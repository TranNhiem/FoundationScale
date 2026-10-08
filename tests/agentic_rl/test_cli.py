"""Tests for ``agentic_rl.cli``: ``build_parser``, ``--dry-run``, the real-run
wiring, and the 0/5/95/96 exit taxonomy.

WHAT IS CLAIMED: ``--dry-run`` on a valid config builds every non-GPU/non-
server object and exits 0, printing the resolved config with provenance; a
SUBPROCESS run of ``--dry-run`` (the module form, ``python -m
foundationscale.agentic_rl``) imports no torch and no transformers (checked
via that child's own ``sys.modules``, since the parent's import graph already
carries both from other test modules); a declared refusal (config file,
``--set``, or a plane ``ValueError``) exits 96; an argparse usage error exits
96 the same way ``train/cli.py``'s own translation does; an unhandled crash
during command-line binding, dry-run construction, real-run construction or
``RLTrainer.run()`` each exit 5; ``RLTrainer.run()`` raising ``TrainerRefusal``
(construction or mid-run) exits 96; a ``run()`` returning 0 step reports exits
95; a normal successful run exits 0. The real run is exercised with
``transformers.AutoTokenizer.from_pretrained``, ``EngineFleet.start_all``/
``stop_all`` and ``RLTrainer`` itself monkeypatched -- no GPU, server or real
tokenizer download is ever touched.

WHAT IS NOT CLAIMED: any real training, any real engine server, or any real
HF tokenizer/model load.
"""

from __future__ import annotations

import dataclasses
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from foundationscale.agentic_rl.cli import (
    EXIT_PASS,
    EXIT_RED,
    EXIT_REFUSE,
    EXIT_UNMEASURED,
    _HFChatTokenizer,
    build_parser,
    main,
)
from foundationscale.agentic_rl.engines.fleet import EngineFleet
from foundationscale.rl.trainer import TrainerRefusal
from foundationscale.train.loop import (
    EXIT_PASS as REAL_EXIT_PASS,
)
from foundationscale.train.loop import (
    EXIT_RED as REAL_EXIT_RED,
)
from foundationscale.train.loop import (
    EXIT_REFUSE as REAL_EXIT_REFUSE,
)
from foundationscale.train.loop import (
    EXIT_UNMEASURED as REAL_EXIT_UNMEASURED,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_EXAMPLE_CONFIG = _REPO_ROOT / "examples" / "agentic_rl" / "music_s0.json"


def test_exit_constants_match_train_loops_real_values() -> None:
    # Drift guard for cli.py's deliberate re-statement (see its own module
    # docstring for why these are NOT imported from train.loop: that module
    # imports torch/transformers at module scope, which would defeat
    # --dry-run's own no-torch claim).
    assert EXIT_PASS == REAL_EXIT_PASS
    assert EXIT_RED == REAL_EXIT_RED
    assert EXIT_UNMEASURED == REAL_EXIT_UNMEASURED
    assert EXIT_REFUSE == REAL_EXIT_REFUSE


def _raw_config(tasks_path: str, **section_overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "policy": {"model_path": "Qwen/Qwen2.5-1.5B-Instruct"},
        "trainer": {"max_steps": 1},
        "engine": {
            "kind": "vllm",
            "command": ["python3", "-m", "vllm.entrypoints.openai.api_server", "--port", "8000"],
            "port": 8000,
            "served_model_name": "Qwen/Qwen2.5-1.5B-Instruct",
            "log_dir": "/tmp",
            "reload_mode": "collective_rpc",
        },
        "env": {"backend": "local", "allow_unsandboxed": True},
        "harness": {"parser": "qwen_xml"},
        "budget": {"step_limit": 2, "max_response_tokens": 64, "max_observation_chars": 128},
        "sampling": {
            "temperature": 1.0,
            "top_p": 1.0,
            "top_k": 0,
            "min_p": 0.0,
            "presence_penalty": 0.0,
            "repetition_penalty": 1.0,
            "max_new_tokens": 16,
        },
        "rollout": {
            "tasks_path": tasks_path,
            "tasks_per_step": 1,
            "group_size": 2,
            "max_concurrency": 2,
        },
        "reward": {"abc2midi_bin": "abc2midi"},
        "publish_root": "/tmp/publish",
    }
    for section, overrides in section_overrides.items():
        base[section] = overrides
    return base


def _write_tasks(tmp_path: Path) -> str:
    path = tmp_path / "tasks.jsonl"
    rows = [
        {"uid": "t1", "messages": [{"role": "user", "content": "hello"}]},
        {"uid": "t2", "messages": [{"role": "user", "content": "world"}]},
    ]
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return str(path)


def _write_config(tmp_path: Path, raw: dict[str, Any]) -> str:
    path = tmp_path / "config.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    return str(path)


# ---------------------------------------------------------------------------
# build_parser
# ---------------------------------------------------------------------------


def test_build_parser_requires_config() -> None:
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])


def test_build_parser_collects_repeated_set_flags() -> None:
    parser = build_parser()
    args = parser.parse_args(["--config", "x.json", "--set", "a.b=1", "--set", "c.d=2"])
    assert args.set_overrides == ["a.b=1", "c.d=2"]
    assert args.dry_run is False


def test_build_parser_prog_name() -> None:
    assert build_parser().prog == "foundationscale-agentic-rl"


# ---------------------------------------------------------------------------
# _HFChatTokenizer
# ---------------------------------------------------------------------------


class _FakeHFTokenizer:
    def apply_chat_template(
        self,
        messages: Any,
        *,
        tools: Any,
        add_generation_prompt: bool,
        tokenize: bool,
        return_dict: bool = True,
    ) -> list[int]:
        assert return_dict is False, "the adapter must ask for raw ids, not a mapping"
        self.last_call = (messages, tools, add_generation_prompt, tokenize)
        return [1, 2, 3]

    def decode(self, ids: Any) -> str:
        return f"decoded:{list(ids)}"


def test_hf_chat_tokenizer_render_forwards_to_apply_chat_template() -> None:
    fake = _FakeHFTokenizer()
    adapter = _HFChatTokenizer(tokenizer=fake)
    result = adapter.render(
        [{"role": "user", "content": "hi"}], tools=None, add_generation_prompt=True
    )
    assert result == [1, 2, 3]
    assert fake.last_call == ([{"role": "user", "content": "hi"}], None, True, True)


def test_hf_chat_tokenizer_decode_forwards_and_stringifies() -> None:
    adapter = _HFChatTokenizer(tokenizer=_FakeHFTokenizer())
    assert adapter.decode([1, 2]) == "decoded:[1, 2]"


# ---------------------------------------------------------------------------
# --dry-run: success
# ---------------------------------------------------------------------------


def test_dry_run_on_a_valid_config_exits_pass_and_prints_resolved_config(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    tasks_path = _write_tasks(tmp_path)
    config_path = _write_config(tmp_path, _raw_config(tasks_path))
    code = main(["--config", config_path, "--dry-run"])
    assert code == EXIT_PASS
    printed = json.loads(capsys.readouterr().out)
    assert printed["policy"]["model_path"] == "Qwen/Qwen2.5-1.5B-Instruct"
    assert printed["provenance"]["trainer.max_steps"] == "config"


def test_dry_run_applies_set_overrides(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    tasks_path = _write_tasks(tmp_path)
    config_path = _write_config(tmp_path, _raw_config(tasks_path))
    code = main(["--config", config_path, "--set", "publish_root=/elsewhere", "--dry-run"])
    assert code == EXIT_PASS
    printed = json.loads(capsys.readouterr().out)
    assert printed["publish_root"] == "/elsewhere"


def test_the_example_config_dry_runs_clean(capsys: pytest.CaptureFixture[str]) -> None:
    code = main(["--config", str(_EXAMPLE_CONFIG), "--dry-run"])
    assert code == EXIT_PASS
    capsys.readouterr()


# ---------------------------------------------------------------------------
# --dry-run: refusals and crashes
# ---------------------------------------------------------------------------


def test_dry_run_refuses_a_bad_declaration(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    tasks_path = _write_tasks(tmp_path)
    # port=0 fails EngineServerSpec's own construction-time refusal during
    # _build_non_gpu_objects, downstream of config.py's own (looser) checks.
    raw = _raw_config(
        tasks_path,
        engine={
            "kind": "vllm",
            "command": ["python3"],
            "port": 0,
            "served_model_name": "m",
            "reload_mode": "collective_rpc",
        },
    )
    config_path = _write_config(tmp_path, raw)
    code = main(["--config", config_path, "--dry-run"])
    assert code == EXIT_REFUSE
    assert "[fs:agentic-rl:refuse]" in capsys.readouterr().out


def test_dry_run_adjudicates_an_unexpected_crash_as_red(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    tasks_path = _write_tasks(tmp_path)
    config_path = _write_config(tmp_path, _raw_config(tasks_path))

    def _boom(config: Any) -> dict[str, Any]:
        raise RuntimeError("boom")

    monkeypatch.setattr("foundationscale.agentic_rl.cli._build_non_gpu_objects", _boom)
    code = main(["--config", config_path, "--dry-run"])
    assert code == EXIT_RED
    assert "[fs:agentic-rl:red]" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# main(): command-line / config-loading boundary
# ---------------------------------------------------------------------------


def test_main_refuses_an_unparseable_command_line(capsys: pytest.CaptureFixture[str]) -> None:
    code = main([])
    assert code == EXIT_REFUSE
    assert "[fs:agentic-rl:refuse]" in capsys.readouterr().out


def test_main_help_exits_via_systemexit_zero() -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(["--help"])
    assert excinfo.value.code == 0


def test_main_refuses_a_config_refusal(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    missing = str(tmp_path / "nope.json")
    code = main(["--config", missing])
    assert code == EXIT_REFUSE
    assert "config refused" in capsys.readouterr().out


def test_main_adjudicates_an_unexpected_binding_crash_as_red(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    tasks_path = _write_tasks(tmp_path)
    config_path = _write_config(tmp_path, _raw_config(tasks_path))

    def _boom(path: str, overrides: Any) -> Any:
        raise RuntimeError("binding boom")

    monkeypatch.setattr("foundationscale.agentic_rl.cli.load_config", _boom)
    code = main(["--config", config_path])
    assert code == EXIT_RED
    assert "[fs:agentic-rl:red]" in capsys.readouterr().out


def test_main_wraps_a_non_agentic_config_refusal_value_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # A ValueError that is NOT an AgenticConfigRefusal escapes the inner
    # except and is adjudicated RED by the outer boundary -- it is not a
    # declared refusal this plane named.
    tasks_path = _write_tasks(tmp_path)
    config_path = _write_config(tmp_path, _raw_config(tasks_path))

    def _raise_plain_value_error(path: str, overrides: Any) -> Any:
        raise ValueError("not an AgenticConfigRefusal")

    monkeypatch.setattr("foundationscale.agentic_rl.cli.load_config", _raise_plain_value_error)
    code = main(["--config", config_path])
    assert code == EXIT_RED


# ---------------------------------------------------------------------------
# Real run: fully monkeypatched (no GPU, no server, no real tokenizer)
# ---------------------------------------------------------------------------


class _FakeTokenizer:
    pass


class _StepReport:
    pass


@pytest.fixture(autouse=True)
def _patch_engine_fleet_lifecycle(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(EngineFleet, "start_all", lambda self: None)
    monkeypatch.setattr(EngineFleet, "stop_all", lambda self: None)


@pytest.fixture(autouse=True)
def _patch_tokenizer(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "transformers.AutoTokenizer.from_pretrained", lambda *a, **k: _FakeTokenizer()
    )


def _real_run_config_path(tmp_path: Path) -> str:
    tasks_path = _write_tasks(tmp_path)
    return _write_config(tmp_path, _raw_config(tasks_path))


def test_real_run_success_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    class _FakeRLTrainer:
        def __init__(self, config: Any) -> None:
            self.config = config

        def run(self) -> list[_StepReport]:
            return [_StepReport()]

    monkeypatch.setattr("foundationscale.agentic_rl.cli.RLTrainer", _FakeRLTrainer)
    code = main(["--config", _real_run_config_path(tmp_path)])
    assert code == EXIT_PASS


class _FakeRLTrainer:
    def __init__(self, config: Any) -> None:
        self.config = config

    def run(self) -> list[_StepReport]:
        return [_StepReport()]


def _without_weight_sync(monkeypatch: pytest.MonkeyPatch) -> None:
    # _build_real_run always wires a DiskWeightSync (see cli.py); this
    # simulates the absence the engine.allow_stale_policy guard exists for,
    # by replacing the real build's result with weight_sync=None -- the
    # fleet/harness/etc. stay REAL, only weight_sync is stripped.
    import foundationscale.agentic_rl.cli as cli_module

    real_build = cli_module._build_real_run

    def _patched(config: Any) -> Any:
        return dataclasses.replace(real_build(config), weight_sync=None)

    monkeypatch.setattr(cli_module, "_build_real_run", _patched)


def test_real_run_refuses_multi_step_with_no_weight_sync(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _without_weight_sync(monkeypatch)
    tasks_path = _write_tasks(tmp_path)
    raw = _raw_config(tasks_path, trainer={"max_steps": 5})
    code = main(["--config", _write_config(tmp_path, raw)])
    assert code == EXIT_REFUSE
    assert "allow_stale_policy" in capsys.readouterr().out


def test_real_run_one_step_with_no_weight_sync_is_not_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # max_steps<=1 never reaches a second, potentially-stale rollout, so the
    # guard does not fire.
    _without_weight_sync(monkeypatch)
    monkeypatch.setattr("foundationscale.agentic_rl.cli.RLTrainer", _FakeRLTrainer)
    code = main(["--config", _real_run_config_path(tmp_path)])
    assert code == EXIT_PASS


def test_real_run_allow_stale_policy_declares_the_opt_out(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _without_weight_sync(monkeypatch)
    monkeypatch.setattr("foundationscale.agentic_rl.cli.RLTrainer", _FakeRLTrainer)
    tasks_path = _write_tasks(tmp_path)
    base_engine = _raw_config(tasks_path)["engine"]
    raw = _raw_config(
        tasks_path,
        trainer={"max_steps": 5},
        engine={**base_engine, "allow_stale_policy": True},
    )
    code = main(["--config", _write_config(tmp_path, raw)])
    assert code == EXIT_PASS


def test_real_run_multi_step_with_weight_sync_is_not_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The normal path: _build_real_run's own DiskWeightSync is wired, so
    # max_steps>1 proceeds without needing allow_stale_policy at all.
    monkeypatch.setattr("foundationscale.agentic_rl.cli.RLTrainer", _FakeRLTrainer)
    tasks_path = _write_tasks(tmp_path)
    raw = _raw_config(tasks_path, trainer={"max_steps": 5})
    code = main(["--config", _write_config(tmp_path, raw)])
    assert code == EXIT_PASS


def test_real_run_zero_reports_is_unmeasured(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    class _EmptyRLTrainer:
        def __init__(self, config: Any) -> None:
            pass

        def run(self) -> list[_StepReport]:
            return []

    monkeypatch.setattr("foundationscale.agentic_rl.cli.RLTrainer", _EmptyRLTrainer)
    code = main(["--config", _real_run_config_path(tmp_path)])
    assert code == EXIT_UNMEASURED
    assert "[fs:agentic-rl:unmeasured]" in capsys.readouterr().out


def test_real_run_trainer_refusal_at_construction_exits_refuse(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    class _RefusingRLTrainer:
        def __init__(self, config: Any) -> None:
            raise TrainerRefusal("group_size=1: need at least 2")

    monkeypatch.setattr("foundationscale.agentic_rl.cli.RLTrainer", _RefusingRLTrainer)
    code = main(["--config", _real_run_config_path(tmp_path)])
    assert code == EXIT_REFUSE
    assert "[fs:agentic-rl:refuse]" in capsys.readouterr().out


def test_real_run_trainer_refusal_mid_run_exits_refuse(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class _MidRunRefusingRLTrainer:
        def __init__(self, config: Any) -> None:
            pass

        def run(self) -> list[_StepReport]:
            raise TrainerRefusal("temperature<=0 with group_size>1")

    monkeypatch.setattr("foundationscale.agentic_rl.cli.RLTrainer", _MidRunRefusingRLTrainer)
    code = main(["--config", _real_run_config_path(tmp_path)])
    assert code == EXIT_REFUSE


def test_real_run_unexpected_crash_during_run_exits_red(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    class _CrashingRLTrainer:
        def __init__(self, config: Any) -> None:
            pass

        def run(self) -> list[_StepReport]:
            raise RuntimeError("unexpected bug")

    monkeypatch.setattr("foundationscale.agentic_rl.cli.RLTrainer", _CrashingRLTrainer)
    code = main(["--config", _real_run_config_path(tmp_path)])
    assert code == EXIT_RED
    assert "[fs:agentic-rl:red]" in capsys.readouterr().out


def test_real_run_build_refusal_exits_refuse(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    tasks_path = _write_tasks(tmp_path)
    raw = _raw_config(
        tasks_path,
        engine={
            "kind": "vllm",
            "command": ["python3"],
            "port": 0,
            "served_model_name": "m",
            "reload_mode": "collective_rpc",
        },
    )
    config_path = _write_config(tmp_path, raw)
    code = main(["--config", config_path])
    assert code == EXIT_REFUSE


def test_real_run_build_crash_exits_red(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def _boom(config: Any) -> Any:
        raise RuntimeError("build boom")

    monkeypatch.setattr("foundationscale.agentic_rl.cli._build_real_run", _boom)
    code = main(["--config", _real_run_config_path(tmp_path)])
    assert code == EXIT_RED
    assert "[fs:agentic-rl:red]" in capsys.readouterr().out


def test_real_run_stops_the_fleet_even_when_start_all_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stopped = []
    monkeypatch.setattr(
        EngineFleet, "start_all", lambda self: (_ for _ in ()).throw(RuntimeError("no server"))
    )
    monkeypatch.setattr(EngineFleet, "stop_all", lambda self: stopped.append(True))
    code = main(["--config", _real_run_config_path(tmp_path)])
    assert code == EXIT_RED
    assert stopped == [True]


# ---------------------------------------------------------------------------
# Subprocess: the module form, no torch/no transformers on --dry-run
# ---------------------------------------------------------------------------


def test_dry_run_subprocess_imports_no_torch_or_transformers() -> None:
    script = (
        "import sys\n"
        "from foundationscale.agentic_rl.cli import main\n"
        "code = main(['--config', 'examples/agentic_rl/music_s0.json', '--dry-run'])\n"
        "print('TORCH=' + str('torch' in sys.modules))\n"
        "print('TRANSFORMERS=' + str('transformers' in sys.modules))\n"
        "raise SystemExit(code)\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", script],
        cwd=str(_REPO_ROOT),
        env={"PYTHONPATH": "src", "PATH": "/usr/bin:/bin"},
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode == 0, proc.stderr
    assert "TORCH=False" in proc.stdout
    assert "TRANSFORMERS=False" in proc.stdout


def test_module_form_dry_run_exits_zero_via_python_dash_m() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "foundationscale.agentic_rl",
            "--config",
            str(_EXAMPLE_CONFIG),
            "--dry-run",
        ],
        cwd=str(_REPO_ROOT),
        env={"PYTHONPATH": "src", "PATH": "/usr/bin:/bin"},
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode == 0, proc.stderr
    printed = json.loads(proc.stdout)
    assert printed["policy"]["model_path"]


class _BatchEncodingTokenizer:
    """Mimics transformers 5: apply_chat_template(tokenize=True) -> a mapping."""

    def __init__(self, payload: Any) -> None:
        self.payload = payload

    def apply_chat_template(self, messages: Any, **kwargs: Any) -> Any:
        return self.payload

    def decode(self, ids: Any) -> str:
        return "x"


def test_hf_chat_tokenizer_reads_ids_out_of_a_batch_encoding() -> None:
    # Regression from the first GPU smoke run: list(BatchEncoding) gave the KEY
    # strings ("input_ids", ...) instead of token ids.
    from foundationscale.agentic_rl.cli import _HFChatTokenizer

    for payload in ({"input_ids": [5, 6, 7], "attention_mask": [1, 1, 1]}, [5, 6, 7], [[5, 6, 7]]):
        adapter = _HFChatTokenizer(_BatchEncodingTokenizer(payload))
        assert adapter.render(
            [{"role": "user", "content": "hi"}], tools=None, add_generation_prompt=True
        ) == [5, 6, 7]


def test_hf_chat_tokenizer_refuses_unreadable_shapes() -> None:
    from foundationscale.agentic_rl.cli import _HFChatTokenizer
    from foundationscale.agentic_rl.config import AgenticConfigRefusal

    with pytest.raises(AgenticConfigRefusal, match="input_ids"):
        _HFChatTokenizer(_BatchEncodingTokenizer({"attention_mask": [1]})).render(
            [], tools=None, add_generation_prompt=True
        )
    with pytest.raises(AgenticConfigRefusal, match="2 sequences"):
        _HFChatTokenizer(_BatchEncodingTokenizer([[1], [2]])).render(
            [], tools=None, add_generation_prompt=True
        )
