"""Laptop proof for tools/real_checkpoint_probe.py against known ground truth.

Why this suite exists
---------------------
The probe's exit code is about to be quoted on a GPU cluster as evidence that
the framework's checkpoint gates work on real artifacts. Everything here is
built so that the correct answer is known *before* the probe runs: safetensors
files written by this file (including one whose header deliberately puts four
expert FQNs on one byte span — the incident's mechanism, constructed, not
assumed) and config.json files whose denominators are stated exactly. If the
probe cannot reproduce ground truth on a 200-byte artifact it owns nothing to
say about a 16 GB one.

The load-bearing assertions are about honesty, not just verdicts: abstentions
must be *stated*, a fabricated-denominator route must not exist, a requested
control that cannot run must weigh against CLEAR, and a probe bug must surface
as exit 3 — never as Python's exit 1, which a pipeline will misfile as
"gates blocked the checkpoint".
"""

from __future__ import annotations

import importlib.util
import json
import struct
import sys
from pathlib import Path
from typing import Any

import pytest
import torch

from foundationscale.gates.core import Verdict

REPO_ROOT = Path(__file__).resolve().parents[1]
_PROBE_PATH = REPO_ROOT / "tools" / "real_checkpoint_probe.py"

_SPEC = importlib.util.spec_from_file_location("real_checkpoint_probe", _PROBE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
probe = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = probe
_SPEC.loader.exec_module(probe)

_ST_DTYPE = {"bfloat16": "BF16"}


def _write_st_file(path: Path, tensors: dict[str, torch.Tensor]) -> None:
    from safetensors.torch import save_file

    save_file({k: v.contiguous() for k, v in tensors.items()}, str(path))


def _write_header_shard(
    path: Path,
    entries: list[tuple[str, str, tuple[int, ...], int, int]],
) -> None:
    """Write a safetensors file whose header states *exactly* these offsets.

    Offsets are caller-controlled so a test can put several expert names on one
    byte span — aliasing ground truth this suite constructs rather than hopes
    for. Only header plus a nominal data pad are written: the reader under test
    is metadata-only, and a test that needed real tensor bytes would be testing
    the wrong layer.
    """
    header = {
        name: {
            "dtype": _ST_DTYPE[dtype],
            "shape": list(shape),
            "data_offsets": [start, end],
        }
        for name, dtype, shape, start, end in entries
    }
    blob = json.dumps(header).encode("utf-8")
    blob += b" " * ((8 - len(blob) % 8) % 8)
    data_len = max((entry[4] for entry in entries), default=0)
    path.write_bytes(struct.pack("<Q", len(blob)) + blob + b"\0" * data_len)


def _dense_checkpoint(tmp_path: Path) -> Path:
    """A small dense artifact: the Gemma-family shape of *no routed experts*."""
    ckpt = tmp_path / "dense"
    ckpt.mkdir()
    _write_st_file(
        ckpt / "model.safetensors",
        {
            "model.embed_tokens.weight": torch.zeros((8, 4), dtype=torch.bfloat16),
            "model.layers.0.mlp.down_proj.weight": torch.zeros((4, 8), dtype=torch.bfloat16),
        },
    )
    (ckpt / "config.json").write_text(json.dumps({"model_type": "probe_dense"}), encoding="utf-8")
    return ckpt


def _moe_checkpoint(tmp_path: Path, *, aliased: bool) -> Path:
    """1 MoE layer x 2 expert weights x 4 experts; spans distinct or shared 4:1."""
    ckpt = tmp_path / ("moe_aliased" if aliased else "moe_healthy")
    ckpt.mkdir()
    stem = "model.layers.0.mlp.experts"
    if aliased:
        entries: list[tuple[str, str, tuple[int, ...], int, int]] = []
        for fc in (1, 2):
            base = 0 if fc == 1 else 8
            for i in range(4):
                # weight0..3 of one linear_fc all name the same 8 bytes.
                entries.append(
                    (f"{stem}.linear_fc{fc}.weight{i}", "bfloat16", (2, 2), base, base + 8)
                )
        entries.append(("model.embed_tokens.weight", "bfloat16", (8, 4), 16, 80))
        _write_header_shard(ckpt / "model.safetensors", entries)
    else:
        tensors: dict[str, torch.Tensor] = {
            "model.embed_tokens.weight": torch.zeros((8, 4), dtype=torch.bfloat16)
        }
        for fc in (1, 2):
            for i in range(4):
                tensors[f"{stem}.linear_fc{fc}.weight{i}"] = torch.zeros(
                    (2, 2), dtype=torch.bfloat16
                )
        _write_st_file(ckpt / "model.safetensors", tensors)
    config = {
        "model_type": "probe_moe",
        "num_local_experts": 4,
        "num_moe_layers": 1,
    }
    (ckpt / "config.json").write_text(json.dumps(config), encoding="utf-8")
    return ckpt


def _run(capsys: pytest.CaptureFixture[str], *argv: str) -> tuple[int, str, str]:
    code = probe.main(list(argv))
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def _report(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    ckpt: Path,
    *extra: str,
) -> tuple[int, dict[str, Any], str]:
    out_json = tmp_path / "probe-report.json"
    code, out, _err = _run(capsys, str(ckpt), "--json", str(out_json), *extra)
    report: dict[str, Any] = json.loads(out_json.read_text(encoding="utf-8"))
    return code, report, out


def test_dense_checkpoint_abstains_stated_and_completeness_blocks(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # The honest answer on a clean dense artifact is NOT "clear": expert gates
    # must SAY they abstained, and completeness must block, because config.json
    # states no tensor list. The word CLEAR must not appear anywhere.
    ckpt = _dense_checkpoint(tmp_path)
    code, report, out = _report(tmp_path, capsys, ckpt)
    assert code == probe.EXIT_BLOCKED
    assert "CLEAR (exit 0)" not in out
    assert "dense model" in out  # the abstention is a stated reason, not silence
    assert f"checkpoint.save_complete={Verdict.VACUOUS.value}" in out
    assert report["control"] is None
    assert report["exit_code"] == code
    # The probe audits the framework's own no-PASS-over-zero-coverage rule on
    # every run; a breach would land here and would have blocked the exit code.
    assert report["framework_invariant_breaches"] == []


def test_healthy_sharded_moe_distinctness_passes_and_run_still_blocks(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # Ground truth: 4 experts x 2 weights, each its own span. Distinctness must
    # reach a real PASS; byte volume must SKIP for want of a stated denominator;
    # completeness VACUOUS still owns the exit code.
    ckpt = _moe_checkpoint(tmp_path, aliased=False)
    code, report, out = _report(tmp_path, capsys, ckpt)
    assert code == probe.EXIT_BLOCKED
    for bad in (Verdict.FAIL, Verdict.VACUOUS, Verdict.UNDERCOVERED, Verdict.ERROR):
        assert f"checkpoint.expert_distinctness={bad.value}" not in out
    assert "checkpoint.expert_bytes: run manifest does not declare" in out
    assert report["framework_invariant_breaches"] == []


def test_aliased_expert_spans_are_caught_from_metadata_alone(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # The incident's mechanism, assembled header-first: 8 right-shaped expert
    # FQNs name 2 physical spans. This is the property the cluster run relies
    # on, proven here against a span map the test itself wrote.
    ckpt = _moe_checkpoint(tmp_path, aliased=True)
    code, _report_dict, out = _report(tmp_path, capsys, ckpt)
    assert code == probe.EXIT_BLOCKED
    assert f"checkpoint.expert_distinctness={Verdict.FAIL.value}" in out
    assert "aliased" in out


def test_inject_alias_control_fires_on_real_distinct_spans(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    ckpt = _moe_checkpoint(tmp_path, aliased=False)
    code, report, _out = _report(tmp_path, capsys, ckpt, "--inject-alias", "3")
    control = report["control"]
    assert control["status"] == "fired"
    assert control["aliased"] == 3
    # The unmodified artifact does NOT block this gate, so the block is
    # attributable to the injection — the flag exists to say when it is not.
    assert control["confounded"] is False
    assert control["aliasing_leg_observed"] is True
    assert code == probe.EXIT_BLOCKED  # save-completeness still owns the exit


def test_inject_alias_control_says_confounded_on_already_blocking_artifact(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # On an artifact that already blocks the gate, "fired" proves only that the
    # gate blocks — the control must say it cannot attribute the block.
    ckpt = _moe_checkpoint(tmp_path, aliased=True)
    _code, report, _out = _report(tmp_path, capsys, ckpt, "--inject-alias", "2")
    assert report["control"]["status"] == "fired"
    assert report["control"]["confounded"] is True


def test_skipped_control_weighs_against_clear(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # A requested MUST_FIRE control that cannot run leaves "the detector fires
    # on real aliasing" unverified; that must appear as a blocking reason, not
    # ride along as a quiet pass — the all([]) result one level up.
    ckpt = _dense_checkpoint(tmp_path)
    code, report, _out = _report(tmp_path, capsys, ckpt, "--inject-alias", "4")
    assert report["control"]["status"] == "skipped"
    assert "inapplicable" in report["control"]["reason"]
    assert any("--inject-alias" in reason for reason in report["blocking_reasons"])
    assert code == probe.EXIT_BLOCKED


def test_missing_config_is_unmeasured_not_clear(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # Without the independent denominator source the probe has nothing to
    # compare against; that is exit 3, never a verdict about the checkpoint.
    ckpt = _dense_checkpoint(tmp_path)
    (ckpt / "config.json").unlink()
    code, _out, err = _run(capsys, str(ckpt))
    assert code == probe.EXIT_UNMEASURED
    assert "config file not found" in err


def test_invalid_config_json_is_unmeasured(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    ckpt = _dense_checkpoint(tmp_path)
    (ckpt / "config.json").write_text("{not json", encoding="utf-8")
    code, _out, err = _run(capsys, str(ckpt))
    assert code == probe.EXIT_UNMEASURED
    assert "config unreadable or not valid JSON" in err


def test_non_object_config_is_unmeasured(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    ckpt = _dense_checkpoint(tmp_path)
    (ckpt / "config.json").write_text("[1, 2, 3]", encoding="utf-8")
    code, _out, err = _run(capsys, str(ckpt))
    assert code == probe.EXIT_UNMEASURED
    assert "config is not a JSON object" in err


def test_non_checkpoint_directory_is_unmeasured(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    not_a_ckpt = tmp_path / "not_a_ckpt"
    not_a_ckpt.mkdir()
    code, _out, err = _run(capsys, str(not_a_ckpt))
    assert code == probe.EXIT_UNMEASURED
    assert "no recognizable checkpoint layout" in err


def test_zero_tensor_export_is_refused_not_vacuously_cleared(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # A header declaring zero keys is all([]) raw material; read_metadata
    # raises, and the probe must surface that as "could not measure".
    ckpt = tmp_path / "empty_export"
    ckpt.mkdir()
    _write_header_shard(ckpt / "model.safetensors", [])
    (ckpt / "config.json").write_text("{}", encoding="utf-8")
    code, _out, err = _run(capsys, str(ckpt))
    assert code == probe.EXIT_UNMEASURED
    assert "declares 0 tensors" in err


def test_probe_bug_after_measurement_exits_three_not_one(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    # The contract's sharpest edge: an exception while rendering must be
    # "could not measure" (3). Plain exit 1 means "a gate reached a blocking
    # verdict", and a cluster pipeline will file the run accordingly.

    def _boom(_inventory: dict[str, Any]) -> None:
        raise RuntimeError("synthetic probe bug")

    monkeypatch.setattr(probe, "_print_inventory", _boom)
    code, _out, err = _run(capsys, str(_dense_checkpoint(tmp_path)))
    assert code == probe.EXIT_UNMEASURED
    assert "a probe bug is not a checkpoint verdict" in err


def test_inject_alias_of_one_is_a_usage_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # N=1 would "alias" one tensor onto itself — a control-shaped no-op. It is
    # refused as a usage error (argparse's 2), not measured as anything.
    ckpt = _moe_checkpoint(tmp_path, aliased=False)
    with pytest.raises(SystemExit) as excinfo:
        probe.main([str(ckpt), "--inject-alias", "1"])
    assert excinfo.value.code == 2
    capsys.readouterr()


def test_config_can_never_supply_the_completeness_denominator() -> None:
    # This locks the probe's own rule: an HF config states hyperparameters, not
    # tensor FQNs. The day this fails, someone has taught derive_declared to
    # guess the declared tensor set — the fabricated-denominator defect the
    # probe exists to refuse, and the only road to a false CLEAR.
    declared = probe.derive_declared({"num_experts": 4, "num_moe_layers": 1})
    assert declared["declared_fqns"] is None
    assert declared["expected_expert_bytes"] is None


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (8, 8),
        (8.0, 8),  # integral float: used, and the coercion is written in notes
        (7.5, None),  # truncating 7.5 to 7 mints a count the config never stated
        ("8", None),  # a string that converts cleanly is still not a declaration
        (True, None),
        (None, None),
        (-2, None),  # a negative count is not a denominator
        (float("inf"), None),  # int(inf) raises OverflowError; must be absent quietly
    ],
)
def test_scoped_int_accepts_only_stated_json_integers(raw: Any, expected: int | None) -> None:
    notes: list[str] = []
    value, _basis = probe._scoped_int({"num_experts": raw}, ("num_experts",), "num_experts", notes)
    assert value == expected
    if isinstance(raw, int) and not isinstance(raw, bool) and raw >= 0:
        assert notes == []
    else:
        # Every non-plain-integer decision is written down. A denominator with
        # no visible provenance is how the probe would start telling stories.
        assert notes != []


def test_scoped_int_prefers_text_config_scope() -> None:
    # Gemma-style nesting: the LM scope is where the expert count lives, so it
    # outranks a multimodal top-level value.
    config = {"num_experts": 2, "text_config": {"num_experts": 4}}
    value, basis = probe._scoped_int(config, ("num_experts",), "num_experts", [])
    assert value == 4
    assert basis is not None and basis.startswith("text_config.")


def test_malformed_expert_count_never_mints_a_denominator(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # End-to-end for the _scoped_int contract: with `num_experts: 7.5` in the
    # config, the probe must report the dense classification (and a note), not
    # a declared 7 against which a 7-shard artifact would "match count".
    ckpt = _moe_checkpoint(tmp_path, aliased=False)
    (ckpt / "config.json").write_text(
        json.dumps({"num_experts": 7.5, "num_moe_layers": 1}), encoding="utf-8"
    )
    _code, report, _out = _report(tmp_path, capsys, ckpt)
    assert report["declared"]["num_experts"] == 0
    assert any("not an integral number" in note for note in report["declared"]["notes"])
