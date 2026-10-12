"""Held-out loss evaluation (VLM): runner argv/result parsing and the skill's rules.

Nothing here evaluates a checkpoint: ``VlmHeldoutRunner`` talks to a faked
``subprocess.run`` and every skill rule/verdict runs through a duck-typed fake
runner handed to the skill's runner injection -- no subprocess is ever spawned.
"""
from __future__ import annotations

import inspect
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

import foundationskills.skills.evaluation.runner as runner_module
from foundationskills.core.contract import SkillContext, SkillResult
from foundationskills.core.status import Status
from foundationskills.skills.evaluation.policy import load_policy
from foundationskills.skills.evaluation.runner import (
    PINNED_LM_EVAL,
    HeldoutOutcome,
    HeldoutRequest,
    VlmHeldoutRunner,
)
from foundationskills.skills.evaluation.skill import EvalSkill

# -- train/eval frame budget and rows ---------------------------------------------------------------------------

RUN_MANIFEST = {"environment": {"values": {"FOUNDATIONSCALE_TRAIN_VIDEO_FRAMES": "16"}}}  # trained on 16 frames

IMAGE_ROWS = [{"text": "one", "image": "a.png"}, {"text": "two", "image": "b.png"}]
VIDEO_ROWS = [{"text": "clip", "image": "c.png", "video": "clip.mp4"}]
EMPTY_VIDEO_ROWS = [{"text": "still", "image": "d.png", "video": ""}]  # empty video field, not a video row

RESULT_KEYS = {
    "metric", "parameters", "rows_read", "rows_used", "rows_dropped", "drop_reasons", "tokens_scored",
    "total_nll", "mean_nll_nats", "mean_nll_bits", "perplexity", "peak_gpu_memory_gb", "wall_seconds", "versions",
}

NEW_RULES = ("EV-IN-006", "EV-IN-007", "EV-IN-008", "EV-RU-001", "EV-HO-005", "EV-HO-006")


def heldout_result(mean_nll: float = 2.0, rows_used: int = 25, rows_dropped: int = 1) -> dict[str, Any]:
    """A fake ``eval_heldout_loss.py --out`` JSON (one key per script output field)."""
    tokens = 1280
    return {
        "metric": "heldout_nll",
        "parameters": {"total": 8_000_000_000, "trainable": 4_000_000},
        "rows_read": rows_used + rows_dropped,
        "rows_used": rows_used,
        "rows_dropped": rows_dropped,
        "drop_reasons": {},
        "tokens_scored": tokens,
        "total_nll": mean_nll * tokens,
        "mean_nll_nats": mean_nll,
        "mean_nll_bits": mean_nll / math.log(2),
        "perplexity": math.exp(mean_nll),
        "peak_gpu_memory_gb": 42.0,
        "wall_seconds": 31.5,
        "versions": {"python": "3.11.8", "torch": "2.3.0", "transformers": "4.41.2"},
    }


def outcome(mean_nll: float = 2.0, rows_used: int = 25, rows_dropped: int = 1,
            rc: int = 0, stderr_tail: str = "") -> HeldoutOutcome:
    """One ``VlmHeldoutRunner`` answer; a non-zero ``rc`` carries no result JSON."""
    if rc != 0:
        return HeldoutOutcome(rc=rc, result=None, stderr_tail=stderr_tail)
    return HeldoutOutcome(rc=rc, result=heldout_result(mean_nll, rows_used, rows_dropped), stderr_tail=stderr_tail)


# -- test inputs -----------------------------------------------------------------------------------------------


def write_data(path: Path, rows: list[dict[str, Any]]) -> Path:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


def write_manifest(tmp_path: Path) -> Path:
    """An FS run manifest recording the training frame budget at ``environment.values``."""
    path = tmp_path / "run_manifest.json"
    path.write_text(json.dumps(RUN_MANIFEST), encoding="utf-8")
    return path


def write_policy_doc(path: Path) -> Path | None:
    """Smallest judge-free policy document that ``load_policy`` accepts (else None).

    Held-out scoring compares mean NLL with ``abs_epsilon`` and never reads the
    policy bands, but the request envelope may still ask for a parseable policy --
    so hand over the first minimal shape the loader takes, or no ``policy`` key.
    """
    for doc in ('{"version": "1", "tasks": {}}', '{"tasks": {}}', '{"version": "1"}', "{}"):
        path.write_text(doc, encoding="utf-8")
        try:
            load_policy(str(path))
        except Exception:
            continue
        return path
    return None


def make_request(tmp_path: Path, rows: list[dict[str, Any]] = IMAGE_ROWS,
                 manifest: Path | None = None, **heldout_overrides: Any) -> dict[str, Any]:
    """A held-out evaluation request: ``benchmarks`` stays empty when ``heldout`` is set."""
    for name in ("base-model", "checkpoint", "eval-cache"):
        (tmp_path / name).mkdir(exist_ok=True)
    data = write_data(tmp_path / "heldout.jsonl", rows)
    script = tmp_path / "eval_heldout_loss.py"
    script.write_text("# stub: the fake runner never executes it\n", encoding="utf-8")
    request: dict[str, Any] = {
        "checkpoint": str(tmp_path / "checkpoint"),
        "base": str(tmp_path / "base-model"),
        "benchmarks": [],
        "eval_cache": str(tmp_path / "eval-cache"),
        "baseline_cache": str(tmp_path / "baseline-cache"),
        "out": str(tmp_path / "eval_report.json"),
        "heldout": {
            "data": str(data),
            "max_length": 2048,
            "video_frames": None,
            "image_column": "image",
            "video_column": "video",
            "limit": None,
            "heldout_script": str(script),
            "abs_epsilon": 0.25,
            **heldout_overrides,
        },
    }
    if manifest is not None:
        request["run_manifest"] = str(manifest)
    policy = write_policy_doc(tmp_path / "eval_policy.yaml")
    if policy is not None:
        request["policy"] = str(policy)
    return request


def make_ctx(tmp_path: Path) -> SkillContext:
    """Minimal skill context: the held-out path only reads ``artifacts_dir``."""
    return SkillContext(workdir=tmp_path)


def runner_request(tmp_path: Path, **overrides: Any) -> HeldoutRequest:
    values: dict[str, Any] = {
        "model": str(tmp_path / "ckpt"),
        "adapter": None,
        "data": str(tmp_path / "heldout.jsonl"),
        "max_length": 2048,
        "video_frames": 16,
        "image_column": "image",
        "video_column": "video",
        "limit": 64,
        "out": str(tmp_path / "heldout" / "result.json"),
        "script": str(tmp_path / "eval_heldout_loss.py"),
        "python": sys.executable,
    }
    values.update(overrides)
    return HeldoutRequest(**values)


def flag_value(argv: list[str], flag: str) -> str | None:
    """Value following ``flag`` in an argv list (None when the flag is absent)."""
    if flag not in argv:
        return None
    return argv[argv.index(flag) + 1]


def flatten(*chunks: Any) -> str:
    """Findings/messages flattened into one searchable text (rule ids match inside)."""
    parts: list[str] = []
    for chunk in chunks:
        if chunk is None:
            continue
        if isinstance(chunk, str):
            parts.append(chunk)
        else:
            parts.extend(repr(item) for item in chunk)
    return " ".join(parts)


def said(skill: EvalSkill, result: SkillResult, ctx: SkillContext) -> str:
    """Everything the skill said: findings, its refusal text, and any handoff findings."""
    handoff: tuple[Any, ...] = ()
    if isinstance(result.payload, dict) and result.artifacts:
        handoff = tuple(skill.check_handoff(result, ctx))
    return flatten(result.findings, result.refusal, handoff)


def inputs_reported(tmp_path: Path, request: dict[str, Any]) -> str:
    skill = build_skill(FakeHeldoutRunner(request))
    return flatten(skill.check_inputs(request, make_ctx(tmp_path)))


def heldout_row(payload: dict[str, Any]) -> dict[str, Any]:
    """The ``heldout_nll`` report row wherever the report nests it."""
    rows = payload.get("benchmarks") or []
    if isinstance(rows, dict):
        rows = list(rows.values())
    for row in rows:
        if isinstance(row, dict) and row.get("name") == "heldout_nll":
            return row
    raise AssertionError(f"no heldout_nll row in report keys {sorted(payload)}")


# -- fakes -----------------------------------------------------------------------------------------------------


class FakeHeldoutRunner:
    """A ``VlmHeldoutRunner`` stand-in: records every ``HeldoutRequest`` and answers
    each role from a canned ``HeldoutOutcome`` (no subprocess is ever spawned)."""

    def __init__(self, request: dict[str, Any], base: HeldoutOutcome | None = None,
                 checkpoint: HeldoutOutcome | None = None) -> None:
        self.base_model = Path(request["base"])
        self.checkpoint_model = Path(request["checkpoint"])
        self.outcomes = {
            "base": base if base is not None else outcome(2.0),
            "checkpoint": checkpoint if checkpoint is not None else outcome(1.5),
        }
        self.requests: list[HeldoutRequest] = []

    def run(self, request: HeldoutRequest) -> HeldoutOutcome:
        self.requests.append(request)
        return self.outcomes[self.role_of(request)]

    def role_of(self, request: HeldoutRequest) -> str:
        """The base role scores the plain base model; the checkpoint role is a full
        checkpoint directory, or the base model plus a PEFT adapter directory."""
        if request.adapter is not None or Path(request.model) == self.checkpoint_model:
            return "checkpoint"
        return "base"

    def calls(self, role: str) -> HeldoutRequest:
        found = [request for request in self.requests if self.role_of(request) == role]
        assert len(found) == 1, f"expected exactly one {role} call, got {len(found)}"
        return found[0]


def build_skill(fake: FakeHeldoutRunner) -> EvalSkill:
    """``EvalSkill`` with ``fake`` behind its runner injection (and its dedicated
    held-out knob when the skill grew one), with the pinned lm-eval present."""
    params = inspect.signature(EvalSkill.__init__).parameters
    kwargs: dict[str, Any] = {"runner": fake, "version_probe": lambda: PINNED_LM_EVAL}
    for name in ("heldout_runner", "heldout", "vlm_heldout_runner"):
        if name in params:
            kwargs[name] = fake
            break
    return EvalSkill(**kwargs)


def run_heldout(tmp_path: Path, request: dict[str, Any], **outcomes: HeldoutOutcome):
    """``FakeHeldoutRunner``, skill, context and the result of one held-out run."""
    fake = FakeHeldoutRunner(request, **outcomes)
    skill = build_skill(fake)
    ctx = make_ctx(tmp_path)
    return fake, skill, ctx, skill.run(request, ctx)


@pytest.fixture
def subprocess_calls(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """Faked ``subprocess.run``: record argv/kwargs and write a canned ``--out`` JSON."""

    calls: list[dict[str, Any]] = []

    def fake_run(argv: Any, **kwargs: Any) -> subprocess.CompletedProcess:
        calls.append({"argv": argv, "kwargs": kwargs})
        out = Path(argv[argv.index("--out") + 1])
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(heldout_result()), encoding="utf-8")
        return subprocess.CompletedProcess(argv, 0, "", "")

    monkeypatch.setattr(runner_module.subprocess, "run", fake_run)
    return calls


# == VlmHeldoutRunner: argv and result parsing ====================================================================


def test_heldout_argv_is_a_list_with_every_flag(tmp_path: Path, subprocess_calls: list) -> None:
    request = runner_request(tmp_path)
    ran = VlmHeldoutRunner().run(request)

    assert ran.rc == 0
    call = subprocess_calls[0]
    argv = call["argv"]
    assert isinstance(argv, list) and all(isinstance(part, str) for part in argv)  # never a shell string
    assert argv[0] == request.python and request.script in argv
    assert flag_value(argv, "--model") == request.model
    assert flag_value(argv, "--data") == request.data
    assert flag_value(argv, "--max-length") == str(request.max_length)
    assert flag_value(argv, "--image-column") == request.image_column
    assert flag_value(argv, "--video-frames") == str(request.video_frames)
    assert flag_value(argv, "--video-column") == request.video_column
    assert flag_value(argv, "--limit") == str(request.limit)
    assert flag_value(argv, "--out") == request.out
    assert "--adapter" not in argv  # the base role carries no adapter
    assert Path(call["kwargs"]["cwd"]) == Path(request.out).parent  # runs in the --out dir
    assert call["kwargs"].get("env") in (None, dict(os.environ))  # the environment inherits


def test_adapter_flag_is_emitted_only_for_the_checkpoint_role(tmp_path: Path, subprocess_calls: list) -> None:
    adapter = str(tmp_path / "adapter")
    roles = (
        runner_request(tmp_path, model=str(tmp_path / "base-model")),  # base role
        runner_request(tmp_path, model=str(tmp_path / "base-model"), adapter=adapter),  # checkpoint role
    )
    for request in roles:
        assert VlmHeldoutRunner().run(request).rc == 0

    base_argv = subprocess_calls[0]["argv"]
    checkpoint_argv = subprocess_calls[1]["argv"]
    assert "--adapter" not in base_argv
    assert flag_value(checkpoint_argv, "--adapter") == adapter


def test_optional_flags_are_left_out(tmp_path: Path, subprocess_calls: list) -> None:
    request = runner_request(tmp_path, video_frames=None, limit=None)
    assert VlmHeldoutRunner().run(request).rc == 0

    argv = subprocess_calls[0]["argv"]
    assert "--video-frames" not in argv
    assert "--limit" not in argv
    assert "--adapter" not in argv


def test_the_result_json_is_parsed_into_the_outcome(tmp_path: Path, subprocess_calls: list) -> None:
    request = runner_request(tmp_path)
    ran = VlmHeldoutRunner().run(request)

    assert ran.rc == 0
    assert RESULT_KEYS <= set(ran.result or {})  # every key of the script's interface
    assert ran.result == json.loads(Path(request.out).read_text(encoding="utf-8"))
    assert ran.result["metric"] == "heldout_nll"
    assert ran.result["rows_used"] == 25 and ran.result["rows_dropped"] == 1
    assert ran.result["mean_nll_nats"] == pytest.approx(2.0)
    assert ran.result["perplexity"] == pytest.approx(math.exp(2.0))
    assert ran.result["versions"]["torch"] == "2.3.0"
    assert ran.stderr_tail == ""


def test_exit_96_carries_no_result(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(argv: Any, **kwargs: Any) -> subprocess.CompletedProcess:
        return subprocess.CompletedProcess(argv, 96, "", "video rows present but --video-frames not given")

    monkeypatch.setattr(runner_module.subprocess, "run", refuse)
    ran = VlmHeldoutRunner().run(runner_request(tmp_path))

    assert ran.rc == 96
    assert ran.result is None
    assert "video rows" in ran.stderr_tail  # the refusal reason is surfaced


# == skill: declarations =========================================================================================


def test_scope_accepts_vlm_models() -> None:
    assert "vlm" in EvalSkill.scope.model_types


def test_every_new_rule_is_declared_with_a_must_fire_fixture() -> None:
    skill = EvalSkill(version_probe=lambda: PINNED_LM_EVAL)
    declared = repr(EvalSkill.rules)
    fixtures = skill.must_fire_fixtures()
    for rule_id in NEW_RULES:
        assert rule_id in declared, f"{rule_id} missing from EvalSkill.rules"
        assert rule_id in fixtures, f"{rule_id} has no MUST_FIRE fixture"


# == skill: input rules ==========================================================================================


def test_ev_in_006_fires_when_the_held_out_data_is_missing(tmp_path: Path) -> None:
    request = make_request(tmp_path, data=str(tmp_path / "gone.jsonl"))
    assert "EV-IN-006" in inputs_reported(tmp_path, request)


def test_ev_in_006_fires_when_the_eval_script_is_missing(tmp_path: Path) -> None:
    request = make_request(tmp_path, heldout_script=str(tmp_path / "gone.py"))
    assert "EV-IN-006" in inputs_reported(tmp_path, request)


def test_ev_in_006_quiet_with_readable_data_and_the_script(tmp_path: Path) -> None:
    request = make_request(tmp_path)  # data parses, resolve_default overriden by heldout_script
    assert "EV-IN-006" not in inputs_reported(tmp_path, request)


def test_ev_in_007_fires_for_video_rows_without_video_frames(tmp_path: Path) -> None:
    request = make_request(tmp_path, rows=VIDEO_ROWS)  # video_frames stays unset
    assert "EV-IN-007" in inputs_reported(tmp_path, request)


def test_ev_in_007_quiet_when_video_frames_are_given(tmp_path: Path) -> None:
    request = make_request(tmp_path, rows=VIDEO_ROWS, video_frames=16)
    assert "EV-IN-007" not in inputs_reported(tmp_path, request)


def test_ev_in_007_quiet_without_video_rows(tmp_path: Path) -> None:
    request = make_request(tmp_path, rows=EMPTY_VIDEO_ROWS)  # empty video field, no frames needed
    assert "EV-IN-007" not in inputs_reported(tmp_path, request)


def test_ev_in_008_fires_on_a_frame_budget_mismatch_against_the_run_manifest(tmp_path: Path) -> None:
    manifest = write_manifest(tmp_path)  # records FOUNDATIONSCALE_TRAIN_VIDEO_FRAMES == "16"
    request = make_request(tmp_path, rows=VIDEO_ROWS, video_frames=8, manifest=manifest)

    assert RUN_MANIFEST["environment"]["values"]["FOUNDATIONSCALE_TRAIN_VIDEO_FRAMES"] == "16"
    assert "EV-IN-008" in inputs_reported(tmp_path, request)


def test_ev_in_008_quiet_when_frames_match_the_training_budget(tmp_path: Path) -> None:
    manifest = write_manifest(tmp_path)  # 16 frames trained, 16 frames scored
    request = make_request(tmp_path, rows=VIDEO_ROWS, video_frames=16, manifest=manifest)
    assert "EV-IN-008" not in inputs_reported(tmp_path, request)


def test_ev_in_008_quiet_without_a_run_manifest(tmp_path: Path) -> None:
    request = make_request(tmp_path, rows=VIDEO_ROWS, video_frames=8)  # training budget unknown
    assert "EV-IN-008" not in inputs_reported(tmp_path, request)


# == skill: run and handoff rules ================================================================================


@pytest.mark.parametrize("role", ["base", "checkpoint"])
def test_ev_ru_001_zero_scored_rows_are_refused_as_vacuous(tmp_path: Path, role: str) -> None:
    request = make_request(tmp_path)
    scored = outcome(2.0, rows_used=4, rows_dropped=0)
    _, skill, ctx, result = run_heldout(
        tmp_path, request,
        base=outcome(2.0, rows_used=0, rows_dropped=0) if role == "base" else scored,
        checkpoint=outcome(2.0, rows_used=0, rows_dropped=0) if role == "checkpoint" else scored,
    )

    assert result.status is Status.REFUSED
    assert "EV-RU-001" in said(skill, result, ctx)
    assert not Path(request["out"]).exists()  # refusals write no report


def test_ev_ru_001_quiet_with_a_single_scored_row(tmp_path: Path) -> None:
    request = make_request(tmp_path)
    _, skill, ctx, result = run_heldout(
        tmp_path, request,
        base=outcome(2.0, rows_used=1, rows_dropped=0),
        checkpoint=outcome(1.0, rows_used=1, rows_dropped=0),
    )

    assert result.status is Status.PASS  # one row is enough to score
    assert "EV-RU-001" not in said(skill, result, ctx)


@pytest.mark.parametrize("disagree_in", ["rows_used", "rows_dropped"])
def test_ev_ho_005_row_disagreement_between_roles_is_unmeasured(tmp_path: Path, disagree_in: str) -> None:
    request = make_request(tmp_path)
    base_result = heldout_result(2.0, rows_used=25, rows_dropped=1)
    checkpoint_result = heldout_result(1.5, rows_used=25, rows_dropped=1)
    checkpoint_result[disagree_in] = base_result[disagree_in] + 1  # one role saw a different slice
    _, skill, ctx, result = run_heldout(
        tmp_path, request,
        base=HeldoutOutcome(rc=0, result=base_result, stderr_tail=""),
        checkpoint=HeldoutOutcome(rc=0, result=checkpoint_result, stderr_tail=""),
    )

    assert result.status is Status.UNMEASURED
    assert "EV-HO-005" in said(skill, result, ctx)


def test_ev_ho_005_quiet_when_both_roles_scored_the_same_rows(tmp_path: Path) -> None:
    request = make_request(tmp_path)
    _, skill, ctx, result = run_heldout(
        tmp_path, request,
        base=outcome(2.0, rows_used=25, rows_dropped=1),
        checkpoint=outcome(1.5, rows_used=25, rows_dropped=1),
    )

    assert result.status is Status.PASS
    assert "EV-HO-005" not in said(skill, result, ctx)


@pytest.mark.parametrize("missing", ["rows_used", "rows_dropped", "mean_nll_nats"])
def test_ev_ho_006_incomplete_rows_are_reported_red(tmp_path: Path, missing: str) -> None:
    request = make_request(tmp_path)
    base_result = heldout_result(2.0)
    checkpoint_result = heldout_result(1.5)
    del base_result[missing]
    del checkpoint_result[missing]  # symmetric: nothing disagrees, a row is just incomplete
    _, skill, ctx, result = run_heldout(
        tmp_path, request,
        base=HeldoutOutcome(rc=0, result=base_result, stderr_tail=""),
        checkpoint=HeldoutOutcome(rc=0, result=checkpoint_result, stderr_tail=""),
    )

    assert result.status is Status.RED
    assert "EV-HO-006" in said(skill, result, ctx)


def test_ev_ho_006_quiet_with_complete_rows(tmp_path: Path) -> None:
    request = make_request(tmp_path)
    _, skill, ctx, result = run_heldout(
        tmp_path, request,
        base=outcome(2.0),
        checkpoint=outcome(1.5),
    )

    assert result.status is Status.PASS
    assert "EV-HO-006" not in said(skill, result, ctx)


def test_exit_96_from_the_script_is_a_skill_refusal(tmp_path: Path) -> None:
    request = make_request(tmp_path, rows=VIDEO_ROWS, video_frames=16)
    refused = HeldoutOutcome(rc=96, result=None,
                             stderr_tail="refused: video rows present but --video-frames not given")
    _, skill, ctx, result = run_heldout(tmp_path, request, base=refused, checkpoint=refused)

    assert result.status is Status.REFUSED
    assert result.refusal
    assert "video rows" in said(skill, result, ctx)  # the eval script's exit-96 reason is surfaced
    assert not Path(request["out"]).exists()


# == skill: one-sided abs-epsilon verdict on identical rows ======================================================


def test_same_rows_pass_only_when_the_gain_beats_abs_epsilon(tmp_path: Path) -> None:
    request = make_request(tmp_path)  # abs_epsilon 0.25
    fake, _, _, result = run_heldout(tmp_path, request,
                                     base=outcome(2.0), checkpoint=outcome(1.5))  # gain 0.5 > 0.25

    assert result.status is Status.PASS
    row = heldout_row(result.payload)
    assert row["name"] == "heldout_nll" and row["metric"] == "mean_nll_nats"
    assert row["score"] == pytest.approx(1.5)  # tuned mean NLL
    assert row["baseline"] == pytest.approx(2.0)  # base mean NLL
    assert row["higher_is_better"] is False
    assert row["rows_used"] == row["baseline_rows_used"] == 25
    assert row["rows_dropped"] == row["baseline_rows_dropped"] == 1
    assert row["tokens_scored"] == 1280
    assert row["perplexity"] == pytest.approx(math.exp(1.5))
    assert row["baseline_perplexity"] == pytest.approx(math.exp(2.0))
    assert str(row["status"]).lower() == "pass"
    assert Path(request["out"]).exists()

    # base and checkpoint roles see the very same rows and knobs
    base_call, checkpoint_call = fake.calls("base"), fake.calls("checkpoint")
    for field in ("data", "max_length", "video_frames", "image_column", "video_column", "limit"):
        assert getattr(base_call, field) == getattr(checkpoint_call, field), field
    assert base_call.adapter is None
    assert checkpoint_call.adapter is None  # a full checkpoint carries no adapter either


def test_a_gain_inside_the_band_passes_as_non_regression(tmp_path: Path) -> None:
    """The evaluation skill is a non-regression gate (as in lm-eval mode): PASS means the checkpoint is not worse
    than base beyond abs_epsilon. Claiming a gain is measured_or_unmeasured / auto_research territory."""
    request = make_request(tmp_path)  # abs_epsilon 0.25
    _, _, _, result = run_heldout(tmp_path, request, base=outcome(2.0), checkpoint=outcome(1.875))
    assert result.status is Status.PASS
    row = heldout_row(result.payload)
    assert row["score"] == pytest.approx(1.875) and row["baseline"] == pytest.approx(2.0)


def test_a_regression_beyond_abs_epsilon_is_red(tmp_path: Path) -> None:
    request = make_request(tmp_path)  # abs_epsilon 0.25
    _, _, _, result = run_heldout(tmp_path, request, base=outcome(2.0), checkpoint=outcome(2.5))  # NLL up by 0.5
    assert result.status is Status.RED
    assert str(heldout_row(result.payload)["status"]).lower() in ("red", "fail")


def test_heldout_report_binds_the_scoring_script_not_lm_eval(tmp_path: Path) -> None:
    """Real-GPU finding (A4.1): the held-out report copied lm-eval's pinned 0.4.12 as its harness version.
    Held-out mode never runs lm-eval; its provenance is the scoring script, bound by sha256."""
    from foundationskills.skills.evaluation.skill import _heldout_harness

    script = tmp_path / "eval_heldout_loss.py"
    script.write_text("print('x')\n")
    harness = _heldout_harness({"heldout_script": str(script)})
    assert harness["name"] == "fs-heldout" and harness["pinned"] is None
    assert harness["version"].startswith("sha256:") and "0.4.12" not in str(harness)
