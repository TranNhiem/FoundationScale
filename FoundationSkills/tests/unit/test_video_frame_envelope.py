"""Measured video-frame envelope (A4.4): every frame count beyond max_frames=16 is UNMEASURED,
never PASS (or RED when a measured regression outranks the envelope verdict).  Envelope unit,
TrainingPlanner, TrainingEmit and EvalSkill; no subprocess is spawned and ``foundationscale`` is
imported only to branch, never skipped."""
from __future__ import annotations

import importlib.util
import inspect
import json
import math
import sys
from pathlib import Path
from typing import Any

import pytest

from foundationskills.core import SkillContext, Status
from foundationskills.interfaces.fs.capabilities import FSCapabilities, _probe_video_key
from foundationskills.interfaces.fs.video_envelope import (
    MEASURED_FRAME_ENVELOPE, beyond_envelope, frame_envelope,
)
from foundationskills.skills.evaluation.policy import load_policy
from foundationskills.skills.evaluation.runner import PINNED_LM_EVAL, HeldoutOutcome, HeldoutRequest
from foundationskills.skills.evaluation.skill import EvalSkill
from foundationskills.skills.training.emit_skill import TrainingEmitSkill
from foundationskills.skills.training.skill import TrainingPlannerSkill

IMAGE_ROWS = [{"text": "one", "image": "a.png"}, {"text": "two", "image": "b.png"}]
VIDEO_ROWS = [{"text": "clip", "image": "c.png", "video": "clip.mp4"}]
RUN_MANIFEST = {"environment": {"values": {"FOUNDATIONSCALE_TRAIN_VIDEO_FRAMES": "16"}}}


def _fake_caps() -> FSCapabilities:
    return FSCapabilities(available=True, fs_version="0.0.0+77bfa65",
        train_flags=frozenset({"--model", "--dataset", "--output-dir"}),
        train_objectives=("sft",), sharding_strategies=("ddp", "fsdp"),
        executed_axes=("tp", "cp"), refused_axes=("pp", "ep"), axes_measured=True,
        rl_algorithms=("dr_grpo", "gspo", "dapo"),
        rl_runnable={"dr_grpo": None, "gspo": None, "dapo": None},
        families={"gemma4": ("gemma4",)}, backends=("ddp", "fsdp"))


# every flag the emitted spec may carry (copied from tests/unit/fs_interface/test_emit_skill.py)
_ALL_FLAGS = frozenset(
    {
        "--model", "--dataset", "--output-dir", "--max-steps",
        "--per-device-batch-size", "--learning-rate", "--save-interval",
        "--max-sequence-length", "--objective", "--seed", "--dp",
        "--tp", "--pp", "--ep", "--cp", "--nodes", "--gpus-per-node",
        "--profile-name", "--profile-path", "--dry-run", "--precision",
        "--adapter", "--adapter-rank", "--adapter-alpha", "--adapter-target",
        "--adapter-dropout", "--optimizer", "--gradient-accumulation-steps",
        "--lr-scheduler-type", "--warmup-steps", "--sharding-strategy",
        "--gradient-checkpointing",
    }
)


def _emit_caps() -> FSCapabilities:
    return FSCapabilities(available=True, fs_version="0.1.0-test", train_flags=_ALL_FLAGS,
        train_objectives=("sft",), sharding_strategies=("ddp", "fsdp"),
        executed_axes=("tp", "cp"), refused_axes=("pp", "ep"), axes_measured=True,
        families={"gemma4": ("gemma4",)}, backends=("ddp", "fsdp"))


def make_ctx(tmp: Path, caps: FSCapabilities | None = None) -> SkillContext:
    return SkillContext(workdir=tmp, capabilities=caps or _fake_caps())


def _frames(obj: Any, n: int) -> Any:
    """Rebuild ``obj`` with every ``video_frames`` key set to ``n`` (request goal and plan stages)."""
    if isinstance(obj, dict):
        return {k: (n if k == "video_frames" else _frames(v, n)) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_frames(v, n) for v in obj]
    return obj


def flatten(x: Any) -> str:
    if isinstance(x, dict):
        return " ".join(flatten(v) for v in x.values())
    if isinstance(x, (list, tuple)):
        return " ".join(flatten(v) for v in x)
    return str(x)


def said(_skill: Any, result: Any, _ctx: Any) -> str:
    parts = [repr(f) for f in getattr(result, "findings", []) or []]
    for k in ("refusal", "notes", "reason"):
        v = getattr(result, k, None)
        if isinstance(v, str):
            parts.append(v)
    return " | ".join(parts)


def heldout_result(mean_nll: float = 2.0, rows_used: int = 25, rows_dropped: int = 1) -> dict[str, Any]:
    t = 1280
    return {"metric": "heldout_nll",
            "parameters": {"total": 8_000_000_000, "trainable": 4_000_000},
            "rows_read": rows_used + rows_dropped, "rows_used": rows_used,
            "rows_dropped": rows_dropped, "drop_reasons": {}, "tokens_scored": t,
            "total_nll": mean_nll * t, "mean_nll_nats": mean_nll,
            "mean_nll_bits": mean_nll / math.log(2),
            "perplexity": math.exp(mean_nll), "peak_gpu_memory_gb": 42.0, "wall_seconds": 31.5,
            "versions": {"python": "3.11.8", "torch": "2.3.0", "transformers": "4.41.2"}}


def outcome(mean_nll: float = 2.0, rows_used: int = 25, rows_dropped: int = 1,
            rc: int = 0, stderr_tail: str = "") -> HeldoutOutcome:
    if rc != 0:
        return HeldoutOutcome(rc=rc, result=None, stderr_tail=stderr_tail)
    return HeldoutOutcome(rc=rc, result=heldout_result(mean_nll, rows_used, rows_dropped),
                          stderr_tail=stderr_tail)


def make_request(tmp: Path, rows: list[dict[str, Any]] | None = None, **over: Any) -> dict[str, Any]:
    rows = IMAGE_ROWS if rows is None else rows
    for n in ("base-model", "checkpoint", "eval-cache"):
        (tmp / n).mkdir(exist_ok=True)
    (tmp / "run_manifest.json").write_text(json.dumps(RUN_MANIFEST), encoding="utf-8")
    data = tmp / "heldout.jsonl"
    data.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    script = tmp / "eval_heldout_loss.py"
    script.write_text("# stub\n", encoding="utf-8")
    request: dict[str, Any] = {
        "checkpoint": str(tmp / "checkpoint"), "base": str(tmp / "base-model"),
        "benchmarks": [], "eval_cache": str(tmp / "eval-cache"),
        "baseline_cache": str(tmp / "baseline-cache"), "out": str(tmp / "eval_report.json"),
        "run_manifest": str(tmp / "run_manifest.json"),
        "heldout": {"data": str(data), "max_length": 2048, "video_frames": None,
                    "image_column": "image", "video_column": "video", "limit": None,
                    "heldout_script": str(script), "abs_epsilon": 0.25, **over},
    }
    pol = tmp / "eval_policy.yaml"
    for doc in ('{"version": "1", "tasks": {}}', '{"tasks": {}}', '{"version": "1"}', "{}"):
        pol.write_text(doc, encoding="utf-8")
        try:
            load_policy(str(pol))
        except Exception:
            continue
        request["policy"] = str(pol)
        break
    return request


class _FakeHeldoutRunner:
    def __init__(self, request: dict[str, Any], base: HeldoutOutcome | None = None,
                 checkpoint: HeldoutOutcome | None = None) -> None:
        self.base_model = Path(request["base"])
        self.checkpoint_model = Path(request["checkpoint"])
        self.outcomes = {"base": base or outcome(2.0), "checkpoint": checkpoint or outcome(1.5)}

    def run(self, request: HeldoutRequest) -> HeldoutOutcome:
        if getattr(request, "adapter", None) is not None or Path(request.model) == self.checkpoint_model:
            return self.outcomes["checkpoint"]
        return self.outcomes["base"]


def _skill_over(f: _FakeHeldoutRunner) -> EvalSkill:
    params = inspect.signature(EvalSkill.__init__).parameters
    kw: dict[str, Any] = {"runner": f, "version_probe": lambda: PINNED_LM_EVAL}
    for n in ("heldout_runner", "heldout", "vlm_heldout_runner"):
        if n in params:
            kw[n] = f
            break
    return EvalSkill(**kw)


def run_heldout(tmp: Path, request: dict[str, Any], **out: HeldoutOutcome):
    fake = _FakeHeldoutRunner(request, **out)
    skill = _skill_over(fake)
    ctx = make_ctx(tmp)
    return skill, ctx, skill.run(request, ctx)


def _invoke(cls: Any, name: str, tmp: Path, monkeypatch, frames: int | None = None):
    skill = cls()
    fixture = skill.must_fire_fixtures()[name]
    if frames is not None:
        fixture = _frames(fixture, frames)
    if "fake_plan" in fixture:
        mod = sys.modules[cls.__module__]
        if hasattr(mod, "plan"):
            monkeypatch.setattr(mod, "plan", lambda *a, **kw: fixture["fake_plan"])
    caps = _emit_caps() if cls is TrainingEmitSkill else None
    return skill.execute(dict(fixture["request"]), make_ctx(tmp, caps))


# == envelope unit ==


def test_frame_envelope_is_a_copy_of_the_measured_ledger() -> None:
    env = frame_envelope()
    assert env["max_frames"] == MEASURED_FRAME_ENVELOPE["max_frames"] == 16
    env["max_frames"] = 1
    assert MEASURED_FRAME_ENVELOPE["max_frames"] == 16


@pytest.mark.parametrize("raw", [None, "x", True, 16, 1])
def test_beyond_envelope_in_scope_or_unparseable_is_none(raw: Any) -> None:
    assert beyond_envelope(raw) is None


def test_beyond_envelope_flags_17_as_unmeasured() -> None:
    msg = beyond_envelope(17)
    assert msg is not None and "UNMEASURED" in msg and "16" in msg


def test_probe_video_key_measured_or_failure() -> None:
    key = _probe_video_key()
    value = key.get("value")
    if importlib.util.find_spec("foundationscale") is not None:
        assert value["frame_envelope"]["max_frames"] == 16
        return
    # Without foundationscale the probe must not invent numbers; it reports failure.
    flat = flatten(value).lower()
    fe = value.get("frame_envelope") if isinstance(value, dict) else None
    claims_measured = isinstance(fe, dict) and isinstance(fe.get("max_frames"), int)
    has_failure = any(kw in flat for kw in ("fail", "error", "unavailable", "import", "not importable"))
    assert has_failure or not claims_measured, f"probe should report a failure, got {value!r}"


# == TrainingPlannerSkill ==


def test_tr_pl_010_video_over_envelope_is_unmeasured(monkeypatch, tmp_path) -> None:
    r = _invoke(TrainingPlannerSkill, "TR-PL-010", tmp_path, monkeypatch)
    assert r.status is Status.UNMEASURED and r.exit_code == 95
    assert "TR-PL-010" in {f.rule_id for f in r.findings}


def test_tr_pl_010_quiet_at_measured_frames(monkeypatch, tmp_path) -> None:
    r = _invoke(TrainingPlannerSkill, "TR-PL-010", tmp_path, monkeypatch, frames=16)
    assert "TR-PL-010" not in {f.rule_id for f in r.findings}
    assert r.status is not Status.UNMEASURED


def test_tr_in_006_refused_names_the_rule(monkeypatch, tmp_path) -> None:
    r = _invoke(TrainingPlannerSkill, "TR-IN-006", tmp_path, monkeypatch)
    assert r.status is Status.REFUSED and r.exit_code == 96
    assert "TR-IN-006" in r.refusal


# == TrainingEmitSkill (FS-IN-006 doubles as the emit_train ValueError case) ==


def test_fs_in_006_refused_missing_video_frames(monkeypatch, tmp_path) -> None:
    """"FS-IN-006" surfaces the documented emit_train ValueError (a video stage with no
    ``video_frames``) as a refusal; tested through TrainingEmitSkill because emit_train's
    signature is not visible here and guessing it is forbidden."""
    r = _invoke(TrainingEmitSkill, "FS-IN-006", tmp_path, monkeypatch)
    assert r.status is Status.REFUSED and r.exit_code == 96
    assert "FS-IN-006" in r.refusal


def test_fs_ho_007_video_over_envelope_is_unmeasured(monkeypatch, tmp_path) -> None:
    r = _invoke(TrainingEmitSkill, "FS-HO-007", tmp_path, monkeypatch)
    assert r.status is Status.UNMEASURED and r.exit_code == 95
    assert "FS-HO-007" in {f.rule_id for f in r.findings}


def test_fs_ho_007_quiet_at_measured_frames(monkeypatch, tmp_path) -> None:
    r = _invoke(TrainingEmitSkill, "FS-HO-007", tmp_path, monkeypatch, frames=16)
    assert "FS-HO-007" not in {f.rule_id for f in r.findings}
    assert r.status is Status.PASS, [(f.rule_id, f.message) for f in r.findings]
    assert r.payload["specs"][0]["env"]["FOUNDATIONSCALE_TRAIN_VIDEO_FRAMES"] == "16"


# == EvalSkill held-out ==


def test_ev_ho_009_video_rows_over_envelope_is_unmeasured(tmp_path) -> None:
    request = make_request(tmp_path, VIDEO_ROWS, video_frames=32)
    skill, ctx, r = run_heldout(tmp_path, request, base=outcome(2.0), checkpoint=outcome(1.5))
    assert r.status is Status.UNMEASURED
    assert "EV-HO-009" in said(skill, r, ctx)


def test_ev_ho_009_quiet_at_measured_frames(tmp_path) -> None:
    request = make_request(tmp_path, VIDEO_ROWS, video_frames=16)
    skill, ctx, r = run_heldout(tmp_path, request, base=outcome(2.0), checkpoint=outcome(1.5))
    assert r.status is Status.PASS
    assert "EV-HO-009" not in said(skill, r, ctx)


def test_ev_ho_009_quiet_without_video_rows(tmp_path) -> None:
    request = make_request(tmp_path, IMAGE_ROWS, video_frames=32)
    skill, ctx, r = run_heldout(tmp_path, request, base=outcome(2.0), checkpoint=outcome(1.5))
    assert r.status is Status.PASS
    assert "EV-HO-009" not in said(skill, r, ctx)


def test_ev_ho_009_measured_regression_stays_red(tmp_path) -> None:
    request = make_request(tmp_path, VIDEO_ROWS, video_frames=32)
    skill, ctx, r = run_heldout(tmp_path, request, base=outcome(2.0), checkpoint=outcome(4.0))
    assert r.status is Status.RED
