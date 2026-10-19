"""A4.3 `--suite speech-longform`: FS WER + gates over base/tuned predictions; a loop gate FAIL is RED whatever the WER."""

from __future__ import annotations

import json
from pathlib import Path

from foundationskills.core.contract import SkillContext
from foundationskills.core.status import Status
from foundationskills.skills.evaluation.skill import EvalSkill

WORDS = ("revenue grew in the quarter driven by strong demand across our cloud and data center segments "
         "while margins improved on lower component costs and better pricing ").split()


def _ref(n: int, offset: int) -> str:
    return " ".join(WORDS[(i + offset) % len(WORDS)] for i in range(n))


def _write(path: Path, rows: list[dict]) -> str:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return str(path)


def _suite(tmp_path: Path, tuned_text: dict[str, str] | None = None, drop: str | None = None) -> dict:
    calls = {f"call{i}": _ref(1000, i * 7) for i in range(3)}
    manifest = _write(tmp_path / "longform.jsonl", [{"id": c, "text": t, "duration": 1800.0} for c, t in calls.items()])
    base = _write(tmp_path / "base.jsonl", [{"id": c, "text": t, "pred_text": t} for c, t in calls.items()])
    tuned_rows = [{"id": c, "text": t, "pred_text": (tuned_text or {}).get(c, t)} for c, t in calls.items() if c != drop]
    tuned = _write(tmp_path / "tuned.jsonl", tuned_rows)
    return {"suite": "speech-longform", "manifest": manifest, "base_predictions": base, "tuned_predictions": tuned,
            "out": str(tmp_path / "eval" / "speech_longform.json")}


def test_identical_predictions_pass(tmp_path) -> None:
    result = EvalSkill().execute(_suite(tmp_path), SkillContext(workdir=tmp_path))
    assert result.status is Status.PASS, [f.message for f in result.findings]
    assert result.payload["benchmarks"][0]["score"] == 0.0


def test_a_repetition_loop_is_red_whatever_the_wer(tmp_path) -> None:
    # call0's tuned transcript is perfect except one inserted loop of 30 words: WER 1% overall, still RED
    looped = _ref(1000, 0).split()
    looped[500:500] = ["thank"] * 30
    req = _suite(tmp_path, tuned_text={"call0": " ".join(looped)})
    result = EvalSkill().execute(req, SkillContext(workdir=tmp_path))
    assert result.status is Status.RED and "EV-HO-007" in [f.rule_id for f in result.findings]
    assert result.payload["benchmarks"][0]["score"] < 0.02  # the WER is good; the loop still blocks
    assert result.payload["speech_longform"]["gates"]["repetition_loops"]["blocking"] is True


def test_a_call_without_a_prediction_is_unmeasured(tmp_path) -> None:
    result = EvalSkill().execute(_suite(tmp_path, drop="call2"), SkillContext(workdir=tmp_path))
    assert result.status is Status.UNMEASURED and "EV-HO-008" in [f.rule_id for f in result.findings]


def test_a_missing_prediction_file_is_refused(tmp_path) -> None:
    req = _suite(tmp_path)
    req["tuned_predictions"] = str(tmp_path / "nope.jsonl")
    result = EvalSkill().execute(req, SkillContext(workdir=tmp_path))
    assert result.status is Status.REFUSED and "EV-IN-009" in [f.rule_id for f in result.findings]



def test_negative_control_is_recorded_and_a_blind_census_refuses(tmp_path, monkeypatch) -> None:
    """A4.5: the hand-added looping control must be caught before any call is scored."""
    import dataclasses

    import foundationscale.train.speech_metrics as metrics

    request = _suite(tmp_path)
    result = EvalSkill().execute(request, SkillContext(workdir=tmp_path))
    control = result.payload["speech_longform"]["negative_control"]
    assert control["gate_verdict"] == "FAIL" and control["loop_words"] > 0

    real = metrics.count_loop_words
    monkeypatch.setattr(metrics, "count_loop_words",
                        lambda pairs: dataclasses.replace(real(pairs), hypothesis_loop_words=0))
    blind = EvalSkill().execute({**request, "out": str(tmp_path / "eval" / "blind.json")}, SkillContext(workdir=tmp_path))
    assert blind.status is Status.REFUSED and "negative control" in (blind.refusal or "")
