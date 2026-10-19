"""``speech-longform`` suite (A4.3): whole-call long-form ASR scoring of a base and a tuned model's predictions.

A faithful composition of the measured campaign scorer (validation_campaigns/speech_longform/score_longform.py) over
FoundationScale's own pieces, never a reimplementation of them:
- ``foundationscale.train.speech_metrics``: ``normalize_transcript``, ``word_errors`` (cross-check), ``count_runaway``,
  ``count_loop_words``;
- ``foundationscale.gates.speech_gates``: ``RunawayHypothesisGate`` and ``RepetitionLoopGate``.

Inputs are JSONL: a manifest of whole calls ``{id | audio_filepath, text, duration}`` and two NeMo output manifests
(the same rows plus ``pred_text``). WER = total word edit errors / total reference words, by a rolling edit distance
cross-checked against FS's aligner on the first 300 words of every call. A call with no prediction is MISSING
(undercoverage), never skipped. Verdict: a FAILing gate is RED whatever the WER (measured: Canary-1B-flash beam-4 had
the best WER, 12.81%, and 285 loop words, so it FAILED; EVIDENCE.md:116-121); otherwise any MISSING call is UNMEASURED;
otherwise PASS.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

SUITE = "speech-longform"


class SpeechScoreRefused(ValueError):
    """The inputs cannot carry a WER (empty reference, unreadable file, the two edit distances disagree)."""


def _load(path: str) -> dict[str, dict]:
    try:
        rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    except (OSError, ValueError) as exc:
        raise SpeechScoreRefused(f"{path}: unreadable JSONL ({exc})") from exc
    out: dict[str, dict] = {}
    for row in rows:
        key = row.get("id") or (Path(row["audio_filepath"]).stem if row.get("audio_filepath") else None)
        if key is None:
            raise SpeechScoreRefused(f"{path}: a row has neither id nor audio_filepath")
        out[str(key)] = row
    return out


def _errors(ref: list[str], hyp: list[str]) -> int:
    if not ref:
        raise SpeechScoreRefused("a call with an empty reference cannot carry a WER")
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        cur = [i] + [0] * len(hyp)
        for j, h in enumerate(hyp, 1):
            cur[j] = min(prev[j - 1] + (r != h), prev[j] + 1, cur[j - 1] + 1)
        prev = cur
    return prev[-1]


def _gate(result: Any) -> dict[str, Any]:
    return {"gate_id": result.gate_id, "verdict": result.verdict.value, "blocking": bool(result.blocking),
            "detail": result.detail, "checked": result.coverage.checked, "expected": result.coverage.expected,
            "unit": result.coverage.unit}


def score(manifest: str, base_predictions: str, tuned_predictions: str) -> dict[str, Any]:
    """The long-form report: per-call WER and length ratio, totals, both gates, missing calls and the verdict."""
    from foundationscale.gates.speech_gates import (  # noqa: PLC0415
        RepetitionLoopContext,
        RepetitionLoopGate,
        RunawayHypothesisContext,
        RunawayHypothesisGate,
    )
    from foundationscale.train.speech_metrics import (  # noqa: PLC0415
        count_loop_words,
        count_runaway,
        normalize_transcript,
        word_errors,
    )

    calls = _load(manifest)
    preds = {"base": _load(base_predictions), "tuned": _load(tuned_predictions)}
    per_call: dict[str, dict[str, Any]] = {}
    totals = {side: {"errors": 0, "ref_words": 0} for side in preds}
    pairs: dict[str, list[tuple[str, str]]] = {side: [] for side in preds}
    missing: dict[str, list[str]] = {side: [] for side in preds}
    for call, row in sorted(calls.items()):
        ref = normalize_transcript(row["text"])
        per_call[call] = {"hours": round(float(row.get("duration") or 0) / 3600, 2), "ref_words": len(ref)}
        for side in preds:
            pred = preds[side].get(call)
            if pred is None:
                per_call[call][side] = "MISSING"
                missing[side].append(call)
                continue
            hyp = normalize_transcript(pred.get("pred_text", ""))
            fs_errors = word_errors(" ".join(ref[:300]), " ".join(hyp[:300])).errors
            if fs_errors != _errors(ref[:300], hyp[:300]):
                raise SpeechScoreRefused(f"FS align and the rolling edit distance disagree on call {call}")
            errors = _errors(ref, hyp)
            totals[side]["errors"] += errors
            totals[side]["ref_words"] += len(ref)
            pairs[side].append((row["text"], pred.get("pred_text", "")))
            per_call[call][side] = {"wer": round(errors / len(ref), 4), "len_ratio": round(len(hyp) / len(ref), 3)}
    both = [c for c in calls if all(isinstance(per_call[c][s], dict) for s in preds)]
    runaway = RunawayHypothesisGate().run(RunawayHypothesisContext(
        rows_expected=len(calls), rows_checked=len(both),
        base_runaway=count_runaway(pairs["base"]).rows_runaway if both else 0,
        tuned_runaway=count_runaway(pairs["tuned"]).rows_runaway if both else 0))
    base_loops, tuned_loops = count_loop_words(pairs["base"]), count_loop_words(pairs["tuned"])
    loops = RepetitionLoopGate().run(RepetitionLoopContext(
        rows_expected=len(calls), rows_checked=len(both),
        reference_words=base_loops.reference_words if both else 0,
        base_loop_words=base_loops.hypothesis_loop_words if both else 0,
        tuned_loop_words=tuned_loops.hypothesis_loop_words if both else 0))
    for side in totals:
        n = totals[side]["ref_words"]
        totals[side]["wer"] = round(totals[side]["errors"] / n, 4) if n else None
    gates = {"runaway": _gate(runaway), "repetition_loops": _gate(loops)}
    # FS gates block on FAIL and on UNDERCOVERED; only a FAIL is a measured failure (RED). Undercoverage (a call
    # with no prediction) is UNMEASURED: it is never a pass, and it is not a measured regression either.
    if any(g["verdict"] == "FAIL" for g in gates.values()):
        verdict = "RED"
    elif any(missing.values()) or any(g["blocking"] for g in gates.values()):
        verdict = "UNMEASURED"
    else:
        verdict = "PASS"
    return {"suite": SUITE, "calls": per_call, "totals": totals, "gates": gates, "missing": missing,
            "loop_words": {"base": base_loops.hypothesis_loop_words, "tuned": tuned_loops.hypothesis_loop_words,
                           "reference_words": base_loops.reference_words},
            "verdict": verdict}
