"""Whole-call WER for NeMo chunked long-form inference, scored by FoundationScale.

Inputs are NeMo output manifests (original fields + `pred_text`) for a base and a tuned model
over the same calls. Text goes through FoundationScale's normalizer
(`speech_metrics.normalize_transcript`). The edit distance on whole calls (about 8k words) is a
rolling-row Levenshtein: O(m) memory, the same total as `speech_metrics.align`, but no S/D/I
breakdown. `align` keeps its full table for the backtrace, which is O(n*m) memory and
impractical here. The two are re-checked against each other on a 300-word slice of every call,
and the run refuses if they disagree.

Reported per call and over all calls: WER, the hypothesis/reference length ratio (well under 1
means audio was dropped; far over 1 means a runaway), and the speech.runaway_hypotheses and the
speech.repetition_loops verdicts of tuned vs base over the calls (the loops are counted by
count_loop_words over the same calls: a repetition loop inside one 40 s chunk of an hour-long call
adds 4-12% of that call's words and sails under the whole-call runaway yardstick). Calls missing
from a prediction file are UNDERCOVERED, never skipped. Exit 5 when either gate blocks.
Usage: score_longform.py --manifest M.json --base pred_base.json --tuned pred_tuned.json [--json OUT]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from foundationscale.gates.speech_gates import (
    RepetitionLoopContext,
    RepetitionLoopGate,
    RunawayHypothesisContext,
    RunawayHypothesisGate,
)
from foundationscale.train.speech_metrics import (
    count_loop_words,
    count_runaway,
    normalize_transcript,
    word_errors,
)


def load(path: str) -> dict[str, dict]:
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    return {str(r.get("id") or Path(r["audio_filepath"]).stem): r for r in rows}


def errors(ref_words: list[str], hyp_words: list[str]) -> int:
    if not ref_words:
        raise SystemExit("REFUSE (96): a call with an empty reference cannot carry a WER")
    prev = list(range(len(hyp_words) + 1))
    for i, r in enumerate(ref_words, 1):
        cur = [i] + [0] * len(hyp_words)
        for j, h in enumerate(hyp_words, 1):
            cur[j] = min(prev[j - 1] + (r != h), prev[j] + 1, cur[j - 1] + 1)
        prev = cur
    return prev[-1]


def cross_check(ref_words: list[str], hyp_words: list[str], call: str) -> None:
    ref, hyp = ref_words[:300], hyp_words[:300]
    fs = word_errors(" ".join(ref), " ".join(hyp)).errors
    rolling = errors(ref, hyp)
    if fs != rolling:
        raise SystemExit(f"REFUSE (96): FS align ({fs}) and rolling ({rolling}) disagree on call {call}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--tuned", required=True)
    ap.add_argument("--json")
    args = ap.parse_args()
    calls = load(args.manifest)
    preds = {"base": load(args.base), "tuned": load(args.tuned)}
    report: dict[str, dict] = {}
    totals = {"base": [0, 0], "tuned": [0, 0]}
    pairs: dict[str, list[tuple[str, str]]] = {"base": [], "tuned": []}
    for call, row in sorted(calls.items()):
        ref = normalize_transcript(row["text"])
        report[call] = {"hours": round(row["duration"] / 3600, 2), "ref_words": len(ref)}
        for side in ("base", "tuned"):
            p = preds[side].get(call)
            if p is None:
                report[call][side] = "MISSING"
                continue
            hyp = normalize_transcript(p.get("pred_text", ""))
            cross_check(ref, hyp, call)
            e = errors(ref, hyp)
            totals[side][0] += e
            totals[side][1] += len(ref)
            pairs[side].append((row["text"], p.get("pred_text", "")))
            report[call][side] = {"wer": round(e / len(ref), 4), "len_ratio": round(len(hyp) / len(ref), 3)}
    both = [c for c in calls if all(isinstance(report[c][s], dict) for s in ("base", "tuned"))]
    gate = RunawayHypothesisGate().run(RunawayHypothesisContext(
        rows_expected=len(calls), rows_checked=len(both),
        base_runaway=count_runaway(pairs["base"]).rows_runaway if both else 0,
        tuned_runaway=count_runaway(pairs["tuned"]).rows_runaway if both else 0,
    ))
    # Loops are LOCAL: a repetition loop inside one 40 s NeMo chunk of an
    # hour-long call adds 4-12% of that call's words and sails under the
    # whole-call runaway yardstick above, so the loop words are counted over the
    # same scored calls and judged against the base model's own count.
    base_loops = count_loop_words(pairs["base"])
    tuned_loops = count_loop_words(pairs["tuned"])
    loop_gate = RepetitionLoopGate().run(RepetitionLoopContext(
        rows_expected=len(calls), rows_checked=len(both),
        reference_words=base_loops.reference_words if both else 0,
        base_loop_words=base_loops.hypothesis_loop_words if both else 0,
        tuned_loop_words=tuned_loops.hypothesis_loop_words if both else 0,
    ))
    for call, r in report.items():
        print(call, r)
    for side in ("base", "tuned"):
        e, n = totals[side]
        print(f"{side}: WER {e / n:.4f} over {n} reference words" if n else f"{side}: nothing scored")
    print(f"[{gate.verdict.value}] {gate.gate_id}: {gate.coverage.checked}/{gate.coverage.expected} "
          f"{gate.coverage.unit} -- {gate.detail}")
    print(f"[{loop_gate.verdict.value}] {loop_gate.gate_id}: "
          f"{loop_gate.coverage.checked}/{loop_gate.coverage.expected} "
          f"{loop_gate.coverage.unit} -- {loop_gate.detail}")
    if args.json:
        Path(args.json).write_text(json.dumps({"calls": report, "totals": totals,
                                               "runaway": gate.verdict.value,
                                               "repetition_loops": loop_gate.verdict.value},
                                              indent=2))
    return 5 if (gate.blocking or loop_gate.blocking) else 0


if __name__ == "__main__":
    sys.exit(main())
