"""Pick a long-form model on a held-out DEV set, under both speech gates; never look at test.

Every candidate (a seed x checkpoint decode of the dev calls) is scored with the long-form scorer's
own code: whole-call WER, speech.runaway_hypotheses and speech.repetition_loops against the base
model's decode of the same calls. Candidates that a gate blocks are out whatever their WER. Among
the rest, the lowest dev WER wins. The printout lists every candidate, so the pass rate of a
recipe is visible, not only its best draw.
Usage: select_under_gates.py --manifest dev.json --base dev_base.json CAND.json [CAND.json ...]
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from score_longform import errors, load

from foundationscale.gates.speech_gates import (
    RepetitionLoopContext,
    RepetitionLoopGate,
    RunawayHypothesisContext,
    RunawayHypothesisGate,
)
from foundationscale.train.speech_metrics import count_loop_words, count_runaway, normalize_transcript


def score(calls: dict[str, dict], preds: dict[str, dict]) -> tuple[float | None, list[tuple[str, str]]]:
    e = n = 0
    pairs = []
    for call, row in calls.items():
        p = preds.get(call)
        if p is None:
            return None, pairs
        ref = normalize_transcript(row["text"])
        e += errors(ref, normalize_transcript(p.get("pred_text", "")))
        n += len(ref)
        pairs.append((row["text"], p.get("pred_text", "")))
    return e / n, pairs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--json")
    ap.add_argument("candidates", nargs="+")
    args = ap.parse_args()
    calls = load(args.manifest)
    base_wer, base_pairs = score(calls, load(args.base))
    if base_wer is None:
        raise SystemExit("REFUSE (96): the base decode is missing a dev call")
    base_run = count_runaway(base_pairs).rows_runaway
    base_loop = count_loop_words(base_pairs)
    table = []
    # Whole-call edit distance is about a minute per call in pure Python: candidates in parallel.
    with ProcessPoolExecutor(max_workers=min(32, len(args.candidates))) as pool:
        scored = list(pool.map(score, [calls] * len(args.candidates),
                               [load(p) for p in args.candidates]))
    for path, (wer, pairs) in zip(args.candidates, scored, strict=True):
        name = Path(path).stem
        if wer is None:
            table.append({"candidate": name, "status": "MISSING CALL"})
            continue
        run = RunawayHypothesisGate().run(RunawayHypothesisContext(
            rows_expected=len(calls), rows_checked=len(pairs),
            base_runaway=base_run, tuned_runaway=count_runaway(pairs).rows_runaway))
        loops = count_loop_words(pairs)
        loop = RepetitionLoopGate().run(RepetitionLoopContext(
            rows_expected=len(calls), rows_checked=len(pairs),
            reference_words=loops.reference_words,
            base_loop_words=base_loop.hypothesis_loop_words,
            tuned_loop_words=loops.hypothesis_loop_words))
        table.append({"candidate": name, "dev_wer": round(wer, 4), "loop_words": loops.hypothesis_loop_words,
                      "runaway": run.verdict.value, "loops": loop.verdict.value,
                      "passes": not run.blocking and not loop.blocking})
    print(f"base dev WER {base_wer:.4f}, loop words {base_loop.hypothesis_loop_words}")
    for t in sorted(table, key=lambda t: t.get("dev_wer", 9)):
        print(t)
    passing = [t for t in table if t.get("passes")]
    chosen = min(passing, key=lambda t: t["dev_wer"]) if passing else None
    print(f"passing {len(passing)} of {len(table)}; chosen: {chosen['candidate'] if chosen else 'NONE'}")
    if args.json:
        Path(args.json).write_text(json.dumps({"base_dev_wer": base_wer, "candidates": table,
                                               "chosen": chosen}, indent=2))
    return 0 if chosen else 5


if __name__ == "__main__":
    sys.exit(main())
