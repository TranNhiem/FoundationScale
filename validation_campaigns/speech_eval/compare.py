"""One campaign report: is the tuned model better than the base -- and on whose numbers?

A 300-row eval manufactured "gains" that vanished at 2,000-2,700 rows: two models
scored on two different subsets of an eval set disagree about the subsets'
difficulty, not about the models, and no arithmetic on two independent corpus
rates can tell the difference. Only the PAIRED comparison can -- the same
utterances, one row per model, the gap bootstrapped over utterances -- and that
is what this prints, computed by ``train.speech_metrics.paired_bootstrap``.

Two provenance rules hold here, both learned from a comparison table that cited
the wrong checkpoints because run directories were picked by name:

  each number travels       the model (and adapter) printed beside each rate is
  with its checkpoint       the one the eval FILE recorded -- top-level
                            ``"model"``, or ``args.model`` with the optional
                            ``args.adapter`` -- and never anything read out of a
                            file or directory NAME. A rate without its checkpoint
                            is a rate nobody can re-derive; a rate beside the
                            wrong checkpoint is worse than that.

  coverage comes from       ``rows_expected`` is the manifest's own count of
  the manifest, never       non-empty JSONL lines, never the eval artifacts'
  from the artifact         row counts (an artifact must not certify its own
                            shortness). Rows are aligned BY ID over the
                            intersection of the two results, and the shortfall is
                            printed beside the rates -- a gap computed over 300 of
                            2,500 utterances is the failure that started this.

The ``speech.runaway_hypotheses`` verdict is computed exactly as
``runaway_verdict.py`` computes it (``count_runaway`` over the aligned rows,
``RunawayHypothesisGate().run(RunawayHypothesisContext(...))``) and is printed as
that script prints it, so the two reports cannot disagree about one corpus.

The ``speech.repetition_loops`` verdict follows it over the SAME aligned rows
(``count_loop_words``): the runaway count judges whole rows and a repetition loop
inside one 40 s chunk of an hour-long call adds only 4-12% of that call's words
(the measured Canary-1B Earnings collapse put 1,428 loop words beside 50,400
reference words while every row stayed under the runaway yardstick), so the loop
words are counted locally and judged against the base model's own loop count.

Exit code follows the run contract: 0 when the report is complete and neither
gate blocks, 5 when either does -- or when the comparison refused, because a
gap that could not be measured is not a green report.

Usage: compare.py --manifest evals.jsonl --base base.json --tuned tuned.json \\
                  [--metric wer] [--resamples 2000] [--seed 0] [--json out.json]
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

from foundationscale.gates.speech_gates import (
    RepetitionLoopContext,
    RepetitionLoopGate,
    RunawayHypothesisContext,
    RunawayHypothesisGate,
)
from foundationscale.train.speech_metrics import (
    TRANSCRIPT_NORMALIZER_ID,
    count_loop_words,
    count_runaway,
    paired_bootstrap,
)

__all__ = [
    "EvalResult",
    "load_eval",
    "manifest_rows_expected",
    "main",
]


@dataclass(frozen=True)
class EvalResult:
    """One eval artifact: its rows by id, and the model THAT FILE recorded.

    Frozen because this IS the provenance: a model path substituted in afterwards
    is how one comparison table published base numbers and tuned numbers under the
    wrong checkpoints. ``model`` is ``None`` when the artifact recorded none -- the
    report then prints ``<unrecorded>`` rather than inventing a path out of the
    file's name, because a guessed provenance is the defect this refuses to repeat.

    ``rows`` is keyed by the row's own ``id``: two artifacts over one manifest are
    comparable row for row and nowhere else.
    """

    path: str
    rows: dict[str, tuple[str, str]]
    model: str | None
    adapter: str | None

    def describe(self) -> str:
        """The model (+ adapter) as one printable line: these numbers' home address."""
        if self.model is None:
            return f"<unrecorded> (eval {self.path})"
        if self.adapter is None:
            return f"{self.model} (eval {self.path})"
        return f"{self.model} +adapter {self.adapter} (eval {self.path})"

    def as_manifest(self) -> dict[str, str | int | None]:
        """The JSON-ready provenance: the eval file, the model it recorded, its size."""
        return {
            "eval": self.path,
            "model": self.model,
            "adapter": self.adapter,
            "rows": len(self.rows),
        }


def load_eval(path: str) -> EvalResult:
    """``path`` as rows by id plus the provenance recorded INSIDE it.

    The model is read from the artifact's own ``model`` key or its ``args`` block
    (``args.model``, plus the optional ``args.adapter``) and from nowhere else --
    not from the filename, not from the run directory it sits in. A duplicate row
    id raises rather than dropping a row in silence: here a dropped row is a gap
    quietly widened on one side of the comparison.
    """
    payload = json.loads(Path(path).read_text())
    rows: dict[str, tuple[str, str]] = {}
    for row in payload["all"]:
        identifier = row["id"]
        if identifier in rows:
            raise ValueError(
                f"{path}: row id {identifier!r} appears twice -- a duplicate id would "
                "drop one utterance from one side of a paired comparison only"
            )
        rows[identifier] = (row["reference"], row["hypothesis"])
    args = payload.get("args") or {}
    model = payload.get("model") or args.get("model")
    adapter = args.get("adapter")
    return EvalResult(path=path, rows=rows, model=model, adapter=adapter)


def manifest_rows_expected(path: str) -> int:
    """``rows_expected``: the manifest's own count of non-empty JSONL lines.

    The denominator comes from the manifest and never from the artifacts: an eval
    file is not permitted to certify that it answered everything, which is exactly
    how a 300-row result passed for a full one.
    """
    return sum(1 for line in Path(path).read_text().splitlines() if line.strip())


def main() -> int:
    """Print the base-vs-tuned comparison with provenance, then the runaway and loop verdicts."""
    ap = argparse.ArgumentParser(description="Paired base-vs-tuned ASR comparison with provenance")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--tuned", required=True)
    ap.add_argument("--metric", default="wer", choices=("wer", "cer"))
    ap.add_argument("--resamples", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", default=None, metavar="OUT")
    args = ap.parse_args()

    expected = manifest_rows_expected(args.manifest)
    base = load_eval(args.base)
    tuned = load_eval(args.tuned)
    # Only rows both models answered are comparable; a row one of them lost shows up as
    # UNDERCOVERED against the manifest's own count.
    shared = sorted(base.rows.keys() & tuned.rows.keys())
    pairs_base = [base.rows[i] for i in shared]
    pairs_tuned = [tuned.rows[i] for i in shared]

    comparison = None
    refused = None
    try:
        comparison = paired_bootstrap(
            pairs_base,
            pairs_tuned,
            metric=args.metric,
            resamples=args.resamples,
            seed=args.seed,
        )
    except ValueError as error:
        refused = str(error)

    base_runaway = count_runaway(pairs_base)
    tuned_runaway = count_runaway(pairs_tuned)
    outcome = RunawayHypothesisGate().run(
        RunawayHypothesisContext(
            rows_expected=expected,
            rows_checked=len(shared),
            base_runaway=base_runaway.rows_runaway,
            tuned_runaway=tuned_runaway.rows_runaway,
        )
    )
    verdict_line = (
        f"[{outcome.verdict.value}] {outcome.gate_id}: {outcome.coverage.checked}/{outcome.coverage.expected} "
        f"{outcome.coverage.unit} -- {outcome.detail}"
    )

    # Loops are LOCAL: the runaway gate above judges whole rows and a repetition
    # loop inside one 40 s chunk of an hour-long call adds 4-12% of that call's
    # words, so the loop words are counted inside the same aligned rows and
    # judged against the base model's own count. The references are aligned
    # (paired_bootstrap refuses any two lists whose references differ), so the
    # base side states the corpus reference words once.
    base_loops = count_loop_words(pairs_base)
    tuned_loops = count_loop_words(pairs_tuned)
    loop_outcome = RepetitionLoopGate().run(
        RepetitionLoopContext(
            rows_expected=expected,
            rows_checked=len(shared),
            reference_words=base_loops.reference_words,
            base_loop_words=base_loops.hypothesis_loop_words,
            tuned_loop_words=tuned_loops.hypothesis_loop_words,
        )
    )
    loop_line = (
        f"[{loop_outcome.verdict.value}] {loop_outcome.gate_id}: {loop_outcome.coverage.checked}/{loop_outcome.coverage.expected} "
        f"{loop_outcome.coverage.unit} -- {loop_outcome.detail}"
    )

    shortfall = expected - len(shared)
    print(f"base : {base.describe()}")
    print(f"tuned: {tuned.describe()}")
    print(
        f"rows : {len(shared)} used / {expected} expected"
        + (f" -- {shortfall} short of the manifest (coverage)" if shortfall > 0 else "")
    )
    print(f"metric {args.metric}  normalizer {TRANSCRIPT_NORMALIZER_ID}")
    if comparison is None:
        print(f"paired comparison: REFUSED -- {refused}")
    else:
        if comparison.rows != len(shared):
            print(
                f"scored: {comparison.rows} rows -- {len(shared) - comparison.rows} refused "
                "(empty reference, as corpus_error_rate refuses them)"
            )
        print(f"base  : {comparison.metric} {comparison.base_rate:.4f}")
        print(f"tuned : {comparison.metric} {comparison.tuned_rate:.4f}")
        level = f"{int(round(comparison.confidence * 100))}%"
        word = "significant" if comparison.significant else "not significant"
        print(
            f"diff  : {comparison.diff:+.4f} with {level} CI "
            f"[{comparison.ci_low:+.4f}, {comparison.ci_high:+.4f}] -- {word}"
        )
        print(
            f"p_tuned_better: {comparison.p_tuned_better:.3f} over {comparison.resamples} "
            f"resamples (seed {comparison.seed})"
        )
    print(verdict_line)
    print(loop_line)

    if args.json is not None:
        report = {
            "manifest": args.manifest,
            "rows_expected": expected,
            "rows_used": len(shared),
            "base": base.as_manifest(),
            "tuned": tuned.as_manifest(),
            "comparison": None if comparison is None else comparison.as_manifest(),
            "comparison_refused": refused,
            "runaway": {
                "base": base_runaway.as_manifest(),
                "tuned": tuned_runaway.as_manifest(),
                "gate": {
                    "gate_id": outcome.gate_id,
                    "verdict": outcome.verdict.value,
                    "blocking": outcome.blocking,
                    "detail": outcome.detail,
                    "coverage_checked": outcome.coverage.checked,
                    "coverage_expected": outcome.coverage.expected,
                    "coverage_unit": outcome.coverage.unit,
                    "line": verdict_line,
                },
            },
            "repetition_loops": {
                "base": base_loops.as_manifest(),
                "tuned": tuned_loops.as_manifest(),
                "gate": {
                    "gate_id": loop_outcome.gate_id,
                    "verdict": loop_outcome.verdict.value,
                    "blocking": loop_outcome.blocking,
                    "detail": loop_outcome.detail,
                    "coverage_checked": loop_outcome.coverage.checked,
                    "coverage_expected": loop_outcome.coverage.expected,
                    "coverage_unit": loop_outcome.coverage.unit,
                    "line": loop_line,
                },
            },
        }
        Path(args.json).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    return 5 if (outcome.blocking or loop_outcome.blocking or refused is not None) else 0


if __name__ == "__main__":
    sys.exit(main())
