"""speech.runaway_hypotheses over two eval results (base vs fine-tuned) on the same manifest.

Both inputs are eval JSONs carrying every row (`all`: id, reference, hypothesis), as written by
speech_p4/eval_wer.py, speech_canary/eval_wer_nemo.py and eval_wer_salm.py. rows_expected comes
from the manifest itself, never from the eval artifact. Exit code follows the run contract:
0 PASS, 5 RED (the gate blocks).

Usage: runaway_verdict.py --manifest JSONL --base BASE.json --tuned TUNED.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from foundationscale.gates.speech_gates import RunawayHypothesisContext, RunawayHypothesisGate
from foundationscale.train.speech_metrics import count_runaway


def rows_by_id(path: str) -> dict[str, tuple[str, str]]:
    return {r["id"]: (r["reference"], r["hypothesis"]) for r in json.loads(Path(path).read_text())["all"]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--tuned", required=True)
    args = ap.parse_args()
    expected = sum(1 for line in Path(args.manifest).read_text().splitlines() if line.strip())
    base, tuned = rows_by_id(args.base), rows_by_id(args.tuned)
    # Only rows both models answered are comparable; a row one of them lost shows up as
    # UNDERCOVERED against the manifest's own count.
    shared = sorted(base.keys() & tuned.keys())
    b = count_runaway(base[i] for i in shared)
    t = count_runaway(tuned[i] for i in shared)
    r = RunawayHypothesisGate().run(
        RunawayHypothesisContext(
            rows_expected=expected,
            rows_checked=len(shared),
            base_runaway=b.rows_runaway,
            tuned_runaway=t.rows_runaway,
        )
    )
    print(f"[{r.verdict.value}] {r.gate_id}: {r.coverage.checked}/{r.coverage.expected} "
          f"{r.coverage.unit} -- {r.detail}")
    return 5 if r.blocking else 0


if __name__ == "__main__":
    sys.exit(main())
