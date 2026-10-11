"""speech.repetition_loops on base vs fine-tuned decodes of the same test manifest.

Usage: loop_verdict.py BASE_EVAL.json TUNED_EVAL.json ROWS_EXPECTED
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from foundationscale.gates.speech_gates import RepetitionLoopContext, RepetitionLoopGate
from foundationscale.train.speech_metrics import count_loop_words


def rows_by_id(path: str) -> dict[str, dict[str, str]]:
    return {r["id"]: r for r in json.loads(Path(path).read_text())["all"]}


def main() -> None:
    base, tuned = rows_by_id(sys.argv[1]), rows_by_id(sys.argv[2])
    shared = sorted(base.keys() & tuned.keys())
    b = count_loop_words((base[i]["reference"], base[i]["hypothesis"]) for i in shared)
    t = count_loop_words((tuned[i]["reference"], tuned[i]["hypothesis"]) for i in shared)
    gate = RepetitionLoopGate().run(
        RepetitionLoopContext(
            rows_expected=int(sys.argv[3]),
            rows_checked=len(shared),
            reference_words=b.reference_words,
            base_loop_words=b.hypothesis_loop_words,
            tuned_loop_words=t.hypothesis_loop_words,
        )
    )
    print(
        f"[{gate.verdict.value}] {gate.gate_id}: {gate.coverage.checked}/"
        f"{gate.coverage.expected} -- {gate.detail}"
    )


if __name__ == "__main__":
    main()
