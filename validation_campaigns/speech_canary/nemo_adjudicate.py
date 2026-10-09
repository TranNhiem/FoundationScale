"""FoundationScale verdicts on a NeMo-lane checkpoint it did not train (the Canary lane).

Runs the same speech gates as train() does after a HF-native save:
  * speech.audio_row_coverage over the converted manifest (coverage.json from nemo_finetune.py);
  * speech.tower_movement over the encoder, base vs fine-tuned, parameters only (digests of
    named_parameters, dtype-tagged), plus the decoder as a second exercised module.
Exit code follows the run contract: 0 PASS, 5 RED (a gate blocks), 95 UNMEASURED.

Usage: nemo_adjudicate.py --base nvidia/canary-1b-flash --finetuned DIR/finetuned.nemo
                          --coverage DIR/coverage.json --out DIR/adjudication.json
                          [--frozen transf_decoder]  (declared frozen: must be bit-identical)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from foundationscale.gates.speech_gates import (
    AudioRowCoverageContext,
    AudioRowCoverageGate,
    TowerMovementContext,
    TowerMovementGate,
)
from foundationscale.train.speech_adjudication import digests_from_named_tensors


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--finetuned", required=True)
    ap.add_argument("--coverage", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--frozen", action="append", default=[])
    args = ap.parse_args()
    from nemo.collections.asr.models import ASRModel

    cov = json.loads(Path(args.coverage).read_text())
    results = [
        AudioRowCoverageGate().run(
            AudioRowCoverageContext(
                rows_expected=int(cov["rows_expected"]),
                rows_checked=int(cov["rows_checked"]),
                rows_refused=int(cov["rows_refused"]),
                refused=cov.get("refused", {}),
            )
        )
    ]
    towers = ("encoder", "transf_decoder")

    def digests(model: object) -> dict[str, str]:
        return digests_from_named_tensors(model.named_parameters(), list(towers))  # type: ignore[attr-defined]

    base = digests(ASRModel.from_pretrained(args.base, map_location="cpu"))
    tuned = digests(ASRModel.restore_from(args.finetuned, map_location="cpu"))
    extra: list[tuple[str, bool]] = []
    for prefix in towers:
        if prefix in args.frozen:
            # Declared frozen: movement would be the defect, so the verdict inverts. Every tensor
            # under the prefix must hash identically; none checked proves nothing and blocks.
            names = [n for n in base if n == prefix or n.startswith(prefix + ".")]
            changed = [n for n in names if tuned.get(n) != base[n]]
            ok = bool(names) and not changed
            extra.append((f"[{'PASS' if ok else 'RED'}] speech.frozen_unchanged/{prefix}: "
                          f"{len(names) - len(changed)}/{len(names)} tower parameters -- "
                          f"{len(changed)} changed {changed[:3]}", not ok))
            continue
        results.append(
            TowerMovementGate().run(
                TowerMovementContext(
                    tower_prefix=prefix, base_digests=base, saved_digests=tuned, exercised=True
                )
            )
        )
    lines = [
        f"[{r.verdict.value}] {r.gate_id}: {r.coverage.checked}/{r.coverage.expected} "
        f"{r.coverage.unit} -- {r.detail}"
        for r in results
    ] + [line for line, _ in extra]
    for line in lines:
        print("ADJ", line[:200])
    blocking = any(r.blocking for r in results) or any(b for _, b in extra)
    Path(args.out).write_text(json.dumps({"lines": lines, "blocking": blocking}, indent=2))
    return 5 if blocking else 0


if __name__ == "__main__":
    sys.exit(main())
