"""FoundationScale verdicts on a NeMo-lane checkpoint it did not train (the Canary lane).

PHASE 1.2a move target for ``validation_campaigns/speech_canary/nemo_adjudicate.py``. Behaviour
is preserved verbatim -- same gate calls, same printed ADJ lines, same 5/0 exit codes.

Runs the same speech gates as ``train()`` does after a HF-native save:

* ``speech.audio_row_coverage`` over the converted manifest (``coverage.json`` from
  :mod:`foundationscale.upstream.nemo.finetune`);
* ``speech.tower_movement`` over the encoder, base vs fine-tuned, parameters only (digests of
  ``named_parameters``, dtype-tagged), plus the decoder as a second exercised module.

Exit code follows the run contract: 0 PASS, 5 RED (a gate blocks), 95 UNMEASURED (currently
folded into the gate's own ``blocking`` semantics -- see ``adjudicate_digests`` for the exact
aggregation).

The verdict ASSEMBLY is factored into :func:`adjudicate_digests` (pure: takes two digest dicts
plus the tower/frozen/coverage inputs and returns the printed lines + blocking flag). That is
what lets the PASS/RED/frozen-identity/VACUOUS shapes be tested in CI with plain dicts of
strings -- no NeMo model in sight.

Usage::

    python -m foundationscale.upstream.nemo.adjudicate --base nvidia/canary-1b-flash \\
        --finetuned DIR/finetuned.nemo --coverage DIR/coverage.json \\
        --out DIR/adjudication.json [--frozen transf_decoder]
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Collection, Mapping, Sequence
from pathlib import Path
from typing import Any

__all__ = ["TOWERS", "adjudicate_digests", "main", "parse_args"]


TOWERS: tuple[str, ...] = ("encoder", "transf_decoder")
"""The two exercised towers of the Canary AED. Order matters: it is the order verdict lines
appear in for the non-frozen case, matching the original's ``towers = ("encoder",
"transf_decoder")``."""


def adjudicate_digests(
    base: Mapping[str, str],
    tuned: Mapping[str, str],
    towers: Sequence[str],
    frozen: Collection[str],
    coverage: Mapping[str, Any],
) -> tuple[list[str], bool]:
    """Assemble verdict lines + blocking flag from two tower-digest dicts. Pure.

    ``base`` and ``tuned`` are ``name -> digest`` maps (e.g. from
    ``digests_from_named_tensors(..., TOWERS)``). ``towers`` is the tower prefix list in the
    order verdicts should be emitted for the non-frozen case; ``frozen`` is the set of prefixes
    the run declared FROZEN, for which movement is the DEFECT and the verdict logic inverts.
    ``coverage`` is the coverage dict from ``convert`` (four keys: ``rows_expected``,
    ``rows_checked``, ``rows_refused``, ``refused``).

    Returns ``(lines, blocking)`` where ``lines`` is the list of ``ADJ``-prefixable strings (a
    ``results``-derived line per gate plus one line per frozen tower) and ``blocking`` is
    ``True`` when any gate (or any frozen check) blocks. Same aggregation as the original:
    ``any(r.blocking for r in results) or any(b for _, b in extra)``.

    Line ORDER matches the original exactly: ALL coverage-then-movement lines (frozen towers are
    SKIPPED from ``results``) first, then the frozen-unchanged lines in tower order.
    """
    from foundationscale.gates.speech_gates import (
        AudioRowCoverageContext,
        AudioRowCoverageGate,
        TowerMovementContext,
        TowerMovementGate,
    )

    results = [
        AudioRowCoverageGate().run(
            AudioRowCoverageContext(
                rows_expected=int(coverage["rows_expected"]),
                rows_checked=int(coverage["rows_checked"]),
                rows_refused=int(coverage["rows_refused"]),
                refused=dict(coverage.get("refused", {})),
            )
        )
    ]
    extra: list[tuple[str, bool]] = []
    for prefix in towers:
        if prefix in frozen:
            # Declared frozen: movement would be the defect, so the verdict inverts. Every tensor
            # under the prefix must hash identically; none checked proves nothing and blocks.
            names = [n for n in base if n == prefix or n.startswith(prefix + ".")]
            changed = [n for n in names if tuned.get(n) != base[n]]
            ok = bool(names) and not changed
            extra.append(
                (
                    f"[{'PASS' if ok else 'RED'}] speech.frozen_unchanged/{prefix}: "
                    f"{len(names) - len(changed)}/{len(names)} tower parameters -- "
                    f"{len(changed)} changed {changed[:3]}",
                    not ok,
                )
            )
            continue
        results.append(
            TowerMovementGate().run(
                TowerMovementContext(
                    tower_prefix=prefix,
                    base_digests=dict(base),
                    saved_digests=dict(tuned),
                    exercised=True,
                )
            )
        )
    lines = [
        f"[{r.verdict.value}] {r.gate_id}: {r.coverage.checked}/{r.coverage.expected} "
        f"{r.coverage.unit} -- {r.detail}"
        for r in results
    ] + [line for line, _ in extra]
    blocking = any(r.blocking for r in results) or any(b for _, b in extra)
    return lines, blocking


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the CLI. Pure (argparse-only).

    Arg names identical to ``nemo_adjudicate.py``'s: ``--base``, ``--finetuned``, ``--coverage``,
    ``--out``, ``--frozen`` (append). ``--frozen`` names a prefix the run DECLARED frozen -- the
    gate family's ``--frozen`` flag semantics are "movement would be the defect".
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--finetuned", required=True)
    ap.add_argument("--coverage", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--frozen", action="append", default=[])
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the adjudication gates over a fine-tuned checkpoint. Return 0 PASS, 5 RED."""
    args = parse_args(argv)
    from nemo.collections.asr.models import ASRModel  # type: ignore[import-not-found]

    from foundationscale.train.speech_adjudication import digests_from_named_tensors

    cov = json.loads(Path(args.coverage).read_text())
    towers = TOWERS

    def digests(model: object) -> dict[str, str]:
        return digests_from_named_tensors(model.named_parameters(), list(towers))  # type: ignore[attr-defined]

    base = digests(ASRModel.from_pretrained(args.base, map_location="cpu"))
    tuned = digests(ASRModel.restore_from(args.finetuned, map_location="cpu"))
    lines, blocking = adjudicate_digests(base, tuned, towers, args.frozen, cov)
    for line in lines:
        print("ADJ", line[:200])
    Path(args.out).write_text(json.dumps({"lines": lines, "blocking": blocking}, indent=2))
    return 5 if blocking else 0


if __name__ == "__main__":
    raise SystemExit(main())
