"""FoundationScale verdicts on a SALM checkpoint it did not train (the NeMo speech-lm lane).

PHASE 1.2b move target for ``validation_campaigns/speech_canary/salm_adjudicate.py``. Behaviour is
preserved verbatim -- same gate calls, same printed ADJ lines (byte for byte), same 5/0 exit codes.

Runs the speech gates plus the SALM freeze contract:
  * speech.audio_row_coverage over the converted manifest (coverage.json from salm_finetune);
  * speech.tower_movement (exercised=True) over perception.encoder and perception.proj,
    the trainable pieces of the perception stack;
  * speech.tower_movement over the PEFT tensors alone (its line is labelled "lora"): the only
    trainable LLM weights per freeze_params / prevent_freeze_params;
  * salm.frozen_llm_unchanged: every non-LoRA weight under llm. and embed_tokens. is bit-identical
    base vs fine-tuned -- a silent thaw would look like an ordinary fine-tune and no movement
    verdict would ever say so;
  * salm.param_names_unchanged: a name present in one model and not the other blocks; renaming
    would hide movement from every digest comparison above.
Exit code follows the run contract: 0 PASS, 5 RED (a gate blocks), 95 UNMEASURED.

The verdict ASSEMBLY is factored into pure transforms over plain ``(name, digest)`` and
``(name, tensor)`` input -- :func:`lora_filter` (the PEFT-tensor selector the freeze contract
lives on), :func:`snapshot_digests`, :func:`frozen_line`, :func:`names_line`, :func:`local_line`,
:func:`fmt` -- so every PASS/RED/frozen/vacuous shape here is testable in CI with plain dicts of
strings and no NeMo model in sight. Line formats are the campaign's, byte for byte.

Usage::

    python -m foundationscale.upstream.nemo.salm_adjudicate --base nvidia/canary-qwen-2.5b \\
        --finetuned DIR --coverage DIR/coverage.json --out DIR/adjudication.json
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

__all__ = [
    "digests",
    "fmt",
    "frozen_line",
    "local_line",
    "lora_filter",
    "main",
    "names_line",
    "parse_args",
    "snapshot",
    "DEFAULT_SCOPES",
    "SalmScopes",
    "scopes_for_model",
    "snapshot_digests",
]


def lora_filter(items: Sequence[tuple[str, Any]], lora: str = "any") -> list[tuple[str, Any]]:
    """Keep the PEFT tensors ("only"), drop them ("none"), or accept both ("any"). Pure.

    Marker is ``.lora_`` in the parameter name (PEFT's ``lora_A`` / ``lora_B`` modules). The
    verdicts disagree about LoRA in exactly one way and it matters: ``salm.frozen_llm_unchanged``
    must NEVER see a lora tensor (it is supposed to move), while the ``lora`` movement line keeps
    only those. A filter that leaked one either way would turn a silent thaw into a PASS (or the
    trained LoRA into a spurious RED).
    """
    if lora != "any":
        return [it for it in items if (".lora_" in it[0]) == (lora == "only")]
    return list(items)


def digests(
    items: Sequence[tuple[str, Any]], prefixes: Sequence[str], lora: str = "any"
) -> dict[str, str]:
    """Digest the named parameters under prefixes; ``lora`` keeps PEFT tensors ("only"), drops
    them ("none"), or accepts both ("any"). The freeze verdict must never see a lora tensor: it
    is supposed to move.

    Pure over the ``(name, tensor)`` pairs handed in (``digests_from_named_tensors`` is the only
    hashing step), so the digest maps every verdict compares are buildable in plain CI.
    """
    from foundationscale.train.speech_adjudication import digests_from_named_tensors

    return digests_from_named_tensors(lora_filter(items, lora), list(prefixes))


@dataclass(frozen=True)
class SalmScopes:
    """Which parameters must move and which must stay bit-identical (Phase 2: from the registry).

    ``trainable`` prefixes are checked for movement; ``frozen`` prefixes must be bit-identical
    except PEFT tensors (``.lora_``), which must move. LoRA lives under the first frozen prefix.
    """

    trainable: tuple[str, ...]
    frozen: tuple[str, ...]


# Canary-Qwen's released scope; equal to the registry entry "canary-qwen-2.5b".
DEFAULT_SCOPES = SalmScopes(
    trainable=("perception.encoder", "perception.proj"), frozen=("llm", "embed_tokens")
)


def scopes_for_model(model_id: str) -> SalmScopes:
    """The scopes the model registry declares for ``model_id`` (a NeMo speech_llm entry)."""
    from foundationscale.upstream.models import Backend, ModelKind, get_model

    entry = get_model(model_id)
    if entry.backend is not Backend.NEMO or entry.kind is not ModelKind.SPEECH_LLM:
        raise ValueError(f"{model_id!r} is not a NeMo speech_llm registry entry")
    if not entry.trainable_prefixes or not entry.frozen_prefixes:
        raise ValueError(f"{model_id!r} declares no trainable/frozen scopes to adjudicate")
    return SalmScopes(trainable=entry.trainable_prefixes, frozen=entry.frozen_prefixes)


def snapshot_digests(
    items: Sequence[tuple[str, Any]], scopes: SalmScopes = DEFAULT_SCOPES
) -> dict[str, Any]:
    """Everything the verdicts below need of one model, from its ``(name, tensor)`` pairs. Pure.

    Keys: ``names``; ``trainable`` (prefix -> digests, in declared order); ``lora`` (PEFT tensors
    under the frozen scope) and ``frozen`` (non-LoRA tensors under the frozen scope).
    """
    return {
        "names": {n for n, _ in items},
        "trainable": {p: digests(items, [p]) for p in scopes.trainable},
        "lora": digests(items, list(scopes.frozen), lora="only"),
        "frozen": digests(items, list(scopes.frozen), lora="none"),
    }


def snapshot(model: object, scopes: SalmScopes = DEFAULT_SCOPES) -> dict[str, Any]:
    """Everything the verdicts below need of one model, so the next model may be loaded (each is
    ~2.5B params; only the digests fit in memory side by side)."""
    return snapshot_digests(list(model.named_parameters()), scopes)  # type: ignore[attr-defined]


def fmt(result: object, label: str | None = None) -> str:
    # label overrides gate_id: tower_movement runs three times here and the lines must tell apart.
    return (
        f"[{result.verdict.value}] {label or result.gate_id}: "  # type: ignore[attr-defined]
        f"{result.coverage.checked}/{result.coverage.expected} "  # type: ignore[attr-defined]
        f"{result.coverage.unit} -- {result.detail}"  # type: ignore[attr-defined]
    )


def local_line(ok: bool, name: str, checked: int, expected: int, detail: str) -> tuple[str, bool]:
    return f"[{'PASS' if ok else 'RED'}] {name}: {checked}/{expected} params -- {detail}", not ok


def frozen_line(base: dict[str, str], tuned: dict[str, str]) -> tuple[str, bool]:
    """Bit-identity of every non-LoRA LLM / embedding weight (freeze_params: they never train)."""
    shared = sorted(base.keys() & tuned.keys())
    changed = [n for n in shared if base[n] != tuned[n]]
    renamed = sorted(base.keys() ^ tuned.keys())
    checked, expected = len(shared), len(base.keys() | tuned.keys())
    ok = checked > 0 and not changed and not renamed
    detail = f"{len(changed)} changed {len(renamed)} new/missing"
    if changed:
        detail += f"; changed up to 5 {changed[:5]}"
    if renamed:
        detail += f"; new/missing up to 5 {renamed[:5]}"
    return local_line(ok, "salm.frozen_llm_unchanged", checked, expected, detail)


def names_line(base_names: set[str], tuned_names: set[str]) -> tuple[str, bool]:
    """Parameter names must match exactly: a rename hides movement from every digest above."""
    checked, expected = len(base_names & tuned_names), len(base_names | tuned_names)
    renamed = sorted(base_names ^ tuned_names)
    ok = checked > 0 and not renamed
    detail = "identical keys" if ok else f"new/missing up to 5 {renamed[:5]}"
    return local_line(ok, "salm.param_names_unchanged", checked, expected, detail)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the CLI. Pure (argparse-only).

    Arg names identical to ``salm_adjudicate.py``'s: ``--base``, ``--finetuned``, ``--coverage``,
    ``--out``.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--finetuned", required=True)
    ap.add_argument("--coverage", required=True)
    ap.add_argument("--out", required=True)
    # Phase 2: scopes from the model registry; without it, Canary-Qwen's released scope.
    ap.add_argument("--model-id", default=None)
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the adjudication gates over a fine-tuned SALM checkpoint. Return 0 PASS, 5 RED."""
    args = parse_args(argv)
    from nemo.collections.speechlm2 import SALM  # type: ignore[import-not-found]

    from foundationscale.gates.speech_gates import (
        AudioRowCoverageContext,
        AudioRowCoverageGate,
        TowerMovementContext,
        TowerMovementGate,
    )

    cov = json.loads(Path(args.coverage).read_text())
    coverage = AudioRowCoverageGate().run(
        AudioRowCoverageContext(
            rows_expected=int(cov["rows_expected"]),
            rows_checked=int(cov["rows_checked"]),
            rows_refused=int(cov["rows_refused"]),
            refused=cov.get("refused", {}),
        )
    )
    results: list[tuple[str, bool]] = [(fmt(coverage), coverage.blocking)]

    # SALM.from_pretrained is HFHubMixin's and takes no map_location; HF already loads to CPU and
    # .to("cpu") pins it so the digests hash saved values, not a GPU copy.
    scopes = scopes_for_model(args.model_id) if args.model_id else DEFAULT_SCOPES
    base_snap = snapshot(SALM.from_pretrained(args.base).to("cpu"), scopes)
    tuned_model = SALM.from_pretrained(args.finetuned).to("cpu")
    tuned_snap = snapshot(tuned_model, scopes)
    del tuned_model  # the temporary is released at statement end; this one is not.

    checks = [
        (base_snap["trainable"][p], tuned_snap["trainable"][p], p, f"tower_movement/{p}")
        for p in scopes.trainable
    ]
    # tower_prefix is the PEFT namespace (LoRA lives under the first frozen prefix).
    checks.append((base_snap["lora"], tuned_snap["lora"], scopes.frozen[0], "lora"))
    for base_d, tuned_d, prefix, label in checks:
        r = TowerMovementGate().run(
            TowerMovementContext(
                tower_prefix=prefix,
                base_digests=base_d,
                saved_digests=tuned_d,
                exercised=True,
            )
        )
        results.append((fmt(r, label), r.blocking))

    results.append(frozen_line(base_snap["frozen"], tuned_snap["frozen"]))
    results.append(names_line(base_snap["names"], tuned_snap["names"]))

    lines = [line for line, _ in results]
    for line in lines:
        print("ADJ", line[:200])
    blocking = any(b for _, b in results)
    Path(args.out).write_text(json.dumps({"lines": lines, "blocking": blocking}, indent=2))
    return 5 if blocking else 0


if __name__ == "__main__":
    raise SystemExit(main())
