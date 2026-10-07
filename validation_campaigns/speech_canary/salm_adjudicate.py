"""FoundationScale verdicts on a SALM checkpoint it did not train (the NeMo speech-lm lane).

Runs the speech gates plus the SALM freeze contract:
  * speech.audio_row_coverage over the converted manifest (coverage.json from salm_finetune.py);
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

Usage: salm_adjudicate.py --base nvidia/canary-qwen-2.5b --finetuned DIR
                          --coverage DIR/coverage.json --out DIR/adjudication.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from foundationscale.gates.speech_gates import (
    AudioRowCoverageContext,
    AudioRowCoverageGate,
    TowerMovementContext,
    TowerMovementGate,
)
from foundationscale.train.speech_adjudication import digests_from_named_tensors


def digests(model: object, prefixes: list[str], lora: str = "any") -> dict[str, str]:
    """Digest the named parameters under prefixes; ``lora`` keeps PEFT tensors ("only"), drops
    them ("none"), or accepts both ("any"). The freeze verdict must never see a lora tensor: it
    is supposed to move."""
    items = list(model.named_parameters())  # type: ignore[attr-defined]
    if lora != "any":
        items = [it for it in items if (".lora_" in it[0]) == (lora == "only")]
    return digests_from_named_tensors(items, prefixes)


def snapshot(model: object) -> dict[str, Any]:
    """Everything the verdicts below need of one model, so the next model may be loaded (each is
    ~2.5B params; only the digests fit in memory side by side)."""
    return {
        "names": {n for n, _ in model.named_parameters()},  # type: ignore[attr-defined]
        "encoder": digests(model, ["perception.encoder"]),
        "proj": digests(model, ["perception.proj"]),
        "lora": digests(model, ["llm"], lora="only"),
        "frozen": digests(model, ["llm", "embed_tokens"], lora="none"),
    }


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
    """Bit-identity of every non-LoRA LLM / embedding weight (freeze_params says these never train)."""
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--finetuned", required=True)
    ap.add_argument("--coverage", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    from nemo.collections.speechlm2 import SALM

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
    base_snap = snapshot(SALM.from_pretrained(args.base).to("cpu"))
    tuned_model = SALM.from_pretrained(args.finetuned).to("cpu")
    tuned_snap = snapshot(tuned_model)
    del tuned_model  # the temporary is released at statement end; this one is not.

    checks = (
        ("encoder", "perception.encoder", "tower_movement/perception.encoder"),
        ("proj", "perception.proj", "tower_movement/perception.proj"),
        ("lora", "llm", "lora"),  # tower_prefix is the PEFT namespace; the line says what it is
    )
    for key, prefix, label in checks:
        r = TowerMovementGate().run(
            TowerMovementContext(
                tower_prefix=prefix,
                base_digests=base_snap[key],
                saved_digests=tuned_snap[key],
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
    sys.exit(main())
