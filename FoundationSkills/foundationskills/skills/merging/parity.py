"""Numeric parity of a merged checkpoint against its base + LoRA adapter (A4.7). Run as a script; writes JSON.

FS's merge script verifies structure only (parameter counts, no ``lora_`` tensors left), so a merge with the wrong LoRA
scale, a skipped target module or the wrong adapter passes it while changing the function the model computes. This
check runs identical token inputs through three forwards:

- ``adapter``: base + adapter through PEFT, unmerged (the function the merge must reproduce);
- ``base``: the same model with the adapter disabled (``disable_adapter()``), i.e. no merge at all;
- ``merged``: the merged checkpoint.

The verdict is self-calibrating: ``ratio = mean|merged - adapter| / mean|base - adapter|``, the merge error as a fraction
of the adapter's own effect on the logits. A missing or wrongly scaled merge sits near 1; a correct merge differs only by
the rounding of the merged weights. PASS iff ``ratio <= max_ratio`` and the next-token argmax of merged agrees with adapter
on at least ``min_agreement`` of positions. An adapter whose own effect is below ``min_effect`` cannot be told apart from
the base, so the check is UNMEASURED.

usage: python parity.py --base B --adapter A --merged M --out parity.json [--device cuda:0] [--dtype float32]
exit: 0 PASS, 5 RED (outside tolerance), 95 UNMEASURED (no measurable adapter effect), 96 refused (unloadable inputs)
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

PROMPTS = (
    "Summarise the main idea of the following paragraph in one sentence: The quarterly report shows revenue growth "
    "driven by cloud services, while hardware sales declined for the third consecutive quarter.",
    "Explain step by step how to compute the median of the list 7, 3, 9, 1, 5.",
    "Translate into French: The meeting has been moved to Thursday afternoon because the room is unavailable.",
    "Write a short Python function that returns the n-th Fibonacci number iteratively.",
    "A video shows a person picking up a cup, drinking, and placing it back on the table. Describe the sequence.",
)
EXIT = {"PASS": 0, "RED": 5, "UNMEASURED": 95, "REFUSED": 96}


def _load(path: str, dtype, device: str, trust: bool):
    import transformers  # noqa: PLC0415

    for cls in ("AutoModelForImageTextToText", "AutoModelForCausalLM"):
        try:
            model = getattr(transformers, cls).from_pretrained(path, torch_dtype=dtype, trust_remote_code=trust)
            return model.to(device).eval()
        except (ValueError, KeyError, AttributeError):
            continue
    raise ValueError(f"{path}: neither AutoModelForImageTextToText nor AutoModelForCausalLM can load it")


def _logits(model, batches) -> list:
    import torch  # noqa: PLC0415

    out = []
    with torch.no_grad():
        for ids in batches:
            out.append(model(input_ids=ids).logits.float().cpu())
    return out


def run(args: argparse.Namespace) -> dict:
    import torch  # noqa: PLC0415
    from peft import PeftModel  # noqa: PLC0415
    from transformers import AutoTokenizer  # noqa: PLC0415

    dtype = getattr(torch, args.dtype)
    tok = AutoTokenizer.from_pretrained(args.base, trust_remote_code=args.trust_remote_code)
    batches = [tok(p, return_tensors="pt").input_ids.to(args.device) for p in PROMPTS]
    model = PeftModel.from_pretrained(_load(args.base, dtype, args.device, args.trust_remote_code), args.adapter).eval()
    adapter = _logits(model, batches)
    with model.disable_adapter():
        base = _logits(model, batches)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    merged = _logits(_load(args.merged, dtype, args.device, args.trust_remote_code), batches)

    def mean_abs(a, b) -> float:
        total = sum(float((x - y).abs().sum()) for x, y in zip(a, b))
        return total / sum(x.numel() for x in a)

    effect, error = mean_abs(base, adapter), mean_abs(merged, adapter)
    agree = sum(int((m.argmax(-1) == p.argmax(-1)).sum()) for m, p in zip(merged, adapter))
    positions = sum(int(p.shape[1]) for p in adapter)
    base_agree = sum(int((b.argmax(-1) == p.argmax(-1)).sum()) for b, p in zip(base, adapter))
    ratio = error / effect if effect > 0 else None
    agreement = agree / positions
    if effect < args.min_effect:
        verdict, reason = "UNMEASURED", f"adapter effect {effect:.3g} < min_effect {args.min_effect}: merged cannot be told from base"
    elif ratio <= args.max_ratio and agreement >= args.min_agreement:
        verdict, reason = "PASS", "merged reproduces base + adapter within tolerance"
    else:
        verdict, reason = "RED", f"ratio {ratio:.4f} (max {args.max_ratio}) / argmax agreement {agreement:.4f} (min {args.min_agreement})"
    return {"verdict": verdict, "reason": reason, "ratio": ratio, "merge_error_mean_abs": error,
            "adapter_effect_mean_abs": effect, "argmax_agreement": agreement,
            "base_vs_adapter_argmax_agreement": base_agree / positions, "positions": positions,
            "prompts": len(PROMPTS), "dtype": args.dtype, "device": args.device,
            "max_ratio": args.max_ratio, "min_agreement": args.min_agreement, "min_effect": args.min_effect,
            "base": args.base, "adapter": args.adapter, "merged": args.merged}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--base", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--merged", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dtype", default="float32", help="forward dtype (float32 keeps the comparison above bf16 noise)")
    ap.add_argument("--max-ratio", dest="max_ratio", type=float, default=0.1)
    ap.add_argument("--min-agreement", dest="min_agreement", type=float, default=0.95)
    ap.add_argument("--min-effect", dest="min_effect", type=float, default=1e-3)
    ap.add_argument("--trust-remote-code", dest="trust_remote_code", action="store_true")
    args = ap.parse_args(argv)
    try:
        report = run(args)
    except (OSError, ValueError, ImportError) as exc:
        report = {"verdict": "REFUSED", "reason": f"{type(exc).__name__}: {exc}"}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: report.get(k) for k in ("verdict", "ratio", "argmax_agreement", "reason")}))
    return EXIT[report["verdict"]]


if __name__ == "__main__":
    sys.exit(main())
