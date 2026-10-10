#!/usr/bin/env python3
"""
merge_lora.py -- merge a PEFT/LoRA adapter into its base vision-language model.

Standalone tutorial script (FoundationScale examples/vlm-2xH200/scripts/) for a
VM with 2x H200.  It is checkpoint-agnostic and works for the Gemma-4 family
(model_type "gemma4", "gemma4_unified") and the Qwen3.6 family (model_type
"qwen3_5", "qwen3_5_moe") checkpoints used in the tutorial.

What the script does
--------------------
1. Loads the base VLM with `AutoModelForImageTextToText` (torch_dtype from
   --dtype, `device_map` from --device-map -- "cpu" by default) and records the
   base parameter count *before* any LoRA weights exist.
2. Wraps it with `PeftModel.from_pretrained(base, adapter)` and counts the LoRA
   modules that are about to be merged (every nn.Module exposing `lora_A`).
3. Calls `merge_and_unload()` which folds every LoRA update into the base
   weights and removes the adapter modules.
4. Saves the merged model (`save_pretrained(..., safe_serialization=True)` -> safetensors)
   and copies the processor (tokenizer + image processor + processor config) so
   that `--out` is a self-contained checkpoint.
5. Verifies the saved artifact by reloading it from disk: the merged parameter
   count must equal the base parameter count and no `lora_` weight may remain.

The run summary is printed and written as JSON (--results, default
<out>/merge_lora_results.json).  The exit code is 0 only if every check passed.

Example
-------
python3 merge_lora.py \
    --base    /models/gemma4-unified-27b \
    --adapter /checkpoints/vqa-lora \
    --out     /models/gemma4-unified-27b-vqa-merged \
    --dtype   bfloat16
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import torch
import transformers
from peft import PeftModel
from transformers import AutoConfig, AutoModelForImageTextToText, AutoProcessor

# Supported dtypes for loading / storing the merged checkpoint.
DTYPE_BY_NAME = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}

# Checkpoint families covered by the tutorial; anything else only triggers a warning.
KNOWN_MODEL_TYPES = ("gemma4", "gemma4_unified", "qwen3_5", "qwen3_5_moe")


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def param_stats(named_parameters) -> dict:
    """Count parameter tensors and scalar parameters of a (sub)model."""
    tensors = 0
    numel = 0
    for _, tensor in named_parameters:
        tensors += 1
        numel += tensor.numel()
    return {"param_tensors": tensors, "param_numel": numel}


def lora_stats(model) -> dict:
    """Describe the LoRA weights present in `model` (called *before* merging).

    Counts every nn.Module that exposes `lora_A`, i.e. the modules whose LoRA
    updates `merge_and_unload()` will fold into the base weights.
    """
    lora_modules = 0  # modules with `lora_A`
    adapter_targets = 0  # concreteLinear/tensor targets (len of lora_A dicts)
    for _, module in model.named_modules():
        lora_a = getattr(module, "lora_A", None)
        if lora_a is not None and len(lora_a) > 0:
            lora_modules += 1
            adapter_targets += len(lora_a)
    lora_numel = sum(p.numel() for name, p in model.named_parameters() if "lora_" in name)
    return {
        "lora_modules_merged": lora_modules,  # <- "number of LoRA modules merged"
        "lora_adapter_targets_merged": adapter_targets,
        "lora_param_numel_merged": lora_numel,
    }


def lora_param_names(model) -> list[str]:
    """Names of any leftover LoRA weights (should be empty after merging)."""
    return sorted(name for name, _ in model.named_parameters() if "lora_" in name)


def human_bytes(count: float) -> str:
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(count) < 1024.0 or unit == "TiB":
            return f"{count:.2f} {unit}"
        count /= 1024.0
    return f"{count:.2f} TiB"


def directory_size(path: Path) -> int:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=False)
        handle.write("\n")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="merge_lora.py",
        description=(
            "Merge a PEFT/LoRA adapter into its base VLM (Gemma-4 or Qwen3.6) and save "
            "a standalone checkpoint, together with the processor."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--base",
        type=Path,
        required=True,
        metavar="MODEL_DIR",
        help="Directory of the base VLM checkpoint the adapter was trained on "
        "(model_type gemma4/gemma4_unified or qwen3_5/qwen3_5_moe).",
    )
    parser.add_argument(
        "--adapter",
        type=Path,
        required=True,
        metavar="ADAPTER_DIR",
        help="Directory with the LoRA adapter (adapter_config.json + adapter_model.safetensors).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        metavar="MERGED_DIR",
        help="Output directory for the merged checkpoint (created if missing).",
    )
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=sorted(DTYPE_BY_NAME),
        help="torch dtype used to load the base model and to store the merged weights.",
    )
    parser.add_argument(
        "--device-map",
        default="cpu",
        metavar="MAP",
        help="device_map passed to from_pretrained: 'cpu' (default), 'cuda:0', or an "
        "accelerate map such as 'auto'/'balanced'/'sequential'.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Pass trust_remote_code=True to transformers (only for custom model code).",
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=None,
        metavar="RESULTS_JSON",
        help="Where to write the JSON results (default: <out>/merge_lora_results.json).",
    )
    return parser.parse_args(argv)


# --------------------------------------------------------------------------- #
# work
# --------------------------------------------------------------------------- #
def merge_and_verify(args: argparse.Namespace, results: dict) -> dict:
    dtype = DTYPE_BY_NAME[args.dtype]
    torch.set_grad_enabled(False)  # merging and saving are pure inference work

    # ---- 0. checkpoint metadata -------------------------------------------------
    config = AutoConfig.from_pretrained(str(args.base), trust_remote_code=args.trust_remote_code)
    model_type = getattr(config, "model_type", "unknown")
    if model_type not in KNOWN_MODEL_TYPES:
        print(
            f"[warn] unexpected model_type {model_type!r}; the tutorial covers {KNOWN_MODEL_TYPES}"
        )
    results["base"] = {
        "path": str(args.base.resolve()),
        "model_type": model_type,
        "architectures": list(getattr(config, "architectures", []) or []),
        "torch_dtype": str(dtype),
        "device_map": args.device_map,
    }
    results["adapter"] = {"path": str(args.adapter.resolve())}
    results["merged"] = {"path": str(args.out.resolve()), "torch_dtype": str(dtype)}

    # ---- 1. base model (count parameters *before* any LoRA weight exists) -------
    print(
        f"[1/5] loading base model from {args.base} "
        f"(device_map={args.device_map}, torch_dtype={args.dtype})"
    )
    base_model = AutoModelForImageTextToText.from_pretrained(
        str(args.base),
        torch_dtype=dtype,  # bfloat16 for the VLM weights
        device_map=args.device_map,
        trust_remote_code=args.trust_remote_code,
        low_cpu_mem_usage=True,
    )
    base = param_stats(base_model.named_parameters())
    results["base"].update(base)
    print(
        f"      base parameters: {base['param_numel']:,} scalars in "
        f"{base['param_tensors']:,} tensors"
    )

    # ---- 2. LoRA adapter, counted before merging --------------------------------
    print(f"[2/5] loading LoRA adapter from {args.adapter}")
    peft_model = PeftModel.from_pretrained(base_model, str(args.adapter))
    lora = lora_stats(peft_model)
    results["adapter"].update(lora)
    print(
        f"      LoRA modules merged (modules with lora_A, counted before merging): "
        f"{lora['lora_modules_merged']:,}"
    )
    print(
        f"      LoRA adapter targets merged: {lora['lora_adapter_targets_merged']:,}   "
        f"LoRA scalars folded: {lora['lora_param_numel_merged']:,} "
        f"({human_bytes(lora['lora_param_numel_merged'] * dtype.itemsize)})"
    )

    # ---- 3. fold LoRA updates into the base weights and drop the adapter --------
    print("[3/5] merge_and_unload() -- folding LoRA updates into the base weights")
    merged_model = peft_model.merge_and_unload()
    merged = param_stats(merged_model.named_parameters())
    leftover_in_memory = lora_param_names(merged_model)
    results["merged"].update(merged)
    print(
        f"      merged parameters: {merged['param_numel']:,} scalars in "
        f"{merged['param_tensors']:,} tensors"
    )

    # ---- 4. save merged model + processor ---------------------------------------
    print(f"[4/5] saving merged checkpoint to {args.out} (safe_serialization=True)")
    args.out.mkdir(parents=True, exist_ok=True)
    merged_model.save_pretrained(str(args.out), safe_serialization=True)
    processor = AutoProcessor.from_pretrained(
        str(args.base), trust_remote_code=args.trust_remote_code
    )
    processor.save_pretrained(str(args.out))  # tokenizer + image processor + processor config
    results["merged"]["processor_class"] = type(processor).__name__
    results["merged"]["size_bytes"] = directory_size(args.out)
    print(f"      merged checkpoint size: {human_bytes(results['merged']['size_bytes'])}")

    # ---- 5. verify the saved artifact by reloading it from disk -----------------
    print("[5/5] verifying the saved checkpoint by reloading it from disk (device_map=cpu)")
    del merged_model, peft_model, base_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    reloaded = AutoModelForImageTextToText.from_pretrained(
        str(args.out),
        torch_dtype=dtype,
        device_map="cpu",
        trust_remote_code=args.trust_remote_code,
        low_cpu_mem_usage=True,
    )
    reloaded_stats = param_stats(reloaded.named_parameters())
    leftover_on_disk = lora_param_names(reloaded)
    processor_ok = (
        AutoProcessor.from_pretrained(str(args.out), trust_remote_code=args.trust_remote_code)
        is not None
    )
    del reloaded
    gc.collect()

    checks = {
        "merged_param_numel_equals_base_param_numel": merged["param_numel"] == base["param_numel"],
        "merged_param_tensors_equals_base_param_tensors": merged["param_tensors"]
        == base["param_tensors"],
        "reloaded_param_numel_equals_base_param_numel": reloaded_stats["param_numel"]
        == base["param_numel"],
        "reloaded_param_tensors_equals_base_param_tensors": reloaded_stats["param_tensors"]
        == base["param_tensors"],
        "no_lora_weights_left_in_merged_model": not leftover_in_memory,
        "no_lora_weights_left_on_disk": not leftover_on_disk,
        "processor_saved_and_reloadable": processor_ok,
    }
    results["verification"] = {
        "checks": checks,
        "reloaded": reloaded_stats,
        "leftover_lora_params_in_memory": leftover_in_memory,
        "leftover_lora_params_on_disk": leftover_on_disk,
        "passed": all(checks.values()),
    }
    return results


def print_summary(results: dict, args: argparse.Namespace) -> None:
    base = results["base"]
    adapter = results["adapter"]
    merged = results["merged"]
    verify = results["verification"]
    line = "=" * 74
    print(line)
    print("merge_lora.py -- summary")
    print(line)
    print(f"base model dir       : {base['path']}")
    print(
        f"base model_type      : {base['model_type']}   torch_dtype={base['torch_dtype']}   "
        f"device_map={base['device_map']}"
    )
    print(
        f"base parameters      : {base['param_numel']:,} scalars in "
        f"{base['param_tensors']:,} tensors"
    )
    print(f"LoRA adapter dir     : {adapter['path']}")
    print(
        f"LoRA modules merged  : {adapter['lora_modules_merged']:,} "
        f"(modules with lora_A, counted before merging)"
    )
    print(f"LoRA targets merged  : {adapter['lora_adapter_targets_merged']:,}")
    print(f"LoRA scalars folded  : {adapter['lora_param_numel_merged']:,}")
    print(f"merged output dir    : {merged['path']}")
    print(
        f"merged parameters    : {merged['param_numel']:,} scalars in "
        f"{merged['param_tensors']:,} tensors"
    )
    print(
        f"merged checkpoint    : {human_bytes(merged['size_bytes'])} "
        f"(safetensors + processor of class {merged['processor_class']})"
    )
    print("-" * 74)
    print("verification")
    for name, ok in verify["checks"].items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    print(f"  [{'PASS' if verify['passed'] else 'FAIL'}] all checks passed")
    print(line)
    print(f"JSON results written to: {args.results}")
    print(line)


# --------------------------------------------------------------------------- #
# entry point
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.base = args.base.expanduser()
    args.adapter = args.adapter.expanduser()
    args.out = args.out.expanduser()
    args.results = (args.results or (args.out / "merge_lora_results.json")).expanduser()

    for label, path in (("base", args.base), ("adapter", args.adapter)):
        if not path.is_dir():
            print(f"[error] --{label} directory does not exist: {path}", file=sys.stderr)
            return 2

    started = time.time()
    results = {
        "script": "merge_lora.py",
        "command": " ".join(sys.argv[:1] + (argv if argv is not None else sys.argv[1:])),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "gpu_count": torch.cuda.device_count(),
        "args": {
            "base": str(args.base),
            "adapter": str(args.adapter),
            "out": str(args.out),
            "dtype": args.dtype,
            "device_map": args.device_map,
            "trust_remote_code": args.trust_remote_code,
            "results": str(args.results),
        },
    }

    try:
        merge_and_verify(args, results)
    except Exception as exc:  # keep the partial record for the student to inspect
        results["error"] = f"{type(exc).__name__}: {exc}"
        results["ok"] = False
        results["elapsed_seconds"] = round(time.time() - started, 3)
        write_json(args.results, results)
        traceback.print_exc()
        print(f"[error] merge failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    results["ok"] = results["verification"]["passed"]
    results["elapsed_seconds"] = round(time.time() - started, 3)
    write_json(args.results, results)
    print_summary(results, args)
    return 0 if results["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
