#!/usr/bin/env python3
"""eval_wer.py — Held-out ASR word error rate for a Gemma-4 audio model.

Honesty rule: every number is measured, nothing assumed.
"""

import argparse
import json
import os
import platform
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import transformers

from foundationscale.train.speech_metrics import (
    corpus_error_rate,
    word_errors,
)


def step_env(args):  # noqa: ARG001 - kept for a uniform step signature
    e = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>"),
    }
    try:
        import peft

        e["peft"] = peft.__version__
    except Exception as ex:
        e["peft"] = f"UNAVAILABLE: {ex!r}"
    if torch.cuda.is_available():
        e["gpu_name"] = torch.cuda.get_device_name(0)
    else:
        e["gpu_name"] = "N/A (no CUDA)"
    return e


def step_load(args, device):
    from transformers import AutoProcessor, Gemma4ForConditionalGeneration

    proc_path = args.processor or args.model
    processor = AutoProcessor.from_pretrained(proc_path)
    try:
        model = Gemma4ForConditionalGeneration.from_pretrained(args.model, dtype=torch.bfloat16)
    except TypeError:
        model = Gemma4ForConditionalGeneration.from_pretrained(
            args.model, torch_dtype=torch.bfloat16
        )
    model = model.to(device)
    model.eval()

    adapter_used = None
    if args.adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, args.adapter)
        adapter_used = args.adapter

    return model, processor, adapter_used


def step_audio(path, sampling_rate):
    wave2d, sr = sf.read(path, dtype="float32", always_2d=True)
    if sr != sampling_rate:
        raise RuntimeError(
            f"sample rate mismatch: file {path!r} has sr={sr}, "
            f"processor.feature_extractor.sampling_rate={sampling_rate}; resampling NOT allowed"
        )
    mono = (
        wave2d.mean(axis=1).astype(np.float32)
        if wave2d.shape[1] > 1
        else wave2d[:, 0].astype(np.float32)
    )
    return mono, int(sr)


def step_generate(model, processor, rows_batch, args, device):
    messages = [
        [
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio": r["_wave"]},
                    {"type": "text", "text": args.prompt},
                ],
            }
        ]
        for r in rows_batch
    ]
    processor.tokenizer.padding_side = "left"
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        add_generation_prompt=True,
        processor_kwargs={"padding": True},
    )
    inputs = {k: v.to(device) for k, v in inputs.items() if torch.is_tensor(v)}
    for k, v in inputs.items():
        if v.is_floating_point():
            inputs[k] = v.to(torch.bfloat16)
    out = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
    prompt_len = inputs["input_ids"].shape[1]
    new_tokens = out[:, prompt_len:]
    hyps = processor.tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
    # Truncated = the row stopped only because it ran out of budget: no stop id in
    # its new tokens. The stop set is the generation config's (Gemma-4 ends a turn
    # with <end_of_turn>, not the tokenizer's eos), so read it there.
    stop = model.generation_config.eos_token_id
    stop_ids = set(stop if isinstance(stop, (list, tuple)) else [stop]) - {None}
    stop_ids.add(processor.tokenizer.eos_token_id)
    truncated = 0
    for i in range(new_tokens.shape[0]):
        if not stop_ids.intersection(new_tokens[i].tolist()):
            truncated += 1
    return [h.strip() for h in hyps], truncated


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--processor", default=None)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--prompt", default="Transcribe this audio exactly.")
    args = ap.parse_args()

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    results = {
        "env": step_env(args),
        "args": vars(args),
        "n_rows": 0,
        "rows_scored": 0,
        "truncated_count": 0,
        "samples": [],
    }
    pairs = []
    out_path = Path(args.out)
    t0 = time.time()
    try:
        model, processor, adapter_used = step_load(args, device)
        results["adapter_used"] = adapter_used
        sampling_rate = int(processor.feature_extractor.sampling_rate)

        rows = []
        with Path(args.manifest).open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
        if args.limit is not None:
            rows = rows[: args.limit]
        results["n_rows"] = len(rows)

        truncated_total = 0
        for start in range(0, len(rows), args.batch_size):
            batch = rows[start : start + args.batch_size]
            for r in batch:
                wave, _ = step_audio(r["audio"], sampling_rate)
                r["_wave"] = wave
            with torch.inference_mode():
                hyps, trun = step_generate(model, processor, batch, args, device)
            truncated_total += trun
            for i, r in enumerate(batch):
                ref = r["answer"]
                hyp = hyps[i]
                pairs.append((ref, hyp))
                results["rows_scored"] = len(pairs)
                if len(results["samples"]) < 25:
                    sample = {"id": r["id"], "reference": ref, "hypothesis": hyp}
                    if ref.strip():
                        sample["wer"] = word_errors(ref, hyp).rate()
                    results["samples"].append(sample)
        results["truncated_count"] = truncated_total

        results["wer"] = corpus_error_rate(pairs, metric="wer", expected=len(rows)).as_manifest()
        results["cer"] = corpus_error_rate(pairs, metric="cer", expected=len(rows)).as_manifest()
    except SystemExit:
        raise
    except Exception:
        results["fatal"] = traceback.format_exc()
        raise
    finally:
        results["elapsed_s"] = time.time() - t0
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")

    print(f"n={results['n_rows']}")
    print(f"WER={results['wer'].get('error_rate')}")
    print(f"CER={results['cer'].get('error_rate')}")
    print(f"truncated={results['truncated_count']}")
    print(f"seconds={results['elapsed_s']:.1f}")


if __name__ == "__main__":
    try:
        main()
    except SystemExit as se:
        sys.exit(se.code if se.code is not None else 0)
    except Exception:
        traceback.print_exc()
        sys.exit(1)
