#!/usr/bin/env python3
"""eval_wer.py — Held-out ASR word error rate for a FoundationScale P3 speech model.

Evaluates all three speech kinds declared by foundationscale.train.speech_kinds
(audio_llm, seq2seq, ctc), dispatching on the checkpoint's own config.

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

from foundationscale.train.speech_kinds import (
    load_speech_model,
    speech_model_kind,
)
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
    """Load processor + model per the checkpoint's declared speech kind.

    Dispatch:
      - ``speech_model_kind(AutoConfig.from_pretrained(model))`` decides the kind;
        an unknown kind refuses at exit 96 (same mechanism as speech_kinds).
      - ``load_speech_model(kind, model, dtype=torch.bfloat16)`` builds the model.
      - If ``--adapter`` is given, wrap with ``PeftModel.from_pretrained``.

    Returns (model, processor, adapter_used, kind).
    """
    from transformers import AutoConfig, AutoProcessor

    proc_path = args.processor or args.model
    processor = AutoProcessor.from_pretrained(proc_path)

    config = AutoConfig.from_pretrained(args.model)
    kind = speech_model_kind(config)
    if kind is None:
        print(
            f"REFUSAL (exit 96): this checkpoint's model type "
            f"{getattr(config, 'model_type', None)!r} is not a known speech kind; "
            f"its geometry is undeclared, refusing rather than guessing.",
            file=sys.stderr,
        )
        raise SystemExit(96)

    model = load_speech_model(kind, args.model, dtype=torch.bfloat16)
    model = model.to(device)
    model.eval()

    adapter_used = None
    if args.adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, args.adapter)
        adapter_used = args.adapter

    return model, processor, adapter_used, kind


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


def step_generate(model, processor, rows_batch, args, device, kind):
    """Return (hyps, truncated_count) for one batch of loaded waveforms.

    Dispatches per kind.  Truncation means "the row stopped only because it ran out
    of budget": for audio_llm and seq2seq we measure it by looking for a stop id in
    the generated tokens after the prompt; for ctc it is not applicable and is
    reported as 0 with a note.
    """
    waves = [r["_wave"] for r in rows_batch]

    if kind == "audio_llm":
        return _generate_audio_llm(model, processor, waves, args, device)
    if kind == "seq2seq":
        return _generate_seq2seq(model, processor, waves, args, device)
    if kind == "ctc":
        return _generate_ctc(model, processor, waves, args, device)

    print(
        f"REFUSAL (exit 96): speech kind {kind!r} has no inference path "
        "(known: audio_llm, seq2seq, ctc), refusing rather than guessing.",
        file=sys.stderr,
    )
    raise SystemExit(96)


def _generate_audio_llm(model, processor, waves, args, device):
    """audio_llm (Gemma-4, Qwen2-Audio): chat-template batched greedy generation."""
    messages = [
        [
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio": w},
                    {"type": "text", "text": args.prompt},
                ],
            }
        ]
        for w in waves
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
    # its new tokens.  The stop set is the generation config's (Gemma-4 ends a turn
    # with <end_of_turn>, not the tokenizer's eos), so read it there.
    stop = model.generation_config.eos_token_id
    stop_ids = set(stop if isinstance(stop, (list, tuple)) else [stop]) - {None}
    stop_ids.add(processor.tokenizer.eos_token_id)
    truncated = 0
    for i in range(new_tokens.shape[0]):
        if not stop_ids.intersection(new_tokens[i].tolist()):
            truncated += 1
    return [h.strip() for h in hyps], truncated


def _generate_seq2seq(model, processor, waves, args, device):
    """seq2seq (Whisper): batched greedy generation with language/task control."""
    sr = int(processor.feature_extractor.sampling_rate)
    feats = processor.feature_extractor(waves, sampling_rate=sr, return_tensors="pt")
    input_features = feats.input_features.to(device=device, dtype=torch.bfloat16)
    out = model.generate(
        input_features=input_features,
        language=args.language,
        task="transcribe",
        max_new_tokens=args.max_new_tokens,
    )
    hyps = [h.strip() for h in processor.tokenizer.batch_decode(out, skip_special_tokens=True)]

    # Truncated = no stop id in the generated sequence after the prompt.
    # The stop set is the generation config's eos ids plus the tokenizer's eos.
    stop = model.generation_config.eos_token_id
    stop_ids = set(stop if isinstance(stop, (list, tuple)) else [stop]) - {None}
    stop_ids.add(processor.tokenizer.eos_token_id)
    # Whisper's generate returns the batch-longest row WITHOUT its trailing stop token
    # (the shorter rows are padded with it), so "no stop id" flagged exactly one row per
    # batch -- 19 of 300 at batch 16, measured. A row can only be truncated if the
    # sequence reached the max_new_tokens budget; below it, nothing was cut.
    truncated = 0
    if out.shape[1] >= args.max_new_tokens:
        for i in range(out.shape[0]):
            if out[i, -1].item() not in stop_ids:
                truncated += 1
    return hyps, truncated


def _generate_ctc(model, processor, waves, args, device):  # noqa: ARG001
    """ctc (Parakeet): batched argmax frame logits -> batch decode.

    Truncation does not apply to ctc: every frame emits a label (or blank), so
    generation cannot "stop early".  ``truncated_count`` is 0 with a note.
    """
    sr = int(processor.feature_extractor.sampling_rate)
    feats = processor.feature_extractor(
        waves,
        sampling_rate=sr,
        return_tensors="pt",
        return_attention_mask=True,
    )
    input_features = feats.input_features.to(device=device, dtype=torch.bfloat16)
    attention_mask = feats.attention_mask.to(device) if feats.attention_mask is not None else None
    logits = model(
        input_features=input_features,
        attention_mask=attention_mask,
    ).logits
    pred = logits.argmax(-1)
    hyps = [h.strip() for h in processor.tokenizer.batch_decode(pred)]
    return hyps, 0


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
    ap.add_argument("--language", default=None)
    args = ap.parse_args()

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    results = {
        "env": step_env(args),
        "args": vars(args),
        "n_rows": 0,
        "rows_scored": 0,
        "truncated_count": 0,
        "samples": [],
        # Every row, so two models scored on the same manifest can be compared pairwise
        # (paired bootstrap); the corpus rate alone cannot say whether a gap is noise.
        "all": [],
    }
    pairs = []
    out_path = Path(args.out)
    t0 = time.time()
    try:
        model, processor, adapter_used, kind = step_load(args, device)
        results["adapter_used"] = adapter_used
        results["kind"] = kind
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
                hyps, trun = step_generate(model, processor, batch, args, device, kind)
            truncated_total += trun
            for i, r in enumerate(batch):
                ref = r["answer"]
                hyp = hyps[i]
                pairs.append((ref, hyp))
                results["rows_scored"] = len(pairs)
                results["all"].append({"id": r["id"], "reference": ref, "hypothesis": hyp})
                if len(results["samples"]) < 25:
                    sample = {"id": r["id"], "reference": ref, "hypothesis": hyp}
                    if ref.strip():
                        sample["wer"] = word_errors(ref, hyp).rate()
                    results["samples"].append(sample)
        results["truncated_count"] = truncated_total
        if kind == "ctc":
            results["truncated_note"] = "not applicable to ctc"

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

    print(f"kind={results.get('kind')}")
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
