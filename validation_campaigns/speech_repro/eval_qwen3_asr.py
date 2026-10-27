#!/usr/bin/env python3
"""eval_qwen3_asr.py — Held-out ASR word error rate for Alibaba's Qwen3-ASR (-hf release).

Decodes a held-out manifest with Qwen3-ASR and reports corpus word error rate, reproducing
Qwen's published LibriSpeech numbers (1.7B: clean 1.63 / other 3.38, percent).  It reads the
same manifest eval_wer.py reads (JSON lines: id, audio, answer) and writes the same output JSON
(the "all" list of {id, reference, hypothesis} plus the WER/CER manifests computed with the very
helpers eval_wer.py imports), so score_repro.py can score this run unchanged.  Greedy decoding,
bfloat16, max_new_tokens 1024 -- the published eval setting.  ``--num-beams N`` (N > 1) switches to
beam search with ``--length-penalty``; the greedy generate() call is unchanged when N is 1.
``--prefetch-workers`` loads the next batch's audio in threads while the GPU decodes the current one;
the batches and waveforms are identical to the serial loader's.

Runs in the hf-26.10 profile (transformers 5.19; venv ~/envs/hf-cand-519). Qwen3-ASR trains as an
audio_llm, but the generic audio-LLM eval would build a chat prompt; decoding must go through the
model's own transcription API (Qwen3ASRForConditionalGeneration via AutoModelForMultimodalLM +
AutoProcessor.apply_transcription_request, landed in transformers >= 5.13).

Honesty rule: every number is measured, nothing assumed.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
import traceback
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import torch

from foundationscale.train.speech_metrics import (
    corpus_error_rate,
    word_errors,
)
from foundationscale.upstream.profiles import write_profile_record


def _eval_wer_helpers() -> Any:
    """Import speech_p4/eval_wer.py via importlib, and return it as a module.

    It refuses Qwen3-ASR at exit 96 (its speech kinds do not declare this geometry), so its
    generation helpers cannot be reused -- but its step_env and step_audio are exactly the env
    record and "no resampling, ever" audio loading every eval run should share, and they live
    only in that script.
    """
    path = Path(__file__).resolve().parent.parent / "speech_p4" / "eval_wer.py"
    spec = importlib.util.spec_from_file_location("speech_repro_eval_wer", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load eval_wer helpers from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def step_load(args: argparse.Namespace) -> tuple[Any, Any]:
    """Load Qwen3-ASR processor + model as the official -hf model card prescribes.

    AutoModelForMultimodalLM resolves to Qwen3ASRForConditionalGeneration for this checkpoint;
    bfloat16 on cuda is the published inference recipe.
    """
    from transformers import AutoModelForMultimodalLM, AutoProcessor

    # fine-tuned checkpoints carry weights only; their processor is the base model's
    processor = AutoProcessor.from_pretrained(args.processor or args.model)
    # no device_map: it needs a newer accelerate than the hf-26.04 container ships
    model = AutoModelForMultimodalLM.from_pretrained(args.model, dtype=torch.bfloat16)
    model.to("cuda").eval()
    return model, processor


def generate_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    """The generate() arguments: greedy (the published setting) unless --num-beams > 1."""
    if args.num_beams < 1:
        raise ValueError(f"--num-beams {args.num_beams}: must be >= 1")
    kwargs: dict[str, Any] = {"max_new_tokens": args.max_new_tokens, "do_sample": False}
    if args.num_beams > 1:
        kwargs.update(num_beams=args.num_beams, length_penalty=args.length_penalty, early_stopping=True)
    return kwargs


def iter_audio_batches(
    rows: list[dict[str, Any]], batch_size: int, load: Callable[[str], Any], workers: int
) -> Iterator[tuple[list[dict[str, Any]], list[Any]]]:
    """Yield (batch, waveforms) in manifest order.

    With workers > 0 the next batch's audio is loaded in a thread pool while the caller decodes
    the current batch; the batches and waveforms are the same as with workers == 0. Waveforms are
    not kept on the rows, so each batch's audio is freed once it has been decoded.
    """
    if workers <= 0:
        for start in range(0, len(rows), batch_size):
            batch = rows[start : start + batch_size]
            yield batch, [load(r["audio"]) for r in batch]
        return
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending = [pool.submit(load, r["audio"]) for r in rows[:batch_size]]
        for start in range(0, len(rows), batch_size):
            batch, current = rows[start : start + batch_size], pending
            nxt = rows[start + batch_size : start + 2 * batch_size]
            pending = [pool.submit(load, r["audio"]) for r in nxt]
            yield batch, [f.result() for f in current]


def step_generate(
    model: Any, processor: Any, waves: list[Any], args: argparse.Namespace
) -> tuple[list[str], int]:
    """Return (hyps, truncated_count) for one batch of loaded waveforms.

    Batched greedy transcription through ``apply_transcription_request`` (language=None means
    Qwen3-ASR runs its own language identification, as at eval time).  Truncation is judged the
    way eval_wer.py judges it on its audio_llm path: a row is truncated iff its new tokens hold
    no stop id, i.e. it ended only by spending the whole max_new_tokens budget.
    """
    inputs = processor.apply_transcription_request(audio=waves, language=args.language).to(
        model.device, model.dtype
    )
    prompt_len = int(inputs["input_ids"].shape[1])
    out = model.generate(**inputs, **generate_kwargs(args))
    # per row: only Qwen3ASRProcessor.decode honours return_format (transformers 5.19);
    # batch_decode ignores it and returns "language English<asr_text>..." verbatim
    texts = [
        processor.decode(row, skip_special_tokens=True, return_format="transcription_only")
        for row in out[:, prompt_len:]
    ]

    # The stop set is the generation config's eos ids plus the tokenizer's eos -- identical to
    # eval_wer.py's audio_llm path.
    stop = model.generation_config.eos_token_id
    stop_ids = set(stop if isinstance(stop, (list, tuple)) else [stop]) - {None}
    stop_ids.add(processor.tokenizer.eos_token_id)
    truncated = 0
    for i in range(out.shape[0]):
        if not stop_ids.intersection(out[i, prompt_len:].tolist()):
            truncated += 1
    return [t.strip() for t in texts], truncated


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--processor", default=None)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--language", default=None)
    ap.add_argument("--num-beams", type=int, default=1, help="1 = greedy (published setting); >1 = beam search")
    ap.add_argument("--length-penalty", type=float, default=1.0, help="beam search only")
    ap.add_argument("--prefetch-workers", type=int, default=8, help="threads loading the next batch's audio; 0 = serial")
    args = ap.parse_args()
    generate_kwargs(args)  # refuse a bad --num-beams before loading anything

    helpers = _eval_wer_helpers()
    results: dict[str, Any] = {
        "env": helpers.step_env(args),
        "args": vars(args),
        "kind": "qwen3_asr",
        "adapter_used": None,
        "n_rows": 0,
        "rows_scored": 0,
        "truncated_count": 0,
        "samples": [],
        # Every row, so two models scored on the same manifest can be compared pairwise
        # (paired bootstrap); the corpus rate alone cannot say whether a gap is noise.
        "all": [],
    }
    pairs: list[tuple[str, str]] = []
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # The profile record is written up front: it says which candidate profile this decode ran
    # under (hf-26.10), whatever the decode outcome.
    write_profile_record(Path(args.out).parent, None)
    t0 = time.time()
    try:
        model, processor = step_load(args)
        sampling_rate = int(processor.feature_extractor.sampling_rate)

        rows: list[dict[str, Any]] = []
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

        def load(path: str) -> Any:
            return helpers.step_audio(path, sampling_rate)[0]

        for batch, waves in iter_audio_batches(rows, args.batch_size, load, args.prefetch_workers):
            with torch.inference_mode():
                hyps, trun = step_generate(model, processor, waves, args)
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
