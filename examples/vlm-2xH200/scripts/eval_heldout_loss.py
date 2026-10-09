#!/usr/bin/env python3
"""eval_heldout_loss.py — held-out teacher-forced NLL for a FoundationScale VLM.

The metric is "mean negative log-likelihood in nats per assistant token".  Every
row is tokenised with FoundationScale's *own* TRAIN collator
(`train_conversation_collator_or_refuse`), which is the same call the trainer
makes: it masks prompt / image / padding tokens to -100 (so only assistant tokens
are scored) and *drops* rows longer than `max_length`.  The number printed here is
therefore directly comparable with the loss column of the FoundationScale
trainer logs; perplexity is exp(mean NLL).

Good for a 2x H200 VM (the backbone is sharded over both GPUs with
device_map="auto", rows are scored one at a time):

    python eval_heldout_loss.py \
        --model /models/qwen2p5_vl_7b --adapter /runs/lora_sft \
        --data heldout.jsonl --max-length 16384 --limit 500 --out results.json

`--data` is FoundationScale canonical JSONL: one object per line with
`conversations: [{"from": ..., "value": ...}, ...]`, an optional image list in the
`--image-column`, and a `source` string.

Writes a JSON summary and prints mean NLL, perplexity and rows used/dropped.
"""

from __future__ import annotations

import argparse
import gc
import inspect
import json
import math
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import transformers
from peft import PeftModel
from transformers import AutoModelForImageTextToText, AutoProcessor

# The one piece of FoundationScale we rely on: the batch builder used in training.
from foundationscale.train.conversation import train_conversation_collator_or_refuse

IGNORE = -100  # collator label for "prompt / media / padding: never score me"
CE_CHUNK = 256  # sequence positions per fp32 cross-entropy chunk (~150 MB at vocab 152k)


# --------------------------------------------------------------------------- CLI


class _Help(argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter):
    pass


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        prog="eval_heldout_loss.py",
        description=(
            "Teacher-forced mean NLL over assistant tokens on a held-out JSONL file, "
            "computed with FoundationScale's own TRAIN collator so the masking matches "
            "training exactly."
        ),
        formatter_class=_Help,
        epilog=(
            "example:\n"
            "  python eval_heldout_loss.py --model /models/vlm_7b --adapter /runs/lora \\\n"
            "      --data heldout.jsonl --max-length 16384 --limit 500 --out results.json\n"
        ),
    )
    ap.add_argument(
        "--model",
        required=True,
        metavar="BASE_DIR",
        help="HF model directory for AutoModelForImageTextToText (loaded in bf16 on both GPUs)",
    )
    ap.add_argument(
        "--adapter",
        default=None,
        metavar="ADAPTER_DIR",
        help="optional PEFT/LoRA adapter directory, wrapped with PeftModel.from_pretrained",
    )
    ap.add_argument(
        "--data",
        required=True,
        metavar="heldout.jsonl",
        help="held-out rows in FoundationScale canonical JSONL form "
        "(conversations [{from,value}], optional image list, source)",
    )
    ap.add_argument(
        "--max-length",
        type=int,
        default=16384,
        metavar="TOKENS",
        help="collator max_length; rows longer than this are dropped (overlong='drop')",
    )
    ap.add_argument(
        "--image-column",
        default="image",
        metavar="COL",
        help="column holding each row's image list ('none' declares no image column)",
    )
    ap.add_argument(
        "--video-frames",
        type=int,
        default=None,
        metavar="N",
        help="frames per clip for rows with a video; required when the data has video "
        "rows, and should match FOUNDATIONSCALE_TRAIN_VIDEO_FRAMES used in training",
    )
    ap.add_argument(
        "--video-column", default="video", metavar="COL", help="column holding the clip path"
    )
    ap.add_argument(
        "--video-cache-dir",
        default=None,
        metavar="DIR",
        help="decoded-frame cache (default: .fs_video_frames next to the data file)",
    )
    ap.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help="only score the first N rows of --data (default: every row)",
    )
    ap.add_argument(
        "--out",
        default="results.json",
        metavar="FILE",
        help="where to write the JSON summary",
    )
    return ap.parse_args(argv)


# ------------------------------------------------------------------ data sources


def read_rows(path: str, limit: int | None):
    """Yield the canonical JSONL rows of `path` in file order (at most `limit`)."""
    yielded = 0
    with Path(path).open(encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            if limit is not None and yielded >= limit:
                break
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"error: {path}:{lineno}: not valid JSON ({exc})") from exc
            if not isinstance(row, dict):
                raise SystemExit(
                    f"error: {path}:{lineno}: expected a JSON object, got {type(row).__name__}"
                )
            yielded += 1
            yield row


def load_processor(model_dir: str, adapter_dir: str | None):
    """Prefer the processor saved with the run (it carries the chat template that
    training used); fall back to the one shipped with the base model."""
    for candidate in (adapter_dir, model_dir):
        if not candidate:
            continue
        files = {p.name for p in Path(candidate).glob("*.json")}
        if "preprocessor_config.json" in files and "tokenizer_config.json" in files:
            print(f"  processor        {candidate}")
            return AutoProcessor.from_pretrained(candidate)
    print(f"  processor        {model_dir}")
    return AutoProcessor.from_pretrained(model_dir)


def _from_pretrained_bf16(model_dir: str):
    """Backbone in bf16, sharded over the 2 H200s with `device_map="auto"`.

    transformers renamed `from_pretrained(torch_dtype=...)` to `dtype=...`; a kwarg
    the installed release does not know is silently written into the *config* and the
    weights then load in fp32 (twice the memory, OOM on this box).  Try both names and
    keep whichever really produced bf16 weights.
    """
    for kw in ("torch_dtype", "dtype"):
        model = AutoModelForImageTextToText.from_pretrained(
            model_dir, device_map="auto", **{kw: torch.bfloat16}
        )
        got = next(model.parameters()).dtype
        if got == torch.bfloat16:
            return model
        print(f"  note: {kw}= was ignored (weights came back as {got}); retrying", file=sys.stderr)
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    raise SystemExit("error: this transformers release would not load the backbone in bf16")


def load_model(model_dir: str, adapter_dir: str | None):
    """Backbone (+ optional adapter) in bf16 / eval mode / no gradients."""
    model = _from_pretrained_bf16(model_dir)
    if adapter_dir:
        # PEFT only allocates the LoRA weights; the base stays sharded as loaded.
        model = PeftModel.from_pretrained(model, adapter_dir, is_trainable=False)
        print(f"  adapter          {adapter_dir}")
    else:
        print("  adapter          (none)")
    model.eval()
    if getattr(model.config, "use_cache", None) is not None:
        model.config.use_cache = False  # teacher forcing never reads a KV cache
    return model


# ------------------------------------------------------------------- batch / loss


def collate_one(
    collator: Any, row: Mapping[str, Any], row_index: int, drop_reasons: dict[str, int]
) -> Any | None:
    """Collate exactly one row; return None when the TRAIN collator dropped it.

    With `overlong="drop"` the collator silently removes rows that exceed
    `max_length`; for a mini-batch of one that shows up as an empty encoding (or as a
    refusal/pad error on an empty batch).  Both are recorded as an overlong drop so the
    "rows used / dropped" bookkeeping stays in step with training.
    """
    try:
        batch = collator([row])
    except Exception as exc:  # noqa: BLE001 - the collator refuses several batch shapes
        tag = f"collator_error:{type(exc).__name__}"
        drop_reasons[tag] = drop_reasons.get(tag, 0) + 1
        print(f"  warning: row {row_index} dropped ({type(exc).__name__}: {exc})", file=sys.stderr)
        return None
    if _batch_is_empty(batch):
        drop_reasons["overlong"] = drop_reasons.get("overlong", 0) + 1
        return None
    return batch


def _batch_is_empty(batch: Any) -> bool:
    """True when every row of the mini-batch was dropped (overlong='drop')."""
    if not isinstance(batch, Mapping) or not batch:
        return True
    ids = batch.get("input_ids")
    if ids is None:
        return True
    if torch.is_tensor(ids):
        return ids.numel() == 0
    return len(ids) == 0 or len(ids[0]) == 0


def move_to(batch: Any, device) -> dict[str, Any]:
    """Tensors go to the first device of the sharding; accelerate moves intermediate
    activations across the 2 GPUs on the way forward."""
    return {
        k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()
    }


def forward_kwargs(model: Any, batch: Mapping[str, Any]) -> dict[str, Any]:
    """Tensors whose names `forward` accepts (the collator may add bookkeeping keys)."""
    params = inspect.signature(model.forward).parameters
    accepts_kwargs = any(p.kind is p.VAR_KEYWORD for p in params.values())
    return {
        k: v for k, v in batch.items() if torch.is_tensor(v) and (accepts_kwargs or k in params)
    }


def teacher_forced_nll(outputs: Any, labels: torch.Tensor) -> tuple[float, int]:
    """(total NLL in nats, number of assistant tokens scored) for one row.

    Only `labels != -100` counts: the TRAIN collator has already masked prompt, media
    and padding tokens to -100, leaving exactly the tokens the trainer is supervised on.
    Per-row totals are accumulated in Python floats and divided by the token count at
    the very end, i.e. the mean is token-weighted (long rows count more).

    If we get logits back we sum the cross entropy ourselves in fp32 chunks -- a full
    [len, vocab] fp32 logit tensor is several GB at max_length=16384, and fp32 keeps the
    sum exact.  Otherwise we fall back to `outputs.loss`, which is already the *mean*
    over the scored tokens and only needs re-weighting by that count.
    """
    n_tok = int((labels[:, 1:] != IGNORE).sum())  # labels[0] has no predecessor
    if n_tok == 0:
        return 0.0, 0

    logits = getattr(outputs, "logits", None)
    full_logits = (
        logits is not None
        and logits.dim() == 3
        and logits.shape[0] == labels.shape[0]
        and logits.shape[1] == labels.shape[1]
    )
    if full_logits:
        # logits[:, i] is the distribution for labels[:, i + 1].
        targets = labels[:, 1:]
        total = 0.0
        for start in range(0, targets.shape[1], CE_CHUNK):
            stop = min(start + CE_CHUNK, targets.shape[1])
            chunk = targets[:, start:stop]
            if not bool((chunk != IGNORE).any()):
                continue
            lp = logits[:, start:stop, :].float()
            total += float(
                F.cross_entropy(lp.transpose(1, 2), chunk, ignore_index=IGNORE, reduction="sum")
            )
            del lp  # free the fp32 copy before the next chunk
        return total, n_tok

    loss = getattr(outputs, "loss", None)
    if loss is not None:
        return float(loss) * n_tok, n_tok
    raise SystemExit("error: the model returned neither logits nor a loss; cannot score a row")


# ------------------------------------------------------------------------- output


def summarize(result: Mapping[str, Any]) -> None:
    mean, ppl = result["mean_nll_nats"], result["perplexity"]
    reasons = result["drop_reasons"] or {}
    reason_note = ", ".join(f"{k}={v}" for k, v in sorted(reasons.items())) or "(none)"
    print()
    print("== eval_heldout_loss: summary " + "=" * 40)
    print(f"  model            {result['model']}")
    print(f"  adapter          {result['adapter'] or '(none)'}")
    print(f"  data             {result['data']}")
    print(f"  max length       {result['max_length']} tokens (overlong rows dropped)")
    print(f"  rows read        {result['rows_read']}")
    print(f"  rows used        {result['rows_used']}")
    print(f"  rows dropped     {result['rows_dropped']} ({reason_note})")
    print(f"  tokens scored    {result['tokens_scored']:,} assistant tokens")
    if mean is None:
        print("  mean NLL         n/a (no assistant tokens were scored)")
    else:
        print(f"  mean NLL         {mean:.6f} nats/token ({mean / math.log(2):.4f} bits/token)")
        print(f"  perplexity       {ppl:.4f}")
    print(f"  devices          {', '.join(result['devices']) or '(cpu)'}")
    print(f"  wall time        {result['wall_seconds']:.1f} s")
    print(f"  results JSON     {result['results_path']}")
    print("=" * 70)


# ----------------------------------------------------------------------------- main


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.max_length < 8:
        raise SystemExit("error: --max-length must be at least 8 tokens")
    if args.limit is not None and args.limit < 1:
        raise SystemExit("error: --limit must be >= 1")
    started = time.perf_counter()
    image_column = (
        None
        if args.image_column.strip().lower() in ("", "none", "null", "-")
        else args.image_column
    )

    print("== eval_heldout_loss: setup")
    print(f"  data             {args.data}")
    print(f"  model            {args.model}")
    processor = load_processor(args.model, args.adapter)

    # Exactly the collator the trainer builds for SFT: assistant-only labels and
    # overlong rows dropped/truncated exactly as in training (no dummy media injected,
    # so text-only rows are never padded out with fake images).
    # Video rows need the same frame budget training used: frames are sampled
    # inside each row's [start, end] and passed to the processor as native video.
    frames_for = None
    if args.video_frames is not None:
        import functools

        from foundationscale import video as fs_video

        data_dir = Path(args.data).resolve().parent
        frames_for = functools.partial(
            fs_video.frames_for_row,
            video_column=args.video_column,
            budget=fs_video.FrameBudget(frames=args.video_frames),
            cache_dir=args.video_cache_dir or str(data_dir / ".fs_video_frames"),
            base_dir=str(data_dir),
        )
    collator = train_conversation_collator_or_refuse(
        processor,
        image_column=image_column,
        video_column=args.video_column if frames_for is not None else None,
        max_length=args.max_length,
        overlong="drop",
        inject_dummy_media=False,
        frames_for=frames_for,
    )
    model = load_model(args.model, args.adapter)
    device = getattr(model, "device", None) or next(model.parameters()).device

    n_params = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    hf_map = getattr(model, "hf_device_map", None) or {}
    shards_per_device = {
        d: sum(1 for v in hf_map.values() if str(v) == d)
        for d in sorted({str(v) for v in hf_map.values()})
    }
    print(f"  parameters       {n_params:,} total, {n_train:,} trainable")
    print(f"  shard map        {shards_per_device or '(unsharded)'}")
    print()

    rows_read = rows_used = rows_dropped = 0
    drop_reasons: dict[str, int] = {}
    total_nll = 0.0
    total_tok = 0

    with torch.no_grad():  # teacher forcing: eval mode, no grads, batch size 1
        for row in read_rows(args.data, args.limit):
            rows_read += 1
            batch = collate_one(collator, row, rows_read, drop_reasons)
            if batch is None:
                rows_dropped += 1
                continue
            if "labels" not in batch:
                raise SystemExit("error: the TRAIN collator returned no `labels`; nothing to score")
            batch = move_to(batch, device)
            outputs = model(**forward_kwargs(model, batch))
            nll, n_tok = teacher_forced_nll(outputs, batch["labels"])
            rows_used += 1
            total_nll += nll
            total_tok += n_tok
            if rows_used % 20 == 0:
                print(
                    f"    {rows_used} rows used, running mean NLL {total_nll / total_tok:.4f} nats",
                    flush=True,
                )

    mean_nll = (total_nll / total_tok) if total_tok else None
    # exp() overflows past ~709; clamp instead of writing an invalid `Infinity` to JSON.
    perplexity = math.exp(min(mean_nll, 700.0)) if mean_nll is not None else None

    result = {
        "script": "eval_heldout_loss.py",
        "model": str(args.model),
        "adapter": str(args.adapter) if args.adapter else None,
        "data": str(args.data),
        "image_column": image_column,
        "max_length": args.max_length,
        "limit": args.limit,
        "dtype": "bfloat16",
        "device_map": "auto",
        "devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        "shards_per_device": shards_per_device,
        "collator": (
            "foundationscale.train.conversation.train_conversation_collator_or_refuse("
            "overlong='drop', inject_dummy_media=False, pad_to_max_length=False)"
        ),
        "metric": (
            "token-weighted mean NLL over labels != -100 (assistant tokens); "
            "perplexity = exp(mean NLL)"
        ),
        "parameters": {"total": n_params, "trainable": n_train},
        "rows_read": rows_read,
        "rows_used": rows_used,
        "rows_dropped": rows_dropped,
        "drop_reasons": drop_reasons,
        "tokens_scored": total_tok,
        "total_nll": total_nll,
        "mean_nll_nats": mean_nll,
        "mean_nll_bits": (mean_nll / math.log(2)) if mean_nll is not None else None,
        "perplexity": perplexity,
        "peak_gpu_memory_gb": {
            f"cuda:{i}": round(torch.cuda.max_memory_allocated(i) / 2**30, 3)
            for i in range(torch.cuda.device_count())
        },
        "wall_seconds": round(time.perf_counter() - started, 3),
        "versions": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "transformers": transformers.__version__,
        },
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)
        fh.write("\n")
    result["results_path"] = str(out_path)

    summarize(result)
    if rows_used == 0:
        print(
            f"error: no row was scored ({rows_dropped} of {rows_read} were dropped); "
            "--max-length is probably too small",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
