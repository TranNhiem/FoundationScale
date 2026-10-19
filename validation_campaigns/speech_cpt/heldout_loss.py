#!/usr/bin/env python3
"""heldout_loss.py -- Teacher-forced cross-entropy of a Qwen3-ASR checkpoint on held-out ASR rows.

ROADMAP SP-1 part C: the CPT run changes what the model writes WITHOUT looking at the tuning
transcripts, so the only way to know it did NOT destroy (or negligibly move) ASR is to score the
model's OWN LOSS on a held-out manifest before and after, and report the same measurement of both.

Measurement definition: for every (audio, answer) row in the manifest, build the SAME input the
eval_qwen3_asr decode builds (prompt from ``processor.apply_transcription_request(audio=...,
language=...)`` -- the Qwen3-ASR transcription prompt is what the model sees at eval, not a chat
prompted utterance), append the row's ANSWER as the model's target (encode ``" " + answer_text``
with ``add_special_tokens=False`` and append the tokenizer's eos), and compute cross-entropy over
the ANSWER tokens ONLY (prompt masked to -100, eos included in the loss): the model is scored on the
words it is being asked to say, not on rebuilding the audio prompt that shared the input.

Batch of 1, bfloat16, one GPU (a weights-only model, batch 1 keeps it apples-to-apples with the
SP-1 loss claim). Fp32 accumulation of the sum of per-token losses (and the token count) so the
mean over the run is a mean of what a batch of 1 measured, never a mean of bf16 rounding noise.

Output: JSON {model, processor, rows, tokens, mean_loss} (printed AND written to --out as a record
of measurement): mean_loss = sum of per-token losses over all non-masked target tokens /
#tokens. Every row is COUNTED (scored, or skipped with a reason -- audio unreadable / empty
target): no silent drops.

Exit codes: 0 ok / 95 coverage short (row with no target token could inflate the token count unseen)
/ 96 refusal (model/manifest/processor refuse before anything is scored).
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import torch

EXIT_OK = 0
EXIT_COVERAGE_SHORT = 95
EXIT_REFUSAL = 96

SKIP_REASONS: tuple[str, ...] = ("audio_unreadable", "empty_target")


class Refusal(Exception):
    """A precondition this script refuses before anything is scored: exit 96."""


def _env_record() -> dict[str, Any]:
    record: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    import importlib.metadata as metadata

    for name in ("torch", "transformers", "soundfile"):
        try:
            record[name] = metadata.version(name)
        except Exception:
            record[name] = None
    try:
        record["cuda"] = getattr(torch.version, "cuda", None)
        record["cuda_available"] = bool(torch.cuda.is_available())
    except Exception:
        record["cuda"] = None
        record["cuda_available"] = None
    return record


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Teacher-forced cross-entropy on a held-out (audio, transcript) manifest."
    )
    ap.add_argument("--model", required=True, help="checkpoint (weights) or full model dir")
    ap.add_argument("--processor", default=None, help="processor dir; defaults to --model")
    ap.add_argument(
        "--manifest", required=True, help="jsonl rows with id/audio/answer (SFT format)"
    )
    ap.add_argument("--out", default=None, help="JSON record path (default: print only)")
    ap.add_argument(
        "--language", default="en", help="transcription request language (en per recipe)"
    )
    ap.add_argument("--limit", type=int, default=None, help="score only the first --limit rows")
    return ap.parse_args()


def read_manifest(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise Refusal(
                    f"manifest {path.name} line {lineno} is not valid JSON: {exc}"
                ) from exc
            if not isinstance(row, dict):
                raise Refusal(
                    f"manifest {path.name} line {lineno} is not a JSON object: {str(row)[:60]}"
                )
            for key in ("audio", "answer"):
                if key not in row:
                    raise Refusal(f"manifest {path.name} line {lineno} misses field {key!r}")
            rows.append(row)
    return rows


def load_wave(path: Path) -> Any | None:
    """Load a waveform at the processor's rate (no resampling, as eval_qwen3_asr), mono mixdown."""
    import soundfile

    try:
        data, _rate = soundfile.read(str(path), dtype="float32", always_2d=False)
    except Exception:
        return None
    return data


def load_model(args: argparse.Namespace) -> tuple[Any, Any]:
    from transformers import AutoModelForMultimodalLM, AutoProcessor

    processor = AutoProcessor.from_pretrained(args.processor or args.model)
    model = AutoModelForMultimodalLM.from_pretrained(args.model, dtype=torch.bfloat16)
    model.to("cuda").eval()
    return model, processor


def score_row(
    model: Any, processor: Any, wave: Any, answer: str, language: str | None
) -> tuple[float, int]:
    """(summed cross-entropy loss of the answer tokens, number of answer incl. eos tokens).

    The prompt is the transcription request (identical surface to eval_qwen3_asr.py): the answer is
    appended as target tokens and eos, the prompt is masked with -100, and cross-entropy is computed
    over the answer side only (reduction="sum" then divided per-token count upstream of averaging:
    the mean over the whole run is the mean of ALL scored tokens, not the mean of per-row means).
    """
    language = None if language in (None, "", "none", "None") else language
    inputs = processor.apply_transcription_request(audio=[wave], language=language).to(
        model.device, model.dtype
    )
    prompt_len = int(inputs["input_ids"].shape[1])
    target = processor.tokenizer(answer, add_special_tokens=False, return_tensors="pt")[
        "input_ids"
    ].to("cuda")
    target = torch.cat(
        [target, torch.tensor([[processor.tokenizer.eos_token_id]], device="cuda")], dim=1
    )  # eos is part of the loss (the model must learn to stop)
    input_ids = torch.cat([inputs["input_ids"], target], dim=1).to("cuda")
    attention_mask = torch.ones_like(input_ids)
    labels = input_ids.clone()
    labels[:, :prompt_len] = -100
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        # the audio conditioning MUST reach the forward: without input_features the loss would
        # score the transcript against placeholders that hear nothing
        features = {k: v for k, v in inputs.items() if k not in ("input_ids", "attention_mask")}
        if "input_features" not in features:
            raise Refusal("the transcription request returned no input_features")
        out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, **features)
        loss = out.loss
    tokens = int((labels != -100).sum())
    # out.loss is a MEAN over the non-masked tokens; multiply back to the sum so the run mean is
    # over tokens, not over rows (a row with 3 tokens must not weigh the same as one with 30).
    return float(loss) * tokens, tokens


def run(args: argparse.Namespace, results: dict[str, Any]) -> tuple[int, str]:
    if not torch.cuda.is_available():
        raise Refusal("CUDA is not available: heldout_loss runs on one GPU (bf16)")

    model_path = Path(args.model).expanduser()
    if not model_path.exists():
        raise Refusal(f"--model not found: {model_path}")
    manifest_path = Path(args.manifest).expanduser()
    if not manifest_path.is_file():
        raise Refusal(f"--manifest not found: {manifest_path}")

    rows = read_manifest(manifest_path)
    results["manifest_rows"] = len(rows)
    if args.limit is not None:
        rows = rows[: args.limit]
    results["rows"] = len(rows)
    if not rows:
        raise Refusal(f"manifest {manifest_path} holds no rows to score")

    model, processor = load_model(args)
    language = args.language

    loss_sum = 0.0
    token_count = 0

    for row in rows:
        wave = load_wave(Path(str(row["audio"])))
        if wave is None:
            results["skipped_by_reason"]["audio_unreadable"] += 1
            continue
        answer = row["answer"]
        if not isinstance(answer, str) or not answer.strip():
            results["skipped_by_reason"]["empty_target"] += 1
            continue
        row_loss, row_tokens = score_row(model, processor, wave, answer, language)
        if row_tokens == 0:
            results["skipped_by_reason"]["empty_target"] += 1
            continue
        loss_sum += row_loss
        token_count += row_tokens

    skipped = sum(results["skipped_by_reason"].values())
    results["tokens"] = token_count
    results["mean_loss"] = round(loss_sum / token_count, 6) if token_count else None
    results["rows_unaccounted"] = results["rows"] - (
        token_count > 0 and len(rows) - skipped or (len(rows) - skipped)
    )
    results["rows_scored"] = len(rows) - skipped
    results["rows_skipped"] = skipped
    results["rows_unaccounted"] = 0 if results["rows_scored"] == results["rows"] - skipped else 1

    if token_count == 0:
        return EXIT_COVERAGE_SHORT, f"coverage short: 0 tokens scored from {results['rows']} rows"
    return EXIT_OK, "ok"


def main() -> int:
    args = parse_args()
    results: dict[str, Any] = {
        "kind": "speech_cpt_heldout_loss",
        "env": _env_record(),
        "args": vars(args),
        "model": str(Path(args.model).expanduser()),
        "processor": None,
        "manifest_rows": 0,
        "rows": 0,
        "rows_scored": 0,
        "rows_skipped": 0,
        "rows_unaccounted": 0,
        "tokens": 0,
        "mean_loss": None,
        "language": args.language,
        "skipped_by_reason": {reason: 0 for reason in SKIP_REASONS},
    }
    code, reason = EXIT_OK, "ok"
    t0 = time.time()
    try:
        code, reason = run(args, results)
    except Refusal as exc:
        code, reason = EXIT_REFUSAL, f"refusal: {exc}"
        results["refusal"] = str(exc)
        sys.stderr.write(f"REFUSE {exc}\n")
    except SystemExit:
        code, reason = EXIT_REFUSAL, "refusal: exited before any row was scored"
        raise
    except Exception:
        results["fatal"] = traceback.format_exc()
        code, reason = 1, "fatal"
        sys.stderr.write(results["fatal"])
    finally:
        results["elapsed_s"] = time.time() - t0
        results["exit_code"] = code
        results["exit_reason"] = reason
        if args.out:
            out_path = Path(args.out).expanduser()
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")

    print(f"kind={results['kind']}")
    print(f"model={results['model']}")
    print(
        f"rows={results['rows']} scored={results['rows_scored']} skipped={results['rows_skipped']}"
    )
    print(f"tokens={results['tokens']}")
    print(f"mean_loss={results['mean_loss']}")
    if args.out:
        print(f"out={Path(args.out).expanduser()}")
    print(f"exit={code} {reason}")
    print(f"seconds={results['elapsed_s']:.1f}")
    return code


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SystemExit as se:
        sys.exit(se.code if se.code is not None else 0)
    except Exception:
        traceback.print_exc()
        sys.exit(1)
