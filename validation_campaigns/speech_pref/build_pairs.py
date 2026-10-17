#!/usr/bin/env python3
"""build_pairs.py -- ASR preference pairs for Qwen3-ASR on Earnings-22 (SP-2 part B).

One pair record per manifest utterance whose transcripts disagree. The completions are what the
SFT checkpoint actually EMITS -- ``--n-samples`` sampled transcripts (do_sample) plus the greedy
one -- scored each with ``speech_metrics.word_errors(ref, hyp).rate()`` under
``normalize_transcript``; chosen is the LOWEST-WER transcript and rejected the HIGHEST-WER one. So
the preference is one a word error rate may state, and the DPO logprobs run over completion tokens
SFT trained to produce (the transcript text + eos). Hypotheses ``speech_metrics`` flags are dropped
before ranking -- ``is_runaway`` or ``loop_words``: a decoder that stopped listening is not a
preference, and ranking one into a reward is how a fine-tune learns to run away.

An utterance is skipped whenever the pair would NOT be a preference: the reference normalizes empty
(an empty denominator is a refusal, never a 0.0), fewer than two hypotheses survive, chosen and
rejected normalize to ONE text (one transcript counted twice), or their WERs tie (no direction).
Every skip is COUNTED BY REASON in ``<out stem>.record.json`` -- no silent drops, so the pair set is
auditable against the manifest it came from.

The pair rows are exactly PreferenceTrainConfig's audio-mode contract records:

    {"prompt": AUDIO_PROMPT_SENTINEL, "audio": <absolute wav/flac>, "chosen": ..., "rejected": ...,
     "chosen_wer": ..., "rejected_wer": ..., "id": ...}

``AUDIO_PROMPT_SENTINEL`` comes from ``foundationscale.rl.preference_trainer`` (its literal
``"<|asr_audio|>"`` ONLY if that import fails, and the record names which was used): the prompt side
is the AUDIO, rebuilt by the Qwen3-ASR transcription request in ``train.audio`` with ``language`` --
say ``--language none`` to pair exactly what eval_qwen3_asr.py decodes (Qwen3-ASR's own language
identification).

Decoding is eval_qwen3_asr.py's, row-by-row workaround included: AutoModelForMultimodalLM
(Qwen3ASRForConditionalGeneration) in bfloat16, ``apply_transcription_request(audio=waves,
language=...)`` for the prompts, and ONE ``processor.decode(row, skip_special_tokens=True,
return_format="transcription_only")`` per row -- only Qwen3ASRProcessor.decode honours
``return_format`` (transformers 5.19) and ``batch_decode`` would return "language
English<asr_text>..." verbatim. Audio is loaded with soundfile at the processor's rate and NEVER
resampled.

Exit codes: 0 ok / 95 coverage short (no pair for any row, or rows lost without a counted reason)
/ 96 refusal (manifest, columns or config refused before anything is written).
``--resume`` keeps the ids already in ``--out`` and pairs the rest.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import random
import sys
import time
import traceback
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import soundfile
import torch

from foundationscale.train.speech_metrics import (
    TRANSCRIPT_NORMALIZER_ID,
    is_runaway,
    loop_words,
    normalize_transcript,
    word_errors,
)

EXIT_OK = 0
EXIT_COVERAGE_SHORT = 95
EXIT_REFUSAL = 96

AUDIO_COLUMN = "audio"
"""Record key the audio path is published under: train_dpo passes audio_column="audio"."""

AUDIO_PROMPT_FALLBACK = "<|asr_audio|>"

SKIP_REASONS: tuple[str, ...] = (
    "missing_audio_field",
    "missing_text_field",
    "audio_path_missing",
    "audio_unreadable",
    "sample_rate_mismatch",
    "reference_normalizes_empty",
    "fewer_than_two_valid_hypotheses",
    "tie_wer",
    "chosen_rejected_identical",
)

DROP_REASONS: tuple[str, ...] = ("runaway", "loop")

# eval_qwen3_asr.py reads {"id","audio","answer"}; the sibling spellings cover the other
# speech manifests unchanged. A family with no key is refused, never guessed.
AUDIO_KEYS = ("audio", "audio_filepath", "wav", "path", "file", "audio_file", "filename")
TEXT_KEYS = ("answer", "text", "transcript", "transcription", "reference", "sentence")
ID_KEYS = ("id", "utt_id", "uttid", "key", "name")


class Refusal(Exception):
    """A precondition this script refuses before it writes anything: exit 96."""


def env_record() -> dict[str, Any]:
    """The versions a second run must match to reproduce these pairs (eval's step_env, inlined)."""
    record: dict[str, Any] = {"python": platform.python_version(), "platform": platform.platform()}
    for name in ("torch", "transformers", "soundfile"):
        try:
            record[name] = importlib.metadata.version(name)
        except Exception:
            record[name] = None
    try:
        record["cuda"] = getattr(torch.version, "cuda", None)
        record["cuda_available"] = bool(torch.cuda.is_available())
    except Exception:
        record["cuda"] = None
        record["cuda_available"] = None
    return record


def load_audio_prompt_sentinel() -> tuple[str, str]:
    """(the audio-mode prompt sentinel, and which source it came from).

    ``foundationscale.rl.preference_trainer`` owns the constant and must be used where it can be
    imported: PreferenceTrainer REFUSES a record whose prompt is anything else, so pairs written
    against a stale literal would train nothing. The literal is a rescue valve for writing pairs
    before that module is on the path -- and the record says which of the two was used rather than
    letting a stale ``<|asr_audio|>`` pass as current.
    """
    try:
        from foundationscale.rl.preference_trainer import AUDIO_PROMPT_SENTINEL
    except Exception as exc:
        return (
            AUDIO_PROMPT_FALLBACK,
            f"literal {AUDIO_PROMPT_FALLBACK!r} (import failed: {type(exc).__name__})",
        )
    return (
        AUDIO_PROMPT_SENTINEL,
        "foundationscale.rl.preference_trainer.AUDIO_PROMPT_SENTINEL",
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "Build ASR preference pairs (audio-mode contract records) "
            "from an SFT Qwen3-ASR checkpoint."
        )
    )
    ap.add_argument("--model", required=True, help="SFT checkpoint dir (weights only)")
    ap.add_argument(
        "--processor", default=None, help="processor dir; fine-tuned checkpoints carry weights only"
    )
    ap.add_argument(
        "--manifest", required=True, help="jsonl rows with an audio path and a transcript"
    )
    ap.add_argument("--out", required=True, help="pairs jsonl (contract records)")
    ap.add_argument(
        "--n-samples",
        type=int,
        default=8,
        help="sampled transcripts per utterance, plus the greedy one",
    )
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--limit", type=int, default=None, help="pair only the first --limit manifest rows"
    )
    ap.add_argument(
        "--language",
        default="en",
        help=(
            "transcription request language; none = Qwen3-ASR's own "
            "(eval_qwen3_asr.py's eval setting)"
        ),
    )
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument(
        "--max-new-tokens",
        type=int,
        default=1024,
        help="the published eval setting: sampled and greedy completions spend the same budget",
    )
    ap.add_argument(
        "--resume", action="store_true", help="keep the ids already in --out and pair the rest"
    )
    return ap.parse_args(argv)


def read_manifest(path: Path) -> list[dict[str, Any]]:
    """Every non-empty line as one JSON object; a line that is not one is REFUSED, not lost."""
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise Refusal(f"manifest line {lineno} is not valid JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise Refusal(f"manifest line {lineno} is not a JSON object: {str(row)[:60]}")
            rows.append(row)
    return rows


def detect_keys(rows: list[dict[str, Any]]) -> dict[str, str | None]:
    """Which record keys carry the audio path, the transcript and the id.

    eval_qwen3_asr.py's names (``audio`` / ``answer`` / ``id``) are tried first; the neighbouring
    spellings cover the sibling corpora's manifests unchanged. A family with no key is left None for
    the caller to refuse -- a row is never scored against a guessed column.
    """
    present: set[str] = set()
    for row in rows:
        present.update(row)
    lowered: dict[str, str] = {}
    for key in present:
        lowered.setdefault(key.lower(), key)

    def pick(candidates: Sequence[str]) -> str | None:
        for candidate in candidates:
            if candidate in present:
                return candidate
        for candidate in candidates:
            if candidate in lowered:
                return lowered[candidate]
        return None

    return {"audio": pick(AUDIO_KEYS), "text": pick(TEXT_KEYS), "id": pick(ID_KEYS)}


def step_fields(
    row: dict[str, Any],
    index: int,
    keys: dict[str, str | None],
    manifest_dir: Path,
) -> tuple[dict[str, Any] | None, str | None]:
    """(id / reference / audio path | None, skip reason | None) for one manifest row.

    A field that is not a non-empty string is no field at all -- a JSON null in the transcript
    column can neither be scored nor paired -- and is counted under its missing reason. The
    reference that normalizes to nothing is refused here and NOW: ``EditCounts.rate()`` has no
    rate over zero reference words and 0.0 would turn "nothing measured" into "perfect". The
    audio path is published ABSOLUTE: a pair set pointing at paths relative to a directory
    nobody recorded is not portable to the box that reads it next.
    """
    audio_value = row.get(keys["audio"]) if keys["audio"] else None
    text_value = row.get(keys["text"]) if keys["text"] else None
    if not isinstance(audio_value, str) or not audio_value.strip():
        return None, "missing_audio_field"
    if not isinstance(text_value, str):
        return None, "missing_text_field"
    if not normalize_transcript(text_value):
        return None, "reference_normalizes_empty"

    id_value = row.get(keys["id"]) if keys["id"] else None
    rid = str(id_value) if id_value is not None else f"row-{index:06d}"

    path = Path(audio_value).expanduser()
    base = "absolute"
    if path.is_absolute():
        path = path.resolve()
    else:
        candidate = manifest_dir / path
        if candidate.is_file():
            path, base = candidate.resolve(), "manifest"
        else:
            path, base = path.resolve(), "cwd"
    if not path.is_file():
        return None, "audio_path_missing"
    return {"id": rid, "ref": text_value, "audio": path, "base": base}, None


def step_wave(path: Path, expected_rate: int) -> tuple[Any, str | None, str | None]:
    """(mono float32 wave | None, multichannel note | None, skip reason | None).

    NO RESAMPLING, ever: a waveform whose rate is not the processor's is refused and COUNTED
    (``sample_rate_mismatch``) rather than interpolated -- the interpolation is not the audio the
    transcript belongs to, and every pair built on it would be a preference about a resampler.
    Several channels are averaged to mono and counted (``rows_mixed_to_mono``) rather than done in
    silence.
    """
    try:
        data, rate = soundfile.read(str(path), dtype="float32", always_2d=False)
    except Exception:
        return None, None, "audio_unreadable"
    if int(rate) != int(expected_rate):
        return None, None, "sample_rate_mismatch"
    note = None
    if getattr(data, "ndim", 1) == 2:
        data = data.mean(axis=1).astype("float32")
        note = "mixed_to_mono"
    return data, note, None


def step_load(args: argparse.Namespace) -> tuple[Any, Any]:
    """Qwen3-ASR processor + model exactly as eval_qwen3_asr.step_load (the -hf card recipe)."""
    from transformers import AutoModelForMultimodalLM, AutoProcessor

    # fine-tuned checkpoints carry weights only; their processor is the base model's
    processor = AutoProcessor.from_pretrained(args.processor or args.model)
    # no device_map: it needs a newer accelerate than the hf-26.04 container ships
    model = AutoModelForMultimodalLM.from_pretrained(args.model, dtype=torch.bfloat16)
    model.to("cuda").eval()
    return model, processor


def step_generate(
    model: Any,
    processor: Any,
    waves: list[Any],
    *,
    language: str | None,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float = 1.0,
    top_p: float = 1.0,
) -> tuple[list[str], list[bool]]:
    """(transcripts, ran-out-of-budget flags) for one batch, decoded as eval_qwen3_asr.py decodes.

    ``apply_transcription_request`` builds the prompt (language=None is Qwen3-ASR's own language
    identification, the eval setting), generation spends ``max_new_tokens`` on the completion side
    only, and each row is decoded ON ITS OWN with ``processor.decode(row, skip_special_tokens=True,
    return_format="transcription_only")``: only Qwen3ASRProcessor.decode honours ``return_format``
    (transformers 5.19) and ``batch_decode`` would return "language English<asr_text>..." verbatim.
    A row that ended only by spending the whole budget (no stop id in its new tokens) is flagged per
    ROW here rather than counted in a total, because pair construction must know WHICH completion
    ran out before it ranks it.
    """
    inputs = processor.apply_transcription_request(audio=waves, language=language).to(
        model.device, model.dtype
    )
    prompt_len = int(inputs["input_ids"].shape[1])
    if do_sample:
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
        )
    else:
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    new_tokens = out[:, prompt_len:]
    texts = [
        processor.decode(row, skip_special_tokens=True, return_format="transcription_only")
        for row in new_tokens
    ]

    # eval_qwen3_asr.py's stop set: the generation config's eos ids plus the tokenizer's eos.
    stop = model.generation_config.eos_token_id
    stop_ids = set(stop if isinstance(stop, (list, tuple)) else [stop]) - {None}
    stop_ids.add(processor.tokenizer.eos_token_id)
    truncated = [
        not stop_ids.intersection(new_tokens[i].tolist()) for i in range(new_tokens.shape[0])
    ]
    return [t.strip() for t in texts], [bool(flag) for flag in truncated]


def score_candidates(
    ref: str, candidates: Sequence[tuple[str, str]]
) -> tuple[list[tuple[str, str, float]], dict[str, int]]:
    """((kind, text, WER) per survivor of ``candidates``, drops counted per detector).

    ``word_errors(ref, hyp).rate()`` under ``normalize_transcript`` is the ranked score and the ONLY
    score here: it is the claim the pair record publishes beside its completions (``chosen_wer`` /
    ``rejected_wer``), and it is never blended with anything else. Runaway before loops, so a
    transcript flagged by both is counted once, under the first detector to see it.
    """
    scored: list[tuple[str, str, float]] = []
    dropped = {"runaway": 0, "loop": 0}
    for kind, text in candidates:
        if is_runaway(ref, text):
            dropped["runaway"] += 1
            continue
        if loop_words(normalize_transcript(text)) > 0:
            dropped["loop"] += 1
            continue
        scored.append((kind, text, word_errors(ref, text).rate()))
    return scored, dropped


def pick_pair(
    scored: list[tuple[str, str, float]],
) -> tuple[tuple[str, str, float] | None, tuple[str, str, float] | None, str | None]:
    """(chosen, rejected, skip reason) over the WER-scored survivors.

    chosen is the LOWEST-WER survivor, rejected the HIGHEST-WER one; ties are broken by candidate
    order (greedy first, then samples in seed order) so one seed writes exactly one pair and no
    dict ordering can decide it. A pair that would not be a preference is refused with the reason it
    is refused: equal WERs state no direction, and two transcripts normalizing to one are one
    transcript counted twice. The identical-text check comes first because identical normalized text
    ALWAYS ties in WER (the WER is derived from those tokens) -- both reasons stay reachable.
    """
    if len(scored) < 2:
        return None, None, "fewer_than_two_valid_hypotheses"
    order = sorted(range(len(scored)), key=lambda i: (scored[i][2], i))
    best = scored[order[0]]
    worst = scored[order[-1]]
    if normalize_transcript(best[1]) == normalize_transcript(worst[1]):
        return None, None, "chosen_rejected_identical"
    if best[2] == worst[2]:
        return None, None, "tie_wer"
    return best, worst, None


def run_campaign(
    args: argparse.Namespace, results: dict[str, Any], out_path: Path
) -> tuple[int, str]:
    """Fill ``results`` and write the pairs to ``out_path``; return (exit code, reason)."""
    if args.n_samples < 1:
        raise Refusal(
            f"--n-samples {args.n_samples} must be >= 1: one sampled transcript minimum, "
            f"plus the greedy one"
        )
    if args.batch_size < 1:
        raise Refusal(f"--batch-size {args.batch_size} must be >= 1")
    if args.max_new_tokens < 1:
        raise Refusal(f"--max-new-tokens {args.max_new_tokens} must be >= 1")
    if args.temperature <= 0.0:
        raise Refusal(
            f"--temperature {args.temperature} must be > 0: this is sampling, not decoding"
        )
    if not 0.0 < args.top_p <= 1.0:
        raise Refusal(f"--top-p {args.top_p} must be in (0, 1]")

    manifest_path = Path(args.manifest).expanduser()
    if not manifest_path.is_file():
        raise Refusal(f"manifest not found: {manifest_path}")
    rows = read_manifest(manifest_path)
    results["manifest_rows"] = len(rows)
    if args.limit is not None:
        rows = rows[: args.limit]
    results["rows_seen"] = len(rows)
    if not rows:
        raise Refusal(f"manifest {manifest_path} holds no rows to pair")
    keys = detect_keys(rows)
    results["manifest_keys"] = keys
    if keys["audio"] is None:
        raise Refusal(
            f"manifest {manifest_path} has no audio column ({', '.join(AUDIO_KEYS)} tried): "
            "an audio-mode pair set refuses a guessed column"
        )
    if keys["text"] is None:
        raise Refusal(
            f"manifest {manifest_path} has no transcript column ({', '.join(TEXT_KEYS)} tried): "
            "a WER ranked pair needs the reference it was ranked against"
        )

    sentinel, sentinel_source = load_audio_prompt_sentinel()
    results["audio_prompt_sentinel"] = sentinel
    results["audio_prompt_sentinel_source"] = sentinel_source
    language = None if args.language in ("", "none", "None") else args.language
    results["language"] = language

    model, processor = step_load(args)
    sampling_rate = int(processor.feature_extractor.sampling_rate)
    results["sampling_rate"] = sampling_rate

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    already: set[str] = set()
    if args.resume and out_path.is_file():
        for line in out_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                already.add(str(json.loads(line).get("id")))
        results["resumed_ids_out"] = len(already)
    mode = "a" if args.resume else "w"

    with out_path.open(mode, encoding="utf-8") as out_f:
        for start in range(0, len(rows), args.batch_size):
            chunk = rows[start : start + args.batch_size]
            prepared: list[dict[str, Any]] = []
            for offset, row in enumerate(chunk):
                index = start + offset
                fields, reason = step_fields(row, index, keys, manifest_path.parent)
                if fields is None:
                    results["skipped_by_reason"][reason] += 1
                    continue
                if fields["id"] in already:
                    results["rows_resumed"] += 1
                    continue
                wave, note, reason = step_wave(fields["audio"], sampling_rate)
                if reason:
                    results["skipped_by_reason"][reason] += 1
                    continue
                results["audio_path_base"][fields["base"]] += 1
                if note == "mixed_to_mono":
                    results["rows_mixed_to_mono"] += 1
                fields["wave"] = wave
                prepared.append(fields)
            if not prepared:
                continue

            waves = [work["wave"] for work in prepared]
            hyps: list[list[tuple[str, str]]] = [[] for _ in prepared]
            flags: list[list[bool]] = [[] for _ in prepared]
            with torch.inference_mode():
                for pass_index in range(args.n_samples + 1):
                    if pass_index == 0:
                        kind = "greedy"
                        texts, truncated = step_generate(
                            model,
                            processor,
                            waves,
                            language=language,
                            max_new_tokens=args.max_new_tokens,
                            do_sample=False,
                        )
                    else:
                        kind = f"sample-{pass_index - 1}"
                        torch.manual_seed(args.seed + pass_index - 1)
                        texts, truncated = step_generate(
                            model,
                            processor,
                            waves,
                            language=language,
                            max_new_tokens=args.max_new_tokens,
                            do_sample=True,
                            temperature=args.temperature,
                            top_p=args.top_p,
                        )
                    for i, text in enumerate(texts):
                        hyps[i].append((kind, text))
                        flags[i].append(truncated[i])

            for j, work in enumerate(prepared):
                candidates = hyps[j]
                results["hypotheses_generated"] += len(candidates)
                results["truncated_hypotheses"] += sum(1 for flag in flags[j] if flag)
                kept, dropped = score_candidates(work["ref"], candidates)
                for reason in DROP_REASONS:
                    results["hypotheses_dropped"][reason] += dropped[reason]
                chosen, rejected, reason = pick_pair(kept)
                if reason is not None:
                    results["skipped_by_reason"][reason] += 1
                    continue
                assert chosen is not None and rejected is not None
                record = {
                    "prompt": sentinel,
                    AUDIO_COLUMN: str(work["audio"]),
                    "chosen": chosen[1],
                    "rejected": rejected[1],
                    "chosen_wer": chosen[2],
                    "rejected_wer": rejected[2],
                    "id": work["id"],
                }
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                results["pairs_written"] += 1
                results["chosen_source"]["greedy" if chosen[0] == "greedy" else "sampled"] += 1
                results["rejected_source"]["greedy" if rejected[0] == "greedy" else "sampled"] += 1
                if len(results["samples"]) < 10:
                    results["samples"].append(
                        {
                            "id": work["id"],
                            "chosen_kind": chosen[0],
                            "rejected_kind": rejected[0],
                            "chosen_wer": chosen[2],
                            "rejected_wer": rejected[2],
                        }
                    )

    skipped = sum(results["skipped_by_reason"].values())
    results["rows_skipped"] = skipped
    available = results["pairs_written"] + results["rows_resumed"]
    results["pairs_available"] = available
    results["rows_without_pair"] = results["rows_seen"] - available
    unaccounted = results["rows_seen"] - available - skipped
    results["rows_unaccounted"] = unaccounted
    if unaccounted != 0:
        return (
            EXIT_COVERAGE_SHORT,
            f"coverage short: {unaccounted} rows left the run with neither a pair "
            f"nor a counted reason",
        )
    if available == 0:
        return EXIT_COVERAGE_SHORT, f"coverage short: no pair from {results['rows_seen']} rows"
    return EXIT_OK, "ok"


def main() -> int:
    args = parse_args()
    out_path = Path(args.out).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    record_path = out_path.parent / (out_path.stem + ".record.json")
    results: dict[str, Any] = {
        "kind": "asr_preference_pairs",
        "env": env_record(),
        "args": vars(args),
        "out": str(out_path),
        "normalizer": TRANSCRIPT_NORMALIZER_ID,
        "audio_column": AUDIO_COLUMN,
        "audio_prompt_sentinel": None,
        "audio_prompt_sentinel_source": None,
        "manifest_rows": 0,
        "manifest_keys": None,
        "language": None,
        "sampling_rate": None,
        "rows_seen": 0,
        "rows_resumed": 0,
        "resumed_ids_out": 0,
        "pairs_written": 0,
        "rows_skipped": 0,
        "rows_without_pair": 0,
        "rows_unaccounted": 0,
        "rows_mixed_to_mono": 0,
        "audio_path_base": {"absolute": 0, "manifest": 0, "cwd": 0},
        "skipped_by_reason": {reason: 0 for reason in SKIP_REASONS},
        "hypotheses_generated": 0,
        "hypotheses_dropped": {reason: 0 for reason in DROP_REASONS},
        "truncated_hypotheses": 0,
        "chosen_source": {"greedy": 0, "sampled": 0},
        "rejected_source": {"greedy": 0, "sampled": 0},
        "samples": [],
    }
    code, reason = EXIT_OK, "ok"
    t0 = time.time()
    try:
        code, reason = run_campaign(args, results, out_path)
    except Refusal as exc:
        code, reason = EXIT_REFUSAL, f"refusal: {exc}"
        results["refusal"] = str(exc)
        sys.stderr.write(f"REFUSE {exc}\n")
    except SystemExit:
        code, reason = EXIT_REFUSAL, "refusal: exited before the pairs were written"
        raise
    except Exception:
        results["fatal"] = traceback.format_exc()
        code, reason = 1, "fatal"
        sys.stderr.write(results["fatal"])
    finally:
        results["elapsed_s"] = time.time() - t0
        results["exit_code"] = code
        results["exit_reason"] = reason
        record_path.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")

    print(f"kind={results['kind']}")
    print(f"rows={results['rows_seen']} manifest_rows={results['manifest_rows']}")
    print(
        f"pairs_written={results['pairs_written']} resumed={results['rows_resumed']} "
        f"skipped={results['rows_skipped']} unaccounted={results['rows_unaccounted']}"
    )
    for skip_reason, count in results["skipped_by_reason"].items():
        print(f"skip.{skip_reason}={count}")
    print(
        f"hypotheses={results['hypotheses_generated']} dropped={results['hypotheses_dropped']} "
        f"truncated={results['truncated_hypotheses']}"
    )
    print(
        f"sentinel={results['audio_prompt_sentinel']!r} ({results['audio_prompt_sentinel_source']})"
    )
    print(f"pairs={out_path}")
    print(f"record={record_path}")
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
