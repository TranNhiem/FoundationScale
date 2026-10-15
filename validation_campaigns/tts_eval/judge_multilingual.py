from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

__all__ = ["main"]


def _load_seedtts_score() -> Any:
    """Import the sibling ``seedtts_score.py`` module via importlib.

    Returns its module object so the SIM subprocess helpers (``_run_sim`` and the
    ``_run`` abort convention) are reused verbatim rather than duplicated here.
    Raises ImportError if the file is not loadable.
    """
    path = Path(__file__).resolve().parent / "seedtts_score.py"
    spec = importlib.util.spec_from_file_location("tts_eval_seedtts_score", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load seedtts_score helpers from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _resolve_sim_helpers() -> tuple[Any, Any, Any]:
    """Return ``(run_sim, run_cmd, seedtts_score_module)``.

    Requires ``_run_sim(rows, seed_code, ckpt, python, workdir, device)`` (rows keyed
    on ``wav``/``prompt`` returning ``{wav: score}``) and ``_run(cmd, cwd)`` from
    ``seedtts_score.py``.  We refuse at runtime with a clear NotImplementedError naming
    what is missing rather than copying the upstream subprocess logic here.
    """
    module = _load_seedtts_score()
    missing: list[str] = []
    if not hasattr(module, "_run_sim"):
        missing.append("_run_sim(rows, seed_code, ckpt, python, workdir, device)")
    if not hasattr(module, "_run"):
        missing.append("_run(cmd, cwd)")
    if missing:
        raise NotImplementedError(
            "seedtts_score.py does not expose the SIM helpers this judge reuses: "
            + "; ".join(missing)
        )
    return module._run_sim, module._run, module


def _parse_row(line: str, lineno: int) -> dict[str, Any]:
    """Parse one JSONL manifest row; exit 96 naming the line if malformed.

    Each row must be a JSON object with string ``id``, string ``audio`` and string
    ``text``; ``prompt_audio`` is optional (string) and enables SIM scoring.
    """
    try:
        obj = json.loads(line)
    except json.JSONDecodeError as exc:
        sys.stderr.write(f"judge_multilingual: line {lineno}: invalid JSON: {exc}\n")
        sys.exit(96)
    if not isinstance(obj, dict):
        sys.stderr.write(f"judge_multilingual: line {lineno}: expected a JSON object\n")
        sys.exit(96)
    for key in ("id", "audio", "text"):
        if key not in obj or not isinstance(obj[key], str):
            sys.stderr.write(
                f"judge_multilingual: line {lineno}: missing or non-string field {key!r}\n"
            )
            sys.exit(96)
    prompt = obj.get("prompt_audio")
    if prompt is not None and not isinstance(prompt, str):
        sys.stderr.write(f"judge_multilingual: line {lineno}: 'prompt_audio' must be a string\n")
        sys.exit(96)
    return {
        "id": obj["id"],
        "audio": obj["audio"],
        "text": obj["text"],
        "prompt_audio": prompt,
    }


def _whisper_hypotheses(rows: list[dict[str, Any]], path: Path, language: str) -> dict[str, str]:
    """Transcribe with Whisper-large-v3 loaded from a LOCAL ``path``.

    Greedy, ``language`` forced, task transcribe, audio as float32 at 16 kHz. Clips over 30 s
    need Whisper's long-form mode, which requires timestamp tokens: ``return_timestamps=True``,
    as upstream VoiceBench's Whisper cascade (src/models/naive.py) passes.
    """
    import torch
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    processor = WhisperProcessor.from_pretrained(str(path))
    model = WhisperForConditionalGeneration.from_pretrained(str(path))
    model.to("cuda" if _has_cuda() else "cpu").eval()
    sr = int(processor.feature_extractor.sampling_rate)

    out: dict[str, str] = {}
    for r in rows:
        waveform = _load_audio(r["audio"], sr)
        feats = processor(
            waveform,
            sampling_rate=sr,
            return_tensors="pt",
            truncation=False,
            padding="longest",
            return_attention_mask=True,
        )
        with torch.inference_mode():
            generated = model.generate(
                feats.input_features.to(model.device, model.dtype),
                attention_mask=feats.attention_mask.to(model.device),
                language=language,
                task="transcribe",
                return_timestamps=True,
                do_sample=False,
            )
        text = processor.batch_decode(generated, skip_special_tokens=True)[0]
        out[r["id"]] = text.strip()
    return out


_LANGUAGE_NAMES = {"vi": "Vietnamese", "en": "English", "zh": "Chinese"}


def _qwen3omni_hypotheses(rows: list[dict[str, Any]], path: Path, language: str) -> dict[str, str]:
    """Transcribe with Qwen3-Omni's thinker (talker disabled, text out), greedy.

    Measures how well the omni model hears a language (plan phase O2's starting point), with the
    same normalisation and scoring as the ASR judges. The instruction names the language.
    """
    import torch
    from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor

    model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(str(path), dtype=torch.bfloat16)
    model.to("cuda" if _has_cuda() else "cpu")
    model.disable_talker()
    model.eval()
    processor = Qwen3OmniMoeProcessor.from_pretrained(str(path))
    sample_rate = int(processor.feature_extractor.sampling_rate)
    name = _LANGUAGE_NAMES.get(language, language)
    out: dict[str, str] = {}
    for r in rows:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio": "<array>"},
                    {"type": "text", "text": f"Transcribe the {name} audio into text."},
                ],
            },
        ]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(
            text=text,
            audio=[_load_audio(r["audio"], sample_rate)],
            return_tensors="pt",
            padding=True,
            use_audio_in_video=False,
        ).to(model.device, model.dtype)
        with torch.inference_mode():
            generated = model.generate(
                **inputs, return_audio=False, max_new_tokens=512, do_sample=False
            )
        ids = generated[0] if isinstance(generated, tuple) else generated
        ids = getattr(ids, "sequences", ids)
        prompt_len = int(inputs["input_ids"].shape[1])
        out[r["id"]] = processor.batch_decode(ids[:, prompt_len:], skip_special_tokens=True)[
            0
        ].strip()
    return out


def _qwen3asr_hypotheses(rows: list[dict[str, Any]], path: Path, language: str) -> dict[str, str]:
    """Transcribe with Qwen3-ASR (local ``path``) using the eval_qwen3_asr.py API.

    ``AutoProcessor`` + ``AutoModelForMultimodalLM`` with ``apply_transcription_request``
    (``language`` forced) and row-by-row ``processor.decode(..., return_format=
    "transcription_only")`` — ``batch_decode`` ignores the return format and echoes
    "language English<asr_text>..." verbatim.
    """
    import torch
    from transformers import AutoModelForMultimodalLM, AutoProcessor

    processor = AutoProcessor.from_pretrained(str(path))
    model = AutoModelForMultimodalLM.from_pretrained(str(path), dtype=torch.bfloat16)
    model.to("cuda" if _has_cuda() else "cpu").eval()
    sample_rate = int(processor.feature_extractor.sampling_rate)

    out: dict[str, str] = {}
    batch_size = _current_batch_size()
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        waves = []
        for r in batch:
            waves.append(_load_audio(r["audio"], sample_rate))
        inputs = processor.apply_transcription_request(audio=waves, language=language).to(
            model.device, model.dtype
        )
        prompt_len = int(inputs["input_ids"].shape[1])
        with torch.inference_mode():
            generated = model.generate(**inputs, max_new_tokens=1024, do_sample=False)
        for i, r in enumerate(batch):
            text = processor.decode(
                generated[i, prompt_len:],
                skip_special_tokens=True,
                return_format="transcription_only",
            )
            out[r["id"]] = text.strip()
    return out


def _load_audio(path: str, target_sr: int) -> Any:
    """``path`` as mono float32 in [-1, 1] at ``target_sr`` (scipy resample only if needed).

    soundfile decodes to float32 whatever the file's sample format, so int16 and float WAVs
    reach the judges on the same scale.
    """
    import numpy as np
    import soundfile as sf
    from scipy.signal import resample

    data, rate = sf.read(path, dtype="float32", always_2d=False)
    if data.ndim > 1:
        data = data.mean(axis=1)
    if int(rate) != target_sr:
        data = resample(data, int(round(len(data) * target_sr / int(rate)))).astype(np.float32)
    return data


def _has_cuda() -> bool:
    """True if a CUDA device is available to torch (kept optional so imports stay light)."""
    import torch

    return torch.cuda.is_available()


def _current_batch_size() -> int:
    """Return the CLI-configured batch size (module-level so helpers stay sync-free)."""
    return _BatchSize


def _normalizer() -> Any:
    """Return Whisper's ``BasicTextNormalizer`` (multilingual: keeps diacritics)."""
    from transformers.models.whisper.english_normalizer import BasicTextNormalizer

    return BasicTextNormalizer()


def _wer_errors(normalize: Any, reference: str, hypothesis: str) -> tuple[int, int]:
    """Return ``(errors, reference_length)`` for one row after multilingual normalisation.

    Both strings pass through Whisper's ``BasicTextNormalizer``; WER is word-level
    edit distance (substitutions + deletions + insertions) computed with jiwer's
    ``compute_measures`` over the normalised strings.
    """
    import jiwer

    ref = normalize(reference).strip()
    hyp = normalize(hypothesis).strip()
    if not ref:
        return 0, 0
    measures = jiwer.compute_measures(ref, hyp)
    errors = int(measures["substitutions"] + measures["deletions"] + measures["insertions"])
    reference_length = len(ref.split())
    return errors, reference_length


def main(argv: Sequence[str] | None = None) -> int:
    """Score speech for languages the upstream Seed-TTS scorer does not cover.

    Reads a JSONL manifest of utterances (``id``, ``audio``, ``text``, optional
    ``prompt_audio``) and reports ASR word error rate with either Whisper-large-v3 or
    Qwen3-ASR as judge, plus speaker-similarity (SIM) via the sibling ``seedtts_score.py``
    subprocess helpers when ``prompt_audio`` is provided.

    Text normalisation is transformers' ``BasicTextNormalizer`` (multilingual: keeps
    diacritics), and per-row word error counts come from ``jiwer.compute_measures``.
    Corpus WER is the **pooled** form  —  sum(errors) / sum(reference_length) × 100  —
    which differs from Seed-TTS's mean-of-rows averaging in ``average_wer.py``; the
    per-row numbers in ``all`` are kept in a per-file schema compatible with
    ``speech_p7/paired_bootstrap.py`` so pairwise comparisons remain possible.
    """
    global _BatchSize

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--asr", choices=["whisper", "qwen3asr", "qwen3omni"], required=True)
    ap.add_argument("--language", required=True, help='e.g. "vi" for Whisper/Qwen3-ASR')
    ap.add_argument("--whisper", help="local path to Whisper-large-v3")
    ap.add_argument(
        "--qwen3asr", help="local path to Qwen3-ASR (or Qwen3-Omni for --asr qwen3omni)"
    )
    ap.add_argument("--wavlm-ckpt", help="WavLM speaker-verification checkpoint")
    ap.add_argument("--seed-code", help="root of a seed-tts-eval checkout")
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-sim", action="store_true")
    args = ap.parse_args(argv)

    _BatchSize = int(args.batch_size)
    manifest_path = Path(args.manifest).resolve()
    out_path = Path(args.out).resolve()

    raw_lines = manifest_path.read_text(encoding="utf-8").splitlines()
    if args.limit is not None:
        raw_lines = raw_lines[: args.limit]
    rows = [_parse_row(line, i) for i, line in enumerate(raw_lines, 1) if line.strip()]
    rows_expected = len(rows)

    missing: list[str] = []
    present: list[dict[str, Any]] = []
    for r in rows:
        audio_path = Path(r["audio"])
        if not audio_path.exists():
            missing.append(r["audio"])
            continue
        present.append(r)

    normalize = _normalizer()
    judge_path: str = ""
    if args.asr == "whisper":
        if not args.whisper:
            sys.stderr.write("judge_multilingual: --whisper is required for --asr whisper\n")
            return 96
        judge_path = str(Path(args.whisper).resolve())
        hypotheses = _whisper_hypotheses(present, Path(judge_path), args.language)
    else:
        if not args.qwen3asr:
            sys.stderr.write(f"judge_multilingual: --qwen3asr is required for --asr {args.asr}\n")
            return 96
        judge_path = str(Path(args.qwen3asr).resolve())
        if args.asr == "qwen3omni":
            hypotheses = _qwen3omni_hypotheses(present, Path(judge_path), args.language)
        else:
            hypotheses = _qwen3asr_hypotheses(present, Path(judge_path), args.language)

    sims: dict[str, float] = {}
    want_sim = not args.no_sim and any(r["prompt_audio"] for r in present)
    if want_sim:
        if not args.wavlm_ckpt or not args.seed_code:
            sys.stderr.write(
                "judge_multilingual: --wavlm-ckpt and --seed-code are required "
                "when rows carry prompt_audio and --no-sim is not given\n"
            )
            return 96
        run_sim, _run, _module = _resolve_sim_helpers()
        workdir = out_path.parent / ".judge_multilingual_tmp"
        workdir.mkdir(parents=True, exist_ok=True)
        sim_rows = [
            {
                "wav": str(Path(r["audio"]).resolve()),
                "prompt": str(Path(r["prompt_audio"]).resolve()),
            }
            for r in present
            if r["prompt_audio"]
        ]
        if sim_rows:
            sims = {
                Path(k).resolve().as_posix(): v
                for k, v in run_sim(
                    sim_rows,
                    Path(args.seed_code).resolve(),
                    args.wavlm_ckpt,
                    args.python,
                    workdir,
                    args.device,
                ).items()
            }

    all_rows: list[dict[str, Any]] = []
    total_errors = 0
    total_reference_length = 0
    for r in present:
        hyp = hypotheses.get(r["id"], "")
        errors, reference_length = _wer_errors(normalize, r["text"], hyp)
        total_errors += errors
        total_reference_length += reference_length
        sim = sims.get(Path(r["audio"]).resolve().as_posix())
        all_rows.append(
            {
                "id": r["id"],
                "reference": normalize(r["text"]).strip(),
                "hypothesis": normalize(hyp).strip(),
                "errors": errors,
                "reference_length": reference_length,
                "wer": round(errors / reference_length, 6) if reference_length else 0.0,
                "sim": sim,
            }
        )

    corpus_wer = (
        round(total_errors / total_reference_length * 100, 3) if total_reference_length else 0.0
    )
    sim_vals = [x["sim"] for x in all_rows if x["sim"] is not None]
    corpus_sim = round(sum(sim_vals) / len(sim_vals) * 100, 3) if sim_vals else None
    for x in all_rows:
        if x["sim"] is not None:
            x["sim"] = round(x["sim"] * 100, 3)

    result = {
        "language": args.language,
        "asr": args.asr,
        "n_rows": rows_expected,
        "rows_scored": len(all_rows),
        "missing": missing,
        "wer": corpus_wer,
        "sim": corpus_sim,
        "judge": {"asr": judge_path, "normalizer": "whisper BasicTextNormalizer"},
        "all": all_rows,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    return 95 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
