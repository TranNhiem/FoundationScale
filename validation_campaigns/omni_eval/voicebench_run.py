#!/usr/bin/env python3
"""voicebench_run.py — VoiceBench objectively scored subsets: O0 (cascade) baseline, O1 to follow.

Wraps the upstream VoiceBench repo (its main.py loop shape and its evaluate.py scorer) instead of
copying it.  It answers, for one assistant arm and one subset, two questions a voice-assistant
deployment must justify with measured evidence:

  1. end-to-end answer quality — how high does this arm score on VoiceBench's objectively scored
     subsets (openbookqa, mmsu, ifeval, advbench, bbh), scored by upstream evaluate.py so the
     numbers are the upstream numbers and stay independently checkable;
  2. per-row latency split — ASR transcription seconds ∶ LLM time-to-response seconds.  A cascade
     (Qwen3-ASR-1.7B + a chat LLM) is only the right O1 topology if its scores hold while its
     split stays cheaper than an end-to-end omni model; those are the two numbers the split
     records.  The split is per-row (means in the summary JSON, raw per-row values in a sidecar
     file) because a mean alone cannot show the tail that drives perceived wait.

The generation loop mirrors upstream main.py's audio path exactly in record shape: one row of
non-audio fields plus "response", named <assistant>-<subset>-test-audio.jsonl, so upstream
evaluate.py scores the file unchanged.  Audio is read from LOCAL parquet (column "audio" is
struct{bytes, path}) because the campaign's eval data is a frozen local snapshot — no
load_dataset (network fetch is not reproducible) — and is decoded with soundfile, mixed down to
mono and resampled to 16 kHz (upstream casts Audio(sampling_rate=16_000)).  The handed-over
waveform is the dict {"array": float32 ndarray, "sampling_rate": 16000} that upstream assistants
receive.

Arms (registry, so O1 adds without changing the runner):
  * cascade (O0) — Qwen3-ASR-1.7B via the qwen3_asr API (AutoProcessor.apply_transcription_request,
    row-wise, language=None so its own LID runs, return_format="transcription_only") + chat LLM via
    AutoTokenizer/AutoModelForCausalLM, bfloat16, greedy, max_new_tokens 2048, with upstream
    naive.py's audio-path system prompt and the transcript as the user turn.
  * qwen3_omni (O1) — stub that raises NotImplementedError("O1") at construction.

Per-row fs_* metrics go to a sidecar JSONL, NOT into the scored file: evaluate.py forwards rows
verbatim (json.loads → list → evaluator.evaluate), so the only evidence of what an evaluator does
with an unknown key is in evaluator source that is not among the wrapped files (mcq/ifeval inspect
key sets such as target/key/kwargs).  Putting fs_* keys into the scored file would change the
input the score is computed on without proof that every one of the five evaluators tolerates the
superset; keeping the fed file byte-shaped as upstream produce makes the score comparable to any
upstream run.  The fs_* full rows are written to <stem>-test-audio.fs.jsonl alongside.

Honesty rule: every number is measured, nothing assumed.  rows_expected = parquet rows after
--limit; a row whose generation raises is recorded with response "" and counted in "failed" and is
never dropped from the scored file, so the score is over exactly the expected row set; any failed
row still scores and the run exits 95.  If upstream evaluate.py itself exits non-zero, the run is
refused (exit 96) with the stderr tail — it is better to fall back to score_phase.py with
thousands_fs_calibration.json than to publish a number a broken scorer produced.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import subprocess
import sys
import time
import traceback
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import soundfile
import torch

TARGET_SR = 16_000  # upstream voicebench casts Audio(sampling_rate=16_000)
MAX_TRANSCRIPTION_TOKENS = 1024  # Qwen3-ASR published eval setting
MAX_RESPONSE_TOKENS = 2048  # upstream base.generate_audio default
SYSTEM_PROMPT_AUDIO = (
    "You are a helpful assistant who tries to help answer the user's question. "
    "Please note that the user's query is transcribed from speech, and the transcription may "
    "contain errors."
)  # verbatim upstream src/models/naive.py audio-path system prompt

ASSISTANT_ARMS = ("cascade", "qwen3_omni")  # registry keys (kept in sync at construction time)
EVALUATOR_BY_SUBSET: dict[str, str] = {
    "openbookqa": "mcq",
    "mmsu": "mcq",
    "ifeval": "ifeval",
    "advbench": "harm",
    "bbh": "bbh",
}


def _load_base_class(vb_code: Path) -> type:
    """Upstream src/models/base.py's VoiceAssistant, loaded straight from its file.

    Importing it as ``src.models.base`` would run ``src/models/__init__.py``, which imports every
    upstream assistant and their dependencies (mini_omni needs ``lightning``, ...). base.py
    itself depends only on torch and time, so it is loaded alone, by path.
    """
    import importlib.util

    path = vb_code / "src" / "models" / "base.py"
    spec = importlib.util.spec_from_file_location("voicebench_base", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load VoiceBench base class from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    base = getattr(module, "VoiceAssistant", None)
    if base is None:
        raise RuntimeError(f"{path} does not expose VoiceAssistant")
    return base


def _load_bf16(factory: Any, name: str) -> Any:
    """Load *name* in bfloat16 across the transformers 4.x `torch_dtype` / 5.x `dtype` rename.

    The rename is a toolchain boundary this run straddles, not a measured result, so both names
    are probed rather than assumed; the first one the installed transformers accepts wins.
    """
    try:
        return factory.from_pretrained(name, dtype=torch.bfloat16)
    except (TypeError, ValueError):
        return factory.from_pretrained(name, torch_dtype=torch.bfloat16)


def _build_cascade(base: type, asr_model: str, llm: str, device: str) -> Any:
    """Build the O0 cascade assistant as a subclass of upstream's VoiceAssistant."""

    class CascadeAssistant(base):
        def __init__(
            self,
            asr_model: str,
            llm: str,
            device: str,
        ) -> None:
            from transformers import (
                AutoModelForCausalLM,
                AutoModelForMultimodalLM,
                AutoProcessor,
                AutoTokenizer,
            )

            self.device = torch.device(device)
            self.asr_processor = AutoProcessor.from_pretrained(asr_model)
            self.asr_model = _load_bf16(AutoModelForMultimodalLM, asr_model)
            self.asr_model.to(self.device).eval()
            self.llm_tokenizer = AutoTokenizer.from_pretrained(llm)
            self.llm_model = _load_bf16(AutoModelForCausalLM, llm)
            self.llm_model.to(self.device).eval()
            self.last_metrics: dict[str, Any] = {}

        def transcribe(
            self,
            array: np.ndarray,
        ) -> tuple[str, float]:
            """Return (row-wise transcript, transcription seconds) for one 16 kHz waveform."""
            started = time.perf_counter()
            inputs = self.asr_processor.apply_transcription_request(
                audio=[array],  # row-wise ([array], not a batch) so one row can be timed
                language=None,  # Qwen3-ASR runs its own language identification, as at eval time
            ).to(self.asr_model.device, self.asr_model.dtype)  # features must match bf16 weights
            prompt_len = int(inputs["input_ids"].shape[1])
            with torch.inference_mode():
                out = self.asr_model.generate(
                    **inputs,
                    max_new_tokens=MAX_TRANSCRIPTION_TOKENS,
                    do_sample=False,
                )
            transcript = self.asr_processor.decode(
                out[0, prompt_len:],
                skip_special_tokens=True,
                return_format="transcription_only",
            ).strip()
            return transcript, time.perf_counter() - started

        def respond(
            self,
            transcript: str,
            max_new_tokens: int = MAX_RESPONSE_TOKENS,
        ) -> tuple[str, float]:
            """Return (chat response, time-to-response seconds) for one transcript turn."""
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT_AUDIO},
                {"role": "user", "content": transcript},
            ]
            started = time.perf_counter()
            # transformers 5.x returns a BatchEncoding here; ask for it explicitly (ids + mask)
            enc = self.llm_tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True,
            ).to(self.llm_model.device)
            prompt_len = int(enc["input_ids"].shape[1])
            with torch.inference_mode():
                out = self.llm_model.generate(
                    **enc,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                )
            response = self.llm_tokenizer.decode(out[0, prompt_len:], skip_special_tokens=True)
            return response.strip(), time.perf_counter() - started

        @torch.no_grad()
        def generate_audio(
            self,
            audio: dict[str, Any],
            max_new_tokens: int = MAX_RESPONSE_TOKENS,
        ) -> str:
            transcript, asr_seconds = self.transcribe(audio["array"])
            # max_new_tokens is honoured: upstream generate_ttft times a call with max_new_tokens=1
            response, llm_seconds = self.respond(transcript, max_new_tokens)
            self.last_metrics = {
                "fs_transcript": transcript,
                "fs_asr_seconds": asr_seconds,
                "fs_llm_seconds": llm_seconds,
            }
            return response

    return CascadeAssistant(asr_model=asr_model, llm=llm, device=device)


def _build_qwen3_omni(
    base: type,
    asr_model: str,
    llm: str,
    device: str,
) -> Any:
    """Registry-time stub for the O1 arm; O1 replaces this body without touching the runner."""

    class Qwen3OmniAssistant(base):
        def __init__(
            self,
            asr_model: str,
            llm: str,
            device: str,
        ) -> None:
            del asr_model, llm, device
            raise NotImplementedError("O1 (Qwen3-Omni) is not yet enabled; O0 is the cascade arm.")

        @torch.no_grad()
        def generate_audio(
            self,
            audio: dict[str, Any],
            max_new_tokens: int = MAX_RESPONSE_TOKENS,
        ) -> str:
            del audio, max_new_tokens
            raise NotImplementedError("O1")

    return Qwen3OmniAssistant(asr_model=asr_model, llm=llm, device=device)


def build_registry(
    vb_code: Path,
    asr_model: str | None,
    llm: str | None,
    device: str,
) -> dict[str, Callable[[], Any]]:
    factories: dict[str, Callable[[], Any]] = {
        "cascade": lambda: _build_cascade(base, asr_model or "", llm or "", device),
        "qwen3_omni": lambda: _build_qwen3_omni(base, asr_model or "", llm or "", device),
    }
    if set(factories) != set(ASSISTANT_ARMS):
        raise RuntimeError("ASSISTANT_ARMS and the assistant registry disagree")
    base = _load_base_class(vb_code)
    return factories


def load_rows(
    data_dir: Path,
    subset: str,
    limit: int | None,
) -> list[dict[str, Any]]:
    """Load LOCAL VoiceBench parquet rows for *subset* (a frozen snapshot, not load_dataset)."""
    import pyarrow.parquet

    paths = sorted((data_dir / subset).glob("*.parquet"))
    if not paths:
        raise SystemExit(f"no parquet files under {data_dir / subset}")
    rows: list[dict[str, Any]] = []
    for path in paths:
        rows.extend(pyarrow.parquet.read_table(path).to_pylist())
    if limit is not None:
        rows = rows[:limit]
    return rows


def _resample_to_16k(
    wave: np.ndarray,
    sr: int,
) -> np.ndarray:
    """Resample mono float32 audio to 16 kHz (upstream Audio(sampling_rate=16_000)).

    Most fixture audio is 16 kHz hex-native, so this is normally a no-op; the conversion branch is
    a resample_poly filter and is documented as such (not identical to the upstream library's
    filter, which is a known and stated deviation).
    """
    if int(sr) == TARGET_SR:
        return wave
    if int(sr) <= 0:
        raise SystemExit(f"unreadable sample rate {sr}")
    from scipy.signal import resample_poly

    divisor = math.gcd(int(sr), TARGET_SR)
    return resample_poly(wave, TARGET_SR // divisor, int(sr) // divisor).astype(np.float32)


def decode_audio(
    audio_field: dict[str, Any],
) -> dict[str, Any]:
    """Decode the parquet audio struct to the dict shape upstream assistants receive."""
    raw = audio_field.get("bytes")
    if raw is None:
        raise SystemExit(f"audio struct without bytes: {sorted(audio_field)}")
    wave, sr = soundfile.read(io.BytesIO(raw), dtype="float32", always_2d=False)
    if wave.ndim == 2:  # upstream HF Audio decodes to mono
        wave = wave.mean(axis=1)
    array = _resample_to_16k(wave.astype(np.float32, copy=False), int(sr))
    return {"array": array, "sampling_rate": TARGET_SR}


def _json_default(
    value: object,
) -> str:
    return str(value)


def _mean(
    values: list[float],
) -> float:
    return round(float(sum(values) / len(values)), 4) if values else 0.0


def run_generation(
    assistant: Any,
    rows: list[dict[str, Any]],
    jsonl_path: Path,
    sidecar_path: Path,
) -> tuple[int, int]:
    """Run the upstream-shaped generation loop; return (rows_generated, failed).

    Resumable: every finished row is appended (and flushed) to ``<jsonl>.partial`` as it
    completes, so an interrupted run (a released GPU, a killed job) restarts where it stopped.
    Rows that succeeded are reused; rows that failed are retried. The final JSONL is written in
    row order, the shape upstream evaluate.py reads, and the progress file is then removed.
    """
    partial = jsonl_path.with_name(jsonl_path.name + ".partial")
    done: dict[int, dict[str, Any]] = {}
    if partial.exists():
        for line in partial.read_text(encoding="utf-8").splitlines():
            if line.strip():
                entry = json.loads(line)
                if "fs_error" not in entry["metrics"]:
                    done[int(entry["row"])] = entry
    with partial.open("a", encoding="utf-8") as progress:
        for index, item in enumerate(rows):
            if index in done:
                continue
            audio_field = item.get("audio") or {}
            tmp = {k: v for k, v in item.items() if k != "audio"}  # upstream main.py record shape
            try:
                if audio_field:
                    audio = decode_audio(audio_field)
                else:
                    audio = {"array": np.zeros(0, np.float32)}
                tmp["response"] = assistant.generate_audio(audio)
                metrics = dict(assistant.last_metrics)
            except Exception:
                tmp["response"] = ""
                metrics = {"fs_transcript": "", "fs_asr_seconds": 0.0, "fs_llm_seconds": 0.0}
                metrics["fs_error"] = traceback.format_exc()
            entry = {"row": index, "record": tmp, "metrics": metrics}
            progress.write(json.dumps(entry, default=_json_default) + "\n")
            progress.flush()
            done[index] = entry
    ordered = [done[i] for i in sorted(done)]
    failed = sum(1 for e in ordered if "fs_error" in e["metrics"])
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for entry in ordered:
            handle.write(json.dumps(entry["record"], default=_json_default) + "\n")
    with sidecar_path.open("w", encoding="utf-8") as handle:
        for entry in ordered:
            row = {"row": entry["row"], **entry["metrics"]}
            handle.write(json.dumps(row, default=_json_default) + "\n")
    partial.unlink()
    return len(ordered) - failed, failed


def run_scoring(
    vb_code: Path,
    eval_python: str,
    jsonl_path: Path,
    evaluator: str,
    eval_txt_path: Path,
) -> tuple[str, str | None]:
    """Run upstream evaluate.py as a subprocess; return (stdout, stderr tail or None)."""
    if eval_python and not Path(eval_python).exists():
        return "", f"--eval-python {eval_python} not found (fall back to score_phase.py)"
    if not (vb_code / "evaluate.py").is_file():
        return "", f"upstream evaluate.py not found in {vb_code}"
    cmd = [
        eval_python or sys.executable,
        str(vb_code / "evaluate.py"),
        "--src_file",
        str(jsonl_path),
        "--evaluator",
        evaluator,
    ]
    result = subprocess.run(cmd, cwd=str(vb_code), capture_output=True, text=True, check=False)
    eval_txt_path.parent.mkdir(parents=True, exist_ok=True)
    # upstream reports its score through loguru, which writes to stderr: keep both streams
    output = result.stdout + result.stderr
    eval_txt_path.write_text(output, encoding="utf-8")
    if result.returncode != 0:
        return output, f"evaluate.py exit {result.returncode}: tail {output[-2000:]}"
    return output, None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="VoiceBench O0/O1 runner (local parquet in, upstream evaluate.py scores)"
    )
    parser.add_argument("--assistant", required=True, choices=list(ASSISTANT_ARMS))
    parser.add_argument("--subset", required=True, choices=sorted(EVALUATOR_BY_SUBSET))
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--vb-code", required=True, type=Path)
    parser.add_argument("--eval-python", required=True, help="python that runs evaluate.py")
    parser.add_argument("--asr-model", dest="asr_model", default=None, help="Qwen3-ASR path")
    parser.add_argument("--llm", default=None, help="chat LLM path (cascade)")
    parser.add_argument("--out-dir", dest="out_dir", required=True, type=Path)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    evaluator = EVALUATOR_BY_SUBSET[args.subset]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{args.assistant}-{args.subset}"
    jsonl_path = args.out_dir / f"{stem}-test-audio.jsonl"
    sidecar_path = args.out_dir / f"{stem}-test-audio.fs.jsonl"
    eval_txt_path = args.out_dir / f"{stem}.eval.txt"
    summary_path = args.out_dir / f"{stem}.summary.json"

    if args.assistant == "cascade" and (not args.asr_model or not args.llm):
        raise SystemExit("cascade needs --asr-model and --llm")

    rows = load_rows(args.data_dir, args.subset, args.limit)
    rows_expected = len(rows)

    registry = build_registry(args.vb_code, args.asr_model, args.llm, args.device)
    assistant = registry[args.assistant]()
    started = time.perf_counter()
    rows_generated, failed = run_generation(assistant, rows, jsonl_path, sidecar_path)
    elapsed_s = time.perf_counter() - started

    metrics_rows = [
        json.loads(line) for line in sidecar_path.read_text(encoding="utf-8").splitlines() if line
    ]
    asr_seconds = [t for r in metrics_rows if (t := float(r.get("fs_asr_seconds", 0.0))) > 0]
    llm_seconds = [t for r in metrics_rows if (t := float(r.get("fs_llm_seconds", 0.0))) > 0]

    eval_stdout, scoring_error = run_scoring(
        args.vb_code, args.eval_python, jsonl_path, evaluator, eval_txt_path
    )
    summary: dict[str, Any] = {
        "assistant": args.assistant,
        "subset": args.subset,
        "rows_expected": rows_expected,
        "rows_generated": rows_generated,
        "failed": failed,
        "evaluator": evaluator,
        "eval_stdout": eval_stdout,
        "models": {"asr": args.asr_model, "llm": args.llm},
        "mean_llm_seconds": _mean(llm_seconds),
        "mean_asr_seconds": _mean(asr_seconds),
    }
    if scoring_error is not None:
        summary["scoring_error"] = scoring_error  # refused run keeps the evidence, not a score
        summary["elapsed_s"] = round(elapsed_s, 2)
        text = json.dumps(summary, indent=2, default=_json_default)
        summary_path.write_text(text, encoding="utf-8")
        print(f"REFUSED scoring: {scoring_error}", file=sys.stderr)
        raise SystemExit(96)

    summary["elapsed_s"] = round(elapsed_s, 2)
    summary_path.write_text(json.dumps(summary, indent=2, default=_json_default), encoding="utf-8")
    print(f"assistant={args.assistant} subset={args.subset}")
    print(f"rows_expected={summary['rows_expected']} rows_generated={summary['rows_generated']}")
    print(f"failed={summary['failed']}")
    print(f"evaluator={evaluator}")
    print(f"mean_asr_seconds={summary['mean_asr_seconds']}")
    print(f"mean_llm_seconds={summary['mean_llm_seconds']}")
    print(json.dumps({"eval_stdout_tail": eval_stdout[-600:]}))
    if failed:
        raise SystemExit(95)
    raise SystemExit(0)


if __name__ == "__main__":
    main()
