#!/usr/bin/env python3
"""pause_cut_merge.py -- long-form inference for clip-trained ASR models: cut, decode, merge.

Models trained on clips cannot take a whole recording in one pass: Qwen3-ASR through
validation_campaigns/speech_repro/eval_qwen3_asr.py degrades on audio far longer than its training
clips, and Gemma-4 refuses more than 30 s of audio. This tool cuts every recording into pieces of at
most ``--max-s`` seconds, writes them as an FS eval manifest that any FS evaluator decodes, and then
merges the piece hypotheses back into one hypothesis per recording, in time order.

Cutting (``--mode pause``, the default) places each cut inside a PAUSE: frame log energy (25 ms
frames, 10 ms hop) is compared with the recording's median frame energy minus ``PAUSE_DB``; runs of
at least ``PAUSE_MIN_S`` below it are pauses, and the cut goes at the middle of the longest pause in
the second half of the current ``max_s`` window (so pieces stay long), then of the whole window,
then -- when the speech has no pause at all -- at ``max_s`` itself. A trailing piece shorter than
``--min-s`` joins the previous one when that keeps it within ``max_s``. ``--mode fixed`` cuts
uniform ``--chunk-s`` windows, the comparison every chunked long-form recipe starts from.

Inputs: JSON-lines manifests of whole recordings, either FS eval rows ({"id","audio","answer",...})
or NeMo long-form rows ({"audio_filepath","duration","text","id"?}); the recording id is ``id`` or
the audio file's stem. Audio must already be 16 kHz (refused otherwise; multi-channel is averaged).

Outputs: ``cut`` writes 16 kHz mono FLAC pieces to ``OUT_DIR/audio/<rec>__<k:04d>.flac`` and
``OUT_DIR/segments.jsonl`` (FS eval rows, id ``<rec>::<k:04d>``, empty answer). ``merge`` reads an
FS evaluator output for those segments and writes the recording-level eval JSON (the shape
eval_qwen3_asr.py writes: {"all": [{"id","reference","hypothesis"}]}) and, with ``--nemo-out``, the
NeMo-style prediction rows (original fields + ``pred_text``) that score_longform.py reads. A
recording with no decoded piece keeps an empty hypothesis and is counted: merge exits 5.

usage: pause_cut_merge.py cut --manifest M --out-dir D [--mode pause|fixed] [--max-s 28] [--chunk-s 30]
       pause_cut_merge.py merge --segments-eval E --manifest M --out EVAL_JSON [--nemo-out PRED_JSONL]

WHAT IS NOT CLAIMED: that pause cuts never split a word (a pause detector sees energy, not words);
that pause cutting beats fixed windows for every model (EVIDENCE.md measures it for one).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

SR = 16000
FRAME_LEN = 400  # 25 ms at 16 kHz
HOP_LEN = 160  # 10 ms at 16 kHz
PAUSE_DB = 15.0  # a frame is "quiet" when its energy is this far below the recording's median
PAUSE_MIN_S = 0.2  # a pause is at least this long
PROMPT = "Transcribe this audio exactly."


def frame_log_energy(x: np.ndarray, sr: int = SR) -> np.ndarray:
    """Per-frame 10*log10(mean square) over 25 ms frames with a 10 ms hop."""
    frame, hop = int(0.025 * sr), int(0.010 * sr)
    sq = np.square(np.asarray(x, dtype=np.float64))
    if sq.size == 0:
        return np.zeros(0, dtype=np.float64)
    if sq.size < frame:
        return np.asarray([10.0 * np.log10(sq.mean() + 1e-12)])
    csum = np.concatenate(([0.0], np.cumsum(sq)))
    starts = np.arange(1 + (sq.size - frame) // hop) * hop
    return 10.0 * np.log10((csum[starts + frame] - csum[starts]) / frame + 1e-12)


def _pauses(energy: np.ndarray, sr: int) -> list[tuple[int, int]]:
    """Runs of quiet frames at least PAUSE_MIN_S long, as (center_sample, length_frames)."""
    if energy.size == 0:
        return []
    quiet = energy < float(np.median(energy)) - PAUSE_DB
    hop, frame = int(0.010 * sr), int(0.025 * sr)
    min_frames = int(round(PAUSE_MIN_S / 0.010))
    runs: list[tuple[int, int]] = []
    i = 0
    while i < quiet.size:
        if not quiet[i]:
            i += 1
            continue
        j = i
        while j + 1 < quiet.size and quiet[j + 1]:
            j += 1
        if j - i + 1 >= min_frames:
            center = ((i + j) // 2) * hop + frame // 2
            runs.append((center, j - i + 1))
        i = j + 1
    return runs


def pause_bounds(x: np.ndarray, sr: int = SR, max_s: float = 28.0, min_s: float = 1.0) -> list[int]:
    """Sample boundaries [0, b1, ..., len(x)] with every piece <= max_s, cut inside pauses."""
    if max_s <= 0 or min_s < 0:
        raise ValueError(f"max_s must be > 0 and min_s >= 0, got {max_s}, {min_s}")
    n, max_len, min_len = len(x), int(max_s * sr), int(min_s * sr)
    pauses = _pauses(frame_log_energy(x, sr), sr)
    bounds = [0]
    while n - bounds[-1] > max_len:
        start = bounds[-1]
        end = start + max_len
        late = [p for p in pauses if start + max_len // 2 <= p[0] <= end]
        anywhere = [p for p in pauses if start + min_len < p[0] <= end]
        pick = late or anywhere
        if pick:
            # the longest pause; a later one wins a tie, so pieces stay long
            cut = max(pick, key=lambda p: (p[1], p[0]))[0]
        else:
            cut = end
        bounds.append(int(cut))
    bounds.append(n)
    if len(bounds) > 2 and bounds[-1] - bounds[-2] < min_len and bounds[-1] - bounds[-3] <= max_len:
        del bounds[-2]
    return bounds


def fixed_bounds(n_samples: int, sr: int = SR, chunk_s: float = 30.0) -> list[int]:
    """Uniform ``chunk_s`` windows covering all ``n_samples``."""
    if chunk_s <= 0:
        raise ValueError(f"chunk_s must be > 0, got {chunk_s}")
    step = int(chunk_s * sr)
    return list(range(0, n_samples, step)) + [n_samples] if n_samples else [0, 0]


def _rec_id(row: dict[str, Any]) -> str:
    audio = row.get("audio") or row.get("audio_filepath")
    if not audio:
        raise ValueError(f"row has neither 'audio' nor 'audio_filepath': {sorted(row)}")
    return str(row.get("id") or Path(str(audio)).stem)


def _load_16k(path: str) -> np.ndarray:
    x, sr = sf.read(path, dtype="float32", always_2d=True)
    if sr != SR:
        raise ValueError(f"{path}: {sr} Hz audio; pause_cut_merge needs 16 kHz (resample it first)")
    return x.mean(axis=1)


def cut_rows(
    rows: list[dict[str, Any]],
    out_dir: str,
    mode: str = "pause",
    max_s: float = 28.0,
    chunk_s: float = 30.0,
    prompt: str = PROMPT,
) -> list[dict[str, Any]]:
    """Cut every recording into pieces; write FLAC pieces; return FS eval rows for the pieces."""
    if mode not in ("pause", "fixed"):
        raise ValueError(f"mode must be 'pause' or 'fixed', got {mode!r}")
    audio_dir = Path(out_dir) / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    pieces: list[dict[str, Any]] = []
    for row in rows:
        rec = _rec_id(row)
        x = _load_16k(str(row.get("audio") or row.get("audio_filepath")))
        if mode == "pause":
            bounds = pause_bounds(x, SR, max_s=max_s)
        else:
            bounds = fixed_bounds(len(x), SR, chunk_s=chunk_s)
        safe = rec.replace("/", "_")
        for k in range(len(bounds) - 1):
            path = audio_dir / f"{safe}__{k:04d}.flac"
            tmp = path.with_suffix(".tmp.flac")
            sf.write(tmp, x[bounds[k] : bounds[k + 1]], SR, format="FLAC", subtype="PCM_16")
            os.replace(tmp, path)
            pieces.append(
                {
                    "id": f"{rec}::{k:04d}",
                    "text": prompt,
                    "answer": "",
                    "audio": str(path),
                    "duration": round((bounds[k + 1] - bounds[k]) / SR, 3),
                }
            )
    return pieces


def merge(seg_eval: dict[str, Any], rows: list[dict[str, Any]]) -> tuple[list[dict], list[dict], int]:
    """Join piece hypotheses per recording in numeric piece order.

    Returns (eval_rows, nemo_rows, missing): eval_rows in the eval_qwen3_asr.py shape, nemo_rows =
    the original manifest rows plus ``pred_text``, and the number of recordings with no decoded piece
    (kept with an empty hypothesis, never dropped).
    """
    by_rec: dict[str, dict[int, str]] = {}
    for item in seg_eval["all"]:
        rec, _, k = str(item["id"]).rpartition("::")
        if not rec or not k.isdigit():
            raise ValueError(f"segment id {item['id']!r} is not '<recording>::<k>'")
        by_rec.setdefault(rec, {})[int(k)] = str(item.get("hypothesis", "")).strip()
    eval_rows, nemo_rows, missing = [], [], 0
    for row in rows:
        rec = _rec_id(row)
        parts = by_rec.get(rec, {})
        if not parts:
            missing += 1
        hyp = " ".join(p for _, p in sorted(parts.items()) if p)
        reference = row.get("answer", row.get("text", ""))
        eval_rows.append({"id": rec, "reference": reference, "hypothesis": hyp})
        nemo_rows.append({**row, "pred_text": hyp})
    return eval_rows, nemo_rows, missing


def _read_jsonl(path: str) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _write_atomic(path: str, text: str) -> None:
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
    os.replace(tmp, path)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("cut")
    c.add_argument("--manifest", required=True)
    c.add_argument("--out-dir", required=True)
    c.add_argument("--mode", default="pause", choices=("pause", "fixed"))
    c.add_argument("--max-s", type=float, default=28.0)
    c.add_argument("--chunk-s", type=float, default=30.0)
    m = sub.add_parser("merge")
    m.add_argument("--segments-eval", required=True)
    m.add_argument("--manifest", required=True)
    m.add_argument("--out", required=True)
    m.add_argument("--nemo-out")
    args = ap.parse_args(argv)
    rows = _read_jsonl(args.manifest)
    if args.cmd == "cut":
        pieces = cut_rows(rows, args.out_dir, args.mode, args.max_s, args.chunk_s)
        seg_path = str(Path(args.out_dir) / "segments.jsonl")
        _write_atomic(seg_path, "".join(json.dumps(p, ensure_ascii=False) + "\n" for p in pieces))
        durs = [p["duration"] for p in pieces] or [0.0]
        summary = {"recordings": len(rows), "pieces": len(pieces), "max_s": max(durs),
                   "median_s": float(np.median(durs)), "mode": args.mode}
        print(json.dumps(summary))
        return 0
    with open(args.segments_eval, encoding="utf-8") as f:
        seg_eval = json.load(f)
    eval_rows, nemo_rows, missing = merge(seg_eval, rows)
    payload = {"strategy": "pause_cut_merge", "all": eval_rows, "missing": missing}
    _write_atomic(args.out, json.dumps(payload, ensure_ascii=False))
    if args.nemo_out:
        _write_atomic(args.nemo_out, "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in nemo_rows))
    print(json.dumps({"recordings": len(eval_rows), "missing": missing}))
    return 5 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
