#!/usr/bin/env python3
"""build_concat.py -- long-form recordings from a clip manifest, by joining consecutive clips.

Joins CONSECUTIVE clips of the same source recording (a call, an episode) with a fixed silence
gap into recordings between ``--min-s`` and ``--max-s`` seconds. Used two ways: as a long-form TEST
built from held-out clips (a test whose every word has a clip-level reference), and as long-form
TRAINING rows that teach a clip-trained model to transcribe minutes of audio in one pass.

Source recording and order come from ``source_id`` shaped ``<call>/<k>.wav`` (Earnings-22 test
clips), else from an id shaped ``<prefix>_<call>_<k>`` (Earnings-22 training clips); a row with
neither is refused. Within a call, clips are walked in order and a recording is closed when the next
clip would push it past ``--max-s``; a recording shorter than ``--min-s`` is dropped and counted.
Audio must be 16 kHz (refused otherwise). The reference is the clip answers joined in order.

Output: FS eval rows {"id": "<call>_c<n:03d>", "text", "answer", "audio", "duration", "source_id",
"clips"} to ``--out`` (JSON lines) and 16 kHz mono FLAC under ``--audio-dir``.
usage: build_concat.py --manifest CLIPS.jsonl --out OUT.jsonl --audio-dir DIR [--min-s 60]
                       [--max-s 300] [--gap-s 0.3] [--seed 0] [--limit-recordings N]

WHAT IS NOT CLAIMED: that consecutive clip ids are contiguous in time (gaps between clips are
replaced by ``--gap-s`` of silence), or that a concatenation sounds like a natural recording.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

SR = 16000
_SOURCE_RE = re.compile(r"^(?P<call>[^/]+)/(?P<k>\d+)\.\w+$")
_ID_RE = re.compile(r"^[^_]+_(?P<call>.+)_(?P<k>\d+)$")


def parse_call_order(row: dict[str, Any]) -> tuple[str, int]:
    """(call, order) for one clip row, from ``source_id`` or from its id; refuses neither."""
    match = _SOURCE_RE.match(str(row.get("source_id", "")))
    if match is None:
        match = _ID_RE.match(str(row.get("id", "")))
    if match is None:
        raise ValueError(
            f"row {row.get('id')!r}: neither source_id '<call>/<k>.wav' nor id '<prefix>_<call>_<k>'"
        )
    return match["call"], int(match["k"])


def plan_recordings(
    rows: list[dict[str, Any]], min_s: float = 60.0, max_s: float = 300.0, gap_s: float = 0.3
) -> tuple[list[list[dict[str, Any]]], int]:
    """Group one call's clips (already in order) into recordings; returns (plans, dropped_short)."""
    plans: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    length = 0.0
    dropped = 0
    for row in rows:
        dur = float(row["duration"])
        added = dur + (gap_s if current else 0.0)
        if current and length + added > max_s:
            if length >= min_s:
                plans.append(current)
            else:
                dropped += 1
            current, length, added = [], 0.0, dur
        current.append(row)
        length += added
    if current:
        if length >= min_s:
            plans.append(current)
        else:
            dropped += 1
    return plans, dropped


def _concat_audio(rows: list[dict[str, Any]], gap_s: float) -> np.ndarray:
    gap = np.zeros(int(gap_s * SR), dtype=np.float32)
    parts: list[np.ndarray] = []
    for i, row in enumerate(rows):
        x, sr = sf.read(row["audio"], dtype="float32", always_2d=True)
        if sr != SR:
            raise ValueError(f"{row['audio']}: {sr} Hz audio; build_concat needs 16 kHz")
        if i:
            parts.append(gap)
        parts.append(x.mean(axis=1))
    return np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--audio-dir", required=True)
    ap.add_argument("--min-s", type=float, default=60.0)
    ap.add_argument("--max-s", type=float, default=300.0)
    ap.add_argument("--gap-s", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit-recordings", type=int)
    args = ap.parse_args(argv)
    with open(args.manifest, encoding="utf-8") as f:
        clips = [json.loads(line) for line in f if line.strip()]
    calls: dict[str, list[tuple[int, dict[str, Any]]]] = {}
    for row in clips:
        call, k = parse_call_order(row)
        calls.setdefault(call, []).append((k, row))
    plans: list[tuple[str, list[dict[str, Any]]]] = []
    dropped = 0
    for call in sorted(calls):
        ordered = [row for _, row in sorted(calls[call], key=lambda t: t[0])]
        made, short = plan_recordings(ordered, args.min_s, args.max_s, args.gap_s)
        dropped += short
        plans += [(call, p) for p in made]
    if args.limit_recordings is not None:
        random.Random(args.seed).shuffle(plans)
        plans = sorted(plans[: args.limit_recordings], key=lambda t: (t[0], t[1][0]["id"]))
    Path(args.audio_dir).mkdir(parents=True, exist_ok=True)
    out_rows, per_call = [], Counter()
    for call, plan in plans:
        n = per_call[call]
        per_call[call] += 1
        rid = f"{call}_c{n:03d}"
        audio = _concat_audio(plan, args.gap_s)
        path = Path(args.audio_dir) / f"{rid}.flac"
        tmp = path.with_suffix(".tmp.flac")
        sf.write(tmp, audio, SR, format="FLAC", subtype="PCM_16")
        os.replace(tmp, path)
        out_rows.append(
            {
                "id": rid,
                "text": plan[0].get("text", "Transcribe this audio exactly."),
                "answer": " ".join(str(r["answer"]).strip() for r in plan),
                "audio": str(path),
                "duration": round(audio.size / SR, 3),
                "source_id": call,
                "clips": [r["id"] for r in plan],
            }
        )
    tmp_out = f"{args.out}.tmp"
    with open(tmp_out, "w", encoding="utf-8") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in out_rows)
    os.replace(tmp_out, args.out)
    summary = {
        "recordings": len(out_rows),
        "hours": round(sum(r["duration"] for r in out_rows) / 3600, 2),
        "clips_used": sum(len(r["clips"]) for r in out_rows),
        "dropped_short": dropped,
        "per_call": dict(per_call),
    }
    print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    sys.exit(main())
