"""Long Earnings training segments (30-40 s) built from consecutive clips of one call.

Motivation: a Canary fine-tune trained on clips of at most 25 s falls into repetition loops on 40 s
chunks of whole calls (speech_longform/EVIDENCE.md). This builder gives it long targets. It
takes the robust split's train segments of the 119 non-test calls (the 6 Earnings-22 test calls
are excluded by source_id, as in speech_p8), orders each call's segments by start_ts, and joins
consecutive ones until the next would pass --max-seconds. Each output is the joined audio plus
the joined transcripts, so audio and text stay aligned even where validation/test segments of
the same call sit between two train segments. Conversion to 16 kHz mono is declared and counted.
Every output path is written once (refused otherwise), named split_call_firstsegment_lastsegment.
Usage: build_earnings_long.py [--shards 8] [--limit 2000] [--min-seconds 30] [--max-seconds 40]
"""

from __future__ import annotations

import argparse
import io
import json
import os
from collections import defaultdict
from math import gcd
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
from huggingface_hub import HfFileSystem, hf_hub_download
from scipy.signal import resample_poly

REPO = "sanchit-gandhi/earnings22_robust_split"
HOME = Path(os.environ.get("CLUSTER_HOME", str(Path.home())))
ROOT = HOME / "datasets/speech/earnings22_robust"
MAN = HOME / "datasets/speech/manifests"


def to_16k_mono(raw: bytes) -> tuple[np.ndarray, str | None]:
    wave, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=True)
    note = None if (sr == 16000 and wave.shape[1] == 1) else f"converted_from_{sr}Hz_{wave.shape[1]}ch"
    wave = wave.mean(axis=1)
    if sr != 16000:
        g = gcd(sr, 16000)
        wave = resample_poly(wave, 16000 // g, sr // g).astype("float32")
    return wave, note


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", type=int, default=8)
    ap.add_argument("--limit", type=int, default=2000)
    ap.add_argument("--min-seconds", type=float, default=30.0)
    ap.add_argument("--max-seconds", type=float, default=40.0)
    args = ap.parse_args()
    fs = HfFileSystem(token=False)
    excluded: set[str] = set()
    for f in fs.glob("datasets/hf-audio/open-asr-leaderboard/earnings22/test-*.parquet"):
        with fs.open(f) as h:
            excluded |= {i.split("/")[0] for i in pq.read_table(h, columns=["id"]).column("id").to_pylist()}
    if len(excluded) != 6:
        raise SystemExit(f"REFUSE (96): expected 6 test calls to exclude, found {sorted(excluded)}")
    every = sorted(f.split("/")[-1] for f in fs.glob(f"datasets/{REPO}/data/train-*.parquet"))
    shards = [every[round(i * (len(every) - 1) / max(args.shards - 1, 1))] for i in range(args.shards)]
    by_call: dict[str, list[dict]] = defaultdict(list)
    counts = {"excluded_test_call": 0, "empty_text": 0}
    for shard in shards:
        local = hf_hub_download(REPO, f"data/{shard}", repo_type="dataset", local_dir=str(ROOT), token=False)
        for r in pq.read_table(local).to_pylist():
            if r["source_id"] in excluded:
                counts["excluded_test_call"] += 1
            elif not (r.get("sentence") or "").strip():
                counts["empty_text"] += 1
            else:
                by_call[r["source_id"]].append(r)
    (ROOT / "flac_long").mkdir(parents=True, exist_ok=True)
    rows, written, converted = [], set(), 0
    for call in sorted(by_call):
        segs = sorted(by_call[call], key=lambda r: float(r["start_ts"]))
        group: list[tuple[dict, np.ndarray]] = []
        seconds = 0.0

        def flush() -> None:
            nonlocal converted
            if len(group) < 2 or seconds < args.min_seconds:
                return
            first, last = group[0][0], group[-1][0]
            path = ROOT / "flac_long" / f"train_{call}_{first['segment_id']}_{last['segment_id']}.flac"
            if path in written:
                raise SystemExit(f"REFUSE (96): {path.name} would be written twice")
            written.add(path)
            wave = np.concatenate([w for _, w in group])
            sf.write(str(path), wave, 16000)
            rows.append({"id": path.stem, "source_id": call, "text": "Transcribe this audio exactly.",
                         "answer": " ".join(g["sentence"].strip() for g, _ in group),
                         "audio": str(path), "duration": round(len(wave) / 16000, 3),
                         "segments": len(group)})

        for r in segs:
            wave, note = to_16k_mono(r["audio"]["bytes"])
            converted += note is not None
            dur = len(wave) / 16000
            if seconds + dur > args.max_seconds:
                flush()
                group, seconds = [], 0.0
                if len(rows) >= args.limit:
                    break
            if dur > args.max_seconds:
                continue
            group.append((r, wave))
            seconds += dur
        flush()
        if len(rows) >= args.limit:
            break
    rows = rows[: args.limit]
    out = MAN / f"earnings_long{len(rows)}.jsonl"
    out.write_text("".join(json.dumps(x) + "\n" for x in rows))
    calls = {x["source_id"] for x in rows}
    assert not calls & excluded, "leak: a test call reached the manifest"
    print(json.dumps({"manifest": str(out), "rows": len(rows), "calls": len(calls),
                      "hours": round(sum(x["duration"] for x in rows) / 3600, 2),
                      "mean_seconds": round(sum(x["duration"] for x in rows) / max(1, len(rows)), 1),
                      "segments_per_row": round(sum(x["segments"] for x in rows) / max(1, len(rows)), 1),
                      "converted_segments": converted, **counts}))


if __name__ == "__main__":
    main()
