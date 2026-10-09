"""Earnings-domain training manifest from sanchit-gandhi/earnings22_robust_split, test calls excluded.

The "robust split" divides each call's segments across train/validation/test, so its train split
contains every call of the Earnings-22 test set we evaluate on (hf-audio/open-asr-leaderboard,
6 calls). Training on it as-is would leak the eval. This builder drops every segment whose
source_id is a leaderboard-test call (counted as excluded_test_call), so the remaining training
calls share no recording with the eval.

Most source audio is not 16 kHz mono. Without --convert such rows are refused, as everywhere
else. With --convert they are resampled to 16 kHz (polyphase, anti-aliased) and downmixed (channel mean)
HERE, at build time. That is a declared data-prep step: every converted row is counted by its
original rate/channel count and marked in its manifest row. The training loader still never
resamples. Other rules: 1-25 s, non-empty transcript. Every refused row is counted by reason.
Shards are taken evenly across the split (they are ordered by call), so many calls are covered.
--split validation builds the checkpoint-selection dev set the same way (test calls excluded).
Usage: build_earnings_train.py [--split train|validation] [--shards 8] [--limit 6000] [--convert]
"""

from __future__ import annotations

import argparse
import io
import json
import os
from pathlib import Path

import pyarrow.parquet as pq
import soundfile as sf
from huggingface_hub import HfFileSystem, hf_hub_download

REPO = "sanchit-gandhi/earnings22_robust_split"
HOME = Path(os.environ.get("CLUSTER_HOME", str(Path.home())))
ROOT = HOME / "datasets/speech/earnings22_robust"
MAN = HOME / "datasets/speech/manifests"


def leaderboard_test_calls(fs: HfFileSystem) -> set[str]:
    calls: set[str] = set()
    for f in fs.glob("datasets/hf-audio/open-asr-leaderboard/earnings22/test-*.parquet"):
        with fs.open(f) as h:
            calls |= {i.split("/")[0] for i in pq.read_table(h, columns=["id"]).column("id").to_pylist()}
    return calls


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=("train", "validation"), default="train")
    ap.add_argument("--shards", type=int, default=8)
    ap.add_argument("--limit", type=int, default=6000)
    ap.add_argument("--convert", action="store_true")
    args = ap.parse_args()
    from math import gcd

    from scipy.signal import resample_poly

    fs = HfFileSystem(token=False)
    excluded = leaderboard_test_calls(fs)
    if not excluded:
        raise SystemExit("REFUSE (96): found no leaderboard test calls; cannot prove the split is leak-free")
    every = sorted(f.split("/")[-1] for f in fs.glob(f"datasets/{REPO}/data/{args.split}-*.parquet"))
    shards = [every[round(i * (len(every) - 1) / max(args.shards - 1, 1))] for i in range(args.shards)]
    per_shard = -(-args.limit // len(shards))  # ceil: spread the rows over the shards too
    (ROOT / "flac").mkdir(parents=True, exist_ok=True)
    rows, refused, converted, seen, written = [], {}, {}, 0, set()
    for shard in shards:
        local = hf_hub_download(REPO, f"data/{shard}", repo_type="dataset", local_dir=str(ROOT), token=False)
        kept_here = 0
        for r in pq.read_table(local).to_pylist():
            if kept_here == per_shard:
                break
            seen += 1
            note = None
            reason = None
            text = (r.get("sentence") or "").strip()
            if r["source_id"] in excluded:
                reason = "excluded_test_call"
            elif not text:
                reason = "empty_text"
            else:
                wave, sr = sf.read(io.BytesIO(r["audio"]["bytes"]), dtype="float32")
                if (sr != 16000 or wave.ndim != 1) and args.convert:
                    note = f"converted_from_{sr}Hz_{1 if wave.ndim == 1 else wave.shape[1]}ch"
                    if wave.ndim != 1:
                        wave = wave.mean(axis=1)
                    if sr != 16000:
                        g = gcd(sr, 16000)
                        wave = resample_poly(wave, 16000 // g, sr // g).astype("float32")
                        sr = 16000
                    converted[note] = converted.get(note, 0) + 1
                dur = len(wave) / sr
                if sr != 16000:
                    reason = "sample_rate_mismatch"
                elif wave.ndim != 1:
                    reason = "not_mono"
                elif not 1.0 <= dur <= 25.0:
                    reason = "duration_out_of_range"
            if reason:
                refused[reason] = refused.get(reason, 0) + 1
                continue
            # segment_id repeats across calls (it is a per-call index): a name built from it alone
            # let later rows overwrite earlier rows' audio, so 6,000 rows pointed at 820 files and
            # most transcripts trained against another segment's audio. Name by split + call +
            # segment, and refuse to write any path twice.
            path = ROOT / "flac" / f"{args.split}_{r['source_id']}_{r['segment_id']}.flac"
            if path in written:
                raise SystemExit(f"REFUSE (96): {path.name} would be written twice")
            written.add(path)
            sf.write(str(path), wave, sr)
            rows.append({"id": path.stem, "source_id": r["source_id"],
                         "text": "Transcribe this audio exactly.", "answer": text,
                         "audio": str(path), "duration": round(dur, 3), "conversion": note})
            kept_here += 1
            if len(rows) == args.limit:
                break
        if len(rows) == args.limit:
            break
    out = MAN / f"earnings_{args.split}{len(rows)}.jsonl"
    out.write_text("".join(json.dumps(x) + "\n" for x in rows))
    calls = {x["source_id"] for x in rows}
    assert not calls & excluded, "leak: a test call reached the manifest"
    print(json.dumps({"manifest": str(out), "rows_seen": seen, "rows_kept": len(rows),
                      "refused": refused, "converted": converted, "shards": shards, "train_calls": len(calls), "excluded_test_calls": sorted(excluded),
                      "hours": round(sum(x["duration"] for x in rows) / 3600, 2)}))


if __name__ == "__main__":
    main()
