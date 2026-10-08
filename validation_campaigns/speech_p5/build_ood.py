"""Build up-to-300-utterance OOD eval manifests (1-25 s, 16 kHz) from hf-audio/open-asr-leaderboard.

Usage: build_ood.py [voxpopuli] [ami] [earnings22]   (default: all three)

Rows keep the published transcript and use the same JSONL shape as the training manifests
(text = prompt, answer = transcript, audio = flac path). Clips outside 1-25 s, not at 16 kHz, or
with an empty transcript are skipped (the eval refuses to resample or truncate).
"""

from __future__ import annotations

import io
import json
import os
import sys
import urllib.request
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import soundfile as sf

BASE = "https://huggingface.co/datasets/hf-audio/open-asr-leaderboard/resolve/main/"
HOME = Path(os.environ.get("CLUSTER_HOME", str(Path.home())))
ROOT = HOME / "datasets/speech/ood"
MAN = HOME / "datasets/speech/manifests"
SHARDS = {
    "voxpopuli": ["voxpopuli/test-00002-of-00004.parquet"],
    "ami": ["ami/test-00004-of-00015.parquet"],
    "earnings22": [
        "earnings22/test-00003-of-00005.parquet",
        "earnings22/test-00002-of-00005.parquet",
    ],
}
# OOD_LIMIT raises the cap for a tighter WER estimate; extra shards follow the default one, so the
# first 300 rows (and their flac files) are identical to the default build.
LIMIT = int(os.environ.get("OOD_LIMIT", "300"))
EXTRA_SHARDS = {"ami": ["ami/test-00005-of-00015.parquet", "ami/test-00006-of-00015.parquet"]}


def load_table(shards: list[str]) -> pa.Table:
    tables = []
    for shard in shards:
        local = ROOT / shard
        local.parent.mkdir(parents=True, exist_ok=True)
        if not local.exists():
            urllib.request.urlretrieve(BASE + shard, local)  # noqa: S310 - fixed https URL
        tables.append(pq.read_table(local))
    return pa.concat_tables(tables)


def build(name: str) -> None:
    table = load_table(SHARDS[name] + (EXTRA_SHARDS.get(name, []) if LIMIT > 300 else []))
    text_col = next(c for c in ("text", "transcription", "sentence") if c in table.column_names)
    out_dir = ROOT / name / "flac"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, rec in enumerate(table.to_pylist()):
        wave, sr = sf.read(io.BytesIO(rec["audio"]["bytes"]), dtype="float32", always_2d=True)
        # Mono, like the training manifests: some sets (Earnings-22) ship stereo, and NeMo's
        # lhotse loader refuses multi-channel input (FoundationScale's loader mixes down).
        wave = wave.mean(axis=1)
        dur = len(wave) / sr
        text = (rec[text_col] or "").strip()
        if sr != 16000 or not (1.0 <= dur <= 25.0) or not text:
            continue
        path = out_dir / f"{name}_{i:05d}.flac"
        sf.write(path, wave, sr)
        rows.append(
            {
                "id": f"{name}_{i:05d}",
                "text": "Transcribe this audio exactly.",
                "answer": text,
                "audio": str(path),
                "duration": round(dur, 3),
            }
        )
        if len(rows) == LIMIT:
            break
    with (MAN / f"ood_{name}{LIMIT}.jsonl").open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    hours = round(sum(r["duration"] for r in rows) / 3600, 2)
    print("OOD", name, "rows", len(rows), "hours", hours)


if __name__ == "__main__":
    for set_name in sys.argv[1:] or list(SHARDS):
        build(set_name)
