"""AMI (ihm) TRAIN clips for a mixed Canary recipe: 1-25 s, 16 kHz mono, LibriSpeech-style text.

AMI train and test meetings are disjoint by design, so this does not leak the OOD AMI test clips.
"""

from __future__ import annotations

import io
import json
import os
import sys
import urllib.request
from pathlib import Path

import pyarrow.parquet as pq
import soundfile as sf

HOME = Path(os.environ.get("CLUSTER_HOME", str(Path.home())))
ROOT = HOME / "datasets/speech/ami_train"
BASE = "https://huggingface.co/datasets/edinburghcstr/ami/resolve/main/ihm/"
LIMIT = int(sys.argv[1]) if len(sys.argv) > 1 else 6000
rows = []
(ROOT / "flac").mkdir(parents=True, exist_ok=True)
for k in range(42):
    shard = f"train-{k:05d}-of-00042.parquet"
    local = ROOT / shard
    if not local.exists():
        urllib.request.urlretrieve(BASE + shard, local)  # noqa: S310
    t = pq.read_table(local)
    for i, rec in enumerate(t.to_pylist()):
        wave, sr = sf.read(io.BytesIO(rec["audio"]["bytes"]), dtype="float32", always_2d=True)
        wave = wave.mean(axis=1)
        dur = len(wave) / sr
        text = " ".join((rec.get("text") or "").split()).lower().capitalize()
        if sr != 16000 or not (1.0 <= dur <= 25.0) or not text:
            continue
        path = ROOT / "flac" / f"ami_train_{k:02d}_{i:05d}.flac"
        sf.write(path, wave, sr)
        rows.append(
            {
                "id": path.stem,
                "text": "Transcribe this audio exactly.",
                "answer": text,
                "audio": str(path),
                "duration": round(dur, 3),
            }
        )
        if len(rows) >= LIMIT:
            break
    if len(rows) >= LIMIT:
        break
man = HOME / "datasets/speech/manifests/ami_train6k.jsonl"
with man.open("w") as f:
    for r in rows:
        f.write(json.dumps(r) + "\n")
short = sum(r["duration"] < 2.0 for r in rows)
print(
    "AMI_TRAIN rows",
    len(rows),
    "hours",
    round(sum(r["duration"] for r in rows) / 3600, 2),
    "clips<2s",
    short,
    "| sample:",
    rows[0]["answer"][:60],
)
