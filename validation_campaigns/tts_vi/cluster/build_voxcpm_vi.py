"""Commercial-safe Vietnamese TTS training manifest for VoxCPM2 (T2b).

Sources (all CC BY 4.0): FLEURS-vi train+dev (cased, punctuated), InfoRe-25h (one paid reader),
FPT-FOSD (100 h); InfoRe and FOSD transcripts are lowercase without punctuation. Audio decoded to
16 kHz mono WAV (VoxCPM2's AudioVAE encoder rate). Rows: {"audio","text","duration","dataset_id",
"source","licence"}. A per-source dev slice is held out; FLEURS test is never used (it is eval).
"""

from __future__ import annotations

import csv
import io
import json
import random
from math import gcd
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
from scipy.signal import resample_poly

H = Path.home()
OUT = H / "datasets/speech/vi_tts/voxcpm"
(OUT / "wav").mkdir(parents=True, exist_ok=True)
SR = 16000
rows: list[dict] = []


def put(src, did, uid, data, sr, text):
    if data.ndim > 1:
        data = data.mean(axis=1)
    if sr != SR:
        g = gcd(sr, SR)
        data = resample_poly(data, SR // g, sr // g)
    data = np.clip(data, -1, 1).astype(np.float32)
    path = OUT / "wav" / f"{src}_{uid}.wav"
    if not path.exists():
        sf.write(str(path), data, SR, subtype="PCM_16")
    rows.append(
        {
            "audio": str(path),
            "text": " ".join(text.split()),
            "duration": round(len(data) / SR, 3),
            "dataset_id": did,
            "source": src,
            "licence": "CC-BY-4.0",
        }
    )


F = H / "datasets/speech/fleurs_vi/data/vi_vn"
for split in ("train", "dev"):
    with (F / f"{split}.tsv").open(encoding="utf-8") as handle:
        reader = csv.reader(handle, delimiter="\t", quoting=csv.QUOTE_NONE)
        for rec in reader:
            audio_path = F / "audio" / split / rec[1]
            data, sr = sf.read(audio_path, dtype="float32")
            put("fleurs", 0, rec[1][:-4], data, sr, rec[2])
for did, name in ((1, "infore1_25hours"), (2, "fpt_fosd")):
    count = 0
    for parquet_path in sorted((H / "datasets/speech/vi_tts" / name / "data").glob("*.parquet")):
        table = pq.read_table(parquet_path)
        audios = table.column("audio").to_pylist()
        transcriptions = table.column("transcription").to_pylist()
        for audio_entry, transcription in zip(audios, transcriptions, strict=True):
            data, sr = sf.read(io.BytesIO(audio_entry["bytes"]), dtype="float32")
            put(name, did, f"{count:06d}", data, sr, transcription)
            count += 1
audio_paths = [rec["audio"] for rec in rows]
assert len(audio_paths) == len(set(audio_paths)), "duplicate audio paths"
random.Random(0).shuffle(rows)
dev: dict[str, list[dict]] = {}
train: list[dict] = []
for rec in rows:
    if dev.setdefault(rec["source"], []).__len__() < 100 and rec["duration"] <= 15:
        dev[rec["source"]].append(rec)
    else:
        train.append(rec)
for name, part in (
    ("train", train),
    ("dev", [item for collection in dev.values() for item in collection]),
):
    with (OUT / f"{name}.jsonl").open("w", encoding="utf-8") as out_handle:
        for rec in part:
            out_handle.write(json.dumps(rec, ensure_ascii=False) + "\n")
hours = {
    source: round(sum(rec["duration"] for rec in train if rec["source"] == source) / 3600, 1)
    for source in dev
}
print(
    "train",
    len(train),
    "dev",
    sum(len(collection) for collection in dev.values()),
    "hours",
    hours,
    "max dur",
    max(rec["duration"] for rec in rows),
    "empty text",
    sum(1 for rec in rows if not rec["text"]),
)
