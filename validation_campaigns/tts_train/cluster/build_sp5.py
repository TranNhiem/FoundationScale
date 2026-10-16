"""SP-5 data: K LibriTTS-R (CC BY 4.0) train-clean-100 speakers for multi-speaker VoxCPM2 LoRA.

The K speakers with the most utterances; per speaker, HELD utterances held out (shuffled, seed 0)
and one fixed 6-12 s train clip as that speaker's cloning prompt. Writes, under <out>:
wav16/ (16 kHz, the AudioVAE encoder rate), train.jsonl / dev.jsonl (VoxCPM manifests),
testset/en/meta.lst (clone each held-out sentence from its speaker's prompt) and judge_rows.jsonl
(SIM against the real held-out recording). usage: build_sp5.py <libritts_r dir> <out> [K] [HELD]
"""

import io
import json
import random
import sys
from collections import defaultdict
from math import gcd
from pathlib import Path

import pyarrow.parquet as pq
import soundfile as sf
from scipy.signal import resample_poly

D, OUT = Path(sys.argv[1]), Path(sys.argv[2])
K = int(sys.argv[3]) if len(sys.argv) > 3 else 8
HELD = int(sys.argv[4]) if len(sys.argv) > 4 else 20
SR = 16000
(OUT / "wav16").mkdir(parents=True, exist_ok=True)
(OUT / "wav24").mkdir(parents=True, exist_ok=True)
(OUT / "testset/en").mkdir(parents=True, exist_ok=True)
files = sorted((D / "data/train.clean.100").glob("*.parquet"))
cols = pq.read_schema(files[0]).names
by_spk = defaultdict(list)
for f in files:
    for r in pq.read_table(f, columns=[c for c in cols if c != "audio"]).to_pylist():
        by_spk[str(r["speaker_id"])].append((f, r))
speakers = sorted(by_spk, key=lambda s: -len(by_spk[s]))[:K]
need = {r["id"] for s in speakers for _, r in by_spk[s]}
audio = {}
for f in sorted({f for s in speakers for f, _ in by_spk[s]}):
    t = pq.read_table(f, columns=["id", "audio"])
    for i, a in zip(t.column("id").to_pylist(), t.column("audio").to_pylist(), strict=True):
        if i in need:
            audio[i] = a["bytes"]

train, dev, meta, judge, summary = [], [], [], [], {}
for s in speakers:
    rows = []
    for _, r in by_spk[s]:
        data, sr = sf.read(io.BytesIO(audio[r["id"]]), dtype="float32")
        p24 = OUT / "wav24" / f"{r['id']}.wav"
        sf.write(p24, data, sr, subtype="PCM_16")
        g = gcd(sr, SR)
        p16 = OUT / "wav16" / f"{r['id']}.wav"
        sf.write(p16, resample_poly(data, SR // g, sr // g), SR, subtype="PCM_16")
        text = r["text_normalized"] if "text_normalized" in r else r["text_original"]
        rows.append(
            {"id": r["id"], "p16": str(p16), "p24": str(p24), "text": text, "s": len(data) / sr}
        )
    random.Random(0).shuffle(rows)
    held, tr = rows[:HELD], rows[HELD:]
    ref = max((r for r in tr if 6 <= r["s"] <= 12), key=lambda r: r["s"])
    for r in tr:
        train.append(
            {"audio": r["p16"], "text": r["text"], "duration": round(r["s"], 3), "dataset_id": 0}
        )
    for r in held:
        dev.append(
            {"audio": r["p16"], "text": r["text"], "duration": round(r["s"], 3), "dataset_id": 0}
        )
        meta.append(f"{r['id']}|{ref['text']}|{ref['p24']}|{r['text']}")
        judge.append({"id": r["id"], "text": r["text"], "prompt_audio": r["p24"], "speaker": s})
    summary[s] = {
        "train": len(tr),
        "train_min": round(sum(r["s"] for r in tr) / 60, 1),
        "held": len(held),
    }

(OUT / "train.jsonl").write_text("".join(json.dumps(r) + "\n" for r in train))
(OUT / "dev.jsonl").write_text("".join(json.dumps(r) + "\n" for r in dev))
(OUT / "testset/en/meta.lst").write_text("".join(m + "\n" for m in meta))
(OUT / "judge_rows.jsonl").write_text("".join(json.dumps(j) + "\n" for j in judge))
(OUT / "speakers.json").write_text(json.dumps(summary, indent=1))
print(json.dumps({"speakers": len(speakers), "train": len(train), "held": len(dev)}))
