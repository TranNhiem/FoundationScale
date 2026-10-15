"""T2a data: one LibriTTS-R (CC BY 4.0) train-clean-100 speaker for Qwen3-TTS single-speaker SFT.

Speaker = most total audio. 60 utterances held out (eval: real recordings to compare against); the
rest -> train_raw.jsonl in Qwen's prepare_data.py format {"audio" (24 kHz wav), "text",
"ref_audio"} with ONE fixed ref_audio on every row, as Qwen's README recommends (its trainer takes
the speaker embedding from the first batch). LibriTTS-R audio is native 24 kHz.
"""

import io
import json
import random
from collections import defaultdict
from pathlib import Path

import pyarrow.parquet as pq
import soundfile as sf

D = Path.home() / "datasets/speech/libritts_r"
OUT = D / "t2a"
(OUT / "wav").mkdir(parents=True, exist_ok=True)
by_spk = defaultdict(list)
files = sorted((D / "data/train.clean.100").glob("*.parquet"))
cols = pq.read_schema(files[0]).names
print("columns", cols)
for f in files:
    t = pq.read_table(f, columns=[c for c in cols if c != "audio"])
    for r in t.to_pylist():
        by_spk[str(r["speaker_id"])].append((f, r))


def dur(r):
    return float(r.get("duration") or 0) if isinstance(r, dict) else 0


spk = max(by_spk, key=lambda s: len(by_spk[s]))
items = by_spk[spk]
print("speaker", spk, "utterances", len(items))
need = {r["id"] for _, r in items}
audio = {}
for f in sorted({f for f, _ in items}):
    t = pq.read_table(f, columns=["id", "audio"])
    for i, a in zip(t.column("id").to_pylist(), t.column("audio").to_pylist(), strict=True):
        if i in need:
            audio[i] = a
rows = []
for _, r in items:
    data, sr = sf.read(io.BytesIO(audio[r["id"]]["bytes"]), dtype="float32")
    assert sr == 24000, (r["id"], sr)
    p = str(OUT / "wav" / f"{r['id']}.wav")
    sf.write(p, data, sr, subtype="PCM_16")
    rows.append(
        {
            "id": r["id"],
            "audio": p,
            "text": r["text_normalized"] if "text_normalized" in r else r["text_original"],
            "seconds": len(data) / sr,
        }
    )
random.Random(0).shuffle(rows)
held, train = rows[:60], rows[60:]
ref = max((r for r in train if 6 <= r["seconds"] <= 12), key=lambda r: r["seconds"])
with (OUT / "train_raw.jsonl").open("w") as o:
    for r in train:
        o.write(
            json.dumps({"audio": r["audio"], "text": r["text"], "ref_audio": ref["audio"]}) + "\n"
        )
with (OUT / "heldout.jsonl").open("w") as o:
    for r in held:
        o.write(
            json.dumps(
                {
                    "id": r["id"],
                    "audio": r["audio"],
                    "text": r["text"],
                    "ref_audio": ref["audio"],
                    "ref_text": ref["text"],
                }
            )
            + "\n"
        )
print(
    "train",
    len(train),
    round(sum(r["seconds"] for r in train) / 60, 1),
    "min; heldout",
    len(held),
    "; ref",
    ref["id"],
    round(ref["seconds"], 1),
    "s",
)
