"""Full LibriSpeech dev-clean eval manifest (2703 utterances), strict: 16 kHz, 1-25 s, counted refusals."""

import json
from pathlib import Path

import soundfile as sf

root = Path("/home/hhri-ai/hh28144/datasets/speech/LibriSpeech/dev-clean")
out = Path("/home/hhri-ai/hh28144/datasets/speech/manifests/devclean_full.jsonl")
rows, refused = [], 0
for t in sorted(root.rglob("*.trans.txt")):
    for line in t.read_text().splitlines():
        uid, txt = line.split(" ", 1)
        audio = t.parent / f"{uid}.flac"
        info = sf.info(str(audio))
        if info.samplerate != 16000 or not 1.0 <= info.duration <= 25.0:
            refused += 1
            continue
        rows.append({"id": uid, "text": "Transcribe this audio exactly.", "answer": txt,
                     "audio": str(audio), "duration": round(info.duration, 3)})
out.write_text("".join(json.dumps(r) + "\n" for r in rows))
print("devclean_full", len(rows), "refused", refused)
