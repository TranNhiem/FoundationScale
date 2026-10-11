"""Full LibriSpeech split manifest for rule-4 reproduction: every utterance, no duration cap.

Model cards report WER on the WHOLE test split, so nothing is filtered by length here (the
training builders cap at 25-40 s; a reproduction must not). Audio must be 16 kHz mono (refused and
counted otherwise). Usage: build_librispeech.py SPLIT   (e.g. test-clean, test-other)
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import soundfile as sf

HOME = Path(os.environ.get("CLUSTER_HOME", str(Path.home())))


def main() -> None:
    split = sys.argv[1]
    root = HOME / "datasets/speech/LibriSpeech" / split
    out = HOME / "datasets/speech/manifests" / f"librispeech_{split.replace('-', '_')}.jsonl"
    rows, refused = [], {}
    for t in sorted(root.rglob("*.trans.txt")):
        for line in t.read_text().splitlines():
            uid, txt = line.split(" ", 1)
            audio = t.parent / f"{uid}.flac"
            info = sf.info(str(audio))
            reason = "not_16k" if info.samplerate != 16000 else "not_mono" if info.channels != 1 else None
            if reason:
                refused[reason] = refused.get(reason, 0) + 1
                continue
            rows.append({"id": uid, "text": "Transcribe this audio exactly.", "answer": txt,
                         "audio": str(audio), "duration": round(info.duration, 3)})
    out.write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(json.dumps({"manifest": str(out), "rows": len(rows), "refused": refused,
                      "max_seconds": max(r["duration"] for r in rows),
                      "hours": round(sum(r["duration"] for r in rows) / 3600, 2)}))


if __name__ == "__main__":
    main()
