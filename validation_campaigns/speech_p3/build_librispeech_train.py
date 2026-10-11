"""Build train_clean_100_le25s.jsonl, the LibriSpeech manifest the speech campaigns train on.

Targets are SENTENCE-CASED, not LibriSpeech's raw upper case: whitespace collapsed, lowercased,
first character upper-cased, no punctuation added. Regenerating the GB200 manifest with this script
gives the same 28,539 rows field for field. Every LibriSpeech-trained result in
validation_campaigns/speech_p3..p7 used these targets. Raw upper-case targets are the leading
suspect for a Whisper-large-v3 regression seen on another stack (2.25 -> 3.32 on dev-clean, proper
nouns breaking into upper-case pieces such as "SWEET WATER"); unconfirmed. Qwen3-ASR's loss on
upper-case references is 1.61, against 0.014 on its own cased output (speech_q3asr/EVIDENCE.md).

Usage: build_librispeech_train.py LIBRISPEECH_SPLIT_DIR OUT.jsonl [--max-seconds 25] [--seed 0]
Rows: {"id", "text" (the prompt), "answer", "audio" (absolute path), "duration" (seconds, 3 dp)}.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import soundfile as sf

PROMPT = "Transcribe this audio exactly."


def sentence_case(text: str) -> str:
    return " ".join(text.split()).lower().capitalize()


def rows(split_dir: Path, max_seconds: float) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for trans in sorted(split_dir.rglob("*.trans.txt")):
        for line in trans.read_text().splitlines():
            utt_id, _, text = line.partition(" ")
            audio = (trans.parent / f"{utt_id}.flac").resolve()
            info = sf.info(str(audio))
            duration = round(info.frames / info.samplerate, 3)
            if duration > max_seconds:
                continue
            out.append(
                {
                    "id": utt_id,
                    "text": PROMPT,
                    "answer": sentence_case(text),
                    "audio": str(audio),
                    "duration": duration,
                }
            )
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("split_dir")
    ap.add_argument("out")
    ap.add_argument("--max-seconds", type=float, default=25.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    built = rows(Path(args.split_dir), args.max_seconds)
    if len({r["audio"] for r in built}) != len(built):
        raise SystemExit("REFUSE (96): duplicate audio paths")
    random.Random(args.seed).shuffle(built)
    with Path(args.out).open("w") as f:
        for r in built:
            f.write(json.dumps(r) + "\n")
    print(f"{len(built)} rows -> {args.out}")


if __name__ == "__main__":
    main()
