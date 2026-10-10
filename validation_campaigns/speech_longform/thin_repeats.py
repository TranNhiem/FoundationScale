"""Cap runs of one repeated word in training transcripts (audio untouched): a loop-attractor test.

Earnings transcripts repeat words 3-5x more often than LibriSpeech/AMI (speech_longform/
EVIDENCE.md), and a fine-tune that learned to emit repeats loops under greedy decoding. This
rewrites each `answer` so that a run of the same word (compared after FoundationScale's
normalizer, so "and- and- And" is one run) keeps at most --keep surface tokens. Every changed row
and every dropped token is counted and printed; nothing else is altered.
Usage: thin_repeats.py IN.jsonl OUT.jsonl [--keep 2]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from foundationscale.train.speech_metrics import normalize_transcript


def thin(text: str, keep: int) -> tuple[str, int]:
    out: list[str] = []
    last, run, dropped = None, 0, 0
    for token in text.split():
        norm = normalize_transcript(token)
        key = norm[0] if len(norm) == 1 else None
        if key is not None and key == last:
            run += 1
        else:
            last, run = key, 1
        if key is not None and run > keep:
            dropped += 1
            continue
        out.append(token)
    return " ".join(out), dropped


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--keep", type=int, default=2)
    args = ap.parse_args()
    rows = [json.loads(line) for line in Path(args.src).read_text().splitlines() if line.strip()]
    changed = dropped = 0
    for r in rows:
        r["answer"], d = thin(r["answer"], args.keep)
        changed += d > 0
        dropped += d
    Path(args.dst).write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(json.dumps({"rows": len(rows), "rows_changed": changed, "tokens_dropped": dropped, "keep": args.keep}))


if __name__ == "__main__":
    main()
