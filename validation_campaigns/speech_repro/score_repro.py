"""Score a reproduction run two ways: FoundationScale's normalizer and Whisper's English normalizer.

Model cards (and the Open ASR Leaderboard they usually cite) normalize with Whisper's
EnglishTextNormalizer; FoundationScale normalizes with speech_metrics.normalize_transcript. Rule 4
(docs/research/upstream_integration.md) asks for the gap AND the normalizer difference, so both are
reported. The Whisper normalizer is applied first and the same FS word alignment counts errors.
Usage: score_repro.py EVAL.json --whisper-normalizer path/to/normalizer.json --card-wer 1.61 [--tolerance 0.3]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from foundationscale.train.speech_metrics import corpus_error_rate


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("eval_json")
    ap.add_argument("--whisper-normalizer", required=True)
    ap.add_argument("--card-wer", type=float, required=True)
    ap.add_argument("--tolerance", type=float, default=0.3)
    args = ap.parse_args()
    from transformers.models.whisper.english_normalizer import EnglishTextNormalizer

    english = json.loads(Path(args.whisper_normalizer).read_text())
    whisper_norm = EnglishTextNormalizer(english)
    rows = json.loads(Path(args.eval_json).read_text())["all"]
    pairs = [(r["reference"], r["hypothesis"]) for r in rows]
    fs = corpus_error_rate(pairs, metric="wer", expected=len(pairs))
    wpairs = [(whisper_norm(a), whisper_norm(b)) for a, b in pairs]
    wpairs = [(a, b) for a, b in wpairs if a.strip()]
    wh = corpus_error_rate(wpairs, metric="wer", expected=len(wpairs))
    out = {"rows": len(pairs), "card_wer": args.card_wer, "tolerance": args.tolerance,
           "wer_fs_normalizer": round(100 * fs.rate(), 3),
           "wer_whisper_normalizer": round(100 * wh.rate(), 3)}
    out["gap_whisper_normalizer"] = round(out["wer_whisper_normalizer"] - args.card_wer, 3)
    out["reproduced"] = abs(out["gap_whisper_normalizer"]) <= args.tolerance
    print(json.dumps(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
