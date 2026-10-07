"""Held-out WER for a NeMo ASR model (Canary), scored by foundationscale.train.speech_metrics.

The FoundationScale side of the NeMo lane: NeMo decodes (it owns the model), FoundationScale
measures with the same normaliser and corpus WER as every HF-native model, so the numbers compare.
Usage: eval_wer_nemo.py --model <hf id or .nemo> --manifest JSONL --out JSON [--batch-size 16]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from foundationscale.train.speech_metrics import corpus_error_rate, word_errors


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--pnc", default="yes", help="Canary punctuation/capitalisation prompt")
    args = ap.parse_args()
    import soundfile as sf
    import torch
    from nemo.collections.asr.models import ASRModel

    rows = [json.loads(line) for line in Path(args.manifest).read_text().splitlines()]
    for r in rows:  # same strict rule as the HF eval: no resampling
        if sf.info(r["audio"]).samplerate != 16000:
            raise SystemExit(f"REFUSE (96): {r['audio']} is not 16 kHz; resampling is not allowed")
    t0 = time.time()
    if args.model.endswith(".nemo"):
        model = ASRModel.restore_from(args.model, map_location="cuda")
    else:
        model = ASRModel.from_pretrained(args.model, map_location="cuda")
    model = model.to(torch.bfloat16).eval()
    out = model.transcribe(
        [r["audio"] for r in rows],
        batch_size=args.batch_size,
        source_lang="en",
        target_lang="en",
        pnc=args.pnc,
    )
    hyps = [h.text if hasattr(h, "text") else str(h) for h in out]
    pairs = [(r["answer"], h) for r, h in zip(rows, hyps, strict=True)]
    wer = corpus_error_rate(pairs, metric="wer", expected=len(rows))
    cer = corpus_error_rate(pairs, metric="cer", expected=len(rows))
    result = {
        "model": args.model,
        "class": type(model).__name__,
        "n_rows": len(rows),
        "wer": {**wer.as_manifest(), "rate": wer.rate()},
        "cer": {**cer.as_manifest(), "rate": cer.rate()},
        "elapsed_s": round(time.time() - t0, 1),
        "samples": [
            {
                "id": r["id"],
                "reference": r["answer"],
                "hypothesis": h,
                "wer": word_errors(r["answer"], h).rate(),
            }
            for r, h in list(zip(rows, hyps, strict=True))[:25]
        ],
        # Every row, with its duration, so regressions can be located (e.g. by clip length).
        "all": [
            {
                "id": r["id"],
                "duration": r.get("duration"),
                "reference": r["answer"],
                "hypothesis": h,
            }
            for r, h in zip(rows, hyps, strict=True)
        ],
    }
    Path(args.out).write_text(json.dumps(result, indent=2))
    print(f"n={len(rows)}\nWER={wer.rate()}\nCER={cer.rate()}\nclass={type(model).__name__}")


if __name__ == "__main__":
    main()
