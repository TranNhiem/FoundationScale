"""Held-out WER for a NeMo speechlm2 SALM model (Canary-Qwen), scored by FoundationScale.

NeMo owns loading and generation (SALM's own chat API with its audio locator tag); FoundationScale
measures with the same normaliser and corpus WER as every other model.
Usage: eval_wer_salm.py --model nvidia/canary-qwen-2.5b --manifest JSONL --out JSON
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from foundationscale.train.speech_metrics import corpus_error_rate


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-new-tokens", type=int, default=128)
    args = ap.parse_args()
    import soundfile as sf
    import torch
    from nemo.collections.speechlm2.models import SALM

    rows = [json.loads(line) for line in Path(args.manifest).read_text().splitlines()]
    for r in rows:
        if sf.info(r["audio"]).samplerate != 16000:
            raise SystemExit(f"REFUSE (96): {r['audio']} is not 16 kHz; resampling is not allowed")
    t0 = time.time()
    model = SALM.from_pretrained(args.model).bfloat16().eval().cuda()
    hyps: list[str] = []
    for i in range(0, len(rows), args.batch_size):
        chunk = rows[i : i + args.batch_size]
        prompts = [
            [
                {
                    "role": "user",
                    "content": f"Transcribe the following: {model.audio_locator_tag}",
                    "audio": [r["audio"]],
                }
            ]
            for r in chunk
        ]
        with torch.inference_mode():
            ids = model.generate(prompts=prompts, max_new_tokens=args.max_new_tokens)
        hyps.extend(model.tokenizer.ids_to_text(x.cpu()) for x in ids)
    pairs = [(r["answer"], h) for r, h in zip(rows, hyps, strict=True)]
    wer = corpus_error_rate(pairs, metric="wer", expected=len(rows))
    cer = corpus_error_rate(pairs, metric="cer", expected=len(rows))
    Path(args.out).write_text(
        json.dumps(
            {
                "model": args.model,
                "class": type(model).__name__,
                "n_rows": len(rows),
                "wer": {**wer.as_manifest(), "rate": wer.rate()},
                "cer": {**cer.as_manifest(), "rate": cer.rate()},
                "elapsed_s": round(time.time() - t0, 1),
                "all": [
                    {
                        "id": r["id"],
                        "duration": r.get("duration"),
                        "reference": r["answer"],
                        "hypothesis": h,
                    }
                    for r, h in zip(rows, hyps, strict=True)
                ],
            },
            indent=2,
        )
    )
    print(f"n={len(rows)}\nWER={wer.rate()}\nCER={cer.rate()}\nclass={type(model).__name__}")


if __name__ == "__main__":
    main()
