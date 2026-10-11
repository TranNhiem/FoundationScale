"""Held-out WER for a NeMo ASR model (Canary), scored by ``foundationscale.train.speech_metrics``.

The FoundationScale side of the NeMo lane: NeMo decodes (it owns the model), FoundationScale
measures with the same normaliser and corpus WER as every HF-native model, so the numbers compare.

PHASE 1.2a move target for ``validation_campaigns/speech_canary/eval_wer_nemo.py``. Behaviour is
preserved verbatim -- same args, same refusal for non-16 kHz audio, same output JSON keys
including ``all``. The per-row (row -> hypothesis) pairing is factored into :func:`pair_rows`
(pure) so the JSON shape of the eval is testable in CI without a NeMo container.

Usage::

    python -m foundationscale.upstream.nemo.decode --model <hf id or .nemo> \\
        --manifest JSONL --out JSON [--batch-size 16] [--pnc yes]

Exit code follows the run contract: 0 OK, 96 REFUSE (a non-16 kHz clip -- same strict rule as
the HF eval: no resampling).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

__all__ = ["main", "pair_rows", "parse_args"]


def pair_rows(rows: Sequence[Mapping[str, Any]], hyps: Sequence[str]) -> list[dict[str, Any]]:
    """Pair each manifest row with its decoded hypothesis into JSON-record form. Pure.

    STRICT length matching -- mirrors ``zip(..., strict=True)`` in the original -- so a missing
    or extra hypothesis raises instead of being silently zipped away. Every row must land in the
    output or be refused by name.

    Each record has four keys, in this order: ``id``, ``duration`` (the manifest row's own
    optional ``duration`` field -- NOT re-measured here, so a regression can be located by clip
    length), ``reference`` (the row's ``answer``), ``hypothesis``. Those four keys are the shape
    of the ``all`` entry in the eval JSON; the ``samples`` entry derives from these and adds
    ``wer`` (drops ``duration``, preserving the original's key order: ``id``, ``reference``,
    ``hypothesis``, ``wer``).
    """
    if len(rows) != len(hyps):
        raise ValueError(
            f"pair_rows: {len(rows)} rows but {len(hyps)} hypotheses (strict pairing required)"
        )
    return [
        {
            "id": r["id"],
            "duration": r.get("duration"),
            "reference": r["answer"],
            "hypothesis": h,
        }
        for r, h in zip(rows, hyps, strict=True)
    ]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the CLI. Pure (argparse-only).

    Arg names identical to ``eval_wer_nemo.py``'s: ``--model`` (hf id or ``.nemo``),
    ``--manifest``, ``--out``, ``--batch-size`` (default 16), ``--pnc`` (default ``yes`` -- the
    Canary punctuation/capitalisation prompt).
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--pnc", default="yes", help="Canary punctuation/capitalisation prompt")
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Decode a held-out manifest and write a WER/CER eval JSON. Return 0 OK, 96 REFUSE."""
    args = parse_args(argv)
    import soundfile as sf  # type: ignore[import-untyped]
    import torch
    from nemo.collections.asr.models import ASRModel  # type: ignore[import-not-found]

    from foundationscale.train.speech_metrics import corpus_error_rate, word_errors

    rows = [json.loads(line) for line in Path(args.manifest).read_text().splitlines()]
    for r in rows:  # same strict rule as the HF eval: no resampling
        if sf.info(r["audio"]).samplerate != 16000:
            print(
                f"REFUSE (96): {r['audio']} is not 16 kHz; resampling is not allowed",
                file=sys.stderr,
            )
            return 96
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
    per_row = pair_rows(rows, hyps)
    pairs = [(p["reference"], p["hypothesis"]) for p in per_row]
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
                "id": p["id"],
                "reference": p["reference"],
                "hypothesis": p["hypothesis"],
                "wer": word_errors(p["reference"], p["hypothesis"]).rate(),
            }
            for p in per_row[:25]
        ],
        # Every row, with its duration, so regressions can be located (e.g. by clip length).
        "all": per_row,
    }
    Path(args.out).write_text(json.dumps(result, indent=2))
    print(f"n={len(rows)}\nWER={wer.rate()}\nCER={cer.rate()}\nclass={type(model).__name__}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
