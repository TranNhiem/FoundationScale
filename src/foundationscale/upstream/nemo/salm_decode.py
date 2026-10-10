"""Held-out WER for a NeMo speechlm2 SALM model (Canary-Qwen), scored by FoundationScale.

PHASE 1.2b move target for ``validation_campaigns/speech_canary/eval_wer_salm.py``. Behaviour is
preserved verbatim -- same prompt (``"Transcribe the following: " + audio_locator_tag``), same
batched ``SALM.generate``, same JSON keys including ``all``, same refusal for non-16 kHz audio.

NeMo owns loading and generation (SALM's own chat API with its audio locator tag); FoundationScale
measures with the same normaliser and corpus WER as every other model. The prompt construction and
the row/hypothesis pairing are factored into :func:`chat_prompt` / :func:`prompts_for` /
:func:`pair_rows` (pure), so the JSON shape of the eval is testable in CI without a NeMo container.

Usage::

    python -m foundationscale.upstream.nemo.salm_decode --model nvidia/canary-qwen-2.5b \\
        --manifest JSONL --out JSON [--batch-size 16] [--max-new-tokens 128]

Exit codes are the campaign script's, unchanged: 0 OK, and a non-16 kHz clip is refused exactly as
``eval_wer_salm.py`` refuses it -- ``raise SystemExit("REFUSE (96): ...")`` (the message on stderr,
status 1). No resampling, the same strict rule as the HF eval.
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

__all__ = ["chat_prompt", "main", "pair_rows", "parse_args", "prompts_for"]


def chat_prompt(row: Mapping[str, Any], audio_locator_tag: str) -> list[dict[str, Any]]:
    """The one-message SALM chat prompt for one manifest row. Pure.

    Same instruction the training ``tags`` and ``SALM.generate`` were scored with:
    ``f"Transcribe the following: {audio_locator_tag}"`` followed by the row's clip. A changed
    prompt would decode the corpus as a different task and the WER would silently stop being a
    continuation of the earlier runs.
    """
    return [
        {
            "role": "user",
            "content": f"Transcribe the following: {audio_locator_tag}",
            "audio": [row["audio"]],
        }
    ]


def prompts_for(
    rows: Sequence[Mapping[str, Any]], audio_locator_tag: str
) -> list[list[dict[str, Any]]]:
    """The per-row prompts of one decode batch, in row order (generation order is row order, so the
    hypotheses below pair with rows by position). Pure."""
    return [chat_prompt(r, audio_locator_tag) for r in rows]


def pair_rows(rows: Sequence[Mapping[str, Any]], hyps: Sequence[str]) -> list[dict[str, Any]]:
    """Pair each manifest row with its decoded hypothesis into the ``all`` JSON records. Pure.

    STRICT length matching -- mirrors ``zip(..., strict=True)`` in the original -- so a missing or
    extra hypothesis raises instead of being silently zipped away. Every row must land in the
    output or be refused by name.

    Each record has four keys, in this order: ``id``, ``duration`` (the manifest row's own optional
    ``duration`` field -- NOT re-measured here, so a regression can be located by clip length),
    ``reference`` (the row's ``answer``), ``hypothesis``.
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

    Arg names identical to ``eval_wer_salm.py``'s: ``--model``, ``--manifest``, ``--out``,
    ``--batch-size`` (default 16), ``--max-new-tokens`` (default 128).
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-new-tokens", type=int, default=128)
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Decode a held-out manifest and write a WER/CER eval JSON. Return 0 on success; the non-16 kHz
    refusal is the campaign script's ``SystemExit``, preserved exactly (see module docstring)."""
    args = parse_args(argv)
    import soundfile as sf  # type: ignore[import-untyped]
    import torch
    from nemo.collections.speechlm2.models import SALM  # type: ignore[import-not-found]

    from foundationscale.train.speech_metrics import corpus_error_rate

    rows = [json.loads(line) for line in Path(args.manifest).read_text().splitlines()]
    for r in rows:  # same strict rule as the HF eval: no resampling
        if sf.info(r["audio"]).samplerate != 16000:
            raise SystemExit(f"REFUSE (96): {r['audio']} is not 16 kHz; resampling is not allowed")
    t0 = time.time()
    model = SALM.from_pretrained(args.model).bfloat16().eval().cuda()
    hyps: list[str] = []
    for i in range(0, len(rows), args.batch_size):
        chunk = rows[i : i + args.batch_size]
        prompts = prompts_for(chunk, model.audio_locator_tag)
        with torch.inference_mode():
            ids = model.generate(prompts=prompts, max_new_tokens=args.max_new_tokens)
        hyps.extend(model.tokenizer.ids_to_text(x.cpu()) for x in ids)
    per_row = pair_rows(rows, hyps)
    pairs = [(p["reference"], p["hypothesis"]) for p in per_row]
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
                # Every row, with its duration, so regressions can be located (e.g. by clip length).
                "all": per_row,
            },
            indent=2,
        )
    )
    print(f"n={len(rows)}\nWER={wer.rate()}\nCER={cer.rate()}\nclass={type(model).__name__}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
