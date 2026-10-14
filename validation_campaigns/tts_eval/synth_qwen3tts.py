"""Synthesize Seed-TTS-eval rows with Qwen3-TTS Base (voice clone) in Qwen's published protocol.

Qwen3-TTS README, "Evaluation": bfloat16, ``max_new_tokens=2048``, every other sampling parameter
from the checkpoint's ``generate_config.json``, ``language="auto"`` on the Seed test set. Each row
clones its prompt wav + prompt text and speaks the target text; the output is ``<filename>.wav``
in --out, scored by seedtts_score.py. Runs in the qwen-tts worker env (tts-qwen), not the FS lane:
qwen-tts pins transformers 4.57.3.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def read_meta(path: Path) -> list[tuple[str, str, str, str]]:
    rows = []
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        parts = line.split("|")
        if len(parts) not in (4, 5):
            raise SystemExit(f"REFUSE (96): {path}:{n} has {len(parts)} fields, expected 4 or 5")
        rows.append((parts[0], parts[1], parts[2], parts[3]))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--testset-dir", required=True)
    ap.add_argument("--lang", choices=["en", "zh"], required=True)
    ap.add_argument("--list", default="meta.lst")
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    import soundfile as sf
    import torch
    from qwen_tts import Qwen3TTSModel

    lang_dir = Path(args.testset_dir) / args.lang
    rows = read_meta(lang_dir / args.list)[: args.limit]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    todo = [r for r in rows if not (out / f"{r[0]}.wav").exists()]  # resumable

    torch.manual_seed(args.seed)
    model = Qwen3TTSModel.from_pretrained(args.model, dtype=torch.bfloat16, device_map="cuda:0")
    start = time.time()
    for i in range(0, len(todo), args.batch_size):
        batch = todo[i : i + args.batch_size]
        wavs, sr = model.generate_voice_clone(
            text=[r[3] for r in batch],
            language=["auto"] * len(batch),
            ref_audio=[str(lang_dir / r[2]) for r in batch],
            ref_text=[r[1] for r in batch],
            max_new_tokens=args.max_new_tokens,
        )
        for r, wav in zip(batch, wavs, strict=True):
            sf.write(out / f"{r[0]}.wav", wav, sr)
    record = {
        "model": args.model,
        "rows": len(rows),
        "synthesized_now": len(todo),
        "seconds": round(time.time() - start, 1),
        "protocol": {
            "dtype": "bfloat16",
            "max_new_tokens": args.max_new_tokens,
            "language": "auto",
            "sampling": "checkpoint generate_config.json",
        },
        "seed": args.seed,
    }
    (out / "synth_record.json").write_text(json.dumps(record, indent=2))
    print(json.dumps(record))


if __name__ == "__main__":
    main()
