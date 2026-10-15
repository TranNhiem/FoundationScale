"""Synthesize Seed-TTS-eval style rows with VoxCPM2 ("ultimate cloning": prompt wav + its text).

VoxCPM2's README publishes Seed-TTS results (en 1.84 WER / 75.3 SIM, zh 0.97 / 79.5) but not the
inference settings behind them, so this uses the README's documented cloning call and defaults
(cfg_value=2.0, inference_timesteps=10, the prompt clip also passed as reference_wav_path) and
records them; a gap to the published numbers is attributed to settings first. Runs in the voxcpm
worker env (tts-voxcpm). Output: ``<filename>.wav`` in --out, scored by seedtts_score.py.
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
    ap.add_argument("--lang", required=True)
    ap.add_argument("--list", default="meta.lst")
    ap.add_argument("--out", required=True)
    ap.add_argument("--cfg-value", type=float, default=2.0)
    ap.add_argument("--inference-timesteps", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lora-ckpt", default=None, help="VoxCPM LoRA checkpoint dir (T2b)")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    import soundfile as sf
    import torch
    from voxcpm import VoxCPM

    lang_dir = Path(args.testset_dir) / args.lang
    rows = read_meta(lang_dir / args.list)[: args.limit]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    todo = [r for r in rows if not (out / f"{r[0]}.wav").exists()]  # resumable

    torch.manual_seed(args.seed)
    lora_kwargs: dict = {}
    if args.lora_ckpt:
        # upstream's own LoRA loading (scripts/test_voxcpm_lora_infer.py): config from the
        # checkpoint's lora_config.json, weights from the checkpoint directory
        from voxcpm.model.voxcpm import LoRAConfig

        info = json.loads((Path(args.lora_ckpt) / "lora_config.json").read_text())
        lora_kwargs = {
            "lora_config": LoRAConfig(**info["lora_config"]),
            "lora_weights_path": str(args.lora_ckpt),
        }
    model = VoxCPM.from_pretrained(args.model, load_denoiser=False, **lora_kwargs)
    sr = model.tts_model.sample_rate
    start = time.time()
    for name, prompt_text, prompt_wav, target_text in todo:
        clip = str(lang_dir / prompt_wav)
        wav = model.generate(
            text=target_text,
            prompt_wav_path=clip,
            prompt_text=prompt_text,
            reference_wav_path=clip,
            cfg_value=args.cfg_value,
            inference_timesteps=args.inference_timesteps,
        )
        sf.write(out / f"{name}.wav", wav, sr)
    record = {
        "model": args.model,
        "lora_ckpt": args.lora_ckpt,
        "rows": len(rows),
        "synthesized_now": len(todo),
        "seconds": round(time.time() - start, 1),
        "sample_rate": sr,
        "protocol": {
            "mode": "prompt_wav + prompt_text, reference_wav = prompt clip",
            "cfg_value": args.cfg_value,
            "inference_timesteps": args.inference_timesteps,
            "published_settings": "UNVERIFIED (README states none for its Seed-TTS table)",
        },
        "seed": args.seed,
    }
    (out / "synth_record.json").write_text(json.dumps(record, indent=2))
    print(json.dumps(record))


if __name__ == "__main__":
    main()
