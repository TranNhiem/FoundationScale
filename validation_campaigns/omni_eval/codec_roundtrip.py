"""SP-6 feasibility gate: is Qwen3-Omni's Code2Wav the public Qwen3-TTS-Tokenizer-12Hz codec?

Training the talker needs codec targets for new speech, and Qwen3-Omni ships no encoder. The public
Qwen3-TTS-Tokenizer-12Hz has the same quantizer layout (16 x 2048, 12.5 Hz at 24 kHz). This encodes
real recordings with that tokenizer and decodes the same codes twice: with the tokenizer's own
decoder and with Omni's Code2Wav (loaded alone from the checkpoint's ``code2wav.*`` weights).
If Omni's decode is as intelligible as the tokenizer's, the codebooks are shared and exact talker
targets exist. Two steps, two worker envs:

  encode       (tts-qwen env)    manifest -> <out>/codes/<id>.npy, <out>/tokdec/<id>.wav
  omni-decode  (hf-cand-519 env) <out>/codes -> <out>/omnidec/<id>.wav

Each step writes ``<out>/<arm>.judge.jsonl`` for judge_multilingual.py (``--no-sim``).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _rows(manifest: Path, limit: int) -> list[dict]:
    return [json.loads(x) for x in manifest.read_text(encoding="utf-8").splitlines() if x.strip()][
        :limit
    ]


def _judge_manifest(out: Path, arm: str, rows: list[dict]) -> None:
    with (out / f"{arm}.judge.jsonl").open("w", encoding="utf-8") as f:
        for r in rows:
            audio = r["audio"] if arm == "gt" else str(out / arm / f"{r['id']}.wav")
            f.write(
                json.dumps({"id": r["id"], "audio": audio, "text": r["text"]}, ensure_ascii=False)
                + "\n"
            )


def encode(args: argparse.Namespace) -> None:
    import numpy as np
    import soundfile as sf
    from qwen_tts import Qwen3TTSTokenizer

    rows = _rows(args.manifest, args.limit)
    (args.out / "codes").mkdir(parents=True, exist_ok=True)
    (args.out / "tokdec").mkdir(parents=True, exist_ok=True)
    tok = Qwen3TTSTokenizer.from_pretrained(args.tokenizer, device_map="cuda:0")
    for r in rows:
        enc = tok.encode([r["audio"]])
        codes = enc.audio_codes[0].cpu().numpy()  # (frames, 16)
        np.save(args.out / "codes" / f"{r['id']}.npy", codes)
        wavs, sr = tok.decode(enc)
        sf.write(args.out / "tokdec" / f"{r['id']}.wav", wavs[0], sr)
    _judge_manifest(args.out, "gt", rows)
    _judge_manifest(args.out, "tokdec", rows)
    print(
        json.dumps(
            {"step": "encode", "rows": len(rows), "frames_per_row_first": int(codes.shape[0])}
        )
    )


def omni_decode(args: argparse.Namespace) -> None:
    import numpy as np
    import soundfile as sf
    import torch
    from safetensors import safe_open
    from transformers import AutoConfig
    from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import Qwen3OmniMoeCode2Wav

    rows = _rows(args.manifest, args.limit)
    config = AutoConfig.from_pretrained(args.omni)
    model = Qwen3OmniMoeCode2Wav._from_config(config.code2wav_config)
    index = json.loads((Path(args.omni) / "model.safetensors.index.json").read_text())["weight_map"]
    state = {}
    for shard in sorted({v for k, v in index.items() if k.startswith("code2wav.")}):
        with safe_open(str(Path(args.omni) / shard), framework="pt") as f:
            for key in f.keys():  # noqa: SIM118 - safe_open is not a dict
                if key.startswith("code2wav."):
                    state[key.removeprefix("code2wav.")] = f.get_tensor(key)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise SystemExit(
            f"REFUSE (96): code2wav weights mismatch missing={missing[:5]} unexpected={unexpected[:5]}"
        )
    model.to("cuda", torch.float32).eval()
    (args.out / "omnidec").mkdir(parents=True, exist_ok=True)
    for r in rows:
        codes = torch.from_numpy(np.load(args.out / "codes" / f"{r['id']}.npy")).long()
        codes = codes.T.unsqueeze(0).to("cuda")  # (1, 16, frames)
        with torch.inference_mode():
            wav = model.chunked_decode(codes).reshape(-1).float().cpu().numpy()
        sf.write(args.out / "omnidec" / f"{r['id']}.wav", wav, 24000)
    _judge_manifest(args.out, "omnidec", rows)
    print(json.dumps({"step": "omni-decode", "rows": len(rows), "weights": len(state)}))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["encode", "omni-decode"])
    ap.add_argument("--manifest", required=True, type=Path, help="jsonl: id, audio, text")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--tokenizer", help="Qwen3-TTS-Tokenizer-12Hz dir (encode)")
    ap.add_argument("--omni", help="Qwen3-Omni checkpoint dir (omni-decode)")
    args = ap.parse_args()
    encode(args) if args.step == "encode" else omni_decode(args)


if __name__ == "__main__":
    main()
