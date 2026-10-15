"""T2a eval synthesis: speak held-out sentences of the T2a speaker, base clone vs fine-tuned voice.

--mode clone: the Base model voice-clones from the fixed reference clip + its text (zero-shot).
--mode custom: the SFT checkpoint speaks as its trained speaker (``generate_custom_voice``).
Both use Qwen's README protocol (bf16, max_new_tokens 2048, checkpoint sampling defaults). Output:
<out>/<id>.wav and <out>/judge.jsonl for judge_multilingual.py, with the speaker's REAL recording of
the same sentence as prompt_audio, so SIM measures closeness to the actual voice.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--heldout", required=True)
    ap.add_argument("--mode", choices=["clone", "custom"], required=True)
    ap.add_argument("--speaker", default="speaker_test", help="SFT speaker_name (custom mode)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import soundfile as sf
    import torch
    from qwen_tts import Qwen3TTSModel

    rows = [json.loads(x) for x in Path(args.heldout).read_text().splitlines() if x.strip()]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    model = Qwen3TTSModel.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="cuda:0", attn_implementation="sdpa"
    )
    for r in rows:
        if args.mode == "clone":
            wavs, sr = model.generate_voice_clone(
                text=r["text"],
                language="English",
                ref_audio=r["ref_audio"],
                ref_text=r["ref_text"],
                max_new_tokens=2048,
            )
        else:
            wavs, sr = model.generate_custom_voice(
                text=r["text"], language="English", speaker=args.speaker, max_new_tokens=2048
            )
        sf.write(out / f"{r['id']}.wav", wavs[0], sr)
    with (out / "judge.jsonl").open("w") as f:
        for r in rows:
            row = {
                "id": r["id"],
                "audio": str(out / f"{r['id']}.wav"),
                "text": r["text"],
                "prompt_audio": r["audio"],
            }
            f.write(json.dumps(row) + "\n")
    print(json.dumps({"mode": args.mode, "rows": len(rows), "model": args.model}))


if __name__ == "__main__":
    main()
