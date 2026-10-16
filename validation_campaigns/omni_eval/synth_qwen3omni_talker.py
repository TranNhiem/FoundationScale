"""O2 re-scope probe: can Qwen3-Omni's talker speak a language? Read-aloud of test sentences.

Each target sentence of a Seed-TTS style meta.lst (``name|prompt_text|prompt_wav|target_text``) is
given to Qwen3-Omni as text with an instruction to read it aloud verbatim; the talker's 24 kHz
audio is written to ``<out>/<name>.wav`` and the thinker's text to ``thinker.jsonl`` (so a judge
error can be split into "the thinker changed the words" and "the talker mispronounced them").
``judge.jsonl`` is written for judge_multilingual.py (``--no-sim``: there is no reference voice).
Greedy thinker, card's default speaker; resumable.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

LANGUAGE_NAMES = {"en": "English", "zh": "Chinese", "vi": "Vietnamese"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--meta", required=True, type=Path)
    ap.add_argument("--lang", required=True)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--speaker", default="Ethan")
    ap.add_argument("--limit", type=int, default=200)
    args = ap.parse_args()

    import soundfile as sf
    import torch
    from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor

    rows = []
    for line in args.meta.read_text(encoding="utf-8").splitlines()[: args.limit]:
        parts = line.split("|")
        rows.append((parts[0], parts[3]))
    args.out.mkdir(parents=True, exist_ok=True)
    model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(args.model, dtype=torch.bfloat16)
    model.to("cuda")
    model.eval()
    processor = Qwen3OmniMoeProcessor.from_pretrained(args.model)
    name = LANGUAGE_NAMES.get(args.lang, args.lang)
    thinker_path = args.out / "thinker.jsonl"
    seen = set()
    if thinker_path.exists():
        seen = {json.loads(x)["id"] for x in thinker_path.read_text(encoding="utf-8").splitlines()}
    start = time.time()
    with thinker_path.open("a", encoding="utf-8") as log:
        for uid, sentence in rows:
            if uid in seen and (args.out / f"{uid}.wav").exists():
                continue
            instruction = (
                f"Read the following {name} sentence aloud, word for word, in {name}:\n{sentence}"
            )
            messages = [{"role": "user", "content": [{"type": "text", "text": instruction}]}]
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = processor(text=text, return_tensors="pt", padding=True).to(
                model.device, model.dtype
            )
            with torch.inference_mode():
                text_ids, audio = model.generate(
                    **inputs,
                    speaker=args.speaker,
                    thinker_return_dict_in_generate=True,
                    thinker_do_sample=False,
                    use_audio_in_video=False,
                )
            prompt_len = int(inputs["input_ids"].shape[1])
            ids = getattr(text_ids, "sequences", text_ids)  # tensor in transformers 5.x
            said = processor.batch_decode(ids[:, prompt_len:], skip_special_tokens=True)[0].strip()
            wav = audio.reshape(-1).float().cpu().numpy() if audio is not None else []
            sf.write(args.out / f"{uid}.wav", wav, 24000)
            log.write(
                json.dumps({"id": uid, "target": sentence, "thinker": said}, ensure_ascii=False)
                + "\n"
            )
            log.flush()
    with (args.out / "judge.jsonl").open("w", encoding="utf-8") as f:
        for uid, sentence in rows:
            f.write(
                json.dumps(
                    {"id": uid, "audio": str(args.out / f"{uid}.wav"), "text": sentence},
                    ensure_ascii=False,
                )
                + "\n"
            )
    print(
        json.dumps(
            {"rows": len(rows), "seconds": round(time.time() - start, 1), "speaker": args.speaker}
        )
    )


if __name__ == "__main__":
    main()
