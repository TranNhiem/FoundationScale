"""MiniMax TTS-Multilingual-Test-Set (CC BY-SA 4.0), Vietnamese split, laid out Seed-TTS style.

<out>/vi/meta.lst: id|prompt_text|prompt-wavs/<speaker>.wav|target_text (99 rows, 2 voices), the
MP3 prompts decoded to WAV. Also writes <out>/vi/judge_manifest_<synth_tag>.jsonl builders' inputs.
"""

from __future__ import annotations

import sys
from pathlib import Path

import soundfile as sf

M = Path("~/datasets/speech/tts_eval/minimax_multilingual").expanduser()
out = sys.argv[1]
vi = Path(out) / "vi"
(vi / "prompt-wavs").mkdir(parents=True, exist_ok=True)
with (M / "speaker" / "prompt_text.txt").open(encoding="utf-8") as f:
    prompts = dict(line.split("|", 1) for line in f.read().splitlines() if "|" in line)
spk_file = {}
for spk in ("vietnamese_female", "vietnamese_male"):
    (name,) = [p.name for p in sorted((M / "speaker" / "vietnamese" / spk).iterdir())]
    data, sr = sf.read(str(M / "speaker" / "vietnamese" / spk / name), dtype="float32")
    sf.write(str(vi / "prompt-wavs" / f"{spk}.wav"), data, sr)
    spk_file[spk] = (name, prompts[name].strip())
rows = []
with (M / "text" / "vietnamese.txt").open(encoding="utf-8") as f:
    for i, line in enumerate(f.read().splitlines()):
        if not line.strip():
            continue
        spk, text = line.split("|", 1)
        rows.append(f"vi_{spk}_{i:03d}|{spk_file[spk][1]}|prompt-wavs/{spk}.wav|{text.strip()}")
(vi / "meta.lst").write_text("\n".join(rows) + "\n", encoding="utf-8")
print("rows", len(rows), "speakers", {k: v[0] for k, v in spk_file.items()})
