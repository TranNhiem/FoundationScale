"""FLEURS-vi test (CC BY 4.0) as a Vietnamese cloning eval, Seed-TTS meta layout.

Each of the 857 test rows is a target sentence; its prompt is a DIFFERENT test utterance of the
same gender and a different sentence id, chosen deterministically (seed 0), so SIM measures
cloning, not self-reconstruction. FLEURS test is held out from all training
(vi_tts/build_voxcpm_vi.py uses train+dev only). <out>/vi/meta.lst:
id|prompt_text|prompt-wavs/<file>|target_text.
"""

from __future__ import annotations

import csv
import random
import sys
from pathlib import Path

H = Path("~").expanduser()
F = H / "datasets" / "speech" / "fleurs_vi" / "data" / "vi_vn"
out = sys.argv[1]
vi = Path(out) / "vi"
(vi / "prompt-wavs").mkdir(parents=True, exist_ok=True)
with (F / "test.tsv").open(encoding="utf-8", newline="") as handle:
    rows = list(csv.reader(handle, delimiter="\t", quoting=csv.QUOTE_NONE))
rng = random.Random(0)
lines = []
for row in rows:
    pool = [p for p in rows if p[6] == row[6] and p[0] != row[0]]
    prompt = rng.choice(pool)
    link = vi / "prompt-wavs" / prompt[1]
    if not link.exists():
        link.symlink_to(F / "audio" / "test" / prompt[1])
    lines.append(f"fleurs_vi_{row[1][:-4]}|{prompt[2]}|prompt-wavs/{prompt[1]}|{row[2]}")
with (vi / "meta.lst").open("w", encoding="utf-8") as handle:
    handle.write("\n".join(lines) + "\n")
print("rows", len(lines), "distinct prompts", len({item.split("|")[2] for item in lines}))
