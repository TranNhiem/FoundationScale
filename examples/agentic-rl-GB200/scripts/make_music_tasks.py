"""Generate music-composition tasks (train + held-out eval JSONL) for the agentic music example.

    python make_music_tasks.py --out-dir data/ [--train 900] [--eval 100] [--seed 20261008]

Each task asks for one piece of a given instrumentation, key, tempo, metre and length, written as
ABC notation and submitted through the episode's tools. The prompts are generated here from a
fixed vocabulary with a seeded RNG, so the set is reproducible and carries no third-party text.
The measured runs in this example used a different 900/100 split of prompts with the same
shape; numbers on this generated set will differ.
"""

import argparse
import json
import random
from pathlib import Path

INSTRUMENTS = [
    ("solo piano", 1), ("solo cello", 1), ("solo violin", 1), ("flute and piano", 2),
    ("string quartet", 4), ("clarinet and guitar", 2), ("brass quintet", 5), ("solo guitar", 1),
]  # fmt: skip
KEYS = ["C major", "G major", "D major", "A minor", "E minor", "F major", "B-flat major", "D minor"]
TEMPI = [(62, "adagio"), (76, "andante"), (96, "moderato"), (120, "allegro"), (144, "vivace")]
METRES = ["4/4", "3/4", "6/8", "2/4"]
TEXTURES = [
    "a lyrical melody over a simple accompaniment",
    "one line outlining the harmony through wide arpeggios",
    "a call-and-response between two voices",
    "a dance with a steady bass and a syncopated tune",
]
SYSTEM = "You are a composer. Write the piece as ABC notation and submit it with the submit tool."


def task(rng: random.Random, uid: str) -> dict:
    inst, voices = rng.choice(INSTRUMENTS)
    bpm, word = rng.choice(TEMPI)
    bars = rng.choice([16, 24, 32, 48, 64])
    prompt = (
        f"Compose a piece for {inst} in {rng.choice(KEYS)}, {bpm} BPM {word}, "
        f"{rng.choice(METRES)}, about {bars} bars, in {voices} voice(s). "
        f"Texture: {rng.choice(TEXTURES)}. Write it in ABC notation and submit it."
    )
    return {
        "uid": uid,
        "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}],
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--train", type=int, default=900)
    p.add_argument("--eval", type=int, default=100)
    p.add_argument("--seed", type=int, default=20261008)
    a = p.parse_args()
    rng = random.Random(a.seed)
    a.out_dir.mkdir(parents=True, exist_ok=True)
    for name, n, prefix in (("music_train", a.train, "tr"), ("music_eval", a.eval, "ev")):
        path = a.out_dir / f"{name}.jsonl"
        with path.open("w") as fh:
            for i in range(n):
                fh.write(json.dumps(task(rng, f"{prefix}-{i:04d}")) + "\n")
        print(f"{path}: {n} tasks")


if __name__ == "__main__":
    main()
