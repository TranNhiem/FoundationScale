"""Build tool-using math tasks (train + held-out eval JSONL) for agentic RL with reward.kind="math".

    python make_math_tasks.py --out-dir data/math [--train-n 0] [--seed 0]

Downloads public datasets from Hugging Face (plain HTTPS, no token needed):

| split | dataset | license |
|---|---|---|
| train | agentica-org/DeepScaleR-Preview-Dataset (deepscaler.json, ~40k problems) | MIT |
| eval | HuggingFaceH4/MATH-500 (test.jsonl, 500 problems from the MATH test set) | MIT (MATH) |
| eval | opencompass/AIME2025 (aime2025-I/II.jsonl, 30 problems) | MIT |

Training problems whose text also appears in an eval set are dropped (whitespace- and
case-insensitive match), and the count is printed. Each task is one JSON line: a uid, the
opening messages (a system prompt naming the bash and submit tools, and the problem) and
metadata.answer, the reference the math reward checks the final answer against.
"""

import argparse
import json
import random
import re
import urllib.request
from pathlib import Path

HF = "https://huggingface.co/datasets"
SYSTEM = (
    "You are an expert mathematician with a Python interpreter. Use the bash tool to run Python "
    '(for example: python3 -c "print(2**10)") whenever computing or checking helps. Reason step '
    "by step, then give the final answer as \\boxed{...} and submit it with the submit tool."
)


def fetch(repo: str, path: str) -> bytes:
    with urllib.request.urlopen(f"{HF}/{repo}/resolve/main/{path}", timeout=120) as resp:
        return resp.read()


def jsonl(data: bytes) -> list[dict]:
    return [json.loads(line) for line in data.decode().splitlines() if line.strip()]


def problem_of(row: dict) -> str:
    return str(row.get("problem") or row.get("question") or "").strip()


def key(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def task(uid: str, row: dict, source: str) -> dict:
    return {
        "uid": uid,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": problem_of(row)},
        ],
        "metadata": {"answer": str(row["answer"]).strip(), "source": source},
    }


def write(path: Path, tasks: list[dict]) -> None:
    with path.open("w") as fh:
        for t in tasks:
            fh.write(json.dumps(t, ensure_ascii=False) + "\n")
    print(f"{path}: {len(tasks)} tasks")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--train-n", type=int, default=0, help="subsample the train set (0 = all)")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)

    math500 = jsonl(fetch("HuggingFaceH4/MATH-500", "test.jsonl"))
    aime = []
    for part in ("aime2025-I.jsonl", "aime2025-II.jsonl"):
        aime += jsonl(fetch("opencompass/AIME2025", part))
    train = json.loads(fetch("agentica-org/DeepScaleR-Preview-Dataset", "deepscaler.json"))

    held_out = {key(problem_of(r)) for r in math500 + aime}
    kept = [r for r in train if problem_of(r) and str(r.get("answer") or "").strip()]
    clean = [r for r in kept if key(problem_of(r)) not in held_out]
    print(f"train: {len(kept)} usable, {len(kept) - len(clean)} dropped as eval overlap")
    if a.train_n:
        clean = random.Random(a.seed).sample(clean, min(a.train_n, len(clean)))

    write(
        a.out_dir / "math_train.jsonl",
        [task(f"dsr-{i:05d}", r, "deepscaler") for i, r in enumerate(clean)],
    )
    write(
        a.out_dir / "math500_eval.jsonl",
        [task(f"m500-{i:03d}", r, "math500") for i, r in enumerate(math500)],
    )
    write(
        a.out_dir / "aime25_eval.jsonl",
        [task(f"aime25-{i:02d}", r, "aime2025") for i, r in enumerate(aime)],
    )


if __name__ == "__main__":
    main()
