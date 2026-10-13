#!/usr/bin/env python3
"""Convert HuggingFaceH4/ultrachat_200k parquet shards to ShareGPT-style rows.

Writes ``{"id", "conversations": [{"from": "human"|"gpt", "value"}]}`` per line,
the shape ``examples/vlm-2xH200/data/prepare_data.py text`` validates and the
FoundationScale conversation collator trains on. System turns are dropped and a
conversation with fewer than two turns is skipped; nothing is truncated here --
overlong rows are dropped at train time (FOUNDATIONSCALE_TRAIN_OVERLONG=drop).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

ROLE = {"user": "human", "assistant": "gpt"}


def convert(src: Path, pattern: str, out: Path, limit: int) -> int:
    paths = sorted(src.glob(pattern))
    if not paths:
        raise SystemExit(f"no parquet matches {pattern!r} under {src}")
    written = 0
    with out.open("w", encoding="utf-8") as fout:
        for path in paths:
            for row in pd.read_parquet(path).itertuples():
                turns = [
                    {"from": ROLE[m["role"]], "value": m["content"]}
                    for m in row.messages
                    if m["role"] in ROLE
                ]
                if len(turns) < 2:
                    continue
                fout.write(
                    json.dumps(
                        {"id": f"{row.prompt_id}_{written}", "conversations": turns},
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                written += 1
                if written >= limit:
                    return written
    return written


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--input-dir", required=True, help="local ultrachat_200k snapshot (has data/*.parquet)"
    )
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--train-rows", type=int, default=20000)
    ap.add_argument("--test-rows", type=int, default=1000)
    a = ap.parse_args()
    src, out = Path(a.input_dir), Path(a.out_dir)
    n_train = convert(
        src, "data/train_sft-*.parquet", out / "ultrachat_raw_train.jsonl", a.train_rows
    )
    n_test = convert(src, "data/test_sft-*.parquet", out / "ultrachat_raw_test.jsonl", a.test_rows)
    print(json.dumps({"train_rows": n_train, "test_rows": n_test}))


if __name__ == "__main__":
    main()
