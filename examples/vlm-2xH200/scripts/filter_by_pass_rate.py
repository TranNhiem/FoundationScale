#!/usr/bin/env python3
"""
DAPO-style difficulty filtering for GRPO RL data (multimodal multiple choice).

Why
---
GRPO normalises the rewards *inside* a question's sampled group.  If every
sample gets the same reward (the model always wins or always loses), the group
advantage is exactly zero and the update vanishes -- those rows only cost
compute.  Sampling ``--samples`` completions per question and keeping the rows
whose pass rate sits in a target band (``--min-rate`` .. ``--max-rate``) is the
cheap way to remove them (DAPO "dynamic sampling / difficulty filtering").

Input (one JSON object per line)::
    {"id": "q42",
     "conversations": [{"from": "human", "value": "…Choices A. B. C. D. … Answer: X"},
                       {"from": "gpt",   "value": "Answer: X"}],
     "image": "/abs/path/to/fig.png",
     "gold": "B"}

Method
------
The question is rendered as one chat message (image + text) and sampled
``--samples`` times with ``n`` in a single ``llm.chat`` call (the whole dataset
goes in as one batch).  Every sample is judged with the exact answer parser of
the companion MCQ benchmark script: an unparsable sample counts as a **wrong**
answer, so models that ramble push their own pass rate down.

Output
------
* ``--out-keep``   : the kept rows unchanged, plus a ``pass_rate`` field.
* ``--out-stats``  : CLI summary as JSON (histogram, kept/too-easy/too-hard,
                     unparsable fraction, per-row pass rates).
* stdout           : the same histogram / counters.

Example
-------
    python filter_by_pass_rate.py \\
        --model /models/Qwen2.5-VL-7B-Instruct \\
        --data  /rl/train.jsonl --out-keep /rl/train_filtered.jsonl \\
        --out-stats filter_stats.json --samples 8 \\
        --min-rate 0.125 --max-rate 0.875 --tp 2 --disable-thinking
"""

from __future__ import annotations

import argparse
import base64
import json
import re
import time
from pathlib import Path

from vllm import LLM, SamplingParams

# ---------------------------------------------------------------------------
# Thinking-mode tags.  The raw tag text is built by concatenation so that it
# never appears literally in this source file.
# ---------------------------------------------------------------------------
_THINK_OPEN = "<" + "think>"
_THINK_CLOSE = "<" + "/think>"
_THINK_BLOCK = re.compile(re.escape(_THINK_OPEN) + r".*?" + re.escape(_THINK_CLOSE), re.DOTALL)


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------
def sniff_mime(blob: bytes) -> str:
    """Guess the MIME type of an image from its magic bytes (JPEG fallback)."""
    if blob[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if blob[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if blob[:4] == b"RIFF" and blob[8:12] == b"WEBP":
        return "image/webp"
    if blob[:3] == b"GIF":
        return "image/gif"
    return "image/jpeg"


def image_path_to_data_url(path: str) -> str:
    """Turn the row's PNG path into a base64 ``data:`` URL for the chat API."""
    path = (path or "").strip()
    if not path:
        raise ValueError("row has an empty 'image' field")
    if path.startswith("data:"):  # already usable as-is
        return path
    raw = Path(path).read_bytes()  # absolute (or relative) image file path
    return f"data:{sniff_mime(raw)};base64,{base64.b64encode(raw).decode()}"


# ---------------------------------------------------------------------------
# Prompt / answer / parsing helpers
# ---------------------------------------------------------------------------
def strip_thinking(text: str) -> str:
    """Remove every ``<think>...`` reasoning block from the raw output."""
    text = _THINK_BLOCK.sub(" ", text)  # finished blocks
    if _THINK_OPEN in text:  # unterminated block -> drop the tail
        text = text.split(_THINK_OPEN, 1)[0]
    return text.strip()


def parse_option_letter(raw: str, letters: str) -> str | None:
    """Return the option letter the model committed to, else ``None``.

    Order of preference, because a model that reasons before answering writes
    ordinary English first ("A chart shows ..."), and the first standalone
    capital in that text is an article, not an answer:

    1. an explicit final answer: "Answer: C", "the answer is (C)", "**C**";
    2. a bare short reply such as "C", "(C)" or "C.";
    3. the LAST standalone option letter on the last non-empty line.
    """
    if not letters:
        return None
    text = strip_thinking(raw).strip()
    if not text:
        return None
    opts = re.escape("".join(sorted(letters)))
    explicit = [
        r"answer\s*(?:is|:)?\s*[:\-]?\s*\(?\**\s*([" + opts + r"])\b",
        r"\*\*\(?([" + opts + r"])\)?\*\*",
        r"(?:option|choice)\s+\(?([" + opts + r"])\b",
    ]
    for pattern in explicit:
        found = [f.upper() for f in re.findall(pattern, text, flags=re.IGNORECASE)]
        found = [f for f in found if f in letters]
        if found:
            return found[-1]
    bare = re.fullmatch(r"\(?([" + opts + r"])\)?[.:]?", text)
    if bare:
        return bare.group(1)
    last_line = [line for line in text.splitlines() if line.strip()][-1]
    standalone = r"(?<![0-9A-Za-z])([" + opts + r"])(?![0-9A-Za-z])"
    found = re.findall(standalone, last_line)
    return found[-1] if found else None


def normalize_letter(cell: str) -> str | None:
    """Map "B", "b.", "Answer: C" to a single upper letter (or ``None``)."""
    match = re.search(r"[A-Za-z]", cell or "")
    return match.group(0).upper() if match else None


def extract_option_letters(question: str) -> str:
    """Pick the option letters out of "Choices A. B. C. D."-style stems."""
    letters: set[str] = set()
    # line-initial option markers:  "A. the chart shows", "B) four circles"
    for match in re.finditer(r"(?m)^[ \t]*([A-Za-z])[.)][ \t]+\S", question):
        letters.add(match.group(1).upper())
    if len(letters) < 2:
        # inline listing:  "Choices A. B. C. D."  /  "A) … B) …"
        for match in re.finditer(r"\b([A-Z])[.)][ \t]", question):
            letters.add(match.group(1).upper())
    return "".join(sorted(letters))


def strip_image_marker(text: str) -> str:
    """Drop the standalone ``<image>`` placeholder left in the human turn."""
    out = re.sub(r"<\s*image\s*/?\s*>", "", text, flags=re.IGNORECASE)
    out = re.sub(r"[ \t]+\n", "\n", out)
    return out.strip()


# ---------------------------------------------------------------------------
# RL JSONL loading
# ---------------------------------------------------------------------------
def load_rows(data_path: str, limit: int | None) -> list[dict]:
    """Read the RL JSONL and validate the schema used by the RL pipeline."""
    rows: list[dict] = []
    with Path(data_path).open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)

            convs = row.get("conversations") or []
            human = [c for c in convs if (c or {}).get("from") == "human"]
            if not human:
                raise SystemExit(f"{data_path}:{line_no}: no 'human' turn in conversations")
            question = human[0].get("value", "")
            if not (row.get("image") or "").strip():
                raise SystemExit(f"{data_path}:{line_no}: missing 'image' path")
            gold = normalize_letter(row.get("gold", ""))
            if gold is None:
                raise SystemExit(f"{data_path}:{line_no}: unusable 'gold' field")

            letters = extract_option_letters(question)
            letters = "".join(sorted(set(letters) | {gold}))
            if len(letters) < 2:  # very defensive fallback
                letters = "ABCD" if gold in "ABCD" else gold

            rows.append({**row, "_question": question, "_gold": gold, "_letters": letters})

            if limit is not None and len(rows) >= limit:
                break
    return rows


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="filter_by_pass_rate.py",
        description="Keep GRPO rows whose sampled pass rate is inside [--min-rate, --max-rate].",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model", required=True, help="Local model dir (or hub id) of the VLM policy."
    )
    parser.add_argument(
        "--data", required=True, help="RL JSONL to measure (schema documented above)."
    )
    parser.add_argument(
        "--out-keep", required=True, help="JSONL with the kept rows (+ 'pass_rate')."
    )
    parser.add_argument("--out-stats", default=None, help="Optional JSON with the printed summary.")
    parser.add_argument(
        "--samples", type=int, default=8, help="Completions per question (the group size)."
    )
    parser.add_argument(
        "--temperature", type=float, default=1.0, help="Sampling temperature (GRPO value)."
    )
    parser.add_argument(
        "--max-num-seqs",
        type=int,
        default=256,
        help="concurrent sequences in vLLM (Qwen3.6-27B needs <= 844 on one H200)",
    )
    parser.add_argument("--max-tokens", type=int, default=400, help="Generation budget per sample.")
    parser.add_argument(
        "--min-rate", type=float, default=0.125, help="Rows below this pass rate are too hard."
    )
    parser.add_argument(
        "--max-rate", type=float, default=0.875, help="Rows above this pass rate are too easy."
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Only measure the first N rows (default: all)."
    )
    parser.add_argument("--tp", type=int, default=1, help="Tensor-parallel size.")
    parser.add_argument(
        "--max-model-len", type=int, default=8192, help="Maximum context length handed to vLLM."
    )
    parser.add_argument(
        "--disable-thinking",
        action="store_true",
        help="Slide in chat_template_kwargs={'enable_thinking': False} to llm.chat.",
    )
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(argv=None) -> None:
    args = parse_args(argv)
    if args.samples < 1:
        raise SystemExit("--samples must be >= 1")
    if not (0.0 <= args.min_rate <= args.max_rate <= 1.0):
        raise SystemExit("need 0 <= --min-rate <= --max-rate <= 1")

    rows = load_rows(args.data, args.limit)
    if not rows:
        raise SystemExit(f"{args.data}: no rows loaded")
    print(f"loaded {len(rows)} question(s) from {args.data}")

    # One conversation per question; the outer list is the batch -> ONE llm.chat call.
    conversations: list[list[dict]] = []
    for row in rows:
        content = [
            {"type": "image_url", "image_url": {"url": image_path_to_data_url(row["image"])}},
            {"type": "text", "text": strip_image_marker(row["_question"])},
        ]
        conversations.append([{"role": "user", "content": content}])

    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tp,
        max_model_len=args.max_model_len,
        limit_mm_per_prompt={"image": 1},
        trust_remote_code=True,
        # Qwen3.6's linear-attention layers need one cache block per running
        # sequence; vLLM's default of 1024 exceeded the 844 blocks available for
        # Qwen3.6-27B on one H200 and the engine refused to start.
        max_num_seqs=args.max_num_seqs,
    )
    sampling = SamplingParams(
        n=args.samples, temperature=args.temperature, max_tokens=args.max_tokens
    )

    started = time.perf_counter()
    if args.disable_thinking:
        outputs = llm.chat(
            messages=conversations,
            sampling_params=sampling,
            chat_template_kwargs={"enable_thinking": False},
        )
    else:
        outputs = llm.chat(messages=conversations, sampling_params=sampling)
    elapsed = time.perf_counter() - started

    # ---------------- scoring -------------------------------------------
    per_row: list[dict] = []
    kept, too_easy, too_hard = [], [], []
    unparsable_samples = 0
    total_samples = 0
    histogram: dict[str, int] = {f"{k}/{args.samples}": 0 for k in range(args.samples + 1)}

    for row, output in zip(rows, outputs, strict=True):
        correct = 0
        unparsable = 0
        for sample in output.outputs:  # exactly --samples texts
            pred = parse_option_letter(sample.text, row["_letters"])
            total_samples += 1
            unparsable += pred is None
            correct += pred == row["_gold"]
        unparsable_samples += unparsable

        pass_rate = correct / args.samples
        bucket = f"{correct}/{args.samples}"
        histogram[bucket] += 1

        record = {
            "id": row.get("id"),
            "pass_rate": round(pass_rate, 6),
            "correct": correct,
            "samples": args.samples,
            "unparsable": unparsable,
            "gold": row["_gold"],
        }
        per_row.append(record)

        if pass_rate < args.min_rate:  # GRPO advantage would be ~0 (all wrong)
            too_hard.append(row)
        elif pass_rate > args.max_rate:  # GRPO advantage would be ~0 (all right)
            too_easy.append(row)
        else:  # informative group -> learn from it
            row["pass_rate"] = round(pass_rate, 6)
            kept.append(row)

    # ---------------- outputs -------------------------------------------
    out_keep = Path(args.out_keep)
    out_keep.parent.mkdir(parents=True, exist_ok=True)
    with out_keep.open("w", encoding="utf-8") as handle:
        for row in kept:
            clean = {k: v for k, v in row.items() if not k.startswith("_")}
            handle.write(json.dumps(clean, ensure_ascii=False) + "\n")

    fraction_unparsable = unparsable_samples / total_samples if total_samples else 0.0
    summary = {
        "model": args.model,
        "data": args.data,
        "out_keep": str(out_keep),
        "samples": args.samples,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "min_rate": args.min_rate,
        "max_rate": args.max_rate,
        "limit": args.limit,
        "tensor_parallel_size": args.tp,
        "max_model_len": args.max_model_len,
        "disable_thinking": bool(args.disable_thinking),
        "num_rows": total_samples // args.samples,
        "kept": len(kept),
        "too_easy": len(too_easy),
        "too_hard": len(too_hard),
        "histogram": histogram,
        "total_samples": total_samples,
        "unparsable_samples": unparsable_samples,
        "unparsable_fraction": round(fraction_unparsable, 6),
        "generation_seconds": round(elapsed, 3),
        "rows": per_row,
    }

    if args.out_stats:
        out_stats = Path(args.out_stats)
        out_stats.parent.mkdir(parents=True, exist_ok=True)
        out_stats.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    # ---------------- summary -------------------------------------------
    n_rows = summary["num_rows"]

    def pct(num: int, den: int) -> str:
        return f"{100.0 * num / den:.2f}%" if den else "n/a"

    print("=" * 72)
    print(f" model       : {args.model}")
    print(f" data        : {args.data}")
    print(
        f" samples     : {args.samples}   temperature : {args.temperature}"
        f"   thinking : {'disabled' if args.disable_thinking else 'enabled'}"
    )
    print(f" band        : [{args.min_rate}, {args.max_rate}]   (GRPO learns only here)")
    print("-" * 72)
    print(f" pass-rate histogram  ({args.samples} samples per question):")
    width = max(len(str(n_rows)), 1)
    for k in range(args.samples + 1):
        key = f"{k}/{args.samples}"
        count = histogram[key]
        bar = "#" * min(count, 50)
        print(f"   {key:>6} : {str(count).rjust(width)}  {bar}")
    print("-" * 72)
    print(f" questions   : {n_rows}")
    print(f" kept        : {len(kept)}   ({pct(len(kept), n_rows)})")
    print(
        f" too easy    : {len(too_easy)}   ({pct(len(too_easy), n_rows)})"
        f"   (pass_rate > {args.max_rate})"
    )
    print(
        f" too hard    : {len(too_hard)}   ({pct(len(too_hard), n_rows)})"
        f"   (pass_rate < {args.min_rate})"
    )
    print(
        f" unparsable  : {unparsable_samples}/{total_samples} samples"
        f"  ({pct(unparsable_samples, total_samples)})"
    )
    print(f" generation  : {elapsed:.1f} s")
    print("-" * 72)
    print(f" keep  -> {out_keep}")
    if args.out_stats:
        print(f" stats -> {args.out_stats}")
    print("=" * 72)


if __name__ == "__main__":
    main()
