#!/usr/bin/env python3
"""
Multiple-choice VLM benchmark (VLMEvalKit TSV format) powered by vLLM.

The TSV is tab-separated with a header row and columns are matched by *name*,
so extra columns (``bench``, ``l2_category``, ``image_path``, ...) are ignored.
Required columns : ``question``, ``answer``, ``image``
Optional columns : ``index`` (row id), ``hint`` (extra prompt line),
                   ``category`` (per-category metrics)
Options          : every column whose header is a single letter A-Z
                   (``A B C D``, optionally ``E`` ... ``Z``)

The ``image`` cell is a base64 payload as written by VLMEvalKit
(`/9j/...` = JPEG, `iVBORw0KGgo...` = PNG); plain file paths and ready-made
``data:`` URLs are accepted too.  Images travel to the model as data URLs.

All questions are scored with ONE ``llm.chat`` call (one conversation per row),
which keeps both H200s fully busy while still returning outputs in input order.

Example
-------
    python eval_mcq_vllm.py --model /models/Qwen2.5-VL-7B-Instruct \\
        --tsv /data/MMBench_dev.tsv --name MMBench-dev-tp2 --limit 200 --tp 2

JSON output contains the run configuration, overall accuracy, per-category
accuracy, the number of unparsable outputs and one entry per scored question.
"""

from __future__ import annotations

import argparse
import base64
import csv
import json
import re
import sys
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

# Anything that looks like a base64 body (letters, '+', '/', '=', whitespace).
_BASE64_RE = re.compile(r"[A-Za-z0-9+/=\s]+")


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


def image_cell_to_data_url(cell: str) -> str:
    """Convert one TSV image cell into a ``data:`` URL for the vLLM chat API."""
    cell = (cell or "").strip()
    if not cell:
        raise ValueError("empty image cell")
    if cell.startswith("data:"):  # already a data URL
        return cell
    if len(cell) >= 64 and _BASE64_RE.fullmatch(cell):  # base64 payload in the TSV
        payload = re.sub(r"\s+", "", cell)
        mime = sniff_mime(base64.b64decode(payload))
        return f"data:{mime};base64,{payload}"
    raw = Path(cell).read_bytes()  # path to an image file
    return f"data:{sniff_mime(raw)};base64,{base64.b64encode(raw).decode()}"


# ---------------------------------------------------------------------------
# Prompt / answer / parsing helpers
# ---------------------------------------------------------------------------
def build_prompt(question: str, options: dict[str, str], hint: str) -> str:
    """Render the multiple-choice question exactly as documented."""
    lines: list[str] = []
    hint = (hint or "").strip()
    if hint:
        lines.append("Hint: " + hint)  # 1. optional hint line
    lines.append((question or "").strip())  # 2. the question
    for letter, text in sorted(options.items()):
        lines.append(f"{letter}. {text.strip()}")  # 3. options "A. ..."
    lines.append("Answer with the option's letter from the given choices directly.")
    return "\n".join(lines)


def normalize_answer_cell(cell: str) -> str | None:
    """Map an ``answer`` cell ("A", "b.", "Answer: C") to a single upper letter."""
    match = re.search(r"[A-Za-z]", cell or "")
    return match.group(0).upper() if match else None


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


# ---------------------------------------------------------------------------
# TSV loading
# ---------------------------------------------------------------------------
def load_rows(tsv_path: str, limit: int | None):
    """Read the TSV and return ``(option_columns, rows)``.

    ``option_columns``: list of ``(letter, raw header name)`` for A-Z columns.
    ``rows``: dicts with the columns the scorer needs (plus parsed options).
    """
    with Path(tsv_path).open(newline="", encoding="utf-8") as handle:
        # Benchmark TSVs embed images as base64; some fields exceed csv's 128 KiB default.
        csv.field_size_limit(sys.maxsize)
        reader = csv.DictReader(handle, delimiter="\t")
        if not reader.fieldnames:
            raise SystemExit(f"{tsv_path}: empty file (no header row)")

        header = {name.strip(): name for name in reader.fieldnames if name}
        option_columns = sorted(
            (name.strip().upper(), header[name.strip()])
            for name in reader.fieldnames
            if name and re.fullmatch(r"[A-Za-z]", name.strip())
        )
        for required in ("question", "answer", "image"):
            if required not in header:
                raise SystemExit(f"{tsv_path}: missing required column '{required}'")
        if len(option_columns) < 2:
            raise SystemExit(
                f"{tsv_path}: need at least two single-letter option columns (A, B, ...)"
            )

        def cell(raw: dict, key: str) -> str:
            col = header.get(key)
            return (raw.get(col) or "").strip() if col else ""

        rows: list[dict] = []
        skipped = 0
        for position, raw in enumerate(reader):
            if limit is not None and len(rows) >= limit:
                break
            options = {letter: (raw.get(col) or "").strip() for letter, col in option_columns}
            options = {letter: text for letter, text in options.items() if text}
            answer = normalize_answer_cell(cell(raw, "answer"))
            if len(options) < 2 or answer not in options or not cell(raw, "image"):
                skipped += 1  # unusable row (no options / no image / no label)
                continue
            rows.append(
                {
                    "index": cell(raw, "index") or str(position),
                    "question": cell(raw, "question"),
                    "hint": cell(raw, "hint"),
                    "category": cell(raw, "category") or "uncategorised",
                    "image": cell(raw, "image"),
                    "options": options,
                    "answer": answer,
                }
            )
    return option_columns, rows, skipped


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="eval_mcq_vllm.py",
        description="Score a multiple-choice VLM benchmark (VLMEvalKit TSV) with vLLM on 2x H200.",
        epilog="Example: python eval_mcq_vllm.py --model /models/Qwen2.5-VL-7B-Instruct "
        "--tsv /data/MMBench_dev.tsv --name MMBench-dev --limit 200 --tp 2",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model", required=True, help="Local model directory (or hub id) of the VLM to evaluate."
    )
    parser.add_argument(
        "--tsv", required=True, help="VLMEvalKit TSV file with the questions and base64 images."
    )
    parser.add_argument(
        "--name",
        required=True,
        help="Run name stored in the JSON results and printed in the summary.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only score the first N usable questions (default: all).",
    )
    parser.add_argument(
        "--tp", type=int, default=2, help="Tensor-parallel size (the VM has 2x H200)."
    )
    parser.add_argument(
        "--max-model-len", type=int, default=16384, help="Maximum context length handed to vLLM."
    )
    parser.add_argument(
        "--out",
        default="results.json",
        help="JSON file that receives scores and per-row predictions.",
    )
    parser.add_argument(
        "--max-num-seqs",
        type=int,
        default=256,
        help="concurrent sequences in vLLM (Qwen3.6-27B needs <= 844 on one H200)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="generation budget; models that reason before answering need room to finish "
        "(measured: 32 tokens left 26%% of MMStar answers unparsable for gemma-4-12B-it)",
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

    _, rows, skipped = load_rows(args.tsv, args.limit)
    if not rows:
        raise SystemExit(f"{args.tsv}: no usable questions loaded")
    print(f"loaded {len(rows)} question(s) from {args.tsv} (skipped {skipped} unusable row(s))")

    # One conversation per question; the outer list is the batch -> ONE llm.chat call.
    conversations: list[list[dict]] = []
    for row in rows:
        prompt = build_prompt(row["question"], row["options"], row["hint"])
        content = [
            {"type": "image_url", "image_url": {"url": image_cell_to_data_url(row["image"])}},
            {"type": "text", "text": prompt},
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
    sampling = SamplingParams(temperature=0, max_tokens=args.max_tokens)

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
    predictions: list[dict] = []
    correct = 0
    unparsable = 0
    per_category: dict[str, list[int]] = {}  # category -> [correct, total]

    for row, output in zip(rows, outputs, strict=False):
        raw_text = output.outputs[0].text
        letters = "".join(sorted(row["options"]))
        pred = parse_option_letter(raw_text, letters)
        is_unparsable = pred is None
        is_correct = pred == row["answer"]
        correct += int(is_correct)
        unparsable += int(is_unparsable)

        bucket = per_category.setdefault(row["category"], [0, 0])
        bucket[0] += int(is_correct)
        bucket[1] += 1

        predictions.append(
            {
                "index": row["index"],
                "category": row["category"],
                "answer": row["answer"],
                "pred": pred,
                "correct": bool(is_correct),
                "unparsable": bool(is_unparsable),
                "output": raw_text,
                "prompt": build_prompt(row["question"], row["options"], row["hint"]),
            }
        )

    total = len(predictions)
    accuracy = correct / total if total else 0.0

    def percent(num: int, den: int) -> str:
        return f"{100.0 * num / den:.2f}%" if den else "n/a"

    results = {
        "name": args.name,
        "model": args.model,
        "tsv": args.tsv,
        "tensor_parallel_size": args.tp,
        "max_model_len": args.max_model_len,
        "disable_thinking": bool(args.disable_thinking),
        "num_questions": total,
        "num_correct": correct,
        "overall_accuracy": round(accuracy, 6),
        "num_unparsable": unparsable,
        "num_skipped": skipped,
        "generation_seconds": round(elapsed, 3),
        "category_accuracy": {
            cat: {"correct": hit, "total": num, "accuracy": round(hit / num, 6)}
            for cat, (hit, num) in sorted(per_category.items())
        },
        "predictions": predictions,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    # ---------------- summary -------------------------------------------
    print("=" * 72)
    print(f" run         : {args.name}")
    print(f" model       : {args.model}")
    print(f" tsv         : {args.tsv}")
    print(
        f" tensor para : {args.tp}   max-model-len : {args.max_model_len}"
        f"   thinking : {'disabled' if args.disable_thinking else 'enabled'}"
    )
    print("-" * 72)
    print(f" questions   : {total}   (skipped rows: {skipped})")
    print(f" correct     : {correct}")
    print(f" accuracy    : {percent(correct, total)}")
    print(f" unparsable  : {unparsable}")
    print(f" generation  : {elapsed:.1f} s")
    print(" per-category accuracy:")
    name_width = max((len(cat) for cat in per_category), default=8)
    for cat, (hit, num) in sorted(per_category.items()):
        print(f"   {cat.ljust(name_width)}  {percent(hit, num):>7}  ({hit}/{num})")
    print("-" * 72)
    print(f" results -> {out_path}")
    print("=" * 72)


if __name__ == "__main__":
    main()


# ============================================================================
# FILE: examples/vlm-2xH200/scripts/smoke_test_vllm.py
# ============================================================================
