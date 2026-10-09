#!/usr/bin/env python3
"""prepare_rl_data.py — build small, tutorial-friendly datasets for RL fine-tuning.

This single script holds two little ETL jobs, one per subcommand:

``preference``
    Download the public preference dataset ``trl-lib/ultrafeedback_binarized``
    and rewrite its first ``--rows`` rows into a compact DPO-ready JSONL file
    with one record per line::

        {"prompt": <user text>, "chosen": <good assistant reply>,
         "rejected": <bad assistant reply>}

    The original ``chosen`` / ``rejected`` columns are chat transcripts
    (lists of ``{"role", "content"}`` messages): the prompt is the user
    message, while ``chosen`` / ``rejected`` keep only the *last* assistant
    answer of each transcript.  Rows in which any of the three fields is empty
    are skipped and reported.

``scienceqa``
    Download ``derek-thomas/ScienceQA``, keep only **multimodal** questions
    (an image plus at least two answer choices), export every image as a PNG
    and write one multimodal conversational record per saved image::

        {"id": "scienceqa-<i>",
         "conversations": [{"from": "human", "value": <question + choices +
              answer-format instruction>},
                           {"from": "gpt", "value": "Answer: <gold letter>"}],
         "image": <absolute path of the PNG>,
         "gold": "<letter>"}

    The gold answer column of ScienceQA is a *choice index*; it is converted to
    the corresponding letter ("ABCDE..."[index]). ``<i>`` is the zero-based
    index of the emitted row, so ``scienceqa-<i>.png`` and ``scienceqa-<i>``
    always match.

.. warning::
   **Training must use the ``train`` split.** The ``test`` split of ScienceQA is
   reserved for evaluation and must never be mixed into the training JSONL.

Only the standard library, ``datasets`` and ``Pillow`` are required.

Examples
--------
    python prepare_rl_data.py preference --out data/pref/dpo_train.jsonl \
        --rows 2000 --seed 0
    python prepare_rl_data.py scienceqa --out-dir data/rl_scienceqa \
        --rows 300 --split train

Both subcommand print a small summary (rows read / written / skipped and the
files that were produced) before exiting.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from io import BytesIO
from itertools import islice
from pathlib import Path
from typing import Any

from datasets import load_dataset  # NOTE: only `datasets`, `Pillow` + stdlib
from PIL import Image

# --------------------------------------------------------------------------- #
# constants
# --------------------------------------------------------------------------- #

ULTRAFEEDBACK = "trl-lib/ultrafeedback_binarized"
SCIENCEQA = "derek-thomas/ScienceQA"

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"  # "ABCDE..."[answer index]

ROLE_ASSISTANT = {"assistant", "gpt", "model", "bot"}
ROLE_USER = {"user", "human"}
PICTURE_MODES = {"1", "L", "LA", "P", "RGB", "RGBA"}  # PNG-safe image modes

# Verbatim prompt ending used by the ScienceQA conversations.
ANSWER_INSTRUCTION = (
    "You may reason briefly, but you MUST end your response with the letter of the "
    "correct option in exactly this form on its own line: 'Answer: X' (where X is the "
    "letter, e.g. Answer: C)."
)


# --------------------------------------------------------------------------- #
# generic helpers for chat-shaped data
# --------------------------------------------------------------------------- #
def _to_text(value: Any) -> str:
    """Flatten a message ``content`` field (``str`` | list of parts | ``None``) to text."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (bytes, bytearray)):
        return value.decode("utf-8", errors="replace").strip()
    if isinstance(value, Sequence):  # multimodal content: [{"type": "text", ...}, ...]
        parts: list[str] = []
        for part in value:
            piece = part.get("text", part.get("content", "")) if isinstance(part, Mapping) else part
            piece = "" if piece is None else str(piece).strip()
            if piece:
                parts.append(piece)
        return "\n".join(parts).strip()
    return str(value).strip()


def _is_role(message: Any, roles: set[str]) -> bool:
    return isinstance(message, Mapping) and str(message.get("role", "")).strip().lower() in roles


def _message_text(messages: Any, roles: set[str], *, first: bool) -> str:
    """Content of the first/last message whose role matches ``roles``."""
    if isinstance(messages, str):
        return messages.strip()
    if not isinstance(messages, Sequence):
        return ""
    ordered = list(messages) if first else list(reversed(list(messages)))
    for message in ordered:
        if _is_role(message, roles):
            text = _to_text(message.get("content"))
            if text:
                return text
    return ""


def _assistant_reply(value: Any) -> str:
    """Last assistant answer of a transcript (a ``str`` is returned as-is)."""
    return _message_text(value, ROLE_ASSISTANT, first=False)


def _user_prompt(value: Any) -> str:
    """First user (human) message of a transcript (a ``str`` is returned as-is)."""
    return _message_text(value, ROLE_USER, first=True)


def _prompt_text(row: Mapping[str, Any]) -> str:
    """Best-effort prompt extraction: transcripts first, then plain-text columns."""
    for key in ("chosen", "rejected", "prompt", "messages", "conversation"):
        value = row.get(key)
        if isinstance(value, str):
            text = value.strip()
            if text:
                return text
        else:
            text = _user_prompt(value)
            if text:
                return text
    return ""


def _record_count(args_rows: int) -> int | None:
    """``--rows`` guard: ``N`` caps the work, ``N <= 0`` means "all rows"."""
    return int(args_rows) if int(args_rows) > 0 else None


def _write_jsonl_line(handle, record: Mapping[str, Any]) -> None:
    handle.write(json.dumps(record, ensure_ascii=False) + "\n")


# --------------------------------------------------------------------------- #
# subcommand 1: preference pairs for DPO
# --------------------------------------------------------------------------- #
def cmd_preference(args: argparse.Namespace) -> int:
    """Convert UltraFeedback-binarized rows into ``{"prompt", "chosen", "rejected"}``."""
    limit = _record_count(args.rows)
    banner = "all rows" if limit is None else f"the first {limit} rows"
    print(f"[preference] loading {ULTRAFEEDBACK} split 'train' (taking {banner}) ...")
    dataset = load_dataset(ULTRAFEEDBACK, split="train")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    scanned = written = skipped = 0
    with out_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in islice(dataset, limit):
            scanned += 1
            prompt = _prompt_text(row)
            chosen = _assistant_reply(row.get("chosen"))
            rejected = _assistant_reply(row.get("rejected"))
            if not (prompt and chosen and rejected):  # empty fields -> drop the row
                skipped += 1
                continue
            _write_jsonl_line(handle, {"prompt": prompt, "chosen": chosen, "rejected": rejected})
            written += 1

    print("[preference] summary:")
    print(f"  source          : {ULTRAFEEDBACK} (split 'train')")
    print(f"  rows scanned    : {scanned}")
    print(f"  rows written    : {written}")
    print(f"  rows skipped    : {skipped} (empty prompt / chosen / rejected)")
    print(f"  seed            : {args.seed} (extraction is deterministic, nothing is sampled)")
    print(f"  output jsonl    : {str(Path(out_path).absolute())}")
    return 0


# --------------------------------------------------------------------------- #
# image helper for the multimodal job
# --------------------------------------------------------------------------- #
def _to_image(value: Any) -> Image.Image | None:
    """Decode the ``image`` column (``PIL.Image``, dict with bytes/path, or bytes)."""
    if value is None or value is False:
        return None
    if isinstance(value, Image.Image):
        return value
    if isinstance(value, Mapping):
        blob, path = value.get("bytes"), value.get("path")
        if isinstance(blob, (bytes, bytearray)):
            return Image.open(BytesIO(bytes(blob)))
        if isinstance(path, str) and path:
            return Image.open(path)
        return None
    if isinstance(value, (bytes, bytearray)):
        return Image.open(BytesIO(bytes(value)))
    return None


def _save_png(image: Image.Image, path: Path) -> None:
    """Write ``image`` to ``path`` as PNG (converted to 8-bit RGB when needed)."""
    if image.mode not in PICTURE_MODES:
        image = image.convert("RGB")
    image.save(path, format="PNG")


def _gold_letter(answer: Any, n_choices: int) -> str | None:
    """Map ScienceQA's answer column (a choice *index*) to its letter, e.g. 2 -> 'C'."""
    idx: int | None = None
    if isinstance(answer, bool):
        return None
    if isinstance(answer, (int, float)) and float(answer).is_integer():
        idx = int(answer)
    elif isinstance(answer, str):
        token = answer.strip()
        if token.lstrip("+-").isdigit():
            idx = int(token)
        elif token.upper() in LETTERS[:n_choices]:  # already a letter
            return token.upper()
    if idx is None or not (0 <= idx < min(n_choices, len(LETTERS))):
        return None
    return LETTERS[idx]


# --------------------------------------------------------------------------- #
# subcommand 2: multimodal ScienceQA conversations
# --------------------------------------------------------------------------- #
def cmd_scienceqa(args: argparse.Namespace) -> int:
    """Export image-based ScienceQA questions as conversational SFT rows + PNGs."""
    limit = _record_count(args.rows)
    if args.split == "test":
        print(
            "[scienceqa] WARNING: the 'test' split is reserved for evaluation; "
            "use --split train for training data."
        )

    banner = "all rows" if limit is None else f"up to {limit} rows"
    print(f"[scienceqa] loading {SCIENCEQA} split '{args.split}' (collecting {banner}) ...")
    dataset = load_dataset(SCIENCEQA, split=args.split)

    out_dir = Path(args.out_dir)
    img_dir = out_dir / "images"
    out_dir.mkdir(parents=True, exist_ok=True)
    img_dir.mkdir(parents=True, exist_ok=True)
    # Training data must come from the `train` split -> data/rl_scienceqa/scienceqa_train.jsonl
    jsonl_name = (
        "scienceqa_train.jsonl" if args.split == "train" else f"scienceqa_{args.split}.jsonl"
    )
    jsonl_path = out_dir / jsonl_name

    scanned = saved = 0
    no_image = few_choices = no_question = bad_answer = 0
    with jsonl_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in dataset:
            if limit is not None and saved >= limit:
                break
            scanned += 1

            question = _to_text(row.get("question"))
            choices = (
                [_to_text(c) for c in (row.get("choices") or [])]
                if isinstance(row.get("choices"), Sequence)
                else []
            )
            image = _to_image(row.get("image"))

            if image is None:
                no_image += 1
                continue
            if len(choices) < 2:
                few_choices += 1
                continue
            if not question:
                no_question += 1
                continue
            letter = _gold_letter(row.get("answer"), len(choices))
            if letter is None:
                bad_answer += 1
                continue

            index = saved
            row_id = f"scienceqa-{index}"
            png_path = img_dir / f"{row_id}.png"
            _save_png(image, png_path)

            choices_block = "\n".join(f"{LETTERS[k]}. {choice}" for k, choice in enumerate(choices))
            human_value = f"{question}\n\nChoices:\n{choices_block}\n\n{ANSWER_INSTRUCTION}"
            record = {
                "id": row_id,
                "conversations": [
                    {"from": "human", "value": human_value},
                    {"from": "gpt", "value": f"Answer: {letter}"},
                ],
                "image": str(Path(png_path).absolute()),
                "gold": letter,
            }
            _write_jsonl_line(handle, record)
            saved += 1

    print("[scienceqa] summary:")
    print(f"  source          : {SCIENCEQA} (split '{args.split}')")
    print(f"  rows scanned    : {scanned}")
    print(f"  rows written    : {saved} ({saved} image(s) saved as PNG)")
    skipped_total = no_image + few_choices + no_question + bad_answer
    print(
        f"  rows skipped    : {skipped_total} "
        f"(no image: {no_image}, <2 choices: {few_choices}, "
        f"empty question: {no_question}, bad answer index: {bad_answer})"
    )
    print(f"  images dir      : {str(Path(img_dir).absolute())}")
    print(f"  output jsonl    : {str(Path(jsonl_path).absolute())}")
    return 0


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="prepare_rl_data.py",
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    pref = sub.add_parser(
        "preference",
        help="build a DPO preference JSONL from trl-lib/ultrafeedback_binarized",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    pref.add_argument(
        "--out",
        default="data/pref/dpo_train.jsonl",
        help="destination JSONL file for the preference pairs",
    )
    pref.add_argument(
        "--rows", type=int, default=2000, help="how many source rows to read (<= 0 means all)"
    )
    pref.add_argument(
        "--seed",
        type=int,
        default=0,
        help="accepted for reproducible pipelines (extraction is deterministic)",
    )
    pref.set_defaults(func=cmd_preference)

    sci = sub.add_parser(
        "scienceqa",
        help="build multimodal (image + text) rows from derek-thomas/ScienceQA",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sci.add_argument(
        "--out-dir",
        default="data/rl_scienceqa",
        help="output directory (images land in <out-dir>/images)",
    )
    sci.add_argument(
        "--rows", type=int, default=300, help="stop after this many kept rows (<= 0 means all)"
    )
    sci.add_argument(
        "--split",
        default="train",
        help="dataset split to export; training MUST use 'train' "
        "('test' is reserved for evaluation)",
    )
    sci.set_defaults(func=cmd_scienceqa)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
