#!/usr/bin/env python3
"""prepare_rl_data.py — build small, tutorial-friendly datasets for RL fine-tuning.

This single script holds three little ETL jobs, one per subcommand:

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

``preference_mm``
    Build MULTIMODAL preference pairs for the offline preference trainer's
    ``image_column``/``video_column`` surface, two independent sub-jobs
    selected by ``--modality`` (``image``, ``video``, or ``both``, the
    default):

    * **image** pairs come from the public ``openbmb/RLAIF-V-Dataset``
      (``question``/``chosen``/``rejected`` are already plain strings, and
      its ``image`` column decodes to a real ``PIL.Image`` through
      ``datasets``' own Image feature -- MEASURED against the dataset's
      actual schema, 2026-10, not guessed). Each kept row becomes one image
      saved to ``<out-dir>/images/`` and one JSONL record::

          {"prompt": "<image> <question>", "image": <absolute PNG path>,
           "chosen": <chosen answer>, "rejected": <rejected answer>}

      These ARE human/model preference judgements over the SAME image (the
      dataset's own RLAIF annotation), the same semantics as ``preference``'s
      UltraFeedback pairs.

    * **video** pairs are built from OUR canonical video rows
      (``--video-jsonl``, default ``data/prepared/video.train.jsonl``):
      conversational rows carrying a ``<video>`` marker, a ``video`` clip
      path, and ``start``/``end`` segment seconds. ``chosen`` is a row's own
      REAL final assistant answer for its segment; ``rejected`` is the final
      answer of a DIFFERENT row's segment (a different video when the pool
      allows it), paired with THIS row's own prompt and clip::

          {"prompt": <row's human turn>, "video": <clip path>,
           "start": <float>, "end": <float>,
           "chosen": <this row's own answer>,
           "rejected": <a DIFFERENT row's answer, same prompt/clip>}

      .. warning::
         Unlike the image pairs above, this is a GROUNDING construction, not
         a human preference judgement: "rejected" is a plausible-sounding
         answer that does not describe what THIS clip actually shows, not an
         answer a human rated worse for the same clip. Treat it as a cheap
         proxy for "the model must look at the pixels", not as aligned
         preference supervision.

Only the standard library, ``datasets`` and ``Pillow`` are required.

Examples
--------
    python prepare_rl_data.py preference --out data/pref/dpo_train.jsonl \
        --rows 2000 --seed 0
    python prepare_rl_data.py scienceqa --out-dir data/rl_scienceqa \
        --rows 300 --split train
    python prepare_rl_data.py preference_mm --out-dir data/dpo_mm \
        --n-images 500 --n-video-pairs 500

Every subcommand prints a small summary (rows read / written / skipped and
the files that were produced) before exiting.
"""

from __future__ import annotations

import argparse
import json
import random
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
RLAIF_V = "openbmb/RLAIF-V-Dataset"

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
    # Accepts BOTH {"role": ..., "content": ...} (UltraFeedback's own
    # transcripts) and ShareGPT's {"from": ..., "value": ...} (our own
    # data/prepare_data.py's canonical video/image conversations) -- the
    # same dual-format convention train/conversation.py's _role_and_text
    # already uses, so a canonical video row does not need a second,
    # divergent parser here.
    if not isinstance(message, Mapping):
        return False
    raw_role = message.get("role", message.get("from", ""))
    return str(raw_role).strip().lower() in roles


def _message_text(messages: Any, roles: set[str], *, first: bool) -> str:
    """Content of the first/last message whose role matches ``roles``."""
    if isinstance(messages, str):
        return messages.strip()
    if not isinstance(messages, Sequence):
        return ""
    ordered = list(messages) if first else list(reversed(list(messages)))
    for message in ordered:
        if _is_role(message, roles):
            text = _to_text(message.get("content", message.get("value")))
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
# subcommand 3: multimodal preference pairs (image + video)
# --------------------------------------------------------------------------- #
def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a JSONL file into a list of dicts; blank lines are skipped."""
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                rows.append(json.loads(stripped))
    return rows


def _first_parquet_shard(dataset_id: str) -> str:
    """Download just the FIRST data shard of ``dataset_id`` and return its local path.

    Uses ``huggingface_hub.hf_hub_download`` (a resumable, chunked file
    download) rather than ``datasets``' own streaming reader. MEASURED: on a
    constrained link, streaming RLAIF-V-Dataset's ~1.1 GB first shard via
    ``load_dataset(..., streaming=True)`` repeatedly timed out or dropped
    mid-transfer with no net progress (fsspec's live HTTP range-reads do not
    resume cleanly after a drop); ``hf_hub_download``'s own resumable
    download of the exact same file over the exact same link completed in
    under a minute. One shard is enough for a tutorial-sized ``--n-images``
    prefix (RLAIF-V-Dataset's ~83k rows span ~14 shards, so shard 0 alone
    holds several thousand).
    """
    from huggingface_hub import HfApi, hf_hub_download

    info = HfApi().dataset_info(dataset_id, files_metadata=False)
    parquet_files = sorted(s.rfilename for s in info.siblings if s.rfilename.endswith(".parquet"))
    if not parquet_files:
        raise RuntimeError(f"{dataset_id}: no .parquet file found in the dataset repo")
    return hf_hub_download(repo_id=dataset_id, repo_type="dataset", filename=parquet_files[0])


def cmd_preference_mm_images(args: argparse.Namespace) -> int:
    """Convert ``openbmb/RLAIF-V-Dataset`` rows into image DPO pairs.

    MEASURED schema (datasets-server, 2026-10):
    ``question``/``chosen``/``rejected`` are already plain strings (unlike
    UltraFeedback's chat transcripts), and ``image`` decodes to a real
    ``PIL.Image`` through ``datasets``' own Image feature -- so this reuses
    ``_to_text``/``_to_image``/``_save_png`` with no transcript parsing.

    A positive ``--n-images`` (the common case) downloads only the FIRST
    data shard (see :func:`_first_parquet_shard`) rather than the whole
    ~83k-row, tens-of-GB dataset ``preference``'s plain ``load_dataset``
    pattern would pull down just to slice a few hundred rows from the
    front. ``--n-images <= 0`` ("all rows") instead falls back to the plain,
    full, non-streaming load -- it genuinely needs every shard.
    """
    limit = _record_count(args.n_images)
    banner = "all rows" if limit is None else f"the first {limit} rows"
    if limit is None:
        print(f"[preference_mm] loading {args.dataset} split 'train' (taking {banner}) ...")
        dataset = load_dataset(args.dataset, split="train")
    else:
        print(f"[preference_mm] downloading {args.dataset}'s first shard (taking {banner}) ...")
        shard_path = _first_parquet_shard(args.dataset)
        dataset = load_dataset("parquet", data_files=shard_path, split="train")

    out_dir = Path(args.out_dir)
    img_dir = out_dir / "images"
    out_dir.mkdir(parents=True, exist_ok=True)
    img_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "preference_mm_images.jsonl"

    scanned = written = skipped = 0
    with out_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in islice(dataset, limit):
            scanned += 1
            prompt = _to_text(row.get("question"))
            chosen = _to_text(row.get("chosen"))
            rejected = _to_text(row.get("rejected"))
            image = _to_image(row.get("image"))
            if not (prompt and chosen and rejected) or image is None:
                skipped += 1
                continue
            row_id = f"preference_mm_image-{written}"
            png_path = img_dir / f"{row_id}.png"
            _save_png(image, png_path)
            _write_jsonl_line(
                handle,
                {
                    "prompt": f"<image> {prompt}",
                    "image": str(Path(png_path).absolute()),
                    "chosen": chosen,
                    "rejected": rejected,
                },
            )
            written += 1

    print("[preference_mm] image summary:")
    print(f"  source          : {args.dataset} (split 'train')")
    print(f"  rows scanned    : {scanned}")
    print(f"  rows written    : {written}")
    print(f"  rows skipped    : {skipped} (empty prompt/chosen/rejected, or no image)")
    print(f"  images dir      : {str(Path(img_dir).absolute())}")
    print(f"  output jsonl    : {str(Path(out_path).absolute())}")
    if limit is not None and scanned < limit:
        print(
            f"  NOTE            : the first shard only held {scanned} row(s), fewer than "
            f"the requested {limit}; pass --n-images <= 0 to read every shard instead"
        )
    return 0


def cmd_preference_mm_video(args: argparse.Namespace) -> int:
    """Pair our canonical video rows into GROUNDING video DPO pairs.

    chosen = a row's own real final assistant answer for its segment;
    rejected = a DIFFERENT row's final answer (a different video when the
    pool allows it), paired with THIS row's own prompt and clip. See the
    module docstring's warning: this is a grounding construction, not a
    human preference judgement.
    """
    video_path = Path(args.video_jsonl)
    try:
        video_rows = _read_jsonl(video_path)
    except OSError as exc:
        print(f"[preference_mm] ERROR: could not read {video_path}: {exc}")
        return 1

    usable: list[dict[str, Any]] = []
    no_prompt = no_answer = no_clip = 0
    for row in video_rows:
        prompt = _user_prompt(row.get("conversations"))
        answer = _assistant_reply(row.get("conversations"))
        clip = row.get("video")
        if not prompt:
            no_prompt += 1
            continue
        if not answer:
            no_answer += 1
            continue
        if not clip:
            no_clip += 1
            continue
        usable.append(row)

    rng = random.Random(args.seed)
    limit = _record_count(args.n_video_pairs)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "preference_mm_video.jsonl"

    written = 0
    with out_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in usable:
            if limit is not None and written >= limit:
                break
            own_clip = row.get("video")
            # Prefer a DIFFERENT video; fall back to a different ROW of the
            # same video only when the pool is too small to avoid it.
            candidates = [r for r in usable if r is not row and r.get("video") != own_clip]
            if not candidates:
                candidates = [r for r in usable if r is not row]
            if not candidates:
                continue
            other = rng.choice(candidates)
            record = {
                "prompt": _user_prompt(row.get("conversations")),
                "video": own_clip,
                "start": row.get("start"),
                "end": row.get("end"),
                "chosen": _assistant_reply(row.get("conversations")),
                "rejected": _assistant_reply(other.get("conversations")),
            }
            _write_jsonl_line(handle, record)
            written += 1

    print("[preference_mm] video summary:")
    print(f"  source          : {video_path} (grounding construction, NOT human preference)")
    print(f"  rows read       : {len(video_rows)}")
    print(
        f"  rows usable     : {len(usable)} "
        f"(no prompt: {no_prompt}, no answer: {no_answer}, no clip: {no_clip})"
    )
    print(f"  pairs written   : {written}")
    print(f"  output jsonl    : {str(Path(out_path).absolute())}")
    return 0


def cmd_preference_mm(args: argparse.Namespace) -> int:
    """Dispatch the image and/or video multimodal-preference sub-jobs."""
    if args.modality in ("image", "both"):
        code = cmd_preference_mm_images(args)
        if code != 0:
            return code
    if args.modality in ("video", "both"):
        code = cmd_preference_mm_video(args)
        if code != 0:
            return code
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

    mm = sub.add_parser(
        "preference_mm",
        help="build multimodal (image+video) DPO preference pairs",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    mm.add_argument(
        "--out-dir",
        default="data/dpo_mm",
        help="output directory (images land in <out-dir>/images; two JSONL files are written)",
    )
    mm.add_argument(
        "--modality",
        choices=["image", "video", "both"],
        default="both",
        help="which multimodal pair job(s) to run",
    )
    mm.add_argument(
        "--dataset", default=RLAIF_V, help="HF dataset id for the image preference pairs"
    )
    mm.add_argument(
        "--n-images", type=int, default=500, help="how many RLAIF-V rows to read (<=0 means all)"
    )
    mm.add_argument(
        "--video-jsonl",
        default="data/prepared/video.train.jsonl",
        help="our canonical video rows (conversations + <video> marker + start/end) to pair",
    )
    mm.add_argument(
        "--n-video-pairs",
        type=int,
        default=500,
        help="how many video pairs to write (<=0 means all usable rows)",
    )
    mm.add_argument(
        "--seed",
        type=int,
        default=0,
        help="rejected-pairing shuffle seed (REAL here, unlike the other subcommands' --seed)",
    )
    mm.set_defaults(func=cmd_preference_mm)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
