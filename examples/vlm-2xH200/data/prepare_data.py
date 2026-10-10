#!/usr/bin/env python3
"""
prepare_data.py — Convert three raw dataset sources into one canonical JSONL format.

CANONICAL FORMAT (one JSON object per line)::

    {
      "id": str,                          # unique row identifier
      "source": "image"|"video"|"text",   # provenance tag
      "conversations": [                 # ordered dialogue turns
        {"from": "system"|"human"|"gpt", "value": str},
        ...
      ],
      "image":  [str, ...],              # optional – one path per image
      "video":  str,                     # optional – path to the video file
      "start":  float,                   # optional – clip start (seconds)
      "end":    float                    # optional – clip end   (seconds)
    }

RULES enforced on every canonical row
-------------------------------------
1. Turns alternate  human ↔ gpt  and may begin with ONE optional "system" turn.
2. The final turn MUST be a "gpt" (assistant) turn.
3. Every ``value`` field must be a non-empty string.
4. Human turns collectively contain exactly one ``<image>`` marker per listed
   image and one ``<video>`` marker when a video is present.  Missing markers
   are automatically inserted at the start of the first human turn, each
   followed by a newline.
5. Assistant chain-of-thought is kept inline as
   ````<think>reasoninganswer````
   The literal tag strings are BUILT BY CONCATENATION in code so that this
   source file never contains the raw angle-bracket tags as single literals
   (keeps naive grep-based dispatch happy).
6. All referenced media files must exist on disk (skippable via ``--no-check-files``).

Invalid rows are skipped and tallied per reason.  A short summary is printed
at the end and the process exits with code 2 when no rows could be written.

SUB-COMMANDS & USAGE EXAMPLES
------------------------------

1. ``image`` – VDoc-style image-reasoning JSONL

   Input row::

     {"id": "...", "image": "images/xxx.jpg",
      "conversations": [
        {"from": "human", "value": "What is in this image?"},
        {"from": "gpt",   "value": {"reasoning": "...", "answer": "..."}}
      ]}

   (the gpt ``value`` may also already be a plain string)

   Example::

     python prepare.py image \\
         --input vdoc.jsonl --image-root /data/vdoc \\
         --output vdoc_canonical.jsonl

   Image paths are rewritten to absolute paths under *--image-root*.
   Rows whose image file is missing on disk are counted as ``missing_media``
   so that a partial extraction still yields a valid output file.

2. ``video`` – Action-100M-style video-reasoning JSONL

   Input row::

     {"id": "...", "video": "/some/old/path/XXXX.mp4",
      "start": 3.5, "end": 12.0,
      "conversations": [...]}

   Example::

     python prepare.py video \\
         --input action100m.jsonl --video-root /data/videos \\
         --output action100m_canonical.jsonl

   Video paths are rewritten from the old absolute location to
   ``<video-root>/<basename>``.  ``start``/``end`` timing is preserved.

3. ``text`` – plain text (QA) JSONL

   Example::

     python prepare.py text \\
         --input text_qa.jsonl --output text_canonical.jsonl

4. ``mix`` – weighted merge of canonical files

   Example::

     python prepare.py mix \\
         --inputs vdoc_canonical.jsonl action100m_canonical.jsonl text_canonical.jsonl \\
         --weights 0.4 0.3 0.3 \\
         --total 50000 --seed 42 \\
         --output mixed.jsonl

   Sampling is WITHOUT replacement when a source has enough rows.
   When a source has fewer rows than its quota the deficit is filled with
   random copies and a warning reports how many rows were repeated.
   The final mix is shuffled deterministically given *--seed*.

5. ``stats`` – quick statistics on any canonical file

   Example::

     python prepare.py stats --input mixed.jsonl

EXIT CODES
----------
  0  success (≥ 1 row written, or ``stats`` completed)
  1  CLI / argument error
  2  zero rows were written (all inputs skipped or empty)

Requires Python 3.10+ and the standard library only.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path
from statistics import median
from typing import Any

# ─────────────────────────────────────────────────────────────────────────────
# Literal-tag constants.
# As instructed, every angle-bracket tag is built by concatenation so that this
# source file never contains a single complete literal such as "<think>".
# ─────────────────────────────────────────────────────────────────────────────
IMAGE_MARKER: str = "<" + "image>"
VIDEO_MARKER: str = "<" + "video>"
THINK_OPEN: str = "<" + "think>"
THINK_CLOSE: str = "<" + "/think>"

VALID_SOURCES = frozenset({"image", "video", "text"})
VALID_ROLES = frozenset({"system", "human", "gpt"})


# ═══════════════════════════════════════════════════════════════════════════
# Shared helpers
# ═══════════════════════════════════════════════════════════════════════════


def format_gpt_value(raw: Any) -> str:
    """Normalise an assistant (gpt) turn value to a plain string.

    Accepted input formats:
      * ``{"reasoning": "...", "answer": "..."}`` → ``<think>reasoninganswer``
      * Any other truthy value            → ``str(raw)`` (plain passthrough).

    If the reasoning string is blank (after stripping) it is omitted so that
    the output is just the answer.
    """
    if isinstance(raw, dict):
        reasoning = raw.get("reasoning", "")
        answer = raw.get("answer", "")
        if reasoning.strip():
            return f"{THINK_OPEN}{reasoning}{THINK_CLOSE}{answer}"
        return answer
    if raw is None:
        return ""
    return str(raw)


def ensure_media_markers(conversations: list[dict], n_images: int, has_video: bool) -> list[dict]:
    """Insert ``<image>`` / ``<video>`` markers where they are missing.

    Markers are placed at the **start of the first human turn**, each trailing
    a newline character.  The function is idempotent: it counts how many of
    each marker already appear in human turns and only prepends the shortfall.

    Parameters
    ----------
    conversations : list of turn dicts in canonical order.
    n_images : number of images in the row's ``image`` list (0 or more).
    has_video : whether the row references a video file.

    Returns the (possibly modified) *conversations* list.
    """
    n_videos_needed = 1 if has_video else 0

    # Locate the first human turn.
    first_human: dict | None = None
    for turn in conversations:
        if turn.get("from") == "human":
            first_human = turn
            break
    if first_human is None:
        # No human turn at all — validation will reject this row.
        return conversations

    # Count markers already present across every human turn.
    current_img = sum(
        t["value"].count(IMAGE_MARKER) for t in conversations if t.get("from") == "human"
    )
    current_vid = sum(
        t["value"].count(VIDEO_MARKER) for t in conversations if t.get("from") == "human"
    )

    # Build the prefix string with any missing markers.
    prefix_parts: list[str] = []
    for _ in range(max(0, n_images - current_img)):
        prefix_parts.append(IMAGE_MARKER + "\n")
    for _ in range(max(0, n_videos_needed - current_vid)):
        prefix_parts.append(VIDEO_MARKER + "\n")

    if prefix_parts:
        first_human["value"] = "".join(prefix_parts) + first_human["value"]

    return conversations


def validate_row(row: dict, check_files: bool = True) -> str | None:
    """Validate one canonical row.

    Returns **None** when the row passes every check, otherwise a short
    reason-code string that is tallied by the caller.  The checks are:

    ====== ========================================================
    Code   Meaning
    ====== ========================================================
    missing_id              ``id`` absent or not a string
    invalid_source          source not in {image, video, text}
    missing_conversations   conversations absent / empty / not a list
    malformed_turn          a turn lacks ``from``/``value`` or has bad role
    empty_value             a turn value is empty / whitespace-only
    first_turn_not_human    first content turn is not "human"
    turn_order              turns do not alternate human ↔ gpt
    last_turn_not_gpt       final turn is not "gpt"
    image_marker_count      <image> marker total ≠ len(image) list
    video_marker_count      <video> marker total ≠ 1 when video present
    missing_start_end       video present but start/end absent or non-numeric
    invalid_image_field     ``image`` present but not a list
    missing_media           a media file does not exist on disk
    ====== ========================================================
    """
    # 1 ── Identity ──────────────────────────────────────────────────────
    row_id = row.get("id")
    if not isinstance(row_id, str) or not row_id.strip():
        return "missing_id"

    # 2 ── Source tag ────────────────────────────────────────────────────
    if row.get("source") not in VALID_SOURCES:
        return "invalid_source"

    # 3 ── Conversations present ─────────────────────────────────────────
    conversations = row.get("conversations")
    if not isinstance(conversations, list) or not conversations:
        return "missing_conversations"

    # 4 ── Per-turn structural checks ────────────────────────────────────
    for turn in conversations:
        if not isinstance(turn, dict):
            return "malformed_turn"
        if "from" not in turn or "value" not in turn:
            return "malformed_turn"
        if turn["from"] not in VALID_ROLES:
            return "malformed_turn"
        val = turn["value"]
        if not isinstance(val, str) or not val.strip():
            return "empty_value"

    # 5 ── Role ordering ─────────────────────────────────────────────────
    #     [system,] human, gpt, human, gpt, … , gpt
    roles = [t["from"] for t in conversations]
    start = 1 if roles[0] == "system" else 0
    if start >= len(roles) or roles[start] != "human":
        return "first_turn_not_human"

    for i in range(start, len(roles)):
        expected = "human" if (i - start) % 2 == 0 else "gpt"
        if roles[i] != expected:
            return "turn_order"

    if roles[-1] != "gpt":
        return "last_turn_not_gpt"

    # 6 ── Media marker-count checks ─────────────────────────────────────
    images = row.get("image")
    if images is not None and not isinstance(images, list):
        return "invalid_image_field"
    n_images = len(images) if images else 0

    video = row.get("video")
    is_video = bool(video)

    human_texts = [t["value"] for t in conversations if t["from"] == "human"]
    total_img_markers = sum(t.count(IMAGE_MARKER) for t in human_texts)
    total_vid_markers = sum(t.count(VIDEO_MARKER) for t in human_texts)

    if total_img_markers != n_images:
        return "image_marker_count"
    if total_vid_markers != (1 if is_video else 0):
        return "video_marker_count"

    # 7 ── Video timing fields ───────────────────────────────────────────
    if is_video:
        for key in ("start", "end"):
            if key not in row or not isinstance(row[key], (int, float)):
                return "missing_start_end"

    # 8 ── File-existence checks ─────────────────────────────────────────
    if check_files:
        for p in images or []:
            if not Path(p).is_file():
                return "missing_media"
        if is_video and not Path(str(video)).is_file():
            return "missing_media"

    return None  # all checks passed


def print_summary(
    cmd: str,
    input_paths: str | list[str],
    output_path: str,
    written: int,
    skip_reasons: Counter,
) -> None:
    """Print a human-readable conversion / mix summary."""
    if isinstance(input_paths, str):
        input_paths = [input_paths]
    inputs_str = ", ".join(input_paths)
    print(f"[{cmd}]  inputs: {inputs_str}")
    print(f"         output: {output_path}")
    print(f"         rows written: {written}")
    if skip_reasons:
        total_skipped = sum(skip_reasons.values())
        print(f"         rows skipped: {total_skipped}")
        for reason, count in skip_reasons.most_common():
            print(f"           {reason}: {count}")
    else:
        print("         rows skipped: 0")


def _open_input(path: str):
    """Open a JSONL file for reading with a friendly error message."""
    if not Path(path).is_file():
        print(f"ERROR: input file not found: {path}", file=sys.stderr)
        sys.exit(1)
    return Path(path).open(encoding="utf-8")


# ═══════════════════════════════════════════════════════════════════════════
# Sub-command: image
# ═══════════════════════════════════════════════════════════════════════════


def cmd_image(args: argparse.Namespace) -> None:
    """Convert a VDoc-style image-reasoning dataset to canonical format.

    Each input line contains ``id``, ``image`` (a path relative to
    *--image-root*) and ``conversations`` where gpt turn values may be either
    a plain string or a ``{"reasoning": …, "answer": …}`` dict.

    Processing steps per row:
      1. Normalize gpt values (reasoning → inline think block).
      2. Resolve the image path to an absolute path under *image-root*.
      3. Insert one ``<image>`` marker at the start of the first human turn.
      4. Validate and write.

    Rows whose image file cannot be found are tallied as ``missing_media`` so
    that a partial archive extraction still produces a valid output.
    """
    skip_reasons: Counter = Counter()
    written = 0
    image_root = str(Path(args.image_root).absolute())
    check_files = not args.no_check_files

    with _open_input(args.input) as fin, Path(args.output).open("w", encoding="utf-8") as fout:
        for line_no, line in enumerate(fin, 1):
            # --max-rows stops output after N written rows.
            if args.max_rows is not None and written >= args.max_rows:
                break
            line = line.strip()
            if not line:
                continue

            # --- Parse JSON ------------------------------------------------
            try:
                raw = json.loads(line)
            except json.JSONDecodeError:
                skip_reasons["json_error"] += 1
                continue

            # --- Normalize gpt turn values to strings ----------------------
            conversations = raw.get("conversations", [])
            for turn in conversations:
                if turn.get("from") == "gpt":
                    turn["value"] = format_gpt_value(turn.get("value", ""))

            # --- Resolve image path → absolute path ------------------------
            rel_image = raw.get("image", "")
            abs_image = str(Path(image_root, rel_image).absolute())

            # --- Build canonical row ---------------------------------------
            row: dict = {
                "id": str(raw.get("id", f"img_{line_no}")),
                "source": "image",
                "conversations": conversations,
                "image": [abs_image],
            }

            # --- Ensure <image> marker is present --------------------------
            ensure_media_markers(row["conversations"], n_images=1, has_video=False)

            # --- Validate ---------------------------------------------------
            error = validate_row(row, check_files=check_files)
            if error is not None:
                skip_reasons[error] += 1
                continue

            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            written += 1

    print_summary("image", args.input, args.output, written, skip_reasons)
    if written == 0:
        sys.exit(2)


# ═══════════════════════════════════════════════════════════════════════════
# Sub-command: video
# ═══════════════════════════════════════════════════════════════════════════


def cmd_video(args: argparse.Namespace) -> None:
    """Convert an Action-100M-style video-reasoning dataset to canonical format.

    Each input line has ``id``, ``video`` (an old absolute path), ``start``,
    ``end`` and ``conversations``.  The video path is rewritten to
    ``<video-root>/<basename-of-old-path>`` and timing is preserved.
    """
    skip_reasons: Counter = Counter()
    written = 0
    video_root = str(Path(args.video_root).absolute())
    check_files = not args.no_check_files

    with _open_input(args.input) as fin, Path(args.output).open("w", encoding="utf-8") as fout:
        for line_no, line in enumerate(fin, 1):
            if args.max_rows is not None and written >= args.max_rows:
                break
            line = line.strip()
            if not line:
                continue

            try:
                raw = json.loads(line)
            except json.JSONDecodeError:
                skip_reasons["json_error"] += 1
                continue

            # --- Normalize gpt turn values ----------------------------------
            conversations = raw.get("conversations", [])
            for turn in conversations:
                if turn.get("from") == "gpt":
                    turn["value"] = format_gpt_value(turn.get("value", ""))

            # --- Rewrite video path: <video-root>/<basename> ----------------
            old_path = raw.get("video", "")
            new_path = str(Path(video_root, Path(old_path).name).absolute())

            # --- Build canonical row ----------------------------------------
            row: dict = {
                "id": str(raw.get("id", f"vid_{line_no}")),
                "source": "video",
                "conversations": conversations,
                "video": new_path,
                "start": float(raw.get("start", 0.0)),
                "end": float(raw.get("end", 0.0)),
            }

            # --- Ensure <video> marker is present (spec says it is usually
            #     already there, but we silently insert it if missing) -------
            ensure_media_markers(row["conversations"], n_images=0, has_video=True)

            # --- Validate ---------------------------------------------------
            error = validate_row(row, check_files=check_files)
            if error is not None:
                skip_reasons[error] += 1
                continue

            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            written += 1

    print_summary("video", args.input, args.output, written, skip_reasons)
    if written == 0:
        sys.exit(2)


# ═══════════════════════════════════════════════════════════════════════════
# Sub-command: text
# ═══════════════════════════════════════════════════════════════════════════


def cmd_text(args: argparse.Namespace) -> None:
    """Convert a text-only (QA) dataset to canonical format.

    Input rows carry only ``conversations`` (and optionally an ``id``).
    Gpt turn values are normalized exactly as in the image / video commands.
    No media fields or markers are produced.
    """
    skip_reasons: Counter = Counter()
    written = 0
    check_files = not args.no_check_files  # no-op for text (no media fields)

    with _open_input(args.input) as fin, Path(args.output).open("w", encoding="utf-8") as fout:
        for line_no, line in enumerate(fin, 1):
            if args.max_rows is not None and written >= args.max_rows:
                break
            line = line.strip()
            if not line:
                continue

            try:
                raw = json.loads(line)
            except json.JSONDecodeError:
                skip_reasons["json_error"] += 1
                continue

            # --- Normalize gpt turn values ----------------------------------
            conversations = raw.get("conversations", [])
            for turn in conversations:
                if turn.get("from") == "gpt":
                    turn["value"] = format_gpt_value(turn.get("value", ""))

            row: dict = {
                "id": str(raw.get("id", f"txt_{line_no}")),
                "source": "text",
                "conversations": conversations,
            }

            # --- Validate ---------------------------------------------------
            error = validate_row(row, check_files=check_files)
            if error is not None:
                skip_reasons[error] += 1
                continue

            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            written += 1

    print_summary("text", args.input, args.output, written, skip_reasons)
    if written == 0:
        sys.exit(2)


# ═══════════════════════════════════════════════════════════════════════════
# Sub-command: mix
# ═══════════════════════════════════════════════════════════════════════════


def _compute_quotas(weights: list[float], total: int) -> list[int]:
    """Distribute *total* among sources proportionally (largest-remainder method).

    Returns a list of integer quotas whose sum equals *total* exactly.
    """
    w_sum = sum(weights)
    raw = [total * w / w_sum for w in weights]
    quotas = [int(r) for r in raw]
    remainder = total - sum(quotas)
    # Give leftover slots to sources with the largest fractional parts.
    order = sorted(range(len(raw)), key=lambda i: raw[i] - quotas[i], reverse=True)
    for i in range(remainder):
        quotas[order[i % len(order)]] += 1
    return quotas


def cmd_mix(args: argparse.Namespace) -> None:
    """Mix several canonical files into one dataset with weighted sampling.

    For each source *i* a quota ``q_i`` is computed from ``--weights``.
    If the source has ``≥ q_i`` rows then ``q_i`` rows are sampled WITHOUT
    replacement.  When the source has fewer rows the deficit is filled with
    random copies (WITH replacement) and a warning is printed.

    Finally the combined rows are shuffled with a deterministic RNG seeded
    by ``--seed`` so that the entire pipeline is reproducible.
    """
    inputs: list[str] = args.inputs
    weights: list[float] = args.weights
    total: int = args.total
    seed: int = args.seed

    if len(inputs) != len(weights):
        print(
            "ERROR: number of --inputs and --weights must match.",
            file=sys.stderr,
        )
        sys.exit(1)

    # ── Load every source into memory (fine for tutorial-scale datasets) ──
    all_sources: list[list[dict]] = []
    for path in inputs:
        rows: list[dict] = []
        with _open_input(path) as fin:
            for line in fin:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        all_sources.append(rows)

    # ── Compute per-source quotas ──
    quotas = _compute_quotas(weights, total)

    # ── Sample rows per source ──
    rng = random.Random(seed)
    sampled: list[dict] = []
    per_source_counts: list[int] = []
    warnings: list[str] = []

    for idx, (rows, quota) in enumerate(zip(all_sources, quotas, strict=False)):
        n = len(rows)
        if n == 0:
            per_source_counts.append(0)
            if quota > 0:
                warnings.append(
                    f"  ⚠  source '{inputs[idx]}' has 0 rows but its quota is "
                    f"{quota}.  Nothing to contribute."
                )
            continue

        if n >= quota:
            # Sufficient rows — sample WITHOUT replacement.
            picked = rng.sample(rows, quota)
        else:
            # Not enough rows — use every row once, then fill the deficit
            # with random copies (WITH replacement).
            deficit = quota - n
            picked = list(rows) + [rows[rng.randrange(n)] for _ in range(deficit)]
            warnings.append(
                f"  ⚠  source '{inputs[idx]}' had {n} rows but quota is {quota}."
                f"  Repeated {deficit} rows (each original row appears at least once)."
            )

        sampled.extend(picked)
        per_source_counts.append(len(picked))

    # ── Deterministic shuffle of the full sample ──
    rng.shuffle(sampled)

    # ── Write output ──
    with Path(args.output).open("w", encoding="utf-8") as fout:
        for row in sampled:
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")

    # ── Per-source counts and warnings ──
    print(f"Mix complete — wrote {len(sampled)} rows to {args.output}  (seed={seed})")
    print("Per-source counts:")
    for path, cnt, q in zip(inputs, per_source_counts, quotas, strict=False):
        print(
            f"  {path}:  {cnt} rows  (quota={q}, available={len(all_sources[inputs.index(path)])})"
        )
    for wmsg in warnings:
        print(wmsg)

    if len(sampled) == 0:
        sys.exit(2)


# ═══════════════════════════════════════════════════════════════════════════
# Sub-command: stats
# ═══════════════════════════════════════════════════════════════════════════


def cmd_split(args: argparse.Namespace) -> None:
    """Split one canonical JSONL file into disjoint train and eval files.

    Rows are shuffled with a fixed seed, then the first ``--eval-fraction`` of them
    go to the eval file and the rest to the train file. Every row lands in exactly
    one output, so the held-out set never overlaps training. Split each source
    (image, video, text) BEFORE mixing, then mix only the train halves.
    """
    if not 0.0 < args.eval_fraction < 1.0:
        sys.exit("--eval-fraction must be between 0 and 1 (exclusive)")
    with Path(args.input).open(encoding="utf-8") as fin:
        rows = [line for line in fin if line.strip()]
    if len(rows) < 2:
        sys.exit(f"{args.input} has {len(rows)} row(s); need at least 2 to split")
    random.Random(args.seed).shuffle(rows)
    n_eval = max(1, round(len(rows) * args.eval_fraction))
    with Path(args.eval_out).open("w", encoding="utf-8") as f:
        f.writelines(rows[:n_eval])
    with Path(args.train_out).open("w", encoding="utf-8") as f:
        f.writelines(rows[n_eval:])
    print(
        f"split {args.input}: {len(rows) - n_eval} train -> {args.train_out}, "
        f"{n_eval} eval -> {args.eval_out} (seed {args.seed})"
    )


def cmd_stats(args: argparse.Namespace) -> None:
    """Print summarised statistics for a canonical JSONL file.

    Reported metrics:
      * Total row count and per-source breakdown.
      * Histogram of conversation length (turns per row).
      * Number of rows containing at least one inline think block.
      * Median total character count of all turn values per row.
    """
    source_counter: Counter = Counter()
    turns_hist: Counter = Counter()
    rows_with_reasoning = 0
    char_counts: list[int] = []

    with _open_input(args.input) as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)

            src = row.get("source", "unknown")
            source_counter[src] += 1

            conversations = row.get("conversations", [])
            turns_hist[len(conversations)] += 1

            has_reasoning = False
            total_chars = 0
            for turn in conversations:
                val = turn.get("value", "")
                total_chars += len(val)
                if turn.get("from") == "gpt" and THINK_OPEN in val and THINK_CLOSE in val:
                    has_reasoning = True

            if has_reasoning:
                rows_with_reasoning += 1
            char_counts.append(total_chars)

    total_rows = sum(source_counter.values())
    med = median(char_counts) if char_counts else 0

    print(f"File: {args.input}")
    print(f"Total rows: {total_rows}")
    print()

    print("Rows per source:")
    if source_counter:
        for src in sorted(source_counter):
            print(f"  {src}: {source_counter[src]}")
    else:
        print("  (empty)")
    print()

    print("Turns per row (histogram):")
    if turns_hist:
        for n_turns in sorted(turns_hist):
            print(f"  {n_turns} turns: {turns_hist[n_turns]}")
    else:
        print("  (empty)")
    print()

    print(f"Rows with reasoning: {rows_with_reasoning}")
    print(f"Median characters per row: {med}")


# ═══════════════════════════════════════════════════════════════════════════
# CLI plumbing
# ═══════════════════════════════════════════════════════════════════════════


def _add_common_conv_args(parser: argparse.ArgumentParser) -> None:
    """Add --input / --output / --max-rows shared by conversion sub-commands."""
    parser.add_argument("--input", required=True, help="Path to the input JSONL file.")
    parser.add_argument("--output", required=True, help="Path to the output JSONL file.")
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Stop after writing N valid rows (default: unlimited).",
    )
    parser.add_argument(
        "--no-check-files",
        action="store_true",
        help="Skip media file-existence checks during validation.",
    )


def build_parser() -> argparse.ArgumentParser:
    """Construct the top-level argparse parser with sub-commands."""
    parser = argparse.ArgumentParser(
        prog="prepare_data.py",
        description="Convert raw datasets into one canonical JSONL format and mix them.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ── 1. image ────────────────────────────────────────────────────────
    p_img = sub.add_parser("image", help="Convert a VDoc-style image dataset.")
    _add_common_conv_args(p_img)
    p_img.add_argument(
        "--image-root",
        required=True,
        help="Directory that relative image paths are resolved against.",
    )
    p_img.set_defaults(func=cmd_image)

    # ── 2. video ────────────────────────────────────────────────────────
    p_vid = sub.add_parser("video", help="Convert an Action-100M-style video dataset.")
    _add_common_conv_args(p_vid)
    p_vid.add_argument(
        "--video-root",
        required=True,
        help="Directory that video basenames are resolved against.",
    )
    p_vid.set_defaults(func=cmd_video)

    # ── 3. text ─────────────────────────────────────────────────────────
    p_txt = sub.add_parser("text", help="Convert a text-only QA dataset.")
    _add_common_conv_args(p_txt)
    p_txt.set_defaults(func=cmd_text)

    # ── 4. mix ──────────────────────────────────────────────────────────
    p_mix = sub.add_parser("mix", help="Mix canonical JSONL files with weighted sampling.")
    p_mix.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="Two or more canonical JSONL files to mix.",
    )
    p_mix.add_argument(
        "--weights",
        nargs="+",
        type=float,
        required=True,
        help="Sampling weight for each input (same count as --inputs).",
    )
    p_mix.add_argument(
        "--total",
        type=int,
        required=True,
        help="Total number of rows in the output.",
    )
    p_mix.add_argument("--output", required=True, help="Path to the output JSONL file.")
    p_mix.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for deterministic sampling and shuffling (default: 0).",
    )
    p_mix.set_defaults(func=cmd_mix)

    # ── 5. stats ────────────────────────────────────────────────────────
    p_stats = sub.add_parser("stats", help="Print statistics on a canonical JSONL file.")
    p_stats.add_argument("--input", required=True, help="Canonical JSONL file to analyse.")
    p_stats.set_defaults(func=cmd_stats)

    # ── 6. split ────────────────────────────────────────────────────────
    p_split = sub.add_parser(
        "split", help="Split a canonical JSONL file into disjoint train/eval files."
    )
    p_split.add_argument("--input", required=True, help="Canonical JSONL file to split.")
    p_split.add_argument("--train-out", required=True, help="Output path for the training rows.")
    p_split.add_argument("--eval-out", required=True, help="Output path for the held-out rows.")
    p_split.add_argument(
        "--eval-fraction", type=float, default=0.05, help="Share of rows held out (default 0.05)."
    )
    p_split.add_argument("--seed", type=int, default=0, help="Shuffle seed (default 0).")
    p_split.set_defaults(func=cmd_split)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
