#!/usr/bin/env python3
"""
prepare_librispeech.py — Convert local LibriSpeech parquet shards to FLAC + JSONL.

Reads HuggingFace ``openslr/librispeech_asr`` parquet shards (config ``clean``)
that have already been downloaded to local disk, e.g.::

    <root>/clean/train.100/0000.parquet
    <root>/clean/train.100/0001.parquet
    <root>/clean/validation/*.parquet
    <root>/clean/test/*.parquet

and produces (a) one 16 kHz mono FLAC file per utterance and (b) one JSONL
manifest with one row per utterance.

CANONICAL MANIFEST FORMAT (one JSON object per line)::

    {
      "id": "1272-128104-0000",
      "text": "Transcribe this audio exactly.",
      "answer": "<transcript text, as-is from the dataset>",
      "audio": "/abs/path/to/the/flac/file.flac",
      "duration": 5.855
    }

RULES enforced on every manifest row
------------------------------------
1. ``text`` is ALWAYS the literal prompt "Transcribe this audio exactly."; it
   is the training prompt and is never derived from the dataset.
2. ``answer`` is the raw transcript from the parquet ``text`` column with only
   leading/trailing whitespace stripped — no case, punctuation or number
   normalisation of any kind.
3. ``duration`` is recomputed from the decoded samples
   (``len(array) / sampling_rate``) and rounded to 3 decimals.  Any
   ``duration`` / ``num_samples`` column in the parquet row is ignored.
4. ``audio`` is an absolute path to a 16-bit PCM FLAC file named ``<id>.flac``
   written under ``--audio-out-dir``.
5. ``id`` is globally unique within one run.  A colliding id is SKIPPED and
   counted as ``duplicate_id`` — never renamed, never overwritten.
6. Audio must already be 16 kHz mono (LibriSpeech is).  Anything else is
   skipped (``bad_sample_rate`` / ``not_mono``): this script NEVER resamples
   and NEVER mixes channels down.

The embedded ``audio`` column is decoded defensively — see
``_decode_audio_cell`` — because the parquet export may store either encoded
file ``bytes`` or already-decoded ``array`` + ``sampling_rate`` samples.

Rows are streamed: every accepted utterance is written to disk and appended to
the manifest immediately, one ``json.dumps(...) + "\\n"`` at a time.

USAGE
-----
::

    python prepare_librispeech.py \\
        --input /data/librispeech/clean/train.100 \\
        --audio-out-dir /data/librispeech/flac/train.100 \\
        --manifest-out /data/librispeech/manifests/train.100.jsonl

    python prepare_librispeech.py \\
        --input /data/librispeech/clean/validation \\
        --audio-out-dir /data/librispeech/flac/validation \\
        --manifest-out /data/librispeech/manifests/validation.jsonl

Both invocations share one duration policy (``--min-duration`` /
``--max-duration``) so the eval set is filtered exactly like the train set.

EXIT CODES
----------
  0  success (>= 1 row written)
  1  CLI / argument error
  2  zero rows were written (all rows skipped or no input)

Requires Python 3.10+.  Third-party requirements: ``pyarrow``, ``soundfile``
and ``numpy`` (no ``datasets``, ``librosa`` or ``torchaudio``).
"""

from __future__ import annotations

import argparse
import glob
import io
import json
import random
import sys
from collections import Counter
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TextIO

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

PROMPT_TEXT: str = "Transcribe this audio exactly."
TARGET_SAMPLE_RATE: int = 16_000
FLAC_SUBTYPE: str = "PCM_16"

SKIP_REASONS: tuple[str, ...] = (
    "too_long",
    "too_short",
    "duplicate_id",
    "bad_sample_rate",
    "not_mono",
    "decode_error",
)

_ROW_BATCH_SIZE: int = 64
_MAX_WARNINGS_PER_REASON: int = 5

# Everything a corrupt row or shard can realistically raise: pyarrow and
# soundfile both derive their errors from these builtins.
_ROW_ERRORS: tuple[type[BaseException], ...] = (
    OSError,
    ValueError,
    TypeError,
    RuntimeError,
    KeyError,
    IndexError,
    MemoryError,
    NotImplementedError,
)


# ─────────────────────────────────────────────────────────────────────────────
# Run state
# ─────────────────────────────────────────────────────────────────────────────


def _empty_skip_reasons() -> Counter[str]:
    """Create a skip-reason counter pre-seeded with every known reason."""
    return Counter({reason: 0 for reason in SKIP_REASONS})


@dataclass
class RunState:
    """Mutable state shared across every shard of one invocation."""

    audio_out_dir: Path
    manifest: TextIO
    max_duration: float
    min_duration: float
    id_prefix: str
    limit: int | None
    shards: int = 0
    rows_seen: int = 0
    rows_written: int = 0
    total_duration: float = 0.0
    min_written: float | None = None
    max_written: float | None = None
    seen_ids: set[str] = field(default_factory=set)
    skip_reasons: Counter[str] = field(default_factory=_empty_skip_reasons)
    warned: Counter[str] = field(default_factory=Counter)

    @property
    def done(self) -> bool:
        """True once ``--limit`` valid rows have been written."""
        return self.limit is not None and self.rows_written >= self.limit


def _warn(state: RunState, reason: str, message: str) -> None:
    """Print *message* to stderr, capped per *reason* to avoid log spam."""
    if state.warned[reason] < _MAX_WARNINGS_PER_REASON:
        print(f"WARNING: {message}", file=sys.stderr)
    state.warned[reason] += 1


# ─────────────────────────────────────────────────────────────────────────────
# Row-level helpers
# ─────────────────────────────────────────────────────────────────────────────


def _decode_audio_cell(cell: Any) -> tuple[np.ndarray, int]:
    """Decode one embedded ``audio`` cell into ``(samples, sample_rate)``.

    Two parquet encodings are supported:

    * ``{"bytes": <encoded file bytes>, "path": ...}`` — decoded with
      :func:`soundfile.read` from an in-memory buffer.
    * ``{"array": <samples>, "sampling_rate": <hz>}`` — used as-is.

    A bare ``bytes``-like cell is accepted too.  Anything else raises a
    :class:`RuntimeError` naming the keys that were found, so an unexpected
    schema fails loudly on the first row instead of silently producing empty
    output.
    """
    if isinstance(cell, (bytes, bytearray, memoryview)):
        samples, samplerate = sf.read(io.BytesIO(bytes(cell)))
        return np.asarray(samples), int(samplerate)

    if not isinstance(cell, Mapping):
        raise RuntimeError(
            "unsupported audio cell type "
            f"{type(cell).__name__}; expected a mapping with 'bytes' or "
            "'array' + 'sampling_rate'"
        )

    raw_bytes = cell.get("bytes")
    if raw_bytes is not None:
        samples, samplerate = sf.read(io.BytesIO(bytes(raw_bytes)))
        return np.asarray(samples), int(samplerate)

    if cell.get("array") is not None and cell.get("sampling_rate") is not None:
        return np.asarray(cell["array"]), int(cell["sampling_rate"])

    keys = sorted(str(key) for key in cell)
    raise RuntimeError(
        f"unsupported audio cell schema; available keys: {keys} "
        "(expected 'bytes' or 'array' + 'sampling_rate')"
    )


def _iter_parquet_rows(shard_path: Path) -> Iterator[dict[str, Any]]:
    """Yield raw parquet rows one at a time from a single shard file."""
    parquet_file = pq.ParquetFile(str(shard_path))
    for batch in parquet_file.iter_batches(batch_size=_ROW_BATCH_SIZE):
        yield from batch.to_pylist()


def _sanitize_id(raw: str) -> str:
    """Make an id safe to use as a filename stem (files are named ``<id>.flac``)."""
    cleaned = raw.strip()
    for bad in ("/", "\\", "\x00"):
        cleaned = cleaned.replace(bad, "_")
    return cleaned


def _name_stem(value: Any) -> str:
    """Return the filename stem of a path-like value (``""`` when unusable)."""
    if not isinstance(value, str) or not value.strip():
        return ""
    return Path(value.strip()).stem


def _build_id(row: Mapping[str, Any], fallback: str) -> str:
    """Build a stable utterance id from whatever the parquet row provides.

    Preference order:

    1. a literal ``id`` column (used directly),
    2. the stem of a ``file`` / ``filename`` column,
    3. the stem of the ``audio`` struct's ``path``,
    4. ``{speaker_id}-{chapter_id}-{utterance...}`` when those columns exist,
    5. *fallback* (``<shard-stem>-<row-index>``).
    """
    direct = row.get("id")
    if direct is not None and not isinstance(direct, (Mapping, list, tuple)):
        candidate = _sanitize_id(str(direct))
        if candidate:
            return candidate

    for key in ("file", "filename"):
        candidate = _sanitize_id(_name_stem(row.get(key)))
        if candidate:
            return candidate

    audio = row.get("audio")
    if isinstance(audio, Mapping):
        candidate = _sanitize_id(_name_stem(audio.get("path")))
        if candidate:
            return candidate

    speaker = row.get("speaker_id")
    chapter = row.get("chapter_id")
    if speaker is not None and chapter is not None:
        utterance = ""
        for key in ("utterance_id", "utterance_idx", "utterance_index", "utt_id"):
            value = row.get(key)
            if value is not None and not isinstance(value, (Mapping, list, tuple)):
                utterance = str(value).strip()
                break
        parts = [str(speaker).strip(), str(chapter).strip()]
        parts.append(utterance or fallback)
        candidate = _sanitize_id("-".join(parts))
        if candidate:
            return candidate

    return _sanitize_id(fallback) or fallback


def _answer_text(row: Mapping[str, Any]) -> str:
    """Return the transcript, SENTENCE-CASED, not LibriSpeech's raw upper case.

    MEASURED, 2026-10-11: training Whisper-large-v3 on LibriSpeech's raw upper-case transcripts
    (``"SWEETWATER SOMEONE DREW THAT WEAPON OUT"``) pushes it onto upper-case BPE pieces it rarely
    uses, which fragments words it already transcribes correctly zero-shot into multiple pieces
    (``"Sweetwater"`` -> ``"SWEET WATER"``, ``"mountebank"`` -> ``"MOUNTAIN BANK"``) -- a measured
    WER regression (2.25% -> 3.32% on dev-clean) root-caused to this exact target format, not to
    the training recipe or the software stack. Every LibriSpeech-trained result in this repo's own
    speech_p3..p7 campaigns used sentence-cased targets instead: whitespace collapsed, lower-cased,
    then only the first character upper-cased -- no punctuation added. This mirrors
    ``validation_campaigns/speech_p3/build_librispeech_train.py`` and
    ``validation_campaigns/speech_p6/build_ami_train.py``'s own transform exactly, so this script's
    manifest trains the same target distribution those campaigns measured.

    The EVAL manifest (chapter 02 §3) is NOT rebuilt with this change -- it keeps LibriSpeech's raw
    upper-case reference text as-is. ``eval_wer.py``'s normalizer casefolds both the reference and
    the hypothesis before scoring (``foundationscale.train.speech_metrics.normalize_transcript``),
    so an upper-case reference scores identically to a sentence-cased one; only the TRAINING target
    format mattered here, not the eval reference's case.
    """
    raw = row.get("text")
    text = raw.strip() if isinstance(raw, str) else str(raw or "").strip()
    return " ".join(text.split()).lower().capitalize()


# ─────────────────────────────────────────────────────────────────────────────
# Processing
# ─────────────────────────────────────────────────────────────────────────────


def _process_row(row: Mapping[str, Any], fallback_id: str, state: RunState) -> None:
    """Decode, filter and write one raw parquet row.

    Order: decode → sample-rate / channel checks → duration → min/max duration
    filter → id (with ``--id-prefix``) → duplicate check → FLAC write →
    manifest append.  Every rejection is tallied in ``state.skip_reasons``.
    """
    try:
        array, samplerate = _decode_audio_cell(row.get("audio"))
    except _ROW_ERRORS as exc:
        state.skip_reasons["decode_error"] += 1
        _warn(state, "decode_error", f"cannot decode audio for {fallback_id}: {exc}")
        return

    # Never resample: refuse anything that is not already 16 kHz.
    if samplerate != TARGET_SAMPLE_RATE:
        state.skip_reasons["bad_sample_rate"] += 1
        return

    # Never mix down: a (n, 1) column is still mono, anything else is not.
    if array.ndim == 2 and array.shape[1] == 1:
        array = array.reshape(-1)
    if array.ndim != 1:
        state.skip_reasons["not_mono"] += 1
        return

    # Duration is recomputed from the decoded samples, never taken from the row.
    seconds = array.shape[0] / float(samplerate)
    if seconds > state.max_duration:
        state.skip_reasons["too_long"] += 1
        return
    if seconds < state.min_duration:
        state.skip_reasons["too_short"] += 1
        return

    row_id = state.id_prefix + _build_id(row, fallback_id)
    if row_id in state.seen_ids:
        state.skip_reasons["duplicate_id"] += 1
        return
    # Reserved even if the write fails, so a filename can never be overwritten.
    state.seen_ids.add(row_id)

    audio_path = (state.audio_out_dir / f"{row_id}.flac").resolve()
    try:
        sf.write(str(audio_path), array, samplerate, subtype=FLAC_SUBTYPE, format="FLAC")
    except _ROW_ERRORS as exc:
        state.skip_reasons["decode_error"] += 1
        _warn(state, "decode_error", f"cannot write {audio_path}: {exc}")
        return

    duration = round(seconds, 3)
    manifest_row = {
        "id": row_id,
        "text": PROMPT_TEXT,
        "answer": _answer_text(row),
        "audio": str(audio_path),
        "duration": duration,
    }
    state.manifest.write(json.dumps(manifest_row, ensure_ascii=False) + "\n")

    state.rows_written += 1
    state.total_duration += duration
    if state.min_written is None or duration < state.min_written:
        state.min_written = duration
    if state.max_written is None or duration > state.max_written:
        state.max_written = duration


def process_shard(shard_path: Path, state: RunState) -> None:
    """Stream one parquet shard into FLAC files and manifest rows."""
    try:
        for index, row in enumerate(_iter_parquet_rows(shard_path)):
            if state.done:
                break
            state.rows_seen += 1
            _process_row(row, f"{shard_path.stem}-{index}", state)
    except _ROW_ERRORS as exc:
        # A corrupt shard must never abort the whole run.
        state.skip_reasons["decode_error"] += 1
        _warn(state, "shard", f"cannot read shard {shard_path}: {exc}")


def resolve_shards(input_spec: str) -> list[Path]:
    """Expand one ``--input`` value into a sorted list of parquet shard paths.

    Accepts a directory (``*.parquet`` inside it, sorted), an explicit parquet
    file path, or a glob pattern.  Exits with code 1 when nothing matches.
    """
    path = Path(input_spec)
    if path.is_file():
        return [path]

    candidates: list[Path] = (
        [path] if path.is_dir() else sorted(Path(match) for match in glob.glob(input_spec))  # noqa: PTH207
    )

    shards: list[Path] = []
    for candidate in candidates:
        if candidate.is_dir():
            shards.extend(sorted(candidate.glob("*.parquet")))
        elif candidate.is_file() and candidate.suffix == ".parquet":
            shards.append(candidate)

    if not shards:
        print(f"ERROR: no *.parquet shards matched --input {input_spec}", file=sys.stderr)
        sys.exit(1)
    return shards


def process_shard_dir(input_spec: str, state: RunState) -> None:
    """Process one ``--input`` entry (shard directory, glob or parquet file)."""
    for shard_path in resolve_shards(input_spec):
        state.shards += 1
        process_shard(shard_path, state)
        if state.done:
            break


# ─────────────────────────────────────────────────────────────────────────────
# Reporting
# ─────────────────────────────────────────────────────────────────────────────


def print_summary(state: RunState) -> None:
    """Print the end-of-run summary as a single JSON line on stdout."""
    summary = {
        "shards": state.shards,
        "rows_seen": state.rows_seen,
        "rows_written": state.rows_written,
        "skipped": dict(sorted(state.skip_reasons.items())),
        "total_hours": round(state.total_duration / 3600.0, 2),
        "min_duration": state.min_written,
        "max_duration": state.max_written,
    }
    print(json.dumps(summary, ensure_ascii=False))


# ─────────────────────────────────────────────────────────────────────────────
# CLI plumbing
# ─────────────────────────────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    """Construct the argparse parser for a single input → single output run."""
    parser = argparse.ArgumentParser(
        prog="prepare_librispeech.py",
        description=(
            "Convert local openslr/librispeech_asr parquet shards into 16 kHz "
            "mono FLAC files plus a JSONL manifest."
        ),
    )
    parser.add_argument(
        "--input",
        dest="inputs",
        action="append",
        nargs="+",
        required=True,
        metavar="DIR_OR_GLOB",
        help=(
            "Directory of *.parquet shards, an explicit parquet file, or a glob "
            "pattern. Repeatable."
        ),
    )
    parser.add_argument(
        "--audio-out-dir",
        required=True,
        help="Directory for the decoded FLAC files (created if missing).",
    )
    parser.add_argument(
        "--manifest-out",
        required=True,
        help="Output JSONL manifest path (parent dirs created if missing).",
    )
    parser.add_argument(
        "--max-duration",
        type=float,
        default=25.0,
        help="Skip rows longer than this many seconds (default: 25.0).",
    )
    parser.add_argument(
        "--min-duration",
        type=float,
        default=0.0,
        help="Skip rows shorter than this many seconds (default: 0.0).",
    )
    parser.add_argument(
        "--id-prefix",
        default="",
        help="String prepended to every id (default: empty).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Stop after writing N valid rows (default: unlimited).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed; unused unless sampling is added (default: 0).",
    )
    return parser


def _validate_args(args: argparse.Namespace) -> str | None:
    """Validate CLI arguments; return an error message or ``None``."""
    if args.min_duration < 0:
        return "--min-duration must be >= 0"
    if args.max_duration < args.min_duration:
        return "--max-duration must be >= --min-duration"
    if args.limit is not None and args.limit < 0:
        return "--limit must be >= 0"
    return None


def main() -> None:
    """CLI entry point: one input set → one audio dir + one manifest."""
    parser = build_parser()
    try:
        args = parser.parse_args()
    except SystemExit as exc:
        # argparse exits 2 on usage errors; this script reserves 2 for
        # "zero rows written", so CLI/argument errors must exit 1.
        raise SystemExit(0 if exc.code in (0, None) else 1) from None

    error = _validate_args(args)
    if error is not None:
        print(f"ERROR: {error}", file=sys.stderr)
        sys.exit(1)

    random.seed(args.seed)  # deterministic if sampling is ever introduced.

    input_specs = [spec for group in args.inputs for spec in group]
    audio_out_dir = Path(args.audio_out_dir).expanduser().resolve()
    audio_out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = Path(args.manifest_out).expanduser()
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    with manifest_path.open("w", encoding="utf-8") as manifest:
        state = RunState(
            audio_out_dir=audio_out_dir,
            manifest=manifest,
            max_duration=args.max_duration,
            min_duration=args.min_duration,
            id_prefix=args.id_prefix,
            limit=args.limit,
        )
        for input_spec in input_specs:
            process_shard_dir(input_spec, state)
            if state.done:
                break

    print_summary(state)
    if state.rows_written == 0:
        sys.exit(2)


if __name__ == "__main__":
    main()
