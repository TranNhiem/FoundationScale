"""The data contracts FoundationScale owns at the upstream boundary.

WHY WE OWN THESE, measured on the current estate (2026-09/10):

1. The speech manifest. The conversion call in the NeMo lane wrote a manifest
   and then re-derived its statistics by opening every audio file with
   soundfile. That conflates TWO different questions -- "is this row well
   formed?" (pure, answerable from the JSON alone, cheap enough to run over
   6,000 rows before a GPU is even allocated) and "is this audio what the row
   claims?" (belongs to the worker that has the audio stack). Running the first
   question without soundfile is not a simplification; it is what lets the
   control plane validate manifests from campaigns it cannot play back.

2. Duplicate detection is a first-class counter because it has already burned
   us once: a builder bug left 6,000 rows on 820 unique recordings, and row
   coverage said 6,000/6,000. Every row loaded. Only two independent counters
   -- rows_read and distinct audio paths -- expose that. Duplicate ids and
   duplicate audio paths therefore get their own fields on the report, not a
   note in a log line.

3. The run config and its CLI projection. ``nemo_finetune.py``'s flags are
   argparse names living in a campaign script; a typo (``--train_manifest``)
   would be silently unread while argparse happily applied every other flag.
   ``to_argv`` produces the exact flag list from typed data, so the CLI surface
   is generated from one typed source and a config round-trips without drift.

4. Problem codes are STABLE STRINGS, not prose, because they are counted into
   reports and diffed between runs. An unrecognised manifest key is reported as
   ``unknown_key:<k>``: a misspelled ``duraton`` must be a counted problem, not
   a row that silently loses its duration. Every refusal is named and countable.

Nothing here reads audio, opens a model, or imports anything outside the stdlib.
Wrong input is a counted problem (reader) or a ``ValueError`` naming the key
(API); it is never a silent default.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

__all__ = [
    "SPEECH_MANIFEST_SCHEMA_VERSION",
    "ManifestReport",
    "SpeechRunConfig",
    "load_run_config",
    "manifest_row_problems",
    "read_speech_manifest",
    "to_argv",
    "to_nemo_aed_row",
]

SPEECH_MANIFEST_SCHEMA_VERSION = 1
"""Bumped whenever a row shape changes incompatibly. Readers record it so a
stale JSONL is refused as stale rather than half-interpreted."""

# Field vocabulary. Required keys are named in order, because the problem list
# for a row keeps that order and callers diff problem lists.
_REQUIRED_KEYS: tuple[str, ...] = ("id", "audio", "answer", "duration")
_OPTIONAL_KEYS: tuple[str, ...] = ("text", "source_id", "conversion", "segments")
_KNOWN_KEYS: frozenset[str] = frozenset(_REQUIRED_KEYS + _OPTIONAL_KEYS)

_PNC_VALUES: frozenset[str] = frozenset({"yes", "no"})
"""Canary's punctuation-and-capitals switch vocabulary. Two values, hard-coded
here because "True" and "auto" have both appeared in hand-written configs and
neither is a value NeMo's prompt formatter understands."""


def _is_number(value: Any) -> bool:
    """int or float, and explicitly NOT bool: ``True`` is an int in Python and
    would otherwise pass a duration check and survive as 1 second."""
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def manifest_row_problems(row: object) -> list[str]:
    """Every problem in one speech-manifest row, as stable codes.

    Codes (stable; counted, diffed and asserted against in tests):

    * ``missing:<k>``      -- a required key is absent.
    * ``bad_type:<k>``     -- a key is present but its value is of the wrong
      kind: a non-string ``id``/``audio``/``answer``/``text``/``source_id``, a
      ``bool`` or non-number ``duration``, a ``conversion`` that is neither
      ``str`` nor ``None``, a ``segments`` that is not an ``int >= 1``. Empty or
      whitespace-only ``id``/``audio`` are ``bad_type`` too: the field is
      declared non-empty, so emptiness is a failure of the declared type. The
      ``segments`` range (``>= 1``) is likewise PART of its declared type
      (``int >= 1``), so an out-of-range value reports ``bad_type:segments``
      rather than inventing a code the report vocabulary does not have.
    * ``empty_answer``     -- ``answer`` is a string but empty or whitespace-
      only. Kept apart from ``bad_type:answer`` because it is a DATA problem
      (the recording exists, the transcript is blank) and the workers count it
      under their own name for it.
    * ``bad_duration``     -- ``duration`` is a number of the right type but not
      usable: zero, negative, or not finite (``NaN``/``inf`` are real floats and
      would otherwise pass a ``> 0`` test by accident).
    * ``unknown_key:<k>``  -- a key this schema does not describe. Unknown keys
      are PROBLEMS, not trivia: a misspelled required key must surface as both
      ``missing:<k>`` and ``unknown_key:<misspelling>`` rather than being
      dropped on the floor.

    A row that is not a mapping at all (the reader sees whatever valid JSON a
    line holds) reports ``bad_type:row`` and nothing else. The function never
    raises: it is called per line over files we did not write.

    Ordering of the returned list is deterministic (required-key problems in
    declaration order, unknown keys sorted, then field checks in declaration
    order) so two runs' problem lists can be compared directly.
    """
    if not isinstance(row, Mapping):
        return ["bad_type:row"]

    problems: list[str] = []
    for key in _REQUIRED_KEYS:
        if key not in row:
            problems.append(f"missing:{key}")
    for key in sorted(k for k in row if k not in _KNOWN_KEYS):
        problems.append(f"unknown_key:{key}")

    if "id" in row:
        value = row["id"]
        if not isinstance(value, str) or not value.strip():
            problems.append("bad_type:id")
    if "audio" in row:
        value = row["audio"]
        if not isinstance(value, str) or not value.strip():
            problems.append("bad_type:audio")
    if "answer" in row:
        value = row["answer"]
        if not isinstance(value, str):
            problems.append("bad_type:answer")
        elif not value.strip():
            problems.append("empty_answer")
    if "duration" in row:
        value = row["duration"]
        if not _is_number(value):
            problems.append("bad_type:duration")
        elif not math.isfinite(float(value)) or float(value) <= 0.0:
            problems.append("bad_duration")
    if "text" in row and not isinstance(row["text"], str):
        problems.append("bad_type:text")
    if "source_id" in row and not isinstance(row["source_id"], str):
        problems.append("bad_type:source_id")
    if "conversion" in row:
        value = row["conversion"]
        if value is not None and not isinstance(value, str):
            problems.append("bad_type:conversion")
    if "segments" in row:
        value = row["segments"]
        if not _is_int(value) or value < 1:
            problems.append("bad_type:segments")
    return problems


@dataclass(frozen=True)
class ManifestReport:
    """What a manifest read found. Frozen: a report is a measurement.

    ``problems`` maps a problem code to how many rows reported it. Codes absent
    from the dict occurred zero times -- the dict is never padded with zeros,
    because ``report.problems.get(code, 0)`` already answers that and a padded
    dict would grow a key for every code anyone ever invents.
    """

    rows_read: int
    rows_valid: int
    problems: dict[str, int]
    duplicate_ids: int
    duplicate_audio_paths: int

    def __post_init__(self) -> None:
        for name in ("rows_read", "rows_valid", "duplicate_ids", "duplicate_audio_paths"):
            value = getattr(self, name)
            if not _is_int(value) or value < 0:
                raise ValueError(f"ManifestReport.{name} must be a non-negative int, got {value!r}")
        if self.rows_valid > self.rows_read:
            raise ValueError(
                f"ManifestReport claims {self.rows_valid} valid rows of {self.rows_read} read"
            )
        for code, count in self.problems.items():
            if not isinstance(code, str) or not code:
                raise ValueError(f"ManifestReport.problems has non-string code {code!r}")
            if not _is_int(count) or count < 0:
                raise ValueError(f"ManifestReport.problems[{code!r}] must be a non-negative int")
        if sum(self.problems.values()) == self.rows_read - self.rows_valid:
            pass
        elif sum(self.problems.values()) < self.rows_read - self.rows_valid:
            raise ValueError(
                "ManifestReport problem counts "
                f"{sum(self.problems.values())} cannot cover the "
                f"{self.rows_read - self.rows_valid} "
                "rows refused: every refused row must report at least one counted problem code"
            )

    @property
    def ok(self) -> bool:
        """True only when nothing at all was wrong: no problem code, no
        duplicate id, no duplicate audio path. A manifest with 5,900 valid rows
        and 100 counted problems is NOT ok -- campaigns run on it would be
        silently a different corpus from the one they were specified against."""
        return not self.problems and self.duplicate_ids == 0 and self.duplicate_audio_paths == 0

    def as_manifest(self) -> dict[str, Any]:
        """The JSON-serialisable form written alongside a coverage report.

        Deterministic: the problems dict is emitted sorted by code.
        """
        return {
            "schema_version": SPEECH_MANIFEST_SCHEMA_VERSION,
            "rows_read": self.rows_read,
            "rows_valid": self.rows_valid,
            "problems": dict(sorted(self.problems.items())),
            "duplicate_ids": self.duplicate_ids,
            "duplicate_audio_paths": self.duplicate_audio_paths,
            "ok": self.ok,
        }


def read_speech_manifest(path: str | Path) -> tuple[list[dict[str, Any]], ManifestReport]:
    """Read an FS speech manifest (JSONL) and VALIDATE it, without raising.

    Returns ``(valid_rows, report)``. Every JSONL line that is not a valid row
    is counted against a problem code on the report; the row is absent from the
    returned list. Bad rows never raise -- a campaign's manifest is exactly the
    kind of file that acquires one bad line from a shell redirect, and losing
    the other 5,999 rows to a traceback is not verification, it is an outage.
    A missing file still raises ``FileNotFoundError``: that is not a bad row, it
    is a missing input and must fail closed.

    Duplicate ids and duplicate audio paths are counted separately from
    per-row problems, and they are counted over every row whose ``id``/``audio``
    IS a usable non-empty string -- including rows the validator otherwise
    refuses. Reason: a collision must never hide behind another problem. The
    builder bug this exists for produced 6,000 transcript rows on 820
    recordings and every one of those rows was individually well formed;
    two counters -- rows_read vs rows_valid -- is the only place that shape of
    corruption is visible at all.

    ``duplicate_audio_paths == 1`` for three rows on two recordings (the second
    row on a repeated path is one collision), matching the campaign scripts'
    own counting.
    """
    manifest_path = Path(path)
    text = manifest_path.read_text(encoding="utf-8")  # FileNotFoundError propagates, by design

    problems: dict[str, int] = {}
    valid_rows: list[dict[str, Any]] = []
    rows_read = 0
    seen_ids: set[str] = set()
    seen_audio: set[str] = set()
    duplicate_ids = 0
    duplicate_audio_paths = 0

    def bump(code: str) -> None:
        problems[code] = problems.get(code, 0) + 1

    for line in text.splitlines():
        if not line.strip():
            continue  # blank lines (trailing newline, blank separators) are not rows
        rows_read += 1
        try:
            row = json.loads(line)
        except ValueError:
            bump("invalid_json")
            continue
        row_problems = manifest_row_problems(row)
        if isinstance(row, Mapping):
            row_id = row.get("id")
            if isinstance(row_id, str) and row_id.strip():
                if row_id in seen_ids:
                    duplicate_ids += 1
                else:
                    seen_ids.add(row_id)
            row_audio = row.get("audio")
            if isinstance(row_audio, str) and row_audio.strip():
                if row_audio in seen_audio:
                    duplicate_audio_paths += 1
                else:
                    seen_audio.add(row_audio)
        for code in row_problems:
            bump(code)
        if not row_problems:
            valid_rows.append(dict(row))

    report = ManifestReport(
        rows_read=rows_read,
        rows_valid=len(valid_rows),
        problems=problems,
        duplicate_ids=duplicate_ids,
        duplicate_audio_paths=duplicate_audio_paths,
    )
    return valid_rows, report


def to_nemo_aed_row(
    row: Mapping[str, Any],
    *,
    pnc: str,
    source_lang: str = "en",
    target_lang: str = "en",
) -> dict[str, Any]:
    """Translate one OWNED manifest row to a NeMo AED manifest row. Pure.

    This is the contract crossing, extracted from ``nemo_finetune.convert``
    WITHOUT the audio half of that function: soundfile's ``info`` is the
    worker's business (it needs the audio stack and the audio files), and
    making this translation depend on it would put a file read between two
    pieces of data. The duration written is the ROW's ``duration`` field; the
    worker re-measures it from the audio and refuses the row when they
    disagree, which is one check with two honest questions.

    Field mapping (fixed by the checkpoints' inherited ``train_ds``):
    ``audio_filepath`` = ``audio``, ``duration`` = ``duration``, ``text`` =
    ``answer`` (the NeMo example scripts write ``text`` while the checkpoint's
    default names the field ``answer``; the worker sets ``text_field``
    explicitly because inheriting it silently empties the targets), plus
    ``source_lang``/``target_lang``, ``taskname`` = ``asr``.

    ``pnc`` must be ``"yes"`` or ``"no"``; anything else is refused by name --
    a value like ``"auto"`` would reach the prompt formatter and produce a
    prompt format nothing scores.
    """
    if pnc not in _PNC_VALUES:
        raise ValueError(f"to_nemo_aed_row: pnc must be one of {sorted(_PNC_VALUES)}, got {pnc!r}")
    for name, value in (("source_lang", source_lang), ("target_lang", target_lang)):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"to_nemo_aed_row: {name} must be a non-empty string, got {value!r}")
    problems = manifest_row_problems(row)
    if problems:
        raise ValueError(
            "to_nemo_aed_row: refusing row with problems "
            + ", ".join(problems)
            + "; run read_speech_manifest first and translate only valid rows"
        )
    return {
        "audio_filepath": row["audio"],
        "duration": row["duration"],
        "text": row["answer"],
        "source_lang": source_lang,
        "target_lang": target_lang,
        "taskname": "asr",
        "pnc": pnc,
    }


@dataclass(frozen=True)
class SpeechRunConfig:
    """One speech fine-tune, as typed data.

    Fields and defaults mirror ``nemo_finetune.py``'s argparse surface exactly,
    including what a zero or None MEANS (``save_every=0``: no snapshots,
    ``seed=None``: unseeded, ``freeze=()``: nothing frozen). Those meanings are
    data, not CLI convention, so a config file and a command line agree by
    construction rather than by somebody remembering both.

    Frozen, and ``freeze`` is a tuple: a run config is the input to a
    reproduction claim and must not change under anyone's feet.
    """

    model: str
    train_manifest: str
    out_dir: str
    max_steps: int = 500
    batch_size: int = 8
    lr: float = 1e-5
    warmup: int = 50
    pnc: str = "no"
    max_duration: float = 25.0
    seed: int | None = None
    save_every: int = 0
    freeze: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        # Construction validates too: a config built in code must be as sound as
        # one loaded from a mapping, or the type would be a lie for one caller.
        for name in ("model", "train_manifest", "out_dir"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    f"SpeechRunConfig.{name} must be a non-empty string, got {value!r}"
                )
        if not _is_int(self.max_steps) or self.max_steps <= 0:
            raise ValueError(
                f"SpeechRunConfig.max_steps must be a positive int, got {self.max_steps!r}"
            )
        if not _is_int(self.batch_size) or self.batch_size <= 0:
            raise ValueError(
                f"SpeechRunConfig.batch_size must be a positive int, got {self.batch_size!r}"
            )
        if not _is_number(self.lr) or self.lr <= 0.0:
            raise ValueError(f"SpeechRunConfig.lr must be a positive number, got {self.lr!r}")
        if not _is_int(self.warmup) or self.warmup < 0:
            raise ValueError(
                f"SpeechRunConfig.warmup must be a non-negative int, got {self.warmup!r}"
            )
        if self.pnc not in _PNC_VALUES:
            raise ValueError(
                f"SpeechRunConfig.pnc must be one of {sorted(_PNC_VALUES)}, got {self.pnc!r}"
            )
        if not _is_number(self.max_duration) or self.max_duration <= 0.0:
            raise ValueError(
                f"SpeechRunConfig.max_duration must be a positive number, got {self.max_duration!r}"
            )
        if self.seed is not None and not _is_int(self.seed):
            raise ValueError(f"SpeechRunConfig.seed must be an int or None, got {self.seed!r}")
        if not _is_int(self.save_every) or self.save_every < 0:
            raise ValueError(
                f"SpeechRunConfig.save_every must be a non-negative int, got {self.save_every!r}"
            )
        freeze: object = self.freeze
        if isinstance(freeze, (str, bytes)) or not isinstance(freeze, (tuple, list)):
            # A bare string would iterate into one freeze prefix per CHARACTER
            # and freeze nothing, silently.
            raise ValueError(
                f"SpeechRunConfig.freeze must be a tuple of parameter-name prefixes, got {freeze!r}"
            )
        for prefix in freeze:
            if not isinstance(prefix, str) or not prefix.strip():
                raise ValueError(
                    f"SpeechRunConfig.freeze must contain non-empty strings, got {prefix!r}"
                )
        object.__setattr__(self, "freeze", tuple(freeze))
        object.__setattr__(self, "lr", float(self.lr))
        object.__setattr__(self, "max_duration", float(self.max_duration))


def load_run_config(mapping: Mapping[str, Any]) -> SpeechRunConfig:
    """Build a ``SpeechRunConfig`` from a plain mapping, refusing by name.

    Refusals (all ``ValueError``, all naming the offending key):

    * the mapping is not a mapping;
    * an UNKNOWN key is present -- every key is named, because a run config is
      often hand-edited and ``--max_duration`` silently unread is the failure
      this check exists to forbid;
    * a REQUIRED key (``model``, ``train_manifest``, ``out_dir``) is missing;
    * a value has the wrong type. ``bool`` is refused where an int is wanted:
      Python's ``True`` is an ``int`` and would otherwise become
      ``max_steps=1`` and run a one-step "fine-tune";
    * a value is out of range (non-positive steps/batch/lr/max_duration,
      negative warmup/save_every);
    * ``pnc`` is not ``"yes"``/``"no"``;
    * ``freeze`` is a bare string, or holds anything but non-empty strings.

    ``freeze`` may arrive as a list (what JSON gives you) and is stored as a
    tuple. Values are NOT coerced silently: only ``lr``/``max_duration`` widen
    int -> float, which is the documented numeric type of the field.
    """
    if not isinstance(mapping, Mapping):
        raise ValueError(
            f"load_run_config: expected a mapping of run-config keys, got {type(mapping).__name__}"
        )
    unknown = sorted(
        key for key in mapping if key not in {f for f in SpeechRunConfig.__dataclass_fields__}
    )
    if unknown:
        raise ValueError(
            "load_run_config: unknown key(s) "
            + ", ".join(repr(key) for key in unknown)
            + "; known keys are "
            + ", ".join(sorted(f for f in SpeechRunConfig.__dataclass_fields__))
        )
    missing = [key for key in ("model", "train_manifest", "out_dir") if key not in mapping]
    if missing:
        raise ValueError(
            "load_run_config: missing required key(s) " + ", ".join(repr(key) for key in missing)
        )

    values: dict[str, Any] = {}
    for key in ("model", "train_manifest", "out_dir"):
        value = mapping[key]
        if not isinstance(value, str) or not value.strip():
            raise ValueError(
                f"load_run_config: key {key!r} must be a non-empty string, "
                f"got {value!r} ({type(value).__name__})"
            )
        values[key] = value

    for key, minimum in (("max_steps", 1), ("batch_size", 1), ("warmup", 0), ("save_every", 0)):
        if key in mapping:
            value = mapping[key]
            if not _is_int(value):
                raise ValueError(
                    f"load_run_config: key {key!r} must be an int "
                    f"(bool is not an int here), got {value!r} ({type(value).__name__})"
                )
            if value < minimum:
                raise ValueError(
                    f"load_run_config: key {key!r} must be >= {minimum}, got {value!r}"
                )
            values[key] = value

    for key in ("lr", "max_duration"):
        if key in mapping:
            number = mapping[key]
            if not _is_number(number):
                raise ValueError(
                    f"load_run_config: key {key!r} must be a number "
                    f"(bool is not a number here), got {number!r} ({type(number).__name__})"
                )
            if not math.isfinite(float(number)) or number <= 0.0:
                raise ValueError(
                    f"load_run_config: key {key!r} must be a positive finite number, got {number!r}"
                )
            values[key] = float(number)

    if "pnc" in mapping:
        value = mapping["pnc"]
        if not isinstance(value, str) or value not in _PNC_VALUES:
            raise ValueError(
                f"load_run_config: key 'pnc' must be one of {sorted(_PNC_VALUES)}, "
                f"got {value!r} ({type(value).__name__})"
            )
        values["pnc"] = value

    if "seed" in mapping:
        value = mapping["seed"]
        if value is not None and not _is_int(value):
            raise ValueError(
                f"load_run_config: key 'seed' must be an int or None, "
                f"got {value!r} ({type(value).__name__})"
            )
        values["seed"] = value

    if "freeze" in mapping:
        value = mapping["freeze"]
        if isinstance(value, (str, bytes)) or not isinstance(value, (tuple, list)):
            raise ValueError(
                "load_run_config: key 'freeze' must be a list of parameter-name prefixes, "
                f"got {value!r} ({type(value).__name__}); a bare string would freeze one "
                "prefix per character"
            )
        for prefix in value:
            if not isinstance(prefix, str) or not prefix.strip():
                raise ValueError(
                    f"load_run_config: key 'freeze' must contain non-empty strings, got {prefix!r}"
                )
        values["freeze"] = tuple(value)

    return SpeechRunConfig(**values)


def to_argv(cfg: SpeechRunConfig) -> list[str]:
    """The ``nemo_finetune.py`` command line for one run config.

    Flag names are the EXACT argparse names of the campaign script (``--model``
    ``--train`` ``--out-dir`` ``--max-steps`` ``--batch-size`` ``--lr``
    ``--warmup`` ``--pnc`` ``--max-duration`` ``--seed`` ``--save-every``
    ``--freeze``) -- not the config's field names, not prettified: a worker
    subprocess is exec'd with this list and a renamed flag would be silently
    ignored by argparse while every other flag applied.

    Deterministic and total: ``to_argv(load_run_config(x))`` is byte-identical
    across processes (order comes from the dataclass fields, and numbers are
    rendered with ``str``), which is what makes a run reproducible from a
    recorded config alone.

    Emission rules mirror the script's own semantics: ``--seed`` is omitted
    entirely when the seed is ``None`` (the plant reads an absent flag as
    unseeded), ``--save-every`` is omitted when it is ``0`` (disabled), and each
    freeze prefix gets its OWN ``--freeze`` (the script declares it
    ``action="append"``; one flag per prefix, never a joined list).
    """
    if not isinstance(cfg, SpeechRunConfig):
        raise ValueError(f"to_argv: expected SpeechRunConfig, got {type(cfg).__name__}")
    argv = [
        "--model",
        cfg.model,
        "--train",
        cfg.train_manifest,
        "--out-dir",
        cfg.out_dir,
        "--max-steps",
        str(cfg.max_steps),
        "--batch-size",
        str(cfg.batch_size),
        "--lr",
        str(cfg.lr),
        "--warmup",
        str(cfg.warmup),
        "--pnc",
        cfg.pnc,
        "--max-duration",
        str(cfg.max_duration),
    ]
    if cfg.seed is not None:
        argv += ["--seed", str(cfg.seed)]
    if cfg.save_every:
        argv += ["--save-every", str(cfg.save_every)]
    for prefix in cfg.freeze:
        argv += ["--freeze", prefix]
    return argv
