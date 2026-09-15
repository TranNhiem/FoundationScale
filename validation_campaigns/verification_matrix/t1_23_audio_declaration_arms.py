#!/usr/bin/env python3
"""T1-23: "declaring audio REFUSES cleanly today" -- or is the field silently dropped?

Row claim
    declaring audio REFUSES cleanly today
Run arm
    audio corpus
Control arm
    "must be 96, not a silent drop of the audio field"

What is actually true of the shipped source, measured and not re-derived: the
string "audio" appears ZERO times in src/foundationscale/ -- there is no audio
surface. The text path tokenizes only ``batch["text"]`` and then calls
`` .map(..., remove_columns=columns)`` -- there is no audio surface. The text
path tokenizes only ``batch["text"]`` and then calls
``.map(..., remove_columns=columns)``, which deletes EVERY original column, so
an extra field -- audio included -- vanishes with no log line and no manifest
entry. A corpus with no "text" column at all is refused with exit 96 ("the
thin path requires a 'text' column", interpolating the columns it saw). The one
guarded non-text surface is an IMAGE column declared through
FOUNDATIONSCALE_TRAIN_IMAGE_COLUMN: if that declared column is absent from the
dataset, the trainer refuses with exit 96 and a message that NAMES the column.
The run manifest records cfg.dataset (the corpus PATH) and the full argv; it
does not record a column list.

So the interesting question is not "does it refuse when there is nothing to
train on" -- it does, for the missing text, not for the audio. The interesting
question is what happens when a corpus carries BOTH text and audio.

The three arms -- each an independent subprocess of the REAL entry point,
``python -m foundationscale.train.cli``, the thing under test being the thing
that ships, not an in-process reimplementation -- differ only in corpus and
environment; every trainer flag is held constant across arms:

    A audio_only     2 rows of {"audio": "clip0.wav"} -- no text column at all.
                     Today: exit 96 naming the columns it saw.
    B text_audio     2 rows of {"text": <sentence>, "audio": "clip0.wav"}.
                     Today: it trains. The question is whether ANYTHING tells
                     the user the audio field was discarded.
    C image_control  2 rows of {"text": <sentence>}, run with
                     FOUNDATIONSCALE_TRAIN_IMAGE_COLUMN set to a column name
                     that is NOT in the corpus. Today: exit 96 naming it.

WHY ARM C EXISTS
    Arm C is the POSITIVE CONTROL and the reason this file can conclude
    anything at all. Without it, arm B's silence has two explanations that
    cannot be told apart: the framework is silently dropping a declared
    modality, or this instrument simply cannot see a refusal. Arm C forces the
    framework to refuse over a declared non-text column and forces the
    instrument to observe it. If arm C does not refuse with a 96 naming its
    declared column, this file REFUSES with 96 rather than report a verdict on
    arm B: a control that does not fire, or fires for the wrong cause, cannot
    acquit a silence.

THE SELF-HIT TRAP
    Arm B's verdict turns on a substring scan for "audio" over the arm's
    combined output and its run manifest -- but the manifest records the corpus
    PATH and the full argv. Were the corpus file called corpus_audio.jsonl, or
    an output directory t23_text_audio/, the scanner would find its own needle
    and score the arm GREEN for the wrong reason: the framework said nothing,
    the harness said it. Therefore every path this file creates is arm_a /
    arm_b / arm_c and corpus.jsonl -- no case-insensitive "audio" anywhere --
    argv cleanliness is ASSERTED before the first arm runs and again when the
    payload is built, a dirty argv is 96 CANNOT-MEASURE (the measurement would
    be reading its own fingerprint, not a warning), and the record VALUE is
    clip0.wav so that only the record KEY ("audio") carries the word and only
    the framework can put it in a log line. The image-column environment
    variable is set for arm C from a COPY of os.environ and explicitly REMOVED
    for arms A and B, so an operator who happens to have it exported cannot
    change what the row measures.

VERDICT ORDER (load-bearing; pinned by --self-test controls, not comments)
    1   an arm absent, or its rc None                    -> 95 UNMEASURED
    2   an arm with argv_clean False                     -> 96 REFUSE (names it)
    3   image_control rc != 96                           -> 96 REFUSE
    4   image_control 96 but output lacks its column     -> 96 REFUSE
    5   audio_only rc == 0                               -> 5 RED
    6   audio_only rc not in (0, 96)                     -> 96 REFUSE
    7   audio_only 96 naming neither requirement         -> 96 REFUSE
    8   text_audio rc == 96                              -> 0 PASS (claim true)
    9   text_audio rc not in (0, 96)                     -> 96 REFUSE
    10  text_audio rc == 0, tokenized None or 0          -> 96 REFUSE
    11  text_audio rc == 0, tokenized > 0, "audio" said  -> 0 PASS
    12  text_audio rc == 0, tokenized > 0, never said    -> 5 RED (the drop)

WHAT THIS FILE DOES NOT MEASURE
    It measures the declaration surface of the shipped text path on a small
    model -- not whether some other entry point elsewhere in the framework
    handles audio, and not whether the framework ought to support audio at
    all. A subprocess timeout is a real outcome, recorded as rc=None (rule 1
    turns it into 95); it is never swallowed.

EXIT CONTRACT
    0 claim upheld, 5 the control observation refuted it, 95 ran but the
    deciding quantity could not be measured, 96 refused to run at all. Never
    1 or 2: argparse's error() is overridden to exit 96. The verdict function
    is pure -- no I/O, no torch, no datasets -- and --self-test drives it over
    synthetic payloads under a bare interpreter (``python3 -S -E``, no
    site-packages): nothing outside the standard library is imported at module
    scope, and torch is never imported in this process at all. No GPU is
    probed -- this row does not need one, and refusing over a GPU it does not
    use would be a false 96.
"""

from __future__ import annotations

import argparse
import json
import os  # stdlib only: the self-test runs under `python3 -S`
import re
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any, NoReturn

# The row directory is a sibling import root, exactly as `python3 t1_23_...py`
# gives it. This row already abstained on an escape, but collapsed every escape
# to 96 -- so a dead CUDA link read as a harness bug. The shared classifier
# keeps the environment/harness split the #417 rule is built on.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from t1_interpreter_floor import classify_boundary_exception  # noqa: E402

GREEN = 0
RED = 5
UNMEASURED = 95
REFUSE = 96

ROW_ID = "T1-23"
PROG = "t1_23_audio_declaration_arms"
CLAIM = "declaring audio REFUSES cleanly today"
CONTROL = "must be 96, not a silent drop of the audio field"

VERDICT_NAMES = {GREEN: "GREEN", RED: "RED", UNMEASURED: "UNMEASURED", REFUSE: "REFUSE"}

# The scan needle, lowercase. Every path this file creates is checked against
# it; only corpus record KEYS may carry it.
NEEDLE = "audio"

# Arms, in verdict order, mapped to directory names free of the needle.
ARM_NAMES: tuple[str, ...] = ("audio_only", "text_audio", "image_control")
ARM_DIRS: dict[str, str] = {
    "audio_only": "arm_a",
    "text_audio": "arm_b",
    "image_control": "arm_c",
}
CORPUS_FILENAME = "corpus.jsonl"
MANIFEST_GLOB = "*manifest*.json"

IMAGE_COLUMN_ENV = "FOUNDATIONSCALE_TRAIN_IMAGE_COLUMN"
DECLARED_IMAGE_COLUMN = "t23_img_probe"

# Common trainer flags, held constant so arms differ only in corpus and env.
# --dp: the trainer's topology requires the parallel-degree product to equal
# nodes x gpus_per_node; this row declares no other degree, so dp IS the world
# size -- whatever the optimizer-arms row passes for the same shape.
MAX_STEPS = 2
PER_DEVICE_BATCH_SIZE = 1
NODES = 1
GPUS_PER_NODE = 1
SEED = 42
SAVE_INTERVAL = 1000

# A small model and two steps is minutes, not hours. A timeout is a real
# outcome -- rc=None in the payload -- never an exception to swallow.
ARM_TIMEOUT_S = 3600

_TOKENIZED_RE = re.compile(r"(\d+)\s+examples tokenized")


# ---------------------------------------------------------------------------
# Import bootstrap (same helper shape as the optimizer-arms row)
# ---------------------------------------------------------------------------


def _repo_src_root(start: Path) -> Path | None:
    """Walk upward from ``start`` for the ``src/`` of a src-layout checkout, or None.

    The marker is ``src/foundationscale/__init__.py`` -- the package's own module file,
    not a directory that merely happens to be called ``src``.
    """
    for parent in [start, *start.parents]:
        candidate = parent / "src"
        if (candidate / "foundationscale" / "__init__.py").is_file():
            return candidate
    return None


def _ensure_package_importable() -> str | None:
    """Make ``foundationscale`` importable; return the reason it is not, or None.

    An installed distribution cannot be assumed, so this falls back to the
    checkout's own ``src/``. The found root is ALSO exported through PYTHONPATH,
    because the trainer runs as a subprocess and ``sys.path`` does not cross a
    process boundary.
    """
    try:
        import foundationscale  # noqa: F401  -- probing importability, not using it
    except ImportError:
        pass
    else:
        return None
    here = Path(__file__).resolve().parent
    src = _repo_src_root(here)
    if src is None:
        return f"no src/foundationscale/__init__.py in any ancestor of {here}"
    sys.path.insert(0, os.fspath(src))
    try:
        import foundationscale  # noqa: F401  -- probing importability, not using it
    except ImportError as exc:
        return f"{src} is on sys.path and `import foundationscale` still failed: {exc}"
    existing = os.environ.get("PYTHONPATH")
    os.environ["PYTHONPATH"] = (
        os.fspath(src) if not existing else os.fspath(src) + os.pathsep + existing
    )
    return None


# ---------------------------------------------------------------------------
# Arms: corpora, flags, and the no-self-hit assertion
# ---------------------------------------------------------------------------


def _argv_dirty_count(argv: list[str]) -> int:
    """How many argv elements carry the needle, case-insensitively. Must be 0."""
    return sum(1 for element in argv if NEEDLE in element.lower())


def _corpus_rows(arm: str) -> list[dict[str, str]]:
    """Two JSON Lines rows for ``arm``.

    The audio field's VALUE is clip0.wav -- never the word itself -- so only
    the KEY carries the needle, and only the framework can put it in a log
    line. The text sentences are short, declarative, and needle-free.
    """
    text_0 = "A red kite crossed the valley."
    text_1 = "The kettle rattled on the stove."
    if arm == "audio_only":
        return [{NEEDLE: "clip0.wav"}, {NEEDLE: "clip1.wav"}]
    if arm == "text_audio":
        return [
            {"text": text_0, NEEDLE: "clip0.wav"},
            {"text": text_1, NEEDLE: "clip1.wav"},
        ]
    return [{"text": text_0}, {"text": text_1}]


def _arm_flags(args: argparse.Namespace, arm_dir: Path, corpus: Path) -> dict[str, str]:
    """One arm's trainer invocation as flag -> value, constant across arms."""
    flags = {
        "--model": args.model,
        "--dataset": os.fspath(corpus),
        "--output-dir": os.fspath(arm_dir),
        "--max-steps": str(MAX_STEPS),
        "--per-device-batch-size": str(PER_DEVICE_BATCH_SIZE),
        "--nodes": str(NODES),
        "--gpus-per-node": str(GPUS_PER_NODE),
        "--dp": str(NODES * GPUS_PER_NODE),
        "--seed": str(SEED),
        "--save-interval": str(SAVE_INTERVAL),
    }
    if args.profile_name is not None:
        flags["--profile-name"] = args.profile_name
    else:
        flags["--profile-path"] = os.fspath(args.profile_path)
    return flags


def _prepare_arms(args: argparse.Namespace, work_dir: Path) -> dict[str, tuple[Path, list[str]]]:
    """Materialise every corpus and argv BEFORE the first subprocess starts.

    The self-hit assertion runs here -- before running anything -- because an
    argv carrying the needle would make arm B's scan read its own fingerprint.
    A failure aborts the run before a single trainer is launched.
    """
    prepared: dict[str, tuple[Path, list[str]]] = {}
    for name in ARM_NAMES:
        arm_dir = work_dir / ARM_DIRS[name]
        arm_dir.mkdir(parents=True, exist_ok=True)
        corpus = arm_dir / CORPUS_FILENAME
        corpus.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in _corpus_rows(name))
        )
        cmd = [sys.executable, "-m", "foundationscale.train.cli"]
        for flag, value in _arm_flags(args, arm_dir, corpus).items():
            cmd.extend([flag, value])
        assert _argv_dirty_count(cmd) == 0, (
            f"{name}: argv carries the '{NEEDLE}' needle ({cmd}); "
            "arm B's scan would match its own fingerprint"
        )
        prepared[name] = (arm_dir, cmd)
    return prepared


# ---------------------------------------------------------------------------
# Running one arm and reading what it left behind
# ---------------------------------------------------------------------------


def _as_text(stream: Any) -> str:
    """Coerce a captured stream (str, bytes or None) to text."""
    if stream is None:
        return ""
    if isinstance(stream, bytes):
        return stream.decode("utf-8", "replace")
    return stream


def _read_manifest_text(arm_dir: Path) -> str:
    """Concatenated text of every *manifest*.json under ``arm_dir``, or ""."""
    parts: list[str] = []
    for path in sorted(arm_dir.rglob(MANIFEST_GLOB)):
        if not path.is_file():
            continue
        try:
            parts.append(path.read_text(errors="replace"))
        except OSError:
            continue
    return "\n".join(parts)


def _parse_tokenized(output: str) -> int | None:
    """The integer from the "N examples tokenized" line, or None if it never appeared."""
    match = _TOKENIZED_RE.search(output)
    return int(match.group(1)) if match is not None else None


def _run_arm(name: str, arm_dir: Path, cmd: list[str]) -> dict[str, Any]:
    """Run one arm as an independent subprocess and project the outcome.

    ``output`` is stdout merged with stderr (true interleave is unrecoverable
    from two separately captured pipes, so stdout is kept ahead of stderr with
    a newline boundary between them). Arm C's environment is a COPY of
    os.environ with the image-column variable set; arms A and B get the same
    copy with the variable explicitly REMOVED, so an exported operator setting
    cannot change what the row measures.
    """
    env = dict(os.environ)
    if name == "image_control":
        env[IMAGE_COLUMN_ENV] = DECLARED_IMAGE_COLUMN
    else:
        env.pop(IMAGE_COLUMN_ENV, None)
    rc: int | None
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=ARM_TIMEOUT_S)
        rc = proc.returncode
        output = proc.stdout
        if proc.stderr:
            if output and not output.endswith("\n"):
                output += "\n"
            output += proc.stderr
    except subprocess.TimeoutExpired as exc:
        rc = None
        output = (
            f"arm timed out after {ARM_TIMEOUT_S}s; recorded as rc=None (rule 1)\n"
            + _as_text(exc.output)
            + _as_text(exc.stderr)
        )
    entry: dict[str, Any] = {
        "rc": rc,
        "output": output,
        "manifest_text": _read_manifest_text(arm_dir),
        "tokenized": _parse_tokenized(output),
        "argv_clean": _argv_dirty_count(cmd) == 0,
        "argv": list(cmd),
    }
    if name == "image_control":
        entry["declared_column"] = DECLARED_IMAGE_COLUMN
    return entry


def _build_payload(arms: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """The JSON-shaped payload the verdict judges. Re-asserts the no-hit invariant."""
    for name in ARM_NAMES:
        assert _argv_dirty_count(list(arms[name]["argv"])) == 0, (
            f"{name}: payload argv carries the '{NEEDLE}' needle; the scan would "
            "match its own fingerprint"
        )
    return {"row": ROW_ID, "claim": CLAIM, "control_arm": CONTROL, "arms": dict(arms)}


# ---------------------------------------------------------------------------
# The verdict -- PURE (payload in -> (rc, reason) out). The only verdict site.
# ---------------------------------------------------------------------------


def audio_declaration_verdict(payload: Mapping[str, Any]) -> tuple[int, str]:
    """Adjudicate the row from payload fields only. No I/O, no torch, no datasets.

    Decision order is load-bearing and pinned by --self-test controls; see the
    module docstring. Every reason is one line, names its arm, and carries the
    number it turned on; none carries a pipe character (these strings are
    quoted into a markdown table).
    """
    raw_arms = payload.get("arms")
    arms: Mapping[str, Any] = raw_arms if isinstance(raw_arms, Mapping) else {}
    entries: dict[str, Mapping[str, Any]] = {}

    # Rule 1: an arm that did not run, or never returned an exit code, cannot
    # report a null.
    for name in ARM_NAMES:
        entry = arms.get(name)
        if not isinstance(entry, Mapping):
            return UNMEASURED, (
                f"{name}: absent from the payload; an arm that did not run cannot report a null"
            )
        entries[name] = entry
        if entry.get("rc") is None:
            return UNMEASURED, (
                f"{name}: rc is None -- the arm never returned an exit code, and an "
                "arm that did not run cannot report a null"
            )

    # Rule 2: a dirty argv means the scan would match its own needle. That is
    # CANNOT-MEASURE, not a result.
    for name in ARM_NAMES:
        if not entries[name].get("argv_clean"):
            argv = entries[name].get("argv") or []
            offenders = [str(e) for e in argv if NEEDLE in str(e).lower()]
            first = offenders[0] if offenders else "<no argv recorded>"
            return REFUSE, (
                f"{name}: argv_clean is False -- argv element {first!r} carries the "
                f"'{NEEDLE}' needle; the scan would match its own fingerprint"
            )

    # Rules 3-4: the positive control must refuse, and must refuse naming the
    # column it declared -- otherwise a silence in text_audio is uninterpretable.
    control = entries["image_control"]
    rc_c = control.get("rc")
    out_c = str(control.get("output") or "")
    declared = control.get("declared_column")
    if rc_c != REFUSE:
        return REFUSE, (
            f"image_control: exited {rc_c}, expected 96 -- the positive control did "
            "not fire, so a silence in text_audio is uninterpretable"
        )
    if not isinstance(declared, str) or not declared or declared not in out_c:
        return REFUSE, (
            f"image_control: exited 96 but its output never names its declared "
            f"column {declared!r} -- it refused for some other reason, and a "
            "control that fires for the wrong cause is not one"
        )

    # Rules 5-7: the audio-only corpus must be refused, and the refusal must be
    # about the corpus this arm wrote, not about something unrelated.
    only = entries["audio_only"]
    rc_a = only.get("rc")
    out_a = str(only.get("output") or "")
    if rc_a == GREEN:
        return RED, (
            "audio_only: exited 0 -- a corpus whose only non-trivial field is "
            f"'{NEEDLE}' trained anyway; the declaration was refused nowhere"
        )
    if rc_a != REFUSE:
        return REFUSE, (
            f"audio_only: exited {rc_a}, neither 0 nor 96 -- the arm failed for a "
            "reason that is not a refusal, and a crash measures nothing"
        )
    names_text_requirement = "'text' column" in out_a
    names_audio_column = NEEDLE in out_a.lower()
    if not (names_text_requirement or names_audio_column):
        return REFUSE, (
            "audio_only: exited 96 but the output names neither the 'text' column "
            f"requirement nor the {NEEDLE} column -- a refusal about something "
            "unrelated to the corpus this arm carried"
        )

    # Rules 8-12: the deciding arm. A refusal is the claim literally true; a
    # training run is judged on whether it SAID the field went away.
    both = entries["text_audio"]
    rc_b = both.get("rc")
    if rc_b == REFUSE:
        return GREEN, (
            "text_audio: exited 96 -- the framework refused the corpus carrying "
            "both text and audio; the row's claim, literally true"
        )
    if rc_b != GREEN:
        return REFUSE, (
            f"text_audio: exited {rc_b}, neither 0 nor 96 -- a crash measures "
            "nothing, and no verdict on the audio field can be read from it"
        )
    tokenized = both.get("tokenized")
    if not isinstance(tokenized, int) or tokenized == 0:
        return REFUSE, (
            f"text_audio: exited 0 with tokenized={tokenized} -- it 'succeeded' "
            "without tokenizing anything, so its silence is the silence of an "
            "empty set, not of a dropped field"
        )
    haystack = (str(both.get("output") or "") + "\n" + str(both.get("manifest_text") or "")).lower()
    if NEEDLE in haystack:
        return GREEN, (
            f"text_audio: exited 0, {tokenized} examples tokenized, and the run "
            f"SAYS the {NEEDLE} field went away (in its output or its manifest) "
            "-- a documented drop is not a silent one"
        )
    return RED, (
        f"text_audio: exited 0, {tokenized} examples tokenized, and neither a log "
        f"line nor a manifest entry mentions the {NEEDLE} field the corpus "
        "carried -- the silent drop"
    )


# ---------------------------------------------------------------------------
# Self-test: named controls driving the pure verdict on synthetic payloads.
# Runs under a bare interpreter: standard library only, no GPU, no trainer.
# ---------------------------------------------------------------------------


def _synthetic_baseline() -> dict[str, Any]:
    """A fully healthy payload: control refuses naming its column, audio_only
    refuses naming the requirement, text_audio trains and SAYS the field dropped."""
    argv = ["python3", "-m", "foundationscale.train.cli", "--dataset", "corpus.jsonl"]
    return {
        "row": ROW_ID,
        "synthetic": True,
        "arms": {
            "audio_only": {
                "rc": 96,
                "output": (
                    "REFUSED: the thin path requires a 'text' column; "
                    "dataset columns seen: ['audio']"
                ),
                "manifest_text": "",
                "tokenized": None,
                "argv_clean": True,
                "argv": list(argv),
            },
            "text_audio": {
                "rc": 0,
                "output": (
                    "2 examples tokenized\ndataset columns: ['text', 'audio']; training on 'text'"
                ),
                "manifest_text": "",
                "tokenized": 2,
                "argv_clean": True,
                "argv": list(argv),
            },
            "image_control": {
                "rc": 96,
                "output": (
                    f"REFUSED: image column '{DECLARED_IMAGE_COLUMN}' is declared "
                    "but dataset has no such column"
                ),
                "manifest_text": "",
                "tokenized": None,
                "argv_clean": True,
                "argv": list(argv),
                "declared_column": DECLARED_IMAGE_COLUMN,
            },
        },
    }


def _arms(
    *,
    a: dict[str, Any] | None = None,
    b: dict[str, Any] | None = None,
    c: dict[str, Any] | None = None,
    drop: tuple[str, ...] = (),
) -> dict[str, Any]:
    """A fresh baseline payload with per-arm field overrides applied."""
    base = _synthetic_baseline()["arms"]
    for name, override in (("audio_only", a), ("text_audio", b), ("image_control", c)):
        if override:
            base[name].update(override)
    for name in drop:
        del base[name]
    return {"arms": base}


def _self_test() -> int:
    silent_run = {
        "rc": 0,
        "output": "2 examples tokenized\ntrain_loss 1.9302",
        "manifest_text": "",
        "tokenized": 2,
    }
    dirty_b = {
        **_arms(b={"argv_clean": False, "argv": ["python3", "--dataset", "corpus_audio.jsonl"]})[
            "arms"
        ]["text_audio"],
        "output": "2 examples tokenized",
        "manifest_text": "",
    }
    controls: list[tuple[str, str, dict[str, Any], int, tuple[str, ...]]] = [
        (
            "SC1",
            "everything healthy, text_audio mentions audio",
            _synthetic_baseline(),
            GREEN,
            ("text_audio",),
        ),
        (
            "SC2",
            "the silent drop (RED must fire, count quoted)",
            _arms(
                b={
                    "rc": 0,
                    "output": "2 examples tokenized\ntrain_loss 1.9302",
                    "manifest_text": "",
                    "tokenized": 2,
                }
            ),
            RED,
            ("text_audio", "2"),
        ),
        (
            "SC3",
            "audio_only trained anyway, rc 0 (RED must fire)",
            _arms(a={"rc": 0, "output": "2 examples tokenized", "tokenized": 2}),
            RED,
            ("audio_only",),
        ),
        (
            "SC4",
            "an arm absent from the payload (UNMEASURED must fire)",
            _arms(drop=("text_audio",)),
            UNMEASURED,
            ("text_audio",),
        ),
        (
            "SC5",
            "an arm present with rc None (UNMEASURED must fire)",
            _arms(b={"rc": None}),
            UNMEASURED,
            ("text_audio",),
        ),
        (
            "SC6",
            "a dirty argv on an arm (REFUSE must fire, naming it)",
            _arms(a={"argv_clean": False, "argv": ["python3", "corpus_audio.jsonl"]}),
            REFUSE,
            ("audio_only",),
        ),
        (
            "SC7",
            "image_control rc 0: the positive control did not fire (REFUSE)",
            _arms(c={"rc": 0, "output": "2 examples tokenized", "tokenized": 2}),
            REFUSE,
            ("image_control",),
        ),
        (
            "SC8",
            "image_control rc 5: not the expected refusal (REFUSE)",
            _arms(c={"rc": 5}),
            REFUSE,
            ("image_control",),
        ),
        (
            "SC9",
            "image_control refused without naming its column (REFUSE)",
            _arms(c={"output": "REFUSED: profile section missing from config"}),
            REFUSE,
            ("image_control", DECLARED_IMAGE_COLUMN),
        ),
        (
            "SC10",
            "audio_only rc 1: a crash is not a refusal (REFUSE)",
            _arms(a={"rc": 1, "output": "Traceback (most recent call last):"}),
            REFUSE,
            ("audio_only",),
        ),
        (
            "SC11",
            "text_audio rc 1: a crash measures nothing (REFUSE)",
            _arms(b={"rc": 1, "output": "Traceback (most recent call last):"}),
            REFUSE,
            ("text_audio",),
        ),
        (
            "SC12",
            "text_audio refused, rc 96: the claim, literally true (GREEN)",
            _arms(b={"rc": 96, "output": "modality column 'audio' has no loader"}),
            GREEN,
            ("text_audio",),
        ),
        (
            "SC13",
            "text_audio rc 0, tokenized 0, no mention: 96, NOT 5 (order pin)",
            _arms(b={"rc": 0, "output": "0 examples tokenized", "tokenized": 0}),
            REFUSE,
            ("text_audio", "0"),
        ),
        (
            "SC14",
            "dirty argv AND the silent drop: 96 rule 2 beats rule 12 (order pin)",
            _arms(b=dirty_b),
            REFUSE,
            ("text_audio",),
        ),
        (
            "SC15",
            "an arm missing AND image_control broken: 95, rule 1 first (order pin)",
            _arms(c={"rc": 0, "output": "2 examples tokenized"}, drop=("audio_only",)),
            UNMEASURED,
            ("audio_only",),
        ),
        (
            "SC16",
            "mention present but tokenized 0: 96, rule 10 beats rule 11 (order pin)",
            _arms(b={"rc": 0, "output": "0 examples tokenized; ignored 'audio'", "tokenized": 0}),
            REFUSE,
            ("text_audio",),
        ),
        (
            "SC17",
            "the mention is capitalised 'Audio' (case pin, GREEN must fire)",
            _arms(b={"output": "2 examples tokenized\nnote: Audio stream not loaded"}),
            GREEN,
            ("text_audio",),
        ),
        (
            "SC18",
            "audio_only rc 96 whose message is a missing GPU (rule 7 REFUSE)",
            _arms(a={"rc": 96, "output": "REFUSED: no accelerator device visible"}),
            REFUSE,
            ("audio_only",),
        ),
        (
            "SC19",
            "tokenized line never appeared: 96 on None, not on 0 (rule 10 pin)",
            _arms(b={"rc": 0, "output": "run complete", "tokenized": None}),
            REFUSE,
            ("text_audio",),
        ),
        (
            "SC20",
            "audio_only's refusal names the audio column alone (rule 7 passes it)",
            _arms(
                a={"rc": 96, "output": "REFUSED: dataset column 'audio' has no loader"},
                b={"rc": 96, "output": "modality column 'audio' has no loader"},
            ),
            GREEN,
            ("text_audio",),
        ),
        (
            "SC21",
            "the mention is in the manifest, not the output (rule 11 manifest pin)",
            _arms(
                b={
                    "output": "2 examples tokenized",
                    "manifest_text": '{"dropped_columns": ["audio"]}',
                }
            ),
            GREEN,
            ("text_audio",),
        ),
        (
            "SC22",
            "a dirty argv on image_control is still rule 2 (any-arm pin)",
            _arms(c={"argv_clean": False, "argv": ["python3", "--output-dir", "t23_audio_run"]}),
            REFUSE,
            ("image_control",),
        ),
    ]
    del silent_run  # the shape lives in SC2; the name was documentation

    passed = 0
    for control_id, desc, payload, want, mentions in controls:
        got, reason = audio_declaration_verdict(payload)
        ok = (
            got == want
            and all(fragment in reason for fragment in mentions)
            and "|" not in reason
            and "\n" not in reason
            and ("UNMEASURED" not in reason or want == UNMEASURED)
        )
        if ok:
            passed += 1
        print(f"[{'PASS' if ok else 'FAIL'}] {control_id} {desc} rc={got} want={want}")
    print(f"{ROW_ID} self-test: {passed}/{len(controls)} controls PASS")
    return GREEN if passed == len(controls) else RED


# ---------------------------------------------------------------------------
# The run
# ---------------------------------------------------------------------------


def _run(args: argparse.Namespace) -> int:
    if (args.profile_name is None) == (args.profile_path is None):
        print("REFUSED: exactly one of --profile-name / --profile-path is required")
        return REFUSE
    reason = _ensure_package_importable()
    if reason is not None:
        print(f"REFUSED: foundationscale is not importable -- {reason}")
        return REFUSE

    work_dir: Path = args.work_dir
    work_dir.mkdir(parents=True, exist_ok=True)
    prepared = _prepare_arms(args, work_dir)

    print(f"claim:        {CLAIM}")
    print(f"control arm:  {CONTROL}")
    print(
        f"ground truth: '{NEEDLE}' appears zero times in src/foundationscale; "
        "arm C ({DECLARED_IMAGE_COLUMN}) is the positive control".replace(
            "DECLARED_IMAGE_COLUMN", DECLARED_IMAGE_COLUMN
        )
    )

    arms: dict[str, dict[str, Any]] = {}
    for name in ARM_NAMES:
        arm_dir, cmd = prepared[name]
        entry = _run_arm(name, arm_dir, cmd)
        arms[name] = entry
        rc_shown = "-" if entry["rc"] is None else str(entry["rc"])
        print(
            f"ARM {name:<14} rc={rc_shown:<4} tokenized={entry['tokenized']} "
            f"argv_clean={entry['argv_clean']}",
            flush=True,
        )

    payload = _build_payload(arms)
    if args.out is not None:
        args.out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")

    code, why = audio_declaration_verdict(payload)
    print("=" * 72)
    print(f"{ROW_ID} VERDICT {VERDICT_NAMES[code]}: {why}")
    if args.out is not None:
        print(f"payload: {args.out}")
    return code


# ---------------------------------------------------------------------------
# Argument parsing -- argparse's exit 2 is out of contract
# ---------------------------------------------------------------------------


class _RefusingArgumentParser(argparse.ArgumentParser):
    """ArgumentParser whose errors REFUSE (96) instead of argparse's default 2."""

    def error(self, message: str) -> NoReturn:
        sys.stderr.write(f"REFUSE({REFUSE}): {message}\n")
        raise SystemExit(REFUSE)


def _build_parser() -> argparse.ArgumentParser:
    parser = _RefusingArgumentParser(
        prog=PROG,
        description=(
            "T1-23 verification row: three corpora, one declaration surface, one "
            "positive control. Exit codes: 0 claim upheld, 5 the observation "
            "refuted it, 95 UNMEASURED, 96 REFUSE."
        ),
    )
    parser.add_argument(
        "--model",
        default=None,
        help="passed straight through to the trainer (required for a run; no defensible default)",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=None,
        help="directory for the arm_a / arm_b / arm_c corpora, trainer output "
        "and manifests (required for a run; no defensible default)",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--profile-name", default=None, help="passed through to the trainer")
    group.add_argument(
        "--profile-path", type=Path, default=None, help="passed through to the trainer"
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="optional path for the JSON payload the verdict was read from",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="drive audio_declaration_verdict over synthetic payloads; "
        "bare-interpreter safe, no other args needed",
    )
    return parser


# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.self_test:
        return _self_test()

    missing = [
        flag
        for flag, value in (("--model", args.model), ("--work-dir", args.work_dir))
        if value is None
    ]
    if missing:
        sys.stderr.write(
            f"REFUSE({REFUSE}): required argument(s) with no defensible default "
            f"missing: {', '.join(missing)} (or pass --self-test)\n"
        )
        return REFUSE

    try:
        return _run(args)
    except Exception as exc:  # noqa: BLE001 -- an escape is classified, never adjudicated
        import traceback

        traceback.print_exc()
        code, why = classify_boundary_exception(exc)
        name = "UNMEASURED" if code == UNMEASURED else "CANNOT_MEASURE"
        sys.stderr.write(
            f"{ROW_ID} VERDICT {name}: unexpected {type(exc).__name__} escaped the "
            f"run body: {exc}; classified {name} ({code}): {why}; refusing to "
            "adjudicate a measurement that cannot account for itself\n"
        )
        return code


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001 - never 1, never 2, never 5
        import traceback

        traceback.print_exc()
        _code, _why = classify_boundary_exception(exc)
        sys.exit(_code)
