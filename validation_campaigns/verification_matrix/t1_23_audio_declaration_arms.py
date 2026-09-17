#!/usr/bin/env python3
"""T1-23: declaring audio must refuse by name; merely carrying it must not.

Row claim
    declaring audio REFUSES cleanly and NAMES the modality
Run arm
    a text+audio corpus with FOUNDATIONSCALE_TRAIN_AUDIO_COLUMN set
Control arm
    the same corpus with nothing declared must TRAIN, and the console must name
    the dropped audio column; refusing it would be name-sniffing

Subject change measured here: fix #490 gave src/foundationscale/train/loop.py a
real undeclared-modality surface. FOUNDATIONSCALE_TRAIN_AUDIO_COLUMN and
FOUNDATIONSCALE_TRAIN_VIDEO_COLUMN are module-level declarations. An EMPTY env
value reads as undeclared. Immediately after the image declaration is read, and
before the model, tokenizer, or dataset are touched, a non-empty declaration
refuses with exit 96, emits [fs:train:refuse], and writes a refused run
manifest carrying extra.exit and extra.<modality>_column. The refusal text
names the modality, the environment variable, the declared column, and "text"
as what this plane can train. It is deliberately NOT the old "requires a 'text'
column" failure.

In the text-only arm -- no image column declared -- tokenization still trains on
batch["text"] exactly as before. What changed is that a non-text dataset column
is now named before tokenization: a [fs:train:data] line says text is being
tokenized and that columns such as ['audio'] are dropped and contribute nothing
to the loss.

The load-bearing distinction is DECLARATION, not DATA:

  * a corpus whose rows contain a key called "audio", with no declaration, is a
    legitimate text corpus with an extra field. It must train. The drop must be
    announced.
  * the same corpus with FOUNDATIONSCALE_TRAIN_AUDIO_COLUMN=audio is an
    operator declaration that audio was meant to be trained. This plane cannot
    train audio, so it must refuse by name with 96.

Refusing merely because a column is named "audio" would break ordinary text
corpora that happen to carry such a field. Training it silently would return
the original defect. Either direction of over-correction is RED here.

The four arms -- each an independent subprocess of the REAL entry point,
``python -m foundationscale.train.cli``, the thing under test being the thing
that ships, not an in-process reimplementation -- differ only in corpus and
environment; every trainer flag is held constant across arms:

    audio_only      2 rows of {"audio": "clip0.wav"} -- no text column and no
                    declaration. It should still be refused by the old missing
                    text requirement.
    text_audio      2 rows of {"text": <sentence>, "audio": "clip0.wav"}, with
                    every modality declaration explicitly removed. Today: train,
                    and the [fs:train:data] notice must name the dropped field.
    image_control   2 rows of {"text": <sentence>}, run with
                    FOUNDATIONSCALE_TRAIN_IMAGE_COLUMN set to a column absent
                    from the corpus. Today: exit 96 naming it.
    declared_audio  the same corpus as text_audio, but with
                    FOUNDATIONSCALE_TRAIN_AUDIO_COLUMN=audio. Today: exit 96
                    whose refusal names audio. This is the deciding arm.

WHY IMAGE_CONTROL STILL EXISTS
    Declared_audio is now the deciding arm, but that does not make the image
    control redundant. It checks that the pre-existing declared non-text
    surface still refuses for its own declared column and gives this instrument
    an independent observation that a 96 naming a declaration can be seen
    through the same subprocess and scanner. If image_control instead exits 0,
    or refuses for an unrelated reason, the row cannot acquit or convict the
    declaration machinery on this checkout and the adjudicator refuses rather
    than pretend a control survived. The new arm then decides the claim; the
    old control keeps the measurement honest.

WHY AUDIO_ONLY STILL EXISTS
    A corpus with no usable text must not start training just because the
    harness is now paying attention to modality declarations. With no
    FOUNDATIONSCALE_TRAIN_AUDIO_COLUMN exported there is no declared modality
    to refuse, so the ordinary missing 'text' refusal remains the expected
    surface. If that arm exits 0, the framework trained a corpus it has no
    training field for, which is RED. If it exits 96 naming neither the text
    requirement nor the corpus's audio field, the refusal cannot be tied to
    this corpus and the row refuses to adjudicate it.

THE SELF-HIT TRAP
    text_audio's verdict turns on whether the child process's console output
    contains the [fs:train:data] marker and the word "audio". The manifest can
    record the corpus PATH and full argv. Were the corpus named
    corpus_audio.jsonl, or an output directory t23_text_audio/, the scanner
    would find its own needle and score a framework silence GREEN because the
    harness said the word. Therefore every path placed on the child's argv is
    arm_a / arm_b / arm_c / arm_d and corpus.jsonl -- no case-insensitive
    "audio" anywhere -- argv cleanliness is ASSERTED before the first arm runs
    and again when the payload is built, and a dirty argv is REFUSE
    CANNOT-MEASURE, not a warning. The record VALUE is clip0.wav so that only
    the record KEY carries the word.

    The declaration itself is supplied through the CHILD ENVIRONMENT, not
    argv. declared_audio gets FOUNDATIONSCALE_TRAIN_AUDIO_COLUMN=audio from a
    copy of os.environ. All other arms explicitly remove the audio, video, and
    image declaration variables so an operator who happens to have one exported
    -- including an empty one, which the subject correctly treats as undeclared
    -- cannot change which arm is being measured.

VERDICT ORDER (load-bearing; pinned by --self-test controls, not comments)
    1   an arm absent, or its rc None                    -> 95 UNMEASURED
    2   an arm with argv_clean False                     -> 96 REFUSE (names it)
    3   image_control rc 0                               -> 96 REFUSE
    4   image_control rc neither 0 nor 96                -> 95 UNMEASURED
    5   image_control 96 without its declared column     -> 96 REFUSE
    6   audio_only rc 0                                  -> 5 RED
    7   audio_only rc neither 0 nor 96                   -> 95 UNMEASURED
    8   audio_only 96 naming neither requirement         -> 96 REFUSE
    9   declared_audio rc 0                              -> 5 RED
    10  declared_audio rc neither 0 nor 96               -> 95 UNMEASURED
    11  declared_audio 96 not naming audio               -> 5 RED
    12  text_audio rc 96                                 -> 5 RED (name-sniffing)
    13  text_audio rc neither 0 nor 96                   -> 95 UNMEASURED
    14  text_audio rc 0 with tokenized None or 0         -> 95 UNMEASURED
    15  text_audio rc 0 and the drop notice names audio  -> 0 GREEN
    16  text_audio rc 0 and it does not                  -> 5 RED (silent drop)

Rule 11 is the original defect shape in miniature: the right exit code with the
old missing-'text' explanation is not a clean modality refusal, so it is RED
rather than acquitted on the number alone. Rule 12 is the opposite defect:
refusing an undeclared corpus on a column name is name-sniffing and breaks
legitimate text data. Rule 1 and the unexpected-rc rules keep G honest: an arm
that is absent, crashes, or times out is 95, never RED.

WHAT THIS FILE DOES NOT MEASURE
    It measures the declaration and dropped-column surface of one shipped
    training path on a small model and a two-row corpus. It does not decide
    whether FoundationScale ought to support audio, whether another entry point
    has an audio loader, or whether a dropped column would have helped the
    loss. A subprocess timeout is a real outcome, recorded as rc=None, and rule
    1 turns it into 95; it is never swallowed.

EXIT CONTRACT
    0 claim upheld, 5 the observation refuted it, 95 ran but the deciding
    quantity could not be measured, 96 refused to run at all. Never 1 or 2:
    argparse's error() is overridden to exit 96. The verdict function is pure
    -- no I/O, no torch, no datasets -- and --self-test drives it over
    synthetic payloads under a bare interpreter (``python3 -S -E``, no
    site-packages): nothing outside the standard library is imported at module
    scope, and torch is never imported in this process at all. No GPU is probed
    -- this row does not need one, and refusing over a GPU it does not use
    would be a false 96.
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
CLAIM = "declaring audio REFUSES cleanly and NAMES the modality"
CONTROL = "an undeclared audio column TRAINS with the drop named; refusing it is name-sniffing"

VERDICT_NAMES = {GREEN: "GREEN", RED: "RED", UNMEASURED: "UNMEASURED", REFUSE: "REFUSE"}

# The scan needle, lowercase. Every argv element this file creates is checked
# against it; only corpus record KEYS and the declared_audio child environment
# may carry it.
NEEDLE = "audio"

# The existing arms keep their names and directories. The new deciding arm is
# appended rather than used to renumber or rename them.
ARM_NAMES: tuple[str, ...] = (
    "audio_only",
    "text_audio",
    "image_control",
    "declared_audio",
)
ARM_DIRS: dict[str, str] = {
    "audio_only": "arm_a",
    "text_audio": "arm_b",
    "image_control": "arm_c",
    "declared_audio": "arm_d",
}
CORPUS_FILENAME = "corpus.jsonl"
MANIFEST_GLOB = "*manifest*.json"

IMAGE_COLUMN_ENV = "FOUNDATIONSCALE_TRAIN_IMAGE_COLUMN"
AUDIO_COLUMN_ENV = "FOUNDATIONSCALE_TRAIN_AUDIO_COLUMN"
VIDEO_COLUMN_ENV = "FOUNDATIONSCALE_TRAIN_VIDEO_COLUMN"
DECLARED_IMAGE_COLUMN = "t23_img_probe"
DECLARED_AUDIO_COLUMN = NEEDLE
MODALITY_DECLARATION_ENVS = (IMAGE_COLUMN_ENV, AUDIO_COLUMN_ENV, VIDEO_COLUMN_ENV)

DROP_NOTICE_MARKER = "[fs:train:data]"

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
    if arm in ("text_audio", "declared_audio"):
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
    argv carrying the needle would make text_audio's notice scan read its own
    fingerprint. A failure aborts the run before a single trainer is launched.
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
            "text_audio's notice scan would match its own fingerprint"
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


def _clean_declaration_environment(env: dict[str, str]) -> None:
    """Remove every inherited modality declaration, empty ones included."""
    for variable in MODALITY_DECLARATION_ENVS:
        env.pop(variable, None)


def _arm_environment(name: str) -> dict[str, str]:
    """The one environment in which ``name`` differs from the other arms.

    Only declared_audio receives the new non-empty audio declaration. Only
    image_control receives the older image declaration. Every other arm has all
    declaration variables explicitly removed, because the subject defines an
    empty inherited value as undeclared and an operator's shell is not part of
    the measurement.
    """
    env = dict(os.environ)
    _clean_declaration_environment(env)
    if name == "image_control":
        env[IMAGE_COLUMN_ENV] = DECLARED_IMAGE_COLUMN
    elif name == "declared_audio":
        env[AUDIO_COLUMN_ENV] = DECLARED_AUDIO_COLUMN
    return env


def _run_arm(name: str, arm_dir: Path, cmd: list[str]) -> dict[str, Any]:
    """Run one arm as an independent subprocess and project the outcome.

    ``output`` is stdout merged with stderr (true interleave is unrecoverable
    from two separately captured pipes, so stdout is kept ahead of stderr with
    a newline boundary between them). A timeout becomes rc=None, which rule 1
    turns into 95.
    """
    rc: int | None
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            env=_arm_environment(name),
            timeout=ARM_TIMEOUT_S,
        )
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
    elif name == "declared_audio":
        entry["declaration_env"] = AUDIO_COLUMN_ENV
        entry["declared_column"] = DECLARED_AUDIO_COLUMN
    return entry


def _build_payload(arms: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """The JSON-shaped payload the verdict judges. Re-asserts the no-hit invariant."""
    for name in ARM_NAMES:
        assert _argv_dirty_count(list(arms[name]["argv"])) == 0, (
            f"{name}: payload argv carries the '{NEEDLE}' needle; the notice scan "
            "would match its own fingerprint"
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

    # Rule 2: a dirty argv means the notice scan would match its own needle.
    # That is CANNOT-MEASURE, not a result.
    for name in ARM_NAMES:
        if not entries[name].get("argv_clean"):
            argv = entries[name].get("argv") or []
            offenders = [str(e) for e in argv if NEEDLE in str(e).lower()]
            first = offenders[0] if offenders else "<no argv recorded>"
            return REFUSE, (
                f"{name}: argv_clean is False -- argv element {first!r} carries the "
                f"'{NEEDLE}' needle; the notice scan would match its own fingerprint"
            )

    # Rules 3-5: the older positive control must refuse, and must refuse naming
    # the column it declared. A crash is 95; a clean train or a refusal for an
    # unrelated cause makes the declaration surface uninterpretable.
    control = entries["image_control"]
    rc_c = control.get("rc")
    out_c = str(control.get("output") or "")
    declared = control.get("declared_column")
    if rc_c == GREEN:
        return REFUSE, (
            "image_control: exited 0, expected 96 -- the positive control did not "
            "fire, so declaration behaviour is uninterpretable"
        )
    if rc_c != REFUSE:
        return UNMEASURED, (
            f"image_control: exited {rc_c}, neither 0 nor the expected 96 -- the "
            "arm errored before it could serve as a control"
        )
    if not isinstance(declared, str) or not declared or declared not in out_c:
        return REFUSE, (
            f"image_control: exited 96 but its output never names its declared "
            f"column {declared!r} -- it refused for some other reason, and a "
            "control that fires for the wrong cause is not one"
        )

    # Rules 6-8: with nothing declared, the audio-only corpus still has no text
    # field. It must not train; its refusal must be about this corpus.
    only = entries["audio_only"]
    rc_a = only.get("rc")
    out_a = str(only.get("output") or "")
    if rc_a == GREEN:
        return RED, (
            "audio_only: exited 0 -- a corpus whose only non-trivial field is "
            f"'{NEEDLE}' trained anyway; neither text nor a declared modality "
            "refusal stopped it"
        )
    if rc_a != REFUSE:
        return UNMEASURED, (
            f"audio_only: exited {rc_a}, neither 0 nor 96 -- the arm errored "
            "rather than demonstrating the expected refusal"
        )
    names_text_requirement = "'text' column" in out_a
    names_audio_column = NEEDLE in out_a.lower()
    if not (names_text_requirement or names_audio_column):
        return REFUSE, (
            "audio_only: exited 96 but the output names neither the 'text' column "
            f"requirement nor the {NEEDLE} column -- a refusal about something "
            "unrelated to the corpus this arm carried"
        )

    # Rules 9-11: THE deciding arm. A non-empty declaration must refuse by
    # modality name. Training anyway is RED; 96 with the old missing-'text'
    # explanation is right code, wrong reason, and also RED.
    declared_audio = entries["declared_audio"]
    rc_d = declared_audio.get("rc")
    out_d = str(declared_audio.get("output") or "")
    if rc_d == GREEN:
        return RED, (
            "declared_audio: exited 0 -- FOUNDATIONSCALE_TRAIN_AUDIO_COLUMN was "
            f"set to {DECLARED_AUDIO_COLUMN!r} and the run trained anyway; the "
            "declaration surface is still being ignored"
        )
    if rc_d != REFUSE:
        return UNMEASURED, (
            f"declared_audio: exited {rc_d}, neither 0 nor 96 -- the deciding arm "
            "errored before the declaration could be judged"
        )
    if NEEDLE not in out_d.lower():
        return REFUSE and RED, (
            "declared_audio: exited 96 but the refusal never names 'audio' -- "
            "right code, wrong reason (the old missing-'text' failure shape, "
            "not a modality refusal)"
        )

    # Rules 12-16: the undeclared corpus must train, but no longer silently.
    # A refusal here is name-sniffing: the key existed in DATA, not in ENV.
    both = entries["text_audio"]
    rc_b = both.get("rc")
    if rc_b == REFUSE:
        return RED, (
            "text_audio: exited 96 although nothing was declared -- the trainer "
            "refused on the column name rather than the declaration; "
            f"name-sniffing would break legitimate text corpora carrying {NEEDLE}"
        )
    if rc_b != GREEN:
        return UNMEASURED, (
            f"text_audio: exited {rc_b}, neither 0 nor 96 -- the undeclared-corpus "
            "arm errored, so training and the drop notice cannot be judged"
        )
    tokenized = both.get("tokenized")
    if not isinstance(tokenized, int) or tokenized == 0:
        return UNMEASURED, (
            f"text_audio: exited 0 with tokenized={tokenized} -- it 'succeeded' "
            "without tokenizing anything, so no statement about a dropped field "
            "can be read from it"
        )
    out_b = str(both.get("output") or "")
    has_drop_notice = DROP_NOTICE_MARKER in out_b
    names_dropped_modality = NEEDLE in out_b.lower()
    if has_drop_notice and names_dropped_modality:
        return GREEN, (
            f"text_audio: exited 0, {tokenized} examples tokenized, and its "
            f"{DROP_NOTICE_MARKER} notice names the dropped {NEEDLE} column -- "
            "the undeclared corpus trains and the drop is not silent"
        )
    return RED, (
        f"text_audio: exited 0, {tokenized} examples tokenized, but no "
        f"{DROP_NOTICE_MARKER} notice names the {NEEDLE} column it dropped -- "
        "the silent drop is back"
    )


# ---------------------------------------------------------------------------
# Self-test: named controls driving the pure verdict on synthetic payloads.
# Runs under a bare interpreter: standard library only, no GPU, no trainer.
# ---------------------------------------------------------------------------


def _synthetic_baseline() -> dict[str, Any]:
    """A fully healthy payload under the fixed declaration surface."""
    argv = ["python3", "-m", "foundationscale.train.cli", "--dataset", "corpus.jsonl"]
    notice = (
        "[fs:train:data] text-only arm: tokenizing 'text'; columns ['audio'] are "
        "dropped and contribute nothing to the loss"
    )
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
                "output": f"{notice}\n2 examples tokenized",
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
            "declared_audio": {
                "rc": 96,
                "output": (
                    "[fs:train:refuse] declared audio column 'audio' from "
                    "FOUNDATIONSCALE_TRAIN_AUDIO_COLUMN cannot be trained; this "
                    "plane can train text"
                ),
                "manifest_text": (
                    '{"stage": "refused", "config": {"extra.exit": 96, '
                    '"extra.audio_column": "audio"}}'
                ),
                "tokenized": None,
                "argv_clean": True,
                "argv": list(argv),
                "declaration_env": AUDIO_COLUMN_ENV,
                "declared_column": DECLARED_AUDIO_COLUMN,
            },
        },
    }


def _arms(
    *,
    a: dict[str, Any] | None = None,
    b: dict[str, Any] | None = None,
    c: dict[str, Any] | None = None,
    d: dict[str, Any] | None = None,
    drop: tuple[str, ...] = (),
) -> dict[str, Any]:
    """A fresh baseline payload with per-arm field overrides applied."""
    base = _synthetic_baseline()["arms"]
    for name, override in (
        ("audio_only", a),
        ("text_audio", b),
        ("image_control", c),
        ("declared_audio", d),
    ):
        if override:
            base[name].update(override)
    for name in drop:
        del base[name]
    return {"arms": base}


def _self_test() -> int:
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
            "new deciding arm refuses naming audio; undeclared arm trains naming the drop",
            _synthetic_baseline(),
            GREEN,
            ("text_audio",),
        ),
        (
            "SC2",
            "the silent drop returned (RED must fire, count quoted)",
            _arms(
                b={
                    "rc": 0,
                    "output": "2 examples tokenized\ntrain_loss 1.9302",
                    "manifest_text": "",
                    "tokenized": 2,
                }
            ),
            RED,
            ("text_audio", "2", "silent drop"),
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
            "image_control rc 5: the control errored, so row is UNMEASURED",
            _arms(c={"rc": 5}),
            UNMEASURED,
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
            "audio_only rc 1: an errored arm is 95, never RED",
            _arms(a={"rc": 1, "output": "Traceback (most recent call last):"}),
            UNMEASURED,
            ("audio_only",),
        ),
        (
            "SC11",
            "text_audio rc 1: an errored arm is 95, never RED",
            _arms(b={"rc": 1, "output": "Traceback (most recent call last):"}),
            UNMEASURED,
            ("text_audio",),
        ),
        (
            "SC12",
            "undeclared text_audio refused, rc 96: name-sniffing (RED)",
            _arms(b={"rc": 96, "output": "[fs:train:refuse] column 'audio' seen"}),
            RED,
            ("text_audio", "name-sniffing"),
        ),
        (
            "SC13",
            "text_audio rc 0, tokenized 0: 95 before either 5 branch",
            _arms(
                b={
                    "rc": 0,
                    "output": "0 examples tokenized",
                    "manifest_text": "",
                    "tokenized": 0,
                }
            ),
            UNMEASURED,
            ("text_audio", "0"),
        ),
        (
            "SC14",
            "dirty argv AND the silent drop: 96 rule 2 beats rule 16 (order pin)",
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
            "drop notice present but tokenized 0: 95, rule 14 beats rule 15",
            _arms(
                b={
                    "rc": 0,
                    "output": "[fs:train:data] ignored 'audio'\n0 examples tokenized",
                    "tokenized": 0,
                }
            ),
            UNMEASURED,
            ("text_audio",),
        ),
        (
            "SC17",
            "the notice says capitalised 'Audio' (case pin, GREEN must fire)",
            _arms(
                b={
                    "output": (
                        "[fs:train:data] text-only arm: tokenizing 'text'; Audio "
                        "is dropped\n2 examples tokenized"
                    )
                }
            ),
            GREEN,
            ("text_audio",),
        ),
        (
            "SC18",
            "audio_only rc 96 whose message is a missing GPU (rule 8 REFUSE)",
            _arms(a={"rc": 96, "output": "REFUSED: no accelerator device visible"}),
            REFUSE,
            ("audio_only",),
        ),
        (
            "SC19",
            "tokenized line never appeared: 95 on None, not 0 (rule 14 pin)",
            _arms(b={"rc": 0, "output": "run complete", "tokenized": None}),
            UNMEASURED,
            ("text_audio",),
        ),
        (
            "SC20",
            "audio_only's refusal names the audio column alone (rule 8 passes it)",
            _arms(a={"rc": 96, "output": "REFUSED: dataset column 'audio' has no loader"}),
            GREEN,
            ("text_audio",),
        ),
        (
            "SC21",
            "audio appears only in the manifest, not the console notice (RED)",
            _arms(
                b={
                    "output": "2 examples tokenized",
                    "manifest_text": '{"dropped_columns": ["audio"]}',
                }
            ),
            RED,
            ("text_audio", "silent drop"),
        ),
        (
            "SC22",
            "a dirty argv on image_control is still rule 2 (any-arm pin)",
            _arms(c={"argv_clean": False, "argv": ["python3", "--output-dir", "t23_audio_run"]}),
            REFUSE,
            ("image_control",),
        ),
        (
            "SC23",
            "declared_audio exits 96 naming audio (new deciding arm acquits)",
            _arms(
                d={
                    "output": (
                        "[fs:train:refuse] FOUNDATIONSCALE_TRAIN_AUDIO_COLUMN "
                        "declared column 'audio'; this plane trains text"
                    )
                }
            ),
            GREEN,
            ("text_audio",),
        ),
        (
            "SC24",
            "declared_audio exits 96 with the old missing-'text' reason (RED)",
            _arms(
                d={
                    "output": (
                        "REFUSED: the thin path requires a 'text' column; dataset "
                        "columns seen: ['clips']"
                    )
                }
            ),
            RED,
            ("declared_audio", "right code", "wrong reason"),
        ),
        (
            "SC25",
            "declared_audio trains anyway, rc 0 (RED must fire)",
            _arms(d={"rc": 0, "output": "2 examples tokenized", "tokenized": 2}),
            RED,
            ("declared_audio",),
        ),
        (
            "SC26",
            "declared_audio rc 1: an errored deciding arm is 95, never RED",
            _arms(d={"rc": 1, "output": "Traceback (most recent call last):"}),
            UNMEASURED,
            ("declared_audio",),
        ),
        (
            "SC27",
            "declared_audio absent: 95, rule 1 applies to the new arm too",
            _arms(drop=("declared_audio",)),
            UNMEASURED,
            ("declared_audio",),
        ),
        (
            "SC28",
            "declared_audio's refusal says capitalised Audio (case pin, GREEN)",
            _arms(d={"output": "[fs:train:refuse] AUDIO column declared; text only"}),
            GREEN,
            ("text_audio",),
        ),
        (
            "SC29",
            "text_audio names audio without the [fs:train:data] notice (RED)",
            _arms(b={"output": "2 examples tokenized\nnote: audio stream not loaded"}),
            RED,
            ("text_audio", "silent drop"),
        ),
    ]

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
        "ground truth: declaring audio through "
        f"{AUDIO_COLUMN_ENV}={DECLARED_AUDIO_COLUMN!r} must refuse naming the "
        "modality; the same corpus with every declaration removed must train "
        f"after a {DROP_NOTICE_MARKER} drop notice"
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
            "T1-23 verification row: four corpora, one new modality declaration, "
            "one positive control. Exit codes: 0 claim upheld, 5 the observation "
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
        help="directory for the arm_a / arm_b / arm_c / arm_d corpora, trainer "
        "output and manifests (required for a run; no defensible default)",
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
