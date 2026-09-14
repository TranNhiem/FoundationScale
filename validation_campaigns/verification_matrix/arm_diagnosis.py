"""Turn a failed arm subprocess into a diagnosable reason string (#441).

WHY THIS EXISTS
---------------
Pass p439 ran four T1-9 arms on a GB200 tray. Every one of them came back

    torchrun launcher exited 1 (...#171); last line: ====================

-- the banner. Not a single arm record named what went wrong, so the pass could
not be diagnosed from its own receipts and the cause had to be recovered by
re-running one arm by hand with the streams inherited instead of captured. That
hand run found the answer on the FIRST try, printed in plain language::

    [fs:train:refuse] transformers 5.13.0 rejected the declared config at
    Trainer/TrainingArguments construction: ... Refusing (96) rather than
    retrying with a guessed value

So the information was never missing. It was discarded, for two independent
reasons, and both have to be fixed or the next pass is just as blind:

1. WRONG STREAM. Every ``[fs:train:*]`` marker the trainer emits -- including
   Step.REFUSE -- is a bare ``print()`` in ``foundationscale.train.loop`` (see
   :225 there), so it goes to STDOUT. The rows read ``proc.stderr`` and fall
   back to stdout only when stderr is *empty*. Under torchrun stderr is never
   empty: the ``ChildFailedError`` banner always fills it. So the one stream
   carrying FoundationScale's own diagnostic vocabulary was in no row's
   denominator, on every failure, by construction.

2. WRONG SLICE. ``lines[-1]`` of a torchrun failure is always the closing
   ``====`` rule. t1_12's three-line tail and t1_11's N-line tail are the same
   defect with a larger constant: the banner is ~20 lines, so a tail that does
   not reach past it reads as a tail that found nothing.

THE RULE HERE
-------------
Search BOTH streams for the trainer's OWN declared markers first -- those are
statements the framework chose to make about itself, and they outrank anything
inferred from a traceback. Only if no marker is present fall back to a bounded
tail, and take that tail from the stream that actually has content. Always
persist a bounded excerpt of both streams on the record, so the NEXT failure
that this module's marker vocabulary cannot name is still diagnosable without a
second GPU allocation.

This module is deliberately dependency-free: it must be importable by a row
running under an interpreter that has no torch and no foundationscale, because
that is exactly the situation in which an arm fails early and needs explaining.
"""

from __future__ import annotations

import signal

# The trainer's declared refusal/abort vocabulary, most specific first. These
# are the literal Step values from src/foundationscale/train/loop.py. They are
# duplicated rather than imported on purpose: this helper must work when the
# package cannot be imported at all, which is one of the failure modes it
# exists to explain. tests/train/test_arm_diagnosis.py pins the two lists
# together so the duplication cannot drift silently -- in BOTH directions: no
# marker here that the trainer does not emit, and no Step in the trainer that
# is left unclassified. The second direction is the load-bearing one, because a
# new refusal Step that never reached this tuple would make an arm fail in a
# way the module whose whole job is naming failures could not name.
#
# Progress markers are deliberately absent. This tuple is scanned for the LAST
# matching line in a failed arm's output, so admitting [fs:train:done] would
# let a run that printed "done" and then died be "diagnosed" as done.
DIAGNOSTIC_MARKERS: tuple[str, ...] = (
    "[fs:train:refuse]",
    "[fs:train:red]",
    "[fs:train:blocked]",
    "[fs:train:unmeasured]",
)

# How much of each stream to keep on the record. Generous enough to contain a
# torchrun banner plus the trainer output above it, bounded so an arm record
# cannot become a log file.
EXCERPT_LINES = 40
EXCERPT_CHARS = 4000

# A line that is only banner furniture carries no diagnosis. Used to pick the
# last line that says something, not the last line that exists.
_FURNITURE_PREFIXES = ("=", "-", "*", "~")


def _is_furniture(line: str) -> bool:
    """True for a rule line (``=====``) or an empty line -- no diagnostic content."""
    stripped = line.strip()
    if not stripped:
        return True
    head = stripped[0]
    return head in _FURNITURE_PREFIXES and set(stripped) == {head}


def signal_note(returncode: int) -> str:
    """Name the signal behind a negative return code, or "" for a real exit.

    subprocess reports a signalled child as -N. That is the one case where
    empty streams are a DIAGNOSIS rather than the absence of one: an arm the
    OOM killer took says nothing at all, and "both streams are empty" without
    "killed by SIGKILL" sends the reader looking for a bug that is not there.
    """
    if returncode >= 0:
        return ""
    try:
        name = signal.Signals(-returncode).name
    except ValueError:
        name = f"signal {-returncode}"
    return f" [killed by {name}]"


def _tail(text: str | None, limit: int = EXCERPT_LINES) -> list[str]:
    lines = (text or "").strip().splitlines()
    return lines[-limit:]


def excerpt(text: str | None) -> str:
    """A bounded, record-safe excerpt of one stream."""
    joined = "\n".join(_tail(text))
    if len(joined) > EXCERPT_CHARS:
        joined = "..." + joined[-EXCERPT_CHARS:]
    return joined


def declared_markers(stdout: str | None, stderr: str | None) -> list[str]:
    """Every line from either stream carrying a declared trainer marker.

    stdout is searched first because that is where the markers are emitted; a
    marker appearing on stderr would mean the trainer redirected, which is
    worth surfacing rather than hiding.
    """
    found: list[str] = []
    for stream in (stdout, stderr):
        for line in (stream or "").splitlines():
            if any(marker in line for marker in DIAGNOSTIC_MARKERS):
                found.append(line.strip())
    return found


def diagnose(
    returncode: int,
    stdout: str | None,
    stderr: str | None,
    *,
    prefix: str,
) -> tuple[str, dict[str, str]]:
    """Build (reason, excerpts) for a non-zero arm subprocess.

    ``prefix`` is the caller's own framing of the exit code -- typically the
    #171 sentence about torchrun flattening the trainer's declared code. The
    diagnosis is appended to it, so the record states BOTH that the code is the
    launcher's and what the trainer actually said.

    The returned ``excerpts`` mapping is meant to be written onto the arm
    record verbatim. It is always populated, including when a marker was found:
    a marker names the refusal but does not always carry the context, and the
    whole point of #441 is that the next unexplained failure must not require
    another allocation.
    """
    excerpts = {"stdout_tail": excerpt(stdout), "stderr_tail": excerpt(stderr)}
    prefix = f"{prefix}{signal_note(returncode)}"

    markers = declared_markers(stdout, stderr)
    if markers:
        # The LAST declared marker is the trainer's final word on itself.
        reason = f"{prefix}; the trainer declared: {markers[-1]}"
        return reason, excerpts

    # No declared marker. Fall back to the last line with content, taken from
    # whichever stream has any -- not from stderr unconditionally, which is what
    # made the stdout-only markers unreachable in the first place.
    for stream, name in ((stderr, "stderr"), (stdout, "stdout")):
        speaking = [ln for ln in _tail(stream) if not _is_furniture(ln)]
        if speaking:
            reason = (
                f"{prefix}; no [fs:train:*] marker in either stream, so this is "
                f"not a declared refusal -- last speaking line of {name}: "
                f"{speaking[-1].strip()[:200]}"
            )
            return reason, excerpts

    reason = (
        f"{prefix}; both streams are empty, so the failure is undiagnosable "
        "from the subprocess -- treat this as an environment fault, not a verdict"
    )
    return reason, excerpts


__all__ = [
    "DIAGNOSTIC_MARKERS",
    "EXCERPT_CHARS",
    "EXCERPT_LINES",
    "declared_markers",
    "diagnose",
    "excerpt",
    "signal_note",
]
