"""Interpreter floor for the T1 adjudicators -- an unmet toolchain is REFUSE, not RED.

The repository declares ``requires-python = ">=3.10"`` in its packaging metadata and
formats to ``target-version = "py310"``, and these adjudicators use 3.10+ syntax
(``zip(..., strict=True)``). Nothing enforced that floor at runtime: measured on this
machine's system interpreter, Python 3.9.6, the syntax raised INSIDE the run body and
the main()-boundary handler faithfully adjudicated it against the scientific claim --

    T1-9 VERDICT RED: unexpected TypeError escaped the run body:
    zip() takes no keyword arguments

A reader would conclude the optimizer knob failed to move the loss curve. It did not
fail; it was never measured. The claim is innocent and the toolchain is at fault, so
the honest code is REFUSE (96) -- "I decline to measure" -- not RED (5). This is not a
hypothetical: #138 measured a 3.6.8 interpreter on this estate's login nodes, and #376
/#388 are the same class (a below-floor ``python3`` reading as broken gates rather than
as one unmet precondition). It is also the mirror of #83: a verdict carrying no
interpreter provenance cannot be attributed.

This module is deliberately stdlib-only and syntax-compatible with interpreters BELOW
the floor it enforces. A guard that cannot be imported by the interpreter it is meant
to reject is not a guard. Annotations are safe because ``from __future__ import
annotations`` leaves them unevaluated strings; what is NOT safe here, and is therefore
absent, is ``zip(strict=)``, runtime ``X | Y``, ``match``, and ``tomllib``.
"""

from __future__ import annotations

import sys
from collections.abc import Callable

# The floor the repository already declares. Stated here as the single runtime source;
# the packaging metadata is not readable from a standalone campaign script.
MIN_PYTHON = (3, 10)


def python_floor_reason(
    version: tuple[int, ...] | None = None,
    executable: str | None = None,
) -> str | None:
    """Return None if the interpreter meets the floor, else a reason naming both versions.

    ``version`` and ``executable`` are injectable ONLY so a self-test can exercise the
    refusing branch on a green interpreter. Without them the below-floor arm would be
    unreachable on any machine that can run the suite, and an assertion that never fires
    is not a control -- it is decoration that reads as coverage.
    """
    have_tuple = tuple(sys.version_info[:3]) if version is None else tuple(version)
    where = sys.executable if executable is None else executable
    if have_tuple[:2] >= MIN_PYTHON:
        return None
    want = ".".join(str(part) for part in MIN_PYTHON)
    have = ".".join(str(part) for part in have_tuple)
    return (
        f"interpreter is Python {have}, below the declared floor of {want} "
        f"(pyproject requires-python). The adjudicator uses {want}+ syntax, so the "
        f"claim cannot be measured here -- this is an unmet precondition, NOT a failed "
        f"claim. Interpreter: {where}"
    )


def floor_controls(row: str, record: Callable[[str, bool, str], None]) -> None:
    """Run the two floor controls for ``row``, reporting through the caller's ``record``.

    ``record(label, passed, detail)`` matches the callback every T1 self-test already
    builds, so the controls land in the module's own count rather than a second tally.
    """
    below = python_floor_reason(version=(3, 9, 6), executable="/usr/bin/python3")
    record(
        f"{row} F1 a below-floor interpreter yields a reason naming both versions",
        below is not None and "3.9.6" in below and "3.10" in below,
        repr(below),
    )
    at_floor = python_floor_reason(version=(3, 10, 0), executable="/usr/bin/python3")
    record(
        f"{row} F2 an at-floor interpreter yields no reason (the guard is not blanket)",
        at_floor is None,
        repr(at_floor),
    )
