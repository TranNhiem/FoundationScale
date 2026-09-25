
"""Verdict statuses and their process exit codes."""
from __future__ import annotations

from enum import Enum


class Status(str, Enum):
    """Terminal verdict for a skill execution.

    Contract: every status maps to exactly one permitted exit code. PASS means
    positive evidence; absence of evidence is UNMEASURED, never PASS.
    """

    PASS = "PASS"
    RED = "RED"
    UNMEASURED = "UNMEASURED"
    REFUSED = "REFUSED"

    @property
    def exit_code(self) -> int:
        return _EXIT_CODE[self]

    @classmethod
    def from_exit_code(cls, code: int) -> "Status":
        for status in cls:
            if status.exit_code == code:
                return status
        raise ValueError(f"invalid FoundationSkills exit code: {code!r}")


_EXIT_CODE: dict[Status, int] = {
    Status.PASS: 0,
    Status.RED: 5,
    Status.UNMEASURED: 95,
    Status.REFUSED: 96,
}
EXIT_CODES = frozenset({0, 5, 95, 96})


def worst(*statuses: Status | str) -> Status:
    """Return the worst status. Empty input is UNMEASURED, not a vacuous pass."""
    if not statuses:
        return Status.UNMEASURED
    parsed = [s if isinstance(s, Status) else Status(str(s)) for s in statuses]
    if Status.REFUSED in parsed:
        return Status.REFUSED
    if Status.RED in parsed:
        return Status.RED
    if Status.UNMEASURED in parsed:
        return Status.UNMEASURED
    return Status.PASS
