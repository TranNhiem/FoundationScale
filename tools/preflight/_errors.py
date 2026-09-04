"""Typed failures for the pre-flight blocklist."""

from __future__ import annotations

from collections.abc import Sequence

# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class ToolError(RuntimeError):
    """The tool itself could not run (exit 2). Not a clearance, not a block: an abort."""


class ConfigError(RuntimeError):
    """The config file parsed but fails validation (exit 1, BLOCKED, keys named).

    Aggregates EVERY problem rather than raising on the first: an operator fixing
    one missing key per run discovers the last one after the launch window closed.
    """

    def __init__(self, problems: Sequence[str]) -> None:
        self.problems = tuple(problems)
        super().__init__(f"config invalid: {len(problems)} problem(s)")


class ArtifactError(RuntimeError):
    """A declared artifact is missing or unreadable. Checks convert this to BLOCK."""
