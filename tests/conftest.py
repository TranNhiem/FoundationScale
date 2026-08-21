"""Shared pytest fixtures for the gate-contract suite.

Deliberately minimal. The package is expected to be importable via
``pip install -e .`` (src layout) or a ``pythonpath = ["src"]`` entry in the
pytest configuration. There is intentionally no ``sys.path`` manipulation here:
a test suite that can only find the framework by path hacks would silently pass
against a stale checkout — the same class of silent success this framework
exists to prevent.

The skip guard
--------------
CI must never report green over a suite that skipped. This repository's
founding incident was reproduced by a test CI never ran, because the extra that
provides torch was never installed and the skips were tolerated. A tolerated
skip is the same shape as ``all([]) is True``: the check was absent *and* it
bought confidence.

Set ``FS_FORBID_SKIPS=1`` and the hooks below turn any skip into a build
failure: ``pytest_terminal_summary`` names every skipped test with the reason
pytest recorded for it (the reason is the actionable half — "could not import
'torch'" calls for installing an extra, while a platform guard calls for fixing
the CI matrix), and ``pytest_sessionfinish`` flips an otherwise-green exit
status to failure. Leave the variable unset on a developer laptop: skips are
still listed, named, but tolerated.

The guard is kept honest the same way every gate here is kept honest: with a
MUST_FIRE probe. The ``check`` job in ``.github/workflows/ci.yml`` (and
``make skip-guard-probe``) generates a deliberately skipped test, runs pytest
against it with the guard armed, and fails unless the run both fails and names
the probe. A guard whose probe exits 0 is a guard that has rotted into a
no-op — and that now fails CI too.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import pytest

from foundationscale.gates.core import GateRegistry

if TYPE_CHECKING:
    # These have no public import path; imported for annotations only. Annotations
    # here are strings (PEP 563), so nothing below imports private modules at
    # runtime — the runtime objects are whatever pytest hands the hooks.
    from _pytest.reports import CollectReport, TestReport
    from _pytest.terminal import TerminalReporter

_SKIP_GUARD_ENV = "FS_FORBID_SKIPS"

# Per-process ledger of (nodeid, reason) for every skip observed this run,
# filled by the report hooks below and read by the summary/exit hooks. CI runs
# pytest in a single process, so a module-level list is the whole ledger; under
# pytest-xdist each worker would keep its own and the controller would need to
# merge them — out of scope until CI adopts xdist.
_SKIPPED: list[tuple[str, str]] = []


@pytest.fixture
def fresh_registry() -> GateRegistry:
    """An isolated registry per test.

    Tests must never register against the global ``REGISTRY`` from here: gates
    registered there would leak between tests and could mask a ``required=``
    assertion by making a gate present that a later test expects to be missing —
    the registry-level analogue of ``all([]) is True``.
    """
    return GateRegistry()


def _guard_armed() -> bool:
    """True only when skips are explicitly forbidden (CI); never by default."""
    return os.environ.get(_SKIP_GUARD_ENV) == "1"


def _skip_reason(longrepr: object) -> str:
    """Extract the human-readable reason from a skipped report's longrepr.

    Skips carry ``(path, lineno, reason)`` tuples; anything unexpected is
    stringified whole. The reason is always recorded: it is what lets a
    maintainer tell a missing dependency from a platform guard.
    """
    if isinstance(longrepr, tuple) and len(longrepr) == 3:
        return str(longrepr[2])
    return str(longrepr)


def pytest_runtest_logreport(report: TestReport) -> None:
    """Record skips from any run phase — setup (skipif, fixture skip), call,
    teardown — each arrives as its own report with ``skipped`` set."""
    if report.skipped:
        _SKIPPED.append((report.nodeid, _skip_reason(report.longrepr)))


def pytest_collectreport(report: CollectReport) -> None:
    """Record module-level skips (``pytest.skip(..., allow_module_level=True)``),
    which never reach the runtest phase."""
    if report.skipped:
        _SKIPPED.append((report.nodeid, _skip_reason(report.longrepr)))


@pytest.hookimpl(trylast=True)
def pytest_terminal_summary(terminalreporter: TerminalReporter) -> None:
    """Name every skipped test and its reason, armed or not.

    "N tests skipped" is not actionable. When the guard is armed this section is
    the failure message; when it is not (a laptop) it is a visible reminder that
    CI would fail these exact tests. ``trylast`` keeps it at the end of the
    summary where a failing build's last screenful lives.
    """
    if not _SKIPPED:
        if _guard_armed():
            terminalreporter.write_line(
                f"{_SKIP_GUARD_ENV}=1: skip guard armed — zero skips observed."
            )
        return

    count = len(_SKIPPED)
    armed = _guard_armed()
    if armed:
        terminalreporter.section(
            f"{_SKIP_GUARD_ENV}=1 — {count} skip(s): every one is a failure here",
            red=True,
            bold=True,
        )
    else:
        terminalreporter.section(
            f"{count} skip(s) — tolerated locally, FAILED by CI ({_SKIP_GUARD_ENV}=1)",
            yellow=True,
        )
    for nodeid, reason in _SKIPPED:
        terminalreporter.write_line(f"  {nodeid}")
        terminalreporter.write_line(f"    reason: {reason}")
    if armed:
        for line in (
            "Act on each reason above:",
            "  - missing module / 'could not import' -> a dependency is absent; install",
            "    the extra that provides it (this repo's founding incident: CI installed",
            "    .[dev] only, so 41 checkpoint tests skipped and CI stayed green).",
            "  - a platform guard -> the test cannot run on this runner; fix the CI",
            "    matrix. A tolerated skip in CI is a check that cannot fail.",
        ):
            terminalreporter.write_line(line)


@pytest.hookimpl(trylast=True)
def pytest_sessionfinish(session: pytest.Session) -> None:
    """Flip a green board to red when skips were observed and the guard is armed.

    ``session.exitstatus`` is read by pytest after all ``pytest_sessionfinish``
    hooks complete — pytest's own ``--suppress-no-test-exit-code`` option is
    implemented by assigning to it from this same hook — so the assignment below
    is the supported way to change the process exit code from a plugin. An
    already-red status is never touched: this hook can only make a run fail,
    never make one pass.
    """
    if _guard_armed() and _SKIPPED and session.exitstatus == pytest.ExitCode.OK:
        session.exitstatus = pytest.ExitCode.TESTS_FAILED
