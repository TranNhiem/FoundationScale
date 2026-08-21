"""Shared pytest fixtures for the gate-contract suite.

Deliberately minimal. The package is expected to be importable via
``pip install -e .`` (src layout) or a ``pythonpath = ["src"]`` entry in the
pytest configuration. There is intentionally no ``sys.path`` manipulation here:
a test suite that can only find the framework by path hacks would silently pass
against a stale checkout — the same class of silent success this framework
exists to prevent.
"""

from __future__ import annotations

import pytest

from foundationscale.gates.core import GateRegistry


@pytest.fixture
def fresh_registry() -> GateRegistry:
    """An isolated registry per test.

    Tests must never register against the global ``REGISTRY`` from here: gates
    registered there would leak between tests and could mask a ``required=``
    assertion by making a gate present that a later test expects to be missing —
    the registry-level analogue of ``all([]) is True``.
    """
    return GateRegistry()
