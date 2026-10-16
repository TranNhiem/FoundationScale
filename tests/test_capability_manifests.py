"""A5.2: each foundationscale capability_manifest.py is well-formed with a resolvable verify."""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest

import foundationscale

ROOT = Path(foundationscale.__file__).resolve().parent
MANIFESTS = sorted(
    ".".join(("foundationscale", *p.relative_to(ROOT).with_suffix("").parts))
    for p in ROOT.rglob("capability_manifest.py")
    if "__pycache__" not in p.parts
)


def test_manifests_exist() -> None:
    assert "foundationscale.capability_manifest" in MANIFESTS
    assert "foundationscale.agentic_rl.capability_manifest" in MANIFESTS


@pytest.mark.parametrize("name", MANIFESTS)
def test_entries_are_well_formed_and_verifiable(name: str) -> None:
    entries = importlib.import_module(name).MANIFEST
    assert isinstance(entries, list) and entries
    keys = [e["key"] for e in entries]
    assert len(keys) == len(set(keys))
    for e in entries:
        assert e["compare"] in ("equal", "subset", "superset")
        module, _, attr = e["verify"].partition(":")
        target = importlib.import_module(module)
        for part in attr.split("."):
            target = getattr(target, part)
        assert callable(target)
        measured = target()  # the measurement runs here without error
        assert isinstance(measured, (list, tuple))
