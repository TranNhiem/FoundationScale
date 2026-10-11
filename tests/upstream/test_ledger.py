from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from foundationscale.upstream.ledger import (
    LEDGER,
    LedgerEntry,
    LedgerKind,
    entries_for,
    ledger_problems,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def _entry(**overrides: object) -> LedgerEntry:
    fields: dict[str, object] = {
        "id": "x",
        "kind": LedgerKind.WORKAROUND,
        "framework": "nemo",
        "path": "f.py",
        "anchor": "MARK",
        "why": "because",
        "upstream_version": "nemo 3.1",
        "expiry_check": "it stops failing",
    }
    fields.update(overrides)
    return LedgerEntry(**fields)  # type: ignore[arg-type]


def test_the_real_tree_matches_the_ledger() -> None:
    """MUST_PASS: every recorded copy/workaround is where the ledger says it is."""
    assert ledger_problems(REPO_ROOT) == []


def test_a_moved_anchor_is_reported(tmp_path: Path) -> None:
    """MUST_FIRE: code changed without updating the ledger fails, never goes stale silently."""
    (tmp_path / "f.py").write_text("something else\n")
    assert ledger_problems(tmp_path, (_entry(),)) == ["anchor_missing:x"]


def test_a_missing_file_is_reported(tmp_path: Path) -> None:
    assert ledger_problems(tmp_path, (_entry(path="gone.py"),)) == ["path_missing:x"]


def test_duplicate_ids_and_empty_fields_are_reported(tmp_path: Path) -> None:
    (tmp_path / "f.py").write_text("MARK\n")
    problems = ledger_problems(tmp_path, (_entry(), _entry(why="  ")))
    assert problems == ["duplicate_id:x", "empty_field:x:why"]


def test_consistent_fake_ledger_passes(tmp_path: Path) -> None:
    (tmp_path / "f.py").write_text("before MARK after\n")
    assert ledger_problems(tmp_path, (_entry(),)) == []


def test_entries_for_and_coverage_of_the_record() -> None:
    assert {e.id for e in entries_for("nemo")} == {
        "nemo-salm-strict-loading",
        "nemo-datamodule-null-validation",
        "nemo-canary-train-cfg-strip",
        "nemo-canary-text-field",
    }
    assert len(entries_for("transformers")) == 4
    assert entries_for("nonexistent") == ()
    kinds = {e.kind for e in LEDGER}
    assert kinds == {LedgerKind.COPY, LedgerKind.WORKAROUND, LedgerKind.PRIVATE_API}
    assert all(e.expiry_check.strip() for e in LEDGER)


def test_entry_is_frozen() -> None:
    entry = _entry()
    with pytest.raises(dataclasses.FrozenInstanceError):
        entry.id = "other"  # type: ignore[misc]
