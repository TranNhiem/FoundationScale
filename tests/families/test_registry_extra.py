"""Coverage tests for FamilySpec validation branches the registry suite misses.

Pins registry.py lines 101 and 112: an empty ``name`` and a tower declared
with an empty prefix must each raise ValueError with the named message.
Either registration, if allowed through, would make every later family lookup
or tower scoping silently MISS rather than loudly refuse -- the failure mode
the whole ``__post_init__`` validation exists to prevent at import time.
"""

from __future__ import annotations

import pytest

from foundationscale.families.registry import FamilySpec


def _valid_kwargs() -> dict[str, object]:
    return {
        "name": "demo",
        "model_types": ("demo_type",),
        "language_prefixes": ("model.language_model",),
        "towers": (),
        "adapter_leaf_modules": ("q_proj",),
        "expert_count_path": (),
    }


def test_empty_family_name_rejected_with_named_message() -> None:
    """Pins registry.py line 101: ``name=""`` raises ValueError naming the
    field, before any other check runs.

    If deleted: a nameless family registers fine, its announcements and
    refusals print an empty label, and corpus/registry diagnostics become
    unattributable -- a name that is 'human label used in announcements and
    refusals' must exist to be printed.
    """
    kwargs = _valid_kwargs()
    kwargs["name"] = ""
    with pytest.raises(ValueError, match="FamilySpec.name must be non-empty") as excinfo:
        FamilySpec(**kwargs)  # type: ignore[arg-type]
    assert "non-empty" in str(excinfo.value)


def test_tower_with_empty_prefix_rejected_with_named_message() -> None:
    """Pins registry.py line 112: a tower declaring an empty prefix raises
    ValueError naming the FAMILY and the defect.

    If deleted: an empty prefix passes the duplicate/unknown-modality checks
    below it, then matches EVERY module name during scoping (any string
    startswith '') -- turning adapter target selection globally blind while
    reporting success.
    """
    kwargs = _valid_kwargs()
    kwargs["towers"] = (("", "image"),)
    with pytest.raises(ValueError, match="declares a tower with an empty prefix") as excinfo:
        FamilySpec(**kwargs)  # type: ignore[arg-type]
    assert "'demo'" in str(excinfo.value)
