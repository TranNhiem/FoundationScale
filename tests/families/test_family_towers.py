"""Pins for the towers-as-pairs registry change and DDP dormancy derivation.

Runs under FS_FORBID_SKIPS=1: every branch is explicit; no skip/xfail anywhere.
"""

from __future__ import annotations

import pytest

from foundationscale.families.registry import REGISTRY, FamilySpec
from foundationscale.families.towers import (
    FamilyRefusal,
    derive_find_unused_parameters,
    resolve_module_path,
)


def _spec(name: str) -> FamilySpec:
    for spec in REGISTRY:
        if spec.name == name:
            return spec
    raise AssertionError(f"registry has no family named {name!r}")


_QWEN = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


def _towerless() -> FamilySpec:
    # A language-only family: no towers declared at all.
    return FamilySpec(
        name="towerless",
        model_types=("towerless_test_only",),
        language_prefixes=("model.layers",),
        towers=(),
        adapter_leaf_modules=_QWEN,
        expert_count_path=("text_config", "num_experts"),
    )


def test_text_only_qwen35_derives_true() -> None:
    # REGRESSION: previously the leaf-attribute table never matched
    # "model.visual", the dormant list was empty, the flag stayed False, and
    # every multi-rank qwen3.5 run aborted on first backward.
    spec = _spec("qwen3.5")
    assert derive_find_unused_parameters(spec, frozenset({"text"})) is True


def test_gemma4_text_only_derives_true() -> None:
    # Both vision and audio towers exist; a text-only declaration exercises
    # neither, so DDP must expect unused parameters.
    spec = _spec("gemma4")
    assert derive_find_unused_parameters(spec, frozenset({"text"})) is True


def test_gemma4_full_omni_derives_false() -> None:
    spec = _spec("gemma4")
    assert derive_find_unused_parameters(spec, frozenset({"text", "image", "audio"})) is False


def test_towerless_family_derives_false() -> None:
    # Floor is False: w/o towers there is no declared reason to expect unused
    # parameters, so a genuinely unused one must still abort the run.
    assert derive_find_unused_parameters(_towerless(), frozenset({"text"})) is False


def test_unregistered_family_refuses_with_96_text() -> None:
    with pytest.raises(FamilyRefusal) as excinfo:
        derive_find_unused_parameters(None, frozenset({"text", "image", "audio"}))
    refusal = excinfo.value
    assert refusal.exit_status == 96
    assert "not registered" in refusal.reason
    assert "--adapter-target" in refusal.reason  # the refusal states both ways forward
    assert "FamilySpec" in refusal.reason


def test_mtp_is_not_a_modality_tower() -> None:
    spec = _spec("qwen3.5")
    pairs = dict(spec.tower_modalities)
    assert pairs == {"model.visual": "image"}
    assert "mtp" not in pairs


def test_mtp_does_not_drive_find_unused_when_image_declared() -> None:
    # mtp is out of adapter scope but no modality exercises it; it must not,
    # on its own, force dormancy handling on.
    spec = _spec("qwen3.5")
    assert derive_find_unused_parameters(spec, frozenset({"text", "image"})) is False


def test_tower_prefixes_still_returns_every_prefix_including_mtp() -> None:
    # adapters.py reads this unchanged and relies on "mtp" being listed to
    # keep the adapter out of the lookahead head.
    assert _spec("qwen3.5").tower_prefixes == ("model.visual", "mtp")
    assert _spec("gemma4").tower_prefixes == (
        "model.vision_tower",
        "model.audio_tower",
        "model.embed_vision",
    )


def test_duplicate_tower_prefix_rejected_at_registration() -> None:
    with pytest.raises(ValueError, match="twice"):
        FamilySpec(
            name="dupe",
            model_types=("dupe_test_only",),
            language_prefixes=("model.layers",),
            towers=(("model.visual", "image"), ("model.visual", None)),
            adapter_leaf_modules=_QWEN,
            expert_count_path=(),
        )


def test_unknown_modality_string_rejected_at_registration() -> None:
    with pytest.raises(ValueError, match="unknown"):
        FamilySpec(
            name="typo",
            model_types=("typo_test_only",),
            language_prefixes=("model.layers",),
            towers=(("model.visual", "img"),),  # "img" is not a modality
            adapter_leaf_modules=_QWEN,
            expert_count_path=(),
        )


def test_resolve_module_path_walks_dotted_paths() -> None:
    class Inner:
        visual = object()
        vision_tower = object()

    class Outer:
        model = Inner()

    model = Outer()
    assert resolve_module_path(model, "model.visual") is Inner.visual
    assert resolve_module_path(model, "model.vision_tower") is Inner.vision_tower
    # Absent components announce via None -- never via a vacuous match.
    assert resolve_module_path(model, "model.audio_tower") is None
    assert resolve_module_path(model, "model.visual.deep") is None


class _Leaf:
    """Stands in for a loaded tower module."""


class _Nested:
    """A composite VLM: towers hang off ``.model``, not off the root."""

    def __init__(self, **towers: object) -> None:
        self.model = type("_Inner", (), towers)()


def _spec(name: str) -> FamilySpec:
    for spec in REGISTRY:
        if spec.name == name:
            return spec
    raise AssertionError(f"registry has no family {name!r}")


def test_qwen35_visual_tower_is_found_where_the_hardcoded_table_missed_it() -> None:
    """If deleted: the measured #523 abort comes back.

    The table this replaced listed the bare attributes ``vision_tower`` and
    ``audio_tower`` -- Gemma4's spelling. qwen3.5 calls its tower
    ``model.visual``, so the walk found nothing, left ddp_find_unused_parameters
    False, and EVERY multi-rank qwen3.5 run aborted on the first backward. This
    pins that the tower is located from the family's own declaration.
    """
    from foundationscale.train.loop import _dormant_modality_towers

    model = _Nested(visual=_Leaf())

    dormant = _dormant_modality_towers(model, family=_spec("qwen3.5"), image_declared=False)

    assert dormant == ["model.visual"]


def test_declaring_the_image_column_exercises_the_vision_tower() -> None:
    """If deleted: a multimodal run would declare its own vision tower dormant
    and tell DDP to tolerate unused parameters it should be reporting."""
    from foundationscale.train.loop import _dormant_modality_towers

    model = _Nested(visual=_Leaf())

    assert _dormant_modality_towers(model, family=_spec("qwen3.5"), image_declared=True) == []


def test_a_none_modality_tower_never_forces_the_flag_on_its_own() -> None:
    """If deleted: qwen3.5's ``mtp`` head -- out of adapter scope but not a
    modality any declaration could exercise -- would force
    ddp_find_unused_parameters on for every text-only run."""
    from foundationscale.train.loop import _dormant_modality_towers

    # qwen3.5 declares ``mtp`` at the ROOT, not under ``.model``. Nesting it
    # would make this pass because the tower was never RESOLVED, which says
    # nothing about the modality rule the test exists to pin.
    model = _Nested()
    model.mtp = _Leaf()
    assert resolve_module_path(model, "mtp") is not None, (
        "fixture does not expose the declared path; the assertion below would pass vacuously"
    )

    assert _dormant_modality_towers(model, family=_spec("qwen3.5"), image_declared=False) == []


def test_a_declared_tower_absent_from_the_checkpoint_is_not_dormant() -> None:
    """If deleted: a family that CAN carry an audio tower would force the flag
    on checkpoints that do not carry one, hiding real unused parameters."""
    from foundationscale.train.loop import _dormant_modality_towers

    model = _Nested()  # declares nothing at all

    assert _dormant_modality_towers(model, family=_spec("gemma4"), image_declared=False) == []


def test_unregistered_family_yields_no_dormancy_claim() -> None:
    """If deleted: None could be read as "no towers", which is a measurement
    this function cannot make for a family nobody declared."""
    from foundationscale.train.loop import _dormant_modality_towers

    assert (
        _dormant_modality_towers(_Nested(visual=_Leaf()), family=None, image_declared=False) == []
    )
