"""Tests for ``_dormant_modality_towers`` (see #504, refactored by #523).

The helper must key dormancy on the run's declaration and the loaded module
tree -- never on the data -- so that a text-only run on a VLM, a multimodal
run on an omni model, and a plain LLM stay three distinguishable cases.

#523 changed WHERE the tower list comes from, not what dormancy means. It used
to walk a two-row literal in the training loop naming ``vision_tower`` and
``audio_tower`` -- Gemma4's spelling, and only Gemma4's. The towers now come
from the family's own ``FamilySpec.towers`` declaration, so every case below is
stated against a real registered family rather than against a table that lived
one import away from the code it constrained. Each test keeps the assertion it
had; what changed is that the expected names are the DECLARED PATHS.
"""

from types import SimpleNamespace

import pytest

from foundationscale.families.registry import REGISTRY, FamilySpec
from foundationscale.train.loop import _dormant_modality_towers


def _spec(name: str) -> FamilySpec:
    for spec in REGISTRY:
        if spec.name == name:
            return spec
    raise AssertionError(f"registry has no family {name!r}")


GEMMA4 = "gemma4"


def test_plain_llm_with_neither_tower_reports_nothing_dormant():
    """If this fails, plain LLMs would silently tolerate genuinely unused parameters that DDP should
    report."""
    model = SimpleNamespace()
    assert _dormant_modality_towers(model, family=_spec(GEMMA4), image_declared=False) == []


def test_vision_tower_exercised_by_image_declaration_is_not_dormant():
    """If this fails, a multimodal run on a VLM would wrongly flag its actively trained vision tower
    as dormant."""
    model = SimpleNamespace(model=SimpleNamespace(vision_tower=object()))
    assert _dormant_modality_towers(model, family=_spec(GEMMA4), image_declared=True) == []


def test_vision_tower_without_image_declaration_is_dormant():
    """If this fails, a text-only run on a VLM would leave its vision tower frozen without anyone
    being told."""
    model = SimpleNamespace(model=SimpleNamespace(vision_tower=object()))
    assert _dormant_modality_towers(model, family=_spec(GEMMA4), image_declared=False) == [
        "model.vision_tower"
    ]


def test_omni_model_with_image_declared_leaves_only_audio_dormant():
    """If this fails, a multimodal run on an omni model would either miss the dormant audio tower or
    flag the exercised vision tower."""
    model = SimpleNamespace(model=SimpleNamespace(vision_tower=object(), audio_tower=object()))
    assert _dormant_modality_towers(model, family=_spec(GEMMA4), image_declared=True) == [
        "model.audio_tower"
    ]


def test_omni_model_without_image_declaration_reports_towers_in_declaration_order():
    """If this fails, the announcement would name towers in an order divorced from the family's
    declaration, breaking the declaration-order contract."""
    model = SimpleNamespace(model=SimpleNamespace(vision_tower=object(), audio_tower=object()))
    dormant = _dormant_modality_towers(model, family=_spec(GEMMA4), image_declared=False)
    assert dormant == ["model.vision_tower", "model.audio_tower"]
    # Order is the DECLARATION's, not this test's: same list, derived independently.
    assert dormant == [
        path
        for path, modality in _spec(GEMMA4).towers
        if modality is not None and getattr(model.model, path.rsplit(".", 1)[-1], None) is not None
    ]


def test_towers_nested_under_inner_model_attribute_are_still_found():
    """If this fails, composite VLMs that nest towers one level down would have their dormant towers
    missed entirely."""
    model = SimpleNamespace(model=SimpleNamespace(vision_tower=object()))
    assert _dormant_modality_towers(model, family=_spec(GEMMA4), image_declared=False) == [
        "model.vision_tower"
    ]


def test_a_tower_at_the_model_ROOT_is_still_found():
    """If this fails, the root fallback #523 kept is gone.

    The declared path is ``model.vision_tower``, but the pre-#523 code also
    probed the bare attribute on the root, and that was load-bearing: the same
    checkpoint exposes different roots under different auto-classes -- the
    measured reason qwen3.5 declares TWO language prefixes. This is the load
    path nobody exercises until a multi-rank run aborts on it.
    """
    model = SimpleNamespace(vision_tower=object())
    assert _dormant_modality_towers(model, family=_spec(GEMMA4), image_declared=False) == [
        "model.vision_tower"
    ]


def test_tower_attribute_explicitly_set_to_none_counts_as_absent():
    """If this fails, a model whose tower attribute exists but is None would be announced as having
    a dormant tower that isn't really there."""
    model = SimpleNamespace(model=SimpleNamespace(vision_tower=None))
    assert _dormant_modality_towers(model, family=_spec(GEMMA4), image_declared=False) == []


def test_inner_model_object_lacking_towers_reports_nothing_dormant():
    """If this fails, models with a .model submodule but no towers would be misreported as having
    dormant modality towers."""
    model = SimpleNamespace(model=SimpleNamespace(layers=object()))
    assert _dormant_modality_towers(model, family=_spec(GEMMA4), image_declared=False) == []


def test_qwen35_visual_tower_is_found_where_the_hardcoded_table_could_not_be():
    """If this fails, the measured #523 abort returns.

    The deleted table listed Gemma4's spellings. qwen3.5 calls its tower
    ``model.visual``, so the walk found nothing, ddp_find_unused_parameters
    stayed False, and every multi-rank qwen3.5 run died on the first backward.
    """
    model = SimpleNamespace(model=SimpleNamespace(visual=object()))
    assert _dormant_modality_towers(model, family=_spec("qwen3.5"), image_declared=False) == [
        "model.visual"
    ]


def test_an_unregistered_family_makes_no_dormancy_claim():
    """If this fails, None reads as "no towers" -- a measurement this function cannot make for a
    family nobody declared."""
    model = SimpleNamespace(model=SimpleNamespace(vision_tower=object()))
    assert _dormant_modality_towers(model, family=None, image_declared=False) == []


@pytest.mark.parametrize("image_declared", [True, False])
def test_a_none_modality_tower_never_forces_the_flag(image_declared: bool):
    """If this fails, qwen3.5's ``mtp`` head -- out of adapter scope but not a modality any
    declaration could exercise -- forces find_unused_parameters on for every run."""
    model = SimpleNamespace(mtp=object())
    assert (
        _dormant_modality_towers(model, family=_spec("qwen3.5"), image_declared=image_declared)
        == []
    )
