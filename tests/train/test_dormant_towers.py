"""Tests for ``_dormant_modality_towers`` (see #504).

The helper must key dormancy on the run's declaration and the loaded module
tree -- never on the data -- so that a text-only run on a VLM, a multimodal
run on an omni model, and a plain LLM stay three distinguishable cases.
"""

from types import SimpleNamespace

from foundationscale.train.loop import _MODALITY_TOWERS, _dormant_modality_towers


def test_plain_llm_with_neither_tower_reports_nothing_dormant():
    """If this fails, plain LLMs would silently tolerate genuinely unused parameters that DDP should
    report."""
    model = SimpleNamespace()
    assert _dormant_modality_towers(model, image_declared=False) == []


def test_vision_tower_exercised_by_image_declaration_is_not_dormant():
    """If this fails, a multimodal run on a VLM would wrongly flag its actively trained vision tower
    as dormant."""
    model = SimpleNamespace(vision_tower=object())
    assert _dormant_modality_towers(model, image_declared=True) == []


def test_vision_tower_without_image_declaration_is_dormant():
    """If this fails, a text-only run on a VLM would leave its vision tower frozen without anyone
    being told."""
    model = SimpleNamespace(vision_tower=object())
    assert _dormant_modality_towers(model, image_declared=False) == ["vision_tower"]


def test_omni_model_with_image_declared_leaves_only_audio_dormant():
    """If this fails, a multimodal run on an omni model would either miss the dormant audio tower or
    flag the exercised vision tower."""
    model = SimpleNamespace(vision_tower=object(), audio_tower=object())
    assert _dormant_modality_towers(model, image_declared=True) == ["audio_tower"]


def test_omni_model_without_image_declaration_reports_both_towers_in_declaration_order():
    """If this fails, the announcement would name towers in an order divorced from _MODALITY_TOWERS,
    breaking the declaration-order contract."""
    model = SimpleNamespace(vision_tower=object(), audio_tower=object())
    dormant = _dormant_modality_towers(model, image_declared=False)
    assert dormant == ["vision_tower", "audio_tower"]
    assert dormant == [attr for attr, _ in _MODALITY_TOWERS]


def test_towers_nested_under_inner_model_attribute_are_still_found():
    """If this fails, composite VLMs that nest towers one level down would have their dormant towers
    missed entirely."""
    model = SimpleNamespace(model=SimpleNamespace(vision_tower=object()))
    assert _dormant_modality_towers(model, image_declared=False) == ["vision_tower"]


def test_tower_attribute_explicitly_set_to_none_counts_as_absent():
    """If this fails, a model whose tower attribute exists but is None would be announced as having
    a dormant tower that isn't really there."""
    model = SimpleNamespace(vision_tower=None)
    assert _dormant_modality_towers(model, image_declared=False) == []


def test_inner_model_object_lacking_towers_reports_nothing_dormant():
    """If this fails, models with a .model submodule but no towers would be misreported as having
    dormant modality towers."""
    model = SimpleNamespace(model=SimpleNamespace(layers=object()))
    assert _dormant_modality_towers(model, image_declared=False) == []
