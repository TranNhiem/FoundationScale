"""Tests for the model-family registry.

These run under ``FS_FORBID_SKIPS=1``, so there is no skip, no importorskip and
no xfail anywhere in this file. Nothing here touches torch, the network, or a
checkpoint: the registry is data, and data is testable as data.

The config dictionaries below are the SHAPES measured from the seven real
checkpoints on the estate on 2026-09-20, reduced to the keys the registry reads.
Keeping them here means a future vendor rename breaks a test rather than a run.
"""

from __future__ import annotations

import pytest

from foundationscale.families.registry import (
    REGISTRY,
    FamilySpec,
    _assert_no_duplicate_model_types,
    resolve_family,
    unregistered_family_reason,
)

# The measured estate, reduced to what resolution reads. `experts` is carried so
# the composite-config point stays visible: it is stated on text_config, never at
# the top level, on every one of these.
MEASURED_CHECKPOINTS: tuple[tuple[str, dict[str, object], str], ...] = (
    (
        "gemma-4-E4B-it",
        {"model_type": "gemma4", "text_config": {"model_type": "gemma4_text"}},
        "gemma4",
    ),
    (
        "gemma-4-12B-it",
        {"model_type": "gemma4_unified", "text_config": {"model_type": "gemma4_unified_text"}},
        "gemma4",
    ),
    (
        "gemma-4-26B-A4B",
        {"model_type": "gemma4", "text_config": {"model_type": "gemma4_text", "num_experts": 128}},
        "gemma4",
    ),
    (
        "gemma-4-31B",
        {"model_type": "gemma4", "text_config": {"model_type": "gemma4_text"}},
        "gemma4",
    ),
    (
        "Qwen3.5-27B",
        {"model_type": "qwen3_5", "text_config": {"model_type": "qwen3_5_text"}},
        "qwen3.5",
    ),
    (
        "Qwen3.5-35B-A3B",
        {
            "model_type": "qwen3_5_moe",
            "text_config": {"model_type": "qwen3_5_moe_text", "num_experts": 256},
        },
        "qwen3.5",
    ),
    (
        "Qwen3.5-122B-A10B",
        {
            "model_type": "qwen3_5_moe",
            "text_config": {"model_type": "qwen3_5_moe_text", "num_experts": 256},
        },
        "qwen3.5",
    ),
)


@pytest.mark.parametrize(("label", "config", "expected"), MEASURED_CHECKPOINTS)
def test_every_measured_checkpoint_resolves(
    label: str, config: dict[str, object], expected: str
) -> None:
    spec = resolve_family(config)
    assert spec is not None, f"{label} did not resolve to any registered family"
    assert spec.name == expected, f"{label} resolved to {spec.name!r}, expected {expected!r}"


def test_resolves_by_top_level_model_type() -> None:
    assert resolve_family({"model_type": "gemma4"}) is not None


def test_resolves_by_nested_text_config_model_type() -> None:
    # The top level states something nobody claims; the text sub-config states
    # something we know. A composite VLM config really does look like this.
    spec = resolve_family(
        {"model_type": "some_wrapper", "text_config": {"model_type": "qwen3_5_text"}}
    )
    assert spec is not None
    assert spec.name == "qwen3.5"


def test_unknown_family_resolves_to_none_not_a_default() -> None:
    assert resolve_family({"model_type": "llama_9_ultra"}) is None


def test_config_without_any_model_type_resolves_to_none() -> None:
    assert resolve_family({}) is None


def test_path_like_hints_do_not_influence_resolution() -> None:
    # A directory called gemma-4-31B must not resolve anything. Resolution reads
    # model_type and only model_type; a path is a naming convention.
    assert resolve_family({"_name_or_path": "/weights/Google/Gemma4/gemma-4-31B"}) is None


def test_refusal_names_what_was_tried_and_what_is_known() -> None:
    reason = unregistered_family_reason({"model_type": "llama_9_ultra"})
    assert "llama_9_ultra" in reason
    assert "gemma4" in reason and "qwen3_5" in reason
    # A refusal that does not say how to proceed is just a stop, and "declare
    # them somehow" is not saying how: it has to name the flag and the class.
    assert "--adapter-target" in reason
    assert "FamilySpec" in reason


def test_refusal_is_explicit_when_no_model_type_is_stated() -> None:
    reason = unregistered_family_reason({})
    assert "none" in reason.lower()


def test_no_model_type_is_claimed_twice() -> None:
    claims: dict[str, str] = {}
    for spec in REGISTRY:
        for model_type in spec.model_types:
            assert model_type not in claims, (
                f"{model_type!r} claimed by {claims.get(model_type)!r} and {spec.name!r}"
            )
            claims[model_type] = spec.name


def test_duplicate_claim_is_rejected_at_registration() -> None:
    a = FamilySpec(
        name="a",
        model_types=("shared_type",),
        language_prefixes=("model.layers",),
        towers=(),
        adapter_leaf_modules=("q_proj",),
        expert_count_path=(),
    )
    b = FamilySpec(
        name="b",
        model_types=("shared_type",),
        language_prefixes=("model.layers",),
        towers=(),
        adapter_leaf_modules=("q_proj",),
        expert_count_path=(),
    )
    with pytest.raises(ValueError, match="shared_type"):
        _assert_no_duplicate_model_types((a, b))


def test_overlapping_language_and_tower_prefixes_are_rejected() -> None:
    with pytest.raises(ValueError, match="overlapping scopes"):
        FamilySpec(
            name="overlapping",
            model_types=("x",),
            language_prefixes=("model",),
            towers=(("model.vision_tower", "image"),),
            adapter_leaf_modules=("q_proj",),
            expert_count_path=(),
        )


def test_reversed_overlap_is_also_rejected() -> None:
    with pytest.raises(ValueError, match="overlapping scopes"):
        FamilySpec(
            name="reversed",
            model_types=("x",),
            language_prefixes=("model.language_model.layers",),
            towers=(("model.language_model", None),),
            adapter_leaf_modules=("q_proj",),
            expert_count_path=(),
        )


@pytest.mark.parametrize("field", ["model_types", "language_prefixes", "adapter_leaf_modules"])
def test_load_bearing_fields_may_not_be_empty(field: str) -> None:
    kwargs: dict[str, object] = {
        "name": "empty",
        "model_types": ("x",),
        "language_prefixes": ("model.layers",),
        "towers": (),
        "adapter_leaf_modules": ("q_proj",),
        "expert_count_path": (),
    }
    kwargs[field] = ()
    with pytest.raises(ValueError, match=field):
        FamilySpec(**kwargs)  # type: ignore[arg-type]


def test_tower_prefixes_may_be_empty_because_some_families_have_no_towers() -> None:
    spec = FamilySpec(
        name="text_only_family",
        model_types=("x",),
        language_prefixes=("model.layers",),
        towers=(),
        adapter_leaf_modules=("q_proj",),
        expert_count_path=(),
    )
    assert spec.tower_prefixes == ()


def test_registry_declares_the_two_families_measured_on_this_estate() -> None:
    names = {spec.name for spec in REGISTRY}
    assert names == {"gemma4", "qwen3.5"}


def test_gemma4_declares_an_audio_tower_and_qwen_does_not() -> None:
    # Not a trivia test. This asymmetry is the reason dormancy and adapter scope
    # cannot be global: the same text-only corpus leaves two towers unused on
    # Gemma4 and one on Qwen3.5.
    by_name = {spec.name: spec for spec in REGISTRY}
    assert any("audio" in p for p in by_name["gemma4"].tower_prefixes)
    assert not any("audio" in p for p in by_name["qwen3.5"].tower_prefixes)


def test_expert_count_path_is_nested_not_flat() -> None:
    # A flat config.get("num_experts") reads None on every MoE checkpoint here
    # and concludes the model is dense.
    for spec in REGISTRY:
        assert spec.expert_count_path == ("text_config", "num_experts")
