"""Tests for the speech model registry (PHASE 2a, docs/research/upstream_integration.md).

Three things must hold here and nothing may quietly stop holding:

* the six real entries are self-consistent -- ``registry_problems() == []`` against the
  REAL family registry (the lazy default import is exercised, not bypassed), so the
  tree's own catalogue can never accumulate a contradiction;
* every refusal code fires on a deliberately broken entry -- a code that can never fire
  is not a check, and a test that only walks the clean path cannot tell a working
  invariant from a missing one. Each crafted entry breaks exactly one rule so the
  reported code is attributable;
* ``reproduced`` compares numbers the way acceptance must: inclusive at the boundary,
  False whenever either side of the comparison is missing. The real entries sit at the
  failure end of that -- every card reports test-clean while we hold dev-clean -- so all
  six are EXPERIMENTAL and none claims reproduction.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from foundationscale.families.registry import REGISTRY as FAMILY_REGISTRY
from foundationscale.upstream.models import (
    MODELS,
    Backend,
    ModelEntry,
    ModelKind,
    ReferenceResult,
    SupportStatus,
    entries_for_backend,
    get_model,
    registry_problems,
)

_FAMILY_NAMES = frozenset(spec.name for spec in FAMILY_REGISTRY)

# One consistent entry per backend, mutated below into one inconsistency at a time. A
# fixture that broke two rules at once could pass on the wrong code.
_BASE_HF = ModelEntry(
    id="base-hf",
    backend=Backend.HF,
    upstream_ref="org/base-hf",
    kind=ModelKind.AUDIO_LLM,
    license_weights="apache-2.0",
    family="gemma4",
)
_BASE_NEMO = ModelEntry(
    id="base-nemo",
    backend=Backend.NEMO,
    upstream_ref="nvidia/base-nemo",
    kind=ModelKind.AED,
    license_weights="cc-by-4.0",
    nemo_class="nemo.collections.asr.models.EncDecMultiTaskModel",
    trainable_prefixes=("encoder", "transf_decoder"),
)


def _problems(*models: ModelEntry) -> list[str]:
    """registry_problems for crafted entries, with family names stated explicitly.

    Crafted entries name their own families so these tests cannot drift with a
    registration elsewhere in the tree; the real registry's names are exercised on the
    real MODELS below.
    """
    return registry_problems(models, family_names=_FAMILY_NAMES)


def _reference(
    upstream_value: float | None,
    measured: float | None,
    tolerance: float = 0.5,
) -> ReferenceResult:
    """A reference with a single controlled variable, for the reproduced boundary.

    0.5 and x.0/x.5 values are exactly representable in binary floating point, so
    "exactly at tolerance" is the boundary itself and not a rounding artifact.
    """
    return ReferenceResult(
        task="librispeech_test_clean",
        metric="wer",
        upstream_value=upstream_value,
        source="https://example.invalid/model-card",
        tolerance=tolerance,
        measured=measured,
    )


def _supported(reference: ReferenceResult | None) -> ModelEntry:
    """The consistent HF base entry, claiming SUPPORTED on top of ``reference``."""
    return replace(_BASE_HF, status=SupportStatus.SUPPORTED, reference=reference)


def test_crafted_base_entries_are_consistent() -> None:
    """The fixtures must start clean, or every code-fired test below is untrustworthy."""
    assert _problems(_BASE_HF, _BASE_NEMO) == []


def test_duplicate_id() -> None:
    assert _problems(_BASE_HF, replace(_BASE_HF)) == ["duplicate_id:base-hf"]


@pytest.mark.parametrize("field", ["id", "upstream_ref", "license_weights"])
def test_empty_field_names_the_field(field: str) -> None:
    """Identification and licensing fields may never be blank, and the code names which."""
    entry = replace(_BASE_HF, **{field: ""})
    assert _problems(entry) == [f"empty_field:{entry.id}:{field}"]


def test_hf_without_family() -> None:
    assert _problems(replace(_BASE_HF, family="")) == ["hf_without_family:base-hf"]


def test_unknown_family() -> None:
    """A family name outside the given set is refused, naming the name it saw."""
    entry = replace(_BASE_HF, family="nonexistent_family")
    assert _problems(entry) == ["unknown_family:base-hf:nonexistent_family"]


def test_hf_with_nemo_class() -> None:
    entry = replace(_BASE_HF, nemo_class="nemo.collections.asr.models.EncDecMultiTaskModel")
    assert _problems(entry) == ["hf_with_nemo_class:base-hf"]


def test_nemo_without_class() -> None:
    entry = replace(_BASE_NEMO, nemo_class="")
    assert _problems(entry) == ["nemo_without_class:base-nemo"]


def test_nemo_without_scopes() -> None:
    """NeMo entries must declare what trains: adjudication reads those prefixes."""
    entry = replace(_BASE_NEMO, trainable_prefixes=())
    assert _problems(entry) == ["nemo_without_scopes:base-nemo"]


def test_overlapping_scopes() -> None:
    """The same prefix trainable and frozen is a contradiction, not an ordering hint."""
    entry = replace(
        _BASE_NEMO,
        trainable_prefixes=("encoder", "transf_decoder"),
        frozen_prefixes=("transf_decoder",),
    )
    assert _problems(entry) == ["overlapping_scopes:base-nemo"]


def test_lora_marker_without_frozen() -> None:
    entry = replace(_BASE_HF, lora_marker=".lora_")
    assert _problems(entry) == ["lora_marker_without_frozen:base-hf"]


def test_lora_marker_with_frozen_scope_is_clean() -> None:
    """The designed carve-out -- PEFT tensors inside a frozen scope -- is not a problem."""
    entry = replace(_BASE_NEMO, frozen_prefixes=("llm",), lora_marker=".lora_")
    assert _problems(entry) == []


def test_supported_without_reference_is_reported() -> None:
    assert _problems(_supported(None)) == ["supported_without_reproduction:base-hf"]


def test_supported_without_measured_value_is_reported() -> None:
    """The card's number alone is not a reproduction; 'we ran it and it looked fine' is not data."""
    assert _problems(_supported(_reference(upstream_value=4.0, measured=None))) == [
        "supported_without_reproduction:base-hf"
    ]


def test_supported_with_failed_reproduction_is_reported() -> None:
    """A measured value outside the registered tolerance must not carry a SUPPORTED label."""
    assert _problems(_supported(_reference(upstream_value=4.0, measured=6.0, tolerance=0.3))) == [
        "supported_without_reproduction:base-hf"
    ]


def test_supported_with_reproduction_is_clean() -> None:
    """And the one path that earns SUPPORTED reports nothing."""
    assert _problems(_supported(_reference(upstream_value=4.0, measured=4.1, tolerance=0.3))) == []


def test_reproduced_exactly_at_tolerance_passes() -> None:
    """The boundary is inclusive: tolerance is the width of 'same number, different rounding'."""
    assert _reference(upstream_value=2.0, measured=2.5).reproduced


def test_reproduced_beyond_tolerance_fails() -> None:
    assert not _reference(upstream_value=2.0, measured=2.6).reproduced


def test_reproduced_without_measurement_fails() -> None:
    """Missing data is not zero and not agreement -- until we measure, nothing reproduces."""
    assert not _reference(upstream_value=2.0, measured=None).reproduced


def test_reproduced_without_upstream_value_fails() -> None:
    """A measured number with no card claim cannot reproduce it (the current card state)."""
    assert not _reference(upstream_value=None, measured=2.0).reproduced


def test_reproduced_with_no_numbers_at_all_fails() -> None:
    assert not _reference(upstream_value=None, measured=None).reproduced


def test_registry_holds_the_six_speech_models() -> None:
    """Pin the transcription itself: ids, in declaration order."""
    assert len(MODELS) == 6
    assert [entry.id for entry in MODELS] == [
        "gemma-4-e4b-it",
        "whisper-large-v3",
        "qwen2-audio-7b-instruct",
        "parakeet-ctc-1.1b",
        "canary-1b-flash",
        "canary-qwen-2.5b",
    ]


def test_upstream_refs_are_transcribed_exactly_as_upstream_spells_them() -> None:
    """A locator is a case-sensitive contract (google/gemma-4-E4B-it), so it is pinned here."""
    assert {entry.id: entry.upstream_ref for entry in MODELS} == {
        "gemma-4-e4b-it": "google/gemma-4-E4B-it",
        "whisper-large-v3": "openai/whisper-large-v3",
        "qwen2-audio-7b-instruct": "Qwen/Qwen2-Audio-7B-Instruct",
        "parakeet-ctc-1.1b": "nvidia/parakeet-ctc-1.1b",
        "canary-1b-flash": "nvidia/canary-1b-flash",
        "canary-qwen-2.5b": "nvidia/canary-qwen-2.5b",
    }


def test_every_reference_targets_a_card_test_set_with_a_fixed_tolerance() -> None:
    """Each claim names the card's own LibriSpeech test split and a tolerance set in advance."""
    for entry in MODELS:
        reference = entry.reference
        assert reference is not None, f"{entry.id} declares no reference"
        assert reference.task in ("librispeech_test_clean", "librispeech_test_other")
        assert reference.metric == "wer"
        assert reference.tolerance == 0.3


def test_supported_means_reproduced_and_only_reproduced_models_are_supported() -> None:
    """Rule 4: the three models whose cards publish a LibriSpeech result were reproduced on the
    full split (validation_campaigns/speech_repro, 2026-10-10); the others publish none and stay
    experimental."""
    assert len(MODELS) == 6
    supported = {e.id for e in MODELS if e.status is SupportStatus.SUPPORTED}
    assert supported == {"parakeet-ctc-1.1b", "canary-1b-flash", "canary-qwen-2.5b"}
    for entry in MODELS:
        assert entry.reference is not None
        if entry.status is SupportStatus.SUPPORTED:
            assert entry.reference.reproduced, entry.id
            assert entry.reference.source.startswith("https://huggingface.co/")
        else:
            assert entry.reference.upstream_value is None and entry.reference.measured is None


def test_entries_for_backend_counts() -> None:
    """4 HF transformers entries, 2 NeMo entries, in registry order for both spellings."""
    assert [entry.id for entry in entries_for_backend(Backend.HF)] == [
        "gemma-4-e4b-it",
        "whisper-large-v3",
        "qwen2-audio-7b-instruct",
        "parakeet-ctc-1.1b",
    ]
    assert [entry.id for entry in entries_for_backend(Backend.NEMO)] == [
        "canary-1b-flash",
        "canary-qwen-2.5b",
    ]
    assert len(entries_for_backend(Backend.HF)) == 4
    assert len(entries_for_backend(Backend.NEMO)) == 2
    assert entries_for_backend("hf") == entries_for_backend(Backend.HF)
    assert entries_for_backend("nemo") == entries_for_backend(Backend.NEMO)


def test_every_hf_family_is_a_registered_family() -> None:
    """The registry's family names must point at real FamilySpec registrations, and the
    NeMo entries must not carry a family at all (their architecture is ``nemo_class``)."""
    for entry in MODELS:
        if entry.backend is Backend.HF:
            assert entry.family in _FAMILY_NAMES, (
                f"{entry.id} declares family {entry.family!r}, which no FamilySpec "
                f"registers; registered names are {sorted(_FAMILY_NAMES)}"
            )
        else:
            assert entry.family == ""


def test_get_model_finds_each_registered_id() -> None:
    for entry in MODELS:
        assert get_model(entry.id) == entry


def test_get_model_refuses_an_unknown_id_naming_the_known_ones() -> None:
    with pytest.raises(ValueError) as excinfo:
        get_model("not-a-model")
    message = str(excinfo.value)
    assert "not-a-model" in message
    for known in (
        "gemma-4-e4b-it",
        "whisper-large-v3",
        "qwen2-audio-7b-instruct",
        "parakeet-ctc-1.1b",
        "canary-1b-flash",
        "canary-qwen-2.5b",
    ):
        assert known in message


def test_real_models_pass_registry_problems() -> None:
    """The tree's own catalogue is consistent, against the lazy real-registry default
    AND against the names read here -- the two paths must agree."""
    assert registry_problems() == []
    assert registry_problems(MODELS, family_names=_FAMILY_NAMES) == []


def test_reproduced_boundary_survives_float_addition() -> None:
    """1.61 + 0.3 is 1.9100000000000001 in floating point; exactly-at-tolerance must still pass."""
    assert _reference(upstream_value=1.61, measured=1.61 + 0.3, tolerance=0.3).reproduced
    assert not _reference(upstream_value=1.61, measured=1.61 + 0.31, tolerance=0.3).reproduced
