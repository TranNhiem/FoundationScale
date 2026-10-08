"""Unit tests for --fused-loss's dispatch/refusal logic (train/fused_loss.py).

:func:`fused_loss_refusal_reason` and the model_type tuples are pure and
import-free (no liger_kernel/transformers needed), so they are fully covered
here. :func:`apply_fused_loss` itself (the actual monkeypatch) needs
liger_kernel/transformers/peft and real models to mean anything, and is
exercised by the GPU proof and tests/train/test_fused_loss_parity.py instead.
"""

from __future__ import annotations

import pytest

from foundationscale.train.fused_loss import (
    FUSED_LOSS_BACKENDS,
    FUSED_LOSS_SUPPORTED_MODEL_TYPES,
    LIGER_CUSTOM_MODEL_TYPES,
    LIGER_GENERIC_MODEL_TYPES,
    apply_fused_loss,
    fused_loss_refusal_reason,
)


def test_covered_model_types_are_not_refused() -> None:
    for model_type in FUSED_LOSS_SUPPORTED_MODEL_TYPES:
        assert fused_loss_refusal_reason("liger", model_type) is None


def test_unsupported_model_type_is_refused_naming_it_and_the_list() -> None:
    refusal = fused_loss_refusal_reason("liger", "llama")
    assert refusal is not None
    assert "llama" in refusal
    for model_type in FUSED_LOSS_SUPPORTED_MODEL_TYPES:
        assert model_type in refusal


def test_none_model_type_is_refused_without_crashing() -> None:
    refusal = fused_loss_refusal_reason("liger", None)
    assert refusal is not None
    assert "model_type" in refusal


def test_unknown_backend_is_refused() -> None:
    refusal = fused_loss_refusal_reason("not-a-real-backend", "gemma4")
    assert refusal is not None
    assert "not-a-real-backend" in refusal


def test_generic_and_custom_lists_are_disjoint_and_union_is_supported() -> None:
    assert set(LIGER_GENERIC_MODEL_TYPES).isdisjoint(LIGER_CUSTOM_MODEL_TYPES)
    assert set(FUSED_LOSS_SUPPORTED_MODEL_TYPES) == set(LIGER_GENERIC_MODEL_TYPES) | set(
        LIGER_CUSTOM_MODEL_TYPES
    )


def test_measured_coverage_matches_the_installed_liger_kernel_version() -> None:
    # Pinned to what was MEASURED against liger_kernel 0.8.4's
    # MODEL_TYPE_TO_APPLY_LIGER_FN at patch-design time (see the module
    # docstring); a future liger_kernel bump that adds or removes one of
    # these should be a deliberate edit here, not a silent drift.
    assert LIGER_GENERIC_MODEL_TYPES == (
        "gemma4",
        "qwen3_5",
        "qwen3_5_text",
        "qwen3_5_moe",
        "qwen3_5_moe_text",
    )
    assert LIGER_CUSTOM_MODEL_TYPES == ("gemma4_unified",)
    assert FUSED_LOSS_BACKENDS == ("liger",)


def test_apply_fused_loss_rejects_unsupported_model_type_as_a_caller_defect() -> None:
    # Reaching apply_fused_loss with an unsupported model_type is a CALLER
    # defect (the caller must check fused_loss_refusal_reason first), so this
    # raises rather than refusing -- the same split cli.py's _build_config
    # draws between a rejected declaration (ValueError, refuse) and a crash.
    with pytest.raises(ValueError, match="unsupported model_type"):
        apply_fused_loss(object(), "llama")
