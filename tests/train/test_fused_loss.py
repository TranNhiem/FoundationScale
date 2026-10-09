"""Unit tests for --fused-loss's dispatch/refusal logic (train/fused_loss.py).

:func:`fused_loss_refusal_reason` and the model_type tuples are pure and
import-free (no liger_kernel/transformers needed), so they are fully covered
here. The rest of this file drives the ACTUAL monkeypatch
(:func:`apply_fused_loss`, :func:`_ensure_liger_flce_is_dtensor_safe` and
:func:`_fs_gemma4_unified_lce_forward`) with ``liger_kernel`` stubbed into
``sys.modules`` -- real torch (CPU), a pure-torch reference standing in for
the fused kernel itself. Two things stay genuinely GPU-only and are left to
the GPU proof and tests/train/test_fused_loss_parity.py instead: whether the
REAL liger_kernel/transformers resolve these exact module paths (this file
only proves this module's OWN branching and plumbing), and the numeric parity
between the fused and stock forward on a real model.
"""

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from foundationscale.train.fused_loss import (
    FUSED_LOSS_BACKENDS,
    FUSED_LOSS_SUPPORTED_MODEL_TYPES,
    LIGER_CUSTOM_MODEL_TYPES,
    LIGER_GENERIC_MODEL_TYPES,
    _ensure_liger_flce_is_dtensor_safe,
    _fs_gemma4_unified_lce_forward,
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
    # The check runs BEFORE any liger import, so this is true even where
    # liger_kernel is not installed at all -- confirmed by the absence of any
    # sys.modules stub in this test, unlike every test below it in this file.
    assert "liger_kernel" not in sys.modules


# ---------------------------------------------------------------------------
# _ensure_liger_flce_is_dtensor_safe / apply_fused_loss -- with liger_kernel
# stubbed into sys.modules (the package is not installed in this environment;
# see the module docstring's "CI has no GPU ... no liger_kernel"). Only the
# ONE function this plane actually calls (liger_fused_linear_cross_entropy)
# is faked; everything around it is this module's own real code.
# ---------------------------------------------------------------------------


def _install_fake_liger_functional(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    """Stub liger_kernel.transformers.functional, the DTensor-safety choke point.

    Every level of the dotted path must be present in sys.modules for
    ``import liger_kernel.transformers.functional as _liger_functional`` to
    resolve without a real liger_kernel installed -- Python's import system
    walks each segment and is satisfied by a pre-registered entry at each one.
    """
    functional_mod = ModuleType("liger_kernel.transformers.functional")

    def _original_flce(
        input: Any, weight: Any, target: Any, bias: Any = None, *args: Any, **kwargs: Any
    ) -> str:
        return "unwrapped-loss"

    functional_mod.liger_fused_linear_cross_entropy = _original_flce  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "liger_kernel", ModuleType("liger_kernel"))
    monkeypatch.setitem(
        sys.modules, "liger_kernel.transformers", ModuleType("liger_kernel.transformers")
    )
    monkeypatch.setitem(sys.modules, "liger_kernel.transformers.functional", functional_mod)
    return functional_mod


def test_ensure_liger_flce_is_dtensor_safe_materializes_dtensor_args(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The measured FSDP2 fix: DTensor args are full_tensor()'d; plain ones pass through.

    A fake DTensor class stands in for ``torch.distributed.tensor.DTensor`` --
    real DTensor construction needs an initialized process group this test has
    none of, and the fix's own logic (materialize IF DTensor, else pass
    through) does not care which concrete class answers ``isinstance``, only
    that the wrapper asks the question and acts on the answer.
    """
    functional_mod = _install_fake_liger_functional(monkeypatch)
    original = functional_mod.liger_fused_linear_cross_entropy

    class _FakeDTensor:
        def __init__(self, local: Any) -> None:
            self._local = local

        def full_tensor(self) -> Any:
            return self._local

    dtensor_mod = ModuleType("torch.distributed.tensor")
    dtensor_mod.DTensor = _FakeDTensor  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "torch.distributed.tensor", dtensor_mod)

    _ensure_liger_flce_is_dtensor_safe()

    wrapped = functional_mod.liger_fused_linear_cross_entropy
    assert wrapped is not original  # the module attribute was really replaced

    sharded_weight = _FakeDTensor("full-weight")
    result = wrapped("plain-input", sharded_weight, "plain-target")
    assert result == "unwrapped-loss"  # the real function still ran, on materialized args

    # Re-call the ORIGINAL directly to see exactly what the wrapper passed it.
    captured: dict[str, Any] = {}

    def _spy(input: Any, weight: Any, target: Any, bias: Any = None, **kwargs: Any) -> str:
        captured["args"] = (input, weight, target, bias)
        return "spied-loss"

    # Rebuild the wrapper around the spy to inspect its materialized call --
    # `_ensure_liger_flce_is_dtensor_safe` is idempotent per-function (it
    # checks `_fs_dtensor_safe` on the CURRENT target), so wrapping a fresh
    # callable produces a fresh wrapper around it.
    functional_mod.liger_fused_linear_cross_entropy = _spy
    _ensure_liger_flce_is_dtensor_safe()
    functional_mod.liger_fused_linear_cross_entropy("plain-input", sharded_weight, "plain-target")
    assert captured["args"] == ("plain-input", "full-weight", "plain-target", None)


def test_apply_fused_loss_generic_model_type_applies_liger_kernel_and_names_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A LIGER_GENERIC_MODEL_TYPES model_type dispatches to _apply_liger_kernel_to_instance.

    Every other kernel swap is explicitly OFF -- this flag's contract is
    "only the loss changes" (see the module docstring's NUMERICS section) --
    asserted here as the exact flags dict, not merely that the call happened.
    """
    _install_fake_liger_functional(monkeypatch)  # for _ensure_liger_flce_is_dtensor_safe

    calls: dict[str, Any] = {}

    def _apply_liger_kernel_to_instance(
        *,
        model: Any,
        rope: bool,
        cross_entropy: bool,
        fused_linear_cross_entropy: bool,
        rms_norm: bool,
        geglu: bool,
        swiglu: bool,
        layer_norm: bool,
    ) -> None:
        calls["model"] = model
        calls["flags"] = {
            "rope": rope,
            "cross_entropy": cross_entropy,
            "fused_linear_cross_entropy": fused_linear_cross_entropy,
            "rms_norm": rms_norm,
            "geglu": geglu,
            "swiglu": swiglu,
            "layer_norm": layer_norm,
        }

    transformers_pkg = ModuleType("liger_kernel.transformers")
    transformers_pkg._apply_liger_kernel_to_instance = _apply_liger_kernel_to_instance  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "liger_kernel.transformers", transformers_pkg)

    fake_model = object()
    message = apply_fused_loss(fake_model, "gemma4")

    assert calls["model"] is fake_model
    assert calls["flags"] == {
        "rope": False,
        "cross_entropy": False,
        "fused_linear_cross_entropy": True,
        "rms_norm": False,
        "geglu": False,
        "swiglu": False,
        "layer_norm": False,
    }
    assert "gemma4" in message
    assert "_apply_liger_kernel_to_instance" in message


def test_apply_fused_loss_gemma4_unified_patches_the_forward_method(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The FS-owned custom path: no _apply_liger_kernel_to_instance needed at all."""
    _install_fake_liger_functional(monkeypatch)

    class _FakeModel:
        pass

    model = _FakeModel()
    message = apply_fused_loss(model, "gemma4_unified")

    assert model.forward.__func__ is _fs_gemma4_unified_lce_forward  # type: ignore[attr-defined]
    assert "gemma4_unified" in message
    assert "FS-owned forward patch" in message


# ---------------------------------------------------------------------------
# _fs_gemma4_unified_lce_forward -- a tiny fake model, real torch, and a
# pure-torch reference standing in for LigerForCausalLMLoss (so the logic
# AROUND the fused kernel -- softcapping, return_dict, labels-vs-no-labels --
# is exercised, not the kernel's own fused numerics, which is GPU-only).
# ---------------------------------------------------------------------------


def _install_fake_liger_loss_utils(monkeypatch: pytest.MonkeyPatch, fake_loss: Any) -> None:
    loss_utils_mod = ModuleType("liger_kernel.transformers.model.loss_utils")
    loss_utils_mod.LigerForCausalLMLoss = fake_loss  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "liger_kernel", ModuleType("liger_kernel"))
    monkeypatch.setitem(
        sys.modules, "liger_kernel.transformers", ModuleType("liger_kernel.transformers")
    )
    monkeypatch.setitem(
        sys.modules,
        "liger_kernel.transformers.model",
        ModuleType("liger_kernel.transformers.model"),
    )
    monkeypatch.setitem(sys.modules, "liger_kernel.transformers.model.loss_utils", loss_utils_mod)


class _FakeLmHead:
    """``self.lm_head``: callable (the non-fused path) AND carries ``.weight``
    (what the fused path hands LigerForCausalLMLoss instead of calling it)."""

    def __init__(self, weight: Any) -> None:
        self.weight = weight

    def __call__(self, hidden_states: Any) -> Any:
        return hidden_states @ self.weight.t()


def _fake_self(
    hidden_states: Any, lm_head_weight: Any, final_logit_softcapping: float | None
) -> Any:
    class _InnerModel:
        """``self.model(...)``: a plain class (not SimpleNamespace) so it is
        actually CALLABLE via ``__call__``, matching the stock forward's own
        ``outputs = self.model(...)`` shape."""

        def __call__(self, **kwargs: Any) -> Any:
            return SimpleNamespace(
                last_hidden_state=hidden_states,
                past_key_values="pkv",
                hidden_states="hs",
                attentions="attn",
                image_hidden_states="ihs",
                audio_hidden_states="ahs",
                shared_kv_states="skv",
            )

    text_config = SimpleNamespace(
        hidden_size=hidden_states.shape[-1], final_logit_softcapping=final_logit_softcapping
    )
    return SimpleNamespace(
        model=_InnerModel(),
        config=SimpleNamespace(get_text_config=lambda: text_config),
        lm_head=_FakeLmHead(lm_head_weight),
    )


def _pure_torch_liger_loss(
    *,
    hidden_states: Any,
    lm_head_weight: Any,
    labels: Any,
    hidden_size: int,
    final_logit_softcapping: float | None,
    **kwargs: Any,
) -> Any:
    """A pure-torch reference standing in for liger's fused kernel.

    Exercises exactly what the production code hands it (hidden_states,
    lm_head_weight, labels, hidden_size, final_logit_softcapping, plus
    whatever else arrived in **kwargs) without claiming to BE the fused
    kernel -- this is the module-under-test's plumbing, not a numerics test.
    """
    import torch

    assert hidden_size == hidden_states.shape[-1]
    logits = hidden_states @ lm_head_weight.t()
    if final_logit_softcapping is not None:
        logits = torch.tanh(logits / final_logit_softcapping) * final_logit_softcapping
    return torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.size(-1)), labels.reshape(-1)
    )


def test_gemma4_unified_lce_forward_with_labels_takes_the_fused_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import torch

    _install_fake_liger_loss_utils(monkeypatch, _pure_torch_liger_loss)
    hidden_states = torch.randn(1, 3, 4)
    lm_head_weight = torch.randn(5, 4)
    labels = torch.randint(0, 5, (1, 3))
    self_obj = _fake_self(hidden_states, lm_head_weight, final_logit_softcapping=None)

    result = _fs_gemma4_unified_lce_forward(self_obj, labels=labels)

    assert result.logits is None  # the skip-logits convention: nothing downstream reads it
    assert result.loss is not None
    assert result.past_key_values == "pkv"

    # return_dict=False: a plain tuple, not the dataclass, with loss first.
    tuple_result = _fs_gemma4_unified_lce_forward(self_obj, labels=labels, return_dict=False)
    assert isinstance(tuple_result, tuple)
    assert torch.equal(tuple_result[0], result.loss)


def test_gemma4_unified_lce_forward_without_labels_returns_plain_logits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import torch

    _install_fake_liger_loss_utils(monkeypatch, _pure_torch_liger_loss)
    hidden_states = torch.randn(1, 3, 4)
    lm_head_weight = torch.randn(5, 4)
    self_obj = _fake_self(hidden_states, lm_head_weight, final_logit_softcapping=None)

    result = _fs_gemma4_unified_lce_forward(self_obj, labels=None)

    assert result.loss is None
    assert result.logits is not None
    assert result.logits.shape == (1, 3, 5)


def test_gemma4_unified_lce_forward_without_labels_applies_softcapping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import torch

    _install_fake_liger_loss_utils(monkeypatch, _pure_torch_liger_loss)
    hidden_states = torch.randn(1, 3, 4) * 10  # large enough for softcapping to bite
    lm_head_weight = torch.randn(5, 4) * 10
    self_obj = _fake_self(hidden_states, lm_head_weight, final_logit_softcapping=5.0)

    result = _fs_gemma4_unified_lce_forward(self_obj, labels=None)

    assert result.loss is None
    assert result.logits is not None
    # tanh(x / cap) * cap is bounded in [-cap, cap] -- the softcapping contract.
    # (tanh saturates to exactly +/-1.0 in float32 well before its argument
    # reaches infinity, so some entries legitimately equal the cap exactly.)
    assert bool((result.logits.abs() <= 5.0 + 1e-5).all())
    assert bool((result.logits.abs() > 4.9).any())  # genuinely large, not a no-op
