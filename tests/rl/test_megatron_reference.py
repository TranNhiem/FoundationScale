"""CPU tests for the frozen-reference helpers in the megatron RL driver.

Importing ``foundationscale.rl.megatron.driver`` pulls no megatron (every
megatron import is lazy), so these run on a CPU box with torch installed.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from foundationscale.rl.megatron.driver import (
    needs_reference,
    reference_gap,
    swapped_parameters,
)


def _policy_params() -> list[torch.nn.Parameter]:
    """Parameters of two nn.Linear layers, in named_parameters order."""
    layers = (torch.nn.Linear(3, 2), torch.nn.Linear(2, 1))
    return [p for layer in layers for p in layer.parameters()]


def _bits(t: torch.Tensor) -> bytes:
    """Host bytes of ``t``'s content: a bit-exact equality marker."""
    return t.detach().cpu().contiguous().numpy().tobytes()


def test_swap_installs_replacements_and_restores_bit_exact() -> None:
    params = _policy_params()
    originals = [p.detach().clone() for p in params]
    replacements = [torch.randn_like(p) for p in params]
    with swapped_parameters(params, replacements):
        for p, r in zip(params, replacements, strict=True):
            assert torch.equal(p, r)
    for p, o in zip(params, originals, strict=True):
        assert _bits(p) == _bits(o)


def test_swap_restores_when_body_raises() -> None:
    params = _policy_params()
    originals = [p.detach().clone() for p in params]
    replacements = [torch.randn_like(p) for p in params]
    with (
        pytest.raises(RuntimeError, match="body failed"),
        swapped_parameters(params, replacements),
    ):
        raise RuntimeError("body failed")
    for p, o in zip(params, originals, strict=True):
        assert _bits(p) == _bits(o)


def test_length_mismatch_refuses_and_leaves_params_untouched() -> None:
    params = _policy_params()  # 4 tensors: 2 Linear layers
    originals = [p.detach().clone() for p in params]
    with (
        # The counts in the refusal are its operator-facing contract, so they are pinned.
        pytest.raises(ValueError, match="4 parameters but 1 replacements"),
        swapped_parameters(params, [torch.rand(5, 5)]),
    ):
        pass
    for p, o in zip(params, originals, strict=True):
        assert _bits(p) == _bits(o)


def test_shape_mismatch_refuses_before_touching_any_parameter() -> None:
    params = _policy_params()
    originals = [p.detach().clone() for p in params]
    # Valid replacements everywhere except index 2 (Linear(2, 1).weight is
    # (1, 2)); a swap of index 0 would show validation was not up-front.
    replacements = [torch.randn_like(p) for p in params]
    replacements[2] = torch.randn(2, 1)
    with (
        # The index and both shapes are what the operator needs; pin exactly those.
        pytest.raises(
            ValueError,
            match=r"parameter 2 has shape \(1, 2\) but its replacement has shape \(2, 1\)",
        ),
        swapped_parameters(params, replacements),
    ):
        pass
    for p, o in zip(params, originals, strict=True):
        assert _bits(p) == _bits(o)


def test_needs_reference() -> None:
    assert needs_reference(SimpleNamespace(kl_weight=0.0)) is False
    assert needs_reference(SimpleNamespace(kl_weight=0.04)) is True
    assert needs_reference(SimpleNamespace()) is False  # objective without a KL term
    assert needs_reference(SimpleNamespace(kl_weight=None)) is False


def test_reference_gap_identical_is_exactly_zero() -> None:
    logprobs = torch.tensor([[1.5, -2.0, 0.25]])
    mask = torch.ones_like(logprobs)
    assert reference_gap(logprobs, logprobs.clone(), mask) == 0.0


def test_reference_gap_known_mean_over_mask() -> None:
    # diffs are [100.0, 0.25, 0.5]; the 100.0 sits under a masked-out position.
    old = torch.tensor([[100.0, 1.0, 3.0]])
    ref = torch.tensor([[0.0, 0.75, 2.5]])
    mask = torch.tensor([[0.0, 1.0, 1.0]])
    # 0.25 and 0.5 are exact binary fractions, so the float64 mean is exactly 0.375.
    assert reference_gap(old, ref, mask) == 0.375


def test_reference_gap_empty_mask_abstains() -> None:
    # Nothing supervised means nothing measured: None, never the passing 0.0.
    old = torch.tensor([[1.0, 2.0]])
    ref = torch.tensor([[0.0, 5.0]])
    assert reference_gap(old, ref, torch.zeros(1, 2)) is None
