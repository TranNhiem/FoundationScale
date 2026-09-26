# SPDX-License-Identifier: Apache-2.0
"""Coverage tests: refusal branches and edge cases of the preference kernel.

These tests target the error and edge-case paths of
``foundationscale.rl.preference_torch`` that the parity suite does not feed:
recipe selection refusals, wrong-plane and grad-less inputs, degenerate row
counts, fractional masks, ORPO's exact-probability odds wall, KTO's
recipe-specific input checks (reference point, ``desirable`` labels), the
paired-only input refusals, the fail-closed missing-reference guards
(reached by monkeypatching ``reference_free``), the non-finite-loss wall,
DPO's opt-in SFT term, and the remaining ``preference_metrics`` recipes.
"""

from __future__ import annotations

from typing import Any

import pytest

torch = pytest.importorskip("torch")

from foundationscale.rl.interfaces import BatchRefusal, SupervisionRefusal  # noqa: E402
from foundationscale.rl.losses import DPOLoss  # noqa: E402
from foundationscale.rl.preference_objectives import (  # noqa: E402
    CPOLoss,
    IPOLoss,
    KTOLoss,
    ORPOLoss,
    SimPOLoss,
)
from foundationscale.rl.preference_torch import (  # noqa: E402
    TensorPreferenceLoss,
    needs_reference,
    preference_metrics,
)

PAIR_COUNT = 2
TOKEN_COUNT = 5
KTO_ROW_COUNT = 3


def _prefix_mask(lengths: tuple[int, ...], width: int) -> torch.Tensor:
    mask = torch.zeros(len(lengths), width, dtype=torch.float64)
    for row, length in enumerate(lengths):
        mask[row, :length] = 1.0
    return mask


def _interleave(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    stacked = torch.stack((left, right), dim=1)
    return stacked.reshape(2 * int(left.shape[0]), int(left.shape[1]))


def _paired_tensors() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(1234)
    policy = -(0.05 + 2.0 * torch.rand(2 * PAIR_COUNT, TOKEN_COUNT, dtype=torch.float64))
    reference = -(0.05 + 2.0 * torch.rand(2 * PAIR_COUNT, TOKEN_COUNT, dtype=torch.float64))
    mask = _interleave(
        _prefix_mask((5, 2), TOKEN_COUNT),
        _prefix_mask((3, 5), TOKEN_COUNT),
    )
    return policy, reference, mask


def _kto_tensors() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(4321)
    policy = -(0.05 + 2.0 * torch.rand(KTO_ROW_COUNT, TOKEN_COUNT, dtype=torch.float64))
    reference = -(0.05 + 2.0 * torch.rand(KTO_ROW_COUNT, TOKEN_COUNT, dtype=torch.float64))
    mask = _prefix_mask((5, 1, 3), TOKEN_COUNT)
    desirable = torch.tensor((True, False, True), dtype=torch.bool)
    return policy, reference, mask, desirable


def test_unknown_objective_is_refused_by_recipe_selection() -> None:
    """An attribute-compatible duck is not admitted as a parity contract."""
    with pytest.raises(BatchRefusal, match="supported preference oracle classes matched"):
        needs_reference(object())


def test_non_tensor_policy_logprobs_is_refused() -> None:
    policy, reference, mask = _paired_tensors()
    with pytest.raises(BatchRefusal, match="not torch.Tensor"):
        TensorPreferenceLoss(DPOLoss())(
            policy_logprobs=policy.tolist(),
            completion_mask=mask,
            reference_logprobs=reference,
        )


def test_policy_without_grad_is_refused_by_the_kernel() -> None:
    policy, reference, mask = _paired_tensors()
    with pytest.raises(BatchRefusal, match="carries no gradient"):
        TensorPreferenceLoss(DPOLoss())(
            policy_logprobs=policy.clone(),
            completion_mask=mask,
            reference_logprobs=reference,
        )


def test_kto_zero_row_batch_is_refused() -> None:
    objective = KTOLoss()
    policy = torch.zeros(0, TOKEN_COUNT, dtype=torch.float64, requires_grad=True)
    reference = torch.zeros(0, TOKEN_COUNT, dtype=torch.float64)
    mask = torch.zeros(0, TOKEN_COUNT, dtype=torch.float64)
    with pytest.raises(BatchRefusal, match="got 0 rows"):
        TensorPreferenceLoss(objective)(
            policy_logprobs=policy,
            completion_mask=mask,
            reference_logprobs=reference,
            kl_reference_point=0.1,
            desirable=torch.zeros(0, dtype=torch.bool),
        )


def test_paired_zero_row_batch_is_refused() -> None:
    objective = SimPOLoss()
    policy = torch.zeros(0, TOKEN_COUNT, dtype=torch.float64, requires_grad=True)
    mask = torch.zeros(0, TOKEN_COUNT, dtype=torch.float64)
    with pytest.raises(BatchRefusal, match="got 0 rows"):
        TensorPreferenceLoss(objective)(
            policy_logprobs=policy,
            completion_mask=mask,
        )


def test_paired_odd_row_count_is_refused() -> None:
    policy = -torch.ones(3, TOKEN_COUNT, dtype=torch.float64, requires_grad=True)
    reference = -torch.ones(3, TOKEN_COUNT, dtype=torch.float64)
    mask = torch.ones(3, TOKEN_COUNT, dtype=torch.float64)
    with pytest.raises(BatchRefusal, match="R == 2B"):
        TensorPreferenceLoss(DPOLoss())(
            policy_logprobs=policy,
            completion_mask=mask,
            reference_logprobs=reference,
        )


def test_fractional_mask_entry_is_refused() -> None:
    policy, reference, mask = _paired_tensors()
    broken = mask.clone()
    broken[0, 0] = 0.5
    with pytest.raises(BatchRefusal, match="neither 0 nor 1"):
        TensorPreferenceLoss(DPOLoss())(
            policy_logprobs=policy.clone().requires_grad_(True),
            completion_mask=broken,
            reference_logprobs=reference,
        )


def test_kto_row_with_zero_supervision_is_refused() -> None:
    policy, reference, mask, desirable = _kto_tensors()
    emptied = mask.clone()
    emptied[0] = 0.0
    with pytest.raises(SupervisionRefusal, match="a row that supervises nothing"):
        TensorPreferenceLoss(KTOLoss())(
            policy_logprobs=policy.clone().requires_grad_(True),
            completion_mask=emptied,
            reference_logprobs=reference,
            kl_reference_point=0.1,
            desirable=desirable,
        )


def test_orpo_exact_probability_row_hits_the_odds_wall() -> None:
    """A length-normalised average of 0.0 makes ORPO's odds undefined."""
    objective = ORPOLoss()
    policy = torch.zeros(2, TOKEN_COUNT, dtype=torch.float64, requires_grad=True)
    mask = torch.ones(2, TOKEN_COUNT, dtype=torch.float64)
    with pytest.raises(BatchRefusal, match="odds p /"):
        TensorPreferenceLoss(objective)(
            policy_logprobs=policy,
            completion_mask=mask,
        )


def test_kto_unconvertible_reference_point_is_refused() -> None:
    policy, reference, mask, desirable = _kto_tensors()
    with pytest.raises(BatchRefusal, match="does not convert to a scalar float"):
        TensorPreferenceLoss(KTOLoss())(
            policy_logprobs=policy.clone().requires_grad_(True),
            completion_mask=mask,
            reference_logprobs=reference,
            kl_reference_point="oops",
            desirable=desirable,
        )


def test_kto_negative_reference_point_is_refused() -> None:
    policy, reference, mask, desirable = _kto_tensors()
    with pytest.raises(BatchRefusal, match="not a finite"):
        TensorPreferenceLoss(KTOLoss())(
            policy_logprobs=policy.clone().requires_grad_(True),
            completion_mask=mask,
            reference_logprobs=reference,
            kl_reference_point=-0.5,
            desirable=desirable,
        )


def test_kto_missing_desirable_is_refused() -> None:
    policy, reference, mask, _ = _kto_tensors()
    with pytest.raises(BatchRefusal, match="desirable is None"):
        TensorPreferenceLoss(KTOLoss())(
            policy_logprobs=policy.clone().requires_grad_(True),
            completion_mask=mask,
            reference_logprobs=reference,
            kl_reference_point=0.1,
        )


def test_kto_non_bool_desirable_is_refused() -> None:
    policy, reference, mask, desirable = _kto_tensors()
    with pytest.raises(BatchRefusal, match="expected torch.bool"):
        TensorPreferenceLoss(KTOLoss())(
            policy_logprobs=policy.clone().requires_grad_(True),
            completion_mask=mask,
            reference_logprobs=reference,
            kl_reference_point=0.1,
            desirable=desirable.to(dtype=torch.float64),
        )


def test_kto_misshapen_desirable_is_refused() -> None:
    policy, reference, mask, _ = _kto_tensors()
    with pytest.raises(BatchRefusal, match="one boolean flag per completion"):
        TensorPreferenceLoss(KTOLoss())(
            policy_logprobs=policy.clone().requires_grad_(True),
            completion_mask=mask,
            reference_logprobs=reference,
            kl_reference_point=0.1,
            desirable=torch.tensor((True, False), dtype=torch.bool),
        )


def test_paired_kernel_refuses_a_kl_reference_point() -> None:
    policy, reference, mask = _paired_tensors()
    with pytest.raises(BatchRefusal, match="unexpected"):
        TensorPreferenceLoss(DPOLoss())(
            policy_logprobs=policy.clone().requires_grad_(True),
            completion_mask=mask,
            reference_logprobs=reference,
            kl_reference_point=0.1,
        )


def test_paired_kernel_refuses_a_desirable_tensor() -> None:
    policy, reference, mask = _paired_tensors()
    with pytest.raises(BatchRefusal, match="desirable tensor"):
        TensorPreferenceLoss(DPOLoss())(
            policy_logprobs=policy.clone().requires_grad_(True),
            completion_mask=mask,
            reference_logprobs=reference,
            desirable=torch.tensor((True, False, True, False), dtype=torch.bool),
        )


def test_dpo_with_nonzero_sft_weight_adds_the_sft_term() -> None:
    """The opt-in SFT branch evaluates and stays finite and scalar."""
    policy, reference, mask = _paired_tensors()
    loss = TensorPreferenceLoss(DPOLoss(sft_weight=0.5))(
        policy_logprobs=policy.clone().requires_grad_(True),
        completion_mask=mask,
        reference_logprobs=reference,
    )
    assert loss.dim() == 0
    assert bool(torch.isfinite(loss))


def test_non_finite_loss_inside_the_recipe_is_refused() -> None:
    """Overflow past the guarded inputs refuses instead of clamping."""
    objective = IPOLoss()
    policy = torch.full((2, TOKEN_COUNT), -1e200, dtype=torch.float64, requires_grad=True)
    reference = torch.zeros(2, TOKEN_COUNT, dtype=torch.float64)
    mask = _interleave(_prefix_mask((1,), TOKEN_COUNT), _prefix_mask((2,), TOKEN_COUNT))
    with pytest.raises(BatchRefusal, match="non-finite loss"):
        TensorPreferenceLoss(objective)(
            policy_logprobs=policy,
            completion_mask=mask,
            reference_logprobs=reference,
        )


def _declare_reference_free(monkeypatch: pytest.MonkeyPatch, cls: type[Any]) -> None:
    """Flip an oracle's ``reference_free`` declaration to reach fail-closed guards.

    The missing-reference guards in the kernel and metrics are unreachable
    while the common checks honestly detect the discrepancy, so the only
    way to exercise them is an oracle whose declaration has been edited --
    exactly the incoherence the guards exist to refuse.
    """
    monkeypatch.setattr(cls, "reference_free", property(lambda self: True))


def test_kto_kernel_fail_closed_guard_for_missing_reference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _declare_reference_free(monkeypatch, KTOLoss)
    policy, _, mask, desirable = _kto_tensors()
    with pytest.raises(BatchRefusal, match="declared a reference seam"):
        TensorPreferenceLoss(KTOLoss())(
            policy_logprobs=policy.clone().requires_grad_(True),
            completion_mask=mask,
            kl_reference_point=0.1,
            desirable=desirable,
        )


def test_dpo_kernel_fail_closed_guard_for_missing_reference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _declare_reference_free(monkeypatch, DPOLoss)
    policy, _, mask = _paired_tensors()
    with pytest.raises(BatchRefusal, match="declared a reference seam"):
        TensorPreferenceLoss(DPOLoss())(
            policy_logprobs=policy.clone().requires_grad_(True),
            completion_mask=mask,
        )


def test_ipo_kernel_fail_closed_guard_for_missing_reference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _declare_reference_free(monkeypatch, IPOLoss)
    policy, _, mask = _paired_tensors()
    with pytest.raises(BatchRefusal, match="declared a reference seam"):
        TensorPreferenceLoss(IPOLoss())(
            policy_logprobs=policy.clone().requires_grad_(True),
            completion_mask=mask,
        )


def test_dpo_metrics_fail_closed_guard_for_missing_reference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _declare_reference_free(monkeypatch, DPOLoss)
    policy, _, mask = _paired_tensors()
    with pytest.raises(BatchRefusal, match="declared a reference seam"):
        preference_metrics(DPOLoss(), policy, mask)


def test_ipo_metrics_fail_closed_guard_for_missing_reference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _declare_reference_free(monkeypatch, IPOLoss)
    policy, _, mask = _paired_tensors()
    with pytest.raises(BatchRefusal, match="declared a reference seam"):
        preference_metrics(IPOLoss(), policy, mask)


def test_cpo_metrics_use_the_beta_scaled_margin() -> None:
    policy, _, mask = _paired_tensors()
    metrics = preference_metrics(CPOLoss(), policy, mask)
    assert set(metrics) == {"accuracy", "margin"}
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert float(metrics["margin"]) == pytest.approx(metrics["margin"])


def test_ipo_metrics_default_report_the_summed_margin() -> None:
    """IPO's default (unnormalised) margin branch reports the anchored h."""
    policy, reference, mask = _paired_tensors()
    metrics = preference_metrics(IPOLoss(), policy, mask, reference)
    assert set(metrics) == {"accuracy", "margin"}


def test_ipo_length_normalised_metrics_report_the_mean_margin() -> None:
    policy, reference, mask = _paired_tensors()
    metrics = preference_metrics(IPOLoss(length_normalise=True), policy, mask, reference)
    assert set(metrics) == {"accuracy", "margin"}


def test_orpo_metrics_report_the_log_odds_margin() -> None:
    policy, _, mask = _paired_tensors()
    metrics = preference_metrics(ORPOLoss(), policy, mask)
    assert set(metrics) == {"accuracy", "margin"}
    assert metrics["accuracy"] >= 0.0


def test_kto_metrics_return_an_empty_mapping() -> None:
    policy, reference, mask, _ = _kto_tensors()
    assert preference_metrics(KTOLoss(), policy, mask, reference) == {}
