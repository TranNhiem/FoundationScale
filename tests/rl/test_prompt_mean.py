"""The ``prompt_mean`` reduction: torch-free oracle, kernel mirror, refusals.

``prompt_mean`` is the first reduction in this plane whose denominator is
PER GROUP and therefore depends on group ids, which ``TensorPolicyLoss``
never receives. So the denominator shape has exactly one owner --
``prompt_mean_row_weights`` -- and the tensor kernel takes its output
verbatim as ``reduction_row_weights``. On any disagreement between the two
planes the TENSOR side is what is wrong, never the oracle.

WHAT IS CLAIMED: the row-weight function matches a hand computation over
unequal groups and a zero-token row, and refuses every malformed input; the
tensor scalar agrees with the oracle's ``LossOutput.loss`` to rtol 1e-6 over
random batches on several seeds (with and without an abstained reward); the
two reductions are distinguishable when group token totals differ and
coincide when they do not; and every kernel-side refusal for the new plane
has a failing mutant.

WHAT IS NOT CLAIMED: any convergence, benchmark or paper-equivalence
property, and any guarantee beyond the batches measured.
"""

from __future__ import annotations

import random
from typing import Any

import pytest

torch = pytest.importorskip("torch")

from foundationscale.rl.advantage import AdvantageRefusal, SessionGroupAdvantage  # noqa: E402
from foundationscale.rl.group_policy_objectives import (  # noqa: E402
    AgenticGRPOLoss,
    prompt_mean_row_weights,
)
from foundationscale.rl.interfaces import (  # noqa: E402
    BatchRefusal,
    ExperienceBatch,
    LossConfigRefusal,
    SupervisionRefusal,
)
from foundationscale.rl.torch_backend import TensorPolicyLoss  # noqa: E402

_ROWS = 4
_TOKENS = 6


class _Reduction:
    """Minimal objective stub: the kernel reads exactly these four axes."""

    def __init__(self, reduction: str) -> None:
        self.ratio_scope = "token"
        self.reduction = reduction
        self.clip_bounds = (0.8, 1.2)
        self.weight = 1.0


# ---------------------------------------------------------------------------
# prompt_mean_row_weights: the one owner of the denominator shape
# ---------------------------------------------------------------------------


def test_row_weights_hand_computed_over_unequal_groups_and_a_zero_token_row() -> None:
    """Hand computation: active groups at 4 and 6 supervised tokens.

    rows 0,1 are group "a" (3 + 1 = 4 tokens) and rows 2,3 are group "b"
    (5 + 1 = 6 tokens); row 4 is group "c" with 0 supervised tokens and is
    excluded from P. P == 2, so the weights are 1/(2 * 4) for rows 0 and 1,
    1/(2 * 6) for rows 2 and 3, and a literal 0.0 for the zero-token row.
    """
    weights = prompt_mean_row_weights(("a", "a", "b", "b", "c"), (3, 1, 5, 1, 0))
    assert weights == pytest.approx((0.125, 0.125, 1.0 / 12.0, 1.0 / 12.0, 0.0))


def test_weighted_row_sum_is_the_declared_prompt_mean() -> None:
    # Derivation, not a restatement: for ANY per-row term the weighted sum
    # must equal (1/P) * sum_g (terms_g / tokens_g). This is the identity the
    # two planes are held to, stated independently of both.
    group_ids = ("a", "a", "b", "b")
    counts = (2, 1, 3, 1)
    row_terms = (1.0, 2.0, 3.0, 4.0)
    weights = prompt_mean_row_weights(group_ids, counts)
    surrogate = sum(w * t for w, t in zip(weights, row_terms, strict=True))
    expected = 0.5 * ((1.0 + 2.0) / 3.0 + (3.0 + 4.0) / 4.0)
    assert surrogate == pytest.approx(expected)


def test_row_weights_refuse_a_length_mismatch_naming_both_counts() -> None:
    with pytest.raises(BatchRefusal) as exc_info:
        prompt_mean_row_weights(("a", "b"), (1,))
    message = str(exc_info.value)
    assert "2 group ids" in message
    assert "1 supervised-token counts" in message


def test_row_weights_refuse_a_bool_or_non_integer_count() -> None:
    for bad in (True, 1.5, "2", None):
        with pytest.raises(BatchRefusal, match="must be an int of at least 0"):
            prompt_mean_row_weights(("a",), (bad,))  # type: ignore[arg-type]


def test_row_weights_refuse_a_negative_count() -> None:
    with pytest.raises(BatchRefusal, match="cannot be negative"):
        prompt_mean_row_weights(("a", "a"), (3, -1))


def test_row_weights_refuse_an_unhashable_group_id() -> None:
    with pytest.raises(BatchRefusal, match="not hashable"):
        prompt_mean_row_weights(([1], [1]), (1, 1))  # type: ignore[arg-type]


def test_row_weights_refuse_zero_active_groups() -> None:
    with pytest.raises(SupervisionRefusal, match="never 0.0"):
        prompt_mean_row_weights(("a", "a"), (0, 0))


def test_row_weights_refuse_an_empty_input() -> None:
    with pytest.raises(SupervisionRefusal, match="0 of 0 prompt group"):
        prompt_mean_row_weights((), ())


def test_row_weights_refuse_a_non_iterable_pair() -> None:
    with pytest.raises(BatchRefusal, match="was not iterable"):
        prompt_mean_row_weights(("a",), 7)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# The oracle itself: one hand-computed loss
# ---------------------------------------------------------------------------


def _experience(columns: dict[str, Any]) -> ExperienceBatch:
    return ExperienceBatch(columns=columns, required=tuple(columns))


def test_agentic_grpo_prompt_mean_loss_matches_the_hand_computation() -> None:
    """Hand computation over two groups of two sessions.

    Group "a" rewards (0.0, 2.0) -> centred weights -1, +1; group "b"
    rewards (0.0, 6.0) -> -3, +3. current == old everywhere, so every ratio
    is exactly 1.0 and every clipped term equals its advantage weight.

        mask         (1, 0)  (1, 1)  (1, 1)  (1, 1)
        supervised       1       2       2       2
        row weight     1/6     1/6     1/8     1/8
        row terms       -1      +2      -6      +6
        contributes   -1/6    +1/3    -3/4    +3/4

    P == 2 active groups ("a": 3 supervised tokens, "b": 4), so the
    surrogate is 1/6 and the loss is -1/6.
    """
    objective = AgenticGRPOLoss(group_size=2)
    batch = _experience(
        {
            "prompt_ids": ("a", "a", "b", "b"),
            "rewards": (0.0, 2.0, 0.0, 6.0),
            "loss_mask": ((1, 0), (1, 1), (1, 1), (1, 1)),
            "old_logprobs": ((-1.0, -1.0), (-1.0, -1.0), (-1.0, -1.0), (-1.0, -1.0)),
        }
    )
    output, advantage = objective.compute_with_report(
        lambda _batch: ((-1.0, -1.0), (-1.0, -1.0), (-1.0, -1.0), (-1.0, -1.0)),
        batch,
    )
    assert advantage.rows == (0, 1, 2, 3)
    assert output.loss == pytest.approx(-1.0 / 6.0)
    (component,) = output.components
    assert component.name == "agentic_grpo_policy_loss"
    assert component.contribution == pytest.approx(-1.0 / 6.0)


# ---------------------------------------------------------------------------
# Oracle vs kernel parity
# ---------------------------------------------------------------------------


def _oracle_and_tensor(
    seed: int, *, rewards: tuple[Any, ...] | None = None
) -> tuple[float, torch.Tensor]:
    rng = random.Random(seed)
    prompt_ids = ("g0", "g0", "g1", "g1")
    if rewards is None:
        rewards = tuple(float(rng.uniform(0.5, 3.0)) for _ in range(_ROWS))
    current = tuple(tuple(rng.uniform(-3.0, -0.5) for _ in range(_TOKENS)) for _ in range(_ROWS))
    old = tuple(tuple(value + rng.uniform(-0.30, 0.30) for value in row) for row in current)
    mask = (
        (1, 1, 1, 0, 0, 0),
        (1, 1, 0, 0, 0, 0),
        (1, 1, 1, 1, 0, 0),
        (1, 0, 0, 0, 0, 0),
    )
    columns: dict[str, Any] = {
        "prompt_ids": prompt_ids,
        "rewards": rewards,
        "loss_mask": mask,
        "old_logprobs": old,
    }
    batch = _experience(columns)
    objective = AgenticGRPOLoss(
        group_size=2,
        advantage_fn=SessionGroupAdvantage(min_group_size=2),
    )

    def forward_fn(_batch: ExperienceBatch) -> Any:
        return current

    oracle_output, advantage = objective.compute_with_report(forward_fn, batch)
    used = advantage.rows
    kept_group_ids = [prompt_ids[row] for row in used]
    kept_counts = [sum(1 for entry in mask[row] if entry) for row in used]
    row_weights = prompt_mean_row_weights(kept_group_ids, kept_counts)
    tensor_loss = TensorPolicyLoss(objective)(
        current_logprobs=torch.tensor(
            [current[row] for row in used], dtype=torch.float64, requires_grad=True
        ),
        old_logprobs=torch.tensor([old[row] for row in used], dtype=torch.float64),
        advantages=torch.tensor(
            [list(weights) for weights in advantage.weights], dtype=torch.float64
        ),
        mask=torch.tensor([mask[row] for row in used], dtype=torch.float64),
        reduction_row_weights=torch.tensor(row_weights, dtype=torch.float64),
    )
    return oracle_output.loss, tensor_loss


@pytest.mark.parametrize("seed", (0, 1, 2, 7, 11))
def test_tensor_loss_matches_the_oracle_across_seeds(seed: int) -> None:
    oracle_loss, tensor_loss = _oracle_and_tensor(seed)
    assert float(tensor_loss) == pytest.approx(oracle_loss, rel=1e-6, abs=1e-9)


def test_tensor_loss_matches_the_oracle_with_an_abstained_reward() -> None:
    # g0 partially abstains and shrinks below its minimum, so the compaction
    # path and a single-active-group denominator both ride through the
    # parity: P shrinks to 1 and both planes say so identically.
    oracle_loss, tensor_loss = _oracle_and_tensor(3, rewards=(1.0, None, 2.0, 6.0))
    assert float(tensor_loss) == pytest.approx(oracle_loss, rel=1e-6, abs=1e-9)


# ---------------------------------------------------------------------------
# Distinguishability: prompt_mean is not token_mean
# ---------------------------------------------------------------------------


def _two_reduction_losses(mask: Any, advantages: Any) -> tuple[float, float]:
    rows = len(mask)
    columns = len(mask[0])
    current = torch.zeros((rows, columns), dtype=torch.float64, requires_grad=True)
    old = torch.zeros((rows, columns), dtype=torch.float64)
    adv = torch.tensor(advantages, dtype=torch.float64)
    mask_f = torch.tensor(mask, dtype=torch.float64)
    counts = [sum(1 for entry in row if entry) for row in mask]
    weights = torch.tensor(
        prompt_mean_row_weights(("a", "a", "b", "b"), counts), dtype=torch.float64
    )
    prompt = float(
        TensorPolicyLoss(_Reduction("prompt_mean"))(
            current_logprobs=current,
            old_logprobs=old,
            advantages=adv,
            mask=mask_f,
            reduction_row_weights=weights,
        )
    )
    token = float(
        TensorPolicyLoss(_Reduction("token_mean"))(
            current_logprobs=current,
            old_logprobs=old,
            advantages=adv,
            mask=mask_f,
        )
    )
    return prompt, token


def test_prompt_mean_differs_from_token_mean_when_group_totals_differ() -> None:
    # group "a" = rows 0,1 at 3 supervised tokens and weight 1; group "b"
    # = rows 2,3 at 5 supervised tokens and weight 3. token_mean reads
    # (3 + 15) / 8; prompt_mean reads (1/2) * (3/3 + 15/5).
    mask = ((1, 1, 0, 0), (1, 0, 0, 0), (1, 1, 1, 0), (1, 1, 0, 0))
    advantages = ((1, 1, 1, 1), (1, 1, 1, 1), (3, 3, 3, 3), (3, 3, 3, 3))
    prompt, token = _two_reduction_losses(mask, advantages)
    assert token == pytest.approx(-18.0 / 8.0)
    assert prompt == pytest.approx(-2.0)
    assert prompt != pytest.approx(token)


def test_prompt_mean_agrees_with_token_mean_when_group_totals_match() -> None:
    # The control that keeps the previous test honest: with equal per-group
    # token totals the two denominators coincide, so a kernel guessing the
    # wrong one is only caught by the differing case above.
    mask = ((1, 1, 0, 0), (1, 0, 0, 0), (1, 0, 0, 0), (1, 1, 0, 0))
    advantages = ((1, 1, 1, 1), (1, 1, 1, 1), (3, 3, 3, 3), (3, 3, 3, 3))
    prompt, token = _two_reduction_losses(mask, advantages)
    assert prompt == pytest.approx(token)


# ---------------------------------------------------------------------------
# Kernel-side refusals for the new plane
# ---------------------------------------------------------------------------


def test_prompt_mean_without_row_weights_refuses_naming_the_missing_plane() -> None:
    with pytest.raises(BatchRefusal, match="required reduction_row_weights"):
        TensorPolicyLoss(_Reduction("prompt_mean"))(
            current_logprobs=torch.zeros((2, 3), requires_grad=True),
            old_logprobs=torch.zeros((2, 3)),
            advantages=torch.ones(2),
            mask=torch.ones((2, 3)),
        )


def test_prompt_mean_row_weights_length_mismatch_refuses_naming_both_counts() -> None:
    with pytest.raises(BatchRefusal) as exc_info:
        TensorPolicyLoss(_Reduction("prompt_mean"))(
            current_logprobs=torch.zeros((3, 4), requires_grad=True),
            old_logprobs=torch.zeros((3, 4)),
            advantages=torch.ones(3),
            mask=torch.ones((3, 4)),
            reduction_row_weights=torch.ones(2),
        )
    message = str(exc_info.value)
    assert "reduction_row_weights has 2 entries" in message
    assert "current_logprobs has 3 rows" in message


def test_row_weights_with_another_reduction_refuse_naming_the_owner() -> None:
    for reduction in ("token_mean", "sequence_mean", "constant"):
        with pytest.raises(BatchRefusal, match="one owner"):
            TensorPolicyLoss(_Reduction(reduction))(
                current_logprobs=torch.zeros((2, 3), requires_grad=True),
                old_logprobs=torch.zeros((2, 3)),
                advantages=torch.ones(2),
                mask=torch.ones((2, 3)),
                reduction_row_weights=torch.ones(2),
            )


def test_row_weights_with_shape_rows_by_one_refuse() -> None:
    with pytest.raises(BatchRefusal, match="2 dims"):
        TensorPolicyLoss(_Reduction("prompt_mean"))(
            current_logprobs=torch.zeros((2, 3), requires_grad=True),
            old_logprobs=torch.zeros((2, 3)),
            advantages=torch.ones(2),
            mask=torch.ones((2, 3)),
            reduction_row_weights=torch.ones((2, 1)),
        )


def test_row_weights_are_load_bearing() -> None:
    # A kernel that ignored the plane -- ``surrogate = surrogate_terms.sum()``
    # -- passes every equivalence leg above, because each supplies the SAME
    # weights to both planes. Perturbing exactly one entry must move the
    # scalar, and it moves it by exactly the perturbed row's term sum.
    mask = ((1, 1), (1, 1))
    advantages = ((2.0, 2.0), (5.0, 5.0))
    current = torch.zeros((2, 2), dtype=torch.float64, requires_grad=True)
    old = torch.zeros((2, 2), dtype=torch.float64)
    adv = torch.tensor(advantages, dtype=torch.float64)
    mask_f = torch.tensor(mask, dtype=torch.float64)
    base = torch.tensor((0.25, 0.25), dtype=torch.float64)
    moved = torch.tensor((0.25, 0.35), dtype=torch.float64)

    def price(weights: torch.Tensor) -> float:
        return float(
            TensorPolicyLoss(_Reduction("prompt_mean"))(
                current_logprobs=current,
                old_logprobs=old,
                advantages=adv,
                mask=mask_f,
                reduction_row_weights=weights,
            )
        )

    first = price(base)
    second = price(moved)
    # row 1 contributes 10.0 of masked terms; +0.10 of weight moves it by
    # 1.0 in the surrogate and by 1.0 (with the sign flip) in the loss.
    assert second == pytest.approx(first - 1.0)


def test_prompt_mean_row_weights_that_are_not_a_tensor_refuse_naming_the_type() -> None:
    with pytest.raises(BatchRefusal, match="is a list, not a torch.Tensor"):
        TensorPolicyLoss(_Reduction("prompt_mean"))(
            current_logprobs=torch.zeros((2, 3), requires_grad=True),
            old_logprobs=torch.zeros((2, 3)),
            advantages=torch.ones(2),
            mask=torch.ones((2, 3)),
            reduction_row_weights=[0.5, 0.5],
        )


# ---------------------------------------------------------------------------
# The token_mean ablation arm: same objective, one batch-wide denominator
# ---------------------------------------------------------------------------


def _hand_batch() -> ExperienceBatch:
    return _experience(
        {
            "prompt_ids": ("a", "a", "b", "b"),
            "rewards": (0.0, 2.0, 0.0, 6.0),
            "loss_mask": ((1, 0), (1, 1), (1, 1), (1, 1)),
            "old_logprobs": ((-1.0, -1.0), (-1.0, -1.0), (-1.0, -1.0), (-1.0, -1.0)),
        }
    )


def test_agentic_grpo_token_mean_loss_matches_the_hand_computation() -> None:
    """The prompt_mean oracle's batch, re-priced with ONE denominator.

    Row terms are -1, +2, -6, +6 (sum 1) over 1 + 2 + 2 + 2 = 7 supervised
    tokens, so the surrogate is 1/7 and the loss -1/7 -- distinct from the
    prompt_mean arm's -1/6 on the same batch, which is what makes the A/B
    measure the reduction.
    """
    objective = AgenticGRPOLoss(group_size=2, reduction_mode="token_mean")
    assert objective.reduction == "token_mean"
    output, _advantage = objective.compute_with_report(
        lambda _batch: ((-1.0, -1.0), (-1.0, -1.0), (-1.0, -1.0), (-1.0, -1.0)),
        _hand_batch(),
    )
    assert output.loss == pytest.approx(-1.0 / 7.0)


def test_agentic_grpo_default_reduction_stays_prompt_mean() -> None:
    assert AgenticGRPOLoss(group_size=2).reduction == "prompt_mean"


@pytest.mark.parametrize("mode", ["sequence_mean", "TOKEN_MEAN", "", None, 1])
def test_agentic_grpo_refuses_an_undeclared_reduction_mode(mode: Any) -> None:
    with pytest.raises(LossConfigRefusal, match="1 of 2 reductions"):
        AgenticGRPOLoss(group_size=2, reduction_mode=mode)


def test_agentic_grpo_token_mean_never_prices_an_unsupervised_row() -> None:
    """The token_mean denominator is never 0: an unsupervised row is refused upstream."""
    objective = AgenticGRPOLoss(group_size=2, reduction_mode="token_mean")
    batch = _experience(
        {
            "prompt_ids": ("a", "a"),
            "rewards": (0.0, 2.0),
            "loss_mask": ((0, 0), (0, 0)),
            "old_logprobs": ((-1.0, -1.0), (-1.0, -1.0)),
        }
    )
    with pytest.raises(AdvantageRefusal, match="supervises 0 of 2 positions"):
        objective.compute_with_report(lambda _batch: ((-1.0, -1.0), (-1.0, -1.0)), batch)


def test_token_mean_registry_binding_differs_only_in_the_reduction() -> None:
    from foundationscale.rl.group_policy import (
        agentic_grpo_algorithm,
        agentic_grpo_token_mean_algorithm,
    )

    control = agentic_grpo_algorithm()._objective
    arm = agentic_grpo_token_mean_algorithm()._objective
    assert (control.reduction, arm.reduction) == ("prompt_mean", "token_mean")
    assert arm.clip_bounds == control.clip_bounds
    assert arm.group_size == control.group_size
    assert arm.advantage_fn == control.advantage_fn
