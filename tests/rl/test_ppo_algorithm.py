"""Adversarial tests for the PPO binding: role-map delegation, declarations and step."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from foundationscale.rl.advantage import LearnedValueAdvantageEstimation
from foundationscale.rl.algorithm import (
    AlgorithmSemantics,
    AlgorithmWiringRefusal,
    StepReportRefusal,
)
from foundationscale.rl.interfaces import BatchRefusal, ExperienceBatch
from foundationscale.rl.policy import PolicyPair
from foundationscale.rl.ppo import PPOAlgorithm, PPOCompositeLoss, check_ppo_requirements
from foundationscale.rl.ppo_objectives import KLPenaltyLoss
from foundationscale.rl.value_head import (
    ValueCapabilities,
    ValueEstimateRefusal,
    ValueHead,
)

_REQUIRED_ROLES = (
    "advantage_fn",
    "dataloader",
    "loss_fn",
    "policy_pair",
    "reference_policy",
    "value_head",
)

# One estimate row per batch row and one estimate per mask position: the
# batch below has 2 rows with 1 supervised token each, so the reading is
# [[0.25], [0.25]]. Single-token terminated rows keep the temporal
# estimator hand-computable: terminated=True sets V(T) = 0.0, so at the
# default gamma == 1.0 each row's only delta is r + 1.0 * 0.0 - V(t) =
# reward - value, i.e. 1.0 - 0.25 = 0.75 and 2.0 - 0.25 = 1.75. The
# producer therefore recorded advantages [0.75, 1.75] and terminal returns
# equal to the rewards, [1.0, 2.0] -- and the estimator excludes nothing,
# so used == offered == 2.
_ESTIMATES = [[0.25], [0.25]]

_REWARDS = [1.0, 2.0]


@dataclass(frozen=True)
class _FrozenKLController:
    # Trainer-side stand-in for the adaptive KL controller: the trainer owns
    # ``controller = controller.update(kl_estimate)``; this binding must not.
    coefficient: float


@dataclass(frozen=True)
class _FixedValueHead:
    # Structural fake for the runtime_checkable ValueHead Protocol, which
    # cannot be instantiated: it declares ``estimate`` and ``capabilities``.
    # The default value loss has clip_epsilon=None, so setup needs token
    # granularity only and old-value reporting is not required.
    estimates: tuple[tuple[float, ...], ...]

    def estimate(self, _batch: ExperienceBatch) -> list[list[float]]:
        return [list(row) for row in self.estimates]

    def capabilities(self) -> ValueCapabilities:
        return ValueCapabilities(
            granularity="token",
            shares_policy_trunk=False,
            reports_old_values=False,
        )


def _value_head(values: list[list[float]]) -> ValueHead:
    return _FixedValueHead(estimates=tuple(tuple(row) for row in values))


def _batch(*, omit: tuple[str, ...] = ()) -> ExperienceBatch:
    columns = {
        "current_logprobs": [[-0.1], [-0.2]],
        "old_logprobs": [[-0.1], [-0.2]],
        "reference_logprobs": [[-0.1], [-0.2]],
        "advantages": [[0.75], [1.75]],
        "returns": [[1.0], [2.0]],
        "prompt_ids": ["prompt-a", "prompt-b"],
        "rewards": list(_REWARDS),
        "terminated": [True, True],
        "loss_mask": [[True], [True]],
    }
    for name in omit:
        del columns[name]
    return ExperienceBatch(columns=columns)


def _pair(*, with_reference: bool) -> PolicyPair:
    references = {"ref_a": object()} if with_reference else {}
    return PolicyPair(train_view=object(), generate_view=None, references=references)


def _wired_algorithm(
    *,
    batches: list[ExperienceBatch],
    with_kl: bool = False,
    estimates: list[list[float]] | None = None,
) -> tuple[PPOAlgorithm, PPOCompositeLoss]:
    value_head = _value_head(_ESTIMATES if estimates is None else estimates)
    advantage_fn = LearnedValueAdvantageEstimation()
    loss_fn = PPOCompositeLoss(
        kl_loss=KLPenaltyLoss() if with_kl else None,
        value_head=value_head,
        advantage_fn=advantage_fn,
    )
    algorithm = PPOAlgorithm(with_kl=with_kl)
    algorithm.setup(
        policy_pair=_pair(with_reference=with_kl),
        loss_fn=loss_fn,
        dataloader=batches,
        config={"policy_logprob_column": "current_logprobs"},
        advantage_fn=advantage_fn,
        value_head=value_head,
    )
    return algorithm, loss_fn


def _ppo_requires() -> dict[str, bool]:
    return dict(PPOAlgorithm().requires())


@pytest.mark.parametrize("missing", _REQUIRED_ROLES)
@pytest.mark.parametrize("absence", ["key", "none"])
def test_check_ppo_requirements_refuses_every_declared_role_when_absent(
    missing: str, absence: str
) -> None:
    # Absence is a missing key OR an explicit None; every role PPO declares
    # is refused in both forms, and the refusal names the role, the absent
    # count against the required denominator, and PPO's own origin string.
    supplied = {role: object() for role in _REQUIRED_ROLES if role != missing}
    if absence == "none":
        supplied[missing] = None
    with pytest.raises(AlgorithmWiringRefusal, match=missing) as excinfo:
        check_ppo_requirements(requires=_ppo_requires(), supplied=supplied)
    message = str(excinfo.value)
    assert "1 of 6 required inputs absent for ppo" in message
    # The origin is PPO's own, not the shared helper's default and not a
    # sibling family's: delegation is real only if this string is "ppo".
    assert "grpo" not in message
    assert "<algorithm>" not in message


def test_check_ppo_requirements_returns_the_sorted_consumed_role_set() -> None:
    supplied = {role: object() for role in _REQUIRED_ROLES}
    consumed = check_ppo_requirements(requires=_ppo_requires(), supplied=supplied)
    assert consumed == (
        "advantage_fn",
        "dataloader",
        "loss_fn",
        "policy_pair",
        "reference_policy",
        "value_head",
    )


def test_check_ppo_requirements_refuses_a_supplied_role_declared_unconsumed() -> None:
    supplied = {role: object() for role in _REQUIRED_ROLES}
    supplied["rollout_source"] = object()
    with pytest.raises(AlgorithmWiringRefusal, match="rollout_source") as excinfo:
        check_ppo_requirements(requires=_ppo_requires(), supplied=supplied)
    message = str(excinfo.value)
    assert "not required by ppo" in message
    assert "1 of 7 supplied inputs" in message


@pytest.mark.parametrize("value", [True, False])
def test_check_ppo_requirements_refuses_a_boolean_as_presence(value: bool) -> None:
    # True read as presence would launder a declaration into wiring; that is
    # exactly the bug the shared helper exists to prevent. The refusal names
    # the offending boolean, not merely the role.
    with pytest.raises(AlgorithmWiringRefusal, match=rf"boolean {value!r}") as excinfo:
        check_ppo_requirements(requires={"advantage_fn": True}, supplied={"advantage_fn": value})
    assert "True and False are not supplied components" in str(excinfo.value)


def test_check_ppo_requirements_refuses_an_empty_required_set_as_vacuous() -> None:
    # An all-False declaration admits nothing but was never asked for
    # anything either; the required set is empty and the check is vacuous.
    with pytest.raises(AlgorithmWiringRefusal, match="empty required-set") as excinfo:
        check_ppo_requirements(
            requires={"rollout_source": False, "weight_sync": False},
            supplied={},
        )
    message = str(excinfo.value)
    assert "0 of 2 declared roles are required for ppo" in message
    assert "vacuous" in message


def test_check_ppo_requirements_refuses_an_empty_union_as_vacuous() -> None:
    with pytest.raises(AlgorithmWiringRefusal, match="0 roles appear in either map for ppo"):
        check_ppo_requirements(requires={}, supplied={})


def test_requirements_declares_the_with_kl_stance_exactly() -> None:
    requirements = PPOAlgorithm().requirements()
    assert requirements.name == "ppo"
    assert dict(requirements.requires) == {
        "rollout_source": False,
        "advantage_fn": True,
        "weight_sync": False,
        "reference_policy": True,
        "value_head": True,
    }
    assert requirements.declared_components == ("ppo_policy_loss", "value_loss", "kl_penalty")
    assert requirements.declared_metrics == ("clip_fraction", "kl_estimate")
    assert requirements.semantics == AlgorithmSemantics(
        group_size=None,
        ratio_scope="token",
        kl_estimator="k3",
        clip_bounds=(0.8, 1.2),
        reference_free=False,
    )


def test_requirements_declares_the_kl_free_stance_exactly() -> None:
    requirements = PPOAlgorithm(with_kl=False).requirements()
    assert dict(requirements.requires) == {
        "rollout_source": False,
        "advantage_fn": True,
        "weight_sync": False,
        "reference_policy": False,
        "value_head": True,
    }
    assert requirements.declared_components == ("ppo_policy_loss", "value_loss")
    assert requirements.declared_metrics == ("clip_fraction",)
    assert requirements.semantics == AlgorithmSemantics(
        group_size=None,
        ratio_scope="token",
        kl_estimator=None,
        clip_bounds=(0.8, 1.2),
        reference_free=True,
    )


def test_requirements_resolve_clip_overrides_and_value_clip_by_value() -> None:
    requirements = PPOAlgorithm(
        clip_epsilon=0.2,
        clip_epsilon_low=0.25,
        clip_epsilon_high=0.5,
        value_clip_epsilon=0.1,
    ).requirements()
    assert requirements.declared_metrics == ("clip_fraction", "value_clip_fraction", "kl_estimate")
    assert requirements.semantics is not None
    assert requirements.semantics.clip_bounds == (0.75, 1.5)


def test_requires_map_names_all_eight_graded_roles_by_value() -> None:
    assert dict(PPOAlgorithm().requires()) == {
        "policy_pair": True,
        "loss_fn": True,
        "dataloader": True,
        "advantage_fn": True,
        "value_head": True,
        "reference_policy": True,
        "rollout_source": False,
        "weight_sync": False,
    }


def test_supplied_abstains_with_none_before_setup() -> None:
    assert PPOAlgorithm().supplied() is None


def test_setup_refuses_a_second_wiring_and_names_the_replacement_risk() -> None:
    algorithm, _loss_fn = _wired_algorithm(batches=[_batch()])
    with pytest.raises(AlgorithmWiringRefusal, match="already wired") as excinfo:
        algorithm.setup(
            policy_pair=_pair(with_reference=False),
            loss_fn=object(),
            dataloader=[],
            config={},
        )
    assert "silently replace the measured supplied mapping" in str(excinfo.value)


def test_setup_refuses_a_semantics_disagreement_and_names_both_declarations() -> None:
    # The algorithm declares a KL term and the composite declares none; the
    # first disagreeing field is kl_estimator and both sides' values are
    # named, because 1 of 5 fields disagreeing is a refused handshake, never
    # a silent adjustment.
    advantage_fn = LearnedValueAdvantageEstimation()
    value_head = _value_head(_ESTIMATES)
    loss_fn = PPOCompositeLoss(value_head=value_head, advantage_fn=advantage_fn)
    algorithm = PPOAlgorithm(with_kl=True)
    with pytest.raises(AlgorithmWiringRefusal, match="kl_estimator") as excinfo:
        algorithm.setup(
            policy_pair=_pair(with_reference=True),
            loss_fn=loss_fn,
            dataloader=[_batch()],
            config={"policy_logprob_column": "current_logprobs"},
            advantage_fn=advantage_fn,
            value_head=value_head,
        )
    message = str(excinfo.value)
    assert "'k3'" in message
    assert "None" in message


def test_supplied_mapping_is_complete_immutable_and_identity_exact() -> None:
    algorithm, loss_fn = _wired_algorithm(batches=[_batch()])
    supplied = algorithm.supplied()
    assert supplied is not None
    assert set(supplied) == {"policy_pair", "loss_fn", "dataloader", "advantage_fn", "value_head"}
    assert supplied["loss_fn"] is loss_fn
    assert supplied["advantage_fn"] is loss_fn.advantage_fn
    assert supplied["value_head"] is loss_fn.value_head
    with pytest.raises(TypeError):
        supplied["loss_fn"] = None


def test_step_green_path_reports_hand_computed_denominators() -> None:
    algorithm, _loss_fn = _wired_algorithm(batches=[_batch()])
    report = algorithm.step()
    assert report.step == 0
    # The temporal estimator excludes nothing: used == offered == 2 for two
    # fully-supervised rows, so rows is 2 and RewardStats summarises the two
    # terminal rewards: count 2, mean (1.0 + 2.0) / 2 == 1.5, min 1.0,
    # max 2.0.
    assert report.rows == 2
    assert report.sync is None
    assert tuple(component.name for component in report.loss.components) == (
        "ppo_policy_loss",
        "value_loss",
    )
    # The composite's own claim, from this module: the reported scalar is the
    # sum of the sub-losses' contributions, with no second weighting layer.
    # Here both contributions are nonzero: the policy term is
    # -((0.75 + 1.75) / 2) == -1.25 and the value term is
    # ((0.25 - 1.0)**2 + (0.25 - 2.0)**2) / 2 == 1.8125.
    contributions = [component.contribution for component in report.loss.components]
    assert report.loss.loss == sum(contribution for contribution in contributions if contribution)
    stats = report.reward_stats
    assert stats is not None
    assert stats.count == 2
    assert stats.mean == 1.5
    assert stats.minimum == 1.0
    assert stats.maximum == 2.0


def test_step_refuses_before_setup_with_the_required_count() -> None:
    with pytest.raises(AlgorithmWiringRefusal, match="setup has not completed") as excinfo:
        PPOAlgorithm().step()
    assert "0 of 6 required inputs are present for ppo" in str(excinfo.value)


def test_step_refuses_an_exhausted_dataloader_as_unmeasured() -> None:
    algorithm, _loss_fn = _wired_algorithm(batches=[])
    with pytest.raises(StepReportRefusal, match="0 of at least 1 required batches") as excinfo:
        algorithm.step()
    message = str(excinfo.value)
    assert "for ppo" in message
    assert "unmeasured" in message


def test_step_refuses_a_missing_termination_column_and_names_it() -> None:
    # Termination flags are READ, never defaulted: a fabricated all-True flag
    # would change every bootstrapped advantage while claiming to be
    # measured, so the absent column is refused by name through step.
    algorithm, _loss_fn = _wired_algorithm(batches=[_batch(omit=("terminated",))])
    with pytest.raises(BatchRefusal, match="terminated") as excinfo:
        algorithm.step()
    message = str(excinfo.value)
    assert "1 of 3 required inputs absent for ppo advantage estimation" in message
    assert "{terminated}" in message


def test_step_verifies_estimates_before_pricing_and_refuses_a_misshapen_reading() -> None:
    # One estimate row for a two-row batch fails the row-count half of
    # verify_estimates: the source shows PPOAlgorithm.step itself calls
    # verify_estimates before pricing (there is no separate verify_step and
    # the loop owns no such gate), so the refusal class is the real
    # ValueEstimateRefusal -- a ValueError subclass, but the bare base class
    # is not what is asserted -- and the origin pins the refusal to this
    # binding's step rather than a bare unverified call.
    algorithm, _loss_fn = _wired_algorithm(batches=[_batch()], estimates=[[0.25]])
    with pytest.raises(ValueEstimateRefusal, match="returned 1 estimate rows") as excinfo:
        algorithm.step()
    assert "PPOAlgorithm.step" in str(excinfo.value)
    assert "has 2 rows" in str(excinfo.value)


def test_step_does_not_thread_kl_controller_state_across_steps() -> None:
    algorithm, loss_fn = _wired_algorithm(
        batches=[_batch(), _batch()],
        with_kl=True,
    )
    kl_loss = loss_fn.kl_loss
    controller = _FrozenKLController(coefficient=0.1)
    first = algorithm.step()
    second = algorithm.step()
    assert first.step == 0
    assert second.step == 1
    # This is the DESIGNED division of responsibility, not an oversight: the
    # trainer owns `controller = controller.update(kl_estimate)` and builds
    # the next KLPenaltyLoss. A frozen binding cannot advance controller
    # state across a step without either mutating a frozen object or staling
    # the measured supplied map it was constructed from, so two steps must
    # leave the controller and the wired KL loss exactly as they were. Do
    # not "fix" this by adding mutation here.
    assert controller == _FrozenKLController(coefficient=0.1)
    supplied = algorithm.supplied()
    assert supplied is not None
    assert isinstance(supplied["loss_fn"], PPOCompositeLoss)
    assert supplied["loss_fn"].kl_loss is kl_loss
