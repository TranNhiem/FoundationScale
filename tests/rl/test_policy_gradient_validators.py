"""Input-validation adversary tests for the stage-3e policy-gradient validators.

Every malformed input here is pushed through a PUBLIC entry point -- the loss
constructors, the loss call/compute surface, or the three public ``check_*``
role helpers of ``foundationscale.rl.policy_gradient``. No private helper is
imported: each targeted guard (``_checked_name``, ``_positive_finite``,
``_outer_sequence``, ``_row_sequence``, ``_mask_entry``, ``_logprob_value``,
``_reward_scalar``, ``_advantage_weight``, ``_token_ratio``,
``_supplied_is_present`` and ``_check_policy_gradient_roles``) is exercised by
the public caller that reaches it in production order, so the refusals
asserted here are the ones real callers observe.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import pytest

from foundationscale.rl.advantage import LeaveOneOutAdvantage
from foundationscale.rl.algorithm import AlgorithmWiringRefusal
from foundationscale.rl.interfaces import BatchRefusal, LossConfigRefusal
from foundationscale.rl.policy_gradient import (
    ReinforceBaselineLoss,
    ReinforcePlusPlusLoss,
    RLOOPolicyLoss,
    check_reinforce_baseline_requirements,
    check_reinforce_pp_requirements,
    check_rloo_requirements,
)

__all__ = ()


@dataclass(frozen=True, slots=True)
class _Batch:
    """Minimal ExperienceBatch-shaped fake with explicit row count.

    Unlike the main test module's fake, ``__len__`` never inspects the column
    VALUES, which is exactly what lets a malformed (non-iterable, ragged,
    non-finite) column survive long enough to reach the guard under test.
    """

    source: Mapping[str, Any]
    length: int

    @property
    def columns(self) -> tuple[str, ...]:
        return tuple(self.source)

    def column(self, name: str) -> Any:
        return self.source[name]

    def __len__(self) -> int:
        return self.length


def _rigged_advantage(report: Any) -> LeaveOneOutAdvantage:
    """Build a LeaveOneOutAdvantage subclass whose compute returns ``report``.

    The subclass exists only to satisfy RLOOPolicyLoss's isinstance and
    ``min_group_size`` checks; the report is consumed structurally (attribute
    access) by ``compute_with_report``, so a plain namespace suffices.
    """

    class _Rigged(LeaveOneOutAdvantage):
        def compute(self, *, prompt_ids: Any, rewards: Any, mask: Any) -> Any:
            _ = (prompt_ids, rewards, mask)
            return report

    return _Rigged()


def _weight_report(raw: Any) -> Any:
    """A two-row advantage report whose first token weight is ``raw``."""
    return SimpleNamespace(
        weights=((raw,), (0.5,)),
        rows=(0, 1),
        used=2,
        offered=2,
        rewards=None,
    )


def _rloo_group_batch() -> _Batch:
    return _Batch(
        {
            "prompt_ids": ("p", "p"),
            "rewards": (1.0, 2.0),
            "loss_mask": ((1,), (1,)),
            "old_logprobs": ((-1.0,), (-1.0,)),
        },
        2,
    )


def test_checked_name_refuses_empty_and_non_string_names() -> None:
    """Every name-bearing config field refuses a non-name at construction.

    WHAT IS CLAIMED: an empty string and non-string values (an int, bytes)
    for three different name fields across all three losses refuse with the
    shared "non-empty strings" refusal, and the message names the exact field
    that failed -- which is what the module header promises over letting the
    name drift into a later gate error.

    WHAT IS NOT CLAIMED: that any VALID name is checked for collisions or
    uniqueness; only absence-of-name shapes are refused here.
    """
    with pytest.raises(LossConfigRefusal, match="must be non-empty strings") as exc_info:
        RLOOPolicyLoss(component_name="")
    assert "component_name=''" in str(exc_info.value)
    with pytest.raises(LossConfigRefusal, match="must be non-empty strings") as exc_info:
        ReinforceBaselineLoss(mask_column=12)
    assert "mask_column=12" in str(exc_info.value)
    with pytest.raises(LossConfigRefusal, match="must be non-empty strings") as exc_info:
        ReinforcePlusPlusLoss(reward_column=b"rewards")
    assert "reward_column=b'rewards'" in str(exc_info.value)


def test_positive_finite_refuses_boolean_weights_before_numeric_read() -> None:
    """A boolean True is refused as a weight before it can read as 1.0.

    WHAT IS CLAIMED: the bool branch fires ahead of the numeric test for two
    distinct ``_positive_finite`` call sites (RLOO weight, Reinforce++
    kl_beta), each naming its own mechanism string in the refusal.

    WHAT IS NOT CLAIMED: anything about other non-positive or non-finite
    weights; those numeric branches are covered by the existing test module.
    """
    with pytest.raises(
        LossConfigRefusal,
        match="True is not a declared RLOO component weight",
    ):
        RLOOPolicyLoss(weight=True)
    with pytest.raises(LossConfigRefusal, match="True is not a reward-penalty weight"):
        ReinforcePlusPlusLoss(kl_beta=True)


def test_outer_sequence_refuses_non_iterable_column_and_forward_output() -> None:
    """A non-iterable column or forward output refuses before any row is read.

    WHAT IS CLAIMED: a plain int as the rewards column, and a plain int as
    the forward output, both refuse with the "1 of 1 required inputs was not
    iterable" denominator, naming the offending field and the actual type.

    WHAT IS NOT CLAIMED: that a non-iterable row reaches the same refusal;
    per-row iterability is the separate ``_row_sequence`` guard.
    """
    loss_fn = ReinforceBaselineLoss()
    bad_rewards = _Batch({"rewards": 5, "loss_mask": ((1,), (1,))}, 2)
    with pytest.raises(BatchRefusal, match="1 of 1 required inputs was not iterable") as exc_info:
        loss_fn(lambda _batch: ((-1.0,), (-2.0,)), bad_rewards)
    message = str(exc_info.value)
    assert "rewards=5" in message
    assert "(int)" in message
    fine_batch = _Batch({"rewards": (1.0, 2.0), "loss_mask": ((1,), (1,))}, 2)
    with pytest.raises(BatchRefusal, match="1 of 1 required inputs was not iterable") as exc_info:
        loss_fn(lambda _batch: 9, fine_batch)
    assert "forward_fn(batch)=9" in str(exc_info.value)


def test_row_sequence_refuses_non_iterable_token_rows() -> None:
    """A non-iterable token row refuses with its row index named.

    WHAT IS CLAIMED: an int as a mask row, and an int as a forward-output
    row, both refuse with the "1 of 1 rows was not iterable" message naming
    the field, the row, and the value; the batch-level checks (column
    presence, outer lengths) all passed first, so this is the row-level
    guard and not an earlier one.

    WHAT IS NOT CLAIMED: that the contents of a well-formed iterable row are
    valid; per-entry validity is the mask/logprob guards measured elsewhere.
    """
    loss_fn = ReinforceBaselineLoss()
    bad_mask_row = _Batch({"rewards": (1.0, 2.0), "loss_mask": (7, (1,))}, 2)
    with pytest.raises(BatchRefusal, match="1 of 1 rows was not iterable") as exc_info:
        loss_fn(lambda _batch: ((-1.0,), (-2.0,)), bad_mask_row)
    assert "field loss_mask row 0=7" in str(exc_info.value)
    fine_batch = _Batch({"rewards": (1.0, 2.0), "loss_mask": ((1,), (1,))}, 2)
    with pytest.raises(BatchRefusal, match="1 of 1 rows was not iterable") as exc_info:
        loss_fn(lambda _batch: (7, (-2.0,)), fine_batch)
    assert "field forward_fn output row 0=7" in str(exc_info.value)


def test_mask_entry_reads_booleans_as_their_truth_value() -> None:
    """The bool branch of the mask guard is deliberate, tested behaviour.

    Hand computation: rewards (2, 4), mean return 3.0 as the abstained
    baseline; row 0's mask is False (skipped), row 1's mask is True with
    advantage +1 times logprob -2.0, so the single supervised token prices
    to contribution 2.0 and the above-baseline fraction is 1 of 2 rows.

    WHAT IS CLAIMED: bool mask entries are accepted and read exactly as
    their truth value -- False skips the token, True supervises it -- and
    the resulting loss equals the hand computation.

    WHAT IS NOT CLAIMED: that bool masks are good practice; only that the
    helper's bool-first branch behaves as the code intends.
    """
    loss_fn = ReinforceBaselineLoss()
    batch = _Batch({"rewards": (2.0, 4.0), "loss_mask": ((False,), (True,))}, 2)
    output = loss_fn(lambda _batch: ((-1.0,), (-2.0,)), batch)
    assert output.loss == pytest.approx(2.0)
    assert output.components[0].contribution == pytest.approx(2.0)
    assert output.metrics[0].value == pytest.approx(0.5)


def test_mask_entry_refuses_unconvertible_and_out_of_range_entries() -> None:
    """Only 0/1-shaped entries survive; strings and other floats refuse.

    WHAT IS CLAIMED: a non-numeric string entry takes the
    convert-then-refuse path (the try/except arm) and a float like 2.0 takes
    the out-of-range arm, both landing on the same refusal naming row,
    position, and the offending entry.

    WHAT IS NOT CLAIMED: that digit-shaped strings like "1" refuse -- they
    convert to 1.0 and are accepted; only genuinely non-0/1 entries fail.
    """
    loss_fn = ReinforceBaselineLoss()
    stringy = _Batch({"rewards": (1.0, 2.0), "loss_mask": (("x",), (1,))}, 2)
    with pytest.raises(
        BatchRefusal,
        match="supervision mask entries must be 0 or 1",
    ) as exc_info:
        loss_fn(lambda _batch: ((-1.0,), (-2.0,)), stringy)
    assert "row 0, position 0 is 'x'" in str(exc_info.value)
    out_of_range = _Batch({"rewards": (1.0, 2.0), "loss_mask": ((2.0,), (1,))}, 2)
    with pytest.raises(
        BatchRefusal,
        match="supervision mask entries must be 0 or 1",
    ) as exc_info:
        loss_fn(lambda _batch: ((-1.0,), (-2.0,)), out_of_range)
    assert "row 0, position 0 is 2.0" in str(exc_info.value)


def test_logprob_value_refuses_unconvertible_and_non_finite_entries() -> None:
    """Each token log-probability must convert to one finite float.

    WHAT IS CLAIMED: a None entry, and a string token row (a str IS a
    sequence, so it flows one level deeper and its characters are refused as
    non-floats, never silently accepted as a scalar), hit the
    does-not-convert arm; positive infinity and NaN hit the not-finite arm.
    All four name the field, the row, and the position.

    WHAT IS NOT CLAIMED: that converting digit strings refuse -- float("1")
    is a legitimate finite reading by this module's rules.
    """
    loss_fn = ReinforceBaselineLoss()
    fine_batch = _Batch({"rewards": (1.0, 2.0), "loss_mask": ((1,), (1,))}, 2)
    with pytest.raises(BatchRefusal, match="does not convert to a scalar float") as exc_info:
        loss_fn(lambda _batch: ((None,), (-2.0,)), fine_batch)
    assert "field current_logprobs at row 0, position 0" in str(exc_info.value)
    string_row = _Batch({"rewards": (1.0, 2.0), "loss_mask": ((1, 1), (1,))}, 2)
    with pytest.raises(BatchRefusal, match="does not convert to a scalar float") as exc_info:
        loss_fn(lambda _batch: ("xy", (-2.0,)), string_row)
    # The refusal names the offending entry's TYPE, not its repr: the row "xy"
    # is a sequence, so position 0 is the one-character str "x", and what the
    # operator is told is that a str arrived where a float was required.
    assert "at row 0, position 0 does not convert to a scalar float (str)" in str(exc_info.value)
    with pytest.raises(BatchRefusal, match="which is not finite") as exc_info:
        loss_fn(lambda _batch: ((math.inf,), (-2.0,)), fine_batch)
    assert "is inf" in str(exc_info.value)
    with pytest.raises(BatchRefusal, match="which is not finite") as exc_info:
        loss_fn(lambda _batch: ((math.nan,), (-2.0,)), fine_batch)
    assert "is nan" in str(exc_info.value)


def test_reward_scalar_refuses_unconvertible_and_non_finite_rewards() -> None:
    """Each reward must be one finite scalar before any mean is derived.

    WHAT IS CLAIMED: a None reward hits the does-not-convert arm, and both
    infinity and NaN hit the not-finite arm, each naming the row and value;
    the refusals fire before returns_mean is computed, so no poisoned mean
    can enter the baseline.

    WHAT IS NOT CLAIMED: that the batch mean itself can be non-finite from
    finite rewards (overflow of sum); only per-reward entry validity is
    measured here.
    """
    loss_fn = ReinforceBaselineLoss()
    none_reward = _Batch({"rewards": (None, 2.0), "loss_mask": ((1,), (1,))}, 2)
    with pytest.raises(BatchRefusal, match="does not convert to a scalar float") as exc_info:
        loss_fn(lambda _batch: ((-1.0,), (-2.0,)), none_reward)
    assert "reward at row 0 is None" in str(exc_info.value)
    inf_reward = _Batch({"rewards": (math.inf, 2.0), "loss_mask": ((1,), (1,))}, 2)
    with pytest.raises(BatchRefusal, match="which is not finite") as exc_info:
        loss_fn(lambda _batch: ((-1.0,), (-2.0,)), inf_reward)
    assert "reward at row 0 is inf" in str(exc_info.value)
    nan_reward = _Batch({"rewards": (math.nan, 2.0), "loss_mask": ((1,), (1,))}, 2)
    with pytest.raises(BatchRefusal, match="which is not finite") as exc_info:
        loss_fn(lambda _batch: ((-1.0,), (-2.0,)), nan_reward)
    assert "reward at row 0 is nan" in str(exc_info.value)


def test_advantage_weight_refuses_bool_unconvertible_and_non_finite() -> None:
    """The loss refuses a bad per-token advantage weight before pricing it.

    WHAT IS CLAIMED: through the PUBLIC ``compute_with_report`` entry point,
    a bool weight, an unconvertible weight, and a NaN weight each refuse
    with the row-and-position-named BatchRefusal -- the real LeaveOneOut
    estimator only ever emits finite floats, so a minimal subclass feeding a
    caller-built report is used to deliver the malformed weight to the
    guard.

    WHAT IS NOT CLAIMED: that the real estimator could ever produce these
    shapes -- it cannot; the rigged subclass is a stand-in purely to reach
    the loss-side guard. The entry point is public, but the data path is
    staged rather than estimator-produced.
    """
    batch = _rloo_group_batch()
    forward = ((-1.0,), (-1.0,))
    bool_weight = RLOOPolicyLoss(advantage_fn=_rigged_advantage(_weight_report(True)))
    with pytest.raises(BatchRefusal, match="a bool is not a measured token weight") as exc_info:
        bool_weight.compute_with_report(lambda _batch: forward, batch)
    assert "advantage weight at row 0, position 0 is True" in str(exc_info.value)
    text_weight = RLOOPolicyLoss(advantage_fn=_rigged_advantage(_weight_report("heavy")))
    with pytest.raises(BatchRefusal, match="does not convert to a scalar float") as exc_info:
        text_weight.compute_with_report(lambda _batch: forward, batch)
    assert "advantage weight at row 0, position 0" in str(exc_info.value)
    nan_weight = RLOOPolicyLoss(advantage_fn=_rigged_advantage(_weight_report(math.nan)))
    with pytest.raises(BatchRefusal, match="which is not finite") as exc_info:
        nan_weight.compute_with_report(lambda _batch: forward, batch)
    assert "advantage weight at row 0, position 0 is nan" in str(exc_info.value)


def test_token_ratio_refuses_overflow_and_underflow_before_use() -> None:
    """An unrepresentable or zeroed-out token ratio refuses rather than reads.

    WHAT IS CLAIMED: through the real estimator's weights, a log-ratio of
    +1000 overflows exp() and refuses as not representable (naming the exact
    ``exp(1000.0)``), while a log-ratio of -1000 underflows to exactly 0.0
    and refuses rather than manufacturing a measured-looking zero
    contribution -- the two failure shapes of the one ratio guard.

    WHAT IS NOT CLAIMED: that ratios near but inside those bounds are
    stable or trusted; only the two refusal shapes are measured here.
    """
    loss_fn = RLOOPolicyLoss()
    batch = _Batch(
        {
            "prompt_ids": ("p", "p"),
            "rewards": (1.0, 2.0),
            "loss_mask": ((1,), (1,)),
            "old_logprobs": ((0.0,), (0.0,)),
        },
        2,
    )
    with pytest.raises(BatchRefusal, match="is not representable") as exc_info:
        loss_fn(lambda _batch: ((1000.0,), (0.0,)), batch)
    assert "exp(1000.0)" in str(exc_info.value)
    with pytest.raises(BatchRefusal, match="not a positive finite value") as exc_info:
        loss_fn(lambda _batch: ((-1000.0,), (0.0,)), batch)
    assert "token ratio 0.0" in str(exc_info.value)


def test_role_check_refuses_non_mapping_arguments() -> None:
    """Both mapping arguments must implement Mapping before any key is read.

    WHAT IS CLAIMED: a list ``requires`` and an int ``supplied`` refuse with
    the 1-of-1-mapping denominator, on two different public check helpers --
    the refusal fires before dict() conversion could mask the type.

    WHAT IS NOT CLAIMED: anything about mapping CONTENTS; key/value shape
    refusals are separate guards measured next.
    """
    with pytest.raises(
        AlgorithmWiringRefusal,
        match="1 of 1 requirement mappings must implement Mapping",
    ) as exc_info:
        check_rloo_requirements(requires=["loss_fn"], supplied={})
    assert "requires=['loss_fn']" in str(exc_info.value)
    with pytest.raises(
        AlgorithmWiringRefusal,
        match="1 of 1 supplied mappings must implement Mapping",
    ) as exc_info:
        check_reinforce_baseline_requirements(requires={"loss_fn": True}, supplied=42)
    assert "supplied=42" in str(exc_info.value)


def test_role_check_refuses_bad_names_and_non_bool_requirement_values() -> None:
    """Names and requirement truth values are graded before presence checks.

    WHAT IS CLAIMED: an empty-string and a non-string requirement name
    refuse naming the bad-name count (1 of 1, 1 of 2 denominators), on both
    the requires side and the supplied side; and a non-bool requirement
    value refuses with its own denominator because a required role must be
    measured True or declared unrequired False, never a truthy stand-in.

    WHAT IS NOT CLAIMED: that the same bad names would survive a different
    check helper; all three helpers share the one implementation, and only
    two origins are sampled here.
    """
    with pytest.raises(
        AlgorithmWiringRefusal,
        match="names are not non-empty strings",
    ) as exc_info:
        check_reinforce_pp_requirements(requires={"": True}, supplied={"loss_fn": object()})
    assert "1 of 1 names" in str(exc_info.value)
    with pytest.raises(
        AlgorithmWiringRefusal,
        match="names are not non-empty strings",
    ) as exc_info:
        check_rloo_requirements(
            requires={"loss_fn": True},
            supplied={"loss_fn": object(), 7: object()},
        )
    assert "1 of 2 names" in str(exc_info.value)
    with pytest.raises(
        AlgorithmWiringRefusal,
        match="requirement values are not bool",
    ) as exc_info:
        check_reinforce_baseline_requirements(
            requires={"loss_fn": 1},
            supplied={"loss_fn": object()},
        )
    assert "1 of 1 requirement" in str(exc_info.value)


def test_role_check_refuses_absent_boolean_and_unrequired_supplied_roles() -> None:
    """Presence is measured by the actual object, never None or a boolean.

    WHAT IS CLAIMED: a None-valued required role reads as ABSENT (the
    ``_supplied_is_present`` None arm) so the absent-required refusal names
    1 of 1 required inputs for the rloo origin; True and False as supplied
    values refuse as not-supplied-components; and a present but unrequired
    role refuses naming the 1-of-2 denominator for the reinforce_pp origin.

    WHAT IS NOT CLAIMED: that the order of these refusals is arbitrary --
    boolean laundering is caught at presence-read time, before the absent
    and unrequired set checks would otherwise mis-measure the wiring.
    """
    with pytest.raises(AlgorithmWiringRefusal, match="required inputs absent") as exc_info:
        check_rloo_requirements(requires={"loss_fn": True}, supplied={"loss_fn": None})
    message = str(exc_info.value)
    assert "1 of 1 required inputs absent for rloo" in message
    assert "('loss_fn',)" in message
    assert "1 supplied keys" in message
    for boolean in (True, False):
        with pytest.raises(
            AlgorithmWiringRefusal,
            match="are not supplied components",
        ) as exc_info:
            check_reinforce_baseline_requirements(
                requires={"loss_fn": True},
                supplied={"loss_fn": boolean},
            )
        assert f"the boolean {boolean!r}" in str(exc_info.value)
    with pytest.raises(
        AlgorithmWiringRefusal,
        match="are not required by reinforce_pp",
    ) as exc_info:
        check_reinforce_pp_requirements(
            requires={"loss_fn": True, "weight_sync": False},
            supplied={"weight_sync": object(), "loss_fn": object()},
        )
    message = str(exc_info.value)
    assert "1 of 2 supplied inputs" in message
    assert "('weight_sync',)" in message


def test_role_check_returns_sorted_consumed_roles_over_none_absences() -> None:
    """The happy path returns exactly the required roles that are present.

    WHAT IS CLAIMED: a supplied mapping mixing real objects for required
    roles with None for a declared-False role yields the sorted tuple of
    consumed roles, with the None entry counted as absent rather than
    rejected -- proving the None arm of presence is a measurement input,
    not only a refusal trigger.

    WHAT IS NOT CLAIMED: that the returned role OBJECTS are usable; the
    check measures presence only, and this test measures only the tuple.
    """
    consumed = check_reinforce_pp_requirements(
        requires={
            "policy_pair": True,
            "reference_policy": True,
            "advantage_fn": False,
        },
        supplied={
            "advantage_fn": None,
            "reference_policy": object(),
            "policy_pair": object(),
        },
    )
    assert consumed == ("policy_pair", "reference_policy")
