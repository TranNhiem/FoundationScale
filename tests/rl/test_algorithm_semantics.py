"""Tests for the D1/D2 requirements surface: AlgorithmSemantics and the
requires/supplied mapping handshake."""

from __future__ import annotations

import dataclasses

import pytest

from foundationscale.rl.algorithm import (
    AlgorithmRequirements,
    AlgorithmSemantics,
    AlgorithmWiringRefusal,
    check_algorithm_wiring,
)


def test_semantics_default_construction_is_full_abstention() -> None:
    # Abstention case 1: an algorithm that constrains no seam is
    # representable as all-None, and None on every field is a stated
    # non-constraint -- not a default smuggled in as a measurement.
    semantics = AlgorithmSemantics()
    assert semantics.group_size is None
    assert semantics.ratio_scope is None
    assert semantics.kl_estimator is None
    assert semantics.clip_bounds is None
    assert semantics.reference_free is None
    # slots=True is not optional: no per-instance dict may exist.
    assert not hasattr(semantics, "__dict__")


def test_semantics_is_frozen_against_mutation() -> None:
    semantics = AlgorithmSemantics(group_size=8)
    with pytest.raises(dataclasses.FrozenInstanceError):
        semantics.group_size = 9  # type: ignore[misc]


def test_semantics_group_size_zero_and_bool_are_refused_not_abstention() -> None:
    # 0 is a measured zero, True sorts as K=1; neither is an abstention.
    with pytest.raises(AlgorithmWiringRefusal) as zero_exc:
        AlgorithmSemantics(group_size=0)
    assert "group_size" in str(zero_exc.value)
    with pytest.raises(AlgorithmWiringRefusal) as bool_exc:
        AlgorithmSemantics(group_size=True)  # type: ignore[arg-type]
    assert "group_size" in str(bool_exc.value)
    assert AlgorithmSemantics(group_size=1).group_size == 1


def test_semantics_ratio_scope_outside_declared_union_is_refused() -> None:
    assert AlgorithmSemantics(ratio_scope="token").ratio_scope == "token"
    assert AlgorithmSemantics(ratio_scope="sequence").ratio_scope == "sequence"
    with pytest.raises(AlgorithmWiringRefusal) as excinfo:
        AlgorithmSemantics(ratio_scope="group")  # type: ignore[arg-type]
    assert "ratio_scope" in str(excinfo.value)


def test_semantics_clip_bounds_must_be_an_ordered_pair() -> None:
    assert AlgorithmSemantics(clip_bounds=(0.8, 1.2)).clip_bounds == (0.8, 1.2)
    with pytest.raises(AlgorithmWiringRefusal) as excinfo:
        AlgorithmSemantics(clip_bounds=(1.2, 0.8))
    assert "clip_bounds" in str(excinfo.value)


def test_requirements_empty_requires_is_refused_as_vacuous() -> None:
    # The founding defect at the declaration layer: an empty required-set
    # is refused, never silently satisfied.
    with pytest.raises(AlgorithmWiringRefusal) as excinfo:
        AlgorithmRequirements(name="grpo", requires={}, declared_components=("pg_loss",))
    message = str(excinfo.value)
    assert "EMPTY" in message
    assert "0 roles" in message


def test_requirements_non_bool_role_value_is_refused() -> None:
    # A truthy stand-in (1, "yes") would let an unmeasured value pose as
    # a declaration; only True/False are declarations here.
    with pytest.raises(AlgorithmWiringRefusal) as excinfo:
        AlgorithmRequirements(
            name="ppo",
            requires={"critic": 1},  # type: ignore[dict-item]
            declared_components=("policy_loss",),
        )
    assert "critic" in str(excinfo.value)


def test_requirements_offline_shape_kept_and_sync_without_rollout_refused() -> None:
    # requires advantage without rollout is the legitimate offline shape
    # (algorithm.py's original comment); it must survive the mapping rework.
    offline = AlgorithmRequirements(
        name="dpo-style",
        requires={"advantage_fn": True},
        declared_components=("preference_loss",),
    )
    assert offline.requires["advantage_fn"] is True
    with pytest.raises(AlgorithmWiringRefusal) as excinfo:
        AlgorithmRequirements(
            name="incoherent",
            requires={"weight_sync": True},
            declared_components=("pg_loss",),
        )
    assert "weight_sync" in str(excinfo.value)
    assert "rollout_source" in str(excinfo.value)


def test_wiring_required_role_absent_refuses_naming_both_counts() -> None:
    # The mapping half of the check reads nothing off the pair, so this
    # refusal fires before the pair's type is ever examined.
    requirements = AlgorithmRequirements(
        name="grpo",
        requires={"advantage_fn": True, "rollout_source": True},
        declared_components=("pg_loss",),
    )
    with pytest.raises(AlgorithmWiringRefusal) as excinfo:
        check_algorithm_wiring(
            requirements,
            policy_pair=object(),  # type: ignore[arg-type]
            loss_fn=object(),  # type: ignore[arg-type]
            supplied={"advantage_fn": object()},
        )
    message = str(excinfo.value)
    assert "1 of 2" in message
    assert "rollout_source" in message


def test_wiring_supplied_role_with_no_requirement_enters_denominator_and_refuses() -> None:
    # D2's reason for existing: a critic no declaration names must sit IN
    # the denominator via the union, not outside it. requires alone would
    # print CLEAR over this wiring; the union refuses it as 1 of 2.
    requirements = AlgorithmRequirements(
        name="sft",
        requires={"rollout_source": False},
        declared_components=("sft_loss",),
    )
    with pytest.raises(AlgorithmWiringRefusal) as excinfo:
        check_algorithm_wiring(
            requirements,
            policy_pair=object(),  # type: ignore[arg-type]
            loss_fn=object(),  # type: ignore[arg-type]
            supplied={"critic": object()},
        )
    message = str(excinfo.value)
    assert "1 of 2" in message
    assert "critic" in message
    assert "does not consume" in message


def test_wiring_none_entry_in_supplied_is_abstention_not_presence() -> None:
    # Abstention case 2: a role stated on the supplied side with value
    # None is absence, not a stub. The wiring must PASS the role stage and
    # fail later -- on the foreign pair object -- proving by the refusal's
    # wording that no role disagreement was ever recorded against None.
    requirements = AlgorithmRequirements(
        name="sft",
        requires={"rollout_source": False},
        declared_components=("sft_loss",),
    )
    with pytest.raises(AlgorithmWiringRefusal) as excinfo:
        check_algorithm_wiring(
            requirements,
            policy_pair=object(),  # type: ignore[arg-type]
            loss_fn=object(),  # type: ignore[arg-type]
            supplied={"rollout_source": None},
        )
    message = str(excinfo.value)
    assert "PolicyPair" in message
    assert "does not consume" not in message


def test_wiring_empty_union_is_refused_as_vacuous_defense_in_depth() -> None:
    # The in-function guard: a requirements object that bypassed
    # __post_init__ (as a hand-rolled stand-in for this contract could)
    # must still not produce a CLEAR over an empty denominator.
    requirements = object.__new__(AlgorithmRequirements)
    object.__setattr__(requirements, "name", "ghost")
    object.__setattr__(requirements, "requires", {})
    object.__setattr__(requirements, "declared_components", ("pg_loss",))
    object.__setattr__(requirements, "declared_metrics", ())
    object.__setattr__(requirements, "semantics", None)
    with pytest.raises(AlgorithmWiringRefusal) as excinfo:
        check_algorithm_wiring(
            requirements,
            policy_pair=object(),  # type: ignore[arg-type]
            loss_fn=object(),  # type: ignore[arg-type]
            supplied={},
        )
    message = str(excinfo.value)
    assert "0 roles named in requires" in message
    assert "0 roles named in supplied" in message
