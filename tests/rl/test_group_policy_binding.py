"""Additional coverage for the group-relative policy binding's refusal arms.

WHAT IS CLAIMED: these tests measure the uncovered surface of
``foundationscale.rl.group_policy`` -- the construction-time refusal for a
non-objective, the identity refusals in ``setup`` that fire before any
wiring handshake is consulted, the pre-setup ``step`` refusal, and the
declared surfaces of a real binding derived from a real objective. Every
object constructed here is real: the objectives are the module's own
shipped loss classes and the algorithm is bound through the module's own
public constructor, so nothing asserted here depends on a stub.

WHAT IS NOT CLAIMED: that the post-handshake ``setup`` arms or the ``step``
pricing path are exercised here. Those arms sit behind
``check_algorithm_wiring``, whose ``PolicyPair`` and batch construction live
outside this module, and a guessed construction of either would assert
behaviour this module does not own; per the binding's own docstrings, the
cross-check and role-mirror comparisons are edit-time tripwires no public
construction can fire, so they are deliberately not chased.
"""

from __future__ import annotations

import pytest

from foundationscale.rl.algorithm import AlgorithmWiringRefusal
from foundationscale.rl.group_policy import (
    SequencePolicyAlgorithm,
    check_group_policy_requirements,
)
from foundationscale.rl.group_policy_objectives import (
    DAPOLoss,
    DrGRPOLoss,
    GSPOLoss,
)


def test_init_refuses_non_objective_and_names_member_count() -> None:
    # An arbitrary object carries none of the nine protocol members, and the
    # refusal must say so -- a vague "wrong type" would hide which seams an
    # objective author still owes.
    with pytest.raises(AlgorithmWiringRefusal, match=r"must carry a SequenceObjective"):
        SequencePolicyAlgorithm(name="bad", objective=object())


def test_init_refusal_names_the_absent_member_diagnostic() -> None:
    # The refusal message embeds the absent-members tuple; a bare object is
    # missing all nine, so the denominator itself is the distinctive text.
    with pytest.raises(AlgorithmWiringRefusal, match=r"of the 9 required members"):
        SequencePolicyAlgorithm(name="bad", objective=42)


def test_semantics_derives_ratio_scope_and_clip_bounds_from_objective() -> None:
    # The binding must not restate geometry: the algorithm-side record is
    # read off the carried objective, so both sides are the same reading.
    objective = GSPOLoss()
    algorithm = SequencePolicyAlgorithm(name="gspo_test", objective=objective)
    semantics = algorithm.semantics()
    assert semantics.ratio_scope == objective.ratio_scope
    assert semantics.clip_bounds == objective.clip_bounds


def test_semantics_derives_reference_stance_from_objective_semantics() -> None:
    # reference_free comes from the objective's own semantics declaration,
    # never re-declared by hand in the binding.
    objective = DrGRPOLoss()
    algorithm = SequencePolicyAlgorithm(name="dr_grpo_test", objective=objective)
    assert algorithm.semantics().reference_free == objective.semantics().reference_free


def test_requires_declares_the_static_role_denominator() -> None:
    # The static map names every consumed AND unconsumed role; dropping a
    # False entry would let an unconsumed role read as unmeasured.
    algorithm = SequencePolicyAlgorithm(name="x", objective=DAPOLoss())
    requires = algorithm.requires()
    assert requires["rollout_source"] is True
    assert requires["advantage_fn"] is True
    assert requires["reference_policy"] is False
    assert requires["weight_sync"] is False


def test_requires_mapping_is_immutable() -> None:
    # The denominator is a MappingProxyType: a post-construction write would
    # move the gate's target, so mutation must refuse.
    algorithm = SequencePolicyAlgorithm(name="x", objective=GSPOLoss())
    with pytest.raises(TypeError):
        algorithm.requires()["rollout_source"] = False  # type: ignore[index]


def test_supplied_abstains_with_none_before_setup() -> None:
    # None -- not an empty map -- is the pre-setup reading: an empty map
    # would falsely look like a completed wiring check over zero roles.
    algorithm = SequencePolicyAlgorithm(name="x", objective=GSPOLoss())
    assert algorithm.supplied() is None


def test_setup_refuses_a_loss_that_is_not_the_carried_objective() -> None:
    # Identity, not equality: a second GSPOLoss agreeing everywhere still
    # doubles the one declared objective role, and the refusal must say two
    # objects were handed for one role before any handshake runs.
    carried = GSPOLoss()
    impostor = GSPOLoss()
    algorithm = SequencePolicyAlgorithm(name="gspo_test", objective=carried)
    with pytest.raises(AlgorithmWiringRefusal, match=r"2 distinct objects for 1"):
        algorithm.setup(
            policy_pair=object(),  # type: ignore[arg-type]
            loss_fn=impostor,
            dataloader=iter(()),
            config={"policy_logprob_column": "logprobs"},
        )


def test_setup_refuses_a_foreign_advantage_function() -> None:
    # The GRPO precedent: the wired estimator must BE the objective's own,
    # because two estimators disagreeing about what an advantage is cannot
    # both be right; a bare object fails identity unconditionally.
    objective = GSPOLoss()
    algorithm = SequencePolicyAlgorithm(name="gspo_test", objective=objective)
    with pytest.raises(
        AlgorithmWiringRefusal, match=r"2 distinct objects for 1 declared estimator"
    ):
        algorithm.setup(
            policy_pair=object(),  # type: ignore[arg-type]
            loss_fn=objective,
            dataloader=iter(()),
            config={"policy_logprob_column": "logprobs"},
            advantage_fn=object(),  # type: ignore[arg-type]
        )


def test_step_before_setup_refuses_as_unmeasured() -> None:
    # A step with no wiring is not a zero-row step: the refusal names the
    # absent supplied mapping as the cause rather than reporting nothing.
    algorithm = SequencePolicyAlgorithm(name="x", objective=GSPOLoss())
    with pytest.raises(AlgorithmWiringRefusal, match=r"setup has not completed"):
        algorithm.step()


def test_step_before_setup_refusal_names_the_zero_of_three_count() -> None:
    # The refusal carries its own denominator -- 0 of 3 required inputs -- so
    # a caller can tell a wiring gap apart from a data gap by the count.
    algorithm = SequencePolicyAlgorithm(name="x", objective=DrGRPOLoss())
    with pytest.raises(AlgorithmWiringRefusal, match=r"0 of 3 required inputs"):
        algorithm.step()


def test_check_group_policy_requirements_returns_sorted_consumed_roles() -> None:
    # The helper is the family's own re-entry into the shared role-map check
    # under its own origin: a fully satisfied required set returns the sorted
    # consumed roles, which later feed the two-denominator comparison.
    consumed = check_group_policy_requirements(
        requires={"rollout_source": True, "advantage_fn": True},
        supplied={"rollout_source": object(), "advantage_fn": object()},
        origin="binding_test",
    )
    assert consumed == ("advantage_fn", "rollout_source")


def test_each_factory_derives_a_binding_agreeing_with_its_objective_class() -> None:
    # Direct construction is the documented seam beside the factories: for
    # each shipped objective class, the derived record must read the
    # objective's own scope and clip declarations, never a restated copy.
    for name, objective in (
        ("gspo", GSPOLoss()),
        ("dr_grpo", DrGRPOLoss()),
        ("dapo", DAPOLoss()),
    ):
        algorithm = SequencePolicyAlgorithm(name=name, objective=objective)
        assert algorithm.semantics().ratio_scope == objective.ratio_scope
        assert algorithm.semantics().clip_bounds == objective.clip_bounds
        assert algorithm.requirements().name == name
