"""Registry and GRPO binding tests at the pure stdlib seam."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, get_origin

import pytest

from foundationscale import rl as rl_package
from foundationscale.rl.advantage import (
    AdvantageRefusal,
    AdvantageResult,
    GroupNormalisedAdvantage,
    LeaveOneOutAdvantage,
    RewardStats,
)
from foundationscale.rl.algorithm import (
    Algorithm,
    AlgorithmWiringRefusal,
    StepReport,
    StepReportRefusal,
    verify_step,
)
from foundationscale.rl.grpo import (
    GRPOAlgorithm,
    GRPOPolicyLoss,
    _Wiring,
    check_grpo_requirements,
)
from foundationscale.rl.interfaces import (
    BatchRefusal,
    LossConfigRefusal,
    SupervisionRefusal,
)
from foundationscale.rl.registry import (
    AlgorithmRegistryRefusal,
    available_algorithm_names,
    lookup_algorithm,
    register_algorithm,
    reset_algorithm_registry,
)

__all__ = ()


@dataclass(frozen=True, slots=True)
class _Batch:
    source: Mapping[str, Any]

    @property
    def columns(self) -> tuple[str, ...]:
        return tuple(self.source)

    def column(self, name: str) -> Any:
        return self.source[name]

    def __len__(self) -> int:
        if not self.source:
            return 0
        return len(next(iter(self.source.values())))


def _grpo_batch(*, rewards: tuple[float, float] = (2.0, 4.0)) -> _Batch:
    return _Batch(
        {
            "prompt_ids": ("prompt-a", "prompt-a"),
            "rewards": rewards,
            "loss_mask": ((1, 1), (1, 1)),
            "old_logprobs": ((-2.2, -1.2), (-1.2, -1.2)),
            "reference_logprobs": ((-2.2, -1.2), (-1.2, -1.2)),
        }
    )


def test_registry_default_reset_restores_the_thirteen_default_names() -> None:
    """The supported reset leaves exactly the thirteen default bindings.

    WHAT IS CLAIMED: reset removes test entries and reinstalls all thirteen
    built-in names, and a second reset restores the SAME set. The thirteen are
    hand-stated here rather than read back from the registry, because a test
    that asked the registry what it holds would agree with any answer.

    Restoring the same set every time is a property of WHERE the install
    lives: the registry installs the built-ins itself, so a family module
    that is never re-imported still comes back. A family that registered on
    its own import would survive only the first reset.

    WHAT IS NOT CLAIMED: that this list is COMPLETE with respect to the
    bindings the package ships. Hand-stating the set is what makes the claim
    checkable, and it is also what makes it one-directional: it catches a
    binding that appears, and it cannot catch one that never arrives. That
    second direction is a different measurement and is taken by
    ``test_every_exported_algorithm_binding_is_reachable_through_the_registry``.
    Also not claimed: that the thirteen bindings are interchangeable, or any
    of them is wired -- lookup constructs, setup is a separate handshake.
    """
    register_algorithm("z_test_algorithm", GRPOAlgorithm)
    reset_algorithm_registry()
    expected = (
        "cpo",
        "dapo",
        "dpo",
        "dr_grpo",
        "grpo",
        "gspo",
        "ipo",
        "kto",
        "orpo",
        "reinforce_baseline",
        "reinforce_pp",
        "rloo",
        "simpo",
    )
    assert available_algorithm_names() == expected
    reset_algorithm_registry()
    assert available_algorithm_names() == expected


def _is_plain_class(candidate: object) -> bool:
    """Return True for a real class and False for a parametrised generic alias.

    ``isinstance(x, type)`` is NOT that predicate, and the gap between them is
    interpreter-dependent: ``isinstance(Callable[[X], Y], type)`` is True on
    Python 3.10 and False on 3.11+. A package that exports a type ALIAS
    therefore reaches ``issubclass`` on the OLDEST supported interpreter only,
    where it raises ``TypeError: issubclass() arg 1 must be a class`` --
    ``ForwardFn = Callable[[ExperienceBatch], Any]`` is such an alias, and it
    is exported. The local gate runs one interpreter, so a divergence of this
    shape is invisible until CI's 3.10 leg runs it; that is how it shipped.

    ``get_origin`` is the half that means the same thing everywhere: it is
    non-None for every parametrised alias on every version this package
    supports, so the composite selects the same objects on all of them and the
    sweep's denominator stops depending on the interpreter.
    """
    return isinstance(candidate, type) and get_origin(candidate) is None


def test_the_exported_class_filter_excludes_a_parametrised_alias() -> None:
    """``_is_plain_class`` discriminates alias from class on the running version.

    WHAT IS CLAIMED: the filter guarding the ``__all__`` sweep below rejects the
    alias the package actually exports, accepts a real binding, and is
    load-bearing rather than decorative.

    The reject leg alone would be VACUOUS on 3.11+, where ``isinstance`` already
    excludes the alias and ``get_origin`` is never consulted -- it would pass on
    every interpreter except the one that was broken. So the two legs that carry
    the measurement are the version-stable ones: ``get_origin`` is asserted
    non-None directly, and ``issubclass`` is shown to RAISE on the same object,
    which it does on every supported version. Together they say the filter is
    the only thing standing between the sweep and a hard error.

    WHAT IS NOT CLAIMED: that ``ForwardFn`` is the only non-class export, or
    that the sweep's other exclusions are correct; the protocol exclusion is
    argued at its own site.
    """
    # Read the alias off the PACKAGE, which is where the sweep reads it from.
    # Importing it from its defining module instead would measure a same-named
    # object rather than the one the sweep is handed.
    alias = rl_package.ForwardFn
    assert "ForwardFn" in rl_package.__all__, (
        "the sweep no longer sees a parametrised alias, so this control guards "
        "a hypothetical; re-point it at whatever non-class name __all__ exports"
    )
    assert get_origin(alias) is not None
    with pytest.raises(TypeError, match="issubclass"):
        issubclass(alias, Algorithm)  # type: ignore[arg-type]
    assert _is_plain_class(alias) is False
    # Accept side: a filter that rejected everything would empty the sweep's
    # denominator and make its "no exported bindings found" assertion the only
    # thing left standing.
    assert _is_plain_class(GRPOAlgorithm) is True


def test_every_exported_algorithm_binding_is_reachable_through_the_registry() -> None:
    """Every ``Algorithm`` the package exports can be resolved by name.

    WHAT IS CLAIMED: for each public name in ``foundationscale.rl`` that is a
    class satisfying the ``Algorithm`` protocol, some registered name
    constructs an instance of exactly that class. The reachable side is
    measured by CONSTRUCTING through ``lookup_algorithm`` rather than by
    reading the registry mapping, so a name bound to a factory that cannot
    build is not counted as reachable.

    This is the direction the reset test above cannot see. A binding that
    ships a module, an ``__init__`` export and a full test suite, and is then
    simply never added to the default install, is invisible to a test that
    pins the install against a hand-written list -- the list and the install
    are the same statement made twice, so they agree by construction. That is
    exactly how ``PPOAlgorithm`` shipped: ``lookup_algorithm("ppo")`` refuses
    with "0 of 13 available algorithm names matched" while the class is
    public, zero-argument, and passes the registry's own runtime
    ``isinstance`` check.

    An unreachable binding may be DECLARED, and the declaration must say why.
    A declaration is not a suppression: the second assertion below fails when
    a declared name becomes reachable, so the note cannot outlive the gap it
    describes. The abstention exists because the alternative -- deleting the
    test until the gap closes -- is how the gap stayed invisible in the first
    place.

    WHAT IS NOT CLAIMED: that the registry holds NO name beyond the exported
    bindings -- registering an extra name is the documented extension point,
    and the reset test is what pins the DEFAULT set. Also not claimed: that a
    reachable binding is correct, or that its ``setup`` succeeds against any
    particular role map; nor that a DECLARED binding is unusable -- an
    explicit ``register_algorithm`` call reaches it, only the default install
    does not.
    """
    declared_unreachable = {
        "PPOAlgorithm": (
            "finding #359: PPOAlgorithm is not a STATIC subtype of Algorithm. Its "
            "setup narrows advantage_fn to LearnedValueAdvantageEstimation, which "
            "advantage.py declares a SIBLING of AdvantageFn rather than a subtype, "
            "and a parameter type is contravariant -- so a setup accepting less "
            "than the protocol promises cannot stand in for it. isinstance cannot "
            "see this: Algorithm is runtime_checkable, which measures attribute "
            "PRESENCE only. Registering the binding is what makes mypy check the "
            "registry's Callable[[], Algorithm] slot, and it is red. The "
            "resolution is a separate protocol slot per seam rather than a union, "
            "because both seams are structurally just __call__ and no runtime "
            "check can separate them; that touches all five algorithm families."
        ),
    }

    reset_algorithm_registry()
    exported: dict[str, type] = {}
    for name in rl_package.__all__:
        candidate = getattr(rl_package, name)
        # _is_plain_class, not a bare isinstance(candidate, type): the package
        # exports a type ALIAS, and on Python 3.10 an alias answers True to
        # that check and then explodes inside issubclass. See the helper.
        if not _is_plain_class(candidate) or not issubclass(candidate, Algorithm):
            continue
        # ``Algorithm`` itself satisfies ``issubclass(Algorithm, Algorithm)``, and a
        # contract is not a binding: nothing constructs the protocol, so counting it
        # would make the denominator permanently short by one. The exclusion keys on
        # what the object IS -- a protocol -- and not on the name ``Algorithm``, so a
        # second protocol added to this package is excluded for the same reason
        # rather than needing its own entry.
        if getattr(candidate, "_is_protocol", False):
            continue
        exported[name] = candidate
    assert exported, "no exported Algorithm bindings found; the denominator is vacuous"
    unknown_declarations = sorted(set(declared_unreachable) - set(exported))
    assert unknown_declarations == [], (
        f"{len(unknown_declarations)} declared-unreachable name(s) are not exported "
        f"Algorithm bindings at all: {unknown_declarations}; a declaration that names "
        f"nothing suppresses nothing and hides a rename"
    )

    reachable = {type(lookup_algorithm(name)) for name in available_algorithm_names()}
    undeclared = sorted(
        name
        for name, binding in exported.items()
        if binding not in reachable and name not in declared_unreachable
    )
    assert undeclared == [], (
        f"{len(undeclared)} of {len(exported)} exported Algorithm binding(s) cannot "
        f"be resolved by any registered name and are not declared: {undeclared}; "
        f"registered names ({len(available_algorithm_names())}): "
        f"{available_algorithm_names()}"
    )

    stale = sorted(name for name in declared_unreachable if exported[name] in reachable)
    assert stale == [], (
        f"{len(stale)} of {len(declared_unreachable)} declared-unreachable binding(s) "
        f"are now reachable: {stale}; the declaration outlived the gap and must be "
        f"deleted, not carried"
    )


def test_registry_refuses_duplicate_name_without_overwrite() -> None:
    """A second registration cannot launder a different binding into one name.

    WHAT IS CLAIMED: the refusal names ``grpo``, the one attempted
    registration, and the four names already registered.

    WHAT IS NOT CLAIMED: which factory object was attempted; identity is not
    needed to establish the duplicate key.
    """
    reset_algorithm_registry()
    first = lookup_algorithm("grpo")
    with pytest.raises(
        AlgorithmRegistryRefusal,
        match="duplicate registration requested",
    ) as exc_info:
        register_algorithm("grpo", lambda: GRPOAlgorithm())
    message = str(exc_info.value)
    assert "field name='grpo'" in message
    assert "1 of 1 new registrations" in message
    assert "1 of 13 registered names" in message
    assert lookup_algorithm("grpo") is not first
    reset_algorithm_registry()


def test_registry_unknown_name_names_key_and_available_count() -> None:
    """Lookup absence is reported against every available registry name.

    WHAT IS CLAIMED: the requested key and all thirteen available names appear.

    WHAT IS NOT CLAIMED: that the requested spelling was close to ``grpo``;
    no correction or distance measurement exists.
    """
    reset_algorithm_registry()
    with pytest.raises(
        AlgorithmRegistryRefusal,
        match="0 of 13 available algorithm names matched",
    ) as exc_info:
        lookup_algorithm("missing")
    message = str(exc_info.value)
    assert "requested key name='missing'" in message
    assert (
        "available names (13): ('cpo', 'dapo', 'dpo', 'dr_grpo', 'grpo', "
        "'gspo', 'ipo', 'kto', 'orpo', 'reinforce_baseline', 'reinforce_pp', "
        "'rloo', 'simpo')"
    ) in message
    reset_algorithm_registry()


def test_registry_lookup_returns_fresh_algorithm_object() -> None:
    """Lookup constructs rather than sharing a configured algorithm.

    WHAT IS CLAIMED: two lookups are distinct objects satisfying the runtime
    ``Algorithm`` protocol check in the registry.

    WHAT IS NOT CLAIMED: that either returned object is wired; setup remains
    a separate handshake.
    """
    reset_algorithm_registry()
    first = lookup_algorithm("grpo")
    second = lookup_algorithm("grpo")
    assert isinstance(first, GRPOAlgorithm)
    assert isinstance(second, GRPOAlgorithm)
    assert first is not second
    assert first.supplied() is None
    assert second.supplied() is None


def test_registry_available_names_are_sorted_after_extra_registration() -> None:
    """Insertion order cannot become reported registry order.

    WHAT IS CLAIMED: ``a_first`` sorts before every default entry.

    WHAT IS NOT CLAIMED: case-insensitive or natural-language sorting; this
    test measures ordinary tuple sorting of exact strings.
    """
    reset_algorithm_registry()
    register_algorithm("a_first", lambda: GRPOAlgorithm())
    assert available_algorithm_names() == (
        "a_first",
        "cpo",
        "dapo",
        "dpo",
        "dr_grpo",
        "grpo",
        "gspo",
        "ipo",
        "kto",
        "orpo",
        "reinforce_baseline",
        "reinforce_pp",
        "rloo",
        "simpo",
    )
    assert tuple(sorted(available_algorithm_names())) == available_algorithm_names()
    reset_algorithm_registry()


def test_grpo_semantics_are_the_concrete_declared_values() -> None:
    """The algorithm-side GRPO declaration states its exact geometry.

    WHAT IS CLAIMED: group size, token scope, k3, clipping, and reference
    dependence are all present and non-abstaining for this binding.

    WHAT IS NOT CLAIMED: that another GRPO implementation must choose K=2;
    that value is this concrete default, not a theorem of the algorithm.
    """
    semantics = GRPOAlgorithm().semantics()
    assert semantics.group_size == 2
    assert semantics.ratio_scope == "token"
    assert semantics.kl_estimator == "k3"
    assert semantics.clip_bounds == (0.8, 1.2)
    assert semantics.reference_free is False


def test_grpo_declared_requirements_refuse_missing_advantage() -> None:
    """An absent required role is measured against the union denominator.

    WHAT IS CLAIMED: four supplied required inputs cannot produce CLEAR
    while the fifth required input, ``advantage_fn``, is absent.

    WHAT IS NOT CLAIMED: that the supplied stand-ins are valid loss or model
    objects; this test isolates presence from type validation.
    """
    algorithm = GRPOAlgorithm()
    requires = dict(algorithm.requires())
    required_names = tuple(name for name, required in requires.items() if required)
    supplied = {name: object() for name in required_names if name != "advantage_fn"}
    with pytest.raises(
        Exception,
        match="1 of 5 required inputs absent for grpo",
    ) as exc_info:
        check_grpo_requirements(
            requires=requires,
            supplied=supplied,
            origin="grpo",
        )
    message = str(exc_info.value)
    assert "advantage_fn" in message
    assert "7 roles from 7 requires keys and 4 supplied keys" in message


def test_grpo_end_to_end_advantage_loss_and_step_numbers() -> None:
    """A hand-computed two-sample GRPO batch reaches a verified step report.

    Hand computation: rewards 2 and 4 have mean 3 and population std 1, so
    their advantages are -1 and +1. Each of the four current-old log-ratio
    terms is 0.2. Its raw ratio is exp(0.2) but clipping uses 1.2, giving
    policy objective sum

        2 * min(-exp(0.2), -1.2) + 2 * min(exp(0.2), 1.2)

    Every k3 term is exp(-0.2) - (-0.2) - 1 and the KL weight is 0.04.

    WHAT IS CLAIMED: estimator weights, component values, total loss, and
    ``verify_step`` agree with those hand-computed values.

    WHAT IS NOT CLAIMED: numerical equivalence to an external framework or
    any convergence result.
    """
    loss_fn = GRPOPolicyLoss()
    output, advantage = loss_fn.compute_with_report(
        lambda batch: ((-2.0, -1.0), (-1.0, -1.0)),
        _grpo_batch(),
    )
    assert advantage.weights == ((-1.0, -1.0), (1.0, 1.0))
    assert advantage.rows == (0, 1)
    assert advantage.used == 2
    assert advantage.offered == 2
    assert advantage.rewards.count == 2
    assert advantage.rewards.mean == pytest.approx(3.0)
    assert advantage.rewards.std == pytest.approx(1.0)

    ratio = math.exp(0.2)
    expected_policy = -((2.0 * min(-ratio, -1.2) + 2.0 * min(ratio, 1.2)) / 4.0)
    expected_kl = 0.04 * (math.expm1(-0.2) + 0.2)
    assert output.components[0].name == "grpo_policy_loss"
    assert output.components[0].contribution == pytest.approx(expected_policy)
    assert output.components[1].name == "grpo_kl"
    assert output.components[1].contribution == pytest.approx(expected_kl)
    assert output.loss == pytest.approx(expected_policy + expected_kl)

    report = StepReport(
        step=0,
        loss=output,
        rows=advantage.used,
        reward_stats=advantage.rewards,
    )
    assert verify_step(report, GRPOAlgorithm().requirements()) == 2


def test_grpo_loss_refuses_absent_required_batch_columns() -> None:
    """The required-column check names both the absent and required counts.

    WHAT IS CLAIMED: one prompt-id column cannot satisfy the five-column
    GRPO batch schema.

    WHAT IS NOT CLAIMED: whether the remaining four values would have been
    valid had they been present; absence is measured before shape.
    """
    loss_fn = GRPOPolicyLoss()
    with pytest.raises(
        BatchRefusal,
        match="4 of 5 required inputs absent for grpo",
    ) as exc_info:
        loss_fn(
            lambda batch: ((-1.0, -1.0),),
            _Batch({"prompt_ids": ("prompt-a", "prompt-a")}),
        )
    message = str(exc_info.value)
    assert "rewards" in message
    assert "loss_mask" in message
    assert "old_logprobs" in message
    assert "reference_logprobs" in message


def test_grpo_refuses_group_and_estimator_size_disagreement() -> None:
    """A minimum-size estimator cannot impersonate an exact GRPO K.

    WHAT IS CLAIMED: a loss declaring K=3 while owning a K=2 estimator is a
    construction-time configuration refusal.

    WHAT IS NOT CLAIMED: that K=2 or K=3 is universally correct; only their
    disagreement is refused.
    """
    with pytest.raises(
        Exception,
        match="1 configured GRPO group size disagrees with 1 estimator minimum",
    ) as exc_info:
        GRPOPolicyLoss(group_size=3)
    message = str(exc_info.value)
    assert "group_size=3" in message
    assert "min_group_size=2" in message


def test_grpo_degenerate_group_cannot_report_zero_gradient() -> None:
    """A zero-variance group leaves no unmeasured zero-weight denominator.

    WHAT IS CLAIMED: all-equal rewards in the only group cannot produce a
    successful GRPO loss merely by emitting zeros.

    WHAT IS NOT CLAIMED: that a measured advantage of zero is generally
    invalid; this case is invalid because no relative group signal exists.
    """
    loss_fn = GRPOPolicyLoss()
    with pytest.raises(
        AdvantageRefusal,
        match="empty sequence: nothing used means nothing measured",
    ):
        loss_fn.compute_with_report(
            lambda batch: ((-2.0, -1.0), (-1.0, -1.0)),
            _grpo_batch(rewards=(1.0, 1.0)),
        )


@dataclass(frozen=True)
class _ScriptedAdvantage(GroupNormalisedAdvantage):
    """Return a caller-chosen compacted result, skipping input validation.

    WHAT IS CLAIMED: the estimator slot yields exactly the scripted weights
    over the first offered rows, measured against the offered count plus
    ``extra_offered``.

    WHAT IS NOT CLAIMED: that the script is a plausible advantage result;
    several tests script states the real estimator would refuse, so that
    grpo's own shape and weight refusals can be measured first.
    """

    script_weights: tuple[tuple[Any, ...], ...] = ((-1.0, -1.0), (1.0, 1.0))
    extra_offered: int = 0

    def compute(
        self,
        *,
        prompt_ids: Sequence[str],
        rewards: Sequence[float],
        mask: Sequence[Sequence[int]],
    ) -> AdvantageResult:
        del mask  # scripted: grpo re-validates the raw batch mask itself
        weights = tuple(tuple(row) for row in self.script_weights)
        cleaned = tuple(float(reward) for reward in rewards)
        return AdvantageResult(
            weights=weights,
            rewards=RewardStats.over(cleaned[: len(weights)]),
            used=len(weights),
            offered=len(prompt_ids) + self.extra_offered,
            method="ScriptedAdvantage",
            rows=tuple(range(len(weights))),
        )


def test_grpo_config_refuses_boolean_group_size() -> None:
    """WHAT IS CLAIMED: True is refused as a group size, not read as 1.

    WHAT IS NOT CLAIMED: that any integer of at least 2 is universally
    correct; only the bool laundering is measured here.
    """
    with pytest.raises(
        LossConfigRefusal,
        match="True is not a GRPO group size",
    ) as exc_info:
        GRPOPolicyLoss(group_size=True)
    message = str(exc_info.value)
    assert "group_size=True" in message
    assert "a group size is an integer of at least 2" in message


def test_grpo_config_refuses_group_size_below_two() -> None:
    """WHAT IS CLAIMED: a singleton group has no baseline and is refused.

    WHAT IS NOT CLAIMED: that 2 is the optimum K; only the below-2 state is
    measured.
    """
    with pytest.raises(
        LossConfigRefusal,
        match="at least 2 samples",
    ) as exc_info:
        GRPOPolicyLoss(group_size=1)
    message = str(exc_info.value)
    assert "group_size=1" in message
    assert "there is no within-group baseline to estimate" in message


def test_grpo_config_refuses_non_finite_clip_bound() -> None:
    """WHAT IS CLAIMED: a non-finite clip bound is refused with its field.

    WHAT IS NOT CLAIMED: anything about finite-but-degenerate intervals;
    those are a distinct refusal measured separately.
    """
    with pytest.raises(
        LossConfigRefusal,
        match="a GRPO ratio clip bound must be finite",
    ) as exc_info:
        GRPOPolicyLoss(clip_high=float("inf"))
    message = str(exc_info.value)
    assert "clip_high=inf" in message


def test_grpo_config_refuses_degenerate_clip_bounds() -> None:
    """WHAT IS CLAIMED: both sides of the clip interval are named when the
    interval is degenerate.

    WHAT IS NOT CLAIMED: which ordering a future relaxed bound might take;
    only the two degenerate shapes of this declaration are measured.
    """
    with pytest.raises(
        LossConfigRefusal,
        match="a degenerate interval cannot",
    ) as exc_zero:
        GRPOPolicyLoss(clip_low=0.0)
    assert "clip_bounds=(0.0, 1.2)" in str(exc_zero.value)
    with pytest.raises(
        LossConfigRefusal,
        match="must be strictly ordered",
    ) as exc_swapped:
        GRPOPolicyLoss(clip_low=1.5, clip_high=1.2)
    assert "clip_bounds=(1.5, 1.2)" in str(exc_swapped.value)


def test_grpo_config_refuses_non_positive_component_weight() -> None:
    """WHAT IS CLAIMED: zero and negative component weights are refused with
    the offending field named.

    WHAT IS NOT CLAIMED: the magnitude of any admissible weight; only the
    non-positive edge is measured.
    """
    with pytest.raises(
        LossConfigRefusal,
        match="a positive finite weight",
    ) as exc_kl:
        GRPOPolicyLoss(kl_weight=0.0)
    assert "kl_weight=0.0" in str(exc_kl.value)
    with pytest.raises(
        LossConfigRefusal,
        match="outside the coverage denominator",
    ):
        GRPOPolicyLoss(weight=-2.5)


def test_grpo_config_refuses_shared_component_name() -> None:
    """WHAT IS CLAIMED: two components cannot share one denominator name.

    WHAT IS NOT CLAIMED: anything about name spelling; only the collision
    of the two declared names is measured.
    """
    with pytest.raises(
        LossConfigRefusal,
        match="2 components would share 1 name",
    ) as exc_info:
        GRPOPolicyLoss(
            component_name="grpo_policy_loss",
            kl_component_name="grpo_policy_loss",
        )
    message = str(exc_info.value)
    assert "are both 'grpo_policy_loss'" in message
    assert "collapse the objective denominator" in message


def test_grpo_config_refuses_wrong_advantage_estimator() -> None:
    """WHAT IS CLAIMED: a different estimator cannot fill the GRPO slot.

    WHAT IS NOT CLAIMED: that LeaveOneOutAdvantage is invalid elsewhere;
    only its presence in the GRPO slot is refused.
    """
    with pytest.raises(
        LossConfigRefusal,
        match="1 of 1 estimator slots",
    ) as exc_info:
        GRPOPolicyLoss(advantage_fn=LeaveOneOutAdvantage())
    message = str(exc_info.value)
    assert "LeaveOneOutAdvantage" in message
    assert "not an undeclared substitute" in message


def test_grpo_config_refuses_empty_column_name() -> None:
    """WHAT IS CLAIMED: an empty string is not a column name.

    WHAT IS NOT CLAIMED: any other naming convention; only absence
    laundering is measured.
    """
    with pytest.raises(
        LossConfigRefusal,
        match="absence of a name is not a name",
    ) as exc_info:
        GRPOPolicyLoss(prompt_id_column="")
    message = str(exc_info.value)
    assert "prompt_id_column=''" in message
    assert "non-empty strings" in message


def test_grpo_batch_refuses_zero_row_batch() -> None:
    """WHAT IS CLAIMED: a zero-row batch is unmeasured, never 0.0.

    WHAT IS NOT CLAIMED: that one row is sufficient for GRPO; the group
    geometry check above this one still applies.
    """
    loss_fn = GRPOPolicyLoss()
    with pytest.raises(
        BatchRefusal,
        match="never 0.0",
    ) as exc_info:
        loss_fn.compute_with_report(
            lambda batch: (),
            _Batch(
                {
                    "prompt_ids": (),
                    "rewards": (),
                    "loss_mask": (),
                    "old_logprobs": (),
                    "reference_logprobs": (),
                }
            ),
        )
    message = str(exc_info.value)
    assert "0 of at least 1 required batch rows were supplied for grpo" in message


def test_grpo_batch_refuses_non_iterable_required_column() -> None:
    """WHAT IS CLAIMED: a scalar column cannot stand in for per-row values.

    WHAT IS NOT CLAIMED: which reward values the rows would have carried;
    the outer iterable requirement is checked before any value semantics.
    """
    loss_fn = GRPOPolicyLoss()
    with pytest.raises(
        BatchRefusal,
        match="was not iterable",
    ) as exc_info:
        loss_fn.compute_with_report(
            lambda batch: ((-1.0, -1.0), (-1.0, -1.0)),
            _Batch(
                {
                    "prompt_ids": ("prompt-a", "prompt-a"),
                    "rewards": 7,
                    "loss_mask": ((1, 1), (1, 1)),
                    "old_logprobs": ((-1.0, -1.0), (-1.0, -1.0)),
                    "reference_logprobs": ((-1.0, -1.0), (-1.0, -1.0)),
                }
            ),
        )
    message = str(exc_info.value)
    assert "field rewards=7" in message
    assert "one outer entry per batch row is required" in message


def test_grpo_batch_refuses_prompt_id_count_mismatch() -> None:
    """WHAT IS CLAIMED: the offered denominator names both row counts.

    WHAT IS NOT CLAIMED: which of the two counts is the truthful one; the
    loss measures only their disagreement.
    """
    loss_fn = GRPOPolicyLoss()
    with pytest.raises(
        BatchRefusal,
        match="must be the one offered denominator",
    ) as exc_info:
        loss_fn.compute_with_report(
            lambda batch: ((-1.0, -1.0), (-1.0, -1.0), (-1.0, -1.0)),
            _Batch(
                {
                    "rewards": (2.0, 4.0, 6.0),
                    "prompt_ids": ("prompt-a", "prompt-a"),
                    "loss_mask": ((1, 1), (1, 1), (1, 1)),
                    "old_logprobs": ((-1.0, -1.0), (-1.0, -1.0), (-1.0, -1.0)),
                    "reference_logprobs": ((-1.0, -1.0), (-1.0, -1.0), (-1.0, -1.0)),
                }
            ),
        )
    message = str(exc_info.value)
    assert "field prompt_ids: 2 values were supplied for 3 batch rows" in message


def test_grpo_batch_refuses_unhashable_prompt_id() -> None:
    """WHAT IS CLAIMED: an unhashable prompt id cannot group, and is named.

    WHAT IS NOT CLAIMED: that any particular hashable type is admissible;
    only the ungroupable state is measured.
    """
    loss_fn = GRPOPolicyLoss()
    with pytest.raises(
        BatchRefusal,
        match="not hashable",
    ) as exc_info:
        loss_fn.compute_with_report(
            lambda batch: ((-1.0, -1.0), (-1.0, -1.0)),
            _Batch(
                {
                    "prompt_ids": ([1], [1]),
                    "rewards": (2.0, 4.0),
                    "loss_mask": ((1, 1), (1, 1)),
                    "old_logprobs": ((-1.0, -1.0), (-1.0, -1.0)),
                    "reference_logprobs": ((-1.0, -1.0), (-1.0, -1.0)),
                }
            ),
        )
    message = str(exc_info.value)
    assert "prompt id at row 0 is [1]" in message


def test_grpo_batch_refuses_prompt_group_smaller_than_k() -> None:
    """WHAT IS CLAIMED: a group short of K names the prompt and both counts.

    WHAT IS NOT CLAIMED: that K=2 is universally correct; only the declared-
    K versus offered-group disagreement is measured.
    """
    loss_fn = GRPOPolicyLoss()
    with pytest.raises(
        BatchRefusal,
        match="1 prompt group disagrees with 1 declared K",
    ) as exc_info:
        loss_fn.compute_with_report(
            lambda batch: ((-1.0, -1.0), (-1.0, -1.0), (-1.0, -1.0)),
            _Batch(
                {
                    "prompt_ids": ("prompt-a", "prompt-a", "prompt-b"),
                    "rewards": (2.0, 4.0, 6.0),
                    "loss_mask": ((1, 1), (1, 1), (1, 1)),
                    "old_logprobs": ((-1.0, -1.0), (-1.0, -1.0), (-1.0, -1.0)),
                    "reference_logprobs": ((-1.0, -1.0), (-1.0, -1.0), (-1.0, -1.0)),
                }
            ),
        )
    message = str(exc_info.value)
    assert "prompt 'prompt-b' carries 1 of 3 batch rows" in message
    assert "GRPO declares group_size=2" in message


def test_grpo_batch_refuses_overreported_advantage_denominator() -> None:
    """WHAT IS CLAIMED: an estimator result naming more rows than the batch
    offers cannot anchor the step.

    WHAT IS NOT CLAIMED: that the real estimator can produce such a result;
    this measures the binding's own denominator handshake over a scripted
    estimator output.
    """
    loss_fn = GRPOPolicyLoss(advantage_fn=_ScriptedAdvantage(extra_offered=1))
    with pytest.raises(
        BatchRefusal,
        match="cannot anchor a GRPO step",
    ) as exc_info:
        loss_fn.compute_with_report(
            lambda batch: ((-1.0, -1.0), (-1.0, -1.0)),
            _grpo_batch(),
        )
    message = str(exc_info.value)
    assert "AdvantageResult.offered=3 but the batch contains 2 rows" in message


def test_grpo_batch_refuses_mismatched_forward_row_count() -> None:
    """WHAT IS CLAIMED: the forward contract is one token row per batch row.

    WHAT IS NOT CLAIMED: that the forward values themselves were valid; the
    row-count handshake fails before any token value is read.
    """
    loss_fn = GRPOPolicyLoss()
    with pytest.raises(
        BatchRefusal,
        match="one token row per batch row is required",
    ) as exc_info:
        loss_fn.compute_with_report(
            lambda batch: ((-1.0, -1.0),),
            _grpo_batch(),
        )
    message = str(exc_info.value)
    assert "forward_fn returned 1 rows for a batch of 2 rows" in message


def test_grpo_batch_refuses_extra_mask_rows() -> None:
    """WHAT IS CLAIMED: the mask row count is still the batch denominator,
    reported with both counts.

    WHAT IS NOT CLAIMED: that the extra mask row's contents were valid; the
    count disagreement refuses before any per-token reading.
    """
    loss_fn = GRPOPolicyLoss(advantage_fn=_ScriptedAdvantage())
    with pytest.raises(
        BatchRefusal,
        match="mask rows were supplied",
    ) as exc_info:
        loss_fn.compute_with_report(
            lambda batch: ((-1.0, -1.0), (-1.0, -1.0)),
            _Batch(
                {
                    "prompt_ids": ("prompt-a", "prompt-a"),
                    "rewards": (2.0, 4.0),
                    "loss_mask": ((1, 1), (1, 1), (1, 1)),
                    "old_logprobs": ((-1.0, -1.0), (-1.0, -1.0)),
                    "reference_logprobs": ((-1.0, -1.0), (-1.0, -1.0)),
                }
            ),
        )
    message = str(exc_info.value)
    assert "field loss_mask: 3 mask rows were supplied for 2 batch rows" in message


def test_grpo_batch_refuses_short_old_logprob_rows() -> None:
    """WHAT IS CLAIMED: a short old-logprob column is named with both counts.

    WHAT IS NOT CLAIMED: that the present row's values were finite; the row
    count disagreement refuses before any token reading.
    """
    loss_fn = GRPOPolicyLoss()
    with pytest.raises(
        BatchRefusal,
        match="log-probability rows",
    ) as exc_info:
        loss_fn.compute_with_report(
            lambda batch: ((-1.0, -1.0), (-1.0, -1.0)),
            _Batch(
                {
                    "prompt_ids": ("prompt-a", "prompt-a"),
                    "rewards": (2.0, 4.0),
                    "loss_mask": ((1, 1), (1, 1)),
                    "old_logprobs": ((-1.0, -1.0),),
                    "reference_logprobs": ((-1.0, -1.0), (-1.0, -1.0)),
                }
            ),
        )
    message = str(exc_info.value)
    assert (
        "field old_logprobs: 1 old log-probability rows were supplied for 2 batch rows" in message
    )


def test_grpo_batch_refuses_extra_reference_rows() -> None:
    """WHAT IS CLAIMED: an over-long reference column is named with counts.

    WHAT IS NOT CLAIMED: that two of the three rows were valid; the count
    disagreement refuses before any token reading.
    """
    loss_fn = GRPOPolicyLoss()
    with pytest.raises(
        BatchRefusal,
        match="reference rows were supplied",
    ) as exc_info:
        loss_fn.compute_with_report(
            lambda batch: ((-1.0, -1.0), (-1.0, -1.0)),
            _Batch(
                {
                    "prompt_ids": ("prompt-a", "prompt-a"),
                    "rewards": (2.0, 4.0),
                    "loss_mask": ((1, 1), (1, 1)),
                    "old_logprobs": ((-1.0, -1.0), (-1.0, -1.0)),
                    "reference_logprobs": ((-1.0, -1.0), (-1.0, -1.0), (-1.0, -1.0)),
                }
            ),
        )
    message = str(exc_info.value)
    assert "field reference_logprobs: 3 reference rows were supplied for 2 batch rows" in message


def test_grpo_batch_refuses_non_iterable_forward_row() -> None:
    """WHAT IS CLAIMED: a scalar forward row cannot carry per-token values.

    WHAT IS NOT CLAIMED: what the row's tokens would have been; the grasp
    on iteration is measured before any token reading.
    """
    loss_fn = GRPOPolicyLoss()
    with pytest.raises(
        BatchRefusal,
        match="1 of 1 rows was not iterable",
    ) as exc_info:
        loss_fn.compute_with_report(
            lambda batch: (7, 7),
            _grpo_batch(),
        )
    message = str(exc_info.value)
    assert "field forward_fn output row 0=7" in message
    assert "GRPO requires one entry per token position" in message


def test_grpo_batch_refuses_non_convertible_mask_entry() -> None:
    """WHAT IS CLAIMED: a non-numeric mask entry is refused with coordinates.

    WHAT IS NOT CLAIMED: that 0/1 floats are the only admissible numbers;
    that shape is measured by the same refusal the real estimator shares.
    """
    loss_fn = GRPOPolicyLoss(advantage_fn=_ScriptedAdvantage())
    with pytest.raises(
        BatchRefusal,
        match="supervision mask entries must be 0 or 1",
    ) as exc_info:
        loss_fn.compute_with_report(
            lambda batch: ((-1.0, -1.0), (-1.0, -1.0)),
            _Batch(
                {
                    "prompt_ids": ("prompt-a", "prompt-a"),
                    "rewards": (2.0, 4.0),
                    "loss_mask": ((1, "x"), (1, 1)),
                    "old_logprobs": ((-1.0, -1.0), (-1.0, -1.0)),
                    "reference_logprobs": ((-1.0, -1.0), (-1.0, -1.0)),
                }
            ),
        )
    message = str(exc_info.value)
    assert "mask entry at row 0, position 1 is 'x'" in message


def test_grpo_batch_refuses_mismatched_token_denominators() -> None:
    """WHAT IS CLAIMED: all five token denominators are reported when the
    forward row is wider than the rest.

    WHAT IS NOT CLAIMED: which of the five counts is authoritative; only
    their disagreement is measured.
    """
    loss_fn = GRPOPolicyLoss()
    with pytest.raises(
        BatchRefusal,
        match="all 5 token denominators must be equal",
    ) as exc_info:
        loss_fn.compute_with_report(
            lambda batch: ((-1.0, -1.0, -1.0), (-1.0, -1.0, -1.0)),
            _grpo_batch(),
        )
    message = str(exc_info.value)
    assert "row 0: mask/advantage/current/old/reference lengths" in message
    assert "(2, 2, 3, 2, 2)" in message


def test_grpo_batch_refuses_bool_advantage_weight() -> None:
    """WHAT IS CLAIMED: a bool weight is not a measured token weight.

    WHAT IS NOT CLAIMED: that the real estimator could emit one; this
    measures the binding's own price-book discipline over a scripted row.
    """
    loss_fn = GRPOPolicyLoss(
        advantage_fn=_ScriptedAdvantage(script_weights=((True, -1.0), (1.0, 1.0)))
    )
    with pytest.raises(
        BatchRefusal,
        match="a bool is not a measured token weight",
    ) as exc_info:
        loss_fn.compute_with_report(
            lambda batch: ((-1.0, -1.0), (-1.0, -1.0)),
            _grpo_batch(),
        )
    message = str(exc_info.value)
    assert "advantage weight at row 0, position 0 is True" in message


def test_grpo_batch_refuses_non_convertible_advantage_weight() -> None:
    """WHAT IS CLAIMED: a non-numeric weight is refused with its type.

    WHAT IS NOT CLAIMED: any coercion; the refusal names the str type.
    """
    loss_fn = GRPOPolicyLoss(
        advantage_fn=_ScriptedAdvantage(script_weights=(("x", -1.0), (1.0, 1.0)))
    )
    with pytest.raises(
        BatchRefusal,
        match="does not convert to a scalar float",
    ) as exc_info:
        loss_fn.compute_with_report(
            lambda batch: ((-1.0, -1.0), (-1.0, -1.0)),
            _grpo_batch(),
        )
    message = str(exc_info.value)
    assert "advantage weight at row 0, position 0 does not convert" in message
    assert "(str)" in message


def test_grpo_batch_refuses_non_finite_advantage_weight() -> None:
    """WHAT IS CLAIMED: a NaN weight cannot be priced as gradient.

    WHAT IS NOT CLAIMED: any amortisation of such a weight; the refusal is
    the only outcome measured.
    """
    loss_fn = GRPOPolicyLoss(
        advantage_fn=_ScriptedAdvantage(script_weights=((float("nan"), -1.0), (1.0, 1.0)))
    )
    with pytest.raises(
        BatchRefusal,
        match="a non-finite advantage as gradient",
    ) as exc_info:
        loss_fn.compute_with_report(
            lambda batch: ((-1.0, -1.0), (-1.0, -1.0)),
            _grpo_batch(),
        )
    message = str(exc_info.value)
    assert "advantage weight at row 0, position 0 is nan" in message


def test_grpo_batch_refuses_non_convertible_current_logprob() -> None:
    """WHAT IS CLAIMED: a non-numeric current reading names its column role.

    WHAT IS NOT CLAIMED: which component produced the bad string; only the
    token reading contract is measured.
    """
    loss_fn = GRPOPolicyLoss()
    with pytest.raises(
        BatchRefusal,
        match="does not convert to a scalar float",
    ) as exc_info:
        loss_fn.compute_with_report(
            lambda batch: (("x", -1.0), (-1.0, -1.0)),
            _grpo_batch(),
        )
    message = str(exc_info.value)
    assert "field current_logprobs at row 0, position 0 does not convert" in message
    assert "(str)" in message


def test_grpo_batch_refuses_non_finite_old_logprob() -> None:
    """WHAT IS CLAIMED: a NaN old log-probability is named with its field.

    WHAT IS NOT CLAIMED: where the NaN originated upstream; the token
    reading refuses before the ratio is even formed.
    """
    loss_fn = GRPOPolicyLoss()
    with pytest.raises(
        BatchRefusal,
        match="which is not finite",
    ) as exc_info:
        loss_fn.compute_with_report(
            lambda batch: ((-2.0, -1.0), (-1.0, -1.0)),
            _Batch(
                {
                    "prompt_ids": ("prompt-a", "prompt-a"),
                    "rewards": (2.0, 4.0),
                    "loss_mask": ((1, 1), (1, 1)),
                    "old_logprobs": ((float("nan"), -1.2), (-1.2, -1.2)),
                    "reference_logprobs": ((-2.2, -1.2), (-1.2, -1.2)),
                }
            ),
        )
    message = str(exc_info.value)
    assert "field old_logprobs at row 0, position 0 is nan" in message


def test_grpo_batch_refuses_overflowing_token_ratio() -> None:
    """WHAT IS CLAIMED: an unrepresentable ratio refuses before clipping.

    WHAT IS NOT CLAIMED: any clamping behaviour; the refusal names the
    exact log-ratio that overflowed.
    """
    loss_fn = GRPOPolicyLoss()
    with pytest.raises(
        BatchRefusal,
        match="is not representable",
    ) as exc_info:
        loss_fn.compute_with_report(
            lambda batch: ((1000.0, -1.0), (-1.0, -1.0)),
            _grpo_batch(),
        )
    message = str(exc_info.value)
    assert "row 0, position 0: token ratio exp(1002.2)" in message
    assert "1 of 1 token ratios failed before clipping" in message


def test_grpo_batch_refuses_underflowed_token_ratio() -> None:
    """WHAT IS CLAIMED: a ratio that underflows to 0.0 cannot manufacture a
    measured-looking zero contribution.

    WHAT IS NOT CLAIMED: that very small ratios above zero are refused;
    the boundary measured is exactly the non-positive one.
    """
    loss_fn = GRPOPolicyLoss()
    with pytest.raises(
        BatchRefusal,
        match="is not a positive finite value",
    ) as exc_info:
        loss_fn.compute_with_report(
            lambda batch: ((-1000.0, -1.0), (-1.0, -1.0)),
            _grpo_batch(),
        )
    message = str(exc_info.value)
    assert "row 0, position 0: token ratio 0.0" in message
    assert "silently underflowed ratio" in message


def test_grpo_batch_refuses_overflowing_k3_projection() -> None:
    """WHAT IS CLAIMED: an unrepresentable k3 projection refuses with the
    exact reference log-ratio.

    WHAT IS NOT CLAIMED: any clamping; the refusal is the only measured
    outcome of the overflow.
    """
    loss_fn = GRPOPolicyLoss()
    with pytest.raises(
        BatchRefusal,
        match="is not representable",
    ) as exc_info:
        loss_fn.compute_with_report(
            lambda batch: ((0.0, 0.0), (0.0, 0.0)),
            _Batch(
                {
                    "prompt_ids": ("prompt-a", "prompt-a"),
                    "rewards": (2.0, 4.0),
                    "loss_mask": ((1, 1), (1, 1)),
                    "old_logprobs": ((0.0, 0.0), (0.0, 0.0)),
                    "reference_logprobs": ((1000.0, -1.0), (-1.0, -1.0)),
                }
            ),
        )
    message = str(exc_info.value)
    assert "row 0, position 0: k3 projection exp(1000.0)" in message
    assert "1 of 1 reference-token readings failed" in message


def test_grpo_loss_refuses_zero_supervised_tokens() -> None:
    """WHAT IS CLAIMED: no supervised token means no measured loss, refused
    with the used-row count.

    WHAT IS NOT CLAIMED: any zero fallback; the refusal is what keeps an
    empty denominator from reporting a perfect loss.

    The scripted estimator sidesteps the real estimator's own empty-mask
    refusal so that the binding's last-ditch supervision refusal is the
    measured statement.
    """
    loss_fn = GRPOPolicyLoss(advantage_fn=_ScriptedAdvantage())
    with pytest.raises(
        SupervisionRefusal,
        match="the GRPO loss is unmeasured",
    ) as exc_info:
        loss_fn.compute_with_report(
            lambda batch: ((-1.0, -1.0), (-1.0, -1.0)),
            _Batch(
                {
                    "prompt_ids": ("prompt-a", "prompt-a"),
                    "rewards": (2.0, 4.0),
                    "loss_mask": ((0, 0), (0, 0)),
                    "old_logprobs": ((-1.0, -1.0), (-1.0, -1.0)),
                    "reference_logprobs": ((-1.0, -1.0), (-1.0, -1.0)),
                }
            ),
        )
    message = str(exc_info.value)
    assert "0 supervised tokens remain across 2 used rows" in message


def test_grpo_masked_position_carries_no_gradient_or_denominator() -> None:
    """A masked position is absent from both numerator and denominator.

    Hand computation: advantages are -1 and +1 broadcast over the row
    masks, so the supervised weights are (-1 at row 0 position 0) and
    (+1, +1 at row 1). With current == old == reference at every position
    the ratio is 1 (unclipped) and every k3 term is 0; the policy mean is
    (-1 + 1 + 1) / 3 supervised tokens.

    WHAT IS CLAIMED: the masked position contributes literally nothing --
    bool and integer mask entries are both read, its weight reads 0.0 by
    broadcast, and the supervised-token denominator is 3, not 4.

    WHAT IS NOT CLAIMED: that masking and a measured zero advantage are the
    same reading; they are distinct and remain so.
    """
    loss_fn = GRPOPolicyLoss()
    output, advantage = loss_fn.compute_with_report(
        lambda batch: ((-1.0, -1.0), (-1.0, -1.0)),
        _Batch(
            {
                "prompt_ids": ("prompt-a", "prompt-a"),
                "rewards": (2.0, 4.0),
                "loss_mask": ((True, 0), (1, 1)),
                "old_logprobs": ((-1.0, -1.0), (-1.0, -1.0)),
                "reference_logprobs": ((-1.0, -1.0), (-1.0, -1.0)),
            }
        ),
    )
    assert advantage.weights == ((-1.0, 0.0), (1.0, 1.0))
    assert advantage.rewards.count == 2
    assert output.components[0].contribution == pytest.approx(-1.0 / 3.0)
    assert output.components[1].contribution == pytest.approx(0.0)
    assert output.loss == pytest.approx(-1.0 / 3.0)


def test_grpo_declaration_mirrors_configured_component_names() -> None:
    """WHAT IS CLAIMED: the declaration carries exactly the configured
    component pair and no metrics.

    WHAT IS NOT CLAIMED: that the declared components will be observed; only
    the declared set itself is measured.
    """
    loss_fn = GRPOPolicyLoss(
        component_name="policy_objective",
        kl_component_name="reference_divergence",
    )
    declaration = loss_fn.declaration()
    assert declaration.components == ("policy_objective", "reference_divergence")
    assert declaration.metrics == ()


def test_grpo_loss_semantics_track_the_loss_configuration() -> None:
    """WHAT IS CLAIMED: the loss-side declaration takes its group size and
    clip bounds from the loss's own fields.

    WHAT IS NOT CLAIMED: that the algorithm agrees; agreement is the setup
    handshake, measured separately.
    """
    loss_fn = GRPOPolicyLoss(
        group_size=3,
        clip_low=0.5,
        clip_high=1.5,
        advantage_fn=GroupNormalisedAdvantage(min_group_size=3),
    )
    semantics = loss_fn.semantics()
    assert semantics.group_size == 3
    assert semantics.ratio_scope == "token"
    assert semantics.kl_estimator == "k3"
    assert semantics.clip_bounds == (0.5, 1.5)
    assert semantics.reference_free is False


def test_grpo_call_is_the_first_half_of_compute_with_report() -> None:
    """WHAT IS CLAIMED: ``__call__`` returns exactly the report's loss side,
    with components named as declared.

    WHAT IS NOT CLAIMED: that the advantage denominator is discarded beyond
    this surface; only the loss value and declared names are measured.
    """
    loss_fn = GRPOPolicyLoss()

    def forward(batch: Any) -> tuple[tuple[float, float], tuple[float, float]]:
        return ((-2.0, -1.0), (-1.0, -1.0))

    called = loss_fn(forward, _grpo_batch())
    reported, _advantage = loss_fn.compute_with_report(forward, _grpo_batch())
    assert called.loss == reported.loss
    assert called.components == reported.components
    names = tuple(component.name for component in called.components)
    assert names == loss_fn.declaration().components


def test_grpo_requirements_refuses_non_mapping_requires() -> None:
    """WHAT IS CLAIMED: the requires argument must itself be a Mapping.

    WHAT IS NOT CLAIMED: any duck-typed acceptance of Mapping-shaped data;
    the protocol check is on the container, not its contents.
    """
    with pytest.raises(
        AlgorithmWiringRefusal,
        match="must implement Mapping",
    ) as exc_info:
        check_grpo_requirements(
            requires=[("policy_pair", True)],
            supplied={"policy_pair": object()},
        )
    assert "field requires=" in str(exc_info.value)


def test_grpo_requirements_refuses_non_mapping_supplied() -> None:
    """WHAT IS CLAIMED: the supplied argument must itself be a Mapping.

    WHAT IS NOT CLAIMED: that a sequence of supplied names would even state
    absence for checks; only the container contract is measured.
    """
    with pytest.raises(
        AlgorithmWiringRefusal,
        match="must implement Mapping",
    ) as exc_info:
        check_grpo_requirements(
            requires={"policy_pair": True},
            supplied=["policy_pair"],
        )
    assert "field supplied=" in str(exc_info.value)


def test_grpo_requirements_refuses_unnamed_requires_roles() -> None:
    """WHAT IS CLAIMED: non-string or empty requires names are enumerated.

    WHAT IS NOT CLAIMED: which surrounding roles were valid; the refusal
    reports only the offending names.
    """
    with pytest.raises(
        AlgorithmWiringRefusal,
        match="names are not non-empty strings",
    ) as exc_info:
        check_grpo_requirements(
            requires={1: True, "": False},
            supplied={},
        )
    message = str(exc_info.value)
    assert "field requires: 2 of 2 names are not non-empty strings" in message
    assert "(1, '')" in message


def test_grpo_requirements_refuses_unnamed_supplied_roles() -> None:
    """WHAT IS CLAIMED: an integer supplied key is a bad name, not a role.

    WHAT IS NOT CLAIMED: whether the role it tried to name was required;
    naming is checked before requirement semantics.
    """
    with pytest.raises(
        AlgorithmWiringRefusal,
        match="names are not non-empty strings",
    ) as exc_info:
        check_grpo_requirements(
            requires={"policy_pair": True},
            supplied={"policy_pair": object(), 1: object()},
        )
    message = str(exc_info.value)
    assert "field supplied: 1 of 2 names are not non-empty strings" in message
    assert "(1,)" in message


def test_grpo_requirements_refuses_non_bool_requirement_values() -> None:
    """WHAT IS CLAIMED: requirement values are True or False, nothing else.

    WHAT IS NOT CLAIMED: any truthiness liberalisation; only exact bools
    state the requirement denominator.
    """
    with pytest.raises(
        AlgorithmWiringRefusal,
        match="values are not bool",
    ) as exc_info:
        check_grpo_requirements(
            requires={"policy_pair": "yes"},
            supplied={"policy_pair": object()},
        )
    message = str(exc_info.value)
    assert "field requires: 1 of 1 requirement values are not bool" in message
    assert "('policy_pair',)" in message


def test_grpo_requirements_refuses_empty_role_union() -> None:
    """WHAT IS CLAIMED: an empty union passes nothing by measuring nothing.

    WHAT IS NOT CLAIMED: that any particular role belongs to grpo; this
    measures only the vacuous-check state.
    """
    with pytest.raises(
        AlgorithmWiringRefusal,
        match="0 roles appear in either map",
    ) as exc_info:
        check_grpo_requirements(requires={}, supplied={})
    message = str(exc_info.value)
    assert "for grpo" in message
    assert "pass vacuously by measuring nothing" in message


def test_grpo_requirements_refuses_all_false_requirements() -> None:
    """WHAT IS CLAIMED: a requirement map declaring every role unrequired
    is refused as vacuous.

    WHAT IS NOT CLAIMED: that unrequired roles are ungraded; a False role
    is still present in the denominator, as the count shows.
    """
    with pytest.raises(
        AlgorithmWiringRefusal,
        match="an empty required-set is refused as vacuous",
    ) as exc_info:
        check_grpo_requirements(
            requires={"rollout_source": False, "weight_sync": False},
            supplied={},
        )
    message = str(exc_info.value)
    assert "field requires: 0 of 2 declared roles are required for grpo" in message


def test_grpo_requirements_refuses_unrequired_supplied_roles() -> None:
    """WHAT IS CLAIMED: a supplied-but-unrequired role names the overflow
    and the full checked denominator.

    WHAT IS NOT CLAIMED: that the extra object was invalid; presence
    outside the requirement map is the violation, not the object.
    """
    with pytest.raises(
        AlgorithmWiringRefusal,
        match="are not required by grpo",
    ) as exc_info:
        check_grpo_requirements(
            requires={"policy_pair": True},
            supplied={"policy_pair": object(), "telemetry": object()},
        )
    message = str(exc_info.value)
    assert "field supplied: 1 of 2 supplied inputs are not required" in message
    assert "('telemetry',)" in message
    assert "2 roles from 1 requires keys and 2 supplied keys" in message


def test_grpo_requirements_refuses_boolean_presence() -> None:
    """WHAT IS CLAIMED: a boolean supplied value is neither presence nor
    absence and refuses with the boolean named.

    WHAT IS NOT CLAIMED: that False reads as absence; absence is no key or
    None, never False.
    """
    with pytest.raises(
        AlgorithmWiringRefusal,
        match="not supplied components",
    ) as exc_info:
        check_grpo_requirements(
            requires={"policy_pair": True},
            supplied={"policy_pair": False},
        )
    message = str(exc_info.value)
    assert "field supplied contains the boolean False" in message
    assert "absence must be represented by no key or by None" in message


def test_grpo_requirements_none_is_absence_and_success_is_sorted() -> None:
    """WHAT IS CLAIMED: an explicit None reads as absence with the role
    named, while a fully consistent wiring returns the sorted consumed set.

    WHAT IS NOT CLAIMED: that the consumed set equals the offered key set
    in general; this wiring's unrequired role is simply not offered.
    """
    with pytest.raises(
        AlgorithmWiringRefusal,
        match="required inputs absent",
    ) as exc_info:
        check_grpo_requirements(
            requires={"policy_pair": True},
            supplied={"policy_pair": None},
        )
    message = str(exc_info.value)
    assert "field supplied: 1 of 1 required inputs absent for grpo" in message
    assert "('policy_pair',)" in message

    roles = check_grpo_requirements(
        requires={"loss_fn": True, "dataloader": True, "rollout_source": False},
        supplied={"dataloader": object(), "loss_fn": object()},
    )
    assert roles == ("dataloader", "loss_fn")
    assert tuple(sorted(roles)) == roles


def test_grpo_setup_refuses_already_wired_algorithm() -> None:
    """WHAT IS CLAIMED: a second setup cannot silently replace the measured
    supplied mapping.

    WHAT IS NOT CLAIMED: that the new arguments were otherwise valid; the
    wiring-state check precedes every other setup check.
    """
    algorithm = GRPOAlgorithm()
    loss_fn = GRPOPolicyLoss()
    algorithm._wiring = _Wiring(
        loss_fn=loss_fn,
        dataloader=iter(()),
        policy_logprob_column="current_logprobs",
    )
    with pytest.raises(
        AlgorithmWiringRefusal,
        match="already wired",
    ) as exc_info:
        algorithm.setup(
            policy_pair=object(),
            loss_fn=loss_fn,
            dataloader=iter(()),
            config={},
            advantage_fn=loss_fn.advantage_fn,
        )
    message = str(exc_info.value)
    assert "1 of 1 GRPO algorithm objects was already wired" in message
    assert "replace the measured supplied mapping" in message


def test_grpo_setup_refuses_non_grpo_loss() -> None:
    """WHAT IS CLAIMED: policy-as-grpo loss substitutes are refused, with
    the received type named.

    WHAT IS NOT CLAIMED: that the substitute computes anything sensible;
    only the slot type is measured.
    """
    with pytest.raises(
        AlgorithmWiringRefusal,
        match="GRPO requires GRPOPolicyLoss",
    ) as exc_info:
        GRPOAlgorithm().setup(
            policy_pair=object(),
            loss_fn=object(),
            dataloader=iter(()),
            config={},
            advantage_fn=GroupNormalisedAdvantage(),
        )
    message = str(exc_info.value)
    assert "field loss_fn=" in message
    assert "1 of 1 loss slots carries object" in message


def test_grpo_setup_refuses_absent_advantage_role() -> None:
    """WHAT IS CLAIMED: the estimator role must be present at setup.

    WHAT IS NOT CLAIMED: that the loss carries no estimator of its own; the
    check measures that it was not handed through both sides of the wiring.
    """
    loss_fn = GRPOPolicyLoss()
    with pytest.raises(
        AlgorithmWiringRefusal,
        match="field advantage_fn",
    ) as exc_info:
        GRPOAlgorithm().setup(
            policy_pair=object(),
            loss_fn=loss_fn,
            dataloader=iter(()),
            config={},
            advantage_fn=None,
        )
    message = str(exc_info.value)
    assert "1 of 1 required inputs absent for grpo" in message
    assert "{'advantage_fn'}" in message


def test_grpo_setup_refuses_distinct_advantage_instances() -> None:
    """WHAT IS CLAIMED: the setup object must be the loss's own estimator,
    not a chemically identical second instance.

    WHAT IS NOT CLAIMED: that the second estimator's declaration disagreed;
    identity, not equality, is the measured requirement.
    """
    loss_fn = GRPOPolicyLoss()
    with pytest.raises(
        AlgorithmWiringRefusal,
        match="2 distinct objects for 1 declared role",
    ) as exc_info:
        GRPOAlgorithm().setup(
            policy_pair=object(),
            loss_fn=loss_fn,
            dataloader=iter(()),
            config={},
            advantage_fn=GroupNormalisedAdvantage(),
        )
    message = str(exc_info.value)
    assert "must wire the same GroupNormalisedAdvantage instance" in message


def test_grpo_setup_refuses_semantics_disagreement() -> None:
    """WHAT IS CLAIMED: a loss whose independent declaration disagrees with
    the algorithm's is refused with the disagreeing field named.

    WHAT IS NOT CLAIMED: that the algorithm's declaration is universally
    correct; only the disagreement between the two records is measured.
    """
    mismatched_loss = GRPOPolicyLoss()
    with pytest.raises(
        AlgorithmWiringRefusal,
        match="1 of 5 semantics fields disagrees",
    ) as exc_clip:
        GRPOAlgorithm(clip_low=0.5, clip_high=0.9).setup(
            policy_pair=object(),
            loss_fn=mismatched_loss,
            dataloader=iter(()),
            config={},
            advantage_fn=mismatched_loss.advantage_fn,
        )
    clip_message = str(exc_clip.value)
    assert "field clip_bounds" in clip_message
    assert "algorithm declares (0.5, 0.9) but loss declares (0.8, 1.2)" in clip_message

    larger_loss = GRPOPolicyLoss(
        group_size=3,
        advantage_fn=GroupNormalisedAdvantage(min_group_size=3),
    )
    with pytest.raises(
        AlgorithmWiringRefusal,
        match="the 2 declarations",
    ) as exc_group:
        GRPOAlgorithm().setup(
            policy_pair=object(),
            loss_fn=larger_loss,
            dataloader=iter(()),
            config={},
            advantage_fn=larger_loss.advantage_fn,
        )
    group_message = str(exc_group.value)
    assert "field group_size" in group_message
    assert "algorithm declares 2 but loss declares 3" in group_message


def test_grpo_step_refuses_before_setup_completes() -> None:
    """WHAT IS CLAIMED: pre-setup, step measures absence rather than guessing.

    WHAT IS NOT CLAIMED: any partial wiring state; only the unmeasured
    pre-setup state is asserted.
    """
    with pytest.raises(
        AlgorithmWiringRefusal,
        match="setup has not completed",
    ) as exc_info:
        GRPOAlgorithm().step()
    message = str(exc_info.value)
    assert "0 of 5 required inputs are present for grpo" in message


def test_grpo_step_prices_configured_column_and_advances_step() -> None:
    """A wired step prices the configured column and counts its own steps.

    Hand computation: with current readings 0.2 above both old and
    reference rows, every log-ratio is 0.2 and every reference log-ratio is
    -0.2, matching the hand-computed policy and k3 geometry of the
    end-to-end fixture, routed through the ``policy_logprob_column``.

    WHAT IS CLAIMED: two successive steps report step indices 0 and 1, the
    step's rows equal the estimator's used count, and a third step over an
    exhausted dataloader refuses rather than reporting zero rows.

    WHAT IS NOT CLAIMED: any rollout freshness or on-policy property; the
    binding measures batch pricing, not origin.
    """
    batch = _Batch(
        {
            "prompt_ids": ("prompt-a", "prompt-a"),
            "rewards": (2.0, 4.0),
            "loss_mask": ((1, 1), (1, 1)),
            "old_logprobs": ((-2.0, -1.0), (-1.0, -1.0)),
            "reference_logprobs": ((-2.0, -1.0), (-1.0, -1.0)),
            "current_logprobs": ((-1.8, -0.8), (-0.8, -0.8)),
        }
    )
    algorithm = GRPOAlgorithm()
    algorithm._wiring = _Wiring(
        loss_fn=GRPOPolicyLoss(),
        dataloader=iter((batch, batch)),
        policy_logprob_column="current_logprobs",
    )

    ratio = math.exp(0.2)
    expected_policy = -((2.0 * min(-ratio, -1.2) + 2.0 * min(ratio, 1.2)) / 4.0)
    expected_kl = 0.04 * (math.expm1(-0.2) + 0.2)

    first = algorithm.step()
    assert first.step == 0
    assert first.rows == 2
    assert first.reward_stats.count == 2
    assert first.reward_stats.mean == pytest.approx(3.0)
    assert first.sync is None
    assert first.loss.loss == pytest.approx(expected_policy + expected_kl)
    assert first.loss.components[0].name == "grpo_policy_loss"
    assert first.loss.components[1].contribution == pytest.approx(expected_kl)

    second = algorithm.step()
    assert second.step == 1
    assert second.loss.loss == pytest.approx(expected_policy + expected_kl)

    with pytest.raises(
        StepReportRefusal,
        match="unmeasured rather than a step with zero rows",
    ) as exc_info:
        algorithm.step()
    message = str(exc_info.value)
    assert "0 of at least 1 required batches remain for grpo" in message
