"""The group-relative policy binding: one parameterised algorithm, three objectives.

``policy_gradient.py`` writes one algorithm class per loss because each
policy-gradient family genuinely fixes different semantic seams. The three
group-relative policy objectives bound here -- GSPO, Dr.GRPO and DAPO -- do
not. The measured difference between docs/RL_ALGORITHMS.md's critic-free row
(:125, Dr.GRPO and DAPO) and its sequence-level row (:129, GSPO) is exactly
two declarations -- ``ratio_scope`` and ``reduction`` -- plus the objective's
own advantage estimator, clip bounds and KL stance. One parameterised
:class:`SequencePolicyAlgorithm` therefore binds all three through one seam,
and three thin factories construct it with the right objective for
``registry.py``: ``gspo_algorithm()``, ``dr_grpo_algorithm()`` and
``dapo_algorithm()``. ``group_policy`` names the invariant all three share: a
group-relative advantage over prompt groups with a clipped importance ratio.

THE DENOMINATOR SEAM, stated once. ``reduction``
(``token_mean``/``sequence_mean``/``constant``) is independent of the ratio
geometry, and it is what actually separates GSPO from Dr.GRPO from DAPO, so
it rides on the objective's own declaration and is never restated here. Per
docs/RL_ALGORITHMS.md's open question (:137), ``ratio_scope`` keeps the
closed union ``Literal["token", "sequence"]``: a sequence ratio that is NOT
length-normalised is ``exp(SUM_t log_ratio_t)``, which over a response of
several hundred tokens leaves any usable floating-point range for any
non-trivial policy shift, so it is not a geometry anything binds. Length
normalisation is not an option on a sequence ratio; it is what makes one
exist. That is an ARGUMENT, not a measurement.

WHAT THIS MODULE CLAIMS: a wired group-relative policy algorithm prices
exactly one caller-supplied batch per step through the objective it was
constructed with, declares the rollout-owning role map, derives its
semantics' ratio scope, clip bounds and reference stance FROM the objective
instance so the two cannot drift, and reports ``rows`` as the batch's row
count -- one row being one sampled response in a prompt group.

WHAT THIS MODULE DOES NOT CLAIM: that DAPO's dynamic sampling or overlong
reward shaping is implemented anywhere here -- the REFILL half of dynamic
sampling is a rollout-loop behaviour, and this package has no rollout loop to
put it in, so ``expects_dynamic_sampling`` is a declared and deliberately
UNENFORCED expectation carried by the objective; that any model forward, any
optimiser, or any producer of the consumed batch columns exists here; that
the five-field semantics cross-check can fire -- both sides derive from the
one objective, so it is a maintenance tripwire; that a rollout actually
occurred -- the rollout source role is declared and graded for presence
only; or that the loss arithmetic is restated here -- the objectives own it.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Literal, Protocol, runtime_checkable

from foundationscale.rl.advantage import AdvantageFn
from foundationscale.rl.algorithm import (
    AlgorithmRequirements,
    AlgorithmSemantics,
    AlgorithmWiringRefusal,
    StepReport,
    StepReportRefusal,
    check_algorithm_wiring,
    check_role_map,
)
from foundationscale.rl.group_policy_objectives import (
    DAPOLoss,
    DrGRPOLoss,
    GSPOLoss,
)
from foundationscale.rl.interfaces import (
    BatchRefusal,
    ExperienceBatch,
    LossFn,
)
from foundationscale.rl.policy import PolicyPair
from foundationscale.rl.rollout import RolloutSource
from foundationscale.rl.weightsync import WeightSync

__all__ = (
    "SequenceObjective",
    "SequencePolicyAlgorithm",
    "check_group_policy_requirements",
    "dapo_algorithm",
    "dr_grpo_algorithm",
    "gspo_algorithm",
)


@runtime_checkable
class SequenceObjective(LossFn, Protocol):
    """The structural contract an objective must satisfy to be bound here.

    WHAT IS CLAIMED: a satisfying object exposes the members this binding
    actually reads -- ``advantage_fn``, ``clip_bounds``, ``ratio_scope`` and
    ``required_columns`` to derive the wiring, the semantics and the batch
    schema, ``semantics()`` for the five-field cross-check, and
    ``expects_dynamic_sampling`` so a rollout-half expectation stays visible
    rather than silently absent -- in addition to everything
    :class:`LossFn` already demands of any loss.

    This is a protocol rather than a union of the three shipped objective
    classes because a union is a CLOSED set: a fourth group-relative
    objective could not be bound without editing this module, which is the
    opposite of what a seam is for.

    WHAT IS NOT CLAIMED: that a satisfying object computes a group-relative
    policy objective. ``@runtime_checkable`` measures attribute PRESENCE
    only, never signatures and never meaning, so the ``isinstance`` check in
    :meth:`SequencePolicyAlgorithm.__init__` is a structural smoke test and
    not a handshake. What makes a binding safe is not that check but the
    schema and semantics measurements taken afterwards from the object's own
    declarations.
    """

    def semantics(self) -> AlgorithmSemantics: ...

    @property
    def advantage_fn(self) -> AdvantageFn: ...

    @property
    def clip_bounds(self) -> tuple[float, float]: ...

    @property
    def expects_dynamic_sampling(self) -> bool: ...

    @property
    def ratio_scope(self) -> Literal["token", "sequence"]: ...

    @property
    def reduction(self) -> Literal["token_mean", "sequence_mean", "constant"]: ...

    @property
    def required_columns(self) -> tuple[str, ...]: ...


# The members :class:`SequenceObjective` demands, stated here rather than
# read from its ``__protocol_attrs__``: that attribute exists only on Python
# 3.12+, and this package supports 3.10, so deriving the diagnostic from it
# would replace a clean refusal with an AttributeError on the two oldest
# supported interpreters. Stating a set twice is only safe when something
# measures both statements independently, so the protocol test module drives
# the protocol's REAL member set through ``isinstance`` -- adding a member
# here that the protocol does not demand, or omitting one it does, fails
# there, with each direction caught by a DIFFERENT leg.
_OBJECTIVE_MEMBERS: tuple[str, ...] = (
    "__call__",
    "advantage_fn",
    "clip_bounds",
    "declaration",
    "expects_dynamic_sampling",
    "ratio_scope",
    "reduction",
    "required_columns",
    "semantics",
)


def _absent_members(objective: object) -> tuple[str, ...]:
    """Name the objective members ``objective`` does not expose, in sorted order."""
    return tuple(name for name in _OBJECTIVE_MEMBERS if not hasattr(objective, name))


def check_group_policy_requirements(
    *,
    requires: Mapping[str, bool],
    supplied: Mapping[str, Any],
    origin: str = "group_policy",
) -> tuple[str, ...]:
    """Check the group-relative policy family's role map over both key sets.

    WHAT IS CLAIMED on success: every required role names a supplied object,
    no required set was emptily satisfied, no supplied required-role value is
    ``None``, and the returned tuple is the sorted set of roles actually
    consumed.

    WHAT IS NOT CLAIMED: that any role object is a valid model, rollout
    source, estimator, or loss. Type and semantic checks remain with the
    component handshakes; this helper checks the declared denominator only.
    The implementation is shared with every other family's role-map check --
    only the ``origin`` this binding names differs.
    """
    return check_role_map(requires=requires, supplied=supplied, origin=origin)


@dataclass(frozen=True, slots=True)
class _SequencePolicyWiring:
    loss_fn: SequenceObjective
    dataloader: Iterator[ExperienceBatch]
    policy_logprob_columns: tuple[str, ...]
    required_columns: tuple[str, ...]


class SequencePolicyAlgorithm:
    """Concrete rollout-owning group-relative policy algorithm bound to one objective.

    The family is uniform in a way the wider policy-gradient plane is not:
    each of the three objectives consumes a rollout source and a
    group-relative advantage function, none consumes a weight sync, and none
    calls a reference model by default -- Dr.GRPO's optional KL, when
    enabled, is the objective's own configuration, not this binding's. One
    parameterised binding therefore carries the objective instance it was
    constructed with and derives its ratio scope, clip bounds and reference
    stance FROM that instance rather than restating them.

    ``rows`` in the step report is the batch's row count, meaning ONE THING
    across all three objectives: one row is one sampled response in a prompt
    group, one unit of group-relative supervision. Whether the objective
    reduces over tokens or over whole sequences is the objective's own
    ``reduction`` declaration and changes no unit in this report.

    WHAT IS CLAIMED: one successful step invokes the carried objective on
    the next dataloader batch through a forward function that reads the
    config-named policy log-probability columns, and returns a
    ``StepReport`` whose ``loss`` is the objective's own ``LossOutput`` and
    whose ``rows`` is the batch length.

    WHAT IS NOT CLAIMED: that a rollout actually ran this step -- the
    rollout source is declared and graded for presence at setup, and no
    batch generation is performed here; that DAPO's dynamic refill happened
    -- that is a rollout-loop behaviour this package has no rollout loop to
    hold, and the objective's ``expects_dynamic_sampling`` declaration is
    visible but enforced by nothing in this module; that the dataloader rows
    are fresh or correctly grouped; that another batch exists after any
    step; or that the loss arithmetic is re-implemented here.
    """

    __slots__ = (
        "_name",
        "_next_step",
        "_objective",
        "_requirements",
        "_requires",
        "_semantics",
        "_supplied",
        "_wiring",
    )

    def __init__(self, *, name: str, objective: SequenceObjective) -> None:
        """Construct the binding's declarations FROM the carried objective.

        The three factories are the ordinary callers; direct construction is
        the seam for objectives whose column or metric names were customised
        beyond the factories' knob surface. ``name`` is the one statement
        that cannot be derived from the objective -- the losses carry no
        family name of their own, and inventing one from the type would be a
        restated countable.

        WHAT IS CLAIMED: an object outside the protocol is refused at
        construction with the missing members named; the resulting
        declarations take ``ratio_scope`` and ``clip_bounds`` from the
        objective's own properties, take ``reference_free`` from the
        objective's own semantics declaration, declare the rollout source
        and advantage function consumed, declare weight sync and -- exactly
        when the objective is reference-free -- the reference policy
        unconsumed with False; and read the declared components and metric
        names from the objective's own ``declaration()``.

        WHAT IS NOT CLAIMED: that the derived declarations were compared
        against a second statement -- the objective IS the single source,
        and the five-field cross-check in ``setup`` is a construction-time
        tripwire, not a live seam.
        """
        # The branch below is a SINGLE raise on purpose, mirroring
        # preference.py: mypy narrows the annotated parameter to Never here
        # and reports any other statement as unreachable -- correctly, for a
        # typed caller. The untyped callers are the real ones: the registry
        # constructs through a factory, and nothing static stands between a
        # caller's object and this slot.
        if not isinstance(objective, SequenceObjective):
            raise AlgorithmWiringRefusal(
                f"objective={objective!r}: 1 of 1 objective slots must carry "
                f"a SequenceObjective; {type(objective).__name__} does not "
                f"expose {_absent_members(objective)} "
                f"of the {len(_OBJECTIVE_MEMBERS)} "
                f"required members -- the group-relative policy binding "
                f"cannot price an objective whose batch shape it cannot read"
            )
        self._name = name
        self._objective = objective
        # Every semantics field is DERIVED from the objective's own
        # declarations, never restated by hand: ratio scope and clip bounds
        # from the objective's properties, reference-freeness from the
        # objective's own semantics() statement. group_size abstains with
        # None because the objective declares no group size to this seam,
        # and a number there would be a measured value posing as an
        # unconstrained seam; kl_estimator is likewise the objective's own
        # statement (None where the family carries no KL).
        objective_semantics = objective.semantics()
        self._semantics = AlgorithmSemantics(
            group_size=objective_semantics.group_size,
            ratio_scope=objective.ratio_scope,
            kl_estimator=objective_semantics.kl_estimator,
            clip_bounds=objective.clip_bounds,
            reference_free=objective_semantics.reference_free,
        )
        declaration = objective.declaration()
        reference_free = objective_semantics.reference_free is not False
        self._requirements = AlgorithmRequirements(
            name=name,
            # Every role this surface grades is named, including the ones it
            # does NOT consume: weight_sync is False throughout the family,
            # and reference_policy is False exactly when the objective is
            # reference-free -- an unconsumed frozen model is a stub that
            # reads as a working wiring. A role declared False is graded,
            # and a supplied-but-unconsumed one is refused, never silently
            # ignored.
            requires={
                "rollout_source": True,
                "advantage_fn": True,
                "weight_sync": False,
                "reference_policy": not reference_free,
            },
            declared_components=declaration.components,
            declared_metrics=tuple(metric.name for metric in declaration.metrics),
            semantics=self._semantics,
        )
        self._requires: Mapping[str, bool] = MappingProxyType(
            {
                "policy_pair": True,
                "loss_fn": True,
                "dataloader": True,
                "advantage_fn": True,
                "reference_policy": False,
                "rollout_source": True,
                "weight_sync": False,
            }
        )
        self._supplied: Mapping[str, Any] | None = None
        self._wiring: _SequencePolicyWiring | None = None
        self._next_step = 0

    def requirements(self) -> AlgorithmRequirements:
        """Return the role and objective declaration measured by gates.

        WHAT IS CLAIMED: the record is this binding's declaration: the
        rollout-owning role map with every unconsumed role named False, and
        the component and metric denominators read off the carried
        objective's own declaration.

        WHAT IS NOT CLAIMED: that the declaration is sufficient evidence the
        mathematics are correct; it is the denominator the gates measure.
        """
        return self._requirements

    def semantics(self) -> AlgorithmSemantics:
        """Return this binding's algorithm-side semantics declaration.

        WHAT IS CLAIMED: ``ratio_scope`` and ``clip_bounds`` came from the
        objective's own properties, and ``kl_estimator``, ``group_size`` and
        ``reference_free`` came from the objective's own semantics
        declaration -- nothing here is restated by hand.

        WHAT IS NOT CLAIMED: that algorithm and loss currently agree as an
        achievement; agreement holds because both statements derive from the
        one objective, and ``setup`` still measures it as a tripwire.
        """
        return self._semantics

    def requires(self) -> Mapping[str, bool]:
        """Return the static role-requirement denominator.

        WHAT IS CLAIMED: every entry is an immutable string-to-bool
        declaration, and the mapping contains at least one required role.

        WHAT IS NOT CLAIMED: that required roles are currently present;
        presence is represented only by ``supplied()`` after setup.
        """
        return self._requires

    def supplied(self) -> Mapping[str, Any] | None:
        """Return the actual post-setup wiring, or abstain before setup.

        WHAT IS CLAIMED: after successful setup, the immutable mapping names
        exactly the objects handed to this algorithm during that setup --
        the pair, the objective, the dataloader, the rollout source and the
        advantage function, and nothing else, because the family consumes
        nothing else.

        WHAT IS NOT CLAIMED before setup: any wiring measurement. The
        pre-setup value is ``None`` because the supplied set was not
        measured; an empty map would falsely read as a completed check.
        """
        return self._supplied

    def setup(
        self,
        *,
        policy_pair: PolicyPair,
        loss_fn: Any,
        dataloader: Iterable[ExperienceBatch],
        config: Mapping[str, Any],
        rollout_source: RolloutSource | None = None,
        advantage_fn: AdvantageFn | None = None,
        weight_sync: WeightSync | None = None,
    ) -> None:
        """Wire the binding and perform its declaration-to-loss handshake.

        ``loss_fn`` must BE the objective this algorithm was constructed
        with -- identity, not equality; two objective instances that agree
        everywhere still double the one declared objective role. A supplied
        ``advantage_fn`` must BE the objective's own ``advantage_fn`` -- the
        GRPO precedent, and the check that stops two estimators
        disagreeing about what an advantage is. ``config`` must name the
        policy log-probability column the forward function will read:
        ``policy_logprob_column``.

        setup peeks NO batch. All schema enforcement is per-step: peeking
        one batch to schema-check it early would make setup data-dependent
        -- a loader that blocks or raises on its first item would report a
        wiring fault for a data fact -- and it would buy nothing, because
        step re-checks the declared columns on EVERY batch and so already
        refuses the first one.

        WHAT IS CLAIMED on success: the handed loss is the carried
        objective; the handed advantage function, when one was handed, is
        the objective's own; the five-field semantics cross-check between
        this algorithm's record and the loss's own ``semantics()`` passes
        over exactly ``("group_size", "ratio_scope", "kl_estimator",
        "clip_bounds", "reference_free")``; the existing wiring handshake
        accepted the rollout-owning role map, including refusing any
        reference model the pair carries when the objective is
        reference-free; the dataloader was iterable; the config named a
        non-empty policy log-probability column; and the two independent
        role denominators name the same consumed set.

        WHAT IS NOT CLAIMED: that any batch was inspected -- schema is
        checked per step; that an empty dataloader is refused here -- an
        absent batch is a data fact and the step-time refusal keeps it at
        call time; that the columns carry well-typed values, which is the
        objective's own per-step measurement; or that a rollout was
        generated -- the rollout source's presence is graded, its product
        is not consumed here.
        """
        if self._wiring is not None:
            raise AlgorithmWiringRefusal(
                f"field setup: 1 of 1 {self._name} algorithm objects was "
                f"already wired; calling setup twice would silently replace "
                f"the measured supplied mapping"
            )
        if loss_fn is not self._objective:
            raise AlgorithmWiringRefusal(
                f"field loss_fn={loss_fn!r}: the setup object and the "
                f"algorithm-carried objective are 2 distinct objects for 1 "
                f"declared objective role; the {self._name} binding must "
                f"wire the same {type(self._objective).__name__} instance "
                f"through both sides"
            )
        if advantage_fn is not None and advantage_fn is not self._objective.advantage_fn:
            raise AlgorithmWiringRefusal(
                f"field advantage_fn={advantage_fn!r}: the setup object and "
                f"the objective-carried advantage function are 2 distinct "
                f"objects for 1 declared estimator role; the {self._name} "
                f"binding must wire the same advantage estimator through "
                f"both sides, because two estimators that disagree about "
                f"what an advantage is cannot both be right"
            )
        # The five-field cross-check is a MAINTENANCE tripwire, not a live
        # seam: both records derive from the one objective instance, so no
        # construction the public API permits can make them disagree. It is
        # run, and required, because an EDIT that stops deriving either side
        # from the objective arrives through the source, not through a caller.
        loss_semantics = loss_fn.semantics()
        for field_name in (
            "group_size",
            "ratio_scope",
            "kl_estimator",
            "clip_bounds",
            "reference_free",
        ):
            algorithm_value = getattr(self._semantics, field_name)
            loss_value = getattr(loss_semantics, field_name)
            if algorithm_value != loss_value:  # pragma: no cover -- derived from one objective
                raise AlgorithmWiringRefusal(
                    f"field {field_name}: algorithm declares "
                    f"{algorithm_value!r} but loss declares {loss_value!r}; "
                    f"1 of 5 semantics fields disagrees between the 2 "
                    f"declarations"
                )
        consumed = check_algorithm_wiring(
            self._requirements,
            policy_pair=policy_pair,
            loss_fn=loss_fn,
            # reference_policy is deliberately absent from the supplied
            # mapping: the pair attests it -- and with a reference-free
            # objective declaring it False, a pair carrying a frozen
            # reference is refused, because an unconsumed frozen model is a
            # stub that reads as a working wiring.
            supplied={
                "rollout_source": rollout_source,
                "advantage_fn": advantage_fn
                if advantage_fn is not None
                else self._objective.advantage_fn,
                "weight_sync": weight_sync,
            },
            origin="SequencePolicyAlgorithm.setup",
        )
        try:
            dataloader_iterator = iter(dataloader)
        except TypeError as exc:
            raise AlgorithmWiringRefusal(
                f"field dataloader={dataloader!r}: 1 of 1 mandatory "
                f"step inputs was not iterable "
                f"({type(dataloader).__name__})"
            ) from exc
        if not isinstance(config, Mapping):
            raise AlgorithmWiringRefusal(
                f"field config={config!r}: 1 of 1 setup configurations must implement Mapping"
            )
        policy_columns: list[str] = []
        for key in ("policy_logprob_column",):
            value = config.get(key)
            if not isinstance(value, str) or not value:
                raise AlgorithmWiringRefusal(
                    f"field {key}: 0 of 1 required config values names a "
                    f"non-empty batch column for {self._name}; current-policy "
                    f"token log-probabilities cannot be attributed to the "
                    f"existing LossFn surface without that column"
                )
            policy_columns.append(value)
        # required_columns is a PROPERTY on every objective, read with no
        # call parentheses. The declared schema is RECORDED here and
        # measured in step, never measured here: setup is the wiring
        # handshake and touches no data, and step re-checks these same
        # columns on EVERY batch, so a peek here would buy nothing and
        # would make setup data-dependent.
        required_columns = tuple(loss_fn.required_columns)
        supplied = MappingProxyType(
            {
                "policy_pair": policy_pair,
                "loss_fn": loss_fn,
                "dataloader": dataloader,
                "rollout_source": rollout_source,
                "advantage_fn": advantage_fn
                if advantage_fn is not None
                else self._objective.advantage_fn,
            }
        )
        mapped_consumed = check_group_policy_requirements(
            requires=self._requires,
            supplied=supplied,
            origin=self._name,
        )
        handshake_roles = tuple(sorted(consumed + ("policy_pair", "loss_fn", "dataloader")))
        # This comparison is a MAINTENANCE tripwire, not a runtime one: the
        # two denominators are literal mappings written side by side in
        # __init__, and both supplied maps are built here from the same
        # validated arguments, so no public construction can make them
        # disagree. What it catches is an EDIT that changes one mirror and
        # not the other -- one countable declared in two objects.
        if mapped_consumed != handshake_roles:  # pragma: no cover -- mirrors agree by construction
            raise AlgorithmWiringRefusal(
                f"mapped roles {mapped_consumed!r} disagree with the "
                f"existing handshake roles {consumed!r} plus the 3 "
                f"mandatory setup roles; 2 independently checked role "
                f"denominators must name the same consumed set"
            )
        self._supplied = supplied
        self._wiring = _SequencePolicyWiring(
            loss_fn=loss_fn,
            dataloader=dataloader_iterator,
            policy_logprob_columns=tuple(policy_columns),
            required_columns=required_columns,
        )

    def step(self) -> StepReport:
        """Price the next batch through the carried objective and report rows.

        ``rows`` is the batch's row count, and it means the same thing for
        all three objectives: one row is one sampled response in a prompt
        group -- one unit of group-relative supervision. Which reduction
        the objective applies inside the loss is its own declaration and
        changes no unit in this report. ``reward_stats`` and ``sync``
        abstain as ``None`` here because the advantage estimator's own
        result record -- advantage.py's ``AdvantageResult.rows`` -- names
        the surviving samples, and a weight-sync step is a cadence fact the
        family never owns.

        WHAT IS CLAIMED: the batch's columns cover the wiring's declared
        schema before the objective is invoked, the forward function reads
        exactly the config-named policy column, and the report carries the
        objective's own ``LossOutput`` with ``rows`` equal to the batch
        length.

        WHAT IS NOT CLAIMED: that a rollout ran this step; that another
        batch exists after this one; that any advantage was recomputed here
        rather than by the objective it was wired to; or that the result is
        finite.
        """
        wiring = self._wiring
        if wiring is None:
            raise AlgorithmWiringRefusal(
                f"field supplied: 0 of 3 required inputs are present for "
                f"{self._name} because setup has not completed"
            )
        try:
            batch = next(wiring.dataloader)
        except StopIteration as exc:
            raise StepReportRefusal(
                f"0 of at least 1 required batches remain for {self._name}; "
                f"the step is unmeasured rather than a step with zero rows"
            ) from exc
        needed_columns = wiring.required_columns + wiring.policy_logprob_columns
        missing = tuple(name for name in needed_columns if name not in batch.columns)
        if missing:
            raise BatchRefusal(
                f"step {self._next_step} of {self._name}: {len(missing)} of "
                f"{len(needed_columns)} required batch columns are absent "
                f"({', '.join(missing)}); this batch carries "
                f"{tuple(batch.columns)} -- a mid-stream batch must carry "
                f"the schema the setup handshake measured, because a column "
                f"present at setup and absent now is a changed denominator"
            )
        # The plane's own ForwardFn seam (Callable[[ExperienceBatch], Any]);
        # the objective, not this binding, owns what the returned scores
        # mean -- token log-probabilities under the current policy.
        column = wiring.policy_logprob_columns[0]

        def forward_fn(current_batch: ExperienceBatch) -> Any:
            return current_batch.column(column)

        output = wiring.loss_fn(forward_fn, batch)
        report = StepReport(
            step=self._next_step,
            loss=output,
            rows=len(batch),
            reward_stats=None,
            sync=None,
        )
        self._next_step += 1
        return report


def gspo_algorithm() -> SequencePolicyAlgorithm:
    """Construct a ``SequencePolicyAlgorithm`` binding ``GSPOLoss`` as ``gspo``.

    Zero-argument, so ``registry.py`` may register the factory itself; every
    knob defaults to ``GSPOLoss``'s own defaults.

    WHAT IS CLAIMED: the returned algorithm carries a fresh ``GSPOLoss`` and
    declares a SEQUENCE-scope ratio -- the geometric mean over a response's
    token ratios -- with the tight default clip interval the objective
    carries. The interval's VALUE is a default, not a measurement; the
    load-bearing claim is its SCALE: a sequence ratio concentrates far
    closer to 1 than a token ratio does, so GRPO's (0.8, 1.2) applied to it
    would never bind.

    WHAT IS NOT CLAIMED: that the default epsilon is a measured constant --
    it is not, and the structural claim tested against it is only that the
    interval sits strictly inside GRPO's; or that any run of this algorithm
    has been trained.
    """
    objective = GSPOLoss()
    return SequencePolicyAlgorithm(name="gspo", objective=objective)


def dr_grpo_algorithm() -> SequencePolicyAlgorithm:
    """Construct a ``SequencePolicyAlgorithm`` binding ``DrGRPOLoss``
    as ``dr_grpo``.

    Zero-argument, so ``registry.py`` may register the factory itself.

    WHAT IS CLAIMED: the returned algorithm declares a token-scope ratio
    with a CONSTANT-length reduction -- the batch sum divided by ``rows *
    constant_length`` against a CONFIGURED max length, never the observed
    one, which is precisely Dr.GRPO's length-bias fix -- and a KL weight of
    0.0 by default. Both readings of Dr.GRPO's KL stance ("dropped" and
    "optional") are then representable, and the default matches the
    standard recipe.

    WHAT IS NOT CLAIMED: that the KL term can never be enabled -- the
    objective's own ``kl_weight`` carries it, off by default; or that the
    constant length read out of the objective is tuned.
    """
    objective = DrGRPOLoss()
    return SequencePolicyAlgorithm(name="dr_grpo", objective=objective)


def dapo_algorithm() -> SequencePolicyAlgorithm:
    """Construct a ``SequencePolicyAlgorithm`` binding ``DAPOLoss`` as
    ``dapo``.

    Zero-argument, so ``registry.py`` may register the factory itself.

    WHAT IS CLAIMED: the returned algorithm declares a token-scope ratio
    with ASYMMETRIC clip bounds and declares, through the objective's
    ``expects_dynamic_sampling = True``, that DAPO's dynamic sampling is
    EXPECTED of the rollout side. Filtering zero-variance groups already
    happens in ``GroupNormalisedAdvantage``; the REFILL half --
    regenerating until the batch is full of mixed-outcome groups -- is a
    rollout-loop behaviour, and nothing in this module or this package
    enforces it. A declared, unenforced expectation is honest; silently
    omitting it is the defect.

    WHAT IS NOT CLAIMED: that dynamic sampling or overlong reward shaping
    is implemented anywhere here -- neither is, both are rollout-side, and
    this package has no rollout loop to put them in; or that the asymmetric
    interval is tuned.
    """
    objective = DAPOLoss()
    return SequencePolicyAlgorithm(name="dapo", objective=objective)
