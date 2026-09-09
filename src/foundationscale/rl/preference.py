"""The preference binding: one parameterised algorithm driving six objectives.

``policy_gradient.py`` writes one algorithm class per loss because each
policy-gradient family genuinely fixes different semantic seams (group
size, ratio scope, clip bounds). The six preference objectives do not:
none reads a rollout source, none forms an importance ratio, none reads
a group, and all six consume a caller-supplied batch of preference records
-- pairs for DPO, IPO, ORPO, SimPO and CPO, flagged singletons for KTO.
One parameterised :class:`PreferenceAlgorithm` therefore binds all six,
and six thin factories construct it with the right objective for
``registry.py``: ``dpo_algorithm()``, ``ipo_algorithm()``,
``kto_algorithm()``, ``orpo_algorithm()``, ``simpo_algorithm()`` and
``cpo_algorithm()``.

THE REFERENCE SEAM, stated once. DPO, IPO and KTO are reference-ANCHORED
in the paper sense, but what they CONSUME is reference scores read from
batch columns as sequence-level sums; they never call a reference model.
The role map therefore declares ``reference_policy`` False for all six,
so a policy pair carrying a frozen reference is REFUSED at wiring -- an
unconsumed frozen model is a stub that reads as a working DPO wiring --
while the columns are demanded through the objective's
``required_columns`` schema. "A reference was wired" and "reference
columns were present" are different claims, and only the second is true
of this family.

WHAT THIS MODULE CLAIMS: a wired preference algorithm prices exactly one
caller-supplied batch per step through the objective it was constructed
with, declares the offline role map with every unconsumed role named
False, derives its semantics reference-freeness and its declared
objective denominator FROM the objective instance so the two cannot
drift, and reports ``rows`` as the batch's row count -- one row being
one preference pair, or one flagged singleton under KTO.

WHAT THIS MODULE DOES NOT CLAIM: that any reference model, any model
forward, any producer of the consumed batch columns, or any optimiser
exists here; that the five-field semantics cross-check can fire -- both
sides derive from the one objective, so it is a maintenance tripwire;
or that the loss arithmetic is restated here -- the objectives own it.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Protocol, runtime_checkable

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
from foundationscale.rl.interfaces import (
    BatchRefusal,
    ExperienceBatch,
    ForwardFn,
    LossFn,
)
from foundationscale.rl.losses import DPOLoss
from foundationscale.rl.policy import PolicyPair
from foundationscale.rl.preference_objectives import (
    CPOLoss,
    IPOLoss,
    KTOLoss,
    ORPOLoss,
    SimPOLoss,
)
from foundationscale.rl.rollout import RolloutSource
from foundationscale.rl.weightsync import WeightSync

__all__ = (
    "PreferenceAlgorithm",
    "PreferenceObjective",
    "check_preference_requirements",
    "cpo_algorithm",
    "dpo_algorithm",
    "ipo_algorithm",
    "kto_algorithm",
    "orpo_algorithm",
    "simpo_algorithm",
)


@runtime_checkable
class PreferenceObjective(LossFn, Protocol):
    """The structural contract an objective must satisfy to be bound here.

    WHAT IS CLAIMED: a satisfying object exposes the members this binding
    actually reads -- ``reference_free`` and ``required_columns`` to derive
    the wiring and the batch schema, ``semantics()`` for the five-field
    cross-check -- in addition to everything :class:`LossFn` already
    demands of any loss.

    This is a protocol rather than a union of the six shipped objective
    classes because a union is a CLOSED set: a seventh objective could not
    be bound without editing this module, which is the opposite of what a
    seam is for. Opening it costs no discrimination. ``reference_free`` is
    declared by exactly the six preference objectives and by no other loss
    in the plane, so the protocol admits the same six the union named and
    still refuses ``SFTLoss``.

    WHAT IS NOT CLAIMED: that a satisfying object computes a preference
    objective. ``@runtime_checkable`` measures attribute PRESENCE only,
    never signatures and never meaning, so the ``isinstance`` check in
    :meth:`PreferenceAlgorithm.__init__` is a structural smoke test and not
    a handshake. What makes a binding safe is not that check but the schema
    and semantics measurements taken afterwards from the object's own
    declarations.
    """

    def semantics(self) -> AlgorithmSemantics: ...

    @property
    def reference_free(self) -> bool: ...

    @property
    def required_columns(self) -> tuple[str, ...]: ...


# The members :class:`PreferenceObjective` demands, stated here rather than
# read from its ``__protocol_attrs__``: that attribute exists only on Python
# 3.12+, and this package supports 3.10, so deriving the diagnostic from it
# would replace a clean refusal with an AttributeError on the two oldest
# supported interpreters. Stating a set twice is only safe when something
# measures both statements independently, so tests/rl/test_preference_protocol.py
# drives the protocol's REAL member set through ``isinstance`` -- adding a
# member here that the protocol does not demand, or omitting one it does,
# fails there, and each direction is caught by a DIFFERENT leg.
_OBJECTIVE_MEMBERS: tuple[str, ...] = (
    "__call__",
    "declaration",
    "reference_free",
    "required_columns",
    "semantics",
)


def _absent_members(objective: object) -> tuple[str, ...]:
    """Name the objective members ``objective`` does not expose, in sorted order."""
    return tuple(name for name in _OBJECTIVE_MEMBERS if not hasattr(objective, name))


def check_preference_requirements(
    *,
    requires: Mapping[str, bool],
    supplied: Mapping[str, Any],
    origin: str = "preference",
) -> tuple[str, ...]:
    """Check the preference family's role map over the union of both key sets.

    WHAT IS CLAIMED on success: every required role names a supplied object,
    no required set was emptily satisfied, no supplied required-role value is
    ``None``, and the returned tuple is the sorted set of roles actually
    consumed.

    WHAT IS NOT CLAIMED: that any role object is a valid model, dataloader,
    or loss. Type and semantic checks remain with the component handshakes;
    this helper checks the declared denominator only. The implementation is
    shared with every other family's role-map check -- only the ``origin``
    this binding names differs.
    """
    return check_role_map(requires=requires, supplied=supplied, origin=origin)


@dataclass(frozen=True, slots=True)
class _PreferenceWiring:
    loss_fn: PreferenceObjective
    dataloader: Iterator[ExperienceBatch]
    policy_logprob_columns: tuple[str, ...]
    required_columns: tuple[str, ...]


class PreferenceAlgorithm:
    """Concrete offline preference algorithm bound to one of the six objectives.

    The family is uniform in a way the policy-gradient families are not:
    none of the six objectives consumes a rollout source, an advantage
    function, or a weight sync, and none calls a reference model -- the
    anchored three read reference scores out of batch columns. One
    parameterised binding therefore carries the objective instance it was
    constructed with, and derives its semantic stance and its declared
    objective denominator FROM that instance rather than restating them.

    ``rows`` in the step report is the batch's row count, meaning ONE
    THING across all six objectives: one row is one preference pair for
    DPO, IPO, ORPO, SimPO and CPO, and one flagged singleton completion
    for KTO -- one unit of preference supervision either way. There is no
    per-pair field whose denominator could quietly switch meaning between
    objectives.

    WHAT IS CLAIMED: one successful step invokes the carried objective on
    the next dataloader batch through a forward function that reads the
    config-named policy log-probability columns, and returns a
    ``StepReport`` whose ``loss`` is the objective's own ``LossOutput``
    and whose ``rows`` is the batch length.

    WHAT IS NOT CLAIMED: that a reference model was consulted -- none is,
    and a wired one is refused; that the dataloader rows are fresh,
    deduplicated, or correctly labelled; that another batch exists after
    any step; that any forward implementation is invoked by this
    torch-free object; or that the loss arithmetic is re-implemented
    here.
    """

    __slots__ = (
        "_name",
        "_next_step",
        "_objective",
        "_paired",
        "_requirements",
        "_requires",
        "_semantics",
        "_supplied",
        "_wiring",
    )

    def __init__(self, *, name: str, objective: PreferenceObjective) -> None:
        """Construct the binding's declarations FROM the carried objective.

        The six factories are the ordinary callers; direct construction is
        the seam for objectives whose column or metric names were
        customised beyond the factories' knob surface. ``name`` is the one
        statement that cannot be derived from the objective -- the losses
        carry no family name of their own, and inventing one from the type
        would be a restated countable.

        WHAT IS CLAIMED: an object outside the six preference classes is
        refused at construction; the resulting declarations take
        ``reference_free`` from the objective's property, take the
        declared components and metric names from the objective's own
        ``declaration()``, and declare every optional role unconsumed with
        False -- including ``reference_policy``, because the anchored
        objectives read columns, never a model.

        WHAT IS NOT CLAIMED: that the derived declarations were compared
        against a second statement -- the objective IS the single source,
        and the wiring gate's comparisons against its ``declaration()``
        are construction-time tripwires here, not live seams.
        """
        # The branch below is a SINGLE raise on purpose. mypy narrows the
        # annotated parameter to Never here and reports any other statement
        # as unreachable -- correctly, for a typed caller. The check is kept
        # because the untyped callers are the real ones: the registry
        # constructs through a factory, and nothing static stands between a
        # caller's object and this slot. Naming the missing members inside
        # the raise keeps the diagnostic without adding a second statement,
        # which is the same shape ``registry.py`` uses for its own
        # ``isinstance(algorithm, Algorithm)`` smoke test.
        if not isinstance(objective, PreferenceObjective):
            raise AlgorithmWiringRefusal(
                f"objective={objective!r}: 1 of 1 objective slots must carry "
                f"a PreferenceObjective; {type(objective).__name__} does not "
                f"expose {_absent_members(objective)} "
                f"of the {len(_OBJECTIVE_MEMBERS)} "
                f"required members -- the preference binding cannot price an "
                f"objective whose batch shape it cannot read"
            )
        self._name = name
        self._objective = objective
        # Pairedness is read from the objective's OWN declared schema, not
        # from its concrete class. A class check (``is not KTOLoss``) would
        # default every unrecognised objective to paired, which is the one
        # closed-world assumption that survives opening the union -- and it
        # would fail silently, by mis-shaping the forward closure rather
        # than by refusing. A half-declared schema is refused rather than
        # guessed, because naming a chosen column and no rejected one is an
        # objective whose unit of supervision cannot be attributed.
        declared_columns = tuple(objective.required_columns)
        chosen_columns = tuple(n for n in declared_columns if n.startswith("chosen_"))
        rejected_columns = tuple(n for n in declared_columns if n.startswith("rejected_"))
        if bool(chosen_columns) != bool(rejected_columns):
            raise AlgorithmWiringRefusal(
                f"objective={type(objective).__name__}: its declared schema "
                f"names {len(chosen_columns)} chosen and "
                f"{len(rejected_columns)} rejected column(s) "
                f"({declared_columns}); a preference objective is paired "
                f"only if it declares both sides, and 1 of 2 sides is "
                f"absent -- pairedness decides the forward closure's shape "
                f"and cannot be inferred from half a declaration"
            )
        self._paired = bool(chosen_columns)
        # reference_free is DERIVED from the objective's own property, so the
        # algorithm's semantics and the loss's semantics cannot be written
        # independently and drift; the other four fields abstain with None
        # because a preference objective constrains no group, ratio, KL
        # estimator, or clip seam, and a number there would be a measured
        # value posing as an unconstrained seam.
        self._semantics = AlgorithmSemantics(reference_free=objective.reference_free)
        declaration = objective.declaration()
        self._requirements = AlgorithmRequirements(
            name=name,
            # Every role this surface grades is named, including the four it
            # does NOT consume: reference_policy is False even for the
            # reference-anchored objectives, because they read reference
            # scores from batch columns and never call a reference model.
            # A role declared False is graded, and a supplied-but-unconsumed
            # one is refused, never silently ignored.
            requires={
                "rollout_source": False,
                "advantage_fn": False,
                "weight_sync": False,
                "reference_policy": False,
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
                "advantage_fn": False,
                "reference_policy": False,
                "rollout_source": False,
                "weight_sync": False,
            }
        )
        self._supplied: Mapping[str, Any] | None = None
        self._wiring: _PreferenceWiring | None = None
        self._next_step = 0

    def requirements(self) -> AlgorithmRequirements:
        """Return the role and objective declaration measured by gates.

        WHAT IS CLAIMED: the record is this binding's declaration: the
        offline role map with every unconsumed role named False, and the
        component and metric denominators read off the carried objective's
        own declaration.

        WHAT IS NOT CLAIMED: that the declaration is sufficient evidence the
        mathematics are correct; it is the denominator the gates measure.
        """
        return self._requirements

    def semantics(self) -> AlgorithmSemantics:
        """Return this binding's algorithm-side semantics declaration.

        WHAT IS CLAIMED: ``reference_free`` came from the objective's own
        property, and the other four fields are ``None`` abstentions --
        this family constrains no group, ratio, KL, or clip seam.

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
        the pair, the objective, and the dataloader, and nothing else,
        because the family consumes nothing else.

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
        with -- identity, not equality; two preference losses that agree
        everywhere still double the one declared objective role. ``config``
        must name the policy log-probability columns the forward function
        will read: ``policy_chosen_logprob_column`` and
        ``policy_rejected_logprob_column`` for the five paired objectives,
        ``policy_logprob_column`` for KTO's unpaired rows.

        WHAT IS CLAIMED on success: the handed loss is the carried
        objective; the five-field semantics cross-check between this
        algorithm's record and the loss's own ``semantics()`` passes over
        exactly ``("group_size", "ratio_scope", "kl_estimator",
        "clip_bounds", "reference_free")``; the existing wiring handshake
        accepted the offline role map, including refusing any reference
        model the pair carries; the first batch's columns cover the
        objective's ``required_columns`` -- a PROPERTY, read without call
        parentheses -- plus the config-named policy columns, and the batch
        is replayed rather than consumed; and the two independent role
        denominators name the same consumed set.

        WHAT IS NOT CLAIMED: that batches after the first were inspected --
        they are schema-checked per step instead; that an empty dataloader
        is refused here -- an absent batch is a data fact and the
        step-time refusal keeps it at call time; or that the columns carry
        well-typed values, which is the objective's own per-step
        measurement.
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
            # mapping: the pair attests it -- and with the family declaring
            # it False, a pair carrying a frozen reference is refused,
            # because an unconsumed frozen model is a stub that reads as a
            # working DPO wiring. The anchored objectives demand reference
            # COLUMNS from the batch schema below, never a model.
            supplied={
                "rollout_source": rollout_source,
                "advantage_fn": advantage_fn,
                "weight_sync": weight_sync,
            },
            origin="PreferenceAlgorithm.setup",
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
        configured_keys: tuple[str, ...]
        if self._paired:
            configured_keys = (
                "policy_chosen_logprob_column",
                "policy_rejected_logprob_column",
            )
        else:
            configured_keys = ("policy_logprob_column",)
        policy_columns: list[str] = []
        for key in configured_keys:
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
        # call parentheses; the schema the objective declared plus the
        # policy columns this binding declared is the full contract a batch
        # must carry.
        # The declared schema is RECORDED here and measured in step, never
        # measured here. Setup is the wiring handshake for every family in
        # this plane: it reads the roles it was handed and touches no data.
        # Peeking one batch to schema-check it early would make setup
        # data-dependent -- a loader that blocks or raises on its first item
        # would report a wiring fault for a data fact -- and it would buy
        # nothing, because step re-checks these same columns on EVERY batch
        # and so already refuses the first one. Storing the iterator is
        # separate and IS house idiom: ppo.py and grpo.py both call
        # ``iter(dataloader)`` in setup and consume with ``next()`` in step.
        required_columns = tuple(loss_fn.required_columns)
        supplied = MappingProxyType(
            {
                "policy_pair": policy_pair,
                "loss_fn": loss_fn,
                "dataloader": dataloader,
            }
        )
        mapped_consumed = check_preference_requirements(
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
        self._wiring = _PreferenceWiring(
            loss_fn=loss_fn,
            dataloader=dataloader_iterator,
            policy_logprob_columns=tuple(policy_columns),
            required_columns=required_columns,
        )

    def step(self) -> StepReport:
        """Price the next batch through the carried objective and report rows.

        ``rows`` is the batch's row count, and it means the same thing for
        all six objectives: one row is one unit of preference supervision --
        one (chosen, rejected) pair for the five paired objectives, one
        flagged singleton completion for KTO. No per-pair field exists in
        this report, so no denominator silently changes units between
        objectives. ``reward_stats`` and ``sync`` abstain as ``None``,
        because this family consumes no advantage function and no weight
        sync.

        WHAT IS CLAIMED: the batch's columns cover the wiring's declared
        schema before the objective is invoked, the forward function reads
        exactly the config-named policy columns, and the report carries the
        objective's own ``LossOutput`` with ``rows`` equal to the batch
        length.

        WHAT IS NOT CLAIMED: that another batch exists after this one; that
        any reference model ran -- none does in this family; or that the
        result is finite.
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
        # Both branches build the plane's own ForwardFn seam
        # (Callable[[ExperienceBatch], Any]); the objective, not this
        # binding, owns what the returned scores mean.
        forward_fn: ForwardFn
        if self._paired:
            chosen_column = wiring.policy_logprob_columns[0]
            rejected_column = wiring.policy_logprob_columns[1]

            def paired_forward(current_batch: ExperienceBatch) -> Any:
                chosen_scores = current_batch.column(chosen_column)
                rejected_scores = current_batch.column(rejected_column)
                return tuple(zip(chosen_scores, rejected_scores, strict=True))

            forward_fn = paired_forward
        else:
            column = wiring.policy_logprob_columns[0]

            def single_forward(current_batch: ExperienceBatch) -> Any:
                return current_batch.column(column)

            forward_fn = single_forward

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


def dpo_algorithm(
    *,
    beta: float = 0.1,
    weight: float = 1.0,
    sft_weight: float = 0.0,
) -> PreferenceAlgorithm:
    """Construct a ``PreferenceAlgorithm`` binding ``DPOLoss`` as ``dpo``.

    Zero-argument-callable-compatible: every knob defaults to ``DPOLoss``'s
    own defaults, so ``registry.py`` may register the factory itself.

    WHAT IS CLAIMED: the returned algorithm carries a ``DPOLoss`` built with
    exactly the given knobs, including the optional auxiliary SFT term when
    -- and only when -- ``sft_weight`` is non-zero, and declares the
    reference-anchored, column-reading role map.

    WHAT IS NOT CLAIMED: that the defaults are tuned values, that column or
    metric names are customisable here (construct ``PreferenceAlgorithm``
    directly with a pre-built objective for that), or that a reference
    model runs anywhere.
    """
    objective = DPOLoss(beta=beta, weight=weight, sft_weight=sft_weight)
    return PreferenceAlgorithm(name="dpo", objective=objective)


def ipo_algorithm(*, tau: float = 1.0, weight: float = 1.0) -> PreferenceAlgorithm:
    """Construct a ``PreferenceAlgorithm`` binding ``IPOLoss`` as ``ipo``.

    WHAT IS CLAIMED: the returned algorithm carries an ``IPOLoss`` built
    with exactly the given knobs, with ``tau`` named deliberately rather
    than borrowing DPO's ``beta``, and declares the reference-anchored,
    column-reading role map.

    WHAT IS NOT CLAIMED: that the factory is anything but a constructor --
    zero-argument-callable-compatible for the registry, with no tuned
    opinion about ``tau``.
    """
    objective = IPOLoss(tau=tau, weight=weight)
    return PreferenceAlgorithm(name="ipo", objective=objective)


def kto_algorithm(
    *,
    beta: float = 0.1,
    weight: float = 1.0,
    lambda_desirable: float = 1.0,
    lambda_undesirable: float = 1.0,
) -> PreferenceAlgorithm:
    """Construct a ``PreferenceAlgorithm`` binding ``KTOLoss`` as ``kto``.

    WHAT IS CLAIMED: the returned algorithm is the family's UNPAIRED member:
    its forward function reads a single policy log-probability column, its
    ``rows`` count flagged singletons rather than pairs, and its objective
    demands the recorded KL reference-point column rather than estimating
    it in-batch.

    WHAT IS NOT CLAIMED: that ``rows`` changes meaning under KTO -- one row
    is one unit of preference supervision here exactly as a pair is for the
    other five -- or that any choice of lambda values is endorsed.
    """
    objective = KTOLoss(
        beta=beta,
        weight=weight,
        lambda_desirable=lambda_desirable,
        lambda_undesirable=lambda_undesirable,
    )
    return PreferenceAlgorithm(name="kto", objective=objective)


def orpo_algorithm(*, lambda_: float = 1.0) -> PreferenceAlgorithm:
    """Construct a ``PreferenceAlgorithm`` binding ``ORPOLoss`` as ``orpo``.

    WHAT IS CLAIMED: the returned algorithm carries an ``ORPOLoss`` built
    with exactly the given odds-ratio weight, declares BOTH of ORPO's
    components, and is reference-FREE: its role map admits no reference
    model and its schema demands no reference columns.

    WHAT IS NOT CLAIMED: that the SFT term can be switched off through this
    factory -- removing it is a different objective, not a configuration of
    this one.
    """
    objective = ORPOLoss(lambda_=lambda_)
    return PreferenceAlgorithm(name="orpo", objective=objective)


def simpo_algorithm(
    *,
    beta: float = 1.0,
    gamma: float = 0.0,
    weight: float = 1.0,
) -> PreferenceAlgorithm:
    """Construct a ``PreferenceAlgorithm`` binding ``SimPOLoss`` as ``simpo``.

    WHAT IS CLAIMED: the returned algorithm carries a ``SimPOLoss`` built
    with exactly the given knobs, and is reference-FREE like ORPO and CPO:
    length-normalised rewards are derived from the batch's masks, never
    accepted as columns.

    WHAT IS NOT CLAIMED: that ``gamma=0.0`` is a tuned value -- it is the
    corpus idiom of defaulting every knob, and tuned runs are expected to
    set it.
    """
    objective = SimPOLoss(beta=beta, gamma=gamma, weight=weight)
    return PreferenceAlgorithm(name="simpo", objective=objective)


def cpo_algorithm(*, beta: float = 0.1, lambda_: float = 1.0) -> PreferenceAlgorithm:
    """Construct a ``PreferenceAlgorithm`` binding ``CPOLoss`` as ``cpo``.

    WHAT IS CLAIMED: the returned algorithm carries a ``CPOLoss`` built
    with exactly the given knobs, declares BOTH of CPO's components, and
    is reference-FREE by the paper's uniform-reference approximation.

    WHAT IS NOT CLAIMED: that the dropped reference term was small -- that
    assumption is unverifiable here and stays the objective's own
    documented caveat, not a claim this binding adds to.
    """
    objective = CPOLoss(beta=beta, lambda_=lambda_)
    return PreferenceAlgorithm(name="cpo", objective=objective)
