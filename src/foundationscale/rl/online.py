"""The online binding: one parameterised algorithm driving four objectives.

``preference.py`` showed that one parameterised binding can carry a whole
family when every member consumes the same shape of thing: a
caller-supplied batch, priced through the objective it was constructed
with. The four online objectives -- ``OnlineDPOLoss``,
``IterativeDPOLoss``, ``RAFTLoss`` and ``BestOfNLoss`` in
``online_objectives.py`` -- have that same shape: none reads a rollout
source, none forms an importance ratio, none calls a reference model
(the two DPO variants read reference scores as pre-summed batch
columns), and all four price the next dataloader batch through a
caller-supplied forward function. One parameterised
:class:`OnlineAlgorithm` therefore binds all four, and four thin
factories construct it with the right objective for ``registry.py``:
``online_dpo_algorithm()``, ``iterative_dpo_algorithm()``,
``raft_algorithm()`` and ``best_of_n_algorithm()``.

WHY THIS IS NOT PreferenceAlgorithm, stated once. Two differences are
documentation-honest and one is load-bearing. First, these objectives'
supervision is drawn from the CURRENT policy's own samples, which the
binding cannot verify -- ``rollout_source`` is still False because
rollout production is upstream of the step, and only the producer knows
whether the samples are on-policy. Second, ``rows`` does NOT mean one
unit of supervision for every member here the way it does across the
preference family: for ``best_of_n`` one row is one SAMPLED completion
and several rows form one group, so the loss prices one supervision unit
per GROUP, not per row. That varying denominator is what the explicit
:class:`RowsUnit` declaration records, and it is the reason this family
gets its own binding instead of being hosted by
``PreferenceAlgorithm``.

WHAT THIS MODULE CLAIMS: a wired online algorithm prices exactly one
caller-supplied batch per step through the objective it was constructed
with; declares the offline role map with every unconsumed role named
False, including ``rollout_source`` (an Algorithm PRICES A BATCH -- who
produced it is upstream, the convention ``grpo.py`` established);
derives pairedness from the objective's own declared schema, never from
the concrete class; refuses a ``rows_unit`` that contradicts that
derivation; and reports ``rows`` as the batch's row count, meaning
exactly what the constructed ``rows_unit`` says.

WHAT THIS MODULE DOES NOT CLAIM: that the batch samples came from the
current policy -- the binding cannot verify provenance and does not
pretend to; that the ``isinstance`` check in the constructor
discriminates an online loss from a preference loss -- it cannot and is
documented as the member-presence smoke test it is; that any producer of
the consumed batch columns, any reward model, any snapshot store, or any
optimiser exists here; or that the loss arithmetic is restated here --
the objectives own it.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from enum import Enum
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
from foundationscale.rl.online_objectives import (
    BestOfNLoss,
    IterativeDPOLoss,
    OnlineDPOLoss,
    RAFTLoss,
)
from foundationscale.rl.policy import PolicyPair
from foundationscale.rl.rollout import RolloutSource
from foundationscale.rl.weightsync import WeightSync

__all__ = (
    "OnlineAlgorithm",
    "OnlineObjective",
    "RowsUnit",
    "best_of_n_algorithm",
    "iterative_dpo_algorithm",
    "online_dpo_algorithm",
    "raft_algorithm",
)


class RowsUnit(Enum):
    """What ONE batch row counts, declared per member.

    The declaration exists because the online family is the one place in
    the plane where "one row" does not uniformly mean "one unit of
    supervision": a preference pair is one unit outright, a RAFT survivor
    is one unit outright, but a best-of-N SAMPLE is only a member of the
    group the objective actually supervises over. The value is stated by
    the factory and refused against the schema-derived pairedness in
    :meth:`OnlineAlgorithm.__init__` -- a PREFERENCE_PAIR claim on an
    unpaired objective, or a single-completion claim on a paired one, is
    two statements about one forward shape that cannot both be true.

    WHAT IS CLAIMED: each member states, once, what ``StepReport.rows``
    counts for the binding constructed with it.

    WHAT IS NOT CLAIMED: that the declaration makes the batch on-policy,
    well-grouped, or correctly rewarded -- those are producer facts; or
    that ``rows`` is convertible to a supervision-unit count here -- for
    SAMPLED_COMPLETION the group count lives inside the objective, not in
    this report.
    """

    PREFERENCE_PAIR = "preference_pair"  # online_dpo, iterative_dpo
    SURVIVING_COMPLETION = "surviving_completion"  # raft
    SAMPLED_COMPLETION = "sampled_completion"  # best_of_n


@runtime_checkable
class OnlineObjective(LossFn, Protocol):
    """The structural contract an objective must satisfy to be bound here.

    WHAT IS CLAIMED: a satisfying object exposes the members this binding
    actually reads -- ``reference_free`` and ``required_columns`` to
    derive the wiring and the batch schema, ``semantics()`` for the
    five-field cross-check -- in addition to everything :class:`LossFn`
    already demands of any loss. This is a protocol for the reason
    ``preference.py`` stated: a union of the four shipped classes is a
    CLOSED set, and a fifth online objective must bind without an edit
    here.

    WHAT IS NOT CLAIMED: that a satisfying object computes an ONLINE
    objective. ``@runtime_checkable`` answers member PRESENCE only, never
    signatures and never meaning, and it was measured at HEAD that all
    four online objectives satisfy ``PreferenceObjective`` today -- the
    two protocols demand the same members, so an ``isinstance`` check
    cannot discriminate an online loss from a preference one, and this
    binding does not pretend it can. The check's real job is narrower: to
    refuse an object that is missing members outright, before the schema
    and semantics measurements below read declarations from something
    that has none. On-policy provenance is likewise NOT what this
    protocol expresses -- where the samples came from is upstream of
    every measurement the binding makes.
    """

    def semantics(self) -> AlgorithmSemantics: ...

    @property
    def reference_free(self) -> bool: ...

    @property
    def required_columns(self) -> tuple[str, ...]: ...


# The members :class:`OnlineObjective` demands, stated here rather than
# read from its ``__protocol_attrs__`` for the reason preference.py
# states: that attribute exists only on Python 3.12+, and this package
# supports 3.10, so deriving the diagnostic from it would replace a clean
# refusal with an AttributeError on the two oldest supported
# interpreters. Stating a set twice is only safe when something measures
# both statements independently, so tests/rl/test_online_protocol.py
# drives the protocol's REAL member set through ``isinstance`` -- adding
# a member here that the protocol does not demand, or omitting one it
# does, fails there, and each direction is caught by a DIFFERENT leg.
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


@dataclass(frozen=True, slots=True)
class _OnlineWiring:
    loss_fn: OnlineObjective
    dataloader: Iterator[ExperienceBatch]
    policy_logprob_columns: tuple[str, ...]
    required_columns: tuple[str, ...]


class OnlineAlgorithm:
    """Concrete online algorithm bound to one of the four online objectives.

    The family is uniform in the way ``preference.py`` measured: none of
    the four objectives consumes a rollout source, an advantage function,
    or a weight sync, and none calls a reference model -- the two
    reference-anchored DPO variants read reference scores out of batch
    columns. ``rollout_source`` is declared False NOT because the samples
    are offline -- they are drawn from the current policy -- but because
    rollout production is upstream of the step: the binding PRICES the
    batch it is handed, and only the producer knows its provenance. One
    parameterised binding therefore carries the objective instance it was
    constructed with, and derives its semantic stance and its declared
    objective denominator FROM that instance rather than restating them.

    ``rows`` in the step report is the batch's row count, and its meaning
    is the constructed ``rows_unit``: one row is one preference pair for
    ``online_dpo`` and ``iterative_dpo`` (PREFERENCE_PAIR -- one unit of
    supervision outright) and one surviving completion for ``raft``
    (SURVIVING_COMPLETION -- one unit outright), but one SAMPLED
    completion for ``best_of_n`` (SAMPLED_COMPLETION): several rows form
    one group there, the objective selects and prices one winner per
    group, so ``rows`` counts samples drawn, not units of supervision.
    That differing denominator is the reason ``rows_unit`` is an explicit
    declaration rather than a comment.

    WHAT IS CLAIMED: one successful step invokes the carried objective on
    the next dataloader batch through a forward function that reads the
    config-named policy log-probability columns, and returns a
    ``StepReport`` whose ``loss`` is the objective's own ``LossOutput``
    and whose ``rows`` is the batch length, interpreted per
    ``rows_unit``.

    WHAT IS NOT CLAIMED: that the batch samples came from the current
    policy -- the binding cannot verify provenance and says so rather
    than pretending the wiring encodes it; that a reference model was
    consulted -- none is, and a wired one is refused; that the dataloader
    rows are fresh, well-grouped, or correctly rewarded; or that any
    forward implementation is invoked by this torch-free object.
    """

    __slots__ = (
        "_name",
        "_next_step",
        "_objective",
        "_paired",
        "_requirements",
        "_requires",
        "_rows_unit",
        "_semantics",
        "_supplied",
        "_wiring",
    )

    def __init__(self, *, name: str, objective: OnlineObjective, rows_unit: RowsUnit) -> None:
        """Construct the binding's declarations FROM the carried objective.

        The four factories are the ordinary callers; direct construction
        is the seam for objectives whose column or metric names were
        customised beyond the factories' knob surface. ``name`` is the
        one statement that cannot be derived from the objective, and
        ``rows_unit`` is the one declaration the objective's schema
        cannot make -- the schema states whether rows are PAIRED, never
        whether a row is a whole unit of supervision.

        WHAT IS CLAIMED: an object outside the protocol's structural
        member set is refused at construction; pairedness is derived from
        the objective's own declared ``chosen_``/``rejected_`` columns,
        with a half-declared schema refused; a ``rows_unit`` of
        PREFERENCE_PAIR is refused against an unpaired schema and a
        single-completion unit against a paired one, with the refusal
        naming both sides; the declarations take ``reference_free`` from
        the objective's property, take the component and metric names
        from its own ``declaration()``, and declare every optional role
        unconsumed with False -- including ``rollout_source`` and
        ``reference_policy``.

        WHAT IS NOT CLAIMED: that the ``isinstance`` check proves the
        objective is an ONLINE objective -- it proves member presence
        only, and the four preference objectives would pass it too; that
        ``rollout_source=False`` asserts a fresh-sample cadence -- it is
        the house role map with the rationale that production is upstream
        of pricing, and the binding cannot verify the samples came from
        the current policy; or that the derived declarations were
        compared against a second statement -- the objective IS the
        single source.
        """
        # The branch below is a SINGLE raise on purpose, for the reason
        # preference.py states: mypy narrows the annotated parameter to
        # Never here, but the untyped callers are the real ones -- the
        # registry constructs through a factory, and nothing static
        # stands between a caller's object and this slot. Naming the
        # missing members inside the raise keeps the diagnostic without
        # adding a second statement. The check is a structural smoke
        # test, not a discriminator: an online loss and a preference
        # loss are member-identical, and only the memberless are caught.
        if not isinstance(objective, OnlineObjective):
            raise AlgorithmWiringRefusal(
                f"objective={objective!r}: 1 of 1 objective slots must carry "
                f"an OnlineObjective; {type(objective).__name__} does not "
                f"expose {_absent_members(objective)} "
                f"of the {len(_OBJECTIVE_MEMBERS)} "
                f"required members -- the online binding cannot price an "
                f"objective whose batch shape it cannot read"
            )
        self._name = name
        self._objective = objective
        # Pairedness is read from the objective's OWN declared schema,
        # not from its concrete class -- the preference.py derivation,
        # adopted unchanged. A class check would default every
        # unrecognised objective to one shape, the closed-world
        # assumption that survives opening the union, and it would fail
        # silently by mis-shaping the forward closure rather than by
        # refusing. A half-declared schema is refused rather than
        # guessed, because naming a chosen column and no rejected one is
        # an objective whose unit of supervision cannot be attributed.
        declared_columns = tuple(objective.required_columns)
        chosen_columns = tuple(n for n in declared_columns if n.startswith("chosen_"))
        rejected_columns = tuple(n for n in declared_columns if n.startswith("rejected_"))
        if bool(chosen_columns) != bool(rejected_columns):
            raise AlgorithmWiringRefusal(
                f"objective={type(objective).__name__}: its declared schema "
                f"names {len(chosen_columns)} chosen and "
                f"{len(rejected_columns)} rejected column(s) "
                f"({declared_columns}); an objective is paired only if it "
                f"declares both sides, and 1 of 2 sides is absent -- "
                f"pairedness decides the forward closure's shape and "
                f"cannot be inferred from half a declaration"
            )
        self._paired = bool(chosen_columns)
        # rows_unit is resolved to the forward shape it implies, then
        # MEASURED against the schema-derived pairedness. The declaration
        # exists to be checked, not stored: the schema can state that
        # rows are paired, but it cannot state that one row is one unit
        # of supervision, so the explicit unit and the derived shape are
        # the two independent statements whose agreement this refusal
        # protects.
        unit_expects_paired: bool
        if rows_unit is RowsUnit.PREFERENCE_PAIR:
            unit_expects_paired = True
        elif rows_unit in (RowsUnit.SURVIVING_COMPLETION, RowsUnit.SAMPLED_COMPLETION):
            # The two single-completion units share a forward shape and are
            # tested together; they stay separate MEMBERS because they name
            # different denominators downstream (a survivor is one unit of
            # supervision, a sample is one member of a group), which is a
            # reporting distinction, not a shape distinction.
            unit_expects_paired = False
        else:
            raise AlgorithmWiringRefusal(
                f"rows_unit={rows_unit!r}: 0 of {len(RowsUnit)} RowsUnit "
                f"members matched; one of "
                f"{tuple(member.value for member in RowsUnit)} must declare "
                f"what one batch row counts, and an unrecognised value "
                f"cannot name the step report's denominator"
            )
        if unit_expects_paired != self._paired:
            # The units that WOULD have matched are named, not left to the
            # caller to re-derive: the refusal is read by someone who wrote
            # one of the two declarations wrong and cannot yet tell which,
            # and naming the matching set turns "these disagree" into a
            # repair instruction without deciding WHICH side is the error
            # -- the binding cannot know whether the schema or the unit was
            # the mistake.
            matching = tuple(
                member.value
                for member in RowsUnit
                if (member is RowsUnit.PREFERENCE_PAIR) is self._paired
            )
            raise AlgorithmWiringRefusal(
                f"rows_unit={rows_unit.value!r} declares one row is one "
                f"{'PREFERENCE PAIR' if unit_expects_paired else 'single completion'} "
                f"(a{'' if unit_expects_paired else 'n'} "
                f"{'paired' if unit_expects_paired else 'unpaired'} "
                f"forward shape), but objective="
                f"{type(objective).__name__} derives a"
                f"{'' if self._paired else 'n'} "
                f"{'paired' if self._paired else 'unpaired'} schema from "
                f"{len(chosen_columns)} chosen and "
                f"{len(rejected_columns)} rejected column(s) "
                f"({declared_columns}); 1 of 2 sides of this construction "
                f"-- the declared row unit and the derived schema -- names "
                f"a forward shape the other contradicts, and the binding "
                f"prices one shape per step, never an average of two. The "
                f"{len(matching)} of {len(RowsUnit)} RowsUnit member(s) that "
                f"match this schema: {matching}"
            )
        # RETAINED, not merely checked. StepReport carries a bare `rows`
        # integer and no unit, so if the declaration were consumed by the
        # refusal above and dropped, nothing downstream could say what
        # that integer counts -- the report would state a denominator
        # whose meaning lives only in the factory that is no longer on
        # the stack. Keeping it is what makes the `rows_unit` property
        # below able to answer for the number.
        self._rows_unit = rows_unit
        # reference_free is DERIVED from the objective's own property, so
        # the algorithm's semantics and the loss's semantics cannot be
        # written independently and drift; the other four fields abstain
        # with None because an online objective constrains no group,
        # ratio, KL estimator, or clip seam, and a number there would be
        # a measured value posing as an unconstrained seam.
        self._semantics = AlgorithmSemantics(reference_free=objective.reference_free)
        declaration = objective.declaration()
        self._requirements = AlgorithmRequirements(
            name=name,
            # Every role this surface grades is named, including the four
            # it does NOT consume. rollout_source is False NOT because
            # the samples are offline but because rollout production is
            # upstream of the step (grpo.py's stated convention: an
            # Algorithm PRICES A BATCH), and reference_policy is False
            # even for the anchored objectives, because they read
            # reference scores from batch columns and never call a
            # reference model. A role declared False is graded, and a
            # supplied-but-unconsumed one is refused, never silently
            # ignored.
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
        self._wiring: _OnlineWiring | None = None
        self._next_step = 0

    @property
    def rows_unit(self) -> RowsUnit:
        """Return what one row of this binding's ``StepReport`` counts.

        WHAT IS CLAIMED: the value this binding was constructed with,
        after it was measured against the schema-derived pairedness. It
        is exposed because :class:`StepReport` reports ``rows`` as a bare
        integer: a reader holding the report and the algorithm can say
        whether that integer counts preference pairs, surviving
        completions, or sampled completions, and for SAMPLED_COMPLETION
        that it is NOT a count of supervision units.

        WHAT IS NOT CLAIMED: that the batch honoured the declaration.
        The unit is a statement about how this binding INTERPRETS the
        rows it is handed, not a measurement of the rows themselves --
        nothing here counts groups, verifies that a best-of-N group is
        complete, or detects a producer that emitted a different shape.
        """
        return self._rows_unit

    def requirements(self) -> AlgorithmRequirements:
        """Return the role and objective declaration measured by gates.

        WHAT IS CLAIMED: the record is this binding's declaration: the
        role map with every unconsumed role named False -- rollout_source
        included, because pricing a batch is not producing one -- and the
        component and metric denominators read off the carried
        objective's own declaration.

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
        with -- identity, not equality; two online losses that agree
        everywhere still double the one declared objective role.
        ``config`` must name the policy log-probability columns the
        forward function will read: ``policy_chosen_logprob_column`` and
        ``policy_rejected_logprob_column`` for the two paired objectives,
        ``policy_logprob_column`` for the two single-completion
        objectives. Supplying a ``rollout_source`` is REFUSED with the
        role map: the binding prices batches, it does not produce them.

        WHAT IS CLAIMED on success: the handed loss is the carried
        objective; the five-field semantics cross-check between this
        algorithm's record and the loss's own ``semantics()`` passes over
        exactly ``("group_size", "ratio_scope", "kl_estimator",
        "clip_bounds", "reference_free")``; the existing wiring handshake
        accepted the role map, including refusing any reference model the
        pair carries; the objective's ``required_columns`` declaration --
        a PROPERTY, read without call parentheses, which for
        ``BestOfNLoss`` already names its group and reward columns --
        plus the config-named policy columns is recorded as the batch
        contract; and the two independent role denominators name the same
        consumed set.

        WHAT IS NOT CLAIMED: that any batch was inspected here -- the
        schema is measured per step, so a loader that blocks or raises on
        its first item reports a data fault at step time, not a wiring
        fault here; that the dataloader has another batch; that the
        samples came from the current policy; or that the columns carry
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
            # COLUMNS from the batch schema below, never a model. Likewise
            # rollout_source is False though the data is on-policy: the
            # binding prices the batch, and production stays upstream.
            supplied={
                "rollout_source": rollout_source,
                "advantage_fn": advantage_fn,
                "weight_sync": weight_sync,
            },
            origin="OnlineAlgorithm.setup",
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
        # must carry. For BestOfNLoss that contract ALREADY includes its
        # group and reward columns -- the binding supplies log-probability
        # sequences through the forward closure and never reads a reward,
        # so no member gets special-cased here.
        # The declared schema is RECORDED here and measured in step, never
        # measured here. Setup is the wiring handshake for every family in
        # this plane: it reads the roles it was handed and touches no data.
        # Peeking one batch to schema-check it early would make setup
        # data-dependent -- a loader that blocks or raises on its first item
        # would report a wiring fault for a data fact -- and it would buy
        # nothing, because step re-checks these same columns on EVERY batch
        # and so already refuses the first one. Storing the iterator is
        # separate and IS house idiom: ppo.py, grpo.py and preference.py all
        # call ``iter(dataloader)`` in setup and consume with ``next()`` in
        # step.
        required_columns = tuple(loss_fn.required_columns)
        supplied = MappingProxyType(
            {
                "policy_pair": policy_pair,
                "loss_fn": loss_fn,
                "dataloader": dataloader,
            }
        )
        mapped_consumed = check_role_map(
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
        self._wiring = _OnlineWiring(
            loss_fn=loss_fn,
            dataloader=dataloader_iterator,
            policy_logprob_columns=tuple(policy_columns),
            required_columns=required_columns,
        )

    def step(self) -> StepReport:
        """Price the next batch through the carried objective and report rows.

        ``rows`` is the batch's row count, and its meaning is exactly the
        constructed ``rows_unit``: one preference pair per row for
        ``online_dpo`` and ``iterative_dpo`` and one surviving completion
        per row for ``raft`` -- one unit of supervision in both cases --
        but one SAMPLED completion per row for ``best_of_n``, where
        several rows form one group and the objective prices one winner
        per group, so ``rows`` counts samples drawn rather than
        supervision units. ``reward_stats`` and ``sync`` abstain as
        ``None``, because this family consumes no advantage function and
        no weight sync.

        WHAT IS CLAIMED: the batch's columns cover the wiring's declared
        schema before the objective is invoked, the forward function
        reads exactly the config-named policy columns, and the report
        carries the objective's own ``LossOutput`` with ``rows`` equal to
        the batch length.

        WHAT IS NOT CLAIMED: that another batch exists after this one;
        that the batch's samples were produced by the current policy --
        provenance is upstream and only the producer knows it; that any
        reference model ran -- none does in this family; or that the
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
                f"the schema the wiring recorded, because a column present "
                f"at setup and absent now is a changed denominator"
            )
        # Both branches build the plane's own ForwardFn seam
        # (Callable[[ExperienceBatch], Any]); the objective, not this
        # binding, owns what the returned scores mean -- including
        # BestOfNLoss's own group and reward readings, which the batch
        # contract above already covers.
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


def online_dpo_algorithm(
    *,
    beta: float = 0.1,
    weight: float = 1.0,
) -> OnlineAlgorithm:
    """Construct an ``OnlineAlgorithm`` binding ``OnlineDPOLoss`` as ``online_dpo``.

    Zero-argument-callable: every knob defaults to ``OnlineDPOLoss``'s
    own defaults, so ``registry.py`` may register the factory itself
    rather than the unconfigured class.

    WHAT IS CLAIMED: the returned algorithm carries an ``OnlineDPOLoss``
    built with exactly the given knobs, declares the reference-anchored,
    column-reading role map, and declares ``PREFERENCE_PAIR``: one batch
    row is one (chosen, rejected) pair and one unit of supervision.

    WHAT IS NOT CLAIMED: that the pairs are fresh -- the binding prices
    the batch it is handed and cannot verify on-policy provenance; that
    the defaults are tuned values; or that a reference model runs
    anywhere.
    """
    objective = OnlineDPOLoss(beta=beta, weight=weight)
    return OnlineAlgorithm(
        name="online_dpo",
        objective=objective,
        rows_unit=RowsUnit.PREFERENCE_PAIR,
    )


def iterative_dpo_algorithm(
    *,
    beta: float = 0.1,
    weight: float = 1.0,
) -> OnlineAlgorithm:
    """Construct an ``OnlineAlgorithm`` binding ``IterativeDPOLoss`` as ``iterative_dpo``.

    Zero-argument-callable: every knob defaults to ``IterativeDPOLoss``'s
    own defaults, so ``registry.py`` may register the factory itself
    rather than the unconfigured class.

    WHAT IS CLAIMED: the returned algorithm carries an
    ``IterativeDPOLoss`` built with exactly the given knobs -- the
    objective that refuses a batch mixing reference snapshots -- and
    declares ``PREFERENCE_PAIR``: one batch row is one
    snapshot-consistent pair and one unit of supervision.

    WHAT IS NOT CLAIMED: that any snapshot refresh cadence is enforced --
    the objective measures snapshot uniformity inside one batch, and the
    schedule belongs to the producer; or that the defaults are tuned
    values.
    """
    objective = IterativeDPOLoss(beta=beta, weight=weight)
    return OnlineAlgorithm(
        name="iterative_dpo",
        objective=objective,
        rows_unit=RowsUnit.PREFERENCE_PAIR,
    )


def raft_algorithm(*, weight: float = 1.0) -> OnlineAlgorithm:
    """Construct an ``OnlineAlgorithm`` binding ``RAFTLoss`` as ``raft``.

    Zero-argument-callable: every knob defaults to ``RAFTLoss``'s own
    defaults, so ``registry.py`` may register the factory itself rather
    than the unconfigured class.

    WHAT IS CLAIMED: the returned algorithm carries a ``RAFTLoss`` built
    with exactly the given weight, is reference-FREE, and declares
    ``SURVIVING_COMPLETION``: the batch arrives already filtered
    best-of-N upstream, and one row is one surviving completion -- one
    unit of supervision.

    WHAT IS NOT CLAIMED: that the survivors were actually the best of
    anything -- the ranking is the producer's deed, and the batch
    carries no trace of the N it was drawn from; or that the binding
    knows anything about the filter.
    """
    objective = RAFTLoss(weight=weight)
    return OnlineAlgorithm(
        name="raft",
        objective=objective,
        rows_unit=RowsUnit.SURVIVING_COMPLETION,
    )


def best_of_n_algorithm(*, weight: float = 1.0) -> OnlineAlgorithm:
    """Construct an ``OnlineAlgorithm`` binding ``BestOfNLoss`` as ``best_of_n``.

    Zero-argument-callable: every knob defaults to ``BestOfNLoss``'s own
    defaults, so ``registry.py`` may register the factory itself rather
    than the unconfigured class.

    WHAT IS CLAIMED: the returned algorithm carries a ``BestOfNLoss``
    built with exactly the given weight, is reference-FREE, and declares
    ``SAMPLED_COMPLETION``: one batch row is one SAMPLED completion,
    several rows form one group, and the objective -- not this binding --
    selects and prices the argmax-reward winner per group, so the step
    report's ``rows`` counts samples drawn, not the groups supervised
    over. The group's reward and identity columns reach the objective
    through the recorded batch schema, never through this binding's
    forward closure.

    WHAT IS NOT CLAIMED: that ``rows`` can recover the group count --
    that denominator lives inside the objective's metrics; or that the
    binding knows what N is.
    """
    objective = BestOfNLoss(weight=weight)
    return OnlineAlgorithm(
        name="best_of_n",
        objective=objective,
        rows_unit=RowsUnit.SAMPLED_COMPLETION,
    )
