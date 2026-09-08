"""The ``Algorithm`` contract: what an algorithm DECLARES, measured twice.

This module is the top of the rl contract stack (design section 3.1):
an algorithm declares the components it needs, and components it does
not consume are absent, never stubbed. ``AlgorithmRequirements`` states
that declaration ONCE -- which optional components the algorithm
consumes, and which names its objective and its metric channel will
decompose into -- and two gates read it: :func:`check_algorithm_wiring`
at setup, comparing the declaration against what the run was actually
HANDED, and :func:`verify_step` after every step, comparing it against
what the step actually OBSERVED. A declaration is what puts a quantity
in a denominator: ``declared_components`` is the objective's
denominator and ``declared_metrics`` is the metric channel's, and each
gate measures against the declaration it was given rather than
inferring one.

The algorithm owns step semantics ONLY -- the loop owns manifest
emission, save gating, and exit codes; the sizing argument lives on
the ``Algorithm`` protocol below. ``StepReport`` is the per-step
record: it holds the ``LossOutput`` itself rather than copying
``components``/``metrics`` up into its own fields, because two copies
of one countable is exactly the drift class this repository keeps
finding. Abstention is ``None`` everywhere here -- ``reward_stats``,
``sync`` -- never ``0.0`` and never ``0``.

WHAT THIS MODULE DOES NOT CLAIM: anything about whether a declaration
is CORRECT for the algorithm's mathematics -- a declaration puts a
quantity in a denominator and is not evidence the quantity is the
right one; anything about WHICH rows a step's denominator counted --
when an advantage function compacts a batch the survivors are named by
``AdvantageResult.rows``, not here; and anything about transports,
rollout mechanics, or loss internals -- those contracts are siblings,
consumed or absent, never re-stated here.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any, Literal, Protocol, runtime_checkable

from foundationscale.rl.advantage import AdvantageFn, RewardStats
from foundationscale.rl.interfaces import ExperienceBatch, LossFn, LossOutput
from foundationscale.rl.policy import PolicyPair
from foundationscale.rl.rollout import RolloutSource
from foundationscale.rl.weightsync import SyncReport, WeightSync

__all__ = (
    "Algorithm",
    "AlgorithmRequirements",
    "AlgorithmSemantics",
    "AlgorithmWiringRefusal",
    "StepReport",
    "StepReportRefusal",
    "check_algorithm_wiring",
    "verify_step",
)


class AlgorithmWiringRefusal(ValueError):
    # Raised when what an algorithm DECLARES it consumes and what it was
    # HANDED disagree -- in either direction. A required component that
    # is absent and an unrequired component that was supplied are the
    # same defect seen from two sides, and the second is the more
    # dangerous: a component the algorithm does not consume is a stub by
    # another name, and wiring that carries one reads as wiring that
    # works. CONSTRUCTION/CONFIGURATION-side only -- distinct from
    # StepReportRefusal, which is the call-time data half.
    pass


class StepReportRefusal(ValueError):
    # Raised when one step's report is internally incoherent, or does
    # not account for what the algorithm declared. Refused at call time,
    # not construction time: this is a fact about data a step produced,
    # not about how the run was wired. CALL-TIME DATA-side only --
    # distinct from AlgorithmWiringRefusal, which is the configuration
    # half.
    pass


def _checked_names(
    values: tuple[str, ...], *, field: str, refusal: type[ValueError]
) -> tuple[str, ...]:
    # Shared name-tuple validation for the two declarations: every
    # entry a non-empty str (absence of a name is not a name -- a gate
    # cannot grade a quantity it cannot name), no name twice (one
    # quantity, one denominator slot -- a duplicated name usually hides
    # the entry the caller meant to declare).
    names = tuple(values)
    seen: dict[str, int] = {}
    for index, name in enumerate(names):
        if not isinstance(name, str) or not name:
            raise refusal(
                f"{field}[{index}]={name!r}: every declared name must "
                f"be a non-empty str; absence of a name is not a name, "
                f"and a gate cannot grade a quantity it cannot name"
            )
        if name in seen:
            raise refusal(
                f"{field}[{index}] repeats {name!r}, first declared at "
                f"{field}[{seen[name]}]: one quantity, one denominator "
                f"slot -- a duplicated name usually hides the entry the "
                f"caller meant to declare"
            )
        seen[name] = index
    return names


def _first_repeat(names: tuple[str, ...]) -> str | None:
    # First duplicated name, or None. A repeat must be caught BEFORE
    # any set comparison, because two entries with one name collapse
    # into one frozenset member and the comparison would hide it.
    seen: set[str] = set()
    for name in names:
        if name in seen:
            return name
        seen.add(name)
    return None


def _set_disagreement(
    left: tuple[str, ...],
    right: tuple[str, ...],
    *,
    left_label: str,
    right_label: str,
) -> str | None:
    # Both directions of a name-set disagreement, each against ITS OWN
    # denominator, or None when the two sides name the same set. Two
    # sets of equal size can still be different sets, so a message
    # built from counts alone can state no reason at all -- the names
    # are what make the refusal actionable and the denominators are
    # what stop a bare count being read against the wrong total.
    #
    # One helper rather than four copies of the count-and-name idiom:
    # a restated countable is the drift class this repository keeps
    # filing (findings #233, #243), and a comparison helper that has
    # drifted reports agreement it never measured.
    #
    # Both sides must already be duplicate-free -- every caller runs
    # _first_repeat first -- because a repeat collapses in frozenset
    # and would make len(left) overstate its own denominator.
    left_set = frozenset(left)
    right_set = frozenset(right)
    if left_set == right_set:
        return None
    absent_right = tuple(n for n in left if n not in right_set)
    absent_left = tuple(n for n in right if n not in left_set)
    absent_right_names = ", ".join(absent_right) or "none"
    absent_left_names = ", ".join(absent_left) or "none"
    return (
        f"{len(absent_right)} of {len(left)} {left_label} are not "
        f"{right_label} ({absent_right_names}) and "
        f"{len(absent_left)} of {len(right)} {right_label} are not "
        f"{left_label} ({absent_left_names})"
    )


@dataclass(frozen=True, slots=True)
class AlgorithmSemantics:
    """The algorithm's second, independent declaration: the semantics it
    consumes, stated beside -- never instead of -- the role-presence layer.

    The ``requires`` mapping on ``AlgorithmRequirements`` answers "is a
    rollout source wired in?". It cannot answer "does this loss and this
    algorithm agree about what a group is?", because presence says nothing
    about geometry. This record carries the second question's answers:
    group size K, ratio geometry, KL estimator, clip bounds, and
    reference-freeness. Every field defaults to ``None`` and ``None``
    means exactly one thing: THIS ALGORITHM DOES NOT CONSTRAIN THAT SEAM.
    ``None`` is an abstention, not a default and not a measurement --
    ``group_size=0`` is refused below, because a measured zero posing as
    an unconstrained seam is the inversion of this repository's rule that
    ``0.0`` means measured and ``None`` means unmeasured.

    The declaration exists so that TWO objects state each seam -- this
    record on the algorithm side, and a matching declaration on the loss
    side -- and a disagreement between two independent statements can be
    refused. One shared constant has no second side to disagree with, and
    a loss that read its geometry from the algorithm would believe it and
    train wrongly.

    ``ratio_scope`` is loss geometry (design record D6): the advantage
    function never sees logprobs and cannot carry it. The closed union
    ``("token", "sequence")`` covers the algorithm families rated in the
    design record; whether a third geometry needs a member is a stated
    open question there, not a gap closed here by permissiveness.

    Mis-stated fields refuse at construction with
    ``AlgorithmWiringRefusal`` -- a bad semantics declaration is a
    configuration-side defect knowable before step one, in the same class
    as a bad ``requires`` entry, and refusing there costs nothing while
    refusing at step zero costs an allocation.

    WHAT IS CLAIMED: these five fields are the algorithm's own statements
    about the semantic seams it consumes, each optional, each
    independently stated, with ``None`` as the abstention value on every
    seam.

    WHAT IS NOT CLAIMED: that the stated values are CORRECT for the
    algorithm's mathematics -- a declaration puts a seam in a denominator
    and is not evidence the value is the right one; that ``ratio_scope``
    is a permanently closed union -- ``token | sequence`` covers the rated
    families and no third geometry is either claimed or precluded; and
    that any gate COMPARES this record today -- ``check_algorithm_wiring``
    does not read it, because no loss-side semantics declaration exists on
    ``LossDeclaration`` yet and whether that comparison should be
    structural or advisory is a live open question in the design record.
    """

    group_size: int | None = None
    ratio_scope: Literal["token", "sequence"] | None = None
    kl_estimator: str | None = None
    clip_bounds: tuple[float, float] | None = None
    reference_free: bool | None = None

    def __post_init__(self) -> None:
        # bool is checked BEFORE int because isinstance(True, int) is
        # True; a True group size would sort as 1 and read as K=1.
        if self.group_size is not None:
            if isinstance(self.group_size, bool):
                raise AlgorithmWiringRefusal(
                    f"group_size={self.group_size!r}: True is not a group "
                    f"size -- K is a positive int, or None to abstain"
                )
            if not isinstance(self.group_size, int) or self.group_size < 1:
                raise AlgorithmWiringRefusal(
                    f"group_size={self.group_size!r}: K is a positive int, "
                    f"or None to abstain (this algorithm does not constrain "
                    f"group size) -- 0 is not an unconstrained seam, it is a "
                    f"measured zero posing as an abstention, and abstention "
                    f"here is None, never 0"
                )
        if self.ratio_scope is not None and self.ratio_scope not in ("token", "sequence"):
            raise AlgorithmWiringRefusal(
                f"ratio_scope={self.ratio_scope!r}: the declared union is "
                f"('token', 'sequence') and covers the rated algorithm "
                f"families; None abstains. A third geometry is a stated open "
                f"question in the design record -- it is not closed here by "
                f"admitting an unrated string"
            )
        if self.kl_estimator is not None and (
            not isinstance(self.kl_estimator, str) or not self.kl_estimator
        ):
            raise AlgorithmWiringRefusal(
                f"kl_estimator={self.kl_estimator!r}: an estimator is named "
                f"by a non-empty str, or None to abstain -- absence of a "
                f"name is not a name, and a gate cannot grade a seam it "
                f"cannot name"
            )
        if self.clip_bounds is not None:
            bounds = tuple(self.clip_bounds)
            if len(bounds) != 2:
                raise AlgorithmWiringRefusal(
                    f"clip_bounds={self.clip_bounds!r}: a (low, high) pair, "
                    f"got {len(bounds)} entries -- or None to abstain"
                )
            low, high = bounds
            if (
                isinstance(low, bool)
                or isinstance(high, bool)
                or not isinstance(low, (int, float))
                or not isinstance(high, (int, float))
            ):
                raise AlgorithmWiringRefusal(
                    f"clip_bounds={self.clip_bounds!r}: both bounds are real "
                    f"numbers -- a clip bound is a measured quantity, and an "
                    f"untypable one cannot be compared against any loss-side "
                    f"declaration"
                )
            if not low < high:
                raise AlgorithmWiringRefusal(
                    f"clip_bounds=({low!r}, {high!r}): low must be strictly "
                    f"below high -- a reversed or degenerate interval clips "
                    f"everything or nothing and either way the declaration "
                    f"describes no real objective"
                )
            object.__setattr__(self, "clip_bounds", bounds)
        if self.reference_free is not None and not isinstance(self.reference_free, bool):
            raise AlgorithmWiringRefusal(
                f"reference_free={self.reference_free!r}: True or False, or "
                f"None to abstain -- a truthy stand-in would let an "
                f"unmeasured value pose as a declaration"
            )


@dataclass(frozen=True, slots=True)
class AlgorithmRequirements:
    """What an algorithm states it consumes and what it will put in the
    objective's denominator, stated ONCE.

    One declaration, read by both :func:`check_algorithm_wiring` (does
    the wiring the algorithm was handed match what it declared?) and
    :func:`verify_step` (does the step the algorithm produced account
    for what it declared?). Design section 3.1: an algorithm declares
    the components it needs, and components it does not consume are
    absent, never stubbed.

    ``requires`` is that declaration's role-presence half, as DATA: a
    mapping from role name (``"rollout_source"``, ``"advantage_fn"``,
    ``"weight_sync"``, ``"reference_policy"``, and any role a new family
    adds -- a critic, a reward model) to ``True`` (the algorithm
    consumes it) or ``False`` (it must not be wired). Data rather than a
    fixed field per role because the wiring check's denominator is built
    from these keys unioned with the supplied side's keys: a new role
    enters the denominator by being NAMED, not by a core edit, and a
    core edit is exactly where a hard-coded enumeration stays silently
    stale. The mapping may NOT be empty: an algorithm that declares no
    roles at all puts its whole wiring outside the check's denominator
    and every agreement over it would be vacuous -- ``all([]) is True``
    is this repository's founding defect and is refused below. An
    algorithm that consumes no optional component names its roles with
    ``False``.

    ``declared_components`` names the loss terms the objective
    decomposes into and may NOT be empty. ``declared_metrics`` names the
    readings the loss does not optimise and MAY be empty: the #316
    metric channel answers the empty case with a declared SKIP, so
    absence stays visible in the denominator.

    ``semantics`` is the second, independent declaration (design record
    D1): the semantic seams the algorithm consumes. It MAY be ``None``
    -- an algorithm that fixes no group size, no ratio geometry, no KL
    estimator, no clip bounds, and no reference-freeness stance
    CONSTRAINS NOTHING and abstains, rather than posing a default the
    mathematics never stated.

    WHAT IS CLAIMED: these are the algorithm's own declarations, and
    both ``check_algorithm_wiring`` and ``verify_step`` measure against
    them.

    WHAT IS NOT CLAIMED: anything about whether the declaration is
    CORRECT for the algorithm's mathematics -- a declaration puts a
    quantity in a denominator and is not evidence that the quantity is
    the right one; and any opinion about role names this repository has
    not seen -- cross-checks exist only for the named pairings below,
    and a role no check knows is measured for presence only, never for
    coherence.
    """

    name: str
    requires: Mapping[str, bool]
    declared_components: tuple[str, ...]
    declared_metrics: tuple[str, ...] = ()
    semantics: AlgorithmSemantics | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise AlgorithmWiringRefusal(
                f"name={self.name!r}: an algorithm with no name cannot be "
                f"attributed in a report or a manifest"
            )
        if not isinstance(self.requires, Mapping):
            raise AlgorithmWiringRefusal(
                f"requires={self.requires!r}: the role declaration is a "
                f"Mapping of role name to bool, not a "
                f"{type(self.requires).__name__} -- a bare list of names "
                f"cannot say False, and False is a declaration this "
                f"surface must be able to refuse against"
            )
        if not self.requires:
            raise AlgorithmWiringRefusal(
                f"{self.name} declares an EMPTY requires mapping: 0 roles "
                f"named on the declaration side. A wiring check over zero "
                f"roles has an empty denominator and every agreement over "
                f"it is vacuous -- all([]) is True is this repository's "
                f"founding defect. Name every role the algorithm knows "
                f"about; an algorithm that consumes no optional component "
                f"declares its roles with False, never with silence"
            )
        frozen_requires: dict[str, bool] = {}
        for role, value in self.requires.items():
            if not isinstance(role, str) or not role:
                raise AlgorithmWiringRefusal(
                    f"requires has a role named {role!r}: every role name "
                    f"must be a non-empty str -- absence of a name is not "
                    f"a name, and a wiring check cannot grade a role it "
                    f"cannot name"
                )
            if not isinstance(value, bool):
                raise AlgorithmWiringRefusal(
                    f"requires[{role!r}]={value!r}: role requirements are "
                    f"booleans -- True declares the role consumed, False "
                    f"declares it refused-if-supplied; a "
                    f"{type(value).__name__} stand-in would let an "
                    f"unmeasured value pose as a declaration"
                )
            frozen_requires[role] = value
        object.__setattr__(self, "requires", frozen_requires)
        declared_components = tuple(self.declared_components)
        if not declared_components:
            raise AlgorithmWiringRefusal(
                f"{self.name} declares NO loss components: an algorithm that "
                f"declares no loss component puts its whole objective outside "
                f"the gate's denominator, and every coverage check over it "
                f"would pass vacuously -- by measuring nothing"
            )
        object.__setattr__(
            self,
            "declared_components",
            _checked_names(
                declared_components, field="declared_components", refusal=AlgorithmWiringRefusal
            ),
        )
        # declared_metrics MAY be empty -- the #316 metric channel
        # answers the empty case with a declared SKIP, so absence stays
        # visible in the denominator. Only its entries need checking.
        object.__setattr__(
            self,
            "declared_metrics",
            _checked_names(
                tuple(self.declared_metrics),
                field="declared_metrics",
                refusal=AlgorithmWiringRefusal,
            ),
        )
        metric_names = frozenset(self.declared_metrics)
        both = tuple(n for n in self.declared_components if n in metric_names)
        if both:
            raise AlgorithmWiringRefusal(
                f"{len(both)} names appear in BOTH declared_components "
                f"({len(self.declared_components)}) and declared_metrics "
                f"({len(self.declared_metrics)}): {', '.join(both)}; a "
                f"component is a term OF the loss and a metric is a reading "
                f"the loss does not optimise -- a name in both is the same "
                f"quantity in two denominators, and the two gates that read "
                f"those denominators would each grade it as covered"
            )
        if self.semantics is not None and not isinstance(self.semantics, AlgorithmSemantics):
            raise AlgorithmWiringRefusal(
                f"semantics={self.semantics!r}: None (the algorithm "
                f"constrains no seam) or an AlgorithmSemantics -- anything "
                f"else is a declaration this contract cannot read"
            )
        # The converse is deliberately NOT refused: requires
        # "advantage_fn" without "rollout_source" is a legitimate
        # offline shape -- advantages over rollouts a previous run
        # generated -- and refusing it would make offline RL
        # unrepresentable. These two pairings are the only role-name
        # cross-checks this class makes; a role name it has not seen is
        # measured for presence by check_algorithm_wiring and judged no
        # further here.
        if self.requires.get("weight_sync", False) and not self.requires.get(
            "rollout_source", False
        ):
            raise AlgorithmWiringRefusal(
                f"{self.name} declares the weight_sync role without the "
                f"rollout_source role: a weight sync moves the training "
                f"view into the generation view, and with no rollout there "
                f"is no generation view to move it to -- the sync has no "
                f"destination and a report of one describes a transfer to "
                f"nowhere"
            )


@dataclass(frozen=True, slots=True)
class StepReport:
    """One step's record: what the objective was, over how many rows,
    plus the reward statistics and weight sync if the algorithm
    consumes those.

    Deliberate NON-restatement: ``StepReport`` holds the ``LossOutput``
    itself and does NOT copy ``components``/``metrics`` up into its own
    fields. Two copies of one countable is exactly the drift class this
    repository keeps finding; the report holds the observation, not a
    second edition of it.

    Deliberate NON-refusal: a NaN or infinite ``loss.loss`` ROUND-TRIPS
    and is not refused. A diverged step is the single event a step
    record most needs to carry, and a validator that deleted it would
    make divergence unrepresentable -- the report would go quiet at
    exactly the moment it is needed. This is the opposite call from
    ``SyncReport.seconds``, where NaN IS refused, and the difference is
    that ``seconds`` is an instrument reading while ``loss`` is the
    observation.

    ``reward_stats`` and ``sync`` abstain as ``None``, never ``0`` and
    never a zero-count stand-in. This class has NO properties and NO
    derived quantities -- no ``ok``, no ``loss_per_row``: a report that
    grades itself is a detector inside its own denominator, and an
    aggregate over steps is not a property of one step.

    WHAT IS CLAIMED: ``rows`` is the denominator the gradient was
    priced against.

    WHAT IS NOT CLAIMED: anything about WHICH rows. When an advantage
    function compacts a batch, ``AdvantageResult.rows`` names the
    survivors, and a caller zipping positionally needs that, not this
    count.
    """

    step: int
    loss: LossOutput
    rows: int
    reward_stats: RewardStats | None = None
    sync: SyncReport | None = None

    def __post_init__(self) -> None:
        # bool is checked BEFORE int because isinstance(True, int) is
        # True; a True step index would sort as 1 and read as step one.
        if isinstance(self.step, bool):
            raise StepReportRefusal(f"step={self.step!r}: True is not a step index")
        if not isinstance(self.step, int) or self.step < 0:
            raise StepReportRefusal(f"step={self.step!r}: a step index is an int >= 0")
        if isinstance(self.rows, bool):
            raise StepReportRefusal(f"rows={self.rows!r}: True is not a row count")
        if not isinstance(self.rows, int):
            raise StepReportRefusal(f"rows={self.rows!r}: a row count is an int")
        if self.rows < 1:
            raise StepReportRefusal(
                f"rows={self.rows}: zero is not a denominator -- a step over "
                f"zero rows measured nothing, and returning it lets a caller "
                f"divide by it or read it as a step with nothing to do"
            )
        if not isinstance(self.loss, LossOutput):
            raise StepReportRefusal(
                f"loss={self.loss!r}: not a LossOutput -- a step report "
                f"carries the objective it was priced against, not a "
                f"stand-in for one"
            )
        if not self.loss.components:
            raise StepReportRefusal(
                "loss.components is EMPTY: a step whose objective decomposed "
                "into zero components has its whole objective outside the "
                "coverage gate's denominator. LossOutput permits the empty "
                "tuple because a loss object is not a step; a STEP that "
                "reports one is unmeasured"
            )
        if self.reward_stats is not None and not isinstance(self.reward_stats, RewardStats):
            raise StepReportRefusal(
                f"reward_stats={self.reward_stats!r}: None (unconsumed) or a "
                f"RewardStats -- anything else is a summary this contract "
                f"cannot read"
            )
        if self.reward_stats is not None and self.reward_stats.count != self.rows:
            # RewardStats summarises the samples the advantage function
            # ACTUALLY USED and rows is the gradient's denominator. If a
            # real algorithm ever needs these to differ, the contract is
            # drawn wrong and must be widened deliberately -- not by
            # relaxing this check.
            raise StepReportRefusal(
                f"reward_stats summarises {self.reward_stats.count} samples "
                f"but step {self.step} was priced against {self.rows} rows "
                f"({self.reward_stats.count} of {self.rows}): if they "
                f"differ, one step carries two denominators and each gate "
                f"prices against whichever it happens to read"
            )
        if self.sync is not None and not isinstance(self.sync, SyncReport):
            raise StepReportRefusal(
                f"sync={self.sync!r}: None (no sync this step) or a "
                f"SyncReport -- anything else is a transfer record this "
                f"contract cannot read"
            )


@runtime_checkable
class Algorithm(Protocol):
    """Owns step semantics ONLY -- nothing wider, nothing narrower.

    The loop owns manifest emission, save gating, and exit codes
    (design section 3.1). One level wider -- the algorithm owns the
    loop, the manifest, the save path -- and the adjudication plane is
    bypassed; one level narrower -- the algorithm is only a loss
    function -- and rollout-owning algorithms like GRPO have nowhere to
    live. The optional parameters of ``setup`` default to ``None`` and
    are ABSENT when unconsumed -- never stubbed.

    ``@runtime_checkable`` checks method PRESENCE only, never
    signatures, so ``isinstance`` here is a structural smoke test and
    not the handshake. The handshake is :func:`check_algorithm_wiring`.
    """

    def requirements(self) -> AlgorithmRequirements: ...

    def setup(
        self,
        *,
        policy_pair: PolicyPair,
        loss_fn: LossFn,
        dataloader: Iterable[ExperienceBatch],
        config: Mapping[str, Any],
        rollout_source: RolloutSource | None = None,
        advantage_fn: AdvantageFn | None = None,
        weight_sync: WeightSync | None = None,
    ) -> None: ...

    def step(self) -> StepReport: ...


def check_algorithm_wiring(
    requirements: AlgorithmRequirements,
    *,
    policy_pair: PolicyPair,
    loss_fn: LossFn,
    supplied: Mapping[str, Any],
    origin: str = "<algorithm>",
) -> tuple[str, ...]:
    """The setup handshake: the declaration measured against the wiring.

    WHAT IS CLAIMED on success: every role named in the UNION of the
    requirements' ``requires`` keys and the caller's ``supplied`` keys
    agrees in both directions -- required roles are present, unconsumed
    roles are absent; the reference-policy requirement and the pair's
    ``references`` agree in BOTH directions; a rollout-owning algorithm
    was handed a pair with a generation view; and the algorithm's
    ``declared_components`` / ``declared_metrics`` ARE the names the
    wired loss's own ``declaration()`` states. The returned tuple is the
    SORTED set of role names the declaration marks consumed -- the
    roles, not the count of arguments passed, are what a caller records,
    and the set is open: a role this repository has never heard of (a
    critic, a reward model) flows through the same both-direction check
    with no core edit.

    WHAT IS NOT CLAIMED: that any supplied component WORKS -- the only
    thing this function calls is ``loss_fn.declaration()``, which is by
    contract the statement a run RECORDS and not a forward pass; nothing
    about signatures -- the runtime protocol check sees method presence
    only, which is why this handshake exists; anything about role
    COHERENCE beyond the named pairings (weight sync needs a
    destination, rollouts need a generation view) -- an unknown role is
    measured for presence only; and anything about the ``semantics``
    declaration -- that record is not read here, because no loss-side
    semantics declaration exists to compare it against yet and whether
    that comparison is structural or advisory is a stated open question
    in the design record.

    ORDERING, deliberately changed from the field-per-role surface: the
    mapping half of this check reads nothing off the pair, so it runs
    BEFORE the pair's type is established -- a role disagreement is
    knowable from the two mappings alone, and the pair's views are only
    ever read after ``isinstance`` has settled what they are. The
    refusal that names its disagreement still precedes any read that
    could not.

    The loss cross-check is here rather than left to
    :func:`verify_step` because the two are the SAME countable declared
    twice, in two objects, and the step-time check can only speak after
    an allocation has been burned -- the class findings #170 and #183
    filed against the launch plane. At setup it costs one method call.

    The component-role check collects EVERY disagreement across the
    union and refuses ONCE: a refusal that stops at the first role
    makes the operator rediscover the same class once per role.

    Deliberate NON-refusal: a pair carrying a ``generate_view`` that
    the algorithm does not require is NOT refused, while unrequired
    ``references`` are. The asymmetry is real, not an oversight. A
    generation view is a VIEW onto weights the pair already holds, and
    one pair is legitimately handed to a sequence of phases -- an SFT
    warm-up then a policy-gradient phase -- so refusing it would force
    the pair to be torn down and rebuilt between phases. A reference is
    a SECOND, separately loaded frozen model whose only purpose is to
    price an objective; if no objective prices against it, it is a
    loaded model that nothing reads.
    """
    requires = requirements.requires
    role_names = frozenset(requires) | frozenset(supplied)
    if not role_names:
        raise AlgorithmWiringRefusal(
            f"{origin} wired {requirements.name} but the wiring denominator "
            f"is empty: 0 roles named in requires ({len(requires)} entries) "
            f"and 0 roles named in supplied ({len(supplied)} entries). A "
            f"wiring check whose denominator is the empty set reports CLEAR "
            f"over nothing, and every agreement over it passes vacuously -- "
            f"all([]) is True remains the founding defect this surface "
            f"exists to refuse"
        )
    for role in supplied:
        if not isinstance(role, str) or not role:
            raise AlgorithmWiringRefusal(
                f"{origin} supplied a role named {role!r}: every role name "
                f"must be a non-empty str -- absence of a name is not a "
                f"name, and a wiring check cannot grade a role it cannot "
                f"name"
            )
    if supplied.get("reference_policy") is not None:
        raise AlgorithmWiringRefusal(
            f"{origin} handed {requirements.name} a supplied mapping that "
            f"names reference_policy: a reference policy's supplied side is "
            f"the policy pair's references, and stating it in a second "
            f"place is one countable declared in two objects. Remove the "
            f"mapping entry and let the pair attest"
        )
    generic_roles = tuple(sorted(role_names - {"reference_policy"}))
    disagreements: list[str] = []
    for role in generic_roles:
        required = requires.get(role, False)
        provided = supplied.get(role)
        if required and provided is None:
            disagreements.append(
                f"{role} is required by {requirements.name} but {origin} supplied none"
            )
        elif not required and provided is not None:
            disagreements.append(
                f"{role} was supplied by {origin} but {requirements.name} does "
                f"not consume it -- a component the algorithm does not consume "
                f"is absent, never stubbed, and a supplied-but-unconsumed "
                f"component is a stub by another name that reads as wiring "
                f"that works"
            )
    if disagreements:
        raise AlgorithmWiringRefusal(
            # The denominator is generic_roles, NOT role_names. reference_policy
            # is in role_names but the loop above cannot reach it -- it is graded
            # against the pair's references further down, by its own refusal. A
            # count stated over a set one member larger than the set it ranges
            # over understates the disagreement rate on every algorithm that
            # declares a reference policy, which is most of them.
            f"{len(disagreements)} of {len(generic_roles)} component roles "
            f"disagree between what {requirements.name} declares and what "
            f"{origin} was handed: {'; '.join(disagreements)}"
        )
    if not isinstance(policy_pair, PolicyPair):
        raise AlgorithmWiringRefusal(
            f"{origin} wired {requirements.name} against a "
            f"{type(policy_pair).__name__} rather than a PolicyPair: every "
            f"other refusal here names the disagreement it found, and reading "
            f"the views off a foreign object would raise an AttributeError "
            f"that names nothing"
        )
    wants_reference = requires.get("reference_policy", False)
    if wants_reference and not policy_pair.references:
        raise AlgorithmWiringRefusal(
            f"{origin} wired {requirements.name}, which declares a reference "
            f"policy, but the policy pair carries no references -- the model "
            f"the objective is priced against is absent"
        )
    if not wants_reference and policy_pair.references:
        keys = tuple(policy_pair.references)
        raise AlgorithmWiringRefusal(
            f"{origin} wired {requirements.name}, which declares no reference "
            f"policy, but the policy pair carries {len(keys)} references "
            f"({', '.join(keys)}): an unconsumed frozen model is a stub that "
            f"reads as a working DPO wiring"
        )
    if requires.get("rollout_source", False) and policy_pair.generate_view is None:
        raise AlgorithmWiringRefusal(
            f"{origin} wired {requirements.name}, which declares rollouts, "
            f"but the policy pair has no generation view -- a rollout source "
            f"has nothing to generate from. This is a cross-check between "
            f"two contracts that neither can make alone: the pair does not "
            f"know what the algorithm declares, and the algorithm does not "
            f"know what the pair carries"
        )
    declaration = loss_fn.declaration()
    repeated_loss_component = _first_repeat(declaration.components)
    if repeated_loss_component is not None:
        raise AlgorithmWiringRefusal(
            f"{origin} wired {requirements.name} to a loss whose own "
            f"declaration names the component {repeated_loss_component!r} "
            f"twice -- the loss side of this comparison has no unambiguous "
            f"denominator, so no agreement with it can be measured"
        )
    component_disagreement = _set_disagreement(
        requirements.declared_components,
        declaration.components,
        left_label="components the algorithm declares",
        right_label="components the loss declares",
    )
    if component_disagreement is not None:
        raise AlgorithmWiringRefusal(
            f"{origin} wired {requirements.name} to a loss that declares a "
            f"different objective: {component_disagreement}. These are ONE "
            f"countable declared twice, in two objects, and until they agree "
            f"each gate prices against whichever declaration it happens to "
            f"read"
        )
    loss_metric_names = tuple(m.name for m in declaration.metrics)
    repeated_loss_metric = _first_repeat(loss_metric_names)
    if repeated_loss_metric is not None:
        raise AlgorithmWiringRefusal(
            f"{origin} wired {requirements.name} to a loss whose own "
            f"declaration names the metric {repeated_loss_metric!r} twice -- "
            f"the loss side of this comparison has no unambiguous "
            f"denominator, so no agreement with it can be measured"
        )
    metric_disagreement = _set_disagreement(
        requirements.declared_metrics,
        loss_metric_names,
        left_label="metrics the algorithm declares",
        right_label="metrics the loss declares",
    )
    if metric_disagreement is not None:
        raise AlgorithmWiringRefusal(
            f"{origin} wired {requirements.name} to a loss that declares "
            f"different diagnostics: {metric_disagreement}. An undeclared "
            f"metric is refused outright by the #316 channel and a declared "
            f"one the loss never produces reads as a covered quantity, so "
            f"the two declarations have to be the same set before step one"
        )
    return tuple(sorted(role for role in role_names if requires.get(role, False)))


def verify_step(
    report: StepReport,
    requirements: AlgorithmRequirements,
    *,
    origin: str = "<algorithm>",
) -> int:
    """The per-step measurement: the step's observation against the declaration.

    WHAT IS CLAIMED on success: the observed component names ARE the
    declared component names (as sets, duplicate-free on both sides),
    the observed metric names ARE the declared metric names, reward
    statistics are present exactly when the algorithm consumes an
    advantage function, and no weight sync was reported by an algorithm
    that consumes none. The returned ``int`` is the number of DECLARED
    loss components the step OBSERVED -- the observed count MEASURED,
    not the declared count restated. Returning
    ``len(report.loss.components)`` rather than
    ``len(requirements.declared_components)`` is only meaningful
    because the set comparisons below have already proved the two sets
    equal, so the denominator downstream uses is the count this step
    actually produced.

    WHAT IS NOT CLAIMED: that a sync happened -- a weight sync runs on
    a CADENCE, so a missing ``sync`` under a declared ``weight_sync``
    role is a step between syncs and passes; that the loss is finite --
    a diverged step round-trips by design (see ``StepReport``); and
    anything about WHICH rows ``rows`` counted.
    """
    observed_components = tuple(c.name for c in report.loss.components)
    repeated_component = _first_repeat(observed_components)
    if repeated_component is not None:
        raise StepReportRefusal(
            f"{origin}: step {report.step} from {requirements.name} reports "
            f"the component {repeated_component!r} twice -- two entries with "
            f"one name collapse in any set comparison, so this is checked "
            f"before the set comparison that would hide it"
        )
    component_disagreement = _set_disagreement(
        observed_components,
        requirements.declared_components,
        left_label="observed components",
        right_label="declared components",
    )
    if component_disagreement is not None:
        raise StepReportRefusal(
            f"{origin}: step {report.step} from {requirements.name} does not "
            f"account for its declared objective: {component_disagreement}. "
            f"Two sets of equal size can still be different sets, so the "
            f"counts alone do not settle it -- declaration is what puts a "
            f"quantity in a denominator (design section 9 stage 2, "
            f"conditions (a) and (c))"
        )
    observed_metrics = tuple(m.name for m in report.loss.metrics)
    repeated_metric = _first_repeat(observed_metrics)
    if repeated_metric is not None:
        raise StepReportRefusal(
            f"{origin}: step {report.step} from {requirements.name} reports "
            f"the metric {repeated_metric!r} twice -- two entries with one "
            f"name collapse in any set comparison, so this is checked before "
            f"the set comparison that would hide it"
        )
    # Both sides empty is agreement and must PASS: _set_disagreement
    # answers two empty tuples with None, and the #316 metric channel
    # answers a declared empty metric set with a declared SKIP, not a
    # refusal.
    metric_disagreement = _set_disagreement(
        observed_metrics,
        requirements.declared_metrics,
        left_label="observed metrics",
        right_label="declared metrics",
    )
    if metric_disagreement is not None:
        raise StepReportRefusal(
            f"{origin}: step {report.step} from {requirements.name} does not "
            f"account for its declared metrics: {metric_disagreement}. Two "
            f"sets of equal size can still be different sets, so the counts "
            f"alone do not settle it"
        )
    if report.reward_stats is not None and not requirements.requires.get("advantage_fn", False):
        raise StepReportRefusal(
            f"{origin}: step {report.step} from {requirements.name} reports "
            f"reward statistics ({report.reward_stats.count} samples) but "
            f"the algorithm declares no advantage function -- an algorithm "
            f"that consumes no advantage function has no reward statistics "
            f"to report, and a reported one came from a path the contract "
            f"does not describe"
        )
    if report.reward_stats is None and requirements.requires.get("advantage_fn", False):
        raise StepReportRefusal(
            f"{origin}: step {report.step} from {requirements.name} reports "
            f"no reward statistics though the algorithm declares an "
            f"advantage function -- every step that consumed one summarised "
            f"the samples it used (rows >= 1), so there is no zero-sample "
            f"state in which the summary could be legitimately absent"
        )
    if report.sync is not None and not requirements.requires.get("weight_sync", False):
        raise StepReportRefusal(
            f"{origin}: step {report.step} from {requirements.name} reports "
            f"a weight sync (transport {report.sync.transport!r}) but the "
            f"algorithm declares no weight sync -- a reported one came from "
            f"a path the contract does not describe"
        )
    # Deliberate asymmetry with the reward-statistics checks above: a
    # MISSING sync under a declared weight_sync role is NOT refused.
    # Reward statistics are produced by every step that consumes an
    # advantage function, whereas a weight sync happens on a CADENCE. A
    # step with no sync is a step BETWEEN syncs, not a step that lost
    # one, and refusing it would make every cadence other than
    # every-step unrepresentable.
    return len(report.loss.components)
