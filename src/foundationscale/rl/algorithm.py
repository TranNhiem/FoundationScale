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
from typing import Any, Protocol, runtime_checkable

from foundationscale.rl.advantage import AdvantageFn, RewardStats
from foundationscale.rl.interfaces import ExperienceBatch, LossFn, LossOutput
from foundationscale.rl.policy import PolicyPair
from foundationscale.rl.rollout import RolloutSource
from foundationscale.rl.weightsync import SyncReport, WeightSync

__all__ = (
    "Algorithm",
    "AlgorithmRequirements",
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
class AlgorithmRequirements:
    """What an algorithm states it consumes and what it will put in the
    objective's denominator, stated ONCE.

    One declaration, read by both :func:`check_algorithm_wiring` (does
    the wiring the algorithm was handed match what it declared?) and
    :func:`verify_step` (does the step the algorithm produced account
    for what it declared?). Design section 3.1: an algorithm declares
    the components it needs, and components it does not consume are
    absent, never stubbed. The four ``requires_*`` flags name the
    optional components. ``declared_components`` names the loss terms
    the objective decomposes into and may NOT be empty -- an algorithm
    that declares no loss component puts its whole objective outside
    the gate's denominator. ``declared_metrics`` names the readings the
    loss does not optimise and MAY be empty: the #316 metric channel
    answers the empty case with a declared SKIP, so absence stays
    visible in the denominator rather than being hidden by a check over
    nothing.

    WHAT IS CLAIMED: these are the algorithm's own declarations, and
    both ``check_algorithm_wiring`` and ``verify_step`` measure against
    them.

    WHAT IS NOT CLAIMED: anything about whether the declaration is
    CORRECT for the algorithm's mathematics. A declaration is what puts
    a quantity in a denominator; it is not evidence that the quantity
    is the right one.
    """

    name: str
    requires_rollout: bool
    requires_advantage: bool
    requires_weight_sync: bool
    requires_reference_policy: bool
    declared_components: tuple[str, ...]
    declared_metrics: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise AlgorithmWiringRefusal(
                f"name={self.name!r}: an algorithm with no name cannot be "
                f"attributed in a report or a manifest"
            )
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
        # The converse is deliberately NOT refused: requires_advantage
        # without requires_rollout is a legitimate offline shape --
        # advantages over rollouts a previous run generated -- and
        # refusing it would make offline RL unrepresentable.
        if self.requires_weight_sync and not self.requires_rollout:
            raise AlgorithmWiringRefusal(
                f"{self.name} declares requires_weight_sync without "
                f"requires_rollout: a weight sync moves the training view "
                f"into the generation view, and with no rollout there is no "
                f"generation view to move it to -- the sync has no "
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
    rollout_source: RolloutSource | None = None,
    advantage_fn: AdvantageFn | None = None,
    weight_sync: WeightSync | None = None,
    origin: str = "<algorithm>",
) -> tuple[str, ...]:
    """The setup handshake: the declaration measured against the wiring.

    WHAT IS CLAIMED on success: every optional component the algorithm
    requires was supplied, none it does not require was, the reference
    policy requirement and the pair's ``references`` agree in BOTH
    directions, a rollout-owning algorithm was handed a pair with a
    generation view, and the algorithm's ``declared_components`` /
    ``declared_metrics`` ARE the names the wired loss's own
    ``declaration()`` states. The returned tuple is the SORTED set of
    optional role names actually consumed -- a subset of
    ("advantage_fn", "reference_policy", "rollout_source",
    "weight_sync"). That tuple, not the count of arguments passed, is
    what a caller records.

    WHAT IS NOT CLAIMED: that any supplied component WORKS -- the only
    thing this function calls is ``loss_fn.declaration()``, which is by
    contract the statement a run RECORDS and not a forward pass; and
    nothing about signatures -- the runtime protocol check sees method
    presence only, which is why this handshake exists.

    The loss cross-check is here rather than left to
    :func:`verify_step` because the two are the SAME countable declared
    twice, in two objects, and the step-time check can only speak after
    an allocation has been burned -- the class findings #170 and #183
    filed against the launch plane. At setup it costs one method call.

    The component-role check collects EVERY disagreement across the
    three roles and refuses ONCE: a refusal that stops at the first
    role makes the operator rediscover the same class three times.

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
    if not isinstance(policy_pair, PolicyPair):
        raise AlgorithmWiringRefusal(
            f"{origin} wired {requirements.name} against a "
            f"{type(policy_pair).__name__} rather than a PolicyPair: every "
            f"other refusal here names the disagreement it found, and reading "
            f"the views off a foreign object would raise an AttributeError "
            f"that names nothing"
        )
    roles: tuple[tuple[str, bool, Any], ...] = (
        ("rollout_source", requirements.requires_rollout, rollout_source),
        ("advantage_fn", requirements.requires_advantage, advantage_fn),
        ("weight_sync", requirements.requires_weight_sync, weight_sync),
    )
    disagreements: list[str] = []
    for role, required, supplied in roles:
        if required and supplied is None:
            disagreements.append(
                f"{role} is required by {requirements.name} but {origin} supplied none"
            )
        elif not required and supplied is not None:
            disagreements.append(
                f"{role} was supplied by {origin} but {requirements.name} does "
                f"not consume it -- a component the algorithm does not consume "
                f"is absent, never stubbed, and a supplied-but-unconsumed "
                f"component is a stub by another name that reads as wiring "
                f"that works"
            )
    if disagreements:
        raise AlgorithmWiringRefusal(
            f"{len(disagreements)} of {len(roles)} component roles disagree "
            f"between what {requirements.name} declares and what {origin} was "
            f"handed: {'; '.join(disagreements)}"
        )
    if requirements.requires_reference_policy and not policy_pair.references:
        raise AlgorithmWiringRefusal(
            f"{origin} wired {requirements.name}, which declares a reference "
            f"policy, but the policy pair carries no references -- the model "
            f"the objective is priced against is absent"
        )
    if not requirements.requires_reference_policy and policy_pair.references:
        keys = tuple(policy_pair.references)
        raise AlgorithmWiringRefusal(
            f"{origin} wired {requirements.name}, which declares no reference "
            f"policy, but the policy pair carries {len(keys)} references "
            f"({', '.join(keys)}): an unconsumed frozen model is a stub that "
            f"reads as a working DPO wiring"
        )
    if requirements.requires_rollout and policy_pair.generate_view is None:
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
    consumed = [role for role, required, _supplied in roles if required]
    if requirements.requires_reference_policy:
        consumed.append("reference_policy")
    return tuple(sorted(consumed))


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
    a CADENCE, so a missing ``sync`` under ``requires_weight_sync`` is
    a step between syncs and passes; that the loss is finite -- a
    diverged step round-trips by design (see ``StepReport``); and
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
    if report.reward_stats is not None and not requirements.requires_advantage:
        raise StepReportRefusal(
            f"{origin}: step {report.step} from {requirements.name} reports "
            f"reward statistics ({report.reward_stats.count} samples) but "
            f"the algorithm declares no advantage function -- an algorithm "
            f"that consumes no advantage function has no reward statistics "
            f"to report, and a reported one came from a path the contract "
            f"does not describe"
        )
    if report.reward_stats is None and requirements.requires_advantage:
        raise StepReportRefusal(
            f"{origin}: step {report.step} from {requirements.name} reports "
            f"no reward statistics though the algorithm declares an "
            f"advantage function -- every step that consumed one summarised "
            f"the samples it used (rows >= 1), so there is no zero-sample "
            f"state in which the summary could be legitimately absent"
        )
    if report.sync is not None and not requirements.requires_weight_sync:
        raise StepReportRefusal(
            f"{origin}: step {report.step} from {requirements.name} reports "
            f"a weight sync (transport {report.sync.transport!r}) but the "
            f"algorithm declares no weight sync -- a reported one came from "
            f"a path the contract does not describe"
        )
    # Deliberate asymmetry with the reward-statistics checks above: a
    # MISSING sync under requires_weight_sync is NOT refused. Reward
    # statistics are produced by every step that consumes an advantage
    # function, whereas a weight sync happens on a CADENCE. A step with
    # no sync is a step BETWEEN syncs, not a step that lost one, and
    # refusing it would make every cadence other than every-step
    # unrepresentable.
    return len(report.loss.components)
