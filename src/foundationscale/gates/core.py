"""The gate contract.

A *gate* is a correctness check that runs at a defined point in a job's lifecycle and
can block. This module defines what a gate is, what it may return, and — the part that
matters — what it is structurally prevented from returning.

Why this exists
---------------
The audit that preceded this framework found the same failure repeatedly, in
checkpointing, export, rewards, the RL trust region and throughput: **the dangerous
failures are the ones that report success.** A checkpoint that was 87.5% wrong passed
`rc=0`, resume, healthy loss, tensor counts and dtypes for two full training runs.

The sharpest instance was not in the training code at all. The tool written to *detect*
silent success silently succeeded: asked whether every expert tensor matched, it
reported ``all_identity: True`` on a corrupt artifact — because the expert tensors were
absent, the comparison set was empty, and ``all([])`` is ``True``.

So this module treats one rule as non-negotiable:

    A gate that inspected nothing did not pass. It returns VACUOUS, and VACUOUS blocks.

The rule is enforced by the framework, not by the gate author. :meth:`Gate.ok` cannot
return ``PASS`` with zero coverage no matter what the author writes; it downgrades. An
author who wants to assert "nothing to check here" must say so explicitly with
:meth:`Gate.skip` and supply a reason, which is recorded and surfaced.

The three properties every gate has
-----------------------------------
1. **A verdict** — and ``PASS`` is only one of two non-blocking outcomes.
2. **Coverage** — how many units it actually examined, and out of how many. An
   unqualified count is not a fact: a gate reporting "3 layers checked" out of 205 is
   :attr:`Verdict.UNDERCOVERED` unless it explicitly declares itself a sample, and a
   gate reporting 500 examined out of 256 expected is :attr:`Verdict.OVERCOVERED` —
   the denominator binds in both directions, because a numerator that outruns it is
   not coverage, it is a contradiction.
3. **Controls** — at least one deliberately broken input the gate *must* flag. A gate
   with no control is not a gate; :func:`verify_controls` fails it in CI. This makes the
   audit's own review rule executable: *every claim that something does not exist must
   name the positive control proving its detector could have fired.*

Gates fail closed. An exception inside a gate is :attr:`Verdict.ERROR`, and ERROR
blocks. This is deliberate and is also drawn from the record: a reward-module import
failure silently disabled a degeneracy veto, and a verifier exception counted as a pass.
"""

from __future__ import annotations

import json
import time
import traceback
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, ClassVar

__all__ = [
    "Lifecycle",
    "Verdict",
    "AbstentionKind",
    "Coverage",
    "GateResult",
    "GateReport",
    "GateBlocked",
    "Control",
    "ControlKind",
    "Gate",
    "GateRegistry",
    "REGISTRY",
    "register",
    "run_event",
    "verify_controls",
    "ControlFailure",
]


class Lifecycle(str, Enum):
    """Points at which gates run.

    These are the moments where a defect either gets caught or gets baked into an
    artifact. Each corresponds to a real incident class in the audited estate.
    """

    LAUNCH = "launch"
    """Before any process starts: topology validity, config resolution, manifest write."""

    BUILD = "build"
    """After the model object is constructed, before weights load or training starts."""

    DATA = "data"
    """After the data pipeline renders a batch: supervision masks, template parity."""

    STEP_ZERO = "step_zero"
    """After the first optimizer step: objective identity, trust region, trainable set."""

    FIRST_SAVE = "first_save"
    """At the first checkpoint of a run — the cheapest place to catch a save defect."""

    SAVE = "save"
    """Every subsequent checkpoint."""

    EXPORT = "export"
    """After a checkpoint is converted to a serving format."""

    PROMOTE = "promote"
    """Before an artifact is declared servable. The last gate before the blast radius."""


class Verdict(str, Enum):
    """The outcome of a gate.

    There are two ways to not-block (:attr:`PASS`, :attr:`SKIP`) and five ways to
    block. The asymmetry is intentional: it is much easier to accidentally produce a
    meaningless success than a meaningless failure. The fifth block,
    :attr:`OVERCOVERED`, joined when the denominator learned to bind in both
    directions: "500 of 256 examined" is as false a claim as "3 of 205 checked".
    """

    PASS = "PASS"
    """Checked a non-vacuous, sufficient set of units and found no defect."""

    FAIL = "FAIL"
    """Found a defect."""

    VACUOUS = "VACUOUS"
    """Reported no defect while inspecting **nothing**. Blocks.

    This is not a pedantic distinction. It is the literal shape of the ``all([]) is
    True`` bug that a verification tool shipped while checking for exactly this class
    of bug in someone else's code.
    """

    UNDERCOVERED = "UNDERCOVERED"
    """Inspected some units but fewer than expected, without declaring itself a sample.

    Blocks. "19 of 23 checked" is a different claim from "checked", and only one of
    them is what a green check mark communicates.
    """

    OVERCOVERED = "OVERCOVERED"
    """Reported MORE units examined than the denominator declares exist. Blocks.

    ``checked > expected`` means at least one of the two numbers is wrong:
    units were double-counted (an aliased name, a re-swept batch), the sweep
    ranged over a superset of the declared population, or ``expected`` came
    from a stale config. Each shape invalidates the result exactly as
    thoroughly as undercoverage does — the claim contradicts its own
    denominator — so this is :attr:`UNDERCOVERED`'s mirror: a blocking
    verdict, not a warning annotation.

    There is deliberately no author pardon (no ``sampled``-equivalent). A
    sample is a deliberate choice to examine *fewer*, declared while both
    numbers remain true; no choice makes 500 a subset of 256, so there is
    nothing to bless. The honest remediations belong to the author: correct
    the count, correct ``expected`` to the true population, or pass
    ``expected=None`` when no denominator is knowable.
    """

    SKIP = "SKIP"
    """Explicitly declined to verify. Requires a reason. Does not block, but is reported.

    Two different abstentions wear this verdict — "the property does not exist
    here" and "the property exists but my evidence cannot settle it". The split
    is machine-readable in :attr:`GateResult.abstention` (:class:`AbstentionKind`);
    aggregators pricing a verified/applicable denominator must read THAT FIELD,
    never the prose reason. An undeclared kind (``None``) prices exactly like
    :attr:`AbstentionKind.NOT_ESTABLISHED`: it never leaves a denominator.
    """

    ERROR = "ERROR"
    """The gate itself raised. Blocks — gates fail closed."""

    @property
    def blocking(self) -> bool:
        """Whether this verdict should stop the job or prevent promotion."""
        return self in (
            Verdict.FAIL,
            Verdict.VACUOUS,
            Verdict.UNDERCOVERED,
            Verdict.OVERCOVERED,
            Verdict.ERROR,
        )

    @property
    def symbol(self) -> str:
        return {
            Verdict.PASS: "ok",
            Verdict.FAIL: "FAIL",
            Verdict.VACUOUS: "VACUOUS",
            Verdict.UNDERCOVERED: "UNDER",
            Verdict.OVERCOVERED: "OVER",
            Verdict.SKIP: "skip",
            Verdict.ERROR: "ERROR",
        }[self]


class AbstentionKind(str, Enum):
    """Why a gate abstained — machine-readable, because composites must price it.

    :attr:`Verdict.SKIP` is two different statements wearing one verdict, and an
    aggregator that prices them identically is wrong in both directions:

    * Charging "not established" as applicable-and-verified fabricates coverage.
    * Charging "not applicable" as applicable-and-missing blocks every run whose
      declared scope legitimately lacks the property — a composite that blocks
      every dense model at its first save teaches operators to route around the
      gate, which is how verification tools die in the audited estate.

    The split therefore lives HERE, as data — never in the skip reason string,
    which prose-sniffing aggregators would paraphrase (the defect this codebase
    already removed once). ``None`` on :attr:`GateResult.abstention` is NOT a
    synonym for either member: it means "this call site was never audited for
    the distinction", and consumers must price it exactly like
    :attr:`NOT_ESTABLISHED` — it stays inside every denominator.
    """

    NOT_APPLICABLE = "not_applicable"
    """The property does not exist in this run's DECLARED scope.

    Legitimate only when grounded in a POSITIVE declaration — an explicit
    ``num_experts == 0``, never an absence of evidence. "I found no experts" is
    not "this model has no experts"; that inference is the founding incident.
    A composite may remove a NOT_APPLICABLE verdict from its applicable
    denominator, must NAME it, and must never count it as verified.
    """

    NOT_ESTABLISHED = "not_established"
    """The property may well exist; the available evidence could not settle it.

    The canonical instance is per-expert identity inside a stacked MoE tensor:
    the experts exist, and metadata cannot see whether they are distinct. This
    kind MUST remain in every denominator — "could not check" is charged
    against the sweep, never excused from it.
    """


@dataclass(frozen=True)
class Coverage:
    """How much a gate actually looked at.

    Every gate result carries one. It is the difference between "the experts match" and
    "the experts I compared match, and I compared 3,840 of 3,840".

    Args:
        checked: Number of units actually examined. Zero means vacuous, always.
        unit: What is being counted, plural, lowercase — ``"experts"``, ``"tensors"``,
            ``"export dirs"``. Appears in rendered output, so make it read naturally.
        expected: Total units that *should* have been examined, when knowable. Leave
            ``None`` when the denominator genuinely is not known in advance; do not
            fabricate one to make a ratio look complete. The denominator binds in
            BOTH directions when it is given: ``checked < expected`` blocks as
            :attr:`Verdict.UNDERCOVERED` unless declared a sample, and ``checked >
            expected`` blocks as :attr:`Verdict.OVERCOVERED` with no declaration
            available. If the true population moved since ``expected`` was computed,
            update ``expected`` — do not ship a contradicted ratio.
        sampled: Set ``True`` to declare deliberately partial coverage. This converts
            what would be :attr:`Verdict.UNDERCOVERED` into a pass — so it requires
            ``sample_reason`` and is surfaced in every rendering. It pardons
            undercoverage only: a count above the denominator is not a partial
            anything, and :attr:`is_over` coverage blocks regardless.
        sample_reason: Why partial coverage is acceptable here. Required if ``sampled``.
    """

    checked: int
    unit: str
    expected: int | None = None
    sampled: bool = False
    sample_reason: str = ""

    def __post_init__(self) -> None:
        if self.checked < 0:
            raise ValueError(f"coverage cannot be negative: {self.checked}")
        if self.expected is not None and self.expected < 0:
            raise ValueError(f"expected cannot be negative: {self.expected}")
        if self.sampled and not self.sample_reason.strip():
            raise ValueError(
                "Coverage(sampled=True) requires sample_reason. Declaring a sample is "
                "how a gate is allowed to check less than everything; it is not a way "
                "to avoid saying so."
            )

    @property
    def is_vacuous(self) -> bool:
        """True if nothing at all was examined."""
        return self.checked == 0

    @property
    def is_short(self) -> bool:
        """True if fewer units were examined than expected, sampling aside."""
        return self.expected is not None and self.checked < self.expected

    @property
    def is_over(self) -> bool:
        """True if MORE units were examined than the denominator declares.

        The mirror of :attr:`is_short`, shipped as a sibling rather than a
        widening of it: "short" means ``checked < expected`` — that reading is
        pinned by the contract tests, 300-of-205 explicitly NOT short — and
        folding overage into the same word would silently retarget every
        existing consumer of it. An over-covered claim contradicts itself (the
        units examined cannot outnumber the units that exist), so
        :meth:`Gate.ok` blocks it outright; no ``sampled`` pardon applies,
        because sampling declares a partial sweep and a count above the
        denominator is not a partial anything.
        """
        return self.expected is not None and self.checked > self.expected

    @property
    def fraction(self) -> float | None:
        # May exceed 1.0: that IS the overage signal, and capping it here would
        # file the evidence down to fit the chart. Consumers that need a
        # bounded ratio must clamp on their own side, knowingly.
        if self.expected in (None, 0):
            return None
        return self.checked / float(self.expected)

    def __str__(self) -> str:
        if self.expected is None:
            base = f"{self.checked} {self.unit}"
        else:
            base = f"{self.checked}/{self.expected} {self.unit}"
        if self.sampled:
            base += f" (sample: {self.sample_reason})"
        if self.is_over:
            # "500/256 reward samples" printed bare reads as a typo, not a
            # defect — and this string is rendered where no verdict sits
            # beside it (the coverage column is the whole result line until
            # the detail follows). "3/205" stays bare because it is a
            # believable statement that needs a verdict to judge it; "500/256"
            # refutes itself on its face, so the string must not present it
            # neutrally.
            base += " (over: checked exceeds expected — one of them is wrong)"
        return base

    @classmethod
    def none(cls, unit: str) -> Coverage:
        """Zero coverage. Any result built on this downgrades to VACUOUS."""
        return cls(checked=0, unit=unit)


@dataclass(frozen=True)
class GateResult:
    """What a gate returns.

    Construct these via :meth:`Gate.ok`, :meth:`Gate.fail` and :meth:`Gate.skip` rather
    than directly — those apply the coverage rule that this class exists to serve.
    """

    gate_id: str
    verdict: Verdict
    coverage: Coverage
    detail: str = ""
    evidence: Mapping[str, Any] = field(default_factory=dict)
    duration_s: float = 0.0
    abstention: AbstentionKind | None = None
    """Machine-readable kind of a :attr:`Verdict.SKIP` verdict.

    ``None`` for every other verdict, and for a SKIP whose kind the gate never
    declared (a legacy or unaudited call site — priced like NOT_ESTABLISHED by
    consumers, never like NOT_APPLICABLE). Carried on the result rather than
    encoded in ``detail`` so that composites never parse prose to price an
    abstention.
    """

    def __post_init__(self) -> None:
        # An abstention kind on a non-SKIP verdict would let a PASSED result
        # smuggle an inapplicability claim past every aggregator that trusts the
        # verdict alone. Refuse at construction: the type itself keeps the two
        # channels consistent, the same doctrine as Gate.ok()'s downgrades.
        if self.abstention is not None and self.verdict is not Verdict.SKIP:
            raise ValueError(
                f"abstention kind {self.abstention!r} is only meaningful on "
                f"Verdict.SKIP, got {self.verdict!r} — a pass cannot carry an "
                f"inapplicability"
            )

    @property
    def blocking(self) -> bool:
        return self.verdict.blocking

    def render(self) -> str:
        line = f"[{self.verdict.symbol:>7}] {self.gate_id}: {self.coverage}"
        if self.detail:
            line += f" — {self.detail}"
        return line

    def to_dict(self) -> dict[str, Any]:
        return {
            "gate": self.gate_id,
            "verdict": self.verdict.value,
            # Serialized for every result (None when not a declared-kind SKIP):
            # a downstream aggregator reading JSON must be able to price an
            # abstention without re-parsing the detail prose.
            "abstention": self.abstention.value if self.abstention is not None else None,
            "checked": self.coverage.checked,
            "expected": self.coverage.expected,
            "unit": self.coverage.unit,
            "sampled": self.coverage.sampled,
            "sample_reason": self.coverage.sample_reason or None,
            "detail": self.detail,
            "evidence": dict(self.evidence),
            "duration_s": round(self.duration_s, 4),
        }


_EMPTY_SWEEP_GATE_PREFIX = "registry.empty_sweep."
"""Gate-id prefix for the framework-synthesized result that blocks an empty sweep.

A registry run that executes zero gates appends exactly one result whose id is
this prefix plus the event value, so a report's vacuity is identifiable from its
own contents — :attr:`GateReport.is_vacuous` does not have to trust a bare count.
The prefix is reserved: :meth:`GateRegistry.register` refuses author gates under
it, so the marker cannot be spoofed.
"""


@dataclass(frozen=True)
class GateReport:
    """The result of running every gate registered for one lifecycle event."""

    event: Lifecycle
    results: tuple[GateResult, ...]
    missing: tuple[str, ...] = ()
    """Gates the caller declared required that did not run. Blocks the whole report.

    A registry that silently ran zero gates is the same failure as a gate that
    silently checked zero units, one level up — so an empty sweep blocks on its
    own now (see :meth:`GateRegistry.run`); ``required`` adds the named-gate leg
    on top of that floor.
    """

    registered: int | None = None
    """How many gates were registered for this event when the sweep ran.

    The sweep-level denominator: a verdict over "3 gates" is as unqualified as a
    gate over "3 tensors" if nobody says out of how many. The runner that produced
    the report sets it; a hand-built report leaves it ``None``, and :meth:`render`
    then claims no denominator rather than inventing one.
    """

    allow_empty: bool = False
    """Whether this report's event was *declared* legitimately gateless.

    The runner sets this from the registry's ``event_allow_empty`` opt-out; a
    hand-built report gets ``False`` and so fails closed. This field is what
    lets :attr:`ok` enforce :attr:`is_vacuous` on the type itself: a report
    emptied by filtering, merging, or plain non-invocation blocks, and only a
    sweep the integrator explicitly declared gateless does not. Anything
    looser reopens the report-layer ``all([])``; anything stricter breaks the
    one declared extension point.
    """

    @property
    def blocking(self) -> tuple[GateResult, ...]:
        return tuple(r for r in self.results if r.blocking)

    @property
    def is_vacuous(self) -> bool:
        """True when this report records no executed gate.

        Either there are no results at all — ``all([])`` is ``True``, the
        framework's namesake bug, and here it is the right answer: a report over
        nothing is a report over nothing — or every result is the framework's
        synthesized empty-sweep marker from :meth:`GateRegistry.run`. A vacuous
        report is proof of nothing whether or not the caller opted out of
        blocking via ``GateRegistry(event_allow_empty=...)``.
        """
        return all(r.gate_id.startswith(_EMPTY_SWEEP_GATE_PREFIX) for r in self.results)

    @property
    def is_unverified(self) -> bool:
        """True when gates executed and not one of them verified anything.

        Distinct from :attr:`is_vacuous`, and deliberately NOT folded into it.
        Vacuity names "no gate ran"; this names "gates ran and every single one
        declined". The only non-blocking verdicts are :attr:`Verdict.PASS` and
        :attr:`Verdict.SKIP`, so a report that reaches :attr:`ok`'s final clause
        with no blocking results and real gate ids is a sweep whose every unit
        is a declared, reasoned, per-gate SKIP — the audit's ``all([]) is True``
        wearing per-gate costumes. Each abstention is a first-class outcome; an
        aggregate of nothing but abstentions still examined zero units, and the
        verdict-bearing position must say so (the SKIP lines below the headline
        are detail, not verdict).

        Only PASS and FAIL are post-examination verdicts: both assert the gate
        actually looked at its units. VACUOUS examined nothing by definition,
        ERROR never rendered on the artifact, UNDERCOVERED and OVERCOVERED block
        on their own — those four never reach this property's deciding case
        anyway, but the set is named over full verdicts so any future verdict
        defaults to "not verification" unless explicitly added. OVERCOVERED
        inherits that default exactly as the design intends: a gate that
        examined plenty but contradicts its own denominator has still verified
        nothing.

        Empty results are excluded by design: a no-results report is VACUOUS,
        :attr:`is_vacuous` already owns that shape along with its ``allow_empty``
        pardon, and the two properties must not both claim it — otherwise a
        declared-gateless sweep would block on the abstention rule, which exists
        for *executed* gates. That polarity split is load-bearing; the report-
        vacuity test suite pins both sides of it.
        """
        return bool(self.results) and not any(
            r.verdict in (Verdict.PASS, Verdict.FAIL) for r in self.results
        )

    @property
    def ok(self) -> bool:
        """True only when nothing blocks and the sweep verified something.

        The first two clauses are the old definition. The third closes the
        report-layer ``all([])``: over an empty report, "no blocking results"
        and "no missing gates" are both trivially true, so a hand-built or
        filtered-down report over zero executed gates used to read as success
        here — the empty-comparison verdict restated one level up.
        :attr:`is_vacuous` already named the condition; now it gates on it. The
        sole pardon is a runner-declared :attr:`allow_empty`, where gateless
        was a decision, not an accident.

        The fourth clause closes the per-gate-costume restatement: a report
        whose every result is SKIP has no blocking results, no missing gates,
        and real (non-marker) gate ids — and examined zero units.
        :attr:`allow_empty` does NOT pardon this clause. Its declaration is
        "this event is legitimately gateless", and a report carrying SKIP
        results is not gateless: gates were registered, wired and run, and each
        individually declined — precisely the pattern (a renamed ctx field, a
        wholesale ``missing_ctx="report-skip"``) that must surface instead of
        passing. An environment where no gate applies is expressed by not
        selecting the gates, making the event genuinely gateless; then the
        declaration matches the fact and the third clause pardons it.
        """
        return (
            not self.blocking
            and not self.missing
            and (not self.is_vacuous or self.allow_empty)
            and not self.is_unverified
        )

    def render(self) -> str:
        # The empty-sweep marker is a result, not a gate that ran. The footer has
        # always excluded it; the head once did not, rendering "1 run" directly
        # above "0 gates ran of 0 registered". One report emitting two run counts
        # that disagree is the audit's whole complaint in miniature, so the count
        # is computed once, here, and shared by both lines.
        ran = sum(1 for r in self.results if not r.gate_id.startswith(_EMPTY_SWEEP_GATE_PREFIX))
        head = f"gates @ {self.event.value}: {ran} run"
        if self.ok:
            head += " — all clear"
            if self.is_vacuous:
                # Reachable only via allow_empty: "all clear" over zero gates is
                # a claim broader than its evidence unless the declaration that
                # permitted it is shown.
                head += " (gateless by declaration: event_allow_empty)"
            else:
                skipped = sum(1 for r in self.results if r.verdict is Verdict.SKIP)
                if skipped:
                    # An ok report contains only PASS and SKIP results (every
                    # other verdict blocks, and the marker never coexists with a
                    # verdict of ok), so ran - skipped IS the verified count.
                    # "all clear" alone would claim all N gates verified; the
                    # parenthetical keeps the headline inside its evidence while
                    # the per-gate skip lines carry the reasons.
                    head += f" ({ran - skipped} of {ran} verified; {skipped} declared SKIP)"
        else:
            bits = []
            if self.blocking:
                bits.append(f"{len(self.blocking)} blocking")
            elif self.is_vacuous:
                # Hand-built report over zero results: nothing executed, so the
                # blocking tuple is empty and it is the vacuity itself that must
                # be named here — "0 blocking" would read as a near-miss.
                bits.append("VACUOUS — no gates ran")
            elif self.is_unverified:
                # Nothing blocked, nothing is missing — and nothing verified.
                # The head must name the abstention sweep WITH its denominator,
                # or "2 run" reads as two verifications above a wall of skips.
                bits.append(f"0 of {ran} verified — every gate abstained (SKIP)")
            if self.missing:
                bits.append(f"{len(self.missing)} MISSING")
            head += " — " + ", ".join(bits)
        lines = [head]
        lines += ["  " + r.render() for r in self.results]
        lines += [f"  [MISSING] {g}: required but never ran" for g in self.missing]
        if self.registered is not None:
            lines.append(
                f"  — {ran} gates ran of {self.registered} registered for {self.event.value}"
            )
        return "\n".join(lines)

    def to_json(self, **kwargs: Any) -> str:
        return json.dumps(
            {
                "event": self.event.value,
                "ok": self.ok,
                "missing": list(self.missing),
                "registered": self.registered,
                "allow_empty": self.allow_empty,
                # The abstention-sweep state travels on the wire: otherwise a
                # downstream consumer sees a bare "ok": false and must re-derive
                # WHY from per-gate verdicts — or worse, sees "ok": true over a
                # healthy-but-partial sweep with the skips invisible at the top.
                "unverified": self.is_unverified,
                "results": [r.to_dict() for r in self.results],
            },
            **kwargs,
        )

    def raise_if_blocking(self) -> None:
        """Fail closed. Call this at every gate site that is allowed to stop the job."""
        if self.ok:
            return
        raise GateBlocked(self)


class GateBlocked(RuntimeError):
    """Raised by :meth:`GateReport.raise_if_blocking`."""

    def __init__(self, report: GateReport) -> None:
        self.report = report
        super().__init__(report.render())


class ControlKind(str, Enum):
    """What a control asserts about the gate."""

    MUST_FIRE = "must_fire"
    """A deliberately defective input. If the gate does not block on it, the gate is
    broken and must not be trusted on real inputs.

    This is what the audit means by "positive control": proof the detector can fire.
    """

    MUST_PASS = "must_pass"
    """A known-good input. Guards against a gate that blocks on everything, which is
    just as useless and tends to get disabled.

    "Does not block" is not the assertion: the control must produce its DECLARED
    outcome — :attr:`Verdict.PASS` by default, or an explicit
    :attr:`Control.expect_skip` reason when the healthy input is one the gate
    genuinely cannot adjudicate. An abstention discovered at run time rather
    than declared on the fixture certified nothing about healthy-input
    behaviour, and :func:`verify_controls` now fails it.
    """


@dataclass(frozen=True)
class Control:
    """A fixture that proves a gate works.

    Controls are executable, not documentation. :func:`verify_controls` runs them and is
    intended to be wired into CI, so a gate cannot rot into a no-op unnoticed.
    """

    name: str
    kind: ControlKind
    make_ctx: Callable[[], Any]
    """Builds the context to hand the gate. Called fresh per run; may create tmp files."""

    note: str = ""
    """What defect this fixture injects, in one line."""

    expect_skip: str = ""
    """Declares that a :attr:`ControlKind.MUST_PASS` control is EXPECTED to abstain.

    Empty — the default — means the control must reach :attr:`Verdict.PASS`.
    The strict reading is the default deliberately, so the stricter rule cannot
    be satisfied by omission: doctrine (1) applied to the declaration itself.
    Set a non-empty reason only when the fixture is known-healthy AND genuinely
    unadjudicable by this gate (the live case: per-expert identity inside a
    stacked MoE tensor is metadata-invisible, so a PASS there would be a claim
    broader than the gate's evidence). :func:`verify_controls` checks the
    declaration against reality in BOTH directions: abstain without declaring
    and the run fails; declare and then reach PASS anyway and the run also
    fails, because the gate demonstrably CAN adjudicate the fixture and the
    declaration has become a stale claim narrower than its evidence. Illegal
    on :attr:`ControlKind.MUST_FIRE`: a deliberately defective input the gate
    is expected to abstain on has proven nothing about the defect — that
    fixture is a MUST_PASS control or it is nothing — so the combination is
    refused here, at construction, rather than priced at run time.
    """

    def __post_init__(self) -> None:
        if self.expect_skip:
            if not self.expect_skip.strip():
                raise ValueError(
                    f"Control {self.name!r}: expect_skip must carry the reason the "
                    f"gate cannot adjudicate this fixture — a blank declaration is "
                    f"an exemption, and an unreasoned exemption is the exact defect "
                    f"this field exists to remove"
                )
            if self.kind is ControlKind.MUST_FIRE:
                raise ValueError(
                    f"Control {self.name!r}: expect_skip is illegal on MUST_FIRE — "
                    f"a positive control the detector is expected to abstain on "
                    f"never fired, so nothing is proven; the fixture belongs under "
                    f"MUST_PASS or not at all"
                )


class ControlFailure(AssertionError):
    """A gate failed its own control."""


class Gate(ABC):
    """Base class for all gates.

    Subclasses set the three class attributes, implement :meth:`check`, and declare
    controls of BOTH kinds — at least one :attr:`ControlKind.MUST_FIRE` proving the
    detector can block, and at least one :attr:`ControlKind.MUST_PASS` proving it
    does not block on a healthy input. :func:`verify_controls` enforces both. One
    kind without the other is half a proof, and was accepted as a whole one until
    an adversarial sweep traced the zero-trip loop in verify_controls: a gate with
    only MUST_FIRE controls had its healthy-input behaviour verified zero times.

    Return results through :meth:`ok`, :meth:`fail` and :meth:`skip`. Those helpers
    apply the coverage rule; constructing :class:`GateResult` by hand bypasses it, which
    is exactly the hole this class exists to close.

    Example::

        class ExpertBytesGate(Gate):
            id = "checkpoint.expert_bytes"
            description = "Expert parameter bytes match the model's declared shape"
            events = (Lifecycle.FIRST_SAVE, Lifecycle.SAVE)

            def check(self, ctx):
                experts = ctx.expert_tensors()          # may be empty!
                bad = [e for e in experts if e.implied_nbytes != ctx.expected_nbytes(e)]
                cov = Coverage(len(experts), "expert tensors",
                               expected=ctx.declared_expert_count)
                if bad:
                    return self.fail(f"{len(bad)} experts with wrong byte count", cov,
                                     evidence={"offenders": [e.name for e in bad[:8]]})
                return self.ok("all expert byte counts match", cov)

            def controls(self):
                return [
                    Control("aliased-16-of-128", ControlKind.MUST_FIRE,
                            make_aliased_ckpt,
                            note="128 experts collapsed to 16 by local-name save"),
                    Control("intact-128-of-128", ControlKind.MUST_PASS,
                            make_intact_ckpt,
                            note="declared shapes and byte counts all correct — "
                                 "without a healthy fixture, a gate that blocked "
                                 "on every input would verify green"),
                ]

    If ``experts`` comes back empty — the corrupt-artifact case — ``self.ok`` returns
    ``VACUOUS`` rather than ``PASS``, and the author did not have to remember to handle
    it.
    """

    id: ClassVar[str]
    description: ClassVar[str]
    events: ClassVar[tuple[Lifecycle, ...]]
    context_type: ClassVar[type | None] = None
    """The concrete context this gate consumes, for typed sweeps.

    ``None`` (the default) keeps the legacy single-context broadcast of
    :meth:`GateRegistry.run`. Declaring a real type lets :func:`run_event` mix this
    gate into one sweep keyed by context type, and turns "the integrator never
    wired my context" from a raw TypeError inside :meth:`check` into a named,
    blocking ERROR that identifies the missing type. Gates that accept more than
    their declared type (a path, an adapter object) say so by overriding
    :meth:`coerce_context`.
    """

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if ABC in cls.__bases__:
            return
        for attr in ("id", "description", "events"):
            if not getattr(cls, attr, None):
                raise TypeError(f"{cls.__name__} must define a non-empty '{attr}'")
        if not isinstance(cls.events, tuple) or not all(
            isinstance(e, Lifecycle) for e in cls.events
        ):
            raise TypeError(f"{cls.__name__}.events must be a tuple of Lifecycle members")
        context_type = getattr(cls, "context_type", None)
        if context_type is not None and not isinstance(context_type, type):
            raise TypeError(
                f"{cls.__name__}.context_type must be a type or None (legacy "
                f"broadcast), got {context_type!r}"
            )

    # -- result constructors ------------------------------------------------------

    def ok(
        self,
        detail: str,
        coverage: Coverage,
        *,
        evidence: Mapping[str, Any] | None = None,
    ) -> GateResult:
        """Report no defect found — subject to the coverage rule.

        This does **not** always produce :attr:`Verdict.PASS`. If ``coverage`` is
        vacuous the verdict is :attr:`Verdict.VACUOUS`; if it is short of ``expected``
        and not declared a sample, the verdict is :attr:`Verdict.UNDERCOVERED`; if it
        EXCEEDS ``expected`` the verdict is :attr:`Verdict.OVERCOVERED` — a claim
        whose numerator outruns its own denominator is not a pass, and no sample
        declaration pardons it. The gate author cannot override this, which is the
        point.
        """
        if coverage.is_vacuous:
            return self._result(
                Verdict.VACUOUS,
                coverage,
                detail=(
                    f"gate examined 0 {coverage.unit} and therefore proves nothing"
                    + (f" (claimed: {detail})" if detail else "")
                ),
                evidence=evidence,
            )
        if coverage.is_short and not coverage.sampled:
            return self._result(
                Verdict.UNDERCOVERED,
                coverage,
                detail=(
                    f"examined {coverage.checked} of {coverage.expected} "
                    f"{coverage.unit}; declare Coverage(sampled=True, sample_reason=...) "
                    f"if partial coverage is intended" + (f" (claimed: {detail})" if detail else "")
                ),
                evidence=evidence,
            )
        if coverage.is_over:
            # Placed AFTER the sampled pardon, and deliberately NOT conditioned on
            # it: sampled= declares a partial sweep and can only pardon shortage.
            # A numerator above the denominator is a contradiction, not a partial
            # result, so nothing an author can attach to a Coverage lifts this
            # block. Before this branch existed the chain's final else was
            # default-success, and overage was the last coverage classification
            # that reached PASS unopposed.
            return self._result(
                Verdict.OVERCOVERED,
                coverage,
                detail=(
                    f"examined {coverage.checked} of {coverage.expected} "
                    f"{coverage.unit} — the numerator exceeds the denominator, so at "
                    f"least one of them is wrong (double-counted units, a superset "
                    f"sweep, or a stale expected); fix the count, correct expected, "
                    f"or pass expected=None" + (f" (claimed: {detail})" if detail else "")
                ),
                evidence=evidence,
            )
        return self._result(Verdict.PASS, coverage, detail=detail, evidence=evidence)

    def fail(
        self,
        detail: str,
        coverage: Coverage,
        *,
        evidence: Mapping[str, Any] | None = None,
    ) -> GateResult:
        """Report a defect. Always blocks; coverage is recorded but never softens it."""
        return self._result(Verdict.FAIL, coverage, detail=detail, evidence=evidence)

    def skip(self, reason: str, *, kind: AbstentionKind | None = None) -> GateResult:
        """Declare the gate non-verifying on this input. Requires a reason; does not block.

        ``kind`` makes the abstention machine-readable so composites can price
        it. :attr:`AbstentionKind.NOT_APPLICABLE` is only legitimate behind a
        POSITIVE declaration that the property does not exist in this run's
        scope (an explicit ``num_experts == 0``, never an absence of evidence);
        a composite may then remove the gate from its applicable denominator.
        Everywhere else pass :attr:`AbstentionKind.NOT_ESTABLISHED` or leave
        the default ``None``: both stay in every denominator, and ``None``
        deliberately records that the call site was never audited for the
        distinction rather than laundering legacy code into an audited kind.
        """
        if not reason.strip():
            raise ValueError("skip() requires a reason — an unexplained skip is a hole")
        return self._result(
            Verdict.SKIP, Coverage.none("units"), detail=reason, abstention=kind
        )

    def _result(
        self,
        verdict: Verdict,
        coverage: Coverage,
        *,
        detail: str = "",
        evidence: Mapping[str, Any] | None = None,
        abstention: AbstentionKind | None = None,
    ) -> GateResult:
        return GateResult(
            gate_id=self.id,
            verdict=verdict,
            coverage=coverage,
            detail=detail,
            evidence=dict(evidence or {}),
            abstention=abstention,
        )

    # -- the gate itself ----------------------------------------------------------

    @abstractmethod
    def check(self, ctx: Any) -> GateResult:
        """Run the check. Must return via :meth:`ok`/:meth:`fail`/:meth:`skip`."""

    @abstractmethod
    def controls(self) -> Sequence[Control]:
        """Fixtures proving this gate works.

        At least one :attr:`ControlKind.MUST_FIRE` AND one :attr:`ControlKind.MUST_PASS`,
        both enforced in CI by :func:`verify_controls`: the MUST_FIRE proves the
        detector can fire; the MUST_PASS proves it is not firing on everything. A
        gate that blocks unconditionally is useless and gets disabled — and without
        a declared healthy fixture, no run would ever notice it had become one.
        """

    def run(self, ctx: Any) -> GateResult:
        """Invoke :meth:`check` with timing, and convert any exception to ERROR."""
        t0 = time.perf_counter()
        try:
            result = self.check(ctx)
        except Exception as exc:  # noqa: BLE001 — fail closed, deliberately broad
            return GateResult(
                gate_id=self.id,
                verdict=Verdict.ERROR,
                coverage=Coverage.none("units"),
                detail=f"{type(exc).__name__}: {exc}",
                evidence={"traceback": traceback.format_exc(limit=12)},
                duration_s=time.perf_counter() - t0,
            )
        # mypy calls this branch unreachable, because `check()` is annotated
        # `-> GateResult`. That annotation is a promise from the gate author, and this
        # framework's entire premise is that promises about verification are the thing
        # most worth checking. A gate that returns `True` type-checks clean and, without
        # this branch, would sail through `report.ok` as a truthy non-result. Keep the
        # branch, keep the test that covers it, and silence the static claim instead.
        if not isinstance(result, GateResult):
            return GateResult(  # type: ignore[unreachable]
                gate_id=self.id,
                verdict=Verdict.ERROR,
                coverage=Coverage.none("units"),
                detail=f"check() returned {type(result).__name__}, expected GateResult",
                duration_s=time.perf_counter() - t0,
            )
        object.__setattr__(result, "duration_s", time.perf_counter() - t0)
        return result

    def coerce_context(self, ctx: Any) -> Any | None:  # noqa: ARG002 — refusing hook
        """Adapt a foreign context to :attr:`context_type`, or refuse by returning ``None``.

        :func:`run_event` calls this when a bare (non-mapping) context is not
        already an instance of the declared type. Return ``None`` for inputs this
        gate does not recognise — the dispatcher turns that into a named, blocking
        "unwired, not healthy" ERROR, so never raise ``TypeError`` here: that would
        reproduce the opaque failure this hook exists to replace. Other exceptions
        (I/O while adapting, a lazy import) may propagate; adaptation is gate-author
        code and lands in the sweep's ERROR conversion exactly like a failure inside
        :meth:`check`.
        """
        return None


class GateRegistry:
    """Holds gates and runs them by lifecycle event.

    The registry exists so that gate invocation is a property of the *event*, not of
    whichever launcher script happens to be running. In the audited estate the export
    byte check lived as a copy-pasted heredoc in one script and was simply absent from
    the other, which is how a truncated export reached ``rc=0``.
    """

    def __init__(self, *, event_allow_empty: Iterable[Lifecycle] = ()) -> None:
        """The single opt-out from the empty-sweep rule lives here.

        Args:
            event_allow_empty: Lifecycle events that may legitimately run with
                zero registered gates — a deliberate extension point, e.g. a
                harness that populates the registry later. Every other event
                that matches no gate produces a blocking VACUOUS report from
                :meth:`run`. The allowed-empty report is still
                :attr:`GateReport.is_vacuous`; only the block is lifted.
        """
        self._gates: dict[str, Gate] = {}
        self._event_allow_empty = frozenset(Lifecycle(e) for e in event_allow_empty)

    def register(self, gate: Gate) -> Gate:
        if gate.id.startswith(_EMPTY_SWEEP_GATE_PREFIX):
            raise ValueError(
                f"gate id {gate.id!r} starts with {_EMPTY_SWEEP_GATE_PREFIX!r}, "
                f"which is reserved for the framework's empty-sweep markers — "
                f"an author-named marker could spoof GateReport.is_vacuous"
            )
        if gate.id in self._gates:
            raise ValueError(f"duplicate gate id: {gate.id!r}")
        self._gates[gate.id] = gate
        return gate

    def get(self, gate_id: str) -> Gate:
        return self._gates[gate_id]

    def __contains__(self, gate_id: object) -> bool:
        return gate_id in self._gates

    def __len__(self) -> int:
        return len(self._gates)

    def __iter__(self) -> Iterator[Gate]:
        return iter(self._gates.values())

    def for_event(self, event: Lifecycle) -> list[Gate]:
        return [g for g in self._gates.values() if event in g.events]

    def run(
        self,
        event: Lifecycle,
        ctx: Any,
        *,
        required: Iterable[str] | None = None,
    ) -> GateReport:
        """Run every gate registered for ``event``.

        Args:
            event: The lifecycle point being gated.
            ctx: Passed unchanged to each gate's ``check``.
            required: Gate ids the caller asserts must run here. Any that are not
                registered for this event are reported in :attr:`GateReport.missing`
                and block. Use it at sites where a missing gate is itself the bug —
                which, on the evidence, is most of them.

        If no gate is registered for ``event``, the report is blocking VACUOUS
        by construction: a sweep that ran nothing is not an all-clear, with or
        without ``required``. The only opt-out is the registry's
        ``event_allow_empty`` declaration.
        """
        gates = self.for_event(event)
        results = [g.run(ctx) for g in gates]
        missing: tuple[str, ...] = ()
        if required is not None:
            ran = {g.id for g in gates}
            missing = tuple(sorted(set(required) - ran))
        if not gates and event not in self._event_allow_empty:
            # The sweep-level all([]): zero gates ran, so this report must not
            # be able to read as success. The synthesized result is blocking
            # VACUOUS and records the population it drew from, so "registered
            # 10, matched 0 at this event" is a fact in the evidence rather
            # than an inference.
            results.append(
                GateResult(
                    gate_id=f"{_EMPTY_SWEEP_GATE_PREFIX}{event.value}",
                    verdict=Verdict.VACUOUS,
                    coverage=Coverage.none("gates"),
                    detail=(
                        f"no gates ran for {event.value}; a sweep over nothing "
                        f"proves nothing (construct GateRegistry("
                        f"event_allow_empty=...) if gateless is deliberate)"
                    ),
                    evidence={
                        "event": event.value,
                        "registered_gates": len(self._gates),
                    },
                )
            )
        return GateReport(
            event=event,
            results=tuple(results),
            missing=missing,
            registered=len(gates),
            # ok now enforces vacuity on the type; the declared opt-out must
            # travel with the report or this runner's own allowance would block.
            allow_empty=event in self._event_allow_empty,
        )


REGISTRY = GateRegistry()
"""Process-wide default registry."""


def register(cls: type[Gate]) -> type[Gate]:
    """Class decorator: instantiate a gate and add it to :data:`REGISTRY`."""
    REGISTRY.register(cls())
    return cls


_MISSING = object()
"""Sentinel for "no supplied context matched the gate's declared type".

Distinct from ``None`` deliberately: ``None`` is a legitimate context value and
must never be able to impersonate *absent*.
"""


class _AmbiguousMatch(Exception):
    """Several supplied contexts satisfy the gate's declared type; picking is a guess."""

    def __init__(self, names: list[str]) -> None:
        self.names = names
        super().__init__(", ".join(names))


def _resolve_context(gate: Gate, contexts: Mapping[type, Any]) -> Any:
    """Pick the one context ``gate.context_type`` names, or return ``_MISSING``.

    Order is exact-then-subclass: an exact type-key hit wins; a single subclass hit
    is honoured; two or more raise :class:`_AmbiguousMatch`, because dispatch must
    never arbitrate silently. Callers convert both outcomes into results — this
    function must not itself decide a verdict.
    """
    ct = gate.context_type
    assert ct is not None  # legacy-broadcast gates never reach the resolver
    if ct in contexts:
        return contexts[ct]
    subtype_hits = [
        (key, value)
        for key, value in contexts.items()
        if isinstance(key, type) and issubclass(key, ct)
    ]
    if len(subtype_hits) == 1:
        return subtype_hits[0][1]
    if subtype_hits:
        raise _AmbiguousMatch([key.__name__ for key, _ in subtype_hits])
    return _MISSING


def _dispatch_error(gate: Gate, exc: BaseException, started: float) -> GateResult:
    """An ERROR for failures *around* ``check()`` — the same conversion, one frame up.

    Context adaptation and lazy context factories are gate-author code just like
    ``check``; letting their exceptions escape the sweep would be the
    verifier-exception-counted-as-pass failure with a new callstack.
    """
    return GateResult(
        gate_id=gate.id,
        verdict=Verdict.ERROR,
        coverage=Coverage.none("units"),
        detail=f"{type(exc).__name__}: {exc}",
        evidence={"traceback": traceback.format_exc(limit=12)},
        duration_s=time.perf_counter() - started,
    )


def _run_dispatched_gate(gate: Gate, contexts: Any, missing_ctx: str) -> GateResult:
    """Resolve this gate's context and run it, converting wiring defects into results.

    Nothing here can return PASS without the gate's ``check`` having run: every
    outcome that is not "the gate received its declared context" is an ERROR (or an
    explicitly declared SKIP), so an unwired sweep cannot read as green.
    """

    def unwired(ct_name: str) -> GateResult:
        # Built through gate._result, not ok()/fail(): this verdict is about the
        # sweep's wiring, and the artifact was never examined at all.
        if missing_ctx == "report-skip":
            return gate._result(
                Verdict.SKIP,
                Coverage.none(ct_name),
                detail=(
                    f"no context of type {ct_name} supplied for gate {gate.id}; "
                    f"caller declared missing_ctx='report-skip', so this gate "
                    f"established nothing and is reported as SKIP, never PASS"
                ),
            )
        return gate._result(
            Verdict.ERROR,
            Coverage.none(ct_name),
            detail=(
                f"no context of type {ct_name} supplied for gate {gate.id} — unwired, not healthy"
            ),
        )

    ct = gate.context_type
    if ct is None:
        # Legacy broadcast: a typed map offers several contexts and choosing one
        # would be a guess; a bare context is handed over as-is, exactly as
        # GateRegistry.run has always done.
        if isinstance(contexts, Mapping):
            return gate._result(
                Verdict.ERROR,
                Coverage.none("contexts"),
                detail=(
                    f"gate {gate.id} declares no context_type and the sweep carries "
                    f"{len(contexts)} typed contexts — broadcasting any one of them "
                    f"would be a guess. Declare context_type on the gate class, or "
                    f"use GateRegistry.run with a single context."
                ),
            )
        return gate.run(contexts)

    ct_name = ct.__name__
    started = time.perf_counter()
    try:
        if isinstance(contexts, Mapping):
            resolved = _resolve_context(gate, contexts)
        elif isinstance(contexts, ct):
            resolved = contexts
        else:
            coerced = gate.coerce_context(contexts)
            resolved = coerced if coerced is not None else _MISSING
    except _AmbiguousMatch as amb:
        return gate._result(
            Verdict.ERROR,
            Coverage.none(ct_name),
            detail=(
                f"ambiguous context for gate {gate.id}: supplied contexts "
                f"{', '.join(amb.names)} all satisfy {ct_name}; dispatch refuses "
                f"to pick one arbitrarily"
            ),
        )
    except Exception as exc:  # noqa: BLE001 — coerce_context is gate-author code
        return _dispatch_error(gate, exc, started)
    if resolved is _MISSING:
        return unwired(ct_name)
    ctx = resolved
    if callable(ctx) and not isinstance(ctx, type):
        # A zero-argument factory, invoked now — once per consuming gate — inside
        # the same ERROR conversion Gate.run applies to check(). A factory for a
        # type no gate in this sweep consumes is never invoked.
        started = time.perf_counter()
        try:
            ctx = ctx()
        except Exception as exc:  # noqa: BLE001
            return _dispatch_error(gate, exc, started)
    return gate.run(ctx)


def run_event(
    registry: GateRegistry,
    event: Lifecycle | str,
    contexts: Mapping[type, Any],
    *,
    required: Iterable[str] = (),
    gate_ids: Iterable[str] | None = None,
    exclude: Iterable[str] = (),
    missing_ctx: str = "block",
) -> GateReport:
    """Run one lifecycle event, handing every gate the context it declared.

    This is the multi-context counterpart to :meth:`GateRegistry.run`. A training
    job registers gates from several context families — checkpoint, objective,
    parity — whose contexts are not interchangeable. ``run`` broadcasts one object
    to all of them and the mismatch dies inside ``check`` as a raw TypeError one
    frame down; ``run_event`` dispatches on :attr:`Gate.context_type` and turns
    wiring facts into verdicts:

    * No matching context: blocking ERROR — ``no context of type X supplied for
      gate Y — unwired, not healthy`` — unless the caller explicitly passes
      ``missing_ctx="report-skip"``, which surfaces the gate as SKIP instead. The
      default fails closed; an abstention must be declared.
    * A legacy gate (``context_type is None``) meeting a typed context map refuses
      to guess which entry it should get: blocking ERROR naming the fix.
    * Two supplied contexts both satisfying the declared type is ambiguous:
      blocking ERROR. Dispatch never picks arbitrarily.
    * A mapping value may be a zero-argument callable, invoked lazily once per
      consuming gate inside the sweep's ERROR conversion — a raising context
      factory is an ERROR verdict with a traceback, never an escape, and a factory
      for a type no gate consumes is never invoked.

    Args:
        registry: The registry to draw from.
        event: The lifecycle event, as a member or its value string.
        contexts: A mapping ``{context_type: context}`` covering the registered
            gates, or (during migration) one bare context object, matched by
            ``isinstance`` then ``coerce_context``. A ``Mapping`` is *always* read
            as a typed context map; to broadcast a mapping-shaped context itself,
            use :meth:`GateRegistry.run`.
        required: Ids that MUST run; any that did not are reported in
            :attr:`GateReport.missing` and block, as in :meth:`GateRegistry.run`.
        gate_ids: Restrict the sweep to these ids. An id not registered for this
            event lands in ``missing`` — silently dropping a typo would read as
            "all selected gates clear" over fewer gates than were asked for.
        exclude: Ids to leave out. Excluding a gate that was also required or
            selected is a contradiction; it lands in ``missing`` and blocks.
        missing_ctx: ``"block"`` (default) or ``"report-skip"``. Any other value
            raises ``ValueError`` — a misspelled abstention mode must not quietly
            weaken the sweep.

    A selection that runs zero gates inherits the registry's empty-sweep rule:
    blocking VACUOUS unless the event was declared ``event_allow_empty``. Gate
    failures never raise from here — they are verdicts; bad sweep arguments do.
    """
    event = Lifecycle(event)
    if missing_ctx not in ("block", "report-skip"):
        raise ValueError(
            f"missing_ctx must be 'block' or 'report-skip', got {missing_ctx!r} — "
            f"a misspelled abstention mode must not read as consent"
        )
    excluded = frozenset(exclude)
    event_gates = registry.for_event(event)
    unmatched: set[str] = set()
    if gate_ids is not None:
        wanted = set(gate_ids)
        unmatched = wanted - {g.id for g in event_gates}
        selected = [g for g in event_gates if g.id in wanted and g.id not in excluded]
    else:
        selected = [g for g in event_gates if g.id not in excluded]

    results = [_run_dispatched_gate(gate, contexts, missing_ctx) for gate in selected]

    ran = {g.id for g in selected}
    missing = tuple(sorted((set(required) | unmatched) - ran))
    if not selected and event not in registry._event_allow_empty:
        # The sweep-level all([]), as in GateRegistry.run: zero gates ran, so the
        # report must not be able to read as success — with a hint aimed at the
        # selection when the selection (not the registry) is what emptied it.
        results.append(
            GateResult(
                gate_id=f"{_EMPTY_SWEEP_GATE_PREFIX}{event.value}",
                verdict=Verdict.VACUOUS,
                coverage=Coverage.none("gates"),
                detail=(
                    f"no gates ran for {event.value}; a sweep over nothing proves "
                    f"nothing (broaden the gate_ids/exclude selection, or construct "
                    f"GateRegistry(event_allow_empty=...) if gateless is deliberate)"
                ),
                evidence={
                    "event": event.value,
                    "registered_gates": len(registry._gates),
                },
            )
        )
    return GateReport(
        event=event,
        results=tuple(results),
        missing=missing,
        registered=len(event_gates),
        # Stamped here as in GateRegistry.run: the report, not the runner, is
        # responsible for refusing a vacuous pass.
        allow_empty=event in registry._event_allow_empty,
    )


def verify_controls(
    registry: GateRegistry | None = None,
    *,
    gate_ids: Iterable[str] | None = None,
) -> list[str]:
    """Run every gate's controls. Returns a list of human-readable failures.

    Wire this into CI. It enforces three things a code review reliably misses:

    1. Every gate declares controls of BOTH kinds — at least one
       :attr:`ControlKind.MUST_FIRE` and at least one :attr:`ControlKind.MUST_PASS`.
       The MUST_PASS half is not implied by check (3): that check lives INSIDE the
       loop over declared controls, so a gate shipping only MUST_FIRE had its
       healthy-input behaviour verified ZERO times while every MUST_FIRE control
       held — and this function returned "all controls held" for what could be a
       detector that blocks on literally everything. A check nested in a
       data-driven loop cannot see the absence of its own trip; only an
       existence guard outside the loop can. (This is the ``compare_keys``
       NO_ELEMENTS defect, one meta level up.)
    2. Each MUST_FIRE control actually makes the gate block. This is the executable
       form of the review rule *name the positive control that proves your detector
       could have fired*.
    3. Each MUST_PASS control produces its DECLARED outcome. The default
       declaration is :attr:`Verdict.PASS`: a healthy fixture must affirmatively
       pass, because testing "not blocking" used to accept SKIP — and an
       abstention verifies neither "the gate's healthy-input behaviour is
       correct" nor "the gate is not firing on everything" (the purpose stated
       at the top of this module's controls documentation). A known-healthy
       fixture the gate genuinely cannot adjudicate declares
       :attr:`Control.expect_skip` with its reason; declaring it and then
       reaching PASS anyway is equally a failure — a stale declaration is a
       claim the evidence has outgrown, and doctrine (5) binds both directions.
    4. Each gate's MUST_PASS set must reach at least one real PASS in total.
       Every individual abstention may be honest and declared while the SET
       still certifies nothing: a gate that has never affirmatively accepted
       any input could block — or abstain — on everything and return green
       here. This is the zero-trip guard for the abstention lane, the same
       shape as the missing-MUST_PASS existence check above it, computed over
       observed verdicts on the far side of the loop that cannot see its own
       absence of trips.

    It also enforces three things about itself, because a verifier that loses
    findings to its own exceptions is the ``all([])`` bug one level up:

    * A gate whose :meth:`Gate.controls` raises is reported as a failure naming
      the gate and the exception — it is not re-raised from here and not
      skipped. Its controls were never shown to run; that is a finding.
    * An id in ``gate_ids`` that is not registered is reported as a failure
      naming the id — it is not raised as ``KeyError`` and not dropped. A
      dropped id would verify nothing and return ``[]``, reporting success
      over zero work.
    * A call that targets zero gates — an empty registry, gate modules never
      imported, or a ``gate_ids`` selection that matched nothing — is reported
      as a failure naming the fact. Returning ``[]`` from that call would be
      "all controls held" over zero controls.

    An empty return value means every control produced its DECLARED outcome —
    block, pass, or a declared and reasoned abstention — and every gate proved
    at least one affirmative healthy-input PASS. Anything less is a named
    failure, including the shapes "no gate was targeted", "the abstention was
    discovered rather than declared", and "every healthy fixture abstained".
    """
    reg = registry if registry is not None else REGISTRY
    failures: list[str] = []
    targets: list[Gate] = []
    if gate_ids is None:
        targets.extend(reg)
    else:
        for gate_id in gate_ids:
            if gate_id in reg:
                targets.append(reg.get(gate_id))
            else:
                failures.append(
                    f"{gate_id}: not registered — asked to verify controls for a gate "
                    f"that is not in the registry; dropping the id would let a typo "
                    f"read as 'all controls held'"
                )

    if not targets:
        # The verifier-layer all([]): whatever the cause — empty registry,
        # never-imported gate modules, a gate_ids selection that matched
        # nothing — returning [] here reports success over zero verified gates.
        failures.append(
            "verify_controls: 0 gates targeted — verified nothing. Import the "
            "gate package before calling, and check the gate_ids selection."
        )

    for gate in targets:
        try:
            controls = list(gate.controls())
        except Exception as exc:  # noqa: BLE001 — controls() is gate-author code
            failures.append(
                f"{gate.id}: controls() raised {type(exc).__name__}: {exc} — a gate "
                f"whose control list cannot be built has never been shown to fire"
            )
            continue
        if not any(c.kind is ControlKind.MUST_FIRE for c in controls):
            failures.append(
                f"{gate.id}: declares no MUST_FIRE control — a gate that has never "
                f"been shown to fire is not evidence of anything"
            )
        if not any(c.kind is ControlKind.MUST_PASS for c in controls):
            # The zero-trip guard for the MUST_PASS branch of the loop below —
            # same shape, same lesson, as the MUST_FIRE guard directly above,
            # and conspicuously missing until an adversarial sweep traced it: a
            # gate whose check() returns fail() unconditionally BLOCKS its own
            # MUST_FIRE control (verdict: holding), evaluates the per-control
            # MUST_PASS check zero times, and this function returned [] — "all
            # controls held" over zero healthy-input evaluations, certifying a
            # detector that blocks on everything as proven. An empty controls
            # list lands here too and earns both messages; each missing kind is
            # its own independently true, independently actionable finding.
            failures.append(
                f"{gate.id}: declares no MUST_PASS control — never shown to pass "
                f"a healthy input, so this gate could block on EVERYTHING and "
                f"verify green over zero healthy-input evaluations (and gates "
                f"that block unconditionally are the ones that get disabled)"
            )
        must_pass_total = sum(1 for c in controls if c.kind is ControlKind.MUST_PASS)
        must_pass_affirmed = 0
        # The affirmed counter is the healthy-input half's numerator. Per
        # doctrine (1) it is initialised to zero and may only be INCREMENTED by
        # an observed PASS, so an all-abstaining, all-erroring or all-blocking
        # MUST_PASS set cannot coast through the post-loop guard on a
        # success-valued initialiser. "At least one MUST_PASS declared" (the
        # existence guard above) never implied "at least one MUST_PASS passed":
        # the measured estate carried 16 MUST_PASS controls of which 2 returned
        # SKIP and were *accepted* — two controls certifying nothing — and had
        # the one affirming control regressed to SKIP as well, every gate would
        # still have certified green over zero healthy-input affirmations.
        for ctrl in controls:
            try:
                result = gate.run(ctrl.make_ctx())
            except Exception as exc:  # noqa: BLE001
                failures.append(
                    f"{gate.id}/{ctrl.name}: fixture raised {type(exc).__name__}: {exc}"
                )
                continue
            if ctrl.kind is ControlKind.MUST_FIRE and not result.blocking:
                failures.append(
                    f"{gate.id}/{ctrl.name}: MUST_FIRE control did not block "
                    f"(got {result.verdict.value}: {result.detail}). "
                    f"The defect was present and the gate reported success."
                )
            elif ctrl.kind is ControlKind.MUST_PASS:
                if result.blocking:
                    failures.append(
                        f"{gate.id}/{ctrl.name}: MUST_PASS control blocked "
                        f"(got {result.verdict.value}: {result.detail})"
                    )
                elif result.verdict is Verdict.PASS:
                    if ctrl.expect_skip:
                        # Doctrine (5) in the other direction: an abstention
                        # DECLARED for a fixture the gate now affirmatively
                        # adjudicates is a claim narrower than the evidence.
                        # The gate can judge this input today, so shipping the
                        # declaration would exempt this control from the proof
                        # it is now able to carry. Delete it.
                        failures.append(
                            f"{gate.id}/{ctrl.name}: MUST_PASS control declared "
                            f"expect_skip ({ctrl.expect_skip!r}) but the gate "
                            f"reached PASS — the declaration is stale; delete "
                            f"expect_skip so this control proves healthy-input "
                            f"behaviour again"
                        )
                    else:
                        must_pass_affirmed += 1
                elif not ctrl.expect_skip:
                    # The F2 acceptance, closed: "result not blocking" used to
                    # certify a SKIPPED MUST_PASS, and an abstention verifies
                    # neither half of the control's purpose. The expectation
                    # therefore lives on the fixture, where omission defaults
                    # to strictness: an author who expects SKIP says so, with
                    # the reason the input is genuinely unadjudicable; nothing
                    # else earns the abstention lane.
                    failures.append(
                        f"{gate.id}/{ctrl.name}: MUST_PASS control abstained "
                        f"(SKIP: {result.detail}) with no expect_skip declaration "
                        f"— an abstention certifies nothing about healthy-input "
                        f"behaviour; declare expect_skip=<reason> on the Control "
                        f"if this input is genuinely outside the gate's scope, "
                        f"or fix the gate to reach PASS here"
                    )
        if must_pass_total and must_pass_affirmed == 0:
            # The abstention lane's zero-trip guard, deliberately OUTSIDE the
            # data-driven loop that cannot see its own absence of trips: the
            # gate declared healthy fixtures (the no-MUST_PASS existence check
            # is satisfied), every fixture may even have behaved exactly as
            # declared — and the gate has still never been shown to accept
            # anything. At certification time that detector is indistinguishable
            # from one that abstains or blocks on EVERYTHING, and detectors
            # like that are the ones operators learn to route around.
            failures.append(
                f"{gate.id}: 0 of {must_pass_total} MUST_PASS control(s) reached "
                f"PASS — every healthy fixture abstained, blocked or errored, so "
                f"healthy-input behaviour was verified zero times; at least one "
                f"fixture that affirmatively passes is required, or this gate "
                f"is unproven in precisely the direction that gets gates disabled"
            )
    return failures
