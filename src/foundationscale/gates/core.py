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
   :attr:`Verdict.UNDERCOVERED` unless it explicitly declares itself a sample.
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

    There are two ways to not-block (:attr:`PASS`, :attr:`SKIP`) and four ways to block.
    The asymmetry is intentional: it is much easier to accidentally produce a
    meaningless success than a meaningless failure.
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

    SKIP = "SKIP"
    """Explicitly not applicable. Requires a reason. Does not block, but is reported."""

    ERROR = "ERROR"
    """The gate itself raised. Blocks — gates fail closed."""

    @property
    def blocking(self) -> bool:
        """Whether this verdict should stop the job or prevent promotion."""
        return self in (Verdict.FAIL, Verdict.VACUOUS, Verdict.UNDERCOVERED, Verdict.ERROR)

    @property
    def symbol(self) -> str:
        return {
            Verdict.PASS: "ok",
            Verdict.FAIL: "FAIL",
            Verdict.VACUOUS: "VACUOUS",
            Verdict.UNDERCOVERED: "UNDER",
            Verdict.SKIP: "skip",
            Verdict.ERROR: "ERROR",
        }[self]


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
            fabricate one to make a ratio look complete.
        sampled: Set ``True`` to declare deliberately partial coverage. This converts
            what would be :attr:`Verdict.UNDERCOVERED` into a pass — so it requires
            ``sample_reason`` and is surfaced in every rendering.
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
    def fraction(self) -> float | None:
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
            "checked": self.coverage.checked,
            "expected": self.coverage.expected,
            "unit": self.coverage.unit,
            "sampled": self.coverage.sampled,
            "sample_reason": self.coverage.sample_reason or None,
            "detail": self.detail,
            "evidence": dict(self.evidence),
            "duration_s": round(self.duration_s, 4),
        }


@dataclass(frozen=True)
class GateReport:
    """The result of running every gate registered for one lifecycle event."""

    event: Lifecycle
    results: tuple[GateResult, ...]
    missing: tuple[str, ...] = ()
    """Gates the caller declared required that did not run. Blocks the whole report.

    A registry that silently ran zero gates is the same failure as a gate that
    silently checked zero units, one level up.
    """

    @property
    def blocking(self) -> tuple[GateResult, ...]:
        return tuple(r for r in self.results if r.blocking)

    @property
    def ok(self) -> bool:
        return not self.blocking and not self.missing

    def render(self) -> str:
        head = f"gates @ {self.event.value}: {len(self.results)} run"
        if self.ok:
            head += " — all clear"
        else:
            bits = []
            if self.blocking:
                bits.append(f"{len(self.blocking)} blocking")
            if self.missing:
                bits.append(f"{len(self.missing)} MISSING")
            head += " — " + ", ".join(bits)
        lines = [head]
        lines += ["  " + r.render() for r in self.results]
        lines += [f"  [MISSING] {g}: required but never ran" for g in self.missing]
        return "\n".join(lines)

    def to_json(self, **kwargs: Any) -> str:
        return json.dumps(
            {
                "event": self.event.value,
                "ok": self.ok,
                "missing": list(self.missing),
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
    just as useless and tends to get disabled."""


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


class ControlFailure(AssertionError):
    """A gate failed its own control."""


class Gate(ABC):
    """Base class for all gates.

    Subclasses set the three class attributes, implement :meth:`check`, and declare at
    least one :class:`Control` of kind :attr:`ControlKind.MUST_FIRE`.

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
                bad = [e for e in experts if e.nbytes != ctx.expected_nbytes(e)]
                cov = Coverage(len(experts), "expert tensors",
                               expected=ctx.declared_expert_count)
                if bad:
                    return self.fail(f"{len(bad)} experts with wrong byte count", cov,
                                     evidence={"offenders": [e.name for e in bad[:8]]})
                return self.ok("all expert byte counts match", cov)

            def controls(self):
                return [Control("aliased-16-of-128", ControlKind.MUST_FIRE,
                                make_aliased_ckpt,
                                note="128 experts collapsed to 16 by local-name save")]

    If ``experts`` comes back empty — the corrupt-artifact case — ``self.ok`` returns
    ``VACUOUS`` rather than ``PASS``, and the author did not have to remember to handle
    it.
    """

    id: ClassVar[str]
    description: ClassVar[str]
    events: ClassVar[tuple[Lifecycle, ...]]

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
        and not declared a sample, the verdict is :attr:`Verdict.UNDERCOVERED`. The
        gate author cannot override this, which is the point.
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

    def skip(self, reason: str) -> GateResult:
        """Declare the gate not applicable. Requires a reason; does not block."""
        if not reason.strip():
            raise ValueError("skip() requires a reason — an unexplained skip is a hole")
        return self._result(Verdict.SKIP, Coverage.none("units"), detail=reason)

    def _result(
        self,
        verdict: Verdict,
        coverage: Coverage,
        *,
        detail: str = "",
        evidence: Mapping[str, Any] | None = None,
    ) -> GateResult:
        return GateResult(
            gate_id=self.id,
            verdict=verdict,
            coverage=coverage,
            detail=detail,
            evidence=dict(evidence or {}),
        )

    # -- the gate itself ----------------------------------------------------------

    @abstractmethod
    def check(self, ctx: Any) -> GateResult:
        """Run the check. Must return via :meth:`ok`/:meth:`fail`/:meth:`skip`."""

    @abstractmethod
    def controls(self) -> Sequence[Control]:
        """Fixtures proving this gate works. At least one MUST_FIRE, enforced in CI."""

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


class GateRegistry:
    """Holds gates and runs them by lifecycle event.

    The registry exists so that gate invocation is a property of the *event*, not of
    whichever launcher script happens to be running. In the audited estate the export
    byte check lived as a copy-pasted heredoc in one script and was simply absent from
    the other, which is how a truncated export reached ``rc=0``.
    """

    def __init__(self) -> None:
        self._gates: dict[str, Gate] = {}

    def register(self, gate: Gate) -> Gate:
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
        """
        gates = self.for_event(event)
        results = tuple(g.run(ctx) for g in gates)
        missing: tuple[str, ...] = ()
        if required is not None:
            ran = {g.id for g in gates}
            missing = tuple(sorted(set(required) - ran))
        return GateReport(event=event, results=results, missing=missing)


REGISTRY = GateRegistry()
"""Process-wide default registry."""


def register(cls: type[Gate]) -> type[Gate]:
    """Class decorator: instantiate a gate and add it to :data:`REGISTRY`."""
    REGISTRY.register(cls())
    return cls


def verify_controls(
    registry: GateRegistry | None = None,
    *,
    gate_ids: Iterable[str] | None = None,
) -> list[str]:
    """Run every gate's controls. Returns a list of human-readable failures.

    Wire this into CI. It enforces three things a code review reliably misses:

    1. Every gate declares at least one :attr:`ControlKind.MUST_FIRE` control. A gate
       with no control has never been shown to detect anything.
    2. Each MUST_FIRE control actually makes the gate block. This is the executable
       form of the review rule *name the positive control that proves your detector
       could have fired*.
    3. Each MUST_PASS control does not block, so gates that block unconditionally are
       caught before someone disables them.

    An empty return value means all controls held.
    """
    reg = registry if registry is not None else REGISTRY
    failures: list[str] = []
    targets = list(reg) if gate_ids is None else [reg.get(g) for g in gate_ids]

    for gate in targets:
        controls = list(gate.controls())
        if not any(c.kind is ControlKind.MUST_FIRE for c in controls):
            failures.append(
                f"{gate.id}: declares no MUST_FIRE control — a gate that has never "
                f"been shown to fire is not evidence of anything"
            )
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
            elif ctrl.kind is ControlKind.MUST_PASS and result.blocking:
                failures.append(
                    f"{gate.id}/{ctrl.name}: MUST_PASS control blocked "
                    f"(got {result.verdict.value}: {result.detail})"
                )
    return failures
