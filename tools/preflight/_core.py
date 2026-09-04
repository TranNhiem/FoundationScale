"""Check results, the clearance algebra, and the check/lane registry."""

from __future__ import annotations

import time
import traceback
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from ._base import (
    Coverage,
    Verdict,
)

if TYPE_CHECKING:
    from ._fixtures import _World


# ---------------------------------------------------------------------------
# Check results + the clearance algebra
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CheckResult:
    check_id: str
    title: str
    verdict: Verdict
    coverage: Coverage
    detail: str = ""
    evidence: Mapping[str, Any] = field(default_factory=dict)
    duration_s: float = 0.0

    @property
    def blocking(self) -> bool:
        # For CLEAR purposes every non-PASS blocks (see clearance predicate), but
        # this property mirrors core semantics for report anatomy.
        return self.verdict is not Verdict.PASS

    def render(self) -> str:
        line = f"[{self.verdict.symbol:>7}] {self.check_id}: {self.coverage}"
        if self.detail:
            line += f" — {self.detail}"
        if self.verdict is not Verdict.PASS:
            # Design item 4's vocabulary, verbatim, on every non-clearing line:
            # a reader must never have to know that SKIP is 'non-blocking' in
            # gate-land to understand that this launch is not cleared.
            line += "  (NOT-VERIFIED)"
        return line

    def to_dict(self) -> dict[str, Any]:
        return {
            "check_id": self.check_id,
            "title": self.title,
            "verdict": self.verdict.value,
            "checked": self.coverage.checked,
            "expected": self.coverage.expected,
            "unit": self.coverage.unit,
            "detail": self.detail,
            "evidence": dict(self.evidence),
            "duration_s": round(self.duration_s, 4),
        }


def _finalize(
    check_id: str,
    title: str,
    verdict: Verdict,
    coverage: Coverage,
    detail: str,
    evidence: Mapping[str, Any] | None,
) -> CheckResult:
    """The Gate.ok downgrade ladder, restated for checks.

    Same order, same semantics: a requested PASS over zero units becomes
    VACUOUS; short-of-denominator becomes UNDERCOVERED; numerator outrunning
    the denominator becomes OVERCOVERED. The check author cannot override any
    of the three, which — as in Gate.ok — is the point.
    """
    evidence = dict(evidence or {})
    if verdict is Verdict.PASS:
        if coverage.is_vacuous:
            return CheckResult(
                check_id,
                title,
                Verdict.VACUOUS,
                coverage,
                f"check examined 0 {coverage.unit} and therefore proves nothing"
                + (f" (claimed: {detail})" if detail else ""),
                evidence,
            )
        if coverage.is_short and not coverage.sampled:
            return CheckResult(
                check_id,
                title,
                Verdict.UNDERCOVERED,
                coverage,
                f"examined {coverage.checked} of {coverage.expected} {coverage.unit}"
                + (f" (claimed: {detail})" if detail else ""),
                evidence,
            )
        if coverage.is_over:
            return CheckResult(
                check_id,
                title,
                Verdict.OVERCOVERED,
                coverage,
                f"examined {coverage.checked} of {coverage.expected} {coverage.unit} — "
                f"the numerator outruns the denominator; one of them is wrong"
                + (f" (claimed: {detail})" if detail else ""),
                evidence,
            )
    return CheckResult(check_id, title, verdict, coverage, detail, evidence)


def _discipline(res: CheckResult) -> CheckResult:
    """PASS with an EMPTY evidence map is not a PASS — design item 4 made the
    evidence payload part of the record shape, and an absent one is the claim
    'we counted and hashed' detached from every count and hash. Downgrade to
    ERROR: it is an author defect in this file, not a property of the run."""
    if res.verdict is Verdict.PASS and not res.evidence:
        return CheckResult(
            res.check_id,
            res.title,
            Verdict.ERROR,
            res.coverage,
            "check returned PASS with an empty evidence map — the verdict schema "
            "requires counts/hashes/files attached to every clearance",
            {"author_defect": True},
            res.duration_s,
        )
    return res


def _is_clear(results: Sequence[CheckResult]) -> bool:
    # The single sentence this whole file exists to enforce.
    return bool(results) and all(
        r.verdict is Verdict.PASS and r.coverage.checked > 0 for r in results
    )


# ---------------------------------------------------------------------------
# Check registry
# ---------------------------------------------------------------------------


@dataclass
class _Lane:
    """One MUST_FIRE control: apply() corrupts a fresh fixture world; the check
    must then NOT return PASS. description names the injected defect."""

    name: str
    description: str
    apply: Callable[[_World], None]


@dataclass
class _Check:
    id: str
    title: str
    section: str | None  # config section handed to fn; None for meta checks
    fn: Callable[[dict, dict, dict, dict, Sequence[_Check] | None], CheckResult]
    lanes: tuple[_Lane, ...] = ()


_REGISTRY_ORDER: list[_Check] = []

REGISTRY: dict[str, _Check] = {}


def _register(
    check_id: str,
    title: str,
    section: str | None,
    lanes: Sequence[_Lane] = (),
) -> Callable[[Callable[..., CheckResult]], Callable[..., CheckResult]]:
    def deco(fn: Callable[..., CheckResult]) -> Callable[..., CheckResult]:
        if check_id in REGISTRY:
            raise ValueError(f"duplicate preflight check id {check_id!r}")
        chk = _Check(check_id, title, section, fn, tuple(lanes))
        REGISTRY[check_id] = chk
        _REGISTRY_ORDER.append(chk)
        return fn

    return deco


def _execute(
    chk: _Check,
    cfg: dict,
    env: Mapping[str, str],
    shared: dict,
    registry: Sequence[_Check] | None = None,
) -> CheckResult:
    """One check, timed, fail-closed. Mirrors Gate.run's conversion discipline:
    an exception inside a check is ERROR and blocks; a check that returns
    anything but a CheckResult is ERROR and blocks."""
    t0 = time.perf_counter()
    try:
        section = cfg.get(chk.section, {}) if chk.section else {}
        res = chk.fn(cfg, section, env, shared, registry)
        if not isinstance(res, CheckResult):
            res = CheckResult(
                chk.id,
                chk.title,
                Verdict.ERROR,
                Coverage.none("units"),
                f"check returned {type(res).__name__}, expected CheckResult",
            )
    except Exception as exc:  # noqa: BLE001 — fail closed, deliberately broad
        res = CheckResult(
            chk.id,
            chk.title,
            Verdict.ERROR,
            Coverage.none("units"),
            f"{type(exc).__name__}: {exc}",
            {"traceback": traceback.format_exc(limit=12)},
        )
    res = _discipline(res)
    return CheckResult(
        res.check_id,
        res.title,
        res.verdict,
        res.coverage,
        res.detail,
        res.evidence,
        time.perf_counter() - t0,
    )


def _shared_or_error(chk: _Check, shared: dict, need: Sequence[str]) -> CheckResult | None:
    """Cross-check denominators (manifest hash, pinned corpus order) exist only
    if frozen_manifest ran and passed. Unwired shared state BLOCKS the consumer
    with the cause named — doctrine 4 applied to the tool's own plumbing."""
    missing = [k for k in need if k not in shared]
    if missing:
        return CheckResult(
            chk.id,
            chk.title,
            Verdict.ERROR,
            Coverage.none("units"),
            f"frozen_manifest did not establish {missing!r}; downstream checks "
            f"may not source denominators from anywhere else (design item 1)",
        )
    return None


def _stub_fn(*_a: Any, **_k: Any) -> CheckResult:  # never called; placeholder for _shared_or_error
    raise AssertionError("stub")
