"""``foundationscale.agentic_rl.integrity`` — reward integrity gates (P0.7, section A).

Binding API for plan §3.7 threats 4, 5, 6, 8, 9.  Doctrine (FS):

  * Refusals raise :class:`IntegrityRefusal` (a ``ValueError`` subclass) and name
    BOTH counts ("k of n ...").
  * Unmeasured is ``None``, never ``0.0``.
  * ``bool`` is never accepted where a number is required.
  * All dataclasses are frozen and validate their own invariants in
    ``__post_init__``.

Stdlib only (plus :mod:`foundationscale.agentic_rl.trajectory_v2`).
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, NoReturn

from foundationscale.agentic_rl.trajectory_v2 import Episode, EpisodeStatus, episode_reward

__all__ = [
    "IntegrityRefusal",
    "IngestPolicy",
    "ingest_reward",
    "INFRA_STATUSES",
    "MaskDecision",
    "MaskReport",
    "mask_failures",
    "StageBudget",
    "check_stage",
    "audit_selected",
    "AuditOutcome",
    "AuditReport",
    "run_fp_audit",
    "require_no_false_positives",
    "SmokeResult",
    "check_task_smoke",
]


class IntegrityRefusal(ValueError):
    """Raised when a reward-integrity gate refuses to proceed."""


def _refuse(msg: str) -> NoReturn:
    raise IntegrityRefusal(msg)


def _is_number(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _is_finite(x: Any) -> bool:
    if not _is_number(x):
        return False
    try:
        return math.isfinite(float(x))
    except OverflowError:
        return False


def _is_nonneg_finite(x: Any) -> bool:
    if not _is_finite(x):
        return False
    try:
        return float(x) >= 0.0
    except OverflowError:
        return False


def _is_pos_finite(x: Any) -> bool:
    if not _is_finite(x):
        return False
    try:
        return float(x) > 0.0
    except OverflowError:
        return False


def _is_nonempty_str(x: Any) -> bool:
    return isinstance(x, str) and len(x) > 0


# ============================================================================
# threat 9: fail-closed reward ingestion
# ============================================================================


@dataclass(frozen=True)
class IngestPolicy:
    """Bounds and snapping rules for :func:`ingest_reward`."""

    low: float = 0.0
    high: float = 1.0
    snap_binary: bool = False

    def __post_init__(self) -> None:
        if not _is_finite(self.low):
            _refuse(f"IngestPolicy.low: expected finite float, got {self.low!r}")
        if not _is_finite(self.high):
            _refuse(f"IngestPolicy.high: expected finite float, got {self.high!r}")
        if float(self.low) >= float(self.high):
            _refuse(
                f"IngestPolicy: expected low < high, got low={self.low!r} and high={self.high!r}"
            )
        snap_binary: object = self.snap_binary
        if not isinstance(snap_binary, bool):
            _refuse(f"IngestPolicy.snap_binary: expected bool, got {snap_binary!r}")


def ingest_reward(raw: object, policy: IngestPolicy | None = None) -> float:
    """Ingest one raw reward value fail-closed.

    Accepts ``int``/``float`` (never ``bool``) or a ``str`` that parses with
    ``float()``.  Refuses ``None``, NaN, +/-Infinity, values outside
    ``[policy.low, policy.high]``, and (when ``policy.snap_binary``) anything
    not exactly ``low`` or ``high``.  Every refusal names the raw value.
    """
    if policy is None:
        policy = IngestPolicy()
    if not isinstance(policy, IngestPolicy):
        _refuse(f"ingest_reward.policy: expected IngestPolicy, got {type(policy).__name__}")

    if type(raw) is bool:
        _refuse(f"ingest_reward: refusing bool raw value {raw!r} (bool is not a number here)")
    if isinstance(raw, str):
        try:
            value = float(raw)
        except ValueError:
            _refuse(f"ingest_reward: refusing raw value {raw!r} (str does not parse as a float)")
        except OverflowError:
            _refuse(f"ingest_reward: refusing raw value {raw!r} (str overflows float range)")
    elif isinstance(raw, (int, float)) and _is_number(raw):
        try:
            value = float(raw)
        except OverflowError:
            _refuse(f"ingest_reward: refusing raw value {raw!r} (overflows float range)")
    else:
        _refuse(
            f"ingest_reward: refusing raw value {raw!r} "
            f"(expected int, float or str, got {type(raw).__name__})"
        )

    if not math.isfinite(value):
        _refuse(f"ingest_reward: refusing raw value {raw!r} (not finite)")
    if value < float(policy.low) or value > float(policy.high):
        _refuse(
            f"ingest_reward: refusing raw value {raw!r} (outside [{policy.low!r}, {policy.high!r}])"
        )
    if policy.snap_binary and value not in (float(policy.low), float(policy.high)):
        _refuse(
            f"ingest_reward: refusing raw value {raw!r} "
            f"(snap_binary requires exactly {policy.low!r} or {policy.high!r})"
        )
    return value


# ============================================================================
# threat 5: mask failed sandboxes, never score them
# ============================================================================

INFRA_STATUSES: frozenset[EpisodeStatus] = frozenset(
    {
        EpisodeStatus.SANDBOX_ERROR,
        EpisodeStatus.INFRA_ERROR,
        EpisodeStatus.VERIFIER_ERROR,
        EpisodeStatus.HARNESS_ERROR,
    }
)


@dataclass(frozen=True)
class MaskDecision:
    """Per-episode masking decision."""

    episode_id: str
    status: EpisodeStatus
    reward: float | None
    masked: bool
    reason: str

    def __post_init__(self) -> None:
        if not _is_nonempty_str(self.episode_id):
            _refuse("MaskDecision.episode_id: expected non-empty str")
        status: object = self.status
        if not isinstance(status, EpisodeStatus):
            _refuse(f"MaskDecision.status: expected EpisodeStatus, got {status!r}")
        if self.reward is not None and not _is_finite(self.reward):
            _refuse(f"MaskDecision.reward: expected None or finite float, got {self.reward!r}")
        masked: object = self.masked
        if not isinstance(masked, bool):
            _refuse(f"MaskDecision.masked: expected bool, got {masked!r}")
        if not isinstance(self.reason, str):
            _refuse(f"MaskDecision.reason: expected str, got {self.reason!r}")
        if self.masked:
            if self.reward is not None:
                _refuse(
                    f"MaskDecision: masked episode {self.episode_id!r} must have reward None, "
                    f"got {self.reward!r}"
                )
            if self.reason == "":
                _refuse(
                    f"MaskDecision: masked episode {self.episode_id!r} must name a non-empty reason"
                )
        else:
            if self.reason != "":
                _refuse(
                    f"MaskDecision: unmasked episode {self.episode_id!r} must have reason '', "
                    f"got {self.reason!r}"
                )


@dataclass(frozen=True)
class MaskReport:
    """Aggregate masking report over one episode batch."""

    decisions: tuple[MaskDecision, ...]
    total: int
    masked: int
    scored: int
    passed: int
    pass_rate: float | None
    pass_threshold: float = 1.0

    def __post_init__(self) -> None:
        decisions = self.decisions
        if not isinstance(decisions, tuple):
            _refuse(f"MaskReport.decisions: expected tuple, got {type(decisions).__name__}")
        bad = sum(1 for d in decisions if not isinstance(d, MaskDecision))
        if bad:
            _refuse(f"MaskReport.decisions: {bad} of {len(decisions)} entries are not MaskDecision")
        for name in ("total", "masked", "scored", "passed"):
            v: object = getattr(self, name)
            if type(v) is not int or v < 0:
                _refuse(f"MaskReport.{name}: expected int >= 0, got {v!r}")
        if self.total != len(decisions):
            _refuse(
                f"MaskReport: total must equal len(decisions); "
                f"{self.total} of {len(decisions)} decisions recorded"
            )
        masked_count = sum(1 for d in decisions if d.masked)
        if self.masked != masked_count:
            _refuse(
                f"MaskReport: masked must equal count of masked decisions; "
                f"{self.masked} of {masked_count} masked decisions"
            )
        scored_count = sum(1 for d in decisions if not d.masked)
        if self.scored != scored_count:
            _refuse(
                f"MaskReport: scored must equal count of unmasked decisions; "
                f"{self.scored} of {scored_count} unmasked decisions"
            )
        if self.masked + self.scored != self.total:
            _refuse(
                f"MaskReport: masked + scored must equal total; "
                f"{self.masked} + {self.scored} != {self.total} of {self.total} episodes"
            )
        if self.passed > self.scored:
            _refuse(f"MaskReport: passed must be <= scored; {self.passed} of {self.scored} scored")
        for d in decisions:
            if d.masked and d.reward is not None:
                _refuse(
                    f"MaskReport: masked episode {d.episode_id!r} must have reward None, "
                    f"got {d.reward!r}"
                )
            if not d.masked and d.reward is None:
                _refuse(
                    f"MaskReport: unmasked episode {d.episode_id!r} must have a non-None reward"
                )
        if self.scored == 0:
            if self.pass_rate is not None:
                _refuse(
                    "MaskReport.pass_rate: expected None when scored == 0 (unmeasured), "
                    f"got {self.pass_rate!r}"
                )
        else:
            if not _is_finite(self.pass_rate):
                _refuse(
                    f"MaskReport.pass_rate: expected finite float when scored > 0, "
                    f"got {self.pass_rate!r}"
                )
            pr = self.pass_rate
            if pr is None or not math.isclose(float(pr), self.passed / self.scored):
                _refuse(
                    f"MaskReport.pass_rate: expected {self.passed / self.scored!r} "
                    f"({self.passed} of {self.scored} scored), got {self.pass_rate!r}"
                )
        if not _is_finite(self.pass_threshold):
            _refuse(
                f"MaskReport.pass_threshold: expected finite float, got {self.pass_threshold!r}"
            )
        recomputed = sum(
            1
            for d in decisions
            if not d.masked and d.reward is not None and d.reward >= self.pass_threshold
        )
        if recomputed != self.passed:
            _refuse(
                f"MaskReport: passed must equal decisions at or above pass_threshold "
                f"{self.pass_threshold!r}; {self.passed} of {recomputed} recorded"
            )


def mask_failures(episodes: Sequence[Episode], *, pass_threshold: float = 1.0) -> MaskReport:
    """Mask failed sandboxes and unmeasured episodes; never score them.

    Episodes whose status is in :data:`INFRA_STATUSES` are masked (reward
    ``None``) even when a reward event exists.  An ``INFRA_STATUSES`` episode
    whose reward is ``>= pass_threshold`` is a hard refusal ("sandbox failure
    counted as a pass: k of n episodes ...") naming the ids.  An ``OK`` episode
    with no reward is masked with reason ``"no_reward"``.  An empty input is
    refused.
    """
    if not _is_finite(pass_threshold):
        _refuse(f"mask_failures.pass_threshold: expected finite float, got {pass_threshold!r}")
    eps = tuple(episodes)
    if len(eps) == 0:
        _refuse("mask_failures.episodes: expected a non-empty sequence, got 0 of 0 episodes")
    bad_ep = sum(1 for e in eps if not isinstance(e, Episode))
    if bad_ep:
        _refuse(f"mask_failures.episodes: {bad_ep} of {len(eps)} entries are not Episode")

    n = len(eps)
    decisions: list[MaskDecision] = []
    offending: list[str] = []
    masked = 0
    scored = 0
    passed = 0

    for ep in eps:
        reward = episode_reward(ep)
        if ep.status in INFRA_STATUSES:
            if reward is not None and reward >= float(pass_threshold):
                offending.append(ep.episode_id)
            decisions.append(
                MaskDecision(
                    episode_id=ep.episode_id,
                    status=ep.status,
                    reward=None,
                    masked=True,
                    reason=ep.status.value,
                )
            )
            masked += 1
            continue
        if reward is None:
            decisions.append(
                MaskDecision(
                    episode_id=ep.episode_id,
                    status=ep.status,
                    reward=None,
                    masked=True,
                    reason="no_reward",
                )
            )
            masked += 1
            continue
        decisions.append(
            MaskDecision(
                episode_id=ep.episode_id,
                status=ep.status,
                reward=reward,
                masked=False,
                reason="",
            )
        )
        scored += 1
        if reward >= float(pass_threshold):
            passed += 1

    if offending:
        k = len(offending)
        _refuse(
            f"sandbox failure counted as a pass: {k} of {n} episodes "
            f"have an INFRA status with reward >= {pass_threshold!r} "
            f"(ids: {', '.join(offending)})"
        )

    pass_rate: float | None = None if scored == 0 else passed / scored
    return MaskReport(
        decisions=tuple(decisions),
        total=n,
        masked=masked,
        scored=scored,
        passed=passed,
        pass_rate=pass_rate,
        pass_threshold=pass_threshold,
    )


# ============================================================================
# threat 4: stage timeouts
# ============================================================================


@dataclass(frozen=True)
class StageBudget:
    """Timeout budget for one pipeline stage."""

    task_timeout_s: float
    max_timeout_s: float
    override_timeout_s: float | None = None
    multiplier: float = 1.0
    grace_s: float = 5.0

    def __post_init__(self) -> None:
        if not _is_pos_finite(self.task_timeout_s):
            _refuse(
                f"StageBudget.task_timeout_s: expected finite float > 0, "
                f"got {self.task_timeout_s!r}"
            )
        if not _is_pos_finite(self.max_timeout_s):
            _refuse(
                f"StageBudget.max_timeout_s: expected finite float > 0, got {self.max_timeout_s!r}"
            )
        if self.override_timeout_s is not None and not _is_pos_finite(self.override_timeout_s):
            _refuse(
                f"StageBudget.override_timeout_s: expected None or finite float > 0, "
                f"got {self.override_timeout_s!r}"
            )
        if not _is_pos_finite(self.multiplier):
            _refuse(f"StageBudget.multiplier: expected finite float > 0, got {self.multiplier!r}")
        if not _is_nonneg_finite(self.grace_s):
            _refuse(f"StageBudget.grace_s: expected finite float >= 0, got {self.grace_s!r}")

    def effective_s(self) -> float:
        """``min(override or task, max) * multiplier``."""
        base = (
            float(self.override_timeout_s)
            if self.override_timeout_s is not None
            else float(self.task_timeout_s)
        )
        return min(base, float(self.max_timeout_s)) * float(self.multiplier)


def check_stage(stage: str, budget: StageBudget, elapsed_s: float | None) -> None:
    """Refuse on unmeasured, invalid, or over-budget stage elapsed time."""
    if not _is_nonempty_str(stage):
        _refuse(f"check_stage.stage: expected non-empty str, got {stage!r}")
    if not isinstance(budget, StageBudget):
        _refuse(f"check_stage.budget: expected StageBudget, got {type(budget).__name__}")
    if elapsed_s is None:
        _refuse(f"stage {stage}: elapsed unmeasured")
    if not _is_nonneg_finite(elapsed_s):
        _refuse(f"stage {stage}: elapsed must be a finite float >= 0, got {elapsed_s!r}")
    limit = budget.effective_s() + float(budget.grace_s)
    if float(elapsed_s) > limit:
        _refuse(
            f"stage overrun: stage {stage} elapsed {elapsed_s!r}s exceeds "
            f"effective {budget.effective_s()!r}s + grace {budget.grace_s!r}s "
            f"(limit {limit!r}s)"
        )


# ============================================================================
# threat 6: verifier false-positive audit
# ============================================================================


def audit_selected(episode_id: str, *, sample_rate: float, salt: str = "fs-fp-audit") -> bool:
    """Deterministic audit selection for one episode id."""
    if not _is_nonempty_str(episode_id):
        _refuse(f"audit_selected.episode_id: expected non-empty str, got {episode_id!r}")
    if not _is_finite(sample_rate) or not (0.0 <= float(sample_rate) <= 1.0):
        _refuse(f"audit_selected.sample_rate: expected finite float in [0, 1], got {sample_rate!r}")
    if not _is_nonempty_str(salt):
        _refuse(f"audit_selected.salt: expected non-empty str, got {salt!r}")
    digest = hashlib.sha256(f"{salt}:{episode_id}".encode()).hexdigest()[:8]
    return int(digest, 16) / 0xFFFFFFFF < float(sample_rate)


@dataclass(frozen=True)
class AuditOutcome:
    """Outcome of one audited verifier pass."""

    episode_id: str
    primary_reward: float
    audit_reward: float | None
    false_positive: bool | None

    def __post_init__(self) -> None:
        if not _is_nonempty_str(self.episode_id):
            _refuse("AuditOutcome.episode_id: expected non-empty str")
        if not _is_finite(self.primary_reward):
            _refuse(
                f"AuditOutcome.primary_reward: expected finite float, got {self.primary_reward!r}"
            )
        if self.audit_reward is not None and not _is_finite(self.audit_reward):
            _refuse(
                f"AuditOutcome.audit_reward: expected None or finite float, "
                f"got {self.audit_reward!r}"
            )
        fp: object = self.false_positive
        if fp is not None and not isinstance(fp, bool):
            _refuse(f"AuditOutcome.false_positive: expected bool or None, got {fp!r}")
        if self.audit_reward is None:
            if fp is not None:
                _refuse(
                    "AuditOutcome.false_positive: expected None when audit_reward is None "
                    f"(audit abstained), got {fp!r}"
                )
        elif not isinstance(fp, bool):
            _refuse(
                "AuditOutcome.false_positive: expected bool when audit_reward is not None, "
                f"got {fp!r}"
            )


@dataclass(frozen=True)
class AuditReport:
    """Aggregate report over one false-positive audit run."""

    outcomes: tuple[AuditOutcome, ...]
    verifier_passes: int
    audited: int
    abstained: int
    false_positives: int
    audited_fraction: float
    false_positive_rate: float | None

    def __post_init__(self) -> None:
        outcomes = self.outcomes
        if not isinstance(outcomes, tuple):
            _refuse(f"AuditReport.outcomes: expected tuple, got {type(outcomes).__name__}")
        bad = sum(1 for o in outcomes if not isinstance(o, AuditOutcome))
        if bad:
            _refuse(f"AuditReport.outcomes: {bad} of {len(outcomes)} entries are not AuditOutcome")
        for name in ("verifier_passes", "audited", "abstained", "false_positives"):
            v: object = getattr(self, name)
            if type(v) is not int or v < 0:
                _refuse(f"AuditReport.{name}: expected int >= 0, got {v!r}")
        if self.verifier_passes < 1:
            _refuse(
                f"AuditReport.verifier_passes: expected >= 1, got {self.verifier_passes} "
                f"of {self.verifier_passes} passes"
            )
        if self.audited > self.verifier_passes:
            _refuse(
                f"AuditReport: audited must be <= verifier_passes; "
                f"{self.audited} of {self.verifier_passes} passes audited"
            )
        if self.audited != len(outcomes):
            _refuse(
                f"AuditReport: audited must equal len(outcomes); "
                f"{self.audited} of {len(outcomes)} outcomes recorded"
            )
        abstained_count = sum(1 for o in outcomes if o.audit_reward is None)
        if self.abstained != abstained_count:
            _refuse(
                f"AuditReport: abstained must equal count of outcomes with audit_reward None; "
                f"{self.abstained} of {abstained_count} abstained outcomes"
            )
        fp_count = sum(1 for o in outcomes if o.false_positive is True)
        if self.false_positives != fp_count:
            _refuse(
                f"AuditReport: false_positives must equal count of outcomes with "
                f"false_positive True; {self.false_positives} of {fp_count} false positives"
            )
        if self.abstained > self.audited:
            _refuse(
                f"AuditReport: abstained must be <= audited; "
                f"{self.abstained} of {self.audited} audited"
            )
        if self.false_positives > self.audited - self.abstained:
            _refuse(
                f"AuditReport: false_positives must be <= audited - abstained; "
                f"{self.false_positives} of {self.audited - self.abstained} decided"
            )
        if not _is_finite(self.audited_fraction):
            _refuse(
                f"AuditReport.audited_fraction: expected finite float, "
                f"got {self.audited_fraction!r}"
            )
        if not math.isclose(float(self.audited_fraction), self.audited / self.verifier_passes):
            _refuse(
                f"AuditReport.audited_fraction: expected "
                f"{self.audited / self.verifier_passes!r} "
                f"({self.audited} of {self.verifier_passes} passes), "
                f"got {self.audited_fraction!r}"
            )
        decided = self.audited - self.abstained
        if decided == 0:
            if self.false_positive_rate is not None:
                _refuse(
                    "AuditReport.false_positive_rate: expected None when "
                    f"audited - abstained == 0 (unmeasured), got {self.false_positive_rate!r}"
                )
        else:
            if not _is_finite(self.false_positive_rate):
                _refuse(
                    f"AuditReport.false_positive_rate: expected finite float when "
                    f"audited - abstained > 0, got {self.false_positive_rate!r}"
                )
            fpr = self.false_positive_rate
            if fpr is None or not math.isclose(float(fpr), self.false_positives / decided):
                _refuse(
                    f"AuditReport.false_positive_rate: expected "
                    f"{self.false_positives / decided!r} "
                    f"({self.false_positives} of {decided} decided), "
                    f"got {self.false_positive_rate!r}"
                )


def run_fp_audit(
    passes: Mapping[str, float],
    audit: Callable[[str], float | None],
    *,
    sample_rate: float = 0.05,
    pass_threshold: float = 1.0,
    min_audited: int = 1,
    salt: str = "fs-fp-audit",
    forced: Iterable[str] = (),
) -> AuditReport:
    """Audit a sample of verifier passes for false positives.

    ``passes`` maps episode id -> primary reward; only entries ``>=
    pass_threshold`` count as verifier passes.  Selected ids are those chosen by
    :func:`audit_selected` plus ``forced`` (every forced id must be present in
    ``passes``).  If the selected count is below
    ``min(min_audited, verifier_passes)``, the lexicographically smallest
    unselected passes are added until the quota is met.  ``audit(id)`` returns
    the independent verifier's reward (``None`` = abstain); a false positive is
    ``audit_reward < pass_threshold``.  No passes -> refusal.  Exceptions from
    ``audit()`` propagate.
    """
    if not isinstance(passes, Mapping):
        _refuse(f"run_fp_audit.passes: expected a Mapping, got {type(passes).__name__}")
    if not callable(audit):
        _refuse(f"run_fp_audit.audit: expected a callable, got {type(audit).__name__}")
    if not _is_finite(sample_rate) or not (0.0 <= float(sample_rate) <= 1.0):
        _refuse(f"run_fp_audit.sample_rate: expected finite float in [0, 1], got {sample_rate!r}")
    if not _is_finite(pass_threshold):
        _refuse(f"run_fp_audit.pass_threshold: expected finite float, got {pass_threshold!r}")
    if type(min_audited) is not int or min_audited < 1:
        _refuse(f"run_fp_audit.min_audited: expected int >= 1, got {min_audited!r}")
    if not _is_nonempty_str(salt):
        _refuse(f"run_fp_audit.salt: expected non-empty str, got {salt!r}")

    forced_ids = tuple(forced)
    bad_forced = sum(1 for f in forced_ids if not _is_nonempty_str(f))
    if bad_forced:
        _refuse(
            f"run_fp_audit.forced: {bad_forced} of {len(forced_ids)} entries are not non-empty str"
        )

    pass_ids: dict[str, float] = {}
    for key, value in passes.items():
        if not _is_nonempty_str(key):
            _refuse(f"run_fp_audit.passes: key {key!r} is not a non-empty str")
        if not _is_finite(value):
            _refuse(f"run_fp_audit.passes[{key!r}]: expected finite float, got {value!r}")
        if float(value) >= float(pass_threshold):
            pass_ids[key] = float(value)

    verifier_passes = len(pass_ids)
    if verifier_passes == 0:
        _refuse(
            "run_fp_audit.passes: expected at least 1 verifier pass "
            f"(reward >= {pass_threshold!r}), got 0 of {len(passes)} entries"
        )

    missing = [f for f in forced_ids if f not in pass_ids]
    if missing:
        _refuse(
            f"run_fp_audit.forced: {len(missing)} of {len(forced_ids)} forced ids are not "
            f"verifier passes (ids: {', '.join(missing)})"
        )

    selected: list[str] = sorted(set(forced_ids))
    selected_set = set(selected)
    for ep_id in sorted(pass_ids):
        if ep_id in selected_set:
            continue
        if audit_selected(ep_id, sample_rate=float(sample_rate), salt=salt):
            selected.append(ep_id)
            selected_set.add(ep_id)

    quota = min(min_audited, verifier_passes)
    if len(selected) < quota:
        for ep_id in sorted(pass_ids):
            if ep_id in selected_set:
                continue
            selected.append(ep_id)
            selected_set.add(ep_id)
            if len(selected) >= quota:
                break

    outcomes: list[AuditOutcome] = []
    abstained = 0
    false_positives = 0
    for ep_id in selected:
        audit_reward = audit(ep_id)
        if audit_reward is not None and not _is_finite(audit_reward):
            _refuse(
                f"run_fp_audit.audit({ep_id!r}): expected None or finite float, "
                f"got {audit_reward!r}"
            )
        if audit_reward is None:
            fp: bool | None = None
            abstained += 1
        else:
            fp = float(audit_reward) < float(pass_threshold)
            if fp:
                false_positives += 1
        outcomes.append(
            AuditOutcome(
                episode_id=ep_id,
                primary_reward=pass_ids[ep_id],
                audit_reward=audit_reward,
                false_positive=fp,
            )
        )

    audited = len(outcomes)
    decided = audited - abstained
    return AuditReport(
        outcomes=tuple(outcomes),
        verifier_passes=verifier_passes,
        audited=audited,
        abstained=abstained,
        false_positives=false_positives,
        audited_fraction=audited / verifier_passes,
        false_positive_rate=None if decided == 0 else false_positives / decided,
    )


def require_no_false_positives(report: AuditReport) -> None:
    """Refuse when any audited pass is a verifier false positive."""
    if not isinstance(report, AuditReport):
        _refuse(
            f"require_no_false_positives.report: expected AuditReport, got {type(report).__name__}"
        )
    if report.false_positives > 0:
        ids = [o.episode_id for o in report.outcomes if o.false_positive is True]
        _refuse(
            f"verifier false positives: {report.false_positives} of {report.audited} audited "
            f"(ids: {', '.join(ids)})"
        )


# ============================================================================
# threat 8: oracle / empty smoke
# ============================================================================


@dataclass(frozen=True)
class SmokeResult:
    """Oracle and empty-prompt smoke rewards for one task."""

    task_id: str
    oracle_reward: float | None
    empty_reward: float | None

    def __post_init__(self) -> None:
        if not _is_nonempty_str(self.task_id):
            _refuse("SmokeResult.task_id: expected non-empty str")
        if self.oracle_reward is not None and not _is_finite(self.oracle_reward):
            _refuse(
                f"SmokeResult.oracle_reward: expected None or finite float, "
                f"got {self.oracle_reward!r}"
            )
        if self.empty_reward is not None and not _is_finite(self.empty_reward):
            _refuse(
                f"SmokeResult.empty_reward: expected None or finite float, "
                f"got {self.empty_reward!r}"
            )


def check_task_smoke(results: Sequence[SmokeResult], *, pass_threshold: float = 1.0) -> None:
    """Refuse when any task fails its oracle / empty-prompt smoke check.

    Each task must have ``oracle_reward >= pass_threshold`` AND
    ``empty_reward < pass_threshold``; ``None`` counts as broken on either side.
    """
    if not _is_finite(pass_threshold):
        _refuse(f"check_task_smoke.pass_threshold: expected finite float, got {pass_threshold!r}")
    rs = tuple(results)
    if len(rs) == 0:
        _refuse("check_task_smoke.results: expected a non-empty sequence, got 0 of 0 tasks")
    bad = sum(1 for r in rs if not isinstance(r, SmokeResult))
    if bad:
        _refuse(f"check_task_smoke.results: {bad} of {len(rs)} entries are not SmokeResult")

    n = len(rs)
    broken: list[str] = []
    for r in rs:
        failures: list[str] = []
        if r.oracle_reward is None:
            failures.append("oracle_reward unmeasured (None)")
        elif float(r.oracle_reward) < float(pass_threshold):
            failures.append(f"oracle_reward {r.oracle_reward!r} < {pass_threshold!r}")
        if r.empty_reward is None:
            failures.append("empty_reward unmeasured (None)")
        elif float(r.empty_reward) >= float(pass_threshold):
            failures.append(f"empty_reward {r.empty_reward!r} >= {pass_threshold!r}")
        if failures:
            broken.append(f"{r.task_id} ({'; '.join(failures)})")

    if broken:
        k = len(broken)
        _refuse(f"task smoke broken: {k} of {n} tasks broken (ids: {'; '.join(broken)})")
