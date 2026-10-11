"""``foundationscale.agentic_rl.evaluation`` — per-(model, harness) evaluation.

Every ``(model, harness)`` pair is reported separately with pass rate, tokens
used, latency, no-action turns and failure categories over
``fs.trajectory/v2`` :class:`Episode` objects.

Doctrine (FS):
  * Refusal messages name BOTH counts ("k of n ...").
  * Unmeasured is ``None``, never ``0.0``.
  * An empty group is refused, never a 0% row.
  * Mapping fields are frozen with ``types.MappingProxyType``.
  * Sequence fields are coerced to ``tuple`` in ``__post_init__``.

Stdlib only: dataclasses, enum, math, statistics, types.MappingProxyType, typing.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from statistics import median
from types import MappingProxyType
from typing import Any, NoReturn

from foundationscale.agentic_rl.trajectory_v2 import (
    Episode,
    EpisodeStatus,
    episode_reward,
)

__all__ = [
    "EvaluationRefusal",
    "FailureCategory",
    "EpisodeMeasure",
    "PairReport",
    "measure_episode",
    "report_by_pair",
    "render_markdown",
    "wilson_interval",
]

_WILSON_Z_DEFAULT = 1.959964
_DASH = "\u2014"


class EvaluationRefusal(ValueError):
    """Raised when an evaluation input violates a binding API invariant."""


def _refuse(msg: str) -> NoReturn:
    raise EvaluationRefusal(msg)


def _is_int(x: Any) -> bool:
    """True iff ``x`` is a real ``int`` (``bool`` is never an ``int`` here)."""
    return type(x) is int


def _is_number(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _is_nonneg_finite(x: Any) -> bool:
    return _is_number(x) and math.isfinite(float(x)) and float(x) >= 0.0


def _is_nonempty_str(x: Any) -> bool:
    return isinstance(x, str) and len(x) > 0


class FailureCategory(str, Enum):
    """Failure taxonomy for one evaluated episode."""

    PASSED = "passed"
    WRONG_ANSWER = "wrong_answer"
    NO_REWARD = "no_reward"
    TIMEOUT = "timeout"
    CONTEXT_OVERFLOW = "context_overflow"
    HARNESS_ERROR = "harness_error"
    SANDBOX_ERROR = "sandbox_error"
    INFRA_ERROR = "infra_error"
    VERIFIER_ERROR = "verifier_error"
    FIDELITY_MISMATCH = "fidelity_mismatch"


_STATUS_CATEGORY: Mapping[EpisodeStatus, FailureCategory] = MappingProxyType(
    {
        EpisodeStatus.TIMEOUT: FailureCategory.TIMEOUT,
        EpisodeStatus.CONTEXT_OVERFLOW: FailureCategory.CONTEXT_OVERFLOW,
        EpisodeStatus.HARNESS_ERROR: FailureCategory.HARNESS_ERROR,
        EpisodeStatus.SANDBOX_ERROR: FailureCategory.SANDBOX_ERROR,
        EpisodeStatus.INFRA_ERROR: FailureCategory.INFRA_ERROR,
        EpisodeStatus.VERIFIER_ERROR: FailureCategory.VERIFIER_ERROR,
        EpisodeStatus.FIDELITY_MISMATCH: FailureCategory.FIDELITY_MISMATCH,
    }
)


@dataclass(frozen=True)
class EpisodeMeasure:
    """Per-episode measurement over one ``fs.trajectory/v2`` Episode."""

    episode_id: str
    model: str
    model_version: int
    harness: str
    harness_version: str
    task_id: str
    category: FailureCategory
    reward: float | None
    sampled_tokens: int
    context_tokens: int
    turns: int
    no_action_turns: int
    actions_known: bool
    wall_ms: float | None

    def __post_init__(self) -> None:
        if not _is_nonempty_str(self.episode_id):
            _refuse("EpisodeMeasure.episode_id: expected non-empty str")
        if not _is_nonempty_str(self.model):
            _refuse("EpisodeMeasure.model: expected non-empty str")
        model_version: object = self.model_version
        if not _is_int(model_version) or self.model_version < 0:
            _refuse(f"EpisodeMeasure.model_version: expected int >= 0, got {self.model_version!r}")
        if not _is_nonempty_str(self.harness):
            _refuse("EpisodeMeasure.harness: expected non-empty str")
        if not _is_nonempty_str(self.harness_version):
            _refuse("EpisodeMeasure.harness_version: expected non-empty str")
        if not _is_nonempty_str(self.task_id):
            _refuse("EpisodeMeasure.task_id: expected non-empty str")
        category: object = self.category
        if not isinstance(category, FailureCategory):
            _refuse("EpisodeMeasure.category: expected FailureCategory")
        if self.reward is not None and not _is_number(self.reward):
            _refuse(f"EpisodeMeasure.reward: expected float or None, got {self.reward!r}")
        sampled_tokens: object = self.sampled_tokens
        if not _is_int(sampled_tokens) or self.sampled_tokens < 0:
            _refuse(
                f"EpisodeMeasure.sampled_tokens: expected int >= 0, got {self.sampled_tokens!r}"
            )
        context_tokens: object = self.context_tokens
        if not _is_int(context_tokens) or self.context_tokens < 0:
            _refuse(
                f"EpisodeMeasure.context_tokens: expected int >= 0, got {self.context_tokens!r}"
            )
        turns: object = self.turns
        if not _is_int(turns) or self.turns < 0:
            _refuse(f"EpisodeMeasure.turns: expected int >= 0, got {self.turns!r}")
        no_action_turns: object = self.no_action_turns
        if not _is_int(no_action_turns) or self.no_action_turns < 0:
            _refuse(
                f"EpisodeMeasure.no_action_turns: expected int >= 0, got {self.no_action_turns!r}"
            )
        if self.no_action_turns > self.turns:
            _refuse(
                f"EpisodeMeasure.no_action_turns: {self.no_action_turns} of {self.turns} turns "
                "are no-action turns (cannot exceed turns)"
            )
        actions_known: object = self.actions_known
        if not isinstance(actions_known, bool):
            _refuse(f"EpisodeMeasure.actions_known: expected bool, got {self.actions_known!r}")
        if self.wall_ms is not None and not _is_nonneg_finite(self.wall_ms):
            _refuse(
                f"EpisodeMeasure.wall_ms: expected None or finite float >= 0, got {self.wall_ms!r}"
            )


@dataclass(frozen=True)
class PairReport:
    """Aggregated report for one ``(model, harness)`` pair."""

    model: str
    harness: str
    episodes: int
    passed: int
    pass_rate: float
    pass_rate_ci95: tuple[float, float]
    mean_reward: float | None
    mean_sampled_tokens: float
    mean_turns: float
    no_action_turn_rate: float | None
    median_wall_ms: float | None
    failures: Mapping[str, int]
    tasks: int

    def __post_init__(self) -> None:
        if not _is_nonempty_str(self.model):
            _refuse("PairReport.model: expected non-empty str")
        if not _is_nonempty_str(self.harness):
            _refuse("PairReport.harness: expected non-empty str")
        episodes: object = self.episodes
        if not _is_int(episodes) or self.episodes < 1:
            _refuse(f"PairReport.episodes: expected int >= 1, got {self.episodes!r}")
        passed: object = self.passed
        if not _is_int(passed) or not 0 <= self.passed <= self.episodes:
            _refuse(f"PairReport.passed: expected 0 <= int <= {self.episodes}, got {self.passed!r}")
        if not _is_number(self.pass_rate) or not 0.0 <= float(self.pass_rate) <= 1.0:
            _refuse(f"PairReport.pass_rate: expected float in [0, 1], got {self.pass_rate!r}")
        ci = self.pass_rate_ci95
        if not (isinstance(ci, tuple) and len(ci) == 2):
            _refuse(f"PairReport.pass_rate_ci95: expected a 2-tuple, got {ci!r}")
        lo, hi = ci
        if not (_is_number(lo) and _is_number(hi) and 0.0 <= float(lo) <= float(hi) <= 1.0):
            _refuse(
                f"PairReport.pass_rate_ci95: expected (lo, hi) with 0 <= lo <= hi <= 1, got {ci!r}"
            )
        if self.mean_reward is not None and not _is_number(self.mean_reward):
            _refuse(f"PairReport.mean_reward: expected float or None, got {self.mean_reward!r}")
        if not _is_nonneg_finite(self.mean_sampled_tokens):
            _refuse(
                "PairReport.mean_sampled_tokens: expected finite float >= 0, "
                f"got {self.mean_sampled_tokens!r}"
            )
        if not _is_nonneg_finite(self.mean_turns):
            _refuse(f"PairReport.mean_turns: expected finite float >= 0, got {self.mean_turns!r}")
        if self.no_action_turn_rate is not None:
            if not _is_number(self.no_action_turn_rate):
                _refuse(
                    "PairReport.no_action_turn_rate: expected float or None, "
                    f"got {self.no_action_turn_rate!r}"
                )
            if not 0.0 <= float(self.no_action_turn_rate) <= 1.0:
                _refuse(
                    "PairReport.no_action_turn_rate: expected float in [0, 1], "
                    f"got {self.no_action_turn_rate!r}"
                )
        if self.median_wall_ms is not None and not _is_nonneg_finite(self.median_wall_ms):
            _refuse(
                "PairReport.median_wall_ms: expected None or finite float >= 0, "
                f"got {self.median_wall_ms!r}"
            )
        if not isinstance(self.failures, Mapping):
            _refuse(f"PairReport.failures: expected a Mapping, got {type(self.failures).__name__}")
        keys = list(self.failures.keys())
        if keys != sorted(keys):
            _refuse(f"PairReport.failures: expected keys sorted by key, got {keys!r}")
        bad_keys = sum(1 for k in keys if not _is_nonempty_str(k))
        if bad_keys:
            _refuse(f"PairReport.failures: {bad_keys} of {len(keys)} keys are not non-empty str")
        bad_vals = sum(1 for v in self.failures.values() if not (_is_int(v) and v > 0))
        if bad_vals:
            _refuse(
                f"PairReport.failures: {bad_vals} of {len(self.failures)} counts are not ints > 0 "
                "(only non-zero counts are stored)"
            )
        total_failures = sum(self.failures.values())
        if total_failures != self.episodes - self.passed:
            _refuse(
                f"PairReport.failures: counts sum to {total_failures} of "
                f"{self.episodes - self.passed} non-passed episodes"
            )
        tasks: object = self.tasks
        if not _is_int(tasks) or not 1 <= self.tasks <= self.episodes:
            _refuse(f"PairReport.tasks: expected 1 <= int <= {self.episodes}, got {self.tasks!r}")
        object.__setattr__(
            self,
            "pass_rate_ci95",
            (float(self.pass_rate_ci95[0]), float(self.pass_rate_ci95[1])),
        )
        object.__setattr__(
            self,
            "failures",
            MappingProxyType({str(k): int(v) for k, v in self.failures.items()}),
        )


def measure_episode(
    ep: Episode,
    *,
    pass_threshold: float = 1.0,
    tool_calls_per_generation: Mapping[str, int] | None = None,
) -> EpisodeMeasure:
    """Measure one :class:`Episode` into an :class:`EpisodeMeasure`.

    The category comes from ``ep.status`` (non-``OK`` statuses map 1:1;
    ``TIMEOUT`` maps to ``TIMEOUT`` regardless of the trainable flag), then the
    reward via :func:`trajectory_v2.episode_reward` with the ``"single"``
    reducer (more than one EPISODE-scope event propagates the refusal).

    ``sampled_tokens`` sums ``loss_mask == 1`` over all segments;
    ``context_tokens`` sums ``loss_mask == 0`` response-side tokens over all
    segments; ``turns`` counts generation spans over all segments.

    A span counts as an action when
    ``tool_calls_per_generation.get(gen_id, 0) > 0``; the LAST span of the
    episode is never a no-action turn (it ends the episode).  When
    ``tool_calls_per_generation`` is ``None``, ``no_action_turns`` is 0 and
    ``actions_known`` is ``False``.
    """
    if not isinstance(ep, Episode):
        _refuse(f"measure_episode.ep: expected Episode, got {type(ep).__name__}")
    if not _is_number(pass_threshold):
        _refuse(f"measure_episode.pass_threshold: expected float, got {pass_threshold!r}")
    if tool_calls_per_generation is not None and not isinstance(tool_calls_per_generation, Mapping):
        _refuse(
            "measure_episode.tool_calls_per_generation: expected a Mapping or None, "
            f"got {type(tool_calls_per_generation).__name__}"
        )

    reward = episode_reward(ep, reducer="single")

    if ep.status != EpisodeStatus.OK:
        category = _STATUS_CATEGORY[ep.status]
    elif reward is None:
        category = FailureCategory.NO_REWARD
    elif float(reward) >= float(pass_threshold):
        category = FailureCategory.PASSED
    else:
        category = FailureCategory.WRONG_ANSWER

    sampled_tokens = 0
    context_tokens = 0
    span_ids: list[str] = []
    for seg in ep.segments:
        for mask_bit in seg.loss_mask:
            if mask_bit == 1:
                sampled_tokens += 1
            else:
                context_tokens += 1
        for span in seg.generation_spans:
            span_ids.append(span[2])

    turns = len(span_ids)
    actions_known = tool_calls_per_generation is not None
    no_action_turns = 0
    if actions_known:
        assert tool_calls_per_generation is not None
        for pos, gen_id in enumerate(span_ids):
            if pos == turns - 1:
                continue
            if tool_calls_per_generation.get(gen_id, 0) <= 0:
                no_action_turns += 1

    run_ms = ep.timings.run_ms
    wall_ms = float(run_ms) if run_ms is not None else None

    return EpisodeMeasure(
        episode_id=ep.episode_id,
        model=ep.model.name,
        model_version=ep.model.version,
        harness=ep.harness.name,
        harness_version=ep.harness.version,
        task_id=ep.env.task_id,
        category=category,
        reward=None if reward is None else float(reward),
        sampled_tokens=sampled_tokens,
        context_tokens=context_tokens,
        turns=turns,
        no_action_turns=no_action_turns,
        actions_known=actions_known,
        wall_ms=wall_ms,
    )


def wilson_interval(successes: int, n: int, z: float = _WILSON_Z_DEFAULT) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion.

    ``center = (p + z^2 / (2n)) / (1 + z^2 / n)`` and
    ``half = z * sqrt(p(1 - p) / n + z^2 / (4n^2)) / (1 + z^2 / n)``, clipped
    to ``[0, 1]``.  Requires ``n >= 1`` and ``0 <= successes <= n``.
    """
    if not _is_int(n) or n < 1:
        _refuse(f"wilson_interval.n: expected int >= 1, got {n!r}")
    if not _is_int(successes) or not 0 <= successes <= n:
        _refuse(f"wilson_interval.successes: expected 0 <= int <= {n}, got {successes!r}")
    if not _is_number(z) or not math.isfinite(float(z)) or float(z) <= 0.0:
        _refuse(f"wilson_interval.z: expected finite float > 0, got {z!r}")

    p = successes / n
    zf = float(z)
    denom = 1.0 + zf * zf / n
    center = (p + zf * zf / (2.0 * n)) / denom
    half = zf * math.sqrt(p * (1.0 - p) / n + zf * zf / (4.0 * n * n)) / denom
    lo = min(1.0, max(0.0, center - half))
    hi = min(1.0, max(0.0, center + half))
    return (lo, hi)


def report_by_pair(measures: Sequence[EpisodeMeasure]) -> tuple[PairReport, ...]:
    """Group measures by ``(model, harness)`` and build one :class:`PairReport` per pair.

    Groups are sorted by ``(model, harness)``.  Empty input raises
    :class:`EvaluationRefusal`; an empty group is never reported as a 0% row.
    """
    if not isinstance(measures, Sequence):
        _refuse(f"report_by_pair.measures: expected a sequence, got {type(measures).__name__}")
    items = tuple(measures)
    if len(items) == 0:
        _refuse("report_by_pair.measures: expected a non-empty sequence, got 0 of 0 measures")
    bad = sum(1 for m in items if not isinstance(m, EpisodeMeasure))
    if bad:
        _refuse(f"report_by_pair.measures: {bad} of {len(items)} entries are not EpisodeMeasure")

    groups: dict[tuple[str, str], list[EpisodeMeasure]] = {}
    for m in items:
        groups.setdefault((m.model, m.harness), []).append(m)

    reports: list[PairReport] = []
    for key in sorted(groups):
        group = groups[key]
        n = len(group)

        passed = sum(1 for m in group if m.category is FailureCategory.PASSED)
        pass_rate = passed / n
        ci = wilson_interval(passed, n)

        rewards = [float(m.reward) for m in group if m.reward is not None]
        mean_reward = (sum(rewards) / len(rewards)) if rewards else None

        mean_sampled_tokens = sum(m.sampled_tokens for m in group) / n
        mean_turns = sum(m.turns for m in group) / n

        known = [m for m in group if m.actions_known]
        na_denom = sum(m.turns - 1 for m in known)
        if known and na_denom > 0:
            no_action_turn_rate = sum(m.no_action_turns for m in known) / na_denom
        else:
            no_action_turn_rate = None

        walls = [float(m.wall_ms) for m in group if m.wall_ms is not None]
        median_wall_ms = float(median(walls)) if walls else None

        counts: dict[str, int] = {}
        for m in group:
            if m.category is not FailureCategory.PASSED:
                counts[m.category.value] = counts.get(m.category.value, 0) + 1
        failures = MappingProxyType({k: counts[k] for k in sorted(counts)})

        tasks = len({m.task_id for m in group})

        reports.append(
            PairReport(
                model=key[0],
                harness=key[1],
                episodes=n,
                passed=passed,
                pass_rate=pass_rate,
                pass_rate_ci95=ci,
                mean_reward=mean_reward,
                mean_sampled_tokens=mean_sampled_tokens,
                mean_turns=mean_turns,
                no_action_turn_rate=no_action_turn_rate,
                median_wall_ms=median_wall_ms,
                failures=failures,
                tasks=tasks,
            )
        )

    return tuple(reports)


def _fmt(value: float | None, digits: int) -> str:
    return _DASH if value is None else f"{value:.{digits}f}"


def _fmt_failures(failures: Mapping[str, int]) -> str:
    if not failures:
        return _DASH
    return ", ".join(f"{key}={count}" for key, count in failures.items())


def render_markdown(reports: Sequence[PairReport]) -> str:
    """Render one Markdown table over ``reports``.

    Columns: model | harness | episodes | pass rate (95% CI) | mean reward |
    mean sampled tokens | mean turns | no-action turn rate | median wall s |
    failures.  Unmeasured values render as ``"\\u2014"``.
    """
    if not reports:
        raise EvaluationRefusal("render_markdown: 0 reports given; refusing an empty table")
    if not isinstance(reports, Sequence):
        _refuse(f"render_markdown.reports: expected a sequence, got {type(reports).__name__}")
    items = tuple(reports)
    bad = sum(1 for r in items if not isinstance(r, PairReport))
    if bad:
        _refuse(f"render_markdown.reports: {bad} of {len(items)} entries are not PairReport")

    header = (
        "| model | harness | episodes | pass rate (95% CI) | mean reward | "
        "mean sampled tokens | mean turns | no-action turn rate | median wall s | failures |"
    )
    separator = "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |"
    lines = [header, separator]
    for r in items:
        lo, hi = r.pass_rate_ci95
        pass_cell = f"{r.pass_rate:.3f} [{lo:.3f}, {hi:.3f}]"
        wall_s = None if r.median_wall_ms is None else r.median_wall_ms / 1000.0
        lines.append(
            f"| {r.model} | {r.harness} | {r.episodes} | {pass_cell} | "
            f"{_fmt(r.mean_reward, 3)} | {r.mean_sampled_tokens:.1f} | "
            f"{r.mean_turns:.2f} | {_fmt(r.no_action_turn_rate, 3)} | "
            f"{_fmt(wall_s, 3)} | {_fmt_failures(r.failures)} |"
        )
    return "\n".join(lines) + "\n"
