"""Stability guardrails for the FoundationScale agentic RL plane.

This module is the STABILITY PLANE of the agentic RL feature: it measures what a
rollout step produced, tracks the exponential moving average of the measured
reward, and refuses to let a collapse pass unremarked. It depends on the
standard library and ``foundationscale.agentic_rl.contracts`` only.

Three rules run through every type here, and each names the failure it
prevents:

* Unmeasured is ``None``, never ``0.0``. A step whose trajectories all abstained
  carries ``reward_mean=None`` and ``zero_reward_fraction=None``; the guard's
  EMA is then unchanged and the step raises no alarm, because an unmeasured step
  is not a measured drop. A ``0.0`` mean is a real measurement of a real zero.
* Refusals name both counts. Every :class:`StabilityRefusal` carries the failed
  field and BOTH sides of any count or bound it compares -- a refusal that says
  only "too small" leaves the caller unable to see what was compared.
* ``True`` is never ``1``. Every int check is ``type(x) is int`` and every bool
  check is ``type(x) is bool``, so a boolean in place of a count or a step is
  refused rather than read as its numeric value.

WHAT IS NOT CLAIMED: anything about WHY a reward moved (the harness and the
scorer own that); any action taken on an alarm -- :class:`StabilityGuard` reports
a verdict and the HOST decides, raising :class:`CollapseDetected` when the
configured action is ``"stop"``; and any persistence of guard state across
processes, which is the host's concern.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TypeGuard

from foundationscale.agentic_rl.contracts import Trajectory

__all__ = (
    "CollapseDetected",
    "GuardConfig",
    "GuardVerdict",
    "RolloutStats",
    "StabilityGuard",
    "StabilityRefusal",
    "rollout_stats",
    "verdict_line",
)


class StabilityRefusal(ValueError):
    """Raised when a rollout statistic, a guard config or a verdict violates its contract.

    Every message names the failed field and BOTH sides of any count or bound it
    compares. Fail closed: never coerce a bool to a count, never substitute
    ``0.0`` for an unmeasured mean, never pad a best-k tuple to length.
    """

    pass


class CollapseDetected(RuntimeError):
    """Raised by the host when the guard alarms and the configured action is ``"stop"``.

    The guard itself never raises this: it reports a verdict and the host decides.
    The host raises it AFTER publishing the checkpoint, so the last checkpoint
    exists when the run stops.
    """

    pass


def _is_int(value: object) -> TypeGuard[int]:
    # `type(value) is int`, deliberately NOT isinstance: `True` is an `int`
    # under isinstance, and a bool silently read as a count or a step is exactly
    # the coercion this plane refuses ("1 is not True").
    return type(value) is int


def _is_bool(value: object) -> TypeGuard[bool]:
    return type(value) is bool


def _is_real(value: object) -> TypeGuard[float]:
    # Narrowed to float for typing only: an int entry is stored VERBATIM, and a
    # bool is refused because `type(value) is float`/`is int` excludes it.
    return type(value) in (int, float)


def _described(value: object) -> str:
    # A refusal names what it refused: value first, concrete type second, so a
    # bool never reads as an int nor text as a count.
    return f"{value!r} ({type(value).__name__})"


@dataclass(frozen=True)
class RolloutStats:
    """What one rollout step measured, stated or refused.

    ``scored`` counts trajectories whose ``reward`` is not ``None``; ``abstained``
    counts those whose ``reward`` is ``None``. ``reward_mean`` and
    ``zero_reward_fraction`` are the mean and the zero fraction over the SCORED
    trajectories only, and both are ``None`` when ``scored == 0`` -- an unmeasured
    step is never reported as a measured ``0.0``.
    """

    step: int
    scored: int
    abstained: int
    reward_mean: float | None
    zero_reward_fraction: float | None

    def __post_init__(self) -> None:
        where = f"rollout stats step={self.step!r}"
        if not _is_int(self.step):
            raise StabilityRefusal(
                f"{where}: field 'step' is {_described(self.step)}: a step is a real int "
                f"(type(x) is int) -- 1 is not True -- and a bool would alias the step it "
                f"numerically equals"
            )
        if self.step < 0:
            raise StabilityRefusal(
                f"{where}: field 'step' is {self.step}: steps are >= 0 and order the run"
            )
        if not _is_int(self.scored):
            raise StabilityRefusal(
                f"{where}: field 'scored' is {_described(self.scored)}: a count is a real int "
                f"-- 1 is not True"
            )
        if not _is_int(self.abstained):
            raise StabilityRefusal(
                f"{where}: field 'abstained' is {_described(self.abstained)}: a count is a real "
                f"int -- 1 is not True"
            )
        if self.scored < 0:
            raise StabilityRefusal(
                f"{where}: field 'scored' is {self.scored}: a count of scored trajectories is >= 0"
            )
        if self.abstained < 0:
            raise StabilityRefusal(
                f"{where}: field 'abstained' is {self.abstained}: a count of abstained "
                f"trajectories is >= 0"
            )
        total = self.scored + self.abstained
        if total == 0:
            raise StabilityRefusal(
                f"{where}: field 'scored' is {self.scored} and field 'abstained' is "
                f"{self.abstained}: the two counts sum to 0, and stats over zero trajectories "
                f"measure nothing -- an empty step is refused rather than reported as a "
                f"measured mean of nothing"
            )
        if self.scored == 0:
            if self.reward_mean is not None:
                raise StabilityRefusal(
                    f"{where}: field 'reward_mean' is {_described(self.reward_mean)} but field "
                    f"'scored' is 0: a mean over zero scored trajectories is unmeasured and "
                    f"stays None -- 0.0 is a real mean of real zeros"
                )
            if self.zero_reward_fraction is not None:
                raise StabilityRefusal(
                    f"{where}: field 'zero_reward_fraction' is "
                    f"{_described(self.zero_reward_fraction)} but field 'scored' is 0: a "
                    f"fraction over zero scored trajectories is unmeasured and stays None -- "
                    f"0.0 is a real fraction of real zeros"
                )
        else:
            if self.reward_mean is None:
                raise StabilityRefusal(
                    f"{where}: field 'reward_mean' is None but field 'scored' is "
                    f"{self.scored}: a mean over scored trajectories is measured and stated -- "
                    f"None is reserved for the unmeasured case where scored is 0"
                )
            if not _is_real(self.reward_mean):
                raise StabilityRefusal(
                    f"{where}: field 'reward_mean' is {_described(self.reward_mean)}: a mean is "
                    f"a real number"
                )
            if self.zero_reward_fraction is None:
                raise StabilityRefusal(
                    f"{where}: field 'zero_reward_fraction' is None but field 'scored' is "
                    f"{self.scored}: a fraction over scored trajectories is measured and "
                    f"stated -- None is reserved for the unmeasured case where scored is 0"
                )
            if not _is_real(self.zero_reward_fraction):
                raise StabilityRefusal(
                    f"{where}: field 'zero_reward_fraction' is "
                    f"{_described(self.zero_reward_fraction)}: a fraction is a real number"
                )
            if not 0.0 <= float(self.zero_reward_fraction) <= 1.0:
                raise StabilityRefusal(
                    f"{where}: field 'zero_reward_fraction' is {self.zero_reward_fraction}: a "
                    f"fraction of scored trajectories lies in [0, 1] and this one names no "
                    f"proportion at all"
                )


def rollout_stats(step: int, trajectories: Sequence[Trajectory]) -> RolloutStats:
    """Measure one rollout step from its trajectories' ``reward`` fields.

        WHAT IS CLAIMED on success: a :class:`RolloutStats` whose ``scored`` and
    ``abstained`` counts sum to ``len(trajectories)``, whose ``reward_mean`` is the
    mean over scored rewards and whose ``zero_reward_fraction`` is the fraction of
    scored rewards equal to ``0.0`` -- both ``None`` exactly when ``scored == 0``.

        WHAT IS NOT CLAIMED: anything about the trajectories' content beyond
    ``reward`` (the contracts plane already validated them), and any ordering of
    the input beyond the counts it contributes to.

        Two refusals. A non-``Trajectory`` element is refused naming its position,
    because a duck-typed object could carry a ``reward`` this plane would then
    average into a measurement. An empty sequence is refused naming both counts
    (0 scored, 0 abstained): stats over zero trajectories measure nothing, and a
    ``None`` mean over an empty step would read as an unmeasured step rather than
    a step that never happened.
    """
    if not _is_int(step):
        raise StabilityRefusal(
            f"rollout_stats(): field 'step' is {_described(step)}: a step is a real int "
            f"(type(x) is int) -- 1 is not True"
        )
    items = tuple(trajectories)
    if not items:
        raise StabilityRefusal(
            "rollout_stats(): trajectories is empty: scored is 0 and abstained is 0, and "
            "stats over zero trajectories measure nothing -- an empty step is refused rather "
            "than reported as an unmeasured mean"
        )
    scored = 0
    abstained = 0
    reward_sum = 0.0
    zero_count = 0
    for position, trajectory in enumerate(items):
        if not isinstance(trajectory, Trajectory):
            raise StabilityRefusal(
                f"rollout_stats(): element {position} is {_described(trajectory)}, not a Trajectory"
            )
        reward = trajectory.reward
        if reward is None:
            abstained += 1
            continue
        scored += 1
        reward_sum += float(reward)
        if reward == 0.0:
            zero_count += 1
    if scored == 0:
        return RolloutStats(
            step=step,
            scored=scored,
            abstained=abstained,
            reward_mean=None,
            zero_reward_fraction=None,
        )
    return RolloutStats(
        step=step,
        scored=scored,
        abstained=abstained,
        reward_mean=reward_sum / scored,
        zero_reward_fraction=zero_count / scored,
    )


@dataclass(frozen=True)
class GuardConfig:
    """How the guard reads a rollout's reward history.

    ``ema_alpha`` in ``(0, 1]`` weights the newest measured mean; ``drop_from_peak``
    in ``(0, 1)`` is the fraction of the peak the EMA must fall before an alarm;
    ``min_steps`` is how many MEASURED steps must precede any alarm (an unmeasured
    step does not count); ``best_k`` is how many steps to track as the best by EMA
    (``0`` disables tracking); ``action`` is what the HOST does on an alarm --
    ``"warn"`` reports and continues, ``"stop"`` raises :class:`CollapseDetected`
    after publishing.
    """

    ema_alpha: float = 0.3
    drop_from_peak: float = 0.5
    min_steps: int = 10
    best_k: int = 0
    action: str = "warn"

    def __post_init__(self) -> None:
        where = "guard config"
        if not _is_real(self.ema_alpha):
            raise StabilityRefusal(
                f"{where}: field 'ema_alpha' is {_described(self.ema_alpha)}: alpha is a real "
                f"number"
            )
        if not 0.0 < float(self.ema_alpha) <= 1.0:
            raise StabilityRefusal(
                f"{where}: field 'ema_alpha' is {self.ema_alpha}: alpha lies in (0, 1] -- 0 "
                f"would freeze the EMA at its first value and a value above 1 would weight the "
                f"newest mean past the whole history"
            )
        if not _is_real(self.drop_from_peak):
            raise StabilityRefusal(
                f"{where}: field 'drop_from_peak' is {_described(self.drop_from_peak)}: a drop "
                f"fraction is a real number"
            )
        if not 0.0 < float(self.drop_from_peak) < 1.0:
            raise StabilityRefusal(
                f"{where}: field 'drop_from_peak' is {self.drop_from_peak}: a drop fraction "
                f"lies in (0, 1) -- 0 would alarm on any non-rise and 1 would require the EMA "
                f"to reach 0 before alarming"
            )
        if not _is_int(self.min_steps):
            raise StabilityRefusal(
                f"{where}: field 'min_steps' is {_described(self.min_steps)}: a count is a real "
                f"int -- 1 is not True"
            )
        if self.min_steps < 1:
            raise StabilityRefusal(
                f"{where}: field 'min_steps' is {self.min_steps}: at least 1 measured step "
                f"must precede any alarm, and a guard that alarms on 0 steps would fire before "
                f"it measured anything"
            )
        if not _is_int(self.best_k):
            raise StabilityRefusal(
                f"{where}: field 'best_k' is {_described(self.best_k)}: a count is a real int "
                f"-- 1 is not True"
            )
        if self.best_k < 0:
            raise StabilityRefusal(
                f"{where}: field 'best_k' is {self.best_k}: best-k is >= 0, and 0 disables "
                f"best-k tracking"
            )
        if not isinstance(self.action, str):
            raise StabilityRefusal(
                f"{where}: field 'action' is {_described(self.action)}: action is a str naming "
                f"what the host does on an alarm"
            )
        if self.action not in ("warn", "stop"):
            raise StabilityRefusal(
                f"{where}: field 'action' is {self.action!r}: action is 'warn' or 'stop' -- "
                f"anything else names no host behaviour and would leave an alarm with no "
                f"declared consequence"
            )


@dataclass(frozen=True)
class GuardVerdict:
    """What the guard saw at one step, stated or refused.

    ``ema`` and ``peak`` are ``None`` until the first measured step; ``peak_step``
    is the step whose EMA set the peak and is ``None`` alongside them. ``alarm``
    is True only on a measured step where the EMA has fallen to or below
    ``peak * (1 - drop_from_peak)`` after ``min_steps`` measured steps. ``best_steps``
    is the current best-k steps by EMA, best first (ties broken by earlier step),
    and ``evicted`` names the steps that just left that set.
    """

    step: int
    ema: float | None
    peak: float | None
    peak_step: int | None
    alarm: bool
    best_steps: tuple[int, ...]
    evicted: tuple[int, ...]

    def __post_init__(self) -> None:
        where = f"guard verdict step={self.step!r}"
        if not _is_int(self.step):
            raise StabilityRefusal(
                f"{where}: field 'step' is {_described(self.step)}: a step is a real int "
                f"(type(x) is int) -- 1 is not True"
            )
        if self.step < 0:
            raise StabilityRefusal(
                f"{where}: field 'step' is {self.step}: steps are >= 0 and order the run"
            )
        if not _is_bool(self.alarm):
            raise StabilityRefusal(
                f"{where}: field 'alarm' is {_described(self.alarm)}: True or False only -- the "
                f"host's stop decision is derived from this flag alone, and a truthy non-bool "
                f"(the text 'false' is truthy) would stop a healthy run"
            )
        if (self.ema is None) != (self.peak is None):
            raise StabilityRefusal(
                f"{where}: field 'ema' is {_described(self.ema)} and field 'peak' is "
                f"{_described(self.peak)}: the two are measured together or unmeasured together "
                f"-- half a pair leaves a reader unable to say which one is missing"
            )
        if (self.ema is None) != (self.peak_step is None):
            raise StabilityRefusal(
                f"{where}: field 'ema' is {_described(self.ema)} and field 'peak_step' is "
                f"{_described(self.peak_step)}: the peak's step is stated exactly when the peak "
                f"is"
            )
        if self.ema is not None:
            if not _is_real(self.ema):
                raise StabilityRefusal(
                    f"{where}: field 'ema' is {_described(self.ema)}: an EMA is a real number"
                )
            if not _is_real(self.peak):
                raise StabilityRefusal(
                    f"{where}: field 'peak' is {_described(self.peak)}: a peak is a real number"
                )
            if not _is_int(self.peak_step):
                raise StabilityRefusal(
                    f"{where}: field 'peak_step' is {_described(self.peak_step)}: a step is a "
                    f"real int -- 1 is not True"
                )
            if self.peak_step < 0:
                raise StabilityRefusal(
                    f"{where}: field 'peak_step' is {self.peak_step}: steps are >= 0"
                )
            if float(self.peak) < float(self.ema):
                raise StabilityRefusal(
                    f"{where}: field 'peak' is {self.peak} but field 'ema' is {self.ema}: the "
                    f"peak is the maximum EMA seen and cannot sit below the current one"
                )
        if self.alarm and self.ema is None:
            raise StabilityRefusal(
                f"{where}: field 'alarm' is True but field 'ema' is None: an alarm is a "
                f"measured drop and an unmeasured step is never a drop"
            )
        for name in ("best_steps", "evicted"):
            declared: object = getattr(self, name)
            if not isinstance(declared, tuple):
                raise StabilityRefusal(
                    f"{where}: field {name!r} is {_described(declared)}: this field is a tuple "
                    f"of steps, frozen at construction"
                )
            for position, entry in enumerate(declared):
                if not _is_int(entry):
                    raise StabilityRefusal(
                        f"{where}: field {name!r} entry {position} is {_described(entry)}: a "
                        f"step is a real int -- 1 is not True"
                    )
                if entry < 0:
                    raise StabilityRefusal(
                        f"{where}: field {name!r} entry {position} is {entry}: steps are >= 0"
                    )
        if len(set(self.best_steps)) != len(self.best_steps):
            raise StabilityRefusal(
                f"{where}: field 'best_steps' carries {len(self.best_steps)} entries but only "
                f"{len(set(self.best_steps))} distinct steps: one step appears once in the "
                f"best-k set"
            )
        if len(set(self.evicted)) != len(self.evicted):
            raise StabilityRefusal(
                f"{where}: field 'evicted' carries {len(self.evicted)} entries but only "
                f"{len(set(self.evicted))} distinct steps: one step leaves the best-k set once"
            )


class StabilityGuard:
    """Tracks the EMA of measured reward means and reports a verdict per step.

    The guard is stateful and single-use per run: :meth:`update` is called once per
    rollout step in step order, and each call returns the verdict for that step.
    An unmeasured step (``reward_mean is None``) leaves the EMA unchanged and
    raises no alarm -- an unmeasured step is not a measured drop -- but still
    reports the current best-k set with no evictions.
    """

    def __init__(self, config: GuardConfig) -> None:
        if not isinstance(config, GuardConfig):
            raise StabilityRefusal(
                f"StabilityGuard(): config is {_described(config)}, not a GuardConfig: the "
                f"guard reads its thresholds from a declared config and never from defaults it "
                f"would have to invent"
            )
        self._config = config
        self.last_verdict: GuardVerdict | None = None
        self._ema: float | None = None
        self._peak: float | None = None
        self._peak_step: int | None = None
        self._measured_steps = 0
        self._best: dict[int, float] = {}

    @property
    def config(self) -> GuardConfig:
        """The declared configuration this guard reads its thresholds from."""
        return self._config

    @property
    def measured_steps(self) -> int:
        """How many steps carried a measured mean so far (unmeasured steps excluded)."""
        return self._measured_steps

    def update(self, stats: RolloutStats) -> GuardVerdict:
        """Fold one step's stats into the guard and return that step's verdict.

        ``reward_mean=None`` leaves the EMA and peak unchanged and reports
        ``alarm=False`` for this step. Otherwise the EMA is
        ``alpha * mean + (1 - alpha) * ema`` (the first measured step sets
        ``ema = mean``), the peak is the maximum EMA seen, and the alarm fires when
        ``measured_steps >= min_steps`` and ``peak > 0`` and
        ``ema <= peak * (1 - drop_from_peak)``.

        Best-k tracking keeps the ``best_k`` steps with the highest EMA, ties broken
        by earlier step; ``evicted`` names the steps that just left the set.
        """
        if not isinstance(stats, RolloutStats):
            raise StabilityRefusal(
                f"StabilityGuard.update(): stats is {_described(stats)}, not a RolloutStats: "
                f"the guard folds a declared measurement and never a value it would have to "
                f"interpret"
            )
        mean = stats.reward_mean
        if mean is not None:
            self._measured_steps += 1
            alpha = float(self._config.ema_alpha)
            if self._ema is None:
                self._ema = float(mean)
            else:
                self._ema = alpha * float(mean) + (1.0 - alpha) * self._ema
            if self._peak is None or self._ema > self._peak:
                self._peak = self._ema
                self._peak_step = stats.step

        alarm = False
        if (
            mean is not None
            and self._measured_steps >= self._config.min_steps
            and self._peak is not None
            and self._peak > 0
            and self._ema is not None
            and self._ema <= self._peak * (1.0 - float(self._config.drop_from_peak))
        ):
            alarm = True

        evicted: tuple[int, ...] = ()
        if self._config.best_k > 0 and mean is not None and self._ema is not None:
            previous = set(self._best)
            self._best[stats.step] = self._ema
            ordered = sorted(self._best.items(), key=lambda item: (-item[1], item[0]))
            kept = ordered[: self._config.best_k]
            self._best = dict(kept)
            evicted = tuple(sorted(previous - set(self._best)))
        best_steps = tuple(
            step for step, _ in sorted(self._best.items(), key=lambda item: (-item[1], item[0]))
        )
        verdict = GuardVerdict(
            step=stats.step,
            ema=self._ema,
            peak=self._peak,
            peak_step=self._peak_step,
            alarm=alarm,
            best_steps=best_steps,
            evicted=evicted,
        )
        self.last_verdict = verdict
        return verdict


def verdict_line(v: GuardVerdict) -> str:
    """Render a verdict as one log line: a fixed prefix plus sorted-key JSON.

    The prefix is what a log reader greps for; the JSON body is sorted by key so
    two lines for the same verdict are byte-identical and diff cleanly.
    """
    if not isinstance(v, GuardVerdict):
        raise StabilityRefusal(
            f"verdict_line(): v is {_described(v)}, not a GuardVerdict: the line renders a "
            f"declared verdict and never a value it would have to interpret"
        )
    payload = {
        "alarm": v.alarm,
        "best_steps": list(v.best_steps),
        "ema": v.ema,
        "evicted": list(v.evicted),
        "peak": v.peak,
        "peak_step": v.peak_step,
        "step": v.step,
    }
    return "[agentic-rl:stability] " + json.dumps(payload, sort_keys=True)
