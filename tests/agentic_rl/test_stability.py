"""Tests for part A of the BINDING stability API: stats, guard, and verdict line."""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

from foundationscale.agentic_rl import contracts
from foundationscale.agentic_rl.stability import (
    CollapseDetected,
    GuardConfig,
    GuardVerdict,
    RolloutStats,
    StabilityGuard,
    StabilityRefusal,
    rollout_stats,
    verdict_line,
)

PREFIX = "[agentic-rl:stability] "


def _trajectory(reward: float | None, *, session: str = "s0") -> contracts.Trajectory:
    """Build one minimal valid v1 trajectory carrying the given reward or abstention."""
    prompt = contracts.Turn(
        index=0,
        kind=contracts.SegmentKind.USER,
        token_ids=(1, 2),
        generated=False,
        logprobs=None,
    )
    response = contracts.Turn(
        index=1,
        kind=contracts.SegmentKind.ASSISTANT,
        token_ids=(3, 4),
        generated=True,
        logprobs=(-0.5, -0.5),
    )
    return contracts.Trajectory(
        uid="u0",
        session_id=session,
        harness="h0",
        prompt_turns=(prompt,),
        turns=(response,),
        reward=reward,
        abstention_reason=None if reward is not None else "malformed tool call",
        termination=contracts.Termination.STOP,
        policy_version_min=None,
        policy_version_max=None,
    )


def _stats(step: int, mean: float | None) -> RolloutStats:
    """Build a RolloutStats with the given step and mean, scored/abstained consistent."""
    if mean is None:
        return RolloutStats(
            step=step, scored=0, abstained=2, reward_mean=None, zero_reward_fraction=None
        )
    return RolloutStats(
        step=step, scored=2, abstained=0, reward_mean=mean, zero_reward_fraction=0.0
    )


def test_rollout_stats_means_and_fractions() -> None:
    """Scored and abstained trajectories yield the mean and zero fraction over scored."""
    trajectories = (
        _trajectory(0.0, session="a"),
        _trajectory(2.0, session="b"),
        _trajectory(None, session="c"),
    )
    stats = rollout_stats(3, trajectories)
    assert stats.step == 3
    assert stats.scored == 2
    assert stats.abstained == 1
    assert stats.reward_mean == pytest.approx(1.0)
    assert stats.zero_reward_fraction == pytest.approx(0.5)


def test_rollout_stats_none_when_nothing_scored() -> None:
    """With no scored trajectories the mean and zero fraction are None, never 0.0."""
    stats = rollout_stats(0, (_trajectory(None, session="a"), _trajectory(None, session="b")))
    assert stats.scored == 0
    assert stats.abstained == 2
    assert stats.reward_mean is None
    assert stats.zero_reward_fraction is None


def test_guard_config_refuses_alpha_zero() -> None:
    """An ema_alpha of 0.0 is refused: alpha must lie in (0, 1]."""
    with pytest.raises(ValueError):
        GuardConfig(ema_alpha=0.0)


def test_guard_config_refuses_drop_one() -> None:
    """A drop_from_peak of 1.0 is refused: drop must lie in (0, 1)."""
    with pytest.raises(ValueError):
        GuardConfig(drop_from_peak=1.0)


def test_guard_config_refuses_negative_min_steps() -> None:
    """A negative min_steps is refused: min_steps must be >= 0."""
    with pytest.raises(ValueError):
        GuardConfig(min_steps=-1)


def test_guard_config_refuses_bad_action() -> None:
    """An action outside {'warn', 'stop'} is refused."""
    with pytest.raises(ValueError):
        GuardConfig(action="halt")


def test_guard_config_refuses_bool_as_int() -> None:
    """A bool passed where an int is required is refused, never read as 1."""
    with pytest.raises(ValueError):
        GuardConfig(min_steps=True)
    with pytest.raises(ValueError):
        GuardConfig(best_k=True)


def test_ema_recurrence_numerically() -> None:
    """The EMA follows ema_t = alpha*mean + (1-alpha)*ema_{t-1} exactly."""
    guard = StabilityGuard(GuardConfig(ema_alpha=0.5, min_steps=100))
    first = guard.update(_stats(0, 1.0))
    assert first.ema == pytest.approx(1.0)
    second = guard.update(_stats(1, 0.0))
    assert second.ema == pytest.approx(0.5)
    third = guard.update(_stats(2, 0.0))
    assert third.ema == pytest.approx(0.25)


def test_peak_tracking() -> None:
    """The peak is the maximum EMA seen and peak_step names where it was set."""
    guard = StabilityGuard(GuardConfig(ema_alpha=1.0, min_steps=100))
    guard.update(_stats(0, 0.2))
    verdict = guard.update(_stats(1, 0.9))
    assert verdict.peak == pytest.approx(0.9)
    assert verdict.peak_step == 1
    later = guard.update(_stats(2, 0.3))
    assert later.peak == pytest.approx(0.9)
    assert later.peak_step == 1


def test_no_alarm_before_min_steps() -> None:
    """No alarm fires before min_steps measured means, however deep the drop."""
    guard = StabilityGuard(GuardConfig(ema_alpha=1.0, drop_from_peak=0.5, min_steps=5))
    guard.update(_stats(0, 1.0))
    for step in range(1, 4):  # measured steps 2..4: still below min_steps=5
        verdict = guard.update(_stats(step, 0.0))
        assert verdict.alarm is False
    assert guard.update(_stats(4, 0.0)).alarm is True  # the 5th measured step may alarm


def test_alarm_after_sustained_drop() -> None:
    """An alarm fires once measured steps reach min_steps and ema falls to the threshold."""
    guard = StabilityGuard(GuardConfig(ema_alpha=1.0, drop_from_peak=0.5, min_steps=3))
    guard.update(_stats(0, 1.0))
    guard.update(_stats(1, 1.0))
    verdict = guard.update(_stats(2, 0.4))
    assert verdict.alarm is True


def test_none_mean_leaves_ema_unchanged_and_never_alarms() -> None:
    """A None-mean step leaves the EMA and peak untouched and never alarms."""
    guard = StabilityGuard(GuardConfig(ema_alpha=1.0, drop_from_peak=0.5, min_steps=1))
    guard.update(_stats(0, 1.0))
    guard.update(_stats(1, 0.0))
    verdict = guard.update(_stats(2, None))
    assert verdict.ema == pytest.approx(0.0)
    assert verdict.peak == pytest.approx(1.0)
    assert verdict.alarm is False


def test_best_k_ordering_and_ties() -> None:
    """Best-k keeps the highest EMA steps best-first, ties resolved to the earlier step."""
    guard = StabilityGuard(GuardConfig(ema_alpha=1.0, min_steps=100, best_k=2))
    guard.update(_stats(0, 0.5))
    guard.update(_stats(1, 0.5))
    verdict = guard.update(_stats(2, 0.2))
    assert verdict.best_steps == (0, 1)


def test_best_k_evictions_reported() -> None:
    """A step leaving the best-k set is reported in evicted on the step that displaced it."""
    guard = StabilityGuard(GuardConfig(ema_alpha=1.0, min_steps=100, best_k=1))
    guard.update(_stats(0, 0.5))
    verdict = guard.update(_stats(1, 0.9))
    assert verdict.best_steps == (1,)
    assert verdict.evicted == (0,)


def test_collapse_shape_alarm_and_best_steps() -> None:
    """The measured collapse alarms within 6 steps and best_steps holds a step near the peak."""
    guard = StabilityGuard(GuardConfig(ema_alpha=0.3, drop_from_peak=0.5, min_steps=10, best_k=3))
    alarm_step: int | None = None
    verdicts: list[GuardVerdict] = []
    for step in range(42):
        mean = 0.16 + (0.56 - 0.16) * step / 41
        verdicts.append(guard.update(_stats(step, mean)))
    for step in range(42, 52):
        verdict = guard.update(_stats(step, 0.05))
        verdicts.append(verdict)
        if verdict.alarm and alarm_step is None:
            alarm_step = step
    assert alarm_step is not None
    assert alarm_step <= 47 + 6
    final = verdicts[-1]
    assert any(abs(step - 42) <= 2 for step in final.best_steps)


def test_verdict_line_is_prefixed_sorted_key_json() -> None:
    """verdict_line is the prefix plus JSON with sorted keys and the verdict's fields."""
    guard = StabilityGuard(GuardConfig(ema_alpha=1.0, min_steps=100, best_k=1))
    verdict = guard.update(_stats(0, 0.5))
    line = verdict_line(verdict)
    assert line.startswith(PREFIX)
    payload = json.loads(line[len(PREFIX) :])
    assert list(payload) == sorted(payload)
    assert payload["step"] == 0
    assert payload["ema"] == pytest.approx(0.5)
    assert payload["alarm"] is False
    assert payload["best_steps"] == [0]
    assert payload["evicted"] == []
    assert line == PREFIX + json.dumps(payload, sort_keys=True)


def test_collapse_detected_is_runtime_error() -> None:
    """CollapseDetected is a RuntimeError the host raises on a stop-action alarm."""
    assert issubclass(CollapseDetected, RuntimeError)


def test_stats_refuses_bool_step() -> None:
    """A bool step is refused: 1 is not True and a step is a real int."""
    with pytest.raises(StabilityRefusal, match="1 is not True"):
        RolloutStats(step=True, scored=1, abstained=0, reward_mean=0.0, zero_reward_fraction=0.0)


def test_stats_refuses_negative_step() -> None:
    """A negative step is refused: steps are >= 0 and order the run."""
    with pytest.raises(StabilityRefusal, match="steps are >= 0"):
        RolloutStats(step=-1, scored=1, abstained=0, reward_mean=0.0, zero_reward_fraction=0.0)


def test_stats_refuses_bool_scored() -> None:
    """A bool scored count is refused: a count is a real int."""
    with pytest.raises(StabilityRefusal, match="field 'scored'"):
        RolloutStats(step=0, scored=True, abstained=0, reward_mean=0.0, zero_reward_fraction=0.0)


def test_stats_refuses_bool_abstained() -> None:
    """A bool abstained count is refused: a count is a real int."""
    with pytest.raises(StabilityRefusal, match="field 'abstained'"):
        RolloutStats(step=0, scored=1, abstained=False, reward_mean=0.0, zero_reward_fraction=0.0)


def test_stats_refuses_negative_scored() -> None:
    """A negative scored count is refused: counts of scored trajectories are >= 0."""
    with pytest.raises(StabilityRefusal, match="scored trajectories is >= 0"):
        RolloutStats(step=0, scored=-1, abstained=1, reward_mean=0.0, zero_reward_fraction=0.0)


def test_stats_refuses_negative_abstained() -> None:
    """A negative abstained count is refused: counts of abstained trajectories are >= 0."""
    with pytest.raises(StabilityRefusal, match="abstained"):
        RolloutStats(step=0, scored=1, abstained=-1, reward_mean=0.0, zero_reward_fraction=0.0)


def test_stats_refuses_zero_total() -> None:
    """Stats whose counts sum to 0 are refused: an empty step measures nothing."""
    with pytest.raises(StabilityRefusal, match="sum to 0"):
        RolloutStats(step=0, scored=0, abstained=0, reward_mean=None, zero_reward_fraction=None)


def test_stats_refuses_mean_when_unscored() -> None:
    """A stated mean with scored == 0 is refused: unmeasured stays None, never 0.0."""
    with pytest.raises(StabilityRefusal, match="stays None"):
        RolloutStats(step=0, scored=0, abstained=1, reward_mean=0.0, zero_reward_fraction=None)


def test_stats_refuses_fraction_when_unscored() -> None:
    """A stated zero fraction with scored == 0 is refused: unmeasured stays None."""
    with pytest.raises(StabilityRefusal, match="zero_reward_fraction"):
        RolloutStats(step=0, scored=0, abstained=1, reward_mean=None, zero_reward_fraction=0.0)


def test_stats_refuses_none_mean_when_scored() -> None:
    """A None mean with scored > 0 is refused: a measured mean is stated."""
    with pytest.raises(StabilityRefusal, match="reward_mean' is None"):
        RolloutStats(step=0, scored=1, abstained=0, reward_mean=None, zero_reward_fraction=0.0)


def test_stats_refuses_non_real_mean() -> None:
    """A non-numeric mean is refused: a mean is a real number."""
    with pytest.raises(StabilityRefusal, match="a mean is a real number"):
        RolloutStats(step=0, scored=1, abstained=0, reward_mean="0.5", zero_reward_fraction=0.0)


def test_stats_refuses_none_fraction_when_scored() -> None:
    """A None zero fraction with scored > 0 is refused: a measured fraction is stated."""
    with pytest.raises(StabilityRefusal, match="zero_reward_fraction' is None"):
        RolloutStats(step=0, scored=1, abstained=0, reward_mean=0.5, zero_reward_fraction=None)


def test_stats_refuses_non_real_fraction() -> None:
    """A non-numeric zero fraction is refused: a fraction is a real number."""
    with pytest.raises(StabilityRefusal, match="a fraction is a real number"):
        RolloutStats(step=0, scored=1, abstained=0, reward_mean=0.5, zero_reward_fraction="0.0")


def test_stats_refuses_fraction_out_of_range() -> None:
    """A zero fraction outside [0, 1] is refused: it names no proportion at all."""
    with pytest.raises(StabilityRefusal, match="names no"):
        RolloutStats(step=0, scored=1, abstained=0, reward_mean=0.5, zero_reward_fraction=1.5)


def test_rollout_stats_refuses_bool_step() -> None:
    """rollout_stats refuses a bool step: 1 is not True."""
    with pytest.raises(StabilityRefusal, match="1 is not True"):
        rollout_stats(True, (_trajectory(0.0),))


def test_rollout_stats_refuses_empty_sequence() -> None:
    """rollout_stats refuses an empty sequence: stats over zero trajectories measure nothing."""
    with pytest.raises(StabilityRefusal, match="trajectories is empty"):
        rollout_stats(0, ())


def test_rollout_stats_refuses_non_trajectory_element() -> None:
    """rollout_stats refuses a non-Trajectory element naming its position."""
    with pytest.raises(StabilityRefusal, match="element 1"):
        rollout_stats(0, (_trajectory(0.0), object()))


def test_guard_config_refuses_non_real_alpha() -> None:
    """A non-numeric ema_alpha is refused: alpha is a real number."""
    with pytest.raises(StabilityRefusal, match="alpha is a real number"):
        GuardConfig(ema_alpha="0.3")


def test_guard_config_refuses_alpha_above_one() -> None:
    """An ema_alpha above 1.0 is refused: alpha lies in (0, 1]."""
    with pytest.raises(StabilityRefusal, match="alpha lies in"):
        GuardConfig(ema_alpha=1.5)


def test_guard_config_refuses_non_real_drop() -> None:
    """A non-numeric drop_from_peak is refused: a drop fraction is a real number."""
    with pytest.raises(StabilityRefusal, match="drop fraction is a real number"):
        GuardConfig(drop_from_peak="0.5")


def test_guard_config_refuses_drop_zero() -> None:
    """A drop_from_peak of 0.0 is refused: a drop fraction lies in (0, 1)."""
    with pytest.raises(StabilityRefusal, match="drop fraction"):
        GuardConfig(drop_from_peak=0.0)


def test_guard_config_refuses_non_int_min_steps() -> None:
    """A non-int min_steps is refused: a count is a real int."""
    with pytest.raises(StabilityRefusal, match="min_steps"):
        GuardConfig(min_steps=2.5)


def test_guard_config_refuses_zero_min_steps() -> None:
    """A min_steps of 0 is refused: at least 1 measured step must precede any alarm."""
    with pytest.raises(StabilityRefusal, match="at least 1 measured step"):
        GuardConfig(min_steps=0)


def test_guard_config_refuses_non_int_best_k() -> None:
    """A non-int best_k is refused: a count is a real int."""
    with pytest.raises(StabilityRefusal, match="best_k"):
        GuardConfig(best_k=1.5)


def test_guard_config_refuses_negative_best_k() -> None:
    """A negative best_k is refused: best-k is >= 0 and 0 disables tracking."""
    with pytest.raises(StabilityRefusal, match="best-k is >= 0"):
        GuardConfig(best_k=-1)


def test_guard_config_refuses_non_str_action() -> None:
    """A non-str action is refused: action is a str naming the host behaviour."""
    with pytest.raises(StabilityRefusal, match="action is a str"):
        GuardConfig(action=1)


def test_verdict_refuses_bool_step() -> None:
    """A bool step in a verdict is refused: 1 is not True."""
    with pytest.raises(StabilityRefusal, match="1 is not True"):
        GuardVerdict(
            step=True,
            ema=None,
            peak=None,
            peak_step=None,
            alarm=False,
            best_steps=(),
            evicted=(),
        )


def test_verdict_refuses_negative_step() -> None:
    """A negative step in a verdict is refused: steps are >= 0."""
    with pytest.raises(StabilityRefusal, match="steps are >= 0"):
        GuardVerdict(
            step=-1,
            ema=None,
            peak=None,
            peak_step=None,
            alarm=False,
            best_steps=(),
            evicted=(),
        )


def test_verdict_refuses_non_bool_alarm() -> None:
    """A non-bool alarm is refused: True or False only, never a truthy text."""
    with pytest.raises(StabilityRefusal, match="True or False only"):
        GuardVerdict(
            step=0,
            ema=None,
            peak=None,
            peak_step=None,
            alarm="false",
            best_steps=(),
            evicted=(),
        )


def test_verdict_refuses_half_ema_peak_pair() -> None:
    """An ema without a peak is refused: the two are measured or unmeasured together."""
    with pytest.raises(StabilityRefusal, match="half a pair"):
        GuardVerdict(
            step=0,
            ema=0.5,
            peak=None,
            peak_step=None,
            alarm=False,
            best_steps=(),
            evicted=(),
        )


def test_verdict_refuses_half_ema_peak_step_pair() -> None:
    """An ema without a peak_step is refused: the peak's step is stated exactly when the peak is."""
    with pytest.raises(StabilityRefusal, match="peak_step"):
        GuardVerdict(
            step=0,
            ema=0.5,
            peak=0.5,
            peak_step=None,
            alarm=False,
            best_steps=(),
            evicted=(),
        )


def test_verdict_refuses_non_real_ema() -> None:
    """A non-numeric ema is refused: an EMA is a real number."""
    with pytest.raises(StabilityRefusal, match="an EMA is a real number"):
        GuardVerdict(
            step=0,
            ema="0.5",
            peak=0.5,
            peak_step=0,
            alarm=False,
            best_steps=(),
            evicted=(),
        )


def test_verdict_refuses_non_real_peak() -> None:
    """A non-numeric peak is refused: a peak is a real number."""
    with pytest.raises(StabilityRefusal, match="a peak is a real number"):
        GuardVerdict(
            step=0,
            ema=0.5,
            peak="0.5",
            peak_step=0,
            alarm=False,
            best_steps=(),
            evicted=(),
        )


def test_verdict_refuses_bool_peak_step() -> None:
    """A bool peak_step is refused: a step is a real int."""
    with pytest.raises(StabilityRefusal, match="peak_step"):
        GuardVerdict(
            step=0,
            ema=0.5,
            peak=0.5,
            peak_step=True,
            alarm=False,
            best_steps=(),
            evicted=(),
        )


def test_verdict_refuses_negative_peak_step() -> None:
    """A negative peak_step is refused: steps are >= 0."""
    with pytest.raises(StabilityRefusal, match="steps are >= 0"):
        GuardVerdict(
            step=0,
            ema=0.5,
            peak=0.5,
            peak_step=-1,
            alarm=False,
            best_steps=(),
            evicted=(),
        )


def test_verdict_refuses_peak_below_ema() -> None:
    """A peak below the current ema is refused: the peak is the maximum EMA seen."""
    with pytest.raises(StabilityRefusal, match="cannot sit below"):
        GuardVerdict(
            step=0,
            ema=0.9,
            peak=0.5,
            peak_step=0,
            alarm=False,
            best_steps=(),
            evicted=(),
        )


def test_verdict_refuses_alarm_on_unmeasured_step() -> None:
    """An alarm with ema None is refused: an unmeasured step is never a drop."""
    with pytest.raises(StabilityRefusal, match="unmeasured step is never a drop"):
        GuardVerdict(
            step=0,
            ema=None,
            peak=None,
            peak_step=None,
            alarm=True,
            best_steps=(),
            evicted=(),
        )


def test_verdict_refuses_non_tuple_best_steps() -> None:
    """A non-tuple best_steps is refused: this field is a tuple of steps."""
    with pytest.raises(StabilityRefusal, match="this field is a tuple"):
        GuardVerdict(
            step=0,
            ema=None,
            peak=None,
            peak_step=None,
            alarm=False,
            best_steps=[0],
            evicted=(),
        )


def test_verdict_refuses_non_tuple_evicted() -> None:
    """A non-tuple evicted is refused: this field is a tuple of steps."""
    with pytest.raises(StabilityRefusal, match="this field is a tuple"):
        GuardVerdict(
            step=0,
            ema=None,
            peak=None,
            peak_step=None,
            alarm=False,
            best_steps=(),
            evicted=[0],
        )


def test_verdict_refuses_bool_best_step_entry() -> None:
    """A bool entry in best_steps is refused: a step is a real int."""
    with pytest.raises(StabilityRefusal, match="1 is not True"):
        GuardVerdict(
            step=0,
            ema=None,
            peak=None,
            peak_step=None,
            alarm=False,
            best_steps=(True,),
            evicted=(),
        )


def test_verdict_refuses_negative_evicted_entry() -> None:
    """A negative entry in evicted is refused: steps are >= 0."""
    with pytest.raises(StabilityRefusal, match="steps are >= 0"):
        GuardVerdict(
            step=0,
            ema=None,
            peak=None,
            peak_step=None,
            alarm=False,
            best_steps=(),
            evicted=(-1,),
        )


def test_verdict_refuses_duplicate_best_steps() -> None:
    """Duplicate best_steps entries are refused: one step appears once in the best-k set."""
    with pytest.raises(StabilityRefusal, match="distinct steps"):
        GuardVerdict(
            step=0,
            ema=None,
            peak=None,
            peak_step=None,
            alarm=False,
            best_steps=(1, 1),
            evicted=(),
        )


def test_verdict_refuses_duplicate_evicted() -> None:
    """Duplicate evicted entries are refused: one step leaves the best-k set once."""
    with pytest.raises(StabilityRefusal, match="distinct steps"):
        GuardVerdict(
            step=0,
            ema=None,
            peak=None,
            peak_step=None,
            alarm=False,
            best_steps=(),
            evicted=(2, 2),
        )


def test_guard_refuses_non_config() -> None:
    """StabilityGuard refuses a non-GuardConfig config: it never invents thresholds."""
    with pytest.raises(StabilityRefusal, match="not a GuardConfig"):
        StabilityGuard({"ema_alpha": 0.3})  # type: ignore[arg-type]


def test_guard_update_refuses_non_stats() -> None:
    """StabilityGuard.update refuses a non-RolloutStats value: it folds declared measurements."""
    guard = StabilityGuard(GuardConfig())
    with pytest.raises(StabilityRefusal, match="not a RolloutStats"):
        guard.update({"reward_mean": 0.5})  # type: ignore[arg-type]


def test_guard_config_property_returns_declared_config() -> None:
    """The guard's config property returns the declared configuration it reads thresholds from."""
    config = GuardConfig(ema_alpha=0.7, min_steps=3, best_k=2, action="stop")
    guard = StabilityGuard(config)
    assert guard.config is config


def test_guard_measured_steps_counts_only_measured() -> None:
    """measured_steps counts measured means only, excluding unmeasured steps."""
    guard = StabilityGuard(GuardConfig(min_steps=100))
    guard.update(_stats(0, 1.0))
    guard.update(_stats(1, None))
    guard.update(_stats(2, 0.5))
    assert guard.measured_steps == 2


def test_verdict_line_refuses_non_verdict() -> None:
    """verdict_line refuses a non-GuardVerdict value: it renders declared verdicts only."""
    with pytest.raises(StabilityRefusal, match="not a GuardVerdict"):
        verdict_line({"step": 0})  # type: ignore[arg-type]


def test_verdict_line_renders_unmeasured_verdict() -> None:
    """verdict_line renders an unmeasured verdict with null ema, peak and peak_step."""
    verdict = GuardVerdict(
        step=7,
        ema=None,
        peak=None,
        peak_step=None,
        alarm=False,
        best_steps=(),
        evicted=(),
    )
    line = verdict_line(verdict)
    payload = json.loads(line[len(PREFIX) :])
    assert payload["ema"] is None
    assert payload["peak"] is None
    assert payload["peak_step"] is None
    assert payload["step"] == 7
    assert payload["best_steps"] == []
    assert payload["evicted"] == []


def test_verdict_replace_keeps_valid_shape() -> None:
    """dataclasses.replace on a valid verdict keeps the measured pair consistent."""
    verdict = _stats(0, 0.5)
    assert replace(verdict, step=1).step == 1


class _RecordingSync:
    """Stands in for DiskWeightSync: records protect/unprotect calls."""

    def __init__(self) -> None:
        self.protected: list[int] = []
        self.unprotected: list[int] = []

    def protect(self, step: int) -> None:
        self.protected.append(step)

    def unprotect(self, step: int) -> None:
        self.unprotected.append(step)


def _apply(guard: StabilityGuard, step: int) -> _RecordingSync:
    from types import SimpleNamespace

    from foundationscale.agentic_rl.rollout_host import RolloutHost

    sync = _RecordingSync()
    RolloutHost._apply_guard(SimpleNamespace(guard=guard, weight_sync=sync), step)  # type: ignore[arg-type]
    return sync


def test_host_protects_the_checkpoint_that_produced_a_best_rollout() -> None:
    """A best rollout at step r was sampled with checkpoint r-1, so r-1 is protected."""
    guard = StabilityGuard(GuardConfig(ema_alpha=1.0, best_k=1, min_steps=1))
    guard.update(_stats(0, 0.1))
    guard.update(_stats(1, 0.2))
    guard.update(_stats(2, 0.9))
    sync = _apply(guard, 2)
    assert sync.protected == [1]


def test_host_stops_on_a_collapse_when_action_is_stop() -> None:
    """With action="stop", an alarm raises CollapseDetected after the publish."""
    guard = StabilityGuard(
        GuardConfig(ema_alpha=1.0, drop_from_peak=0.5, min_steps=2, action="stop")
    )
    guard.update(_stats(0, 0.6))
    guard.update(_stats(1, 0.6))
    guard.update(_stats(2, 0.05))
    with pytest.raises(CollapseDetected, match="fell to"):
        _apply(guard, 2)


def test_host_only_warns_on_a_collapse_by_default() -> None:
    """The default action is "warn": an alarm never stops the run."""
    guard = StabilityGuard(GuardConfig(ema_alpha=1.0, drop_from_peak=0.5, min_steps=2))
    guard.update(_stats(0, 0.6))
    guard.update(_stats(1, 0.6))
    assert guard.update(_stats(2, 0.05)).alarm is True
    _apply(guard, 2)  # does not raise


def test_host_guard_is_a_no_op_before_any_verdict_or_without_a_sync() -> None:
    """No verdict yet, or no weight sync to protect against: nothing happens, nothing raises."""
    from types import SimpleNamespace

    from foundationscale.agentic_rl.rollout_host import RolloutHost

    guard = StabilityGuard(GuardConfig(best_k=1))
    assert _apply(guard, 0).protected == []
    guard.update(_stats(1, 0.5))
    RolloutHost._apply_guard(SimpleNamespace(guard=guard, weight_sync=None), 1)  # type: ignore[arg-type]


def test_host_unprotects_a_checkpoint_when_its_rollout_leaves_the_best_k() -> None:
    """An evicted best rollout r releases checkpoint r-1 back to normal pruning."""
    guard = StabilityGuard(GuardConfig(ema_alpha=1.0, best_k=1, min_steps=1))
    guard.update(_stats(1, 0.2))
    guard.update(_stats(2, 0.9))
    sync = _apply(guard, 2)
    assert sync.unprotected == [0]
