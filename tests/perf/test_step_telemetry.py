"""Pinned arms for ``foundationscale.perf.telemetry.StepTelemetry.summary()``.

``summary()`` is a branch tree in which every branch returns a DIFFERENT
reason string, and each reason is the behaviour: it names the input that was
missing. Every test here drives the real callback hooks --
on_train_begin, on_step_begin, on_pre_optimizer_step, on_optimizer_step,
on_step_end -- with ``types.SimpleNamespace`` stand-ins for
args/state/control, because the hooks read attributes and nothing more; no
Trainer is constructed. Wall-clock time is the one doubled input:
``time.perf_counter`` is replaced by an advance-only clock so each step
duration, gap and optimizer bracket is an exact, hand-checkable number, and
every expected value below is computed by hand at the asserting line, not
re-derived through the module's own arithmetic.

Since finding #518, every tuple summary() returns is already manifest-shaped:
(number, "measured"|"derived") for values, (reason_string, "unmeasured") for
gaps -- the reason living in the VALUE slot, never None, because a None value
is the one shape the provenance manifest refuses.

No skips: torch is installed but this box has no CUDA, and the no-CUDA
memory branch IS the behaviour under test -- asserted as a real outcome,
never patched away. The measured memory branches are reached by doubling
``_read_cuda_memory_bytes`` itself, the seam the module defines for exactly
this, and each doubled return names the branch it is meant to reach.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from types import SimpleNamespace

import pytest

from foundationscale.perf import telemetry
from foundationscale.perf.telemetry import (
    PERF_TELEMETRY_UNITS,
    DevicePeak,
    FlopsModel,
    StepTelemetry,
    _entry,
)
from foundationscale.provenance.manifest import TelemetryEntry

# Batch geometry present in almost every run driven here: 8 * 2 = 16 samples
# per optimizer step with a world of one, chosen so samples/s divides evenly
# by the step times used below.
_ARGS = SimpleNamespace(per_device_train_batch_size=8, gradient_accumulation_steps=2)
_STATE = SimpleNamespace()
_CONTROL = SimpleNamespace()

# Sentinel distinguishing "state carries no token counter at all" from
# "the counter was fed" -- the two are different facts to the callback.
_ABSENT = object()

# flops_per_token = 6 * 8000 + 12 * 2 * 16 * 32 = 48_000 + 12_288 = 60_288,
# computed once here and re-stated at every assertion that consumes it so a
# reader never has to trust this comment over the arithmetic.
_FLOPS = FlopsModel(
    parameters=10_000,
    non_embedding_parameters=8_000,
    layers=2,
    hidden_size=16,
    sequence_length=32,
)
_PEAK = DevicePeak(
    name="unit-test-peak",
    bf16_dense_tflops=100.0,
    source="vendor datasheet, dense bf16 row",
)


class _Clock:
    """An advance-only stand-in for ``time.perf_counter``.

    pytest itself calls perf_counter while the patch is live (fixture
    bookkeeping, duration reporting), so the double answers EVERY call with
    the current value and only moves when a test advances it. A scripted
    next-value sequence would be consumed by callers the test does not
    control and would silently retime the hooks it claims to measure.
    """

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        if seconds < 0:
            raise ValueError(f"the fake clock is monotonic; cannot advance {seconds!r}")
        self.now += seconds


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> _Clock:
    """Replace ``time.perf_counter`` with an advance-only clock for one test."""
    fake = _Clock()
    monkeypatch.setattr(time, "perf_counter", fake)
    return fake


def _run_step(
    callback: StepTelemetry,
    clock: _Clock,
    *,
    step_s: float,
    gap_s: float = 0.0,
    optimizer_s: float | None = None,
    tokens: object = _ABSENT,
) -> None:
    """Drive one complete hook cycle for a single training step.

    ``gap_s`` elapses before on_step_begin (the between-step interval for
    every step after the first), ``step_s`` inside the step, and
    ``optimizer_s`` inside the pre-optimizer -> optimizer bracket when
    given. The clock only moves where this helper advances it, so every
    interval lands in exactly one telemetry bucket and any interval the
    caller omits is genuinely absent rather than zero.
    """
    if optimizer_s is not None and optimizer_s > step_s:
        raise ValueError("the optimizer bracket cannot be longer than the step")
    clock.advance(gap_s)
    callback.on_step_begin(_ARGS, _STATE, _CONTROL)
    if optimizer_s is None:
        clock.advance(step_s)
    else:
        clock.advance(step_s - optimizer_s)
        callback.on_pre_optimizer_step(_ARGS, _STATE, _CONTROL)
        clock.advance(optimizer_s)
        callback.on_optimizer_step(_ARGS, _STATE, _CONTROL)
    state = _STATE if tokens is _ABSENT else SimpleNamespace(num_input_tokens_seen=tokens)
    callback.on_step_end(_ARGS, state, _CONTROL)


def _drive_two_reading_run(callback: StepTelemetry, clock: _Clock) -> None:
    """A warmup-free two-step run: 500 tokens at t=2.0, 2000 tokens at t=8.0.

    The window between the two token readings is 6.0 s carrying 1500
    tokens, so the steady-state rate this run must yield is 250.0 tokens/s;
    the four inputs are chosen so that any other pairing of them gives a
    different answer.
    """
    callback.on_train_begin(_ARGS, _STATE, _CONTROL)
    _run_step(callback, clock, step_s=2.0, tokens=500)
    _run_step(callback, clock, step_s=3.0, gap_s=3.0, tokens=2000)


def test_zero_steps_every_derived_entry_is_unmeasured_with_a_named_reason() -> None:
    """An untouched callback reports every derivable entry with source
    "unmeasured" and the UNMEASURED reason string sitting in the VALUE
    slot -- never a None value, which is precisely the shape finding #518
    showed the provenance manifest refuses -- and the only numbers it
    publishes are the four identity counters, none of them the float 0.0.

    The honest form is a loop over the WHOLE summary dict: a hand-listed
    subset of keys would silently stop covering any key summary() grows
    later, and an entry that decayed to 0.0, or back to the old (None,
    reason) layout this module used to emit, is exactly the defect this
    test exists to catch.
    """
    summary = StepTelemetry().summary()
    numbered_keys: set[str] = set()
    for key, entry in summary.items():
        assert isinstance(entry, tuple) and len(entry) == 2, key
        value, source = entry
        assert source in ("measured", "derived", "unmeasured"), key
        # The manifest refuses a None value under ANY source -- the exact
        # shape finding #518 shipped -- so None is asserted away here
        # regardless of which provenance label the entry carries.
        assert value is not None, key
        if source == "unmeasured":
            assert isinstance(value, str) and value, key
            assert value.startswith("UNMEASURED"), key
            # A reason names its input; a bare token does not.
            assert len(value) > 40, key
        else:
            assert not (isinstance(value, float) and value == 0.0), key
            numbered_keys.add(key)
    assert numbered_keys == {
        "perf_steps_observed",
        "perf_steps_warmup",
        "perf_steps_steady",
        "perf_world_size",
    }
    assert summary["perf_steps_observed"] == (0, "measured")
    assert summary["perf_steps_warmup"] == (0, "measured")
    assert summary["perf_steps_steady"] == (0, "measured")


def test_zero_steps_every_summary_entry_constructs_a_valid_telemetry_entry() -> None:
    """END-TO-END PIN for finding #518: every entry of a zero-step
    summary must be accepted by the manifest constructor the perf plane
    feeds, passed key, value and source exactly as summary() emitted
    them.

    The old shape -- (None, reason) for an unmeasured metric -- made this
    constructor raise ValueError at the end of a finished run, because a
    None value violates the TelemetryEntry contract under every admissible
    source and the old reason string is not a source at all. Against that
    shape this test fails on the first unmeasured entry it feeds, so its
    pass here is the statement that the perf plane's output is accepted by
    the manifest that stores it.
    """
    summary = StepTelemetry().summary()
    for key, (value, source) in summary.items():
        TelemetryEntry(key=key, value=value, source=source)


def test_a_short_mixed_run_feeds_measured_and_unmeasured_entries_to_the_manifest(
    clock: _Clock,
) -> None:
    """The same end-to-end pin on a run that is genuinely MIXED: one
    driven steady step leaves the token metrics and (on this no-CUDA box)
    the memory metrics unmeasured while the step counters are measured,
    and both classes are asserted present BEFORE any constructor runs --
    so the test cannot pass by every entry happening to be measured, and
    remains a real statement that the manifest accepts the perf plane's
    full vocabulary of sources.
    """
    callback = StepTelemetry(warmup_steps=0)
    callback.on_train_begin(_ARGS, _STATE, _CONTROL)
    _run_step(callback, clock, step_s=1.0)
    summary = callback.summary()
    sources = {source for _, source in summary.values()}
    assert "measured" in sources
    assert "unmeasured" in sources
    for key, (value, source) in summary.items():
        TelemetryEntry(key=key, value=value, source=source)


def test_empty_steady_window_reason_names_warmup_and_the_counts() -> None:
    """With only warmup steps observed, the step-time reason states that
    the steady-state window is empty and says how many steps the warmup
    bucket is holding -- carried in the value slot under the "unmeasured"
    source, as every reason now is.

    warmup_steps=5 swallowing 3 driven steps and warmup_steps=3 swallowing
    the same 3 steps produce DIFFERENT reason strings, because the reason
    carries the declaration and the counts that explain the gap.
    """
    clock_less = _Clock()
    callback = StepTelemetry(warmup_steps=5)
    callback.on_train_begin(_ARGS, _STATE, _CONTROL)
    for _ in range(3):
        _run_step(callback, clock_less, step_s=1.0)
    reason, source = callback.summary()["perf_step_time_mean_s"]
    assert source == "unmeasured"
    assert "steady-state window is empty" in reason
    assert "warmup_steps=5" in reason
    assert "3 step(s)" in reason


def test_warmup_steps_are_timed_counted_but_excluded_from_the_mean(clock: _Clock) -> None:
    """Warmup steps land in their own bucket: they are timed and counted,
    and the steady-state mean is over the steady bucket alone.

    The durations [10, 20 | 1, 1] with warmup_steps=2 give a steady mean of
    1.0 where including the warmup steps would give 8.0 -- the sequence is
    constructed so the wrong answer and the right answer are far apart.
    """
    callback = StepTelemetry(warmup_steps=2)
    callback.on_train_begin(_ARGS, _STATE, _CONTROL)
    for duration in (10.0, 20.0, 1.0, 1.0):
        _run_step(callback, clock, step_s=duration)
    summary = callback.summary()
    assert summary["perf_steps_observed"] == (4, "measured")
    assert summary["perf_steps_warmup"] == (2, "measured")
    assert summary["perf_steps_steady"] == (2, "measured")
    assert summary["perf_step_time_mean_s"][0] == pytest.approx(1.0)
    assert summary["perf_step_time_mean_s"][0] != pytest.approx(8.0)
    assert summary["perf_step_time_p50_s"][0] == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("warmup", "steps"),
    [(0, 4), (3, 5), (10, 4), (1, 1)],
    ids=["no-warmup", "mixed", "all-warmup", "single-step"],
)
def test_steps_observed_is_always_warmup_plus_steady(
    warmup: int, steps: int, clock: _Clock
) -> None:
    """perf_steps_observed is exactly the warmup count plus the steady
    count, in every partition of the run -- no step is dropped and none is
    double-counted at the bucket boundary.
    """
    callback = StepTelemetry(warmup_steps=warmup)
    callback.on_train_begin(_ARGS, _STATE, _CONTROL)
    for _ in range(steps):
        _run_step(callback, clock, step_s=1.0)
    summary = callback.summary()
    expected_warmup = min(steps, warmup)
    expected_steady = max(steps - warmup, 0)
    assert summary["perf_steps_warmup"] == (expected_warmup, "measured")
    assert summary["perf_steps_steady"] == (expected_steady, "measured")
    assert summary["perf_steps_observed"] == (
        expected_warmup + expected_steady,
        "measured",
    )


def test_percentiles_use_linear_interpolation_not_nearest_rank(clock: _Clock) -> None:
    """The p50/p90 convention is linear interpolation on rank (n-1)*q.

    On the sorted even-length list [1, 2, 3, 10]: rank50 = 1.5 gives
    2 + (3-2)*0.5 = 2.5 where nearest-rank gives 2 (or 3), and rank90 = 2.7
    gives 3 + (10-3)*0.7 = 7.9 where nearest-rank gives 10. The hand
    values are computed in this comment so the convention, not the code,
    is the source of truth.
    """
    callback = StepTelemetry(warmup_steps=0)
    callback.on_train_begin(_ARGS, _STATE, _CONTROL)
    for duration in (10.0, 1.0, 3.0, 2.0):
        _run_step(callback, clock, step_s=duration)
    summary = callback.summary()
    assert summary["perf_step_time_p50_s"] == (pytest.approx(2.5), "derived")
    assert summary["perf_step_time_p90_s"] == (pytest.approx(7.9), "derived")
    assert summary["perf_step_time_p50_s"][0] != pytest.approx(2.0)
    assert summary["perf_step_time_p90_s"][0] != pytest.approx(10.0)


def test_single_steady_step_percentile_is_the_step_itself(clock: _Clock) -> None:
    """A one-element steady window has p50 == p90 == that element, and the
    percentiles are still DERIVED values rather than unmeasured gaps.
    """
    callback = StepTelemetry(warmup_steps=0)
    callback.on_train_begin(_ARGS, _STATE, _CONTROL)
    _run_step(callback, clock, step_s=4.0)
    summary = callback.summary()
    assert summary["perf_step_time_mean_s"] == (pytest.approx(4.0), "derived")
    assert summary["perf_step_time_p50_s"] == (4.0, "derived")
    assert summary["perf_step_time_p90_s"] == (4.0, "derived")


def test_stall_fraction_is_between_time_over_between_plus_step_time(clock: _Clock) -> None:
    """The stall fraction is summed between-step time divided by summed
    (between + step) time over the steady window.

    Driven sequence: step 1.0 s; gap 3.0 s then step 2.0 s; gap 1.0 s then
    step 4.0 s. between_sum = 4.0, step_sum = 7.0, so the fraction is
    4/11 and the between-step mean is (3+1)/2 = 2.0 -- asserted as the
    arithmetic, not as the code re-typed.
    """
    callback = StepTelemetry(warmup_steps=0)
    callback.on_train_begin(_ARGS, _STATE, _CONTROL)
    _run_step(callback, clock, step_s=1.0)
    _run_step(callback, clock, step_s=2.0, gap_s=3.0)
    _run_step(callback, clock, step_s=4.0, gap_s=1.0)
    summary = callback.summary()
    assert summary["perf_between_steps_mean_s"] == (pytest.approx(2.0), "derived")
    assert summary["perf_step_time_mean_s"] == (pytest.approx(7.0 / 3.0), "derived")
    assert summary["perf_dataloader_stall_fraction"] == (
        pytest.approx(4.0 / 11.0),
        "derived",
    )


def test_first_step_fabricates_no_zero_gap(clock: _Clock) -> None:
    """The first step of a run has no predecessor, and no 0.0 "no stall"
    observation is invented for it.

    Gaps of 5.0 s and 1.0 s before the second and third steps give a
    between-step mean of 3.0 over TWO samples; a fabricated first-step zero
    would dilute it to 2.0, which is asserted away explicitly.
    """
    callback = StepTelemetry(warmup_steps=0)
    callback.on_train_begin(_ARGS, _STATE, _CONTROL)
    _run_step(callback, clock, step_s=1.0)
    _run_step(callback, clock, step_s=1.0, gap_s=5.0)
    _run_step(callback, clock, step_s=1.0, gap_s=1.0)
    summary = callback.summary()
    assert summary["perf_between_steps_mean_s"][0] == pytest.approx(3.0)
    assert summary["perf_between_steps_mean_s"][0] != pytest.approx(2.0)


def test_between_step_negative_controls_name_the_missing_gap(clock: _Clock) -> None:
    """A one-step steady window leaves between_steps and the stall
    fraction unmeasured with the no-gap reason, an empty steady window
    leaves them with the empty-window reason, and the two reasons are
    DIFFERENT strings -- collapsing them would hide whether steps or gaps
    were missing. Both reasons travel in the value slot under the
    "unmeasured" source.
    """
    one_step = StepTelemetry(warmup_steps=0)
    one_step.on_train_begin(_ARGS, _STATE, _CONTROL)
    _run_step(one_step, clock, step_s=2.0)
    stalled = one_step.summary()
    between_reason, between_source = stalled["perf_between_steps_mean_s"]
    stall_reason, stall_source = stalled["perf_dataloader_stall_fraction"]
    assert between_source == "unmeasured" and stall_source == "unmeasured"
    assert "no between-step gap was recorded" in between_reason
    assert stall_reason == between_reason

    warmup_only = StepTelemetry(warmup_steps=3)
    warmup_only.on_train_begin(_ARGS, _STATE, _CONTROL)
    _run_step(warmup_only, clock, step_s=1.0)
    _run_step(warmup_only, clock, step_s=1.0)
    empty_reason = warmup_only.summary()["perf_between_steps_mean_s"][0]
    assert "steady-state window is empty" in empty_reason
    assert empty_reason != between_reason


def test_optimizer_sync_never_fired_names_the_missing_hook(clock: _Clock) -> None:
    """When on_pre_optimizer_step never fires, the optimizer metric is
    unmeasured, its reason string carried in the value slot and naming
    the hook that was absent -- never a stale or partial bracket, and
    never 0.0.
    """
    callback = StepTelemetry(warmup_steps=0)
    callback.on_train_begin(_ARGS, _STATE, _CONTROL)
    _run_step(callback, clock, step_s=1.0)
    _run_step(callback, clock, step_s=1.0)
    reason, source = callback.summary()["perf_optimizer_sync_mean_s"]
    assert source == "unmeasured"
    assert "on_pre_optimizer_step never fired" in reason
    assert reason.startswith("UNMEASURED")


def test_optimizer_brackets_in_warmup_only_report_the_empty_window(clock: _Clock) -> None:
    """Optimizer brackets observed entirely inside warmup are recorded but
    excluded: the steady metric falls to the empty-window reason, which is
    a DIFFERENT string from the never-fired reason -- the distinction is
    between "the hook does not exist" and "the hook fired before the
    window opened". Both come back as (reason, "unmeasured").
    """
    warmed = StepTelemetry(warmup_steps=3)
    warmed.on_train_begin(_ARGS, _STATE, _CONTROL)
    _run_step(warmed, clock, step_s=1.0, optimizer_s=0.5)
    _run_step(warmed, clock, step_s=1.0, optimizer_s=0.5)
    warmed_reason, warmed_source = warmed.summary()["perf_optimizer_sync_mean_s"]
    assert warmed_source == "unmeasured"
    assert "steady-state window is empty" in warmed_reason

    never = StepTelemetry(warmup_steps=0)
    never.on_train_begin(_ARGS, _STATE, _CONTROL)
    _run_step(never, clock, step_s=1.0)
    never_reason = never.summary()["perf_optimizer_sync_mean_s"][0]
    assert "on_pre_optimizer_step never fired" in never_reason
    assert warmed_reason != never_reason


def test_optimizer_sync_mean_is_over_steady_brackets(clock: _Clock) -> None:
    """The optimizer metric is the mean of the steady-state pre-optimizer
    -> optimizer brackets: two 0.25 s brackets inside 1.0 s steps give a
    mean of exactly 0.25 s, labelled derived.
    """
    callback = StepTelemetry(warmup_steps=0)
    callback.on_train_begin(_ARGS, _STATE, _CONTROL)
    _run_step(callback, clock, step_s=1.0, optimizer_s=0.25)
    _run_step(callback, clock, step_s=1.0, optimizer_s=0.25)
    assert callback.summary()["perf_optimizer_sync_mean_s"] == (
        pytest.approx(0.25),
        "derived",
    )


def test_tokens_total_unmeasured_without_a_counter_and_the_reason_names_it(
    clock: _Clock,
) -> None:
    """A state that never carries num_input_tokens_seen leaves
    tokens_total AND tokens_per_second unmeasured, and both reasons --
    carried in the value slot, as every reason now is -- name the
    attribute and the TrainingArguments flag that would have fed it.
    """
    callback = StepTelemetry(warmup_steps=0)
    callback.on_train_begin(_ARGS, _STATE, _CONTROL)
    _run_step(callback, clock, step_s=1.0)
    _run_step(callback, clock, step_s=1.0)
    summary = callback.summary()
    total_reason, total_source = summary["perf_tokens_total"]
    rate_reason, rate_source = summary["perf_tokens_per_second"]
    assert total_source == "unmeasured" and rate_source == "unmeasured"
    assert "num_input_tokens_seen" in total_reason
    assert "include_num_input_tokens_seen" in total_reason
    assert rate_reason == total_reason


def test_a_zero_token_counter_is_an_unread_counter(clock: _Clock) -> None:
    """num_input_tokens_seen == 0 at step end is treated as the signature
    of a counter the trainer never fed, not as a measured zero: the entry
    is unmeasured with the same named-counter reason as the absent case,
    the reason sitting in the value slot, and no (0, "measured") is
    minted.
    """
    callback = StepTelemetry(warmup_steps=0)
    callback.on_train_begin(_ARGS, _STATE, _CONTROL)
    _run_step(callback, clock, step_s=1.0, tokens=0)
    _run_step(callback, clock, step_s=1.0, tokens=0)
    reason, source = callback.summary()["perf_tokens_total"]
    assert source == "unmeasured"
    assert "num_input_tokens_seen" in reason
    assert "include_num_input_tokens_seen" in reason


def test_one_steady_token_reading_is_a_level_not_a_rate(clock: _Clock) -> None:
    """A single steady-state token reading makes tokens_total MEASURED but
    tokens_per_second UNMEASURED with the level-not-a-rate reason, and the
    same reason flows down the chain to model TFLOP/s when a FlopsModel is
    declared -- the rate's gap, not the level's, is what propagates, each
    hop arriving as (reason, "unmeasured").
    """
    callback = StepTelemetry(warmup_steps=0, flops_model=_FLOPS)
    callback.on_train_begin(_ARGS, _STATE, _CONTROL)
    _run_step(callback, clock, step_s=2.0, tokens=500)
    summary = callback.summary()
    assert summary["perf_tokens_total"] == (500, "measured")
    rate_reason, rate_source = summary["perf_tokens_per_second"]
    assert rate_source == "unmeasured"
    assert "one reading is a level, not a rate" in rate_reason
    device_reason, device_source = summary["perf_tokens_per_second_per_device"]
    assert device_source == "unmeasured" and device_reason == rate_reason
    tflops_reason, tflops_source = summary["perf_model_tflops_per_second"]
    assert tflops_source == "unmeasured" and "level, not a rate" in tflops_reason


def test_two_token_readings_give_the_difference_over_the_elapsed_window(
    clock: _Clock, monkeypatch: pytest.MonkeyPatch
) -> None:
    """tokens/s is (last - first steady reading) / wall time between the
    readings: (2000 - 500) over the 6.0 s window from t=2.0 to t=8.0 is
    exactly 250.0 tokens/s, and per-device divides it by the world size --
    here a doubled world of 4 gives 62.5.
    """
    callback = StepTelemetry(warmup_steps=0)
    _drive_two_reading_run(callback, clock)
    monkeypatch.setattr(telemetry, "_world_size", lambda: 4)
    summary = callback.summary()
    assert summary["perf_tokens_total"] == (2000, "measured")
    assert summary["perf_tokens_per_second"] == (pytest.approx(250.0), "derived")
    assert summary["perf_tokens_per_second_per_device"] == (
        pytest.approx(62.5),
        "derived",
    )
    assert summary["perf_world_size"] == (4, "measured")


def test_warmup_only_token_readings_leave_the_steady_rate_unmeasured(clock: _Clock) -> None:
    """Tokens counted only during warmup still make tokens_total measured
    (the last reading is real), but the steady-state RATE is unmeasured
    with a reason -- in the value slot -- saying no steady-state step
    carried a count: a third, distinct token gap, different from both the
    unread counter and the one-reading cases.
    """
    callback = StepTelemetry(warmup_steps=1)
    callback.on_train_begin(_ARGS, _STATE, _CONTROL)
    _run_step(callback, clock, step_s=1.0, tokens=100)
    _run_step(callback, clock, step_s=1.0)
    _run_step(callback, clock, step_s=1.0)
    summary = callback.summary()
    assert summary["perf_tokens_total"] == (100, "measured")
    rate_reason, rate_source = summary["perf_tokens_per_second"]
    assert rate_source == "unmeasured"
    assert "no steady-state step carried a token count" in rate_reason
    assert "one reading is a level, not a rate" not in rate_reason


def test_samples_per_second_comes_from_the_declared_batch_geometry(
    clock: _Clock, monkeypatch: pytest.MonkeyPatch
) -> None:
    """samples/s is (per_device_batch * grad_accum * world) / mean steady
    step time: 8 * 2 * 1 over a 2.0 s mean step is 8.0 samples/s with a
    world of one, and 32.0 with a doubled world of 4.
    """
    callback = StepTelemetry(warmup_steps=0)
    callback.on_train_begin(_ARGS, _STATE, _CONTROL)
    _run_step(callback, clock, step_s=2.0)
    _run_step(callback, clock, step_s=2.0)
    assert callback.summary()["perf_samples_per_second"] == (
        pytest.approx(8.0),
        "derived",
    )
    monkeypatch.setattr(telemetry, "_world_size", lambda: 4)
    assert callback.summary()["perf_samples_per_second"] == (
        pytest.approx(32.0),
        "derived",
    )


@pytest.mark.parametrize(
    ("args", "needle"),
    [
        (None, "on_train_begin never ran"),
        (SimpleNamespace(gradient_accumulation_steps=2), "per_device_train_batch_size"),
        (
            SimpleNamespace(per_device_train_batch_size=True, gradient_accumulation_steps=2),
            "per_device_train_batch_size",
        ),
        (
            SimpleNamespace(per_device_train_batch_size=8, gradient_accumulation_steps=True),
            "gradient_accumulation_steps",
        ),
    ],
    ids=["begin-never-ran", "per-device-missing", "per-device-bool", "accum-bool"],
)
def test_samples_per_second_negative_controls_name_the_missing_field(
    args: SimpleNamespace | None, needle: str, clock: _Clock
) -> None:
    """Without a readable positive-integer batch geometry, samples/s is
    unmeasured, its reason (in the value slot) naming the exact field
    that failed -- and a bool field is a flag, not a count, so True fails
    the same guard as an absent field.
    """
    callback = StepTelemetry(warmup_steps=0)
    if args is not None:
        callback.on_train_begin(args, _STATE, _CONTROL)
    _run_step(callback, clock, step_s=1.0)
    _run_step(callback, clock, step_s=1.0)
    reason, source = callback.summary()["perf_samples_per_second"]
    assert source == "unmeasured"
    assert reason.startswith("UNMEASURED")
    assert needle in reason


def test_mfu_unmeasured_without_a_declared_peak_even_when_tflops_is_known(
    clock: _Clock,
) -> None:
    """With a FlopsModel and a token counter, model TFLOP/s is derivable
    and IS derived -- but MFU and perf_device_peak_tflops stay unmeasured,
    both carrying the single no-peak reason in the value slot, because
    there is no default peak to fall back to; the reason names device_peak
    AND the env declaration.
    """
    callback = StepTelemetry(warmup_steps=0, flops_model=_FLOPS)
    _drive_two_reading_run(callback, clock)
    summary = callback.summary()
    assert summary["perf_model_tflops_per_second"][1] == "derived"
    mfu_reason, mfu_source = summary["perf_mfu"]
    peak_reason, peak_source = summary["perf_device_peak_tflops"]
    assert mfu_source == "unmeasured" and peak_source == "unmeasured"
    assert "device peak" in mfu_reason
    assert "FS_DEVICE_PEAK_TFLOPS" in mfu_reason
    assert peak_reason == mfu_reason


def test_mfu_unmeasured_without_a_flops_model_even_when_a_peak_is_declared(
    clock: _Clock,
) -> None:
    """The dual gap: a declared peak is reported (measured), the token
    rate is derived, but TFLOP/s and MFU are unmeasured with the
    no-FlopsModel reason in the value slot -- and that reason is a
    DIFFERENT string from the no-peak reason, so an operator can tell
    which of the two inputs is missing.
    """
    with_peak_no_flops = StepTelemetry(warmup_steps=0, device_peak=_PEAK)
    _drive_two_reading_run(with_peak_no_flops, clock)
    summary = with_peak_no_flops.summary()
    assert summary["perf_device_peak_tflops"] == (100.0, "measured")
    assert summary["perf_tokens_per_second"][1] == "derived"
    for key in ("perf_model_tflops_per_second", "perf_mfu"):
        reason, source = summary[key]
        assert source == "unmeasured", key
        assert "FlopsModel" in reason, key

    no_peak_with_flops = StepTelemetry(warmup_steps=0, flops_model=_FLOPS)
    _drive_two_reading_run(no_peak_with_flops, clock)
    no_peak_reason = no_peak_with_flops.summary()["perf_mfu"][0]
    no_flops_reason = summary["perf_mfu"][0]
    assert "device peak" in no_peak_reason
    assert no_flops_reason != no_peak_reason


def test_mfu_is_derived_from_a_declared_peak_and_the_stated_flops_formula(
    clock: _Clock, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With both declarations in place: flops_per_token is
    6*8000 + 12*2*16*32 = 60_288, the 250.0 tokens/s window gives
    250 * 60_288 / 1e12 = 1.5072e-05 model TFLOP/s, and MFU divides the
    per-device figure by the declared 100.0 TFLOP/s peak. The peak entry
    is labelled measured (the operator's sourced declaration is the
    instrument of record) and MFU derived (the declaration's epistemic
    weight sits in the derivation, not in the peak).
    """
    callback = StepTelemetry(warmup_steps=0, flops_model=_FLOPS, device_peak=_PEAK)
    _drive_two_reading_run(callback, clock)
    expected_tflops = 250.0 * 60_288 / 1e12
    summary = callback.summary()
    assert summary["perf_model_tflops_per_second"] == (
        pytest.approx(expected_tflops),
        "derived",
    )
    assert summary["perf_model_tflops_per_second_per_device"][0] == pytest.approx(expected_tflops)
    assert summary["perf_device_peak_tflops"] == (100.0, "measured")
    assert summary["perf_mfu"] == (pytest.approx(expected_tflops / 100.0), "derived")

    monkeypatch.setattr(telemetry, "_world_size", lambda: 4)
    summary4 = callback.summary()
    assert summary4["perf_model_tflops_per_second_per_device"][0] == pytest.approx(
        expected_tflops / 4
    )
    assert summary4["perf_mfu"][0] == pytest.approx(expected_tflops / 4 / 100.0)


def test_memory_entries_on_a_cpu_machine_share_one_cuda_reason() -> None:
    """On this no-CUDA box the three memory counters and the utilisation
    fraction are all unmeasured, sharing ONE identical reason string in
    the value slot, and that reason mentions CUDA. The unmeasured branch
    IS the behaviour under test here; nothing is patched.
    """
    summary = StepTelemetry().summary()
    keys = (
        "perf_peak_memory_allocated_bytes",
        "perf_peak_memory_reserved_bytes",
        "perf_device_memory_total_bytes",
        "perf_memory_utilisation_fraction",
    )
    entries = [summary[key] for key in keys]
    assert all(source == "unmeasured" for _, source in entries)
    reason = entries[0][0]
    assert all(entry_reason == reason for entry_reason, _ in entries)
    assert isinstance(reason, str)
    assert reason.startswith("UNMEASURED")
    assert "CUDA" in reason


def test_memory_measured_branch_and_the_utilisation_fraction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When the CUDA probe returns counters, the three byte entries are
    measured as-read and the utilisation fraction is peak allocated over
    device total: 30/120 = 0.25 exactly, labelled derived.
    """
    monkeypatch.setattr(
        telemetry,
        "_read_cuda_memory_bytes",
        lambda: (30_000_000, 60_000_000, 120_000_000),
    )
    summary = StepTelemetry().summary()
    assert summary["perf_peak_memory_allocated_bytes"] == (30_000_000, "measured")
    assert summary["perf_peak_memory_reserved_bytes"] == (60_000_000, "measured")
    assert summary["perf_device_memory_total_bytes"] == (120_000_000, "measured")
    assert summary["perf_memory_utilisation_fraction"] == (
        pytest.approx(0.25),
        "derived",
    )


def test_memory_utilisation_unmeasured_when_the_device_reports_zero_total(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A device total of 0 bytes makes the fraction unmeasurable: the
    three counters are still measured as-read, but the utilisation entry
    is unmeasured with a reason (in the value slot) naming the zero total
    rather than a division by zero disguised as a number.
    """
    monkeypatch.setattr(telemetry, "_read_cuda_memory_bytes", lambda: (10, 20, 0))
    summary = StepTelemetry().summary()
    assert summary["perf_device_memory_total_bytes"] == (0, "measured")
    reason, source = summary["perf_memory_utilisation_fraction"]
    assert source == "unmeasured"
    assert "0 total bytes" in reason


def test_units_table_covers_summary_and_only_three_keys_are_unitless() -> None:
    """CONSISTENCY GATE: every key in PERF_TELEMETRY_UNITS is emitted by
    summary(), and exactly the three documented unitless ratios --
    perf_dataloader_stall_fraction, perf_mfu and
    perf_memory_utilisation_fraction -- are the summary keys the units
    table does not list. Both sets are derived from the code, so a key
    added on one side only fails immediately.
    """
    summary_keys = set(StepTelemetry().summary())
    unit_keys = set(PERF_TELEMETRY_UNITS)
    assert unit_keys - summary_keys == set()
    assert summary_keys - unit_keys == {
        "perf_dataloader_stall_fraction",
        "perf_mfu",
        "perf_memory_utilisation_fraction",
    }
    for key, unit in PERF_TELEMETRY_UNITS.items():
        assert isinstance(unit, str) and unit.strip(), key


def test_entry_degrades_none_none_to_a_reason_naming_the_perf_plane() -> None:
    """_entry(None, None) is never (None, None): it degrades to None
    paired with a reason that names the perf plane itself as the defect,
    and it does not raise -- the instrument must not turn a finished run
    into a traceback. (_entry is summary()'s internal seam; summary()
    re-routes its output into the manifest-shaped layout the tests above
    pin.)
    """
    value, reason = _entry(None, None)
    assert value is None
    assert isinstance(reason, str)
    assert "perf plane" in reason
    assert (value, reason) != (None, None)


def test_entry_pairs_a_value_with_its_source_and_a_gap_with_its_reason() -> None:
    """The non-degenerate shapes of the internal seam: a present value
    carries its source (default "derived") and discards the reason, while
    an absent value carries the caller's reason verbatim.
    """
    assert _entry(3.5, "irrelevant") == (3.5, "derived")
    assert _entry(7, "irrelevant", source="measured") == (7, "measured")
    assert _entry(None, "UNMEASURED: the input was absent") == (
        None,
        "UNMEASURED: the input was absent",
    )


def test_negative_warmup_is_refused_at_statement_time() -> None:
    """A negative warmup_steps is a config the operator did not mean --
    it would classify every step as steady-state -- and it is refused in
    the constructor with the field named.
    """
    with pytest.raises(ValueError, match="warmup_steps"):
        StepTelemetry(warmup_steps=-1)


def test_on_train_begin_resets_state_so_runs_cannot_leak(clock: _Clock) -> None:
    """A callback reused across two runs averages nothing across them:
    after a second on_train_begin, the summary is rebuilt from the new
    run alone, so five steps before the reset contribute nothing to a
    one-step run after it -- proven by the steady mean becoming
    unmeasured again (reason in the value slot, source "unmeasured")
    rather than silently absorbing the old samples.
    """
    callback = StepTelemetry(warmup_steps=3)
    callback.on_train_begin(_ARGS, _STATE, _CONTROL)
    for _ in range(5):
        _run_step(callback, clock, step_s=1.0)
    first = callback.summary()
    assert first["perf_steps_observed"] == (5, "measured")
    assert first["perf_steps_steady"] == (2, "measured")

    callback.on_train_begin(_ARGS, _STATE, _CONTROL)
    _run_step(callback, clock, step_s=1.0)
    second = callback.summary()
    assert second["perf_steps_observed"] == (1, "measured")
    assert second["perf_steps_warmup"] == (1, "measured")
    assert second["perf_steps_steady"] == (0, "measured")
    assert second["perf_step_time_mean_s"][1] == "unmeasured"


def test_step_end_without_a_begin_records_nothing(clock: _Clock) -> None:
    """An on_step_end with no matching begin is ignored: no duration is
    manufactured from stale state and the observed count stays zero.
    """
    callback = StepTelemetry()
    callback.on_step_end(_ARGS, _STATE, _CONTROL)
    callback.on_step_end(_ARGS, _STATE, _CONTROL)
    assert callback.summary()["perf_steps_observed"] == (0, "measured")


def test_a_fully_declared_run_reports_no_unmeasured_entry(
    clock: _Clock, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Closure check driving the WHOLE dict again: with batch geometry,
    token counter, optimizer hooks, a FlopsModel, a declared peak and a
    doubled CUDA probe all in place, no entry is None and every
    provenance is one of the two value-carrying labels -- the positive
    image of the zero-steps sweep.
    """
    monkeypatch.setattr(
        telemetry,
        "_read_cuda_memory_bytes",
        lambda: (10_000, 20_000, 40_000),
    )
    callback = StepTelemetry(warmup_steps=0, flops_model=_FLOPS, device_peak=_PEAK)
    callback.on_train_begin(_ARGS, _STATE, _CONTROL)
    _run_step(callback, clock, step_s=2.0, optimizer_s=0.5, tokens=500)
    _run_step(callback, clock, step_s=3.0, gap_s=3.0, optimizer_s=1.0, tokens=2000)
    summary = callback.summary()
    for key, (value, provenance) in summary.items():
        assert value is not None, key
        assert provenance in ("measured", "derived"), key


@pytest.mark.parametrize("tflops", [0.0, -5.0, float("nan"), float("inf")])
def test_device_peak_refuses_non_positive_or_non_finite_tflops(tflops: float) -> None:
    """DevicePeak range-checks its own declaration in __post_init__: a
    non-positive or non-finite peak would make MFU a division by zero or
    a negative efficiency, and it is refused with the field named.
    """
    with pytest.raises(ValueError, match="bf16_dense_tflops"):
        DevicePeak(name="bad", bf16_dense_tflops=tflops, source="datasheet")


def test_device_peak_refuses_a_source_that_names_nothing() -> None:
    """A peak whose source is blank is refused at statement time: a peak
    with no provenance cannot be audited, and an unauditable peak makes
    the MFU derived from it unauditable.
    """
    with pytest.raises(ValueError, match="must name where the number came from"):
        DevicePeak(name="bad", bf16_dense_tflops=100.0, source="   ")


def test_device_peak_from_env_treats_absence_as_not_declared(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Absent env is not declared: from_env returns None when either
    variable is missing, and None keeps MFU honestly unmeasured rather
    than coerced from a partial statement.
    """
    monkeypatch.delenv("FS_DEVICE_PEAK_TFLOPS", raising=False)
    monkeypatch.delenv("FS_DEVICE_PEAK_SOURCE", raising=False)
    monkeypatch.delenv("FS_DEVICE_PEAK_NAME", raising=False)
    assert DevicePeak.from_env() is None
    monkeypatch.setenv("FS_DEVICE_PEAK_TFLOPS", "100")
    assert DevicePeak.from_env() is None


def test_device_peak_from_env_reads_a_complete_declaration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A complete env declaration round-trips into a DevicePeak carrying
    the operator's number and source, with a default name that records
    which channel the declaration arrived through.
    """
    monkeypatch.setenv("FS_DEVICE_PEAK_TFLOPS", "989.5")
    monkeypatch.setenv("FS_DEVICE_PEAK_SOURCE", "H100 SXM datasheet, dense bf16 row")
    monkeypatch.delenv("FS_DEVICE_PEAK_NAME", raising=False)
    peak = DevicePeak.from_env()
    assert peak is not None
    assert peak.bf16_dense_tflops == 989.5
    assert peak.source == "H100 SXM datasheet, dense bf16 row"
    assert "FS_DEVICE_PEAK_TFLOPS" in peak.name


def test_device_peak_from_env_refuses_a_malformed_declaration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Declared-but-unusable is a different state from absent: a
    non-numeric tflops and a blank source both raise with the offending
    field named, rather than being silently demoted to unmeasured.
    """
    monkeypatch.setenv("FS_DEVICE_PEAK_TFLOPS", "fast")
    monkeypatch.setenv("FS_DEVICE_PEAK_SOURCE", "datasheet")
    with pytest.raises(ValueError, match="FS_DEVICE_PEAK_TFLOPS"):
        DevicePeak.from_env()
    monkeypatch.setenv("FS_DEVICE_PEAK_TFLOPS", "100")
    monkeypatch.setenv("FS_DEVICE_PEAK_SOURCE", "  ")
    with pytest.raises(ValueError, match="source"):
        DevicePeak.from_env()


def test_flops_per_token_is_the_stated_six_n_plus_attention_formula() -> None:
    """flops_per_token is 6 * non_embedding + 12 * layers * hidden * seq:
    6*8000 + 12*2*16*32 = 60_288. The `parameters` field is provenance,
    not an input -- a model differing only in it computes the same
    per-token figure.
    """
    assert _FLOPS.flops_per_token == 60_288
    same_formula = FlopsModel(
        parameters=5,
        non_embedding_parameters=8_000,
        layers=2,
        hidden_size=16,
        sequence_length=32,
    )
    assert same_formula.flops_per_token == _FLOPS.flops_per_token


def test_from_hf_config_reads_text_config_before_the_flat_config() -> None:
    """On a composite config the layer/geometry fields are read from
    text_config FIRST: the flat 99-layer attributes belong to another
    sub-model and must not win, and when text_config lacks a field there
    is NO fallback to the flat config -- a wrong value is worse than
    None.
    """
    flat = SimpleNamespace(num_hidden_layers=3, hidden_size=64)
    from_flat = FlopsModel.from_hf_config(
        flat, sequence_length=128, parameters=1_000, non_embedding_parameters=900
    )
    assert from_flat is not None
    assert from_flat.layers == 3 and from_flat.hidden_size == 64

    composite = SimpleNamespace(
        text_config=SimpleNamespace(num_hidden_layers=2, hidden_size=32),
        num_hidden_layers=99,
        hidden_size=99,
    )
    from_text = FlopsModel.from_hf_config(
        composite, sequence_length=128, parameters=1_000, non_embedding_parameters=900
    )
    assert from_text is not None
    assert from_text.layers == 2 and from_text.hidden_size == 32

    broken_text = SimpleNamespace(
        text_config=SimpleNamespace(num_hidden_layers=2),
        num_hidden_layers=99,
        hidden_size=99,
    )
    assert (
        FlopsModel.from_hf_config(
            broken_text, sequence_length=128, parameters=1, non_embedding_parameters=1
        )
        is None
    )


def test_from_hf_config_refuses_missing_and_flag_shaped_fields() -> None:
    """A bool is a flag, not a layer count: num_hidden_layers=True and a
    missing hidden_size both return None, because a partial FlopsModel is
    a guessed formula and the module never guesses.
    """
    assert (
        FlopsModel.from_hf_config(
            SimpleNamespace(num_hidden_layers=True, hidden_size=8),
            sequence_length=8,
            parameters=10,
            non_embedding_parameters=5,
        )
        is None
    )
    assert (
        FlopsModel.from_hf_config(
            SimpleNamespace(num_hidden_layers=2),
            sequence_length=8,
            parameters=10,
            non_embedding_parameters=5,
        )
        is None
    )
    assert (
        FlopsModel.from_hf_config(
            SimpleNamespace(num_hidden_layers=2, hidden_size=8),
            sequence_length=8,
            parameters=10,
            non_embedding_parameters=5,
        )
        is not None
    )


class _Param:
    """A parameter stand-in: the model walker reads only ``numel()``."""

    def __init__(self, numel: int) -> None:
        self._numel = numel

    def numel(self) -> int:
        return self._numel


class Embedding:
    """An embedding-module stand-in whose CLASS NAME is the payload.

    FlopsModel.from_model identifies embeddings by ``type(module).__name__``
    precisely so the telemetry module never imports torch; this stand-in
    honours the same contract, so the class must keep the name ``Embedding``.
    """

    def __init__(self, param: _Param) -> None:
        self._param = param

    def parameters(self, recurse: bool = False) -> Iterator[_Param]:
        return iter([self._param])


class _FakeModel:
    """A model stand-in with one dense parameter and one embedding tensor
    shared by two modules -- the tied lm_head/embedding shape."""

    def __init__(self) -> None:
        self.config = SimpleNamespace(num_hidden_layers=1, hidden_size=4)
        self._dense = _Param(50)
        self._shared = _Param(100)

    def named_parameters(self) -> Iterator[tuple[str, _Param]]:
        return iter(
            [
                ("dense.weight", self._dense),
                ("embed.weight", self._shared),
                ("lm_head.weight", self._shared),
            ]
        )

    def named_modules(self) -> Iterator[tuple[str, Embedding]]:
        return iter(
            [
                ("embed", Embedding(self._shared)),
                ("lm_head", Embedding(self._shared)),
            ]
        )


def test_from_model_counts_a_tied_embedding_once_by_identity() -> None:
    """Parameters are walked by object identity, so the tied tensor the
    lm_head shares with the embedding is counted once in the total (150,
    not 250) and subtracted once from the non-embedding count (50) -- and
    a model exposing no config yields no FlopsModel at all rather than
    one built on a guessed geometry.
    """
    model = FlopsModel.from_model(_FakeModel(), sequence_length=8)
    assert model is not None
    assert model.parameters == 150
    assert model.non_embedding_parameters == 50
    # 6*50 + 12*1*4*8 = 300 + 384 = 684.
    assert model.flops_per_token == 684
    assert FlopsModel.from_model(object(), sequence_length=8) is None
