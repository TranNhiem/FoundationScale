"""Tests for ``foundationscale.agentic_rl.evaluation`` (binding API)."""

from __future__ import annotations

import pytest

from foundationscale.agentic_rl.evaluation import (
    EvaluationRefusal,
    FailureCategory,
    measure_episode,
    render_markdown,
    report_by_pair,
    wilson_interval,
)
from foundationscale.agentic_rl.trajectory_v2 import (
    ApiFormat,
    EnvKind,
    EnvMeta,
    Episode,
    EpisodeStatus,
    Fidelity,
    FidelityStatus,
    GatewayMeta,
    HarnessMeta,
    ModelMeta,
    RewardEvent,
    RewardScope,
    Segment,
    Termination,
    Timings,
    validate_episode,
)

NAN = float("nan")


def _segment(
    *,
    segment_index: int = 0,
    prompt_ids: tuple[int, ...] = (1, 2),
    response_ids: tuple[int, ...] = (10, 11, 12, 13, 14, 15),
    loss_mask: tuple[int, ...] = (1, 1, 0, 0, 1, 1),
    generation_spans: tuple[tuple[int, int, str], ...] = ((0, 2, "g0"), (4, 6, "g1")),
) -> Segment:
    """Build a valid Segment with NaN logprobs and -1 versions on mask-0 tokens."""
    logprobs = tuple(NAN if m == 0 else -0.5 for m in loss_mask)
    policy_version = tuple(-1 if m == 0 else 3 for m in loss_mask)
    return Segment(
        segment_index=segment_index,
        prompt_ids=prompt_ids,
        response_ids=response_ids,
        loss_mask=loss_mask,
        logprobs=logprobs,
        policy_version=policy_version,
        generation_spans=generation_spans,
    )


def _episode(
    *,
    episode_id: str = "ep-1",
    task_id: str = "task-1",
    status: EpisodeStatus = EpisodeStatus.OK,
    segments: tuple[Segment, ...] | None = None,
    reward_events: tuple[RewardEvent, ...] = (),
    model: str = "m1",
    harness: str = "h1",
    timings: Timings | None = None,
    trainable_on_timeout: bool = False,
    fidelity: Fidelity | None = None,
) -> Episode:
    """Build a valid Episode with default metadata and the given segments."""
    if segments is None:
        segments = (_segment(),)
    if timings is None:
        timings = Timings()
    if fidelity is None:
        fidelity = (
            Fidelity(status=FidelityStatus.MISMATCH, detail="mismatch")
            if status is EpisodeStatus.FIDELITY_MISMATCH
            else Fidelity(status=FidelityStatus.VERIFIED)
        )
    ep = Episode(
        episode_id=episode_id,
        group_id="grp-1",
        task_prompt_ids=(1, 2),
        status=status,
        termination=Termination.STOP,
        segments=segments,
        reward_events=reward_events,
        model=ModelMeta(name=model, version=1, checkpoint_sha="sha"),
        harness=HarnessMeta(name=harness, version="1"),
        env=EnvMeta(kind=EnvKind.FS_LOCAL, task_id=task_id),
        gateway=GatewayMeta(api_format=ApiFormat.CHAT_COMPLETIONS, session_id="s1"),
        timings=timings,
        fidelity=fidelity,
        trainable_on_timeout=trainable_on_timeout,
    )
    validate_episode(ep)
    return ep


def _reward(value: float | None) -> tuple[RewardEvent, ...]:
    """One EPISODE-scope reward event with the given value (None is an abstention)."""
    return (
        RewardEvent(
            source="verifier",
            verifier_id="v",
            verifier_version="1",
            value=value,
            scope=RewardScope.EPISODE,
        ),
    )


def test_category_mapping_for_every_episode_status() -> None:
    """Every EpisodeStatus maps 1:1 to its FailureCategory, TIMEOUT regardless of trainable."""
    expected = {
        EpisodeStatus.TIMEOUT: FailureCategory.TIMEOUT,
        EpisodeStatus.CONTEXT_OVERFLOW: FailureCategory.CONTEXT_OVERFLOW,
        EpisodeStatus.HARNESS_ERROR: FailureCategory.HARNESS_ERROR,
        EpisodeStatus.SANDBOX_ERROR: FailureCategory.SANDBOX_ERROR,
        EpisodeStatus.INFRA_ERROR: FailureCategory.INFRA_ERROR,
        EpisodeStatus.VERIFIER_ERROR: FailureCategory.VERIFIER_ERROR,
        EpisodeStatus.FIDELITY_MISMATCH: FailureCategory.FIDELITY_MISMATCH,
    }
    for status, category in expected.items():
        ep = _episode(status=status, trainable_on_timeout=True)
        assert measure_episode(ep).category is category
    ok = _episode(status=EpisodeStatus.OK, reward_events=_reward(1.0))
    assert measure_episode(ok).category is FailureCategory.PASSED


def test_passed_vs_wrong_answer_around_pass_threshold() -> None:
    """Reward at or above pass_threshold is PASSED; just below is WRONG_ANSWER."""
    at = _episode(episode_id="a", reward_events=_reward(0.8))
    below = _episode(episode_id="b", reward_events=_reward(0.7999))
    assert measure_episode(at, pass_threshold=0.8).category is FailureCategory.PASSED
    assert measure_episode(below, pass_threshold=0.8).category is FailureCategory.WRONG_ANSWER


def test_no_reward_when_no_reward_event() -> None:
    """An OK episode with no reward event is NO_REWARD with reward None."""
    m = measure_episode(_episode())
    assert m.category is FailureCategory.NO_REWARD
    assert m.reward is None


def test_sampled_context_tokens_and_turns_on_two_span_segment() -> None:
    """Sampled/context token counts and turn counts come from the loss mask and spans."""
    m = measure_episode(_episode())
    assert m.sampled_tokens == 4
    assert m.context_tokens == 2
    assert m.turns == 2


def test_no_action_turns_last_span_never_counted() -> None:
    """Only non-final spans without tool calls count; the last span never counts."""
    counts = {"g0": 0, "g1": 0}
    m = measure_episode(_episode(), tool_calls_per_generation=counts)
    assert m.no_action_turns == 1
    assert m.actions_known is True
    both = {"g0": 1, "g1": 2}
    assert measure_episode(_episode(), tool_calls_per_generation=both).no_action_turns == 0


def test_actions_known_false_when_tool_calls_not_given() -> None:
    """Without tool_calls_per_generation no_action_turns is 0 and actions_known is False."""
    m = measure_episode(_episode())
    assert m.no_action_turns == 0
    assert m.actions_known is False


def test_report_by_pair_grouping_sorting_and_pass_rate() -> None:
    """Pairs group by (model, harness), sort by it, and pass_rate is passed/episodes."""
    measures = (
        measure_episode(_episode(episode_id="e1", model="b", reward_events=_reward(1.0))),
        measure_episode(_episode(episode_id="e2", model="a", reward_events=_reward(0.0))),
        measure_episode(_episode(episode_id="e3", model="a", reward_events=_reward(1.0))),
    )
    reports = report_by_pair(measures)
    assert [(r.model, r.harness) for r in reports] == [("a", "h1"), ("b", "h1")]
    assert reports[0].episodes == 2
    assert reports[0].passed == 1
    assert reports[0].pass_rate == 0.5
    assert reports[1].pass_rate == 1.0


def test_wilson_interval_known_value() -> None:
    """Wilson score interval for 8 of 10 is about (0.490, 0.943)."""
    low, high = wilson_interval(8, 10)
    assert low == pytest.approx(0.4902, abs=5e-4)
    assert high == pytest.approx(0.9433, abs=5e-4)


def test_mean_reward_none_when_no_rewards() -> None:
    """mean_reward is None when no episode in the pair carries a reward."""
    report = report_by_pair((measure_episode(_episode()),))[0]
    assert report.mean_reward is None


def test_failures_counts_only_non_zero_and_sorted() -> None:
    """failures maps category values to counts, omits zeros, and is sorted by key."""
    measures = (
        measure_episode(_episode(episode_id="f1", status=EpisodeStatus.TIMEOUT)),
        measure_episode(_episode(episode_id="f2", status=EpisodeStatus.TIMEOUT)),
        measure_episode(_episode(episode_id="f3", status=EpisodeStatus.INFRA_ERROR)),
    )
    failures = report_by_pair(measures)[0].failures
    assert dict(failures) == {"infra_error": 1, "timeout": 2}
    assert list(failures) == sorted(failures)


def test_tasks_distinct_count() -> None:
    """tasks counts distinct task_ids across the pair's episodes."""
    measures = (
        measure_episode(_episode(episode_id="t1", task_id="x")),
        measure_episode(_episode(episode_id="t2", task_id="x")),
        measure_episode(_episode(episode_id="t3", task_id="y")),
    )
    assert report_by_pair(measures)[0].tasks == 2


def test_median_wall_ms_none_when_no_timings() -> None:
    """median_wall_ms is None when no episode carries timings.run_ms."""
    assert report_by_pair((measure_episode(_episode()),))[0].median_wall_ms is None
    timed = measure_episode(_episode(timings=Timings(run_ms=100.0)))
    assert report_by_pair((timed,))[0].median_wall_ms == 100.0


def test_empty_input_refused() -> None:
    """Empty measure sequences are refused, never reported as a 0% row."""
    with pytest.raises(EvaluationRefusal):
        report_by_pair(())
    with pytest.raises(EvaluationRefusal):
        render_markdown(())


def test_render_markdown_header_and_none_em_dash() -> None:
    """render_markdown emits a header row and renders None values as an em dash."""
    report = report_by_pair((measure_episode(_episode()),))[0]
    text = render_markdown((report,))
    lines = text.splitlines()
    assert lines[0].startswith("| model | harness |")
    assert "—" in text


def test_episode_measure_refusals() -> None:
    """Invalid EpisodeMeasure fields are refused at construction."""
    from dataclasses import replace

    base = measure_episode(_episode())
    cases = [
        ("episode_id", ""),
        ("episode_id", 1),
        ("model", ""),
        ("model", None),
        ("model_version", -1),
        ("model_version", 1.5),
        ("model_version", True),
        ("harness", ""),
        ("harness", 3),
        ("harness_version", ""),
        ("harness_version", None),
        ("task_id", ""),
        ("task_id", 0),
        ("category", "passed"),
        ("category", None),
        ("reward", "1.0"),
        ("reward", True),
        ("sampled_tokens", -1),
        ("sampled_tokens", 1.5),
        ("sampled_tokens", True),
        ("context_tokens", -1),
        ("context_tokens", 2.5),
        ("context_tokens", False),
        ("turns", -1),
        ("turns", 2.5),
        ("turns", True),
        ("no_action_turns", -1),
        ("no_action_turns", 1.5),
        ("no_action_turns", True),
        ("no_action_turns", 3),
        ("actions_known", 1),
        ("actions_known", "yes"),
        ("wall_ms", -1.0),
        ("wall_ms", NAN),
        ("wall_ms", float("inf")),
        ("wall_ms", "10"),
    ]
    for field, value in cases:
        with pytest.raises(EvaluationRefusal):
            replace(base, **{field: value})


def test_pair_report_refusals() -> None:
    """Invalid PairReport fields are refused at construction."""
    from dataclasses import replace

    base = report_by_pair((measure_episode(_episode()),))[0]
    cases = [
        ("model", ""),
        ("model", None),
        ("harness", ""),
        ("harness", 2),
        ("episodes", 0),
        ("episodes", 1.5),
        ("episodes", True),
        ("passed", -1),
        ("passed", 2),
        ("passed", 0.5),
        ("passed", True),
        ("pass_rate", -0.1),
        ("pass_rate", 1.5),
        ("pass_rate", True),
        ("pass_rate_ci95", [0.0, 1.0]),
        ("pass_rate_ci95", (0.0,)),
        ("pass_rate_ci95", (0.5, 0.2)),
        ("pass_rate_ci95", (-0.1, 0.5)),
        ("pass_rate_ci95", (0.5, 1.5)),
        ("pass_rate_ci95", ("a", "b")),
        ("mean_reward", "x"),
        ("mean_reward", True),
        ("mean_sampled_tokens", -1.0),
        ("mean_sampled_tokens", NAN),
        ("mean_sampled_tokens", float("inf")),
        ("mean_sampled_tokens", "1"),
        ("mean_turns", -0.5),
        ("mean_turns", NAN),
        ("mean_turns", float("inf")),
        ("mean_turns", True),
        ("no_action_turn_rate", "x"),
        ("no_action_turn_rate", True),
        ("no_action_turn_rate", -0.1),
        ("no_action_turn_rate", 1.5),
        ("median_wall_ms", -1.0),
        ("median_wall_ms", NAN),
        ("median_wall_ms", float("inf")),
        ("median_wall_ms", "5"),
        ("failures", [("a", 1)]),
        ("failures", {"b": 1, "a": 1}),
        ("failures", {"": 1}),
        ("failures", {1: 1}),
        ("failures", {"a": 0}),
        ("failures", {"a": -1}),
        ("failures", {"a": 1.5}),
        ("failures", {"a": True}),
        ("failures", {"a": 2}),
        ("failures", {}),
        ("tasks", 0),
        ("tasks", 2),
        ("tasks", 1.5),
        ("tasks", True),
    ]
    for field, value in cases:
        with pytest.raises(EvaluationRefusal):
            replace(base, **{field: value})


def test_measure_episode_refusals() -> None:
    """measure_episode refuses non-Episode input and invalid arguments."""
    with pytest.raises(EvaluationRefusal):
        measure_episode("not-an-episode")
    with pytest.raises(EvaluationRefusal):
        measure_episode(_episode(), pass_threshold="1.0")
    with pytest.raises(EvaluationRefusal):
        measure_episode(_episode(), pass_threshold=True)
    with pytest.raises(EvaluationRefusal):
        measure_episode(_episode(), tool_calls_per_generation=[("g0", 1)])


def test_no_action_rate_none_when_only_final_turns() -> None:
    """Action-known single-span episodes have no countable turns, so the rate is unmeasured."""
    m = measure_episode(
        _episode(
            segments=(
                _segment(response_ids=(10, 11), loss_mask=(1, 1), generation_spans=((0, 2, "g0"),)),
            )
        ),
        tool_calls_per_generation={},
    )
    assert m.actions_known and m.turns == 1
    assert report_by_pair((m,))[0].no_action_turn_rate is None


def test_wilson_interval_refusals() -> None:
    """wilson_interval refuses invalid n, successes and z."""
    with pytest.raises(EvaluationRefusal):
        wilson_interval(0, 0)
    with pytest.raises(EvaluationRefusal):
        wilson_interval(1, 1.5)
    with pytest.raises(EvaluationRefusal):
        wilson_interval(1, True)
    with pytest.raises(EvaluationRefusal):
        wilson_interval(-1, 10)
    with pytest.raises(EvaluationRefusal):
        wilson_interval(11, 10)
    with pytest.raises(EvaluationRefusal):
        wilson_interval(1.5, 10)
    with pytest.raises(EvaluationRefusal):
        wilson_interval(True, 10)
    with pytest.raises(EvaluationRefusal):
        wilson_interval(1, 10, z=0.0)
    with pytest.raises(EvaluationRefusal):
        wilson_interval(1, 10, z=-1.0)
    with pytest.raises(EvaluationRefusal):
        wilson_interval(1, 10, z=NAN)
    with pytest.raises(EvaluationRefusal):
        wilson_interval(1, 10, z="1.96")


def test_report_by_pair_refusals() -> None:
    """report_by_pair refuses non-sequence input and non-EpisodeMeasure entries."""
    with pytest.raises(EvaluationRefusal):
        report_by_pair(3)
    with pytest.raises(EvaluationRefusal):
        report_by_pair("abc")
    with pytest.raises(EvaluationRefusal):
        report_by_pair((measure_episode(_episode()), "not-a-measure"))
    with pytest.raises(EvaluationRefusal):
        report_by_pair((None, None))


def test_render_markdown_refusals() -> None:
    """render_markdown refuses empty, non-sequence and non-PairReport input."""
    from foundationscale.agentic_rl.evaluation import PairReport

    report = report_by_pair((measure_episode(_episode()),))[0]
    with pytest.raises(EvaluationRefusal):
        render_markdown(())
    with pytest.raises(EvaluationRefusal):
        render_markdown(5)
    with pytest.raises(EvaluationRefusal):
        render_markdown("nope")
    with pytest.raises(EvaluationRefusal):
        render_markdown((report, "not-a-report"))
    with pytest.raises(EvaluationRefusal):
        render_markdown((PairReport, PairReport))


def test_no_action_turn_rate_over_non_final_turns() -> None:
    """The rate divides no-action turns by non-final turns of action-known episodes only."""
    idle = measure_episode(_episode(episode_id="a"), tool_calls_per_generation={})
    acted = measure_episode(_episode(episode_id="b"), tool_calls_per_generation={"g0": 1})
    unknown = measure_episode(_episode(episode_id="c"))
    assert (idle.no_action_turns, acted.no_action_turns) == (1, 0)
    assert report_by_pair((idle, acted, unknown))[0].no_action_turn_rate == 0.5
