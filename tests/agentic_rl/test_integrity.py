"""Tests for section A of the BINDING reward-integrity API (P0.7)."""

from __future__ import annotations

import pytest

from foundationscale.agentic_rl import integrity as ig
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
    validate_episode,
)

NAN = float("nan")


def _segment(index: int = 0) -> Segment:
    """Build a valid segment: mask-1 tokens have finite logprobs and pv >= 0."""
    return Segment(
        segment_index=index,
        prompt_ids=(1, 2),
        response_ids=(3, 4, 5, 6),
        loss_mask=(1, 0, 1, 0),
        logprobs=(-0.5, NAN, -1.5, NAN),
        policy_version=(7, -1, 7, -1),
        generation_spans=((0, 1, f"g{index}a"), (2, 3, f"g{index}b")),
    )


def _episode(
    eid: str = "ep-0",
    status: EpisodeStatus = EpisodeStatus.OK,
    reward: float | None = None,
) -> Episode:
    """Build a valid Episode with an optional single EPISODE-scope reward event."""
    events = ()
    if reward is not None:
        events = (
            RewardEvent(
                source="verifier",
                verifier_id="v1",
                verifier_version="1.0",
                value=reward,
                scope=RewardScope.EPISODE,
            ),
        )
    ep = Episode(
        episode_id=eid,
        group_id="grp",
        task_prompt_ids=(1, 2),
        status=status,
        termination=Termination.STOP,
        segments=(_segment(),),
        reward_events=events,
        model=ModelMeta(name="m", version=1, checkpoint_sha="sha"),
        harness=HarnessMeta(name="h", version="1"),
        env=EnvMeta(kind=EnvKind.FS_LOCAL, task_id="t1"),
        gateway=GatewayMeta(api_format=ApiFormat.WHITE_BOX, session_id="s1"),
        fidelity=Fidelity(status=FidelityStatus.VERIFIED),
    )
    validate_episode(ep)
    return ep


def test_ingest_reward_accepts_int_float_str() -> None:
    """ingest_reward accepts int, float and numeric str, returning float."""
    assert ig.ingest_reward(1) == 1.0
    assert ig.ingest_reward(0.25) == 0.25
    assert ig.ingest_reward("0.5") == 0.5
    assert isinstance(ig.ingest_reward(0), float)


@pytest.mark.parametrize("raw", [None, float("nan"), float("inf"), float("-inf")])
def test_ingest_reward_refuses_nonfinite(raw: object) -> None:
    """ingest_reward refuses None, NaN and +/-Infinity naming the raw value."""
    with pytest.raises(ig.IntegrityRefusal) as e:
        ig.ingest_reward(raw)
    assert repr(raw) in str(e.value)


@pytest.mark.parametrize("raw", [True, False])
def test_ingest_reward_refuses_bool(raw: bool) -> None:
    """ingest_reward refuses bools since type(raw) is bool."""
    with pytest.raises(ig.IntegrityRefusal):
        ig.ingest_reward(raw)


@pytest.mark.parametrize("raw", [-0.1, 1.5, "2.0", "abc", object()])
def test_ingest_reward_refuses_out_of_range_or_unparsable(raw: object) -> None:
    """ingest_reward refuses out-of-range and unparsable values naming the raw value."""
    with pytest.raises(ig.IntegrityRefusal) as e:
        ig.ingest_reward(raw)
    assert repr(raw) in str(e.value)


def test_ingest_reward_snap_binary() -> None:
    """snap_binary accepts exactly low or high and refuses anything between."""
    pol = ig.IngestPolicy(low=0.0, high=1.0, snap_binary=True)
    assert ig.ingest_reward(0.0, pol) == 0.0
    assert ig.ingest_reward(1, pol) == 1.0
    with pytest.raises(ig.IntegrityRefusal):
        ig.ingest_reward(0.5, pol)


@pytest.mark.parametrize(
    "low,high",
    [(0.0, 0.0), (1.0, 0.0), (float("nan"), 1.0), (0.0, float("inf")), (True, 1.0)],
)
def test_ingest_policy_refuses_bad_bounds(low: object, high: object) -> None:
    """IngestPolicy requires finite bounds with low < high."""
    with pytest.raises(ig.IntegrityRefusal):
        ig.IngestPolicy(low=low, high=high)  # type: ignore[arg-type]


def test_mask_failures_exact_counts() -> None:
    """mask_failures reports exact total/masked/scored/passed counts and pass_rate."""
    eps = [
        _episode("a", EpisodeStatus.OK, 1.0),
        _episode("b", EpisodeStatus.OK, 0.5),
        _episode("c", EpisodeStatus.OK, None),
    ]
    rep = ig.mask_failures(eps)
    assert (rep.total, rep.masked, rep.scored, rep.passed) == (3, 1, 2, 1)
    assert rep.pass_rate == 0.5
    assert [d.masked for d in rep.decisions] == [False, False, True]
    assert rep.decisions[2].reward is None
    assert rep.decisions[2].reason == "no_reward"
    assert rep.decisions[0].reason == ""


@pytest.mark.parametrize(
    "status",
    sorted(ig.INFRA_STATUSES, key=lambda s: s.value),
)
def test_mask_failures_masks_infra(status: EpisodeStatus) -> None:
    """Every INFRA_STATUSES episode is masked with reward None and reason = status."""
    ep = _episode("x", status, 0.0)
    rep = ig.mask_failures([ep])
    d = rep.decisions[0]
    assert d.masked is True
    assert d.reward is None
    assert d.reason == status.value
    assert (rep.total, rep.masked, rep.scored, rep.passed) == (1, 1, 0, 0)


def test_mask_failures_infra_statuses_exact_set() -> None:
    """INFRA_STATUSES is exactly the four documented statuses."""
    assert (
        frozenset(
            {
                EpisodeStatus.SANDBOX_ERROR,
                EpisodeStatus.INFRA_ERROR,
                EpisodeStatus.VERIFIER_ERROR,
                EpisodeStatus.HARNESS_ERROR,
            }
        )
        == ig.INFRA_STATUSES
    )


def test_mask_failures_fire_infra_pass() -> None:
    """MUST_FIRE: an infra episode with reward >= pass_threshold is refused naming ids."""
    eps = [_episode("a", EpisodeStatus.OK, 1.0), _episode("bad", EpisodeStatus.SANDBOX_ERROR, 1.0)]
    with pytest.raises(ig.IntegrityRefusal) as e:
        ig.mask_failures(eps)
    msg = str(e.value)
    assert "1 of 2" in msg
    assert "bad" in msg


def test_mask_failures_fire_infra_pass_multi() -> None:
    """MUST_FIRE: multiple infra passes are counted and all ids are named."""
    eps = [
        _episode("i1", EpisodeStatus.INFRA_ERROR, 1.0),
        _episode("i2", EpisodeStatus.VERIFIER_ERROR, 1.0),
        _episode("ok", EpisodeStatus.OK, 1.0),
    ]
    with pytest.raises(ig.IntegrityRefusal) as e:
        ig.mask_failures(eps)
    assert "2 of 3" in str(e.value)
    assert "i1" in str(e.value) and "i2" in str(e.value)


def test_mask_failures_pass_rate_none_when_no_scored() -> None:
    """pass_rate is None (never 0.0) when scored == 0."""
    rep = ig.mask_failures([_episode("a", EpisodeStatus.OK, None)])
    assert rep.pass_rate is None
    assert (rep.scored, rep.passed) == (0, 0)


def test_mask_failures_empty_refuses() -> None:
    """mask_failures refuses empty input."""
    with pytest.raises(ig.IntegrityRefusal):
        ig.mask_failures([])


def test_mask_failures_pass_threshold() -> None:
    """pass_threshold counts only scored rewards >= threshold."""
    eps = [_episode("a", EpisodeStatus.OK, 0.8), _episode("b", EpisodeStatus.OK, 0.9)]
    rep = ig.mask_failures(eps, pass_threshold=0.85)
    assert (rep.scored, rep.passed) == (2, 1)
    assert rep.pass_rate == 0.5


def test_mask_failures_abstention_value_none() -> None:
    """A reward event with value None is an abstention and yields reward None."""
    ep = Episode(
        episode_id="ab",
        group_id="grp",
        task_prompt_ids=(1,),
        status=EpisodeStatus.OK,
        termination=Termination.STOP,
        segments=(_segment(),),
        reward_events=(
            RewardEvent(
                source="v",
                verifier_id="v1",
                verifier_version="1",
                value=None,
                scope=RewardScope.EPISODE,
            ),
        ),
        model=ModelMeta(name="m", version=1, checkpoint_sha="sha"),
        harness=HarnessMeta(name="h", version="1"),
        env=EnvMeta(kind=EnvKind.FS_LOCAL, task_id="t"),
        gateway=GatewayMeta(api_format=ApiFormat.WHITE_BOX, session_id="s"),
    )
    rep = ig.mask_failures([ep])
    assert rep.decisions[0].reward is None
    assert rep.decisions[0].masked is True
    assert rep.decisions[0].reason == "no_reward"


def test_stage_budget_effective_s() -> None:
    """effective_s is min(override or task, max) * multiplier."""
    b = ig.StageBudget(task_timeout_s=10.0, max_timeout_s=8.0, multiplier=2.0)
    assert b.effective_s() == 16.0
    b2 = ig.StageBudget(task_timeout_s=10.0, max_timeout_s=8.0, override_timeout_s=5.0)
    assert b2.effective_s() == 5.0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"task_timeout_s": 0.0, "max_timeout_s": 1.0},
        {"task_timeout_s": 1.0, "max_timeout_s": 0.0},
        {"task_timeout_s": 1.0, "max_timeout_s": 1.0, "multiplier": 0.0},
        {"task_timeout_s": 1.0, "max_timeout_s": 1.0, "grace_s": -1.0},
        {"task_timeout_s": float("nan"), "max_timeout_s": 1.0},
    ],
)
def test_stage_budget_refuses_bad_values(kwargs: dict) -> None:
    """StageBudget requires finite positive fields and non-negative grace."""
    with pytest.raises(ig.IntegrityRefusal):
        ig.StageBudget(**kwargs)


def test_check_stage_pass_within_grace() -> None:
    """check_stage passes when elapsed <= effective + grace."""
    b = ig.StageBudget(task_timeout_s=10.0, max_timeout_s=10.0, grace_s=5.0)
    ig.check_stage("run", b, 15.0)
    ig.check_stage("run", b, 0.0)


def test_check_stage_fire_overrun() -> None:
    """MUST_FIRE: elapsed beyond effective + grace raises 'stage overrun'."""
    b = ig.StageBudget(task_timeout_s=10.0, max_timeout_s=10.0, grace_s=5.0)
    with pytest.raises(ig.IntegrityRefusal) as e:
        ig.check_stage("run", b, 15.0001)
    assert "stage overrun" in str(e.value)
    assert "run" in str(e.value)


def test_check_stage_fire_unmeasured() -> None:
    """MUST_FIRE: elapsed None is refused as unmeasured, never treated as 0.0."""
    b = ig.StageBudget(task_timeout_s=10.0, max_timeout_s=10.0)
    with pytest.raises(ig.IntegrityRefusal) as e:
        ig.check_stage("reward", b, None)
    assert "elapsed unmeasured" in str(e.value)
    assert "reward" in str(e.value)


@pytest.mark.parametrize("elapsed", [-1.0, float("nan"), float("inf"), float("-inf")])
def test_check_stage_fire_bad_elapsed(elapsed: float) -> None:
    """check_stage refuses negative and non-finite elapsed values."""
    b = ig.StageBudget(task_timeout_s=10.0, max_timeout_s=10.0)
    with pytest.raises(ig.IntegrityRefusal):
        ig.check_stage("run", b, elapsed)


def test_audit_selected_deterministic() -> None:
    """audit_selected is deterministic and matches the documented hash formula."""
    import hashlib

    eid = "ep-42"
    digest = hashlib.sha256(f"fs-fp-audit:{eid}".encode()).hexdigest()[:8]
    expected = int(digest, 16) / 0xFFFFFFFF < 0.5
    assert ig.audit_selected(eid, sample_rate=0.5) is expected
    assert ig.audit_selected(eid, sample_rate=0.5) == ig.audit_selected(eid, sample_rate=0.5)


def test_audit_selected_salt_changes_selection() -> None:
    """A different salt yields a different deterministic selection stream."""
    ids = [f"ep-{i}" for i in range(200)]
    a = [ig.audit_selected(i, sample_rate=0.5) for i in ids]
    b = [ig.audit_selected(i, sample_rate=0.5, salt="other") for i in ids]
    assert a != b


@pytest.mark.parametrize("rate", [-0.1, 1.5, float("nan")])
def test_audit_selected_refuses_bad_rate(rate: float) -> None:
    """audit_selected requires sample_rate in (0, 1]."""
    with pytest.raises(ig.IntegrityRefusal):
        ig.audit_selected("ep", sample_rate=rate)


def test_run_fp_audit_exact_counts() -> None:
    """run_fp_audit reports exact verifier_passes/audited/abstained/false_positives."""
    passes = {"a": 1.0, "b": 1.0, "c": 0.2}
    rep = ig.run_fp_audit(passes, lambda i: 1.0, sample_rate=0.0, min_audited=2)
    assert rep.verifier_passes == 2
    assert rep.audited == 2
    assert rep.abstained == 0
    assert rep.false_positives == 0
    assert rep.audited_fraction == 1.0
    assert rep.false_positive_rate == 0.0


def test_run_fp_audit_ignores_below_threshold() -> None:
    """Only entries >= pass_threshold count as verifier passes."""
    passes = {"a": 1.0, "b": 0.99, "c": 0.0}
    rep = ig.run_fp_audit(passes, lambda i: 1.0, sample_rate=0.0, min_audited=1)
    assert rep.verifier_passes == 1
    assert [o.episode_id for o in rep.outcomes] == ["a"]


def test_run_fp_audit_min_audited_topup_lexicographic() -> None:
    """min_audited top-up adds the lexicographically smallest unselected passes."""
    passes = {"z": 1.0, "a": 1.0, "m": 1.0}
    rep = ig.run_fp_audit(passes, lambda i: 1.0, sample_rate=0.0, min_audited=2)
    assert [o.episode_id for o in rep.outcomes] == ["a", "m"]


def test_run_fp_audit_min_audited_capped_by_passes() -> None:
    """min_audited is capped at verifier_passes."""
    passes = {"a": 1.0}
    rep = ig.run_fp_audit(passes, lambda i: 1.0, sample_rate=0.0, min_audited=5)
    assert rep.audited == 1


def test_run_fp_audit_forced_ids() -> None:
    """Forced ids are always selected even at sample_rate 0."""
    passes = {"a": 1.0, "b": 1.0, "c": 1.0}
    rep = ig.run_fp_audit(passes, lambda i: 1.0, sample_rate=0.0, min_audited=1, forced=["c"])
    assert [o.episode_id for o in rep.outcomes] == ["c"]


def test_run_fp_audit_forced_unknown_refuses() -> None:
    """Forced ids must be present in passes else refuse."""
    with pytest.raises(ig.IntegrityRefusal) as e:
        ig.run_fp_audit({"a": 1.0}, lambda i: 1.0, sample_rate=0.0, forced=["zz"])
    assert "zz" in str(e.value)


def test_run_fp_audit_abstain_none() -> None:
    """audit returning None abstains and false_positive is None."""
    rep = ig.run_fp_audit({"a": 1.0}, lambda i: None, sample_rate=0.0, min_audited=1)
    o = rep.outcomes[0]
    assert o.audit_reward is None
    assert o.false_positive is None
    assert rep.abstained == 1
    assert rep.false_positive_rate is None


def test_run_fp_audit_false_positive_rate_denominator() -> None:
    """false_positive_rate divides by audited - abstained and is None when that is 0."""
    passes = {"a": 1.0, "b": 1.0, "c": 1.0}
    rep = ig.run_fp_audit(
        passes,
        lambda i: 0.0 if i == "a" else (None if i == "b" else 1.0),
        sample_rate=0.0,
        min_audited=3,
    )
    assert (rep.audited, rep.abstained, rep.false_positives) == (3, 1, 1)
    assert rep.false_positive_rate == 0.5


def test_run_fp_audit_no_passes_refuses() -> None:
    """run_fp_audit refuses when there are no verifier passes."""
    with pytest.raises(ig.IntegrityRefusal):
        ig.run_fp_audit({}, lambda i: 1.0)


def test_run_fp_audit_audit_exception_propagates() -> None:
    """Exceptions raised by the audit callable propagate unchanged."""

    def boom(eid: str) -> float | None:
        raise RuntimeError("audit down")

    with pytest.raises(RuntimeError, match="audit down"):
        ig.run_fp_audit({"a": 1.0}, boom, sample_rate=0.0, min_audited=1)


def test_run_fp_audit_seeded_false_positive_fires() -> None:
    """Seeded selection finds a false positive and require_no_false_positives MUST_FIREs."""
    passes = {f"ep-{i}": 1.0 for i in range(200)}
    rep = ig.run_fp_audit(
        passes,
        lambda i: 0.0 if i == "ep-3" else 1.0,
        sample_rate=0.05,
        forced=("ep-3",),
    )
    assert rep.audited > 0
    assert rep.false_positives == 1
    with pytest.raises(ig.IntegrityRefusal) as e:
        ig.require_no_false_positives(rep)
    msg = str(e.value)
    assert "1 of" in msg
    assert "ep-3" in msg


def test_require_no_false_positives_pass_clean() -> None:
    """MUST_PASS: a report with zero false positives passes."""
    rep = ig.run_fp_audit({"a": 1.0}, lambda i: 1.0, sample_rate=0.0, min_audited=1)
    ig.require_no_false_positives(rep)


def test_require_no_false_positives_names_counts_and_ids() -> None:
    """The refusal names 'k of m audited' and every false-positive id."""
    passes = {"a": 1.0, "b": 1.0, "c": 1.0}
    rep = ig.run_fp_audit(
        passes,
        lambda i: 0.0 if i in ("a", "c") else 1.0,
        sample_rate=0.0,
        min_audited=3,
    )
    with pytest.raises(ig.IntegrityRefusal) as e:
        ig.require_no_false_positives(rep)
    msg = str(e.value)
    assert "2 of 3" in msg
    assert "a" in msg and "c" in msg


def test_check_task_smoke_pass() -> None:
    """MUST_PASS: oracle >= threshold and empty < threshold for every task."""
    results = [
        ig.SmokeResult("t1", 1.0, 0.0),
        ig.SmokeResult("t2", 0.9, 0.1),
    ]
    ig.check_task_smoke(results, pass_threshold=0.5)


def test_check_task_smoke_fire_broken() -> None:
    """MUST_FIRE: broken tasks are refused naming 'k of n tasks broken' and each id."""
    results = [
        ig.SmokeResult("t1", 1.0, 0.0),
        ig.SmokeResult("t2", 0.0, 0.0),
        ig.SmokeResult("t3", 1.0, 1.0),
    ]
    with pytest.raises(ig.IntegrityRefusal) as e:
        ig.check_task_smoke(results, pass_threshold=0.5)
    msg = str(e.value)
    assert "2 of 3" in msg
    assert "t2" in msg and "t3" in msg


@pytest.mark.parametrize(
    "oracle,empty",
    [(None, 0.0), (1.0, None), (None, None), (0.0, 1.0)],
)
def test_check_task_smoke_fire_none_or_inverted(oracle: float | None, empty: float | None) -> None:
    """None counts as broken on either side and inverted rewards are broken."""
    with pytest.raises(ig.IntegrityRefusal) as e:
        ig.check_task_smoke([ig.SmokeResult("t", oracle, empty)])
    assert "1 of 1" in str(e.value)
    assert "t" in str(e.value)


def test_check_task_smoke_empty_refuses() -> None:
    """check_task_smoke refuses an empty result sequence."""
    with pytest.raises(ig.IntegrityRefusal):
        ig.check_task_smoke([])


def test_check_task_smoke_names_failed_side() -> None:
    """The refusal names which side (oracle/empty) failed for each broken task."""
    with pytest.raises(ig.IntegrityRefusal) as e:
        ig.check_task_smoke([ig.SmokeResult("tX", 0.0, 1.0)])
    msg = str(e.value)
    assert "oracle" in msg and "empty" in msg


def test_integrity_refusal_is_value_error() -> None:
    """IntegrityRefusal is a ValueError subclass."""
    assert issubclass(ig.IntegrityRefusal, ValueError)


def test_mask_decision_and_report_frozen() -> None:
    """MaskDecision and MaskReport are frozen dataclasses."""
    from dataclasses import FrozenInstanceError

    rep = ig.mask_failures([_episode("a", EpisodeStatus.OK, 1.0)])
    with pytest.raises(FrozenInstanceError):
        rep.total = 0  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        rep.decisions[0].reward = 0.0  # type: ignore[misc]


def test_audit_outcome_and_report_frozen() -> None:
    """AuditOutcome and AuditReport are frozen dataclasses."""
    from dataclasses import FrozenInstanceError

    rep = ig.run_fp_audit({"a": 1.0}, lambda i: 1.0, sample_rate=0.0, min_audited=1)
    with pytest.raises(FrozenInstanceError):
        rep.audited = 0  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        rep.outcomes[0].false_positive = True  # type: ignore[misc]


def test_smoke_result_frozen() -> None:
    """SmokeResult is a frozen dataclass."""
    from dataclasses import FrozenInstanceError

    r = ig.SmokeResult("t", 1.0, 0.0)
    with pytest.raises(FrozenInstanceError):
        r.oracle_reward = 0.0  # type: ignore[misc]


def test_stage_budget_frozen() -> None:
    """StageBudget is a frozen dataclass."""
    from dataclasses import FrozenInstanceError

    b = ig.StageBudget(task_timeout_s=1.0, max_timeout_s=1.0)
    with pytest.raises(FrozenInstanceError):
        b.grace_s = 0.0  # type: ignore[misc]


def test_ingest_policy_frozen() -> None:
    """IngestPolicy is a frozen dataclass."""
    from dataclasses import FrozenInstanceError

    p = ig.IngestPolicy()
    with pytest.raises(FrozenInstanceError):
        p.low = 1.0  # type: ignore[misc]


def test_mask_failures_decision_order_matches_input() -> None:
    """MaskReport.decisions preserve input order with one decision per episode."""
    eps = [_episode(f"e{i}", EpisodeStatus.OK, 1.0) for i in range(5)]
    rep = ig.mask_failures(eps)
    assert [d.episode_id for d in rep.decisions] == [f"e{i}" for i in range(5)]


def test_run_fp_audit_outcomes_order_is_deterministic() -> None:
    """Selected outcomes come in a deterministic order across runs."""
    passes = {f"ep-{i}": 1.0 for i in range(50)}
    r1 = ig.run_fp_audit(passes, lambda i: 1.0, sample_rate=0.3, min_audited=3)
    r2 = ig.run_fp_audit(passes, lambda i: 1.0, sample_rate=0.3, min_audited=3)
    assert [o.episode_id for o in r1.outcomes] == [o.episode_id for o in r2.outcomes]


def test_run_fp_audit_audited_fraction_exact() -> None:
    """audited_fraction is audited / verifier_passes with verifier_passes >= 1."""
    passes = {f"p{i}": 1.0 for i in range(10)}
    rep = ig.run_fp_audit(passes, lambda i: 1.0, sample_rate=0.0, min_audited=4)
    assert rep.audited == 4
    assert rep.audited_fraction == 0.4


def test_run_fp_audit_primary_reward_recorded() -> None:
    """Each AuditOutcome records the primary reward of its episode."""
    passes = {"a": 1.0, "b": 0.75}
    rep = ig.run_fp_audit(passes, lambda i: 1.0, sample_rate=0.0, min_audited=2, pass_threshold=0.5)
    assert {o.episode_id: o.primary_reward for o in rep.outcomes} == {"a": 1.0, "b": 0.75}


def test_check_stage_override_beats_task() -> None:
    """override_timeout_s takes precedence over task_timeout_s in effective_s."""
    b = ig.StageBudget(
        task_timeout_s=100.0, max_timeout_s=100.0, override_timeout_s=1.0, grace_s=0.0
    )
    with pytest.raises(ig.IntegrityRefusal):
        ig.check_stage("s", b, 1.5)


def test_mask_failures_non_infra_status_scored() -> None:
    """Non-infra statuses with a reward are scored, not masked."""
    ep = _episode("t", EpisodeStatus.TIMEOUT, 1.0)
    rep = ig.mask_failures([ep])
    assert rep.decisions[0].masked is False
    assert rep.decisions[0].reward == 1.0
    assert (rep.scored, rep.passed) == (1, 1)


def test_check_task_smoke_threshold_boundary() -> None:
    """Oracle exactly at threshold passes; empty exactly at threshold fails."""
    ig.check_task_smoke([ig.SmokeResult("t", 1.0, 0.5)], pass_threshold=1.0)
    with pytest.raises(ig.IntegrityRefusal):
        ig.check_task_smoke([ig.SmokeResult("t", 1.0, 1.0)], pass_threshold=1.0)


def test_run_fp_audit_forced_and_selected_union() -> None:
    """Forced ids are unioned with hash-selected ids without duplicates."""
    passes = {f"ep-{i}": 1.0 for i in range(30)}
    rep = ig.run_fp_audit(
        passes, lambda i: 1.0, sample_rate=0.5, min_audited=1, forced=["ep-0", "ep-1"]
    )
    ids = [o.episode_id for o in rep.outcomes]
    assert len(ids) == len(set(ids))
    assert {"ep-0", "ep-1"} <= set(ids)
    assert rep.audited == len(ids)


def test_ingest_reward_refuses_bad_policy_type() -> None:
    """ingest_reward refuses a policy that is not an IngestPolicy."""
    with pytest.raises(ig.IntegrityRefusal) as e:
        ig.ingest_reward(0.5, "policy")  # type: ignore[arg-type]
    assert "IngestPolicy" in str(e.value)


def test_ingest_policy_refuses_non_bool_snap_binary() -> None:
    """IngestPolicy.snap_binary must be a bool."""
    with pytest.raises(ig.IntegrityRefusal):
        ig.IngestPolicy(snap_binary=1)  # type: ignore[arg-type]


def test_mask_decision_refuses_bad_fields() -> None:
    """MaskDecision validates episode_id, status, reward, masked and reason."""
    from foundationscale.agentic_rl.trajectory_v2 import EpisodeStatus as ES

    cases = [
        dict(episode_id="", status=ES.OK, reward=None, masked=True, reason="r"),
        dict(episode_id="e", status="OK", reward=None, masked=True, reason="r"),
        dict(episode_id="e", status=ES.OK, reward=float("nan"), masked=False, reason=""),
        dict(episode_id="e", status=ES.OK, reward=None, masked=1, reason="r"),
        dict(episode_id="e", status=ES.OK, reward=None, masked=True, reason=3),
    ]
    for kwargs in cases:
        with pytest.raises(ig.IntegrityRefusal):
            ig.MaskDecision(**kwargs)  # type: ignore[arg-type]


def test_mask_decision_refuses_masked_with_reward() -> None:
    """A masked decision must have reward None."""
    from foundationscale.agentic_rl.trajectory_v2 import EpisodeStatus as ES

    with pytest.raises(ig.IntegrityRefusal):
        ig.MaskDecision(episode_id="e", status=ES.OK, reward=1.0, masked=True, reason="r")


def test_mask_decision_refuses_masked_empty_reason() -> None:
    """A masked decision must name a non-empty reason."""
    from foundationscale.agentic_rl.trajectory_v2 import EpisodeStatus as ES

    with pytest.raises(ig.IntegrityRefusal):
        ig.MaskDecision(episode_id="e", status=ES.OK, reward=None, masked=True, reason="")


def test_mask_decision_refuses_unmasked_reason() -> None:
    """An unmasked decision must have reason ''."""
    from foundationscale.agentic_rl.trajectory_v2 import EpisodeStatus as ES

    with pytest.raises(ig.IntegrityRefusal):
        ig.MaskDecision(episode_id="e", status=ES.OK, reward=1.0, masked=False, reason="why")


def test_mask_report_refuses_bad_decisions() -> None:
    """MaskReport.decisions must be a tuple of MaskDecision."""
    with pytest.raises(ig.IntegrityRefusal):
        ig.MaskReport(decisions=[], total=0, masked=0, scored=0, passed=0, pass_rate=None)  # type: ignore[arg-type]
    with pytest.raises(ig.IntegrityRefusal) as e:
        ig.MaskReport(decisions=("x",), total=1, masked=1, scored=0, passed=0, pass_rate=None)
    assert "1 of 1" in str(e.value)


def test_mask_report_refuses_bad_counts() -> None:
    """MaskReport count fields must be ints >= 0 and mutually consistent."""
    base = dict(decisions=(), total=0, masked=0, scored=0, passed=0, pass_rate=None)
    with pytest.raises(ig.IntegrityRefusal):
        ig.MaskReport(**{**base, "total": -1})
    with pytest.raises(ig.IntegrityRefusal):
        ig.MaskReport(**{**base, "masked": 1.0})  # type: ignore[arg-type]
    with pytest.raises(ig.IntegrityRefusal):
        ig.MaskReport(**{**base, "total": 2, "masked": 1, "scored": 0})
    with pytest.raises(ig.IntegrityRefusal):
        ig.MaskReport(**{**base, "total": 1, "scored": 1, "passed": 2})
    with pytest.raises(ig.IntegrityRefusal):
        ig.MaskReport(**{**base, "total": 1, "scored": 1, "passed": 0})


def test_mask_report_refuses_bad_pass_rate() -> None:
    """pass_rate must be None when scored == 0 and finite when scored > 0."""
    base = dict(decisions=(), total=0, masked=0, scored=0, passed=0)
    with pytest.raises(ig.IntegrityRefusal):
        ig.MaskReport(**base, pass_rate=0.0)
    with pytest.raises(ig.IntegrityRefusal):
        ig.MaskReport(**{**base, "scored": 1, "passed": 1}, pass_rate=None)
    with pytest.raises(ig.IntegrityRefusal):
        ig.MaskReport(**{**base, "scored": 1, "passed": 1}, pass_rate=float("nan"))


def test_mask_failures_refuses_bad_threshold_and_entries() -> None:
    """mask_failures validates pass_threshold and episode entries."""
    with pytest.raises(ig.IntegrityRefusal):
        ig.mask_failures([_episode("a")], pass_threshold=float("inf"))
    with pytest.raises(ig.IntegrityRefusal) as e:
        ig.mask_failures([_episode("a"), "x"])  # type: ignore[list-item]
    assert "1 of 2" in str(e.value)


def test_stage_budget_refuses_bad_override() -> None:
    """StageBudget.override_timeout_s must be None or finite > 0."""
    with pytest.raises(ig.IntegrityRefusal):
        ig.StageBudget(task_timeout_s=1.0, max_timeout_s=1.0, override_timeout_s=0.0)


def test_check_stage_refuses_bad_stage_and_budget() -> None:
    """check_stage validates the stage name and budget type."""
    b = ig.StageBudget(task_timeout_s=1.0, max_timeout_s=1.0)
    with pytest.raises(ig.IntegrityRefusal):
        ig.check_stage("", b, 0.0)
    with pytest.raises(ig.IntegrityRefusal) as e:
        ig.check_stage("s", "b", 0.0)  # type: ignore[arg-type]
    assert "StageBudget" in str(e.value)


def test_audit_selected_refuses_bad_id_and_salt() -> None:
    """audit_selected requires a non-empty episode_id and salt."""
    with pytest.raises(ig.IntegrityRefusal):
        ig.audit_selected("", sample_rate=0.5)
    with pytest.raises(ig.IntegrityRefusal):
        ig.audit_selected("e", sample_rate=0.5, salt="")


def test_audit_outcome_refuses_bad_fields() -> None:
    """AuditOutcome validates ids, rewards and the false_positive triple."""
    cases = [
        dict(episode_id="", primary_reward=1.0, audit_reward=None, false_positive=None),
        dict(episode_id="e", primary_reward=float("nan"), audit_reward=None, false_positive=None),
        dict(episode_id="e", primary_reward=1.0, audit_reward=float("inf"), false_positive=None),
        dict(episode_id="e", primary_reward=1.0, audit_reward=None, false_positive=1),
        dict(episode_id="e", primary_reward=1.0, audit_reward=None, false_positive=True),
        dict(episode_id="e", primary_reward=1.0, audit_reward=1.0, false_positive=None),
    ]
    for kwargs in cases:
        with pytest.raises(ig.IntegrityRefusal):
            ig.AuditOutcome(**kwargs)  # type: ignore[arg-type]


def test_audit_report_refuses_bad_outcomes_and_counts() -> None:
    """AuditReport validates outcomes type, counts and their consistency."""
    base = dict(
        outcomes=(),
        verifier_passes=1,
        audited=0,
        abstained=0,
        false_positives=0,
        audited_fraction=0.0,
        false_positive_rate=None,
    )
    with pytest.raises(ig.IntegrityRefusal):
        ig.AuditReport(**{**base, "outcomes": []})  # type: ignore[arg-type]
    with pytest.raises(ig.IntegrityRefusal) as e:
        ig.AuditReport(**{**base, "outcomes": ("x",), "audited": 1})
    assert "1 of 1" in str(e.value)
    with pytest.raises(ig.IntegrityRefusal):
        ig.AuditReport(**{**base, "verifier_passes": 0})
    with pytest.raises(ig.IntegrityRefusal) as e:
        ig.AuditReport(**{**base, "verifier_passes": 1, "audited": 2})
    with pytest.raises(ig.IntegrityRefusal) as e:
        ig.AuditReport(**{**base, "verifier_passes": 2, "audited": 1})
    with pytest.raises(ig.IntegrityRefusal) as e:
        ig.AuditReport(**{**base, "audited": 1, "abstained": 2})
    with pytest.raises(ig.IntegrityRefusal) as e:
        ig.AuditReport(**{**base, "audited": 1, "abstained": 1, "false_positives": 1})


def test_audit_report_refuses_bad_rates() -> None:
    """AuditReport validates audited_fraction and false_positive_rate."""
    base = dict(
        outcomes=(),
        verifier_passes=1,
        audited=0,
        abstained=0,
        false_positives=0,
    )
    with pytest.raises(ig.IntegrityRefusal):
        ig.AuditReport(**base, audited_fraction=float("nan"), false_positive_rate=None)
    with pytest.raises(ig.IntegrityRefusal):
        ig.AuditReport(**base, audited_fraction=0.0, false_positive_rate=0.0)
    with pytest.raises(ig.IntegrityRefusal):
        ig.AuditReport(
            **{**base, "audited": 1, "false_positives": 1},
            audited_fraction=1.0,
            false_positive_rate=None,
        )
    with pytest.raises(ig.IntegrityRefusal):
        ig.AuditReport(
            **{**base, "audited": 1, "false_positives": 1},
            audited_fraction=1.0,
            false_positive_rate=float("inf"),
        )


def test_run_fp_audit_refuses_bad_args() -> None:
    """run_fp_audit validates passes, audit, sample_rate, pass_threshold, min_audited, salt."""
    with pytest.raises(ig.IntegrityRefusal):
        ig.run_fp_audit([], lambda i: 1.0)  # type: ignore[arg-type]
    with pytest.raises(ig.IntegrityRefusal):
        ig.run_fp_audit({"a": 1.0}, "audit")  # type: ignore[arg-type]
    with pytest.raises(ig.IntegrityRefusal):
        ig.run_fp_audit({"a": 1.0}, lambda i: 1.0, sample_rate=2.0)
    with pytest.raises(ig.IntegrityRefusal):
        ig.run_fp_audit({"a": 1.0}, lambda i: 1.0, pass_threshold=float("nan"))
    with pytest.raises(ig.IntegrityRefusal):
        ig.run_fp_audit({"a": 1.0}, lambda i: 1.0, min_audited=-1)
    with pytest.raises(ig.IntegrityRefusal):
        ig.run_fp_audit({"a": 1.0}, lambda i: 1.0, salt="")


def test_run_fp_audit_refuses_bad_forced_entries() -> None:
    """run_fp_audit.forced entries must be non-empty strings."""
    with pytest.raises(ig.IntegrityRefusal) as e:
        ig.run_fp_audit({"a": 1.0}, lambda i: 1.0, forced=("", "b"))
    assert "1 of 2" in str(e.value)


def test_run_fp_audit_refuses_bad_pass_entries() -> None:
    """run_fp_audit.passes keys must be non-empty strings and values finite."""
    with pytest.raises(ig.IntegrityRefusal):
        ig.run_fp_audit({"": 1.0}, lambda i: 1.0)
    with pytest.raises(ig.IntegrityRefusal) as e:
        ig.run_fp_audit({"a": float("nan")}, lambda i: 1.0)
    assert "'a'" in str(e.value)


def test_run_fp_audit_refuses_bad_audit_reward() -> None:
    """run_fp_audit refuses a non-finite audit reward naming the episode id."""
    with pytest.raises(ig.IntegrityRefusal) as e:
        ig.run_fp_audit({"a": 1.0}, lambda i: float("nan"), sample_rate=0.0, min_audited=1)
    assert "'a'" in str(e.value)


def test_require_no_false_positives_refuses_bad_report() -> None:
    """require_no_false_positives requires an AuditReport."""
    with pytest.raises(ig.IntegrityRefusal) as e:
        ig.require_no_false_positives("report")  # type: ignore[arg-type]
    assert "AuditReport" in str(e.value)


def test_smoke_result_refuses_bad_fields() -> None:
    """SmokeResult validates task_id and both reward fields."""
    cases = [
        dict(task_id="", oracle_reward=1.0, empty_reward=0.0),
        dict(task_id="t", oracle_reward=float("nan"), empty_reward=0.0),
        dict(task_id="t", oracle_reward=1.0, empty_reward=float("inf")),
    ]
    for kwargs in cases:
        with pytest.raises(ig.IntegrityRefusal):
            ig.SmokeResult(**kwargs)  # type: ignore[arg-type]


def test_check_task_smoke_refuses_bad_threshold_and_entries() -> None:
    """check_task_smoke validates pass_threshold and result entries."""
    with pytest.raises(ig.IntegrityRefusal):
        ig.check_task_smoke([ig.SmokeResult("t", 1.0, 0.0)], pass_threshold=float("nan"))
    with pytest.raises(ig.IntegrityRefusal) as e:
        ig.check_task_smoke([ig.SmokeResult("t", 1.0, 0.0), "x"])  # type: ignore[list-item]
    assert "1 of 2" in str(e.value)


def test_ingest_reward_refuses_huge_int() -> None:
    """ingest_reward refuses an int that overflows float range."""
    with pytest.raises(ig.IntegrityRefusal):
        ig.ingest_reward(10**400)


def test_ingest_policy_refuses_huge_high() -> None:
    """IngestPolicy.high must be finite even for huge ints."""
    with pytest.raises(ig.IntegrityRefusal):
        ig.IngestPolicy(high=10**400)


def test_mask_report_refuses_masked_count_mismatch() -> None:
    """MaskReport refuses when masked disagrees with the masked decisions."""
    from foundationscale.agentic_rl.trajectory_v2 import EpisodeStatus as ES

    d = ig.MaskDecision(episode_id="e", status=ES.OK, reward=None, masked=True, reason="r")
    with pytest.raises(ig.IntegrityRefusal):
        ig.MaskReport(decisions=(d,), total=1, masked=0, scored=1, passed=0, pass_rate=None)


def test_mask_report_refuses_scored_count_mismatch() -> None:
    """MaskReport refuses when scored disagrees with the unmasked decisions."""
    from foundationscale.agentic_rl.trajectory_v2 import EpisodeStatus as ES

    d = ig.MaskDecision(episode_id="e", status=ES.OK, reward=1.0, masked=False, reason="")
    with pytest.raises(ig.IntegrityRefusal):
        ig.MaskReport(decisions=(d,), total=1, masked=1, scored=0, passed=0, pass_rate=None)


def test_mask_report_refuses_total_mismatch_with_decisions() -> None:
    """MaskReport refuses when total disagrees with len(decisions)."""
    from foundationscale.agentic_rl.trajectory_v2 import EpisodeStatus as ES

    d = ig.MaskDecision(episode_id="e", status=ES.OK, reward=1.0, masked=False, reason="")
    with pytest.raises(ig.IntegrityRefusal):
        ig.MaskReport(decisions=(d,), total=2, masked=0, scored=2, passed=0, pass_rate=0.0)


def test_mask_report_refuses_masked_plus_scored_mismatch() -> None:
    """MaskReport refuses when masked + scored != total despite per-decision counts."""
    from foundationscale.agentic_rl.trajectory_v2 import EpisodeStatus as ES

    d = ig.MaskDecision(episode_id="e", status=ES.OK, reward=1.0, masked=False, reason="")
    with pytest.raises(ig.IntegrityRefusal):
        ig.MaskReport(decisions=(d,), total=1, masked=0, scored=1, passed=0, pass_rate=0.0)


def test_audit_report_refuses_abstained_count_mismatch() -> None:
    """AuditReport refuses when abstained disagrees with outcomes."""
    o = ig.AuditOutcome(episode_id="e", primary_reward=1.0, audit_reward=None, false_positive=None)
    with pytest.raises(ig.IntegrityRefusal):
        ig.AuditReport(
            outcomes=(o,),
            verifier_passes=1,
            audited=1,
            abstained=0,
            false_positives=0,
            audited_fraction=1.0,
            false_positive_rate=0.0,
        )


def test_audit_report_refuses_false_positive_count_mismatch() -> None:
    """AuditReport refuses when false_positives disagrees with outcomes."""
    o = ig.AuditOutcome(episode_id="e", primary_reward=1.0, audit_reward=0.0, false_positive=True)
    with pytest.raises(ig.IntegrityRefusal):
        ig.AuditReport(
            outcomes=(o,),
            verifier_passes=1,
            audited=1,
            abstained=0,
            false_positives=0,
            audited_fraction=1.0,
            false_positive_rate=0.0,
        )


def test_audit_report_refuses_audited_count_mismatch_with_outcomes() -> None:
    """AuditReport refuses when audited disagrees with len(outcomes)."""
    o = ig.AuditOutcome(episode_id="e", primary_reward=1.0, audit_reward=1.0, false_positive=False)
    with pytest.raises(ig.IntegrityRefusal):
        ig.AuditReport(
            outcomes=(o,),
            verifier_passes=1,
            audited=2,
            abstained=0,
            false_positives=0,
            audited_fraction=1.0,
            false_positive_rate=0.0,
        )


def test_audit_outcome_refuses_fp_with_abstained_audit_reward() -> None:
    """AuditOutcome refuses false_positive set when audit_reward is None."""
    with pytest.raises(ig.IntegrityRefusal):
        ig.AuditOutcome(episode_id="e", primary_reward=1.0, audit_reward=None, false_positive=False)


def test_require_no_false_positives_on_valid_report_from_outcomes() -> None:
    """require_no_false_positives passes on a valid report built from outcomes."""
    outcomes = (
        ig.AuditOutcome(episode_id="a", primary_reward=1.0, audit_reward=1.0, false_positive=False),
        ig.AuditOutcome(episode_id="b", primary_reward=1.0, audit_reward=None, false_positive=None),
    )
    report = ig.AuditReport(
        outcomes=outcomes,
        verifier_passes=2,
        audited=2,
        abstained=1,
        false_positives=0,
        audited_fraction=1.0,
        false_positive_rate=0.0,
    )
    ig.require_no_false_positives(report)


def test_mask_report_refuses_pass_rate_mismatch_with_counts() -> None:
    """MaskReport refuses a pass_rate that disagrees with passed/scored."""
    from foundationscale.agentic_rl.trajectory_v2 import EpisodeStatus as ES

    d = ig.MaskDecision(episode_id="e", status=ES.OK, reward=1.0, masked=False, reason="")
    with pytest.raises(ig.IntegrityRefusal):
        ig.MaskReport(decisions=(d,), total=1, masked=0, scored=1, passed=1, pass_rate=0.5)


def test_audit_report_refuses_audited_fraction_mismatch_with_counts() -> None:
    """AuditReport refuses an audited_fraction that disagrees with audited/verifier_passes."""
    o = ig.AuditOutcome(episode_id="e", primary_reward=1.0, audit_reward=1.0, false_positive=False)
    with pytest.raises(ig.IntegrityRefusal):
        ig.AuditReport(
            outcomes=(o,),
            verifier_passes=2,
            audited=1,
            abstained=0,
            false_positives=0,
            audited_fraction=1.0,
            false_positive_rate=0.0,
        )


def test_audit_report_refuses_false_positive_rate_mismatch_with_counts() -> None:
    """AuditReport refuses a false_positive_rate that disagrees with fp/decided."""
    o = ig.AuditOutcome(episode_id="e", primary_reward=1.0, audit_reward=0.0, false_positive=True)
    with pytest.raises(ig.IntegrityRefusal):
        ig.AuditReport(
            outcomes=(o,),
            verifier_passes=1,
            audited=1,
            abstained=0,
            false_positives=1,
            audited_fraction=1.0,
            false_positive_rate=0.0,
        )
