"""Adversarial tests for ``foundationscale.gates.agentic_gates``.

Why this suite exists
----------------------
These five gates are the lifecycle-boundary re-check of invariants
``agentic_rl.contracts`` already enforces inside its own constructors
(``Trajectory``, ``flatten``) and ``rl.weightsync`` enforces inside
``SyncReport`` -- the SEAM, not the constructor, is where a defect built by
some OTHER adapter would otherwise slip through uncaught. The tests here try
every way to buy a PASS or a silent skip out of a defective input the way the
checkpoint-plane incidents actually happened: a loss mask entry outside
``{0, 1}``, a per-token column that runs its own length, supervision on
non-assistant/tool-call text, an INFRA row carrying a live mask, a
reward/abstention pair that contradicts itself, a 100%-infra batch, a
look-alike-zero ``offered`` count, an unmeasured or measured-stale weight
sync, an unmeasured rollout policy version, excess lag, mismatched or
over-tolerance parity pairs, and a prompt-bytes mismatch. Every "X is
rejected" test sits next to a positive control proving the detector could
have fired, and every VACUOUS/SKIP path is pinned by its
:class:`~foundationscale.gates.core.AbstentionKind` (or lack of one), never by
prose alone.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from foundationscale.gates.agentic_gates import (
    PromptBytesGate,
    PromptBytesGateContext,
    RolloutAbstentionGate,
    RolloutGateContext,
    StalenessGate,
    TrajectoryIntegrityGate,
    WeightSyncGateContext,
    WeightSyncParityGate,
    prompt_bytes_sha256,
)
from foundationscale.gates.core import (
    REGISTRY,
    AbstentionKind,
    Control,
    ControlKind,
    GateRegistry,
    Lifecycle,
    Verdict,
    verify_controls,
)

if TYPE_CHECKING:
    from foundationscale.gates.core import Gate


# ---------------------------------------------------------------------------
# Local context builders -- deliberately independent of the module's own
# private fixtures, so this suite cannot rot in lockstep with them.
# ---------------------------------------------------------------------------


def _row(
    *,
    response_ids: tuple[int, ...],
    loss_mask: tuple[int, ...],
    segment_kind: tuple[str, ...],
    termination: str = "stop",
    reward: float | None = 1.0,
    abstention_reason: str | None = None,
) -> dict[str, Any]:
    n = len(response_ids)
    return {
        "response_ids": response_ids,
        "loss_mask": loss_mask,
        "segment_kind": segment_kind,
        "termination": termination,
        "reward": reward,
        "abstention_reason": abstention_reason,
        "rollout_logprobs": (None,) * n,
        "turn_index": (0,) * n,
        "tool_call_error_mask": (0,) * n,
        "repetition_mask": (0,) * n,
    }


_BATCH_COLUMNS = (
    "response_ids",
    "loss_mask",
    "rollout_logprobs",
    "turn_index",
    "segment_kind",
    "tool_call_error_mask",
    "repetition_mask",
    "reward",
    "abstention_reason",
    "termination",
)


def _batch(rows: list[dict[str, Any]]) -> dict[str, tuple[Any, ...]]:
    return {name: tuple(row[name] for row in rows) for name in _BATCH_COLUMNS}


def _rollout_ctx(
    rows: list[dict[str, Any]], *, declared_max_infra_rate: float | None = None, step: int = 0
) -> RolloutGateContext:
    return RolloutGateContext(
        batch_columns=_batch(rows), declared_max_infra_rate=declared_max_infra_rate, step=step
    )


def _healthy_rows() -> list[dict[str, Any]]:
    return [
        _row(
            response_ids=(1, 2, 3),
            loss_mask=(0, 1, 1),
            segment_kind=("system", "assistant", "assistant"),
            reward=2.0,
        ),
        _row(
            response_ids=(4, 5),
            loss_mask=(0, 0),
            segment_kind=("assistant", "assistant"),
            termination="infra",
            reward=None,
            abstention_reason="infra:EnvInfraError",
        ),
    ]


def _sync_ctx(
    *,
    offered: int = 2,
    failed: int = 0,
    is_stale: bool | None = False,
    seconds: float | None = 1.5,
    step: int = 0,
    policy_version: int = 3,
    rollout_policy_versions: tuple[int | None, ...] = (2, 3),
    declared_max_lag: int = 1,
    parity: tuple[tuple[tuple[float, ...], tuple[float, ...]], ...] | None = None,
    declared_parity_atol: float | None = None,
) -> WeightSyncGateContext:
    return WeightSyncGateContext(
        offered=offered,
        failed=failed,
        is_stale=is_stale,
        seconds=seconds,
        step=step,
        policy_version=policy_version,
        rollout_policy_versions=rollout_policy_versions,
        declared_max_lag=declared_max_lag,
        parity=parity,
        declared_parity_atol=declared_parity_atol,
    )


def _control_by_name(gate: Gate, name: str) -> Control:
    return next(c for c in gate.controls() if c.name == name)


@pytest.fixture
def trajectory_gate() -> TrajectoryIntegrityGate:
    return TrajectoryIntegrityGate()


@pytest.fixture
def abstention_gate() -> RolloutAbstentionGate:
    return RolloutAbstentionGate()


@pytest.fixture
def staleness_gate() -> StalenessGate:
    return StalenessGate()


@pytest.fixture
def parity_gate() -> WeightSyncParityGate:
    return WeightSyncParityGate()


@pytest.fixture
def prompt_gate() -> PromptBytesGate:
    return PromptBytesGate()


# ---------------------------------------------------------------------------
# 1. TrajectoryIntegrityGate
# ---------------------------------------------------------------------------


class TestTrajectoryIntegrityGate:
    def test_healthy_two_row_batch_passes_with_full_row_coverage(
        self, trajectory_gate: TrajectoryIntegrityGate
    ) -> None:
        result = trajectory_gate.run(_rollout_ctx(_healthy_rows()))
        assert result.verdict is Verdict.PASS
        assert not result.blocking
        assert result.coverage.checked == 2
        assert result.coverage.expected == 2
        assert result.coverage.unit == "rows"

    def test_empty_batch_is_vacuous_never_pass(
        self, trajectory_gate: TrajectoryIntegrityGate
    ) -> None:
        result = trajectory_gate.run(_rollout_ctx([]))
        assert result.verdict is Verdict.VACUOUS
        assert result.blocking
        assert result.coverage.checked == 0

    def test_missing_required_column_fails_naming_it(
        self, trajectory_gate: TrajectoryIntegrityGate
    ) -> None:
        ctx = _rollout_ctx(_healthy_rows())
        cols = dict(ctx.batch_columns)
        del cols["turn_index"]
        result = trajectory_gate.run(
            RolloutGateContext(batch_columns=cols, declared_max_infra_rate=None, step=0)
        )
        assert result.verdict is Verdict.FAIL
        assert "turn_index" in result.detail

    def test_per_token_length_mismatch_fails_naming_both_counts(
        self, trajectory_gate: TrajectoryIntegrityGate
    ) -> None:
        ctx = _rollout_ctx(
            [_row(response_ids=(1, 2), loss_mask=(0, 1), segment_kind=("system", "assistant"))]
        )
        cols = dict(ctx.batch_columns)
        cols["loss_mask"] = ((0,),)  # one entry for a two-token response
        result = trajectory_gate.run(
            RolloutGateContext(batch_columns=cols, declared_max_infra_rate=None, step=0)
        )
        assert result.verdict is Verdict.FAIL
        assert "row 0" in result.detail
        assert "'loss_mask'" in result.detail
        assert "1 token entries" in result.detail
        assert "'response_ids' has 2" in result.detail

    def test_loss_mask_value_outside_0_1_fails(
        self, trajectory_gate: TrajectoryIntegrityGate
    ) -> None:
        ctx = _rollout_ctx(
            [_row(response_ids=(1, 2), loss_mask=(0, 2), segment_kind=("system", "assistant"))]
        )
        result = trajectory_gate.run(ctx)
        assert result.verdict is Verdict.FAIL
        assert "loss_mask entry 2" in result.detail

    def test_loss_mask_bool_entry_fails_one_is_not_true(
        self, trajectory_gate: TrajectoryIntegrityGate
    ) -> None:
        ctx = _rollout_ctx(
            [_row(response_ids=(1,), loss_mask=(True,), segment_kind=("assistant",))]
        )
        result = trajectory_gate.run(ctx)
        assert result.verdict is Verdict.FAIL
        assert "1 is not True" in result.detail

    def test_loss_mask_one_on_non_supervisable_segment_fails(
        self, trajectory_gate: TrajectoryIntegrityGate
    ) -> None:
        ctx = _rollout_ctx([_row(response_ids=(1,), loss_mask=(1,), segment_kind=("tool_result",))])
        result = trajectory_gate.run(ctx)
        assert result.verdict is Verdict.FAIL
        assert "tool_result" in result.detail

    def test_infra_row_with_nonzero_loss_mask_fails(
        self, trajectory_gate: TrajectoryIntegrityGate
    ) -> None:
        ctx = _rollout_ctx(
            [
                _row(
                    response_ids=(1, 2),
                    loss_mask=(0, 1),
                    segment_kind=("assistant", "assistant"),
                    termination="infra",
                    reward=None,
                    abstention_reason="infra:EnvInfraError",
                )
            ]
        )
        result = trajectory_gate.run(ctx)
        assert result.verdict is Verdict.FAIL
        assert "infra" in result.detail

    @pytest.mark.parametrize(
        ("reward", "abstention_reason"),
        [(None, None), (1.0, "measured but also abstained")],
    )
    def test_reward_abstention_biconditional_violation_fails(
        self,
        trajectory_gate: TrajectoryIntegrityGate,
        reward: float | None,
        abstention_reason: str | None,
    ) -> None:
        ctx = _rollout_ctx(
            [
                _row(
                    response_ids=(1,),
                    loss_mask=(1,),
                    segment_kind=("assistant",),
                    reward=reward,
                    abstention_reason=abstention_reason,
                )
            ]
        )
        result = trajectory_gate.run(ctx)
        assert result.verdict is Verdict.FAIL
        assert "biconditional" in result.detail

    def test_first_offending_row_is_named_not_a_later_one(
        self, trajectory_gate: TrajectoryIntegrityGate
    ) -> None:
        rows = [
            _row(response_ids=(1,), loss_mask=(2,), segment_kind=("assistant",)),  # row 0: bad
            _row(response_ids=(1,), loss_mask=(2,), segment_kind=("assistant",)),  # row 1: also bad
        ]
        result = trajectory_gate.run(_rollout_ctx(rows))
        assert result.verdict is Verdict.FAIL
        assert "row 0" in result.detail
        assert "row 1" not in result.detail


# ---------------------------------------------------------------------------
# 2. RolloutAbstentionGate
# ---------------------------------------------------------------------------


class TestRolloutAbstentionGate:
    def test_healthy_rate_within_bound_passes(self, abstention_gate: RolloutAbstentionGate) -> None:
        rows = [
            _row(response_ids=(1,), loss_mask=(1,), segment_kind=("assistant",), reward=1.0),
            _row(
                response_ids=(1,),
                loss_mask=(0,),
                segment_kind=("assistant",),
                termination="infra",
                reward=None,
                abstention_reason="infra:transport",
            ),
            _row(response_ids=(1,), loss_mask=(1,), segment_kind=("assistant",), reward=1.0),
            _row(response_ids=(1,), loss_mask=(1,), segment_kind=("assistant",), reward=1.0),
        ]
        result = abstention_gate.run(_rollout_ctx(rows, declared_max_infra_rate=0.5))
        assert result.verdict is Verdict.PASS
        assert result.coverage.checked == 4

    def test_rate_exceeding_bound_fails_naming_both_counts(
        self, abstention_gate: RolloutAbstentionGate
    ) -> None:
        rows = [
            _row(
                response_ids=(1,),
                loss_mask=(0,),
                segment_kind=("assistant",),
                termination="infra",
                reward=None,
                abstention_reason="infra:transport",
            ),
            _row(
                response_ids=(1,),
                loss_mask=(0,),
                segment_kind=("assistant",),
                termination="infra",
                reward=None,
                abstention_reason="infra:transport",
            ),
            _row(response_ids=(1,), loss_mask=(1,), segment_kind=("assistant",), reward=1.0),
        ]
        result = abstention_gate.run(_rollout_ctx(rows, declared_max_infra_rate=0.1))
        assert result.verdict is Verdict.FAIL
        assert result.blocking
        assert "2/3" in result.detail
        assert "declared_max_infra_rate=0.1" in result.detail

    def test_hundred_percent_infra_under_a_generous_bound_passes_not_vacuous(
        self, abstention_gate: RolloutAbstentionGate
    ) -> None:
        # The subtlety this gate deliberately does NOT special-case: a 100%
        # infra rate is measured and compared like any other value, never
        # treated as its own vacuity.
        rows = [
            _row(
                response_ids=(1,),
                loss_mask=(0,),
                segment_kind=("assistant",),
                termination="infra",
                reward=None,
                abstention_reason="infra:a",
            ),
            _row(
                response_ids=(1,),
                loss_mask=(0,),
                segment_kind=("assistant",),
                termination="infra",
                reward=None,
                abstention_reason="infra:b",
            ),
        ]
        result = abstention_gate.run(_rollout_ctx(rows, declared_max_infra_rate=1.0))
        assert result.verdict is Verdict.PASS

    def test_empty_batch_is_vacuous(self, abstention_gate: RolloutAbstentionGate) -> None:
        result = abstention_gate.run(_rollout_ctx([], declared_max_infra_rate=0.5))
        assert result.verdict is Verdict.VACUOUS
        assert result.blocking

    def test_undeclared_bound_abstains_not_established(
        self, abstention_gate: RolloutAbstentionGate
    ) -> None:
        rows = [
            _row(
                response_ids=(1,),
                loss_mask=(0,),
                segment_kind=("assistant",),
                termination="infra",
                reward=None,
                abstention_reason="infra:transport",
            ),
        ]
        result = abstention_gate.run(_rollout_ctx(rows, declared_max_infra_rate=None))
        assert result.verdict is Verdict.SKIP
        assert not result.blocking
        assert result.abstention is AbstentionKind.NOT_ESTABLISHED


# ---------------------------------------------------------------------------
# 3. StalenessGate
# ---------------------------------------------------------------------------


class TestStalenessGate:
    def test_healthy_push_within_lag_passes(self, staleness_gate: StalenessGate) -> None:
        result = staleness_gate.run(_sync_ctx())
        assert result.verdict is Verdict.PASS
        assert result.coverage.checked == 1 + 2  # 1 (is_stale) + 2 rollout versions

    def test_malformed_offered_bool_is_vacuous_never_buys_not_applicable(
        self, staleness_gate: StalenessGate
    ) -> None:
        ctx = WeightSyncGateContext(
            offered=False,  # type: ignore[arg-type]
            failed=0,
            is_stale=None,
            seconds=None,
            step=0,
            policy_version=1,
            rollout_policy_versions=(),
            declared_max_lag=1,
        )
        result = staleness_gate.run(ctx)
        assert result.verdict is Verdict.VACUOUS
        assert result.blocking
        assert result.abstention is None

    def test_no_sync_offered_abstains_not_applicable(self, staleness_gate: StalenessGate) -> None:
        result = staleness_gate.run(
            _sync_ctx(offered=0, is_stale=None, seconds=None, rollout_policy_versions=())
        )
        assert result.verdict is Verdict.SKIP
        assert not result.blocking
        assert result.abstention is AbstentionKind.NOT_APPLICABLE

    def test_unmeasured_staleness_fails(self, staleness_gate: StalenessGate) -> None:
        result = staleness_gate.run(_sync_ctx(is_stale=None))
        assert result.verdict is Verdict.FAIL
        assert "never measured" in result.detail

    def test_measured_stale_true_fails(self, staleness_gate: StalenessGate) -> None:
        result = staleness_gate.run(_sync_ctx(is_stale=True))
        assert result.verdict is Verdict.FAIL
        assert "is_stale=True" in result.detail

    def test_unmeasured_rollout_version_fails(self, staleness_gate: StalenessGate) -> None:
        result = staleness_gate.run(_sync_ctx(policy_version=3, rollout_policy_versions=(3, None)))
        assert result.verdict is Verdict.FAIL
        assert "rollout session 1" in result.detail
        assert "unmeasured (None)" in result.detail

    def test_lag_exceeded_fails_naming_both_counts(self, staleness_gate: StalenessGate) -> None:
        result = staleness_gate.run(
            _sync_ctx(policy_version=5, rollout_policy_versions=(1,), declared_max_lag=1)
        )
        assert result.verdict is Verdict.FAIL
        assert "trails policy_version=5 by 4" in result.detail
        assert "declared_max_lag=1" in result.detail

    def test_empty_rollout_versions_with_healthy_is_stale_passes_not_skip(
        self, staleness_gate: StalenessGate
    ) -> None:
        # The fix pinned here: a WEIGHT_SYNC sweep where StalenessGate also
        # abstained (alongside WeightSyncParityGate, which abstains until a
        # GPU probe measures parity) would leave the whole event with zero
        # PASS/FAIL results -- `GateReport.is_unverified` would then block
        # every real publish() call forever. Coverage=1 keeps this a real,
        # scoped PASS.
        result = staleness_gate.run(_sync_ctx(rollout_policy_versions=()))
        assert result.verdict is Verdict.PASS
        assert result.coverage.checked == 1


# ---------------------------------------------------------------------------
# 4. WeightSyncParityGate
# ---------------------------------------------------------------------------


class TestWeightSyncParityGate:
    def test_healthy_parity_within_atol_passes(self, parity_gate: WeightSyncParityGate) -> None:
        ctx = _sync_ctx(
            parity=(((0.0, -1.0), (0.01, -1.02)),),
            declared_parity_atol=0.05,
        )
        result = parity_gate.run(ctx)
        assert result.verdict is Verdict.PASS
        assert result.coverage.checked == 2

    def test_diff_exceeding_atol_fails_naming_the_worst_token(
        self, parity_gate: WeightSyncParityGate
    ) -> None:
        ctx = _sync_ctx(
            parity=(((0.0, -1.0), (0.0, -1.5)),),
            declared_parity_atol=0.1,
        )
        result = parity_gate.run(ctx)
        assert result.verdict is Verdict.FAIL
        assert "row 0 token 1" in result.detail
        assert "declared_parity_atol=0.1" in result.detail

    def test_length_mismatch_fails_naming_both_lengths(
        self, parity_gate: WeightSyncParityGate
    ) -> None:
        ctx = _sync_ctx(parity=(((0.0, -1.0), (0.0,)),), declared_parity_atol=0.1)
        result = parity_gate.run(ctx)
        assert result.verdict is Verdict.FAIL
        assert "2 entries" in result.detail
        assert "1 entries" in result.detail

    def test_empty_pairs_is_vacuous(self, parity_gate: WeightSyncParityGate) -> None:
        ctx = _sync_ctx(parity=(), declared_parity_atol=0.1)
        result = parity_gate.run(ctx)
        assert result.verdict is Verdict.VACUOUS
        assert result.blocking

    def test_parity_none_abstains_not_established(self, parity_gate: WeightSyncParityGate) -> None:
        ctx = _sync_ctx(parity=None, declared_parity_atol=None)
        result = parity_gate.run(ctx)
        assert result.verdict is Verdict.SKIP
        assert result.abstention is AbstentionKind.NOT_ESTABLISHED

    def test_undeclared_atol_with_measured_parity_abstains_not_established(
        self, parity_gate: WeightSyncParityGate
    ) -> None:
        ctx = _sync_ctx(parity=(((0.0,), (0.0,)),), declared_parity_atol=None)
        result = parity_gate.run(ctx)
        assert result.verdict is Verdict.SKIP
        assert result.abstention is AbstentionKind.NOT_ESTABLISHED


# ---------------------------------------------------------------------------
# 5. PromptBytesGate
# ---------------------------------------------------------------------------


class TestPromptBytesSha256:
    def test_stable_across_calls(self) -> None:
        messages = ({"role": "system", "content": "be helpful"},)
        tools = ({"name": "bash"},)
        assert prompt_bytes_sha256(messages, tools) == prompt_bytes_sha256(messages, tools)

    def test_insensitive_to_dict_key_order(self) -> None:
        a = ({"role": "system", "content": "x"},)
        forward = prompt_bytes_sha256(a, ({"name": "bash", "parameters": {}},))
        backward = prompt_bytes_sha256(a, ({"parameters": {}, "name": "bash"},))
        assert forward == backward

    def test_any_content_change_moves_the_digest(self) -> None:
        base = prompt_bytes_sha256(({"role": "system", "content": "x"},), ())
        changed = prompt_bytes_sha256(({"role": "system", "content": "y"},), ())
        assert base != changed

    def test_is_a_sha256_hex_digest(self) -> None:
        digest = prompt_bytes_sha256(({"role": "system", "content": "x"},), ())
        assert len(digest) == 64
        assert all(c in "0123456789abcdef" for c in digest)


class TestPromptBytesGate:
    def test_matching_digests_pass(self, prompt_gate: PromptBytesGate) -> None:
        digest = prompt_bytes_sha256(({"role": "system", "content": "x"},), ())
        result = prompt_gate.run(
            PromptBytesGateContext(observed_sha256=digest, declared_sha256=digest)
        )
        assert result.verdict is Verdict.PASS
        assert result.coverage.checked == 1

    def test_mismatch_fails_naming_both_digests(self, prompt_gate: PromptBytesGate) -> None:
        result = prompt_gate.run(
            PromptBytesGateContext(observed_sha256="a" * 64, declared_sha256="b" * 64)
        )
        assert result.verdict is Verdict.FAIL
        assert "a" * 64 in result.detail
        assert "b" * 64 in result.detail

    def test_unmeasured_observation_beside_a_declaration_fails(
        self, prompt_gate: PromptBytesGate
    ) -> None:
        result = prompt_gate.run(
            PromptBytesGateContext(observed_sha256=None, declared_sha256="a" * 64)
        )
        assert result.verdict is Verdict.FAIL
        assert "never computed" in result.detail

    def test_undeclared_sha256_abstains_not_established(self, prompt_gate: PromptBytesGate) -> None:
        result = prompt_gate.run(
            PromptBytesGateContext(observed_sha256="a" * 64, declared_sha256=None)
        )
        assert result.verdict is Verdict.SKIP
        assert not result.blocking
        assert result.abstention is AbstentionKind.NOT_ESTABLISHED


# ---------------------------------------------------------------------------
# Lifecycle extension (D1): append-only, every existing member unchanged.
# ---------------------------------------------------------------------------


class TestLifecycleExtension:
    def test_new_members_have_the_declared_values(self) -> None:
        assert Lifecycle.ROLLOUT.value == "rollout"
        assert Lifecycle.EPISODE.value == "episode"
        assert Lifecycle.WEIGHT_SYNC.value == "weight_sync"

    def test_every_pre_existing_member_is_unchanged(self) -> None:
        # Pinned so a future edit to this enum cannot silently rename or drop
        # one of the original eight members while adding the three new ones.
        assert Lifecycle.LAUNCH.value == "launch"
        assert Lifecycle.BUILD.value == "build"
        assert Lifecycle.DATA.value == "data"
        assert Lifecycle.STEP_ZERO.value == "step_zero"
        assert Lifecycle.FIRST_SAVE.value == "first_save"
        assert Lifecycle.SAVE.value == "save"
        assert Lifecycle.EXPORT.value == "export"
        assert Lifecycle.PROMOTE.value == "promote"

    def test_full_membership_is_exactly_eleven_in_declared_order(self) -> None:
        assert [member.value for member in Lifecycle] == [
            "launch",
            "build",
            "data",
            "step_zero",
            "first_save",
            "save",
            "export",
            "promote",
            "rollout",
            "episode",
            "weight_sync",
        ]


# ---------------------------------------------------------------------------
# Controls and registration
# ---------------------------------------------------------------------------


class TestControlsAndRegistration:
    _IDS = {
        "agentic_rl.trajectory_integrity",
        "agentic_rl.rollout_abstention",
        "agentic_rl.weight_sync_staleness",
        "agentic_rl.weight_sync_parity",
        "agentic_rl.prompt_bytes",
    }

    def test_all_five_gates_are_registered_in_the_global_registry(self) -> None:
        registered = {gate.id for gate in REGISTRY}
        assert registered >= self._IDS

    def test_each_gate_runs_at_its_declared_lifecycle_event(self) -> None:
        rollout_ids = {g.id for g in REGISTRY.for_event(Lifecycle.ROLLOUT)}
        weight_sync_ids = {g.id for g in REGISTRY.for_event(Lifecycle.WEIGHT_SYNC)}
        assert rollout_ids >= {
            "agentic_rl.trajectory_integrity",
            "agentic_rl.rollout_abstention",
            "agentic_rl.prompt_bytes",
        }
        assert weight_sync_ids >= {
            "agentic_rl.weight_sync_staleness",
            "agentic_rl.weight_sync_parity",
        }

    def test_every_gate_declares_both_control_kinds(self) -> None:
        for gate in (
            TrajectoryIntegrityGate(),
            RolloutAbstentionGate(),
            StalenessGate(),
            WeightSyncParityGate(),
            PromptBytesGate(),
        ):
            kinds = {c.kind for c in gate.controls()}
            assert ControlKind.MUST_FIRE in kinds, gate.id
            assert ControlKind.MUST_PASS in kinds, gate.id

    def test_all_module_controls_hold_under_verify_controls(self) -> None:
        registry = GateRegistry()
        registry.register(TrajectoryIntegrityGate())
        registry.register(RolloutAbstentionGate())
        registry.register(StalenessGate())
        registry.register(WeightSyncParityGate())
        registry.register(PromptBytesGate())
        assert verify_controls(registry) == []

    @pytest.mark.parametrize(
        ("gate", "control_name", "expected"),
        [
            (TrajectoryIntegrityGate(), "missing-column", Verdict.FAIL),
            (TrajectoryIntegrityGate(), "per-token-length-mismatch", Verdict.FAIL),
            (TrajectoryIntegrityGate(), "loss-mask-out-of-range", Verdict.FAIL),
            (TrajectoryIntegrityGate(), "loss-mask-bool-entry", Verdict.FAIL),
            (TrajectoryIntegrityGate(), "loss-mask-wrong-segment", Verdict.FAIL),
            (TrajectoryIntegrityGate(), "infra-row-nonzero-loss-mask", Verdict.FAIL),
            (TrajectoryIntegrityGate(), "reward-abstention-mismatch", Verdict.FAIL),
            (TrajectoryIntegrityGate(), "empty-batch", Verdict.VACUOUS),
            (TrajectoryIntegrityGate(), "healthy-batch", Verdict.PASS),
            (RolloutAbstentionGate(), "rate-exceeds-bound", Verdict.FAIL),
            (RolloutAbstentionGate(), "empty-batch", Verdict.VACUOUS),
            (RolloutAbstentionGate(), "healthy", Verdict.PASS),
            (RolloutAbstentionGate(), "all-infra-under-generous-bound", Verdict.PASS),
            (RolloutAbstentionGate(), "undeclared-bound", Verdict.SKIP),
            (StalenessGate(), "malformed-offered-bool", Verdict.VACUOUS),
            (StalenessGate(), "unmeasured-staleness", Verdict.FAIL),
            (StalenessGate(), "measured-stale", Verdict.FAIL),
            (StalenessGate(), "unmeasured-rollout-version", Verdict.FAIL),
            (StalenessGate(), "lag-exceeded", Verdict.FAIL),
            (StalenessGate(), "healthy", Verdict.PASS),
            (StalenessGate(), "no-sync-offered", Verdict.SKIP),
            (StalenessGate(), "empty-rollout-versions", Verdict.PASS),
            (WeightSyncParityGate(), "exceeds-atol", Verdict.FAIL),
            (WeightSyncParityGate(), "length-mismatch", Verdict.FAIL),
            (WeightSyncParityGate(), "empty-pairs", Verdict.VACUOUS),
            (WeightSyncParityGate(), "healthy", Verdict.PASS),
            (WeightSyncParityGate(), "parity-unmeasured", Verdict.SKIP),
            (WeightSyncParityGate(), "atol-undeclared", Verdict.SKIP),
            (PromptBytesGate(), "mismatch", Verdict.FAIL),
            (PromptBytesGate(), "unmeasured-observation", Verdict.FAIL),
            (PromptBytesGate(), "match", Verdict.PASS),
            (PromptBytesGate(), "undeclared", Verdict.SKIP),
        ],
    )
    def test_control_produces_its_exact_verdict(
        self, gate: Gate, control_name: str, expected: Verdict
    ) -> None:
        control = _control_by_name(gate, control_name)
        result = gate.run(control.make_ctx())
        assert result.verdict is expected
