"""Agentic RL gates: five detectors over a rollout batch and a weight-sync push.

Why this module exists
-----------------------
The agentic RL plane (``foundationscale.agentic_rl``) builds its own correctness
discipline directly into its data types: ``contracts.Trajectory`` refuses a
reward/abstention mismatch at construction, ``contracts.flatten`` refuses a
per-token column whose length disagrees with its row's ``response_ids``, and
``weight_sync.SyncReport`` refuses a non-total transfer record. Those are SEAM
checks on the objects that cross from one module into another; this module is
the GATE layer on top of them, run at the lifecycle events this slice adds to
:class:`~foundationscale.gates.core.Lifecycle` (``ROLLOUT``, ``WEIGHT_SYNC``) —
late, declared, controlled checkpoints that a training loop can refuse to pass,
exactly like every other gate in this package. A seam check that only a
dataclass's own author remembers to call is the same "copy-pasted into one
script, absent from the other" failure the checkpoint gates were built to
remove; a registered gate run through :func:`~foundationscale.gates.core.run_event`
cannot be forgotten by a caller the way a constructor call can.

Every gate here follows the pattern in :mod:`foundationscale.gates.example`
and :mod:`foundationscale.gates.objective_gates`, read first per the slice
spec: a small frozen, torch-free context dataclass; a ``check`` that returns
only through ``self.ok``/``self.fail``/``self.skip``; coverage that is never
fabricated; and controls of both kinds, including fixtures that prove the
framework's own "nothing inspected is VACUOUS, never PASS" rule is not
bypassed here either.

The five gates, and the incident shape each one targets
---------------------------------------------------------
1. :class:`TrajectoryIntegrityGate` (``ROLLOUT``) — per-row structural
   integrity of a flattened agentic batch: a loss-mask value outside
   ``{0, 1}``, a per-token column that runs its own length, a supervised
   token sitting on non-assistant/tool-call text, an INFRA row carrying a
   non-zero mask, or a reward/abstention pair that disagrees with itself.
   Every one of these is already refused by ``contracts.py`` for a batch that
   genuinely went through ``flatten()`` — this gate is defense in depth for a
   context an adapter built some other way, and its coverage is the row
   count, never special-cased to zero.
2. :class:`RolloutAbstentionGate` (``ROLLOUT``) — the fraction of rows an
   infrastructure failure, not the model, ended. A rate above the declared
   bound is a FAIL; an undeclared bound is a declared abstention, never a
   silent pass; and — the one place this gate deliberately does NOT special-
   case — a 100% infra batch is measured and compared like any other rate,
   because the framework's VACUOUS door is for EXAMINING NOTHING (zero rows),
   not for a rate that happens to be 1.0.
3. :class:`StalenessGate` (``WEIGHT_SYNC``) — after a weight-sync push, is the
   measured staleness claim present and healthy, and does every rollout
   session's policy version trail the just-published one by no more than the
   declared lag.
4. :class:`WeightSyncParityGate` (``WEIGHT_SYNC``) — when a parity probe
   supplies engine/learner logprob pairs, do they agree within the declared
   tolerance; an unmeasured parity abstains honestly rather than asserting a
   claim nobody checked.
5. :class:`PromptBytesGate` (``ROLLOUT``) — the rendered system prompt and
   tool schemas hash to the value declared at launch, so a silent template or
   tool-schema drift between declaration and rollout is a named FAIL, not a
   policy quietly trained against a prompt it never agreed to.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .core import (
    AbstentionKind,
    Control,
    ControlKind,
    Coverage,
    Gate,
    GateResult,
    Lifecycle,
    register,
)

__all__ = [
    "RolloutGateContext",
    "WeightSyncGateContext",
    "PromptBytesGateContext",
    "TrajectoryIntegrityGate",
    "RolloutAbstentionGate",
    "StalenessGate",
    "WeightSyncParityGate",
    "PromptBytesGate",
    "prompt_bytes_sha256",
]


# ---------------------------------------------------------------------------
# Contexts
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RolloutGateContext:
    """Everything :class:`TrajectoryIntegrityGate` and :class:`RolloutAbstentionGate`
    need from one step's flattened agentic batch.

    ``batch_columns`` is exactly the shape of
    ``rl.interfaces.ExperienceBatch.columns`` (``{column_name: per_row_values}``)
    so a caller already holding a built ``ExperienceBatch`` passes its
    ``.columns`` straight through with no adapter; a gate here reads only the
    columns it declares needing, and never opens a trajectory, a tokenizer, or
    a file. ``declared_max_infra_rate`` is ``None`` when the run never declared
    a bound — a legitimate, honestly-abstaining state, never a license to
    invent ``1.0`` or ``0.0``.
    """

    batch_columns: Mapping[str, Sequence[Any]]
    declared_max_infra_rate: float | None
    step: int


@dataclass(frozen=True)
class WeightSyncGateContext:
    """Everything :class:`StalenessGate` and :class:`WeightSyncParityGate` need
    from one weight-sync push.

    ``offered``/``failed``/``is_stale``/``seconds`` mirror
    ``rl.weightsync.SyncReport``'s own fields (offered server count, failed-rank
    count, the measured staleness claim, the measured push duration) without
    importing that module: the gate contract stays decoupled from the agentic
    RL plane's own types, exactly as :class:`~foundationscale.gates.example.ExpertCheckContext`
    stays decoupled from any real checkpoint reader. ``offered`` is the
    positive declaration :class:`StalenessGate` reads to decide whether a sync
    was attempted at all — ``0`` means it was not, a malformed (bool, negative,
    non-int) value establishes nothing and is never read as a declaration.

    ``rollout_policy_versions`` is the policy version each in-flight rollout
    session is running, ``None`` where a session's version is itself
    unmeasured. ``declared_max_lag`` has no ``None`` state: unlike
    ``declared_max_infra_rate`` it is a config field with a standing default
    (``1`` for S0), always in force.

    ``parity``/``declared_parity_atol`` are ``None`` until a GPU probe slice
    takes the measurement; see :class:`WeightSyncParityGate`.
    """

    offered: int
    failed: int
    is_stale: bool | None
    seconds: float | None
    step: int
    policy_version: int
    rollout_policy_versions: Sequence[int | None]
    declared_max_lag: int
    parity: Sequence[tuple[Sequence[float], Sequence[float]]] | None = None
    declared_parity_atol: float | None = None


@dataclass(frozen=True)
class PromptBytesGateContext:
    """What :class:`PromptBytesGate` needs: an observed and a declared sha256
    of the rendered system prompt plus tool schemas (see
    :func:`prompt_bytes_sha256`). Either may be ``None`` — a declaration never
    made, or an observation never taken — and neither absence is read as a
    match or a mismatch; it is a declared abstention (see the gate's own
    ``check``).
    """

    observed_sha256: str | None
    declared_sha256: str | None


def prompt_bytes_sha256(
    messages: Sequence[Mapping[str, Any]], tools: Sequence[Mapping[str, Any]]
) -> str:
    """Canonical sha256 of a rendered system prompt plus its tool schemas.

    Serialised through ``json.dumps(..., sort_keys=True, separators=(",", ":"))``
    so the digest is stable across dict key order and incidental whitespace: two
    callers rendering the same semantic prompt hash identically, and a merely
    reordered mapping is never mistaken for a changed prompt. This function
    computes an ``observed_sha256``; it is never itself a gate, and never reads
    a ``declared_sha256`` — that comparison belongs to :class:`PromptBytesGate`.
    """
    canonical = json.dumps(
        {"messages": list(messages), "tools": list(tools)},
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# 1. TrajectoryIntegrityGate
# ---------------------------------------------------------------------------

_PER_TOKEN_COLUMNS: tuple[str, ...] = (
    "response_ids",
    "loss_mask",
    "rollout_logprobs",
    "turn_index",
    "segment_kind",
    "tool_call_error_mask",
    "repetition_mask",
)
_ROW_SCALAR_COLUMNS: tuple[str, ...] = ("reward", "abstention_reason", "termination")
_TRAJECTORY_REQUIRED_COLUMNS: tuple[str, ...] = _PER_TOKEN_COLUMNS + _ROW_SCALAR_COLUMNS

_SUPERVISABLE_SEGMENT_KINDS = frozenset({"assistant", "tool_call"})
_INFRA_TERMINATION = "infra"


def _row_dict(
    *,
    response_ids: tuple[int, ...],
    loss_mask: tuple[int, ...],
    segment_kind: tuple[str, ...],
    termination: str = "stop",
    reward: float | None = 1.0,
    abstention_reason: str | None = None,
    rollout_logprobs: tuple[float | None, ...] | None = None,
    turn_index: tuple[int, ...] | None = None,
    tool_call_error_mask: tuple[int, ...] | None = None,
    repetition_mask: tuple[int, ...] | None = None,
) -> dict[str, Any]:
    """One row of ``RolloutGateContext.batch_columns``, as a plain dict keyed by
    column name — the row-oriented shape :func:`_columns_from_rows` transposes
    into the columnar shape the context (and real ``ExperienceBatch``) carries.
    Unset per-token columns default to an all-zero (or all-``None``, for
    ``rollout_logprobs``) tuple the length of ``response_ids``, since most
    controls below care about one column at a time and spelling out every
    column on every fixture would bury the one that matters.
    """
    n = len(response_ids)
    return {
        "response_ids": response_ids,
        "loss_mask": loss_mask,
        "segment_kind": segment_kind,
        "termination": termination,
        "reward": reward,
        "abstention_reason": abstention_reason,
        "rollout_logprobs": rollout_logprobs if rollout_logprobs is not None else (None,) * n,
        "turn_index": turn_index if turn_index is not None else (0,) * n,
        "tool_call_error_mask": (
            tool_call_error_mask if tool_call_error_mask is not None else (0,) * n
        ),
        "repetition_mask": repetition_mask if repetition_mask is not None else (0,) * n,
    }


def _columns_from_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, tuple[Any, ...]]:
    return {name: tuple(row[name] for row in rows) for name in _TRAJECTORY_REQUIRED_COLUMNS}


def _trajectory_ctx(rows: Sequence[Mapping[str, Any]], **overrides: Any) -> RolloutGateContext:
    fields: dict[str, Any] = {"declared_max_infra_rate": None, "step": 0}
    fields.update(overrides)
    return RolloutGateContext(batch_columns=_columns_from_rows(rows), **fields)


def _missing_column_ctx() -> RolloutGateContext:
    rows = [_row_dict(response_ids=(1,), loss_mask=(1,), segment_kind=("assistant",))]
    ctx = _trajectory_ctx(rows)
    cols = dict(ctx.batch_columns)
    del cols["turn_index"]
    return RolloutGateContext(
        batch_columns=cols, declared_max_infra_rate=ctx.declared_max_infra_rate, step=ctx.step
    )


def _length_mismatch_ctx() -> RolloutGateContext:
    rows = [_row_dict(response_ids=(1, 2), loss_mask=(0, 1), segment_kind=("system", "assistant"))]
    ctx = _trajectory_ctx(rows)
    cols = dict(ctx.batch_columns)
    cols["loss_mask"] = ((0,),)  # one entry for a two-token response
    return RolloutGateContext(
        batch_columns=cols, declared_max_infra_rate=ctx.declared_max_infra_rate, step=ctx.step
    )


def _loss_mask_out_of_range_ctx() -> RolloutGateContext:
    rows = [_row_dict(response_ids=(1, 2), loss_mask=(0, 2), segment_kind=("system", "assistant"))]
    return _trajectory_ctx(rows)


def _loss_mask_bool_entry_ctx() -> RolloutGateContext:
    rows = [_row_dict(response_ids=(1,), loss_mask=(True,), segment_kind=("assistant",))]
    return _trajectory_ctx(rows)


def _loss_mask_wrong_segment_ctx() -> RolloutGateContext:
    rows = [_row_dict(response_ids=(1,), loss_mask=(1,), segment_kind=("tool_result",))]
    return _trajectory_ctx(rows)


def _infra_row_nonzero_loss_mask_ctx() -> RolloutGateContext:
    rows = [
        _row_dict(
            response_ids=(1, 2),
            loss_mask=(0, 1),
            segment_kind=("assistant", "assistant"),
            termination="infra",
            reward=None,
            abstention_reason="infra:EnvInfraError",
        )
    ]
    return _trajectory_ctx(rows)


def _reward_abstention_mismatch_ctx() -> RolloutGateContext:
    rows = [
        _row_dict(
            response_ids=(1,),
            loss_mask=(1,),
            segment_kind=("assistant",),
            termination="stop",
            reward=None,
            abstention_reason=None,
        )
    ]
    return _trajectory_ctx(rows)


def _empty_trajectory_batch_ctx() -> RolloutGateContext:
    return RolloutGateContext(
        batch_columns={name: () for name in _TRAJECTORY_REQUIRED_COLUMNS},
        declared_max_infra_rate=None,
        step=0,
    )


def _healthy_trajectory_batch_ctx() -> RolloutGateContext:
    rows = [
        _row_dict(
            response_ids=(1, 2, 3),
            loss_mask=(0, 1, 1),
            segment_kind=("system", "assistant", "assistant"),
            termination="stop",
            reward=1.0,
            abstention_reason=None,
        ),
        _row_dict(
            response_ids=(4, 5),
            loss_mask=(0, 0),
            segment_kind=("assistant", "assistant"),
            termination="infra",
            reward=None,
            abstention_reason="infra:EnvInfraError",
        ),
    ]
    return _trajectory_ctx(rows, declared_max_infra_rate=0.9)


@register
class TrajectoryIntegrityGate(Gate):
    """Per-row structural integrity of a flattened agentic rollout batch.

    Five invariants, checked in this order for the first offending row so a
    FAIL always names the earliest defect a caller would want to fix first:
    required columns present; every per-token column the same length as
    ``response_ids`` for its row; every ``loss_mask`` entry a genuine ``0`` or
    ``1`` (``True``/``False`` excluded — ``1 is not True`` throughout this
    plane); ``loss_mask`` is ``1`` only where ``segment_kind`` is
    ``assistant``/``tool_call``; an INFRA-terminated row carries an all-zero
    ``loss_mask``; and ``reward is None`` exactly when ``abstention_reason`` is
    a non-empty string. ``contracts.py`` already enforces every one of these
    for a batch built through ``flatten()`` — this gate is the same invariant
    re-checked at the lifecycle boundary, for a context an adapter built some
    other way, following :mod:`foundationscale.gates.example`'s doctrine that a
    seam check known only to one constructor is the "absent from the other
    script" failure one level up.

    The empty batch is NOT special-cased: zero rows means ``self.ok`` is
    called over ``Coverage(checked=0, ...)``, which the framework downgrades to
    VACUOUS. The ``empty-batch`` control exists to prove nobody "helpfully"
    short-circuits that.
    """

    id = "agentic_rl.trajectory_integrity"
    description = (
        "Every row of a flattened agentic rollout batch is structurally sound: "
        "a {0,1} loss mask, aligned per-token columns, supervision only on "
        "assistant/tool-call tokens, an all-zero mask on INFRA rows, and a "
        "reward/abstention biconditional"
    )
    events = (Lifecycle.ROLLOUT,)
    context_type = RolloutGateContext

    def check(self, ctx: RolloutGateContext) -> GateResult:
        cols = ctx.batch_columns
        missing = [name for name in _TRAJECTORY_REQUIRED_COLUMNS if name not in cols]
        if missing:
            return self.fail(
                f"batch is missing {len(missing)} of {len(_TRAJECTORY_REQUIRED_COLUMNS)} "
                f"required column(s): {missing} — a row cannot be examined without its "
                f"own columns",
                Coverage.none("rows"),
                evidence={"missing_columns": missing, "present_columns": sorted(cols)},
            )
        response_ids = cols["response_ids"]
        num_rows = len(response_ids)
        cov = Coverage(checked=num_rows, unit="rows", expected=num_rows)
        for row in range(num_rows):
            resp_len = len(response_ids[row])
            for name in _PER_TOKEN_COLUMNS[1:]:
                entry_len = len(cols[name][row])
                if entry_len != resp_len:
                    return self.fail(
                        f"row {row}: column {name!r} has {entry_len} token entries but "
                        f"'response_ids' has {resp_len} for this row — per-token columns "
                        f"must carry one entry per response token",
                        cov,
                        evidence={
                            "row": row,
                            "column": name,
                            "entry_len": entry_len,
                            "response_len": resp_len,
                        },
                    )
            loss_mask_row = cols["loss_mask"][row]
            for token, value in enumerate(loss_mask_row):
                if type(value) is bool or value not in (0, 1):
                    return self.fail(
                        f"row {row} token {token}: loss_mask entry {value!r} "
                        f"({type(value).__name__}) is not a genuine 0 or 1 — 1 is not "
                        f"True",
                        cov,
                        evidence={"row": row, "token": token, "loss_mask_value": repr(value)},
                    )
            segment_kind_row = cols["segment_kind"][row]
            for token, (mask_value, kind) in enumerate(
                zip(loss_mask_row, segment_kind_row, strict=True)
            ):
                if mask_value == 1 and kind not in _SUPERVISABLE_SEGMENT_KINDS:
                    return self.fail(
                        f"row {row} token {token}: loss_mask is 1 but segment_kind is "
                        f"{kind!r}, not assistant/tool_call — only engine-generated "
                        f"assistant or tool-call tokens may be supervised",
                        cov,
                        evidence={"row": row, "token": token, "segment_kind": str(kind)},
                    )
            termination = cols["termination"][row]
            if termination == _INFRA_TERMINATION and any(v != 0 for v in loss_mask_row):
                return self.fail(
                    f"row {row}: termination is {_INFRA_TERMINATION!r} but loss_mask "
                    f"{tuple(loss_mask_row)!r} carries a non-zero entry — the run, not "
                    f"the model, ended this attempt, so none of its tokens is a choice "
                    f"the policy can be held to",
                    cov,
                    evidence={"row": row, "loss_mask": list(loss_mask_row)},
                )
            reward = cols["reward"][row]
            abstention_reason = cols["abstention_reason"][row]
            reward_is_none = reward is None
            abstains = bool(abstention_reason)
            if reward_is_none != abstains:
                return self.fail(
                    f"row {row}: reward={reward!r} and abstention_reason="
                    f"{abstention_reason!r} violate the biconditional — reward is None "
                    f"exactly when abstention_reason is a non-empty str",
                    cov,
                    evidence={
                        "row": row,
                        "reward": reward,
                        "abstention_reason": abstention_reason,
                    },
                )
        return self.ok(f"all {num_rows} rows pass trajectory integrity checks", cov)

    def controls(self) -> list[Control]:
        return [
            Control(
                "missing-column",
                ControlKind.MUST_FIRE,
                _missing_column_ctx,
                note="batch_columns is missing 'turn_index' entirely",
            ),
            Control(
                "per-token-length-mismatch",
                ControlKind.MUST_FIRE,
                _length_mismatch_ctx,
                note="loss_mask carries 1 entry for a 2-token response_ids row",
            ),
            Control(
                "loss-mask-out-of-range",
                ControlKind.MUST_FIRE,
                _loss_mask_out_of_range_ctx,
                note="loss_mask entry is 2, outside {0, 1}",
            ),
            Control(
                "loss-mask-bool-entry",
                ControlKind.MUST_FIRE,
                _loss_mask_bool_entry_ctx,
                note="loss_mask entry is True: 1 is not True throughout this plane",
            ),
            Control(
                "loss-mask-wrong-segment",
                ControlKind.MUST_FIRE,
                _loss_mask_wrong_segment_ctx,
                note="loss_mask=1 on a tool_result token, never assistant/tool_call",
            ),
            Control(
                "infra-row-nonzero-loss-mask",
                ControlKind.MUST_FIRE,
                _infra_row_nonzero_loss_mask_ctx,
                note="termination=infra but loss_mask carries a 1 — the run, not the "
                "model, ended this attempt",
            ),
            Control(
                "reward-abstention-mismatch",
                ControlKind.MUST_FIRE,
                _reward_abstention_mismatch_ctx,
                note="reward=None with abstention_reason=None: the biconditional's "
                "FAIL side, never scored and never explained",
            ),
            Control(
                "empty-batch",
                ControlKind.MUST_FIRE,
                _empty_trajectory_batch_ctx,
                note="zero rows: the all([]) trap — must VACUOUS-block, never pass",
            ),
            Control(
                "healthy-batch",
                ControlKind.MUST_PASS,
                _healthy_trajectory_batch_ctx,
                note="one STOP row and one INFRA row, both internally consistent",
            ),
        ]


# ---------------------------------------------------------------------------
# 2. RolloutAbstentionGate
# ---------------------------------------------------------------------------


def _abstention_ctx(
    reasons: Sequence[str | None], *, declared_max_infra_rate: float | None
) -> RolloutGateContext:
    return RolloutGateContext(
        batch_columns={"abstention_reason": tuple(reasons)},
        declared_max_infra_rate=declared_max_infra_rate,
        step=0,
    )


def _rate_exceeds_bound_ctx() -> RolloutGateContext:
    return _abstention_ctx(
        ("infra:transport", "infra:transport", None), declared_max_infra_rate=0.1
    )


def _all_infra_under_generous_bound_ctx() -> RolloutGateContext:
    # 100% infra is NOT itself vacuous: it is measured and compared like any
    # other rate. A generous declared bound must let it PASS; only a batch of
    # ZERO rows is examined-nothing.
    return _abstention_ctx(("infra:a", "infra:b"), declared_max_infra_rate=1.0)


def _empty_abstention_batch_ctx() -> RolloutGateContext:
    return _abstention_ctx((), declared_max_infra_rate=0.5)


def _healthy_abstention_ctx() -> RolloutGateContext:
    return _abstention_ctx(("infra:transport", None, None, None), declared_max_infra_rate=0.5)


def _undeclared_bound_ctx() -> RolloutGateContext:
    return _abstention_ctx(("infra:transport", None), declared_max_infra_rate=None)


@register
class RolloutAbstentionGate(Gate):
    """The fraction of a rollout batch's rows an infrastructure failure ended.

    ``infra_rate = (rows whose abstention_reason starts with "infra:") / rows``,
    FAIL when it exceeds ``declared_max_infra_rate``. Two abstention doors, each
    for a DIFFERENT reason a verdict cannot be reached:

    * Zero rows in the batch: ``self.ok`` over ``Coverage.none("rows")``
      downgrades to the framework's own VACUOUS — this gate does not
      special-case it, exactly as :mod:`foundationscale.gates.example` does
      not special-case an empty expert set.
    * ``declared_max_infra_rate is None``: the run never declared a bound, so
      the measured rate (named in the reason) has nothing to compare against.
      :attr:`~foundationscale.gates.core.AbstentionKind.NOT_ESTABLISHED`,
      because the property — "the infra rate is acceptable" — may well be
      true; this gate's evidence just cannot settle it without a declaration.

    What this gate deliberately does NOT do: treat a 100% infra rate as its
    own abstention door. A batch where every attempt infra-failed is measured
    exactly like one where a tenth did — ``infra_rate=1.0`` is compared
    against the declared bound like any other value, and FAILs or PASSes on
    that comparison alone. Rows failing the WAY a harness crashes is
    information this rate exists to surface, not an exemption from
    reporting it.
    """

    id = "agentic_rl.rollout_abstention"
    description = (
        "The rollout batch's measured infra-abstention rate is within the "
        "declared bound; an undeclared bound abstains honestly instead of "
        "passing an unadjudicated rate"
    )
    events = (Lifecycle.ROLLOUT,)
    context_type = RolloutGateContext

    def check(self, ctx: RolloutGateContext) -> GateResult:
        abstention_reasons = ctx.batch_columns.get("abstention_reason", ())
        total = len(abstention_reasons)
        if total == 0:
            return self.ok(
                "batch carries zero rows — there is nothing to measure an infra rate over",
                Coverage.none("rows"),
            )
        infra_count = sum(
            1
            for reason in abstention_reasons
            if isinstance(reason, str) and reason.startswith("infra:")
        )
        if ctx.declared_max_infra_rate is None:
            return self.skip(
                f"no declared_max_infra_rate was recorded for this run: the measured "
                f"infra rate is {infra_count}/{total} but there is no declared bound "
                f"to adjudicate it against",
                kind=AbstentionKind.NOT_ESTABLISHED,
            )
        infra_rate = infra_count / total
        cov = Coverage(checked=total, unit="rows", expected=total)
        evidence = {
            "infra_count": infra_count,
            "total": total,
            "infra_rate": infra_rate,
            "declared_max_infra_rate": ctx.declared_max_infra_rate,
        }
        if infra_rate > ctx.declared_max_infra_rate:
            return self.fail(
                f"infra rate {infra_count}/{total} ({infra_rate:.4f}) exceeds "
                f"declared_max_infra_rate={ctx.declared_max_infra_rate}",
                cov,
                evidence=evidence,
            )
        return self.ok(
            f"infra rate {infra_count}/{total} ({infra_rate:.4f}) is within "
            f"declared_max_infra_rate={ctx.declared_max_infra_rate}",
            cov,
            evidence=evidence,
        )

    def controls(self) -> list[Control]:
        return [
            Control(
                "rate-exceeds-bound",
                ControlKind.MUST_FIRE,
                _rate_exceeds_bound_ctx,
                note="2/3 rows infra-abstained against a declared bound of 0.1",
            ),
            Control(
                "empty-batch",
                ControlKind.MUST_FIRE,
                _empty_abstention_batch_ctx,
                note="zero rows: the all([]) trap, must VACUOUS-block rather than pass",
            ),
            Control(
                "healthy",
                ControlKind.MUST_PASS,
                _healthy_abstention_ctx,
                note="1/4 rows infra-abstained against a declared bound of 0.5",
            ),
            Control(
                "all-infra-under-generous-bound",
                ControlKind.MUST_PASS,
                _all_infra_under_generous_bound_ctx,
                note="100% infra is measured and compared like any other rate, not "
                "treated as its own vacuity — only zero rows is examined-nothing",
            ),
            Control(
                "undeclared-bound",
                ControlKind.MUST_PASS,
                _undeclared_bound_ctx,
                note="no declared_max_infra_rate: the measured rate has nothing to compare against",
                expect_skip=(
                    "no declared_max_infra_rate was recorded for this run, so the "
                    "measured infra rate cannot be adjudicated"
                ),
            ),
        ]


# ---------------------------------------------------------------------------
# 3. StalenessGate
# ---------------------------------------------------------------------------


def _sync_ctx(
    *,
    offered: int = 1,
    failed: int = 0,
    is_stale: bool | None = False,
    seconds: float | None = 1.0,
    step: int = 0,
    policy_version: int = 1,
    rollout_policy_versions: Sequence[int | None] = (1,),
    declared_max_lag: int = 1,
    parity: Sequence[tuple[Sequence[float], Sequence[float]]] | None = None,
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


def _malformed_offered_ctx() -> WeightSyncGateContext:
    return WeightSyncGateContext(
        # bool is a subtype of int (mypy admits it; the whole point of this fixture
        # is the RUNTIME type(offered) is bool guard in StalenessGate.check, which
        # a static int annotation cannot express): the exact malformed look-alike
        # zero the "no sync offered" door must refuse.
        offered=False,
        failed=0,
        is_stale=None,
        seconds=None,
        step=0,
        policy_version=1,
        rollout_policy_versions=(),
        declared_max_lag=1,
        parity=None,
        declared_parity_atol=None,
    )


def _no_sync_offered_ctx() -> WeightSyncGateContext:
    return _sync_ctx(offered=0, is_stale=None, seconds=None, rollout_policy_versions=())


def _unmeasured_staleness_ctx() -> WeightSyncGateContext:
    return _sync_ctx(offered=1, is_stale=None)


def _measured_stale_ctx() -> WeightSyncGateContext:
    return _sync_ctx(is_stale=True)


def _unmeasured_rollout_version_ctx() -> WeightSyncGateContext:
    return _sync_ctx(policy_version=2, rollout_policy_versions=(2, None))


def _lag_exceeded_ctx() -> WeightSyncGateContext:
    return _sync_ctx(policy_version=5, rollout_policy_versions=(1,), declared_max_lag=1)


def _healthy_staleness_ctx() -> WeightSyncGateContext:
    return _sync_ctx(policy_version=2, rollout_policy_versions=(1, 2), declared_max_lag=1)


def _empty_rollout_versions_ctx() -> WeightSyncGateContext:
    return _sync_ctx(rollout_policy_versions=())


@register
class StalenessGate(Gate):
    """After a weight-sync push, is the fleet's measured staleness healthy and
    is every SUPPLIED rollout session's policy version within the declared lag.

    Checked in order: ``offered`` must be a genuine non-negative int — a
    look-alike zero (``False``) establishes nothing and must never buy the
    no-sync-offered door, exactly the ``malformed-dense-count-bool`` lesson
    :mod:`foundationscale.gates.example` encodes for ``declared_expert_count``.
    ``offered == 0`` is then a POSITIVE declaration that no sync was attempted
    at this event: :attr:`~foundationscale.gates.core.AbstentionKind.NOT_APPLICABLE`,
    staleness is a property this event does not have. Otherwise a sync WAS
    offered: ``is_stale`` must be a measured bool (``None`` is an unmeasured
    claim and FAILs; a measured ``True`` FAILs too — a realization's own
    staleness measurement is not something this gate ignores once taken).
    Finally every ``rollout_policy_versions`` entry must satisfy
    ``policy_version - v <= declared_max_lag`` (an unmeasured, ``None`` entry
    FAILs).

    Coverage is ``1`` (the push's own measured ``is_stale`` fact) plus the
    number of rollout policy versions supplied, NEVER zero once ``is_stale``
    is confirmed ``False`` — an EMPTY ``rollout_policy_versions`` is not a
    reason to call this gate's examination vacuous or to abstain: the push's
    own staleness measurement is one real, non-fabricated unit this gate DID
    examine, and the detail names exactly how many (if any) rollout sessions
    were additionally checked for lag. A caller supplying no per-session
    versions gets an honest PASS scoped to what was measured, never a
    fabricated claim about sessions nobody reported — and never a SKIP that
    would leave the whole ``WEIGHT_SYNC`` event with no non-abstaining result
    at all whenever :class:`WeightSyncParityGate` also legitimately abstains
    (``parity`` is ``None`` until a GPU probe slice exists).
    """

    id = "agentic_rl.weight_sync_staleness"
    description = (
        "A weight-sync push reports measured, non-stale staleness, and every "
        "supplied rollout policy version trails the trained policy by no more "
        "than the declared lag"
    )
    events = (Lifecycle.WEIGHT_SYNC,)
    context_type = WeightSyncGateContext

    def check(self, ctx: WeightSyncGateContext) -> GateResult:
        offered = ctx.offered
        if type(offered) is bool or not isinstance(offered, int) or offered < 0:
            return self.ok(
                f"offered is {offered!r} ({type(offered).__name__}), not a genuine "
                f"non-negative integer: a malformed denominator establishes nothing, "
                f"and a value that merely compares equal to zero must never buy the "
                f"no-sync-offered abstention",
                Coverage.none("staleness checks"),
                evidence={"offered_raw": repr(offered)},
            )
        if offered == 0:
            return self.skip(
                "offered == 0 — a positive declaration that no weight sync was "
                "attempted at this event, so staleness is a property this event "
                "does not have",
                kind=AbstentionKind.NOT_APPLICABLE,
            )
        if ctx.is_stale is None:
            return self.fail(
                f"a weight sync was offered to {offered} server(s) but is_stale was "
                f"never measured — an unmeasured staleness claim cannot be reported "
                f"healthy",
                Coverage.none("staleness checks"),
                evidence={"offered": offered, "failed": ctx.failed, "seconds": ctx.seconds},
            )
        if ctx.is_stale:
            return self.fail(
                "the weight-sync report measured is_stale=True: the published weights are stale",
                Coverage.none("staleness checks"),
                evidence={"offered": offered, "failed": ctx.failed, "seconds": ctx.seconds},
            )
        versions = tuple(ctx.rollout_policy_versions)
        # checked = 1 (the push's own confirmed-non-stale measurement) + however
        # many rollout sessions were supplied: NEVER zero from this point on, so
        # a push with zero supplied versions still reaches a real PASS scoped to
        # what it measured, rather than a vacuous or skipped non-result.
        cov = Coverage(checked=1 + len(versions), unit="staleness checks")
        for index, version in enumerate(versions):
            if version is None:
                return self.fail(
                    f"rollout session {index}: policy version is unmeasured (None) — "
                    f"lag against policy_version={ctx.policy_version} cannot be "
                    f"established",
                    cov,
                    evidence={"index": index, "policy_version": ctx.policy_version},
                )
            lag = ctx.policy_version - version
            if lag > ctx.declared_max_lag:
                return self.fail(
                    f"rollout session {index}: policy version {version} trails "
                    f"policy_version={ctx.policy_version} by {lag}, exceeding "
                    f"declared_max_lag={ctx.declared_max_lag}",
                    cov,
                    evidence={
                        "index": index,
                        "rollout_policy_version": version,
                        "policy_version": ctx.policy_version,
                        "lag": lag,
                        "declared_max_lag": ctx.declared_max_lag,
                    },
                )
        return self.ok(
            f"is_stale confirmed False; {len(versions)} supplied rollout policy "
            f"version(s) all within declared_max_lag={ctx.declared_max_lag} of "
            f"policy_version={ctx.policy_version}",
            cov,
        )

    def controls(self) -> list[Control]:
        return [
            Control(
                "malformed-offered-bool",
                ControlKind.MUST_FIRE,
                _malformed_offered_ctx,
                note="offered=False: a look-alike zero must not buy the "
                "no-sync-offered door — VACUOUS-blocks instead",
            ),
            Control(
                "unmeasured-staleness",
                ControlKind.MUST_FIRE,
                _unmeasured_staleness_ctx,
                note="a sync was offered but is_stale is None",
            ),
            Control(
                "measured-stale",
                ControlKind.MUST_FIRE,
                _measured_stale_ctx,
                note="is_stale measured True: the realization's own claim is stale",
            ),
            Control(
                "unmeasured-rollout-version",
                ControlKind.MUST_FIRE,
                _unmeasured_rollout_version_ctx,
                note="one rollout session's policy version is None",
            ),
            Control(
                "lag-exceeded",
                ControlKind.MUST_FIRE,
                _lag_exceeded_ctx,
                note="a rollout session trails the trained policy by 4, over a "
                "declared_max_lag of 1",
            ),
            Control(
                "healthy",
                ControlKind.MUST_PASS,
                _healthy_staleness_ctx,
                note="is_stale=False and every rollout version within declared_max_lag",
            ),
            Control(
                "no-sync-offered",
                ControlKind.MUST_PASS,
                _no_sync_offered_ctx,
                note="offered=0: no weight sync was attempted at this event",
                expect_skip=(
                    "offered == 0 is a positive declaration that no sync was "
                    "attempted; staleness is a property this event does not have"
                ),
            ),
            Control(
                "empty-rollout-versions",
                ControlKind.MUST_PASS,
                _empty_rollout_versions_ctx,
                note="is_stale=False but rollout_policy_versions is empty — the real "
                "production shape until a later slice tracks per-session versions: "
                "a real, scoped PASS (coverage=1), never a skip or a vacuity",
            ),
        ]


# ---------------------------------------------------------------------------
# 4. WeightSyncParityGate
# ---------------------------------------------------------------------------


def _parity_ctx(
    parity: Sequence[tuple[Sequence[float], Sequence[float]]] | None,
    declared_parity_atol: float | None,
) -> WeightSyncGateContext:
    return _sync_ctx(
        offered=0,
        is_stale=None,
        seconds=None,
        rollout_policy_versions=(),
        parity=parity,
        declared_parity_atol=declared_parity_atol,
    )


def _parity_exceeds_atol_ctx() -> WeightSyncGateContext:
    return _parity_ctx(parity=(((0.0, -1.0), (0.0, -1.5)),), declared_parity_atol=0.1)


def _parity_length_mismatch_ctx() -> WeightSyncGateContext:
    return _parity_ctx(parity=(((0.0, -1.0), (0.0,)),), declared_parity_atol=0.1)


def _parity_empty_pairs_ctx() -> WeightSyncGateContext:
    return _parity_ctx(parity=(), declared_parity_atol=0.1)


def _parity_healthy_ctx() -> WeightSyncGateContext:
    return _parity_ctx(parity=(((0.0, -1.0), (0.01, -1.02)),), declared_parity_atol=0.05)


def _parity_unmeasured_ctx() -> WeightSyncGateContext:
    return _parity_ctx(parity=None, declared_parity_atol=None)


def _parity_no_declared_atol_ctx() -> WeightSyncGateContext:
    return _parity_ctx(parity=(((0.0,), (0.0,)),), declared_parity_atol=None)


@register
class WeightSyncParityGate(Gate):
    """When a parity probe supplies engine/learner logprob pairs, the maximum
    absolute difference over every aligned token must not exceed the declared
    tolerance.

    ``parity is None`` is a declared
    :attr:`~foundationscale.gates.core.AbstentionKind.NOT_ESTABLISHED`: the
    measurement was not taken (``parity None until the GPU probe slice
    measures it``), never asserted as agreement. ``declared_parity_atol is
    None`` with ``parity`` supplied abstains the same way — a measured
    difference cannot be adjudicated without a declared tolerance. A row whose
    two sequences carry different lengths FAILs naming both lengths, because
    parity has no answer for a token either side never scored. An empty
    ``parity`` tuple (declared and present, but zero rows) is NOT
    special-cased: ``self.ok`` over zero aligned tokens downgrades to VACUOUS,
    the same rule as every other gate in this module.
    """

    id = "agentic_rl.weight_sync_parity"
    description = (
        "Engine-reported and learner-recomputed logprobs agree within the "
        "declared tolerance over every aligned sampled token; an untaken "
        "measurement abstains honestly instead of asserting agreement"
    )
    events = (Lifecycle.WEIGHT_SYNC,)
    context_type = WeightSyncGateContext

    def check(self, ctx: WeightSyncGateContext) -> GateResult:
        if ctx.parity is None:
            return self.skip(
                "parity was not measured at this event (parity=None) — the GPU "
                "probe slice that takes this measurement has not run yet",
                kind=AbstentionKind.NOT_ESTABLISHED,
            )
        if ctx.declared_parity_atol is None:
            return self.skip(
                "parity was measured but no declared_parity_atol was recorded — a "
                "measured difference cannot be adjudicated without a declared "
                "tolerance",
                kind=AbstentionKind.NOT_ESTABLISHED,
            )
        pairs = tuple(ctx.parity)
        worst_diff = -1.0
        worst_row = -1
        worst_token = -1
        worst_engine = 0.0
        worst_learner = 0.0
        total_tokens = 0
        for row, (engine_logprobs, learner_logprobs) in enumerate(pairs):
            engine_seq = tuple(engine_logprobs)
            learner_seq = tuple(learner_logprobs)
            if len(engine_seq) != len(learner_seq):
                return self.fail(
                    f"row {row}: engine_logprobs has {len(engine_seq)} entries but "
                    f"learner_logprobs has {len(learner_seq)} entries — parity cannot be "
                    f"computed over misaligned sequences",
                    Coverage(checked=total_tokens, unit="aligned tokens"),
                    evidence={
                        "row": row,
                        "engine_len": len(engine_seq),
                        "learner_len": len(learner_seq),
                    },
                )
            for token, (engine_value, learner_value) in enumerate(
                zip(engine_seq, learner_seq, strict=True)
            ):
                diff = abs(engine_value - learner_value)
                total_tokens += 1
                if diff > worst_diff:
                    worst_diff = diff
                    worst_row, worst_token = row, token
                    worst_engine, worst_learner = engine_value, learner_value
        cov = Coverage(checked=total_tokens, unit="aligned tokens", expected=total_tokens)
        if total_tokens and worst_diff > ctx.declared_parity_atol:
            return self.fail(
                f"row {worst_row} token {worst_token}: |engine {worst_engine!r} - "
                f"learner {worst_learner!r}| = {worst_diff} exceeds "
                f"declared_parity_atol={ctx.declared_parity_atol}",
                cov,
                evidence={
                    "row": worst_row,
                    "token": worst_token,
                    "engine_logprob": worst_engine,
                    "learner_logprob": worst_learner,
                    "diff": worst_diff,
                    "declared_parity_atol": ctx.declared_parity_atol,
                },
            )
        return self.ok(
            f"all {total_tokens} aligned tokens are within "
            f"declared_parity_atol={ctx.declared_parity_atol} (max diff "
            f"{max(worst_diff, 0.0)})",
            cov,
        )

    def controls(self) -> list[Control]:
        return [
            Control(
                "exceeds-atol",
                ControlKind.MUST_FIRE,
                _parity_exceeds_atol_ctx,
                note="a token pair differs by 0.5 against a declared atol of 0.1",
            ),
            Control(
                "length-mismatch",
                ControlKind.MUST_FIRE,
                _parity_length_mismatch_ctx,
                note="engine_logprobs has 2 entries, learner_logprobs has 1",
            ),
            Control(
                "empty-pairs",
                ControlKind.MUST_FIRE,
                _parity_empty_pairs_ctx,
                note="parity=() — declared and present but zero rows: the all([]) "
                "trap, must VACUOUS-block rather than pass",
            ),
            Control(
                "healthy",
                ControlKind.MUST_PASS,
                _parity_healthy_ctx,
                note="every aligned token within a declared atol of 0.05",
            ),
            Control(
                "parity-unmeasured",
                ControlKind.MUST_PASS,
                _parity_unmeasured_ctx,
                note="parity=None: the GPU probe slice has not measured it yet",
                expect_skip=(
                    "parity was not measured at this event — honest absence, never "
                    "asserted agreement"
                ),
            ),
            Control(
                "atol-undeclared",
                ControlKind.MUST_PASS,
                _parity_no_declared_atol_ctx,
                note="parity measured but declared_parity_atol is None",
                expect_skip=(
                    "a measured parity difference cannot be adjudicated without a "
                    "declared tolerance"
                ),
            ),
        ]


# ---------------------------------------------------------------------------
# 5. PromptBytesGate
# ---------------------------------------------------------------------------


def _prompt_mismatch_ctx() -> PromptBytesGateContext:
    return PromptBytesGateContext(observed_sha256="a" * 64, declared_sha256="b" * 64)


def _prompt_unmeasured_observation_ctx() -> PromptBytesGateContext:
    return PromptBytesGateContext(observed_sha256=None, declared_sha256="a" * 64)


def _prompt_match_ctx() -> PromptBytesGateContext:
    digest = prompt_bytes_sha256(
        messages=({"role": "system", "content": "be helpful and concise"},),
        tools=({"name": "bash", "parameters": {"command": "str"}},),
    )
    return PromptBytesGateContext(observed_sha256=digest, declared_sha256=digest)


def _prompt_undeclared_ctx() -> PromptBytesGateContext:
    return PromptBytesGateContext(observed_sha256="a" * 64, declared_sha256=None)


@register
class PromptBytesGate(Gate):
    """The observed sha256 of the rendered system prompt plus tool schemas
    matches the value declared at launch (see :func:`prompt_bytes_sha256`).

    ``declared_sha256 is None`` is a declared
    :attr:`~foundationscale.gates.core.AbstentionKind.NOT_ESTABLISHED`: the
    claim was never made, so there is nothing to compare an observation
    against. With a declaration in force, ``observed_sha256 is None`` FAILs —
    an unmeasured observation beside a standing declaration is itself a
    defect, never silently passed — and a mismatch FAILs naming both digests.
    """

    id = "agentic_rl.prompt_bytes"
    description = (
        "The rendered system prompt and tool schemas hash to the value "
        "declared at launch; an unrecorded declaration abstains honestly "
        "instead of passing an unadjudicated observation"
    )
    events = (Lifecycle.ROLLOUT,)
    context_type = PromptBytesGateContext

    def check(self, ctx: PromptBytesGateContext) -> GateResult:
        if ctx.declared_sha256 is None:
            return self.skip(
                "no declared_sha256 was recorded for this run — the prompt-bytes "
                "claim was never made, so there is nothing to compare an "
                "observation against",
                kind=AbstentionKind.NOT_ESTABLISHED,
            )
        if ctx.observed_sha256 is None:
            return self.fail(
                f"declared_sha256={ctx.declared_sha256!r} is recorded but "
                f"observed_sha256 was never computed — the rendered prompt bytes "
                f"were not measured at this event",
                Coverage.none("prompt renders"),
                evidence={"declared_sha256": ctx.declared_sha256},
            )
        cov = Coverage(checked=1, unit="prompt renders", expected=1)
        if ctx.observed_sha256 != ctx.declared_sha256:
            return self.fail(
                f"observed sha256 {ctx.observed_sha256!r} does not match declared "
                f"{ctx.declared_sha256!r}: the rendered system prompt and/or tool "
                f"schemas changed since the declaration",
                cov,
                evidence={
                    "observed_sha256": ctx.observed_sha256,
                    "declared_sha256": ctx.declared_sha256,
                },
            )
        return self.ok(f"observed sha256 matches declared ({ctx.declared_sha256!r})", cov)

    def controls(self) -> list[Control]:
        return [
            Control(
                "mismatch",
                ControlKind.MUST_FIRE,
                _prompt_mismatch_ctx,
                note="observed and declared digests differ",
            ),
            Control(
                "unmeasured-observation",
                ControlKind.MUST_FIRE,
                _prompt_unmeasured_observation_ctx,
                note="declared_sha256 is recorded but observed_sha256 is None",
            ),
            Control(
                "match",
                ControlKind.MUST_PASS,
                _prompt_match_ctx,
                note="observed and declared digests are identical",
            ),
            Control(
                "undeclared",
                ControlKind.MUST_PASS,
                _prompt_undeclared_ctx,
                note="declared_sha256 is None: the claim was never made",
                expect_skip=(
                    "no declared_sha256 was recorded for this run, so there is "
                    "nothing to compare an observation against"
                ),
            ),
        ]
