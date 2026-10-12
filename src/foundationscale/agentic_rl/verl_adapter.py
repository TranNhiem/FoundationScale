"""``foundationscale.agentic_rl.verl_adapter`` — verl AgentLoop -> fs.trajectory/v2 adapter.

Reads verl's ``AgentLoopOutput`` duck-typed (attribute OR mapping access) and NEVER
imports verl at module import time.  Stdlib + ``trajectory_v2`` only.

Upstream contract (pinned ``9058412fb308e9f2e31a794ad1d18d97d4286485``):
``verl/experimental/agent_loop/agent_loop.py:88-115``.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Mapping, Sequence
from importlib import metadata
from typing import Any, NoReturn

from foundationscale.agentic_rl.trajectory_v2 import (
    EnvMeta,
    Episode,
    EpisodeStatus,
    GatewayMeta,
    HarnessMeta,
    ModelMeta,
    RewardEvent,
    RewardScope,
    Segment,
    Termination,
)

VERL_PIN = "9058412fb308e9f2e31a794ad1d18d97d4286485"

__all__ = [
    "VERL_PIN",
    "VerlAdapterRefusal",
    "episode_from_agent_loop_output",
    "require_pinned_verl",
    "samples_from_rollout_batch",
]


class VerlAdapterRefusal(ValueError):
    """Raised when a verl ``AgentLoopOutput`` cannot be adapted to v2."""


def _refuse(msg: str) -> NoReturn:
    raise VerlAdapterRefusal(msg)


def _is_int(x: Any) -> bool:
    """True iff ``x`` is a real ``int`` (``bool`` is never an ``int`` here)."""
    return type(x) is int


def _is_number(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _is_finite(x: Any) -> bool:
    return _is_number(x) and math.isfinite(float(x))


def _read(out: object, field_name: str, *, optional: bool = False) -> Any:
    """Read ``field_name`` from ``out`` via getattr, falling back to mapping access.

    ``optional`` fields mirror verl's ``Optional[...] = None`` defaults: absent reads as
    ``None``. Required fields absent are refused."""
    value: Any = getattr(out, field_name, _MISSING)
    if value is not _MISSING:
        return value
    if isinstance(out, Mapping):
        try:
            return out[field_name]
        except KeyError:
            pass
    if optional:
        return None
    _refuse(f"verl AgentLoopOutput.{field_name}: missing field")


class _Missing:
    __slots__ = ()


_MISSING = _Missing()


def _as_sequence(value: Any, field_name: str) -> tuple[Any, ...]:
    if isinstance(value, tuple):
        return value
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(value)
    _refuse(f"verl AgentLoopOutput.{field_name}: expected a sequence, got {type(value).__name__}")


def _nested_experts(value: Any, field_name: str) -> tuple[tuple[tuple[int, ...], ...], ...]:
    rows = _as_sequence(value, field_name)
    out_rows: list[tuple[tuple[int, ...], ...]] = []
    for i, row in enumerate(rows):
        layers_raw = _as_sequence(row, f"{field_name}[{i}]")
        layers: list[tuple[int, ...]] = []
        for j, layer in enumerate(layers_raw):
            ids = _as_sequence(layer, f"{field_name}[{i}][{j}]")
            layers.append(tuple(ids))
        out_rows.append(tuple(layers))
    return tuple(out_rows)


def episode_from_agent_loop_output(
    out: object,
    *,
    episode_id: str,
    group_id: str,
    policy_version: int,
    model: ModelMeta,
    env: EnvMeta,
    gateway: GatewayMeta,
    harness_version: str = VERL_PIN,
) -> Episode:
    """Adapt one verl ``AgentLoopOutput`` into a one-segment v2 :class:`Episode`.

    Refusals (:class:`VerlAdapterRefusal`) name the field and both counts.
    """
    prompt_ids_raw = _read(out, "prompt_ids")
    response_ids_raw = _read(out, "response_ids")
    response_mask_raw = _read(out, "response_mask")
    response_logprobs_raw = _read(out, "response_logprobs", optional=True)
    routed_experts_raw = _read(out, "routed_experts", optional=True)
    reward_score_raw = _read(out, "reward_score", optional=True)

    prompt_ids = _as_sequence(prompt_ids_raw, "prompt_ids")
    response_ids = _as_sequence(response_ids_raw, "response_ids")
    response_mask = _as_sequence(response_mask_raw, "response_mask")
    if response_logprobs_raw is None:
        response_logprobs: tuple[Any, ...] | None = None
    else:
        response_logprobs = _as_sequence(response_logprobs_raw, "response_logprobs")

    n = len(response_ids)
    if n == 0:
        _refuse("verl AgentLoopOutput.response_ids: empty response (0 of 0 tokens)")

    if len(response_mask) != n:
        _refuse(
            f"verl AgentLoopOutput.response_mask: {len(response_mask)} of {n} entries "
            "(len(response_mask) must equal len(response_ids))"
        )
    if response_logprobs is not None and len(response_logprobs) != n:
        _refuse(
            f"verl AgentLoopOutput.response_logprobs: {len(response_logprobs)} of {n} entries "
            "(len(response_logprobs) must equal len(response_ids))"
        )

    bad_mask = sum(1 for m in response_mask if not (_is_int(m) and m in (0, 1)))
    if bad_mask:
        _refuse(
            f"verl AgentLoopOutput.response_mask: {bad_mask} of {n} entries are not 0/1 ints "
            "(bool is not int here)"
        )

    mask = tuple(int(m) for m in response_mask)
    trainable = sum(1 for m in mask if m == 1)
    if response_logprobs is None and trainable > 0:
        _refuse(
            f"verl AgentLoopOutput.response_logprobs: None with {trainable} of {n} mask==1 tokens; "
            "rollout produced no logprobs: enable calculate_log_probs in the verl rollout config"
        )

    if not _is_int(policy_version) or policy_version < 0:
        _refuse(f"policy_version: expected int >= 0, got {policy_version!r}")

    logprobs: list[float] = []
    policy_versions: list[int] = []
    for i, m in enumerate(mask):
        if m == 1:
            assert response_logprobs is not None
            lp = response_logprobs[i]
            if not _is_finite(lp):
                _refuse(
                    f"verl AgentLoopOutput.response_logprobs[{i}]: non-finite logprob {lp!r} "
                    f"on a mask-1 token ({i} of {n} entries checked)"
                )
            logprobs.append(float(lp))
            policy_versions.append(policy_version)
        else:
            logprobs.append(math.nan)
            policy_versions.append(-1)

    spans: list[tuple[int, int, str]] = []
    k = 0
    i = 0
    while i < n:
        if mask[i] != 1:
            i += 1
            continue
        start = i
        while i < n and mask[i] == 1:
            i += 1
        spans.append((start, i, f"g{k}"))
        k += 1

    routed_experts: tuple[tuple[tuple[int, ...], ...], ...] | None = None
    if routed_experts_raw is not None:
        rows = _nested_experts(routed_experts_raw, "routed_experts")
        expected = len(prompt_ids) + n
        if len(rows) != expected:
            _refuse(
                f"verl AgentLoopOutput.routed_experts: {len(rows)} of {expected} rows "
                "(must cover prompt_ids + response_ids)"
            )
        routed_experts = rows

    reward_events: tuple[RewardEvent, ...] = ()
    if reward_score_raw is not None:
        if not _is_finite(reward_score_raw):
            _refuse(
                f"verl AgentLoopOutput.reward_score: expected finite float or None, "
                f"got {reward_score_raw!r}"
            )
        reward_events = (
            RewardEvent(
                source="verl_reward_score",
                verifier_id="verl_reward_score",
                verifier_version=harness_version,
                value=float(reward_score_raw),
                scope=RewardScope.EPISODE,
            ),
        )

    segment = Segment(
        segment_index=0,
        prompt_ids=tuple(prompt_ids),
        response_ids=tuple(response_ids),
        loss_mask=mask,
        logprobs=tuple(logprobs),
        policy_version=tuple(policy_versions),
        generation_spans=tuple(spans),
        routed_experts=routed_experts,
    )

    return Episode(
        episode_id=episode_id,
        group_id=group_id,
        task_prompt_ids=tuple(prompt_ids),
        status=EpisodeStatus.OK,
        termination=Termination.STOP,
        segments=(segment,),
        reward_events=reward_events,
        model=model,
        harness=HarnessMeta(name="verl_agent_loop", version=harness_version),
        env=env,
        gateway=gateway,
    )


def _batch_rows(value: Any, field_name: str) -> tuple[tuple[Any, ...], ...]:
    """Read a ``[B, ...]`` batch array via ``.tolist()`` when present, else as nested sequences."""
    materialized: Any = value
    tolist = getattr(value, "tolist", _MISSING)
    if tolist is not _MISSING and callable(tolist):
        materialized = tolist()
    rows = _as_sequence(materialized, field_name)
    out_rows: list[tuple[Any, ...]] = []
    for i, row in enumerate(rows):
        out_rows.append(_as_sequence(row, f"{field_name}[{i}]"))
    return tuple(out_rows)


def _batch_vector(value: Any, field_name: str) -> tuple[Any, ...]:
    """Read a ``[B]`` batch vector via ``.tolist()`` when present, else as a sequence."""
    materialized: Any = value
    tolist = getattr(value, "tolist", _MISSING)
    if tolist is not _MISSING and callable(tolist):
        materialized = tolist()
    return _as_sequence(materialized, field_name)


def samples_from_rollout_batch(
    batch: Mapping[str, Any],
    non_tensor: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Unpack verl's padded rollout batch into ``AgentLoopOutput``-shaped dicts.

    Torch-free: every array is read via ``.tolist()`` when present, else used as
    nested sequences.  Refusals (:class:`VerlAdapterRefusal`) name both counts.
    """
    for required in ("prompts", "responses", "response_mask", "attention_mask"):
        if required not in batch:
            _refuse(f"rollout batch.{required}: missing field")

    prompts = _batch_rows(batch["prompts"], "prompts")
    responses = _batch_rows(batch["responses"], "responses")
    response_mask = _batch_rows(batch["response_mask"], "response_mask")
    attention_mask = _batch_rows(batch["attention_mask"], "attention_mask")

    if not prompts:
        _refuse("rollout batch.prompts: empty batch (0 of 0 rows)")
    b = len(prompts)
    p = len(prompts[0])
    r = len(responses[0]) if responses else 0

    for name, rows in (
        ("prompts", prompts),
        ("responses", responses),
        ("response_mask", response_mask),
        ("attention_mask", attention_mask),
    ):
        if len(rows) != b:
            _refuse(
                f"rollout batch.{name}: {len(rows)} of {b} rows "
                "(batch sizes must agree across keys)"
            )

    for i in range(b):
        if len(prompts[i]) != p:
            _refuse(
                f"rollout batch.prompts[{i}]: {len(prompts[i])} of {p} columns "
                "(prompt width must be constant across rows)"
            )
        if len(responses[i]) != r:
            _refuse(
                f"rollout batch.responses[{i}]: {len(responses[i])} of {r} columns "
                "(response width must be constant across rows)"
            )
        if len(response_mask[i]) != r:
            _refuse(
                f"rollout batch.response_mask[{i}]: {len(response_mask[i])} of {r} columns "
                "(response_mask width must equal responses width)"
            )
        if len(attention_mask[i]) != p + r:
            _refuse(
                f"rollout batch.attention_mask[{i}]: {len(attention_mask[i])} of {p + r} columns "
                "(attention_mask width must equal P + R)"
            )

    logprobs_rows: tuple[tuple[Any, ...], ...] | None = None
    if "rollout_log_probs" in batch and batch["rollout_log_probs"] is not None:
        logprobs_rows = _batch_rows(batch["rollout_log_probs"], "rollout_log_probs")
        if len(logprobs_rows) != b:
            _refuse(
                f"rollout batch.rollout_log_probs: {len(logprobs_rows)} of {b} rows "
                "(batch sizes must agree across keys)"
            )
        for i in range(b):
            if len(logprobs_rows[i]) != r:
                _refuse(
                    f"rollout batch.rollout_log_probs[{i}]: {len(logprobs_rows[i])} of {r} columns "
                    "(rollout_log_probs width must equal responses width)"
                )

    rm_rows: tuple[tuple[Any, ...], ...] | None = None
    if "rm_scores" in batch and batch["rm_scores"] is not None:
        rm_rows = _batch_rows(batch["rm_scores"], "rm_scores")
        if len(rm_rows) != b:
            _refuse(
                f"rollout batch.rm_scores: {len(rm_rows)} of {b} rows "
                "(batch sizes must agree across keys)"
            )
        for i in range(b):
            if len(rm_rows[i]) != r:
                _refuse(
                    f"rollout batch.rm_scores[{i}]: {len(rm_rows[i])} of {r} columns "
                    "(rm_scores width must equal responses width)"
                )

    num_turns: tuple[Any, ...] | None = None
    if non_tensor is not None and "__num_turns__" in non_tensor:
        num_turns = _batch_vector(non_tensor["__num_turns__"], "non_tensor.__num_turns__")
        if len(num_turns) != b:
            _refuse(
                f"non_tensor.__num_turns__: {len(num_turns)} of {b} rows "
                "(batch sizes must agree across keys)"
            )

    samples: list[dict[str, Any]] = []
    for i in range(b):
        prompt_row = attention_mask[i][:p]
        resp_row = attention_mask[i][p : p + r]

        # Left-padded prompt: zeros, then ones. The violation is a 0 after a 1.
        seen_one = False
        for j, m in enumerate(prompt_row):
            if not _is_number(m):
                _refuse(
                    f"rollout batch.attention_mask[{i}][{j}]: expected a number, "
                    f"got {m!r} ({j} of {p} prompt columns checked)"
                )
            if float(m) > 0:
                seen_one = True
            elif seen_one:
                _refuse(
                    f"rollout batch.attention_mask[{i}]: prompt mask is not left-padded "
                    f"(a 1 followed by a 0 at column {j}; {p} prompt columns checked)"
                )

        # Right-padded response: ones, then zeros. The violation is a 1 after a 0.
        seen_zero = False
        for j, m in enumerate(resp_row):
            if not _is_number(m):
                _refuse(
                    f"rollout batch.attention_mask[{i}][{p + j}]: expected a number, "
                    f"got {m!r} ({j} of {r} response columns checked)"
                )
            if float(m) <= 0:
                seen_zero = True
            elif seen_zero:
                _refuse(
                    f"rollout batch.attention_mask[{i}]: response mask is not right-padded "
                    f"(a 0 followed by a 1 at column {p + j}; {r} response columns checked)"
                )

        prompt_ids = [tok for tok, m in zip(prompts[i], prompt_row) if float(m) > 0]
        valid = [j for j, m in enumerate(resp_row) if float(m) > 0]
        response_ids = [responses[i][j] for j in valid]
        mask_values = [response_mask[i][j] for j in valid]

        if logprobs_rows is not None:
            response_logprobs: list[Any] | None = [logprobs_rows[i][j] for j in valid]
        else:
            response_logprobs = None

        reward_score: Any = None
        if rm_rows is not None and valid:
            reward_score = rm_rows[i][valid[-1]]

        if num_turns is not None:
            turns = int(num_turns[i])
        else:
            turns = 0

        samples.append(
            {
                "prompt_ids": prompt_ids,
                "response_ids": response_ids,
                "response_mask": mask_values,
                "response_logprobs": response_logprobs,
                "reward_score": reward_score,
                "num_turns": turns,
            }
        )
    return samples


def require_pinned_verl(
    *,
    pin: str = VERL_PIN,
    distribution: str = "verl",
    read_text: Callable[[str, str], str | None] | None = None,
) -> str:
    """Verify the installed verl distribution is pinned to ``pin``; return its commit id.

    Reads the installed distribution's PEP 610 ``direct_url.json``.  Refuses when
    verl is not installed, when the file is absent (a PyPI/wheel install has no
    commit), when ``vcs_info.commit_id`` is absent, or when it does not equal
    ``pin`` (both SHAs are named).
    """
    if read_text is None:

        def _default_read_text(dist_name: str, filename: str) -> str | None:
            try:
                dist = metadata.distribution(dist_name)
            except metadata.PackageNotFoundError:
                _refuse(f"require_pinned_verl: distribution {distribution!r} is not installed")
            return dist.read_text(filename)

        reader: Callable[[str, str], str | None] = _default_read_text
    else:
        reader = read_text

    text = reader(distribution, "direct_url.json")
    if text is None:
        _refuse(
            f"require_pinned_verl: {distribution} has no direct_url.json "
            "(a PyPI/wheel install carries no commit); expected a pinned VCS install"
        )

    try:
        payload: Any = json.loads(text)
    except (TypeError, ValueError) as exc:
        _refuse(f"require_pinned_verl: {distribution} direct_url.json is not valid JSON: {exc}")

    if not isinstance(payload, Mapping):
        _refuse(
            f"require_pinned_verl: {distribution} direct_url.json: expected a JSON object, "
            f"got {type(payload).__name__}"
        )
    vcs_info = payload.get("vcs_info")
    if not isinstance(vcs_info, Mapping):
        _refuse(
            f"require_pinned_verl: {distribution} direct_url.json: missing vcs_info "
            f"(no vcs_info.commit_id: not a VCS install)"
        )
    commit_id = vcs_info.get("commit_id")
    if not isinstance(commit_id, str) or len(commit_id) == 0:
        _refuse(f"require_pinned_verl: {distribution} direct_url.json: missing vcs_info.commit_id")
    if commit_id != pin:
        _refuse(
            f"require_pinned_verl: installed verl commit {commit_id!r} != pinned commit {pin!r}"
        )
    return commit_id
