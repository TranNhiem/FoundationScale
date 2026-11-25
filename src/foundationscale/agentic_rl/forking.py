"""Resume a forked episode from a restored environment (ARL-18 slice 4).

A fork continues an episode from a response boundary: the environment comes back from a
committed snapshot (``TaskSandboxEnvironment.restore``) and the loop comes back from the
:class:`~foundationscale.agentic_rl.harness.base.EpisodeCheckpoint` captured at the same
boundary (``NativeToolLoop.run_episode(resume=...)``). :class:`ForkBoundary` binds the two
together with the trace identity they were taken under; :func:`resume_fork` checks that an
``fs_trace.ForkPoint`` names exactly that boundary before running anything.

Every check is exact: the parent trace and branch ids, the number of kept turns and the
token prefix itself (compared as token ids -- never decoded and re-encoded). A mismatch is a
:class:`ForkRefusal`; there is no prompt-only restart and no replay of historical tool
calls. The restored environment's signature is checked against the snapshot by
``restore()`` itself; here the boundary's recorded signature must also equal the snapshot's.

Credit: the forked outcome is a complete episode (kept turns plus new ones) scored by the
normal verifier, but the kept generated turns carry ``credited=False`` (ARL-19), so
``Trajectory.loss_mask()`` -- and therefore the trainer -- trains only on the tokens this
attempt sampled after the fork. The kept turns keep their provenance and logprobs.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from foundationscale.agentic_rl.fs_trace import ForkPoint
from foundationscale.agentic_rl.harness.base import (
    EpisodeBudget,
    EpisodeCheckpoint,
    EpisodeOutcome,
    EpisodeTask,
    GenerationClient,
)

if TYPE_CHECKING:
    from foundationscale.agentic_rl.envs.sandbox_task import TaskSandboxEnvironment
    from foundationscale.agentic_rl.sandbox.snapshot_store import SnapshotStore


class ForkRefusal(ValueError):
    """A fork whose point and boundary do not describe the same state."""


class ResumableHarness(Protocol):
    async def run_episode(
        self,
        task: EpisodeTask,
        *,
        client: GenerationClient,
        env: TaskSandboxEnvironment,
        budget: EpisodeBudget,
        resume: EpisodeCheckpoint | None = None,
    ) -> EpisodeOutcome: ...


@dataclass(frozen=True)
class ForkBoundary:
    """A response boundary that can be forked: the environment snapshot and the loop
    checkpoint taken there, with the trace identity they belong to."""

    snapshot_id: str
    env_signature: str
    checkpoint: EpisodeCheckpoint
    parent_trace_id: str
    parent_branch_id: str


def check_fork(point: ForkPoint, boundary: ForkBoundary) -> None:
    """Refuse unless ``point`` names exactly ``boundary`` (ids, kept turns, token prefix)."""
    if point.parent_trace_id != boundary.parent_trace_id:
        raise ForkRefusal(
            f"fork: point.parent_trace_id {point.parent_trace_id!r} != boundary "
            f"{boundary.parent_trace_id!r}"
        )
    if point.parent_branch_id != boundary.parent_branch_id:
        raise ForkRefusal(
            f"fork: point.parent_branch_id {point.parent_branch_id!r} != boundary "
            f"{boundary.parent_branch_id!r}"
        )
    if point.branch_id == boundary.parent_branch_id:
        raise ForkRefusal("fork: point.branch_id equals its parent branch: not a fork")
    kept = len(boundary.checkpoint.turns)
    if point.turns_kept != kept:
        raise ForkRefusal(
            f"fork: point.turns_kept={point.turns_kept} but the boundary checkpoint holds "
            f"{kept} trajectory turn(s)"
        )
    prefix = boundary.checkpoint.token_ids
    if point.prompt_token_ids != prefix:
        raise ForkRefusal(
            f"fork: the point's token prefix ({len(point.prompt_token_ids)} ids) differs "
            f"from the boundary checkpoint's ({len(prefix)} ids): the model would continue "
            f"from a context the environment was not snapshotted in"
        )


async def resume_fork(
    point: ForkPoint,
    boundary: ForkBoundary,
    *,
    harness: ResumableHarness,
    task: EpisodeTask,
    client: GenerationClient,
    env: TaskSandboxEnvironment,
    budget: EpisodeBudget,
    store: SnapshotStore,
) -> EpisodeOutcome:
    """Restore ``env`` from the boundary's snapshot and continue the episode from its
    checkpoint under ``point.branch_id``. ``env`` must be fresh (``restore`` replaces
    ``start``); the caller still owns ``verify()`` and ``close()``."""
    check_fork(point, boundary)
    if task.uid != boundary.checkpoint.task_uid:
        raise ForkRefusal(
            f"fork: task {task.uid!r} is not the checkpoint's task {boundary.checkpoint.task_uid!r}"
        )
    recorded = store.get(boundary.snapshot_id).env_signature
    if recorded != boundary.env_signature:
        raise ForkRefusal(
            f"fork: snapshot {boundary.snapshot_id!r} records signature {recorded!r}, the "
            f"boundary {boundary.env_signature!r}"
        )
    await env.restore(store, boundary.snapshot_id)
    outcome = await harness.run_episode(
        task, client=client, env=env, budget=budget, resume=boundary.checkpoint
    )
    kept = len(boundary.checkpoint.turns)
    prefix = outcome.trajectory_turns[:kept]
    if prefix != boundary.checkpoint.turns:
        raise ForkRefusal(
            "fork: the resumed outcome does not start with the checkpoint's turns: the kept "
            "prefix cannot be identified, so credit could not be withheld from it"
        )
    # ARL-19: the kept prefix was sampled by the PARENT episode; it stays generated (real
    # provenance and logprobs) but gets no policy-gradient credit in this attempt.
    uncredited = tuple(
        dataclasses.replace(turn, credited=False) if turn.generated else turn for turn in prefix
    )
    return dataclasses.replace(
        outcome,
        trajectory_turns=uncredited + outcome.trajectory_turns[kept:],
        metadata={
            **outcome.metadata,
            "fork_parent_trace_id": point.parent_trace_id,
            "fork_parent_branch_id": point.parent_branch_id,
            "fork_branch_id": point.branch_id,
            "fork_turns_kept": str(point.turns_kept),
        },
    )
