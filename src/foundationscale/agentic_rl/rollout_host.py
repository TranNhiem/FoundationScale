"""``RolloutHost``: the ``RLTrainConfig.rollout_source`` realisation for agentic RL.

Draws ``tasks_per_step`` tasks from a ``TaskSource``, runs ``group_size``
attempts of each concurrently (one fresh environment instance per attempt,
created/started/closed every time), scores each with a ``RewardFn``, and
flattens the results into the ``ExperienceBatch`` the trainer (slice 4a,
``foundationscale.rl.trainer``) consumes via ``rollout(step)``. ``publish``
realises the OPTIONAL sibling half of that same interface: pushing the
just-trained policy's weights to the generation fleet after an optimizer step.

Sync wrapper over an async core: the trainer is synchronous, so
``rollout(step)`` opens its own event loop with ``asyncio.run`` around
``rollout_async``; nothing in this module assumes a loop is already running
except ``rollout_async`` and ``_run_episode`` themselves.
"""

from __future__ import annotations

import asyncio
import json
import sys
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from foundationscale.agentic_rl.contracts import Termination, Trajectory, flatten
from foundationscale.agentic_rl.engines import EngineInfraError
from foundationscale.agentic_rl.engines.fleet import EngineFleet
from foundationscale.agentic_rl.envs.base import EnvBackend, EnvInfraError, EnvSpec
from foundationscale.agentic_rl.harness.base import (
    EpisodeBudget,
    EpisodeOutcome,
    EpisodeTask,
    GenerationClient,
    HarnessAdapter,
    to_trajectory,
)
from foundationscale.agentic_rl.rewards.base import RewardFn
from foundationscale.agentic_rl.tasks import TaskSource
from foundationscale.agentic_rl.weight_sync import DiskWeightSync
from foundationscale.gates.agentic_gates import RolloutGateContext, WeightSyncGateContext
from foundationscale.gates.core import REGISTRY, Lifecycle, run_event
from foundationscale.rl.interfaces import ExperienceBatch

__all__ = (
    "RolloutHost",
    "RolloutHostRefusal",
)

_INFRA_EXCEPTIONS: tuple[type[Exception], ...] = (EnvInfraError, EngineInfraError)


def _described(value: object) -> str:
    return f"{type(value).__name__} {value!r}"


class RolloutHostRefusal(ValueError):
    """A config error constructing or calling ``RolloutHost`` -- never an episode's
    own outcome, which is handled per-episode (see the module docstring).
    """


def _real_int_at_least(value: object, minimum: int, *, where: str, name: str) -> int:
    if type(value) is not int or value < minimum:
        raise RolloutHostRefusal(
            f"{where}: field {name!r} is {_described(value)}: it must be a real int "
            f">= {minimum} (bool excluded)"
        )
    return value


def _zero_turn_infra_trajectory(
    task: EpisodeTask, *, session_id: str, harness_name: str, reason: str, step: int
) -> Trajectory:
    """An honest, EMPTY ``Trajectory`` for an episode that infra-failed before the
    harness ever returned an ``EpisodeOutcome`` (e.g. the environment failed to
    start).

    ``RolloutHost`` owns no tokenizer and nothing was ever rendered -- not even a
    prompt -- so there is no real text on either side to report. Rather than
    inventing placeholder tokens for text nothing tokenised (a byte-encoding of
    the seed messages that could be mistaken for a real tokenizer's output),
    ``prompt_turns`` and ``turns`` are both ``()``: ``contracts.Trajectory``
    admits exactly this shape under ``Termination.INFRA`` (see its docstring),
    and ``flatten`` keeps the row with empty per-token columns.
    """
    return Trajectory(
        uid=task.uid,
        session_id=session_id,
        harness=harness_name,
        prompt_turns=(),
        turns=(),
        reward=None,
        abstention_reason=reason,
        termination=Termination.INFRA,
        policy_version_min=step,
        policy_version_max=step,
    )


@dataclass(frozen=True)
class RolloutHost:
    """Generates and scores one step's experience batch for the agentic RL trainer.

    Exactly one of ``fleet``/``client_factory`` must be given: ``fleet`` for a real
    SGLang deployment (round-robin by session index), ``client_factory`` to inject
    a scripted/fake ``GenerationClient`` in tests. ``weight_sync`` is optional --
    ``publish`` is a no-op when it is ``None``, matching the trainer-side
    interface's "optional ``publish``" contract. ``sharding`` must match the
    trainer's OWN sharding mode (``RLTrainConfig.sharding``): ``publish`` passes
    it straight to ``rl.distributed.save_checkpoint``, which needs it to gather a
    sharded model correctly.

    ``gates`` (default ``True``) runs the slice-6 agentic gates
    (``foundationscale.gates.agentic_gates``) through
    ``foundationscale.gates.core.run_event``: ``Lifecycle.ROLLOUT`` at the end of
    ``rollout_async``, over the just-flattened batch, and ``Lifecycle.WEIGHT_SYNC``
    in ``publish``, after a complete push. A blocking report raises
    ``RolloutHostRefusal`` naming the rendered report; set ``gates=False`` to run
    with the gate layer off (tests exercise both). ``declared_max_infra_rate``
    (``None`` by default -- no declared bound, so the rollout-abstention gate
    abstains rather than inventing one) and ``declared_max_lag`` (``1`` for S0)
    are the two declared thresholds those gates read; see
    ``config.py``'s ``rollout.max_infra_rate`` / ``engine.max_policy_lag``.
    """

    tasks: TaskSource
    tasks_per_step: int
    group_size: int
    harness: HarnessAdapter
    env_backend: EnvBackend
    env_spec: EnvSpec
    reward: RewardFn
    budget: EpisodeBudget
    max_concurrency: int
    fleet: EngineFleet | None = None
    client_factory: Callable[[int], GenerationClient] | None = None
    group_by_harness: bool = False
    weight_sync: DiskWeightSync | None = None
    sharding: str = "none"
    gates: bool = True
    declared_max_infra_rate: float | None = None
    declared_max_lag: int = 1

    def __post_init__(self) -> None:
        where = "RolloutHost"
        _real_int_at_least(self.tasks_per_step, 1, where=where, name="tasks_per_step")
        _real_int_at_least(self.group_size, 2, where=where, name="group_size")
        _real_int_at_least(self.max_concurrency, 1, where=where, name="max_concurrency")
        if type(self.group_by_harness) is not bool:
            raise RolloutHostRefusal(
                f"{where}: field 'group_by_harness' is {_described(self.group_by_harness)}: "
                f"it must be a real bool (type(x) is bool) -- 1 is not True"
            )
        if (self.fleet is None) == (self.client_factory is None):
            raise RolloutHostRefusal(
                f"{where}: field 'fleet' is {_described(self.fleet)} and field "
                f"'client_factory' is {_described(self.client_factory)}: exactly one of "
                f"the two must be given -- neither leaves no client to generate with, and "
                f"both leaves an ambiguous choice this plane refuses to make for the caller"
            )
        if self.weight_sync is not None and not isinstance(self.weight_sync, DiskWeightSync):
            raise RolloutHostRefusal(
                f"{where}: field 'weight_sync' is {_described(self.weight_sync)}, not a "
                f"DiskWeightSync"
            )
        if self.sharding not in ("none", "ddp", "fsdp"):
            raise RolloutHostRefusal(
                f"{where}: field 'sharding' is {_described(self.sharding)}: it must be one "
                f"of 'none', 'ddp', 'fsdp' -- rl.distributed.save_checkpoint needs this to "
                f"gather a sharded model correctly, and an undeclared value would guess"
            )
        if type(self.gates) is not bool:
            raise RolloutHostRefusal(
                f"{where}: field 'gates' is {_described(self.gates)}: it must be a real "
                f"bool (type(x) is bool) -- 1 is not True"
            )
        if self.declared_max_infra_rate is not None:
            rate = self.declared_max_infra_rate
            if (
                type(rate) is bool
                or not isinstance(rate, (int, float))
                or not (0.0 <= float(rate) <= 1.0)
            ):
                raise RolloutHostRefusal(
                    f"{where}: field 'declared_max_infra_rate' is {_described(rate)}: it "
                    f"must be None or a real number in [0.0, 1.0] -- an infra rate is a "
                    f"fraction of a batch's rows and cannot exceed 1.0"
                )
        _real_int_at_least(self.declared_max_lag, 0, where=where, name="declared_max_lag")

    def _client_for(self, session_index: int) -> GenerationClient:
        if self.client_factory is not None:
            return self.client_factory(session_index)
        assert self.fleet is not None  # __post_init__ guarantees exactly one is set
        return self.fleet.client_for(session_index)

    async def _run_episode(
        self,
        task: EpisodeTask,
        session_id: str,
        session_index: int,
        step: int,
        semaphore: asyncio.Semaphore,
    ) -> Trajectory:
        async with semaphore:
            client = self._client_for(session_index)
            instance = self.env_backend.create(self.env_spec, instance_id=session_id)
            outcome: EpisodeOutcome | None = None
            try:
                await instance.start()
                outcome = await self.harness.run_episode(
                    task, client=client, env=instance, budget=self.budget
                )
                reward_value: float | None
                abstention: str | None
                extra_metadata: dict[str, str] | None = None
                if outcome.termination is Termination.INFRA:
                    reward_value, abstention = None, outcome.abstention_reason
                else:
                    verdict = await self.reward.score(task, outcome)
                    reward_value = verdict.value
                    if reward_value is None:
                        # ABSTAINED: verdict.reason is the abstention's own name.
                        abstention = verdict.reason
                    else:
                        # MEASURED: verdict.reason (if any) is an informational note
                        # on a real score, never an abstention -- promoting it to
                        # abstention_reason here would make Trajectory.__post_init__
                        # read a scored attempt as one with nothing scored (and raise
                        # TrajectoryRefusal for carrying both). The note still
                        # survives, just in metadata rather than abstention_reason.
                        abstention = None
                        if verdict.reason is not None:
                            extra_metadata = {"reward_note": verdict.reason}
                return to_trajectory(
                    outcome,
                    uid=task.uid,
                    session_id=session_id,
                    harness=self.harness.name,
                    reward=reward_value,
                    abstention_reason=abstention,
                    versions=(step, step),
                    extra_metadata=extra_metadata,
                )
            except _INFRA_EXCEPTIONS as exc:
                reason = f"infra:{type(exc).__name__}"
                if outcome is not None:
                    # run_episode() returned a real EpisodeOutcome before reward
                    # scoring raised: reuse its REAL prompt_turns/turns rather
                    # than the no-outcome synthetic fallback below. Guaranteed
                    # non-empty: reward scoring only runs when
                    # outcome.termination is not INFRA, which requires at
                    # least one generated turn (contracts.Trajectory).
                    return Trajectory(
                        uid=task.uid,
                        session_id=session_id,
                        harness=self.harness.name,
                        prompt_turns=outcome.prompt_turns,
                        turns=outcome.trajectory_turns,
                        reward=None,
                        abstention_reason=reason,
                        termination=Termination.INFRA,
                        policy_version_min=step,
                        policy_version_max=step,
                    )
                return _zero_turn_infra_trajectory(
                    task,
                    session_id=session_id,
                    harness_name=self.harness.name,
                    reason=reason,
                    step=step,
                )
            finally:
                await instance.close()

    async def rollout_async(self, step: int) -> ExperienceBatch:
        """The async core of ``rollout``: draw this step's tasks, run every attempt
        concurrently under ``max_concurrency``, flatten the results, and -- unless
        ``gates=False`` -- run the ``Lifecycle.ROLLOUT`` gates over the flattened
        batch before returning it.
        """
        if type(step) is not int or step < 0:
            raise RolloutHostRefusal(
                f"RolloutHost.rollout: parameter 'step' is {_described(step)}: it must be "
                f"a real int >= 0"
            )
        tasks = self.tasks.batch(step, self.tasks_per_step)
        semaphore = asyncio.Semaphore(self.max_concurrency)
        episodes = []
        session_index = 0
        for task in tasks:
            for replica in range(self.group_size):
                session_id = f"{task.uid}#{replica}"
                episodes.append(self._run_episode(task, session_id, session_index, step, semaphore))
                session_index += 1
        trajectories = await asyncio.gather(*episodes)
        batch = flatten(trajectories, group_by_harness=self.group_by_harness)
        print(rollout_summary(step, trajectories), file=sys.stderr, flush=True)
        if self.gates:
            gate_ctx = RolloutGateContext(
                batch_columns=batch.columns,
                declared_max_infra_rate=self.declared_max_infra_rate,
                step=step,
            )
            # missing_ctx="report-skip": PromptBytesGate is also registered for
            # Lifecycle.ROLLOUT but this call site has no PromptBytesGateContext to
            # give it yet (the prompt-bytes observation is a harness-layer concern,
            # not wired here) -- it must surface as a declared SKIP, never a
            # blocking "unwired" ERROR, every time this event runs.
            report = run_event(
                REGISTRY,
                Lifecycle.ROLLOUT,
                {RolloutGateContext: gate_ctx},
                missing_ctx="report-skip",
            )
            if not report.ok:
                raise RolloutHostRefusal(
                    f"RolloutHost.rollout: step {step}: ROLLOUT gates blocked:\n{report.render()}"
                )
        return batch

    def rollout(self, step: int) -> ExperienceBatch:
        """Synchronous entry point the trainer calls: runs ``rollout_async`` to
        completion in its own event loop.
        """
        return asyncio.run(self.rollout_async(step))

    def publish(self, model: Any, tokenizer: Any, ctx: Any, step: int) -> None:
        """Push the policy trained through ``step`` to the generation fleet.

        No-op when ``weight_sync`` is ``None``. The save itself is built HERE,
        around the REAL ``model``/``tokenizer``/``ctx`` this call receives --
        never a closure ``weight_sync.save_fn`` would have had to capture before
        the policy existed, which no caller constructing this host before
        ``RLTrainer.run()`` loads the model could ever build (``weight_sync.
        DiskWeightSync.save_fn`` exists only for ITS OWN ``sync()`` convenience
        path, a non-distributed, non-``publish`` caller; see that module's
        docstring). ``rl.distributed.save_checkpoint`` is imported lazily here,
        not at module scope, so importing this module (unconditionally, by
        ``cli.py``, even for ``--dry-run``) never imports torch.

        ``save_checkpoint`` is COLLECTIVE and is called on EVERY rank,
        unconditionally; only ``ctx.rank == 0`` additionally pushes to the
        fleet -- see ``weight_sync.DiskWeightSync``'s module docstring for why
        ``sync`` itself must not be called on just one rank here. A push that
        does not fully succeed (a fleet member unreachable, a reload that
        reported failure) raises -- CONTINUING training on an unreplaced
        checkpoint would serve the OLD policy's rollouts as if they were
        on-policy, silently, for every step after this one; the trainer's own
        total exit-code boundary (``agentic_rl.cli._run``) adjudicates this
        RED, the correct verdict for an infra fault discovered mid-run.
        """
        if type(step) is not int or step < 0:
            raise RolloutHostRefusal(
                f"RolloutHost.publish: parameter 'step' is {_described(step)}: it must be "
                f"a real int >= 0"
            )
        if self.weight_sync is None:
            return
        from foundationscale.rl.distributed import save_checkpoint

        target_step = step + 1
        path = self.weight_sync.path_for(target_step)
        save_checkpoint(model, tokenizer, path, ctx, sharding=self.sharding, step=target_step)
        if ctx.rank == 0:
            report = self.weight_sync.push(path)
            if not report.complete:
                raise RolloutHostRefusal(
                    f"RolloutHost.publish: step {step}: the weight push to the fleet was "
                    f"NOT complete ({len(report.transferred)} of {len(report.offered)} "
                    f"server(s) transferred, failed_ranks={report.failed_ranks}): "
                    f"continuing would serve the stale, pre-step-{target_step} policy on "
                    f"every subsequent rollout without saying so"
                )
            if self.gates:
                # rollout_policy_versions=(): RolloutHost does not yet track which
                # policy version each in-flight rollout session is running (that
                # instrumentation is a later slice's job) -- an honest empty
                # sequence, never a fabricated one. StalenessGate still reaches a
                # real, scoped PASS from is_stale alone when the sequence is empty
                # (see its own docstring); it never fabricates a per-session lag
                # claim nobody measured.
                gate_ctx = WeightSyncGateContext(
                    offered=len(report.offered),
                    failed=len(report.failed_ranks),
                    # report.is_stale is DERIVED by DiskWeightSync.push from the
                    # push just having completed synchronously (S0) -- it is
                    # never separately measured here, by the gate, or anywhere
                    # else; see weight_sync.DiskWeightSync.push's own docstring.
                    is_stale=report.is_stale,
                    seconds=report.seconds,
                    step=step,
                    policy_version=target_step,
                    rollout_policy_versions=(),
                    declared_max_lag=self.declared_max_lag,
                    parity=None,
                    declared_parity_atol=None,
                )
                gate_report = run_event(REGISTRY, Lifecycle.WEIGHT_SYNC, gate_ctx)
                if not gate_report.ok:
                    raise RolloutHostRefusal(
                        f"RolloutHost.publish: step {step}: WEIGHT_SYNC gates blocked:\n"
                        f"{gate_report.render()}"
                    )


def rollout_summary(step: int, trajectories: tuple[Trajectory, ...] | list[Trajectory]) -> str:
    """One JSON line describing what a rollout step produced, for the run log.

    Counts are over every trajectory; reward statistics are over MEASURED rewards
    only (an abstained ``None`` is counted under ``abstained``, never averaged in
    as 0.0), and are ``None`` when no reward was measured at all.
    """
    measured = [t.reward for t in trajectories if t.reward is not None]
    reasons = Counter(t.abstention_reason for t in trajectories if t.reward is None)
    notes = Counter(t.metadata.get("reward_note") for t in trajectories if t.reward is not None)
    groups: dict[str, list[float]] = {}
    for t in trajectories:
        if t.reward is not None:
            groups.setdefault(t.uid, []).append(t.reward)
    payload = {
        "rollout_step": step,
        "trajectories": len(trajectories),
        "abstained": sum(reasons.values()),
        "abstention_reasons": dict(reasons),
        "terminations": dict(Counter(t.termination.value for t in trajectories)),
        "reward_min": min(measured) if measured else None,
        "reward_mean": sum(measured) / len(measured) if measured else None,
        "reward_max": max(measured) if measured else None,
        "groups_with_spread": sum(1 for v in groups.values() if len(v) > 1 and max(v) > min(v)),
        "groups": len(groups),
        "reward_notes": {str(k): v for k, v in notes.items() if k is not None},
        "supervised_tokens": sum(t.supervised_token_count for t in trajectories),
    }
    return "[agentic-rl:rollout] " + json.dumps(payload, sort_keys=True)
