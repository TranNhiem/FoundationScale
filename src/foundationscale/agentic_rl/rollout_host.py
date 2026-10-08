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
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from foundationscale.agentic_rl.contracts import SegmentKind, Termination, Trajectory, Turn, flatten
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


def _synthetic_infra_trajectory(
    task: EpisodeTask, *, session_id: str, harness_name: str, reason: str, step: int
) -> Trajectory:
    """An honest placeholder ``Trajectory`` for an episode that infra-failed before
    the harness ever returned an ``EpisodeOutcome`` (e.g. the environment failed to
    start).

    ``contracts.Trajectory`` refuses an empty ``prompt_turns``/``turns`` tuple under
    EVERY termination, including INFRA, and ``RolloutHost`` owns no tokenizer to
    render even a real prompt in this situation. The turns built here NEVER claim
    to be chat-template-rendered or policy-sampled (every one has
    ``generated=False``): their token ids are a DECLARED raw UTF-8 byte encoding of
    real text -- each seed message's own role and content, and the infra failure's
    own reason string -- never a guess, a default, or readable as a real
    tokenizer's output.
    """
    prompt_turns: list[Turn] = []
    for index, message in enumerate(task.messages):
        role = message["role"]
        kind = SegmentKind.SYSTEM if role == "system" else SegmentKind.USER
        raw = f"{role}:{message['content']}".encode()
        prompt_turns.append(
            Turn(index=index, kind=kind, token_ids=tuple(raw), generated=False, logprobs=None)
        )
    response_turn = Turn(
        index=len(prompt_turns),
        kind=SegmentKind.OBSERVATION,
        token_ids=tuple(reason.encode()),
        generated=False,
        logprobs=None,
    )
    return Trajectory(
        uid=task.uid,
        session_id=session_id,
        harness=harness_name,
        prompt_turns=tuple(prompt_turns),
        turns=(response_turn,),
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
    interface's "optional ``publish``" contract.
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
                return _synthetic_infra_trajectory(
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
        concurrently under ``max_concurrency``, and flatten the results.
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
        return flatten(trajectories, group_by_harness=self.group_by_harness)

    def rollout(self, step: int) -> ExperienceBatch:
        """Synchronous entry point the trainer calls: runs ``rollout_async`` to
        completion in its own event loop.
        """
        return asyncio.run(self.rollout_async(step))

    def publish(self, _model: Any, _tokenizer: Any, ctx: Any, step: int) -> None:
        """Push the policy trained through ``step`` to the generation fleet.

        No-op when ``weight_sync`` is ``None``. ``_model``/``_tokenizer`` are
        unused by this realisation (``weight_sync.save_fn`` is a closure the
        caller already built around them) and kept only so this method matches
        the trainer-side ``publish(model, tokenizer, ctx, step)`` interface.

        ``save_fn`` is COLLECTIVE and is called on EVERY rank, unconditionally;
        only ``ctx.rank == 0`` additionally pushes to the fleet -- see
        ``weight_sync.DiskWeightSync``'s module docstring for why ``sync`` itself
        must not be called on just one rank here.
        """
        if type(step) is not int or step < 0:
            raise RolloutHostRefusal(
                f"RolloutHost.publish: parameter 'step' is {_described(step)}: it must be "
                f"a real int >= 0"
            )
        if self.weight_sync is None:
            return
        target_step = step + 1
        path = self.weight_sync.path_for(target_step)
        self.weight_sync.save_fn(path)
        if ctx.rank == 0:
            self.weight_sync.push(path)
