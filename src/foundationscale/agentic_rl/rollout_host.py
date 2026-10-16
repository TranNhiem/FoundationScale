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
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
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
from foundationscale.agentic_rl.stability import (
    CollapseDetected,
    StabilityGuard,
    rollout_stats,
    verdict_line,
)
from foundationscale.agentic_rl.tasks import TaskSource
from foundationscale.agentic_rl.weight_sync import DiskWeightSync
from foundationscale.gates.agentic_gates import RolloutGateContext, WeightSyncGateContext
from foundationscale.gates.core import REGISTRY, Lifecycle, run_event
from foundationscale.rl.interfaces import ExperienceBatch
from foundationscale.rl.weightsync import SyncReport

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


def _all_engine_infra(trajectories: Sequence[Trajectory]) -> bool:
    """True iff ``trajectories`` is non-empty and EVERY row in it is an ENGINE infra
    failure -- the one signature of a hung (or dead) generation engine that
    ``engine_recovery_retries`` is allowed to act on.

    "Engine" here is read off the row's own ``abstention_reason``: the harness's
    catch of a mid-episode engine fault names the kind
    (``"infra:engine_<kind>"``, e.g. ``"infra:engine_timeout"``), and this
    module's own outer catch of a fault that reached it before any outcome names
    the exception type (``"infra:EngineInfraError"``). Both are engine faults;
    anything else -- an empty sequence, an env failure
    (``"infra:EnvInfraError"``), a reward abstention, a measured row -- is NOT
    this shape and must not trigger a fleet restart.
    """
    if not trajectories:
        return False
    for trajectory in trajectories:
        if trajectory.termination is not Termination.INFRA:
            return False
        reason = trajectory.abstention_reason
        if not isinstance(reason, str):
            return False
        if not (reason.startswith("infra:engine_") or reason == "infra:EngineInfraError"):
            return False
    return True


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

    ``engine_recovery_retries`` (default ``0`` -- recovery OFF) enables ENGINE
    hang recovery in ``rollout_async``. Why it exists: a vLLM engine can hang
    after an in-place weight reload -- every request times out -- and a 100-step
    run would then die at the infra gate through no fault of the policy.
    Recovery fires only when EVERY trajectory in a step is an engine infra
    failure (see ``_all_engine_infra``): it stops and restarts the fleet,
    re-pushes the last published checkpoint (``_state["last_pushed_path"]``,
    recorded by ``publish``) so the restarted engine serves the CURRENT policy
    rather than the base model, and retries the SAME step with the SAME tasks. A
    failed attempt's summary is replaced by one
    ``[agentic-rl:engine-recovery]`` JSON line to stderr. It never fires on a
    PARTIAL failure (some attempts ran) or on env failures, and if the retries
    are exhausted the step's infra rows still reach the infra gate, which
    refuses as it always did. A positive count needs a ``fleet`` (there must be
    an engine to stop and start back up).
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
    # The multimodal base model dir the trainer actually loaded (e.g. a
    # Qwen3.5-VL checkpoint loaded as a text-only CausalLM) -- None means
    # "do not try to complete the published checkpoint for serving" (either
    # the base model IS already text-only, or policy.servable_publish=False
    # declared the opt-out; see cli.py and
    # foundationscale.agentic_rl.servable). When given, `publish` calls
    # `servable.complete_for_serving` on rank 0 between the collective save
    # and the push -- see `publish`'s own docstring.
    servable_base_model_dir: str | None = None
    # Optional collapse guard (foundationscale.agentic_rl.stability). Fed once per rollout
    # step; after each publish it protects the checkpoint that produced a best-k reward and
    # can stop the run on a collapse.
    guard: StabilityGuard | None = None
    # How many times rollout_async may restart the fleet and re-run a step whose
    # every attempt was an ENGINE infra failure (0 = recovery OFF; see the class
    # docstring). Needs `fleet`.
    engine_recovery_retries: int = 0
    # Private mutable state on this frozen dataclass (the dict's CONTENTS are
    # mutable, never the field binding): "last_pushed_path" is the path of the
    # last checkpoint `publish` fully pushed, so `_recover_engines` can re-push
    # it after restarting the fleet -- a restarted engine must serve the CURRENT
    # policy, never the base model.
    _state: dict[str, Any] = field(default_factory=dict, init=False, repr=False, compare=False)

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
        if self.servable_base_model_dir is not None and (
            not isinstance(self.servable_base_model_dir, str) or not self.servable_base_model_dir
        ):
            raise RolloutHostRefusal(
                f"{where}: field 'servable_base_model_dir' is "
                f"{_described(self.servable_base_model_dir)}: it must be None or a "
                f"non-empty str"
            )
        _real_int_at_least(
            self.engine_recovery_retries, 0, where=where, name="engine_recovery_retries"
        )
        if self.engine_recovery_retries > 0 and self.fleet is None:
            raise RolloutHostRefusal(
                f"{where}: field 'engine_recovery_retries' is {self.engine_recovery_retries} "
                f"but field 'fleet' is None: engine recovery needs a fleet to restart -- a "
                f"caller injecting only 'client_factory' declares no engine to stop and start "
                f"back up"
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
                return _zero_turn_infra_trajectory(
                    task,
                    session_id=session_id,
                    harness_name=self.harness.name,
                    reason=reason,
                    step=step,
                )
            finally:
                await instance.close()

    async def _gather_step(self, step: int, tasks: Sequence[EpisodeTask]) -> list[Trajectory]:
        """Build and run EVERY attempt of ``tasks`` for ``step``, returning one
        ``Trajectory`` per ``tasks_per_step * group_size`` attempt.

        Called once per attempt at the step (once, or once more per engine
        recovery retry) with the SAME task list: session ids (``f"{uid}#{k}"``)
        and session indices are derived purely from ``tasks``/``group_size``, so
        a retried gather produces exactly the rows the first one would have.
        Each call builds a FRESH semaphore -- the counting state of a previous
        call's semaphore is spent on episodes that have since finished/failed,
        and a recovery retry must not inherit it.
        """
        semaphore = asyncio.Semaphore(self.max_concurrency)
        episodes = []
        session_index = 0
        for task in tasks:
            for replica in range(self.group_size):
                session_id = f"{task.uid}#{replica}"
                episodes.append(self._run_episode(task, session_id, session_index, step, semaphore))
                session_index += 1
        return list(await asyncio.gather(*episodes))

    def _recover_engines(self, step: int) -> None:
        """Restart the generation fleet and make it serve the CURRENT policy again.

        A hung engine is not repaired by another request: stop and start it. The
        last checkpoint ``publish`` fully pushed (``_state["last_pushed_path"]``)
        is then re-pushed so the restarted engine serves the current policy
        rather than whatever the base command loads -- a restart with no re-push
        would serve the BASE model and silently put the training step off-policy
        on the next gather. ``weight_sync.mode == "restart"`` pushes by
        stop/starting each server itself around the checkpoint, so it is called
        directly (there is no separate fleet restart to do here); otherwise the
        fleet is restarted through ``self.fleet`` and the checkpoint is
        re-pushed whenever one is both recorded and wanted.

        Synchronous, and ``rollout_async`` runs it in a worker thread
        (``asyncio.to_thread``): ``DiskWeightSync.push`` drives the fleet with its
        own ``asyncio.run()``, which raises if called on a thread whose event loop
        is already running -- measured on GB200, where the first real recovery
        crashed exactly that way -- and a restart blocks for minutes.

        Raises ``RolloutHostRefusal`` rather than returning a report: a recovery
        that cannot restart the fleet, or that re-pushed only PART of it, leaves
        the run generating on a stale/partial fleet -- the same dishonesty
        ``publish`` itself refuses -- so it must not be retried past.
        """
        path = self._state.get("last_pushed_path")
        restart_mode = self.weight_sync is not None and self.weight_sync.mode == "restart"
        if restart_mode and path is None:
            # The fleet's declared command carries the "{model_path}" placeholder, which
            # only cli.py's first launch substitutes; start_all would pass it verbatim.
            raise RolloutHostRefusal(
                f"RolloutHost.rollout: step {step}: engine recovery cannot restart a "
                f"weight_sync mode='restart' fleet before any checkpoint was published -- "
                f"there is no path to substitute into the command's model placeholder"
            )
        report: SyncReport | None
        if restart_mode:
            # restart mode stops/starts each server itself, with the checkpoint.
            assert self.weight_sync is not None and path is not None
            report = self.weight_sync.push(path)
        else:
            fleet = self.fleet
            assert fleet is not None  # __post_init__: engine_recovery_retries > 0 needs one
            fleet.stop_all()
            try:
                fleet.start_all()
            except EngineInfraError as exc:
                raise RolloutHostRefusal(
                    f"RolloutHost.rollout: step {step}: engine recovery failed to restart the "
                    f"fleet: {exc}"
                ) from exc
            report = (
                self.weight_sync.push(path)
                if (self.weight_sync is not None and path is not None)
                else None
            )
        if report is not None and not report.complete:
            raise RolloutHostRefusal(
                f"RolloutHost.rollout: step {step}: engine recovery's re-push of {path} was NOT "
                f"complete ({len(report.transferred)} of {len(report.offered)} server(s) "
                f"transferred, failed_ranks={report.failed_ranks}): the restarted fleet would "
                f"keep serving a stale checkpoint on every subsequent rollout without saying so"
            )

    async def rollout_async(self, step: int) -> ExperienceBatch:
        """The async core of ``rollout``: draw this step's tasks, run every attempt
        concurrently under ``max_concurrency``, flatten the results, and -- unless
        ``gates=False`` -- run the ``Lifecycle.ROLLOUT`` gates over the flattened
        batch before returning it.

        With ``engine_recovery_retries > 0``, a step whose EVERY attempt is an
        ENGINE infra failure is recovered and re-gathered in place (see the class
        docstring and ``_recover_engines``): the failed attempt's summary is
        replaced by one ``[agentic-rl:engine-recovery]`` line per attempt, and the
        summary/guard/gates below run ONCE, on the FINAL trajectories only. The
        tasks are drawn once per call -- a retry re-runs the same step's tasks,
        never a fresh draw.
        """
        if type(step) is not int or step < 0:
            raise RolloutHostRefusal(
                f"RolloutHost.rollout: parameter 'step' is {_described(step)}: it must be "
                f"a real int >= 0"
            )
        tasks = self.tasks.batch(step, self.tasks_per_step)
        trajectories = await self._gather_step(step, tasks)
        attempt = 0
        while attempt < self.engine_recovery_retries and _all_engine_infra(trajectories):
            attempt += 1
            print(
                "[agentic-rl:engine-recovery] "
                + json.dumps(
                    {
                        "rollout_step": step,
                        "attempt": attempt,
                        "of": self.engine_recovery_retries,
                        "abstention_reasons": dict(
                            Counter(t.abstention_reason for t in trajectories)
                        ),
                    },
                    sort_keys=True,
                ),
                file=sys.stderr,
                flush=True,
            )
            # In a worker thread: the re-push drives the fleet with its own asyncio.run(), which
            # cannot run on this loop's thread, and the restart blocks for minutes.
            await asyncio.to_thread(self._recover_engines, step)
            trajectories = await self._gather_step(step, tasks)
        batch = flatten(trajectories, group_by_harness=self.group_by_harness)
        print(rollout_summary(step, trajectories), file=sys.stderr, flush=True)
        if self.guard is not None:
            verdict = self.guard.update(rollout_stats(step, trajectories))
            print(verdict_line(verdict), file=sys.stderr, flush=True)
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
        unconditionally; only ``ctx.rank == 0`` additionally completes the
        checkpoint for serving (when ``servable_base_model_dir`` is set) and
        pushes to the fleet -- see ``weight_sync.DiskWeightSync``'s module
        docstring for why ``sync`` itself must not be called on just one rank
        here. ``servable.complete_for_serving`` runs AFTER the collective save
        and BEFORE the push, so the fleet never reloads/restarts from a
        directory a fresh engine process could not itself load from cold --
        see ``foundationscale.agentic_rl.servable``'s module docstring for why
        a multimodal base model trained as text-only needs this at all; it is
        imported lazily here for the same reason ``save_checkpoint`` is
        (``safetensors.torch`` pulls in torch, and this module must stay
        importable, torch-free, for ``cli.py --dry-run``). A push that does
        not fully succeed (a fleet member unreachable, a reload that reported
        failure) raises -- CONTINUING training on an unreplaced checkpoint
        would serve the OLD policy's rollouts as if they were on-policy,
        silently, for every step after this one; the trainer's own total
        exit-code boundary (``agentic_rl.cli._run``) adjudicates this RED, the
        correct verdict for an infra fault discovered mid-run. A COMPLETE push
        records its path in ``_state["last_pushed_path"]`` -- the checkpoint
        ``_recover_engines`` re-pushes after a fleet restart, so a recovered
        engine serves the current policy (see the class docstring).
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
            if self.servable_base_model_dir is not None:
                from foundationscale.agentic_rl.servable import complete_for_serving

                complete_for_serving(
                    path,
                    self.servable_base_model_dir,
                    cache_dir=f"{self.weight_sync.publish_root}/.serving_cache",
                )
            report = self.weight_sync.push(path)
            if not report.complete:
                raise RolloutHostRefusal(
                    f"RolloutHost.publish: step {step}: the weight push to the fleet was "
                    f"NOT complete ({len(report.transferred)} of {len(report.offered)} "
                    f"server(s) transferred, failed_ranks={report.failed_ranks}): "
                    f"continuing would serve the stale, pre-step-{target_step} policy on "
                    f"every subsequent rollout without saying so"
                )
            self._state["last_pushed_path"] = path
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
            if self.guard is not None:
                self._apply_guard(step)

    def _apply_guard(self, step: int) -> None:
        """Retention and stop decisions for the checkpoint just published at ``step``.

        The trainer calls ``rollout(r)``, trains, then ``publish(r)``, and ``publish(r)``
        writes ``step_{r+1}`` (``target_step``). So rollout r samples the weights in
        ``step_r`` -- written by ``publish(r - 1)`` -- and a best-k reward at rollout r
        protects ``step_r``. When this runs (after ``publish(r)``'s prune) ``step_r`` is
        still inside the ``keep_last >= 2`` window; ``step_{r-1}`` is not, which is why
        protecting r - 1 silently kept nothing on a GB200 run. Rollout 0 sampled the base
        model, which has no published directory.
        """
        guard = self.guard
        assert guard is not None
        verdict = guard.last_verdict
        if verdict is None or self.weight_sync is None:
            return
        for rollout_step in verdict.best_steps:
            if 1 <= rollout_step <= step + 1:
                self.weight_sync.protect(rollout_step)
        for rollout_step in verdict.evicted:
            if rollout_step >= 1:
                self.weight_sync.unprotect(rollout_step)
        if verdict.alarm and guard.config.action == "stop":
            raise CollapseDetected(
                f"RolloutHost.publish: step {step}: rollout reward EMA {verdict.ema} fell to "
                f"<= {1 - guard.config.drop_from_peak:.2f} of its peak {verdict.peak} "
                f"(rollout step {verdict.peak_step}); the run stops here with checkpoint "
                f"{step} published and the best-k checkpoints kept"
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
