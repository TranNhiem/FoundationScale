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
import faulthandler
import inspect
import json
import os
import re
import sys
import threading
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from foundationscale.agentic_rl.contracts import Termination, Trajectory, flatten
from foundationscale.agentic_rl.curriculum import FrontierTaskSource
from foundationscale.agentic_rl.engines import EngineInfraError
from foundationscale.agentic_rl.engines.fleet import EngineFleet
from foundationscale.agentic_rl.envs.base import (
    EnvBackend,
    EnvInfraError,
    EnvSpec,
    TaskAwareEnvBackend,
    VerifiableEnvironment,
)
from foundationscale.agentic_rl.fs_trace import (
    LiveTraceWriter,
    TraceWriter,
    trace_from_trajectory,
    trace_id_for_session,
)
from foundationscale.agentic_rl.harness.base import (
    EpisodeBudget,
    EpisodeOutcome,
    EpisodeTask,
    GenerationClient,
    HarnessAdapter,
    segment_trajectories,
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


_RECOVERY_RERUN = ("all", "engine_rows")


def _deadline_exit(code: int) -> None:
    """End the process immediately with ``code`` (seam for tests: monkeypatch this)."""
    os._exit(code)


def _is_engine_infra(trajectory: Trajectory) -> bool:
    """True iff this row is an ENGINE infra failure (same reading as ``_all_engine_infra``)."""
    reason = trajectory.abstention_reason
    return (
        trajectory.termination is Termination.INFRA
        and isinstance(reason, str)
        and (reason.startswith("infra:engine_") or reason == "infra:EngineInfraError")
    )


def _needs_engine_recovery(trajectories: Sequence[Trajectory], max_rate: float | None) -> bool:
    """Whether a step's rows call for an engine restart and a retry of the same step.

    Every row an engine failure (a fully hung engine), or -- when the infra gate declares a
    ``max_rate`` -- engine failures alone exceeding that rate: a hang that times out only PART
    of a step (GB200 swe_v1_s6, step 19: 21 of 32 engine_timeout) would be refused by the gate
    anyway, so retrying on a restarted engine can only help. A step under the rate trains as is.
    """
    if _all_engine_infra(trajectories):
        return True
    if max_rate is None or not trajectories:
        return False
    engine = sum(1 for t in trajectories if _is_engine_infra(t))
    return engine / len(trajectories) > max_rate


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
    Recovery fires when EVERY trajectory in a step is an engine infra failure, or
    when engine failures alone exceed ``declared_max_infra_rate`` (the step the
    infra gate would refuse anyway; see ``_needs_engine_recovery``): it stops and
    restarts the fleet,
    re-pushes the last published checkpoint (``_state["last_pushed_path"]``,
    recorded by ``publish``) so the restarted engine serves the CURRENT policy
    rather than the base model, and retries the SAME step with the SAME tasks. A
    failed attempt's summary is replaced by one
    ``[agentic-rl:engine-recovery]`` JSON line to stderr. It never fires on a
    partial failure the gate would accept, or on env failures, and if the retries
    are exhausted the step's infra rows still reach the infra gate, which
    refuses as it always did. A positive count needs a ``fleet`` (there must be
    an engine to stop and start back up).
    """

    tasks: TaskSource | FrontierTaskSource
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
    # Which rows an engine-recovery retry re-runs: "all" re-gathers every attempt of the
    # step (the original behaviour); "engine_rows" re-runs ONLY the rows that ended as
    # ENGINE infra failures and keeps every other row as measured -- a recovery re-pushes
    # the SAME checkpoint, so kept and re-run rows share one policy version (checked, and
    # refused if they do not; see ``_merge_recovered``).
    recovery_rerun: str = "all"
    # The policy version the fleet serves when it is LAUNCHED (cli.py launches every
    # server from policy.model_path, the version-0 weights the trainer starts from). Seeds
    # the per-server version ledger (``_state["server_versions"]``) that ``publish`` and
    # ``_recover_engines`` advance from their own push reports.
    fleet_launch_version: int = 0
    # Dynamic sampling (ARL-G06, DAPO-style): "off" trains on the drawn tasks as they come;
    # "refill" drops groups whose rows are ALL measured with ONE reward value (zero spread ->
    # zero centred advantage -> no gradient) and runs up to ``max_refill_rounds`` rounds of
    # fresh tasks until ``tasks_per_step`` groups carry signal (see ``_refill``).
    dynamic_sampling: str = "off"
    max_refill_rounds: int = 2
    # Wall-clock deadline (seconds) on one whole ``publish`` -- save, serving completion,
    # push and any push recovery. None (default) = no deadline. On expiry every thread's
    # stack is dumped to stderr and the process exits RED (5): GB200 arl_pg_s10/arl_g_s10
    # each hung ~2 days between writing the checkpoint shards and the push (no reload
    # ever reached the engine), holding 4 GPUs, with nothing able to notice or recover.
    publish_deadline_s: float | None = None
    # ARL-21: when set, every step's final trajectories are written as fs.trace/v1 span traces
    # to <trace_dir>/step_<NNNNNN>.jsonl (agentic_rl.fs_trace); None (default) writes nothing.
    trace_dir: str | None = None
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
        if self.recovery_rerun not in _RECOVERY_RERUN:
            raise RolloutHostRefusal(
                f"{where}: field 'recovery_rerun' is {_described(self.recovery_rerun)}: it "
                f"must be one of {', '.join(repr(v) for v in _RECOVERY_RERUN)}"
            )
        _real_int_at_least(self.fleet_launch_version, 0, where=where, name="fleet_launch_version")
        if self.dynamic_sampling not in ("off", "refill"):
            raise RolloutHostRefusal(
                f"{where}: field 'dynamic_sampling' is {_described(self.dynamic_sampling)}: it "
                f"must be one of 'off', 'refill'"
            )
        _real_int_at_least(self.max_refill_rounds, 1, where=where, name="max_refill_rounds")
        if self.trace_dir is not None and (
            not isinstance(self.trace_dir, str) or not self.trace_dir
        ):
            raise RolloutHostRefusal(
                f"{where}: field 'trace_dir' is {_described(self.trace_dir)}: it must be None or "
                f"a non-empty str"
            )
        if self.publish_deadline_s is not None and (
            type(self.publish_deadline_s) is bool
            or not isinstance(self.publish_deadline_s, (int, float))
            or not (0.0 < float(self.publish_deadline_s) < float("inf"))
        ):
            raise RolloutHostRefusal(
                f"{where}: field 'publish_deadline_s' is {_described(self.publish_deadline_s)}: "
                f"it must be None or a finite number of seconds > 0"
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

    def _server_count(self) -> int | None:
        """How many servers sessions are round-robined over, or ``None`` when this
        host cannot know (an injected ``client_factory`` with no fleet anywhere)."""
        # Only a fleet's round-robin (``session_index % len(servers)``) maps a session
        # to a server this host can name; an injected client_factory routes opaquely.
        # A fleet that does not expose its servers (a test double) cannot be mapped either.
        servers = getattr(self.fleet, "servers", None) if self.fleet is not None else None
        return len(servers) if servers else None

    def _server_versions(self) -> list[int] | None:
        """The per-server policy-version ledger, seeded on first use from
        ``fleet_launch_version``; ``None`` when the server count is unknowable.

        Entry ``i`` is the version of the checkpoint server ``i`` last COMPLETED
        loading, advanced only from push reports (``publish``, ``_recover_engines``)
        -- the same evidence the push itself is judged on, never the trainer's own
        step counter. Sessions read their version from the server they ran on, so
        ``StalenessGate`` checks what each session was actually served."""
        ledger = self._state.get("server_versions")
        if ledger is None:
            count = self._server_count()
            if count is None:
                return None
            ledger = [self.fleet_launch_version] * count
            self._state["server_versions"] = ledger
        return ledger

    def _record_push(self, report: SyncReport | None, version: int) -> None:
        """Advance the ledger after a push: every server the report names as
        transferred now serves ``version``. ``report is None`` (a fleet restart
        with nothing re-pushed) means every server relaunched from its declared
        command, i.e. serves ``fleet_launch_version`` again."""
        ledger = self._server_versions()
        if ledger is None:
            return
        if report is None:
            ledger[:] = [self.fleet_launch_version] * len(ledger)
            return
        for name in report.transferred:
            index = int(name.rsplit("_", 1)[1])  # DiskWeightSync names servers "server_<i>"
            ledger[index] = version

    def _session_version(self, session_index: int, step: int) -> int:
        """The policy version the server for ``session_index`` serves (ledger), or
        ``step`` -- the legacy asserted value, recorded as unmeasured -- when no
        ledger exists."""
        ledger = self._server_versions()
        if ledger is None:
            return step
        return ledger[session_index % len(ledger)]

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
            version = self._session_version(session_index, step)
            if isinstance(self.env_backend, TaskAwareEnvBackend):
                instance = self.env_backend.create_for_task(
                    self.env_spec, task, instance_id=session_id
                )
            else:
                instance = self.env_backend.create(self.env_spec, instance_id=session_id)
            outcome: EpisodeOutcome | None = None
            try:
                await instance.start()
                outcome = await self.harness.run_episode(
                    task,
                    client=client,
                    env=instance,
                    budget=self.budget,
                    **self._live_trace_kwargs(step, session_id, version),
                )
                reward_value: float | None
                abstention: str | None
                extra_metadata: dict[str, str] | None = None
                if outcome.termination is Termination.INFRA:
                    reward_value, abstention = None, outcome.abstention_reason
                else:
                    if isinstance(instance, VerifiableEnvironment):
                        # e.g. hidden tests run against the agent's edits; an EnvInfraError
                        # here lands in the infra branch below, never as a scored zero.
                        checked = await instance.verify()
                        if self.trace_dir is not None:
                            # ARL-21: name the verifier's keys so _write_traces can join a
                            # verifier span to the reward (only when tracing: otherwise the
                            # metadata stays exactly what it was).
                            checked = {**checked, "verifier_keys": ",".join(sorted(checked))}
                        outcome = replace(outcome, metadata={**outcome.metadata, **checked})
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
                final = to_trajectory(
                    outcome,
                    uid=task.uid,
                    session_id=session_id,
                    harness=self.harness.name,
                    reward=reward_value,
                    abstention_reason=abstention,
                    versions=(version, version),
                    extra_metadata=extra_metadata,
                )
                # ARL-17: closed segments of a compacted episode wait here, keyed by session,
                # and are expanded only at flatten time -- every per-session count (gates,
                # recovery, refill, curriculum) keeps seeing one row per session.
                segments = self._state.setdefault("segments", {})
                if outcome.segments:
                    segments[session_id] = segment_trajectories(outcome, final)
                else:
                    segments.pop(session_id, None)
                return final
            except _INFRA_EXCEPTIONS as exc:
                reason = f"infra:{type(exc).__name__}"
                # ARL-17: an abstained attempt has no segments; never let an earlier attempt's
                # stored segments stand in for it at flatten time.
                self._state.get("segments", {}).pop(session_id, None)
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
                        policy_version_min=version,
                        policy_version_max=version,
                    )
                return _zero_turn_infra_trajectory(
                    task,
                    session_id=session_id,
                    harness_name=self.harness.name,
                    reason=reason,
                    step=version,
                )
            finally:
                try:
                    await instance.close()
                except EnvInfraError as exc:
                    # The episode's row is already final; a sandbox that failed to delete is
                    # an operator problem (a leaked container), not a reason to lose the step.
                    print(
                        f"[agentic-rl:env] close failed for {session_id}: {exc}",
                        file=sys.stderr,
                        flush=True,
                    )

    async def _gather_step(
        self,
        step: int,
        tasks: Sequence[EpisodeTask],
        only: frozenset[int] | None = None,
    ) -> list[Trajectory]:
        """Build and run EVERY attempt of ``tasks`` for ``step``, returning one
        ``Trajectory`` per ``tasks_per_step * group_size`` attempt.

        Called once per attempt at the step (once, or once more per engine
        recovery retry) with the SAME task list: session ids (``f"{uid}#{k}"``)
        and session indices are derived purely from ``tasks``/``group_size``, so
        a retried gather produces exactly the rows the first one would have.
        Each call builds a FRESH semaphore -- the counting state of a previous
        call's semaphore is spent on episodes that have since finished/failed,
        and a recovery retry must not inherit it.

        ``only``, when given, runs just those session indices (in index order) --
        the ``recovery_rerun="engine_rows"`` retry; each keeps its session id and
        index, hence its server.
        """
        semaphore = asyncio.Semaphore(self.max_concurrency)
        episodes = []
        session_index = 0
        for task in tasks:
            for replica in range(self.group_size):
                if only is None or session_index in only:
                    session_id = f"{task.uid}#{replica}"
                    episodes.append(
                        self._run_episode(task, session_id, session_index, step, semaphore)
                    )
                session_index += 1
        return list(await asyncio.gather(*episodes))

    def _expand_segments(self, trajectories: list[Trajectory]) -> list[Trajectory]:
        """ARL-17: replace each compacted session's row by all of its segment rows (the
        recorded rows end with that same final segment, renamed ``::seg<n>``); sessions
        that never compacted pass through unchanged. The side table is consumed."""
        segments: dict[str, tuple[Trajectory, ...]] = self._state.pop("segments", {})
        rows: list[Trajectory] = []
        for trajectory in trajectories:
            expanded = segments.get(trajectory.session_id)
            rows.extend(expanded if expanded is not None else (trajectory,))
        return rows

    def _merge_recovered(
        self, step: int, previous: list[Trajectory], rerun: list[Trajectory], indices: list[int]
    ) -> list[Trajectory]:
        """Splice re-run rows into ``previous`` at ``indices``. Refuses when a kept
        measured row and a re-run row carry different policy versions: one step's
        batch must come from one policy, and a recovery that changed the served
        version would mix two policies' samples under one advantage baseline."""
        merged = list(previous)
        for index, row in zip(indices, rerun, strict=True):
            merged[index] = row
        rerun_set = set(indices)
        kept = {t.policy_version_min for i, t in enumerate(merged) if i not in rerun_set}
        fresh = {t.policy_version_min for t in rerun}
        if kept and fresh and kept != fresh:
            raise RolloutHostRefusal(
                f"RolloutHost.rollout: step {step}: recovery_rerun='engine_rows' would mix "
                f"policy versions {sorted(map(str, kept))} (kept rows) and "
                f"{sorted(map(str, fresh))} (re-run rows) "
                f"in one batch; use recovery_rerun='all' or fix the re-push"
            )
        return merged

    def _live_trace_kwargs(self, step: int, session_id: str, version: int) -> dict[str, Any]:
        """ARL-21 crash evidence: with ``trace_dir`` set and a harness whose ``run_episode``
        accepts ``observer``/``policy_version``, a ``LiveTraceWriter`` that fsyncs a start and
        an end record around every generation and tool call to
        ``<trace_dir>/live/step_<NNNNNN>/<session>.jsonl``. Otherwise ``{}`` -- the call is
        exactly the pre-ARL-21 one."""
        if self.trace_dir is None:
            return {}
        accepts = self._state.get("harness_accepts_observer")
        if accepts is None:
            params = inspect.signature(self.harness.run_episode).parameters
            accepts = "observer" in params and "policy_version" in params
            self._state["harness_accepts_observer"] = accepts
        if not accepts:
            return {}
        directory = Path(self.trace_dir) / "live" / f"step_{step:06d}"
        directory.mkdir(parents=True, exist_ok=True)
        safe = re.sub(r"[^A-Za-z0-9._-]", "_", session_id)
        writer = LiveTraceWriter(
            directory / f"{safe}.jsonl",
            trace_id=trace_id_for_session(
                run_id=Path(self.trace_dir).name,
                rollout_step=step,
                session_id=session_id,
                invocation=0,  # unique session ids per step; matches _write_traces
            ),
            session_id=session_id,
        )
        return {"observer": writer, "policy_version": version}

    def _write_traces(self, step: int, trajectories: Sequence[Trajectory]) -> None:
        """ARL-21: one fs.trace/v1 trace per final trajectory of ``step``, appended to
        ``<trace_dir>/step_<NNNNNN>.jsonl``. The run identity in every trace id is the trace
        directory's name, so traces of different runs never share ids, and a session's live
        file (``_live_trace_kwargs``) carries the same trace id. A write failure propagates
        -- a declared trace that silently went missing would read as a complete record."""
        assert self.trace_dir is not None
        directory = Path(self.trace_dir)
        directory.mkdir(parents=True, exist_ok=True)
        writer = TraceWriter(directory / f"step_{step:06d}.jsonl")
        for trajectory in trajectories:
            meta = trajectory.metadata
            keys = [k for k in meta.get("verifier_keys", "").split(",") if k]
            # invocation 0: a session id is unique within a step, and the live writer
            # (_live_trace_kwargs) derives the same trace id, so the two files join.
            writer.write_episode(
                trace_from_trajectory(
                    trajectory,
                    run_id=directory.name,
                    rollout_step=step,
                    invocation=0,
                    verifier_metadata={k: meta[k] for k in keys if k in meta} if keys else None,
                    reward_note=meta.get("reward_note"),
                )
            )

    def _group_key(self, trajectory: Trajectory) -> str:
        return (
            f"{trajectory.uid}::{trajectory.harness}" if self.group_by_harness else trajectory.uid
        )

    def _droppable_groups(self, trajectories: Sequence[Trajectory]) -> set[str]:
        """Groups whose rows are ALL measured and share ONE reward: zero spread, so a centred
        baseline gives every row zero advantage. A group with ANY abstained/infra row is never
        droppable -- dropping it would hide infra from ``RolloutAbstentionGate``."""
        rewards: dict[str, list[float | None]] = {}
        for trajectory in trajectories:
            rewards.setdefault(self._group_key(trajectory), []).append(trajectory.reward)
        return {
            key
            for key, values in rewards.items()
            if all(v is not None for v in values) and len(set(values)) == 1
        }

    async def _refill(
        self, step: int, round0: list[Trajectory], reserve: Sequence[EpisodeTask]
    ) -> list[Trajectory]:
        """DAPO-style dynamic sampling: keep round 0's informative groups, run reserve tasks in
        chunks of ``tasks_per_step`` until ``tasks_per_step`` groups carry signal or
        ``max_refill_rounds`` is spent, and return the kept rows (round 0's rows first).

        Kept: every non-droppable group, in draw order, up to ``tasks_per_step`` groups. When
        no group anywhere is informative the step returns ``round0`` unchanged -- the trainer
        then reports the step UNMEASURED exactly as without refill. One
        ``[agentic-rl:dynamic-sampling]`` line reports rounds run and groups kept/dropped.
        """
        gathered = list(round0)
        rounds = 0
        offset = 0

        def informative(rows: Sequence[Trajectory]) -> list[str]:
            drop = self._droppable_groups(rows)
            order: list[str] = []
            for row in rows:
                key = self._group_key(row)
                if key not in drop and key not in order:
                    order.append(key)
            return order

        while (
            len(informative(gathered)) < self.tasks_per_step
            and rounds < self.max_refill_rounds
            and offset < len(reserve)
        ):
            chunk = reserve[offset : offset + self.tasks_per_step]
            offset += len(chunk)
            rounds += 1
            gathered.extend(await self._gather_step(step, chunk))
        keep = informative(gathered)[: self.tasks_per_step]
        all_groups = {self._group_key(t) for t in gathered}
        print(
            "[agentic-rl:dynamic-sampling] "
            + json.dumps(
                {
                    "rollout_step": step,
                    "rounds": rounds,
                    "groups_gathered": len(all_groups),
                    "groups_kept": len(keep),
                    "groups_dropped_zero_spread": len(all_groups) - len(set(informative(gathered))),
                },
                sort_keys=True,
            ),
            file=sys.stderr,
            flush=True,
        )
        # Every gathered row (dropped groups included) is a measurement of its task's
        # difficulty; the curriculum (if any) reads them from here.
        self._state["last_gathered"] = list(gathered)
        if not keep:
            return round0
        kept = set(keep)
        return [t for t in gathered if self._group_key(t) in kept]

    def _recover_engines(self, step: int) -> SyncReport | None:
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

        Returns the re-push's report (``None`` when nothing was re-pushed). Raises
        ``RolloutHostRefusal`` instead of returning an incomplete one: a recovery
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
        self._record_push(report, self._state.get("last_pushed_version", self.fleet_launch_version))
        return report

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
        if self.dynamic_sampling == "refill":
            # One deterministic draw covers round 0 and every refill round (capped at the pool,
            # so no task repeats within a step and no (uid, session_id) collides in flatten).
            pool = self.tasks.batch(
                step,
                min(len(self.tasks), self.tasks_per_step * (1 + self.max_refill_rounds)),
            )
            tasks, reserve = pool[: self.tasks_per_step], pool[self.tasks_per_step :]
        else:
            tasks, reserve = self.tasks.batch(step, self.tasks_per_step), ()
        self._state["segments"] = {}  # ARL-17: segments never carry over from an aborted step
        trajectories = await self._gather_step(step, tasks)
        attempt = 0
        while attempt < self.engine_recovery_retries and _needs_engine_recovery(
            trajectories, self.declared_max_infra_rate
        ):
            attempt += 1
            print(
                "[agentic-rl:engine-recovery] "
                + json.dumps(
                    {
                        "rollout_step": step,
                        "attempt": attempt,
                        "of": self.engine_recovery_retries,
                        "abstention_reasons": dict(
                            Counter(t.abstention_reason or "measured" for t in trajectories)
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
            if self.recovery_rerun == "engine_rows":
                indices = [i for i, t in enumerate(trajectories) if _is_engine_infra(t)]
                rerun = await self._gather_step(step, tasks, only=frozenset(indices))
                trajectories = self._merge_recovered(step, trajectories, rerun, indices)
            else:
                trajectories = await self._gather_step(step, tasks)
        # Summary and stability guard read ROUND 0 only: an unbiased sample of the drawn tasks.
        # Refilled batches over-represent groups with spread, which would inflate the reward EMA.
        round0 = trajectories
        self._state["last_gathered"] = list(round0)
        if self.dynamic_sampling == "refill":
            trajectories = await self._refill(step, trajectories, reserve)
        if isinstance(self.tasks, FrontierTaskSource):
            # ARL-20: every measured rollout this step (refill rounds included) updates the
            # difficulty registry the NEXT step's draw is weighted by.
            self.tasks.observe(self._state["last_gathered"])
            print(
                "[agentic-rl:curriculum] "
                + json.dumps({"rollout_step": step, **self.tasks.stats()}, sort_keys=True),
                file=sys.stderr,
                flush=True,
            )
        # The measured per-session versions publish hands StalenessGate; () when no
        # ledger exists (nothing measured -- never the asserted step).
        # The freshest version the fleet COULD serve at this rollout: the last completed push
        # (or the launch weights). Staleness is judged against it, not against the trainer's
        # step counter -- a step the trainer skips (UNMEASURED: no gradient, so no update and
        # no publish) leaves the weights, and therefore the frontier, unchanged.
        self._state["rollout_frontier"] = self._state.get(
            "last_pushed_version", self.fleet_launch_version
        )
        self._state["last_step_versions"] = (
            tuple(t.policy_version_min for t in trajectories)
            if self._server_versions() is not None
            else ()
        )
        if self.trace_dir is not None:
            self._write_traces(step, trajectories)
        batch = flatten(self._expand_segments(trajectories), group_by_harness=self.group_by_harness)
        print(rollout_summary(step, round0), file=sys.stderr, flush=True)
        if self.guard is not None:
            verdict = self.guard.update(rollout_stats(step, round0))
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
        """Publish under ``publish_deadline_s`` (see ``_publish_body`` for the publish itself).

        With a deadline declared, a daemon timer armed before the body and cancelled after it
        (success or exception) fires only if the body is still running when the deadline
        passes: it prints a ``[fs:agentic-rl:red]`` line naming the step and the deadline,
        dumps every thread's stack (``faulthandler``) so the hang is diagnosable, and ends
        the process with exit code 5 (RED, inside the 0/5/95/96 contract) via
        ``_deadline_exit``. A hung save or serving completion can block in an uninterruptible
        file-system call, so raising into the stuck thread is not an option; ending the
        process is the only way to free the GPUs and let the run's supervisor move on.
        """
        if self.publish_deadline_s is None:
            self._publish_body(model, tokenizer, ctx, step)
            return
        deadline = float(self.publish_deadline_s)

        def expire() -> None:
            # The exit is unconditional: the message and the stack dump are best effort (a
            # stderr without a file descriptor makes faulthandler raise), the RED exit is not.
            try:
                print(
                    f"[fs:agentic-rl:red] RolloutHost.publish: step {step} did not finish "
                    f"within publish_deadline_s={deadline:g}s -- dumping every thread's stack "
                    f"and exiting RED (5) rather than holding the GPUs on a hung save/push",
                    file=sys.stderr,
                    flush=True,
                )
                try:
                    faulthandler.dump_traceback(file=sys.stderr, all_threads=True)
                except (OSError, ValueError, AttributeError):
                    fallback = sys.__stderr__  # the process's original stderr; None if closed
                    if fallback is not None:
                        faulthandler.dump_traceback(file=fallback, all_threads=True)
                sys.stderr.flush()
            finally:
                _deadline_exit(5)

        timer = threading.Timer(deadline, expire)
        timer.daemon = True
        timer.start()
        try:
            self._publish_body(model, tokenizer, ctx, step)
        finally:
            timer.cancel()

    def _publish_body(self, model: Any, tokenizer: Any, ctx: Any, step: int) -> None:
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
            pushes = 0
            while not report.complete and pushes < self.engine_recovery_retries:
                # A hang that timed out only SOME of the step's episodes (under max_infra_rate)
                # never triggers rollout recovery; the step trains and the push then meets the
                # hung engine (GB200 math_grpo_s5, step 4: 9/32 engine_timeout, push 0 of 1).
                # Restart the fleet and push THIS checkpoint, as rollout recovery does.
                pushes += 1
                print(
                    "[agentic-rl:engine-recovery] "
                    + json.dumps(
                        {
                            "publish_step": step,
                            "attempt": pushes,
                            "of": self.engine_recovery_retries,
                            "failed_ranks": list(report.failed_ranks),
                        },
                        sort_keys=True,
                    ),
                    file=sys.stderr,
                    flush=True,
                )
                self._state["last_pushed_path"] = path
                self._state["last_pushed_version"] = target_step
                recovered = self._recover_engines(step)
                if recovered is not None:
                    report = recovered
            if not report.complete:
                raise RolloutHostRefusal(
                    f"RolloutHost.publish: step {step}: the weight push to the fleet was "
                    f"NOT complete ({len(report.transferred)} of {len(report.offered)} "
                    f"server(s) transferred, failed_ranks={report.failed_ranks}): "
                    f"continuing would serve the stale, pre-step-{target_step} policy on "
                    f"every subsequent rollout without saying so"
                )
            self._state["last_pushed_path"] = path
            self._state["last_pushed_version"] = target_step
            self._record_push(report, target_step)
            if self.gates:
                # rollout_policy_versions: the per-session versions the LAST rollout
                # measured off the server ledger (see _server_versions) -- () when no
                # ledger exists (client_factory hosts), never a fabricated sequence.
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
                    # Measured sessions are judged in UPDATES: the weights published here are the
                    # rollout frontier plus one update, so an on-policy session lags by exactly 1
                    # and only a server left on older weights lags by more -- even when a skipped
                    # (UNMEASURED) step made target_step jump past frontier + 1. Without measured
                    # sessions the trainer's own target_step stands, as before.
                    policy_version=(
                        self._state["rollout_frontier"] + 1
                        if self._state.get("last_step_versions")
                        else target_step
                    ),
                    rollout_policy_versions=self._state.get("last_step_versions", ()),
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
        # Generated turns carrying an unparseable tool call / a repetition hit: the
        # per-turn facts ProcessShaping (agentic_process_grpo) penalises, counted in
        # every run so a shaped and an unshaped arm report the same quantity.
        "tool_call_error_turns": sum(
            1
            for t in trajectories
            for turn in t.turns
            if turn.generated and any(c.parse_error is not None for c in turn.tool_calls)
        ),
        "repetition_turns": sum(
            1 for t in trajectories for turn in t.turns if turn.generated and turn.repetition_hit
        ),
        "generated_turns": sum(1 for t in trajectories for turn in t.turns if turn.generated),
    }
    return "[agentic-rl:rollout] " + json.dumps(payload, sort_keys=True)
