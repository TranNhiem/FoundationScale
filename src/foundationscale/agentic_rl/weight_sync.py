"""``DiskWeightSync``: the S0 (synchronous, whole-checkpoint-over-disk) realisation
of ``foundationscale.rl.weightsync.WeightSync``.

Realises that contract's two protocol methods -- ``capabilities()`` and a
``sync``-named call returning a ``SyncReport`` -- but DEVIATES from its literal
``sync(self, mapping: Mapping[str, str]) -> SyncReport`` signature: a disk
realisation moves one whole HF checkpoint directory, not a set of named
train-side/generate-side parameter pairs, so there is no parameter-name mapping
to take. ``sync(step: int)`` takes the step to publish instead; "offered" in the
returned ``SyncReport`` names the fleet's SERVERS (``"server_<i>"``), not
parameter names, and "transferred"/"skipped" partition them by whether that
server's ``update_weights_from_disk`` call succeeded.
``isinstance(obj, WeightSync)`` still holds for this class: the Protocol is
``@runtime_checkable``, which checks method PRESENCE only, never signatures
(see ``rl/weightsync.py``'s own docstring on this point).

``save_fn`` is OPTIONAL and serves exactly one caller: ``sync(step)``, a
convenience for the no-distributed-ranks case (this module's own tests, a
single-process smoke run) that calls ``save_fn`` itself, then pushes.
``rollout_host.RolloutHost.publish`` -- the REAL, collective, multi-rank
caller -- does NOT use ``sync`` or ``save_fn`` at all: it builds its save
around the ACTUAL ``model``/``tokenizer``/``ctx`` it receives from
``RLTrainer.run()`` (via ``rl.distributed.save_checkpoint``, called on every
rank directly, in lockstep) and then calls THIS module's ``push`` (below) only
on rank 0. The reason ``save_fn`` cannot serve ``publish`` too: ``save_fn``
would have to be a closure built BEFORE the real policy exists (whoever
constructs a ``DiskWeightSync`` builds it long before ``RLTrainer.run()`` ever
loads a model), so a caller may freely leave it ``None`` when ``publish`` is
the only path it uses -- ``sync()`` refuses, naming the gap, if ever called
without one. ``push`` (this module's addition beyond the two Protocol methods,
the same way ``engines.sglang.SGLangClient`` adds control calls beyond
``harness.base.GenerationClient``'s one method) needs no ``save_fn`` at all:
it only ever pushes an ALREADY-saved checkpoint.

``mode`` chooses HOW ``push`` delivers an already-saved checkpoint, and
``SyncReport`` semantics (offered/transferred/skipped/failed_ranks/is_stale)
are IDENTICAL across both:

* ``"reload_endpoint"`` (default, unchanged behaviour) -- ``fleet.
  update_weights_from_disk(path)``: every server's client is asked to reload
  the checkpoint from disk over its own HTTP control endpoint, with no
  interruption to serving.
* ``"restart"`` -- stops each ``EngineServer`` and starts it again with
  ``path`` substituted into its declared ``{model_path}`` command placeholder
  (``EngineServerSpec.command``). This is the GUARANTEED-correct S0 path for
  an engine (vLLM, today) with no PROVEN reload endpoint: a restart always
  picks up the new checkpoint because it is argv, not an RPC whose shape and
  availability are unprobed (see ``engines.vllm``'s UNPROBED ASSUMPTIONS on
  ``reload_weights``). It costs the serving interruption a reload avoids, and
  it requires every fleet server to be a generic ``EngineServer`` whose
  command declares the placeholder -- a legacy ``SGLangServer`` has no such
  field and this mode names it and refuses rather than silently no-op it.
"""

from __future__ import annotations

import asyncio
import re
import shutil
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from foundationscale.agentic_rl.engines import EngineInfraError
from foundationscale.agentic_rl.engines.fleet import EngineFleet, EngineServer
from foundationscale.rl.weightsync import SyncCapabilities, SyncReport

__all__ = (
    "DiskWeightSync",
    "DiskWeightSyncRefusal",
)

_MODES = ("reload_endpoint", "restart")
_MODEL_PATH_PLACEHOLDER = "{model_path}"

_STEP_DIR_RE = re.compile(r"^step_(\d+)$")


def _described(value: object) -> str:
    return f"{type(value).__name__} {value!r}"


class DiskWeightSyncRefusal(ValueError):
    """A config error constructing or calling ``DiskWeightSync`` -- not a sync-call
    failure, which is report-valued (see ``SyncReport``), never exception-valued.
    """


@dataclass(frozen=True)
class DiskWeightSync:
    """Save one HF checkpoint to ``{publish_root}/step_{step:06d}`` and push it to
    every server in ``fleet`` via ``update_weights_from_disk``.

    ``save_fn``, when given, writes the checkpoint directory for ``sync()``
    ONLY (a closure around the collective save, e.g.
    ``rl.distributed.save_checkpoint``, built by a non-distributed caller who
    already has the model in scope) -- ``None`` is lawful for a caller that
    only ever uses ``RolloutHost.publish``, which builds its own save around
    the real model/tokenizer/ctx it receives from ``RLTrainer.run()`` and never
    reads this field; see the module docstring. ``keep_last`` published step
    directories are kept under ``publish_root``; older ones are deleted --
    never anything outside ``publish_root``.
    """

    publish_root: str
    fleet: EngineFleet
    save_fn: Callable[[str], None] | None = None
    keep_last: int = 2
    mode: Literal["reload_endpoint", "restart"] = "reload_endpoint"

    def __post_init__(self) -> None:
        where = "DiskWeightSync"
        if self.save_fn is not None and not callable(self.save_fn):
            raise DiskWeightSyncRefusal(
                f"{where}: field 'save_fn' is {_described(self.save_fn)}: it must be "
                f"None or callable -- a non-callable, non-None value cannot write a "
                f"checkpoint, and None is lawful for a caller that only ever uses "
                f"RolloutHost.publish (see the module docstring) rather than sync()"
            )
        if not isinstance(self.publish_root, str) or not self.publish_root:
            raise DiskWeightSyncRefusal(
                f"{where}: field 'publish_root' is {_described(self.publish_root)}: it "
                f"must be a non-empty str -- an empty root names no directory to publish "
                f"under"
            )
        if not isinstance(self.fleet, EngineFleet):
            raise DiskWeightSyncRefusal(
                f"{where}: field 'fleet' is {_described(self.fleet)}, not an EngineFleet"
            )
        if type(self.keep_last) is not int or self.keep_last < 1:
            raise DiskWeightSyncRefusal(
                f"{where}: field 'keep_last' is {_described(self.keep_last)}: it must be "
                f"a real int >= 1 -- keeping fewer than one published checkpoint would "
                f"delete the directory a push just succeeded against"
            )
        if self.mode not in _MODES:
            raise DiskWeightSyncRefusal(
                f"{where}: field 'mode' is {_described(self.mode)}: it must be one of {_MODES}"
            )

    def capabilities(self) -> SyncCapabilities:
        """This realisation's claim: one transport ("disk"), and it populates both
        ``SyncReport.failed_ranks`` and ``SyncReport.is_stale``.
        """
        return SyncCapabilities(
            transports=("disk",), reports_per_rank_failure=True, reports_staleness=True
        )

    def path_for(self, step: int) -> str:
        """The canonical published directory for ``step``: ``{publish_root}/step_{step:06d}``."""
        if type(step) is not int or step < 0:
            raise DiskWeightSyncRefusal(
                f"DiskWeightSync.path_for: parameter 'step' is {_described(step)}: it must "
                f"be a real int >= 0"
            )
        return f"{self.publish_root}/step_{step:06d}"

    def push(self, path: str) -> SyncReport:
        """Push the ALREADY-SAVED checkpoint at ``path`` to every fleet server, prune
        stale published directories, and return the measured ``SyncReport``.

        ``offered`` names the fleet's servers as ``"server_<index>"``; a server whose
        ``update_weights_from_disk`` call failed is recorded in BOTH ``skipped`` (it
        did not receive the update) and ``failed_ranks`` (it failed, rather than
        having been intentionally left out) -- see the module docstring.
        ``bytes_moved`` stays ``None`` (unmeasured: this transport never counts
        bytes), ``seconds`` is the measured wall time of the push, and ``is_stale``
        is always ``False`` here -- DERIVED from this call being a COMPLETE,
        SYNCHRONOUS reload (the S0 schedule), never a separate measurement taken
        on the fleet or the published checkpoint: the caller observes the push
        complete before using its result, so staleness cannot have crept in
        between push and observation -- a claim specific to THIS synchronous
        realisation, never a general truth of the ``WeightSync`` contract (whose own
        docstring leaves staleness open, ``None``, for any realisation that cannot
        make this guarantee).
        """
        if not isinstance(path, str) or not path:
            raise DiskWeightSyncRefusal(
                f"DiskWeightSync.push: parameter 'path' is {_described(path)}: it must be "
                f"a non-empty str"
            )
        start = time.monotonic()
        if self.mode == "restart":
            results = self._restart_push(path)
        else:
            results = asyncio.run(self.fleet.update_weights_from_disk(path))
        elapsed = time.monotonic() - start
        offered = tuple(f"server_{i}" for i in range(len(results)))
        transferred = tuple(name for name, ok in zip(offered, results, strict=True) if ok)
        skipped = tuple(name for name, ok in zip(offered, results, strict=True) if not ok)
        failed_ranks = tuple(i for i, ok in enumerate(results) if not ok)
        self._prune()
        return SyncReport(
            transport="disk",
            offered=offered,
            transferred=transferred,
            skipped=skipped,
            failed_ranks=failed_ranks,
            bytes_moved=None,
            seconds=elapsed,
            is_stale=False,
        )

    def _restart_push(self, path: str) -> tuple[bool, ...]:
        """``mode="restart"``'s delivery: stop each server, substitute ``path`` into
        its declared ``{model_path}`` command placeholder, and start it again.

        The substitution is computed FRESH from ``server.spec.command`` on every
        call and passed to ``EngineServer.start(command=...)`` -- ``server.spec``
        itself is NEVER mutated, so the declared template (placeholder included)
        survives for the NEXT restart too, rather than being consumed by the
        first one.

        Sequential, deliberately: a restart kills and relaunches real subprocesses,
        and the S0 guarantee this mode exists for is correctness over throughput
        (see the module docstring). One bool per server, in ``self.fleet.servers``
        order, ``True`` iff that server's restart completed and became healthy.
        Refuses BEFORE touching any server if any one of them cannot run this mode
        -- a legacy ``SGLangServer`` (no declared command/placeholder) or an
        ``EngineServer`` whose command never names the placeholder -- naming the
        server index, so a fleet either restarts wholesale or not at all rather
        than being left with some servers already stopped.
        """
        for index, server in enumerate(self.fleet.servers):
            if not isinstance(server, EngineServer):
                raise DiskWeightSyncRefusal(
                    f"DiskWeightSync.push: mode='restart' requires every fleet server "
                    f"to be an EngineServer with a declared {_MODEL_PATH_PLACEHOLDER!r} "
                    f"command placeholder -- server_{index} is a "
                    f"{type(server).__name__}, which has no such placeholder"
                )
            if not any(_MODEL_PATH_PLACEHOLDER in part for part in server.spec.command):
                raise DiskWeightSyncRefusal(
                    f"DiskWeightSync.push: mode='restart' requires server_{index}'s "
                    f"command to declare the {_MODEL_PATH_PLACEHOLDER!r} placeholder at "
                    f"least once -- its command is {server.spec.command!r}"
                )
        results: list[bool] = []
        for server in self.fleet.servers:
            assert isinstance(server, EngineServer)  # checked for every server above
            server.stop()
            substituted_command = tuple(
                part.replace(_MODEL_PATH_PLACEHOLDER, path) for part in server.spec.command
            )
            try:
                server.start(command=substituted_command)
            except EngineInfraError:
                results.append(False)
                continue
            results.append(True)
        return tuple(results)

    def sync(self, step: int) -> SyncReport:
        """Save ``step``'s checkpoint via ``save_fn``, then ``push`` it.

        Convenience for a NON-distributed caller (e.g. this module's own tests):
        calls ``save_fn`` itself exactly once. Refuses if ``save_fn`` is ``None``
        -- a caller using only ``RolloutHost.publish`` (which never reads
        ``save_fn``, see the module docstring) may construct this with
        ``save_fn=None`` and must not then call ``sync`` too. A multi-rank
        caller must NOT call this on one rank while calling ``save_fn``
        directly on the others -- see the module docstring.
        """
        if self.save_fn is None:
            raise DiskWeightSyncRefusal(
                "DiskWeightSync.sync: field 'save_fn' is None: sync() needs a "
                "non-distributed save closure declared at construction -- a "
                "distributed caller publishes through RolloutHost.publish instead, "
                "which never reads save_fn"
            )
        path = self.path_for(step)
        self.save_fn(path)
        return self.push(path)

    def _prune(self) -> None:
        """Delete every ``step_<N>`` directory directly under ``publish_root`` except
        the ``keep_last`` highest-numbered ones. CONFINED to ``publish_root``:
        ``publish_root`` is resolved to its canonical real path first, and a
        candidate is only deleted when it is a real (non-symlink) directory whose
        OWN resolved path is a direct child of that canonical root -- so a symlink
        planted under ``publish_root``, or any other escape, can never point this
        deletion anywhere else. Never touches a non-matching entry.

        This also means ``{publish_root}/.serving_cache`` (``servable.
        complete_for_serving``'s extras-file cache, when ``RolloutHost.
        servable_base_model_dir`` is set) is NEVER pruned here: its name never
        matches ``_STEP_DIR_RE``, by construction, not by an extra exclusion
        this function would have to remember to keep. A pruned step directory
        may hold one of that cache's files as a hard link
        (``model-serving-extras.safetensors``); deleting the directory only
        removes THIS link -- the cache's own copy, and the inode, survive as
        long as the cache keeps its link to it.
        """
        root = Path(self.publish_root)
        if not root.is_dir():
            return
        resolved_root = root.resolve()
        numbered: list[tuple[int, Path]] = []
        for child in root.iterdir():
            if child.is_symlink():
                continue
            if not child.is_dir():
                continue
            match = _STEP_DIR_RE.match(child.name)
            if match is None:
                continue
            resolved_child = child.resolve()
            if resolved_child.parent != resolved_root:
                continue
            numbered.append((int(match.group(1)), resolved_child))
        numbered.sort(key=lambda pair: pair[0])
        stale = numbered[: max(0, len(numbered) - self.keep_last)]
        for _, directory in stale:
            shutil.rmtree(directory)
