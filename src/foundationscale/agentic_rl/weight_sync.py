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

``sync(step)`` is a convenience for the no-distributed-ranks case: it calls
``save_fn`` itself, then pushes. A collective, multi-rank caller (see
``rollout_host.RolloutHost.publish``) must NOT use it directly, because
``save_fn`` wraps a COLLECTIVE checkpoint save (every rank must call it exactly
once, in lockstep) while only rank 0 may push to the serving fleet; calling
``sync`` on rank 0 and `save_fn` separately on every other rank would call
``save_fn`` once on every rank except rank 0, which would call it TWICE (once
via the direct call every rank makes, once again inside ``sync``) -- breaking
the one-call-per-rank symmetry a real collective save depends on. ``publish``
therefore calls ``path_for`` + ``save_fn`` directly on every rank, and ``push``
(this module's addition beyond the two Protocol methods, the same way
``engines.sglang.SGLangClient`` adds control calls beyond
``harness.base.GenerationClient``'s one method) only on rank 0.
"""

from __future__ import annotations

import asyncio
import re
import shutil
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from foundationscale.agentic_rl.engines.fleet import EngineFleet
from foundationscale.rl.weightsync import SyncCapabilities, SyncReport

__all__ = (
    "DiskWeightSync",
    "DiskWeightSyncRefusal",
)

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

    ``save_fn`` writes the checkpoint directory; the caller supplies a closure
    around the collective save (e.g. ``rl.distributed.save_checkpoint``) so this
    module stays torch-free and training-backend-agnostic. ``keep_last`` published
    step directories are kept under ``publish_root``; older ones are deleted --
    never anything outside ``publish_root``.
    """

    save_fn: Callable[[str], None]
    publish_root: str
    fleet: EngineFleet
    keep_last: int = 2

    def __post_init__(self) -> None:
        where = "DiskWeightSync"
        if not callable(self.save_fn):
            raise DiskWeightSyncRefusal(
                f"{where}: field 'save_fn' is {_described(self.save_fn)}: it must be "
                f"callable -- a non-callable cannot write a checkpoint"
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
        is asserted ``False`` because this call is synchronous: the caller observes
        the push complete before using its result, so staleness cannot have crept in
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

    def sync(self, step: int) -> SyncReport:
        """Save ``step``'s checkpoint via ``save_fn``, then ``push`` it.

        Convenience for a NON-distributed caller (e.g. this module's own tests):
        calls ``save_fn`` itself exactly once. A multi-rank caller must NOT call
        this on one rank while calling ``save_fn`` directly on the others -- see
        the module docstring.
        """
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
