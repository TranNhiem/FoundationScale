from __future__ import annotations

"""Weight-sync transport cost on GB200 (PHASE3_DESIGN.md, section 7, item 3).

After N optimiser steps in RL post-training, the policy weights must reach the
rollout generator. This module prices the three candidate transports, plus one
positive control, as a function of payload size on the estate's GB200 tray.

ARMS
----
* full_copy          : each rank moves the payload D2H into a PINNED host
                       buffer and back H2D. This is the portable
                       "save-and-reload" path minus the filesystem. On GB200
                       host/device share NVLink-C2C, so this number is
                       estate-specific and is the reason this script exists.
* collective         : dist.broadcast from rank 0's GPU to every rank. Pure
                       device-to-device, no host involvement.
* reshard            : training holds the tensor sharded over TP=world_size,
                       inference wants it whole; dist.all_gather_into_tensor
                       reconstitutes it.
* full_copy_pageable : POSITIVE CONTROL, identical to full_copy but with a
                       NON-pinned host buffer. It MUST be slower than pinned
                       full_copy by a PREDECLARED margin with non-overlapping
                       bootstrap intervals; if it is not, the harness is not
                       moving the bytes it claims (copy elided / not what we
                       think) and the whole run is adjudicated RED, not CLEAR.
                       Construction of the control REFUSES (96) if the host
                       allocator did not honour pin_memory as requested.

WHAT IS MEASURED
----------------
* Per (transport, size): poison+prime before EACH timed iteration (never
  inside the timed window), barrier-aligned CUDA-event timing, warmups run
  to a size-dependent floor AND a stability precondition, then an adaptive
  repeat count. Every rank keeps its OWN event samples; there is NO common
  clock across ranks, and none is implied. Ranks exchange per-rank medians
  after the window; the reported aggregate cost is the SLOWEST RANK'S MEDIAN,
  named as such, because that is what the training loop waits for. Per-rank
  spread plus IQR are reported -- never a bare mean; variance is the finding.
* A 1 MiB payload for every transport, labelled "per_call_floor". Cadence
  semantics turn on whether cost is per-byte or per-call, so the fixed
  per-call floor is reported separately from the sweep.
* Blocking collective forms only. Verified against the installed torch
  2.13.0 source: with async_op=False, distributed_c10d calls work.wait()
  itself in broadcast and all_reduce (and all_gather_into_tensor delegates
  to the same contract), so an end event recorded on the current stream
  after the call returns is correctly ordered behind the NCCL work.

WHAT "MOVED" MEANS (per transport, they differ -- and every row says so)
------------------------------------------------------------------------
* full_copy / full_copy_pageable: each rank moves `size` D2H and `size` H2D
  through its own host buffer, concurrently with the other ranks, so
  moved_bytes = 2 * size per rank; gbps_basis = per_rank_roundtrip.
* collective: rank 0 sends `size` bytes to each of world_size-1 consumers,
  so moved_bytes = size * (world_size - 1); gbps_basis = aggregate_broadcast.
* reshard: each rank receives size * (world_size-1) / world_size into its
  gathered output, so aggregate moved_bytes = size * (world_size - 1);
  gbps_basis = aggregate_allgather.
Every result row carries `moved_bytes`, `byte_model` (this prose, per row)
and `gbps_basis`. Two rows whose gbps_basis differ are NOT comparable as
raw GB/s, and the row itself is what says so; comparing them without
reading the basis is a reader error the schema is designed to surface.
gbps values below are moved_bytes / median_seconds / 1e9 (decimal GB).

TIMED-ITERATION CORRECTNESS (poison gate)
-----------------------------------------
A one-time pre-timing equality check does not prove the TIMED iterations
moved anything. Before every timed iteration k the payload sources are
rewritten with an iteration-distinct deterministic pattern and every
destination is poisoned with a per-iteration-distinct scalar provably
absent from the payload family. The timed loop performs NO verification
inside the window. After the window closes and the device is drained, the
destination for the final iteration k_last is verified to hold exactly
payload(k_last) -- neither the poison nor any earlier iteration's payload
can satisfy that check, so a dropped/no-op final iteration cannot pass as
CLEAR. Verification is reduced across ranks with all_reduce(MIN), so a
consumer that did not receive the bytes fails the row even when rank 0
passes locally. A cell whose destination is unverified is UNMEASURED.

STABILITY PRECONDITION AND ADAPTIVE SAMPLING
--------------------------------------------
* Three warmups are not enough. Warmup runs to a size-dependent floor
  (small payloads get more, large fewer; warmup is dominated by per-call
  effects) AND the last three warmup durations -- reduced MAX across ranks
  so every rank sees the same series -- must agree within WARMUP_TOL. The
  loop extends up to WARMUP_CAP; a cell that never stabilises is
  UNMEASURED with the last-three durations named, never published as CLEAR.
* Ten samples are weak at the 1 MiB floor. The timed repeat count is scaled
  inversely with the measured cost of one iteration against a wall-clock
  budget, clamped to [REPEAT_MIN, REPEAT_MAX]; every row emits n_samples
  and the IQR so precision is visible rather than assumed.

CACHE AND ALLOCATOR DISCIPLINE
------------------------------
* Buffer reuse measures cache, not transport. Each arm rotates over a ring
  of >= RING_BUFFERS (=3) independently pre-allocated source/destination
  buffer sets. All allocation happens BEFORE any timing, so allocator cost
  is never inside the window, and empty_cache() is never called anywhere
  inside warmup or the timed loop (only between cells, after timing has
  fully closed).

LATE NCCL FAILURE POLICY
------------------------
TORCH_NCCL_ASYNC_ERROR_HANDLING=1 is set and recorded before communicator
creation. An asynchronous NCCL fault can surface AFTER rows were already
classified successful, so every (transport, size) cell ends with a barrier
plus a counting health probe, any escaped late failure retro-invalidates
all rows recorded so far to UNMEASURED if the process group then fails
health, and a final health check runs before adjudication. A row must
never stay CLEAR because the error arrived late.

RECORDED CONTEXT (recorded, never silently accepted, never modified)
--------------------------------------------------------------------
* NCCL configuration: NCCL_ALGO, NCCL_PROTO, NCCL_NVLS_ENABLE, NCCL_IB_HCA,
  NCCL_SOCKET_IFNAME, NCCL_DEBUG (unset shown as null), plus
  TORCH_NCCL_ASYNC_ERROR_HANDLING / TORCH_NCCL_BLOCKING_WAIT,
  CUDA_LAUNCH_BLOCKING and PYTORCH_CUDA_ALLOC_CONF.
* torch / CUDA / NCCL versions, TF32 and cudnn flags, and per rank the
  device name + GPU UUID.
* Per rank: the NUMA node(s) of the CPU set the rank is pinned to
  (os.sched_getaffinity -> /sys/devices/system/node), the raw affinity
  list, and a pre-run occupancy probe (free vs total device memory).
  Affinity is RECORDED, not changed; this tray is on a SHARED account and
  if another tenant's footprint exceeds FOREIGN_BYTES_LIMIT the run is
  reported UNMEASURED rather than publishing a contended number.

WHAT IS NOT MEASURED
--------------------
* The filesystem (no checkpoint is written or read).
* The vLLM / inference-process boundary (no IPC, no shared-memory handoff,
  no serializer; the full_copy arms emulate it with a per-rank host buffer).
* Inter-node transport: this is a single-tray (single-node) measurement only.
* The CPU-side cost of building the state_dict out of the training model.
* Any common timebase between ranks: there is none and none is implied.

CORRECTNESS GATE
----------------
Every arm must move a deterministic, iteration-distinct payload EXACTLY
(torch.equal on bf16 bits) to a destination that was poisoned immediately
beforehand; the check runs after the timed window closes and is reduced
across all ranks. An arm that cannot be shown to have moved the bytes is
reported UNMEASURED and its timing is withheld.

EXIT CODES (four-state contract; only rank 0 decides, all ranks follow)
----------------------------------------------------------------------
  0  CLEAR       every arm measured, control fired with margin
  5  RED         the control failed (pageable not slower than pinned by the
                 predeclared margin with non-overlapping intervals)
  95 UNMEASURED  at least one arm could not be measured (verification
                 failed, warmup never stabilised, contention, late NCCL
                 failure, world_size < 2, or no CUDA device present) --
                 the gate degrades to UNMEASURED, never crashes, never RED
  96 REFUSE      preconditions absent (not under torchrun, torch built
                 without NCCL, invalid LOCAL_RANK, or pin_memory not
                 honoured while constructing the control arms)

Launch:
  torchrun --nproc_per_node=4 this_module.py --out result.json
Self-test (no GPU, no torch.distributed; exercises the adjudication and
control layer over synthetic rows):
  python3 this_module.py --self-test
"""

import argparse
import json
import math
import os
import random
import statistics
import sys
from typing import Any, Callable, NamedTuple

# Finding #354. torch is a REAL requirement of the measurement path and NOT a
# requirement of --self-test, which the docstring above already declares runnable
# with "no GPU, no torch.distributed" over synthetic rows. Importing it at module
# scope collapsed that distinction: on any interpreter without torch the module
# died with a ModuleNotFoundError traceback and exit 1 -- a code outside the
# 0/5/95/96 contract this file publishes twelve lines above, and a verdict that
# depended on which python3 happened to be first on PATH (#83/#111/#229 again).
#
# `from __future__ import annotations` is on line 1, so the 16 `torch.device` and
# `torch.Tensor` annotations below are strings at runtime and cost nothing when the
# binding is None. Only genuine runtime uses reach the guard in main().
try:
    import torch
    import torch.distributed as dist

    TORCH_IMPORT_ERROR: str | None = None
except ImportError as _torch_exc:
    torch = None  # type: ignore[assignment]
    dist = None  # type: ignore[assignment]
    TORCH_IMPORT_ERROR = str(_torch_exc)

# --------------------------------------------------------------------------
# Contract constants
# --------------------------------------------------------------------------
EXIT_CLEAR: int = 0
EXIT_RED: int = 5
EXIT_UNMEASURED: int = 95
EXIT_REFUSE: int = 96

MIB: int = 1 << 20
GIB: int = 1 << 30

# Sweep list is a module constant so it can be shortened by a flag.
SIZES_BYTES: list[int] = [64 * MIB, 256 * MIB, 1 * GIB, 4 * GIB, 16 * GIB]
PER_CALL_FLOOR_BYTES: int = 1 * MIB

WARMUP_DEFAULT: int = 3
REPEAT_DEFAULT: int = 10

# Stability precondition (#8): warmups run to a size-dependent floor and
# must end with three consecutive durations agreeing within WARMUP_TOL,
# extending up to WARMUP_CAP before the cell is declared UNMEASURED.
WARMUP_TOL_DEFAULT: float = 0.10
WARMUP_CAP: int = 40
WARMUP_REF_BYTES: int = 32 * MIB

# Adaptive sampling (#18): repeat scales inversely with the measured
# per-iteration cost against a wall-clock budget, clamped.
REPEAT_MIN: int = 10
REPEAT_MAX: int = 500
REPEAT_BUDGET_S_DEFAULT: float = 0.5

# Cache discipline (#9/#10): rotate over this many independent buffer sets.
RING_BUFFERS: int = 3

# Positive control (#17): predeclared margin and interval discipline.
CONTROL_MARGIN_DEFAULT: float = 0.05
BOOTSTRAP_RESAMPLES: int = 2000
BOOTSTRAP_SEED: int = 20260725

# Shared-account guard (#12): device memory already held by someone else
# beyond this footprint means the number would be contended.
FOREIGN_BYTES_LIMIT: int = 2 * GIB

TRANSPORTS: tuple[str, ...] = (
    "full_copy",
    "full_copy_pageable",
    "collective",
    "reshard",
)

GBPS_BASES: tuple[str, str, str] = (
    "per_rank_roundtrip",
    "aggregate_broadcast",
    "aggregate_allgather",
)

NOT_MEASURED: list[str] = [
    "filesystem checkpoint write/read",
    "vLLM/inference process boundary (no IPC or shared-memory handoff)",
    "inter-node transport (single GB200 tray only)",
    "CPU-side cost of building the state_dict",
    "any common cross-rank timebase (there is none; aggregates use the "
    "slowest rank's median, named as such)",
]

NCCL_ENV_KEYS: tuple[str, ...] = (
    "NCCL_ALGO",
    "NCCL_PROTO",
    "NCCL_NVLS_ENABLE",
    "NCCL_IB_HCA",
    "NCCL_SOCKET_IFNAME",
    "NCCL_DEBUG",
    "TORCH_NCCL_ASYNC_ERROR_HANDLING",
    "TORCH_NCCL_BLOCKING_WAIT",
    "CUDA_LAUNCH_BLOCKING",
    "PYTORCH_CUDA_ALLOC_CONF",
)

TIMEBASE_NOTE: str = (
    "cross-rank timings share no common clock; each rank's median is a "
    "per-rank quantity, and every aggregate figure is derived from the "
    "SLOWEST rank's median and named as such"
)

_PATTERN_CHUNK: int = 1 << 24  # pattern is generated/verified in chunks

SCHEMA: str = "foundationscale.weight_sync_bench/v1"

_VERDICTS: tuple[str, ...] = ("CLEAR", "RED", "UNMEASURED", "REFUSE")


class _PreconditionRefusal(Exception):
    """Construction-refusal (exit 96). Deliberately NOT a RuntimeError so the
    per-cell allocation gate cannot swallow it into an UNMEASURED row."""


class _LateFailure(RuntimeError):
    """A late/asynchronous failure surfaced by the post-cell health check."""


# --------------------------------------------------------------------------
# Deterministic payload pattern (rank- AND index-dependent; never zeros,
# never constant -- a memcpy of zeros and a no-op are indistinguishable).
# Payload(k) at index i is idx*1.0009765625 + (3 + seed(k)*977) in fp32
# rounded to bf16; with seed >= 0 every value is >= 3, so any NEGATIVE
# scalar is provably absent from every payload: that is the poison.
# Distinct k differ at index 0 by 977, far above the bf16 ulp there.
# --------------------------------------------------------------------------
def _pattern_chunk(offset: int, n: int, seed: float, device: torch.device) -> torch.Tensor:
    idx = torch.arange(offset, offset + n, dtype=torch.float32, device=device)
    idx.mul_(1.0009765625).add_(3.0 + seed * 977.0)
    return idx.to(torch.bfloat16)


def _pattern_into(out: torch.Tensor, offset: int, seed: float) -> None:
    numel = out.numel()
    start = 0
    while start < numel:
        n = min(_PATTERN_CHUNK, numel - start)
        out[start : start + n] = _pattern_chunk(offset + start, n, seed, out.device)
        start += n


def _verify_pattern(t: torch.Tensor, offset: int, seed: float) -> bool:
    numel = t.numel()
    start = 0
    while start < numel:
        n = min(_PATTERN_CHUNK, numel - start)
        if not bool(torch.equal(t[start : start + n], _pattern_chunk(offset + start, n, seed, t.device))):
            return False
        start += n
    return True


def _iteration_seed(base: float, k: int) -> float:
    return base + float(k)


def _poison_value(k: int) -> float:
    return -float(k + 2)  # per-iteration-distinct and provably not payload


# --------------------------------------------------------------------------
# Arm construction. Each builder returns an ArmBundle over a ring of
# RING_BUFFERS independent buffer sets, all allocated before any timing:
#   prime(k)  -- untimed: rewrite payload(k) sources, poison destinations
#   op(k)     -- the timed transport for iteration k (ring slot k % ring)
#   verify(k) -- untimed: destination holds payload(k) exactly
# moved_bytes semantics are per-transport; gbps_basis labels which
# definition the row's GB/s rests on. See the module docstring.
# --------------------------------------------------------------------------
class ArmBundle(NamedTuple):
    prime: Callable[[int], None]
    op: Callable[[int], None]
    verify: Callable[[int], bool]
    moved_bytes: int
    byte_model: str
    gbps_basis: str
    keepalive: list[torch.Tensor]


def _arm_full_copy(size: int, pinned: bool, rank: int, device: torch.device) -> ArmBundle:
    numel = size // 2
    srcs: list[torch.Tensor] = []
    hosts: list[torch.Tensor] = []
    dsts: list[torch.Tensor] = []
    for _ in range(RING_BUFFERS):
        srcs.append(torch.empty(numel, dtype=torch.bfloat16, device=device))
        hosts.append(torch.empty(numel, dtype=torch.bfloat16, pin_memory=pinned))
        dsts.append(torch.empty(numel, dtype=torch.bfloat16, device=device))

    # Control-arm construction assertion (#17a): the host allocator must
    # have honoured pin_memory exactly as requested, else the pinned vs
    # pageable comparison measures nothing and the harness must REFUSE.
    honoured = all(bool(h.is_pinned()) for h in hosts)
    if pinned and not honoured:
        raise _PreconditionRefusal(
            "pin_memory=True was not honoured by the host allocator; the "
            "pinned/pageable positive control cannot be constructed honestly"
        )
    if not pinned and any(bool(h.is_pinned()) for h in hosts):
        raise _PreconditionRefusal(
            "a nominally pageable control buffer is pinned; the positive "
            "control cannot be constructed honestly"
        )

    base = float(rank)

    def prime(k: int) -> None:
        j = k % RING_BUFFERS
        _pattern_into(srcs[j], 0, _iteration_seed(base, k))
        dsts[j].fill_(_poison_value(k))

    def op(k: int) -> None:
        j = k % RING_BUFFERS
        hosts[j].copy_(srcs[j], non_blocking=True)
        dsts[j].copy_(hosts[j], non_blocking=True)

    def verify(k: int) -> bool:
        j = k % RING_BUFFERS
        return _verify_pattern(dsts[j], 0, _iteration_seed(base, k))

    moved = 2 * size  # per rank: size D2H + size H2D, ranks copy concurrently
    model = (
        "per-rank D2H(size) + H2D(size) round trip through a %s host buffer; "
        "ranks copy concurrently; host buffer is per-rank (process-boundary "
        "handoff NOT measured)"
    ) % ("pinned" if pinned else "pageable")
    keepalive: list[torch.Tensor] = []
    keepalive.extend(srcs)
    keepalive.extend(hosts)
    keepalive.extend(dsts)
    return ArmBundle(prime, op, verify, moved, model, "per_rank_roundtrip", keepalive)


def _arm_collective(size: int, rank: int, world: int, device: torch.device) -> ArmBundle:
    numel = size // 2
    bufs = [torch.empty(numel, dtype=torch.bfloat16, device=device) for _ in range(RING_BUFFERS)]

    def prime(k: int) -> None:
        j = k % RING_BUFFERS
        if rank == 0:
            _pattern_into(bufs[j], 0, _iteration_seed(0.0, k))
        else:
            bufs[j].fill_(_poison_value(k))  # consumer buffer starts provably stale

    def op(k: int) -> None:
        # Blocking form: with async_op=False, torch (2.13.0, distributed_c10d
        # broadcast) waits on the work itself, so the end event recorded
        # after this call returns is correctly ordered behind the NCCL work.
        dist.broadcast(bufs[k % RING_BUFFERS], src=0)

    def verify(k: int) -> bool:
        return _verify_pattern(bufs[k % RING_BUFFERS], 0, _iteration_seed(0.0, k))

    moved = size * (world - 1)  # rank0 -> size bytes to each of world-1 consumers
    model = "rank0 broadcasts size bytes to each of world_size-1 consumers"
    return ArmBundle(prime, op, verify, moved, model, "aggregate_broadcast", list(bufs))


def _arm_reshard(size: int, rank: int, world: int, device: torch.device) -> ArmBundle:
    numel = size // 2
    if numel % world != 0:
        raise ValueError("numel %d not divisible by world_size %d" % (numel, world))
    shard_n = numel // world
    shards: list[torch.Tensor] = []
    outs: list[torch.Tensor] = []
    for _ in range(RING_BUFFERS):
        shards.append(torch.empty(shard_n, dtype=torch.bfloat16, device=device))
        outs.append(torch.empty(numel, dtype=torch.bfloat16, device=device))

    def prime(k: int) -> None:
        j = k % RING_BUFFERS
        _pattern_into(shards[j], rank * shard_n, _iteration_seed(0.0, k))
        outs[j].fill_(_poison_value(k))

    def op(k: int) -> None:
        dist.all_gather_into_tensor(outs[k % RING_BUFFERS], shards[k % RING_BUFFERS])

    def verify(k: int) -> bool:
        return _verify_pattern(outs[k % RING_BUFFERS], 0, _iteration_seed(0.0, k))

    moved = size * (world - 1)  # each rank receives size*(world-1)/world; aggregate over ranks
    model = (
        "each rank receives size*(world_size-1)/world_size into its output; "
        "aggregate moved_bytes = size*(world_size-1)"
    )
    keepalive = list(shards)
    keepalive.extend(outs)
    return ArmBundle(prime, op, verify, moved, model, "aggregate_allgather", keepalive)


def _build_arm(transport: str, size: int, rank: int, world: int, device: torch.device) -> ArmBundle:
    if transport == "full_copy":
        return _arm_full_copy(size, True, rank, device)
    if transport == "full_copy_pageable":
        return _arm_full_copy(size, False, rank, device)
    if transport == "collective":
        return _arm_collective(size, rank, world, device)
    if transport == "reshard":
        return _arm_reshard(size, rank, world, device)
    raise ValueError("unknown transport %r" % (transport,))


# --------------------------------------------------------------------------
# Distributed helpers
# --------------------------------------------------------------------------
def _unanimous(value: bool, device: torch.device) -> bool:
    flag = torch.tensor([1 if value else 0], dtype=torch.int64, device=device)
    dist.all_reduce(flag, op=dist.ReduceOp.MIN)
    return int(flag.item()) == 1


def _fan_in_reason(local: str | None, world: int) -> str:
    bag: list[Any] = [None] * world
    dist.all_gather_object(bag, local)
    for item in bag:
        if item:
            return str(item)
    return local or "unknown failure"


def _health_check(device: torch.device, world: int) -> str | None:
    """Barrier plus a counting probe. Returns None when healthy, else the
    reason -- used to RETRO-INVALIDATE rows an async NCCL failure may have
    stranded (a row must never stay CLEAR because the error arrived late)."""
    try:
        dist.barrier()
        probe = torch.ones(1, dtype=torch.int64, device=device)
        dist.all_reduce(probe, op=dist.ReduceOp.SUM)
        torch.cuda.synchronize()
    except RuntimeError as exc:
        return "post-cell barrier/health probe raised: %s" % (exc,)
    got = int(probe.item())
    if got != world:
        return "health probe counted %d ranks, expected %d" % (got, world)
    return None


def _retro_invalidate(rows: list[dict[str, Any]], unmeasured: list[dict[str, Any]], reason: str) -> None:
    """Demote every already-recorded row to UNMEASURED. A late async failure
    that proves the process group was unhealthy withdraws all prior evidence."""
    for r in list(rows):
        unmeasured.append({
            "transport": r["transport"],
            "bytes": r["bytes"],
            "label": r["label"],
            "reason": reason,
            "retro_invalidated": True,
        })
    rows.clear()


def _record_late(
    unmeasured: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    transport: str,
    size: int,
    label: str,
    exc: BaseException,
    device: torch.device,
    world: int,
) -> bool:
    """Record a cell that threw late; if the process group is now unhealthy,
    retro-invalidate everything recorded so far and stop the sweep."""
    unmeasured.append({
        "transport": transport,
        "bytes": size,
        "label": label,
        "reason": "late or asynchronous failure escaped the cell: %s" % (exc,),
        "late_failure": True,
    })
    health = _health_check(device, world)
    if health is not None:
        _retro_invalidate(
            rows,
            unmeasured,
            "process group unhealthy after a late failure (%s); all prior "
            "rows retro-invalidated rather than admitted as evidence" % (health,),
        )
        return True
    return False


# --------------------------------------------------------------------------
# Warmup-to-stability and the timed window. Neither phase ever verifies or
# calls empty_cache() inside its loop; priming is untimed and synchronised
# away before the barrier that opens each timed iteration.
# --------------------------------------------------------------------------
def _stabilised_warmup(
    bundle: ArmBundle,
    device: torch.device,
    k0: int,
    min_warmup: int,
    cap: int,
    tol: float,
) -> tuple[int, list[float], str | None]:
    """Run untimed iterations until the last three MAX-reduced durations
    agree within tol of their median (same reduced series on every rank, so
    the stability decision is identical everywhere). Returns the next
    iteration index, the discarded warmup durations, and a failure reason
    if the cap was hit without stabilising."""
    scratch = torch.empty(1, dtype=torch.float64, device=device)
    durs: list[float] = []
    k = k0
    while True:
        bundle.prime(k)
        torch.cuda.synchronize()
        dist.barrier()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        bundle.op(k)
        end.record()
        torch.cuda.synchronize()
        scratch.fill_(start.elapsed_time(end) / 1000.0)
        # Warmup stability is judged on the slowest rank so every rank sees
        # the same series and takes the same decision without extra traffic.
        dist.all_reduce(scratch, op=dist.ReduceOp.MAX)
        durs.append(float(scratch.item()))
        k += 1
        if len(durs) >= min_warmup:
            last3 = durs[-3:]
            med = float(statistics.median(last3))
            if med > 0.0 and (max(last3) - min(last3)) / med <= tol:
                return k, durs, None
            if med == 0.0 and max(last3) == 0.0:
                return k, durs, None
            if len(durs) >= cap:
                return k, durs, (
                    "warmup never stabilised: last three durations %s s, spread "
                    "%.1f%% of median exceeds tolerance %.1f%% after %d "
                    "iterations (cap %d)"
                    % (
                        ["%.6f" % d for d in last3],
                        (100.0 * (max(last3) - min(last3)) / med) if med > 0.0 else float("inf"),
                        100.0 * tol,
                        len(durs),
                        cap,
                    )
                )


def _measure(
    bundle: ArmBundle,
    device: torch.device,
    k0: int,
    repeat: int,
) -> tuple[int, list[float]]:
    """The timed window. Per iteration: poison+prime (untimed) -> drain ->
    barrier -> CUDA events around the blocking op -> drain. Each rank keeps
    its OWN samples; there is no common clock and none is implied. No
    verification, no allocation, no empty_cache inside the loop."""
    samples: list[float] = []
    k = k0
    for _ in range(repeat):
        bundle.prime(k)
        torch.cuda.synchronize()  # priming fully drained before the window opens
        dist.barrier()  # ranks start the timed region together
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        bundle.op(k)
        end.record()
        torch.cuda.synchronize()  # events resolved before reading elapsed_time
        samples.append(start.elapsed_time(end) / 1000.0)
        k += 1
    return k, samples


def _min_warmup(size: int, base_warmup: int) -> int:
    """Size-dependent warmup floor: small payloads pay per-call effects and
    get more warmups; large payloads reach steady state in the base count."""
    scaled = max(1, int(math.ceil(WARMUP_REF_BYTES / float(size))))
    return min(WARMUP_CAP, max(base_warmup, scaled))


def _choose_repeat(est_iter_s: float, floor: int, budget_s: float) -> int:
    """Adaptive repeat (#18): inversely proportional to the measured cost of
    one iteration against a wall-clock budget, clamped to [REPEAT_MIN, MAX]."""
    lo = max(REPEAT_MIN, floor)
    if est_iter_s <= 0.0:
        return lo
    return int(min(REPEAT_MAX, max(lo, int(round(budget_s / est_iter_s)))))


# --------------------------------------------------------------------------
# One (transport, size) cell: allocation gate -> pre-flight move check ->
# warmup-to-stability -> adaptive timed window -> post-window verification
# -> health check. A failure at any stage degrades THIS row only; a health
# failure escalates to the caller, which retro-invalidates prior evidence.
# --------------------------------------------------------------------------
def _measure_arm(
    transport: str,
    size: int,
    label: str,
    rank: int,
    world: int,
    device: torch.device,
    warmup: int,
    repeat_floor: int,
    budget_s: float,
    tol: float,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    local_reason: str | None = None
    bundle: ArmBundle | None = None

    if world < 2 and transport in ("collective", "reshard"):
        local_reason = "%s requires world_size >= 2, got %d" % (transport, world)
    else:
        try:
            bundle = _build_arm(transport, size, rank, world, device)
        except torch.cuda.OutOfMemoryError:
            local_reason = "CUDA OOM allocating %d-byte payload (ring of %d) for %s" % (
                size, RING_BUFFERS, transport,
            )
            torch.cuda.empty_cache()
        except (RuntimeError, ValueError, MemoryError) as exc:
            local_reason = "allocation failed for %s @ %d B: %s" % (transport, size, exc)

    if not _unanimous(bundle is not None, device):
        reason = _fan_in_reason(local_reason, world)
        return None, {"transport": transport, "bytes": size, "label": label, "reason": reason}

    assert bundle is not None

    # Pre-flight: prove the mechanism CAN move the bytes once, before any
    # timing is spent. (This alone would prove nothing about the timed
    # iterations -- that is the post-window poison gate below.)
    preflight_ok = False
    try:
        bundle.prime(0)
        bundle.op(0)
        torch.cuda.synchronize()
        preflight_ok = bool(bundle.verify(0))
        if not preflight_ok:
            local_reason = "payload equality check FAILED after %s (pre-flight)" % transport
    except torch.cuda.OutOfMemoryError:
        local_reason = "CUDA OOM during pre-flight of %s" % transport
        torch.cuda.empty_cache()
    except RuntimeError as exc:
        local_reason = "runtime error during pre-flight of %s: %s" % (transport, exc)

    if not _unanimous(preflight_ok, device):
        reason = _fan_in_reason(local_reason, world)
        torch.cuda.empty_cache()
        return None, {"transport": transport, "bytes": size, "label": label, "reason": reason}

    # Stability precondition: same reduced series on every rank.
    min_warm = _min_warmup(size, warmup)
    k, warm_durs, stab_fail = _stabilised_warmup(
        bundle, device, 1, min_warm, WARMUP_CAP, tol,
    )
    if stab_fail is not None:
        reason = stab_fail
        if not _unanimous(True, device):
            reason = _fan_in_reason(reason, world)
        del bundle
        torch.cuda.empty_cache()
        return None, {"transport": transport, "bytes": size, "label": label, "reason": reason}

    est_iter_s = float(statistics.median(warm_durs[-3:]))
    repeat = _choose_repeat(est_iter_s, repeat_floor, budget_s)

    # The timed window: poisoned and re-primed before every iteration.
    k, samples = _measure(bundle, device, k, repeat)
    final_k = k - 1

    # Post-window poison gate (#2): the destination must hold EXACTLY the
    # final iteration's payload -- not the poison, not an earlier payload.
    # Runs strictly after the timed window, never inside it, and is reduced
    # across ranks so a consumer that missed the transfer fails the row even
    # when rank 0 passes locally. Unverified destination => UNMEASURED.
    torch.cuda.synchronize()
    dest_ok_local = bool(bundle.verify(final_k))
    if not _unanimous(dest_ok_local, device):
        reason = (
            "timed-window destination verification failed for %s at iteration "
            "%d: a rank's destination could not be verified to hold the "
            "expected post-poison payload, so this row is UNMEASURED, not CLEAR"
            % (transport, final_k)
        )
        del bundle
        torch.cuda.empty_cache()
        return None, {"transport": transport, "bytes": size, "label": label, "reason": reason}

    # Health check closing the cell: an async NCCL failure must not be
    # allowed to strand already-classified rows.
    health = _health_check(device, world)
    if health is not None:
        del bundle
        torch.cuda.empty_cache()
        raise _LateFailure(health)

    local_median = float(statistics.median(samples))
    gathered: list[Any] | None = [None] * world if rank == 0 else None
    dist.gather_object(
        {"median": local_median, "samples": [float(s) for s in samples]},
        gathered,
        dst=0,
    )

    row: dict[str, Any] = {
        "transport": transport,
        "bytes": size,
        "label": label,  # "sweep" or "per_call_floor"
        "n_samples": len(samples),
        "warmup_iterations": len(warm_durs),
        "median_s": local_median,
        "moved_bytes": bundle.moved_bytes,
        "byte_model": bundle.byte_model,
        "gbps_basis": bundle.gbps_basis,
        "poison_gate": "destination poisoned and re-primed per iteration; "
                       "verified only after the timed window closed",
        "equality_ok": True,
        "equality_scope": "verified on every rank, reduced all_reduce(MIN), "
                          "after the timed window",
        "cross_rank_timebase": TIMEBASE_NOTE,
    }
    if rank == 0 and gathered is not None and all(g is not None for g in gathered):
        per_rank = [float(g["median"]) for g in gathered]
        slowest = max(range(world), key=lambda i: per_rank[i])
        slow_samples = [float(x) for x in gathered[slowest]["samples"]]
        aggregate = per_rank[slowest]  # SLOWEST rank's median, named as such
        if len(slow_samples) > 1:
            q1, _, q3 = statistics.quantiles(slow_samples, n=4)
        else:
            q1 = q3 = slow_samples[0]
        row.update({
            "median_s": aggregate,
            "slowest_rank": slowest,
            "per_rank_median_s": per_rank,
            "min_s": float(min(per_rank)),
            "max_s": float(max(per_rank)),
            "q1_s": float(q1),
            "q3_s": float(q3),
            "iqr_s": float(q3 - q1),
            "samples_s": slow_samples,
            "gbps_median": (bundle.moved_bytes / aggregate) / 1e9 if aggregate > 0.0 else 0.0,
            "aggregate": "median is the SLOWEST rank's median (named as such); "
                         "there is no common clock across ranks",
        })
    else:
        row["samples_s"] = [float(s) for s in samples]

    del bundle
    torch.cuda.empty_cache()  # between cells, after timing fully closed
    return row, None


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------
def _render_table(rows: list[dict[str, Any]], unmeasured: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    lines.append("%-20s %12s %14s %6s %12s %12s %12s %12s %12s  %-22s %s" % (
        "transport", "payload", "label", "n", "median_s", "rankmin_s", "rankmax_s",
        "iqr_s", "GB/s(med)", "gbps_basis", "moved_bytes",
    ))
    lines.append("-" * 156)
    for r in rows:
        lines.append("%-20s %12d %14s %6d %12.6f %12.6f %12.6f %12.6f %12.2f  %-22s %d" % (
            r["transport"], r["bytes"], r["label"],
            r.get("n_samples", 0),
            r["median_s"], r.get("min_s", 0.0), r.get("max_s", 0.0),
            r.get("iqr_s", 0.0), r.get("gbps_median", 0.0),
            r.get("gbps_basis", "?"), r["moved_bytes"],
        ))
    for u in unmeasured:
        lines.append("%-20s %12d %14s  -- UNMEASURED: %s" % (
            u["transport"], u["bytes"], u["label"], u["reason"],
        ))
    return "\n".join(lines)


def _row_discloses_basis(row: dict[str, Any]) -> bool:
    """Schema guard for the byte-definition disclosure (#13): a row without
    all three fields cannot be published."""
    return (
        isinstance(row.get("moved_bytes"), int)
        and isinstance(row.get("byte_model"), str)
        and bool(row.get("byte_model"))
        and row.get("gbps_basis") in GBPS_BASES
    )


def _bootstrap_median_ci(samples: list[float], resamples: int, seed: int) -> tuple[float, float]:
    """Percentile bootstrap 95% interval for the median, deterministically
    seeded so the control decision is reproducible."""
    if not samples:
        return (0.0, 0.0)
    if len(samples) == 1:
        return (samples[0], samples[0])
    rng = random.Random(seed)
    n = len(samples)
    meds = [float(statistics.median(rng.choices(samples, k=n))) for _ in range(resamples)]
    meds.sort()
    lo = meds[int(0.025 * (resamples - 1))]
    hi = meds[int(0.975 * (resamples - 1))]
    return (lo, hi)


def _adjudicate(
    rows: list[dict[str, Any]],
    unmeasured: list[dict[str, Any]],
    world: int,
    margin: float,
) -> tuple[dict[str, Any], str, int]:
    """Positive control (#17): pageable full_copy MUST be slower than pinned
    full_copy at every sweep size where both were measured, by BOTH
      (a) a predeclared margin: pageable median >= pinned median * (1+margin)
      (b) non-overlapping bootstrap 95%% intervals for the two medians.
    A bare inequality is not evidence: a 0.1%% difference 'firing' the control
    is a control that failed to fire. On failure the numbers are named. If the
    control cannot validate the harness, the run is RED, not CLEAR."""
    by_key: dict[tuple[str, int], dict[str, Any]] = {}
    for r in rows:
        if r["label"] == "sweep":
            by_key[(r["transport"], r["bytes"])] = r
    pairs: list[int] = sorted(
        size for size in {s for (_, s) in by_key}
        if ("full_copy", size) in by_key and ("full_copy_pageable", size) in by_key
    )
    violations: list[dict[str, Any]] = []
    observations: list[dict[str, Any]] = []
    pair_details: list[dict[str, Any]] = []
    required_ratio = 1.0 + margin
    for size in pairs:
        pinned_med = float(by_key[("full_copy", size)]["median_s"])
        pageable_med = float(by_key[("full_copy_pageable", size)]["median_s"])
        ci_pinned = _bootstrap_median_ci(
            list(by_key[("full_copy", size)].get("samples_s", [pinned_med])),
            BOOTSTRAP_RESAMPLES, BOOTSTRAP_SEED + size,
        )
        ci_pageable = _bootstrap_median_ci(
            list(by_key[("full_copy_pageable", size)].get("samples_s", [pageable_med])),
            BOOTSTRAP_RESAMPLES, BOOTSTRAP_SEED + size + 1,
        )
        observed_ratio = pageable_med / pinned_med if pinned_med > 0.0 else 0.0
        overlap = not (ci_pageable[0] > ci_pinned[1])
        reasons: list[str] = []
        if pageable_med < pinned_med * required_ratio:
            reasons.append(
                "margin not met: pageable median %.6fs is only %.2f%% above "
                "pinned median %.6fs; the predeclared margin is >= %.1f%%"
                % (
                    pageable_med, 100.0 * (observed_ratio - 1.0),
                    pinned_med, 100.0 * margin,
                )
            )
        if overlap:
            reasons.append(
                "bootstrap 95%% intervals overlap: pinned [%.6f, %.6f], "
                "pageable [%.6f, %.6f]" % (
                    ci_pinned[0], ci_pinned[1], ci_pageable[0], ci_pageable[1],
                )
            )
        # The elision question has TWO instruments and they are not equal.
        # The poison gate is DIRECT: the destination is poisoned and re-primed
        # before every timed iteration and must hold exactly the final
        # iteration's payload afterwards, or the row never reaches this
        # function at all -- it is demoted to UNMEASURED at :735. The
        # pinned-vs-pageable margin is an INDIRECT proxy for the same
        # question, and it is hardware-calibrated. On coherent-memory parts
        # (Grace-Blackwell NVLink-C2C) a large payload is bandwidth-bound and
        # the staging advantage washes out: measured on 4x GB200 the gap is
        # 71% at 1 MiB, 6.0% at 64 MiB and 0.75% at 256 MiB, so one global
        # margin REDs a run whose bytes provably moved (#321).
        #
        # So key on WHICH INSTRUMENT ANSWERED, never on the hardware -- a
        # "GB200 exception" would be an allowlist over our own residue, and
        # widening the margin globally would blind the control at 1 MiB and
        # 64 MiB where it works and carries real signal. Where the poison gate
        # covered BOTH rows it has already refuted elision, so the margin is
        # recorded as an OBSERVATION; where it did not, the margin is the only
        # instrument left and still adjudicates RED.
        poison_covered = bool(
            by_key[("full_copy", size)].get("poison_gate")
            and by_key[("full_copy_pageable", size)].get("poison_gate")
        )
        detail = {
            "bytes": size,
            "pinned_median_s": pinned_med,
            "pageable_median_s": pageable_med,
            "observed_ratio": observed_ratio,
            "required_ratio": required_ratio,
            "ci95_pinned_s": [ci_pinned[0], ci_pinned[1]],
            "ci95_pageable_s": [ci_pageable[0], ci_pageable[1]],
            "intervals_overlap": overlap,
            "poison_gate_covered": poison_covered,
            "elision_adjudicated_by": (
                "poison_gate" if poison_covered else "pinned_pageable_margin"
            ),
            "ok": not reasons,
            "reasons": reasons,
        }
        pair_details.append(detail)
        if reasons and poison_covered:
            observations.append(detail)
        elif reasons:
            violations.append(detail)
    control = {
        "name": "pageable_slower_than_pinned",
        "fired": len(pairs) > 0,
        "paired_sizes": pairs,
        "ok": len(pairs) > 0 and not violations,
        "margin_required": margin,
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "violations": violations,
        "observations": observations,
        "pair_details": pair_details,
        "detail": (
            "full_copy_pageable must beat pinned full_copy upward by the "
            "predeclared margin (%.1f%%) at the medians AND the two bootstrap "
            "95%% intervals must not overlap, at every paired sweep size; a "
            "bare inequality is treated as a control that failed to fire. "
            "pin_memory honouring is asserted at arm construction (refusal "
            "96) so the control cannot silently compare pinned with pinned. "
            "This control is an INDIRECT elision detector and its margin is "
            "hardware-calibrated (#321): where the poison gate covered both "
            "rows of a pair it has already refuted elision directly, so a "
            "sub-margin gap there is recorded under 'observations' and does "
            "not sink the verdict; where the poison gate did NOT cover the "
            "pair this control is the only instrument left and a sub-margin "
            "gap is a violation. Each pair names its adjudicator in "
            "'elision_adjudicated_by'"
        ) % (100.0 * margin),
    }

    if world < 2:
        return control, "UNMEASURED", EXIT_UNMEASURED
    if control["fired"] and violations:
        return control, "RED", EXIT_RED
    if not rows:
        # Zero measured units is UNMEASURED, never PASS.
        control["ok"] = False
        control["zero_units"] = (
            "no (transport, size) cells were measured; zero units is "
            "UNMEASURED by contract"
        )
        return control, "UNMEASURED", EXIT_UNMEASURED
    if not control["fired"]:
        # No paired sweep size ever exercised the elision control, so no
        # timing in this run is admitted as evidence at all.
        control["ok"] = False
        return control, "UNMEASURED", EXIT_UNMEASURED
    if unmeasured:
        # A withheld cell narrows what the verdict COVERS; it does not
        # discard the cells that did measure. This mirrors the standing-gate
        # plane's "GREEN WITH DECLARED ABSTENTIONS" (#322): the run is clear
        # over a NAMED denominator and certifies nothing whatsoever about
        # the withheld cells.
        #
        # The subtlety that makes this safe to exit 0 on: a withheld cell can
        # leave its pinned/pageable PAIR incomplete, and the control skips an
        # incomplete pair silently. So the sizes the control actually
        # adjudicated are recorded explicitly and the withheld cells are
        # named -- a clear verdict must never read as covering a payload size
        # the elision control never examined.
        control["abstentions"] = [
            {
                "transport": u.get("transport"),
                "bytes": u.get("bytes"),
                "label": u.get("label"),
                "reason": u.get("reason"),
            }
            for u in unmeasured
        ]
        control["elision_adjudicated_sizes"] = sorted(
            {int(d["bytes"]) for d in pair_details}
        )
        control["abstention_detail"] = (
            "%d of %d attempted cells were withheld and are named in "
            "'abstentions'; this verdict certifies NOTHING about them. The "
            "elision control adjudicated only the payload sizes listed in "
            "'elision_adjudicated_sizes' -- a size whose pinned/pageable pair "
            "lost a row is not among them and was never examined (#322)."
            % (len(unmeasured), len(rows) + len(unmeasured))
        )
        return control, "CLEAR_WITH_ABSTENTIONS", EXIT_CLEAR
    return control, "CLEAR", EXIT_CLEAR


# --------------------------------------------------------------------------
# Recorded context: NCCL configuration, per-rank facts, and the shared-
# account occupancy probe. Recorded, never silently accepted, never changed.
# --------------------------------------------------------------------------
def _capture_env() -> dict[str, str | None]:
    return {k: os.environ.get(k) for k in NCCL_ENV_KEYS}


def _expand_cpulist(spec: str) -> set[int]:
    out: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            out.update(range(int(a), int(b) + 1))
        else:
            out.add(int(part))
    return out


def _cpu_numa_nodes(affinity: list[int]) -> list[int] | None:
    """NUMA nodes of the CPU set this rank is pinned to, via
    /sys/devices/system/node/nodeN/cpulist. Record-only; returns None where
    the sysfs topology is unavailable."""
    try:
        root = "/sys/devices/system/node"
        aff = set(affinity)
        nodes: list[int] = []
        for name in sorted(os.listdir(root)):
            if not name.startswith("node"):
                continue
            try:
                node_id = int(name[4:])
            except ValueError:
                continue
            try:
                with open(os.path.join(root, name, "cpulist"), "r", encoding="utf-8") as fh:
                    spec = fh.read()
            except OSError:
                continue
            if aff & _expand_cpulist(spec):
                nodes.append(node_id)
        return nodes
    except OSError:
        return None


def _rank_facts(rank: int, local_rank: int, device: torch.device) -> dict[str, Any]:
    props = torch.cuda.get_device_properties(device)
    mem_free, mem_total = torch.cuda.mem_get_info(device)
    try:
        affinity = sorted(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        affinity = []
    return {
        "rank": rank,
        "local_rank": local_rank,
        "device_name": str(props.name),
        "gpu_uuid": str(props.uuid),
        "capability": "%d.%d" % (props.major, props.minor),
        "cpu_affinity": affinity,
        "numa_nodes": _cpu_numa_nodes(affinity),
        "mem_free_bytes": int(mem_free),
        "mem_total_bytes": int(mem_total),
        "mem_used_at_probe_bytes": int(mem_total - mem_free),
    }


def _occupancy_gate(
    facts: list[dict[str, Any]],
) -> tuple[bool, str]:
    """Shared account: if another tenant's footprint on any rank's device
    exceeds FOREIGN_BYTES_LIMIT, the numbers would be contended -- report
    UNMEASURED rather than publishing them."""
    foreign: list[str] = []
    for f in facts:
        if int(f["mem_used_at_probe_bytes"]) > FOREIGN_BYTES_LIMIT:
            foreign.append(
                "rank %d device %s already holds %.2f GiB at probe "
                "(limit %.2f GiB) -- another tenant is present"
                % (
                    f["rank"], f["gpu_uuid"],
                    f["mem_used_at_probe_bytes"] / float(GIB),
                    FOREIGN_BYTES_LIMIT / float(GIB),
                )
            )
    return (len(foreign) > 0), "; ".join(foreign)


def _all_cells(sizes: list[int]) -> list[tuple[int, str]]:
    cells = [(s, "sweep") for s in sizes]
    cells.append((PER_CALL_FLOOR_BYTES, "per_call_floor"))
    return cells


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------
def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Weight-sync transport cost on GB200")
    p.add_argument("--out", default="weight_sync_transport_gb200.json",
                   help="JSON output path (rank 0 only)")
    p.add_argument("--warmup", type=int, default=WARMUP_DEFAULT,
                   help="minimum warmup iterations per cell; the floor also "
                        "scales up for small payloads, and warmups extend "
                        "until the last three agree within --warmup-tol "
                        "(cap %d) or the cell is UNMEASURED" % WARMUP_CAP)
    p.add_argument("--repeat", type=int, default=REPEAT_DEFAULT,
                   help="minimum timed samples per cell; the count scales "
                        "inversely with the measured per-iteration cost "
                        "against --repeat-budget-s, clamped to [%d, %d]"
                        % (REPEAT_MIN, REPEAT_MAX))
    p.add_argument("--repeat-budget-s", type=float, default=REPEAT_BUDGET_S_DEFAULT,
                   help="wall-clock budget per cell for the adaptive timed window")
    p.add_argument("--warmup-tol", type=float, default=WARMUP_TOL_DEFAULT,
                   help="stability tolerance for the last three warmups")
    p.add_argument("--control-margin", type=float, default=CONTROL_MARGIN_DEFAULT,
                   help="predeclared control margin: pageable median must exceed "
                        "pinned median by at least this fraction, with "
                        "non-overlapping bootstrap 95%% intervals")
    p.add_argument("--sizes", default=None,
                   help="comma-separated payload sizes in MiB; overrides the sweep constant")
    p.add_argument("--smoke", action="store_true",
                   help="shorten the sweep to the first two entries of SIZES_BYTES")
    p.add_argument("--self-test", action="store_true",
                   help="exercise the adjudication and control layer over "
                        "synthetic rows; no GPU and no torch.distributed "
                        "required (safe for CI on a machine with no cluster)")
    return p.parse_args(argv)


def _resolve_sizes(args: argparse.Namespace) -> list[int]:
    if args.sizes:
        return [max(MIB, (int(tok) * MIB) & ~1) for tok in args.sizes.split(",") if tok.strip()]
    if args.smoke:
        return list(SIZES_BYTES[:2])
    return list(SIZES_BYTES)


def _refuse(msg: str) -> int:
    sys.stderr.write("REFUSE: %s\n" % msg)
    sys.stderr.flush()
    return EXIT_REFUSE


def main(argv: list[str]) -> int:
    args = _parse_args(argv)

    # The self-test harness runs before any cluster precondition: no GPU,
    # no torch.distributed, no torchrun, and -- since #354 -- no torch at all.
    # This is what makes the gate landable in CI on a machine with no cluster.
    if args.self_test:
        return _self_test()

    # --- preconditions ---
    # #354: torch is imported conditionally so an absent install cannot crash the
    # torch-free self-test above. Past this line every arm needs it, and an absent
    # dependency is UNMEASURED -- the same degradation the CUDA check below makes,
    # for the same reason: a measurement that could not be taken is not a failure
    # of the thing being measured.
    if TORCH_IMPORT_ERROR is not None:
        sys.stderr.write(
            f"UNMEASURED: torch is not importable on this interpreter "
            f"({TORCH_IMPORT_ERROR}); the transport measurement needs it, so no "
            f"cell was swept. This is not RED: nothing was measured to be wrong.\n"
        )
        sys.stderr.flush()
        return EXIT_UNMEASURED
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        return _refuse("RANK/WORLD_SIZE not in environment; launch under torchrun")
    if not torch.cuda.is_available():
        # Degrade to UNMEASURED, not a crash and never RED, when no CUDA
        # device is present on the tray.
        sys.stderr.write("UNMEASURED: CUDA is not available on this host\n")
        sys.stderr.flush()
        return EXIT_UNMEASURED
    if not dist.is_nccl_available():
        return _refuse("NCCL backend is not available in this torch build")

    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if local_rank >= torch.cuda.device_count():
        return _refuse("LOCAL_RANK %d but only %d CUDA devices" % (local_rank, torch.cuda.device_count()))

    # Set AND record before communicator creation so an asynchronous NCCL
    # failure surfaces at a synchronisation point instead of never.
    os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "1"

    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    code = EXIT_UNMEASURED
    try:
        dist.init_process_group(backend="nccl")
        try:
            code = _run(args, rank, world, local_rank, device)
        finally:
            dist.destroy_process_group()
    except _PreconditionRefusal as exc:
        # Raised identically on every rank during control-arm construction;
        # the tray shares one host allocator, so all ranks refuse together.
        if rank == 0:
            sys.stderr.write("REFUSE: %s\n" % (exc,))
            sys.stderr.flush()
        return EXIT_REFUSE
    except Exception as exc:  # never let a bare exception escape as exit 1
        if rank == 0:
            sys.stderr.write("UNMEASURED: unexpected harness error: %r\n" % (exc,))
            sys.stderr.flush()
        return EXIT_UNMEASURED
    return code


def _run(
    args: argparse.Namespace,
    rank: int,
    world: int,
    local_rank: int,
    device: torch.device,
) -> int:
    sizes = _resolve_sizes(args)
    rows: list[dict[str, Any]] = []
    unmeasured: list[dict[str, Any]] = []
    cells = _all_cells(sizes)
    total_cells = len(cells) * len(TRANSPORTS)

    # --- recorded context and the shared-account occupancy probe ---
    local_facts = _rank_facts(rank, local_rank, device)
    facts_bag: list[Any] = [None] * world
    dist.all_gather_object(facts_bag, local_facts)
    rank_facts: list[dict[str, Any]] = [dict(f) for f in facts_bag]
    contended, contention_reason = _occupancy_gate(rank_facts)

    if contended:
        # Publish nothing downstream of a contended device: every cell is
        # reported UNMEASURED with the tenant named, rather than emitting a
        # number measured through someone else's traffic.
        for transport in TRANSPORTS:
            for size, label in cells:
                unmeasured.append({
                    "transport": transport,
                    "bytes": size,
                    "label": label,
                    "reason": "device contention: %s" % contention_reason,
                })
    else:
        aborted = False
        for size, label in cells:
            for transport in TRANSPORTS:
                try:
                    row, fail = _measure_arm(
                        transport, size, label, rank, world, device,
                        args.warmup, args.repeat, args.repeat_budget_s,
                        args.warmup_tol,
                    )
                except RuntimeError as exc:
                    # A late or asynchronous failure (incl. a NCCL error
                    # surfacing at a later sync): retro-invalidate prior
                    # evidence if the group proves unhealthy, then stop.
                    aborted = _record_late(
                        unmeasured, rows, transport, size, label, exc, device, world,
                    )
                    if aborted:
                        break
                else:
                    if row is not None:
                        rows.append(row)
                    if fail is not None:
                        unmeasured.append(fail)
            if aborted:
                break

        if not aborted:
            final_health = _health_check(device, world)
            if final_health is not None:
                _retro_invalidate(
                    rows, unmeasured,
                    "final process-group health check failed: %s" % final_health,
                )

    # Row schema guard: a row that does not disclose its byte basis cannot
    # be published; fold it into unmeasured before anything sees it.
    if rank == 0:
        kept: list[dict[str, Any]] = []
        for r in rows:
            if _row_discloses_basis(r):
                kept.append(r)
            else:
                unmeasured.append({
                    "transport": r.get("transport", "?"),
                    "bytes": r.get("bytes", 0),
                    "label": r.get("label", "?"),
                    "reason": "row failed byte-basis disclosure schema "
                              "(moved_bytes/byte_model/gbps_basis)",
                })
        rows = kept

    # Computed identically on every rank; the DECISION is rank 0's alone.
    control_on_r0: list[Any] = [None]
    verdict_on_r0: list[Any] = [None]
    code_on_r0: list[Any] = [None]
    if rank == 0:
        control, verdict, code = _adjudicate(rows, unmeasured, world, args.control_margin)
        control_on_r0[0] = control
        verdict_on_r0[0] = verdict
        code_on_r0[0] = code

        nccl_ver: Any = None
        try:
            nccl_ver = list(torch.cuda.nccl.version())
        except Exception:
            nccl_ver = None
        try:
            cudnn_ver: Any = torch.backends.cudnn.version()
        except Exception:
            cudnn_ver = None

        payload: dict[str, Any] = {
            "schema": SCHEMA,
            "provenance": {
                "python_executable": sys.executable,
                "python_version": sys.version,
            },
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "nccl": nccl_ver,
            "cudnn": cudnn_ver,
            "flags": {
                "tf32_matmul": bool(torch.backends.cuda.matmul.allow_tf32),
                "tf32_cudnn": bool(torch.backends.cudnn.allow_tf32),
                "cudnn_enabled": bool(torch.backends.cudnn.enabled),
                "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
                "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
            },
            "nccl_environment": _capture_env(),
            "affinity_policy": "recorded, not modified (shared account)",
            "occupancy": {
                "foreign_bytes_limit": FOREIGN_BYTES_LIMIT,
                "contended": contended,
                "ranks": rank_facts,
            },
            "world_size": world,
            "device_name": torch.cuda.get_device_name(device),
            "capability": "%d.%d" % torch.cuda.get_device_capability(device),
            "warmup": args.warmup,
            "warmup_stability": {
                "tol": args.warmup_tol, "cap": WARMUP_CAP,
                "ref_bytes": WARMUP_REF_BYTES,
            },
            "repeat": args.repeat,
            "repeat_adaptive": {
                "budget_s": args.repeat_budget_s,
                "clamp": [REPEAT_MIN, REPEAT_MAX],
            },
            "control_margin": args.control_margin,
            "buffer_ring": RING_BUFFERS,
            "rows": rows,
            "controls": control,
            "unmeasured": unmeasured,
            "verdict": verdict,
            "denominator": {
                "measured_cells": len(rows),
                "total_cells": total_cells,
                "outside": NOT_MEASURED,
                "unmeasured_cells": len(unmeasured),
            },
            "not_measured": NOT_MEASURED,
            "cross_rank_timebase": TIMEBASE_NOTE,
            "aggregation": (
                "every aggregate figure is derived from the SLOWEST rank's "
                "median and is named as such; per-rank timings share no "
                "common clock"
            ),
            "timing_method": (
                "poison+prime before each timed iteration (untimed, drained); "
                "dist.barrier() before each timed iteration; torch.cuda.Event "
                "pairs around the blocking op (blocking collectives wait on "
                "their work under torch 2.13.0, so the end event is "
                "correctly ordered); torch.cuda.synchronize() before reading "
                "elapsed_time; each rank keeps its own samples and ranks "
                "exchange medians after the window; warmups run to a "
                "size-dependent floor and a last-three stability "
                "precondition; repeat scales against a wall-clock budget "
                "clamped to [10, 500]; median, per-rank spread and IQR "
                "reported, never a bare mean"
            ),
            "exit_code_contract": {"CLEAR": 0, "RED": 5, "UNMEASURED": 95, "REFUSE": 96},
        }

        print("=" * 78)
        print("WEIGHT-SYNC TRANSPORT COST -- GB200, world_size=%d" % world)
        print("Timing: CUDA events around the op, barrier-aligned, blocking "
              "collectives; per-rank samples, no common clock; aggregate = "
              "SLOWEST rank's median (named as such)")
        print("Poison gate: destinations poisoned and re-primed before "
              "every timed iteration, verified only after the window "
              "closed, reduced across ranks (MIN).")
        print("=" * 78)
        print("DENOMINATOR: %d of %d (transport,size) cells measured and "
              "admitted (%d transports x (%d sweep sizes + 1 per-call-floor "
              "size))."
              % (len(rows), total_cells, len(TRANSPORTS), len(sizes)))
        # State the rule that is ENFORCED, not a narrower one (#322). Three
        # distinct things make this run abstain and they are not the same
        # thing, so the banner names all three rather than only the first.
        print("ABSTENTION RULE: zero measured cells is UNMEASURED, never "
              "PASS; a control that never fired is UNMEASURED, never PASS; "
              "and any withheld cell narrows the verdict to "
              "CLEAR_WITH_ABSTENTIONS over a named denominator, which "
              "certifies nothing about the cells it withheld.")
        print("OUTSIDE THE DENOMINATOR:")
        for item in NOT_MEASURED:
            print("  - %s" % item)
        if unmeasured:
            print("  - %d attempted cells withheld as UNMEASURED (see "
                  "'unmeasured' in the JSON)" % len(unmeasured))
        print("-" * 78)
        print(_render_table(rows, unmeasured))
        print("-" * 78)
        print("ADJUDICATION (positive control): pageable_slower_than_pinned "
              "fired=%s ok=%s (predeclared margin %.1f%% + non-overlapping "
              "bootstrap 95%% intervals)"
              % (control["fired"], control["ok"], 100.0 * args.control_margin))
        if control["violations"]:
            for v in control["violations"]:
                print("  VIOLATION @ %d B: pinned_median=%.6fs pageable_median=%.6fs "
                      "(observed ratio %.4f, required >= %.4f); pinned CI95 [%.6f, %.6f] "
                      "pageable CI95 [%.6f, %.6f] -- the control failed to fire:"
                      % (
                          v["bytes"], v["pinned_median_s"], v["pageable_median_s"],
                          v["observed_ratio"], v["required_ratio"],
                          v["ci95_pinned_s"][0], v["ci95_pinned_s"][1],
                          v["ci95_pageable_s"][0], v["ci95_pageable_s"][1],
                      ))
                for reason in v["reasons"]:
                    print("      - %s" % reason)
        for o in control["observations"]:
            print("  OBSERVATION @ %d B: pinned_median=%.6fs pageable_median=%.6fs "
                  "(observed ratio %.4f, required >= %.4f) -- NOT a violation: the "
                  "poison gate covered BOTH rows of this pair and has already "
                  "refuted elision directly, so this indirect margin is recorded "
                  "rather than adjudicated (#321):"
                  % (
                      o["bytes"], o["pinned_median_s"], o["pageable_median_s"],
                      o["observed_ratio"], o["required_ratio"],
                  ))
            for reason in o["reasons"]:
                print("      - %s" % reason)
        if control["violations"]:
            pass
        elif control["fired"] and control["observations"]:
            print("  pageable exceeded pinned by the predeclared margin with "
                  "non-overlapping intervals at every pair the margin "
                  "adjudicated; the pairs listed above as OBSERVATION were "
                  "adjudicated by the poison gate instead. The H2D/D2H path is "
                  "really moving bytes.")
        elif control["fired"]:
            print("  pageable exceeded pinned by the predeclared margin with "
                  "non-overlapping intervals at all paired sweep sizes; the "
                  "H2D/D2H path is really moving bytes.")
        else:
            print("  control never fired (no paired sweep sizes); no timing "
                  "in this run is admitted as evidence.")
        print("Per-call floor: see rows labelled 'per_call_floor' (1 MiB); "
              "cadence semantics turn on per-byte vs per-call cost.")
        if control.get("abstentions"):
            print("-" * 78)
            print("DECLARED ABSTENTIONS -- this verdict certifies NOTHING "
                  "about the following %d cell(s):"
                  % len(control["abstentions"]))
            for a in control["abstentions"]:
                print("  - %s @ %s B (%s): %s"
                      % (a["transport"], a["bytes"], a["label"], a["reason"]))
            print("  The elision control adjudicated only these payload "
                  "sizes: %s B. A size whose pinned/pageable pair lost a row "
                  "is absent from that list and was never examined."
                  % ", ".join(
                      str(b) for b in control["elision_adjudicated_sizes"]
                  ))
        print("VERDICT: %s (exit %d)" % (verdict, code))

        try:
            with open(args.out, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, indent=2)
        except OSError as exc:
            sys.stderr.write("warning: could not write %s: %s\n" % (args.out, exc))

    # Only rank 0 decides the exit code; every rank exits with it.
    decision = torch.zeros(1, dtype=torch.int64, device=device)
    if rank == 0:
        decision.fill_(int(code_on_r0[0]))
    dist.broadcast(decision, src=0)
    return int(decision.item())


# --------------------------------------------------------------------------
# Self-test harness (--self-test): exercises the adjudication and control
# layer over synthetic result rows. Uses NO CUDA and NO torch.distributed,
# so it runs in CI on a machine with no cluster. Each arm is explicitly
# labelled MUST_FIRE / MUST_PASS / MUST_BE_UNMEASURED and prints what it
# wanted and what it got.
# --------------------------------------------------------------------------
def _synth_row(
    transport: str,
    size: int,
    label: str,
    median: float,
    half_spread: float,
    n: int,
    gbps_basis: str,
    byte_model: str,
    moved: int,
) -> dict[str, Any]:
    step = (2.0 * half_spread) / max(1, n - 1)
    samples = sorted(median - half_spread + i * step for i in range(n))
    return {
        "transport": transport,
        "bytes": size,
        "label": label,
        "n_samples": n,
        "median_s": float(statistics.median(samples)),
        "samples_s": samples,
        "moved_bytes": moved,
        "byte_model": byte_model,
        "gbps_basis": gbps_basis,
    }


def _synth_pin_pageable(
    sizes: list[int],
    pinned_med: float,
    pageable_med: float,
    spread: float,
    n: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for s in sizes:
        rows.append(_synth_row(
            "full_copy", s, "sweep", pinned_med, spread, n,
            "per_rank_roundtrip", "synthetic pinned round trip", 2 * s,
        ))
        rows.append(_synth_row(
            "full_copy_pageable", s, "sweep", pageable_med, spread, n,
            "per_rank_roundtrip", "synthetic pageable round trip", 2 * s,
        ))
    return rows


def _arm_outcome(
    got: tuple[dict[str, Any], str, int],
    want_verdict: str,
    want_code: int,
    want_control_ok: bool | None,
) -> tuple[bool, str]:
    control, verdict, code = got
    ok = verdict == want_verdict and code == want_code
    if want_control_ok is not None:
        ok = ok and bool(control.get("ok")) == want_control_ok
    return ok, "verdict=%s exit=%d control.ok=%s violations=%d" % (
        verdict, code, control.get("ok"), len(control.get("violations", [])),
    )


def _self_test() -> int:
    margin = CONTROL_MARGIN_DEFAULT
    sweep = [64 * MIB, 256 * MIB]

    def arm_control_passes() -> tuple[str, tuple[bool, str]]:
        rows = _synth_pin_pageable(sweep, 1.00, 1.30, 0.004, 60)
        # Extra per-call-floor rows with EQUAL medians must not be paired:
        # the control only ever reads rows labelled "sweep".
        floor_rows = _synth_pin_pageable(
            [PER_CALL_FLOOR_BYTES], 1.0, 1.0, 0.004, 60,
        )
        for r in floor_rows:
            r["label"] = "per_call_floor"
        rows += floor_rows
        got = _adjudicate(rows, [], 4, margin)
        ok, desc = _arm_outcome(got, "CLEAR", EXIT_CLEAR, True)
        return "want: verdict=CLEAR exit=0 control.ok=True; got: %s" % desc, (ok, desc)

    def arm_control_catches_elided() -> tuple[str, tuple[bool, str]]:
        rows = _synth_pin_pageable(sweep, 1.00, 1.001, 0.004, 60)
        got = _adjudicate(rows, [], 4, margin)
        ok, desc = _arm_outcome(got, "RED", EXIT_RED, False)
        control = got[0]
        named = bool(control["violations"]) and all(
            v["pinned_median_s"] > 0.0 and v["pageable_median_s"] > 0.0 and v["reasons"]
            for v in control["violations"]
        )
        return (
            "want: verdict=RED exit=5 control.ok=False with the numbers "
            "named in each violation; got: %s (numbers named: %s)" % (desc, named)
        ), (ok and named, desc)

    def arm_control_rejects_thin_margin() -> tuple[str, tuple[bool, str]]:
        rows = _synth_pin_pageable(sweep, 1.00, 1.02, 0.004, 80)
        got = _adjudicate(rows, [], 4, margin)
        ok, desc = _arm_outcome(got, "RED", EXIT_RED, False)
        return "want: verdict=RED exit=5 (2%% gap < 5%% margin); got: %s" % desc, (ok, desc)

    def arm_poison_gate_downgrades_thin_margin() -> tuple[str, tuple[bool, str]]:
        # The twin of arm 3, differing in exactly one variable: whether the
        # DIRECT instrument covered the pair. Same medians, same margin, same
        # sizes -- only the poison gate is added. If the verdict flips, the
        # poison gate is what flipped it, and the arm has isolated its own
        # cause rather than merely passing (#321).
        def _thin_rows(poisoned: bool) -> list[dict[str, Any]]:
            rows = _synth_pin_pageable(sweep, 1.00, 1.02, 0.004, 80)
            if poisoned:
                for r in rows:
                    r["poison_gate"] = (
                        "destination poisoned and re-primed per iteration; "
                        "verified only after the timed window closed"
                    )
            return rows

        got = _adjudicate(_thin_rows(True), [], 4, margin)
        ok, desc = _arm_outcome(got, "CLEAR", EXIT_CLEAR, True)
        control = got[0]
        # The downgrade must be RECORDED, not silent: one observation per pair,
        # each naming the poison gate as its adjudicator, and no violations.
        recorded = (
            len(control["observations"]) == len(sweep)
            and not control["violations"]
            and all(
                o["poison_gate_covered"]
                and o["elision_adjudicated_by"] == "poison_gate"
                and o["reasons"]
                for o in control["observations"]
            )
        )
        # And the control variable: strip the poison gate, keep everything
        # else, and the same numbers must still go RED.
        twin = _adjudicate(_thin_rows(False), [], 4, margin)
        twin_reds, _ = _arm_outcome(twin, "RED", EXIT_RED, False)
        twin_control = twin[0]
        isolated = twin_reds and len(twin_control["violations"]) == len(sweep) and all(
            v["elision_adjudicated_by"] == "pinned_pageable_margin"
            for v in twin_control["violations"]
        )
        return (
            "want: the SAME sub-margin gap is CLEAR with an observation per "
            "pair when the poison gate covered it, and RED without it; got: "
            "%s (recorded: %s, twin still RED: %s)" % (desc, recorded, isolated)
        ), (ok and recorded and isolated, desc)

    def arm_partial_sweep_abstains_by_name() -> tuple[str, tuple[bool, str]]:
        # A withheld cell must NARROW the verdict, not discard the cells that
        # measured -- and must never launder a real RED into a pass (#322).
        base = _synth_pin_pageable(sweep, 1.00, 1.30, 0.004, 60)
        # The realistic shape, and the one the safety claim is about: the
        # PINNED twin at 1 GiB was withheld, so that pair is incomplete and
        # the elision control never examined that size at all.
        lone = _synth_row(
            "full_copy_pageable", GIB, "sweep", 1.30, 0.004, 60,
            "per_rank_roundtrip", "synthetic pageable round trip", 2 * GIB,
        )
        withheld = [{
            "transport": "full_copy",
            "bytes": GIB,
            "label": "sweep",
            "reason": "warmup never stabilised",
        }]
        got = _adjudicate(base + [lone], withheld, 4, margin)
        ok, desc = _arm_outcome(got, "CLEAR_WITH_ABSTENTIONS", EXIT_CLEAR, True)
        control = got[0]
        named = (
            len(control.get("abstentions", [])) == 1
            and control["abstentions"][0]["bytes"] == GIB
            and bool(control["abstentions"][0]["reason"])
        )
        # Load-bearing: the size whose pair lost a row must be ABSENT from
        # the sizes the elision control adjudicated, or the verdict would
        # read as covering a payload size nothing examined.
        scoped = (
            control.get("elision_adjudicated_sizes") == sorted(sweep)
            and GIB not in control.get("elision_adjudicated_sizes", [])
        )
        # Twin A: drop the abstention, same rows, and it is a plain CLEAR --
        # so the abstention is what narrowed it, not the numbers.
        twin_clear, _ = _arm_outcome(
            _adjudicate(base, [], 4, margin), "CLEAR", EXIT_CLEAR, True,
        )
        # Twin B: an abstention must not outrank a violation.
        elided = _synth_pin_pageable(sweep, 1.00, 1.001, 0.004, 60)
        twin_red, _ = _arm_outcome(
            _adjudicate(elided, withheld, 4, margin), "RED", EXIT_RED, False,
        )
        return (
            "want: a withheld cell yields CLEAR_WITH_ABSTENTIONS naming that "
            "cell, with the incomplete pair's size ABSENT from the "
            "adjudicated set; the same rows without it are a plain CLEAR; "
            "and an abstention never launders a RED; got: %s (named: %s, "
            "scoped: %s, plain-CLEAR twin: %s, RED twin: %s)"
            % (desc, named, scoped, twin_clear, twin_red)
        ), (ok and named and scoped and twin_clear and twin_red, desc)

    def arm_zero_units() -> tuple[str, tuple[bool, str]]:
        um = [
            {"transport": t, "bytes": sweep[0], "label": "sweep", "reason": "synthetic"}
            for t in TRANSPORTS
        ]
        got = _adjudicate([], um, 4, margin)
        ok, desc = _arm_outcome(got, "UNMEASURED", EXIT_UNMEASURED, False)
        return ("want: verdict=UNMEASURED exit=95 -- zero measured units is "
                "UNMEASURED, never PASS; got: %s" % desc), (ok, desc)

    def arm_control_cannot_fire() -> tuple[str, tuple[bool, str]]:
        rows = [r for r in _synth_pin_pageable(sweep, 1.0, 1.3, 0.004, 60)
                if r["transport"] == "full_copy"]
        um = [
            {"transport": "full_copy_pageable", "bytes": s, "label": "sweep",
             "reason": "synthetic allocation failure"}
            for s in sweep
        ]
        got = _adjudicate(rows, um, 4, margin)
        ok, desc = _arm_outcome(got, "UNMEASURED", EXIT_UNMEASURED, False)
        return ("want: verdict=UNMEASURED exit=95 (control could not fire); "
                "got: %s" % desc), (ok, desc)

    def arm_world_lt_2() -> tuple[str, tuple[bool, str]]:
        rows = _synth_pin_pageable(sweep, 1.00, 1.30, 0.004, 60)
        got = _adjudicate(rows, [], 1, margin)
        ok, desc = _arm_outcome(got, "UNMEASURED", EXIT_UNMEASURED, None)
        return ("want: verdict=UNMEASURED exit=95 for world_size=1 (degrade, "
                "not RED, not crash); got: %s" % desc), (ok, desc)

    def arm_byte_basis_disclosed() -> tuple[str, tuple[bool, str]]:
        good = _synth_pin_pageable(sweep, 1.0, 1.3, 0.004, 20)
        bases = {r["gbps_basis"] for r in good}
        disclosed = all(_row_discloses_basis(r) for r in good)
        bad = _synth_row("collective", sweep[0], "sweep", 1.0, 0.01, 20,
                         "aggregate_broadcast", "synthetic", 3 * sweep[0])
        del bad["gbps_basis"]
        rejected = not _row_discloses_basis(bad)
        ok = disclosed and rejected and "per_rank_roundtrip" in bases
        desc = "all measured rows disclose basis: %s; doctored row rejected: %s" % (
            disclosed, rejected,
        )
        return ("want: every row carries moved_bytes + byte_model + gbps_basis, "
                "and a row missing the basis is rejected; got: %s" % desc), (ok, desc)

    def arm_retro_invalidation() -> tuple[str, tuple[bool, str]]:
        rows = _synth_pin_pageable(sweep, 1.0, 1.3, 0.004, 20)
        um: list[dict[str, Any]] = [{
            "transport": "collective", "bytes": sweep[1], "label": "sweep",
            "reason": "synthetic late NCCL error",
        }]
        n_um_before = len(um)
        _retro_invalidate(rows, um, "synthetic process-group health failure")
        demoted = um[n_um_before:]
        ok = (
            len(rows) == 0
            and len(demoted) == 2 * len(sweep)
            and all(d.get("retro_invalidated") for d in demoted)
        )
        desc = ("rows after invalidation: %d; newly unmeasured: %d; all "
                "marked retro_invalidated: %s"
                % (len(rows), len(demoted),
                   all(d.get("retro_invalidated") for d in demoted)))
        return ("want: all prior rows demoted to UNMEASURED with "
                "retro_invalidated=True when a late error proves the group "
                "unhealthy; got: %s" % desc), (ok, desc)

    arms: list[tuple[str, str, Callable[[], tuple[str, tuple[bool, str]]]]] = [
        ("MUST_PASS", "control passes on a genuine pinned/pageable gap "
         "(and per-call-floor rows are never paired)", arm_control_passes),
        ("MUST_FIRE", "control catches an elided copy (pageable ~= pinned) "
         "and names the numbers", arm_control_catches_elided),
        ("MUST_FIRE", "control rejects a sub-margin gap (2% < 5%)",
         arm_control_rejects_thin_margin),
        ("MUST_PASS", "a sub-margin gap the POISON GATE covered is an "
         "observation, not a violation -- and the twin without the gate is "
         "still RED", arm_poison_gate_downgrades_thin_margin),
        ("MUST_PASS", "a WITHHELD cell narrows the verdict to "
         "CLEAR_WITH_ABSTENTIONS over a named denominator -- it does not "
         "discard the cells that measured, and it never launders a RED",
         arm_partial_sweep_abstains_by_name),
        ("MUST_BE_UNMEASURED", "zero measured units is UNMEASURED, never PASS",
         arm_zero_units),
        ("MUST_BE_UNMEASURED", "control that cannot fire yields UNMEASURED",
         arm_control_cannot_fire),
        ("MUST_BE_UNMEASURED", "world_size < 2 degrades to UNMEASURED",
         arm_world_lt_2),
        ("MUST_PASS", "every emitted row discloses its byte basis",
         arm_byte_basis_disclosed),
        ("MUST_FIRE", "a late async failure retro-invalidates earlier CLEAR rows",
         arm_retro_invalidation),
    ]

    print("SELF-TEST: adjudication and control layer over synthetic rows; "
          "no CUDA, no torch.distributed")
    print("interpreter: %s" % sys.executable)
    print("version: %s" % sys.version.replace("\n", " "))
    print("SELF-TEST DENOMINATOR: %d of %d registered arms exercised; an arm "
          "outside this list makes no claim." % (len(arms), len(arms)))
    print("-" * 78)

    failed = 0
    for i, (expect, title, fn) in enumerate(arms, 1):
        want, (ok, _) = fn()
        status = "OK" if ok else "MISMATCH"
        if not ok:
            failed += 1
        print("ARM %d/%d [%s] %s" % (i, len(arms), expect, title))
        print("  %s -> %s" % (want, status))
    print("-" * 78)
    if failed:
        print("SELF-TEST RESULT: %d of %d arms mismatched their label "
              "(exit %d) -- the gate's adjudication layer is not behaving "
              "as specified" % (failed, len(arms), EXIT_RED))
        return EXIT_RED
    print("SELF-TEST RESULT: %d of %d arms behaved as labelled (exit %d)"
          % (len(arms), len(arms), EXIT_CLEAR))
    return EXIT_CLEAR


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))