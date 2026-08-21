"""Checkpoint gates: the save-path checks that would have caught the 87.5%-wrong MoE run.

Why this module exists
----------------------
The audit's defining checkpoint incident: a MoE trainer saved expert tensors under
*rank-local* names (``...linear_fc1.weight0`` ... ``weight15``), so 128 experts
collapsed to 16 distinct storages, each written by 8 ranks. On disk it was
unmistakable — 16 small shards per layer where the healthy checkpoint has one fused
``...linear_fc1.weight`` of shape ``(128, 1408, 2816)``; 5.71 GB where 45.70 GB was
correct. It passed ``rc=0``, resume, healthy loss curves, tensor counts and dtype
checks for two full training runs. The detection tool that was supposed to catch it
reported ``all_identity: true`` on the corrupt artifact because the expert set was
empty and ``all([]) is True``.

So every gate here (a) counts what it examined against a declared denominator (an
unqualified count is not a fact), (b) treats an absent expert set on a declared-MoE
model as VACUOUS, never as "all identity", and (c) ships fixtures proving it fires —
including the empty-expert-set fixture, which exists solely to prevent the
detector-itself-silently-passing bug from recurring in this codebase.

Context protocol
----------------
Gates consume :class:`CheckpointGateContext`. At runtime it is built from an on-disk
checkpoint via :meth:`CheckpointGateContext.from_path`, which lazily imports
``foundationscale.checkpoint`` (a torch-free metadata summary plus the run manifest
that declares what the checkpoint *should* contain). Controls build synthetic
contexts directly in ``fixtures.py``, so ``verify_controls`` runs with no torch and
no large I/O. If the real module's API drifts, the adapter in ``from_path`` is the
single place to fix — that is why it exists as a function instead of inline code
scattered across four gates.
"""

from __future__ import annotations

import math
import os
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

from .core import (
    Control,
    ControlKind,
    Coverage,
    Gate,
    GateResult,
    Lifecycle,
    Verdict,
    register,
)

__all__ = [
    "TensorMeta",
    "CheckpointGateContext",
    "ExpertDistinctnessGate",
    "ExpertByteVolumeGate",
    "SaveCompletenessGate",
    "FirstSaveGate",
]

_DTYPE_BYTES: dict[str, int] = {
    "bfloat16": 2,
    "float16": 2,
    "float32": 4,
    "float": 4,
    "float64": 8,
    "int8": 1,
    "uint8": 1,
    "int16": 2,
    "int32": 4,
    "int64": 8,
    "bool": 1,
}

# Healthy:  "...mlp.experts.experts.experts.linear_fc1.weight"      (fused, no index)
# Corrupt:  "...mlp.experts.experts.linear_fc1.weight0" .. "weight15" (local names)
_EXPERT_WEIGHT_RE = re.compile(r".*experts.*linear_fc[12]\.weight\d*$")
_SHARD_SUFFIX_RE = re.compile(r"^(?P<stem>.*linear_fc[12]\.weight)(?P<idx>\d+)$")


@dataclass(frozen=True)
class TensorMeta:
    """One entry of checkpoint tensor metadata.

    ``storage_id`` identifies the underlying byte storage (DCP: file+offset). It is
    what makes *aliasing* visible without reading a tensor: two FQNs that name the
    same storage are the same tensor wearing two names — the "aliased 8 ways"
    signature. ``kind`` separates real tensors from metadata byte blobs: a 26B
    checkpoint's metadata carries ~8,970 keys of which ~8,042 are ``_extra_state``
    entries, and counting those as tensors is how a completeness check passes
    trivially on a gutted checkpoint.
    """

    fqn: str
    shape: tuple[int, ...]
    dtype: str
    storage_id: str | None = None
    kind: str = "tensor"  # "tensor" | "extra_state"

    @property
    def nbytes(self) -> int:
        return math.prod(self.shape) * _DTYPE_BYTES.get(self.dtype, 4)


@dataclass(frozen=True)
class CheckpointGateContext:
    """Everything the checkpoint gates need, with no torch dependency.

    ``declared_*`` fields come from the run manifest — the checkpoint's own
    statement of what it was supposed to contain. Comparing what *is* there against
    only what *is* there is the vacuity trap; comparing it against what the run
    *declared* is the check.
    """

    tensors: tuple[TensorMeta, ...]
    declared_fqns: tuple[str, ...] | None
    num_experts: int | None  # declared experts per MoE layer; 0/None => dense model
    num_moe_layers: int | None
    expected_expert_bytes: int | None
    origin: str

    @classmethod
    def from_path(cls, path: str | os.PathLike[str]) -> CheckpointGateContext:
        """Build a context from an on-disk checkpoint.

        Lazily imports ``foundationscale.checkpoint`` so this package imports
        without torch. Assumed API (kept deliberately narrow): ``read_metadata``
        returns an object with ``.tensors: Mapping[str, TensorStorageMeta]`` where
        each meta has ``.shape``/``.dtype`` and optionally ``.storage_id`` /
        ``.is_extra_state``; ``load_manifest`` returns the run manifest (or
        ``None``) optionally carrying ``declared_fqns``, ``num_experts``,
        ``num_moe_layers`` and ``expected_expert_bytes``.
        """
        from foundationscale import checkpoint as fsckpt  # lazy: torch-backed

        meta = fsckpt.read_metadata(os.fspath(path))
        manifest = fsckpt.load_manifest(os.fspath(path))
        tensors = tuple(
            TensorMeta(
                fqn=fqn,
                shape=tuple(tm.shape),
                dtype=str(tm.dtype).removeprefix("torch."),
                storage_id=getattr(tm, "storage_id", None),
                kind=(
                    "extra_state"
                    if ("_extra_state" in fqn or getattr(tm, "is_extra_state", False))
                    else "tensor"
                ),
            )
            for fqn, tm in meta.tensors.items()
        )
        return cls(
            tensors=tensors,
            declared_fqns=(tuple(getattr(manifest, "declared_fqns", ())) if manifest else None),
            num_experts=getattr(manifest, "num_experts", None) if manifest else None,
            num_moe_layers=(getattr(manifest, "num_moe_layers", None) if manifest else None),
            expected_expert_bytes=(
                getattr(manifest, "expected_expert_bytes", None) if manifest else None
            ),
            origin=os.fspath(path),
        )


def _coerce(ctx: Any) -> CheckpointGateContext:
    """Accept a context, a checkpoint path, or any object with a ``.tensors``."""
    if isinstance(ctx, CheckpointGateContext):
        return ctx
    if isinstance(ctx, (str, Path)):
        return CheckpointGateContext.from_path(ctx)
    if hasattr(ctx, "tensors"):
        return CheckpointGateContext(
            tensors=tuple(ctx.tensors),
            declared_fqns=getattr(ctx, "declared_fqns", None),
            num_experts=getattr(ctx, "num_experts", None),
            num_moe_layers=getattr(ctx, "num_moe_layers", None),
            expected_expert_bytes=getattr(ctx, "expected_expert_bytes", None),
            origin=getattr(ctx, "origin", repr(ctx)),
        )
    raise TypeError(
        f"checkpoint gates need a CheckpointGateContext or path, got {type(ctx).__name__}"
    )


def _is_real_tensor(t: TensorMeta) -> bool:
    """True for actual parameter/buffer tensors — never for metadata byte blobs."""
    return t.kind == "tensor" and "_extra_state" not in t.fqn


def _expert_weights(ctx: CheckpointGateContext) -> list[TensorMeta]:
    return [t for t in ctx.tensors if _is_real_tensor(t) and _EXPERT_WEIGHT_RE.match(t.fqn)]


def _split_layout(
    experts: list[TensorMeta],
) -> tuple[list[TensorMeta], dict[str, list[TensorMeta]]]:
    """Split expert weights into fused tensors and per-stem shard groups."""
    fused: list[TensorMeta] = []
    shards: dict[str, list[TensorMeta]] = defaultdict(list)
    for t in experts:
        m = _SHARD_SUFFIX_RE.match(t.fqn)
        if m:
            shards[m.group("stem")].append(t)
        else:
            fused.append(t)
    return fused, shards


def _declared_tensor_count(ctx: CheckpointGateContext, *, sharded: bool) -> int | None:
    """How many expert-weight tensors the checkpoint should hold, if knowable."""
    if ctx.num_moe_layers is None:
        return None
    weights_per_layer = 2  # linear_fc1 + linear_fc2
    if sharded:
        if not ctx.num_experts:
            return None
        return ctx.num_experts * ctx.num_moe_layers * weights_per_layer
    return ctx.num_moe_layers * weights_per_layer


@register
class ExpertDistinctnessGate(Gate):
    """Catches 128 experts collapsed to 16 by a local-name save.

    Two signatures, both present in the real incident and checked independently:

    1. *Count*: a sharded layout stores one tensor per expert. 16 on disk against 128
       declared is not "checkpoint format variation"; it is 87.5% of the experts
       missing, overwritten in place as each rank wrote its local ``weight0..15``.
    2. *Aliasing*: distinct FQNs sharing one ``storage_id`` are one tensor. If the
       count ever looks right again (a more subtle version of the same bug), this is
       the check that still fires.

    The empty-expert-set control is load-bearing: on a declared-MoE model, finding
    zero expert tensors must be VACUOUS, because "no mismatches found" is what
    ``all([])`` reported on the corrupt artifact for months.
    """

    id: ClassVar[str] = "checkpoint.expert_distinctness"
    description: ClassVar[str] = (
        "Expert tensors exist, are present at the declared count, and occupy "
        "distinct storage — the 128-experts-aliased-to-16 incident"
    )
    events: ClassVar[tuple[Lifecycle, ...]] = (Lifecycle.FIRST_SAVE, Lifecycle.SAVE)

    def check(self, ctx: Any) -> GateResult:
        c = _coerce(ctx)
        experts = _expert_weights(c)

        if not experts:
            if not c.num_experts:
                return self.skip("context declares no experts and none are present (dense model)")
            # A MoE model whose checkpoint has zero expert tensors has not "passed an
            # identity check on every expert". ok() downgrades this to VACUOUS, which
            # is the exact difference between this gate and the audit tool it replaces.
            return self.ok(
                f"model declares {c.num_experts} experts but the checkpoint contains "
                f"no expert tensors",
                Coverage.none("expert tensors"),
                evidence={"origin": c.origin},
            )

        fused, shards = _split_layout(experts)
        problems: list[str] = []
        offenders: list[str] = []

        for t in fused:
            if c.num_experts and (not t.shape or t.shape[0] != c.num_experts):
                problems.append(
                    f"{t.fqn}: fused leading dim "
                    f"{t.shape[0] if t.shape else '?'} != declared experts {c.num_experts}"
                )
                offenders.append(t.fqn)

        for stem, members in sorted(shards.items()):
            if c.num_experts and len(members) != c.num_experts:
                problems.append(
                    f"{stem}<i>: {len(members)} expert shards on disk, config declares "
                    f"{c.num_experts} — the local-name save signature (16 of 128)"
                )
                offenders.extend(t.fqn for t in members[:4])
            # storage_id unknown -> fall back to the FQN, which is unique by
            # construction; aliasing is then simply undetectable from metadata, and
            # the byte-volume gate remains the coarse net.
            storages = {t.storage_id if t.storage_id is not None else t.fqn for t in members}
            if len(storages) < len(members):
                problems.append(
                    f"{stem}<i>: {len(members)} expert FQNs share {len(storages)} "
                    f"distinct storages — experts are aliased to the same bytes"
                )
                offenders.extend(t.fqn for t in members[:4])

        coverage = Coverage(
            checked=len(experts),
            unit="expert tensors",
            expected=_declared_tensor_count(c, sharded=bool(shards)),
        )
        if problems:
            return self.fail(
                problems[0] + (f" (+{len(problems) - 1} more)" if len(problems) > 1 else ""),
                coverage,
                evidence={
                    "problems": problems[:16],
                    "offenders": offenders[:8],
                    "origin": c.origin,
                },
            )
        layout = "fused" if fused and not shards else "sharded"
        # Claim only what this branch actually checked: per-stem storage aliasing is
        # verified only when shards exist, and a declared count exists only when the
        # manifest supplied one.
        if c.num_experts and shards:
            detail = (
                f"expert shard counts match the declared {c.num_experts} experts and "
                f"every shard group occupies distinct storage ({layout})"
            )
        elif c.num_experts:
            detail = (
                f"expert weights match the declared {c.num_experts}-expert shape "
                f"({layout}; fused tensors carry no per-expert storages to compare)"
            )
        elif shards:
            detail = (
                "expert shard storages are distinct within every group, but no expert "
                "count was declared — presence at the declared count could not be "
                f"verified ({layout})"
            )
        else:
            detail = (
                "expert tensors are present, but the context declares no expert count "
                "and the layout is fused, so neither count nor storage distinctness "
                f"was verifiable here ({layout})"
            )
        return self.ok(detail, coverage)

    def controls(self) -> list[Control]:
        from . import fixtures as fx

        return [
            Control(
                "aliased-16-of-128",
                ControlKind.MUST_FIRE,
                fx.aliased_local_names_ctx,
                note="local-name expert save: 16 shards of 128 declared, 8-way aliased storage",
            ),
            Control(
                "empty-expert-set",
                ControlKind.MUST_FIRE,
                fx.empty_expert_set_ctx,
                note="MoE declared, zero expert tensors present — the all([]) is True trap",
            ),
            Control(
                "right-count-but-aliased",
                ControlKind.MUST_FIRE,
                fx.right_count_aliased_storage_ctx,
                note="shard count matches but FQNs share storages (count check alone is blind)",
            ),
            Control(
                "healthy-fused",
                ControlKind.MUST_PASS,
                fx.healthy_fused_moe_ctx,
                note="one fused (experts, in, out) tensor per weight per layer",
            ),
        ]


@register
class ExpertByteVolumeGate(Gate):
    """The cheap coarse net: total expert bytes vs. declared expert bytes.

    The real bug's signature was 5.71 GB on disk where 45.70 GB was correct — an
    exact 1/8 ratio, catchable in milliseconds from DCP metadata alone, without
    reading a single tensor. This gate exists so that even if every semantic check
    were bypassed, the bytes themselves disagree. A byte deficit of >=1% fails;
    overage is not flagged here (padding/optimizer states make it ambiguous) — that
    is distinctness' job.
    """

    id: ClassVar[str] = "checkpoint.expert_bytes"
    description: ClassVar[str] = (
        "Total expert tensor bytes match the declared volume (5.71 GB vs 45.70 GB "
        "was visible from metadata alone)"
    )
    events: ClassVar[tuple[Lifecycle, ...]] = (Lifecycle.FIRST_SAVE, Lifecycle.SAVE)

    _DEFICIT_PER_MILLE: ClassVar[int] = 10  # fail if actual < expected * 99%

    def check(self, ctx: Any) -> GateResult:
        c = _coerce(ctx)
        experts = _expert_weights(c)

        if not experts:
            if not c.num_experts:
                return self.skip("context declares no experts and none are present (dense model)")
            return self.ok(
                f"model declares {c.num_experts} experts but checkpoint has no "
                f"expert tensors — there is nothing to sum",
                Coverage.none("expert tensors"),
            )

        if c.expected_expert_bytes is None:
            return self.skip(
                "run manifest does not declare expected expert byte volume; without "
                "the denominator a byte count is an unqualified count, not a fact"
            )

        actual = sum(t.nbytes for t in experts)
        _, shards = _split_layout(experts)
        coverage = Coverage(
            checked=len(experts),
            unit="expert tensors",
            expected=_declared_tensor_count(c, sharded=bool(shards)),
        )
        if actual * 1000 < c.expected_expert_bytes * (1000 - self._DEFICIT_PER_MILLE):
            return self.fail(
                f"expert bytes {actual:,} of declared {c.expected_expert_bytes:,} "
                f"(ratio {actual / c.expected_expert_bytes:.3f}; the incident ratio "
                f"was 0.125 = 5.71/45.70 GB)",
                coverage,
                evidence={
                    "actual_bytes": actual,
                    "expected_bytes": c.expected_expert_bytes,
                    "ratio": round(actual / c.expected_expert_bytes, 4),
                    "origin": c.origin,
                },
            )
        return self.ok(
            f"expert byte volume {actual:,} matches declared {c.expected_expert_bytes:,}",
            coverage,
        )

    def controls(self) -> list[Control]:
        from . import fixtures as fx

        return [
            Control(
                "eighth-the-bytes",
                ControlKind.MUST_FIRE,
                fx.aliased_local_names_ctx,
                note="8-way expert aliasing shrinks expert bytes to exactly 1/8",
            ),
            Control(
                "no-experts-at-all",
                ControlKind.MUST_FIRE,
                fx.empty_expert_set_ctx,
                note="zero experts to sum must not read as 'bytes match'",
            ),
            Control(
                "healthy-fused",
                ControlKind.MUST_PASS,
                fx.healthy_fused_moe_ctx,
                note="byte volume matches the manifest exactly",
            ),
        ]


@register
class SaveCompletenessGate(Gate):
    """Every declared tensor FQN is present in the written checkpoint.

    The counting rule is the substance: a 26B checkpoint's metadata carries ~8,970
    keys of which ~8,042 are ``_extra_state`` byte blobs. A completeness check that
    counts *keys* passes trivially while real tensors are missing. This gate
    intersects declared FQNs with present FQNs after filtering both sides to real
    tensors, so the denominator is the thing that matters.
    """

    id: ClassVar[str] = "checkpoint.save_complete"
    description: ClassVar[str] = (
        "Every tensor FQN the run declared is present (counting real tensors only — "
        "a 26B checkpoint has 8,970 metadata keys but only ~928 are tensors)"
    )
    events: ClassVar[tuple[Lifecycle, ...]] = (Lifecycle.FIRST_SAVE, Lifecycle.SAVE)

    def check(self, ctx: Any) -> GateResult:
        c = _coerce(ctx)
        present = {t.fqn for t in c.tensors if _is_real_tensor(t)}

        if c.declared_fqns is None:
            # SKIP means "not applicable", but completeness is applicable to every
            # checkpoint; what is missing is the denominator. That is missing
            # evidence and must block, so route through ok() with zero coverage —
            # the framework's enforced downgrade makes this VACUOUS, never a pass.
            return self.ok(
                "no run manifest found beside the checkpoint, so completeness could "
                "not be established: there is no declared tensor set to compare the "
                "present set against — 'what is there matches what is there' is "
                "not a check",
                Coverage.none("declared tensors"),
                evidence={"origin": c.origin},
            )
        declared = sorted(f for f in c.declared_fqns if "_extra_state" not in f)
        missing = [f for f in declared if f not in present]
        coverage = Coverage(
            checked=len(declared) - len(missing),
            unit="tensors",
            expected=len(declared),
        )
        if missing:
            return self.fail(
                f"{len(missing)} of {len(declared)} declared tensors absent — "
                f"checkpoint is missing a shard",
                coverage,
                evidence={"missing": missing[:16], "origin": c.origin},
            )
        return self.ok(
            f"all {len(declared)} declared tensors present (excluding "
            f"{len(c.declared_fqns) - len(declared)} _extra_state metadata blobs)",
            coverage,
        )

    def controls(self) -> list[Control]:
        from . import fixtures as fx

        return [
            Control(
                "missing-shard",
                ControlKind.MUST_FIRE,
                fx.missing_shard_ctx,
                note="shard_2 never written: 2 of 6 declared tensors absent",
            ),
            Control(
                "bloated-metadata",
                ControlKind.MUST_PASS,
                fx.bloated_extra_state_ctx,
                note="64 real tensors + 512 _extra_state blobs, mirroring 928/8,970 — "
                "proves the blob filter does not inflate coverage",
            ),
            Control(
                "healthy-fused",
                ControlKind.MUST_PASS,
                fx.healthy_fused_moe_ctx,
                note="declared == present",
            ),
        ]


@register
class FirstSaveGate(Gate):
    """Runs the checkpoint gates as one composite at the *first* save.

    The cheapest moment to catch a save defect is before 2,400 steps are written on
    top of it — in the incident, the corrupt format persisted for two full runs
    because nothing looked at the first checkpoint with suspicion. In the audited
    estate the equivalent check lived as a copy-pasted heredoc in one launcher and
    was simply absent from the other; making this a registered gate means its
    absence shows up as ``missing`` in the FIRST_SAVE report instead of as silence.
    """

    id: ClassVar[str] = "checkpoint.first_save"
    description: ClassVar[str] = (
        "Composite: distinctness + byte volume + completeness at the first "
        "checkpoint of a run, the cheapest place a save defect can be caught"
    )
    events: ClassVar[tuple[Lifecycle, ...]] = (Lifecycle.FIRST_SAVE,)
    _subgates: ClassVar[tuple[type[Gate], ...]] = (
        ExpertDistinctnessGate,
        ExpertByteVolumeGate,
        SaveCompletenessGate,
    )

    def check(self, ctx: Any) -> GateResult:
        sub = tuple(cls_().run(ctx) for cls_ in self._subgates)
        passed = [r for r in sub if r.verdict is Verdict.PASS]
        abstained = [r for r in sub if r.verdict is Verdict.SKIP]
        blocking = [r for r in sub if r.blocking]
        # Coverage counts verified properties, not invoked sub-gates: a SKIP ran but
        # established nothing, so it cannot count as "checked". With expected pinned
        # to the full sweep, ok() downgrades any partial sweep to UNDERCOVERED (and
        # an abstention-only sweep to VACUOUS) instead of letting PASS stand over it.
        coverage = Coverage(len(passed), "sub-gates", expected=len(self._subgates))
        if blocking:
            return self.fail(
                "first save is defective: "
                + "; ".join(f"{r.gate_id}={r.verdict.value}" for r in blocking),
                coverage,
                evidence={r.gate_id: r.to_dict() for r in blocking},
            )
        if abstained:
            verified_msg = ", ".join(r.gate_id for r in passed)
            missing_msg = "; ".join(f"{r.gate_id} skipped: {r.detail}" for r in abstained)
            return self.ok(
                f"verified {len(passed)}/{len(self._subgates)} first-save properties "
                f"({verified_msg}); not established: {missing_msg}",
                coverage,
                evidence={r.gate_id: r.verdict.value for r in sub},
            )
        return self.ok(
            "verified at first save: " + ", ".join(r.gate_id for r in passed),
            coverage,
            evidence={r.gate_id: r.verdict.value for r in sub},
        )

    def controls(self) -> list[Control]:
        from . import fixtures as fx

        return [
            Control(
                "aliased-first-save",
                ControlKind.MUST_FIRE,
                fx.aliased_local_names_ctx,
                note="the composite must block the incident artifact at save #1",
            ),
            Control(
                "empty-expert-first-save",
                ControlKind.MUST_FIRE,
                fx.empty_expert_set_ctx,
                note="a silent composite reproduce the all([]) detector one level up",
            ),
            Control(
                "healthy-first-save",
                ControlKind.MUST_PASS,
                fx.healthy_fused_moe_ctx,
                note="a correct first save must not be blocked",
            ),
        ]
