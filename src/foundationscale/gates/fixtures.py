"""Deterministic synthetic expert sets for gate controls.

Why this module exists
----------------------
Every gate must ship with controls — deliberately broken inputs the gate is required
to block — and those fixtures have to be *boring*: fully deterministic, tiny, and
buildable without torch, safetensors, numpy or real checkpoints. A control that is
even slightly non-deterministic will eventually flake in CI, and a flaky control
teaches the team to ignore the controls job, which is exactly how detector-rot
starts in the audited estate.

These builders synthesise the four shapes of MoE expert data that matter to the
reference gate and, in practice, to most future checkpoint gates:

* :func:`make_healthy_experts` — the known-good input (MUST_PASS controls; without
  one, a gate that blocks on everything rots undetected until someone disables it).
* :func:`make_aliased_experts` — the central audit incident: 128 experts saved as
  16 experts replicated 8 times. Every shape, count and dtype is correct; only the
  *content* repeats. A gate that compares names, shapes or counts cannot see this.
* :func:`make_local_name_experts` — the on-disk signature of the same incident:
  keys spelled ``...linear_fc1.weight0`` .. ``...weight15``, i.e. a trailing LOCAL
  index where a global expert index should live. Diagnosable from names alone,
  before a single byte of tensor content is read.
* :func:`make_empty_experts` — the verification-tool incident: an artifact with no
  expert tensors at all, on which ``all([])`` is ``True``. Any control suite that
  purports to guard against silent success must contain this case.

"Tensors" here are plain ``bytes`` derived from a fixed seed via SHA-256, so the
same call always produces the same bytes on every platform and Python process.
There is deliberately no ``random`` module use anywhere in this file.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

from .checkpoint_gates import CheckpointGateContext, TensorMeta

__all__ = [
    "ExpertSet",
    "parse_global_expert_index",
    "make_healthy_experts",
    "make_aliased_experts",
    "make_local_name_experts",
    "make_empty_experts",
]

SEED = "foundationscale-fixtures-v1"
"""Fixed seed for all synthesised bytes.

Changing this changes every control fixture byte; it is versioned so that a change
is a deliberate, reviewable event and never an accident of reordering a call site.
"""

DEFAULT_TENSOR_BYTES = 256
"""Size of each synthesised tensor. Small enough that a 128-expert control stays tiny."""

_PER_EXPERT_PARAMS: tuple[str, ...] = ("linear_fc1.weight", "linear_fc2.weight")

_GLOBAL_INDEX_RE = re.compile(r"\.experts\.(\d+)\.")


def parse_global_expert_index(name: str) -> int | None:
    """Extract the global expert index from a well-formed expert tensor name.

    The canonical save format embeds the expert identity in the path, e.g.
    ``layers.3.experts.42.linear_fc1.weight``. Returns ``None`` for names that do
    not carry a global index — most importantly the corrupted local-name form
    ``layers.3.experts.linear_fc1.weight42``, which is the signature this module
    exists to reproduce.
    """
    m = _GLOBAL_INDEX_RE.search(name)
    return int(m.group(1)) if m else None


def _blob(*parts: object, nbytes: int = DEFAULT_TENSOR_BYTES, seed: str = SEED) -> bytes:
    """Deterministic pseudo-tensor content derived from ``parts``.

    Any structural parameter (layer, expert, param name, alias source) that should
    make content distinct is passed as a part. Hashing the parts rather than using
    ``random.Random`` keeps the output stable across process restarts regardless of
    call order — call order is precisely the kind of thing that drifts in CI.
    """
    out = bytearray()
    counter = 0
    while len(out) < nbytes:
        h = hashlib.sha256()
        h.update(seed.encode("utf-8"))
        for part in parts:
            h.update(str(part).encode("utf-8"))
            h.update(b"\x00")
        h.update(counter.to_bytes(4, "big"))
        out.extend(h.digest())
        counter += 1
    return bytes(out[:nbytes])


@dataclass(frozen=True)
class ExpertSet:
    """A synthetic set of expert tensors plus the metadata a reader would supply.

    The fields mirror exactly what a real checkpoint reader hands a gate, so a
    control written against an :class:`ExpertSet` reads the same as the production
    call path:

    Args:
        tensors: Checkpoint key to raw tensor content.
        declared_expert_count: What the *model config* says the expert count is.
            This must come from the model, never from counting keys in the artifact —
            counting the artifact is how "16 tensors found, everything present,
            rc=0" happened on a 128-expert model.
        expert_index: Checkpoint key to its *global* expert index, for keys that
            carry one. Empty when keys are spelled with local indices — the gate is
            expected to catch that from the names themselves.
    """

    tensors: Mapping[str, bytes] = field(default_factory=dict)
    declared_expert_count: int = 0
    expert_index: Mapping[str, int] = field(default_factory=dict)


def _global_names(
    num_experts: int,
    num_layers: int,
    *,
    source_expert: Callable[[int], int] | None = None,
    nbytes: int = DEFAULT_TENSOR_BYTES,
    seed: str = SEED,
) -> tuple[dict[str, bytes], dict[str, int]]:
    """Build global-form expert names; ``source_expert(i)`` picks whose bytes expert ``i`` gets.

    With the identity function this produces a healthy set. With ``lambda i: i % 16``
    it produces the audited incident verbatim: every expert's bytes are those of the
    expert 16 positions below, so shapes/counts/dtypes stay perfect while the
    content silently repeats.
    """
    src = source_expert or (lambda i: i)
    tensors: dict[str, bytes] = {}
    index: dict[str, int] = {}
    for layer in range(num_layers):
        for expert in range(num_experts):
            for param in _PER_EXPERT_PARAMS:
                name = f"layers.{layer}.experts.{expert}.{param}"
                tensors[name] = _blob("global", layer, src(expert), param, seed=seed, nbytes=nbytes)
                index[name] = expert
    return tensors, index


def make_healthy_experts(
    num_experts: int = 8,
    *,
    num_layers: int = 2,
    nbytes: int = DEFAULT_TENSOR_BYTES,
    seed: str = SEED,
) -> ExpertSet:
    """A known-good expert set: globally-spelled names, unique content per expert.

    This is the MUST_PASS fixture. Its purpose is to guard against gate behaviour
    that is the opposite of silent success but just as costly in practice — a gate
    that blocks on everything, which in production gets disabled rather than fixed
    and is thereafter as useless as a gate that blocks on nothing.
    """
    tensors, index = _global_names(num_experts, num_layers, nbytes=nbytes, seed=seed)
    return ExpertSet(tensors=tensors, declared_expert_count=num_experts, expert_index=index)


def make_aliased_experts(
    num_experts: int = 128,
    *,
    period: int = 16,
    num_layers: int = 2,
    nbytes: int = DEFAULT_TENSOR_BYTES,
    seed: str = SEED,
) -> ExpertSet:
    """The incident itself: ``num_experts`` experts stored as ``period`` replicated ones.

    Expert ``i`` receives the exact bytes of expert ``i % period`` in every layer
    and every parameter, so with the default arguments 128 experts are 16 experts
    replicated 8 times — the audited run's geometry, reproduced deliberately so the
    control reads as a regression test for that exact defect rather than a
    hypothetical one. Tensor count, shapes, dtypes and byte sizes are all perfectly
    correct, which is the entire point: only content comparison sees this.

    Args:
        num_experts: Experts the model declares and the artifact appears to hold.
        period: How many genuinely distinct experts exist on disk. Must divide
            ``num_experts`` evenly, as 16 divided 128.
    """
    if num_experts % period != 0:
        raise ValueError(
            f"num_experts ({num_experts}) must be a multiple of period ({period}); "
            f"the incident being modelled is an exact replication"
        )
    tensors, index = _global_names(
        num_experts,
        num_layers,
        source_expert=lambda i: i % period,
        nbytes=nbytes,
        seed=seed,
    )
    return ExpertSet(tensors=tensors, declared_expert_count=num_experts, expert_index=index)


def make_local_name_experts(
    num_local: int = 16,
    *,
    declared_expert_count: int = 128,
    num_layers: int = 1,
    nbytes: int = DEFAULT_TENSOR_BYTES,
    seed: str = SEED,
) -> ExpertSet:
    """The on-disk signature of the incident: local indices baked into tensor names.

    Produces keys of the form ``layers.0.experts.linear_fc1.weight0`` ..
    ``...weight15`` — a trailing LOCAL expert index where a per-expert path should
    be. This is what a checkpoint looks like when the saving module held only its
    local shard under unqualified names. The defect is visible in the key set
    alone: no global expert identity survives in these names, so content
    comparison is impossible and the artifact must be rejected before serve.
    """
    tensors: dict[str, bytes] = {}
    for layer in range(num_layers):
        for local in range(num_local):
            for param in _PER_EXPERT_PARAMS:
                tensors[f"layers.{layer}.experts.{param}{local}"] = _blob(
                    "local", layer, local, param, seed=seed, nbytes=nbytes
                )
    return ExpertSet(
        tensors=tensors,
        declared_expert_count=declared_expert_count,
        expert_index={},  # no global identity exists in these names, by construction
    )


def make_empty_experts(declared_expert_count: int = 128) -> ExpertSet:
    """An artifact with no expert tensors at all, against a model that declares many.

    This is the positive control for the audit's sharpest finding: the verification
    tool that reported ``all_identity: True`` on a corrupt artifact because its
    comparison set was empty and ``all([])`` is ``True``. Any gate claiming to guard
    expert integrity must demonstrably NOT report success on this input. Per the
    gate contract, ``Gate.ok`` downgrades zero-coverage results to VACUOUS no
    matter what the author writes; this fixture exists to prove nobody designed
    around that.
    """
    return ExpertSet(tensors={}, declared_expert_count=declared_expert_count, expert_index={})


# -----------------------------------------------------------------------------
# CheckpointGateContext fixtures for the checkpoint gates
#
# These fixtures deliberately contain metadata only. The checkpoint gates are
# metadata gates: they inspect names, shapes, declared counts, byte volume and
# storage identity without reading launcher payloads. Storage identifiers are
# therefore generated with the same deterministic hash machinery as the byte
# fixtures above.

__all__ = [
    *__all__,
    "aliased_local_names_ctx",
    "bloated_extra_state_ctx",
    "empty_expert_set_ctx",
    "healthy_fused_moe_ctx",
    "healthy_sharded_moe_ctx",
    "missing_shard_ctx",
    "mixed_expert_layout_ctx",
    "right_count_aliased_storage_ctx",
    "stacked_aliased_layers_ctx",
    "stacked_hf_moe_ctx",
    "stacked_underfilled_ctx",
    "unknown_expert_layout_ctx",
]

_CONTEXT_ORIGIN_ROOT = "fixture://checkpoint-gates"
_EXPERT_DTYPE = "bfloat16"
_EXPERT_INNER_SHAPE: tuple[int, int] = (16, 32)

_DECLARED_EXPERT_COUNT = 128
_DECLARED_MOE_LAYERS = 1
_LOCAL_SHARD_COUNT = 16
_LOCAL_ALIAS_STORAGE_COUNT = 2
_RIGHT_COUNT_ALIAS_STORAGE_COUNT = 16


def _storage_id(kind: str, *parts: object) -> str:
    """Return a deterministic identifier for one synthesised byte storage."""
    return _blob(kind, *parts, seed=SEED, nbytes=32).hex()


def _fused_expert_fqn(layer: int, parameter: str) -> str:
    return f"layers.{layer}.mlp.experts.experts.{parameter}"


def _sharded_expert_fqn(layer: int, parameter: str, index: int) -> str:
    return f"{_fused_expert_fqn(layer, parameter)}{index}"


def _one_expert_weight_bytes() -> int:
    return TensorMeta(
        fqn="fixture.declared_one_expert_weight",
        shape=_EXPERT_INNER_SHAPE,
        dtype=_EXPERT_DTYPE,
    ).implied_nbytes


def _declared_expert_bytes(num_experts: int, num_layers: int) -> int:
    return _one_expert_weight_bytes() * num_experts * num_layers * len(_PER_EXPERT_PARAMS)


def _make_context(
    *,
    name: str,
    tensors: tuple[TensorMeta, ...],
    declared_fqns: tuple[str, ...] | None,
    num_experts: int,
    num_moe_layers: int,
    expected_expert_bytes: int,
) -> CheckpointGateContext:
    return CheckpointGateContext(
        tensors=tensors,
        declared_fqns=declared_fqns,
        num_experts=num_experts,
        num_moe_layers=num_moe_layers,
        expected_expert_bytes=expected_expert_bytes,
        origin=f"{_CONTEXT_ORIGIN_ROOT}/{name}",
    )


def aliased_local_names_ctx() -> CheckpointGateContext:
    """The 16-shards-of-128 checkpoint incident, reproduced from metadata.

    The run declares 128 experts, but each expert weight exists only as
    ``weight0`` .. ``weight15``. Each stem has two distinct storages, each
    shared by eight shards, preserving the incident's 8-way over-write. The
    summed expert bytes are exactly one eighth of the declared volume —
    5.71 GB against 45.70 GB in the audited checkpoint.
    """
    tensors: list[TensorMeta] = []
    declared_fqns: list[str] = []

    for layer in range(_DECLARED_MOE_LAYERS):
        for parameter in _PER_EXPERT_PARAMS:
            declared_fqns.extend(
                _sharded_expert_fqn(layer, parameter, expert)
                for expert in range(_DECLARED_EXPERT_COUNT)
            )
            for local_expert in range(_LOCAL_SHARD_COUNT):
                tensors.append(
                    TensorMeta(
                        fqn=_sharded_expert_fqn(layer, parameter, local_expert),
                        shape=(1, *_EXPERT_INNER_SHAPE),
                        dtype=_EXPERT_DTYPE,
                        storage_id=_storage_id(
                            "aliased-local-shard",
                            layer,
                            parameter,
                            local_expert % _LOCAL_ALIAS_STORAGE_COUNT,
                        ),
                    )
                )

    return _make_context(
        name="aliased-local-names",
        tensors=tuple(tensors),
        declared_fqns=tuple(declared_fqns),
        num_experts=_DECLARED_EXPERT_COUNT,
        num_moe_layers=_DECLARED_MOE_LAYERS,
        expected_expert_bytes=_declared_expert_bytes(
            _DECLARED_EXPERT_COUNT,
            _DECLARED_MOE_LAYERS,
        ),
    )


def empty_expert_set_ctx() -> CheckpointGateContext:
    """A declared-MoE checkpoint with no expert tensors: the ``all([])`` trap.

    The checkpoint is otherwise coherent and contains present, declared
    non-expert tensors, so a block cannot be attributed to an unrelated
    malformed fixture. The missing expert set is the defect. On this input,
    success would mean \"every expert matched\" without comparing any expert.
    """
    num_layers = 2
    tensors = tuple(
        TensorMeta(
            fqn=f"layers.{layer}.attention.self_attention.linear_proj.weight",
            shape=(256, 512),
            dtype="float32",
            storage_id=_storage_id("empty-expert-dense", layer),
        )
        for layer in range(num_layers)
    )
    return _make_context(
        name="empty-expert-set",
        tensors=tensors,
        declared_fqns=tuple(meta.fqn for meta in tensors),
        num_experts=_DECLARED_EXPERT_COUNT,
        num_moe_layers=num_layers,
        expected_expert_bytes=_declared_expert_bytes(
            _DECLARED_EXPERT_COUNT,
            num_layers,
        ),
    )


def right_count_aliased_storage_ctx() -> CheckpointGateContext:
    """The subtle aliasing variant the tensor-count check cannot see.

    All 128 declared shards are present for every expert weight, so a gate
    that only counts names is green. Every stem nevertheless contains only 16
    distinct storages, each addressed by eight different tensor names: the
    count is correct, but 112 expert identities are aliases.
    """
    tensors: list[TensorMeta] = []
    declared_fqns: list[str] = []

    for layer in range(_DECLARED_MOE_LAYERS):
        for parameter in _PER_EXPERT_PARAMS:
            for expert in range(_DECLARED_EXPERT_COUNT):
                fqn = _sharded_expert_fqn(layer, parameter, expert)
                declared_fqns.append(fqn)
                tensors.append(
                    TensorMeta(
                        fqn=fqn,
                        shape=(1, *_EXPERT_INNER_SHAPE),
                        dtype=_EXPERT_DTYPE,
                        storage_id=_storage_id(
                            "right-count-aliased-storage",
                            layer,
                            parameter,
                            expert % _RIGHT_COUNT_ALIAS_STORAGE_COUNT,
                        ),
                    )
                )

    return _make_context(
        name="right-count-aliased-storage",
        tensors=tuple(tensors),
        declared_fqns=tuple(declared_fqns),
        num_experts=_DECLARED_EXPERT_COUNT,
        num_moe_layers=_DECLARED_MOE_LAYERS,
        expected_expert_bytes=_declared_expert_bytes(
            _DECLARED_EXPERT_COUNT,
            _DECLARED_MOE_LAYERS,
        ),
    )


def healthy_fused_moe_ctx(
    num_experts: int = 8,
    *,
    num_layers: int = 2,
) -> CheckpointGateContext:
    """A non-vacuous known-good fused MoE checkpoint.

    Each MoE layer has one ``linear_fc1`` and one ``linear_fc2`` weight whose
    leading dimension is the declared expert count, with unique storage for
    every tensor. This is the healthy counterpart of the audited incident and
    the positive control against a checkpoint gate that blocks everything.
    """
    tensors: list[TensorMeta] = []

    for layer in range(num_layers):
        for parameter in _PER_EXPERT_PARAMS:
            fqn = _fused_expert_fqn(layer, parameter)
            tensors.append(
                TensorMeta(
                    fqn=fqn,
                    shape=(num_experts, *_EXPERT_INNER_SHAPE),
                    dtype=_EXPERT_DTYPE,
                    storage_id=_storage_id(
                        "healthy-fused-expert",
                        layer,
                        parameter,
                        num_experts,
                    ),
                )
            )

    context_tensors = tuple(tensors)
    return _make_context(
        name=f"healthy-fused-{num_layers}x{num_experts}",
        tensors=context_tensors,
        declared_fqns=tuple(meta.fqn for meta in context_tensors),
        num_experts=num_experts,
        num_moe_layers=num_layers,
        expected_expert_bytes=sum(meta.implied_nbytes for meta in context_tensors),
    )


def missing_shard_ctx() -> CheckpointGateContext:
    """A save in which one shard was never written while launch still exited 0.

    Shards 0 and 1 contribute all four of their tensors; shard 2 contributes
    none, leaving two of the six declared tensors absent. Filtering to real
    tensors is essential: the defect must not disappear underneath unrelated
    checkpoint metadata keys.
    """

    def shard_fqn(shard: int, layer: int) -> str:
        return f"checkpoint.shards.{shard}.layers.{layer}.weight"

    declared_fqns = tuple(shard_fqn(shard, layer) for shard in range(3) for layer in range(2))
    present = tuple(
        TensorMeta(
            fqn=shard_fqn(shard, layer),
            shape=(256, 256),
            dtype="float32",
            storage_id=_storage_id("missing-shard-present", shard, layer),
        )
        for shard in range(2)
        for layer in range(2)
    )
    return _make_context(
        name="missing-shard-2",
        tensors=present,
        declared_fqns=declared_fqns,
        num_experts=0,
        num_moe_layers=0,
        expected_expert_bytes=0,
    )


def bloated_extra_state_ctx() -> CheckpointGateContext:
    """64 real tensors buried under 512 ``_extra_state`` metadata blobs.

    This mirrors the audited 26B checkpoint's 928 real tensors among roughly
    8,970 metadata keys, at fixture scale. Counting every metadata entry would
    report 576 healthy units; the fixture proves completeness is measured
    against only the 64 tensors that actually matter.
    """
    real_count = 64
    blobs_per_tensor = 8

    real_tensors = tuple(
        TensorMeta(
            fqn=f"model.layers.{index}.linear.weight",
            shape=(64, 64),
            dtype="bfloat16",
            storage_id=_storage_id("bloated-real-tensor", index),
        )
        for index in range(real_count)
    )
    metadata_blobs = tuple(
        TensorMeta(
            fqn=f"{real_tensors[index].fqn}._extra_state.{blob_index}",
            shape=(8,),
            dtype="uint8",
            storage_id=_storage_id(
                "bloated-extra-state",
                index,
                blob_index,
            ),
            kind="extra_state",
        )
        for index in range(real_count)
        for blob_index in range(blobs_per_tensor)
    )
    tensors = real_tensors + metadata_blobs
    return _make_context(
        name="bloated-extra-state",
        tensors=tensors,
        declared_fqns=tuple(meta.fqn for meta in tensors),
        num_experts=0,
        num_moe_layers=0,
        expected_expert_bytes=0,
    )


# -----------------------------------------------------------------------------
# Layout fixtures beyond the Megatron family
#
# The real-artifact probe found the selector underneath these gates matched only
# Megatron-Core naming. The builders below cover the other two families the gates
# now see: the PER-EXPERT layout done correctly (healthy counterpart of the
# incident), and the STACKED layout (every expert of a layer on dim 0 of one
# tensor), including its two metadata-visible corruption signatures and the
# fail-closed unknown family. Naming mirrors the probed Gemma-4 checkpoint.


def healthy_sharded_moe_ctx(
    num_experts: int = 8,
    *,
    num_layers: int = 2,
) -> CheckpointGateContext:
    """A known-good PER-EXPERT checkpoint: one FQN and one storage span per expert.

    Same local-name SHAPE as the incident fixture (``...linear_fc1.weight<i>``),
    but at the full declared count with a distinct storage per shard — the only
    layout family in which expert distinctness can fully verify from metadata,
    and therefore the healthy fixture for any composite that must reach 3/3.
    """
    tensors: list[TensorMeta] = []
    for layer in range(num_layers):
        for parameter in _PER_EXPERT_PARAMS:
            for expert in range(num_experts):
                tensors.append(
                    TensorMeta(
                        fqn=_sharded_expert_fqn(layer, parameter, expert),
                        shape=(1, *_EXPERT_INNER_SHAPE),
                        dtype=_EXPERT_DTYPE,
                        storage_id=_storage_id(
                            "healthy-sharded-expert",
                            layer,
                            parameter,
                            expert,
                        ),
                    )
                )
    context_tensors = tuple(tensors)
    return _make_context(
        name=f"healthy-sharded-{num_layers}x{num_experts}",
        tensors=context_tensors,
        declared_fqns=tuple(meta.fqn for meta in context_tensors),
        num_experts=num_experts,
        num_moe_layers=num_layers,
        expected_expert_bytes=sum(meta.implied_nbytes for meta in context_tensors),
    )


_STACKED_PROJECTIONS: tuple[tuple[str, tuple[int, int]], ...] = (
    ("gate_up_proj", (16, 32)),
    ("down_proj", (32, 16)),
)


def _stacked_expert_fqn(layer: int, projection: str) -> str:
    return f"model.language_model.layers.{layer}.experts.{projection}"


def _stacked_declared_volume(num_experts: int, num_layers: int) -> int:
    """The volume a correct run would declare: experts x per-slice bytes x weights."""
    return sum(
        TensorMeta(
            fqn=f"fixture.stacked-declared.{layer}.{projection}",
            shape=(num_experts, *inner),
            dtype=_EXPERT_DTYPE,
        ).implied_nbytes
        for layer in range(num_layers)
        for projection, inner in _STACKED_PROJECTIONS
    )


def stacked_hf_moe_ctx(
    num_experts: int = 8,
    *,
    num_layers: int = 2,
) -> CheckpointGateContext:
    """A clean STACKED MoE checkpoint in HF naming — the probed artifact's layout.

    One ``gate_up_proj`` and one ``down_proj`` per layer with every expert on dim 0,
    each tensor on its own storage span, and the declared volume matching the
    implied bytes exactly. Nothing metadata can flag, and nothing metadata can
    prove: the distinctness gate must abstain (SKIP, never "distinct"), the byte
    gate must reach a real verdict, and completeness must match.
    """
    tensors: list[TensorMeta] = []
    for layer in range(num_layers):
        for projection, inner in _STACKED_PROJECTIONS:
            tensors.append(
                TensorMeta(
                    fqn=_stacked_expert_fqn(layer, projection),
                    shape=(num_experts, *inner),
                    dtype=_EXPERT_DTYPE,
                    storage_id=_storage_id(
                        "stacked-hf-expert",
                        layer,
                        projection,
                        num_experts,
                    ),
                )
            )
    context_tensors = tuple(tensors)
    return _make_context(
        name=f"stacked-hf-{num_layers}x{num_experts}",
        tensors=context_tensors,
        declared_fqns=tuple(meta.fqn for meta in context_tensors),
        num_experts=num_experts,
        num_moe_layers=num_layers,
        expected_expert_bytes=sum(meta.implied_nbytes for meta in context_tensors),
    )


def stacked_aliased_layers_ctx(num_experts: int = 8) -> CheckpointGateContext:
    """Two stacked tensors in DIFFERENT layers pointing at one storage span.

    Whole-tensor, cross-layer aliasing is the only aliasing signature that
    survives stacking: every leading dim is correct, every sibling ratio is
    consistent, and yet the two layers' down projections are the same bytes. The
    distinctness gate must fire on the shared span.
    """
    layers = 2
    shared_down_proj_storage = _storage_id("stacked-aliased-down-proj", num_experts)
    tensors: list[TensorMeta] = []
    for layer in range(layers):
        for projection, inner in _STACKED_PROJECTIONS:
            tensors.append(
                TensorMeta(
                    fqn=_stacked_expert_fqn(layer, projection),
                    shape=(num_experts, *inner),
                    dtype=_EXPERT_DTYPE,
                    storage_id=(
                        shared_down_proj_storage
                        if projection == "down_proj"
                        else _storage_id("stacked-aliased-ok", layer, projection)
                    ),
                )
            )
    context_tensors = tuple(tensors)
    return _make_context(
        name="stacked-cross-layer-alias",
        tensors=context_tensors,
        declared_fqns=tuple(meta.fqn for meta in context_tensors),
        num_experts=num_experts,
        num_moe_layers=layers,
        expected_expert_bytes=_stacked_declared_volume(num_experts, layers),
    )


def stacked_underfilled_ctx(num_experts: int = 8) -> CheckpointGateContext:
    """One stacked tensor holds 1/8 of a layer's experts; its siblings hold them all.

    Layer 0's down_proj has leading dim ``num_experts // 8`` while every other
    stacked weight carries the full declared count — the incident's exact 0.125
    ratio in stacked clothing, visible twice to metadata: leading dim vs declared
    count, and implied bytes vs sibling projections across layers.
    """
    layers = 2
    tensors: list[TensorMeta] = []
    for layer in range(layers):
        for projection, inner in _STACKED_PROJECTIONS:
            # Exactly one projection in one layer is starved to 1/8 of the
            # declared count — the incident's 0.125 ratio, in stacked clothing.
            starved = layer == 0 and projection == "down_proj"
            leading = num_experts // 8 if starved else num_experts
            tensors.append(
                TensorMeta(
                    fqn=_stacked_expert_fqn(layer, projection),
                    shape=(leading, *inner),
                    dtype=_EXPERT_DTYPE,
                    storage_id=_storage_id(
                        "stacked-underfilled",
                        layer,
                        projection,
                        leading,
                    ),
                )
            )
    context_tensors = tuple(tensors)
    return _make_context(
        name="stacked-a-fraction-of-experts",
        tensors=context_tensors,
        declared_fqns=tuple(meta.fqn for meta in context_tensors),
        num_experts=num_experts,
        num_moe_layers=layers,
        expected_expert_bytes=_stacked_declared_volume(num_experts, layers),
    )


def unknown_expert_layout_ctx(num_experts: int = 8) -> CheckpointGateContext:
    """Expert-named tensors matching no layout family: the gates must block.

    ``weight_bank`` under an ``experts`` segment might be a perfectly fine
    format — but no code in this repo knows its per-expert semantics, so both
    treating it as verified and treating the model as dense would be claims
    without evidence. Fail-closed is the only honest answer.
    """
    layers = 2
    tensors = tuple(
        TensorMeta(
            fqn=f"model.layers.{layer}.mlp.experts.weight_bank",
            shape=(num_experts, *_EXPERT_INNER_SHAPE),
            dtype=_EXPERT_DTYPE,
            storage_id=_storage_id("unknown-expert-layout", layer),
        )
        for layer in range(layers)
    )
    return _make_context(
        name="unknown-expert-layout",
        tensors=tensors,
        declared_fqns=tuple(meta.fqn for meta in tensors),
        num_experts=num_experts,
        num_moe_layers=layers,
        expected_expert_bytes=sum(meta.implied_nbytes for meta in tensors),
    )


def mixed_expert_layout_ctx(num_experts: int = 8) -> CheckpointGateContext:
    """Per-expert shards and stacked tensors inside ONE checkpoint: a MIXED layout.

    Layer 0 carries the full declared expert count as one FQN and one storage span
    per expert (the per-expert family); both layers also carry Gemma-style stacked
    projections with every expert on leading dim 0 (the stacked family). Each
    family on its own would verify clean, which is precisely the trap: examined
    independently they report full denominators over their own subsets while
    nothing examined the checkpoint as a whole. Whether this is a heterogeneous
    model or a half-converted artifact is not decidable from layout metadata, so
    the distinctness gate must block on the mixture itself, not fold either
    family into the other's count.
    """
    layers = 2
    tensors: list[TensorMeta] = []
    for layer in range(layers):
        for projection, inner in _STACKED_PROJECTIONS:
            tensors.append(
                TensorMeta(
                    fqn=_stacked_expert_fqn(layer, projection),
                    shape=(num_experts, *inner),
                    dtype=_EXPERT_DTYPE,
                    storage_id=_storage_id(
                        "mixed-stacked-expert",
                        layer,
                        projection,
                    ),
                )
            )
    for parameter in _PER_EXPERT_PARAMS:
        for expert in range(num_experts):
            tensors.append(
                TensorMeta(
                    fqn=_sharded_expert_fqn(0, parameter, expert),
                    shape=(1, *_EXPERT_INNER_SHAPE),
                    dtype=_EXPERT_DTYPE,
                    storage_id=_storage_id(
                        "mixed-per-expert-shard",
                        parameter,
                        expert,
                    ),
                )
            )
    context_tensors = tuple(tensors)
    return _make_context(
        name="mixed-expert-layout",
        tensors=context_tensors,
        declared_fqns=tuple(meta.fqn for meta in context_tensors),
        num_experts=num_experts,
        num_moe_layers=layers,
        expected_expert_bytes=sum(meta.implied_nbytes for meta in context_tensors),
    )
