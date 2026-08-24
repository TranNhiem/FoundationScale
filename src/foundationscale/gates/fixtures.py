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

Defect-bearing builders REFUSE parameterisations that would synthesise no defect
--------------------------------------------------------------------------------
A MUST_FIRE fixture that contains no defect is worse than no fixture. The gate
contract downgrades zero-work results to VACUOUS, and VACUOUS BLOCKS — so a
MUST_FIRE control built on an empty or defect-free "broken" fixture is satisfied
by the gate's own vacuity tripwire, never by detection, and the control suite
goes green over a detector that was exercised zero times on its defect. The
guard therefore lives at the source, not in consumer discipline. The builders
below raise ``ValueError`` on any parameterisation whose output could not
contain the defect their name promises: :func:`make_aliased_experts` requires a
source map that actually replicates bytes (``period`` strictly smaller than
``num_experts``, at least one expert, at least one layer),
:func:`make_local_name_experts` requires at least one local-named tensor, and
:func:`make_empty_experts` requires a declared count of at least one (its
defect is absence-while-declared — declaring zero experts makes it a dense
artifact containing nothing to fire on). The HEALTHY builder is deliberately
not guarded this way: ``make_healthy_experts(num_experts=0)`` is a legitimate
dense-model MUST_PASS input, and refusing it would mint a defect where none
exists. Guards refuse stories; they do not refuse layouts.

"Tensors" here are plain ``bytes`` derived from a fixed seed via SHA-256, so the
same call always produces the same bytes on every platform and Python process.
There is deliberately no ``random`` module use anywhere in this file.
"""

from __future__ import annotations

import hashlib
import math
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
            Must be at least 1: zero experts means zero tensors, and an empty
            "aliased" fixture carries no defect for any gate to detect.
        period: How many genuinely distinct experts exist on disk. Must divide
            ``num_experts`` evenly, as 16 divided 128, and must be STRICTLY
            smaller than ``num_experts``. At equality the source map is the
            identity, every synthesised expert is byte-distinct, and the
            fixture is a healthy expert set wearing a MUST_FIRE label. Build
            that with :func:`make_healthy_experts`; this builder must not mint it.

    Raises:
        ValueError: On any parameterisation whose output would not materially
            contain the replication defect — see the module-docstring section
            "Defect-bearing builders REFUSE...".
    """
    if num_experts < 1:
        raise ValueError(
            f"make_aliased_experts(num_experts={num_experts}): the fixture would "
            f"contain ZERO tensors. Its only way to make a gate block is the "
            f"gate's vacuity tripwire, which means the aliasing detector itself "
            f"was exercised zero times while the MUST_FIRE control read green. "
            f"Refusing at the source."
        )
    if num_layers < 1:
        raise ValueError(
            f"make_aliased_experts(num_layers={num_layers}): zero layers means "
            f"zero tensors — the same vacuous-MUST_FIRE trap as num_experts=0, "
            f"one loop level up."
        )
    if period < 1:
        # This used to fall through to `num_experts % period` and die as a bare
        # ZeroDivisionError — a crash wearing guard's clothing. A refusal with a
        # reason is the contract every other guard in this file keeps.
        raise ValueError(
            f"make_aliased_experts(period={period}): the period is the count of "
            f"genuinely distinct experts on disk and must be at least 1"
        )
    if num_experts % period != 0:
        raise ValueError(
            f"num_experts ({num_experts}) must be a multiple of period ({period}); "
            f"the incident being modelled is an exact replication"
        )
    if period >= num_experts:
        raise ValueError(
            f"period ({period}) must be strictly smaller than num_experts "
            f"({num_experts}): at equality source_expert(i) = i % {period} = i, "
            f"the map is the identity, and every expert's bytes are distinct — "
            f"the fixture would contain NO DEFECT, and a dependent MUST_FIRE "
            f"control could hold only via a gate's vacuity tripwire, never via "
            f"alias detection. If a distinct-bytes expert set is what a test "
            f"needs, that set is healthy: build it with make_healthy_experts "
            f"and label the control MUST_PASS."
        )
    tensors, index = _global_names(
        num_experts,
        num_layers,
        source_expert=lambda i: i % period,
        nbytes=nbytes,
        seed=seed,
    )
    # Post-build proof that the defect is materially present, not merely
    # intended. The parameter guards above make replication a theorem of
    # _global_names AS WRITTEN TODAY; this check is what keeps the promise if
    # _global_names or _blob ever change — say, hashing the destination expert
    # index into every payload, a one-line edit that would silently un-alias
    # every fixture built here and demote every dependent MUST_FIRE control to
    # a vacuity check again. A fixture must fail loudly the day it stops
    # containing its defect.
    if len(set(tensors.values())) == len(tensors):
        raise RuntimeError(
            f"fixture invariant broken: {len(tensors)} tensors synthesised and "
            f"every payload is byte-distinct. The replication source map lost "
            f"its effect; investigate _global_names/_blob before trusting any "
            f"control built on this fixture."
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

    Raises:
        ValueError: If the call would synthesise zero tensors. The defect this
            builder ships is a NAMING signature, and zero tensors means zero
            names — nothing to detect, so a MUST_FIRE control built on it would
            hold only via a gate's vacuity tripwire while claiming the name
            check was exercised.
    """
    if num_layers < 1 or num_local < 1:
        raise ValueError(
            f"make_local_name_experts(num_local={num_local}, num_layers={num_layers}): "
            f"the fixture would contain zero tensors. The defect is a naming "
            f"signature; zero tensors means zero names, and a claim that the "
            f"signature was checked, made over zero names, is the all([]) "
            f"verdict one level up. Refusing at the source."
        )
    tensors: dict[str, bytes] = {}
    for layer in range(num_layers):
        for local in range(num_local):
            for param in _PER_EXPERT_PARAMS:
                tensors[f"layers.{layer}.experts.{param}{local}"] = _blob(
                    "local", layer, local, param, seed=seed, nbytes=nbytes
                )
    # Same post-build proof as make_aliased_experts: the defect must be visible
    # in the artifact, not just in the builder's intent. Every synthesised key
    # must lack a parseable GLOBAL expert index — that absence is the fixture's
    # entire claim, and a key that regains one (a renamed format string above)
    # silently converts the corrupt-signature fixture into a healthy global one.
    globally_named = [k for k in tensors if parse_global_expert_index(k) is not None]
    if not tensors or globally_named:
        raise RuntimeError(
            f"fixture invariant broken: {len(tensors)} tensors synthesised, "
            f"{len(globally_named)} of which (e.g. {globally_named[:3]}) carry a "
            f"parseable global expert index — this fixture exists because its "
            f"names hold NO global identity."
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

    ``declared_expert_count`` must be at least 1: this fixture's defect is
    absence WHILE DECLARED. A declared count of zero makes the absence declared
    and expected — a dense-model artifact containing no defect — and gates
    typically respond to that with an abstention ("dense, not applicable"), so
    a MUST_FIRE control built on it would exercise a skip path, never the
    empty-comparison tripwire it exists to pin.
    """
    if declared_expert_count < 1:
        raise ValueError(
            f"make_empty_experts(declared_expert_count={declared_expert_count}): "
            f"this fixture's defect is experts being ABSENT WHILE DECLARED. With "
            f"a declared count of zero it mints a dense-model artifact — no "
            f"defect, no tripwire. If a dense artifact is the intent, that is a "
            f"MUST_PASS input; label it so instead of borrowing this name."
        )
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
    "dense_declared_ctx",
    "empty_expert_set_ctx",
    "healthy_fused_moe_ctx",
    "healthy_sharded_moe_ctx",
    "manifestless_moe_ctx",
    "missing_shard_ctx",
    "mixed_expert_layout_ctx",
    "right_count_aliased_storage_ctx",
    "stacked_aliased_layers_ctx",
    "stacked_hf_moe_ctx",
    "stacked_underfilled_ctx",
    "unknown_expert_layout_ctx",
    "unpriceable_dtype_ctx",
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


def _priced_nbytes(meta: TensorMeta) -> int:
    """Price a fixture tensor, refusing outright when the dtype is unpriceable.

    ``implied_nbytes`` is now ``None`` outside the price table, and the fixtures
    bake declared volumes into manifest fields, so a guessed width here would
    poison every control's denominator at once. Fixture dtypes are all
    priceable; a None is therefore a fixture bug, and fixtures fail loudly.
    """
    nbytes = meta.implied_nbytes
    if nbytes is None:
        raise ValueError(
            f"fixture dtype {meta.dtype!r} is outside the price table; controls "
            "must declare volumes from dtypes the gates can actually price"
        )
    return nbytes


def _fused_expert_fqn(layer: int, parameter: str) -> str:
    return f"layers.{layer}.mlp.experts.experts.{parameter}"


def _sharded_expert_fqn(layer: int, parameter: str, index: int) -> str:
    return f"{_fused_expert_fqn(layer, parameter)}{index}"


def _one_expert_weight_bytes() -> int:
    return _priced_nbytes(
        TensorMeta(
            fqn="fixture.declared_one_expert_weight",
            shape=_EXPERT_INNER_SHAPE,
            dtype=_EXPERT_DTYPE,
        )
    )


def _declared_expert_bytes(num_experts: int, num_layers: int) -> int:
    return _one_expert_weight_bytes() * num_experts * num_layers * len(_PER_EXPERT_PARAMS)


def _make_context(
    *,
    name: str,
    tensors: tuple[TensorMeta, ...],
    declared_fqns: tuple[str, ...] | None,
    num_experts: int | None,
    num_moe_layers: int | None,
    expected_expert_bytes: int | None,
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
        expected_expert_bytes=sum(_priced_nbytes(meta) for meta in context_tensors),
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
        expected_expert_bytes=sum(_priced_nbytes(meta) for meta in context_tensors),
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
        _priced_nbytes(
            TensorMeta(
                fqn=f"fixture.stacked-declared.{layer}.{projection}",
                shape=(num_experts, *inner),
                dtype=_EXPERT_DTYPE,
            )
        )
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
        expected_expert_bytes=sum(_priced_nbytes(meta) for meta in context_tensors),
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
        expected_expert_bytes=sum(_priced_nbytes(meta) for meta in tensors),
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
        expected_expert_bytes=sum(_priced_nbytes(meta) for meta in context_tensors),
    )


def manifestless_moe_ctx() -> CheckpointGateContext:
    """NO run manifest at all: every declared_* field is None — the gutted-MoE trap.

    This is exactly what ``CheckpointGateContext.from_path`` builds for a
    checkpoint saved with no manifest beside it. The artifact is otherwise
    coherent — present, well-formed dense tensors — so no block can be blamed
    on a malformed fixture. If the run was MoE and the expert tensors were
    stripped or never written, the artifact arrives indistinguishable from a
    true dense model, and "zero experts examined" over an UNKNOWN declaration
    must VACUOUS-block, never inherit the dense-model SKIP. The explicit-0
    twin of this door is pinned inline in test_checkpoint_gate_gaps.
    """
    num_layers = 2
    tensors = tuple(
        TensorMeta(
            fqn=f"layers.{layer}.attention.self_attention.linear_proj.weight",
            shape=(256, 512),
            dtype="float32",
            storage_id=_storage_id("manifestless-dense", layer),
        )
        for layer in range(num_layers)
    )
    return CheckpointGateContext(
        tensors=tensors,
        declared_fqns=None,
        num_experts=None,
        num_moe_layers=None,
        expected_expert_bytes=None,
        origin=f"{_CONTEXT_ORIGIN_ROOT}/manifestless-moe",
    )


def unpriceable_dtype_ctx(num_experts: int = 8, *, num_layers: int = 2) -> CheckpointGateContext:
    """A coherent STACKED MoE checkpoint in a dtype the gates cannot price.

    Gemma-style names, correct leading dims, distinct storage per tensor —
    every metadata observable is clean except that ``float8_e4m3fn`` (a dtype
    dcp_meta legitimately parses from real safetensors headers) has no entry
    in ``_DTYPE_BYTES``. The declared volume is computed at the TRUE 1-byte
    width inside the fixture — fixtures may know float8's element size
    precisely so the block can never be excused by a wrong manifest; the gates
    may not guess it. That asymmetry is the control: the byte gate must refuse
    to price, and the distinctness abstention must not claim sibling byte
    pricing was examined.
    """
    tensors: list[TensorMeta] = []
    for layer in range(num_layers):
        for projection, inner in _STACKED_PROJECTIONS:
            tensors.append(
                TensorMeta(
                    fqn=_stacked_expert_fqn(layer, projection),
                    shape=(num_experts, *inner),
                    dtype="float8_e4m3fn",
                    storage_id=_storage_id("unpriceable-expert-dtype", layer, projection),
                )
            )
    context_tensors = tuple(tensors)
    return _make_context(
        name=f"unpriceable-dtype-{num_layers}x{num_experts}",
        tensors=context_tensors,
        declared_fqns=tuple(meta.fqn for meta in context_tensors),
        num_experts=num_experts,
        num_moe_layers=num_layers,
        expected_expert_bytes=sum(math.prod(meta.shape) for meta in context_tensors),
    )


def dense_declared_ctx() -> CheckpointGateContext:
    """A checkpoint from a run that POSITIVELY declared itself dense.

    ``num_experts=0`` here is a DECLARATION — "this model has no experts" — and
    is categorically different from ``None`` ("nothing was declared"), which is
    :func:`manifestless_moe_ctx`'s shape and must VACUOUS-block through the
    expert gates' UNKNOWN doors. This fixture exists to pin the shrink-safe
    half of the composite's applicable-denominator pricing: the two expert
    gates abstain NOT_APPLICABLE on it, completeness verifies the four real
    tensors it does carry, and the composite must therefore reach 1/1
    applicable — non-blocking, with both abstentions NAMED in its detail. If
    this context ever produces "verified 3/3", bare "verified", or a block, the
    distinction the composite exists to enforce has rotted.
    """
    num_layers = 2
    tensors = tuple(
        TensorMeta(
            fqn=f"layers.{layer}.attention.self_attention.linear_proj.weight",
            shape=(256, 512),
            dtype="float32",
            storage_id=_storage_id("dense-declared", layer),
        )
        for layer in range(num_layers)
    )
    return _make_context(
        name="dense-declared",
        tensors=tensors,
        declared_fqns=tuple(meta.fqn for meta in tensors),
        num_experts=0,
        num_moe_layers=0,
        expected_expert_bytes=0,
    )
