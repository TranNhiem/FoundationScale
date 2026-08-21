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
