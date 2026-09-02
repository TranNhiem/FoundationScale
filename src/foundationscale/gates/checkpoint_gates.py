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

Layout coverage on real artifacts
---------------------------------
The expert selector sees three PER-EXPERT namings — Megatron local-name shards
(``...linear_fc[12].weight<i>``; the audited incident), Megatron global names
(``...experts.42.linear_fc1.weight``) and Mixtral/Qwen ``...experts.<i>.<proj>.weight``
— plus the STACKED layout that dominates HF MoE (``...experts.gate_up_proj`` /
``...experts.down_proj`` on Gemma-4; Megatron's fused, suffix-less
``...linear_fc1.weight`` is the same thing in older spelling). Stacked changes the
epistemics, measured on a real 48.07 GiB Gemma-4 checkpoint that selected 0 tensors
under the old Megatron-only selector: per-expert aliasing is *not observable at all*
from metadata, because N duplicated slices cost exactly the one storage span that N
distinct slices cost. The distinctness gate therefore verifies everything stacking
still permits (leading dims, cross-layer span sharing, sibling byte ratios) and then
ABSTAINS with a stated reason rather than passing; the byte-volume gate, by contrast,
prices stacked layouts exactly and reaches a real verdict. Expert-ish names matching
none of these families block as an unrecognized layout — never read as a dense model.

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
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

from .core import (
    AbstentionKind,
    Control,
    ControlKind,
    Coverage,
    Gate,
    GateResult,
    Lifecycle,
    Verdict,
    register,
)

if TYPE_CHECKING:
    from foundationscale.provenance.manifest import DeclaredCheckpoint

__all__ = [
    "TensorMeta",
    "CheckpointGateContext",
    "ExpertDistinctnessGate",
    "ExpertByteVolumeGate",
    "SaveCompletenessGate",
    "FirstSaveGate",
    # Naming vocabulary, shared with whoever produces the declaration these
    # gates adjudicate against. See the note beside their definitions.
    "mentions_expert",
    "matches_expert_family",
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

# Per-expert family B — each expert is its own FQN carrying the global expert index
# as a dotted path segment:
#   ...block_sparse_moe.experts.0.w1.weight      (Mixtral)
#   ...mlp.experts.0.gate_proj.weight            (Qwen-MoE)
# Megatron's global spelling (...experts.42.linear_fc1.weight) also lands here.
_PER_EXPERT_MEMBER_RE = re.compile(
    r"^(?P<prefix>.*\.experts\.)(?P<idx>\d+)\.(?P<suffix>[A-Za-z]\w*(?:\.\w+)*)$"
)

# STACKED family — every expert of a layer inside ONE tensor on dim 0, hence one
# storage span, so per-expert identity inside it is metadata-invisible:
#   model.language_model.layers.3.experts.down_proj      (128, 2816, 704)  (Gemma-4)
#   ...mlp.experts.gate_up_proj.weight                                    (GPT-OSS)
# Megatron's fused ...linear_fc[12].weight matches _EXPERT_WEIGHT_RE first and is the
# same stacked epistemology under an older name; the classifier folds it in.
_STACKED_WEIGHT_RE = re.compile(r".*\.experts\.[a-z0-9_]*?(?:proj|fc\d)(?:_bias|\.bias|\.weight)?$")


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
    def implied_nbytes(self) -> int | None:
        """Bytes this FQN *claims* to occupy, computed from shape and dtype alone.

        This is metadata-implied, not measured: 128 aliased FQNs each imply their
        own bytes while naming one underlying storage, so summing this per FQN
        prices the count-correct variant of the aliasing incident at exactly the
        declared volume. Physical accounting lives in :func:`_distinct_storage_bytes`,
        which prices each storage identity once; only a reader-supplied storage
        map can do better.

        Returns ``None`` when ``dtype`` is outside :data:`_DTYPE_BYTES`. The old
        behaviour defaulted unknown dtypes to 4 bytes *silently*: a 1-byte
        float8_e4m3fn expert set (a dtype dcp_meta parses from real safetensors
        headers) was priced at 4x its true volume with no signal anywhere, and a
        manifest declaring the true volume then "matched". A guessed element
        width is a fabricated denominator, and fabricating it inside the price is
        the same class of lie as reporting ``all([]) is True``. Consumers must
        handle ``None`` explicitly — byte accounting blocks, message-only uses
        degrade the message — and the return type makes mypy enforce that they do.
        """
        per_element = _DTYPE_BYTES.get(self.dtype)
        if per_element is None:
            return None
        return math.prod(self.shape) * per_element


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
    # Declared experts per MoE layer. 0 is a DECLARATION of a dense model and
    # earns the dense-model SKIP; None means nothing was declared at all (the
    # from_path shape of a manifestless checkpoint), and the expert gates must
    # read None as UNKNOWN and fail closed on it. Conflating the two is what
    # let a gutted MoE checkpoint strip its experts and skip past both expert
    # gates with the prose "context declares no experts" — a declaration the
    # context never made.
    num_experts: int | None
    num_moe_layers: int | None
    expected_expert_bytes: int | None
    origin: str
    expert_storage_bytes: int | None = None
    """Physical expert bytes measured from the reader's storage map, when it can
    supply one; ``None`` means the byte gate must fall back to per-FQN implied
    bytes and say so in its PASS. Preferred over the implied sum wherever it
    exists — it is the one number aliasing cannot inflate."""

    @classmethod
    def from_path(
        cls,
        path: str | os.PathLike[str],
        *,
        declared: DeclaredCheckpoint | None = None,
    ) -> CheckpointGateContext:
        """Build a context from an on-disk checkpoint.

        Lazily imports ``foundationscale.checkpoint`` so this package imports
        without torch. Assumed API (kept deliberately narrow): ``read_metadata``
        returns an object with ``.tensors: Mapping[str, TensorStorageMeta]`` where
        each meta has ``.shape``/``.dtype`` and optionally ``.storage_id`` /
        ``.is_extra_state``; ``load_manifest`` returns the run manifest (or
        ``None``). The four denominators the gates adjudicate come from the
        manifest's ``declared`` block (a ``DeclaredCheckpoint`` produced at
        launch); adapters written before the block existed may instead expose
        ``declared_fqns``, ``num_experts``, ``num_moe_layers`` and
        ``expected_expert_bytes`` as flat attributes. ``read_metadata`` may
        additionally expose ``.expert_storage_bytes`` — expert bytes measured
        from the reader's storage map.

        An explicit ``declared`` argument overrides the manifest's block, for
        callers that resolved the denominators separately from manifest storage.
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

        block = declared if declared is not None else getattr(manifest, "declared", None)
        if block is not None:
            # An empty declared_fqns list inside a block normalizes to None: the
            # block says "no list was captured", and completeness must abstain,
            # not pass "all 0 declared tensors present".
            declared_fqns = tuple(block.declared_fqns) or None
            num_experts = block.num_experts
            num_moe_layers = block.num_moe_layers
            expected_expert_bytes = block.expected_expert_bytes
        else:
            # Pre-DeclaredCheckpoint adapters carry flat attributes. Absent
            # attributes must become None here — defaulting to an empty tuple
            # would hand the completeness gate a zero-length denominator it
            # auto-satisfies.
            flat_fqns = getattr(manifest, "declared_fqns", None)
            declared_fqns = None if flat_fqns is None else (tuple(flat_fqns) or None)
            num_experts = getattr(manifest, "num_experts", None)
            num_moe_layers = getattr(manifest, "num_moe_layers", None)
            expected_expert_bytes = getattr(manifest, "expected_expert_bytes", None)

        expert_storage_bytes = getattr(meta, "expert_storage_bytes", None)
        if expert_storage_bytes is None:
            # Recognized layouts only: an unrecognized expert-ish name must not
            # pollute the physical sum any more than it would the implied one.
            expert_tensors = [
                t for t in tensors if _is_real_tensor(t) and _matches_expert_family(t.fqn)
            ]
            if expert_tensors:
                physical, storage_complete = _distinct_storage_bytes(expert_tensors)
                # A None sum means some expert dtype is outside the price table:
                # the physical figure stays unset so the byte gate reaches its own
                # unpriceable-dtype refusal with the FQNs attached, instead of
                # quietly reverting to an implied sum it could not compute either.
                if storage_complete and physical is not None:
                    expert_storage_bytes = physical
        return cls(
            tensors=tensors,
            declared_fqns=declared_fqns,
            num_experts=num_experts,
            num_moe_layers=num_moe_layers,
            expected_expert_bytes=expected_expert_bytes,
            origin=os.fspath(path),
            expert_storage_bytes=expert_storage_bytes,
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
            expert_storage_bytes=getattr(ctx, "expert_storage_bytes", None),
        )
    raise TypeError(
        f"checkpoint gates need a CheckpointGateContext or path, got {type(ctx).__name__}"
    )


def _is_real_tensor(t: TensorMeta) -> bool:
    """True for actual parameter/buffer tensors — never for metadata byte blobs."""
    return t.kind == "tensor" and "_extra_state" not in t.fqn


def _checked_num_experts(c: CheckpointGateContext) -> tuple[CheckpointGateContext, str | None]:
    """Type the one denominator the gates used to launder through ``== 0``.

    ``CheckpointGateContext.num_experts`` is annotated ``int | None``, and every
    producer flowing through ``DeclaredCheckpoint`` validation honors that — but
    contexts also arrive via :func:`_coerce`'s duck-typed ``getattr`` and via
    flat-attribute manifest adapters, where nothing types the value. Python's
    equalities then launder wrong types into real declarations: ``False == 0``,
    ``0.0 == 0`` and ``0j == 0`` are ALL True, so each used to buy the
    dense-model SKIP — and with it FirstSaveGate's inapplicable-denominator
    shrink — without any dense declaration ever being stated (a YAML
    ``moe: false`` flattened into the field, an integrator's ``bool(...)`` glue).

    Returns ``(context, None)`` when the field is a genuine non-negative int or
    truly absent (None). Otherwise the value normalizes to None on the returned
    context copy and the second element names the offense; the gates block
    VACUOUS on that reason before any classification, because a denominator the
    gate cannot name is no denominator at all — and doctrine (4) bills malformed
    exactly where it bills missing.
    """
    value = c.num_experts
    if value is None:
        return c, None
    # bool before int: isinstance(True, int) is True, and a boolean is never a
    # count however eagerly it compares like one.
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return replace(c, num_experts=None), (
            f"the declared expert count is {value!r} ({type(value).__name__}), "
            f"not a genuine non-negative integer. Only a real int may speak "
            f"here: 0 is a positive dense declaration and makes the expert "
            f"properties inapplicable, so letting a bool/float/complex "
            f"look-alike equal to 0 buy that shrink re-mints the founding "
            f"all([]) defect through a type check the manifest layer performs "
            f"and this layer skipped"
        )
    return c, None


def _expert_named(fqn: str) -> bool:
    """Cheap broad net: does the FQN carry an ``expert``/``experts`` path segment?

    Segment-exact on purpose: a substring test would pull in names that merely
    mention experts, like Gemma's ``router.per_expert_scale`` (a per-expert SCALING
    vector, not an expert weight), and classifying those as expert failures would
    accuse clean artifacts. Weird hyphenated names like ``fooexperts.bar`` are
    caught by :func:`_matches_expert_family` below instead.
    """
    return any(seg in {"expert", "experts"} for seg in fqn.lower().split("."))


def _matches_expert_family(fqn: str) -> bool:
    """True when the name is an expert weight in any layout this module can verify."""
    return bool(
        _EXPERT_WEIGHT_RE.match(fqn)
        or _PER_EXPERT_MEMBER_RE.match(fqn)
        or _STACKED_WEIGHT_RE.match(fqn)
    )


# PUBLIC, and deliberately so. The producer of a run's declaration has to sort
# tensor names into dense-vs-expert with the SAME vocabulary this module
# adjudicates them by; a second copy of the three regexes above, living in the
# training entry point, is #150's exact shape — a writer and a reader each
# internally consistent, never introduced, drifting on the first model whose
# layout only one of them learned about. Exported as names rather than
# re-implemented so that drift is impossible rather than merely unlikely.
#
# The two answer DIFFERENT questions and the difference is load-bearing:
#   mentions_expert       — is this name expert-related AT ALL? The broad test.
#                           Used to decide whether a model may be declared dense,
#                           where the fail-closed answer is "any mention at all
#                           means do not declare dense".
#   matches_expert_family — is this an expert WEIGHT in a layout this module can
#                           actually verify? The narrow test. Used to price
#                           expected bytes, where counting a name we cannot
#                           verify would inflate a denominator.
mentions_expert = _expert_named
matches_expert_family = _matches_expert_family


def _expert_weights(ctx: CheckpointGateContext) -> list[TensorMeta]:
    """Recognized-layout expert weights — the set byte accounting may safely price."""
    return [t for t in ctx.tensors if _is_real_tensor(t) and _matches_expert_family(t.fqn)]


def _expert_weight_candidates(tensors: Sequence[TensorMeta]) -> Sequence[TensorMeta]:
    """All real tensors that look expert-related, recognized layout or not.

    Selecting wider than :func:`_expert_weights` is the point: the unrecognized
    tail must surface as a named, blocking UNKNOWN-layout verdict instead of an
    invisible zero selection.
    """
    return [
        t
        for t in tensors
        if _is_real_tensor(t) and (_expert_named(t.fqn) or _matches_expert_family(t.fqn))
    ]


def _distinct_storage_bytes(experts: list[TensorMeta]) -> tuple[int | None, bool]:
    """Sum bytes over *distinct physical storages*, not over FQNs.

    One ``implied_nbytes`` is counted per distinct storage identity, so 128
    right-shaped FQNs aliased to one storage price that storage once — the
    count-correct variant of the incident, which per-FQN summation cannot see.

    Returns ``(physical_bytes, complete)``. The byte figure is ``None`` when any
    tensor's dtype is outside the price table: pricing only the recognizable
    storages would state a total over a self-chosen subset and read it as the
    whole checkpoint, which is the unrecognized-layout failure wearing
    arithmetic. ``complete`` is True only when every tensor carried a storage
    identity; tensors without one fall back to their FQN as the key (unique by
    construction), which prices each unidentifiable name as its own storage and
    marks the result incomplete — honest about being per-FQN accounting under
    another name.
    """
    seen: dict[str, int] = {}
    complete = True
    unpriceable = False
    for t in experts:
        nbytes = t.implied_nbytes
        if nbytes is None:
            # Poison the figure, not the loop: the caller must learn that this
            # total cannot be stated, never receive a partial sum wearing the
            # shape of a full one.
            unpriceable = True
            continue
        if t.storage_id is None:
            complete = False
            key = t.fqn
        else:
            key = t.storage_id
        if key not in seen:
            seen[key] = nbytes
    if unpriceable:
        return None, complete
    return sum(seen.values()), complete


def _split_expert_layouts(
    candidates: Sequence[TensorMeta],
) -> tuple[dict[str, list[TensorMeta]], list[TensorMeta], list[TensorMeta]]:
    """Classify expert-named tensors into shard groups, stacked tensors, and the tail.

    * Per-expert shard groups: Megatron local-name shards (``...linear_fc1.weight7``)
      AND the Mixtral/Qwen/Megatron-global form ``...experts.<i>.<proj>.weight``.
      The group key is the shared stem with the index slot marked ``<i>``, so every
      stem's members can be counted against the declared expert count and checked
      for within-stem storage aliasing, exactly as the incident detector did.
    * Stacked: one tensor holds the whole layer's experts on dim 0 (HF
      ``...experts.gate_up_proj`` / Megatron's fused ``...linear_fc[12].weight``).
      One FQN, one storage span: per-expert identity inside it is invisible here.
    * Unknown: expert-named but matching no family. Returned separately so callers
      fail closed on it; an unrecognized MoE layout is not a dense model.
    """
    shard_groups: dict[str, list[TensorMeta]] = defaultdict(list)
    stacked: list[TensorMeta] = []
    unknown: list[TensorMeta] = []
    for t in candidates:
        m = _SHARD_SUFFIX_RE.match(t.fqn)
        if m:
            shard_groups[f"{m.group('stem')}<i>"].append(t)
            continue
        m = _PER_EXPERT_MEMBER_RE.match(t.fqn)
        if m:
            shard_groups[f"{m.group('prefix')}<i>.{m.group('suffix')}"].append(t)
            continue
        if _EXPERT_WEIGHT_RE.match(t.fqn) or _STACKED_WEIGHT_RE.match(t.fqn):
            stacked.append(t)
            continue
        unknown.append(t)
    return dict(shard_groups), stacked, unknown


def _layer_normalized_stem(fqn: str) -> str:
    """Blank per-layer index segments so a projection's cross-layer siblings group.

    ``model.language_model.layers.3.experts.down_proj`` and ``...layers.7...`` map
    to one stem; indices embedded INSIDE a segment (``fc1``, ``w2``) stay put.
    """
    return ".".join("{}" if seg.isdigit() else seg for seg in fqn.split("."))


def _stacked_layout_problems(
    stacked: Sequence[TensorMeta],
    num_experts: int | None,
) -> tuple[list[str], list[str]]:
    """Everything metadata can still check on a stacked MoE layout.

    Three signatures, each a stacked-layout form of the incident:
      * a leading dim that is not the declared expert count — only a fraction of
        the layer's experts actually made it to storage;
      * one projection pricing a fraction of its siblings across layers (1/E of
        the bytes) — the same defect seen bottom-up, no declared denominator
        required;
      * two stacked tensors in different layers sharing one storage span — the
        ONLY aliasing signature that survives stacking, since each tensor is one
        span by construction.

    ``num_experts`` distinguishes UNKNOWN from DECLARED-DENSE: ``None`` skips the
    leading-dim comparison (no denominator exists to compare against), while an
    explicit ``0`` is a dense declaration that any stacked expert tensor
    contradicts, and must fire. Returns ``(problems, offender_fqns)``. Anything
    NOT caught here is not "fine"; per-expert slice identity inside a stacked
    tensor is simply unobservable from metadata, and the caller's verdict
    wording owns that distinction — it must abstain from claiming distinctness,
    never upgrade this list's quietness into proof of it.
    """
    problems: list[str] = []
    offenders: list[str] = []

    for t in stacked:
        if num_experts is not None and (not t.shape or t.shape[0] != num_experts):
            problems.append(
                f"{t.fqn}: stacked expert tensor's leading dim "
                f"{t.shape[0] if t.shape else '?'} != declared experts {num_experts} — "
                "the stacked form of the local-name save: only a fraction of the "
                "layer's experts was stored"
            )
            offenders.append(t.fqn)

    # Sibling projections across layers must agree in shape and dtype: layer 3's
    # down_proj at 1/E of the bytes of every other layer's down_proj is the
    # local-name save signature without any declared denominator.
    siblings: dict[str, list[TensorMeta]] = defaultdict(list)
    for t in stacked:
        siblings[_layer_normalized_stem(t.fqn)].append(t)
    for stem, members in sorted(siblings.items()):
        shapes: dict[tuple[tuple[int, ...], str], list[TensorMeta]] = defaultdict(list)
        for t in members:
            shapes[(t.shape, t.dtype)].append(t)
        if len(shapes) <= 1:
            continue
        # On ties prefer the larger leading dim as the norm, so an underfilled
        # minority tensor can never become the reference its siblings are accused of.
        consistent = max(
            shapes.values(),
            key=lambda ts: (len(ts), ts[0].shape[0] if ts[0].shape else 0),
        )
        norm_shape = consistent[0].shape
        norm_dtype = consistent[0].dtype
        norm_bytes = consistent[0].implied_nbytes
        for (shape, _dtype), ts in shapes.items():
            if ts is consistent:
                continue
            for t in ts:
                this_bytes = t.implied_nbytes
                if this_bytes is not None and norm_bytes is not None:
                    evidence_clause = (
                        f"shape {shape} implies {this_bytes:,} bytes, but "
                        f"{len(consistent)} other layer(s) of {stem} are "
                        f"{norm_shape} {norm_dtype} ({norm_bytes:,} bytes, ratio "
                        f"{this_bytes / norm_bytes:.3f})"
                    )
                else:
                    # The shape disagreement is the defect; the byte ratio is
                    # exhibit formatting. A guessed price for an unrecognized
                    # dtype would fabricate the exhibit, so the message names
                    # the pricing failure and lets the shape disagreement stand
                    # on its own.
                    evidence_clause = (
                        f"shape {shape} disagrees with {len(consistent)} other "
                        f"layer(s) of {stem} at {norm_shape} {norm_dtype}; the "
                        "byte ratio cannot be stated because an unrecognized "
                        "dtype makes at least one sibling unpriceable"
                    )
                problems.append(
                    f"{t.fqn}: {evidence_clause} — a stacked tensor holding "
                    "a fraction of its sibling projections is the stacked-layout "
                    "form of the 16-of-128 save"
                )
                offenders.append(t.fqn)

    # Whole-tensor aliasing: two FQNs priced over one span. Tensors without storage
    # identity cannot be compared at all; the abstention wording records that gap.
    spans: dict[str, list[TensorMeta]] = defaultdict(list)
    for t in stacked:
        if t.storage_id is not None:
            spans[t.storage_id].append(t)
    for span_members in spans.values():
        if len(span_members) > 1:
            fqns = sorted(t.fqn for t in span_members)
            problems.append(
                f"{len(fqns)} stacked expert tensors in different layers share one "
                f"storage span ({fqns[0]} .. {fqns[-1]}): whole-tensor aliasing — "
                "the one aliasing signature that survives stacking"
            )
            offenders.extend(fqns[:4])
    return problems, offenders


# The full per-MoE-layer projection sets of every expert naming family this
# module knows how to verify. This table is where "how many weights per layer"
# may legitimately come from, and its provenance is the entire point:
#
# * Not from the manifest — none declares it. CheckpointGateContext carries
#   num_experts, num_moe_layers, expected_expert_bytes and declared_fqns; no
#   projection-count field exists for from_path to populate.
# * Not from a universal constant. The previous `weights_per_layer = 2` priced
#   Megatron's fc1+fc2 for EVERY family, so a cleanly saved 24-tensor Mixtral
#   checkpoint (w1/w2/w3 per expert per layer) reported 24/16 — a real
#   examination whose stated denominator refuted it, which is precisely the
#   shape Verdict.OVERCOVERED now blocks. An unwarranted constant standing in
#   for a fact the artifact never stated is the defect this file already
#   removed from implied_nbytes; the count deserved the same removal.
# * Never from the OBSERVED stems. A denominator derived from the numerator is
#   circular in the direction that kills it: under uniform shrinkage (one
#   projection never written on ANY layer) a stem-count-derived expected
#   equals checked by construction and can only ratify the gutting, while in
#   the cases where it could disagree the per-stem count check has already
#   fired with a more precise message. A number that cannot fail on its own is
#   decoration, and doctrine (5) grades decoration as a defect.
#
# The tables instead restate, as data, what the selector asserts when it
# classifies a name: "this is Mixtral-family naming" already means "one MoE
# layer of it is w1, w2 and w3 per expert". Matching a saved population
# AGAINST the closed set (subset, never equality-from-counts) yields a width
# that can and does disagree with what was saved.
_PER_EXPERT_PROJECTION_FAMILIES: tuple[frozenset[str], ...] = (
    # Megatron per-expert weights — local suffix (``…linear_fc1.weight<i>``,
    # the incident) and global spelling (``…experts.<i>.linear_fc1.weight``).
    frozenset({"linear_fc1.weight", "linear_fc2.weight"}),
    # Mixtral: ``…block_sparse_moe.experts.<i>.w{1,2,3}.weight``.
    frozenset({"w1.weight", "w2.weight", "w3.weight"}),
    # Qwen-MoE: ``…mlp.experts.<i>.{gate,up,down}_proj.weight``.
    frozenset({"gate_proj.weight", "up_proj.weight", "down_proj.weight"}),
)

# Stacked families are keyed on the token after the FINAL ``.experts.``
# segment, with one trailing ``.weight`` removed, so Gemma-4's bare spelling
# (``...experts.down_proj``) and GPT-OSS's suffixed spelling
# (``...experts.gate_up_proj.weight``) denote the same membership; Megatron's
# fused ``...experts.experts.linear_fc1.weight`` collapses onto its last token
# by the same cut. Biases are NOT normalised in: a ``gate_up_proj.bias`` token
# matches no entry below, so a checkpoint whose projections carry biases
# abstains from the aggregate count rather than being priced against
# weight-only tables it half-matches.
_STACKED_PROJECTION_FAMILIES: tuple[frozenset[str], ...] = (
    frozenset({"linear_fc1", "linear_fc2"}),  # Megatron fused
    frozenset({"gate_up_proj", "down_proj"}),  # Gemma-4 / GPT-OSS
    # Split-projection HF MoE (Qwen3-style stacked). No fixture exercises it;
    # the entry exists so an artifact that DOES spell it gets a true
    # denominator instead of a Gemma-shaped 2.
    frozenset({"gate_proj", "up_proj", "down_proj"}),
)


def _shard_projection_token(group_key: str) -> str | None:
    """The projection one per-expert shard STEM names, e.g. ``w1.weight``.

    Group keys come from :func:`_split_expert_layouts` in exactly two shapes:
    per-expert members as ``<prefix><i>.<suffix>`` (the suffix IS the token)
    and Megatron local shards as ``<stem…linear_fcN.weight><i>``. A key in any
    other shape is a shape this module did not emit; returning None lets the
    caller's denominator go absent rather than be built on a guessed token.
    """
    marker = "<i>."
    if marker in group_key:
        return group_key.rsplit(marker, 1)[1]
    if group_key.endswith("<i>"):
        match = re.search(r"linear_fc[12]\.weight$", group_key[: -len("<i>")])
        return match.group(0) if match else None
    return None


def _stacked_projection_token(fqn: str) -> str | None:
    """The projection one stacked tensor names, normalised per the table above.

    Every family regex that sorts a name into the stacked list contains the
    ``experts`` substring, but only the dotted segment is a safe anchor (the
    classifier's own docs warn about spellings like ``fooexperts.bar``); a
    name with no ``.experts.`` segment returns None and poisons the total.
    """
    if ".experts." not in fqn:
        return None
    return fqn.rsplit(".experts.", 1)[1].removesuffix(".weight")


def _family_layer_width(
    tokens: set[str],
    families: tuple[frozenset[str], ...],
) -> int | None:
    """Per-layer weight count of the ONE family whose full set contains ``tokens``.

    Subset semantics are the anti-circularity device: the saved population is
    matched against the family's declared set, so fc1-alone resolves to
    ``{linear_fc1, linear_fc2}`` → width 2 and a checkpoint that never wrote
    fc2 still carries fc2 in its expectation, registering as UNDERCOVERED
    instead of redefining the denominator to fit the artifact. Zero containing
    families means the stems name structure no table defines (a novel
    projection, a bias spelling); MORE than one means this population cannot
    distinguish the families (a lone ``down_proj`` is both a complete Gemma
    layer and a gutted split-projection one). Both return None: an unqualified
    count is a stated abstention; a guessed one is a fabricated fact.
    """
    if not tokens:
        return None
    containing = [family for family in families if tokens <= family]
    if len(containing) != 1:
        return None
    return len(containing[0])


def _declared_tensor_count(
    ctx: CheckpointGateContext,
    *,
    shard_groups: Mapping[str, Sequence[TensorMeta]],
    stacked: Sequence[TensorMeta],
) -> int | None:
    """How many expert-weight tensors the checkpoint should hold, if knowable.

    The three multiplicands now have three separate, defensible provenances:
    ``num_experts`` and ``num_moe_layers`` from the manifest, weights-per-layer
    from the closed naming-family tables above. Two of them stay absent
    (``None``) rather than guessed:

    * Zero or undeclared MoE layers DECLARE NO EXPERT POPULATION: the
      denominator is absent, not zero. Returning 0 rendered real coverage as
      ``N/0 expert tensors`` — an unqualified count wearing a denominator's
      clothes — and doctrine (2) does not grade that on severity. (The old
      docstring's aside about "the non-blocking direction" is deleted with
      the constant: that direction no longer exists, and a comment asserting
      it would be doctrine (5) in prose.)
    * A stem population resolving to no single family has no truthfully
      computable total; the examined count stands unqualified.

    What this number is FOR, stated plainly so its absence is never read as
    lost verification: on the ok() path every observed stem already provably
    holds ``num_experts`` members (otherwise the per-stem count check has
    failed the gate with the stem named), so the member-level count is fully
    redundant upstream of here. The one defect class only this aggregate can
    see is an ENTIRELY ABSENT (layer, projection) stem — nothing in the loop
    over observed stems can indict a stem that is not in its data. The family
    tables preserve precisely that increment and nothing more.

    When both families coexist (reachable only from the byte gate — the
    distinctness gate refuses the artifact as MIXED upstream), the shard
    family's table prices the total: a mixed artifact's blocking verdicts have
    already been decided by the gates that own them, and this count is the
    byte gate's audit trail for what was priced, never its real denominator —
    that remains ``expected_expert_bytes``, manifest-stated and untouched.
    """
    if not ctx.num_moe_layers:
        return None
    if shard_groups:
        # Keep falsiness here deliberately: with the gates' count checks now
        # `is not None`, a declared-0 manifest beside real shards fails at the
        # contradiction before this helper's value can dress any verdict.
        if not ctx.num_experts:
            return None
        tokens: list[str] = []
        for group_key in shard_groups:
            token = _shard_projection_token(group_key)
            if token is None:
                return None
            tokens.append(token)
        width = _family_layer_width(set(tokens), _PER_EXPERT_PROJECTION_FAMILIES)
        if width is None:
            return None
        return ctx.num_experts * ctx.num_moe_layers * width
    tokens = []
    for tensor in stacked:
        token = _stacked_projection_token(tensor.fqn)
        if token is None:
            return None
        tokens.append(token)
    width = _family_layer_width(set(tokens), _STACKED_PROJECTION_FAMILIES)
    if width is None:
        return None
    return ctx.num_moe_layers * width


@register
class ExpertDistinctnessGate(Gate):
    """Catches expert-count and expert-identity save corruption from metadata alone.

    PER-EXPERT layouts — one FQN per expert, each with its own storage span:
    Megatron local-name shards (``...linear_fc1.weight0`` .. ``weight15``, the
    incident), Megatron global names (``...experts.42.linear_fc1.weight``), Mixtral
    (``...block_sparse_moe.experts.0.w1.weight``), Qwen-MoE
    (``...mlp.experts.0.gate_proj.weight``). This is the layout the gate was built
    for, and its two incident signatures are checked there exactly as they always
    were:

    1. *Count*: a per-expert layout stores one tensor per expert. 16 on disk against
       128 declared is not "checkpoint format variation"; it is 87.5% of the experts
       missing, overwritten in place as each rank wrote its local ``weight0..15``.
    2. *Aliasing*: distinct FQNs sharing one ``storage_id`` are one tensor. If the
       count ever looks right again (a more subtle version of the same bug), this is
       the check that still fires.

    STACKED layouts — every expert of a layer inside ONE tensor on dim 0, hence ONE
    storage span: HF's dominant MoE form (``...experts.down_proj`` on Gemma-4,
    ``...mlp.experts.gate_up_proj.weight`` on GPT-OSS), and Megatron's fused
    ``...linear_fc[12].weight``, which is the same layout in older spelling. The
    premise "N expert FQNs map to N storage spans" is void here. What metadata CAN
    still see, this gate still checks and still fails on (see
    :func:`_stacked_layout_problems`): a leading dim that is not the declared
    count, a tensor pricing a fraction of its sibling projections across layers,
    and two stacked tensors in different layers on one storage span.

    What metadata can NEVER see under a stacked layout is the incident itself: E
    duplicated expert-slices occupy exactly the storage span of E distinct ones.
    So the no-defect stacked outcome is an explicit SKIP, never a PASS. Justification
    against doctrine point (5): a claim broader than its evidence is a defect even
    when the code is correct, and an honest, specific abstention is a first-class
    correct outcome strictly better than a check that pretends. The gate examined
    everything metadata can show on every stacked tensor, found nothing, and still
    does not know whether the experts are distinct — so it says precisely that,
    naming the layout, the tensor count, the claimed expert count, and the evidence
    that WOULD settle it (a data-level per-slice hash comparison, which is a tensor
    read this gate deliberately never performs). A FAIL would be false here (nothing
    wrong was found); a PASS would pretend (distinctness is unproven). The composite
    first-save gate counts only verified properties, so on metadata alone a stacked
    checkpoint's distinctness is reported "not established" there — blocked for a
    true reason instead of silently credited.

    UNKNOWN layout: expert-named tensors matching no family above BLOCK. An
    unrecognised MoE layout is not a dense model, and fail-closed beats guessing.

    MIXED layout: per-expert shards and stacked tensors coexisting in one
    checkpoint block as their own named verdict. Neither family's denominator is
    defined over the mixture, and layout metadata cannot tell a legitimately
    heterogeneous model (dense-MoE blocks beside stacked-MoE blocks) from a
    half-converted checkpoint in which one family is a leftover — so the gate
    refuses rather than folding unexamined tensors into any verified count.

    The empty-expert-set control is load-bearing: on a declared-MoE model, finding
    zero expert tensors under ANY recognized layout must be VACUOUS, because "no
    mismatches found" is what ``all([])`` reported on the corrupt artifact for
    months.

    Its stricter twin is the no-manifest door: ``num_experts is None`` means the
    context declared *nothing* (exactly what ``from_path`` builds beside a
    manifestless checkpoint), and a gutted MoE artifact — experts stripped or
    never written — reaches it byte-for-byte identical to a true dense model.
    So ``None`` blocks as VACUOUS through the enforced-ok path, and only an
    explicit ``0`` earns the dense-model SKIP. None means UNKNOWN, 0 means
    dense: doctrine (4) is precisely that a missing denominator blocks rather
    than abstains politely with a fabricated reason.
    """

    id: ClassVar[str] = "checkpoint.expert_distinctness"
    description: ClassVar[str] = (
        "Expert tensors exist, are present at the declared count, and occupy "
        "distinct storage — the 128-experts-aliased-to-16 incident"
    )
    events: ClassVar[tuple[Lifecycle, ...]] = (Lifecycle.FIRST_SAVE, Lifecycle.SAVE)
    context_type: ClassVar[type | None] = CheckpointGateContext

    def coerce_context(self, ctx: Any) -> CheckpointGateContext | None:
        """The module adapter (:func:`_coerce`) promoted to the dispatch contract.

        Paths and ``.tensors``-shaped objects still adapt, exactly as they did when
        the coercion lived only at the top of :meth:`check`; anything this gate
        does not recognise returns ``None`` so a typed sweep reports it unwired
        instead of dying on a TypeError one frame down.
        """
        try:
            return _coerce(ctx)
        except TypeError:
            return None

    def check(self, ctx: Any) -> GateResult:
        c = _coerce(ctx)
        declared_num_experts = c.num_experts
        c, malformed = _checked_num_experts(c)
        if malformed is not None:
            # Block BEFORE the empty-set doors below: a malformed count must
            # never reach the `== 0` dense door, and never read as the None
            # (UNKNOWN) door either — its own named reason blocks VACUOUS, with
            # the raw value kept as evidence.
            return self.ok(
                malformed,
                Coverage.none("expert tensors"),
                evidence={
                    "declared_num_experts_raw": repr(declared_num_experts),
                    "origin": c.origin,
                },
            )
        candidates = _expert_weight_candidates(c.tensors)
        shard_groups, stacked, unknown = _split_expert_layouts(candidates)

        if not candidates:
            if c.num_experts is None:
                # None is not 0. This is exactly the context from_path builds
                # when NO MANIFEST EXISTS, so nothing anywhere declares the
                # model dense: a gutted MoE checkpoint (experts stripped or
                # never written) arrives at this branch looking identical to a
                # true dense model, and the old falsy test answered both with
                # the dense-model SKIP and prose claiming "context declares no
                # experts" — a declaration the context never made. Missing
                # denominator BLOCKS: ok() over zero coverage takes the
                # framework's enforced-VACUOUS path, the same one the
                # completeness gate uses for a missing manifest, and doctrine
                # (1) is honoured because Coverage.none names the 0 examined.
                return self.ok(
                    "run manifest does not declare an expert count and the "
                    "checkpoint contains no expert tensors, so zero expert "
                    "tensors were examined against an unknown declaration — "
                    "'none found' is indistinguishable from 'none saved', and "
                    "establishes nothing",
                    Coverage.none("expert tensors"),
                    evidence={"origin": c.origin},
                )
            if c.num_experts == 0:
                # An explicit 0 is the ONLY ground on which distinctness is
                # INAPPLICABLE rather than unestablished: a positive declaration
                # of a dense model, distinguishable from the None (UNKNOWN) door
                # directly above, which can never reach this branch. The kind
                # travels as data so composites may remove this gate from the
                # applicable denominator without parsing this string.
                return self.skip(
                    "context declares no experts and none are present (dense model)",
                    kind=AbstentionKind.NOT_APPLICABLE,
                )
            # A MoE model whose checkpoint has zero expert tensors has not "passed an
            # identity check on every expert". ok() downgrades this to VACUOUS, which
            # is the exact difference between this gate and the audit tool it replaces.
            return self.ok(
                f"model declares {c.num_experts} experts but the checkpoint contains "
                f"no expert tensors",
                Coverage.none("expert tensors"),
                evidence={"origin": c.origin},
            )

        examined = len(stacked) + sum(len(members) for members in shard_groups.values())

        problems: list[str] = []
        offenders: list[str] = []

        # Per-expert shard groups — the incident family, checked exactly as it always
        # was: declared-count per stem, then storage aliasing within the stem.
        # Per-expert shard groups — the incident family, checked exactly as it always
        # was: declared-count per stem, then storage aliasing within the stem.
        for stem, members in sorted(shard_groups.items()):
            # `is not None`, deliberately not truthiness: an explicit 0 is a
            # DECLARED dense model, and expert shards physically present beside
            # that declaration contradict it — a mismatch the falsy version
            # folded into "no count was declared" and skipped.
            if c.num_experts is not None and len(members) != c.num_experts:
                problems.append(
                    f"{stem}: {len(members)} expert shards on disk, config declares "
                    f"{c.num_experts} — the local-name save signature (16 of 128)"
                )
                offenders.extend(t.fqn for t in members[:4])
            # storage_id unknown -> fall back to the FQN, which is unique by
            # construction; aliasing is then undetectable from this metadata, and
            # the PASS wording below must not claim a distinctness it never
            # examined — the storage_identity flag governs that. The byte-volume
            # gate prices physical storage whenever identity exists.
            storages = {t.storage_id if t.storage_id is not None else t.fqn for t in members}
            if len(storages) < len(members):
                problems.append(
                    f"{stem}: {len(members)} expert FQNs share {len(storages)} "
                    f"distinct storages — experts are aliased to the same bytes"
                )
                offenders.extend(t.fqn for t in members[:4])

        storage_identity = bool(shard_groups) and all(
            t.storage_id is not None for members in shard_groups.values() for t in members
        )

        stacked_problems, stacked_offenders = _stacked_layout_problems(stacked, c.num_experts)
        problems.extend(stacked_problems)
        offenders.extend(stacked_offenders)

        if shard_groups and stacked:
            # Both families are non-empty, so each per-family check above ran over
            # its own subset under a full-sounding denominator: the composite of
            # two partial verifications is not a verification of the checkpoint as
            # a whole. The mixture is a defect of evidence, not of values — block
            # on it directly, with the per-family findings kept as evidence.
            shard_count = sum(len(members) for members in shard_groups.values())
            # An unrecognized tail is reachable here — MIXED is decided before the
            # UNKNOWN refusal below, so without this clause the detail would
            # describe a two-family checkpoint while a third population sat in
            # the evidence dict, unmentioned in the sentence an operator reads.
            unknown_note = (
                ""
                if not unknown
                else (
                    f" A further {len(unknown)} expert-named tensor(s) match no "
                    f"recognized layout at all (first: {unknown[0].fqn}) and are "
                    "outside both denominators."
                )
            )
            return self.fail(
                f"MIXED expert layout: {shard_count} per-expert shard tensor(s) in "
                f"{len(shard_groups)} group(s) coexist with {len(stacked)} stacked "
                f"expert tensor(s) (first stacked: {stacked[0].fqn}; first shard "
                f"group: {sorted(shard_groups)[0]}), and the mixture itself is the "
                "defect: metadata cannot tell a legitimately heterogeneous model "
                "(dense-MoE blocks beside stacked-MoE blocks) from a half-converted "
                "checkpoint in which one family is a leftover, and getting that "
                "wrong in either direction silently changes the denominators — "
                "unexamined tensors fold into a verified count, or leftover "
                "storages are priced and verified twice. What would settle it: an "
                "explicit declared layout in the run manifest, or a per-layer "
                "expected-layout map. Refusing to pass over the ambiguity." + unknown_note,
                Coverage(checked=examined + len(unknown), unit="expert tensors"),
                evidence={
                    "per_expert_tensor_count": shard_count,
                    "per_expert_groups": sorted(shard_groups)[:8],
                    "stacked_tensor_count": len(stacked),
                    "stacked_fqns": [t.fqn for t in stacked[:8]],
                    "unrecognized_fqns": [t.fqn for t in unknown[:8]],
                    "recognized_problems": problems[:16],
                    "would_settle": (
                        "a manifest-declared expert layout, or a per-layer expected-layout map"
                    ),
                    "origin": c.origin,
                },
            )

        if unknown:
            # Fail closed on the names themselves: an unrecognized MoE layout is
            # not a dense model. What the recognized half already established is
            # kept as evidence, not merged into the verdict.
            return self.fail(
                f"{len(unknown)} expert-named tensor(s) match no recognized MoE "
                f"layout (first: {unknown[0].fqn}); an unrecognized expert layout "
                "cannot be verified from metadata and is not a dense model — "
                "refusing to pass over it",
                Coverage(checked=examined + len(unknown), unit="expert tensors"),
                evidence={
                    "unrecognized_fqns": [t.fqn for t in unknown[:8]],
                    "recognized_problems": problems[:16],
                    "origin": c.origin,
                },
            )

        if stacked:
            # No honest single denominator exists: a pure stacked checkpoint's
            # manifest declares experts and layers, not tensor counts, and a mixed
            # layout has no one per-tensor count. checked is real (every matched
            # tensor was examined); expected stays None rather than being fabricated.
            coverage = Coverage(checked=examined, unit="expert tensors")
        elif not storage_identity:
            # The aliasing half of this gate compared unique names, not bytes.
            # The coverage record carries the degraded basis so a downstream
            # reader cannot confuse this PASS with "distinct storage verified".
            coverage = Coverage(
                checked=examined,
                unit="expert tensors",
                expected=_declared_tensor_count(c, shard_groups=shard_groups, stacked=stacked),
                sampled=True,
                sample_reason="no storage identity",
            )
        else:
            # The denominator's weights-per-layer factor resolves through the
            # naming family the stems matched — fc1/fc2 for Megatron, w1..w3
            # for Mixtral, gate/up/down for Qwen — so a healthy non-Megatron
            # checkpoint is no longer accused of exceeding a Megatron-shaped
            # expectation, and an entirely absent stem still reads as short.
            coverage = Coverage(
                checked=examined,
                unit="expert tensors",
                expected=_declared_tensor_count(c, shard_groups=shard_groups, stacked=stacked),
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
        if stacked:
            return self._stacked_abstention(c, stacked, shard_groups, storage_identity, coverage)
        # Only per-expert shards remain: the gate's original claims, unchanged,
        # with the layout fixed in prose.
        if c.num_experts and storage_identity:
            detail = (
                f"expert shard counts match the declared {c.num_experts} experts and "
                f"every shard group occupies distinct storage (sharded)"
            )
        elif c.num_experts:
            detail = (
                f"expert shard counts match the declared {c.num_experts} experts, but "
                f"storage distinctness could not be examined: the metadata carries "
                f"no storage identity for these shards, so aliasing to shared bytes "
                f"cannot be excluded (sharded)"
            )
        else:
            detail = (
                "expert shard storages are distinct within every group, but no expert "
                "count was declared — presence at the declared count could not be "
                "verified (sharded)"
            )
        return self.ok(detail, coverage)

    def _stacked_abstention(
        self,
        c: CheckpointGateContext,
        stacked: Sequence[TensorMeta],
        shard_groups: dict[str, list[TensorMeta]],
        storage_identity: bool,
        coverage: Coverage,
    ) -> GateResult:
        """The clean-stacked verdict: an explicit, fully reasoned SKIP, never a PASS.

        ``ok()`` cannot express this outcome — PASS would claim a property the gate
        cannot examine, and its UNDERCOVERED downgrade would describe the gap as
        under-sampling when it is actually physics. ``skip()`` would discard the
        coverage record of the 60-odd tensors that genuinely WERE examined. This
        goes through ``_result`` — the same mechanism the framework itself uses for
        framework-level abstentions — so the SKIP keeps the true examined count,
        with the complete reasoning carried in the detail string and evidence.
        """
        checked_claims: list[str] = []
        if c.num_experts is not None:
            checked_claims.append(
                f"all {len(stacked)} leading dims equal the declared {c.num_experts} experts"
            )
        else:
            checked_claims.append("no declared expert count, so leading dims were unchecked")
        nameless = [t for t in stacked if t.storage_id is None]
        if nameless:
            checked_claims.append(
                f"{len(nameless)} of {len(stacked)} stacked tensors carry no storage "
                "identity, so even cross-tensor span aliasing could not be fully examined"
            )
        else:
            checked_claims.append(
                f"all {len(stacked)} stacked storage spans are distinct across layers "
                "and projections"
            )
        unpriceable = [t for t in stacked if t.implied_nbytes is None]
        if unpriceable:
            # The sibling check above could not price these tensors, so the
            # pricing-consistency claim would be doctrine (5)'s defect verbatim —
            # a claim broader than its evidence, emitted by correct code. Name
            # the gap with the dtype attached instead.
            checked_claims.append(
                f"{len(unpriceable)} of {len(stacked)} stacked tensors carry a "
                f"dtype outside the price table (first: {unpriceable[0].dtype!r}), "
                "so sibling byte pricing was NOT examined — only their shapes "
                "could be compared"
            )
        else:
            checked_claims.append("sibling projections price consistently across layers")

        shard_note = ""
        if shard_groups:
            sharded_examined = sum(len(m) for m in shard_groups.values())
            basis = "count and storage" if storage_identity else "count only (no storage identity)"
            shard_note = (
                f"; the {sharded_examined} per-expert shard(s) also present were "
                f"verified by {basis}"
            )
        # UNKNOWN (None) and DECLARED-DENSE (0) are different statements; the
        # falsy version rendered a declared 0 as unknowable. Logical experts are
        # only computable when both declared fields actually exist.
        logical_experts = (
            c.num_experts * c.num_moe_layers
            if c.num_experts is not None and c.num_moe_layers is not None
            else None
        )
        return self._result(
            Verdict.SKIP,
            coverage,
            # The kind IS the verdict's meaning, one level down: the experts
            # exist (this layout is MoE by construction), and metadata cannot
            # settle their distinctness. NOT_ESTABLISHED keeps this abstention
            # charged against every composite denominator — as data, so the
            # composite never has to parse the prose below to price it.
            abstention=AbstentionKind.NOT_ESTABLISHED,
            detail=(
                f"STACKED MoE layout: {len(stacked)} expert tensor(s) each hold an "
                f"entire layer's experts on leading dim 0 (first: {stacked[0].fqn})"
                f"{shard_note}. Everything checkpoint metadata can show WAS checked — "
                + "; ".join(checked_claims)
                + " — and showed nothing wrong. Per-expert identity INSIDE a stacked "
                "tensor is unobservable from metadata by construction: N duplicated "
                "slices occupy exactly the one storage span that N distinct slices "
                "would, so this gate does not know whether the experts are distinct "
                "and ABSTAINS instead of claiming it (an honest, specific abstention "
                "is a first-class correct outcome; a PASS here would be a claim "
                "broader than its evidence). What would settle it: a data-level "
                "per-slice comparison (hash every expert slice within each stacked "
                "tensor) — a tensor read this gate deliberately never performs."
            ),
            evidence={
                "layout": "stacked",
                "stacked_tensor_count": len(stacked),
                "stacked_fqns": [t.fqn for t in stacked[:8]],
                "declared_experts": c.num_experts,
                "logical_experts_claimed": logical_experts,
                "per_expert_identity": "unobservable-from-metadata",
                "would_settle": "data-level per-slice hash of each expert slice",
                "origin": c.origin,
            },
        )

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
                "manifestless-expert-set",
                ControlKind.MUST_FIRE,
                fx.manifestless_moe_ctx,
                note="NO manifest at all: num_experts is None (UNKNOWN), not an "
                "explicit 0 (dense) — the gutted-MoE twin of the empty-expert-set "
                "trap; must VACUOUS-block, never take the dense-model SKIP",
            ),
            Control(
                "malformed-dense-count-bool",
                ControlKind.MUST_FIRE,
                lambda: CheckpointGateContext(
                    tensors=(),
                    declared_fqns=None,
                    num_experts=False,
                    num_moe_layers=None,
                    expected_expert_bytes=None,
                    origin="synthetic:malformed-dense-count-bool",
                ),
                note="num_experts=False satisfies `== 0` and used to buy the "
                "dense-model NOT_APPLICABLE SKIP — the YAML `moe: false` "
                "flattening path. 0.0 and 0j launder the same way (pinned in "
                "tests). A bool/float/complex declared count must VACUOUS-block "
                "as a malformed denominator, never shrink the first-save "
                "denominator",
            ),
            Control(
                "right-count-but-aliased",
                ControlKind.MUST_FIRE,
                fx.right_count_aliased_storage_ctx,
                note="shard count matches but FQNs share storages (count check alone is blind)",
            ),
            Control(
                "healthy-sharded",
                ControlKind.MUST_PASS,
                fx.healthy_sharded_moe_ctx,
                note="per-expert FQNs with distinct storage — the only family in "
                "which distinctness can fully verify",
            ),
            Control(
                "stacked-clean",
                ControlKind.MUST_PASS,
                fx.stacked_hf_moe_ctx,
                note="clean Gemma-style stacked layout: expect an explicit SKIP "
                "abstention — non-blocking, and never a 'distinct' claim",
                expect_skip=(
                    "per-expert identity inside a stacked tensor is "
                    "metadata-invisible by construction — N duplicated slices "
                    "occupy exactly the one storage span N distinct slices "
                    "occupy — so this gate abstains NOT_ESTABLISHED on every "
                    "stacked layout; the affirmative half of its healthy-input "
                    "proof is carried by healthy-sharded above"
                ),
            ),
            Control(
                "stacked-cross-layer-alias",
                ControlKind.MUST_FIRE,
                fx.stacked_aliased_layers_ctx,
                note="two stacked tensors in different layers on one storage span — "
                "the only aliasing metadata can still see",
            ),
            Control(
                "stacked-a-fraction-of-experts",
                ControlKind.MUST_FIRE,
                fx.stacked_underfilled_ctx,
                note="one stacked tensor at 1/8 of the declared expert count — the "
                "incident ratio in stacked clothing",
            ),
            Control(
                "unknown-expert-layout",
                ControlKind.MUST_FIRE,
                fx.unknown_expert_layout_ctx,
                note="expert-named tensors matching no layout family must block — "
                "an unrecognized MoE layout is not a dense model",
            ),
            Control(
                "mixed-expert-layout",
                ControlKind.MUST_FIRE,
                fx.mixed_expert_layout_ctx,
                note="per-expert shards beside stacked tensors: the mixture has no "
                "honest single denominator and must block as its own named verdict",
            ),
            Control(
                "healthy-fused",
                ControlKind.MUST_PASS,
                fx.healthy_fused_moe_ctx,
                note="legacy Megatron fused names are the same stacked epistemology "
                "under an older name — now expect the stated SKIP abstention",
                expect_skip=(
                    "Megatron's fused ...linear_fc[12].weight holds a whole "
                    "layer's experts in one tensor — the stacked layout under "
                    "an older name — so per-expert identity inside it is just "
                    "as metadata-invisible and the gate abstains "
                    "NOT_ESTABLISHED here too, by the same construction as "
                    "stacked-clean"
                ),
            ),
        ]


@register
class ExpertByteVolumeGate(Gate):
    """The cheap coarse net: total expert bytes vs. declared expert bytes.

    The real bug's signature was 5.71 GB on disk where 45.70 GB was correct — an
    exact 1/8 ratio, catchable in milliseconds from DCP metadata alone, without
    reading a single tensor. This gate exists so that even if every semantic check
    were bypassed, the bytes themselves disagree.

    Bytes are priced over *distinct storage*, not per FQN: 128 shape-correct FQNs
    aliased to one tensor imply the full declared volume while one physical
    tensor exists — the count-correct variant of the incident, invisible to a
    per-FQN sum. When the metadata carries no storage identity the gate falls
    back to per-FQN implied bytes and its PASS is labelled metadata-implied,
    because a claim broader than its evidence is a defect. A byte deficit of
    strictly more than 1% fails (the boundary is pinned: exactly 99% passes);
    overage is not flagged here (padding/optimizer states make it ambiguous) —
    that is distinctness' job.

    Unlike distinctness, this gate prices STACKED layouts correctly and completely:
    a stacked tensor holding N experts on its leading dim implies exactly the bytes
    the manifest should declare, so on a clean stacked checkpoint (the Gemma-4
    family) this gate reaches a real, non-vacuous PASS where the distinctness gate
    must abstain. The denominator is still never derived from the artifact itself —
    pricing a checkpoint against itself is the vacuity trap — so without a
    Unlike distinctness, this gate prices STACKED layouts correctly and completely:
    a stacked tensor holding N experts on its leading dim implies exactly the bytes
    the manifest should declare, so on a clean stacked checkpoint (the Gemma-4
    family) this gate reaches a real, non-vacuous PASS where the distinctness gate
    must abstain. The denominator is still never derived from the artifact itself —
    pricing a checkpoint against itself is the vacuity trap — so without a
    manifest-supplied ``expected_expert_bytes`` the gate abstains and names the
    missing denominator, exactly as before.

    Two further denominator failures now block instead of shading into green.
    ``num_experts is None`` over an empty expert set is UNKNOWN, not dense (the
    enforced-VACUOUS path; only an explicit ``0`` earns the SKIP), and any expert
    dtype outside the price table blocks the whole pricing path rather than being
    costed at a silent 4 bytes/element — the float8 case the old default inflated
    to 4x volume and then matched against an honest manifest.
    """

    id: ClassVar[str] = "checkpoint.expert_bytes"
    description: ClassVar[str] = (
        "Total expert tensor bytes match the declared volume (5.71 GB vs 45.70 GB "
        "was visible from metadata alone)"
    )
    events: ClassVar[tuple[Lifecycle, ...]] = (Lifecycle.FIRST_SAVE, Lifecycle.SAVE)
    context_type: ClassVar[type | None] = CheckpointGateContext

    _DEFICIT_PER_MILLE: ClassVar[int] = 10  # fail if actual < expected * 99%

    def coerce_context(self, ctx: Any) -> CheckpointGateContext | None:
        """The module adapter (:func:`_coerce`) promoted to the dispatch contract.

        Paths and ``.tensors``-shaped objects still adapt, exactly as they did when
        the coercion lived only at the top of :meth:`check`; anything this gate
        does not recognise returns ``None`` so a typed sweep reports it unwired
        instead of dying on a TypeError one frame down.
        """
        try:
            return _coerce(ctx)
        except TypeError:
            return None

    def check(self, ctx: Any) -> GateResult:
        c = _coerce(ctx)
        declared_num_experts = c.num_experts
        c, malformed = _checked_num_experts(c)
        if malformed is not None:
            # Same door as in ExpertDistinctnessGate: a malformed denominator
            # blocks here, typed and named, before any empty-set classification
            # can turn it into a dense-model SKIP.
            return self.ok(
                malformed,
                Coverage.none("expert tensors"),
                evidence={
                    "declared_num_experts_raw": repr(declared_num_experts),
                    "origin": c.origin,
                },
            )
        candidates = _expert_weight_candidates(c.tensors)
        shard_groups, stacked, unknown = _split_expert_layouts(candidates)
        experts = _expert_weights(c)

        if not candidates:
            if c.num_experts is None:
                # None is not 0: from_path yields exactly this shape when no
                # manifest exists at all. A gutted MoE checkpoint and a true
                # dense model are indistinguishable from here, and the old falsy
                # test answered both with the dense-model SKIP — a fabricated
                # reason over a missing denominator. Missing denominator BLOCKS
                # via ok() over zero coverage: the framework's enforced-VACUOUS
                # path, mirroring the declared-but-absent branch below and the
                # completeness gate's missing-manifest branch.
                return self.ok(
                    "run manifest does not declare an expert count and the "
                    "checkpoint contains no expert tensors, so zero expert "
                    "tensors were examined against an unknown declaration: "
                    "'nothing to sum' cannot be told apart from 'nothing was "
                    "ever saved', and no byte-volume claim is established",
                    Coverage.none("expert tensors"),
                    evidence={"origin": c.origin},
                )
            if c.num_experts == 0:
                # Same door as the distinctness gate: an explicit 0 is a
                # POSITIVE dense declaration; the None (UNKNOWN) door above
                # blocks as VACUOUS and never reaches here. There is no expert
                # byte volume to measure on a declared-dense run — the property
                # itself is absent, so the abstention is priced NOT_APPLICABLE.
                return self.skip(
                    "context declares no experts and none are present (dense model)",
                    kind=AbstentionKind.NOT_APPLICABLE,
                )
            return self.ok(
                f"model declares {c.num_experts} experts but checkpoint has no "
                f"expert tensors — there is nothing to sum",
                Coverage.none("expert tensors"),
            )
        if unknown:
            # Fail closed, mirroring the distinctness gate: bytes in an expert
            # naming family this gate does not understand cannot be priced against
            # the declared volume, and silently dropping them would price a
            # PARTIAL set as though it were the whole.
            return self.fail(
                f"{len(unknown)} expert-named tensor(s) match no recognized MoE "
                f"layout (first: {unknown[0].fqn}); the expert byte volume of an "
                "unrecognized layout cannot be priced honestly from metadata",
                Coverage(checked=len(experts), unit="expert tensors"),
                evidence={
                    "unrecognized_fqns": [t.fqn for t in unknown[:8]],
                    "origin": c.origin,
                },
            )

        unpriceable = [t for t in experts if t.implied_nbytes is None]
        if unpriceable:
            # An unknown dtype costed at the old silent default of 4 bytes was
            # this gate inflating a float8 expert set 4x and then matching an
            # honest manifest against the inflated figure — a fabricated
            # denominator smuggled into the numerator. Checked AFTER the layout
            # refusal so an unrecognized naming family keeps its verdict, but
            # BEFORE the manifest denominators because the failure is
            # artifact-side: even with no manifest this must block, not skip.
            return self.ok(
                f"{len(unpriceable)} of {len(experts)} expert tensor(s) carry a "
                f"dtype this gate cannot price (first: {unpriceable[0].fqn}: "
                f"{unpriceable[0].dtype!r}); guessing the element width would "
                "state a byte volume in guessed units, so none is stated",
                Coverage.none("expert tensors"),
                evidence={
                    "unpriceable_dtypes": sorted({t.dtype for t in unpriceable}),
                    "unpriceable_fqns": [t.fqn for t in unpriceable[:8]],
                    "origin": c.origin,
                },
            )

        if c.expected_expert_bytes is None:
            # Experts EXIST here (the empty-candidates doors are upstream) and
            # only the external denominator is missing: the property is
            # unestablished, never inapplicable. Naming the kind is what keeps
            # this abstention inside every composite's denominator.
            return self.skip(
                "run manifest does not declare expected expert byte volume; without "
                "the denominator a byte count is an unqualified count, not a fact",
                kind=AbstentionKind.NOT_ESTABLISHED,
            )
        if c.expected_expert_bytes <= 0:
            # A non-positive denominator is malformed, not absent: "matches
            # declared 0" would be a claim over nothing, so this takes the same
            # enforced-VACUOUS path as every other missing-denominator branch.
            return self.ok(
                "declared expert byte volume is not positive "
                f"({c.expected_expert_bytes}) — there is no denominator to "
                "measure against",
                Coverage.none("expert tensors"),
                evidence={"origin": c.origin},
            )

        # The unpriceable guard above returned on any None, so every element of
        # this comprehension is int; the filter is for mypy, not control flow.
        implied = sum(p for p in (t.implied_nbytes for t in experts) if p is not None)
        physical, storage_complete = _distinct_storage_bytes(experts)
        if c.expert_storage_bytes is not None:
            # The reader's storage map measured physical bytes directly; that
            # number outranks anything summable from per-FQN metadata.
            physical = c.expert_storage_bytes
            storage_complete = True
        # "storage_complete" and "physical is not None" are two spellings of one
        # fact, and until now they were kept in sync by hand: the guard set the
        # flag while leaving the value None, so every comparison below was
        # written against a total that might not exist, ruled out only by a flag
        # no checker could follow. Collapse them — a non-None `physical` IS
        # storage completeness — so "compare against a sum of nothing" becomes
        # unrepresentable rather than merely commented against. The None case is
        # unreachable for the set priced above (None only arises from an
        # unpriceable dtype, which returned earlier); this keeps it that way by
        # construction instead of by assertion.
        # Every branch below therefore tests ``physical is not None`` rather than
        # the flag: the test that decides the branch is the same test that makes
        # the value safe to read inside it, which no amount of flag discipline
        # can guarantee.
        if not storage_complete:
            physical = None
        del storage_complete

        coverage = Coverage(
            checked=len(experts),
            unit="expert tensors",
            # The tensor-count denominator prices the CLASSIFIED structure:
            # shard stems carry per-expert membership, stacked tensors one
            # weight per layer, weights-per-layer read from the family table
            # the classifier matched — never a Megatron-shaped constant. This
            # count is the audit trail for what was priced; the gate's real
            # denominator remains expected_expert_bytes, manifest-stated.
            expected=_declared_tensor_count(c, shard_groups=shard_groups, stacked=stacked),
        )
        expected = c.expected_expert_bytes
        measured = implied if physical is None else physical
        evidence = {
            "implied_expert_bytes": implied,
            "physical_expert_bytes": physical,
            "storage_identity": ("absent-or-partial" if physical is None else "complete"),
            "expected_expert_bytes": expected,
            "origin": c.origin,
        }
        if measured * 1000 < expected * (1000 - self._DEFICIT_PER_MILLE):
            basis = (
                "metadata-implied (no storage identity)"
                if physical is None
                else "measured over distinct storage"
            )
            return self.fail(
                f"expert byte volume {measured:,} {basis} is below the declared "
                f"{expected:,} (ratio {measured / expected:.3f}; the incident "
                f"ratio was 0.125 = 5.71/45.70 GB)",
                coverage,
                evidence={**evidence, "ratio": round(measured / expected, 4)},
            )
        if physical is not None and implied > physical:
            # Count and shape look right while storage is aliased: the FQNs price
            # N tensors but distinct storage prices fewer. Not a volume deficit —
            # say exactly what it is so it is never mistaken for one.
            return self.fail(
                f"expert FQNs imply {implied:,} bytes across {len(experts)} tensors, "
                f"but only {physical:,} bytes exist in distinct storage — the count "
                "looks right because multiple FQNs price one physical tensor",
                coverage,
                evidence=evidence,
            )
        if physical is not None and physical > implied:
            # The other direction of the same disagreement. It used to share the
            # branch above, which meant a checkpoint with MORE physical bytes
            # than its expert names account for was reported as aliasing — a
            # sentence stating the opposite of the measurement, and doctrine (5)
            # counts a wrong stated reason as a defect even when blocking was
            # the right call. The two are not the same finding: aliasing hides
            # missing weights behind shared spans, whereas a surplus means the
            # measured span carries bytes no expert FQN names (a reader storage
            # map covering non-expert tensors, padding, or an FQN set that is
            # not the whole expert population). Either way the total agreeing
            # with the manifest does not establish that the right bytes were
            # counted, so this blocks too — under its own description.
            return self.fail(
                f"expert storage measures {physical:,} bytes but the {len(experts)} "
                f"expert FQN(s) account for only {implied:,} — {physical - implied:,} "
                "byte(s) of the measured span are named by nothing, so this total "
                "cannot be read as the expert volume even though it matches the "
                "declared figure",
                coverage,
                evidence=evidence,
            )
        if physical is not None:
            return self.ok(
                f"expert byte volume {physical:,} measured over distinct storage "
                f"matches declared {expected:,}",
                coverage,
                evidence=evidence,
            )
        return self.ok(
            f"metadata-implied only (no storage identity; aliasing cannot be "
            f"excluded): expert byte volume {implied:,} matches declared {expected:,}",
            coverage,
            evidence=evidence,
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
                "right-count-aliased-storage",
                ControlKind.MUST_FIRE,
                fx.right_count_aliased_storage_ctx,
                note="counts and shapes are right, but the FQNs share storages — "
                "only physical byte pricing fires here; the count-correct variant",
            ),
            Control(
                "no-experts-at-all",
                ControlKind.MUST_FIRE,
                fx.empty_expert_set_ctx,
                note="zero experts to sum must not read as 'bytes match'",
            ),
            Control(
                "manifestless-expert-set",
                ControlKind.MUST_FIRE,
                fx.manifestless_moe_ctx,
                note="no manifest at all: num_experts None is UNKNOWN, not an "
                "explicit 0 — the empty expert set must VACUOUS-block, not take "
                "the dense-model SKIP",
            ),
            Control(
                "malformed-dense-count-bool",
                ControlKind.MUST_FIRE,
                lambda: CheckpointGateContext(
                    tensors=(),
                    declared_fqns=None,
                    # Deliberately ill-typed, and the ignore is the point: this
                    # control exists BECAUSE a float can reach num_experts at
                    # runtime (a JSON config states `0.0`, and json.load hands
                    # back a float that satisfies `== 0`). The annotation says
                    # int | None; the wire does not honour annotations. Silencing
                    # the checker here keeps the runtime door under test instead
                    # of deleting the fixture that proves the door shuts.
                    num_experts=0.0,  # type: ignore[arg-type]
                    num_moe_layers=None,
                    expected_expert_bytes=None,
                    origin="synthetic:malformed-dense-count-float",
                ),
                note="num_experts=0.0 satisfies `== 0` and used to price this "
                "gate as dense-model inapplicable. The byte twin of the "
                "distinctness control above: a look-alike of 0 is not a dense "
                "declaration and must VACUOUS-block",
            ),
            Control(
                "unpriceable-expert-dtype",
                ControlKind.MUST_FIRE,
                fx.unpriceable_dtype_ctx,
                note="float8_e4m3fn is a real safetensors dtype with no price-"
                "table entry; the old 4-byte default inflated this 4x and "
                "matched the manifest, PASSing — pricing must block and name "
                "the dtype",
            ),
            Control(
                "healthy-fused",
                ControlKind.MUST_PASS,
                fx.healthy_fused_moe_ctx,
                note="byte volume matches the manifest exactly",
            ),
            Control(
                "stacked-clean",
                ControlKind.MUST_PASS,
                fx.stacked_hf_moe_ctx,
                note="stacked layout: implied bytes == declared volume == distinct-"
                "storage bytes — a real, non-vacuous PASS where distinctness must "
                "abstain; the denominator comes from the manifest, not the artifact",
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
    context_type: ClassVar[type | None] = CheckpointGateContext

    def coerce_context(self, ctx: Any) -> CheckpointGateContext | None:
        """The module adapter (:func:`_coerce`) promoted to the dispatch contract.

        Paths and ``.tensors``-shaped objects still adapt, exactly as they did when
        the coercion lived only at the top of :meth:`check`; anything this gate
        does not recognise returns ``None`` so a typed sweep reports it unwired
        instead of dying on a TypeError one frame down.
        """
        try:
            return _coerce(ctx)
        except TypeError:
            return None

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
        if not declared:
            # An empty declaration is not a denominator: "all 0 declared tensors
            # present" is the all([]) trap in completeness clothing. Take the
            # same enforced-VACUOUS path as the missing-manifest branch above.
            return self.ok(
                "the declared tensor set is empty after excluding _extra_state "
                "metadata blobs — there is no tensor set for the checkpoint to "
                "be complete against",
                Coverage.none("declared tensors"),
                evidence={"origin": c.origin},
            )
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
                "empty-declaration",
                ControlKind.MUST_FIRE,
                lambda: CheckpointGateContext(
                    tensors=(
                        TensorMeta(
                            "model.layers.0.mlp.experts.linear_fc1.weight",
                            (8, 4, 4),
                            "bfloat16",
                        ),
                    ),
                    declared_fqns=(),
                    num_experts=None,
                    num_moe_layers=None,
                    expected_expert_bytes=None,
                    origin="synthetic:empty-declaration",
                ),
                note="a zero-length declared tensor set must block, not auto-"
                "satisfy as 'all 0 declared tensors present'",
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

    The composite denominator counts APPLICABLE properties
    ------------------------------------------------------
    A sub-gate that abstains with ``abstention=NOT_APPLICABLE`` — reachable only
    through a POSITIVE declaration (``num_experts == 0``; ``None`` is UNKNOWN and
    takes the VACUOUS door, which blocks) — is removed from the denominator,
    because a property that provably does not exist can be neither verified nor
    missing: charging the two expert sub-gates against a declared-dense run
    blocks every dense model at its first save for a defect that is not there,
    and blocks-then-gets-disabled is how verification dies in the audited
    estate. Any other SKIP (``NOT_ESTABLISHED``, or an undeclared ``None``)
    stays in the denominator: "I could not check" is not "there was nothing to
    check". The kind is read off the machine-readable field, never off the
    reason string. The two shapes the shrink must never swallow are pinned by
    controls below: a declared-MoE artifact with ZERO expert tensors produces
    VACUOUS, not SKIP (``empty-expert-first-save``), and a stacked MoE artifact
    produces NOT_ESTABLISHED and stays 2/3 UNDERCOVERED
    (``stacked-first-save``). Inapplicable gates are NAMED in the detail and
    evidence: a shrunken-denominator PASS reads "verified 1/1 applicable …;
    2 inapplicable (named)", never "verified 3/3", never bare "verified".
    """

    id: ClassVar[str] = "checkpoint.first_save"
    description: ClassVar[str] = (
        "Composite: distinctness + byte volume + completeness at the first "
        "checkpoint of a run, the cheapest place a save defect can be caught"
    )
    events: ClassVar[tuple[Lifecycle, ...]] = (Lifecycle.FIRST_SAVE,)
    context_type: ClassVar[type | None] = CheckpointGateContext
    _subgates: ClassVar[tuple[type[Gate], ...]] = (
        ExpertDistinctnessGate,
        ExpertByteVolumeGate,
        SaveCompletenessGate,
    )

    def coerce_context(self, ctx: Any) -> CheckpointGateContext | None:
        """The module adapter (:func:`_coerce`) promoted to the dispatch contract.

        Paths and ``.tensors``-shaped objects still adapt, exactly as they did when
        the coercion lived only at the top of :meth:`check`; anything this gate
        does not recognise returns ``None`` so a typed sweep reports it unwired
        instead of dying on a TypeError one frame down.
        """
        try:
            return _coerce(ctx)
        except TypeError:
            return None

    def check(self, ctx: Any) -> GateResult:
        sub = tuple(cls_().run(ctx) for cls_ in self._subgates)
        passed = [r for r in sub if r.verdict is Verdict.PASS]
        skipped = [r for r in sub if r.verdict is Verdict.SKIP]
        blocking = [r for r in sub if r.blocking]
        # Abstentions are priced off the machine-readable kind, never off the
        # reason string — prose-sniffing is the paraphrase defect this codebase
        # already deleted once. NOT_APPLICABLE leaves the denominator: the
        # property provably does not exist in this run's declared scope, and the
        # POSITIVE-declaration requirement is enforced upstream (the UNKNOWN
        # door is VACUOUS, and VACUOUS landed in `blocking` above before any of
        # this pricing runs). Everything else — NOT_ESTABLISHED, or an
        # undeclared None from an unaudited call site — stays in.
        inapplicable = [r for r in skipped if r.abstention is AbstentionKind.NOT_APPLICABLE]
        unresolved = [r for r in skipped if r.abstention is not AbstentionKind.NOT_APPLICABLE]
        applicable = len(self._subgates) - len(inapplicable)
        # Coverage still counts VERIFIED properties against the applicable
        # total. ok() enforces the rest by construction: an unverified-but-
        # applicable remainder downgrades to UNDERCOVERED, and an all-
        # inapplicable sweep has checked == 0 and downgrades to VACUOUS —
        # "every property was beside the point" is not a pass shape either.
        coverage = Coverage(len(passed), "sub-gates", expected=applicable)
        if blocking:
            return self.fail(
                "first save is defective: "
                + "; ".join(f"{r.gate_id}={r.verdict.value}" for r in blocking),
                coverage,
                evidence={r.gate_id: r.to_dict() for r in blocking},
            )
        if unresolved:
            verified_msg = ", ".join(r.gate_id for r in passed) or "none"
            missing_msg = "; ".join(f"{r.gate_id}: {r.detail}" for r in unresolved)
            na_msg = ""
            if inapplicable:
                na_msg = (
                    "; inapplicable by positive declaration (removed from the "
                    "denominator, never counted as verified): "
                    + ", ".join(r.gate_id for r in inapplicable)
                )
            return self.ok(
                f"verified {len(passed)}/{applicable} applicable first-save "
                f"properties ({verified_msg}); not established: {missing_msg}{na_msg}",
                coverage,
                evidence={
                    **{r.gate_id: r.verdict.value for r in sub},
                    "inapplicable": [r.gate_id for r in inapplicable],
                    "unresolved": [r.gate_id for r in unresolved],
                },
            )
        if inapplicable:
            # Where the denominator shrank, say so with the names attached —
            # this string must never collapse to "verified" over fewer
            # properties than the sweep declared.
            return self.ok(
                f"verified {len(passed)}/{applicable} applicable first-save "
                f"properties ({', '.join(r.gate_id for r in passed)}); "
                f"{len(inapplicable)} inapplicable by positive declaration "
                "(removed from the denominator, never counted as verified): "
                + "; ".join(f"{r.gate_id}: {r.detail}" for r in inapplicable),
                coverage,
                evidence={
                    **{r.gate_id: r.verdict.value for r in sub},
                    "inapplicable": [r.gate_id for r in inapplicable],
                },
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
                "stacked-first-save",
                ControlKind.MUST_FIRE,
                fx.stacked_hf_moe_ctx,
                note="clean stacked MoE: distinctness abstains (SKIP), so the "
                "composite can verify only 2/3 properties and is UNDERCOVERED — "
                "blocked for a true reason instead of silently green",
            ),
            Control(
                "healthy-first-save",
                ControlKind.MUST_PASS,
                fx.healthy_sharded_moe_ctx,
                note="a correct first save must not be blocked — per-expert layout, "
                "the only family in which every sub-gate can fully verify",
            ),
            Control(
                "dense-first-save",
                ControlKind.MUST_PASS,
                fx.dense_declared_ctx,
                note="declared-dense run (num_experts == 0, a POSITIVE "
                "declaration): the expert properties do not exist, so the "
                "composite must verify 1/1 APPLICABLE and name the two "
                "inapplicable sub-gates — never block, never read as 'verified "
                "3/3'. The MUST_FIRE half of the same distinction already "
                "exists: empty-expert-first-save (declared-but-absent experts "
                "stay VACUOUS-blocking) and stacked-first-save (could-not-check "
                "stays 2/3)",
            ),
        ]
