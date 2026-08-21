"""Layout-family coverage tests for the checkpoint expert gates.

Why this module exists
----------------------
The expert gates were built around the audited incident's Megatron local-name
spelling. The first probe of real production artifacts showed the selector was
match-nothing outside that family: pointed at a 48.07 GiB HF-format Gemma-4 MoE
checkpoint (1,013 tensors, 60 stacked expert weights over 30 layers), every
expert gate blocked with the FALSE stated reason "the checkpoint contains no
expert tensors". The shipped fix makes the selector layout-aware across four
outcomes — PER-EXPERT, STACKED, UNKNOWN, and MIXED (two families in one
checkpoint, which blocks on the ambiguity itself) — and these tests pin that
behaviour family by family, so no future selector edit can silently shrink (or
silently inflate) the set again.

This module supersedes an earlier draft written against a WRONG assumption:
that a clean STACKED layout could reach ``Verdict.PASS`` on distinctness. Under
the shipped semantics it cannot and must not. N duplicated expert slices inside
one stacked tensor occupy exactly the one storage span N distinct slices would,
so per-expert identity is unobservable from metadata by construction, and
doctrine (5) — a claim broader than its evidence is a defect — makes the honest
outcome an explicit SKIP with a stated reason, never a silent pass and never a
false alarm. Every stacked expectation here is re-derived from the shipped gate
code, not carried over from the draft; where the draft asserted PASS on a
stacked context, these tests assert SKIP and not-FAIL, both.

The router tripwire is armed in every fixture: ``router.per_expert_scale`` is
the real FQN from the production checkpoint, it contains the substring
``expert`` but no expert path segment, and a selector that matched substrings
instead of segments would swallow it and inflate coverage. Its absence from the
selection and its presence in the context are both asserted by name.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace

import pytest

from foundationscale.gates import checkpoint_gates as cg
from foundationscale.gates import fixtures as fx
from foundationscale.gates.checkpoint_gates import (
    CheckpointGateContext,
    ExpertByteVolumeGate,
    ExpertDistinctnessGate,
    TensorMeta,
)
from foundationscale.gates.core import REGISTRY, Verdict, verify_controls

# Measured from the real Gemma-4 26B-A4B production artifact (2 safetensors
# shards, 1,013 tensors, all bfloat16, 128 experts per layer, 30 MoE layers).
_GEMMA_LAYERS = 30
_GEMMA_EXPERTS = 128
_GEMMA_DOWN_SHAPE: tuple[int, ...] = (128, 2816, 704)
_GEMMA_GATE_UP_SHAPE: tuple[int, ...] = (128, 1408, 2816)
# 30 x (128*2816*704 + 128*1408*2816) x 2 bytes — the production figure.
_GEMMA_EXPERT_BYTES = 45_675_970_560

_PER_EXPERT_SHAPE: tuple[int, ...] = (64, 32)
_FUSED_INNER_SHAPE: tuple[int, ...] = (16, 32)
_FUSED_PARAMS: tuple[str, ...] = ("linear_fc1.weight", "linear_fc2.weight")
_EXPERT_DTYPE = "bfloat16"

_CHECKPOINT_GATE_IDS = [
    "checkpoint.expert_distinctness",
    "checkpoint.expert_bytes",
    "checkpoint.save_complete",
    "checkpoint.first_save",
]


# ---------------------------------------------------------------------------
# Context builders. Everything is metadata-only, in the fixtures.py tradition.
# ---------------------------------------------------------------------------


def _router_meta(fqn: str, num_experts: int, storage_key: str) -> TensorMeta:
    """One Gemma-4 router scaling vector.

    This name is the trap the selector must not fall into: ``per_expert_scale``
    contains the substring ``expert`` but no expert path segment, so a
    substring-matching selector would swallow the router and inflate every
    expert count by construction.
    """
    return TensorMeta(
        fqn=fqn,
        shape=(num_experts,),
        dtype=_EXPERT_DTYPE,
        storage_id=storage_key,
    )


def _ctx(
    tensors: list[TensorMeta],
    *,
    num_experts: int | None,
    num_moe_layers: int,
    expected_expert_bytes: int | None,
    origin: str,
) -> CheckpointGateContext:
    """Assemble a context declaring exactly the tensors supplied — routers included."""
    return CheckpointGateContext(
        tensors=tuple(tensors),
        declared_fqns=tuple(t.fqn for t in tensors),
        num_experts=num_experts,
        num_moe_layers=num_moe_layers,
        expected_expert_bytes=expected_expert_bytes,
        origin=origin,
    )


def _per_expert_ctx(
    *,
    family: str,
    fqn_for: Callable[[int, int, str], str],
    params: tuple[str, ...],
    num_layers: int,
    num_experts: int,
    alias_mod: int | None = None,
    expected_expert_bytes: int | None = None,
) -> CheckpointGateContext:
    """One PER-EXPERT naming family, clean or with count-correct storage aliasing.

    ``alias_mod`` collapses storage identity: expert ``i``'s tensor then names
    the storage span of expert ``i % alias_mod``, so every name is present at
    the declared count while only ``alias_mod`` byte spans exist — the
    count-correct variant of the incident, which a name/count check is blind to.

    ``expected_expert_bytes`` overrides the derived denominator, which is always
    the UN-aliased implied volume. Overriding it models the manifest that was
    written from the same broken save: it declares what is physically there, so
    the byte gate's deficit test passes and only the implied-vs-physical
    disagreement is left to catch the incident.
    """
    tensors: list[TensorMeta] = []
    expert_bytes = 0
    for layer in range(num_layers):
        for expert in range(num_experts):
            source = expert if alias_mod is None else expert % alias_mod
            for param in params:
                tensor = TensorMeta(
                    fqn=fqn_for(layer, expert, param),
                    shape=_PER_EXPERT_SHAPE,
                    dtype=_EXPERT_DTYPE,
                    storage_id=f"{family}:L{layer}:{param}:e{source}",
                )
                tensors.append(tensor)
                expert_bytes += tensor.implied_nbytes
        tensors.append(
            _router_meta(
                f"model.layers.{layer}.router.per_expert_scale",
                num_experts,
                f"{family}:L{layer}:router",
            )
        )
    return _ctx(
        tensors,
        num_experts=num_experts,
        num_moe_layers=num_layers,
        expected_expert_bytes=(
            expert_bytes if expected_expert_bytes is None else expected_expert_bytes
        ),
        origin=f"test://{family}/per-expert",
    )


def _megatron_local_ctx(
    *, num_layers: int = 2, num_experts: int = 4, alias_mod: int | None = None
) -> CheckpointGateContext:
    """Megatron local shards: ``...experts.linear_fc1.weight<i>`` — the incident family."""
    return _per_expert_ctx(
        family="megatron-local",
        fqn_for=lambda layer, expert, param: f"model.layers.{layer}.mlp.experts.{param}{expert}",
        params=_FUSED_PARAMS,
        num_layers=num_layers,
        num_experts=num_experts,
        alias_mod=alias_mod,
    )


def _megatron_global_ctx(
    *,
    num_layers: int = 2,
    num_experts: int = 4,
    alias_mod: int | None = None,
    expected_expert_bytes: int | None = None,
) -> CheckpointGateContext:
    """Megatron global names: ``...mlp.experts.<i>.linear_fc1.weight``."""
    return _per_expert_ctx(
        family="megatron-global",
        fqn_for=lambda layer, expert, param: f"model.layers.{layer}.mlp.experts.{expert}.{param}",
        params=_FUSED_PARAMS,
        num_layers=num_layers,
        num_experts=num_experts,
        alias_mod=alias_mod,
        expected_expert_bytes=expected_expert_bytes,
    )


def _mixtral_ctx(
    *, num_layers: int = 2, num_experts: int = 4, alias_mod: int | None = None
) -> CheckpointGateContext:
    """Mixtral: ``...block_sparse_moe.experts.<i>.w{1,2,3}.weight``."""
    return _per_expert_ctx(
        family="mixtral",
        fqn_for=lambda layer, expert, param: (
            f"model.layers.{layer}.block_sparse_moe.experts.{expert}.{param}.weight"
        ),
        params=("w1", "w2", "w3"),
        num_layers=num_layers,
        num_experts=num_experts,
        alias_mod=alias_mod,
    )


def _qwen_ctx(
    *, num_layers: int = 2, num_experts: int = 4, alias_mod: int | None = None
) -> CheckpointGateContext:
    """Qwen-MoE: ``...mlp.experts.<i>.{gate,up,down}_proj.weight``."""
    return _per_expert_ctx(
        family="qwen",
        fqn_for=lambda layer, expert, param: (
            f"model.layers.{layer}.mlp.experts.{expert}.{param}.weight"
        ),
        params=("gate_proj", "up_proj", "down_proj"),
        num_layers=num_layers,
        num_experts=num_experts,
        alias_mod=alias_mod,
    )


def _megatron_fused_ctx(
    *,
    num_layers: int,
    num_experts: int,
    starved: set[tuple[int, str]] | None = None,
) -> CheckpointGateContext:
    """Megatron's fused spelling of the STACKED family, optionally underfilled.

    Fused names (``...experts.experts.linear_fc1.weight``, no trailing index)
    carry every expert of the layer on dim 0 of one tensor — the identical
    epistemology as Gemma-4's HF spelling, so the gates must treat them alike:
    metadata-visible corruptions still block, clean metadata still only
    abstains. ``starved`` names (layer, parameter) pairs stored at leading dim
    1 where ``num_experts`` were declared.
    """
    starved = starved or set()
    tensors: list[TensorMeta] = []
    expert_bytes = 0
    for layer in range(num_layers):
        for param in _FUSED_PARAMS:
            leading = 1 if (layer, param) in starved else num_experts
            tensor = TensorMeta(
                fqn=f"layers.{layer}.mlp.experts.experts.{param}",
                shape=(leading, *_FUSED_INNER_SHAPE),
                dtype=_EXPERT_DTYPE,
                storage_id=f"megatron-fused:L{layer}:{param}:{leading}",
            )
            tensors.append(tensor)
            expert_bytes += tensor.implied_nbytes
        tensors.append(
            _router_meta(
                f"model.layers.{layer}.router.per_expert_scale",
                num_experts,
                f"megatron-fused:L{layer}:router",
            )
        )
    return _ctx(
        tensors,
        num_experts=num_experts,
        num_moe_layers=num_layers,
        expected_expert_bytes=expert_bytes,
        origin="test://megatron-fused",
    )


def _stacked_gemma_ctx(
    *,
    num_layers: int = _GEMMA_LAYERS,
    declared_experts: int | None = _GEMMA_EXPERTS,
    expected_expert_bytes: int | str | None = "auto",
    down_shape: tuple[int, ...] = _GEMMA_DOWN_SHAPE,
    gate_up_shape: tuple[int, ...] = _GEMMA_GATE_UP_SHAPE,
    alias_down_proj_across_layers: bool = False,
) -> CheckpointGateContext:
    """A Gemma-4 26B-A4B-shaped stacked MoE context with the real FQNs and shapes.

    ``expected_expert_bytes="auto"`` derives the declared volume from the
    tensors' implied bytes — with the standard shapes this reproduces the
    measured production figure; an explicit int models an independent, possibly
    WRONG, manifest denominator, because a byte gate that only ever passes has
    proven nothing.
    """
    tensors: list[TensorMeta] = []
    for layer in range(num_layers):
        down_storage = (
            "st-shard:experts.down_proj#aliased-across-layers"
            if alias_down_proj_across_layers
            else f"st-shard:L{layer}:experts.down_proj"
        )
        for projection, shape, storage in (
            ("down_proj", down_shape, down_storage),
            ("gate_up_proj", gate_up_shape, f"st-shard:L{layer}:experts.gate_up_proj"),
        ):
            tensors.append(
                TensorMeta(
                    fqn=f"model.language_model.layers.{layer}.experts.{projection}",
                    shape=shape,
                    dtype=_EXPERT_DTYPE,
                    storage_id=storage,
                )
            )
        tensors.append(
            _router_meta(
                f"model.language_model.layers.{layer}.router.per_expert_scale",
                down_shape[0],
                f"st-shard:L{layer}:router.per_expert_scale",
            )
        )
    implied = sum(t.implied_nbytes for t in tensors if ".experts." in t.fqn)
    declared_bytes = implied if isinstance(expected_expert_bytes, str) else expected_expert_bytes
    return _ctx(
        tensors,
        num_experts=declared_experts,
        num_moe_layers=num_layers,
        expected_expert_bytes=declared_bytes,
        origin="probe://gemma-4-26b-a4b/stacked",
    )


def _mixed_layout_ctx() -> CheckpointGateContext:
    """One Megatron per-expert group AND one Gemma-4 stacked pair in one context.

    Both populations are clean: the shards verify by count and storage, the
    stacked tensors abstain. The gate must account for BOTH in its coverage —
    arbitrate to a single layout and one population's examination would vanish
    from the record.
    """
    tensors: list[TensorMeta] = [
        TensorMeta(
            fqn=f"model.layers.0.mlp.experts.linear_fc1.weight{i}",
            shape=_PER_EXPERT_SHAPE,
            dtype=_EXPERT_DTYPE,
            storage_id=f"megatron-shard-{i}",
        )
        for i in range(2)
    ]
    tensors += [
        TensorMeta(
            fqn="model.language_model.layers.0.experts.gate_up_proj",
            shape=(2, 64, 32),
            dtype=_EXPERT_DTYPE,
            storage_id="stacked:gate_up_proj",
        ),
        TensorMeta(
            fqn="model.language_model.layers.0.experts.down_proj",
            shape=(2, 32, 64),
            dtype=_EXPERT_DTYPE,
            storage_id="stacked:down_proj",
        ),
    ]
    tensors.append(_router_meta("model.layers.0.router.per_expert_scale", 2, "mixed:router"))
    return _ctx(
        tensors,
        num_experts=2,
        num_moe_layers=1,
        expected_expert_bytes=sum(t.implied_nbytes for t in tensors if "router" not in t.fqn),
        origin="test://mixed-layouts",
    )


def _with_routers(ctx: CheckpointGateContext, *, prefix: str) -> CheckpointGateContext:
    """Append one ``router.per_expert_scale`` per MoE layer to a fixture context.

    Every fixture this module runs carries the trap name: the selector must
    ignore it, completeness must still account for it, and a change that
    started counting routers as expert tensors would trip the exact-count
    assertions against the truth (routers are present; experts exclude them).
    """
    if ctx.num_moe_layers is None:
        raise ValueError("fixture layer count unknown; routers must mirror the MoE layers")
    num_experts = ctx.num_experts or 1
    routers = [
        _router_meta(
            f"{prefix}.layers.{layer}.router.per_expert_scale",
            num_experts,
            f"router:{prefix}:{layer}:{num_experts}",
        )
        for layer in range(ctx.num_moe_layers)
    ]
    declared = list(ctx.declared_fqns or ()) + [r.fqn for r in routers]
    return CheckpointGateContext(
        tensors=(*ctx.tensors, *routers),
        declared_fqns=tuple(declared),
        num_experts=ctx.num_experts,
        num_moe_layers=ctx.num_moe_layers,
        expected_expert_bytes=ctx.expected_expert_bytes,
        origin=f"{ctx.origin}+routers",
    )


# ---------------------------------------------------------------------------
# Shared exact-count assertions for the PER-EXPERT clean and aliased cases.
# ---------------------------------------------------------------------------


def _assert_clean_per_expert(
    ctx: CheckpointGateContext,
    *,
    checked: int,
    expected: int | None,
    router_count: int,
) -> None:
    """Selection and distinctness assertions shared by the clean PER-EXPERT cases.

    Asserts in both directions on the router tripwire: it is genuinely present
    in the context (the trap is armed) and genuinely absent from the selection
    (the trap was not taken).
    """
    router_fqns = {t.fqn for t in ctx.tensors if t.fqn.endswith("router.per_expert_scale")}
    experts = cg._expert_weights(ctx)
    expert_fqns = {t.fqn for t in experts}

    assert len(router_fqns) == router_count
    assert "model.layers.0.router.per_expert_scale" in router_fqns
    assert len(experts) == checked
    assert not (router_fqns & expert_fqns)
    assert "model.layers.0.router.per_expert_scale" not in expert_fqns

    result = ExpertDistinctnessGate().run(ctx)
    assert result.verdict is Verdict.PASS, result.detail
    assert result.coverage.checked == checked
    assert result.coverage.expected == expected


def _assert_aliased_per_expert(ctx: CheckpointGateContext, *, checked: int) -> None:
    """Selection still sees every name; distinctness must fire on shared storage."""
    experts = cg._expert_weights(ctx)
    assert len(experts) == checked  # the count looks right — that is the point

    result = ExpertDistinctnessGate().run(ctx)
    assert result.verdict is Verdict.FAIL, result.detail
    assert "aliased to the same bytes" in result.detail
    assert result.coverage.checked == checked


# ---------------------------------------------------------------------------
# SELECTION, STACKED (Gemma-4 HF) — the probe-regression pin.
# ---------------------------------------------------------------------------


def test_stacked_hf_selects_all_60_expert_tensors_and_never_the_router() -> None:
    """The regression pin for the real-artifact finding.

    Measured on the production checkpoint: 30 layers x {gate_up_proj, down_proj}
    = 60 stacked expert weights, plus 30 router vectors. On the pre-fix
    Megatron-only selector this exact context selected 0 tensors and the
    distinctness gate answered VACUOUS, *claiming* the checkpoint contained no
    expert tensors — a false reason over 42.5 GiB of MoE weights. Deleting this
    test lets the selector narrow again (back to a linear_fc-only pattern, or
    to a substring match that swallows the router) with nothing noticing.
    """
    ctx = _stacked_gemma_ctx()
    assert len(ctx.tensors) == 90  # 60 expert weights + 30 routers

    experts = cg._expert_weights(ctx)
    expert_fqns = {t.fqn for t in experts}
    router_fqns = {t.fqn for t in ctx.tensors if ".router." in t.fqn}

    assert len(experts) == 60
    assert len(cg._expert_weight_candidates(ctx.tensors)) == 60
    assert len(router_fqns) == 30
    assert not (router_fqns & expert_fqns)
    assert "model.language_model.layers.0.router.per_expert_scale" in router_fqns
    assert "model.language_model.layers.0.router.per_expert_scale" not in expert_fqns
    assert "model.language_model.layers.0.experts.down_proj" in expert_fqns

    result = ExpertDistinctnessGate().run(ctx)
    assert result.verdict is Verdict.SKIP, result.detail
    assert result.evidence.get("layout") == "stacked"
    assert result.evidence.get("stacked_tensor_count") == 60
    # The pre-fix false reason must never come back.
    assert "no expert tensors" not in result.detail


# ---------------------------------------------------------------------------
# MUST_PASS, STACKED: an explicit SKIP abstention — never PASS, never FAIL.
# ---------------------------------------------------------------------------


def test_stacked_hf_clean_abstains_as_skip_never_pass_never_fail() -> None:
    """The honest clean verdict for a stacked checkpoint, pinned both ways.

    Per-expert identity inside a stacked tensor is metadata-invisible by
    construction, so the gate checks everything metadata CAN show (leading
    dims, sibling pricing, whole-tensor span sharing) and then ABSTAINS.
    Deleting this test lets a future change either promote the abstention to a
    PASS — doctrine (5)'s defect, a claim broader than its evidence — or
    regress it to a FAIL false alarm that teaches operators to bypass the gate.
    """
    ctx = _with_routers(fx.stacked_hf_moe_ctx(), prefix="model.language_model")

    result = ExpertDistinctnessGate().run(ctx)

    assert result.verdict is Verdict.SKIP, result.detail
    assert result.verdict is not Verdict.PASS
    assert result.verdict is not Verdict.FAIL
    assert not result.blocking
    assert result.coverage.checked == 4  # every stacked tensor WAS examined
    assert result.coverage.expected is None  # no honest per-tensor denominator exists

    text = result.detail.lower()
    assert "stacked moe layout" in text
    assert "abstains" in text
    assert "unobservable from metadata" in text
    assert result.evidence.get("layout") == "stacked"
    assert result.evidence.get("stacked_tensor_count") == 4
    assert result.evidence.get("declared_experts") == 8
    assert result.evidence.get("per_expert_identity") == "unobservable-from-metadata"
    # The gate examined four kinds of evidence and says so; it does not claim
    # the per-expert-layout conclusion, which would be false about this checkpoint.
    assert "occupies distinct storage" not in result.detail


def test_megatron_fused_clean_abstains_as_skip_not_pass() -> None:
    """Megatron's suffix-less fused names are the same stacked epistemology.

    ``...experts.experts.linear_fc1.weight`` carries every expert of a layer on
    dim 0 of one tensor: one FQN, one storage span, per-expert identity
    invisible. Deleting this test lets a future change special-case the older
    spelling into a full PASS on distinctness — the false claim the stacked
    abstention exists to prevent — without any tripwire.
    """
    ctx = _with_routers(fx.healthy_fused_moe_ctx(), prefix="model")

    experts = cg._expert_weights(ctx)
    assert len(experts) == 4  # fc1 + fc2 on each of 2 layers
    assert "model.layers.0.router.per_expert_scale" not in {t.fqn for t in experts}

    result = ExpertDistinctnessGate().run(ctx)
    assert result.verdict is Verdict.SKIP, result.detail
    assert result.verdict is not Verdict.PASS
    assert result.verdict is not Verdict.FAIL
    assert result.evidence.get("layout") == "stacked"
    assert result.evidence.get("stacked_tensor_count") == 4


# ---------------------------------------------------------------------------
# MUST_FIRE, STACKED: corruptions that metadata CAN see.
# ---------------------------------------------------------------------------


def test_stacked_hf_leading_dim_starvation_fails_and_names_the_tensor() -> None:
    """A stacked tensor holding 1/8 of the declared experts must FAIL, not abstain.

    Layer 0's down_proj has leading dim 1 where 8 experts were declared — the
    incident's 0.125 ratio in stacked clothing. Abstention only covers what
    metadata cannot see; this is what it can. Deleting this test lets the
    leading-dim check rot (e.g. silently skipping it "because stacked") and a
    starved artifact receives the benign abstention wording instead of a block.
    """
    ctx = _with_routers(fx.stacked_underfilled_ctx(), prefix="model.language_model")

    result = ExpertDistinctnessGate().run(ctx)

    assert result.verdict is Verdict.FAIL, result.detail
    assert "leading dim 1 != declared experts 8" in result.detail
    assert "model.language_model.layers.0.experts.down_proj" in result.detail


def test_stacked_hf_cross_layer_span_alias_fails_and_names_the_tensors() -> None:
    """Two whole-layer stacked tensors on one storage span: the aliasing metadata sees.

    Per-expert aliasing inside a stacked tensor is invisible; two layers'
    tensors answering to one span is aliasing at tensor granularity and must
    block even here. Deleting this test lets the span-sharing check be dropped
    by someone reasoning "stacked layouts abstain" — forgetting that abstention
    is conditional on the observable checks finding nothing.
    """
    ctx = _stacked_gemma_ctx(
        num_layers=2,
        declared_experts=4,
        down_shape=(4, 2816, 704),
        gate_up_shape=(4, 1408, 2816),
        alias_down_proj_across_layers=True,
    )

    result = ExpertDistinctnessGate().run(ctx)

    assert result.verdict is Verdict.FAIL, result.detail
    assert "share one storage span" in result.detail
    assert "model.language_model.layers.0.experts.down_proj" in result.detail
    assert "model.language_model.layers.1.experts.down_proj" in result.detail


def test_megatron_fused_leading_dim_starvation_fails_and_names_the_tensor() -> None:
    """The same starvation signature under the older fused spelling must also fire.

    Deleting this test lets the observable-stacked checks drift to HF-shaped
    names only, so a fused-format checkpoint saved at a fraction of its declared
    experts would slide into the abstention branch on a technicality of naming.
    """
    ctx = _megatron_fused_ctx(
        num_layers=2,
        num_experts=8,
        starved={(0, "linear_fc1.weight")},
    )

    result = ExpertDistinctnessGate().run(ctx)

    assert result.verdict is Verdict.FAIL, result.detail
    assert "leading dim 1 != declared experts 8" in result.detail
    assert "layers.0.mlp.experts.experts.linear_fc1.weight" in result.detail


# ---------------------------------------------------------------------------
# PER-EXPERT families: SELECTION + MUST_PASS with a full denominator.
# ---------------------------------------------------------------------------


def test_megatron_local_shard_clean_passes_with_full_denominator() -> None:
    """The incident's own name shape, stored correctly: distinct storage per shard.

    Deleting this test leaves no known-good control for the exact naming
    pattern the incident used, so a selector change that stops recognizing
    ``weight<i>`` suffixes (or starts rejecting them as unknown) rots silently —
    and with it the gate's reach over the very checkpoint family it was built
    for.
    """
    ctx = _megatron_local_ctx()
    # 4 experts x {fc1, fc2} x 2 layers = 16; denominator prices fc1+fc2 pairs.
    _assert_clean_per_expert(ctx, checked=16, expected=16, router_count=2)


def test_megatron_global_clean_passes_with_full_denominator() -> None:
    """Megatron's global ``experts.<i>.`` spelling: distinct FQN and span per expert.

    Deleting this test lets the per-expert member pattern lose the
    ``<index>.`` requirement (or gain a spurious one) unnoticed; the family
    would then fall to UNKNOWN and block every healthy Megatron-global
    checkpoint — a false alarm no other test here is shaped to catch.
    """
    ctx = _megatron_global_ctx()
    _assert_clean_per_expert(ctx, checked=16, expected=16, router_count=2)


def test_mixtral_clean_passes_with_full_examination() -> None:
    """Mixtral ``block_sparse_moe.experts.<i>.w<n>.weight``: 24 tensors examined.

    The declared-count denominator helper prices two weights per MoE layer
    (fc1 + fc2), so with three projections per expert it reports 16 where 24
    were examined — checked strictly exceeds expected, which is the
    non-blocking direction for the coverage rule. Deleting this test lets
    Mixtral names fall out of the selector unnoticed; the gates would then
    VACUOUS-block a healthy Mixtral save as "no expert tensors".
    """
    ctx = _mixtral_ctx()
    _assert_clean_per_expert(ctx, checked=24, expected=16, router_count=2)


def test_qwen_clean_passes_with_full_examination() -> None:
    """Qwen-MoE ``mlp.experts.<i>.{gate,up,down}_proj.weight``: 24 tensors examined.

    Same denominator note as Mixtral (fc1/fc2 pricing vs three projections).
    Deleting this test leaves the Qwen family — the family whose ``gate_proj``
    suffix encodes the projection a router substring-search would confuse —
    without a known-good control.
    """
    ctx = _qwen_ctx()
    _assert_clean_per_expert(ctx, checked=24, expected=16, router_count=2)


# ---------------------------------------------------------------------------
# PER-EXPERT families: MUST_FIRE on real storage aliasing (N names, one span).
# ---------------------------------------------------------------------------


def test_megatron_local_shard_short_count_still_fails() -> None:
    """The incident itself must still block after the selector gained reach.

    16 shards on disk where 128 experts are declared. Deleting this test lets a
    selector rework blunt the check the whole module exists for — the one
    regression no amount of new-family coverage excuses.
    """
    ctx = _with_routers(fx.aliased_local_names_ctx(), prefix="model")

    result = ExpertDistinctnessGate().run(ctx)

    assert result.verdict is Verdict.FAIL, result.detail
    assert "16 expert shards on disk" in result.detail
    assert "config declares 128" in result.detail


def test_megatron_global_storage_aliasing_fails_distinctness_and_bytes() -> None:
    """Count-correct aliasing needs BOTH signatures: shared spans and short bytes.

    Every one of the 16 FQNs is present — only the storage identities collide —
    so distinctness fires on the shared spans. The byte gate then has two ways
    to say the same incident, and which one it reaches depends entirely on what
    the manifest declares:

      * against an honest denominator (the full un-aliased volume) the aliasing
        reads as a straight DEFICIT — half the declared bytes are not there;
      * against a manifest written from the same broken save, the deficit test
        passes at ratio 1.000 and only the implied-vs-physical disagreement is
        left to fire.

    The second is the dangerous one: every count and every total agrees and the
    checkpoint is still wrong. Asserting only the first — which is what a
    fixture that always declares the full volume can reach — leaves the
    implied-vs-physical branch untested, so a refactor could delete it and the
    suite would still be green on this family.
    """
    ctx = _megatron_global_ctx(alias_mod=2)
    _assert_aliased_per_expert(ctx, checked=16)

    honest = ExpertByteVolumeGate().run(ctx)
    assert honest.verdict is Verdict.FAIL, honest.detail
    assert "measured over distinct storage is below the declared" in honest.detail
    assert honest.evidence.get("implied_expert_bytes") == 65_536
    assert honest.evidence.get("physical_expert_bytes") == 32_768
    assert honest.evidence.get("ratio") == 0.5

    # The same artifact, priced against what is physically on disk.
    complicit = _megatron_global_ctx(alias_mod=2, expected_expert_bytes=32_768)
    byte_result = ExpertByteVolumeGate().run(complicit)
    assert byte_result.verdict is Verdict.FAIL, byte_result.detail
    assert "bytes exist in distinct storage" in byte_result.detail
    assert "multiple FQNs price one physical tensor" in byte_result.detail
    assert byte_result.coverage.checked == 16
    assert "ratio" not in byte_result.evidence  # not a deficit; do not report one


def test_mixtral_storage_aliasing_fails() -> None:
    """The count-correct aliased variant in Mixtral names must FAIL.

    Deleting this test lets the within-stem aliasing check be weakened to
    Megatron-only stems (e.g. keyed on ``linear_fc``) while Mixtral shard
    groups silently compare unique names and pass.
    """
    ctx = _mixtral_ctx(alias_mod=2)
    _assert_aliased_per_expert(ctx, checked=24)
    assert "4 expert FQNs share 2 distinct storages" in (ExpertDistinctnessGate().run(ctx).detail)


def test_qwen_storage_aliasing_fails() -> None:
    """The count-correct aliased variant in Qwen names must FAIL.

    Deleting this test leaves the ``gate_proj``/``up_proj``/``down_proj``
    suffix family without a MUST_FIRE pin distinct from Mixtral's ``w<n>``;
    a suffix-pattern regression that recognized only one style would survive.
    """
    ctx = _qwen_ctx(alias_mod=2)
    _assert_aliased_per_expert(ctx, checked=24)


# ---------------------------------------------------------------------------
# UNKNOWN layout: fail closed, naming the FQN — never a silent drop.
# ---------------------------------------------------------------------------


def test_unknown_expert_layout_blocks_and_names_the_fqn() -> None:
    """An expert-named tensor matching no family must block both gates, by name.

    The anti-silent-drop control: if the selector simply ignored what it did
    not understand, both the denominator and the verdict would describe a
    self-chosen subset as though it were the whole checkpoint. The set-based
    assertion below is the proof nothing is dropped AND nothing extra (the
    routers) is absorbed. Deleting this test lets a "helpful" selector change
    start quietly swallowing unrecognized expert layouts into either bucket.
    """
    ctx = _with_routers(fx.unknown_expert_layout_ctx(), prefix="model")

    candidate_fqns = {t.fqn for t in cg._expert_weight_candidates(ctx.tensors)}
    assert candidate_fqns == {
        "model.layers.0.mlp.experts.weight_bank",
        "model.layers.1.mlp.experts.weight_bank",
    }
    assert cg._expert_weights(ctx) == []  # unrecognized != silently priced

    result = ExpertDistinctnessGate().run(ctx)
    assert result.verdict is Verdict.FAIL, result.detail
    assert "match no recognized MoE layout" in result.detail
    assert "model.layers.0.mlp.experts.weight_bank" in result.detail
    assert result.coverage.checked == 2

    byte_result = ExpertByteVolumeGate().run(ctx)
    assert byte_result.verdict is Verdict.FAIL, byte_result.detail
    assert "model.layers.0.mlp.experts.weight_bank" in byte_result.detail


# ---------------------------------------------------------------------------
# MIXED layout: the mixture itself is the defect — block, do not arbitrate.
# ---------------------------------------------------------------------------


def test_mixed_layout_blocks_with_both_populations_accounted() -> None:
    """Per-expert shards PLUS stacked tensors in one checkpoint must BLOCK.

    Both populations here are individually clean, and that is exactly why this
    is not an abstention. Each per-family check ran over its own subset under a
    denominator that sounds like the whole checkpoint, and the composite of two
    partial verifications is not a verification: metadata cannot separate a
    legitimately heterogeneous model from a half-converted one, and the two
    readings move the denominators in opposite directions. So the mixture is a
    defect of evidence, and the gate refuses rather than picking a winner.

    An earlier draft of this test asserted SKIP, on the theory that an
    unverifiable stacked population downgrades the whole verdict to abstention.
    That is wrong in the direction that matters: SKIP is non-blocking, so a
    half-converted checkpoint would have shipped with a note. Deleting this
    test lets a refactor arbitrate a mixed checkpoint down to "pure stacked"
    (the shards' verified distinctness vanishes from the record) or "pure
    sharded" (the stacked tensors fold into a verified count), and either way
    the coverage number stops describing the artifact.
    """
    ctx = _mixed_layout_ctx()

    experts = cg._expert_weights(ctx)
    assert len(experts) == 4  # 2 shards + 2 stacked; the router is neither

    result = ExpertDistinctnessGate().run(ctx)
    assert result.verdict is Verdict.FAIL, result.detail
    assert result.blocking
    assert result.coverage.checked == 4  # both populations, not the larger one
    assert result.coverage.expected is None  # no honest cross-layout denominator
    assert "MIXED expert layout" in result.detail
    assert "2 per-expert shard tensor(s) in 1 group(s)" in result.detail
    assert "coexist with 2 stacked expert tensor(s)" in result.detail
    assert result.evidence.get("per_expert_tensor_count") == 2
    assert result.evidence.get("stacked_tensor_count") == 2
    # No per-family problem was found; the verdict rests on the mixture alone,
    # so a reader is not left hunting for a value defect that does not exist.
    assert result.evidence.get("recognized_problems") == []
    assert "manifest" in result.evidence.get("would_settle", "")


def test_mixed_layout_detail_also_names_an_unrecognized_tail() -> None:
    """A third, unrecognized population must appear in the sentence, not only the dict.

    MIXED is decided before the UNKNOWN refusal, so a checkpoint carrying all
    three families reaches the mixed message. Without this test the detail can
    describe a two-family checkpoint while the unrecognized tensors sit in
    ``evidence`` unmentioned — a stated reason narrower than the finding, and
    a coverage count that silently omits them.
    """
    ctx = _mixed_layout_ctx()
    stray = TensorMeta(
        fqn="model.layers.9.mlp.experts.weight_bank",
        shape=(2, 64, 32),
        dtype=_EXPERT_DTYPE,
        storage_id="unknown:weight_bank",
    )
    ctx = replace(
        ctx,
        tensors=(*ctx.tensors, stray),
        declared_fqns=(*(ctx.declared_fqns or ()), stray.fqn),
    )

    result = ExpertDistinctnessGate().run(ctx)

    assert result.verdict is Verdict.FAIL, result.detail
    assert "MIXED expert layout" in result.detail
    assert "match no recognized layout at all" in result.detail
    assert stray.fqn in result.detail
    assert result.coverage.checked == 5  # 2 shards + 2 stacked + 1 unrecognized
    assert result.evidence.get("unrecognized_fqns") == [stray.fqn]


# ---------------------------------------------------------------------------
# Byte volume over the stacked layout, in BOTH directions.
# ---------------------------------------------------------------------------


def test_stacked_byte_volume_matches_the_measured_declared_volume() -> None:
    """The byte gate reaches a real PASS on the Gemma-4 layout where distinctness abstains.

    Stacked implied bytes are fully observable, so against the manifest's
    declared volume this gate is non-abstaining — and the auto-derived
    denominator here reproduces the number measured on the production artifact.
    Deleting this test lets the byte gate lose stacked reach (e.g. revert to
    per-FQN pricing only and VACUOUS-block on a 48 GiB healthy artifact).
    """
    ctx = _stacked_gemma_ctx()
    assert ctx.expected_expert_bytes == _GEMMA_EXPERT_BYTES  # 45,675,970,560

    result = ExpertByteVolumeGate().run(ctx)
    assert result.verdict is Verdict.PASS, result.detail
    assert result.coverage.checked == 60
    assert result.coverage.expected == 60
    assert result.evidence.get("physical_expert_bytes") == _GEMMA_EXPERT_BYTES


def test_stacked_byte_volume_fails_against_a_wrong_denominator() -> None:
    """The same clean artifact against an independent WRONG manifest must FAIL.

    A gate that only ever passes has proven nothing; the positive control here
    is the doubled declared volume, which must produce a byte deficit at ratio
    exactly 0.5 — pinned so a loosened deficit threshold cannot launder a real
    shortfall back into green.
    """
    ctx = _stacked_gemma_ctx(expected_expert_bytes=2 * _GEMMA_EXPERT_BYTES)

    result = ExpertByteVolumeGate().run(ctx)
    assert result.verdict is Verdict.FAIL, result.detail
    assert result.evidence.get("ratio") == 0.5
    assert "ratio 0.500" in result.detail


# ---------------------------------------------------------------------------
# The reader's measured storage map: it OUTRANKS the metadata sum, both ways.
# ---------------------------------------------------------------------------


def test_measured_storage_bytes_override_the_metadata_sum() -> None:
    """A reader-measured shortfall must beat a metadata sum that says all is well.

    ``expert_storage_bytes`` exists because the storage-id sum is only as good
    as the storage ids: a reader that walked the actual file can measure what
    metadata merely implies. Until this test, nothing in the suite ever made
    the two disagree, so the override line was executed only where it changed
    no answer — a branch that is exercised but not measured. Here the metadata
    sum reproduces the declared volume exactly (this context PASSes without the
    override) while the reader measures the incident's 1/8, and the gate must
    take the reader's number and fail. Deleting this test lets the override be
    dropped or inverted with the whole suite still green.
    """
    clean = _stacked_gemma_ctx()
    assert ExpertByteVolumeGate().run(clean).verdict is Verdict.PASS

    ctx = replace(clean, expert_storage_bytes=_GEMMA_EXPERT_BYTES // 8)

    result = ExpertByteVolumeGate().run(ctx)
    assert result.verdict is Verdict.FAIL, result.detail
    assert result.evidence.get("physical_expert_bytes") == _GEMMA_EXPERT_BYTES // 8
    assert result.evidence.get("implied_expert_bytes") == _GEMMA_EXPERT_BYTES
    assert "measured over distinct storage is below the declared" in result.detail
    assert "ratio 0.125" in result.detail  # the incident ratio, reproduced exactly


def test_measured_storage_surplus_blocks_under_its_own_description() -> None:
    """More measured bytes than the expert names account for is not aliasing.

    The implied-vs-physical guard is two-directional, so before this test a
    surplus took the aliasing branch and was reported as "multiple FQNs price
    one physical tensor" — a sentence stating the opposite of the measurement.
    A surplus means the measured span carries bytes no expert FQN names, which
    is why the total matching the manifest proves nothing about which bytes
    were counted. Both readings block; only one of them is true at a time, and
    an operator acts on the sentence. Deleting this test lets the two collapse
    back into one message.
    """
    surplus = 4_096
    measured = _GEMMA_EXPERT_BYTES + surplus
    ctx = replace(
        _stacked_gemma_ctx(expected_expert_bytes=measured),
        expert_storage_bytes=measured,
    )

    result = ExpertByteVolumeGate().run(ctx)

    assert result.verdict is Verdict.FAIL, result.detail
    assert f"{surplus:,} byte(s) of the measured span are named by nothing" in result.detail
    assert "multiple FQNs price one physical tensor" not in result.detail
    assert "ratio" not in result.evidence  # it is not a deficit and must not read as one
    assert result.evidence.get("physical_expert_bytes") == measured
    assert result.evidence.get("implied_expert_bytes") == _GEMMA_EXPERT_BYTES


# ---------------------------------------------------------------------------
# The composite at first save: a stacked abstention is coverage 2/3, blocking.
# ---------------------------------------------------------------------------


def test_first_save_reports_two_of_three_on_a_clean_stacked_save() -> None:
    """The composite counts verified properties, not invoked sub-gates.

    On a clean stacked save, distinctness established nothing (SKIP), so the
    composite verifies 2 of 3 properties and its ok() downgrades to
    UNDERCOVERED — blocking for a true reason instead of silently crediting a
    distinctness nobody established. Deleting this test lets the composite
    start counting a SKIP as a checked property, and stacked first saves go
    green over an unexamined claim — the incident's shape one level up. The
    healthy per-expert control pins the opposite failure: over-blocking.
    """
    composite = REGISTRY.get("checkpoint.first_save")

    stacked_ctx = _with_routers(fx.stacked_hf_moe_ctx(), prefix="model.language_model")
    stacked = composite.run(stacked_ctx)
    assert stacked.verdict is Verdict.UNDERCOVERED, stacked.detail
    assert stacked.blocking
    assert stacked.coverage.checked == 2
    assert stacked.coverage.expected == 3
    assert "checkpoint.expert_distinctness skipped" in stacked.detail

    healthy_ctx = _with_routers(fx.healthy_sharded_moe_ctx(), prefix="model")
    healthy = composite.run(healthy_ctx)
    assert healthy.verdict is Verdict.PASS, healthy.detail
    assert healthy.coverage.checked == 3
    assert healthy.coverage.expected == 3


# ---------------------------------------------------------------------------
# The gates' own controls must hold.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("gate_id", _CHECKPOINT_GATE_IDS)
def test_checkpoint_gate_controls_hold(gate_id: str) -> None:
    """Every checkpoint gate keeps at least one working MUST_FIRE control.

    Deleting this test lets the control suites rot in place — e.g. the
    stacked-clean control flipping from SKIP to PASS without anyone noticing
    that the abstention had become a claim — because verify_controls only runs
    where someone wires it.
    """
    assert verify_controls(REGISTRY, gate_ids=[gate_id]) == []
