"""CPU tests for ``_find_tower_modules`` -- the FSDP2 + multimodal-generate() fix.

MEASURED BUG this fixes (gemma-4-12B-it, GRPO + images, 2026-10-09): under
``sharding='fsdp'``, ``generate()``'s multimodal preprocessing calls
``get_image_features()`` -> ``self.embed_vision(...)`` directly, BEFORE the
decode loop's first ``forward()``. ``_no_split_modules`` for gemma4_unified
names only the text decoder layer, so the vision embedder was never its own
``fully_shard`` unit and stayed DTensor-sharded with no unshard hook to fire
for a direct method call -- ``aten.native_layer_norm`` crashed on "mixed
torch.Tensor and DTensor". Confirmed pre-existing and LoRA-independent via
an ``adapter=None`` control.

These tests exercise ``_find_tower_modules`` against the REAL
``foundationscale.families`` registry (gemma4's actual declared
``towers``), not a stand-in -- the whole point is that this reuses the
SAME registry the SFT plane's LoRA target scoping already trusts, rather
than guessing module names here. torch IS installed in this CPU
environment, so the fake models below are real small ``nn.Module`` trees.

FS_FORBID_SKIPS=1 clean -- nothing here may skip.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from torch import nn

from foundationscale.rl.distributed import (
    DistContext,
    _find_tower_modules,
    unshard_for_generation,
    wrap_fsdp2,
)


class _VisionTower(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(4, 4)


class _LanguageModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList(nn.Linear(4, 4) for _ in range(2))


class _InnerModel(nn.Module):
    """Mirrors the real tree: <root>.model.{language_model, embed_vision}."""

    def __init__(self, *, with_vision: bool = True, alias: str = "embed_vision") -> None:
        super().__init__()
        self.language_model = _LanguageModel()
        if with_vision:
            setattr(self, alias, _VisionTower())


class _FakeVLM(nn.Module):
    def __init__(
        self, model_type: str, *, with_vision: bool = True, alias: str = "embed_vision"
    ) -> None:
        super().__init__()
        self.model = _InnerModel(with_vision=with_vision, alias=alias)
        self.config = SimpleNamespace(to_dict=lambda: {"model_type": model_type})


def test_find_tower_modules_resolves_the_registered_gemma4_embed_vision_path() -> None:
    model = _FakeVLM("gemma4_unified", alias="embed_vision")
    towers = _find_tower_modules(model)
    assert towers == [model.model.embed_vision]


class _FakePeftWrapper:
    """Mimics peft's attribute delegation: ``.model`` is the ORIGINAL
    wrapped object, not ``original.model`` -- the exact shape that made the
    first version of this fix a silent no-op (MEASURED, see
    ``_find_tower_modules``'s docstring). ``.config`` is copied directly,
    the same way ``PeftModel.__init__`` does it (NOT through the ``.model``
    delegation chain), so family resolution is unaffected by this wrapper.
    """

    def __init__(self, wrapped: nn.Module) -> None:
        self.model = wrapped
        self.config = wrapped.config

    def get_base_model(self) -> nn.Module:
        return self.model

    def parameters(self):  # pragma: no cover -- not exercised by these tests
        return self.model.parameters()


def test_find_tower_modules_resolves_correctly_through_a_peft_wrapper() -> None:
    # Regression for the measured off-by-one: resolving "model.embed_vision"
    # against the PEFT WRAPPER (whose own ".model" already equals the raw
    # model) would read it as raw_model.embed_vision -- absent -- and
    # silently return []. Resolving against get_base_model() instead must
    # find the SAME live module the unwrapped case does.
    raw = _FakeVLM("gemma4_unified", alias="embed_vision")
    wrapped = _FakePeftWrapper(raw)
    towers = _find_tower_modules(wrapped)
    assert towers == [raw.model.embed_vision]


def test_find_tower_modules_through_a_peft_wrapper_without_get_base_model() -> None:
    # A wrapper that merely LOOKS peft-shaped (has .model delegation) but
    # exposes no get_base_model must fall back to resolving against itself
    # -- the pre-fix behaviour -- rather than crashing.
    class _BareDelegator:
        def __init__(self, wrapped: nn.Module) -> None:
            self.model = wrapped
            self.config = wrapped.config

    raw = _FakeVLM("gemma4_unified", alias="embed_vision")
    wrapper = _BareDelegator(raw)
    assert _find_tower_modules(wrapper) == []


def test_find_tower_modules_resolves_the_registered_gemma4_vision_tower_alias() -> None:
    # The SAME family declares BOTH "model.vision_tower" (E4B/26B/31B) and
    # "model.embed_vision" (12B/unified) -- only the attribute that actually
    # exists on this checkpoint resolves.
    model = _FakeVLM("gemma4", alias="vision_tower")
    towers = _find_tower_modules(model)
    assert towers == [model.model.vision_tower]


def test_find_tower_modules_dedups_when_two_declared_paths_alias_one_object() -> None:
    model = _FakeVLM("gemma4_unified", alias="embed_vision")
    # Alias BOTH declared gemma4 vision paths to the SAME live module --
    # resolve_module_path must not be allowed to double-wrap it.
    model.model.vision_tower = model.model.embed_vision
    towers = _find_tower_modules(model)
    assert len(towers) == 1
    assert towers[0] is model.model.embed_vision


def test_find_tower_modules_skips_a_resolved_module_with_zero_parameters() -> None:
    model = _FakeVLM("gemma4_unified", alias="embed_vision")
    model.model.embed_vision = nn.Module()  # resolves, but owns no parameters
    assert _find_tower_modules(model) == []


def test_find_tower_modules_returns_empty_for_a_text_only_checkpoint() -> None:
    # The family IS registered (gemma4), but THIS checkpoint has no vision
    # tower attribute at all -- every declared tower path resolves to None.
    model = _FakeVLM("gemma4_unified", with_vision=False)
    assert _find_tower_modules(model) == []


def test_find_tower_modules_returns_empty_for_an_unregistered_family() -> None:
    # Safe degradation, unlike plan_adapter_targets' hard refusal for the
    # same "family unknown" case: wrap_fsdp2's behaviour is unchanged.
    model = _FakeVLM("totally_unregistered_model_type")
    assert _find_tower_modules(model) == []


def test_find_tower_modules_returns_empty_when_config_has_no_to_dict() -> None:
    class _NoConfigModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = nn.Linear(2, 2)

    assert _find_tower_modules(_NoConfigModel()) == []


def test_find_tower_modules_returns_empty_when_to_dict_is_not_a_mapping() -> None:
    class _WeirdConfig:
        def to_dict(self) -> Any:
            return "not-a-dict"

    class _Model(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = nn.Linear(2, 2)
            self.config = _WeirdConfig()

    assert _find_tower_modules(_Model()) == []


# ---------------------------------------------------------------------------
# unshard_for_generation
# ---------------------------------------------------------------------------
#
# MEASURED SECOND BUG (gemma-4-12B-it, GRPO rollout, 2026-10-09): even with
# the vision tower wrapped as its own fully_shard unit, the NEXT forward
# (the root's own first regular forward, inside generate()'s _prefill) still
# saw embed_tokens un-materialised -- the out-of-band tower call apparently
# disturbs whichever unit FSDP2 treats as the coordinating root.
#
# MEASURED THIRD BUG, same date: explicitly unshard()-ing EVERY FSDP2 unit
# (root, every decoder block, every tower) fixed the second bug but broke
# decoder blocks, which DO go through the normal hook path: their real
# forward's own lazy-init collided with this function's explicit one --
# "RuntimeError: FSDP state has already been lazily initialized". Decoder
# blocks are never reached out-of-band, so they need no help; the fix is
# scoped to exactly ROOT + TOWERS, leaving every decoder block on its own
# single, hook-triggered lazy-init. A KNOWN REMAINING LIMITATION (a real
# 30-step run still fails at step 2, see the function's own docstring) is
# NOT fixed here -- reset_iter_state() was tried and made it WORSE
# (AttributeError on its own precondition), so it is not part of this
# function at all. FSDPModule itself needs a real process group to
# construct for real, so these tests monkeypatch the class
# `unshard_for_generation` imports, to exercise the root+towers-only
# scoping / unshard-then-return-a-reshard-callable logic without one.


def test_unshard_for_generation_unshards_root_and_towers_not_decoder_blocks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import torch.distributed.fsdp as fsdp_mod

    calls: list[tuple[str, str]] = []

    class _FakeFSDPModule:
        def unshard(self) -> None:
            calls.append(("unshard", self._tag))

        def reshard(self) -> None:
            calls.append(("reshard", self._tag))

    class _DecoderBlock(nn.Module, _FakeFSDPModule):
        def __init__(self) -> None:
            nn.Module.__init__(self)
            self._tag = "decoder_block"
            self.linear = nn.Linear(2, 2)

    class _VisionTower(nn.Module, _FakeFSDPModule):
        def __init__(self) -> None:
            nn.Module.__init__(self)
            self._tag = "tower"
            self.proj = nn.Linear(4, 4)

    class _LanguageModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layers = nn.ModuleList([_DecoderBlock()])

    class _InnerModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.language_model = _LanguageModel()
            self.embed_vision = _VisionTower()

    class _Root(nn.Module, _FakeFSDPModule):
        def __init__(self) -> None:
            nn.Module.__init__(self)
            self._tag = "root"
            self.model = _InnerModel()
            self.config = SimpleNamespace(to_dict=lambda: {"model_type": "gemma4_unified"})

    monkeypatch.setattr(fsdp_mod, "FSDPModule", _FakeFSDPModule, raising=False)

    model = _Root()
    reshard = unshard_for_generation(model)
    # ORDER is load-bearing (see this function's docstring, point (3)):
    # root MUST unshard before any tower, or torch's _lazy_init() lets the
    # tower self-elect as root and raises when the true root's own
    # lazy-init later reaches it. The decoder block is left for its own hook.
    assert calls == [("unshard", "root"), ("unshard", "tower")]
    calls.clear()
    reshard()
    assert sorted(calls) == [("reshard", "root"), ("reshard", "tower")]


def test_unshard_for_generation_unshards_the_root_even_with_zero_towers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # MEASURED, and the reason an earlier "no tower -> no-op" version of
    # this function was wrong: Qwen3.6-27B (text-only, zero registered
    # towers) hit the SAME "mixed Tensor/DTensor" crash at embed_tokens on
    # its very first generate() call. The root cause is universal --
    # peft.PeftModel.generate() always redirects to get_base_model(),
    # bypassing the root's own forward hook for EVERY peft+FSDP2 model,
    # with or without a tower -- so the root must always be unsharded here.
    import torch.distributed.fsdp as fsdp_mod

    calls: list[str] = []

    class _FakeFSDPModule:
        def unshard(self) -> None:
            calls.append("unshard")

        def reshard(self) -> None:
            calls.append("reshard")

    class _TextOnlyModel(nn.Module, _FakeFSDPModule):
        def __init__(self) -> None:
            nn.Module.__init__(self)
            self.linear = nn.Linear(2, 2)
            self.config = SimpleNamespace(to_dict=lambda: {"model_type": "totally_unregistered"})

    monkeypatch.setattr(fsdp_mod, "FSDPModule", _FakeFSDPModule, raising=False)

    model = _TextOnlyModel()
    reshard = unshard_for_generation(model)
    assert calls == ["unshard"]
    reshard()
    assert calls == ["unshard", "reshard"]


def test_unshard_for_generation_is_a_noop_on_a_non_fsdp_model() -> None:
    class _PlainModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = nn.Linear(2, 2)

    reshard = unshard_for_generation(_PlainModel())
    reshard()  # must not raise: zero units found, zero units resharded


def test_unshard_for_generation_ignores_a_tower_not_actually_fsdp_wrapped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # MEASURED (gemma-4-12B-it, sharding='ddp'): _find_tower_modules
    # resolves the vision embedder from the family registry regardless of
    # sharding -- under ddp it was never fully_shard()-wrapped at all, and
    # calling unshard() on it unconditionally raised AttributeError
    # ('Gemma4UnifiedVisionEmbedder' object has no attribute 'unshard').
    # A tower must be touched only when it is ACTUALLY an FSDPModule.
    import torch.distributed.fsdp as fsdp_mod

    class _FakeFSDPModule:
        def unshard(self) -> None:  # pragma: no cover -- must never be called here
            raise AssertionError("unshard() called on a non-FSDP-wrapped tower")

    class _PlainVisionTower(nn.Module):  # NOT an _FakeFSDPModule -- plain, like under ddp
        def __init__(self) -> None:
            super().__init__()
            self.proj = nn.Linear(4, 4)

    class _Root(nn.Module):  # also NOT FSDP-wrapped, matching sharding='ddp'
        def __init__(self) -> None:
            super().__init__()
            self.model = SimpleNamespace(embed_vision=_PlainVisionTower())
            self.config = SimpleNamespace(to_dict=lambda: {"model_type": "gemma4_unified"})

    monkeypatch.setattr(fsdp_mod, "FSDPModule", _FakeFSDPModule, raising=False)

    model = _Root()
    reshard = unshard_for_generation(model)  # must not raise
    reshard()  # must not raise


# ---------------------------------------------------------------------------
# wrap_fsdp2's shard_towers parameter
# ---------------------------------------------------------------------------
#
# MEASURED DEADLOCK (Qwen3.6-27B, preference trainer, mixed image+video+text
# DPO, 2026-10-10 -- see wrap_fsdp2's own docstring for the full account): a
# separately-sharded tower's all-gather is its own collective, fired only
# when that submodule's forward() runs. The online RL trainer's rollouts are
# safe because every row carries the SAME declared modality; the preference
# trainer's per-row heterogeneous batches are not -- at a given micro-batch
# index, one rank's row can carry media while the peer's does not, so one
# rank issues the tower's all-gather and the other never does. These tests
# exercise the DECISION LOGIC -- which modules get their own fully_shard
# unit -- against fakes for fully_shard/MixedPrecisionPolicy/
# init_device_mesh (each of which needs a real CUDA process group to
# construct for real), never touching a GPU.


def _patch_fsdp2_primitives(monkeypatch: pytest.MonkeyPatch) -> list[Any]:
    """Patch wrap_fsdp2's three FSDP2 primitives with CPU-safe fakes.

    Returns the list ``fully_shard`` was called on, in call order -- the
    one piece of evidence that distinguishes "towers get their own unit"
    from "towers fold into the root unit".
    """
    import torch.distributed.device_mesh as mesh_mod
    import torch.distributed.fsdp as fsdp_mod

    calls: list[Any] = []

    def _fake_fully_shard(module: Any, **_kwargs: Any) -> Any:
        calls.append(module)
        return module

    monkeypatch.setattr(fsdp_mod, "fully_shard", _fake_fully_shard, raising=False)
    monkeypatch.setattr(
        fsdp_mod, "MixedPrecisionPolicy", lambda **_kwargs: SimpleNamespace(), raising=False
    )
    monkeypatch.setattr(
        mesh_mod, "init_device_mesh", lambda *_a, **_kw: SimpleNamespace(), raising=False
    )
    return calls


_WRAP_CTX = DistContext(rank=0, world_size=2, local_rank=0, device="cpu", is_distributed=True)


class _FakeQwen35VisionBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(4, 4)


class _FakeQwen35DecoderLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(4, 4)


class _FakeQwen35VisualTower(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(_FakeQwen35VisionBlock() for _ in range(2))


class _FakeQwen35LanguageModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList(_FakeQwen35DecoderLayer() for _ in range(2))


class _FakeQwen35InnerModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.language_model = _FakeQwen35LanguageModel()
        self.visual = _FakeQwen35VisualTower()


class _FakeQwen35VLM(nn.Module):
    """Mirrors Qwen3.6-27B's REAL shape (measured, transformers 5.18):
    ``_no_split_modules`` names BOTH the text decoder layer class AND the
    vision block class (``Qwen3_5DecoderLayer``, ``Qwen3_5VisionBlock``),
    and the real family registry's ``qwen3.5`` entry declares
    ``("model.visual", "image")`` as its tower. ``model_type="qwen3_5"`` is
    resolved against the REAL ``foundationscale.families`` registry here,
    the same way the gemma4 fixtures above do -- not a stand-in.
    """

    _no_split_modules = ["_FakeQwen35DecoderLayer", "_FakeQwen35VisionBlock"]

    def __init__(self) -> None:
        super().__init__()
        self.model = _FakeQwen35InnerModel()
        self.config = SimpleNamespace(to_dict=lambda: {"model_type": "qwen3_5"})


def test_wrap_fsdp2_shards_the_tower_in_its_own_unit_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _patch_fsdp2_primitives(monkeypatch)
    model = _FakeVLM("gemma4_unified", alias="embed_vision")

    wrap_fsdp2(model, _WRAP_CTX)

    # One fully_shard call for the decoder-block ModuleList's own child, one
    # for the tower, one for the root -- the tower gets its OWN unit.
    assert model.model.embed_vision in calls
    assert calls[-1] is model  # root is always wrapped last
    assert calls.index(model.model.embed_vision) < calls.index(model)


def test_wrap_fsdp2_shard_towers_false_folds_the_tower_into_the_root_unit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _patch_fsdp2_primitives(monkeypatch)
    model = _FakeVLM("gemma4_unified", alias="embed_vision")

    wrap_fsdp2(model, _WRAP_CTX, shard_towers=False)

    # The tower is NEVER its own fully_shard unit -- only decoder blocks and
    # the root are. Its parameters are therefore unsharded as a by-product
    # of the root's own hook, which fires on every top-level forward
    # regardless of that row's modality.
    assert model.model.embed_vision not in calls
    assert calls[-1] is model


def test_wrap_fsdp2_shard_towers_false_still_shards_decoder_blocks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # shard_towers must be scoped to TOWERS only -- decoder blocks (which
    # every row's forward always touches, media or not, so their collective
    # is never at risk of the mismatch) must still get their own units.
    calls = _patch_fsdp2_primitives(monkeypatch)
    model = _FakeVLM("gemma4_unified", alias="embed_vision")

    wrap_fsdp2(model, _WRAP_CTX, shard_towers=False)

    decoder_layers = list(model.model.language_model.layers)
    assert decoder_layers, "fixture must declare at least one decoder layer"
    for layer in decoder_layers:
        assert layer in calls


def test_wrap_fsdp2_shard_towers_false_is_a_noop_with_zero_declared_towers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A text-only checkpoint has nothing _find_tower_modules would have
    # wrapped anyway -- shard_towers=False changes nothing observable.
    calls_true = _patch_fsdp2_primitives(monkeypatch)
    model_true = _FakeVLM("gemma4_unified", with_vision=False)
    wrap_fsdp2(model_true, _WRAP_CTX, shard_towers=True)

    calls_false = _patch_fsdp2_primitives(monkeypatch)
    model_false = _FakeVLM("gemma4_unified", with_vision=False)
    wrap_fsdp2(model_false, _WRAP_CTX, shard_towers=False)

    assert len(calls_true) == len(calls_false)


# ---------------------------------------------------------------------------
# wrap_fsdp2's shard_towers=False, second path: _no_split_modules ALSO
# naming the tower's OWN internal block class.
#
# MEASURED REMAINING HANG (Qwen3.6-27B, preference trainer, mixed
# image+video+text DPO, 2026-10-11 -- a LIVE py-spy --native capture on 2x
# H200 during an actual stall): with ``shard_towers=False`` already applied
# (the fix directly above), the SAME class of deadlock still occurred. Root
# cause: transformers 5.18's ``Qwen3_5VisionModel._no_split_modules`` (and
# ``Qwen3VLVisionModel``'s) is ``["Qwen3_5DecoderLayer", "Qwen3_5VisionBlock"]``
# -- HF names the VISION transformer's own block class there too, for its
# own unrelated device_map sharding granularity. ``find_decoder_blocks``'s
# PRIMARY match path walks the WHOLE model matching class names in
# ``_no_split_modules`` -- UNCONDITIONALLY, never gated by ``shard_towers``
# -- so it independently discovers every ``Qwen3_5VisionBlock`` instance
# living inside ``model.visual`` and hands each one its OWN ``fully_shard``
# unit, exactly reproducing the asymmetric-all-gather hang that
# ``shard_towers=False`` was built to close, via a second, unrelated code
# path ``_find_tower_modules`` was never scoped to touch. gemma4 never hit
# this because ``Gemma4UnifiedTextDecoderLayer`` is the ONLY entry in its
# ``_no_split_modules`` (see this module's own docstring) -- a family
# difference, not a mechanism this fix can keep ignoring. Confirmed on the
# real cluster run: rank0's row carried an image (vision block executed),
# rank1's SAME-index row was text-only (vision block never executed) --
# rank0 hung inside the vision block's forward, rank1 hung waiting at the
# training loop's own cross-rank ``agree_all`` barrier for it.
def test_wrap_fsdp2_shard_towers_false_also_folds_no_split_modules_vision_blocks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _patch_fsdp2_primitives(monkeypatch)
    model = _FakeQwen35VLM()

    wrap_fsdp2(model, _WRAP_CTX, shard_towers=False)

    vision_blocks = list(model.model.visual.blocks)
    decoder_layers = list(model.model.language_model.layers)
    assert vision_blocks, "fixture must declare vision blocks"
    assert decoder_layers, "fixture must declare decoder layers"
    for block in vision_blocks:
        assert block not in calls, (
            "a vision block matched ONLY via _no_split_modules (not via "
            "_find_tower_modules) must still fold into the root unit when "
            "shard_towers=False -- its own all-gather is exactly the "
            "asymmetric collective that hangs a heterogeneous-modality "
            "micro-batch, the same failure shard_towers=False exists to close"
        )
    assert model.model.visual not in calls
    for layer in decoder_layers:
        assert layer in calls, "decoder layers must still get their own unit"
    assert calls[-1] is model


def test_wrap_fsdp2_shard_towers_true_is_unchanged_for_no_split_modules_vision_blocks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # shard_towers=True is the online RL trainer's path, already safe
    # because every row in a rollout batch carries the SAME declared
    # modality (see wrap_fsdp2's own docstring). This fix must not touch
    # that path: _no_split_modules-matched vision blocks are wrapped
    # individually exactly as before (a pre-existing, harmless nesting
    # under the tower's own separate unit), never folded into root.
    calls = _patch_fsdp2_primitives(monkeypatch)
    model = _FakeQwen35VLM()

    wrap_fsdp2(model, _WRAP_CTX, shard_towers=True)

    for block in model.model.visual.blocks:
        assert block in calls
    assert model.model.visual in calls
    for layer in model.model.language_model.layers:
        assert layer in calls
    assert calls[-1] is model
