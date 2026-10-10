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

from foundationscale.rl.distributed import _find_tower_modules, unshard_for_generation


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
