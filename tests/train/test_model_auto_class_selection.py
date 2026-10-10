"""Tests for ``_media_capable_auto_class``/``_model_can_consume_pixels``.

MEASURED defect these two functions close: transformers 5.18.0 registers
``"qwen3_5"``/``"qwen3_5_moe"`` under ``AutoModelForCausalLM`` ONLY as the
text-only "VLM compatibility" classes ``Qwen3_5ForCausalLM``/
``Qwen3_5MoeForCausalLM`` -- which explicitly skip ``model.visual.*``
weights on load and silently absorb ``pixel_values``/``pixel_values_videos``
through a bare ``**kwargs`` without ever reading them, so a media-declared
run on Qwen3.6-27B/35B-A3B trained with the pixels silently dropped under a
multimodal label. ``gemma4_unified`` has no such split (both auto-mappings
already resolve it to the one ``Gemma4UnifiedForConditionalGeneration``
class), so every gemma4 assertion below is the no-op control.

``Qwen3_5Config``/``LlamaConfig`` are both constructible offline with no
checkpoint and no network -- a real registered-VLM config and a real
plain-text config, the same two shapes train/loop.py resolves against on a
real run.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from foundationscale.families.registry import REGISTRY, FamilySpec
from foundationscale.train.loop import _media_capable_auto_class, _model_can_consume_pixels


def _spec(name: str) -> FamilySpec:
    for spec in REGISTRY:
        if spec.name == name:
            return spec
    raise AssertionError(f"registry has no family {name!r}")


# ---------------------------------------------------------------------------
# _media_capable_auto_class
# ---------------------------------------------------------------------------


def test_text_only_is_always_causal_lm_regardless_of_config_class() -> None:
    from transformers import AutoModelForCausalLM, LlamaConfig, Qwen3_5Config

    for config in (Qwen3_5Config(), LlamaConfig()):
        auto_class, name, verify = _media_capable_auto_class(config, media_declared=False)
        assert auto_class is AutoModelForCausalLM
        assert name == "AutoModelForCausalLM"
        assert verify is False


def test_media_declared_on_a_registered_vlm_picks_image_text_to_text() -> None:
    from transformers import AutoModelForImageTextToText, Qwen3_5Config

    auto_class, name, verify = _media_capable_auto_class(Qwen3_5Config(), media_declared=True)
    assert auto_class is AutoModelForImageTextToText
    assert name == "AutoModelForImageTextToText"
    assert verify is True


def test_media_declared_on_an_unregistered_model_type_falls_back_unchanged() -> None:
    from transformers import AutoModelForCausalLM, LlamaConfig

    auto_class, name, verify = _media_capable_auto_class(LlamaConfig(), media_declared=True)
    assert auto_class is AutoModelForCausalLM
    assert name == "AutoModelForCausalLM"
    assert verify is False


def test_media_declared_but_image_text_to_text_class_unavailable_falls_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A transformers build (or test double) with no AutoModelForImageTextToText
    at all degrades to the unchanged path instead of an uncaught ImportError --
    MEASURED: test_conversation_fused_loss_fsdp_wiring.py's own fake
    transformers module is exactly this shape. ``transformers`` lazy-loads its
    top-level names (a plain ``monkeypatch.delattr`` is re-resolved by its own
    ``__getattr__`` on the next access), so the module itself is replaced in
    ``sys.modules`` with a proxy that forwards everything EXCEPT the one name,
    which is what makes the real import machinery raise ``ImportError``."""
    import sys
    import types

    import transformers as real_transformers
    from transformers import AutoModelForCausalLM, Qwen3_5Config

    class _ProxyModule(types.ModuleType):
        def __getattr__(self, name: str) -> Any:
            if name == "AutoModelForImageTextToText":
                raise AttributeError(name)
            return getattr(real_transformers, name)

    monkeypatch.setitem(sys.modules, "transformers", _ProxyModule("transformers"))
    auto_class, name, verify = _media_capable_auto_class(Qwen3_5Config(), media_declared=True)
    assert auto_class is AutoModelForCausalLM
    assert name == "AutoModelForCausalLM"
    assert verify is False


def test_media_declared_but_image_text_to_text_has_no_mapping_falls_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A stand-in Auto* double (no ``_model_mapping`` attribute at all) --
    MEASURED: test_parallelism_execution.py's ``_install_fake_runtime``
    resolves every ``Auto*`` name to one such double."""
    import transformers
    from transformers import AutoModelForCausalLM, Qwen3_5Config

    class _BareAutoDouble:
        @classmethod
        def from_pretrained(cls, *_a: Any, **_k: Any) -> Any:
            raise AssertionError("not reached by this test")

    monkeypatch.setattr(transformers, "AutoModelForImageTextToText", _BareAutoDouble, raising=False)
    auto_class, name, verify = _media_capable_auto_class(Qwen3_5Config(), media_declared=True)
    assert auto_class is AutoModelForCausalLM
    assert name == "AutoModelForCausalLM"
    assert verify is False


def test_gemma4_unified_resolves_to_the_same_class_either_way() -> None:
    """The no-op control: gemma4_unified has no CausalLM/ConditionalGeneration
    split, so media_declared changes nothing about WHICH class loads."""
    from transformers import AutoModelForImageTextToText, Gemma4UnifiedConfig

    _, name_text_only, verify_text_only = _media_capable_auto_class(
        Gemma4UnifiedConfig(), media_declared=False
    )
    auto_class_media, name_media, verify_media = _media_capable_auto_class(
        Gemma4UnifiedConfig(), media_declared=True
    )
    assert name_text_only == "AutoModelForCausalLM"
    assert verify_text_only is False
    assert auto_class_media is AutoModelForImageTextToText
    assert name_media == "AutoModelForImageTextToText"
    assert verify_media is True


# ---------------------------------------------------------------------------
# _model_can_consume_pixels
# ---------------------------------------------------------------------------


class _ModelWithPixelParam:
    def __init__(self, *, tower: bool) -> None:
        self.config = {"model_type": "qwen3_5"}
        self.model = SimpleNamespace(visual=SimpleNamespace()) if tower else SimpleNamespace()

    def forward(self, input_ids: Any = None, pixel_values: Any = None, **kwargs: Any) -> Any:
        raise AssertionError("not actually called by this test")


class _ModelWithoutPixelParam:
    def __init__(self, *, tower: bool) -> None:
        self.config = {"model_type": "qwen3_5"}
        self.model = SimpleNamespace(visual=SimpleNamespace()) if tower else SimpleNamespace()

    def forward(self, input_ids: Any = None, **kwargs: Any) -> Any:
        raise AssertionError("not actually called by this test")


def test_pixel_param_and_tower_both_present_is_capable() -> None:
    assert _spec("qwen3.5").towers[0] == ("model.visual", "image")  # the path this test relies on
    assert _model_can_consume_pixels(_ModelWithPixelParam(tower=True)) is True


def test_no_pixel_param_is_incapable_even_with_the_tower_present() -> None:
    """MEASURED shape: Qwen3_5ForCausalLM has the family's language layers but
    its forward() has no pixel_values parameter at all (bare **kwargs)."""
    assert _model_can_consume_pixels(_ModelWithoutPixelParam(tower=True)) is False


def test_pixel_param_present_but_tower_absent_is_incapable() -> None:
    """A forward() that NAMES pixel_values on a module tree with no vision
    tower attached (weights skipped on load, or never instantiated)."""
    assert _model_can_consume_pixels(_ModelWithPixelParam(tower=False)) is False


def test_unregistered_family_is_incapable_regardless_of_the_signature() -> None:
    class _Unregistered:
        def __init__(self) -> None:
            self.config = {"model_type": "not-a-real-family"}
            self.model = SimpleNamespace(visual=SimpleNamespace())

        def forward(self, input_ids: Any = None, pixel_values: Any = None, **kwargs: Any) -> Any:
            raise AssertionError("not actually called by this test")

    assert _model_can_consume_pixels(_Unregistered()) is False


def test_no_forward_at_all_is_incapable() -> None:
    model = SimpleNamespace(config={"model_type": "qwen3_5"}, model=SimpleNamespace())
    assert _model_can_consume_pixels(model) is False
