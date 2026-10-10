"""Tests for ``trainer._load_causal_lm``'s media-aware auto-class selection.

The RL-plane counterpart of tests/train/test_model_auto_class_selection.py.

MEASURED defect (GRPO with images, Qwen3.6-35B-A3B / Qwen3.6-27B): transformers
5.18.0 registers ``qwen3_5``/``qwen3_5_moe`` under ``AutoModelForCausalLM``
ONLY as the text-only ``Qwen3_5ForCausalLM``/``Qwen3_5MoeForCausalLM``
classes, which skip ``model.visual`` on load and silently absorb
``pixel_values``/``image_grid_thw`` through a bare ``**kwargs`` without ever
reading them. The RL trainer's old loader tried ``AutoModelForCausalLM``
FIRST and only fell back to ``AutoModelForImageTextToText`` on an exception
-- but that first call SUCCEEDS on this family, so the fallback never fires
and the wrong, pixel-blind class trains (surfacing later, at the first
``generate()``, as "model_kwargs are not used by the model").

``_load_causal_lm`` closes this by asking train/loop.py's own
``_media_capable_auto_class``/``_model_can_consume_pixels`` BEFORE
``from_pretrained`` whenever the corpus declares images, and refusing (exit
96) if the selected class still cannot consume pixels. A text-only corpus
keeps the exact historical try/except fallback, untouched.

Hermetic: real ``transformers`` (``Qwen3_5Config`` is constructible offline,
no checkpoint, no network) with only ``from_pretrained`` entry points
monkeypatched -- so ``AutoModelForImageTextToText._model_mapping`` membership
is the REAL mapping transformers ships, not a stand-in.

``import transformers`` is repeated INSIDE every test, immediately before
patching, rather than hoisted to module scope: MEASURED, transformers' own
``_LazyModule`` package object is not stable across a full test-suite run
(another test constructing a real HF model, elsewhere in tests/rl/, leaves
``sys.modules['transformers']`` bound to a different object than this file
saw at collection time). ``_load_causal_lm`` itself always resolves
``transformers`` fresh via its own lazy ``from transformers import ...``
inside the function, so a test patching a stale, orphaned module object
would silently patch nothing and exercise the REAL loader instead -- which
is exactly backwards for a regression suite over this bug. A local
``import transformers`` re-binds to whatever ``sys.modules`` currently holds,
matching what the code under test will see.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

pytest.importorskip("torch", reason="trainer.py itself refuses torch-free hosts")

import foundationscale.rl.trainer as trainer_module  # noqa: E402


class _PixelCapableModel:
    """A loaded instance ``_model_can_consume_pixels`` accepts: forward()
    names ``pixel_values`` and the family's declared image tower resolves."""

    def __init__(self) -> None:
        self.config = {"model_type": "qwen3_5"}
        self.model = SimpleNamespace(visual=SimpleNamespace())

    def forward(self, input_ids: Any = None, pixel_values: Any = None, **kwargs: Any) -> Any:
        raise AssertionError("forward() is not called by this test")


class _PixelBlindModel:
    """MEASURED shape: Qwen3_5ForCausalLM -- the family's language layers are
    present (the tower resolves) but forward() is a bare **kwargs catch-all
    with no pixel_values parameter, so it cannot actually consume pixels."""

    def __init__(self) -> None:
        self.config = {"model_type": "qwen3_5"}
        self.model = SimpleNamespace(visual=SimpleNamespace())

    def forward(self, input_ids: Any = None, **kwargs: Any) -> Any:
        raise AssertionError("forward() is not called by this test")


def _fake_auto_config(config: Any) -> Any:
    """A double for the ``AutoConfig`` class with a fixed ``from_pretrained``."""
    return SimpleNamespace(from_pretrained=lambda model_id: config)


def _refuse_if_called(label: str):
    def _boom(model_id: str) -> Any:
        raise AssertionError(f"{label}.from_pretrained must not be called here")

    return _boom


# ---------------------------------------------------------------------------
# Image corpora: the class choice this bug is about
# ---------------------------------------------------------------------------


def test_image_corpus_on_a_registered_vlm_config_selects_image_text_to_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import transformers
    from transformers import Qwen3_5Config

    monkeypatch.setattr(transformers, "AutoConfig", _fake_auto_config(Qwen3_5Config()))
    calls: list[str] = []

    def _load_itt(model_id: str) -> Any:
        calls.append(model_id)
        return _PixelCapableModel()

    monkeypatch.setattr(transformers.AutoModelForImageTextToText, "from_pretrained", _load_itt)
    monkeypatch.setattr(
        transformers.AutoModelForCausalLM,
        "from_pretrained",
        _refuse_if_called("AutoModelForCausalLM"),
    )

    loaded = trainer_module._load_causal_lm("fake/qwen3.6-35b-a3b", needs_images=True)

    assert calls == ["fake/qwen3.6-35b-a3b"]
    assert isinstance(loaded, _PixelCapableModel)


def test_image_corpus_on_an_unregistered_config_falls_back_unverified(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A config in NEITHER mapping (e.g. a plain-text checkpoint used with an
    image corpus) is left on AutoModelForCausalLM with no pixel-capability
    verification -- the same no-op _media_capable_auto_class itself declares."""
    import transformers
    from transformers import LlamaConfig

    monkeypatch.setattr(transformers, "AutoConfig", _fake_auto_config(LlamaConfig()))
    sentinel = object()
    monkeypatch.setattr(
        transformers.AutoModelForCausalLM, "from_pretrained", lambda model_id: sentinel
    )

    loaded = trainer_module._load_causal_lm("fake/plain-text", needs_images=True)

    assert loaded is sentinel


def test_image_corpus_refuses_when_the_selected_class_cannot_consume_pixels(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import transformers
    from transformers import Qwen3_5Config

    monkeypatch.setattr(transformers, "AutoConfig", _fake_auto_config(Qwen3_5Config()))
    monkeypatch.setattr(
        transformers.AutoModelForImageTextToText,
        "from_pretrained",
        lambda model_id: _PixelBlindModel(),
    )

    with pytest.raises(SystemExit) as excinfo:
        trainer_module._load_causal_lm("fake/qwen3.6-35b-a3b", needs_images=True)

    assert excinfo.value.code == 96
    err = capsys.readouterr().err
    assert "cannot consume pixels" in err
    assert "qwen3_5" in err
    assert "fake/qwen3.6-35b-a3b" in err


def test_image_corpus_refuses_when_the_config_itself_fails_to_load(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import transformers

    def _boom(model_id: str) -> Any:
        raise OSError("no such repo")

    monkeypatch.setattr(transformers, "AutoConfig", SimpleNamespace(from_pretrained=_boom))

    with pytest.raises(SystemExit) as excinfo:
        trainer_module._load_causal_lm("fake/missing", needs_images=True)

    assert excinfo.value.code == 96
    err = capsys.readouterr().err
    assert "config load failed" in err
    assert "no such repo" in err


def test_image_corpus_refuses_when_the_selected_class_load_raises(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import transformers
    from transformers import Qwen3_5Config

    monkeypatch.setattr(transformers, "AutoConfig", _fake_auto_config(Qwen3_5Config()))

    def _boom(model_id: str) -> Any:
        raise OSError("no such weights")

    monkeypatch.setattr(transformers.AutoModelForImageTextToText, "from_pretrained", _boom)

    with pytest.raises(SystemExit) as excinfo:
        trainer_module._load_causal_lm("fake/qwen3.6-35b-a3b", needs_images=True)

    assert excinfo.value.code == 96
    assert "AutoModelForImageTextToText" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# Text-only corpora: byte-identical to before this function existed
# ---------------------------------------------------------------------------


def test_text_corpus_loads_directly_under_causal_lm_and_never_touches_autoconfig(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import transformers

    monkeypatch.setattr(
        transformers, "AutoConfig", SimpleNamespace(from_pretrained=_refuse_if_called("AutoConfig"))
    )
    sentinel = object()
    monkeypatch.setattr(
        transformers.AutoModelForCausalLM, "from_pretrained", lambda model_id: sentinel
    )

    loaded = trainer_module._load_causal_lm("fake/llama", needs_images=False)

    assert loaded is sentinel


def test_text_corpus_falls_back_to_image_text_to_text_on_a_causal_lm_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Preserves the historical fallback: a config only registered under
    AutoModelForImageTextToText (no images declared) still loads -- unlike
    the image-corpus path, no pixel-capability verification is performed."""
    import transformers

    sentinel = object()

    def _boom(model_id: str) -> Any:
        raise OSError("unrecognized configuration class for AutoModelForCausalLM")

    monkeypatch.setattr(transformers.AutoModelForCausalLM, "from_pretrained", _boom)
    monkeypatch.setattr(
        transformers.AutoModelForImageTextToText, "from_pretrained", lambda model_id: sentinel
    )

    loaded = trainer_module._load_causal_lm("fake/vlm-only", needs_images=False)

    assert loaded is sentinel


def test_text_corpus_refuses_when_both_auto_classes_fail(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import transformers

    def _boom(model_id: str) -> Any:
        raise OSError("no such weights")

    monkeypatch.setattr(transformers.AutoModelForCausalLM, "from_pretrained", _boom)
    monkeypatch.setattr(transformers.AutoModelForImageTextToText, "from_pretrained", _boom)

    with pytest.raises(SystemExit) as excinfo:
        trainer_module._load_causal_lm("fake/missing", needs_images=False)

    assert excinfo.value.code == 96
    assert "under both auto classes" in capsys.readouterr().err
