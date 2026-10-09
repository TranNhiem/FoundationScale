"""Unit tests for the FSDP1+peft empty-adapter save fix (train/fsdp_peft_save.py).

These are the fakes-only legs: the dispatch logic (:func:`is_fsdp1_peft_save`)
and the Trainer-subclass wiring (:func:`fsdp1_peft_save_trainer_class`) are
both pure Python plus plain attribute access, so they are exercised here with
no torch/transformers/peft installed. The GPU proof (real FSDP1 LoRA run,
adapter reloads with no missing keys) is the complement to this file, not a
substitute for it -- see the task's GPU-proof runs (a)/(b).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from types import ModuleType

import pytest

from foundationscale.train.fsdp_peft_save import (
    _atomically_overwrite_adapter,
    build_peft_fsdp1_skeleton_factory,
    fsdp1_peft_save_trainer_class,
    is_fsdp1_peft_save,
)
from foundationscale.train.loop import _model_ties_word_embeddings

# ---------------------------------------------------------------------------
# 1. is_fsdp1_peft_save -- pure dispatch, all eight combinations
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("is_fsdp_enabled", "fsdp_version", "has_factory", "expected"),
    [
        (True, 1, True, True),  # the one case the fix exists for
        (True, 1, False, False),  # FSDP1, but no adapter at all
        (True, 2, True, False),  # FSDP2 (untied Qwen): unaffected by design
        (True, 2, False, False),
        (True, None, True, False),  # FSDP enabled but version unreadable
        (False, 1, True, False),  # DDP (or no distribution): never engages
        (False, 2, True, False),
        (False, None, False, False),
    ],
)
def test_is_fsdp1_peft_save_dispatch(
    is_fsdp_enabled: bool, fsdp_version: int | None, has_factory: bool, expected: bool
) -> None:
    assert (
        is_fsdp1_peft_save(
            is_fsdp_enabled=is_fsdp_enabled,
            fsdp_version=fsdp_version,
            has_peft_skeleton_factory=has_factory,
        )
        is expected
    )


# ---------------------------------------------------------------------------
# 2. build_peft_fsdp1_skeleton_factory -- capture, with a fake PeftModel
# ---------------------------------------------------------------------------


class _FakeBaseModel:
    def __init__(self) -> None:
        self.config = object()


class _FakePeftModel:
    def __init__(self, peft_config: dict[str, object]) -> None:
        self._base = _FakeBaseModel()
        self.peft_config = peft_config

    def get_base_model(self) -> _FakeBaseModel:
        return self._base


def test_build_peft_fsdp1_skeleton_factory_refuses_empty_peft_config() -> None:
    with pytest.raises(ValueError, match="at least one adapter config"):
        build_peft_fsdp1_skeleton_factory(_FakePeftModel({}))


def test_build_peft_fsdp1_skeleton_factory_returns_a_zero_arg_callable() -> None:
    factory = build_peft_fsdp1_skeleton_factory(_FakePeftModel({"default": object()}))
    assert callable(factory)
    # Capture happens eagerly (base class/config/dtype/peft_config read NOW,
    # before FSDP can touch the live model); only the meta-device CONSTRUCTION
    # is deferred to save time, which is why calling the factory itself needs
    # real torch/peft and is exercised by the GPU proof instead of here.


def test_factory_builds_the_meta_skeleton_before_peft_and_does_not_swallow_its_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Calling the returned factory: construction-order and error-propagation, with real torch.

    Real ``torch`` is used (CPU, meta device -- no allocation, no CUDA needed);
    only ``peft`` is stubbed, via the same ``sys.modules`` stand-in pattern
    ``tests/train/test_train_precision_adapter.py`` uses for the same package.
    The base class is built under ``torch.device("meta")`` -- proven by
    recording the device active inside its own ``__init__`` -- STRICTLY BEFORE
    ``peft.get_peft_model`` is ever called, and a failure raised by
    ``get_peft_model`` propagates verbatim rather than being caught and
    re-guessed: ``FsdpPeftSaveTrainer._save`` (fsdp_peft_save.py) calls
    ``factory()`` with no try/except of its own, so a defect here must surface
    loudly, never silently, at the one place -- checkpoint save time -- this
    whole module exists to get right.
    """
    calls: list[str] = []

    class _FakeBaseCls:
        def __init__(self, config: object) -> None:
            calls.append("base_cls")
            self.config = config
            # Proves the skeleton really is built with no live storage: the
            # context manager set by `with torch.device("meta"):` makes this
            # the default device for any tensor this constructor might make.
            import torch

            assert torch.get_default_device() == torch.device("meta")

    def _boom_get_peft_model(skeleton: object, config: object, *, adapter_name: str) -> None:
        calls.append("get_peft_model")
        raise RuntimeError("peft refused this skeleton")

    peft_module = ModuleType("peft")
    peft_module.get_peft_model = _boom_get_peft_model  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "peft", peft_module)

    # get_base_model()'s result only needs to answer `type(...)` as
    # `_FakeBaseCls` (so the factory reconstructs THAT class) and `.config` --
    # built via `object.__new__` so capturing it does not itself ring the
    # constructor's own `calls.append("base_cls")`, which must fire exactly
    # once, inside the factory, under the meta-device context manager.
    base_model_stub = object.__new__(_FakeBaseCls)
    base_model_stub.config = object()

    peft_model = _FakePeftModel({"default": object()})
    monkeypatch.setattr(peft_model, "get_base_model", lambda: base_model_stub)
    factory = build_peft_fsdp1_skeleton_factory(peft_model)

    with pytest.raises(RuntimeError, match="peft refused this skeleton"):
        factory()

    # ORDER is the load-bearing assertion: the skeleton exists before peft is
    # ever asked to wrap it, exactly as the module docstring's THE FIX section
    # describes ("Build a FRESH PeftModel skeleton ... and hand it the
    # ALREADY-CORRECT state_dict").
    assert calls == ["base_cls", "get_peft_model"]


# ---------------------------------------------------------------------------
# 3. fsdp1_peft_save_trainer_class -- the Trainer subclass wiring, with fakes
# ---------------------------------------------------------------------------


class _RecordingBaseTrainer:
    """A minimal stand-in for transformers.Trainer, recording ``_save`` calls.

    ``fallback_dir`` must be a REAL, existing directory: the fixed path now
    calls ``_atomically_overwrite_adapter``, which needs a real filesystem
    location for ``tempfile.mkdtemp``.
    """

    def __init__(self, fallback_dir: str = "") -> None:
        self.save_calls: list[tuple[object, object]] = []
        self.is_fsdp_enabled = False
        self.args = type("Args", (), {"output_dir": fallback_dir})()

    def _save(self, output_dir: object = None, state_dict: object = None) -> None:
        self.save_calls.append((output_dir, state_dict))


class _HostileBaseTrainer(_RecordingBaseTrainer):
    """Counts reads of is_fsdp_enabled/accelerator and returns a TRUTHY
    non-bool sentinel for each -- the shape a bare test double with a
    catch-all ``__getattr__`` takes elsewhere in this suite (it returns a
    no-op function for any missing attribute, which is truthy, rather than
    raising AttributeError). ``getattr(obj, name, False)`` would NOT fall
    back to its default here, because the attribute access succeeds; a
    dispatch that read these unconditionally would therefore see a truthy
    "is_fsdp_enabled" even when nothing of the sort was ever declared. This
    class exists to prove _save's short-circuit (``if factory is not None``)
    means that landmine is never stepped on when there is nothing to fix --
    the access COUNT stays zero, not merely "happened to resolve False".
    """

    def __init__(self) -> None:
        super().__init__()
        self.hostile_reads = 0

    @property
    def is_fsdp_enabled(self) -> object:  # type: ignore[override]
        self.hostile_reads += 1
        return lambda: None  # truthy, and NOT a bool -- the __getattr__ shape

    @is_fsdp_enabled.setter
    def is_fsdp_enabled(self, value: object) -> None:
        pass

    @property
    def accelerator(self) -> object:
        self.hostile_reads += 1
        return lambda: None


def test_untouched_path_calls_only_super_save_and_skips_attribute_access() -> None:
    """No factory -> byte-identical to calling base_trainer_cls._save directly.

    Asserting ``hostile_reads == 0`` is the load-bearing part: it proves
    ``is_fsdp_enabled``/``accelerator`` were never even LOOKED AT, not merely
    that they happened to resolve falsy -- see _HostileBaseTrainer's docstring
    for why a mere getattr-with-default would not catch this.
    """
    cls = fsdp1_peft_save_trainer_class(_HostileBaseTrainer)
    trainer = cls()
    trainer._fs_peft_skeleton_factory = None
    trainer._save("/out", {"some": "state"})
    assert trainer.save_calls == [("/out", {"some": "state"})]
    assert trainer.hostile_reads == 0


def test_untouched_path_default_factory_is_none() -> None:
    """The class attribute default, with no instance assignment at all."""
    cls = fsdp1_peft_save_trainer_class(_HostileBaseTrainer)
    trainer = cls()
    trainer._save("/out", {"x": 1})
    assert trainer.save_calls == [("/out", {"x": 1})]
    assert trainer.hostile_reads == 0


def _fake_accelerator(fsdp_version: int) -> object:
    plugin = type("Plugin", (), {"fsdp_version": fsdp_version})()
    state = type("State", (), {"fsdp_plugin": plugin})()
    return type("Accelerator", (), {"state": state})()


def _fsdp1_enabled_trainer(factory, fallback_dir: str = "") -> _RecordingBaseTrainer:
    cls = fsdp1_peft_save_trainer_class(_RecordingBaseTrainer)
    trainer = cls(fallback_dir)
    trainer.is_fsdp_enabled = True
    trainer.accelerator = _fake_accelerator(1)
    trainer._fs_peft_skeleton_factory = factory
    return trainer


class _FakeSkeleton:
    """Writes a real file, so the atomic-write path has something to move."""

    def __init__(self, calls: list[tuple[str, dict]]) -> None:
        self._calls = calls

    def save_pretrained(self, output_dir: str, state_dict: dict) -> None:
        self._calls.append((output_dir, state_dict))
        Path(output_dir, "adapter_model.safetensors").write_text("fake-adapter-bytes")


def test_fixed_path_calls_super_save_then_overwrites_with_skeleton(tmp_path: Path) -> None:
    calls: list[tuple[str, dict]] = []
    trainer = _fsdp1_enabled_trainer(lambda: _FakeSkeleton(calls))
    state_dict = {"base_model.model.lora_A.default.weight": "tensor"}
    ckpt_dir = str(tmp_path)
    trainer._save(ckpt_dir, state_dict)
    # super()._save was called first -- the stock behaviour, including its
    # own (broken-for-this-case) adapter write -- and THEN overwritten.
    assert trainer.save_calls == [(ckpt_dir, state_dict)]
    assert len(calls) == 1
    assert calls[0][1] == state_dict
    # The skeleton wrote into a TEMP subdirectory, not ckpt_dir directly...
    assert calls[0][0] != ckpt_dir
    assert Path(calls[0][0]).parent == tmp_path
    # ...and the final file landed in ckpt_dir via os.replace, with the temp
    # directory cleaned up afterward.
    assert (tmp_path / "adapter_model.safetensors").read_text() == "fake-adapter-bytes"
    assert not Path(calls[0][0]).exists()
    assert list(tmp_path.iterdir()) == [tmp_path / "adapter_model.safetensors"]


def test_fixed_path_uses_args_output_dir_when_output_dir_is_none(tmp_path: Path) -> None:
    calls: list[tuple[str, dict]] = []
    trainer = _fsdp1_enabled_trainer(lambda: _FakeSkeleton(calls), fallback_dir=str(tmp_path))
    trainer._save(None, {"k": "v"})
    assert (tmp_path / "adapter_model.safetensors").read_text() == "fake-adapter-bytes"


def test_fixed_path_refuses_loudly_on_missing_state_dict(tmp_path: Path) -> None:
    trainer = _fsdp1_enabled_trainer(lambda: pytest.fail("factory must not be built"))
    ckpt_dir = str(tmp_path)
    with pytest.raises(RuntimeError, match="state_dict is None"):
        trainer._save(ckpt_dir, None)
    # super()._save still ran (it is called unconditionally, first) -- only
    # the overwrite step refused.
    assert trainer.save_calls == [(ckpt_dir, None)]


def test_fsdp2_with_a_factory_is_untouched(tmp_path: Path) -> None:
    """FSDP2 (the untied-Qwen arm): a factory may exist, but version=2 must not engage it."""
    cls = fsdp1_peft_save_trainer_class(_RecordingBaseTrainer)
    trainer = cls()
    trainer.is_fsdp_enabled = True
    trainer.accelerator = _fake_accelerator(2)
    trainer._fs_peft_skeleton_factory = lambda: pytest.fail("must not be built under FSDP2")
    ckpt_dir = str(tmp_path)
    trainer._save(ckpt_dir, {"k": "v"})
    assert trainer.save_calls == [(ckpt_dir, {"k": "v"})]


# ---------------------------------------------------------------------------
# 4. _atomically_overwrite_adapter -- temp-dir-then-os.replace, directly
# ---------------------------------------------------------------------------


def test_atomic_overwrite_replaces_existing_broken_file(tmp_path: Path) -> None:
    # The stock _save's own (broken) 40-byte write, already on disk.
    (tmp_path / "adapter_model.safetensors").write_bytes(b"\x00" * 40)
    calls: list[tuple[str, dict]] = []
    skeleton = _FakeSkeleton(calls)
    _atomically_overwrite_adapter(skeleton, str(tmp_path), {"k": "v"})
    assert (tmp_path / "adapter_model.safetensors").read_text() == "fake-adapter-bytes"
    assert list(tmp_path.iterdir()) == [tmp_path / "adapter_model.safetensors"]


def test_atomic_overwrite_cleans_up_temp_dir_even_on_failure(tmp_path: Path) -> None:
    class _ExplodingSkeleton:
        def save_pretrained(self, output_dir: str, state_dict: dict) -> None:
            raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        _atomically_overwrite_adapter(_ExplodingSkeleton(), str(tmp_path), {})
    # No leftover temp directory, and the write never touched the real dir.
    assert list(tmp_path.iterdir()) == []


def test_atomic_overwrite_uses_a_real_rename(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    replace_calls: list[tuple[str, str]] = []
    real_replace = Path.replace

    # Recorded at Path.replace, not os.replace: on Python 3.10 pathlib binds
    # os.replace inside its accessor at import time, so patching os.replace
    # never sees the call there (3.12+ calls os.replace directly).
    def _recording_replace(self: Path, target: str | os.PathLike[str]) -> Path:
        replace_calls.append((str(self), str(target)))
        return real_replace(self, target)

    monkeypatch.setattr(Path, "replace", _recording_replace)
    calls: list[tuple[str, dict]] = []
    _atomically_overwrite_adapter(_FakeSkeleton(calls), str(tmp_path), {})
    assert len(replace_calls) == 1
    src, dst = replace_calls[0]
    assert Path(dst) == tmp_path / "adapter_model.safetensors"
    # src is <temp-subdir-of-tmp_path>/adapter_model.safetensors -- one level
    # deeper than dst, since the temp dir is itself a child of tmp_path.
    assert Path(src).parent.parent == tmp_path


# ---------------------------------------------------------------------------
# 5. _model_ties_word_embeddings -- the fact loop.py uses to decide whether a
# skeleton-factory build FAILURE must refuse at START (will this run actually
# be FSDP1?), shared with the FSDP-version-selection site. Imported from
# train/loop.py: the question this module's fix depends on, even though the
# function itself lives beside the rest of the FSDP-version decision.
# ---------------------------------------------------------------------------


class _NS:
    def __init__(self, **kwargs: object) -> None:
        self.__dict__.update(kwargs)


def test_model_ties_word_embeddings_true() -> None:
    model = _NS(config=_NS(tie_word_embeddings=True))
    assert _model_ties_word_embeddings(model) is True


def test_model_ties_word_embeddings_false() -> None:
    model = _NS(config=_NS(tie_word_embeddings=False))
    assert _model_ties_word_embeddings(model) is False


def test_model_ties_word_embeddings_reads_nested_text_config() -> None:
    model = _NS(config=_NS(text_config=_NS(tie_word_embeddings=True)))
    assert _model_ties_word_embeddings(model) is True


def test_model_ties_word_embeddings_missing_attribute_defaults_false() -> None:
    model = _NS(config=_NS())
    assert _model_ties_word_embeddings(model) is False


def test_model_ties_word_embeddings_no_config_defaults_false() -> None:
    assert _model_ties_word_embeddings(_NS()) is False
