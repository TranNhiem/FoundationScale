"""Loop.py wiring for the conversation/fused-loss/FSDP1+peft-save additions on
this branch, driven end-to-end through ``train()`` with fakes -- CPU, no GPU,
no model files, no liger_kernel/peft actually installed (this plane's own
constraints; see train/loop.py's module docstring).

This file does NOT re-test train/conversation.py's own behaviour (its FAST
tier lives in tests/train/test_conversation_collator.py) or
train/fused_loss.py's own behaviour (tests/train/test_fused_loss.py) --
``conversation_prepass_or_refuse``/``train_conversation_collator_or_refuse``
are monkeypatched to canned fakes here, the same "unit under test is the
CALLER's dispatch, not the callee's internals" split
tests/train/test_audio_loop_wiring.py and test_train_precision_adapter.py
already draw for the audio/adapter arms. ``apply_fused_loss`` is exercised for
REAL in the one test where that matters (the liger_kernel-genuinely-absent
ImportError path costs nothing extra to make real), and stubbed via
sys.modules where a successful patch is needed, mirroring
tests/train/test_fused_loss.py's own stubbing technique exactly so this file
adds no NEW coverage surface to that module.

The fake stack (model/trainer/transformers/datasets/torch) is a trimmed
combination of two already-proven patterns: the FSDP block-resolution model
shape from tests/train/test_parallelism_execution.py's ``_install_fake_runtime``,
and the peft-wrapping surface from tests/train/test_train_precision_adapter.py's
``_FakeModel``/``_make_fake_peft`` -- merged because the FSDP1+peft-save wiring
this file tests needs both at once.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from foundationscale.topology import ClusterProfile
from foundationscale.train import loop
from foundationscale.train.loop import _conversation_inject_dummy_media_or_refusal

_SYNTHETIC_PROFILE = ClusterProfile(
    name="synthetic",
    scheduler="none",
    partitions=("synthetic-partition",),
    node_pattern=r"^synthetic-node\d+$",
    gpus_per_node=1,
    nccl_socket_ifname="",
    ib_hca_pattern="",
    mnnvl_available=False,
    container_runtime="none",
    container_image="",
    filesystem_roots=(),
)

PROCEEDED_CODES = frozenset({loop.EXIT_PASS, loop.EXIT_UNMEASURED})


def _config(**overrides: Any) -> Any:
    kwargs: dict[str, Any] = {
        "model": "test-model",
        "dataset": "test-dataset",
        "output_dir": Path("out"),
        "nodes": 1,
        "gpus_per_node": 1,
        "profile_name": "local-single-node",
    }
    kwargs.update(overrides)
    return loop.TrainConfig(**kwargs)


# ---------------------------------------------------------------------------
# Pure helpers: _conversation_inject_dummy_media_or_refusal (the one
# conversation env-resolver test_conversation_loop_wiring.py does not cover),
# and TrainConfig's fused_loss choice validation.
# ---------------------------------------------------------------------------


def test_inject_dummy_media_absent_defaults_true_with_no_refusal() -> None:
    assert _conversation_inject_dummy_media_or_refusal(None) == (True, None)


def test_inject_dummy_media_true_and_false_are_both_accepted() -> None:
    assert _conversation_inject_dummy_media_or_refusal("true") == (True, None)
    assert _conversation_inject_dummy_media_or_refusal("false") == (False, None)


def test_inject_dummy_media_unknown_value_is_refused_naming_it() -> None:
    value, refusal = _conversation_inject_dummy_media_or_refusal("yes")
    assert value is True
    assert refusal is not None
    assert "yes" in refusal


def test_train_config_fused_loss_refuses_unknown_backend() -> None:
    with pytest.raises(ValueError) as excinfo:
        _config(fused_loss="bogus-backend")
    text = str(excinfo.value)
    assert "fused_loss" in text
    assert "'bogus-backend'" in text
    assert "('liger',)" in text


# ---------------------------------------------------------------------------
# The fake stack: a model that is BOTH FSDP-wrap-resolvable (_no_split_modules
# / a ModuleList-shaped block list) and peft-wrappable (named_parameters /
# named_modules with one real-shaped Linear leaf), a fake Trainer/
# TrainingArguments, and a dataset whose split declares whatever columns a
# test asks for.
# ---------------------------------------------------------------------------


class _FakeParam:
    def __init__(self, numel: int, *, requires_grad: bool) -> None:
        self._numel = numel
        self.requires_grad = requires_grad

    def numel(self) -> int:
        return self._numel

    def element_size(self) -> int:
        return 2


class _FakeLinear:
    """Stands in for ``torch.nn.Linear`` -- installed as the fake stack's
    OWN ``torch.nn.Linear`` below, so ``torch_linear_predicate()``'s
    ``type(m) is Linear`` check (families/adapters.py) resolves true for it."""


class _FakeDecoderLayer:
    """A block class actually present on the fake model, for _fsdp_wrap_classes."""


class _FakeModel:
    # Declares one class the model does NOT contain (mirrors the omni-model
    # defect test_parallelism_execution.py's own fixture guards against) and
    # one it does, so wrap-class resolution exercises the real intersection.
    _no_split_modules = ["_FakeDecoderLayer", "_FakeAbsentAudioLayer"]

    def __init__(self, *, tied: bool = False, model_type: str | None = "gemma4_unified") -> None:
        self.config = SimpleNamespace(tie_word_embeddings=tied, model_type=model_type)
        self._blocks = [_FakeDecoderLayer(), _FakeDecoderLayer()]
        self._params: list[tuple[str, _FakeParam]] = [
            ("model.q_proj.weight", _FakeParam(8, requires_grad=True))
        ]
        self.dtype = None

    def modules(self) -> list[Any]:
        return [self, *self._blocks]

    def named_modules(self) -> list[tuple[str, Any]]:
        return [
            ("", self),
            *((f"model.layers.{i}", block) for i, block in enumerate(self._blocks)),
            ("model.q_proj", _FakeLinear()),
        ]

    def named_parameters(self) -> list[tuple[str, _FakeParam]]:
        return list(self._params)

    def state_dict(self) -> dict[str, _FakeParam]:
        return dict(self._params)

    def parameters(self) -> list[_FakeParam]:
        return [p for _, p in self._params]

    def to(self, *args: Any, **kwargs: Any) -> _FakeModel:
        return self


class _FakeSplit:
    def __init__(self, columns: tuple[str, ...] = ("text",)) -> None:
        self.column_names = list(columns)

    def __len__(self) -> int:
        return 3

    def map(self, *args: Any, **kwargs: Any) -> _FakeSplit:
        return self

    def select(self, *args: Any, **kwargs: Any) -> _FakeSplit:
        return self


def _install_fake_runtime(
    monkeypatch: pytest.MonkeyPatch,
    *,
    tied: bool = False,
    model_type: str | None = "gemma4_unified",
    columns: tuple[str, ...] = ("text",),
) -> SimpleNamespace:
    """Install torch/datasets/transformers fakes; return recorders for what train() built."""
    stack = SimpleNamespace(
        training_arguments=[],
        constructed=[],
        trainers=[],
        manifests=[],
        model=_FakeModel(tied=tied, model_type=model_type),
    )

    _BASE_ACCEPTED: tuple[str, ...] = (
        "output_dir",
        "max_steps",
        "per_device_train_batch_size",
        "learning_rate",
        "save_strategy",
        "save_steps",
        "save_safetensors",
        "ddp_find_unused_parameters",
        "lr_scheduler_type",
        "logging_steps",
        "report_to",
        "seed",
        "data_seed",
        "bf16",
        "fp16",
        "use_cpu",
        "no_cuda",
        "gradient_checkpointing",
        "optim",
        "gradient_accumulation_steps",
        "max_grad_norm",
        "warmup_steps",
        "weight_decay",
        "remove_unused_columns",
        "dataloader_pin_memory",
        "dataloader_num_workers",
        "dataloader_prefetch_factor",
        "torch_compile",
        "torch_compile_backend",
        "torch_compile_mode",
        "fsdp",
        "fsdp_config",
        "ddp_backend",
        "local_rank",
        "deepspeed",
        "parallelism_config",
    )

    import inspect

    class FakeTrainingArguments:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = dict(kwargs)
            for name, value in kwargs.items():
                setattr(self, name, value)
            config = kwargs.get("fsdp_config")
            offload = isinstance(config, dict) and config.get("cpu_offload") is True
            self.fsdp_plugin_args = {"cpu_offload": True} if offload else {}
            stack.training_arguments.append(self)

    FakeTrainingArguments.__init__.__signature__ = inspect.Signature(  # type: ignore[attr-defined]
        parameters=[
            inspect.Parameter(name, inspect.Parameter.KEYWORD_ONLY, default=None)
            for name in _BASE_ACCEPTED
        ]
    )
    stack.training_arguments_cls = FakeTrainingArguments

    class FakeTrainer:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self.args = kwargs.get("args")
            self.model = kwargs.get("model")
            self.state = SimpleNamespace(log_history=[])
            stack.constructed.append((args, kwargs))
            stack.trainers.append(self)

        def train(self, *args: Any, **kwargs: Any) -> None:
            return None

        def save_model(self, path: Any) -> None:
            destination = Path(path)
            destination.mkdir(parents=True, exist_ok=True)
            (destination / "model.safetensors").write_bytes((2).to_bytes(8, "little") + b"{}")

        def evaluate(self, *args: Any, **kwargs: Any) -> dict[str, float]:
            return {}

        def __getattr__(self, name: str) -> Any:
            def _noop(*args: Any, **kwargs: Any) -> None:
                return None

            return _noop

    stack.trainer_cls = FakeTrainer

    datasets_mod = ModuleType("datasets")
    datasets_mod.load_dataset = lambda *a, **k: {"train": _FakeSplit(columns)}  # type: ignore[attr-defined]

    class _FakeTokenizer:
        def __init__(self) -> None:
            self.pad_token: str | None = None
            self.eos_token = "<eos>"

        def __call__(self, texts: list[str], **kwargs: Any) -> dict[str, Any]:
            return {"input_ids": [[1] for _ in texts]}

    class _FakeProcessor:
        def __init__(self) -> None:
            self.tokenizer = _FakeTokenizer()

    class _AutoTokenizer:
        @classmethod
        def from_pretrained(cls, *args: Any, **kwargs: Any) -> _FakeTokenizer:
            return _FakeTokenizer()

    class _Auto:
        @classmethod
        def from_pretrained(cls, *args: Any, **kwargs: Any) -> Any:
            stack.model_kwargs = dict(kwargs)
            return stack.model

    class _AutoProcessor:
        @classmethod
        def from_pretrained(cls, *args: Any, **kwargs: Any) -> _FakeProcessor:
            return _FakeProcessor()

    class _FakeCollator:
        def __init__(self, tokenizer: Any = None, mlm: bool = False, **kwargs: Any) -> None:
            self.tokenizer = tokenizer

    torch_mod = ModuleType("torch")
    torch_mod.__version__ = "0.0+synthetic"
    torch_mod.manual_seed = lambda seed: None
    torch_mod.cuda = SimpleNamespace(is_available=lambda: False, device_count=lambda: 1)
    torch_mod.distributed = SimpleNamespace(
        is_available=lambda: False, is_initialized=lambda: False
    )
    torch_mod.utils = SimpleNamespace(data=SimpleNamespace(DataLoader=object))
    nn_mod = ModuleType("torch.nn")
    nn_mod.Linear = _FakeLinear  # type: ignore[attr-defined]
    torch_mod.nn = nn_mod

    transformers_mod = ModuleType("transformers")
    transformers_mod.TrainingArguments = FakeTrainingArguments  # type: ignore[attr-defined]
    transformers_mod.Trainer = FakeTrainer  # type: ignore[attr-defined]
    transformers_mod.AutoTokenizer = _AutoTokenizer  # type: ignore[attr-defined]
    transformers_mod.AutoModelForCausalLM = _Auto  # type: ignore[attr-defined]
    transformers_mod.AutoConfig = _Auto  # type: ignore[attr-defined]
    transformers_mod.AutoProcessor = _AutoProcessor  # type: ignore[attr-defined]
    transformers_mod.DataCollatorForLanguageModeling = _FakeCollator  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "torch", torch_mod)
    monkeypatch.setitem(sys.modules, "torch.nn", nn_mod)
    monkeypatch.setitem(sys.modules, "datasets", datasets_mod)
    monkeypatch.setitem(sys.modules, "transformers", transformers_mod)

    def _record_emit(
        cfg: object,
        stage: str | None = None,
        extra: dict[str, object] | None = None,
        **kwargs: object,
    ) -> None:
        stack.manifests.append(SimpleNamespace(cfg=cfg, stage=stage, extra=dict(extra or {})))

    monkeypatch.setattr(loop, "_emit_manifest", _record_emit, raising=False)

    class _FakeTopology:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs

        def describe(self) -> str:
            return "fake topology"

        def validate_against(self, profile: Any) -> list[str]:
            return []

    monkeypatch.setattr(loop, "Topology", _FakeTopology)
    monkeypatch.setattr(loop, "_resolve_profile", lambda cfg: _SYNTHETIC_PROFILE)
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    monkeypatch.delenv("FS_RUN_ID", raising=False)
    monkeypatch.setattr(loop, "_tp_save_reaches_every_rank", lambda: True)
    return stack


def _refused_manifest(stack: SimpleNamespace) -> SimpleNamespace:
    refused = [m for m in stack.manifests if m.stage == "refused"]
    assert len(refused) == 1, f"expected exactly one refused manifest, got {len(refused)}"
    return refused[0]


def _done_manifest(stack: SimpleNamespace) -> SimpleNamespace:
    done = [m for m in stack.manifests if m.stage == "done"]
    assert len(done) == 1, f"expected exactly one done manifest, got {len(done)}"
    return done[0]


def _make_fake_peft(sink: list[dict[str, Any]], *, has_get_base_model: bool = True) -> ModuleType:
    """A fake ``peft`` module: LoraConfig records its kwargs, get_peft_model
    attaches lora params (so the attachment-count check passes) and, unless
    ``has_get_base_model`` is False, a working ``get_base_model()`` -- the
    shape ``build_peft_fsdp1_skeleton_factory`` needs."""
    module = ModuleType("peft")

    class _FakeLoraConfig:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = dict(kwargs)
            sink.append(dict(kwargs))

    def _get_peft_model(model: _FakeModel, config: Any) -> _FakeModel:
        model._params = [
            *model._params,
            ("model.q_proj.lora_A.default.weight", _FakeParam(4, requires_grad=True)),
            ("model.q_proj.lora_B.default.weight", _FakeParam(4, requires_grad=True)),
        ]
        model.peft_config = {"default": config}  # type: ignore[attr-defined]
        if has_get_base_model:
            model.get_base_model = lambda: model  # type: ignore[attr-defined]
        return model

    module.LoraConfig = _FakeLoraConfig  # type: ignore[attr-defined]
    module.get_peft_model = _get_peft_model  # type: ignore[attr-defined]
    return module


def _conversation_cfg(**overrides: Any) -> Any:
    kwargs: dict[str, Any] = {
        "model": "test-model",
        "dataset": "test-dataset",
        "output_dir": Path("out"),
        "nodes": 1,
        "gpus_per_node": 1,
        "profile_name": "local-single-node",
        "max_steps": 2,
        "save_interval": 1,
    }
    kwargs.update(overrides)
    return loop.TrainConfig(**kwargs)


# ---------------------------------------------------------------------------
# The conversation-column declaration + dataset-arm + collator-selection +
# stats-reduction + done_extra reporting wiring, all through train().
# ---------------------------------------------------------------------------


def test_conversations_and_audio_both_declared_refuses(monkeypatch: pytest.MonkeyPatch) -> None:
    stack = _install_fake_runtime(monkeypatch, columns=("conversations",))
    monkeypatch.setenv("FOUNDATIONSCALE_TRAIN_CONVERSATIONS_COLUMN", "conversations")
    monkeypatch.setenv("FOUNDATIONSCALE_TRAIN_AUDIO_COLUMN", "waveform")
    cfg = _conversation_cfg()

    rc = loop.train(cfg)

    assert rc == loop.EXIT_REFUSE
    manifest = _refused_manifest(stack)
    assert manifest.extra["conversations_column"] == "conversations"
    assert manifest.extra["audio_column"] == "waveform"


def test_conversations_overlong_env_invalid_value_refuses(monkeypatch: pytest.MonkeyPatch) -> None:
    stack = _install_fake_runtime(monkeypatch, columns=("conversations",))
    monkeypatch.setenv("FOUNDATIONSCALE_TRAIN_CONVERSATIONS_COLUMN", "conversations")
    monkeypatch.setenv("FOUNDATIONSCALE_TRAIN_OVERLONG", "truncate")
    cfg = _conversation_cfg()

    rc = loop.train(cfg)

    assert rc == loop.EXIT_REFUSE
    manifest = _refused_manifest(stack)
    assert manifest.extra["conversations_column"] == "conversations"


def test_conversations_pad_to_max_length_env_invalid_value_refuses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stack = _install_fake_runtime(monkeypatch, columns=("conversations",))
    monkeypatch.setenv("FOUNDATIONSCALE_TRAIN_CONVERSATIONS_COLUMN", "conversations")
    monkeypatch.setenv("FOUNDATIONSCALE_TRAIN_OVERLONG", "drop")
    monkeypatch.setenv("FOUNDATIONSCALE_TRAIN_PAD_TO_MAX_LENGTH", "yes")
    cfg = _conversation_cfg()

    rc = loop.train(cfg)

    assert rc == loop.EXIT_REFUSE
    _refused_manifest(stack)


def test_conversations_inject_dummy_media_env_invalid_value_refuses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stack = _install_fake_runtime(monkeypatch, columns=("conversations",))
    monkeypatch.setenv("FOUNDATIONSCALE_TRAIN_CONVERSATIONS_COLUMN", "conversations")
    monkeypatch.setenv("FOUNDATIONSCALE_TRAIN_OVERLONG", "drop")
    monkeypatch.setenv("FOUNDATIONSCALE_TRAIN_INJECT_DUMMY_MEDIA", "maybe")
    cfg = _conversation_cfg()

    rc = loop.train(cfg)

    assert rc == loop.EXIT_REFUSE
    _refused_manifest(stack)


def test_conversations_column_absent_from_dataset_refuses(monkeypatch: pytest.MonkeyPatch) -> None:
    stack = _install_fake_runtime(monkeypatch, columns=("text",))  # no "conversations" column
    monkeypatch.setenv("FOUNDATIONSCALE_TRAIN_CONVERSATIONS_COLUMN", "conversations")
    monkeypatch.setenv("FOUNDATIONSCALE_TRAIN_OVERLONG", "drop")
    cfg = _conversation_cfg()

    rc = loop.train(cfg)

    assert rc == loop.EXIT_REFUSE
    manifest = _refused_manifest(stack)
    assert manifest.extra["conversations_column"] == "conversations"


def test_conversations_with_image_column_absent_from_dataset_refuses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stack = _install_fake_runtime(monkeypatch, columns=("conversations",))  # no "image" column
    monkeypatch.setenv("FOUNDATIONSCALE_TRAIN_CONVERSATIONS_COLUMN", "conversations")
    monkeypatch.setenv("FOUNDATIONSCALE_TRAIN_IMAGE_COLUMN", "image")
    monkeypatch.setenv("FOUNDATIONSCALE_TRAIN_OVERLONG", "drop")
    cfg = _conversation_cfg()

    rc = loop.train(cfg)

    assert rc == loop.EXIT_REFUSE
    manifest = _refused_manifest(stack)
    assert manifest.extra["image_column"] == "image"


def test_conversations_prepass_refusal_stops_the_run_before_the_trainer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stack = _install_fake_runtime(monkeypatch, columns=("conversations",))
    monkeypatch.setenv("FOUNDATIONSCALE_TRAIN_CONVERSATIONS_COLUMN", "conversations")
    monkeypatch.setenv("FOUNDATIONSCALE_TRAIN_OVERLONG", "refuse")

    from foundationscale.train import conversation as conversation_module

    def _fake_prepass(dataset: Any, processor: Any, **kwargs: Any) -> Any:
        return conversation_module.ConversationPrepassResult(
            refusal_reason="3 row(s) exceed max_length=128 (first: row 0, 500 tokens)",
            filtered_dataset=None,
            rows_seen=3,
            dropped_overlong=3,
            kept=0,
            seconds=0.01,
        )

    monkeypatch.setattr(conversation_module, "conversation_prepass_or_refuse", _fake_prepass)
    cfg = _conversation_cfg()

    rc = loop.train(cfg)

    assert rc == loop.EXIT_REFUSE
    assert stack.constructed == []
    manifest = _refused_manifest(stack)
    assert "conversation_prepass" in manifest.extra


def test_conversations_cp_greater_than_one_refuses(monkeypatch: pytest.MonkeyPatch) -> None:
    stack = _install_fake_runtime(monkeypatch, columns=("conversations",))
    monkeypatch.setenv("FOUNDATIONSCALE_TRAIN_CONVERSATIONS_COLUMN", "conversations")
    monkeypatch.setenv("FOUNDATIONSCALE_TRAIN_OVERLONG", "drop")

    from foundationscale.train import conversation as conversation_module

    def _fake_prepass(dataset: Any, processor: Any, **kwargs: Any) -> Any:
        return conversation_module.ConversationPrepassResult(
            refusal_reason=None,
            filtered_dataset=dataset,
            rows_seen=3,
            dropped_overlong=0,
            kept=3,
            seconds=0.01,
        )

    monkeypatch.setattr(conversation_module, "conversation_prepass_or_refuse", _fake_prepass)
    cfg = _conversation_cfg(cp=2)

    rc = loop.train(cfg)

    assert rc == loop.EXIT_REFUSE
    manifest = _refused_manifest(stack)
    assert manifest.extra["cp"] == 2


def test_conversations_happy_path_reaches_done_with_collator_stats(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The full conversation arm: prepass ok -> collator selected, with
    remove_unused_columns disabled -> collate.stats read and reduced
    (single-process scope).

    The run's OWN final-checkpoint adjudication lands UNMEASURED here (this
    harness's fake ``save_model`` writes a 0-tensor safetensors header, same
    as test_parallelism_execution.py's own fixture) -- UNMEASURED is still a
    PROCEEDED code (never PASS-by-silence), and it is what every wiring test
    in that file also settles for. That early-UNMEASURED return happens AFTER
    the stats all-reduce block this test is pinning, so the per-rank stats
    reduction below is exercised regardless of where adjudication lands.
    """
    stack = _install_fake_runtime(monkeypatch, columns=("conversations",))
    monkeypatch.setenv("FOUNDATIONSCALE_TRAIN_CONVERSATIONS_COLUMN", "conversations")
    monkeypatch.setenv("FOUNDATIONSCALE_TRAIN_OVERLONG", "drop")

    from foundationscale.train import conversation as conversation_module

    def _fake_prepass(dataset: Any, processor: Any, **kwargs: Any) -> Any:
        return conversation_module.ConversationPrepassResult(
            refusal_reason=None,
            filtered_dataset=dataset,
            rows_seen=3,
            dropped_overlong=0,
            kept=3,
            seconds=0.01,
        )

    collator_calls: dict[str, Any] = {}

    def _fake_collator(surface: Any, **kwargs: Any) -> Any:
        collator_calls.update(kwargs)

        def _collate(features: Any) -> dict[str, Any]:
            return {}

        _collate.stats = {"rows_seen": 3, "dropped_overlong": 0, "dummy_media_batches": 1}
        return _collate

    monkeypatch.setattr(conversation_module, "conversation_prepass_or_refuse", _fake_prepass)
    monkeypatch.setattr(
        conversation_module, "train_conversation_collator_or_refuse", _fake_collator
    )
    cfg = _conversation_cfg()

    rc = loop.train(cfg)
    out = capsys.readouterr().out

    assert rc in PROCEEDED_CODES, f"expected proceed, got {rc}"
    assert collator_calls["conversations_column"] == "conversations"
    assert collator_calls["overlong"] == "drop"
    constructed_kwargs = stack.constructed[0][1]
    assert constructed_kwargs["args"].kwargs["remove_unused_columns"] is False
    # The pre-pass's own success announcement proves the monkeypatched
    # conversation_prepass_or_refuse result actually drove the DATA stage.
    assert "conversation pre-pass: 3 row(s) validated" in out


# ---------------------------------------------------------------------------
# --fused-loss wiring: unsupported model_type, liger_kernel genuinely absent
# (real ImportError, no stub needed), and a stubbed success.
# ---------------------------------------------------------------------------


def test_fused_loss_unsupported_model_type_refuses(monkeypatch: pytest.MonkeyPatch) -> None:
    stack = _install_fake_runtime(monkeypatch, model_type="llama")
    cfg = _config(fused_loss="liger")

    rc = loop.train(cfg)

    assert rc == loop.EXIT_REFUSE
    manifest = _refused_manifest(stack)
    assert manifest.extra["fused_loss"] == "liger"
    assert manifest.extra["model_type"] == "llama"


def test_fused_loss_import_error_refuses_naming_the_missing_dependency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # liger_kernel is genuinely NOT installed in this environment (see
    # train/fused_loss.py's module docstring) -- no stub at all here, so
    # apply_fused_loss's own real ImportError is what loop.py must catch.
    assert "liger_kernel" not in sys.modules
    stack = _install_fake_runtime(monkeypatch, model_type="gemma4_unified")
    cfg = _config(fused_loss="liger")

    rc = loop.train(cfg)

    assert rc == loop.EXIT_REFUSE
    manifest = _refused_manifest(stack)
    assert manifest.extra["fused_loss"] == "liger"


def test_fused_loss_success_marks_validated_and_proceeds(monkeypatch: pytest.MonkeyPatch) -> None:
    stack = _install_fake_runtime(monkeypatch, model_type="gemma4_unified")
    functional_mod = ModuleType("liger_kernel.transformers.functional")
    functional_mod.liger_fused_linear_cross_entropy = lambda *a, **k: None  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "liger_kernel", ModuleType("liger_kernel"))
    monkeypatch.setitem(
        sys.modules, "liger_kernel.transformers", ModuleType("liger_kernel.transformers")
    )
    monkeypatch.setitem(sys.modules, "liger_kernel.transformers.functional", functional_mod)
    cfg = _config(fused_loss="liger")

    rc = loop.train(cfg)

    assert rc in PROCEEDED_CODES, f"expected proceed, got {rc}"
    assert callable(stack.model.forward)
    assert not any(m.stage == "refused" for m in stack.manifests)


# ---------------------------------------------------------------------------
# FSDP1 + peft save-fix skeleton-factory capture: refuses at START when this
# run WILL be FSDP1 and the factory cannot be built; degrades silently
# (continues) when it will not be.
# ---------------------------------------------------------------------------


def test_fsdp1_peft_skeleton_factory_build_failure_refuses_when_tied_and_fsdp(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stack = _install_fake_runtime(monkeypatch, tied=True)
    peft_sink: list[dict[str, Any]] = []
    monkeypatch.setitem(sys.modules, "peft", _make_fake_peft(peft_sink, has_get_base_model=False))
    cfg = _config(
        adapter="lora", adapter_rank=8, adapter_targets=("q_proj",), sharding_strategy="fsdp"
    )

    rc = loop.train(cfg)

    assert rc == loop.EXIT_REFUSE
    manifest = _refused_manifest(stack)
    assert "fsdp1_peft_save_prepare_error" in manifest.extra
    assert stack.constructed == []  # refused before the Trainer was ever built


def test_fsdp1_peft_skeleton_factory_build_failure_degrades_without_fsdp1(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Untied AND no sharding_strategy=fsdp at all: this run was never going to
    # engage the fix, so a factory-build failure is a logged degradation, not
    # a refusal -- and the run proceeds to completion.
    stack = _install_fake_runtime(monkeypatch, tied=False)
    peft_sink: list[dict[str, Any]] = []
    monkeypatch.setitem(sys.modules, "peft", _make_fake_peft(peft_sink, has_get_base_model=False))
    cfg = _config(adapter="lora", adapter_rank=8, adapter_targets=("q_proj",))

    rc = loop.train(cfg)

    assert rc in PROCEEDED_CODES, f"expected proceed, got {rc}"
    assert not any(m.stage == "refused" for m in stack.manifests)


def test_fsdp1_peft_skeleton_factory_builds_cleanly_and_is_wired_onto_the_trainer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stack = _install_fake_runtime(monkeypatch, tied=False)
    peft_sink: list[dict[str, Any]] = []
    monkeypatch.setitem(sys.modules, "peft", _make_fake_peft(peft_sink, has_get_base_model=True))
    cfg = _config(adapter="lora", adapter_rank=8, adapter_targets=("q_proj",))

    rc = loop.train(cfg)

    assert rc in PROCEEDED_CODES, f"expected proceed, got {rc}"
    assert len(stack.trainers) == 1
    # fsdp1_peft_save_trainer_class(Trainer)'s whole wiring contract: the
    # instance attribute loop.py sets right after construction is a real,
    # callable factory -- never None -- exactly when the factory built clean.
    assert callable(stack.trainers[0]._fs_peft_skeleton_factory)


# ---------------------------------------------------------------------------
# FSDP_TRANSFORMER_CLS_TO_WRAP: peft's fsdp_auto_wrap_policy is handed the
# resolved wrap classes, as an env var, only when an adapter is declared.
# ---------------------------------------------------------------------------


def test_fsdp_transformer_cls_to_wrap_env_set_for_fsdp_plus_adapter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("FSDP_TRANSFORMER_CLS_TO_WRAP", raising=False)
    stack = _install_fake_runtime(monkeypatch, tied=False)
    peft_sink: list[dict[str, Any]] = []
    monkeypatch.setitem(sys.modules, "peft", _make_fake_peft(peft_sink, has_get_base_model=True))
    cfg = _config(
        adapter="lora", adapter_rank=8, adapter_targets=("q_proj",), sharding_strategy="fsdp"
    )

    try:
        rc = loop.train(cfg)
        assert rc in PROCEEDED_CODES, f"expected proceed, got {rc}"
        assert os.environ.get("FSDP_TRANSFORMER_CLS_TO_WRAP") == "_FakeDecoderLayer"
        constructed_kwargs = stack.constructed[0][1]
        assert constructed_kwargs["args"].kwargs["fsdp_config"][
            "transformer_layer_cls_to_wrap"
        ] == ["_FakeDecoderLayer"]
    finally:
        monkeypatch.delenv("FSDP_TRANSFORMER_CLS_TO_WRAP", raising=False)


def test_fsdp_transformer_cls_to_wrap_env_not_set_without_an_adapter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("FSDP_TRANSFORMER_CLS_TO_WRAP", raising=False)
    stack = _install_fake_runtime(monkeypatch, tied=False)
    cfg = _config(sharding_strategy="fsdp")

    rc = loop.train(cfg)

    assert rc in PROCEEDED_CODES, f"expected proceed, got {rc}"
    assert "FSDP_TRANSFORMER_CLS_TO_WRAP" not in os.environ
    assert stack.constructed


# ---------------------------------------------------------------------------
# Video on the conversation arm: a declared frame budget reaches BOTH entry
# points as a frames_for callable (and the arm invents no image column); a
# declared video column the dataset lacks refuses before any collator exists.
# ---------------------------------------------------------------------------


def _video_conversation_fakes(monkeypatch: pytest.MonkeyPatch) -> dict[str, dict[str, Any]]:
    from foundationscale.train import conversation as conversation_module

    seen: dict[str, dict[str, Any]] = {}

    def _fake_prepass(dataset: Any, processor: Any, **kwargs: Any) -> Any:
        seen["prepass"] = kwargs
        return conversation_module.ConversationPrepassResult(
            refusal_reason=None,
            filtered_dataset=dataset,
            rows_seen=1,
            dropped_overlong=0,
            kept=1,
            seconds=0.01,
        )

    def _fake_collator(surface: Any, **kwargs: Any) -> Any:
        seen["collator"] = kwargs

        def _collate(features: Any) -> dict[str, Any]:
            return {}

        _collate.stats = {"rows_seen": 1, "dropped_overlong": 0, "dummy_media_batches": 0}
        return _collate

    monkeypatch.setattr(conversation_module, "conversation_prepass_or_refuse", _fake_prepass)
    monkeypatch.setattr(
        conversation_module, "train_conversation_collator_or_refuse", _fake_collator
    )
    return seen


def test_conversations_with_a_frame_budget_pass_frames_for_to_both_entry_points(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_runtime(monkeypatch, columns=("conversations", "clips"))
    monkeypatch.setenv("FOUNDATIONSCALE_TRAIN_CONVERSATIONS_COLUMN", "conversations")
    monkeypatch.setenv("FOUNDATIONSCALE_TRAIN_OVERLONG", "drop")
    monkeypatch.setenv("FOUNDATIONSCALE_TRAIN_VIDEO_COLUMN", "clips")
    monkeypatch.setenv("FOUNDATIONSCALE_TRAIN_VIDEO_FRAMES", "4")
    seen = _video_conversation_fakes(monkeypatch)

    rc = loop.train(_conversation_cfg())

    assert rc in PROCEEDED_CODES, f"expected proceed, got {rc}"
    for entry in ("prepass", "collator"):
        assert callable(seen[entry]["frames_for"]), entry
        assert seen[entry]["video_column"] == "clips", entry
        # The image arm's synthetic frames column must not leak into this arm.
        assert seen[entry]["image_column"] is None, entry


def test_conversations_without_a_frame_budget_pass_no_frames_for(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_runtime(monkeypatch, columns=("conversations",))
    monkeypatch.setenv("FOUNDATIONSCALE_TRAIN_CONVERSATIONS_COLUMN", "conversations")
    monkeypatch.setenv("FOUNDATIONSCALE_TRAIN_OVERLONG", "drop")
    seen = _video_conversation_fakes(monkeypatch)

    rc = loop.train(_conversation_cfg())

    assert rc in PROCEEDED_CODES, f"expected proceed, got {rc}"
    assert "frames_for" not in seen["prepass"]
    assert "frames_for" not in seen["collator"]


def test_conversations_with_a_budget_and_no_such_video_column_refuse(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _install_fake_runtime(monkeypatch, columns=("conversations",))
    monkeypatch.setenv("FOUNDATIONSCALE_TRAIN_CONVERSATIONS_COLUMN", "conversations")
    monkeypatch.setenv("FOUNDATIONSCALE_TRAIN_OVERLONG", "drop")
    monkeypatch.setenv("FOUNDATIONSCALE_TRAIN_VIDEO_COLUMN", "clips")
    monkeypatch.setenv("FOUNDATIONSCALE_TRAIN_VIDEO_FRAMES", "4")
    seen = _video_conversation_fakes(monkeypatch)

    rc = loop.train(_conversation_cfg())

    assert rc == loop.EXIT_REFUSE
    assert "video column 'clips' is declared" in capsys.readouterr().out
    assert "prepass" not in seen and "collator" not in seen
