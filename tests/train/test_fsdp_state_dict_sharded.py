"""The FSDP sharded-checkpoint option (#544), the offload fabric pair (#545),
and the model/dataset RED line's innermost frame (#548).

#544: sharding_strategy="fsdp" + fsdp_state_dict="sharded" makes accelerate
write one DCP shard per rank under pytorch_model_fsdp_0/ instead of a full
state dict gathered onto rank 0's host -- the gather that exhausted 956 GiB
of host RAM on a 26B cpu-offload save (~106 GB of model + ~202 GB of
optimizer). The saved keys are all wrapped "model.<fqn>", and the save gate's
single entries reader must strip exactly that one wrapper.

#545: with cpu_optimizer_offload under FSDP2, accelerate builds
cuda:nccl,cpu:gloo only when ACCELERATE_USE_FSDP and FSDP_OFFLOAD_PARAMS are
exported before the process group exists; otherwise step 0 dies with "No
backend type associated with device type cpu".

#548: the model/dataset RED line names the innermost traceback frame because
a FileNotFoundError without a filename reported nothing actionable on a
2-node run.

The fake-runtime seam mirrors tests/train/test_parallelism_execution.py:
TrainingArguments kwargs are read back off the captured instance, and
REACHED-SITE is asserted separately from OUTCOME.
"""

from __future__ import annotations

import inspect
import json
import os
import struct
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch
import torch.distributed.checkpoint as dcp

from foundationscale.topology import ClusterProfile
from foundationscale.train import cli as train_cli
from foundationscale.train import loop

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

# The fake TrainingArguments signature. loop.py introspects it and refuses any
# kwarg it does not contain, so a name missing here turns a proceed arm into a
# refusal arm that looks like a subject failure.
_ACCEPTED: tuple[str, ...] = (
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
    "bf16",
    "fp16",
    "gradient_checkpointing",
    "optim",
    "gradient_accumulation_steps",
    "max_grad_norm",
    "warmup_steps",
    "remove_unused_columns",
    "dataloader_num_workers",
    "dataloader_prefetch_factor",
    "torch_compile",
    "torch_compile_backend",
    "torch_compile_mode",
    "fsdp",
    "fsdp_config",
    "include_num_input_tokens_seen",
    "parallelism_config",
)


def _base_kwargs(tmp_path: Path) -> dict[str, object]:
    return {
        "model": "synthetic-model",
        "dataset": "synthetic-dataset",
        "output_dir": tmp_path / "out",
        "nodes": 1,
        "gpus_per_node": 1,
        "profile_name": "synthetic-profile",
        "max_steps": 2,
        "save_interval": 1,
    }


def _install_fake_runtime(
    monkeypatch: pytest.MonkeyPatch,
    *,
    dist: object | None = None,
    load_error: BaseException | None = None,
) -> SimpleNamespace:
    """Install the optional-extra fakes and return recorders. Same seam as
    tests/train/test_parallelism_execution.py, with two dials: ``dist``
    replaces the fake torch.distributed namespace (for the #545 backend guard)
    and ``load_error`` makes every from_pretrained raise (for the frame test).
    """
    stack = SimpleNamespace(training_arguments=[], constructed=[], manifests=[], model_kwargs=None)

    class FakeTrainingArguments:
        def __init__(self, **kwargs: object) -> None:
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
            for name in _ACCEPTED
        ]
    )

    class FakeTrainer:
        def __init__(self, *args: object, **kwargs: object) -> None:
            self.args = kwargs.get("args")
            self.state = SimpleNamespace(log_history=[])
            stack.constructed.append((args, kwargs))

        def train(self, *args: object, **kwargs: object) -> None:
            return None

        def save_model(self, path: object) -> None:
            # Faithful to production: the save WRITES and the loop inspects
            # what landed (here: layout (a), root safetensors).
            destination = Path(path)  # type: ignore[arg-type]
            destination.mkdir(parents=True, exist_ok=True)
            (destination / "model.safetensors").write_bytes((2).to_bytes(8, "little") + b"{}")

        def __getattr__(self, name: str):
            def _noop(*args: object, **kwargs: object) -> None:
                return None

            return _noop

    def _fake_sharded_final(trainer: object, final_dir: Path) -> None:
        # Faithful to production (#544): a sharded final save writes per-rank
        # DCP shards under pytorch_model_fsdp_0/, never root safetensors.
        dcp = final_dir / loop._FSDP_DCP_SUBDIRNAME
        dcp.mkdir(parents=True, exist_ok=True)
        (dcp / "__0_0.distcp").write_bytes(b"shard")
        (dcp / loop._DCP_METADATA_FILENAME).write_bytes(b"meta")

    monkeypatch.setattr(loop, "_save_final_fsdp_sharded", _fake_sharded_final)

    class _FakeSplit:
        column_names = ["text"]

        def __len__(self) -> int:
            return 3

        def map(self, *args: object, **kwargs: object) -> _FakeSplit:
            return self

    datasets_mod = ModuleType("datasets")
    datasets_mod.load_dataset = lambda *args, **kwargs: {"train": _FakeSplit()}

    class _FakeDecoderLayer:
        """A block class that IS present in the fake model."""

    class _FakeModel(SimpleNamespace):
        # Faithful to the omni defect: a declared block class the build never
        # instantiates, so the wrap-class intersection is exercised.
        _no_split_modules = ["_FakeDecoderLayer", "_FakeAbsentAudioLayer"]

        def __init__(self) -> None:
            super().__init__(
                pad_token=None,
                eos_token="</s>",
                config=SimpleNamespace(use_cache=False, tie_word_embeddings=False),
                parameters=lambda: [],
                state_dict=lambda: {},
                to=lambda *a, **k: None,
            )
            self._blocks = [_FakeDecoderLayer(), _FakeDecoderLayer()]

        def modules(self):
            return [self, *self._blocks]

        def named_modules(self):
            return [("", self)] + [(f"model.layers.{i}", b) for i, b in enumerate(self._blocks)]

    class _Auto:
        @classmethod
        def from_pretrained(cls, *args: object, **kwargs: object) -> object:
            if load_error is not None:
                raise load_error
            stack.model_kwargs = dict(kwargs)
            return _FakeModel()

    class _FakeCollator:
        def __init__(self, tokenizer: object = None, mlm: bool = False, **kwargs: object) -> None:
            self.tokenizer = tokenizer
            self.mlm = mlm

    torch_mod = ModuleType("torch")
    torch_mod.__version__ = "0.0+synthetic"
    torch_mod.manual_seed = lambda seed: None
    torch_mod.cuda = SimpleNamespace(is_available=lambda: False, device_count=lambda: 1)
    torch_mod.distributed = dist or SimpleNamespace(
        is_available=lambda: False, is_initialized=lambda: False
    )
    torch_mod.utils = SimpleNamespace(data=SimpleNamespace(DataLoader=object))

    transformers_mod = ModuleType("transformers")
    transformers_mod.TrainingArguments = FakeTrainingArguments  # type: ignore[attr-defined]
    transformers_mod.Trainer = FakeTrainer  # type: ignore[attr-defined]
    stack.training_arguments_cls = FakeTrainingArguments
    transformers_mod.AutoTokenizer = _Auto  # type: ignore[attr-defined]
    transformers_mod.AutoModelForCausalLM = _Auto  # type: ignore[attr-defined]
    transformers_mod.DataCollatorForLanguageModeling = _FakeCollator  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "torch", torch_mod)
    monkeypatch.setitem(sys.modules, "datasets", datasets_mod)
    monkeypatch.setitem(sys.modules, "transformers", transformers_mod)

    def _record_emit(
        cfg: object, stage: str | None = None, extra: dict | None = None, **kwargs: object
    ) -> None:
        stack.manifests.append(SimpleNamespace(cfg=cfg, stage=stage, extra=dict(extra or {})))

    monkeypatch.setattr(loop, "_emit_manifest", _record_emit, raising=False)

    class _FakeTopology:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs

        def describe(self) -> str:
            return "fake topology (validation patched out; not the unit under test)"

        def validate_against(self, profile: object) -> list[str]:
            return []

    monkeypatch.setattr(loop, "Topology", _FakeTopology)
    monkeypatch.setattr(loop, "_resolve_profile", lambda cfg: _SYNTHETIC_PROFILE)
    for var in (
        "WORLD_SIZE",
        "LOCAL_WORLD_SIZE",
        "RANK",
        "LOCAL_RANK",
        "FS_RUN_ID",
        "FOUNDATIONSCALE_TRAIN_IMAGE_COLUMN",
    ):
        monkeypatch.delenv(var, raising=False)
    return stack


def _only_training_arguments(stack: SimpleNamespace) -> SimpleNamespace:
    assert len(stack.training_arguments) == 1, (
        f"expected exactly one TrainingArguments, got {len(stack.training_arguments)}"
    )
    return stack.training_arguments[0]


# --------------------------------------------------------------------------
# #544: the two-layout entries reader.
# --------------------------------------------------------------------------


def _save_dcp_layout(ckpt_dir: Path, state: dict[str, object]) -> Path:
    """Write a real layout-(b) store with torch's own single-process save."""
    subdir = ckpt_dir / "pytorch_model_fsdp_0"
    subdir.mkdir(parents=True)
    writer = dcp.FileSystemWriter(str(subdir))
    try:
        # torch >= 2.4-ish: explicit no_dist so no process group is needed.
        dcp.save(state, storage_writer=writer, no_dist=True)
    except TypeError:
        # Older signature: save falls back to a no-dist plan on its own.
        dcp.save(state, storage_writer=writer)
    assert (subdir / ".metadata").is_file(), "the fixture produced no DCP .metadata"
    return subdir


def _write_root_safetensors(ckpt: Path) -> None:
    blob = json.dumps({"c.weight": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}).encode()
    (ckpt / "model.safetensors").write_bytes(struct.pack("<Q", len(blob)) + blob + b"\x00" * 4)


def test_layout_b_entries_are_stripped_and_dtypes_normalised(tmp_path: Path) -> None:
    """model.<fqn> becomes <fqn>; bfloat16/float32 become BF16/F32."""
    ckpt = tmp_path / "checkpoint-10"
    _save_dcp_layout(
        ckpt,
        {
            "model": {
                "a.weight": torch.zeros(2, 2, dtype=torch.bfloat16),
                "b.bias": torch.zeros(2, dtype=torch.float32),
            }
        },
    )

    entries = loop._checkpoint_weight_entries(ckpt)

    assert dict(entries) == {"a.weight": "BF16", "b.bias": "F32"}
    # OUTCOME on the wrapper: no surviving key may carry it.
    assert all(not name.startswith("model.") for name, _ in entries)


def test_layout_a_still_reads_through_safetensors(tmp_path: Path) -> None:
    """Today's layout answers through _safetensors_entries, unchanged."""
    ckpt = tmp_path / "checkpoint-3"
    ckpt.mkdir()
    _write_root_safetensors(ckpt)

    assert loop._checkpoint_weight_entries(ckpt) == [("c.weight", "F32")]


def test_layout_b_key_without_the_wrapper_raises(tmp_path: Path) -> None:
    """A namespace this plane did not write is not silently accepted."""
    ckpt = tmp_path / "checkpoint-11"
    _save_dcp_layout(ckpt, {"rogue.weight": torch.zeros(2)})

    with pytest.raises(ValueError, match="does not carry the"):
        loop._checkpoint_weight_entries(ckpt)


def test_both_layouts_present_is_ambiguous_and_raises(tmp_path: Path) -> None:
    """Two artifacts claiming one save must not be adjudicated by guessing."""
    ckpt = tmp_path / "checkpoint-12"
    ckpt.mkdir()
    _save_dcp_layout(ckpt, {"model": {"a.weight": torch.zeros(1)}})
    _write_root_safetensors(ckpt)

    with pytest.raises(ValueError, match="BOTH"):
        loop._checkpoint_weight_entries(ckpt)


def test_neither_layout_is_the_historical_empty_case(tmp_path: Path) -> None:
    """[] so the callers' vacuous comparisons refuse, per existing semantics."""
    ckpt = tmp_path / "checkpoint-99"
    ckpt.mkdir()

    assert loop._checkpoint_weight_entries(ckpt) == []


# --------------------------------------------------------------------------
# #544: CLI + config threading.
# --------------------------------------------------------------------------

_BASE_ARGV = [
    "--model",
    "synthetic/model",
    "--dataset",
    "synthetic/data",
    "--output-dir",
    "/tmp/does-not-need-to-exist",
    "--nodes",
    "1",
    "--gpus-per-node",
    "1",
    "--profile-name",
    "synthetic-profile",
]


def test_cli_default_state_dict_is_full() -> None:
    args = train_cli.build_parser().parse_args(_BASE_ARGV)

    assert args.fsdp_state_dict == "full"


def test_cli_parses_sharded_and_threads_it_with_provenance() -> None:
    argv = [*_BASE_ARGV, "--fsdp-state-dict", "sharded"]
    args = train_cli.build_parser().parse_args(argv)
    assert args.fsdp_state_dict == "sharded"

    cfg = train_cli._build_config(argv, args)

    assert cfg.fsdp_state_dict == "sharded"
    # The operator typed it, so its provenance must read cli, not default.
    assert cfg.cli_declared is not None and "fsdp_state_dict" in cfg.cli_declared


def test_train_config_rejects_an_unknown_state_dict_value(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="fsdp_state_dict"):
        loop.TrainConfig(**_base_kwargs(tmp_path), fsdp_state_dict="quartered")


# --------------------------------------------------------------------------
# #544: refusals and wiring.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("strategy", [None, "ddp"])
def test_sharded_without_fsdp_refuses_with_a_refused_manifest(
    strategy: str | None, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """No FSDP, no state_dict_type: refuse at START, before anything imports."""
    kwargs = _base_kwargs(tmp_path / ("none" if strategy is None else strategy))
    if strategy is not None:
        kwargs["sharding_strategy"] = strategy
    rc = loop.train(loop.TrainConfig(**kwargs, fsdp_state_dict="sharded"))

    out = capsys.readouterr().out
    assert rc == loop.EXIT_REFUSE
    assert "fsdp_state_dict='sharded'" in out
    assert "[fs:train:trainer]" not in out
    manifest = kwargs["output_dir"] / loop.MANIFEST_NAME  # type: ignore[operator]
    assert manifest.is_file()
    assert '"refused"' in manifest.read_text()


def test_sharded_puts_state_dict_type_into_the_built_fsdp_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Wired AND still present in what TrainingArguments was actually handed."""
    stack = _install_fake_runtime(monkeypatch)
    cfg = loop.TrainConfig(
        **_base_kwargs(tmp_path), sharding_strategy="fsdp", fsdp_state_dict="sharded"
    )

    rc = loop.train(cfg)

    assert rc in PROCEEDED_CODES
    fsdp_config = _only_training_arguments(stack).kwargs["fsdp_config"]
    assert fsdp_config["state_dict_type"] == "SHARDED_STATE_DICT"
    assert fsdp_config["transformer_layer_cls_to_wrap"] == ["_FakeDecoderLayer"]


def test_default_full_leaves_state_dict_type_absent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Undeclared means unchanged: the fsdp_config dict is byte-identical."""
    stack = _install_fake_runtime(monkeypatch)
    rc = loop.train(loop.TrainConfig(**_base_kwargs(tmp_path), sharding_strategy="fsdp"))

    assert rc in PROCEEDED_CODES
    fsdp_config = _only_training_arguments(stack).kwargs["fsdp_config"]
    assert "state_dict_type" not in fsdp_config


def test_sharded_refuses_when_transformers_drops_the_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The #538 rule on the #544 axis: a silently dropped key refuses the run."""
    stack = _install_fake_runtime(monkeypatch)
    real_init = stack.training_arguments_cls.__init__

    def dropping_init(self: object, **kwargs: object) -> None:
        real_init(self, **kwargs)
        self.fsdp_config = {k: v for k, v in self.fsdp_config.items() if k != "state_dict_type"}

    dropping_init.__signature__ = real_init.__signature__  # type: ignore[attr-defined]
    monkeypatch.setattr(stack.training_arguments_cls, "__init__", dropping_init)
    cfg = loop.TrainConfig(
        **_base_kwargs(tmp_path), sharding_strategy="fsdp", fsdp_state_dict="sharded"
    )

    rc = loop.train(cfg)
    out = capsys.readouterr().out

    assert rc == loop.EXIT_REFUSE
    assert "SHARDED_STATE_DICT" in out
    assert "[fs:train:trainer]" not in out


# --------------------------------------------------------------------------
# #545: the offload fabric pair.
# --------------------------------------------------------------------------


def test_offload_under_fsdp_exports_the_accelerate_env_pair(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """accelerate reads its env when the process group is built, not after."""
    _install_fake_runtime(monkeypatch)
    monkeypatch.delenv("ACCELERATE_USE_FSDP", raising=False)
    monkeypatch.delenv("FSDP_OFFLOAD_PARAMS", raising=False)

    rc = loop.train(
        loop.TrainConfig(
            **_base_kwargs(tmp_path), sharding_strategy="fsdp", cpu_optimizer_offload=True
        )
    )

    assert rc in PROCEEDED_CODES
    assert os.environ["ACCELERATE_USE_FSDP"] == "true"
    assert os.environ["FSDP_OFFLOAD_PARAMS"] == "true"


def test_no_offload_leaves_the_env_pair_unset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Exporting the pair without the declaration would reshape healthy runs."""
    _install_fake_runtime(monkeypatch)
    monkeypatch.delenv("ACCELERATE_USE_FSDP", raising=False)
    monkeypatch.delenv("FSDP_OFFLOAD_PARAMS", raising=False)

    rc = loop.train(loop.TrainConfig(**_base_kwargs(tmp_path), sharding_strategy="fsdp"))

    assert rc in PROCEEDED_CODES
    assert "ACCELERATE_USE_FSDP" not in os.environ
    assert "FSDP_OFFLOAD_PARAMS" not in os.environ


def test_offload_under_fsdp_refuses_when_the_built_group_has_no_cpu_backend(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The measured failure mode: an NCCL-only group under an offload declaration."""
    nccl_only_dist = SimpleNamespace(
        is_available=lambda: True,
        is_initialized=lambda: True,
        get_world_size=lambda: 1,  # one rank: the fabric probe has nothing to warm
        get_backend_config=lambda: "cuda:nccl",
    )
    _install_fake_runtime(monkeypatch, dist=nccl_only_dist)

    rc = loop.train(
        loop.TrainConfig(
            **_base_kwargs(tmp_path), sharding_strategy="fsdp", cpu_optimizer_offload=True
        )
    )
    out = capsys.readouterr().out

    assert rc == loop.EXIT_REFUSE
    assert "cpu:gloo" in out
    assert "No backend type associated with device type cpu" in out
    assert "[fs:train:trainer]" not in out


# --------------------------------------------------------------------------
# #548: the model/dataset RED names where it happened.
# --------------------------------------------------------------------------


def test_model_construction_red_names_the_innermost_frame(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A filename-less FileNotFoundError now says file:line of its origin."""
    _install_fake_runtime(monkeypatch, load_error=FileNotFoundError("checkpoint not found"))

    rc = loop.train(loop.TrainConfig(**_base_kwargs(tmp_path)))
    out = capsys.readouterr().out

    assert rc == loop.EXIT_RED
    assert "[fs:train:red]" in out
    assert "model/dataset construction failed" in out
    assert "innermost frame" in out
    # The raise site is this file's fake from_pretrained.
    assert "test_fsdp_state_dict_sharded.py" in out
