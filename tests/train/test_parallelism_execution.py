"""Pins for the parallelism axes that moved from REFUSED to EXECUTED.

WHY THEY MOVED: sharding is what makes a 26B or 31B checkpoint reachable at all.
Under replication every rank holds the whole model plus its gradients and AdamW
state -- roughly 8 bytes per parameter -- so adding GPUs never lowers per-GPU
memory, and a 26B checkpoint is unmeasurable on a 189 GiB device at any width.
Measured on a GB200 tray, 8 GPUs, seq 2048, batch 1: gemma-4-26B-A4B needs
65.4 GiB per rank sharded and does not fit replicated. `sharding_strategy=fsdp`
now binds transformers' FSDP integration, `cpu_optimizer_offload=true` binds the
offload inside it, and `--tp` / `--cp` bind accelerate's ParallelismConfig.

WHAT STAYS REFUSED, and why that is not laziness:

* `pp` and `ep` -- accelerate's ParallelismConfig carries dp_replicate_size,
  dp_shard_size, tp_size, cp_size and sp_size, and NO pipeline or expert field.
  There is nothing here to bind them to, so a degree above 1 would be recorded
  and never executed. That is finding #375 and the house answer is refusal.
* `zero3` -- ZeRO and DeepSpeed remain adjudicated and unbuilt.
* `cpu_optimizer_offload=true` without fsdp -- the only offload backend reachable
  from this plane lives inside FSDP. Under replication each rank owns a whole
  optimizer and there is no route off device.
* `tp`/`cp` beside a REPLICATED data dimension -- accelerate composes tensor and
  context parallelism with SHARDED data parallelism only. Left to reach the
  backend this arrives as a ValueError adjudicated RED (5), the code for a
  defect, when what happened is an operator describing a mesh that does not
  exist. It is decidable from the declaration, so it is answered 96 at START.

CONTRACT, measured end-to-end on CPU with torch/transformers/datasets faked in
``sys.modules`` exactly where ``train()`` imports them. DECLARED means the kwarg
reaches the TrainingArguments object the run ACTUALLY constructed, read back off
that captured instance rather than off the TrainConfig this module assembled
(#291, #294, #372). Every test asserts REACHED-SITE separately from OUTCOME.
"""

from __future__ import annotations

import inspect
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from foundationscale.topology import ClusterProfile
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
)


def _base_kwargs(tmp_path: Path, gpus_per_node: int = 1) -> dict[str, object]:
    """Minimal real TrainConfig kwargs; gpus_per_node carries the mesh width.

    A literal dict rather than a derivation over "parameters without a default":
    the profile requirement is a CROSS-FIELD invariant in __post_init__ --
    exactly one of profile / profile_path / profile_name -- and all three
    default to None, so that rule fills none of them and every test dies in the
    constructor.
    """
    return {
        "model": "synthetic-model",
        "dataset": "synthetic-dataset",
        "output_dir": tmp_path / "out",
        "nodes": 1,
        "gpus_per_node": gpus_per_node,
        "profile_name": "synthetic-profile",
        "max_steps": 2,
        "save_interval": 1,
    }


def _install_fake_runtime(
    monkeypatch: pytest.MonkeyPatch,
    *,
    parallelism: bool = True,
    blocks: bool = True,
    tied: bool = False,
) -> SimpleNamespace:
    """Install the optional-extra fakes and return recorders for what train() touched.

    ``parallelism`` controls whether the fake TrainingArguments signature carries
    a ``parallelism_config`` parameter. That switch is the whole instrument for
    the older-toolchain arm: ``_parallelism_backend`` decides availability by
    introspecting the signature of whatever ``transformers`` is importable, which
    inside these tests is this fake.
    """
    stack = SimpleNamespace(
        training_arguments=[],
        constructed=[],
        manifests=[],
        # head_counts: a test sets it and the fake model's config carries it;
        # None leaves the config bare. model_kwargs: from_pretrained records.
        head_counts=None,
        model_kwargs=None,
    )
    accepted = _BASE_ACCEPTED + (("parallelism_config",) if parallelism else ())

    class FakeTrainingArguments:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = dict(kwargs)
            for name, value in kwargs.items():
                setattr(self, name, value)
            # transformers derives fsdp_plugin_args from fsdp_config; the fake
            # models only the one key train() re-reads after construction.
            config = kwargs.get("fsdp_config")
            offload = isinstance(config, dict) and config.get("cpu_offload") is True
            self.fsdp_plugin_args = {"cpu_offload": True} if offload else {}
            stack.training_arguments.append(self)

    FakeTrainingArguments.__init__.__signature__ = inspect.Signature(  # type: ignore[attr-defined]
        parameters=[
            inspect.Parameter(name, inspect.Parameter.KEYWORD_ONLY, default=None)
            for name in accepted
        ]
    )

    stack.training_arguments_cls = FakeTrainingArguments

    class FakeTrainer:
        def __init__(self, *args: object, **kwargs: object) -> None:
            self.args = kwargs.get("args")
            self.model = kwargs.get("model") or SimpleNamespace(
                config=SimpleNamespace(use_cache=False)
            )
            self.state = SimpleNamespace(log_history=[])
            # accelerate's root mesh: only the dimensions above 1, canonical order.
            pc = getattr(self.args, "parallelism_config", None)
            if pc is not None:
                dims = [
                    (name, int(getattr(pc, f"{name}_size", None) or 1))
                    for name in ("dp_replicate", "dp_shard", "cp", "tp")
                ]
                live = [(n, v) for n, v in dims if v > 1]
                self.accelerator = SimpleNamespace(
                    torch_device_mesh=SimpleNamespace(
                        mesh_dim_names=tuple(n for n, _ in live),
                        mesh=SimpleNamespace(shape=tuple(v for _, v in live)),
                    )
                )
            stack.constructed.append((args, kwargs))

        def train(self, *args: object, **kwargs: object) -> None:
            return None

        def save_model(self, path: object) -> None:
            # Production's save_model WRITES and the loop inspects what landed,
            # globbing the artifact format rather than trusting a flag. A fixture
            # kinder than production tests the fixture.
            destination = Path(path)  # type: ignore[arg-type]
            destination.mkdir(parents=True, exist_ok=True)
            (destination / "model.safetensors").write_bytes((2).to_bytes(8, "little") + b"{}")

        def evaluate(self, *args: object, **kwargs: object) -> dict[str, float]:
            return {}

        def __getattr__(self, name: str):
            def _noop(*args: object, **kwargs: object) -> None:
                return None

            return _noop

    class _FakeSplit:
        column_names = ["text"]

        def __len__(self) -> int:
            return 3

        def map(self, *args: object, **kwargs: object) -> _FakeSplit:
            return self

        def select(self, *args: object, **kwargs: object) -> _FakeSplit:
            return self

    # A real dict, and no __getitem__ on the split: the thin path asks
    # ``"train" in raw``, and against an object with __getitem__ but no
    # __contains__ Python walks raw[0], raw[1], ... forever.
    datasets_mod = ModuleType("datasets")
    datasets_mod.load_dataset = lambda *args, **kwargs: {"train": _FakeSplit()}

    class _FakeDecoderLayer:
        """A block class that IS present in the fake model."""

    class _FakeModel(SimpleNamespace):
        # Faithful to the defect this fixture exists to catch: an omni
        # checkpoint declares block classes the causal-LM build never
        # instantiates. gemma-4-26B-A4B declares Gemma4AudioLayer, and
        # transformers' own auto_wrap resolution died on it at the first step,
        # on a tray, after the weights were resident. A fake that declared only
        # classes it contains could not tell a correct intersection from no
        # intersection at all.
        _no_split_modules = ["_FakeDecoderLayer", "_FakeAbsentAudioLayer"] if blocks else []

        def __init__(self) -> None:
            super().__init__(
                pad_token=None,
                eos_token="</s>",
                # Head counts land on the config only when a test declares them:
                # _tp_head_refusal skips absent fields (#540), so the default
                # None arm must leave the config bare rather than carrying Nones.
                config=SimpleNamespace(
                    use_cache=False,
                    tie_word_embeddings=tied,
                    **(stack.head_counts or {}),
                ),
                parameters=lambda: [],
                state_dict=lambda: {},
                to=lambda *a, **k: None,
            )
            self._blocks = [_FakeDecoderLayer(), _FakeDecoderLayer()] if blocks else []

        def modules(self):
            return [self, *self._blocks]

        def named_modules(self):
            return [("", self)] + [(f"model.layers.{i}", b) for i, b in enumerate(self._blocks)]

    class _Auto:
        @classmethod
        def from_pretrained(cls, *args: object, **kwargs: object) -> object:
            # Last write wins: the model load is the last thing train() asks
            # this surface for, so the stack reads back the load's kwargs.
            stack.model_kwargs = dict(kwargs)
            return _FakeModel()

    class _FakeCollator:
        def __init__(self, tokenizer: object = None, mlm: bool = False) -> None:
            self.tokenizer = tokenizer
            self.mlm = mlm

    torch_mod = ModuleType("torch")
    torch_mod.__version__ = "0.0+synthetic"
    torch_mod.manual_seed = lambda seed: None
    torch_mod.cuda = SimpleNamespace(is_available=lambda: False, device_count=lambda: 1)
    torch_mod.distributed = SimpleNamespace(
        is_available=lambda: False, is_initialized=lambda: False
    )
    torch_mod.utils = SimpleNamespace(data=SimpleNamespace(DataLoader=object))

    transformers_mod = ModuleType("transformers")
    transformers_mod.TrainingArguments = FakeTrainingArguments  # type: ignore[attr-defined]
    transformers_mod.Trainer = FakeTrainer  # type: ignore[attr-defined]
    stack.trainer_cls = FakeTrainer
    transformers_mod.AutoTokenizer = _Auto  # type: ignore[attr-defined]
    transformers_mod.AutoModelForCausalLM = _Auto  # type: ignore[attr-defined]
    transformers_mod.AutoConfig = _Auto  # type: ignore[attr-defined]
    transformers_mod.DataCollatorForLanguageModeling = _FakeCollator  # type: ignore[attr-defined]

    def _transformers_getattr(name: str) -> object:
        if name.startswith("Auto"):
            return _Auto
        raise AttributeError(
            f"the fake transformers module does not define {name!r}; the thin path "
            "reached a symbol this fixture never modelled -- add it here rather than "
            "returning a stub, so the next gap names itself too"
        )

    transformers_mod.__getattr__ = _transformers_getattr  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "torch", torch_mod)
    monkeypatch.setitem(sys.modules, "datasets", datasets_mod)
    monkeypatch.setitem(sys.modules, "transformers", transformers_mod)

    def _record_emit(
        cfg: object,
        stage: str | None = None,
        extra: dict[str, object] | None = None,
        **kwargs: object,
    ) -> None:
        stack.manifests.append(
            SimpleNamespace(cfg=cfg, stage=stage, extra=dict(extra or {}), kwargs=kwargs)
        )

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
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    monkeypatch.delenv("FS_RUN_ID", raising=False)
    return stack


def _refused_manifest(stack: SimpleNamespace) -> SimpleNamespace:
    refused = [manifest for manifest in stack.manifests if manifest.stage == "refused"]
    assert len(refused) == 1, f"expected exactly one refused manifest, got {len(refused)}"
    return refused[0]


def _only_training_arguments(stack: SimpleNamespace) -> SimpleNamespace:
    assert len(stack.training_arguments) == 1, (
        f"expected exactly one TrainingArguments, got {len(stack.training_arguments)}"
    )
    return stack.training_arguments[0]


def test_accepted_sharding_strategies_are_ddp_and_fsdp() -> None:
    """Pinned here so the accepted set cannot widen without a reviewer seeing it."""
    assert loop.SHARDING_STRATEGIES == ("ddp", "fsdp")


def test_fsdp_reaches_training_arguments(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """sharding_strategy=fsdp proceeds and ARRIVES as the fsdp kwarg."""
    stack = _install_fake_runtime(monkeypatch)
    cfg = loop.TrainConfig(**_base_kwargs(tmp_path), sharding_strategy="fsdp")

    rc = loop.train(cfg)

    assert rc in PROCEEDED_CODES, f"proceeding arm exited {rc}"
    constructed = _only_training_arguments(stack)
    # The legacy STRING form, not a bare True: the cluster toolchain is
    # transformers 5.13, which knows only the string, and 5.17 still parses it.
    assert constructed.kwargs["fsdp"] == "full_shard auto_wrap"
    # The wrap list is DERIVED from the loaded model, and the absent class the
    # fake declares must not appear: transformers resolves _no_split_modules by
    # class name and raises on a name the model does not contain.
    config = constructed.kwargs["fsdp_config"]
    assert config["transformer_layer_cls_to_wrap"] == ["_FakeDecoderLayer"]
    assert "cpu_offload" not in config


def test_ddp_and_omitted_pass_no_fsdp_key(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Replication is the absence of the kwarg, not fsdp=False."""
    for declared in ("ddp", None):
        stack = _install_fake_runtime(monkeypatch)
        kwargs: dict[str, object] = {}
        if declared is not None:
            kwargs["sharding_strategy"] = declared
        rc = loop.train(loop.TrainConfig(**_base_kwargs(tmp_path / str(declared)), **kwargs))
        assert rc in PROCEEDED_CODES
        constructed = _only_training_arguments(stack)
        assert "fsdp" not in constructed.kwargs, (
            f"sharding_strategy={declared!r} added an fsdp key; replication must "
            "not reach the FSDP integration at all"
        )


def test_zero3_still_refuses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """ZeRO remains adjudicated and unbuilt, so declaring it is still a refusal."""
    stack = _install_fake_runtime(monkeypatch)
    cfg = loop.TrainConfig(**_base_kwargs(tmp_path), sharding_strategy="zero3")

    rc = loop.train(cfg)
    out = capsys.readouterr().out

    assert "[fs:train:refuse]" in out
    assert "[fs:train:trainer]" not in out
    assert rc == loop.EXIT_REFUSE
    assert stack.constructed == []


def test_offload_under_fsdp_reaches_fsdp_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """cpu_optimizer_offload binds the offload inside FSDP, and only there."""
    stack = _install_fake_runtime(monkeypatch)
    cfg = loop.TrainConfig(
        **_base_kwargs(tmp_path), sharding_strategy="fsdp", cpu_optimizer_offload=True
    )

    rc = loop.train(cfg)

    assert rc in PROCEEDED_CODES
    constructed = _only_training_arguments(stack)
    assert constructed.kwargs["fsdp"] == "full_shard auto_wrap"
    assert constructed.kwargs["fsdp_config"] == {
        "transformer_layer_cls_to_wrap": ["_FakeDecoderLayer"],
        "cpu_offload": True,
    }


def test_offload_config_survives_the_real_transformers_parser(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The dict train() builds must mean offload to TRANSFORMERS, not to this file.

    The test above pins the dict; this one hands it to the real TrainingArguments
    and reads the plugin arguments transformers derives. The dict used to say
    "offload_params", which the parser drops silently, and a mirror assertion
    passed over it while two hardware runs trained un-offloaded (#538).
    """
    from transformers import TrainingArguments

    stack = _install_fake_runtime(monkeypatch)
    cfg = loop.TrainConfig(
        **_base_kwargs(tmp_path), sharding_strategy="fsdp", cpu_optimizer_offload=True
    )
    assert loop.train(cfg) in PROCEEDED_CODES
    built = _only_training_arguments(stack).kwargs
    monkeypatch.undo()
    real = TrainingArguments(
        output_dir=str(tmp_path / "real"),
        fsdp=built["fsdp"],
        fsdp_config=dict(built["fsdp_config"]),
        report_to=[],
    )
    assert real.fsdp_plugin_args.get("cpu_offload") is True


def test_offload_absent_from_built_plugin_args_refuses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """If transformers ever drops the key again, the run refuses instead of training."""
    stack = _install_fake_runtime(monkeypatch)
    real_init = stack.training_arguments_cls.__init__

    def dropping_init(self: object, **kwargs: object) -> None:
        real_init(self, **kwargs)
        self.fsdp_plugin_args = {}  # type: ignore[attr-defined]

    # _parallelism_backend introspects this signature; keep it.
    dropping_init.__signature__ = real_init.__signature__  # type: ignore[attr-defined]
    monkeypatch.setattr(stack.training_arguments_cls, "__init__", dropping_init)
    cfg = loop.TrainConfig(
        **_base_kwargs(tmp_path), sharding_strategy="fsdp", cpu_optimizer_offload=True
    )

    rc = loop.train(cfg)
    out = capsys.readouterr().out

    assert rc == loop.EXIT_REFUSE
    assert "cpu_offload" in out
    assert "[fs:train:trainer]" not in out


def test_offload_without_fsdp_refuses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Offload declared with no strategy at all: refused, naming what it saw."""
    stack = _install_fake_runtime(monkeypatch)
    cfg = loop.TrainConfig(**_base_kwargs(tmp_path), cpu_optimizer_offload=True)

    rc = loop.train(cfg)
    out = capsys.readouterr().out

    assert "[fs:train:refuse]" in out
    assert "fsdp" in out
    assert "[fs:train:trainer]" not in out
    assert rc == loop.EXIT_REFUSE
    assert stack.constructed == []
    manifest = _refused_manifest(stack)
    assert manifest.extra["cpu_optimizer_offload"] is True
    assert manifest.extra["sharding_strategy"] is None


def test_offload_with_ddp_refuses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Offload declared beside EXPLICIT replication is a different fact.

    Separate from the omitted arm because the two declarations differ and the
    message must name the strategy it actually saw; one test would pass on a
    refusal that printed the same sentence for both.
    """
    stack = _install_fake_runtime(monkeypatch)
    cfg = loop.TrainConfig(
        **_base_kwargs(tmp_path), sharding_strategy="ddp", cpu_optimizer_offload=True
    )

    rc = loop.train(cfg)
    out = capsys.readouterr().out

    assert "[fs:train:refuse]" in out
    assert "'ddp'" in out
    assert rc == loop.EXIT_REFUSE
    assert stack.constructed == []
    manifest = _refused_manifest(stack)
    assert manifest.extra["sharding_strategy"] == "ddp"


def test_tp_reaches_parallelism_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """tp=2 proceeds and arrives as tp_size on a real ParallelismConfig."""
    stack = _install_fake_runtime(monkeypatch)
    cfg = loop.TrainConfig(
        **_base_kwargs(tmp_path, gpus_per_node=2), tp=2, sharding_strategy="fsdp"
    )

    rc = loop.train(cfg)

    assert rc in PROCEEDED_CODES
    constructed = _only_training_arguments(stack)
    assert constructed.kwargs["parallelism_config"].tp_size == 2


def test_cp_reaches_parallelism_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """cp is a second, independently declarable dimension on the same backend."""
    stack = _install_fake_runtime(monkeypatch)
    cfg = loop.TrainConfig(**_base_kwargs(tmp_path, gpus_per_node=2), cp=2)

    rc = loop.train(cfg)

    assert rc in PROCEEDED_CODES
    constructed = _only_training_arguments(stack)
    assert constructed.kwargs["parallelism_config"].cp_size == 2


def test_tp_with_sharded_dp_sets_dp_shard_size(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """2D parallel: a declared dp beside tp becomes a SHARD, never a replica."""
    stack = _install_fake_runtime(monkeypatch)
    cfg = loop.TrainConfig(
        **_base_kwargs(tmp_path, gpus_per_node=4), tp=2, dp=2, sharding_strategy="fsdp"
    )

    rc = loop.train(cfg)

    assert rc in PROCEEDED_CODES
    parallelism = _only_training_arguments(stack).kwargs["parallelism_config"]
    assert parallelism.tp_size == 2
    assert parallelism.dp_shard_size == 2
    # Never a replica: accelerate rejects tp beside replicated data outright, so
    # a dp_replicate_size above 1 here would mean the wiring built a mesh the
    # backend refuses.
    assert parallelism.dp_replicate_size in (None, 1)


def test_tp_with_replicated_dp_refuses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The composition rule, answered 96 at START rather than RED at the backend.

    Left to reach accelerate this raises ValueError and the thin path adjudicates
    it RED (5) -- the code for a defect -- when what happened is an operator
    describing a mesh that does not exist. This test is what holds the check in
    front of the model load: remove it and the arm still "fails", but with the
    wrong code, after the allocation is burned, and naming a backend exception
    instead of the declaration.
    """
    stack = _install_fake_runtime(monkeypatch)
    cfg = loop.TrainConfig(**_base_kwargs(tmp_path, gpus_per_node=4), tp=2, dp=2)

    rc = loop.train(cfg)
    out = capsys.readouterr().out

    assert "[fs:train:refuse]" in out
    assert "[fs:train:trainer]" not in out
    assert rc == loop.EXIT_REFUSE, f"expected REFUSE 96, got {rc}"
    assert stack.constructed == []
    manifest = _refused_manifest(stack)
    assert manifest.extra["tp"] == 2
    assert manifest.extra["dp"] == 2


@pytest.mark.parametrize("degree", ["pp", "ep"])
def test_pipeline_and_expert_degrees_still_refuse(
    degree: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Named independently: a guard on the product would pass pp=2, ep=1."""
    stack = _install_fake_runtime(monkeypatch)
    cfg = loop.TrainConfig(**_base_kwargs(tmp_path / degree, gpus_per_node=2), **{degree: 2})

    rc = loop.train(cfg)
    out = capsys.readouterr().out

    assert "[fs:train:refuse]" in out
    assert f"{degree}=2" in out
    assert rc == loop.EXIT_REFUSE
    assert stack.constructed == []


def test_tp_refuses_when_the_backend_is_absent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """An older toolchain must refuse tp, not drop it.

    The kwarg-introspection guard further down would also catch an unknown
    parallelism_config, but it would do so after the allocation is burned and
    while naming the kwarg rather than the declaration the operator made. This
    arm proves the check happens at START instead.
    """
    stack = _install_fake_runtime(monkeypatch, parallelism=False)
    cfg = loop.TrainConfig(**_base_kwargs(tmp_path, gpus_per_node=2), tp=2)

    rc = loop.train(cfg)
    out = capsys.readouterr().out

    assert "[fs:train:refuse]" in out
    assert "parallelism_config" in out
    assert "[fs:train:trainer]" not in out
    assert rc == loop.EXIT_REFUSE
    assert stack.constructed == []


def test_fsdp_refuses_when_no_block_class_resolves(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A model with no resolvable blocks is refused, not wrapped at the root.

    An empty wrap list is not a harmless default: FSDP would shard the root
    module only, hand back most of the memory saving the declaration asked for,
    and record sharding_strategy=fsdp in the manifest while doing it. The
    failure this guards is silent in exactly the way a memory number is -- the
    run completes, and only the peak GiB says anything was wrong.
    """
    stack = _install_fake_runtime(monkeypatch, blocks=False)
    cfg = loop.TrainConfig(**_base_kwargs(tmp_path), sharding_strategy="fsdp")

    rc = loop.train(cfg)
    out = capsys.readouterr().out

    assert "[fs:train:refuse]" in out
    assert "no transformer block class" in out
    assert "[fs:train:trainer]" not in out
    assert rc == loop.EXIT_REFUSE
    assert stack.constructed == []


def test_tied_embeddings_select_fsdp_version_1(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A tied model pins FSDP version 1 and keeps per-layer wrapping.

    Measured on a tray, in order, for gemma-4-26B-A4B (tied, no lm_head tensor):
    FSDP2 + per-layer wrap and FSDP2 + NO_WRAP both refused the shared
    embedding/output tensor at the first step; FSDP1 + NO_WRAP accepted the tie
    and then OOMed on one 48 GiB flat buffer. FSDP1 + per-layer wrap trained.
    This pins that exact combination, and announces it.
    """
    stack = _install_fake_runtime(monkeypatch, tied=True)
    rc = loop.train(loop.TrainConfig(**_base_kwargs(tmp_path), sharding_strategy="fsdp"))
    out = capsys.readouterr().out

    assert rc in PROCEEDED_CODES
    kwargs = _only_training_arguments(stack).kwargs
    assert kwargs["fsdp"] == "full_shard auto_wrap"
    assert kwargs["fsdp_config"]["version"] == 1
    assert kwargs["fsdp_config"]["transformer_layer_cls_to_wrap"] == ["_FakeDecoderLayer"]
    assert "FSDP version 1 is pinned" in out


def test_untied_model_leaves_fsdp_version_to_the_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The version pin is caused by the tie and appears only with it."""
    stack = _install_fake_runtime(monkeypatch, tied=False)
    loop.train(loop.TrainConfig(**_base_kwargs(tmp_path), sharding_strategy="fsdp"))

    assert "version" not in _only_training_arguments(stack).kwargs["fsdp_config"]


def test_wrap_classes_fall_back_to_structure_when_declared_names_are_absent() -> None:
    """Real torch modules: a declared-but-absent class does not blank the wrap list.

    This is the Gemma4AudioLayer shape -- the model names a class it never
    instantiates -- with nothing valid to intersect, so the derivation must find
    the blocks by structure: the children of a ModuleList whose name ends in
    "layers". A fake could not exercise that walk; it needs real nn.Modules.
    """
    import torch.nn as nn

    class Block(nn.Module):
        pass

    class Model(nn.Module):
        _no_split_modules = ["Gemma4AudioLayer"]

        def __init__(self) -> None:
            super().__init__()
            self.model = nn.Module()
            self.model.layers = nn.ModuleList([Block(), Block()])

    assert loop._fsdp_wrap_classes(Model()) == ["Block"]


def test_wrap_classes_intersect_declared_names_with_present_ones() -> None:
    """Declared names that ARE present win over the structural walk."""
    import torch.nn as nn

    class DecoderLayer(nn.Module):
        pass

    class Model(nn.Module):
        _no_split_modules = ["DecoderLayer", "AbsentVisionLayer"]

        def __init__(self) -> None:
            super().__init__()
            self.layers = nn.ModuleList([DecoderLayer()])

    assert loop._fsdp_wrap_classes(Model()) == ["DecoderLayer"]


def test_wrap_classes_look_through_a_peft_wrapper() -> None:
    """A PEFT-wrapped model carries _no_split_modules on base_model.model."""
    import torch.nn as nn

    class DecoderLayer(nn.Module):
        pass

    class Inner(nn.Module):
        _no_split_modules = ["DecoderLayer"]

        def __init__(self) -> None:
            super().__init__()
            self.layers = nn.ModuleList([DecoderLayer()])

    class Wrapper(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.base_model = nn.Module()
            self.base_model.model = Inner()

    assert loop._fsdp_wrap_classes(Wrapper()) == ["DecoderLayer"]


def test_wrap_classes_are_empty_when_nothing_resolves() -> None:
    """No declared class present and no layers ModuleList: an empty list, which
    the caller refuses rather than wrapping the root alone."""
    import torch.nn as nn

    class Model(nn.Module):
        _no_split_modules = ["Nowhere"]

        def __init__(self) -> None:
            super().__init__()
            self.proj = nn.Linear(2, 2)

    assert loop._fsdp_wrap_classes(Model()) == []


def _torchrun_env(monkeypatch: pytest.MonkeyPatch, world: int) -> None:
    """The env torchrun sets, which is what routes train() into the comparison.

    The real Topology goes back in as well, since the comparison reads its fields
    and the fixture's stand-in carries none, over a profile wide enough to hold
    the ranks.
    """
    import dataclasses

    from foundationscale.topology import Topology

    monkeypatch.setattr(loop, "Topology", Topology)
    profile = dataclasses.replace(_SYNTHETIC_PROFILE, gpus_per_node=world)
    monkeypatch.setattr(loop, "_resolve_profile", lambda cfg: profile)
    for name, value in (
        ("WORLD_SIZE", world),
        ("LOCAL_WORLD_SIZE", world),
        ("RANK", 0),
        ("LOCAL_RANK", 0),
    ):
        monkeypatch.setenv(name, str(value))


def test_tp_mesh_under_torchrun_is_not_blocked_by_the_preload_pin(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """tp=2 dp=2 on 4 ranks proceeds, and the mesh is what the declaration is held to.

    The tests above run on the driver path, where WORLD_SIZE is unset and the
    declared-vs-effective comparison is skipped -- which is how a pre-load pin of
    tp=1 blocked every hardware tp run while every unit test passed (#539).
    """
    stack = _install_fake_runtime(monkeypatch)
    _torchrun_env(monkeypatch, 4)
    cfg = loop.TrainConfig(
        **_base_kwargs(tmp_path, gpus_per_node=4), tp=2, dp=2, sharding_strategy="fsdp"
    )

    rc = loop.train(cfg)
    out = capsys.readouterr().out

    assert rc in PROCEEDED_CODES, f"exited {rc}: {out[-2000:]}"
    assert "effective_overrides" not in out
    assert "device mesh measured {'tp': 2, 'cp': 1, 'dp': 2}" in out
    assert len(stack.constructed) == 1


def test_mesh_that_differs_from_the_declaration_refuses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Standing the pre-load check down is only honest if the mesh check can fire."""
    stack = _install_fake_runtime(monkeypatch)
    _torchrun_env(monkeypatch, 4)
    real_init = stack.trainer_cls.__init__

    def flattening_init(self: object, *args: object, **kwargs: object) -> None:
        real_init(self, *args, **kwargs)
        mesh = SimpleNamespace(mesh_dim_names=("dp_shard",), mesh=SimpleNamespace(shape=(4,)))
        self.accelerator = SimpleNamespace(torch_device_mesh=mesh)  # type: ignore[attr-defined]

    monkeypatch.setattr(stack.trainer_cls, "__init__", flattening_init)
    cfg = loop.TrainConfig(
        **_base_kwargs(tmp_path, gpus_per_node=4), tp=2, dp=2, sharding_strategy="fsdp"
    )

    rc = loop.train(cfg)
    out = capsys.readouterr().out

    assert rc == loop.EXIT_REFUSE
    assert "accelerate built {'tp': 1, 'cp': 1, 'dp': 4}" in out
    assert "training starts" not in out


def test_unreadable_mesh_refuses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """No mesh to read is a mismatch, not a pass: the pre-load check already stood down."""
    stack = _install_fake_runtime(monkeypatch)
    _torchrun_env(monkeypatch, 2)
    real_init = stack.trainer_cls.__init__

    def meshless_init(self: object, *args: object, **kwargs: object) -> None:
        real_init(self, *args, **kwargs)
        self.accelerator = SimpleNamespace(torch_device_mesh=None)  # type: ignore[attr-defined]

    monkeypatch.setattr(stack.trainer_cls, "__init__", meshless_init)
    cfg = loop.TrainConfig(**_base_kwargs(tmp_path, gpus_per_node=2), tp=2)

    rc = loop.train(cfg)

    assert rc == loop.EXIT_REFUSE
    assert "unreadable" in capsys.readouterr().out


def test_tp_plan_and_tp_size_reach_the_model_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """tp=2 hands the LOAD the plan, not only ParallelismConfig the degree.

    accelerate asserts model.tp_size == parallelism_config.tp_size inside
    train(); a model loaded without a tp plan has tp_size None, and that is
    what a 26B tp=4 run raised on hardware after every earlier stage passed
    (#540). "auto" binds the plan the model class ships.
    """
    stack = _install_fake_runtime(monkeypatch)
    _torchrun_env(monkeypatch, 2)
    cfg = loop.TrainConfig(
        **_base_kwargs(tmp_path, gpus_per_node=2), tp=2, sharding_strategy="fsdp"
    )

    rc = loop.train(cfg)

    assert rc in PROCEEDED_CODES
    assert len(stack.constructed) == 1
    assert stack.model_kwargs["tp_plan"] == "auto"
    assert stack.model_kwargs["tp_size"] == 2


def test_no_tp_passes_no_tp_kwargs_to_the_model_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """tp=1 is the absence of the kwargs, not tp_plan=None."""
    stack = _install_fake_runtime(monkeypatch)
    cfg = loop.TrainConfig(**_base_kwargs(tmp_path))

    rc = loop.train(cfg)

    assert rc in PROCEEDED_CODES
    assert "tp_plan" not in stack.model_kwargs
    assert "tp_size" not in stack.model_kwargs


def test_tp_that_does_not_divide_a_declared_head_count_refuses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The check the first forward used to perform, answered before the Trainer.

    Nothing downstream checked head divisibility: on the 26B (2 global KV
    heads) tp=4 loaded, built its mesh, and died in the first forward with a
    reshape error (#540). The refusal names the field and the count so the
    operator sees the declaration, not a backend traceback.
    """
    stack = _install_fake_runtime(monkeypatch)
    stack.head_counts = {"num_global_key_value_heads": 2}
    cfg = loop.TrainConfig(**_base_kwargs(tmp_path, gpus_per_node=4), tp=4)

    rc = loop.train(cfg)
    out = capsys.readouterr().out

    assert "[fs:train:refuse]" in out
    assert "num_global_key_value_heads=2" in out
    assert "[fs:train:trainer]" not in out
    assert rc == loop.EXIT_REFUSE
    assert stack.constructed == []
    manifest = _refused_manifest(stack)
    assert manifest.extra["tp"] == 4


def test_tied_model_with_tp_and_fsdp_refuses_before_the_trainer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """No FSDP version runs tp on a tied model, so refuse with the reason named.

    accelerate composes tp/cp with FSDP version 2 only, and a tied model needs
    version 1. Measured on the 26B: the Trainer refuses at construction with
    "ParallelismConfig is only compatible DistributedType.FSDP (version 2)".
    This arm proves the refusal fires before that construction is attempted
    (#540): proof it fired early is a trainer that was never built.
    """
    stack = _install_fake_runtime(monkeypatch, tied=True)
    cfg = loop.TrainConfig(
        **_base_kwargs(tmp_path, gpus_per_node=4), tp=2, dp=2, sharding_strategy="fsdp"
    )

    rc = loop.train(cfg)
    out = capsys.readouterr().out

    assert "[fs:train:refuse]" in out
    assert "FSDP version 2" in out
    assert "[fs:train:trainer]" not in out
    assert rc == loop.EXIT_REFUSE
    assert stack.constructed == []
    manifest = _refused_manifest(stack)
    assert manifest.extra["sharding_strategy"] == "fsdp"


def test_tp_head_refusal_is_none_when_every_declared_count_divides() -> None:
    """All three head fields declared and divisible: nothing to refuse."""
    model = SimpleNamespace(
        config=SimpleNamespace(
            num_attention_heads=8, num_key_value_heads=4, num_global_key_value_heads=2
        )
    )

    assert loop._tp_head_refusal(model, 2) is None


def test_tp_head_refusal_skips_fields_the_model_does_not_declare() -> None:
    """Only a DECLARED count can be violated; an absent field is not a zero."""
    assert loop._tp_head_refusal(SimpleNamespace(config=SimpleNamespace()), 4) is None


def test_tp_head_refusal_reads_through_text_config_nesting() -> None:
    """Multimodal configs nest the head counts under text_config."""
    config = SimpleNamespace(text_config=SimpleNamespace(num_global_key_value_heads=2))

    assert loop._tp_head_refusal(SimpleNamespace(config=config), 4) == (
        "tp=4 does not divide num_global_key_value_heads=2"
    )


def test_tp_without_fsdp_refuses_because_its_first_save_deadlocks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Pure tp trains on hardware and then hangs at the first checkpoint.

    Measured on the 26B at tp=2 (#540): ten steps, then transformers' Trainer
    called save_pretrained on the writing rank only, whose tensor-parallel
    gather is a collective the other rank never joins. A run that cannot save
    cannot be adjudicated, so it refuses before the Trainer is built.
    """
    stack = _install_fake_runtime(monkeypatch)
    _torchrun_env(monkeypatch, 2)
    cfg = loop.TrainConfig(**_base_kwargs(tmp_path, gpus_per_node=2), tp=2)

    rc = loop.train(cfg)
    out = capsys.readouterr().out

    assert rc == loop.EXIT_REFUSE
    assert "deadlocks" in out
    assert stack.constructed == []
    assert _refused_manifest(stack).extra["tp"] == 2
