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
    monkeypatch: pytest.MonkeyPatch, *, parallelism: bool = True
) -> SimpleNamespace:
    """Install the optional-extra fakes and return recorders for what train() touched.

    ``parallelism`` controls whether the fake TrainingArguments signature carries
    a ``parallelism_config`` parameter. That switch is the whole instrument for
    the older-toolchain arm: ``_parallelism_backend`` decides availability by
    introspecting the signature of whatever ``transformers`` is importable, which
    inside these tests is this fake.
    """
    stack = SimpleNamespace(training_arguments=[], constructed=[], manifests=[])
    accepted = _BASE_ACCEPTED + (("parallelism_config",) if parallelism else ())

    class FakeTrainingArguments:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = dict(kwargs)
            for name, value in kwargs.items():
                setattr(self, name, value)
            stack.training_arguments.append(self)

    FakeTrainingArguments.__init__.__signature__ = inspect.Signature(  # type: ignore[attr-defined]
        parameters=[
            inspect.Parameter(name, inspect.Parameter.KEYWORD_ONLY, default=None)
            for name in accepted
        ]
    )

    class FakeTrainer:
        def __init__(self, *args: object, **kwargs: object) -> None:
            self.args = kwargs.get("args")
            self.model = kwargs.get("model") or SimpleNamespace(
                config=SimpleNamespace(use_cache=False)
            )
            self.state = SimpleNamespace(log_history=[])
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

    class _Auto:
        @classmethod
        def from_pretrained(cls, *args: object, **kwargs: object) -> SimpleNamespace:
            return SimpleNamespace(
                pad_token=None,
                eos_token="</s>",
                config=SimpleNamespace(use_cache=False),
                parameters=lambda: [],
                state_dict=lambda: {},
                to=lambda *a, **k: None,
            )

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
    assert "fsdp_config" not in constructed.kwargs


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
    assert constructed.kwargs["fsdp_config"] == {"offload_params": True}


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
    cfg = loop.TrainConfig(**_base_kwargs(tmp_path, gpus_per_node=2), tp=2)

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
