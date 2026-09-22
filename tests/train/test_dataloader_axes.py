"""Pins for the dataloader declaration axes added after the input-pipeline measurement.

WHY THESE AXES EXIST: a single-GPU run of ``examples/train_tiny.py`` on a 144-core
host recorded ``perf_dataloader_stall_fraction = 0.549`` -- the GPU idle, waiting for
batches, for 55% of wall time -- while the loop's own host-budget advisory printed
"dataloader workers: 0". The advisory could see the stall and TrainConfig had no
declaration that could say anything about it. An MFU measured over that run describes
the input pipeline, not the kernel, and the manifest did not distinguish the cases.

CONTRACT, measured end-to-end on CPU with torch/transformers/datasets faked in
``sys.modules`` exactly where ``train()`` imports them:

  DECLARED  -- the kwarg reaches the TrainingArguments object the run ACTUALLY
               constructed. It is read back off that captured instance, never off the
               TrainConfig this test assembled: a comparator re-reading its own input
               cannot disagree with itself (#291, #294, #372).
  OMITTED   -- NO key is added to the TrainingArguments kwargs at all. None means NOT
               DECLARED (#342); coercing the omission into the engine default would
               launder an abstention into the appearance of a statement.
  REFUSED   -- prefetch declared while workers is absent or explicitly 0 exits
               EXIT_REFUSE before any Trainer exists, because torch's DataLoader
               rejects prefetch_factor when num_workers == 0 at the FIRST BATCH, after
               the model is resident and the allocation is burned.

Every test asserts REACHED-SITE separately from OUTCOME. An exit code is not a site:
the refused arms must show ``[fs:train:refuse]`` and no constructed Trainer, and the
proceed arms must show ``[fs:train:trainer]`` with exactly one construction.
"""

from __future__ import annotations

import inspect
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from foundationscale.topology import ClusterProfile
from foundationscale.train import loop

# A REAL ClusterProfile, constructed rather than looked up: which profiles the package
# ships is irrelevant here, and construction fails at collection if a new required
# field appears. nccl_socket_ifname is deliberately blank so apply_fabric_declaration
# takes the "declares no interface" branch and leaves os.environ untouched for every
# module collected after this one.
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

# What a PROCEEDING arm is allowed to exit with. The loop's contract has exactly four
# states -- 0 PASS, 5 RED, 95 UNMEASURED, 96 REFUSE -- so "proceeded" means landing on
# one of the two that are neither a refusal nor a defect. A fully faked run reaches 95
# rather than 0 by design: no real step was taken, so the outcome is genuinely not
# measured, and this module's subject is the wiring, not the training.
#
# Every name is read off loop directly, with NO getattr default. This set was first
# written as getattr(loop, "EXIT_OK", 0) / "EXIT_DEPS", 97 / "EXIT_CONFIG", 2 -- three
# names that loop does not define, so all three fell through to invented literals. The
# set that resulted admitted 2 and 97, which the contract forbids outright, while
# excluding the 95 every proceeding arm here actually returns. A default on a constant
# lookup converts "this name is gone" into "this name is whatever I guessed", which is
# the failure the four-state contract exists to prevent.
PROCEEDED_CODES = frozenset({loop.EXIT_PASS, loop.EXIT_UNMEASURED})

# What the fake TrainingArguments accepts by NAME. loop.py introspects this signature
# and refuses any kwarg it does not contain, so the proceed legs need every kwarg the
# thin path can pass -- including both new dataloader axes -- while the omitted leg
# still proves absence by inspecting the captured kwargs dict.
_FAKE_ACCEPTED_KWARGS: tuple[str, ...] = (
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
    "ddp_backend",
    "local_rank",
    "deepspeed",
    "fsdp",
)


def _base_kwargs(tmp_path: Path) -> dict[str, object]:
    """Minimal real TrainConfig kwargs, spelled out the way the sibling modules spell them.

    Written as a literal dict rather than derived by introspecting which parameters
    lack a default, which is the shape this file was first drafted with. That
    derivation cannot construct a legal TrainConfig: the profile requirement is a
    CROSS-FIELD invariant checked in __post_init__ -- exactly one of profile /
    profile_path / profile_name -- and all three carry a None default, so a
    "fill every parameter that has no default" rule fills none of them and every
    test dies in the constructor. A filler keyed on substrings of the field NAME is
    also the wrong instrument for a second reason: it silently invents a plausible
    value for any future required field, so the field arrives untested instead of
    failing loudly here.

    profile_name is a placeholder, as in the sibling: the refused arms exit before
    profile resolution runs, and the proceeding arms have _resolve_profile patched.
    """
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


def _install_fake_runtime(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """Install the optional-extra fakes and return recorders for what train() touched."""
    stack = SimpleNamespace(training_arguments=[], constructed=[], manifests=[])

    class FakeTrainingArguments:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = dict(kwargs)
            for name, value in kwargs.items():
                setattr(self, name, value)
            self.dataloader_num_workers = kwargs.get("dataloader_num_workers")
            self.dataloader_prefetch_factor = kwargs.get("dataloader_prefetch_factor")
            stack.training_arguments.append(self)

    FakeTrainingArguments.__init__.__signature__ = inspect.Signature(  # type: ignore[attr-defined]
        parameters=[
            inspect.Parameter(name, inspect.Parameter.KEYWORD_ONLY, default=None)
            for name in _FAKE_ACCEPTED_KWARGS
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
            # Production's save_model WRITES, and the loop inspects what landed --
            # it globs the artifact format rather than trusting a construction-time
            # flag, so a save_model that returns None without creating the directory
            # makes the loop die listing a path that does not exist. A fixture
            # kinder or lazier than production tests the fixture.
            destination = Path(path)  # type: ignore[arg-type]
            destination.mkdir(parents=True, exist_ok=True)
            # A minimal safetensors shard: an 8-byte little-endian header length
            # followed by an empty JSON header.
            (destination / "model.safetensors").write_bytes((2).to_bytes(8, "little") + b"{}")

        def evaluate(self, *args: object, **kwargs: object) -> dict[str, float]:
            return {}

        def __getattr__(self, name: str):
            def _noop(*args: object, **kwargs: object) -> None:
                return None

            return _noop

    class _FakeSplit:
        # column_names must contain "text": the thin path REFUSES a split without
        # it, and a refusal here would be a proceed arm that never reached the
        # wiring it exists to measure.
        column_names = ["text"]

        def __len__(self) -> int:
            return 3

        def map(self, *args: object, **kwargs: object) -> _FakeSplit:
            return self

        def select(self, *args: object, **kwargs: object) -> _FakeSplit:
            return self

    # NO __getitem__ on the split, and load_dataset returns a real dict rather than
    # a single dataset object. This is not stylistic. The thin path asks
    # ``"train" in raw``; against an object that defines __getitem__ but no
    # __contains__, Python falls back to the old iteration protocol and walks
    # raw[0], raw[1], ... comparing each to "train". An index-ignoring __getitem__
    # never raises IndexError, so that membership test never terminates -- the
    # suite hangs with no failure and no output. load_dataset really does return a
    # split-keyed mapping, so the dict is also the accurate fake.
    datasets_mod = ModuleType("datasets")
    datasets_mod.load_dataset = lambda *args, **kwargs: {"train": _FakeSplit()}

    class _Auto:
        @classmethod
        def from_pretrained(cls, *args: object, **kwargs: object) -> SimpleNamespace:
            # state_dict() -> {} is load-bearing, not padding: without it the
            # checkpoint declaration cannot be derived and the run prints that
            # "every checkpoint gate will fail closed on UNKNOWN". That noise has
            # nothing to do with the dataloader axes, and a fake that provokes it
            # invites a future reader to debug the fixture instead of the subject.
            # The empty dict is the honest dense branch: no expert keys, no
            # expert-named tensors, both sources agreeing.
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
    # The thin path seeds torch before building anything. A module object raises
    # AttributeError for names it does not define, which is what this fake wants:
    # an unfaked call fails loudly at the line that made it rather than returning a
    # plausible stub, and train() adjudicates that escape RED (5).
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
        # Auto* is a family with one shape, so one stub answers for all of them.
        # Everything else raises AttributeError deliberately, which train()
        # adjudicates RED (5) naming the symbol. The blanket `return
        # SimpleNamespace()` this started with was worse than no fake at all: the
        # thin path calls DataCollatorForLanguageModeling(...), a SimpleNamespace
        # instance is not callable, and the module died with "'types.SimpleNamespace'
        # object is not callable" -- an error naming the fake rather than the symbol
        # the fake was missing. A stub that answers for every name cannot say which
        # name it failed to model.
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

    # Topology and profile resolution are replaced wholesale, as in the sibling
    # modules, rather than by monkeypatching methods onto the real Topology. Both
    # are out of scope here -- each has its own suite -- and patching a method onto
    # the real class leaves the rest of the real class live, so an unrelated change
    # in it turns this module RED about something it makes no claim on.
    #
    # _resolve_profile MUST be patched, not merely the topology: _base_kwargs names
    # a profile that the registry does not ship, deliberately, so that this module
    # does not move when the shipped registry does. Without the patch every
    # PROCEEDING arm dies in resolution with an unknown-profile refusal and never
    # reaches TrainingArguments -- which is a site-not-reached failure wearing the
    # costume of a wiring failure.
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


def test_prefetch_without_workers_refuses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Prefetch declared, workers absent: REFUSE 96 and record the rejected pair."""
    stack = _install_fake_runtime(monkeypatch)
    cfg = loop.TrainConfig(**_base_kwargs(tmp_path), dataloader_prefetch_factor=2)

    rc = loop.train(cfg)
    out = capsys.readouterr().out

    # REACHED-SITE
    assert "[fs:train:refuse]" in out
    assert "--dataloader-num-workers" in out
    assert "[fs:train:trainer]" not in out

    # OUTCOME
    assert rc == loop.EXIT_REFUSE
    assert stack.constructed == [], "a refused run constructed a Trainer"
    manifest = _refused_manifest(stack)
    assert manifest.extra["dataloader_prefetch_factor"] == 2
    assert "dataloader_num_workers" in manifest.extra
    assert manifest.extra["dataloader_num_workers"] is None


def test_prefetch_with_explicit_zero_workers_refuses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Prefetch declared with workers EXPLICITLY 0: same refusal, different fact."""
    stack = _install_fake_runtime(monkeypatch)
    cfg = loop.TrainConfig(
        **_base_kwargs(tmp_path),
        dataloader_num_workers=0,
        dataloader_prefetch_factor=2,
    )

    rc = loop.train(cfg)
    out = capsys.readouterr().out

    # REACHED-SITE. The "0" wording is the whole point of this second test: without it,
    # test 1 would pass on a refusal that printed one sentence for both causes and could
    # not tell an explicit 0 from no declaration at all.
    assert "[fs:train:refuse]" in out
    assert "0" in out
    assert "not declared" not in out
    assert "[fs:train:trainer]" not in out

    # OUTCOME
    assert rc == loop.EXIT_REFUSE
    assert stack.constructed == [], "a refused run constructed a Trainer"
    manifest = _refused_manifest(stack)
    assert manifest.extra["dataloader_prefetch_factor"] == 2
    assert manifest.extra["dataloader_num_workers"] == 0


def test_declared_workers_reach_training_arguments(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Workers declared alone proceeds and ARRIVES on the constructed TrainingArguments."""
    stack = _install_fake_runtime(monkeypatch)
    cfg = loop.TrainConfig(**_base_kwargs(tmp_path), dataloader_num_workers=2)

    rc = loop.train(cfg)
    out = capsys.readouterr().out

    # REACHED-SITE
    assert "[fs:train:trainer]" in out
    assert "[fs:train:refuse]" not in out

    # OUTCOME
    assert rc in PROCEEDED_CODES, f"proceeding arm exited {rc}, not a non-refused contract state"
    assert rc != loop.EXIT_REFUSE
    assert len(stack.training_arguments) == 1
    constructed = stack.training_arguments[0]
    assert constructed.kwargs["dataloader_num_workers"] == 2
    assert constructed.dataloader_num_workers == 2
    assert "dataloader_prefetch_factor" not in constructed.kwargs


def test_declared_prefetch_and_workers_both_reach_training_arguments(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The honourable pair proceeds and BOTH values arrive on TrainingArguments."""
    stack = _install_fake_runtime(monkeypatch)
    cfg = loop.TrainConfig(
        **_base_kwargs(tmp_path),
        dataloader_num_workers=2,
        dataloader_prefetch_factor=4,
    )

    rc = loop.train(cfg)
    out = capsys.readouterr().out

    # REACHED-SITE
    assert "[fs:train:trainer]" in out
    assert "[fs:train:refuse]" not in out

    # OUTCOME
    assert rc in PROCEEDED_CODES, f"proceeding arm exited {rc}, not a non-refused contract state"
    assert rc != loop.EXIT_REFUSE
    assert len(stack.training_arguments) == 1
    constructed = stack.training_arguments[0]
    assert constructed.kwargs["dataloader_num_workers"] == 2
    assert constructed.kwargs["dataloader_prefetch_factor"] == 4
    assert constructed.dataloader_num_workers == 2
    assert constructed.dataloader_prefetch_factor == 4


def test_undeclared_axes_pass_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Neither axis declared: proceed, and add NEITHER key to TrainingArguments kwargs."""
    stack = _install_fake_runtime(monkeypatch)
    cfg = loop.TrainConfig(**_base_kwargs(tmp_path))

    rc = loop.train(cfg)
    out = capsys.readouterr().out

    # REACHED-SITE
    assert "[fs:train:trainer]" in out
    assert "[fs:train:refuse]" not in out

    # OUTCOME. Asserting only that the run succeeded would NOT catch a regression that
    # passes the engine defaults explicitly: presence-with-default-value is
    # indistinguishable from a declaration, so the abstention rule is key ABSENCE.
    assert rc in PROCEEDED_CODES, f"proceeding arm exited {rc}, not a non-refused contract state"
    assert rc != loop.EXIT_REFUSE
    assert len(stack.training_arguments) == 1
    constructed = stack.training_arguments[0]
    assert "dataloader_num_workers" not in constructed.kwargs
    assert "dataloader_prefetch_factor" not in constructed.kwargs
