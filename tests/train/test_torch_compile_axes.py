"""Pins for the three torch.compile declaration axes.

WHY THESE AXES EXIST: compilation moves both numbers a throughput claim rests on,
in opposite directions, and neither move was recordable. Measured on a GB200 tray,
gemma-4-E4B at seq 2048, per-rank batch 2: compile off 403.4 ms/step at 106.43 GiB,
compile on 305.2 ms at 79.80 GiB -- 1.32x faster on 25 GiB less. On the 12B dense
checkpoint the eager arm did not fit at all at the same batch, so compilation was
not an optimisation there but the difference between a run and an OOM. Two runs
whose manifests were byte-identical could differ by a third in throughput, and
nothing in either manifest said why.

CONTRACT, measured end-to-end on CPU with torch/transformers/datasets faked in
``sys.modules`` exactly where ``train()`` imports them:

  DECLARED  -- the kwarg reaches the TrainingArguments object the run ACTUALLY
               constructed, read back off that captured instance rather than off
               the TrainConfig this module assembled (#291, #294, #372).
  OMITTED   -- NO key is added to the TrainingArguments kwargs at all. None means
               NOT DECLARED (#342).
  REFUSED   -- backend or mode declared while torch_compile is anything other
               than True exits EXIT_REFUSE before any Trainer exists.

THE REFUSAL IS NOT A STYLE RULE. transformers' TrainingArguments sets
``torch_compile = True`` whenever backend or mode is present. So a run declaring a
backend alone COMPILES, and the manifest records torch_compile as whatever the
operator left it as. Two rejected states, and the second is the worse one:

  torch_compile omitted  -> manifest says None, the abstention reading
  torch_compile = false  -> manifest makes a POSITIVE claim that contradicts the run

An abstention that turns out to be wrong is a gap. A false statement is a defect,
and it is the one a downstream reader cannot detect, because nothing else in the
artifact disagrees with it. Both are refused; the message says which, and the pair
of tests below is what keeps the two causes distinguishable -- a single refusal
that printed one sentence for both would pass a test written against either.

Every test asserts REACHED-SITE separately from OUTCOME. An exit code is not a
site: the refused arms must show ``[fs:train:refuse]`` with no constructed
Trainer, and the proceed arms must show ``[fs:train:trainer]``.
"""

from __future__ import annotations

import inspect
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from foundationscale.topology import ClusterProfile
from foundationscale.train import loop

# A REAL ClusterProfile, constructed rather than looked up: which profiles the
# package ships is irrelevant here, and construction fails at collection if a new
# required field appears. nccl_socket_ifname is deliberately blank so
# apply_fabric_declaration takes the "declares no interface" branch and leaves
# os.environ untouched for every module collected after this one.
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

# What a PROCEEDING arm is allowed to exit with. The contract has exactly four
# states -- 0 PASS, 5 RED, 95 UNMEASURED, 96 REFUSE -- so "proceeded" means
# landing on one of the two that are neither a refusal nor a defect. A fully faked
# run reaches 95 rather than 0 by design: no real step was taken, so the outcome
# is genuinely not measured, and this module's subject is the wiring.
PROCEEDED_CODES = frozenset({loop.EXIT_PASS, loop.EXIT_UNMEASURED})

# What the fake TrainingArguments accepts by NAME. loop.py introspects this
# signature and refuses any kwarg it does not contain, so the proceed legs need
# every kwarg the thin path can pass -- including all three compile axes -- while
# the omitted leg still proves absence by inspecting the captured kwargs dict.
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
    "torch_compile",
    "torch_compile_backend",
    "torch_compile_mode",
    "ddp_backend",
    "local_rank",
    "deepspeed",
    "fsdp",
)


def _base_kwargs(tmp_path: Path) -> dict[str, object]:
    """Minimal real TrainConfig kwargs, spelled as the sibling modules spell them.

    A literal dict rather than a derivation over "parameters without a default":
    the profile requirement is a CROSS-FIELD invariant checked in __post_init__ --
    exactly one of profile / profile_path / profile_name -- and all three default
    to None, so that rule fills none of them and every test dies in the
    constructor. A name-substring filler is wrong for a second reason: it invents
    a plausible value for any future required field, so the field arrives untested
    instead of failing loudly here.
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
            # flag, so a save_model that returns None without creating the
            # directory makes the loop die listing a path that does not exist. A
            # fixture kinder than production tests the fixture.
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
    # a single dataset object. The thin path asks ``"train" in raw``; against an
    # object defining __getitem__ but no __contains__, Python falls back to the old
    # iteration protocol and walks raw[0], raw[1], ... comparing each to "train".
    # An index-ignoring __getitem__ never raises IndexError, so that membership
    # test never terminates and the suite hangs with no failure and no output.
    datasets_mod = ModuleType("datasets")
    datasets_mod.load_dataset = lambda *args, **kwargs: {"train": _FakeSplit()}

    class _Auto:
        @classmethod
        def from_pretrained(cls, *args: object, **kwargs: object) -> SimpleNamespace:
            # state_dict() -> {} is load-bearing: without it the checkpoint
            # declaration cannot be derived and the run prints that every
            # checkpoint gate will fail closed on UNKNOWN. That noise has nothing
            # to do with the compile axes, and a fake that provokes it invites a
            # reader to debug the fixture instead of the subject.
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
    # A module object raises AttributeError for names it does not define, which is
    # what this fake wants: an unfaked call fails loudly at the line that made it
    # rather than returning a plausible stub, and train() adjudicates that RED (5).
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
        # adjudicates RED (5) naming the symbol. A blanket stub cannot say which
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

    # Topology and profile resolution are replaced wholesale rather than by
    # monkeypatching methods onto the real Topology: both are out of scope here,
    # each has its own suite, and patching a method onto the real class leaves the
    # rest of it live, so an unrelated change turns this module RED about
    # something it makes no claim on. _resolve_profile MUST be patched, not merely
    # the topology: _base_kwargs names a profile the registry does not ship, so
    # that this module does not move when the shipped registry does.
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


def test_all_three_compile_axes_are_declared() -> None:
    """The tuple names them, so the manifest and the drift gate both cover them."""
    for axis in ("torch_compile", "torch_compile_backend", "torch_compile_mode"):
        assert axis in loop.DECLARATION_AXES, f"{axis} is wired but not declared"


def test_backend_without_compile_refuses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Backend declared, torch_compile absent: REFUSE 96 and record the rejected pair."""
    stack = _install_fake_runtime(monkeypatch)
    cfg = loop.TrainConfig(**_base_kwargs(tmp_path), torch_compile_backend="inductor")

    rc = loop.train(cfg)
    out = capsys.readouterr().out

    # REACHED-SITE
    assert "[fs:train:refuse]" in out
    assert "--torch-compile" in out
    assert "not declared" in out
    assert "[fs:train:trainer]" not in out

    # OUTCOME
    assert rc == loop.EXIT_REFUSE
    assert stack.constructed == [], "a refused run constructed a Trainer"
    manifest = _refused_manifest(stack)
    assert manifest.extra["torch_compile_backend"] == "inductor"
    assert "torch_compile" in manifest.extra
    assert manifest.extra["torch_compile"] is None


def test_mode_without_compile_refuses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Mode is a second, independently declarable trigger and is refused on its own.

    Separate from the backend arm because either flag alone flips compilation on.
    A single test against the backend would pass while mode-alone compiled
    silently -- the two are the same defect reached through different doors.
    """
    stack = _install_fake_runtime(monkeypatch)
    cfg = loop.TrainConfig(**_base_kwargs(tmp_path), torch_compile_mode="max-autotune")

    rc = loop.train(cfg)
    out = capsys.readouterr().out

    # REACHED-SITE. The message must name the axis that was actually declared and
    # must NOT name the one that was not, or it cannot tell the two doors apart.
    assert "[fs:train:refuse]" in out
    assert "torch_compile_mode" in out
    assert "torch_compile_backend" not in out
    assert "[fs:train:trainer]" not in out

    # OUTCOME
    assert rc == loop.EXIT_REFUSE
    assert stack.constructed == []
    manifest = _refused_manifest(stack)
    assert manifest.extra["torch_compile_mode"] == "max-autotune"
    assert manifest.extra["torch_compile_backend"] is None


def test_explicit_false_with_backend_refuses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The worse of the two rejected states: a manifest that CONTRADICTS the run.

    An omitted torch_compile records None and the artifact merely fails to say.
    An explicit false records a positive claim that the run did not compile, while
    transformers compiles it anyway. The guard is therefore ``is not True`` rather
    than ``is None``, and this test is what holds it there: written as ``is None``
    the loop proceeds here, ships the false claim, and the other two tests stay
    green.
    """
    stack = _install_fake_runtime(monkeypatch)
    cfg = loop.TrainConfig(
        **_base_kwargs(tmp_path),
        torch_compile=False,
        torch_compile_backend="inductor",
    )

    rc = loop.train(cfg)
    out = capsys.readouterr().out

    # REACHED-SITE. "explicitly false" is the whole point: without it this arm and
    # the omitted arm would print one sentence for two different facts.
    assert "[fs:train:refuse]" in out
    assert "explicitly false" in out
    assert "[fs:train:trainer]" not in out

    # OUTCOME
    assert rc == loop.EXIT_REFUSE
    assert stack.constructed == []
    manifest = _refused_manifest(stack)
    assert manifest.extra["torch_compile"] is False


def test_declared_compile_reaches_training_arguments(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """torch_compile alone proceeds and ARRIVES on the constructed TrainingArguments."""
    stack = _install_fake_runtime(monkeypatch)
    cfg = loop.TrainConfig(**_base_kwargs(tmp_path), torch_compile=True)

    rc = loop.train(cfg)
    out = capsys.readouterr().out

    # REACHED-SITE
    assert "[fs:train:trainer]" in out
    assert "[fs:train:refuse]" not in out

    # OUTCOME
    assert rc in PROCEEDED_CODES, f"proceeding arm exited {rc}, not a non-refused state"
    assert len(stack.training_arguments) == 1
    constructed = stack.training_arguments[0]
    assert constructed.kwargs["torch_compile"] is True
    assert constructed.torch_compile is True
    assert "torch_compile_backend" not in constructed.kwargs
    assert "torch_compile_mode" not in constructed.kwargs


def test_declared_trio_reaches_training_arguments(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The honourable triple proceeds and all THREE values arrive."""
    stack = _install_fake_runtime(monkeypatch)
    cfg = loop.TrainConfig(
        **_base_kwargs(tmp_path),
        torch_compile=True,
        torch_compile_backend="inductor",
        torch_compile_mode="default",
    )

    rc = loop.train(cfg)
    out = capsys.readouterr().out

    # REACHED-SITE
    assert "[fs:train:trainer]" in out
    assert "[fs:train:refuse]" not in out

    # OUTCOME
    assert rc in PROCEEDED_CODES
    constructed = stack.training_arguments[0]
    assert constructed.kwargs["torch_compile"] is True
    assert constructed.kwargs["torch_compile_backend"] == "inductor"
    assert constructed.kwargs["torch_compile_mode"] == "default"


def test_explicit_false_alone_is_declared_not_omitted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """An explicit false is a STATEMENT and must reach TrainingArguments as one.

    Without a backend or mode beside it there is nothing for transformers to flip,
    so this is the legitimate way to say "this run did not compile" -- and the key
    has to be present and False, not absent. Coercing it to an omission would
    launder a statement into an abstention, the #342 defect read backwards.
    """
    stack = _install_fake_runtime(monkeypatch)
    cfg = loop.TrainConfig(**_base_kwargs(tmp_path), torch_compile=False)

    rc = loop.train(cfg)

    assert rc in PROCEEDED_CODES
    constructed = stack.training_arguments[0]
    assert "torch_compile" in constructed.kwargs
    assert constructed.kwargs["torch_compile"] is False


def test_undeclared_compile_axes_pass_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """None means NOT DECLARED: no key reaches TrainingArguments at all."""
    stack = _install_fake_runtime(monkeypatch)
    cfg = loop.TrainConfig(**_base_kwargs(tmp_path))

    rc = loop.train(cfg)

    assert rc in PROCEEDED_CODES
    constructed = stack.training_arguments[0]
    for axis in ("torch_compile", "torch_compile_backend", "torch_compile_mode"):
        assert axis not in constructed.kwargs, (
            f"{axis} was undeclared but a key reached TrainingArguments; an "
            "abstention was laundered into the engine default"
        )
