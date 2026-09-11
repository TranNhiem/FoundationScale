"""Pins for three train-plane declaration axes: optimizer,
gradient_accumulation_steps, max_grad_norm.

Each axis follows ONE contract, measured end-to-end here on CPU, in seconds,
without torch/transformers/datasets installed -- the loop imports the optional
extra INSIDE train(), so fake modules injected below reach exactly the import
sites a real run uses:

  DECLARED  -- the kwarg reaches the TrainingArguments object the run
               ACTUALLY constructed. It is introspected off the captured
               instance, never re-read from a dict this test assembled: a
               test that checks its own input is the tautology this campaign
               already shipped three times (#291, #294, #372).
  OMITTED   -- NO kwarg is added at all, so the transformers engine default
               applies and nothing is claimed. Coercing an omission into the
               default and recording THAT would launder an abstention into a
               measured statement (#342). Asserted as key ABSENCE from the
               constructed kwargs, because presence-with-default-value is
               indistinguishable from a declaration.
  RECORDED  -- the run manifest carries {key, value, source} in BOTH cases.
               The omitted case is an explicit absence (key present, value
               "None"), distinguishable from a key this loop never populated
               (key absent).

MUST-FIRE: an optimizer name outside transformers' accepted vocabulary is a
REJECTED DECLARATION. The fake TrainingArguments below plays authority over
its own accepted vocabulary, standing in for transformers -- and train() must
turn that rejection into REFUSE (96) with NO Trainer constructed, never a
traceback (exit 1 is in none of the four declared states, #171's shape). If
the (ValueError, TypeError) refusal arm in train() were deleted, this test
comes out differently: that differential is the point of the arm.
"""

from __future__ import annotations

import inspect
import json
import struct
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from foundationscale.train import loop

# What our fake TrainingArguments accepts by NAME. Mirrors what real
# transformers accepts for the thin path; loop.py introspects this signature
# and REFUSES on any kwarg it does not contain, so the fake must be complete
# for the legs that are supposed to reach construction -- and must NOT contain
# "optimizer", so a regression that wires the CLI name instead of the engine
# name ("optim") is REFUSED before construction and caught by the declared-arm
# assertions below rather than silently recorded.
_FAKE_ACCEPTED_KWARGS: tuple[str, ...] = (
    "output_dir",
    "max_steps",
    "per_device_train_batch_size",
    "learning_rate",
    "save_strategy",
    "save_steps",
    "seed",
    "logging_steps",
    "report_to",
    "ddp_find_unused_parameters",
    "bf16",
    "fp16",
    "optim",
    "gradient_accumulation_steps",
    "max_grad_norm",
    "gradient_checkpointing",
    "lr_scheduler_type",
    "warmup_steps",
)
# The fake's optim vocabulary: the set over which it plays transformers'
# authority. "sgd" is in it so the DECLARED arm constructs successfully;
# "not_a_real_optimizer_name" is deliberately out of it for the MUST-FIRE arm.
_FAKE_OPTIMS: tuple[str, ...] = (
    "adamw_torch",
    "adamw_torch_fused",
    "adafactor",
    "sgd",
)


def _install_fake_stack(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """Inject fake torch/transformers/datasets and stub the machine-fact plane.

    Returns a recorders namespace. Everything stateful is built fresh inside
    this function so each test gets empty recorder lists -- a cross-test
    recorder would make REACHED-SITE assertions unfalsifiable.
    """
    stack = SimpleNamespace(training_arguments=[], trainer_instances=[], load_calls=[])

    class FakeTrainingArguments:
        def __init__(self, **kwargs: object) -> None:
            optim = kwargs.get("optim")
            if optim is not None and optim not in _FAKE_OPTIMS:
                # Real transformers raises ValueError from
                # TrainingArguments.__post_init__ for an unknown optim name.
                # train() must convert exactly this into REFUSE (96).
                raise ValueError(
                    f"{optim!r} is not a valid optimizer name; "
                    f"accepted values are {sorted(_FAKE_OPTIMS)}"
                )
            self.received_kwargs: dict[str, object] = dict(kwargs)
            for name, value in kwargs.items():
                setattr(self, name, value)
            # Engine defaults, so introspection of the constructed object
            # sees SOMETHING on every axis either way. The omitted arm does
            # NOT assert these values: they are transformers' defaults, the
            # package deliberately does not claim them, and this package's
            # contract is only that they were never wired (#342).
            self.optim = kwargs.get("optim", "adamw_torch")
            self.gradient_accumulation_steps = kwargs.get("gradient_accumulation_steps", 1)
            self.max_grad_norm = kwargs.get("max_grad_norm", 1.0)
            stack.training_arguments.append(self)

    FakeTrainingArguments.__init__.__signature__ = inspect.Signature(  # type: ignore[attr-defined]
        [inspect.Parameter("self", inspect.Parameter.POSITIONAL_OR_KEYWORD)]
        + [
            inspect.Parameter(name, inspect.Parameter.KEYWORD_ONLY)
            for name in _FAKE_ACCEPTED_KWARGS
        ]
    )

    class FakeTokenizer:
        pad_token = None
        eos_token = "</s>"

        @classmethod
        def from_pretrained(cls, ref: str) -> FakeTokenizer:
            return cls()

        def __call__(self, texts, truncation: bool = False, max_length: int | None = None) -> dict:
            seqs = texts if isinstance(texts, list) else [texts]
            return {"input_ids": [[1, 2] for _ in seqs]}

    class FakeSplit:
        column_names = ["text"]

        def map(self, fn, batched: bool = False, remove_columns=None) -> list:
            fn({"text": ["alpha", "beta"]})  # exercise the tokenizer path for real
            return [{"input_ids": [1, 2]}, {"input_ids": [3, 4]}]

    class FakeModel:
        def __init__(self) -> None:
            self.config = SimpleNamespace(tie_word_embeddings=False)

        def state_dict(self) -> dict:
            return {"layer0.weight": object(), "layer0.bias": object()}

    class FakeAutoModelForCausalLM:
        @classmethod
        def from_pretrained(cls, ref: str, **kwargs: object) -> FakeModel:
            model = FakeModel()
            model.loader_kwargs = dict(kwargs)
            return model

    class FakeDataCollator:
        def __init__(self, tokenizer=None, mlm: bool = False) -> None:
            self.tokenizer = tokenizer
            self.mlm = mlm

    class FakeTrainer:
        def __init__(
            self,
            model=None,
            args=None,
            train_dataset=None,
            data_collator=None,
            callbacks=None,
        ) -> None:
            self.model = model
            self.args = args
            stack.trainer_instances.append(self)

        def train(self) -> None:
            return None

        def save_model(self, path: str) -> None:
            # Write the declaration it was measured against, as REAL
            # safetensors bytes, so the final-save format assertion and the
            # real checkpoint-gate sweep have something honest to read back.
            out = Path(path)
            out.mkdir(parents=True, exist_ok=True)
            header = json.dumps(
                {
                    "layer0.weight": {"dtype": "F32", "shape": [2], "data_offsets": [0, 8]},
                    "layer0.bias": {"dtype": "F32", "shape": [2], "data_offsets": [8, 16]},
                }
            ).encode("utf-8")
            (out / "model.safetensors").write_bytes(
                struct.pack("<Q", len(header)) + header + (b"\x00" * 16)
            )

    torch_mod = ModuleType("torch")
    torch_mod.manual_seed = lambda *a, **k: None  # type: ignore[attr-defined]

    datasets_mod = ModuleType("datasets")

    def _load_dataset(ref: str, **kwargs: object) -> dict:
        stack.load_calls.append(ref)
        return {"train": FakeSplit()}

    datasets_mod.load_dataset = _load_dataset  # type: ignore[attr-defined]

    transformers_mod = ModuleType("transformers")
    transformers_mod.__version__ = "0.0-synthetic"  # type: ignore[attr-defined]
    transformers_mod.AutoTokenizer = FakeTokenizer  # type: ignore[attr-defined]
    transformers_mod.AutoModelForCausalLM = FakeAutoModelForCausalLM  # type: ignore[attr-defined]
    transformers_mod.DataCollatorForLanguageModeling = FakeDataCollator  # type: ignore[attr-defined]
    transformers_mod.Trainer = FakeTrainer  # type: ignore[attr-defined]
    transformers_mod.TrainingArguments = FakeTrainingArguments  # type: ignore[attr-defined]
    transformers_mod.TrainerCallback = object  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "torch", torch_mod)
    monkeypatch.setitem(sys.modules, "datasets", datasets_mod)
    monkeypatch.setitem(sys.modules, "transformers", transformers_mod)

    # The machine-fact plane is not the instrument under test. Stub it narrowly:
    # a synthetic profile with the fields the prologue PRINTS, and an empty
    # consistency result, so the run proceeds to Trainer construction. The
    # axis tests assert REACHED-SITE separately, so an over-strict stub cannot
    # pass them by stopping the run early.
    monkeypatch.setattr(
        loop,
        "_resolve_profile",
        lambda cfg: SimpleNamespace(
            name="synthetic", scheduler="none", gpus_per_node=cfg.gpus_per_node
        ),
    )
    monkeypatch.setattr(loop.Topology, "validate_against", lambda self, profile: [])
    # WORLD_SIZE from the ambient environment would pull the declared-vs-
    # effective comparator into the run; these pins do not measure it.
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    monkeypatch.delenv("FS_RUN_ID", raising=False)
    return stack


def _drive(tmp_path: Path, **overrides: object) -> tuple[int, dict]:
    # Unset adapter/precision/objective-adjacent knobs all default to
    # abstained; objective="sft" because this entry implements sft and data.
    cfg = loop.TrainConfig(
        model="synthetic/model-id",
        dataset="synthetic/dataset-id",
        output_dir=tmp_path / "out",
        nodes=1,
        gpus_per_node=1,
        profile_name="synthetic",
        objective="sft",
        max_steps=2,
        save_interval=1,
        **overrides,
    )
    rc = loop.train(cfg)
    manifest_path = cfg.output_dir / loop.MANIFEST_NAME
    assert manifest_path.is_file(), (
        f"train() returned {rc} without leaving {loop.MANIFEST_NAME}: every exit "
        "arm the thin path can reach is obliged to leave a manifest"
    )
    return rc, json.loads(manifest_path.read_text(encoding="utf-8"))


def _config_entry(manifest: dict, key: str) -> dict:
    config = manifest.get("config")
    assert isinstance(config, dict), f"manifest carries no config mapping: {manifest!r}"
    assert key in config, (
        f"config key {key!r} is ABSENT from the manifest -- an absent key is "
        "indistinguishable from a loop version that never populated the field, "
        f"which is the defect provenance exists to catch. Present: {sorted(config)}"
    )
    entry = config[key]
    assert isinstance(entry, dict), f"config entry {key!r} is not a record: {entry!r}"
    for field in ("key", "value", "source"):
        assert field in entry, f"config entry {key!r} lacks {field!r}: {entry!r}"
    return entry


@pytest.mark.parametrize(
    ("cfg_field", "engine_kwarg", "declared"),
    [
        ("optimizer", "optim", "sgd"),
        ("gradient_accumulation_steps", "gradient_accumulation_steps", 4),
        ("max_grad_norm", "max_grad_norm", 0.5),
    ],
    ids=["optimizer-renamed-to-optim", "gradient-accumulation-steps", "max-grad-norm"],
)
def test_declared_axis_reaches_constructed_training_arguments(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    cfg_field: str,
    engine_kwarg: str,
    declared: object,
) -> None:
    stack = _install_fake_stack(monkeypatch)
    rc, manifest = _drive(tmp_path, **{cfg_field: declared})
    out = capsys.readouterr().out

    # OUTCOME first-person guard: the leg must not have been REFUSED. REFUSE
    # here would mean the wiring was rejected or dropped (e.g. "optimizer"
    # passed instead of "optim" -- the engine name for this axis differs, and
    # asserting "optimizer" in the kwargs would be asserting the bug).
    # 0/5/95 are all admissible: the final adjudication over this synthetic
    # checkpoint is not the instrument this module measures.
    assert rc in (loop.EXIT_PASS, loop.EXIT_RED, loop.EXIT_UNMEASURED), (
        f"declared {cfg_field}={declared!r} was REFUSED (rc={rc}): {out}"
    )
    # REACHED-SITE, asserted SEPARATELY from the outcome below (doctrine: a
    # control that never reaches the site it measures is vacuous).
    assert "[fs:train:trainer]" in out, f"run stopped before Trainer construction: {out}"
    assert len(stack.training_arguments) == 1, (
        f"expected exactly one constructed TrainingArguments, got {len(stack.training_arguments)}"
    )
    assert len(stack.trainer_instances) == 1

    # (a) DECLARED -- introspected off the constructed object, not re-read
    # from anything this test passed in.
    constructed = stack.training_arguments[0]
    assert getattr(constructed, engine_kwarg) == declared, (
        f"TrainingArguments.{engine_kwarg} is {getattr(constructed, engine_kwarg)!r}, "
        f"expected the declared {declared!r}. On a tree where the axis is not "
        "wired, this is the engine default instead -- this assert is that "
        "differential."
    )

    # (c) RECORDED -- key present, carrying the declared value and a source.
    entry = _config_entry(manifest, cfg_field)
    assert entry["key"] == cfg_field
    assert entry["value"] == str(declared), (
        f"manifest records {entry['value']!r} for {cfg_field!r}, not the declared {declared!r}"
    )
    # "config", not "cli": this leg builds TrainConfig in Python and never runs
    # the parser, so there is no command line for the record to point at. The
    # emitter used to stamp "cli" on every key unconditionally, which made this
    # assertion pass by construction and made the manifest claim a provenance
    # it had not measured. The cli/default split over a REAL argv is measured
    # in test_manifest_records_declared_axes.py, which is where the parser is
    # the thing under test.
    assert entry["source"] == "config"


def test_omitted_axes_add_no_kwarg_and_record_an_explicit_absence(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    stack = _install_fake_stack(monkeypatch)
    rc, manifest = _drive(tmp_path)
    out = capsys.readouterr().out

    assert rc in (loop.EXIT_PASS, loop.EXIT_RED, loop.EXIT_UNMEASURED), (
        f"a run declaring NOTHING on the three axes was REFUSED (rc={rc}): {out}"
    )
    assert "[fs:train:trainer]" in out, f"run stopped before Trainer construction: {out}"
    assert len(stack.training_arguments) == 1

    # (b) OMITTED -- key ABSENCE, not presence-with-default-value. Only
    # absence keeps the engine default unclaimed (#342).
    constructed = stack.training_arguments[0]
    for engine_kwarg in ("optim", "gradient_accumulation_steps", "max_grad_norm"):
        assert engine_kwarg not in constructed.received_kwargs, (
            f"{engine_kwarg!r} was added to TrainingArguments kwargs though the "
            f"axis was never declared: {constructed.received_kwargs}"
        )

    # (c) RECORDED as an explicit absence: key present + value None != key
    # absent. _config_entry already fails on a missing key, so passing both
    # asserts means the two states are distinguishable.
    for cfg_field in ("optimizer", "gradient_accumulation_steps", "max_grad_norm"):
        entry = _config_entry(manifest, cfg_field)
        assert entry["key"] == cfg_field
        assert entry["value"] == "None", (
            f"omitted {cfg_field!r} records {entry['value']!r} instead of an "
            "explicit absence -- that is the #342 laundering this axis forbids"
        )
        # See the note on the declared leg above: no parser ran here, so
        # "config" is the honest source and "cli" would be a second false
        # claim sitting on top of the absence this block exists to protect.
        assert entry["source"] == "config"


def test_unsupported_optimizer_refuses_before_any_trainer_exists(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """MUST-FIRE arm. The fake TrainingArguments rejects the name exactly the
    way real transformers does (ValueError from __post_init__). A tree where
    train()'s ValueError/TypeError refusal arm is deleted lets that escape as
    a traceback -- exit 1, an undeclared state (#171) -- and this test FAILS
    with an exception instead of asserting 96. A tree where the arm wrongly
    RE-DECLARES a guessed value constructs a Trainer and fails the
    no-construction assertion. Both broken shapes come out different.
    """
    stack = _install_fake_stack(monkeypatch)
    rc, manifest = _drive(tmp_path, optimizer="not_a_real_optimizer_name")
    out = capsys.readouterr().out

    assert rc == loop.EXIT_REFUSE, f"rc={rc}, expected 96 (REFUSE): {out}"
    # REACHED-SITE for THIS arm: the rejection path specifically, not some
    # earlier refusal. If the optimizer reached the kwargs under a WRONG name,
    # loop.py's introspection refusal would fire instead -- also 96, but the
    # wrong control. The marker separates the two.
    assert "[fs:train:refuse]" in out
    assert "rejected the declared config at Trainer/TrainingArguments construction" in out
    # No Trainer, and not even a completed TrainingArguments: the constructor
    # is what rejected the declaration.
    assert stack.trainer_instances == [], (
        "a Trainer was constructed for a refused declaration -- a control that "
        "cannot fail is the defect this campaign keeps finding"
    )
    assert stack.training_arguments == []

    # The refused run still leaves an honest artifact: the rejection is
    # recorded as DECLARED, so a reader can check what was asked for.
    entry = _config_entry(manifest, "optimizer")
    assert entry["value"] == "not_a_real_optimizer_name"
    assert _config_entry(manifest, "stage")["value"] == "refused"
