"""Tests for the declared-precision and adapter (LoRA) planes of the thin loop.

No GPU, no network, no torch: the observed side of every comparison is a
hand-written safetensors byte stream (8-byte header length, JSON header, zero
payload), so the comparator is always evaluated over tensor metadata it did
NOT construct -- the negative control this suite owes any detector.

The peft wrapping path IS exercised here, but through sys.modules stand-ins
for torch/transformers/datasets/peft, never against real peft. What that
covers is the loop's own branching -- the missing-dependency refusal, the
zero-attachment refusal, attachment counting, the adapter-scoped checkpoint
declaration, and the wrap-failure RED. What it does NOT cover is peft's real
target resolution against a live module graph: the fakes supply
`named_parameters` prefixes directly, so a change in how peft names or matches
modules would not be caught by anything in this file. That experiment needs
torch+peft on a GPU host and is named in the module RISKS of loop.py.
"""

from __future__ import annotations

import json
import struct
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from foundationscale.train.cli import build_parser
from foundationscale.train.loop import (
    ADAPTERS,
    EXIT_RED,
    EXIT_REFUSE,
    EXIT_UNMEASURED,
    PRECISIONS,
    FoundationScaleSaveGate,
    PrecisionAgreement,
    TrainConfig,
    _declare_checkpoint,
    _dtype_histogram,
    _manifest_payload,
    check_precision_agreement,
    train,
)


def _config(**overrides: Any) -> TrainConfig:
    kwargs: dict[str, Any] = {
        "model": "test-model",
        "dataset": "test-dataset",
        "output_dir": Path("out"),
        "nodes": 1,
        "gpus_per_node": 1,
        "profile_name": "local-single-node",
    }
    kwargs.update(overrides)
    return TrainConfig(**kwargs)


_DTYPE_BYTES = {"BF16": 2, "F16": 2, "F32": 4}


def _write_safetensors(path: Path, dtypes: dict[str, str]) -> None:
    """Write a real safetensors file with one-element tensors of the given dtypes."""
    header: dict[str, Any] = {}
    offset = 0
    for name, dtype in dtypes.items():
        nbytes = _DTYPE_BYTES[dtype]
        header[name] = {
            "dtype": dtype,
            "shape": [1],
            "data_offsets": [offset, offset + nbytes],
        }
        offset += nbytes
    blob = json.dumps(header).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(struct.pack("<Q", len(blob)) + blob + b"\x00" * offset)


def test_train_config_declared_fields_default_to_none() -> None:
    cfg = _config()
    assert cfg.precision is None
    assert cfg.adapter is None
    assert cfg.adapter_rank is None
    assert cfg.adapter_alpha is None
    assert cfg.adapter_targets is None
    assert cfg.adapter_dropout is None


def test_precision_and_adapter_constants_are_sorted_and_complete() -> None:
    assert tuple(sorted(PRECISIONS)) == PRECISIONS
    assert {"bf16", "fp16", "fp32", "nvfp4"} <= set(PRECISIONS)
    assert tuple(sorted(ADAPTERS)) == ADAPTERS
    assert "lora" in ADAPTERS
    # Every accepted precision the package plane can EXECUTE validates cleanly;
    # nvfp4 is declarable (the refusal is train()'s job, not the schema's).
    for value in PRECISIONS:
        assert _config(precision=value).precision == value


def test_train_config_precision_refuses_unknown_value_naming_set() -> None:
    with pytest.raises(ValueError) as excinfo:
        _config(precision="bf19")
    text = str(excinfo.value)
    assert "precision" in text
    assert "'bf19'" in text
    assert "('bf16', 'fp16', 'fp32', 'nvfp4')" in text


def test_train_config_adapter_refuses_unknown_value_naming_set() -> None:
    with pytest.raises(ValueError) as excinfo:
        _config(adapter="adalora", adapter_rank=8)
    text = str(excinfo.value)
    assert "adapter" in text
    assert "'adalora'" in text
    assert "('lora',)" in text


def test_train_config_adapter_refuses_partial_specification() -> None:
    with pytest.raises(ValueError) as excinfo:
        _config(adapter_rank=8)
    text = str(excinfo.value)
    assert "adapter_rank" in text
    assert "adapter is None" in text
    assert "('lora',)" in text
    with pytest.raises(ValueError) as excinfo_two:
        _config(adapter_dropout=0.1)
    text_two = str(excinfo_two.value)
    assert "adapter_dropout" in text_two
    assert "adapter is None" in text_two


def test_train_config_adapter_refuses_missing_rank() -> None:
    with pytest.raises(ValueError) as excinfo:
        _config(adapter="lora")
    text = str(excinfo.value)
    assert "adapter" in text
    assert "'lora'" in text
    assert "adapter_rank" in text
    assert "None" in text


def test_train_config_adapter_refuses_non_positive_rank() -> None:
    with pytest.raises(ValueError) as excinfo:
        _config(adapter="lora", adapter_rank=0)
    text = str(excinfo.value)
    assert "adapter_rank=0" in text
    assert "positive" in text


def test_train_config_adapter_refuses_empty_targets_as_vacuous() -> None:
    with pytest.raises(ValueError) as excinfo:
        _config(adapter="lora", adapter_rank=8, adapter_targets=())
    text = str(excinfo.value)
    assert "adapter_targets" in text
    assert "()" in text
    assert "vacuous" in text


def test_train_config_adapter_targets_normalise_list_to_tuple() -> None:
    cfg = _config(adapter="lora", adapter_rank=8, adapter_targets=["q_proj", "v_proj"])
    assert cfg.adapter_targets == ("q_proj", "v_proj")
    assert isinstance(cfg.adapter_targets, tuple)


_CLI_MIN = [
    "--model",
    "m",
    "--dataset",
    "d",
    "--output-dir",
    "o",
    "--nodes",
    "1",
    "--gpus-per-node",
    "1",
    "--profile-name",
    "local-single-node",
]


def test_cli_new_flags_default_to_none_not_a_value() -> None:
    args = build_parser().parse_args(_CLI_MIN)
    assert args.precision is None
    assert args.adapter is None
    assert args.adapter_rank is None
    assert args.adapter_alpha is None
    assert args.adapter_target is None
    assert args.adapter_dropout is None


def test_cli_adapter_target_is_repeatable_and_ordered() -> None:
    args = build_parser().parse_args(
        [*_CLI_MIN, "--adapter-target", "q_proj", "--adapter-target", "v_proj"]
    )
    assert args.adapter_target == ["q_proj", "v_proj"]


def test_manifest_payload_records_precision_and_adapter_separately() -> None:
    undeclared = _manifest_payload(_config(), stage="train")
    assert undeclared["config"]["precision"] is None  # round-trips as None, never coerced
    assert undeclared["config"]["adapter"] is None
    assert undeclared["config"]["adapter_rank"] is None
    declared = _manifest_payload(
        _config(
            precision="bf16",
            adapter="lora",
            adapter_rank=8,
            adapter_alpha=16.0,
            adapter_targets=("q_proj",),
        ),
        stage="train",
    )
    assert declared["config"]["precision"] == "bf16"
    assert declared["config"]["adapter"] == "lora"
    assert declared["config"]["adapter_rank"] == 8
    assert declared["config"]["adapter_alpha"] == 16.0
    assert declared["config"]["adapter_targets"] == ["q_proj"]
    assert declared["config"]["adapter_dropout"] is None


def test_precision_agreement_passes_on_agreement() -> None:
    result = check_precision_agreement("bf16", {"BF16": 3})
    assert result.status == "pass"
    assert result.observed == {"BF16": 3}
    # Documented autocast tolerance: fp32 master weights beside half dtypes are
    # an agreement, and the comparator's docstring says so explicitly.
    assert check_precision_agreement("fp16", {"F16": 1, "F32": 2}).status == "pass"


def test_precision_agreement_red_names_both_sides_and_counts() -> None:
    result = check_precision_agreement("fp32", {"BF16": 2, "F32": 1})
    assert result.status == "red"
    text = result.message
    assert "'fp32'" in text
    assert "'BF16': 2" in text  # the rejected side, with its count
    assert "'F32': 1" in text  # 袖口 the conforming side, with its count
    assert "3 tensor" in text


def test_precision_agreement_abstains_when_nothing_declared() -> None:
    result = check_precision_agreement(None, {"BF16": 4})
    assert result.status == "abstain"
    assert result.status != "pass"
    # Abstention still reports what was observed -- an invisible abstention is
    # indistinguishable from a check that ran.
    assert "'BF16': 4" in result.message
    assert result.observed == {"BF16": 4}


def test_precision_agreement_refuses_a_vacuous_observation() -> None:
    for observed in (None, {}):
        result = check_precision_agreement("bf16", observed)
        assert result.status == "refuse"
        assert "vacuous" in result.message
        assert result.observed is None


def test_precision_agreement_refuses_declared_precision_without_backend() -> None:
    # nvfp4 is declarable in the schema but has no accepted-dtype mapping; the
    # comparator must refuse, not borrow another precision's rule.
    result = check_precision_agreement("nvfp4", {"F32": 3})
    assert result.status == "refuse"
    assert "'nvfp4'" in result.message


def test_dtype_histogram_reads_tensors_it_did_not_construct(tmp_path: Path) -> None:
    # NEGATIVE CONTROL: the bytes below are built in this test, by hand; the
    # detector under test never touched them. A detector that cannot report
    # agreement over a set it did not construct is worthless -- three of those
    # have shipped here before.
    _write_safetensors(
        tmp_path / "model.safetensors",
        {"a.weight": "BF16", "b.weight": "F32", "c.weight": "BF16"},
    )
    assert _dtype_histogram(tmp_path) == {"BF16": 2, "F32": 1}
    # And over that same externa byte stream the comparator reports both ways:
    assert check_precision_agreement("bf16", _dtype_histogram(tmp_path)).status == "pass"
    assert check_precision_agreement("fp32", _dtype_histogram(tmp_path)).status == "red"
    assert _dtype_histogram(tmp_path / "no-shards-here") is None


def _drive_first_save(
    tmp_path: Path, precision: str | None, dtypes: dict[str, str]
) -> tuple[FoundationScaleSaveGate, Any, PrecisionAgreement | None]:
    ckpt = tmp_path / "checkpoint-7"
    ckpt.mkdir(parents=True)
    _write_safetensors(ckpt / "model.safetensors", dtypes)

    def _unbuildable(_: Path | str) -> Any:
        # The precision check runs BEFORE and INDEPENDENTLY of the checkpoint
        # context, so an unbuildable context must not silence it.
        raise RuntimeError("no checkpoint context in this test")

    gate = FoundationScaleSaveGate(
        context_builder=_unbuildable,
        declared_precision=precision,
    )
    control = SimpleNamespace(should_training_stop=False)
    gate.on_save(
        SimpleNamespace(output_dir=str(tmp_path)),
        SimpleNamespace(global_step=7),
        control,
    )
    return gate, control, gate.precision_agreement


def test_save_gate_first_save_precision_disagreement_stops_training(tmp_path: Path) -> None:
    gate, control, agreement = _drive_first_save(tmp_path, "fp32", {"w.weight": "BF16"})
    assert agreement is not None
    assert agreement.status == "red"
    # RED must stop the run through the SAME path a blocking gate uses:
    assert gate.blocked is True
    assert control.should_training_stop is True
    assert any(
        record.get("verdicts", {}).get("precision.observed_vs_declared") == "red"
        for record in gate.records
    )


def test_save_gate_first_save_undeclared_precision_abstains(tmp_path: Path) -> None:
    gate, control, agreement = _drive_first_save(tmp_path, None, {"w.weight": "BF16"})
    assert agreement is not None
    assert agreement.status == "abstain"
    assert gate.blocked is False
    assert control.should_training_stop is False


def test_train_refuses_nvfp4_before_touching_a_backend(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # torch-free by construction: the refusal fires before the optional-deps
    # import block, so this asserts reality on any host.
    rc = train(_config(output_dir=tmp_path, precision="nvfp4"))
    assert rc == EXIT_REFUSE
    out = capsys.readouterr().out
    assert "no nvfp4 backend" in out
    assert "precision='nvfp4'" in out
    manifest = tmp_path / "run_manifest.json"
    assert manifest.exists()  # the refusal leaves a STATEMENT, not silence
    assert '"nvfp4"' in manifest.read_text(encoding="utf-8")


class _FakeParam:
    """A tensor stand-in for the loop's parameter accounting."""

    def __init__(self, numel: int, *, requires_grad: bool) -> None:
        self._numel = numel
        self.requires_grad = requires_grad

    def numel(self) -> int:
        return self._numel

    def element_size(self) -> int:
        return 2


class _FakeModel:
    """A stand-in model: state_dict()/named_parameters() over a fixed list.

    Carries a config with tying off (so no aliases are declared away) and an
    optional ``peft_config`` attribute, which is exactly the probe
    ``_declare_checkpoint`` uses to scope the declaration to adapter tensors.
    """

    def __init__(self, params: list[tuple[str, _FakeParam]], peft_config: Any = None) -> None:
        self._params = list(params)
        self.peft_config = peft_config
        self.config = SimpleNamespace(tie_word_embeddings=False)

    def state_dict(self) -> dict[str, _FakeParam]:
        return dict(self._params)

    def named_parameters(self) -> list[tuple[str, _FakeParam]]:
        return list(self._params)


class _FakeTokenizer:
    def __init__(self) -> None:
        self.pad_token: str | None = None  # exercises the pad_token <- eos arm
        self.eos_token = "<eos>"

    def __call__(self, texts: list[str], *, truncation: bool, max_length: int) -> dict[str, Any]:
        return {"input_ids": [[1] for _ in texts]}


class _FakeDataset:
    """One 'text' column, the only one the thin path requires."""

    column_names = ["text"]

    def map(self, fn: Any, *, batched: bool, remove_columns: list[str]) -> _FakeDataset:
        fn({"text": ["a training example"]})  # the loop's lambda must consume this
        return self

    def __len__(self) -> int:
        return 2


def _install_fake_training_stack(
    monkeypatch: pytest.MonkeyPatch,
    *,
    base_model: _FakeModel,
    args_sink: list[dict[str, Any]],
    training_arguments_class: Any = None,
) -> None:
    """Install torch/transformers/datasets stand-ins into ``sys.modules``.

    The real absence of the train extra is a REFUSE at Step.DEPS, so the only
    torch-free way past it is a sys.modules stand-in. The fakes run the loop
    through model/dataset construction, the adapter block, and the trainer,
    ending at a final save that writes ZERO shards -- the no-shard UNMEASURED
    arm -- which keeps the run off the real gate registry. ``args_sink``
    receives the kwargs the default TrainingArguments was constructed with.
    """
    torch_module = ModuleType("torch")

    def _manual_seed(seed: int) -> None:
        return None

    torch_module.manual_seed = _manual_seed

    datasets_module = ModuleType("datasets")

    def _load_dataset(ref: str) -> dict[str, Any]:
        return {"train": _FakeDataset()}

    datasets_module.load_dataset = _load_dataset

    class _FakeAutoTokenizer:
        @classmethod
        def from_pretrained(cls, ref: str) -> _FakeTokenizer:
            return _FakeTokenizer()

    class _FakeAutoModel:
        @classmethod
        def from_pretrained(cls, ref: str) -> _FakeModel:
            return base_model

    class _FakeCollator:
        def __init__(self, *, tokenizer: Any, mlm: bool) -> None:
            self.mlm = mlm

    if training_arguments_class is None:

        class _DefaultTrainingArguments:
            # Explicit names, including bf16/fp16 and save_safetensors, so the
            # loop's introspection drop-check passes and we can observe which
            # flags the declaration plane actually wired in.
            def __init__(
                self,
                *,
                output_dir: str,
                max_steps: int,
                per_device_train_batch_size: int,
                learning_rate: float,
                save_strategy: str,
                save_steps: int,
                seed: int,
                logging_steps: int,
                report_to: list[Any],
                ddp_find_unused_parameters: bool,
                save_safetensors: bool,
                bf16: bool = False,
                fp16: bool = False,
            ) -> None:
                self.output_dir = output_dir
                args_sink.append(
                    {
                        "max_steps": max_steps,
                        "save_steps": save_steps,
                        "save_safetensors": save_safetensors,
                        "logging_steps": logging_steps,
                        "bf16": bf16,
                        "fp16": fp16,
                    }
                )

        training_arguments_class = _DefaultTrainingArguments

    class _FakeTrainer:
        def __init__(
            self,
            *,
            model: Any,
            args: Any,
            train_dataset: Any,
            data_collator: Any,
            callbacks: list[Any],
        ) -> None:
            self.model = model
            self.callbacks = callbacks

        def train(self) -> None:
            return None

        def save_model(self, out: str) -> None:
            # An empty final directory: the run must adjudicate UNMEASURED,
            # proving train() refuses to read 0 shards as a clean save.
            Path(out).mkdir(parents=True, exist_ok=True)

    transformers_module = ModuleType("transformers")
    transformers_module.AutoTokenizer = _FakeAutoTokenizer
    transformers_module.AutoModelForCausalLM = _FakeAutoModel
    transformers_module.DataCollatorForLanguageModeling = _FakeCollator
    transformers_module.Trainer = _FakeTrainer
    transformers_module.TrainingArguments = training_arguments_class

    monkeypatch.setitem(sys.modules, "torch", torch_module)
    monkeypatch.setitem(sys.modules, "datasets", datasets_module)
    monkeypatch.setitem(sys.modules, "transformers", transformers_module)


def _make_fake_peft(
    sink: list[dict[str, Any]],
    *,
    wrapped: _FakeModel | None = None,
    wrap_error: str | None = None,
) -> ModuleType:
    """A fake ``peft`` module whose LoraConfig records exactly what it got."""
    module = ModuleType("peft")

    class _FakeLoraConfig:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = dict(kwargs)
            sink.append(dict(kwargs))

    def _get_peft_model(model: Any, config: Any) -> _FakeModel:
        if wrap_error is not None:
            raise RuntimeError(wrap_error)
        assert wrapped is not None, "test must supply a wrapped model"
        wrapped.peft_config = config  # the probe _declare_checkpoint reads
        return wrapped

    module.LoraConfig = _FakeLoraConfig
    module.get_peft_model = _get_peft_model
    return module


def test_train_refuses_lora_when_peft_is_missing_naming_peft_and_hint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # THE load-bearing refusal: adapter='lora' with peft absent must STOP the
    # run naming the dep and the remedy -- never degrade to a full fine-tune
    # the operator did not ask for. peft is forced absent via sys.modules=None,
    # which makes `from peft import ...` raise ImportError on any host.
    args_sink: list[dict[str, Any]] = []
    _install_fake_training_stack(
        monkeypatch,
        base_model=_FakeModel([("w.weight", _FakeParam(4, requires_grad=True))]),
        args_sink=args_sink,
    )
    monkeypatch.setitem(sys.modules, "peft", None)

    rc = train(_config(output_dir=tmp_path, adapter="lora", adapter_rank=8))

    assert rc == EXIT_REFUSE
    out = capsys.readouterr().out
    assert "[fs:train:refuse]" in out
    assert "adapter='lora' is declared" in out
    assert "'peft'" in out  # the missing dependency is NAMED, not implied
    assert "foundationscale[train]" in out  # the install hint travels with it
    assert "full fine-tune" in out  # ...and the silent fallback is named as such
    assert "[fs:train:adapter]" not in out  # no attach marker: nothing was wrapped
    assert args_sink == []  # refused BEFORE any TrainingArguments were built


def test_train_refuses_a_lora_attachment_of_zero_modules_as_vacuous(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # Wrapping SUCCEEDED, but the declared targets matched no module: a real,
    # measured zero (the loop enumerated the names to get it), so REFUSE with
    # the attachment count and the declared targets in the message.
    lora_sink: list[dict[str, Any]] = []
    wrapped = _FakeModel(
        [
            ("w.weight", _FakeParam(4, requires_grad=False)),
            ("v.weight", _FakeParam(4, requires_grad=False)),
        ]
    )
    _install_fake_training_stack(
        monkeypatch,
        base_model=_FakeModel([("w.weight", _FakeParam(4, requires_grad=True))]),
        args_sink=[],
    )
    monkeypatch.setitem(sys.modules, "peft", _make_fake_peft(lora_sink, wrapped=wrapped))

    rc = train(
        _config(
            output_dir=tmp_path,
            adapter="lora",
            adapter_rank=8,
            adapter_targets=("q_proj",),
        )
    )

    assert rc == EXIT_REFUSE
    out = capsys.readouterr().out
    assert "[fs:train:adapter]" in out  # the refusal reports through the adapter step
    assert "attached to 0 modules" in out
    assert "vacuous" in out
    assert "'q_proj'" in out  # the operator's declared targets are named
    # Wrapping genuinely ran: peft got r and the targets, and UNSET knobs were
    # omitted from the LoraConfig rather than defaulted at this seam.
    assert lora_sink == [{"r": 8, "target_modules": ["q_proj"]}]


def test_train_measures_lora_attachment_and_scopes_the_declaration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # End-to-end through the fake stack: attachment counting (2 distinct
    # module prefixes, 24/88 trainable), the adapter branch of the checkpoint
    # declaration (3 lora_ tensors out of a 4-entry state dict), and the
    # no-shard UNMEASURED adjudication after the final save.
    lora_sink: list[dict[str, Any]] = []
    wrapped = _FakeModel(
        [
            ("base.q_proj.lora_A.default.weight", _FakeParam(8, requires_grad=True)),
            ("base.q_proj.lora_B.default.weight", _FakeParam(8, requires_grad=True)),
            ("base.v_proj.lora_A.default.weight", _FakeParam(8, requires_grad=True)),
            ("base.q_proj.weight", _FakeParam(64, requires_grad=False)),
        ]
    )
    _install_fake_training_stack(
        monkeypatch,
        base_model=_FakeModel([("w.weight", _FakeParam(4, requires_grad=True))]),
        args_sink=[],
    )
    monkeypatch.setitem(sys.modules, "peft", _make_fake_peft(lora_sink, wrapped=wrapped))

    rc = train(
        _config(
            output_dir=tmp_path,
            adapter="lora",
            adapter_rank=8,
            adapter_alpha=16.0,
            adapter_targets=("q_proj", "v_proj"),
            adapter_dropout=0.1,
        )
    )

    assert rc == EXIT_UNMEASURED
    out = capsys.readouterr().out
    assert "lora attached to 2 module(s)" in out
    assert "24/88 parameters trainable" in out
    # The saved peft artifact carries ADAPTER tensors only, so the declaration
    # denominator is the 3 lora_ names, not the 4 state-dict entries -- a
    # full-model denominator here would fake-RED every healthy LoRA run.
    assert "0 of 3 declared tensors" in out
    # Every knob the operator stated reached peft verbatim -- tuple -> list:
    assert lora_sink == [
        {
            "r": 8,
            "lora_alpha": 16.0,
            "target_modules": ["q_proj", "v_proj"],
            "lora_dropout": 0.1,
        }
    ]
    assert isinstance(lora_sink[0]["target_modules"], list)
    manifest_text = (tmp_path / "run_manifest.json").read_text(encoding="utf-8")
    assert "adapter.attached_modules" in manifest_text
    assert "peft-wrapped model" in manifest_text  # declaration.adapter_scope survives


def test_train_marks_lora_wrap_construction_failure_as_red_naming_the_cause(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # A wrapping exception is a construction failure (RED 5), not a refusal:
    # peft WAS present and the config was stated, the wrap itself failed.
    lora_sink: list[dict[str, Any]] = []
    wrapped = _FakeModel([("w.weight", _FakeParam(4, requires_grad=False))])
    _install_fake_training_stack(
        monkeypatch,
        base_model=_FakeModel([("w.weight", _FakeParam(4, requires_grad=True))]),
        args_sink=[],
    )
    monkeypatch.setitem(
        sys.modules, "peft", _make_fake_peft(lora_sink, wrapped=wrapped, wrap_error="boom")
    )

    rc = train(_config(output_dir=tmp_path, adapter="lora", adapter_rank=8))

    assert rc == EXIT_RED
    out = capsys.readouterr().out
    assert "[fs:train:red]" in out
    assert "peft wrapping of adapter='lora' failed" in out
    assert "RuntimeError('boom')" in out  # the underlying cause rides the marker


def test_declare_checkpoint_scopes_denominator_to_adapter_tensors_when_peft_wrapped() -> None:
    # The 2e contract stated directly on the declaration: under a peft wrapper
    # only lora_ tensors are declared, the frozen base weights are not, and the
    # adapter_scope note records which arm ran and how many tensors that was.
    model = _FakeModel(
        [
            ("base.q.lora_A.default.weight", _FakeParam(8, requires_grad=True)),
            ("base.q.lora_B.default.weight", _FakeParam(8, requires_grad=True)),
            ("base.q.weight", _FakeParam(64, requires_grad=False)),
        ],
        peft_config=object(),
    )
    declared, notes = _declare_checkpoint(model)
    assert sorted(declared.declared_fqns) == [
        "base.q.lora_A.default.weight",
        "base.q.lora_B.default.weight",
    ]
    assert notes["declaration.state_dict_keys"] == "3"  # the honest INPUT size
    scope = notes["declaration.adapter_scope"]
    assert "peft-wrapped model" in scope
    assert "declared 2 adapter tensor(s)" in scope
    # The dense-vs-MoE basis counts the SCOPED denominator (2), not the input 3:
    assert "0 of 2 declared tensors" in notes["declaration.basis"]


def test_train_wires_fp32_as_explicitly_disabled_flags_not_omission(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # fp32's contract is to TURN bf16/fp16 OFF, not to leave TrainingArguments'
    # environment-leaning defaults in charge; asserted off the arguments the
    # constructor was actually called with, not off the config.
    args_sink: list[dict[str, Any]] = []
    _install_fake_training_stack(
        monkeypatch,
        base_model=_FakeModel([("w.weight", _FakeParam(4, requires_grad=True))]),
        args_sink=args_sink,
    )

    rc = train(_config(output_dir=tmp_path, precision="fp32"))

    assert rc == EXIT_UNMEASURED
    assert args_sink[0]["bf16"] is False
    assert args_sink[0]["fp16"] is False
    # Introspection accepted save_safetensors, so the loop did wire it:
    assert args_sink[0]["save_safetensors"] is True
    # Cadence bounded by the run's own length: min(10, max_steps=20):
    assert args_sink[0]["logging_steps"] == 10


def test_train_refuses_when_training_arguments_drops_thin_path_knobs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # The introspection refusal: a TrainingArguments whose signature no longer
    # accepts the thin path's kwargs must be a REFUSE naming the dropped keys,
    # never a run that silently ignores max_steps or save_steps.
    class _MinimalArgs:
        def __init__(self, *, output_dir: str) -> None:
            self.output_dir = output_dir

    args_sink: list[dict[str, Any]] = []
    _install_fake_training_stack(
        monkeypatch,
        base_model=_FakeModel([("w.weight", _FakeParam(4, requires_grad=True))]),
        args_sink=args_sink,
        training_arguments_class=_MinimalArgs,
    )

    rc = train(_config(output_dir=tmp_path))

    assert rc == EXIT_REFUSE
    out = capsys.readouterr().out
    assert "TrainingArguments does not accept" in out
    assert "'max_steps'" in out  # a dropped knob is NAMED in the refusal
    assert "foundationscale[train]" in out  # the remedy hint travels with it
    assert args_sink == []  # refused BEFORE the constructor call
