# SPDX-License-Identifier: Apache-2.0
"""Coverage tests for ``preference_trainer`` refusal and edge branches.

These target the loader validation legs, the constructor refusal surface, the
encoding refusals, the micro-batched step slicing, and the load-surface
exit-96 branches of ``PreferenceTrainer.run``. Heavy paths that need a real
model use the same in-process tiny-Llama fixture pattern as the smoke tests;
encoding and step branches use a fake tokenizer and a homemade tiny torch
module so they never touch the loader.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

torch = pytest.importorskip("torch")

from foundationscale.rl import preference_trainer as pt  # noqa: E402
from foundationscale.rl.preference_torch import TensorPreferenceLoss  # noqa: E402
from foundationscale.rl.preference_trainer import (  # noqa: E402
    PreferenceTrainConfig,
    PreferenceTrainer,
    load_pairs_jsonl,
)
from foundationscale.rl.registry import reset_algorithm_registry  # noqa: E402
from foundationscale.rl.trainer import TrainerRefusal  # noqa: E402

_SPECIAL_TOKENS = ("<unk>", "<pad>", "<eos>")
_WORD_TOKENS = tuple(f"w{i}" for i in range(125))

_CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{{ message['role'] }}: {{ message['content'] }}\n"
    "{% endfor %}"
    "{% if add_generation_prompt %}assistant: {% endif %}"
)


@pytest.fixture(autouse=True)
def _offline_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    monkeypatch.setenv("HF_DATASETS_OFFLINE", "1")
    monkeypatch.setenv("TOKENIZERS_PARALLELISM", "false")


@pytest.fixture(autouse=True)
def _default_registry() -> None:
    reset_algorithm_registry()


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> Path:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


def _paired_dataset(tmp_path: Path) -> Path:
    rows = [
        {"prompt": "w1 w2", "chosen": "w3 w4", "rejected": "w5 w6"},
        {"prompt": "w7 w8", "chosen": "w9 w10", "rejected": "w11 w12"},
    ]
    return _write_jsonl(tmp_path / "pairs.jsonl", rows)


def _build_model_dir(
    tmp_path: Path,
    *,
    with_template: bool = True,
    with_pad: bool = True,
    with_eos: bool = True,
) -> Path:
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import AutoModelForCausalLM, LlamaConfig, PreTrainedTokenizerFast

    model_dir = tmp_path / "tiny-preference-model"
    config = LlamaConfig(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        vocab_size=128,
        max_position_embeddings=128,
    )
    AutoModelForCausalLM.from_config(config).save_pretrained(str(model_dir))

    vocab = {tok: i for i, tok in enumerate(_SPECIAL_TOKENS + _WORD_TOKENS)}
    wordlevel = Tokenizer(WordLevel(vocab=vocab, unk_token="<unk>"))
    wordlevel.pre_tokenizer = Whitespace()
    fast = PreTrainedTokenizerFast(
        tokenizer_object=wordlevel,
        unk_token="<unk>",
        pad_token="<pad>" if with_pad else None,
        eos_token="<eos>" if with_eos else None,
    )
    if with_template:
        fast.chat_template = _CHAT_TEMPLATE
    fast.save_pretrained(str(model_dir))
    return model_dir


def _cfg(model: str, dataset: str, **overrides: Any) -> PreferenceTrainConfig:
    kwargs: dict[str, Any] = {
        "model": model,
        "dataset": dataset,
        "pairs_per_step": 2,
        "max_steps": 1,
        "learning_rate": 1e-3,
        "device": "cpu",
    }
    kwargs.update(overrides)
    return PreferenceTrainConfig(**kwargs)


# ---------------------------------------------------------------------------
# load_pairs_jsonl validation legs
# ---------------------------------------------------------------------------


def test_loader_refuses_non_string_path() -> None:
    with pytest.raises(TrainerRefusal, match="non-empty string"):
        load_pairs_jsonl("   ", paired=True)


def test_loader_refuses_blank_line(tmp_path: Path) -> None:
    path = tmp_path / "blank.jsonl"
    path.write_text('{"prompt": "w1", "chosen": "w2", "rejected": "w3"}\n\n', encoding="utf-8")
    with pytest.raises(TrainerRefusal, match="the line is blank"):
        load_pairs_jsonl(str(path), paired=True)


def test_loader_refuses_invalid_json(tmp_path: Path) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text("{not json}\n", encoding="utf-8")
    with pytest.raises(TrainerRefusal, match="invalid JSON"):
        load_pairs_jsonl(str(path), paired=True)


def test_loader_refuses_non_object_json(tmp_path: Path) -> None:
    path = tmp_path / "list.jsonl"
    path.write_text("[1, 2, 3]\n", encoding="utf-8")
    with pytest.raises(TrainerRefusal, match="not object"):
        load_pairs_jsonl(str(path), paired=True)


def test_loader_paired_schema_missing_fields(tmp_path: Path) -> None:
    path = _write_jsonl(
        tmp_path / "ktoish.jsonl", [{"prompt": "w1", "completion": "w2", "label": True}]
    )
    with pytest.raises(TrainerRefusal, match="a paired objective requires fields"):
        load_pairs_jsonl(str(path), paired=True)


def test_loader_kto_schema_missing_fields(tmp_path: Path) -> None:
    path = _write_jsonl(tmp_path / "pairish.jsonl", [{"prompt": "w1", "chosen": "w2"}])
    with pytest.raises(TrainerRefusal, match="a KTO objective requires fields"):
        load_pairs_jsonl(str(path), paired=False)


def test_loader_refuses_empty_text_field(tmp_path: Path) -> None:
    rows = [{"prompt": "w1", "chosen": "", "rejected": "w2"}]
    path = _write_jsonl(tmp_path / "empty.jsonl", rows)
    with pytest.raises(TrainerRefusal, match="non-empty string"):
        load_pairs_jsonl(str(path), paired=True)


def test_loader_accepts_numeric_zero_one_labels(tmp_path: Path) -> None:
    rows = [
        {"prompt": "w1", "completion": "w2", "label": 1},
        {"prompt": "w3", "completion": "w4", "label": 0},
    ]
    path = _write_jsonl(tmp_path / "numeric.jsonl", rows)
    records = load_pairs_jsonl(str(path), paired=False)
    assert [record["label"] for record in records] == [True, False]
    assert all(isinstance(record["label"], bool) for record in records)


def test_loader_refuses_bad_label(tmp_path: Path) -> None:
    rows = [{"prompt": "w1", "completion": "w2", "label": "yes"}]
    path = _write_jsonl(tmp_path / "badlabel.jsonl", rows)
    with pytest.raises(TrainerRefusal, match="not bool, 0 or 1"):
        load_pairs_jsonl(str(path), paired=False)


def test_loader_refuses_unreadable_file(tmp_path: Path) -> None:
    with pytest.raises(TrainerRefusal, match="could not be"):
        load_pairs_jsonl(str(tmp_path / "missing.jsonl"), paired=True)


def test_loader_refuses_non_utf8_file(tmp_path: Path) -> None:
    path = tmp_path / "bytes.jsonl"
    path.write_bytes(b"\xff\xfe\x00bad\n")
    with pytest.raises(TrainerRefusal, match="not UTF-8"):
        load_pairs_jsonl(str(path), paired=True)


def test_loader_refuses_empty_file(tmp_path: Path) -> None:
    path = tmp_path / "empty.jsonl"
    path.write_text("", encoding="utf-8")
    with pytest.raises(TrainerRefusal, match="0 of 0 lines"):
        load_pairs_jsonl(str(path), paired=True)


# ---------------------------------------------------------------------------
# Constructor refusals
# ---------------------------------------------------------------------------


def test_config_refuses_blank_model() -> None:
    with pytest.raises(TrainerRefusal, match="a non-empty string is"):
        PreferenceTrainer(_cfg("  ", "d.jsonl"))


def test_config_refuses_bool_learning_rate() -> None:
    with pytest.raises(TrainerRefusal, match="finite"):
        PreferenceTrainer(_cfg("m", "d.jsonl", learning_rate=True))


def test_config_refuses_nonpositive_learning_rate() -> None:
    with pytest.raises(TrainerRefusal, match="descent"):
        PreferenceTrainer(_cfg("m", "d.jsonl", learning_rate=-1e-3))


def test_config_refuses_non_integer_denominator() -> None:
    with pytest.raises(TrainerRefusal, match="an integer is required"):
        PreferenceTrainer(_cfg("m", "d.jsonl", pairs_per_step=2.5))


def test_config_refuses_below_minimum_denominator() -> None:
    with pytest.raises(TrainerRefusal, match="is the minimum"):
        PreferenceTrainer(_cfg("m", "d.jsonl", max_steps=0))


def test_config_refuses_non_bool_master_weights() -> None:
    with pytest.raises(TrainerRefusal, match="master_weights"):
        PreferenceTrainer(_cfg("m", "d.jsonl", master_weights="yes"))


def test_config_refuses_negative_kto_reference_point() -> None:
    with pytest.raises(TrainerRefusal, match="non-negative"):
        PreferenceTrainer(_cfg("m", "d.jsonl", kto_reference_point=-1.0))


def test_unsupported_configured_knob_refuses() -> None:
    with pytest.raises(TrainerRefusal, match="unsupported by"):
        PreferenceTrainer(_cfg("m", "d.jsonl", algorithm="dpo", gamma=1.0))


def test_objective_none_binding_refuses(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pt, "lookup_algorithm", lambda name: SimpleNamespace())
    with pytest.raises(TrainerRefusal, match="0 of 1 required"):
        PreferenceTrainer(_cfg("m", "d.jsonl", algorithm="mystery"))


def test_non_preference_objective_is_family_mismatch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pt, "lookup_algorithm", lambda name: SimpleNamespace(_objective=object()))
    with pytest.raises(TrainerRefusal, match="family mismatch"):
        PreferenceTrainer(_cfg("m", "d.jsonl", algorithm="mystery"))


def test_each_objective_family_accepts_its_supported_knobs() -> None:
    cases = (
        ("dpo", {"beta": 0.2, "sft_weight": 0.1}, "beta", 0.2),
        ("ipo", {"tau": 0.5, "ipo_length_normalise": True}, "tau", 0.5),
        ("kto", {"beta": 0.3, "kto_reference_point": 0.1}, "beta", 0.3),
        ("orpo", {"lambda_": 0.7}, "lambda_", 0.7),
        ("simpo", {"beta": 2.0, "gamma": 0.5}, "gamma", 0.5),
        ("cpo", {"beta": 0.4, "lambda_": 0.6}, "lambda_", 0.6),
    )
    for algorithm, knobs, attribute, expected in cases:
        trainer = PreferenceTrainer(_cfg("m", "d.jsonl", algorithm=algorithm, **knobs))
        assert getattr(trainer._objective, attribute) == pytest.approx(expected), (
            f"{algorithm}: knob {attribute} did not round-trip through the config"
        )


# ---------------------------------------------------------------------------
# Encoding refusals via the private encoding surface (only reachable way)
# ---------------------------------------------------------------------------


class _FakeTokenizer:
    """Whitespace word tokenizer with deterministic small ids."""

    pad_token_id = 0

    def __call__(self, text: str, *, add_special_tokens: bool = False) -> dict[str, list[int]]:
        ids = [(sum(ord(char) for char in word) % 60) + 1 for word in text.split()]
        return {"input_ids": ids}

    def apply_chat_template(self, messages: Any, **kwargs: Any) -> Any:
        return " ".join(message["content"] for message in messages)


def _trainer(**overrides: Any) -> PreferenceTrainer:
    overrides.setdefault("max_length", 16)
    return PreferenceTrainer(_cfg("m", "d.jsonl", **overrides))


def _record(**fields: Any) -> dict[str, Any]:
    record = {"prompt": "w1 w2", "chosen": "w3", "rejected": "w4", "_line": 1}
    record.update(fields)
    return record


def test_encode_refuses_tokenizer_failure() -> None:
    class Raiser(_FakeTokenizer):
        def __call__(self, text: str, *, add_special_tokens: bool = False) -> dict[str, list[int]]:
            raise ValueError("kaboom")

    trainer = _trainer()
    with pytest.raises(TrainerRefusal, match="tokenizer encoding failed"):
        trainer._encode_completion(
            tokenizer=Raiser(),
            record=_record(),
            prompt_text="w1",
            completion_text="w2",
            side="chosen",
        )


def test_encode_refuses_zero_token_prompt() -> None:
    class EmptyPrompt(_FakeTokenizer):
        def apply_chat_template(self, messages: Any, **kwargs: Any) -> Any:
            return ""

    trainer = _trainer()
    with pytest.raises(TrainerRefusal, match="0 tokens"):
        trainer._encode_record_completion(
            tokenizer=EmptyPrompt(), record=_record(), completion="w3", side="chosen"
        )


def test_encode_refuses_zero_token_completion() -> None:
    class EmptyCompletion(_FakeTokenizer):
        def __call__(self, text: str, *, add_special_tokens: bool = False) -> dict[str, list[int]]:
            ids = super().__call__(text, add_special_tokens=add_special_tokens)["input_ids"]
            return {"input_ids": ids if ":" not in text else []}

    trainer = _trainer()
    with pytest.raises(TrainerRefusal, match="encoded to 0 tokens"):
        trainer._encode_completion(
            tokenizer=EmptyCompletion(),
            record=_record(),
            prompt_text="p: x",
            completion_text="w3: w4",
            side="chosen",
        )


def test_encode_refuses_overlong_prompt() -> None:
    trainer = _trainer(max_length=4)
    with pytest.raises(TrainerRefusal, match="truncating a prompt"):
        trainer._encode_completion(
            tokenizer=_FakeTokenizer(),
            record=_record(),
            prompt_text="w1 w2 w3 w4 w5",
            completion_text="w6",
            side="chosen",
        )


def test_encode_refuses_prompt_consuming_all_positions() -> None:
    trainer = _trainer(max_length=4)
    with pytest.raises(TrainerRefusal, match="leaving"):
        trainer._encode_completion(
            tokenizer=_FakeTokenizer(),
            record=_record(),
            prompt_text="w1 w2 w3 w4",
            completion_text="w5",
            side="chosen",
        )


def test_encode_truncates_overlong_completion_and_counts() -> None:
    trainer = _trainer(max_length=6)
    result = trainer._encode_completion(
        tokenizer=_FakeTokenizer(),
        record=_record(),
        prompt_text="w1 w2",
        completion_text="w3 w4 w5 w6 w7 w8",
        side="chosen",
    )
    assert result.truncated is True
    assert len(result.inputs) == 6
    assert trainer.truncated_completion_count == 1
    assert sum(result.target_mask) == 4


def test_encode_refuses_template_exception() -> None:
    class BadTemplate(_FakeTokenizer):
        def apply_chat_template(self, messages: Any, **kwargs: Any) -> Any:
            raise RuntimeError("no template")

    trainer = _trainer()
    with pytest.raises(TrainerRefusal, match="apply_chat_template failed"):
        trainer._encode_record_completion(
            tokenizer=BadTemplate(), record=_record(), completion="w3", side="chosen"
        )


def test_encode_refuses_non_string_template_result() -> None:
    class ListTemplate(_FakeTokenizer):
        def apply_chat_template(self, messages: Any, **kwargs: Any) -> Any:
            return ["not", "a", "string"]

    trainer = _trainer()
    with pytest.raises(TrainerRefusal, match="not str"):
        trainer._encode_record_completion(
            tokenizer=ListTemplate(), record=_record(), completion="w3", side="chosen"
        )


def test_encode_batch_refuses_non_bool_kto_label() -> None:
    trainer = _trainer(algorithm="kto")
    record = {"prompt": "w1 w2", "completion": "w3", "label": "yes", "_line": 7}
    with pytest.raises(TrainerRefusal, match="was not a bool"):
        trainer._encode_batch(
            records=(record,),
            tokenizer=_FakeTokenizer(),
            pad_token_id=0,
            device="cpu",
        )


def test_measure_weight_delta_refuses_without_probe() -> None:
    trainer = _trainer()
    with pytest.raises(TrainerRefusal, match="probe parameters"):
        trainer._measure_weight_delta()


# ---------------------------------------------------------------------------
# _one_step micro-batching and metric contracts (private step, only way in)
# ---------------------------------------------------------------------------


class _TinyCausalLM(torch.nn.Module):
    def __init__(self, vocab: int = 64) -> None:
        super().__init__()
        self.embed = torch.nn.Embedding(vocab, 8)
        self.head = torch.nn.Linear(8, vocab)

    def forward(self, input_ids: Any, attention_mask: Any = None) -> Any:
        return SimpleNamespace(logits=self.head(self.embed(input_ids)))


def _step_fixture(algorithm: str, **overrides: Any) -> tuple[PreferenceTrainer, dict[str, Any]]:
    trainer = _trainer(algorithm=algorithm, max_length=32, **overrides)
    model = _TinyCausalLM()
    probe = model.embed.weight
    trainer._probe_parameter = probe
    trainer._probe_initial = probe.detach().to(device="cpu", dtype=torch.float32, copy=True)
    kwargs: dict[str, Any] = {
        "step": 0,
        "policy_model": model,
        "reference_model": _TinyCausalLM(),
        "tokenizer": _FakeTokenizer(),
        "optimizer": torch.optim.AdamW(model.parameters(), lr=1e-3),
        "device": "cpu",
        "pad_token_id": 0,
        "loss_fn": TensorPreferenceLoss(objective=trainer._objective),
    }
    return trainer, kwargs


def test_one_step_micro_batched_paired() -> None:
    trainer, kwargs = _step_fixture("dpo", logprob_micro_batch=1)
    records = (
        _record(_line=1),
        _record(prompt="w7 w8", chosen="w9", rejected="w10", _line=2),
    )
    report = trainer._one_step(records=records, **kwargs)
    assert report.rows == 2
    assert report.loss.loss == report.loss.loss  # finite (NaN fails equality)
    assert len(trainer.history) == 1


def test_one_step_micro_batch_larger_than_rows_uses_full_reference_slice() -> None:
    trainer, kwargs = _step_fixture("dpo", logprob_micro_batch=64)
    records = (_record(_line=1),)
    report = trainer._one_step(records=records, **kwargs)
    assert report.rows == 1


def test_one_step_refuses_incomplete_paired_metrics(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pt, "preference_metrics", lambda *a, **k: {"accuracy": 1.0})
    trainer, kwargs = _step_fixture("dpo")
    with pytest.raises(TrainerRefusal, match="did not return exactly"):
        trainer._one_step(records=(_record(_line=1),), **kwargs)


def test_one_step_refuses_paired_metrics_for_kto(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pt, "preference_metrics", lambda *a, **k: {"accuracy": 1.0})
    trainer, kwargs = _step_fixture("kto")
    records = ({"prompt": "w1 w2", "completion": "w3", "label": True, "_line": 1},)
    with pytest.raises(TrainerRefusal, match="paired metrics"):
        trainer._one_step(records=records, **kwargs)


# ---------------------------------------------------------------------------
# run() load-surface refusals (exit 96) and TrainerRefusal legs
# ---------------------------------------------------------------------------


def test_run_refuses_exit_96_without_torch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setitem(__import__("sys").modules, "torch", None)
    cfg = _cfg("no-model", str(_paired_dataset(tmp_path)), algorithm="dpo")
    with pytest.raises(SystemExit) as excinfo:
        PreferenceTrainer(cfg).run()
    assert excinfo.value.code == 96
    assert "torch" in capsys.readouterr().err


def test_run_refuses_exit_96_without_transformers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setitem(__import__("sys").modules, "transformers", None)
    cfg = _cfg("no-model", str(_paired_dataset(tmp_path)), algorithm="dpo")
    with pytest.raises(SystemExit) as excinfo:
        PreferenceTrainer(cfg).run()
    assert excinfo.value.code == 96
    assert "transformers" in capsys.readouterr().err


def test_run_refuses_when_tokenizer_cannot_load(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    empty_dir = tmp_path / "empty-model"
    empty_dir.mkdir()
    cfg = _cfg(str(empty_dir), str(_paired_dataset(tmp_path)), algorithm="dpo")
    with pytest.raises(SystemExit) as excinfo:
        PreferenceTrainer(cfg).run()
    assert excinfo.value.code == 96
    assert "AutoTokenizer failed to load" in capsys.readouterr().err


def test_run_refuses_tokenizer_without_pad_or_eos(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    model_dir = _build_model_dir(tmp_path, with_pad=False, with_eos=False)
    cfg = _cfg(str(model_dir), str(_paired_dataset(tmp_path)), algorithm="dpo")
    with pytest.raises(SystemExit) as excinfo:
        PreferenceTrainer(cfg).run()
    assert excinfo.value.code == 96
    assert "neither a pad token nor an eos token" in capsys.readouterr().err


def test_run_falls_back_to_eos_as_pad_token(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    model_dir = _build_model_dir(tmp_path, with_pad=False)
    cfg = _cfg(str(model_dir), str(_paired_dataset(tmp_path)), algorithm="simpo")
    reports = PreferenceTrainer(cfg).run()
    assert len(reports) == 1
    assert "using eos_token as the explicit pad token" in capsys.readouterr().err


def test_run_refuses_when_policy_model_load_fails(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    model_dir = _build_model_dir(tmp_path)
    (model_dir / "config.json").unlink()
    cfg = _cfg(str(model_dir), str(_paired_dataset(tmp_path)), algorithm="dpo")
    with pytest.raises(SystemExit) as excinfo:
        PreferenceTrainer(cfg).run()
    assert excinfo.value.code == 96
    assert "policy model load failed" in capsys.readouterr().err


def test_run_refuses_when_every_parameter_is_frozen(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model_dir = _build_model_dir(tmp_path)

    from transformers import AutoModelForCausalLM, LlamaConfig

    class _FrozenLoader:
        @staticmethod
        def from_pretrained(path: str) -> Any:
            model = AutoModelForCausalLM.from_config(LlamaConfig(vocab_size=128, hidden_size=32))
            model.requires_grad_(False)
            return model

    import transformers

    monkeypatch.setattr(transformers, "AutoModelForCausalLM", _FrozenLoader)
    cfg = _cfg(str(model_dir), str(_paired_dataset(tmp_path)), algorithm="dpo")
    with pytest.raises(TrainerRefusal, match="0 of its parameters require"):
        PreferenceTrainer(cfg).run()


def test_run_refuses_exit_96_when_reference_load_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    model_dir = _build_model_dir(tmp_path)

    import transformers

    real_from_pretrained = transformers.AutoModelForCausalLM.from_pretrained
    calls = {"count": 0}

    def fail_second(path: str) -> Any:
        calls["count"] += 1
        if calls["count"] == 2:
            raise RuntimeError("reference unavailable")
        return real_from_pretrained(path)

    monkeypatch.setattr(transformers.AutoModelForCausalLM, "from_pretrained", fail_second)
    cfg = _cfg(str(model_dir), str(_paired_dataset(tmp_path)), algorithm="dpo")
    with pytest.raises(SystemExit) as excinfo:
        PreferenceTrainer(cfg).run()
    assert excinfo.value.code == 96
    assert "reference model load failed" in capsys.readouterr().err


def test_run_forced_master_weights_auto_device_and_truncation(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """One SimPO step with master weights forced on, device auto-selected, and
    completions that exceed the remaining budget are truncated and printed."""
    import torch

    # Auto-selection must land on the same device on every host; MPS has no float64.
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    model_dir = _build_model_dir(tmp_path)
    long_side = " ".join(f"w{i}" for i in range(20, 40))
    rows = [
        {"prompt": "w1 w2", "chosen": long_side, "rejected": "w5 w6"},
        {"prompt": "w7 w8", "chosen": "w9 w10", "rejected": "w11 w12"},
    ]
    dataset = _write_jsonl(tmp_path / "truncate.jsonl", rows)
    cfg = _cfg(
        str(model_dir),
        str(dataset),
        algorithm="simpo",
        max_length=16,
        master_weights=True,
        device=None,
    )
    trainer = PreferenceTrainer(cfg)
    reports = trainer.run()
    captured = capsys.readouterr()

    assert len(reports) == 1
    assert trainer.truncated_completion_count > 0
    assert "MasterWeightOptimizer(host-fp32)" in captured.err
    assert "forced by config master_weights=True" in captured.err
    assert "truncated" in captured.err
