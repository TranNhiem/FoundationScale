# SPDX-License-Identifier: Apache-2.0
"""End-to-end CPU smoke tests for the offline preference trainer loop.

These tests run the REAL ``PreferenceTrainer.run()`` path against a tiny
random-weight Llama causal LM and a WordLevel tokenizer built in-process and
saved to disk, so nothing here needs a network, a hub id, or an accelerator.
The model is 2 layers of hidden size 32 over a 128-token vocabulary: small
enough that one forward over the four-row paired batches below is a fraction
of a second on CPU.

WHAT IS CLAIMED: DPO trains for three real steps and its step-1 loss is the
policy-equals-reference value ln(2); SimPO never loads a reference model;
KTO trains on the unpaired label schema with the paper's per-row families;
and the four documented refusal surfaces (missing chat template, mixed
JSONL schema, registry family mismatch, media-carrying records) refuse with
their named reasons rather than a traceback or a silent downgrade.

WHAT IS NOT CLAIMED: convergence, benchmark behaviour, parity with the
oracle plane (that is the tensor-parity suite's job), or anything about a
real model or corpus.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pytest

torch = pytest.importorskip("torch")

from foundationscale.rl.preference_trainer import (  # noqa: E402 (torch importorskip first)
    PreferenceTrainConfig,
    PreferenceTrainer,
    load_pairs_jsonl,
)
from foundationscale.rl.registry import reset_algorithm_registry  # noqa: E402
from foundationscale.rl.trainer import TrainerRefusal  # noqa: E402

# The model config below declares vocab_size=128, so the tokenizer vocab must
# be exactly 3 specials + 125 word tokens == 128 ids; an out-of-range id would
# surface as an embedding index error that looks like a training defect while
# being a fixture defect.
_SPECIAL_TOKENS = ("<unk>", "<pad>", "<eos>")
_WORD_TOKENS = tuple(f"w{i}" for i in range(125))

# A minimal chat template, set explicitly: the trainer REFUSES a tokenizer
# without one rather than concatenating strings, so the happy-path fixture must
# carry one, and the refusal test builds a second model dir that omits it.
_CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{{ message['role'] }}: {{ message['content'] }}\n"
    "{% endfor %}"
    "{% if add_generation_prompt %}assistant: {% endif %}"
)


@pytest.fixture(autouse=True)
def _offline_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Guarantee no helper in this module can contact a hub."""
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    monkeypatch.setenv("HF_DATASETS_OFFLINE", "1")
    monkeypatch.setenv("TOKENIZERS_PARALLELISM", "false")


@pytest.fixture(autouse=True)
def _default_registry() -> None:
    """Reinstall the built-in registry so these runs never inherit a test's entry."""
    reset_algorithm_registry()


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> Path:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


def _build_model_dir(tmp_path: Path, *, with_template: bool = True) -> Path:
    """A real, loadable causal LM + tokenizer on disk in ONE directory.

    The layout mirrors what ``PreferenceTrainer`` assumes when it resolves
    ``config.model``: ``save_pretrained`` of the config-built model writes
    config.json and weights, and ``save_pretrained`` of the
    ``PreTrainedTokenizerFast`` writes a plain tokenizer.json, so both
    ``AutoModelForCausalLM`` and the prompt surface resolve from the same
    local path with no network.
    """
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
    model = AutoModelForCausalLM.from_config(config)
    model.save_pretrained(str(model_dir))

    vocab = {tok: i for i, tok in enumerate(_SPECIAL_TOKENS + _WORD_TOKENS)}
    wordlevel = Tokenizer(WordLevel(vocab=vocab, unk_token="<unk>"))
    wordlevel.pre_tokenizer = Whitespace()
    fast = PreTrainedTokenizerFast(
        tokenizer_object=wordlevel,
        unk_token="<unk>",
        pad_token="<pad>",
        eos_token="<eos>",
    )
    if with_template:
        fast.chat_template = _CHAT_TEMPLATE
    fast.save_pretrained(str(model_dir))
    return model_dir


def _paired_dataset(tmp_path: Path) -> Path:
    rows = [
        {"prompt": "w1 w2", "chosen": "w3 w4", "rejected": "w5 w6"},
        {"prompt": "w7 w8", "chosen": "w9 w10", "rejected": "w11 w12"},
    ]
    return _write_jsonl(tmp_path / "pairs.jsonl", rows)


def _kto_dataset(tmp_path: Path) -> Path:
    rows = [
        {"prompt": "w1 w2", "completion": "w3 w4", "label": True},
        {"prompt": "w5 w6", "completion": "w7 w8", "label": False},
    ]
    return _write_jsonl(tmp_path / "kto.jsonl", rows)


def _cfg(model_dir: Path, dataset: Path, **overrides: Any) -> PreferenceTrainConfig:
    """The smallest honest run declaration: CPU device pinned, two rows a step.

    ``device="cpu"`` is set EXPLICITLY rather than relying on the auto-select:
    a control that only passes on hosts with no accelerator measures the
    machine, not the fixture. ``learning_rate=1e-3`` keeps the weight movement
    assertable on a random-weight fp32 model without needing host fp32 masters
    (the auto path already selects direct AdamW for fp32 params).
    """
    kwargs: dict[str, Any] = {
        "model": str(model_dir),
        "dataset": str(dataset),
        "pairs_per_step": 2,
        "max_steps": 1,
        "learning_rate": 1e-3,
        "device": "cpu",
    }
    kwargs.update(overrides)
    return PreferenceTrainConfig(**kwargs)


def _metric_map(report: Any) -> dict[str, float]:
    return {observation.name: observation.value for observation in report.loss.metrics}


def test_dpo_three_steps_first_loss_is_ln2_and_weights_move(tmp_path: Path) -> None:
    """DPO happy path: 3 measured steps, step-1 loss == ln2, weights move.

    At step 0 policy and reference are two loads of the SAME directory with no
    intervening update, so ``pi == ref`` on every row, every margin is zero,
    and the DPO loss is softplus(0) == ln(2) (both scale knobs wash out:
    ``weight`` defaults to 1.0 and ``sft_weight`` to 0.0). tolerance 1e-4,
    because the two forwards are the same arithmetic on the same weights.
    """
    model_dir = _build_model_dir(tmp_path)
    dataset = _paired_dataset(tmp_path)
    cfg = _cfg(model_dir, dataset, algorithm="dpo", max_steps=3)

    trainer = PreferenceTrainer(cfg)
    reports = trainer.run()

    assert len(reports) == 3, f"expected 3 measured steps, got {len(reports)}"
    assert len(trainer.history) == 3

    first = trainer.history[0]
    assert abs(first["loss"] - math.log(2)) < 1e-4, (
        f"step-0 loss {first['loss']!r} is not ln(2) within 1e-4; the "
        f"policy-equals-reference invariant is broken before any update"
    )
    assert abs(first["margin"]) < 1e-4, (
        f"step-0 margin {first['margin']!r} is not ~0.0 on a policy==reference step"
    )
    assert first["accuracy"] is not None and first["margin"] is not None

    # Accuracy and margin ride the metrics channel of the LossOutput, named
    # exactly as preference_metrics emits them.
    metrics = _metric_map(reports[0])
    assert "accuracy" in metrics and "margin" in metrics

    # The probe-parameter L2 reading must move once updates have landed.
    assert trainer.history[2]["weight_delta_l2"] > 0.0, (
        "weight_delta_l2 is 0 after 3 optimiser steps; the loop ran but the model did not move"
    )
    assert reports[0].rows == 2


def test_simpo_loads_no_reference_model(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """SimPO is reference-FREE: the second model is never loaded, and says so.

    The proof is the recorded line on stderr -- the loader branch for the
    reference model prints exactly one of two sentences, and a run that
    silently loaded a frozen copy would carry the other one. Asserting the
    printed decision is asserting the wiring that produced it.
    """
    model_dir = _build_model_dir(tmp_path)
    dataset = _paired_dataset(tmp_path)
    cfg = _cfg(model_dir, dataset, algorithm="simpo")

    reports = PreferenceTrainer(cfg).run()
    captured = capsys.readouterr()

    assert len(reports) == 1
    assert "reference-free; no second model was loaded" in captured.err, (
        f"SimPO did not record skipping the reference model:\n{captured.err}"
    )
    assert "reference model loaded" not in captured.err
    assert reports[0].loss.loss > 0.0
    metrics = _metric_map(reports[0])
    assert "accuracy" in metrics and "margin" in metrics


def test_kto_run_on_label_schema(tmp_path: Path) -> None:
    """KTO happy path: unpaired records with bool labels train for one step.

    With policy == reference on the first step, the trainer's measured z0 is
    ~0.0 and every row's ``r`` is ~0.0, so each row scores sigmoid(0) == 0.5
    and the loss is 0.5 (weight and both lambdas default to 1.0). The two
    row families must be counted, and the kl_reference_point the rows were
    priced against must be recorded on the report.
    """
    model_dir = _build_model_dir(tmp_path)
    dataset = _kto_dataset(tmp_path)
    cfg = _cfg(model_dir, dataset, algorithm="kto")

    trainer = PreferenceTrainer(cfg)
    reports = trainer.run()

    assert len(reports) == 1
    first = trainer.history[0]
    assert abs(first["loss"] - 0.5) < 1e-3, (
        f"KTO step-0 loss {first['loss']!r} is not sigmoid(0) == 0.5 within "
        f"1e-3 on a policy==reference step"
    )
    # KTO has no pair accuracy to report -- the accuracy/margin history keys
    # exist but abstain, and the metrics channel carries counts instead.
    assert first["accuracy"] is None and first["margin"] is None
    metrics = _metric_map(reports[0])
    assert metrics.get("desirable_count") == 1.0
    assert metrics.get("undesirable_count") == 1.0
    assert abs(metrics.get("kl_reference_point", float("nan"))) < 1e-3, (
        "the recorded KTO reference point is not ~0.0 on a policy==reference batch"
    )
    assert reports[0].rows == 2


def test_missing_chat_template_refuses_exit_96(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A tokenizer without a chat template is REFUSED (exit 96), never
    silently concatenated -- the refusal names the model and the reason."""
    model_dir = _build_model_dir(tmp_path, with_template=False)
    dataset = _paired_dataset(tmp_path)
    cfg = _cfg(model_dir, dataset, algorithm="dpo")

    with pytest.raises(SystemExit) as excinfo:
        PreferenceTrainer(cfg).run()
    captured = capsys.readouterr()

    assert excinfo.value.code == 96
    assert "no chat template" in captured.err, (
        f"the refusal did not name the missing chat template:\n{captured.err}"
    )
    assert str(model_dir) in captured.err


def test_mixed_jsonl_schema_is_refused(tmp_path: Path) -> None:
    """A record carrying BOTH families' fields cannot name one row denominator.

    Tested against ``load_pairs_jsonl`` directly: the refusal is a loader
    property, and driving it through a full run would spend a model load to
    reach a check that fires before any model is touched.
    """
    rows = [
        {
            "prompt": "w1 w2",
            "chosen": "w3 w4",
            "rejected": "w5 w6",
            "completion": "w7 w8",
            "label": True,
        }
    ]
    path = _write_jsonl(tmp_path / "mixed.jsonl", rows)

    with pytest.raises(TrainerRefusal, match="mixed record"):
        load_pairs_jsonl(str(path), paired=True)


def test_group_relative_algorithm_is_a_family_mismatch(tmp_path: Path) -> None:
    """``algorithm='grpo'`` resolves in the registry but is not a preference
    objective, so construction refuses and names the offending key.

    No model or dataset is needed: the objective is resolved inside
    ``PreferenceTrainer.__init__``, well before either path is opened, so the
    refusal cannot depend on loadable files.
    """
    cfg = _cfg(
        tmp_path / "no-model",
        tmp_path / "no-data.jsonl",
        algorithm="grpo",
    )
    with pytest.raises(TrainerRefusal, match=r"algorithm 'grpo'") as excinfo:
        PreferenceTrainer(cfg)
    assert "preference" in str(excinfo.value).lower(), (
        "the family-mismatch refusal dropped the word that makes it actionable"
    )


def test_record_carrying_an_image_field_refuses_exit_96(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A media-carrying record refuses (exit 96) rather than training on its
    prompt alone -- the offline preference trainer is text-only, and says so."""
    model_dir = _build_model_dir(tmp_path)
    rows = [
        {
            "prompt": "w1 w2",
            "chosen": "w3 w4",
            "rejected": "w5 w6",
            "image": "photo.png",
        }
    ]
    dataset = _write_jsonl(tmp_path / "media.jsonl", rows)
    cfg = _cfg(model_dir, dataset, algorithm="dpo")

    with pytest.raises(SystemExit) as excinfo:
        PreferenceTrainer(cfg).run()
    captured = capsys.readouterr()

    assert excinfo.value.code == 96
    assert "text-only" in captured.err, (
        f"the media refusal did not say the trainer is text-only:\n{captured.err}"
    )
