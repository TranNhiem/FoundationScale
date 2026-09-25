# SPDX-License-Identifier: Apache-2.0
"""#546: ratio_mean / clip_fraction are checkable from the public step record.

WHAT IS CLAIMED: the ``LossOutput.metrics`` of a measured step carries
``MetricObservation`` entries named ``ratio_mean`` and ``clip_fraction``,
measured off the same kept old/current log-probability tensors the kernel
priced. End-to-end through the trainer seam, a deterministic model gives the
step-1 invariant exactly -- ratio 1.0, clip fraction 0.0, because old and
current are read off the same weights. At the helper level, a perturbed
current reading moves both metrics to values computed here by hand, for the
token scope and for the sequence scope (the ratio the kernel forms when the
objective declares ``ratio_scope == "sequence"``); an objective without
``clip_bounds`` OMITS the clip fraction rather than reporting 0.0.

WHAT IS NOT CLAIMED: that any StepReport field was added -- none is; the
observation rides in LossOutput.metrics because the StepReport contract
forbids derived quantities -- or anything about convergence or real models.
"""

from __future__ import annotations

import math
import sys
import types

import pytest

torch = pytest.importorskip("torch")

import foundationscale.rl.trainer as trainer_module  # noqa: E402 (torch importorskip first)
from foundationscale.rl.trainer import (  # noqa: E402 (torch importorskip first)
    RLTrainConfig,
    RLTrainer,
    _ratio_and_clip_metrics,
)

_PAD = 0
_TOK_A = 61
_TOK_B = 62
_VOCAB = 96


class _Objective:
    """Just the two declarations the metrics read off an objective."""

    def __init__(self, ratio_scope, clip_bounds=(0.8, 1.2)):
        self.ratio_scope = ratio_scope
        if clip_bounds is not None:
            self.clip_bounds = clip_bounds


def _named(metrics):
    return {metric.name: metric.value for metric in metrics}


# --- helper level: identical inputs give the invariant exactly -------------


def test_identical_old_and_current_give_ratio_one_and_clip_zero() -> None:
    old = torch.full((2, 3), -0.5)
    mask = torch.ones((2, 3))
    metrics = _named(
        _ratio_and_clip_metrics(
            objective=_Objective("token"),
            current_logprobs=old.clone(),
            old_logprobs=old,
            mask=mask,
        )
    )
    assert metrics["ratio_mean"] == 1.0
    assert metrics["clip_fraction"] == 0.0


def test_perturbed_current_moves_the_metrics_as_computed_by_hand_token_scope() -> None:
    old = torch.full((2, 3), -0.5)
    cur = old.clone()
    cur[0, 1] += math.log(1.5)  # ratio 1.5, above the (0.8, 1.2) band
    cur[1, 1] += math.log(0.7)  # ratio 0.7, below the band
    mask = torch.tensor([[1, 1, 1], [1, 1, 0]], dtype=torch.float32)
    metrics = _named(
        _ratio_and_clip_metrics(
            objective=_Objective("token"),
            current_logprobs=cur,
            old_logprobs=old,
            mask=mask,
        )
    )
    # Five supervised positions; ratios 1.0, 1.5, 1.0 on row 0 and 1.0, 0.7
    # on row 1.
    assert metrics["ratio_mean"] == pytest.approx((1.0 + 1.5 + 1.0 + 1.0 + 0.7) / 5, rel=1e-6)
    assert metrics["clip_fraction"] == pytest.approx(2 / 5, rel=1e-9)


def test_sequence_scope_uses_the_kernels_per_row_ratio() -> None:
    old = torch.full((2, 3), -0.5)
    cur = old.clone()
    cur[0, 1] += math.log(1.5)
    cur[1, 1] += math.log(0.5)
    mask = torch.tensor([[1, 1, 1], [1, 1, 0]], dtype=torch.float32)
    metrics = _named(
        _ratio_and_clip_metrics(
            objective=_Objective("sequence"),
            current_logprobs=cur,
            old_logprobs=old,
            mask=mask,
        )
    )
    # The kernel's sequence ratio is exp(masked mean of log-ratios) PER ROW.
    row0 = math.exp(math.log(1.5) / 3)
    row1 = math.exp(math.log(0.5) / 2)
    assert metrics["ratio_mean"] == pytest.approx((row0 + row1) / 2, rel=1e-6)
    # sqrt(0.5) < 0.8 puts row 1 outside the band; row 0 sits inside it.
    assert metrics["clip_fraction"] == pytest.approx(0.5, rel=1e-9)


def test_clip_fraction_is_omitted_not_zeroed_without_a_declared_band() -> None:
    old = torch.full((1, 2), -0.3)
    cur = old.clone()
    cur[0, 0] += 2.0
    metrics = _ratio_and_clip_metrics(
        objective=_Objective("token", clip_bounds=None),
        current_logprobs=cur,
        old_logprobs=old,
        mask=torch.ones((1, 2)),
    )
    assert [metric.name for metric in metrics] == ["ratio_mean"]


def test_a_fully_masked_batch_omits_both_metrics() -> None:
    metrics = _ratio_and_clip_metrics(
        objective=_Objective("token"),
        current_logprobs=torch.zeros((1, 2)),
        old_logprobs=torch.zeros((1, 2)),
        mask=torch.zeros((1, 2)),
    )
    assert metrics == ()


# --- end-to-end: the first measured step carries the invariant -------------


class _FakeTokenizer:
    def __init__(self) -> None:
        self.chat_template = "fake-template"
        self.pad_token_id = _PAD
        self.eos_token = "<eos>"
        self.padding_side = "right"

    def apply_chat_template(self, conversations, tokenize=False, add_generation_prompt=True):
        return [
            ";".join(f"{turn['role']}:{turn['content']}" for turn in conversation)
            for conversation in conversations
        ]

    def __call__(
        self, *, text, return_tensors=None, padding=None, add_special_tokens=None, images=None
    ):
        rows = [[(ord(ch) % 32) + 8 for ch in line] for line in text]
        width = max(len(row) for row in rows)
        input_ids = torch.tensor(
            [[_PAD] * (width - len(row)) + row for row in rows], dtype=torch.long
        )
        return {"input_ids": input_ids, "attention_mask": (input_ids != _PAD).long()}

    def batch_decode(self, sequences, skip_special_tokens=True):
        return [
            "".join("A" if token == _TOK_A else "B" for token in row) for row in sequences.tolist()
        ]


class _FakeModel(torch.nn.Module):
    """Deterministic embedding-logits model: the old pass and the current
    pass read IDENTICALLY until the optimizer moves something, which is the
    step-1 invariant under test."""

    def __init__(self) -> None:
        super().__init__()
        self.embedding = torch.nn.Embedding(_VOCAB, _VOCAB)
        with torch.no_grad():
            self.embedding.weight.copy_(torch.eye(_VOCAB) * 2.0)

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        return types.SimpleNamespace(logits=self.embedding(input_ids))

    @torch.no_grad()
    def generate(
        self,
        *,
        input_ids,
        attention_mask=None,
        max_new_tokens,
        num_return_sequences,
        do_sample,
        temperature,
        top_p,
        top_k,
        pad_token_id,
        **kwargs,
    ):
        expanded = input_ids.repeat_interleave(num_return_sequences, dim=0)
        continuation = torch.tensor(
            [
                [_TOK_A if row % 2 == 0 else _TOK_B] * max_new_tokens
                for row in range(expanded.shape[0])
            ],
            dtype=torch.long,
        )
        return torch.cat([expanded, continuation], dim=1)


def test_first_measured_step_report_carries_ratio_one_and_clip_zero(monkeypatch) -> None:
    tokenizer = _FakeTokenizer()
    model = _FakeModel()
    samples = (
        types.SimpleNamespace(
            sample_id="s0",
            prompt_turns=(("user", "Pick A."),),
            gold="A",
            images=(),
            video=None,
        ),
        types.SimpleNamespace(
            sample_id="s1",
            prompt_turns=(("user", "Now pick the letter A, please."),),
            gold="A",
            images=(),
            video=None,
        ),
    )
    monkeypatch.setattr(
        trainer_module, "load_sharegpt", lambda dataset, gold_key=None: list(samples)
    )
    monkeypatch.setattr(
        trainer_module,
        "resolve_prompt_surface",
        lambda model_id, needs_images: types.SimpleNamespace(
            kind="tokenizer", surface=tokenizer, reason="fake", supports_images=False
        ),
    )
    loader = types.SimpleNamespace(from_pretrained=lambda *args, **kwargs: model)
    fake_transformers = types.ModuleType("transformers")
    fake_transformers.AutoModelForCausalLM = loader
    fake_transformers.AutoModelForImageTextToText = loader
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    reports = RLTrainer(
        RLTrainConfig(
            model="fake/model",
            dataset="unused.jsonl",
            group_size=2,
            prompts_per_step=2,
            max_steps=1,
            max_new_tokens=2,
            device="cpu",
        )
    ).run()

    assert len(reports) == 1
    metrics = _named(reports[0].loss.metrics)
    assert set(metrics) == {"ratio_mean", "clip_fraction"}, (
        "the step must carry BOTH observability metrics, by name, in the "
        "LossOutput.metrics channel -- not as new StepReport fields"
    )
    # Step 1: old and current logprobs come from the SAME weights, so every
    # supervised ratio is exactly 1 and nothing sits outside the clip band.
    assert metrics["ratio_mean"] == 1.0
    assert metrics["clip_fraction"] == 0.0
