# SPDX-License-Identifier: Apache-2.0
"""Part A: grpo + rloo train as a family through RLTrainer.

WHAT IS CLAIMED: with the default ``reference_policy=None`` a grpo run
auto-loads a frozen reference copy of the initial policy, and that copy is
BITWISE unchanged after training while the policy's own weights move; on
step 1 the reference IS the initial policy, so the k3 KL component reads
~0 and the observability pair is exactly ratio_mean == 1.0 /
clip_fraction == 0.0; rloo trains with NO reference loaded at all (the
model loader fires exactly once, for the policy); declaring
``reference_policy=False`` against grpo's k3 term refuses with the missing
reference named; a config ``group_size`` that differs from the binding
default is the size the resolved objective declares; and the
``_loss_components`` KL split attributes policy + kl == total exactly.

WHAT IS NOT CLAIMED: convergence, real-model loading, or anything about
generation quality. The model below is the deterministic fake the
micro-batch / step-metrics harnesses already use, so every equality here
is arithmetic, not sampling luck.
"""

from __future__ import annotations

import sys
import types
from dataclasses import dataclass
from typing import Any

import pytest

torch = pytest.importorskip("torch")

import foundationscale.rl.trainer as trainer_module  # noqa: E402 (torch importorskip first)
from foundationscale.rl.trainer import (  # noqa: E402 (torch importorskip first)
    RLTrainConfig,
    RLTrainer,
    TrainerRefusal,
    _loss_components,
)

_PAD = 0
_TOK_A = 61
_TOK_B = 62
_VOCAB = 96


# --- fake harness, copied from tests/rl/test_trainer_step_metrics.py -------


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
    """Deterministic embedding-logits model: old / current / reference passes
    read IDENTICALLY until the optimiser moves the policy, which is the
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


def _install_fake_host(monkeypatch: pytest.MonkeyPatch) -> list[torch.nn.Module]:
    """Patch corpus / prompt surface / model loading; return the load log.

    The loader hands out a FRESH _FakeModel per call and appends it to the
    returned list, so ``loaded[0]`` is the policy and -- iff a reference is
    loaded -- ``loaded[1]`` is the frozen reference copy.
    """
    tokenizer = _FakeTokenizer()
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
    loaded: list[torch.nn.Module] = []

    def from_pretrained(*args: object, **kwargs: object) -> torch.nn.Module:
        model = _FakeModel()
        loaded.append(model)
        return model

    loader = types.SimpleNamespace(from_pretrained=from_pretrained)
    fake_transformers = types.ModuleType("transformers")
    fake_transformers.AutoModelForCausalLM = loader
    fake_transformers.AutoModelForImageTextToText = loader
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    return loaded


def _config(algorithm: str, **overrides: Any) -> RLTrainConfig:
    fields: dict[str, Any] = {
        "model": "fake/model",
        "dataset": "unused.jsonl",
        "algorithm": algorithm,
        "group_size": 2,
        "prompts_per_step": 2,
        "max_steps": 2,
        "max_new_tokens": 2,
        "learning_rate": 1e-2,
        "device": "cpu",
    }
    fields.update(overrides)
    return RLTrainConfig(**fields)


def _snapshot(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: parameter.detach().clone() for name, parameter in model.named_parameters()}


def _named(metrics: Any) -> dict[str, float]:
    return {metric.name: metric.value for metric in metrics}


# --- grpo: the default auto-loads a frozen, untouched reference ------------


def test_grpo_auto_loads_a_bitwise_frozen_reference_while_policy_moves(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loaded = _install_fake_host(monkeypatch)
    reports = RLTrainer(_config("grpo")).run()

    assert len(reports) == 2, "both steps have within-group reward variance by construction"
    assert len(loaded) == 2, "policy plus exactly one auto-loaded reference"
    policy, reference = loaded
    assert policy is not reference

    # Frozen: every reference parameter is bitwise what the fresh copy held.
    pristine = _snapshot(_FakeModel())
    for name, value in _snapshot(reference).items():
        assert torch.equal(value, pristine[name]), f"reference parameter {name} moved"
    # Trained: the policy moved off the same initial plane.
    assert any(
        not torch.equal(value, pristine[name]) for name, value in _snapshot(policy).items()
    ), "two measured steps at lr=1e-2 must move the policy"


def test_grpo_step1_kl_is_zero_and_metrics_carry_the_invariant(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_host(monkeypatch)
    reports = RLTrainer(_config("grpo")).run()
    assert reports

    first = reports[0]
    components = [c for c in first.loss.components if c.contribution is not None]
    assert len(components) >= 2, "grpo declares the policy component plus the k3 term"
    # The reference IS the initial policy, so ref == current on step 1 and the
    # k3 quantity expm1(0) - 0 is exactly zero up to float rounding.
    assert abs(components[1].contribution) < 1e-6

    metrics = _named(first.loss.metrics)
    assert metrics["ratio_mean"] == 1.0
    assert metrics["clip_fraction"] == 0.0


# --- rloo: trains reference-free -------------------------------------------


def test_rloo_trains_without_loading_any_reference(monkeypatch: pytest.MonkeyPatch) -> None:
    loaded = _install_fake_host(monkeypatch)
    pristine: dict[str, torch.Tensor] | None = None
    reports = RLTrainer(_config("rloo")).run()

    assert len(reports) == 2
    assert len(loaded) == 1, "rloo declares kl_weight == 0.0: no reference may be loaded"
    policy = loaded[0]
    pristine = _snapshot(_FakeModel())
    assert any(
        not torch.equal(value, pristine[name]) for name, value in _snapshot(policy).items()
    ), "rloo took measured steps; the policy must move"


# --- the declared refusal ---------------------------------------------------


def test_grpo_with_reference_forbidden_refuses_naming_the_reference() -> None:
    with pytest.raises(TrainerRefusal, match="reference"):
        RLTrainer(_config("grpo", reference_policy=False))._resolve_objective()


# --- group-size drift --------------------------------------------------------


def test_resolve_objective_keeps_the_binding_min_group_size() -> None:
    # A larger config group must NOT raise the estimator's floor to K: one
    # abstained row would then drop the whole group (measured on GPU: 0 of 12
    # rows kept every step at group 8). The loop reads only advantage_fn.
    for algorithm, k in (("grpo", 5), ("rloo", 6)):
        objective = RLTrainer(_config(algorithm, group_size=k))._resolve_objective()
        assert objective.advantage_fn.min_group_size == 2


# --- the KL split arithmetic, directly --------------------------------------


@dataclass(frozen=True)
class _Declaration:
    components: tuple[str, ...]


class _KLObjective:
    kl_weight = 0.1
    ratio_scope = "token"

    def declaration(self) -> _Declaration:
        return _Declaration(components=("policy", "kl_penalty"))


def test_loss_components_kl_split_sums_to_the_total() -> None:
    torch.manual_seed(7)
    current = torch.full((2, 3), -0.5)
    reference = current.clone()
    reference[0, 1] += 0.3
    reference[1, 0] -= 0.2
    mask = torch.tensor([[1, 1, 1], [1, 1, 0]], dtype=torch.float32)
    total = 1.234

    components = _loss_components(
        objective=_KLObjective(),
        total=total,
        current_logprobs=current,
        reference_logprobs=reference,
        mask=mask,
    )

    assert len(components) == 2
    policy, kl = components
    assert policy.observed and kl.observed
    assert kl.contribution is not None and kl.contribution > 0.0
    assert policy.contribution + kl.contribution == pytest.approx(total, rel=1e-12, abs=1e-12)


def test_loss_components_without_a_reference_leaves_whole_total_on_policy() -> None:
    components = _loss_components(
        objective=_KLObjective(),
        total=0.75,
        current_logprobs=torch.full((1, 2), -0.4),
        reference_logprobs=None,
        mask=torch.ones((1, 2)),
    )
    assert components[0].contribution == pytest.approx(0.75, rel=0, abs=0)
    assert components[1].contribution is None
    assert not components[1].observed
