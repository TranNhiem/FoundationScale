"""PPO step: value head, GAE, step-1 invariant, and FSDP collective lockstep.

WHAT IS CLAIMED: ``ppo_objective`` reads the registry's ``ppo`` binding
(clip band, k3 KL stance, declared components); ``RLTrainer`` resolves that
binding to the PPO step objective instead of refusing it; the value head
starts at V == 0; ``gae`` matches a hand-computed recursion; at step 0 the
graded pass reproduces the no-grad pass, so ratio_mean == 1 and
clip_fraction == 0; under a simulated 2-rank world a rank whose rows all
abstain issues the same ordered collectives and the same number of model
forwards as a rank whose rows all score; and a step where every rank
abstains returns None after exactly one collective.

WHAT IS NOT CLAIMED: that FSDP's own backward collectives match (real GPUs
only), or that the value head learns anything useful in one step.
"""

from __future__ import annotations

import importlib
from types import SimpleNamespace
from typing import Any

import pytest

torch = pytest.importorskip("torch")

import foundationscale.rl.distributed as dist_mod  # noqa: E402
from foundationscale.rl.corpus import Sample  # noqa: E402
from foundationscale.rl.ppo_step import (  # noqa: E402
    KL_WEIGHT,
    PPOStepObjective,
    build_value_head,
    gae,
    ppo_objective,
    ppo_step,
)
from foundationscale.rl.registry import lookup_algorithm  # noqa: E402
from foundationscale.rl.rewards import MCQLetterReward  # noqa: E402
from foundationscale.rl.trainer import RLTrainConfig, RLTrainer  # noqa: E402

HIDDEN = 4
FULL = ["A", "B", "A", "B"]
NULL = ["123 !!!"] * 4


def _sample(rid: str) -> Sample:
    return Sample(
        sample_id=rid,
        prompt_turns=(("user", "Pick A or B. Q? Answer with a single letter."),),
        response="A",
        gold="A",
    )


class _PPOModel(torch.nn.Module):  # type: ignore[misc]
    """Logits and final hidden state both depend on one trainable scalar."""

    def __init__(self) -> None:
        super().__init__()
        self.w = torch.nn.Parameter(torch.tensor(0.5))
        self.config = SimpleNamespace(hidden_size=HIDDEN)
        self.forwards: list[bool] = []

    def generate(self, **kwargs: Any) -> Any:
        rows = kwargs["input_ids"].shape[0] * kwargs["num_return_sequences"]
        width = kwargs["input_ids"].shape[1] + 2
        return torch.arange(1, width + 1).unsqueeze(0).repeat(rows, 1)

    def forward(self, input_ids: Any, attention_mask: Any = None, **kw: Any) -> Any:
        b, w = input_ids.shape
        self.forwards.append(torch.is_grad_enabled())
        logits = torch.zeros(b, w, 8) + self.w * torch.arange(8.0)
        hidden = torch.ones(b, w, HIDDEN) * self.w
        return SimpleNamespace(logits=logits, hidden_states=(hidden,))


class _Tokenizer:
    pad_token_id = 0

    def __init__(self, decoded: list[str]) -> None:
        self._decoded = decoded

    def batch_decode(self, sequences: Any, skip_special_tokens: bool = True) -> list[str]:
        return list(self._decoded)


def _install_world(
    monkeypatch: pytest.MonkeyPatch, *, peer_votes: bool, peer_sum: float
) -> list[str]:
    """Record every collective; the peer votes ``peer_votes`` and adds ``peer_sum``."""
    trace: list[str] = []

    def agree_all(flag: bool, ctx: Any) -> bool:
        trace.append("agree_all")
        return bool(flag) and peer_votes

    def agree_max(value: int, ctx: Any) -> int:
        trace.append("agree_max")
        return int(value)

    def all_reduce_sum(value: float, ctx: Any) -> float:
        trace.append("all_reduce_sum")
        return float(value) + peer_sum

    def all_reduce_grads_mean(module: Any, ctx: Any) -> None:
        trace.append("all_reduce_grads_mean")

    # Patch the distributed module ppo_step itself holds, resolved now: another
    # test module may have re-imported foundationscale.rl.* under us.
    live = importlib.import_module("foundationscale.rl.ppo_step").dist_mod
    monkeypatch.setattr(live, "agree_all", agree_all)
    monkeypatch.setattr(live, "agree_max", agree_max)
    monkeypatch.setattr(live, "all_reduce_sum", all_reduce_sum)
    monkeypatch.setattr(live, "all_reduce_grads_mean", all_reduce_grads_mean)
    return trace


def _objective() -> PPOStepObjective:
    objective = ppo_objective(lookup_algorithm("ppo"))
    assert objective is not None
    return objective


def _run(
    decoded: list[str],
    monkeypatch: pytest.MonkeyPatch,
    *,
    peer_votes: bool = False,
    peer_sum: float = 4.0,
    distributed: bool = True,
) -> tuple[Any, list[str], _PPOModel, Any]:
    trace = _install_world(monkeypatch, peer_votes=peer_votes, peer_sum=peer_sum)
    monkeypatch.setattr(
        "foundationscale.rl.trainer.encode_prompts",
        lambda surface, chunk, device: {
            "input_ids": torch.ones(2, 3, dtype=torch.long),
            "attention_mask": torch.ones(2, 3, dtype=torch.long),
        },
    )
    trainer = RLTrainer(
        config=RLTrainConfig(
            model="never-loaded",
            dataset="never-read",
            algorithm="ppo",
            group_size=2,
            max_steps=1,
            prompts_per_step=2,
            temperature=1.0,
        )
    )
    model = _PPOModel()
    value_head = build_value_head(model, "cpu")
    ctx = dist_mod.DistContext(
        rank=1 if distributed else 0,
        world_size=2 if distributed else 1,
        local_rank=1 if distributed else 0,
        device="cpu",
        is_distributed=distributed,
    )
    report = ppo_step(
        trainer,
        step=0,
        chunk=[_sample("p0"), _sample("p1")],
        model=model,
        tokenizer=_Tokenizer(decoded),
        surface=SimpleNamespace(),
        reward=MCQLetterReward(),
        objective=_objective(),
        optimizer=torch.optim.SGD(model.parameters(), lr=0.0),
        ref_model=None,
        device="cpu",
        ctx=ctx,
        value_head=value_head,
        value_optimizer=torch.optim.SGD(value_head.parameters(), lr=0.1),
    )
    return report, trace, model, value_head


def test_objective_reads_the_registry_binding() -> None:
    objective = _objective()
    assert objective.clip_bounds == pytest.approx((0.8, 1.2))
    assert objective.kl_weight == KL_WEIGHT
    assert objective.value_clip_epsilon is None
    assert set(objective.components) == {"ppo_policy_loss", "value_loss", "kl_penalty"}
    assert "clip_fraction" in objective.metrics


def test_trainer_resolves_ppo_instead_of_refusing() -> None:
    trainer = RLTrainer(
        config=RLTrainConfig(model="never-loaded", dataset="never-read", algorithm="ppo")
    )
    assert isinstance(trainer._resolve_objective(), PPOStepObjective)


def test_value_head_starts_at_zero() -> None:
    head = build_value_head(_PPOModel(), "cpu")
    assert head.weight.dtype == torch.float32
    assert torch.count_nonzero(head(torch.randn(3, HIDDEN))) == 0


def test_gae_matches_a_hand_computed_recursion() -> None:
    # gamma=1, lam=0.5, terminal reward 1 on the last token, V=0.5 everywhere:
    # A2 = 1 - 0.5 = 0.5; A1 = (0 + 0.5 - 0.5) + 0.5*A2 = 0.25; A0 = 0.5*A1.
    rewards = torch.tensor([[0.0, 0.0, 1.0]])
    values = torch.full((1, 3), 0.5)
    mask = torch.ones(1, 3)
    advantages = gae(rewards, values, mask, gamma=1.0, lam=0.5)
    assert advantages[0].tolist() == pytest.approx([0.125, 0.25, 0.5])
    masked = gae(rewards, values, torch.tensor([[0.0, 1.0, 1.0]]), gamma=1.0, lam=0.5)
    assert masked[0, 0].item() == 0.0, "an off-mask position carries no advantage"


def test_step_zero_ratio_is_one_and_nothing_clips(
    capsys: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    report, _, _, value_head = _run(FULL, monkeypatch, peer_sum=0.0, distributed=False)
    assert report is not None
    metrics = {m.name: m.value for m in report.loss.metrics}
    assert metrics["ratio_mean"] == pytest.approx(1.0)
    assert metrics["clip_fraction"] == 0.0
    assert report.rows == 4
    assert torch.count_nonzero(value_head.bias) == 1, "the critic took a step"
    assert "PPO invariant" in capsys.readouterr().err


def test_null_rank_matches_a_full_rank(capsys: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    full_report, full_trace, full_model, _ = _run(FULL, monkeypatch)
    null_report, null_trace, null_model, _ = _run(NULL, monkeypatch)
    assert full_report is not None and null_report is not None
    assert "all_reduce_grads_mean" in full_trace, "positive control: the fake world recorded"
    assert null_trace == full_trace
    assert null_model.forwards == full_model.forwards
    assert null_report.rows == 4  # no local rows: names the peer's 4
    assert null_report.reward_stats is None
    assert "participating with zero loss" in capsys.readouterr().err


def test_every_rank_null_is_unmeasured(capsys: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    report, trace, model, _ = _run(NULL, monkeypatch, peer_votes=True)
    assert report is None
    assert trace == ["agree_all"]
    assert model.forwards == []
    assert "UNMEASURED" in capsys.readouterr().err
