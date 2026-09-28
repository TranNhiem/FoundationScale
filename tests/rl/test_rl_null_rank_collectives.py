"""A NULL rank must issue exactly the collectives a rank with real rows issues.

WHAT IS CLAIMED: under a simulated 2-rank world whose other rank has real
rows, ``RLTrainer._one_step`` on a rank where every row abstains records the
same ordered sequence of collective calls (agree_all / agree_max /
all_reduce_sum) as a rank whose rows all score, and returns a StepReport
rather than leaving early. MEASURED origin: the early-return null path
deadlocked FSDP on 4x GB200 because the other ranks went on to further votes,
forwards and reductions.

WHAT IS NOT CLAIMED: that the backward/step collectives FSDP itself issues
match; those live inside the model and are exercised on real GPUs only.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

torch = pytest.importorskip("torch")

import foundationscale.rl.distributed as dist_mod  # noqa: E402
import foundationscale.rl.trainer as trainer_mod  # noqa: E402
from foundationscale.rl.corpus import Sample  # noqa: E402
from foundationscale.rl.rewards import MCQLetterReward  # noqa: E402
from foundationscale.rl.trainer import RLTrainConfig, RLTrainer  # noqa: E402


def _sample(rid: str) -> Sample:
    return Sample(
        sample_id=rid,
        prompt_turns=(("user", "Pick A or B. Q? Answer with a single letter."),),
        response="A",
        gold="A",
    )


class _ParamModel(torch.nn.Module):  # type: ignore[misc]
    """Log-probs depend on one trainable scalar, so backward has a graph."""

    def __init__(self) -> None:
        super().__init__()
        self.w = torch.nn.Parameter(torch.tensor(0.5))
        self.forwards: list[bool] = []  # grad_enabled per forward call

    def generate(self, **kwargs: Any) -> Any:
        rows = kwargs["input_ids"].shape[0] * kwargs["num_return_sequences"]
        width = kwargs["input_ids"].shape[1] + 2
        return torch.arange(1, width + 1).unsqueeze(0).repeat(rows, 1)

    def forward(self, input_ids: Any, attention_mask: Any = None, **kw: Any) -> Any:
        b, w = input_ids.shape
        self.forwards.append(torch.is_grad_enabled())
        logits = torch.zeros(b, w, 8)
        logits = logits + self.w * torch.arange(8.0)
        return SimpleNamespace(logits=logits)


class _Tokenizer:
    pad_token_id = 0

    def __init__(self, decoded: list[str]) -> None:
        self._decoded = decoded

    def batch_decode(self, sequences: Any, skip_special_tokens: bool = True) -> list[str]:
        return list(self._decoded)


class _UnitAdvantage:
    """Every row kept, alternating +1/-1 so the advantage is never zero."""

    def compute(self, **kwargs: Any) -> Any:
        n = len(kwargs["rewards"])
        width = max(len(row) for row in kwargs["mask"])
        return SimpleNamespace(
            rows=tuple(range(n)),
            weights=tuple(((1.0 if i % 2 == 0 else -1.0),) * width for i in range(n)),
        )


def _loss_fn(**kw: Any) -> Any:
    return -(kw["current_logprobs"] * kw["advantages"] * kw["mask"]).sum() / kw["mask"].sum()


def _install_fake_world(
    monkeypatch: pytest.MonkeyPatch, other_rank_votes: bool, peer_slices: int = 0
) -> list[tuple[str, Any]]:
    """Record every collective; the peer rank votes ``other_rank_votes``."""
    trace: list[tuple[str, Any]] = []

    def agree_all(flag: bool, ctx: Any) -> bool:
        trace.append(("agree_all", None))
        return bool(flag) and other_rank_votes

    def agree_max(value: int, ctx: Any) -> int:
        trace.append(("agree_max", None))
        return max(int(value), peer_slices)

    def all_reduce_sum(value: float, ctx: Any) -> float:
        trace.append(("all_reduce_sum", None))
        return float(value) + 4.0  # the peer contributes 4 measured rows

    monkeypatch.setattr(dist_mod, "agree_all", agree_all)
    monkeypatch.setattr(dist_mod, "agree_max", agree_max)
    monkeypatch.setattr(dist_mod, "all_reduce_sum", all_reduce_sum)
    return trace


def _run(
    decoded: list[str],
    monkeypatch: pytest.MonkeyPatch,
    other_rank_votes: bool,
    *,
    micro_batch: int = 0,
    peer_slices: int = 0,
    model: Any = None,
) -> Any:
    trace = _install_fake_world(monkeypatch, other_rank_votes, peer_slices)
    monkeypatch.setattr(
        trainer_mod,
        "encode_prompts",
        lambda surface, chunk, device: {
            "input_ids": torch.ones(2, 3, dtype=torch.long),
            "attention_mask": torch.ones(2, 3, dtype=torch.long),
        },
    )
    trainer = RLTrainer(
        config=RLTrainConfig(
            model="never-loaded",
            dataset="never-read",
            group_size=2,
            max_steps=1,
            prompts_per_step=2,
            temperature=1.0,
            logprob_micro_batch=micro_batch,
        )
    )
    model = model if model is not None else _ParamModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
    ctx = dist_mod.DistContext(
        rank=1, world_size=2, local_rank=1, device="cpu", is_distributed=True
    )
    report = trainer._one_step(
        step=0,
        chunk=[_sample("p0"), _sample("p1")],
        model=model,
        tokenizer=_Tokenizer(decoded),
        surface=SimpleNamespace(),
        reward=MCQLetterReward(),
        objective=SimpleNamespace(
            advantage_fn=_UnitAdvantage(),
            declaration=lambda: SimpleNamespace(components=("surrogate",), metrics=()),
        ),
        loss_fn=_loss_fn,
        optimizer=optimizer,
        ref_model=None,
        device="cpu",
        ctx=ctx,
    )
    return report, trace


FULL = ["A", "B", "A", "B"]
NULL = ["123 !!!"] * 4


def test_null_rank_issues_the_same_collectives_as_a_full_rank(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    full_report, full_trace = _run(FULL, monkeypatch, other_rank_votes=False)
    null_report, null_trace = _run(NULL, monkeypatch, other_rank_votes=False)
    assert full_report is not None
    assert null_report is not None
    assert full_trace, "positive control: the fake world recorded nothing"
    assert sum(1 for name, _ in full_trace if name == "agree_all") >= 4
    assert null_trace == full_trace


def test_null_rank_reports_zero_loss_participation(
    capsys: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    report, _ = _run(NULL, monkeypatch, other_rank_votes=False)
    assert report is not None
    assert report.reward_stats is None
    assert report.rows == 4  # no local rows: names the peer's 4 measured rows
    assert "participating with zero loss" in capsys.readouterr().err


def test_every_rank_null_is_unmeasured(capsys: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    report, trace = _run(NULL, monkeypatch, other_rank_votes=True)
    assert report is None
    assert [name for name, _ in trace] == ["agree_all"]


def test_null_rank_issues_the_same_model_forwards_as_a_full_rank(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """FSDP all-gathers once per forward, so the forward SEQUENCE must match.

    MEASURED origin: a 1-row null rank took the whole-batch path (2 forwards)
    beside a sliced full rank (3 per slice) and hung 2x GB200 in an all-gather.
    """
    full_model, null_model = _ParamModel(), _ParamModel()
    _run(FULL, monkeypatch, False, micro_batch=1, peer_slices=4, model=full_model)
    _run(NULL, monkeypatch, False, micro_batch=1, peer_slices=4, model=null_model)
    assert len(full_model.forwards) == 12, "positive control: 4 slices x 3 passes"
    assert null_model.forwards == full_model.forwards
