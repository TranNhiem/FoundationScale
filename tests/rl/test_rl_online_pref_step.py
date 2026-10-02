"""Online/iterative DPO pricing and FSDP collective alignment for the new step.

WHAT IS CLAIMED: at step 0, when the reference IS the policy, every pair's DPO
loss is -log sigmoid(0) == ln 2, so the reported loss is ln 2 for both
objectives; a NULL rank (no local pair, whether because every decode abstains
or because every in-prompt reward ties so no pair can be formed) issues the
identical collective trace and the identical policy-forward sequence as a full
rank, participates with zero loss, and returns a report naming the peers'
pairs; an every-rank-null step is UNMEASURED after exactly one agree_all;
``maybe_refresh_reference`` copies policy weights into the reference exactly
when ``(step + 1) % cadence == 0``, never at cadence 0, and refuses a
shape-mismatched pair loudly; the routing helpers agree with the contract.

WHAT IS NOT CLAIMED: anything about ``online_pref_step`` internals the
contract does not fix (private helpers, metric names, reduction order beyond
the traced collectives), or that FSDP's own backward collectives align --
those live inside the model and are exercised on real GPUs only.
"""

from __future__ import annotations

import copy
import math
from types import SimpleNamespace
from typing import Any

import pytest

torch = pytest.importorskip("torch")

import foundationscale.rl.distributed as dist_mod  # noqa: E402
from foundationscale.rl.corpus import Sample  # noqa: E402
from foundationscale.rl.online_objectives import IterativeDPOLoss, OnlineDPOLoss  # noqa: E402
from foundationscale.rl.online_pref_step import (  # noqa: E402
    is_online_pref,
    maybe_refresh_reference,
    online_pref_step,
    refresh_cadence,
)
from foundationscale.rl.rewards import MCQLetterReward  # noqa: E402
from foundationscale.rl.trainer import RLTrainConfig, RLTrainer  # noqa: E402

# group_size=2 over 2 gold-"A" prompts: A scores 1.0, B scores 0.0, and the
# unscorable decode abstains entirely -- the same reward alphabet the
# null-rank template exercises.
FULL = ["A", "B", "A", "B"]
NO_SCORE = ["123 !!!"] * 4
TIED = ["A", "A", "A", "A"]


def _sample(rid: str) -> Sample:
    return Sample(
        sample_id=rid,
        prompt_turns=(("user", "Pick A or B. Q? Answer with a single letter."),),
        response="A",
        gold="A",
    )


class _ParamModel(torch.nn.Module):  # type: ignore[misc]
    """Log-probs depend on one trainable scalar, so backward has a graph.

    ``forwards`` records grad-enabled per policy forward; the reference is a
    deep copy, so its forwards never pollute this list.
    """

    def __init__(self) -> None:
        super().__init__()
        self.w = torch.nn.Parameter(torch.tensor(0.5))
        self.forwards: list[bool] = []

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


def _install_fake_world(
    monkeypatch: pytest.MonkeyPatch,
    other_null_votes: bool,
    *,
    peer_pairs: float = 4.0,
    peer_slices: int = 0,
) -> list[tuple[str, None]]:
    """Record every collective; the peer votes ``other_null_votes`` on null,
    holds ``peer_pairs`` pairs, and needs ``peer_slices`` slices."""

    trace: list[tuple[str, None]] = []

    def agree_all(flag: bool, ctx: Any) -> bool:
        trace.append(("agree_all", None))
        return bool(flag) and other_null_votes

    def agree_max(value: int, ctx: Any) -> int:
        trace.append(("agree_max", None))
        return max(int(value), peer_slices)

    def all_reduce_sum(value: float, ctx: Any) -> float:
        trace.append(("all_reduce_sum", None))
        return float(value) + peer_pairs

    monkeypatch.setattr(dist_mod, "agree_all", agree_all)
    monkeypatch.setattr(dist_mod, "agree_max", agree_max)
    monkeypatch.setattr(dist_mod, "all_reduce_sum", all_reduce_sum)
    return trace


def _run(
    decoded: list[str],
    monkeypatch: pytest.MonkeyPatch,
    other_null_votes: bool,
    *,
    micro_batch: int = 0,
    peer_slices: int = 0,
    peer_pairs: float = 4.0,
    model: Any = None,
    objective: Any = None,
) -> tuple[Any, list[tuple[str, None]]]:
    trace = _install_fake_world(
        monkeypatch, other_null_votes, peer_pairs=peer_pairs, peer_slices=peer_slices
    )
    # Patch by dotted path: the step imports the trainer lazily, and a suite
    # that re-imports foundationscale.rl.* (the torch-free census) would leave
    # a module-object patch on a stale copy.
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
            group_size=2,
            max_steps=1,
            prompts_per_step=2,
            temperature=1.0,
            logprob_micro_batch=micro_batch,
        )
    )
    model = model if model is not None else _ParamModel()
    ref_model = copy.deepcopy(model)
    for parameter in ref_model.parameters():
        parameter.requires_grad_(False)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
    ctx = dist_mod.DistContext(
        rank=1, world_size=2, local_rank=1, device="cpu", is_distributed=True
    )
    report = online_pref_step(
        trainer,
        step=0,
        chunk=[_sample("p0"), _sample("p1")],
        model=model,
        tokenizer=_Tokenizer(decoded),
        surface=SimpleNamespace(),
        reward=MCQLetterReward(),
        objective=objective if objective is not None else OnlineDPOLoss(beta=0.1),
        optimizer=optimizer,
        ref_model=ref_model,
        device="cpu",
        ctx=ctx,
    )
    return report, trace


@pytest.mark.parametrize(
    "objective",
    [OnlineDPOLoss(beta=0.1), IterativeDPOLoss(beta=0.1)],
    ids=["online_dpo", "iterative_dpo"],
)
def test_step_zero_reports_ln2_when_reference_equals_policy(
    objective: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """WHAT IS CLAIMED: ref == policy prices every pair at -log sigmoid(0) == ln 2.

    Rewards split inside each prompt (gold "A"; decodes A then B), so both
    prompts yield exactly one pair, and at step 0 every pair's price is ln 2
    independent of beta. The peer contributes zero pairs, so the global pair
    count is the local one and the reduced loss is the plain pair mean.
    """
    report, _ = _run(FULL, monkeypatch, True, objective=objective, peer_pairs=0.0)
    assert report is not None
    assert report.rows == 4  # rows counts the 4 local scored completions
    assert report.loss.loss == pytest.approx(math.log(2), abs=1e-5)


@pytest.mark.parametrize("null_decodes", [NO_SCORE, TIED])
def test_null_rank_matches_full_rank_collectives_and_forwards(
    null_decodes: list[str], capsys: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """WHAT IS CLAIMED: a pair-less rank mirrors a full rank line for line.

    MEASURED origin (see test_rl_null_rank_collectives): an early-return null
    path deadlocked FSDP on 4x GB200 while the peers went on to the votes,
    the per-slice forwards and the backward. Whether the rank is null because
    every decode abstains or because every in-prompt reward ties, it runs the
    agreed slice count on dummy pairs, steps, and returns a report naming the
    peers' global pair count.
    """
    full_model, null_model = _ParamModel(), _ParamModel()
    full_report, full_trace = _run(
        FULL, monkeypatch, False, micro_batch=1, peer_slices=3, model=full_model
    )
    null_report, null_trace = _run(
        null_decodes, monkeypatch, False, micro_batch=1, peer_slices=3, model=null_model
    )
    assert full_trace, "positive control: the fake world recorded nothing"
    assert any(name == "agree_max" for name, _ in full_trace), "positive control"
    assert full_report is not None and full_report.rows == 4  # the 4 local scored completions
    # 3 agreed slices (2 real locally, padded to the peer's 3), and the
    # contract fixes one policy forward per slice.
    assert len(full_model.forwards) == 3, "positive control: one policy forward per slice"
    assert null_trace == full_trace
    assert null_model.forwards == full_model.forwards
    assert null_report is not None
    assert null_report.rows == 4  # zero local pairs: names the peer's 4
    assert "participating with zero loss" in capsys.readouterr().err


def test_every_rank_null_is_unmeasured_after_one_agree_all(
    capsys: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """WHAT IS CLAIMED: when no rank holds a pair, the step is skipped as UNMEASURED.

    Exactly one collective (agree_all) is issued, None is returned, and no
    agree_max, forward, or optimiser step is claimed for a step that priced
    nothing.
    """
    report, trace = _run(NO_SCORE, monkeypatch, True)
    assert report is None
    assert [name for name, _ in trace] == ["agree_all"]
    assert "UNMEASURED" in capsys.readouterr().err


def test_maybe_refresh_reference_copies_exactly_on_cadence() -> None:
    """WHAT IS CLAIMED: cadence 3 copies at steps 2 and 5 of steps 0..5 only.

    On a copy step the reference becomes elementwise equal to the policy; on
    every other step it is left untouched.
    """
    torch.manual_seed(0)
    policy = torch.nn.Linear(4, 4)
    ref = copy.deepcopy(policy)
    refreshed: list[bool] = []
    for step in range(6):
        with torch.no_grad():
            for parameter in policy.parameters():
                parameter.add_(1.0)
        ref_before = [p.detach().clone() for p in ref.parameters()]
        copied = maybe_refresh_reference(step=step, cadence=3, model=policy, ref_model=ref)
        refreshed.append(copied)
        for ref_p, pol_p, before in zip(
            ref.parameters(), policy.parameters(), ref_before, strict=True
        ):
            if copied:
                assert torch.equal(ref_p, pol_p)
            else:
                assert torch.equal(ref_p, before)
    assert refreshed == [False, False, True, False, False, True]


def test_maybe_refresh_reference_cadence_zero_never_copies() -> None:
    """WHAT IS CLAIMED: cadence 0 freezes the reference for the whole run."""
    torch.manual_seed(0)
    policy = torch.nn.Linear(4, 4)
    ref = copy.deepcopy(policy)
    with torch.no_grad():
        for parameter in policy.parameters():
            parameter.add_(1.0)
    ref_before = [p.detach().clone() for p in ref.parameters()]
    for step in range(3):
        assert maybe_refresh_reference(step=step, cadence=0, model=policy, ref_model=ref) is False
    for ref_p, before in zip(ref.parameters(), ref_before, strict=True):
        assert torch.equal(ref_p, before)


def test_maybe_refresh_reference_refuses_shape_mismatch() -> None:
    """WHAT IS CLAIMED: a parameter shape mismatch raises RuntimeError, naming
    the first mismatch rather than copying across it silently."""
    policy = torch.nn.Linear(4, 4)
    ref = torch.nn.Linear(8, 4)  # same parameter count, different first shape
    with pytest.raises(RuntimeError):
        maybe_refresh_reference(step=0, cadence=1, model=policy, ref_model=ref)


def test_is_online_pref_recognises_only_the_two_online_objectives() -> None:
    """WHAT IS CLAIMED: the predicate is True exactly for the two bound classes."""
    assert is_online_pref(OnlineDPOLoss(beta=0.1))
    assert is_online_pref(IterativeDPOLoss(beta=0.1))
    assert not is_online_pref(SimpleNamespace())
    assert not is_online_pref(None)


def test_refresh_cadence_matches_the_contract() -> None:
    """WHAT IS CLAIMED: only iterative DPO refreshes, on the configured cadence.

    Online DPO and any non-online-pref object never refresh; a negative
    cadence is a configuration error and raises.
    """
    assert refresh_cadence(OnlineDPOLoss(beta=0.1), 4) == 0
    assert refresh_cadence(SimpleNamespace(), 4) == 0
    assert refresh_cadence(IterativeDPOLoss(beta=0.1), 4) == 4
    with pytest.raises(ValueError, match="configured"):
        refresh_cadence(IterativeDPOLoss(beta=0.1), -1)
