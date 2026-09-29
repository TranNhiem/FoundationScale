"""The estimator-free tails must keep every rank in collective lockstep.

WHAT IS CLAIMED: under a simulated 2-rank world, the reinforce_baseline,
reinforce_pp and SFT (RAFT / best-of-N) tails issue the same ordered sequence
of collectives and the same number of model forwards on a rank with no signal
(null rank, flat returns, no winners) as on a rank with signal, and return a
StepReport rather than leaving early unless EVERY rank is signal-free.
MEASURED origin: rank-local early returns in these tails deadlocked a 2-rank
FSDP reinforce_baseline run on GB200, and the rb/rpp batch statistics were
computed per rank, so ranks disagreed on the baseline.

WHAT IS NOT CLAIMED: that FSDP's own backward collectives match; those live
inside the model and are exercised on real GPUs only.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

torch = pytest.importorskip("torch")

sys.path.insert(0, str(Path(__file__).parent))
from test_rl_reinforce_sft_coverage import (  # noqa: E402 (sibling test module, no package)
    _RecordingOptimizer,
    _ReinforceBaselineStub,
    _ReinforcePPStub,
)
from test_trainer_group_family import _config  # noqa: E402 (sibling test module, no package)

import foundationscale.rl.distributed as dist_mod  # noqa: E402 (torch importorskip first)
from foundationscale.rl.online_objectives import RAFTLoss  # noqa: E402
from foundationscale.rl.torch_backend import TensorPolicyLoss  # noqa: E402
from foundationscale.rl.trainer import RLTrainer  # noqa: E402

_CTX = dist_mod.DistContext(rank=1, world_size=2, local_rank=1, device="cpu", is_distributed=True)


def _install_fake_world(
    monkeypatch: pytest.MonkeyPatch, *, peer_votes: bool, peer_sum: float = 4.0
) -> list[str]:
    """Record every collective; the peer votes ``peer_votes`` and adds ``peer_sum``."""
    trace: list[str] = []

    def agree_all(flag: bool, ctx: Any) -> bool:
        trace.append("agree_all")
        return bool(flag) and peer_votes

    def all_reduce_sum(value: float, ctx: Any) -> float:
        trace.append("all_reduce_sum")
        return float(value) + peer_sum

    monkeypatch.setattr(dist_mod, "agree_all", agree_all)
    monkeypatch.setattr(dist_mod, "all_reduce_sum", all_reduce_sum)
    return trace


class _Planes:
    """A parameter-backed forward that counts calls (FSDP all-gathers per forward)."""

    def __init__(self, rows: int) -> None:
        self.param = torch.nn.Parameter(torch.full((rows, 2), -0.5))
        self.forwards = 0
        self.row_slices = tuple((i, i + 1) for i in range(rows))
        self.leaf = (self.param * 2.0).detach().requires_grad_(True)

    def forward_slice(self, start: int, end: int) -> Any:
        self.forwards += 1
        return (self.param * 2.0).narrow(0, start, end - start)


def _rb(
    scores: list[float], null_rank: bool, *, trainer: Any = None, extra_slices: int = 1
) -> tuple[Any, Any, Any]:
    trainer = trainer or RLTrainer(_config("reinforce_baseline"))
    planes = _Planes(len(scores))
    report = trainer._reinforce_baseline_tail(
        step=0,
        scores=scores,
        kept_current=planes.leaf,
        kept_mask=torch.ones((len(scores), 2)),
        current_logprobs=planes.leaf,
        row_slices=planes.row_slices,
        forward_slice=planes.forward_slice,
        use_logprob_micro_batching=True,
        objective=_ReinforceBaselineStub(),
        optimizer=_RecordingOptimizer(),
        ctx=_CTX,
        null_rank=null_rank,
        extra_slices=extra_slices,
    )
    return report, planes, trainer


def _rpp(scores: list[float], null_rank: bool) -> tuple[Any, Any]:
    planes = _Planes(len(scores))
    report = RLTrainer(_config("reinforce_pp"))._reinforce_pp_tail(
        step=0,
        scores=scores,
        kept_current=planes.leaf,
        kept_old=planes.leaf.detach(),
        kept_mask=torch.ones((len(scores), 2)),
        kept_ref=planes.leaf.detach(),
        current_logprobs=planes.leaf,
        row_slices=planes.row_slices,
        forward_slice=planes.forward_slice,
        use_logprob_micro_batching=True,
        objective=_ReinforcePPStub(),
        loss_fn=TensorPolicyLoss(objective=_ReinforcePPStub()),
        optimizer=_RecordingOptimizer(),
        ctx=_CTX,
        null_rank=null_rank,
        extra_slices=1,
    )
    return report, planes


def _raft(
    rows: list[tuple[int, float]], null_rank: bool, *, extra_slices: int = 1
) -> tuple[Any, Any]:
    planes = _Planes(len(rows))
    report = RLTrainer(_config("raft"))._sft_tail(
        step=0,
        objective=RAFTLoss(),
        rows=rows,
        response_mask=torch.ones((len(rows), 2)),
        forward_slice=planes.forward_slice,
        row_slices=planes.row_slices,
        use_logprob_micro_batching=True,
        optimizer=_RecordingOptimizer(),
        ctx=_CTX,
        null_rank=null_rank,
        extra_slices=extra_slices,
    )
    return report, planes


def test_rb_zero_rank_matches_a_signal_rank(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    signal_trace = _install_fake_world(monkeypatch, peer_votes=False)
    signal_report, signal_planes, _ = _rb([1.0, 0.0, 1.0, 0.0], null_rank=False)
    zero_trace = _install_fake_world(monkeypatch, peer_votes=False)
    zero_report, zero_planes, _ = _rb([1.0, 1.0, 1.0, 1.0], null_rank=False)
    assert signal_report is not None and zero_report is not None
    assert signal_trace, "positive control: the fake world recorded nothing"
    assert zero_trace == signal_trace
    assert zero_planes.forwards == signal_planes.forwards
    assert zero_report.rows == 4  # no local weight: names the peer's 4 measured rows
    assert "participating with zero loss" in capsys.readouterr().err


def test_rb_null_rank_matches_a_signal_rank(monkeypatch: pytest.MonkeyPatch) -> None:
    signal_trace = _install_fake_world(monkeypatch, peer_votes=False)
    _, signal_planes, _ = _rb([1.0, 0.0, 1.0, 0.0], null_rank=False)
    null_trace = _install_fake_world(monkeypatch, peer_votes=False)
    null_report, null_planes, _ = _rb([0.0, 0.0, 0.0, 0.0], null_rank=True)
    assert null_report is not None
    assert null_trace == signal_trace
    assert null_planes.forwards == signal_planes.forwards


def test_rb_every_rank_zero_is_unmeasured(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_world(monkeypatch, peer_votes=True)
    report, _, trainer = _rb([1.0, 1.0], null_rank=False)
    assert report is None
    assert trainer._reinforce_baseline is None, "an unmeasured step must not seed the EMA"


def test_rb_baseline_uses_the_global_batch_mean(monkeypatch: pytest.MonkeyPatch) -> None:
    # Local rows sum to 2 over 4; the fake peer adds 4.0 to every reduction, so
    # the global mean is (2 + 4) / (4 + 4) = 0.75, not the local 0.5.
    _install_fake_world(monkeypatch, peer_votes=False)
    _, _, trainer = _rb([1.0, 0.0, 1.0, 0.0], null_rank=False)
    assert trainer._reinforce_baseline == pytest.approx(0.75)


def test_rpp_null_rank_matches_a_full_rank(monkeypatch: pytest.MonkeyPatch) -> None:
    full_trace = _install_fake_world(monkeypatch, peer_votes=False)
    full_report, full_planes = _rpp([1.0, 0.0, 1.0, 0.0], null_rank=False)
    null_trace = _install_fake_world(monkeypatch, peer_votes=False)
    null_report, null_planes = _rpp([0.0, 0.0, 0.0, 0.0], null_rank=True)
    assert full_report is not None and null_report is not None
    assert full_trace, "positive control: the fake world recorded nothing"
    assert null_trace == full_trace
    assert null_planes.forwards == full_planes.forwards
    assert null_report.rows == 4


def test_raft_flat_rank_matches_a_winner_rank(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    # group_size in _config decides the grouping; both rows of every group share
    # one reward on the flat rank and differ on the winner rank.
    group = RLTrainer(_config("raft")).config.group_size
    winner_rows = [(i, float(i % group == 0)) for i in range(2 * group)]
    flat_rows = [(i, 1.0) for i in range(2 * group)]
    winner_trace = _install_fake_world(monkeypatch, peer_votes=False)
    winner_report, winner_planes = _raft(winner_rows, null_rank=False)
    flat_trace = _install_fake_world(monkeypatch, peer_votes=False)
    flat_report, flat_planes = _raft(flat_rows, null_rank=False)
    assert winner_report is not None and flat_report is not None
    assert winner_trace, "positive control: the fake world recorded nothing"
    assert flat_trace == winner_trace
    assert flat_planes.forwards == winner_planes.forwards
    assert "participating with zero loss" in capsys.readouterr().err


def test_raft_every_rank_flat_is_unmeasured(monkeypatch: pytest.MonkeyPatch) -> None:
    trace = _install_fake_world(monkeypatch, peer_votes=True)
    report, planes = _raft([(0, 1.0), (1, 1.0)], null_rank=False)
    assert report is None
    assert trace == ["agree_all"]
    assert planes.forwards == 0


def test_rb_short_null_rank_pads_to_the_peer_forward_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A null rank with 1 row beside a 4-slice peer must run 3 padding slices.

    MEASURED origin: the rank with fewer micro-batch slices stopped all-gathering
    first and hung its FSDP peer; the pad slices are what keep the counts equal.
    """
    _install_fake_world(monkeypatch, peer_votes=False)
    _, full_planes, _ = _rb([1.0, 0.0, 1.0, 0.0], null_rank=False, extra_slices=0)
    _install_fake_world(monkeypatch, peer_votes=False)
    _, null_planes, _ = _rb([0.0], null_rank=True, extra_slices=3)
    assert full_planes.forwards == 4, "positive control: one backward forward per slice"
    assert null_planes.forwards == full_planes.forwards


def test_raft_short_flat_rank_pads_to_the_peer_forward_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    group = RLTrainer(_config("raft")).config.group_size
    _install_fake_world(monkeypatch, peer_votes=False)
    winner_rows = [(i, float(i % group == 0)) for i in range(2 * group)]
    _, winner_planes = _raft(winner_rows, null_rank=False, extra_slices=0)
    _install_fake_world(monkeypatch, peer_votes=False)
    _, flat_planes = _raft([(i, 1.0) for i in range(group)], null_rank=False, extra_slices=group)
    assert winner_planes.forwards == 4 * group, "positive control: slices x (read + backward)"
    assert flat_planes.forwards == winner_planes.forwards
