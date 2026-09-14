"""Rank scoping for the FINAL save: the ranks must LEAVE with one agreed code (#445).

#444 scoped the per-checkpoint save gate. This is its twin at the other site that
adjudicates a checkpoint, and the defect there was wider: the final save in
``_train()`` is written by ONE rank, but every rank then globbed ``final_dir``,
counted its shards and ran the save gates over it. On a rank that never wrote,
``final_dir`` either does not exist -- ``FileNotFoundError`` out of ``iterdir()``,
which #380 adjudicates RED and torchrun flattens to a bare 1 (#171) -- or exists
empty, which reads as UNMEASURED. Both were verdicts about the MODEL taken from
an artifact that was never that rank's to hold.

Measured on a GB200 tray (p444, WIDTH=2): every T1-9 arm died with
``[Errno 2] ... '<out>/final'`` at ``local_rank: 1`` while rank 0 saved cleanly,
and T1-12's eager_r1 arm reported "0 safetensors shards ... (contents: [])".

Three things are pinned here, because the fix is only as good as the weakest one:

* ``_agree_on_exit`` agrees on SEVERITY, not on the numeric code. The codes are
  0/5/95/96, so a bare MAX would rank REFUSE(96) above RED(5) -- a real defect
  masked by a machine that would not let a peer measure.
* Abstention is not a vote for PASS, and an all-abstain run is UNMEASURED.
* Every return in the region actually goes through the collective. That is an
  AST leg rather than a behavioural one: a future edit that adds a bare
  ``return EXIT_RED`` below the final save reintroduces exactly #445, and no
  single-process test can see it.

The AST leg selects the SMALLEST function containing the final save. Naming the
enclosing function by hand gets ``train()``, which merely wraps ``_train()`` and
contains zero of these returns -- the guard would pass vacuously over an empty
set. The >= 8 floor exists so that a selector which silently matches nothing
fails loudly instead.
"""

from __future__ import annotations

import ast
import sys
import types
from pathlib import Path

import pytest

from foundationscale.train import loop as loop_mod
from foundationscale.train.loop import (
    _ABSTAIN_SEVERITY,
    _EXIT_SEVERITY,
    EXIT_PASS,
    EXIT_RED,
    EXIT_REFUSE,
    EXIT_UNMEASURED,
    _agree_on_exit,
)

# The final save routes 8 returns through the collective. A selector that drifts
# and matches a smaller region must fail, not quietly certify the remainder.
_MIN_ROUTED_RETURNS = 8


class _FakeTensor:
    """The one-element severity tensor, with just the surface _agree_on_exit uses."""

    def __init__(self, value: int) -> None:
        self.value = int(value)

    def item(self) -> int:
        return self.value


class _FakeReduceOp:
    MAX = "MAX"


def _install_fake_dist(
    monkeypatch: pytest.MonkeyPatch,
    *,
    peers: list[int],
    backend: str = "gloo",
    cuda: bool = False,
) -> dict[str, object]:
    """Stand a process group up in-process; return a record of what was asked of it.

    ``peers`` are the OTHER ranks' severities. all_reduce(MAX) is simulated as
    max(local, *peers), which is the whole contract _agree_on_exit depends on.
    Every entry is installed with monkeypatch.setitem/setattr so the real torch
    is restored at teardown -- the suite imports torch elsewhere, and leaving a
    double in sys.modules would make later modules' verdicts depend on ordering.
    """
    record: dict[str, object] = {"op": None, "device": None, "reduced": False}

    def _all_reduce(tensor: _FakeTensor, op: object = None) -> None:
        record["op"] = op
        record["reduced"] = True
        tensor.value = max([tensor.value, *peers])

    def _tensor(values: list[int], dtype: object = None, device: object = None) -> _FakeTensor:
        record["device"] = device
        return _FakeTensor(values[0])

    fake_dist = types.SimpleNamespace(
        is_available=lambda: True,
        is_initialized=lambda: True,
        get_backend=lambda: backend,
        all_reduce=_all_reduce,
        ReduceOp=_FakeReduceOp,
    )
    fake_torch = types.SimpleNamespace(
        distributed=fake_dist,
        tensor=_tensor,
        int32="int32",
        device=lambda kind, index: f"{kind}:{index}",
        cuda=types.SimpleNamespace(is_available=lambda: cuda, current_device=lambda: 0),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "torch.distributed", fake_dist)
    return record


# --------------------------------------------------------------------------
# A. Single process: the collective is absent, so the local answer stands
# --------------------------------------------------------------------------


@pytest.mark.parametrize("code", [EXIT_PASS, EXIT_RED, EXIT_UNMEASURED, EXIT_REFUSE])
def test_a_single_process_run_keeps_its_own_code(
    monkeypatch: pytest.MonkeyPatch, code: int
) -> None:
    """With no process group the agreement is the identity -- all four states.

    Would break if the helper rewrote codes unconditionally: a plain
    ``python -m foundationscale.train`` run has no peers to agree with, and its
    RED must stay RED rather than being laundered through a severity round-trip.
    """
    monkeypatch.setattr(
        sys.modules["torch.distributed"], "is_initialized", lambda: False, raising=False
    )
    assert _agree_on_exit(code) == code


def test_a_single_process_abstention_is_unmeasured_not_pass(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Abstaining alone means nothing was written anywhere -- 95, never 0.

    Would break if ``None`` fell through to a default of EXIT_PASS: a run that
    saved no final checkpoint at all would report success.
    """
    monkeypatch.setattr(
        sys.modules["torch.distributed"], "is_initialized", lambda: False, raising=False
    )
    assert _agree_on_exit(None) == EXIT_UNMEASURED


# --------------------------------------------------------------------------
# B. Distributed: worst-wins over SEVERITY, not over the numeric code
# --------------------------------------------------------------------------


def test_a_peers_red_outranks_this_ranks_pass(monkeypatch: pytest.MonkeyPatch) -> None:
    """RED on any rank is RED on every rank.

    Would break if the reduce were MIN, or if the severity map were inverted:
    the writing rank's measured defect would be discarded by a peer that had
    nothing to report, which is the disagreement #171 renders undiagnosable.
    """
    _install_fake_dist(monkeypatch, peers=[_EXIT_SEVERITY[EXIT_RED]])
    assert _agree_on_exit(EXIT_PASS) == EXIT_RED


def test_red_outranks_refuse_despite_96_being_the_larger_number(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The regression this whole ranking exists for: 5 must beat 96.

    A bare all_reduce(MAX) over the CODES returns 96, so a rank that measured a
    real defect would be overruled by a rank that merely could not measure.
    Would break the moment someone "simplifies" the severity map away.
    """
    _install_fake_dist(monkeypatch, peers=[_EXIT_SEVERITY[EXIT_REFUSE]])
    assert _agree_on_exit(EXIT_RED) == EXIT_RED


def test_refuse_outranks_unmeasured_and_pass(monkeypatch: pytest.MonkeyPatch) -> None:
    """A machine that would not let a rank measure outranks silence and success."""
    _install_fake_dist(monkeypatch, peers=[_EXIT_SEVERITY[EXIT_PASS]])
    assert _agree_on_exit(EXIT_REFUSE) == EXIT_REFUSE
    _install_fake_dist(monkeypatch, peers=[_EXIT_SEVERITY[EXIT_UNMEASURED]])
    assert _agree_on_exit(EXIT_REFUSE) == EXIT_REFUSE


def test_unmeasured_outranks_pass(monkeypatch: pytest.MonkeyPatch) -> None:
    """Silence beats success: UNMEASURED is not PASS, across ranks as within one."""
    _install_fake_dist(monkeypatch, peers=[_EXIT_SEVERITY[EXIT_UNMEASURED]])
    assert _agree_on_exit(EXIT_PASS) == EXIT_UNMEASURED


def test_an_abstaining_rank_does_not_drag_down_a_peers_pass(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The non-writing rank adopts the writing rank's verdict, and adds nothing.

    This is the whole point of #445: on the tray, rank 1 had no artifact. Its
    abstention must leave rank 0's PASS intact. Would break if abstain were
    mapped to UNMEASURED's severity instead of -1, which would turn every
    healthy 2-rank run into a 95.
    """
    _install_fake_dist(monkeypatch, peers=[_EXIT_SEVERITY[EXIT_PASS]])
    assert _agree_on_exit(None) == EXIT_PASS


def test_an_abstaining_rank_still_adopts_a_peers_red(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Abstention is not immunity -- the rank that wrote a bad checkpoint speaks for all."""
    _install_fake_dist(monkeypatch, peers=[_EXIT_SEVERITY[EXIT_RED]])
    assert _agree_on_exit(None) == EXIT_RED


def test_every_rank_abstaining_is_unmeasured_and_says_so(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Nobody wrote a final checkpoint: 95, and the reason is printed, not implied.

    Would break if the all-abstain branch returned EXIT_PASS (a run that never
    saved reporting success) or returned 95 silently (#444's rule that an
    abstention is declared, never inferred from an absent line).
    """
    _install_fake_dist(monkeypatch, peers=[_ABSTAIN_SEVERITY])
    assert _agree_on_exit(None) == EXIT_UNMEASURED
    out = capsys.readouterr().out
    assert "fs:train:unmeasured" in out
    assert "#445" in out


def test_an_unknown_code_is_treated_as_red(monkeypatch: pytest.MonkeyPatch) -> None:
    """An out-of-contract code fails closed rather than agreeing on a lie.

    Would break if the severity lookup defaulted to PASS: a code outside
    0/5/95/96 -- which is already a defect (#161) -- would be agreed away.
    """
    _install_fake_dist(monkeypatch, peers=[_EXIT_SEVERITY[EXIT_PASS]])
    assert _agree_on_exit(7) == EXIT_RED


def test_the_reduce_is_a_max_and_the_device_follows_the_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """NCCL puts the tensor on the current CUDA device; gloo leaves it on host.

    Would break if the device were unconditionally None under nccl, which makes
    all_reduce raise on a real tray, or unconditionally cuda under gloo, which
    fails on a CPU-only host.
    """
    rec = _install_fake_dist(monkeypatch, peers=[0], backend="nccl", cuda=True)
    _agree_on_exit(EXIT_PASS)
    assert rec["reduced"] is True
    assert rec["op"] == _FakeReduceOp.MAX
    assert rec["device"] == "cuda:0"

    rec = _install_fake_dist(monkeypatch, peers=[0], backend="gloo", cuda=False)
    _agree_on_exit(EXIT_PASS)
    assert rec["device"] is None


def test_nccl_without_cuda_does_not_ask_for_a_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A nccl backend on a host whose CUDA is gone must not call current_device().

    Would break if the guard checked only the backend: current_device() raises
    when no CUDA is available, and #445's own docstring promises that a rank
    which cannot reach its peers reports that, rather than dying inside the
    agreement it was asked to make.
    """
    rec = _install_fake_dist(monkeypatch, peers=[0], backend="nccl", cuda=False)
    assert _agree_on_exit(EXIT_PASS) == EXIT_PASS
    assert rec["device"] is None


# --------------------------------------------------------------------------
# C. AST: every return in the final-save region goes through the collective
# --------------------------------------------------------------------------


def _final_save_region() -> tuple[ast.FunctionDef, int, str]:
    """Return (enclosing function, line of the final_dir assignment, source).

    The enclosing function is the SMALLEST one whose span contains that line.
    Selecting by name instead yields ``train()``, which wraps ``_train()`` and
    holds none of these returns -- the guard would then certify an empty set.
    """
    src = Path(loop_mod.__file__).read_text(encoding="utf-8")
    lines = src.splitlines()
    hits = [i + 1 for i, ln in enumerate(lines) if ln.strip().startswith("final_dir = Path(")]
    assert len(hits) == 1, f"expected exactly one final_dir assignment, found {hits}"
    target = hits[0]

    tree = ast.parse(src)
    candidates = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.lineno <= target
        and (node.end_lineno or node.lineno) >= target
    ]
    assert candidates, "no function encloses the final save"
    smallest = min(candidates, key=lambda n: (n.end_lineno or n.lineno) - n.lineno)
    assert isinstance(smallest, ast.FunctionDef)
    return smallest, target, src


def test_the_final_save_region_is_inside_the_private_trainer() -> None:
    """Pin the selector itself: it must find _train, not its public wrapper.

    Would break if the selection reverted to a by-name lookup of ``train`` --
    the failure mode that made my first scan report "0 returns" and read green.
    """
    fn, target, _ = _final_save_region()
    assert fn.name == "_train"
    assert fn.lineno < target < (fn.end_lineno or fn.lineno)


def test_every_return_after_the_final_save_is_agreed_across_ranks() -> None:
    """No bare ``return EXIT_*`` may survive below the final save.

    This is the regression guard for #445 itself. A return that skips
    _agree_on_exit lets one rank leave with a different code than its peers,
    and torchrun flattens the disagreement to a bare 1 (#171) -- indistinguishable
    from a crash. Would break the moment an edit adds an early return to the
    shard census, the gate loop or the manifest tail.
    """
    fn, target, _ = _final_save_region()
    returns = [
        node
        for node in ast.walk(fn)
        if isinstance(node, ast.Return) and node.lineno >= target and node.value is not None
    ]
    assert len(returns) >= _MIN_ROUTED_RETURNS, (
        f"only {len(returns)} returns found below the final save; the selector has "
        f"drifted and this guard would pass over a region it is not measuring"
    )
    unagreed = [
        node.lineno
        for node in returns
        if not (
            isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Name)
            and node.value.func.id == "_agree_on_exit"
        )
    ]
    assert not unagreed, f"returns at lines {unagreed} leave without agreeing across ranks (#445)"


def test_the_abstain_branch_is_writer_scoped_and_returns_none() -> None:
    """The abstention must be gated on _wrote_this_checkpoint and vote None.

    Would break if the branch were rewritten to return a concrete code: a
    non-writing rank would then assert a verdict about an artifact it does not
    hold, which is #445 restated. Structural rather than behavioural because
    reaching this line for real needs a live Trainer and a process group.
    """
    fn, _, src = _final_save_region()
    lines = src.splitlines()
    guards = [
        node
        for node in ast.walk(fn)
        if isinstance(node, ast.If)
        and "_wrote_this_checkpoint" in ast.dump(node.test)
        and "final_dir" in "\n".join(lines[node.lineno - 1 : (node.end_lineno or node.lineno)])
    ]
    assert len(guards) == 1, f"expected one writer-scoped abstain guard, found {len(guards)}"
    abstains = [
        node
        for node in ast.walk(guards[0])
        if isinstance(node, ast.Return)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Name)
        and node.value.func.id == "_agree_on_exit"
        and len(node.value.args) == 1
        and isinstance(node.value.args[0], ast.Constant)
        and node.value.args[0].value is None
    ]
    assert len(abstains) == 1, "the abstain branch must return _agree_on_exit(None)"


def test_save_model_is_called_on_every_rank() -> None:
    """trainer.save_model must sit ABOVE the writer test, never inside it.

    Under FSDP and DeepSpeed save_model is itself a collective that gathers
    shards from the ranks holding them. Scoping the CALL to the writing rank
    would hang every peer it is waiting on -- a deadlock, which is strictly
    worse than the crash #445 fixes. Would break if a later edit "optimised"
    the call by moving it under the rank guard.
    """
    fn, target, _ = _final_save_region()
    saves = [
        node.lineno
        for node in ast.walk(fn)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "save_model"
    ]
    assert len(saves) == 1, f"expected exactly one save_model call, found {saves}"
    guards = [
        node
        for node in ast.walk(fn)
        if isinstance(node, ast.If) and "_wrote_this_checkpoint" in ast.dump(node.test)
    ]
    assert guards, "no writer-scoped guard found"
    assert saves[0] < guards[0].lineno, (
        "save_model is below the writer guard; a non-writing rank would skip a "
        "collective its peers are blocked on"
    )
    assert target < saves[0], "save_model must follow the final_dir assignment"
