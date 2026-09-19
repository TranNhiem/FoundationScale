"""The declared-covers-execution floor: REFUSE a silently widened run (#492).

``_effective_topology`` reconciles the declared topology against the runtime, but it
sources every field from torchrun's environment, so it returns ``None`` on a plain
``python -m foundationscale.train.cli`` -- the invocation on the README's front page.
On that path nothing compared the declaration to anything, while ``transformers`` sized
the run from the VISIBLE device set and wrapped the model in ``nn.DataParallel``.

Measured 2026-09-18 on a GB200 tray, two arms differing only in how many GPUs the step
exposed, both declaring ``--nodes 1 --gpus-per-node 1 --dp 1``. One visible device:
exit 0, eight steps, PASS. Two visible devices: a hang at step 0 of 8 that did not die
on SIGTERM, was SIGKILLed at 600 s, left a defunct process holding a 682 MiB CUDA
context, and drove the node into ``draining`` with ``Kill task failed``. The cost of
widening silently on this hardware is a cluster node, not a mis-recorded run.

The numbers the doubles carry are measured, not invented: ``TrainingArguments`` built on
that tray reported ``n_gpu=1`` / ``parallel_mode=NOT_PARALLEL`` at one visible device and
``n_gpu=2`` / ``parallel_mode=NOT_DISTRIBUTED`` -- transformers' own name for
DataParallel -- at two.

Four separations are pinned, because conflating any two is a live defect:

* Cannot-measure vs. clean vs. refuse. No ``n_gpu`` means NO measurement line at all:
  printing a clean one would certify a comparison that never happened.
* Clean vs. refuse across the paths that must NOT fire. A CPU host (``n_gpu == 0``) and
  every torchrun rank (``n_gpu == 1`` against a wide declaration) stay clean. A guard
  that refused either would sink CI and ``examples/train_tiny.py``; one-sidedness is the
  contract, not an oversight.
* Local vs. agreed. ``_agree_on_stop`` is called exactly once on every path, and a peer's
  refusal must become this rank's refusal with its own wording.
* The declaration vs. the execution. The gate compares ``cfg.nodes * cfg.gpus_per_node``
  against what transformers PUBLISHES it will drive, never against ``device_count()``.

The last leg is source-order, not behavioural: the call must sit AFTER
``args = _TrainingArguments(``, which is what resolves ``n_gpu`` and ``parallel_mode``,
and BEFORE ``trainer = Trainer(``, whose ``__init__`` performs the DataParallel wrap the
gate exists to pre-empt. Every behavioural leg passes wherever the call sits.
"""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any

import pytest

from foundationscale.train import loop
from foundationscale.train.loop import (
    Step,
    TrainConfig,
    _execution_widens_beyond_declaration,
)

_MEASURE = "topology.declared_covers_execution"


def _cfg(tmp_path: Path, **overrides: Any) -> TrainConfig:
    """A valid config; overrides name the single axis under test."""
    fields: dict[str, Any] = {
        "model": "fake-model",
        "dataset": "fake-dataset",
        "output_dir": tmp_path,
        "nodes": 1,
        "gpus_per_node": 1,
        "profile_name": "local-single-node",
        "dp": 1,
    }
    fields.update(overrides)
    return TrainConfig(**fields)


class _Mode:
    """Carries ``.name`` the way ``transformers.ParallelMode`` does, and nothing else."""

    def __init__(self, name: str) -> None:
        self.name = name


class _Args:
    """The two attributes the gate reads off ``TrainingArguments``, and no others.

    Deliberately NARROW. A double carrying more surface than the gate touches would let
    a future read of some third attribute pass here and fail against the real object;
    absent attributes are constructed by simply not setting them, which is the
    cannot-measure case the first leg pins.
    """

    def __init__(self, **attrs: Any) -> None:
        for key, value in attrs.items():
            setattr(self, key, value)


class _MarkRecorder:
    """Every ``_mark`` emission in order, so a missing line fails as loudly as a wrong one."""

    def __init__(self) -> None:
        self.calls: list[tuple[object, str]] = []

    def __call__(self, step: object, message: str = "") -> None:
        # _mark's signature exactly -- no ``extra``, no ``**kwargs``. A double WIDER
        # than the thing it replaces accepts calls the real function raises TypeError
        # on, certifying a call site that cannot run.
        self.calls.append((step, message))

    def messages(self, step: object) -> list[str]:
        return [message for recorded, message in self.calls if recorded is step]


def _capture_marks(monkeypatch: pytest.MonkeyPatch) -> _MarkRecorder:
    recorder = _MarkRecorder()
    monkeypatch.setattr(loop, "_mark", recorder)
    return recorder


def _record_agreement(monkeypatch: pytest.MonkeyPatch, *, verdict: bool) -> list[bool]:
    """Replace the collective with a recorder that always answers ``verdict``."""
    calls: list[bool] = []

    def _fake_agree_on_stop(local_stop: bool) -> bool:
        calls.append(local_stop)
        return verdict

    monkeypatch.setattr(loop, "_agree_on_stop", _fake_agree_on_stop)
    return calls


# --------------------------------------------------------------------------
# A. Cannot-measure: no reading, therefore no line
# --------------------------------------------------------------------------


def test_absent_n_gpu_returns_false_and_emits_no_measurement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No ``n_gpu`` is unmeasurable, which is neither clean nor refuse.

    Would break if the gate fabricated a clean line for a TrainingArguments that
    resolved no accelerator count -- an operator could then not tell "measured and
    fine" from "never looked", the distinction the sibling memory floor also pins.
    """
    recorder = _capture_marks(monkeypatch)
    calls = _record_agreement(monkeypatch, verdict=False)

    refused = _execution_widens_beyond_declaration(_cfg(tmp_path), _Args())

    assert refused is False
    assert calls == [False]
    fabrications = [m for _, m in recorder.calls if _MEASURE in m]
    assert not fabrications, f"marks emitted without a measurement: {fabrications}"


def test_boolean_n_gpu_is_unmeasurable_not_a_quiet_pass(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``bool`` is an ``int`` subclass, and ``True > 1`` is False -- a silent clean pass.

    A stub that parks a flag on ``n_gpu`` must read as unmeasurable. Would break if the
    isinstance guard dropped its ``not isinstance(n_gpu, bool)`` clause: the comparison
    would then succeed arithmetically and certify a device count nobody measured.
    """
    recorder = _capture_marks(monkeypatch)
    calls = _record_agreement(monkeypatch, verdict=False)

    refused = _execution_widens_beyond_declaration(_cfg(tmp_path), _Args(n_gpu=True))

    assert refused is False
    assert calls == [False]
    assert not [m for _, m in recorder.calls if _MEASURE in m]


# --------------------------------------------------------------------------
# B. Clean: the paths that must never fire
# --------------------------------------------------------------------------


def test_cpu_host_reporting_zero_accelerators_stays_clean(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``n_gpu == 0`` is the CPU run CI and examples/train_tiny.py take every push.

    Would break if the comparison were widened from ``>`` to ``!=`` -- the tempting
    "declaration must equal execution" form, which refuses every CPU run in the repo.
    The VERDICT is what this pins; the marker text is asserted by the #503 control
    below, which is the one that cares that zero accelerators does not read as a pass.
    """
    recorder = _capture_marks(monkeypatch)
    _record_agreement(monkeypatch, verdict=False)

    refused = _execution_widens_beyond_declaration(
        _cfg(tmp_path), _Args(n_gpu=0, parallel_mode=_Mode("NOT_PARALLEL"))
    )

    assert refused is False
    line = recorder.messages(Step.VALIDATED)
    assert len(line) == 1, line


def test_zero_accelerators_against_a_gpu_declaration_is_not_reported_as_ok(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """#503: a CPU fallback must not render as ``[   ok]``, and must not refuse either.

    A GPU job whose devices went away reports the SAME ``n_gpu == 0`` as a deliberate
    CPU run, so the line cannot decide between them -- but it can decline to call one
    of them a pass. Measured on a tray: a node with a GPU in reset state poisoned CUDA
    for the whole node and the job trained on CPU, announced by nothing louder than a
    ``pin_memory`` UserWarning. Both halves are asserted here, because a marker that
    started blocking would take CI's own CPU path down with it.
    """
    recorder = _capture_marks(monkeypatch)
    _record_agreement(monkeypatch, verdict=False)

    refused = _execution_widens_beyond_declaration(
        _cfg(tmp_path), _Args(n_gpu=0, parallel_mode=_Mode("NOT_PARALLEL"))
    )

    assert refused is False
    (line,) = recorder.messages(Step.VALIDATED)
    assert "[   ok]" not in line, line
    assert "[ON CPU]" in line, line
    assert "no throughput, timing or memory number" in line, line
    assert "#503" in line, line


def test_one_visible_device_matching_the_declaration_stays_clean(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The measured green arm: one declared, one driven, NOT_PARALLEL, exit 0."""
    recorder = _capture_marks(monkeypatch)
    _record_agreement(monkeypatch, verdict=False)

    refused = _execution_widens_beyond_declaration(
        _cfg(tmp_path), _Args(n_gpu=1, parallel_mode=_Mode("NOT_PARALLEL"))
    )

    assert refused is False
    (line,) = recorder.messages(Step.VALIDATED)
    assert "[   ok]" in line
    assert "NOT_PARALLEL" in line, "the mechanism transformers names belongs in the log"


def test_a_torchrun_rank_under_a_wide_declaration_stays_clean(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every rank reports ``n_gpu == 1`` while the job declares eight. Never fires.

    This is the legitimate multi-accelerator path. Would break if the gate compared a
    per-process count against a per-node one in the other direction, or refused any
    disagreement: an eight-GPU torchrun job would then refuse on all eight ranks.
    """
    recorder = _capture_marks(monkeypatch)
    _record_agreement(monkeypatch, verdict=False)

    refused = _execution_widens_beyond_declaration(
        _cfg(tmp_path, gpus_per_node=8, dp=8),
        _Args(n_gpu=1, parallel_mode=_Mode("DISTRIBUTED")),
    )

    assert refused is False
    (line,) = recorder.messages(Step.VALIDATED)
    assert "[   ok]" in line


# --------------------------------------------------------------------------
# C. Refuse: the measured incident
# --------------------------------------------------------------------------


def test_two_visible_devices_against_one_declared_refuses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The measured red arm, and the refusal has to be actionable to be worth anything.

    Both numbers appear, so the operator can see which one they meant, and the remedy
    names the environment variable that narrows the visible set. Would break if the
    gate refused without saying what to change -- the difference between a wedge and a
    fixable one is entirely in this message.
    """
    recorder = _capture_marks(monkeypatch)
    calls = _record_agreement(monkeypatch, verdict=True)

    refused = _execution_widens_beyond_declaration(
        _cfg(tmp_path), _Args(n_gpu=2, parallel_mode=_Mode("NOT_DISTRIBUTED"))
    )

    assert refused is True
    assert calls == [True], "the local answer must be the one put to the collective"
    (measurement,) = recorder.messages(Step.VALIDATED)
    assert "[REFUSE]" in measurement and "NOT_DISTRIBUTED" in measurement
    (refusal,) = recorder.messages(Step.REFUSE)
    assert "2 accelerator(s)" in refusal and "declares 1" in refusal
    assert "CUDA_VISIBLE_DEVICES" in refusal, "a refusal with no remedy is a wedge"
    assert "#492" in refusal


def test_a_declaration_wider_than_the_execution_is_not_this_gates_question(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Declared four, driving two: narrower, so clean HERE by deliberate design.

    One-sidedness is the contract. Narrowing is a different claim that was never
    measured, and a gate that answered it on this evidence would be guessing.
    """
    recorder = _capture_marks(monkeypatch)
    _record_agreement(monkeypatch, verdict=False)

    refused = _execution_widens_beyond_declaration(
        _cfg(tmp_path, gpus_per_node=4, dp=4),
        _Args(n_gpu=2, parallel_mode=_Mode("NOT_DISTRIBUTED")),
    )

    assert refused is False
    assert "[   ok]" in recorder.messages(Step.VALIDATED)[0]


# --------------------------------------------------------------------------
# D. Agreement: one rank's refusal is every rank's refusal
# --------------------------------------------------------------------------


def test_a_peer_refusal_becomes_this_ranks_refusal_in_its_own_words(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Locally clean, peer refused: this rank must stop, and must not claim it measured it.

    Would break if the refusal text were built from the local numbers unconditionally --
    a rank that saw a matching topology would print a contradiction of its own clean
    measurement one line above.
    """
    recorder = _capture_marks(monkeypatch)
    calls = _record_agreement(monkeypatch, verdict=True)

    refused = _execution_widens_beyond_declaration(
        _cfg(tmp_path), _Args(n_gpu=1, parallel_mode=_Mode("NOT_PARALLEL"))
    )

    assert refused is True
    assert calls == [False], "the local answer was clean and must be reported as such"
    (refusal,) = recorder.messages(Step.REFUSE)
    assert "a peer rank" in refusal


def test_the_collective_is_called_exactly_once_on_the_unmeasurable_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Unconditional, so the collective itself can never be the unmatched call.

    A gate that skipped ``_agree_on_stop`` when it could not measure would leave ranks
    that CAN measure waiting in an all_reduce with no partner -- the hang #444/#445
    exist to prevent, reintroduced by the guard meant to prevent one.
    """
    _capture_marks(monkeypatch)
    calls = _record_agreement(monkeypatch, verdict=False)

    _execution_widens_beyond_declaration(_cfg(tmp_path), _Args())

    assert calls == [False]


# --------------------------------------------------------------------------
# E. Placement: only source inspection can see it
# --------------------------------------------------------------------------


def test_the_gate_sits_between_training_arguments_and_trainer_in_source() -> None:
    """Pinned from both sides, and neither bound is cosmetic.

    The LOWER bound -- after ``args = _TrainingArguments(`` -- is where ``n_gpu`` and
    ``parallel_mode`` come into existence; asked earlier there is nothing to read and
    the gate is unmeasurable on every host, which is the inert shape. The UPPER bound --
    before ``trainer = Trainer(`` -- is the point of no return: ``Trainer.__init__``
    performs the ``nn.DataParallel`` wrap, so asking afterwards reinstates exactly the
    hang the gate exists to refuse. Behavioural legs cannot see either bound.
    """
    src = inspect.getsource(loop._train)
    args_line = src.index("args = _TrainingArguments(")
    gate = src.index("if _execution_widens_beyond_declaration(")
    trainer = src.index("trainer = Trainer(")
    assert args_line < gate, (
        "the gate runs before _TrainingArguments resolves n_gpu/parallel_mode, so it "
        "can never measure anything and is inert on every host (#492)"
    )
    assert gate < trainer, (
        "the gate runs after Trainer.__init__ has already wrapped the model in "
        "nn.DataParallel -- the wedge it exists to refuse has happened (#492)"
    )
