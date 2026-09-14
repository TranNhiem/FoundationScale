"""The pre-training memory floor: REFUSE foreign occupation, AGREED across ranks (#447).

#447 is the incident this gate exists for. Two foreign ``VLLM::EngineCore`` processes
held 153.9/186.2 GiB of a GB200 tray's 189.5 GiB per device, rank 1 died inside
``Trainer.__init__`` with "15.69 MiB is free", and #380's boundary handler adjudicated
memory this framework never allocated as a defect claim against it. Occupied memory is
an environment fact, so the answer is REFUSE (96), never RED, and the measurement must
say so on every path -- clean ones included, because a measurement that speaks only on
failure is indistinguishable from one that never ran.

Four separations are pinned here, because conflating any two of them is a live defect:

* Cannot-measure vs. clean vs. refuse. A host without CUDA -- and a device that is not
  CUDA -- returns False and prints NO ``memory.weights_fit`` line at all: fabricating a
  clean result there would certify a comparison that never happened.
* Ours vs. foreign. ``memory_reserved`` is subtracted out of the occupation figure:
  "we filled the device" and "someone else did" call for opposite responses, so the
  foreign number must exclude this process's reservation exactly.
* Local vs. agreed. A per-rank refusal lets rank 0 sail into a collective that rank 1
  has already left -- the hang #444/#445 exist to prevent -- so ``_agree_on_stop`` is
  called exactly once, unconditionally, on every path including the happy one, and any
  rank's True must become this rank's "a peer rank" refusal.
* The device asked vs. the device measured. The gate takes the Trainer's OWN device
  (``args.device``) and honours its index; reading ``current_device()`` instead measured
  GPU 0 from every rank, and on the tray GPU 0 was the healthy one -- the gate passed
  while rank 1 still OOM'd on GPU 1.

The last leg is source-order, not behavioural: the call must sit AFTER
``args = _TrainingArguments(`` -- accelerate's PartialState, built there, is the only
thing that initializes the process group ``_agree_on_stop`` needs, and run earlier the
agreement silently degrades to a per-rank guess -- and BEFORE ``trainer = Trainer(``,
whose ``__init__`` performs the allocation the gate exists to pre-empt. Every behavioural
leg above passes wherever the call sits; only source inspection sees placement. Dev
machines have torch but no CUDA, so the CUDA surface is patched attribute-by-attribute
on the real module and restored by monkeypatch at teardown.
"""

from __future__ import annotations

import inspect

import pytest
import torch

from foundationscale.train import loop
from foundationscale.train.loop import EXIT_REFUSE, Step, _device_memory_preempted, _gib

_GIB = 1024**3


class _Parameter:
    """A model parameter carrying exactly the surface the floor sums over."""

    def __init__(self, numel: int, element_size: int) -> None:
        self._numel = numel
        self._element_size = element_size

    def numel(self) -> int:
        return self._numel

    def element_size(self) -> int:
        return self._element_size


class _Model:
    """The smallest object whose ``.parameters()`` the floor can walk."""

    def __init__(self, parameters: list[_Parameter]) -> None:
        self._parameters = parameters

    def parameters(self) -> list[_Parameter]:
        return self._parameters


def _device(device_type: str = "cuda", index: int | None = 0) -> torch.device:
    """A real ``torch.device`` -- the gate only ever reads ``.type`` and ``.index``.

    Constructing one needs no CUDA driver, so it works on the CUDA-less dev machines
    this suite runs on. A namespace double would carry the same two attributes but
    could silently drift from what TrainingArguments actually hands the gate; the real
    object cannot. ``index=None`` builds the bare ``torch.device("cuda")`` of the
    single-visible-device case, where the index's ABSENCE is the thing under test.
    """
    if index is None:
        return torch.device(device_type)
    return torch.device(device_type, index)


class _MarkRecorder:
    """Every ``_mark`` emission in order, so a missing line fails as loudly as a wrong one."""

    def __init__(self) -> None:
        self.calls: list[tuple[object, str]] = []

    def __call__(self, step: object, message: str = "") -> None:
        # The signature is _mark's, exactly -- no ``extra``, no ``**kwargs``. A
        # double WIDER than the thing it replaces accepts calls the real
        # function raises TypeError on, so the suite would certify a call site
        # that cannot run (the mirror of #252, where a double was NARROWER).
        # _mark(step, msg) prints; the structured ``extra={...}`` idiom belongs
        # to the manifest writer, which is a different function.
        self.calls.append((step, message))

    def messages(self, step: object) -> list[str]:
        return [message for recorded, message in self.calls if recorded is step]


def _capture_marks(monkeypatch: pytest.MonkeyPatch) -> _MarkRecorder:
    recorder = _MarkRecorder()
    monkeypatch.setattr(loop, "_mark", recorder)
    return recorder


def _record_agreement(monkeypatch: pytest.MonkeyPatch, *, verdict: bool) -> list[bool]:
    """Replace the collective with a recorder that always answers ``verdict``.

    all_reduce(MAX) is out of scope here -- its own tests cover severity semantics.
    What this function owes is calling it exactly once with the local answer and
    obeying whatever it hands back, which is all the recorder needs to observe.
    """
    calls: list[bool] = []

    def _fake_agree_on_stop(local_stop: bool) -> bool:
        calls.append(local_stop)
        return verdict

    monkeypatch.setattr(loop, "_agree_on_stop", _fake_agree_on_stop)
    return calls


def _install_cuda(
    monkeypatch: pytest.MonkeyPatch,
    *,
    free_bytes: int,
    total_bytes: int,
    reserved_by_us: int,
    current_index: int = 0,
) -> None:
    """Patch the real ``torch.cuda`` surface attribute-by-attribute.

    torch itself must stay real -- the suite imports it elsewhere, and a double in
    ``sys.modules`` would leak into later modules -- so only the four lookups the gate
    performs are replaced, and monkeypatch restores them all at teardown. ``mem_get_info``
    answers the same figures for any index; the per-index variant below is the one for
    the failure this gate exists for, where the devices DISAGREE.
    """
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: current_index)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device: (free_bytes, total_bytes))
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda device: reserved_by_us)


def _install_cuda_per_index(
    monkeypatch: pytest.MonkeyPatch,
    *,
    memory_by_index: dict[int, tuple[int, int]],
    current_index: int,
    reserved_by_us: int = 0,
) -> None:
    """Patch ``torch.cuda`` so ``mem_get_info`` answers per device index.

    The single-figure variant above cannot express the #447 failure shape -- one
    healthy GPU next to one preempted one -- so this one takes the whole map, keyed
    by the integer index the gate hands ``mem_get_info``, and lets
    ``current_device()`` point somewhere else entirely. That divergence IS the test.
    """
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: current_index)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device: memory_by_index[device])
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda device: reserved_by_us)


# --------------------------------------------------------------------------
# A. Measurement: what the floor does with the numbers it can (and cannot) see
# --------------------------------------------------------------------------


def test_a_host_without_cuda_returns_false_and_emits_no_measurement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cannot-measure is not refuse, and it is not clean either: no line is printed.

    Would break if the guard were rewritten to skip measurement but keep marking --
    a ``memory.weights_fit`` line on a CUDA-less host would fabricate a clean result
    from a comparison that never ran, and an operator reading the log could not tell
    "measured and fine" from "never looked".
    """
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    recorder = _capture_marks(monkeypatch)
    calls = _record_agreement(monkeypatch, verdict=False)

    refused = _device_memory_preempted(_Model([_Parameter(_GIB, 4)]), _device())

    assert refused is False
    assert calls == [False]
    fabrications = [m for _, m in recorder.calls if "memory.weights_fit" in m]
    assert not fabrications, f"marks emitted without a measurement: {fabrications}"


def test_free_memory_far_above_the_floor_is_clean_and_says_so(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The happy path emits ``[   ok] memory.weights_fit`` -- clean is declared, not implied.

    Would break if the VALIDATED mark moved inside the refusal branch: the gate would
    still refuse correctly, but a silent success is indistinguishable in the log from
    a gate that never ran, which is the same reason topology.validate_summary always
    prints its summary.
    """
    _install_cuda(monkeypatch, free_bytes=100 * _GIB, total_bytes=189 * _GIB, reserved_by_us=0)
    recorder = _capture_marks(monkeypatch)
    _record_agreement(monkeypatch, verdict=False)
    model = _Model([_Parameter(_GIB // 2, 4)])  # 2.00 GiB of weights

    assert _device_memory_preempted(model, _device()) is False
    ok = [m for m in recorder.messages(Step.VALIDATED) if "[   ok] memory.weights_fit" in m]
    assert len(ok) == 1, f"expected one clean measurement line, saw {recorder.calls}"


def test_free_memory_below_the_floor_refuses_with_the_figures_stated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A local shortfall returns True and the REFUSE mark carries exit 96 and both byte counts.

    Would break if the message dropped a figure or the exit code: a refusal logged
    without them is an accusation with no evidence -- undebuggable on a tray where
    the memory belongs to someone else's process. The code is asserted in the text
    rather than in a structured field because _mark's whole signature is
    ``(step, msg)``; the contract's constants are stated at the return site (#445),
    so the message IS the record.
    """
    free = 1 * _GIB
    weights = 2 * _GIB
    _install_cuda(monkeypatch, free_bytes=free, total_bytes=189 * _GIB, reserved_by_us=0)
    recorder = _capture_marks(monkeypatch)
    _record_agreement(monkeypatch, verdict=True)
    model = _Model([_Parameter(weights // 4, 4)])

    assert _device_memory_preempted(model, _device()) is True
    refusals = recorder.messages(Step.REFUSE)
    assert len(refusals) == 1
    message = refusals[0]
    assert f"refused ({EXIT_REFUSE})" in message
    assert EXIT_REFUSE == 96
    # Both figures, in the operator's units. A shortfall reported as one number
    # cannot be acted on: "1.00 GiB free" and "the weights need 2.00 GiB" are
    # what say whether to wait for a neighbour or to ask for a bigger device.
    assert loop._gib(free) in message
    assert loop._gib(weights) in message
    assert "below the model's weights" in message
    assert "CANNOT-MEASURE and never RED" in message


def test_free_memory_exactly_equal_to_the_floor_is_not_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The comparison is ``free < floor``: equality fits. This is the off-by-one control.

    Would break if the test were ``<=``: a device whose free bytes exactly hold the
    weights would be refused, and every run sized to the advertised device capacity
    would die as CANNOT-MEASURE on hardware that could in fact start.
    """
    _install_cuda(monkeypatch, free_bytes=_GIB, total_bytes=10 * _GIB, reserved_by_us=0)
    recorder = _capture_marks(monkeypatch)
    calls = _record_agreement(monkeypatch, verdict=False)
    model = _Model([_Parameter(_GIB, 1)])  # floor == free, byte for byte

    assert _device_memory_preempted(model, _device()) is False
    assert calls == [False], "equality must ask the collective about False, not True"
    assert not recorder.messages(Step.REFUSE)


# --------------------------------------------------------------------------
# B. Attribution: whose bytes occupy the device
# --------------------------------------------------------------------------


def test_foreign_occupation_excludes_this_process_reservation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """100 total - 10 free - 30 ours = 60 foreign, exactly; our reservation is not blamed.

    Would break if ``memory_reserved`` were dropped from the subtraction: a healthy
    training process's own reads would be reported as a foreign occupant, "WE filled
    the device" and "someone else did" become indistinguishable, and the operator is
    sent hunting a process that does not exist -- or worse, is told the device is
    clean while a peer holds it.
    """
    _install_cuda(
        monkeypatch,
        free_bytes=10 * _GIB,
        total_bytes=100 * _GIB,
        reserved_by_us=30 * _GIB,
    )
    recorder = _capture_marks(monkeypatch)
    _record_agreement(monkeypatch, verdict=False)

    assert _device_memory_preempted(_Model([]), _device()) is False
    lines = [m for m in recorder.messages(Step.VALIDATED) if "memory.weights_fit" in m]
    assert len(lines) == 1
    assert "this process reserves 30.00 GiB" in lines[0]
    assert "another process holds 60.00 GiB" in lines[0]


# --------------------------------------------------------------------------
# C. Agreement: any rank's refusal is every rank's refusal
# --------------------------------------------------------------------------


def test_a_peers_refusal_with_local_fit_says_a_peer_rank_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Local memory is fine, the collective says stop: refuse, and name the peer.

    Would break if the second mark reused the local-shortfall wording: on the tray the
    ranks disagreed (~34.7 GiB free on rank 0, none on rank 1), so rank 0's log must
    say the refusal came from elsewhere -- otherwise the operator debugs the one rank
    whose device was healthy.
    """
    _install_cuda(monkeypatch, free_bytes=50 * _GIB, total_bytes=189 * _GIB, reserved_by_us=0)
    recorder = _capture_marks(monkeypatch)
    calls = _record_agreement(monkeypatch, verdict=True)
    model = _Model([_Parameter(_GIB // 2, 4)])

    assert _device_memory_preempted(model, _device()) is True
    assert calls == [False], "the local answer was clean; only the collective may refuse"
    refusals = recorder.messages(Step.REFUSE)
    assert len(refusals) == 1
    assert "a peer rank could not fit the model weights" in refusals[0]


def test_agree_on_stop_is_called_exactly_once_on_the_happy_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One call, unconditionally, with the local answer as the argument.

    Would break if the collective moved inside the measured branch: a CUDA-less rank
    would skip all_reduce while its CUDA peers enter it, and the unmatched collective
    hangs the run precisely the way #444/#445 describe -- the agreement itself must
    never be the thing that goes unmatched across ranks.
    """
    _install_cuda(monkeypatch, free_bytes=100 * _GIB, total_bytes=189 * _GIB, reserved_by_us=0)
    _capture_marks(monkeypatch)
    calls = _record_agreement(monkeypatch, verdict=False)

    assert _device_memory_preempted(_Model([]), _device()) is False
    assert calls == [False], f"expected exactly one agreement call with False, saw {calls}"


def test_the_floor_is_the_sum_of_numel_times_element_size(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two parameters totalling 2 GiB report "the weights alone need 2.00 GiB".

    Would break if the sum dropped element_size (conflating parameters with bytes --
    fp16 vs fp32 differs 2x and the refusal would fire on devices that fit) or if the
    sum over parameters() were replaced by a single tensor: the floor is decidable
    from facts only, and the fact it sums over is every parameter's real allocation.
    """
    _install_cuda(monkeypatch, free_bytes=50 * _GIB, total_bytes=189 * _GIB, reserved_by_us=0)
    recorder = _capture_marks(monkeypatch)
    _record_agreement(monkeypatch, verdict=False)
    model = _Model(
        [
            _Parameter(_GIB // 4, 4),  # 1.00 GiB
            _Parameter(_GIB // 2, 2),  # 1.00 GiB
        ]
    )

    assert _device_memory_preempted(model, _device()) is False
    lines = [m for m in recorder.messages(Step.VALIDATED) if "memory.weights_fit" in m]
    assert len(lines) == 1
    assert "the weights alone need 2.00 GiB" in lines[0]


# --------------------------------------------------------------------------
# D. Device selection: the index asked for is the index measured
# --------------------------------------------------------------------------


def test_an_explicit_device_index_is_honoured_over_current_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """current_device() says GPU 0 (ample), the device says index 1 (full): REFUSE.

    This is the exact regression that made the gate inert on #447, and it must fail
    loudly if ``current_device()`` ever comes back: then every rank would measure GPU
    0 -- ample here, as its ~34.7 GiB were on the tray -- the collective would be asked
    about False, and rank 1 would still die on GPU 1 two minutes in. The load-bearing
    assertion is on the LOCAL answer handed to the collective (``calls == [True]``), so
    a silent return to the global fails this leg even with the verdict agreeing to stop.
    """
    _install_cuda_per_index(
        monkeypatch,
        memory_by_index={0: (100 * _GIB, 189 * _GIB), 1: (1 * _GIB, 189 * _GIB)},
        current_index=0,
    )
    recorder = _capture_marks(monkeypatch)
    calls = _record_agreement(monkeypatch, verdict=True)
    model = _Model([_Parameter(_GIB // 2, 4)])  # 2.00 GiB of weights

    assert _device_memory_preempted(model, _device(index=1)) is True
    assert calls == [True], "the device named index 1, so its shortfall must be the local answer"
    refusals = recorder.messages(Step.REFUSE)
    assert len(refusals) == 1
    assert "below the model's weights" in refusals[0]
    assert loop._gib(1 * _GIB) in refusals[0]


def test_a_bare_cuda_device_without_an_index_falls_back_to_current_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``torch.device("cuda")`` carries no index; there the global is the truthful source.

    Would break if the ``index is None`` fallback were dropped: ``mem_get_info`` would
    receive ``None`` and misbehave or raise inside the gate. It would equally break if
    the fallback read anything but ``current_device()`` -- the single-visible-device
    case is the one place the global IS this rank's own GPU, so it is the only correct
    source when no index exists to honour.
    """
    _install_cuda_per_index(
        monkeypatch,
        memory_by_index={0: (100 * _GIB, 189 * _GIB), 1: (1 * _GIB, 189 * _GIB)},
        current_index=1,
    )
    _capture_marks(monkeypatch)
    calls = _record_agreement(monkeypatch, verdict=True)
    model = _Model([_Parameter(_GIB // 2, 4)])  # 2.00 GiB of weights

    assert _device_memory_preempted(model, _device(index=None)) is True
    assert calls == [True], "the global named GPU 1, and its shortfall must be the local answer"


def test_a_cpu_device_is_not_measured_and_emits_no_measurement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A CPU device cannot be probed for CUDA memory: False, and no ``weights_fit`` line.

    Would break if the ``device.type == "cuda"`` guard were dropped: ``mem_get_info``
    would be called for a CPU device -- raising on real torch or fabricating figures
    on a stub -- and the VALIDATED line would certify a comparison that never happened
    on a device that was never going to run CUDA at all. The patched figures below are
    chosen so that a guard-less gate WOULD refuse; only the guard keeps this rank out
    of the refusal path.
    """
    # One GiB free against a 4 GiB floor: measured, this would refuse.
    _install_cuda(monkeypatch, free_bytes=_GIB, total_bytes=189 * _GIB, reserved_by_us=0)
    recorder = _capture_marks(monkeypatch)
    calls = _record_agreement(monkeypatch, verdict=False)

    assert _device_memory_preempted(_Model([_Parameter(_GIB, 4)]), _device("cpu")) is False
    assert calls == [False], "an unmeasured device asks the collective about False, not True"
    fabrications = [m for _, m in recorder.calls if "memory.weights_fit" in m]
    assert not fabrications, f"a CPU device was 'measured': {fabrications}"


# --------------------------------------------------------------------------
# E. Formatting and wiring: the string, and the gate's position in _train
# --------------------------------------------------------------------------


def test_gib_formats_two_decimals_with_a_gib_suffix_including_zero() -> None:
    """Bytes render as ``N.NN GiB`` -- two decimals always, zero included.

    Would break if the format lost its fixed precision (operators diffing logs read
    "1 GiB" and "1.00 GiB" as different magnitudes) or its suffix: the message's
    audience acts on GiB, and a bare digit-soup figure is exactly the reading error
    this helper exists to prevent.
    """
    assert _gib(0) == "0.00 GiB"
    assert _gib(_GIB) == "1.00 GiB"
    assert _gib(_GIB + _GIB // 2) == "1.50 GiB"
    assert _gib(_GIB // 4) == "0.25 GiB"
    assert _gib(60 * _GIB) == "60.00 GiB"


def test_the_gate_sits_between_training_arguments_and_trainer_in_source() -> None:
    """``_device_memory_preempted(model, args.device)`` sits between two pinned lines.

    The LOWER bound -- after ``args = _TrainingArguments(`` -- is what makes the
    collective REAL: accelerate's PartialState, built inside TrainingArguments'
    ``__post_init__``, is the only thing in this framework that initializes
    torch.distributed. Run earlier and ``_agree_on_stop`` finds no process group and
    silently returns the LOCAL answer -- while ``current_device()`` is still 0 on every
    rank, so all of them measure GPU 0. It is that combination, not either fact alone,
    that was inert on #447: GPU 0 had ~34.7 GiB free against a ~3 GiB floor, every
    rank passed, and rank 1 still died on GPU 1. The UPPER bound -- before
    ``trainer = Trainer(`` -- keeps the gate from being inert the other way:
    ``Trainer.__init__`` -> ``_move_model_to_device`` is the allocation itself, so
    asking afterwards reinstates the crash the gate exists to refuse. Behavioural legs
    cannot see either bound -- every one above passes wherever the call sits.
    """
    src = inspect.getsource(loop._train)
    args_line = src.index("args = _TrainingArguments(")
    # Anchored on the call, not on its argument list. This leg is about WHERE the
    # gate sits; pinning the exact arguments too would make it fail on a change it
    # has no opinion about, and a leg that cries on unrelated edits gets deleted.
    gate = src.index("if _device_memory_preempted(")
    trainer = src.index("trainer = Trainer(")
    assert args_line < gate, (
        "the gate runs before _TrainingArguments, so torch.distributed is not yet "
        "initialized: _agree_on_stop silently returns the local answer and one rank's "
        "refusal never reaches its peers -- the inert-gate shape of #447"
    )
    assert gate < trainer, (
        "the memory gate runs after the Trainer moves the model to the device; "
        "it would ask whether the weights fit in memory they already occupy (#447)"
    )
