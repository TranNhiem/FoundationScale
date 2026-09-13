"""Pin the two uncovered arms: the ``logging_steps`` raise and the CUDA readable arm.

Targets in ``foundationscale.train.loop``:

* Lines 415-419 -- the ``logging_steps must be >= 1`` raise in the config
  validation. ``logging_steps=0`` does not mean "log less"; it means the run
  emits no training log at all and is UNMEASURED by construction, so it is a
  config error made at statement time. The test asserts the MESSAGE, not only
  the type, because the message is what the operator acts on. If the
  implementation instead clamped 0 to a default or dropped the raise, this
  test fails -- it can fail.

* Line 1522 -- the ``return True, ""`` arm of ``_cuda_availability``.
  ``(True, "")`` states: this machine HAS a CUDA peak-memory counter the run
  can read, so the manifest gets telemetry and there is no UNMEASURED
  explanation to record. On the CPU-only boxes this suite must run on, the
  real ``torch.cuda.is_available()`` is False and this arm is unreachable
  through real torch -- the function takes the torch module as an injected
  argument precisely so the decision is a unit with inputs, and the tests
  exercise it through that seam with a fake ``cuda`` namespace. The real
  ``torch.cuda`` is never touched. Both False arms are asserted as paired
  controls with their distinct verbatim reasons, so the ``(True, "")``
  assertion is distinguised from "no counter, recorded why" rather than
  passing regardless of which outcome the function returned.

No skips anywhere: this suite runs under FS_FORBID_SKIPS=1.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from foundationscale.train.loop import (
    TrainConfig,
    _cuda_availability,
)


class _FakeCudaModule:
    """Stand-in exposing exactly the seam the decision under test reads.

    The decision function calls ``cuda.is_available()`` and nothing else on the
    ``cuda`` attribute, so this fake is not narrower along any axis the code
    under test can observe. The real ``torch.cuda`` is never imported,
    called, or introspected here -- the house rule for CPU-only boxes.
    """

    def __init__(self, *, available: bool) -> None:
        self._available = available

    def is_available(self) -> bool:
        return self._available


def _cfg(tmp_path: Path, **overrides: object) -> TrainConfig:
    """The smallest TrainConfig that reaches ``__post_init__`` validation.

    These tests construct the config ONLY to exercise statement-time
    validation -- no training runs, so nothing here needs to exist on disk.
    The five required keyword-only fields get cheap placeholders (a str model
    name, a str dataset name, a tmp_path output_dir, nodes=1,
    gpus_per_node=1) and the field under test is passed by name through
    ``overrides``. The real fields are named directly: no candidate-spelling
    scan, so a wrong signature fails here instead of being adapted around.

    ``profile_path`` is supplied because ``__post_init__`` refuses unless
    exactly one of profile / profile_path / profile_name is given, and that
    refusal is raised BEFORE the logging_steps bound these tests pin -- an
    omitted profile would make every test here fail on the wrong sentence.
    The file is never opened (resolution happens inside ``train()``), so a
    path under tmp_path that does not exist is the honest declaration: a
    profile was named, and this test never reads it.
    """
    kwargs: dict = {
        "model": "placeholder-model",
        "dataset": "placeholder-dataset",
        "output_dir": tmp_path / "out",
        "nodes": 1,
        "gpus_per_node": 1,
        "profile_path": tmp_path / "cluster_profile.json",
    }
    kwargs.update(overrides)
    return TrainConfig(**kwargs)


def test_logging_steps_zero_is_refused_with_an_operator_facing_message(tmp_path: Path) -> None:
    """Pin line 416: ``logging_steps=0`` raises ValueError, and the message
    names the field, the bound, the concrete harm, and the UNMEASURED
    consequence the operator must understand."""
    with pytest.raises(ValueError, match=r"logging_steps must be >= 1") as excinfo:
        _cfg(tmp_path, logging_steps=0)
    message = str(excinfo.value)
    assert "logging_steps" in message and ">= 1" in message, (
        f"wanted the refusal to name the field and the bound so the operator knows "
        f"exactly which declaration to fix; got {message!r}"
    )
    assert "0 would emit no training log at all" in message, (
        f"wanted the message to state the concrete harm of logging_steps=0 rather "
        f"than only a bare bound; got {message!r}"
    )
    assert "UNMEASURED by construction" in message, (
        f"wanted the message to name the four-state consequence so the refusal "
        f"reads as protecting measurement, not as pedantry; got {message!r}"
    )


def test_logging_steps_valid_values_construct_without_refusal(tmp_path: Path) -> None:
    """Paired control: the check fires ON the boundary, not around it.

    If validation refused None (leave the Trainer default) or 1 (the boundary
    minimum), the raise pinned above would be over-firing and the zero-test
    would be pinning a bug's neighbour.
    """
    cfg_minimum = _cfg(tmp_path, logging_steps=1)
    assert cfg_minimum.logging_steps == 1, (
        f"wanted logging_steps=1 (the boundary minimum) to survive validation; "
        f"got {cfg_minimum.logging_steps!r}"
    )
    cfg_unset = _cfg(tmp_path, logging_steps=None)
    assert cfg_unset.logging_steps is None, (
        f"wanted logging_steps=None (defer to the Trainer default) to skip the "
        f"check entirely; got {cfg_unset.logging_steps!r}"
    )


def test_cuda_peak_memory_decision_true_pair_pins_line_1522() -> None:
    """``(True, "")`` means: the CUDA peak-memory counter EXISTS and is
    readable, so telemetry is recorded and the manifest gets no UNMEASURED
    text. A non-empty reason here would misreport a readable counter as an
    environment gap."""
    fake_torch = SimpleNamespace(cuda=_FakeCudaModule(available=True))
    available, reason = _cuda_availability(fake_torch)
    assert available is True, (
        f"wanted available=True when cuda.is_available() is True (a counter exists "
        f"to read); got available={available!r}, reason={reason!r}"
    )
    assert reason == "", (
        f"wanted an empty reason on the readable arm -- there is nothing to "
        f"explain when measurement succeeded; got reason={reason!r}"
    )


def test_cuda_peak_memory_decision_false_pair_when_no_cuda_module() -> None:
    """Paired False control, branch one: the build exposes no ``torch.cuda``
    at all -- a different fact about the machine than an unavailable driver,
    and the reason must say so."""
    fake_torch = SimpleNamespace()  # getattr(module, "cuda", None) -> None
    available, reason = _cuda_availability(fake_torch)
    assert available is False, (
        f"wanted available=False when the module has no cuda attribute; "
        f"got available={available!r}, reason={reason!r}"
    )
    assert reason == (
        "UNMEASURED: this torch build exposes no torch.cuda module, so this "
        "run has no CUDA peak-memory counter to read"
    ), (
        f"wanted the no-CUDA-module UNMEASURED wording verbatim so manifests stay "
        f"branch-recognisable across runs; got {reason!r}"
    )


def test_cuda_peak_memory_decision_false_pair_when_cuda_unavailable() -> None:
    """Paired False control, branch two: ``torch.cuda`` exists but reports
    itself unavailable -- the situation on every CPU-only box this suite runs
    on, reached through the fake rather than the real probe."""
    fake_torch = SimpleNamespace(cuda=_FakeCudaModule(available=False))
    available, reason = _cuda_availability(fake_torch)
    assert available is False, (
        f"wanted available=False when cuda.is_available() is False; "
        f"got available={available!r}, reason={reason!r}"
    )
    assert reason == (
        "UNMEASURED: torch.cuda.is_available() is False, so this run has "
        "no CUDA peak-memory counter to read"
    ), (
        f"wanted the is_available-False UNMEASURED wording verbatim so manifests "
        f"stay branch-recognisable across runs; got {reason!r}"
    )


def test_cuda_peak_memory_decision_distinguishes_all_three_outcomes() -> None:
    """The ``(True, "")`` pin above is only meaningful if the decision ACTUALLY
    distinguishes states: both False arms must differ from the True arm AND
    from each other, per the docstring's contract that a manifest reader can
    tell which fact held. If the implementation collapsed two branches onto
    one string, or answered False on the readable arm, this test fails."""
    no_module = _cuda_availability(SimpleNamespace())
    cuda_unavailable = _cuda_availability(SimpleNamespace(cuda=_FakeCudaModule(available=False)))
    cuda_readable = _cuda_availability(SimpleNamespace(cuda=_FakeCudaModule(available=True)))
    false_reasons = {no_module[1], cuda_unavailable[1]}
    assert len(false_reasons) == 2, (
        f"wanted the two False branches to carry DISTINCT reasons so the two "
        f"facts are not conflated in the manifest; got identical reasons "
        f"{no_module[1]!r} from both arms"
    )
    assert cuda_readable == (True, ""), (
        f"wanted the readable arm to be exactly (True, '') and nothing else; got {cuda_readable!r}"
    )
    assert cuda_readable not in (no_module, cuda_unavailable), (
        f"wanted the readable outcome to differ from both UNMEASURED outcomes; "
        f"got {cuda_readable!r} colliding with a False arm"
    )
