"""SKIP disclosure on the save-gate PASS line (#505).

FoundationScaleSaveGate._adjudicate printed "PASS M/M gates" whenever no gate
blocked, even when some of those gates had returned Verdict.SKIP. The
denominator counts gates RUN over gates REGISTERED; it says nothing about how
many VERIFIED anything. Three abstentions inside a bare "PASS 4/4 gates" is
the overstatement the method's own header forbids -- an abstention that is
visible, and never a PASS -- and GateReport.summary already disclosed the
skips one layer down. The two summaries for the SAME gate set disagreed, and
only the quieter one was honest.

The fix mirrors GateReport.summary: a clear report with K skips emits
"PASS M/M gates over <ckpt> (M-K of M verified; K declared SKIP)", while a
report with zero skips keeps the byte-identical bare line -- no trailing
parenthetical at all -- so anything grepping the old string sees no diff
until there is something to disclose. A blocking report still emits RED and
is untouched by the caveat.

run_event is stubbed with a fabricated report: the finding lives in the
formatting of the emitted line, not in gate execution, and fabricating the
report is what makes the fix attributable. If the bare-PASS leg passes while
the disclosure leg does not, the fixture has drifted and the guard is being
credited for something it did not do.
"""

from __future__ import annotations

import logging
import types
from pathlib import Path

import pytest

from foundationscale.train import loop
from foundationscale.train.loop import FoundationScaleSaveGate

_STEP = 7
# run_event is stubbed, so the event is only read for event.value in the
# records dict; a bare namespace answers exactly that and nothing more.
_EVENT = types.SimpleNamespace(value="on_save")


def _ckpt(tmp_path: Path, step: int) -> Path:
    """Build a minimal on-disk checkpoint; return the checkpoint dir itself."""
    ckpt = tmp_path / f"checkpoint-{step}"
    ckpt.mkdir()
    (ckpt / "config.json").write_text("{}")
    (ckpt / "generation_config.json").write_text("{}")
    (ckpt / "rng_state_0.pth").write_bytes(b"\xde\xad\xbe\xef" * 4)
    return ckpt


def _result(gate_id: str, verdict: object) -> types.SimpleNamespace:
    return types.SimpleNamespace(gate_id=gate_id, verdict=verdict, detail="")


def _report(
    results: list[types.SimpleNamespace],
    blocking: list[types.SimpleNamespace] | None = None,
) -> types.SimpleNamespace:
    # registered=None exercises the len(results) fallback for the denominator.
    return types.SimpleNamespace(results=results, registered=None, blocking=blocking or [])


def _gate(monkeypatch: pytest.MonkeyPatch, report: object) -> FoundationScaleSaveGate:
    gate = FoundationScaleSaveGate(registry=[])
    # The context is opaque here BY DESIGN: run_event is stubbed just below, so
    # nothing reads it. Building a real one would drag the whole checkpoint
    # decode surface into a test about one emitted line, and an undecodable
    # fixture would silently divert every case into the UNMEASURED branch --
    # which is exactly how the first draft of this file passed nothing.
    gate.context_builder = lambda ckpt_dir: types.SimpleNamespace(ckpt_dir=ckpt_dir)
    monkeypatch.setattr(loop, "run_event", lambda *a, **k: report)
    return gate


def _emitted(capsys: pytest.CaptureFixture[str], caplog: pytest.LogCaptureFixture) -> str:
    captured = capsys.readouterr()
    return captured.out + captured.err + caplog.text


def test_skip_verdicts_are_disclosed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO)
    ckpt = _ckpt(tmp_path, _STEP)
    report = _report(
        [
            _result("precision", loop.Verdict.PASS),
            _result("format", loop.Verdict.PASS),
            _result("parity", loop.Verdict.SKIP),
        ]
    )
    gate = _gate(monkeypatch, report)

    stop = gate._adjudicate(_EVENT, ckpt)

    text = _emitted(capsys, caplog)
    assert stop is False
    assert "PASS 3/3 gates" in text
    assert "(2 of 3 verified; 1 declared SKIP)" in text


def test_multiple_skips_are_counted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO)
    ckpt = _ckpt(tmp_path, _STEP)
    report = _report(
        [
            _result("precision", loop.Verdict.PASS),
            _result("parity", loop.Verdict.SKIP),
            _result("freshness", loop.Verdict.SKIP),
            _result("lineage", loop.Verdict.SKIP),
        ]
    )
    gate = _gate(monkeypatch, report)

    stop = gate._adjudicate(_EVENT, ckpt)

    text = _emitted(capsys, caplog)
    assert stop is False
    assert "PASS 4/4 gates" in text
    assert "(1 of 4 verified; 3 declared SKIP)" in text


def test_zero_skips_keeps_bare_pass_line(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO)
    ckpt = _ckpt(tmp_path, _STEP)
    report = _report(
        [
            _result("precision", loop.Verdict.PASS),
            _result("format", loop.Verdict.PASS),
        ]
    )
    gate = _gate(monkeypatch, report)

    stop = gate._adjudicate(_EVENT, ckpt)

    text = _emitted(capsys, caplog)
    assert stop is False
    assert "declared SKIP" not in text
    assert "verified" not in text
    line = next(line for line in text.splitlines() if "PASS" in line)
    assert line.rstrip().endswith(f"PASS 2/2 gates over {ckpt}")


def test_blocking_report_still_emits_red(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO)
    ckpt = _ckpt(tmp_path, _STEP)
    # A SKIP sits in the results on purpose: the RED branch must not grow the
    # caveat -- the disclosure belongs to the PASS line and nowhere else.
    report = _report(
        [
            _result("precision", loop.Verdict.PASS),
            _result("parity", loop.Verdict.SKIP),
        ],
        blocking=[types.SimpleNamespace(gate_id="freshness")],
    )
    gate = _gate(monkeypatch, report)

    stop = gate._adjudicate(_EVENT, ckpt)

    text = _emitted(capsys, caplog)
    assert stop is True
    assert "RED 2/2 gates" in text
    assert "freshness" in text
    assert "should_training_stop=True" in text
    assert "declared SKIP" not in text
