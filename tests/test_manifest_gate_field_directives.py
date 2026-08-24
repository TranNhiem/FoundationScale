"""#83 general-form control: the manifest must never again name a gate
record field the gate does not write.

The core check runs over the JOINED runtime strings the emitter
produces, not the raw source: the original false directive was split
across adjacent string literals ("...cross-check against " / "the
gate's adjudicating-torch record..."), so a raw-source grep reads green
while the shipped artifact stays false. Field existence uses substring
grep-semantics, exactly the check that found #83 at count 0.

SCOPE, stated: the runtime scan examines the branch of
_training_stack_entries() this interpreter executes (torch importable
here or not). Because of that, the historical observed-bad wording is
ALSO pinned out of the raw emitter source, so both prose branches stand
under a control for the wording that was actually observed red. A
novel directive shape on the unexecuted branch that matches neither the
regex nor the pins is the stated residual, not a hidden one.

MUST_PASS: today's emitted strings name zero gate fields absent from
tools/live_save_gate.py, over a denominator of 3 emitted entries.
MUST_FIRE: the detector is fed a planted directive and must flag it --
observed red on every run, so the detector cannot rot quietly and a
zero-directive manifest can never masquerade as a passed measurement.
First execution of this control is this change's next CI run.
"""

import importlib.util
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GATE_PATH = ROOT / "tools" / "live_save_gate.py"
EMITTER_PATH = ROOT / "tools" / "emit_run_manifest.py"

DIRECTIVE = re.compile(
    r"cross-check\s+against\s+the\s+gate'?s\s+([A-Za-z][A-Za-z0-9_.\-]*)",
    re.IGNORECASE,
)

HISTORICAL_PHRASES = (
    "the gate records the adjudicating torch regardless",
    "cross-check against the gate",
    "adjudicating-torch record",
)


def _load_emitter():
    spec = importlib.util.spec_from_file_location(
        "emit_run_manifest_under_control", EMITTER_PATH
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {EMITTER_PATH}")  # FAIL CLOSED
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _fields_named_in(strings):
    return [f for s in strings for f in DIRECTIVE.findall(s)]


def _violations(named, gate_src):
    return sorted({f for f in named if f not in gate_src})


def test_must_pass_emitted_strings_name_no_gate_field_the_gate_lacks():
    gate_src = GATE_PATH.read_text(encoding="utf-8")
    assert len(gate_src) > 1000, "unreadable gate source would void this leg"
    entries = _load_emitter()._training_stack_entries()
    # DENOMINATOR: 3 emitted entries (python_executable, python_version,
    # torch_record) examined for directives.
    assert len(entries) == 3
    named = _fields_named_in([v for _, v in entries])
    bad = _violations(named, gate_src)
    assert not bad, (
        f"{len(named)} gate-field directive(s) in emitted strings; "
        f"{len(bad)} never written by tools/live_save_gate.py: {bad} "
        f"(#83 shape)"
    )


def test_must_fire_planted_directive_is_flagged():
    gate_src = GATE_PATH.read_text(encoding="utf-8")
    probe = [
        "cross-check against the gate's never_written_field_qz; "
        "a mismatch indicts the measurement"
    ]
    named = _fields_named_in(probe)
    assert named == ["never_written_field_qz"]
    assert _violations(named, gate_src) == ["never_written_field_qz"]


def test_regression_pin_83_torch_record_is_still_unwritten_gate_side():
    # Today's measured fact (grep -c -> 0). If the gate ever IMPLEMENTS
    # torch provenance (option (a)), this pin MUST go red first and be
    # updated in the same change that updates the emitter's prose --
    # the prose and the gate may never drift apart silently again.
    gate_src = GATE_PATH.read_text(encoding="utf-8")
    assert "torch_record" not in gate_src


def test_regression_pins_historical_wording_absent_from_emitter_source():
    # Units examined: 3 historical phrases over the full emitter source
    # (both prose branches, executed or not). These were observed red
    # against the pre-fix tree; they may never return in any branch.
    emit_src = EMITTER_PATH.read_text(encoding="utf-8")
    for phrase in HISTORICAL_PHRASES:
        assert phrase not in emit_src, (
            f"historical #83 wording returned to the emitter: {phrase!r}"
        )
