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
ALSO pinned out of the emitter source with adjacent string literals
joined -- the original false directive was split across literals, so
only a joined-source pin observes the wording the interpreter ships --
and both prose branches stand under that control, executed or not. A
novel directive shape on the unexecuted branch that matches neither the
regex nor the pins is the stated residual, not a hidden one.

MUST_PASS: today's emitted strings name zero gate fields absent from
src/foundationscale/gates/adjudication.py, over a denominator of 3 emitted entries.
MUST_FIRE: the detector is fed a planted directive and must flag it --
observed red on every run, so the detector cannot rot quietly and a
zero-directive manifest can never masquerade as a passed measurement.
First execution of this control is this change's next CI run.
"""

import argparse
import importlib.metadata
import importlib.util
import json
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
# The gate's decision API is a library module since T2_lib_script_boundary#0.
GATE_PATH = ROOT / "src" / "foundationscale" / "gates" / "adjudication.py"
EMITTER_PATH = ROOT / "tools" / "emit_run_manifest.py"

DIRECTIVE = re.compile(
    r"cross-check\s+against\s+the\s+gate'?s\s+([A-Za-z][A-Za-z0-9_.\-]*)",
    re.IGNORECASE,
)

# Historical pre-fix wording, pinned so it can never return in any branch.
# Phrase 2 is pinned at its full SHIPPED length -- joined across the
# literal split ("...cross-check against " / "the gate's adjudicating-
# torch record...") quoted above -- not as the bare topic substring
# "cross-check against the gate", which also matched the honest
# retraction prose this tree now ships: a false alarm priced like a
# false green (doctrine 5). The full joined directive names the
# nonexistent RECORD; the record, not the cross-check, was the defect,
# and the joined spelling is specific enough to pin (option (a)).
HISTORICAL_PHRASES = (
    "the gate records the adjudicating torch regardless",
    "cross-check against the gate's adjudicating-torch record",
    "adjudicating-torch record",
)

# The pre-fix emitter shipped the false directive split across adjacent
# string literals. This sample reconstructs that exact as-shipped SHAPE:
# the literal split points are verbatim from the split quoted in this
# module's docstring; the connective prose around them is representative
# and claims nothing more (doctrine 5). It exists as the pin detector's
# MUST_FIRE sample: the join+pin pipeline must always go red on it.
_HISTORICAL_WORDING_AS_SHIPPED = (
    '"the gate records the adjudicating torch regardless of exit path, "\n'
    '    "so record this build for cross-check against "\n'
    '    "the gate\'s adjudicating-torch record; a mismatch would "\n'
    '    "indict the measurement, never the model"'
)

# Join implicit string-literal concatenation (incl. r/b/f/u-prefixed
# literals) so a source pin sees what the interpreter ships. Without
# this, any phrase long enough to exclude honest prose but split across
# literals -- the original #83 evasion -- reads green while shipping red.
# Kept as an uncompiled string: this module's import block is outside
# the listed lines, so no new top-level import is added; re is imported
# inside the helper where it is used.
_ADJACENT_STRING_LITERALS = r"([\"'])\n\s*[rbfuRBFU]{0,2}\1"


def _join_adjacent_string_literals(src):
    """See source as the interpreter ships it: concatenation made visible.

    The substitution removes only quote characters, newlines, whitespace,
    and literal prefixes -- never phrase characters -- so any phrase raw
    scanning would catch is still caught. Identical quote characters
    abutting a seam that is not truly adjacent literals may over-join;
    that can at worst add an across-seam spelling (a false alarm, priced
    under doctrine 5), never a false green (fail closed, doctrine 4).
    """
    import re

    return re.sub(_ADJACENT_STRING_LITERALS, "", src)


def _load_emitter():
    spec = importlib.util.spec_from_file_location(
        "emit_run_manifest_under_control", EMITTER_PATH
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {EMITTER_PATH}")  # FAIL CLOSED
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.modules.pop(spec.name, None)
    return mod


def _load_gate():
    # exec, not subprocess: the gate's helpers then see THIS interpreter, so
    # the value legs below cross-check the shipped record against an
    # independent measurement of the same python (sys.executable, find_spec)
    # taken in this very process. A load failure is a test failure (FAIL
    # CLOSED), never a skip.
    spec = importlib.util.spec_from_file_location(
        "live_save_gate_under_control", GATE_PATH
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {GATE_PATH}")  # FAIL CLOSED
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.modules.pop(spec.name, None)
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
        f"{len(bad)} never written by "
        f"src/foundationscale/gates/adjudication.py: {bad} "
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


def test_regression_pin_83_torch_record_is_written_and_truthful(
    tmp_path, monkeypatch
):
    """INVERTED PIN. This function was the absence pin ("torch_record" not
    in gate_src); its own comment ordered that it go red in the change that
    implements gate-side provenance and be updated in that same change. This
    is that change: gone red, now inverted into a presence-AND-VALUE
    assertion (units: 1 readability guard + 3 field spellings over the gate
    source, 5 measured-value comparisons on the provenance record, 5
    must-pass checks on the adjudication entry and env channel, and 3
    assertions over a really-written UNMEASURED refusal record).
    """
    monkeypatch.delenv("LIVE_SAVE_GATE_EXPECT_INTERPRETER", raising=False)
    gate_src = GATE_PATH.read_text(encoding="utf-8")
    assert len(gate_src) > 1000, "unreadable gate source would void this pin"
    # PRESENCE, under the manifest's own spelling (one gate field per
    # emitted training-stack entry): the fields the manifest directs
    # operators to cross-check must exist as fields the gate really writes.
    for field in ("python_executable", "python_version", "torch_record"):
        assert f'"{field}"' in gate_src, (
            f"gate does not write {field!r}: the manifest directive is "
            f"unattributable again (#83 shape)"
        )
    # VALUE: a present string is not a measurement. The shipped helper's
    # output is compared, value by value, against an INDEPENDENT reading of
    # this very interpreter taken here -- a hardcoded record (the decoy
    # repair) turns this leg red.
    gate = _load_gate()
    prov = gate._interpreter_provenance()
    detected_here = importlib.util.find_spec("torch") is not None
    assert prov["python_executable"] == sys.executable
    assert prov["python_version"] == sys.version.split()[0]
    assert prov["torch_record"]["detected"] is detected_here, (
        "gate-reported torch presence disagrees with find_spec taken in "
        "this same interpreter -- the record is authored, not measured"
    )
    if detected_here:
        assert prov["torch_record"]["dist_version"] == (
            importlib.metadata.version("torch")
        )
    else:
        assert prov["torch_record"]["origin"] is None
    # MUST_PASS of the adjudication half: the truthful expectation passes
    # both the bare referee and the full report-entry path, and the entry
    # states the met expectation.
    expected_here = "container" if detected_here else "host"
    gate._refuse_on_interpreter_mismatch(prov, expected_here)
    entry = gate._interpreter_report_entry(expected_here)
    assert entry["expected_interpreter"] == expected_here
    assert entry["expectation_met"] is True
    # NO expectation: a stated abstention, never a counted pass (doctrine 1)
    # -- the entry pairs the provenance with expectation_met: null plus a
    # note, instead of manufacturing a comparison that never ran.
    uncontested = gate._interpreter_report_entry(None)
    assert uncontested["expected_interpreter"] is None
    assert uncontested["expectation_met"] is None
    assert "no expectation" in uncontested["expectation_note"]
    # The UNMEASURED refusal record (audit finding 2) carries the same
    # measured values -- exercised for real, not merely grepped, and kept
    # env-free here so the transcription legs below cannot contaminate it.
    refusal_path = tmp_path / "refusal.json"
    args = argparse.Namespace(
        ckpt_dir="ckpt_qz",
        event="save",
        run_kind="auto",
        json_out=str(refusal_path),
    )
    gate._record_refusal(args, "planted refusal for the #83 control qz")
    refusal = json.loads(refusal_path.read_text(encoding="utf-8"))
    assert refusal["verdict"] == "UNMEASURED"
    assert refusal["interpreter"]["python_executable"] == sys.executable
    assert refusal["interpreter"]["torch_record"]["detected"] is (
        detected_here
    )
    # The env channel: fallback resolves when the kwarg abstains, and the
    # kwarg wins when both speak. The resolver only validates vocabulary and
    # precedence; adjudication itself is the referee's job.
    monkeypatch.setenv("LIVE_SAVE_GATE_EXPECT_INTERPRETER", expected_here)
    assert gate._resolve_expected_interpreter(None) == expected_here
    other = "host" if expected_here == "container" else "container"
    assert gate._resolve_expected_interpreter(other) == other


def test_must_fire_83_refusal_observed_on_absent_lying_or_malformed(monkeypatch):
    """MUST_FIRE for the recorded-vs-expected half (doctrine 3): nine
    refusal legs, each a pytest.raises gate -- if any refusal path goes
    quiet, that leg fails HERE, so the comparator can never rot to vacuity
    green. Every field the referee compares against a fresh probe has its
    own planted lie below. First execution of this control is this
    change's CI run."""
    monkeypatch.delenv("LIVE_SAVE_GATE_EXPECT_INTERPRETER", raising=False)
    gate = _load_gate()
    prov = gate._interpreter_provenance()
    detected = prov["torch_record"]["detected"]
    truthful = "container" if detected else "host"
    untruthful = "host" if detected else "container"
    lying_detected = {
        **prov,
        "torch_record": {**prov["torch_record"], "detected": not detected},
    }
    lying_executable = {**prov, "python_executable": "/nonexistent/python"}
    # (1)-(2) record ABSENT refuses WHETHER OR NOT an expectation is on the
    # table -- an unattributable verdict is not made attributable by the
    # caller's silence (unreadable is not empty).
    with pytest.raises(gate.GateUnmeasured):
        gate._refuse_on_interpreter_mismatch(None, truthful)
    with pytest.raises(gate.GateUnmeasured):
        gate._refuse_on_interpreter_mismatch(None, None)
    # (3)-(4) record PRESENT AND LYING -- caught against a fresh probe of
    # this interpreter even with NO expectation to contradict, one planted
    # lie per compared field: the load-bearing torch bit, then the
    # executable path.
    with pytest.raises(gate.GateUnmeasured, match="torch_record"):
        gate._refuse_on_interpreter_mismatch(lying_detected, None)
    with pytest.raises(gate.GateUnmeasured, match="python_executable"):
        gate._refuse_on_interpreter_mismatch(lying_executable, None)
    # (5) honest record against the swapped expectation -- the two-pythons
    # mixup #83 exists to catch; refuses BEFORE any gate runs.
    with pytest.raises(gate.GateUnmeasured, match="interpreter mismatch"):
        gate._refuse_on_interpreter_mismatch(prov, untruthful)
    # (6) the full report-entry path refuses the swapped expectation too.
    with pytest.raises(gate.GateUnmeasured):
        gate._interpreter_report_entry(untruthful)
    # (7)-(9) the expectation vocabulary fails closed on every channel: the
    # resolver kwarg, a direct referee call that bypassed the resolver, and
    # the env var.
    with pytest.raises(gate.GateUnmeasured):
        gate._resolve_expected_interpreter("wsl2-something")
    with pytest.raises(gate.GateUnmeasured):
        gate._refuse_on_interpreter_mismatch(prov, "wsl2-something")
    monkeypatch.setenv("LIVE_SAVE_GATE_EXPECT_INTERPRETER", "wsl2-something")
    with pytest.raises(gate.GateUnmeasured):
        gate._resolve_expected_interpreter(None)


def test_regression_pins_historical_wording_absent_from_emitter_source():
    # Units examined: 3 historical phrases over the full emitter source
    # with adjacent string literals joined (both prose branches, executed
    # or not). Joined, because the pre-fix tree shipped phrase 2 split
    # across two literals: a raw-source-only pin of the full shipped
    # wording reads green on that shape while the artifact ships false --
    # the exact #83 evasion this control exists against. Joining only
    # ever ADDS the across-literal spellings an interpreter would see.
    emit_src = _join_adjacent_string_literals(
        EMITTER_PATH.read_text(encoding="utf-8")
    )
    for phrase in HISTORICAL_PHRASES:
        assert phrase not in emit_src, (
            f"historical #83 wording returned to the emitter: {phrase!r}"
        )
    # MUST_FIRE for this detector itself (doctrine 3): its only red to
    # date was a false alarm on honest prose, so it had never been
    # observed firing and was not yet a control. The historical wording
    # in its as-shipped split-literal shape must go red through the same
    # join+pin pipeline on every run; this assertion is option (a)'s
    # required confirmation that the pin still fails on the historical
    # wording -- executable, not prosaic. A pin rotted to vacuity fails
    # HERE, never silently.
    planted = _join_adjacent_string_literals(_HISTORICAL_WORDING_AS_SHIPPED)
    fired = [phrase for phrase in HISTORICAL_PHRASES if phrase in planted]
    assert fired == list(HISTORICAL_PHRASES), (
        f"the historical-wording pin no longer fires on the #83 wording "
        f"it exists to exclude: caught {fired!r}, expected all "
        f"{list(HISTORICAL_PHRASES)!r} -- the detector, not the wording, "
        f"has moved"
    )
