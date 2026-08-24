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
