"""Build-suite certification for fs_argv_preflight.py.

The subject carries 12 self-test controls, but those controls only matter if
they actually EXECUTE inside the build's suite. This module runs the self-test
entry point in process and independently pins the load-bearing properties
(denominator, zero failures, oracle agreement, derived mode set) rather than
trusting the subject's own reporting.

Paths are resolved relative to this file's directory with pathlib -- never the
current working directory, because pytest may be invoked from anywhere.
"""

import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import fs_argv_preflight as preflight  # noqa: E402

TRAINER = HERE / "h100" / "gen" / "fs_train.fixed.py"
BACKEND = HERE / "h100" / "gen" / "fs_container_backend.bound.sh"
GATE = HERE / "gate_launch_doc.py"

_SUMMARY_RE = re.compile(r"^SELF-TEST (\d+)/(\d+) ok \((\d+) unmeasured\)$")


def _summary_numbers(lines):
    """(n_ok, n_total, n_unmeasured) from the self-test summary line."""
    for line in lines:
        m = _SUMMARY_RE.match(line)
        if m:
            return int(m.group(1)), int(m.group(2)), int(m.group(3))
    raise AssertionError("no SELF-TEST summary line found in output: %r" % (lines,))


def test_self_test_passes(capsys):
    # The whole point of this module: the subject's controls must run inside
    # THIS suite. Invoke the entry point in process -- not via a second
    # interpreter, so a broken interpreter path cannot make this vacuously
    # green -- and require the PASS exit code.
    exit_code = preflight.self_test(None)
    out = capsys.readouterr().out
    assert exit_code == preflight.EXIT_PASS, (
        "self-test returned exit %d (required %d=PASS); output:\n%s"
        % (exit_code, preflight.EXIT_PASS, out))
    n_ok, n_total, n_unmeasured = _summary_numbers(out.splitlines())
    # The summary line must report zero failures, independently of the exit
    # code, so a broken exit-code mapping cannot hide a FAILED control.
    n_failed = n_total - n_ok - n_unmeasured
    assert n_failed == 0, (
        "summary reports %d failed control(s): %d/%d ok (%d unmeasured)"
        % (n_failed, n_ok, n_total, n_unmeasured))


def test_every_control_ran(capsys):
    # Denominator check: a self-test that silently stopped emitting controls
    # would otherwise still print an ok summary, so count the emitted
    # `control ` lines and require the count to equal the summary's claimed
    # total AND to be at least the 12 controls the subject documents.
    preflight.self_test(None)
    out = capsys.readouterr().out
    lines = out.splitlines()
    control_lines = [line for line in lines if line.startswith("control ")]
    _, n_total, _ = _summary_numbers(lines)
    assert len(control_lines) == n_total, (
        "summary claims %d control(s) but %d `control ` line(s) were emitted"
        % (n_total, len(control_lines)))
    assert n_total >= 12, (
        "only %d control(s) ran; the subject documents 12 mandatory controls"
        % n_total)


def test_no_control_reported_failed(capsys):
    # A FAILED string anywhere in the output is a red control, regardless of
    # what the aggregate exit code or summary arithmetic claims.
    preflight.self_test(None)
    out = capsys.readouterr().out
    failed = [line for line in out.splitlines() if "FAILED" in line]
    assert not failed, "control(s) reported FAILED: %r" % (failed,)


def test_declared_flags_matches_the_gate_oracle():
    # Two oracles that can disagree is the defect class this project keeps
    # finding, so the agreement between the subject's declared_flags and the
    # gate's trainer_flags is asserted here as well as inside the subject.
    # Absence of either artifact is asserted explicitly, naming the missing
    # path, rather than passing quietly.
    assert GATE.is_file(), "gate oracle module missing: %s" % GATE
    assert TRAINER.is_file(), "real trainer missing: %s" % TRAINER
    import gate_launch_doc
    src = TRAINER.read_text()
    ours = preflight.declared_flags(src)
    theirs = gate_launch_doc.trainer_flags(src)
    assert ours == theirs, (
        "flag oracles disagree on %s: declared_flags=%s, trainer_flags=%s"
        % (TRAINER, sorted(ours), sorted(theirs)))
    assert ours, "both oracles returned an empty flag set for %s" % TRAINER


def test_mode_set_is_derived_from_the_backend(tmp_path):
    # The backend's `case "$mode" in ...` arm is the only thing that actually
    # enforces the mode at launch, so a second hard-coded list here could
    # drift from it. Pin derivation by mutation: add an alternative to a COPY
    # of the backend and require the re-derived set to contain it -- a
    # hard-coded list would fail this.
    assert BACKEND.is_file(), "real backend missing: %s" % BACKEND
    src = BACKEND.read_text()
    modes, provenance = preflight.declared_modes(src)
    assert modes, "derived mode set is empty for %s (%s)" % (BACKEND, provenance)
    assert re.search(r"line \d+", provenance), (
        "provenance does not name a line number: %r" % (provenance,))

    extra = "pytestextra"
    assert extra not in modes, "test alternative %r already in %s" % (extra, modes)
    lines = src.splitlines(True)
    patched = None
    for i, line in enumerate(lines):
        m = preflight._MODE_CASE_RE.search(line)
        if m:
            lines[i] = line[:m.end(1)] + "|" + extra + line[m.end(1):]
            patched = "".join(lines)
            break
    assert patched is not None, (
        "could not locate the case arm in a copy of %s to patch" % BACKEND)
    copy = tmp_path / "fs_container_backend.with_extra.sh"
    copy.write_text(patched)

    modes2, provenance2 = preflight.declared_modes(copy.read_text())
    assert extra in modes2, (
        "added alternative %r not in re-derived set %s (%s); the mode set is "
        "not derived from the backend" % (extra, sorted(modes2), provenance2))
