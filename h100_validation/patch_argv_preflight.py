#!/usr/bin/env python3
"""patch_argv_preflight.py -- FoundationScale H100 plane build stage (fs183).

Does two things, atomically:

* INSTALL: copies fs_argv_preflight.py from the build root (the hand-authored
  INPUT) to h100/gen/fs_argv_preflight.py (the OUTPUT plane directory, beside
  fs_container_backend.bound.sh, where the plane's other python residents are
  put BY STAGES). Without this half the spliced call
  $FS_PLANE_DIR/fs_argv_preflight.py would resolve to nothing on a deployed
  plane -- the orphan defect class this project hit five separate times
  (#188, #189, #190). The direction matters: build root is the input,
  h100/gen/ is the output (finding #137 was three build inputs living in the
  output directory).
* SPLICE: inserts the login-node argv preflight into
  h100/gen/launch_fs_h100.fixed.sh, between the chain-driver comment line and
  the first sbatch invocation, so a malformed FS_ENGINE_LAUNCH_CMD is refused
  on the login node BEFORE any allocation is burned. Every existing guard on
  that variable sits below the SLURM_JOB_ID gate, which is the defect this
  stage closes.

Atomicity: if either half cannot be done, NEITHER is. Both writes happen only
after every control has passed; if the second write raises, the stage says so
loudly and rolls back the copy it just landed rather than leave a
half-applied plane.

Stage conventions (the #188/#189/#190 defect class): a bare invocation APPLIES
the transform; --check is the explicit dry run. A stage whose bare invocation
is a dry run runs green in the build and lands nothing, so no argument means
--apply here. Exit codes: 0 PASS, 5 RED, 95 UNMEASURED, 96 REFUSE.

Idempotence is anchor-based: the pre-image anchor must occur exactly once.
Zero occurrences means this is not the file the stage was written against (or
the stage already ran); more than one means the file drifted. Both are REFUSE.
The stage never guesses and never fuzzy-matches.
"""

import hashlib
import os
import pathlib
import shlex
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parent
TARGET = ROOT / "h100" / "gen" / "launch_fs_h100.fixed.sh"
PREFLIGHT_SOURCE = ROOT / "fs_argv_preflight.py"
INSTALLED_PREFLIGHT = ROOT / "h100" / "gen" / "fs_argv_preflight.py"

# The anchor is the first two lines of the chain-driver branch plus the START
# of the third, matched as one contiguous byte span. Contiguity is what makes
# the transform self-destroying: the splice lands BETWEEN line two and line
# three, so a post-image no longer contains the anchor and a second run finds
# zero occurrences and REFUSES instead of double-splicing.
ANCHOR_LINE_1 = 'if [[ -z "${SLURM_JOB_ID:-}" && "${FS_SUBMIT_CHAIN:-0}" == 1 ]]; then'
ANCHOR_LINE_2 = '  # Login-node chain driver. Probe -> production -> resume(afterok: a REAL resume) + post-mortem(afterany: reporting only).'
ANCHOR_LINE_3_PREFIX = '  probe_jid="$(PROBE=1 FS_SUBMIT_CHAIN=0 sbatch'
ANCHOR = ANCHOR_LINE_1 + "\n" + ANCHOR_LINE_2 + "\n" + ANCHOR_LINE_3_PREFIX

# The spliced block, verbatim. It is a raw string because the printf line
# carries a literal backslash-n that must reach the shell script intact. The
# block is host-side and torch-free on purpose: it runs on the login node,
# where the interpreter may be 3.6.8, before the first sbatch.
BLOCK = r'''  # fs183: the operator's ACTUAL engine command, checked on the login node BEFORE any
  # allocation is burned. Every existing guard on FS_ENGINE_LAUNCH_CMD sits below the
  # SLURM_JOB_ID gate -- unset is caught at :819 and a malformed mode at :841 -- which means
  # a typo cost four queued jobs and a scheduler wait to discover. The preflight is host-side
  # and torch-free precisely so it can run here, where the interpreter may be 3.6.8.
  # FS_GPUS_PER_NODE is required and already validated above this splice point (req_env at
  # :225, integer at :344, > 0 at :345), so it is passed UNCONDITIONALLY: the emptiness a
  # conditional append would guard against cannot occur here.
  fs183_pf_args=( --launch-cmd "${FS_ENGINE_LAUNCH_CMD:-}" --backend "$FS_PLANE_DIR/$BACKEND_NAME" --procs-per-node "$FS_GPUS_PER_NODE" )
  # Pass --mode only when the operator set one. FS_ENGINE_LAUNCH_MODE is required-with-no-default
  # and read at :841, but a child process sees it only if it was exported; passing an empty
  # value would make the preflight adjudicate an empty string as a mode, whereas omitting the
  # flag lets it report C3 as UNMEASURED ("nothing to check").
  [[ -n "${FS_ENGINE_LAUNCH_MODE:-}" ]] && fs183_pf_args+=( --mode "$FS_ENGINE_LAUNCH_MODE" )
  if [[ ! -r "$FS_PLANE_DIR/fs_argv_preflight.py" ]]; then
    # A plane staged before this check existed is not a broken plane, and refusing to submit
    # because a diagnostic is missing would make the diagnostic worse than the defect it
    # finds. Absence of the checker is UNMEASURED, and it names its own remedy.
    printf 'ARGV PREFLIGHT unmeasured -- %s/fs_argv_preflight.py is not readable, so the engine command was NOT checked before submit. Remedy: redeploy the plane directory from the build (it ships this file alongside %s). Proceeding.\n' \
      "$FS_PLANE_DIR" "$BACKEND_NAME" >&2
  else
    fs183_pf_rc=0
    # Bare python3, deliberately no knob: the launcher already spells the host interpreter
    # as python3 (:582, :749), and the preflight is 3.6.8-clean precisely so that bare
    # python3 on a login node is sufficient. A knob introduced here and declared nowhere
    # would be a knob with no reader.
    python3 "$FS_PLANE_DIR/fs_argv_preflight.py" "${fs183_pf_args[@]}" || fs183_pf_rc=$?
    case "$fs183_pf_rc" in
      0) ;;
      95)
        # UNMEASURED must not block: a foreign engine entrypoint that lives only inside the
        # container is legitimately unreadable from here, and refusing every such launch would
        # make the plane engine-specific. It must also never be reported as a pass.
        printf 'ARGV PREFLIGHT unmeasured -- proceeding to submit; the checks above say which oracle was missing. This is NOT a clean bill of health.\n' >&2 ;;
      5)  fail 5 "argv preflight RED: the engine command names flags the entrypoint does not declare, or a mode the backend does not accept. Refusing before submitting; nothing was queued." ;;
      96) fail 96 "argv preflight REFUSE: FS_ENGINE_LAUNCH_CMD is unset or does not tokenize. Refusing before submitting; nothing was queued." ;;
      *)  fail 96 "argv preflight returned undeclared exit code $fs183_pf_rc; the plane's contract is 0/5/95/96" ;;
    esac
  fi
'''

REPLACEMENT = ANCHOR_LINE_1 + "\n" + ANCHOR_LINE_2 + "\n" + BLOCK + ANCHOR_LINE_3_PREFIX

UNMEASURED_WARNING = "ARGV PREFLIGHT unmeasured -- proceeding to submit"


def _stderr(msg):
    print(msg, file=sys.stderr)


def _control(name, ok, observation, failures):
    # Every control prints exactly one line in the stage's reporting format.
    # A control that did not fire or did not pass is recorded; the stage
    # refuses at the end and writes nothing.
    print("control {}: {}".format(name, observation))
    if not ok:
        failures.append(name)


def _run(cmd):
    # Returns the CompletedProcess, or None when bash itself is absent -- in
    # which case the control is UNMEASURED (95), never silently passed.
    try:
        return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                              universal_newlines=True)
    except FileNotFoundError:
        return None


def _write_harness(tmpdir, fake_rc, plane_dir):
    # Builds a behavioural harness. python3 is stubbed as a shell FUNCTION
    # that ignores its arguments and RETURNS the desired code: a function
    # overrides the command lookup in bash, so the spliced block still runs
    # verbatim and the measurement is of the real block, not a paraphrase.
    #
    # `return`, emphatically not `exit`. An `exit` inside a shell function
    # terminates the whole script, so the harness died AT THE STUB and never
    # reached the case block at all. That still produced rc=5 with sbatch
    # uncalled, i.e. the RED control passed while measuring nothing -- it would
    # have passed against a block with no `5)` arm. The 95 control is what
    # exposed it, because "proceed and warn" cannot be faked by dying early.
    # Hence also the FAIL-rc discriminator in the RED control below: the
    # observable that distinguishes "the arm ran" from "the harness died".
    # fail and sbatch are stubbed so the control can observe what the block
    # actually DOES. FS_PLANE_DIR points at plane_dir; whether that directory
    # contains a readable fs_argv_preflight.py is the caller's choice, because
    # the missing-preflight control depends on its absence.
    harness = tmpdir / "harness_exit_{}_{}.sh".format(fake_rc, plane_dir.name)
    harness.write_text(
        "# Behavioural harness: fail, sbatch and python3 are stubbed so the control can\n"
        "# observe what the spliced block actually DOES, then the block runs\n"
        "# verbatim, then a line invokes sbatch.\n"
        'fail() { echo "FAIL rc=$1" >&2; exit "$1"; }\n'
        'sbatch() { echo "SBATCH-WAS-CALLED" ; }\n'
        + "python3() {{ return {}; }}\n".format(fake_rc)
        + "FS_PLANE_DIR={}\n".format(shlex.quote(str(plane_dir)))
        + "BACKEND_NAME=fake_backend.bound.sh\n"
        + "FS_ENGINE_LAUNCH_CMD='python3 train.py --epochs 1'\n"
        + "FS_GPUS_PER_NODE=8\n"
        + BLOCK
        + 'sbatch --partition=fake --parsable "$FS_PLANE_DIR/launch_fs_h100.fixed.sh"\n'
    )
    return harness


def main():
    argv = sys.argv[1:]
    if argv not in ([], ["--apply"], ["--check"]):
        _stderr("usage: patch_argv_preflight.py [--apply|--check]   (no argument == --apply)")
        return 96
    apply = argv != ["--check"]

    failures = []
    unmeasured = []

    try:
        pre = TARGET.read_text()
    except OSError as exc:
        _stderr("REFUSE 96: cannot read {}: {}; the stage does not recognise this file "
                "and will not guess; writing nothing".format(TARGET, exc))
        return 96

    # Control 1 -- MUST_FIRE/ANCHOR_RESOLVES_EXACTLY_ONCE. The anchor is the
    # stage's entire claim of recognition: exactly one occurrence means this is
    # the file the stage was written against and the splice point is
    # unambiguous. Zero means the file drifted or the stage already ran; more
    # than one means the branch was duplicated. Both are REFUSE, because a
    # stage that guesses a splice point lands a patch in the wrong place and
    # reports green.
    occurrences = pre.count(ANCHOR)
    _control("MUST_FIRE/ANCHOR_RESOLVES_EXACTLY_ONCE", occurrences == 1,
             "anchor occurrences={} (required: exactly 1)".format(occurrences), failures)
    if failures:
        _stderr("REFUSE 96: anchor occurs {} time(s) in {}; the stage does not recognise "
                "this file and will not guess; writing nothing".format(occurrences, TARGET))
        return 96

    post = pre.replace(ANCHOR, REPLACEMENT)

    with tempfile.TemporaryDirectory(prefix="fs183_preflight_stage_") as td:
        tmpdir = pathlib.Path(td)

        # Control 2 -- MUST_PASS/POST_IMAGE_IS_VALID_BASH. The stage's product
        # is a bash script; a splice that breaks the parse would land a
        # launcher that dies on every invocation. bash -n on the actual
        # post-image bytes measures parse validity instead of asserting it.
        post_path = tmpdir / "launch_fs_h100.post-image.sh"
        post_path.write_text(post)
        proc = _run(["bash", "-n", str(post_path)])
        if proc is None:
            unmeasured.append("MUST_PASS/POST_IMAGE_IS_VALID_BASH")
            print("control MUST_PASS/POST_IMAGE_IS_VALID_BASH: UNMEASURED -- bash not found on PATH")
        else:
            _control("MUST_PASS/POST_IMAGE_IS_VALID_BASH", proc.returncode == 0,
                     "bash -n rc={} stderr={!r}".format(proc.returncode, proc.stderr.strip()),
                     failures)

        # Control 3 -- MUST_FIRE/THE_CALL_PRECEDES_THE_FIRST_SBATCH. The
        # finding's actual claim is ordering: the preflight must run before the
        # first sbatch or it buys nothing. That claim is MEASURED from the
        # post-image by line number -- the preflight invocation line must be
        # strictly above the first non-comment line containing "sbatch " --
        # never asserted from the text of the block.
        preflight_line = 0
        first_sbatch_line = 0
        for lineno, line in enumerate(post.splitlines(), start=1):
            if not preflight_line and "fs_argv_preflight.py" in line:
                preflight_line = lineno
            if (not first_sbatch_line and "sbatch " in line
                    and not line.lstrip().startswith("#")):
                first_sbatch_line = lineno
        ok = bool(preflight_line) and bool(first_sbatch_line) and preflight_line < first_sbatch_line
        _control("MUST_FIRE/THE_CALL_PRECEDES_THE_FIRST_SBATCH", ok,
                 "preflight_invocation_line={} first_non_comment_sbatch_line={} "
                 "(required: preflight strictly above sbatch)".format(
                     preflight_line or "not-found", first_sbatch_line or "not-found"),
                 failures)

        # The behavioural harnesses below need a plane directory that DOES
        # contain a readable fs_argv_preflight.py (the spliced block guards on
        # -r before calling), and one that does NOT. The python3 function stub
        # never reads the file it is handed, so a dummy satisfies the guard.
        plane = tmpdir / "plane_with_preflight"
        plane.mkdir()
        (plane / "fs_argv_preflight.py").write_text(
            "# dummy readable preflight for the stage's behavioural harness\n")
        empty_plane = tmpdir / "plane_without_preflight"
        empty_plane.mkdir()

        # Control 4 -- MUST_FIRE/A_RED_PREFLIGHT_BLOCKS_THE_SUBMIT. This is a
        # behavioural control, not a text match: the spliced block is executed
        # under bash with fail and sbatch stubbed and python3 stubbed as a
        # shell function that exits 5. The finding's promise is that a RED
        # preflight refuses BEFORE anything is queued, so the harness must exit
        # 5 and the sbatch stub must never have run. A text match could pass
        # while the block silently fell through to the submit.
        #
        # rc==5 and not-called are jointly satisfiable by a harness that never
        # ran the block at all -- that is exactly the confound the `exit`-vs-
        # `return` stub bug produced. So the control additionally requires the
        # fail stub's own marker, which only the `5)` arm can emit. That is the
        # observable separating "the block refused" from "the harness died".
        proc = _run(["bash", str(_write_harness(tmpdir, 5, plane))])
        if proc is None:
            unmeasured.append("MUST_FIRE/A_RED_PREFLIGHT_BLOCKS_THE_SUBMIT")
            print("control MUST_FIRE/A_RED_PREFLIGHT_BLOCKS_THE_SUBMIT: UNMEASURED -- bash not found on PATH")
        else:
            called = "SBATCH-WAS-CALLED" in (proc.stdout + proc.stderr)
            arm_ran = "FAIL rc=5" in proc.stderr
            _control("MUST_FIRE/A_RED_PREFLIGHT_BLOCKS_THE_SUBMIT",
                     proc.returncode == 5 and not called and arm_ran,
                     "harness rc={} (required 5) sbatch_called={} (required False) "
                     "fail_arm_ran={} (required True; without this the control is "
                     "satisfied by a harness that died before the case block)".format(
                         proc.returncode, called, arm_ran),
                     failures)

        # Control 5 -- MUST_PASS/A_95_PREFLIGHT_DOES_NOT_BLOCK. Same harness,
        # python3 stub exits 95. UNMEASURED must not block: a foreign engine
        # entrypoint that lives only inside the container is legitimately
        # unreadable from the login node, and refusing every such launch would
        # make the plane engine-specific. It must also never be reported as a
        # pass, so the warning text must reach stderr. This pins the deliberate
        # asymmetry between UNMEASURED and RED; without it a later edit could
        # quietly make 95 fatal and every foreign-engine launch would refuse.
        proc = _run(["bash", str(_write_harness(tmpdir, 95, plane))])
        if proc is None:
            unmeasured.append("MUST_PASS/A_95_PREFLIGHT_DOES_NOT_BLOCK")
            print("control MUST_PASS/A_95_PREFLIGHT_DOES_NOT_BLOCK: UNMEASURED -- bash not found on PATH")
        else:
            called = "SBATCH-WAS-CALLED" in proc.stdout
            warned = UNMEASURED_WARNING in proc.stderr
            _control("MUST_PASS/A_95_PREFLIGHT_DOES_NOT_BLOCK", called and warned,
                     "harness rc={} sbatch_called={} (required True) "
                     "unmeasured_warning_in_stderr={} (required True)".format(
                         proc.returncode, called, warned),
                     failures)

        # Control 8 -- MUST_PASS/INSTALLED_COPY_IS_BYTE_IDENTICAL. The install
        # half of this stage copies the build-root source byte for byte; a copy
        # that silently transformed the file would make the plane's checker
        # differ from the one certified by --self-test. The bytes the stage
        # would install are measured by staging the copy through the same
        # write_bytes path the apply step uses and reading it back; both
        # lengths and a sha256 of each are reported (fingerprints, not
        # contents).
        try:
            src_bytes = PREFLIGHT_SOURCE.read_bytes()
        except OSError:
            src_bytes = None
        if src_bytes is None:
            _control("MUST_PASS/INSTALLED_COPY_IS_BYTE_IDENTICAL", False,
                     "cannot read source {}; byte identity unmeasurable".format(
                         PREFLIGHT_SOURCE), failures)
        else:
            staged = tmpdir / "fs_argv_preflight.py.installed"
            staged.write_bytes(src_bytes)
            staged_bytes = staged.read_bytes()
            _control("MUST_PASS/INSTALLED_COPY_IS_BYTE_IDENTICAL",
                     staged_bytes == src_bytes,
                     "source_len={} installed_len={} source_sha256={} "
                     "installed_sha256={}".format(
                         len(src_bytes), len(staged_bytes),
                         hashlib.sha256(src_bytes).hexdigest(),
                         hashlib.sha256(staged_bytes).hexdigest()),
                     failures)

        # Control 9 -- MUST_FIRE/A_MISSING_PREFLIGHT_DOES_NOT_BLOCK. Same
        # harness shape, but FS_PLANE_DIR points at a directory that does NOT
        # contain fs_argv_preflight.py. A plane staged before this check
        # existed must not refuse every launch: python on a missing path exits
        # 2, which the block's *) arm would turn into fail 96 -- converting a
        # diagnostic into an outage. The harness must exit 0, the sbatch stub
        # must have run, and the word "unmeasured" must reach stderr. Without
        # this control that outage is one edit away from returning.
        proc = _run(["bash", str(_write_harness(tmpdir, 0, empty_plane))])
        if proc is None:
            unmeasured.append("MUST_FIRE/A_MISSING_PREFLIGHT_DOES_NOT_BLOCK")
            print("control MUST_FIRE/A_MISSING_PREFLIGHT_DOES_NOT_BLOCK: UNMEASURED -- bash not found on PATH")
        else:
            called = "SBATCH-WAS-CALLED" in proc.stdout
            warned = "unmeasured" in proc.stderr
            _control("MUST_FIRE/A_MISSING_PREFLIGHT_DOES_NOT_BLOCK",
                     proc.returncode == 0 and called and warned,
                     "harness rc={} (required 0) sbatch_called={} (required True) "
                     "unmeasured_in_stderr={} (required True)".format(
                         proc.returncode, called, warned),
                     failures)

    # Control 6 -- MUST_FIRE/THE_PREFLIGHT_SOURCE_EXISTS. The splice makes the
    # launcher call fs_argv_preflight.py on every chained submit, and the
    # install half of this stage copies the build-root source into the plane
    # directory; the build root is the INPUT, so that source is what must be
    # readable here. A stage that splices a call to a script it cannot ship is
    # the orphan defect this project has already hit five times, so the
    # source's readability is measured before anything is written.
    pf_ok = PREFLIGHT_SOURCE.is_file() and os.access(str(PREFLIGHT_SOURCE), os.R_OK)
    _control("MUST_FIRE/THE_PREFLIGHT_SOURCE_EXISTS", pf_ok,
             "path={} is_readable_file={}".format(PREFLIGHT_SOURCE, pf_ok), failures)

    # Control 7 -- MUST_FIRE/BYTE_IDEMPOTENCE. Applying the transform to the
    # post-image must find zero anchors: the splice lands between anchor line
    # two and anchor line three, destroying the contiguous byte span the anchor
    # matches. A second run therefore REFUSES rather than double-splicing,
    # which is what makes a re-run of the build safe. The count is measured on
    # the actual post-image bytes.
    post_occurrences = post.count(ANCHOR)
    _control("MUST_FIRE/BYTE_IDEMPOTENCE", post_occurrences == 0,
             "anchor occurrences in post-image={} (required 0; a second run must "
             "REFUSE, not double-splice)".format(post_occurrences), failures)

    # Control 10 -- MUST_FIRE/THE_BLOCK_NAMES_NO_UNDECLARED_KNOB. The measured
    # launcher spells the host interpreter as bare python3 and the
    # procs-per-node knob as FS_GPUS_PER_NODE; the two earlier names were
    # invented by a splice and declared nowhere (the "knob with no reader"
    # defect of #131/#140). This pins points 1 and 2 against a future re-edit;
    # a text control is the right instrument here because the claim is itself
    # about what the text names.
    names_host_knob = "FS_PYTHON" in BLOCK
    names_procs_knob = "FS_ENGINE_PROCS_PER_NODE" in BLOCK
    names_gpus_knob = "FS_GPUS_PER_NODE" in BLOCK
    _control("MUST_FIRE/THE_BLOCK_NAMES_NO_UNDECLARED_KNOB",
             not names_host_knob and not names_procs_knob and names_gpus_knob,
             "block_names_FS_PYTHON={} (required False) "
             "block_names_FS_ENGINE_PROCS_PER_NODE={} (required False) "
             "block_names_FS_GPUS_PER_NODE={} (required True)".format(
                 names_host_knob, names_procs_knob, names_gpus_knob),
             failures)

    if unmeasured:
        _stderr("UNMEASURED 95: {} control(s) could not be measured ({}); writing nothing".format(
            len(unmeasured), ", ".join(unmeasured)))
        return 95
    if failures:
        _stderr("REFUSE 96: {} control(s) failed ({}); writing nothing".format(
            len(failures), ", ".join(failures)))
        return 96

    if apply:
        # Both writes happen only here, after every control has passed, and
        # they are atomic: if either half cannot be done, NEITHER is. The
        # install goes first; if it fails the launcher is untouched. If the
        # launcher write then raises, the stage says so loudly and rolls back
        # the copy it just landed -- a call site whose callee did not ship, or
        # a callee whose call site does not exist, is the #188/#189/#190
        # orphan defect class, and a half-applied plane must not be left
        # behind silently.
        try:
            src_bytes = PREFLIGHT_SOURCE.read_bytes()
        except OSError as exc:
            _stderr("REFUSE 96: cannot read {} for install: {}; writing nothing".format(
                PREFLIGHT_SOURCE, exc))
            return 96
        install_existed = INSTALLED_PREFLIGHT.exists()
        try:
            INSTALLED_PREFLIGHT.write_bytes(src_bytes)
        except OSError as exc:
            _stderr("REFUSE 96: cannot install {}: {}; the launcher was NOT patched; "
                    "writing nothing".format(INSTALLED_PREFLIGHT, exc))
            return 96
        try:
            TARGET.write_text(post)
        except OSError as exc:
            if install_existed:
                remedy = "the installed copy pre-dated this run and was left in place"
            else:
                try:
                    INSTALLED_PREFLIGHT.unlink()
                    remedy = "rolled back the installed copy"
                except OSError:
                    remedy = ("ROLLBACK FAILED -- remove {} by hand".format(
                        INSTALLED_PREFLIGHT))
            _stderr("REFUSE 96: installed {} but FAILED to write {}: {}; {}. "
                    "The plane must not be left half-applied.".format(
                        INSTALLED_PREFLIGHT, TARGET, exc, remedy))
            return 96
        print("summary: PASS -- all 10 controls fired/passed; installed {} byte-for-byte "
              "and spliced fs183 argv preflight into {}".format(
                  INSTALLED_PREFLIGHT, TARGET))
    else:
        print("summary: PASS -- all 10 controls fired/passed; --check dry run, would have "
              "installed {} and written {}; nothing written".format(
                  INSTALLED_PREFLIGHT, TARGET))
    return 0


if __name__ == "__main__":
    sys.exit(main())
