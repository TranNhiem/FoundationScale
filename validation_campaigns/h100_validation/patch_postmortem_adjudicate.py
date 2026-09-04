#!/usr/bin/env python3
"""#187: the post-mortem afterany link prints one sentence and exits 0 -- recording PASS
over zero adjudications in exactly the case the link exists for, production DIED.

MEASURED BEFORE WRITING (on the generated launcher, h100/gen/launch_fs_h100.fixed.sh,
798 lines):

  :584  chain driver comment names the link `post-mortem(afterany: reporting only)`
  :597-:600  the entire post-mortem branch:
            if [[ "${FS_PHASE:-}" == post-mortem ]]; then
              printf '... adjudication link reached; reporting only, no training launched. ...'
              exit 0
            fi
        In this plane's four-state contract 0 means PASS, so a chain whose production
        job died leaving unread checkpoints ends with a post-mortem job sacct reports
        COMPLETED 0:0 -- indistinguishable from a clean sweep. Observed live: job
        37343, 8 lines of output, 3 seconds, ExitCode 0:0, ZERO adjudicator
        invocations. That is all([]) is True wearing a Slurm job costume: zero
        adjudications reported as success, over a denominator that is never printed.
        The branch's own printf calls itself the "adjudication link" and adjudicates
        nothing -- claim and evidence do not meet.
  :778-:786  the training failure path: nonzero rc -> fs_hard_stop_training, END
        FAILED, `exit "$rc"` -- so the adjudication tail at :788-:798 is NEVER reached
        on a failed run. The checkpoints a failed run left on disk are the ones
        nothing ever reads, and they are precisely the ones most worth reading.
  :538  run_adjudicators dispatches a .py spec through run_in_container (:555).
  :648-:649  fs_backend_init / fs_backend_runtime_setup -- the container runtime does
        not exist until HERE, 51 lines AFTER the post-mortem branch at :597.

WHY THE FIX HAS THIS SHAPE. The obvious one-line fix -- call
`adjudicate_tree "$OUT_DIR" post-mortem` inside the :597 branch -- is WRONG, and the
ordering above is the measurement that rules it out: run_adjudicators needs
run_in_container, run_in_container needs the runtime, and the runtime is not
initialised until 51 lines below the branch. Adjudicating at :597 would die on an
uninitialised backend. So the post-mortem link becomes the ordinary training path
minus the training: it falls through backend init, runtime setup, the bind plane, the
in-container GPU census and fs_compose_launch; an internal skip flag bypasses only the
one run_in_container call that launches training; and the EXISTING adjudication tail
at :788-:798 is reused verbatim. Reusing it is the whole point: same adjudicate_tree,
same checkpoint_observed denominator, same `[[ "$checkpoint_observed" -gt 0 ]] || fail 95`
guard, same `END rc=... phase=%s checkpoint_saves_adjudicated=%s` line, which already
interpolates ${FS_PHASE:-train} and so prints phase=post-mortem with no change. A
second copy of that tail would be a second thing to keep in sync, and this build has
already been bitten by exactly that (two oracles, the hard-coded one wins).

The skip flag, FS_SKIP_TRAIN, is deliberately launcher-INTERNAL: not exported, not on
any allowlist, not documented as an operator knob. An operator-settable "skip the
training" flag on a training launcher is a way to produce a green run that trained
nothing. Under `set -u` every read is ${FS_SKIP_TRAIN:-0}; the default-expansion is
the whole guard, so no separate initialiser line exists.

KNOWN BOUNDED CONDITION (recorded, deliberately not "fixed"): the post-mortem link's
afterany dependency is on the production job ALONE, so when production SUCCEEDS the
post-mortem job can run concurrently with the resume job, which writes new checkpoints
into the same OUT_DIR. Post-mortem therefore adjudicates the tree as it stands at that
moment; an adjudicator refusal on a half-written checkpoint is a loud 96, never a
silent pass, and that is the correct failure direction. The dependency must NOT be
widened to afterany:$prod:$resume: Slurm's kill-invalid-depend would then cancel the
post-mortem link itself when resume is cancelled after a failed production -- i.e. it
would delete the link exactly in the case the link exists for.

STAGE MECHANICS (shared with the sibling stages): fail closed on a missing or
unreadable launcher (96); each anchor must occur EXACTLY once or the stage refuses,
naming the anchor and the count observed -- zero occurrences is NOT "already applied",
it is a generator drift to refuse, because a stage that silently no-ops when its
anchor is gone is how a patch quietly stops being applied. Idempotence is measured,
not assumed: an already-patched file is detected by the presence of FS_SKIP_TRAIN and
reported as a no-op (0) WITHOUT rewriting, and that check runs before the anchor
counts. Post-conditions are verified on the produced text BEFORE it is written:
  1. `exit 0` no longer appears inside the post-mortem branch;
  2. FS_SKIP_TRAIN occurs exactly twice -- once assigned, once read;
  3. the `run_in_container ... bash -lc "$LAUNCH_CMD"` line still occurs exactly once;
  4. rc= is assigned on BOTH arms of the new guard (rc is read unguarded at :782, and
     under set -u an unset rc would abort);
  5. the file still parses as bash (`bash -n` on a temporary copy; if bash itself is
     unavailable that is UNMEASURED -- printed, exit 95, never skipped).

EXIT CODES: 0 applied (or already applied), 95 UNMEASURED, 96 REFUSE. Never any other
code -- and `raise SystemExit("text")` exits 1, so messages are printed and the number
is returned explicitly.
"""

from __future__ import annotations

import os
import pathlib
import re
import subprocess
import sys
import tempfile

# Resolved against THIS FILE, not the cwd: the stage must run from the repository root
# with no arguments, exactly like every other stage in this build.
LAUNCH = pathlib.Path(__file__).resolve().parent / "h100" / "gen" / "launch_fs_h100.fixed.sh"

# Idempotence marker: the flag this stage introduces. Its presence means the stage has
# already run; its absence means nothing about the anchors, which are counted next.
MARK = "FS_SKIP_TRAIN"

# --- Edit 1: the post-mortem branch (:597-:600) -----------------------------------------
ANCHOR_1 = (
    'if [[ "${FS_PHASE:-}" == post-mortem ]]; then\n'
    "  printf 'POST-MORTEM afterany adjudication link reached; reporting only, no training launched. OUT_DIR=%s\\n' \"$OUT_DIR\" | tee -a \"$RUN_LOG\"\n"
    "  exit 0\n"
    "fi\n"
)
# The replacement sets the skip flag and does NOT exit. The printf keeps announcing the
# link but drops "reporting only" -- after this patch reporting is no longer all it
# does. The in-launcher comment must not contain the flag's literal name: post-condition
# 2 counts exactly two occurrences of it in the whole file (one assignment, one read),
# and a comment naming it would be a third.
REPLACEMENT_1 = (
    'if [[ "${FS_PHASE:-}" == post-mortem ]]; then\n'
    "  # fs187: this link fires afterany precisely so it runs when production DIED -- the\n"
    "  # checkpoints a failed run left on disk are the ones most worth reading. The\n"
    "  # unconditional zero-exit that used to stand here recorded PASS over zero\n"
    "  # adjudications: all([]) in a Slurm job costume. (The literal is elided rather\n"
    "  # than written out because the stage that emits this branch post-condition-scans\n"
    "  # it, and a scanner that matches its own explanatory prose has no denominator.)\n"
    "  # The skip flag set below is deliberately launcher-internal -- NOT\n"
    "  # exported, NOT on any allowlist: an operator-settable \"skip the training\" flag\n"
    "  # on a training launcher is a way to produce a green run that trained nothing.\n"
    "  FS_SKIP_TRAIN=1\n"
    "  printf 'POST-MORTEM afterany adjudication link reached; no training launched, the checkpoint tree production left behind is adjudicated below. OUT_DIR=%s\\n' \"$OUT_DIR\" | tee -a \"$RUN_LOG\"\n"
    "fi\n"
)

# --- Edit 2: the training launch (:778-:781) --------------------------------------------
ANCHOR_2 = (
    "set +e\n"
    'run_in_container --workdir "$OUT_DIR" "${top_args[@]}" -- bash -lc "$LAUNCH_CMD" 2>&1 | tee -a "$RUN_LOG"\n'
    'rc="${PIPESTATUS[0]}"\n'
    "set -e\n"
)
# The four original lines are preserved byte-for-byte apart from two spaces of
# indentation: PIPESTATUS stays bash (not zsh), and the set +e / set -e pair stays
# INSIDE the else arm -- moving it outside would disarm set -e across the skip arm too.
# rc is assigned on both arms because it is read unguarded at :782 and set -u would
# abort on an unset rc.
REPLACEMENT_2 = (
    'if [[ "${FS_SKIP_TRAIN:-0}" == 1 ]]; then\n'
    "  # fs187: only the training launch is skipped. Backend init, runtime setup, the\n"
    "  # bind plane, the in-container GPU census and fs_compose_launch have all already\n"
    "  # run on the ordinary path above, and the shared adjudication tail below is what\n"
    "  # produces the verdict -- reused verbatim, not duplicated.\n"
    "  printf 'POST-MORTEM: no training launched; adjudicating the checkpoint tree production left behind. OUT_DIR=%s\\n' \"$OUT_DIR\" | tee -a \"$RUN_LOG\"\n"
    "  rc=0\n"
    "else\n"
    "  set +e\n"
    '  run_in_container --workdir "$OUT_DIR" "${top_args[@]}" -- bash -lc "$LAUNCH_CMD" 2>&1 | tee -a "$RUN_LOG"\n'
    '  rc="${PIPESTATUS[0]}"\n'
    "  set -e\n"
    "fi\n"
)

LAUNCH_LINE = 'run_in_container --workdir "$OUT_DIR" "${top_args[@]}" -- bash -lc "$LAUNCH_CMD"'


def _branch_region(text: str, start_marker: str) -> str:
    """The text of one if..fi block, from its opening line to the next line that is `fi`."""
    start = text.index(start_marker)
    end = text.index("\nfi\n", start) + len("\nfi\n")
    return text[start:end]


def main() -> int:
    # Fail closed: a missing or unreadable input is not a zero, it is an unread
    # measurement.
    if not LAUNCH.exists():
        print(f"REFUSE 96: launcher not found: {LAUNCH}", file=sys.stderr)
        return 96
    try:
        text = LAUNCH.read_text("utf-8")
    except OSError as exc:
        print(f"REFUSE 96: launcher unreadable: {LAUNCH}: {exc}", file=sys.stderr)
        return 96

    # Idempotence as a measurement, checked FIRST, before the anchor counts: an
    # already-patched file is a reported no-op, not a failure and not a rewrite.
    if MARK in text:
        print(f"fs187: already applied ({MARK} present in {LAUNCH.name}) -- no-op")
        return 0

    # Each anchor must occur EXACTLY once. Zero occurrences is NOT "already applied" --
    # the marker check above is the only idempotence test, so a missing anchor here
    # means the generator drifted, and a stage that silently no-ops when its anchor is
    # gone is how a patch quietly stops being applied.
    for name, anchor in (("edit 1 (post-mortem branch)", ANCHOR_1),
                         ("edit 2 (training launch)", ANCHOR_2)):
        n = text.count(anchor)
        if n != 1:
            print(f"REFUSE 96: anchor for {name} occurs {n}x (need exactly 1) in "
                  f"{LAUNCH.name}; the generator emitted a different shape than #187 "
                  f"measured -- refusing rather than patching blind", file=sys.stderr)
            return 96

    new = text.replace(ANCHOR_1, REPLACEMENT_1, 1).replace(ANCHOR_2, REPLACEMENT_2, 1)

    # Post-conditions, verified on the produced text BEFORE anything is written.
    # 1: the post-mortem branch no longer contains a zero-exit STATEMENT. The claim is
    # about a statement, so the detector is anchored to one -- a bare substring search
    # also matched this stage's own explanatory prose, which is the scanner standing
    # inside its own denominator. The prose was elided AND the detector was narrowed to
    # what it actually claims; narrowing alone would have been an exemption, and eliding
    # alone would have left the next comment free to break it again.
    if re.search(r"^\s*exit\s+0\b",
                 _branch_region(new, 'if [[ "${FS_PHASE:-}" == post-mortem ]]; then'),
                 re.M):
        print("REFUSE 96: post-condition 1 failed -- a zero-exit statement still stands "
              "inside the post-mortem branch; the link would still record PASS over zero "
              "adjudications", file=sys.stderr)
        return 96

    # 2: the flag occurs exactly twice -- once assigned, once read.
    n_mark = new.count(MARK)
    if n_mark != 2:
        print(f"REFUSE 96: post-condition 2 failed -- {MARK} occurs {n_mark}x (need "
              f"exactly 2: one assignment, one read)", file=sys.stderr)
        return 96

    # 3: the training launch line still occurs exactly once.
    n_launch = new.count(LAUNCH_LINE)
    if n_launch != 1:
        print(f"REFUSE 96: post-condition 3 failed -- the run_in_container launch line "
              f"occurs {n_launch}x (need exactly 1)", file=sys.stderr)
        return 96

    # 4: rc= is assigned on both arms of the new guard.
    guard = _branch_region(new, 'if [[ "${FS_SKIP_TRAIN:-0}" == 1 ]]; then')
    arms = guard.split("\nelse\n")
    if len(arms) != 2 or not all(re.search(r"^\s*rc=", arm, re.M) for arm in arms):
        print("REFUSE 96: post-condition 4 failed -- rc= is not assigned on both arms "
              "of the skip guard; rc is read unguarded at the failure test and set -u "
              "would abort", file=sys.stderr)
        return 96

    # 5: the file still parses as bash. If bash itself is unavailable the parse is
    # UNMEASURED -- printed and exited 95, never skipped.
    with tempfile.NamedTemporaryFile("w", suffix=".sh", delete=False) as fh:
        fh.write(new)
        tmp = fh.name
    try:
        try:
            r = subprocess.run(["bash", "-n", tmp], capture_output=True, text=True)
        except FileNotFoundError:
            print("UNMEASURED 95: bash is unavailable, so the patched launcher cannot "
                  "be parse-checked; UNMEASURED is not PASS", file=sys.stderr)
            return 95
    finally:
        os.unlink(tmp)
    if r.returncode != 0:
        print(f"REFUSE 96: post-condition 5 failed -- bash -n: {r.stderr.strip()[:300]}",
              file=sys.stderr)
        return 96

    LAUNCH.write_text(new, "utf-8")
    print("fs187: applied -- edit 1: post-mortem afterany link sets the internal skip "
          "flag instead of exit 0; edit 2: training launch wrapped in the skip guard "
          "with rc assigned on both arms, shared adjudication tail reused verbatim")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
