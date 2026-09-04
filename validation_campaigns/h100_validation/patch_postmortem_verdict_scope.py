#!/usr/bin/env python3
"""#193: two fixes that are each correct in isolation compose into a dead branch.

#187 made the post-mortem afterany chain link stop exiting 0 over zero
adjudications. It did that by setting a launcher-internal skip flag in the
FS_PHASE == post-mortem branch and routing the link down the SHARED adjudication
tail. Its own comment states the design intent:

    # fs187: only the training launch is skipped. Backend init, runtime setup, the
    # bind plane, the in-container GPU census and fs_compose_launch have all already
    # run on the ordinary path above, and the shared adjudication tail below is what
    # produces the verdict -- reused verbatim, not duplicated.

#171 later added a success-arm guard directly above that tail: rc == 0 is not PASS
until the trainer's declaration is checked -- a run that exits clean having
declared nothing is the vacuous-pass hole. The guard maps (rc, RUN_LOG) through
fs_map_run_verdict and exits before adjudication when the mapping is not 0. On the
post-mortem path NO TRAINER RAN, so RUN_LOG carries no verdict line by
construction; the mapper correctly returns 95 and the launcher exits before
adjudicate_tree is ever called. The adjudication tail is unreachable on the only
path #187 exists to serve.

MEASURED BEFORE WRITING -- job 37347, 8xH100, FS_PHASE=post-mortem, over a tree
holding 2 checkpoint directories:

  * all 8 srun steps COMPLETED;
    FS_COLLECTIVE world=8 got=28.0 expected=28.0 spread=0.0 verdict=OK;
    fs129: collective probe PASS
  * the run log is 51 lines and contains the string ADJUDICATE 0 times
  * its last line is
    END rc=0 mapped_rc=95 phase=train FAILED (fs_map_run_verdict: verdict=NONE rc=0 mapped=95 -- no verdict line; exited clean having declared nothing (the vacuous-pass hole, closed))
  * sacct: 37347 FAILED 95:0 00:02:36

The 95 is honest -- the run genuinely measured nothing -- which is why this hid:
the exit code is CORRECT and the REASON is wrong. #187 is not reopened by this
finding; it is DEFEATED by it.

A second, smaller defect has the same shape, and the fix closes it for free: the
post-mortem branch prints an operator-facing banner promising that the tree "is
adjudicated below". That sentence is false in the shipped artifact today. The
banner is NOT edited -- this fix makes an existing claim true rather than
weakening it.

THE FIX. The #171 guard's unstated precondition is "a trainer ran and was
supposed to declare a verdict". The post-mortem arm violates that precondition,
so:

  Edit 1 scopes the success-arm mapper to the arm that launched a trainer, keyed
  on the same launcher-internal skip flag the #187 skip block already reads. The
  post-mortem arm takes the else branch, which logs ONE line recording that the
  trainer-verdict mapper was deliberately not applied -- no trainer was launched,
  so there is no declaration to check -- and names the adjudication tail as this
  arm's verdict source. That line is not decoration: an arm that skips a check
  must say so in the log, or the next reader cannot tell "checked and passed"
  from "not checked".

  Edit 2 rewords exactly one message. The tail's two refusals keep firing
  unchanged -- the positivity refusal (a post-mortem over an EMPTY tree still
  exits 95; this is the whole of #187 and it survives) and the fs176
  observed == found refusal (partial coverage is RED, not PASS) -- but the
  positivity refusal's message said "after training", false on an arm where no
  training happened. It now names the state in words true on both arms.
  adjudicate_tree itself, checkpoint_observed, checkpoint_found and the fs176
  logic and threshold are not touched. The final END rc=0 line was CHECKED and
  needed no change: it already interpolates ${FS_PHASE:-train}, so it already
  prints phase=post-mortem on this arm and a reader can tell a post-mortem
  adjudication from a training one. Nothing that did not need changing was
  changed.

STAGE MECHANICS (shared with the sibling stages): fail closed on a missing or
unreadable launcher (96). Each anchor must occur EXACTLY once or the stage
refuses, naming the anchor and the count observed -- zero occurrences is NOT
"already applied", it is generator drift to refuse, because a stage that
silently no-ops when its anchor is gone is how a patch quietly stops being
applied. The MUST_FIRE premise is checked BEFORE the anchors: if the success-arm
mapper is already inside a skip-flag guard, the pre-image does not exhibit the
defect and the stage REFUSES -- it does not silently no-op on a file it does not
recognise. Post-conditions are verified on the produced text BEFORE anything is
written:

  1. the dead branch is gone, verified STRUCTURALLY, not by grep for this
     stage's own comment: the text between the skip block's closing fi and the
     adjudicate_tree call is scanned with if/fi depth tracking (comment lines
     stripped, so neither the launcher's prose nor this stage's can stand
     inside the denominator), and any exit statement -- or verdict-mapper call
     -- at depth zero refuses;
  2. adjudicate_tree's body is byte-identical before and after;
  3. the transform is byte-idempotent: applying it to its own output changes
     nothing (measured, not assumed -- the already-patched shape is a REFUSE
     via the MUST_FIRE premise, never a silent no-op);
  4. the file still parses as bash (bash -n on a temporary copy; if bash itself
     is unavailable that is UNMEASURED -- printed, exit 95, never skipped).

CONTROLS run on every invocation, printed under a CONTROLS heading, and never
touch the repository: real bash is driven over SYNTHETIC fragments -- the
genuine tail region lifted from the pre-image and from the post-image -- built
in a temp directory with stub fs_map_run_verdict / fail / adjudicate_tree /
run_in_container functions. Each control prints one line naming what was driven
and what was observed; if any MUST_FIRE fails to fire, or any MUST_PASS fails,
the stage refuses and writes nothing.

  MUST_FIRE/DEAD_BRANCH               pre-image, skip flag set, mapper returns
                                      95: exit 95 and adjudicate_tree NEVER
                                      called -- the job 37347 drill; without it
                                      the stage asserts a defect it never
                                      observed.
  MUST_PASS/REACHES_TAIL              post-image, same drive: adjudicate_tree IS
                                      called and the run reaches the END rc=0
                                      line.
  MUST_FIRE/EMPTY_TREE_STILL_95       post-image, stub observes ZERO
                                      checkpoints: exit 95. Proves the fix does
                                      not reopen #187; the most important
                                      control here.
  MUST_FIRE/PARTIAL_COVERAGE_STILL_5  post-image, stub observes 1 of 2: exit 5
                                      (fs176 survives).
  MUST_PASS/TRAIN_ARM_UNCHANGED       post-image, skip flag UNSET, mapper
                                      returns 95: exit 95 and the tail is never
                                      reached. The #171 guard is untouched on
                                      the arm it was written for; a fix that
                                      widens a hole to close a branch is worse
                                      than the branch.

EXIT CODES: 0 applied, 95 UNMEASURED, 96 REFUSE. Never any other code -- and
`raise SystemExit("text")` exits 1, so messages are printed and the number is
returned explicitly.
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

# Structural landmarks (not edited, but counted exactly once and used to delimit the
# regions the post-conditions and controls are extracted from).
SKIP_BLOCK_OPEN = 'if [[ "${FS_SKIP_TRAIN:-0}" == 1 ]]; then'
GUARD_OPEN = 'if [[ "${FS_SKIP_TRAIN:-0}" != 1 ]]; then'
ADJUDICATE_DEF = "adjudicate_tree() {"
ADJUDICATE_CALL = 'adjudicate_tree "$adj_root"'

# --- Edit 1: scope the #171 success-arm verdict mapper to the arm that launched a
# trainer. The anchor is the whole success-arm block, comment included, so the comment
# stays attached to the code it describes when both are re-indented one level.
ANCHOR_1 = (
    "# fs175: finding #171, success arm -- rc==0 is not PASS until the trainer's\n"
    "# declaration is checked: a run that exits clean having declared nothing is the\n"
    "# vacuous-pass hole, and before this block it fell straight through to\n"
    "# adjudication. Map (rc, log) and refuse anything that is not 0 BEFORE\n"
    "# adjudicating. This is the whole point of the doctrine and it costs these\n"
    "# lines only.\n"
    'fs175_reason="${RUN_LOG}.fs175_reason"\n'
    'mapped_rc="$(fs_map_run_verdict "$rc" "$RUN_LOG" 2>"$fs175_reason")" || fail 96 "fs175: verdict mapper unavailable/failed (backend drift); refusing to guess a state"\n'
    'map_reason="$(cat "$fs175_reason" 2>/dev/null || true)"; rm -f "$fs175_reason"\n'
    'if [[ "$mapped_rc" != 0 ]]; then\n'
    "  printf 'END rc=%s mapped_rc=%s phase=train FAILED (%s)\\n' \"$rc\" \"$mapped_rc\" \"$map_reason\" | tee -a \"$RUN_LOG\"\n"
    '  exit "$mapped_rc"\n'
    "fi\n"
)
# The original block is preserved byte-for-byte apart from two spaces of indentation.
# The else arm's printf is a load-bearing log line, not decoration: it records that
# the mapper was DELIBERATELY not applied on this arm, why, and where this arm's
# verdict comes from instead.
REPLACEMENT_1 = (
    "# fs193: finding #193 -- the mapper below has an unstated precondition: a\n"
    "# trainer ran and was supposed to declare a verdict. The post-mortem arm\n"
    "# violates it (no trainer ran, so the run log carries no verdict line by\n"
    "# construction), and an unguarded mapper on that arm maps to 95 and leaves\n"
    "# before the adjudication tail is ever reached -- measured on job 37347,\n"
    "# 51 log lines, zero adjudications. The mapper is therefore scoped to the\n"
    "# arm that launched a trainer; the post-mortem arm's verdict source is the\n"
    "# shared adjudication tail, which is what fs187 routed it to.\n"
    'if [[ "${FS_SKIP_TRAIN:-0}" != 1 ]]; then\n'
    "  # fs175: finding #171, success arm -- rc==0 is not PASS until the trainer's\n"
    "  # declaration is checked: a run that exits clean having declared nothing is the\n"
    "  # vacuous-pass hole, and before this block it fell straight through to\n"
    "  # adjudication. Map (rc, log) and refuse anything that is not 0 BEFORE\n"
    "  # adjudicating. This is the whole point of the doctrine and it costs these\n"
    "  # lines only.\n"
    '  fs175_reason="${RUN_LOG}.fs175_reason"\n'
    '  mapped_rc="$(fs_map_run_verdict "$rc" "$RUN_LOG" 2>"$fs175_reason")" || fail 96 "fs175: verdict mapper unavailable/failed (backend drift); refusing to guess a state"\n'
    '  map_reason="$(cat "$fs175_reason" 2>/dev/null || true)"; rm -f "$fs175_reason"\n'
    '  if [[ "$mapped_rc" != 0 ]]; then\n'
    "    printf 'END rc=%s mapped_rc=%s phase=train FAILED (%s)\\n' \"$rc\" \"$mapped_rc\" \"$map_reason\" | tee -a \"$RUN_LOG\"\n"
    '    exit "$mapped_rc"\n'
    "  fi\n"
    "else\n"
    "  printf 'fs193: trainer-verdict mapper deliberately not applied on this arm -- no trainer was launched, so there is no declaration to check; the adjudication tail below is the verdict source for this arm\\n' | tee -a \"$RUN_LOG\"\n"
    "fi\n"
)

# --- Edit 2: the positivity refusal's message said "after training", which is false
# on the post-mortem arm. The refusal itself, its 95, and everything it compares are
# untouched; only the words change, to a phrasing true on both arms that still names
# the state.
ANCHOR_2 = (
    '[[ "$checkpoint_observed" -gt 0 ]] || fail 95 \'no checkpoint-save units observed after training; UNMEASURED is not PASS\'\n'
)
REPLACEMENT_2 = (
    '[[ "$checkpoint_observed" -gt 0 ]] || fail 95 \'no checkpoint-save units observed to adjudicate; UNMEASURED is not PASS\'\n'
)

# Stub harness prepended to the lifted tail fragment for every control. The stubs
# stand in for fs_map_run_verdict / fail / adjudicate_tree / run_in_container; the
# adjudicate_tree stub records its invocation in $CALLS and publishes the
# observed/found counters the scenario dictates, exactly as the real walker would.
CONTROL_PRELUDE = (
    "set -u\n"
    'OUT_DIR="$CTL_DIR/out"\n'
    'RUN_LOG="$CTL_DIR/run.log"\n'
    'CALLS="$CTL_DIR/calls"\n'
    'mkdir -p "$OUT_DIR"\n'
    ': >"$RUN_LOG"\n'
    "top_args=(stub)\n"
    "LAUNCH_CMD=':'\n"
    "checkpoint_observed=0\n"
    "checkpoint_found=0\n"
    'fs_map_run_verdict() { printf \'%s\\n\' "$CTL_MAP_RC"; printf \'stub mapper: verdict=NONE rc=%s mapped=%s -- no verdict line\\n\' "$1" "$CTL_MAP_RC" >&2; return 0; }\n'
    'fail() { printf \'FATAL rc=%s %s\\n\' "$1" "$2" >&2; exit "$1"; }\n'
    'adjudicate_tree() { printf \'called root=%s phase=%s\\n\' "$1" "$2" >>"$CALLS"; checkpoint_observed="$CTL_OBSERVED"; checkpoint_found="$CTL_FOUND"; return 0; }\n'
    "run_in_container() { return 0; }\n"
)


def _tail_region(text: str) -> str:
    """The text between the skip block's closing `fi` and the adjudicate_tree call --
    the region in which the #193 dead branch lived."""
    start = text.index(SKIP_BLOCK_OPEN)
    fi = text.index("\nfi\n", start) + len("\nfi\n")
    end = text.index(ADJUDICATE_CALL, fi)
    return text[fi:end]


def _fn_body(text: str) -> str:
    """adjudicate_tree's whole body, from its definition line to its closing brace."""
    start = text.index(ADJUDICATE_DEF)
    end = text.index("\n}\n", start) + len("\n}\n")
    return text[start:end]


def _control_fragment(text: str) -> str:
    """The genuine tail region (skip block through EOF) lifted verbatim for the
    controls -- the controls drive the real code, not a hand-copied sketch of it."""
    return text[text.index(SKIP_BLOCK_OPEN):]


def _unconditional_exits(region: str) -> list[str]:
    """Lines in REGION that would run an `exit` -- or the verdict mapper -- with no
    enclosing conditional. Structural, not textual: if/fi depth is tracked and
    comment lines are stripped, so neither the launcher's prose nor this stage's own
    comments can satisfy or trip the scan (a scanner that matches its own
    explanatory prose has no denominator)."""
    bad: list[str] = []
    depth = 0
    for line in region.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if re.match(r"^fi\b", stripped):
            depth = max(0, depth - 1)
            continue
        if re.match(r"^if\b", stripped):
            depth += 1
            continue
        if depth == 0 and (re.match(r"^exit\b", stripped) or "fs_map_run_verdict" in stripped):
            bad.append(line)
    return bad


def _run_control(name: str, fragment: str, drove: str, *, skip_train: bool,
                 phase: str, map_rc: int, observed: int, found: int,
                 expect_rc: int, expect_calls: bool, expect_end0: bool = False,
                 expect_log: str | None = None) -> bool | None:
    """Drive real bash over one synthetic fragment in a temp directory. Returns True
    on pass, False on fail, None when bash itself is unavailable (UNMEASURED)."""
    with tempfile.TemporaryDirectory(prefix="fs193_ctl_") as td:
        script = pathlib.Path(td) / "fragment.sh"
        script.write_text(CONTROL_PRELUDE + fragment, "utf-8")
        env = dict(os.environ)
        env.pop("FS_SKIP_TRAIN", None)
        env.pop("FS_PHASE", None)
        env.update({"CTL_DIR": td, "CTL_MAP_RC": str(map_rc),
                    "CTL_OBSERVED": str(observed), "CTL_FOUND": str(found),
                    "FS_PHASE": phase})
        if skip_train:
            env["FS_SKIP_TRAIN"] = "1"
        try:
            r = subprocess.run(["bash", str(script)], capture_output=True, text=True,
                               env=env, cwd=td)
        except FileNotFoundError:
            return None
        calls_path = pathlib.Path(td) / "calls"
        calls = calls_path.read_text("utf-8").splitlines() if calls_path.exists() else []
        log_path = pathlib.Path(td) / "run.log"
        log = log_path.read_text("utf-8") if log_path.exists() else ""
        ok = (r.returncode == expect_rc
              and (len(calls) > 0) == expect_calls
              and (not expect_end0 or "END rc=0 " in log)
              and (expect_log is None or expect_log in log))
        print(f"CONTROL {name}: drove {drove}; observed exit={r.returncode} "
              f"adjudicate_tree_calls={len(calls)}"
              + (f" end_rc0_in_log={'yes' if 'END rc=0 ' in log else 'no'}" if expect_end0 else "")
              + f" (expected exit={expect_rc} calls={'>0' if expect_calls else '0'})")
        return ok


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

    # MUST_FIRE premise, checked FIRST, before the anchor counts: if the success-arm
    # mapper is already inside a skip-flag guard, this pre-image does not exhibit the
    # defect. Unlike #187's already-applied no-op, this stage REFUSES -- a stage that
    # silently no-ops on a file it does not recognise is how a patch quietly stops
    # being applied, and here the absence of the defect is either "already patched"
    # or "generator drifted", both of which are findings, not zeros.
    if GUARD_OPEN in text:
        print("REFUSE 96: MUST_FIRE premise failed -- the success-arm verdict mapper "
              "is already scoped inside a skip-flag guard, so this pre-image does not "
              "exhibit the #193 dead branch; refusing rather than no-oping on a file "
              "this stage does not recognise", file=sys.stderr)
        return 96

    # Each anchor must occur EXACTLY once. Zero occurrences is NOT "already applied"
    # -- the premise check above is the only already-patched test, so a missing
    # anchor here means the generator drifted.
    for name, anchor in (("edit 1 (success-arm verdict-mapper block)", ANCHOR_1),
                         ("edit 2 (positivity-refusal message)", ANCHOR_2),
                         ("skip block (#187)", SKIP_BLOCK_OPEN),
                         ("adjudicate_tree definition", ADJUDICATE_DEF),
                         ("adjudicate_tree call", ADJUDICATE_CALL)):
        n = text.count(anchor)
        if n != 1:
            print(f"REFUSE 96: anchor for {name} occurs {n}x (need exactly 1) in "
                  f"{LAUNCH.name}; the generator emitted a different shape than #193 "
                  f"measured -- refusing rather than patching blind", file=sys.stderr)
            return 96

    new = text.replace(ANCHOR_1, REPLACEMENT_1, 1).replace(ANCHOR_2, REPLACEMENT_2, 1)

    # Post-conditions, verified on the produced text BEFORE anything is written.
    # 1: the dead branch is gone, verified structurally -- no exit statement and no
    # verdict-mapper call survives at conditional depth zero in the region between
    # the skip block and the adjudicate_tree call, and the scoping guard stands
    # exactly once.
    n_guard = new.count(GUARD_OPEN)
    if n_guard != 1:
        print(f"REFUSE 96: post-condition 1 failed -- the skip-flag guard around the "
              f"success-arm mapper occurs {n_guard}x (need exactly 1)", file=sys.stderr)
        return 96
    bad = _unconditional_exits(_tail_region(new))
    if bad:
        print(f"REFUSE 96: post-condition 1 failed -- the region between the skip "
              f"block and the adjudicate_tree call still permits the dead branch: "
              f"{len(bad)} unconditional exit/mapper line(s), first: {bad[0].strip()!r}",
              file=sys.stderr)
        return 96

    # 2: adjudicate_tree's body is byte-identical before and after. #193 scopes a
    # guard and rewords one message; it does not touch the walker.
    if _fn_body(text) != _fn_body(new):
        print("REFUSE 96: post-condition 2 failed -- adjudicate_tree's body is not "
              "byte-identical before and after; the walker, its counters and the "
              "fs176 refusal are out of scope for this fix", file=sys.stderr)
        return 96

    # 3: the transform is byte-idempotent -- measured, not assumed. Applying the same
    # two edits to the output must change nothing.
    if new.replace(ANCHOR_1, REPLACEMENT_1, 1).replace(ANCHOR_2, REPLACEMENT_2, 1) != new:
        print("REFUSE 96: post-condition 3 failed -- the transform is not "
              "byte-idempotent: applying it to its own output changes the output",
              file=sys.stderr)
        return 96

    # 4: the file still parses as bash. If bash itself is unavailable the parse is
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
        print(f"REFUSE 96: post-condition 4 failed -- bash -n: {r.stderr.strip()[:300]}",
              file=sys.stderr)
        return 96

    # Controls: real bash over synthetic fragments lifted from the pre-image and the
    # post-image, in a temp directory, with stubbed functions. The repository is
    # never touched to run a control.
    print("CONTROLS")
    pre_frag = _control_fragment(text)
    post_frag = _control_fragment(new)
    controls = (
        ("MUST_FIRE/DEAD_BRANCH", pre_frag,
         "the PRE-image tail fragment with FS_SKIP_TRAIN=1 and a stub verdict mapper "
         "returning 95 (the job 37347 drill)",
         dict(skip_train=True, phase="post-mortem", map_rc=95, observed=2, found=2,
              expect_rc=95, expect_calls=False, expect_log="mapped_rc=95")),
        ("MUST_PASS/REACHES_TAIL", post_frag,
         "the POST-image fragment with FS_SKIP_TRAIN=1 and the same 95-returning mapper",
         dict(skip_train=True, phase="post-mortem", map_rc=95, observed=2, found=2,
              expect_rc=0, expect_calls=True, expect_end0=True)),
        ("MUST_FIRE/EMPTY_TREE_STILL_95", post_frag,
         "the POST-image fragment with FS_SKIP_TRAIN=1 and an adjudicate_tree stub "
         "observing ZERO checkpoints (the #187 drill)",
         dict(skip_train=True, phase="post-mortem", map_rc=0, observed=0, found=0,
              expect_rc=95, expect_calls=True)),
        ("MUST_FIRE/PARTIAL_COVERAGE_STILL_5", post_frag,
         "the POST-image fragment with FS_SKIP_TRAIN=1 and a stub observing 1 of 2 "
         "checkpoints (the fs176 drill)",
         dict(skip_train=True, phase="post-mortem", map_rc=0, observed=1, found=2,
              expect_rc=5, expect_calls=True)),
        ("MUST_PASS/TRAIN_ARM_UNCHANGED", post_frag,
         "the POST-image fragment with FS_SKIP_TRAIN UNSET and a mapper returning 95 "
         "(the arm #171 was written for)",
         dict(skip_train=False, phase="train", map_rc=95, observed=2, found=2,
              expect_rc=95, expect_calls=False)),
    )
    for name, fragment, drove, kw in controls:
        res = _run_control(name, fragment, drove, **kw)
        if res is None:
            print("UNMEASURED 95: bash is unavailable, so the controls cannot run; "
                  "UNMEASURED is not PASS", file=sys.stderr)
            return 95
        if not res:
            print(f"REFUSE 96: control {name} failed -- a MUST_FIRE did not fire or "
                  f"a MUST_PASS did not pass; refusing and writing nothing",
                  file=sys.stderr)
            return 96

    LAUNCH.write_text(new, "utf-8")
    print("fs193: applied -- edit 1: success-arm verdict mapper scoped to the arm "
          "that launched a trainer (skip-flag guard, with a logged deliberately-not-"
          "applied line on the post-mortem arm); edit 2: positivity-refusal message "
          "reworded to be true on both arms; adjudicate_tree byte-identical; the "
          "END rc=0 line already interpolated ${FS_PHASE:-train} and was left "
          "unchanged")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
