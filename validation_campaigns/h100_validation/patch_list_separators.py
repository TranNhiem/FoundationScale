#!/usr/bin/env python3
"""Stage for the launch plane: patch finding #139 in h100/gen/launch_fs_h100.fixed.sh.

Line 18 of the launcher sets a global IFS of newline+tab -- a deliberate safety
setting -- and two operator-facing list knobs are then split under it while
telling the operator something else. FS_ALLOWED_PATH_ROOTS is documented as
"Set it to the space-separated absolute root(s)", yet under the global IFS
"/work /home" arrives as ONE root (measured: 1 word where default IFS gives 2),
so the check refuses and the refusal names the operator's MODEL_DIR when the
actual fault is the separator. ADJUDICATORS_RAW states no separator at all --
"Each entry is invoked as ...", entries plural -- and its read yields a
1-element array on space-separated input, whose downstream refusal names an
adjudicator path. Both sites eventually refuse, so nothing silently passes;
they just blame the wrong value, and the cause is invisible without reading
line 18. The correct idiom already exists in-file (fs_tl_seconds splits
walltimes under a function-local `IFS=:` near line 118): it was simply never
applied to the two knobs an operator actually types.

This stage rewrites both splits under a LOCALLY-SCOPED IFS that names space,
tab AND newline -- space because the prose promises it, tab/newline because
that is what operators are currently forced to use and must keep working --
and it widens site 2's comment AND refusal message so prose and parser agree
in both directions. Message-and-parser drifting apart IS the defect; fixing
the parser and leaving the prose would be the same bug facing the other way.
The global IFS is explicitly NOT touched: deleting line 18 is the tempting
wrong fix, trading an invisible mis-split for glob-and-space expansion across
the rest of the script. A gate proves it is still there.

Exit codes (House Doctrine #5: UNMEASURED is a declared state, not a fail):
  0  every gate green, file rewritten in place (or confirmed already patched)
  2  at least one gate red; nothing was written
  3  target unreadable; nothing measured; refused
"""

from __future__ import annotations

import argparse
import os
import pathlib
import re
import subprocess
import sys
import tempfile

TARGET_REL = pathlib.Path("h100") / "gen" / "launch_fs_h100.fixed.sh"

# Literal text of the deliberate safety setting on line 18. The patch must
# leave its occurrence count untouched; a fix that widens a split by removing
# the guard would green every behavioural test and still be a defect.
GLOBAL_IFS = "IFS=$'\\n\\t'"

# The widened separator set, as literal file text: space, tab, newline. Space
# because the refusal prose promises it; tab/newline because that is the form
# the old IFS silently forced on operators, and breaking them to fix the prose
# would just be the defect rotated ninety degrees.
LOCAL_IFS_VALUE = "$' \\t\\n'"

# --- SITE 1: the FS_ALLOWED_PATH_ROOTS split inside fs_path_under_allowed_root.
SITE1_OLD = (
    '  [[ -n "$p" ]] || return 1\n'
    '  # shellcheck disable=SC2086\n'
    '  for root in ${FS_ALLOWED_PATH_ROOTS}; do\n'
)
SITE1_LOCAL_LINE = "  local IFS=" + LOCAL_IFS_VALUE + "\n"
SITE1_NEW = (
    '  [[ -n "$p" ]] || return 1\n'
    '  # fs139: widen the split to include space. The refusal above tells the\n'
    '  # operator to set "space-separated" roots, and prose and parser must\n'
    '  # agree -- fixing the parser but leaving the prose would be the same\n'
    '  # defect facing the other way. The split is widened with a\n'
    '  # function-local IFS, NOT by editing the global safety IFS on line 18:\n'
    '  # touching that line would re-enable glob-and-space splitting for every\n'
    '  # other expansion in the script, and is the tempting wrong fix.\n'
    + SITE1_LOCAL_LINE
    + '  # shellcheck disable=SC2086\n'
    '  for root in ${FS_ALLOWED_PATH_ROOTS}; do\n'
)

# --- SITE 2: the adjudicator read. An assignment-prefix IFS is scoped to this
# one builtin, which is the narrowest possible widening.
SITE2_READ = 'read -r -a ADJUDICATORS <<< "$ADJUDICATORS_RAW"'
SITE2_READ_NEW = "IFS=" + LOCAL_IFS_VALUE + " " + SITE2_READ
# These comment lines must never contain the token "fail " -- the refusal-line
# matcher below keys on `fail ... adjudicator`, and a comment that looks like
# the anchor would turn a unique match into an ambiguity we then refuse on.
SITE2_COMMENT_LINES = (
    'fs139: this list used to be split under the global safety IFS while its',
    'comment said "Each entry is invoked as ..." -- entries, plural, separator',
    'never stated -- and its refusal then named an adjudicator path, blaming',
    'the value when the fault was the invisible separator from line 18.',
    'Prose and parser now agree, in both directions: one adjudicator per word,',
    'separated by space, tab or newline (space because that is what an',
    'operator types; tab/newline because that is what the old behaviour forced',
    'and must keep working). A claim broader than its evidence is a defect',
    'even when the code is correct -- the stale refusal message was that.',
    'The assignment-prefix IFS scopes to this read alone, so the global',
    'safety setting is not weakened for the rest of the script.',
)
# Appended inside the site-2 refusal string so the message finally names the
# separator it always assumed. Distinct phrasing from the comment above keeps
# occurrence counts single-valued.
REFUSAL_CLAUSE = " -- entries are space/tab/newline-separated, one adjudicator per word"
# Anchored on the ENV VAR, not on the word "adjudicator". The looser form matched three refusals
# in the site-2 block (the empty-RAW guard, the zero-length-array guard and the empty-token guard),
# so the stage correctly refused rather than guessing -- but refusing forever is not a fix. Site 2
# is specifically the guard on the operator-facing variable, and naming that variable is what makes
# the anchor identify one line instead of a neighbourhood.
REFUSAL_RE = re.compile(r"^[^\n]*fail [^\n]*FS_CHECKPOINT_ADJUDICATORS[^\n]*$", re.MULTILINE)

# The pre-existing in-file CORRECT control (fs_tl_seconds, ~line 118): proof
# that the right idiom was already known to this file. Gate D diffs its
# occurrence counts before/after -- the fix must converge on this form, never
# disturb it.
CONTROL_IDIOM = ('local IFS=:\n', 'local -a f; read -r -a f <<< "$t"')

# Behavioural snippets: every split-under-test runs under the same global IFS
# the launcher sets on line 18, on a two-entry SPACE-separated input -- the
# exact documented form that mis-split. The _NEW snippets embed the very
# lines this stage writes, so the behaviour gate cannot pass on prose alone.
GLOBAL_SETUP = GLOBAL_IFS + "\n"
SNIP_SITE1_NEW = (
    GLOBAL_SETUP
    + "FS_ALLOWED_PATH_ROOTS='/work /home'\n"
    + "count=0\n"
    + "probe() {\n"
    + "  local p=/x root\n"
    + SITE1_LOCAL_LINE
    + "  for root in ${FS_ALLOWED_PATH_ROOTS}; do\n"
    + "    count=$((count + 1))\n"
    + "  done\n"
    + "}\n"
    + "probe\n"
    + "printf '%s\\n' \"$count\"\n"
)
SNIP_SITE2_NEW = (
    GLOBAL_SETUP
    + "ADJUDICATORS_RAW='/opt/a.sh /opt/b.sh'\n"
    + SITE2_READ_NEW + "\n"
    + "printf '%s\\n' \"${#ADJUDICATORS[@]}\"\n"
)
SNIP_SITE1_OLD = (
    GLOBAL_SETUP
    + "FS_ALLOWED_PATH_ROOTS='/work /home'\n"
    + "count=0\n"
    + "for root in ${FS_ALLOWED_PATH_ROOTS}; do count=$((count + 1)); done\n"
    + "printf '%s\\n' \"$count\"\n"
)
SNIP_SITE2_OLD = (
    GLOBAL_SETUP
    + "read -r -a ADJUDICATORS <<< '/opt/a.sh /opt/b.sh'\n"
    + "printf '%s\\n' \"${#ADJUDICATORS[@]}\"\n"
)


class Refusal(Exception):
    """Raised when the file's shape makes a blind edit possible; we patch nothing."""


def patch(text: str) -> tuple[str, str]:
    """Return (new_text, mode) with mode 'applied' or 'noop'; raise Refusal otherwise.

    Idempotency is load-bearing: a build stage that cannot tell 'already
    patched' from 'unexpected shape' must refuse, not re-anchor on its own
    output or patch a first hit among many.
    """
    s1o = text.count(SITE1_OLD)
    s2o = sum(1 for ln in text.splitlines() if ln.strip() == SITE2_READ)
    s1n = text.count("local IFS=" + LOCAL_IFS_VALUE)
    s2n = sum(1 for ln in text.splitlines() if ln.strip() == SITE2_READ_NEW)

    if s1o == 0 and s2o == 0 and s1n == 1 and s2n == 1:
        return text, "noop"
    if not (s1o == 1 and s2o == 1 and s1n == 0 and s2n == 0):
        # A multi-match anchor must refuse: patching the first hit of an
        # ambiguous anchor is how a stage silently edits the wrong site and
        # reports success over a denominator it invented.
        raise Refusal(f"anchors site1-old={s1o} site2-old={s2o} site1-new={s1n} site2-new={s2n}")

    out = text.replace(SITE1_OLD, SITE1_NEW, 1)

    lines = out.splitlines(keepends=True)
    idx = next(i for i, ln in enumerate(lines) if ln.strip() == SITE2_READ)
    indent = lines[idx][: len(lines[idx]) - len(lines[idx].lstrip())]
    comment = "".join(indent + "# " + c + "\n" for c in SITE2_COMMENT_LINES)
    lines[idx] = comment + indent + SITE2_READ_NEW + "\n"
    out = "".join(lines)

    # The refusal message must gain the separator it always assumed. Exactly
    # one `fail ... adjudicator` line is expected; zero means the anchor
    # drifted, more than one means we cannot know which refusal is site 2's,
    # and either way a blind edit is worse than a refusal.
    matches = REFUSAL_RE.findall(out)
    if len(matches) != 1:
        raise Refusal(f"site-2 refusal line: expected exactly 1, measured {len(matches)}")
    line = matches[0]
    stripped, tail = line.rstrip(" \t"), ""
    if stripped.endswith("\\"):
        stripped, tail = stripped[:-1].rstrip(" \t"), " \\"
    if not stripped.endswith('"'):
        raise Refusal("site-2 refusal line is not a plain quoted fail message; refusing to edit blind")
    out = out.replace(line, stripped[:-1] + REFUSAL_CLAUSE + '"' + tail, 1)

    return out, "applied"


def _words(script: str) -> tuple[int, str]:
    proc = subprocess.run(["bash", "-s"], input=script, capture_output=True, text=True)
    out = [ln for ln in proc.stdout.splitlines() if ln.strip()]
    return proc.returncode, (out[-1].strip() if out else "")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    args = ap.parse_args()
    target = pathlib.Path(args.root) / TARGET_REL

    gates: list[tuple[str, bool, str]] = []

    if not (target.is_file() and os.access(target, os.R_OK)):
        # Fail CLOSED and say what happened: a gate table over a file we could
        # not read would be a claim broader than its evidence, so this is
        # UNMEASURED with its own exit code -- laundered into neither PASS nor
        # FAIL.
        print("gate table:")
        print(f"  P1 target readable at h100/gen/launch_fs_h100.fixed.sh: UNMEASURED")
        print("0/1 gates green; 1 UNMEASURED (exit 3); nothing written")
        return 3

    original = target.read_text(encoding="utf-8")
    gates.append(("P1 target exists and is readable", True, f"{len(original)} bytes"))

    s1o = original.count(SITE1_OLD)
    s2o = sum(1 for ln in original.splitlines() if ln.strip() == SITE2_READ)
    s1n = original.count("local IFS=" + LOCAL_IFS_VALUE)
    s2n = sum(1 for ln in original.splitlines() if ln.strip() == SITE2_READ_NEW)
    actionable = (s1o == 1 and s2o == 1 and s1n == 0 and s2n == 0) or (
        s1o == 0 and s2o == 0 and s1n == 1 and s2n == 1
    )
    gates.append(
        ("PRE site-1 and site-2 anchors each in exactly one actionable state (pristine, or already patched)",
         actionable, f"old {s1o}+{s2o}, new {s1n}+{s2n}; anything else is hand-drift and refuses")
    )
    gates.append(
        ("PRE global safety IFS present before patch",
         original.count(GLOBAL_IFS) >= 1,
         f"{original.count(GLOBAL_IFS)} occurrence(s); this stage widens two splits -- it must not touch line 18")
    )
    gates.append(
        ("PRE in-file correct control idiom present before patch (baselined for gate D)",
         all(original.count(c) >= 1 for c in CONTROL_IDIOM),
         f"{[original.count(c) for c in CONTROL_IDIOM]} occurrence(s)")
    )
    if not all(ok for _, ok, _ in gates):
        return _fail(gates)

    try:
        patched, mode = patch(original)
        gates.append((f"PRE patch applicable (anchors unique, refusal line editable), mode {mode}",
                      True, f"{len(original)} -> {len(patched)} bytes"))
    except Refusal as exc:
        gates.append(("PRE patch applicable (anchors unique, refusal line editable)", False, str(exc)[:120]))
        return _fail(gates)

    gates.append(
        ("A1 site 1 rewritten: function-local IFS + split present exactly once, old block gone",
         patched.count("local IFS=" + LOCAL_IFS_VALUE) == 1 and SITE1_OLD not in patched,
         f"{patched.count('local IFS=' + LOCAL_IFS_VALUE)} rewritten / {patched.count(SITE1_OLD)} stale occurrence(s)")
    )
    gates.append(
        ("A2 site 2 rewritten: IFS-scoped read present, bare read gone, separator comment inserted",
         sum(1 for ln in patched.splitlines() if ln.strip() == SITE2_READ_NEW) == 1
         and sum(1 for ln in patched.splitlines() if ln.strip() == SITE2_READ) == 0
         and patched.count("# fs139: this list used to be split") == 1,
         "verified by searching the patched text, not by trusting the replace")
    )
    gates.append(
        ("A3 prose and parser agree at site 2: refusal message now names its separator",
         patched.count(REFUSAL_CLAUSE) == 1,
         f"{patched.count(REFUSAL_CLAUSE)} occurrence(s) of the separator clause")
    )

    # bash -n runs on a temp copy of the PATCHED text, because syntax-checking
    # the pre-patch file would green a gate over bytes we never ship.
    fd, tmp = tempfile.mkstemp(dir=str(target.parent), suffix=".fs139-check.sh")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(patched)
        try:
            proc = subprocess.run(["bash", "-n", tmp], capture_output=True, text=True)
            gates.append(("B bash -n clean on the patched launcher",
                          proc.returncode == 0, f"rc {proc.returncode}; {proc.stderr.strip()[:120]}"))
        except OSError as exc:
            # No bash, no verification; an unverified artifact is never written.
            gates.append(("B bash -n clean on the patched launcher", False, f"bash unavailable: {exc}"))
    finally:
        pathlib.Path(tmp).unlink(missing_ok=True)

    gates.append(
        ("C global safety IFS still present, count unchanged (the fix is local, not a deletion of line 18)",
         0 < original.count(GLOBAL_IFS) == patched.count(GLOBAL_IFS),
         f"{original.count(GLOBAL_IFS)} -> {patched.count(GLOBAL_IFS)} occurrence(s)")
    )
    gates.append(
        ("D correct pre-existing idiom near line 118 untouched",
         all(original.count(c) == patched.count(c) for c in CONTROL_IDIOM),
         f"before {[original.count(c) for c in CONTROL_IDIOM]}, after {[patched.count(c) for c in CONTROL_IDIOM]}")
    )

    try:
        second, mode2 = patch(patched)
        gates.append(("IDEM re-application is a byte-exact no-op",
                      mode2 == "noop" and second == patched,
                      f"second pass mode {mode2}, {len(second)}/{len(patched)} bytes identical"))
    except Refusal as exc:
        gates.append(("IDEM re-application is a byte-exact no-op", False, str(exc)[:120]))

    # The MUST_FIRE pair is not optional ceremony: a behavioural check that has
    # never been observed going red is not known to measure anything. Running
    # the ORIGINAL forms and asserting they still mis-split (1 word) is the
    # control going red on command.
    for label, script, want in (
        ("BEHAVIOUR MUST_PASS site 1: patched split of '/work /home' yields 2 words under the line-18 IFS",
         SNIP_SITE1_NEW, "2"),
        ("BEHAVIOUR MUST_PASS site 2: patched read of two space-separated adjudicators yields 2 elements",
         SNIP_SITE2_NEW, "2"),
        ("BEHAVIOUR MUST_FIRE site 1: ORIGINAL split is observed yielding 1 word (the control can go red)",
         SNIP_SITE1_OLD, "1"),
        ("BEHAVIOUR MUST_FIRE site 2: ORIGINAL read is observed yielding 1 element (the control can go red)",
         SNIP_SITE2_OLD, "1"),
    ):
        try:
            rc, got = _words(script)
            gates.append((label, rc == 0 and got == want,
                          f"rc {rc}, measured {got} word(s) of 2 expected, required {want}"))
        except OSError as exc:
            gates.append((label, False, f"bash unavailable: {exc}"))

    _report(gates)
    if not all(ok for _, ok, _ in gates):
        print("refusing to write: at least one gate red; the pre-patch file is untouched")
        return 2

    if mode == "applied":
        # mkstemp makes 0600; re-impose the launcher's mode so patching in
        # place cannot silently strip +x off the artifact that gets executed.
        mode_bits = target.stat().st_mode & 0o7777
        fd, tmp = tempfile.mkstemp(dir=str(target.parent), suffix=".tmp")
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(patched)
        os.chmod(tmp, mode_bits)
        os.replace(tmp, target)

    print(
        f"{'patched' if mode == 'applied' else 'verified already-patched'} {target}: "
        f"2/2 split sites rewritten (site 1 local-IFS, site 2 read-scoped IFS), "
        f"1/1 refusal messages reconciled with their parser, "
        f"global safety IFS intact at {patched.count(GLOBAL_IFS)} occurrence(s), "
        f"behaviour 2/2 words at both sites with both MUST_FIRE controls observed at 1; "
        f"{len(original)} -> {len(patched)} bytes"
    )
    return 0


def _fail(gates: list[tuple[str, bool, str]]) -> int:
    _report(gates)
    return 2


def _report(gates: list[tuple[str, bool, str]]) -> None:
    print("gate table:")
    for label, ok, note in gates:
        print(f"  {label}: {'PASS' if ok else 'FAIL'}{(' (' + note + ')') if note else ''}")
    # The predicate was missing here, so this counted EVERY gate and printed "5/5 gates green"
    # over a table containing a FAIL. The stage still refused -- behaviour was correct -- but the
    # summary line said the opposite of the table above it, and the summary is what gets read.
    green = sum(1 for _, ok, _ in gates if ok)
    print(f"{green}/{len(gates)} gates green")


if __name__ == "__main__":
    raise SystemExit(main())
