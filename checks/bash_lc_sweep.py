#!/usr/bin/env python3
"""BASH-LC sweep -- the inner-shell SOURCE-injection detector (BLOCKER 2 class).

Usage:
  bash_lc_sweep.py LAUNCHER [LAUNCHER ...]        classify every bash -lc site
  bash_lc_sweep.py --self-test                    run the detector's own
                                                  MUST_FIRE / MUST_PASS controls
                                                  against synthetic fixtures
  bash_lc_sweep.py --reinstate-blocker2 SRC DST   copy SRC to DST with BLOCKER
                                                  2's exact unsafe line put
                                                  back (MUST_FIRE setup helper;
                                                  requires the fixed line to
                                                  occur exactly once in SRC)

Classification:
  unsafe   = double-quoted inner source containing $VAR / ${VAR} / $(...)
             OUTSIDE inner single quotes: the outer shell splices the value
             into the inner shell's SOURCE text (word-splitting; $(...)
             executes).
  audited  = single-quoted source with data arguments ($1-style); the
             :513/:1002 nesting idiom (double-quoted source, every expansion
             inside inner single quotes -- enumerated, not blessed: a single
             quote in the value would reopen it, argv is the fix); a
             non-literal argument (enumerated for a human); an EVAL-SHAPED
             source (see below); or a %q-RENDERED substitution (see below).
  declared = an unsafe splice that a DECLARED_SPLICES entry names, with a
             reason, an anchor and the exact token. Declarations are pinned to
             bytes and go RED when they stop describing the tree.

Why this file grew three categories (#239). Its first version asked one
question -- "is there a bare expansion inside double-quoted inner source?" --
and reported five REDs on a tree where two were genuine and three were not
splices at all. A detector that cannot tell "value spliced INTO literal
source" from "value IS the source" produces unattributable REDs, and an
unattributable RED is discarded by the next reader, which is how a real
blocker (the two that WERE genuine, now repaired at lora:726 and lora:1264)
hides inside its own noise. The categories below are enumerated with their
residual risk stated; none of them is an allowlist, and the difference is that
each one is a rule about SHAPE that a control can fire against, not a list of
lines that are excused.

  EVAL-SHAPED -- `bash -lc "$CMD"`, where the whole inner source is one
  expansion and nothing else. The value is not spliced into surrounding
  literal source; it IS the source, which is what `eval` means and what these
  two call sites intend (both launchers build $CMD as the training command and
  the launcher contract suite's conjunct c4 pins this exact spelling). Residual
  risk, stated rather than dismissed: whoever writes $CMD owns its contents
  absolutely. The narrowness is the control -- `bash -lc "$CMD --extra"` is a
  splice again and MUST_FIRE proves this detector still says so.

  %q-RENDERED -- `$(printf '%q ' "${ARR[@]}")`. The substitution's stdout is
  shell-quoted BY printf, so what lands in the inner source is already escaped
  bytes; this is the recommended render, not the defect. Residual risk: only
  `printf %q` is recognised, and only at the head of the substitution.

Pure comment lines are not sites, and neither is a `bash -lc` inside a
TRAILING comment on a code line (#239 D1): comment text must never make this
sweep red or inflate its denominator (doctrine 5, symmetric). The `#` is only
a comment when it is unquoted and at start-of-word, so `bash -lc "x # $V"`
stays a site and stays RED -- there is a MUST_FIRE for exactly that.

Doctrine wiring:
  (1) 0 sites is RED -- both launchers visibly use bash -lc, so zero means the
      sweep broke, i.e. UNMEASURED, not a pass.
  (2) the site count, the comment-mention count, and every site are printed on
      the wire; the partition audited+declared+unsafe must equal the site
      count or the sweep indicts its own classifier.
  (3) the MUST_FIRE lives in .github/workflows/ci.yml: it calls
      --reinstate-blocker2 and demands this sweep refuse the copy. --self-test
      adds the controls that live with the detector, and CI runs it BEFORE the
      live sweep the way `packaging` already does: a detector whose controls
      misbehave has no licence to report a verdict.
  (4) unreadable input is RED, not empty.
  (5) a DECLARED_SPLICES entry that matches nothing is RED, and one that
      matches more than one site is RED. A declaration is a claim about the
      tree; when the tree moves the claim must move with it, and the only way
      to keep an allowlist honest is to make going stale cost more than
      deleting the entry.
Output avoids the verify_summary-ingested tokens by design: no 'green',
'passed', or 'SUMMARY' on any path, and no suite domain prefix at line start.

Exit codes are 0/1 here, not the 0/5/95/96 namespace: every gate under checks/
still speaks 0/1 and CI's legs test for nonzero. Migrating the namespace is a
separate change that must move the callers in the same commit.
"""

from __future__ import annotations

import io
import re
import sys
import tempfile
from contextlib import redirect_stdout
from pathlib import Path

BLOCKER2_FIXED = 'bash -lc \'python3 "$1"\' _ "$COT_PROBE_PY"'
BLOCKER2_BROKEN = 'bash -lc "python3 $COT_PROBE_PY"'

# Deliberate splices, each pinned to the bytes that make it deliberate. `anchor`
# is a substring of the site's own line and `token` is the exact expansion; both
# must be observed or the entry is stale and the sweep goes RED. That is the
# whole difference between this and an allowlist: an allowlist survives the code
# it excuses being deleted, and this does not.
DECLARED_SPLICES = (
    {
        "file": "launch_g4e4b_lora_1tray.sh",
        "anchor": "${census_cuda_prefix}torchrun --nnodes=1",
        "token": "${census_cuda_prefix}",
        "reason": (
            "an ENV-ASSIGNMENT PREFIX, set at :699/:701 to either '' or"
            " 'CUDA_VISIBLE_DEVICES= ' -- it must word-split into zero or one"
            " assignment word ahead of torchrun, which is what quoting would"
            " break; the launcher contract suite pins this spelling as f41"
        ),
    },
    {
        "file": "launch_g4e4b_lora_1tray.sh",
        "anchor": "--overrides $CLI_OVERRIDES",
        "token": "$CLI_OVERRIDES",
        "reason": (
            "deliberately unquoted per the comment at :1232, so the replay"
            " probe receives the BYTE-IDENTICAL word-split the training"
            " invocation will receive; quoting it would make the probe verify"
            " a different argv than the one that runs"
        ),
    },
)


def _comment_start(line):
    """Index of the `#` that begins a trailing comment, or -1.

    A `#` opens a comment only when it is unquoted AND at start-of-word, which
    is why `bash -lc "x # $V"` is still a site: the `#` there is inside the
    inner source, not a comment, and the expansion after it is a real splice.
    Limitation, stated: quoting is tracked within one physical line, so a
    double-quoted string continued across a backslash-newline can desynchronise
    this. Both launchers keep every `bash -lc` invocation on one physical line
    and the site count is printed, so a desync is visible rather than silent.
    """
    in_s = in_d = False
    i = 0
    while i < len(line):
        c = line[i]
        if c == "\\" and not in_s:
            i += 2
            continue
        if c == "'" and not in_d:
            in_s = not in_s
        elif c == '"' and not in_s:
            in_d = not in_d
        elif c == "#" and not in_s and not in_d and (i == 0 or line[i - 1] in " \t"):
            return i
        i += 1
    return -1


def _inner_source(line, at):
    """Return (kind, body) for the argument after `bash -lc` at index `at`.

    kind is 'nonliteral' (no quote follows), 'single' (single-quoted source) or
    'double' (double-quoted source, body returned with escapes left in place).
    """
    j = at + len("bash -lc")
    while j < len(line) and line[j] in " \t":
        j += 1
    if j >= len(line) or line[j] not in "'\"":
        return "nonliteral", ""
    if line[j] == "'":
        return "single", ""
    k = j + 1
    body = []
    while k < len(line):
        c = line[k]
        if c == "\\":
            k += 2
            continue
        if c == '"':
            break
        body.append(c)
        k += 1
    return "double", "".join(body)


def _cmdsub_span(body, i):
    """Given body[i:i+2] == '$(', return (inner_text, index_of_closing_paren)."""
    depth = 0
    j = i + 1
    while j < len(body):
        if body[j] == "(":
            depth += 1
        elif body[j] == ")":
            depth -= 1
            if depth == 0:
                return body[i + 2 : j], j
        j += 1
    return body[i + 2 :], len(body) - 1


_PCT_Q = re.compile(r"\s*printf\s+(?:(['\"])%q\s*\1|%q)\s")
_VAR = re.compile(r"\$\{?[A-Za-z_][A-Za-z_0-9]*\}?")
_WHOLE_VAR = re.compile(r"^\$\{?[A-Za-z_][A-Za-z_0-9]*\}?$")


def _scan(body):
    """Every expansion in `body` that sits outside inner single quotes.

    Returns a list of (token, kind, narration) with kind in {'var', 'cmdsub',
    'cmdsub_q'}. 'cmdsub_q' is the %q-rendered form and is safe; the other two
    are splices unless the site as a whole is eval-shaped or declared.
    """
    hits = []
    inner_single = False
    idx = 0
    while idx < len(body):
        ch = body[idx]
        if ch == "'":
            inner_single = not inner_single
        elif ch == "$" and not inner_single:
            if body[idx : idx + 2] == "$(":
                sub, close = _cmdsub_span(body, idx)
                if _PCT_Q.match(sub):
                    hits.append(
                        (
                            "$(printf '%q ' ...)",
                            "cmdsub_q",
                            "%q-RENDERED substitution -- printf shell-quotes its"
                            " output, so what reaches the inner source is already"
                            " escaped bytes, not raw value",
                        )
                    )
                else:
                    hits.append(
                        (
                            "$(...)",
                            "cmdsub",
                            "$(...) executes in the OUTER shell and its stdout"
                            " splices into inner SOURCE",
                        )
                    )
                idx = close
            else:
                m = _VAR.match(body, idx)
                if m:
                    hits.append(
                        (
                            m.group(0),
                            "var",
                            f"{m.group(0)} expanded bare -- the outer shell"
                            " splices the value into the inner shell's SOURCE"
                            " text (word-splits; $(...) content executes)",
                        )
                    )
                    idx = m.end() - 1
        idx += 1
    return hits


def classify(files, declarations=DECLARED_SPLICES):
    if len(files) < 2:
        print(
            f"BASH-LC RED: the sweep requires both launchers on argv; got"
            f" {len(files)} -- a partial sweep is UNMEASURED (doctrine 1)"
        )
        return 1
    decls = [{"d": d, "hits": 0} for d in declarations]
    total = 0
    mentions = 0
    unsafe = []
    audited = []
    declared = []
    for fn in files:
        base = Path(fn).name
        try:
            with Path(fn).open(encoding="utf-8") as f:
                lines = f.read().splitlines()
        except OSError as e:
            print(f"BASH-LC RED: unreadable {fn}: {e} -- unreadable is not empty (doctrine 4)")
            return 1
        for n, line in enumerate(lines, 1):
            cs = _comment_start(line)
            pos = 0
            while True:
                at = line.find("bash -lc", pos)
                if at < 0:
                    break
                pos = at + 1
                if 0 <= cs < at:
                    # A mention inside a trailing (or whole-line) comment. Not a
                    # site: it cannot execute, so it must neither redden the
                    # sweep nor pad its denominator (#239 D1).
                    mentions += 1
                    continue
                total += 1
                kind, body = _inner_source(line, at)
                if kind == "nonliteral":
                    audited.append(
                        f"{fn}:{n} [non-literal argument -- enumerated for a human,"
                        " counted as examined]"
                    )
                    continue
                if kind == "single":
                    audited.append(
                        f"{fn}:{n} [single-quoted source; expansions passed as data ($1-style)]"
                    )
                    continue
                hits = _scan(body)
                if not hits:
                    audited.append(
                        f"{fn}:{n} [double-quoted source; every expansion nested"
                        " inside inner single quotes -- the :513/:1002 idiom;"
                        " a single quote in the value would reopen it]"
                    )
                    continue
                if len(hits) == 1 and hits[0][1] == "var" and _WHOLE_VAR.match(body.strip()):
                    audited.append(
                        f"{fn}:{n} [EVAL-SHAPED: the whole inner source is"
                        f" {hits[0][0]} and nothing else -- the value is not"
                        " spliced INTO literal source, it IS the source, which is"
                        " what eval means; whoever builds that variable owns its"
                        " contents absolutely]"
                    )
                    continue
                residual = []
                site_declared = []
                for token, hkind, narration in hits:
                    if hkind == "cmdsub_q":
                        continue
                    match = None
                    for entry in decls:
                        d = entry["d"]
                        if d["file"] == base and d["anchor"] in line and d["token"] == token:
                            match = entry
                            break
                    if match is not None:
                        match["hits"] += 1
                        site_declared.append(f"{token} -- {match['d']['reason']}")
                    else:
                        residual.append(narration)
                if residual:
                    unsafe.append(f"{fn}:{n} {' | '.join(residual)}")
                elif site_declared:
                    declared.append(f"{fn}:{n} {' | '.join(site_declared)}")
                else:
                    audited.append(
                        f"{fn}:{n} [every expansion is a %q-RENDERED substitution"
                        " -- printf emits shell-quoted bytes, which is the"
                        " recommended render rather than the defect]"
                    )
    print(
        f"BASH-LC sweep: examined {total} 'bash -lc' site(s) across"
        f" {len(files)} launcher file(s); {mentions} further occurrence(s) were"
        " comment text and are not sites"
    )
    for s in audited:
        print(f"  audited: {s}")
    for s in declared:
        print(f"  declared: {s}")
    if total == 0:
        print(
            "BASH-LC RED: 0 sites found, but both launchers visibly use 'bash -lc'"
            " -- zero means the sweep broke, i.e. UNMEASURED (doctrine 1)"
        )
        return 1
    part = len(audited) + len(declared) + len(unsafe)
    if part != total:
        print(
            f"BASH-LC RED: classifier partition {part} != {total} examined sites"
            " -- some site fell through every branch, so the verdict describes"
            " fewer sites than it counted (doctrine 2)"
        )
        return 1
    rc = 0
    stale = [e["d"] for e in decls if e["hits"] == 0]
    if stale:
        print(
            f"BASH-LC RED: {len(stale)} DECLARED_SPLICES entr(y|ies) matched"
            " nothing on this tree -- a declaration that no longer describes"
            " the code is an allowlist, and an allowlist that outlives its"
            " subject excuses whatever moves in next (doctrine 5):"
        )
        for d in stale:
            print(f"  RED: stale declaration {d['file']} :: {d['token']} @ '{d['anchor']}'")
        rc = 1
    dupes = [e for e in decls if e["hits"] > 1]
    if dupes:
        print(
            f"BASH-LC RED: {len(dupes)} DECLARED_SPLICES entr(y|ies) matched more"
            " than one site -- a declaration must name one splice, or it is"
            " blessing sites nobody read:"
        )
        for e in dupes:
            print(
                f"  RED: ambiguous declaration {e['d']['file']} ::"
                f" {e['d']['token']} matched {e['hits']} sites"
            )
        rc = 1
    if unsafe:
        print(
            f"BASH-LC RED: {len(unsafe)} unsafe site(s) splice outer-shell"
            " expansions into inner SOURCE text:"
        )
        for u in unsafe:
            print(f"  RED: {u}")
        print('  pass paths as data instead:  bash -lc \'python3 "$1"\' _ "$VAR"')
        rc = 1
    if rc:
        return rc
    print(
        f"BASH-LC ok: {total}/{total} sites classified"
        f" ({len(audited)} audited, {len(declared)} declared, 0 unsafe)"
    )
    return 0


def reinstate(src, dst):
    try:
        with Path(src).open(encoding="utf-8") as f:
            t = f.read()
    except OSError as e:
        print(
            f"BASH-LC MUST_FIRE SETUP RED: unreadable {src}: {e}"
            " -- unreadable is not empty (doctrine 4)"
        )
        return 1
    if t.count(BLOCKER2_FIXED) != 1:
        print(
            f"BASH-LC MUST_FIRE SETUP RED: fixed probe line occurs"
            f" {t.count(BLOCKER2_FIXED)} time(s) in {src}; expected exactly 1"
            " -- cannot isolate the reinstatement; has the fix landed,"
            " or has it been reverted?"
        )
        return 1
    t = t.replace(BLOCKER2_FIXED, BLOCKER2_BROKEN, 1)
    try:
        with Path(dst).open("w", encoding="utf-8") as f:
            f.write(t)
    except OSError as e:
        print(f"BASH-LC MUST_FIRE SETUP RED: cannot write {dst}: {e}")
        return 1
    print(f"BASH-LC MUST_FIRE SETUP ok: unsafe splice reinstated on the copy at {dst}")
    return 0


# ---------------------------------------------------------------------------
# --self-test: the detector's own controls.
#
# Every category this file grew in #239 widens what it will NOT report, and a
# detector that only ever gets wider is on a one-way trip to reporting nothing.
# So each widening ships a MUST_FIRE that proves the neighbouring shape is
# still caught: eval-shaped is exempt but eval-shaped-plus-one-word is not,
# %q-rendered is exempt but a plain substitution is not, comment text is not a
# site but a `#` inside the quoted source is. The MUST_PASSes are the other
# half -- a detector that reddens everything states no verdict either.
# ---------------------------------------------------------------------------

_CLEAN_PARTNER = 'run_in_container bash -lc \'python3 "$1"\' _ "$SCRIPT"\n'


# declarations defaults to EMPTY here, not to DECLARED_SPLICES. The real
# declarations are pinned to the real launchers, so against a synthetic fixture
# they are -- correctly -- stale, and every MUST_PASS went red for a reason
# other than the one it names. That is the staleness rule working, so the fix
# is to scope the fixtures out of it rather than to soften it; the rule keeps
# its own MUST_FIRE below, and the live run is where the real entries must
# still be observed.
def _run_fixture(tmp, name, text_a, text_b=_CLEAN_PARTNER, declarations=()):
    a = Path(tmp) / f"{name}_a.sh"
    b = Path(tmp) / f"{name}_b.sh"
    a.write_text(text_a, encoding="utf-8")
    b.write_text(text_b, encoding="utf-8")
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = classify([str(a), str(b)], declarations=declarations)
    return rc, buf.getvalue()


def self_test():
    fired = []
    held = []
    failures = []

    def must_fire(name, expect_rc, out_needle, rc, out):
        if rc == expect_rc and out_needle in out:
            fired.append(name)
        else:
            failures.append(
                f"MUST_FIRE {name}: expected rc {expect_rc} and"
                f" {out_needle!r} in output; got rc {rc}\n"
                + "\n".join(f"      | {ln}" for ln in out.splitlines())
            )

    def must_pass(name, rc, out, needle=None):
        if rc == 0 and (needle is None or needle in out):
            held.append(name)
        else:
            failures.append(
                f"MUST_PASS {name}: expected rc 0"
                + (f" and {needle!r} in output" if needle else "")
                + f"; got rc {rc}\n"
                + "\n".join(f"      | {ln}" for ln in out.splitlines())
            )

    bogus = (
        {
            "file": "nosuchlauncher.sh",
            "anchor": "no such anchor",
            "token": "$NOPE",
            "reason": "deliberately stale, to prove staleness is fatal",
        },
    )

    with tempfile.TemporaryDirectory() as tmp:
        # ---- MUST_FIRE -----------------------------------------------------
        rc, out = _run_fixture(tmp, "bare", 'bash -lc "python3 $VAR"\n')
        must_fire("bare-splice", 1, "$VAR expanded bare", rc, out)

        rc, out = _run_fixture(tmp, "evalx", 'bash -lc "$CMD --extra"\n')
        must_fire("eval-shaped-plus-one-word-is-still-a-splice", 1, "$CMD expanded bare", rc, out)

        rc, out = _run_fixture(tmp, "sub", 'bash -lc "python3 $(cat f)"\n')
        must_fire("plain-substitution", 1, "$(...) executes in the OUTER shell", rc, out)

        rc, out = _run_fixture(tmp, "hashq", 'bash -lc "python3 x # $VAR"\n')
        must_fire("hash-inside-quotes-is-not-a-comment", 1, "$VAR expanded bare", rc, out)

        rc, out = _run_fixture(tmp, "notq", 'bash -lc "p $(printf \'%s \' "${A[@]}")"\n')
        must_fire("printf-without-%q-is-not-a-render", 1, "$(...) executes", rc, out)

        rc, out = _run_fixture(tmp, "stale", _CLEAN_PARTNER, declarations=bogus)
        must_fire("stale-declaration", 1, "stale declaration", rc, out)

        rc, out = _run_fixture(tmp, "zero", "echo no sites here\n", "echo none here either\n")
        must_fire("zero-sites-is-unmeasured", 1, "0 sites found", rc, out)

        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = classify([str(Path(tmp) / "bare_a.sh")])
        must_fire(
            "single-file-is-a-partial-sweep", 1, "a partial sweep is UNMEASURED", rc, buf.getvalue()
        )

        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = classify([str(Path(tmp) / "does_not_exist.sh"), str(Path(tmp) / "bare_b.sh")])
        must_fire("unreadable-is-not-empty", 1, "unreadable is not empty", rc, buf.getvalue())

        # ---- MUST_PASS -----------------------------------------------------
        rc, out = _run_fixture(tmp, "data", 'bash -lc \'python3 "$1"\' _ "$V"\n')
        must_pass("data-passing-argv", rc, out, "expansions passed as data")

        rc, out = _run_fixture(tmp, "nest", "bash -lc \"cd '$REPO' && python3 '$P'\"\n")
        must_pass("nested-single-quote-idiom", rc, out, ":513/:1002 idiom")

        rc, out = _run_fixture(tmp, "eval", 'bash -lc "$CMD"\n')
        must_pass("eval-shaped-whole-source", rc, out, "EVAL-SHAPED")

        rc, out = _run_fixture(tmp, "pctq", 'bash -lc "p $(printf \'%q \' "${A[@]}")"\n')
        must_pass("%q-rendered-substitution", rc, out, "%q-RENDERED")

        rc, out = _run_fixture(
            tmp,
            "cmt",
            "bash -lc \"cd '$REPO' && x\"\n"
            'echo hi   # historical: this used to be bash -lc "$CMD"\n'
            '# and a whole-line mention of bash -lc "$OTHER"\n',
        )
        must_pass("comment-mentions-are-not-sites", rc, out, "examined 2 'bash -lc' site(s)")
        must_pass("comment-mentions-are-counted-aloud", rc, out, "2 further occurrence(s)")

        decl = (
            {
                "file": "decl_a.sh",
                "anchor": "--overrides $SPLICED",
                "token": "$SPLICED",
                "reason": "fixture declaration",
            },
        )
        rc, out = _run_fixture(
            tmp, "decl", 'bash -lc "python3 p --overrides $SPLICED"\n', declarations=decl
        )
        must_pass("declared-splice-is-reported-not-hidden", rc, out, "  declared: ")

    if failures:
        print(f"BASH-LC SELF-TEST RED: {len(failures)} control(s) misbehaved:")
        for f in failures:
            print(f"  RED: {f}")
        return 1
    if not fired or not held:
        print(
            f"BASH-LC SELF-TEST RED: {len(fired)} MUST_FIRE and {len(held)} MUST_PASS"
            " control(s) ran -- a self-test with an empty arm is all([]), which is"
            " True and means nothing (doctrine 1)"
        )
        return 1
    print(
        f"BASH-LC SELF-TEST ok: {len(fired)} MUST_FIRE control(s) fired and"
        f" {len(held)} MUST_PASS control(s) held"
    )
    for n in fired:
        print(f"  fired: {n}")
    for n in held:
        print(f"  held:  {n}")
    return 0


def main(argv):
    if argv[:1] == ["--self-test"]:
        if len(argv) != 1:
            print("BASH-LC SELF-TEST RED: --self-test takes no further arguments")
            return 1
        return self_test()
    if argv[:1] == ["--reinstate-blocker2"]:
        if len(argv) != 3:
            print("BASH-LC MUST_FIRE SETUP RED: --reinstate-blocker2 takes SRC and DST")
            return 1
        return reinstate(argv[1], argv[2])
    return classify(argv)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
