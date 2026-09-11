# Shared prelude for the launcher-contract and checks-gate suites.
#
# Sourced, never executed: it carries the two verdict primitives (ok/no) and
# the counters they move, so that two suites can report in one vocabulary. It
# is deliberately tiny -- everything a single suite alone needs stays in that
# suite. LDIR resolves from THIS file's directory, which is the same
# launchers/ directory either caller lives in, so the launcher paths below are
# unchanged by the split.
# Positive-control harness for the launcher edits. Every check names what it
# would have caught; a check that cannot fail is not a check.
LDIR=${LDIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}
LORA=$LDIR/launch_g4e4b_lora_1tray.sh
FULL=$LDIR/launch_g4e4b_fullft_1tray.sh
# abstain tallies NAMED abstentions (the fix28 estate battery while the estate
# files are absent, and — in fix30 — the fix28b snippet battery whose 127
# lines were not in the packet): a first-class verdict that states its
# denominator and adds NOTHING to pass or fail. A suite may not green itself
# on checks that never ran (doctrines 1/5), so abstentions print on their own
# line at the end and the frozen "controls:" line stays byte-identical.
pass=0; fail=0; abstain=0
ok(){ printf '  PASS  %s\n' "$1"; pass=$((pass+1)); }
no(){ printf '  FAIL  %s\n' "$1"; fail=$((fail+1)); }

# ---- #376: one resolved interpreter, and one named precondition ------------
#
# Both suites shell out to bare `python3` in ~150 places -- inside heredocs,
# inside fixtures a leg writes at run time, inside the sub-suite one of them
# execs. That binds the verdict to whatever `python3` the caller's PATH
# happened to hold. On this developer's Mac (/usr/bin/python3 is 3.9.6, below
# the >=3.10 floor pyproject.toml declares) the checks suite measured
# "controls: 22 passed, 9 failed": six gate self-tests and three MUST_FIRE
# discrimination legs, each announcing its gate as broken. No gate was broken.
# Nine reds for one unmet precondition is not a measurement -- it is noise
# deep enough to bury whatever the suite was actually run for.
#
# Two moves, in this order:
#
#   1. Resolve ONE interpreter by the rule the Makefile already uses (repo
#      .venv first, else bare python3; FS_SUITE_PY overrides), and put a
#      one-entry shim directory in FRONT of PATH so every bare `python3`
#      binds to it -- including the call sites a leg generates at run time,
#      which no textual rewrite of these files could ever reach. Rewriting
#      the ~150 literal sites was the alternative and is worse on both axes:
#      its failure mode is a missed site that quietly keeps the old
#      interpreter, and many of those sites ASSERT about the literal string
#      `python3` rather than calling it. The shim holds `python3` and nothing
#      else -- prepending .venv/bin itself would also shadow ruff, mypy,
#      pytest and torchrun for every leg, a far larger change than the one
#      being made -- and it is an exec wrapper rather than a symlink so the
#      interpreter reports the venv it actually belongs to.
#      Note the direction of travel: this makes the verdict LESS
#      environment-dependent, not more (#83/#111/#229). The suite now names
#      the interpreter it ran under instead of silently inheriting one.
#
#   2. Measure that interpreter against the floor. Below it, print ONE named
#      abstention and exit 95 (UNMEASURED) without running a single leg. A
#      suite that cannot run its gates has not found them defective, and
#      saying so in one line is the whole finding. Exit 5 would claim the
#      gates are broken; exit 0 would be a green over controls that never
#      ran. No "controls:" line is printed on that path, because the frozen
#      summary line is a report from a suite that RAN -- "0 passed, 0 failed"
#      would be a claim about the gates, and this is a claim about the host.
#
# The shim path is deterministic rather than `mktemp -d` because a
# prelude-level EXIT trap could not survive: test_launcher_contracts.sh:383
# installs its own sandbox trap and bash keeps exactly one EXIT trap per
# shell, so the cleanup would be silently dropped. A fixed per-uid path is
# rewritten in place by the next run instead of accumulating.
FS_SUITE_PY_FLOOR=${FS_SUITE_PY_FLOOR:-3.10}
if [ -z "${FS_SUITE_PY:-}" ]; then
  if [ -x "$LDIR/../.venv/bin/python3" ]; then
    FS_SUITE_PY=$(cd "$LDIR/.." && pwd)/.venv/bin/python3
  else
    FS_SUITE_PY=$(command -v python3 2>/dev/null)
  fi
fi
_fs_py_state=OK
_fs_py_ver=
if [ -z "$FS_SUITE_PY" ] || [ ! -x "$FS_SUITE_PY" ]; then
  _fs_py_state=ABSENT
else
  _fs_py_ver=$("$FS_SUITE_PY" -c 'import sys; print("%d.%d.%d" % sys.version_info[:3])' 2>/dev/null)
  "$FS_SUITE_PY" -c 'import sys
floor = tuple(int(p) for p in sys.argv[1].split("."))
raise SystemExit(0 if sys.version_info[:len(floor)] >= floor else 1)' "$FS_SUITE_PY_FLOOR" 2>/dev/null \
    || _fs_py_state=BELOW_FLOOR
fi
if [ "$_fs_py_state" = OK ]; then
  # Only intervene when the resolved interpreter is NOT already what a bare
  # `python3` resolves to. On a machine with no repo venv the two are the same
  # and prepending anything would shadow that machine's other tools for no gain.
  if [ "$(command -v python3 2>/dev/null)" != "$FS_SUITE_PY" ]; then
    FS_SUITE_PY_SHIM=${TMPDIR:-/tmp}/fs-suite-py.$(id -u)
    if mkdir -p "$FS_SUITE_PY_SHIM" 2>/dev/null \
       && printf '#!/bin/sh\nexec "%s" "$@"\n' "$FS_SUITE_PY" > "$FS_SUITE_PY_SHIM/python3" 2>/dev/null \
       && chmod +x "$FS_SUITE_PY_SHIM/python3" 2>/dev/null; then
      PATH=$FS_SUITE_PY_SHIM:$PATH
    else
      _fs_py_state=SHIM_FAILED
    fi
  fi
fi
if [ "$_fs_py_state" != OK ]; then
  case $_fs_py_state in
    ABSENT)
      printf '  ABSTAIN  interpreter precondition: no executable python3 resolved (FS_SUITE_PY=%s)\n' \
        "${FS_SUITE_PY:-<empty>}" ;;
    BELOW_FLOOR)
      printf '  ABSTAIN  interpreter precondition: %s is %s, below the >=%s floor pyproject.toml declares\n' \
        "$FS_SUITE_PY" "${_fs_py_ver:-unknown}" "$FS_SUITE_PY_FLOOR" ;;
    SHIM_FAILED)
      printf '  ABSTAIN  interpreter precondition: resolved %s but could not bind it (shim dir %s not writable)\n' \
        "$FS_SUITE_PY" "${FS_SUITE_PY_SHIM:-<unset>}" ;;
  esac
  printf '           remedy: python3 -m venv .venv && make install, or set FS_SUITE_PY to a >=%s interpreter\n' "$FS_SUITE_PY_FLOOR"
  printf 'UNMEASURED (95): 0 of this suite'"'"'s controls ran -- an unmet host precondition, not a gate verdict\n'
  printf 'abstentions: 1 named\n'
  exit 95
fi
export PATH FS_SUITE_PY
printf 'interpreter: %s (%s), floor >=%s\n' "$FS_SUITE_PY" "${_fs_py_ver:-unknown}" "$FS_SUITE_PY_FLOOR"

# ---- #377: each suite anchors its own control count ------------------------
#
# Three developer-facing documents stated 146 and 27 controls while the suites
# ran 147 and 29, in eight places, for months. `make countables` was CLEAR the
# whole time, because this countable is in no census denominator -- and it
# cannot be put in one the way the others are. The census is static, and this
# number is not derivable statically: a control is one if/else whose single
# `ok` branch competes with several `no` branches, so verdict SITES outnumber
# verdicts by 5x (150 sites for 29 controls in the checks suite), and the
# launcher suite folds a sub-suite's tally in bulk rather than one verdict at
# a time. A static `ok`-site count reads 144 there against a measured 147. The
# only instrument that knows the number is the suite itself, while running, so
# the anchor lives here rather than in tools/countables_census.py.
#
# Wall-clock time is deliberately NOT anchored, though the same documents
# state it. The launcher suite measured 28.9s idle and 63.6s under concurrent
# load on one machine at one commit, with user time unmoved at ~4.1s. A
# countable whose value is set by what else the machine is doing manufactures
# reds that say nothing about the code; those figures are published as dated
# observations instead, and docs/TESTING.md says so at the site.
CONTROL_DOC_FILES=${CONTROL_DOC_FILES:-"docs/TESTING.md docs/GETTING_STARTED.md docs/DEVELOPMENT.md"}

# stdout: one row per "N controls" claim in the developer docs, as
# `file:line<TAB>suite-or-NONE<TAB>N`. The suite is recognised by its basename
# appearing ON THE SAME LINE, and by nothing else. Association is where this
# class of gate fails -- #233 inferred which countable a bare number claimed
# and produced 865 false drifts -- so a claim that does not name its subject
# is reported as unattributed rather than guessed at.
#
# The scope is those three files and no others. "8/8 controls" appears 17
# times in the campaign reports about entirely different denominators, so a
# repo-wide `*.md` denominator would redden a dozen true statements.
_control_claims(){
  local root f
  root=$(cd "$LDIR/.." && pwd)
  for f in $CONTROL_DOC_FILES; do
    [ -f "$root/$f" ] || { printf 'MISSING\t%s\t-\n' "$f"; continue; }
    awk -v name="$f" '
      {
        t = $0
        while (match(t, /[0-9]+ controls/)) {
          m = substr(t, RSTART, RLENGTH); sub(/ controls/, "", m)
          s = "NONE"
          if (index($0, "test_launcher_contracts.sh")) s = "test_launcher_contracts.sh"
          else if (index($0, "test_checks_gates.sh"))  s = "test_checks_gates.sh"
          printf "%s:%d\t%s\t%s\n", name, FNR, s, m
          t = substr(t, RSTART + RLENGTH)
        }
      }' "$root/$f"
  done
}

# One control. Every "N controls" claim in the three developer docs must name
# the suite it describes on its own line; one that names neither suite is a
# number a maintainer cannot check and a future edit cannot be held to.
assert_control_claims_attributed(){
  local rows orphans missing
  rows=$(_control_claims)
  missing=$(printf '%s\n' "$rows" | awk -F'\t' '$1=="MISSING"{printf "%s ", $2}')
  if [ -n "$missing" ]; then
    no "control-count attribution: ${missing% } is named in CONTROL_DOC_FILES and does not exist, so the denominator is short and any CLEAR from this control would be over a set smaller than the one declared"
    return
  fi
  orphans=$(printf '%s\n' "$rows" | awk -F'\t' '$2=="NONE"{printf "%s ", $1}')
  if [ -n "$orphans" ]; then
    no "control-count attribution: ${orphans% } state a control count without naming test_launcher_contracts.sh or test_checks_gates.sh on the same line -- an unattributed countable is one no gate can hold to a measurement"
  else
    ok "control-count attribution: every control count in ${CONTROL_DOC_FILES// /, } names its suite on the same line, so each claim has an owner that can be measured"
  fi
}

# One control. $1 = this suite's basename; the total compared is the number of
# controls THIS RUN adjudicated, counting this one. pass+fail is used rather
# than pass alone because it does not move when a load-sensitive leg goes red
# (#93): the suite runs the same controls either way, and the documented
# figure is a denominator, not a score.
assert_documented_control_total(){
  local suite=$1 total claims bad
  total=$((pass + fail + 1))
  claims=$(_control_claims | awk -F'\t' -v s="$suite" '$2==s{print $1"="$3}')
  if [ -z "$claims" ]; then
    no "documented control total ($suite): the three developer docs state a count for this suite in 0 places, so its denominator is published nowhere and a change to it is invisible"
    return
  fi
  bad=$(printf '%s\n' "$claims" | awk -F'=' -v t="$total" '$2!=t{printf "%s ", $0}')
  if [ -n "$bad" ]; then
    no "documented control total ($suite): this run adjudicated $total controls; the docs say ${bad% }. Update every site, or -- if you enabled an env-gated battery such as FIX28_ESTATE_GEMMA4_VL -- note that the published figure describes the default environment and your denominator is legitimately larger"
  else
    ok "documented control total ($suite): $total adjudicated this run, and all $(printf '%s\n' "$claims" | wc -l | tr -d ' ') published statements agree -- the count that #377 left stale in eight places is now checked by the only instrument that can measure it"
  fi
}

