#!/usr/bin/env bash
# Positive-control harness for the checks/*.py gates. Every check names what it
# would have caught; a check that cannot fail is not a check.
#
# Split out of test_launcher_contracts.sh (finding #257): these legs certify
# the repository's own gate scripts, not the launchers, and appending each new
# gate's self-test to a file named for the launchers is what made that file a
# 5,562-line outlier. A new checks/*.py gate belongs here.
#
# Every gate is certified in a PAIR: MUST_PASS (the gate clears a tree it must
# not redden) and MUST_FIRE (the gate reddens a planted defect). A gate with
# only the first half is not proven able to refuse.
# shellcheck source=launchers/_suite_prelude.sh
. "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_suite_prelude.sh"
# --- MUST_PASS: workflow YAML audit (checks/wf_yaml_audit.py) ----------------
# The auditor's own MUST_FIRE lives in .github/workflows/ci.yml (the
# doctor-blocker1 rig), so THIS leg is the only workflow-parse measurement on
# this wire and its denominator goes ON the wire: the auditor script must be
# readable (unreadable is not empty, doctrine 4), a glob that stays literal
# means zero files were handed over (UNMEASURED, never PASS -- missing is not
# zero, doctrines 1/4), the auditor's rc must be 0 over every *.yml handed to
# it, and the auditor's OWN "examined N" count must equal the number of files
# the glob handed it -- the claim counts what the auditor examined, not what
# the shell globbed (doctrine 2), so an auditor that silently under-scans its
# argv goes red here. Files travel via "$@" (space-safe); the scope is exactly
# .github/workflows/*.yml -- nothing wider is claimed than was measured
# (doctrine 5, symmetric).
if [ ! -r "checks/wf_yaml_audit.py" ]; then
  wfy_msg="MUST_PASS FAILED (workflow YAML audit) UNMEASURED:"
  wfy_msg="$wfy_msg checks/wf_yaml_audit.py is not readable -- unreadable is not empty"
  wfy_msg="$wfy_msg (doctrine 4); the auditor cannot run, so nothing was examined"
  no "$wfy_msg"
else
  set -- .github/workflows/*.yml
  if [ ! -f "$1" ]; then
    wfy_msg="MUST_PASS FAILED (workflow YAML audit) UNMEASURED: the glob"
    wfy_msg="$wfy_msg .github/workflows/*.yml stayed literal -- 0 files handed to the"
    wfy_msg="$wfy_msg auditor; missing is not zero, UNMEASURED is never PASS (doctrines 1/4)"
    no "$wfy_msg"
  else
    wfy_want=$#
    wfy_rc=0
    wfy_out=$(python3 checks/wf_yaml_audit.py "$@" 2>&1) || wfy_rc=$?
    wfy_n=$(printf '%s\n' "$wfy_out" |
      sed -n 's/^WF-YAML ok: examined \([0-9][0-9]*\) workflow file(s);.*/\1/p' | head -n 1)
    wfy_mode=$(printf '%s\n' "$wfy_out" |
      sed -n 's/^WF-YAML ok: .*accepted by //p' | head -n 1)
    if [ "$wfy_rc" -ne 0 ]; then
      wfy_msg="MUST_PASS FAILED (workflow YAML audit): auditor rc=$wfy_rc over"
      wfy_msg="$wfy_msg $wfy_want handed .github/workflows/*.yml files -- auditor output:"
      wfy_msg="$wfy_msg $(printf '%s\n' "$wfy_out" | tr '\n' ' ')"
      no "$wfy_msg"
    elif [ -z "$wfy_n" ]; then
      wfy_msg="MUST_PASS FAILED (workflow YAML audit) UNMEASURED: rc=0 but the auditor"
      wfy_msg="$wfy_msg printed no 'WF-YAML ok: examined N' denominator line over $wfy_want"
      wfy_msg="$wfy_msg handed files -- the measuring unit printed no denominator"
      wfy_msg="$wfy_msg (doctrine 2); output: $(printf '%s\n' "$wfy_out" | tr '\n' ' ')"
      no "$wfy_msg"
    elif [ "$wfy_n" -ne "$wfy_want" ]; then
      wfy_msg="MUST_PASS FAILED (workflow YAML audit): auditor examined=$wfy_n but was"
      wfy_msg="$wfy_msg handed $wfy_want .github/workflows/*.yml files -- a silent"
      wfy_msg="$wfy_msg under-scan; the claim counts what the auditor examined (doctrine 2)"
      no "$wfy_msg"
    else
      wfy_msg="MUST_PASS workflow YAML audit: checks/wf_yaml_audit.py examined $wfy_n of"
      wfy_msg="$wfy_msg $wfy_want handed .github/workflows/*.yml files, rc=0, and its own"
      wfy_msg="$wfy_msg denominator matches the wire -- accepted by ${wfy_mode:-unknown mode}"
      ok "$wfy_msg"
    fi
  fi
fi

# --- MUST_FIRE (workflow YAML audit): a freshly wired detector that has never
# been observed firing is not a control (doctrine 3). Plant a malformed
# workflow in a temp dir and demand the auditor go red ON IT by name. The
# payload over-dedents out of a '|' block scalar and then re-indents, which
# is red under BOTH of the auditor's code paths -- a PyYAML parse error where
# PyYAML is importable, and the structural fallback's under-cut shape where
# it is not -- because a fire that only one parser mode can see never happens
# on hosts running the other mode. rc!=0 WITHOUT the auditor's own WF-YAML
# RED line (e.g. a Python traceback, which also exits 1) does NOT count as
# firing -- a crashed detector is not a discriminating one -- and a plant
# failure is UNREACHABLE-red, never green.
wfy_bad_dir=$(mktemp -d "${TMPDIR:-/tmp}/fs-wf-yaml-bad.XXXXXX")
wfy_bad_rc=99
wfy_bad_out="(auditor never ran -- plant failed)"
if [ -d "$wfy_bad_dir" ]; then
  wfy_bad_file=$wfy_bad_dir/planted_bad.yml
  {
    printf 'jobs:\n  build:\n    steps:\n      - run: |\n'
    printf '          echo planted\n'
    printf 'bad-column-zero-continuation\n'
    printf '          echo reindented\n'
  } > "$wfy_bad_file"
  if [ -s "$wfy_bad_file" ]; then
    wfy_bad_rc=0
    wfy_bad_out=$(python3 checks/wf_yaml_audit.py "$wfy_bad_file" 2>&1) || wfy_bad_rc=$?
  fi
fi
[ -n "$wfy_bad_dir" ] && rm -rf "$wfy_bad_dir" || true
if [ "$wfy_bad_rc" -eq 99 ]; then
  wfy_msg="MUST_FIRE UNREACHABLE (workflow YAML audit): could not plant the malformed"
  wfy_msg="$wfy_msg workflow (temp dir or planted file unusable) -- a MUST_FIRE that cannot"
  wfy_msg="$wfy_msg plant its mutation is UNREACHABLE-red, never green"
  no "$wfy_msg"
elif [ "$wfy_bad_rc" -eq 0 ]; then
  wfy_msg="MUST_FIRE UNREACHABLE (workflow YAML audit): the planted malformed workflow"
  wfy_msg="$wfy_msg came back rc=0 -- the auditor does not discriminate, so wiring it above"
  wfy_msg="$wfy_msg changed nothing (output: $(printf '%s\n' "$wfy_bad_out" | tr '\n' ' '))"
  no "$wfy_msg"
elif ! printf '%s\n' "$wfy_bad_out" | grep -q 'WF-YAML RED'; then
  wfy_msg="MUST_FIRE UNREACHABLE (workflow YAML audit): planted file rc=$wfy_bad_rc but"
  wfy_msg="$wfy_msg the auditor never indicted it by name (no 'WF-YAML RED' line) --"
  wfy_msg="$wfy_msg red from a crash is not discrimination; output:"
  wfy_msg="$wfy_msg $(printf '%s\n' "$wfy_bad_out" | tr '\n' ' ')"
  no "$wfy_msg"
else
  wfy_msg="MUST_FIRE workflow YAML audit: planted malformed workflow (over-dedent out of"
  wfy_msg="$wfy_msg a '|' block scalar -- refused by PyYAML AND by the structural fallback)"
  wfy_msg="$wfy_msg was indicted by name (rc=$wfy_bad_rc):"
  wfy_msg="$wfy_msg $(printf '%s\n' "$wfy_bad_out" | grep -m1 'WF-YAML RED')"
  ok "$wfy_msg"
fi

echo "== fix238-gatewiring: countables_drift + packaging_reachability real legs =="

# --- finding #238: two gate files (checks/countables_drift.py,
# checks/packaging_reachability.py) enter the repo in the same commit as
# these legs. The anti-orphan gate scans every launchers/*.py + checks/*.py
# and refuses any basename with no word-boundary call site in this suite, so
# both would be indicted as orphans on arrival. A comment-mention would
# satisfy the grep and measure NOTHING, so what follows are real legs that
# execute both gates and put their denominators on the wire.
#
# ENVIRONMENT (measured, not assumed): the CI job that runs this suite
# installs NOTHING -- no setup-python, no pip -- so every invocation below is
# the runner's bare system `python3` (both gates are stdlib-only, verified on
# Python 3.9.6), and every invocation carries `-S`. The `-S` is load-bearing,
# not hygiene: packaging_reachability's verdict otherwise depends on whether
# foundationscale happens to be pip-installed in the ambient environment
# (MEASURED: rc=0 installed, rc=95 not), so the same suite text would go
# green on one runner and abstain on another -- an environment-dependent
# verdict, which is this repo's #83/#229 defect class. `-S` drops
# site-packages and forces the not-installed condition in BOTH environments,
# so the verdict below is a property of the gate, not of the host.

# --- MUST_PASS: countables_drift self-test (checks/countables_drift.py) -----
# MEASURED: `python3 -S checks/countables_drift.py --self-test` exits rc=0
# and its last line is exactly
#   self-test denominator: 17 of 17 controls (10 MUST_FIRE, 7 MUST_PASS)
# rc=0 alone is NOT the measurement: a self-test whose control set silently
# shrinks to 1 still exits 0, so the trailing "N of N controls" is parsed and
# held to a FLOOR of N >= 17 -- the claim counts what the self-test examined
# (doctrine 2), and a shrunken control set goes red here. If the wording of
# that line ever changes, THIS leg goes red and must be updated in the same
# commit -- unreadable is not empty, and unparseable is not passing.
#
# Floor history: 8 -> 14 (#243) -> 17 (#244, the three legs that check the
# census/gate exclusion handshake). A floor left at its historical value while
# the real count grows is not conservative, it is that many controls the leg
# would let disappear in silence.
if [ ! -r "checks/countables_drift.py" ]; then
  f238_msg="MUST_PASS FAILED (countables_drift self-test) UNMEASURED:"
  f238_msg="$f238_msg checks/countables_drift.py is not readable -- unreadable is not"
  f238_msg="$f238_msg empty (doctrine 4); the gate cannot run, so 0 of 17 controls were measured"
  no "$f238_msg"
else
  f238_rc=0
  f238_out=$(python3 -S checks/countables_drift.py --self-test 2>&1) || f238_rc=$?
  f238_last=$(printf '%s\n' "$f238_out" | tail -n 1)
  f238_have=$(printf '%s\n' "$f238_last" |
    sed -n 's/^self-test denominator: \([0-9][0-9]*\) of \([0-9][0-9]*\) controls.*/\1/p')
  f238_want=$(printf '%s\n' "$f238_last" |
    sed -n 's/^self-test denominator: \([0-9][0-9]*\) of \([0-9][0-9]*\) controls.*/\2/p')
  if [ "$f238_rc" -ne 0 ]; then
    f238_msg="MUST_PASS FAILED (countables_drift self-test): rc=$f238_rc over the gate's"
    f238_msg="$f238_msg own 17-control fixture set -- output:"
    f238_msg="$f238_msg $(printf '%s\n' "$f238_out" | tr '\n' ' ')"
    no "$f238_msg"
  elif [ -z "$f238_have" ] || [ -z "$f238_want" ]; then
    f238_msg="MUST_PASS FAILED (countables_drift self-test) UNMEASURED: rc=0 but the last"
    f238_msg="$f238_msg line is not the declared 'self-test denominator: N of N controls'"
    f238_msg="$f238_msg wording -- the measuring unit printed no denominator (doctrine 2);"
    f238_msg="$f238_msg update this leg in the same commit as the wording change."
    f238_msg="$f238_msg Last line: $f238_last"
    no "$f238_msg"
  elif [ "$f238_have" -ne "$f238_want" ]; then
    f238_msg="MUST_PASS FAILED (countables_drift self-test): denominator $f238_have of"
    f238_msg="$f238_msg $f238_want controls is not self-consistent -- the self-test examined"
    f238_msg="$f238_msg fewer controls than it claims to have (doctrine 2)"
    no "$f238_msg"
  elif [ "$f238_have" -lt 17 ]; then
    f238_msg="MUST_PASS FAILED (countables_drift self-test): control set shrank to"
    f238_msg="$f238_msg $f238_have of $f238_want, below the measured floor of 17 -- a self-test"
    f238_msg="$f238_msg that quietly drops controls still exits 0, so the floor is the control"
    no "$f238_msg"
  else
    f238_msg="MUST_PASS countables_drift self-test: rc=0 under python3 -S, denominator"
    f238_msg="$f238_msg $f238_have of $f238_want controls (>= the measured floor of 17): $f238_last"
    ok "$f238_msg"
  fi
fi

# --- MUST_FIRE: countables_drift refuses an empty denominator ----------------
# MEASURED: `python3 -S checks/countables_drift.py` with NO path arguments at
# all exits rc=96 exactly -- the declared REFUSE code. The assertion is 96,
# not merely nonzero: collapsing it to nonzero would accept a crash (rc=1/2)
# as a control firing, and a crashed detector is not a discriminating one.
# This is doctrine 1 pinned as a control: all([]) is True, so a gate asked to
# certify a corpus it never read (0 of 0 units) must REFUSE -- zero units is
# UNMEASURED, never PASS.
if [ ! -r "checks/countables_drift.py" ]; then
  f238_msg="MUST_FIRE UNREACHABLE (countables_drift empty-denominator refusal)"
  f238_msg="$f238_msg UNMEASURED: checks/countables_drift.py is not readable -- unreadable"
  f238_msg="$f238_msg is not empty (doctrine 4); the refusal cannot be exercised, 0 of 1"
  f238_msg="$f238_msg refusal paths measured"
  no "$f238_msg"
else
  f238_rc=0
  f238_out=$(python3 -S checks/countables_drift.py 2>&1) || f238_rc=$?
  if [ "$f238_rc" -eq 96 ]; then
    f238_msg="MUST_FIRE countables_drift: invoked over 0 of 0 path arguments it refused"
    f238_msg="$f238_msg with rc=96 (REFUSE), declining to certify a corpus it never read --"
    f238_msg="$f238_msg zero units is UNMEASURED, never PASS (doctrine 1)"
    ok "$f238_msg"
  else
    f238_msg="MUST_FIRE UNREACHABLE (countables_drift empty-denominator refusal): rc=$f238_rc"
    f238_msg="$f238_msg over 0 path arguments, expected exactly 96 -- rc=0 would launder an"
    f238_msg="$f238_msg empty corpus into a PASS (doctrine 1), and any other nonzero collapses"
    f238_msg="$f238_msg a crash into a control firing; output:"
    f238_msg="$f238_msg $(printf '%s\n' "$f238_out" | tr '\n' ' ')"
    no "$f238_msg"
  fi
fi

# --- MUST_PASS: packaging_reachability self-test -----------------------------
# MEASURED: `python3 -S checks/packaging_reachability.py --self-test` exits
# rc=0 and its last line is exactly
#   SELF-TEST DENOMINATOR: 7 of 7 -- 5x MUST_FIRE produced nonzero finding counts; 2x MUST_PASS stayed clean over a nonzero denominator
# Same floor reasoning as countables_drift: rc=0 survives a control set that
# silently shrinks to 1, so "N of N" is parsed and held to N >= 7. A wording
# change reds THIS leg and is updated in the same commit.
if [ ! -r "checks/packaging_reachability.py" ]; then
  f238_msg="MUST_PASS FAILED (packaging_reachability self-test) UNMEASURED:"
  f238_msg="$f238_msg checks/packaging_reachability.py is not readable -- unreadable is not"
  f238_msg="$f238_msg empty (doctrine 4); the gate cannot run, so 0 of 7 controls were measured"
  no "$f238_msg"
else
  f238_rc=0
  f238_out=$(python3 -S checks/packaging_reachability.py --self-test 2>&1) || f238_rc=$?
  f238_last=$(printf '%s\n' "$f238_out" | tail -n 1)
  f238_have=$(printf '%s\n' "$f238_last" |
    sed -n 's/^SELF-TEST DENOMINATOR: \([0-9][0-9]*\) of \([0-9][0-9]*\) .*/\1/p')
  f238_want=$(printf '%s\n' "$f238_last" |
    sed -n 's/^SELF-TEST DENOMINATOR: \([0-9][0-9]*\) of \([0-9][0-9]*\) .*/\2/p')
  if [ "$f238_rc" -ne 0 ]; then
    f238_msg="MUST_PASS FAILED (packaging_reachability self-test): rc=$f238_rc over the"
    f238_msg="$f238_msg gate's own 7-control fixture set -- output:"
    f238_msg="$f238_msg $(printf '%s\n' "$f238_out" | tr '\n' ' ')"
    no "$f238_msg"
  elif [ -z "$f238_have" ] || [ -z "$f238_want" ]; then
    f238_msg="MUST_PASS FAILED (packaging_reachability self-test) UNMEASURED: rc=0 but the"
    f238_msg="$f238_msg last line is not the declared 'SELF-TEST DENOMINATOR: N of N' wording"
    f238_msg="$f238_msg -- the measuring unit printed no denominator (doctrine 2); update this"
    f238_msg="$f238_msg leg in the same commit as the wording change. Last line: $f238_last"
    no "$f238_msg"
  elif [ "$f238_have" -ne "$f238_want" ]; then
    f238_msg="MUST_PASS FAILED (packaging_reachability self-test): denominator $f238_have"
    f238_msg="$f238_msg of $f238_want controls is not self-consistent -- the self-test examined"
    f238_msg="$f238_msg fewer controls than it claims to have (doctrine 2)"
    no "$f238_msg"
  elif [ "$f238_have" -lt 7 ]; then
    f238_msg="MUST_PASS FAILED (packaging_reachability self-test): control set shrank to"
    f238_msg="$f238_msg $f238_have of $f238_want, below the measured floor of 7 -- a self-test"
    f238_msg="$f238_msg that quietly drops controls still exits 0, so the floor is the control"
    no "$f238_msg"
  else
    f238_msg="MUST_PASS packaging_reachability self-test: rc=0 under python3 -S,"
    f238_msg="$f238_msg denominator $f238_have of $f238_want controls (>= the measured floor"
    f238_msg="$f238_msg of 7): $f238_last"
    ok "$f238_msg"
  fi
fi

# --- MUST_FIRE: packaging_reachability declares abstention -------------------
# MEASURED: `python3 -S checks/packaging_reachability.py` (foundationscale
# forced not-installed by -S, see the banner comment) exits rc=95 exactly --
# the declared UNMEASURED code -- and its last line begins
#   UNMEASURED: distribution 'foundationscale' is not installed
# This is finding #56's rule pinned as a control: an abstaining gate must
# publish a DECLARED abstention state, never exit 0 -- rc=0 here would
# launder an unmeasured distribution into a pass. The reason text is required
# alongside the code so that a future rc=95 raised for an unrelated cause
# cannot read as THIS control firing, and any other nonzero (a crash, rc=1/2)
# is not discrimination.
if [ ! -r "checks/packaging_reachability.py" ]; then
  f238_msg="MUST_FIRE UNREACHABLE (packaging_reachability declared abstention)"
  f238_msg="$f238_msg UNMEASURED: checks/packaging_reachability.py is not readable --"
  f238_msg="$f238_msg unreadable is not empty (doctrine 4); the abstention cannot be"
  f238_msg="$f238_msg exercised, 0 of 1 abstention paths measured"
  no "$f238_msg"
else
  f238_rc=0
  f238_out=$(python3 -S checks/packaging_reachability.py 2>&1) || f238_rc=$?
  f238_last=$(printf '%s\n' "$f238_out" | tail -n 1)
  if [ "$f238_rc" -ne 95 ]; then
    f238_msg="MUST_FIRE UNREACHABLE (packaging_reachability declared abstention):"
    f238_msg="$f238_msg rc=$f238_rc with foundationscale forced not-installed under -S,"
    f238_msg="$f238_msg expected exactly 95 -- rc=0 would launder the abstention into a pass"
    f238_msg="$f238_msg (#56), any other nonzero is a crash, not a declared state; output:"
    f238_msg="$f238_msg $(printf '%s\n' "$f238_out" | tr '\n' ' ')"
    no "$f238_msg"
  elif ! printf '%s\n' "$f238_last" |
      grep -q "^UNMEASURED: distribution 'foundationscale' is not installed"; then
    f238_msg="MUST_FIRE UNREACHABLE (packaging_reachability declared abstention): rc=95"
    f238_msg="$f238_msg but the last line does not name the not-installed reason -- an"
    f238_msg="$f238_msg rc=95 raised for an unrelated cause must not read as this control"
    f238_msg="$f238_msg firing (#56). Last line: $f238_last"
    no "$f238_msg"
  else
    f238_msg="MUST_FIRE packaging_reachability: 0 of 1 distributions (foundationscale)"
    f238_msg="$f238_msg installed under -S, and the gate published a DECLARED abstention --"
    f238_msg="$f238_msg rc=95 with reason, never exit 0 (#56): $f238_last"
    ok "$f238_msg"
  fi
fi

# --- named abstention: the CLEAR direction of packaging_reachability ---------
# Under `-S` site-packages is dropped on EVERY host, so on this wire the
# packaging gate can only ever be measured in its not-installed direction
# (leg above). Its CLEAR direction -- rc=0 with foundationscale genuinely
# installed and importable -- is unreachable here BY CONSTRUCTION, and an
# unreachable direction is a zero-run denominator, which doctrine 1 says must
# be recorded by name so it can never read as coverage. `make packaging`
# covers the installed direction in an installed environment. Recorded as 0
# of 1 directions; adds 0 to pass and 0 to fail.
printf '  ABSTAIN  fix238-gatewiring: packaging_reachability CLEAR direction (rc=0 with foundationscale installed and importable) — 0 of 1 directions measurable under python3 -S on this wire; `make packaging` covers the installed direction in an installed environment; adds 0 to pass and 0 to fail\n'
abstain=$((abstain+1))

echo "== fix245-trainingplane: training_plane_probe real legs =="

# --- finding #245: checks/training_plane_probe.py enters the repo in the same
# commit as these legs, for the #238 reason -- the anti-orphan gate below scans
# every checks/*.py and refuses any basename with no word-boundary call site in
# this suite. A comment-mention would satisfy that grep and measure NOTHING.
#
# WHAT THE PROBE IS FOR. Four review documents asserted that the package
# "contains no training code". That sentence came from an ad-hoc campaign
# command that looked for six training PRIMITIVES at module scope; the package
# DELEGATES to transformers.Trainer with function-scope imports, so all six
# read zero over a tree that demonstrably trains. The zero was literally true
# and materially misleading, and no committed instrument existed to go red when
# `train/` landed. The probe reports two INDEPENDENT axes -- primitives (A) and
# delegation (B) -- plus a doc axis (C) that hunts the retired phrasings.
#
# Every invocation below carries `python3 -S`, same as fix238 and for the same
# reason: the probe is stdlib-only, and `-S` makes the verdict a property of
# the gate rather than of whatever happens to be installed on the runner
# (#83/#229 class).

# --- MUST_PASS: training_plane_probe self-test -------------------------------
# MEASURED: `python3 -S checks/training_plane_probe.py --self-test` exits rc=0
# and its last line is exactly
#   self-test denominator: 9 of 9 controls (5 MUST_FIRE, 4 MUST_PASS)
# -- deliberately the same wording checks/countables_drift.py prints, parsed by
# the same sed expression. rc=0 alone is NOT the measurement: a self-test whose
# control set silently shrinks to 1 still exits 0, so the trailing "N of N" is
# parsed and held to a FLOOR of 9.
#
# Floor history: 9 at birth (#245). Two of those nine exist because the probe
# reproduced, one level up, the very defect it was written to catch: control 8
# pins that ZERO DOCS SCANNED is UNMEASURED rather than inheriting the source
# axes' clean reading, and control 9 pins that the doc denominator reaches the
# repo ROOT -- README.md is 1 of the 8 tracked *.md that a docs/-only scan
# cannot see, and it is the file deliverable 9 rewrites with exactly this
# claim. A floor left behind while the real count grows is not conservative,
# it is that many controls the leg would let disappear in silence.
if [ ! -r "checks/training_plane_probe.py" ]; then
  f245_msg="MUST_PASS FAILED (training_plane_probe self-test) UNMEASURED:"
  f245_msg="$f245_msg checks/training_plane_probe.py is not readable -- unreadable is not"
  f245_msg="$f245_msg empty (doctrine 4); the gate cannot run, so 0 of 9 controls were measured"
  no "$f245_msg"
else
  f245_rc=0
  f245_out=$(python3 -S checks/training_plane_probe.py --self-test 2>&1) || f245_rc=$?
  f245_last=$(printf '%s\n' "$f245_out" | tail -n 1)
  f245_have=$(printf '%s\n' "$f245_last" |
    sed -n 's/^self-test denominator: \([0-9][0-9]*\) of \([0-9][0-9]*\) controls.*/\1/p')
  f245_want=$(printf '%s\n' "$f245_last" |
    sed -n 's/^self-test denominator: \([0-9][0-9]*\) of \([0-9][0-9]*\) controls.*/\2/p')
  if [ "$f245_rc" -ne 0 ]; then
    f245_msg="MUST_PASS FAILED (training_plane_probe self-test): rc=$f245_rc over the gate's"
    f245_msg="$f245_msg own 9-control fixture set -- output:"
    f245_msg="$f245_msg $(printf '%s\n' "$f245_out" | tr '\n' ' ')"
    no "$f245_msg"
  elif [ -z "$f245_have" ] || [ -z "$f245_want" ]; then
    f245_msg="MUST_PASS FAILED (training_plane_probe self-test) UNMEASURED: rc=0 but the last"
    f245_msg="$f245_msg line is not the declared 'self-test denominator: N of N controls'"
    f245_msg="$f245_msg wording -- the measuring unit printed no denominator (doctrine 2);"
    f245_msg="$f245_msg update this leg in the same commit as the wording change."
    f245_msg="$f245_msg Last line: $f245_last"
    no "$f245_msg"
  elif [ "$f245_have" -ne "$f245_want" ]; then
    f245_msg="MUST_PASS FAILED (training_plane_probe self-test): denominator $f245_have of"
    f245_msg="$f245_msg $f245_want controls is not self-consistent -- the self-test examined"
    f245_msg="$f245_msg fewer controls than it claims to have (doctrine 2)"
    no "$f245_msg"
  elif [ "$f245_have" -lt 9 ]; then
    f245_msg="MUST_PASS FAILED (training_plane_probe self-test): control set shrank to"
    f245_msg="$f245_msg $f245_have of $f245_want, below the measured floor of 9 -- a self-test"
    f245_msg="$f245_msg that quietly drops controls still exits 0, so the floor is the control"
    no "$f245_msg"
  else
    f245_msg="MUST_PASS training_plane_probe self-test: rc=0 under python3 -S, denominator"
    f245_msg="$f245_msg $f245_have of $f245_want controls (>= the measured floor of 9): $f245_last"
    ok "$f245_msg"
  fi
fi

# --- MUST_FIRE: training_plane_probe refuses outside a git repository --------
# MEASURED: run with cwd outside any git worktree, the probe exits rc=96
# exactly -- the declared REFUSE code. The assertion is 96, not merely nonzero:
# collapsing it to nonzero would accept a crash (rc=1/2) as a control firing,
# and a crashed detector is not a discriminating one.
#
# This is the #244 lesson pinned as a control. The probe's denominator is the
# git index and nothing else; the alternative -- an rglob minus a blocklist --
# is what let a stale build/ tree DOUBLE this package's measured line count.
# So when git cannot answer, the only honest verdict is REFUSE. A filesystem
# fallback here would silently redefine "the repository" and every axis count
# downstream would be a number about the wrong set of files.
if [ ! -r "checks/training_plane_probe.py" ]; then
  f245_msg="MUST_FIRE UNREACHABLE (training_plane_probe non-repo refusal) UNMEASURED:"
  f245_msg="$f245_msg checks/training_plane_probe.py is not readable -- unreadable is not"
  f245_msg="$f245_msg empty (doctrine 4); the refusal cannot be exercised, 0 of 1 refusal"
  f245_msg="$f245_msg paths measured"
  no "$f245_msg"
else
  f245_abs=$(pwd)/checks/training_plane_probe.py
  f245_tmp=$(mktemp -d)
  f245_rc=0
  # A temp dir is not a git worktree on any runner this suite targets, but it
  # is NOT assumed: if the sandbox happens to sit inside one, the probe would
  # answer about THAT repo and the leg would be measuring nothing, so the
  # non-repo precondition is established first and the leg abstains by name if
  # it cannot be.
  if (cd "$f245_tmp" && git rev-parse --show-toplevel >/dev/null 2>&1); then
    f245_msg="MUST_FIRE UNREACHABLE (training_plane_probe non-repo refusal) UNMEASURED:"
    f245_msg="$f245_msg $f245_tmp is itself inside a git worktree, so the non-repo condition"
    f245_msg="$f245_msg could not be established -- 0 of 1 refusal paths measured; the probe"
    f245_msg="$f245_msg would have answered about the enclosing repo, not refused"
    no "$f245_msg"
  else
    f245_out=$( (cd "$f245_tmp" && python3 -S "$f245_abs" 2>&1) ) || f245_rc=$?
    if [ "$f245_rc" -eq 96 ]; then
      f245_msg="MUST_FIRE training_plane_probe: invoked outside any git worktree it refused"
      f245_msg="$f245_msg with rc=96 (REFUSE) rather than falling back to a filesystem walk --"
      f245_msg="$f245_msg a blocklist walk cannot define 'the repository' (#244)"
      ok "$f245_msg"
    else
      f245_msg="MUST_FIRE UNREACHABLE (training_plane_probe non-repo refusal): rc=$f245_rc"
      f245_msg="$f245_msg outside a git worktree, expected exactly 96 -- rc=0 would mean the"
      f245_msg="$f245_msg probe invented a denominator from the filesystem (#244), and any"
      f245_msg="$f245_msg other nonzero collapses a crash into a control firing; output:"
      f245_msg="$f245_msg $(printf '%s\n' "$f245_out" | tr '\n' ' ')"
      no "$f245_msg"
    fi
  fi
  rm -rf "$f245_tmp"
fi

# --- MUST_PASS: the live tree's own two-axis verdict --------------------------
# The self-test proves the instrument discriminates on fixtures; this leg runs
# it against THIS repository, which is the reading the review documents cite.
# MEASURED on the commit that introduces the probe: rc=0, axis A total 0 over
# 24 git-tracked src/*.py, axis B nonzero, 23 git-tracked *.md clean on axis C.
#
# The assertion is rc=0 AND a nonzero source denominator. rc=0 alone would be
# satisfied by a probe that scanned nothing: 0 of 0 files exits 95 today, but
# the point of restating the denominator here is that the number reaching the
# wire is the number the documents quote.
if [ ! -r "checks/training_plane_probe.py" ]; then
  f245_msg="MUST_PASS FAILED (training_plane_probe live verdict) UNMEASURED:"
  f245_msg="$f245_msg checks/training_plane_probe.py is not readable -- unreadable is not"
  f245_msg="$f245_msg empty (doctrine 4); 0 of 1 live verdicts measured"
  no "$f245_msg"
else
  f245_rc=0
  f245_out=$(python3 -S checks/training_plane_probe.py 2>&1) || f245_rc=$?
  f245_den=$(printf '%s\n' "$f245_out" |
    sed -n 's/^AXIS A module_scope_torch_import: [0-9][0-9]* of \([0-9][0-9]*\).*/\1/p')
  if [ "$f245_rc" -ne 0 ]; then
    f245_msg="MUST_PASS FAILED (training_plane_probe live verdict): rc=$f245_rc on this tree."
    f245_msg="$f245_msg rc=5 means a tracked *.md still asserts a retired bare-absence form"
    f245_msg="$f245_msg (the #245 sentence is back); rc=95 means a denominator went empty;"
    f245_msg="$f245_msg rc=96 means git could not answer. Output:"
    f245_msg="$f245_msg $(printf '%s\n' "$f245_out" | tr '\n' ' ')"
    no "$f245_msg"
  elif [ -z "$f245_den" ]; then
    f245_msg="MUST_PASS FAILED (training_plane_probe live verdict) UNMEASURED: rc=0 but no"
    f245_msg="$f245_msg 'AXIS A module_scope_torch_import: N of M' line was found -- the probe"
    f245_msg="$f245_msg reported no denominator, and unparseable is not passing (doctrine 2)."
    f245_msg="$f245_msg Output: $(printf '%s\n' "$f245_out" | tr '\n' ' ')"
    no "$f245_msg"
  elif [ "$f245_den" -lt 1 ]; then
    f245_msg="MUST_PASS FAILED (training_plane_probe live verdict): source denominator is"
    f245_msg="$f245_msg $f245_den git-tracked src/*.py -- zero units is UNMEASURED, never a"
    f245_msg="$f245_msg clean reading (doctrine 1)"
    no "$f245_msg"
  else
    f245_msg="MUST_PASS training_plane_probe live verdict: rc=0 over $f245_den git-tracked"
    f245_msg="$f245_msg src/*.py, both axes on the wire and no tracked *.md asserting a retired"
    f245_msg="$f245_msg bare-absence form"
    ok "$f245_msg"
  fi
fi

echo "== fix247-makefiletooling: checks/makefile_tooling.py real legs =="
# --- finding #247 (REOPENS #232 as a class): five further Makefile recipes
# invoked a tool by BARE NAME -- pytest, ruff, mypy, pip, python3 -- so
# `make lint` resolved against the developer's PATH rather than against the
# repository, and died `No module named ruff` on the machine the target
# exists to serve. The repair routed all 22 invocations through $(PY).
#
# This block is the detector's CALL SITE. A gate file entering checks/ with
# no leg here is the #86 orphan class, which the fix78-orphan block below
# already refuses -- #238 says the gate, its legs, the Makefile target and
# the CI step land in ONE commit, and this is the leg half of that.
#
# Four legs, because the gate declares four states and a state that no leg
# ever reaches is a state that is written down rather than measured
# (#198/#200): CLEAR on this tree, RED on a planted bare name, UNMEASURED on
# an empty denominator, and the self-test that proves the discrimination.

# --- MUST_PASS: makefile_tooling self-test -----------------------------------
# MEASURED on the commit that introduces the gate: `python3 -S
# checks/makefile_tooling.py --self-test` exits 0 and prints
# `self-test: 21 of 21 controls ok (11 MUST_FIRE, 10 MUST_PASS)`.
# The floor is a floor: controls may be ADDED, never silently dropped, and a
# shrinking control set is how a detector quietly stops discriminating.
f247_floor=21
if [ ! -r "checks/makefile_tooling.py" ]; then
  f247_msg="MUST_PASS FAILED (makefile_tooling self-test) UNMEASURED:"
  f247_msg="$f247_msg checks/makefile_tooling.py is not readable -- unreadable is not empty"
  f247_msg="$f247_msg and it is not clean (doctrine 4); 0 of 1 self-tests measured"
  no "$f247_msg"
else
  f247_rc=0
  f247_out=$(python3 -S checks/makefile_tooling.py --self-test 2>&1) || f247_rc=$?
  f247_have=$(printf '%s\n' "$f247_out" |
    sed -n 's/^self-test: \([0-9][0-9]*\) of \([0-9][0-9]*\) controls ok.*/\1/p')
  f247_tot=$(printf '%s\n' "$f247_out" |
    sed -n 's/^self-test: \([0-9][0-9]*\) of \([0-9][0-9]*\) controls ok.*/\2/p')
  if [ "$f247_rc" -ne 0 ]; then
    f247_msg="MUST_PASS FAILED (makefile_tooling self-test): rc=$f247_rc under python3 -S."
    f247_msg="$f247_msg A red self-test means the instrument no longer discriminates, so its"
    f247_msg="$f247_msg verdict on the live Makefile is worth nothing. Output:"
    f247_msg="$f247_msg $(printf '%s\n' "$f247_out" | tr '\n' ' ')"
    no "$f247_msg"
  elif [ -z "$f247_have" ] || [ -z "$f247_tot" ]; then
    f247_msg="MUST_PASS FAILED (makefile_tooling self-test) UNMEASURED: rc=0 but no"
    f247_msg="$f247_msg 'self-test: N of M controls ok' line was found -- a control suite that"
    f247_msg="$f247_msg reports no denominator has not reported (doctrine 2). Output:"
    f247_msg="$f247_msg $(printf '%s\n' "$f247_out" | tr '\n' ' ')"
    no "$f247_msg"
  elif [ "$f247_have" != "$f247_tot" ]; then
    f247_msg="MUST_PASS FAILED (makefile_tooling self-test): $f247_have of $f247_tot controls"
    f247_msg="$f247_msg ok but rc=0 -- the exit code and the summary disagree, and one side of"
    f247_msg="$f247_msg the contract is lying (doctrine 6)"
    no "$f247_msg"
  elif [ "$f247_tot" -lt "$f247_floor" ]; then
    f247_msg="MUST_PASS FAILED (makefile_tooling self-test): control set shrank to $f247_tot,"
    f247_msg="$f247_msg below the $f247_floor measured when the gate landed -- controls may be"
    f247_msg="$f247_msg added, never dropped, and a green run over a shrunken set is doctrine 1"
    no "$f247_msg"
  else
    f247_msg="MUST_PASS makefile_tooling self-test: rc=0 under python3 -S, $f247_have of"
    f247_msg="$f247_msg $f247_tot controls ok, at or above the $f247_floor-control floor"
    ok "$f247_msg"
  fi
fi

# --- MUST_PASS: the live tree's own verdict ----------------------------------
# The self-test proves the instrument discriminates on fixtures; this leg runs
# it against THIS repository's Makefile, which is the reading the #247 repair
# claims. MEASURED on the landing commit: rc=0 over 54 recipe lines.
#
# The assertion is rc=0 AND a nonzero recipe-line denominator. rc=0 alone is
# satisfied by a gate that read nothing: 0 recipe lines exits 95 today, but
# restating the denominator here is what keeps the number on the wire equal
# to the number the Makefile's own comment quotes.
if [ ! -r "checks/makefile_tooling.py" ]; then
  f247_msg="MUST_PASS FAILED (makefile_tooling live verdict) UNMEASURED:"
  f247_msg="$f247_msg checks/makefile_tooling.py is not readable -- unreadable is not empty"
  f247_msg="$f247_msg (doctrine 4); 0 of 1 live verdicts measured"
  no "$f247_msg"
else
  f247_rc=0
  f247_out=$(python3 -S checks/makefile_tooling.py 2>&1) || f247_rc=$?
  f247_den=$(printf '%s\n' "$f247_out" |
    sed -n 's/^CLEAR makefile_tooling: 0 bare tool invocations over \([0-9][0-9]*\) recipe.*/\1/p')
  if [ "$f247_rc" -ne 0 ]; then
    f247_msg="MUST_PASS FAILED (makefile_tooling live verdict): rc=$f247_rc on this tree."
    f247_msg="$f247_msg rc=5 means a recipe line invokes a tool by bare name again (#232/#247);"
    f247_msg="$f247_msg rc=95 means the Makefile went unreadable or lost every recipe line;"
    f247_msg="$f247_msg rc=96 means the gate crashed, which is not a verdict. Output:"
    f247_msg="$f247_msg $(printf '%s\n' "$f247_out" | tr '\n' ' ')"
    no "$f247_msg"
  elif [ -z "$f247_den" ]; then
    f247_msg="MUST_PASS FAILED (makefile_tooling live verdict) UNMEASURED: rc=0 but no"
    f247_msg="$f247_msg 'CLEAR makefile_tooling: 0 bare tool invocations over N recipe lines'"
    f247_msg="$f247_msg line was found -- unparseable is not passing (doctrine 2). Output:"
    f247_msg="$f247_msg $(printf '%s\n' "$f247_out" | tr '\n' ' ')"
    no "$f247_msg"
  elif [ "$f247_den" -lt 1 ]; then
    f247_msg="MUST_PASS FAILED (makefile_tooling live verdict): recipe-line denominator is"
    f247_msg="$f247_msg $f247_den -- zero units is UNMEASURED, never a pass (doctrine 1)"
    no "$f247_msg"
  else
    f247_msg="MUST_PASS makefile_tooling live verdict: rc=0, 0 bare tool invocations over"
    f247_msg="$f247_msg $f247_den recipe lines of this repository's Makefile"
    ok "$f247_msg"
  fi
fi

# --- MUST_FIRE: a planted bare tool name is RED ------------------------------
# The gate is copied into a throwaway tree whose Makefile invokes `ruff` bare.
# Copying rather than editing the real Makefile is deliberate: a control that
# mutates the tree it guards can leave the repository dirty on any early exit,
# and #239 was exactly a leg committed red against its own target.
#
# This leg also proves the gate resolves its Makefile from its OWN location
# (parents[1]) rather than from the working directory -- if it read $PWD it
# would score the real Makefile here and stay green, which is the failure this
# leg would then be blind to.
f247_tmp=$(mktemp -d 2>/dev/null || mktemp -d -t fs247)
if [ ! -r "checks/makefile_tooling.py" ] || [ -z "$f247_tmp" ] || [ ! -d "$f247_tmp" ]; then
  f247_msg="MUST_FIRE UNREACHABLE (makefile_tooling planted bare name) UNMEASURED: could not"
  f247_msg="$f247_msg stage a throwaway tree (gate readable? mktemp ok?) -- an unreachable"
  f247_msg="$f247_msg control is a declared state, not a silent pass (doctrine 5)"
  no "$f247_msg"
else
  mkdir -p "$f247_tmp/checks"
  cp checks/makefile_tooling.py "$f247_tmp/checks/makefile_tooling.py"
  printf 'PY := python3\n\nlint:\n\truff check src\n\ntest:\n\t$(PY) -m pytest\n' \
    > "$f247_tmp/Makefile"
  f247_rc=0
  f247_out=$(python3 -S "$f247_tmp/checks/makefile_tooling.py" 2>&1) || f247_rc=$?
  if [ "$f247_rc" -eq 5 ] && printf '%s\n' "$f247_out" | grep -q 'bare `ruff` in command position'; then
    f247_msg="MUST_FIRE makefile_tooling: a recipe line reading 'ruff check src' was scored"
    f247_msg="$f247_msg rc=5 (RED) and named as a bare command word, while the sibling"
    f247_msg="$f247_msg '\$(PY) -m pytest' line in the same fixture was not flagged -- the gate"
    f247_msg="$f247_msg discriminates the defect from its own fix"
    ok "$f247_msg"
  else
    f247_msg="MUST_FIRE UNREACHABLE (makefile_tooling planted bare name): rc=$f247_rc on a"
    f247_msg="$f247_msg Makefile whose recipe invokes 'ruff' bare, expected exactly 5 with the"
    f247_msg="$f247_msg tool named. rc=0 means the detector is blind to the #232/#247 shape;"
    f247_msg="$f247_msg rc=95 means it never found the planted Makefile, which would mean it"
    f247_msg="$f247_msg reads \$PWD rather than its own location. Output:"
    f247_msg="$f247_msg $(printf '%s\n' "$f247_out" | tr '\n' ' ')"
    no "$f247_msg"
  fi
  rm -rf "$f247_tmp"
fi

# --- MUST_FIRE: an empty denominator is UNMEASURED, not CLEAR ----------------
# A Makefile with no TAB-indented recipe has zero units to scan. `all([])` is
# True, so the natural implementation returns 0 and reads as "clean". This leg
# pins the refusal: the gate must exit 95, and 95 must not be 0.
f247_tmp=$(mktemp -d 2>/dev/null || mktemp -d -t fs247)
if [ ! -r "checks/makefile_tooling.py" ] || [ -z "$f247_tmp" ] || [ ! -d "$f247_tmp" ]; then
  f247_msg="MUST_FIRE UNREACHABLE (makefile_tooling empty denominator) UNMEASURED: could not"
  f247_msg="$f247_msg stage a throwaway tree -- unreachable is a declared state (doctrine 5)"
  no "$f247_msg"
else
  mkdir -p "$f247_tmp/checks"
  cp checks/makefile_tooling.py "$f247_tmp/checks/makefile_tooling.py"
  printf 'PY := python3\n\n.PHONY: all\nall:\n' > "$f247_tmp/Makefile"
  f247_rc=0
  f247_out=$(python3 -S "$f247_tmp/checks/makefile_tooling.py" 2>&1) || f247_rc=$?
  if [ "$f247_rc" -eq 95 ] && printf '%s\n' "$f247_out" | grep -q '^UNMEASURED makefile_tooling:'; then
    f247_msg="MUST_FIRE makefile_tooling: over a Makefile with 0 recipe lines the gate exited"
    f247_msg="$f247_msg 95 (UNMEASURED) and said so, rather than exiting 0 over an empty"
    f247_msg="$f247_msg denominator -- zero units is not a pass (doctrine 1)"
    ok "$f247_msg"
  else
    f247_msg="MUST_FIRE UNREACHABLE (makefile_tooling empty denominator): rc=$f247_rc over a"
    f247_msg="$f247_msg Makefile with no TAB-indented recipe, expected exactly 95. rc=0 is the"
    f247_msg="$f247_msg vacuous truth itself -- a gate reporting CLEAR over nothing scanned."
    f247_msg="$f247_msg Output: $(printf '%s\n' "$f247_out" | tr '\n' ' ')"
    no "$f247_msg"
  fi
  rm -rf "$f247_tmp"
fi

echo "== fix78-orphan: every launchers/*.py and checks/*.py harness helper carries at least one call site in this suite (anti-orphan, #86 class) =="
# This control was earned the hard way: the F78 census-writer driver
# shipped a full round with ZERO call sites while its two legs burned red
# on a dead inline heredoc -- an orphan helper is the #86 defect class: a
# control that never RUNS is not a control (doctrine 3), and an
# unreferenced helper rots in silence because nothing can see it rot. The
# detector is a SINGLE shared function (the rule is shared, not
# duplicated) run over the real tree with the examined denominator printed
# (MUST_PASS) and over a COPY rigged with a planted decoy (MUST_FIRE).
# Matching is a WORD-BOUNDARY fixed-string grep for the helper's basename
# (a substring match would let a basename that prefixes another helper's
# name borrow that helper's call site and false-green; -w still matches
# 'launchers/<base>.py' occurrences, so real call sites are not lost). A
# comment-only mention would still count as a call site -- stated, not
# hidden (doctrine 5) -- which is why this leg spells no real helper's
# basename in its own text and why the decoy's basename is ASSEMBLED AT
# RUNTIME: a fixed decoy name written into this leg would be found by the
# very grep under test, and the fire rig could never go red. A zero-file
# sweep is UNMEASURED-red, never a vacuous pass (doctrine 1), and the fire
# rig must ALSO observe the same detector return to green once the decoy's
# call site is planted on the copy: a constant-red is no more a control
# than a constant-green (doctrine 3, symmetric).
f78_orph_scan() {
  # $1 = root holding launchers/ and checks/ ; $2 = suite text grepped for
  # call sites. stdout: one ORPH_CALLSITE_OK|<base> or ORPH_ORPHAN|<base>
  # per examined file (one unit per line, doctrine 2), then
  # ORPH_HELPERS=<n> and ORPH_ORPHANS=<csv|none>. rc: 0 = n>0 and no
  # orphans; 1 = at least one orphan indicted; 2 = zero files examined
  # (unmeasured -- never a pass, doctrine 1).
  local oroot=$1 osuite=$2 ofile obase on=0 orph=""
  for ofile in "$oroot"/launchers/*.py "$oroot"/checks/*.py; do
    [ -f "$ofile" ] || continue
    on=$((on+1))
    obase=${ofile##*/}
    if grep -Fwq -- "$obase" "$osuite"; then
      printf 'ORPH_CALLSITE_OK|%s\n' "$obase"
    else
      printf 'ORPH_ORPHAN|%s\n' "$obase"
      orph=${orph:+$orph,}$obase
    fi
  done
  printf 'ORPH_HELPERS=%d\n' "$on"
  printf 'ORPH_ORPHANS=%s\n' "${orph:-none}"
  [ "$on" -eq 0 ] && return 2
  [ -n "$orph" ] && return 1
  return 0
}
# fix#257 (B3 split 3): this gate's denominator is every launchers/*.py and
# checks/*.py helper, but their CALL SITES are now spread across TWO suite
# files. Reading $0 would indict every helper called only from the other suite,
# so the corpus is both suites concatenated. This is load-bearing rather than
# defensive, and the number was measured on the split commit: of the 11 helpers
# in the denominator, 5 (countables_drift, makefile_tooling,
# packaging_reachability, training_plane_probe, wf_yaml_audit) have ZERO
# citations in test_launcher_contracts.sh and would have gone red on the first
# push -- and 4 more run the other way, cited only in the launcher suite.
# Fail-closed: if either member is
# unreadable the corpus path is pointed at a nonexistent file, which the
# [ -r ] guard below routes to the UNMEASURED-red arm rather than to a pass --
# an unreadable member is not an empty one (doctrine 4).
f78_orph_suite=$(mktemp "${TMPDIR:-/tmp}/fs-f78-corpus.XXXXXX")
for f78_orph_member in "$LDIR/test_launcher_contracts.sh" "$LDIR/test_checks_gates.sh"; do
  if [ -r "$f78_orph_member" ]; then
    cat "$f78_orph_member" >> "$f78_orph_suite"
  else
    rm -f "$f78_orph_suite"
    f78_orph_suite=$f78_orph_member  # unreadable: carry the name into the red
    break
  fi
done
# The corpus is a temp path, which is useless in a verdict a human reads. The
# messages name the real inputs instead; the corpus is an implementation
# detail of how they are grepped, not the thing being claimed about.
f78_orph_label="test_launcher_contracts.sh + test_checks_gates.sh"
f78_orph_passed=1
f78_orph_n=0
f78_orph_orphs=unknown
if [ -r "$f78_orph_suite" ]; then
  f78_orph_real=$(f78_orph_scan . "$f78_orph_suite")
  f78_orph_rcc=$?
  f78_orph_n=$(printf '%s\n' "$f78_orph_real" | grep -m1 '^ORPH_HELPERS=' | cut -d= -f2)
  f78_orph_orphs=$(printf '%s\n' "$f78_orph_real" | grep -m1 '^ORPH_ORPHANS=' | cut -d= -f2-)
  if [ "$f78_orph_rcc" -eq 0 ] && [ "${f78_orph_n:-0}" -gt 0 ] && [ "${f78_orph_orphs:-<missing>}" = none ]; then
    f78_orph_passed=0
  fi
fi
if [ "$f78_orph_passed" -eq 0 ]; then
  ok "MUST_PASS no orphan harness helpers: $f78_orph_n of $f78_orph_n examined files (launchers/*.py + checks/*.py) carry at least one word-boundary call site in $f78_orph_label (orphans: none) -- this leg exists because a writer driver shipped orphaned for a full round while its legs burned red; that defect class is now indicted by name (#86 class)"
else
  no "MUST_PASS FAILED (no orphan harness helpers): examined ${f78_orph_n:-0} files matching launchers/*.py + checks/*.py against call sites in $f78_orph_label (suite readable: $( [ -r "$f78_orph_suite" ] && echo yes || echo no )) -- orphans (zero call-site basenames): ${f78_orph_orphs:-unknown}$( [ "${f78_orph_n:-0}" = 0 ] && printf '; ZERO files examined is UNMEASURED, never PASS (doctrine 1)' ) -- wire the helper into the suite or delete it; an unreferenced helper rots in silence"
fi
# MUST_FIRE (orphan detector, SET-based): same COPIES discipline as before -- the
# real tree and the real suite are never modified -- and the SAME f78_orph_scan
# the MUST_PASS leg uses is driven with || rc=$? capture so a nonzero verdict
# survives set -e instead of killing the suite (one seam, one source of truth).
# The decoy's basename is assembled at runtime ($$ = this shell's pid, so the
# full name occurs nowhere in any text the grep can see until its call site is
# planted on the copy). The decoy must be indicted BY NAME: the raw red output
# must carry its own ORPH_ORPHAN|<decoy> line -- CSV membership alone is derived
# data and would still pass if the scan's per-name indictment grammar drifted.
# Discrimination is SET-based, not decoy-alone: the decoy must be IN the red
# orphan set, ABSENT from the green set, and the two sets must be EQUAL once
# the decoy is subtracted -- so a pre-existing orphan elsewhere in the copied
# tree can neither disarm this control (the old "decoy alone + green rc 0" rig
# failed exactly when the estate was dirtiest) nor count as detector noise;
# global cleanliness stays owned by the MUST_PASS leg above. Red rc is forced
# to exactly 1: rc 2 is zero-files-examined and anything above 1 is the scan
# itself breaking -- a crash must never count as "fired". Green is observed by
# planting the decoy's call site ON THE COPY and re-scanning: the detector
# must stop indicting a helper whose call site exists (a constant-red is no
# more a control than a constant-green, doctrine 3 symmetric), and green rc
# must track the green set (0 iff empty, 1 iff nonempty) so the scan cannot
# contradict its own printed verdict. Sets are de-spaced before every
# comparison so comma/comma-space CSV formatting cannot false-fail the
# control, and both denominators are recomputed live: each scan must examine
# exactly realc+1 copied units. Any copy/plant failure leaves
# f78_orph_ffired=1: a MUST_FIRE that could not plant its mutation is
# UNREACHABLE-red, never green.
f78_orph_ffired=1
f78_orph_why=setup-failed
f78_orph_rigbad=""
f78_orph_froot=$(mktemp -d "${TMPDIR:-/tmp}/fs-f78-orphan.XXXXXX")
if [ -d "$f78_orph_froot" ] && [ -r "$f78_orph_suite" ]; then
  mkdir -p "$f78_orph_froot/launchers" "$f78_orph_froot/checks"
  f78_orph_scopy=$f78_orph_froot/suite-copy.txt
  f78_orph_realc=0
  for f78_orph_src in launchers/*.py; do
    if [ -f "$f78_orph_src" ]; then
      cp "$f78_orph_src" "$f78_orph_froot/launchers/"
      f78_orph_realc=$((f78_orph_realc+1))
    fi
  done
  for f78_orph_src in checks/*.py; do
    if [ -f "$f78_orph_src" ]; then
      cp "$f78_orph_src" "$f78_orph_froot/checks/"
      f78_orph_realc=$((f78_orph_realc+1))
    fi
  done
  cp "$f78_orph_suite" "$f78_orph_scopy"
  f78_orph_decoy="zz_orphan_decoy_$$.py"
  f78_orph_dpath=$f78_orph_froot/launchers/$f78_orph_decoy
  printf '# planted decoy: defines nothing, is called by nothing\n' > "$f78_orph_dpath"
  if [ "$f78_orph_realc" -gt 0 ] && [ -s "$f78_orph_scopy" ]; then
    f78_orph_redrc=0
    f78_orph_red=$(f78_orph_scan "$f78_orph_froot" "$f78_orph_scopy") || f78_orph_redrc=$?
    f78_orph_fn=$(printf '%s\n' "$f78_orph_red" | grep -m1 '^ORPH_HELPERS=' | cut -d= -f2)
    f78_orph_rset=$(printf '%s\n' "$f78_orph_red" | grep -m1 '^ORPH_ORPHANS=' | cut -d= -f2-)
    f78_orph_rset=$(printf '%s' "$f78_orph_rset" | tr -d ' ')
    if [ -z "$f78_orph_rset" ] || [ "$f78_orph_rset" = none ]; then f78_orph_rset=-; fi
    printf 'python3 launchers/%s  # planted call site (copy only)\n' \
      "$f78_orph_decoy" >> "$f78_orph_scopy"
    f78_orph_greenrc=0
    f78_orph_green=$(f78_orph_scan "$f78_orph_froot" "$f78_orph_scopy") || f78_orph_greenrc=$?
    f78_orph_gn=$(printf '%s\n' "$f78_orph_green" | grep -m1 '^ORPH_HELPERS=' | cut -d= -f2)
    f78_orph_gset=$(printf '%s\n' "$f78_orph_green" | grep -m1 '^ORPH_ORPHANS=' | cut -d= -f2-)
    f78_orph_gset=$(printf '%s' "$f78_orph_gset" | tr -d ' ')
    if [ -z "$f78_orph_gset" ] || [ "$f78_orph_gset" = none ]; then f78_orph_gset=-; fi
    f78_orph_rminus=$(printf '%s' ",$f78_orph_rset," | sed "s/,$f78_orph_decoy,/,/")
    f78_orph_rminus=${f78_orph_rminus#,}
    f78_orph_rminus=${f78_orph_rminus%,}
    if [ -z "$f78_orph_rminus" ]; then f78_orph_rminus=-; fi
    f78_orph_why="red-rc=$f78_orph_redrc red-set=$f78_orph_rset green-rc=$f78_orph_greenrc"
    f78_orph_why="$f78_orph_why green-set=$f78_orph_gset red-minus-decoy=$f78_orph_rminus"
    f78_orph_why="$f78_orph_why examined=${f78_orph_fn:-?}->${f78_orph_gn:-?}"
    f78_orph_why="$f78_orph_why want=$((f78_orph_realc+1))-per-scan"
    case ",$f78_orph_rset," in
      *",$f78_orph_decoy,"*) ;;
      *) f78_orph_rigbad="$f78_orph_rigbad decoy-not-in-red-set" ;;
    esac
    case ",$f78_orph_gset," in
      *",$f78_orph_decoy,"*) f78_orph_rigbad="$f78_orph_rigbad decoy-still-in-green-set" ;;
    esac
    printf '%s\n' "$f78_orph_red" | grep -qF "ORPH_ORPHAN|$f78_orph_decoy" \
      || f78_orph_rigbad="$f78_orph_rigbad decoy-not-indicted-by-name"
    if [ "$f78_orph_redrc" -ne 1 ]; then
      f78_orph_rigbad="$f78_orph_rigbad red-rc=$f78_orph_redrc-not-exactly-1"
    fi
    if [ "$f78_orph_gset" = "-" ]; then
      if [ "$f78_orph_greenrc" -ne 0 ]; then
        f78_orph_rigbad="$f78_orph_rigbad green-rc=$f78_orph_greenrc-on-empty-set"
      fi
    elif [ "$f78_orph_greenrc" -ne 1 ]; then
      f78_orph_rigbad="$f78_orph_rigbad green-rc=$f78_orph_greenrc-on-nonempty-set"
    fi
    if [ "$f78_orph_rminus" != "$f78_orph_gset" ]; then
      f78_orph_rigbad="$f78_orph_rigbad sets-differ-beyond-the-decoy"
    fi
    if [ "$f78_orph_fn" != "$f78_orph_gn" ] \
       || [ "${f78_orph_fn:-0}" -ne $((f78_orph_realc+1)) ]; then
      f78_orph_rigbad="$f78_orph_rigbad denominators-not-red==green==realc+1"
    fi
    if [ -z "$f78_orph_rigbad" ]; then f78_orph_ffired=0; fi
  fi
fi
if [ "$f78_orph_ffired" -eq 0 ]; then
  f78_orph_fmsg="MUST_FIRE orphan detector SET-discriminates: 1 planted decoy helper"
  f78_orph_fmsg="$f78_orph_fmsg (runtime-assembled name occurring nowhere in any greppable"
  f78_orph_fmsg="$f78_orph_fmsg text until its call site lands, zero call sites by"
  f78_orph_fmsg="$f78_orph_fmsg construction) was indicted BY NAME (its own ORPH_ORPHAN|"
  f78_orph_fmsg="$f78_orph_fmsg line) IN the red orphan set and ABSENT from the green set"
  f78_orph_fmsg="$f78_orph_fmsg with red-minus-decoy EQUAL to the green set"
  f78_orph_fmsg="$f78_orph_fmsg ($f78_orph_rminus); both scans examined $((f78_orph_realc+1))"
  f78_orph_fmsg="$f78_orph_fmsg copied units ($f78_orph_realc real + 1 decoy) against a COPY"
  f78_orph_fmsg="$f78_orph_fmsg of this suite, red rc exactly 1, green rc tracking the green"
  f78_orph_fmsg="$f78_orph_fmsg set -- observed firing AND recovering under discrimination"
  f78_orph_fmsg="$f78_orph_fmsg that pre-existing dirt can neither disarm nor impersonate"
  f78_orph_fmsg="$f78_orph_fmsg (doctrine 3, symmetric; #86 class)"
  ok "$f78_orph_fmsg"
else
  f78_orph_fmsg="MUST_FIRE UNREACHABLE (orphan detector): the planted zero-call-site"
  f78_orph_fmsg="$f78_orph_fmsg decoy rig failed set discrimination ($f78_orph_why):"
  f78_orph_fmsg="$f78_orph_fmsg$f78_orph_rigbad [scan rc: 1=orphan found, 2=zero files"
  f78_orph_fmsg="$f78_orph_fmsg examined] -- a control that cannot see a planted orphan,"
  f78_orph_fmsg="$f78_orph_fmsg or cannot stop seeing one whose call site exists, would"
  f78_orph_fmsg="$f78_orph_fmsg wave the next #86-class orphan through exactly the way this"
  f78_orph_fmsg="$f78_orph_fmsg round's orphaned driver shipped"
  no "$f78_orph_fmsg"
fi
[ -n "${f78_orph_froot:-}" ] && rm -rf "$f78_orph_froot" || true
[ -f "$f78_orph_suite" ] && rm -f "$f78_orph_suite"
# --- MUST_PASS: coverage_floor self-test (checks/coverage_floor.py) --------
# MEASURED: `python3 -S checks/coverage_floor.py --self-test` exits rc=0 and
# its last line is the declared
#   SELF-TEST DENOMINATOR: 12 of 12 controls behaved; ...
# rc=0 alone is NOT the measurement: a control set that shrank to one control
# would still be capable of exiting 0. The trailing "N of N" tally is parsed,
# required to be non-empty and self-consistent, and held at N >= 12 -- the 8
# MUST_FIRE + 4 MUST_PASS controls present when this leg was written. A
# wording change reds THIS leg and must update it in the same commit.
if [ ! -r "checks/coverage_floor.py" ]; then
  f252_msg="MUST_PASS FAILED (coverage_floor self-test) UNMEASURED:"
  f252_msg="$f252_msg checks/coverage_floor.py is not readable -- unreadable is not"
  f252_msg="$f252_msg empty; the gate cannot run, so 0 of its declared denominator of 12"
  f252_msg="$f252_msg controls (8 MUST_FIRE + 4 MUST_PASS) were measured. An unreadable"
  f252_msg="$f252_msg measuring unit is failed closed, never skipped."
  no "$f252_msg"
else
  f252_rc=0
  f252_out=$(python3 -S checks/coverage_floor.py --self-test 2>&1) || f252_rc=$?
  f252_last=$(printf '%s\n' "$f252_out" | tail -n 1)
  f252_have=$(printf '%s\n' "$f252_last" |
    sed -n 's/^SELF-TEST DENOMINATOR: \([0-9][0-9]*\) of \([0-9][0-9]*\) controls behaved;.*/\1/p')
  f252_want=$(printf '%s\n' "$f252_last" |
    sed -n 's/^SELF-TEST DENOMINATOR: \([0-9][0-9]*\) of \([0-9][0-9]*\) controls behaved;.*/\2/p')
  if [ "$f252_rc" -ne 0 ]; then
    f252_msg="MUST_PASS FAILED (coverage_floor self-test): rc=$f252_rc over the gate's"
    f252_msg="$f252_msg declared denominator of 12 controls (8 MUST_FIRE + 4 MUST_PASS);"
    f252_msg="$f252_msg 0 of 12 controls were accepted as behaved, so the leg fails closed."
    f252_msg="$f252_msg Output: $(printf '%s\n' "$f252_out" | tr '\n' ' ')"
    no "$f252_msg"
  elif [ -z "$f252_have" ] || [ -z "$f252_want" ]; then
    f252_msg="MUST_PASS FAILED (coverage_floor self-test) UNMEASURED: rc=0 but the last"
    f252_msg="$f252_msg line carries no parseable 'SELF-TEST DENOMINATOR: N of N controls"
    f252_msg="$f252_msg behaved' tally -- the measuring unit printed no denominator, so 0 of"
    f252_msg="$f252_msg 12 declared controls are auditable here. Unparseable is not passing;"
    f252_msg="$f252_msg fail closed and update this leg in the same commit as the wording"
    f252_msg="$f252_msg change. Last line: $f252_last"
    no "$f252_msg"
  elif [ "$f252_have" -ne "$f252_want" ]; then
    f252_msg="MUST_PASS FAILED (coverage_floor self-test): denominator $f252_have of"
    f252_msg="$f252_msg $f252_want controls is not self-consistent -- the self-test examined"
    f252_msg="$f252_msg fewer controls than it claims to have, over the declared denominator"
    f252_msg="$f252_msg of 12. The inconsistency is failed closed because rc=0 cannot certify"
    f252_msg="$f252_msg a partial control set."
    no "$f252_msg"
  elif [ "$f252_have" -lt 12 ]; then
    f252_msg="MUST_PASS FAILED (coverage_floor self-test): control set shrank to"
    f252_msg="$f252_msg $f252_have of $f252_want, below the measured floor of 12 controls"
    f252_msg="$f252_msg (8 MUST_FIRE + 4 MUST_PASS). A shortened self-test can still exit 0,"
    f252_msg="$f252_msg so the floor is the control and this leg fails closed."
    no "$f252_msg"
  else
    f252_msg="MUST_PASS coverage_floor self-test: rc=0 under python3 -S, denominator"
    f252_msg="$f252_msg $f252_have of $f252_want controls (>= the measured floor of 12,"
    f252_msg="$f252_msg 8 MUST_FIRE + 4 MUST_PASS): $f252_last"
    ok "$f252_msg"
  fi
fi

# --- MUST_FIRE: coverage_floor refuses an absent coverage report ------------
# MEASURED: a path deliberately constructed inside a fresh temporary directory,
# verified absent, is passed as `--report`; `python3 -S
# checks/coverage_floor.py --report "$missing"` exits exactly rc=95
# (UNMEASURED), never rc=0. A CLEAR verdict over a report that does not exist
# would be the repository's defining vacuous pass. The assertion is 95, not
# merely nonzero: a crash or refusal code is not evidence that the missing-
# report detector is discriminating as declared.
if [ ! -r "checks/coverage_floor.py" ]; then
  f252_msg="MUST_FIRE UNREACHABLE (coverage_floor absent-report refusal) UNMEASURED:"
  f252_msg="$f252_msg checks/coverage_floor.py is not readable -- unreadable is not empty;"
  f252_msg="$f252_msg 0 of 1 declared absent-report refusal paths could be exercised, and"
  f252_msg="$f252_msg this fail-closed leg never treats an unreadable gate as measured."
  no "$f252_msg"
else
  f252_tmp=""
  f252_tmp=$(mktemp -d "${TMPDIR:-/tmp}/fix252-coveragemissing.XXXXXX" 2>/dev/null) || f252_tmp=""
  if [ -z "$f252_tmp" ] || [ ! -d "$f252_tmp" ]; then
    f252_msg="MUST_FIRE UNREACHABLE (coverage_floor absent-report refusal) UNMEASURED:"
    f252_msg="$f252_msg mktemp -d did not create the scratch directory that carries the"
    f252_msg="$f252_msg absent report, so 0 of 1 declared absent-report refusal paths were"
    f252_msg="$f252_msg measured. The fixture construction itself is failed closed rather"
    f252_msg="$f252_msg than allowing an un-isolated path to stand in for an absent report."
    no "$f252_msg"
  else
    f252_missing="$f252_tmp/coverage-report-that-does-not-exist.json"
    if [ -e "$f252_missing" ] || [ -L "$f252_missing" ]; then
      rm -rf "$f252_tmp"
      f252_msg="MUST_FIRE UNREACHABLE (coverage_floor absent-report refusal) UNMEASURED:"
      f252_msg="$f252_msg the constructed report path already existed before invocation,"
      f252_msg="$f252_msg so 0 of 1 declared ABSENT-report refusal paths were actually"
      f252_msg="$f252_msg constructed. Firing condition failed its own construction and is"
      f252_msg="$f252_msg failed closed; $f252_missing"
      no "$f252_msg"
    else
      f252_rc=0
      f252_out=$(python3 -S checks/coverage_floor.py --report "$f252_missing" 2>&1) || f252_rc=$?
      rm -rf "$f252_tmp"
      if [ "$f252_rc" -eq 95 ]; then
        f252_msg="MUST_FIRE coverage_floor absent-report refusal: over 1 of 1 constructed"
        f252_msg="$f252_msg absent-report paths, the gate exited rc=95 (UNMEASURED), refusing"
        f252_msg="$f252_msg to launder evidence it never read into a pass"
        ok "$f252_msg"
      else
        f252_msg="MUST_FIRE UNREACHABLE (coverage_floor absent-report refusal): rc=$f252_rc"
        f252_msg="$f252_msg over 1 of 1 constructed absent-report paths, expected exactly 95"
        f252_msg="$f252_msg (UNMEASURED). rc=0 would report CLEAR over no evidence, while any"
        f252_msg="$f252_msg other code misclassifies crash, RED, or REFUSE as this detector's"
        f252_msg="$f252_msg declared abstention; the leg therefore fails closed. Output:"
        f252_msg="$f252_msg $(printf '%s\n' "$f252_out" | tr '\n' ' ')"
        no "$f252_msg"
      fi
    fi
  fi
fi

# --- MUST_PASS: ci_suite_extras self-test -----------------------------------
# MEASURED: `python3 -S checks/ci_suite_extras.py --self-test` exits rc=0 and
# publishes an explicit "N of N" control tally. The tally variables must be
# non-empty and equal, and N must be at least 2: one firing-side control and
# one clean-side control are the irreducible minimum for a detector whose job
# is discrimination. rc=0 with a vanished or internally inconsistent tally is
# not evidence; the direct two-workflow discrimination leg below supplies the
# separately constructed historical firing check.
if [ ! -r "checks/ci_suite_extras.py" ]; then
  f252_msg="MUST_PASS FAILED (ci_suite_extras self-test) UNMEASURED:"
  f252_msg="$f252_msg checks/ci_suite_extras.py is not readable -- unreadable is not empty;"
  f252_msg="$f252_msg 0 of the required minimum denominator of 2 controls were measured."
  f252_msg="$f252_msg The gate is failed closed rather than skipped."
  no "$f252_msg"
else
  f252_rc=0
  f252_out=$(python3 -S checks/ci_suite_extras.py --self-test 2>&1) || f252_rc=$?
  f252_last=$(printf '%s\n' "$f252_out" | tail -n 1)
  # ONE house tally format, ONE parser -- the same expression the coverage_floor
  # leg above uses. The first draft of this leg carried three alternative
  # patterns because it was authored in parallel with the gate and had to guess
  # which phrasing the gate would print; it guessed three and the gate printed a
  # fourth, so the tally read as absent and this MUST_PASS would have reported
  # UNMEASURED against a working gate. The fix went into the gate (conform to the
  # house format) rather than here, because a parser that accepts formats no gate
  # emits has dead branches that cannot be exercised by any control -- and an
  # unexercised branch is exactly where a malformed future tally gets accepted.
  f252_have=$(printf '%s\n' "$f252_last" |
    sed -n 's/^SELF-TEST DENOMINATOR: \([0-9][0-9]*\) of \([0-9][0-9]*\) controls behaved;.*/\1/p')
  f252_want=$(printf '%s\n' "$f252_last" |
    sed -n 's/^SELF-TEST DENOMINATOR: \([0-9][0-9]*\) of \([0-9][0-9]*\) controls behaved;.*/\2/p')
  if [ "$f252_rc" -ne 0 ]; then
    f252_msg="MUST_PASS FAILED (ci_suite_extras self-test): rc=$f252_rc over the gate's"
    f252_msg="$f252_msg own control denominator (minimum required: 2 controls; reported"
    f252_msg="$f252_msg tally is checked only after a passing exit). The self-test did not"
    f252_msg="$f252_msg earn a CLEAR verdict, so this leg fails closed. Output:"
    f252_msg="$f252_msg $(printf '%s\n' "$f252_out" | tr '\n' ' ')"
    no "$f252_msg"
  elif [ -z "$f252_have" ] || [ -z "$f252_want" ]; then
    f252_msg="MUST_PASS FAILED (ci_suite_extras self-test) UNMEASURED: rc=0 but the last"
    f252_msg="$f252_msg line is not one of the suite's explicit 'N of N' self-test tally"
    f252_msg="$f252_msg spellings -- the measuring unit printed no denominator, so 0 of the"
    f252_msg="$f252_msg required minimum denominator of 2 controls is auditable. Unparseable"
    f252_msg="$f252_msg is not passing; fail closed and update this leg in the same commit"
    f252_msg="$f252_msg as the wording change. Last line: $f252_last"
    no "$f252_msg"
  elif [ "$f252_have" -ne "$f252_want" ]; then
    f252_msg="MUST_PASS FAILED (ci_suite_extras self-test): denominator $f252_have of"
    f252_msg="$f252_msg $f252_want controls is not self-consistent -- the self-test examined"
    f252_msg="$f252_msg fewer controls than it claims to have. rc=0 cannot certify a partial"
    f252_msg="$f252_msg control set, so the leg fails closed."
    no "$f252_msg"
  elif [ "$f252_have" -lt 2 ]; then
    f252_msg="MUST_PASS FAILED (ci_suite_extras self-test): control tally $f252_have of"
    f252_msg="$f252_msg $f252_want is below the non-vacuous minimum denominator of 2 controls."
    f252_msg="$f252_msg A discriminating gate needs at least one firing-side and one"
    f252_msg="$f252_msg clean-side control; the shrunken set is failed closed."
    no "$f252_msg"
  else
    f252_msg="MUST_PASS ci_suite_extras self-test: rc=0 under python3 -S, control denominator"
    f252_msg="$f252_msg $f252_have of $f252_want (>= the non-vacuous minimum of 2): $f252_last"
    ok "$f252_msg"
  fi
fi

# --- MUST_FIRE: ci_suite_extras reds differing extras and clears reality ----
# MEASURED: a temporary workflow is WRITTEN with exactly two jobs that invoke
# launchers/test_launcher_contracts.sh and two distinct pip extras, [test] and
# [dev]. The fixture text itself is counted before either invocation: 2 suite
# call sites, 1 [test] install, and 1 [dev] install. The doctored workflow must
# exit rc=5 (RED). The real `.github/workflows/ci.yml` must then exit rc=0
# (CLEAR). Requiring both outcomes proves discrimination: rc=5 alone could be
# a stuck-red detector, while rc=0 alone could be a vacuous pass.
if [ ! -r "checks/ci_suite_extras.py" ]; then
  f252_msg="MUST_FIRE UNREACHABLE (ci_suite_extras doctored-workflow discrimination)"
  f252_msg="$f252_msg UNMEASURED: checks/ci_suite_extras.py is not readable -- unreadable is"
  f252_msg="$f252_msg not empty; 0 of 2 required gate invocations (1 doctored RED fixture,"
  f252_msg="$f252_msg 1 real CLEAR workflow) were measured, so the leg fails closed."
  no "$f252_msg"
elif [ ! -r ".github/workflows/ci.yml" ]; then
  f252_msg="MUST_FIRE UNREACHABLE (ci_suite_extras doctored-workflow discrimination)"
  f252_msg="$f252_msg UNMEASURED: .github/workflows/ci.yml is not readable -- unreadable is"
  f252_msg="$f252_msg not the same as clean. 0 of 2 required gate invocations were accepted;"
  f252_msg="$f252_msg without the real CLEAR arm the RED arm cannot prove discrimination,"
  f252_msg="$f252_msg so the leg fails closed."
  no "$f252_msg"
else
  f252_tmp=""
  f252_tmp=$(mktemp -d "${TMPDIR:-/tmp}/fix252-ciworkflow.XXXXXX" 2>/dev/null) || f252_tmp=""
  if [ -z "$f252_tmp" ] || [ ! -d "$f252_tmp" ]; then
    f252_msg="MUST_FIRE UNREACHABLE (ci_suite_extras doctored-workflow discrimination)"
    f252_msg="$f252_msg UNMEASURED: mktemp -d did not create the scratch directory for the"
    f252_msg="$f252_msg 2-job/2-extra fixture, so 0 of 2 required gate invocations were"
    f252_msg="$f252_msg measured. The unconstructed firing condition fails closed."
    no "$f252_msg"
  else
    f252_doctored="$f252_tmp/doctored-ci-suite-extras.yml"
    # The two jobs must EXECUTE THE SUITE by the gate's own definition of that
    # phrase -- a real pytest invocation or a tools/mutate.py run. The first
    # draft of this fixture ran launchers/test_launcher_contracts.sh in both
    # jobs, which is a gate but is not the pytest suite, so the gate correctly
    # answered 95 (zero suite-executing jobs, an empty denominator) and this
    # MUST_FIRE would have failed against a working detector. A firing fixture
    # has to satisfy the detector's denominator rule, not merely look like the
    # defect.
    cat > "$f252_doctored" <<'EOF'
name: fix252 doctored ci-suite extras
on:
  push:
jobs:
  suite-with-test-extra:
    runs-on: ubuntu-latest
    steps:
      - name: Install with test extra
        run: python -m pip install -e ".[test]"
      - name: Run the pytest suite
        run: python -m pytest tests/
  suite-with-dev-extra:
    runs-on: ubuntu-latest
    steps:
      - name: Install with dev extra
        run: python -m pip install -e ".[dev]"
      - name: Run the mutation battery
        run: python tools/mutate.py
EOF
    f252_suite_refs=$(grep -F -c -e 'python -m pytest tests/' \
      -e 'python tools/mutate.py' "$f252_doctored" 2>/dev/null || true)
    f252_test_refs=$(grep -F -c 'python -m pip install -e ".[test]"' "$f252_doctored" 2>/dev/null || true)
    f252_dev_refs=$(grep -F -c 'python -m pip install -e ".[dev]"' "$f252_doctored" 2>/dev/null || true)
    if [ ! -s "$f252_doctored" ]; then
      rm -rf "$f252_tmp"
      f252_msg="MUST_FIRE UNREACHABLE (ci_suite_extras doctored-workflow discrimination)"
      f252_msg="$f252_msg UNMEASURED: writing the 2-job/2-extra temporary workflow produced"
      f252_msg="$f252_msg an empty or missing file, so 0 of 2 required gate invocations were"
      f252_msg="$f252_msg measured. An unwritten fixture cannot establish the claimed firing"
      f252_msg="$f252_msg condition and is failed closed."
      no "$f252_msg"
    elif [ "$f252_suite_refs" -ne 2 ] || [ "$f252_test_refs" -ne 1 ] || [ "$f252_dev_refs" -ne 1 ]; then
      rm -rf "$f252_tmp"
      f252_msg="MUST_FIRE UNREACHABLE (ci_suite_extras doctored-workflow discrimination)"
      f252_msg="$f252_msg UNMEASURED: fixture construction was checked before the gate and"
      f252_msg="$f252_msg found suite-executing steps=$f252_suite_refs (expected 2), [test] installs="
      f252_msg="$f252_msg$f252_test_refs (expected 1), and [dev] installs=$f252_dev_refs"
      f252_msg="$f252_msg (expected 1). 0 of 2 required gate invocations were accepted because"
      f252_msg="$f252_msg the doctored denominator was not the claimed 2 suite jobs with 2"
      f252_msg="$f252_msg distinct extras; fail closed rather than credit a malformed fixture."
      no "$f252_msg"
    else
      f252_doctor_rc=0
      f252_doctor_out=$(python3 -S checks/ci_suite_extras.py --workflow "$f252_doctored" 2>&1) || f252_doctor_rc=$?
      f252_real_rc=0
      f252_real_out=$(python3 -S checks/ci_suite_extras.py --workflow ".github/workflows/ci.yml" 2>&1) || f252_real_rc=$?
      rm -rf "$f252_tmp"
      if [ "$f252_doctor_rc" -ne 5 ]; then
        f252_msg="MUST_FIRE FAILED (ci_suite_extras doctored-workflow discrimination):"
        f252_msg="$f252_msg doctored fixture rc=$f252_doctor_rc, expected exactly 5 (RED),"
        f252_msg="$f252_msg over an independently counted denominator of 2 suite-executing jobs"
        f252_msg="$f252_msg and 2 distinct extras. The real workflow arm returned rc=$f252_real_rc"
        f252_msg="$f252_msg over 1 readable workflow, but only 0 of 1 firing outcomes held;"
        f252_msg="$f252_msg rc=0 would launder differing extras into CLEAR and any nonzero other"
        f252_msg="$f252_msg than 5 is not this detector's declared RED. The leg fails closed."
        f252_msg="$f252_msg Doctored output: $(printf '%s\n' "$f252_doctor_out" | tr '\n' ' ')"
        no "$f252_msg"
      elif [ "$f252_real_rc" -ne 0 ]; then
        f252_msg="MUST_FIRE FAILED (ci_suite_extras doctored-workflow discrimination):"
        f252_msg="$f252_msg the constructed 2-job/2-extra arm fired correctly at rc=5, but"
        f252_msg="$f252_msg the real .github/workflows/ci.yml returned rc=$f252_real_rc instead"
        f252_msg="$f252_msg of 0. Only 1 of 2 discrimination outcomes held; a gate that also"
        f252_msg="$f252_msg rejects its production denominator is stuck RED, not discriminating,"
        f252_msg="$f252_msg so the leg fails closed. Real output:"
        f252_msg="$f252_msg $(printf '%s\n' "$f252_real_out" | tr '\n' ' ')"
        no "$f252_msg"
      else
        f252_msg="MUST_FIRE ci_suite_extras doctored-workflow discrimination: over a counted"
        f252_msg="$f252_msg denominator of 2 suite-executing jobs and 2 distinct extras the"
        f252_msg="$f252_msg doctored fixture exited rc=5 (RED); over the 1 real production"
        f252_msg="$f252_msg workflow it exited rc=0 (CLEAR). Both discrimination outcomes held:"
        f252_msg="$f252_msg the gate is neither vacuous nor stuck red"
        ok "$f252_msg"
      fi
    fi
  fi
fi

echo "abstentions: $abstain named (each named at its site above with its denominator; 0 added to pass or fail)"
echo "controls: $pass passed, $fail failed"
[ "$fail" -eq 0 ]
