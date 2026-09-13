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
elif ! grep -q 'WF-YAML RED' <<<"$wfy_bad_out"; then
  wfy_msg="MUST_FIRE UNREACHABLE (workflow YAML audit): planted file rc=$wfy_bad_rc but"
  wfy_msg="$wfy_msg the auditor never indicted it by name (no 'WF-YAML RED' line) --"
  wfy_msg="$wfy_msg red from a crash is not discrimination; output:"
  wfy_msg="$wfy_msg $(printf '%s\n' "$wfy_bad_out" | tr '\n' ' ')"
  no "$wfy_msg"
else
  wfy_msg="MUST_FIRE workflow YAML audit: planted malformed workflow (over-dedent out of"
  wfy_msg="$wfy_msg a '|' block scalar -- refused by PyYAML AND by the structural fallback)"
  wfy_msg="$wfy_msg was indicted by name (rc=$wfy_bad_rc):"
  wfy_msg="$wfy_msg $(grep -m1 'WF-YAML RED' <<<"$wfy_bad_out")"
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
#   self-test denominator: 19 of 19 controls (12 MUST_FIRE, 7 MUST_PASS)
# rc=0 alone is NOT the measurement: a self-test whose control set silently
# shrinks to 1 still exits 0, so the trailing "N of N controls" is parsed and
# held to a FLOOR of N >= 19 -- the claim counts what the self-test examined
# (doctrine 2), and a shrunken control set goes red here. If the wording of
# that line ever changes, THIS leg goes red and must be updated in the same
# commit -- unreadable is not empty, and unparseable is not passing.
#
# Floor history: 8 -> 14 (#243) -> 17 (#244, the three legs that check the
# census/gate exclusion handshake) -> 19 (#249's markup-blind control, which
# landed while the floor stayed at 17, plus #266's sentence-initial control).
# That gap is the comment below happening in practice: for one commit the leg
# would have let #249's control vanish without going red. A floor left at its historical value while
# the real count grows is not conservative, it is that many controls the leg
# would let disappear in silence.
if [ ! -r "checks/countables_drift.py" ]; then
  f238_msg="MUST_PASS FAILED (countables_drift self-test) UNMEASURED:"
  f238_msg="$f238_msg checks/countables_drift.py is not readable -- unreadable is not"
  f238_msg="$f238_msg empty (doctrine 4); the gate cannot run, so 0 of 19 controls were measured"
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
    f238_msg="$f238_msg own 19-control fixture set -- output:"
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
  elif [ "$f238_have" -lt 19 ]; then
    f238_msg="MUST_PASS FAILED (countables_drift self-test): control set shrank to"
    f238_msg="$f238_msg $f238_have of $f238_want, below the measured floor of 19 -- a self-test"
    f238_msg="$f238_msg that quietly drops controls still exits 0, so the floor is the control"
    no "$f238_msg"
  else
    f238_msg="MUST_PASS countables_drift self-test: rc=0 under python3 -S, denominator"
    f238_msg="$f238_msg $f238_have of $f238_want controls (>= the measured floor of 19): $f238_last"
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
  if [ "$f247_rc" -eq 5 ] && grep -q 'bare `ruff` in command position' <<<"$f247_out"; then
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
  if [ "$f247_rc" -eq 95 ] && grep -q '^UNMEASURED makefile_tooling:' <<<"$f247_out"; then
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

echo "== fix286-mirror: checks/makefile_ci_mirror.py real legs =="
# --- finding #286: the Makefile 'check' tree and .github/workflows/ci.yml
# are two lists of the same instrument panel, and they drifted in BOTH
# directions -- #230 landed a mypy widening on the Makefile side that CI
# never ran, and the reverse prefix defect sat in the workflow at the same
# time. The gate compares WHICH check scripts run on each side and is RED
# on any key that is single-sided.
#
# Two narrowings are deliberate and printed in a banner on every verdict:
# argv is NOT compared (the two sides legitimately differ in output paths
# and sharding), and *.sh suites are in no denominator (measured: ci.yml
# reaches two launcher suites through a loop variable, so widening to shell
# would emit false Makefile-only findings).
#
# Five legs, because the gate declares four states and a state that no leg
# ever reaches is a state that is written down rather than measured
# (#198/#200): the self-test, CLEAR on this tree with the self-wiring
# proof, RED in each drift direction, and UNMEASURED on a Makefile with no
# check: target.

# --- MUST_PASS: makefile_ci_mirror self-test ---------------------------------
# MEASURED on the commit that introduces the gate: `python3 -S
# checks/makefile_ci_mirror.py --self-test` exits 0 and its last line reads
# `SELF-TEST DENOMINATOR: 10 of 10 controls behaved; ...`. The floor is a
# floor: controls may be ADDED, never silently dropped, and a shrinking
# control set is how a detector quietly stops discriminating.
f286_floor=10
if [ ! -r "checks/makefile_ci_mirror.py" ]; then
  f286_msg="MUST_PASS FAILED (makefile_ci_mirror self-test) UNMEASURED:"
  f286_msg="$f286_msg checks/makefile_ci_mirror.py is not readable -- unreadable is not"
  f286_msg="$f286_msg empty and it is not clean (doctrine 4); 0 of 1 self-tests measured"
  no "$f286_msg"
else
  f286_rc=0
  f286_out=$(python3 -S checks/makefile_ci_mirror.py --self-test 2>&1) || f286_rc=$?
  f286_have=$(printf '%s\n' "$f286_out" |
    sed -n 's/^SELF-TEST DENOMINATOR: \([0-9][0-9]*\) of \([0-9][0-9]*\) controls behaved.*/\1/p')
  f286_tot=$(printf '%s\n' "$f286_out" |
    sed -n 's/^SELF-TEST DENOMINATOR: \([0-9][0-9]*\) of \([0-9][0-9]*\) controls behaved.*/\2/p')
  if [ "$f286_rc" -ne 0 ]; then
    f286_msg="MUST_PASS FAILED (makefile_ci_mirror self-test): rc=$f286_rc under python3 -S."
    f286_msg="$f286_msg A red self-test means the instrument no longer discriminates, so its"
    f286_msg="$f286_msg verdict on the live mirror is worth nothing. Output:"
    f286_msg="$f286_msg $(printf '%s\n' "$f286_out" | tr '\n' ' ')"
    no "$f286_msg"
  elif [ -z "$f286_have" ] || [ -z "$f286_tot" ]; then
    f286_msg="MUST_PASS FAILED (makefile_ci_mirror self-test) UNMEASURED: rc=0 but no"
    f286_msg="$f286_msg 'SELF-TEST DENOMINATOR: N of M controls behaved' line was found -- a"
    f286_msg="$f286_msg control suite that reports no denominator has not reported (doctrine 2)."
    f286_msg="$f286_msg Output: $(printf '%s\n' "$f286_out" | tr '\n' ' ')"
    no "$f286_msg"
  elif [ "$f286_have" != "$f286_tot" ]; then
    f286_msg="MUST_PASS FAILED (makefile_ci_mirror self-test): $f286_have of $f286_tot controls"
    f286_msg="$f286_msg behaved but rc=0 -- the exit code and the summary disagree, and one side"
    f286_msg="$f286_msg of the contract is lying (doctrine 6)"
    no "$f286_msg"
  elif [ "$f286_tot" -lt "$f286_floor" ]; then
    f286_msg="MUST_PASS FAILED (makefile_ci_mirror self-test): control set shrank to $f286_tot,"
    f286_msg="$f286_msg below the $f286_floor measured when the gate landed -- controls may be"
    f286_msg="$f286_msg added, never dropped, and a green run over a shrunken set is doctrine 1"
    no "$f286_msg"
  else
    f286_msg="MUST_PASS makefile_ci_mirror self-test: rc=0 under python3 -S, $f286_have of"
    f286_msg="$f286_msg $f286_tot controls behaved, at or above the $f286_floor-control floor"
    ok "$f286_msg"
  fi
fi

# --- MUST_PASS: the live tree's own verdict ----------------------------------
# The self-test proves the instrument discriminates on fixtures; this leg
# runs it against THIS repository's Makefile and ci.yml, which is the
# reading the #286 repair claims. MEASURED on the landing commit: rc=0 over
# 19 mirrored check scripts against 43 CI run steps.
#
# The strongest assertion is the self-wiring proof: the enumerated
# denominator must name checks/makefile_ci_mirror.py itself. The gate runs
# on both sides of the mirror, so it must see ITSELF in both lists -- the
# one thing a mirror gate wired into only one side could never report. A
# gate file entering checks/ with no call site on each side is the #86
# orphan class, which the fix78-orphan block below refuses in general; this
# leg refuses it for this gate in particular.
if [ ! -r "checks/makefile_ci_mirror.py" ]; then
  f286_msg="MUST_PASS FAILED (makefile_ci_mirror live verdict) UNMEASURED:"
  f286_msg="$f286_msg checks/makefile_ci_mirror.py is not readable -- unreadable is not empty"
  f286_msg="$f286_msg (doctrine 4); 0 of 1 live verdicts measured"
  no "$f286_msg"
else
  f286_rc=0
  f286_out=$(python3 -S checks/makefile_ci_mirror.py 2>&1) || f286_rc=$?
  f286_mk=$(printf '%s\n' "$f286_out" |
    sed -n 's/^CLEAR: \([0-9][0-9]*\) check scripts run on both sides of the mirror (Makefile .check. tree against \([0-9][0-9]*\) CI run steps).*/\1/p')
  f286_ci=$(printf '%s\n' "$f286_out" |
    sed -n 's/^CLEAR: \([0-9][0-9]*\) check scripts run on both sides of the mirror (Makefile .check. tree against \([0-9][0-9]*\) CI run steps).*/\2/p')
  if [ "$f286_rc" -ne 0 ]; then
    f286_msg="MUST_PASS FAILED (makefile_ci_mirror live verdict): rc=$f286_rc on this tree."
    f286_msg="$f286_msg rc=5 means a check script is single-sided again -- the #286/#230 drift"
    f286_msg="$f286_msg shape; rc=95 means the Makefile or ci.yml went unreadable or the check:"
    f286_msg="$f286_msg target vanished; rc=96 means the gate crashed, which is not a verdict."
    f286_msg="$f286_msg Output: $(printf '%s\n' "$f286_out" | tr '\n' ' ')"
    no "$f286_msg"
  elif [ -z "$f286_mk" ] || [ -z "$f286_ci" ]; then
    f286_msg="MUST_PASS FAILED (makefile_ci_mirror live verdict) UNMEASURED: rc=0 but no"
    f286_msg="$f286_msg 'CLEAR: N check scripts run on both sides of the mirror ...' line was"
    f286_msg="$f286_msg found -- unparseable is not passing (doctrine 2). Output:"
    f286_msg="$f286_msg $(printf '%s\n' "$f286_out" | tr '\n' ' ')"
    no "$f286_msg"
  elif [ "$f286_mk" -lt 1 ] || [ "$f286_ci" -lt 1 ]; then
    f286_msg="MUST_PASS FAILED (makefile_ci_mirror live verdict): denominator is $f286_mk"
    f286_msg="$f286_msg mirrored scripts against $f286_ci CI run steps -- zero units is"
    f286_msg="$f286_msg UNMEASURED, never a pass (doctrine 1)"
    no "$f286_msg"
  elif ! grep -q 'checks/makefile_ci_mirror\.py' <<<"$f286_out"; then
    f286_msg="MUST_PASS FAILED (makefile_ci_mirror live verdict): CLEAR over $f286_mk scripts"
    f286_msg="$f286_msg but the enumerated denominator does not name checks/makefile_ci_mirror.py"
    f286_msg="$f286_msg -- the gate does not see itself running on both sides, which is the #86"
    f286_msg="$f286_msg orphan class wearing a green verdict. Output:"
    f286_msg="$f286_msg $(printf '%s\n' "$f286_out" | tr '\n' ' ')"
    no "$f286_msg"
  else
    f286_msg="MUST_PASS makefile_ci_mirror live verdict: rc=0, $f286_mk check scripts mirrored"
    f286_msg="$f286_msg against $f286_ci CI run steps, and the denominator names the gate"
    f286_msg="$f286_msg itself -- the mirror is wired into both sides it watches"
    ok "$f286_msg"
  fi
fi

# --- MUST_FIRE: a Makefile-only script is RED --------------------------------
# The exemplar copies its gate into a throwaway tree because that gate
# resolves its Makefile from its own location; THIS gate takes explicit
# --makefile/--ci paths, so no copy is needed and the real tree is never
# mutated -- a control that mutates the tree it guards can leave the
# repository dirty on any early exit, and #239 was exactly a leg committed
# red against its own target.
#
# The fixture must define PY and a check: target: without them the gate
# would abstain at 95 for a reason that has nothing to do with the planted
# drift, and the leg would be measuring a staging defect rather than the
# gate's discrimination -- a fire leg that cannot reach the fire is a pass
# for the wrong reason waiting to happen.
#
# The fixture also runs ruff on BOTH sides, and that is load-bearing rather
# than scenery. A fixture carrying only the planted key drifts in both
# directions at once -- MEASURED: ci.yml's own `ruff check src` came back as
# a CI-only finding -- so the leg would claim to test one direction while
# firing on two, and could not tell a one-directional detector from a
# blanket "the two lists differ". With ruff mirrored, the planted key is the
# ONLY finding, and the shared key going unflagged is the discrimination:
# the gate separates the defect from its own fix.
#
# The must-NOT-fire assertion is anchored to the finding-line shape
# `^  CI-only:` rather than to the bare word. The banner this gate prints on
# every verdict discusses "Makefile-only findings on a correct tree" in
# prose, so an unanchored must-not-contain would match the gate's own
# explanation of itself and fail a correct leg.
f286_tmp=$(mktemp -d 2>/dev/null || mktemp -d -t fs286)
if [ ! -r "checks/makefile_ci_mirror.py" ] || [ -z "$f286_tmp" ] || [ ! -d "$f286_tmp" ]; then
  f286_msg="MUST_FIRE UNREACHABLE (makefile_ci_mirror Makefile-only drift) UNMEASURED: could"
  f286_msg="$f286_msg not stage a throwaway tree (gate readable? mktemp ok?) -- an unreachable"
  f286_msg="$f286_msg control is a declared state, not a silent pass (doctrine 5)"
  no "$f286_msg"
else
  printf 'PY := python3\n\ncheck: lint planted-gate\n\nlint:\n\t$(PY) -m ruff check src\n\nplanted-gate:\n\t$(PY) checks/planted_gate.py\n' \
    > "$f286_tmp/Makefile"
  printf 'name: ci\non: [push]\njobs:\n  lint:\n    steps:\n      - run: ruff check src\n' \
    > "$f286_tmp/ci.yml"
  f286_rc=0
  f286_out=$(python3 -S checks/makefile_ci_mirror.py \
    --makefile "$f286_tmp/Makefile" --ci "$f286_tmp/ci.yml" 2>&1) || f286_rc=$?
  if [ "$f286_rc" -eq 5 ] &&
     grep -q '^  Makefile-only: *checks/planted_gate\.py' <<<"$f286_out" &&
     ! grep -q '^  CI-only:' <<<"$f286_out"; then
    f286_msg="MUST_FIRE makefile_ci_mirror: a check: tree reaching '\$(PY)"
    f286_msg="$f286_msg checks/planted_gate.py' against a ci.yml that never runs it was scored"
    f286_msg="$f286_msg rc=5 (RED) and named under Makefile-only, while the ruff invocation"
    f286_msg="$f286_msg mirrored on both sides of the same fixture produced NO finding -- the"
    f286_msg="$f286_msg gate catches the #230 direction of the drift, Makefile ahead of CI, and"
    f286_msg="$f286_msg discriminates it from a correctly mirrored sibling"
    ok "$f286_msg"
  else
    f286_msg="MUST_FIRE UNREACHABLE (makefile_ci_mirror Makefile-only drift): rc=$f286_rc on a"
    f286_msg="$f286_msg Makefile whose check: tree runs checks/planted_gate.py and a ci.yml that"
    f286_msg="$f286_msg does not, expected exactly 5 with that key under Makefile-only and NO"
    f286_msg="$f286_msg CI-only line. rc=0 means the detector is blind to the #230 shape; rc=95"
    f286_msg="$f286_msg means the fixture lost its check: target or PY and the leg measured"
    f286_msg="$f286_msg staging, not the gate; a CI-only line means the mirrored ruff key was"
    f286_msg="$f286_msg ALSO flagged, so the leg would be firing on two directions while"
    f286_msg="$f286_msg claiming one. Output:"
    f286_msg="$f286_msg $(printf '%s\n' "$f286_out" | tr '\n' ' ')"
    no "$f286_msg"
  fi
  rm -rf "$f286_tmp"
fi

# --- MUST_FIRE: a CI-only script is RED (the other direction) ----------------
# The reverse direction needs its own leg because #286 drifted in BOTH
# directions at once: #230 landed a mypy widening in the Makefile and not
# in CI, and the reverse prefix defect sat in the workflow at the same
# time. A gate controlled in one direction only is half an instrument --
# the single-sided test must fire whichever side grew the extra key.
#
# Same fixture discipline as the leg above, mirrored: ruff runs on BOTH
# sides so the planted CI-only key is the only finding, and the must-NOT-
# fire assertion is anchored to `^  Makefile-only:` because that phrase also
# occurs in the gate's own banner prose.
f286_tmp=$(mktemp -d 2>/dev/null || mktemp -d -t fs286)
if [ ! -r "checks/makefile_ci_mirror.py" ] || [ -z "$f286_tmp" ] || [ ! -d "$f286_tmp" ]; then
  f286_msg="MUST_FIRE UNREACHABLE (makefile_ci_mirror CI-only drift) UNMEASURED: could not"
  f286_msg="$f286_msg stage a throwaway tree (gate readable? mktemp ok?) -- an unreachable"
  f286_msg="$f286_msg control is a declared state, not a silent pass (doctrine 5)"
  no "$f286_msg"
else
  printf 'PY := python3\n\ncheck: lint\n\nlint:\n\t$(PY) -m ruff check src\n' \
    > "$f286_tmp/Makefile"
  printf 'name: ci\non: [push]\njobs:\n  gates:\n    steps:\n      - run: ruff check src\n      - run: python3 tools/planted_ci_only.py\n' \
    > "$f286_tmp/ci.yml"
  f286_rc=0
  f286_out=$(python3 -S checks/makefile_ci_mirror.py \
    --makefile "$f286_tmp/Makefile" --ci "$f286_tmp/ci.yml" 2>&1) || f286_rc=$?
  if [ "$f286_rc" -eq 5 ] &&
     grep -q '^  CI-only: *tools/planted_ci_only\.py' <<<"$f286_out" &&
     ! grep -q '^  Makefile-only:' <<<"$f286_out"; then
    f286_msg="MUST_FIRE makefile_ci_mirror: a ci.yml step running tools/planted_ci_only.py"
    f286_msg="$f286_msg against a check: tree that never reaches it was scored rc=5 (RED) and"
    f286_msg="$f286_msg named under CI-only, with no Makefile-only line -- the gate catches the"
    f286_msg="$f286_msg reverse direction of the #286 drift, CI ahead of the Makefile, and the"
    f286_msg="$f286_msg two directions are separately reachable rather than one shared verdict"
    ok "$f286_msg"
  else
    f286_msg="MUST_FIRE UNREACHABLE (makefile_ci_mirror CI-only drift): rc=$f286_rc on a ci.yml"
    f286_msg="$f286_msg that runs tools/planted_ci_only.py and a Makefile that does not, expected"
    f286_msg="$f286_msg exactly 5 with that key under CI-only and NO Makefile-only line. rc=0"
    f286_msg="$f286_msg means the detector is blind to the reverse drift and is half an"
    f286_msg="$f286_msg instrument; rc=95 means the fixture lost its check: target and the leg"
    f286_msg="$f286_msg measured staging, not the gate. Output:"
    f286_msg="$f286_msg $(printf '%s\n' "$f286_out" | tr '\n' ' ')"
    no "$f286_msg"
  fi
  rm -rf "$f286_tmp"
fi

# --- MUST_FIRE: no check: target is UNMEASURED, not CLEAR --------------------
# A Makefile with recipes but no check: target gives the Makefile side zero
# units to mirror. `all([])` is True, so the natural implementation returns
# 0 and reads as "clean" -- a vacuous green over an empty denominator. This
# leg pins the refusal: the gate must exit 95 and say UNMEASURED, and 95
# must not be 0 (doctrine 1).
f286_tmp=$(mktemp -d 2>/dev/null || mktemp -d -t fs286)
if [ ! -r "checks/makefile_ci_mirror.py" ] || [ -z "$f286_tmp" ] || [ ! -d "$f286_tmp" ]; then
  f286_msg="MUST_FIRE UNREACHABLE (makefile_ci_mirror no check: target) UNMEASURED: could not"
  f286_msg="$f286_msg stage a throwaway tree -- unreachable is a declared state (doctrine 5)"
  no "$f286_msg"
else
  printf 'PY := python3\n\nlint:\n\t$(PY) -m ruff check src\n' > "$f286_tmp/Makefile"
  printf 'name: ci\non: [push]\njobs:\n  lint:\n    steps:\n      - run: ruff check src\n' \
    > "$f286_tmp/ci.yml"
  f286_rc=0
  f286_out=$(python3 -S checks/makefile_ci_mirror.py \
    --makefile "$f286_tmp/Makefile" --ci "$f286_tmp/ci.yml" 2>&1) || f286_rc=$?
  if [ "$f286_rc" -eq 95 ] && grep -q '^UNMEASURED:' <<<"$f286_out"; then
    f286_msg="MUST_FIRE makefile_ci_mirror: over a Makefile with no check: target the gate"
    f286_msg="$f286_msg exited 95 (UNMEASURED) and said so, rather than exiting 0 over an empty"
    f286_msg="$f286_msg denominator -- zero units on one side of the mirror is not a pass"
    f286_msg="$f286_msg (doctrine 1)"
    ok "$f286_msg"
  else
    f286_msg="MUST_FIRE UNREACHABLE (makefile_ci_mirror no check: target): rc=$f286_rc over a"
    f286_msg="$f286_msg Makefile with recipes but no check: target, expected exactly 95. rc=0 is"
    f286_msg="$f286_msg the vacuous truth itself -- a mirror reporting CLEAR with nothing on one"
    f286_msg="$f286_msg side. Output: $(printf '%s\n' "$f286_out" | tr '\n' ' ')"
    no "$f286_msg"
  fi
  rm -rf "$f286_tmp"
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
# defensive, and it was measured on the split commit: of the 11 helpers in the
# denominator THEN, 5 (countables_drift, makefile_tooling,
# packaging_reachability, training_plane_probe, wf_yaml_audit) had ZERO
# citations in test_launcher_contracts.sh and would have gone red on the first
# push -- and 4 more ran the other way, cited only in the launcher suite.
# RE-MEASURED 2026-09-11, when #286 added makefile_ci_mirror.py: the
# denominator is now 17, of which 11 are cited only in this suite and 4 only in
# the launcher suite. The split has widened, not closed, so reading $0 alone
# would now indict 11 helpers rather than 5. The counts are restated rather
# than left at the split-commit figures because a historical marker on a
# sentence that also reads as a present-tense claim shields nothing (#196);
# the CLAIM here -- neither suite is a sufficient corpus on its own -- is what
# the numbers are evidence for, and it got stronger.
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
  f78_orph_n=$(grep -m1 '^ORPH_HELPERS=' <<<"$f78_orph_real" | cut -d= -f2)
  f78_orph_orphs=$(grep -m1 '^ORPH_ORPHANS=' <<<"$f78_orph_real" | cut -d= -f2-)
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
    f78_orph_fn=$(grep -m1 '^ORPH_HELPERS=' <<<"$f78_orph_red" | cut -d= -f2)
    f78_orph_rset=$(grep -m1 '^ORPH_ORPHANS=' <<<"$f78_orph_red" | cut -d= -f2-)
    f78_orph_rset=$(printf '%s' "$f78_orph_rset" | tr -d ' ')
    if [ -z "$f78_orph_rset" ] || [ "$f78_orph_rset" = none ]; then f78_orph_rset=-; fi
    printf 'python3 launchers/%s  # planted call site (copy only)\n' \
      "$f78_orph_decoy" >> "$f78_orph_scopy"
    f78_orph_greenrc=0
    f78_orph_green=$(f78_orph_scan "$f78_orph_froot" "$f78_orph_scopy") || f78_orph_greenrc=$?
    f78_orph_gn=$(grep -m1 '^ORPH_HELPERS=' <<<"$f78_orph_green" | cut -d= -f2)
    f78_orph_gset=$(grep -m1 '^ORPH_ORPHANS=' <<<"$f78_orph_green" | cut -d= -f2-)
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
    grep -qF "ORPH_ORPHAN|$f78_orph_decoy" <<<"$f78_orph_red" \
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
#   SELF-TEST DENOMINATOR: 21 of 21 controls behaved; ...
# rc=0 alone is NOT the measurement: a control set that shrank to one control
# would still be capable of exiting 0. The trailing "N of N" tally is parsed,
# required to be non-empty and self-consistent, and held at N >= 21 -- the 11
# MUST_FIRE + 10 MUST_PASS controls present when this leg was last re-measured.
# A wording change reds THIS leg and must update it in the same commit.
#
# The floor moved 12 -> 16 when the freshness arm landed (#318), 16 -> 20 with
# the #385 drill, and 20 -> 21 with the --update band arm. A floor that lags
# the measured count is the #386 class: it is still a floor, so it stays green
# while silently admitting a shrunk control set, which is the one thing the
# tally exists to refuse. Re-seat it whenever a control lands. Four of the
# twenty-one drive freshness through an INJECTED parser resolver, not the real
# coverage.parser: this leg runs the gate under `python3 -S`, deliberately, so
# that its verdict cannot depend on what happens to be installed. A control
# that needs an optional third-party import is not a control here -- it reports
# UNMEASURED and shrinks the denominator, which is exactly what the tally is
# built to catch. The resolver seam keeps the ghost/orphan comparison -- the
# thing actually under test -- hermetic, and one of the four injects an absent
# resolver so "coverage.parser is missing" is proven REFUSED rather than
# waived, over a report that is CLEAR when the resolver is present.
if [ ! -r "checks/coverage_floor.py" ]; then
  f252_msg="MUST_PASS FAILED (coverage_floor self-test) UNMEASURED:"
  f252_msg="$f252_msg checks/coverage_floor.py is not readable -- unreadable is not"
  f252_msg="$f252_msg empty; the gate cannot run, so 0 of its declared denominator of 21"
  f252_msg="$f252_msg controls (11 MUST_FIRE + 10 MUST_PASS) were measured. An unreadable"
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
    f252_msg="$f252_msg declared denominator of 21 controls (11 MUST_FIRE + 10 MUST_PASS);"
    f252_msg="$f252_msg 0 of 21 controls were accepted as behaved, so the leg fails closed."
    f252_msg="$f252_msg Output: $(printf '%s\n' "$f252_out" | tr '\n' ' ')"
    no "$f252_msg"
  elif [ -z "$f252_have" ] || [ -z "$f252_want" ]; then
    f252_msg="MUST_PASS FAILED (coverage_floor self-test) UNMEASURED: rc=0 but the last"
    f252_msg="$f252_msg line carries no parseable 'SELF-TEST DENOMINATOR: N of N controls"
    f252_msg="$f252_msg behaved' tally -- the measuring unit printed no denominator, so 0 of"
    f252_msg="$f252_msg 21 declared controls are auditable here. Unparseable is not passing;"
    f252_msg="$f252_msg fail closed and update this leg in the same commit as the wording"
    f252_msg="$f252_msg change. Last line: $f252_last"
    no "$f252_msg"
  elif [ "$f252_have" -ne "$f252_want" ]; then
    f252_msg="MUST_PASS FAILED (coverage_floor self-test): denominator $f252_have of"
    f252_msg="$f252_msg $f252_want controls is not self-consistent -- the self-test examined"
    f252_msg="$f252_msg fewer controls than it claims to have, over the declared denominator"
    f252_msg="$f252_msg of 21. The inconsistency is failed closed because rc=0 cannot certify"
    f252_msg="$f252_msg a partial control set."
    no "$f252_msg"
  elif [ "$f252_have" -lt 21 ]; then
    f252_msg="MUST_PASS FAILED (coverage_floor self-test): control set shrank to"
    f252_msg="$f252_msg $f252_have of $f252_want, below the measured floor of 21 controls"
    f252_msg="$f252_msg (11 MUST_FIRE + 10 MUST_PASS). A shortened self-test can still exit 0,"
    f252_msg="$f252_msg so the floor is the control and this leg fails closed."
    no "$f252_msg"
  else
    f252_msg="MUST_PASS coverage_floor self-test: rc=0 under python3 -S, denominator"
    f252_msg="$f252_msg $f252_have of $f252_want controls (>= the measured floor of 21,"
    f252_msg="$f252_msg 11 MUST_FIRE + 10 MUST_PASS): $f252_last"
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

# --- MUST_PASS: doc-pointer gate self-test (checks/doc_pointers.py) ----------
# Finding #281: the README is a contents page and 17 of the chapters it points
# at did not exist. Nothing was red, because a pointer is a DECLARATION and the
# only gate over declarations was gate_stage_orphans, which reads build stages.
#
# Same floor convention as the f238 leg above and for the same reason: rc=0 is
# not the measurement. A self-test whose control set silently shrinks to 1 still
# exits 0, so the trailing "N of N controls" is parsed and held to a FLOOR.
#
# The floor covers BOTH control families -- 17 end-to-end cases and 4 extractor
# controls -- because it is the extractor controls that guard against #186
# recurring here, and they are the ones a floor over end-to-end cases alone
# would let disappear. That is not hypothetical: this gate's first version
# anchored the arrow notation on a closing bracket, so two of the README's own
# pointers (the ones carrying a prose tail) sat in no denominator while it
# printed a green. The extractor controls assert the exact (notation, target,
# line) tuples, which is the only assertion that would have caught it -- an
# rc-only control cannot tell "found the right thing" from "found a different
# thing that is also RED".
#
# Floor history: 21 at introduction (#281).
if [ ! -r "checks/doc_pointers.py" ]; then
  f281_msg="MUST_PASS FAILED (doc_pointers self-test) UNMEASURED:"
  f281_msg="$f281_msg checks/doc_pointers.py is not readable -- unreadable is not empty"
  f281_msg="$f281_msg (doctrine 4); the gate cannot run, so 0 of 21 controls were measured"
  no "$f281_msg"
else
  f281_rc=0
  f281_out=$(python3 -S checks/doc_pointers.py --self-test 2>&1) || f281_rc=$?
  f281_last=$(printf '%s\n' "$f281_out" | tail -n 1)
  f281_have=$(printf '%s\n' "$f281_last" |
    sed -n 's/^self-test denominator: \([0-9][0-9]*\) of \([0-9][0-9]*\) controls.*/\1/p')
  f281_want=$(printf '%s\n' "$f281_last" |
    sed -n 's/^self-test denominator: \([0-9][0-9]*\) of \([0-9][0-9]*\) controls.*/\2/p')
  if [ "$f281_rc" -ne 0 ]; then
    f281_msg="MUST_PASS FAILED (doc_pointers self-test): rc=$f281_rc over the gate's own"
    f281_msg="$f281_msg 21-control fixture set -- output:"
    f281_msg="$f281_msg $(printf '%s\n' "$f281_out" | tr '\n' ' ')"
    no "$f281_msg"
  elif [ -z "$f281_have" ] || [ -z "$f281_want" ]; then
    f281_msg="MUST_PASS FAILED (doc_pointers self-test) UNMEASURED: rc=0 but the last"
    f281_msg="$f281_msg line is not the declared 'self-test denominator: N of N controls'"
    f281_msg="$f281_msg wording -- the measuring unit printed no denominator (doctrine 2);"
    f281_msg="$f281_msg update this leg in the same commit as the wording change."
    f281_msg="$f281_msg Last line: $f281_last"
    no "$f281_msg"
  elif [ "$f281_have" -ne "$f281_want" ]; then
    f281_msg="MUST_PASS FAILED (doc_pointers self-test): denominator $f281_have of"
    f281_msg="$f281_msg $f281_want controls is not self-consistent -- the self-test examined"
    f281_msg="$f281_msg fewer controls than it claims to have (doctrine 2)"
    no "$f281_msg"
  elif [ "$f281_have" -lt 21 ]; then
    f281_msg="MUST_PASS FAILED (doc_pointers self-test): control set shrank to $f281_have"
    f281_msg="$f281_msg of $f281_want, below the measured floor of 21 -- a self-test that"
    f281_msg="$f281_msg quietly drops controls still exits 0, so the floor is the control"
    no "$f281_msg"
  else
    f281_msg="MUST_PASS doc_pointers self-test: rc=0 under python3 -S, denominator"
    f281_msg="$f281_msg $f281_have of $f281_want controls (>= the measured floor of 21):"
    f281_msg="$f281_msg $f281_last"
    ok "$f281_msg"
  fi
fi

# --- MUST_FIRE: doc_pointers discriminates a dangle from a resolved pointer --
# The self-test above exercises the gate's INTERNAL fixture path. This leg
# exercises the shipped CLI over a real git tree, and it is a discrimination
# PAIR on purpose: the same corpus, differing only in whether the pointed-at
# file is in the index, must produce 5 (RED) and then 0 (CLEAR). A gate that is
# merely stuck red satisfies the first arm and fails the second, so one arm
# alone would not prove the gate discriminates.
#
# The assertion is rc=5 exactly, not merely nonzero: collapsing it to nonzero
# would accept a crash (rc=1/2) or a REFUSE (96) as the control firing, and a
# crashed detector is not a discriminating one.
if [ ! -r "checks/doc_pointers.py" ]; then
  f281b_msg="MUST_FIRE FAILED (doc_pointers dangle discrimination) UNMEASURED:"
  f281b_msg="$f281b_msg checks/doc_pointers.py is not readable -- unreadable is not empty"
  f281b_msg="$f281b_msg (doctrine 4); 0 of 2 discrimination arms were measured"
  no "$f281b_msg"
elif ! command -v git >/dev/null 2>&1; then
  f281b_msg="MUST_FIRE FAILED (doc_pointers dangle discrimination) UNMEASURED: git is not"
  f281b_msg="$f281b_msg on PATH, and both the gate's corpus and its resolution universe are"
  f281b_msg="$f281b_msg git's index -- the fixture cannot be built, so 0 of 2 arms ran"
  no "$f281b_msg"
else
  f281b_tmp=$(mktemp -d)
  mkdir -p "$f281b_tmp/docs"
  printf 'Contents page.\n\n[-> docs/PLANTED.md]\n' > "$f281b_tmp/README.md"
  git -C "$f281b_tmp" init -q >/dev/null 2>&1
  git -C "$f281b_tmp" add -A >/dev/null 2>&1
  f281b_dangle_rc=0
  f281b_dangle_out=$(python3 -S checks/doc_pointers.py "$f281b_tmp" 2>&1) || f281b_dangle_rc=$?
  # Identical corpus, one file added to the index. Nothing else changes.
  printf '# Planted\n' > "$f281b_tmp/docs/PLANTED.md"
  git -C "$f281b_tmp" add -A >/dev/null 2>&1
  f281b_clear_rc=0
  f281b_clear_out=$(python3 -S checks/doc_pointers.py "$f281b_tmp" 2>&1) || f281b_clear_rc=$?
  rm -rf "$f281b_tmp"
  if [ "$f281b_dangle_rc" -ne 5 ]; then
    f281b_msg="MUST_FIRE FAILED (doc_pointers dangle discrimination): the planted"
    f281b_msg="$f281b_msg [-> docs/PLANTED.md] with no such path in the index gave"
    f281b_msg="$f281b_msg rc=$f281b_dangle_rc, expected exactly 5 (RED) over a denominator of"
    f281b_msg="$f281b_msg 1 pointer in 1 tracked *.md. rc=0 would launder a dangling pointer"
    f281b_msg="$f281b_msg into CLEAR and any other nonzero is not this gate's declared RED."
    f281b_msg="$f281b_msg Output: $(printf '%s\n' "$f281b_dangle_out" | tr '\n' ' ')"
    no "$f281b_msg"
  elif [ "$f281b_clear_rc" -ne 0 ]; then
    f281b_msg="MUST_FIRE FAILED (doc_pointers dangle discrimination): the dangling arm fired"
    f281b_msg="$f281b_msg correctly at rc=5, but staging docs/PLANTED.md -- the ONLY change --"
    f281b_msg="$f281b_msg still gave rc=$f281b_clear_rc instead of 0. Only 1 of 2 arms held; a"
    f281b_msg="$f281b_msg gate that reddens a corpus whose every pointer resolves is stuck RED,"
    f281b_msg="$f281b_msg not discriminating, so the leg fails closed. Output:"
    f281b_msg="$f281b_msg $(printf '%s\n' "$f281b_clear_out" | tr '\n' ' ')"
    no "$f281b_msg"
  else
    f281b_msg="MUST_FIRE doc_pointers dangle discrimination: over 1 pointer in 1 tracked"
    f281b_msg="$f281b_msg *.md the shipped CLI exited rc=5 (RED) while docs/PLANTED.md was"
    f281b_msg="$f281b_msg absent from the index and rc=0 (CLEAR) once it was staged -- the only"
    f281b_msg="$f281b_msg difference between the arms. Both discrimination outcomes held"
    ok "$f281b_msg"
  fi
fi

# --- MUST_PASS: citation-lines gate self-test (checks/citation_lines.py) -----
# Finding #303: prose cited lines that were not there -- path.ext:N and
# path.ext:N-M tokens whose N (or whose M) ran past the end of the target --
# and nothing was red, because a citation is a DECLARATION and no gate resolved
# the line half of one. This gate proves EXISTENCE only: that the cited line is
# there, never that it says what the prose claims.
#
# Same floor convention as the f281 leg above and for the same reason: rc=0 is
# not the measurement. A self-test whose control set silently shrinks to 1 still
# exits 0, so the trailing "N of N controls" is parsed and held to a FLOOR.
#
# The floor covers ALL THREE control families -- 3 MUST_FIRE (out-of-range,
# range-M-past-end, malformed N>M), 4 MUST_PASS (last-line boundary, ambiguous
# basename left unresolved, out-of-repo path unresolved, bare :NNN declared
# unmeasured), and 1 control ON THE HARNESS -- because the harness control is
# the one that keeps the other seven honest. It re-runs the MUST_FIRE controls
# with the range comparison neutered and asserts they all go silent, and that
# is why an rc-only control would be insufficient here: without it, a
# comparison that always reports "in range" passes every MUST_PASS and the gate
# reads green while measuring nothing.
#
# Floor history: 8 at introduction (#303).
if [ ! -r "checks/citation_lines.py" ]; then
  f303_msg="MUST_PASS FAILED (citation_lines self-test) UNMEASURED:"
  f303_msg="$f303_msg checks/citation_lines.py is not readable -- unreadable is not empty"
  f303_msg="$f303_msg (doctrine 4); the gate cannot run, so 0 of 8 controls were measured"
  no "$f303_msg"
else
  f303_rc=0
  f303_out=$(python3 -S checks/citation_lines.py --self-test 2>&1) || f303_rc=$?
  f303_last=$(printf '%s\n' "$f303_out" | tail -n 1)
  f303_have=$(printf '%s\n' "$f303_last" |
    sed -n 's/^self-test denominator: \([0-9][0-9]*\) of \([0-9][0-9]*\) controls.*/\1/p')
  f303_want=$(printf '%s\n' "$f303_last" |
    sed -n 's/^self-test denominator: \([0-9][0-9]*\) of \([0-9][0-9]*\) controls.*/\2/p')
  if [ "$f303_rc" -ne 0 ]; then
    f303_msg="MUST_PASS FAILED (citation_lines self-test): rc=$f303_rc over the gate's own"
    f303_msg="$f303_msg 8-control fixture set -- output:"
    f303_msg="$f303_msg $(printf '%s\n' "$f303_out" | tr '\n' ' ')"
    no "$f303_msg"
  elif [ -z "$f303_have" ] || [ -z "$f303_want" ]; then
    f303_msg="MUST_PASS FAILED (citation_lines self-test) UNMEASURED: rc=0 but the last"
    f303_msg="$f303_msg line is not the declared 'self-test denominator: N of N controls'"
    f303_msg="$f303_msg wording -- the measuring unit printed no denominator (doctrine 2);"
    f303_msg="$f303_msg update this leg in the same commit as the wording change."
    f303_msg="$f303_msg Last line: $f303_last"
    no "$f303_msg"
  elif [ "$f303_have" -ne "$f303_want" ]; then
    f303_msg="MUST_PASS FAILED (citation_lines self-test): denominator $f303_have of"
    f303_msg="$f303_msg $f303_want controls is not self-consistent -- the self-test examined"
    f303_msg="$f303_msg fewer controls than it claims to have (doctrine 2)"
    no "$f303_msg"
  elif [ "$f303_have" -lt 8 ]; then
    f303_msg="MUST_PASS FAILED (citation_lines self-test): control set shrank to $f303_have"
    f303_msg="$f303_msg of $f303_want, below the measured floor of 8 -- a self-test that"
    f303_msg="$f303_msg quietly drops controls still exits 0, so the floor is the control"
    no "$f303_msg"
  else
    f303_msg="MUST_PASS citation_lines self-test: rc=0 under python3 -S, denominator"
    f303_msg="$f303_msg $f303_have of $f303_want controls (>= the measured floor of 8):"
    f303_msg="$f303_msg $f303_last"
    ok "$f303_msg"
  fi
fi

# --- MUST_FIRE: citation_lines discriminates a past-end line from a real one -
# The self-test above exercises the gate's INTERNAL fixture path. This leg
# exercises the shipped CLI over a real git tree, and it is a discrimination
# PAIR on purpose: the same corpus, differing only in whether the cited line
# exists, must produce 5 (RED) and then 0 (CLEAR). A gate that is merely stuck
# red satisfies the first arm and fails the second, so one arm alone would not
# prove the gate discriminates.
#
# The assertion is rc=5 exactly, not merely nonzero: collapsing it to nonzero
# would accept a crash (rc=1/2) or a REFUSE (96) as the control firing, and a
# crashed detector is not a discriminating one.
#
# The CLEAR arm extends the TARGET from two lines to five rather than editing
# the CITATION, and that is deliberate: it holds the cited token target.py:5
# fixed across both arms, so the difference in verdict is attributable to the
# target's length and to nothing else.
if [ ! -r "checks/citation_lines.py" ]; then
  f303b_msg="MUST_FIRE FAILED (citation_lines past-end discrimination) UNMEASURED:"
  f303b_msg="$f303b_msg checks/citation_lines.py is not readable -- unreadable is not empty"
  f303b_msg="$f303b_msg (doctrine 4); 0 of 2 discrimination arms were measured"
  no "$f303b_msg"
elif ! command -v git >/dev/null 2>&1; then
  f303b_msg="MUST_FIRE FAILED (citation_lines past-end discrimination) UNMEASURED: git is"
  f303b_msg="$f303b_msg not on PATH, and both the gate's corpus and its resolution universe"
  f303b_msg="$f303b_msg are git's index -- the fixture cannot be built, so 0 of 2 arms ran"
  no "$f303b_msg"
else
  f303b_tmp=$(mktemp -d)
  mkdir -p "$f303b_tmp/docs"
  printf 'alpha = 1\nbeta = 2\n' > "$f303b_tmp/target.py"
  printf 'See target.py:5 for the detail.\n' > "$f303b_tmp/docs/a.md"
  git -C "$f303b_tmp" init -q >/dev/null 2>&1
  git -C "$f303b_tmp" add -A >/dev/null 2>&1
  f303b_red_rc=0
  f303b_red_out=$(python3 -S checks/citation_lines.py "$f303b_tmp" 2>&1) || f303b_red_rc=$?
  # Identical corpus, the target extended from two lines to five and re-staged.
  # The citing text is not touched. Nothing else changes.
  printf 'alpha = 1\nbeta = 2\ngamma = 3\ndelta = 4\neps = 5\n' > "$f303b_tmp/target.py"
  git -C "$f303b_tmp" add -A >/dev/null 2>&1
  f303b_clear_rc=0
  f303b_clear_out=$(python3 -S checks/citation_lines.py "$f303b_tmp" 2>&1) || f303b_clear_rc=$?
  rm -rf "$f303b_tmp"
  if [ "$f303b_red_rc" -ne 5 ]; then
    f303b_msg="MUST_FIRE FAILED (citation_lines past-end discrimination): the planted"
    f303b_msg="$f303b_msg target.py:5 over a 2-line target gave rc=$f303b_red_rc, expected"
    f303b_msg="$f303b_msg exactly 5 (RED) over a denominator of 1 citation in 2 tracked"
    f303b_msg="$f303b_msg files. rc=0 would launder a past-end citation into CLEAR and any"
    f303b_msg="$f303b_msg other nonzero is not this gate's declared RED."
    f303b_msg="$f303b_msg Output: $(printf '%s\n' "$f303b_red_out" | tr '\n' ' ')"
    no "$f303b_msg"
  elif [ "$f303b_clear_rc" -ne 0 ]; then
    f303b_msg="MUST_FIRE FAILED (citation_lines past-end discrimination): the past-end arm"
    f303b_msg="$f303b_msg fired correctly at rc=5, but extending target.py to five lines --"
    f303b_msg="$f303b_msg the ONLY change -- still gave rc=$f303b_clear_rc instead of 0. Only"
    f303b_msg="$f303b_msg 1 of 2 arms held; a gate that reddens a corpus whose every citation"
    f303b_msg="$f303b_msg names an existing line is stuck RED, not discriminating, so the leg"
    f303b_msg="$f303b_msg fails closed. Output:"
    f303b_msg="$f303b_msg $(printf '%s\n' "$f303b_clear_out" | tr '\n' ' ')"
    no "$f303b_msg"
  else
    f303b_msg="MUST_FIRE citation_lines past-end discrimination: over 1 citation in 2"
    f303b_msg="$f303b_msg tracked files the shipped CLI exited rc=5 (RED) while target.py had"
    f303b_msg="$f303b_msg 2 lines and rc=0 (CLEAR) once it held 5 -- the only difference"
    f303b_msg="$f303b_msg between the arms, with the cited token target.py:5 untouched. Both"
    f303b_msg="$f303b_msg discrimination outcomes held"
    ok "$f303b_msg"
  fi
fi

# --- MUST_PASS: mutation-scope gate self-test (checks/mutation_scope.py) -----
# Finding #317: the mutation battery's module scope was UNDECLARED. tools/mutate.py's
# MODULE_PATHS covered 9 files; the git index tracked 47; the other 38 sat in no set and
# no gate could say so. A pinned COUNT (tests/tooling/test_mutation_anchor_freshness.py
# pins MODULE_PATHS at 9) is not a denominator: it stops the map shrinking, and says
# nothing about the tree it is supposed to reach. #316 is the proof -- 444 new lines of
# adjudication logic landed in the largest unmapped library file and every gate stayed
# green. This gate partitions the index into covered + pending + out-of-scope and reds
# on undeclared, double-membership, stale, empty-reason and pasteable-reason.
#
# Same floor convention as the f252 and f303 legs above, for the same reason: rc=0 is not
# the measurement. The trailing tally is parsed and held to a FLOOR of 13 so a self-test
# that quietly drops controls cannot still read green.
#
# The floor spans both families -- 6 MUST_FIRE (one per rule R1-R5, with R3 twice because
# a stale path can come from PENDING_ENROLMENT or from MODULE_PATHS and those are
# different code paths) and 7 MUST_PASS (clean synthetic partition, all three UNMEASURED
# arms, the abstain path, the live shipped tree, and the --mutate-py flag). The last of
# those is a control ON THE HARNESS: it is what the MUST_FIRE leg below depends on, and a
# flag that parsed but was never read would leave that leg measuring the shipped tree
# while believing it measured a doctored one.
#
# Floor history: 13 at introduction (#317).
if [ ! -r "checks/mutation_scope.py" ]; then
  f317_msg="MUST_PASS FAILED (mutation_scope self-test) UNMEASURED:"
  f317_msg="$f317_msg checks/mutation_scope.py is not readable -- unreadable is not empty"
  f317_msg="$f317_msg (doctrine 4); the gate cannot run, so 0 of 13 controls were measured"
  no "$f317_msg"
else
  f317_rc=0
  f317_out=$(python3 -S checks/mutation_scope.py --self-test 2>&1) || f317_rc=$?
  f317_last=$(printf '%s\n' "$f317_out" | tail -n 1)
  f317_have=$(printf '%s\n' "$f317_last" |
    sed -n 's/^SELF-TEST DENOMINATOR: \([0-9][0-9]*\) of \([0-9][0-9]*\) controls behaved;.*/\1/p')
  f317_want=$(printf '%s\n' "$f317_last" |
    sed -n 's/^SELF-TEST DENOMINATOR: \([0-9][0-9]*\) of \([0-9][0-9]*\) controls behaved;.*/\2/p')
  if [ "$f317_rc" -ne 0 ]; then
    f317_msg="MUST_PASS FAILED (mutation_scope self-test): rc=$f317_rc over the gate's"
    f317_msg="$f317_msg declared denominator of 13 controls (6 MUST_FIRE + 7 MUST_PASS);"
    f317_msg="$f317_msg 0 of 13 are accepted as behaved, so the leg fails closed. Output:"
    f317_msg="$f317_msg $(printf '%s\n' "$f317_out" | tr '\n' ' ')"
    no "$f317_msg"
  elif [ -z "$f317_have" ] || [ -z "$f317_want" ]; then
    f317_msg="MUST_PASS FAILED (mutation_scope self-test) UNMEASURED: rc=0 but the last"
    f317_msg="$f317_msg line carries no parseable 'SELF-TEST DENOMINATOR: N of N controls"
    f317_msg="$f317_msg behaved' tally -- the measuring unit printed no denominator, so 0 of"
    f317_msg="$f317_msg 13 declared controls are auditable here. Unparseable is not passing;"
    f317_msg="$f317_msg fail closed and update this leg in the same commit as the wording"
    f317_msg="$f317_msg change. Last line: $f317_last"
    no "$f317_msg"
  elif [ "$f317_have" -ne "$f317_want" ]; then
    f317_msg="MUST_PASS FAILED (mutation_scope self-test): denominator $f317_have of"
    f317_msg="$f317_msg $f317_want controls is not self-consistent -- the self-test examined"
    f317_msg="$f317_msg fewer controls than it claims to have. rc=0 cannot certify a partial"
    f317_msg="$f317_msg control set, so the inconsistency fails closed."
    no "$f317_msg"
  elif [ "$f317_have" -lt 13 ]; then
    f317_msg="MUST_PASS FAILED (mutation_scope self-test): control set shrank to $f317_have"
    f317_msg="$f317_msg of $f317_want, below the measured floor of 13 (6 MUST_FIRE +"
    f317_msg="$f317_msg 7 MUST_PASS). A shortened self-test still exits 0, so the floor is"
    f317_msg="$f317_msg the control and this leg fails closed."
    no "$f317_msg"
  else
    f317_msg="MUST_PASS mutation_scope self-test: rc=0 under python3 -S, denominator"
    f317_msg="$f317_msg $f317_have of $f317_want controls (>= the measured floor of 13,"
    f317_msg="$f317_msg 6 MUST_FIRE + 7 MUST_PASS): $f317_last"
    ok "$f317_msg"
  fi
fi

# --- MUST_FIRE: mutation_scope discriminates an undeclared file from a declared one ---
# The self-test above exercises the gate's INTERNAL fixture path over synthetic
# declarations. This leg exercises the shipped CLI over the REAL git index and the REAL
# shipped declaration constants, and it is a discrimination PAIR: the same tree, the same
# 47 files, differing ONLY in whether one path is present in MODULE_PATHS, must give 5
# (RED) and then 0 (CLEAR).
#
# A synthetic tree is deliberately NOT used here. PENDING_ENROLMENT and OUT_OF_SCOPE are
# pinned module constants naming 38 real repository paths, so pointed at a temporary tree
# the gate would correctly fire R3 on all of them and the arms would differ by far more
# than one variable. The one input that is legitimately variable is the covered set, and
# --mutate-py is what exposes it. Nothing in the working tree is touched: both arms read
# COPIES in a temp directory, so a killed suite cannot strand a mutated tools/mutate.py
# (the mutation-battery lesson).
#
# The assertion is rc=5 exactly, not merely nonzero: nonzero would accept a crash (1/2) or
# an UNMEASURED (95) or a REFUSE (96) as the control firing, and none of those is this
# gate's declared RED. The RED arm additionally asserts that R1 names THE PATH THAT WAS
# REMOVED -- a gate that reddens for some other reason would satisfy an rc-only check
# while measuring something else entirely.
if [ ! -r "checks/mutation_scope.py" ]; then
  f317b_msg="MUST_FIRE FAILED (mutation_scope undeclared discrimination) UNMEASURED:"
  f317b_msg="$f317b_msg checks/mutation_scope.py is not readable -- unreadable is not empty"
  f317b_msg="$f317b_msg (doctrine 4); 0 of 2 discrimination arms were measured"
  no "$f317b_msg"
elif [ ! -r "tools/mutate.py" ]; then
  f317b_msg="MUST_FIRE FAILED (mutation_scope undeclared discrimination) UNMEASURED:"
  f317b_msg="$f317b_msg tools/mutate.py is not readable, and it is the covered-set source"
  f317b_msg="$f317b_msg both arms are built from -- the fixture cannot exist, so 0 of 2 ran"
  no "$f317b_msg"
else
  f317b_tmp=$(mktemp -d)
  # Drop exactly ONE entry from a COPY of MODULE_PATHS. The key is chosen by sort order,
  # not hard-coded, so the leg cannot go stale the day the map is re-ordered or renamed.
  f317b_dropped=$(python3 -S - "$f317b_tmp" <<'F317B_PY'
import ast, sys
from pathlib import Path

tmp = Path(sys.argv[1])
src = Path("tools/mutate.py").read_text(encoding="utf-8")
for node in ast.walk(ast.parse(src)):
    name = None
    if isinstance(node, ast.Assign):
        name = next((t.id for t in node.targets if isinstance(t, ast.Name)), None)
    elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
        name = node.target.id
    if name != "MODULE_PATHS" or node.value is None:
        continue
    mapping = ast.literal_eval(node.value)
    if not mapping:
        break
    key = sorted(mapping)[0]
    lines = src.splitlines(keepends=True)
    kept = [ln for ln in lines if not (f'"{key}"' in ln and mapping[key] in ln)]
    if len(kept) == len(lines):
        break  # the entry is not on one line; say nothing rather than mis-doctor
    (tmp / "intact.py").write_text(src, encoding="utf-8")
    (tmp / "doctored.py").write_text("".join(kept), encoding="utf-8")
    print(mapping[key])
    break
F317B_PY
  ) || f317b_dropped=""
  if [ -z "$f317b_dropped" ] || [ ! -r "$f317b_tmp/doctored.py" ] || [ ! -r "$f317b_tmp/intact.py" ]; then
    rm -rf "$f317b_tmp"
    f317b_msg="MUST_FIRE FAILED (mutation_scope undeclared discrimination) UNMEASURED: the"
    f317b_msg="$f317b_msg one-entry-removed copy of tools/mutate.py could not be built --"
    f317b_msg="$f317b_msg MODULE_PATHS is absent, empty, or its first entry is not on a"
    f317b_msg="$f317b_msg single line. Mis-doctoring would make the arms differ by more than"
    f317b_msg="$f317b_msg the one variable, so the leg abstains rather than guess; 0 of 2 arms"
    f317b_msg="$f317b_msg ran."
    no "$f317b_msg"
  else
    f317b_red_rc=0
    f317b_red_out=$(python3 -S checks/mutation_scope.py --mutate-py "$f317b_tmp/doctored.py" 2>&1) ||
      f317b_red_rc=$?
    f317b_clear_rc=0
    f317b_clear_out=$(python3 -S checks/mutation_scope.py --mutate-py "$f317b_tmp/intact.py" 2>&1) ||
      f317b_clear_rc=$?
    rm -rf "$f317b_tmp"
    if [ "$f317b_red_rc" -ne 5 ]; then
      f317b_msg="MUST_FIRE FAILED (mutation_scope undeclared discrimination): removing"
      f317b_msg="$f317b_msg $f317b_dropped from a copy of MODULE_PATHS gave"
      f317b_msg="$f317b_msg rc=$f317b_red_rc, expected exactly 5 (RED). rc=0 would launder an"
      f317b_msg="$f317b_msg undeclared tracked file into CLEAR -- the whole of #317 -- and any"
      f317b_msg="$f317b_msg other nonzero is not this gate's declared RED. Output:"
      f317b_msg="$f317b_msg $(printf '%s\n' "$f317b_red_out" | tr '\n' ' ')"
      no "$f317b_msg"
    elif ! grep -q "R1 UNDECLARED: $f317b_dropped" <<<"$f317b_red_out"; then
      f317b_msg="MUST_FIRE FAILED (mutation_scope undeclared discrimination): the gate"
      f317b_msg="$f317b_msg exited 5, but its R1 row does not name $f317b_dropped -- the one"
      f317b_msg="$f317b_msg path removed. A RED attributed to some other file is not this"
      f317b_msg="$f317b_msg control firing, and rc alone cannot tell the two apart. Output:"
      f317b_msg="$f317b_msg $(printf '%s\n' "$f317b_red_out" | tr '\n' ' ')"
      no "$f317b_msg"
    elif [ "$f317b_clear_rc" -ne 0 ]; then
      f317b_msg="MUST_FIRE FAILED (mutation_scope undeclared discrimination): the undeclared"
      f317b_msg="$f317b_msg arm fired correctly at rc=5 naming $f317b_dropped, but the"
      f317b_msg="$f317b_msg UNMODIFIED copy of the same file -- the only difference -- still"
      f317b_msg="$f317b_msg gave rc=$f317b_clear_rc instead of 0. Only 1 of 2 arms held; a gate"
      f317b_msg="$f317b_msg that reddens a fully-declared partition is stuck RED, not"
      f317b_msg="$f317b_msg discriminating, so the leg fails closed. Output:"
      f317b_msg="$f317b_msg $(printf '%s\n' "$f317b_clear_out" | tr '\n' ' ')"
      no "$f317b_msg"
    else
      f317b_msg="MUST_FIRE mutation_scope undeclared discrimination: over the live 47-file"
      f317b_msg="$f317b_msg index and the shipped declarations, a copy of tools/mutate.py"
      f317b_msg="$f317b_msg missing $f317b_dropped exited rc=5 with R1 naming that exact path,"
      f317b_msg="$f317b_msg and the unmodified copy exited rc=0 -- the covered set the only"
      f317b_msg="$f317b_msg variable, the working tree untouched. Both outcomes held"
      ok "$f317b_msg"
    fi
  fi
fi

# --- rl-static-subtypes gate (checks/rl_static_subtypes.py): three deterministic controls ---
# Finding #360: nothing in the tree proved a shipped RL binding is a STATIC subtype of
# the protocol it claims to implement. Duck-typed agreement at the call site is not
# substitutability: a binding can satisfy every caller today and still drift off the
# protocol the day a consumer reaches for a member the binding never declared. This gate
# runs mypy over every (binding, protocol) pair in the registry and reds when the static
# verdict fails. #359 is the proof the measurement earns its keep: PPOAlgorithm was
# suspected of exactly that drift, and the gate's measurement REFUTED the suspicion --
# all 20 pairs check out -- so the live-tree leg below now stands as the continuous
# control on that refutation instead of a one-off hand audit.
#
# This gate's --self-test documents NO trailing 'DENOMINATOR: N of N controls' tally the
# way mutation_scope's does, so NO denominator is parsed here. Inventing a floor for a
# tally the gate never published would redden on a wording change rather than on a lost
# control.
#
# rc 95 (mypy not installed) under THIS suite is not a host accident and not "the same
# disposition this suite gives every unmeasurable arm": it is a CONSTRUCTION. The suite
# invokes gates as `python3 -S checks/<gate>.py`, and -S skips `import site`, so
# site-packages is off sys.path for that process and find_spec("mypy") is None on EVERY
# host -- CI, and this developer machine where the venv interpreter imports mypy fine.
# -S is load-bearing (#83/#229: it forces the same not-installed condition on every
# runner so a verdict cannot depend on what happens to be installed). The three arms
# below therefore ASSERT rc=95 and what the gate still measures without an analyzer,
# rather than tolerating 95 as a disposition. The real mypy verdict is not lost: it is
# taken by `make rl-static-subtypes`, whose $(PY) is the repo venv (mypy present, no -S),
# in the check: chain.

# --- Arm 1 (f360) -- MUST_BE_UNMEASURED: never silently green without an analyzer ----
if [ ! -r "checks/rl_static_subtypes.py" ]; then
  f360_msg="MUST_BE_UNMEASURED FAILED (rl_static_subtypes self-test):"
  f360_msg="$f360_msg checks/rl_static_subtypes.py is not readable -- unreadable is not"
  f360_msg="$f360_msg empty (doctrine 4); the gate cannot run, so its refusal to report"
  f360_msg="$f360_msg green without an analyzer was not measured"
  no "$f360_msg"
else
  f360_rc=0
  f360_out=$(python3 -S checks/rl_static_subtypes.py --self-test 2>&1) || f360_rc=$?
  if [ "$f360_rc" -eq 95 ] \
     && grep -q 'RL-STATIC-SUBTYPES UNMEASURED' <<<"$f360_out" \
     && grep -q 'mypy' <<<"$f360_out"; then
    f360_msg="MUST_BE_UNMEASURED rl_static_subtypes self-test: rc=95 with"
    f360_msg="$f360_msg 'RL-STATIC-SUBTYPES UNMEASURED' naming mypy -- the gate refused"
    f360_msg="$f360_msg to report green without its analyzer. This is a deterministic"
    f360_msg="$f360_msg control, not a host-dependent one: the suite runs the gate as"
    f360_msg="$f360_msg python3 -S, which skips import site and removes site-packages"
    f360_msg="$f360_msg from sys.path BY CONSTRUCTION, so mypy is absent on every host"
    f360_msg="$f360_msg including CI. The mypy verdict itself is taken by make"
    f360_msg="$f360_msg rl-static-subtypes (repo venv interpreter, mypy present, no -S,"
    f360_msg="$f360_msg in the check: chain). Output:"
    f360_msg="$f360_msg $(printf '%s\n' "$f360_out" | tr '\n' ' ')"
    ok "$f360_msg"
  elif [ "$f360_rc" -eq 0 ]; then
    f360_msg="MUST_BE_UNMEASURED FAILED (rl_static_subtypes self-test): rc=0 under"
    f360_msg="$f360_msg python3 -S -- the gate reported CLEAR while -S guarantees its"
    f360_msg="$f360_msg analyzer is absent. That silent-green fake is the exact defect"
    f360_msg="$f360_msg this control exists to refuse: a subtype verdict with no mypy"
    f360_msg="$f360_msg behind it is manufactured, not measured. Output:"
    f360_msg="$f360_msg $(printf '%s\n' "$f360_out" | tr '\n' ' ')"
    no "$f360_msg"
  elif [ "$f360_rc" -eq 5 ]; then
    f360_msg="MUST_BE_UNMEASURED FAILED (rl_static_subtypes self-test): rc=5 under"
    f360_msg="$f360_msg python3 -S -- the gate reported a RED finding while -S guarantees"
    f360_msg="$f360_msg its analyzer is absent. A finding manufactured out of an absent"
    f360_msg="$f360_msg analyzer reddens the suite over nothing and buries real reds."
    f360_msg="$f360_msg Output: $(printf '%s\n' "$f360_out" | tr '\n' ' ')"
    no "$f360_msg"
  elif [ "$f360_rc" -eq 96 ]; then
    f360_msg="MUST_BE_UNMEASURED FAILED (rl_static_subtypes self-test): rc=96 under"
    f360_msg="$f360_msg python3 -S -- the gate issued a REFUSAL where an abstention"
    f360_msg="$f360_msg (rc=95, UNMEASURED) is the owed verdict for an absent analyzer."
    f360_msg="$f360_msg Output: $(printf '%s\n' "$f360_out" | tr '\n' ' ')"
    no "$f360_msg"
  elif [ "$f360_rc" -eq 95 ]; then
    f360_msg="MUST_BE_UNMEASURED FAILED (rl_static_subtypes self-test): rc=95 but the"
    f360_msg="$f360_msg output does not carry both the 'RL-STATIC-SUBTYPES UNMEASURED'"
    f360_msg="$f360_msg marker and the name 'mypy' -- an abstention that does not name"
    f360_msg="$f360_msg its missing analyzer is unattributable. Output:"
    f360_msg="$f360_msg $(printf '%s\n' "$f360_out" | tr '\n' ' ')"
    no "$f360_msg"
  else
    f360_msg="MUST_BE_UNMEASURED FAILED (rl_static_subtypes self-test): rc=$f360_rc"
    f360_msg="$f360_msg under python3 -S, which is none of the gate's declared verdicts"
    f360_msg="$f360_msg (0 CLEAR / 5 RED / 95 UNMEASURED / 96 REFUSAL). Fail closed."
    f360_msg="$f360_msg Output: $(printf '%s\n' "$f360_out" | tr '\n' ' ')"
    no "$f360_msg"
  fi
fi

# --- Arm 2 (f360b) -- MUST_PASS: the discovery denominator over the live tree --------
# Arm 1 proves the gate refuses to fake a verdict without mypy; this leg is a SECOND
# measurement, not a restatement: the live run's own output proves the gate's STATIC
# DISCOVERY half runs to completion with no analyzer at all -- pairs=, the algorithm-key
# list and the lossfn list are all produced before mypy is ever consulted. A discovery
# layer that silently shrank to zero pairs would keep arm 1 green while making every
# future mypy verdict vacuous, so this leg holds the denominator: rc=95, pairs >= 20,
# the registry:ppo / registry:grpo / registry:dpo keys present, lossfn non-empty.
if [ ! -r "checks/rl_static_subtypes.py" ]; then
  f360b_msg="MUST_PASS FAILED (rl_static_subtypes live discovery):"
  f360b_msg="$f360b_msg checks/rl_static_subtypes.py is not readable -- unreadable is"
  f360b_msg="$f360b_msg not empty (doctrine 4); the live tree's discovery denominator"
  f360b_msg="$f360b_msg was not measured"
  no "$f360b_msg"
else
  f360b_rc=0
  f360b_out=$(python3 -S checks/rl_static_subtypes.py 2>&1) || f360b_rc=$?
  f360b_pairs=$(printf '%s\n' "$f360b_out" |
    sed -n 's/.*pairs=\([0-9][0-9]*\).*/\1/p' | head -1)
  f360b_lossfn=$(printf '%s\n' "$f360b_out" |
    sed -n 's/.*lossfn=\([^;]*\).*/\1/p' | head -1)
  f360b_missing=""
  for f360b_key in registry:ppo registry:grpo registry:dpo; do
    if ! grep -q "$f360b_key" <<<"$f360b_out"; then
      f360b_missing="$f360b_missing $f360b_key"
    fi
  done
  if [ -n "$f360b_pairs" ]; then f360b_n=$f360b_pairs; else f360b_n=absent; fi
  if [ -n "$f360b_missing" ]; then f360b_miss=$f360b_missing; else f360b_miss=none; fi
  if [ "$f360b_rc" -ne 95 ]; then
    f360b_msg="MUST_PASS FAILED (rl_static_subtypes live discovery): observed"
    f360b_msg="$f360b_msg rc=$f360b_rc, expected 95 (the -S construction arm 1 asserts);"
    f360b_msg="$f360b_msg pairs=$f360b_n, missing-keys=$f360b_miss. Any other rc here"
    f360b_msg="$f360b_msg means the gate reached a verdict its absent analyzer cannot"
    f360b_msg="$f360b_msg back, or crashed. Output:"
    f360b_msg="$f360b_msg $(printf '%s\n' "$f360b_out" | tr '\n' ' ')"
    no "$f360b_msg"
  elif [ -z "$f360b_pairs" ] || [ "$f360b_pairs" -lt 20 ]; then
    f360b_msg="MUST_PASS FAILED (rl_static_subtypes live discovery): observed rc=95 but"
    f360b_msg="$f360b_msg pairs=$f360b_n (floor 20), missing-keys=$f360b_miss -- the"
    f360b_msg="$f360b_msg static discovery denominator shrank below the registry's known"
    f360b_msg="$f360b_msg 20 (binding, protocol) pairs. A discovery layer approaching zero"
    f360b_msg="$f360b_msg pairs keeps arm 1 green while making every future mypy verdict"
    f360b_msg="$f360b_msg vacuous. Output:"
    f360b_msg="$f360b_msg $(printf '%s\n' "$f360b_out" | tr '\n' ' ')"
    no "$f360b_msg"
  elif [ -n "$f360b_missing" ]; then
    f360b_msg="MUST_PASS FAILED (rl_static_subtypes live discovery): observed rc=95,"
    f360b_msg="$f360b_msg pairs=$f360b_n, but the algorithm-key list is missing"
    f360b_msg="$f360b_msg required key(s):$f360b_missing -- discovery no longer sees"
    f360b_msg="$f360b_msg bindings the registry ships. Output:"
    f360b_msg="$f360b_msg $(printf '%s\n' "$f360b_out" | tr '\n' ' ')"
    no "$f360b_msg"
  elif [ -z "$f360b_lossfn" ]; then
    f360b_msg="MUST_PASS FAILED (rl_static_subtypes live discovery): observed rc=95,"
    f360b_msg="$f360b_msg pairs=$f360b_n, missing-keys=none, but lossfn= is absent or"
    f360b_msg="$f360b_msg empty -- the loss-function half of static discovery reported"
    f360b_msg="$f360b_msg nothing. Output:"
    f360b_msg="$f360b_msg $(printf '%s\n' "$f360b_out" | tr '\n' ' ')"
    no "$f360b_msg"
  else
    f360b_msg="MUST_PASS rl_static_subtypes live discovery: rc=95 (analyzer absent by"
    f360b_msg="$f360b_msg the -S construction, as arm 1 asserts) yet the STATIC"
    f360b_msg="$f360b_msg DISCOVERY half ran to completion with no analyzer -- measured"
    f360b_msg="$f360b_msg pairs=$f360b_pairs (>= 20), algorithm-keys naming registry:ppo"
    f360b_msg="$f360b_msg by that literal alongside registry:grpo and registry:dpo,"
    f360b_msg="$f360b_msg lossfn=$f360b_lossfn non-empty. This second measurement holds"
    f360b_msg="$f360b_msg the denominator arm 1 cannot: discovery of (binding, protocol)"
    f360b_msg="$f360b_msg pairs over the live registry, where a silent shrink to zero"
    f360b_msg="$f360b_msg would leave arm 1 green and every future mypy verdict vacuous."
    f360b_msg="$f360b_msg registry:ppo named by that literal keeps #359's refutation"
    f360b_msg="$f360b_msg (all 20 pairs check out) under continuous control"
    ok "$f360b_msg"
  fi
fi

# --- Arm 3 (f360c) -- MUST_FIRE: doctored-baseline discrimination ---------------------
# The legs above prove the gate's discovery half measures the live tree; this leg proves
# the baseline/completeness axis can refuse -- and it can, with no analyzer: the
# doctored-baseline arm reaches rc=5 because that axis is adjudicated AHEAD of the mypy
# axis. The fixture is the SHIPPED checks/rl_static_subtypes.baseline.json copied into a
# scratch dir (never the live tree), and the doctored copy differs by EXACTLY ONE added
# ghost binding key, registry:ghost_binding. The doctored copy must score rc=5 with
# output naming that ghost key; the intact copy must NOT score 5 (0 or 95 both accepted
# -- 95 is the -S construction arm 1 asserts). The doctoring is PROVEN, not assumed:
# cmp must report the two files DIFFERENT before either arm runs, because a plant that
# did not land would hand both arms the same bytes and let the MUST_FIRE pass by never
# firing. Nothing outside the scratch dir is touched.
if [ ! -r "checks/rl_static_subtypes.py" ]; then
  f360c_msg="MUST_FIRE FAILED (rl_static_subtypes doctored-baseline discrimination):"
  f360c_msg="$f360c_msg checks/rl_static_subtypes.py is not readable -- unreadable is"
  f360c_msg="$f360c_msg not empty (doctrine 4); 0 of 2 discrimination arms ran"
  no "$f360c_msg"
elif [ ! -r "checks/rl_static_subtypes.baseline.json" ]; then
  f360c_msg="MUST_FIRE FAILED (rl_static_subtypes doctored-baseline discrimination):"
  f360c_msg="$f360c_msg checks/rl_static_subtypes.baseline.json is not readable --"
  f360c_msg="$f360c_msg unreadable is not empty (doctrine 4); the fixture both arms are"
  f360c_msg="$f360c_msg built from does not exist here, and a synthesised one would make"
  f360c_msg="$f360c_msg the arms differ by more than the one variable. 0 of 2 arms ran"
  no "$f360c_msg"
else
  f360c_tmp=$(mktemp -d)
  cp "checks/rl_static_subtypes.baseline.json" "$f360c_tmp/intact.json"
  if [ ! -r "$f360c_tmp/intact.json" ] \
     || ! cmp -s "$f360c_tmp/intact.json" "checks/rl_static_subtypes.baseline.json"; then
    rm -rf "$f360c_tmp"
    f360c_msg="MUST_FIRE FAILED (rl_static_subtypes doctored-baseline discrimination):"
    f360c_msg="$f360c_msg the copied fixture is unreadable or differs from the shipped"
    f360c_msg="$f360c_msg checks/rl_static_subtypes.baseline.json -- the fixture must be"
    f360c_msg="$f360c_msg the real baseline, not a synthesised or corrupted one. 0 of 2"
    f360c_msg="$f360c_msg arms ran"
    no "$f360c_msg"
  else
    # ADD exactly ONE ghost binding key to the doctored copy. JSON in, JSON out through
    # python3 so a malformed baseline fails loudly here instead of being half-edited
    # into something that reds for the wrong reason.
    f360c_edit_rc=0
    python3 -S -c '
import copy, json, sys
with open(sys.argv[1], encoding="utf-8") as fh:
    data = json.load(fh)
pairs = data.get("pairs") if isinstance(data, dict) else data
if not isinstance(pairs, list):
    sys.exit(2)
ghost = copy.deepcopy(pairs[0]) if pairs else {}
if isinstance(ghost, dict):
    ghost["key"] = "registry:ghost_binding"
else:
    ghost = "registry:ghost_binding"
pairs.append(ghost)
with open(sys.argv[2], "w", encoding="utf-8") as fh:
    json.dump(data, fh)
' "$f360c_tmp/intact.json" "$f360c_tmp/doctored.json" || f360c_edit_rc=$?
    if [ "$f360c_edit_rc" -ne 0 ] || [ ! -r "$f360c_tmp/doctored.json" ]; then
      rm -rf "$f360c_tmp"
      f360c_msg="MUST_FIRE FAILED (rl_static_subtypes doctored-baseline"
      f360c_msg="$f360c_msg discrimination): the JSON doctor failed (rc=$f360c_edit_rc)"
      f360c_msg="$f360c_msg or wrote no doctored copy -- the shipped baseline carried no"
      f360c_msg="$f360c_msg 'pairs' list to add the ghost key to. A plant that did not"
      f360c_msg="$f360c_msg plant proves nothing; 0 of 2 arms ran"
      no "$f360c_msg"
    elif cmp -s "$f360c_tmp/intact.json" "$f360c_tmp/doctored.json"; then
      rm -rf "$f360c_tmp"
      f360c_msg="MUST_FIRE FAILED (rl_static_subtypes doctored-baseline"
      f360c_msg="$f360c_msg discrimination): the doctoring changed ZERO bytes -- intact"
      f360c_msg="$f360c_msg and doctored copies are identical, so both arms would read"
      f360c_msg="$f360c_msg the same input and the MUST_FIRE could pass by never firing."
      f360c_msg="$f360c_msg A control whose defect was never planted is not a control;"
      f360c_msg="$f360c_msg 0 of 2 arms ran"
      no "$f360c_msg"
    else
      f360c_red_rc=0
      f360c_red_out=$(python3 -S checks/rl_static_subtypes.py \
        --baseline "$f360c_tmp/doctored.json" 2>&1) || f360c_red_rc=$?
      f360c_clear_rc=0
      f360c_clear_out=$(python3 -S checks/rl_static_subtypes.py \
        --baseline "$f360c_tmp/intact.json" 2>&1) || f360c_clear_rc=$?
      rm -rf "$f360c_tmp"
      if [ "$f360c_red_rc" -ne 5 ]; then
        f360c_msg="MUST_FIRE FAILED (rl_static_subtypes doctored-baseline"
        f360c_msg="$f360c_msg discrimination): adding the one ghost key"
        f360c_msg="$f360c_msg registry:ghost_binding scored rc=$f360c_red_rc, expected"
        f360c_msg="$f360c_msg exactly 5 (RED) -- the baseline/completeness axis is"
        f360c_msg="$f360c_msg adjudicated ahead of the mypy axis, so no analyzer is"
        f360c_msg="$f360c_msg needed for this refusal. rc=0 would mean the gate cannot"
        f360c_msg="$f360c_msg see a baseline pair its discovery cannot reproduce. Output:"
        f360c_msg="$f360c_msg $(printf '%s\n' "$f360c_red_out" | tr '\n' ' ')"
        no "$f360c_msg"
      elif ! grep -q 'registry:ghost_binding' <<<"$f360c_red_out"; then
        f360c_msg="MUST_FIRE FAILED (rl_static_subtypes doctored-baseline"
        f360c_msg="$f360c_msg discrimination): rc=5, but the RED output never names"
        f360c_msg="$f360c_msg registry:ghost_binding -- the one key planted. A RED"
        f360c_msg="$f360c_msg attributed to anything else is not this control firing,"
        f360c_msg="$f360c_msg and rc alone cannot tell the two apart. Output:"
        f360c_msg="$f360c_msg $(printf '%s\n' "$f360c_red_out" | tr '\n' ' ')"
        no "$f360c_msg"
      elif [ "$f360c_clear_rc" -eq 5 ]; then
        f360c_msg="MUST_FIRE FAILED (rl_static_subtypes doctored-baseline"
        f360c_msg="$f360c_msg discrimination): the doctored arm fired correctly at rc=5"
        f360c_msg="$f360c_msg naming registry:ghost_binding, but the INTACT arm -- the"
        f360c_msg="$f360c_msg shipped baseline, the ghost key the only difference --"
        f360c_msg="$f360c_msg also scored rc=5 (observed intact rc=$f360c_clear_rc). A"
        f360c_msg="$f360c_msg gate that reddens its own true baseline is stuck RED, not"
        f360c_msg="$f360c_msg discriminating. Output:"
        f360c_msg="$f360c_msg $(printf '%s\n' "$f360c_clear_out" | tr '\n' ' ')"
        no "$f360c_msg"
      else
        f360c_msg="MUST_FIRE rl_static_subtypes doctored-baseline discrimination: the"
        f360c_msg="$f360c_msg fixture was the shipped baseline copied byte-true (cmp"
        f360c_msg="$f360c_msg proven), ONE ghost binding key registry:ghost_binding was"
        f360c_msg="$f360c_msg added (byte-change proven by cmp), the doctored copy scored"
        f360c_msg="$f360c_msg rc=5 with the output naming that exact ghost key, and the"
        f360c_msg="$f360c_msg intact copy scored rc=$f360c_clear_rc -- not 5 (0 or 95"
        f360c_msg="$f360c_msg both accepted; 95 is the -S construction arm 1 asserts)."
        f360c_msg="$f360c_msg The ghost key was the only variable, and the live tree was"
        f360c_msg="$f360c_msg never touched. Both outcomes held"
        ok "$f360c_msg"
      fi
    fi
  fi
fi

# --- MUST_PASS: campaign self-test runner controls (checks/campaign_self_tests.py) ---
# Finding #352: pyproject.toml sets testpaths = ["tests"], so pytest collects NOTHING under
# validation_campaigns/. Six campaign modules ship a --self-test flag that plants controls
# and returns 0 only if every one passes, and five of the six were invoked by nobody. That
# is #278/#293 on a third population: however good a gate's controls are, a gate that runs
# nowhere states no coverage.
#
# Same floor convention as the f317 leg above, for the same reason: rc=0 is not the
# measurement. The trailing tally is parsed and held to a FLOOR of 16 so a self-test that
# quietly drops controls cannot still read green.
#
# The floor spans both families -- 6 MUST_FIRE (one per declaration rule R1-R6) and 10
# MUST_PASS (planted stubs parse; a --self-test mention in a COMMENT stays out of the
# measured population; a file with no literal stays out; two literal-bearing files are in;
# the clean configuration stays silent; and the runner observes PASS, FAIL, TIMEOUT and
# absent-from-disk as four distinct states). The last four are controls on the RUNNER
# rather than on the declaration audit, and they are what stop this gate certifying five
# modules it never actually executed.
#
# Floor history: 16 at introduction (#352).
if [ ! -r "checks/campaign_self_tests.py" ]; then
  f352_msg="MUST_PASS FAILED (campaign_self_tests self-test) UNMEASURED:"
  f352_msg="$f352_msg checks/campaign_self_tests.py is not readable -- unreadable is not empty"
  f352_msg="$f352_msg (doctrine 4); the gate cannot run, so 0 of 16 controls were measured"
  no "$f352_msg"
else
  f352_rc=0
  f352_out=$(python3 -S checks/campaign_self_tests.py --self-test 2>&1) || f352_rc=$?
  f352_last=$(printf '%s\n' "$f352_out" | tail -n 1)
  f352_have=$(printf '%s\n' "$f352_last" |
    sed -n 's/^SELF-TEST DENOMINATOR: \([0-9][0-9]*\) of \([0-9][0-9]*\) controls behaved;.*/\1/p')
  f352_want=$(printf '%s\n' "$f352_last" |
    sed -n 's/^SELF-TEST DENOMINATOR: \([0-9][0-9]*\) of \([0-9][0-9]*\) controls behaved;.*/\2/p')
  if [ "$f352_rc" -ne 0 ]; then
    f352_msg="MUST_PASS FAILED (campaign_self_tests self-test): rc=$f352_rc over the gate's"
    f352_msg="$f352_msg declared denominator of 16 controls (6 MUST_FIRE + 10 MUST_PASS);"
    f352_msg="$f352_msg 0 of 16 are accepted as behaved, so the leg fails closed. Output:"
    f352_msg="$f352_msg $(printf '%s\n' "$f352_out" | tr '\n' ' ')"
    no "$f352_msg"
  elif [ -z "$f352_have" ] || [ -z "$f352_want" ]; then
    f352_msg="MUST_PASS FAILED (campaign_self_tests self-test) UNMEASURED: rc=0 but the last"
    f352_msg="$f352_msg line carries no parseable 'SELF-TEST DENOMINATOR: N of N controls"
    f352_msg="$f352_msg behaved' tally -- the measuring unit printed no denominator, so 0 of"
    f352_msg="$f352_msg 16 declared controls are auditable here. Unparseable is not passing;"
    f352_msg="$f352_msg fail closed and update this leg in the same commit as the wording"
    f352_msg="$f352_msg change. Last line: $f352_last"
    no "$f352_msg"
  elif [ "$f352_have" -ne "$f352_want" ]; then
    f352_msg="MUST_PASS FAILED (campaign_self_tests self-test): denominator $f352_have of"
    f352_msg="$f352_msg $f352_want controls is not self-consistent -- the self-test examined"
    f352_msg="$f352_msg fewer controls than it claims to have. rc=0 cannot certify a partial"
    f352_msg="$f352_msg control set, so the inconsistency fails closed."
    no "$f352_msg"
  elif [ "$f352_have" -lt 16 ]; then
    f352_msg="MUST_PASS FAILED (campaign_self_tests self-test): control set shrank to"
    f352_msg="$f352_msg $f352_have of $f352_want, below the measured floor of 16 (6 MUST_FIRE"
    f352_msg="$f352_msg + 10 MUST_PASS). A shortened self-test still exits 0, so the floor is"
    f352_msg="$f352_msg the control and this leg fails closed."
    no "$f352_msg"
  else
    f352_msg="MUST_PASS campaign_self_tests self-test: rc=0 under python3 -S, denominator"
    f352_msg="$f352_msg $f352_have of $f352_want controls (>= the measured floor of 16,"
    f352_msg="$f352_msg 6 MUST_FIRE + 10 MUST_PASS): $f352_last"
    ok "$f352_msg"
  fi
fi

# --- MUST_FIRE: campaign_self_tests actually EXECUTES what it certifies ----------------
# The self-test above exercises the gate's internal fixture path over planted stubs. This
# leg exercises the shipped CLI over the REAL git index and the REAL shipped declarations,
# and it is a discrimination PAIR: the same tree, the same RUNNABLE population, differing
# ONLY in --timeout, must give 5 (RED) and then 0 (CLEAR).
#
# The variable is chosen deliberately. The failure this gate exists to prevent is a gate
# that reports coverage without running anything -- #278/#293 -- and a declaration audit
# alone would still exit 0 with every subprocess never spawned. Making every spawn
# unmeetable is the one input that separates "the partition is declared correctly" from
# "the modules were executed and adjudicated". If the CLEAR arm's rc=0 survived an
# unmeetable timeout, the run would be certifying files it never touched.
#
# The assertion is rc=5 exactly, not merely nonzero: nonzero would accept a crash (1/2), an
# UNMEASURED (95) or a REFUSE (96) as the control firing, and none of those is this gate's
# declared RED. The RED arm additionally asserts that EVERY runnable module is named
# TIMEOUT -- an rc-only check would be satisfied by a RED raised for some unrelated reason.
#
# The expected count is READ OFF THE CLEAR ARM ("N of N runnable"), not written here as a
# literal. A literal was here, it said five, and the sixth module (#294's restore control)
# falsified it the day it enrolled -- turning a correct enrolment into a red leg that
# accused the gate. The population is measured on both arms of the same tree, so it moves
# on its own; a count that cannot be parsed, or a zero, fails closed rather than passing
# vacuously over an empty denominator (doctrine 1).
if [ ! -r "checks/campaign_self_tests.py" ]; then
  f352b_msg="MUST_FIRE FAILED (campaign_self_tests execution discrimination) UNMEASURED:"
  f352b_msg="$f352b_msg checks/campaign_self_tests.py is not readable -- unreadable is not"
  f352b_msg="$f352b_msg empty (doctrine 4); 0 of 2 discrimination arms were measured"
  no "$f352b_msg"
else
  f352b_red_rc=0
  f352b_red_out=$(python3 -S checks/campaign_self_tests.py --timeout 0.000001 2>&1) ||
    f352b_red_rc=$?
  f352b_clear_rc=0
  f352b_clear_out=$(python3 -S checks/campaign_self_tests.py 2>&1) || f352b_clear_rc=$?
  f352b_timeouts=$(printf '%s\n' "$f352b_red_out" | grep -c '^TIMEOUT validation_campaigns/')
  # The RUNNABLE population, measured off the gate's own verdict line rather than
  # restated here. awk (ERE) not sed: BSD sed has no \| alternation, and a pattern
  # that silently matches nothing would set the expectation to empty and take this
  # leg straight to the unparseable branch below on every macOS run.
  f352b_pop=$(printf '%s\n' "$f352b_clear_out" |
    awk '/campaign_self_tests: [0-9]+ of [0-9]+ runnable/ {
           for (i = 1; i <= NF; i++) if ($i == "of") { print $(i + 1); exit }
         }')
  if ! grep -q '^[0-9][0-9]*$' <<<"$f352b_pop" || [ "$f352b_pop" -lt 1 ]; then
    f352b_msg="MUST_FIRE FAILED (campaign_self_tests execution discrimination) UNMEASURED: the"
    f352b_msg="$f352b_msg CLEAR arm named no parseable RUNNABLE population (read"
    f352b_msg="$f352b_msg '${f352b_pop:-<empty>}'), so this leg has no denominator to assert"
    f352b_msg="$f352b_msg against. Asserting 0 TIMEOUTs against 0 declared modules would pass"
    f352b_msg="$f352b_msg over an empty population -- exactly the vacuity #352 is about -- so"
    f352b_msg="$f352b_msg the leg fails closed. Output:"
    f352b_msg="$f352b_msg $(printf '%s\n' "$f352b_clear_out" | tr '\n' ' ')"
    no "$f352b_msg"
  elif [ "$f352b_red_rc" -ne 5 ]; then
    f352b_msg="MUST_FIRE FAILED (campaign_self_tests execution discrimination): an unmeetable"
    f352b_msg="$f352b_msg --timeout gave rc=$f352b_red_rc, expected exactly 5 (RED). rc=0 would"
    f352b_msg="$f352b_msg mean the gate reached CLEAR without any module having run -- the"
    f352b_msg="$f352b_msg whole of #352 -- and any other nonzero is not its declared RED."
    f352b_msg="$f352b_msg Output: $(printf '%s\n' "$f352b_red_out" | tr '\n' ' ')"
    no "$f352b_msg"
  elif [ "$f352b_timeouts" -ne "$f352b_pop" ]; then
    f352b_msg="MUST_FIRE FAILED (campaign_self_tests execution discrimination): the gate"
    f352b_msg="$f352b_msg exited 5, but $f352b_timeouts of the $f352b_pop RUNNABLE modules are named"
    f352b_msg="$f352b_msg TIMEOUT. Under an unmeetable timeout every declared module must be"
    f352b_msg="$f352b_msg observed timing out; a smaller count means some module was never"
    f352b_msg="$f352b_msg spawned, and a RED raised for another reason is not this control"
    f352b_msg="$f352b_msg firing. Output: $(printf '%s\n' "$f352b_red_out" | tr '\n' ' ')"
    no "$f352b_msg"
  elif [ "$f352b_clear_rc" -ne 0 ]; then
    f352b_msg="MUST_FIRE FAILED (campaign_self_tests execution discrimination): the unmeetable"
    f352b_msg="$f352b_msg arm fired correctly at rc=5 with all $f352b_pop modules TIMEOUT, but the same"
    f352b_msg="$f352b_msg command at the default timeout -- the only difference -- gave"
    f352b_msg="$f352b_msg rc=$f352b_clear_rc instead of 0. Only 1 of 2 arms held; a gate that"
    f352b_msg="$f352b_msg reddens on a healthy tree is stuck RED, not discriminating, so the"
    f352b_msg="$f352b_msg leg fails closed. Output:"
    f352b_msg="$f352b_msg $(printf '%s\n' "$f352b_clear_out" | tr '\n' ' ')"
    no "$f352b_msg"
  else
    f352b_msg="MUST_FIRE campaign_self_tests execution discrimination: over the live git index"
    f352b_msg="$f352b_msg and the shipped declarations, an unmeetable --timeout exited rc=5"
    f352b_msg="$f352b_msg with all $f352b_pop RUNNABLE modules named TIMEOUT, and the default timeout"
    f352b_msg="$f352b_msg exited rc=0 -- the spawn budget the only variable. The CLEAR is"
    f352b_msg="$f352b_msg therefore contingent on the modules having actually run, which a"
    f352b_msg="$f352b_msg declaration audit alone would not be. Both outcomes held"
    ok "$f352b_msg"
  fi
fi

# --- MUST_PASS: every RUNNABLE campaign module runs under a BARE interpreter -----------
# Finding #394, and this leg is the class behind it rather than the instance. A RUNNABLE
# campaign module imported foundationscale inside its self-test. The runner executes each
# one as [sys.executable, path, "--self-test"] and requires exit 0, so the child inherits
# whatever interpreter is running the suite. Locally that interpreter had `pip install -e .`
# applied and the module ran; the launcher-contracts CI job's python3 never did, the import
# raised ModuleNotFoundError, the child exited 1, and an UNMEASURED harness state was read
# as tree-RED on main.
#
# The instance is repaired. The class is what this leg closes, because nothing else in this
# suite can see it: whether the defect is visible depends entirely on which python3 is on
# PATH and whether anyone installed the package into it, so a developer can run every leg
# here green and still push a commit that reddens CI. That is precisely what happened.
#
# Each module is re-run under an interpreter that cannot see an installed distribution and
# cannot be helped by the ambient environment -- -S drops site-packages (an editable install
# goes invisible, the CI shape), -E makes the interpreter ignore PYTHONPATH, and the env -u
# closes the same hole for anything the module itself spawns. The accepted outcomes are 0
# (it ran and its controls behaved) and 95 (it declared itself UNMEASURED and said why).
# Any other exit is the defect; 1 is the ModuleNotFoundError shape specifically.
#
# The population is READ FROM THE REGISTRY, never declared here. A hand-written list of the
# seven current paths is a second thing that rots: the next RUNNABLE module to enrol would
# sit in no denominator while this leg still read "7 of 7". Reading RUNNABLE out of
# checks/campaign_self_tests.py also beats parsing the runner's output, which only names
# modules it actually reached -- a runner that died early would silently shrink the
# denominator on both sides at once. An empty registry, an unreadable one, or one that has
# lost the anchor is UNMEASURED, never PASS: zero units is not a pass (doctrine 1) and
# unreadable is not empty (doctrine 4).
f394c_anchor=validation_campaigns/verification_matrix/t1_6_symlinked_shard.py
if [ ! -r "checks/campaign_self_tests.py" ]; then
  f394c_msg="MUST_PASS FAILED (RUNNABLE campaign modules under a bare interpreter)"
  f394c_msg="$f394c_msg UNMEASURED: checks/campaign_self_tests.py is not readable --"
  f394c_msg="$f394c_msg unreadable is not empty (doctrine 4), so the RUNNABLE registry could"
  f394c_msg="$f394c_msg not be read and 0 modules were measured"
  no "$f394c_msg"
else
  f394c_derive_rc=0
  f394c_list=$(env -u PYTHONPATH python3 -S -E -c 'import importlib.util as u, sys
s = u.spec_from_file_location("_cst", "checks/campaign_self_tests.py")
m = u.module_from_spec(s)
sys.modules["_cst"] = m
s.loader.exec_module(m)
print("\n".join(sorted(m.RUNNABLE)))' 2>&1) || f394c_derive_rc=$?
  f394c_total=$(printf '%s\n' "$f394c_list" |
    awk '/^validation_campaigns\/.*\.py$/ { n++ } END { print n + 0 }')
  if [ "$f394c_derive_rc" -ne 0 ] || [ "$f394c_total" -lt 1 ]; then
    f394c_msg="MUST_PASS FAILED (RUNNABLE campaign modules under a bare interpreter)"
    f394c_msg="$f394c_msg UNMEASURED: reading the RUNNABLE registry out of"
    f394c_msg="$f394c_msg checks/campaign_self_tests.py exited rc=$f394c_derive_rc and yielded"
    f394c_msg="$f394c_msg $f394c_total module path(s). Zero units is never a pass (doctrine 1)"
    f394c_msg="$f394c_msg -- an empty denominator reported green is the vacuity this refuses."
    f394c_msg="$f394c_msg Output: $(printf '%s\n' "$f394c_list" | tr '\n' ' ')"
    no "$f394c_msg"
  elif ! grep -qxF "$f394c_anchor" <<<"$f394c_list"; then
    f394c_msg="MUST_PASS FAILED (RUNNABLE campaign modules under a bare interpreter)"
    f394c_msg="$f394c_msg UNMEASURED: the registry yielded $f394c_total module(s) and none of"
    f394c_msg="$f394c_msg them is $f394c_anchor -- the module whose ModuleNotFoundError"
    f394c_msg="$f394c_msg reddened main and the reason this leg exists. A derivation that has"
    f394c_msg="$f394c_msg silently lost its own anchor is a broken derivation, not a"
    f394c_msg="$f394c_msg measurement (doctrine 2)"
    no "$f394c_msg"
  else
    # An instrument that has never been observed refusing is not an instrument
    # (doctrine 3). Before trusting the sweep below, plant a module at exactly the
    # shape #394 had -- a module-scope import of a package that is not installed
    # anywhere -- and require the SAME command line to reject it. If the decoy comes
    # back 0 or 95, the rule cannot tell a broken module from a working one and the
    # seven greens underneath it mean nothing, so the leg reports UNMEASURED rather
    # than a pass it did not earn. The decoy imports a name with no distribution on
    # PyPI or in this tree, so it cannot be accidentally satisfied.
    f394c_decoy_dir=$(mktemp -d "${TMPDIR:-/tmp}/fs-394c-decoy.XXXXXX")
    f394c_decoy_rc=0
    f394c_decoy_state="plant failed"
    if [ -d "$f394c_decoy_dir" ]; then
      f394c_decoy=$f394c_decoy_dir/decoy_self_test.py
      printf 'import _fs_no_such_distribution_394\nprint(_fs_no_such_distribution_394)\n' \
        > "$f394c_decoy"
      if [ -s "$f394c_decoy" ]; then
        env -u PYTHONPATH python3 -S -E "$f394c_decoy" --self-test >/dev/null 2>&1 ||
          f394c_decoy_rc=$?
        f394c_decoy_state="rc=$f394c_decoy_rc"
      fi
    fi
    rm -rf "$f394c_decoy_dir"
  fi
  if [ ! -r "checks/campaign_self_tests.py" ] || [ "$f394c_total" -lt 1 ] ||
     ! grep -qxF "$f394c_anchor" <<<"$f394c_list"; then
    : # already adjudicated above
  elif [ "$f394c_decoy_rc" -eq 0 ] || [ "$f394c_decoy_rc" -eq 95 ]; then
    f394c_msg="MUST_PASS FAILED (RUNNABLE campaign modules under a bare interpreter)"
    f394c_msg="$f394c_msg UNMEASURED: the planted decoy -- a module whose only statement is"
    f394c_msg="$f394c_msg an import of a distribution that exists nowhere, i.e. exactly the"
    f394c_msg="$f394c_msg #394 shape -- came back $f394c_decoy_state under the same"
    f394c_msg="$f394c_msg 'env -u PYTHONPATH python3 -S -E' command line, which this leg"
    f394c_msg="$f394c_msg treats as acceptable. A rule that accepts the defect it exists to"
    f394c_msg="$f394c_msg catch cannot attribute the greens beneath it (doctrine 3), so the"
    f394c_msg="$f394c_msg sweep was not run and nothing is claimed"
    no "$f394c_msg"
  else
    f394c_ran=0
    f394c_unm=0
    f394c_bad=""
    f394c_saved_ifs=$IFS
    IFS='
'
    for f394c_mod in $f394c_list; do
      IFS=$f394c_saved_ifs
      case "$f394c_mod" in
        validation_campaigns/*.py) ;;
        *) IFS='
'
           continue ;;
      esac
      f394c_mod_rc=0
      f394c_mod_out=$(env -u PYTHONPATH python3 -S -E "$f394c_mod" --self-test 2>&1) ||
        f394c_mod_rc=$?
      if [ "$f394c_mod_rc" -eq 0 ]; then
        f394c_ran=$((f394c_ran + 1))
      elif [ "$f394c_mod_rc" -eq 95 ]; then
        f394c_unm=$((f394c_unm + 1))
      else
        f394c_last=$(printf '%s\n' "$f394c_mod_out" | tail -n 1)
        [ -z "$f394c_bad" ] || f394c_bad="$f394c_bad ; "
        f394c_bad="$f394c_bad$f394c_mod rc=$f394c_mod_rc (last line:"
        f394c_bad="$f394c_bad ${f394c_last:-<no output>})"
      fi
      IFS='
'
    done
    IFS=$f394c_saved_ifs
    if [ -n "$f394c_bad" ]; then
      f394c_msg="MUST_PASS FAILED (RUNNABLE campaign modules under a bare interpreter):"
      f394c_msg="$f394c_msg $((f394c_ran + f394c_unm)) of $f394c_total module(s) read from the"
      f394c_msg="$f394c_msg RUNNABLE registry exited 0 or 95 under"
      f394c_msg="$f394c_msg 'env -u PYTHONPATH python3 -S -E <module> --self-test'; the rest"
      f394c_msg="$f394c_msg did not, and any other exit is finding #394's class. An exit 1 is"
      f394c_msg="$f394c_msg the ModuleNotFoundError shape: the module leaned on a distribution"
      f394c_msg="$f394c_msg installed into the developer's interpreter and absent from CI's,"
      f394c_msg="$f394c_msg which is an UNMEASURED harness state that reads as tree-RED."
      f394c_msg="$f394c_msg Offenders: $f394c_bad"
      no "$f394c_msg"
    else
      f394c_msg="MUST_PASS bare-interpreter campaign self-tests: $f394c_total of $f394c_total"
      f394c_msg="$f394c_msg RUNNABLE campaign module(s) -- read from the RUNNABLE registry in"
      f394c_msg="$f394c_msg checks/campaign_self_tests.py rather than declared here, and"
      f394c_msg="$f394c_msg containing the anchor $f394c_anchor -- each exited 0 or 95 under"
      f394c_msg="$f394c_msg 'env -u PYTHONPATH python3 -S -E <module> --self-test'"
      f394c_msg="$f394c_msg ($f394c_ran measured rc=0, $f394c_unm declared UNMEASURED at"
      f394c_msg="$f394c_msg rc=95). None of them leans on an installed distribution the CI"
      f394c_msg="$f394c_msg interpreter does not have, which is the #394 defect class."
      f394c_msg="$f394c_msg The same command line REFUSED a planted decoy at that exact"
      f394c_msg="$f394c_msg shape ($f394c_decoy_state), so the rule was observed answering"
      f394c_msg="$f394c_msg both ways in this run and the greens are attributable"
      ok "$f394c_msg"
    fi
  fi
fi

# --- ORPHAN DISCHARGE: checks/verification_matrix.py -------------------------
# This block is also the suite's call site for checks/verification_matrix.py.
# The fix78-orphan leg scans every launchers/*.py and checks/*.py for a call
# site IN THIS SUITE and refuses a file that has none; a Makefile target is
# NOT accepted as a call site. The gate was just added and is indicted BY
# NAME as an orphan -- these two legs are the in-suite call site that
# discharges the indictment.

# --- MUST_PASS: verification-matrix gate self-test (checks/verification_matrix.py)
# Finding #378: matrix.json is the ledger of what this repository claims to
# have verified, and docs/VERIFICATION_MATRIX.md is its rendering. A row
# whose control arm is a placeholder is a row that CANNOT FAIL -- the defect
# this campaign has already filed four times (#291, #294, #372, #203). The
# gate exists to keep that from recurring in the ledger that tracks it.
#
# Same floor convention as the f317 legs above, for the same reason: rc=0 is
# not the measurement. The trailing tally is parsed and held to a FLOOR of 8
# so a self-test that quietly drops controls cannot still read green. A leg
# that only checked rc==0 would pass over a self-test that had silently
# shrunk to one control -- the denominator goes on the wire.
#
# The floor spans four families, as counted by the gate's own banner: 5
# MUST_FIRE (SC1-SC5, one planted defect per check C1-C5), 1
# MUST_PASS_NEGATIVE (SC6, the unmodified files must be CLEAR), 1
# MUST_ABSTAIN (SC7, a missing matrix is UNMEASURED), and 1 MUST_OUTRANK
# (SC8, RED outranks UNMEASURED).
#
# Floor history: 8 at introduction (#378).
if [ ! -r "checks/verification_matrix.py" ]; then
  f378_msg="MUST_PASS FAILED (verification_matrix self-test) UNMEASURED:"
  f378_msg="$f378_msg checks/verification_matrix.py is not readable --"
  f378_msg="$f378_msg unreadable is not empty (doctrine 4); the gate cannot"
  f378_msg="$f378_msg run, so 0 of 8 controls were measured"
  no "$f378_msg"
else
  f378_rc=0
  f378_out=$(python3 -S checks/verification_matrix.py --self-test 2>&1) || f378_rc=$?
  f378_last=$(printf '%s\n' "$f378_out" | tail -n 1)
  f378_have=$(printf '%s\n' "$f378_last" |
    sed -n 's/^self-test denominator: \([0-9][0-9]*\) of \([0-9][0-9]*\) controls .*/\1/p')
  f378_want=$(printf '%s\n' "$f378_last" |
    sed -n 's/^self-test denominator: \([0-9][0-9]*\) of \([0-9][0-9]*\) controls .*/\2/p')
  if [ "$f378_rc" -ne 0 ]; then
    f378_msg="MUST_PASS FAILED (verification_matrix self-test): rc=$f378_rc"
    f378_msg="$f378_msg over the gate's declared denominator of 8 controls"
    f378_msg="$f378_msg (5 MUST_FIRE + 1 MUST_PASS_NEGATIVE + 1 MUST_ABSTAIN"
    f378_msg="$f378_msg + 1 MUST_OUTRANK); 0 of 8 are accepted as behaved,"
    f378_msg="$f378_msg so the leg fails closed. Output:"
    f378_msg="$f378_msg $(printf '%s\n' "$f378_out" | tr '\n' ' ')"
    no "$f378_msg"
  elif [ -z "$f378_have" ] || [ -z "$f378_want" ]; then
    f378_msg="MUST_PASS FAILED (verification_matrix self-test) UNMEASURED:"
    f378_msg="$f378_msg rc=0 but the last line carries no parseable"
    f378_msg="$f378_msg 'self-test denominator: N of M controls' tally --"
    f378_msg="$f378_msg the measuring unit printed no denominator, so 0 of 8"
    f378_msg="$f378_msg declared controls are auditable here. Unparseable is"
    f378_msg="$f378_msg not passing; fail closed and update this leg in the"
    f378_msg="$f378_msg same commit as the wording change. Last line:"
    f378_msg="$f378_msg $f378_last"
    no "$f378_msg"
  elif [ "$f378_have" -ne "$f378_want" ]; then
    f378_msg="MUST_PASS FAILED (verification_matrix self-test): denominator"
    f378_msg="$f378_msg $f378_have of $f378_want controls is not"
    f378_msg="$f378_msg self-consistent -- the self-test examined fewer"
    f378_msg="$f378_msg controls than it claims to have. rc=0 cannot certify"
    f378_msg="$f378_msg a partial control set, so the inconsistency fails"
    f378_msg="$f378_msg closed."
    no "$f378_msg"
  elif [ "$f378_have" -lt 8 ]; then
    f378_msg="MUST_PASS FAILED (verification_matrix self-test): control set"
    f378_msg="$f378_msg shrank to $f378_have of $f378_want, below the"
    f378_msg="$f378_msg measured floor of 8 (5 MUST_FIRE + 1"
    f378_msg="$f378_msg MUST_PASS_NEGATIVE + 1 MUST_ABSTAIN + 1"
    f378_msg="$f378_msg MUST_OUTRANK). A shortened self-test still exits 0,"
    f378_msg="$f378_msg so the floor is the control and this leg fails"
    f378_msg="$f378_msg closed."
    no "$f378_msg"
  else
    f378_msg="MUST_PASS verification_matrix self-test: rc=0 under python3"
    f378_msg="$f378_msg -S, denominator $f378_have of $f378_want controls"
    f378_msg="$f378_msg (>= the measured floor of 8, 5 MUST_FIRE + 1"
    f378_msg="$f378_msg MUST_PASS_NEGATIVE + 1 MUST_ABSTAIN + 1"
    f378_msg="$f378_msg MUST_OUTRANK): $f378_last"
    ok "$f378_msg"
  fi
fi

# --- MUST_FIRE: verification_matrix discriminates a placeholder arm (C2) -----
# The self-test above exercises the gate's INTERNAL controls over temp copies
# it makes itself. This leg exercises the shipped CLI over COPIES of the real
# ledger and the real doc, the plant the only difference between a green arm
# and a red arm.
#
# The plant is a placeholder control_arm on the FIRST row -- the exact #378
# shape, a row that cannot fail. BOTH copies are doctored, and that is
# deliberate: C2 (CONTROL_ARM_SUBSTANTIVE) is the control under test, and
# doctoring only the JSON would ALSO trip C5 (doc-equals-ledger), so the RED
# could not be attributed to the plant. The doc's one table row for that id
# gets the same substitution, keeping the doc equal to the ledger so C2
# alone fires.
if [ ! -r "checks/verification_matrix.py" ] ||
  [ ! -r "validation_campaigns/verification_matrix/matrix.json" ] ||
  [ ! -r "docs/VERIFICATION_MATRIX.md" ]; then
  f378b_msg="MUST_FIRE FAILED (verification_matrix placeholder"
  f378b_msg="$f378b_msg discrimination) UNMEASURED: the gate, the ledger, or"
  f378b_msg="$f378b_msg the doc is not readable -- unreadable is not empty"
  f378b_msg="$f378b_msg (doctrine 4); 0 of 2 arms ran"
  no "$f378b_msg"
else
  f378b_tmp=$(mktemp -d)
  cp "validation_campaigns/verification_matrix/matrix.json" \
    "$f378b_tmp/matrix.json"
  cp "docs/VERIFICATION_MATRIX.md" "$f378b_tmp/VERIFICATION_MATRIX.md"
  f378b_green_rc=0
  f378b_green_out=$(python3 -S checks/verification_matrix.py \
    --matrix "$f378b_tmp/matrix.json" --doc "$f378b_tmp/VERIFICATION_MATRIX.md" 2>&1) ||
    f378b_green_rc=$?
  f378b_denom=$(printf '%s\n' "$f378b_green_out" |
    sed -n 's/^DENOMINATOR: \([0-9][0-9]*\) matrix row(s).*/\1/p')
  if [ "$f378b_green_rc" -ne 0 ]; then
    rm -rf "$f378b_tmp"
    f378b_msg="MUST_FIRE FAILED (verification_matrix placeholder"
    f378b_msg="$f378b_msg discrimination): the UNMODIFIED copies of the live"
    f378b_msg="$f378b_msg ledger and doc gave rc=$f378b_green_rc instead of"
    f378b_msg="$f378b_msg 0 -- the gate is stuck RED, not discriminating, so"
    f378b_msg="$f378b_msg the leg fails closed. Output:"
    f378b_msg="$f378b_msg $(printf '%s\n' "$f378b_green_out" | tr '\n' ' ')"
    no "$f378b_msg"
  elif [ -z "$f378b_denom" ]; then
    rm -rf "$f378b_tmp"
    f378b_msg="MUST_FIRE FAILED (verification_matrix placeholder"
    f378b_msg="$f378b_msg discrimination) UNMEASURED: the green arm exited 0 but"
    f378b_msg="$f378b_msg printed no parseable 'DENOMINATOR: N matrix row(s)'"
    f378b_msg="$f378b_msg line, so this leg has no row count to report. The"
    f378b_msg="$f378b_msg success message below states that count; stating one"
    f378b_msg="$f378b_msg the arm never printed is the defect this campaign"
    f378b_msg="$f378b_msg files as #216/#310, so the leg fails closed rather"
    f378b_msg="$f378b_msg than claim an unmeasured denominator. Output:"
    f378b_msg="$f378b_msg $(printf '%s\n' "$f378b_green_out" | tr '\n' ' ')"
    no "$f378b_msg"
  else
    f378b_doc_rc=0
    f378b_rid=$(python3 -S - "$f378b_tmp/matrix.json" \
      "$f378b_tmp/VERIFICATION_MATRIX.md" <<'PY'
import json
import sys

matrix_path, doc_path = sys.argv[1], sys.argv[2]
with open(matrix_path, encoding="utf-8") as fh:
    data = json.load(fh)
row = data["rows"][0]
rid = row["id"]
old = row["control_arm"]
row["control_arm"] = "n/a"
with open(matrix_path, "w", encoding="utf-8") as fh:
    fh.write(json.dumps(data, indent=2) + "\n")
with open(doc_path, encoding="utf-8") as fh:
    lines = fh.read().splitlines(keepends=True)
for i, line in enumerate(lines):
    body = line.rstrip("\r\n")
    cells = body.split("|")
    if len(cells) >= 5 and cells[1].strip() == rid:
        if old not in cells[4]:
            sys.exit(4)
        cells[4] = cells[4].replace(old, "n/a", 1)
        lines[i] = "|".join(cells) + line[len(body):]
        break
else:
    sys.exit(3)
with open(doc_path, "w", encoding="utf-8") as fh:
    fh.write("".join(lines))
print(rid)
PY
    ) || f378b_doc_rc=$?
    if [ "$f378b_doc_rc" -ne 0 ] || [ -z "$f378b_rid" ]; then
      rm -rf "$f378b_tmp"
      f378b_msg="MUST_FIRE FAILED (verification_matrix placeholder"
      f378b_msg="$f378b_msg discrimination) UNMEASURED: the placeholder plant"
      f378b_msg="$f378b_msg could not be built -- the first row's id or"
      f378b_msg="$f378b_msg control_arm is missing, or the doc carries no"
      f378b_msg="$f378b_msg table row for that id. Mis-doctoring would make"
      f378b_msg="$f378b_msg the arms differ by more than the one variable,"
      f378b_msg="$f378b_msg so the leg abstains rather than guess; 1 of 2"
      f378b_msg="$f378b_msg arms ran."
      no "$f378b_msg"
    else
      f378b_red_rc=0
      f378b_red_out=$(python3 -S checks/verification_matrix.py \
        --matrix "$f378b_tmp/matrix.json" --doc "$f378b_tmp/VERIFICATION_MATRIX.md" 2>&1) ||
        f378b_red_rc=$?
      rm -rf "$f378b_tmp"
      if [ "$f378b_red_rc" -ne 5 ]; then
        f378b_msg="MUST_FIRE FAILED (verification_matrix placeholder"
        f378b_msg="$f378b_msg discrimination): planting 'n/a' as the"
        f378b_msg="$f378b_msg control_arm of $f378b_rid in both copies gave"
        f378b_msg="$f378b_msg rc=$f378b_red_rc, expected exactly 5 (RED)."
        f378b_msg="$f378b_msg rc=0 would launder a row that cannot fail into"
        f378b_msg="$f378b_msg CLEAR -- the whole of #378 -- and any other"
        f378b_msg="$f378b_msg nonzero is not this gate's declared RED."
        f378b_msg="$f378b_msg Output:"
        f378b_msg="$f378b_msg $(printf '%s\n' "$f378b_red_out" | tr '\n' ' ')"
        no "$f378b_msg"
      elif ! printf '%s\n' "$f378b_red_out" |
        grep -q "C2 CONTROL_ARM_SUBSTANTIVE finding: $f378b_rid"; then
        f378b_msg="MUST_FIRE FAILED (verification_matrix placeholder"
        f378b_msg="$f378b_msg discrimination): the gate exited 5, but no C2"
        f378b_msg="$f378b_msg finding names $f378b_rid -- the one row"
        f378b_msg="$f378b_msg planted. A RED attributed to pre-existing dirt"
        f378b_msg="$f378b_msg is not this control firing, and rc alone"
        f378b_msg="$f378b_msg cannot tell the two apart. Output:"
        f378b_msg="$f378b_msg $(printf '%s\n' "$f378b_red_out" | tr '\n' ' ')"
        no "$f378b_msg"
      else
        f378b_msg="MUST_FIRE verification_matrix placeholder discrimination:"
        f378b_msg="$f378b_msg over the copied ledger ($f378b_denom rows"
        f378b_msg="$f378b_msg under adjudication, 5 checks), the unmodified"
        f378b_msg="$f378b_msg pair exited rc=0 and the same pair with"
        f378b_msg="$f378b_msg $f378b_rid's control_arm planted as 'n/a'"
        f378b_msg="$f378b_msg exited rc=5 with C2 naming that exact row --"
        f378b_msg="$f378b_msg the placeholder the only variable, the live"
        f378b_msg="$f378b_msg files untouched. Both outcomes held"
        ok "$f378b_msg"
      fi
    fi
  fi
fi

# --- ORPHAN DISCHARGE: checks/exit_contract_scope.py -------------------------
# This block is also the suite's call site for checks/exit_contract_scope.py.
# The fix78-orphan leg scans every launchers/*.py and checks/*.py for a call
# site IN THIS SUITE and refuses a file that has none; a Makefile target is
# NOT accepted as a call site. The gate was just added and is indicted BY
# NAME as an orphan -- the three legs below are the in-suite call site that
# discharges the indictment.
echo "== fix381-exit-contract: checks/exit_contract_scope.py real legs =="
# --- finding #381: #380 made loop.train total over the 0/5/95/96 contract,
# and #381 measured the OTHER side of the same handoff -- everything
# cli.main does BEFORE `return train(cfg)` was guarded for ValueError and
# nothing else, so a TypeError in parser construction or config binding
# still reached the interpreter and exited 1, one function earlier.
#
# The behavioural pin lives in tests/train/test_train_boundary_is_total.py.
# This gate is the STATIC half, and the two are not redundant: a test
# covers the sites it was written for, while the defect class is "a
# statement was added and nobody wrapped it" -- #380 found 65 of them at
# once. The gate partitions the whole body instead, over three axes
# (RETURN, MODULE-EXIT, ESCAPE) reported with separate denominators,
# because one collapsed number cannot say which axis went inert.
#
# Three legs. The self-test proves the instrument discriminates on
# fixtures; the live leg is the reading the #381 repair actually claims,
# on this tree; and the discrimination leg re-plants the defect INTO A
# COPY of the shipped cli.py, because a self-test built entirely from
# synthetic fixtures can stay green while the parser has quietly stopped
# understanding the real file.

# --- MUST_PASS: exit_contract_scope self-test --------------------------------
# MEASURED on the commit that introduces the gate: `python3 -S
# checks/exit_contract_scope.py --self-test` exits 0 and its last line reads
# `SELF-TEST DENOMINATOR: 15 of 15 controls behaved; 11 MUST_FIRE, 2
# MUST_PASS, 2 MUST_BE_UNMEASURED`. The floor is a floor: controls may be
# ADDED, never silently dropped, and a shrinking control set is how a
# detector quietly stops discriminating.
f381_floor=15
if [ ! -r "checks/exit_contract_scope.py" ]; then
  f381_msg="MUST_PASS FAILED (exit_contract_scope self-test) UNMEASURED:"
  f381_msg="$f381_msg checks/exit_contract_scope.py is not readable -- unreadable is not empty"
  f381_msg="$f381_msg and it is not clean (doctrine 4); 0 of 1 self-tests measured"
  no "$f381_msg"
else
  f381_rc=0
  f381_out=$(python3 -S checks/exit_contract_scope.py --self-test 2>&1) || f381_rc=$?
  f381_have=$(printf '%s\n' "$f381_out" |
    sed -n 's/^SELF-TEST DENOMINATOR: \([0-9][0-9]*\) of \([0-9][0-9]*\) controls behaved.*/\1/p')
  f381_tot=$(printf '%s\n' "$f381_out" |
    sed -n 's/^SELF-TEST DENOMINATOR: \([0-9][0-9]*\) of \([0-9][0-9]*\) controls behaved.*/\2/p')
  if [ "$f381_rc" -ne 0 ]; then
    f381_msg="MUST_PASS FAILED (exit_contract_scope self-test): rc=$f381_rc under python3 -S."
    f381_msg="$f381_msg A red self-test means the instrument no longer discriminates, so its"
    f381_msg="$f381_msg verdict on the shipped entry points is worth nothing. Output:"
    f381_msg="$f381_msg $(printf '%s\n' "$f381_out" | tr '\n' ' ')"
    no "$f381_msg"
  elif [ -z "$f381_have" ] || [ -z "$f381_tot" ]; then
    f381_msg="MUST_PASS FAILED (exit_contract_scope self-test) UNMEASURED: rc=0 but no"
    f381_msg="$f381_msg 'SELF-TEST DENOMINATOR: N of M controls behaved' line was found -- a"
    f381_msg="$f381_msg control suite that reports no denominator has not reported (doctrine 2)."
    f381_msg="$f381_msg Output: $(printf '%s\n' "$f381_out" | tr '\n' ' ')"
    no "$f381_msg"
  elif [ "$f381_have" != "$f381_tot" ]; then
    f381_msg="MUST_PASS FAILED (exit_contract_scope self-test): $f381_have of $f381_tot controls"
    f381_msg="$f381_msg behaved but rc=0 -- the exit code and the summary disagree, and one side"
    f381_msg="$f381_msg of the contract is lying (doctrine 6)"
    no "$f381_msg"
  elif [ "$f381_tot" -lt "$f381_floor" ]; then
    f381_msg="MUST_PASS FAILED (exit_contract_scope self-test): control set shrank to $f381_tot,"
    f381_msg="$f381_msg below the $f381_floor measured when the gate landed -- controls may be"
    f381_msg="$f381_msg added, never dropped, and a green run over a shrunken set is doctrine 1"
    no "$f381_msg"
  else
    f381_msg="MUST_PASS exit_contract_scope self-test: rc=0 under python3 -S, $f381_have of"
    f381_msg="$f381_msg $f381_tot controls behaved, at or above the $f381_floor-control floor"
    ok "$f381_msg"
  fi
fi

# --- MUST_PASS: the live tree's own verdict ----------------------------------
# The self-test proves the instrument discriminates on fixtures; this leg
# runs it against THIS repository's shipped entry points, which is the
# reading the #381 repair claims. MEASURED on the landing commit: rc=0 with
# `2 PROTECTED, 0 UNPROTECTED, 0 UNMEASURED over 2 entry point(s)`.
#
# Three assertions beyond rc, each closing a different way a green verdict
# could be empty. The ESCAPE axis must parse at all (doctrine 2). Its
# denominator must be nonzero -- a gate whose ENTRY_POINTS tuple was
# emptied would exit 0 over nothing, and #381's own blind spot 4 says the
# denominator is DECLARED, never discovered, so shrinking it is a one-line
# edit no other control would see. And the output must name
# cli.py::main PROTECTED by that literal: that single line IS #381's
# repair, and asserting the aggregate alone would let a future edit
# protect train, drop main from the tuple, and still read 1/1 clean.
if [ ! -r "checks/exit_contract_scope.py" ]; then
  f381_msg="MUST_PASS FAILED (exit_contract_scope live verdict) UNMEASURED:"
  f381_msg="$f381_msg checks/exit_contract_scope.py is not readable -- unreadable is not empty"
  f381_msg="$f381_msg (doctrine 4); 0 of 1 live verdicts measured"
  no "$f381_msg"
else
  f381_rc=0
  f381_out=$(python3 -S checks/exit_contract_scope.py 2>&1) || f381_rc=$?
  f381_prot=$(printf '%s\n' "$f381_out" |
    sed -n 's/^ *axis ESCAPE: *\([0-9][0-9]*\) PROTECTED, \([0-9][0-9]*\) UNPROTECTED, \([0-9][0-9]*\) UNMEASURED over \([0-9][0-9]*\) entry point.*/\1/p')
  f381_eps=$(printf '%s\n' "$f381_out" |
    sed -n 's/^ *axis ESCAPE: *\([0-9][0-9]*\) PROTECTED, \([0-9][0-9]*\) UNPROTECTED, \([0-9][0-9]*\) UNMEASURED over \([0-9][0-9]*\) entry point.*/\4/p')
  if [ "$f381_rc" -ne 0 ]; then
    f381_msg="MUST_PASS FAILED (exit_contract_scope live verdict): rc=$f381_rc on this tree."
    f381_msg="$f381_msg rc=5 means a shipped entry point returns outside 0/5/95/96, exits the"
    f381_msg="$f381_msg module outside it, or has an unguarded statement -- #380/#381's own"
    f381_msg="$f381_msg defect class, back; rc=95 means a declared file went unreadable or"
    f381_msg="$f381_msg unparseable; rc=96 means the gate crashed, which is not a verdict."
    f381_msg="$f381_msg Output: $(printf '%s\n' "$f381_out" | tr '\n' ' ')"
    no "$f381_msg"
  elif [ -z "$f381_prot" ] || [ -z "$f381_eps" ]; then
    f381_msg="MUST_PASS FAILED (exit_contract_scope live verdict) UNMEASURED: rc=0 but no"
    f381_msg="$f381_msg 'axis ESCAPE: N PROTECTED, ... over M entry point(s)' line was found --"
    f381_msg="$f381_msg unparseable is not passing (doctrine 2). Output:"
    f381_msg="$f381_msg $(printf '%s\n' "$f381_out" | tr '\n' ' ')"
    no "$f381_msg"
  elif [ "$f381_eps" -lt 1 ] || [ "$f381_prot" -lt 1 ]; then
    f381_msg="MUST_PASS FAILED (exit_contract_scope live verdict): the ESCAPE axis ran over"
    f381_msg="$f381_msg $f381_eps entry point(s) and found $f381_prot PROTECTED -- zero units is"
    f381_msg="$f381_msg UNMEASURED, never a pass (doctrine 1), and ENTRY_POINTS is DECLARED by"
    f381_msg="$f381_msg hand (blind spot 4), so emptying it is a one-line edit that would"
    f381_msg="$f381_msg otherwise read as clean"
    no "$f381_msg"
  elif ! printf '%s\n' "$f381_out" |
    grep -q '^ *PROTECTED  *src/foundationscale/train/cli\.py::main:'; then
    f381_msg="MUST_PASS FAILED (exit_contract_scope live verdict): rc=0 over $f381_eps entry"
    f381_msg="$f381_msg point(s), but no line reads 'PROTECTED"
    f381_msg="$f381_msg src/foundationscale/train/cli.py::main:'. That one line IS #381's repair"
    f381_msg="$f381_msg -- the pre-handoff region of main() being contained. An aggregate that"
    f381_msg="$f381_msg is clean because main was dropped from ENTRY_POINTS is #381 reopened"
    f381_msg="$f381_msg under a green verdict. Output:"
    f381_msg="$f381_msg $(printf '%s\n' "$f381_out" | tr '\n' ' ')"
    no "$f381_msg"
  else
    f381_msg="MUST_PASS exit_contract_scope live verdict: rc=0 with $f381_prot of $f381_eps"
    f381_msg="$f381_msg declared entry point(s) PROTECTED, and cli.py::main -- the boundary"
    f381_msg="$f381_msg #381 repaired -- named PROTECTED by that literal, not merely counted"
    ok "$f381_msg"
  fi
fi

# --- MUST_FIRE: the pre-#381 shape is RED on a COPY of the shipped file ------
# The self-test's fire legs are built from synthetic fixtures, which is
# the right shape for a control -- but a parser can keep passing its own
# fixtures long after it has stopped understanding the real file (a
# renamed decorator, a walrus, a `match`). This leg plants the defect into
# a COPY of the SHIPPED cli.py and requires the gate to see it there.
#
# The gate takes an optional repository root as a positional, so the copy
# is measured in place and the real tree is never mutated -- #239 was
# exactly a leg committed red against its own target, and a control that
# edits the tree it guards can leave the repository dirty on any early
# exit.
#
# Two arms, and the CLEAR arm is the load-bearing one. Copying three files
# into a scratch root could redden the gate for reasons that have nothing
# to do with the plant (a cross-module import that no longer resolves, a
# file the copy missed), and a fire leg that fires for the wrong reason is
# a pass waiting to happen. So the unmodified copy must read rc=0 FIRST;
# only then does the same copy with one statement planted have to read
# rc=5.
#
# The plant is a source construction, not a data splice: one statement is
# inserted immediately before main's `try:`, which is precisely the shape
# #381 found -- a statement added ahead of the guard. Its line number is
# read back out of the mutated file and the finding must name THAT line,
# so a RED inherited from pre-existing dirt cannot be mistaken for this
# control firing. If the insertion does not self-match, the leg reports
# UNMEASURED rather than passing: a positive control that did not plant
# proves nothing.
f381_tmp=$(mktemp -d 2>/dev/null || mktemp -d -t fs381)
f381_dst="$f381_tmp/src/foundationscale/train"
if [ ! -r "checks/exit_contract_scope.py" ] || [ -z "$f381_tmp" ] || [ ! -d "$f381_tmp" ]; then
  f381_msg="MUST_FIRE UNREACHABLE (exit_contract_scope unguarded pre-handoff statement)"
  f381_msg="$f381_msg UNMEASURED: could not stage a throwaway tree (gate readable? mktemp ok?)"
  f381_msg="$f381_msg -- an unreachable control is a declared state, not a silent pass"
  f381_msg="$f381_msg (doctrine 5)"
  no "$f381_msg"
else
  mkdir -p "$f381_dst"
  f381_staged=1
  for f381_f in src/foundationscale/train/cli.py \
                src/foundationscale/train/loop.py \
                src/foundationscale/train/__main__.py; do
    cp "$f381_f" "$f381_dst/" 2>/dev/null || f381_staged=0
  done
  f381_cli="$f381_dst/cli.py"
  if [ "$f381_staged" -ne 1 ] || [ ! -r "$f381_cli" ]; then
    f381_msg="MUST_FIRE UNREACHABLE (exit_contract_scope unguarded pre-handoff statement)"
    f381_msg="$f381_msg UNMEASURED: could not copy the three declared files (cli.py, loop.py,"
    f381_msg="$f381_msg __main__.py) into the scratch root -- the leg would have measured"
    f381_msg="$f381_msg staging, not the gate (doctrine 5)"
    no "$f381_msg"
  else
    f381_clean_rc=0
    f381_clean_out=$(python3 -S checks/exit_contract_scope.py "$f381_tmp" 2>&1) || f381_clean_rc=$?
    awk 'BEGIN{seen=0; done=0}
         /^def main\(/ {seen=1}
         {
           if (seen==1 && done==0 && $0=="    try:") {
             print "    _fs381_plant = int(\"not a number\")  # planted unguarded statement";
             done=1
           }
           print
         }' "$f381_cli" > "$f381_tmp/cli.planted" && mv "$f381_tmp/cli.planted" "$f381_cli"
    f381_line=$(grep -n '_fs381_plant' "$f381_cli" | head -1 | cut -d: -f1)
    if [ "$f381_clean_rc" -ne 0 ]; then
      f381_msg="MUST_FIRE UNREACHABLE (exit_contract_scope unguarded pre-handoff statement):"
      f381_msg="$f381_msg the UNMODIFIED copy of the three shipped files scored"
      f381_msg="$f381_msg rc=$f381_clean_rc in the scratch root, not 0. The fire arm cannot be"
      f381_msg="$f381_msg attributed to the plant while the control arm is already red -- this"
      f381_msg="$f381_msg is the harness manufacturing a finding, and it measures staging"
      f381_msg="$f381_msg rather than discrimination. Output:"
      f381_msg="$f381_msg $(printf '%s\n' "$f381_clean_out" | tr '\n' ' ')"
      no "$f381_msg"
    elif [ -z "$f381_line" ]; then
      f381_msg="MUST_FIRE UNREACHABLE (exit_contract_scope unguarded pre-handoff statement)"
      f381_msg="$f381_msg UNMEASURED: the plant did not self-match -- no '_fs381_plant' line in"
      f381_msg="$f381_msg the mutated copy, so awk found no '    try:' after 'def main(' and"
      f381_msg="$f381_msg nothing was planted. A positive control that did not plant proves"
      f381_msg="$f381_msg nothing, and reporting it as a pass would be doctrine 5"
      no "$f381_msg"
    else
      f381_red_rc=0
      f381_red_out=$(python3 -S checks/exit_contract_scope.py "$f381_tmp" 2>&1) || f381_red_rc=$?
      if [ "$f381_red_rc" -ne 5 ]; then
        f381_msg="MUST_FIRE FAILED (exit_contract_scope unguarded pre-handoff statement):"
        f381_msg="$f381_msg planting one statement at line $f381_line of main(), ahead of its"
        f381_msg="$f381_msg guard, scored rc=$f381_red_rc, expected exactly 5 (RED). rc=0 means"
        f381_msg="$f381_msg the ESCAPE axis is blind to the exact shape #381 found on the real"
        f381_msg="$f381_msg file; any other nonzero is not this gate's declared RED. Output:"
        f381_msg="$f381_msg $(printf '%s\n' "$f381_red_out" | tr '\n' ' ')"
        no "$f381_msg"
      elif ! printf '%s\n' "$f381_red_out" |
        grep -q "cli\.py::main: unguarded statement at line $f381_line"; then
        f381_msg="MUST_FIRE FAILED (exit_contract_scope unguarded pre-handoff statement): rc=5,"
        f381_msg="$f381_msg but no finding reads 'cli.py::main: unguarded statement at line"
        f381_msg="$f381_msg $f381_line' -- the one line planted. A RED attributed to something"
        f381_msg="$f381_msg else in the copied tree is not this control firing, and rc alone"
        f381_msg="$f381_msg cannot tell the two apart (#233's attribution lesson). Output:"
        f381_msg="$f381_msg $(printf '%s\n' "$f381_red_out" | tr '\n' ' ')"
        no "$f381_msg"
      else
        f381_msg="MUST_FIRE exit_contract_scope unguarded pre-handoff statement: an unmodified"
        f381_msg="$f381_msg copy of the three SHIPPED declared files scored rc=0 in a scratch"
        f381_msg="$f381_msg root, and the same copy with one statement planted at line"
        f381_msg="$f381_msg $f381_line -- immediately before main()'s guard, the pre-#381 shape"
        f381_msg="$f381_msg -- scored rc=5 with the ESCAPE finding naming cli.py::main at that"
        f381_msg="$f381_msg exact line. The plant is the only variable and the live tree was"
        f381_msg="$f381_msg never touched"
        ok "$f381_msg"
      fi
    fi
  fi
  rm -rf "$f381_tmp"
fi

# fs377: the last two controls are about this suite's own published size. They
# run last because assert_documented_control_total counts the controls that
# preceded it, plus itself -- see launchers/_suite_prelude.sh for why this
# countable cannot live in the static census.
assert_control_claims_attributed
assert_documented_control_total test_checks_gates.sh

echo "abstentions: $abstain named (each named at its site above with its denominator; 0 added to pass or fail)"
echo "controls: $pass passed, $fail failed"
[ "$fail" -eq 0 ]
