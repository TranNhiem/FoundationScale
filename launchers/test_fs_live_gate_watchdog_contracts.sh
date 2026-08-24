#!/bin/bash
# test_fs_live_gate_watchdog_contracts.sh — defect-2 controls for the
# full-FT live save gate's wall-clock watchdog
# (launchers/launch_g4e4b_fullft_1tray.sh :: fs_live_save_gate).
#
# DENOMINATOR: 8 legs, hardwired. Each leg prints PASS k/8 or FAIL with
# its name; summary is k/8; exit status 0 iff 8/8. A leg that cannot run
# is a FAIL (fail closed); no path exists on which zero legs reads green
# (doctrine 1: nothing here iterates over a possibly-empty set).
#
# WHY THIS FILE EXISTS (measured, not inferred): the pre-repair watchdog
# orphaned its `sleep` grandchild when dismissed; the orphan held the
# caller's pipe, and a command-substitution leg of the contract suite
# (contracts:2520) wedged for the full FS_GATE_TIMEOUT_S=600 on an
# INSTANT gate. That leg's failure mode on this defect class was a HANG,
# so on any finite clock it asserted nothing — a control whose red is
# "the suite never comes back" is not a control. Every timed leg below
# therefore carries its own threshold measured with $SECONDS against a
# small budget, so the red is OBSERVED, here, in seconds. Thresholds are
# justified per leg; $SECONDS is integer, so every bound carries at
# least 2s of truncation slack, and green paths are 50-100x inside them.
#
# THE FUNCTION UNDER TEST IS EXTRACTED FROM THE SHIPPED LAUNCHER (sed
# range below), never pasted into this file: a pasted copy goes dead
# silent the day the launcher moves (the estate's dead-control hazard).
# Extraction boundaries: the function opens at column 0 as
# `fs_live_save_gate() {` and its first column-0 `}` closes it (every
# interior control line is indented; the body holds no column-0 `}`).
# Leg 1 verifies the extraction before anything runs against it.
#
# LoRA asymmetry, ON RECORD (defect-2 work item 5): the LoRA launcher's
# fs_live_save_gate ships NO watchdog at all (measured in the defect
# statement: 0 occurrences of fs_watch_pid / FS_GATE_TIMEOUT_S in its
# gate). The hole the bound closes — a wedged gate never returning, so
# the tripwire loop silently never checks again, a loss of protection
# invisible in any log — is launcher-agnostic, and nothing measured
# shows the LoRA watcher awaiting its gate under any EXTERNAL bound.
# LoRA is therefore NOT exempt on merit; the absence of the bound there
# is an open hole. The port ITSELF is ABSTAINED BY NAME in this shard
# ("LoRA watchdog port") for exactly one reason: this shard was not
# furnished the LoRA gate's verbatim bytes, and a fabricated OLD block
# is the estate's recurring applier-refusal defect — doctrine 5 applied
# to the patch itself. This file accepts FS_GATE_LAUNCHER=<path> so the
# shard that owns the LoRA launcher runs the SAME 8 legs against it the
# day the port lands; if that port chooses different marker token text,
# legs 4-7's token expectations must be re-pinned (leg 7 derives both
# sides from the launcher, legs 4/5/6 read $MARKER below).

set -u  # any unset variable is a harness bug: fail loudly, never vacuously

FS_GATE_LAUNCHER="${FS_GATE_LAUNCHER:-launchers/launch_g4e4b_fullft_1tray.sh}"
MARKER='live_gate wall-clock budget exhausted'
EPI_TOK_A='live_gate could not measure:'
EPI_TOK_B='LIVE GATE VERDICT: CLEAR'

LEGS_TOTAL=8
legs_pass=0

pass() { legs_pass=$((legs_pass + 1)); printf 'PASS %d/%d %s\n' "$legs_pass" "$LEGS_TOTAL" "$1"; }
fail() { printf 'FAIL %d/%d leg(%s) — %s\n' "$legs_pass" "$LEGS_TOTAL" "$1" "$2"; }

[[ -r $FS_GATE_LAUNCHER ]] || { printf 'FAIL 0/8 launcher unreadable: %s (fail closed)\n' "$FS_GATE_LAUNCHER"; exit 1; }

WORK="$(mktemp -d "${TMPDIR:-/tmp}/fsgate-watch.XXXXXXXX")" || { echo 'FAIL 0/8 mktemp'; exit 1; }
trap 'rm -rf "$WORK"' EXIT

# --- run_in_container stub, behaviour selected per leg via STUB_MODE ---
STUB_MODE=rc0
STUB_SLEEP=60
run_in_container() {
  case $STUB_MODE in
    rc*)    [[ ${STUB_MODE#rc} == 0 ]] && printf '%s\n' "$EPI_TOK_B"; return "${STUB_MODE#rc}" ;;
    sleep)  sleep "$STUB_SLEEP" ;;
    ignore) trap '' TERM; sleep "$STUB_SLEEP" ;;
  esac
}

# env the function's payload string expands (unused by the stub, but set:
# unset would abort under set -u before the function is even tested)
REPO="$WORK"; FS_ROOT="$WORK"; HF_MODEL="$WORK"; RESOLVED_CFG="$WORK/cfg"; FQN_MAP="$WORK/map"

# --- extract and source the SHIPPED function ---
sed -n '/^fs_live_save_gate() {/,/^}$/p' "$FS_GATE_LAUNCHER" >"$WORK/gate.fn"
. "$WORK/gate.fn"

# --- leg 1: harness integrity (fail-closed precondition, doctrine 1/4) ---
if [[ -s $WORK/gate.fn ]] && [[ $(tail -n 1 "$WORK/gate.fn") == '}' ]] \
   && [[ $(wc -l <"$WORK/gate.fn" | tr -d ' ') -ge 40 ]] \
   && type fs_live_save_gate >/dev/null 2>&1; then
  pass 'leg1 harness integrity — fs_live_save_gate extracted from the SHIPPED launcher and defined'
else
  fail 'extraction' 'gate.fn empty/truncated or function undefined — refusing to run legs against nothing'
  printf 'WATCHDOG CONTRACTS: PASS %d/%d\n' "$legs_pass" "$LEGS_TOTAL"
  exit 1
fi

# --- leg 2: rc passthrough sweep, 0->0 1->1 3->3 127->127, untouched ---
ok2=0
for want in 0 1 3 127; do
  STUB_MODE="rc$want"; FS_GATE_TIMEOUT_S=30
  cap="$WORK/cap.rc$want"; : >"$cap"
  fs_live_save_gate /iter save "$WORK/rep.rc$want.json" "$cap"
  got=$?
  if [[ $got == "$want" ]]; then ok2=$((ok2 + 1)); else fail 'rc-passthrough' "stub rc $want -> gate rc $got"; fi
done
[[ $ok2 == 4 ]] && pass 'leg2 rc passthrough 4/4 (0->0 1->1 3->3 127->127, bit-exact)'

# --- leg 3: MUST-PASS on elapsed time — the measured reproduction, as detector ---
# budget 30, threshold 5: green path is builtins + instant stub (<0.05s
# real, ~100x headroom); the pre-repair code reads 29-30s here no matter
# the truncation — the 600s wedge scaled 1/20 and OBSERVED as red.
cap="$WORK/cap.timed"; : >"$cap"
STUB_MODE=rc3; FS_GATE_TIMEOUT_S=30
start=$SECONDS
timed_out="$( ( fs_live_save_gate /iter save "$WORK/rep.timed.json" "$cap"; printf 'gate_rc=%s\n' "$?" ) 2>&1 )"
delta=$((SECONDS - start))
if [[ $timed_out == 'gate_rc=3' && $delta -le 5 ]]; then
  pass "leg3 \$( )-caller returns promptly — ${delta}s elapsed vs 30s budget (rc 3 passed through the dead gate's plumbing)"
else
  fail 'elapsed-time positive control' "out=[$timed_out] elapsed=${delta}s (threshold 5; pre-repair reads 29-30 — this leg is the defect's own wedge, armed with a deadline)"
fi

# --- leg 4: consequence-(i) tripwire — a CLEARED gate's record stays clean ---
# budget 3; wait budget+grace+3 so any surviving watchdog code has had
# every chance to discharge; then demand: rc 0, real CLEAR line present,
# marker ABSENT. Fires on the production TERM-deferral semantics where
# the false append was measured, and on the trap-misses-`exit 0` mutant.
cap="$WORK/cap.clear"; : >"$cap"
STUB_MODE=rc0; FS_GATE_TIMEOUT_S=3
fs_live_save_gate /iter save "$WORK/rep.clear.json" "$cap"
rc_clear=$?
sleep 8
if [[ $rc_clear == 0 ]] && ! grep -qF "$MARKER" "$cap" && grep -qF "$EPI_TOK_B" "$cap"; then
  pass 'leg4 cleared gate leaves a clean record — rc 0, CLEAR present, marker absent after budget+grace'
else
  fail 'marker-absence' "rc=$rc_clear; capture now reads: $(cat "$cap")"
fi

# --- leg 5: MUST_FIRE, TERM-responsive executor — bound, marker, 124 ---
# mechanism ~3s (budget 3, TERM kills the stub instantly), bound 8s.
cap="$WORK/cap.slow1"; : >"$cap"
STUB_MODE=sleep; STUB_SLEEP=60; FS_GATE_TIMEOUT_S=3
start=$SECONDS
fs_live_save_gate /iter save "$WORK/rep.slow1.json" "$cap"
rc=$?
delta=$((SECONDS - start))
if [[ $rc == 124 ]] && grep -qF "$MARKER" "$cap" && [[ $delta -le 8 ]]; then
  pass "leg5 MUST_FIRE/TERM — over-budget executor TERM'd, marker recorded, rc 124, in ${delta}s (budget 3)"
else
  fail 'MUST_FIRE/TERM' "rc=$rc elapsed=${delta}s marker_count=$(grep -cF "$MARKER" "$cap")"
fi

# --- leg 6: MUST_FIRE, TERM-IGNORING executor — grace+KILL path ---
# mechanism ~8s (budget 3 + grace 5); window [7,15]: the LOWER bound is
# doctrine 5 on the leg itself — an executor that accidentally honored
# TERM finishes ~3s and goes red here, so the KILL path cannot rot.
cap="$WORK/cap.slow2"; : >"$cap"
STUB_MODE=ignore; STUB_SLEEP=15; FS_GATE_TIMEOUT_S=3
start=$SECONDS
fs_live_save_gate /iter save "$WORK/rep.slow2.json" "$cap"
rc=$?
delta=$((SECONDS - start))
if [[ $rc == 124 ]] && grep -qF "$MARKER" "$cap" && [[ $delta -ge 7 && $delta -le 15 ]]; then
  pass "leg6 MUST_FIRE/grace+KILL — TERM-ignoring executor KILL'd at budget+grace, marker recorded, rc 124, in ${delta}s"
else
  fail 'MUST_FIRE/grace+KILL' "rc=$rc elapsed=${delta}s (want 7..15 for budget 3 + grace 5) marker_count=$(grep -cF "$MARKER" "$cap")"
fi

# --- leg 7: pin the stated epilogue non-consequence (consequence i, rc paths) ---
# "fs_gate_verdict_to_rc cannot be moved by the injected line" rests on
# TEXT DISJUNCTION. Pin it: every marker-bearing line in the launcher
# must contain NEITHER epilogue token. If a future edit to the marker
# prose wanders into either token, this goes red instead of the claim
# going stale.
if grep -F "$MARKER" "$FS_GATE_LAUNCHER" | grep -qF -e "$EPI_TOK_A" -e "$EPI_TOK_B"; then
  fail 'token-overlap' "a marker-bearing line now contains '$EPI_TOK_A' or '$EPI_TOK_B' — the epilogue non-consequence claim just went FALSE"
elif grep -qF "$MARKER" "$FS_GATE_LAUNCHER"; then
  pass "leg7 marker/epilogue token disjunction — no '$MARKER' line contains '$EPI_TOK_A' or '$EPI_TOK_B'"
else
  fail 'marker-presence' "marker token '$MARKER' absent from the launcher entirely — is the watchdog still wired?"
fi

# --- leg 8: invalid knob refuses closed — never an unbounded gate ---
# STUB_MODE=rc0 deliberately: if the gate WRONGLY ran, the stub's CLEAR
# line would land in the capture; its ABSENCE proves refusal happened
# BEFORE any executor spawn — a minted 124 must not carry the tool's
# verdict as camouflage.
cap="$WORK/cap.badknob"; : >"$cap"
STUB_MODE=rc0; FS_GATE_TIMEOUT_S=abc
fs_live_save_gate /iter save "$WORK/rep.badknob.json" "$cap"
rc=$?
if [[ $rc == 124 ]] && grep -qF 'refusing to run the gate UNBOUNDED' "$cap" && ! grep -qF "$EPI_TOK_B" "$cap"; then
  pass 'leg8 invalid FS_GATE_TIMEOUT_S refuses closed — rc 124, refusal narrated to capture, executor never spawned'
else
  fail 'invalid-knob' "rc=$rc; capture: $(cat "$cap")"
fi

printf 'WATCHDOG CONTRACTS: PASS %d/%d\n' "$legs_pass" "$LEGS_TOTAL"
[[ $legs_pass == "$LEGS_TOTAL" ]]
