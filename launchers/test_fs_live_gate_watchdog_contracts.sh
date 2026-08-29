#!/bin/bash
# test_fs_live_gate_watchdog_contracts.sh — defect-2 controls for the
# full-FT live save gate's wall-clock watchdog
# (launchers/launch_g4e4b_fullft_1tray.sh :: fs_live_save_gate).
#
# DENOMINATOR: 9 legs, hardwired. Each leg prints PASS k/9 or FAIL with
# its name; summary is k/9; exit status 0 iff 9/9. A leg that cannot run
# is a FAIL (fail closed); no path exists on which zero legs reads green
# (doctrine 1: nothing here iterates over a possibly-empty set).
#
# WHY THIS FILE EXISTS (measured, not inferred): the pre-repair watchdog
# orphaned its `sleep` grandchild when dismissed; the orphan held the
# caller's pipe, and a command-substitution leg of the contract suite
# (contracts:2520) wedged for the full FS_GATE_TIMEOUT_S=600 on an
# INSTANT gate. That leg's failure mode on this defect class was a HANG,
# so on any finite clock it asserted nothing — a control whose red is
# "the suite never comes back" is not a control. Leg 3 keeps its own
# threshold measured with $SECONDS against a small budget (5s vs a
# <0.05s green path, ~100x inside), so that red is OBSERVED, here, in
# seconds; $SECONDS is integer and leg 3's bound still carries at
# least 2s of truncation slack. Legs 5/6 instead discriminate on
# RECORDED MECHANISM, not the clock: clock discrimination false-greened
# a dead KILL path at a window edge equal to the stub's own sleep, and
# false-alarmed under parallel load. They carry $SECONDS only as a
# generous per-leg HANG GUARD whose failure text says so — never as
# evidence — and leg 9 asserts on the detector's REFUSAL, not on time.
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
# shard that owns the LoRA launcher runs the SAME 9 legs against it the
# day the port lands; if that port chooses different marker token text,
# legs 4-7's and 9's token expectations must be re-pinned (leg 7 derives
# both sides from the launcher, legs 4/5/6/9 read $MARKER below).

set -u  # any unset variable is a harness bug: fail loudly, never vacuously

FS_GATE_LAUNCHER="${FS_GATE_LAUNCHER:-launchers/launch_g4e4b_fullft_1tray.sh}"
MARKER='live_gate wall-clock budget exhausted'
EPI_TOK_A='live_gate could not measure:'
EPI_TOK_B='LIVE GATE VERDICT: CLEAR'

# 8 -> 9: leg 9 is the MUST_FIRE do-nothing control for legs 5/6's
# kill-evidence detector (doctrine 3: a detector never observed going
# red is not a control).
LEGS_TOTAL=9
legs_pass=0

pass() { legs_pass=$((legs_pass + 1)); printf 'PASS %d/%d %s\n' "$legs_pass" "$LEGS_TOTAL" "$1"; }
fail() { printf 'FAIL %d/%d leg(%s) — %s\n' "$legs_pass" "$LEGS_TOTAL" "$1" "$2"; }

[[ -r $FS_GATE_LAUNCHER ]] || { printf 'FAIL 0/9 launcher unreadable: %s (fail closed)\n' "$FS_GATE_LAUNCHER"; exit 1; }

WORK="$(mktemp -d "${TMPDIR:-/tmp}/fsgate-watch.XXXXXXXX")" || { echo 'FAIL 0/9 mktemp'; exit 1; }
trap 'rm -rf "$WORK"' EXIT

# --- run_in_container stub, behaviour selected per leg via STUB_MODE ---
STUB_MODE=rc0
STUB_SLEEP=60
# Per-leg evidence file; the default is the bit bucket so a future leg
# that forgets to point it at a real file reads NOTHING back — its
# TERM_SEEN/SURVIVED_TERM/COMPLETED greps come up empty and read red
# (fail closed, doctrine 4).
STUB_TERMLOG=/dev/null
run_in_container() {
  case $STUB_MODE in
    rc*)    [[ ${STUB_MODE#rc} == 0 ]] && printf '%s\n' "$EPI_TOK_B"; return "${STUB_MODE#rc}" ;;
    # Evidence protocol for the kill legs: the stub RECORDS what happened
    # to it — the executor is the one witness the watchdog cannot mint
    # from its own bookkeeping.
    #   TERM_SEEN      appended by a REAL trap handler when TERM is
    #                  delivered (sleep mode's handler then exits — an
    #                  obeyed TERM; ignore mode's returns — an
    #                  observed-but-refused TERM)
    #   SURVIVED_TERM  appended by ignore mode's loop ONLY after the TERM
    #                  handler returned: the executor ran code post-TERM,
    #                  so it cannot have died on the signal — the retired
    #                  7s lower clock bound, done mechanically (an
    #                  executor that honors TERM never writes this)
    #   COMPLETED      appended only when the sleep runs out on its own;
    #                  a SIGKILLed process cannot write it, so ABSENCE is
    #                  positive evidence the kill landed
    # Sleep ticks at 1s because bash defers a trapped signal over a
    # FOREGROUND sleep until that sleep exits; ticks cap handler latency
    # at ~1s (SURVIVED_TERM at ~2s), far inside the gate's 5s grace. No
    # background child: the KILL leaves no orphaned `sleep` grandchild —
    # this file's own header defect class.
    # sleep: TERM-responsive, observably. The handler MUST exit (143 =
    # 128+SIGTERM, an honest code that cannot masquerade as success),
    # else this mode is indistinguishable from ignore.
    sleep)  trap 'printf "TERM_SEEN\n" >>"$STUB_TERMLOG"; exit 143' TERM
            tick=0
            while [[ $tick -lt $STUB_SLEEP ]]; do sleep 1; tick=$((tick + 1)); done
            printf 'COMPLETED\n' >>"$STUB_TERMLOG" ;;
    # ignore: TERM is OBSERVED but refused — record and KEEP SLEEPING.
    # A bare `trap '' TERM` (the old stub) sets SIG_IGN: no handler can
    # ever run, so it could record NOTHING; evidence must not look like
    # that. This handler does NOT exit, or the leg tests the wrong
    # estate (a TERM-responsive executor is leg 5's shape).
    ignore) trap 'printf "TERM_SEEN\n" >>"$STUB_TERMLOG"; stub_saw_term=1' TERM
            tick=0; stub_saw_term=0
            while [[ $tick -lt $STUB_SLEEP ]]; do
              sleep 1; tick=$((tick + 1))
              if [[ $stub_saw_term == 1 ]]; then
                printf 'SURVIVED_TERM\n' >>"$STUB_TERMLOG"; stub_saw_term=2
              fi
            done
            printf 'COMPLETED\n' >>"$STUB_TERMLOG" ;;
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

# --- leg 5: MUST_FIRE, TERM-responsive executor — mechanism, not clock ---
# DISCRIMINATES ON MECHANISM, NOT CLOCK (same defect class as leg 6;
# changed on the numbers: the retired `delta -le 8` left ~4-5s of
# headroom over the ~3s mechanism, less than the ~6-7s load stretch
# that red'ed leg 6's window under parallel load — M4's false-ALARM
# half. The do-nothing-GREEN half was REFUTED: STUB_SLEEP=60 lands far
# outside 8s and a never-fired watchdog also dies on the rc == 124
# conjunct, so no survived/KILL pair is needed here):
#   TERM_SEEN present  — the watchdog's TERM was DELIVERED (recorded by
#                        the stub's own handler, not inferred from time).
#   COMPLETED absent   — the stub did not run out its own 60s clock; it
#                        died BECAUSE of the TERM (handler exits 143).
# rc 124 and the marker stay conjuncts. The clock is retained ONLY as a
# hang guard at 30s (~10x the mechanism, past the measured ~2x load
# class) — never the discriminator; its red says so, and doctrine 5
# prices a false alarm like a false green: do NOT retighten it.
cap="$WORK/cap.slow1"; : >"$cap"
STUB_TERMLOG="$WORK/ev.slow1"; : >"$STUB_TERMLOG"
STUB_MODE=sleep; STUB_SLEEP=60; FS_GATE_TIMEOUT_S=3
start=$SECONDS
fs_live_save_gate /iter save "$WORK/rep.slow1.json" "$cap"
rc=$?
delta=$((SECONDS - start))
if [[ $rc != 124 ]]; then
  fail 'MUST_FIRE/TERM' "rc=$rc (want 124 from the watchdog timeout path)"
elif ! grep -qF "$MARKER" "$cap"; then
  fail 'MUST_FIRE/TERM' "marker absent — watchdog timeout bookkeeping never ran; capture: $(cat "$cap")"
elif ! grep -qF 'TERM_SEEN' "$STUB_TERMLOG"; then
  fail 'MUST_FIRE/TERM' "TERM_SEEN absent from $STUB_TERMLOG — the watchdog's TERM never reached the executor; its exit was not caused by the watchdog"
elif grep -qF 'COMPLETED' "$STUB_TERMLOG"; then
  fail 'MUST_FIRE/TERM' "COMPLETED present — the stub slept out its own 60s (do-nothing outcome); nothing terminated it"
elif [[ $delta -gt 30 ]]; then
  fail 'MUST_FIRE/TERM' "HANG GUARD ONLY, not evidence about the mechanism: ${delta}s vs ~3s mechanism (budget 3) — the leg wedged; do NOT retighten this against machine load"
else
  pass "leg5 MUST_FIRE/TERM — executor obeyed TERM (TERM_SEEN recorded by the stub, not inferred from the clock), COMPLETED absent, marker, rc 124"
fi

# --- leg 6: MUST_FIRE, TERM-IGNORING executor — grace+KILL path ---
# THE PROPERTY IS MECHANICAL: "TERM was delivered, the process ignored
# it, SIGKILL was required". The stub, not the clock, answers:
#   TERM_SEEN present      — TERM was DELIVERED and observed (handler
#                            appends and keeps sleeping). Its absence
#                            also reds a watchdog that KILL'd at budget
#                            with no grace phase, and one that never
#                            fired at all.
#   SURVIVED_TERM present  — the executor ran code AFTER the TERM
#                            handler returned, so it did NOT die on
#                            TERM: the retired 7s lower bound, done
#                            mechanically (TERM_SEEN alone greens an
#                            executor that honored TERM — leg 5's exact
#                            shape — so the KILL path cannot rot).
#   COMPLETED absent       — the stub did not run out its own clock; a
#                            SIGKILLed process cannot append it.
#                            COMPLETED PRESENT is the do-nothing outcome
#                            (M1) and reds this leg no matter the clock.
# rc 124 and the marker stay conjuncts: necessary bookkeeping, never
# sufficient — the watchdog mints both on a dead KILL path.
# STUB_SLEEP=120: natural completion sits 15x beyond the ~8s budget+grace
# mechanism (the old defect had STUB_SLEEP 15 EQUAL to the window's
# upper bound), unreachable at any plausible load — the worst measured
# stretch was ~2x. Green-path cost is 0s: the KILL lands at ~8s and the
# rest of the sleep never happens; a red path pays at most 120s. Ticks
# of 1s put TERM_SEEN and SURVIVED_TERM on disk far inside the 5s grace.
# The clock survives ONLY as a HANG GUARD at 60s: ~8x the mechanism,
# past the measured load class, and never evidence — do NOT retighten
# it into one; doctrine 5 prices a false alarm like a false green.
cap="$WORK/cap.slow2"; : >"$cap"
STUB_TERMLOG="$WORK/ev.slow2"; : >"$STUB_TERMLOG"
STUB_MODE=ignore; STUB_SLEEP=120; FS_GATE_TIMEOUT_S=3
start=$SECONDS
fs_live_save_gate /iter save "$WORK/rep.slow2.json" "$cap"
rc=$?
delta=$((SECONDS - start))
if [[ $rc != 124 ]]; then
  fail 'MUST_FIRE/grace+KILL' "rc=$rc (want 124 from the watchdog timeout path)"
elif ! grep -qF "$MARKER" "$cap"; then
  fail 'MUST_FIRE/grace+KILL' "marker absent — watchdog timeout bookkeeping never ran; capture: $(cat "$cap")"
elif ! grep -qF 'TERM_SEEN' "$STUB_TERMLOG"; then
  fail 'MUST_FIRE/grace+KILL' "TERM_SEEN absent — TERM never observed by the executor: KILL-at-budget or never-fired watchdog; the grace phase is unproven"
elif ! grep -qF 'SURVIVED_TERM' "$STUB_TERMLOG"; then
  fail 'MUST_FIRE/grace+KILL' "SURVIVED_TERM absent — the executor died on TERM, so the grace+KILL path never ran (the retired 7s lower clock bound, done mechanically)"
elif grep -qF 'COMPLETED' "$STUB_TERMLOG"; then
  fail 'MUST_FIRE/grace+KILL' "COMPLETED present — the stub slept out its own 120s; the KILL never landed (M1's do-nothing outcome, unreachable on a live gate at any load)"
elif [[ $delta -gt 60 ]]; then
  fail 'MUST_FIRE/grace+KILL' "HANG GUARD ONLY, not evidence about the kill: ${delta}s vs ~8s mechanism (budget 3 + grace 5) — the leg wedged; machine load cannot red a 60s bound; do NOT retighten"
else
  pass "leg6 MUST_FIRE/grace+KILL — TERM delivered and ignored (TERM_SEEN + SURVIVED_TERM recorded by the stub), COMPLETED absent (SIGKILL did the work), marker, rc 124 — kill evidence is mechanical, not temporal"
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

# --- leg 9: MUST_FIRE control for legs 5/6's detector — the do-nothing case ---
# doctrine 3 applied to the NEW evidence: legs 5/6's recorded-mechanism
# predicate must be OBSERVED refusing the case M1 showed the old clock
# window could not exclude. BUILD the do-nothing case explicitly: the
# watchdog budget (30s) is LONGER than the stub's own sleep (1s), so the
# stub completes naturally under an idle watchdog. Then demand the WHOLE
# state, each conjunct observed reading the same thing for its own
# reason: rc 0 (no timeout was minted), marker absent (no bookkeeping
# ran), TERM_SEEN absent and SURVIVED_TERM absent (no signal was ever
# sent), and COMPLETED present — proof the stub really ran out its own
# clock, because a refusal measured over a stub that never ran would
# certify nothing (doctrine 1). Budget 30 vs sleep 1 puts this state
# outside the measured load class (the flakes stretched mechanisms
# ~2x; this margin is ~30x). ~1s of wall clock. Keep this conjunct set
# in lockstep with legs 5/6 if their conjuncts ever change.
cap="$WORK/cap.donothing"; : >"$cap"
STUB_TERMLOG="$WORK/ev.donothing"; : >"$STUB_TERMLOG"
STUB_MODE=ignore; STUB_SLEEP=1; FS_GATE_TIMEOUT_S=30
fs_live_save_gate /iter save "$WORK/rep.donothing.json" "$cap"
rc=$?
if [[ $rc == 0 ]] && ! grep -qF "$MARKER" "$cap" \
   && ! grep -qF 'TERM_SEEN' "$STUB_TERMLOG" && ! grep -qF 'SURVIVED_TERM' "$STUB_TERMLOG" \
   && grep -qF 'COMPLETED' "$STUB_TERMLOG"; then
  pass 'leg9 do-nothing control — budget 30 > sleep 1: stub completed naturally under an idle watchdog and every legs-5/6 kill conjunct reads FALSE on the evidence (detector observed going red)'
else
  fail 'do-nothing control' "do-nothing state unclean or kill predicate not refused on it: rc=$rc marker_count=$(grep -cF "$MARKER" "$cap") TERM_SEEN=$(grep -cF 'TERM_SEEN' "$STUB_TERMLOG") SURVIVED_TERM=$(grep -cF 'SURVIVED_TERM' "$STUB_TERMLOG") COMPLETED=$(grep -cF 'COMPLETED' "$STUB_TERMLOG") (want rc 0, all counts 0 except COMPLETED=1) — legs 5/6 could certify a dead KILL path; M1's hole is OPEN"
fi

printf 'WATCHDOG CONTRACTS: PASS %d/%d\n' "$legs_pass" "$LEGS_TOTAL"
[[ $legs_pass == "$LEGS_TOTAL" ]]
