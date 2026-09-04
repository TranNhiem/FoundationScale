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

