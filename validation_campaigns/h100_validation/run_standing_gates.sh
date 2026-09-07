#!/usr/bin/env bash
# run_standing_gates.sh — the automated runner for the standing gate plane (finding #293).
#
# build_h100_plane.sh invokes nine standing gate_*.py scripts near its tail, in the sections
# introduced by "=== standing gate:" banners. Those nine gates have ZERO automated runners:
# no make target and no CI step reaches them. They run only when a human types the full
# build by hand — and that build is slow and, while finding #294 remains open, DESTRUCTIVE:
# it deletes tracked artifacts when a stage refuses. So the nine gates are, in practice,
# unreachable. This script is the automated runner: it runs each standing gate DIRECTLY —
# read-only, non-destructively, in seconds — and reports one verdict over all of them.
#
# SCOPE, stated honestly. This closes the "no automated runner" half of #293. It does NOT
# exercise the build's own roll-call arithmetic — the note_gate / roll_call_gates
# accumulation inside build_h100_plane.sh — because exercising that means running the build,
# and running the build must wait for #294 to make it non-destructive. A full-build smoke
# test is the other half of #293 and is deliberately NOT attempted here. Anything this
# script says about gate VERDICTS is measured; nothing it says about the build's own
# bookkeeping is.
#
# THE DENOMINATOR IS DERIVED, NEVER HARDCODED. The list of gates is extracted from
# build_h100_plane.sh using the exact same anchored pattern the build itself uses at its
# GATE_COUNT= line:
#
#     grep -c '^python3 gate_[a-z_0-9]*\.py || {' build_h100_plane.sh
#
# One derivation rule, two consumers: a hand-kept "9" is a countable that goes stale the
# first time a gate is added or removed, and this campaign has filed that exact class at
# #194, #220, #233 and #266. We do NOT compare our count against the build's GATE_COUNT:
# both numbers come from the same grep on the same file, so that comparison is true by
# construction — a vacuous check, which is finding #203. What must NOT be vacuous is
# whether the derivation found anything at all:
#
#   * an EMPTY derived list exits 96 (CANNOT-MEASURE). A runner that scans a file, finds
#     no gates, and prints "0 red, all clear" is the exact false-green this campaign keeps
#     finding (#157, #233, #241, #303). An empty denominator is never a pass.
#   * a derived gate filename that does not exist on disk, or exists ZERO-BYTE, exits 96,
#     naming the offending files. A gate that cannot be run — or that runs and measures
#     nothing, as python3 does on an empty file — has certified nothing.
#   * a LOOSE-vs-STRICT cross-check exits 96 when a deliberately looser scan for the same
#     idiom sees MORE gate lines in the build than the strict anchored pattern does. The
#     strict shape going quietly narrow — a double space, a flag, an uppercase letter —
#     is how a gate never runs while the runner prints a confident N/N.
#
# The grep is /usr/bin/grep, never bare grep: on the developer machine bare grep can be
# ugrep honouring .gitignore, which would silently shrink the denominator — and a shrunken
# denominator reads exactly like a clean scan.
#
# THE FOUR-STATE CONTRACT is the same one note_gate applies in the build:
#   rc 0        -> green
#   rc 5        -> red      (a finding about the tree)
#   rc 95 or 96 -> refused  (UNMEASURED / CANNOT-MEASURE — an input was absent, e.g. the
#                            estate environment)
#   anything else -> red, with a warning naming the unexpected rc (the note_gate `*)` arm)
#
# EXIT VERDICT over the whole plane:
#   any red                     -> exit 5
#   else a refusal NOT declared -> exit 5   (see --allow-refused)
#   else any refusal at all     -> exit 95, naming the refused gates
#   else                        -> exit 0, "STANDING GATES GREEN — N/N"
#
# EXPECTED OUTCOMES, MEASURED on the current tree, so nobody "fixes" the wrong thing:
#   * Bare checkout, no estate environment sourced: 7 gates green; gate_launch_contract.py
#     and gate_launch_doc.py exit 96 because they require FS_ESTATE_ROOT and
#     FS_ESTATE_IDENT_PAT. This script then exits 95. That is the CORRECT, expected CI
#     outcome for a bare checkout — 95 says "two gates could not measure", NOT "a gate
#     found a defect". Do not treat 95 as a failure and do not silence it: a gate that
#     could not measure has certified nothing, and collapsing 95 into 5 is the #56/#149/#160
#     defect this contract exists to prevent.
#   * With the estate environment sourced: all 9 gates green, exit 0.
#
# WHY --allow-refused EXISTS. 95 is the honest verdict, and it is also unusable as a CI
# signal on its own: a job that accepts 95 accepts EVERY refusal, including one that
# appears tomorrow because a gate broke. --allow-refused NAMES the gates permitted to
# abstain here. A refusal inside that set is a declared abstention and does not sink the
# run; a refusal from any gate outside it is RED, because an abstention nobody declared is
# news. The set is a ceiling, not an equality — measuring MORE than promised is never a
# failure. Passing the flag at all switches the verdict from "95 is tolerable" to "these
# specific gates may abstain and nothing else may", which is the difference between a
# threshold and a contract.
#
# --self-test builds SYNTHETIC campaign directories under mktemp -d — never touching the
# real tree, which is #294's lesson: never mutate the tree you are certifying — and
# re-invokes this script against them with eleven controls (MUST_FIRE, MUST_PASS,
# REFUSAL-IS-NOT-RED, VACUOUS-DENOMINATOR, UNDECLARED-REFUSAL-IS-RED,
# DECLARED-REFUSAL-IS-GREEN, STALE-ALLOWANCE, UNEXPECTED-RC, RED-BEATS-REFUSED,
# LOOSE-STRICT-GAP, COMMENTED-IS-NOT-A-GATE). A broken control is a finding about the
# instrument, so any control failure exits 5.
#
# macOS bash 3.2 compatible: no mapfile/readarray, no associative arrays, no ${x^^}.
# No line-number citations anywhere (#303: coordinates drift); sections are cited by name
# or by quoting a unique anchoring pattern.
set -Eeuo pipefail

usage() {
  cat <<'EOF'
usage: run_standing_gates.sh [--campaign-dir DIR] [--allow-refused LIST] [--self-test]

  --campaign-dir DIR   directory holding build_h100_plane.sh and the gate_*.py files
                       (default: the directory containing this script)
  --allow-refused LIST comma-separated gate filenames that are PERMITTED to refuse
                       (exit 95/96) in this environment. Refusals confined to this set
                       do not sink the run; a refusal from any gate OUTSIDE it is RED.
  --self-test          run the eleven instrument controls against synthetic campaign
                       directories built under mktemp -d; does not touch the real tree
EOF
}

# --allow-refused is a DECLARED ABSTENTION, which is #56's remedy shape: an abstaining
# check must be a state someone wrote down, not a silent pass. Two properties matter.
#
# It names gates instead of counting them. `--max-refused 2` is satisfied by the WRONG
# two — the #199/#233 lesson that identity beats arithmetic. If gate_env_drift starts
# refusing while gate_launch_contract starts passing, a count still reads 2 and says
# nothing; a named set reds immediately.
#
# It is a CEILING, not an equality. The allowed set is what MAY refuse, not what MUST.
# An equality test would red the developer machine, where the estate environment is
# present and nothing refuses at all — punishing the better-measured environment is how
# a gate teaches people to unset things. A subset always passes; an escape never does.
#
# ALLOW_GIVEN tracks whether the flag was PASSED, separately from whether its value is
# non-empty. `--allow-refused ''` is a real and different declaration — "no gate may
# abstain here" — and without the sentinel it would be indistinguishable from not passing
# the flag at all, which is the weaker "95 is tolerable" default. Absence and emptiness
# are two states; a single variable can only hold one of them.
ALLOW_REFUSED=""
ALLOW_GIVEN=0
CAMPAIGN_DIR=""
SELF_TEST=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --campaign-dir)
      # An EMPTY value is rejected exactly like a MISSING one: `--campaign-dir ""` would
      # otherwise fall through to the -z default below and silently certify this script's
      # own directory instead of the caller's intended tree — a wrong-tree run that
      # reads exactly like a right-tree one.
      [[ $# -ge 2 && -n "$2" ]] || { echo "run_standing_gates: --campaign-dir needs a non-empty value" >&2; exit 2; }
      CAMPAIGN_DIR=$2
      shift 2
      ;;
    --allow-refused)
      [[ $# -ge 2 ]] || { echo "run_standing_gates: --allow-refused needs a value" >&2; exit 2; }
      ALLOW_REFUSED=$2
      ALLOW_GIVEN=1
      shift 2
      ;;
    --self-test)
      SELF_TEST=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "run_standing_gates: unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

# Absolute path to THIS script, resolved before any cd, so --self-test can re-invoke it
# against synthetic campaign directories wherever they land.
SELF_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SELF="$SELF_DIR/$(basename "${BASH_SOURCE[0]}")"

# ---------------------------------------------------------------------------
# --self-test: the eleven controls. Each one exists because of a specific way this
# runner could lie, stated with the control below. The synthetic gates are stubs that
# exit a chosen code; the synthetic build_h100_plane.sh carries lines of the form
# `python3 gate_xxx.py || {` so the REAL derivation rule is what finds them — a control
# that bypassed the derivation would certify nothing about it. Control 10 additionally
# carries a line the strict rule does NOT match, to exercise the loose-vs-strict
# cross-check rather than the derivation alone.
# ---------------------------------------------------------------------------
if [[ "$SELF_TEST" == 1 ]]; then
  TMPROOT=$(mktemp -d)
  # Clean up on every exit path, including a control that dies under set -e.
  trap 'rm -rf "$TMPROOT"' EXIT

  controls_passed=0
  controls_failed=0
  # Stated once and consumed by both the per-control line and the summary, so a twelfth
  # control cannot be added while the banner still says eleven — a self-inflicted stale
  # countable of exactly the #220/#233 shape.
  CONTROL_TOTAL=11

  # make_stub_campaign <dir> <name:rc>... — write a stub build file whose gate lines the
  # anchored derivation will find, plus a stub gate script per name that exits <rc>.
  make_stub_campaign() {
    local d=$1; shift
    mkdir -p "$d"
    : > "$d/build_h100_plane.sh"
    local pair name grc
    for pair in "$@"; do
      name=${pair%%:*}
      grc=${pair##*:}
      printf 'python3 %s || {\n' "$name" >> "$d/build_h100_plane.sh"
      printf 'import sys\nprint("stub gate %s exiting %s")\nsys.exit(%s)\n' \
        "$name" "$grc" "$grc" > "$d/$name"
    done
  }

  # run_control <n> <label> <dir> <want-rc> <needle-or-empty> [extra child args...]
  # Re-invokes THIS script against the synthetic dir. --self-test is deliberately NOT
  # forwarded: a child that re-ran the controls would recurse without bound, and a guard
  # that is never stated is a guard that gets removed by the next editor.
  #
  # The child is invoked as `bash "$SELF"`, not `"$SELF"`: direct exec depends on the
  # file's executable bit, and on a checkout that lost the +x bit every control would
  # return rc=126 and the self-test would report failures over a perfectly healthy
  # instrument — a false RED about the instrument itself. The CI step already invokes
  # this script via `bash`, so this also makes the child match the parent's own
  # invocation.
  #
  # The needle is positional and mandatory (pass '' for none) so that everything after it
  # can be forwarded to the child as "$@" — properly quoted, rather than word-split out of
  # one string, which is how a flag value containing a space silently becomes two flags.
  run_control() {
    local n=$1 label=$2 d=$3 want=$4 needle=$5
    shift 5
    local out rc ok=1
    rc=0
    out=$(bash "$SELF" --campaign-dir "$d" "$@" 2>&1) || rc=$?
    if [[ "$rc" != "$want" ]]; then
      ok=0
    fi
    if [[ -n "$needle" ]]; then
      if ! printf '%s\n' "$out" | /usr/bin/grep -qF "$needle"; then
        ok=0
      fi
    fi
    if [[ "$ok" == 1 ]]; then
      printf 'CONTROL %s/%s %-28s PASS (rc=%s, expected %s)\n' \
        "$n" "$CONTROL_TOTAL" "$label" "$rc" "$want"
      controls_passed=$((controls_passed + 1))
    else
      printf 'CONTROL %s/%s %-28s FAIL (rc=%s, expected %s)\n' \
        "$n" "$CONTROL_TOTAL" "$label" "$rc" "$want"
      printf '%s\n' "$out" | sed 's/^/    child| /'
      controls_failed=$((controls_failed + 1))
    fi
    return 0
  }

  # CONTROL 1 — MUST_FIRE. A gate that exits 5 is a finding, and the runner must exit 5
  # AND name the gate. A runner that swallowed a red — or reported "some gate is red"
  # without saying which — would send the operator through all nine by hand, which is the
  # cost this script exists to remove. A green stub rides alongside so the red has to be
  # attributed, not just detected.
  #
  # The needle is the RED roll-call entry `gate_alpha.py(5)`, not the bare filename: the
  # filename appears in the per-gate roll whatever its state, so a bare-name needle proves
  # presence, not attribution — and a control that passes when the runner redds the WRONG
  # gate is not measuring attribution.
  make_stub_campaign "$TMPROOT/c1" gate_alpha.py:5 gate_zzz_ok.py:0
  run_control 1 "MUST_FIRE" "$TMPROOT/c1" 5 "gate_alpha.py(5)"

  # (controls 2-4 below; 5-6 exercise --allow-refused)

  # CONTROL 2 — MUST_PASS. A green gate must produce exit 0. Without this control, a
  # runner that redded or refused EVERYTHING — a broken python3 on PATH, a wrong cwd —
  # would look identical to a runner with findings, and the first real green run would be
  # indistinguishable from a broken instrument.
  make_stub_campaign "$TMPROOT/c2" gate_beta.py:0
  run_control 2 "MUST_PASS" "$TMPROOT/c2" 0 "STANDING GATES GREEN"

  # CONTROL 3 — REFUSAL IS NOT RED. A gate exiting 96 has declared CANNOT-MEASURE: an
  # input was absent. The runner must exit 95, not 5. This is the distinction the whole
  # four-state contract exists for; collapsing it is the #56/#149/#160 defect, and it is
  # the exact collapse that decides whether a bare checkout reads as "broken tree" or as
  # "two gates waiting on the estate environment".
  #
  # The needle is the REFUSED roll-call entry `gate_gamma.py(96)` — the state-labelled
  # form, for the same attribution reason as control 1: a bare filename would pass even
  # if the refusal were attributed to the wrong gate.
  make_stub_campaign "$TMPROOT/c3" gate_gamma.py:96 gate_zzz_ok.py:0
  run_control 3 "REFUSAL_IS_NOT_RED" "$TMPROOT/c3" 95 "gate_gamma.py(96)"

  # CONTROL 4 — VACUOUS DENOMINATOR. A build file with NO matching gate lines must exit
  # 96, never 0. This is the control that stops the runner from reporting all-clear over
  # an empty set — the false-green shape of #157, #233, #241 and #303. The stub carries an
  # INDENTED decoy invocation to prove the column-0 anchor is doing its job: a derivation
  # that matched it would find one gate and exit 0, which is exactly the wrong answer.
  #
  # The needle names the zero-denominator refusal specifically rather than accepting ANY
  # exit 96: a 96 from the wrong arm — a missing build file, a stale allowance, a grep
  # error — would prove nothing about the floor this control exists to pin.
  mkdir -p "$TMPROOT/c4"
  {
    printf '# a stub build with no standing gates at all\n'
    printf '  python3 gate_indented_decoy.py || {\n'
  } > "$TMPROOT/c4/build_h100_plane.sh"
  printf 'import sys\nsys.exit(0)\n' > "$TMPROOT/c4/gate_indented_decoy.py"
  run_control 4 "VACUOUS_DENOMINATOR" "$TMPROOT/c4" 96 "ZERO standing-gate invocations"

  # CONTROLS 5 and 6 are a MATCHED PAIR over the same synthetic tree, and they only mean
  # something together. One tree, two declarations, two verdicts: the ONLY thing that
  # differs is whether the refusing gate was named. Run either alone and you cannot tell an
  # allow-list that works from an allow-list that is ignored — a runner hardwired to exit 0
  # on any refusal passes control 6, and one hardwired to exit 5 passes control 5. The pair
  # is what pins the flag to the verdict.
  make_stub_campaign "$TMPROOT/c56" gate_delta.py:95 gate_zzz_ok.py:0

  # CONTROL 5 — UNDECLARED REFUSAL IS RED. gate_delta refuses; the allow-list names only
  # the OTHER gate, so the refusal is undeclared and must exit 5. This is the arm that
  # stops --allow-refused from degrading into a blanket amnesty: naming SOME gate must not
  # excuse EVERY gate. Note the allow-list here is deliberately a name that is in the
  # denominator but is not the one refusing — a non-empty list that must still red.
  #
  # The needle is the state-labelled `gate_delta.py(95)` form, not the bare name: the
  # verdict this control pins is that THIS gate's refusal went undeclared, and the bare
  # filename would appear in the per-gate roll even if the verdict named nothing.
  run_control 5 "UNDECLARED_IS_RED" "$TMPROOT/c56" 5 "gate_delta.py(95)" \
    --allow-refused gate_zzz_ok.py

  # CONTROL 6 — DECLARED REFUSAL IS GREEN. Same tree, same refusal, now named. Exit 0, and
  # the output must still SAY the gate abstained: a declared abstention that prints nothing
  # is the silent pass of #56, and an operator reading a bare "GREEN" would believe nine
  # gates measured when eight did.
  run_control 6 "DECLARED_IS_GREEN" "$TMPROOT/c56" 0 "DECLARED ABSTENTIONS" \
    --allow-refused gate_delta.py

  # CONTROL 7 — STALE ALLOWANCE. An allow-list naming a gate that is not in the derived
  # denominator must exit 96. Without this arm the allowance would go quietly inert on the
  # day a gate is renamed, and the run would keep passing on a declaration that can no
  # longer match anything — a promise kept only because it is no longer checked (#277).
  # The stub tree here is GREEN, so the 96 can come from nothing but the stale name.
  run_control 7 "STALE_ALLOWANCE" "$TMPROOT/c2" 96 "not in the derived" \
    --allow-refused gate_typo_never_existed.py

  # CONTROL 8 — UNEXPECTED_RC. A stub gate exits 42, a code the four-state contract does
  # not declare. The runner must exit 5 AND name the unexpected rc. This control exists
  # because the `*)` arm of the classification case — "an undeclared code is recorded as
  # RED" — is otherwise exercised by NOTHING: someone could change that arm to default
  # green and the self-test would still report every control passing.
  make_stub_campaign "$TMPROOT/c8" gate_theta.py:42 gate_zzz_ok.py:0
  run_control 8 "UNEXPECTED_RC" "$TMPROOT/c8" 5 "unexpected rc '42'"

  # CONTROL 9 — RED_BEATS_REFUSED. One gate exits 5 and another exits 95; the runner must
  # exit 5, not 95. The red-before-refused precedence — a finding about the tree outranks
  # every refusal, because a refusal reported above a red reads as an amnesty — is
  # otherwise uncontrolled: swapping the two verdict blocks would turn a live finding
  # into the amnesty the header explicitly tells CI to tolerate, and nothing would notice.
  make_stub_campaign "$TMPROOT/c9" gate_red_one.py:5 gate_refused_one.py:95
  run_control 9 "RED_BEATS_REFUSED" "$TMPROOT/c9" 5 "STANDING GATES RED"

  # CONTROL 10 — LOOSE_STRICT_GAP. The stub build carries one well-formed gate line AND
  # one line the LOOSE pattern matches but the STRICT pattern does not (a double space
  # after python3). The runner must exit 96 and print the unmatched line. This is the
  # cross-check's own control: without it, the loose pattern could be deleted outright or
  # degenerated into a copy of the strict one — the #203 self-comparison — and no control
  # would fire.
  mkdir -p "$TMPROOT/c10"
  {
    printf 'python3 gate_strict_ok.py || {\n'
    printf 'python3  gate_loose_only.py || {\n'
  } > "$TMPROOT/c10/build_h100_plane.sh"
  printf 'import sys\nsys.exit(0)\n' > "$TMPROOT/c10/gate_strict_ok.py"
  run_control 10 "LOOSE_STRICT_GAP" "$TMPROOT/c10" 96 "gate_loose_only.py"

  # CONTROL 11 — COMMENTED_MENTION_IS_NOT_A_GATE, and it is the MATCHED PARTNER of control
  # 10 in exactly the way 5 and 6 are matched: same loose shape, same file, the ONLY
  # difference is a leading `#`. Run 10 alone and a loose scan that had simply been deleted
  # would... fail 10, fine — but run 10 alone and a loose scan narrowed to nothing useful
  # by an over-broad filter still passes it. Run 11 alone and a loose scan deleted outright
  # passes. Together they pin the filter to the use/mention axis: the uncommented line must
  # sink the run, the commented one must not.
  #
  # This is not hypothetical. The real build_h100_plane.sh documents its own gate idiom in
  # a comment, with a placeholder filename that keeps the SHAPE, and the first version of
  # the cross-check read that sentence as a tenth gate and refused a healthy nine-gate tree
  # with 96. The needle checks BOTH halves of the fix: the run is green, and the count of
  # elided mentions is PRINTED rather than narrowed away in silence.
  mkdir -p "$TMPROOT/c11"
  {
    printf 'python3 gate_real_one.py || {\n'
    printf '#     python3 gate_documented_example.py || { rc=$?; ... ; }\n'
  } > "$TMPROOT/c11/build_h100_plane.sh"
  printf 'import sys\nsys.exit(0)\n' > "$TMPROOT/c11/gate_real_one.py"
  run_control 11 "COMMENTED_IS_NOT_A_GATE" "$TMPROOT/c11" 0 \
    "strict=1 loose=1 (1 comment mention(s) elided)"

  printf '\nSELF-TEST SUMMARY: %s/%s controls passed, %s failed\n' \
    "$controls_passed" "$CONTROL_TOTAL" "$controls_failed"
  # Written as an if, not `[[ ]] && exit 0`: set -e is on and a false && list is itself a
  # failing compound command, which would exit 1 from inside the summary — a third state
  # nobody declared.
  if [[ "$controls_failed" -eq 0 ]]; then
    exit 0
  else
    echo "A broken control is a finding about the instrument, not about the tree." >&2
    exit 5
  fi
fi

# ---------------------------------------------------------------------------
# Real run: resolve the campaign directory and cd there.
# ---------------------------------------------------------------------------
if [[ -z "$CAMPAIGN_DIR" ]]; then
  CAMPAIGN_DIR=$SELF_DIR
fi
if [[ ! -d "$CAMPAIGN_DIR" ]]; then
  echo "CANNOT-MEASURE (96): campaign dir does not exist: $CAMPAIGN_DIR" >&2
  exit 96
fi
# Guarded cd: an unguarded `cd` failing under set -e exits 1, which is in NO bucket of
# the declared 0/5/95/96 contract — and the -d test above passes on a directory that
# exists but is not searchable by this user. A directory we cannot enter is
# CANNOT-MEASURE, and it is named.
if ! cd "$CAMPAIGN_DIR"; then
  echo "CANNOT-MEASURE (96): cannot enter campaign dir: $CAMPAIGN_DIR" >&2
  echo "  It exists (-d passed) but this user cannot search it, so the gates inside it" >&2
  echo "  cannot be measured from here." >&2
  exit 96
fi

BUILD=build_h100_plane.sh
if [[ ! -f "$BUILD" ]]; then
  echo "CANNOT-MEASURE (96): $BUILD not found in $PWD — the gate denominator is derived" >&2
  echo "  from that file, so without it there is nothing to run and nothing to say." >&2
  exit 96
fi

# This runner's OWN missing dependency must be diagnosed as such, BEFORE the derivation:
# on a host without /usr/bin/grep the derivation silently yields nothing and the floor
# below would blame the BUILD FILE — a false diagnosis that sends the operator to
# re-anchor a derivation that is not broken. The pin itself stays: bare grep can be
# ugrep honouring .gitignore, which would silently shrink the denominator.
if [[ ! -x /usr/bin/grep ]]; then
  echo "CANNOT-MEASURE (96): this runner requires /usr/bin/grep, which is absent or not" >&2
  echo "  executable on this host. The defect is in the instrument's own environment, not" >&2
  echo "  in $BUILD — do not re-anchor the derivation over a missing grep." >&2
  exit 96
fi

# Derive the gate list with the EXACT anchored pattern the build uses at its GATE_COUNT=
# line, then extract the filenames. /usr/bin/grep, not bare grep: on the developer machine
# bare grep can be ugrep honouring .gitignore, which would silently shrink the denominator.
#
# The producer's exit status is CAPTURED, not hidden inside a process substitution — this
# is finding #158 in this campaign's own ledger: a nonzero-exiting producer silently
# truncates the denominator. A grep that errors MID-STREAM emits the matches found so far
# and exits 2, so the list is non-empty but SHORT, the zero-floor below passes, and the
# runner reports "GREEN — 8/8" over a 9-gate plane. rc 1 stays benign: it means no match,
# and the zero-denominator floor already handles that correctly. rc >= 2 is a real grep
# ERROR and is refused on, naming the rc.
grep_rc=0
GREP_OUT=$(/usr/bin/grep '^python3 gate_[a-z_0-9]*\.py || {' "$BUILD") || grep_rc=$?
if [[ "$grep_rc" -ge 2 ]]; then
  echo "CANNOT-MEASURE (96): the derivation grep on $BUILD FAILED with rc=$grep_rc." >&2
  echo "  A grep that errors mid-stream can emit a non-empty but SHORT list, which the" >&2
  echo "  zero-floor below cannot catch — so a grep error is refused on, never read as a" >&2
  echo "  denominator." >&2
  exit 96
fi

GATES=()
while IFS= read -r _g; do
  # An `if`, not `[[ -n "$_g" ]] && GATES+=(...)`: under set -e a false && list is itself a
  # failing compound command, so the LAST loop iteration reading a blank line would exit
  # the whole script from inside the read loop, silently, with a partial denominator.
  if [[ -n "$_g" ]]; then
    GATES+=("$_g")
  fi
done < <(printf '%s\n' "$GREP_OUT" | sed -n 's/^python3 \(gate_[a-z_0-9]*\.py\) || {.*/\1/p')

# THE NON-VACUOUS FLOOR. We do not compare this count against the build's GATE_COUNT —
# both come from the same grep on the same file, so that comparison is true by
# construction (#203). What is not true by construction is that the derivation found
# anything. An empty denominator is CANNOT-MEASURE, never a pass: a runner that scans a
# file, finds no gates, and prints "0 red, all clear" is the exact false-green this
# campaign keeps filing (#157, #233, #241, #303).
if [[ "${#GATES[@]}" -eq 0 ]]; then
  echo "CANNOT-MEASURE (96): the anchored derivation found ZERO standing-gate invocations" >&2
  echo "  in $BUILD. Either the build's gate plane changed shape (and this runner's pattern" >&2
  echo "  must change with it — one derivation rule, two consumers) or the file is not the" >&2
  echo "  build this runner certifies. Reporting GREEN over zero gates is not an option." >&2
  exit 96
fi

# LOOSE-vs-STRICT CROSS-CHECK. The strict pattern above is one rigid shape:
# `python3 gate_new.py --flag || {`, `VAR=x python3 gate_x.py || {`, a double space, or
# an uppercase letter in the filename all match NOTHING, and that gate is never run while
# the runner prints a confident N/N. So alongside the strict anchored pattern we count
# lines matching a deliberately LOOSER shape for the same idiom — any line mentioning
# python3 and a gate_*.py name, in either order. If the loose set is LARGER, something
# that looks like a gate invocation is invisible to the derivation, and that is
# CANNOT-MEASURE: print the lines the loose pattern saw and the strict pattern did not,
# and refuse.
#
# This check is NOT the vacuous self-comparison of #203: the two patterns are DIFFERENT —
# it is not the same grep compared with itself. The strict pattern must stay identical to
# the one the build uses at its GATE_COUNT= line — one derivation rule, two consumers —
# and this cross-check is what stops that shared rule from going quietly narrow.
loose_rc=0
LOOSE_OUT=$(/usr/bin/grep -E 'python3.*gate_[A-Za-z_0-9]*\.py|gate_[A-Za-z_0-9]*\.py.*python3' "$BUILD") || loose_rc=$?
if [[ "$loose_rc" -ge 2 ]]; then
  echo "CANNOT-MEASURE (96): the loose cross-check grep on $BUILD FAILED with" >&2
  echo "  rc=$loose_rc. Same rule as the derivation grep: a producer error is refused on," >&2
  echo "  never read as a count." >&2
  exit 96
fi
strict_count=${#GATES[@]}

# USE vs MENTION, and this cost a false 96 on the real tree before it was written. The
# build script DOCUMENTS its own gate idiom in a comment — a worked example of the
# invocation shape carrying a placeholder filename. A placeholder that keeps the SHAPE is
# still matched by a shape-blind pattern, so the loose scan read that sentence as a tenth
# gate and refused a nine-gate tree that was measuring correctly.
#
# The narrowing is to skip lines whose first non-blank character is `#`. That is
# principled rather than an exception: this scan is looking for INVOCATIONS, and a
# commented line cannot execute, so a comment is a mention and not a use. It is
# deliberately NOT an allowlist naming the offending line — an allowlist would go stale
# silently and would have to grow once per sentence.
#
# But a silent narrowing is exactly how a denominator shrinks, so the elided count is
# PRINTED on every run, green or not. Note the residual hole honestly: a gate line that
# someone COMMENTS OUT is invisible to both patterns, so this check cannot see it. What
# does see it is the roll-call's own N, which falls by one — the same signal the build's
# GATE_COUNT= line gives, because it is the same rule.
loose_lines=""
loose_count=0
loose_comment=0
while IFS= read -r _l; do
  if [[ -z "$_l" ]]; then
    continue
  fi
  _t=${_l#"${_l%%[![:blank:]]*}"}
  case "$_t" in
    '#'*)
      loose_comment=$((loose_comment + 1))
      continue
      ;;
  esac
  loose_count=$((loose_count + 1))
  loose_lines="$loose_lines$_l
"
done < <(printf '%s\n' "$LOOSE_OUT")

printf 'derivation cross-check: strict=%s loose=%s (%s comment mention(s) elided)\n' \
  "$strict_count" "$loose_count" "$loose_comment"

if [[ "$loose_count" -gt "$strict_count" ]]; then
  echo "CANNOT-MEASURE (96): the strict anchored derivation found $strict_count gate" >&2
  echo "  line(s) in $BUILD, but a deliberately looser scan sees $loose_count. Line(s) the" >&2
  echo "  loose pattern saw and the strict pattern did NOT:" >&2
  while IFS= read -r _l; do
    if [[ -n "$_l" ]]; then
      if ! printf '%s\n' "$_l" | /usr/bin/grep -q '^python3 gate_[a-z_0-9]*\.py || {'; then
        printf '    %s\n' "$_l" >&2
      fi
    fi
  done < <(printf '%s\n' "$loose_lines")
  echo "  A gate the strict shape cannot see is a gate that never runs while the runner" >&2
  echo "  prints a confident N/N. Either re-anchor the derivation (and the build's" >&2
  echo "  GATE_COUNT= line with it — one derivation rule, two consumers) or fix the" >&2
  echo "  offending line to the declared shape." >&2
  exit 96
fi

# Every derived filename must exist on disk AND be non-empty. A gate that cannot be run
# has certified nothing, and skipping it would shrink the denominator back into a
# false-green. The non-empty half is `-s`, not merely `-f`: `python3 <empty file>` exits
# 0, so a zero-byte gate — a truncated file from a bad merge — would be recorded GREEN
# having measured nothing.
_missing=""
_empty=""
for _g in "${GATES[@]}"; do
  if [[ ! -f "$_g" ]]; then
    _missing="$_missing $_g"
  elif [[ ! -s "$_g" ]]; then
    _empty="$_empty $_g"
  fi
done
if [[ -n "$_missing" ]]; then
  echo "CANNOT-MEASURE (96): derived standing gate(s) absent on disk:$_missing" >&2
  echo "  The build declares them and the tree does not have them. That is a state to" >&2
  echo "  refuse on, not a list to silently shorten." >&2
  exit 96
fi
if [[ -n "$_empty" ]]; then
  echo "CANNOT-MEASURE (96): derived standing gate(s) are ZERO-BYTE on disk:$_empty" >&2
  echo "  python3 exits 0 on an empty file, so running them would record GREEN over" >&2
  echo "  nothing. A truncated gate is a state to refuse on, not a pass to collect." >&2
  exit 96
fi

N=${#GATES[@]}

# INTERPRETER PROVENANCE (finding #83 in this campaign): a CLEAR verdict with no
# interpreter attribution is unreproducible — the gates are run by whatever `python3`
# PATH offers, so record which interpreter that is and its version. Deliberately NOT
# validated or pinned: the interpreter is a property of the machine, and a gate that
# reddens on it reports the developer's environment as a repo defect. Just record it, so
# the verdict is attributable.
PY3_PATH=$(command -v python3 || true)
PY3_VERSION=""
if [[ -n "$PY3_PATH" ]]; then
  PY3_VERSION=$("$PY3_PATH" --version 2>&1 || true)
fi
echo "=== standing gates: running $N gate(s) derived from $BUILD ==="
echo "    python3: ${PY3_PATH:-<not found on PATH>} — ${PY3_VERSION:-<version unknown>}"

# The allowed-to-refuse set, as a space-padded string so membership is a `case` glob —
# bash 3.2 has no associative arrays. Commas are the caller's separator; spaces are ours.
ALLOW_SET=" $(printf '%s' "$ALLOW_REFUSED" | tr ',' ' ') "

# EVERY NAME IN THE ALLOW-LIST MUST BE IN THE DERIVED DENOMINATOR. A typo, or a gate that
# was renamed since the caller was written, produces an allowance that can never match:
# the declaration goes quietly inert and the gate it was meant to cover starts sinking the
# run — or worse, the reverse, if the name it drifted onto is a real gate. That is #277's
# STALE_DECLARATION shape, and the honest response is to refuse rather than to guess which
# reading the caller meant. 96, not 5: the defect is in the instruction this runner was
# given, not in the tree it was pointed at.
if [[ -n "$ALLOW_REFUSED" ]]; then
  _unknown=""
  for _a in $(printf '%s' "$ALLOW_REFUSED" | tr ',' ' '); do
    _hit=0
    for _g in "${GATES[@]}"; do
      if [[ "$_g" == "$_a" ]]; then
        _hit=1
      fi
    done
    if [[ "$_hit" -eq 0 ]]; then
      _unknown="$_unknown $_a"
    fi
  done
  if [[ -n "$_unknown" ]]; then
    echo "CANNOT-MEASURE (96): --allow-refused names gate(s) that are not in the derived" >&2
    echo "  denominator:$_unknown" >&2
    echo "  An allowance for a gate the build does not invoke is inert, and an inert" >&2
    echo "  allowance is indistinguishable from one that is working. Fix the name or drop" >&2
    echo "  it; do not leave a declaration pointing at nothing." >&2
    exit 96
  fi
fi

gate_red=0
gate_refused=0
gate_undeclared=0
gate_red_names=""
gate_refused_names=""
gate_declared_names=""
gate_undeclared_names=""

for _g in "${GATES[@]}"; do
  # `|| rc=$?`, not a bare `cmd; rc=$?`: under set -e a bare failing command kills the
  # script before the rc is ever read, which would turn the FIRST red gate into the only
  # gate that ran — the early-exit defect #298 removed from the build, one layer down.
  rc=0
  out=$(python3 "$_g" 2>&1) || rc=$?

  # The same four-state contract note_gate applies in the build.
  case $rc in
    0)
      state=green
      ;;
    5)
      state=red
      ;;
    95|96)
      state=refused
      ;;
    *)
      # The note_gate `*)` arm: an undeclared code is recorded as RED and said out loud,
      # because a gate outside its own contract is untrustworthy, not unmeasured.
      state=red
      echo "  run_standing_gates: unexpected rc '$rc' from $_g — recording it as RED" >&2
      ;;
  esac

  # A one-line digest of the gate's own output, so a green line still says something.
  #
  # NOT a bare `tail -1`. Several of these gates END by printing their exit-code legend --
  # gate_ckpt_naming_agreement.py's last line is literally "exit 5  DISAGREEMENT — the writer
  # and the adjudicator disagree about naming". Quoting that beside a `green` marker reads as
  # a red verdict on a gate that passed, which is a false alarm manufactured by the reporter
  # rather than by the tree. So skip legend lines: a line of the form "exit <n> ..." is a
  # gate DOCUMENTING a code, never a gate REPORTING one. Blank lines are skipped too.
  last=$(printf '%s\n' "$out" \
           | /usr/bin/grep -vE '^[[:space:]]*exit[[:space:]]+[0-9]+([[:space:]]|$)' \
           | /usr/bin/grep -vE '^[[:space:]]*$' \
           | tail -1 | cut -c1-120 || true)
  #                                    ^^^^^^^ pipefail is on and `grep -v` exits 1 when it
  # emits NOTHING. A gate whose entire output is legend lines would therefore fail the
  # command substitution and, under `set -e`, kill this runner from inside a cosmetic
  # formatting step -- silently, and only on the one gate shaped that way.
  printf '  %-8s  %-36s  rc=%-3s  %s\n' "$state" "$_g" "$rc" "$last"

  case $state in
    red)
      gate_red=$((gate_red + 1))
      gate_red_names="$gate_red_names $_g($rc)"
      # A red the operator cannot read is useless: print the gate's full captured output.
      if [[ -n "$out" ]]; then
        echo "  --- full output from $_g ---"
        printf '%s\n' "$out" | sed 's/^/    /'
        echo "  --- end $_g ---"
      fi
      ;;
    refused)
      gate_refused=$((gate_refused + 1))
      gate_refused_names="$gate_refused_names $_g($rc)"
      # Split the refusals into DECLARED and UNDECLARED. The distinction is the whole
      # point of the flag: an abstention someone wrote down in advance is a known cost of
      # this environment, and an abstention that appeared on its own is news.
      case "$ALLOW_SET" in
        *" $_g "*)
          gate_declared_names="$gate_declared_names $_g($rc)"
          ;;
        *)
          gate_undeclared=$((gate_undeclared + 1))
          gate_undeclared_names="$gate_undeclared_names $_g($rc)"
          ;;
      esac
      ;;
  esac
done

# Every gate ran, so the green count is a subtraction over a COMPLETE denominator — the
# #302 precondition, satisfied here by construction rather than by a gates_complete flag.
green=$((N - gate_red - gate_refused))

# Roll-call in the same shape the build's roll_call_gates prints.
if [[ $((gate_red + gate_refused)) -gt 0 ]]; then
  printf '\n--- standing gates: %s green, %s red, %s refused (of %s) ---\n' \
    "$green" "$gate_red" "$gate_refused" "$N" >&2
fi
if [[ -n "$gate_red_names" ]]; then
  echo "  RED:     $gate_red_names" >&2
fi
if [[ -n "$gate_refused_names" ]]; then
  echo "  REFUSED: $gate_refused_names" >&2
fi

# Verdict, most severe first — the build's own precedence: RED is a finding about the
# tree and is true whether or not anything else could be measured, so it outranks every
# refusal. A refusal reported above a red would read as an amnesty.
if [[ "$gate_red" -gt 0 ]]; then
  printf '\nSTANDING GATES RED — %s of %s gate(s) reported a finding.\n' \
    "$gate_red" "$N" >&2
  exit 5
fi

# An UNDECLARED refusal, when the caller declared a set, is RED. The caller stated which
# gates may abstain in this environment; a gate outside that set going quiet means the
# environment changed, or the gate broke, or the declaration is stale — and none of those
# three is something to pass over. This arm is what makes the flag a contract rather than
# a threshold: without it, `--allow-refused` would be an amnesty for whatever happened to
# refuse today, which is the #199/#233 "a count of 2 is satisfied by the wrong 2" failure
# wearing a different hat.
#
# Guarded on ALLOW_GIVEN, and control 3 is why. With NO declaration in play there is no
# such thing as an undeclared refusal — every refusal is simply unmeasured, and the honest
# verdict is 95. An earlier revision of this arm was unguarded, which silently promoted the
# default bare-checkout outcome from 95 to 5 and made the header's own worked example
# false. That is the #56/#149/#160 collapse, re-introduced by the very flag added to avoid
# it; the control caught it before the file was installed.
if [[ "$ALLOW_GIVEN" -eq 1 && "$gate_undeclared" -gt 0 ]]; then
  printf '\nSTANDING GATES RED — %s undeclared refusal(s):%s\n' \
    "$gate_undeclared" "$gate_undeclared_names" >&2
  echo "  These gates could not measure, and --allow-refused did not say they might." >&2
  echo "  An abstention nobody declared is a finding: either the environment lost an input" >&2
  echo "  it used to have, or the gate stopped working. Investigate before widening the" >&2
  echo "  allow-list — widening it is how a real regression becomes permanent." >&2
  exit 5
fi

# Every refusal was declared. Say so LOUDLY and exit 0. The volume is the point: #56's
# remedy is that an abstention must be a state someone can see, and a silent 0 here would
# be exactly the collapse this whole contract exists to prevent. The run passed BECAUSE
# the abstentions were declared, not because nothing abstained.
if [[ "$ALLOW_GIVEN" -eq 1 && "$gate_refused" -gt 0 ]]; then
  printf '\nSTANDING GATES GREEN WITH DECLARED ABSTENTIONS — %s of %s measured green,\n' \
    "$green" "$N"
  printf '  %s declared abstention(s):%s\n' "$gate_refused" "$gate_declared_names"
  echo "  Those gates certified NOTHING on this run. They were permitted to abstain here"
  echo "  because their inputs are absent by design in this environment — not because"
  echo "  their subject matter is known good. Something else must measure them."
  exit 0
fi

if [[ "$gate_refused" -gt 0 ]]; then
  printf '\nSTANDING GATES UNMEASURED (95) — %s of %s gate(s) could not measure:%s\n' \
    "$gate_refused" "$N" "$gate_refused_names" >&2
  # The remediation below names no environment and no knob: "source the estate
  # environment" is a dead reference for anyone outside the deployment that has one —
  # the remediation as written could not be followed. Where to look is the gates' own
  # refusal messages; how CI declares the expectation is this runner's flag.
  echo "  A gate that could not measure has certified nothing; that is not the same as a gate" >&2
  echo "  that ran and found nothing, and this runner declines to merge the two. The refusing" >&2
  echo "  gates need DEPLOYMENT-SUPPLIED configuration whose knob names the gates themselves" >&2
  echo "  print in their refusal messages; a public checkout is not expected to have it, so" >&2
  echo "  on a bare checkout this is the EXPECTED outcome — see the header. To make this" >&2
  echo "  usable as a CI signal, the job declares that expectation with --allow-refused," >&2
  echo "  NAMING the gates permitted to abstain; a named set still reds when a gate outside" >&2
  echo "  it goes quiet." >&2
  exit 95
fi

# The all-clear arm prints the MEASURED numerator, not the derived one. Printing `N/N`
# would assert the numerator by construction; the measured green count and the derived
# denominator AGREEING is a real internal-consistency claim, so assert it before printing
# and refuse if they disagree — printing the same number twice is not a check.
if [[ "$green" -ne "$N" ]]; then
  echo "CANNOT-MEASURE (96): internal inconsistency — measured green ($green) does not" >&2
  echo "  equal the derived denominator ($N) with zero red and zero refused. The roll-call" >&2
  echo "  arithmetic is broken, and a verdict printed over it would certify nothing." >&2
  exit 96
fi
printf '\nSTANDING GATES GREEN — %s/%s\n' "$green" "$N"
exit 0
