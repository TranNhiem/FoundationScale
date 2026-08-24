#!/usr/bin/env bash
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

# --- shared source-counting helpers -----------------------------------------
# These live HERE, above every caller, and must stay here. bash resolves a
# function name at CALL time from what has already been sourced, so a
# definition placed after its first use is not a late definition — it is no
# definition at all: the call runs with an empty pipeline and the control
# reads a vacuous zero rather than erroring. fix26b first landed this block
# next to the srun accounting at the bottom, and the OPT_LR_KEY ordering leg
# and both FS_ROOT legs (first callers, ~line 80/95) went red against
# launchers that were correct. Doctrine 1, in the harness: absent input is
# not evidence of absent matches.
#
# fix26 (item B) — the counting rule. The old `^[[:space:]]*` anchor saw a
# call only at the start of its line, so the two `if ! run_in_container`
# sites were invisible: this control read LoRA=1 against its own correct
# denominator of 3 — red forever, while the launchers were RIGHT. The
# rule below counts a symbol wherever bash grammar puts a command:
#   line start, or immediately after  ;  &  |  (  or a backquote
#   (so the `(` arm also covers $( ... ) and plain subshells),
# optionally bridged by `if `/`then ` and a `! ` negation. The trailer
# requires whitespace-or-EOL right after the symbol, so the
# run_in_container() DEFINITION line (name followed by `()`) cannot
# count. grep -c counts LINES; both launchers keep one call per header
# line, and the MUST_FIRE below proves the tally still moves.
# DELIBERATELY EXCLUDED, by construction: `source` lines and every prose
# mention whose nearest preceding separator is a plain space — e.g. the
# full-FT comment "…it is now the run_in_container pid" (its `;` is
# followed by `it`, not by the symbol). One bare-substring count would
# have swept all of that in and turned a precise control into a loose
# one that passes for the wrong reason.
#
# fix26b — the stated blind spot fired. The paragraph above once claimed
# "nothing like that exists in either launcher today"; full-FT :508
# carries `(srun on` inside a CORRECT comment, the operator class took
# the `(`, and zero-srun read full-FT=1 — red forever for a reason no
# legitimate launcher edit may fix (the prose stays; the instrument
# adapts to real source, not the reverse). The repair is to count over
# the text BASH counts: comment tails removed BEFORE matching, once, in
# the helper every leg below shares. What a comment is, for this
# counter: a `#` at a word start (line start, or after blank/operator)
# OUTSIDE quotes. Quote tracking matters in the fail-OPEN direction the
# measured sed candidate ('s/(^|[[:space:];&|(])#.*$/\1/') could not
# see: under the sed,  echo "a # b"; srun ...  would erase the line
# from the quoted `#` on and hide a REAL srun behind a green zero. With
# quote tracking, that srun survives the strip and the control goes red
# as it must. ${#arr[@]}, $#, 2#$x and "#"-inside-strings survive the
# strip untouched (`#` mid-word is never a comment); `#!` at line 1
# disappears, correctly (to bash it IS a comment). Known-safe residuals:
# operator-glued symbols inside QUOTES or heredoc prose still count —
# deliberately, since ignoring quoted text would hand a real site the
# same hiding place; that failure shows as a wrong-number red, never a
# green — and heredoc `#` data may be cut though bash keeps it (data
# was never a call site: harmless). The strip preserves line COUNT 1:1
# (prints every line, truncated or not), so `grep -n` line numbers and
# "first matching line" logic elsewhere in this file stay truthful.
# FS_DECOMMENT_AWK holds the ONE definition so no future leg can
# re-grow the raw-view semantics by accident (house rule: the rule is
# shared, not duplicated). It deliberately contains no single-quote
# byte (quote constants via sprintf) so it can live in one
# single-quoted assignment.
FS_DECOMMENT_AWK='
function decomment(s,   out,i,n,c,q,sq,dq,bs) {
  sq=sprintf("%c",39); dq=sprintf("%c",34); bs=sprintf("%c",92)
  out=""; q=""; n=length(s)
  for (i=1; i<=n; i++) {
    c=substr(s,i,1)
    if (q!="") {                       # quoted region: every char, incl. #, is literal text
      if (q==dq && c==bs && i<n) { out=out c substr(s,++i,1); continue }
      if (c==q) q=""
      out=out c; continue
    }
    if (c==bs && i<n) { out=out c substr(s,++i,1); continue }  # escaped char is literal
    if (c==sq || c==dq) { q=c; out=out c; continue }
    if (c=="#" && (out=="" || substr(out,length(out),1) ~ /[ \t;&|()<>]/)) break
    out=out c
  }
  return out
}'
strip_shell_comments() { # stdin=shell source -> stdout=same line count, comment tails gone
  awk "$FS_DECOMMENT_AWK"'{ print decomment($0) }'
}
pos_pat() { # $1 = symbol -> ERE matching that symbol in shell command position
  printf '(^|[;&|(`])[[:space:]]*((if|then)[[:space:]]+)?(![[:space:]]+)?%s([[:space:]]|$)' "$1"
}
pos_count() { # $1 = symbol, $2 = file -> command-position sites in the comment-stripped view
  # An unreadable subject is BLOCK, not zero (doctrines 1/4): without this
  # guard, a vanished launcher would read srun=0 — a vacuous GREEN zero.
  [ -r "$2" ] || { echo "pos_count: cannot read $2 — refusing a vacuous count" >&2; printf 'UNREADABLE\n'; return 1; }
  strip_shell_comments < "$2" | grep -cE "$(pos_pat "$1")" || true
}


# --- Extract the EXTRA_OVERRIDES block from the full-FT launcher verbatim -----
# sed range, not a paraphrase: a control that tests a copy of the logic proves
# nothing about the logic that ships.
BLOCK=$(sed -n '/^declare -a EXTRA_EFFECTIVE=()/,/^fi$/p' "$FULL")
# Absent logic is a FAILING control, not a harness crash: "the block isn't there"
# is precisely the pre-fix defect these four checks exist to detect. Aborting
# here would have let the reverted baseline exit non-zero for the wrong reason
# and proven nothing about whether the checks can fire.
HAVE_BLOCK=1; [ -n "$BLOCK" ] || { HAVE_BLOCK=0; echo "  (no EXTRA_OVERRIDES block in $FULL — the 4 checks below cannot pass)"; }

run_block(){ # $1 = EXTRA_OVERRIDES value ("" = unset)
  [ "$HAVE_BLOCK" = 1 ] || { echo "NO_BLOCK"; return 127; }
  env -u EXTRA_OVERRIDES bash -c '
    set -euo pipefail
    die(){ echo "DIE: $*" >&2; exit 9; }
    [ -n "${1:-}" ] && export EXTRA_OVERRIDES="$1"
    '"$BLOCK"'
    echo "RECORDED=${#EXTRA_EFFECTIVE[@]}"
    printf "ARG:%s\n" ${EXTRA_EFFECTIVE[@]+"${EXTRA_EFFECTIVE[@]}"}
  ' _ "$1" 2>&1
}

echo "== full-FT EXTRA_OVERRIDES =="
# MUST_FIRE: shadowing a first-class knob must be refused, by name.
out=$(run_block "model.seq_length=8192"); rc=$?
if [ $rc -eq 9 ] && printf '%s' "$out" | grep -q "already records"; then
  ok "MUST_FIRE shadowed knob refused (model.seq_length)"
else no "MUST_FIRE shadowed knob NOT refused (rc=$rc): $out"; fi

# MUST_FIRE: a non-KEY=VALUE entry is unrecordable, so it must not silently ride along.
out=$(run_block "just_garbage"); rc=$?
if [ $rc -eq 9 ] && printf '%s' "$out" | grep -q "not KEY=VALUE"; then
  ok "MUST_FIRE malformed entry refused"
else no "MUST_FIRE malformed entry NOT refused (rc=$rc): $out"; fi

# MUST_PASS: a legitimate, non-colliding override is recorded, not merely tolerated.
out=$(run_block "model.moe_router_topk=2 optimizer.weight_decay=0.05"); rc=$?
if [ $rc -eq 0 ] && printf '%s' "$out" | grep -q "RECORDED=4" \
   && printf '%s' "$out" | grep -q "ARG:model.moe_router_topk=2" \
   && printf '%s' "$out" | grep -q "ARG:optimizer.weight_decay=0.05"; then
  ok "MUST_PASS two clean overrides recorded as 2x(--effective k=v)"
else no "MUST_PASS clean overrides not recorded (rc=$rc): $out"; fi

# MUST_PASS: the empty case must not trip set -u on the array expansion.
out=$(run_block ""); rc=$?
if [ $rc -eq 0 ] && printf '%s' "$out" | grep -q "RECORDED=0"; then
  ok "MUST_PASS unset EXTRA_OVERRIDES survives set -u (empty array expansion)"
else no "MUST_PASS unset case broke (rc=$rc): $out"; fi

echo "== LoRA arm identity =="
# The defect this block was born against: RUN_TAG spelled "L1" literally, so
# the EXPERT_TARGETS=0 and =1 arms shared OUTPUT_DIR and a resume would
# silently cross them. fix28b changed the world under this control —
# CORRECTLY, so the launcher is not what gets repaired: the launcher now
# MEASURES MoE-ness from the base config.json, and on a measured-DENSE base
# (this estate's E4B) the two arms are literally the same run — zero expert
# modules exist for the expert target strings to bind — so an expert-free run
# wearing the "L1" label would be the one-run-two-labels lie in a new
# spelling. The old control's two conjuncts therefore pin the WRONG WORLD
# today (the measured 20 passed / 2 failed: both reds are here) and the
# strengthened contract below is two-state. Each state is simulated with
# $MOE/$ENABLE_MOE_BLOCK/$NUM_EXPERTS set EXPLICITLY — the failing control
# this replaces exercised the dense branch only because $MOE happened to be
# unset in the harness environment, and an unset variable that takes the
# branch you wanted is doctrine 1's vacuous pass wearing a branch:
#   simulated MoE base:   the arms MUST stay distinct and the ET=1 tag MUST
#                         keep the historical '_L1_' spelling. The pre-fix28b
#                         concern, preserved verbatim.
#   simulated dense base: the arms MUST collapse to the SAME 'base4' tag AND
#   (= our measured base  the collapse MUST be LOUD. The WARN is the only
#    per the task)        thing making an honest run of the collapse; a
#                         SILENT collapse is the defect wearing new clothes,
#                         so the WARN here is contract, not decoration. The
#                         loudness needle is MINED from the launcher's own
#                         arm block below — never hand-paraphrased in this
#                         file — so a WARN that drifts or disappears reads
#                         red, not green.
# Mechanism, preserved from the control this replaces: the launcher's REAL
# EXPERT_TARGETS block is sed-extracted (/^EXPERT_TARGETS=/,/^fi$/) and
# evaled — never paraphrased — inside a throwaway subshell, and RUN_TAG's
# right-hand side is then evaled under the same hand-set probe environment as
# before (r32 a64 ep1 gbs64 seq4096). The subshell's stdout carries the
# block's own output (the dense-collapse WARN, when it fires) followed by one
# TAG: line; nothing leaks into harness scope. The helpers sit HERE, above
# their first call site: this file was broken exactly once already by a
# helper defined ~300 lines below its first caller — bash resolves a function
# name at CALL time, so a late definition is no definition at all.
lora_arm_block() { # $1=launcher file -> stdout: the real arm-switching block, verbatim
  sed -n '/^EXPERT_TARGETS=/,/^fi$/p' "$1"
}
run_lora_arm() { # $1=launcher file $2=EXPERT_TARGETS $3=MOE $4=ENABLE_MOE_BLOCK $5=NUM_EXPERTS
                 # stdout: the evaled block's own output (WARN on a dense ET=1 collapse) + one TAG: line
  (
    EXPERT_TARGETS=$2 MOE=$3 ENABLE_MOE_BLOCK=$4 NUM_EXPERTS=$5
    eval "$(lora_arm_block "$1" | sed "s/^EXPERT_TARGETS=.*/EXPERT_TARGETS=$2/")"
    LORA_RANK=32; LORA_ALPHA=64; EP=1; GLOBAL_BATCH_SIZE=64; SEQ_LENGTH=4096
    eval "echo TAG:$(grep -m1 '^RUN_TAG=' "$1" | cut -d= -f2-)"
  )
}
arm_tag_of() { # $1=run_lora_arm output -> the TAG: payload, or empty (a failed eval reads as no tag, never as a pass)
  printf '%s\n' "$1" | sed -n 's/^TAG://p' | tail -1
}
mine_dense_warn() { # $1=launcher file -> the literal prefix of the block's first WARN echo
                    # (up to its first runtime expansion or quote), or empty.
                    # Empty is EVIDENCE OF SILENCE: leg D2 below is red on it.
  lora_arm_block "$1" | grep -m1 'echo .*WARN:' | sed "s/^.*\(WARN:\)/\1/; s/\\\$.*\$//; s/['\"].*\$//"
}
dense_arm_warns() { # $1=launcher file $2=WARN needle; true iff the simulated dense ET=1 arm emits it
  [ -n "$2" ] || return 1   # an empty needle would grep-match everything — a vacuous pass; refuse it
  run_lora_arm "$1" 1 0 False null | grep -qF "$2"
}

# -- leg M (simulated MoE base): the author's original concern, verbatim -----
moe_out0=$(run_lora_arm "$LORA" 0 1 true 128)
moe_out1=$(run_lora_arm "$LORA" 1 1 true 128)
TAG_M0=$(arm_tag_of "$moe_out0"); TAG_M1=$(arm_tag_of "$moe_out1")
if [ -n "$TAG_M0" ] && [ -n "$TAG_M1" ] && [ "$TAG_M0" != "$TAG_M1" ]; then
  ok "MoE-base arms produce distinct RUN_TAG: '$TAG_M0' vs '$TAG_M1'"
else no "MoE-base arms COLLIDE on RUN_TAG='$TAG_M0' — same OUTPUT_DIR, resume would cross arms"; fi
if [ -n "$TAG_M1" ] && printf '%s' "$TAG_M1" | grep -q '_L1_'; then
  ok "MoE-base L1 arm keeps its historical tag spelling"
else no "MoE-base L1 arm tag changed spelling ('$TAG_M1') — breaks continuity with existing MoE-base dirs"; fi

# -- leg D (simulated dense base = this estate's measured base): collapse, loudly
FS_LORA_WARN_NEEDLE=$(mine_dense_warn "$LORA")
den_out0=$(run_lora_arm "$LORA" 0 0 False null)
den_out1=$(run_lora_arm "$LORA" 1 0 False null)
TAG_D0=$(arm_tag_of "$den_out0"); TAG_D1=$(arm_tag_of "$den_out1")
if [ -n "$TAG_D0" ] && [ "$TAG_D0" = "$TAG_D1" ] \
   && printf '%s' "$TAG_D1" | grep -q '_base4_' && ! printf '%s' "$TAG_D1" | grep -q '_L1_'; then
  ok "dense-base arms collapse to one 'base4' tag ('$TAG_D0') — same run must wear one label"
else no "dense-base arms did not collapse to base4 ('$TAG_D0' vs '$TAG_D1') — an expert-free run would wear the L1 name"; fi
if dense_arm_warns "$LORA" "$FS_LORA_WARN_NEEDLE"; then
  ok "dense-base collapse is LOUD (needle mined from the launcher's own block: '${FS_LORA_WARN_NEEDLE}…')"
else no "dense-base collapse is SILENT or its WARN drifted (mined needle: '${FS_LORA_WARN_NEEDLE:-<none: no echo … WARN: line in the arm block>}') — a silent L1->base4 relabel is the one-run-two-labels defect"; fi

# MUST_FIRE for the loudness leg (doctrine 3): a launcher whose dense branch
# collapses SILENTLY must turn leg D red. Construct one on a temp copy by
# REPLACING (not deleting) the block's first WARN echo with a `:` no-op —
# deletion would leave an empty `if … then fi`, which is a syntax error, and
# a firing input that merely fails to parse would certify the wrong thing:
# the defect being simulated is a silent-but-RUNNABLE relabel, so the copy
# must stay parseable while saying nothing. Then prove the firing input
# exists (the needle is GONE from the copy) and demand the same predicate
# the live leg runs report the copy as NOT loud. A construction that cannot
# be proven is reported UNREACHABLE as a failed control, never skipped —
# an unexercised detector proves nothing.
if [ -n "$FS_LORA_WARN_NEEDLE" ]; then
  arm_mf=$(mktemp "${TMPDIR:-/tmp}/fs-lora-arm-warn.XXXXXX") \
    && awk 'BEGIN{d=0} /^EXPERT_TARGETS=/{b=1} b && !d && /echo/ && /WARN:/ {d=1; print "    : # (harness MUST_FIRE construction: dense-collapse WARN silenced on this copy)"; next} {print}' \
         "$LORA" > "$arm_mf"
  arm_mfs=$?
  if [ "$arm_mfs" -eq 0 ] && ! grep -qF "$FS_LORA_WARN_NEEDLE" "$arm_mf" \
     && ! dense_arm_warns "$arm_mf" "$FS_LORA_WARN_NEEDLE"; then
    ok "MUST_FIRE dense-collapse-gone-silent: WARN silenced on a parseable copy -> the loudness leg goes red (needle provably absent from the firing input)"
  else no "MUST_FIRE UNREACHABLE: silently-collapsing launcher copy not constructible (build rc=$arm_mfs) — the dense loudness leg above is unproven"; fi
  [ -n "${arm_mf:-}" ] && rm -f "$arm_mf" || true
else
  no "MUST_FIRE UNREACHABLE: the arm block contains no echo … WARN: line to silence — leg D above is already red, and its firing input cannot be constructed either"
fi

echo "== fix43: the lora_arm_block() extraction surface is measured, not commented =="
# The defect this detector exists against is measured (fix43 receipt): five
# legs in the section above went red from ONE insertion position — the
# fix42 drill's top-level if..fi landing between EXPERT_TARGETS= and the
# arm block's closing fi truncated lora_arm_block()'s sed extraction at
# the INTRUDER's fi; the harness evaled the intruder, LORA_ARM was never
# set, and the WARN-mining MUST_FIRE went UNREACHABLE because its needle
# had fallen out of the extracted text. The repair (drill above
# EXPERT_TARGETS=) was shipped with a comment; a comment is exactly the
# class of guarantee this file refuses everywhere else, so here is the
# detector. WHAT IT KEYS ON, stated because the naive version is proven
# hollow by that same history: not extraction non-emptiness (the fix42-era
# extraction was non-empty — it was WRONG), but the presence of the arm
# DECISION SURFACE inside the extracted text: exactly one terminating
# top-level fi, the 'if [[ "$MOE" != "1" ]]; then' guard, and all three
# LORA_ARM= assignment sites (base4/L1/base4, all indented) that the legs
# above eval. An extraction truncated early carries none of these and goes
# red HERE, at authoring time, with the mechanism named — instead of five
# reds three hundred lines downstream.
f43_extraction_ok() { # $1=launcher file -> rc 0 iff lora_arm_block's extraction
                      # still carries the whole decision surface. The MUST_FIRE
                      # below runs THIS predicate on a doctored copy.
  local x
  x=$(lora_arm_block "$1")
  [ -n "$x" ] || return 1
  [ "$(printf '%s\n' "$x" | grep -c '^fi$' || true)" -eq 1 ] \
    && printf '%s\n' "$x" | grep -qF 'if [[ "$MOE" != "1" ]]; then' \
    && [ "$(printf '%s\n' "$x" | grep -cE '^[[:space:]]+LORA_ARM=' || true)" -eq 3 ]
}
if f43_extraction_ok "$LORA"; then
  ok "fix43 extraction-surface: lora_arm_block() spans EXPERT_TARGETS= to the arm block's OWN fi — 3 LORA_ARM= sites + the \$MOE guard survive in the evaled text (an early-terminating intruder if..fi would strip all three)"
else
  no "fix43 extraction-surface BROKEN: the arm extraction no longer carries the decision surface (exactly 1 top-level fi + \$MOE guard + 3 LORA_ARM= sites) — a top-level if..fi landed between EXPERT_TARGETS= and the arm block (the measured fix42 truncation; move it above EXPERT_TARGETS= or below the arm block's fi)"
fi

# MUST_FIRE (doctrine 3): awk-insert a top-level 'if :; then ... fi'
# intruder into a temp COPY immediately after the EXPERT_TARGETS= line —
# the exact fix42 insertion shape — and require (i) construction proven:
# the copy's extraction contains the intruder and carries NO L1
# assignment, (ii) the SAME predicate reports the copy NOT ok, and (iii)
# the live launcher still satisfies the predicate (the leg above), so the
# flip is constructed, not ambient. On the pre-repair tree the live
# conjunct fails and this leg is red (UNREACHABLE) alongside the leg it
# arms, never skipped.
f43_xt=$(mktemp "${TMPDIR:-/tmp}/fs-f43-extract.XXXXXX") \
  && awk '/^EXPERT_TARGETS=/ && !d { d=1; print; print "if :; then"; print "  : # harness-constructed intruder: the fix42 drill-block shape, a top-level if..fi above the real arm block"; print "fi"; next } { print }' \
       "$LORA" > "$f43_xt"
f43_xs=$?
f43_xcopy=""
[ "$f43_xs" -eq 0 ] && f43_xcopy=$(lora_arm_block "$f43_xt")
f43_xfired=1
if [ "$f43_xs" -eq 0 ] \
   && printf '%s\n' "$f43_xcopy" | grep -qF 'if :; then' \
   && ! printf '%s\n' "$f43_xcopy" | grep -qF 'LORA_ARM=L1' \
   && ! f43_extraction_ok "$f43_xt" \
   && f43_extraction_ok "$LORA"; then
  f43_xfired=0
fi
[ -n "${f43_xt:-}" ] && rm -f "$f43_xt" || true
if [ "$f43_xfired" -eq 0 ]; then
  ok "MUST_FIRE extraction-surface: a constructed top-level if..fi between EXPERT_TARGETS= and the arm block truncates the copy's extraction at the intruder's fi (contains it, carries no LORA_ARM=L1) and turns the predicate red — the comment-only contract is now a measured one"
else
  no "MUST_FIRE UNREACHABLE (extraction-surface): the intruder construction failed or the predicate still greens the doctored copy (awk rc=$f43_xs) — the leg above is an unproven detector"
fi

echo "== LoRA OPT_LR_KEY definition precedes use (set -u blocker) =="
# fix26b audit: `use` was greppable out of prose (a comment containing
# 'OPT_LR_KEY=$LORA_LR' would drag `use` down and loosen the ordering
# proof). Line numbers survive the strip 1:1, so the message's "line $def
# / line $use" stay literally true of the launcher on disk.
def=$(strip_shell_comments < "$LORA" | grep -n '^OPT_LR_KEY=' | head -1 | cut -d: -f1)
use=$(strip_shell_comments < "$LORA" | grep -n 'OPT_LR_KEY=\$LORA_LR' | head -1 | cut -d: -f1)
if [ -n "$def" ] && [ -n "$use" ] && [ "$def" -lt "$use" ]; then
  ok "OPT_LR_KEY defined at line $def, first used at line $use"
else no "OPT_LR_KEY def=$def use=$use — set -u would abort at the provenance gate"; fi

echo "== FS_ROOT resolves a real checkout, not a guessed one =="
# fix26b: greppable-out-of-prose legs. 'HOME/foundationscale' literally
# appears TODAY inside a comment in each launcher (the "measured
# 2026-08-23" note) — the conjunct was already half-satisfiable by prose;
# a comment quoting an assignment line would complete it and mint a green
# over reverted code. Both real assignment lines survive the strip, so
# this is strictly stronger with a frozen green.
for f in "$LORA" "$FULL"; do
  strip_shell_comments < "$f" | grep -q 'FS_CANDIDATES=' && strip_shell_comments < "$f" | grep -q 'HOME/foundationscale' \
    && ok "$(basename "$f"): searches candidates incl. \$HOME/foundationscale" \
    || no "$(basename "$f"): still asserts a single guessed FS_ROOT"
done

echo "== off-Slurm backend: node guard / provenance / drain / launch string =="
BE=$LDIR/fs_container_backend.sh
# These controls source the REAL library the launchers source — a strictly
# stronger guarantee than sed-extracting a copy (same rule this harness
# applies to the EXTRA_OVERRIDES block: no paraphrase is allowed to stand in
# for shipped code). Fail-before state, by construction: on the pre-patch tree
# $BE does not exist, and the srun call sites are still srun, so every check
# in this section FAILS there and PASSES only with the patch. The guard
# scenarios are driven entirely by stubs (hostname / enroot / nvidia-smi) in a
# mktemp sandbox; nothing touches the real tray.
if [ ! -f "$BE" ]; then
  no "fs_container_backend.sh missing — backend absent (controls below cannot pass)"
else
  SANDBOX=$(mktemp -d)
  trap '[ -n "${SANDBOX:-}" ] && rm -rf "$SANDBOX" || true' EXIT

  cat > "$SANDBOX/hostname" <<'SH'
#!/usr/bin/env bash
[ "${1:-}" = "-s" ] && { echo "${STUB_HOST:?}"; exit 0; }
echo "${STUB_HOST:?}.local"
SH
  cat > "$SANDBOX/enroot" <<'SH'
#!/usr/bin/env bash
case "${1:-}" in
  list)   [ -n "${ENROOT_LIST:-}" ] && printf '%s\n' $ENROOT_LIST; exit 0 ;;
  create|start|remove) exit 0 ;;
  *) exit 0 ;;
esac
SH
  cat > "$SANDBOX/nvidia-smi" <<'SH'
#!/usr/bin/env bash
# Four GPUs, all reporting the same stubbed used-memory figure.
for _ in 1 2 3 4; do echo "${STUB_MEM:-0}"; done
SH
  chmod +x "$SANDBOX/hostname" "$SANDBOX/enroot" "$SANDBOX/nvidia-smi"

  # fix26 (item C) — stat adapter, HARNESS-SIDE ONLY (option ii). The
  # backend pins GNU `stat -c %s` / `stat -c %Y`; production keeps them
  # untouched because the production platform is Linux, and that text
  # must not pay for a dev machine's convenience. What actually blocks
  # the dev machine from exercising the guard logic is a one-flag
  # platform difference — the same class the hostname/enroot/nvidia-smi
  # stubs above already exist to absorb — so the sandbox absorbs it too:
  # where the HOST stat rejects GNU syntax, drop a `stat` adapter FIRST
  # on the scenario PATH. It translates exactly the two forms the pinned
  # backend can emit; anything else falls through to the platform stat,
  # which errors loudly on unknown -c forms — fail closed (doctrine 4):
  # a future backend edit adding a third stat form turns controls RED on
  # dev machines instead of silently comparing garbage.
  if ! stat -c %s /dev/null >/dev/null 2>&1; then
    cat > "$SANDBOX/stat" <<'SH'
#!/usr/bin/env bash
# Sandbox-only GNU-syntax stat adapter (fix26 item C). NEVER production
# code, never on a production PATH: scenario() runs env -i with this
# directory first, so the real backend's `stat -c %s|%Y` resolve here on
# BSD hosts. Translates exactly those two forms; every exec re-resolves
# stat through a scrubbed PATH so the adapter cannot re-exec itself.
if [ "${1:-}" = "-c" ]; then
  case "${2:-}" in
    %s) exec env PATH=/usr/bin:/bin stat -f %z "${3:?stat adapter: missing file argument}" ;;
    %Y) exec env PATH=/usr/bin:/bin stat -f %m "${3:?stat adapter: missing file argument}" ;;
  esac
fi
exec env PATH=/usr/bin:/bin stat "$@"
SH
    chmod +x "$SANDBOX/stat"
    FS_STAT_WIRING="sandbox adapter: host stat is not GNU; the backend's 'stat -c %s|%Y' is translated to 'stat -f %z|%m'"
  else
    FS_STAT_WIRING="native GNU stat: host accepts 'stat -c'; the sandbox uses it unmodified"
  fi

  # WIRING MUST_PASS (doctrine 3 applies to wiring, not only detectors):
  # whatever `stat` the scenarios will actually inherit — native GNU on
  # the cluster, the adapter on a BSD dev box — must demonstrably answer
  # the two queries the backend issues. N = 2: a size read and an mtime
  # read against a probe of KNOWN content (a 5-byte file must read 5 and
  # an epoch-shaped mtime). If stat resolution ever breaks on a dev
  # machine, THIS one legible control goes red — instead of five
  # downstream scenario failures that look like five different bugs.
  printf '12345' > "$SANDBOX/stat-probe"   # exactly 5 bytes, no newline
  fs_qs=$(env -i PATH="$SANDBOX:/usr/bin:/bin" stat -c %s "$SANDBOX/stat-probe" 2>&1); fs_qs_rc=$?
  fs_qm=$(env -i PATH="$SANDBOX:/usr/bin:/bin" stat -c %Y "$SANDBOX/stat-probe" 2>&1); fs_qm_rc=$?
  if [ "$fs_qs_rc" -eq 0 ] && [ "$fs_qs" = "5" ] \
     && [ "$fs_qm_rc" -eq 0 ] && printf '%s' "$fs_qm" | grep -qE '^[0-9]+$'; then
    ok "stat wiring verified ($FS_STAT_WIRING): 2 of 2 backend stat queries answered [5 bytes, mtime epoch]"
  else
    no "stat under the scenario PATH cannot answer the backend's two queries (size rc=$fs_qs_rc out='$fs_qs'; mtime rc=$fs_qm_rc out='$fs_qm') — every scenario control below would be a sweep over zero units; BLOCK"
  fi

  # Runs fs_backend_init + fs_backend_runtime_setup in a clean env-emptied
  # bash, against the stubbed tray. fs_die exits, so each scenario is a
  # subprocess whose rc and combined output the caller inspects.
  # args: $1 hostname-s  $2 enroot-list content  $3 MiB used  $4 drain timeout s  $5 record mode (match|mismatch|absent)
  #
  # fix26 (item A): each invocation gets its OWN HOME — a per-call
  # mktemp -d under $SANDBOX, so the ONE existing EXIT trap that reaps
  # $SANDBOX still reaps every scenario HOME with it. Isolation is made
  # STRUCTURAL, not remembered: previously every scenario shared
  # $SANDBOX/home, so the provenance record written by the mismatch
  # scenario survived into the absent scenario, and the orphan MUST_FIRE
  # never once executed against an orphan — it refused with the
  # provenance-mismatch message and reported red for a reason that proved
  # nothing about its stated condition. The other per-run state named in
  # the diagnosis (the truncated image, the append-only tee log that
  # accumulated every run into one file, any future store state an
  # upgraded enroot stub might persist under $HOME/.enroot) is now
  # equally unreachable by later scenarios.
  scenario() {
    local shm
    shm=$(mktemp -d "$SANDBOX/scenario.XXXXXX") || { echo "scenario: mktemp -d failed — refusing to run a scenario against a shared or unknown HOME" >&2; return 99; }
    # FS_ALLOWED_NODE / FS_FORBIDDEN_NODES are REQUIRED by the backend's node
    # guard and deliberately have no default (an unset allowlist must refuse,
    # never pass -- a guard that cannot fire is not a guard). env -i wipes the
    # environment, so the harness has to supply them here or every scenario leg
    # goes red for the wrong reason.
    #
    # They are SYNTHETIC on purpose. The harness used to hard-code the estate's
    # real tray hostnames, which (a) published them from a public repo and (b)
    # coupled the test suite to one site's naming. test-node-a is the allowed
    # tray, test-node-b stands in for another team's node that the standing rule
    # forbids; the pair is what the guard is actually about, so both are needed.
    env -i PATH="$SANDBOX:/usr/bin:/bin" HOME="$shm/home" \
        FS_ALLOWED_NODE=test-node-a FS_FORBIDDEN_NODES=test-node-b \
        STUB_HOST="$1" ENROOT_LIST="$2" STUB_MEM="$3" FS_GPU_DRAIN_TIMEOUT_S="$4" RECMODE="$5" \
        bash -s "$BE" <<'SC'
set -uo pipefail
BE=$1
SQSH="$HOME/img/nemo-demo.sqsh"
REC="$HOME/.enroot/.fs-provenance/fs-g4e4b-nemo-demo.src"
mkdir -p "$HOME/img" "$HOME/.enroot/.fs-provenance" "$HOME/tee"
: > "$SQSH"
# shellcheck disable=SC1090
source "$BE"
fs_backend_init "$HOME/tee"
case "${RECMODE:-match}" in
  match)    { echo "sqsh_path=$SQSH"; echo "sqsh_size=$(stat -c %s "$SQSH")"; echo "sqsh_mtime=$(stat -c %Y "$SQSH")"; echo created_epoch=1; } > "$REC" ;;
  mismatch) { echo "sqsh_path=$SQSH"; echo sqsh_size=424242; echo sqsh_mtime=424242; } > "$REC" ;;
  # This arm MEANS: container present, record never written. The per-call
  # HOME minted above already guarantees that structurally — nothing a
  # previous scenario wrote can exist here at all. The rm -f is kept
  # anyway as a written assertion of the intended start state, so if
  # isolation is ever regressed or bypassed (a future refactor handing
  # scenario() a caller-chosen HOME, say) the arm still certifies "no
  # record" instead of accidentally testing whatever a predecessor left.
  # Belt and braces, on purpose: one mechanism, one assertion, and this
  # comment exists so neither half is "tidied" away as redundant.
  absent)   rm -f "$REC" ;;
esac
fs_backend_runtime_setup "$SQSH" 4 "$HOME/tee/job.out"
printf 'LAUNCH:%s\n' "$(MASTER_PORT=29999 fs_launch_python 4)"
SC
  }

  # MUST_FIRE: the forbidden-node standing rule must be enforceable off-Slurm.
  # Broken to see red: stubbed `hostname -s` reports test-node-b (the synthetic
  # stand-in for another team's node) — if the guard consulted anything mintable
  # (or nothing), this would pass and the control would be vacuous. It must
  # refuse BEFORE any SLURM_* value exists.
  out=$(scenario test-node-b fs-g4e4b-nemo-demo 0 0 match 2>&1); rc=$?
  if [ $rc -ne 0 ] && printf '%s' "$out" | grep -q "STANDING RULE VIOLATION"; then
    ok "MUST_FIRE off-Slurm guard refuses hostname -s = test-node-b (forbidden node)"
  else no "MUST_FIRE off-Slurm guard accepted test-node-b (rc=$rc): $out"; fi

  # MUST_FIRE: a pre-set SLURM_JOB_ID off-Slurm is the self-fulfilling-string
  # smell; the enroot arm must refuse to inherit it. Broken to see red: env
  # injects SLURM_JOB_ID=12345 with FS_BACKEND=enroot.
  out=$(env -i PATH="$SANDBOX:/usr/bin:/bin" HOME="$SANDBOX/h2" STUB_HOST=test-node-a \
        FS_ALLOWED_NODE=test-node-a FS_FORBIDDEN_NODES=test-node-b \
        FS_BACKEND=enroot SLURM_JOB_ID=12345 \
        bash -c 'source "'"$BE"'"; fs_backend_init /tmp' 2>&1); rc=$?
  if [ $rc -ne 0 ] && printf '%s' "$out" | grep -q "pre-set"; then
    ok "MUST_FIRE pre-set SLURM_JOB_ID refused on the enroot arm"
  else no "MUST_FIRE pre-set SLURM_JOB_ID not refused (rc=$rc): $out"; fi

  # ---- controls for the node guard's own configuration (doctrine 3) --------
  # The guard is driven by FS_ALLOWED_NODE / FS_FORBIDDEN_NODES, which replaced
  # the hard-coded tray hostnames. That refactor introduced three new ways to be
  # wrong, and every scenario above supplies BOTH variables correctly — so none
  # of them can observe any of the three. Without the four legs below the
  # fail-closed refusal is an unproven detector, which is the exact defect class
  # this estate exists to eliminate.

  # MUST_FIRE: FS_ALLOWED_NODE is REQUIRED and has NO default. Unset must
  # REFUSE, never pass — the whole point of giving it no default. Broken to see
  # red: default it to anything and this leg goes green while a typo silently
  # disables the standing rule.
  out=$(env -i PATH="$SANDBOX:/usr/bin:/bin" HOME="$SANDBOX/h3" STUB_HOST=test-node-a \
        bash -c 'source "'"$BE"'"; fs_backend_init /tmp' 2>&1); rc=$?
  if [ $rc -ne 0 ] && printf '%s' "$out" | grep -q "FS_ALLOWED_NODE"; then
    ok "MUST_FIRE unset FS_ALLOWED_NODE refuses (fail-closed; message names the variable)"
  else no "MUST_FIRE unset FS_ALLOWED_NODE did not refuse (rc=$rc): $out"; fi

  # MUST_FIRE: EMPTY is a distinct failure from UNSET — `${VAR:-}` conflates
  # them, `${VAR-}` does not. An `export FS_ALLOWED_NODE=` typo, or a lookup
  # that returned nothing, must refuse on the same terms. Broken to see red:
  # write the guard with `${FS_ALLOWED_NODE+set}` and only this leg turns red.
  out=$(env -i PATH="$SANDBOX:/usr/bin:/bin" HOME="$SANDBOX/h4" STUB_HOST=test-node-a \
        FS_ALLOWED_NODE= \
        bash -c 'source "'"$BE"'"; fs_backend_init /tmp' 2>&1); rc=$?
  if [ $rc -ne 0 ] && printf '%s' "$out" | grep -q "FS_ALLOWED_NODE"; then
    ok "MUST_FIRE empty FS_ALLOWED_NODE refuses (empty is not 'configured')"
  else no "MUST_FIRE empty FS_ALLOWED_NODE did not refuse (rc=$rc): $out"; fi

  # MUST_FIRE: deny-before-allow ORDERING. The host below matches BOTH — the
  # allow-prefix 'test-node' would admit it and the denylist must refuse it
  # first. This is the only leg that can see the ordering: an allow-first
  # implementation passes every other control in this file and fails only here.
  # It is the sloppy-allowlist case the denylist exists for.
  out=$(env -i PATH="$SANDBOX:/usr/bin:/bin" HOME="$SANDBOX/h5" STUB_HOST=test-node-b \
        FS_ALLOWED_NODE=test-node FS_FORBIDDEN_NODES=test-node-b \
        bash -c 'source "'"$BE"'"; fs_backend_init /tmp' 2>&1); rc=$?
  if [ $rc -ne 0 ] && printf '%s' "$out" | grep -q "FS_FORBIDDEN_NODES entry"; then
    ok "MUST_FIRE deny-before-allow: denylisted host refused though the allow-prefix matches it"
  else no "MUST_FIRE deny-before-allow failed — allow-prefix admitted a denylisted host (rc=$rc): $out"; fi

  # MUST_PASS: the guard must be LIFTABLE. A refusal that can never be satisfied
  # is not a control either — it is an outage, and it would make the three legs
  # above pass for the wrong reason. Asserting rc=0 alone would be vacuous (an
  # early return also yields 0), so this also requires the backend to say it
  # reached the enroot arm ON the allowed host.
  out=$(env -i PATH="$SANDBOX:/usr/bin:/bin" HOME="$SANDBOX/h6" STUB_HOST=test-node-a \
        FS_ALLOWED_NODE=test-node-a FS_FORBIDDEN_NODES=test-node-b \
        bash -c 'source "'"$BE"'"; fs_backend_init /tmp' 2>&1); rc=$?
  if [ $rc -eq 0 ] && printf '%s' "$out" | grep -q "enroot arm on test-node-a"; then
    ok "MUST_PASS configured FS_ALLOWED_NODE admits the allowed host and reaches the enroot arm"
  else no "MUST_PASS configured FS_ALLOWED_NODE did not admit the allowed host (rc=$rc): $out"; fi

  # MUST_FIRE: recorded source no longer matches the pinned image.
  # Broken to see red: the provenance record claims a size/mtime the (stubbed)
  # image does not have — the g4export-in-reverse case.
  out=$(scenario test-node-a fs-g4e4b-nemo-demo 0 0 mismatch 2>&1); rc=$?
  if [ $rc -ne 0 ] && printf '%s' "$out" | grep -q "provenance mismatch"; then
    ok "MUST_FIRE container-provenance mismatch refuses reuse"
  else no "MUST_FIRE provenance mismatch NOT refused (rc=$rc): $out"; fi

  # MUST_FIRE: existing container with NO record at all — the actual g4export
  # class (s3): name matches, origin unknown. Must refuse, must not auto-rm.
  out=$(scenario test-node-a fs-g4e4b-nemo-demo 0 0 absent 2>&1); rc=$?
  if [ $rc -ne 0 ] && printf '%s' "$out" | grep -q "NO provenance record"; then
    ok "MUST_FIRE orphan container (no provenance record) refuses"
  else no "MUST_FIRE orphan container NOT refused (rc=$rc): $out"; fi

  # ISOLATION SELF-CHECK (fix26 item A; doctrine 3 applied to the harness
  # itself). This is the check that WOULD have caught the shared-HOME
  # defect: recreate the historical ordering on purpose — mismatch THEN
  # absent — and assert two conjuncts on the absent run: (i) output
  # contains "NO provenance record", (ii) output does NOT contain
  # "provenance mismatch". It is the negative conjunct that bites: when
  # the record leaked, the absent run still refused (rc=1) but refused
  # with "provenance mismatch" — the WRONG guard fired, and rc plus the
  # positive grep alone could never tell the difference. A refusal of
  # unknown provenance is no better than a pass of unknown provenance.
  scenario test-node-a fs-g4e4b-nemo-demo 0 0 mismatch >/dev/null 2>&1 || true
  out=$(scenario test-node-a fs-g4e4b-nemo-demo 0 0 absent 2>&1); rc=$?
  if [ $rc -ne 0 ] \
     && printf '%s' "$out" | grep -q "NO provenance record" \
     && ! printf '%s' "$out" | grep -q "provenance mismatch"; then
    ok "self-check: mismatch-then-absent sees NO record and no mismatch text (scenario isolation holds)"
  else no "self-check FAIL: absent-after-mismatch did not see a clean orphan (rc=$rc) — scenario isolation is broken, so the orphan control above is again certifying an unknown condition: $out"; fi

  # MUST_FIRE: drain-poll timeout is a REFUSAL, not a warning that proceeds.
  # Broken to see red: every stubbed GPU reports 60000 MiB and the timeout
  # budget is 0 s, so the first undrained reading must die immediately.
  out=$(scenario test-node-a fs-g4e4b-nemo-demo 60000 0 match 2>&1); rc=$?
  if [ $rc -ne 0 ] && printf '%s' "$out" | grep -q "timed out"; then
    ok "MUST_FIRE drain-poll timeout refuses to launch"
  else no "MUST_FIRE drain timeout did not refuse (rc=$rc): $out"; fi

  # MUST_PASS: simulated <compute-node>, matching container record, idle GPUs:
  # guard chain passes end to end and the emitted rank launcher is torchrun
  # with the tray's real denominator.
  out=$(scenario test-node-a fs-g4e4b-nemo-demo 0 0 match 2>&1); rc=$?
  if [ $rc -eq 0 ] \
     && printf '%s' "$out" | grep -q "provenance matches" \
     && printf '%s' "$out" | grep -q "drain gate: examined 4 GPUs" \
     && printf '%s' "$out" | grep -q "LAUNCH:torchrun --nproc_per_node=4"; then
    ok "MUST_PASS happy path: guard+provenance+drain pass; command carries 'torchrun --nproc_per_node=4'"
  else no "MUST_PASS happy path broke (rc=$rc): $out"; fi
fi

echo "== single-executor accounting (a host-python probe must be impossible to reintroduce silently) =="
# Denominator re-anchored from retired history to a LIVE ENUMERATION (fix43;
# measured cause in the fix43 receipt). The pre-fix43 leg pinned LoRA=3 +
# full-FT=2 "because those were the srun sites the executor patch
# converted" — a tally welded to a historical fact. fix42 then routed a 4th
# LoRA call through the same executor (the step-(7) override-replay probe)
# and the leg went red although the launcher was CORRECT: what the header
# above demands is ROUTING, not history. The contract is therefore the live
# census of WHERE the sites live: LoRA 5 = 1 census (step (5)) + 1 env
# probe (step (6)) + 1 override replay (step (7)) + 1 training invocation +
# 1 ARTIFACT-GATE invocation (the fix44/#77-B1 addition, NAMED: until fix44
# fs_live_save_gate was this launcher's only python call off the executor,
# and it reached hardware that way precisely because this census counts
# positives — a positive census cannot see a bypasser, which is why fix44
# adds the python-call-site complement census further down), each counted
# over its own sed-extracted region by the shared counter, and full-FT 4 =
# 1 tokenizer/CoT probe + 1 training invocation + 1 manifest-emitter
# invocation + 1 ARTIFACT-GATE invocation (fix45-A named the gate into the
# region; fix45-A2 names the emitter — with --full-ft it READS THE BASE
# DCP, the measured hard-block on every historical full-FT launch
# (<compute-node>), so it joined the gate on the adjudicator stack. The cause on
# record in the launcher is PYTHONNOUSERSITE-hides-user-site-torch plus the
# three refused reasons not to un-hide it; what this comment and the
# predicate below pin is the ROUTING — no conjunct depends on the host
# being unable to read a DCP, because on this host it can: arm B measured
# CLEAR. This leg's own author previously pinned the count at 2 with the
# deferral cited: "its file was not in this packet"). If a FIFTH site
# appears, or a site migrates to compensate for a deleted sibling while the
# total holds 4 (the fix40 merged-count blind spot), a site-name conjunct
# goes red on a green total.
# The shared source-counting helpers (FS_DECOMMENT_AWK / strip_shell_comments /
# pos_pat / pos_count) that the counts below use are defined at the TOP of
# this file, above their first caller. See the rationale there.
f43_lora_site_counts() { # $1=launcher file -> stdout "TOTAL N_CENSUS N_ENV N_REPLAY N_TRAIN";
                         # UNREADABLE + rc 1 for a missing subject (fail closed: an
                         # absent launcher is not a zero-site launcher, doctrines 1/4).
  local f=$1 d t5 t6 t7 tt
  [ -r "$f" ] || { printf 'UNREADABLE\n'; return 1; }
  d=$(mktemp -d "${TMPDIR:-/tmp}/fs-f43-sites.XXXXXX") || { printf 'UNREADABLE\n'; return 1; }
  t5=$d/r5; t6=$d/r6; t7=$d/r7; tt=$d/rt
  # Region boundaries are the launcher's own step markers — the same
  # sed-range-over-the-real-file rule this harness applies to every
  # extraction. '^RC=0$' occurs exactly once in the launcher, immediately
  # before the training-invocation region.
  sed -n '/^# (5) /,/^# (6) /p' "$f" > "$t5"
  sed -n '/^# (6) /,/^# (7) /p' "$f" > "$t6"
  sed -n '/^# (7) /,/^RC=0$/p'   "$f" > "$t7"
  sed -n '/^RC=0$/,$p'           "$f" > "$tt"
  printf '%s %s %s %s %s\n' \
    "$(pos_count run_in_container "$f")" \
    "$(pos_count run_in_container "$t5")" \
    "$(pos_count run_in_container "$t6")" \
    "$(pos_count run_in_container "$t7")" \
    "$(pos_count run_in_container "$tt")"
  rm -rf "$d"
}
f43_lora_executor_ok() { # $1=launcher file -> rc 0 iff the LoRA census is the pinned tuple.
                         # fix44: tuple moved 4 1 1 1 1 -> 5 1 1 1 2 with the
                         # artifact-gate's executor call NAMED into the trailing
                         # region (fs_live_save_gate is defined after RC=0).
                         # The MUST_FIRE below runs THIS predicate on a doctored
                         # copy — never a paraphrase of it.
  [ "$(f43_lora_site_counts "$1")" = "5 1 1 1 2" ]
}
f45_full_executor_ok() { # $1=launcher file -> rc 0 iff the full-FT census is
                         # exactly the four enumerated sites. fix45-A argued
                         # the gate was this file's "only torch-importing
                         # python call"; the <compute-node> run refuted the sibling
                         # half of that claim: the manifest emitter ALSO
                         # imports the DCP stack (with --full-ft it censuses
                         # the base checkpoint), and its executor routing is
                         # the delta that produced the first successful
                         # full-FT run. Both torch-importing calls are named
                         # here as conjuncts — the bare total cannot see a
                         # compensating migration (the fix40 merged-count
                         # lesson), so the names are tested, not narrated.
                         # fix45-A2, stated on the predicate itself: no
                         # conjunct below asserts the host is UNABLE to read
                         # a DCP — measured on <compute-node>, it can (arm B
                         # CLEAR); ROUTING is the pinned property, and the
                         # refusal of the cheap host alternative is kept in
                         # the launcher comment the record legs pin. The
                         # section below runs THIS predicate on doctored
                         # copies.
  local f=$1 gate_span
  [ -r "$f" ] || return 1
  gate_span=$(sed -n '/^fs_live_save_gate() {/,/^}/p' "$f" | strip_shell_comments)
  [ "$(pos_count run_in_container "$f")" = "4" ] \
    && printf '%s\n' "$gate_span" | grep -qF 'run_in_container --slurm-ntasks 1 --workdir "$REPO"' \
    && strip_shell_comments < "$f" | grep -qF 'bash -lc "python3 $PROBE"' \
    && strip_shell_comments < "$f" | grep -qF 'run_in_container --workdir "$REPO" bash -lc "$CMD" &' \
    && strip_shell_comments < "$f" | grep -qF "python3 '\$FS_ROOT/tools/emit_run_manifest.py'" \
    && awk '/run_in_container --slurm-ntasks 1/ && /\\$/ {getline nxt; if (nxt ~ /tools\/emit_run_manifest\.py/) found=1} END{exit !found}' "$f"
}
lora_ric=$(pos_count run_in_container "$LORA")
full_ric=$(pos_count run_in_container "$FULL")
lora_srun=$(pos_count srun "$LORA")
full_srun=$(pos_count srun "$FULL")
f43_sites=$(f43_lora_site_counts "$LORA")
if f43_lora_executor_ok "$LORA" && f45_full_executor_ok "$FULL"; then
  ok "run_in_container sites: LoRA 5 = census(1)+env-probe(1)+override-replay(1)+training(1)+artifact-gate(1) [$f43_sites], full-FT 4 = cot-probe(1)+training(1)+manifest-emitter(1)+artifact-gate(1) [$full_ric, every site named as a conjunct] — every container invocation routes through the executor on BOTH launchers; the fix45 series closed the cited-but-unenumerated 'full-FT=2; its file was not in this packet' deferral (gate in fix45-A, emitter in fix45-A2 — the emitter reads the base DCP, measured <compute-node>), and zero dependence on the retired 3+2 historical denominator stands"
else no "run_in_container census drifted: LoRA tuple '$f43_sites' (required '5 1 1 1 2' = census/env-probe/replay/training+artifact-gate), full-FT=$full_ric (required 4 = cot-probe/training/manifest-emitter/artifact-gate, every site grep-named in f45_full_executor_ok) — an unannounced site appeared, a probe wired past the executor, a migrated site compensating for a deleted sibling on a green total, or the executor bypassed (the #77-B1 class: a torch-importing call adjudicating outside the stack that wrote the artifact; the host CAN read the DCP — routing, never incapacity, is the pinned property)"; fi

# MUST_FIRE for the enumeration itself (doctrine 3), complementing the
# raw-counter sensitivity leg below: that twin leg proves the TOTAL moves;
# this one proves the REGION ATTRIBUTION moves. The covered regression is
# the training invocation leaving the executor while a second site appears
# elsewhere — the total holds 4 and only the tuple sees it. Delete the
# first post-RC=0 site (the training call) from a temp COPY and require
# (i) the copy's census to be exactly '4 1 1 1 1' — construction proven:
# the deletion hit the training invocation and nothing else (the fix44
# artifact-gate site, defined later in the trailing region, survives) —
# and (ii) the SAME predicate the live leg runs to report the copy NOT ok.
# The live green '5 1 1 1 2' above is this leg's MUST_PASS twin.
f43_et=$(mktemp "${TMPDIR:-/tmp}/fs-f43-enusites.XXXXXX") \
  && awk -v p="$(pos_pat run_in_container)" "$FS_DECOMMENT_AWK"'BEGIN{t=0;d=0} /^RC=0$/{t=1} t && !d && decomment($0) ~ p {d=1; next} {print}' "$LORA" > "$f43_et"
f43_es=$?
f43_copied=""
[ "$f43_es" -eq 0 ] && f43_copied=$(f43_lora_site_counts "$f43_et")
f43_fired=1
if [ "$f43_es" -eq 0 ] && [ "$f43_copied" = "4 1 1 1 1" ] && ! f43_lora_executor_ok "$f43_et"; then
  f43_fired=0
fi
[ -n "${f43_et:-}" ] && rm -f "$f43_et" || true
if [ "$f43_fired" -eq 0 ]; then
  ok "MUST_FIRE executor-enumeration: removing the training site from a copy moves its census to '$f43_copied' and turns the predicate red — the region attribution can move and the leg reads it (a count nobody can move is wallpaper)"
else
  no "MUST_FIRE UNREACHABLE (executor-enumeration): the training-site deletion did not construct or did not fire (awk rc=$f43_es; copy census '${f43_copied:-<none>}') — the enumeration leg above is an unproven detector"
fi
if [ "$lora_srun" -eq 0 ] && [ "$full_srun" -eq 0 ]; then
  ok "no command-position srun remains in either launcher (LoRA=$lora_srun full-FT=$full_srun)"
else no "raw srun survived in a launcher (LoRA=$lora_srun full-FT=$full_srun) — an off-Slurm-dead call site, or a probe wired past the executor"; fi

# MUST_FIRE leg for the counter itself (fix26 item B; doctrine 3): the
# counts above are a detector only if they MOVE when a call site moves.
# Delete the first site THE COUNTER COUNTS from a temp COPY of the LoRA
# launcher — since fix26b the matcher IS the counter's view (the same
# $FS_DECOMMENT_AWK program, not a paraphrase of it), so the deleted
# line can never be a prose phantom the count itself would ignore, and
# "which line is first" cannot drift between the two legs of one claim
# — and require the tally to fall by EXACTLY one. A counter that reads
# 3 before and 3 after a deletion is wallpaper. The green pinned
# enumeration tuple above (and its own fix43 MUST_FIRE) is the MUST_PASS
# half of that claim; this is the
# firing leg; fix26b ships the third leg below (the numerator unmoved
# by prose). Self-cleaned; any failure to build the copy is a red
# control, never a silent skip.
mft=$(mktemp "${TMPDIR:-/tmp}/fs-ric-mustfire.XXXXXX") \
  && awk -v p="$(pos_pat run_in_container)" "$FS_DECOMMENT_AWK"'BEGIN{d=0} !d && decomment($0) ~ p {d=1; next} {print}' "$LORA" > "$mft"
mfs=$?
mfc=$(pos_count run_in_container "$mft")
[ -n "${mft:-}" ] && rm -f "$mft" || true
if [ "$mfs" -eq 0 ] && [ "$mfc" -eq $((lora_ric - 1)) ]; then
  ok "MUST_FIRE counter sensitivity: deleting one site from a copy drops the count $lora_ric -> $mfc (numerator tracks reality)"
else no "MUST_FIRE counter NOT sensitive: pre=$lora_ric post-deletion=$mfc, copy-build status=$mfs — the counts above prove nothing"; fi

# MUST_NOT_MOVE leg (fix26b; the other half of one claim — doctrine 3).
# The sensitivity leg proves the numerator MOVES with reality; this leg
# proves it moves ONLY with reality — that a comment naming a symbol in
# perfect command-position costume does not move it. fix26 shipped the
# first half alone, which is exactly how a permanently-red zero-control
# reached production on a comment's say-so. Built from the REAL offender
# (the full-FT:508 text, verbatim), the "& run_in_container" prose form
# fix26's own comment predicted, and one INLINE comment carrying both at
# once, to pin the class and not the instance. Both counts must be
# byte-for-byte UNCHANGED against the live LoRA tallies (ric=4, srun=0 — ric
# tracks the live enumeration of Edit-2's leg, so a comment can never pin it
# stale again; the number here is narration, the comparison is dynamic).
mc=$(mktemp "${TMPDIR:-/tmp}/fs-ric-comment.XXXXXX") \
  && { cat "$LORA"; \
       printf '# SRUN_PID is the historical name; it is now the run_in_container pid (srun on\n'; \
       printf '# the sbatch arm, enroot start on the enroot arm).\n'; \
       printf '# see & run_in_container — the executor wraps the call\n'; \
       printf 'SRUN_PID=$!  # historical name; the pid of (srun on the sbatch arm) & run_in_container\n'; \
     } > "$mc"
mcs=$?
mc_ric=$(pos_count run_in_container "$mc")
mc_srun=$(pos_count srun "$mc")
[ -n "${mc:-}" ] && rm -f "$mc" || true
if [ "$mcs" -eq 0 ] && [ "$mc_ric" -eq "$lora_ric" ] && [ "$mc_srun" -eq "$lora_srun" ]; then
  ok "MUST_NOT_MOVE comment immunity: '(srun on' / '& run_in_container' prose appended -> LoRA ric=$mc_ric srun=$mc_srun, unchanged from $lora_ric/$lora_srun (pre-fix26b the zero-control read full-FT srun=1 on exactly this prose)"
else no "MUST_NOT_MOVE comment immunity broke (copy rc=$mcs): ric $lora_ric -> $mc_ric, srun $lora_srun -> $mc_srun — the numerator is reading prose again"; fi

echo "== structural bundle: shelves the patch must keep standing =="
# This predicate is FALSE on the pre-patch tree (no backend file, no source
# lines) and TRUE after: one srun and one `enroot start` in the library, both
# launchers sourcing it, #SBATCH headers preserved, s9 tripwire present on the
# sbatch arm. fix26b audit of what can match PROSE: every leg that asserts
# "this code exists" now reads the comment-stripped view — most load-bearing
# was 'source .*fs_container_backend\.sh', which the launchers' own
# `# shellcheck source=launchers/fs_container_backend.sh` directives already
# matched on their own, so a reverted source line could have stayed green on
# the strength of a comment. The REAL source lines ('source "$(cd ...pwd)"',
# the backend's enroot invocation, the tripwire string in backend code)
# survive the strip, so every frozen green below stays frozen. The two
# #SBATCH legs are the deliberate exception, left RAW with intent: an SBATCH
# directive IS a comment — that is its delivery mechanism — so stripping
# would erase the artifact under test (and sbatch's own parser reads the
# same comment lines; the grep view stays faithful to the consumer view,
# while the ^#SBATCH anchor means prose can satisfy it only by literally
# BEING a job-name directive).
if [ -f "$BE" ] \
   && [ "$(strip_shell_comments < "$BE" | grep -cE '^[[:space:]]*srun[[:space:]]' || true)" -eq 1 ] \
   && strip_shell_comments < "$BE" | grep -q 'enroot start ' \
   && strip_shell_comments < "$LORA" | grep -q 'source .*fs_container_backend\.sh' \
   && strip_shell_comments < "$FULL" | grep -q 'source .*fs_container_backend\.sh' \
   && grep -q '^#SBATCH --job-name=' "$LORA" \
   && grep -q '^#SBATCH --job-name=' "$FULL" \
   && strip_shell_comments < "$BE" | grep -q 'master:8081'; then
  ok "executor single-homed in the library; launchers source it; #SBATCH headers and master:8081 tripwire intact"
else no "structural bundle failed (executor/library/sbiiiatch-header/tripwire accounting)"; fi

echo "== fix28 controls (reintegrated; estate-side abstains by name) =="
# INTEGRITY NOTE, stated before any check: fix28b's 127-line controls snippet
# was LISTED in the fix30 packet but its body was not included in it (0/127
# lines visible here beyond the line 114 the task quotes and the helper/env
# names it spells out: f28, f28_req, f28_bad, FIX28_ESTATE_GEMMA4_VL,
# FIX28_ESTATE_INIT). Minting its launcher-side needle table from imagination
# would be a claim of content never observed — the symmetric doctrine-5
# defect — so this section ships three things and abstains by name on the
# fourth:
#   (i)   the integration MECHANICS Part B requires: helpers under the
#         snippet's own names, wired into this harness's ok/no and pass/fail
#         counters — one accounting, one summary line; nothing in this
#         section calls exit; the harness owns the exit. The bodies of
#         fix28b's snippet legs, when supplied, paste in under these helpers
#         with no restructuring (that is the transplant contract the comment
#         above each helper pins).
#   (ii)  the checks constructible from evidence in THIS packet: bash -n on
#         both launchers (the ground rule) with a constructed MUST_FIRE, and
#         the gated estate A2 tripwire whose needles the task quotes verbatim
#         (Part C), delimiter-repaired and flip-verified.
#   (iii) named abstentions with explicit denominators for the estate files
#         (0/2 — they have not landed) and for the snippet's unseen
#         launcher-side battery (0/127 lines available). Neither adds
#         anything to pass or fail.
# As everywhere in this file, the helpers are defined ABOVE their first call
# site (bash resolves names at call time; a late definition is no definition
# and the call reads vacuous empty input).
f28(){ printf '  fix28: %s\n' "$*"; }                    # annotation line; no counting (mirrors the snippet's label helper)
f28_bad(){ no "fix28: $*"; }                             # private fail helper, retired onto the harness's no()
f28_req(){ # $1=file $2=fixed needle that must be PRESENT $3=label
  if grep -qF -- "$2" "$1"; then ok "$3"; else no "$3 — required pattern absent in $(basename "$1"): $2"; fi
}
f28_nix(){ # $1=file $2=fixed needle that must be ABSENT $3=label
  if grep -qF -- "$2" "$1"; then no "$3 — forbidden pattern present in $(basename "$1"): $2"; else ok "$3"; fi
}
f28_must_fire(){ # $1=file $2=sed -E expr building the defect on a temp copy $3=healthy pattern the
                 # corruption must ERASE $4=defect pattern it must CREATE $5=label
  # A MUST_FIRE whose firing input was never constructed is UNREACHABLE, and
  # unreachable is a FAILED control here, never a skip — an unexercised
  # detector proves nothing (doctrine 3). Both verification needles are
  # therefore load-bearing call-site inputs, not optional decoration.
  local tmp bad=0
  tmp=$(mktemp "${TMPDIR:-/tmp}/fs-f28-mf.XXXXXX") || { no "MUST_FIRE UNREACHABLE ($5): mktemp failed — firing input unbuildable"; return 0; }
  sed -E "$2" "$1" > "$tmp" || bad=1
  if [ "$bad" -eq 0 ] && grep -qF -- "$3" "$tmp"; then bad=1; fi
  if [ "$bad" -eq 0 ] && ! grep -qF -- "$4" "$tmp"; then bad=1; fi
  rm -f "$tmp"
  if [ "$bad" -eq 0 ]; then
    ok "MUST_FIRE reachable ($5): the defect shape constructs on a temp copy of $(basename "$1")"
  else
    no "MUST_FIRE UNREACHABLE ($5): the corruption did not produce the defect shape — the firing input was never exercised, so the check it arms proves nothing"
  fi
}

# --- fix33: E4B region extraction — the one scope every A2 verdict reads -----
# fix30-T1 widened the re-arm grep to FILE scope because the snippet's region
# sentinels were among its 0/127 unseen lines and it refused to invent them;
# it named the resulting blindness and the repair in advance, both of which
# arrived on schedule (see the comment above the estate battery below). The
# sentinels are now MEASURED on the landed estate file (gemma4_vl.py:620
# begin / :957 end; region = lines 620-957 inclusive, 338 lines — the fix33
# packet's numbers, recorded here so a future diff of the landed file
# against these constants is itself legible). Single-definition matters
# doubly for the sentinels: they are ALSO interpolated below as sed range
# addresses to construct the in-region-only and out-of-region-only defect
# copies, so a second spelling anywhere would fork the tripwire's scope from
# its controls' scope. The re-arm regex is byte-identical to the one
# fix30-T1 shipped (the fix33 packet: do not relax the pattern — only the
# input it reads changes); F28_E4B_NONE_PIN is the pin needle; FIRED_PIN is
# the exact string the shared sed creates; the sed itself is one shared BRE
# program with a '#' delimiter (the needle contains '|', no BRE metachar,
# so the fix30-T1 line-114 delimiter-collision class stays unrepresentable).
F28_E4B_REGION_BEGIN='# --- fix28 E4B region begin'
F28_E4B_REGION_END='# --- fix28 E4B region end'
F28_E4B_REARM_RE='recompute_(granularity|method|num_layers)[^#=]*= *"(selective|full)"'
F28_E4B_NONE_PIN='recompute_granularity: "str | None" = None'
F28_E4B_FIRED_PIN='recompute_granularity: "str | None" = "selective"'
F28_E4B_REARM_SED='s#recompute_granularity: "str | None" = None#recompute_granularity: "str | None" = "selective"#'

f28_e4b_region() { # $1 = estate file (real or constructed copy)  $2 = output path
  # rc 0: the E4B region, sentinel lines included, has been written to $2;
  #       stdout is silent. Callers derive the denominator with grep -c ''.
  # rc 1: stdout carries exactly one 'REFUSAL: <what could not be found or
  #       verified>' line for the caller's ok/no message, and nothing in $2
  #       is a region anyone may read a verdict off of.
  # The refusal classes are the fix33 packet's list — a missing or
  # duplicated begin, a missing or duplicated end, a begin that does not
  # precede its end, and an extracted span containing zero recompute_
  # surface lines — and each is a RED BY NAME at every call site, never a
  # pass and never a skip. They exist because the trap they close is
  # doctrine 1 in its purest form: a naive 'sed -n /begin/,/end/p' over a
  # sentinel-less file prints NOTHING, and grep -q over nothing exits 1,
  # which the tripwire's old shape would have reported as "no re-armed knob
  # found" — a green earned over 0 lines, shipped silently on the exact day
  # the estate file was restructured. Pipe discipline (fix33 Task C): the
  # greps here either read files directly or feed cut, which reads to EOF —
  # no writer in this function ever feeds a grep -q, so the broken-pipe
  # class measured at old line 712 is kept out structurally, not by promise.
  local f=$1 out=$2 nb ne bl el n
  [ -r "$f" ] || { printf 'REFUSAL: %s is absent or unreadable — an unreadable artifact BLOCKS (doctrine 4); it never discharges a check\n' "$f"; return 1; }
  nb=$(grep -cF "$F28_E4B_REGION_BEGIN" "$f")
  [ "$nb" -eq 1 ] || { printf 'REFUSAL: begin sentinel "%s" occurs %s times in %s — it must occur exactly once; a renamed or duplicated begin means the scope is unknowable\n' "$F28_E4B_REGION_BEGIN" "$nb" "$(basename "$f")"; return 1; }
  ne=$(grep -cF "$F28_E4B_REGION_END" "$f")
  [ "$ne" -eq 1 ] || { printf 'REFUSAL: end sentinel "%s" occurs %s times in %s — it must occur exactly once; a renamed or duplicated end means the scope is unknowable\n' "$F28_E4B_REGION_END" "$ne" "$(basename "$f")"; return 1; }
  bl=$(grep -nF "$F28_E4B_REGION_BEGIN" "$f" | cut -d: -f1)
  el=$(grep -nF "$F28_E4B_REGION_END" "$f" | cut -d: -f1)
  [ "$bl" -lt "$el" ] || { printf 'REFUSAL: begin sentinel at line %s is not before the end sentinel at line %s — reordered sentinels bracket nothing trustworthy\n' "$bl" "$el"; return 1; }
  sed -n "${bl},${el}p" "$f" > "$out"
  n=$(grep -c '' "$out")
  [ "$(strip_shell_comments < "$out" | grep -c 'recompute_')" -gt 0 ] || { printf 'REFUSAL: extracted region (%s lines, %s-%s) contains zero recompute_ surface lines — the sentinels bracket the wrong span, and an empty sweep must never read as a clean one\n' "$n" "$bl" "$el"; return 1; }
  return 0
}

f28_p0=$pass; f28_f0=$fail; f28_a0=$abstain
FIX28_LORA=$LORA
FIX28_FULLFT=$FULL
[ -f "$FIX28_LORA" ]   || { no "fix28: launch_g4e4b_lora_1tray.sh not found under $LDIR — substituting /dev/null so the dependent legs read red (an absent subject BLOCKS; it never reads as zero)"; FIX28_LORA=/dev/null; }
[ -f "$FIX28_FULLFT" ] || { no "fix28: launch_g4e4b_fullft_1tray.sh not found under $LDIR — substituting /dev/null so the dependent legs read red (an absent subject BLOCKS; it never reads as zero)"; FIX28_FULLFT=/dev/null; }

# --- syntax: the ground rule (every launcher file passes bash -n) ------------
bash -n "$FIX28_LORA" 2>/dev/null; f28_bnl=$?
bash -n "$FIX28_FULLFT" 2>/dev/null; f28_bnf=$?
# /dev/null parses clean; a missing launcher must not ride the syntax leg to green.
[ "$FIX28_LORA" = /dev/null ]   && f28_bnl=99 || true
[ "$FIX28_FULLFT" = /dev/null ] && f28_bnf=99 || true
if [ "$f28_bnl" -eq 0 ] && [ "$f28_bnf" -eq 0 ]; then
  ok "bash -n clean on both launchers (2/2; syntax was never the defect — this pins the ground rule)"
else
  no "bash -n failed (LoRA rc=$f28_bnl, full-FT rc=$f28_bnf)"
fi
# bash -n MUST_FIRE. Construction note: BSD sed has no '0,/re/' address (that
# is a GNU extension; POSIX ranges begin at line 1), and line 1 of a launcher
# is its shebang, which can never be 'fi' — so '1,/^fi$/' is exactly
# equivalent here and portable. A top-level ^fi$ provably EXISTS in the LoRA
# launcher (the arm-switching block that fix28b replaced still ends at a ^fi$
# per the task's own statement), so an absent probe is a real construction
# failure, not an expected shape — hence verified, loudly, before the parse
# refusal is demanded.
f28_t=$(mktemp "${TMPDIR:-/tmp}/fs-f28-bashn.XXXXXX")
if sed '1,/^fi$/s/^fi$/fi_BROKEN_CONTROL_PROBE/' "$FIX28_LORA" > "$f28_t" 2>/dev/null \
   && grep -q fi_BROKEN_CONTROL_PROBE "$f28_t"; then
  if bash -n "$f28_t" 2>/dev/null; then
    no "bash -n MUST_FIRE: a launcher missing its first top-level 'fi' still parses — bash -n would be wallpaper"
  else
    ok "bash -n MUST_FIRE: corrupting the first top-level 'fi' breaks parsing as required"
  fi
else
  no "bash -n MUST_FIRE UNREACHABLE: the control probe could not be spliced into a copy of the LoRA launcher"
fi
rm -f "$f28_t"

# --- estate-side: ABSTAIN BY NAME until the estate files land ----------------
# fix28b's E1/E2 edits into Megatron-Bridge's recipes/gemma4_vl/ land on the
# GB200 by hand later; on this tree the estate files genuinely do not exist.
# While they are absent, this battery does not run and does not pass: two
# NAMED abstentions (one per estate file) with an explicit 0/2 denominator.
# When enabled by exporting FIX28_ESTATE_GEMMA4_VL and FIX28_ESTATE_INIT at
# the files, the A2 battery below runs REGION-scoped. The history is kept
# because the widening was load-bearing when made: fix30-T1 scoped the
# re-arm grep to the whole FILE deliberately — the snippet's sentinel
# spellings were among the 0/127 lines not in that packet, and inventing
# sentinels the estate patch never wrote would have guaranteed a red nobody
# could clear. That widening carried a stated, named blindness — an honest
# recompute default on some OTHER model in the same file would false-fire
# red, loud, never green — and a pre-authorized repair: restore REGION
# scoping from sentinel text measured off the landed estate patch, never
# relax the grep. The false fire arrived exactly as named (the 31B dense
# recipe's legitimate 'selective' default at gemma4_vl.py:507, measured by
# fix33 on a healthy estate whose own runtime agrees: both E4B entry points
# CONSTRUCTED with every recompute knob None), and the landed patch's
# sentinels are now measured (# --- fix28 E4B region begin/end, :620/:957 of
# the measured file). fix33 therefore executes the original plan: the regex
# stays byte-identical, only the input it reads narrowed, and the extraction
# itself (f28_e4b_region above) fails closed and by name — a restructured
# estate file must read as a named red, never as an empty-region green.
if [[ -n "${FIX28_ESTATE_GEMMA4_VL:-}" && -n "${FIX28_ESTATE_INIT:-}" && -f "${FIX28_ESTATE_GEMMA4_VL:-}" && -f "${FIX28_ESTATE_INIT:-}" ]]; then
  # Extract the E4B region ONCE and read every A2 verdict off that
  # extraction — never off the file at large, and (for the four controls
  # below) never off a re-implementation of the extractor, so the live legs
  # and the flip-verifications read the same bytes by construction and one
  # denominator ($f28_rg_n) covers the whole battery. A refusal is ONE named
  # red, not two contentless ones: a region that cannot be established feeds
  # no legs, the section's own denominator legibly shrinks, and the controls
  # (which construct their copies from the real sentinels) are inert without
  # a real region.
  f28_rt=$(mktemp "${TMPDIR:-/tmp}/fs-f28-e4bregion.XXXXXX")
  if ! f28_rmsg=$(f28_e4b_region "$FIX28_ESTATE_GEMMA4_VL" "$f28_rt"); then
    no "fix28 A2 (estate): region extraction REFUSED — ${f28_rmsg#REFUSAL: } — 0 A2 legs examined; the tripwire fails CLOSED (a scope that cannot be established is a BLOCK, never the empty-region green of doctrine 1), and the four scope controls contribute 0 checks without a real region"
  else
    f28_rg_n=$(grep -c '' "$f28_rt")
    # Leg 1 — the None pin, region-scoped (strengthened, same needle). Under
    # fix30-T1's file scope the 26B recipe's occurrence at :283 sat OUTSIDE
    # the region yet silently vouched for the E4B default; the real E4B
    # satisfier is :807's signature default (region-relative 188), measured
    # in the fix33 packet and verified present against the landed file, so
    # the narrower scope starves nothing.
    f28_req "$f28_rt" "$F28_E4B_NONE_PIN" \
      "fix28 A2 (estate): E4B recompute default pinned to None inside the E4B region ($f28_rg_n lines examined)"
    # Leg 2 — the re-arm tripwire, same region, regex byte-identical to
    # fix30-T1 (only its input narrowed, per the fix33 packet) — and grep
    # now reads the region FILE directly. The fix30-T1 shape piped 52 KB of
    # cat|printf into grep -q, which exits on its first match and closed the
    # pipe mid-write: the 'Broken pipe' noise measured at old line 712, an
    # acoustically-trained-to-ignore diagnostic on a control line. The $(
    # cat) round-trip it replaced also silently stripped trailing newlines —
    # a second, then-harmless defect in the same line. Neither can re-grow
    # here: no leg in this block pipes any writer into grep -q (the only
    # pipes, in the MUST_PASS counts below, feed grep -c, which reads to
    # EOF), and the verdict greps read files.
    if grep -qE "$F28_E4B_REARM_RE" "$f28_rt"; then
      no "fix28 A2 (estate): a recompute knob is re-armed inside the E4B region of $FIX28_ESTATE_GEMMA4_VL ($f28_rg_n lines examined): $(grep -m1 -E "$F28_E4B_REARM_RE" "$f28_rt") — the E4B PLE slice is cleared before backward; ANY recompute trains WITHOUT PLE gradients"
    else
      ok "fix28 A2 (estate): no re-armed recompute knob inside the E4B region ($f28_rg_n lines examined; the 31B :507 'selective' default sits outside the region by design and is rightly unseen here)"
    fi

    # Estate A2 MUST_FIRE (fix30-T1's, re-scoped; the delimiter history it
    # once narrated now lives in the constant block above, with the sed it
    # describes). The sed stays UNADDRESSED, so it rewrites BOTH occurrences
    # of the None pin (measured: 26B :283 and E4B :807) — and :807 is
    # in-region, so a region-scoped tripwire still sees the constructed
    # re-arm (the fix33 packet's second claim; confirmed in the diagnosis).
    # Verification remains VERDICT-FLIP, now by construction: the doctored
    # copy is extracted by the SAME f28_e4b_region the live legs use, and on
    # its region the None pin must be GONE while the re-arm regex MATCHES.
    # Any shortfall — failed sed, refused extraction, surviving pin — is
    # UNREACHABLE: a failed control, never a skip.
    f28_t=$(mktemp "${TMPDIR:-/tmp}/fs-f28-a2rearm.XXXXXX")
    sed "$F28_E4B_REARM_SED" "$FIX28_ESTATE_GEMMA4_VL" > "$f28_t"
    f28_c1r=$(mktemp "${TMPDIR:-/tmp}/fs-f28-c1region.XXXXXX")
    f28_c1msg=$(f28_e4b_region "$f28_t" "$f28_c1r"); f28_c1rc=$?
    if [ $f28_c1rc -eq 0 ] && ! grep -qF "$F28_E4B_NONE_PIN" "$f28_c1r" \
       && grep -qE "$F28_E4B_REARM_RE" "$f28_c1r"; then
      ok "MUST_FIRE (fix28 A2 estate): the constructed re-arm flips the region-scoped verdict on the copy (in-region pin erased, in-region re-arm matched — the same extraction the live legs read)"
    else
      no "MUST_FIRE (fix28 A2 estate) UNREACHABLE: the constructed re-arm did not flip the region-scoped tripwire (copy extraction rc=$f28_c1rc ${f28_c1msg:+— $f28_c1msg}) — unproven detector"
    fi
    rm -f "$f28_t" "$f28_c1r"

    # MUST_FIRE (scope): a re-arm constructed INSIDE the region must go red
    # in exactly the scope the live legs read. The sed's address range is
    # the live sentinel pair, so it flips the E4B signature default at :807
    # while the file's other recompute pins (26B :283, 31B :507) stay
    # untouched — out-of-region, rightly invisible to the verdict.
    # NON-VACUITY IS COUNT-PROVEN, not assumed: the healthy region carries
    # $f28_hit0 fired-pin lines and the doctored copy's region must carry
    # EXACTLY ONE MORE (today 0 -> 1), so a sed that silently matched
    # nothing, an address range that missed the region, and an extraction
    # that read the wrong span all fail this leg instead of minting it.
    f28_t=$(mktemp "${TMPDIR:-/tmp}/fs-f28-scopefire.XXXXXX")
    sed "/$F28_E4B_REGION_BEGIN/,/$F28_E4B_REGION_END/ $F28_E4B_REARM_SED" "$FIX28_ESTATE_GEMMA4_VL" > "$f28_t"
    f28_c2r=$(mktemp "${TMPDIR:-/tmp}/fs-f28-c2region.XXXXXX")
    f28_hit0=$(grep -cF "$F28_E4B_FIRED_PIN" "$f28_rt" || true)
    f28_c2msg=$(f28_e4b_region "$f28_t" "$f28_c2r"); f28_c2rc=$?
    f28_hit1=-1
    [ $f28_c2rc -eq 0 ] && f28_hit1=$(grep -cF "$F28_E4B_FIRED_PIN" "$f28_c2r" || true)
    if [ $f28_c2rc -eq 0 ] && [ "$f28_hit1" -eq "$((f28_hit0 + 1))" ] \
       && grep -qE "$F28_E4B_REARM_RE" "$f28_c2r"; then
      ok "MUST_FIRE (scope): an in-region-only re-arm goes red in the region scope (fired-pin lines in-region $f28_hit0 -> $f28_hit1; :283 and :507 untouched) — the live legs' green is therefore informative, not vacuous"
    else
      no "MUST_FIRE (scope) UNREACHABLE: the in-region-only re-arm did not construct or did not fire (copy extraction rc=$f28_c2rc ${f28_c2msg:+— $f28_c2msg}; in-region fired-pin count $f28_hit0 -> $f28_hit1) — region scoping is unproven in the fire direction"
    fi
    rm -f "$f28_t" "$f28_c2r"

    # MUST_PASS (scope, the anti-overreach half — without it, region scoping
    # is indistinguishable from 'the red went away'): a re-arm constructed
    # OUTSIDE the region — the real 31B :507 situation, replayed at the
    # 26B's :283 — must leave the region verdict GREEN even though the
    # byte-identical regex provably matches the same doctored FILE.
    # Construction is count-proven in the mirror direction: the copy's
    # PRE-region span (line 1 through the begin sentinel) must carry exactly
    # one more fired-pin line than the real file's (today 1 -> 2: the native
    # :507 plus the constructed :283), which a do-nothing sed cannot mint —
    # and crucially the delta is taken against the REAL file, so the native
    # :507 cannot mask a collapsed construction. The copy's region must then
    # carry NO regex hit at all.
    f28_pre0=$(sed -n "1,/$F28_E4B_REGION_BEGIN/p" "$FIX28_ESTATE_GEMMA4_VL" | grep -cF "$F28_E4B_FIRED_PIN" || true)
    f28_t=$(mktemp "${TMPDIR:-/tmp}/fs-f28-scopepass.XXXXXX")
    sed "1,/$F28_E4B_REGION_BEGIN/ $F28_E4B_REARM_SED" "$FIX28_ESTATE_GEMMA4_VL" > "$f28_t"
    f28_pre1=$(sed -n "1,/$F28_E4B_REGION_BEGIN/p" "$f28_t" | grep -cF "$F28_E4B_FIRED_PIN" || true)
    f28_c3r=$(mktemp "${TMPDIR:-/tmp}/fs-f28-c3region.XXXXXX")
    f28_c3msg=$(f28_e4b_region "$f28_t" "$f28_c3r"); f28_c3rc=$?
    if [ $f28_c3rc -eq 0 ] && [ "$f28_pre1" -eq "$((f28_pre0 + 1))" ] \
       && grep -qE "$F28_E4B_REARM_RE" "$f28_t" \
       && ! grep -qE "$F28_E4B_REARM_RE" "$f28_c3r"; then
      ok "MUST_PASS (scope): an out-of-region-only re-arm stays green in the region scope while the same regex provably matches the same copy at file scope (pre-region fired-pin lines $f28_pre0 -> $f28_pre1) — the verdict moved because the scope is right, not because the defect was renamed away"
    else
      no "MUST_PASS (scope) FAILED: the out-of-region-only copy was not cleanly green under construction-proof (copy extraction rc=$f28_c3rc ${f28_c3msg:+— $f28_c3msg}; pre-region fired-pin $f28_pre0 -> $f28_pre1) — either the construction collapsed (unproven control) or the tripwire still reads outside its region (the fix33 defect persists)"
    fi
    rm -f "$f28_t" "$f28_c3r"

    # MUST_FIRE (vacuity — doctrine 1 applied to this patch's own new
    # machinery): delete the begin sentinel from a copy and the extraction
    # MUST refuse BY NAME, never produce a green. The failure shape being
    # pinned is specific: a naive between-sentinels sed over a sentinel-less
    # file prints NOTHING, and grep -q over nothing exits 1, which the old
    # tripwire shape would have read as 'no re-armed knob found' — a pass
    # earned over 0 lines. The copy is proven construction-first (the begin
    # sentinel is GONE from it), then the extractor on it must carry a
    # nonzero rc and name the begin sentinel. On the healthy file the same
    # extractor succeeded (the legs above), so the flip is real:extracts <->
    # copy:refuses — and when the REAL file is the restructured one, the
    # refusal arm above is this same behavior firing unconstructed.
    f28_t=$(mktemp "${TMPDIR:-/tmp}/fs-f28-vacuity.XXXXXX")
    sed "/$F28_E4B_REGION_BEGIN/d" "$FIX28_ESTATE_GEMMA4_VL" > "$f28_t"
    f28_c4r=$(mktemp "${TMPDIR:-/tmp}/fs-f28-c4region.XXXXXX")
    f28_c4msg=$(f28_e4b_region "$f28_t" "$f28_c4r"); f28_c4rc=$?
    if [ $f28_c4rc -ne 0 ] && ! grep -qF "$F28_E4B_REGION_BEGIN" "$f28_t" \
       && grep -qF 'begin sentinel' <<<"$f28_c4msg"; then
      ok "MUST_FIRE (vacuity): a copy with the begin sentinel deleted is REFUSED BY NAME, not read as green ('$f28_c4msg') — the empty-region green the fix33 packet names is unrepresentable"
    else
      no "MUST_FIRE (vacuity) FAILED: the begin-sentinel-less copy was not refused by name (extraction rc=$f28_c4rc, msg='${f28_c4msg:-<none>}') — a restructured estate patch would one day read an empty region as ALL CLEAR"
    fi
    rm -f "$f28_t" "$f28_c4r"
  fi
  rm -f "$f28_rt"
else
  abstain=$((abstain + 2))
  echo "  fix28 estate-side controls: ABSTAIN — 0/2 estate files examined (export FIX28_ESTATE_GEMMA4_VL and FIX28_ESTATE_INIT to enable; the region-scoped A2 battery above — 2 legs plus 4 flip-verified controls — gates there). Recorded as named abstentions, adding 0 to the pass count."
fi

# --- the snippet's launcher-side battery: named abstention, not a paraphrase -
# fix28b's launcher-side needle legs (its 1a/1b-style greps against the
# launchers as patched) address launcher text that is 0/2 in this packet,
# carried in a snippet that is 0/127 in this packet. They are NOT minted
# here: a needle imagined against an unread file is a detector with no
# MUST_PASS and no truthful basis — unproven, and unproven is worth nothing.
# They integrate mechanically the moment the snippet body is supplied: paste
# its legs verbatim under the drop-in helpers above (f28/f28_bad are already
# wired to this harness's accounting; the snippet's own fallback plan was a
# one-line-per-helper swap, and these helpers ARE that swap).
abstain=$((abstain + 1))
echo "  fix28 launcher-side needle battery: ABSTAIN — 0/127 snippet lines and 0/2 launcher files available to the fix30 packet; refusing to paraphrase unread controls. Transplant path documented above. Recorded as a named abstention, adding 0 to the pass count."

echo "  fix28 section: $((pass - f28_p0)) of $((pass - f28_p0 + fail - f28_f0)) constructed checks green; $((abstain - f28_a0)) named abstentions (0 added to pass/fail)"

echo
# fix26 (item C) — every run prints WHERE its greens were earned. The
# hostname/enroot/nvidia-smi stubs are by design on EVERY platform (the
# header of this section: nothing touches the real tray), so the whole
# certification question reduces to one fact: did the sandbox stand on
# the production stat semantics directly, or on a translation? Only the
# former may ever be quoted as certification of the off-Slurm launch
# path. The "controls:" line itself is byte-identical to before so any
# existing log parser is untouched; this line adds, never replaces.
fs_platform=$(uname -s)
fs_native_stat=0
case "${FS_STAT_WIRING:-}" in
  native*)  fs_stat_desc="native GNU stat — the same semantics the Linux production path runs"; fs_native_stat=1 ;;
  sandbox*) fs_stat_desc="sandbox stat ADAPTER translating the backend's GNU -c forms to BSD -f" ;;
  *)        fs_stat_desc="stat wiring UNKNOWN — the backend block never ran, and its controls are absent from the tally" ;;
esac
if [ "$fs_platform" = Linux ] && [ "$fs_native_stat" = 1 ]; then
  echo "platform-report: LINUX, $fs_stat_desc. This green certifies the off-Slurm guard chain as it will run on <compute-node> (hostname/enroot/nvidia-smi remain deliberately stubbed: tray state is asserted, not measured)."
else
  echo "platform-report: NON-CERTIFYING RUN — platform '$fs_platform', stat: $fs_stat_desc. These greens certify the guard LOGIC ONLY, under stubs and an adapted stat. This output must never be quoted as certification of a Linux launch path; measure on <compute-node> before trusting a launch."
fi
# Abstentions are reported NEXT TO — never inside — the frozen "controls:"
# tally, which stays byte-identical for any existing log parser (the fix26
# rule above: this line adds, never replaces). An abstention is a first-class
# outcome with its own denominator (doctrine 5), and a green count may never
# include checks that did not run (doctrines 1/2).
echo "== fix35: the artifact adjudicator is wired, three-valued, and cannot ride '|| true' =="
# Fail-before accounting (pre-patch tree): every leg below except the
# '|| true'-absence guard FAILS there (the symbols/needles do not exist; the
# MUST_FIRE extractions come back empty and report UNREACHABLE, which is a
# red control per this file's own rule). The absence guard is green on both
# trees BY CONSTRUCTION — it guards the new wiring against a future
# laundering edit; it is stated here so nobody quotes it as a fail-before leg.
lora_f35=$(pos_count fs_live_save_gate "$LORA")
full_f35=$(pos_count fs_live_save_gate "$FULL")
if [ "$lora_f35" -eq 1 ] && [ "$full_f35" -eq 2 ]; then
  ok "live_save_gate call sites: LoRA=$lora_f35 (post-run) full-FT=$full_f35 (watcher first-save + epilogue final)"
else no "live_save_gate call sites LoRA=$lora_f35 full-FT=$full_f35 — expected 1+2; the adjudicator is (again) unreachable"; fi

f35_lt=$(strip_shell_comments < "$LORA" | grep 'live_save_gate' | grep -c '|| true' || true)
f35_ft=$(strip_shell_comments < "$FULL" | grep 'live_save_gate' | grep -c '|| true' || true)
if [ "$f35_lt" -eq 0 ] && [ "$f35_ft" -eq 0 ]; then
  ok "no live_save_gate site rides '|| true' (LoRA=$f35_lt full-FT=$f35_ft) — rc is captured and TESTED, never swallowed"
else no "a live_save_gate invocation is '|| true'-laundered (LoRA=$f35_lt full-FT=$full_ft) — the founding bug re-spelled"; fi

for f in "$LORA" "$FULL"; do
  if grep -qF 'live_save_gate.py not found' "$f"; then
    ok "$(basename "$f"): launch-time armament refusal present (partial checkout cannot launch)"
  else no "$(basename "$f"): no launch-time refusal for a missing live_save_gate.py — a gate that cannot run must refuse at t=0"; fi
done

if grep -qF 'GATE_JOB_RC=44' "$FULL" && grep -qF 'GATE_JOB_RC=45' "$FULL" && grep -qF 'GATE_JOB_RC=46' "$FULL"; then
  ok "full-FT maps the three-way contract to distinct job rcs (44=BLOCKED, 45=UNMEASURED, 46=infra) — 0/1/3 are not laundered into one"
else no "full-FT three-way rc mapping incomplete (44/45/46) — BLOCKING vs UNMEASURED vs CLEAR must stay distinguishable"; fi
if grep -qF 'FS_ART_GATE_RC=91' "$LORA" && grep -qF 'FS_ART_GATE_RC=92' "$LORA" && grep -qF 'TODAY THE EXPECTED' "$LORA"; then
  ok "LoRA maps BLOCKED->91, infra->92, and names the exit-3 adapter-prefix abstention as today's expected state with its skipped denominator"
else no "LoRA three-way mapping (91/92/expected-3) incomplete — the expected abstention must be stated, not silent, and must never read as a pass"; fi

if grep -qF 'report-first-save.json' "$FULL" && grep -qF 'report-final.json' "$FULL" && grep -qF 'report-lora.json' "$LORA"; then
  ok "gate reports land on disk where each epilogue/tripwire names them (3/3 report paths pinned)"
else no "a gate invocation lacks its on-disk --json report path (first-save/final/lora: 3 expected)"; fi

if grep -qF -- '--fqn-map' "$FULL" && grep -qF 'attempt-*.json' "$FULL"; then
  ok "full-FT materializes --fqn-map from the emitter's own attempt-*.json record (declared_fqns finally have a reader; without it every DCP first save exits 1, healthy or not)"
else no "full-FT passes no --fqn-map producer — on estate DCP saves the gate then VACUOUS-blocks unconditionally (a permanent red that unmints this wiring)"; fi

if grep -qF '(d) FIRST-SAVE ARTIFACT' "$FULL" && grep -qF 'POST-RUN ONLY' "$LORA"; then
  ok "sibling asymmetry stated in both launchers' own text (full-FT live tripwire (d) vs LoRA post-run only)"
else no "a launcher carries a guarantee its text does not state — silent sibling divergence"; fi

# MUST_FIRE, constructed (doctrine 3): sed-extract the REAL rc-mapping
# functions from both launchers (no paraphrase — same rule as the
# EXTRA_OVERRIDES and LoRA-arm extractions above) and feed them constructed
# verdicts. rc=1 must stop the run (nonzero job rc AND stderr saying so);
# rc=3 must never read as a pass and never as a measured block; rc=0 must
# pass through as 0. Empty extraction => UNREACHABLE => a red control.
# fix45 STRENGTHENING (house rule; the two legs this replaces are quoted in
# the patch's 'Existing tests'): their intent — a constructed UNMEASURED
# marks 45-not-clear and a constructed CLEAR stays 0, no laundering in
# either direction — is preserved verbatim in force as legs e0/e1/e3 below,
# now driven by fixtures that satisfy the EVIDENCE CONTRACT the tool has
# written since fix44: an exit 3 marks 45 only beside the tool's refusal
# record carrying refusal_class (fix44 / #77-B3), and an exit 0 stays 0
# only when corroborated by the gate's own printed verdict plus an on-disk
# report. The bare-rc fixtures the old legs drove become legs e2/e4 and now
# demand 46: they construct exactly the states (recordless 3, evidence-less
# 0) the old legs could not distinguish from a recorded 3 and a real CLEAR.
# Fail-before accounting on the pre-fix45 tree: e0 green on both trees
# (unchanged demand, disclosed, never quoted as fail-before); e1-e4 RED
# there — the old mapper answers every 3 with 45 and every 0 with 0
# unconditionally, which was itself the multiplexed decode #77-B2 indicted.
# fix45-A2 addendum (dead-control audit of THESE fixtures, decided: KEEP,
# with the label corrected): nothing in e0-e4 depends on the refuted
# host-incapacity diagnosis. The refusal string in cap-unreadable is the
# tool's REAL refusal vocabulary — gate arm A on <compute-node> measured the tool
# emitting exactly it (under the launcher's own env; the older 'host has no
# torch' reading of that string is refuted and preserved on record in the
# launcher comment). These fixtures exercise the refusal-RECORD contract —
# a record parses identically whatever made the DCP stack unimportable —
# so they fire for the right reason under the corrected diagnosis.
f35_full_fn=$(sed -n '/^fs_gate_verdict_to_rc() {/,/^}/p' "$FULL")
f35_full_rc=$(sed -n '/^fs_gate_refusal_class() {/,/^}/p' "$FULL")
if [ -n "$f35_full_fn" ] && [ -n "$f35_full_rc" ]; then
  f45_msim=$(mktemp -d "${TMPDIR:-/tmp}/fs-f45-mapper.XXXXXX" 2>/dev/null) || f45_msim=""
  [ -n "$f45_msim" ] || { f45_msim="${TMPDIR:-/tmp}/fs-f45-mapper.$$"; mkdir -p "$f45_msim" 2>/dev/null || f45_msim=""; }
  f45_map() { # $1=gate rc $2=report path $3=capture path -> subshell: the REAL extracted helper+mapper, plus telemetry
    ( eval "$f35_full_rc"; eval "$f35_full_fn"
      fs_gate_verdict_to_rc "$1" "fix45-fixture" "$2" "$3"
      echo "JOBRC=$GATE_JOB_RC" ) 2>&1
  }
  # Fixtures. rep-refusal.json is the byte shape the tool writes on every
  # exit 3 (fix44 / #77-B3); cap-unreadable carries refusal text the tool
  # measurably emits (<compute-node> gate arm A; the same string family as LoRA
  # jobs 1787517960364/1787518637847) at fixture paths.
  cat > "$f45_msim/rep-refusal.json" <<'F45CAP'
{
  "checkpoint": "/fixture/iter_0000100",
  "event": "save",
  "run_kind": "full",
  "verdict": "UNMEASURED",
  "exit_code": 3,
  "refusal": "checkpoint unreadable: /fixture/iter_0000100: torch.distributed.checkpoint is unavailable; cannot read DCP",
  "refusal_class": "checkpoint_unreadable",
  "gates_exercised": "0 of 3",
  "controls_exercised": "0 of 3"
}
F45CAP
  printf 'live_gate could not measure: checkpoint unreadable: /fixture/iter_0000100: torch.distributed.checkpoint is unavailable; cannot read DCP\n' > "$f45_msim/cap-unreadable"
  printf 'noise\nLIVE GATE VERDICT: CLEAR (exit 0)\n' > "$f45_msim/cap-clear"
  printf '{}\n' > "$f45_msim/rep-ok.json"

  out=$(f45_map 1 /dev/null /dev/null)
  if printf '%s' "$out" | grep -q 'JOBRC=44' && printf '%s' "$out" | grep -q 'BLOCKED'; then
    ok "MUST_FIRE full-FT e0: a constructed BLOCKED verdict marks the run 44, loudly (the pre-fix45 demand, kept verbatim in force)"
  else no "MUST_FIRE full-FT e0: blocking verdict did not stop the run (output: $out)"; fi

  out=$(f45_map 3 "$f45_msim/rep-refusal.json" "$f45_msim/cap-unreadable")
  if printf '%s' "$out" | grep -q 'JOBRC=45' && printf '%s' "$out" | grep -q 'UNMEASURED' \
     && printf '%s' "$out" | grep -q 'refusal_class=checkpoint_unreadable'; then
    ok "MUST_FIRE full-FT e1: a constructed UNMEASURED with the tool's OWN refusal record marks 45-not-clear WITH the cause named from the record (never narrated) — the pre-fix45 intent, evidence-keyed; on this path NO exit-3 member is an rc-0 abstention (no chosen open knob exists)"
  else no "MUST_FIRE full-FT e1: the evidence-keyed 3→45 decode broke (output: $out)"; fi

  out=$(f45_map 3 "$f45_msim/rep-ABSENT.json" "$f45_msim/cap-unreadable")
  if printf '%s' "$out" | grep -q 'JOBRC=46' && printf '%s' "$out" | grep -q 'claim-vs-disk'; then
    ok "MUST_FIRE full-FT e2: an exit 3 whose refusal record is ABSENT marks 46 — indicted as the #77-B3 claim-vs-disk gap (measured live on the <compute-node> run: two adjudications, rc 3, report ABSENT both times), never laundered into a plain 45: a 3 with no evidence is infrastructure, not a measurement of the tool's own inability"
  else no "MUST_FIRE full-FT e2: a recordless 3 was not indicted at 46 (output: $out)"; fi

  out=$(f45_map 0 "$f45_msim/rep-ok.json" "$f45_msim/cap-clear")
  if printf '%s' "$out" | grep -q 'JOBRC=0' && printf '%s' "$out" | grep -q 'corroborated'; then
    ok "MUST_PASS full-FT e3: a CLEAR corroborated by the gate's own printed verdict AND its on-disk report stays 0 (the pre-fix45 'no laundering in either direction' demand, now with fix44's corroboration on this side too)"
  else no "MUST_PASS full-FT e3: a corroborated CLEAR did not stay 0 (output: $out)"; fi

  out=$(f45_map 0 /dev/null /dev/null)
  if printf '%s' "$out" | grep -q 'JOBRC=46' && printf '%s' "$out" | grep -q 'OVERCLAIM'; then
    ok "MUST_FIRE full-FT e4: a bare rc 0 with neither printed verdict nor report marks 46 OVERCLAIM — the founding bug's shape, refused on this path exactly as on the LoRA sibling"
  else no "MUST_FIRE full-FT e4: an uncorroborated 0 was not refused (output: $out) — rc 0 alone is again minting a pass"; fi
  [ -n "$f45_msim" ] && rm -rf "$f45_msim" || true
else
  no "MUST_FIRE UNREACHABLE: fs_gate_verdict_to_rc/fs_gate_refusal_class not extractable from the full-FT launcher — the 5 constructed-verdict legs above are unproven"
fi
# fix44 STRENGTHENING (house rule): the two legs this replaces constructed
# rc=1 and a bare rc=3 against the OLD 3-arg mapper. The rc=3 construction
# pinned exactly the multiplexed decode that IS defect #77-B2 — its intent
# ("the expected abstention must be stated, not silent, and must never read
# as a pass") is preserved, verbatim in force, as leg m2 below, now keyed on
# the gate's OWN evidence; the class decode, the claim-vs-disk indictment,
# and the CLEAR corroboration are new constructed demands. Fail-before
# accounting on tonight's tree: m1 and m2 are green there by construction
# (unchanged demands, disclosed, never quoted as fail-before legs); m3, m4,
# m5, m6 are RED there (the old mapper scores every rc=3 as the expected
# abstention and never corroborates CLEAR), which pins the multiplexed
# defect itself.
f35_lora_fn=$(sed -n '/^fs_lora_gate_verdict_to_rc() {/,/^}/p' "$LORA")
if [ -n "$f35_lora_fn" ]; then
  f44_msim=$(mktemp -d "${TMPDIR:-/tmp}/fs-f44-mapper.XXXXXX" 2>/dev/null) || f44_msim=""
  [ -n "$f44_msim" ] || { f44_msim="${TMPDIR:-/tmp}/fs-f44-mapper.$$"; mkdir -p "$f44_msim" 2>/dev/null || f44_msim=""; }
  f44_map() { # $1=gate rc $2=report path $3=capture path -> subshell: the REAL extracted mapper, plus its telemetry lines
    ( eval "$f35_lora_fn"
      fs_lora_gate_verdict_to_rc "$1" "fix44-fixture" "$2" "$3"
      echo "ARTRC=$FS_ART_GATE_RC"; echo "STATE=$FS_ART_GATE_STATE" ) 2>&1
  }
  # Fixtures. cap-prefix carries the tool's own refusal words; rep-prefix.json
  # is the byte shape the patched tool now writes on every exit-3 (the
  # '"refusal_class": "adapter_prefix_unpinned"' token is the mapper's
  # corroboration needle, shared with the tool); cap-unreadable reproduces
  # the MEASURED cause text of jobs 1787517960364/1787518637847
  # (torch-less host python reading a healthy DCP, #77-B1) at fixture paths.
  cat > "$f44_msim/cap-prefix" <<'F44CAP'
launcher banner noise
live_gate could not measure: --adapter-prefix was not pinned for a lora adjudication (exit 3 -- a refused measurement, not a checkpoint verdict): whether this estate's adapter saves carry a constant leading segment cannot be established from any independent source this tool may read
F44CAP
  cat > "$f44_msim/rep-prefix.json" <<'F44CAP'
{
  "verdict": "UNMEASURED",
  "exit_code": 3,
  "refusal": "--adapter-prefix was not pinned for a lora adjudication ...",
  "refusal_class": "adapter_prefix_unpinned",
  "gates_exercised": "0 of 3",
  "controls_exercised": "0 of 3"
}
F44CAP
  cat > "$f44_msim/cap-unreadable" <<'F44CAP'
live_gate could not measure: checkpoint unreadable: /fixture/results/checkpoints/iter_0000020: torch.distributed.checkpoint is unavailable; cannot read DCP (path='/fixture/results/checkpoints/iter_0000020')
F44CAP
  cat > "$f44_msim/cap-unknown" <<'F44CAP'
live_gate could not measure: --train-config not found: /fixture/resolved-train-config.json
F44CAP
  printf 'noise\nLIVE GATE VERDICT: CLEAR (exit 0)\n' > "$f44_msim/cap-clear"
  printf '{}\n' > "$f44_msim/rep-ok.json"

  out=$(f44_map 1 "$f44_msim/rep-blocked.json" "$f44_msim/cap-unknown")
  if printf '%s' "$out" | grep -q 'ARTRC=91' && printf '%s' "$out" | grep -q 'BLOCKED'; then
    ok "MUST_FIRE LoRA m1: a constructed BLOCKED verdict stops the run/chain on 91, loudly (the pre-fix44 demand, kept verbatim in force)"
  else no "MUST_FIRE LoRA m1: blocking verdict did not stop the run (output: $out)"; fi

  out=$(f44_map 3 "$f44_msim/rep-prefix.json" "$f44_msim/cap-prefix")
  if printf '%s' "$out" | grep -q 'ARTRC=0' && printf '%s' "$out" | grep -q 'EXPECTED' \
     && printf '%s' "$out" | grep -q '0 of its 3 gates' && printf '%s' "$out" | grep -q 'CONFIRMED'; then
    ok "MUST_PASS LoRA m2: the CONFIRMED adapter-prefix abstention lands rc 0, stated, with its 0-of-3-gates denominator (the pre-fix44 intent, now evidence-keyed) — neither pass nor kill, and never on a bare 3"
  else no "MUST_PASS LoRA m2: the calibrated prefix abstention lost its statement, its denominator, or its rc-0 calibration (output: $out)"; fi

  out=$(f44_map 3 "$f44_msim/rep-missing.json" "$f44_msim/cap-unreadable")
  if printf '%s' "$out" | grep -q 'ARTRC=92' && printf '%s' "$out" | grep -q 'torch.distributed.checkpoint is unavailable' \
     && printf '%s' "$out" | grep -q 'rc-92'; then
    ok "MUST_FIRE LoRA m3: the MEASURED unreadable-DCP cause (torch-less host python, both PROBE runs) decodes to rc-92 with the gate's own words quoted — the multiplexed 'expected abstention' decode is dead"
  else no "MUST_FIRE LoRA m3: an unreadable-artifact 3 was not rc-92 with the real cause quoted (output: $out) — the multiplexed decode persists (#77-B2)"; fi

  out=$(f44_map 3 "$f44_msim/rep-missing.json" "$f44_msim/cap-unknown")
  if printf '%s' "$out" | grep -q 'ARTRC=92' && printf '%s' "$out" | grep -q -- '--train-config not found'; then
    ok "MUST_FIRE LoRA m4: an exit-3 cause the calibration does NOT name (a missing train config) rides the rc-92 class with the cause quoted — unknown members of the class never inherit the calibrated arm"
  else no "MUST_FIRE LoRA m4: an unnamed exit-3 cause was not rc-92 (output: $out) — the calibrated abstention would absorb causes it was never calibrated for"; fi

  out=$(f44_map 3 "$f44_msim/rep-ABSENT.json" "$f44_msim/cap-prefix")
  if printf '%s' "$out" | grep -q 'ARTRC=92' && printf '%s' "$out" | grep -q 'must have written'; then
    ok "MUST_FIRE LoRA m5: the gate claims the prefix abstention but the refusal record it must have written is ABSENT -> rc-92 (#77-B3 indicted, not narrated — the claim never again outruns the disk)"
  else no "MUST_FIRE LoRA m5: a claimed-but-absent refusal record was not indicted at rc-92 (output: $out) — the #77-B3 claim-vs-disk gap persists"; fi

  out=$(f44_map 0 "$f44_msim/rep-ok.json" "$f44_msim/cap-clear")
  if printf '%s' "$out" | grep -q 'ARTRC=0$' && printf '%s' "$out" | grep -q 'corroborated'; then
    ok "MUST_PASS LoRA m6: a CLEAR corroborated by the gate's own printed verdict AND its on-disk report stays rc 0 — corroboration taxes a healthy run nothing"
  else no "MUST_PASS LoRA m6: a corroborated CLEAR did not stay rc 0 (output: $out) — the corroboration demand broke the happy path"; fi
  [ -n "$f44_msim" ] && rm -rf "$f44_msim" || true
else
  no "MUST_FIRE UNREACHABLE: fs_lora_gate_verdict_to_rc not extractable from the LoRA launcher — the 6 constructed-verdict legs above are unproven"
fi

echo "== fix39: the census oracle is the shipped matcher itself, and no shipped target wears the measured-unmatchable shape =="
# Fail-before accounting, in one line: on the pre-fix39 tree every DETECTION
# leg in this section is red — the dense/expert strings were the
# measured-unmatchable dotted-without-'*' shape (grep 42, ModuleMatcher 0,
# population 1556), there was 0/1 census probes on disk, step (5) scored
# targets with grep -cF over a dump using argparse-incompatible flags, and G2
# read the same laundered oracle back. The two MUST_FIRE legs below are green
# on both trees BY CONSTRUCTION — they prove the two detectors fire on a
# constructed regression; stated here so nobody quotes them as fail-before
# legs (the harness's own fix35 absence-guard precedent, restated).
F39_LORA=$LORA
F39_PROBE=$LDIR/lora_target_census.py

# --- helpers, ABOVE their first call site (this file was broken exactly once
# by a helper defined below its first caller; bash resolves names at call time)
f39_target_rows() { # $1=launcher -> "LIST <pattern>", one line per shipped target, mined from the REAL assignment lines (no paraphrase — the rule this file applies to every extraction)
  sed -n 's/^LORA_TARGETS_BASE="\(.*\)"$/BASE \1/p; s/^LORA_TARGETS_EXPERT="\(.*\)"$/EXPERT \1/p' "$1" \
  | while IFS=' ' read -r f39_list f39_csv; do
      for f39_t in ${f39_csv//,/ }; do printf '%s %s\n' "$f39_list" "$f39_t"; done
    done
}
f39_unmatchable() { # $1=launcher -> the rows of the measured-unmatchable shape:
                    # DOTTED and star-free. peft/utils.py:208 anchors the whole
                    # pattern at both ends, so such a pattern matches only an
                    # FQN byte-equal to itself — measured 0/1556 on the live tree.
  f39_target_rows "$1" | awk '$2 ~ /\./ && $2 !~ /\*/'
}
f39_r5_ok() { # $1=launcher -> rc 0 iff the step-(5) region invokes the real-oracle
              # probe with the probe's REAL flags and the laundered census is gone.
  local f39_r5
  f39_r5=$(sed -n '/^# (5) /,/^# (6) /p' "$1" | strip_shell_comments)
  printf '%s\n' "$f39_r5" | grep -qF 'lora_target_census.py' \
    && printf '%s\n' "$f39_r5" | grep -qF 'torchrun --nnodes=1' \
    && printf '%s\n' "$f39_r5" | grep -qF -- '--hf_model_path' \
    && printf '%s\n' "$f39_r5" | grep -qF -- '--targets' \
    && ! printf '%s\n' "$f39_r5" | grep -qF -- '--hf_path ' \
    && ! printf '%s\n' "$f39_r5" | grep -qF -- '--recipe' \
    && ! printf '%s\n' "$f39_r5" | grep -qF 'grep -cF'
}

f39_rows=$(f39_target_rows "$F39_LORA")
f39_rows_n=$(printf '%s\n' "$f39_rows" | grep -c .)
f39_bad=$(f39_unmatchable "$F39_LORA")
if [ -n "$f39_rows" ] && [ "$f39_rows_n" -gt 0 ] && [ -z "$f39_bad" ]; then
  ok "fix39: ${f39_rows_n} shipped targets examined, none of the measured-unmatchable shape (dotted without '*'; ModuleMatcher's anchored wildcard can reach each)"
else
  no "fix39: shipped targets of the measured-unmatchable shape present (dotted, no '*'; 0-of-1556 on the real matcher for exactly this shape): ${f39_bad:-<extraction found ${f39_rows_n:-0} rows — the LORA_TARGETS_* lines themselves are missing, which is also red>}"
fi

# MUST_FIRE for the shape leg (doctrine 3): re-insert the measured-broken
# spelling 'mlp.linear_fc1' in place of its repaired wildcard on a temp COPY
# and demand the predicate surfaces exactly that row. The grep for the
# constructed row IS the construction proof (needle-in); the live leg above
# is the needle-out half, since that row is absent from the healthy file.
f39_t=$(mktemp "${TMPDIR:-/tmp}/fs-f39-shape.XXXXXX") \
  && sed 's/\*\.mlp\.mlp\.linear_fc1/mlp.linear_fc1/' "$F39_LORA" > "$f39_t"
f39_s=$?
f39_mf_out=""
[ "$f39_s" -eq 0 ] && f39_mf_out=$(f39_unmatchable "$f39_t")
[ -n "${f39_t:-}" ] && rm -f "$f39_t" || true
if [ "$f39_s" -eq 0 ] && printf '%s\n' "$f39_mf_out" | grep -qF 'BASE mlp.linear_fc1'; then
  ok "MUST_FIRE fix39-shape: re-inserting the measured-broken 'mlp.linear_fc1' on a copy turns the shape leg red on exactly the row it must (construction verified: the row exists in the firing input)"
else
  no "MUST_FIRE UNREACHABLE (fix39-shape): the pre-fix spelling could not be re-constructed on a copy (sed rc=$f39_s; predicate output: '${f39_mf_out:-<empty>}') — the shape leg above is an unproven detector"
fi

# The probe itself: presence + offline-verifiable syntax. ast.parse (not
# py_compile) so no __pycache__ debris lands next to the launcher. A missing
# python3 is RED, not skipped (doctrine 4: a probe that cannot even be
# syntax-verified offline cannot certify a launch; megatron.bridge itself is
# genuinely importable only in-container, which is stated, not hidden).
if [ -f "$F39_PROBE" ] && command -v python3 >/dev/null 2>&1 \
   && python3 -c "import ast, pathlib, sys; ast.parse(pathlib.Path(sys.argv[1]).read_text(encoding='utf-8'))" "$F39_PROBE" 2>/dev/null; then
  ok "fix39: real-oracle census probe ships beside the launcher and parses clean (1/1 probe files: $F39_PROBE; megatron.bridge import is exercised only in-container, by design)"
else
  no "fix39: census probe missing/unparseable at $F39_PROBE (present: $([ -f "$F39_PROBE" ] && echo yes || echo NO)) — the launcher's step (5) has no real oracle to invoke"
fi

# Step-(5) region wiring: the real probe, the measured torchrun rendezvous,
# the probe's REAL flags — and the pre-fix failure occupants (grep -cF census,
# --hf_path, --recipe) absent from the region, where 'region' is the
# comment-stripped text between the launcher's own (5)/(6) step markers.
if f39_r5_ok "$F39_LORA"; then
  ok "fix39: step (5) invokes lora_target_census.py under torchrun with the tool's real flags (--hf_model_path/--ep/--targets); the grep census and the argparse-incompatible flags are gone from the region"
else
  no "fix39: step (5) is not wired to the real-oracle probe with its measured invocation, or the old grep census / --hf_path / --recipe survived in-region"
fi

# MUST_FIRE for the region leg: rebuild the argparse regression on a temp
# copy (--hf_model_path -> --hf_path) and demand two things: the defect shape
# provably exists in the copy's region (construction proof), and the SAME
# predicate the live leg runs reports the copy NOT ok. Whole-file sed is safe
# here: '--hf_model_path' occurs only inside the step-(5) region post-patch
# (the training command's '--hf_path' for run_recipe.py is untouched, as the
# sed pattern cannot match it).
f39_t=$(mktemp "${TMPDIR:-/tmp}/fs-f39-region.XXXXXX") \
  && sed 's/--hf_model_path/--hf_path/g' "$F39_LORA" > "$f39_t"
f39_s=$?
f39_r5_copy=""
[ "$f39_s" -eq 0 ] && f39_r5_copy=$(sed -n '/^# (5) /,/^# (6) /p' "$f39_t" | strip_shell_comments)
f39_region_fired=1
if [ "$f39_s" -eq 0 ] && printf '%s\n' "$f39_r5_copy" | grep -qF -- '--hf_path $HF_MODEL_PATH' && ! f39_r5_ok "$f39_t"; then
  f39_region_fired=0
fi
[ -n "${f39_t:-}" ] && rm -f "$f39_t" || true
if [ "$f39_region_fired" -eq 0 ]; then
  ok "MUST_FIRE fix39-region: a copy whose probe invocation regresses to the argparse-incompatible --hf_path goes red in-region (construction proven: '--hf_path \$HF_MODEL_PATH' present in the copy's step-(5) region)"
else
  no "MUST_FIRE UNREACHABLE (fix39-region): the invocation regression could not be constructed or does not fire (sed rc=$f39_s) — the region leg above is an unproven detector"
fi

# The probe's three exit states must map to three NAMED launcher refusals —
# BLOCKED means fix the strings, UNMEASURED means fix the census (never
# bypass it), anything else is wiring. Zero laundering of 0/1/3 into one arm.
# fix40 STRENGTHENING (house rule: consistent with this leg's own stated
# intent above, which is preserved verbatim). The three original needles
# pinned message spellings — '(probe rc=1)' / '(probe rc=3)' — written
# pre-measurement on the assumption that the probe's exit code ARRIVES at
# the launcher. Measured false on 2026-08-24: under torchrun every nonzero
# child exit arrives as rc=1 (ChildFailedError), so a leg demanding
# '(probe rc=3)' could only ever pin dead text, and the arm it pinned
# never once fired — presence-of-message is not reachability, the same
# vacuity class as an rc-only assertion on a leg whose experiment never
# ran. The corrected leg keeps the three named refusals (spellings updated
# to the verdict-line era) and ADDS what the measurement requires: the
# deciding switch must key on the CENSUS_VERDICT line (the only
# three-valued signal measured to survive the wrapper), and the
# measured-dead rc=3 arm shape is forbidden outright.
# Fail-before accounting: on the current tree the selector is
# `case "$census_rc" in`, `census_verdicts_n` does not exist, and the dead
# 3) arm is live text — red there, green after the launcher edit it guards.
f40_triage_ok() { # $1=launcher file (real or constructed copy) -> rc 0 iff the
                  # census triage keeps the three NAMED refusals, keys them on
                  # the verdict line, and carries no rc=3 arm. The MUST_FIRE
                  # below runs THIS predicate, never a paraphrase of it.
  local f40_lc
  f40_lc=$(strip_shell_comments < "$1")
  printf '%s\n' "$f40_lc" | grep -qF 'LoRA target census BLOCKED' \
    && printf '%s\n' "$f40_lc" | grep -qF 'LoRA target census UNMEASURED' \
    && printf '%s\n' "$f40_lc" | grep -qF 'LoRA target census infrastructure failure' \
    && printf '%s\n' "$f40_lc" | grep -qF 'case "$census_verdict" in' \
    && printf '%s\n' "$f40_lc" | grep -qF 'census_verdicts_n=' \
    && ! printf '%s\n' "$f40_lc" | grep -qF '3) cat "$CENSUS_OUT"'
}
f39_lc=$(strip_shell_comments < "$F39_LORA")
if f40_triage_ok "$F39_LORA"; then
  ok "fix39/40: three named census refusals kept (BLOCKED / UNMEASURED / infrastructure) and the triage keys on the CENSUS_VERDICT line — the abstention arm is REACHABLE under torchrun, whose rc is one bit (measured 2026-08-24: child exit 3 -> observed rc 1)"
else
  no "fix39/40: census triage incomplete or drifted — a named refusal vanished, the deciding switch does not key on the CENSUS_VERDICT line, or the measured-dead rc=3 arm returned (under torchrun that arm can never fire; it is coverage-shaped wallpaper)"
fi

# MUST_FIRE for the reachability leg (doctrine 3): rebuild TONIGHT's
# measured defect class over a live file — regress only the deciding
# selector, 'case "$census_verdict" in' -> 'case "$census_rc" in', on a
# temp copy — prove the construction non-vacuously (the live file must
# carry EXACTLY ONE verdict selector, the copy none, and the copy must
# carry an rc selector), and demand the SAME predicate the live leg runs
# report the copy NOT ok. On the current tree the construction is
# impossible (0 verdict selectors exist to regress), so this leg reports
# UNREACHABLE there — a red control by this file's own rule, never a skip:
# that red IS the declared fail-before state.
f40_t=$(mktemp "${TMPDIR:-/tmp}/fs-f40-triage.XXXXXX") \
  && sed 's/case "$census_verdict" in/case "$census_rc" in/' "$F39_LORA" > "$f40_t"
f40_s=$?
f40_nsel=$(grep -cF 'case "$census_verdict" in' "$F39_LORA" || true)
f40_csel=-1
if [ "$f40_s" -eq 0 ]; then
  f40_csel=$(grep -cF 'case "$census_verdict" in' "$f40_t" || true)
fi
f40_fired=1
if [ "$f40_s" -eq 0 ] && [ "$f40_nsel" -eq 1 ] && [ "$f40_csel" -eq 0 ] \
   && grep -qF 'case "$census_rc" in' "$f40_t" && ! f40_triage_ok "$f40_t"; then
  f40_fired=0
fi
[ -n "${f40_t:-}" ] && rm -f "$f40_t" || true
if [ "$f40_fired" -eq 0 ]; then
  ok "MUST_FIRE fix40-triage: regressing the deciding selector to the laundered rc on a copy turns the reachability leg red (construction non-vacuous: live verdict-selectors=$f40_nsel, copy=$f40_csel)"
else
  no "MUST_FIRE UNREACHABLE (fix40-triage): the rc-selector regression did not construct or does not fire (sed rc=$f40_s; live verdict-selectors=$f40_nsel, copy=$f40_csel) — the reachability leg above is an unproven detector"
fi

# Post-run G2 must read expected counts from the real-matcher census and must
# no longer read the laundered dump artifact anywhere in code (comments may
# narrate it; the comment-stripped view is what this leg reads).
if printf '%s\n' "$f39_lc" | grep -qF 'target_census.txt' \
   && printf '%s\n' "$f39_lc" | grep -qF 'CENSUS_TARGET' \
   && ! printf '%s\n' "$f39_lc" | grep -qF 'module_dump.txt'; then
  ok "fix39: post-run G2 reads expected counts from the real-matcher census (target_census.txt / CENSUS_TARGET rows); no code path still reads the grep-scored module dump"
else
  no "fix39: post-run gates still read the laundered dump oracle (module_dump.txt present in code) or never read the census file — the drift check is gone or still self-certifying"
fi

# G2's expert-attach expectation must key on the post-decision record the
# arm-identity legs above pin as authoritative. On the measured-dense base
# with default EXPERT_TARGETS=1 the arm branch deliberately drops the expert
# strings; demanding expert attachments of the raw request would be a
# guaranteed false red on the first correct run (doctrine 5, symmetric).
if printf '%s\n' "$f39_lc" | grep -qF 'if [[ "$LORA_ARM" == "L1" ]]'; then
  ok "fix39: G2's expert-attach expectation keys on LORA_ARM — the dense EXPERT_TARGETS=1->base4 relabel can no longer mint a false red demanding adapters the branch deliberately dropped"
else
  no "fix39: G2's expert expectation keys on raw EXPERT_TARGETS — on the measured-dense base with the default =1 that is a guaranteed false red on a correct run"
fi

echo "== fix40: every census-verdict/rc combination resolves to a named outcome, and the probe's one-verdict-per-exit-path premise is pinned =="
# Fail-before accounting, per this file's convention: on the current tree
# the composite triage leg and its MUST_FIRE (fix39 section above,
# strengthened in place) and the two DISAGREEMENT legs below are red — 4
# new reds in total. The probe-carrier guard and its MUST_FIRE are GREEN
# ON BOTH TREES BY CONSTRUCTION — the probe is not edited tonight; its six
# abstention paths already print their verdict lines — so those two are
# invariant guards in the fix35 absence-guard sense, disclosed here so
# nobody quotes them as fail-before legs. Predicted tallies: current tree
# 49 passed / 4 failed / 3 named abstentions; patched tree 53 / 0 / 3.
#
# The word DISAGREEMENT is the census triage's stable token: it appears
# ONLY in the six failure messages of THAT triage — silent-but-rc0,
# doubled verdict, CLEAR with nonzero rc, BLOCKED with rc0, UNMEASURED
# with rc0, unknown verdict — once each. The count below reads the
# comment-stripped view, so prose can never mint one; the corroborated
# arms and the plain infrastructure-failure arm deliberately do NOT carry
# the token, keeping 6 an exact census of the named disagreement arms,
# not a loose mood.
# fix43 addendum — the token's SCOPE is now contract, not observation.
# fix42's replay triage shipped a seventh DISAGREEMENT (launcher step (7),
# "replay verdict CLEAR but rc=..."), the count read 7, and both legs of
# this pair went red against a launcher whose only sin was a shared
# namespace: one grep tally had silently come to span two triages, and a
# deleted census arm compensated by an added replay arm would have been
# invisible inside the merged count (the compensating-error blind spot).
# The repair (launcher Edit 1 of fix43) namespaces the populations: the
# census triage KEEPS this token at 6 (this leg), the replay triage
# carries CONTRADICTION at 6 (censused separately below, same drop-one
# MUST_FIRE). grep matches substrings and \b is not portable to this
# suite's BSD grep, which is why the replay token shares no substring
# with this one — merged counts are refused by construction, and raising
# this leg's constant would be the checksum-armored no-op of doctrine 4.
f40_lc=$(strip_shell_comments < "$F39_LORA")
f40_dn=$(printf '%s\n' "$f40_lc" | grep -c 'DISAGREEMENT' || true)
if [ "$f40_dn" -eq 6 ]; then
  ok "fix40: 6 of 6 verdict/rc disagreement cases are named failure arms (silent-success, doubled verdict, CLEAR+nonzero-rc, BLOCKED+rc0, UNMEASURED+rc0, unknown verdict) — every disagreement BLOCKS by name; none launder into a conviction or a pass"
else
  no "fix40: expected 6 named DISAGREEMENT arms in the census triage, found $f40_dn — an unwired disagreement case is a state the launch classifies silently (the pre-fix40 abstention-as-conviction class)"
fi

# MUST_FIRE (count sensitivity, doctrine 3): doctor ONE DISAGREEMENT token
# out of the stripped launcher text and require the tally to fall by
# EXACTLY one. The live-is-6 conjunct keeps the construction non-vacuous:
# on the current tree the live count is 0, so this leg is red there
# alongside the leg it arms. A count that cannot move is wallpaper.
f40_dt=$(mktemp "${TMPDIR:-/tmp}/fs-f40-disarm.XXXXXX") \
  && printf '%s\n' "$f40_lc" | sed '1,/DISAGREEMENT/s/DISAGREEMENT/NOUN-VERB-MISMATCH/' > "$f40_dt"
f40_ds=$?
f40_dtn=-1
[ "$f40_ds" -eq 0 ] && f40_dtn=$(grep -c 'DISAGREEMENT' "$f40_dt" || true)
[ -n "${f40_dt:-}" ] && rm -f "$f40_dt" || true
if [ "$f40_ds" -eq 0 ] && [ "$f40_dn" -eq 6 ] && [ "$f40_dtn" -eq 5 ]; then
  ok "MUST_FIRE fix40-disagreements: removing one named arm drops the tally 6 -> $f40_dtn (the count tracks the real triage, not prose)"
else
  no "MUST_FIRE UNREACHABLE (fix40-disagreements): the doctored copy did not drop the tally as required (write rc=$f40_ds; live=$f40_dn, copy=$f40_dtn) — the arm-count leg above is an unproven detector"
fi

# fix43 — the replay triage's disagreement census (the fix42 namespace
# breach, repaired at the launcher). Counts CONTRADICTION, reads the SAME
# comment-stripped view as the DISAGREEMENT leg above, and is deliberately
# a SEPARATE tally: two populations, two counts, so the compensating case
# (one arm deleted in each triage) moves each count by one instead of
# netting to green. Expected 6: silent+rc0, doubled verdict, CLEAR with
# nonzero rc, BLOCKED with rc0, UNMEASURED with rc0, unknown verdict —
# once each, with the plain infrastructure-failure arm carrying no token,
# mirroring the census triage's design. Fail-before, measured: 0 on the
# pre-fix43 launcher (fix42 shipped zero of the token there).
f43_cn=$(printf '%s\n' "$f40_lc" | grep -c 'CONTRADICTION' || true)
if [ "$f43_cn" -eq 6 ]; then
  ok "fix43: 6 of 6 replay verdict/rc disagreement cases are named CONTRADICTION arms (silent-success, doubled verdict, CLEAR+nonzero-rc, BLOCKED+rc0, UNMEASURED+rc0, unknown verdict) — a replay triage disagreement BLOCKS by name in its own namespace, never laundering into the census triage's tally"
else
  no "fix43: expected 6 named CONTRADICTION arms in the replay triage, found $f43_cn — a replay disagreement without a name is a state the launch classifies silently, and merging it into the census tally was measured tonight (fix42's 7th token took the fix40 legs red) and refused"
fi

# MUST_FIRE (count sensitivity, doctrine 3): doctor ONE CONTRADICTION
# token out of the stripped launcher text and require the tally to fall by
# EXACTLY one, with the live-is-6 conjunct keeping the construction
# non-vacuous — on the pre-fix43 tree the live count is 0, so this leg is
# red there (UNREACHABLE) alongside the leg it arms, never skipped.
f43_ct=$(mktemp "${TMPDIR:-/tmp}/fs-f43-disarm.XXXXXX") \
  && printf '%s\n' "$f40_lc" | sed '1,/CONTRADICTION/s/CONTRADICTION/TOKEN-REMOVED/' > "$f43_ct"
f43_cs=$?
f43_ctn=-1
[ "$f43_cs" -eq 0 ] && f43_ctn=$(grep -c 'CONTRADICTION' "$f43_ct" || true)
[ -n "${f43_ct:-}" ] && rm -f "$f43_ct" || true
if [ "$f43_cs" -eq 0 ] && [ "$f43_cn" -eq 6 ] && [ "$f43_ctn" -eq 5 ]; then
  ok "MUST_FIRE fix43-contradictions: removing one named replay arm drops the tally 6 -> $f43_ctn (the replay census tracks the real triage, not prose)"
else
  no "MUST_FIRE UNREACHABLE (fix43-contradictions): the doctored copy did not drop the tally as required (write rc=$f43_cs; live=$f43_cn, copy=$f43_ctn) — the replay arm-count leg above is an unproven detector"
fi

# Probe-side carrier guard: the launcher's verdict-line triage stands on
# ONE premise — that the probe prints exactly one CENSUS_VERDICT= line on
# every exit path. GREEN ON BOTH TREES BY CONSTRUCTION (the probe is not
# edited tonight): an invariant guard, disclosed per the section header.
# Denominators: 8 verdict prints (6 UNMEASURED abstention paths + 1 BLOCKED
# + 1 CLEAR) beside 8 vocabulary returns. If a future probe edit adds an
# exit path without its verdict print, the counts diverge and this goes
# red BEFORE the launcher ever has to trust a silent path.
f40_pv=$(grep -c 'CENSUS_VERDICT=' "$F39_PROBE" || true)
f40_pr=$(grep -cE 'return EXIT_(CLEAR|BLOCKED|UNMEASURED)' "$F39_PROBE" || true)
if [ "$f40_pv" -eq 8 ] && [ "$f40_pr" -eq 8 ]; then
  ok "fix40: probe prints exactly one CENSUS_VERDICT= line per exit path (8 prints / 8 vocabulary returns: 6 UNMEASURED + 1 BLOCKED + 1 CLEAR) — the launcher's new decider has a carrier on every path"
else
  no "fix40: probe verdict-carrier accounting drifted (prints=$f40_pv, returns=$f40_pr; both required =8) — a path without a verdict print is a path the launcher triage must refuse to trust"
fi

# MUST_FIRE for the carrier guard: doctor the FIRST verdict print out of a
# probe copy and require the print tally to fall by EXACTLY one (8 -> 7),
# with the live-is-8 conjunct keeping the construction non-vacuous.
f40_pt=$(mktemp "${TMPDIR:-/tmp}/fs-f40-probe.XXXXXX") \
  && sed '1,/CENSUS_VERDICT=/s/CENSUS_VERDICT=/VERDICT-REMOVED/' "$F39_PROBE" > "$f40_pt"
f40_ps=$?
f40_pvc=-1
[ "$f40_ps" -eq 0 ] && f40_pvc=$(grep -c 'CENSUS_VERDICT=' "$f40_pt" || true)
[ -n "${f40_pt:-}" ] && rm -f "$f40_pt" || true
if [ "$f40_ps" -eq 0 ] && [ "$f40_pv" -eq 8 ] && [ "$f40_pvc" -eq 7 ]; then
  ok "MUST_FIRE fix40-carrier: removing one verdict print from a probe copy drops the tally 8 -> $f40_pvc (the guard reads the shipped source, not a paraphrase)"
else
  no "MUST_FIRE UNREACHABLE (fix40-carrier): the doctored probe copy did not drop the tally as required (sed rc=$f40_ps; live=$f40_pv, copy=$f40_pvc) — the carrier guard above is an unproven detector"
fi

echo "== fix41: the UNMEASURED arm must be firable ON DEMAND, exercised through the launcher's own text =="
# Fail-before accounting, per this file's convention: on the current tree all
# SIX legs in this section are RED — the launcher carries no drill knob, no
# banner, no scoped injection, no proceed-refusal, and no drill naming in the
# UNMEASURED arm. fix40 made the arm reachable (the fix39/40 legs above pin
# the keying); nothing has ever fired it through the real launcher, which is
# the gap fix41 exists to close — an arm never observed to fire is not a
# control, per the launcher's own retired-rc=3 comment.
#
# The simulation legs run the launcher's REAL step-(5) text — sed-extracted
# from the drill preamble through the census_bad gate, the same no-paraphrase
# rule this harness applies to EXTRA_OVERRIDES, the LoRA arm block, and the
# fix35 verdict mappers — against ONE declared stub: run_in_container
# becomes canned probe-shaped stdout plus a chosen rc, written through the
# region's own redirect into $CENSUS_OUT (the same bytes the real triage
# reads, via the same file discipline). This is not the drill and never
# claims to be: the genuine abstention (real probe, real wrapper, real tray)
# is delivered by the FS_CENSUS_DRILL_BUILD_FAILURE=1 hardware run these legs
# pin the wiring for. What these legs prove tonight is that the text the
# hardware drill will exercise routes its three fixture classes to the three
# required outcomes — a contract that only greps for the knob would green a
# drill whose text never executes, and the mutations.json rows this section
# arms exist to keep that distinction honest.
F41_LORA=$LORA
F41_SIMDIR=$(mktemp -d "${TMPDIR:-/tmp}/fs-f41-sim.XXXXXX" 2>/dev/null) || F41_SIMDIR=""
[ -n "$F41_SIMDIR" ] || { F41_SIMDIR="${TMPDIR:-/tmp}/fs-f41-sim.$$"; mkdir -p "$F41_SIMDIR" 2>/dev/null || F41_SIMDIR=""; }

f41_region() { # $1=launcher file -> stdout: the REAL drill preamble + census
               # invocation + verdict-line triage + per-target re-verification,
               # verbatim. Empty stdout when the drill marker is absent — the
               # current tree's honest state, reported red below, never skipped.
  sed -n '/^# fix41 — CENSUS MUST_FIRE DRILL/,/^\[\[ "$census_bad" -eq 0 \]\] || exit 1$/p' "$1"
}

f41_sim() { # $1=launcher file $2=drill knob (0|1) $3=fixture (unmeasured|clear|anomaly) $4=stub rc
            # Echoes all output; rc is the region's own rc. 99/98 announce
            # missing drill text / missing scratch dir — both are legs-red
            # answers, never greens.
  [ -n "$F41_SIMDIR" ] || { echo "SIMDIR-MISSING"; return 98; }
  local f41_reg
  f41_reg=$(f41_region "$1")
  [ -n "$f41_reg" ] || { echo "EXTRACTION-EMPTY"; return 99; }
  (
    F41_FIXTURE=$3 F41_STUB_RC=$4
    LORA_TARGETS='linear_qkv,linear_proj,*.mlp.mlp.linear_fc1,*.mlp.mlp.linear_fc2'
    MASTER_PORT=29517; REPO=/tmp/f41; HF_MODEL_PATH=/tmp/f41-hf; EP=1
    CENSUS_PROBE=/tmp/f41/lora_target_census.py
    PREFLIGHT_DIR=$F41_SIMDIR
    CENSUS_OUT=$PREFLIGHT_DIR/target_census.txt
    FS_CENSUS_DRILL_BUILD_FAILURE=$2
    run_in_container() {
      case "$F41_FIXTURE" in
        unmeasured)
          printf '%s\n' \
            'MODEL BUILD FAILED - synthetic harness fixture mimicking the measured RuntimeError shape; the GENUINE abstention is the hardware drill run, not this file' \
            'CENSUS_VERDICT=UNMEASURED (0 of 4 targets certified: model build failed)' ;;
        anomaly)
          printf '%s\n' \
            'CONTROLS FAILED - synthetic fixture: a DIFFERENT abstention path than the drill declares' \
            'CENSUS_VERDICT=UNMEASURED (0 of 4 targets certified: controls failed)' ;;
        clear)
          printf '%s\n' \
            'CENSUS_POPULATION total=1556 leaf=1500 non_leaf=56 is_expert_linear=0' \
            "CENSUS_CONTROL MUST_FIRE 'linear_qkv' -> 42 modules (require > 0): OK" \
            "CENSUS_CONTROL MUST_NOT_FIRE 'zzz_no_such_module_xyz' -> 0 modules (require 0): OK" \
            "CENSUS_CONTROL ANTI_NARROWING non_leaf_population=56 (require > 0), 'linear_proj' matches that are non-leaf: 42 of 42 (require >= 1): OK" \
            'CENSUS_TARGET linear_qkv 42 42' \
            'CENSUS_TARGET linear_proj 42 42' \
            'CENSUS_TARGET *.mlp.mlp.linear_fc1 42 0' \
            'CENSUS_TARGET *.mlp.mlp.linear_fc2 42 0' \
            'CENSUS_VERDICT=CLEAR (4 of 4 shipped targets attach; population 1556; controls 3/3 OK)' ;;
      esac
      return "$F41_STUB_RC"
    }
    eval "$f41_reg"
  )
}

f41_drill_ok() { # $1=launcher file -> rc 0 iff all six conjuncts hold over
                 # the comment-stripped view (the MUST_FIRE below runs THIS
                 # predicate on a doctored copy — never a paraphrase of it).
  local f41_v
  f41_v=$(strip_shell_comments < "$1")
  printf '%s\n' "$f41_v" | grep -qF 'FS_CENSUS_DRILL_BUILD_FAILURE' \
    && printf '%s\n' "$f41_v" | grep -qF 'DRILL: FS_CENSUS_DRILL_BUILD_FAILURE=1' \
    && printf '%s\n' "$f41_v" | grep -qF '${census_cuda_prefix}torchrun --nnodes=1' \
    && printf '%s\n' "$f41_v" | grep -qF 'CENSUS DRILL ARMED BUT THE LAUNCH PROCEEDED' \
    && printf '%s\n' "$f41_v" | grep -qF 'DRILL FIRED' \
    && printf '%s\n' "$f41_v" | grep -qF 'DRILL ANOMALY'
}

# -- leg A (static composite): the drill exists, is audible, is scoped into
# the MEASURED census invocation, refuses silent proceed, and names its
# success/anomaly in the UNMEASURED arm. Any one conjunct missing fails it.
if f41_drill_ok "$F41_LORA"; then
  ok "fix41: census UNMEASURED drill wired — knob read, banner audible with its denominator, CUDA_VISIBLE_DEVICES= injection scoped to the measured census invocation only, proceed-with-drill refusal present, UNMEASURED arm names DRILL FIRED / DRILL ANOMALY"
else
  no "fix41: census drill incomplete — one of knob/banner/scoped-injection/proceed-refusal/drill-naming is absent; an abstention arm that cannot be fired on demand reads as coverage (the fix40 gap) and is not a control (doctrine 3)"
fi

# -- leg B (MUST_FIRE for leg A, doctrine 3): the scariest drill regression
# is an editor keeping the knob, banner and prefix while silently dropping
# the PROCEED REFUSAL — the one conjunct that makes 'armed but cannot fire'
# a failure instead of a mystery launch. Substitute the refusal signature
# away on a temp copy; prove the construction non-vacuously (the needle
# occurs EXACTLY ONCE in the live launcher, ZERO times in the copy, and the
# knob provably survives the sed — the message is doctored, not the drill);
# demand the SAME predicate the live leg runs report the copy NOT ok. On the
# current tree the live count is 0, so this leg is red (UNREACHABLE) there —
# by this file's rule an unbuildable firing input is a failed control, never
# a skip.
f41_t=$(mktemp "${TMPDIR:-/tmp}/fs-f41-refusal.XXXXXX") \
  && sed 's/FATAL: CENSUS DRILL ARMED BUT THE LAUNCH PROCEEDED/FATAL: census drill bookkeeping/' "$F41_LORA" > "$f41_t"
f41_s=$?
f41_nref=$(strip_shell_comments < "$F41_LORA" | grep -cF 'CENSUS DRILL ARMED BUT THE LAUNCH PROCEEDED' || true)
f41_cref=-1
[ "$f41_s" -eq 0 ] && f41_cref=$(strip_shell_comments < "$f41_t" | grep -cF 'CENSUS DRILL ARMED BUT THE LAUNCH PROCEEDED' || true)
f41_fired=1
if [ "$f41_s" -eq 0 ] && [ "$f41_nref" -eq 1 ] && [ "$f41_cref" -eq 0 ] \
   && grep -qF 'FS_CENSUS_DRILL_BUILD_FAILURE:-0' "$f41_t" && ! f41_drill_ok "$f41_t"; then
  f41_fired=0
fi
[ -n "${f41_t:-}" ] && rm -f "$f41_t" || true
if [ "$f41_fired" -eq 0 ]; then
  ok "MUST_FIRE fix41-drill: doctoring the proceed-refusal signature out of a copy (needle live=$f41_nref, copy=$f41_cref; knob provably survives) turns the composite leg red — the drill cannot silently lose its guarantee"
else
  no "MUST_FIRE UNREACHABLE (fix41-drill): refusal-needle live=$f41_nref copy=$f41_cref (sed rc=$f41_s), or the predicate still greens the doctored copy — leg A is an unproven detector"
fi

# -- leg C (simulation, drill ARMED, genuine-UNMEASURED-shaped fixture): the
# drill's whole reason to exist. The launcher's REAL extracted text must
# (i) announce the drill, (ii) land the fixture on the UNMEASURED arm — the
# direct proof that the arm keys on exactly what the drill can deliver —
# (iii) name DRILL FIRED (positive evidence: the payload names the forced
# path), and (iv) BLOCK with rc 1.
out=$(f41_sim "$F41_LORA" 1 unmeasured 1 2>&1); rc=$?
if [ $rc -eq 1 ] \
   && printf '%s' "$out" | grep -q 'DRILL: FS_CENSUS_DRILL_BUILD_FAILURE=1' \
   && printf '%s' "$out" | grep -q 'DRILL FIRED' \
   && printf '%s' "$out" | grep -q 'census UNMEASURED'; then
  ok "fix41 drill routing: armed drill + genuine UNMEASURED shape lands on the verdict-keyed UNMEASURED arm, names DRILL FIRED, and BLOCKS (rc=$rc) — the arm keys on exactly what the drill delivers (suite-internal fixture; the genuine hardware fire is owed to <compute-node>)"
else
  no "fix41 drill routing broken: armed drill + UNMEASURED fixture gave rc=$rc (want 1 with DRILL FIRED) — '$out'"
fi

# -- leg D (simulation, drill ARMED, corroborated CLEAR fixture): THE
# guarantee. A launch that PROCEEDS with the drill set must be a stated
# failure — if the injection ever drops or the trigger drifts, this refusal
# is the only thing between the estate and another never-yet-fired arm that
# reads as coverage. rc 0 here would be the exact defect this section
# refuses.
out=$(f41_sim "$F41_LORA" 1 clear 0 2>&1); rc=$?
if [ $rc -eq 1 ] && printf '%s' "$out" | grep -q 'CENSUS DRILL ARMED BUT THE LAUNCH PROCEEDED'; then
  ok "fix41 drill anti-vacuity: armed drill + corroborated CLEAR is REFUSED by name (rc=$rc) — a drill that cannot fire can never again launder into a launch"
else
  no "fix41 drill anti-vacuity FAILED: armed drill + CLEAR fixture rc=$rc (want 1 with the named refusal) — '$out'"
fi

# -- leg E (MUST_PASS, simulation, drill UNARMED, corroborated CLEAR
# fixture): the control must be inert off the trigger — the empty prefix
# leaves the census invocation byte-identical, no DRILL text may print, and
# a healthy census must proceed (rc 0). Fail-before by extraction like its
# siblings: on the current tree there is no region to run (EXTRACTION-EMPTY).
out=$(f41_sim "$F41_LORA" 0 clear 0 2>&1); rc=$?
if [ $rc -eq 0 ] && ! printf '%s' "$out" | grep -q 'DRILL'; then
  ok "fix41 drill MUST_PASS: unarmed launch with a healthy census proceeds (rc=0, no DRILL output) — the control is inert off its trigger, never a tax on a healthy launch"
else
  no "fix41 drill MUST_PASS broke: unarmed CLEAR fixture rc=$rc or DRILL text leaked off-trigger — '$out'"
fi

# -- leg F (simulation, drill ARMED, WRONG-PATH abstention fixture): the
# positive-evidence rule (the fix41-§4 lesson). The drill may claim success
# ONLY when the abstention names the one path it forced; a 'controls failed'
# abstention under an armed build-failure drill must name DRILL ANOMALY —
# never DRILL FIRED — and still BLOCK. rc-shape alone must never mint the
# MUST_FIRE receipt.
out=$(f41_sim "$F41_LORA" 1 anomaly 1 2>&1); rc=$?
if [ $rc -eq 1 ] && printf '%s' "$out" | grep -q 'DRILL ANOMALY' && ! printf '%s' "$out" | grep -q 'DRILL FIRED'; then
  ok "fix41 drill positive-evidence: a wrong-path abstention under the armed drill names DRILL ANOMALY (never DRILL FIRED) and still BLOCKS — the drill cannot take credit for a fire it did not cause"
else
  no "fix41 drill positive-evidence FAILED: wrong-path fixture rc=$rc — '$out'"
fi

[ -n "$F41_SIMDIR" ] && rm -rf "$F41_SIMDIR" || true

echo "== fix42: the peft.* knob path has an on-demand MUST_FIRE, and the replay triage routes every verdict word to its named arm =="
# Fail-before accounting, per this file's convention — and this section has
# TWO fail-before trees, stated separately because conflating them is how
# false greens get quoted. Tree (i), the PRE-fix42 launcher (the tree that
# made these legs necessary): no drill knob, no refusals, no step (7), no
# REPLAY_VERDICT anywhere — both extractions come back empty, every
# simulation leg announces EXTRACTION-EMPTY and reports red, the static
# composite's predicate is false, and both MUST_FIREs are red (UNREACHABLE)
# by this file's own rule: 15 of 15 legs red there. Tree (ii), tonight's
# attached tree, where fix42's code is landed but this section and the
# fix43 launcher edits are not: the legs pinning already-landed correct
# behavior go green on first sight (disclosed here, like the fix40
# probe-carrier guard, and NOT quoted as fail-before legs), while five go
# red — the MUST_PASS leg demanding the drill-hint note (red: the pre-fix43
# replay_discrim awk -F= extraction could never equal "0"; the launcher
# repair is fix43 Edit 1), and the four legs keyed on the replay triage's
# CONTRADICTION namespace (red: fix42 shipped zero of that token). The
# simulation legs run the launcher's REAL text — same no-paraphrase rule as
# EXTRA_OVERRIDES, the LoRA arm block, the fix35 mappers, and the fix41
# drill region — against ONE declared stub: run_in_container becomes canned
# probe-shaped stdout plus a chosen rc, written through the region's own
# redirect into $REPLAY_OUT, the same bytes the real triage reads.
F42_LORA=$LORA
F42_SIMDIR=$(mktemp -d "${TMPDIR:-/tmp}/fs-f42-sim.XXXXXX" 2>/dev/null) || F42_SIMDIR=""
[ -n "$F42_SIMDIR" ] || { F42_SIMDIR="${TMPDIR:-/tmp}/fs-f42-sim.$$"; mkdir -p "$F42_SIMDIR" 2>/dev/null || F42_SIMDIR=""; }

f42_drill_block() { # $1=launcher file -> stdout: the REAL FS_PEFT_DRILL_RANK
                    # arming block, verbatim. The address anchors at column 0,
                    # which is load-bearing: the same 'if [[ -n
                    # "${FS_PEFT_DRILL_RANK:-}" ]]; then' line recurs twice
                    # more in step (7), INDENTED, so the ^ anchor sees only
                    # the arming block; the range ends at its own top-level fi.
                    # Empty stdout on the pre-fix42 tree, reported red below.
  sed -n '/^if \[\[ -n "${FS_PEFT_DRILL_RANK:-}" \]\]; then$/,/^fi$/p' "$1"
}

f42_replay_region() { # $1=launcher file -> stdout: the REAL step-(7) invocation
                      # + guard + verdict triage, verbatim: from the rc
                      # initialiser to the triage's own esac. There is no nested
                      # case between those anchors (verified by reading), so the
                      # first ^esac$ IS the triage's. Empty on the pre-fix42 tree.
  sed -n '/^replay_rc=0$/,/^esac$/p' "$1"
}

f42_drill_run() { # $1=launcher file $2=drill knob ("" = knob absent from env)
                  # Echoes all block output + one RESULT_RANK line; rc is the
                  # block's own (a refusal exits 1 from inside the subshell).
  local blk
  blk=$(f42_drill_block "$1")
  [ -n "$blk" ] || { echo "EXTRACTION-EMPTY"; return 99; }
  ( LORA_RANK=32
    if [ -n "$2" ]; then FS_PEFT_DRILL_RANK=$2; export FS_PEFT_DRILL_RANK; fi
    eval "$blk"
    echo "RESULT_RANK=$LORA_RANK" )
}

f42_replay_sim() { # $1=launcher file $2=drill knob ("" = unarmed) $3=fixture $4=stub rc
                   # Evals the REAL step-(7) region in a subshell whose
                   # run_in_container writes the fixture through the region's own
                   # redirect. 99/98 announce missing drill text / scratch dir —
                   # leg-red answers, never greens (the f41_sim idiom).
  [ -n "$F42_SIMDIR" ] || { echo "SIMDIR-MISSING"; return 98; }
  local f42_reg
  f42_reg=$(f42_replay_region "$1")
  [ -n "$f42_reg" ] || { echo "EXTRACTION-EMPTY"; return 99; }
  (
    F42_FIXTURE=$3 F42_STUB_RC=$4
    REPO=/tmp/f42sim; HF_MODEL_PATH=/tmp/f42sim-hf; MASTER_PORT=29623
    RECIPE=gemma4_vl_e4b_peft_config; PEFT_SCHEME=lora; SEQ_LENGTH=8192
    REPLAY_PROBE=/tmp/f42sim/peft_override_replay.py
    LORA_RANK=32; LORA_ALPHA=64; LORA_DROPOUT=0.0
    LORA_TARGETS='linear_qkv,linear_proj,*.mlp.mlp.linear_fc1,*.mlp.mlp.linear_fc2'
    CLI_OVERRIDES="peft.dim=$LORA_RANK peft.alpha=$LORA_ALPHA"
    # An ARMED sim models the post-arming state directly (LORA_RANK already
    # perturbed by the arming block, which legs I-L prove separately): the
    # region under test is step (7), not the arming block.
    [ -n "$2" ] && { FS_PEFT_DRILL_RANK=$2; LORA_RANK=$2; }
    REPLAY_OUT=$F42_SIMDIR/override_replay.txt
    run_in_container() {
      case "$F42_FIXTURE" in
        drilled-clear) printf '%s\n' \
          'REPLAY_KNOB_DISCRIMINATING=1 (pre-composition dim=32 alpha=64 dropout=0.0 targets=[fixture])' \
          "REPLAY_PEFT dim=$LORA_RANK alpha=64 dropout=0.0 targets=[fixture] type=LoraConfig transform_intact=True knobs_checked=4" \
          'REPLAY_VERDICT=CLEAR (4 of 4 shipped peft knobs resolved through the real process_config_with_overrides)' ;;
        default-clear) printf '%s\n' \
          'REPLAY_KNOB_DISCRIMINATING=0 (pre-composition dim=32 alpha=64 dropout=0.0 targets=[fixture])' \
          'REPLAY_PEFT dim=32 alpha=64 dropout=0.0 targets=[fixture] type=LoraConfig transform_intact=True knobs_checked=4' \
          'REPLAY_VERDICT=CLEAR (4 of 4 shipped peft knobs resolved; landed-vs-default indistinguishable undrilled)' ;;
        blocked)    printf '%s\n' \
          'REPLAY_PEFT dim=32 alpha=64 dropout=0.0 targets=[fixture] type=LoraConfig transform_intact=True knobs_checked=4' \
          'REPLAY_VERDICT=BLOCKED (1 of 4 shipped knobs did not resolve: dim: shipped=96 resolved=32)' ;;
        unmeasured) printf '%s\n' \
          'REPLAY_VERDICT=UNMEASURED (imports failed: ModuleNotFoundError: megatron)' ;;
        unknown)    printf '%s\n' \
          'REPLAY_VERDICT=HALF-CLEAR (a word outside the shipped vocabulary)' ;;
        doubled)    printf '%s\n' \
          'REPLAY_VERDICT=CLEAR (first print)' \
          'REPLAY_VERDICT=CLEAR (second print — probe breach or foreign writer)' ;;
        silent)     printf '%s\n' \
          'probe noise with no verdict line at all' ;;
      esac
      return "$F42_STUB_RC"
    }
    eval "$f42_reg"
  )
}

f42_drill_ok() { # $1=launcher file -> rc 0 iff the drill exists end-to-end:
                 # knob read, BOTH refusals (non-integer, =32), the perturbation
                 # assignment, the banner, the replay step's DRILL-FIRED demand
                 # on the resolved value, and both refuse-to-launch signatures.
                 # The MUST_FIRE below runs THIS predicate on a doctored copy.
  local v
  v=$(strip_shell_comments < "$1")
  printf '%s\n' "$v" | grep -qF 'FS_PEFT_DRILL_RANK' \
    && printf '%s\n' "$v" | grep -qF 'if [[ -n "${FS_PEFT_DRILL_RANK:-}" ]]; then' \
    && printf '%s\n' "$v" | grep -qF '"$FS_PEFT_DRILL_RANK" =~ ^[1-9][0-9]*$' \
    && printf '%s\n' "$v" | grep -qF '"$FS_PEFT_DRILL_RANK" != "32"' \
    && printf '%s\n' "$v" | grep -qF 'LORA_RANK=$FS_PEFT_DRILL_RANK' \
    && printf '%s\n' "$v" | grep -qF 'DRILL: FS_PEFT_DRILL_RANK=' \
    && printf '%s\n' "$v" | grep -qF '"^REPLAY_PEFT dim=$FS_PEFT_DRILL_RANK "' \
    && printf '%s\n' "$v" | grep -qF 'DRILL FIRED: FS_PEFT_DRILL_RANK=' \
    && printf '%s\n' "$v" | grep -qF 'KNOB DRILL ARMED BUT RESOLVED dim !=' \
    && printf '%s\n' "$v" | grep -qF 'FATAL-AND-DRILL-FIRED'
}

# -- leg G (static composite): the drill and its replay-step demand exist.
if f42_drill_ok "$F42_LORA"; then
  ok "fix42: knob-path drill wired — knob read, both refusals (non-integer / =32-equals-defaults), LORA_RANK perturbed as single source of truth, banner with its reason, step-(7) DRILL-FIRED demand on the resolved dim, and both refuse-to-launch signatures (proceed-on-revert and BLOCK-while-armed)"
else
  no "fix42: knob-path drill incomplete — one of knob/refusals/perturbation/banner/replay-demand/refusal signatures is absent; a drill that cannot discriminate landed-vs-default is the undrilled happy path wearing a knob (doctrine 3)"
fi

# -- leg H (MUST_FIRE for leg G, doctrine 3): the scariest drill regression
# is not deletion but de-fanging — an editor keeps knob, banner and replay
# demand while the '=32' discrimination guard drifts (sed to "0": a rank no
# default holds, so the drill still RUNS and still BLOCKs-passingly while
# perturbing nothing anyone checks). Doctor exactly that on a temp copy;
# prove construction non-vacuously (live needle count EXACTLY 1, copy 0,
# the arming-knob line provably surviving); demand the SAME predicate go
# red on the copy. Pre-fix42 live count is 0 -> red (UNREACHABLE), the
# declared fail-before.
f42_t=$(mktemp "${TMPDIR:-/tmp}/fs-f42-refusal.XXXXXX") \
  && sed 's/FS_PEFT_DRILL_RANK" != "32"/FS_PEFT_DRILL_RANK" != "0"/' "$F42_LORA" > "$f42_t"
f42_s=$?
f42_n32=$(strip_shell_comments < "$F42_LORA" | grep -cF '"$FS_PEFT_DRILL_RANK" != "32"' || true)
f42_c32=-1
[ "$f42_s" -eq 0 ] && f42_c32=$(strip_shell_comments < "$f42_t" | grep -cF '"$FS_PEFT_DRILL_RANK" != "32"' || true)
f42_fired=1
if [ "$f42_s" -eq 0 ] && [ "$f42_n32" -eq 1 ] && [ "$f42_c32" -eq 0 ] \
   && grep -qF '${FS_PEFT_DRILL_RANK:-}' "$f42_t" && ! f42_drill_ok "$f42_t"; then
  f42_fired=0
fi
[ -n "${f42_t:-}" ] && rm -f "$f42_t" || true
if [ "$f42_fired" -eq 0 ]; then
  ok "MUST_FIRE fix42-drill: doctoring the =32 discrimination guard out of a copy (needle live=$f42_n32, copy=$f42_c32; knob provably survives) turns the composite leg red — an undiscriminating drill cannot silently keep its banner"
else
  no "MUST_FIRE UNREACHABLE (fix42-drill): discrimination-needle live=$f42_n32 copy=$f42_c32 (sed rc=$f42_s), or the predicate still greens the doctored copy — leg G is an unproven detector"
fi

# -- legs I-L: the arming block's four behaviours, evaled from the REAL text.
out=$(f42_drill_run "$F42_LORA" abc 2>&1); rc=$?
if [ $rc -ne 0 ] && printf '%s' "$out" | grep -q 'must be a positive integer rank'; then
  ok "fix42 drill refusal: non-integer FS_PEFT_DRILL_RANK is refused by name (rc=$rc) — a typo'd drill must die at arming, not downstream for unrelated reasons"
else no "fix42 drill refusal broke (non-integer): rc=$rc — '$out'"; fi

out=$(f42_drill_run "$F42_LORA" 32 2>&1); rc=$?
if [ $rc -ne 0 ] && printf '%s' "$out" | grep -q 'equals BOTH the recipe default'; then
  ok "fix42 drill refusal: FS_PEFT_DRILL_RANK=32 is refused by name (rc=$rc) — a drill equal to both defaults perturbs nothing and can prove nothing"
else no "fix42 drill refusal broke (=32): rc=$rc — '$out'"; fi

out=$(f42_drill_run "$F42_LORA" 96 2>&1); rc=$?
if [ $rc -eq 0 ] && printf '%s' "$out" | grep -q 'DRILL: FS_PEFT_DRILL_RANK=96' && printf '%s' "$out" | grep -q 'RESULT_RANK=96'; then
  ok "fix42 drill arming: =96 lands LORA_RANK=96 with the DRILL banner (rc=$rc) — the single source of truth feeds tag, manifest, overrides and replay expectation downstream"
else no "fix42 drill arming broke (=96): rc=$rc — '$out'"; fi

out=$(f42_drill_run "$F42_LORA" "" 2>&1); rc=$?
if [ $rc -eq 0 ] && ! printf '%s' "$out" | grep -q 'DRILL' && printf '%s' "$out" | grep -q 'RESULT_RANK=32'; then
  ok "fix42 drill MUST_PASS: knob absent -> block inert, LORA_RANK stays 32, zero DRILL output (rc=$rc) — the control taxes nothing off its trigger"
else no "fix42 drill MUST_PASS broke (unarmed): rc=$rc — '$out'"; fi

# -- leg M: armed drill + replay resolving the DRILL value names DRILL FIRED.
out=$(f42_replay_sim "$F42_LORA" 96 drilled-clear 0 2>&1); rc=$?
if [ $rc -eq 0 ] && printf '%s' "$out" | grep -q 'DRILL FIRED: FS_PEFT_DRILL_RANK=96 resolved through the REAL composition path'; then
  ok "fix42 drill scoring: armed drill + resolved dim=96 prints DRILL FIRED and proceeds (rc=$rc) — positive evidence keyed on the resolved value, never on rc-shape (suite-internal fixture; the genuine hardware fire is owed to <compute-node>)"
else no "fix42 drill scoring broke: armed drilled-clear rc=$rc — '$out'"; fi

# -- leg N (anti-vacuity refusal): armed drill whose perturbation REVERTED to
# the default in composition must be a named refusal, rc 1.
out=$(f42_replay_sim "$F42_LORA" 96 default-clear 0 2>&1); rc=$?
if [ $rc -eq 1 ] && printf '%s' "$out" | grep -q 'KNOB DRILL ARMED BUT RESOLVED dim != 96'; then
  ok "fix42 drill anti-vacuity: armed drill + resolved dim=32 is REFUSED by name (rc=$rc) — a drill whose perturbation silently reverts can never launder into a launch"
else no "fix42 drill anti-vacuity FAILED: armed default-clear rc=$rc — '$out'"; fi

# -- leg O (MUST_PASS): unarmed, flat CLEAR — untaxed, and the drill-hint
# note fires (this leg exposed the pre-fix43 replay_discrim dead parse:
# awk -F= over the payload line could never yield "0"). Fail-before red on
# tonight's tree, green after launcher Edit 1 — declared, per section header.
out=$(f42_replay_sim "$F42_LORA" "" default-clear 0 2>&1); rc=$?
if [ $rc -eq 0 ] && printf '%s' "$out" | grep -q 'replay note:' && ! printf '%s' "$out" | grep -q 'DRILL FIRED'; then
  ok "fix42 drill MUST_PASS: unarmed flat CLEAR proceeds untaxed (rc=0), names its indistinguishability note, and never claims DRILL FIRED — the control is inert off its trigger"
else no "fix42 drill MUST_PASS broke: unarmed default-clear rc=$rc (on pre-fix43 trees this pins the replay_discrim '=0' dead parse) — '$out'"; fi

# -- legs P-U: the replay triage routes every remaining verdict/rc state to
# its named arm. Fixtures are suite-internal; each asserts rc AND the arm's
# own message class, so a leg can never pass because the experiment never ran.
out=$(f42_replay_sim "$F42_LORA" "" blocked 1 2>&1); rc=$?
if [ $rc -eq 1 ] && printf '%s' "$out" | grep -q 'override replay BLOCKED (verdict line' && ! printf '%s' "$out" | grep -q 'FATAL-AND-DRILL-FIRED'; then
  ok "fix42 replay routing: BLOCKED verdict with corroborating nonzero rc lands on the named BLOCKED arm (rc=$rc; unarmed, so no drill conflation) — re-spell guidance delivered, launch stopped"
else no "fix42 replay routing broke (BLOCKED): rc=$rc — '$out'"; fi

out=$(f42_replay_sim "$F42_LORA" "" unmeasured 1 2>&1); rc=$?
if [ $rc -eq 1 ] && printf '%s' "$out" | grep -q 'override replay UNMEASURED (verdict line) — the probe ABSTAINED'; then
  ok "fix42 replay routing: UNMEASURED lands on the named abstention arm with its 0-of-4 denominator (rc=$rc) — a stated abstention BLOCKS, never bypassed"
else no "fix42 replay routing broke (UNMEASURED): rc=$rc — '$out'"; fi

out=$(f42_replay_sim "$F42_LORA" "" unknown 0 2>&1); rc=$?
if [ $rc -eq 1 ] && printf '%s' "$out" | grep -q "unknown verdict 'HALF-CLEAR'" && printf '%s' "$out" | grep -q 'CONTRADICTION'; then
  ok "fix42 replay routing: an off-vocabulary verdict word is refused by name in the CONTRADICTION namespace (rc=$rc) — drift between probe and launcher is never guessed into a classification"
else no "fix42 replay routing broke (unknown verdict): rc=$rc — '$out'"; fi

out=$(f42_replay_sim "$F42_LORA" "" doubled 0 2>&1); rc=$?
if [ $rc -eq 1 ] && printf '%s' "$out" | grep -q 'REPLAY_VERDICT lines in one replay run' && printf '%s' "$out" | grep -q 'CONTRADICTION'; then
  ok "fix42 replay routing: two verdict lines in one run is a named carrier breach (rc=$rc) — evidence of ambiguous provenance is unreadable as evidence"
else no "fix42 replay routing broke (doubled verdict): rc=$rc — '$out'"; fi

out=$(f42_replay_sim "$F42_LORA" "" silent 0 2>&1); rc=$?
if [ $rc -eq 1 ] && printf '%s' "$out" | grep -q 'ZERO REPLAY_VERDICT lines yet rc=0' && printf '%s' "$out" | grep -q 'CONTRADICTION'; then
  ok "fix42 replay routing: zero verdict lines with rc=0 is refused by name (rc=$rc) — the one shape that would otherwise read as a silent pass (doctrines 1/4)"
else no "fix42 replay routing broke (silent+rc0): rc=$rc — '$out'"; fi

out=$(f42_replay_sim "$F42_LORA" "" default-clear 1 2>&1); rc=$?
if [ $rc -eq 1 ] && printf '%s' "$out" | grep -q 'replay verdict CLEAR but rc=1' && printf '%s' "$out" | grep -q 'CONTRADICTION'; then
  ok "fix42 replay routing: CLEAR with a nonzero rc is refused by name (rc=$rc) — a process that printed CLEAR and then failed certifies nothing"
else no "fix42 replay routing broke (CLEAR+rc!=0): rc=$rc — '$out'"; fi

[ -n "$F42_SIMDIR" ] && rm -rf "$F42_SIMDIR" || true

echo "== fix44/45: G3 reads the trainer's REAL census block and asserts realized rank == requested rank AND the rank-invariant frozen base (#76, §A5) =="
# Fail-before accounting, per this file's convention: on the pre-fix44 tree
# all 8 legs below are RED — the old G3 finds no HF-format needle in the
# measured Megatron-Bridge fixtures, and the firing fixtures land
# gate-fails whose messages carry none of the pinned needles. fix45
# addendum to the accounting (per-leg, never merged): on the PRE-fix45
# tree the two MUST_PASS legs are RED (their strengthened conjunct demands
# the frozen-base identity echo the pre-graft G3 never prints) and the new
# frozen-base MUST_FIRE is RED (nothing fires); the remaining five legs
# (rank, HF-format, zero, window, partial) are green on both trees BY
# CONSTRUCTION — same armed arms, fixtures re-shaped not replaced —
# disclosed here so nobody quotes them as fail-before legs.
# The fixtures labelled r32/r96 reproduce the MEASURED bytes of jobs
# 1787518637847 and 1787517960364, now including the third line of the
# measured four-line block ('Total parameters:', <compute-node>); the harness
# evals the REAL sed-extracted G3 block (the no-paraphrase rule this file
# applies to every region), with gate_fail/GATE stubbed so the block's own
# verdict is observable.
f44_sim=$(mktemp -d "${TMPDIR:-/tmp}/fs-f44.XXXXXX" 2>/dev/null) || f44_sim=""
[ -n "$f44_sim" ] || { f44_sim="${TMPDIR:-/tmp}/fs-f44.$$"; mkdir -p "$f44_sim" 2>/dev/null || f44_sim=""; }

f44_g3_run() { # $1=requested LORA_RANK  $2=log fixture
               # -> subshell: the REAL G3 span (from its own header to the G4
               # header, an eval-safe trailing comment) + a G3_GATE line.
  local blk
  blk=$(sed -n '/# G3 trainable census/,/# G4 schedule honesty/p' "$LORA")
  [ -n "$blk" ] || { echo "EXTRACTION-EMPTY"; return 99; }
  (
    GATE=0; OUTLOG=$2; LORA_RANK=$1
    LORA_TARGETS_BASE='linear_qkv,linear_proj,*.mlp.mlp.linear_fc1,*.mlp.mlp.linear_fc2'
    LORA_TARGETS=$LORA_TARGETS_BASE
    MOE=0; TP=1; CP=1
    gate_fail() { echo "GATE FAIL: $1"; GATE=1; }
    eval "$blk"
    echo "G3_GATE=$GATE"
  )
}

# fix45: both healthy fixtures now reproduce THREE lines of the measured
# four-line block (fix45 <compute-node> measurement) — a two-field fixture under
# a three-field parse would silently exercise the wrong arm (the §0(b)
# dead-control class), so every fixture in this section was re-audited
# against the new parse shape; the firing fixtures below carry the audit's
# rationale each.
cat > "$f44_sim/g3-r32.log" <<'F44G3'
 iteration       20/      20 | consumed samples:          320 | elapsed time per iteration (ms): 6214.9
PEFT Statistics:
  Total parameters: 7,750,478,080
  Trainable parameters: 63,078,400
  Trainable percentage: 0.81%
F44G3
cat > "$f44_sim/g3-r96.log" <<'F44G3'
 iteration       20/      20 | consumed samples:          320 | elapsed time per iteration (ms): 6341.2
PEFT Statistics:
  Total parameters: 7,876,634,880
  Trainable parameters: 189,235,200
  Trainable percentage: 2.40%
F44G3
# MUST_FIRE fixtures: each isolates its arm under the THREE-field parse.
# The HF-format fixture pins the deliberate non-acceptance of the pre-fix44
# needle's format (the stack changed -> re-derive; never accept both
# silently); it carries NONE of the three measured lines, so it still
# drives the 0-of-3 arm. The rank fixture is the fix42 silent-revert
# signature at the artifact level: r32's measured numbers under a REQUESTED
# rank of 96. The zero and window fixtures are RE-ANCHORED for the
# three-field parse (§0(b) applied home: a two-field fixture under the new
# parse would land on the ambiguous arm instead of the arm it was built to
# fire — a dead control printing PASS — so both gained the measured
# 'Total parameters: 7,750,478,080' line; the window fixture keeps its
# deliberate self-inconsistency — rank-perfect count AND frozen-perfect
# total beside an out-of-window percentage — so only the window arm can
# fire on it). The partial fixture is deliberately NOT enriched: 1 of the
# 3 measured lines is exactly the state its arm names (half-written log /
# one-line format drift).
cat > "$f44_sim/g3-hf.log" <<'F44G3'
  trainable params: 63,078,400 || all params: 7,787,490,304 || trainable%: 0.81
F44G3
cat > "$f44_sim/g3-zero.log" <<'F44G3'
PEFT Statistics:
  Total parameters: 7,750,478,080
  Trainable parameters: 0
  Trainable percentage: 0.00%
F44G3
cat > "$f44_sim/g3-window.log" <<'F44G3'
PEFT Statistics:
  Total parameters: 7,750,478,080
  Trainable parameters: 63,078,400
  Trainable percentage: 99.99%
F44G3
cat > "$f44_sim/g3-partial.log" <<'F44G3'
  Trainable parameters: 63,078,400
F44G3
# The frozen-base fixture is built from the FAILURE CLASS (§0(b)), not from
# an edge — the identity is rank-INVARIANT, so it has no edge to sit near:
# what the arm guards is any drift of the twice-measured constant. The
# fixture shifts ONLY 'Total parameters' by one param, so the rank identity
# and the window BOTH pass and only the frozen-base arm can fire: the real
# classes it names are a base that thawed by any amount, a trainable scope
# beyond the adapter set, or a census whose lines are not one run's.
cat > "$f44_sim/g3-basewalk.log" <<'F44G3'
PEFT Statistics:
  Total parameters: 7,750,478,081
  Trainable parameters: 63,078,400
  Trainable percentage: 0.81%
F44G3

out=$(f44_g3_run 32 "$f44_sim/g3-r32.log")
if printf '%s' "$out" | grep -q 'G3_GATE=0' \
   && printf '%s' "$out" | grep -qF 'trainable=63078400 (0.81%)' \
   && printf '%s' "$out" | grep -qF 'realized rank == requested rank 32' \
   && printf '%s' "$out" | grep -qF 'frozen-base identity'; then
  ok "fix44/45-G3 MUST_PASS: the MEASURED r32 production bytes (now with the third measured line, Total parameters: 7,750,478,080) parse and clear with realized rank == requested rank 32 AND the rank-invariant frozen-base identity holding to the integer (2 measured logs examined when re-deriving the needle; this fixture is one of them, byte-faithful to the measured block)"
else no "fix44/45-G3 MUST_PASS broke on the measured r32 bytes: $out"; fi

out=$(f44_g3_run 96 "$f44_sim/g3-r96.log")
if printf '%s' "$out" | grep -q 'G3_GATE=0' \
   && printf '%s' "$out" | grep -qF 'trainable=189235200 (2.40%)' \
   && printf '%s' "$out" | grep -qF 'realized rank == requested rank 96' \
   && printf '%s' "$out" | grep -qF 'frozen-base identity'; then
  ok "fix44/45-G3 MUST_PASS: the MEASURED r96 drill bytes (Total 7,876,634,880 - Trainable 189,235,200 = the same 7,687,399,680) clear with realized rank == requested rank 96 AND the frozen-base identity — the drill geometry stays green, so the two identities tax an honest drill run nothing"
else no "fix44/45-G3 MUST_PASS broke on the measured r96 bytes: $out"; fi

out=$(f44_g3_run 96 "$f44_sim/g3-r32.log")
if printf '%s' "$out" | grep -q 'G3_GATE=1' && printf '%s' "$out" | grep -qF 'REALIZED RANK'; then
  ok "fix44-G3 MUST_FIRE: r32's measured numbers under a REQUESTED rank of 96 fire REALIZED RANK != REQUESTED RANK — the silent-revert signature (config claims 96, optimizer got 32) is now caught at the artifact layer, which the (0.10,10.0) window alone can never see — both r32 and r96 sit comfortably inside it, which is precisely why rank needs its own integer identity"
else no "fix44-G3 MUST_FIRE (rank) FAILED: the silent-revert fixture gave: $out"; fi

out=$(f44_g3_run 32 "$f44_sim/g3-hf.log")
if printf '%s' "$out" | grep -q 'G3_GATE=1' && printf '%s' "$out" | grep -qF 'HuggingFace PEFT single-line'; then
  ok "fix44-G3 MUST_FIRE: the pre-fix44 HF-format census line FAILS LOUDLY as a stack change to re-derive — the old needle's format is a named arm, never an accepted second format (this leg RED-proves #76's false alarm cannot silently come back)"
else no "fix44-G3 MUST_FIRE (HF format) FAILED: the legacy needle fixture gave: $out"; fi

out=$(f44_g3_run 32 "$f44_sim/g3-zero.log")
if printf '%s' "$out" | grep -q 'G3_GATE=1' && printf '%s' "$out" | grep -qF 'ZERO trainable params'; then
  ok "fix44-G3 MUST_FIRE: a zero-trainable census still fires the classic silent-LoRA arm (the zero-check survives the re-anchoring verbatim)"
else no "fix44-G3 MUST_FIRE (zero) FAILED: $out"; fi

out=$(f44_g3_run 32 "$f44_sim/g3-window.log")
if printf '%s' "$out" | grep -q 'G3_GATE=1' && printf '%s' "$out" | grep -qF 'outside (0.10,10.0)'; then
  ok "fix44-G3 MUST_FIRE: an UNFROZEN-BASE fixture (pct=99.99 beside a rank-perfect count, so ONLY the window arm can fire) trips the (0.10,10.0) guard — the window stays load-bearing at the one job it claims, and the fixture is the real failure class (base not frozen prints ~100%), not a number picked to sit just outside an edge"
else no "fix44-G3 MUST_FIRE (window) FAILED: $out"; fi

out=$(f44_g3_run 32 "$f44_sim/g3-partial.log")
if printf '%s' "$out" | grep -q 'G3_GATE=1' && printf '%s' "$out" | grep -qF 'ambiguous'; then
  ok "fix44/45-G3 MUST_FIRE: 1 of the 3 measured lines (a half-written log or one-line format drift) lands on the named ambiguous arm with all three distinct-value counts — never laundered into 'missing' or 'found'"
else no "fix44/45-G3 MUST_FIRE (partial) FAILED: $out"; fi

# fix45: the frozen-base identity's own MUST_FIRE, measured through the
# REAL G3 span. Fail-before: RED on the pre-fix45 tree (nothing computes a
# frozen-base value there); construction is non-vacuous by arm isolation
# (the zero-arm cannot fire — trainable is the rank-perfect 63,078,400 —
# and the pct is interior, so ONLY the frozen-base arm can produce GATE=1).
out=$(f44_g3_run 32 "$f44_sim/g3-basewalk.log")
if printf '%s' "$out" | grep -q 'G3_GATE=1' && printf '%s' "$out" | grep -qF 'frozen-base identity'; then
  ok "fix45-G3 MUST_FIRE: a census whose Total no longer reconciles with its Trainable at the rank-invariant frozen base (Total - Trainable = 7,687,399,681 != 7,687,399,680; window AND rank identity both intact, so only the frozen-base arm can fire) trips the identity — the frozen base is now an exact integer identity with a two-measurement provenance, not a percentage hunch"
else no "fix45-G3 MUST_FIRE (frozen-base) FAILED: $out"; fi

echo "== fix44: python call-site census — the complement of the executor census (#77-B1, §4) =="
# The executor census counts USES of run_in_container; it structurally cannot
# see a python invocation that bypasses the executor — which is exactly what
# fs_live_save_gate was, and exactly why #77-B1 reached hardware with a green
# contract suite. This is the missing complement: count python3 in command
# position over the comment-stripped view (the shared machinery, immune to
# the comments, the heredocs, and the 'Training command' echo by
# construction) and require the population to be EXACTLY the enumerated host
# exceptions — host calls that are legitimate because they are torch-free:
#   cfg_get(1): reads config.json text.
#   config-identity(1): reads config.json text.
#   resolved-train-config writer(1): writes the gate's JSON.
#   manifest emitter(1): FoundationScale tool, no torch import.
# The fifth python call — tools/live_save_gate.py, the one that DOES import
# torch — must exist ONLY inside an executor payload, which the negative
# conjunct asserts. Fail-before on tonight's tree: count is 5 and the
# negative conjunct hits, so both legs are RED there.
f44_python_census_ok() { # $1=launcher file (real or doctored copy)
  local lc n
  lc=$(strip_shell_comments < "$1")
  n=$(printf '%s\n' "$lc" | grep -cE "$(pos_pat python3)" || true)
  [ "$n" -eq 4 ] \
    && printf '%s\n' "$lc" | grep -qF 'python3 - "$HF_MODEL_PATH/config.json" "$1"' \
    && printf '%s\n' "$lc" | grep -qF 'python3 - "$HF_MODEL_PATH/config.json" <<' \
    && printf '%s\n' "$lc" | grep -qF 'python3 - "$RESOLVED_CFG"' \
    && printf '%s\n' "$lc" | grep -qF 'python3 "$FS_ROOT/tools/emit_run_manifest.py"' \
    && ! printf '%s\n' "$lc" | grep -E "$(pos_pat python3)" | grep -qF 'live_save_gate'
}
f44_py_n=$(strip_shell_comments < "$LORA" | grep -cE "$(pos_pat python3)" || true)
if f44_python_census_ok "$LORA"; then
  ok "python call sites: $f44_py_n command-position python3 sites, ALL enumerated host exceptions (cfg_get(1)+config-identity(1)+resolved-config-writer(1)+manifest-emitter(1) — each torch-free by its own evidence); zero python3 command-position lines touch live_save_gate (the torch-importing call is executor-routed). Complement of the executor census, 4 of 4 exceptions named — an unenumerated python call is red, never an absence"
else
  no "python call-site census failed: $f44_py_n command-position python3 sites (required 4, all enumerated) or a command-position python3 line touches live_save_gate — an unenumerated host python call exists, and an unenumerated exception is the two-interpreter defect (#77-B1)"
fi
# MUST_FIRE (doctrine 3): re-inject a host-routed gate call on a temp COPY —
# the exact #77-B1 shape — and require (i) construction proven (the copy's
# count moves 4 -> 5 and the injected line names live_save_gate), (ii) the
# SAME predicate reports the copy NOT ok, (iii) the live launcher still
# satisfies the predicate (the leg above), so the flip is constructed, not
# ambient.
f44_pt=$(mktemp "${TMPDIR:-/tmp}/fs-f44-pycensus.XXXXXX") \
  && awk '/^fs_live_save_gate\(\) \{/ && !d {d=1; print; print "  python3 \"$FS_ROOT/tools/live_save_gate.py\" \"$1\"  # fix44-MUST-FIRE injection"; next} {print}' "$LORA" > "$f44_pt"
f44_ps=$?
f44_pn=-1
[ "$f44_ps" -eq 0 ] && f44_pn=$(strip_shell_comments < "$f44_pt" | grep -cE "$(pos_pat python3)" || true)
f44_pfired=1
if [ "$f44_ps" -eq 0 ] && [ "$f44_pn" -eq 5 ] && ! f44_python_census_ok "$f44_pt" \
   && f44_python_census_ok "$LORA"; then
  f44_pfired=0
fi
[ -n "${f44_pt:-}" ] && rm -f "$f44_pt" || true
if [ "$f44_pfired" -eq 0 ]; then
  ok "MUST_FIRE python-census: re-injecting a host-routed live_save_gate call on a copy moves the census 4 -> 5 and turns the predicate red — the complement census can see the #77-B1 shape it exists to refuse"
else
  no "MUST_FIRE UNREACHABLE (python-census): the host-call injection did not construct or did not fire (awk rc=$f44_ps, copy count=$f44_pn) — the complement census is an unproven detector"
fi

echo "== fix44: the artifact gate is executor-routed and returns the gate's rc UNTOUCHED (#77-B1) =="
# Fail-before: all three legs are RED on tonight's tree (the function body
# invokes host python3 directly and knows nothing of run_in_container).
f44_gate_wired_ok() { # $1=launcher file (real or doctored copy). The
                      # negative conjunct is the complement census narrowed to
                      # one function: no python3 in command position inside
                      # fs_live_save_gate — the payload's python3 lives inside
                      # a quoted string and is invisible to pos_pat, so the
                      # conjunct holds exactly when the call is routed.
  local fn
  fn=$(sed -n '/^fs_live_save_gate() {/,/^}/p' "$1" | strip_shell_comments)
  printf '%s\n' "$fn" | grep -qF 'run_in_container --slurm-ntasks 1 --workdir "$REPO"' \
    && printf '%s\n' "$fn" | grep -qF "PYTHONPATH='\$FS_ROOT/src'" \
    && printf '%s\n' "$fn" | grep -qF 'PYTHONNOUSERSITE=1' \
    && printf '%s\n' "$fn" | grep -qF 'return "$fs_gate_rc"' \
    && ! printf '%s\n' "$fn" | grep -qE "$(pos_pat python3)"
}
if f44_gate_wired_ok "$LORA"; then
  ok "fs_live_save_gate routes through run_in_container (--slurm-ntasks 1 --workdir \$REPO idiom), prepends \$FS_ROOT/src to the CONTAINER's forwarded PYTHONPATH, restates PYTHONNOUSERSITE=1 in the payload, returns the captured rc, and carries no command-position python3 — the fifth python call joins the executor"
else
  no "fs_live_save_gate is not executor-routed with the established in-container PYTHONPATH/PYTHONNOUSERSITE idiom and an untouched rc — the torch-importing gate still runs on a host interpreter that cannot read DCP (#77-B1)"
fi
# MUST_FIRE for the wiring leg: doctor the executor call back to a bare
# python3 on a COPY (the #77-B1 shape), prove construction (the executor
# idiom count falls to 0 inside the copy's function), and require the SAME
# predicate to go red while the live file stays green.
f44_gt=$(mktemp "${TMPDIR:-/tmp}/fs-f44-gatewire.XXXXXX") \
  && sed 's/run_in_container --slurm-ntasks 1/python3/' "$LORA" > "$f44_gt"
f44_gs=$?
f44_gw=-1
[ "$f44_gs" -eq 0 ] && f44_gw=$(sed -n '/^fs_live_save_gate() {/,/^}/p' "$f44_gt" | grep -cF 'run_in_container --slurm-ntasks 1' || true)
f44_gfired=1
if [ "$f44_gs" -eq 0 ] && [ "$f44_gw" -eq 0 ] && ! f44_gate_wired_ok "$f44_gt" \
   && f44_gate_wired_ok "$LORA"; then
  f44_gfired=0
fi
[ -n "${f44_gt:-}" ] && rm -f "$f44_gt" || true
if [ "$f44_gfired" -eq 0 ]; then
  ok "MUST_FIRE gate-wiring: reverting the executor call to a bare python3 on a copy (construction proven: executor-idiom count in the copy's function = 0) turns the wiring predicate red — #77-B1 cannot silently regress"
else
  no "MUST_FIRE UNREACHABLE (gate-wiring): the bare-python3 revert did not construct or did not fire (sed rc=$f44_gs, copy executor count=$f44_gw) — the wiring leg is an unproven detector"
fi
# rc passthrough (the #72 lesson): the function's contract is 'returns the
# gate's rc UNTOUCHED', and a layer was just added between gate and rc — so
# PROVE it, never assert it. The REAL extracted function is evaled against a
# stubbed executor returning each contract rc, and the function's own rc plus
# its redirect-created capture file are observed. 4 of 4 rc classes sampled:
# 0 (CLEAR), 1 (BLOCKED), 3 (the multiplexed class the mapper decodes), 127
# (tool vanished).
f44_gate_fn=$(sed -n '/^fs_live_save_gate() {/,/^}/p' "$LORA")
f44_rc_out=$( (
  FS_ROOT=/f44/fsroot REPO=/f44/repo HF_MODEL_PATH=/f44/hf RESOLVED_CFG=/f44/cfg
  run_in_container() { return "${1:-0}"; }
  eval "$f44_gate_fn"
  for stub in 0 1 3 127; do
    run_in_container() { return "${F44_STUB_RC:-0}"; }
    F44_STUB_RC=$stub
    fs_live_save_gate /f44/ck save "$f44_sim/rep-$stub.json" "$f44_sim/cap-$stub"
    r=$?
    echo "MAP_$stub=$r"
    [ -f "$f44_sim/cap-$stub" ] && echo "CAP_$stub=present"
  done
) 2>&1 )
if printf '%s' "$f44_rc_out" | grep -q 'MAP_0=0' \
   && printf '%s' "$f44_rc_out" | grep -q 'MAP_1=1' \
   && printf '%s' "$f44_rc_out" | grep -q 'MAP_3=3' \
   && printf '%s' "$f44_rc_out" | grep -q 'MAP_127=127' \
   && [ "$(printf '%s' "$f44_rc_out" | grep -c 'CAP_.*=present' || true)" -eq 4 ]; then
  ok "fs_live_save_gate returns the gate's rc UNTOUCHED through the executor (4 of 4 sampled rcs survive: 0->0 1->1 3->3 127->127) and every call materializes its capture file — the 0/1/3/other contract the mapper decodes is the gate's own, measured through the REAL extracted function, never a paraphrase"
else
  no "fs_live_save_gate does not return the gate's rc untouched through the executor (observed: $(printf '%s' "$f44_rc_out" | tr '\n' ' ')) — a layer between gate and rc is failing the #72 lesson"
fi

[ -n "$f44_sim" ] && rm -rf "$f44_sim" || true

echo "== fix45-A2: full-FT — gate stays routed under the corrected diagnosis, the emitter joins it on proven text, #82 export census, #84 counter hygiene =="
# Fail-before accounting, per this file's convention: on the pre-fix45-A2
# tree ALL 22 legs below are RED (several MUST_FIREs red via their
# construction conjuncts on the pre-fix tree, disclosed). The diagnosis
# fix45-A pinned is corrected on record in the launcher comment and pinned
# by legs 9-11 here: torch IS installed on the host (2.10.0+cpu, ~/.local);
# the launcher's own PYTHONNOUSERSITE=1 hides it; ROUTING — never
# incapacity — is what every leg in this section asserts. NO leg, fixture,
# or assertion below requires "the host cannot read a DCP": on this host,
# without PYTHONNOUSERSITE=1, it can (arm B measured CLEAR on the real
# 99 GB save). A leg asserting incapacity would be a dead control reading
# exactly like a passing control (the hazard this revision exists to
# refuse). The host python call-site census reads 6 pre-fix (the four
# legitimate stdlib sites + the gate + the emitter) where the enumerated
# population is 4; both torch-importing calls must be INVISIBLE to it.

# --- legs 1-2: the python call-site complement census (#77-B1's shape,
#     now population FOUR; discriminator "reads a DCP", not "is on the
#     host") ---
f45_py_census_ok() { # $1=launcher file (real or doctored copy). The FOUR
                     # enumerated host exceptions of the full-FT file, each
                     # stdlib-only on host-plane text inputs and named with
                     # a one-line justification in the launcher's comment
                     # block. fix45-A2: the emitter is REMOVED from this
                     # list — with --full-ft it reads the base DCP and is
                     # executor-routed; both torch-importing python calls
                     # (gate and emitter) must be INVISIBLE to this census.
  local lc n
  lc=$(strip_shell_comments < "$1")
  n=$(printf '%s\n' "$lc" | grep -cE "$(pos_pat python3)" || true)
  [ "$n" -eq 4 ] \
    && printf '%s\n' "$lc" | grep -qF 'python3 - "$HF_MODEL/config.json" "$1"' \
    && printf '%s\n' "$lc" | grep -qF 'python3 - "$f" <<' \
    && printf '%s\n' "$lc" | grep -qF 'python3 - "$RESOLVED_CFG"' \
    && printf '%s\n' "$lc" | grep -qF 'python3 - "$OUT_DIR/checkpoints" "$FQN_MAP"' \
    && ! printf '%s\n' "$lc" | grep -E "$(pos_pat python3)" | grep -qF 'live_save_gate' \
    && ! printf '%s\n' "$lc" | grep -E "$(pos_pat python3)" | grep -qF 'emit_run_manifest'
}
f45_py_n=$(strip_shell_comments < "$FULL" | grep -cE "$(pos_pat python3)" || true)
if f45_py_census_ok "$FULL"; then
  ok "full-FT python call sites: $f45_py_n command-position python3 sites, ALL enumerated host exceptions (cfg_get(1)+schema-spot-check(1)+resolved-config-writer(1)+fqn-map-materializer(1) — each stdlib-only on host-plane text; the discriminator is 'reads a DCP', never 'is on the host'); zero command-position python3 lines touch live_save_gate or emit_run_manifest (both torch-importing calls are executor-routed). Complement of the executor census, 4 of 4 exceptions named — an unenumerated python call is red, never an absence"
else
  no "full-FT python call-site census failed: $f45_py_n command-position python3 sites (required 4, all enumerated) or a command-position python3 line touches live_save_gate/emit_run_manifest — an unenumerated host python call exists, the #77-B1 / emitter-site class: a torch-importing call adjudicating or producing DCP-plane artifacts outside the stack that writes them. The refusal of the '~/.local torch is RIGHT THERE' shortcut is on record in the launcher comment (the host CAN import torch — routing is the pinned property, not capability)"
fi
# MUST_FIRE (doctrine 3): re-inject the measured #77-B1 shape — a
# host-routed gate call — into a temp COPY of the full-FT launcher, and
# require (i) construction proven (the copy's census moves 4 -> 5), (ii)
# the SAME predicate reports the copy NOT ok, (iii) the live launcher still
# satisfies the predicate (leg 1), so the flip is constructed, not ambient.
f45_pt=$(mktemp "${TMPDIR:-/tmp}/fs-f45-pycensus.XXXXXX") \
  && awk '/^fs_live_save_gate\(\) \{/ && !d {d=1; print; print "  python3 \"$FS_ROOT/tools/live_save_gate.py\" \"$1\"  # fix45-MUST-FIRE injection"; next} {print}' "$FULL" > "$f45_pt"
f45_ps=$?
f45_pn=-1
[ "$f45_ps" -eq 0 ] && f45_pn=$(strip_shell_comments < "$f45_pt" | grep -cE "$(pos_pat python3)" || true)
f45_pfired=1
if [ "$f45_ps" -eq 0 ] && [ "$f45_pn" -eq 5 ] && ! f45_py_census_ok "$f45_pt" \
   && f45_py_census_ok "$FULL"; then
  f45_pfired=0
fi
[ -n "${f45_pt:-}" ] && rm -f "$f45_pt" || true
if [ "$f45_pfired" -eq 0 ]; then
  ok "MUST_FIRE full-FT python-census: re-injecting a host-routed live_save_gate call on a copy moves the census 4 -> 5 and turns the predicate red — the complement census can see the shape it exists to refuse"
else
  no "MUST_FIRE UNREACHABLE (full-FT python-census): the host-call injection did not construct or did not fire (awk rc=$f45_ps, copy count=$f45_pn) — the complement census is an unproven detector"
fi

# --- legs 3-4: the gate wiring itself (conjuncts unchanged from fix45-A —
#     they were always ROUTING assertions; the message now says so) ---
f45_gate_wired_ok() { # $1=launcher file (real or doctored copy). The
                      # negative conjunct is the complement census narrowed
                      # to one function: no python3 in command position inside
                      # fs_live_save_gate — the payload's python3 lives
                      # inside a quoted string and is invisible to pos_pat,
                      # so the conjunct holds exactly when the call is routed.
                      # fix45-A2: every conjunct here is a routing/hygiene
                      # assertion; none asserts host incapacity.
  local fn
  fn=$(sed -n '/^fs_live_save_gate() {/,/^}/p' "$1" | strip_shell_comments)
  printf '%s\n' "$fn" | grep -qF 'run_in_container --slurm-ntasks 1 --workdir "$REPO"' \
    && printf '%s\n' "$fn" | grep -qF "PYTHONPATH='\$FS_ROOT/src'" \
    && printf '%s\n' "$fn" | grep -qF 'PYTHONNOUSERSITE=1' \
    && printf '%s\n' "$fn" | grep -qF -- '--fqn-map' \
    && printf '%s\n' "$fn" | grep -qF 'live_gate wall-clock budget exhausted' \
    && printf '%s\n' "$fn" | grep -qF 'return "$fs_gate_rc"' \
    && ! printf '%s\n' "$fn" | grep -qE "$(pos_pat python3)"
}
if f45_gate_wired_ok "$FULL"; then
  ok "fs_live_save_gate (full-FT) routes through run_in_container (--slurm-ntasks 1 --workdir \$REPO idiom), prepends \$FS_ROOT/src to the CONTAINER's forwarded PYTHONPATH, restates PYTHONNOUSERSITE=1 payload-scoped, passes --fqn-map, carries the wall-clock bound, returns the captured rc, and carries no command-position python3 — the gate adjudicates with the same torch stack that wrote the save, with the cheap host alternative (unset PYTHONNOUSERSITE, arm B's CPU-only 2.10.0) refused on record in the launcher comment"
else
  no "fs_live_save_gate (full-FT) is not executor-routed with the established in-container PYTHONPATH/PYTHONNOUSERSITE/fqn-map/wall-clock/untouched-rc idiom — the torch-importing gate adjudicates outside the stack that wrote the artifact (#77-B1; corrected diagnosis: routing, never incapacity — measured arm B says the host CAN read the save, and the three refused reasons say why it must not)"
fi
# The doctoring is comment-blind-proof by construction (fix26b /
# finding #64 lineage — the same trap that bit here at fix45-A2, measured:
# the gate's comment block mentions run_in_container in prose, and a
# raw-text awk rewrote the word inside a COMMENT, leaving the real
# executor call untouched and the leg honestly UNREACHABLE). The target
# line is CHOSEN on the strip_shell_comments view (line-count-preserving
# and quoting-aware), then EDITED AT THAT LINE NUMBER in the ORIGINAL, so
# the copy stays byte-identical to the real launcher apart from the one
# intended mutation — which matters, because the copy is then fed to the
# REAL predicate. The construction proof is counted on the stripped view
# too, so a surviving comment mention can never masquerade as an
# un-reverted call.
f45_first_real_gate_executor_line() { # $1=launcher file (real or copy);
  # stdout=1-based offset INSIDE fs_live_save_gate of the first NON-COMMENT
  # line mentioning run_in_container; empty stdout iff none exists.
  sed -n '/^fs_live_save_gate() {/,/^}/p' "$1" \
    | strip_shell_comments \
    | awk 'index($0, "run_in_container") { print NR; exit }'
}
f45_revert_gate_executor() { # $1=src, $2=dst. rc 0 iff the doctored copy
  # was constructed: the one real executor call reverted to bare python3
  # AND 0 non-comment run_in_container left inside the copy's
  # fs_live_save_gate, counted on the stripped view. rc 1 = COULD NOT
  # CONSTRUCT (no real executor line, awk failure, or the count refusing
  # 0): the mutator reports failure instead of doctoring a comment and
  # claiming success. The proof is INTERNAL so both callers — the leg and
  # the mutator control below — exercise this one code path, never a
  # re-implementation of it.
  local off
  off=$(f45_first_real_gate_executor_line "$1")
  [ -n "$off" ] || return 1
  awk -v off="$off" '/^fs_live_save_gate\(\) \{/{f=1} f{n+=1; if (n==off) sub(/run_in_container/, "python3")} {print}' "$1" > "$2" || return 1
  [ "$(sed -n '/^fs_live_save_gate() {/,/^}/p' "$2" | strip_shell_comments | grep -cF 'run_in_container' || true)" -eq 0 ]
}
f45_gt=$(mktemp "${TMPDIR:-/tmp}/fs-f45-gatewire.XXXXXX") || f45_gt=""
f45_gs=1
f45_gw=-1
if [ -n "$f45_gt" ] && f45_revert_gate_executor "$FULL" "$f45_gt"; then
  f45_gs=0
  f45_gw=$(sed -n '/^fs_live_save_gate() {/,/^}/p' "$f45_gt" | strip_shell_comments | grep -cF 'run_in_container' || true)
fi
f45_gfired=1
if [ "$f45_gs" -eq 0 ] && [ "$f45_gw" -eq 0 ] && ! f45_gate_wired_ok "$f45_gt" \
   && f45_gate_wired_ok "$FULL"; then
  f45_gfired=0
fi
[ -n "${f45_gt:-}" ] && rm -f "$f45_gt" || true
if [ "$f45_gfired" -eq 0 ]; then
  ok "MUST_FIRE full-FT gate-wiring: reverting the one real (non-comment) executor call to a bare python3 on a copy (construction proven: 0 non-comment run_in_container left in the copy's function, counted on the comment-stripped view) turns the wiring predicate red while the real launcher stays green — #77-B1 cannot silently regress, whatever the prevailing diagnosis says about WHY it must not"
else
  no "MUST_FIRE UNREACHABLE (full-FT gate-wiring): the bare-python3 revert did not construct or did not fire (mutator rc=$f45_gs, non-comment executor count on copy=$f45_gw) — the wiring leg is an unproven detector"
fi

# --- leg 4b (second-order; inserted between the fix45-A2 legs 4 and 5, so
#     the numeric "--- legs N" comment headers below still read as
#     shipped): this guards the DOCTORING (f45_revert_gate_executor), NOT
#     the launcher. Fixture: a copy of $FULL with the one REAL executor
#     line deleted (located on the stripped view) while the comment-prose
#     mentions of run_in_container are left armed. The mutator must report
#     COULD NOT CONSTRUCT on it; a comment-blind mutator (raw first-match —
#     the exact shape measured broken at fix45-A2; fix26b/finding-#64
#     class) would doctor a COMMENT, and whether it then claims success
#     depends on whether its construction proof is also comment-aware. The
#     pinned contract here is unconditional: on input whose only mentions
#     are comments, construction MUST be refused. Denominators are asserted
#     before the mutator even runs — the fixture must still carry >=1 raw
#     (comment) mention AND 0 non-comment ones — else this control could
#     pass vacuously (doctrine 1, doctrine 3). The fixture shares only the
#     line-FINDER with the mutator; the behaviour under test is the
#     production revert+proof itself. ---
f45_gx=$(mktemp "${TMPDIR:-/tmp}/fs-f45-gatenoexec.XXXXXX") || f45_gx=""
f45_gxd=""
f45_gxoff=""
f45_gxraw=-1
f45_gxstrip=-1
f45_gxrc=-1
f45_gxfired=1
if [ -n "$f45_gx" ]; then
  f45_gxoff=$(f45_first_real_gate_executor_line "$FULL")
  if [ -n "$f45_gxoff" ] && awk -v off="$f45_gxoff" '/^fs_live_save_gate\(\) \{/{f=1} f{n+=1; if (n==off) next} {print}' "$FULL" > "$f45_gx"; then
    f45_gxraw=$(sed -n '/^fs_live_save_gate() {/,/^}/p' "$f45_gx" | grep -cF 'run_in_container' || true)
    f45_gxstrip=$(sed -n '/^fs_live_save_gate() {/,/^}/p' "$f45_gx" | strip_shell_comments | grep -cF 'run_in_container' || true)
    if [ "$f45_gxraw" -gt 0 ] && [ "$f45_gxstrip" -eq 0 ]; then
      f45_gxd=$(mktemp "${TMPDIR:-/tmp}/fs-f45-gatectl.XXXXXX") || f45_gxd=""
      if [ -n "$f45_gxd" ]; then
        if f45_revert_gate_executor "$f45_gx" "$f45_gxd"; then
          f45_gxrc=0
        else
          f45_gxrc=1
          f45_gxfired=0
        fi
      fi
    fi
  fi
fi
[ -n "${f45_gx:-}" ] && rm -f "$f45_gx" || true
[ -n "${f45_gxd:-}" ] && rm -f "$f45_gxd" || true
if [ "$f45_gxfired" -eq 0 ]; then
  ok "MUST_FIRE mutator control (gate-wiring doctoring): with the real executor line deleted but $f45_gxraw comment-prose run_in_container mention(s) still armed in the copy, f45_revert_gate_executor reports COULD NOT CONSTRUCT — the mutator provably refuses the comment trap (fix26b / finding #64 class), so the gate-wiring leg's green cannot be a comment-edit masquerading as a revert"
else
  no "MUST_FIRE UNREACHABLE (mutator control, gate-wiring doctoring): the no-executor fixture did not build or the mutator failed to refuse it (executor-line offset=${f45_gxoff:-none}, raw mentions left=$f45_gxraw, non-comment mentions left=$f45_gxstrip, mutator rc=$f45_gxrc) — the doctoring step is an unproven mutator, and a detector never observed to refuse is a dead control reading exactly like a passing one"
fi

# --- legs 5-6: rc passthrough measured through the REAL extracted function,
# plus the watchdog's MUST_FIRE (unchanged from fix45-A; accepted as
# shipped). The passthrough leg mirrors the fix44 LoRA-side leg
# sample-for-sample (the #72 lesson: a layer was added between gate and rc,
# so prove the rc, never assert it); the timeout leg is the doctrine-3
# control for the wall-clock bound.
f45_gate_fn=$(sed -n '/^fs_live_save_gate() {/,/^}/p' "$FULL")
if [ -n "$f45_gate_fn" ]; then
  f45_gate_sim=$(mktemp -d "${TMPDIR:-/tmp}/fs-f45-gate.XXXXXX" 2>/dev/null) || f45_gate_sim=""
  [ -n "$f45_gate_sim" ] || { f45_gate_sim="${TMPDIR:-/tmp}/fs-f45-gate.$$"; mkdir -p "$f45_gate_sim" 2>/dev/null || f45_gate_sim=""; }
  f45_rc_out=$( (
    FS_ROOT=/f45/fsroot REPO=/f45/repo HF_MODEL=/f45/hf RESOLVED_CFG=/f45/cfg FQN_MAP=/f45/map
    eval "$f45_gate_fn"
    for stub in 0 1 3 127; do
      run_in_container() { return "${F45_STUB_RC:-0}"; }
      F45_STUB_RC=$stub
      fs_live_save_gate /f45/ck save "$f45_gate_sim/rep-$stub.json" "$f45_gate_sim/cap-$stub"
      echo "MAP_$stub=$?"
      [ -f "$f45_gate_sim/cap-$stub" ] && echo "CAP_$stub=present"
      grep -qF 'live_gate wall-clock budget exhausted' "$f45_gate_sim/cap-$stub" 2>/dev/null && echo "MARKER_$stub=unexpected"
    done
    run_in_container() { sleep 5; return 0; }
    FS_GATE_TIMEOUT_S=1
    fs_live_save_gate /f45/ck save "$f45_gate_sim/rep-timeout.json" "$f45_gate_sim/cap-timeout"
    echo "MAP_TIMEOUT=$?"
    [ -f "$f45_gate_sim/cap-timeout" ] \
      && grep -qF 'live_gate wall-clock budget exhausted' "$f45_gate_sim/cap-timeout" \
      && echo "MARKER_TIMEOUT=present"
  ) 2>&1 )
  if printf '%s' "$f45_rc_out" | grep -q 'MAP_0=0' \
     && printf '%s' "$f45_rc_out" | grep -q 'MAP_1=1' \
     && printf '%s' "$f45_rc_out" | grep -q 'MAP_3=3' \
     && printf '%s' "$f45_rc_out" | grep -q 'MAP_127=127' \
     && [ "$(printf '%s' "$f45_rc_out" | grep -c 'CAP_.*=present' || true)" -eq 4 ] \
     && ! printf '%s' "$f45_rc_out" | grep -q 'MARKER_.*=unexpected'; then
    ok "fs_live_save_gate (full-FT) returns the gate's rc UNTOUCHED through the executor (4 of 4 sampled rcs survive: 0->0 1->1 3->3 127->127), materializes every capture file, and its watchdog stays silent inside the budget — measured through the REAL extracted function, mirroring the fix44 LoRA-side leg; the 0->0 leg is the positive control"
  else
    no "fs_live_save_gate (full-FT) does not return the gate's rc untouched through the executor (observed: $(printf '%s' "$f45_rc_out" | tr '\n' ' ')) — a layer between gate and rc is failing the #72 lesson"
  fi
  if printf '%s' "$f45_rc_out" | grep -q 'MAP_TIMEOUT=124' \
     && printf '%s' "$f45_rc_out" | grep -q 'MARKER_TIMEOUT=present'; then
    ok "MUST_FIRE bounded wait: an executor that outlives FS_GATE_TIMEOUT_S=1 is TERM/KILL'ed, mints rc 124, and records the cause in the capture — a wedged live gate can never silently stall tripwires (a)-(c) (an arm never observed to fire is not a control, doctrine 3)"
  else
    no "MUST_FIRE UNREACHABLE (bounded wait): the over-budget stub did not mint 124 with the marker (observed: $(printf '%s' "$f45_rc_out" | tr '\n' ' ')) — the wall-clock bound is an unproven detector and the watcher-stall hole is theoretically open"
  fi
  [ -n "$f45_gate_sim" ] && rm -rf "$f45_gate_sim" || true
else
  no "MUST_FIRE UNREACHABLE: fs_live_save_gate not extractable from the full-FT launcher — the rc-passthrough and bounded-wait legs are unproven"
fi

# --- legs 7-8: watcher/epilogue wiring — captures exist, the mapper is
# 4-arg, the launcher-minted 3 bypasses the mapper, the disarm names the
# cause from the record (unchanged from fix45-A; accepted as shipped).
f45_watcher_ok() { # $1=launcher file (real or doctored copy)
  local lc
  lc=$(strip_shell_comments < "$1")
  printf '%s\n' "$lc" | grep -qF 'fs_live_save_gate "$FS1_CKPT" first_save "$FS1_REPORT" "$FS1_CAPTURE"' \
    && printf '%s\n' "$lc" | grep -qF 'fs_live_save_gate "$FINAL_CKPT" save "$FINAL_REPORT" "$FINAL_CAPTURE"' \
    && printf '%s\n' "$lc" | grep -qF 'fs_gate_verdict_to_rc "$FS_GATE_RC" "final save (iter $LAST)" "$FINAL_REPORT" "$FINAL_CAPTURE"' \
    && printf '%s\n' "$lc" | grep -qF 'fs_gate_refusal_class "$FS1_REPORT"' \
    && printf '%s\n' "$lc" | grep -qF 'launcher-minted, NOT a tool refusal' \
    && [ "$(printf '%s\n' "$lc" | grep -cF 'fs_gate_verdict_to_rc "' || true)" -eq 1 ]
}
if f45_watcher_ok "$FULL"; then
  ok "full-FT watcher/epilogue: both gate invocations carry capture files, the epilogue mapper call is the single 4-arg call, the launcher-minted UNMEASURED names itself as minted (never fed to the evidence-demanding mapper), and the live disarm reads the cause from the tool's refusal record — the #77-B2/B3 port is wired at both call sites"
else
  no "full-FT watcher/epilogue wiring drifted — a gate invocation without a capture, a 3-arg mapper call reappearing, the minted-3 bypass lost, or a second mapper call appeared"
fi
f45_wt=$(mktemp "${TMPDIR:-/tmp}/fs-f45-watcher.XXXXXX") \
  && sed 's/fs_live_save_gate "$FS1_CKPT" first_save "$FS1_REPORT" "$FS1_CAPTURE"/fs_live_save_gate "$FS1_CKPT" first_save "$FS1_REPORT"/' "$FULL" > "$f45_wt"
f45_ws=$?
f45_wfired=1
if [ "$f45_ws" -eq 0 ] \
   && ! grep -qF 'fs_live_save_gate "$FS1_CKPT" first_save "$FS1_REPORT" "$FS1_CAPTURE"' "$f45_wt" \
   && ! f45_watcher_ok "$f45_wt" && f45_watcher_ok "$FULL"; then
  f45_wfired=0
fi
[ -n "${f45_wt:-}" ] && rm -f "$f45_wt" || true
if [ "$f45_wfired" -eq 0 ]; then
  ok "MUST_FIRE watcher-wiring: reverting the live invocation to the 3-arg shape on a copy (construction proven: the 4-arg needle absent) turns the wiring predicate red — the capture contract cannot silently regress"
else
  no "MUST_FIRE UNREACHABLE (watcher-wiring): the 3-arg revert did not construct or did not fire (sed rc=$f45_ws) — the wiring leg is an unproven detector"
fi

# --- legs 9-11: BOTH refutions on record — the refuted JUSTIFICATION
# paragraph, the arm-A measurement, the corrected discriminator, and the
# refusal of the cheap fix. Greps against prose by design — the artifact
# under test IS prose — and every needle here is pinned against COMMENT
# prose, which makes every needle hostage to re-wrapping: grep -F is
# line-oriented, so a needle split across two comment lines can never
# match ("a needle containing a newline can never match"), and that is
# exactly how this leg shipped permanently red on the intact launcher —
# measured: 'REFUSED here, on record' straddles the wrap between the
# 'is REFUSED' line and the 'here, on record' line (launcher ~753-754 as
# of this fix; the four sibling needles measured 1/1 each). The stream
# below keeps every raw line (code-position and single-line matches still
# count, exactly as before) and ADDS one folded line per contiguous
# comment block, so a re-wrap of the refusal stays green while a genuine
# deletion or rewording stays red. Blind spot, unchanged from before:
# a rewording that keeps all five substrings verbatim (e.g. deleting the
# three reasons but keeping the REFUSED sentence) stays green — pinning
# the reasons would be a new claim needing new controls legs 10/11 do
# not provide. The dead-control audit of this detector, decided: the
# erased-needle MUST_FIRE (leg 10) still fires for the right reason under
# the corrected diagnosis, because the needle it protects is the arm-A
# MEASURED refusal text, which remains true and on record (arm A rc=3 was
# measured, not alleged). What the old version of this leg could not see —
# a reversion of the CORRECTION with the measurement left intact — is now
# covered by the RELABEL MUST_FIRE (leg 11), which constructs exactly the
# dead-control shape: the old diagnosis restored next to its own refuting
# table.
f45_record_stream() { # $1=file -> stdout: every raw line, PLUS one folded
                      # line per contiguous comment block (leading
                      # "spaces, #, optional one space" stripped, trailing
                      # spaces stripped, block lines joined by single
                      # spaces). A folded match is new evidence ONLY
                      # within one block — nothing can be fabricated
                      # across two separated comment regions. Space-only
                      # patterns on purpose (no [[:space:]], no \t): a
                      # tab-indented comment degrades to raw-only
                      # matching, never worse than the old raw grep.
                      # POSIX awk; no getline.
  awk '
    /^ *#/ {
      line = $0
      sub(/^ *# ?/, "", line)
      sub(/ *$/, "", line)
      if (inblock) { buf = buf " " line } else { buf = line; inblock = 1 }
      print
      next
    }
    {
      if (inblock) { print buf; inblock = 0 }
      print
      next
    }
    END { if (inblock) print buf }
  ' "$1"
}
f45_record_grep() { # $1=fixed needle, $2=file -> needle present in the raw
                    # OR folded view (grep -F: the needle is literal). The
                    # MUST_FIRE construction proofs at legs 10/11 measure
                    # through this same view — see the same-view rule
                    # written down at the #84 counter below.
  f45_record_stream "$2" | grep -qF "$1"
}
f45_record_ok() { # $1=launcher file -> both refuted texts AND the
                  # correction AND the refusal are on record, AND the
                  # gate refusal block itself still stands. All six
                  # needles are comment-pinned prose, so all six go
                  # through the wrap-robust stream — not just the ones
                  # that straddle today's wraps. Denominators, on
                  # record: the DCP refusal text deliberately has TWO
                  # legitimate record sites (the arm-A row at the gate
                  # and the emitter's quote of the same measured error),
                  # so it reds only when the text is gone from BOTH;
                  # the sixth needle, "Three arms, one extracted
                  # function, one artifact", is the signature of the
                  # refusal block alone and itself straddles a wrap
                  # (matches only through the fold), so deleting the
                  # whole block reds this leg even while the quoted
                  # error survives in the emitter comment. A future
                  # re-wrap of ANY needle must not red this leg; a
                  # genuine removal still must, and legs 10/11 are
                  # exactly the controls that prove it.
  f45_record_grep 'read_metadata is torch-free by design' "$1" \
    && f45_record_grep 'torch.distributed.checkpoint is unavailable; cannot read DCP' "$1" \
    && f45_record_grep 'torch IS installed on the host' "$1" \
    && f45_record_grep 'the discriminator is PYTHONNOUSERSITE=1' "$1" \
    && f45_record_grep 'REFUSED here, on record' "$1" \
    && f45_record_grep 'Three arms, one extracted function, one artifact' "$1"
}
if f45_record_ok "$FULL"; then
  ok "fix45-A2: the refuted justification AND the refuted diagnosis stand on record beside the measurements that killed them (arm-A refusal text, user-site torch fact, PYTHONNOUSERSITE discriminator) and the refusal of the cheap host fix is written down, with the gate refusal block pinned by its own three-arms signature (the refusal text itself is denominator-2 BY DESIGN: the gate's arm-A row and the emitter's quote are both legitimate record) — documentation of a measured defect and its correction, under active contract"
else
  no "fix45-A2: the measured refusal text, the corrected discriminator, the on-record refusal, or the gate refusal block's three-arms anchor is missing from the full-FT launcher — a measured defect's documentation (and its correction) must not silently revert; a silent reversion here is how #77-B1 survived review once"
fi
# Same-view rule, mutation side: a mutation must be at least as strong
# as the view the predicate reads. f45_record_ok reads the fold-aware
# stream, which rejoins the emitter comment's wrap-straddled quote of
# this same measured error; a line-oriented sed on the full phrase
# misses that site, the needle survives in the folded view, and the
# construction proof below honestly refuses. Substituting the
# load-bearing token under /g erases the needle on every raw line, and
# a folded line is built only from raw lines — so the needle dies in
# every view the predicate can see.
f45_ct=$(mktemp "${TMPDIR:-/tmp}/fs-f45-record.XXXXXX") \
  && sed 's/torch\.distributed\.checkpoint/the host checkpoint reader/g' "$FULL" > "$f45_ct"
f45_cs=$?
f45_cfired=1
if [ "$f45_cs" -eq 0 ] \
   && ! f45_record_grep 'torch.distributed.checkpoint is unavailable' "$f45_ct" \
   && ! f45_record_ok "$f45_ct" && f45_record_ok "$FULL"; then
  f45_cfired=0
fi
[ -n "${f45_ct:-}" ] && rm -f "$f45_ct" || true
if [ "$f45_cfired" -eq 0 ]; then
  ok "MUST_FIRE record-erasure: erasing the measured refusal text at every site the fold-aware predicate can see it (construction proven: the needle absent from raw AND folded views) turns the record leg red — the measurement is protected because it is MEASURED, not because the old diagnosis needed it"
else
  no "MUST_FIRE UNREACHABLE (record-erasure): the erasure did not construct or did not fire (sed rc=$f45_cs) — leg 9 is an unproven detector"
fi
f45_lt=$(mktemp "${TMPDIR:-/tmp}/fs-f45-relabel.XXXXXX") \
  && sed 's/torch IS installed on the host/torch is absent from the host/' "$FULL" > "$f45_lt"
f45_ls=$?
f45_lfired=1
if [ "$f45_ls" -eq 0 ] \
   && ! f45_record_grep 'torch IS installed on the host' "$f45_lt" \
   && f45_record_grep 'torch is absent from the host' "$f45_lt" \
   && ! f45_record_ok "$f45_lt" && f45_record_ok "$FULL"; then
  f45_lfired=0
fi
[ -n "${f45_lt:-}" ] && rm -f "$f45_lt" || true
if [ "$f45_lfired" -eq 0 ]; then
  ok "MUST_FIRE record-relabel: restoring the refuted 'no torch on the host' diagnosis in place of the correction (construction proven: the correction absent, the old label present) turns the record leg red — the correction itself is under contract, so the comment cannot decay into a dead control asserting incapacity on a host that can"
else
  no "MUST_FIRE UNREACHABLE (record-relabel): the relabel did not construct or did not fire (sed rc=$f45_ls) — leg 9 cannot see a reversion of its own correction"
fi

# --- legs 12-13: the emitter routing — the proven <compute-node> delta, pinned.
f45a2_emit_wired_ok() { # $1=launcher file (real or doctored copy). The
                        # emitter joined the executor for the same reason
                        # the gate did: with --full-ft it imports the DCP
                        # stack (measured: it hard-blocked every historical
                        # launch). Routing is the pinned property here too;
                        # the complement census (leg 1) separately proves no
                        # command-position python3 names it.
  local f=$1 lc
  lc=$(strip_shell_comments < "$f")
  printf '%s\n' "$lc" | grep -qF 'FS_EMIT_ARGS=(' \
    && printf '%s\n' "$lc" | grep -qF "printf '%q ' \"\${FS_EMIT_ARGS[@]}\"" \
    && printf '%s\n' "$lc" | grep -qF "python3 '\$FS_ROOT/tools/emit_run_manifest.py'" \
    && printf '%s\n' "$lc" | grep -qF 'run-manifest emission FAILED' \
    && awk '/run_in_container --slurm-ntasks 1/ && /\\$/ {getline nxt; if (nxt ~ /tools\/emit_run_manifest\.py/) found=1} END{exit !found}' "$f"
}
if f45a2_emit_wired_ok "$FULL"; then
  ok "emit_run_manifest (full-FT) is executor-routed: argv assembled as FS_EMIT_ARGS and flattened through printf '%q' (no hand re-quoting), the payload names the emitter inside the container's quoted layer, the single-CPU executor idiom directly precedes the payload line, and the fail-closed || die stands — the proven <compute-node> shape (declared_fqns=1252 censused from the independent base; the run that reached a GPU at all), pinned"
else
  no "emit_run_manifest (full-FT) is not executor-routed in the proven shape — the DCP-reading emitter is again at the mercy of whatever python the host job env happens to see (the <compute-node> measured hard-block class), or argv quoting was hand-assembled"
fi
f45a2_et=$(mktemp "${TMPDIR:-/tmp}/fs-f45-emitwire.XXXXXX") \
  && awk '{ buf[NR]=$0 } END { for (i=1; i<=NR; i++) { if (buf[i] ~ /run_in_container --slurm-ntasks 1/ && buf[i+1] ~ /tools\/emit_run_manifest\.py/) sub(/run_in_container/, "python3", buf[i]); print buf[i] } }' "$FULL" > "$f45a2_et"
f45a2_es=$?
f45a2_eadj=-1
[ "$f45a2_es" -eq 0 ] && f45a2_eadj=$(awk '/run_in_container --slurm-ntasks 1/ && /\\$/ {getline nxt; if (nxt ~ /tools\/emit_run_manifest\.py/) found=1} END{print (found ? 1 : 0)}' "$f45a2_et")
f45a2_efired=1
if [ "$f45a2_es" -eq 0 ] && [ "$f45a2_eadj" = "0" ] && ! f45a2_emit_wired_ok "$f45a2_et" \
   && f45a2_emit_wired_ok "$FULL"; then
  f45a2_efired=0
fi
[ -n "${f45a2_et:-}" ] && rm -f "$f45a2_et" || true
if [ "$f45a2_efired" -eq 0 ]; then
  ok "MUST_FIRE emitter-wiring: reverting the emitter's executor call to host python on a copy — the exact shape that hard-blocked every historical full-FT launch (construction proven: the executor/payload adjacency absent on the copy, present live) turns the routing predicate red (the complement census, leg 1, also sees this copy — the two detectors share this constructed defect by design, disclosed)"
else
  no "MUST_FIRE UNREACHABLE (emitter-wiring): the host revert did not construct or did not fire (awk rc=$f45a2_es, copy adjacency=$f45a2_eadj) — the emitter routing leg is an unproven detector, and the estate's always-blocked site is unwatched"
fi

# --- legs 14-15: the guard-plane decision (fix45-A2 (a)). The stale
# host-python3 guard asserted a plane the emitter no longer runs on; its
# ABSENCE plus the on-record stated reason is the pinned state. Shallow
# prose-class control, disclosed, same class as legs 9-11.
f45a2_guard_plane_ok() { # $1=launcher file (real or doctored copy)
  ! grep -qF 'host python3 unavailable — cannot emit the run manifest' "$1" \
    && grep -qF 'fix45-A2 (a)' "$1" \
    && grep -qF 'asserted the WRONG plane' "$1"
}
if f45a2_guard_plane_ok "$FULL"; then
  ok "fix45-A2 (a): the wrong-plane host-python3 guard is gone and the decision (no executor availability probe; the first executor call is itself fail-closed before any GPU-second; the two stdlib writers die named on their own plane) is stated where the guard stood — the wrong-plane guard cannot silently come back, and the reasoning cannot be silently dropped"
else
  no "fix45-A2 (a): the wrong-plane guard text reappeared or the stated-reason record is missing — a guard asserting a plane nothing depends on is a dead guard reading like a live one"
fi
f45a2_gt2=$(mktemp "${TMPDIR:-/tmp}/fs-f45-guardplane.XXXXXX") \
  && cat "$FULL" > "$f45a2_gt2" \
  && printf '%s\n%s\n' 'command -v python3 >/dev/null 2>&1 || \' '  die "host python3 unavailable — cannot emit the run manifest; provenance emission is not optional, refusing to launch"' >> "$f45a2_gt2"
f45a2_gs2=$?
f45a2_gfired=1
if [ "$f45a2_gs2" -eq 0 ] \
   && grep -qF 'host python3 unavailable — cannot emit the run manifest' "$f45a2_gt2" \
   && ! f45a2_guard_plane_ok "$f45a2_gt2" && f45a2_guard_plane_ok "$FULL"; then
  f45a2_gfired=0
fi
[ -n "${f45a2_gt2:-}" ] && rm -f "$f45a2_gt2" || true
if [ "$f45a2_gfired" -eq 0 ]; then
  ok "MUST_FIRE guard-plane: re-inserting the stale wrong-plane guard into a copy (construction proven: the needle present on the copy, absent live) turns the decision leg red — the deleted guard cannot re-grow quietly"
else
  no "MUST_FIRE UNREACHABLE (guard-plane): the stale-guard insertion did not construct or did not fire (rc=$f45a2_gs2) — leg 14 is an unproven detector"
fi

# --- legs 16-18: the bind-mount invariant (fix45-A2 (b)) — pinned
# statically, positioned before its first dependent, and EXERCISED through
# the REAL extracted loop (the no-paraphrase rule this file applies to
# every region), with outside-tree rows as doctored inputs it must reject.
f45a2_bind_static_ok() { # $1=launcher file (real or doctored copy)
  local f=$1 g gl el
  g=$(sed -n '/^for _bm in /,/^done$/p' "$f")
  [ -n "$g" ] || return 1
  printf '%s\n' "$g" | grep -qF '"$OUT_DIR" "$FS_ROOT" "$REPO" "$HF_MODEL" "$BASE_CKPT"' \
    && printf '%s\n' "$g" | grep -qF 'bind-mount invariant broken' \
    && printf '%s\n' "$g" | grep -qF '"$HOME"/*' \
    && gl=$(grep -nF 'for _bm in ' "$f" | head -n1 | cut -d: -f1) \
    && el=$(grep -nF "python3 '\$FS_ROOT/tools/emit_run_manifest.py'" "$f" | head -n1 | cut -d: -f1) \
    && [ -n "$gl" ] && [ -n "$el" ] && [ "$gl" -lt "$el" ]
}
f45a2_bind_guard=$(sed -n '/^for _bm in /,/^done$/p' "$FULL")
if [ -n "$f45a2_bind_guard" ]; then
  f45a2_bg=$( (
    die() { echo "DIE:$*"; }
    HOME=/f45a2/home
    echo "[ROW1]"
    OUT_DIR=/f45a2/home/out FS_ROOT=/f45a2/home/fs REPO=/f45a2/home/repo HF_MODEL=/f45a2/home/hf BASE_CKPT=/f45a2/home/base
    eval "$f45a2_bind_guard"
    echo "[ROW2]"
    OUT_DIR=/f45a2/elsewhere
    eval "$f45a2_bind_guard"
    echo "[ROW3]"
    OUT_DIR=/f45a2/home/out; FS_ROOT=/opt/not-home
    eval "$f45a2_bind_guard"
    echo "[END]"
  ) 2>&1 )
  f45a2_row1_dies=$(printf '%s\n' "$f45a2_bg" | sed -n '/\[ROW1\]/,/\[ROW2\]/p' | grep -c 'DIE:' || true)
  f45a2_row2_text=$(printf '%s\n' "$f45a2_bg" | sed -n '/\[ROW2\]/,/\[ROW3\]/p')
  f45a2_row3_text=$(printf '%s\n' "$f45a2_bg" | sed -n '/\[ROW3\]/,/\[END\]/p')
  if f45a2_bind_static_ok "$FULL" && [ "$f45a2_row1_dies" -eq 0 ]; then
    ok "fix45-A2 (b) bind-mount invariant: the guard (five adjudication/provenance paths under \$HOME as spelled) exists, sits BEFORE the emitter call at line $gl < $el (position conjunct — a guard after its first dependent is a story, not a control), and the REAL extracted loop is silent when every path is under \$HOME (MUST_PASS row, exercised, 0 die calls on the in-tree row)"
  else
    no "fix45-A2 (b) bind-mount invariant: guard missing/mispositioned or it false-fires on an all-in-tree path set (row1 die count=$f45a2_row1_dies) — a guard that taxes the <compute-node>-proven layout is doctrine-5 noise"
  fi
  if printf '%s' "$f45a2_row2_text" | grep -q 'DIE:bind-mount invariant broken' \
     && printf '%s' "$f45a2_row2_text" | grep -qF '/f45a2/elsewhere' \
     && printf '%s' "$f45a2_row3_text" | grep -q 'DIE:bind-mount invariant broken' \
     && printf '%s' "$f45a2_row3_text" | grep -qF '/opt/not-home'; then
    ok "MUST_FIRE bind-mount invariant: an \$OUT_DIR outside \$HOME AND an FS_ROOT outside \$HOME each fire the REAL extracted loop's refusal, each naming the offending path (2 of 2 doctored inputs rejected; the arm is observed firing — the <compute-node> run succeeded BECAUSE of this precondition, which was unwritten until now)"
  else
    no "MUST_FIRE UNREACHABLE (bind-mount invariant): outside-tree rows did not fire the named refusal (row2: $(printf '%s' "$f45a2_row2_text" | tr '\n' ' ') ; row3: $(printf '%s' "$f45a2_row3_text" | tr '\n' ' ')) — the guard cannot see the override class it exists to refuse"
  fi
else
  no "MUST_FIRE UNREACHABLE: the bind-mount guard loop is not extractable from the full-FT launcher — legs 16-17 are unproven"
fi
f45a2_bt=$(mktemp "${TMPDIR:-/tmp}/fs-f45-binddel.XXXXXX") \
  && sed '/^for _bm in /,/^done$/d' "$FULL" > "$f45a2_bt"
f45a2_bs=$?
f45a2_bfired=1
if [ "$f45a2_bs" -eq 0 ] \
   && [ -z "$(sed -n '/^for _bm in /,/^done$/p' "$f45a2_bt")" ] \
   && ! f45a2_bind_static_ok "$f45a2_bt" && f45a2_bind_static_ok "$FULL"; then
  f45a2_bfired=0
fi
[ -n "${f45a2_bt:-}" ] && rm -f "$f45a2_bt" || true
if [ "$f45a2_bfired" -eq 0 ]; then
  ok "MUST_FIRE bind-mount deletion: removing the guard from a copy (construction proven: extraction empty on the copy, present live) turns the static predicate red — the launch-time assertion of the invariant this whole patch now depends on cannot be deleted quietly"
else
  no "MUST_FIRE UNREACHABLE (bind-mount deletion): the guard deletion did not construct or did not fire (sed rc=$f45a2_bs) — the bind-mount pin is an unproven detector"
fi

# --- legs 19-20: #82 — the generalized env-export census. EVERY variable a
# container-side python body reads from os.environ must be exported by this
# launcher. Population enumeration: a file-wide os.environ["VAR"] sweep,
# pinned at the measured count of 2 with both members named (a third read
# appearing moves the count conjunct red: extend the census AND ship its
# export — a one-variable fix without the general leg is how #82 waited in
# series behind #77). FOXBRAIN_SFT_JSONLS (exported since its own line in
# the Paths block) is the standing MUST_PASS member; HF_MODEL is the fix.
f45a2_env_census_ok() { # $1=launcher file (real or doctored copy)
  local f=$1 lc n vars
  lc=$(strip_shell_comments < "$f")
  n=$(printf '%s\n' "$lc" | grep -c 'os\.environ' || true)
  vars=$(printf '%s\n' "$lc" | grep -oE 'os\.environ\["[A-Z0-9_]+"\]' | sort -u | sed -E 's/os\.environ\["([A-Z0-9_]+)"\]/\1/' | tr '\n' ' ' | sed 's/ $//' || true)
  [ "$n" -eq 2 ] \
    && [ "$vars" = "FOXBRAIN_SFT_JSONLS HF_MODEL" ] \
    && printf '%s\n' "$lc" | grep -qE '^export FOXBRAIN_SFT_JSONLS=' \
    && printf '%s\n' "$lc" | grep -qE '^export HF_MODEL='
}
if f45a2_env_census_ok "$FULL"; then
  ok "fix45-A2 / #82 env census: 2 of 2 variables read from os.environ by container-side python are exported by the launcher — FOXBRAIN_SFT_JSONLS (standing MUST_PASS member, exported in the Paths block) and HF_MODEL (the measured #82 hard block: KeyError in the preflight probe on every launch until now). Population pinned at 2 by a file-wide sweep: a THIRD os.environ read appearing turns this leg red until its export AND this census are updated together"
else
  no "fix45-A2 / #82 env census drifted: a container-side python body reads a variable the launcher does not export (run_in_container forwards exported env only, s7); the <compute-node> KeyError: 'HF_MODEL' was the second unconditional hard block in series — the next unexported read is the third, and this census exists to see it before hardware does"
fi
f45a2_ht=$(mktemp "${TMPDIR:-/tmp}/fs-f45-hfexport.XXXXXX") \
  && sed 's/^export HF_MODEL=/HF_MODEL=/' "$FULL" > "$f45a2_ht"
f45a2_hs=$?
f45a2_hfired=1
if [ "$f45a2_hs" -eq 0 ] \
   && ! grep -qE '^export HF_MODEL=' "$f45a2_ht" \
   && grep -qE '^HF_MODEL=' "$f45a2_ht" \
   && ! f45a2_env_census_ok "$f45a2_ht" && f45a2_env_census_ok "$FULL"; then
  f45a2_hfired=0
fi
[ -n "${f45a2_ht:-}" ] && rm -f "$f45a2_ht" || true
if [ "$f45a2_hfired" -eq 0 ]; then
  ok "MUST_FIRE #82: un-exporting HF_MODEL on a copy — the BYTE shape measured on <compute-node> (grep -c 'export HF_MODEL' == 0; the assignment itself stays) — turns the census red (construction proven: no export line remains, the bare assignment does), so #82's exact shipped state cannot silently return"
else
  no "MUST_FIRE UNREACHABLE (#82): the un-export revert did not construct or did not fire (sed rc=$f45a2_hs) — the export census is an unproven detector"
fi

# --- legs 21-22: #84 — the malformed-counter class. grep -c already prints
# the count on rc 1; `|| echo 0` appended a second 0. Scoped LOW severity,
# honestly: the tripwire still fired on matches and the malformed test
# evaluated false correctly; what it cost is a permanent syntax error in
# every healthy run's log (measured <compute-node>, both sites) — the noise that
# trains an operator to ignore watcher errors. The leg sweeps FILE-WIDE for
# the antipattern, not just at the two known sites.
f45a2_zero_counter_ok() { # $1=launcher file (real or doctored copy)
  local lc
  lc=$(strip_shell_comments < "$1")
  ! printf '%s\n' "$lc" | grep -qF '|| echo 0' \
    && [ "$(printf '%s\n' "$lc" | grep -cF 'ZC=$(grep -c "ZERO supervised tokens" "$LOG_OUT" 2>/dev/null || true); ZC=${ZC:-0}' || true)" -eq 2 ]
}
if f45a2_zero_counter_ok "$FULL"; then
  ok "fix45-A2 / #84 counter hygiene: both zero-token counters (watcher + epilogue, 2 of 2 sites) take grep's own printed count with the default only on silence, and ZERO instances of the grep -c || echo 0 antipattern remain file-wide — the healthy-run syntax-error noise is gone (the tripwire was never dead; the noise was the defect)"
else
  no "fix45-A2 / #84 counter drifted: a grep -c count is again being appended-to by || echo 0 (the \"0\\n0\" malformed-arithmetic class, measured logging a syntax error every 45 s on <compute-node>), or a ZC site lost the silence-default"
fi
f45a2_zt=$(mktemp "${TMPDIR:-/tmp}/fs-f45-zerocount.XXXXXX") \
  && sed 's/|| true); ZC=\${ZC:-0}/|| echo 0)/' "$FULL" > "$f45a2_zt"
f45a2_zs=$?
f45a2_zn=-1
# Doctrine-3 rule, written where the next person will read it (same class
# as fix26b / finding #64 and the sibling repair at contracts:2497): a
# MUST_FIRE's construction-proof counter MUST MEASURE THE SAME VIEW AS
# THE PREDICATE IT GUARDS. f45a2_zero_counter_ok counts on the
# comment-stripped view, but a raw count here reads 4 on the reverted
# copy — 2 sed-injected code sites plus 2 mentions in the launcher's own
# #84 comments, which quote the antipattern as documentation — against
# this leg's predicted 2, so the detector could never prove it can fire
# (the UNREACHABLE this leg logged, count=4, sed rc=0). strip_shell_comments
# preserves line count, so piping through it makes this count the
# predicate's count: exactly the 2 injected code sites, comment prose
# excluded on purpose — quoting the antipattern in a comment is
# legitimate; appending it in code is the defect.
[ "$f45a2_zs" -eq 0 ] && f45a2_zn=$(strip_shell_comments < "$f45a2_zt" | grep -cF '|| echo 0' || true)
f45a2_zfired=1
if [ "$f45a2_zs" -eq 0 ] && [ "$f45a2_zn" -eq 2 ] && ! f45a2_zero_counter_ok "$f45a2_zt" \
   && f45a2_zero_counter_ok "$FULL"; then
  f45a2_zfired=0
fi
[ -n "${f45a2_zt:-}" ] && rm -f "$f45a2_zt" || true
if [ "$f45a2_zfired" -eq 0 ]; then
  ok "MUST_FIRE #84: restoring the malformed counter at both sites on a copy (construction proven: the antipattern present 2x on the copy, 0x live) turns the hygiene leg red — the log-noise class cannot silently return"
else
  no "MUST_FIRE UNREACHABLE (#84): the double-print revert did not construct or did not fire (sed rc=$f45a2_zs, comment-stripped copy antipattern count=$f45a2_zn, expected 2 of 2 code sites) — the hygiene leg is an unproven detector"
fi

# === watchdog sub-suite wire-in ==============================================
# launchers/test_fs_live_gate_watchdog_contracts.sh measured 8/8 in the
# defect-2 shard's hands, and grep then proved zero invokers: 8 good legs
# nobody runs are 8 non-controls (doctrine 3), so the regression they guard
# -- the watchdog's orphaned `sleep` grandchild wedging any $( ) reader for
# the full 600s budget and stamping 'live_gate wall-clock budget exhausted'
# onto a CLEARED checkpoint's permanent capture -- could return silenced.
# This block runs the sub-suite and folds its REAL counts into $pass/$fail,
# and it has no path from absence or breakage to green.
#
# ROUTE (a) -- parse 'WATCHDOG CONTRACTS: PASS n/N'; route (b)'s one
# composite leg is refused. Doctrine 2 wants the estate denominator honest
# (its other legs + this file's 8, and +9 the day it grows a ninth leg,
# which route (b) would undercount without a word). What breaks (a): a
# format change to that ONE line. That break lands RED below -- no
# parseable summary, or any exit code disagreeing with the parsed numbers,
# is a named FAIL with the raw output re-emitted. The needle is pinned to
# the sub-suite's prose, but fail-CLOSED: green requires rc=0 AND two
# parsed integers equal; every other shape funnels to `no` (even a `[`
# comparator error returns 2, which is elif-false, which is the mismatch
# red). Prose drift can never mint a green.
#
# ABSENCE IS RED, NOT ABSTAINED. This harness abstains batteries it was
# never furnished (fix28 'not in the packet'); this file SHIPS beside this
# one and has a measured 8/8, so its absence is a broken checkout or a bad
# delete. Doctrine 1 forbids vacuous green and doctrine 4 says fail closed:
# `no` names 0 of 8 measured and the frozen exit test below goes nonzero.
#
# CAPTURE IS SAFE -- LEG 3 PROVES IT. `wd_out=$( ... )` is the exact $( )
# shape the orphan defect wedged for 600s; that defect is repaired and the
# sub-suite's leg 3 ('$( )-caller returns promptly -- 0s elapsed vs 30s
# budget') is the positive control showing this capture returns at once.
# stderr is folded into the capture so a bash-level fault inside the
# sub-suite becomes evidence here (guttered below), not silence elsewhere.
# The run goes through `bash FILE`, never `./FILE`: the contract is 'runs
# under bash', and a checkout that drops +x must not dangle the battery.
#
# NO TIMEOUT WRAPPER, DELIBERATELY (considered, refused): there is no
# timeout(1) on macOS bash 3.2, and the estate's one hand-rolled watchdog
# -- `( sleep $BUDGET; kill ) &` against a subshell -- IS the orphan bug
# under guard. The sub-suite self-bounds every timed leg with $SECONDS
# thresholds (measured total well under 45s). Accepted residual, named per
# doctrine 5: if the sub-suite ever wedges anyway, THIS suite blocks and
# the red is 'the run never returns' -- not wrapped in a second watchdog.
#
# OUTPUT: captured, then re-emitted through a '  | ' gutter -- 9 lines do
# not drown the tally, the gutter keeps 'PASS k/8' rows from masquerading
# as main-suite legs, and each fold line below states the exact counter
# arithmetic at its site, so no claim outruns the evidence above it.
WD_CONTRACTS=$LDIR/test_fs_live_gate_watchdog_contracts.sh
if [ ! -f "$WD_CONTRACTS" ]; then
  no "watchdog sub-suite ABSENT: $WD_CONTRACTS not found -- 0 of 8 legs measured; a deleted or unshipped detector is red, never vacuous (doctrines 1/3/4)"
elif [ ! -r "$WD_CONTRACTS" ]; then
  no "watchdog sub-suite UNREADABLE: $WD_CONTRACTS -- 0 of 8 legs measured; failing closed"
else
  if wd_out=$(bash "$WD_CONTRACTS" 2>&1); then wd_rc=0; else wd_rc=$?; fi
  printf '%s\n' "$wd_out" | sed 's/^/  | /'
  wd_line=$(printf '%s\n' "$wd_out" | grep '^WATCHDOG CONTRACTS: PASS [0-9][0-9]*/[0-9][0-9]*$' | tail -n 1 || true)
  wd_p=$(printf '%s\n' "$wd_line" | sed -n 's/^WATCHDOG CONTRACTS: PASS \([0-9][0-9]*\)\/\([0-9][0-9]*\)$/\1/p')
  wd_n=$(printf '%s\n' "$wd_line" | sed -n 's/^WATCHDOG CONTRACTS: PASS \([0-9][0-9]*\)\/\([0-9][0-9]*\)$/\2/p')
  if [ -z "$wd_line" ]; then
    no "watchdog sub-suite gave NO parseable 'WATCHDOG CONTRACTS: PASS n/N' summary (exit $wd_rc) -- raw output above; unparsed is unmeasured, unmeasured is red; 0 legs credited"
  elif [ "$wd_rc" -eq 0 ] && [ "$wd_p" = "$wd_n" ]; then
    pass=$((pass + 10#$wd_p))
    printf '  PASS  watchdog contracts %s/%s green, exit 0 agrees -- +%s folded into pass at this site (route (a): the real denominator, not one composite)\n' "$wd_p" "$wd_n" "$wd_p"
  elif [ "$wd_rc" -ne 0 ] && [ -n "$wd_p" ] && [ "$wd_p" -lt "$wd_n" ]; then
    pass=$((pass + 10#$wd_p)); fail=$((fail + 10#$wd_n - 10#$wd_p))
    printf '  FAIL  watchdog contracts %s of %s green, %s red, exit %s agrees -- +%s pass and +%s fail folded at this site; per-leg evidence above\n' "$wd_p" "$wd_n" "$((10#$wd_n - 10#$wd_p))" "$wd_rc" "$wd_p" "$((10#$wd_n - 10#$wd_p))"
  else
    no "watchdog sub-suite CONTRACT MISMATCH: exit $wd_rc vs summary '$wd_line' -- one side of its contract is lying, 0 legs credited, fail closed (doctrine 4)"
  fi
fi

echo "abstentions: $abstain named (each named at its site above with its denominator; 0 added to pass or fail)"
echo "controls: $pass passed, $fail failed"
[ "$fail" -eq 0 ]
