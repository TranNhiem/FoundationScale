#!/usr/bin/env bash
# launch_fs_h100.sh -- model-agnostic single-node H100 launcher for FoundationScale.
# fs152: the submit partition is an INPUT (FS_PARTITION, required, no default),
# not a property of this file -- see the guard directly below the #SBATCH block.
# Estate facts measured 2026-08-30; do not generalize them into core defaults.
# CRITICAL WALLTIME (fs204/fs153): the maximum walltime is a property of the
# submit PARTITION, measured at submit time by the sinfo probe below -- never a
# property of this file. A ten-day standing rule is NOT portable: the same request
# is correctly ACCEPTED on a partition that allows it and correctly REFUSED where
# it exceeds the measured maximum. Do not re-bake any duration into this header;
# the knob is FS_WALLTIME (required, no default), proven against the oracle below.
# fs152: the `#SBATCH --partition=...` directive that stood here is DELETED, not
# parameterised in place -- an #SBATCH line is a comment to the shell, so an
# expanded-looking form would silently mean something different from what it
# says. The partition now travels as --partition="$FS_PARTITION" on the sbatch
# invocation below, where expansion actually happens.
#SBATCH --nodes=1
# fs204: the four estate-shaped directives -- gpus-per-node, cpus-per-task, mem and
# time -- are DELETED from this header, not parameterised in place. An #SBATCH line
# is a comment to the shell (fs152), so a variable written into one never expands
# and the directive would silently mean something other than what it says. These
# four are properties of the ESTATE'S HARDWARE -- its GPUs per node, its CPUs per
# task, its memory and its partition walltime window -- not of this launcher. They
# now travel as REAL flags on every sbatch invocation below, where expansion
# happens, fed by FS_GPUS_PER_NODE / FS_CPUS_PER_TASK / FS_MEM / FS_WALLTIME, all
# required with no defaults (a default would re-bake the shape this stage removes).
# RETAINED around them: nodes=1, ntasks-per-node=1, job-name, output and error are
# this launcher's own topology contract -- one task per node, torchrun fans out to
# the GPUs inside the allocation -- and name no estate fact.
#SBATCH --ntasks-per-node=1
#SBATCH --job-name=fs-h100
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err

# --- fs152: declared partition policy, replacing a hard-coded partition name --------
# REQUIRED, NO DEFAULT -- the same contract as FS_ALLOWED_NODE, FS_CONTAINER_RUNTIME
# and FS_ALLOWED_PATH_ROOTS. An unconfigured guard is a disabled standing rule; a
# default here would be the deleted literal compiled back in.
[[ -n "${FS_PARTITION:-}" ]] || { echo "REFUSE 96: FS_PARTITION is unset (required, no default by design). Set it to this estate's Slurm submit partition -- the framework refuses to guess a cluster layout." >&2; exit 96; }
# --- fs204: estate node-shape knobs, REQUIRED WITH NO DEFAULTS ------------------
# The same contract as FS_PARTITION directly above and the FS_ALLOWED_NODE /
# FS_CONTAINER_RUNTIME / FS_ALLOWED_PATH_ROOTS family (#123): an unconfigured guard
# is a disabled standing rule, and a default here would be the deleted estate shape
# compiled back in. This block sits ABOVE set -Eeuo pipefail and before fail() is
# defined, so it uses the raw { echo ... >&2; exit 96; } idiom, exactly like the
# FS_PARTITION guard it extends.
[[ -n "${FS_CPUS_PER_TASK:-}" ]] || { echo "REFUSE 96: FS_CPUS_PER_TASK is unset (required, no default by design). Set it to this estate's CPUs per training task -- the framework refuses to guess a cluster layout." >&2; exit 96; }
[[ "$FS_CPUS_PER_TASK" =~ ^[0-9]+$ && "$FS_CPUS_PER_TASK" -gt 0 ]] || { echo "REFUSE 96: FS_CPUS_PER_TASK must be a positive integer; got '$FS_CPUS_PER_TASK'." >&2; exit 96; }
[[ -n "${FS_MEM:-}" ]] || { echo "REFUSE 96: FS_MEM is unset (required, no default by design). Set it to this estate's per-node memory spec (digits, optional K/M/G/T suffix) -- the framework refuses to guess a cluster layout." >&2; exit 96; }
[[ "$FS_MEM" =~ ^[0-9]+[KMGT]?$ ]] || { echo "REFUSE 96: FS_MEM must match ^[0-9]+[KMGT]?$; got '$FS_MEM'." >&2; exit 96; }
[[ -n "${FS_WALLTIME:-}" ]] || { echo "REFUSE 96: FS_WALLTIME is unset (required, no default by design). Set it to the submit walltime for this estate's partition; the VALUE is proven against the measured partition maximum below -- the framework refuses to guess a cluster layout." >&2; exit 96; }

set -Eeuo pipefail
IFS=$'\n\t'
umask 027

# fs142 plane resolver: BEGIN (self-contained; backend helpers do not exist yet)
# Finding #142 was measured on an 8xH100 sbatch job: sbatch stages and renames
# the submitted file, so ${BASH_SOURCE[0]}/. is the spool directory in exactly
# the mode this launcher exists for. SLURM_SUBMIT_DIR was only the submit-time
# CWD and pointed at the plane's parent, so it is deliberately not used here.
# This prologue runs before the backend is parsed. fs_die and every other
# backend helper do not exist yet; Bash builtins, coreutils, and scontrol are
# the complete dependency budget.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd -P)"
BACKEND_NAME="fs_container_backend.bound.sh"
FS142_PLANE_OVERRIDE="${FS_PLANE_DIR:-}"
FS142_PLANE_DIR=""
FS142_PLANE_STEP=0
FS142_CANDIDATE=""
FS142_LAST_BACKEND_DIR="$SCRIPT_DIR"
FS142_STEP1_RESULT="not reached"
FS142_STEP2_RESULT="not reached"
FS142_STEP3_RESULT="not reached"
FS142_STEP4_RESULT="not reached"
FS142_WLM_COMMAND=""
FS142_WLM_KIND="${FS_ALLOCATION:-${SLURM_JOB_ID:+slurm}}"

# Keep the WLM lookup behind a named dispatch. The second workload manager is
# easier to add here than in a launcher whose estate logic has grown into a
# chain of hard-coded Slurm conditionals.
fs142_wlm_command_path() {
  local wlm_kind="$1"
  local batch_info=""
  local -a command_match=()
  case "$wlm_kind" in
  slurm)
    [[ -n "${SLURM_JOB_ID:-}" ]] || return 1
    command -v scontrol >/dev/null 2>&1 || return 1
    batch_info="$(scontrol show job "$SLURM_JOB_ID" 2>/dev/null)" || return 1
    if [[ "$batch_info" =~ (^|[[:space:]])Command=([^[:space:]]+) ]]; then
      command_match=("${BASH_REMATCH[@]}")
      [[ "${command_match[2]}" == /* ]] || return 1
      printf '%s\n' "${command_match[2]}"
      return 0
    fi
    return 1
    ;;
  *)
    return 127
    ;;
  esac
}

# FS_PLANE_DIR is an optional override, not a required-no-default setting:
# step 2 must keep a direct `bash launch...` invocation working without a new
# operator variable.
FS142_CANDIDATE="$FS142_PLANE_OVERRIDE"
if [[ -n "$FS142_CANDIDATE" ]]; then
  FS142_LAST_BACKEND_DIR="$FS142_CANDIDATE"
  if [[ -r "$FS142_CANDIDATE/$BACKEND_NAME" ]]; then
    if FS142_CANDIDATE="$(cd -- "$FS142_CANDIDATE" >/dev/null 2>&1 && pwd -P)"; then
      FS142_PLANE_DIR="$FS142_CANDIDATE"
      FS142_PLANE_STEP=1
      FS142_STEP1_RESULT="FS_PLANE_DIR=$FS142_PLANE_DIR; backend verified"
    else
      FS142_STEP1_RESULT="FS_PLANE_DIR=$FS142_CANDIDATE; physical directory could not be resolved"
    fi
  else
    FS142_STEP1_RESULT="FS_PLANE_DIR=$FS142_CANDIDATE; $BACKEND_NAME was not readable there"
  fi
else
  FS142_STEP1_RESULT="FS_PLANE_DIR=<unset>; no candidate directory was tested"
fi

# In-place execution remains first-class. `bash launch_fs_h100.fixed.sh`, an
# interactive srun, and unmeasured work managers that do not stage scripts all
# reach this answer without relying on Slurm-specific state.
if [[ -z "$FS142_PLANE_DIR" ]]; then
  FS142_CANDIDATE="$SCRIPT_DIR"
  FS142_LAST_BACKEND_DIR="$FS142_CANDIDATE"
  if [[ -r "$FS142_CANDIDATE/$BACKEND_NAME" ]]; then
    FS142_PLANE_DIR="$FS142_CANDIDATE"
    FS142_PLANE_STEP=2
    FS142_STEP2_RESULT="SCRIPT_DIR=$FS142_PLANE_DIR; backend verified"
  else
    FS142_STEP2_RESULT="SCRIPT_DIR=$FS142_CANDIDATE; $BACKEND_NAME was not readable there"
  fi
else
  FS142_STEP2_RESULT="not tested; step 1 resolved the plane"
fi

# A WLM answer is still only a claim. The measured #142 job supplied the
# original `/home/.../probe_sub/planeprobe.sh` through Command=; accepting that
# answer without checking the sibling beside it would turn a WLM defect into a
# falsely "resolved" plane.
if [[ -z "$FS142_PLANE_DIR" ]]; then
  if [[ -n "$FS142_WLM_KIND" ]]; then
    if FS142_WLM_COMMAND="$(fs142_wlm_command_path "$FS142_WLM_KIND")"; then
      FS142_CANDIDATE="$(dirname -- "$FS142_WLM_COMMAND")"
      FS142_LAST_BACKEND_DIR="$FS142_CANDIDATE"
      if [[ -r "$FS142_CANDIDATE/$BACKEND_NAME" ]]; then
        if FS142_CANDIDATE="$(cd -- "$FS142_CANDIDATE" >/dev/null 2>&1 && pwd -P)"; then
          FS142_PLANE_DIR="$FS142_CANDIDATE"
          FS142_PLANE_STEP=3
          FS142_STEP3_RESULT="$FS142_WLM_KIND returned $FS142_WLM_COMMAND; backend verified in $FS142_PLANE_DIR"
        else
          FS142_STEP3_RESULT="$FS142_WLM_KIND returned $FS142_WLM_COMMAND; physical directory could not be resolved"
        fi
      else
        FS142_STEP3_RESULT="$FS142_WLM_KIND returned $FS142_WLM_COMMAND; $BACKEND_NAME was not readable in $FS142_CANDIDATE"
      fi
    else
      FS142_STEP3_RESULT="$FS142_WLM_KIND returned no original script path"
    fi
  else
    FS142_STEP3_RESULT="no known workload manager was detected, so no script path was returned"
  fi
else
  FS142_STEP3_RESULT="not tested; an earlier step resolved the plane"
fi

# Finding #139 already paid for this lesson: a technically true refusal that
# sends the reader to the wrong property is itself the defect. Here the file
# existed while the directory was wrong, so the refused message names the
# attempted answers before naming the remedy.
if [[ -z "$FS142_PLANE_DIR" ]]; then
  FS142_STEP4_RESULT="refused: no verified backend after steps 1 through 3"
  printf 'FATAL[142]: resolution step 1 (FS_PLANE_DIR), returned: %s\n' "$FS142_STEP1_RESULT" >&2
  printf 'FATAL[142]: resolution step 2 (SCRIPT_DIR), returned: %s\n' "$FS142_STEP2_RESULT" >&2
  printf 'FATAL[142]: resolution step 3 (workload-manager dispatch), returned: %s\n' "$FS142_STEP3_RESULT" >&2
  printf 'FATAL[142]: resolution step 4 (refusal), returned: %s\n' "$FS142_STEP4_RESULT" >&2
  printf 'FATAL[142]: directory actually searched last: %s\n' "$FS142_LAST_BACKEND_DIR" >&2
  printf 'FATAL[142]: hypothesis: this looks like a workload-manager-staged copy of the script, not the original\n' >&2
  printf 'FATAL[142]: remedy: set and export FS_PLANE_DIR=<directory containing %s> before submitting\n' "$BACKEND_NAME" >&2
  exit 96
fi

FS142_STEP4_RESULT="accepted verified backend at $FS142_PLANE_DIR/$BACKEND_NAME"
case "$FS142_PLANE_STEP" in
  1) FS142_RESOLUTION_SOURCE="FS_PLANE_DIR" ;;
  2) FS142_RESOLUTION_SOURCE="SCRIPT_DIR" ;;
  3) FS142_RESOLUTION_SOURCE="$FS142_WLM_KIND" ;;
  *) FS142_RESOLUTION_SOURCE="unknown" ;;
esac
printf 'plane directory resolved: %s (resolution step %s: %s)\n' \
  "$FS142_PLANE_DIR" "$FS142_PLANE_STEP" "$FS142_RESOLUTION_SOURCE"

# Later jobs and every child process inherit the measured answer instead of
# independently repeating the sbatch-staging mistake.
export FS_PLANE_DIR="$FS142_PLANE_DIR"
BACKEND="$FS142_PLANE_DIR/$BACKEND_NAME"
# shellcheck source=fs_container_backend.bound.sh
source "$BACKEND"
# fs142 plane resolver: END

# fs175: finding #169 -- the plane publishes exactly four states: 0 PASS, 5 RED,
# 95 UNMEASURED, 96 REFUSE. The measured census of this choke point's call sites
# was 49x rc 96, 1x rc 95, and 1x rc 124 (the compose refusal below): 124 is in
# no namespace this plane publishes, and nothing forbade the next violation. So
# the contract is now enforced HERE, at the single exit choke point: an
# out-of-contract first argument is itself a REFUSE. This must NOT recurse
# through fail (a fail that calls fail on bad input never exits), so the bad-rc
# arm prints directly and exits 96. The FATAL[%s]: %s format is unchanged for
# legal codes, so no log consumer breaks.
fail() {
  local rc="${1:-}"
  if [[ $# -gt 0 ]]; then shift; fi
  case "$rc" in
    0|5|95|96) ;;
    *) printf 'FATAL[96]: fail() called with out-of-contract rc=%s (want one of: 0 5 95 96): %s\n' "$rc" "$*" >&2; exit 96 ;;
  esac
  printf 'FATAL[%s]: %s\n' "$rc" "$*" >&2
  exit "$rc"
}

require_cmd() { command -v "$1" >/dev/null 2>&1 || fail 96 "missing required command: $1"; }
req_env() { local n="$1"; [[ -n "${!n:-}" ]] || fail 96 "required environment variable unset or empty: $n"; }

# Orthogonal axes (A2): this launcher is slurm-allocation + singularity-runtime only.
# Runtime is explicit, never auto-detected from PATH or SLURM_JOB_ID.
# fs126: no default. The backend's own guard calls FS_ALLOCATION "required, no default
# by design ... never inferred", and this line used to infer it -- making that guard
# unreachable through this launcher. Defaulting to EMPTY and refusing is the idiom used
# for FS_CONTAINER_RUNTIME two lines below; an unconfigured guard is a disabled rule.
export FS_ALLOCATION="${FS_ALLOCATION:-}"
[[ "$FS_ALLOCATION" == slurm ]] || fail 96 "FS_ALLOCATION must be exactly 'slurm' for this launcher (no default by design: who allocated these nodes is a separate axis from which runtime launches on them, and it is never inferred from SLURM_JOB_ID). got '${FS_ALLOCATION:-<unset>}'"
export FS_CONTAINER_RUNTIME="${FS_CONTAINER_RUNTIME:-}"
[[ "$FS_CONTAINER_RUNTIME" == singularity ]] || fail 96 "FS_CONTAINER_RUNTIME must be exactly 'singularity' (no default); got '${FS_CONTAINER_RUNTIME:-<unset>}'"
# Legacy selector kept constrained so an old backend cannot drift into auto/pyxis.
export FS_BACKEND="${FS_BACKEND:-slurm-singularity}"
case "$FS_BACKEND" in slurm-singularity|singularity) ;; *) fail 96 "FS_BACKEND must be slurm-singularity here; got '$FS_BACKEND'";; esac

require_cmd sbatch; require_cmd srun; require_cmd singularity; require_cmd sinfo; require_cmd scontrol

# Nothing model-family-specific is accepted or emitted. Inputs are artifacts/config only.
req_env MODEL_DIR; req_env DATASET_DIR; req_env CONFIG_FILE; req_env OUT_DIR_STABLE
req_env IMAGE; req_env FS_GPUS_PER_NODE; req_env PROBE
[[ -d "$MODEL_DIR" ]] || fail 96 "MODEL_DIR unreadable/missing: $MODEL_DIR"
[[ -d "$DATASET_DIR" ]] || fail 96 "DATASET_DIR unreadable/missing: $DATASET_DIR"
[[ -r "$CONFIG_FILE" ]] || fail 96 "CONFIG_FILE unreadable/missing: $CONFIG_FILE"
[[ -r "$IMAGE" && "$IMAGE" == *.sif ]] || fail 96 "IMAGE must be a readable .sif: $IMAGE"

# --- fs123: declared path-root policy, replacing a hard-coded estate root ----
# WHY THE GUARD EXISTS AT ALL: a path the container cannot reach produces a
# failure that blames the model, not the mount plane. Refusing early is right.
# WHY IT IS NO LONGER A LITERAL: the set of reachable roots is a property of the
# ESTATE, not of FoundationScale. Hard-coding one made the framework correct on
# a single filesystem and silently wrong everywhere else.
#
# REQUIRED, NO DEFAULT -- the same contract as FS_ALLOWED_NODE and
# FS_CONTAINER_RUNTIME, for the same stated reason: an unconfigured guard is a
# disabled standing rule. A "sensible default" here would just be a literal
# estate root again, one whose wrongness surfaces later and blames the operator.
#
# A LIST, not a scalar: estates routinely keep assets and scratch on different
# filesystems, and a single-root assumption is how a one-estate literal returns
# wearing a variable's name. Space-separated, deliberately word-split.
[[ -n "${FS_ALLOWED_PATH_ROOTS:-}" ]] || fail 96 \
  "FS_ALLOWED_PATH_ROOTS is unset (required, no default by design). Set it to the space-separated absolute root(s) of this estate that are reachable from inside the container -- the framework refuses to guess a filesystem layout."

# Prefix matching on a PATH BOUNDARY, not on a string. The literal this replaces
# used a trailing-slash glob and got that right; "$p" == "$root"* would not, and
# would admit /workfoo for root /work. Kept as a function so all three call
# sites share one definition and cannot drift.
fs_path_under_allowed_root() {
  local p=$1 root
  [[ -n "$p" ]] || return 1
  # fs139: widen the split to include space. The refusal above tells the
  # operator to set "space-separated" roots, and prose and parser must
  # agree -- fixing the parser but leaving the prose would be the same
  # defect facing the other way. The split is widened with a
  # function-local IFS, NOT by editing the global safety IFS on line 18:
  # touching that line would re-enable glob-and-space splitting for every
  # other expansion in the script, and is the tempting wrong fix.
  local IFS=$' \t\n'
  # shellcheck disable=SC2086
  for root in ${FS_ALLOWED_PATH_ROOTS}; do
    root=${root%/}
    [[ -n "$root" ]] || continue
    [[ "$p" == "$root" || "$p" == "$root"/* ]] && return 0
  done
  return 1
}
fs_path_under_allowed_root "$IMAGE" || fail 96 \
  "IMAGE is outside every declared FS_ALLOWED_PATH_ROOTS entry ($FS_ALLOWED_PATH_ROOTS): $IMAGE"
fs_path_under_allowed_root "$MODEL_DIR" || fail 96 \
  "MODEL_DIR is outside every declared FS_ALLOWED_PATH_ROOTS entry ($FS_ALLOWED_PATH_ROOTS): $MODEL_DIR"
fs_path_under_allowed_root "$DATASET_DIR" || fail 96 \
  "DATASET_DIR is outside every declared FS_ALLOWED_PATH_ROOTS entry ($FS_ALLOWED_PATH_ROOTS): $DATASET_DIR"

# fs146: the ADJUDICATORS parse MOVED here from its original home BELOW the
# bind derivation. Both consumers of the parsed list -- the (a) containment
# check and the (b) bind-plane derivation -- sat above that point, and a
# consumer that cannot be checked until after its input exists is the
# structural cause of #146: the most load-bearing knob in the plane was
# the only unchecked one BECAUSE nothing up here could see it. Moving the
# consumers down was the alternative and was rejected: startup checks buy
# nothing late, and (a) is a defect precisely because a bad value surfaced
# after hours of paid GPU time, not before.

# Checkpoint adjudicators: finding #68 was zero call sites. They must be configured explicitly;
# absence is not success. Each entry is invoked as: <cmd> <checkpoint_dir> <phase> <out_dir>.
ADJUDICATORS_RAW="${FS_CHECKPOINT_ADJUDICATORS:-}"
[[ -n "$ADJUDICATORS_RAW" ]] || fail 96 "FS_CHECKPOINT_ADJUDICATORS empty; cannot call save adjudicators after saves (all([])!=PASS) -- entries are space/tab/newline-separated, one adjudicator per word"
# fs139: this list used to be split under the global safety IFS while its
# comment said "Each entry is invoked as ..." -- entries, plural, separator
# never stated -- and its refusal then named an adjudicator path, blaming
# the value when the fault was the invisible separator from line 18.
# Prose and parser now agree, in both directions: one adjudicator per word,
# separated by space, tab or newline (space because that is what an
# operator types; tab/newline because that is what the old behaviour forced
# and must keep working). A claim broader than its evidence is a defect
# even when the code is correct -- the stale refusal message was that.
# The assignment-prefix IFS scopes to this read alone, so the global
# safety setting is not weakened for the rest of the script.
IFS=$' \t\n' read -r -a ADJUDICATORS <<< "$ADJUDICATORS_RAW"
[[ "${#ADJUDICATORS[@]}" -gt 0 ]] || fail 96 "zero checkpoint adjudicators configured"
for a in "${ADJUDICATORS[@]}"; do [[ -n "$a" ]] || fail 96 'empty adjudicator token'; done

# fs146 (a): EVERY spec gets the containment check the other four executed
# paths already had, and the report carries its denominator ("k of n"), so
# an empty or short-circuited sweep can never read as measured -- all([])
# is UNMEASURED, never PASS. The refusal names the offending spec AND the
# declared roots: blame aimed only at the value sends the operator to edit
# the wrong half of the contract.
_adj_seen=0; _adj_ok=0; _adj_bad=""
for _adj in "${ADJUDICATORS[@]}"; do
  _adj_seen=$((_adj_seen+1))
  if fs_path_under_allowed_root "$_adj"; then
    _adj_ok=$((_adj_ok+1))
  else
    _adj_bad="$_adj_bad $_adj"
  fi
done
printf 'fs146: containment %d of %d adjudicator spec(s) under a declared root\n' "$_adj_ok" "$_adj_seen"
[[ "$_adj_ok" -eq "$_adj_seen" ]] || fail 96 \
  "fs146: adjudicator spec(s) outside every declared FS_ALLOWED_PATH_ROOTS entry ($FS_ALLOWED_PATH_ROOTS):${_adj_bad}"
unset _adj_seen _adj_ok _adj_bad
# fs146 (b): each spec's dirname joins the bind inference below. The
# tempting move is a runbook -- "operators must add these to
# FS_EXTRA_BIND_PATHS" -- and it is rejected here on purpose: this knob is
# REQUIRED and framework-known, and routing a required input through the
# escape hatch makes "required" mean "remembered", which is not a control.
# Membership of an allowed root and membership of the bind plane are
# DIFFERENT properties; inference owes declared inputs both. The existing
# de-duplication below is untouched, so a spec under MODEL_DIR adds no
# redundant mount.
declare -a _adj_dirs=()
for _adj in "${ADJUDICATORS[@]}"; do
  _adj_dirs+=("$(dirname -- "$_adj")")
done
unset _adj
# fs123: three separate tests, not one conjunction. The original ANDed MODEL_DIR and
# DATASET_DIR into a single message, so an operator with one bad path was told both
# were wrong and had to bisect by hand.
[[ "$FS_GPUS_PER_NODE" =~ ^[0-9]+$ ]] || fail 96 "FS_GPUS_PER_NODE must be an integer"
[[ "$FS_GPUS_PER_NODE" -gt 0 ]] || fail 96 "FS_GPUS_PER_NODE must be > 0"

PROBE="$PROBE"
[[ "$PROBE" == 0 || "$PROBE" == 1 ]] || fail 96 "PROBE must be exactly 0 or 1; got '$PROBE'"

# fs153: the hard-coded FS_WALLTIME comparison that stood here is DELETED. It was a
# second, stale oracle for the partition's maximum, and it ran BEFORE the live
# sinfo probe -- so it refused requests the probe would have proven legal, and a
# correct multi-day walltime died against a baked-in constant. One oracle remains:
# the measured maximum below.
# One parser, two call sites. Slurm durations arrive as [D-]HH:MM:SS, D-HH:MM, D-HH, MM:SS, or MM.
# Prints seconds on stdout; rc=1 on anything unparseable. UNLIMITED/INFINITE are returned rc=1 and
# must be handled EXPLICITLY by callers, never silently admitted or rejected.
fs_tl_seconds() {
  local t="$1" d=0 h=0 m=0 s=0 hasdash=0
  [[ -n "$t" && "$t" != UNLIMITED && "$t" != INFINITE ]] || return 1
  if [[ "$t" == *-* ]]; then hasdash=1; d="${t%%-*}"; t="${t#*-}"; fi
  [[ "$d" =~ ^[0-9]+$ ]] || return 1
  local IFS=:
  local -a f; read -r -a f <<< "$t"
  case "${#f[@]}" in
    3) h="${f[0]}"; m="${f[1]}"; s="${f[2]}" ;;
    2) if [[ "$hasdash" == 1 ]]; then h="${f[0]}"; m="${f[1]}"; else m="${f[0]}"; s="${f[1]}"; fi ;;
    1) if [[ "$hasdash" == 1 ]]; then h="${f[0]}"; else m="${f[0]}"; fi ;;
    *) return 1 ;;
  esac
  [[ "$h" =~ ^[0-9]+$ && "$m" =~ ^[0-9]+$ && "$s" =~ ^[0-9]+$ ]] || return 1
  [[ "$m" -lt 60 && "$s" -lt 60 ]] || return 1
  printf '%s\n' "$(( d*86400 + h*3600 + m*60 + s ))"
}

# fs153: the partition is THE ONLY oracle for the maximum. The answer is hard-coded
# nowhere -- the stale literal comparison that once ran above this probe is deleted,
# and FS_WALLTIME itself is proven against this measured value directly below it.
part_max="$(sinfo -h -p "$FS_PARTITION" -o '%l' 2>/dev/null | head -n1 || true)"
[[ -n "$part_max" ]] || fail 96 "$FS_PARTITION partition max probe returned nothing; UNMEASURED is not PASS"
max_unlimited=0
if [[ "$part_max" == UNLIMITED ]]; then
  max_unlimited=1
else
  max_sec="$(fs_tl_seconds "$part_max")" || fail 96 "unparseable ${FS_PARTITION} partition max '$part_max'; UNMEASURED is not PASS"
fi

# fs153: prove the REQUESTED walltime against the MEASURED maximum -- the check the
# deleted stale-guard could never perform, because it compared against a constant.
# fs_tl_seconds returns rc=1 for UNLIMITED and INFINITE, so an unbounded request is
# refused here as not-a-finite-duration; a finite request under an UNLIMITED
# partition maximum is admitted explicitly, never vacuously.
wt_sec="$(fs_tl_seconds "$FS_WALLTIME")" || fail 96 "FS_WALLTIME='$FS_WALLTIME' is not a finite Slurm duration; UNMEASURED is not PASS"
if [[ "$max_unlimited" == 0 ]]; then
  (( wt_sec <= max_sec )) || fail 96 "FS_WALLTIME='$FS_WALLTIME' (${wt_sec}s) exceeds the measured ${FS_PARTITION} partition max '$part_max' (${max_sec}s); refusing instead of clamping"
else
  echo "NOTICE: $FS_PARTITION partition maximum is UNLIMITED; finite FS_WALLTIME='$FS_WALLTIME' (${wt_sec}s) admitted"
fi

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  submit_line="$(scontrol show job "$SLURM_JOB_ID" 2>/dev/null | tr ' ' '\n' | grep '^TimeLimit=' || true)"
  [[ "$submit_line" == TimeLimit=?* ]] || fail 96 "scontrol reported no TimeLimit for job $SLURM_JOB_ID; UNMEASURED is not PASS"
  submitted="${submit_line#TimeLimit=}"
  if [[ "$submitted" == UNLIMITED ]]; then
    [[ "$max_unlimited" == 1 ]] || fail 96 "submitted TimeLimit=UNLIMITED exceeds finite ${FS_PARTITION} max '$part_max'"
  else
    sub_sec="$(fs_tl_seconds "$submitted")" || fail 96 "cannot prove submitted TimeLimit <= ${FS_PARTITION} max; unparseable TimeLimit '$submitted'"
    if [[ "$max_unlimited" == 0 ]]; then
      (( sub_sec <= max_sec )) || fail 96 "cannot prove submitted TimeLimit <= ${FS_PARTITION} max; got '$submitted' (${sub_sec}s) > '$part_max' (${max_sec}s)"
    fi
  fi
else
  # Login node: the probe above already forced the partition max into a proven numeric value,
  # or flagged UNLIMITED explicitly. No walltime can be submitted from here, so report only.
  if [[ "$max_unlimited" == 1 ]]; then
    printf 'NOTICE: %s partition reports UNLIMITED max walltime\n' "$FS_PARTITION"
  else
    printf '%s partition max walltime: %s (%ss)\n' "$FS_PARTITION" "$part_max" "$max_sec"
  fi
fi

# Named-interfaces and NCCL pins are estate facts. Do not inherit GB200 bond0/mlx5 claims.
# If an estate profile supplies interface names, require them to exist; absent config means no pins.
if [[ -n "${FS_NCCL_SOCKET_IFNAME:-}" ]]; then
  ip link show "$FS_NCCL_SOCKET_IFNAME" >/dev/null 2>&1 || fail 96 "FS_NCCL_SOCKET_IFNAME named but absent: $FS_NCCL_SOCKET_IFNAME"
fi
if [[ -n "${FS_NCCL_IB_HCA:-}" ]]; then
  fail 96 "FS_NCCL_IB_HCA pinning is unmeasured on ${FS_PARTITION}; leave unset unless measured and validated"
fi

# fs129: conditional NCCL_NET_PLUGIN seam -- the only sanctioned producer of this name.
# Measured on 8x H100 nemo_25.11.sif (F1): NCCL_NET_PLUGIN=none makes the first all_reduce
# correct (world=8 got=28.0 expected=28.0 spread=0.0) with NVLink UNAFFECTED -- under
# NCCL_DEBUG=INFO the 8-rank log still shows "24 coll channels", 17 lines matching NVLS and 24
# matching "via P2P/CUMEM", so NVLS is still selected and P2P/CUMEM is still the intra-node
# path. Those are log-LINE counts with the patterns stated, not per-rank connection totals: an
# earlier revision of this comment claimed "192 P2P/CUMEM", which was 8 x 24 inferred from the
# rank count rather than a count of anything observed, and a reader of a generated file has no
# way to tell a measurement from an arithmetic guess.
# The framework deliberately does NOT default this to none: on an
# estate where the bundled plugin works it is a performance feature (SHARP offload), and
# converting operator silence into a decision is the #126 defect class (defaulting
# FS_ALLOCATION). F4: an exported-but-EMPTY NCCL_NET_PLUGIN means DISABLED, not unset. F5: the
# backend builds its forwarded env from compgen -e (backend :1138), which LISTS
# exported-but-empty names -- an unconditional export would forward NCCL_NET_PLUGIN= into the
# container and re-create F4 one layer down. So this export exists only when
# FS_NCCL_NET_PLUGIN is non-empty, and the value is validated because a mistyped VALUE is
# worse than an unset one: a typo like NCCL_NET_PLUGN=none changes nothing, but a wrong value
# launches wrong.
if [[ -n "${FS_NCCL_NET_PLUGIN-}" ]]; then
  case "$FS_NCCL_NET_PLUGIN" in
    none)
      export NCCL_NET_PLUGIN="none"
      ;;
    /*.so)
      [[ -f "$FS_NCCL_NET_PLUGIN" ]] || fail 96 "fs129: got FS_NCCL_NET_PLUGIN=$FS_NCCL_NET_PLUGIN -- want none or an absolute path to an existing .so"
      export NCCL_NET_PLUGIN="$FS_NCCL_NET_PLUGIN"
      ;;
    *)
      fail 96 "fs129: got FS_NCCL_NET_PLUGIN=$FS_NCCL_NET_PLUGIN -- refused: accept only none or an absolute path to an existing .so"
      ;;
  esac
fi


# Env crossing the container boundary is allowlist policy owned by backend; launcher exports only explicit facts.
export PYTHONNOUSERSITE=1
# fs116: the SINGULARITYENV_PYTHONNOUSERSITE export that stood here is DELETED.
# Measured to be a no-op: PYTHONNOUSERSITE is on FS_ENV_ALLOWLIST so it already
# crosses on both arms, and the backend's singularity arm exports the
# SINGULARITYENV_ form itself inside run_in_container. What it actually did was
# demonstrate that naming a runtime in the launcher is acceptable -- the habit
# behind three separate runtime-divergence defects.
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export TOKENIZERS_PARALLELISM=false
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=WARN

# Stable production OUT_DIR must not encode a job id and must not equal probe dir.
case "$OUT_DIR_STABLE" in *SLURM_JOB_ID*|*%j*|*JOBID*|*jobid*) fail 96 "OUT_DIR_STABLE must be stable and contain no job id: $OUT_DIR_STABLE";; esac
# fs123: $HOME stays an accepted OUT_DIR root independently of the declared estate
# roots -- it is where an operator without asset-tree write access must be able to
# put outputs, and it is a property of the SESSION rather than the estate.
fs_path_under_allowed_root "$OUT_DIR_STABLE" || [[ -n "${HOME:-}" && "$OUT_DIR_STABLE" == "${HOME%/}"/* ]] || fail 96 \
  "OUT_DIR_STABLE is outside every declared FS_ALLOWED_PATH_ROOTS entry ($FS_ALLOWED_PATH_ROOTS) and outside \$HOME: $OUT_DIR_STABLE"
if [[ "$PROBE" == 1 ]]; then
  OUT_DIR="${OUT_DIR_STABLE%/}_probe"
  [[ "$OUT_DIR" != "$OUT_DIR_STABLE" ]] || fail 96 "probe output collision with production OUT_DIR"
  mkdir -p -- "$OUT_DIR"
  # L4: knobs with no reader are a false claim of configurability. Track provenance, validate the
  # denominators, and state them in the log. Both vars are on FS_ENV_ALLOWLIST, so they now cross
  # the container boundary and are read by the engine entrypoint named in FS_ENGINE_LAUNCH_CMD.
  # No filename appears here because the entrypoint is operator-supplied: naming one would be a
  # claim about the operator's engine, not about this launcher. It must fatal in probe phase with no
  # effective budget (env or config) -- an unbounded 'probe' is not a probe.
  if [[ -n "${FS_ITERATION_BUDGET:-}" ]]; then budget_src=env; else FS_ITERATION_BUDGET=20; budget_src=default; fi
  if [[ -n "${FS_EARLY_SAVE_STEPS:-}" ]]; then save_src=env; else FS_EARLY_SAVE_STEPS=5; save_src=default; fi
  [[ "$FS_ITERATION_BUDGET" =~ ^[0-9]+$ && "$FS_ITERATION_BUDGET" -gt 0 ]] || fail 96 "probe FS_ITERATION_BUDGET must be a positive integer (value='$FS_ITERATION_BUDGET', source=$budget_src); an unbounded probe is not a probe"
  [[ "$FS_EARLY_SAVE_STEPS" =~ ^[0-9]+$ && "$FS_EARLY_SAVE_STEPS" -gt 0 ]] || fail 96 "probe FS_EARLY_SAVE_STEPS must be a positive integer (value='$FS_EARLY_SAVE_STEPS', source=$save_src)"
  [[ "$FS_EARLY_SAVE_STEPS" -lt "$FS_ITERATION_BUDGET" ]] || fail 96 "probe early-save steps ($FS_EARLY_SAVE_STEPS) must be < iteration budget ($FS_ITERATION_BUDGET) (source=$save_src); an early save that cannot fire is not evidence"
  printf 'PROBE denominator: FS_ITERATION_BUDGET=%s (source=%s) FS_EARLY_SAVE_STEPS=%s (source=%s); consumed in-container via FS_ENV_ALLOWLIST by the engine entrypoint named in FS_ENGINE_LAUNCH_CMD\n' "$FS_ITERATION_BUDGET" "$budget_src" "$FS_EARLY_SAVE_STEPS" "$save_src"
else
  OUT_DIR="$OUT_DIR_STABLE"
  mkdir -p -- "$OUT_DIR"
  FS_ITERATION_BUDGET="${FS_ITERATION_BUDGET:-}"
  FS_EARLY_SAVE_STEPS="${FS_EARLY_SAVE_STEPS:-}"
fi
export OUT_DIR FS_ITERATION_BUDGET FS_EARLY_SAVE_STEPS

# --- fs117: POPULATE the declared bind plane -------------------------------
# The backend supplies the MECHANISM (FS_BIND_PATHS, materialised as --mount
# under enroot and --bind under singularity). This is where the run's actual
# requirement is DECLARED. Two things it deliberately does not do:
#
#   * It does not hard-code an estate root. A literal site path here would be
#     the one-time workaround rather than the abstraction: correct on exactly
#     one filesystem, silently wrong on the next one, and invisible until a
#     model failed to load somewhere else.
#   * It does not guess. Every entry derives from a path the launcher ALREADY
#     requires (req_env, above), so the bind set cannot drift away from the set
#     of paths the run actually references. Adding a new path-bearing input
#     without binding it becomes impossible by construction.
#
# Binds are identity (HOST:HOST) on purpose: the container then resolves the
# same string the host validated, so nothing needs rewriting at the boundary.
# Remapping would force every path variable to carry a host form AND a
# container form -- which is precisely how the two runtime arms drifted apart
# in the first place.
#
# An array, not an exported string: bash arrays do not survive `export`, and
# the backend is sourced into THIS shell, so the array is in scope. A scalar
# would collapse a multi-path declaration into one entry and fail closed later
# with a confusing message about an unreadable path.
declare -a FS_BIND_PATHS=()
# FS_EXTRA_BIND_PATHS is intentionally word-split: it is a space-separated
# operator escape hatch for paths the framework cannot infer (a code tree, a
# scratch root). Inference covers the declared inputs; it cannot cover these.
# shellcheck disable=SC2206,SC2086
for _p in "$MODEL_DIR" "$DATASET_DIR" "$(dirname -- "$CONFIG_FILE")" "$OUT_DIR" \
          ${_adj_dirs[@]+"${_adj_dirs[@]}"} \
          ${FS_EXTRA_BIND_PATHS:-}; do
  [[ -n "$_p" ]] || continue
  _dup=0
  for _q in ${FS_BIND_PATHS[@]+"${FS_BIND_PATHS[@]}"}; do
    [[ "$_q" == "$_p" ]] && { _dup=1; break; }
  done
  (( _dup )) || FS_BIND_PATHS+=("$_p")
done
unset _p _q _dup
# Report with a denominator, and refuse an empty derivation. Zero is a legal
# FS_BIND_PATHS in the backend ("0 of 0 declared"), but it is NOT legal here:
# this run demonstrably references a model, a dataset, a config and an output
# directory, so an empty set means the derivation broke, not that nothing was
# needed. Fail closed rather than launch a run that can see none of its inputs.
printf 'fs117: declared %d bind path(s): %s\n' \
  "${#FS_BIND_PATHS[@]}" "${FS_BIND_PATHS[*]}"
[[ ${#FS_BIND_PATHS[@]} -gt 0 ]] || fail 96 \
  "fs117: derived ZERO bind paths from the required inputs and the adjudicator dirnames; an empty bind plane here is a derivation bug, not a legal empty set"

LOG_DIR="${LOG_DIR:-$OUT_DIR/logs}"; mkdir -p -- "$LOG_DIR"
RUN_LOG="$LOG_DIR/launch.${SLURM_JOB_ID:-interactive}.log"

# fs146: the ADJUDICATORS parse and its refusals MOVED UP beside the other
# startup containment checks. Both consumers of the parsed list -- the (a)
# containment loop and the (b) bind-plane derivation -- sit ABOVE this
# point, so a parse that lived here made the knob uncheckable exactly where
# the other four executed paths were checked. Gate D proves the new order
# by line number; do not re-move the parse without re-homing its consumers.

checkpoint_observed=0
# fs176: the walker's INDEPENDENT denominator, published before any adjudicator
# runs. Measured on job 37308: two checkpoint saves on disk against a reported
# denominator of 1, the early save skipped, and the run PASSED. The positivity
# refusal below cannot catch that shape; only observed == found can, and the
# coverage check needs this number to exist whether or not a walk truncated.
checkpoint_found=0
run_adjudicators() {
  local ckpt="$1" phase="$2" ok=0 seen=0
  [[ -d "$ckpt" ]] || { printf 'ADJUDICATOR-SKIP rc=96 checkpoint_dir_missing=%s\n' "$ckpt"; return 96; }
  for spec in "${ADJUDICATORS[@]}"; do
    seen=$((seen+1))
    # fs146 (c): capture, then classify -- never propagate blind. The old
    # `|| return $?` let python3's exit 2 (spec invisible in-container, i.e.
    # missing from the bind plane) escape as `END rc=2 phase=adjudicate`, a
    # code this plane does not declare and therefore cannot attribute. 0
    # stays 0, 95 stays 95 (abstention is not failure), 96 stays 96;
    # ANYTHING else maps to 96 with the ORIGINAL code printed. Mapping
    # undeclared codes to 0 was considered and rejected as strictly worse
    # than the defect: a laundered silent pass is all([]) wearing green.
    local _rc=0
    if [[ -x "$spec" ]]; then
      "$spec" "$ckpt" "$phase" "$OUT_DIR" || _rc=$?
    elif [[ -f "$spec" && "$spec" == *.py ]]; then
      run_in_container --workdir "$OUT_DIR" -- python3 "$spec" "$ckpt" "$phase" "$OUT_DIR" || _rc=$?
    else
      printf 'ADJUDICATOR-REFUSE rc=96 not_executable=%s\n' "$spec" >&2
      return 96
    fi
    case $_rc in
      0) ;;
      95|96) return "$_rc" ;;
      *)
        printf 'ADJUDICATOR-REFUSE rc=96 original_rc=%s spec=%s -- undeclared exit code mapped to 96; leading hypothesis: spec is not bound into the container (absent from FS_BIND_PATHS), not an adjudicator failure\n' "$_rc" "$spec" >&2
        return 96
        ;;
    esac
    ok=$((ok+1)); continue
  done
  [[ "$seen" -gt 0 && "$ok" -eq "$seen" ]] || { printf 'ADJUDICATOR-REFUSE rc=96 seen=%s ok=%s\n' "$seen" "$ok" >&2; return 96; }
  checkpoint_observed=$((checkpoint_observed+1))
  printf 'ADJUDICATORS observed=%s seen=%s ok=%s ckpt=%s phase=%s\n' "$checkpoint_observed" "$seen" "$ok" "$ckpt" "$phase"
}

adjudicate_tree() {
  # fs176: collect first, iterate second. Measured on job 37308 -- a run that
  # PASSED. The output tree held TWO checkpoint saves (the early save the resume
  # proof depends on, and the final save) while this plane reported a denominator
  # of 1 and exited 0: the old body invoked the per-checkpoint runner INSIDE the
  # streaming read loop, the container runtime and interpreter inherited that
  # stdin and drained it, and iteration 2's read hit EOF, so only the FIRST
  # directory find emitted was ever adjudicated. The count was incremented inside
  # the same truncated loop, so numerator and denominator were cut together and
  # the report stayed self-consistent -- a denominator derived from the stream it
  # measures cannot detect its own truncation. And the bare call discarded the
  # runner's rc while the `|| ar=$?` call site suppresses errexit throughout
  # this body, so a per-checkpoint REFUSE was dropped twice over.
  #
  # Structure now: (1) the collection loop forks NOTHING in its body, so its
  # stdin cannot be stolen; (2) every runner invocation is fed </dev/null, so a
  # child that wants stdin gets devnull, never the framework's iterator; (3) the
  # found count is measured from the array BEFORE any adjudicator runs -- an
  # independent denominator -- and the processed count must equal it or the walk
  # REFUSES 96, because an iterator that lost entries is a framework defect, not
  # a checkpoint result; (4) per-checkpoint verdicts are captured explicitly and
  # the WORST outcome is returned by construction, since the call site suppresses
  # errexit. Message vocabulary (ADJUDICATE / rc=) is kept; fields are ADDED,
  # none renamed.
  local root="$1" phase="$2" d rc=0
  local -a dirs=()
  [[ -d "$root" ]] || { printf 'ADJUDICATE rc=96 root_missing=%s\n' "$root" >&2; return 96; }
  while IFS= read -r -d '' d; do dirs+=("$d"); done < <(find "$root" -type d \( -name 'checkpoint*' -o -name 'ckpt*' -o -name 'step_*' \) -print0 2>/dev/null)
  local found=${#dirs[@]} processed=0 ok=0 abstain=0 refuse=0
  checkpoint_found=$found
  [[ "$found" -gt 0 ]] || { printf 'ADJUDICATE rc=95 no_checkpoints_found root=%s phase=%s found=0\n' "$root" "$phase" >&2; return 95; }
  for d in "${dirs[@]}"; do
    rc=0
    run_adjudicators "$d" "$phase" </dev/null || rc=$?
    processed=$((processed+1))
    case $rc in
      0)  ok=$((ok+1)) ;;
      95) abstain=$((abstain+1)) ;;
      96) refuse=$((refuse+1)) ;;
      *)
        printf 'ADJUDICATE rc=96 original_rc=%s ckpt=%s -- undeclared exit code mapped to 96\n' "$rc" "$d" >&2
        refuse=$((refuse+1))
        ;;
    esac
  done
  [[ "$processed" -eq "$found" ]] || { printf 'ADJUDICATE rc=96 iterator_truncated processed=%s found=%s root=%s phase=%s -- the walk lost entries; a framework defect, not a checkpoint result\n' "$processed" "$found" "$root" "$phase" >&2; return 96; }
  printf 'ADJUDICATE complete root=%s phase=%s adjudicated=%s of %s checkpoint dir(s) ok=%s abstain=%s refuse=%s\n' "$root" "$phase" "$processed" "$found" "$ok" "$abstain" "$refuse"
  if (( refuse > 0 )); then return 96; fi
  if (( abstain > 0 )); then return 95; fi
  return 0
}

if [[ -z "${SLURM_JOB_ID:-}" && "${FS_SUBMIT_CHAIN:-0}" == 1 ]]; then
  # Login-node chain driver. Probe -> production -> resume(afterok: a REAL resume) + post-mortem(afterany: reporting only).
  # fs183: the operator's ACTUAL engine command, checked on the login node BEFORE any
  # allocation is burned. Every existing guard on FS_ENGINE_LAUNCH_CMD sits below the
  # SLURM_JOB_ID gate -- unset is caught at :819 and a malformed mode at :841 -- which means
  # a typo cost four queued jobs and a scheduler wait to discover. The preflight is host-side
  # and torch-free precisely so it can run here, where the interpreter may be 3.6.8.
  # FS_GPUS_PER_NODE is required and already validated above this splice point (req_env at
  # :225, integer at :344, > 0 at :345), so it is passed UNCONDITIONALLY: the emptiness a
  # conditional append would guard against cannot occur here.
  fs183_pf_args=( --launch-cmd "${FS_ENGINE_LAUNCH_CMD:-}" --backend "$FS_PLANE_DIR/$BACKEND_NAME" --procs-per-node "$FS_GPUS_PER_NODE" )
  # Pass --mode only when the operator set one. FS_ENGINE_LAUNCH_MODE is required-with-no-default
  # and read at :841, but a child process sees it only if it was exported; passing an empty
  # value would make the preflight adjudicate an empty string as a mode, whereas omitting the
  # flag lets it report C3 as UNMEASURED ("nothing to check").
  [[ -n "${FS_ENGINE_LAUNCH_MODE:-}" ]] && fs183_pf_args+=( --mode "$FS_ENGINE_LAUNCH_MODE" )
  if [[ ! -r "$FS_PLANE_DIR/fs_argv_preflight.py" ]]; then
    # A plane staged before this check existed is not a broken plane, and refusing to submit
    # because a diagnostic is missing would make the diagnostic worse than the defect it
    # finds. Absence of the checker is UNMEASURED, and it names its own remedy.
    printf 'ARGV PREFLIGHT unmeasured -- %s/fs_argv_preflight.py is not readable, so the engine command was NOT checked before submit. Remedy: redeploy the plane directory from the build (it ships this file alongside %s). Proceeding.\n' \
      "$FS_PLANE_DIR" "$BACKEND_NAME" >&2
  else
    fs183_pf_rc=0
    # Bare python3, deliberately no knob: the launcher already spells the host interpreter
    # as python3 (:582, :749), and the preflight is 3.6.8-clean precisely so that bare
    # python3 on a login node is sufficient. A knob introduced here and declared nowhere
    # would be a knob with no reader.
    python3 "$FS_PLANE_DIR/fs_argv_preflight.py" "${fs183_pf_args[@]}" || fs183_pf_rc=$?
    case "$fs183_pf_rc" in
      0) ;;
      95)
        # UNMEASURED must not block: a foreign engine entrypoint that lives only inside the
        # container is legitimately unreadable from here, and refusing every such launch would
        # make the plane engine-specific. It must also never be reported as a pass.
        printf 'ARGV PREFLIGHT unmeasured -- proceeding to submit; the checks above say which oracle was missing. This is NOT a clean bill of health.\n' >&2 ;;
      5)  fail 5 "argv preflight RED: the engine command names flags the entrypoint does not declare, or a mode the backend does not accept. Refusing before submitting; nothing was queued." ;;
      96) fail 96 "argv preflight REFUSE: FS_ENGINE_LAUNCH_CMD is unset or does not tokenize. Refusing before submitting; nothing was queued." ;;
      *)  fail 96 "argv preflight returned undeclared exit code $fs183_pf_rc; the plane's contract is 0/5/95/96" ;;
    esac
  fi
  probe_jid="$(PROBE=1 FS_SUBMIT_CHAIN=0 sbatch --partition="$FS_PARTITION" --gpus-per-node="$FS_GPUS_PER_NODE" --cpus-per-task="$FS_CPUS_PER_TASK" --mem="$FS_MEM" --time="$FS_WALLTIME" --parsable --export=ALL,PROBE=1,FS_SUBMIT_CHAIN=0 "$FS_PLANE_DIR/$(basename "$0")")"
  [[ "$probe_jid" =~ ^[0-9]+$ ]] || fail 96 "probe submit did not return a job id: '$probe_jid'"
  prod_jid="$(PROBE=0 FS_SUBMIT_CHAIN=0 sbatch --partition="$FS_PARTITION" --gpus-per-node="$FS_GPUS_PER_NODE" --cpus-per-task="$FS_CPUS_PER_TASK" --mem="$FS_MEM" --time="$FS_WALLTIME" --parsable --dependency="afterok:$probe_jid" --export=ALL,PROBE=0,FS_SUBMIT_CHAIN=0 "$FS_PLANE_DIR/$(basename "$0")")"
  [[ "$prod_jid" =~ ^[0-9]+$ ]] || fail 96 "production submit did not return a job id: '$prod_jid'"
  resume_jid="$(sbatch --partition="$FS_PARTITION" --gpus-per-node="$FS_GPUS_PER_NODE" --cpus-per-task="$FS_CPUS_PER_TASK" --mem="$FS_MEM" --time="$FS_WALLTIME" --parsable --dependency="afterok:$prod_jid" --export=ALL,FS_PHASE=resume,PROBE=0,FS_SUBMIT_CHAIN=0 "$FS_PLANE_DIR/$(basename "$0")")"
  [[ "$resume_jid" =~ ^[0-9]+$ ]] || fail 96 "resume submit did not return a job id: '$resume_jid'"
  postmortem_jid="$(sbatch --partition="$FS_PARTITION" --gpus-per-node="$FS_GPUS_PER_NODE" --cpus-per-task="$FS_CPUS_PER_TASK" --mem="$FS_MEM" --time="$FS_WALLTIME" --parsable --dependency="afterany:$prod_jid" --export=ALL,FS_PHASE=post-mortem,PROBE=0,FS_SUBMIT_CHAIN=0 "$FS_PLANE_DIR/$(basename "$0")")"
  [[ "$postmortem_jid" =~ ^[0-9]+$ ]] || fail 96 "post-mortem submit did not return a job id: '$postmortem_jid'"
  printf 'CHAIN probe=%s production=%s resume_afterok=%s postmortem_afterany=%s\n' "$probe_jid" "$prod_jid" "$resume_jid" "$postmortem_jid"
  exit 0
fi

if [[ "${FS_PHASE:-}" == post-mortem ]]; then
  # fs187: this link fires afterany precisely so it runs when production DIED -- the
  # checkpoints a failed run left on disk are the ones most worth reading. The
  # unconditional zero-exit that used to stand here recorded PASS over zero
  # adjudications: all([]) in a Slurm job costume. (The literal is elided rather
  # than written out because the stage that emits this branch post-condition-scans
  # it, and a scanner that matches its own explanatory prose has no denominator.)
  # The skip flag set below is deliberately launcher-internal -- NOT
  # exported, NOT on any allowlist: an operator-settable "skip the training" flag
  # on a training launcher is a way to produce a green run that trained nothing.
  FS_SKIP_TRAIN=1
  printf 'POST-MORTEM afterany adjudication link reached; no training launched, the checkpoint tree production left behind is adjudicated below. OUT_DIR=%s\n' "$OUT_DIR" | tee -a "$RUN_LOG"
fi

if [[ "${FS_PHASE:-}" == resume ]]; then
  # Real resume: find the newest production checkpoint, prove its recorded step > 0, then fall
  # through to the common training path with a bounded iteration budget. Resume facts cross the
  # container boundary the same way every other in-container fact does: exported under its
  # PLAIN name and carried by FS_ENV_ALLOWLIST (fs122). There is no runtime-specific path here
  # -- FS_CONTAINER_RUNTIME is a required, never-inferred axis and singularity is what THIS
  # estate happens to have, not what the framework is allowed to assume. FS_RESUME_CKPT,
  # FS_RESUME_STEP, FS_ITERATION_BUDGET and FS_EARLY_SAVE_STEPS are all on the allowlist.
  # Distinct UNMEASURED
  # status for 'cannot perform resume' is 95 -- never 0, never 96.
  resume_ckpt=""; resume_ckpt_mtime=0; resume_found=0
  while IFS= read -r -d '' d; do
    resume_found=$((resume_found+1))
    m="$(stat -c %Y "$d" 2>/dev/null || printf '0')"
    [[ "$m" =~ ^[0-9]+$ ]] || m=0
    if (( m >= resume_ckpt_mtime )); then resume_ckpt_mtime="$m"; resume_ckpt="$d"; fi
  done < <(find "$OUT_DIR" -type d \( -name 'checkpoint*' -o -name 'ckpt*' -o -name 'step_*' \) -print0 2>/dev/null)
  [[ "$resume_found" -gt 0 && -n "$resume_ckpt" ]] || fail 96 "resume requested but no production checkpoints found under $OUT_DIR (found=$resume_found); a missing checkpoint after a production leg is a finding, not a skip"
  resume_base="$(basename -- "$resume_ckpt")"
  resume_step=""
  if [[ "$resume_base" =~ [0-9]+$ ]]; then resume_step="${BASH_REMATCH[0]}"; fi
  [[ "$resume_step" =~ ^[0-9]+$ && "$resume_step" -gt 0 ]] \
    || { printf 'UNMEASURED rc=95 resume cannot prove recorded step > 0 for ckpt=%s under %s\n' "$resume_ckpt" "$OUT_DIR" >&2; exit 95; }
  FS_ITERATION_BUDGET="${FS_RESUME_ITERATION_BUDGET:-5}"
  FS_EARLY_SAVE_STEPS="${FS_RESUME_EARLY_SAVE_STEPS:-2}"
  [[ "$FS_ITERATION_BUDGET" =~ ^[0-9]+$ && "$FS_ITERATION_BUDGET" -gt 0 ]] \
    || { printf 'UNMEASURED rc=95 resume iteration budget invalid: %s\n' "$FS_ITERATION_BUDGET" >&2; exit 95; }
  [[ "$FS_EARLY_SAVE_STEPS" =~ ^[0-9]+$ && "$FS_EARLY_SAVE_STEPS" -le "$FS_ITERATION_BUDGET" ]] \
    || { printf 'UNMEASURED rc=95 resume early-save steps invalid: %s\n' "$FS_EARLY_SAVE_STEPS" >&2; exit 95; }
  export FS_ITERATION_BUDGET FS_EARLY_SAVE_STEPS
  # fs122: PLAIN names, not SINGULARITYENV_*. Both are on FS_ENV_ALLOWLIST, so the
  # backend's single forwarding path carries them on BOTH arms (--env under enroot,
  # an `env K=V` prefix under singularity) from one array. The SINGULARITYENV_ form
  # is deleted rather than kept alongside: a runtime-specific duplicate would keep
  # one arm working for a reason unrelated to the allowlist, which is exactly how
  # this defect stayed invisible.
  export FS_RESUME_CKPT="$resume_ckpt"
  export FS_RESUME_STEP="$resume_step"
  printf 'RESUME from ckpt=%s recorded_step=%s budget=%s early_save=%s (found=%s under %s); restored step must equal recorded step\n' \
    "$resume_ckpt" "$resume_step" "$FS_ITERATION_BUDGET" "$FS_EARLY_SAVE_STEPS" "$resume_found" "$OUT_DIR" | tee -a "$RUN_LOG"
fi

[[ -n "${SLURM_JOB_ID:-}" ]] || fail 96 'not in a Slurm allocation; submit with sbatch (or FS_SUBMIT_CHAIN=1 on login node)'
# fs204: three states, never a vacuous pass. The comparison that stood here gave
# SLURM_GPUS_PER_NODE a parameter-expansion default of the very value it was being
# compared against, so the equality held BY CONSTRUCTION whenever Slurm exported
# nothing -- an
# absent observable reported as agreement. Set: compare, refuse 96 on mismatch.
# Unset: UNMEASURED, stated, and continue -- NOT a refusal (that would make the
# plane specific to Slurm builds that export the variable) and NOT called a pass:
# the binding measurement of the same quantity is the in-container
# torch.cuda.device_count() comparison against FS_GPUS_PER_NODE ~25 lines below.
if [[ -n "${SLURM_GPUS_PER_NODE:-}" ]]; then
  [[ "$SLURM_GPUS_PER_NODE" == "$FS_GPUS_PER_NODE" ]] || fail 96 "SLURM_GPUS_PER_NODE mismatch: $SLURM_GPUS_PER_NODE vs $FS_GPUS_PER_NODE"
else
  echo "UNMEASURED: SLURM_GPUS_PER_NODE is not exported by this Slurm build; treated as unmeasured, never pass. The binding measurement is the in-container torch.cuda.device_count() comparison vs FS_GPUS_PER_NODE below."
fi

SUBMIT_DIR="$OUT_DIR/_submit"; mkdir -p -- "$SUBMIT_DIR"
fs_backend_init "$SUBMIT_DIR"
fs_backend_runtime_setup "$IMAGE" "$FS_GPUS_PER_NODE" "$RUN_LOG"

# L1: measure the training environment INSIDE the runtime that will do the
# training, after fs_backend_runtime_setup. The host probe is DELETED, not
# hardened. Measured on this estate: host python3 is 3.6.8 with no torch, so
# the old probe left $visible empty and every launch died rc=96 before the
# container was ever created. On a host that DID have a system torch the same
# probe would have measured the host CUDA stack -- the very stack the image
# exists to displace. Both outcomes were host properties; both were wrong.
#
# run_in_container is the runner (fs_launch_python only BUILDS a command
# string and takes a gpu count, so it cannot run this). It already resolves
# torch.__file__ in-container and asserts provenance on every invocation, and
# fs_backend_init has already run the provenance self-test, so neither is
# repeated here: a duplicated drill is not a second measurement.
#
# If run_in_container refuses, it fs_die's inside this command substitution's
# subshell, $visible comes back empty, and the regex below fails the launch.
# Fail-closed either way: unmeasured is not pass.
visible="$(run_in_container --slurm-ntasks 1 -- \
  python3 -c 'import torch; print(torch.cuda.device_count())' | tail -n 1)"
[[ "$visible" =~ ^[0-9]+$ ]] || fail 96 'could not measure visible CUDA devices inside the container; unmeasured is not pass'
[[ "$visible" == "$FS_GPUS_PER_NODE" ]] || fail 96 "visible GPUs in container ($visible) != requested FS_GPUS_PER_NODE ($FS_GPUS_PER_NODE)"

# fs129: the collective gate. Mounts are verified from inside the container (fs117 R4) and torch
# provenance from inside (R5), but until now NOTHING verified that a collective completes -- and
# unmeasured is never PASS. This runs $visible ranks via os.fork BEFORE any torch import, so no
# CUDA context is ever inherited across a fork; torch.multiprocessing.spawn is structurally
# impossible inside python3 -c (F7). Each rank contributes its rank id (sum 0..w-1 = 28 at w=8),
# not 1.0 -- a uniform probe sums to w and passes even with ranks silently duplicated onto one
# device (the #124 shape). spread catches a partial reduction. FS_PROBE_CORRUPT=1 poisons rank 0
# as the MUST_FIRE control (observed got=29.0 vs expected=28.0). Same-interpreter rule from #128:
# the process forking ranks is the same interpreter that imports torch (${FS_PYTHON:-python3}).
fs129_collective_probe='import os, sys
w = int(sys.argv[1])
os.environ["MASTER_ADDR"] = "127.0.0.1"
os.environ["MASTER_PORT"] = os.environ.get("FS_PROBE_PORT", "29947")
def child(r, w):
    import torch, torch.distributed as dist
    torch.cuda.set_device(r)
    dist.init_process_group("nccl", rank=r, world_size=w)
    val = float(r) + (1.0 if os.environ.get("FS_PROBE_CORRUPT") == "1" and r == 0 else 0.0)
    t = torch.full((1024, 1024), val, device="cuda:%d" % r, dtype=torch.float32)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    torch.cuda.synchronize()
    rc = 0
    if r == 0:
        got, spread = t.min().item(), (t.max() - t.min()).item()
        exp = float(w * (w - 1) // 2)
        ok = abs(got - exp) < 1e-3 and spread < 1e-3
        print("FS_COLLECTIVE world=%d got=%s expected=%s spread=%s verdict=%s"
              % (w, got, exp, spread, "OK" if ok else "MISMATCH"), flush=True)
        rc = 0 if ok else 1
    dist.barrier()
    dist.destroy_process_group()
    return rc
kids = []
for r in range(1, w):
    pid = os.fork()
    if pid == 0:
        try:
            os._exit(child(r, w))
        except BaseException as e:
            sys.stderr.write("FS_COLLECTIVE child %d FAULT %s: %s\n" % (r, type(e).__name__, e))
            os._exit(1)
    kids.append(pid)
rc = 0
try:
    rc = child(0, w)
except BaseException as e:
    sys.stderr.write("FS_COLLECTIVE rank0 FAULT %s: %s\n" % (type(e).__name__, e))
    rc = 1
for p in kids:
    if os.waitpid(p, 0)[1] != 0:
        rc = 1
sys.exit(rc)'
run_in_container --slurm-ntasks 1 -- \
  "${FS_PYTHON:-python3}" -c "$fs129_collective_probe" "$visible"
fs129_rc=$?
if [[ "$fs129_rc" -ne 0 ]]; then
  fail 96 "fs129: collective plane UNMEASURED or BROKEN -- launch refused (probed world=$visible). Measured failure on 8x H100 nemo_25.11.sif: SIGSEGV inside the FIRST all_reduce with fault frames in /opt/hpcx/ucx/lib/libucs.so.0 while init_process_group(nccl) SUCCEEDED -- the auto-loaded HPC-X NCCL net plugin crashes. MEASURED remedy: FS_NCCL_NET_PLUGIN=none (correct at world=8, NVLink/NVLS unaffected). NCCL_IB_DISABLE=1 does NOT help: it disables a transport, not the plugin LOAD (F2)."
fi
echo "fs129: collective probe PASS -- all_reduce verified across world=$visible ranks"



# Engine remains pluggable. The core launcher does not name NeMo/Megatron/Gemma/Qwen.
# The selected engine adapter must provide a complete in-container command in CONFIG_FILE or FS_ENGINE_LAUNCH_CMD.
LAUNCH_CMD="${FS_ENGINE_LAUNCH_CMD:-}"
LAUNCH_CMD_RAW="$LAUNCH_CMD"  # fs180: captured the operator-supplied command before the composer rewrites LAUNCH_CMD at ingestion; without this capture the record could only ever hold the composed form and 'what the operator asked for' would be lost.
[[ -n "$LAUNCH_CMD" ]] || fail 96 "FS_ENGINE_LAUNCH_CMD unset; engine adapter/config must provide launch command"

# fs123: the write-only `mounts=(...)` array that stood here is DELETED. Measured: zero
# readers -- nothing in this launcher ever expanded it, and the backend's arm-local
# `local -a mounts` is a different variable entirely. It was a dead duplicate of the
# FS_BIND_PATHS plane declared above, and a second mount declaration beside the live
# one is how the next reader concludes that mounts are declared here.

# fs124: rank multiplicity is decided by WHO FORKS THE RANKS, which is
# orthogonal to which container runtime is in use; the deleted runtime-branching
# composer let this launcher measure 8 devices, start exactly 1 process under
# --ntasks-per-node=1, and still exit 0 -- the measurement was checked but
# never load-bearing. FS_ENGINE_LAUNCH_MODE is therefore REQUIRED WITH NO
# DEFAULT (an unconfigured guard is a disabled standing rule):
#   torchrun  compose --nproc_per_node from the MEASURED gpu count;
#   wlm       the workload manager forks ranks, so assert its per-node task
#             count equals the measured count and hand srun that count below;
#   self      the engine forks its own ranks, so it must declare
#             FS_ENGINE_PROCS_PER_NODE == gpus and is stamped engine-declared.
# fs_compose_launch either refuses (nonzero exit, caught here) or echoes
# "<world_size_source>\t<final_cmd>"; parsing on the first tab keeps engine
# commands with spaces intact.
TOPO_OUT="$(fs_compose_launch "${FS_ENGINE_LAUNCH_MODE:-}" "$FS_GPUS_PER_NODE" "$LAUNCH_CMD")" \
  || fail 96 "fs_compose_launch refused: ${TOPO_OUT:-<no output>}"
WORLD_SIZE_SOURCE="${TOPO_OUT%%$'\t'*}"
LAUNCH_CMD="${TOPO_OUT#*$'\t'}"
WORLD_SIZE=$(( FS_GPUS_PER_NODE * ${SLURM_NNODES:-1} ))

# fs124: in wlm mode merely ASSERTING tasks==gpus is not enough --
# run_in_container must actually hand the measured count to srun, otherwise
# the assertion passes and exactly one process still starts: the defect.
top_args=()
if [[ "${FS_ENGINE_LAUNCH_MODE:-}" == wlm ]]; then
  top_args=(--slurm-ntasks "$FS_GPUS_PER_NODE")
fi

fs_begin_log_tee "$RUN_LOG" || true
printf 'BEGIN phase=%s probe=%s out=%s image=%s model_dir=%s dataset_dir=%s config=%s gpus=%s\n' "${FS_PHASE:-train}" "$PROBE" "$OUT_DIR" "$IMAGE" "$MODEL_DIR" "$DATASET_DIR" "$CONFIG_FILE" "$FS_GPUS_PER_NODE"
printf 'LAUNCH_TOPOLOGY mode=%s gpus=%s world_size=%s world_size_source=%s\n' "${FS_ENGINE_LAUNCH_MODE:-unset}" "$FS_GPUS_PER_NODE" "$WORLD_SIZE" "$WORLD_SIZE_SOURCE" | tee -a "$RUN_LOG"

if [[ "${FS_SKIP_TRAIN:-0}" == 1 ]]; then
  # fs187: only the training launch is skipped. Backend init, runtime setup, the
  # bind plane, the in-container GPU census and fs_compose_launch have all already
  # run on the ordinary path above, and the shared adjudication tail below is what
  # produces the verdict -- reused verbatim, not duplicated.
  printf 'POST-MORTEM: no training launched; adjudicating the checkpoint tree production left behind. OUT_DIR=%s\n' "$OUT_DIR" | tee -a "$RUN_LOG"
  rc=0
else
  set +e
# --- fs180: launch provenance writer (finding #180) ----------
# WHY: a completed 8-GPU job's own 936-line log held zero occurrences of the
# launch command, zero of --resume-tolerance, and zero combined of
# --model-path/--dataset-path/--sequence-length: the run logged everything
# about itself except what ran. The trainer's resume knob is flag-only
# (sourced with no environment name) and required, so a value that can only
# arrive on a command line was recorded by nothing, and reconstructing a
# four-hour-old run's invocation from its artifacts was attempted and
# failed. This block writes what ACTUALLY runs, placed after the composer
# rewrote LAUNCH_CMD -- the executed command, not the requested one.
# The path is DERIVED from RUN_LOG by suffix substitution, never recomputed
# from LOG_DIR and SLURM_JOB_ID (finding #150: a writer and a reader
# computing one name two ways disagree). Redaction is by NAME only --
# value-shape matching misfires on paths and misses short values -- and
# every substitution is counted, because a record that has been altered
# must say so: redactions=0 may be treated as exact, redactions>0 means
# values must be re-supplied by hand. The write is fully guarded: a
# failure degrades to an announced write=FAILED state, never a failed
# launch. The name pattern is assembled from single-character classes so
# matching is case-insensitive without a literal word list in this file.
FS180_PROV_PATH="${RUN_LOG%.log}.provenance.json"
FS180_REDACTS=0
_FS180_NPAT="[Kk][Ee][Yy]|[Tt][Oo][Kk][Ee][Nn]|[Ss][Ee][Cc][Rr][Ee][Tt]"
_FS180_NPAT="${_FS180_NPAT}|[Pp][Aa][Ss][Ss][Ww][Oo][Rr][Dd]|[Pp][Aa][Ss][Ss][Ww][Dd]|[Cc][Rr][Ee][Dd][Ee][Nn][Tt][Ii][Aa][Ll]"

fs180_json_escape() {
  local s="$1"
  s="${s//\\/\\\\}"
  s="${s//\"/\\\"}"
  s="${s//$'\n'/\\n}"
  s="${s//$'\r'/\\r}"
  s="${s//$'\t'/\\t}"
  s="${s//[[:cntrl:]]/}"
  printf '%s' "$s"
}

fs180_redact_cmd() {
  # Input: $1. Output: FS180_REDACTED. Side effect: FS180_REDACTS is
  # incremented once per substitution, across the --flag VALUE, --flag=VALUE
  # and NAME=VALUE forms.
  #
  # fs180 MEASURED: the first draft rebuilt the string with ${s/\"$m\"/...},
  # whose pattern carries literal double quotes that a command line does not
  # have. No substitution ever landed, the loop condition stayed true, and one
  # --api-key flag hung bash for >60s -- immediately before exec, on every
  # launch. The rewrite consumes the string left to right: each iteration
  # appends the prefix plus the redacted form to out and advances rest past
  # the match, so termination is structural rather than a side effect of the
  # replacement happening to match. Every ${var%%"$m"*} and ${var#"$pre$m"}
  # quotes its pattern, which is what makes bash match it literally; unquoted,
  # a command line containing * or [ would splice itself.
  local s="$1" out="" rest="$1" re m v flag pre
  re="--[A-Za-z0-9_-]*(${_FS180_NPAT})[A-Za-z0-9_-]*[[:space:]]+([^[:space:]\"'<][^[:space:]]*)"
  while [[ "$rest" =~ $re ]]; do
    m="${BASH_REMATCH[0]}"
    v="${BASH_REMATCH[2]}"
    flag="${m%"$v"}"
    pre="${rest%%"$m"*}"
    out="${out}${pre}${flag}<redacted>"
    rest="${rest#"$pre$m"}"
    FS180_REDACTS=$((FS180_REDACTS + 1))
  done
  s="${out}${rest}"; out=""; rest="$s"
  re="--[A-Za-z0-9_-]*(${_FS180_NPAT})[A-Za-z0-9_-]*=[^[:space:]\"'<][^[:space:]]*"
  while [[ "$rest" =~ $re ]]; do
    m="${BASH_REMATCH[0]}"
    pre="${rest%%"$m"*}"
    out="${out}${pre}${m%%=*}=<redacted>"
    rest="${rest#"$pre$m"}"
    FS180_REDACTS=$((FS180_REDACTS + 1))
  done
  s="${out}${rest}"; out=""; rest="$s"
  re="[A-Za-z0-9_]*(${_FS180_NPAT})[A-Za-z0-9_]*=[^[:space:]\"'<][^[:space:]]*"
  while [[ "$rest" =~ $re ]]; do
    m="${BASH_REMATCH[0]}"
    pre="${rest%%"$m"*}"
    out="${out}${pre}${m%%=*}=<redacted>"
    rest="${rest#"$pre$m"}"
    FS180_REDACTS=$((FS180_REDACTS + 1))
  done
  # A redacted value begins with '<', which every value class above excludes,
  # so a site already substituted can never be counted a second time.
  FS180_REDACTED="${out}${rest}"
}

fs180_emit_provenance() {
  # stdout: one JSON object. Called with its output redirected to
  # FS180_PROV_PATH inside an if-condition, so any failure here degrades to
  # the announced write=FAILED state and can never abort the launch.
  local esc_composed esc_raw mode_json top_json top_state job_json env_body ent v val
  local unexp_body dcl flags
  fs180_redact_cmd "${LAUNCH_CMD:-}"
  esc_composed="$(fs180_json_escape "$FS180_REDACTED")"
  fs180_redact_cmd "${LAUNCH_CMD_RAW:-}"
  esc_raw="$(fs180_json_escape "$FS180_REDACTED")"
  if [[ -n "${FS_ENGINE_LAUNCH_MODE:-}" ]]; then
    mode_json="\"$(fs180_json_escape "$FS_ENGINE_LAUNCH_MODE")\""
  else
    # explicit null, never a silent omission: a reader must be able to
    # distinguish 'not set' from 'the writer forgot'.
    mode_json="null"
  fi
  top_json="null"; top_state="absent"
  if [[ "${top_args[@]+set}" == set && "${#top_args[@]}" -gt 0 ]]; then
    val="$(printf '%s ' "${top_args[@]}")"; val="${val% }"
    top_json="\"$(fs180_json_escape "$val")\""
    top_state="present"
  fi
  if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    job_json="\"$(fs180_json_escape "$SLURM_JOB_ID")\""
  else
    job_json="null"
  fi
  # The fs_env half: the trainer's knobs arrive as env AND as argv, so both
  # halves must be on record for the pair to be reproducible. Name-based
  # redaction keeps the name and replaces the value with the marker.
  env_body=""
  unexp_body=""
  while IFS= read -r v; do
    [[ -n "$v" ]] || continue
    # fs180: declare -p prints the flags as the second word -- reading them
    # that way, instead of globbing the whole line for an x, keeps a VALUE
    # containing x from being read as the export flag.
    dcl="$(declare -p "$v" 2>/dev/null)"; flags="${dcl#declare }"; flags="${flags%% *}"
    if [[ "$flags" != *x* ]]; then
      if [[ -n "$unexp_body" ]]; then unexp_body="${unexp_body}, "; fi
      unexp_body="${unexp_body}\"$(fs180_json_escape "$v")\""
    fi
    if [[ "$v" =~ (${_FS180_NPAT}) ]]; then
      FS180_REDACTS=$((FS180_REDACTS + 1))
      ent="  \"$(fs180_json_escape "$v")\": \"<redacted>\""
    else
      if [[ "${!v+x}" == x ]]; then val="${!v}"; else val=""; fi
      ent="  \"$(fs180_json_escape "$v")\": \"$(fs180_json_escape "$val")\""
    fi
    if [[ -n "$env_body" ]]; then
      env_body="${env_body},
${ent}"
    else
      env_body="$ent"
    fi
  # fs180 MEASURED: compgen -e lists only EXPORTED names, and the launcher
  # sets FS_ITERATION_BUDGET and FS_EARLY_SAVE_STEPS with a bare assignment
  # on the resume arm. run_in_container forwards by NAME from an allowlist,
  # so those two reach the trainer while an exported-only census would have
  # silently dropped the pair of numbers that define the resume segment --
  # a reproducibility record omitting exactly what it exists to hold.
  # compgen -v is the shell's own set; the export state is recorded per name
  # rather than used as a filter, so the record states its denominator.
  done < <(compgen -v FS_ | LC_ALL=C sort)
  printf '{\n'
  printf '  \"launch_cmd_composed\": \"%s\",\n' "$esc_composed"
  printf '  \"launch_cmd_raw\": \"%s\",\n' "$esc_raw"
  printf '  \"engine_launch_mode\": %s,\n' "$mode_json"
  printf '  \"world_size\": %s,\n' "${WORLD_SIZE:-null}"
  printf '  \"world_size_source\": \"%s\",\n' "$(fs180_json_escape "${WORLD_SIZE_SOURCE:-}")"
  printf '  \"gpus_per_node\": %s,\n' "${FS_GPUS_PER_NODE:-null}"
  printf '  \"top_args\": %s,\n' "$top_json"
  printf '  \"top_args_state\": \"%s\",\n' "$top_state"
  printf '  \"out_dir\": \"%s\",\n' "$(fs180_json_escape "${OUT_DIR:-}")"
  printf '  \"image\": \"%s\",\n' "$(fs180_json_escape "${IMAGE:-}")"
  printf '  \"model_dir\": \"%s\",\n' "$(fs180_json_escape "${MODEL_DIR:-}")"
  printf '  \"dataset_dir\": \"%s\",\n' "$(fs180_json_escape "${DATASET_DIR:-}")"
  printf '  \"config_file\": \"%s\",\n' "$(fs180_json_escape "${CONFIG_FILE:-}")"
  printf '  \"phase\": \"%s\",\n' "$(fs180_json_escape "${FS_PHASE:-train}")"
  printf '  \"probe\": %s,\n' "${PROBE:-null}"
  printf '  \"job_id\": %s,\n' "$job_json"
  # hostname is deliberately NOT emitted: it is estate-identifying and this
  # record is written onto a shared filesystem.
  printf '  \"nodes\": %s,\n' "${SLURM_NNODES:-1}"
  printf '  \"fs_env_scope\": \"%s\",\n' 'every FS_ name this shell held at launch (compgen -v), not only the exported subset; export state is recorded per name in fs_env_not_exported rather than used as a filter'
  printf '  \"fs_env\": {\n%s\n  },\n' "$env_body"
  printf '  \"fs_env_not_exported\": [%s],\n' "$unexp_body"
  printf '  \"redactions\": %s\n' "$FS180_REDACTS"
  printf '}\n'
}

if fs180_emit_provenance > "$FS180_PROV_PATH" 2>/dev/null; then
  printf 'PROVENANCE path=%s write=ok redactions=%s\n' "$FS180_PROV_PATH" "$FS180_REDACTS" | tee -a "$RUN_LOG" || true
else
  # degraded, not failed -- and it SAYS so, so a silent gap can never pass
  # for a record.
  printf 'PROVENANCE path=%s write=FAILED redactions=%s (launch continues; run is degraded, not failed)\n' "$FS180_PROV_PATH" "$FS180_REDACTS" | tee -a "$RUN_LOG" || true
fi
unset -f fs180_json_escape fs180_redact_cmd fs180_emit_provenance 2>/dev/null || true
unset _FS180_NPAT FS180_REDACTED FS180_PROV_PATH FS180_REDACTS
# --- end fs180 ---------------------------------------------------------------
  run_in_container --workdir "$OUT_DIR" "${top_args[@]}" -- bash -lc "$LAUNCH_CMD" 2>&1 | tee -a "$RUN_LOG"
  rc="${PIPESTATUS[0]}"
  set -e
fi
if [[ "$rc" -ne 0 ]]; then
  # fs175: finding #174 -- the line this replaces handed the hard-stop helper
  # $$, the launcher's OWN pid, and with no trap anywhere bash took SIGTERM's
  # default action and died on the spot: on job 37304 the END line never ran
  # (a 452-line log of a failed run holds zero END and zero FATAL lines), the
  # exit never ran (sacct: 37304.batch CANCELLED ExitCode 0:15 = signal 15,
  # which an orchestrator reads as a human cancel, not a declared state), and
  # the helper's enroot force-remove -- the arm that exists so no orphaned
  # rank survives -- was unreachable. srun has already returned here, so the
  # ranks are already reaped and only container cleanup remains.
  fs_cleanup_orphans || true
  # fs175: finding #171 -- do NOT propagate $rc verbatim: torchrun flattened
  # the trainer's declared state into it (on 37304: trainer rc 2 -> torchrun
  # 'failed (exitcode: 2) local_rank: 0' -> srun 1), so a declared UNMEASURED
  # surfaced as a generic failure. Map (rc, log) through the backend's verdict
  # mapper; the END line carries the raw srun rc AND the mapped code AND the
  # parsed verdict, so the flattening is visible in the log rather than
  # silently corrected.
  fs175_reason="${RUN_LOG}.fs175_reason"
  mapped_rc="$(fs_map_run_verdict "$rc" "$RUN_LOG" 2>"$fs175_reason")" || fail 96 "fs175: verdict mapper unavailable/failed (backend drift); refusing to guess a state"
  map_reason="$(cat "$fs175_reason" 2>/dev/null || true)"; rm -f "$fs175_reason"
  printf 'END rc=%s mapped_rc=%s phase=train FAILED (%s)\n' "$rc" "$mapped_rc" "$map_reason" | tee -a "$RUN_LOG"
  exit "$mapped_rc"
fi
# fs193: finding #193 -- the mapper below has an unstated precondition: a
# trainer ran and was supposed to declare a verdict. The post-mortem arm
# violates it (no trainer ran, so the run log carries no verdict line by
# construction), and an unguarded mapper on that arm maps to 95 and leaves
# before the adjudication tail is ever reached -- measured on job 37347,
# 51 log lines, zero adjudications. The mapper is therefore scoped to the
# arm that launched a trainer; the post-mortem arm's verdict source is the
# shared adjudication tail, which is what fs187 routed it to.
if [[ "${FS_SKIP_TRAIN:-0}" != 1 ]]; then
  # fs175: finding #171, success arm -- rc==0 is not PASS until the trainer's
  # declaration is checked: a run that exits clean having declared nothing is the
  # vacuous-pass hole, and before this block it fell straight through to
  # adjudication. Map (rc, log) and refuse anything that is not 0 BEFORE
  # adjudicating. This is the whole point of the doctrine and it costs these
  # lines only.
  fs175_reason="${RUN_LOG}.fs175_reason"
  mapped_rc="$(fs_map_run_verdict "$rc" "$RUN_LOG" 2>"$fs175_reason")" || fail 96 "fs175: verdict mapper unavailable/failed (backend drift); refusing to guess a state"
  map_reason="$(cat "$fs175_reason" 2>/dev/null || true)"; rm -f "$fs175_reason"
  if [[ "$mapped_rc" != 0 ]]; then
    printf 'END rc=%s mapped_rc=%s phase=train FAILED (%s)\n' "$rc" "$mapped_rc" "$map_reason" | tee -a "$RUN_LOG"
    exit "$mapped_rc"
  fi
else
  printf 'fs193: trainer-verdict mapper deliberately not applied on this arm -- no trainer was launched, so there is no declaration to check; the adjudication tail below is the verdict source for this arm\n' | tee -a "$RUN_LOG"
fi

# Probe must have produced an early-save checkpoint in *_probe; production must produce saves in stable OUT_DIR.
adj_root="$OUT_DIR"
adjudicate_tree "$adj_root" "${FS_PHASE:-train}" || ar=$?
ar="${ar:-0}"
if [[ "$ar" -ne 0 ]]; then
  printf 'END rc=%s phase=adjudicate FAILED observed_saves=%s\n' "$ar" "$checkpoint_observed" | tee -a "$RUN_LOG"
  exit "$ar"
fi
[[ "$checkpoint_observed" -gt 0 ]] || fail 95 'no checkpoint-save units observed to adjudicate; UNMEASURED is not PASS'
# fs176: positivity is not coverage. Job 37308 PASSED with the observed counter
# at 1 against TWO checkpoint saves on disk; the early save the resume proof
# depends on was never adjudicated and the -gt 0 refusal above certified 1-of-2
# as full. Compare against the independent denominator measured BEFORE any
# adjudicator ran: partial coverage is RED, not PASS.
[[ "$checkpoint_observed" -eq "$checkpoint_found" ]] || fail 5 "fs176: partial adjudication coverage: observed=${checkpoint_observed} found=${checkpoint_found}; only a fraction of the checkpoint saves on disk was adjudicated -- partial coverage is not PASS"
printf 'END rc=0 phase=%s checkpoint_saves_adjudicated=%s checkpoint_saves_found=%s\n' "${FS_PHASE:-train}" "$checkpoint_observed" "$checkpoint_found" | tee -a "$RUN_LOG"
exit 0
