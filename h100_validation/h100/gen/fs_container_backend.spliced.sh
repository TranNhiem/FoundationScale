#!/usr/bin/env bash
# ============================================================================
# fs_container_backend.sh — the ONE in-container executor for the E4B 1-tray
# launchers, with TWO backends behind a single function:
#   slurm  : pyxis + srun        (the historic #SBATCH path — kept whole; Slurm
#                                 may come back, and when it does this arm must
#                                 behave exactly as these scripts always did)
#   enroot : enroot + torchrun   (the ONLY executable path on <compute-node> today)
#
# Both launchers source this file and route EVERY in-container step through it:
# preflight probes and the training run alike. The only backend branch in the
# whole launcher pair lives in run_in_container() below. If a probe could be
# wired to the host interpreter while training ran in the image, preflight
# would prove properties of a python the training run never executes — the
# two-interpreter class of defect this file exists to make impossible.
#
# Measured estate facts encoded here (fix22_shared.md, 2026-08-23):
#   s1  sbatch/srun/squeue/sinfo GONE; the pyxis spank plugin (.so absent) gone
#       with them. Until this file existed every `srun --container-image` line
#       in the launchers was dead code end to end.
#   s3  enroot exists at /usr/bin/enroot; squashfuse does NOT -> `enroot start
#       <img>.sqsh` cannot work; the image must be UNPACKED once via
#       `enroot create`. $HOME/.enroot also holds 'g4export', built from *a*
#       nemo-automodel image of UNRECORDED provenance — which is why container
#       reuse here is conditional on a recorded source match, never on a
#       matching name, and why a mismatch is a hard error rather than a
#       warning or an auto-removal.
#   s4  the only writable container store is shared $HOME (26T avail); the
#       PROVEN value is ENROOT_DATA_PATH=$HOME/.enroot. The unset default
#       ($HOME/.local/share/enroot) is a DIFFERENT store: `enroot list` there
#       shows 13 vllm-judge containers and nothing training-related.
#   s7  $HOME/.local/.../site-packages *.pth files auto-execute at interpreter
#       startup INSIDE the container ($HOME mounts over itself) and shadow its
#       CUDA stack; the failure then masquerades as a broken image. So
#       PYTHONNOUSERSITE=1 is SET below, not merely added to an export list —
#       an exported-but-unset variable is not a 1.
#   s8a $HOME is NOT auto-mounted by enroot; without --mount, HOME inside is an
#       empty dir of the same name and the payload "does not exist". /dev and
#       /sys are mounted for IB, replicated from the off-Slurm run that worked
#       (12/12 steps, 4 trays).
#   s8b NCCL pins: NCCL_SOCKET_IFNAME / GLOO_SOCKET_IFNAME = bond0, and
#       NCCL_IB_HCA left UNSET — the measured failure was that mlx5 prefix-
#       matches all eight mlx5_* devices.
#   s8c NCCL_MNNVL_ENABLE=0 was REQUIRED on a FOUR-tray off-Slurm run (GB200
#       NVL trays select multi-node NVLink, which needs the Slurm-prolog IMEX
#       domain). Whether a SINGLE tray even selects MNNVL is UNVERIFIED — the
#       honest denominator is 4 trays, not 1. We set 0 anyway: one tray has no
#       MNNVL peer, so the pin is at worst inert here and at best prevents an
#       IMEX-less hang.
#   s8d never launch onto a tray still draining (< ~2 GiB used per GPU): the
#       first new rank segfaults in c10::cuda::device_count() and the
#       surviving ranks hang forever. The gate below polls with a bounded
#       timeout, and the timeout is a REFUSAL, not a warning that proceeds.
#   s8e an ssh inside a heredoc needs -n; this file contains no such ssh.
#   s9  the master:8081 tripwire is a STANDING RULE before any Slurm submit.
#       It is moot off-Slurm (no prolog runs) and must NOT be deleted from the
#       sbatch path — the slurm arm enforces it, timeout-bounded so a dead
#       master fails CLOSED instead of hanging a job that could have been good.
#
# Operator knobs: FS_BACKEND=auto|slurm|enroot (default auto — inside a real
#   allocation (SLURM_JOB_ID set, not by us) -> slurm; otherwise enroot).
#   FS_GPU_DRAIN_MAX_MIB (default 2048), FS_GPU_DRAIN_TIMEOUT_S (default 1800).
# ============================================================================

# This is a library: the launchers source it. Executing it directly is a
# usage error, stated rather than silently no-opping.
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  echo "fs_container_backend.sh is meant to be SOURCED by a launcher; executing it decides nothing." >&2
  exit 2
fi

# Callers run under `set -euo pipefail` (LoRA) or `set -uo pipefail` (full-FT).
# Everything below is safe under both: fallible commands sit in conditions,
# carry `|| true`, or refuse via fs_die. Bash 5.2 is the estate standard.

fs_die() { echo "FATAL: $*" >&2; exit 1; }

fs_selftest_torch_provenance() {
  local host_leak=${HOME:-/root}/.local/lib/python3.12/site-packages/torch/__init__.py
  local -a paths=(
    "$host_leak"
    "/usr/local/lib/python3.12/dist-packages/torch/__init__.py"
    "/tmp/usr/local/lib/python3.12/dist-packages/torch/__init__.py"
    ""
  )
  local -a expect=(0 1 0 0)
  local i p rc got exp ok=0 bad=0 fired=0 n=${#paths[@]}
  for i in "${!paths[@]}"; do
    p=${paths[$i]}
    exp=${expect[$i]}
    fs_assert_torch_provenance "$p"
    rc=$?
    if (( rc == 0 )); then got=1; else got=0; fi
    if (( got == exp )); then
      ok=$((ok + 1))
    else
      bad=$((bad + 1))
    fi
    if [[ $p == "$host_leak" ]] && (( exp == 0 && rc != 0 )); then
      fired=1
    fi
  done
  printf '%d of %d\n' "$ok" "$n"
  if (( bad == 0 && fired == 1 )); then
    return 0
  fi
  return 1
}

fs_assert_torch_provenance() {
  local path=${1-}
  local prefix=${FS_CONTAINER_PY_PREFIX:-/usr/local/lib/python3.12/dist-packages}
  prefix=${prefix%/}
  [[ -n $path && -n $prefix && $prefix != "/" ]] || return 1
  [[ $path == "$prefix" || $path == "$prefix/"* ]]
}

fs_env_forward_allowlisted() {
  local name=${1-} item
  [[ $# -eq 1 && -n $name ]] || return 1
  for item in "${FS_ENV_ALLOWLIST[@]-}"; do
    [[ $item == "$name" ]] && return 0
  done
  return 1
}

# ---------------------------------------------------------------------------
# fs_backend_init [submit_dir] — select backend, prove node identity, THEN mint
#
# ORDERING IS THE SECURITY PROPERTY. The guarding path never reads a value this
# process could have written; the minted SLURM_* variables exist solely for
# downstream readers (banner arithmetic, geometry divisions, the run manifest's
# --job-id, the log filenames the post-run gates grep) and are stamped only
# AFTER the ground-truth guard has passed. See census in the change request.
# ---------------------------------------------------------------------------
# ---- fs_backend_init ----
fs_backend_init() {
  local submit_dir=${1:-}

  # ---------------------------------------------------------------------------
  # R1/R2 — TWO axes, both caller-set, NEITHER inferred. The old code ran
  #   if [[ -n SLURM_JOB_ID ]]; then FS_BACKEND=slurm; else FS_BACKEND=enroot; fi
  # which conflated ALLOCATION with RUNTIME and inferred both from a variable
  # Slurm writes. On the measured estate (singularity-ce 4.1.2 ONLY; no enroot,
  # no docker, no podman; sbatch and srun exist but pyxis does NOT, so
  # `srun --container-image` fails) that inference is a live bug: srun existing
  # is not pyxis existing. Auto-detection is barred for the same reason
  # FS_ALLOWED_NODE has no default — an unconfigured guard is a disabled
  # standing rule, and an auto-detected runtime is an accident of $PATH.
  # Both variables are therefore REQUIRED with deliberately NO DEFAULT, and the
  # refusal lives HERE (unlike FS_ALLOWED_NODE, whose refusal must wait for its
  # points of use) because every later arm decision reads exactly these two.
  [[ -n "${FS_CONTAINER_RUNTIME:-}" ]] || \
    fs_die "FS_CONTAINER_RUNTIME is unset/empty (required, no default by design). Export FS_CONTAINER_RUNTIME=enroot|singularity — the container runtime this estate actually has. Auto-detection is barred: it would make the runtime an accident of \$PATH."
  case "$FS_CONTAINER_RUNTIME" in
    enroot|singularity) ;;
    *) fs_die "FS_CONTAINER_RUNTIME must be enroot|singularity, got '$FS_CONTAINER_RUNTIME'" ;;
  esac
  [[ -n "${FS_ALLOCATION:-}" ]] || \
    fs_die "FS_ALLOCATION is unset/empty (required, no default by design). Export FS_ALLOCATION=slurm|local — who allocated these nodes, independent of which container runtime launches on them. It is never inferred from SLURM_JOB_ID: on this estate sbatch exists but pyxis does not, so 'SLURM_JOB_ID set therefore pyxis arm' is a bug, not a heuristic."
  case "$FS_ALLOCATION" in
    slurm|local) ;;
    *) fs_die "FS_ALLOCATION must be slurm|local, got '$FS_ALLOCATION'" ;;
  esac

  # ---------------------------------------------------------------------------
  # R3 — FS_ENV_ALLOWLIST, defined ONCE here; every other function READS it and
  # none redefine it. The forwarding DENYLIST could not be repaired (only
  # extended) because the two runtimes forward in OPPOSITE directions: enroot
  # forwards NOTHING unless told; singularity forwards the HOST ENVIRONMENT
  # unless told not to. A denylist must enumerate every hostile variable, and
  # the one that bit us — host user-site torch 2.9.0+cu128 shadowing container
  # torch 2.11.0a0+nv26.02 under one image, same `python` binary, different
  # sys.path — is not the last. The allowlist makes the direction a property
  # of FoundationScale instead of whichever runtime is loaded; the kept
  # fs_env_forward_denylisted runs as the subordinate second check, belt and
  # braces, and no longer decides.
  #
  # Contents: EXACTLY the variables the enroot arm measurably depends on — the
  # minted SLURM_* set below (MASTER_PORT arithmetic in the launchers reads
  # SLURM_JOB_ID), the rendezvous, and the s7/s8 containment exports. Dropping
  # one silently is a worse failure than the leak this list exists to stop, so
  # nothing here was dropped and nothing unevidenced was added; bash arrays do
  # not survive `export` to children, which is fine — every consumer is a
  # function in THIS shell, and container forwarding is fs_env_forward_allowlisted's
  # job, not the environment's.
  FS_ENV_ALLOWLIST=(
    SLURM_JOB_ID
    SLURM_JOB_NODELIST
    SLURM_NNODES
    SLURM_NTASKS
    SLURM_NTASKS_PER_NODE
    SLURM_SUBMIT_DIR
    MASTER_ADDR
    MASTER_PORT
    PYTHONNOUSERSITE
    HF_HUB_OFFLINE                 # must cross: offline estate must not reach network for HF hub
    TRANSFORMERS_OFFLINE           # must cross: offline estate must not reach network for transformers
    OUT_DIR                        # must cross: trainer needs its output path inside container
    OMP_NUM_THREADS                # must cross: avoid thread oversubscription across 8 ranks
    TOKENIZERS_PARALLELISM         # must cross: avoid fork warnings / tokenizer deadlock risk
    TORCH_NCCL_ASYNC_ERROR_HANDLING # must cross: fail fast instead of hanging on NCCL error
    NCCL_DEBUG                     # must cross: preserve NCCL diagnostics inside container
    FS_ITERATION_BUDGET            # must cross: L4 wires tools/fs_train.py to read it,
                                   #   and fs_train.py runs INSIDE the container
    FS_EARLY_SAVE_STEPS            # must cross: same reader, same reason
    NCCL_SOCKET_IFNAME
    GLOO_SOCKET_IFNAME
    NCCL_MNNVL_ENABLE
  )

  # ---------------------------------------------------------------------------
  # Standing-rule vocabulary — parameterized for publication (docs/DECISIONS.md
  # forbids committing cluster-internal identifiers; in code they live ONLY in
  # these variables):
  #
  #   CLUSTER_HOME — estate root; pseudonym <CLUSTER_HOME>. Defaults to $HOME,
  #     which is BEHAVIOUR-PRESERVING on the allowed tray: measured fact (see
  #     the fix32 note later in this file) that on the allowed node $HOME IS
  #     the account home, so nothing changes there. Parameterizing lets the
  #     contract suite drive THIS code through a sandbox HOME
  #     (`env -i ... HOME="$SANDBOX/h2"`) — extending the fix32 precedent, not
  #     inventing one.
  : "${CLUSTER_HOME:=$HOME}"
  #   FS_ALLOWED_NODE — the one node name/prefix this estate may run on;
  #     pseudonym <compute-node>. REQUIRED, with deliberately NO DEFAULT and NO
  #     check at this point: the refusal lives at the two points of USE below.
  #     Rationale: a guard that cannot fire is not a guard — any default would
  #     let a typo silently lift the standing rule that keeps this estate off
  #     another team's hardware — while refusing HERE would turn red every
  #     contract-suite leg that sources/execs this file on a path that never
  #     reaches a guard. Do not add a default; do not "harden" this into a
  #     source-time refusal.
  #   FS_FORBIDDEN_NODES — optional, space-separated denylist encoding the
  #     standing "never the other team's node" rule; pseudonym
  #     <other-team-node>. Checked BEFORE the allowlist at each guard, so an
  #     explicit denial always beats a sloppy allow-prefix. Empty/unset means
  #     the allowlist alone governs (allowlist is primary; this is defence in
  #     depth).
  : "${FS_FORBIDDEN_NODES:=}"

  # ---------------------------------------------------------------------------
  # Runtime-axis presence checks — orthogonal to allocation (R2). nvidia-smi
  # is required under BOTH runtimes: the drain gate (s8d) is host-side, and a
  # sweep over zero units BLOCKS regardless of which runtime launches.
  case "$FS_CONTAINER_RUNTIME" in
    enroot)
      command -v enroot >/dev/null 2>&1 || \
        fs_die "FS_CONTAINER_RUNTIME=enroot but enroot is not on PATH (s3: expected /usr/bin/enroot on the node named by FS_ALLOWED_NODE, currently '${FS_ALLOWED_NODE:-<unset>}' — wrong node?)."
      ;;
    singularity)
      command -v singularity >/dev/null 2>&1 || \
        fs_die "FS_CONTAINER_RUNTIME=singularity but singularity is not on PATH (measured estate: singularity-ce 4.1.2). Declaring the runtime does not install it."
      ;;
  esac
  command -v nvidia-smi >/dev/null 2>&1 || \
    fs_die "nvidia-smi missing: the drain gate (s8d) can examine 0 of 4 GPUs, and a sweep over zero units BLOCKS."

  if [[ "$FS_ALLOCATION" == slurm ]]; then
    # ------------------------- slurm ALLOCATION arm --------------------------
    # Allocation, not runtime: this arm encodes who owns the node list. It
    # says NOTHING about how containers launch — that is FS_CONTAINER_RUNTIME's
    # axis, and on this estate pyxis is absent so no "slurm implies pyxis"
    # rule may survive here.
    command -v srun >/dev/null 2>&1 || \
      fs_die "FS_ALLOCATION=slurm but srun is not on PATH (s1: measured ABSENT on this estate 2026-08-23). Resubmit with FS_ALLOCATION=local outside an allocation, or bring Slurm back."
    # The two standing-rule checks the launchers historically ran at top
    # level, preserved verbatim in effect: an ALLOWLIST over a value Slurm —
    # not this script — writes.
    [[ -n "${SLURM_JOB_ID:-}" ]] || \
      fs_die "submit via sbatch; the node guard reads SLURM_JOB_NODELIST and cannot verify the allocation interactively."
    # FS_ALLOWED_NODE: REQUIRED, NO DEFAULT — fail closed at the point of use.
    # A guard that cannot fire is not a guard; defaulting this variable would
    # let a typo silently disable the standing rule that keeps this estate off
    # the other team's hardware (<other-team-node> in docs/DECISIONS.md).
    [[ -n "${FS_ALLOWED_NODE:-}" ]] || \
      fs_die "FS_ALLOWED_NODE is unset/empty (required, no default by design). Export FS_ALLOWED_NODE=<compute-node> — the one node this estate may run on — and resubmit. Without it the node guard refuses; it never passes."
    # Denylist BEFORE allowlist: an explicit denial always beats a sloppy
    # allow-prefix. ${FS_FORBIDDEN_NODES:-} is left UNQUOTED in the for-list on
    # purpose — it is a space-separated set and word-splitting IS the iteration.
    fs_guard_seen="${SLURM_JOB_NODELIST:-}"
    for fs_guard_denied in ${FS_FORBIDDEN_NODES:-}; do
      case "$fs_guard_seen" in
        ${fs_guard_denied}*) fs_die "STANDING RULE VIOLATION: landed on '$fs_guard_seen', matching FS_FORBIDDEN_NODES entry '$fs_guard_denied' — the other team's node. scancel this job." ;;
      esac
    done
    # ${FS_ALLOWED_NODE} below is deliberately UNQUOTED in the case pattern: the
    # value must act as a GLOB PREFIX so a Slurm NODELIST like
    # '<node1>,<node2>' matches the configured allowed node. Quoting it would
    # collapse the pattern to a literal string and silently disarm the guard.
    # Do not "fix" the quoting.
    case "$fs_guard_seen" in
      ${FS_ALLOWED_NODE}*) ;;
      *) fs_die "STANDING RULE VIOLATION: landed on '${fs_guard_seen:-<unset>}'. Only nodes matching FS_ALLOWED_NODE ('$FS_ALLOWED_NODE') are allowed; entries in FS_FORBIDDEN_NODES ('${FS_FORBIDDEN_NODES:-none}') — the other team's hardware — are refused first. scancel this job." ;;
    esac
    # s9 tripwire, kept on the sbatch path. The standing rule checks this
    # BEFORE submit from the login node; inside the job it is a cheap
    # re-assertion. `timeout` bounds the connect: an unroutable master must
    # fail CLOSED (refuse the launch), never HUNG.
    ( timeout 15 bash -c '(exec 3<>/dev/tcp/master/8081)' ) || \
      fs_die "standing-rule tripwire: master:8081 is not reachable from here (s9). Refusing a Slurm launch."
    FS_USE_TORCHRUN=0
  else
    # ------------------------- local ALLOCATION arm --------------------------
    # THREAT MODEL — why a pre-set SLURM_* is REFUSED, not honored. Off-Slurm
    # there is no allocator, so any SLURM_JOB_ID / SLURM_JOB_NODELIST found in
    # the environment was written by a caller or a driver script; checking such
    # a string would be the guard that cannot fail — the self-fulfilling shape
    # this repository refuses (and which the LoRA launcher's allowlist comment
    # already narrates). We mint these variables ourselves, AFTER the kernel
    # ground-truth guard below, for the downstream readers that still consume
    # them (see the census). The guarding path never reads the minted copies.
    # This logic belongs to the ALLOCATION axis and is unchanged in effect for
    # both runtimes — enroot and singularity alike launch on the node the
    # kernel reports.
    [[ -z "${SLURM_JOB_ID:-}" ]] || \
      fs_die "SLURM_JOB_ID is pre-set ('$SLURM_JOB_ID') but FS_ALLOCATION=local. Off-Slurm the launcher mints the id itself; a pre-set value is exactly the self-fulfilling guard shape this repository refuses. Unset it and re-run."
    [[ -z "${SLURM_JOB_NODELIST:-}" ]] || \
      fs_die "SLURM_JOB_NODELIST is pre-set ('$SLURM_JOB_NODELIST') off-Slurm — refusing to let a caller-written string stand in for an allocation. Unset it and re-run."
    # (B) Node identity from the KERNEL, once, at guard time. Nothing a caller
    # can export influences `hostname -s`; that is STRICTLY BETTER evidence
    # than the Slurm allocation ever provided, and it keeps the <other-team-node>
    # standing rule provably enforceable without Slurm.
    FS_ACTUAL_HOST=$(hostname -s) || fs_die "hostname -s failed — cannot prove node identity"
    # Same contract as the Slurm arm: FS_ALLOWED_NODE is required and has no
    # default; refuse (fail closed) rather than verify against nothing.
    [[ -n "${FS_ALLOWED_NODE:-}" ]] || \
      fs_die "FS_ALLOWED_NODE is unset/empty (required, no default by design). Export FS_ALLOWED_NODE=<compute-node> — the one node this estate may run on — and retry. Without it the node guard refuses; it never passes."
    # Denylist first — explicit denial beats any allow-prefix. UNQUOTED
    # ${FS_FORBIDDEN_NODES:-}: space-separated set; splitting IS the iteration.
    for fs_guard_denied in ${FS_FORBIDDEN_NODES:-}; do
      case "$FS_ACTUAL_HOST" in
        ${fs_guard_denied}*) fs_die "STANDING RULE VIOLATION (off-Slurm): hostname -s reports '$FS_ACTUAL_HOST', matching FS_FORBIDDEN_NODES entry '$fs_guard_denied' — the other team's node. This value came from the kernel at guard time, not from any environment variable." ;;
      esac
    done
    # ${FS_ALLOWED_NODE} deliberately UNQUOTED in the pattern: glob-prefix match
    # on the kernel-reported short hostname. Quoting would silently disarm the
    # guard; do not "fix" it.
    case "$FS_ACTUAL_HOST" in
      ${FS_ALLOWED_NODE}*) ;;
      *) fs_die "STANDING RULE VIOLATION (off-Slurm): hostname -s reports '$FS_ACTUAL_HOST'; only nodes matching FS_ALLOWED_NODE ('$FS_ALLOWED_NODE') are allowed; FS_FORBIDDEN_NODES ('${FS_FORBIDDEN_NODES:-none}') is refused first. This value came from the kernel at guard time, not from any environment variable." ;;
    esac
    # Minted only now. NUMERIC, because both launchers do MASTER_PORT
    # arithmetic of the form `29400 + SLURM_JOB_ID % 1000`. Unique per
    # invocation: epoch-seconds * 1000 + pid % 1000; the drain gate below
    # additionally serializes same-tray launches, so two minted ids never
    # compete for a MASTER_PORT on this tray. The id is echoed here and lands
    # in the run manifest via `--job-id "${SLURM_JOB_ID}"`, i.e. recorded.
    export SLURM_JOB_ID=$(( $(date +%s) * 1000 + ($$ % 1000) ))
    export SLURM_JOB_NODELIST=$FS_ACTUAL_HOST   # ground truth, not a claim
    export SLURM_NNODES=1
    export SLURM_NTASKS=4                        # 1 tray x 4 GB200; full-FT geometry divides by it
    export SLURM_NTASKS_PER_NODE=4
    [[ -n "$submit_dir" ]] && export SLURM_SUBMIT_DIR=$submit_dir
    # Single-tray rendezvous; resolvable inside the container no matter what
    # /etc/hosts holds. Launchers keep their scontrol-derived value under
    # sbatch and skip that derivation whenever this is already set.
    export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
    # s7 (see header). Exported host-side under BOTH runtimes: for enroot it
    # is the ONLY mechanism (enroot forwards nothing unless told, so the
    # allowlist carrier must move it); for singularity it is belt, with the
    # SINGULARITYENV_ + in-command duplication and the implicit-mount
    # containment living in the singularity launcher arm (R4) — a single
    # uncontrolled export is not a guarantee. The measured incident: one
    # image, two torch majors (host user-site 2.9.0+cu128 vs container
    # 2.11.0a0+nv26.02) selected purely by whose \$HOME is mounted.
    export PYTHONNOUSERSITE=1
    export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-bond0}     # s8b
    export GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-bond0}     # s8b
    unset NCCL_IB_HCA                                          # s8b: prefix-matches all 8 mlx5_*, measured breaking
    export NCCL_MNNVL_ENABLE=${NCCL_MNNVL_ENABLE:-0}           # s8c; denominator 4 trays stated in header
    FS_USE_TORCHRUN=1
    echo "backend: ${FS_CONTAINER_RUNTIME} runtime on local allocation $FS_ACTUAL_HOST; minted SLURM_JOB_ID=$SLURM_JOB_ID (flows to the run manifest via --job-id; log files land under the minted submit dir)"
  fi

  # Backwards-compatible alias for readers not yet on the two-axis vocabulary:
  # the old slurm backend == slurm ALLOCATION; the old enroot backend == local
  # ALLOCATION. It encodes nothing about FS_CONTAINER_RUNTIME and must not be
  # used to select a runtime arm — new code reads the two axes directly.
  if [[ "$FS_ALLOCATION" == slurm ]]; then FS_BACKEND=slurm; else FS_BACKEND=enroot; fi
  # Certify the provenance detector before anything relies on it. Pure bash
  # over a fixed table; the MUST_FIRE case (a host ~/.local path) must be
  # observed being REJECTED, so a detector stuck at always-accept cannot
  # pass this. A drill that cannot fail proves nothing.
  fs_selftest_torch_provenance >/dev/null || fs_die "fs_backend_init: torch-provenance self-test FAILED; the detector that keeps host torch out of the container cannot certify itself, so nothing downstream may trust it"

  export FS_BACKEND FS_CONTAINER_RUNTIME FS_ALLOCATION FS_USE_TORCHRUN
}

# ---------------------------------------------------------------------------
# fs_backend_runtime_setup <sqsh> <expected_gpus> [log_tee_path]
# Slurm arm: pyxis unpacks per-allocation, Slurm captures stdout, and
# --exclusive means the scheduler already guarantees a drained tray -> return.
# Enroot arm: tee the rest of the launcher into the file the post-run gates
# parse, then provenance-checked idempotent container unpack, then drain gate.
# ---------------------------------------------------------------------------
# ---- fs_backend_runtime_setup ----
fs_backend_runtime_setup() {
  local sqsh=$1 gpus=$2 log_tee_path=${3:-}
  [[ -n "$sqsh" ]] || fs_die "fs_backend_runtime_setup: no container image path given"
  # R1: the runtime is a caller decision, never an accident of $PATH. It is
  # validated in fs_backend_init; the case below still dies on anything else
  # rather than guessing — an unconfigured runtime guard is a disabled standing rule.
  [[ -n "${FS_CONTAINER_RUNTIME:-}" ]] || fs_die "fs_backend_runtime_setup: FS_CONTAINER_RUNTIME unset (enroot|singularity required, no default)"
  FS_CONTAINER_SQSH=$sqsh
  export FS_CONTAINER_SQSH
  [[ -n "$log_tee_path" ]] && fs_begin_log_tee "$log_tee_path"
  # R2: this switch keys on RUNTIME alone. Allocation (FS_ALLOCATION: slurm|local)
  # is an independent axis decided elsewhere; nothing in here may read
  # SLURM_JOB_ID — on this estate srun exists but pyxis does not, so "slurm
  # therefore pyxis" was the bug that had to die.
  case "$FS_CONTAINER_RUNTIME" in
    enroot)
      fs_enroot_ensure "$sqsh"
      fs_gpu_drain_wait "$gpus"
      echo "backend: container '$ENROOT_NAME' ready; training will run via torchrun --nproc_per_node=$gpus inside ONE enroot start"
      ;;
    singularity)
      # Fail closed: there is no enroot-style ensure step, so the image check is
      # the only gate before launch. An unreadable or missing .sif is not an
      # empty config to skip — it is a red (doctrine 4).
      [[ -f "$sqsh" && -r "$sqsh" ]] || fs_die "fs_backend_runtime_setup: image '$sqsh' missing or unreadable"
      fs_gpu_drain_wait "$gpus"
      # Containment (PYTHONNOUSERSITE=1, --no-home/--pwd via R4, torch-provenance
      # via R5) is applied per-exec by run_in_container — singularity binds $HOME
      # implicitly, so setup-time state cannot carry that guarantee and this
      # banner must not claim it does.
      echo "backend: singularity image '$sqsh' ready; training will run via torchrun --nproc_per_node=$gpus inside singularity exec"
      ;;
    *)
      fs_die "fs_backend_runtime_setup: unknown FS_CONTAINER_RUNTIME '$FS_CONTAINER_RUNTIME' (want enroot|singularity)"
      ;;
  esac
}

# Off-Slurm, Slurm's --output file capture does not exist — so we capture
# OURSELVES. Everything after this call (probes, training stdout/stderr, the
# post-run gates' own echoes) lands in the same file pyxis/sbatch would have
# written, at the same path the gates compute (SLURM_SUBMIT_DIR was minted
# precisely so the paths agree across backends).
fs_begin_log_tee() {
  local path=$1
  mkdir -p "$(dirname "$path")" || fs_die "cannot create log dir for $path"
  exec > >(tee -a "$path") 2>&1
  echo "backend: off-Slurm there is no #SBATCH --output capture; teeing stdout+stderr to $path (this is the file the post-run gates will parse)"
}

# ---------------------------------------------------------------------------
# fs_enroot_ensure <sqsh> — idempotent, provenance-checked container unpack.
#
# WHAT THIS CHECK CAN AND CANNOT DETECT (required honesty):
#   CAN: the pinned sqsh changed after the unpack (mtime or size drift — any
#        real edit of a 21 GiB file moves mtime); a record pointing at a
#        different path; a container with NO record (the g4export class, s3);
#        a malformed record (unreadable denominator -> BLOCK, doctrine 4).
#   CANNOT: an in-place edit preserving size AND mtime; a deliberately forged
#        record. A full sha256 of 21 GiB every launch is rejected on cost; the
#        record is written ONCE, by us, seconds after `enroot create` returns.
# We NEVER auto-remove a container we did not create: destructive cleanup on a
# shared tray is how g4export's provenance got lost in the first place. A
# mismatch message names the deliberate-remedy command instead.
# ---------------------------------------------------------------------------
fs_enroot_ensure() {
  local sqsh=$1
  [[ -f "$sqsh" ]] || fs_die "container image missing: $sqsh"
  ENROOT_NAME="fs-g4e4b-$(basename "$sqsh" .sqsh)"
  # s4: explicit store — the unset default is a DIFFERENT store on this tray.
  export ENROOT_DATA_PATH=${ENROOT_DATA_PATH:-$HOME/.enroot}
  mkdir -p "$ENROOT_DATA_PATH/.fs-provenance" || fs_die "cannot create $ENROOT_DATA_PATH/.fs-provenance"
  local rec="$ENROOT_DATA_PATH/.fs-provenance/$ENROOT_NAME.src"

  if enroot list 2>/dev/null | awk '{print $1}' | grep -qxF "$ENROOT_NAME"; then
    [[ -f "$rec" ]] || \
      fs_die "enroot container '$ENROOT_NAME' exists in ENROOT_DATA_PATH=$ENROOT_DATA_PATH, but there is NO provenance record at $rec. A name match is not evidence of origin — the 'g4export' container on this tray is exactly such an orphan (s3). Inspect it; if it genuinely came from $sqsh, write the record yourself from today's stat; otherwise remove it deliberately (ENROOT_DATA_PATH=$ENROOT_DATA_PATH enroot remove $ENROOT_NAME). Refusing to guess."
    local rp rs rm_ cs cm_
    rp=$(sed -n 's/^sqsh_path=//p' "$rec")
    rs=$(sed -n 's/^sqsh_size=//p' "$rec")
    rm_=$(sed -n 's/^sqsh_mtime=//p' "$rec")
    [[ -n "$rp" && -n "$rs" && -n "$rm_" ]] || \
      fs_die "provenance record $rec is malformed (need sqsh_path/sqsh_size/sqsh_mtime) — an unreadable denominator BLOCKS; repair or remove the container deliberately."
    cs=$(stat -c %s "$sqsh") || fs_die "cannot stat $sqsh"
    cm_=$(stat -c %Y "$sqsh") || fs_die "cannot stat $sqsh"
    [[ "$rp" == "$sqsh" ]] || \
      fs_die "provenance mismatch for '$ENROOT_NAME': record says built from '$rp', this launcher pins '$sqsh'. Hard error, not a warning (doctrine 4)."
    [[ "$rs" == "$cs" && "$rm_" == "$cm_" ]] || \
      fs_die "provenance mismatch for '$ENROOT_NAME': record has size=$rs mtime=$rm_, current $sqsh has size=$cs mtime=$cm_. The image CHANGED after the unpack — the unpack is stale. Remove deliberately and let it be rebuilt: ENROOT_DATA_PATH=$ENROOT_DATA_PATH enroot remove $ENROOT_NAME"
    echo "backend: enroot container '$ENROOT_NAME' present; provenance matches $sqsh (size=$cs mtime=$cm_)"
  else
    [[ ! -e "$rec" ]] || \
      fs_die "stale provenance record $rec exists but container '$ENROOT_NAME' does not. If you removed the container deliberately, remove the record too; otherwise investigation is owed before a 25 GB re-unpack."
    echo "backend: creating enroot container '$ENROOT_NAME' from $sqsh — one-time unpack (~25 GiB; s3: no squashfuse, so 'enroot start <img>.sqsh' was never an option)"
    enroot create --name "$ENROOT_NAME" "$sqsh" || \
      fs_die "enroot create failed for '$ENROOT_NAME' from $sqsh"
    {
      echo "sqsh_path=$sqsh"
      echo "sqsh_size=$(stat -c %s "$sqsh")"
      echo "sqsh_mtime=$(stat -c %Y "$sqsh")"
      echo "created_epoch=$(date +%s)"
      echo "created_by=fs_container_backend.sh"
    } > "$rec" || fs_die "could not write $rec — an unpack we cannot attest is an unpack we will not run"
    echo "backend: created '$ENROOT_NAME'; provenance recorded at $rec"
  fi
}

# ---------------------------------------------------------------------------
# fs_gpu_drain_wait <expected_gpus> — s8d. Bounded poll; timeout REFUSES.
# Zero GPUs reported is a BLOCK naming 0 as the number examined (doctrine 1),
# and a count that disagrees with the launcher's expected world is a BLOCK,
# not a silently smaller sweep (doctrine 2).
# ---------------------------------------------------------------------------
fs_gpu_drain_wait() {
  local expect_gpus=$1
  local max_mib=${FS_GPU_DRAIN_MAX_MIB:-2048}
  local timeout_s=${FS_GPU_DRAIN_TIMEOUT_S:-1800}
  command -v nvidia-smi >/dev/null 2>&1 || \
    fs_die "nvidia-smi missing — drain gate examined 0 of $expect_gpus GPUs; BLOCK (unreadable denominator)"
  local start now out n max_used
  start=$(date +%s)
  while :; do
    if ! out=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null); then
      fs_die "nvidia-smi query FAILED — the drain gate cannot read its denominator; BLOCK (doctrine 4)"
    fi
    n=$(printf '%s\n' "$out" | grep -cE '^[0-9]+[[:space:]]*$' || true)
    [[ "$n" -gt 0 ]] || fs_die "nvidia-smi reported ZERO GPUs — drain gate examined 0 of $expect_gpus; a sweep over zero units never returns a pass grade (doctrine 1)"
    [[ "$n" == "$expect_gpus" ]] || \
      fs_die "drain gate examined $n GPUs but the launcher's world expects $expect_gpus — tray topology mismatch; refusing to launch on an unverified configuration"
    max_used=$(printf '%s\n' "$out" | sort -n | tail -n 1)
    if (( max_used < max_mib )); then
      echo "drain gate: examined $n GPUs, max used ${max_used} MiB < ${max_mib} MiB — tray is drained (s8d)"
      return 0
    fi
    now=$(date +%s)
    if (( now - start >= timeout_s )); then
      fs_die "GPU-drain gate timed out after ${timeout_s}s: max used ${max_used} MiB across $n GPUs is still >= ${max_mib} MiB. A previous run is still dying; launching now segfaults the first new rank in c10::cuda::device_count() and hangs the rest (s8d). REFUSING to launch — investigate with: nvidia-smi --query-compute-apps=pid,used_memory,process_name --format=csv"
    fi
    echo "drain gate: examined $n GPUs, max used ${max_used} MiB >= ${max_mib} MiB; waiting (elapsed $(( now - start ))s of ${timeout_s}s)..."
    sleep 15
  done
}

# ---------------------------------------------------------------------------
# fs_launch_python <gpus> — echoes the string that replaces bare `python3` in
# front of run_recipe.py. Under sbatch the 4 ranks were supplied by
# --ntasks-per-node=4 (four python3 processes). Off-Slurm, ONE enroot start
# runs ONE torchrun, which forks the same 4 ranks. Entrypoint, recipe flags
# and CLI overrides downstream are IDENTICAL either way — one definition of
# the training command, two executors.
# ---------------------------------------------------------------------------
fs_launch_python() {
  local gpus=$1
  [[ "$gpus" =~ ^[0-9]+$ ]] || fs_die "fs_launch_python: gpu count '$gpus' is not numeric"
  if [[ "${FS_BACKEND:-slurm}" == enroot ]]; then
    printf 'torchrun --nproc_per_node=%s --nnodes=1 --node_rank=0 --master_addr=%s --master_port=%s' \
      "$gpus" "${MASTER_ADDR:-127.0.0.1}" "${MASTER_PORT:?MASTER_PORT must be resolved before the training command is built}"
  else
    printf 'python3'
  fi
}

# ===========================================================================
# fix32 — WHAT THE CONTAINER MUST NEVER INHERIT FROM THE HOST SHELL
#
# MEASURED on <compute-node>, 2026-08-24, container fs-g4e4b-nemo-automodel-26-04_compute
# (case 1 vs case 3 differ ONLY in whether the host PATH is forwarded):
#   case 1: host PATH forwarded (what this file did before fix32)
#           -> sys.executable=<CLUSTER_HOME>/anaconda3/bin/python3;
#              import torch FAILS ("No module named 'torch'"). A host anaconda
#              interpreter wearing the container's name.
#   case 3: host PATH withheld
#           -> /opt/venv/bin/python3; torch 2.11.0a0+eb65b36914.nv26.02;
#              transformers 5.5.0; megatron.bridge OK; E4B entry points 2/2.
# Case 2 (bash -c, non-login) fails identically to case 1, so the login shell
# is NOT the mechanism — the forwarded PATH is. Case 4 showed an absolute
# interpreter path only hides case 1 from the first process, not subprocesses.
#
# fs_env_forward_denylisted() is THE denylist (Task A), and it stays a
# DENYLIST — never an allowlist: pyxis got --export=ALL, the union of what the
# five converted call sites need is ALL (the NCCL_*/GLOO_*/SLURM_*/FS_*/
# MASTER_*/RIC_*/ENROOT_* families minted in fs_backend_init ride in through
# this same forwarding loop), so an allowlist would silently drop the next
# variable a launcher starts depending on and we would be back here.
#
# Membership is bounded by the measured env diff of 2026-08-24 (doctrine 5
# applies to defense too: convicting more than the evidence convicts would be
# a claim broader than its evidence) — EXTENDED by the fix37 hardware rerun
# of that measurement (fix37_shared.md), which collected on the caveat in
# Task B's refusal text ("a still-unknown variable with PATH-like effect")
# in under a week, over the same tray:
#   PROVEN dangerous:
#     PATH                    — case 1 vs case 3 above.
#     SSH_CLIENT, SSH2_CLIENT — PROVEN by the fix37 probes, and by an entirely
#       different mechanism than PATH. Neither name has PATH-like semantics;
#       SSH_CLIENT is a tuple of an IP and two ports. What it trips is a
#       COMPILE-TIME option in another program: Debian/Ubuntu bash is built
#       with SSH_SOURCE_BASHRC, under which a non-interactive `bash -c`
#       sources ~/.bashrc iff SSH_CLIENT or SSH2_CLIENT is present in the
#       environment (bash shell.c, run_startup_files(); the replay stream was
#       bisected down to this one variable, and --norc cleaned the launch).
#       $HOME is bind-mounted (s8a), so ~/.bashrc is the HOST's, whose conda
#       block prepends the host anaconda — the measured in-container PATH had
#       the conda prefix IN FRONT with /opt/venv still intact, the signature
#       that separates this leak from case 1. SSH2_CLIENT is not even set on
#       this host; it was found by reading bash's source, which is the
#       demonstration that membership cannot be enumerated by inspection.
#       GSETTINGS_SCHEMA_DIR_CONDA_BACKUP rides the forward stream today as
#       the same demonstration from the conda side — left FORWARDED by name,
#       because no measured effect convicts it and convicting it would be the
#       doctrine-5 defect in the other direction. That non-enumerability is
#       why fix37 layers 2 and 3 (see run_in_container and FS_CONTAINER_
#       WRAPPER) are the load-bearing arms; this list is hygiene.
#   CATEGORY exclusions (fix37; explicitly NOT measured — the marker is
#   CATEGORY, never PROVEN, per the fix37 packet's provenance rule):
#     SSH_CONNECTION SSH_TTY SSH_AUTH_SOCK SSH_ASKPASS
#     DBUS_SESSION_BUS_ADDRESS XDG_RUNTIME_DIR XDG_SESSION_ID
#     XDG_SESSION_CLASS XDG_SESSION_TYPE — host LOGIN-SESSION descriptors
#     (agent socket, tty, dbus address, runtime dir): category-wrong inside a
#     batch container regardless of tonight's bug. The sockets/paths they
#     name are not mounted here, so today they are inert litter; they are
#     convicted because any container consumer of them would be trusting
#     host-session residue, not because any was observed swapping an
#     interpreter.
#     BASH_ENV, ENV, ZDOTDIR — startup-file POINTERS, again CATEGORY with a
#       measured inert datum: fix37 probe 1 shows BASH_ENV=/etc/bash.bashrc
#       forwarded and CLEAN (a container-side path). The danger is
#       structural and specific: --norc (fix37 layer 2) does NOT disarm
#       BASH_ENV — a non-interactive bash expands it regardless of no_rc —
#       so a host-absolute BASH_ENV would execute host code in the container
#       shell before the wrapper's first line, past layers 1 and 2 alike.
#       ENV/ZDOTDIR are the sh- and zsh-side spellings of the same pointer.
#       Conviction has one honest side effect, stated: the payload's
#       non-interactive bash no longer sources the image's own
#       /etc/bash.bashrc via the forwarded pointer; the probe-1 rows were
#       equally clean without it (the file is interactive-shell furniture),
#       and the wrapper scrub (layer 3) removes it for descendants anyway.
#   Structurally dangerous, currently INERT — excluded anyway, and the comment
#   says "inert today" so no later reader mistakes the exclusion for a second
#   measured bug:
#     LD_LIBRARY_PATH         — unset on the host today, so it is NOT forwarded
#                               today; but the container's own value IS the
#                               CUDA/torch runtime linkage, and any future
#                               caller who exports it silently relinks the
#                               container against host libraries.
#     PYTHONHOME              — relocates the stdlib wholesale. Unset both sides.
#     PYTHONSTARTUP, LD_PRELOAD — classic injection vectors into a foreign
#                               interpreter/libc. Unset both sides today.
#     VIRTUAL_ENV             — unset today; any caller who sources a host venv
#                               exports it and points the container at it.
#     The conda block (CONDA_PREFIX CONDA_EXE CONDA_PYTHON_EXE
#     CONDA_DEFAULT_ENV CONDA_SHLVL CONDA_PROMPT_MODIFIER _CE_CONDA _CE_M) —
#     these ARE forwarded today and every value points at the host anaconda.
#     They are inert ONLY by accident: our payloads run under `bash -lc`, a
#     login NON-interactive shell, which sources /etc/profile and
#     ~/.bash_profile but NOT ~/.bashrc — and the conda hook lives in
#     ~/.bashrc. An image carrying /etc/profile.d/conda.sh, or a future switch
#     to `bash -ic`, re-arms all eight.
#   Kept from before fix32, load-bearing and unchanged: _ PWD OLDPWD SHLVL are
#   caller-shell residue, correct to drop then and now.
# Everything else defaults to FORWARD — including names nobody has invented
# yet (see the predicate's last arm). That is exactly why Task B exists: a
# denylist cannot be complete, so the tripwire below audits its effect.
# ---------------------------------------------------------------------------
# ---- fs_env_forward_denylisted ----
# SUBORDINATE second check since the singularity arm landed (R3, 2026-08-31):
# fs_env_forward_allowlisted decides what crosses the boundary — in BOTH arms,
# so the forwarding direction is a property of FoundationScale, not of
# whichever runtime is loaded. This function can only ever REMOVE a variable
# the allowlist already passed; it can no longer pass one. Kept because every
# entry below carries a measured firing (or a category argument) that the
# allowlist lacks — deleting them would discard controls, not redundancy.
fs_env_forward_denylisted() {
  case "$1" in
    _|PWD|OLDPWD|SHLVL) return 0 ;;   # pre-fix32 residue exclusions; unchanged
    PATH) return 0 ;;                 # PROVEN: case 1 vs case 3, <compute-node> 2026-08-24
    LD_LIBRARY_PATH|LD_PRELOAD|PYTHONHOME|PYTHONSTARTUP|VIRTUAL_ENV) return 0 ;;  # inert today; see block above
    CONDA_PREFIX|CONDA_EXE|CONDA_PYTHON_EXE|CONDA_DEFAULT_ENV|CONDA_SHLVL|CONDA_PROMPT_MODIFIER|_CE_CONDA|_CE_M) return 0 ;;  # inert-by-login-shell-accident today; see block above
    SSH_CLIENT|SSH2_CLIENT) return 0 ;;            # PROVEN: fix37 replay-stream bisect down to SSH_CLIENT + five-cell table (incl. the --norc row), <compute-node> (fix37_shared.md); NOT PATH-like — trips bash's SSH_SOURCE_BASHRC branch over the host-mounted ~/.bashrc
    SSH_CONNECTION|SSH_TTY|SSH_AUTH_SOCK|SSH_ASKPASS|DBUS_SESSION_BUS_ADDRESS|XDG_RUNTIME_DIR|XDG_SESSION_ID|XDG_SESSION_CLASS|XDG_SESSION_TYPE) return 0 ;;  # CATEGORY (fix37), NOT measured: host login-session descriptors are category-wrong in a batch container
    BASH_ENV|ENV|ZDOTDIR) return 0 ;;              # CATEGORY (fix37): startup-file pointers; BASH_ENV measured present-but-inert (fix37 probe 1) — and --norc does NOT disarm it, so this arm plus the FS_CONTAINER_WRAPPER scrub are the whole defence
    *) return 1 ;;                    # not blocked HERE — but the allowlist runs first and defaults to NOT-forward, so reaching this branch no longer means the variable crosses; it means only that this check found no measured reason to stop it
  esac
}

# ---------------------------------------------------------------------------
# fix32 Task B — FS_CONTAINER_WRAPPER: an ARM-AGNOSTIC in-container tripwire
# that the interpreter the container resolves is the container's OWN, plus an
# exec shim for the payload. run_in_container prepends it on BOTH arms, so it
# is cheap enough to run on every launch (one command -v, no imports).
#
# WHY IT EXISTS: Task A is a denylist, and a denylist is unbounded by
# construction — the next colliding host variable is a silent repeat of this
# exact bug. This tripwire is what makes the denylist auditable instead of a
# hope: it measures the EFFECT (which interpreter resolution actually produced)
# rather than trusting the mechanism (which variables we remembered to block).
#
# WHAT IT CHECKS, inside the container, before the payload starts:
#   1. roots: run_in_container hands it the destination side of EVERY host
#      bind-mount as $1..$N before "--" ($HOME on both arms; plus /dev and
#      /sys on the enroot arm). Zero declared roots = a detector that examined
#      0 units = refusal naming 0 (doctrine 1), never a pass.
#   2. PATH must be set and non-empty — an unreadable denominator refuses
#      (doctrine 4).
#   3. `command -v python3` must resolve to an ABSOLUTE path — no resolution,
#      or a non-absolute one, is an ambiguous answer and refuses (doctrine 4).
#   4. The resolved path must NOT sit under any declared host root. A python3
#      under a host mount IS the host's interpreter regardless of which
#      variable put it there — this catches the unbounded denylist tail, not
#      just PATH.
# Failure messages name the resolved interpreter AND the in-container PATH so
# the operator diagnoses from the refusal alone. On success it prints the
# resolution and the NUMBER of roots it checked (every claim carries a
# denominator), then exec's the payload with argv untouched.
#
# DENOMINATOR, STATED HONESTLY (doctrine 5, both directions): the interpreter-
# swap failure mode is MEASURED on the enroot arm only (case 1 vs case 3,
# <compute-node>, 2026-08-24). The slurm/pyxis arm runs this same wrapper BY
# CONSTRUCTION — `srun --export=ALL` plausibly carries the same hazard — but
# whether pyxis ever lets the exported host PATH reach the container is
# UNMEASURED. Nothing here claims the slurm arm was broken, and nothing here
# claims it is "fixed"; the honest claim is: IF a host PATH reaches the
# container and changes the resolution, this refuses, on either arm.
#
# WHAT IT DOES NOT SEE (stated abstentions): PATH a payload mutates for itself
# after exec (measured irrelevant — cases 1 vs 2); the symlink TARGET of the
# resolved interpreter (the resolution path is what executes; target-chasing
# is unmeasured); torchrun explicitly (it lives beside the container python3
# in /opt/venv/bin, so the same resolution implies it — implied, not asserted).
#
# CONTROLS (doctrine 3): MUST_PASS = every clean launch. MUST_FIRE now ships
# TWO drills, one per measured mechanism, with DELIBERATELY different banner
# lines and different refusal signatures — two drills that print the same
# line are one drill, and a drill that rehearses last war's attack certifies
# coverage it does not have (the fix37 control problem):
#   FS_REARM_HOST_PATH_FORWARD=1    — fix32's drill, unchanged: re-forwards
#     exactly the host PATH through the enroot-arm denylist, reproducing
#     case 1 verbatim; refusal signature is the PURE host PATH (no /opt/venv
#     anywhere). The tripwire MUST refuse rc 95.
#   FS_REARM_SSH_SESSION_FORWARD=1  — the fix37 drill: re-arms BOTH conjuncts
#     of the SSH_SOURCE_BASHRC branch, re-admitting SSH_CLIENT/SSH2_CLIENT
#     through the denylist AND invoking this wrapper WITHOUT --norc (the
#     banner names both re-arms and its denominator: N of the 2 measured
#     trigger names present in this host env — on an ssh-less host N=0 is an
#     honest half-drill and says so). Refusal signature on hardware: the
#     host conda prefix PREPENDED to an intact image PATH — the LEG1 shape,
#     distinguishable from the PATH drill even if both banners are missed.
#     SANDBOX HONESTY (doctrine 5): the sandbox has no conda-bearing host
#     ~/.bashrc and no guarantee of a Debian-patched bash, so the sandbox
#     tests pin the drill's DECISIONS (names re-admitted? --norc dropped?
#     banner distinct?) and return 0 by construction; the EFFECT leg — the
#     tripwire refusing rc 95 over a genuinely contaminated resolution — is
#     owed to the <compute-node> rerun, one drill invocation, one refusal.
#   Either knob in production is the same offence stated once: a launch that
#   PROCEEDS with a drill armed is proof the tripwire is dead.
# Both drills are exercised at the decision level in
# tests/test_fix32_container_env_passthrough.py. Exit codes 95-98 remain
# this wrapper's own, never fs_die(1)/gate(90) — fix37 ADDS NO CODES: the
# layer-3 scrub it adds cannot detect, and a control that cannot detect must
# not mint a refusal.
# ---------------------------------------------------------------------------
FS_CONTAINER_WRAPPER='# fix32-container-tripwire-v1 — the shipped text; the test suite executes THIS string via stubs, tying every claim to what actually runs.
set -u
fs_tw_roots=()
while [ "$#" -gt 0 ]; do
  if [ "$1" = "--" ]; then shift; break; fi
  fs_tw_roots+=("$1"); shift
done
if [ "${#fs_tw_roots[@]}" -eq 0 ]; then
  printf "FATAL: fix32 tripwire: 0 host-visible roots were declared — a detector that examined 0 units must never report healthy (doctrine 1). Miswired invocation; refusing.\n" >&2
  exit 98
fi
if [ "$#" -eq 0 ]; then
  printf "FATAL: fix32 tripwire: no payload argv after -- (%d root(s) declared). Miswired invocation; refusing.\n" "${#fs_tw_roots[@]}" >&2
  exit 97
fi
if [ -z "${PATH:-}" ]; then
  printf "FATAL: fix32 tripwire: PATH is unset or empty INSIDE the container — the denominator this check reads is absent, and an unreadable answer is a refusal, never a pass (doctrine 4).\n" >&2
  exit 96
fi
fs_tw_resolved=$(command -v python3) || fs_tw_resolved=""
case "$fs_tw_resolved" in
  /*) ;;
  *)
    printf "FATAL: fix32 tripwire: python3 did not resolve to an absolute path inside the container (command -v printed: \"%s\"). In-container PATH: %s\nAn interpreter that cannot be identified cannot be vouched for; refusing (doctrine 4).\n" "$fs_tw_resolved" "$PATH" >&2
    exit 96 ;;
esac
for fs_tw_root in "${fs_tw_roots[@]}"; do
  case "$fs_tw_resolved" in
    "$fs_tw_root"|"$fs_tw_root"/*)
      printf "FATAL: fix32 tripwire: the container resolved its interpreter from HOST territory.\n  resolved python3 : %s\n  host mount root  : %s (1 of %d declared host-visible roots)\n  in-container PATH: %s\nA host PATH (or a still-unknown variable with PATH-like effect) reached the container and shadowed the image interpreter — the fix32 case 1 shape (<compute-node>, 2026-08-24). Refusing to run the payload under a host python; diff the --env stream against fs_env_forward_denylisted.\n" "$fs_tw_resolved" "$fs_tw_root" "${#fs_tw_roots[@]}" "$PATH" >&2
      exit 95 ;;
  esac
done
# fix37 layer 3 — the positional scrub, placed HERE (after every check, on
# the threshold of the exec) so it reads as what it is: the last environment
# surgery before the payload runs. Layers 1 and 2 leave one live path that
# the fix37 rerun measured reachable: the payload argv is usually a SECOND
# shell (`bash -lc "cd $REPO && python3 ..."`) that this script execs, and
# --norc on the first shell does not reach it. That shell takes the sshd-rc
# branch — and, the day a ~/.profile skeleton exists, the login arm reaches
# the same ~/.bashrc — iff SSH_CLIENT or SSH2_CLIENT is in its environment:
# bash shell.c, run_startup_files(), reads exactly those two names, so
# losing exactly those two names disarms the branch for every descendant at
# any depth. BASH_ENV rides along because --norc does NOT disarm it (a
# non-interactive bash expands BASH_ENV regardless of no_rc); ENV and
# ZDOTDIR are the sh- and zsh-side spellings of the same startup-file
# pointer. The layering is the fix37 lesson applied: the space of trigger
# names cannot be enumerated at the launcher, so the defence that must hold
# lives at the point of use. unset of an already-absent name is silent even
# under set -u — correct: on a clean launch all five names were already
# convicted at the launcher and this line is a no-op with a job.
unset SSH_CLIENT SSH2_CLIENT BASH_ENV ENV ZDOTDIR
printf "fix32 tripwire: python3 resolves to %s — outside all %d host-visible root(s) [%s]; executing payload.\n" "$fs_tw_resolved" "${#fs_tw_roots[@]}" "${fs_tw_roots[*]}"
exec "$@"
'

# ---------------------------------------------------------------------------
# run_in_container [--slurm-ntasks N] [--workdir DIR] [--] CMD [ARG...]
# THE single executor. srun sites replaced by calls here (denominator 5):
#   LoRA 3  (preflight module dump, preflight env probe, training)
#   full-FT 2 (tokenizer/CoT probe, training)
#   --slurm-ntasks : srun-only; CPU probes pass 1 instead of inheriting the
#                    allocation's 4 tasks (their output is a single text file).
#                    The enroot arm always runs a payload exactly once.
#   --workdir      : forwarded to pyxis. enroot start has no workdir flag, so
#                    payloads that need a specific cwd cd themselves — every
#                    converted payload that needs one already does (cd $REPO).
# ---------------------------------------------------------------------------
run_in_container() {
  local ntasks="" workdir=""
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --slurm-ntasks) [[ $# -ge 2 ]] || fs_die "run_in_container: --slurm-ntasks needs a value"; ntasks=$2; shift 2 ;;
      --workdir)      [[ $# -ge 2 ]] || fs_die "run_in_container: --workdir needs a value";      workdir=$2; shift 2 ;;
      --) shift; break ;;
      *) break ;;
    esac
  done
  [[ $# -gt 0 ]] || fs_die "run_in_container called with no payload"

  # fix43 (R1+R2): the two axes the old code conflated. ALLOCATION (does an
  # srun step launch this, or the local shell) and RUNTIME (enroot or
  # singularity) are INDEPENDENT, and this estate proves the conflation was a
  # bug and not a shorthand: srun exists here and pyxis does NOT, so the old
  # "FS_BACKEND==slurm therefore srun --container-image" arm was a rule whose
  # trigger (SLURM_JOB_ID) measured nothing about the thing it selected.
  # srun existing is not pyxis existing. Both axes are caller-set with NO
  # default — the FS_ALLOWED_NODE precedent: an unconfigured guard is a
  # disabled standing rule, and auto-detecting either off $PATH would make
  # the runtime an accident of whichever node compiled the path. fs_backend_init
  # is the validator of record; the cases below exist so an unset or foreign
  # value REFUSES here too instead of falling silently into an else-arm
  # (doctrine 4: fail closed, at the boundary where the arm is chosen).
  case "${FS_CONTAINER_RUNTIME:-}" in
    enroot|singularity) ;;
    *) fs_die "run_in_container: FS_CONTAINER_RUNTIME is '${FS_CONTAINER_RUNTIME:-<unset>}' — REQUIRED (R1), exactly 'enroot' or 'singularity', no default and no auto-detection: the runtime must be a property of the caller's intent, never of \$PATH." ;;
  esac
  case "${FS_ALLOCATION:-}" in
    slurm|local) ;;
    *) fs_die "run_in_container: FS_ALLOCATION is '${FS_ALLOCATION:-<unset>}' — REQUIRED (R2), exactly 'slurm' or 'local', and never inferred from SLURM_JOB_ID: that variable says a job is running, nothing about which container runtime the node has." ;;
  esac

  # fix32: the ENROOT arm bind-mounts $HOME:$HOME (s8a is why the mount
  # exists at all — HOME inside must be the real home — and parameterizing
  # HOME lets the contract suite drive THIS code through a sandbox HOME, the
  # same stub idiom as its hostname/enroot/nvidia-smi binaries and its stat
  # adapter). The SINGULARITY arm deliberately does NOT: singularity binds
  # $HOME, /tmp and $PWD implicitly, with no entry in any mounts array, and
  # the measured incident lives exactly under that implicit $HOME bind — the
  # SAME image importing torch 2.9.0+cu128 from the host user-site
  # (/home/<uid>/.local/lib/python3.12/site-packages, 9.4 GB) versus torch
  # 2.11.0a0+nv26.02 from /usr/local/lib/python3.12/dist-packages depending
  # only on whose HOME was mounted. So $HOME must still be set and absolute,
  # for two distinct reasons now: the enroot arm's mounts are $HOME-relative
  # (an unset HOME would mint a host root matching nothing — a vacuous
  # tripwire), and the singularity arm's host-root census asks whether $PWD
  # sits under $HOME to know whether --no-home has suppressed the implicit
  # CWD bind. A census derived from "is $PWD inside $HOME" when $HOME is
  # unset or relative answers vacuously; that is a refusal, never a default.
  [[ -n "${HOME:-}" && "$HOME" == /* ]] || \
    fs_die "run_in_container: \$HOME is unset or relative ('${HOME:-<unset>}') — the enroot arm's mounts are \$HOME-relative (s8a) and the singularity arm's --no-home census compares \$PWD against \$HOME to decide whether the implicit CWD bind is suppressed; refusing rather than declaring a host root that matches nothing (doctrine 4)."

  # fix37 layer 2 — the tripwire's OWN shell. The fix37 hardware rerun showed
  # a clean launch REFUSED at rc 95 with the host conda prefix prepended to
  # an INTACT image PATH — not a forwarded PATH (the same night's PATH drill
  # shows the difference: the pure host PATH, no /opt/venv), but the wrapper
  # bash itself taking the sshd-rc branch before its first command:
  # SSH_SOURCE_BASHRC makes a non-interactive `bash -c` source ~/.bashrc iff
  # SSH_CLIENT or SSH2_CLIENT is present, and ~/.bashrc is the host's (s8a).
  # That branch is five CONJUNCTS (interactive_shell==0, no_rc==0,
  # login_shell==0, act_like_sh==0, command_execution_string) and no_rc is
  # the one this file controls: --norc. Measured in the five-cell table:
  # SSH_CLIENT set + `bash --norc -c` resolves the IMAGE interpreter; the
  # same probe under plain `bash -c` resolves the host anaconda. BOTH
  # runtimes get it: the wrapper's bash is launched fresh by enroot start /
  # singularity exec either way, and on a slurm allocation the step's default
  # environment propagation can carry the session family to the launcher —
  # the allowlist below is the gate for what enters the CONTAINER, but the
  # vulnerable conjuncts are read by the wrapper's own bash, where --norc is
  # the only barrier in front of the measurement. It protects only the
  # WRAPPER's shell; the payload's own shells are layer 3's job, and
  # BASH_ENV is explicitly NOT covered by --norc (that is why the wrapper
  # scrub removes it). ${norc:+$norc} expands unquoted below so the option
  # VANISHES when the SSH drill re-arms the vulnerable invocation — an
  # empty ARRAY would read unbound under set -u on older bash; the
  # parameter-expansion form is the spelling that cannot misfire.
  local norc=--norc
  if [[ "${FS_REARM_SSH_SESSION_FORWARD:-0}" == "1" ]]; then
    local fs_drill_ssh_n=0
    [[ -z "${SSH_CLIENT:-}" ]]  || fs_drill_ssh_n=$((fs_drill_ssh_n + 1))
    [[ -z "${SSH2_CLIENT:-}" ]] || fs_drill_ssh_n=$((fs_drill_ssh_n + 1))
    norc=
    echo "DRILL: FS_REARM_SSH_SESSION_FORWARD=1 — re-arming the fix37 SSH_SOURCE_BASHRC leak on purpose: invoking the in-container interpreter tripwire WITHOUT --norc below, and re-admitting the measured trigger names where this host exports them ($fs_drill_ssh_n of the 2 measured trigger names, SSH_CLIENT and SSH2_CLIENT; past the allowlist-and-denylist pair that is identical in both runtime arms, and onward to the step through srun's default environment propagation on a slurm allocation). This banner is deliberately different from the FS_REARM_HOST_PATH_FORWARD drill, and the refusal signatures differ too: THIS leak shows the host conda prefix prepended to an otherwise intact image PATH (/opt/venv still present); the PATH drill shows the pure host PATH. The tripwire MUST refuse what follows (rc 95 on hardware); a launch that PROCEEDS is proof the tripwire is dead. A count of 0 means an ssh-less host — an honest half-drill rehearsing only the --norc conjunct; run from an ssh session for the full rehearsal." >&2
  fi

  # fix43 (R3): ONE forwarding decision, built ONCE, enforced IDENTICALLY in
  # both runtime arms — because the measured root cause is a direction flip,
  # not a missing entry. enroot forwards NOTHING unless told, so a denylist
  # was a comfortable shape there; singularity forwards the HOST ENVIRONMENT
  # unless told not to, where a denylist must enumerate every hostile
  # variable and the one that bit us (the user-site consulted because
  # PYTHONNOUSERSITE rode in unset) is never the last. The allowlist decides;
  # the denylist stays as a second, subordinate gate — belt and braces, in
  # that order, so neither a missing allowlist entry nor a missing denylist
  # entry alone can leak. The two drill branches deliberately sit AHEAD of
  # both gates: a MUST_FIRE control that cannot force its own stimulus
  # cannot prove the detector is alive (doctrine 3). argv form
  # (enroot --env / the singularity arm's env-prefix) keeps values with
  # spaces or newlines intact. PYTHONNOUSERSITE rides in as a real 1 because
  # fs_backend_init SET it (s7); PYTHONPATH (EXTRAS-first, README trap 1),
  # CUDA_VISIBLE_DEVICES, HOME, USER, LANG/LC_CTYPE and every variable
  # fs_backend_init mints keep flowing — the enroot-arm keep-set is pinned in
  # tests/test_fix32_container_env_passthrough.py, and R3 makes dropping one
  # silently a worse failure than the leak this split exists to stop.
  local -a forward_env=()
  local v
  while IFS= read -r v; do
    # fix32 MUST_FIRE drill: the one sanctioned way through BOTH gates.
    # FS_REARM_HOST_PATH_FORWARD=1 re-forwards exactly PATH, re-creating the
    # measured case 1 on demand so the in-container interpreter tripwire can
    # be PROVEN able to fire (doctrine 3); the control lives in
    # tests/test_fix32_container_env_passthrough.py and is how a live
    # tripwire is distinguished from a dead one. Never set in production:
    # with it set, a launch that PROCEEDS is proof the tripwire is dead.
    if [[ "$v" == PATH && "${FS_REARM_HOST_PATH_FORWARD:-0}" == "1" ]]; then
      echo "DRILL: FS_REARM_HOST_PATH_FORWARD=1 — re-forwarding the host PATH on purpose to re-create fix32 case 1; the in-container interpreter tripwire MUST refuse what follows." >&2
    # fix37 MUST_FIRE drill, the second measured mechanism. The branch needs
    # BOTH conjuncts rehearsed to be an honest rehearsal, so this drill is
    # split across two sites on purpose: the --norc bypass and the banner
    # (with the drill's denominator) live above this loop; this site only
    # re-admits the two names the branch reads — measured triggers, not a
    # category. The per-name echo keeps the re-admission audible name by
    # name: SSH2_CLIENT was measured DIRTY while being unset on this host,
    # so which of the pair actually rode in tonight is part of the drill's
    # record, never to be inferred.
    elif [[ ( "$v" == SSH_CLIENT || "$v" == SSH2_CLIENT ) && "${FS_REARM_SSH_SESSION_FORWARD:-0}" == "1" ]]; then
      echo "DRILL: FS_REARM_SSH_SESSION_FORWARD=1 — re-admitting $v past the allowlist-and-denylist pair on purpose (the --norc bypass banner above is the drill's other half); the in-container interpreter tripwire MUST refuse what follows." >&2
    else
      # The allowlist decides (R3)...
      fs_env_forward_allowlisted "$v" || continue
      # ...and the denylist may still veto an allowlisted name. Subordinate,
      # never reversed: a name must pass BOTH gates to be forwarded.
      if fs_env_forward_denylisted "$v"; then
        continue
      fi
    fi
    forward_env+=("$v=${!v}")
  done < <(compgen -e)

  # One command line is assembled per runtime, then dispatched per
  # allocation — the independence is structural: nothing about the launcher
  # text depends on who launches it, and nothing about the srun prefix knows
  # which runtime follows it.
  local -a cmd=()
  if [[ "$FS_CONTAINER_RUNTIME" == enroot ]]; then
    # s8a: enroot auto-mounts NOTHING, home least of all — so $HOME is
    # mounted explicitly, and the tripwire's host-root list is DERIVED from
    # those mounts (dst of src:dst; estate paths hold no colons, so ${m#*:}
    # is exact, and keeping one list makes drift between mount and check
    # impossible by construction). On this arm the explicit mounts ARE the
    # complete census: enroot has no implicit bind set for a census to miss.
    [[ -n "${ENROOT_NAME:-}" ]] || fs_die "run_in_container: ENROOT_NAME is unset — the enroot arm refuses to guess a container name (doctrine 4)."
    local -a args=(--rw)
    local -a mounts=("$HOME:$HOME" /dev:/dev /sys:/sys)
    local -a host_roots=()
    local m kv
    for m in "${mounts[@]}"; do
      args+=(--mount "$m")
      host_roots+=("${m#*:}")
    done
    for kv in ${forward_env[@]+"${forward_env[@]}"}; do
      args+=(--env "$kv")
    done
    cmd=(enroot start "${args[@]}" "$ENROOT_NAME" \
      bash ${norc:+$norc} -c "$FS_CONTAINER_WRAPPER" fs-container-tripwire "${host_roots[@]}" -- "$@")
  else
    # The singularity arm. The forwarding direction here is HOST-TO-CONTAINER
    # BY DEFAULT, so containment is built from force, not from enumeration:
    #
    # 1. SINGULARITYENV_PYTHONNOUSERSITE=1 is exported now (R4's
    #    SINGULARITYENV_ leg). Its delivery depends on the launcher's
    #    environment propagation — srun's default — which is exactly why it
    #    is the belt, not the trousers.
    # 2. PYTHONNOUSERSITE=1 is ALSO assigned inside the exec'd command (the
    #    env-prefix below), a leg no launcher can launder, and it is placed
    #    AFTER the allowlisted set so a stray forwarded copy cannot outrank
    #    it. MEASURED 2026-08-31 (job 37234): PYTHONNOUSERSITE=1 ALONE is
    #    sufficient AND is the general containment — it defeats a user-site
    #    wherever it lives, including a PYTHONUSERBASE redirect outside
    #    $HOME; the MUST_FIRE leg was observed RED against a manufactured
    #    hostile user-site while MUST_PASS stayed GREEN with that site still
    #    mounted and reachable.
    # 3. --no-home plus an explicit --pwd contain the implicit BINDS. This
    #    is defence in depth ONLY: it addresses the $HOME instance of the
    #    leak, and the measurement above is why nothing here claims --no-home
    #    is required. --pwd is always explicit (workdir when the caller gave
    #    one, / otherwise — matching the enroot arm, which has no workdir
    #    switch and starts at /): singularity's default CWD handling is one
    #    more implicit behaviour this arm declines to inherit. A --pwd that
    #    does not exist inside the container (e.g. a workdir under the
    #    --no-home'd $HOME) makes singularity itself refuse — fail closed is
    #    the desired answer there, not a silent re-mount of host HOME.
    [[ -n "${FS_CONTAINER_SQSH:-}" ]] || fs_die "run_in_container: FS_CONTAINER_SQSH is unset — the image path comes from the caller (R7); the singularity arm refuses to invent one."
    export SINGULARITYENV_PYTHONNOUSERSITE=1
    local spwd=/
    [[ -n "$workdir" ]] && spwd=$workdir
    # The host-root census on THIS arm cannot be derived from an explicit
    # mounts list — there isn't one. Ground truth: singularity binds $HOME,
    # /tmp, /dev, /sys, /proc and $PWD implicitly, and a census built from a
    # mounts array is incomplete in exactly the arm where it matters, since
    # the host user-site lives under the $HOME nobody asked for. --no-home
    # suppresses the $HOME member; the rest of the implicit set is enumerated
    # here, destination-side, /proc included — a census that omits a real
    # bind would hand the tripwire a blind spot baked in at construction.
    local -a host_roots=(/tmp /dev /sys /proc)
    case "$PWD" in
      "$HOME"|"$HOME"/*) ;;
      /*) host_roots+=("$PWD") ;;
      *) fs_die "run_in_container: \$PWD is not absolute ('$PWD') — the singularity arm's host-root census cannot classify it against the implicit CWD bind; refusing rather than declaring a census that may be short one host root (doctrine 4)." ;;
    esac
    local -a sargs=(exec --no-home --pwd "$spwd")
    # R5 — torch provenance, a check DISTINCT from interpreter resolution and
    # deliberately so: the incident is same `python` binary, different
    # sys.path, so a check on WHICH binary runs cannot fire here and is not
    # credited with covering it. The probe imports torch INSIDE the
    # container, under the same containment legs and the same allowlisted
    # environment as the payload launch (and through the same srun dispatch
    # on a slurm allocation, so the measurement is taken where the claim will
    # run), then hands the resolved __file__ to fs_assert_torch_provenance.
    # Import failure, empty output, and an assert refusal are ALL refusals —
    # doctrine 4, unreadable is not "fine". The cost is one extra torch
    # import per launch; doctrine 5 is symmetric, and a slow measured GREEN
    # costs less than a fast unmeasured one.
    local -a probe=(singularity "${sargs[@]}" "$FS_CONTAINER_SQSH" \
      env ${forward_env[@]+"${forward_env[@]}"} PYTHONNOUSERSITE=1 \
      python3 -c 'import torch; print(torch.__file__)')
    local torch_file=""
    if [[ "$FS_ALLOCATION" == slurm ]]; then
      torch_file=$(srun "${probe[@]}" | tail -n 1) || \
        fs_die "run_in_container: the singularity-arm torch probe FAILED to run (rc nonzero through srun+singularity) — torch provenance is UNMEASURED, and unmeasured is never PASS (doctrine 1); refusing the launch (R5, fail closed)."
    else
      torch_file=$("${probe[@]}" | tail -n 1) || \
        fs_die "run_in_container: the singularity-arm torch probe FAILED to run — torch provenance is UNMEASURED, and unmeasured is never PASS (doctrine 1); refusing the launch (R5, fail closed)."
    fi
    [[ -n "$torch_file" ]] || \
      fs_die "run_in_container: the singularity-arm torch probe produced NO path — torch cannot be imported or its __file__ cannot be read; refuse (R5): unreadable is not empty, missing is not zero, and neither is 'fine' (doctrine 4)."
    fs_assert_torch_provenance "$torch_file" || \
      fs_die "run_in_container: torch resolved to '$torch_file', OUTSIDE the container prefix (expected /usr/local/lib/python3.*/dist-packages; the leaked prefix /home/*/.local/lib/python3.*/site-packages is refused). One image, two torch majors — the leak the containment legs exist to stop made it past them; refusing the launch (R5)."
    cmd=(singularity "${sargs[@]}" "$FS_CONTAINER_SQSH" \
      env ${forward_env[@]+"${forward_env[@]}"} PYTHONNOUSERSITE=1 \
      bash ${norc:+$norc} -c "$FS_CONTAINER_WRAPPER" fs-container-tripwire "${host_roots[@]}" -- "$@")
  fi

  # ALLOCATION, and nothing else, is decided here. No --container-image, no
  # --container-mounts, no --export=ALL: those are pyxis flags, pyxis does
  # not exist on this estate, and `srun --container-image` measured as a
  # failure, not as an arm. ntasks applies only where --ntasks means
  # something; on a local allocation it is inert, as it always was on the
  # non-slurm arm.
  if [[ "$FS_ALLOCATION" == slurm ]]; then
    srun ${ntasks:+--ntasks="$ntasks"} "${cmd[@]}"
  else
    "${cmd[@]}"
  fi
}

# ---------------------------------------------------------------------------
# fs_hard_stop_training <pid> — used by the full-FT live tripwires. The
# escalation (TERM, grace, KILL) is byte-equivalent to the launcher-era lines;
# the enroot arm additionally force-removes exactly the container THIS job
# recorded as its own (RIC_ACTIVE_CONTAINER), because a TERM'd launcher-side
# wrapper does not reliably propagate into an enroot payload, and a tripwire
# must never leave an orphaned rank holding the tray's GPUs. Force-removal
# frees the ~25 GiB unpack; the next launch re-creates it. Our own provenance
# record is removed with it — it attested a container lifetime that just ended
# (keeping it would trip the stale-record refusal in fs_enroot_ensure).
# ---------------------------------------------------------------------------
fs_hard_stop_training() {
  local pid=$1
  kill -TERM "$pid" 2>/dev/null
  sleep 20
  kill -KILL "$pid" 2>/dev/null || true
  if [[ "${FS_BACKEND:-}" == enroot && -n "${RIC_ACTIVE_CONTAINER:-}" ]]; then
    echo "TRIPWIRE: force-removing enroot container '$RIC_ACTIVE_CONTAINER' so no orphaned rank survives" >&2
    enroot remove --force "$RIC_ACTIVE_CONTAINER" >/dev/null 2>&1 || true
    rm -f "${ENROOT_DATA_PATH:-$HOME/.enroot}/.fs-provenance/$RIC_ACTIVE_CONTAINER.src" || true
    RIC_ACTIVE_CONTAINER=""
  fi
}
