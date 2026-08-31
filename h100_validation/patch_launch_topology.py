#!/usr/bin/env python3
"""
Patch for defect #124: the measured GPU count was CHECKED but never became
LOAD-BEARING.

WHY
---
launch_fs_h100.fixed.sh proves `visible CUDA devices == FS_GPUS_PER_NODE`
(8), then calls run_in_container WITHOUT --slurm-ntasks under an sbatch
header of --ntasks-per-node=1. The backend's slurm arm is
`srun ${ntasks:+--ntasks="$ntasks"} "${cmd[@]}"`, so exactly ONE process
starts on 8 measured GPUs while the log line prints gpus=8 and the launcher
exits 0. The function whose entire job was to couple gpus -> launch command,
fs_launch_python(), has ZERO call sites, and its non-enroot branch returned
bare `python3` -- correct only under a foreign sbatch that supplied the ranks
via --ntasks-per-node=4. Branching on FS_BACKEND==enroot conflated two
orthogonal axes: WHO FORKS THE RANKS vs WHICH CONTAINER RUNTIME IS IN USE
the same conflation as prior findings #109/#117/#122.

MEASURED (from the generated artifacts, not hypothesised)
---------------------------------------------------------
  * visible CUDA device count == FS_GPUS_PER_NODE (8) is proven, then never
    becomes load-bearing: no consumer of the measurement exists.
  * sbatch header requests --ntasks-per-node=1; run_in_container is invoked
    with no --slurm-ntasks; the srun arm adds no --ntasks; one process runs.
  * fs_launch_python(): 0 call sites.

WHAT THIS PATCH DOES
--------------------
  * DELETES fs_launch_python and installs ONE runtime-agnostic composer,
    fs_compose_launch <mode> <gpus> <engine_cmd>, keyed on the REQUIRED,
    NO-DEFAULT FS_ENGINE_LAUNCH_MODE (torchrun|wlm|self) -- joining the
    existing required-no-default guard family (an unconfigured guard is a
    disabled standing rule).
  * Makes the launcher CALL the composer, act on its refusal (fail 124),
    pass --slurm-ntasks <gpus> to run_in_container in wlm mode, and stamp a
    LAUNCH_TOPOLOGY provenance line with the world-size source.
  * Leaves FS_ENV_ALLOWLIST UNCHANGED, and gates that it stays unchanged.
    FS_ENGINE_LAUNCH_MODE and FS_ENGINE_PROCS_PER_NODE are host-side
    control-plane inputs, read by fs_compose_launch before any container
    starts -- the same family as FS_CONTAINER_RUNTIME / FS_ALLOCATION /
    FS_BACKEND. The first draft put them on the allowlist and gate_env_drift
    went red (D3: allowlisted with no producer); the gate was right, and the
    draft's own comment conceded nothing in-container reads them.

SCOPE LIMITS
------------
This patch proves COUPLING ONLY: after it, the measured device count is
either composed into the command line (torchrun), asserted against the
workload manager's per-node task count (wlm), or declared by the engine and
stamped engine-declared (self). It does NOT prove that srun honoured the
count at run time, that torchrun rendezvoused, that a self-forking engine
actually forks FS_ENGINE_PROCS_PER_NODE ranks, or that the devices are
healthy. Runtime behaviour is out of scope for a static patch; that is what
the LAUNCH_TOPOLOGY provenance line is for -- a downstream report can audit
it, but this script does not run a training job to prove it.
"""
from __future__ import annotations

import importlib.util
import os
import re
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path
from fs_estate_pat import estate_partition_literal

# #157: the estate's short name is an INPUT. It appears here because this anchor has to
# reproduce the before-text verbatim to find it, and that text names the estate. A
# declared-empty estate (NONE) yields the estate-free anchor, not a hole in a sentence.
_SHORT = estate_partition_literal()
_ON_SHORT = f" on {_SHORT}" if _SHORT else ""

BASE = Path(__file__).resolve().parent
GEN = BASE / "h100" / "gen"
LAUNCHER = GEN / "launch_fs_h100.fixed.sh"
BACKEND = GEN / "fs_container_backend.bound.sh"
DRIFT_GATE = BASE / "gate_env_drift.py"
MARKER = "fs124:"

FAILURES: list[int] = []


def gate(n: int, ok: bool, claim: str) -> None:
    print(f"  {'PASS' if ok else 'FAIL'} G{n}  {claim}")
    if not ok:
        FAILURES.append(n)


# ---------------------------------------------------------------------------
# ANCHOR A and its replacement (launcher). The replacement keeps the fs123
# comment (history of a neighbouring deletion), wires the measured gpu count
# through fs_compose_launch, and drops a site-specific phrase from the old
# fail message because this repo is public.
# ---------------------------------------------------------------------------
ANCHOR_A = r'''# Engine remains pluggable. The core launcher does not name NeMo/Megatron/Gemma/Qwen.
# The selected engine adapter must provide a complete in-container command in CONFIG_FILE or FS_ENGINE_LAUNCH_CMD.
LAUNCH_CMD="${FS_ENGINE_LAUNCH_CMD:-}"
[[ -n "$LAUNCH_CMD" ]] || fail 96 "FS_ENGINE_LAUNCH_CMD unset; engine adapter/config must provide launch command (Megatron-Bridge is absent@@SHORT@@)"

# fs123: the write-only `mounts=(...)` array that stood here is DELETED. Measured: zero
# readers -- nothing in this launcher ever expanded it, and the backend's arm-local
# `local -a mounts` is a different variable entirely. It was a dead duplicate of the
# FS_BIND_PATHS plane declared above, and a second mount declaration beside the live
# one is how the next reader concludes that mounts are declared here.
fs_begin_log_tee "$RUN_LOG" || true
printf 'BEGIN phase=%s probe=%s out=%s image=%s model_dir=%s dataset_dir=%s config=%s gpus=%s\n' "${FS_PHASE:-train}" "$PROBE" "$OUT_DIR" "$IMAGE" "$MODEL_DIR" "$DATASET_DIR" "$CONFIG_FILE" "$FS_GPUS_PER_NODE"

set +e
run_in_container --workdir "$OUT_DIR" -- bash -lc "$LAUNCH_CMD" 2>&1 | tee -a "$RUN_LOG"'''
ANCHOR_A = ANCHOR_A.replace("@@SHORT@@", _ON_SHORT)

REPL_A = r'''# Engine remains pluggable. The core launcher does not name NeMo/Megatron/Gemma/Qwen.
# The selected engine adapter must provide a complete in-container command in CONFIG_FILE or FS_ENGINE_LAUNCH_CMD.
LAUNCH_CMD="${FS_ENGINE_LAUNCH_CMD:-}"
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
  || fail 124 "fs_compose_launch refused: ${TOPO_OUT:-<no output>}"
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

set +e
run_in_container --workdir "$OUT_DIR" "${top_args[@]}" -- bash -lc "$LAUNCH_CMD" 2>&1 | tee -a "$RUN_LOG"'''

# ---------------------------------------------------------------------------
# ANCHOR B and its replacement (backend): fs_launch_python is DELETED and
# replaced by fs_compose_launch. The comment states why the enroot/else
# branch was the bug, not the style.
# ---------------------------------------------------------------------------
ANCHOR_B = r'''# ---------------------------------------------------------------------------
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
}'''

REPL_B = r'''# ---------------------------------------------------------------------------
# fs_compose_launch <mode> <gpus> <engine_cmd> -- echoes:
#     <world_size_source>\t<final_cmd>
# fs124: this REPLACES fs_launch_python, which is DELETED. Its enroot/else
# branch was the bug, not the style: the else arm returned bare `python3`,
# correct only when some unrelated sbatch happened to supply the ranks via
# --ntasks-per-node. That branched on FS_BACKEND (which container runtime is
# in use) instead of on WHO FORKS THE RANKS, and the function had zero call
# sites -- so the launcher proved 8 visible devices, started exactly ONE
# process, and exited 0: the measurement was checked, never load-bearing.
# Rank multiplicity is decided by WHO FORKS THE RANKS:
#   torchrun  FoundationScale composes the prefix from the MEASURED gpu
#             count, so the measurement becomes load-bearing.
#   wlm       the workload manager forks ranks; REFUSE unless the effective
#             per-node task count (SLURM_NTASKS_PER_NODE) is numeric and
#             equals the measured gpu count. The caller additionally passes
#             --slurm-ntasks <gpus> to run_in_container.
#   self      the engine forks its own ranks (deepspeed/accelerate/custom);
#             FoundationScale cannot compose or count those, so it REQUIRES
#             FS_ENGINE_PROCS_PER_NODE == gpus and stamps the source
#             engine-declared, so no report may call it measured.
# Only nnodes/node_rank get single-node defaults (those are safe); the policy
# variables (mode, procs-per-node) never do.
# ---------------------------------------------------------------------------
fs_compose_launch() {
  local mode=${1:-} gpus=${2:-} engine_cmd=${3:-}
  [[ -n "$mode" ]] || fs_die "fs_compose_launch: FS_ENGINE_LAUNCH_MODE unset/empty; required, no default (torchrun|wlm|self)"
  case "$mode" in torchrun|wlm|self) ;; *) fs_die "fs_compose_launch: unrecognised mode '$mode' (torchrun|wlm|self)" ;; esac
  [[ "$gpus" =~ ^[0-9]+$ ]] || fs_die "fs_compose_launch: gpu count '$gpus' is not numeric; refusing to compose from an unmeasured device layer"
  (( gpus >= 1 )) || fs_die "fs_compose_launch: gpu count '$gpus' < 1"
  case "$mode" in
    torchrun)
      printf 'composed\ttorchrun --nproc_per_node=%s --nnodes=%s --node_rank=%s --master_addr=%s --master_port=%s %s\n' \
        "$gpus" "${SLURM_NNODES:-1}" "${SLURM_NODEID:-0}" \
        "${MASTER_ADDR:?MASTER_ADDR must be exported by fs_backend_init before composition}" \
        "${MASTER_PORT:?MASTER_PORT must be exported by fs_backend_init before composition}" \
        "$engine_cmd"
      ;;
    wlm)
      local tpn=${SLURM_NTASKS_PER_NODE:-}
      [[ "$tpn" =~ ^[0-9]+$ ]] || fs_die "fs_compose_launch wlm: SLURM_NTASKS_PER_NODE='$tpn' unset or non-numeric; unmeasured is not pass"
      (( tpn == gpus )) || fs_die "fs_compose_launch wlm: per-node tasks ($tpn) != measured gpus ($gpus); refusing to start $tpn process(es) on $gpus devices"
      printf 'wlm\t%s\n' "$engine_cmd"
      ;;
    self)
      local declared=${FS_ENGINE_PROCS_PER_NODE:-}
      [[ -n "$declared" ]] || fs_die "fs_compose_launch self: the engine forks its own ranks, so FS_ENGINE_PROCS_PER_NODE is required and unset"
      [[ "$declared" == "$gpus" ]] || fs_die "fs_compose_launch self: FS_ENGINE_PROCS_PER_NODE=$declared != measured gpus=$gpus"
      printf 'engine-declared\t%s\n' "$engine_cmd"
      ;;
  esac
}'''

# ---------------------------------------------------------------------------
# ANCHOR C and its replacement (backend allowlist tail).
# ---------------------------------------------------------------------------
ANCHOR_C = r'''                                   #   nothing to compare to and the proof degrades to 'it
                                   #   did not crash', which is not evidence of resuming.
    NCCL_SOCKET_IFNAME
    GLOO_SOCKET_IFNAME
    NCCL_MNNVL_ENABLE
  )'''

# REPL_C == ANCHOR_C: the allowlist is deliberately UNCHANGED.
#
# The first draft of this patch appended FS_ENGINE_LAUNCH_MODE and
# FS_ENGINE_PROCS_PER_NODE here, justified as "launcher-produced policy the
# trainer will read". gate_env_drift.py went red on exactly that (D3: two
# allowlisted names with no producer), and it was right: the justification
# asserted a future in-container consumer that does not exist. The draft's own
# comment conceded "nothing in-container consumes these yet" -- an allowlist
# entry whose stated reason is that nothing reads it is the dead weight D3
# exists to catch, the same shape as MASTER_PORT before it was minted and the
# write-only mounts array deleted in fs123.
#
# Both names are HOST-SIDE control-plane inputs, consumed by fs_compose_launch
# before any container starts -- the same family as FS_CONTAINER_RUNTIME,
# FS_ALLOCATION and FS_BACKEND, which gate_env_drift already declares host-only
# with stated reasons. Nothing crosses the boundary because nothing on the other
# side asks. If a future trainer does read them, THAT is when they earn a place
# on the list, together with the reader that makes the entry true.
REPL_C = ANCHOR_C


def _allowlist_names(backend_text: str) -> list[str]:
    """Every name in FS_ENV_ALLOWLIST, including the ones carrying trailing
    comments.

    Written deliberately to match gate_env_drift.allowlist(): strip the comment
    FIRST, then match. A `^\\s+NAME\\s*$` pattern silently drops every commented
    entry -- 11 of 23 in this file -- and then reports the remainder as if it
    were the whole list. That is how the first draft printed "14 names, was 12"
    against a real list of 23, i.e. a denominator that was not the denominator.
    """
    m = re.search(r"(?ms)^  FS_ENV_ALLOWLIST=\(\n(.*?)^  \)", backend_text)
    if not m:
        return []
    out = []
    for ln in m.group(1).splitlines():
        tok = ln.split("#")[0].strip()
        if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", tok):
            out.append(tok)
    return out


# ---------------------------------------------------------------------------
# G7: execution harness. The emitted composer is extracted from the patched
# backend by regex and executed row-by-row in a cleaned environment.
# ---------------------------------------------------------------------------
def run_g7(patched_backend: str) -> bool:
    defs = re.findall(r"(?m)^fs_compose_launch\(\) \{$", patched_backend)
    m = re.search(r"(?ms)^fs_compose_launch\(\) \{\n.*?^\}$", patched_backend)
    if len(defs) != 1 or not m:
        gate(7, False, f"composer body extraction matched {len(defs)}/1 definitions in patched backend")
        return False
    harness = (
        "fs_die() { printf 'REFUSED: %s\\n' \"$*\" >&2; exit 1; }\n"
        + m.group(0)
        + "\n"
    )

    base_env = {k: v for k, v in os.environ.items()
                if not k.startswith(("FS_", "SLURM_"))}

    def run_row(mode: str, gpus: str, extra: dict[str, str]):
        env = dict(base_env)
        env.update(extra)
        call = (
            harness
            + f"\nfs_compose_launch {shlex.quote(mode)} {shlex.quote(gpus)} engine-training-cmd\n"
        )
        p = subprocess.run(["bash", "-c", call], env=env,
                           capture_output=True, text=True, timeout=30)
        return p.returncode, p.stdout, p.stderr

    torch_env = {"MASTER_ADDR": "127.0.0.1", "MASTER_PORT": "29500"}
    rows = [
        # (label, mode, gpus, env_extra, predicate(rc, out, err))
        ("MUST_PASS torchrun gpus=8 composed, --nproc_per_node=8",
         "torchrun", "8", dict(torch_env),
         lambda rc, out, err: rc == 0 and "--nproc_per_node=8" in out
                              and out.startswith("composed\t")),
        ("MUST_FIRE torchrun pins the count: stdout has NO '--nproc_per_node=1'",
         "torchrun", "8", dict(torch_env),
         lambda rc, out, err: rc == 0 and "--nproc_per_node=1" not in out),
        ("MUST_PASS wlm gpus=8 ntasks_pn=8 admitted, source=wlm",
         "wlm", "8", {"SLURM_NTASKS_PER_NODE": "8"},
         lambda rc, out, err: rc == 0 and out.startswith("wlm\t")),
        ("MUST_FIRE wlm gpus=8 ntasks_pn=1 REFUSED (the live bug)",
         "wlm", "8", {"SLURM_NTASKS_PER_NODE": "1"},
         lambda rc, out, err: rc != 0),
        ("MUST_FIRE wlm gpus=8 ntasks_pn unset REFUSED (unmeasured is not pass)",
         "wlm", "8", {},
         lambda rc, out, err: rc != 0),
        ("MUST_PASS self gpus=8 procs=8 admitted, source=engine-declared",
         "self", "8", {"FS_ENGINE_PROCS_PER_NODE": "8"},
         lambda rc, out, err: rc == 0 and out.startswith("engine-declared\t")),
        ("MUST_FIRE self gpus=8 procs unset REFUSED",
         "self", "8", {},
         lambda rc, out, err: rc != 0),
        ("MUST_FIRE self gpus=8 procs=1 REFUSED",
         "self", "8", {"FS_ENGINE_PROCS_PER_NODE": "1"},
         lambda rc, out, err: rc != 0),
        ("MUST_FIRE mode unset REFUSED (required, no default)",
         "", "8", {},
         lambda rc, out, err: rc != 0),
        ("MUST_FIRE mode=bogus REFUSED",
         "bogus", "8", {},
         lambda rc, out, err: rc != 0),
        ("MUST_FIRE gpus=notanumber REFUSED",
         "torchrun", "notanumber", dict(torch_env),
         lambda rc, out, err: rc != 0),
    ]
    passed, failed = 0, []
    for label, mode, gpus, extra, pred in rows:
        rc, out, err = run_row(mode, gpus, extra)
        if pred(rc, out, err):
            passed += 1
        else:
            failed.append(f"{label} (rc={rc}; stdout={out.strip()!r}; stderr={err.strip()!r})")
    ok = not failed
    gate(7, ok,
         f"{passed}/{len(rows)} rows executed clean (4 MUST_PASS, 7 MUST_FIRE; "
         f"controls EXECUTED in clean env, never asserted by reading)")
    for f in failed:
        print(f"       FAIL row: {f}")
    return ok


def main() -> int:
    if not LAUNCHER.exists() or not BACKEND.exists():
        gate(2, False, f"input files present {int(LAUNCHER.exists()) + int(BACKEND.exists())}/2 "
                       f"({LAUNCHER.name}, {BACKEND.name})")
        return 1

    lt = LAUNCHER.read_text()
    bt = BACKEND.read_text()

    # G1 idempotence + half-applied refusal.
    m_l, m_b = MARKER in lt, MARKER in bt
    if m_l and m_b:
        gate(1, True, "marker fs124: present in 2/2 files; already applied; no-op exit 0")
        return 0
    if m_l != m_b:
        gate(1, False,
             f"half-applied: marker fs124: present in {int(m_l) + int(m_b)}/2 files "
             f"(launcher={m_l}, backend={m_b}); refusing to patch a half-applied state")
        return 1
    gate(1, True, "marker fs124: present in 0/2 files; first application proceeds")

    # G2..G4 anchors unique (exactly once each).
    c_a = lt.count(ANCHOR_A)
    gate(2, c_a == 1, f"launcher anchor A occurs {c_a}/1 time(s)")
    c_b = bt.count(ANCHOR_B)
    gate(3, c_b == 1, f"backend fs_launch_python anchor B occurs {c_b}/1 time(s)")
    c_c = bt.count(ANCHOR_C)
    gate(4, c_c == 1, f"backend allowlist anchor C occurs {c_c}/1 time(s)")
    pre_allow = _allowlist_names(bt)
    if FAILURES:
        return 1

    pl = lt.replace(ANCHOR_A, REPL_A)
    pb = bt.replace(ANCHOR_B, REPL_B).replace(ANCHOR_C, REPL_C)

    # G5 bash -n on both patched texts (written to tempfiles, executed, removed).
    syntax_ok = 0
    for text, tag in ((pl, "launcher"), (pb, "backend")):
        fd, tmp = tempfile.mkstemp(prefix="fs124_", suffix=".sh")
        try:
            with os.fdopen(fd, "w") as fh:
                fh.write(text)
            r = subprocess.run(["bash", "-n", tmp], capture_output=True, text=True)
            if r.returncode == 0:
                syntax_ok += 1
            else:
                print(f"       bash -n {tag}: rc={r.returncode} {r.stderr.strip()}")
        finally:
            os.unlink(tmp)
    gate(5, syntax_ok == 2, f"bash -n clean on {syntax_ok}/2 patched texts (tempfiles, executed)")

    # G6 the allowlist is UNCHANGED, and the two new policy names stay OFF it.
    #
    # Stated positively rather than as an omission: the boundary crossing set is
    # a claim about what in-container code reads, and neither name has an
    # in-container reader. The gate asserts the set is byte-identical AND that
    # the two names are absent, so a later "helpful" addition goes red here
    # rather than quietly re-creating the D3 violation.
    post_allow = _allowlist_names(pb)
    if not pre_allow:
        gate(6, False, "FS_ENV_ALLOWLIST unparseable (0 names) — the gate cannot certify anything")
    else:
        leaked = [n for n in ("FS_ENGINE_LAUNCH_MODE", "FS_ENGINE_PROCS_PER_NODE")
                  if n in post_allow]
        gate(6, post_allow == pre_allow and not leaked,
             f"FS_ENV_ALLOWLIST unchanged at {len(post_allow)}/{len(pre_allow)} names; "
             f"{len(leaked)}/2 topology-policy names on the list "
             f"(want 0 — both are host-side, consumed by fs_compose_launch before "
             f"any container exists)")

    # G7 execute the composer.
    run_g7(pb)

    # Refuse to write while any gate is red.
    if FAILURES:
        print("  WRITE REFUSED: one or more gates red; files untouched")
        return 1

    LAUNCHER.write_text(pl)
    BACKEND.write_text(pb)
    print(f"  WROTE 2/2 files: {LAUNCHER.name}, {BACKEND.name}")

    # G8 integration: env-drift gate on the PATCHED texts, after writing.
    try:
        spec = importlib.util.spec_from_file_location("gate_env_drift", str(DRIFT_GATE))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        issues = mod.check(pl, pb, quiet=True)
    except Exception as exc:  # import/signature drift is itself red
        gate(8, False, f"gate_env_drift import/check raised on patched texts: {exc!r}")
        return 5
    gate(8, issues == [],
         f"env-drift check returned {len(issues)} issue(s)/0 on patched texts (after write)")
    for i in issues or []:
        print(f"       drift: {i}")
    if issues:
        return 5

    # G9 the gate that would have caught #124: call-site counting.
    call_sites = 0
    for line in pl.splitlines():
        if "fs_compose_launch" not in line:
            continue
        s = line.strip()
        if s.startswith("#") or re.match(r"^fs_compose_launch\s*\(\)", s):
            continue
        call_sites += 1
    old_defs = len(re.findall(r"(?m)^\s*fs_launch_python\s*\(\)", pb))
    gate(9, call_sites >= 1 and old_defs == 0,
         f"fs_compose_launch call sites in launcher: {call_sites}/>=1; "
         f"fs_launch_python definitions remaining in backend: {old_defs}/0 -- "
         f"a composer with zero call sites is the defect being fixed (#124), "
         f"so a future regression reads as red here")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
