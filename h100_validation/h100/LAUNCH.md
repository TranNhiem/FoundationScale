# LAUNCH.md — FoundationScale, H100 single-node launch plane (submit partition set at submit time by `$FS_PARTITION`)

> **Read this box before anything below it.** Phase 3 — the first genuine end-to-end
> 8×H100 run (load → distributed → shard → train → save → resume → eval) — has **never
> been executed** on real hardware (matrix E.6). The build plane is green (17/17 stages,
> E.5) and the 8-rank collective probe is measured (E.3), but **every training leg is
> UNMEASURED**. This document tells you how to launch; it does not tell you the launch
> will train. §8 is the trust section.

---

## 1. What this launches, in three sentences

1. This plane submits a **single-node, 8×H100** training job to the Slurm partition
   named by `$FS_PARTITION` — an operator input with no default, refused with exit 96
   when unset (launcher line 28) — runs an engine command **you supply** inside a
   **singularity** container, and adjudicates every checkpoint the run writes — a run
   that saves nothing it can verify is reported as UNMEASURED (exit 95), not as success.
2. The requested canonical interface is:
   `foundation-scale --model <model> --dataset <dataset> --num-gpus 8 --config <config>`
3. **That flag form is NOT YET IMPLEMENTED.** The current plane is env-driven, not
   flag-driven. Each flag maps onto a named environment variable, and the launcher
   refuses (exit 96) on any missing one. Do not type the command above; nothing parses it.

| Requested flag | Environment variable today |
|---|---|
| `--model <model>` | `MODEL_DIR` |
| `--dataset <dataset>` | `DATASET_DIR` |
| `--num-gpus 8` | `FS_GPUS_PER_NODE=8` |
| `--config <config>` | `CONFIG_FILE` |
| *(no flag exists)* | `FS_PARTITION`, `IMAGE`, `OUT_DIR_STABLE`, `FS_ALLOCATION`, `FS_CONTAINER_RUNTIME`, `FS_ALLOWED_PATH_ROOTS`, `FS_ALLOWED_NODE`, `FS_ENGINE_LAUNCH_MODE`, `FS_ENGINE_LAUNCH_CMD`, `FS_NCCL_NET_PLUGIN`, `FS_CHECKPOINT_ADJUDICATORS` — these are **also required** and have no flag counterpart yet |

Engine: the **only** measured-available engine is HuggingFace transformers + torch
FSDP/DDP. `import megatron` → ModuleNotFoundError; `import nemo_automodel` →
PermissionError (measured 2026-08-31). Do not write a Megatron/NeMo launch command.

---

## 2. The 60-second version (Phase 3 probe)

Fill in the `<...>` placeholders, then run the last line from a login node.

```bash
# --- axis guards: exact values, no defaults exist by design ---
export FS_PARTITION=<this estate's Slurm submit partition>    # REQUIRED, no default: launcher line 28 refuses (exit 96) when unset;
                                                              # the name travels as --partition="$FS_PARTITION" on the sbatch invocation
export FS_ALLOCATION=slurm                              # who allocated the nodes; must be exactly 'slurm'
export FS_CONTAINER_RUNTIME=singularity                 # container runtime; must be exactly 'singularity'
export FS_ALLOWED_NODE=<hostname of the allocated node> # standing safety rule: refuse to run on the wrong node

# --- path policy: roots reachable from INSIDE the container (space-separated) ---
export FS_ALLOWED_PATH_ROOTS="<absolute root 1> <absolute root 2>"

# --- the flag-form equivalents, as env vars ---
export MODEL_DIR=<absolute path to model checkpoint dir>      # must sit under an allowed root
export DATASET_DIR=<absolute path to dataset dir>             # must sit under an allowed root
export CONFIG_FILE=<absolute path to engine config file>      # must be readable
export FS_GPUS_PER_NODE=8                                     # this partition has 8× H100 SXM per node

# --- plane-specific required inputs ---
export IMAGE=<absolute path to container image>.sif           # readable .sif, under an allowed root
export OUT_DIR_STABLE=<absolute path to output dir>           # NO job id in the name; under an allowed root or $HOME
export PROBE=1                                                # 1 = bounded probe run (budget 20 / early save 5 by default)

# --- engine wiring (no defaults; an unconfigured guard is a disabled rule) ---
export FS_ENGINE_LAUNCH_MODE=torchrun                         # torchrun|wlm|self; who forks the ranks
export FS_ENGINE_LAUNCH_CMD="python3 <absolute path to fs_phase3_train.py> --model $MODEL_DIR --ckpt-dir \$OUT_DIR/ckpt --steps 20"
#   in torchrun mode the launcher composes --nproc_per_node from the MEASURED gpu count (fs124);
#   supply the inner engine command, not your own torchrun wrapper

# --- measured collective-plane fix (E.3): image default plugin SIGSEGVs in the first all_reduce ---
export FS_NCCL_NET_PLUGIN=none

# --- checkpoint adjudication: absence is not success (all([]) != PASS) ---
export FS_CHECKPOINT_ADJUDICATORS=<absolute path to plane dir>/h100/gen/fs_ckpt_adjudicator.py
#   each spec is containment-checked against FS_ALLOWED_PATH_ROOTS: outside every root it is REFUSED
#   (exit 96), not skipped (fs146a); each spec's dirname is bound into the container automatically (fs146b)
export FS_EXTRA_BIND_PATHS=<absolute path to plane dir>       # escape hatch for trees inference cannot see; see §7

# --- submit the probe ---
sbatch <absolute path to plane dir>/launch_fs_h100.fixed.sh
```

For the full probe → production → resume → post-mortem chain instead, export
`FS_SUBMIT_CHAIN=1` and run the launcher **on the login node** (it submits the chain
itself). See §6.

Leave `FS_WALLTIME` unset. The `#SBATCH --time=7-00:00:00` directive is correct for the
partition named by `$FS_PARTITION`; any other value is refused, not clamped — **including
legitimately shorter values** (launcher walltime guard; see the known limitation at the
end of §7 before you are tempted to set one).

Do not set `FS_PLANE_DIR` unless you need to override plane resolution: the launcher
resolves the plane directory automatically (§3, bypass controls; §7). If you do set it,
set it to the directory containing `fs_container_backend.bound.sh` — the submit chain
re-submits the launcher through it, so a wrong override breaks every hop at once (§6).

---

## 3. Every knob, by category, from the polarity oracle (#127)

Categories are verbatim from the oracle. Citation legend: **L:\<n\>** = launcher line,
**B:\<n\>** = container-backend line. The B: citations are as cited by the oracle against
the generated backend on 2026-08-31. The L: citations have been **re-aligned to the
current launcher**: a self-contained plane-directory resolver (fs142) now occupies
L:34–182 and pushed every later line down, so older oracle line numbers no longer touch
the knobs they were about. The oracle's measured universe is **41 distinct `FS_*`
names**; the tables below reproduce every name the oracle assigns explicitly. Names
present in code but not classified in the oracle's published tables are listed separately
at the end and are **not** silently bucketed.

This document is hand-written from the oracle tables: the Deliverable D generator
(`gate_launch_contract.py`) is red on L2 ("6 of 18") and correctly refuses to write — the
gate's count misses the names reached through locals and the array-length test.

### REQUIRED — unset or empty is fatal, and no default exists

| Name | What it is | If unset | Enforced |
|---|---|---|---|
| `FS_PARTITION` | this estate's Slurm submit partition. The `#SBATCH --partition=` directive was **deleted, not parameterised** (fs152): an `#SBATCH` line is a comment to the shell, so the partition travels as `--partition="$FS_PARTITION"` on the sbatch invocation, where expansion actually happens (L:9–13) | exit 96 refuse — "required, no default by design … the framework refuses to guess a cluster layout" (a default would be the deleted literal compiled back in) | L:28; carried on every chain `sbatch` at L:585–591 |
| `FS_ALLOCATION` | who allocated the nodes; must be exactly `slurm` on this launcher | exit 96 refuse (`${FS_ALLOCATION:-}` empty fallback preserves required-no-default under `set -u`, #126) | B:160; launcher prologue |
| `FS_ALLOWED_NODE` | node this plane is permitted to run on (standing safety rule) | exit 96 refuse | B:290 |
| `FS_CONTAINER_RUNTIME` | container runtime; must be exactly `singularity` | exit 96 refuse | B:154; launcher prologue |
| `FS_ALLOWED_PATH_ROOTS` | space/tab-separated absolute roots reachable in-container | exit 96 refuse — "no default by design … refuses to guess a filesystem layout" | L:227 |
| `FS_ENGINE_LAUNCH_MODE` | who forks ranks: `torchrun` / `wlm` / `self` | exit refuse, "required, no default" (read via local `mode`) | B:613; demanded by `fs_compose_launch` at L:760–761 |
| `FS_CONTAINER_SQSH` | container image path, operator-supplied (R7) | exit refuse | B:1272 |
| `FS_BIND_PATHS` | array of host paths bound into the container; **derived** by the launcher from `MODEL_DIR`, `DATASET_DIR`, `dirname(CONFIG_FILE)`, `OUT_DIR`, each adjudicator spec's dirname (fs146(b), L:317–321), and `FS_EXTRA_BIND_PATHS` | zero entries ⇒ exit 96 — "a derivation bug, not a legal empty set" | L:524 (array-length test); derivation loop L:506–515 |
| `FS_CHECKPOINT_ADJUDICATORS` | space/tab/newline-separated adjudicator commands, one per word; each invoked as `<cmd> <ckpt_dir> <phase> <out_dir>`. Every spec is containment-checked against `FS_ALLOWED_PATH_ROOTS` (fs146(a)): a spec outside every declared root is **REFUSED** (exit 96, naming the offending spec *and* the declared roots), never skipped — and each spec's dirname is then bound into the container (fs146(b)), because a refused-after-hours knob was finding #146 | exit 96 refuse — `all([])!=PASS` (read via local `ADJUDICATORS_RAW`) | L:273 (empty refuse), L:272–296 (zero specs); containment L:296–306 |
| `FS_ENGINE_LAUNCH_CMD` | complete in-container engine command | exit 96 refuse (read via local `LAUNCH_CMD`) | L:738 |
| `FS_NCCL_NET_PLUGIN` | NCCL net-plugin selection; measured correct value on this estate is **`none`** | collective probe refuses the launch; without the fix the 8-rank all_reduce **SIGSEGVs** inside the first collective (E.3, measured) | L:417, L:427, L:729 |

Value note: `FS_WALLTIME`, if set, must be exactly `7-00:00:00` — but it is
VALIDATED-IF-SET, not REQUIRED; leave it unset (below, and §7's known limitation).

### CONDITIONALLY REQUIRED — required only on one branch

| Name | What it is | If unset | Enforced |
|---|---|---|---|
| `FS_ENGINE_PROCS_PER_NODE` | ranks per node, **only** in `self` mode (the engine forks its own ranks and must declare `== gpus`) | in `self` mode: fatal; in `torchrun`/`wlm` mode: **must not be given** | B:651 |

### FORBIDDEN-IF-SET

| Name | What it is | If set | Enforced |
|---|---|---|---|
| `FS_NCCL_IB_HCA` | InfiniBand HCA pinning | exit 96 refuse — pinning is unmeasured on the partition named by `$FS_PARTITION`; leave unset unless measured and validated | L:395 |

The measurement narrowing this refusal now exists (#130); the table reports what the code
**does**, not what it should do.

### VALIDATED-IF-SET — value constrained, absence tolerated

| Name | What it is | If set to a bad value | Enforced |
|---|---|---|---|
| `FS_NCCL_SOCKET_IFNAME` | socket interface name | must resolve via `ip link show`, else 96. **Known defect (#131): validated then discarded — a knob with no reader** | L:392 |
| `FS_WALLTIME` | walltime override | anything other than `7-00:00:00` ⇒ 96 refuse — a hard-coded literal guard that runs **before** the live `sinfo` probe of the partition named by `$FS_PARTITION`: it refuses instead of clamping, and it refuses legitimately *shorter* values too (§7 known limitation) | L:332–333 (literal oracle); live `sinfo` oracle at L:358–365, comparison L:367–387 |
| `FS_GPUS_PER_NODE` | GPUs per node | must be integer > 0 (L:325–326); cross-checked against `SLURM_GPUS_PER_NODE` | L:325–326, L:645. **Note:** the launcher's own `req_env` also refuses when unset (L:206) — set it (§2) even though the value checks are the validated-if-set part |
| `FS_ITERATION_BUDGET` | positive integer step budget | must be positive int; must exceed `FS_EARLY_SAVE_STEPS` | L:465, L:467 |
| `FS_EARLY_SAVE_STEPS` | steps before early save | must be positive int and `<` the budget, "an early save that cannot fire is not evidence" | L:466–467 |
| `FS_BACKEND` | legacy selector | restricted to `slurm-singularity`/`singularity`; has a real default (`slurm-singularity`) | L:199–200 |

### ENVIRONMENT — minted by the script, never an operator input

| Name | What it is | If you set it | Enforced |
|---|---|---|---|
| `FS_ACTUAL_HOST` | actual hostname, minted as `$(hostname -s)` | the script **overwrites** your value | B:337 |

### HAS-A-DEFAULT — cannot be REQUIRED

| Name | Default |
|---|---|
| `FS_BACKEND` | `slurm-singularity` (L:199) |
| `FS_USE_TORCHRUN` | `0`/`1` (B:315, B:394) |
| `FS_EARLY_SAVE_STEPS`, `FS_ITERATION_BUDGET` | on the resume path, default through `FS_RESUME_*` (L:625–626); in probe phase, `5` and `20` with source logged (launcher probe block, L:463–464) |

### Bypass controls and launcher-observed names not in the oracle's published buckets

Listed, not silently classified. The oracle is emphatic about the first one:

| Name | What it is |
|---|---|
| `FS_SUBMIT_CHAIN` | `=1` on a login node runs the four-job submit chain (§6). **It must NOT be classified REQUIRED.** The L:644 guard is on `SLURM_JOB_ID`; `FS_SUBMIT_CHAIN` appears only inside its quoted refusal message — it is what you set to *bypass* the check. A proximity read of that line inverts the knob's meaning. |
| `FS_PLANE_DIR` | **OPTIONAL** operator override naming the plane directory. Resolution is **automatic** and ordered (fs142 resolver, L:34–182): step 1 verifies that `$FS_PLANE_DIR`, if set, contains a readable `fs_container_backend.bound.sh` (L:85–101); step 2 falls back to `SCRIPT_DIR` so a direct `bash launch…` invocation keeps working with no new operator variable (L:106–118); step 3 asks the workload manager for the submitted script's original path (`scontrol show job` on slurm, L:59–80) and **verifies the sibling backend there** rather than trusting the claim (L:120–148); step 4 refuses with `FATAL[142]`, printing all three attempted answers and the last directory searched (L:154–163). The verified answer is exported so later jobs and child processes inherit it (L:178). **Warning:** the submit chain re-submits the launcher *through* this variable — every one of the four `sbatch` calls addresses `"$FS_PLANE_DIR/$(basename "$0")"` (L:585–591) — so overriding it wrongly breaks the entire probe → production → resume → post-mortem chain, not one job. Only set it to override a mis-resolution. |
| `FS_EXTRA_BIND_PATHS` | optional space-separated escape hatch for paths the launcher cannot infer (e.g., a code tree, a scratch root); word-split on purpose (launcher fs117 block, L:502–508). Since fs146(b) the adjudicator dirnames join the bind plane automatically, so pointing this at the adjudicator's tree is now redundant but harmless (de-duplicated, L:502–514) |
| `FS_PHASE` | `train` (default) / `resume` / `post-mortem`, set by the chain driver on the resume and post-mortem hops (launcher chain block, L:589, L:591) |

---

## 4. The artifact map

Every file under `h100/gen/` is **GENERATED** by `build_h100_plane.sh` (Deliverable C,
17/17 stages green, byte-identical across rebuilds). **Hand-editing any of them is
prohibited**: a hand edit puts the fix in the file you read and leaves it out of the file
that runs. Change the stage, rebuild.

| Artifact | What it is | Reads / read by |
|---|---|---|
| `h100/gen/launch_fs_h100.fixed.sh` | the sbatch-submitting launcher (~800 lines; the fs142 plane-directory resolver occupies L:34–182 at the top) | resolves the plane directory (operator override → `SCRIPT_DIR` → workload-manager `Command=` path, refusing `FATAL[142]` if none verify); sources the backend from the resolved directory; derives `FS_BIND_PATHS`; invokes adjudicator entries; composes the launch command |
| `h100/gen/fs_container_backend.bound.sh` | container-runtime backend (`run_in_container`, bind materialization, guards) | sourced by the launcher (from `$FS_PLANE_DIR`, L:179–181); consumes `FS_BIND_PATHS`, `FS_ALLOWED_NODE`, `FS_CONTAINER_RUNTIME`, `FS_ENGINE_LAUNCH_MODE`; built from the spliced base below |
| `h100/gen/fs_container_backend.spliced.sh` | **intermediate**: backend base text from the upstream repo (#136) | read by the backend stages; removed and rebuilt every run; not shipped |
| `h100/gen/fs_train.fixed.py` | in-container training entrypoint | runs inside the container via `run_in_container`; reads `FS_ITERATION_BUDGET`/`FS_EARLY_SAVE_STEPS` across the allowlisted boundary; imports the resolver below |
| `h100/gen/fs_model_root.py` | model-root resolver (#133): config searched per-depth, shallowest populated depth wins, bind closure not declared root | imported by the training entrypoint via the stage-C binding (`load_artifacts` → `resolve_model_root`); before that binding it was an orphan — 12 green tests, zero callers |
| `h100/gen/test_fs_model_root.py` | generated suite for the resolver | run **by the build**; a suite nobody runs is an orphan (#86) |
| `h100/gen/fs_ckpt_adjudicator.py` | the checkpoint adjudicator the launcher's required knob asks for (#141) | invoked per checkpoint dir as `<cmd> <ckpt_dir> <phase> <out_dir>`; its dirname is bound **inside** the container automatically by fs146(b) (see §3, §7) |
| `h100/gen/test_fs_ckpt_adjudicator.py` | generated suite for the adjudicator | run by the build |

Logs: `$OUT_DIR/logs/launch.<jobid>.log` (or `launch.interactive.log`), teed from
`BEGIN` to `END`; checkpoint evidence lands in the `ADJUDICATORS observed=… seen=… ok=…`
and `END … checkpoint_saves_adjudicated=N` lines — the N is the denominator.

---

## 5. The exit-code contract

| rc | Meaning | Operator response |
|---|---|---|
| **0** | Launch succeeded **and** ≥1 checkpoint dir was adjudicated (`checkpoint_saves_adjudicated=N`, N>0) | Read the denominators in the log. A 0 with no adjudicated save is impossible — the script refuses it |
| **95** | **UNMEASURED**: training ran but **no checkpoint-save units were observed** (`adjudicate_tree` found no checkpoint dirs, or zero observed after training). **This is not failure.** `all([])` is True, so zero units measured can never be reported as PASS; the plane reports the abstention instead | **Do not retry blindly.** A blind retry reproduces the same nothing-to-measure. Find out why nothing was saved (budget too small? engine wrote elsewhere? bind hole?) before resubmitting |
| **96** | **REFUSE**: a guard fired — missing/empty required var (including `FS_PARTITION`, whose first refusal is what an operator following an outdated runbook meets on their first submit), path outside allowed roots, adjudicator outside allowed roots, bad value, walltime conflict, forbidden knob set | Fix the configuration named in the `FATAL[96]` message and resubmit. The refusal is the contract working, not a crash; do not bypass it by emptying the guard |
| **124** | `fs_compose_launch` refused the launch topology (mode/gpu-count mismatch, missing mode, `self` mode with no declared procs) | Fix `FS_ENGINE_LAUNCH_MODE` / `FS_ENGINE_PROCS_PER_NODE`; see the refusal text |

The Phase 3 in-container script (`fs_phase3_train.py`) has its own contract: **0** only if
every leg PASS, **1** any FAIL, **3** any UNMEASURED with none FAILed — and only rank 0
prints `FSLEG`/`FSSUMMARY` lines.

---

## 6. The four-job submit chain

Triggered by running the launcher **on a login node** with `FS_SUBMIT_CHAIN=1`. The
launcher submits four jobs and exits:

| Hop | Dependency | What it proves |
|---|---|---|
| 1. **probe** (`PROBE=1`) | — | the bounded path: budget 20 / early-save 5 by default, each with source logged. A probe with no effective budget is not a probe, and the launcher refuses it |
| 2. **production** (`PROBE=0`) | `afterok:probe` | the full run into stable `OUT_DIR_STABLE` — only attempted if the probe exited 0 |
| 3. **resume** (`FS_PHASE=resume`) | `afterok:prod` | a **real** resume from production's checkpoints. This hop exists because a resume proof that never resumed proves nothing: verifying "resume works" against a job that always started fresh is the vacuous-truth defect applied to recovery |
| 4. **post-mortem** | `afterany:prod` | reporting only; runs whether production passed or failed |

`afterok` on hop 3 is load-bearing: resume only runs against a production that actually
produced. Direct single-job submission (`sbatch`, §2) skips the chain and is the right
form for a first bring-up.

Two things to know about how the chain addresses the launcher. Every hop carries the
submit partition explicitly as `--partition="$FS_PARTITION"` (L:585–591), so unset
`FS_PARTITION` is refused before the first job id exists. And every hop re-submits the
launcher **through the resolved plane directory**, as `"$FS_PLANE_DIR/$(basename "$0")"`
(L:585–591): the resolver's answer (or your override) is inherited by all four jobs, so a
wrong `FS_PLANE_DIR` override breaks the probe → production → resume → post-mortem chain
uniformly, not at a single hop.

---

## 7. Troubleshooting — symptom → cause → fix

| Symptom | Cause | Fix | Status |
|---|---|---|---|
| `FATAL[142]: … remedy: set and export FS_PLANE_DIR=<directory containing fs_container_backend.bound.sh>` (older builds: `FATAL: fs_container_backend.sh not readable at /var/spool/slurmd/job<N>/...`) | `sbatch` stages **and renames** the script to `slurm_script`, so the directory beside `BASH_SOURCE` is the spool directory holding no siblings; `SLURM_SUBMIT_DIR` was only the submit-time CWD and is deliberately not trusted | #142 resolver (L:34–182): try the operator override `FS_PLANE_DIR`, then `SCRIPT_DIR`, then the workload-manager-recorded `Command=` path — each candidate accepted only if the backend is **readable beside it**; if all three fail, the refusal prints every attempt and the last directory searched. The remedy `export FS_PLANE_DIR=<plane dir>` still works and is now *step 1* of the resolver; the chain inherits the winning answer (§6) | **FIXED** (#142; the defect itself was measured on a real staged 8×H100 job and the resolver ships in the launcher header) |
| Refusal naming your `FS_ALLOWED_PATH_ROOTS` **value** when the value was right | the variable documented itself "space-separated" while the script's safety `IFS=$'\n\t'` made spaces not split — a two-root value read as one impossible root. Prose and parser disagreed | #139: function-local `IFS=$' \t\n'` at the split sites, prose and parser now agree in both directions; `patch_list_separators.py` is **last** of the launcher-touching stages so no later stage can reintroduce a split site | **FIXED** (#139; visible in the build stage list and both launcher slices) |
| Adjudicator exists on the host, fails inside the container | historically the bind plane derived from `MODEL_DIR`, `DATASET_DIR`, `dirname(CONFIG_FILE)`, `OUT_DIR` **only**; an adjudicator outside those trees wasn't mounted, and the knob *itself* was unchecked at startup (#146) | #146 (a): every spec is containment-checked against `FS_ALLOWED_PATH_ROOTS` at startup and refused (exit 96) if outside every root — refused, not skipped (L:296–306); (b): each spec's dirname now joins the bind-plane derivation automatically (L:317–321, consumed at L:506). `FS_EXTRA_BIND_PATHS` remains the escape hatch for trees the launcher cannot infer (a code tree, a scratch root) | **FIXED** (#146) for adjudicators under declared roots; the escape hatch still covers what inference cannot see |
| `sbatch: error: ERROR: The number of GPUs is not specified.` | the partition named by `$FS_PARTITION` rejects a job with no `--gpus-per-node` | the static `#SBATCH --gpus-per-node=8` directive in the launcher prologue | **FIXED** (directive present in the shipped prologue) |
| `FATAL[96]: FS_WALLTIME='…' conflicts with <partition> max 7-00:00:00; refusing instead of clamping` — **including when your value is legitimately shorter** | see the known limitation below: the wrong oracle runs first | leave `FS_WALLTIME` unset; the `#SBATCH --time=7-00:00:00` directive already requests the maximum and the live probe (second oracle) verifies it against the real partition | **KNOWN LIMITATION** (fail-closed; see below) |

### Known limitation: the walltime guard consults two oracles, and the wrong one runs first

Stated plainly, without excuse: maximum walltime is asserted by two different oracles
that can disagree.

* **Oracle 1 — a hard-coded literal, running first (L:332–334).** If `FS_WALLTIME` is set
  to anything other than exactly `7-00:00:00`, the launcher refuses with exit 96. The
  refusal message names the partition in `$FS_PARTITION`, but the 7-day figure is a
  literal compiled into the file: this oracle asserts a maximum for a partition **it
  never probed**. The inline comment keeps it deliberately, as a guard against the old
  estate's 10-day rule leaking in through the environment.
* **Oracle 2 — a correct live probe (L:358–365; comparison L:367–387).** The launcher
  asks `sinfo` for the real `MaxTime` of whatever partition `$FS_PARTITION` actually
  names, parses it (handling `UNLIMITED` explicitly rather than guessing), and *proves*
  the submitted `TimeLimit` fits — refusing when it cannot prove it.

Because oracle 1 runs first, it refuses values oracle 2 would accept — most importantly
**shorter** ones. `FS_WALLTIME=0-02:00:00` for a cheap smoke run is refused before the
live probe ever executes, so a cheap smoke run through `FS_WALLTIME` is currently
impossible. The behaviour is fail-closed in both directions: nothing is clamped, nothing
is silently admitted, and a walltime the scheduler would have accepted is refused along
with one it would reject. The remedy today is what §2 says: leave `FS_WALLTIME` unset.

---

## 8. What is UNMEASURED

Blunt list. Each item has never run on real hardware in a way these sources record.

* **Phase 3, all of it.** Load, distributed init, FSDP sharding, 20 training steps, DCP
  save, fresh-process resume, deterministic eval — every leg is **UNMEASURED** (matrix
  E.6; every per-model column in E.1 beyond "Builds" is `UNMEASURED`). "Builds" means the
  config resolved and the architecture instantiated on a **meta device**; weights have
  never been materialised.
* **The four-job chain end-to-end.** The chain driver exists in the launcher; no source
  here records a probe → production → resume → post-mortem sequence completing with a
  real resume. Treat the resume leg as **UNMEASURED** until hop 3 is observed.
* **The Phase 3 training script itself** (`fs_phase3_train.py`): specified, with measured
  environment facts; not recorded as executed.
* **Multi-node.** One node only exists on this estate; there is no working multi-node
  launch and none should be attempted or written.
* **`FS_NCCL_IB_HCA` pinning on the partition named by `$FS_PARTITION`.** Unmeasured; the
  code forbids setting it (#130 has produced the measurement but the narrowing is not in
  the sources here).
* **`FS_NCCL_SOCKET_IFNAME`'s effect.** The value is validated and then **discarded**
  (#131) — setting it changes nothing downstream.
* **The flag-form CLI** (`--model --dataset --num-gpus --config`). **Not implemented.**
* **`FS_PLANE_DIR` resolver fallbacks on a full chain.** No longer "remedy named,
  implementation missing": the fs142 resolver ships in the launcher header (L:34–182,
  override at L:85–101, export at L:178), and the staging defect it fixes was itself
  measured on a real staged job. What remains unobserved is a complete four-hop chain
  exercising the resolved plane directory end-to-end — which is the chain item above,
  not a separate unknown.
* **Gemma-4 family.** Not UNMEASURED — measured **NO**: transformers in the container
  rejects `gemma4_unified` and `gemma4` outright; blocked until the architecture
  registration seam exists. No Gemma-4 result may be reported as a framework pass.
* **Megatron / NeMo / Automodel engines.** Measured **absent or unusable** in the
  container (ModuleNotFoundError / PermissionError). The only engine is transformers +
  FSDP/DDP.
* **The enroot arm on this estate.** The runtime here is singularity; the resume-env
  cross-runtime defect (#122) is fixed in code, but no enroot run on this estate is
  recorded in these sources.

What **is** measured (so the list above is not mistaken for "nothing works"): the build
plane rebuilds deterministically with 17/17 stages green (E.5); the model-root resolver's
8/8 estate rows reproduce their independent measurements with the refusals firing on real
data (E.2); and the 8-rank NCCL collective passes with `NCCL_NET_PLUGIN=none`, with NVLS
and P2P/CUMEM still selected and both detector controls observed (E.3). Everything between
that green floor and a trained model is the list above.
