# LAUNCH.md — FoundationScale, H100 single-node launch plane (submit partition set at submit time by `$FS_PARTITION`)

> **Read this box before anything below it.** Phase 3 — the genuine end-to-end 8×H100 run
> (load → distributed → shard → train → save → resume → eval) — **has been executed**:
> job 37310, Qwen3-4B-Instruct-2507, 8/8 legs through this launch path, COMPLETED 00:06:02,
> resume PROVED. The full four-job submit chain has completed end to end with every link
> COMPLETED 0:0 (37340→37341→37342→37343), and a second model family ran the same argv on
> the same corpus (Gemma-3-1b, 37336) with one declared abstention rather than a pass. The
> build plane is green at 42 stages (E.5). **This document tells you how to launch, and the
> launch trains — on two small models, on one node, in one container runtime.** That
> denominator is the limit of the claim, and §8 is the list of everything outside it.
>
> *Until 2026-09-01 this box said Phase 3 had never been executed. It had — for a full
> campaign. A status line that lags its own evidence is a defect in the direction of
> understatement as much as overstatement (#194), and the fix is the same either way: the
> claim states its jobs.*

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
| *(no flag exists)* | `FS_PARTITION`, `IMAGE`, `OUT_DIR_STABLE`, `FS_ALLOCATION`, `FS_CONTAINER_RUNTIME`, `FS_ALLOWED_PATH_ROOTS`, `FS_ALLOWED_NODE`, `FS_ENGINE_LAUNCH_MODE`, `FS_ENGINE_LAUNCH_CMD`, `FS_NCCL_NET_PLUGIN`, `FS_CHECKPOINT_ADJUDICATORS`, `FS_FABRIC_TRIPWIRE` — these are **also required** and have no flag counterpart yet |

Engine: the **only** measured-available engine is HuggingFace transformers + torch
FSDP/DDP. `import megatron` → ModuleNotFoundError; `import nemo_automodel` →
PermissionError (measured 2026-08-31). Do not write a Megatron/NeMo launch command.

---

## 2. The 60-second version (Phase 3 probe)

Fill in the `<...>` placeholders, then run the last line from a login node.

**`<absolute path to plane dir>` means one flat directory** — the one holding
`fs_container_backend.bound.sh`, `launch_fs_h100.fixed.sh`, `fs_train.fixed.py` and
`fs_ckpt_adjudicator.py` side by side. In this repository that directory is `h100/gen/`;
once deployed it is wherever you copied `h100/gen/`'s **contents**, and the `h100/gen/`
prefix does not survive the copy. Every path below is `<plane dir>/<filename>` with
nothing in between — the launcher resolves its own backend as a sibling (fs142), so a
plane where the four files are not siblings does not start.

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
export FS_ENGINE_LAUNCH_CMD="<absolute path to plane dir>/fs_train.fixed.py \
  --model-path $MODEL_DIR --dataset-mode real --dataset-path $DATASET_DIR \
  --text-field <name of the jsonl field holding the training text> \
  --eval-count 8 --batch-size 1 --sequence-length 512 --learning-rate 1e-5 \
  --log-every 1 --seed 1234 --resume-tolerance 0.0005"
#   in torchrun mode the launcher composes --nproc_per_node from the MEASURED gpu count (fs124);
#   supply the inner engine command, not your own torchrun wrapper
#
#   NO INTERPRETER PREFIX. In torchrun mode fs_compose_launch prepends
#   `${FS_PYTHON:-python3} -m torch.distributed.run --nproc_per_node=... ` and this string
#   becomes torch.distributed.run's POSITIONAL argument -- which is the training script.
#   A leading `python3` therefore makes torchrun try to execute the python3 BINARY as a
#   python script. The prefix is the launcher's job precisely because the interpreter that
#   forks the ranks must be the same one that imports torch (fs128); supplying your own
#   splits that pair. The `wlm` and `self` arms pass the string through unchanged, where a
#   prefix is merely redundant -- so the shape that is correct on all three arms is the
#   bare script path.
#
#   EVERY FLAG ABOVE IS REQUIRED (#181). The trainer has no defaults: each knob is sourced
#   through a required-no-default contract, and a missing one is a ContractError -- raised
#   AFTER the allocation is granted, the container is up and the collective probe has passed.
#   That is the most expensive place in the pipeline to discover a typo, so the flag names
#   here are pinned against the trainer's own argparse declaration by gate_launch_doc.py
#   (rules L6-L11); this example cannot drift from the parser without the build going red.
#
#   Three further required knobs are NOT flags here because the launcher supplies them across
#   the container boundary on the allowlist: OUT_DIR (derived from OUT_DIR_STABLE, suffixed
#   `_probe` when PROBE=1), FS_ITERATION_BUDGET and FS_EARLY_SAVE_STEPS (defaulted to 20 and 5
#   by the PROBE arm; see the knob table in section 3 for the enforcing line). Do not pass
#   --iteration-budget on the command line: the flag beats the environment, so it would
#   silently defeat PROBE and run a production-length job.
#
#   --dataset-mode selects a branch and the two branches are mutually exclusive. `real` requires
#   --dataset-path and --text-field and REFUSES --synthetic-samples; `synthetic` requires
#   --synthetic-samples and refuses the other two. Supplying the wrong side is a ContractError,
#   not a warning.

# --- measured collective-plane fix (E.3): image default plugin SIGSEGVs in the first all_reduce ---
export FS_NCCL_NET_PLUGIN=none

# --- checkpoint adjudication: absence is not success (all([]) != PASS) ---
export FS_CHECKPOINT_ADJUDICATORS=<absolute path to plane dir>/fs_ckpt_adjudicator.py
#   each spec is containment-checked against FS_ALLOWED_PATH_ROOTS: outside every root it is REFUSED
#   (exit 96), not skipped (fs146a); each spec's dirname is bound into the container automatically (fs146b)
export FS_EXTRA_BIND_PATHS=<absolute path to plane dir>       # escape hatch for trees inference cannot see; see §7

# --- pre-launch fabric tripwire: REQUIRED, no default (fs163) ---
export FS_FABRIC_TRIPWIRE=<imex-master-host>:<port>|none
#   host:port is probed with a bounded connect before the launch and must fail CLOSED, never hang.
#   `none` is not "off" -- it is the estate DECLARING that it has no pre-launch fabric to probe,
#   which is the honest value on a cluster with no IMEX plane. An unconfigured guard is a
#   disabled standing rule, so there is no default and unset is exit 96 (backend line 325).

# --- submit the probe ---
sbatch --partition="$FS_PARTITION" --export=ALL <absolute path to plane dir>/launch_fs_h100.fixed.sh
#   --partition is REQUIRED on the command line and is not optional here: fs152 DELETED the
#   `#SBATCH --partition=` directive rather than parameterise it, because an #SBATCH line is a
#   shell comment and cannot expand a variable -- a directive reading `--partition=$FS_PARTITION`
#   submits to a partition literally named `$FS_PARTITION`. The launcher's own chain path spells
#   it the same way (launcher lines 585-591), so this is the framework's one submit idiom, not a
#   local workaround. Omit it and sbatch silently uses the cluster default partition.
```

For the full probe → production → resume → post-mortem chain instead, export
`FS_SUBMIT_CHAIN=1` and run the launcher **on the login node** (it submits the chain
itself). See §6.

Set and export the per-submit resource knobs before invoking `sbatch`: `FS_CPUS_PER_TASK`, `FS_MEM` and `FS_WALLTIME` are now required with no default, the same contract as `FS_PARTITION` and `FS_GPUS_PER_NODE`; unset ⇒ REFUSE 96. The shape measured on the estate this plane was validated against is `FS_CPUS_PER_TASK=96 FS_MEM=800G FS_WALLTIME=7-00:00:00` — quoted as an example of the FORM, not as a default. Nothing in the framework supplies these numbers. These facts no longer live in the header — an `#SBATCH` line is a shell comment and cannot expand a variable — so `--gpus-per-node`, `--cpus-per-task`, `--mem` and `--time` travel on the `sbatch` command line. `FS_WALLTIME` is a value knob: the launcher measures the named partition's maximum with `sinfo` at submit time, accepts a ten-day value on a ten-day partition, refuses it on a seven-day one, and quotes the measured maximum.

Do not set `FS_PLANE_DIR` unless you need to override plane resolution: the launcher
resolves the plane directory automatically (§3, bypass controls; §7). If you do set it,
set it to the directory containing `fs_container_backend.bound.sh` — the submit chain
re-submits the launcher through it, so a wrong override breaks every hop at once (§6).

---

## 3. Every knob, by category, from the polarity oracle (#127)

Categories are verbatim from the oracle. Citation legend: **L:\<n\>** = launcher line,
**B:\<n\>** = container-backend line. **Both notations have been re-anchored by
measurement**, and both are now enforced: gate rule L2 resolves every citation in this
document against the file its own notation names — 71 citations, 71 resolved — and a
notation it cannot map to a file is a counted miss, not a citation it quietly skips.
Two separate drifts made that necessary. On the launcher side, a self-contained
plane-directory resolver (fs142) took L:57–205 and pushed every later line down. On the
backend side, 8 of the 12 B: pointers had gone stale by consistent offsets (+9, +29, +64)
as the container backend grew; each was re-anchored to the line where its subject
actually occurs. Before that, L2's denominator was 59 — it read only the L: notation and
scored 59/59 while 12 B: pointers sat in no denominator at all, under a banner claiming
every citation had been read at its line. The oracle's measured universe is **41 distinct `FS_*`
names**; the tables below reproduce every name the oracle assigns explicitly. Names
present in code but not classified in the oracle's published tables are listed separately
at the end and are **not** silently bucketed.

This document is hand-written from the oracle tables: the Deliverable D generator
(`gate_launch_contract.py`) is red on L2 ("6 of 18") and correctly refuses to write — the
gate's count misses the names reached through locals and the array-length test.

### REQUIRED — unset or empty is fatal, and no default exists

| Name | What it is | If unset | Enforced |
|---|---|---|---|
| `FS_PARTITION` | this estate's Slurm submit partition. The `#SBATCH --partition=` directive was **deleted, not parameterised** (fs152): an `#SBATCH` line is a comment to the shell, so the partition travels as `--partition="$FS_PARTITION"` on the sbatch invocation, where expansion actually happens (L:12–16) | exit 96 refuse — "required, no default by design … the framework refuses to guess a cluster layout" (a default would be the deleted literal compiled back in) | L:39; carried on every chain `sbatch` at L:732–738 |
| `FS_ALLOCATION` | who allocated the nodes; must be exactly `slurm` on this launcher | exit 96 refuse (`${FS_ALLOCATION:-}` empty fallback preserves required-no-default under `set -u`, #126) | B:169; launcher prologue |
| `FS_ALLOWED_NODE` | node this plane is permitted to run on (standing safety rule) | exit 96 refuse | B:299 |
| `FS_CONTAINER_RUNTIME` | container runtime; must be exactly `singularity` | exit 96 refuse | B:154; launcher prologue |
| `FS_ALLOWED_PATH_ROOTS` | space/tab-separated absolute roots reachable in-container | exit 96 refuse — "no default by design … refuses to guess a filesystem layout" | L:269 |
| `FS_ENGINE_LAUNCH_MODE` | who forks ranks: `torchrun` / `wlm` / `self` | exit refuse, "required, no default" (read via local `mode`) | B:677; demanded by `fs_compose_launch` at L:920–921 |
| `FS_CONTAINER_SQSH` | container image path, operator-supplied (R7) | exit refuse | B:1336 |
| `FS_BIND_PATHS` | array of host paths bound into the container; **derived** by the launcher from `MODEL_DIR`, `DATASET_DIR`, `dirname(CONFIG_FILE)`, `OUT_DIR`, each adjudicator spec's dirname (fs146(b), L:359–363), and `FS_EXTRA_BIND_PATHS` | zero entries ⇒ exit 96 — "a derivation bug, not a legal empty set" | L:582 (array-length test); derivation loop L:564–573 |
| `FS_CHECKPOINT_ADJUDICATORS` | space/tab/newline-separated adjudicator commands, one per word; each invoked as `<cmd> <ckpt_dir> <phase> <out_dir>`. Every spec is containment-checked against `FS_ALLOWED_PATH_ROOTS` (fs146(a)): a spec outside every declared root is **REFUSED** (exit 96, naming the offending spec *and* the declared roots), never skipped — and each spec's dirname is then bound into the container (fs146(b)), because a refused-after-hours knob was finding #146 | exit 96 refuse — `all([])!=PASS` (read via local `ADJUDICATORS_RAW`) | L:315 (empty refuse), L:314–338 (zero specs); containment L:338–348 |
| `FS_CPUS_PER_TASK` | CPUs per task, carried to `sbatch` as `--cpus-per-task` because an `#SBATCH` line is a comment and cannot expand it | exit 96 refuse — required, no default | presence refusal before queueing; interpolated on the submit command line |
| `FS_MEM` | memory request, carried to `sbatch` as `--mem` for the same comment-expansion reason | exit 96 refuse — required, no default | presence refusal before queueing; interpolated on the submit command line |
| `FS_WALLTIME` | walltime request, carried to `sbatch` as `--time`; a value knob compared against the named partition's maximum **measured** by `sinfo` at submit time. A ten-day value is accepted on a ten-day partition and refused on a seven-day one; the refusal quotes the measured maximum, not a constant | exit 96 refuse — required, no default; over the measured maximum ⇒ 96 | presence refusal; live `sinfo` probe and comparison at submit time |
| `FS_ENGINE_LAUNCH_CMD` | complete in-container engine command | exit 96 refuse (read via local `LAUNCH_CMD`) | L:907 |
| `FS_NCCL_NET_PLUGIN` | NCCL net-plugin selection; measured correct value on this estate is **`none`** | collective probe refuses the launch; without the fix the 8-rank all_reduce **SIGSEGVs** inside the first collective (E.3, measured) | L:473, L:483, L:898 |
| `FS_FABRIC_TRIPWIRE` | pre-launch fabric probe, `host:port` or the sentinel `none` (fs163). The value is format-validated *before* it is interpolated into a `bash -c` string, and the connect is time-bounded so an unroutable master fails CLOSED rather than hanging. `none` is not "disabled": it is the estate **declaring** it has no pre-launch fabric to probe, which is the honest value where there is no IMEX plane — the sentinel exists so that "no tripwire" is a decision on the record instead of an omission | exit 96 refuse — "required, no default by design … the framework refuses to guess a cluster's fabric topology" | B:325 (presence), B:334 / B:338 (format and port range) |

Value note: `FS_WALLTIME` is now REQUIRED with no default (the `FS_PARTITION` / `FS_GPUS_PER_NODE` contract) and is compared against the partition maximum measured by `sinfo` at submit time; the hard-coded literal oracle and its run-first ordering are gone (§7).

### CONDITIONALLY REQUIRED — required only on one branch

| Name | What it is | If unset | Enforced |
|---|---|---|---|
| `FS_ENGINE_PROCS_PER_NODE` | ranks per node, **only** in `self` mode (the engine forks its own ranks and must declare `== gpus`) | in `self` mode: fatal; in `torchrun`/`wlm` mode: **must not be given** | B:715 |

### FORBIDDEN-IF-SET

| Name | What it is | If set | Enforced |
|---|---|---|---|
| `FS_NCCL_IB_HCA` | InfiniBand HCA pinning | exit 96 refuse — pinning is unmeasured on the partition named by `$FS_PARTITION`; leave unset unless measured and validated | L:451 |

The measurement narrowing this refusal now exists (#130); the table reports what the code
**does**, not what it should do.

### VALIDATED-IF-SET — value constrained, absence tolerated

| Name | What it is | If set to a bad value | Enforced |
|---|---|---|---|
| `FS_NCCL_SOCKET_IFNAME` | socket interface name | must resolve via `ip link show`, else 96. **Known defect (#131): validated then discarded — a knob with no reader** | L:448 |
| `FS_GPUS_PER_NODE` | GPUs per node | must be integer > 0 (L:367–368). Agreement with `SLURM_GPUS_PER_NODE` is checked only when Slurm exports it; when Slurm does not, the launcher prints an explicit UNMEASURED line and defers to the in-container `torch.cuda.device_count()` count, which is the real binding check | L:367–368; `req_env` presence refusal — set it (§2); the absent-`SLURM_GPUS_PER_NODE` case is UNMEASURED, not default-agreement |
| `FS_ITERATION_BUDGET` | positive integer step budget | must be positive int; must exceed `FS_EARLY_SAVE_STEPS` | L:523, L:525 |
| `FS_EARLY_SAVE_STEPS` | steps before early save | must be positive int and `<` the budget, "an early save that cannot fire is not evidence" | L:524–525 |
| `FS_BACKEND` | legacy selector | restricted to `slurm-singularity`/`singularity`; has a real default (`slurm-singularity`) | L:241–242 |

### ENVIRONMENT — minted by the script, never an operator input

| Name | What it is | If you set it | Enforced |
|---|---|---|---|
| `FS_ACTUAL_HOST` | actual hostname, minted as `$(hostname -s)` | the script **overwrites** your value | B:366 |

### HAS-A-DEFAULT — cannot be REQUIRED

| Name | Default |
|---|---|
| `FS_BACKEND` | `slurm-singularity` (L:241) |
| `FS_USE_TORCHRUN` | `0`/`1` (B:344, B:423) |
| `FS_EARLY_SAVE_STEPS`, `FS_ITERATION_BUDGET` | on the resume path, default through `FS_RESUME_*` (L:781–782); in probe phase, `5` and `20` with source logged (launcher probe block, L:521–522) |

### Bypass controls and launcher-observed names not in the oracle's published buckets

Listed, not silently classified. The oracle is emphatic about the first one:

| Name | What it is |
|---|---|
| `FS_SUBMIT_CHAIN` | `=1` on a login node runs the four-job submit chain (§6). **It must NOT be classified REQUIRED.** The L:800 guard is on `SLURM_JOB_ID`; `FS_SUBMIT_CHAIN` appears only inside its quoted refusal message — it is what you set to *bypass* the check. A proximity read of that line inverts the knob's meaning. |
| `FS_PLANE_DIR` | **OPTIONAL** operator override naming the plane directory. Resolution is **automatic** and ordered (fs142 resolver, L:57–205): step 1 verifies that `$FS_PLANE_DIR`, if set, contains a readable `fs_container_backend.bound.sh` (L:108–124); step 2 falls back to `SCRIPT_DIR` so a direct `bash launch…` invocation keeps working with no new operator variable (L:129–141); step 3 asks the workload manager for the submitted script's original path (`scontrol show job` on slurm, L:82–103) and **verifies the sibling backend there** rather than trusting the claim (L:143–171); step 4 refuses with `FATAL[142]`, printing all three attempted answers and the last directory searched (L:177–186). The verified answer is exported so later jobs and child processes inherit it (L:201). **Warning:** the submit chain re-submits the launcher *through* this variable — every one of the four `sbatch` calls addresses `"$FS_PLANE_DIR/$(basename "$0")"` (L:732–738) — so overriding it wrongly breaks the entire probe → production → resume → post-mortem chain, not one job. Only set it to override a mis-resolution. |
| `FS_EXTRA_BIND_PATHS` | optional space-separated escape hatch for paths the launcher cannot infer (e.g., a code tree, a scratch root); word-split on purpose (launcher fs117 block, L:560–566). Since fs146(b) the adjudicator dirnames join the bind plane automatically, so pointing this at the adjudicator's tree is now redundant but harmless (de-duplicated, L:560–572) |
| `FS_PHASE` | `train` (default) / `resume` / `post-mortem`, set by the chain driver on the resume and post-mortem hops (launcher chain block, L:736, L:738) |

### 3b. The trainer's own knobs (#170, #181)

Everything above is a *launcher* knob, read on the host before the container starts. The
table below is the other half — the knobs the in-container entrypoint
`h100/gen/fs_train.fixed.py` reads out of the `FS_ENGINE_LAUNCH_CMD` you compose in §2.

These are worth a table of their own for one reason: **the trainer has no defaults.** Every
row is required-no-default, and an absent one raises `ContractError` — *inside* the
container, *after* the allocation is granted, the image is up and the collective probe has
passed. The launcher's guards fail in seconds on a login node; these fail on a scheduled
node minutes in. So the cost of learning a knob's name from its refusal message is not the
same on the two halves of this plane, and this table exists to make the second half cheap.

Both the flag names and the required/optional classification below are **derived from the
parser and its sourcing calls**, not transcribed, and `gate_launch_doc.py` L6–L11 re-derives
them on every build: if a flag is renamed, made optional or moved between dataset-mode
branches and this table is not updated with it, the build goes red.

| flag | type | required | supplied by | applies |
|---|---|---|---|---|
| `--model-path` | str | yes | flag only | always |
| `--dataset-mode` | str | yes | flag only | always — selects the branch below |
| `--eval-count` | int | yes | flag only | always |
| `--batch-size` | int | yes | flag only | always |
| `--sequence-length` | int | yes | flag only | always |
| `--learning-rate` | float | yes | flag only | always |
| `--log-every` | int | yes | flag only | always (refused if it exceeds the iteration budget) |
| `--seed` | int | yes | flag only | always |
| `--resume-tolerance` | float | yes | flag only | always — **restore fidelity only** (#192): the max over ranks of \|own after-resume loss − own before-save loss\|. Exceeding it is RED and final. |
| `--rank-agreement-tolerance` | float | no | flag only | always — the **cross-rank** knob (#192), and a separate question from restore: do the ranks compute the same fixed-eval loss? Leave it unset and the proof reports the before-save spread, the after-resume spread and their signed delta, and declares the absolute claim UNMEASURED; a floor calibrated from the run it judges may never mint `rank_invariant`. Set it only when you have measured what your instrument can resolve. |
| `--iteration-budget` | int | yes | flag, else `$FS_ITERATION_BUDGET` | always — **leave it to the env**, or you defeat `PROBE` |
| `--early-save-steps` | int | yes | flag, else `$FS_EARLY_SAVE_STEPS` | always — same |
| `--output-dir` | str | yes | flag, else `$OUT_DIR` | always — the launcher exports `OUT_DIR` across the allowlist |
| `--dataset-path` | str | yes | flag only | `--dataset-mode real` only; **refused** under `synthetic` |
| `--text-field` | str | yes | flag only | `--dataset-mode real` only; **refused** under `synthetic` |
| `--synthetic-samples` | int | yes | flag only | `--dataset-mode synthetic` only; **refused** under `real` |

Two further flags are declared but never sourced, because they are not run configuration:
`--selftest` runs the entrypoint's own certification table and exits, and `--probe` is a
diagnostic. Neither belongs in a production `FS_ENGINE_LAUNCH_CMD`.

The last three rows are a genuine exclusivity contract, not a convention: supplying a flag
from the branch you did not select is a `ContractError`, exactly as an omission is. The
trainer refuses to guess which of two dataset descriptions you meant.

---

## 4. The artifact map

Every file under `h100/gen/` is **GENERATED** by `build_h100_plane.sh` (Deliverable C,
42 stages green, byte-identical across rebuilds). **Hand-editing any of them is
prohibited**: a hand edit puts the fix in the file you read and leaves it out of the file
that runs. Change the stage, rebuild.

| Artifact | What it is | Reads / read by |
|---|---|---|
| `h100/gen/launch_fs_h100.fixed.sh` | the sbatch-submitting launcher (~800 lines; the fs142 plane-directory resolver occupies L:57–205 at the top) | resolves the plane directory (operator override → `SCRIPT_DIR` → workload-manager `Command=` path, refusing `FATAL[142]` if none verify); sources the backend from the resolved directory; derives `FS_BIND_PATHS`; invokes adjudicator entries; composes the launch command |
| `h100/gen/fs_container_backend.bound.sh` | container-runtime backend (`run_in_container`, bind materialization, guards) | sourced by the launcher (from `$FS_PLANE_DIR`, L:202–204); consumes `FS_BIND_PATHS`, `FS_ALLOWED_NODE`, `FS_CONTAINER_RUNTIME`, `FS_ENGINE_LAUNCH_MODE`; built from the spliced base below |
| `h100/gen/fs_container_backend.spliced.sh` | **intermediate**: backend base text from the upstream repo (#136) | read by the backend stages; removed and rebuilt every run; not shipped |
| `h100/gen/fs_train.fixed.py` | in-container training entrypoint | runs inside the container via `run_in_container`; reads `FS_ITERATION_BUDGET`/`FS_EARLY_SAVE_STEPS` across the allowlisted boundary; imports the resolver below |
| `h100/gen/fs_model_root.py` | model-root resolver (#133): config searched per-depth, shallowest populated depth wins, bind closure not declared root | imported by the training entrypoint via the stage-C binding (`load_artifacts` → `resolve_model_root`); before that binding it was an orphan — 12 green tests, zero callers |
| `h100/gen/test_fs_model_root.py` | generated suite for the resolver | run **by the build**; a suite nobody runs is an orphan (#86) |
| `h100/gen/fs_ckpt_adjudicator.py` | the checkpoint adjudicator the launcher's required knob asks for (#141) | invoked per checkpoint dir as `<cmd> <ckpt_dir> <phase> <out_dir>`; its dirname is bound **inside** the container automatically by fs146(b) (see §3, §7) |
| `h100/gen/test_fs_ckpt_adjudicator.py` | generated suite for the adjudicator | run by the build |

Logs: `$OUT_DIR/logs/launch.<jobid>.log` (or `launch.interactive.log`), teed from
`BEGIN` to `END`; checkpoint evidence lands in the `ADJUDICATORS observed=… seen=… ok=…`
and `END … checkpoint_saves_adjudicated=N` lines — the N is the denominator.

**The provenance record (#180).** Every launch that actually starts a trainer writes
`$OUT_DIR/logs/launch.<jobid>.provenance.json` — the same name as its log with `.log`
replaced, so the pair is derived once and cannot drift — and announces it on the log as
`PROVENANCE path=… write=ok redactions=N`. It holds `launch_cmd_composed` (what ran,
verbatim), `launch_cmd_raw` (what the operator supplied, before the composer prepended the
torchrun wrapper), the world size and its source, the four directory knobs, and `fs_env`:
every `FS_` name this shell held, with `fs_env_not_exported` naming the ones that were set
but not exported. It exists because job 37319's own 936-line log named the launch command
zero times, `--resume-tolerance` zero times, and `--model-path`/`--dataset-path`/
`--sequence-length` zero times combined: the run recorded everything about itself except
what it ran, and the trainer's resume knob is flag-only, so a value that can arrive only on
a command line was held by no artifact.

Three things to know before you rely on it:

* **`redactions` is the record's own honesty field.** Values of flags and names matching
  key/token/secret/password/credential are replaced with `<redacted>` and counted.
  `redactions=0` means the record is exact and can be replayed as-is; `redactions>0` means
  that many values must be re-supplied by hand.
* **A failed write is announced, not silent.** If the record cannot be written the launch
  continues and the log says `write=FAILED`. The run is degraded, not failed — but it is
  no longer self-documenting, and nothing downstream will tell you a second time.
* **The post-mortem link writes no record, deliberately.** It launches no trainer
  (`FS_SKIP_TRAIN=1`), so a record there would assert a command that never ran. Its absence
  on that one arm is the correct state, and that arm prints its own `POST-MORTEM:` line.

---

## 5. The exit-code contract

| rc | Meaning | Operator response |
|---|---|---|
| **0** | Launch succeeded **and** ≥1 checkpoint dir was adjudicated (`checkpoint_saves_adjudicated=N`, N>0) | Read the denominators in the log. A 0 with no adjudicated save is impossible — the script refuses it |
| **95** | **UNMEASURED**: training ran but **no checkpoint-save units were observed** (`adjudicate_tree` found no checkpoint dirs, or zero observed after training). **This is not failure.** `all([])` is True, so zero units measured can never be reported as PASS; the plane reports the abstention instead | **Do not retry blindly.** A blind retry reproduces the same nothing-to-measure. Find out why nothing was saved (budget too small? engine wrote elsewhere? bind hole?) before resubmitting |
| **96** | **REFUSE**: a guard fired — missing/empty required var (including `FS_PARTITION`, whose first refusal is what an operator following an outdated runbook meets on their first submit), path outside allowed roots, adjudicator outside allowed roots, bad value, walltime conflict, forbidden knob set | Fix the configuration named in the `FATAL[96]` message and resubmit. The refusal is the contract working, not a crash; do not bypass it by emptying the guard |
| **124** | `fs_compose_launch` refused the launch topology (mode/gpu-count mismatch, missing mode, `self` mode with no declared procs) | Fix `FS_ENGINE_LAUNCH_MODE` / `FS_ENGINE_PROCS_PER_NODE`; see the refusal text |

The in-container training entrypoint (`h100/gen/fs_train.fixed.py`) has its own contract,
in a **different namespace** from the launcher's — four codes, not the launcher's 0/95/96:

| code | meaning inside the container |
|---|---|
| **0** | the run reached a `MEASURED` verdict (or, under `--selftest`, every table row matched) |
| **1** | `--selftest` only: a self-test row did not match. Never produced by a training run |
| **2** | **the argv was refused.** Either a `ContractError` — a required knob absent, or knobs from both dataset-mode branches supplied — emitting `RUN_SUMMARY_JSON` with `verdict=UNMEASURED, reason=contract_refused`; or `argparse` itself rejecting an unrecognised flag, which also exits 2 but prints a bare usage line and **no JSON at all**. The absence of `RUN_SUMMARY_JSON` is what distinguishes the two |
| **3** | training ran and did not reach `MEASURED`: an `OperationFailure`, emitting `PHASE_JSON` with the failed phase and metric, then `RUN_SUMMARY_JSON` with `verdict=UNMEASURED` |

Only rank 0 prints; the markers are `PHASE_JSON` and `RUN_SUMMARY_JSON`, both emitted by
`_print_json` behind a rank-0 guard. **2 is the code a mistyped `FS_ENGINE_LAUNCH_CMD`
produces**, and it arrives after the allocation is granted — which is why the flag names in
§2 are gated against the parser rather than merely proofread.

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
submit partition explicitly as `--partition="$FS_PARTITION"` (L:732–738), so unset
`FS_PARTITION` is refused before the first job id exists. And every hop re-submits the
launcher **through the resolved plane directory**, as `"$FS_PLANE_DIR/$(basename "$0")"`
(L:732–738): the resolver's answer (or your override) is inherited by all four jobs, so a
wrong `FS_PLANE_DIR` override breaks the probe → production → resume → post-mortem chain
uniformly, not at a single hop.

### The login-node argv preflight (#183)

Before the chain driver fires its first `sbatch`, the launcher runs `fs_argv_preflight.py` against the operator-supplied `FS_ENGINE_LAUNCH_CMD`. The checker is host-side and torch-free, so it runs under the login node's system Python (3.6.8). It reads the command, the backend the plane ships, and the GPU count, and adjudicates six checks: the command tokenizes, the entrypoint it names is readable, every flag it passes is declared by that entrypoint's parser, the launch mode is one the backend accepts, the mode and the process count are consistent, and nothing required is absent. The accepted modes are derived from the backend's own `case "$mode" in ...)` alternatives at check time, not from a typed list, so the checker cannot silently disagree with the backend the day someone adds a mode.

The checker speaks the plane's 0 / 5 / 95 / 96 contract, and the launcher maps each outcome:

| Exit | What it means | What you do |
|------|---------------|-------------|
| `0` | The command passed all six checks. | Nothing. The chain submits silently. |
| `5` RED | The command names flags the entrypoint does not declare, or a mode the backend does not accept. | Fix `FS_ENGINE_LAUNCH_CMD` against the entrypoint's parser and resubmit. Nothing was queued. |
| `96` REFUSE | `FS_ENGINE_LAUNCH_CMD` is unset or does not tokenize. | Set it, or correct it, and resubmit. Nothing was queued. |
| `95` UNMEASURED | An oracle could not be read -- for instance an entrypoint that exists only inside the container. | Read the warning. The launcher proceeds to submit, and the warning is not a clean bill of health. |

The 95 asymmetry is deliberate: a legitimate engine whose entrypoint is readable only inside the container must not be blocked by its own diagnostic. The same reasoning covers the checker itself -- if `fs_argv_preflight.py` is not readable in the plane directory, as in a plane staged before this stage existed, the launcher prints an UNMEASURED warning that names its remedy (redeploy the plane directory from the build, which ships the file alongside the backend) and proceeds. Any exit code outside the contract is refused as undeclared.

The preflight runs on the submit-chain path only -- the login-node branch taken when `FS_SUBMIT_CHAIN=1`. Direct single-job submission (`sbatch`, section 2) does not pass through it, so for that form the original finding stands: the argv is validated after the allocation is granted. No cluster job has yet run the spliced block, so its on-hardware behaviour is UNMEASURED; what is certified is the build-time controls.

---

## 7. Troubleshooting — symptom → cause → fix

| Symptom | Cause | Fix | Status |
|---|---|---|---|
| `FATAL[142]: … remedy: set and export FS_PLANE_DIR=<directory containing fs_container_backend.bound.sh>` (older builds: `FATAL: fs_container_backend.sh not readable at /var/spool/slurmd/job<N>/...`) | `sbatch` stages **and renames** the script to `slurm_script`, so the directory beside `BASH_SOURCE` is the spool directory holding no siblings; `SLURM_SUBMIT_DIR` was only the submit-time CWD and is deliberately not trusted | #142 resolver (L:57–205): try the operator override `FS_PLANE_DIR`, then `SCRIPT_DIR`, then the workload-manager-recorded `Command=` path — each candidate accepted only if the backend is **readable beside it**; if all three fail, the refusal prints every attempt and the last directory searched. The remedy `export FS_PLANE_DIR=<plane dir>` still works and is now *step 1* of the resolver; the chain inherits the winning answer (§6) | **FIXED** (#142; the defect itself was measured on a real staged 8×H100 job and the resolver ships in the launcher header) |
| Refusal naming your `FS_ALLOWED_PATH_ROOTS` **value** when the value was right | the variable documented itself "space-separated" while the script's safety `IFS=$'\n\t'` made spaces not split — a two-root value read as one impossible root. Prose and parser disagreed | #139: function-local `IFS=$' \t\n'` at the split sites, prose and parser now agree in both directions; `patch_list_separators.py` is **last** of the launcher-touching stages so no later stage can reintroduce a split site | **FIXED** (#139; visible in the build stage list and both launcher slices) |
| Adjudicator exists on the host, fails inside the container | historically the bind plane derived from `MODEL_DIR`, `DATASET_DIR`, `dirname(CONFIG_FILE)`, `OUT_DIR` **only**; an adjudicator outside those trees wasn't mounted, and the knob *itself* was unchecked at startup (#146) | #146 (a): every spec is containment-checked against `FS_ALLOWED_PATH_ROOTS` at startup and refused (exit 96) if outside every root — refused, not skipped (L:338–348); (b): each spec's dirname now joins the bind-plane derivation automatically (L:359–363, consumed at L:564). `FS_EXTRA_BIND_PATHS` remains the escape hatch for trees the launcher cannot infer (a code tree, a scratch root) | **FIXED** (#146) for adjudicators under declared roots; the escape hatch still covers what inference cannot see |
| `sbatch: error: ERROR: The number of GPUs is not specified.` | the partition named by `$FS_PARTITION` rejects a job with no `--gpus-per-node` | the prologue no longer carries a static directive (an `#SBATCH` line is a comment); `--gpus-per-node="$FS_GPUS_PER_NODE"` travels on every `sbatch` invocation, and unset `FS_GPUS_PER_NODE` is refused 96 before anything is queued | **FIXED** (estate shape moved from a header literal to a required command-line knob) |
| `FATAL[96]: FS_WALLTIME='…' (…s) exceeds the measured <partition> partition max '…' (…s); refusing instead of clamping` | the value knob is compared against the named partition's maximum measured by `sinfo` at submit time; the message quotes that measured maximum, so a ten-day value is accepted on a ten-day partition and refused only where the measured maximum is shorter | set `FS_WALLTIME` within the measured maximum for `$FS_PARTITION`; shorter and longer values are judged by the live probe alone | **FIXED** (#153; one oracle, the measured one) |
| `FATAL[5]` argv-preflight refusal on a login node, with no job id printed | The named flag is not declared by the entrypoint the command names, or the mode is not one the backend accepts; the refusal happens before anything is queued. | Correct the flag against that entrypoint's parser, or choose a mode from the backend's accepted `case "$mode" in ...)` set, then rerun the submit chain. | **FIXED** (#183; submit-chain path only -- a direct single-job `sbatch` submission is still uncovered). |

### Closed limitation (#153): the walltime guard now has one measured oracle

Stated plainly: maximum walltime is asserted by a single oracle. The compiled-in literal that ran before the probe is deleted, the four node-shape directives are out of the header because an `#SBATCH` line is a shell comment, and `FS_WALLTIME` is required and compared only against `sinfo -h -p "$FS_PARTITION" -o '%l'` measured at submit time.

* **Former oracle 1 — deleted.** The compiled-in `7-00:00:00` literal that refused
  legitimately shorter values before any probe is gone, and with it the claim of a
  maximum for a partition never probed. A ten-day request is now accepted on a ten-day
  partition and refused on a seven-day one, with the measured figure quoted.
* **The surviving oracle — a live probe (L:402–411; pre-submit comparison L:413–421;
  post-submit proof L:423–435).** The launcher asks `sinfo` for the real `MaxTime` of
  whatever partition `$FS_PARTITION` actually names, parses it (handling `UNLIMITED`
  explicitly rather than guessing), and *proves* both the requested `FS_WALLTIME` and
  the scheduler's own reported `TimeLimit` fit — refusing when it cannot prove it.

What changes for you: a short walltime is now a legal request. `FS_WALLTIME=0-02:00:00`
for a cheap smoke run is admitted, because the only question asked is whether the value
fits the measured maximum, and two hours fits every partition this plane can name.
The behaviour is still fail-closed — nothing is clamped and nothing is silently
admitted — but it is now fail-closed against a *measurement* instead of against a
constant, so the set of refused values is the set the scheduler would refuse.

One thing did get stricter, deliberately: `FS_WALLTIME` is required with no default
(§2). There is no longer a compiled-in value to fall back to, because a default here is
one estate's maximum smuggled back into a framework meant to run on several. On a
partition whose maximum is `UNLIMITED` the launcher prints a NOTICE and admits any
finite request; an infinite request is refused as not-a-finite-duration.

---

## 8. What is UNMEASURED

Blunt list. Each item has never run on real hardware in a way these sources record.
Three items that used to head this list have been struck, because they ran:

* ~~**Phase 3, all of it.**~~ **MEASURED.** Job 37310 executed 8/8 legs — load, distributed
  init, FSDP sharding, training steps, DCP save, fresh-process resume, deterministic eval —
  through this launch path with no abstentions. Weights are materialised, not meta-device.
* ~~**The four-job chain end-to-end.**~~ **MEASURED.** 37340 → 37341 → 37342 → 37343,
  qwen3-0.6b, every link COMPLETED 0:0. The resume leg (hop 3) is exact on the measured
  control: `restore_delta 0.0`, 0 of 513 optimizer-state mismatches (37342).
* ~~**The in-container training entrypoint itself**~~ (`h100/gen/fs_train.fixed.py`).
  **MEASURED** as the program every job above ran. Two revisions since then are
  build-certified and not yet exercised on hardware — the split resume tolerance (#192,
  stage 35) and the provenance record (#180, stage 36); see the residual bullet below.
  (Until #181 this bullet named `fs_phase3_train.py`, a script the build produces nowhere —
  the Phase 3 specification's name for the program, carried into a document that ships a
  different one.)
* **Two build-certified fixes that no job has yet exercised.** Stage 35 splits
  `--resume-tolerance` (restore fidelity) from the new optional
  `--rank-agreement-tolerance` (cross-rank agreement); stage 36 makes the launcher write
  `logs/launch.<jobid>.provenance.json` before it execs. Both are green on static gates and
  executed controls at build time. Neither has run on a cluster. The next job on either arm
  discharges both, and until it does, this document's §4 provenance paragraph describes a
  file no run has produced.
* **Anything beyond two small models, one node, one container runtime, one dataset.** The
  executed denominator is Qwen3-4B-Instruct-2507, Qwen3-0.6B and Gemma-3-1b-it. Larger
  declared models (4B/7B/8B/27B rows in E.1.3) have no load/train/checkpoint/resume/eval
  measurement here.
* **The mechanism of the Gemma cross-rank divergence.** Job 37336 abstained on
  `resume.fixed_eval_rank_invariance` at `--resume-tolerance 0.0005` — an abstention, not a
  pass and not a failure. The identical code path produced exact rank agreement on the Qwen
  control at the same threshold (37342), so the divergence is attributable to the model; the
  Gemma checkpoint was a third-party mirror, so canonical-weight equivalence is UNMEASURED
  and that caveat weakens the attribution.
* **Multi-node.** One node only exists on this estate; there is no working multi-node
  launch and none should be attempted or written.
* **`FS_NCCL_IB_HCA` pinning on the partition named by `$FS_PARTITION`.** Unmeasured; the
  code forbids setting it (#130 has produced the measurement but the narrowing is not in
  the sources here).
* **`FS_NCCL_SOCKET_IFNAME`'s effect.** The value is validated and then **discarded**
  (#131) — setting it changes nothing downstream.
* **The flag-form CLI** (`--model --dataset --num-gpus --config`). **Not implemented.**
* **`FS_PLANE_DIR` resolver fallbacks on a full chain.** No longer "remedy named,
  implementation missing": the fs142 resolver ships in the launcher header (L:57–205,
  override at L:108–124, export at L:201), and the staging defect it fixes was itself
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

What **is** measured (so the list above is read at its true size): the build plane is green
at 42 stages (E.5); the model-root resolver's 8/8 estate rows reproduce their independent
measurements with the refusals firing on real data (E.2); the 8-rank NCCL collective passes
with `NCCL_NET_PLUGIN=none`, with NVLS and P2P/CUMEM still selected and both detector
controls observed (E.3); Phase 3 ran 8/8 legs (37310); the four-job chain completed end to
end (37340–37343); resume is exact on the measured control (37342); real checkpoints
adjudicate 3 of 3 under a walker whose denominator can no longer truncate silently (37345);
and the post-mortem link reports over a denominator it actually walked (37348,
`adjudicated=2 of 2 ... ok=2`). Everything between that floor and a *generalizable*
foundation-model training framework is the list above — and the gap is a denominator
problem, not a green-light problem.
