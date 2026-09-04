# D10 — Validation protocol and status

This document states what has been **observed to run**, on which hardware, recorded in
which artifact — and nothing more. Where no execution record exists in this repository,
the state is **UNMEASURED**. "Validated" is used only where a job id or a named artifact
backs it.

Scope note on size, so denominators in this document are anchored: the repository holds
121955 git-tracked .py/.sh/.md lines repo-wide; the validation harness, `validation_campaigns/h100_validation/`
(31313 .py LOC, 63 files); the package, `src/foundationscale/` measures 18915 lines; the
GB200-side launch path, `launchers/*.sh` (10171 LOC, 5 files). No other countables appear
in this document.

---

## PART A — What is proven today

### A.1 H100 (single node, 8× H100 SXM, Slurm, singularity)

**What has actually been executed** (all job ids and artifacts below are recorded in
`validation_campaigns/h100_validation/h100/EVIDENCE.md`, `validation_campaigns/h100_validation/h100/DELIVERABLE_B_validation_report.md`,
`validation_campaigns/h100_validation/h100/DELIVERABLE_E_matrix.md`, and
`validation_campaigns/h100_validation/h100/LAUNCH.md`):

1. **Phase 3 end-to-end run, 8/8 legs, no abstentions.** Job **37310**,
   Qwen3-4B-Instruct-2507, load → distributed init → FSDP sharding → training → DCP save
   → fresh-process resume → deterministic eval. COMPLETED 00:06:02. Resume status PROVED.
   Weights materialised, not meta-device.

2. **The full four-job submit chain, every link COMPLETED 0:0.** Jobs
   **37340 → 37341 → 37342 → 37343**, qwen3-0.6b: probe → production → resume →
   post-mortem. The resume leg is exact on the measured control: `restore_delta 0.0`,
   0 of 513 optimizer-state mismatches (job 37342).

3. **A second model family on the same argv and corpus, with a declared abstention.**
   Job **37336**, Gemma-3-1b-it, abstained on the `resume.fixed_eval_rank_invariance`
   leg at tolerance 0.0005. The identical code path produced exact rank agreement
   (spreads 0.0) on the Qwen control at the same threshold (37342), so the divergence is
   attributable to the model — **with an explicit caveat**: the Gemma checkpoint was a
   third-party mirror, not canonical weights, which weakens the attribution. This is an
   abstention on record, not a pass and not a failure.

4. **Fix re-measurements after the baseline.** Jobs 37344, 37345, 37346 re-measured
   repaired reporting controls (#187, #188, #189); 37347/37348 measured the #193
   composed regression on both sides and closed it (`adjudicated=2 of 2 ... ok=2` on
   37348). Jobs 37369/37370 are the hardware payment for the #192 resume-tolerance
   split: cross_rank_spread_delta populated,
   `rank_agreement_tolerance_source: self-calibrated`, restore verdict PROVED.

5. **Environment facts.** `EVIDENCE.md` M1–M2: one visible node, 8×
   `gpu:H100:8`, 112 CPUs, 2 TB RAM; singularity PRESENT; apptainer and enroot ABSENT on
   this estate. Partition time cap measured at 7 days
   (`DELIVERABLE_E_matrix.md` header; LAUNCH.md §7 describes the live `sinfo` oracle).

6. **The collective plane.** The 8-rank NCCL all_reduce passes **only** with
   `FS_NCCL_NET_PLUGIN=none`; the image's bundled HPC-X plugin SIGSEGVs inside the first
   collective otherwise (measured; recorded in `patch_collective_probe.py`'s stage
   docstring and E.3). The `--nv` singularity flag is load-bearing (job 37284 died
   without it: NVML unreachable, recorded in `patch_nv_runtime.py`).

7. **The build plane.** `validation_campaigns/h100_validation/build_h100_plane.sh` is green at 42 stages
   (per LAUNCH.md's lead box and §4), rebuilding `h100/gen/` from scratch on every
   invocation. This is a **build-time** certification; see the UNMEASURED list for what
   it does not cover.

**Un-measu

red on H100, stated as such:**

- **Multi-node.** Never run. One measured node only. LAUNCH.md §8: no working multi-node
  launch exists and none should be written from this record.
- **MoE and additional architecture families.** UNMEASURED. Executed denominator is two
  small Qwen models plus one mirrored Gemma.
- **Canonical-weight Gemma equivalence.** UNMEASURED (§3.4 caveat).
- **The enroot arm on this estate.** The runtime is absent here (M2). The #122
  cross-runtime resume-env defect is fixed **in code**; no enroot execution on the H100
  estate is recorded anywhere in these sources.
- **The provenance record (#180, stage 36).** Build-certified only. No cluster job has
  yet written `launch.<jobid>.provenance.json`; LAUNCH.md §4 describes a file no run has
  produced. The hardware observation is owed, on the record
  (`DELIVERABLE_B_validation_report.md` §1).
- **Stage 35 (split resume tolerance) on the Qwen/Gemma arms beyond 37369/37370's PROBE
  scope** — build-certified; broader exercise UNMEASURED.
- **`#204` cross-hardware closure.** Verified by build gates (no hard-coded node shape
  remains); **no submit on a second node geometry has ever exercised it**.
- **Megatron / NeMo / Automodel engines.** Measured **absent** in the container
  (ModuleNotFoundError / PermissionError). This is a measured negative, not a gap.
- **Gemma-4 family.** Measured **NO** — the container's transformers rejects the
  architecture. Also a measured negative.

### A.2 GB200 (E4B 1-tray, enroot)

**Nothing in this repository records a validation run on GB200 hardware.** Stated
plainly because the code layout invites the confusion:

- The GB200-side path exists as code: `launchers/launch_g4e4b_fullft_1tray.sh`,
  `launchers/launch_g4e4b_lora_1tray.sh`, and the shared
  `launchers/fs_container_backend.sh`, whose own header records measured estate facts
  for the tray (sbatch/srun gone; enroot the only executable path). Those facts are
  recorded **in comments**, not in an evidence ledger equivalent to EVIDENCE.md.
- Contract-test shells exist (`launchers/test_launcher_contracts.sh`,
  `launchers/test_fs_live_gate_watchdog_contracts.sh`). No execution record of their
  output is in the evidence I was given. UNMEASURED means exactly this: not observed.
- GB200-shaped constants caused measured H100-side defects when they leaked across
  (the `SLURM_NTASKS=4` mint in #125, identified as "a GB200 tray fact"; the stale
  GB200 walltime guard removed by `patch_node_shape.py`). These are defect records,
  not GB200 validity evidence.

**GB200 validation status: UNMEASURED.** Every claim about the GB200 path — that it
launches, trains, checkpoints, resumes — has no recorded denominator of executions.
No honest statement stronger than "the code exists and embeds measured estate facts in
comments" is available from these sources.

---

## PART B — The protocol

Ordered, reproducible procedures to **re-establish** each claim. Each step names its
exit condition using the framework's contract: **0** PASS, **5** RED, **95** UNMEASURED,
**96** REFUSE (launcher/backend namespace). The trainer
(`validation_campaigns/h100_validation/h100/gen/fs_train.fixed.py`) uses a **different** namespace —
0 MEASURED, 1 selftest mismatch, 2 argv refused, 3 ran-but-not-MEASURED — do not
conflate them.

### B.1 H100

**Step H0 — rebuild the plane (login node, no allocation).**
Source the estate environment, then run `validation_campaigns/h100_validation/build_h100_plane.sh`.
*Exit condition:* script exits 0 with all stages green and the three standing checks
(`gate_env_drift.py`, estate blocklist, `bash -n` on both artifacts) passing. Any red
stage leaves the tree at the last good state; do not proceed on a partial build.

**Step H1 — verify publishability and membership.**
Confirm `gate_stage_orphans.py`, `gate_build_inputs.py`, `gate_artifact_linkage.py`,
`gate_exit_contract.py`, `gate_doc_stage_count.py` all report green over
`h100/PUBLISH_SET.txt`.
*Exit condition:* each gate exits 0 over its printed denominator. A gate that prints no
denominator has measured nothing; treat as UNMEASURED.

**Step H2 — set required env (LAUNCH.md §2).** `FS_PARTITION`, `FS_ALLOCATION=slurm`,
`FS_CONTAINER_RUNTIME=singularity`, `FS_ALLOWED_NODE`, `FS_ALLOWED_PATH_ROOTS`,
`MODEL_DIR`, `DATASET_DIR`, `CONFIG_FILE`, `FS_GPUS_PER_NODE`, `IMAGE`,
`OUT_DIR_STABLE`, `FS_ENGINE_LAUNCH_MODE`, `FS_ENGINE_LAUNCH_CMD`,
`FS_NCCL_NET_PLUGIN=none`, `FS_CHECKPOINT_ADJUDICATORS`, `FS_FABRIC_TRIPWIRE`,
`FS_CPUS_PER_TASK`, `FS_MEM`, `FS_WALLTIME`. All required, no defaults.
*Exit condition:* submitting with any unset produces refusal 96 naming the knob —
observing one such refusal deliberately is a legitimate probe of the guard.

**Step H3 — single-job probe.** `export PROBE=1`; submit
`sbatch --partition="$FS_PARTITION" --export=ALL <plane>/launch_fs_h100.fixed.sh`.
*Exit condition:* job exit 0 **and** the log contains
`checkpoint_saves_adjudicated=N` with N>0 and an `END` line. Exit 95 means training ran
with nothing adjudicated — UNMEASURED, do not retry blindly. Exit 96 names the refused
knob. Note: the login-node argv preflight (#183) does **not** run on the single-job
path; a mistyped `FS_ENGINE_LAUNCH_CMD` surfaces as trainer exit 2 after the
allocation.

**Step H4 — full chain.** Unset `PROBE`; `export FS_SUBMIT_CHAIN=1`; run the launcher on
the login node. The argv preflight fires before the first `sbatch`.
*Exit condition:* preflight 0 (chain proceeds) or 95 (proceed with the warning read);
refusals (5: flags/mode RED — fix the command; 96: unset/untokenizable) must be resolved
before queueing. Then all four hops COMPLETED with the resume hop reporting a restore
verdict (PROVED within `--resume-tolerance`, restore fidelity only since #192).

**Step H5 — record.** Append job ids, verdicts, and any abstentions to
`validation_campaigns/h100_validation/h100/EVIDENCE.md`. A claim without a
ledger row is UNMEASURED by this repository's own rule.

### B.2 GB200

**There is no validated GB200 procedure to reproduce.** The following is the protocol an
engineer would run to *establish* the claim, with every divergence from B.1 stated. Each
step's evidence status today is UNMEASURED.

**Divergence 1 — scheduler.** The tray's estate facts (recorded in
`launchers/fs_container_backend.sh`'s header comments): sbatch/srun/squeue/sinfo absent.
Therefore H0-equivalent pre-submit steps that assume Slurm — the `sinfo` walltime
oracle, the sbatch staging problem (#142 resolver), the submit chain — **do not
transfer**. The launchers run against the allocation the tray already represents. Any
document that copies B.1's sbatch idiom to GB200 is inventing a procedure.

**Divergence 2 — container runtime.** Enroot + torchrun, not singularity + `srun`.
Consequences: the `--nv` fix (#167) is singularity-specific — enroot gets driver
libraries from its own hooks; the `SINGULARITYENV_*` injection mechanism does not exist,
which is exactly why #122 (resume env crossing) had to move to the allowlist. On the
tray, the enroot arm is the *only* arm; the slurm arm in the backend is retained dead
code by design.

**Divergence 3 — interconnect.** A GB200 tray is NVLink-domain hardware; the H100
estate's measured `FS_NCCL_NET_PLUGIN=none` requirement does not transfer by analogy.
The correct first measurement on the tray is the fs129-style collective probe through
`run_in_container` — *Exit condition:* the all_reduce completes on all ranks, recorded
with plugin selection in an evidence ledger **that does not yet exist for this
platform**.

**Step G1–G4 sketch (all currently UNMEASURED):** G1: run the launcher contract shells
(`launchers/test_launcher_contracts.sh`) on the tray and record output. G2: LoRA probe
via `launchers/launch_g4e4b_lora_1tray.sh`, bounded budget, exit 0 with an adjudicated
checkpoint. G3: full-FT probe via `launchers/launch_g4e4b_fullft_1tray.sh`, same exit
condition. G4: resume from G3's checkpoint with a restore verdict in the trainer's
0/1/2/3 namespace. Until G1–G4 produce recorded job outputs, **no GB200 claim of any
kind is validated**.

---

## PART C — The regression surface

### C.1 What invalidates what, and which gate catches it

| Change | Claim invalidated | Gate that catches it (path) |
|---|---|---|
| Stage added/renamed; docs still cite old count | "build green at 42 stages" | `validation_campaigns/h100_validation/gate_doc_stage_count.py` |
| Stage or patch omitted from STAGES or PUBLISH_SET | the fix runs/ships at all | `validation_campaigns/h100_validation/gate_stage_orphans.py` |
| Build input parked in the output tree | "rebuilt from scratch" | `validation_campaigns/h100_validation/gate_build_inputs.py` |
| Artifacts drift on run-time filename references | "individually green" but plane cannot start (#142 shape) | `validation_campaigns/h100_validation/gate_artifact_linkage.py` |
| Env export vs allowlist divergence | resume/env crossing (#115/#122 class) | `validation_campaigns/h100_validation/gate_env_drift.py` (build standing check) |
| `SystemExit("msg")` replacing contract codes | 0/5/95/96 vocabulary | `validation_campaigns/h100_validation/gate_exit_contract.py` |
| Checkpoint naming drift between writer and adjudicator | A7b abstains on 100% of real dirs (#150 shape) | `validation_campaigns/h100_validation/gate_ckpt_naming_agreement.py` |
| LAUNCH.md text drifts from launcher/trainer | operator instructions | `validation_campaigns/h100_validation/gate_launch_doc.py` |
| Any countable reworded in a doc | this corpus's drift surface | `checks/countables_drift.py` |
| Splice damages the enroot arm | GB200 path (code-level only) | `validation_campaigns/h100_validation/apply_splice.py` gates G1–G7 |
| Anchored fix misses/over-applies | each #113/#117-class fix | `validation_campaigns/h100_validation/apply_113.py` (A1–A5), `apply_117.py` (B1–B7) |
| Estate identifier into public tree | publishability | blocklist stage inside `build_h100_plane.sh`, patterns from `validation_campaigns/h100_validation/fs_estate_pat.py` |
| Repo-wide countable rotated | repo-wide total | `checks/countables_drift.py` (the logged wording change: "git-tracked ... repo-wide" replaced the retired "measured ..." wording on 2026-09-03) |

Plus standing checks `checks/bash_lc_sweep.py`, `checks/packaging_reachability.py`,
`checks/wf_yaml_audit.py`, and the live-run instruments `tools/live_save_gate.py`,
`tools/preflight.py`, `tools/real_checkpoint_probe.py`.

### C.2 The gaps — invalidating changes NO gate would notice

1. **Hardware drift.** A new container image whose NCCL plugin no longer SIGSEGVs (or
   newly does), a partition whose real MaxTime changed, a driver update — every such
   change invalidates A.1 items 5–6, and the only detector is **re-execution** of B.1.
   No static gate can notice; the claim simply ages. The ledger is dated; treat old rows
   as expiring.
2. **`#180` provenance record remains build-certified only.** If the next cluster job
   fails to write `launch.<jobid>.provenance.json`, no gate fires — the gap is
   acknowledged in `DELIVERABLE_B_validation_report.md` §1 and closes only when a job
   produces the file.
3. **`#205` is OPEN** — the bare-marker detector class remains stage-scoped, so a
   bare-`except`-style marker outside the current stage's scope invalidates the
   defect-class closure with no gate noticing.
4. **`gate_launch_contract.py` is RED on L2 (6 of 18)** and correctly refuses to write;
   LAUNCH.md is hand-written meanwhile. `gate_launch_doc.py` backstops doc↔code drift,
   but the *generated* command (Deliverable D) being red means one intended producer of
   truth is silent — a doc edit that only the generator's L2 would have caught can pass.
5. **GB200, all of it.** Every claim about the tray path has no execution evidence and
   the H100 gates reason about the H100 plane. A change breaking the enroot arm in a way
   `apply_splice.py`'s static gates do not cover (G1–G7 check structure, not runtime
   behaviour) would be invisible until G1–G4 are actually run — which has never been
   recorded.
6. **`FS_NCCL_SOCKET_IFNAME` validated-then-discarded (#131).** A known defect in the
   shipped plane: the knob is checked and ignored. No gate maps "validated" to
   "consumed"; an operator relying on it changes nothing while believing otherwise.
7. **Cross-node-shape closure (#204).** Closing the hard-coded-geometry defect was
   verified by build gates and **never by a submit on a second geometry**. No gate can
   supply that observation; it is a scheduled measurement, not a check.
8. **Model-behaviour regressions behind green infra.** #202 (positional FSDP buffer
   sync) produced correct collectives and swapped tensors — every structural gate green.
   Only the resume/fixed-eval measurements on hardware caught it. Any future
   silent-reorder defect of that class is likewise invisible to C.1's entire table.

The governing rule for all eight gaps: a denominator that has not been re-executed is a
claim on credit. The ledger in `validation_campaigns/h100_validation/h100/EVIDENCE.md` is the only
denominator this repository recognises, and for GB200 that ledger does not yet exist.
