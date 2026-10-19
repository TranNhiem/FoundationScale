---
name: fs-plan
license: MIT
description: "User-invoked slash command: run ONLY when the user types /fs-plan; never choose it on your own. Planning workflow (/fs-plan) for FoundationSkills: composes three model skills in order - training.planner -> probe_first -> training.emit - and stops at the first step that is not PASS (FP-IN-003 on a refusal, FP-HO-001 on RED or a non-executable spec, FP-HO-002 on UNMEASURED). On PASS it SHOWS the plan hash of every emitted fs_launch_spec for the user to type back to /fs-launch; it never launches and never fills a hash into a launch. Do NOT use for launching (/fs-launch), sealing a confirmation (confirm_before_launch) or explaining a status (/fs-status): it plans, probes and emits only."
compatibility: "Needs Python>=3.10 and the foundationskills package (fskills CLI) with FoundationScale available (requires_fs is True): it composes training.planner, probe_first and training.emit through the skills registry. It writes one workflow_record artifact and never touches the scheduler; launching is /fs-launch's job."
metadata:
  version: "0.1.0"
  invocation: "user"
  fs_status_doctrine: "PASS/RED/UNMEASURED/REFUSED (exit 0/5/95/96)"
  when_to_use: '["plan this training run with /fs-plan","/fs-plan build the training plan from my goal spec and the probe report","produce the fs_launch_specs and plan hashes for this goal","compose planner, probe-first and emit for me end to end"]'
---

# fs-plan — the user-invoked planning workflow (/fs-plan)

## Purpose

The plan-to-hash workflow: when the user types `/fs-plan`, it composes three **model** skills in a fixed
order — `training.planner` -> `probe_first` -> `training.emit` — and **stops at the first step that is not
PASS**. On PASS it writes one `workflow_record` and SHOWS the user the plan hash of every emitted
`fs_launch_spec`, to read and type back to `/fs-launch`; it never launches and never fills a hash into a
launch. It composes model skills and never another user skill (`invokes` holds only `training.planner`,
`probe_first`, `training.emit`). Outcome mapping (from `foundationskills.skills._workflow`): a composed
step that REFUSES becomes FP-IN-003 (REFUSED 96, nothing written); a composed step that is RED, or an
emitted spec with `executable` false, becomes FP-HO-001 (RED 5); a composed step that is UNMEASURED becomes
FP-HO-002 (UNMEASURED 95, never PASS).

## When to use

- "Plan this training run: ..." / "Build the training plan and the launch specs for this goal" — i.e. the
  user types `/fs-plan`.
- To go from a `goal_spec` (plus `readiness`, `dataset`, `probe_report`) to emitted `fs_launch_spec`
  artifacts with plan hashes the user can type back to `/fs-launch`.
- To run probe-first planning end to end: planner -> probe_first -> emit, stopping at the first non-PASS step.
- Not for launching (that is `/fs-launch`), sealing a `confirm_before_launch` record without launching
  (that is `confirm_before_launch`), or explaining a result status (that is `/fs-status`).

## Inputs

All inputs are flat request fields (no extra keys are allowed); file fields are artifact paths the skill
reads through `load_payload`.

| Field | Required | Meaning |
|---|---|---|
| `goal` | yes | path to a `goal_spec` artifact — what to train |
| `readiness` | no | path to a readiness report handed to the planner alongside the goal |
| `dataset` | no | path to a `dataset` artifact handed to the emit step |
| `probe_report` | yes | path to the probe report (`fskills probe --deep --json`); nothing is planned against what FS cannot measurably run (FP-IN-002) |
| `model` | no | model path/name handed to the emit step |
| `output_root` | no | run output root handed to the emit step (default: the workdir `runs` directory) |
| `hardware_id` | no | hardware target for planner and emit (default `gb200`) |
| `nodes` | no | node count for the emitted specs (default 1, minimum 1) |
| `gpus_per_node` | no | GPUs per node for the emitted specs (default 1, minimum 1) |
| `out` | no | directory to write the `workflow_record` into |

## Outputs

One `workflow_record` artifact (typed, schema-validated, with provenance) whose payload is `workflow`
(`"fs_plan"`), `steps` (one row per composed step: `training.planner`, `probe_first`, `training.emit`),
`outputs` (`training_plan` path, the `probe_report` path, and `launch_specs`: for each emitted
`fs_launch_spec` a row `{path, stage, plan_hash, executable, missing}`), `verdict` (`"PASS"` or `"RED"`),
`red_step` (when the verdict is RED: the reason and the offending specs) and `show_user`.

Exit codes: 0 PASS (every step PASS and every emitted spec executable — the `show_user` message reads
"Read each plan_hash below and type it back to /fs-launch with its launch_spec; no hash is ever filled in
for you."; the skill **shows** each spec's plan hash and the user types it back to `/fs-launch` — it never
launches and never fills a hash into a launch), 5 RED (FP-HO-001: a composed step came back RED or an
emitted spec is not executable — the workflow_record names the spec's `missing` item), 95 UNMEASURED
(FP-HO-002: never treated as PASS), 96 REFUSED (FP-IN-001/FP-IN-002/FP-IN-003; nothing is written on a
refusal).

## Scope

LLM and VLM work of any family and stage (pretrain, cpt, sft, preference, rl) under the methods
`full`, `lora`, `qlora`, on hardware `gb200`, `dgx-h100`, `local`. It composes the three named model
skills and never another user skill, writes only its `workflow_record`, and never invokes a launcher.
It requires FoundationScale (`requires_fs` is True) because it composes model skills through the registry.

## Validation rules

Outcome semantics (from `core/contract.py`, the same for every skill): an **input**-phase BLOCK refuses
before any work — status REFUSED, exit 96, nothing written. A **handoff** BLOCK fires after the run over
the payload — RED, exit 5. WARN and INFO findings never block; they travel with the result. Phase by
prefix: FP-IN-* rules are input phase, FP-HO-* rules are handoff phase.

| Rule | Severity | Fires when |
|---|---|---|
| FP-IN-001 | BLOCK | no `goal` is given → REFUSED 96: nothing can be planned |
| FP-IN-002 | BLOCK | no `probe_report` is given (probe first: plan only against what the installed FoundationScale measurably runs) → REFUSED 96 |
| FP-IN-003 | BLOCK | a composed step (`training.planner`, `probe_first` or `training.emit`) refused → REFUSED 96; nothing is written; the step's own findings are the evidence |
| FP-HO-001 | BLOCK | a composed step came back RED, or an emitted spec has `executable` false → RED 5; the workflow_record names the spec's `missing` item |
| FP-HO-002 | WARN | a composed step came back UNMEASURED → UNMEASURED 95 (never PASS), with the finding attached |

## Failure handling

- No `goal`: REFUSED (96) naming FP-IN-001; pass `goal` (a goal_spec artifact path) and re-run /fs-plan.
- No `probe_report`: REFUSED (96) naming FP-IN-002 (probe first); run `fskills probe --deep --json >
  probe.json` and pass it as `probe_report`, then re-run /fs-plan.
- A composed step refuses (`training.planner`, `probe_first` or `training.emit`): REFUSED (96) naming
  FP-IN-003 with that step's findings as evidence; fix what those findings name and re-run /fs-plan.
- A composed step is RED, or an emitted spec has `executable` false: RED (5) naming FP-HO-001 — the
  workflow_record carries `steps` and `red_step`; fix the RED step or the spec's `missing` item and
  re-run /fs-plan. Do not treat the spec as launchable and do not hand a hash to /fs-launch.
- A composed step is UNMEASURED: UNMEASURED (95) naming FP-HO-002 — the plan is not a pass. Resolve what
  the step could not measure and re-run /fs-plan. Never treat UNMEASURED as PASS.
- This skill plans and emits only. On a composed model skill's failure, read that step's row in the
  workflow_record (the first non-PASS step names its own findings) or call `FsPlanSkill.diagnose()`;
  do not improvise a fix here and do not launch anything.

## FS interface

CLI entries it relies on: `fskills plan`, `fskills emit`, `fskills hash` (the composed skills compute
these; the hashes are shown, never submitted). It composes `training.planner` -> `probe_first` ->
`training.emit`, emits nothing into the scheduler and declares no APIs. It never invokes another user
skill and never calls a launcher: `interfaces.fs.launch.launch` (the only launch path) belongs to
/fs-launch. It writes only its `workflow_record` artifact and touches no cluster resource. It requires
FoundationScale: on a host with only `foundationskills` installed the composed model skills cannot run —
report REFUSED with the install step (`pip install -e <FoundationScale repo>/FoundationScale`) and never
improvise a plan.

## Worked examples

**Everything given, all steps PASS.**
User: `/fs-plan` — plan from `goal_spec.q27.json`, `probe_report` `probe.q27.json`, `dataset.json`,
`model=/models/mistral-7b`, `nodes=2`, `gpus_per_node=8`, `hardware_id=gb200`.
The skill runs planner -> probe_first -> emit, writes one `workflow_record` with verdict PASS and one
`launch_specs` row per emitted spec, and shows each row's `plan_hash` with show_user: "Read each
plan_hash below and type it back to /fs-launch with its launch_spec; no hash is ever filled in for you."
The user types those hashes to `/fs-launch` — /fs-plan never launches and never fills a hash in.

**Probe report missing.**
User: `/fs-plan` — plan from `goal_spec.q27.json` only.
FP-IN-002 fires: REFUSED (96) saying no `probe_report` is given and to run
`fskills probe --deep --json > probe.json` first. Nothing is planned and nothing is written.

**Emit returns a non-executable spec.**
`training.planner` and `probe_first` PASS, but `training.emit` returns a spec with
`executable: false, missing: ["sft objective"]`. FP-HO-001 fires: RED (5); the workflow_record's
`red_step` names the not-executable spec and its `missing` item. Fix it and re-run /fs-plan; no hash is
offered for /fs-launch.
