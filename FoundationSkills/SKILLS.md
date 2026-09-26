# FoundationSkills: agent entry point

Read this first. It tells an agent which skills exist, how to chain them, and which rules are not negotiable. Each skill's own `SKILL.md` has its full contract: inputs, outputs, scope, validation rules, failure handling, FS interface and worked examples.

## Skills

| Skill | Consumes | Produces | Guide |
|---|---|---|---|
| `data_engine` | `raw_data_ref` | `data_pipeline_spec`, `dataset`, `readiness_report` | `foundationskills/skills/data_engine/SKILL.md` |
| `training.planner` | `goal_spec` (+ `readiness_report`) | `training_plan` | `foundationskills/skills/training/SKILL.md` |
| `training.emit` | `training_plan`, `dataset` | `fs_launch_spec` (+ sbatch) | `foundationskills/skills/training/SKILL.md` |

To find a chain from what you have to what you want, use `SkillRegistry.plan_chain(have, want)`. For example, `{raw_data_ref, goal_spec}` to `fs_launch_spec` gives `data_engine → training.planner → training.emit`.

## Workflow

1. **Understand the goal.** Build a `goal_spec`. For every load-bearing fact that is missing, `foundationskills.agent.intake.missing_facts(goal, skills)` returns the question to ask: base model, data location and size, target capability, hardware and budget. Never invent a load-bearing value.
2. **Probe.** Run `fskills probe --deep`. Plan only against what the installed FoundationScale measurably runs.
3. **Plan.** Run `training.planner`. Present the plan's `decisions` (each has a `because`), the estimates, the feasibility verdict and the `provenance_notes`. Recipes marked `literature` are not yet validated on FoundationScale, so say so.
4. **Data.** If a stage's data is not ready (`data.handoff == "data_engine"`), run `data_engine` for that format. Proceed only if the readiness verdict is PASS. UNMEASURED lists the checks that did not run, so resolve them or tell the user.
5. **Emit.** Run `training.emit` to get one `fs_launch_spec` per stage. A spec with `executable: false` names the missing piece in `missing`. Report it; do not work around it.
6. **Confirm, then launch.** Show the user the spec and `fskills hash --spec ...`. Launch only with `--confirm <hash>` given by the user. Never auto-fill the hash. `--measurement-only` runs a spec whose only gap is the hand-off (for example, FS RL saves no checkpoint), and only when the user asks for a measurement run.
7. **Monitor and iterate.** On failure, call the skill's `diagnose()`, or `TrainingPlannerSkill.diagnose_symptom(name)` for loss spikes, divergence, OOM, reward hacking, reward saturation (UNMEASURED steps), MoE router imbalance and FS refusals. The launcher keeps the full log in `<output_dir>/fskills_launch.log` and recovers FS's own verdict even when torchrun reports exit 1.

## Non-negotiable rules

- The only statuses are PASS / RED / UNMEASURED / REFUSED (exit 0 / 5 / 95 / 96). A refusal names what is missing. A check that did not run is UNMEASURED, never PASS.
- No training job is launched without the user's explicit confirmation of the plan hash.
- Do not claim a capability the probe did not measure, and do not hide a gap the plan reports.
- On the GB200 cluster, generated sbatch files already carry:
  - `--time=10-00:00:00`;
  - `--exclude=r01dgx02`;
  - the IMEX fabric preamble;
  - the measured NCCL environment.

  Do not remove them.

## Adding a skill

See `docs/ARCHITECTURE.md` ("Adding a skill"). In short:
1. subclass `BaseSkill`;
2. declare `consumes`/`produces` artifact types;
3. declare every rule with a MUST_FIRE fixture;
4. ship a `SKILL.md` with the standard sections;
5. register it (in `register_builtin_skills` or through the `foundationskills.skills` entry point).

`tests/contract/test_conformance.py` then checks it automatically.
