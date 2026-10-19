---
name: using-foundationskills
license: MIT
description: "User-invoked slash command: run ONLY when the user types /using-foundationskills; never choose it on your own. Entry point for FoundationSkills: routes a user's intent (or a have→want chain such as {raw_data_ref, goal_spec} to fs_launch_spec) to the right FoundationSkills skill, lists every registered skill (name, invocation, invokes, consumes, produces) and states the workflow and the non-negotiable rules the workflow must keep. Do NOT use for building datasets (use fskills-data-engine), vetting a plan draft against a probe (use fskills-probe-first), writing training plans (fskills-training) or checkpoint scoring (fskills-evaluation): it routes and documents, runs no measured work, writes only a skill_route artifact, and never invents a capability."
compatibility: "Needs Python>=3.10 and the foundationskills package (fskills CLI). It reads only the skill registry (SkillRegistry + register_builtin_skills) and never imports FoundationScale: on a host without the package there is no routed skill list to give, and nothing is planned or emitted."
metadata:
  version: "0.1.0"
  invocation: "user"
  fs_status_doctrine: "PASS/RED/UNMEASURED/REFUSED (exit 0/5/95/96)"
  when_to_use: '["which FoundationSkills skill should I use","how do I start with FoundationSkills","what is the FoundationSkills workflow","I have raw_data_ref and goal_spec and want an fs_launch_spec — which chain runs?","what must I never do in FoundationSkills"]'
---

# using-foundationskills — the FoundationSkills routing entry point

## Purpose

Use this first. It is the entry point that routes a user's intent to the right
FoundationSkills skill and states the workflow and the non-negotiable rules every
later step must keep. It REPLACES the hand-written root `SKILLS.md` as the source
of truth: that file is a generated projection of this skill. Listing the registered
skills is measured (`SkillRegistry` + `register_builtin_skills`), routing is
measured (`SkillRegistry.plan_chain`), and nothing is invented: a `want` no
registered skill produces is refused (UF-IN-001), and an `intent` that matches no
skill is UNMEASURED (UF-IN-002) — absence of a route is never PASS.

## When to use

- "Which FoundationSkills skill should I use?" — before any other skill runs.
- "How do I start with FoundationSkills / what is the workflow / what must I never do?".
- "I have artifacts X and Y and want artifact Z — which chain runs?" (`have`/`want`).
- To see every registered skill with what it consumes and produces.
- Not for building datasets (use fskills-data-engine), vetting a plan draft against
  a probe (use fskills-probe-first), authoring training plans (fskills-training) or
  checkpoint scoring (fskills-evaluation) — route to those and stop here.

## Inputs

| Field | Required | Meaning |
|---|---|---|
| `intent` | no | free-text question; skills are ranked by lower-cased word overlap with their description (words of ≤ 3 letters ignored) |
| `have` | no (with `want`) | artifact type strings you already have, e.g. `["raw_data_ref", "goal_spec"]` |
| `want` | no (with `have`) | artifact type you want, e.g. `fs_launch_spec`; refused by UF-IN-001 if no registered skill produces it |

At least one of `intent` / `have`+`want` is expected; an empty request still
returns the registered skill list.

## Outputs

One `skill_route` artifact (typed, schema-validated, with provenance) whose payload is: `skills` (one row per
registered skill: `name`, `invocation`, `invokes`, `consumes`, `produces`, `description`), `candidates` (the top 3
skill packages by SKILL.md name, `{name, score}`, when `intent` is given: a lexical shortlist scored by IDF-weighted
word overlap with each package's description and when_to_use phrasings; measured top-1 76/98 on the routing
questions, so the agent confirms it rather than trusting it) and `chain` (`SkillRegistry.plan_chain(set(have), want)` as skill
names, when `have` and `want` are both given).

Exit codes: 0 PASS (the route is measured), 95 UNMEASURED (UF-IN-002: the intent matches no registered skill),
96 REFUSED (UF-IN-001: the `want` names an artifact type no registered skill produces, or no chain reaches it).
There is no handoff rule, so RED 5 never fires here.

## Scope

Every registered FoundationSkills skill, both model-invoked and user-invoked, for LLM and VLM work of any family and
stage. It reads the skill registry and each package's SKILL.md frontmatter, writes one `skill_route` artifact, and never
runs another skill, imports FoundationScale or launches anything.

## Workflow

1. **Understand the goal.** Build a `goal_spec`. For every load-bearing fact that is
   missing, `foundationskills.agent.intake.missing_facts(goal, skills)` returns the
   question to ask: base model, data location and size, target capability, hardware
   and budget. Never invent a load-bearing value.
2. **Probe.** Run `fskills probe --deep`. Plan only against what the installed
   FoundationScale measurably runs.
3. **Plan.** Run `training.planner`. Present the plan's `decisions` (each has a
   `because`), the estimates, the feasibility verdict and the `provenance_notes`.
   Recipes marked `literature` are not yet validated on FoundationScale, so say so.
4. **Data.** If a stage's data is not ready (`data.handoff == "data_engine"`), run
   `data_engine` for that format. Proceed only if the readiness verdict is PASS.
   UNMEASURED lists the checks that did not run, so resolve them or tell the user.
5. **Emit.** Run `training.emit` to get one `fs_launch_spec` per stage. A spec with
   `executable: false` names the missing piece in `missing`. Report it; do not work
   around it.
6. **Confirm, then launch.** Show the user the spec and `fskills hash --spec ...`.
   Launch only with `--confirm <hash>` given by the user. Never auto-fill the hash.
   `--measurement-only` runs a spec whose only gap is the hand-off (for example, FS RL
   saves no checkpoint), and only when the user asks for a measurement run.
7. **Monitor and iterate.** On failure, call the skill's `diagnose()`, or
   `TrainingPlannerSkill.diagnose_symptom(name)` for loss spikes, divergence, OOM,
   reward hacking, reward saturation (UNMEASURED steps), MoE router imbalance and FS
   refusals. The launcher keeps the full log in `<output_dir>/fskills_launch.log` and
   recovers FS's own verdict even when torchrun reports exit 1.

To find a chain from what you have to what you want, use
`SkillRegistry.plan_chain(have, want)`. For example, `{raw_data_ref, goal_spec}` to
`fs_launch_spec` gives `data_engine → training.planner → training.emit`.

### Skills

| Skill | Consumes | Produces | Guide |
|---|---|---|---|
| `data_engine` | `raw_data_ref` | `data_pipeline_spec`, `dataset`, `readiness_report` | `foundationskills/skills/data_engine/SKILL.md` |
| `training.planner` | `goal_spec` (+ `readiness_report`) | `training_plan` | `foundationskills/skills/training/SKILL.md` |
| `training.emit` | `training_plan`, `dataset` | `fs_launch_spec` (+ sbatch) | `foundationskills/skills/training/SKILL.md` |
| `evaluation` | `checkpoint` (or an FS `run_manifest.json`) | `eval_report` | `foundationskills/skills/evaluation/SKILL.md` |
| `auto_research` | an approved campaign spec (request) | `auto_research_report` (+ hash-chained ledger) | `foundationskills/skills/auto_research/SKILL.md` |

Discipline skills (A3.1) gate the above: `probe_first` (vet a plan draft against
the measured probe), `measured_or_unmeasured`, `confirm_before_launch` and
`refusal_surface`.

## Non-negotiable rules

- The only statuses are PASS / RED / UNMEASURED / REFUSED (exit 0 / 5 / 95 / 96).
  A refusal names what is missing. A check that did not run is UNMEASURED, never PASS.
- No training job is launched without the user's explicit confirmation of the plan hash.
- Do not claim a capability the probe did not measure, and do not hide a gap the plan
  reports.
- On the GB200 cluster, generated sbatch files already carry:
  - `--time=10-00:00:00`;
  - `--exclude=r01dgx02`;
  - the IMEX fabric preamble;
  - the measured NCCL environment.

  Do not remove them.

## Skill types

There are two kinds of skill (A3.2):

- **Model-invoked skills** (`invocation: "model"`): `data_engine`,
  `training.planner`/`training.emit`, `evaluation`, `auto_research` and the A3.1
  discipline skills. The agent picks one from its description and runs it.
- **User-invoked workflow skills** (`invocation: "user"`), run as slash commands such
  as `/fs-goal`, `/fs-data`, `/fs-plan`, `/fs-launch`, `/fs-eval`, `/fs-research`,
  `/fs-status` and `/fs-handoff`. They run ONLY when the user invokes them. A user
  skill may compose model skills (declared by name in `invokes`) but NEVER another
  user skill. The slash commands are being added in A3.2 — do not assume they all
  exist yet; until one is registered, run the model skill it would compose and say so.

This entry point itself is `invocation: "user"` and composes nothing: it routes and
states the rules.

## Adding a skill

See `docs/ARCHITECTURE.md` ("Adding a skill"). In short:
1. subclass `BaseSkill`; set `invocation` ("model": the agent picks it from its description; "user": a
   slash-command workflow skill) and, for a user skill, `invokes` (the model skills it composes; never a
   user skill). `metadata.invocation` in SKILL.md must agree; `tests/contract/test_invocation.py` checks it;
2. declare `consumes`/`produces` artifact types;
3. declare every rule with a MUST_FIRE fixture;
4. ship a `SKILL.md` with the standard sections, opening with routing frontmatter
   (`name` in lowercase-hyphen form, `license`, a `description` that says what it is
   NOT for, and at least 3 `when_to_use` phrasings);
5. ship `evals/evals.json`: at least 3 cases that should trigger the skill, one routed
   to a sibling skill and one no FoundationSkills skill should take;
   then run `fskills routing-eval run --base-url ... --model ... --write-benchmarks`, which
   measures routing on every package's cases (exit 0/5/95) and regenerates each `BENCHMARK.md`;
   then run `fskills behaviour-eval run --agent-... --judge-... --write-behaviour` (agent and judge must be
   different models), which checks each positive case's `expected_behavior` with and without the SKILL.md
   and regenerates `BEHAVIOUR.md`. It is plan-level (no tool execution), and because the behaviours are
   written from the SKILL.md, read the without-skill baseline and the uplift, not the with-skill ceiling;
   then run `fskills oracle-cases build` (whenever a rule, its MUST_FIRE fixture or its severity changes; a
   contract test enforces it), which writes code-grounded cases to `evals/behaviour.json`, and
   `fskills behaviour-eval run --cases oracle ... --write-behaviour`, which checks that the SKILL.md lets an
   agent predict what the code does (`BEHAVIOUR_ORACLE.md`); `--resume REPORT` re-asks only unmeasured calls;
6. register it (in `register_builtin_skills` or through the `foundationskills.skills` entry point).

`tests/contract/test_conformance.py` then checks it automatically.

## Validation rules

Outcome semantics (from `core/contract.py`, the same for every skill): an **input**-phase
BLOCK refuses before any work — status REFUSED, exit 96, nothing written. A
**handoff**-phase BLOCK does not refuse: the work ran, and the result is RED (exit 5),
never PASS. WARN and INFO findings never block; they travel with the result
(UF-IN-002 makes the run UNMEASURED, exit 95). Phase by prefix: UF-IN-* rules are
input phase.

| Rule | Severity | Fires when |
|---|---|---|
| UF-IN-001 | BLOCK | `want` names an artifact type no registered skill produces (or `SkillRegistry.plan_chain(have, want)` finds no route) → REFUSED 96 |
| UF-IN-002 | WARN | `intent` matches no registered skill (zero word overlap with every description) → UNMEASURED 95; absence of a route is never PASS |

## Failure handling

- `want` unreachable (no registered skill produces it, or no chain from `have`):
  REFUSED (96) naming UF-IN-001 and the producible artifact types; nothing is written.
- `intent` matches no registered skill: UNMEASURED (95) with UF-IN-002 and the full
  `skills` list in the payload — restate the intent against that list, or ask the user
  which skill they mean. Never guess a route.
- The request names work outside every skill's scope (email writing, CI bug fixes):
  no skill is picked; say that no FoundationSkills skill applies and stop.
- On failure of a routed skill, call its `diagnose()` (or
  `TrainingPlannerSkill.diagnose_symptom(name)`); do not improvise a workflow here.

## FS interface

Reads only the skill registry (`SkillRegistry`, `register_builtin_skills`) and a
`fskills skills` listing; computes chains through `SkillRegistry.plan_chain`. This
skill never imports or launches FoundationScale, writes only its skill_route artifact, and touches no
cluster resource. On a host where the `fskills` CLI or the `foundationskills`
package is missing, report REFUSED with the install step
(`pip install -e <FoundationScale repo>/FoundationSkills`, or run where it is
installed) and never improvise a route.

## Worked examples

