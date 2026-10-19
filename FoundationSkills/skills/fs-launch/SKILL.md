---
name: fs-launch
license: MIT
description: "User-invoked slash command: run ONLY when the user types /fs-launch; never choose it on your own. Slash-command launch gate (/fs-launch) for FoundationSkills and THE ONLY launch path: takes an fs_launch_spec artifact path plus the plan hash the USER typed back (user_plan_hash), refuses when it is missing and names `fskills hash --spec` for the USER to read the hash (never echoed, never filled in), composes confirm_before_launch (its confirmation_record is the only confirmation accepted) and then launches through interfaces.fs.launch.launch -- submit true submits to the scheduler, submit false runs it here and waits -- writing one launch_record {launch_spec, plan_hash, confirmation, submitted, returncode, result}. Do NOT use for planning, emitting specs, or confirming without launching."
compatibility: "Needs Python>=3.10, the foundationskills package (fsskills CLI) and FoundationScale (requires_fs is True): interfaces.fs.launch.launch submits through sbatch or runs the spec argv here. It composes confirm_before_launch by composition (registry.get(\"confirm_before_launch\").execute), never by hand, and never submits on anything but the exact token the user typed back."
metadata:
  version: "0.1.0"
  invocation: "user"
  fs_status_doctrine: "PASS/RED/UNMEASURED/REFUSED (exit 0/5/95/96)"
  when_to_use: '["launch this training run","submit the plan to the cluster","start the job with this launch spec","run this fs_launch_spec here and wait for the returncode"]'
---

# fs-launch — THE ONLY launch path (/fs-launch)

## Purpose

The launch gate: when an `fs_launch_spec` has been emitted and the work is ready to go, `/fs-launch` is
the only way it runs. It takes the spec path plus the plan hash the USER typed back, recomputes that hash
exactly as `confirm_before_launch` computes it (`fskills hash --spec`), composes `confirm_before_launch`
(its `confirmation_record` is the only confirmation accepted) and only then calls
`interfaces.fs.launch.launch` — `submit: true` submits to the scheduler, `submit: false` runs it here and
waits. When the user has not typed a hash back it refuses and names `fskills hash --spec` for them to read it; it never echoes or fills in the hash: the confirmation
is the user's sentence, not the agent's. One `launch_record` artifact records what was launched. Do NOT use
for planning, emitting specs, or confirming without launching: plan with `fsskills-training`, emit the spec,
confirm and launch HERE.

## When to use

- "Launch this training run." / "Submit the plan to the cluster." / "Start the job with this launch spec."
- When an `fs_launch_spec` artifact exists and the user wants the scheduler (or this host) to actually run it.
- When a launch came back refused with FL-IN-001/002/003 and the fix is ready (the user has typed the plan
  hash back, the spec was re-emitted): re-run THIS skill — it refuses (naming `fskills hash --spec`) or consumes the hash and launches.
- Not for planning a run or emitting a spec (`fsskills-training`), not for sealing a confirmation record
  without launching (`fsskills-confirm-before-launch`), and not for explaining an old result
  (`fs-status`): launch here, act there.

## Inputs

| Field | Required | Meaning |
|---|---|---|
| `launch_spec` | yes | path to an `fs_launch_spec` artifact (typed envelope or bare payload), hashed exactly as `confirm_before_launch` hashes it |
| `user_plan_hash` | in practice yes | the plan hash the USER typed back (`fskills hash --spec <launch_spec>`); absent/empty is REFUSED by FL-IN-001 — the refusal re-shows the recomputed hash and this skill never fills it in |
| `submit` | no (default `true`) | `true`: submit to the scheduler (sbatch); `false`: run the launcher here and wait (local) |
| `out` | no | directory the `launch_record` is written into (default: the context artifacts dir) |

`submit` and `user_plan_hash` are optional at schema level on purpose: an empty `user_plan_hash` must reach
`check_inputs` so FL-IN-001 can name `fskills hash --spec` instead of a schema error hiding the cause.

## Outputs

One `launch_record` artifact (typed, schema-validated, with provenance) whose payload is: `launch_spec`
(path), `plan_hash` (the recomputed hash that equals what the user typed back), `confirmation` (the composed
`confirm_before_launch` record's artifact id or path), `submitted` (bool), `returncode` (int, or null for an
sbatch submission) and `result` (the launcher's returned dict, JSON-safe: `job_id`, `pid`, `command`,
`dry_run_rc`, `returncode`, `log`, `tail`, `fs_verdict`). See `SCHEMA` for the JSON schema.

Exit codes: 0 PASS when the launch went out (`submitted`) or the local run returned 0; 96 REFUSED (FL-IN-001
missing hash — the user reads it from `fskills hash --spec`; never echoed or filled in; FL-IN-002 the composed `confirm_before_launch` did not PASS;
FL-IN-003 the launcher refused); 5 RED via FL-HO-001 when a local launch (submit false) returned non-zero —
the run itself failed. The composed skill's findings travel only in
`evidence["confirm_before_launch"]` as `{rule_id, severity, message}` rows, never as this skill's findings.

## Scope

Every FoundationSkills `fs_launch_spec`, for LLM and VLM work of any family and stage (scope identical to
`fs-status`). It consumes one `fs_launch_spec`, composes one `confirm_before_launch`, writes one
`launch_record` and invokes nothing else (`invokes` is exactly `("confirm_before_launch",)`). It is THE ONLY
launch path: nothing else may submit a spec.

## Hash display decision

The roadmap asks `/fs-launch` to re-show the plan hash and never fill it in. This skill does it the way the existing
doctrine already measures safe: the USER runs `fskills hash --spec <path>` and reads the hash; the refusal names that
command and never prints the hash. `confirm_before_launch` never echoes the expected hash (CB-IN-001, README.md:45):
a hash an agent can read out of a refusal is a hash an agent can type back on the user's behalf, which is exactly the
auto-fill both skills forbid.

## Validation rules

Outcome semantics (from `core/contract.py`, the same for every skill): an **input**-phase BLOCK refuses
before any work — status REFUSED, exit 96, nothing written. A **handoff**-phase BLOCK is RED 5 after the
work ran. WARN and INFO findings never block; they travel with the result. Phase by prefix: FL-IN-* rules
are input phase, FL-HO-* rules are handoff phase.

| Rule | Severity | Fires when |
|---|---|---|
| FL-IN-001 | BLOCK | `user_plan_hash` is absent or empty → REFUSED 96. The refusal names `fskills hash --spec` for the user instead of echoing the plan hash recomputed from the spec so the USER can type it back; the skill and the agent never fill it in |
| FL-IN-002 | BLOCK | the composed `confirm_before_launch` skill did not PASS (wrong hash, unreadable or non-`fs_launch_spec` spec) → REFUSED 96; its findings are embedded as `{rule_id, severity, message}` rows in `evidence["confirm_before_launch"]`, never returned as findings. Applies instead of FL-IN-001 when the spec itself cannot be read |
| FL-IN-003 | BLOCK | `interfaces.fs.launch` refused the launch (`ConfirmationRequired` or `LaunchRefused`) → REFUSED 96 with the launcher's message verbatim |
| FL-HO-001 | BLOCK | a LOCAL launch (`submit: false`) returned a non-zero `returncode` → RED 5: the run itself failed and its artifacts must not be used |

## Failure handling

- No `user_plan_hash`: REFUSED (96) naming FL-IN-001 with the plan hash re-read from the spec printed in the
  message; ask the user to type it back and re-run. The hash is never auto-filled here or by the agent.
- The composed `confirm_before_launch` refuses (wrong hash, unreadable or non-`fs_launch_spec` spec):
  REFUSED (96) naming FL-IN-002, with the inner `CB-IN-*` findings visible in the evidence block. Nothing
  launches. Fix the spec or get the right hash typed back, then re-run.
- `ConfirmationRequired` or `LaunchRefused` from the launcher (bad token at launch time, `executable` not
  `true`, slurm target without a script, dry-run non-zero): REFUSED (96) naming FL-IN-003 and quoting the
  launcher's message. Fix what it names and re-run — this stays the only launch path.
- A local (`submit: false`) run that returned non-zero: the `launch_record` is written, then handoff FL-HO-001
  makes the verdict RED 5 (the run itself failed). Read `result.tail` / `result.fs_verdict` / `result.log`
  and re-run; never treat a failed run as PASS.
- This skill launches only. For the plan, call `TrainingPlannerSkill.diagnose_symptom(name)`; for an old
  result's meaning, `/fs-status`. Do not improvise a fix here.

## FS interface

Entry `fskills launch`; the only API it drives is `interfaces.fs.launch.launch` (`submit: true` writes and
runs `sbatch`, `submit: false` runs the argv here and waits). It composes `confirm_before_launch` for the
confirmation and writes one `launch_record` artifact — it patches nothing and never bypasses the
confirmation hash. `requires_fs` is True: a host without FoundationScale returns REFUSED (CORE-HOST-001)
from the template with the install step (`pip install -e <FoundationScale repo>/FoundationSkills`), and the
agent never improvises a launch without it.

## Worked examples

- `launch/fs_launch_spec.q27.json` + `user_plan_hash: 3f2a91c0b7de4410` (typed back by the user) +
  `submit: true` → `confirm_before_launch` seals `confirmation_record cb-…`, `launch()` submits via sbatch,
  `launch_record` records `submitted: true`, `returncode: null`, `result.job_id`, PASS 0.
- `launch/fs_launch_spec.rl.json` + the typed-back hash + `submit: false` → runs here and waits; the
  `launch_record` carries `submitted: false` and the real `returncode`. A non-zero returncode is RED 5 by
  FL-HO-001 (the run itself failed) with the tail and `fs_verdict` in `result`.
