---
name: fs-status
license: MIT
description: "User-invoked slash command: run ONLY when the user types /fs-status; never choose it on your own. Slash-command explainer (/fs-status) for FoundationSkills results: takes a journal.jsonl record written by the orchestrator (last record by default, or a step_index) or a skill result object {status, exit_code, refusal, findings} and returns one plain-English status_explanation saying what the status means (PASS 0, RED 5, UNMEASURED 95, REFUSED 96) and what to do next. Side-effect free: it composes and invokes nothing, writes only its own artifact. Do NOT use for fixing the problem, re-running skills, or deciding whether to launch: it explains, it does not act."
compatibility: "Needs Python>=3.10 and the foundationskills package (fskills CLI). It requires no FoundationScale: it reads one journal record or a result object and writes one status_explanation artifact, never imports FoundationScale and never patches (requires_fs is False)."
metadata:
  version: "0.1.0"
  invocation: "user"
  fs_status_doctrine: "PASS/RED/UNMEASURED/REFUSED (exit 0/5/95/96)"
  when_to_use: '["what does exit code 95 mean","explain this fskills result","why did my run come back REFUSED","what does this journal record tell me to do next","explain the statuses my fskills run returned"]'
---

# fs-status — the FoundationSkills result explainer (/fs-status)

## Purpose

The wait-what explainer: when a run comes back and you are not sure what it means,
`/fs-status` translates one orchestrator journal record (or one result object as
printed by the fskills CLI) into a plain-English `status_explanation` and names the
next step. It is side-effect free: it composes nothing, invokes nothing and writes
only its own artifact. It explains, it does not act — it never fixes the problem,
re-runs a skill or decides whether to launch. Reading a record is measured against
the doctrine in `core/status.py` (PASS 0, RED 5, UNMEASURED 95, REFUSED 96): a record
that cannot be read is REFUSED (ST-IN-001), a record whose `status` and `exit_code`
disagree with the doctrine is REFUSED (ST-IN-002) — an inconsistent record cannot be
explained honestly.

## When to use

- "What does exit code 95 mean?" / "Explain this fskills result" / "Why did my run
  come back REFUSED?".
- To read a `journal.jsonl` record (the last one, or a given `step_index`) and say
  what its status means and what to do next.
- To translate a result object `{status, exit_code, refusal, findings}` printed by
  the fskills CLI into one explanation.
- Not for fixing the problem, re-running the skill, or deciding whether to launch
  (fix with the owning skill such as `fskills-probe-first`, launch with
  `fskills-training`): explain here, act there.

## Inputs

| Field | Required | Meaning |
|---|---|---|
| `journal` | one of `journal` / `result` | path to a `journal.jsonl` written by the orchestrator |
| `step_index` | no (with `journal`) | which record to explain; default: the LAST record; out of range is refused by ST-IN-001 |
| `result` | one of `journal` / `result` | a skill result object: `{"status": "PASS|RED|UNMEASURED|REFUSED", "exit_code"?, "refusal"?, "findings"?, "skill"?}` |

## Outputs

One `status_explanation` artifact (typed, schema-validated, with provenance) whose payload is: `source`
(`"journal"` or `"result"`), `skill`, `status`, `exit_code`, `meaning` (one plain-English sentence for
that status), `next_step` (one plain sentence: PASS hands the artifacts on; RED/UNMEASURED/REFUSED fix
what the findings or refusal name and re-run that skill — never treat UNMEASURED as PASS), `refusal`,
`findings` (`{rule_id, severity, message, recovery}` copied from the result when given) and
`artifact_ids` (from the journal record when given). See `SCHEMA` for the JSON schema.

Exit codes: 0 PASS (the explanation itself is measured), 96 REFUSED (ST-IN-001: no readable journal
record and no result object; ST-IN-002: the record's status and exit_code disagree with the doctrine).
There is no handoff rule, so RED 5 never fires here.

## Scope

Every FoundationSkills skill's result, for LLM and VLM work of any family and stage. It reads one
journal record or one result object, writes one `status_explanation` artifact, and never runs another
skill, imports FoundationScale or touches a cluster resource. Side-effect free: it invokes nothing
(`invokes` is empty) and composes no slash command.

## Validation rules

Outcome semantics (from `core/contract.py`, the same for every skill): an **input**-phase BLOCK refuses
before any work — status REFUSED, exit 96, nothing written. WARN and INFO findings never block; they
travel with the result. Phase by prefix: ST-IN-* rules are input phase.

| Rule | Severity | Fires when |
|---|---|---|
| ST-IN-001 | BLOCK | neither a readable journal record (missing file, bad JSON line, step_index out of range) nor a `result` object is given → REFUSED 96 |
| ST-IN-002 | BLOCK | the record's status and exit_code disagree with the doctrine (e.g. status PASS with exit_code 5, or an unknown status) → REFUSED 96: an inconsistent record cannot be explained honestly |

## Failure handling

- No readable journal record and no `result` object: REFUSED (96) naming ST-IN-001 and the read error
  (missing file, bad JSON line, out-of-range `step_index`); nothing is written. Give `journal` (with an
  in-range `step_index`) or a `result` object and re-run.
- The record's status and exit_code disagree with the doctrine: REFUSED (96) naming ST-IN-002; fix the
  record so its status matches its exit code (PASS 0, RED 5, UNMEASURED 95, REFUSED 96) and re-run.
- The explained status is RED, UNMEASURED or REFUSED: the explanation still returns (exit 0) and does
  not hide the bad status — the `status`, `meaning`, `findings`/`refusal` and `next_step` say what failed
  or is missing and to re-run the owning skill. Never treat UNMEASURED as PASS.
- This skill explains only. On the owning skill's failure, call that skill's `diagnose()` (or
  `TrainingPlannerSkill.diagnose_symptom(name)`); do not improvise a fix here.

## FS interface

Reads only a `journal.jsonl` record or a result object (`fskills status` lists them); computes nothing
else and declares no APIs. This skill invokes no other skill, never imports or launches FoundationScale,
writes only its `status_explanation` artifact and touches no cluster resource. It requires no
FoundationScale (`requires_fs` is False): on a host with only `foundationskills` installed it still
explains a record or result. On a host where even the `foundationskills` package is missing, report
REFUSED with the install step (`pip install -e <FoundationScale repo>/FoundationSkills`) and never
improvise an explanation.

## Worked examples
