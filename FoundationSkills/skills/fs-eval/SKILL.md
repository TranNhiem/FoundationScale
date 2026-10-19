---
name: fs-eval
license: MIT
description: "User-invoked slash command: run ONLY when the user types /fs-eval; never choose it on your own. /fs-eval is the A3.2 user-invoked workflow skill: it takes the evaluation request (checkpoint, run_manifest, base, benchmarks, policy, eval_cache, out, baseline_cache, ...) and forwards it UNCHANGED to the ONE composed model skill `evaluation`, run through foundationskills.skills._workflow, then writes one workflow_record {workflow, steps, outputs, verdict} whose outputs name the eval_report BY PATH. Outcome mapping (PASS 0 / RED 5 / UNMEASURED 95 / REFUSED 96): inner REFUSED -> FE-IN-002 REFUSED; inner RED -> FE-HO-001 RED; inner UNMEASURED -> FE-HO-002 UNMEASURED (never PASS). Do NOT use for fixing a regression, running a bare lm-eval harness or judging launch-readiness: /fs-eval records the evaluation verdict, it never repairs a benchmark or tunes a policy."
compatibility: "Needs Python>=3.10 and the foundationskills package (fskills CLI) plus a FoundationScale install (requires_fs is True): the ONE composed model skill — evaluation — drives FoundationScale's harness; /fs-eval itself only forwards the request, runs it through foundationskills.skills._workflow and writes its own workflow_record."
metadata:
  version: "0.1.0"
  invocation: "user"
  fs_status_doctrine: "PASS/RED/UNMEASURED/REFUSED (exit 0/5/95/96)"
  when_to_use: '["run the evaluation workflow over my checkpoint and give me the verdict","record my eval run in one workflow record","did the evaluation regress past the policy band","wrap the evaluation in a workflow record showing the steps and outputs","my /fs-eval run came back 95 — is that usable?"]'
---

# fs-eval — the evaluation model-skill workflow (/fs-eval)

## Purpose

The one-model-skill recorder: `/fs-eval` forwards an evaluation request (`checkpoint`, `run_manifest`,
`base`, `benchmarks`, `policy`, `eval_cache`, `out`, `baseline_cache`, `device`, ...) **unchanged** to the
composed model skill `evaluation`, run through `foundationskills.skills._workflow`, and writes exactly one
`workflow_record` artifact: `{"workflow": "/fs-eval", "steps": [Step.as_row() ...], "outputs": {"eval_report":
<path|None>}, "verdict": "PASS|RED|UNMEASURED"}`. The composed skill's `eval_report` is referenced **BY PATH**
in `outputs` and never copied. The verdict follows the doctrine in `core/status.py` (PASS 0, RED 5,
UNMEASURED 95, REFUSED 96): the inner REFUSED maps to FE-IN-002 (REFUSED 96, nothing written), the inner RED
to FE-HO-001 (RED 5) and the inner UNMEASURED to FE-HO-002 (UNMEASURED 95 — never PASS).

## When to use

- "Run the evaluation workflow over /models/q3 and give me the workflow record: verdict, steps and where the
  eval_report landed."
- "Record my evaluation run in one workflow record — I want the verdict, not a rebuilt report."
- "Did the evaluation regress past the policy band?" / "My /fs-eval run came back 95, is that usable?"
- Not for fixing a regression, running a bare lm-eval harness, or judging launch-readiness: repair with the
  owning `evaluation` skill's `diagnose()`, launch with the planner's skill (`fskills-training`) — record
  here, act there.

## Inputs

The evaluation request is forwarded **unchanged** (any field the `evaluation` skill accepts; the schema is
open — `additionalProperties: True`). The fields that matter:

| Field | Required | Meaning |
|---|---|---|
| `checkpoint` | one of `checkpoint` / `run_manifest` | checkpoint directory the composed evaluation scores |
| `run_manifest` | one of `checkpoint` / `run_manifest` | run manifest naming the checkpoint |
| `eval_cache` / `out` / `base` / `benchmarks` / `policy` / ... | per `evaluation` | passed through untouched (`evaluation` rules EV-IN-001..EV-IN-008 police them) |

Nothing else is required by `/fs-eval`: it adds no knob, it subtracts none.

## Outputs

One `workflow_record` artifact (typed, schema-validated, with provenance) per run, payload `workflow`
(`"/fs-eval"`), `steps` (one `Step.as_row()` per composed run: `skill`, `status`, `exit_code`, `refusal`,
`artifacts`, `findings` — the inner rows live here, never as fs_eval findings), `outputs`
(`{"eval_report": "<path>" | null}`, referencing the `evaluation` artifact BY PATH), `verdict` (`"PASS"`,
`"RED"` or `"UNMEASURED"`; REFUSED writes nothing at all). Exit codes: 0 PASS, 5 RED (FE-HO-001), 95
UNMEASURED (FE-HO-002), 96 REFUSED (FE-IN-001 / FE-IN-002).

## Scope

Every evaluation result, for LLM and VLM work of any family and stage (the scope is identical to
`fs-status`'s). It forwards one request, runs one model skill and writes one record — it never imports
FoundationScale itself, never patches a cluster and never copies the composed skill's artifacts.
Composed skills: exactly **one model skill, `evaluation`** (whose own rules EV-IN-001..EV-HO-003 travel here
only inside `evidence["steps"]` of fs-eval's rules — an undeclared rule id would turn this result RED).
Composition rule: a user skill composes model skills, never another user skill.

## Validation rules

Outcome semantics (from `core/contract.py`, the same for every skill): an **input**-phase BLOCK refuses before
any work — status REFUSED, exit 96, nothing written. WARN and INFO findings never block; they travel with the
result. Phase by prefix: FE-IN-* rules are input phase, FE-HO-* rules are handoff phase.

| Rule | Severity | Phase | Fires when |
|---|---|---|---|
| FE-IN-001 | BLOCK | input | neither `checkpoint` nor `run_manifest` is given (fires from `{}` alone) → REFUSED 96 |
| FE-IN-002 | BLOCK | input | the composed `evaluation` REFUSED (its `refusal` names what is missing) → REFUSED 96, no `workflow_record` |
| FE-HO-001 | BLOCK | handoff | `evaluation` RED: a regression beyond the policy band or a broken `eval_report` (`check_handoff` reads `payload["red_step"]`) → RED 5 |
| FE-HO-002 | WARN | handoff | `evaluation` UNMEASURED (a benchmark or the held-out hit was not measured) → UNMEASURED 95, never PASS |

## Failure handling

- Neither `checkpoint` nor `run_manifest`: REFUSED (96) naming FE-IN-001 before any work; nothing is written.
  Pass one of them (plus the evaluation fields) and re-run.
- The composed `evaluation` REFUSED (FE-IN-002): REFUSED (96) naming FE-IN-002 and the evaluation's `refusal`;
  **no `workflow_record` is written** (a refused run records no report). Give the evaluation what its refusal
  names and re-run `/fs-eval`.
- `evaluation` RED (FE-HO-001): the `workflow_record` is written with `verdict: "RED"` and the result is
  RED (5) — the `eval_report` path is in `outputs`, but the output must not be used. Fix what the evaluation's
  findings name and re-run that evaluation through `/fs-eval`.
- `evaluation` UNMEASURED (FE-HO-002): the record is written with `verdict: "UNMEASURED"` and the WARN finding
  travels with the result (95). Never treat UNMEASURED as PASS; resolve what the findings name and re-run.
- A failing composed skill is diagnosed by that skill's own `diagnose()` (e.g. the evaluation's EV-IN-* /
  EV-HO-* names); `/fs-eval` records, it does not improvise a fix.

## FS interface

One CLI entry (`fskills eval`) plus the composed `evaluation` skill, which drives FoundationScale's harness
(`requires_fs` is True: with no FoundationScale install, the composed evaluation REFUSES and `/fs-eval`
returns REFUSED 96 naming FE-IN-002 — it never launches FoundationScale or a cluster job itself). No APIs are
declared; nothing but the `workflow_record` artifact and the `eval_report` path reference is touched. Reads and
writes no cluster resource of its own.

## Worked examples

1. `/fs-eval` with `{}` → REFUSED 96 naming **FE-IN-001**: neither `checkpoint` nor `run_manifest` is given;
   nothing written. Re-run with `{"checkpoint": "/models/q3", "eval_cache": "/eval/cache", "out": "/out/q3"}`.
2. `evaluation` comes back PASS 0 with an `eval_report` at `/out/q3/eval_report.json` → `/fs-eval` writes
   `workflow_record` `{"workflow": "/fs-eval", "steps": [{"skill": "evaluation", "status": "PASS", ...}],
   "outputs": {"eval_report": "/out/q3/eval_report.json"}, "verdict": "PASS"}` and exits 0; hand the path on.
3. `evaluation` comes back RED 5 (EV-HO-001: GSM8K breached its regression tolerance) → `/fs-eval` writes the
   record with `verdict: "RED"` and returns RED 5 naming **FE-HO-001**; the eval_report path is recorded but is
   not usable output.
4. `evaluation` comes back UNMEASURED 95 (EV-HO-003: a benchmark was unmeasured under a smoke limit) →
   `/fs-eval` writes the record with `verdict: "UNMEASURED"`, returns UNMEASURED 95 naming **FE-HO-002** and
   never calls it a pass.
