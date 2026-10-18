---
name: fs-research
license: MIT
description: "User-invoked slash command: run ONLY when the user types /fs-research; never choose it on your own. Claim-measurement research workflow (/fs-research): takes an optional goal_spec path and a claims list {id, claim, asserted_status?, evidence?, measured?}, composes measured_or_unmeasured to check every claim, and writes one research_findings artifact {goal_spec, claims (corrected), verdict (PASS|UNMEASURED), steps}. REFUSED (96) with no or empty claims (FR-IN-001) or when the composed step refuses (FR-IN-002); UNMEASURED (95) when some claim stays UNMEASURED (FR-HO-001); RED (5) when the composed step came back RED (FR-HO-002). Do NOT use for training or launching (fskills-training) or re-running the checker alone (fskills-measured-or-unmeasured): /fs-research researches and reports, the trainer launches."
compatibility: "Needs Python>=3.10 and the foundationskills package (fskills CLI). Requires no FoundationScale (requires_fs is False): it composes measured_or_unmeasured through foundationskills.skills._workflow.run_step, writes one research_findings artifact and touches no cluster resource."
metadata:
  version: "0.1.0"
  invocation: "user"
  fs_status_doctrine: "PASS/RED/UNMEASURED/REFUSED (exit 0/5/95/96)"
  when_to_use: '["/fs-research check these claims","research whether my claims were measured","are these claims measured or unmeasured","audit every claim against evidence and give me the verdict"]'
---

# fs-research — the claim-measurement research workflow (/fs-research)

## Purpose

The measurement auditor: when you have a list of claims about a run and you need to know which of them
were actually measured, `/fs-research` feeds them to the model skill `measured_or_unmeasured` (composed
once, through `foundationskills.skills._workflow.run_step`, with exactly `{claims: ...}`), reads its
corrected claims back from its `claim_set` artifact and writes one `research_findings` artifact
`{goal_spec, claims, verdict (PASS|UNMEASURED), steps}`. A claim nobody measured is never a pass.

Every status is written against the doctrine in `core/status.py` (PASS 0, RED 5, UNMEASURED 95,
REFUSED 96). The composed findings carry rule ids this skill does not declare, so they travel only as
EVIDENCE (`steps`, `evidence["steps"]`) and never as this skill's findings. Outcome precedence:
REFUSED (FR-IN-001, FR-IN-002) beats RED (FR-HO-002) beats UNMEASURED (FR-HO-001) beats PASS. The
composed step's REFUSED maps to the input-phase BLOCK rule FR-IN-002 (96), its RED to the handoff-phase
BLOCK rule FR-HO-002 (5, fired from `check_handoff` on the PASS run whose payload flag `composed_red`
the RED set), its UNMEASURED to the WARN rule FR-HO-001 (95).

## When to use

- "Research these claims and tell me the verdict" / "Are these claims measured or unmeasured?" /
  "Audit every claim against evidence for goal_spec ..." typed as `/fs-research`.
- To check a claim list `{id, claim, asserted_status?, evidence?, measured?}` against the evidence and
  report which claims are UNMEASURED and why.
- To record what the claims checker said (its own status, exit code, refusal and findings) alongside
  the corrected claims and one `PASS`/`UNMEASURED` verdict.
- Not for training or launching (`fskills-training`) or for re-running the claims checker standalone
  (`fskills-measured-or-unmeasured`): `/fs-research` researches and reports, the checker checks, the
  trainer launches.

## Inputs

| Field | Required | Meaning |
|---|---|---|
| `goal_spec` | no | path to the goal spec this research answers to; recorded in `research_findings.goal_spec` |
| `claims` | **yes** (non-empty) | claim objects exactly as `measured_or_unmeasured` takes them: `{"id", "claim", "asserted_status"?, "evidence"?, "measured"?}` |
| `record_dir` | no | where to write the `research_findings` artifact; default: the context artifacts dir |

## Outputs

One `research_findings` artifact (typed, schema-validated, with provenance) written to `record_dir`
(never to a composed skill's key). Payload: `goal_spec` (path or null), `claims` (the composed skill's
corrected claims, read back from its `claim_set` artifact through `_workflow.load_payload`), `verdict`
(`"PASS"` only when every corrected claim is measured, else `"UNMEASURED"`), `steps` (one row per
composed run: skill, status, exit_code, refusal, artifacts, findings), `composed_status`, and the flag
`composed_red` that `check_handoff` reads. See `SCHEMA` for the JSON schema.

Exit codes: 0 PASS (every claim measured), 5 RED (FR-HO-002, only from `check_handoff`), 95 UNMEASURED
(FR-HO-001, never a pass), 96 REFUSED (FR-IN-001, FR-IN-002) which writes nothing.

## Scope

Claim sets for LLM and VLM work of any family and stage. It composes only `measured_or_unmeasured` (one
`run_step` call, never another user skill), reads only that step's `claim_set` artifact and writes only
its own `research_findings` artifact. It never imports FoundationScale, never patches and touches no
cluster resource (`requires_fs` is False).

## Validation rules

Outcome semantics (from `core/contract.py`, the same for every skill): an **input**-phase BLOCK refuses
before any work — REFUSED 96, nothing written. A WARN finding never blocks; it travels with the result
and makes it UNMEASURED 95. A handoff-phase BLOCK fires only from `check_handoff` on a PASS result.

| Rule | Severity | Fires when | Result |
|---|---|---|---|
| FR-IN-001 | BLOCK input | no `claims` list is given, or the list is empty | REFUSED 96, nothing written |
| FR-IN-002 | BLOCK input | the composed `measured_or_unmeasured` step refused (its findings travel in `evidence["steps"]`) | REFUSED 96, nothing written |
| FR-HO-001 | WARN handoff | some corrected claim is still UNMEASURED (or the composed step itself is) | UNMEASURED 95, `verdict` is `"UNMEASURED"` |
| FR-HO-002 | BLOCK handoff | the composed step came back RED (payload flag `composed_red`) | RED 5 from `check_handoff`, `research_findings` must not be used |

## Failure handling

- No `claims`, or an empty list: REFUSED (96) naming FR-IN-001 with the claim count; nothing is written.
  Pass a non-empty `claims` list and re-run.
- The composed `measured_or_unmeasured` step refused: REFUSED (96) naming FR-IN-002; the reason is its
  own finding message, copied into `evidence["steps"]` and never restated as this skill's finding. Give
  the checker the input its refusal names and re-run `/fs-research`.
- Something was left UNMEASURED: UNMEASURED (95) naming FR-HO-001 and the claim ids; the
  `research_findings` artifact is still written for the audit. Measure what the named claims assert and
  re-run — never treat UNMEASURED as PASS.
- The composed step came back RED: RED (5) naming FR-HO-002; `research_findings` must not be used. Read
  the findings in `evidence["steps"]`, fix them with that skill and re-run `/fs-research` — never hand
  RED `research_findings` on.
- For the owning skill's own failure call its `diagnose()` (or
  `TrainingPlannerSkill.diagnose_symptom(name)`); never improvise a fix here.

## FS interface

Composes `measured_or_unmeasured` through `foundationskills.skills._workflow.run_step` (`fskills research`
lists the entry) and declares no APIs. This skill never imports or launches FoundationScale, writes only
its `research_findings` artifact and touches no cluster resource. It requires no FoundationScale: on a
host with only `foundationskills` installed it still audits a claim list. On a host where even the
`foundationskills` package is missing, report REFUSED with the install step (`pip install -e
<FoundationScale repo>/FoundationSkills`) and never improvise a verdict.

## Worked examples

1. `/fs-research goal_spec goals/measure_loss.goal.json, record into out/research: c1 'the train loss fell
   after step 2k' measured true evidence 'runs/m18/train_loss.json', c2 'the eval set was deduplicated'
   measured false.` → composes `measured_or_unmeasured` with the two claims, reads the corrected claims
   from its `claim_set` artifact, writes `research_findings` to `out/research` and returns UNMEASURED 95
   FR-HO-001 with `verdict: "UNMEASURED"` naming c2.
2. `/fs-research I have goal_spec goals/sft.goal.json but no claims yet.` → REFUSED 96 FR-IN-001 (no
   `claims` list): nothing runs, nothing is written, and the message asks for a non-empty claim list.
3. `/fs-research measured_or_unmeasured came back RED on my claims in out/research (goal_spec
   goals/q2.goal.json), FR-HO-002 fired as RED 5 — re-run and report.` → the composed status is copied
   into `research_findings.steps`, `composed_red` is true, `check_handoff` fires FR-HO-002 and the result
   is RED 5 with the findings in `evidence["steps"]`.
