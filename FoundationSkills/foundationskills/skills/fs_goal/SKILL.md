---
name: fs-goal
license: MIT
description: "User-invoked slash command: run ONLY when the user types /fs-goal; never choose it on your own. Grilling composer (/fs-goal) for one FoundationSkills goal_spec: takes the intake answers so far ({objective, goal?, domain?, preserve_general?, base_model {name_or_path, size_b?, arch?}, sources [{uri, kind, ...}], hardware {gpu, gpus_per_node, nodes}, data_facts {has_verifiable_answers, has_pairs, has_instructions}, rl_algorithm?}), asks ONLY for the facts intake.missing_facts still names and then one confirm question, and writes one goal_spec artifact only when the frontier is empty AND confirmed is true. Do NOT use to plan, train, launch, fix or explain a run: this skill only collects and confirms intake facts (fskills-training plans). Side-effect free: composes and invokes nothing (invokes is empty) and writes only its own goal_spec."
compatibility: "Needs Python>=3.10, the foundationskills package (fskills CLI) and foundationskills.agent.intake (REQUIRED_FACTS, missing_facts, goal_from_answers). It requires no FoundationScale: it reads the user's intake answers and writes one goal_spec artifact, never imports FoundationScale and never patches (requires_fs is False)."
metadata:
  version: "0.1.0"
  invocation: "user"
  fs_status_doctrine: "PASS/RED/UNMEASURED/REFUSED (exit 0/5/95/96)"
  when_to_use: '["grill me for a goal spec","collect my intake answers into a goal_spec","/fs-goal: here are the answers so far","what else do you need to know before planning","finish the goal intake and write the goal spec"]'
---

# fs-goal — the goal-spec grilling composer (/fs-goal)

## Purpose

The grilling composer: `/fs-goal` turns whatever the user has said so far (the intake `answers`) into one
`goal_spec` artifact, asking ONLY for the facts `foundationskills.agent.intake.missing_facts` still names
(`REQUIRED_FACTS` of the skills in `skills`, default `["training.planner", "data_engine"]`) and then one
confirm question. The grilling ends only when the frontier is empty AND the user confirms (`confirmed:
true`). Every round carries the answers so far, so a grilling can stop mid-way and resume with the merged
`answers` — fs-goal never invents an objective, a base model, data sources or hardware the user did not
state. The frontier is measured against the doctrine in `core/status.py` (PASS 0, RED 5, UNMEASURED 95,
REFUSED 96): a missing fact is never a pass (UNMEASURED 95, FG-HO-001), an unconfirmed complete goal is
never a pass (UNMEASURED 95, FG-HO-002), no `answers` object at all is REFUSED 96 (FG-IN-001), and an
answer `goal_from_answers` rejects is RED 5 (FG-HO-003). Only an empty frontier plus a confirmation writes
the `goal_spec` and returns PASS 0.

## When to use

- "/fs-goal collect my intake answers into a goal_spec" / "grill me for a goal spec" / "what else do you
  need to know before planning?".
- To work out which intake facts are still open, ask exactly those questions (one `recommended` value
  each) and stop only when the frontier is empty and the user confirms the preview.
- To write the one `goal_spec` artifact the planner and the data engine consume.
- Not for planning, training, launching, fixing or explaining a run (plan with `fskills-training`, explain
  with `/fs-status`): this skill collects and confirms intake facts only. Do NOT use it to decide
  anything beyond "these are the facts the user gave, and they confirmed them".

## Inputs

| Field | Required | Meaning |
|---|---|---|
| `answers` | yes | object with the intake answer keys of `goal_from_answers`: `objective`, `capabilities?`, `goal?`, `domain?`, `preserve_general?`, `base_model {name_or_path, size_b?, arch?}`, `sources [{uri, kind, ...}]`, `hardware {gpu, gpus_per_node, nodes}`, `data_facts {has_verifiable_answers, has_pairs, has_instructions}`, `rl_algorithm?` |
| `confirmed` | no (default `false`) | the user's confirmation of the preview; `true` (or an answered `yes`) means the grilling may write the `goal_spec` |
| `skills` | no (default `["training.planner", "data_engine"]`) | which skills' `REQUIRED_FACTS` define the frontier |
| `record_dir` | no (default `ctx.artifacts_dir`) | where the `goal_spec` artifact is written |

## Outputs

One `goal_spec` artifact (typed `goal_spec`, the existing goal_spec type, with provenance) — written ONLY
when the frontier is empty AND `confirmed` is true, to `record_dir` (never to a directory some other skill
owns). SCHEMA: none (it produces the existing goal_spec type). Until then the result payload is the
grilling state: `phase` (`questions` | `reject` | `confirm` | `complete`), `questions` (one
`{key, question, recommended}` per open fact; the confirm question is `{'key': 'confirm', 'question':
'Is this goal correct? Answer yes to confirm.', 'recommended': 'yes'}`), `missing_facts`,
`goal_spec_preview` (on `confirm`), `goal_spec` (on `complete`) and `goal_spec_valid` / `goal_spec_error`
(the FG-HO-003 flag `check_handoff` reads).

Exit codes: 0 PASS (empty frontier, user confirmed, valid goal), 95 UNMEASURED (FG-HO-001: facts missing;
FG-HO-002: unconfirmed), 96 REFUSED (FG-IN-001: no `answers` object). 5 RED fires only as FG-HO-003 from
`check_handoff` on a PASS result whose payload carries `goal_spec_valid: false` — `run()` never returns
RED.

## Scope

Any LLM/VLM goal of any family and stage, for full/LoRA/QLoRA on gb200, dgx-h100 or local hardware. It
reads one `answers` object and `foundationskills.agent.intake`, writes one `goal_spec` artifact, and
composes or invokes nothing (`invokes` is empty — it never runs another skill, so no inner rule id can
leak into its findings).

## Validation rules

Outcome semantics (from `core/contract.py`, the same for every skill): an **input**-phase BLOCK refuses
before any work — REFUSED, exit 96, nothing written. A **handoff**-phase BLOCK fires only from
`check_handoff` on a PASS result and turns it RED (exit 5). A handoff WARN is a finding returned from
`run()` with UNMEASURED (exit 95) and never a PASS. Phase by prefix: FG-IN-* rules are input phase,
FG-HO-* rules are handoff phase.

| Rule | Severity | Fires when |
|---|---|---|
| FG-IN-001 | BLOCK | no `answers` object at all is given (absent, or not an object) → REFUSED 96: fs_goal grills what the user said and invents nothing |
| FG-HO-001 | WARN | load-bearing facts are missing from the answers so far (the grilling frontier is not empty) → UNMEASURED 95 with one question per missing fact |
| FG-HO-002 | WARN | the assembled goal is complete but the user has not confirmed it → UNMEASURED 95 with `goal_spec_preview` and the confirm question |
| FG-HO-003 | BLOCK | `goal_from_answers` raised or the assembled goal_spec is invalid (a source without uri/kind, missing hardware, empty objective) → RED 5, fired by `check_handoff` on the PASS result |

## Failure handling

- No `answers` object: REFUSED (96) naming FG-IN-001 and what is missing; nothing is written. Pass the
  intake answer keys the user has given so far and re-run.
- Frontier not empty: UNMEASURED (95) naming FG-HO-001 with the questions; nothing is written. Answer the
  questions (a `yes` accepts the question's `recommended` value, anything else supplies the value) and
  re-run with the merged `answers`. Never treat UNMEASURED as PASS and never skip a question by inventing
  an answer.
- Frontier empty, not confirmed: UNMEASURED (95) naming FG-HO-002 with the preview; nothing is written.
  Review `goal_spec_preview`, answer `yes` (re-run with `confirmed: true`) or send the corrected answers.
- `goal_from_answers` raised or the goal is structurally invalid: RED (5) naming FG-HO-003, fired from
  `check_handoff` on the PASS result whose payload flags `goal_spec_valid: false`; no `goal_spec` is
  written. Fix the answer the message names and re-run — an answer `goal_from_answers` rejects is a real
  contradiction (e.g. a source without `uri` and `kind`), not something to paper over.
- This skill only collects and confirms facts. On a downstream failure call the owning skill's
  `diagnose()`; do not improvise a plan, a fix or a launch here.

## FS interface

Reads only the request's `answers` and `foundationskills.agent.intake` (`fskills goal` lists the frontier
and the assembled spec); computes nothing else and declares no APIs. It composes and invokes no skill,
never imports or launches FoundationScale, writes only its `goal_spec` artifact to `record_dir` (or
`ctx.artifacts_dir`) and touches no cluster resource. It requires no FoundationScale (`requires_fs` is
False): on a host with only `foundationskills` installed it still grills and writes a `goal_spec`. On a
host where even the `foundationskills` package is missing, report REFUSED with the install step (`pip
install -e <FoundationScale repo>/FoundationSkills`) and never improvise a goal.

## Worked examples

Round 1 — two answers, so the frontier is not empty:

    /fs-goal answers {objective 'reply to support tickets politely', domain 'support'}

Result: UNMEASURED 95 with FG-HO-001 and one question per missing fact (`base_model.name_or_path`,
`base_model.size_b`, `base_model.arch`, `data.sources`, `hardware.gpu`, `hardware.nodes`,
`hardware.gpus_per_node`, `data_facts.*`, `preserve_general`, `rl_algorithm`, `goal`) — nothing written.

Round 2 — the user answers the questions (a `yes` per `recommended`, values otherwise):

    /fs-goal answers {objective 'reply to support tickets politely', domain 'support', goal 'general_chat',
    base_model {name_or_path 'hf/Llama-3.1-8B', size_b 8, arch 'dense'}, sources [{uri 's3://data/tickets.jsonl',
    kind 'chat'}], hardware {gpu 'dgx-h100', gpus_per_node 8, nodes 1}, data_facts {has_verifiable_answers false,
    has_pairs false, has_instructions true}, preserve_general true, rl_algorithm 'none'}

Result: UNMEASURED 95 with FG-HO-002 — `goal_spec_preview` plus the confirm question
(`{'key': 'confirm', ... 'recommended': 'yes'}`); nothing written.

Round 3 — the user confirms:

    /fs-goal confirmed true, record_dir run/records — answers unchanged.

Result: PASS 0, one `goal_spec` artifact written to `run/records`, `phase: complete` in the payload.

Contradiction — the user confirmed but a source has no `kind`:

    /fs-goal confirmed true — answers {…, sources [{uri 's3://data/tickets.jsonl'}]}

Result: `goal_from_answers` raises and `run()` returns PASS carrying `goal_spec_valid: false`, which
`check_handoff` turns into RED 5 naming FG-HO-003; no `goal_spec` is written until the source carries
`uri` and `kind`.
