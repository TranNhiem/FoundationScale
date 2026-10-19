---
name: fs-handoff
license: MIT
description: "User-invoked slash command: run ONLY when the user types /fs-handoff; never choose it on your own. Slash-command closer (/fs-handoff) for a FoundationSkills session: takes this session's artifact JSON paths, optional notes and next_steps and writes ONE handoff artifact that references every artifact BY PATH ({path, type, id, sha256}, payload never copied) with notes and next_steps redacted of secrets (sk-, ghp_, AKIA, Bearer, NAME=KEY/TOKEN/SECRET/PASSWORD values) and PII (email, phone, credit_card, ssn, national_id). Do NOT use for re-running a skill, fixing findings or deciding what to run next: it hands a session off, it does not act."
compatibility: "Needs Python>=3.10 and the foundationskills package (fskills CLI). It requires no FoundationScale: it reads artifact JSON envelopes written this session and writes one handoff artifact, never imports FoundationScale and never patches (requires_fs is False)."
metadata:
  version: "0.1.0"
  invocation: "user"
  fs_status_doctrine: "PASS/RED/UNMEASURED/REFUSED (exit 0/5/95/96)"
  when_to_use: '["hand off this session to the next one","write a handoff artifact for these artifacts","carry these notes and next steps over to tomorrow","close the session with a redacted handoff record","what do I give the person who picks this up"]'
---

# fs-handoff — the session handoff writer (/fs-handoff)

## Purpose

The wrap-it-up skill: when a session is done and somebody else picks it up next,
`/fs-handoff` turns what this session produced into ONE `handoff` artifact. It reads
each artifact JSON envelope written this session (its `type`, `id`, `provenance` and
the sha256 of the file's bytes) and references it BY PATH — never copies a payload —
then redacts secrets and PII from `notes` and `next_steps` before writing
{artifacts, notes, next_steps, redactions}. It is side-effect free elsewhere: it
composes nothing, invokes nothing and writes only its own artifact. It hands off, it
does not act — it never re-runs a skill, fixes a finding or decides what to run next.
Reading is measured against two input rules: a request with no `artifacts` list is
REFUSED (FH-IN-001 — there is nothing to hand off), an artifact path that is missing
or is not an artifact envelope (no `type`/`payload`) is REFUSED (FH-IN-002) naming the
path. Redaction is reported, never blocking: FH-HO-001 is INFO.

## When to use

- "Hand off this session to the next one" / "Close it out with a handoff record" /
  "Carry these notes and next steps over to tomorrow".
- To reference everything this session produced BY PATH ({path, type, id, sha256}) in
  one artifact the next session can read without re-deriving anything.
- To scrub `notes` and `next_steps` of secrets (sk-, ghp_, AKIA, `Bearer <token>`,
  and NAME=value lines whose NAME holds KEY/TOKEN/SECRET/PASSWORD — NAME is kept, its
  value is redacted) and PII (email, phone, credit_card, ssn, national_id) before
  anything is shared.
- Not for re-running a skill, fixing findings or choosing the next skill (fix with
  the owning skill such as `fskills-probe-first`, explain a result with `/fs-status`);
  hand off here, act there.

## Inputs

| Field | Required | Meaning |
|---|---|---|
| `artifacts` | yes | list of artifact JSON paths from this session; each file must be an artifact envelope (`type` and `payload`) |
| `notes` | no | free text for the next session; redacted before it is written |
| `next_steps` | no | list of strings: what the next session should do first; every entry redacted |
| `record_dir` | no | where to write the `handoff` artifact; default: the run's artifact directory (`ctx.artifacts_dir`) |

## Outputs

One `handoff` artifact (typed, schema-validated, with provenance) whose payload is `artifacts`
(`[{path, type, id, sha256}, ...]` — one row per input artifact, referenced BY PATH with the
sha256 of the file's bytes pinning it; no payload is copied), `notes` (redacted), `next_steps`
(list, redacted) and `redactions` ({kind: count}, kinds with count 0 omitted: `env_value`,
`sk_token`, `ghp_token`, `akia_key`, `bearer_token`, `email`, `phone`, `credit_card`, `ssn`,
`national_id`). It never uses an untracked temp doc. See `SCHEMA` for the JSON schema.

Exit codes: 0 PASS (the handoff itself is measured), 96 REFUSED (FH-IN-001: no `artifacts` list;
FH-IN-002: an artifact path is missing or is not an artifact envelope). FH-HO-001 is INFO and
travels with the PASS result. There is no handoff-phase BLOCK rule, so RED 5 never fires here.

## Scope

Any FoundationSkills session's artifacts, for LLM and VLM work of any family and stage. It reads
artifact JSON envelopes, writes one `handoff` artifact and never runs another skill, imports
FoundationScale or touches a cluster resource. Side-effect free: it invokes nothing (`invokes` is
empty) and composes no slash command.

## Validation rules

Outcome semantics (from `core/contract.py`, the same for every skill): an **input**-phase BLOCK refuses
before any work — status REFUSED, exit 96, nothing written. WARN and INFO findings never block; they
travel with the result. Phase by prefix: FH-IN-* rules are input phase, FH-HO-* is the handoff phase
(read from the PASS payload on `check_handoff`).

| Rule | Severity | Fires when |
|---|---|---|
| FH-IN-001 | BLOCK | no `artifacts` list is given (absent or empty), so there is nothing to hand off → REFUSED 96 |
| FH-IN-002 | BLOCK | an artifact path is missing/unreadable or the file is not an artifact envelope (no `type` and `payload`) → REFUSED 96 naming that path |
| FH-HO-001 | INFO | redactions were applied to `notes`/`next_steps`: the finding carries the per-kind counts; it reports and never blocks (it is returned from `check_handoff` on a PASS result) |

## Failure handling

- No `artifacts` list: REFUSED (96) naming FH-IN-001; nothing is written. Pass the session's
  artifact JSON paths (with `notes`/`next_steps`) and re-run.
- An artifact path is missing or is not an envelope: REFUSED (96) naming FH-IN-002 and the path;
  nothing is written. Point the list at artifacts written this session (JSON envelope with `type`
  and `payload`) and re-run.
- Redactions applied: the handoff still returns (exit 0) and FH-HO-001 reports the counts — the
  redacted `notes`/`next_steps` are what the next session reads. Never leave a secret or PII in
  a handoff on purpose; an INFO finding is a report, not a licence.
- The explained redaction is not a repair: on a real failure of the producing skill, call that
  skill's `diagnose()` (or `TrainingPlannerSkill.diagnose_symptom(name)`); read a bad result with
  `/fs-status`. Do not improvise a fix here.

## FS interface

Reads only artifact JSON envelopes (`fskills status` lists what the session wrote) and writes one
`handoff` artifact. It declares no APIs: it touches no cluster resource, never imports or launches
FoundationScale and invokes no other skill. It requires no FoundationScale (`requires_fs` is False):
on a host with only `foundationskills` installed it still writes the handoff. On a host where even
the `foundationskills` package is missing, report REFUSED with the install step
(`pip install -e <FoundationScale repo>/FoundationSkills`) and never improvise a handoff.

## Worked examples

1. `/fs-handoff` with artifacts `job/artifacts/probe.json`, `job/artifacts/data_report.json`,
   `job/artifacts/skill_result.json`, notes "the curator approved the curated-q2 set",
   next steps ["resume the trainer at step 3"] and `record_dir` `job/records` → PASS 0 that writes
   one `handoff` artifact holding three rows of {path, type, id, sha256}, the notes and next_steps
   untouched and `redactions: {}` — no FH-HO-001, because nothing was redacted.
2. `/fs-handoff` with notes "reached a.chen@example.com after dropping the key sk-live-9f2c44ab77"
   and next steps ["resume the trainer at step 3"] → PASS 0 with `redactions: {secret_sk: 1,
   email: 1}` and FH-HO-001 INFO naming those counts; the written notes read
   "reached [REDACTED:email] after dropping the key [REDACTED:secret_sk]".
3. `/fs-handoff` with artifacts `[job/artifacts/probe.json, job/scratch/raw.json]` where
   `raw.json` is `{"rows": 1200}` → REFUSED 96 naming FH-IN-002 and `job/scratch/raw.json`
   (no `type`/`payload`): a plain JSON file is not an artifact envelope. Nothing is written.
4. Ops text is scrubbed as a whole value: notes "AWS_SECRET_ACCESS_KEY=AKIAEXAMPLEEXAMPLE and the
   plan hash is 3f2a91c0b7de4410" keep the NAME and the hash, and report `env_value: 1`.
