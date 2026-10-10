---
name: fskills-confirm-before-launch
license: MIT
description: 'Gates an fs_launch_spec behind the user-confirmed plan hash: it recomputes the plan hash (`fskills hash --spec`, the same token `fskills launch --confirm` requires) and REFUSES (96) when the user-supplied hash is absent or mismatches, sealing a confirmation_record with confirmed_by=user and auto_filled=false when they match. Do NOT use to write plans (fskills-training), run probes (fskills-probe-first) or score checkpoints (fskills-evaluation). It never submits a job, never prints or fills the expected hash for a caller to copy.'
compatibility: "Needs Python>=3.10 and the foundationskills package (fskills CLI hash/launch). Confirmation is local and deterministic; submitting still happens only through fskills launch --spec --confirm."
metadata:
  version: "0.1.0"
  invocation: "model"
  fs_status_doctrine: "PASS/RED/UNMEASURED/REFUSED (exit 0/5/95/96)"
  when_to_use: '["Confirm this fs_launch_spec against the hash the user read back to me","Before launching, verify the plan hash matches the spec on disk","We launched without re-showing the hash — what rule blocks that?","Seal the confirmation record for spec q27"]'
---

# confirm_before_launch — FoundationSkills launch confirmation

## Purpose

Gate an `fs_launch_spec` behind the plan hash the user read back. The skill
recomputes the plan hash (`fskills hash --spec`, the same token `fskills launch
--confirm` requires) from the spec payload and REFUSES (96) when the
user-supplied hash is absent (CB-IN-001) or does not match what is on disk
(CB-IN-002, CB-IN-003), or when the spec drifted before the confirmation record
was sealed (CB-HO-001). On a match it seals a `confirmation_record` with
`confirmed_by: user` and `auto_filled: false`, carrying `spec_sha256`
(`sha256_file` of the spec file) as a separate drift seal. The skill never
submits a job and never prints or fills the expected hash for a caller to copy.

## When to use

- When the user read back the plan hash and the confirmation must be sealed before `fskills launch`.
- Before launching: verify the plan hash matches the spec on disk (and the spec did not drift since the hash was shown).
- When asked which rule blocks launching without re-showing the hash (CB-IN-001).
- Not for writing plans (fskills-training), probing capabilities (fskills-probe-first), scoring checkpoints (fskills-evaluation) or submitting a job (that is `fskills launch --confirm` alone).

## Inputs

| Field | Required | Meaning |
|---|---|---|
| `launch_spec` | yes | path of the `fs_launch_spec` JSON (an artifact envelope unwrapped like `cli._load_payload`, or a bare payload); a relative path resolves against the workdir |
| `user_plan_hash` | yes — the skill never fills it | the plan hash the user read back (16 hex, what `fskills hash --spec` prints). Absent/empty is CB-IN-001; a mismatch is CB-IN-002 (an empty string is deliberately not blocked by the schema) |
| `confirmation_id` | no | optional correlation id, recorded in provenance |
| `out` | no | directory the record is written into (default `<workdir>/artifacts`); a `.json` value uses its parent |
| `artifact_id` | no | record/artifact id (default `cb-<spec_sha256[:12]>`) |

## Outputs

`confirmation_record.<artifact_id>.json` (artifact type `confirmation_record`,
written atomically): `launch_spec`, `spec_sha256` (64 hex `sha256_file` of the
spec file, a separate drift seal), `confirmed_hash` (the user's verbatim input,
16 hex), `confirmed_by: "user"`, `auto_filled: false`, plus provenance (skill
and versions, `spec_sha256`, timestamp).

Exit codes: 0 PASS (the record is sealed), 96 REFUSED (no record is written).
This skill's exit set is 0/96: it either confirms or refuses. The *recomputed*
hash is never echoed — only the hash the user themselves supplied is stored.

## Running it on GB200

Only local, deterministic hashing happens here — no GPU is touched. The gate
stands exactly between the two CLI steps of a GB200 (or any) launch: the user
runs `fskills hash --spec <spec>` and reads the 16-hex token aloud; this skill
seals `confirmation_record.<id>.json`; then `fskills launch --spec <spec>
--confirm <token>` is the only submission path and re-hashes the payload the
same way (`require_confirmation`), refusing anything but that exact token.
Submission never happens through this skill.

## Scope

Any `fs_launch_spec` produced for FoundationScale work (training or `fskills
eval` jobs), local/GB200/H100, LLMs of any family, any training stage, full or
adapter runs. The scope is the confirmation gate only: no planning, probing,
scoring, or submitting.

## Validation rules

Outcome semantics (from `core/contract.py`, the same for every skill): an **input**-phase BLOCK refuses before any work - status REFUSED, exit 96, nothing written. A **handoff**-phase BLOCK does not refuse: the work ran, and the result is RED (exit 5), never PASS. WARN and INFO findings never block; they travel with the result. Phase by prefix: CB-IN rules are input phase, CB-HO rules are handoff phase. CB-HO-001 is *detected* in `run`, at the record seal (a drift before the seal REFUSES with 96 and writes no record), and re-checked at handoff over `sha256_file`; the exit set is 0 PASS / 96 REFUSED.

| Rule | Severity | Fires when |
|---|---|---|
| CB-IN-001 | BLOCK | `user_plan_hash` is absent or empty — launching without re-showing the hash is refused, and the skill never auto-fills it |
| CB-IN-002 | BLOCK | the supplied hash is not the plan hash recomputed from the spec payload |
| CB-IN-003 | BLOCK | the `fs_launch_spec` is missing, unreadable or not an fs_launch_spec (nothing to hash) |
| CB-HO-001 | BLOCK | the spec drifted on disk between the input recompute and the record seal (the second plan_hash differs); the handoff re-verify compares `sha256_file` with the sealed `spec_sha256` |

## Failure handling

- 96 REFUSED writes no record; the expected hash is never echoed (SKILLS.md:24, README.md:45).
- No user-supplied hash (CB-IN-001) or a mismatch (CB-IN-002): the refusal text only says to obtain the hash again — the never-auto-filled recomputed value stays inside the comparison.
- Unreadable/foreign `launch_spec` (CB-IN-003): nothing is hashed and nothing is sealed; give a readable `fs_launch_spec`.
- Drift before seal (CB-HO-001): REFUSED 96 with no record — re-emit nothing silently; obtain a fresh hash from the user.
- Drift after seal (CB-HO-001 at handoff): the sealed record no longer stands; the confirmation must be repeated before `fskills launch --confirm`.
- Internal errors (a broken artifact schema, an illegal `artifact_id`) surface as refusals or the template's `CORE-*` RED; nothing here submits a job.

## FS interface

Consumes an `fs_launch_spec` (envelope or bare payload, hashed the same way
`fskills hash --spec` hashes it) and emits a `confirmation_record`. Interfaces:
`fskills hash --spec` / `fskills launch --confirm`. It recomputes the plan hash;
never submits, never auto-fills.

## Worked examples

```python
from pathlib import Path

from foundationskills.core.contract import SkillContext
from foundationskills.skills.confirm_before_launch import ConfirmBeforeLaunchSkill

# the user read the hash back themselves; it is passed in verbatim, never computed for them
result = ConfirmBeforeLaunchSkill().execute(
    {"launch_spec": "artifacts/fs_launch_spec.q27.json", "user_plan_hash": hash_read_back_by_the_user},
    SkillContext(workdir=Path("work/")),
)
# 0 PASS seals work/artifacts/confirmation_record.cb-<sha12>.json (confirmed_by=user, auto_filled=false)
# 96 REFUSED writes no record and echoes no expected hash
```

```bash
# the pair this gate stands between: the token is for the user's eyes only
fskills hash --spec artifacts/fs_launch_spec.q27.json
fskills launch --spec artifacts/fs_launch_spec.q27.json --confirm <token the user read back>
```
