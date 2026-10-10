---
name: fskills-measured-or-unmeasured
license: MIT
description: 'Rewrites a claim set so any claim without a measured evidence pointer (<artifact path>#<json field>, resolvable on disk) is emitted as UNMEASURED and never as PASS, and seals the corrected claim_set with provenance. Do NOT use to probe FoundationScale (use fskills-probe-first), to score checkpoints (fskills-evaluation) or to confirm a launch (fskills-confirm-before-launch). It never upgrades an unmeasured claim and never invents evidence.'
compatibility: "Needs Python>=3.10 and the foundationskills package. Pure stdlib rewriting over evidence pointers; a pointer to a file that is absent is a downgrade to UNMEASURED (exit 95), not a guess."
metadata:
  version: "0.1.0"
  invocation: "model"
  fs_status_doctrine: "PASS/RED/UNMEASURED/REFUSED (exit 0/5/95/96)"
  when_to_use: '["Mark which of these claims are actually measured and which are UNMEASURED","I only have an eval_report pointer for two of these five claims","Turn this claim set into a corrected claim_set where nothing unmeasured reads as PASS"]'
---

# measured_or_unmeasured — FoundationSkills claim-set evidence gate

## Purpose

Rewrite a claim set so that measured-ness is decided by evidence, not by
assertion: a claim is `MEASURED` only when its evidence pointer —
`<artifact path>#<json field>` — resolves to a measured JSON value on disk.
Every other claim is emitted `UNMEASURED` with a `reason`, and the corrected
claim set is sealed as `claim_set.<artifact_id>.json` with provenance. The
skill never upgrades an unmeasured claim and never invents a measurement.

## When to use

- To mark which of a batch of claims are actually measured and which are UNMEASURED.
- When only some claims have `eval_report` pointers and the rest are asserted from memory.
- To turn a claim set into a corrected `claim_set` where nothing unmeasured reads as PASS.
- Not to probe FoundationScale (fskills-probe-first), score checkpoints
  (fskills-evaluation) or confirm a launch (fskills-confirm-first / fskills-confirm-before-launch).

## Inputs

| Field | Required | Meaning |
|---|---|---|
| `claims` | yes | inline claim payload, ≥1 object: `id`, `claim` (strings), optional `asserted_status` (`PASS\|RED\|UNMEASURED\|REFUSED`), `evidence` (pointer or `null`), `measured` (bool) |
| `out` | no | directory the sealed `claim_set.<artifact_id>.json` is written into (default `<workdir>/artifacts`) |
| `artifact_id` | no | artifact id (default `mu-<sha256_json(claims)[:12]>`) |

**Evidence-pointer syntax.** `<path>#<field>` with exactly one `#`: `path` names
a JSON file on disk (absolute, or relative to the working directory) and `field`
walks nested object keys and array indexes separated by `.`
(`artifacts/eval/q27-sft/eval_report.json#benchmarks.0.score`). A missing or
`null`/absent `evidence` is *no pointer*. Any string is a pointer attempt: a
bare sentence ("we ran it last week"), a file that is not on disk, a field that
does not resolve, or a `null` value on disk is **not a measurement** — the claim
is demoted and MU-IN-001/MU-IN-002 fire. A sealed artifact is an envelope; point
into its payload (`...claim_set.<id>.json#payload.claims.0.status`).

**Exit set 0/95.** 0 PASS only when every claim is `MEASURED`; 95 UNMEASURED as
soon as one claim is demoted. This skill never returns 5/96 for a valid request
(every rule is a WARN), though a request that violates the input schema is
REFUSED (96) before any rewrite and writes nothing.

## Outputs

`claim_set.<artifact_id>.json` (artifact type `claim_set`, written atomically as
`<type>.<id>.json`, with provenance). Payload fields: `verdict`
(`PASS|UNMEASURED`, `worst` over the per-claim statuses) and `claims` — one row
per input claim, rewritten to `status` `MEASURED|UNMEASURED` with a `reason`
(which rule, and why) and the `evidence` pointer it was judged by; `id`, `claim`
and any asserted fields are preserved as history.

## Running it on GB200

Nothing here runs on GB200. `fskills-measured-or-unmeasured` is a pure stdlib
rewrite over claim sets and evidence pointers: it emits no `fs_launch_spec` and
submits nothing. GB200/Slurm work belongs to fskills-probe-first (capability
probes), fskills-training (Slurm training) and fskills-evaluation (GB200 eval
jobs); those skills' reports are what the evidence pointers here should name.

## Scope

Claim sets about LLM or VLM work at any training stage, any family, any
backend: the rewrite is claim-level and hardware-free (local execution).

## Validation rules

Outcome semantics (from `core/contract.py`, the same for every skill): an **input**-phase BLOCK refuses before any work — status REFUSED, exit 96, nothing written. A **handoff**-phase BLOCK never passes: the work ran, and the result is RED (exit 5), never PASS. WARN and INFO findings never block; they travel with the result. Phase by prefix: MU-IN-* rules are input phase. Every MU-IN-* rule is a **WARN** on purpose — this skill's plan-table exit set is 0/95 (PASS/UNMEASURED), and an input BLOCK would force REFUSED (96) while a handoff BLOCK would force RED (5), neither legal here; the WARN travels and the claim is demoted to `UNMEASURED`. The sealed verdict is `worst` over the per-claim statuses (`core/status.py`), so one demoted claim makes the whole claim set 95 UNMEASURED and the result is never a silent PASS.

| Rule | Severity | Fires when |
|---|---|---|
| MU-IN-001 | WARN | a claim asserts `asserted_status: "PASS"` (or `measured: true`) with **no** evidence pointer — treated as UNMEASURED, never a silent PASS |
| MU-IN-002 | WARN | an evidence pointer is malformed (not `<path>#<field>`) or unresolvable to a measured value on disk — the claim is demoted |
| MU-IN-003 | WARN | a claim carries no verdict at all (absence of evidence → UNMEASURED) |

## Failure handling

- Pointer to a file that is not on disk / not JSON / field that does not resolve / `null` value: MU-IN-002, claim `UNMEASURED` (exit 95) — a downgrade, never a guess.
- `asserted_status: "PASS"` (or `measured: true`) with no pointer: MU-IN-001, claim `UNMEASURED`.
- No `asserted_status`, no `measured` flag and no resolvable pointer: MU-IN-003, claim `UNMEASURED`. A pointer that resolves is itself the verdict: that claim is `MEASURED` with no finding.
- Evidence is only ever *downgraded* to UNMEASURED; a demoted claim is still written to the sealed `claim_set` with its `reason`.
- A request that violates the input schema is REFUSED (96) with no artifact written; an `artifact_id` containing path separators or dots-traversal fails the artifact writer and is reported RED.

## FS interface

Consumes `claim_set` entries and emits `claim_set` entries — a pure rewrite over
evidence pointers (`resolve_evidence` in `measured_or_unmeasured/evidence.py`).
No FoundationScale call, no CLI, no network.

## Worked examples

```python
from pathlib import Path

from foundationskills.core.contract import SkillContext
from foundationskills.skills.measured_or_unmeasured.skill import MeasuredOrUnmeasuredSkill

skill = MeasuredOrUnmeasuredSkill()
result = skill.execute(
    {
        "claims": [
            {"id": "c1", "claim": "sft holds general ability", "asserted_status": "PASS",
             "evidence": "artifacts/eval/q27-sft/eval_report.json#benchmarks.0.score"},
            {"id": "c2", "claim": "sft keeps instruction following", "asserted_status": "PASS"},
        ],
        "out": "artifacts/claims",
    },
    SkillContext(workdir=Path("artifacts/work")),
)
print(result.exit_code, result.payload["verdict"])   # 95 UNMEASURED: c2 is asserted, never measured
for row in result.payload["claims"]:
    print(row["id"], row["status"], row["reason"])    # c1 MEASURED (pointer resolved) / c2 UNMEASURED (MU-IN-001)
```
