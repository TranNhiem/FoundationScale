---
name: fskills-refusal-surface
license: MIT
description: 'Turns RED/REFUSED findings into a sealed finding_set whose every entry cites a rule id that a registered skill actually declares (undeclared or invented ids become RED citing RS-HO-001, unattributed findings are cited as RS-HO-002). Do NOT use to produce findings (fskills-evaluation, fskills-probe-first), to rewrite claims (fskills-measured-or-unmeasured) or to confirm launches (fskills-confirm-before-launch); to explain why one skill refused or which of its rules fired, load that skill instead. It never invents a rule id and never silently drops a finding.'
compatibility: "Needs Python>=3.10 and the foundationskills package. Pure attribution over findings; the declared-id set is the skills registered in this install plus this skill's own rules."
metadata:
  version: "0.1.0"
  invocation: "model"
  fs_status_doctrine: "PASS/RED/UNMEASURED/REFUSED (exit 0/5/95/96)"
  when_to_use: '["Make these REFUSED findings cite the rule ids that actually exist","Someone wrote rule id MADE-UP-1 in the failure output","Give me a finding_set I can file where every entry is traceable to a SKILL.md rule"]'
---

# refusal_surface — FoundationSkills finding attribution

## Purpose

Turn RED/REFUSED findings into a sealed `finding_set` whose every entry cites a
rule id that a registered skill actually declares. Attribution is pure and
deterministic over a list of findings: no FoundationScale call, no GPU, no job is
emitted. An entry that cites an id nobody declares is an **invented** id and makes
the whole sealing RED (RS-HO-001); an entry with no usable id is attributed as
RS-HO-002 instead of being dropped. The skill **never invents a rule id** (an
invented id lives only in `evidence.source_rule_id` and in payload data) and
**never silently drops a finding**.

## When to use

- When a run's failure output must be filed as a `finding_set` in which every entry is traceable to a SKILL.md rule.
- When someone wrote a rule id that is not written down anywhere (an invented id such as `MADE-UP-1`) and you need to know what happens to it.
- When a finding lost its id and you want it attributed (RS-HO-002) rather than dropped.
- NOT to produce findings (fskills-evaluation, fskills-probe-first), NOT to rewrite claims (fskills-measured-or-unmeasured), NOT to confirm launches (fskills-confirm-before-launch).

## Inputs

| Field | Required | Meaning |
|---|---|---|
| `findings` | yes | list of finding objects; per entry only `message` (string ≥1) is required, plus optional `rule_id` (string), `severity` (`BLOCK`/`WARN`/`INFO`), `evidence` (object), `recovery` (string). No `minItems`: emptiness must reach the input check (RS-IN-001) |
| `source_skill` | no | skill that produced the findings; recorded as `source_skill` in the sealed set (default `""`) |
| `result_status` | no | status of the run the findings came from (`PASS`/`RED`/`UNMEASURED`/`REFUSED`), recorded as `result_status` (default `REFUSED`, findings arrive from a failed run) |
| `out` | no | directory to write `finding_set.<artifact_id>.json` into (a `*.json` path uses its parent); relative paths resolve under the workdir. Default `<workdir>/artifacts` |
| `artifact_id` | no | id of the sealed artifact; default `rs-<sha256_json(findings)[:12]>` (content id of the normalized entries) |

## Outputs

`finding_set.<artifact_id>.json` (artifact type `finding_set`, written atomically, with provenance) and the
same payload on `result.payload`: `source_skill`, `result_status`, `findings`. Every **normalized entry**
has the shape `{"rule_id", "severity", "message", "evidence", "recovery", "cited_by"}`:

- `rule_id` is **always a declared id**: the original when the finding cites one, else `RS-HO-001` (an
  invented/undeclared id) or `RS-HO-002` (an unattributed or malformed finding). `evidence.source_rule_id`
  keeps what arrived (never a dropped finding), so the original id survives on the entry that cites it.
- `severity` is passed through, and normalized to `BLOCK` when what arrived is unknown.
- `cited_by` names the registered skill whose SKILL.md declares `rule_id` (`refusal_surface` for `RS-*`),
  so every entry is traceable to a SKILL.md rule.

Exit codes (this skill's exit set: 0/5/96): 0 PASS (every entry attributed; unattributed or malformed
findings carry an RS-HO-002 WARN), 5 RED (an id was invented — RS-HO-001; the set is still sealed and
written, the sealed entry cites `RS-HO-001`), 96 REFUSED (nothing to surface — RS-IN-001, no artifact).

## Running it on GB200

Nothing to run on GB200: this skill is "attribution only; no FoundationScale call"
(`fs_interface.apits == ()`, it emits a `finding_set` and no `fs_launch_spec`), so it
emits and submits no job. File the sealed `finding_set.<artifact_id>.json` into whatever
chain needs a traceable failure surface; `fskills` has no emit/launch step here.

## Scope

Any LLM/VLM family and every training stage (pretrain/cpt/sft/preference/rl), full or
adapter work on GB200/H100/local — the scope comes with the findings, not with this
skill. Pure attribution over a finding list; it needs no dataset, checkpoint, probe
report or confirmation record.

## Validation rules

Outcome semantics (from `core/contract.py`, the same for every skill): an **input**-phase BLOCK refuses
before any work — status REFUSED, exit 96, nothing written. A **handoff**-phase BLOCK does not refuse:
the work ran, and the result is RED (exit 5), never PASS. WARN and INFO findings never block; they travel
with the result. Every finding must use `self.finding(rule_id, …)` with a declared id. **An undeclared-rule
finding (a `Finding` whose `rule_id` no skill declares) is appended as `CORE-UNDECLARED-RULE` and forces
RED** (`docs/ARCHITECTURE.md:31`) — this skill therefore **never emits a `Finding` with an undeclared id**:
invented ids live only in `Finding.evidence` (`invented_rule_ids`, `source_rule_id`) and in the sealed
payload data. Phase by prefix: RS-IN-* rules are input phase, RS-HO-* rules are handoff phase.

| Rule | Severity | Fires when |
|---|---|---|
| RS-IN-001 | BLOCK | `findings` is empty or not a finding set — nothing to surface (refuse rather than emit an empty surface); no artifact is written |
| RS-HO-001 | BLOCK | an incoming finding cites a rule id **no registered skill declares** (e.g. `PF-IN-999`, `made-up-id`) — invent rule ids ⇒ RED (exit 5); also when the sealed entries cannot be shown to cite declared ids at handoff |
| RS-HO-002 | WARN | an incoming finding is unattributed or malformed — has no usable rule id, or an unknown severity: attributed in the emitted set (id cited as `RS-HO-002`; unknown severity normalized to `BLOCK`) |

## Failure handling

- Empty/invalid finding list: REFUSED (exit 96) naming RS-IN-001; no `finding_set` is written.
- Invented rule id: RED (exit 5) with one RS-HO-001 finding carrying `evidence.invented_rule_ids`; the
  sealed entry cites `RS-HO-001` and keeps the original in `evidence.source_rule_id`. Recover by citing
  the declared id or declaring it with a MUST_FIRE fixture, then reseal.
- Unattributed/malformed finding (no usable rule id, or an unknown severity): PASS (exit 0) with an
  RS-HO-002 WARN; the emitted entry cites `RS-HO-002` and the finding is kept (attributed), never dropped.
- Undeclared-rule findings: they force RED via `CORE-UNDECLARED-RULE` (`docs/ARCHITECTURE.md:31`), which is
  why this skill only ever emits RS-IN-001/RS-HO-001/RS-HO-002.

## FS interface

None. `entries=()`, `emits=("finding_set",)`, `apis=()`. Notes: "attribution only; no FoundationScale
call". It consumes finding lists (RED/REFUSED findings from any skill) and emits one `finding_set`
artifact; it never calls FoundationScale, reads a checkpoint or submits anything.

## Worked examples

```python
from pathlib import Path

from foundationskills.core.contract import SkillContext
from foundationskills.skills.refusal_surface import RefusalSurfaceSkill

request = {
    "findings": [{"rule_id": "MADE-UP-1", "severity": "BLOCK", "message": "failure output with an invented id"}],
    "result_status": "REFUSED",
}
result = RefusalSurfaceSkill().execute(request, SkillContext(workdir=Path("artifacts/rs-demo")))
assert result.exit_code == 5                               # RED: an invented id -> RS-HO-001
entry = result.payload["findings"][0]
assert entry["rule_id"] == "RS-HO-001"                     # the entry cites a declared id ...
assert entry["evidence"]["source_rule_id"] == "MADE-UP-1"  # ... and still keeps what arrived
assert all(f.rule_id != "MADE-UP-1" for f in result.findings)  # never a Finding with an undeclared id
print(result.artifacts[0].path)                            # artifacts/rs-demo/finding_set.rs-<hash>.json
```
