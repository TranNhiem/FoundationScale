---
name: fskills-probe-first
license: MIT
description: 'Refuses a FoundationScale plan draft that was written before the measured probe (fskills probe --deep) or that assumes a capability the probe reports missing/unmeasured, and emits a vetted probe_report (measured capabilities, provenance, deep/partial flag). Do NOT use for building datasets (use fskills-data-engine), training plans (fskills-training) or checkpoint scoring (fskills-evaluation). Does not probe FoundationScale itself and never invents a capability.'
compatibility: "Needs Python>=3.10 and the foundationskills package (fskills CLI). It checks an existing probe JSON; producing one needs FoundationScale importable. With no probe JSON it REFUSES (96) and names the probe step instead of guessing."
metadata:
  version: "0.1.0"
  invocation: "model"
  fs_status_doctrine: "PASS/RED/UNMEASURED/REFUSED (exit 0/5/95/96)"
  when_to_use: '["Before I trust this plan draft, check it against the measured probe","Did the probe actually measure the parallel axes this plan needs?","We skipped the probe and wrote a plan anyway — what breaks?","Turn this probe output into a vetted probe_report bound to plan q27"]'
---

# probe_first — FoundationSkills probe-first plan vetting

## Purpose

Refuse before plan: a FoundationScale plan draft is only as good as the measured
probe it was written against. This skill gates a plan draft (`training_plan` payload
or its artifact envelope) on the measured probe JSON that `fskills probe [--deep] --json`
printed (`FSCapabilities.to_dict()`), and emits a vetted `probe_report` whose every
capability is attributable to that measurement. The skill never probes
FoundationScale itself and never invents a capability: absence of evidence is
never PASS.

## When to use

- Before any plan draft is trusted or handed downstream ("check it against the measured probe").
- When a plan was written before probing, or names parallel axes / RL algorithms the probe did not measure.
- To turn a probe output into a vetted `probe_report` bound to a plan draft (id `artifact_id`).
- Not for building datasets (use fskills-data-engine), writing training plans (fskills-training) or checkpoint scoring (fskills-evaluation).

## Inputs

| Field | Required | Meaning |
|---|---|---|
| `plan_draft` | yes | path to a plan draft: `training_plan.<id>.json` (artifact envelope) or a bare payload JSON |
| `probe_report` | yes in practice (PF-IN-001 refuses without it) | path to the `fskills probe [--deep] --json` output (a dict with `available`) |
| `checks` | one of `checks` or the plan's | array of check objects (keys accepted by `FSCapabilities.check`: `stage`, `algorithm`, `backend`, `tp`, `pp`, `ep`, `cp`, `multi_gpu_rl`, `require_checkpoint`, `answer_kind`); defaults to the plan payload's `checks`, else one `{stage, algorithm}` check per plan `stages` entry (what `fskills plan` emits) |
| `out` | no | directory that gets `probe_report.<id>.json` (default `<workdir>/artifacts`) |
| `artifact_id` | no | id in `probe_report.<id>.json` (default: the plan draft's file stem) |

## Outputs

`probe_report.<id>.json` (artifact type `probe_report`, written atomically through
`core/artifacts.py` with provenance). Payload: `plan_draft`, `checks`
(`[{check, missing}]`, `missing` is `null` only when the measured probe runs that
check), `capabilities` (the probe JSON verbatim — `available`, `backends`,
`executed_axes`, `refused_axes`, `errors`, …) and `verdict` (`PASS`|`UNMEASURED`).
The result carries the same payload plus exit codes 0 PASS, 5 RED, 95 UNMEASURED, 96 REFUSED.

Exit codes: 0 PASS (every check measured and runnable), 5 RED (the emitted report
is missing or altered at handoff), 95 UNMEASURED (the vetted report cannot PASS —
PF-HO-001), 96 REFUSED (nothing is written).

## Running it on GB200

probe_first is a local gate on plan drafts and emits only `probe_report`; it never
submits anything. Produce the probe it consumes on the target estate with
`fskills probe --deep --json` (FoundationScale importable there), hand the output
to this skill, and only then build/confirm the launch spec with fskills-training
(or fskills-evaluation for scoring). A shallow probe (no `--deep`) leaves
`axes_measured: false`, so a plan needing `tp/pp/ep/cp>1` refuses at PF-IN-002.

## Scope

LLM and VLM plans of any family, any stage (pretrain/cpt/sft/preference/rl), full
or PEFT methods, on GB200/H100/local. It checks probe output only; FoundationScale
itself is never imported or executed here.

## Validation rules

Outcome semantics (from `core/contract.py`, the same for every skill): an **input**-phase BLOCK refuses before any work - status REFUSED, exit 96, nothing written. A **handoff**-phase BLOCK does not refuse: the work ran, and the result is RED (exit 5), never PASS. WARN and INFO findings never block; they travel with the result (PF-HO-001 makes the run UNMEASURED, exit 95). Phase by prefix: PF-IN-* rules are input phase, PF-HO-* rules are handoff phase.

| Rule | Severity | Fires when |
|---|---|---|
| PF-IN-001 | BLOCK | `probe_report` is absent, unreadable, or not a measured probe JSON (no `available`) — **the probe was skipped** |
| PF-IN-002 | BLOCK | `FSCapabilities.check` returns a `missing: …` reason for a requested check (unmeasured axis, unrunnable algorithm, refused backend), or the plan draft lists that check as `status: PASS`/`available: true` while it is not measured — **treat UNMEASURED as PASS** |
| PF-IN-003 | BLOCK | no requested checks are named (the plan names none and `checks` is absent) |
| PF-HO-001 | WARN | the vetted report cannot PASS: probe `available` false, `errors` non-empty, or a needed `tp/pp/ep/cp>1` while `axes_measured` is false → UNMEASURED 95 |
| PF-HO-002 | BLOCK | the emitted `probe_report` artifact is missing or altered on disk (`sha256` mismatch) at handoff → RED 5 |

## Failure handling

- No probe JSON: REFUSED (96) naming PF-IN-001 and the probe step (`fskills probe --deep --json`); nothing is written.
- A check the probe cannot run (`missing: …`), claimed available/PASS by the plan: REFUSED (96) with PF-IN-002 and the measured `missing` string as evidence; re-draft the plan against what was measured.
- No checks named anywhere (no `checks`, and the plan has neither `checks` nor `stages`): REFUSED (96) with PF-IN-003.
- Probe unusable-but-present (errors recorded, FS unavailable, needed axis unmeasured): the `probe_report` is still emitted but is UNMEASURED 95 with PF-HO-001 - never PASS.
- The written `probe_report` disappears or is edited before handoff: RED 5 with PF-HO-002; re-run this skill before handing the plan on.

## FS interface

Reads the measured probe JSON printed by `fskills probe [--deep] --json`
(`FSCapabilities.to_dict()`: `available`, `fs_version`, `backends`,
`executed_axes`, `refused_axes`, `axes_measured`, `errors`, …). Producing that
probe needs FoundationScale importable on the machine that ran it. This skill
never imports or launches FoundationScale: measured capabilities per
docs/ARCHITECTURE.md:57-67.

## Worked examples

```python
from pathlib import Path

from foundationskills.core.contract import SkillContext
from foundationskills.skills.probe_first import ProbeFirstSkill

result = ProbeFirstSkill().execute(
    {
        "plan_draft": "drafts/plans/q27.json",          # training_plan payload or envelope
        "probe_report": "artifacts/probe.json",         # fskills probe --deep --json > probe.json
        "checks": [{"stage": "sft", "backend": "fsdp", "pp": 2}],
        "out": "artifacts",
        "artifact_id": "q27",
    },
    SkillContext(Path(".")),
)
print(result.status.value, result.exit_code)  # REFUSED 96 / UNMEASURED 95 / RED 5 / PASS 0
for finding in result.findings:
    print(finding.rule_id, finding.message, finding.evidence.get("missing"))
```

```bash
# Measure first (on the estate you will train on), then vet the plan draft.
fskills probe --deep --json > artifacts/probe.json
```
