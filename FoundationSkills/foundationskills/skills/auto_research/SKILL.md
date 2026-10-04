---
name: fskills-auto-research
license: MIT
version: 0.1.0
description: 'Manages a hash-chained auto-research campaign (M0): validates the campaign spec, opens one
  human approval envelope (campaign_confirm hash), authorises launches with derived tokens, records immutable
  trial results, proposes ideas from the ideas catalog and closes with statistical acceptance against a
  calibrated noise floor (guardrails, keep/discard/crash TSV view) writing auto_research_report. Do NOT use
  for single training runs (fskills-training), nor for evaluation alone (fskills-evaluation), nor (M1, not
  built) for actually submitting jobs to a cluster.'
when_to_use:
- Run an auto research campaign and find a better recipe for val_accuracy on this base model
- Validate my campaign spec (axes, budget, eval policy) before I sign the approval hash
- Authorise a candidate run and give me its derived launch token
- Record these trial results and decide accepted_gain / accepted_flat / no_gain / rejected_regress / unmeasured
- Propose the next three hypotheses from the ideas catalog given these symptoms
- Close the campaign and give me the auto_research report with the ledger TSV view
---

# Auto Research

## Purpose
Drive a bounded experiment campaign against one stated objective (metric + direction) without ever
trusting a claim the evidence cannot support. The skill validates a `campaign_spec`, opens ONE human
approval envelope (`campaign_confirm` = `campaign_hash(spec)`), authorises individual launches with
derived per-launch tokens, records immutable trial results into an append-only hash-chained ledger,
proposes concrete hypotheses (idea cards), and closes with a statistical verdict against a calibrated
noise floor. M0 launches nothing: job submission is M1.

## When to use
- You have a goal ("improve val_accuracy") and want a measured loop of baseline -> propose -> run ->
  record -> decide -> close with an auditable evidence chain.
- You need one approval hash to bound a whole campaign plus derived, per-launch authorisation tokens.
- You need acceptance statistics (tau bands, guardrails) rather than "the last run looked better".

Do NOT use it for a single training run (use the training skill), for evaluation alone (use the
evaluation skill), or for bug fixes/refactors (no FoundationSkills skill should take those).

## Inputs
Request object (all fields optional except those the action needs):
- `action`: `check | launch | record | propose | close`.
- `campaign_spec`: the campaign spec dict (see below), `campaign_confirm`: its human-approved hash,
  `approver`: the human who approved it (required on the first action of a ledger).
- `ledger_dir`: campaign ledger directory (chain.jsonl + objects/); defaults to `<workdir>/ledger`.
- `launch_spec`: one authorised run (launch action), `result`: one trial result (record action).
- `current`: current knob values for the proposer, `symptoms`: observed symptoms (propose action).
- `stop_reason`: why the campaign is being closed (close action).

Spec shape (the object a human hashes and approves):
`{id, objective{metric, direction(max|min), benchmarks[]}, eval_policy{fingerprint: sha256:<hex>, metrics[]},
base{model, fingerprint: sha256:<hex>}, confirm{k, noise_floor_rel, guardrails[], guard_abs_epsilon},
seeds{baseline_repeats, confirm_repeats, seed_list[]}, budget{gpu_hours_total, max_runs, per_run_timeout_h,
reserve_frac}, stopping{no_gain_streak, max_crash_streak}, axes[{key, type, min, max, values}],
cluster{partition, max_nodes, gpus_per_node, time: "10-00:00:00", exclude[]}}`.
Axes are the measured FoundationSkills knobs (`optim.lr`, `optim.warmup_ratio`, `optim.weight_decay`,
`train.method` full|lora, `lora.rank`, `lora.alpha`, `train.seq_len`, `train.global_batch`,
`train.micro_batch`, `train.max_steps`, `train.epochs`, `rl.algorithm` dr_grpo|gspo|dapo, `rl.group_size`,
`rl.temperature`, `rl.kl_coef`, `data.mix`); `parallel.*` and `train.async` are refused (UNSUPPORTED_AXES).

Launch spec shape: `{trial, role(baseline|candidate|confirm), seed, delta{path: value}, nodes, gpus_per_node,
partition, time, gpu_hours_est, commands[]}`.

Result shape: `{trial, role, seed, status(ok|crash), limited, steps, eval_policy_fingerprint,
metrics{name: {value, se}}}`.

## Outputs
- Artifacts: `auto_research_report` (`auto_research_report.json`) with
  `{campaign{id, spec_hash, approver}, outcome, recommendation, decisions{trial: decide()},
  budgets{planned, used_gpu_hours, runs}, ledger{count, head_hash, verified}, tsv}`.
- The ledger (`chain.jsonl` + content-addressed `objects/<sha256>.json`) is the evidence:
  ops `campaign_approved`, `launch_authorised`, `trial_result`, `campaign_closed`, each entry hash-chained
  over its fields; `verify()` reports the first broken seq; results are never updated or deleted.
- Status/exit: PASS (0) | RED (5) | UNMEASURED (95) | REFUSED (96).
- Payload per action: `check` -> `{spec_hash, axes, budget}` (no ledger writes); `launch` ->
  `{launch_token, budget_left}`; `record` -> `{recorded, ledger}`; `propose` -> `{cards}` (no writes);
  `close` -> the report contents.

## Scope
- Stages: sft, preference, rl. Model types: llm, vlm. Families: any.
- Algorithms: dr_grpo, gspo, dapo. Methods: full, lora. Hardware: any (cluster rules are declarative).
- Consumes nothing from FS artifacts; emits `auto_research_report`; M1 wires submission via `training.emit`.

## The loop
1. `check` the campaign spec (objective/eval policy, base fingerprint, budget, axes, cluster rules).
2. A human hashes and approves the spec once: `campaign_confirm = campaign_hash(spec)`.
3. `propose` idea cards (a `baseline` card first, until `baseline_repeats` calibrated baseline results exist).
4. `launch` validates a launch spec and appends `launch_authorised` with a derived `launch_token`
   (M0 authorises only; M1 submits through `training.emit` + an IMEX probe).
5. `record` the returned metrics as an immutable `trial_result` (one result per (trial, seed)).
6. Repeat 3-5 until the budget or `stopping` rule says stop; then `close`: verify the ledger, decide each
   candidate against the baseline on `objective.metric` (and guardrails), write the report, append
   `campaign_closed`.

NeMo-style stop rules carry over: count attempted trials against `budget.max_runs`, convert
`budget.gpu_hours_total` into concrete accounting, and never let an underpowered screening run (limited)
become evidence.

## The approval model
One human approval hash bounds a whole campaign; every launch gets a derived token.
- The envelope is opened by the first action on an empty ledger (`campaign_approved{spec_hash, approver}`).
- `campaign_confirm` must equal `campaign_hash(spec)`, and the ledger's approved hash is checked every
  action: a different hash is refused (AR-AP-001).
- `launch_token = sha256("fs-ar-launch-v1|" + confirm + "|" + canonical(launch_spec))`: bound to the
  approved campaign and to that exact launch spec.

**May (inside an authorised campaign):** append `launch_authorised` entries whose token the ledger derives;
record new trial results (new trial/seed pairs); propose ideas from the catalog; record the close verdict;
read the ledger and derive views (the TSV view is derived only, never read back).

**May not (enforced, not politeness):** never `pkill -u`, `killall`, or `scancel` a job that is not in our
ledger (those command shapes are refused: AR-LN-004); never edit the locked fields (`base`, `model`:
AR-LN-001); never overwrite a recorded result (results are immutable: AR-LG-001); never declare PASS
itself - the acceptance statistics and the exit codes (0/5/95/96) are computed from evidence, not claimed.

## Actions and request fields
| Action | Needs | Writes | Returns |
|---|---|---|---|
| `check` | `campaign_spec` | nothing | spec_hash, axes, budget |
| `launch` | spec + `campaign_confirm` + `approver` + `launch_spec` | `launch_authorised` | launch_token, budget_left |
| `record` | spec + confirm + `result` | `trial_result` | recorded key, ledger head |
| `propose` | spec + confirm + `current` + `symptoms` | approval envelope only | cards |
| `close` | spec + confirm + recorded results + `stop_reason` | `campaign_closed` + report artifact | report contents |

## Acceptance statistics
- Evidence = ok, non-limited results with the metric present, paired by seed; crashes are excluded and
  counted (`crash_excluded:<trial>:<seed>`).
- Noise floor = sample std of `baseline_repeats` baseline repeats (1.4826*MAD when n == 3 if that is > 0,
  else std); `None` means the floor is uncalibrated and nothing can be decided.
- `tau = max(noise_floor, noise_floor_rel * |mean ref|, k * sqrt(mean se_ref^2 + mean se_cand^2))`
  (a missing `se` counts as 0; the sign of `mean_delta` follows `objective.direction`).
- Verdicts: `accepted_gain` (mean_delta > tau), `accepted_flat` (|mean_delta| <= tau and the candidate is
  simpler), `no_gain`, `rejected_regress` (mean_delta < -tau or a guardrail breach), `unmeasured`
  (uncalibrated floor, fewer than `confirm_repeats` paired seeds, a limited run, a missing metric, or an
  eval-policy fingerprint mismatch).
- Guardrails: per guard metric `drop = gsign * (ref - cand)` where `gsign` follows the guard's own
  direction (higher-is-better `max` by default, never `objective.direction`); a breach is
  `drop > max(guard_abs_epsilon, k * sqrt(se_r^2 + se_c^2))` and forces `rejected_regress`.
- `confirm.guardrail_directions: {name: "max"|"min"}` overrides the sign per guard metric.

## Validation rules

Outcome semantics (shared with `core/contract.py`): an **input** BLOCK refuses before any work - status
REFUSED (exit 96), nothing written. A **handoff** BLOCK never refuses: the work ran and the result is RED
(exit 5). WARN never blocks but changes the status where noted (AR-HO-004 makes the close UNMEASURED,
exit 95); INFO travels with a PASS result. Precedence when several apply: REFUSED > UNMEASURED > RED > PASS, with one exception: a broken ledger
(AR-HO-003) is RED even when evidence is also missing - no verdict is trustworthy until the chain verifies.

| Rule id | Phase | Severity | Fires when |
|---|---|---|---|
| AR-IN-001 | input | BLOCK | objective.metric missing or not one of eval_policy.metrics |
| AR-IN-002 | input | BLOCK | base.model missing, or base.fingerprint not `sha256:<64 hex>` |
| AR-IN-003 | input | BLOCK | budget.gpu_hours_total or budget.max_runs absent or <= 0 |
| AR-IN-004 | input | BLOCK | eval_policy.fingerprint missing or malformed |
| AR-IN-005 | input | BLOCK | axis key outside AXIS_PATHS (or in UNSUPPORTED_AXES) or range/values invalid for its kind |
| AR-IN-006 | input | BLOCK | result payload malformed, or a crash record carries metric values |
| AR-AP-001 | input | BLOCK | campaign_confirm missing/wrong hash, ledger approved a different hash, or no approver to open the envelope |
| AR-LN-001 | input | BLOCK | delta outside spec.axes, nodes/gpus/partition outside spec.cluster, or a `base`/`model` override |
| AR-LN-002 | input | BLOCK | budget.max_runs reached, gpu-hour budget exceeded, or a non-confirm run dips into the reserve |
| AR-LN-004 | input | BLOCK | time != `10-00:00:00`, forbidden command (pkill -u / scancel / killall), or a quarantined node is named |
| AR-LN-005 | input | BLOCK | gpu_hours_est implies a wall time above budget.per_run_timeout_h (> 240h refuses) |
| AR-LG-001 | input | BLOCK | recording would overwrite a (trial, seed) result; results are immutable |
| AR-HO-001 | handoff | BLOCK | campaign closed with no accepted gain |
| AR-HO-002 | handoff | BLOCK | best candidate breaches a guardrail band |
| AR-HO-003 | handoff | BLOCK | ledger chain verification failed (reports the first broken seq) |
| AR-HO-004 | handoff | WARN | load-bearing evidence missing (uncalibrated noise floor, screening-only seeds, limited or crashed runs) - status UNMEASURED |
| AR-HO-005 | handoff | INFO | accepted a flat-but-simpler change |

Close outcomes: improved (accepted gain) / flat_simplified / no_gain / regressed / unmeasured, decided in
that order after the ledger check: ledger broken -> RED (AR-HO-003); unmeasured evidence that could change
the outcome -> UNMEASURED (AR-HO-004); best candidate guardrail breach -> RED (AR-HO-002); no accepted
gain/flat -> RED (AR-HO-001).

## Failure handling
- REFUSED (96): fix the request field named in the refusal (spec/confirm/launch/result shape).
- RED (5): the campaign measured something and it did not accept - read `decisions` in the report and the
  `reasons` lines (`guardrail:<name>`, `mean_delta ... > tau ...`), or AR-HO-003 for a damaged chain.
- UNMEASURED (95): calibrate first - `baseline_repeats` ok baseline results with the eval policy
  fingerprint of the spec, and `confirm_repeats` paired seeds per candidate, none of them limited/crashed.
- Nothing in this skill repairs or rewrites a ledger: entries are append-only. Damage is detected
  (AR-HO-003 names the first broken seq), never patched.
- Limitations: the hash chain is not anchored outside the ledger file, so it detects edits that do not
  recompute the chain, not a full rewrite; `gpu_hours_est` is self-declared by the launch request and is
  checked against the budget, not measured. A guardrail with fewer than 2 paired measurements makes the
  close UNMEASURED (`guardrail_unmeasured:<name>`), never an accept.

## FS interface
- Emits: `auto_research_report` artifact in `ctx.artifacts_dir`.
- Consumes: nothing from FS artifacts in M0 (`consumes: ()`).
- APIs: `campaign_hash`, `launch_token`, `check_spec`, `check_launch`, `noise_floor`, `decide`, `propose`,
  `Ledger` (append/verify/results/launches/tsv_view). No processes are launched.

## Increments
- **M0 (now)**: validate, authorise (token), record, propose, decide, close - CPU-only, no submission.
- **M1**: submission via `training.emit` + IMEX probe; the launch token gates the actual job.
- **M2**: seed/claim management (screening vs confirm repeats as first-class ledger ops).
- **M3**: pluggable proposer (the ideas catalog stays the deterministic baseline).

## Worked examples
1. Validate a campaign before signing anything:
   ```python
   request = {"action": "check", "campaign_spec": {
       "id": "ar-2026-03-optim", "objective": {"metric": "val_accuracy", "direction": "max",
       "benchmarks": ["gsm8k"]}, "eval_policy": {"fingerprint": "sha256:..." + 64, "metrics": ["val_accuracy"]},
       "base": {"model": "gemma4-4b", "fingerprint": "sha256:..." + 64},
       "confirm": {"k": 2.0, "noise_floor_rel": 0.005, "guardrails": [], "guard_abs_epsilon": 0.01},
       "seeds": {"baseline_repeats": 3, "confirm_repeats": 3, "seed_list": [101, 102, 103]},
       "budget": {"gpu_hours_total": 24.0, "max_runs": 6, "per_run_timeout_h": 8.0, "reserve_frac": 0.3},
       "stopping": {"no_gain_streak": 3, "max_crash_streak": 2},
       "axes": [{"key": "optim.lr", "type": "log_float", "min": 1e-6, "max": 1e-3}],
       "cluster": {"partition": "rally", "max_nodes": 2, "gpus_per_node": 8,
                   "time": "10-00:00:00", "exclude": ["r01dgx02"]}}}
   ```
   The returned `spec_hash` is what the human approves once as `campaign_confirm`.

2. Authorise one candidate run (M0: nothing is submitted):
   ```python
   request = {"action": "launch", "campaign_spec": spec, "campaign_confirm": confirm_hash,
              "approver": "reviewer", "ledger_dir": "reports/ar-2026-03-optim/ledger",
              "launch_spec": {"trial": "lr-down", "role": "candidate", "seed": 101, "delta": {"optim.lr": 5e-4},
                              "nodes": 1, "gpus_per_node": 8, "partition": "rally", "time": "10-00:00:00",
                              "gpu_hours_est": 4.0, "commands": ["uv run train.py --config campaign.yaml"]}}
   ```

3. Record results and close:
   ```python
   request = {"action": "record", "campaign_spec": spec, "campaign_confirm": confirm_hash,
              "approver": "reviewer", "ledger_dir": ledger_dir,
              "result": {"trial": "lr-down", "role": "candidate", "seed": 101, "status": "ok", "limited": False,
                         "steps": 200, "eval_policy_fingerprint": spec["eval_policy"]["fingerprint"],
                         "metrics": {"val_accuracy": {"value": 0.55, "se": 0.002}}}}
   request = {"action": "close", "campaign_spec": spec, "campaign_confirm": confirm_hash,
              "approver": "reviewer", "ledger_dir": ledger_dir, "stop_reason": "budget exhausted"}
   ```
   `close` returns the report plus the decision per trial and the derived TSV view
   (`seq trial role seed status metric value se`).

## Attribution
Adapted from NeMo-RL nemo-rl-auto-research (Apache-2.0): loop shape, stop rules, ideas catalog, TSV view.
