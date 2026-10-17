---
name: fskills-auto-research
license: MIT
description: 'Manages a hash-chained auto-research campaign (M0): validates the campaign spec, opens one
  human approval envelope (campaign_confirm hash), authorises launches with derived tokens, records immutable
  trial results, proposes ideas from the ideas catalog and closes with statistical acceptance against a
  calibrated noise floor (guardrails, keep/discard/crash TSV view) writing auto_research_report. Do NOT use
  for single training runs (fskills-training), nor for evaluation alone (fskills-evaluation). M1 submits
  emitted trials (`envelope`/`submit`/`cancel`) behind the envelope budget gate and the IMEX fabric probe.'
compatibility: "Requires Python>=3.10 and the foundationskills package (fskills CLI); launches additionally need FoundationScale on a Slurm GPU cluster. Without them the skill reports REFUSED or UNMEASURED and gives the install step instead of improvising."
metadata:
  version: "0.1.0"
  invocation: "model"
  fs_status_doctrine: "PASS/RED/UNMEASURED/REFUSED (exit 0/5/95/96)"
  when_to_use: '["Run an auto research campaign and find a better recipe for val_accuracy on this base model","Validate my campaign spec (axes, budget, eval policy) before I sign the approval hash","Authorise a candidate run and give me its derived launch token","Record these trial results and decide accepted_gain / accepted_flat / no_gain / rejected_regress / unmeasured","Propose the next three hypotheses from the ideas catalog given these symptoms","Close the campaign and give me the auto_research report with the ledger TSV view"]'
---

# Auto Research

## Purpose
Drive a bounded experiment campaign against one stated objective (metric + direction) without ever
trusting a claim the evidence cannot support. The skill validates a `campaign_spec`, opens ONE human
approval envelope (`campaign_confirm` = `campaign_hash(spec)`), authorises individual launches with
derived per-launch tokens, records immutable trial results into an append-only hash-chained ledger,
proposes concrete hypotheses (idea cards), and closes with a statistical verdict against a calibrated
noise floor. M1 gates every emitted trial behind the campaign envelope budget and the IMEX fabric probe.

## When to use
- You have a goal ("improve val_accuracy") and want a measured loop of baseline -> propose -> run ->
  record -> decide -> close with an auditable evidence chain.
- You need one approval hash to bound a whole campaign plus derived, per-launch authorisation tokens.
- You need acceptance statistics (tau bands, guardrails) rather than "the last run looked better".

Do NOT use it for a single training run (use the training skill), for evaluation alone (use the
evaluation skill), or for bug fixes/refactors (no FoundationSkills skill should take those).

## Inputs
Request object; every field is optional except those the action needs:
- `action`: `check | envelope | launch | submit | cancel | record | propose | close`.
- `campaign_spec`, `campaign_confirm` (its human-approved hash) and `approver` (required on a ledger's first action);
  `ledger_dir` (default `<workdir>/ledger`).
- Per action: `envelope` (envelope); `trial_spec`, `launch_token`, `fabric_ttl_s` (submit: a fresh IMEX probe runs
  before every submission, cache TTL capped at 300 s); `launch_spec` (launch); `result` (record); `job_ids` and
  `reason` (cancel); `current` and `symptoms` (propose); `stop_reason` (close); `llm_pool` or `llm_pool_file`
  (propose with the `llm` proposer: the endpoint is fingerprinted, never a URL or credential in the spec).

Spec shape (the object a human hashes and approves): `{id, objective{metric, direction, benchmarks[]},
eval_policy{fingerprint, metrics[]}, base{model, fingerprint}, confirm{k, noise_floor_rel, guardrails[],
guard_abs_epsilon}, seeds{...}, budget{...}, stopping{...}, axes[...], cluster{...}}`, plus an optional
`proposer{...}` block. Axes are the measured FoundationSkills knobs; `parallel.*` and `train.async` are refused.

Every field, block shape, the launch-spec and result shapes, `fabric_ttl_s` and `llm_pool` rules:
`references/inputs.md`.

## Outputs
- Artifact `auto_research_report` (`auto_research_report.json`): campaign id, spec hash and approver; outcome and
  recommendation; per-trial `decisions`; planned vs used budgets; the ledger count, head hash and verification;
  and the proposal replay summary.
- The ledger (`chain.jsonl` + content-addressed `objects/`) is the evidence: append-only, hash-chained ops
  (`campaign_approved`, `launch_authorised`, `trial_result`, `campaign_closed`, `launch_envelope`, `job_submitted`,
  `job_cancelled`, `proposal`); `verify()` reports the first broken seq; results are never updated or deleted.
- Status/exit: PASS (0) | RED (5) | UNMEASURED (95) | REFUSED (96).
- Every action returns a payload (`check` writes nothing to the ledger; a repeated `close` re-reports the sealed
  chain). Render refusals (`sbatch_not_rendered`, `sbatch_render_failed:<why>`, `partition_not_rendered:<p>`)
  happen before any launch.
Full report fields, ledger-op payload keys and per-action payloads: `references/outputs.md`.

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
- The first action on an empty ledger opens it (`campaign_approved{spec_hash, approver}`); `campaign_confirm` must
  equal `campaign_hash(spec)` on every action, and a different hash is refused (AR-AP-001).
- Launch and per-trial tokens are derived from that consent (`fs-ar-launch-v1`, `fs-ar-envelope-v1`,
  `fs-ar-trial-v1`) and budget-decremented in the ledger; REFUSED when runs or measured GPU hours are exhausted.
- The render gate (G1 + G2) re-runs the approval check inside `run()` before anything is emitted, and scans the
  rendered command for forbidden commands (AR-LN-004) and quarantined nodes (AR-LN-005).
Token formulas and the render gate in full: `references/approval-model.md`.

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
| `envelope` | spec + `campaign_confirm` + `approver` + `envelope` | `launch_envelope` | envelope_token |
| `launch` | spec + `campaign_confirm` + `approver` + `launch_spec` | `launch_authorised` | launch_token, budget_left |
| `submit` | spec + confirm + `trial_spec` + `launch_token` | `launch_authorised` + `job_submitted` | job_id, budget_after |
| `cancel` | spec + confirm + `job_ids` + `reason` | `job_cancelled` | cancelled, drops |
| `record` | spec + confirm + `result` | `trial_result` | recorded key, ledger head |
| `propose` | spec + confirm + `current` + `symptoms` (+ `llm_pool` or `llm_pool_file` for proposer `llm`) | `proposal` | cards, drops, proposer, requested, rows_digest, seed, fallback, dropped, `llm_usage` (llm) |
| `claim` | spec + confirm + `trial` | `claim` | claim_id, champion, prev |
| `close` | spec + confirm + recorded results + `stop_reason` | `campaign_closed` + report artifact | report contents (champion, claims, unclaimed_gains) |

Once `campaign_closed` is on the ledger every action except `check` and `close` is refused (AR-LG-002) and a repeated `close` is idempotent.

## Acceptance statistics
- Evidence = ok, non-limited results with the metric present, paired by seed; crashes are excluded and
  counted (`crash_excluded:<trial>:<seed>`).
- Noise floor = sample std of the **eval-only** `baseline_repeats: 3` noise-floor repeats (M1 submits them as
  `job_submitted.kind == 'eval_only'` jobs; AR-HO-006 fires when a repeat is a non-eval_only job **or**, once
  the campaign ledger carries any `launch_envelope` entry, when the repeat's `job_submitted` provenance is
  unknown - the message names `baseline_provenance_unknown:<trial>`), sample
  std (1.4826*MAD when n == 3 if that is > 0, else std); `None` means the floor is uncalibrated and nothing
  can be decided. Ledgers with no `launch_envelope` keep the M0 compat: manual baseline records never fire
  AR-HO-006.
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

Close outcomes: improved (accepted gain) / flat_simplified / no_gain / regressed / unmeasured, decided in
that order after the ledger check: ledger broken -> RED (AR-HO-003); unmeasured evidence that could change
the outcome -> UNMEASURED (AR-HO-004); best candidate guardrail breach -> RED (AR-HO-002); no accepted
gain/flat -> RED (AR-HO-001).

Every rule this skill enforces (the full catalogue, with each rule's exact trigger, is
`references/validation-rules.md`):

| Rule id | Phase | Severity | Fires when |
|---|---|---|---|
| AR-IN-001 | input | BLOCK | objective.metric missing or not in eval_policy.metrics |
| AR-IN-002 | input | BLOCK | base.model or base.fingerprint missing or not a sha256 reference |
| AR-IN-003 | input | BLOCK | budget.gpu_hours_total or budget.max_runs absent or <= 0 |
| AR-IN-004 | input | BLOCK | eval_policy.fingerprint missing or malformed |
| AR-IN-005 | input | BLOCK | axis key outside AXIS_PATHS (or unsupported) or range/values invalid for its kind |
| AR-IN-006 | input | BLOCK | result payload malformed or a crash result carries metric values |
| AR-IN-007 | input | BLOCK | seed plan invalid: seed_list empty/duplicate/non-int, repeats < 1, confirm_repeats beyond the seed list, or cluster.max_in_flight invalid/above max_runs |
| AR-IN-008 | input | BLOCK | proposer config invalid (name outside catalog|optuna, min_rows < 1, require_model not a bool, seed not an int, unknown keys) |
| AR-AP-001 | input | BLOCK | campaign_confirm missing or not the hash of the spec (or ledger approved a different hash) |
| AR-LN-001 | input | BLOCK | delta outside spec.axes, shape/partition mismatch or a locked base/model override |
| AR-LN-002 | input | BLOCK | launch would exceed the run or gpu-hour budget (including the confirm reserve) |
| AR-LN-004 | input | BLOCK | launch carries a forbidden command (pkill -u / scancel / killall) or names a quarantined node |
| AR-LN-005 | input | BLOCK | gpu_hours_est implies a wall time above per_run_timeout_h |
| AR-LN-003 | input | BLOCK | IMEX fabric at submit is not 'ready' (refused or unmeasured); the named reason is carried in the message |
| AR-LN-006 | input | BLOCK | envelope missing or forged, envelope budget != spec budget, the trial token is not derived, or the envelope budget is exhausted (runs or GPU hours) |
| AR-LN-007 | input | BLOCK | cancel names a job id this campaign's ledger never submitted (or requests no job ids) |
| AR-LN-008 | input | BLOCK | in-flight (running or pending) jobs at cluster.max_in_flight; unmeasured job states fail closed |
| AR-LN-009 | input | BLOCK | a baseline/screening submit would spend the run reserve kept for confirm-phase runs |
| AR-LG-001 | input | BLOCK | a recorded trial result would be overwritten (results are immutable; record a new trial) |
| AR-LG-002 | input | BLOCK | a mutating action after campaign_closed (close is final: only check/close stay open) |
| AR-RS-007 | input | BLOCK | claim over an incomplete/unmeasured confirm set (missing, pending, crashed, limited or unpaired seed) or of a decision that is not accepted_gain |
| AR-RS-008 | input | BLOCK | claim on a multi-objective campaign whose decision is not accepted_gain (a trade-off, tie, regression or unmeasured objective never dominates the reference) |
| AR-RS-009 | input | BLOCK | a claim's confirm set holds an unmeasured or caller-asserted (unbound) result |
| AR-IN-009 | input | BLOCK | spec.objectives invalid (2-4 {metric, direction} entries, objectives[0] = objective, distinct metrics inside eval_policy.metrics, none a guardrail) |
| AR-PR-001 | input | BLOCK | proposer.require_model set and the model proposer is unavailable (below min_rows, extra missing, no axes, model error, no in-axes cards) or its replay is not byte-identical |
| AR-PR-002 | input | BLOCK | the optuna-cma proposer cannot hold a categorical axis (refused, never dropped) |
| AR-IN-011 | input | BLOCK | named trial evidence is missing, unreadable or unrecognised |
| AR-IN-012 | input | BLOCK | a caller-asserted result field contradicts its bound evidence, or the eval is not bound to the trained checkpoint |
| AR-IN-013 | input | BLOCK | an rl-stage campaign does not declare rl_min_measured_fraction (0 < f <= 1) and the rl_measured_fraction guardrail |
| AR-IN-010 | input | BLOCK | llm proposer block malformed (pool_key/model missing, URL-shaped, or caps/sampling invalid) |
| AR-PR-003 | handoff | BLOCK | an llm response carries an unsafe card (unknown key, non-scalar delta, or a command/quarantined node in model output) |
| AR-PR-004 | handoff | BLOCK | an llm proposal record overstates provenance (byte_identical / generation replay claimed) |
| AR-PR-005 | input | BLOCK | llm endpoint not pinned by the pool registry, or a credential-shaped token in the ledgered payload |
| AR-HO-001 | handoff | BLOCK | campaign closed with no accepted gain |
| AR-HO-002 | handoff | BLOCK | best candidate breaches a guardrail band |
| AR-HO-003 | handoff | BLOCK | ledger chain verification failed |
| AR-HO-004 | handoff | WARN | load-bearing evidence missing (uncalibrated noise floor, screening-only seeds, limited or crashed runs) - status UNMEASURED |
| AR-HO-005 | handoff | INFO | accepted a flat-but-simpler change |
| AR-HO-006 | handoff | WARN | a noise-floor baseline repeat was submitted as a non-eval_only job (decision 3: the floor is eval-only repeats) - status UNMEASURED |
| AR-HO-007 | handoff | BLOCK | a recorded proposal does not replay byte-identically at close (search provenance drifted) |
| AR-HO-008 | handoff | INFO | multi-objective trade-offs disclosed at close (never claimed; no accepted gain) |
| AR-HO-009 | handoff | BLOCK | an llm proposal's recorded cards do not re-parse from its recorded response (parse provenance drifted) |
| AR-HO-010 | handoff | INFO | the closed ledger holds caller-asserted (unbound) results; each is named and counted |

## Failure handling
Nothing is ever simulated or repaired: a refusal happens before anything is emitted or submitted, and the
ledger is append-only. By outcome:
- REFUSED (96): fix the request field the refusal names (spec, confirm, envelope, launch, trial, fabric, cancel,
  an invalid `proposer` block, or an unpinned or credential-leaking `llm_pool`). The submit gate (G1 + G2)
  re-runs the full check and scans the rendered command for forbidden commands and quarantined nodes first.
- RED (5): the campaign measured something and did not accept it; read `decisions` in the report.
- UNMEASURED (95): calibrate first (`baseline_repeats` ok baseline results under the eval policy); a close-time
  re-check that cannot run is a counted drop, never a pass.
Measured GPU hours, ledger damage detection and the chain's limitations: `references/failure-handling.md`.

## Details on demand
Read these when the task needs them:
- `references/campaign-mechanics.md`: seeds, claims and concurrency (M2); proposers catalog, optuna and llm
  (M3, M4, M5b); multi-objective campaigns (M5a); evidence-bound results (M6a); the increment history.
- `references/validation-rules.md`, `references/failure-handling.md`, `references/inputs.md`,
  `references/outputs.md`, `references/approval-model.md`, `references/worked-examples.md`: the full
  text of the sections summarised above.

## FS interface
- Emits: `auto_research_report` artifact in `ctx.artifacts_dir`.
- Consumes: nothing from FS artifacts in M0 (`consumes: ()`).
- APIs: `campaign_hash`, `launch_token`, `check_spec`, `check_launch`, `noise_floor`, `decide`, `propose`, `proposers.select`, `proposers.replay`, `proposers.reverify`, `proposers.package_version`,
  `squeue.job_states`, the claims/seeds/concurrency/locks helpers,
  `Ledger` (append/verify/results/launches/tsv_view). `squeue` is read-only. No processes are launched.

**Host portability (A1.5).** On a host where the `fskills` CLI or the `foundationskills` package is missing, do not improvise a plan or simulate a result: report REFUSED with the install step (`pip install -e <FoundationScale repo>/FoundationSkills`, or run where it is installed), and report every check that could not run as UNMEASURED, never PASS. `submit` also needs FoundationScale (CORE-HOST-001 with the install step); the ledger actions run anywhere.

## Worked examples
Four end-to-end request sequences are in `references/worked-examples.md`:
1. validate a campaign before signing anything;
2. authorise one candidate run (nothing is submitted);
3. record results and close;
4. consent an envelope, submit one emitted trial, and cancel what was submitted.

## Attribution
Adapted from NeMo-RL nemo-rl-auto-research (Apache-2.0): loop shape, stop rules, ideas catalog, TSV view.
