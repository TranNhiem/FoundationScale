---
name: fskills-auto-research
license: MIT
version: 0.1.0
description: 'Manages a hash-chained auto-research campaign (M0): validates the campaign spec, opens one
  human approval envelope (campaign_confirm hash), authorises launches with derived tokens, records immutable
  trial results, proposes ideas from the ideas catalog and closes with statistical acceptance against a
  calibrated noise floor (guardrails, keep/discard/crash TSV view) writing auto_research_report. Do NOT use
  for single training runs (fskills-training), nor for evaluation alone (fskills-evaluation). M1 submits
  emitted trials (`envelope`/`submit`/`cancel`) behind the envelope budget gate and the IMEX fabric probe.'
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
noise floor. M1 gates every emitted trial behind the campaign envelope budget and the IMEX fabric probe.

## When to use
- You have a goal ("improve val_accuracy") and want a measured loop of baseline -> propose -> run ->
  record -> decide -> close with an auditable evidence chain.
- You need one approval hash to bound a whole campaign plus derived, per-launch authorisation tokens.
- You need acceptance statistics (tau bands, guardrails) rather than "the last run looked better".

Do NOT use it for a single training run (use the training skill), for evaluation alone (use the
evaluation skill), or for bug fixes/refactors (no FoundationSkills skill should take those).

## Inputs
Request object (all fields optional except those the action needs):
- `action`: `check | envelope | launch | submit | cancel | record | propose | close`.
- `campaign_spec`: the campaign spec dict (see below), `campaign_confirm`: its human-approved hash,
  `approver`: the human who approved it (required on the first action of a ledger).
- `ledger_dir`: campaign ledger directory (chain.jsonl + objects/); defaults to `<workdir>/ledger`.
- `launch_spec`: one authorised run (launch action), `result`: one trial result (record action).
- `envelope` (envelope action): `{budget{max_runs, gpu_hours_total} mirroring spec.budget, ...}` - the consented
  campaign envelope; `trial_spec` (submit action): `{trial, role, kind: train|eval_only, delta, seed, nodes,
  gpus_per_node, partition, gpu_hours_est, train_request|eval_request}`.
- `launch_token` (submit): the derived per-trial token `sha256("fs-ar-trial-v1|" + envelope_token + "|" +
  canonical(trial_spec))`; `job_ids` (cancel): job ids THIS campaign's ledger submitted; `reason`: cancel reason.
- `fabric_ttl_s` (submit): the IMEX probe cache TTL in seconds (default 300, hard cap 300: the effective TTL is
  `min(requested, 300)` and a non-finite, negative or non-numeric value falls back to 300). The cache path is
  **fixed** at `<ledger_dir>/fabric.json` (there is no request field for it), a fresh uncached IMEX probe runs
  immediately before every submission (and `submit_trial` refuses unless that state is `ready`), a future-dated
  `refused` entry stays sticky until `fabric.json` is removed (fail closed), while a future-dated `ready` is
  always re-probed.
- `current`: current knob values for the proposer, `symptoms`: observed symptoms (propose action).
- `stop_reason`: why the campaign is being closed (close action).
- `llm_pool` / `llm_pool_file` (propose, proposer `llm` only): the ONLY source of the LLM endpoint - `llm_pool` is a
  registry dict (fs.model_registry/1 `models.<key>` or the same fields flat) and `llm_pool_file` a `.json`/`.yaml`
  path (yaml is read only for this). The spec hash-locks `pool_key` + `model`; the endpoint is fingerprinted
  `sha256(base_url||pool_key||model)` and ledgered - never the URL, never a credential (only the auth env var NAME is
  used). Unpinned (missing/inactive pool entry, model mismatch, credential in the registry) or a credential-shaped
  token in the prompt or response -> AR-PR-005 REFUSED (input BLOCK) before/without appending.

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
- Optional block `proposer{name: catalog (default) | optuna | optuna-cma | llm, min_rows: 10 (>= 1; may be 0 for llm),
  require_model: false, seed: 0}`: selects the (M3/M5b) proposer and is part of the spec hash; validated by AR-IN-008.
- For `proposer.name: llm`, the block `proposer.llm{pool_key (required), model (required), max_cards (1..budget.max_runs,
  default 3), max_calls (default budget.max_runs), max_evidence_chars, parse_spec_version: 1,
  sampling{temperature, max_tokens}}` - a URL-shaped `pool_key`/`model` is refused (the endpoint is never configured
  in the spec; see `llm_pool`). A malformed block is AR-IN-010.

Launch spec shape: `{trial, role(baseline|candidate|confirm), seed, delta{path: value}, nodes, gpus_per_node,
partition, time, gpu_hours_est, commands[]}`.

Result shape: `{trial, role, seed, status(ok|crash), limited, steps, eval_policy_fingerprint,
metrics{name: {value, se}}}`.

## Outputs
- Artifacts: `auto_research_report` (`auto_research_report.json`) with
  `{campaign{id, spec_hash, approver}, outcome, recommendation, decisions{trial: decide()},
  budgets{planned, used_gpu_hours, runs}, ledger{count, head_hash, verified}, tsv,
  proposals{checked, byte_identical, drifted:[i], unmeasured}}`.
- The ledger (`chain.jsonl` + content-addressed `objects/<sha256>.json`) is the evidence:
  ops `campaign_approved`, `launch_authorised`, `trial_result`, `campaign_closed`, `launch_envelope`,
  `job_submitted`, `job_cancelled`, `proposal` (exactly one per `propose` call; payload keys `proposer`,
  `requested`, `seed`, `rows_digest`, `k`, `fallback`, `cards`, `drops`, `stats`, `replay_inputs` ({current,
  symptoms, results_count, launches_count}, or None when not canonicalisable), `package_version`, `replay_status`
  (byte_identical|parse_identical|unmeasured; `byte_identical` is never claimed for llm), `llm_usage`
  ({calls_used, max_calls, tokens_in, tokens_out}, llm only, None when unreported, never 0), and for llm the inline
  `llm` request/response/parse block), each entry hash-chained over
  its fields; `verify()` reports the first
  broken seq; results are never updated or deleted.
- Status/exit: PASS (0) | RED (5) | UNMEASURED (95) | REFUSED (96).
- Payload per action: `check` -> `{spec_hash, axes, budget}` (no ledger writes); `envelope` ->
  `{envelope_token}`; `launch` -> `{launch_token, budget_left}`; `submit` -> `{job_id, budget_after}`;
  `cancel` -> `{cancelled, drops}`; `record` -> `{recorded, ledger}`; `propose` ->
  `{cards, drops, proposer, requested, rows_digest, seed, fallback, dropped, replay_status, llm_usage}` (writes one `proposal` op);
  `close` -> the report contents (`budgets` carry measured/declared/drops from `campaign_usage`; `proposals` carries
  `{checked, byte_identical, drifted:[i], unmeasured}`, plus `parse_identical` / `parse_drifted:[i]` and a
  top-level `llm_usage` for llm campaigns only).
  A repeated `close` re-reports the sealed chain and never appends a second `campaign_closed`.
- Render refusals (`submit`, REFUSED before any launch): `sbatch_not_rendered` (a cluster trial with no sbatch
  would run argv on this host), `sbatch_render_failed:<why>` (train sbatch layering failed),
  `partition_not_rendered:<p>` (the render dropped `trial_spec.partition`). Train trials get their sbatch layered
  like `training.emit`; GB200 trials on fewer than 4 GPUs request `--mem-per-gpu` (default 200G) so they do not
  take the whole node; `eval_request.python` names the eval interpreter (default: the one running the skill).

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
- M1 approval model: one human consent = the campaign envelope token (`fs-ar-envelope-v1` over the approved
  envelope payload); per-trial tokens are derived (`fs-ar-trial-v1`) and budget-decremented in the ledger
  (`budget_after`); REFUSED when exhausted (runs or measured GPU hours).
- G1 + G2 (render gate, defence in depth): the approval gate is re-run inside `run()`'s submit branch before
  `emit_trial` (a direct `run()` call can never skip it, and an absent/empty envelope token never derives a
  trial token), and what is actually gated is the RENDER - `emit_trial` forces `train_request.nodes` /
  `.gpus_per_node` to the trial_spec values (naming each `overrode train_request.<key>`), and the rendered
  argv + `sbatch` (plus the `json.dumps(train_request/eval_request)` blob at `check_inputs` time) are scanned
  with the same AR-LN-004 forbidden-command patterns and AR-LN-005 quarantined-node checks before
  `submit_trial`. 

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

## Seeds, claims and concurrency (M2)
- The seed phase is derived from the trial role (`baseline` -> `baseline`, `candidate` -> `screening`,
  `confirm` -> `confirm`); `seeds.seed_list` (distinct positive ints) plus per-phase repeats validate as
  AR-IN-007 (1 <= repeats per phase, `confirm_repeats` <= len(seed_list)); `phase_unknown:<role>` and
  `seed_not_in_seed_list:<seed>` are AR-LN-001.
- `job_submitted` carries `seed` and the derived `phase`; runs are deduplicated by (trial, seed).
- In-flight cap: a submitted run holds a slot of `cluster.max_in_flight` until the ledger settles it (a
  `trial_result` for its (trial, seed) or its job among `cancelled_ids`); states read from squeue via an
  injected `state_fn` - a job missing from a successful squeue reads `terminal`, an unknown state fails
  closed as `in_flight_unmeasured:<n>` (the count is `None`, never 0) (AR-LN-008).
- Run reserve: `budget.reserve_frac` in [0,1) (default 0.3, `ceil(reserve_frac * max_runs)` run slots) is
  kept for confirm-phase runs; a baseline/screening submit that would spend it is refused
  `reserve_locked_for_confirm:runs` (AR-LN-009) - run slots only, the gpu-hour reserve stays AR-LN-002.
- Claims are explicit only: the `claim` action appends exactly one `claim` op, `prev` = the champion's
  claim_id or `baseline`, `claim_id` = `sha256:<hex>` over the canonical body. The claim rules: a complete
  measured confirm set, `accepted_gain` against the current champion's reference rows, one live claim per
  trial (AR-RS-007).
- `close` re-derives the chain from ledger evidence and never writes claims: it reports `champion` (last
  surviving accepted claim, else the `baseline` root), `claims` (the derived chain with each status) and
  `unclaimed_gains` (sorted trials decided `accepted_gain` with no claim).
- A claim that lost a seed result is downgraded (drop `claim_downgraded:<claim_id>` + AR-HO-004, UNMEASURED)
  without cascading: at close a claimed trial is decided against the rows it was claimed against, every
  other trial against the champion's rows (baseline rows while the chain roots at `baseline`).

## Proposers (M3, M4, M5b)
- The ideas catalog (`propose.py`, untouched) is the deterministic default and the regression oracle
  (byte-identical to the M0 ranking), and the mandatory fallback.
- A model proposer (optuna: TPE) is a lazily imported optional extra, never installed by default; it is
  eligible only with >= `proposer.min_rows` model rows, the extra importable and >= 1 axis (`eligible()`:
  known name, row count, extra imports, one axis).
- A model row (B2): a non-baseline trial with a measured result and recoverable axis parameters; noise-floor
  repeats never count. Parameters come from the result `delta` else the latest launch payload (B6),
  restricted to `spec.axes`.
- The told value is the trial's mean measured objective (direction from `objective.direction`; `None` is never
  told).
- Each optuna card is a full assignment over `spec.axes` with `score: None` (B5); model cards outside the axes
  drop `model_out_of_axes:<idea>`, never clamped.
- `select` re-runs a FRESH proposer over the same inputs and requires byte-identical cards.
- Non-strict failure falls back to the catalog with the counted drop `proposer_fallback_catalog:<reason>`;
  reasons: `below_min_rows:<n>/<m>`, `extra_missing:<name>`, `no_axes`, `model_error`, `no_model_cards`, `unverified`.
- With `require_model` set the same conditions are REFUSED (AR-PR-001: `proposer_unavailable:<reason>` or
  `proposer_unverified:<name>`) and nothing is appended.
- `rows_digest` ("sha256:<hex>") pins spec axes/objective/proposer config + results + launches + `current` +
  `symptoms` (`None` when unmeasurable).
- A card is a suggestion: it reaches a run only through `submit` and every M1/M2 gate. The proposer config is
  part of the spec hash: switching proposer mid-campaign fails `campaign_confirm`.
- `optuna-cma` = Optuna `CmaEsSampler(seed)` behind the same lazy extra (numeric-only): a `categorical` axis is REFUSED by AR-PR-002 (`proposer_axis_unsupported:optuna-cma:<key>`) at check time, never dropped; an axis the installed optuna cannot build is the counted drop `model_axis_skipped:<key>`.
- `select` provenance records `package_version` ("builtin" for the catalog incl. fallback) and every proposal records the replay inputs so it can be replayed from the ledger alone.
- Close re-runs every recorded proposal over its ledger prefix (`reverify`); a drift fires AR-HO-007 and turns the status RED with the outcome unchanged and the
  recommendation prefixed "investigate proposal replay drift before adopting: ".
- A replay that cannot be checked is the counted drop `proposal_replay_unmeasured:<reason>:<i>` (reasons: `legacy_proposal`, `recorded_unmeasured`, `ledger_prefix_missing`, `extra_missing:<name>`, `version_changed:<name>`, `model_error`, `chain_unverified`) and is never a pass.
- A changed optuna version is `version_changed` (unmeasured), not drift: cross-version identity is not claimed. The test registry injection (`registry=`) is test-only.
- The `llm` proposer (M5b) calls ONE OpenAI-compatible chat model - the endpoint comes ONLY from the request
  (`llm_pool` / `llm_pool_file`, AR-PR-005) and transport reuses data_engine's stdlib OpenAI-compatible client: no new
  dependency (`yaml` only to read an `llm_pool_file` `.yaml`). The model only emits typed cards
  (idea/domain/delta/watch/score/rationale); card validation drops, never clamps (`llm_out_of_axes:<idea>`, ...); `proposer.min_rows` may be 0.
- `llm` budgets and taint: `max_calls` counts ledgered calls (exhausted -> the fallback drop
  `llm_call_budget_exhausted`) and `llm_usage` `{calls_used, max_calls, tokens_in, tokens_out}` (`None` when
  unreported, never 0) rides the propose PASS payload and the close report. A forbidden command (pkill -u / scancel /
  killall) or a quarantined node in the response taints the WHOLE response: non-strict -> catalog fallback
  `llm_response_tainted:<i>`, strict -> AR-PR-003 REFUSED (AR-PR-003 is also a close-time audit of the recorded
  responses). Non-strict fallback reasons (`proposer_fallback_catalog:<reason>`; strict -> AR-PR-001
  `proposer_unavailable:<reason>`): `no_calls_left`, `no_axes`, `no_transport`, `transport_error`, `parse_error`,
  `parse_unstable`, `tainted`, `no_model_cards`, `model_error`.
- Record-and-audit (`llm`): the proposal payload stores the full request/response/parse inline (`llm` block),
  `generation: "nondeterministic"`, and `replay_status` `parse_identical|unmeasured` - `byte_identical` is never
  claimed for llm; an overstated claim is AR-PR-004 (handoff BLOCK) and propose refuses it defensively. Close
  re-parses the recorded response (it never calls the model again) and counts `parse_identical` / `parse_drifted` in
  `replay_report` (llm campaigns only); a drift is AR-HO-009 (handoff BLOCK, RED) and the outcome is never moved.

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

## Multi-objective campaigns (M5a)
- `spec.objectives`: 2-4 `{metric, direction}` entries; `objectives[0]` must equal `objective` (metric and
  direction), metrics are distinct, inside `eval_policy.metrics`, and never a guardrail (AR-IN-009). A spec
  without `objectives` takes the single-objective path byte-identically.
- `accept.decide_multi`: one common seed set J (seeds where baseline and candidate both measure EVERY
  objective; `|J| < confirm_repeats` -> unmeasured `n_pairs_below_confirm:<n>/<m>`), and per objective the M4
  tau formula over J only. Each objective classes as win (`delta > tau`), loss (`delta < -tau`), tie, or
  unmeasured (mean_delta `None`, never 0).
- Verdicts, in order: any unmeasured -> `unmeasured`; guardrail breach -> `rejected_regress`; >= 1 win and no
  loss -> `accepted_gain`; win + loss -> `tradeoff`; loss only -> `rejected_regress`; all tie and simpler ->
  `accepted_flat`; else `no_gain`.
- A `tradeoff` is never claimable (AR-RS-008 `claim_refused_not_dominating:<trial>`) and never promoted. A close
  whose only non-baseline outcome is a trade-off is `no_gain` (RED, AR-HO-001) plus AR-HO-008 INFO naming the
  trade-offs. The report then carries `objectives`, `tradeoffs` and
  `frontier {entries, excluded, order: "presentation_only"}`. The frontier is a non-dominated listing for
  humans; it never feeds accept or claim.

## Evidence-bound results (M6a)

- `record` takes named evidence instead of trusting the caller: `evidence: {run_manifests: [<path>...],
  eval_report: <path>}`. The result is derived from those files (`derive_result`): the run manifests give the
  training verdict and `rl_measured_fraction` (the manifest's `measured_fraction`, min over runs), the eval report gives each metric's
  `value`/`se`, and the eval must name the checkpoint the run saved (else AR-IN-012). Each file is stored by
  path and `sha256:` digest; missing/unreadable evidence is AR-IN-011.
- The stored row gains `evidence`, `evidence_class` (`bound` | `asserted` | `crash`), `metric_sources` (per metric
  `eval_report` | `run_manifest` | `asserted` | `absent`) and `derived`. A metric that cannot be derived and was not supplied is
  `{value: null, se: null}` with source `absent` - never a guessed number. A caller-supplied field that disagrees
  with the derivation is AR-IN-012, never an overwrite.
- Status domain is `ok | crash | unmeasured`. A training verdict of UNMEASURED, or `rl_measured_fraction` below
  the declared `rl_min_measured_fraction`, stores the row as `unmeasured`; `decide` reports such a trial as
  `unmeasured` (`manifest_unmeasured:<tag>`), never as a gain or a regression.
- Evidence is required when `spec.evidence_required` is true, or by default when the campaign is rl-stage
  (`is_rl_stage`). Then a claim over a caller-asserted row is AR-RS-009; legacy campaigns keep the M2-M5
  claim behaviour. An unmeasured row never grounds a claim in any campaign.
- An rl-stage campaign must declare `rl_min_measured_fraction` and the `rl_measured_fraction` guardrail
  (direction `max`). The guardrail is judged against the declared floor on the candidate's own rows (every
  confirm row >= `rl_min_measured_fraction`, else `guardrail:rl_measured_fraction:below_floor`), never paired
  against the reference - the base model has no RL fraction. AR-IN-013 at `check`.
- `close` re-hashes every bound row's evidence; a missing or changed file degrades that row to unmeasured
  (AR-HO-004 path); where evidence is required an asserted row is likewise unmeasured at close
  (`asserted_unbound:<trial>:<seed>`). Asserted rows left in the ledger are listed by AR-HO-010.

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
| AR-IN-007 | input | BLOCK | seed plan invalid (seed_list empty/duplicate/non-int, repeats < 1, confirm_repeats beyond the seed list, or `cluster.max_in_flight` invalid/above `budget.max_runs`) |
| AR-IN-008 | input | BLOCK | proposer config invalid (name outside `catalog`/`optuna`/`optuna-cma`/`llm`, `min_rows` < 1 (>= 0 allowed for `llm`), `require_model` not a bool, `seed` not an int, unknown keys) |
| AR-IN-009 | input | BLOCK | spec.objectives invalid (not 2-4 `{metric, direction}` entries, objectives[0] != objective, a duplicate metric, a metric outside eval_policy.metrics, or a guardrail used as an objective) |
| AR-IN-010 | input | BLOCK | the `proposer.llm` block is malformed (missing `pool_key`/`model`, `max_cards` outside 1..budget.max_runs, `max_calls` invalid, `parse_spec_version` != 1, bad `sampling`, a URL-shaped value, or unknown keys) |
| AR-IN-011 | input | BLOCK | named trial evidence (`evidence.run_manifests`, `evidence.eval_report`) is missing, unreadable or unrecognised, or a training/eval verdict is outside PASS/UNMEASURED |
| AR-IN-012 | input | BLOCK | a caller-asserted result field contradicts its bound evidence, or the eval is not bound to the trained checkpoint - REFUSED, the ledger is unchanged |
| AR-IN-013 | input | BLOCK | an rl-stage campaign does not declare `rl_min_measured_fraction` (0 < f <= 1) and the `rl_measured_fraction` guardrail |
| AR-AP-001 | input | BLOCK | campaign_confirm missing/wrong hash, ledger approved a different hash, or no approver to open the envelope |
| AR-LN-001 | input | BLOCK | delta outside spec.axes, nodes/gpus/partition outside spec.cluster, or a `base`/`model` override |
| AR-LN-002 | input | BLOCK | budget.max_runs reached, gpu-hour budget exceeded, or a non-confirm run dips into the reserve |
| AR-LN-004 | input | BLOCK | time != `10-00:00:00`, forbidden command (pkill -u / scancel / killall), or a quarantined node is named |
| AR-LN-005 | input | BLOCK | gpu_hours_est implies a wall time above budget.per_run_timeout_h (> 240h refuses) |
| AR-LN-003 | input | BLOCK | IMEX fabric not `ready` (refused or unmeasured) at submit; the named reason is carried in the message |
| AR-LN-006 | input | BLOCK | envelope missing or forged, envelope budget != spec budget, the trial token is not derived, or the envelope budget is exhausted (runs or GPU hours) |
| AR-LN-007 | input | BLOCK | `cancel` names a job id this campaign's ledger never submitted (or requests no job ids) |
| AR-LN-008 | input | BLOCK | in-flight (running or pending) jobs at `cluster.max_in_flight`; unmeasured job states fail closed (`in_flight_unmeasured:<n>`) |
| AR-LN-009 | input | BLOCK | a baseline/screening submit would spend the run reserve kept for confirm-phase runs (`reserve_locked_for_confirm:runs`) |
| AR-LG-001 | input | BLOCK | recording would overwrite a (trial, seed) result; results are immutable |
| AR-LG-002 | input | BLOCK | a mutating action after `campaign_closed` (close is final: only `check`/`close` stay open) |
| AR-RS-007 | input | BLOCK | claim over an incomplete/unmeasured confirm set (missing, pending, crashed, limited or unpaired seed) or of a decision that is not `accepted_gain` |
| AR-RS-008 | input | BLOCK | multi-objective claim of a candidate whose decision is not `accepted_gain` (a trade-off or worse never claims: `claim_refused_not_dominating:<trial>`) |
| AR-RS-009 | input | BLOCK | a claim's confirm set holds an unmeasured row (always) or a caller-asserted, unbound row (when the campaign requires evidence) |
| AR-PR-001 | input | BLOCK | `proposer.require_model` set and the model proposer is unavailable (below `min_rows`, extra missing, no axes, model error, no in-axes cards) or its replay is not byte-identical |
| AR-PR-002 | input | BLOCK | the optuna-cma proposer cannot hold a categorical axis (refused, never dropped) |
| AR-PR-003 | handoff | BLOCK | an llm response named a forbidden command (pkill -u / scancel / killall) or a quarantined node: the WHOLE response is tainted (non-strict -> catalog fallback `llm_response_tainted:<i>`, strict -> propose REFUSEDs at input); also the close-time audit of recorded responses |
| AR-PR-004 | handoff | BLOCK | an llm proposal claims byte-identical replay (overstated: llm `replay_status` is `parse_identical|unmeasured`) - audited at close, and propose refuses the claim defensively |
| AR-PR-005 | input | BLOCK | the llm endpoint is not pinned (pool entry missing/inactive, model mismatch, credential in the registry) or a credential-shaped token sat in the prompt/response - REFUSED before/without appending |
| AR-HO-001 | handoff | BLOCK | campaign closed with no accepted gain |
| AR-HO-002 | handoff | BLOCK | best candidate breaches a guardrail band |
| AR-HO-003 | handoff | BLOCK | ledger chain verification failed (reports the first broken seq) |
| AR-HO-004 | handoff | WARN | load-bearing evidence missing (uncalibrated noise floor, screening-only seeds, limited or crashed runs) or a claim downgraded at close (`claim_downgraded:<claim_id>`) - status UNMEASURED |
| AR-HO-005 | handoff | INFO | accepted a flat-but-simpler change |
| AR-HO-006 | handoff | WARN | a noise-floor baseline repeat shipped as a non-eval_only job, or (when the ledger has any `launch_envelope`) its `job_submitted` provenance is unknown (`baseline_provenance_unknown:<trial>`) (decision 3: the floor is eval-only repeats) - status UNMEASURED |
| AR-HO-007 | handoff | BLOCK | a recorded proposal does not replay byte-identically at close (search provenance drifted) |
| AR-HO-008 | handoff | INFO | a multi-objective close found only trade-offs (no dominating candidate); they are listed, never claimed, and the outcome stays `no_gain` |
| AR-HO-009 | handoff | BLOCK | a recorded llm response re-parses differently at close (`parse_drifted` in `replay_report`, llm campaigns only, the model is never called again) - RED, the outcome never moves |
| AR-HO-010 | handoff | INFO | the closed ledger holds caller-asserted (unbound) results; each is named and counted |

Close outcomes: improved (accepted gain) / flat_simplified / no_gain / regressed / unmeasured, decided in
that order after the ledger check: ledger broken -> RED (AR-HO-003); unmeasured evidence that could change
the outcome -> UNMEASURED (AR-HO-004); best candidate guardrail breach -> RED (AR-HO-002); no accepted
gain/flat -> RED (AR-HO-001).

## Failure handling
- G1 + G2 (M1 render gate): `run(action="submit")` re-runs the full `_check_submit_request` gate before
  anything is emitted or submitted - any BLOCK finding returns REFUSED (refusal = the first finding message,
  findings attached) with `emit_trial`, `submit_trial`, `launch_fn` and the runner never touched, and an
  absent/empty envelope token can never derive an `fs-ar-trial-v1` token. What actually runs is gated too:
  the rendered argv + `sbatch` (and, at `check_inputs` time, `json.dumps(train_request/eval_request)`) are
  scanned with the AR-LN-004 forbidden-command patterns and the AR-LN-005 quarantined-node names
  (`campaign.scan_command_text` / `campaign.scan_rendered`) - any hit refuses with that rule id, nothing is
  submitted and nothing is appended to the ledger.
- REFUSED (96): fix the request field named in the refusal (spec/confirm/envelope/launch/trial/fabric/cancel
  shape): a refused or unmeasured IMEX fabric (AR-LN-003 - the `sbatch` preamble stays the in-job gate), an
  envelope or derived token that is missing, forged or exhausted (AR-LN-006: runs or measured GPU hours), and
  a `cancel` naming a foreign job id (AR-LN-007: foreign ids never reach `scancel`; they are counted as the
  named drop `refused_foreign_job:<id>`).
- REFUSED (96): an invalid `proposer` block (AR-IN-008: fix the spec's `proposer` block); with `require_model`
  set, a model proposer that is unavailable (`proposer_unavailable:<reason>`) or whose fresh re-proposal is not
  byte-identical (`proposer_unverified:<name>`) is refused (AR-PR-001, nothing appended). Without
  `require_model` those same conditions never crash: the catalog speaks (mandatory fallback) and the drop
  `proposer_fallback_catalog:<reason>` is counted. An `optuna-cma` axis of `type == "categorical"` is refused at
  check time (AR-PR-002: `proposer_axis_unsupported:optuna-cma:<key>`, one per categorical axis in axis order) -
  the axis is never dropped: use `optuna` (TPE holds categorical axes) or remove the categorical axes.
- REFUSED (96) / fallback (llm, M5b): an unpinned or credential-leaking endpoint is AR-PR-005 (fix `llm_pool` /
  `llm_pool_file`: the entry must carry `pool_key` + `model`, be active, and hold only an auth env var NAME) and
  nothing is appended; a credential-shaped token in the prompt/response is refused the same way. A tainted response is
  AR-PR-003 (strict REFUSE) or the catalog fallback `llm_response_tainted:<i>`; llm `max_calls` counts ledgered calls
  and an exhausted budget drops `llm_call_budget_exhausted`. Non-strict llm failure falls back to the catalog with
  `proposer_fallback_catalog:<reason>` (strict: AR-PR-001 `proposer_unavailable:<reason>`): `no_calls_left`, `no_axes`,
  `no_transport`, `transport_error`, `parse_error`, `parse_unstable`, `tainted`, `no_model_cards`, `model_error`. At
  close, an llm response whose re-parse differs is AR-HO-009 (RED, the outcome never moves); an llm replay is never
  claimed byte-identical (AR-PR-004).
- RED (5): the campaign measured something and it did not accept - read `decisions` in the report and the
  `reasons` lines (`guardrail:<name>`, `mean_delta ... > tau ...`), or AR-HO-003 for a damaged chain. A recorded
  proposal that does not replay byte-identically at close (AR-HO-007) turns the status RED with `outcome`
  unchanged and the recommendation prefixed "investigate proposal replay drift before adopting: ".
- Replay unmeasured: a close-time re-check that cannot run is the counted drop
  `proposal_replay_unmeasured:<reason>:<i>` (reasons: `legacy_proposal`, `recorded_unmeasured`,
  `ledger_prefix_missing`, `extra_missing:<name>`, `version_changed:<name>`, `model_error`, `chain_unverified`) -
  a declined claim, never a pass, and it does not change the status.
- UNMEASURED (95): calibrate first - `baseline_repeats` ok baseline results with the eval policy
  fingerprint of the spec, and `confirm_repeats` paired seeds per candidate, none of them limited/crashed.
  M1: the floor must come from **eval-only** `baseline_repeats` jobs (AR-HO-006), and a launch that reports no
  `job_id` is UNMEASURED (`drops: ["no_job_id"]`) and leaves no `job_submitted` entry behind.
- Measured GPU hours (fallback named drops): `sacct` measurement replaces `gpu_hours_est` per job when it is
  available and > 0; otherwise the declared `gpu_hours_est` counts with `sacct_unavailable:<id>`, and a job with
  neither counts 0.0 with `no_accounting:<id>` - one job is never both, and every drop lands in
  `close.budgets.drops`. A measurement of `<= 0` is **not** a measurement: it falls back to declared/uncounted
  and drops `sacct_zero:<id>` instead of `sacct_unavailable`. A job entry with no `job_id` is never measured
  (`measure` is not called for it) and drops `sacct_unavailable:entry:<index>` / `no_accounting:entry:<index>`;
  a repeated `job_id` is counted once and its later entries drop `duplicate_job:<id>` and no hours. The
  envelope gate meters the `job_submitted` payloads **plus** one synthetic `{job_id: None, trial, gpu_hours_est}`
  per `launch_authorised` payload whose trial has no job entry (a budget burn without a job id), and the run
  count is the ledger launch payloads (`Ledger.launches`) reconciled with those job entries - one run attempt
  counts once, whatever recorded it.
- Nothing in this skill repairs or rewrites a ledger: entries are append-only. Damage is detected
  (AR-HO-003 names the first broken seq), never patched.
- Limitations: the hash chain is not anchored outside the ledger file, so it detects edits that do not
  recompute the chain, not a full rewrite; `gpu_hours_est` is self-declared at launch and `sacct`-measured
  hours replace it per job (the fallback is named `sacct_unavailable:<id>` / `no_accounting:<id>` and counted).
  Fabric TTL cache staleness is bounded by `ttl_s` but is not a run-time proof - the in-script `exit 96`
  preamble remains the in-job IMEX gate. A guardrail with fewer than 2 paired measurements makes the
  close UNMEASURED (`guardrail_unmeasured:<name>`), never an accept. Ledger anchoring (Q5) and approver
  binding (Q6) remain open (amendment A5). A model card is a suggestion, not evidence (it reaches a run only
  through `submit` and the M1/M2 gates); TPE determinism holds per optuna version + seed only (the replay is
  checked at propose time and re-checked at close - the re-check proves reproducibility under the installed optuna
  version only); optuna card attribution to a single idea is weaker (a card is a full
  assignment over `spec.axes`, B5). Optuna is not installed by default: model replays on a host without the extra
  are unmeasured (`extra_missing`), not passes.

## FS interface
- Emits: `auto_research_report` artifact in `ctx.artifacts_dir`.
- Consumes: nothing from FS artifacts in M0 (`consumes: ()`).
- APIs: `campaign_hash`, `launch_token`, `check_spec`, `check_launch`, `noise_floor`, `decide`, `propose`, `proposers.select`, `proposers.replay`, `proposers.reverify`, `proposers.package_version`,
  `squeue.job_states`, the claims/seeds/concurrency/locks helpers,
  `Ledger` (append/verify/results/launches/tsv_view). `squeue` is read-only. No processes are launched.

## Increments
- **M0 (done)**: validate, authorise (token), record, propose, decide, close - CPU-only, no submission.
- **M1 (done)**: `envelope`/`submit`/`cancel` via `training.emit` + the IMEX fabric probe + `sacct`-measured
  GPU hours; the envelope budget and the derived per-trial token gate every `job_submitted`.
- **M2 (done)**: seed phases, the explicit claim chain, the in-flight cap + confirm run reserve, the close lock.
- **M3 (done)**: pluggable proposer (`proposer` config block via AR-IN-008, one `proposal` op per `propose`,
  deterministic catalog default + optional optuna TPE extra, mandatory catalog fallback / AR-PR-001 strict mode).
- **M4 (done)**: `optuna-cma` + AR-PR-002 (categorical axes refused at check time), the ledgered replay record
  (`replay_inputs`, `package_version`, `replay_status`) on every `proposal`, and the close-time re-check
  (`proposers.reverify`) + AR-HO-007 drift escalation.
- **M5a (done)**: multi-objective campaigns (`spec.objectives`, AR-IN-009), `decide_multi` over one common
  seed set, AR-RS-008 (trade-offs never claim), AR-HO-008 and the presentation-only frontier.
- **M5b (done)**: LLM proposer (`proposer.llm`, AR-IN-010; request-only endpoint `llm_pool`/`llm_pool_file` + AR-PR-005;
  typed cards that drop, never clamp; record-and-audit = AR-PR-003/-004 + AR-HO-009; `llm_usage` budgets); Ray Tune /
  Vizier stay reference-only.
- **M6a (done)**: evidence-bound `record` (`evidence.run_manifests`/`eval_report`, AR-IN-011/-012), the rl-stage
  floor + guardrail (AR-IN-013), unmeasured/asserted claim refusal (AR-RS-009), close-time re-hash and AR-HO-010.

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

4. Consent an envelope, submit one emitted trial (M1), and cancel what was submitted:
   ```python
   from foundationskills.skills.auto_research.envelope import envelope_token, trial_launch_token

   env = {"budget": {"max_runs": spec["budget"]["max_runs"],
                     "gpu_hours_total": spec["budget"]["gpu_hours_total"]}, "scope": "one campaign"}
   request = {"action": "envelope", "campaign_spec": spec, "campaign_confirm": confirm_hash,
              "approver": "reviewer", "ledger_dir": ledger_dir, "envelope": env}
   env_token = execute(request)["envelope_token"]  # one human consent = the campaign envelope token

   trial_spec = {"trial": "lr-down", "role": "candidate", "kind": "eval_only", "delta": {"optim.lr": 5e-4},
                 "seed": 101, "nodes": 1, "gpus_per_node": 8, "partition": "rally",
                 "gpu_hours_est": 4.0, "eval_request": {"command": ["eval", "run", "gsm8k"]}}
   launch_token = trial_launch_token(env_token, trial_spec)  # derived per-trial, budget-decremented
   request = {"action": "submit", "campaign_spec": spec, "campaign_confirm": confirm_hash,
              "approver": "reviewer", "ledger_dir": ledger_dir, "trial_spec": trial_spec,
              "launch_token": launch_token}
   execute(request)   # -> {"job_id": "4242", "budget_after": {"runs_left": 5, "hours_left": 20.0}}
   request = {"action": "cancel", "campaign_spec": spec, "campaign_confirm": confirm_hash,
              "approver": "reviewer", "ledger_dir": ledger_dir, "job_ids": ["4242"], "reason": "operator"}
   execute(request)   # -> {"cancelled": ["4242"], "drops": []}  (AR-LN-007: foreign ids never reach scancel)
   ```

## Attribution
Adapted from NeMo-RL nemo-rl-auto-research (Apache-2.0): loop shape, stop rules, ideas catalog, TSV view.
