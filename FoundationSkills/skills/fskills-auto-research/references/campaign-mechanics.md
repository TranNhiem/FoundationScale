# Campaign mechanics: seeds, proposers, multi-objective, evidence, increments

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
