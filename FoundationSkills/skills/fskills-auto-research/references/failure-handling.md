# Failure handling: full catalogue

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
