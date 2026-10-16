# Validation rules: full catalogue

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
