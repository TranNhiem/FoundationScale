# Worked examples

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
