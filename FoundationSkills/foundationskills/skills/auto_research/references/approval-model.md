# The approval model: tokens and the render gate in full

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
