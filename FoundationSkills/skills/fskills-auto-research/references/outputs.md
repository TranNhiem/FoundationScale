# Outputs: report, ledger ops and per-action payloads

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
