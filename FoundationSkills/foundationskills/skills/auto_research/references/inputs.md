# Inputs: full field reference

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
