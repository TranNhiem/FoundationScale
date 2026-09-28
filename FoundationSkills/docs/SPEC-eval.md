# Spec: FoundationSkills eval skill

## Assumptions

1. `lm-evaluation-harness 0.4.12` is installed and importable in the eval environment, and `lm-eval run --help` exposes the flags verified in `artifacts/research/eval_harness.md` §7. If the version drifts, the skill refuses (exit 96).
2. `torch 2.12.1`, `transformers 5.13.0`, `peft 0.19.1` are present; the HF backend (`--model hf`) is the only backend. No vLLM, no API backends.
3. A PEFT adapter directory contains `adapter_config.json`. The base model is taken, in order, from `--base`, then the training plan/run manifest, then `base_model_name_or_path` **only if it is an existing local path**. There is no model registry on this estate; an unresolvable base is a refusal, not a fallback.
4. An eval dataset cache root is passed explicitly (`--eval-cache`, no default; absent ⇒ refusal). It is pre-staged by an operator process; this skill never downloads data. (`/datasets` does not exist on the GB200 estate; the path is an open question.)
5. Harness results JSON at `--output_path` contains per-task `results`, and stderr where the metric defines one. Tasks lacking a measurable stderr must have an explicit `abs_epsilon` in policy, or the policy is invalid (refusal).
6. The training skill has already produced a launch result and FS run manifest (`<output_dir>/<stage>/run_manifest.json`, next to `final/` and `checkpoint-N/`) naming the checkpoint and the base model; `fskills eval` consumes those as named inputs and does not guess them.
7. The confirm-hash flow (`plan -> hash -> launch --confirm`) and its hashing helpers live in `foundationskills.core` / `foundationskills.interfaces.fs` and are reusable as-is for a GB200 Slurm eval job.
8. `eval_report.json` allows `additionalProperties` at top level and per benchmark, so all provenance lands inside the report; no sidecar file exists.
9. Exit-code semantics: 96 REFUSED writes **no** report; 5 RED means a measurement completed and failed (or something was skipped); 95 UNMEASURED means a benchmark could not run and nothing measured breached.
10. All benchmarks are judge-free (loglikelihood / exact-match). Any requested task that requires an LLM judge is refused by name.
11. Checkpoint and baseline are always evaluated with identical harness flags. The chat template is applied only when both models ship one (SFT-family); CPT/base comparisons run without it. The flag is recorded per benchmark.
12. A run with `--limit` is a smoke run: it is marked `limited: true` and can never return PASS (best case UNMEASURED, 95).

## Objective

**User stories.**

- As a training operator, after a CPT/SFT/LoRA run, I want `fskills eval` on the run's checkpoint so that I get a PASS/RED verdict against the base model without hand-writing prompts or scorers.
- As an auditor, I want every number in `eval_report.json` attributable to a pinned harness, task-config hash, few-shot count, seed, and chat-template flag, so a third party can reproduce it.
- As a cluster admin, I want eval to run fully offline inside the sqsh/enroot job and to name exactly which dataset snapshot is missing when it cannot run, rather than downloading anything.
- As a CI owner, I want baseline scores cached and keyed by a fingerprint, so repeat evals of the same base model cost nothing and silently-different configs can never reuse a stale baseline.

**Acceptance criteria.**

1. `fskills eval run --checkpoint <dir> --base <dir> --benchmarks a,b --policy eval_policy.yaml` emits a schema-valid `eval_report.json` and exits per the code table.
2. A PEFT adapter checkpoint is merged with its resolved base via `--model_args pretrained=$BASE,peft=$ADAPTER,...`.
3. A benchmark whose dataset is absent from the eval cache is reported `score: null` with the missing dataset named, and forces verdict UNMEASURED (95) unless a measured breach already forced RED.
4. A skipped benchmark (configured, not executed) is RED (5) — skips count as failures.
5. Baseline cache hit is reused and marked `provenance: cached`; miss + base present runs the baseline in the same invocation with identical flags.
6. No network: env exported per subprocess includes `HF_HUB_OFFLINE=1`, `HF_DATASETS_OFFLINE=1`, `HF_HOME=<eval cache>`, `TOKENIZERS_PARALLELISM=false`.
7. `fskills eval emit` produces an fs_launch_spec for a GB200 Slurm eval job; `fskills eval launch --confirm <hash>` is the only submission path.

## Tech Stack

- Python ≥ version used by the training skill; stdlib + existing `foundationskills.core` only for glue.
- **lm-evaluation-harness 0.4.12** (pinned; verified via `importlib.metadata.version("lm_eval")` at skill start, mismatch ⇒ REFUSED).
- transformers 5.13.0, torch 2.12.1, peft 0.19.1 (inherited from the eval environment; asserted, not managed).
- PyYAML (or repo-standard parser) for task configs and policy; pytest for tests.
- No vLLM, no sglang, no API keys, no network access.

## Commands

```bash
# Verdict run against a training run manifest (resolves checkpoint + base).
PYTHONPATH=../src python -m foundationskills.cli eval run \
  --run-manifest <output_dir>/<stage>/run_manifest.json \
  --benchmarks mmlu,hellaswag,gsm8k_cot \
  --policy foundationskills/skills/evaluation/eval_policy.yaml \
  --eval-cache <staged HF_HOME for eval> \
  --out artifacts/eval/<run_id>/eval_report.json

# Explicit form (checkpoint may be a full HF dir or a PEFT adapter dir).
PYTHONPATH=../src python -m foundationskills.cli eval run \
  --checkpoint /models/ckpt/step-20000 [--base /models/base/qwen3.5-8b] \
  --benchmarks mmlu --num-fewshot 5 --seed 1234 \
  --gen-kwargs temperature=0 --eval-cache <staged HF_HOME> \
  --policy foundationskills/skills/evaluation/eval_policy.yaml \
  --out artifacts/eval/<run_id>/eval_report.json

# Print the confirmation hash of an eval plan/spec.
PYTHONPATH=../src python -m foundationskills.cli eval hash --spec fs_eval_launch_spec.json

# Emit a GB200 Slurm spec; nothing is submitted.
PYTHONPATH=../src python -m foundationskills.cli eval emit \
  --checkpoint ... --benchmarks ... --nodes 1 --gpus 4 --out fs_eval_launch_spec.json

# Submit only with a confirm hash (same flow as training).
PYTHONPATH=../src python -m foundationskills.cli eval launch --spec fs_eval_launch_spec.json --confirm <sha256>

# Tests (from FoundationSkills/; the gate after every edit).
PYTHONPATH=../src python -m pytest -q

# Countables loop (from the repo root; repeat until VERDICT CLEAR).
git add -A && python tools/countables_census.py --no-coverage --out /tmp/census.json
python checks/countables_drift.py --census /tmp/census.json --fix docs README.md Makefile .github/workflows/ci.yml
```

Harness subprocess constructed by the runner (never string-concatenated with user text unescaped):

```bash
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HOME=$EVAL_CACHE \
TOKENIZERS_PARALLELISM=false \
lm-eval run --model hf \
  --model_args pretrained=$CKPT[,peft=$ADAPTER],dtype=bfloat16 \
  --tasks $TASKS --num_fewshot $N --seed $SEED \
  [--apply_chat_template] --gen_kwargs temperature=0 \
  --log_samples --output_path $RUN_DIR --include_path $CUSTOM_TASK_DIR
```

Exact task-dependent flags (`--fewshot_as_multiturn`, `--system_instruction`) come from the per-task entry in policy, not from free-form CLI.

## Project Structure

```
foundationskills/skills/evaluation/
  __init__.py
  skill.py            # EvalSkill: inputs -> harness -> policy -> eval_report.json
  runner.py           # HarnessRunner protocol + SubprocessLmEvalRunner (verified flags only)
  policy.py           # eval_policy.yaml load/validate; stderr-band comparison, one-sided
  baseline.py         # fingerprint, cache get/put at artifacts/eval/baselines/<base>/<fp>.json
  resolve.py          # adapter_config.json -> base resolution; run-manifest consumption
  report.py           # eval_report.json build + atomic write (tmp + os.replace)
  emit.py             # fs_launch_spec for the GB200 Slurm eval job (confirm-hash flow)
  eval_policy.yaml    # versioned default policy (policy_version, default band, per-task overrides)
foundationskills/schemas/artifacts/   # unchanged: eval_report.json already exists
tests/unit/evaluation/
  test_policy.py      # tolerance math, one-sided rule, stderr-missing policy error
  test_baseline.py    # fingerprint stability, cache hit/miss, staleness rejection
  test_resolve.py     # adapter base resolution, manifest parsing, refusal messages
  test_report.py      # schema conformance, atomic write, UNMEASURED/skip semantics
  test_skill.py       # exit-code matrix with an injected FakeHarnessRunner (no GPU)
  test_integration_lm_eval.py  # gated on lm_eval importable; tiny CPU slice or honest UNMEASURED
artifacts/eval/baselines/             # runtime cache (not committed)
```

## Code Style

Same shape as `training/skill.py`: dataclass-lite glue, `SkillResult`/`Finding` from `foundationskills.core`, refusals naming inputs, rule ids as `RuleSpec(id, description, Severity, phase)`.

```python
# Same shape as training/skill.py; every rule needs a must-fire negative fixture (core/contract.py).
RULES = (
    RuleSpec("EV-IN-001", "checkpoint missing or not a directory", Severity.BLOCK, "input"),
    RuleSpec("EV-IN-002", "adapter checkpoint with no resolvable local base model", Severity.BLOCK, "input"),
    RuleSpec("EV-IN-003", "eval cache (--eval-cache) missing", Severity.BLOCK, "input"),
    RuleSpec("EV-IN-004", "installed lm_eval is not the pinned 0.4.12", Severity.BLOCK, "input"),
    RuleSpec("EV-HO-001", "a benchmark breached its regression tolerance", Severity.BLOCK, "handoff"),
)
# refusal text names the input:
#   "missing input: base model for adapter <dir> (base_model_name_or_path=<x> is not a local path)"
```

## Testing Strategy

- **Rule fixtures:** every `EV-*` rule has a must-fire negative test (the `core/contract.py` requirement).
- **Unit (no GPU, no network, no lm_eval import):** `test_skill.py` injects a `FakeHarnessRunner` (same pattern as the launch-skill's injected runner) that returns canned results-JSON / raises harness errors. Assert the full exit matrix: PASS(0), RED(5) on breach, RED(5) on skip, UNMEASURED(95) on per-task error + `score: null`, REFUSED(96) with `missing input:`/`precondition failed:` on stderr and **no report file created**.
- `test_policy.py`: exact boundary cases of `drop > max(abs_epsilon, k*sqrt(se_b^2+se_c^2))`, improvements never fail, missing stderr without `abs_epsilon` ⇒ policy error refusal.
- `test_baseline.py`: fingerprint changes when any hashed field changes (harness version, task yaml sha, fewshot, seed set, gen_kwargs, chat-template mode, dtype, backend); cached provenance is marked and never silently refreshed.
- `test_report.py`: output validates against `foundationskills/schemas/artifacts/eval_report.json`; extra fields (stderr, n, harness_version, task sha256, policy fingerprint, command argv, dataset revisions, container hash) present inside the report.
- **Integration:** `test_integration_lm_eval.py` skips via `pytest.importorskip("lm_eval")`; when present, asserts version pin, then runs `--limit 8` on one staged loglikelihood task, or — if the dataset isn't in the cache — asserts the skill reports that benchmark UNMEASURED with the dataset named and exit 95. A skip in this test is failure-honest: it must not be reported as PASS.

## Boundaries

**Always:** run the harness via subprocess `lm-eval run`; pin-check `lm_eval` at start; export the offline env vars per subprocess; fingerprint baselines; write reports atomically; name the missing input in every refusal; keep verdict logic one-sided; treat a skipped benchmark as failure.

**Ask first:** adding a benchmark to policy (judge-free check + staged dataset both required); changing tolerance defaults (`k`, `abs_epsilon`) — calibration runs required; overriding `--base-model` for an adapter; introducing a second harness version alongside 0.4.12.

**Never:** implement or vendor a scorer; accept baseline numbers lacking a matching fingerprint; download datasets or models at run time; auto-submit a Slurm job without `--confirm <hash>`; write `eval_report.json` on refusal; return PASS from a `--limit` run; pass on a skip; enable `trust_remote_code` outside the policy allowlist; invoke lmms-eval or any API backend.

**VLM is out of scope.** lmms-eval is the named follow-up integration, deferred because it is not installed in this environment and is unverified against `transformers 5.13.0` (import/​API-conflict risk per the research record, §6-R2). The policy/report/baseline layers here are designed so a second `HarnessRunner` (`python -m lmms_eval`) drops in without changes to verdict logic.

## Success Criteria

1. `PYTHONPATH=../src python -m pytest -q` passes with all new unit tests; integration test passes or reports UNMEASURED honestly where data is unstaged.
2. On a staged box: base-vs-base eval of one task exits 0 and produces identical scores on repeat with `provenance: cached` for the second run's baseline.
3. Removing one dataset snapshot from the cache produces exit 95, `score: null`, and the missing dataset named — with zero network attempts (verifiable by running with the interface unbound/disabled).
4. A 3-point synthetic drop injected via FakeHarnessRunner yields exit 5 and RED; a 3-point improvement yields exit 0.
5. Every refusal exits 96, writes no report, and its message begins `missing input:` or `precondition failed:`.
6. `eval emit` produces a spec whose `fskills eval hash` matches `launch --confirm` expectations, and `launch` without `--confirm` refuses.
7. `eval_report.json` validates against the existing schema and round-trips through `sha256_json` reproducibility checks.

## Open Questions

1. Eval cache root: `/datasets` does not exist here — which shared path is sanctioned (e.g. under the account's NFS home, or node-local on each GB200), and who owns the offline staging job (`prestage_eval_data.py` equivalent)?
2. Chat template: this spec applies it only when both checkpoint and baseline ship one (assumption 11). Confirm, or should SFT checkpoints also be scored without it against a base baseline?
3. Default benchmark suite and few-shot counts per stage (CPT vs SFT vs RL) — which tasks are contractual for the estate?
4. Tolerance calibration: after base-vs-base noise-floor runs across nodes/seeds, should `eval_policy.yaml` bump to `policy_version: 2` with measured `abs_epsilon` per task?
5. Multi-GPU recipe on one GB200 node: `device_map=auto` vs `accelerate` wrapping for the 27–35B fleet models — bring-up measurement decides; is a 2-hour suite budget the right gate (R1)?
6. Where do RL/agentic evals land — a later Inspect AI-based sibling skill, or extensions of this one?
7. Baseline for a later stage: for SFT after CPT, is the baseline the original base model, the CPT checkpoint, or both (regression vs. gain)?
8. Baseline cache GC policy: retention window / size cap for `artifacts/eval/baselines/`?
