# FoundationScale Agentic RL

Multi-turn, tool-using reinforcement learning on the FoundationScale RL plane:
a policy acts in an environment over many turns (generate → call a tool →
read the observation → generate again), and the whole trajectory is scored
once. This page records what has landed.

## What exists today

| Piece | Where | What it guarantees |
|---|---|---|
| Trajectory contracts | `src/foundationscale/agentic_rl/contracts.py` | `Trajectory` / `Turn` / `ToolCall` are frozen and validated; `loss_mask` is 1 only on engine-generated assistant / tool-call tokens (never on tool results, observations, user or system text) and all-0 on an INFRA trajectory; `reward is None` exactly when an `abstention_reason` is given. `flatten()` emits one row per trajectory with exactly `DECLARED_COLUMNS`; `prompt_ids` is the group key (`uid`, or `uid::harness` with `group_by_harness=True`). |
| Token trace | `src/foundationscale/agentic_rl/token_trace.py` | Incremental token buffer for a multi-turn episode: ids travel with the episode and are never re-tokenised; over-budget responses are refused, not clipped. Adapted from an Apache-2.0 source -- see `THIRD_PARTY_NOTICES.md`. |
| `SessionGroupAdvantage` | `src/foundationscale/rl/advantage.py` | Group baseline that admits `None` rewards (infra failures, unmeasured verdicts) and removes them from the baseline instead of scoring them 0.0, under a declared `partial_group_policy` (`refuse` / `drop_group` / `shrink(min_valid)`). Centred by default; std normalisation is opt-in. Equal-reward groups are dropped by exact equality. |
| `prompt_mean` reduction | `rl/group_policy_objectives.py`, `rl/torch_backend.py`, `rl/trainer.py` | Each prompt group contributes equally to the loss regardless of how many tokens its trajectories hold. HF/FSDP2 lane only; the Megatron lane refuses it by name. |
| `agentic_grpo` algorithm | `rl/group_policy.py`, registry | Token ratio, clip (0.8, 1.2), `SessionGroupAdvantage`, `prompt_mean`, structurally no KL term. |
| `agentic_grpo_token_mean` ablation | `rl/group_policy.py`, registry | The control arm for measuring `prompt_mean`: identical to `agentic_grpo` (estimator, shrink policy, `min_valid=2`, clip) except `AgenticGRPOLoss(reduction_mode="token_mean")`, one denominator over every supervised token in the batch. Select with `policy.algorithm = "agentic_grpo_token_mean"`. Not a recommended default. |
| Environments | `agentic_rl/envs/` | `EnvBackend` registry and `Environment` protocol; the `local` backend runs model-chosen commands in a fresh temp dir with an explicit environment allow-list (plus `HOME`/`PWD` = workdir), its own process group killed on timeout, and head/tail output caps whose dropped bytes are counted. It has no isolation, so it refuses to start unless `EnvSpec.allow_unsandboxed=True`. Infrastructure failures raise `EnvInfraError(kind)` and become `infra:<kind>` abstentions. |
| Environment signature | `agentic_rl/envs/env_signature.py` | ARL-18 slice 1, off by default. `env.signature: "manifest"` (rootfs names, kinds, modes, sizes, symlink targets; no timestamps, symlinks never followed) or `"content"` (also file bytes) makes the `enroot_task` environment fingerprint its started state -- rootfs minus hidden paths and `proc`/`sys`/`dev`/`run`/`tmp`, image file digest, workdir, test command, setup commands and the sealed hidden vault -- and report it as `env_signature` in verifier metadata. A rootfs it cannot read gives `"unmeasured"`, never a partial signature. It is a comparison aid for reset fidelity, not an isolation proof. Paths `enroot start` recreates on every exec (`RUNTIME_ROOTFS_FILES`: the container lock, `/etc/hosts`, `/etc/machine-id`, the apt hook) are excluded. |
| Snapshots | `sandbox/snapshot_store.py`, `envs/sandbox_task.py` | ARL-18 slices 2-3. `SnapshotStore`: a snapshot is visible only once its JSON descriptor is committed (written last); archives are size- and sha256-checked on commit and on every restore; a reservation quota bounds disk use; leased snapshots are never reaped. `TaskSandboxEnvironment.snapshot(store, vault_root=...)` exports the rootfs (`enroot export` under `prlimit --fsize`) while the hidden tests are sealed, clones the vault outside the archive, and refuses live processes and setuid/setgid entries (rootless extraction clears those bits). `restore(store, snapshot_id)` replaces `start()`: no setup replay, an independent vault copy, and the restored signature must equal the recorded one or the container is destroyed (`infra:snapshot_integrity`). |
| Tools and parsers | `agentic_rl/tools.py`, `agentic_rl/markup.py` | `ToolSpec`, the Qwen-XML and Hermes-JSON tool-call parsers (malformed markup is a model-attributable `parse_error`, never an exception) and `validate_call`. Every chat special token and tag is spelled once, in `markup.py`. |
| Native multi-turn loop | `agentic_rl/harness/` | `NativeToolLoop`: generate → parse → execute → observe until `submit`, a final answer, the step limit, the response budget, an engine abort or an infrastructure failure. Token ids come from the engine and are never re-rendered; observation tokens are prefix deltas of the chat template, refused if the template is not prefix-stable. |
| Inference engines | `agentic_rl/engines/` | Token-in/token-out `SGLangClient` (native `/generate`) and `VLLMClient` (OpenAI-compatible `/v1/completions`), each with logprobs and a declared weight-reload/control surface. `EngineServerSpec`/`EngineServer` are the generic pair (full declared argv, the right client class per `kind`); `SGLangServerSpec`/`SGLangServer` keep working unmodified. `EngineFleet.clients()` returns one client per server, `client_for()` round-robins. HTTP field names are unprobed until measured on a cluster; see the "UNPROBED ASSUMPTIONS" section in each client module, especially `engines/vllm.py`'s on `/collective_rpc` reload. |
| Weight sync | `agentic_rl/weight_sync.py` | `DiskWeightSync`: every rank writes the HF checkpoint, rank 0 pushes it, under a declared `mode`. `"reload_endpoint"` (default) asks each server's client to reload from disk over its own control endpoint; `"restart"` stops and relaunches each `EngineServer` with the new checkpoint path substituted into its declared `{model_path}` command placeholder (`EngineServer.start(command=...)` never mutates the declared spec) -- the guaranteed-correct path for an engine (vLLM, today) with no proven reload endpoint. `config.py` refuses, at config time, a `weight_sync_mode="restart"` declaration whose `engine.command` never names the placeholder; `cli.py`'s first fleet launch substitutes `policy.model_path` for it too, so the placeholder never reaches the engine literally. `is_stale=False` is derived from every push being a complete, synchronous reload, never a separate measurement. Old publish directories are pruned strictly inside `publish_root`. |
| Rewards | `agentic_rl/rewards/` | `RewardFn` / `RewardVerdict` (a verdict with no value must carry a reason). The rule-scored music reward is a typed port of an Apache-2.0 scorer: model-attributable failures score a measured 0.0, a missing or failing `abc2midi` abstains. |
| Rollout host | `agentic_rl/rollout_host.py`, `agentic_rl/tasks.py` | Draws tasks, runs `group_size` concurrent sessions per task (each in its own environment), scores them and returns the `flatten()`ed batch; infrastructure failures become INFRA trajectories, programming errors propagate. |
| Serving completion | `agentic_rl/servable.py` | `complete_for_serving(publish_dir, base_model_dir, cache_dir=...)`: when `policy.model_path` is a multimodal model loaded as text-only, `rl.distributed.save_checkpoint` writes a published checkpoint whose `config.json` names a text-only architecture and whose safetensors hold only the language tensors -- a fresh vLLM/SGLang load of that directory fails (reload into a running engine still works). `RolloutHost.publish` calls it on rank 0, between the collective save and the push, whenever `servable_base_model_dir` is set (`policy.servable_publish`, default `true`). It refuses (`ServableRefusal`) if any published tensor name is not a base tensor name or its shape disagrees with the base's, else extracts exactly the base tensors the published checkpoint lacks into one `model-serving-extras.safetensors` file, cached under `{publish_root}/.serving_cache` and hard-linked (falling back to a copy only on a cross-device/permission error) into the published directory; the base model's own shard files are never linked or copied in. Config and any declared preprocessor/chat-template files are copied from the base model, with the published config kept as `config.trained_text_only.json` for provenance, alongside a `fs_servable.json` summary. A no-op (`"already_servable"`) when the published `architectures` already match the base's. |
| Trainer rows path | `rl/trainer.py` | `RLTrainConfig.rollout_source`: each step consumes the agentic batch instead of the built-in single-turn generate-and-score leg. The per-token mask comes from `loss_mask` (tool tokens stay 0), padding is by length, `None` rewards are dropped, old logprobs are still recomputed, and `publish()` runs after every optimizer step. Refused for PPO, online preference and estimator-free algorithms. |

## Gates

Five gates in `src/foundationscale/gates/agentic_gates.py` follow the contract
in `docs/CUSTOM_GATES.md`: each ships `MUST_FIRE`/`MUST_PASS` controls and runs
under `foundationscale-controls`. They run at two `Lifecycle` points added here,
append-only: `ROLLOUT` (after a step's batch is flattened) and `WEIGHT_SYNC`
(after a weight-sync push). Wired through `RolloutHost` (`gates: bool = True`;
`declared_max_infra_rate` / `declared_max_lag` fields, set from
`rollout.max_infra_rate` / `engine.max_policy_lag`), which raises
`RolloutHostRefusal` on a blocking report.

| Gate | Event | What it catches |
|---|---|---|
| `TrajectoryIntegrityGate` | `ROLLOUT` | Per-row structural integrity of the flattened batch: a loss mask entry outside `{0, 1}`, a per-token column that runs its own length, supervision on a non-assistant/tool-call token, an INFRA row with a non-zero mask, or a reward/abstention pair that contradicts itself. |
| `RolloutAbstentionGate` | `ROLLOUT` | The batch's infra-abstention rate against a declared bound; an undeclared bound abstains (`NOT_ESTABLISHED`) rather than inventing one, and a 100% infra rate is measured like any other value. Because an abstaining gate cannot stop a dead engine, the CLI refuses any run with `trainer.max_steps > 1` that leaves `rollout.max_infra_rate` undeclared (0.25 suits single-turn tasks). |
| `StalenessGate` | `WEIGHT_SYNC` | A pushed weight sync's measured `is_stale` claim, and every supplied rollout session's policy version against the declared lag; `offered == 0` abstains `NOT_APPLICABLE`. |
| `WeightSyncParityGate` | `WEIGHT_SYNC` | Engine/learner logprob agreement within a declared tolerance, when a parity probe supplies pairs; `parity=None` abstains `NOT_ESTABLISHED` until a GPU probe slice measures it. |
| `PromptBytesGate` | `ROLLOUT` | The rendered system prompt and tool schemas hash (`prompt_bytes_sha256`) to the value declared at launch; an unrecorded declaration abstains `NOT_ESTABLISHED`. |

**Engine hang recovery.** `rollout.engine_recovery_retries` (default `0`, off)
lets `RolloutHost` recover a step whose every attempt was an engine infra
failure -- the signature of a vLLM engine that hung after an in-place weight
reload. It restarts the fleet, re-pushes the last published checkpoint so the
engine serves the current policy rather than the base model, and re-runs the
same step with the same tasks, logging one `[agentic-rl:engine-recovery]` line
per attempt. Env failures never trigger it; once the retries are spent,
`RolloutAbstentionGate` refuses as before. `rollout.recovery_rerun` declares
what a retry re-runs: `"all"` (default) re-gathers every attempt, while
`"engine_rows"` re-runs only the engine-failed rows and keeps the measured
ones -- refused if the kept and re-run rows report different policy versions.

**SWE tamper guard (DiffGuard).** The `enroot_task` environment fingerprints,
from the host side, every test-harness file the agent could edit to change a
test run without touching a hidden test -- `conftest.py`, `pytest.ini`,
`tox.ini`, `setup.cfg`, `pyproject.toml` under the workdir, and
`sitecustomize.py`/`usercustomize.py`/`*.pth` in site-packages -- at the end
of `start()` and again before `verify()` unseals the hidden tests. It reports
`swe_guard` and `swe_tampered` (changed paths) in the outcome metadata.
`reward.swe_tamper="zero"` scores a tampered episode 0.0
(`test_harness_tampering`) and abstains when the guard was unmeasured;
`"report"` (default) keeps scoring unchanged. The SWE example config uses
`"zero"`.

**Judge rewards.** `reward.kind="judge"` scores the final answer with an
OpenAI-compatible judge model against a declared rubric
(`reward.judge_rubric_path`: weighted criteria, each scored in [0, 1]) --
`rewards/judge.py`. The answer is fenced between per-call random delimiters
and declared data; a deterministic injection floor scores an answer that
addresses the grader as a measured `0.0` (`judge_injection_attempt`). Judge
transport failures abstain as `infra:judge_<kind>`, malformed output as
`judge_malformed`, and `reward.judge_votes > 1` takes the median and abstains
(`judge_disagreement`) when the votes spread past `reward.judge_max_spread`.
The run refuses to start unless `reward.judge_calibration_path` holds a
calibration report for the same rubric with agreement against gold labels
of at least `reward.judge_min_agreement` over `reward.judge_min_n` items
(`calibrate`/`require_calibrated`), and, when `reward.judge_api_key_env` is
declared, unless that variable is set; the key itself is read at call time
and never stored or logged. `judge_base_url` is the server root (the client
appends `/v1/chat/completions`).

**Process shaping.** `policy.algorithm="agentic_process_grpo"` trains with
per-turn penalties on tool-call parse errors and repetition, added to the
advantage after the baseline (never to the reward); see
[RL_ALGORITHMS.md](RL_ALGORITHMS.md).

**Episode traces.** `rollout.trace_dir` (default `null`, off) writes every
step's final trajectories as `fs.trace/v1` span traces to
`<trace_dir>/step_<NNNNNN>.jsonl` (`agentic_rl/fs_trace.py`): a session span,
one span per prompt/response turn (exact token ids, logprobs, flags), one per
tool call (raw arguments and parse verdict), a verifier span holding exactly
what a verifiable environment's `verify()` reported (hidden-test results for
SWE), and a reward span joined to the session it scores and to that verifier
span. Span ids are deterministic hashes of the run (the trace
directory's name), step, session and structural position. `fold(spans)`
rebuilds exactly the original `Trajectory`, and `fork_prefix(spans,
turn_index=k)` names a new branch with the token-id prefix before response
turn `k` -- the contract later snapshot/branching work builds on; nothing is
executed from a fork yet. For crash evidence, the native tool loop also tells
a `LiveTraceWriter` immediately before and after every generation (with the
session's policy version) and every tool execution, fsynced to
`<trace_dir>/live/step_<NNNNNN>/<session>.jsonl`; `last_open_operation()`
returns the operation a hung or killed episode had started but not finished.
A session's live file and its episode trace carry the same trace id.

**Frontier curriculum.** `rollout.curriculum="frontier"` (default `"off"`)
draws each step's tasks with weight `max(floor, 4p(1-p))`, where `p` is a
task's measured pass rate (Beta(1,1) posterior over every rollout, rewards
clipped to [0, 1]): unseen tasks and tasks near a 50% pass rate get full
weight, and a task seen `rollout.curriculum_min_obs` times with all rewards 0
or all 1 drops to `rollout.curriculum_floor`, so it is still revisited. Draws
are deterministic given the seed, the step and the registry; one
`[agentic-rl:curriculum]` line per step reports tasks seen, frontier and
saturated counts. Dynamic sampling discards zero-spread groups after their
rollouts; the curriculum avoids drawing those tasks in the first place.
`rollout.curriculum_revisit_fraction` (default 0.5) reserves that share of
each batch for tasks already measured and not saturated, weighted by
`4p(1-p)`; without it a measured frontier task competes with every unseen
task at the same weight, so in a large pool (GB200 math: about 40k tasks,
about 400 drawn per run) it is practically never revisited and the curriculum
degenerates to uniform sampling.
`rollout.curriculum_state_path` (optional) loads the registry when the file
exists and saves it atomically after every step, so a resumed run -- or a
later run on the same task pool -- keeps the measured difficulty; without it
the registry is in memory and a resumed run starts uniform.

**Dynamic sampling.** On GB200 math runs about half of all groups had zero
reward spread (every sample solved, or every sample failed), so a centred
baseline gave them zero advantage, and 7-10% of steps carried no gradient at
all. `rollout.dynamic_sampling="refill"` (default `"off"`) drops groups whose
rows are all measured with one reward value and runs up to
`rollout.max_refill_rounds` rounds of fresh tasks (one deterministic draw per
step) until `tasks_per_step` groups carry signal. A group with any abstained
or infra row is never dropped, so the infra gate still sees it; the
`[agentic-rl:rollout]` summary and the stability guard read round 0 only (an
unbiased task sample), and one `[agentic-rl:dynamic-sampling]` line reports
rounds and groups kept/dropped. With no informative group anywhere the step
keeps round 0 and is reported UNMEASURED as before.

**Publish deadline.** `rollout.publish_deadline_s` (default `null`, off) puts
a wall-clock deadline on one whole `RolloutHost.publish` -- save, serving
completion, push and any push recovery. On expiry the run prints a
`[fs:agentic-rl:red]` line, dumps every thread's stack and exits RED (5). Two
GB200 math runs each hung about two days between writing the checkpoint shards
and the push (no reload ever reached the engine), holding four GPUs that no
recovery path could see; the example configs set 10800 s.

**Per-session policy versions.** With a real `EngineFleet`, `RolloutHost`
keeps a per-server version ledger: every server starts at
`fleet_launch_version` (0, the `policy.model_path` weights `cli.py` launches),
and `publish`/engine recovery advance exactly the servers their push report
names as transferred. Each session records the version of the server it ran
on (`policy_version_min/max`), and the next `publish` hands those versions to
`StalenessGate` as `rollout_policy_versions`. Hosts built on an injected
`client_factory` cannot name the session-to-server mapping and pass `()`.

## Rules every later piece must keep

- An unmeasured reward is `None`, never `0.0`; infra failures carry
  `abstention_reason="infra:<kind>"` and leave the baseline.
- Rollout logprobs exist only for a declared off-policy correction and are
  never reused as `old_logprobs` (the trainer recomputes them).
- Token ids are never round-tripped through the tokenizer.
- Behaviour switches are declared configuration with provenance, never
  environment variables.

## Running

For a measured end-to-end run (two GB200 GPUs, a 9B policy, the music task, training and evaluation),
follow [examples/agentic-rl-GB200](../examples/agentic-rl-GB200/README.md).

One declared, provenance-tracked JSON config (`agentic_rl/config.py`'s
`AgenticRLConfig`) drives the whole run; see
`examples/agentic_rl/music_s0.json` (the music configuration) for every
section, commented by key via `_doc` strings (JSON has no comments). Validate
it -- no GPU, no server, no torch or transformers import -- before touching
hardware:

```sh
foundationscale-agentic-rl --config examples/agentic_rl/music_s0.json --dry-run
# or, without installing the console script:
python -m foundationscale.agentic_rl --config examples/agentic_rl/music_s0.json --dry-run
```

This builds and validates every object that needs neither (the tool registry
and parser, `SamplingParams`, `EpisodeBudget`, `EnvSpec`, the engine server
spec and an unstarted `EngineFleet`, the reward scorer, the task pool) and
prints the fully resolved config with per-key provenance (`"config"` /
`"cli"` / `"default"`) as JSON, exiting 0 -- with every value under
`engine.env`/`env.env` (the subprocess env allow-lists, which may carry
secrets) redacted to `"<redacted>"` in the printed output only. Override any
declared key with repeatable `--set dotted.key=value` (e.g.
`--set trainer.max_steps=5`); an override whose dotted key is never read by
any declared field (an unknown section, a typo'd field name, or a key this
loader does not allow `--set` for at all, like `engine.env` or
`engine.command`) is refused by name, never silently dropped. A real run
(omit `--dry-run`) additionally loads the tokenizer, builds the
`ChatTokenizer` adapter over `apply_chat_template(..., tokenize=True)`, starts
the engine fleet, and calls
`RLTrainer(RLTrainConfig(..., rollout_source=host)).run()`. Exit codes are the
same four `train/cli.py` uses: `0` pass, `5` red (a crash), `95` unmeasured
(`run()` reported 0 steps), `96` refuse (a declared config problem, or a
`TrainerRefusal`).

## Known gaps

- Colocated weight sync is wired into the CLI's real run: `cli.py` always
  constructs `RolloutHost` with a `DiskWeightSync` (`mode` from
  `engine.weight_sync_mode`, `publish_root` from the config's own
  `publish_root`). `RolloutHost.publish(model, tokenizer, ctx, step)` builds
  its save around the real `model`/`tokenizer`/`ctx` from `RLTrainer.run()`,
  via `rl.distributed.save_checkpoint(..., sharding=trainer.sharding)` --
  never through `weight_sync.save_fn`, which exists only for
  `DiskWeightSync.sync()`'s own non-distributed convenience path (and may be
  `None`). A push that does not fully succeed raises. The CLI refuses (96,
  named) a config declaring `trainer.max_steps > 1` with no weight sync wired,
  unless `engine.allow_stale_policy: true` is explicitly declared and recorded
  in provenance like every other key.
- `engines/vllm.py`'s `/collective_rpc` `reload_weights` assumption is
  unprobed and gated behind an explicit `reload_mode="collective_rpc"` opt-in;
  a cluster probe decides whether it is usable at all. Until then,
  `engine.weight_sync_mode="restart"` is the guaranteed-correct path for vLLM.
- Per-session policy versions come from the server ledger, i.e. from what
  each push REPORTED transferring; nothing yet asks a live engine which
  weights it holds. `client_factory` hosts still pass `()`.
- `parity`/`declared_parity_atol` are always `None` from this call
  site until a GPU probe slice exists to measure engine/learner logprob
  agreement.
- `trainer.{temperature,top_p,top_k,max_new_tokens,group_size}` remain
  vestigial under `rollout_source` except for one early greedy-decoding guard
  (`rl/trainer.py`); the sampling that actually reaches the engine is the
  top-level `sampling` config section instead.
