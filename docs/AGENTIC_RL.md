# FoundationScale Agentic RL

Multi-turn, tool-using reinforcement learning on the FoundationScale RL plane:
a policy acts in an environment over many turns (generate → call a tool →
read the observation → generate again), and the whole trajectory is scored
once. This page is the working record of what has landed; the design study it
implements (feature analysis of an open agentic-RL fork of verl, the gap
analysis and the phased plan) lives outside the repository with the owner.

## What exists today

| Piece | Where | What it guarantees |
|---|---|---|
| Trajectory contracts | `src/foundationscale/agentic_rl/contracts.py` | `Trajectory` / `Turn` / `ToolCall` are frozen and validated; `loss_mask` is 1 only on engine-generated assistant / tool-call tokens (never on tool results, observations, user or system text) and all-0 on an INFRA trajectory; `reward is None` exactly when an `abstention_reason` is given. `flatten()` emits one row per trajectory with exactly `DECLARED_COLUMNS`; `prompt_ids` is the group key (`uid`, or `uid::harness` with `group_by_harness=True`). |
| Token trace | `src/foundationscale/agentic_rl/token_trace.py` | Incremental token buffer for a multi-turn episode: ids travel with the episode and are never re-tokenised; over-budget responses are refused, not clipped. Adapted from an Apache-2.0 source -- see `THIRD_PARTY_NOTICES.md`. |
| `SessionGroupAdvantage` | `src/foundationscale/rl/advantage.py` | Group baseline that ADMITS `None` rewards (infra failures, unmeasured verdicts) and removes them from the baseline instead of scoring them 0.0, under a declared `partial_group_policy` (`refuse` / `drop_group` / `shrink(min_valid)`). Centred by default; std normalisation is opt-in. Equal-reward groups are dropped by exact equality. |
| `prompt_mean` reduction | `rl/group_policy_objectives.py`, `rl/torch_backend.py`, `rl/trainer.py` | Each prompt group contributes equally to the loss regardless of how many tokens its trajectories hold. HF/FSDP2 lane only; the Megatron lane refuses it by name. |
| `agentic_grpo` algorithm | `rl/group_policy.py`, registry | Token ratio, clip (0.8, 1.2), `SessionGroupAdvantage`, `prompt_mean`, structurally no KL term. |
| Environments | `agentic_rl/envs/` | `EnvBackend` registry and `Environment` protocol; the `local` backend runs model-chosen commands in a fresh temp dir with an explicit environment allow-list (plus `HOME`/`PWD` = workdir), its own process group killed on timeout, and head/tail output caps whose dropped bytes are counted. It has no isolation, so it refuses to start unless `EnvSpec.allow_unsandboxed=True`. Infrastructure failures raise `EnvInfraError(kind)` and become `infra:<kind>` abstentions. |
| Tools and parsers | `agentic_rl/tools.py`, `agentic_rl/markup.py` | `ToolSpec`, the Qwen-XML and Hermes-JSON tool-call parsers (malformed markup is a model-attributable `parse_error`, never an exception) and `validate_call`. Every chat special token and tag is spelled once, in `markup.py`. |
| Native multi-turn loop | `agentic_rl/harness/` | `NativeToolLoop`: generate → parse → execute → observe until `submit`, a final answer, the step limit, the response budget, an engine abort or an infrastructure failure. Token ids come from the engine and are never re-rendered; observation tokens are prefix deltas of the chat template, refused if the template is not prefix-stable. |
| Inference engines | `agentic_rl/engines/` | Token-in/token-out `SGLangClient` (native `/generate`) and `VLLMClient` (OpenAI-compatible `/v1/completions`, the engine the GB200 estate actually ships), each with logprobs and a declared weight-reload/control surface. `EngineServerSpec`/`EngineServer` are the GENERIC pair (full declared argv, the right client class per `kind`); `SGLangServerSpec`/`SGLangServer` keep working unmodified. `EngineFleet.clients()` returns one client per server, `client_for()` round-robins. Every HTTP field name is an assumption until probed on the cluster -- see each client module's own "UNPROBED ASSUMPTIONS" docstring section, especially `engines/vllm.py`'s on `/collective_rpc` reload. |
| Weight sync | `agentic_rl/weight_sync.py` | `DiskWeightSync`: every rank writes the HF checkpoint, rank 0 pushes it, under a declared `mode`: `"reload_endpoint"` (default) asks each server's client to reload from disk over its own control endpoint; `"restart"` stops and relaunches each `EngineServer` with the new checkpoint path substituted into its declared `{model_path}` command placeholder (`EngineServer.start(command=...)` never mutates the declared spec, so the same template serves every restart) -- the guaranteed-correct S0 path for an engine (vLLM, today) with no proven reload endpoint. `is_stale=False` only because the S0 schedule is synchronous either way. Old publish directories are pruned strictly inside `publish_root`. |
| Rewards | `agentic_rl/rewards/` | `RewardFn` / `RewardVerdict` (a verdict with no value must carry a reason). The rule-scored music reward is a typed port of an Apache-2.0 scorer: model-attributable failures score a measured 0.0, a missing or failing `abc2midi` abstains. |
| Rollout host | `agentic_rl/rollout_host.py`, `agentic_rl/tasks.py` | Draws tasks, runs `group_size` concurrent sessions per task (each in its own environment), scores them and returns the `flatten()`ed batch; infrastructure failures become INFRA trajectories, programming errors propagate. |
| Trainer rows path | `rl/trainer.py` | `RLTrainConfig.rollout_source`: each step consumes the agentic batch instead of the built-in single-turn generate-and-score leg. The per-token mask comes from `loss_mask` (tool tokens stay 0), padding is by length, `None` rewards are dropped, old logprobs are still recomputed, and `publish()` runs after every optimizer step. Refused for PPO, online preference and estimator-free algorithms. |

## Gates

Five gates in `src/foundationscale/gates/agentic_gates.py` (follow
`docs/CUSTOM_GATES.md`'s contract -- each ships `MUST_FIRE`/`MUST_PASS`
controls, runs under `foundationscale-controls`) run at two new
`Lifecycle` points this slice adds, append-only: `ROLLOUT` (after a step's
batch is flattened) and `WEIGHT_SYNC` (after a weight-sync push). Wired
through `RolloutHost` (`gates: bool = True`; `declared_max_infra_rate` /
`declared_max_lag` fields, set from `rollout.max_infra_rate` /
`engine.max_policy_lag`), which raises `RolloutHostRefusal` on a blocking
report.

| Gate | Event | What it catches |
|---|---|---|
| `TrajectoryIntegrityGate` | `ROLLOUT` | Per-row structural integrity of the flattened batch: a loss mask entry outside `{0, 1}`, a per-token column that runs its own length, supervision on a non-assistant/tool-call token, an INFRA row with a non-zero mask, or a reward/abstention pair that contradicts itself. |
| `RolloutAbstentionGate` | `ROLLOUT` | The batch's infra-abstention rate against a declared bound; an undeclared bound abstains (`NOT_ESTABLISHED`) rather than inventing one, and a 100% infra rate is measured like any other value, never its own vacuity. |
| `StalenessGate` | `WEIGHT_SYNC` | A pushed weight sync's measured `is_stale` claim, and every supplied rollout session's policy version against the declared lag; `offered == 0` abstains `NOT_APPLICABLE`. |
| `WeightSyncParityGate` | `WEIGHT_SYNC` | Engine/learner logprob agreement within a declared tolerance, when a parity probe supplies pairs; `parity=None` abstains `NOT_ESTABLISHED` until a GPU probe slice measures it. |
| `PromptBytesGate` | `ROLLOUT` | The rendered system prompt and tool schemas hash (`prompt_bytes_sha256`) to the value declared at launch; an unrecorded declaration abstains `NOT_ESTABLISHED`. |

## Rules every later piece must keep

- An unmeasured reward is `None`, never `0.0`; infra failures carry
  `abstention_reason="infra:<kind>"` and leave the baseline.
- Rollout logprobs exist only for a declared off-policy correction and are
  never reused as `old_logprobs` (the trainer recomputes them).
- Token ids are never round-tripped through the tokenizer.
- Behaviour switches are declared configuration with provenance, never
  environment variables.

## Running

One declared, provenance-tracked JSON config (`agentic_rl/config.py`'s
`AgenticRLConfig`) drives the whole run; see `examples/agentic_rl/music_s0.json`
(the music arm on one GPU tray) for every section, commented by key via `_doc`
strings (JSON has no comments). Validate it -- no GPU, no server, no torch or
transformers import -- before touching hardware:

```sh
foundationscale-agentic-rl --config examples/agentic_rl/music_s0.json --dry-run
# or, without installing the console script:
python -m foundationscale.agentic_rl --config examples/agentic_rl/music_s0.json --dry-run
```

This builds and validates every object that needs neither (the tool registry
and parser, `SamplingParams`, `EpisodeBudget`, `EnvSpec`, the engine server
spec and an unstarted `EngineFleet`, the reward scorer, the task pool) and
prints the fully resolved config with per-key provenance (`"config"` /
`"cli"` / `"default"`) as JSON, exiting 0. Override any declared key with
repeatable `--set dotted.key=value` (e.g. `--set trainer.max_steps=5`); a
real run (omit `--dry-run`) additionally loads the tokenizer, builds the
`ChatTokenizer` adapter over `apply_chat_template(..., tokenize=True)`, starts
the engine fleet, and calls `RLTrainer(RLTrainConfig(..., rollout_source=host)).run()`.
Exit codes are the same four `train/cli.py` uses: `0` pass, `5` red (a crash),
`95` unmeasured (`run()` reported 0 steps), `96` refuse (a declared config
problem, or a `TrainerRefusal`).

## Known gaps for the next slice

- Colocated weight sync IS wired into the CLI's real run: `cli.py` always
  constructs `RolloutHost` with a `DiskWeightSync` (`mode` from
  `engine.weight_sync_mode`, `publish_root` from the config's own
  `publish_root`). `RolloutHost.publish(model, tokenizer, ctx, step)` builds
  its save AROUND THE REAL `model`/`tokenizer`/`ctx` it receives from
  `RLTrainer.run()`, via `rl.distributed.save_checkpoint(..., sharding=
  trainer.sharding)` -- never through `weight_sync.save_fn`, which now exists
  only for `DiskWeightSync.sync()`'s own non-distributed convenience path (and
  may be `None`). A push that does not fully succeed RAISES (never silently
  continues): serving the stale, pre-publish policy on every later rollout
  without saying so is exactly the failure this exists to prevent. The CLI
  separately REFUSES (96, named) a config declaring `trainer.max_steps > 1`
  with no weight sync wired, unless `engine.allow_stale_policy: true` is
  explicitly declared (recorded in provenance like every other key) --
  belt-and-suspenders against any future path that could leave
  `RolloutHost.weight_sync` `None` for a multi-step run.
- `engines/vllm.py`'s `/collective_rpc` `reload_weights` assumption is
  UNPROBED and gated behind an explicit `reload_mode="collective_rpc"` opt-in;
  a cluster probe decides whether it is usable at all. Until then,
  `engine.weight_sync_mode="restart"` is the guaranteed-correct S0 path for
  vLLM.
- `RolloutHost.publish` does not yet track which policy version each
  in-flight rollout session is running, so `StalenessGate`'s `WEIGHT_SYNC`
  check runs with `rollout_policy_versions=()` -- an honest empty sequence,
  never a fabricated one; the gate still reaches a real PASS from the push's
  own measured `is_stale`, scoped to what it actually examined. Likewise
  `parity`/`declared_parity_atol` are always `None` from this call site until
  a GPU probe slice exists to measure engine/learner logprob agreement.
- `trainer.{temperature,top_p,top_k,max_new_tokens,group_size}` remain
  vestigial under `rollout_source` except for one early greedy-decoding guard
  (`rl/trainer.py`); the sampling that actually reaches the engine is the
  top-level `sampling` config section instead.
