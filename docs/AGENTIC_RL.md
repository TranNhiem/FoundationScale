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
| Inference engines | `agentic_rl/engines/` | Token-in/token-out `SGLangClient` with logprobs and weight reload from disk, and an `EngineFleet` that owns server processes. The HTTP field names follow SGLang's native API and are an assumption until probed on the cluster; the GB200 estate currently ships vLLM, not SGLang. |
| Weight sync | `agentic_rl/weight_sync.py` | `DiskWeightSync`: every rank writes the HF checkpoint, rank 0 tells the fleet to reload it; `is_stale=False` only because the S0 schedule is synchronous. Old publish directories are pruned strictly inside `publish_root`. |
| Rewards | `agentic_rl/rewards/` | `RewardFn` / `RewardVerdict` (a verdict with no value must carry a reason). The rule-scored music reward is a typed port of an Apache-2.0 scorer: model-attributable failures score a measured 0.0, a missing or failing `abc2midi` abstains. |
| Rollout host | `agentic_rl/rollout_host.py`, `agentic_rl/tasks.py` | Draws tasks, runs `group_size` concurrent sessions per task (each in its own environment), scores them and returns the `flatten()`ed batch; infrastructure failures become INFRA trajectories, programming errors propagate. |
| Trainer rows path | `rl/trainer.py` | `RLTrainConfig.rollout_source`: each step consumes the agentic batch instead of the built-in single-turn generate-and-score leg. The per-token mask comes from `loss_mask` (tool tokens stay 0), padding is by length, `None` rewards are dropped, old logprobs are still recomputed, and `publish()` runs after every optimizer step. Refused for PPO, online preference and estimator-free algorithms. |

## Rules every later piece must keep

- An unmeasured reward is `None`, never `0.0`; infra failures carry
  `abstention_reason="infra:<kind>"` and leave the baseline.
- Rollout logprobs exist only for a declared off-policy correction and are
  never reused as `old_logprobs` (the trainer recomputes them).
- Token ids are never round-tripped through the tokenizer.
- Behaviour switches are declared configuration with provenance, never
  environment variables.

## Next

A vLLM client (the engine the GB200 estate has) and a cluster probe of the
engine API, the agentic gates, the `foundationscale-agentic-rl` entry point and
configuration, and a small-scale reproduction on a rule-scored task.
