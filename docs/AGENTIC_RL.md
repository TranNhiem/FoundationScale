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

## Rules every later piece must keep

- An unmeasured reward is `None`, never `0.0`; infra failures carry
  `abstention_reason="infra:<kind>"` and leave the baseline.
- Rollout logprobs exist only for a declared off-policy correction and are
  never reused as `old_logprobs` (the trainer recomputes them).
- Token ids are never round-tripped through the tokenizer.
- Behaviour switches are declared configuration with provenance, never
  environment variables.

## Next

Environment backends and the native multi-turn tool loop, the rollout host and
engine fleet with weight sync, reward services, the agentic gates, and the
`foundationscale-agentic-rl` entry point, ending in a small-scale
reproduction on a rule-scored task.
