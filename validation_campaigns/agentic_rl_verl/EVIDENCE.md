# verl adapter against real verl (2026-10-12)

verl was installed at the pinned commit 9058412fb308e9f2e31a794ad1d18d97d4286485 (`pip install
git+https://github.com/volcengine/verl@<pin>`) in a private copy of the GB200 vLLM container (aarch64).
The copy keeps the container's own torch 2.11, vLLM 0.21 and transformers 5.9; the pin's other
dependencies were installed under a constraints file that freezes those packages.

| check | expected | measured |
|---|---|---|
| `require_pinned_verl()` on the real install | returns the pin | `9058412fb308e9f2e31a794ad1d18d97d4286485` |
| wrong pin | MUST_FIRE | fired, naming both SHAs |
| absent distribution | MUST_FIRE | fired |
| real `AgentLoopOutput` (model, tool, model turns) -> `episode_from_agent_loop_output` | valid v2 Episode | spans g0 and g1, NaN logprobs and version -1 on the 3 tool tokens, version 7 on model tokens, one `verl_reward_score` event of 1.0, flattens to training rows |

Scripts: `verl_pincheck.py` checks the pin guard against the real install; `verl_xcheck.py` checks the
pin guard and the adapter on a real `AgentLoopOutput`. 
## Real AgentLoop run (P0.3 MUST_PASS a, 8-sample scale)

`agentloop_run.py` runs verl's own `AgentLoopManager` (async vLLM rollout, tool agent with an `add` function
tool, max 3 assistant turns) on the 9B policy on one GB200 GPU. It unpacks the padded batch with
`samples_from_rollout_batch` and adapts every sample with `episode_from_agent_loop_output`. The summary is
in `agentloop_run.json`.

| measured | value |
|---|---|
| samples / valid v2 episodes / refusals | 8 / 8 / 0 |
| samples that called the tool | 5 of 8 (4 turns, 10 tool tokens each) |
| token-level check on the real batch | every tool sample splits into two model spans around its tool tokens, e.g. (0, 27) and (37, 50); all tool tokens have NaN logprob and policy_version -1; all model tokens have finite logprobs and version 0 |

Things the real runs exposed, and the fixes:
- `multi_turn.format` defaults to `hermes`, but MiMo-V2.6 emits Qwen3-Coder XML tool calls
  (`<function=add><parameter=a>17</parameter>...`). With `hermes` no call parsed, so no tool ran;
  `qwen3_coder` fixed it.
- No FlashAttention wheel exists for this aarch64 container, so the run uses `override_config.attn_implementation=sdpa`
  and `use_remove_padding=false`.
- A first draft of the batch unpacker inverted both padding checks (unit tests caught it), and the adapter
  treated verl's optional fields as required (the end-to-end test caught it).
- Rewards are all 0.0: the borrowed `openai/gsm8k` rule scorer expects a `#### N` answer line. Reward
  correctness belongs to the parity step, not to this format check.

Parity against FS GRPO on the music task (MUST_PASS b) is measured below.

## Parity: verl GRPO vs FS GRPO on music, 100 steps (2026-10-16)

`verl_parity_launch.sh 7` ran verl's own GRPO trainer with the settings of the FS run `train100_tp1_s7`
(MiMo-V2.6-Distill 9B, top_p 1.0, T 1.0, lr 1e-6, 4 tasks x 8 samples, centred advantage, clip 0.2, no KL),
on one GPU with parameter and optimizer offload. It completed all 100 steps (exit 0).
`parity_stats.py` compares mean rollout reward per window against two FS seeds
(`parity_tp1_s7.json`, `parity_tp1_s6.json`):

| window | FS seed 6 | FS seed 7 | verl | criterion | verdict |
|---|---|---|---|---|---|
| steps 0-19 | 0.070 | 0.133 | 0.085 | inside the FS seed envelope | PASS |
| steps 20-49 | 0.344 | 0.421 | 0.450 | inside the FS seed envelope | 0.029 above the envelope |
| steps 50-99 | 0.559 | 0.563 | 0.580 | within 5% of FS | PASS (3.1% / 3.8%) |

Reading: verl and FS reach the same plateau (late window within 5% of both seeds), and verl's early
learning is inside the FS seed spread. In steps 20-49 verl learns slightly faster than either FS seed.
Two FS seeds make a narrow envelope, so this is a near-miss rather than a measured divergence. The plan's
criterion for the middle window is therefore not met on this evidence; a third FS seed (or a second verl seed)
would settle it. One known difference remains: verl's loss reduction (`seq-mean-token-mean`) is the closest
verl option to FS's `prompt_mean`, not an identical definition.

## FS music reward through verl's reward hook (2026-10-12)

`verl_reward_check.py` builds the hook with verl's own `get_custom_reward_fn` pointed at
`foundationscale.agentic_rl.verl_rewards.music_compute_score` and scores a valid 18-bar ABC tune with
the real `abc2midi` in the verl container. The direct call, the hook called by position and the hook
called by keyword all return 0.468. A malformed tune (duplicated `X:` header) scores 0.0 as rejected,
and text with no ABC scores 0.0 (`no_abc`).
