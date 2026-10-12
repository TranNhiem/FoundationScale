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
pin guard and the adapter on a real `AgentLoopOutput`. Not yet measured: a full AgentLoop run with a vLLM
server (P0.3 MUST_PASS a) and parity against FS GRPO (MUST_PASS b).
