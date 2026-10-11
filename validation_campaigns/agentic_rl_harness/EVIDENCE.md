# Black-box agent loop — real-node evidence (plan v2, P0.5)

**Design (user decision 2026-10-11).** The harness process runs on the host, so its LLM calls reach the
FS gateway directly; only its command execution enters a no-network `EnrootSandbox` through
`foundationscale.agentic_rl.sandbox.mini_swe_env.EnrootMiniSweEnvironment`, which mini-SWE-agent loads
by dotted path as its `environment_class` (no change to mini-SWE-agent).

**What was run.** `agent_e2e.py` on one GB200 tray: vLLM 0.21 serving a 9B Qwen3.5-family model on one
GPU; the FS gateway in front of it (`GatewayCore`, chat-template render, Qwen-XML tool parser); a gateway
session; an unmodified mini-SWE-agent (`DefaultAgent` + `LitellmModel` pointed at the session's
`/v1` base URL) with its stock `default.yaml` prompts; the toy task in `task/` (`calc.add` returns
`a - b`, `test_calc.py` expects 5) rebuilt for arm64 by `build_image`; the verifier runs the tests in
the sandbox and posts the reward to the gateway, which assembles the `fs.trajectory/v2` episode.

**Result (2026-10-11).**

| | |
|---|---|
| Agent exit | `Submitted` after 10.9 s |
| Model calls through the gateway | 5 |
| Fix | `return a - b` → `return a + b` |
| Tests after the agent | `1 passed` → reward **1.0** |
| Episode | 1 segment, 5 sampled spans, 540 trainable tokens, reward event 1.0 |

All five turns merged into one segment: the gateway's message-level continuation reused the exact
sampled tokens across every turn of a real harness.

**Defect this run found and fixed:** mini-SWE-agent's prompt templates read `uname` fields (`release`,
`version`, …); the environment now reports the sandbox's own `uname` values (read once inside the
container), never host values.
