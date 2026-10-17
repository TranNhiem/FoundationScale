# Agentic music collapse: mechanism probe (2026-10-13)

> **Correction (2026-10-18):** these runs used `harness.parser: "hermes_json"`, but this model writes
> Qwen-XML tool calls, so every tool call failed to parse and episodes ran without working tools. The
> reward reads the submitted text, so the comparisons below hold for tool-less episodes. See
> `examples/agentic-rl-GB200/README.md` for the measurement.

Runs: MiMo-V2.6-Distill 9B, music task, `agentic_grpo`, lr 1e-6, 4 tasks x 8, top_p 0.95.
g_s6 used raw-logit scoring (collapsed to ~0 reward by step ~55; final checkpoint evals 0.0).
sc_s6 used consistent sampled-distribution scoring (main b60ce53; no collapse; step-89 checkpoint evals 0.653).

`tail_probe.py` samples 4 fixed responses from the base model once, then measures each checkpoint on those
same 3,072 tokens: mean next-token entropy, mean size of the 0.95 nucleus, and mean top-1 probability.
The output is in `tail_probe.json`.

| step | g_s6 nucleus | g_s6 entropy | sc_s6 nucleus | sc_s6 entropy |
|---|---|---|---|---|
| base | 5.34 | 0.755 | 5.34 | 0.755 |
| 10 | 5.51 | 0.767 | 5.34 | 0.754 |
| 20 | 6.11 | 0.807 | 5.33 | 0.749 |
| 30 | 7.38 | 0.878 | 5.36 | 0.748 |
| 40 | 9.38 | 0.967 | 5.37 | 0.749 |
| 50 | 12.17 | 1.056 | 5.40 | 0.752 |
| 60 | 15.43 | 1.135 | 5.42 | 0.753 |
| last (91 / 89) | 22.57 | 1.223 | 5.32 | 0.749 |

Reading: with top_p 0.95 sampling and raw-logit scoring, the tail of the policy inflates monotonically from
the first checkpoints. Negative-advantage updates raise every unsampled token's probability (softmax
gradient e_a - p), and truncated sampling never draws the tail tokens that would correct it. With
consistent scoring the distribution's shape is unchanged across the whole run. The drift is visible by step 20
(+14% nucleus size), about 30 steps before the reward collapse, so it can serve as an early-warning signal.
