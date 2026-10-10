# Agentic RL gateway — real-engine evidence (plan v2, P0.2e)

**What was run.** `gateway_probe.py` started the FS gateway (`foundationscale.agentic_rl.gateway`) in
front of a real vLLM 0.21 server serving a 9B Qwen3.5-family model on one GB200 GPU, and drove a
two-turn tool-use conversation through each client wire format — OpenAI Chat Completions, OpenAI
Responses, Anthropic Messages and Gemini — plus one streaming chat call, using only HTTP against the
gateway. Turn 1 asks for the weather with one declared tool; turn 2 returns the tool result in that
format's native shape, preserving the assistant turn as the client received it. `fidelity_decode.py`
then decoded every sampled span and every context span of the saved `fs.trajectory/v2` episodes.

**Result (2026-10-11).**

| Format | HTTP | Tool call in turn 1 | Segments | Sampled spans | Trainable tokens | Context tokens | Trained logprobs finite |
|---|---|---|---|---|---|---|---|
| Chat Completions | all 2xx | yes | 1 | 2 | 57 | 12 | yes |
| Responses | all 2xx | yes | 1 | 2 | 57 | 12 | yes |
| Anthropic Messages | all 2xx | yes | 1 | 2 | 74 | 12 | yes |
| Gemini | all 2xx | yes | 1 | 2 | 102 | 16 | yes |

Streaming chat: 5 SSE frames, all parsed.

**Token fidelity.** Every sampled span decodes to the model's own output verbatim — its `<think>`
reasoning, the Qwen-XML tool call (`<tool_call><function=get_weather><parameter=city>Paris…`) and the
closing end-of-turn token. Every context span is only the tool result plus the next assistant header
(`<|im_start|>tool\n18C, sunny<|im_end|><|im_start|>assistant\n`), loss-masked. Nothing on the trained
side was re-rendered or re-tokenized.

**Defects this run found and fixed before the result above** (each now has a unit test):

1. Session base URLs advertised a port-0 placeholder; `serve()` now substitutes the bound port.
2. Re-rendering the model's own turn through the chat template can never reproduce it (the template
   inserts an empty `<think></think>` and renders tool arguments as JSON while the model samples
   Qwen-XML), so every tool-call turn fragmented into a new segment. The gateway now keeps a
   per-session continuation — the served messages and the exact sampled tokens — and renders only new
   messages as a template delta, the same rule as `NativeToolLoop`. A rewritten history still falls
   back to a full render and opens a new segment.
3. The decoded end-of-turn token (`<|im_end|>`) leaked into client-visible content, which made
   Responses/Gemini history echoes mismatch. It is now stripped from text only; the sampled ids keep it.

**Known and deliberate.** No newline separates a sampled `<|im_end|>` from the next `<|im_start|>`:
this matches `NativeToolLoop` (the measured v1 path), so both rollout modes see identical tokens.
