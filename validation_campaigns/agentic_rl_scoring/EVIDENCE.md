# Sampled-distribution scoring cost on real hardware (2026-10-12)

`bench_score.py` loads the 9B agentic policy (bf16, gradient checkpointing, train mode) on one GB200
GPU and times `rl.trainer._token_logprobs` on random token sequences: raw logits (the historical score)
against scoring under the engine's sampling transform. Outputs are in `bench_score_4096.json` and
`bench_score_8192.json`.

| tokens | variant | score s | score + backward s | peak GB | all finite |
|---|---|---|---|---|---|
| 4096 | raw logits | 0.021 | 5.68 | 36.1 | yes |
| 4096 | top_p 0.95 | 0.150 | 5.59 | 36.1 | yes |
| 4096 | T 0.8, top_p 0.95, top_k 20 | 0.143 | 5.61 | 36.1 | yes |
| 8192 | raw logits | 0.032 | 11.07 | 53.1 | yes |
| 8192 | top_p 0.95 | 0.292 | 11.67 | 47.0 | yes |
| 8192 | T 0.8, top_p 0.95, top_k 20 | 0.286 | 11.50 | 47.0 | yes |

The sampled path adds about 0.3 s to an 11 s forward and backward at 8k tokens. Its peak memory is
lower, because the chunked path never upcasts the whole logits tensor at once. A first attempt OOM'd
at 179 GB only because the harness left the model in eval mode, so gradient checkpointing never engaged.
