"""NeMo speechlm2 SALM fine-tune (FoundationScale feeds the data and adjudicates the result).

Same contract as the Canary NeMo-lane scripts, one lane down (SALM = Canary encoder, trained
in full, + a frozen Qwen LLM with LoRA):
1. Convert a FoundationScale audio JSONL (audio, answer, duration) to a NeMo manifest via
   nemo_finetune.convert under the same strict rules: 16 kHz mono only, 1-25 s, non-empty
   text; every refused row is COUNTED and recorded (never silently dropped).
2. Fine-tune the released SALM through NeMo's own speechlm2 API (SALMDataset + DataModule +
   the model's own configure_optimizers) and save via HFHubMixin. NeMo owns the training loop;
   FoundationScale does not wrap it. The released freeze scope and LoRA stay exactly as shipped
   -- nothing about them is re-specified here.

Usage: salm_finetune.py --model nvidia/canary-qwen-2.5b --train JSONL --out-dir DIR
                        [--max-steps 500 --batch-size 8 --lr 1e-4 --warmup 50]
"""

# code moved to src/foundationscale/upstream/nemo/salm_finetune.py (PHASE 1.2b of
# docs/research/upstream_integration.md). This file is kept as a thin wrapper (same USAGE text) so
# callers see no behavioural difference.

from __future__ import annotations

from foundationscale.upstream.nemo.salm_finetune import main

if __name__ == "__main__":
    raise SystemExit(main())
