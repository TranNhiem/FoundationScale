"""Held-out WER for a NeMo speechlm2 SALM model (Canary-Qwen), scored by FoundationScale.

NeMo owns loading and generation (SALM's own chat API with its audio locator tag); FoundationScale
measures with the same normaliser and corpus WER as every other model.
Usage: eval_wer_salm.py --model nvidia/canary-qwen-2.5b --manifest JSONL --out JSON
"""

# code moved to src/foundationscale/upstream/nemo/salm_decode.py (PHASE 1.2b of
# docs/research/upstream_integration.md). This file is kept as a thin wrapper (same USAGE text) so
# callers see no behavioural difference.

from __future__ import annotations

from foundationscale.upstream.nemo.salm_decode import main

if __name__ == "__main__":
    raise SystemExit(main())
