"""Held-out WER for a NeMo ASR model (Canary), scored by foundationscale.train.speech_metrics.

The FoundationScale side of the NeMo lane: NeMo decodes (it owns the model), FoundationScale
measures with the same normaliser and corpus WER as every HF-native model, so the numbers compare.
Usage: eval_wer_nemo.py --model <hf id or .nemo> --manifest JSONL --out JSON [--batch-size 16]
"""

# The worker-side adapter moved to src/foundationscale/upstream/nemo/decode.py (PHASE 1.2a of
# docs/research/upstream_integration.md). This file is kept as a thin wrapper (same USAGE text) so
# the worker's invocation keeps working unchanged.

from foundationscale.upstream.nemo.decode import main

if __name__ == "__main__":
    raise SystemExit(main())

