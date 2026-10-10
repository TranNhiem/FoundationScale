"""NeMo-lane fine-tune for Canary (FoundationScale feeds the data and adjudicates the result).

1. Convert a FoundationScale audio JSONL (audio, answer, duration) to a NeMo AED manifest under
   the same strict rules as the HF loader: 16 kHz mono only, 1-25 s, non-empty text; every refused
   row is COUNTED and recorded (never silently dropped).
2. Fine-tune through NeMo's own API (setup_training_data / setup_optimization / Lightning Trainer)
   and save a .nemo. NeMo owns the training loop here; FoundationScale does not wrap it.

Usage: nemo_finetune.py --model nvidia/canary-1b-flash --train JSONL --out-dir DIR
                        [--max-steps 500 --batch-size 8 --lr 1e-5 --warmup 50 --pnc no]
                        [--save-every N]  (also save DIR/step{N}.nemo, for checkpoint selection)
                        [--freeze PREFIX ...]  (e.g. transf_decoder; adjudicate with --frozen)
"""

# The worker-side adapter moved to src/foundationscale/upstream/nemo/finetune.py (PHASE 1.2a of
# docs/research/upstream_integration.md). This file is kept as a thin wrapper (same USAGE text, same
# module-level ``convert`` that salm_finetune.py imports) so callers see no behavioural difference.

from __future__ import annotations

from pathlib import Path

from foundationscale.upstream.nemo.finetune import convert as _convert
from foundationscale.upstream.nemo.finetune import main


def convert(
    train_jsonl: Path, out_manifest: Path, pnc: str, max_duration: float = 25.0
) -> dict[str, object]:
    """Convert FS -> NeMo AED manifest; delegates to census + to_nemo_aed_row via finetune.convert.

    Kept at module level because ``salm_finetune.py`` imports
    ``from nemo_finetune import convert``. Returns the same coverage dict the original produced:
    ``rows_expected`` / ``rows_checked`` / ``rows_refused`` / ``refused`` (reason -> count).
    """
    return _convert(train_jsonl, out_manifest, pnc, max_duration)


if __name__ == "__main__":
    raise SystemExit(main())

