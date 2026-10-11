"""FoundationScale verdicts on a SALM checkpoint it did not train (the NeMo speech-lm lane).

Runs the speech gates plus the SALM freeze contract:
  * speech.audio_row_coverage over the converted manifest (coverage.json from salm_finetune.py);
  * speech.tower_movement (exercised=True) over perception.encoder and perception.proj,
    the trainable pieces of the perception stack;
  * speech.tower_movement over the PEFT tensors alone (its line is labelled "lora"): the only
    trainable LLM weights per freeze_params / prevent_freeze_params;
  * salm.frozen_llm_unchanged: every non-LoRA weight under llm. and embed_tokens. is bit-identical
    base vs fine-tuned -- a silent thaw would look like an ordinary fine-tune and no movement
    verdict would ever say so;
  * salm.param_names_unchanged: a name present in one model and not the other blocks; renaming
    would hide movement from every digest comparison above.
Exit code follows the run contract: 0 PASS, 5 RED (a gate blocks), 95 UNMEASURED.

Usage: salm_adjudicate.py --base nvidia/canary-qwen-2.5b --finetuned DIR
                          --coverage DIR/coverage.json --out DIR/adjudication.json
"""

# code moved to src/foundationscale/upstream/nemo/salm_adjudicate.py (PHASE 1.2b of
# docs/research/upstream_integration.md). This file is kept as a thin wrapper (same USAGE text) so
# callers see no behavioural difference.

from __future__ import annotations

from foundationscale.upstream.nemo.salm_adjudicate import main

if __name__ == "__main__":
    raise SystemExit(main())
