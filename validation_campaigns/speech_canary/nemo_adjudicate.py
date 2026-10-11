"""FoundationScale verdicts on a NeMo-lane checkpoint it did not train (the Canary lane).

Runs the same speech gates as train() does after a HF-native save:
  * speech.audio_row_coverage over the converted manifest (coverage.json from nemo_finetune.py);
  * speech.tower_movement over the encoder, base vs fine-tuned, parameters only (digests of
    named_parameters, dtype-tagged), plus the decoder as a second exercised module.
Exit code follows the run contract: 0 PASS, 5 RED (a gate blocks), 95 UNMEASURED.

Usage: nemo_adjudicate.py --base nvidia/canary-1b-flash --finetuned DIR/finetuned.nemo
                          --coverage DIR/coverage.json --out DIR/adjudication.json
                          [--frozen transf_decoder]  (declared frozen: must be bit-identical)
"""

# The worker-side adapter moved to src/foundationscale/upstream/nemo/adjudicate.py (PHASE 1.2a of
# docs/research/upstream_integration.md). This file is kept as a thin wrapper (same USAGE text) so
# the adjudicator's invocation keeps working unchanged.

from foundationscale.upstream.nemo.adjudicate import main

if __name__ == "__main__":
    raise SystemExit(main())

