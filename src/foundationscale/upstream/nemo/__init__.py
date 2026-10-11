"""The NeMo worker adapter.

This package is the worker-side adapter for the NeMo (Canary) AED audio lane. It RUNS INSIDE THE
NEMO CONTAINER -- this is where a checkpoint is loaded, a wave file is opened and a Lightning
trainer is set going. It is the ONLY FoundationScale code allowed to import ``nemo`` (or, more
broadly, ``torch`` / ``lightning`` / ``soundfile`` at any point). Everything above the upstream
boundary speaks the typed data in :mod:`foundationscale.upstream.contracts` and never touches a
checkpoint, a wave file, or a container call.

It EXCHANGES ONLY OWNED CONTRACT FILES. A FoundationScale speech manifest goes in (JSONL,
fields ``id``/``audio``/``answer``/``duration``); a NeMo AED manifest and its coverage sidecar
come out, both shapes fixed by the checkpoints' inherited ``train_ds``; an adjudication report
returns to the control plane. Nothing else crosses: a plain ``dict``, an ``OmegaConf`` node or
any NeMo type leaking past this package boundary is a bug.

Four steps, each a ``python -m foundationscale.upstream.nemo.<step>`` entry point. Each exposes
``main(argv=None) -> int`` for the thin campaign wrappers in
``validation_campaigns/speech_canary/`` to call (and for the container to run directly).

Import cost: every module here must import WITHOUT ``nemo``, ``torch``, ``lightning`` or
``soundfile`` installed. The "heavy" imports (whatever the container has and plain CI does not)
live INSIDE the functions that need them. Do not move one to module level to tidy the file --
that is what keeps these modules testable in CI with fakes.

PHASE 1.2a of ``docs/research/upstream_integration.md``: this package is the move target for
``validation_campaigns/speech_canary/nemo_finetune.py``,
``validation_campaigns/speech_canary/eval_wer_nemo.py`` and
``validation_campaigns/speech_canary/nemo_adjudicate.py``. Behaviour is preserved verbatim --
same NeMo calls, same config overrides, same printed COVERAGE/TRAIN_DS/STEP/SNAPSHOT/SAVED
lines, same eval JSON keys, same adjudication lines and exit codes. The refactor is a MOVE onto
the Phase 1.1 contracts, not a rewrite.
"""
