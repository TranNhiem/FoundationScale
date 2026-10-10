# Upstream integration: keeping FoundationScale current with NeMo, Hugging Face and friends

Status: approved 2026-10-10. Executed phase by phase; each phase is behaviour-preserving and verified
against recorded results before the next one starts.

## Principles (the user's design rules, as adapted to the current state)

1. **Own the contracts, not the internals.** FoundationScale owns the data manifest schema, the run
   config schema, the checkpoint metadata, and the evaluation harness (WER/CER, bootstrap, runaway
   and loop metrics, fixed test sets). Adapters use upstream public APIs only.
2. **Wrap, don't copy.** Every unavoidable copy, workaround or private-API use is recorded in the
   ledger (`foundationscale.upstream.ledger`), with why, where, from which upstream version, and an
   expiry check that tells us when upstream no longer needs it.
3. **Isolate dependencies.** Each backend that needs its own stack runs in its own pinned container as
   a worker and exchanges only owned contract files (manifest, run config, checkpoint, eval JSON).
   *Adjustment:* the Hugging Face lane stays in-process, because FoundationScale's own trainer *is*
   the HF backend; wrapping it as a worker would add a layer with no second user (rule 5). The
   worker boundary applies to NeMo first.
4. **Integrated means reproduced.** A model or technique is `supported` only after FoundationScale
   reproduces the upstream reference result on the same test set the upstream reports, scored by our
   harness. The gap and any normalizer difference are reported, and the tolerance is fixed before
   the run. That run then becomes a regression test.
5. **Don't over-abstract early.** A registry or plugin layer is added only where two real backends
   need it. Today that is the model registry (HF and NeMo both host speech models); a technique
   plugin layer waits for a second real technique.
6. **Licenses for code and weights** are recorded per registry entry and per released checkpoint.
7. **Supply chain (added).** Pin upstream by version and digest, review CI and workflow changes like
   code, and never run unreviewed upstream scripts with access to credentials.

*Format adjustment:* the registry and ledger are typed Python data (frozen dataclasses, as in
`families/registry.py`), not YAML. The package supports Python 3.10, where `tomllib` is absent,
and PyYAML is deliberately not a dependency.

## Current state (2026-10-10 review)

- HF lane (`src/foundationscale/train/`): imports transformers/peft directly. Hard-coded model-type
  to kind table and class-name rules (`speech_kinds.py`). The Qwen2-Audio token formula is copied
  (`audio.py`). Private `_compute_audio_num_tokens` and Whisper tokenizer prefix internals are used.
- NeMo lane: campaign scripts only (`validation_campaigns/speech_canary`, `speech_longform`). It
  imports NeMo public classes but carries four workarounds against NeMo internals, keeps a
  per-model adjudicator with hard-coded prefixes, and runs NeMo example scripts by file path.
- Versions: `pyproject.toml` floors only; container tags inside scripts; `peft`/`accelerate`
  unpinned in the NeMo environment; NeMo-lane results carry no version record.
- Licenses (declared on Hugging Face): code Apache-2.0. Weights: Canary/Parakeet CC-BY-4.0;
  Whisper-v3, Qwen2-Audio and Gemma-4-E4B-it Apache-2.0. Data: LibriSpeech and AMI CC-BY-4.0;
  the Earnings-22 derivatives declare none. The original Earnings-22 is believed CC-BY-SA-4.0 and
  must be verified before any model trained on it is released.

## Phases

1. Owned contracts (manifest, run config), the ledger with expiry checks, and the NeMo lane moved
   into `src/foundationscale/upstream/nemo/` as the worker-side adapter. Verified by re-running
   recorded NeMo results through the new code and getting identical numbers.
2. Model registry; scope-driven adjudication replaces the per-model adjudicators.
3. Upstream profiles (container digest + version lock per backend); Level 1 checks in CI, Levels 2-3
   (GPU smoke, reproduction, regression) as tray runs; add the model-card test sets.
4. First checkpoint converter: Parakeet-CTC, NeMo <-> HF.
5. Release tracker; extend the pattern to VLM.

## Shared infrastructure

Shared now: worker isolation with file contracts, the run manifest with profile and license fields,
the model registry, the ledger, the reference-run regression harness, profiles and the Level 1-3
process. Candidates (extracted when a second user appears): checkpoint converters, a technique
plugin layer. Speech only: audio contract, speech metrics, long-form chunking.

## Phase 1.2a: the NeMo AED (Canary) lane as a worker adapter (2026-10-10)

`src/foundationscale/upstream/nemo/{census,finetune,decode,adjudicate}.py`, runnable as
`python -m foundationscale.upstream.nemo.<module>` inside the NeMo container. The campaign scripts
are thin wrappers. NeMo is imported only inside the functions that call it; the census, config
translation, argument parsing and verdict assembly are pure and tested in CI. Verified on GB200:

- decode, old script vs new package on the same node: identical (0 of 2,504 transcripts differ;
  WER equal to 6 decimals). Recorded numbers from another node differ slightly (batched bf16 decoding
  is node-sensitive), so equivalence is always checked on one node.
- adjudication of an existing run: identical verdict lines.
- fine-tune: identical census. Found and fixed: `--seed` did not make training reproducible (two
  old-script runs with one seed differed at step 1), because lhotse's `shard_seed` defaults to
  true randomness. Seeded runs now pin it, and two seeded runs on different GPUs give identical losses.

## Phase 1.2b: the NeMo SALM (Canary-Qwen) lane (2026-10-10)

`src/foundationscale/upstream/nemo/{salm_finetune,salm_decode,salm_adjudicate}.py`, same pattern as
1.2a; the three SALM ledger entries follow the code. Verified on one GB200 node against the old
scripts: decode identical (0 of 300 transcripts differ), re-adjudication of the Canary-Qwen
fine-tune identical (6 of 6 lines), census identical. New `--seed` pins lhotse's seed and
`shard_seed` (default behaviour unchanged): two seeded runs give identical losses.

## Phase 2: the model registry (2026-10-10)

`src/foundationscale/upstream/models.py`: one typed entry per speech model (backend, upstream
reference, kind, license, scopes, requirements, reference result, status). HF entries link to their
existing `families/registry.py` FamilySpec by name, so nothing is duplicated; NeMo entries carry the
trainable/frozen scopes that adjudication reads. `registry_problems()` checks the families exist, the
scopes are consistent, and that no entry is `supported` without a measured reproduction inside
tolerance. All 6 entries are `experimental`: none has been reproduced on its model card's own test
set yet (we hold LibriSpeech dev-clean; the cards report test-clean).

Both NeMo adjudicators take `--model-id` and read their scopes from the registry (defaults unchanged).
Verified on GB200: registry-driven re-adjudication of the Canary (arm B, frozen decoder) and
Canary-Qwen fine-tunes reproduces the recorded verdict lines byte for byte.
