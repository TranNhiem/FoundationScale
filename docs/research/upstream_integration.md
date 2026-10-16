# Upstream integration

How FoundationScale tracks NeMo, Hugging Face and other upstream stacks so the framework stays current without accumulating silent copies and private-API use. Status: approved; executed phase by phase, each phase behaviour-preserving and verified against recorded results before the next starts. Reproduction and regression evidence lives in [EVIDENCE.md](validation_campaigns/speech_repro/EVIDENCE.md) and the campaign records under `validation_campaigns/speech_canary`; this document keeps the rules, the current state, and the plan.

## Principles

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
   harness. The gap and any normalizer difference are reported, and the tolerance is fixed before the
   run. That run then becomes a regression test.
5. **Don't over-abstract early.** A registry or plugin layer is added only where two real backends
   need it. Today that is the model registry (HF and NeMo both host speech models); a technique
   plugin layer waits for a second real technique.
6. **Licenses for code and weights** are recorded per registry entry and per released checkpoint.
7. **Supply chain.** Pin upstream by version and digest, review CI and workflow changes like code,
   and never run unreviewed upstream scripts with access to credentials.

*Format adjustment:* the registry and ledger are typed Python data (frozen dataclasses, as in
`families/registry.py`), not YAML. The package supports Python 3.10, where `tomllib` is absent, and
PyYAML is deliberately not a dependency.

## Current state

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
   into `src/foundationscale/upstream/nemo/` as the worker-side adapter. Done: re-running recorded
   NeMo results through the new code gives identical numbers.
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

## Phase 1.2a: the NeMo AED (Canary) lane as a worker adapter

`src/foundationscale/upstream/nemo/{census,finetune,decode,adjudicate}.py`, runnable as
`python -m foundationscale.upstream.nemo.<module>` inside the NeMo container. The campaign scripts
are thin wrappers. NeMo is imported only inside the functions that call it; the census, config
translation, argument parsing and verdict assembly are pure and tested in CI.

Verification on one GB200 node (batched bf16 decoding is node-sensitive, so equivalence is always
checked on one node): decode identical to the old script (0 of 2,504 transcripts differ; WER equal to
6 decimals), re-adjudication identical verdict lines, fine-tune census identical. Rule for seeding:
`--seed` must also pin lhotse's `shard_seed` (it defaults to true randomness, which broke
reproducibility); with it pinned, two seeded runs on different GPUs give identical losses.

## Phase 1.2b: the NeMo SALM (Canary-Qwen) lane

`src/foundationscale/upstream/nemo/{salm_finetune,salm_decode,salm_adjudicate}.py`, same pattern as
1.2a; the three SALM ledger entries follow the code. Verified on one GB200 node against the old
scripts: decode identical (0 of 300 transcripts differ), re-adjudication of the Canary-Qwen
fine-tune identical (6 of 6 lines), census identical. `--seed` pins lhotse's seed and `shard_seed`;
default behaviour unchanged, and two seeded runs give identical losses.

## Phase 2: the model registry

`src/foundationscale/upstream/models.py`: one typed entry per speech model (backend, upstream
reference, kind, license, scopes, requirements, reference result, status). HF entries link to their
existing `families/registry.py` FamilySpec by name, so nothing is duplicated; NeMo entries carry the
trainable/frozen scopes that adjudication reads. `registry_problems()` checks the families exist, the
scopes are consistent, and that no entry is `supported` without a measured reproduction inside
tolerance. All 6 entries are `experimental` until measured on their model card's own test set (we
hold LibriSpeech dev-clean; the cards report test-clean). All NeMo adjudicators take `--model-id` and
read their scopes from the registry (defaults unchanged); registry-driven re-adjudication of the
Canary (arm B, frozen decoder) and Canary-Qwen fine-tunes reproduces the recorded verdict lines byte
for byte.

## Phase 3 (part 1): profiles and the first reproductions

- `src/foundationscale/upstream/profiles.py`: the exact versions of both backend containers
  (`nemo-26.08`: torch 2.13 / transformers 5.12 / peft 0.21 / NeMo 3.1; `hf-26.04`: torch 2.11 /
  transformers 5.5 / peft 0.18), `installed_versions()` and `profile_mismatches()`. The NeMo
  fine-tune workers write `upstream_profile.json` next to every run. Image digests are not visible
  from the job environment and are recorded as "unrecorded".
- Rule 4 in practice (`validation_campaigns/speech_repro`): Canary-Qwen-2.5B, Parakeet-CTC-1.1B and
  Canary-1B-flash reproduce their model-card LibriSpeech WER within 0.02 points, scored with the
  card's normalizer, and are now `supported` in the registry. Our stricter normalizer reads about
  0.2 points higher.

## Phase 3 (part 2): Level 3 as one command

`foundationscale.upstream.levels.level3_plan()` turns the registry into the regression plan: one step
per `supported` model, with its backend, decode entry point, upstream ref, reference split, card value
and tolerance. `validation_campaigns/speech_repro/run_levels.sh SRC OUT` runs each step in its own
backend container on an idle GPU, scores with the card's normalizer and judges with
`levels.judge()`; it exits 0 when every model passes and 5 on any regression. Promoting a model to
`supported` adds it to the run with no other edit. First full run on current profiles: all 3 PASS
with the same numbers as the reproduction (gaps +0.016, +0.004, +0.014).

## Phase 4: deferred (rule 5)

No current workflow moves a checkpoint between NeMo and Hugging Face (Parakeet is trained and
evaluated in the HF lane), and neither container ships an upstream converter (transformers keeps its
conversion scripts out of the installed package). A converter is built when the first real
cross-ecosystem workflow needs one, by wrapping upstream's script plus an FS equivalence check
(identical outputs on real audio).

## Phase 5: the release tracker

```
python -m foundationscale.upstream.tracker --profile nemo-26.08 --since 2026-04-01 [--json] [--markdown]
```

A read-only digest of upstream releases newer than the profile's pins and of new speech models from
nvidia / Qwen / openai / google, each flagged if already in the registry. Versions come from PyPI, not
GitHub tags: NeMo tags container builds (`25.09-alpha.rc2`) that compare as "newer" than pip's 3.1.0.
New models are filtered server-side per speech tag; filtering an org's latest 50 uploads misses speech
models among nvidia's text and robotics releases. One dead source is recorded and does not abort the
digest. First live digest: transformers 5.13-5.19 newer than the pinned 5.12.1; new unregistered
speech models Qwen3-ASR-0.6B/1.7B (apache-2.0), nemotron-3.5-asr-streaming-0.6b and
parakeet-unified-en-0.6b.

## The first adoption: Qwen3-ASR-1.7B

The process runs end to end from tracker hit to regression test. A candidate profile `hf-cand-519`
(transformers 5.19.0) passes Level 3 for the supported HF model, and Qwen3-ASR reproduces Qwen's
README numbers (test-clean 1.643 vs 1.63, test-other 3.368 vs 3.38;
validation_campaigns/speech_repro/EVIDENCE.md). Two registry fields cover this without a new
abstraction: `profile` names the upstream profile an entry runs in (Level 3 runs that step in it),
and `eval_script` declares a decode-only HF entry's own script, so a model FS cannot train yet (no
FamilySpec) can still be supported for decoding and regression-tested. The candidate reaches
hf-26.04's own peft and accelerate, and the FS trainer runs there. A seeded Parakeet fine-tune is
bit-deterministic within each profile, and the two profiles differ by at most 0.007 loss. Qwen3-ASR
gained a FamilySpec and a collator that wraps `apply_transcription_request`. Fine-tuned in FS on
Earnings-22, it goes 14.54 -> 9.66 on the held-out calls (validation_campaigns/speech_q3asr).

## Promotion: `hf-26.10` is the HF lane

The candidate profile is promoted and renamed `hf-26.10` (the venv keeps its path,
`~/envs/hf-cand-519`). `hf-26.04` is retired but stays in `PROFILES`, so results it produced still
name their stack.

1. **qwen2_audio FamilySpec** re-measured for the transformers 5.19 layout (`model.audio_tower`,
   `model.multi_modal_projector`, `model.language_model`; both calls still positional). A 10-step
   LoRA run with the towers in full PASSes with automatic adapter scoping (224 modules; audio rows
   and placeholders 88/88; tower movement 473/487 and 2/2). On transformers 5.5 it refuses at load.
2. **Gemma-4-E4B training regression**: 30 seeded steps with two runs per profile. Step 1 is
   identical (0.3488). The largest cross-profile difference (0.022) equals the within-profile
   run-to-run difference (0.021), and tower movement is 187 vs 186 of 271. Whisper reads 1.69
   under both profiles; Parakeet differs by at most 0.007.

Level 3 under the promoted default passes 4/4 (gaps +0.004 to +0.014; runner exit 0;
validation_campaigns/speech_repro/EVIDENCE.md).
