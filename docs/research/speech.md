# Speech plane: research record and integration plan

Status: plan. P0 is not started. This document is the research record for adding a training "speech plane" to FoundationScale and the plan to integrate it.

Rules of this document: it is public. Hardware is referred to generically (a GB200 tray). No hostnames, accounts, node names, IP addresses, container registry locations or internal paths appear. FoundationScale positions are cited as `path:line`; upstream positions are cited by their upstream paths. Where the Director decisions and the deep-dives differ, the decisions win.

Summary of the outcome: speech-in training (audio + text into an LLM) runs inside the existing thin plane in Lane A with HF-native audio families and a new optional `speech` dependency extra. Dedicated ASR model types (CTC / transducer / attention-encoder-decoder) land later in Lane B, starting with HF Whisper. TTS and speech-to-speech duplex (Lane C) are deferred and design-only in this cycle. NeMo Speech and icefall are sources of adapted algorithms and reference shapes, not runtime framework dependencies. k2 is rejected for now.

---

## 1. Problem and root cause

### 1.1 What is broken

FoundationScale cannot train speech today. The audio-tower weights in an audio-capable checkpoint measure 0 of 751 tensors moved (`docs/VERIFICATION_MATRIX.md:122-123`): the tower is carried through the run and never exercised. The blocking facts, each with an FS position:

1. **The audio modality is a hard refusal.** `("audio", "FOUNDATIONSCALE_TRAIN_AUDIO_COLUMN")` sits in `UNTRAINABLE_MODALITIES` (`src/foundationscale/train/loop.py:360-363`), read at `src/foundationscale/train/loop.py:4387-4391`; requesting it runs `_untrainable_modality_refusal` (`src/foundationscale/train/loop.py:481-485`) and exits 96 before any row is read. Removing a refusal is a refusal change and needs human approval per `AGENTS.md`.
2. **Exercise marking is image-only.** `exercised = {"image"} if image_declared else set()` (`src/foundationscale/train/loop.py:436`) is the single line that leaves audio out of the exercised set. With audio absent from `exercised`, `model.audio_tower` is classed as a dormant modality tower (`src/foundationscale/train/loop.py:401-433`), `ddp_find_unused_parameters` is forced to `True` (`src/foundationscale/train/loop.py:4998-4999`) and the tower receives no trainable scope: adapters explicitly exclude it from LoRA ("a declaration never asked for", `src/foundationscale/families/adapters.py:147-152`, `:183-188`).
3. **There is no audio dataset contract.** The dataset contract is an unconditional text column (`docs/DATASETS.md:155`). There is no audio row schema, no decode/resample policy, no strict-vs-tolerant rule, and no supervision field for reference text.
4. **`<audio>` is scan-only.** `<audio>` is listed in the modality placeholder set for scanning purposes only (`src/foundationscale/train/loop.py:522`) and produces an UNMEASURED notice (`src/foundationscale/train/loop.py:4656-4662`). Nothing binds N placeholder positions to N waveform segments in a collator and nothing verifies that the counts match.
5. **No length accounting and no bucketing.** There is no audio-to-token length estimator and no 2D `[duration, tokens]` budget, so every audio length in a batch is a claim rather than a measurement. There is also no policy for audio that exceeds a family's fixed audio sequence cap.
6. **Wrong model class risk on Gemma-4.** Gemma-4's causal-LM build never instantiates the audio layer (`docs/STATUS.md:42`), so audio requires the conditional-generation (multimodal) class. Which class FoundationScale loads today is UNVERIFIED; Phase 0 measures it.
7. **No speech metrics and no ASR model types.** There is no WER/CER path with Coverage, no CTC / transducer / encoder-decoder head support, and no TTS.

Supporting constraint: `FS`'s `pyproject.toml` pins `transformers>=4.40`, while native audio families of the kind described here (Gemma-4 audio with feature extractor, mask and placeholder expansion) ship in transformers 5.x as measured in the Megatron-plane container. Which transformers version the thin-plane container carries is UNVERIFIED; Phase 0 measures it.

### 1.2 Root cause

The thin plane was built and verified for text and image. Audio was parked as an intentional fail-closed refusal and all of the surrounding machinery was never written: a declared audio column, a decode and resample path, a collator that resolves placeholders to waveform or feature segments, a length budget, admission of `"audio"` into the exercised set, a trainable scope for the audio tower and its projector, a movement measurement to prove the tower trains, and a WER/CER metric to prove the run learns. These are one unit. Flipping only the refusal at `loop.py:360-363` would produce a run that looks like training while the tower again measures 0 of 751, which is exactly the silent-carrier failure mode the gate doctrine exists to prevent.

---

## 2. Survey

### 2.1 NeMo Speech

**What it is.** NVIDIA NeMo's speech collections: ASR model families (FastConformer, Parakeet and recipes under `examples/asr/`) and `speechlm2`, a speech-LLM stack whose reference model is SALM (speech-audio language model). Launch is Hydra plus PyTorch Lightning (`examples/speechlm2/salm_train.py`, same `hydra_runner` pattern as `examples/asr/speech_to_text_finetune.py`). The model (`nemo/collections/speechlm2/models/salm.py`) composes a pretrained HF causal LLM with an `AudioPerceptionModule` (`nemo/collections/speechlm2/modules/perception`) whose documented flow is preprocessor to encoder to modality adapter to projection, and splices audio frame vectors into the embedding stream at an `audio_locator_tag` placeholder via `replace_placeholders_and_build_targets` (same file), which fail-closes when the replacement count does not match the placeholder count. The data layer is lhotse-based (`nemo/collections/common/data/lhotse/dataloader.py`, `sampling.py`, `audio_loading.py`, `audio_token_estimator.py`).

**Strengths relevant to FoundationScale.**

- Sample-exact audio-to-token length arithmetic (`nemo/collections/common/data/lhotse/audio_token_estimator.py`, centered STFT then conv subsampling then optional chunking). This is what turns claimed lengths into measured lengths.
- 2D `[duration, tokens]` bucketing with strict-mode outlier discard (`FixedBucketBatchSizeConstraint2D`, `MultimodalFixedBucketBatchSizeConstraint2D`, `find_smallest_bucket` in `nemo/collections/common/data/lhotse/sampling.py`), plus token-budgeted dynamic batching (`MultimodalSamplingConstraint`, `PackedTokenConstraint`).
- A strict-versus-fault-tolerant audio loading policy (`nemo/collections/common/data/lhotse/audio_loading.py`) with the fault-tolerant path dropping rows instead of corrupting.
- A fail-closed placeholder-count check (`replace_placeholders_and_build_targets`, `nemo/collections/speechlm2/models/salm.py`) that raises when replacements and placeholders disagree, and label masking of every replaced or pad slot with `-100`.
- Regex parameter freezing with a keep-pattern rescue and an unmatched-pattern warning (`freeze_and_subset` in `nemo/collections/speechlm2/parts/optim_setup.py`) and regex LR multiplier groups (`build_param_groups`, same file).
- Pre-run parallelism compatibility refusals as a pure function (`validate_parallelism_compatibility` in `nemo/collections/speechlm2/parts/parallel.py`): unpacked-pads with CP raises (NaN grads), THD with a non-TE backend raises, THD with TE and fused attention enabled raises on a known-bad capability and warns elsewhere.
- An fp32 dtype pin for operations that assume fp32 under bf16 training (`fp32_precision`, `nemo/collections/speechlm2/parts/precision.py`).

**Weight.** Heavy runtime. The training driver sits on `ModelPT` + Lightning + `exp_manager` + Hydra `exp_manager`-style resume and `.nemo` archives; `speechlm2` additionally pulls a git-pinned `nemo_automodel`, `megatron-fsdp`, `flashoptim` and optional TransformerEngine compiled extras (source builds). FoundationScale delegates stepping to HF `transformers.Trainer` and validates topology itself, so this framework layer is not portable into it.

**License.** Apache-2.0. Adapted files carry attribution and an Apache-2.0 NOTICE entry in FoundationScale.

### 2.2 icefall and k2

**What it is.** icefall is the K2-fsa recipe collection over k2 and lhotse. The canonical ASR recipe is `egs/librispeech/ASR/zipformer/train.py` with `zipformer.py`, `subsampling.py`, `decoder.py`, `joiner.py`, `attention_decoder.py`, `asr_datamodule.py`, `optim.py`, `beam_search.py`. k2 is the loss and decoding library; its pruned RNN-T loss (visible as `k2.RaggedTensor` and `prune_range` in the recipe's `compute_loss` / `model.py`) is the transducer training core.

**Strengths relevant to FoundationScale.**

- One recipe carrying three non-causal head types behind flags: transducer (pruned RNN-T), CTC (`use_ctc`), and an attention encoder-decoder (`use_attention_decoder`), with per-loss warmup scale schedules in `compute_loss` (`egs/librispeech/ASR/zipformer/train.py`).
- The data precondition `remove_short_and_long_utt` (same file): utterances of 1 to 20 seconds whose subsampled frame count `((num_frames - 7) // 2 + 1) // 2` is at least the token count minus 2. This is a hard precondition on model-plus-data compatibility and is the right shape for a Coverage-bearing check.
- Checkpoint averaging math with fp64 accumulation and tied-weight de-duplication (`average_state_dict`, `average_checkpoints_with_averaged_model` in `icefall/checkpoint.py`), plus a bounded-disk checkpoint schema with top-k pruning and mid-epoch resume through sampler state (`save_checkpoint`, `find_checkpoints`, same file).
- `speech_llm/ASR_LLM/whisper_llm_zh/model.py`: `EncoderProjector` (frame stacking plus a two-layer projection) and `_merge_input_ids_with_speech_features`, a splice of speech embeddings at a `default_speech_token_id` that rebuilds `attention_mask`, `labels` and `position_ids`. This is the smallest working template for a custom encoder-plus-LLM family.
- An OOM pre-flight through pessimistic batch scanning (`scan_pessimistic_batches_for_oom` in `egs/librispeech/ASR/zipformer/train.py`).

**Weight.** icefall itself is recipes plus pure-python helpers, but its requirements pull Kaldi-derived compiled wheels (`kaldifst`, `kaldilm`, `kaldialign`, `kaldi-decoder`) and an ONNX export stack; k2 itself is a hand-written CUDA/C++ extension and the hard dependency of the transducer path. Neither k2 nor the Kaldi-derived wheels carry any examined claim of aarch64 or sm_100 support. On a GB200 tray that is UNMEASURED.

**License.** icefall and NeMo are Apache-2.0 (verified). k2 is Apache-2.0 upstream; this was not re-verified here because k2 is rejected regardless.

### 2.3 HF-native audio families (already available)

transformers 5.12.1, as measured in the Megatron-plane container, ships Gemma-4 audio (`feature_extraction_gemma4.Gemma4AudioFeatureExtractor`: 128 mel bins, 16 kHz, 10 ms hop; `processing_gemma4`: given `audio=...` it expands `boa` plus `audio_token` repeated N times plus `eoa`, where N is `input_features_mask.sum`, capped by `audio_seq_length=750`; `modeling_gemma4.Gemma4AudioModel` with `main_input_name` `input_features` and `input_features_mask`) and the native families `qwen2_audio`, `qwen2_5_omni`, `qwen3_omni_moe`, `voxtral`, `granite_speech`, `phi4_multimodal`, `whisper`. Weight: no new framework dependency beyond the transformers pin; this is why Lane A stays inside the thin plane. License: Apache-2.0 upstream (transformers).

### 2.4 Supporting dependencies (verified facts)

| package | version | shape | notes |
|---|---|---|---|
| lhotse | 1.33.0 | pure-python wheel (deps: audioread, SoundFile, cytoolz, intervaltree, numpy, pyyaml) | manifests, Cut schema, `DynamicBucketingSampler` with resumable state |
| kaldialign | 0.12.0 | compiled, but ships aarch64 manylinux wheels for cp310-313 | WER/CER edit distance |
| jiwer | 4.0.0 | pure-python (rapidfuzz dependency) | WER/CER, text normalization helpers |

---

## 3. Comparison table

| aspect | NeMo Speech | icefall / k2 | HF transformers (5.x native audio) |
|---|---|---|---|
| training driver | Hydra + Lightning + `ModelPT` (`examples/speechlm2/salm_train.py`) | plain `train.py` loop over DDP | HF `transformers.Trainer`, already used by the FS thin plane |
| speech-in model | SALM: perception module plus modality adapter plus LLM (`nemo/collections/speechlm2/models/salm.py`) | `whisper_llm_zh` bridge: `EncoderProjector` plus merge function | native audio tower inside the HF checkpoint per family |
| dedicated ASR types | FastConformer / Parakeet recipes (CTC / transducer) | zipformer with CTC, transducer, attention-encoder-decoder in one recipe | Whisper encoder-decoder |
| data and manifests | lhotse CutSet / Shar (`nemo/collections/common/data/lhotse/dataloader.py`) | lhotse CutSet via `asr_datamodule.py` | HF datasets audio column (no manifest layer) |
| length accounting | `AudioTokenEstimator`, measured token counts | frame arithmetic in `remove_short_and_long_utt` | mask sums reported by the processor |
| batch budget / bucketing | 2D `[duration, tokens]` constraints and bins (`sampling.py`) | `DynamicBucketingSampler` with `max_duration` | none natively |
| placeholder splice | `replace_placeholders_and_build_targets`, fail-closed count check | `_merge_input_ids_with_speech_features` | processor-side placeholder expansion per family |
| strict data loading | strict vs fault-tolerant policy, drops visible in loader state (`audio_loading.py`) | `save_bad_model` and continue (deliberately lenient) | none natively |
| trainable scope | regex freeze with keep-rescue (`parts/optim_setup.py`) | per-loss heads with fixed trainables | PEFT-compatible module trees |
| metrics | kaldialign in dependencies; metric glue not in the examined corpus | kaldialign in requirements; usage not in the examined corpus | none natively |
| checkpoint / resume | DCP, `.nemo`, HF dirs (three coexisting surfaces) | top-k single-file checkpoints plus sampler state | Trainer safetensors, already used |
| run-loop weight | heavy: Lightning, Hydra, `exp_manager`, `ModelPT` | medium: own loop, own optimizers, optional DeepSpeed config in one recipe | light: already inside the thin plane |
| compiled extensions | TE / flash-attn source builds in the `compiled` extra | k2 (CUDA/C++), Kaldi-derived wheels, ONNX stack | none required on the audio path |
| aarch64 / sm_100 evidence | upstream parallelism guard keys on `_SM120`; a GB200 tray lands in the upstream warn-only branch | none for k2; UNMEASURED | version in the thin-plane image UNVERIFIED (P0) |
| license | Apache-2.0 | Apache-2.0 | Apache-2.0 |

---

## 4. Decision table

Verb meanings, as decided: **INTEGRATE** = consume as a dependency under a new optional `speech` extra (or as the already-pinned transformers); **ADAPT** = port code into FoundationScale with attribution and an Apache-2.0 NOTICE entry; **REFERENCE** = use the shape only; **REJECT** = do not take.

| component | source | verdict | why |
|---|---|---|---|
| lhotse manifests, Cut schema, `DynamicBucketingSampler` (resumable state) | lhotse 1.33.0 | INTEGRATE | optional `speech` extra; supplies manifest/Cut schema and resumable bucketing. FS row schema mirrors the lhotse Cut fields (id, audio source, sampling_rate, num_samples, duration, supervision text / messages). The thin path must also work with no lhotse against a plain HF-datasets audio column |
| WER/CER edit distance | kaldialign 0.12.0 or jiwer 4.0.0 | INTEGRATE | one of the two, chosen in Phase 2 by measuring both on the same fixture set under the `speech` extra |
| HF-native audio families (gemma4 first, then qwen2_audio, voxtral, granite_speech, qwen3_omni) | transformers 5.x | INTEGRATE | already-pinned dependency; the HF processor owns feature extraction and placeholder expansion, FS owns contract, loading, collating, bucketing, exercised logic, adapter scope, gates and manifest. Registered as `FamilySpec`s |
| HF Whisper encoder-decoder | transformers 5.x | INTEGRATE | Lane B starting point for dedicated ASR; in transformers already, no new stack |
| `AudioTokenEstimator` idea (measured token counts per row) | NeMo `nemo/collections/common/data/lhotse/audio_token_estimator.py` | ADAPT | every audio row needs a measured token count before batching or its length is a claim, not a measurement |
| 2D `[duration, tokens]` bucketing constraint | NeMo `nemo/collections/common/data/lhotse/sampling.py` | ADAPT | closes the missing length-bucketing gap with a two-dimensional budget and measured bin contents |
| strict-vs-fault-tolerant audio loading | NeMo `nemo/collections/common/data/lhotse/audio_loading.py` | ADAPT | FS default is STRICT (an unreadable row refuses with exit 96); the tolerant mode must count every drop in the manifest, never drop silently |
| `replace_placeholders_and_build_targets` fail-closed placeholder-count check | NeMo `nemo/collections/speechlm2/models/salm.py` | ADAPT | for families without a native processor, the replacement count must equal the placeholder count or fail closed; also the `-100` label masking shape |
| `freeze_and_subset` regex freezing with unmatched pattern = fail | NeMo `nemo/collections/speechlm2/parts/optim_setup.py` | ADAPT | adapter and freeze scope becomes measured; the upstream unmatched-pattern warning becomes an FS failure (exit 96) |
| `validate_parallelism_compatibility` as pre-run refusal gates | NeMo `nemo/collections/speechlm2/parts/parallel.py` | ADAPT | three known silent-corruption combinations become pre-run refusals. Upstream's guard keys on `_SM120`; a GB200 tray falls in the upstream warn-only branch, so FS extends the capability coverage or refuses by rule |
| `fp32_precision` | NeMo `nemo/collections/speechlm2/parts/precision.py` | ADAPT | trivial dtype pin under bf16 training; ported as an FS-scoped context manager with no NeMo imports |
| `remove_short_and_long_utt` precondition (subsampling frames >= tokens) | icefall `egs/librispeech/ASR/zipformer/train.py` | ADAPT | a Coverage-bearing precondition for transducer-class families; 0 rows checked is VACUOUS |
| checkpoint averaging math (`average_state_dict`, fp64 accumulation, tied-weight dedupe) | icefall `icefall/checkpoint.py` | ADAPT | optional post-train step behind a gate; adds "claimed versus measured" for the averaged model |
| `EncoderProjector` and `_merge_input_ids_with_speech_features` as a template | icefall `speech_llm/ASR_LLM/whisper_llm_zh/model.py` | ADAPT | the template for a future custom encoder-plus-LLM family: projector shape and splice that rebuilds masks and labels |
| zipformer and NeMo FastConformer / Parakeet recipes and architectures | icefall `egs/librispeech/ASR/zipformer/`, NeMo ASR | REFERENCE | shapes for Lane B non-causal head families; no code taken |
| lhotse Cut/I/O shape and bin-estimation ideas (`CutSet`, `AudioSamples`, bin estimate) | lhotse via NeMo | REFERENCE | adopt the Cut field shape as the FS row schema and precompute bins as a recorded pre-run artifact rather than a startup side effect |
| SALM model stack shape (perception / connector / LLM stacking order) | NeMo `nemo/collections/speechlm2/models/salm.py` | REFERENCE | teaches the audio-tower subtree shape that maps onto `FamilySpec` audio tower declarations |
| packed-sequence splice and per-utterance label shift | NeMo `nemo/collections/speechlm2/parts/packed_sequences.py` | REFERENCE | shape only. THD packing is refused in the thin plane until a backend lane owns it |
| `speechlm2` duplex_s2s and EAR-TTS | NeMo `speechlm2` | REFERENCE | Lane C only; TTS and duplex are deferred this cycle |
| torchaudio `rnnt_loss` (transducer loss candidate) | torchaudio | INTEGRATE if measured, else REJECT | the candidate transducer loss for Lane B. Resolution happens by measurement on the Lane B fixture set; k2 is not the fallback |
| `k2` pruned RNN-T loss and decoders | k2 | REJECT | compiled CUDA/C++ extension with no measured aarch64 or sm_100 support; unmeasured means refuse, not depend |
| `kaldifst`, `kaldilm`, kaldi-decoder decoders | Kaldi-derived | REJECT | no call site in the FS path; compiled wheel risk on aarch64 |
| `ModelPT`, Lightning, Hydra, `exp_manager`, `.nemo` archives | NeMo | REJECT | heavy lifecycle FS replaced with topological validation plus HF Trainer plus save gates |
| `nemo_automodel` and `AutomodelParallelStrategy` | NeMo `speechlm2` | REJECT | TP/PP/EP are refused in the thin plane; this stack belongs to the Megatron lane, which stays text-only |
| TransformerEngine and other compiled extras (flash-attn, mamba, deep_ep, grouped-gemm) | NeMo `compiled` extra | REJECT | source-built CUDA kernels fight container CI and `make check`; upstream's TE fused-attention path is itself a corruption source |
| NeMo DataModule and BroadcastingDataLoader | NeMo | REJECT | a competing control plane over the data loop FS owns |
| lhotse augmentation chains (noise, speed perturbation, RIR, compression, clipping, lowpass) | lhotse / NeMo | REJECT for now | augmentation mutates measured durations under a Coverage doctrine; deferred until a run exists with measured before and after counts |
| icefall `ScaledAdam` and `Eden` | icefall `optim.py` | REJECT for now | would replace HF Trainer's optimizer path and fork the single delegated step |
| DeepSpeed configurations (ZeRO-1 and siblings) | icefall recipe trees | REJECT | no Coverage path in the FS loop; sharding policy is the decision of the run layer, not a recipe config file |
| Whisper encoder monkey-patch | icefall `whisper_llm_zh/whisper_encoder_forward_monkey_patch.py` | REJECT | an untrackable patch point cannot satisfy "claimed versus measured" |
| ONNX / TorchScript / ncnn export paths | icefall export scripts and requirements | REJECT | deployment plane, outside train and gate scope; heavy CI weight |

---

## 5. Architecture

Three lanes, sharing one audio data contract. Lane A is built first and inside the existing thin plane.

**Lane A, speech-in LLMs.** Audio and text into an LLM (ASR-shaped SFT is the first task). Built inside the existing thin plane using HF Trainer. Families are HF-native audio families: Gemma-4 first (its audio tower is already declared in `FamilySpec`, `src/foundationscale/families/registry.py:171-175`), then `qwen2_audio`, `voxtral`, `granite_speech`, `qwen3_omni` as new `FamilySpec` entries. Ownership is split deliberately. The HF processor owns feature extraction and placeholder expansion (for Gemma-4: 128 mel bins, 16 kHz, 10 ms hop; `boa` plus `audio_token` x N plus `eoa` with N taken from `input_features_mask.sum`, capped by `audio_seq_length=750`). FoundationScale owns: the audio dataset contract, loading and resampling, the collator, length bucketing, exercised-tower logic, adapter scope, gates and the run manifest.

**Lane B, dedicated ASR models.** CTC, transducer and attention-encoder-decoder families. Deferred after Lane A. Starts with HF Whisper (encoder-decoder) because it is in transformers. icefall zipformer and NeMo FastConformer / Parakeet are REFERENCE for architecture and recipe shapes. k2's pruned RNN-T loss is REJECT for now (compiled extension, unmeasured on aarch64 sm_100); torchaudio `rnnt_loss` is the candidate to evaluate for the transducer loss. Lane B reuses the Lane A data contract, loader and WER metric.

**Lane C, TTS and speech-to-speech duplex.** Deferred. NeMo `speechlm2` duplex_s2s and EAR-TTS are REFERENCE only. This cycle produces a design note (Phase 5), no code.

Boundary rule that holds for all lanes: 6D parallelism and TP/PP/EP work stays in the Megatron lane (`src/foundationscale/rl/megatron`), text-only. Nothing speech lands there in this cycle.

### Lane A data flow

```
+--------------------------------------------------------------------+
| Lane A: speech-in LLM, thin plane, HF Trainer                      |
+--------------------------------------------------------------------+

 manifest row  {id, audio source, sampling_rate, num_samples,
                duration, supervision text / messages}
   (lhotse-Cut-shaped schema; lhotse optional under `speech`,
    plain HF-datasets audio column also accepted)
        |
        v
 [1] FS row contract + preconditions              (new, src/foundationscale/data/)
     - family declares an audio tower             (families/registry.py)
     - FOUNDATIONSCALE_TRAIN_AUDIO_COLUMN accepted
       only if the family and the processor agree
     - else: still refuse, exit 96                (train/loop.py:360-363, :481-485)
        |
        v
 [2] FS audio loader  (soundfile)
     - decode -> mono mixdown -> resample to processor rate
     - STRICT default: unreadable row = exit 96
     - tolerant mode: drop and COUNT in the manifest (never silent)
        |
        v
 [3] length policy
     - measured token pre-count per row           (AudioTokenEstimator shape)
     - 2D [duration, tokens] bucketing            (lhotse DynamicBucketingSampler
     - duration / frames-vs-tokens preconditions   optional under `speech`)
     - truncation past audio_seq_length: refuse or count
        |
        v
 [4] FS audio collator
     - '<audio>' placeholder POSITIONS resolved    (train/loop.py:522 becomes
       to N audio segments, N per sample,            resolution, not scan-only)
     - features + placeholder expansion delegated
       to the HF processor (boa + audio_token*N + eoa)
     - families without a native processor use the
       SALM-style fail-closed replacement check
     - labels masked -100 over audio and pad slots
        |
        v
 [5] batch -> HF Trainer step
     input_features, input_features_mask,
     input_ids, labels, attention_mask
     - model forward with the audio tower exercised
       ("audio" admitted to exercised,              train/loop.py:436)
     - adapter scope: LoRA on the LLM by default,
       full fine-tune on the audio tower /
       audio projector on explicit declaration     (families/adapters.py)
     - find_unused_parameters derived False        (train/loop.py:4998-4999)
        |
        v
 [6] gates + manifest (per step / per epoch / per run)
     AudioDeclarationGate, AudioPlaceholderCoverageGate,
     AudioLoadCoverageGate, AudioTowerMovementGate,
     WER/CER metric with Coverage
     manifest rows: audio rows checked / expected, total audio
     seconds, sampling rate, processor + feature-extractor config,
     lhotse version
        |
        v
 [7] save gates -> checkpoint + evidence files
     (post-train checkpoint averaging, optional, gate-checked)
```

---

## 6. Phased plan

A phase is DONE only when its exit criteria are verified on a GB200 tray, not just in CI. Criteria marked (CI) are necessary but not sufficient. Evidence files are JSON, committed under `docs/research/evidence/`.

### P0, de-risk (no FoundationScale code)

Goal: answer the two UNVERIFIED questions and prove the audio tower can train at all.

Work: in the thin-plane container on a GB200 tray (2 free GPUs), a plain HF script that (a) records the container's transformers version, (b) loads the Gemma-4 multimodal (conditional-generation) class and records the loaded class name, (c) runs `processor(audio=16 kHz wav, text)` and asserts the count of audio placeholder tokens equals `input_features_mask.sum`, (d) runs forward plus backward and asserts the Gemma-4 audio tower parameters receive non-zero gradients and the loss is finite, (e) confirms a small ASR dataset (for example a LibriSpeech dev-clean subset) is reachable from the container.

Exit criteria (falsifiable): a JSON evidence file containing the measured transformers version, the loaded class name, the measured placeholder count and mask sum (must be equal), measured non-zero gradient norms over at least one audio-tower tensor, a finite loss value, and the declared location and size of the ASR subset. Any assertion failure is a documented blocker returned to planning.

Files touched: `docs/research/evidence/speech-p0-gemma4-audio.json` (new). No source file.

Status: **DONE, PASS (2026-10-06).** Evidence: `validation_campaigns/speech_p0/EVIDENCE.md`. Both UNVERIFIED questions are closed: the thin-plane container carries transformers 5.5.0 with Gemma-4 audio support, and `AutoModelForCausalLM` already builds `Gemma4ForConditionalGeneration` with the audio tower present (271 parameter tensors, 305M parameters). Placeholder counts equal the tower's output lengths, zero-shot WER on four dev-clean utterances is 0.133, and all 271 tower tensors receive a non-zero gradient. The "0 of 751" negative control counts 271 parameters plus 480 clipping buffers.

Maintainer decision (2026-10-06): the refusal change is approved. P1/P2 may accept a declared audio column for families whose processor and towers support it; every other audio case keeps refusing with exit 96.

### P1, audio data contract and collator (TDD, CPU-testable)

Goal: an audio contract that is accepted only when it can be verified, and a collator that resolves `<audio>`.

Work: `FOUNDATIONSCALE_TRAIN_AUDIO_COLUMN` is accepted only when all three hold: the family declares an audio tower and the processor exposes `audio_token`, `boa` and `eoa`, and the feature extractor reports a known `sampling_rate`. Otherwise the run still refuses with exit 96 and a named reason. Row schema mirrors the lhotse Cut fields (id, audio source, sampling_rate, num_samples, duration, supervision text or messages) and must work without lhotse on a plain HF-datasets audio column. Loader: soundfile decode, mono mixdown, resample to the processor rate, STRICT by default (unreadable or unsupported row = exit 96), tolerant mode counts drops in the manifest. Collator mirrors the image path: `<audio>` is resolved to N segments per sample (no longer scan-only); feature extraction and placeholder expansion are delegated to the HF processor for native families; families without a native processor use the fail-closed replacement-count check adapted from SALM. Length preconditions (duration bounds, measured token pre-count) run before batching. Truncation past the family's `audio_seq_length` is refused or counted, never silent. Video stays refused.

Exit criteria (falsifiable): (CI) unit tests prove, per rule, both the accept path and each refusal reason; the collator test proves an N-placeholder sample binds exactly N segments and the label mask is `-100` over audio and pad slots only; a same-count-versus-mismatched-count pair proves the fail-closed check; a tolerant-loader fixture proves drops are counted and strict mode refuses; a truncation fixture proves the event is counted. (tray) the same loaders and collator run against the P0 ASR subset in the thin-plane container and produce one batch whose recorded total audio seconds and sampling rate match the manifest rows.

Files touched: `src/foundationscale/train/loop.py` (contract acceptance and refusal text at `:360-363`, `:481-485`; `<audio>` resolution at `:522`; UNMEASURED notice at `:4656-4662` becomes a measured row); `src/foundationscale/data/audio_rows.py` (new); `src/foundationscale/data/audio_loading.py` (new); `src/foundationscale/data/audio_collator.py` (new); `src/foundationscale/data/token_budget.py` (new, adapted from `nemo/collections/common/data/lhotse/audio_token_estimator.py` and `sampling.py`, with attribution and Apache-2.0 NOTICE); `docs/DATASETS.md` (`:155` text-only contract replaced); `pyproject.toml` (new `speech` extra carrying lhotse and the two WER candidates); `tests/data/test_audio_rows.py` (new); `tests/data/test_audio_loading.py` (new); `tests/data/test_audio_collator.py` (new); `tests/data/test_token_budget.py` (new); `tests/train/test_modality_refusal.py` (audio fixtures added).

### P2, training wiring and gates

Goal: audio declared means audio trained and measured, with gates that cannot pass vacuously.

Work: `"audio"` enters the exercised set when the contract is accepted (`src/foundationscale/train/loop.py:436`) and `derive_find_unused_parameters` then reports `False` (existing test expectations at `tests/families/test_family_towers.py:55-57`); `ddp_find_unused_parameters` stops being forced `True` for these runs (`train/loop.py:4998-4999`). Adapter scope gains explicit declaration: default is LoRA on the LLM and full fine-tune on the audio tower and audio projector when declared; the tower exclusion in `families/adapters.py:147-152`, `:183-188` stays the default behavior and any override requires a declaration and a measured gate pass. WER/CER is implemented with Coverage over kaldialign and jiwer; both are measured on the same fixture set and exactly one remains in the `speech` extra. The run manifest records audio rows checked and expected, total audio seconds, sampling rate, processor and feature-extractor config, and the lhotse version when present. Regex freezing and pre-run parallelism refusals land as gates (see section 7). Removal of the audio refusal entry at `train/loop.py:360-363` is requested from the maintainer as a refusal change under `AGENTS.md` before this phase can be enabled in default runs.

Exit criteria (falsifiable): (CI) every gate in section 7 has a MUST_FIRE test proving it is reached and a MUST_PASS fixture proving it passes, plus its named refusal fixture still exiting 96; the WER fixture is exact (a 3-word reference with 1 wrong word measures 1/3); 0 audios checked in an audio run is VACUOUS and fails; existing image-only expectations are unchanged (the 0 of 751 measurement at `docs/VERIFICATION_MATRIX.md:122-123` is retained as the negative control). (tray) one short CPU-plus-1-GPU audio step in the thin-plane container writes a manifest with the new audio rows and a positive AudioTowerMovementGate reading; the move count is greater than 0 against the 0 of 751 baseline.

Files touched: `src/foundationscale/train/loop.py` (`:436`, `:401-433`, `:4998-4999`); `src/foundationscale/families/adapters.py` (`:147-152`, `:183-188`, new declarations); `src/foundationscale/families/registry.py` (`:56`, `:59-90`, `:171-175`) and the per-family spec module under `src/foundationscale/families/`; `src/foundationscale/gates/speech_gates.py` (new, G1 to G4, G6 to G8); `src/foundationscale/metrics/wer.py` (new, G5); `pyproject.toml` (WER candidate resolved to one package in the `speech` extra); `tests/gates/test_speech_gates.py` (new); `tests/metrics/test_wer.py` (new); `tests/train/test_modality_refusal.py`; `tests/families/test_family_towers.py` (`:55-57` expectations updated where DDP derivation changes); `docs/VERIFICATION_MATRIX.md` (new gate rows); `docs/STATUS.md` (audio status line).

### P3, hardware validation on a GB200 tray

Goal: prove speech training end to end on the real target.

Work: Gemma-4 ASR-shaped SFT on the P0 LibriSpeech subset, two arms (LoRA on the LLM with full fine-tune tower; full fine-tune of LLM plus tower), 1 tray, FSDP as used by the thin plane. Pre-declare the accepted WER improvement margin before the run and record it in the evidence file. Then run one throughput comparison with length bucketing on and off.

Exit criteria (falsifiable): AudioTowerMovementGate reads greater than 0 moved tensors inside the audio-tower subtree on both arms (versus 0 of 751 baseline); WER on the held-out dev split drops versus step 0 by at least the pre-declared margin and the metric carries non-VACUOUS Coverage; the save gate passes and the run resumes from its own checkpoint and finishes; the bucketing on/off throughput numbers are recorded. All values land in one JSON evidence file.

Files touched: `docs/research/evidence/speech-p3-asr-sft.json` (new); `docs/VERIFICATION_MATRIX.md` (0 of 751 row gains its positive control sibling); `docs/STATUS.md`; run config consumed by `src/foundationscale/train/loop.py` (env `FOUNDATIONSCALE_TRAIN_AUDIO_COLUMN` and `speech` extra). No source change expected; any source change here reopens phase sign-off.

### P4, more families and Lane B

Goal: generalize Lane A and open Lane B.

Work: register `qwen2_audio`, `voxtral`, `granite_speech`, `qwen3_omni` as `FamilySpec`s and run their gate fixtures; start Lane B with HF Whisper as an encoder-decoder ASR family; evaluate torchaudio `rnnt_loss` on the Lane B fixture set as the transducer-loss candidate and record the INTEGRATE-or-REJECT decision with measurements; land the adapted icefall checkpoint averaging as an optional post-train step behind a gate; land the `remove_short_and_long_utt` shape as a Coverage-bearing precondition for transducer-class families; keep the `whisper_llm_zh` `EncoderProjector` and merge-function adaptation as the template branch for a later custom encoder-plus-LLM family (not a deliverable of P4).

Exit criteria (falsifiable): (tray) each new family passes AudioDeclarationGate, AudioPlaceholderCoverageGate, AudioLoadCoverageGate and AudioTowerMovementGate on one short run; one Whisper run trains and the WER metric with Coverage reports on a held-out split; the averaging step writes a gate-checked evidence row (averaged tensor count equals expected, tied weights de-duplicated); the transducer-loss candidate carries measured fixture results and an explicit verdict.

Files touched: `src/foundationscale/families/` (new per-family spec modules); `src/foundationscale/train/checkpoint_avg.py` (new, adapted from `icefall/checkpoint.py`, attribution and NOTICE); `src/foundationscale/data/token_budget.py` (Coverage-bearing precondition); `src/foundationscale/gates/speech_gates.py` (averaging gate); `src/foundationscale/metrics/wer.py` (encoder-decoder eval path); `pyproject.toml` (`speech` extra revisited if the transducer candidate integrates); tests per component; `docs/VERIFICATION_MATRIX.md`; `docs/research/evidence/speech-p4-families.json` (new).

### P5, Lane C and speech RL (design only)

Goal: a written design, no code. GRPO with a WER reward through the existing RL plane; TTS and duplex (EAR-TTS, `speechlm2` duplex_s2s) as REFERENCE material only.

Exit criteria (falsifiable): a committed design note covering the reward definition (WER with Coverage, refusal on VACUOUS reward inputs), the reward-model boundary in the RL plane, and the data contract reuse from Lane A. No source file changes in P5.

Files touched: `docs/research/speech-rl-design.md` (new).

---

## 7. New gates

Every gate has a MUST_FIRE condition (the conditions under which the gate must be evaluated; if they do not hold the gate must stay silent and the run must stay on the text-only path) and a MUST_PASS fixture (a concrete input that must produce a measured pass). Refusal fixtures listed with each gate must exit 96. 0 rows checked where rows are expected is VACUOUS and is a failure, not a pass.

**G1 AudioDeclarationGate**

- MUST_FIRE: every run that sets `FOUNDATIONSCALE_TRAIN_AUDIO_COLUMN`, and every run whose `FamilySpec` declares an audio tower, evaluated before the dataloader is built.
- MUST_PASS fixture: an audio family (Gemma-4) with the audio column declared, the processor exposing `audio_token`, `boa`, `eoa`, and a feature extractor reporting `sampling_rate` 16000. Evidence: declaration accepted, modality recorded in the manifest.
- Refusal fixtures (each exit 96 with the named reason): audio column set with (a) a family that declares no audio tower, (b) a processor missing any of `audio_token` / `boa` / `eoa`, (c) an unknown or absent feature-extractor `sampling_rate`.

**G2 AudioPlaceholderCoverageGate**

- MUST_FIRE: every training and eval batch that contains audio rows, before the forward step. 0 rows checked in an audio run is VACUOUS and fails.
- MUST_PASS fixture: a batch of N samples in which, per sample, the count of audio placeholder positions equals the number of supplied audio segments equals `input_features_mask.sum` (for a native-processor family, within the family cap, 750 tokens for Gemma-4). For a family without a native processor the adapted SALM replacement-count check (upstream `nemo/collections/speechlm2/models/salm.py`) must report equal counts. Evidence row: checked = N, expected = N, per-sample counts recorded.
- Refusal fixture: a sample carrying 2 placeholders and 1 audio segment must fail closed exactly as the upstream `ValueError` on replacements does (`nemo/collections/speechlm2/models/salm.py`); recorded as exit 96.

**G3 AudioLoadCoverageGate**

- MUST_FIRE: over every audio source in the loaded manifest set, once per dataset pass.
- MUST_PASS fixture: a set whose rows all decode and resample to the processor rate. Evidence rows: checked = expected = number of audio rows, drops = 0, total audio seconds > 0, measured sampling rate recorded. Tolerant-mode companion fixture: the same set with 2 deliberately corrupt files must record checked = expected and drops = 2 in the manifest. Strict mode on that fixture exits 96 at the first unreadable row. A silent drop in any mode is a failure.

**G4 AudioTowerMovementGate** (positive control for the 0 of 751 negative control)

- MUST_FIRE: any run in which `"audio"` is in the exercised set, after the first optimizer step.
- MUST_PASS fixture: at least 1 optimizer step over an audio-bearing batch on GPU. Measured moved-tensor count inside the `model.audio_tower` subtree (plus the audio projector scope when declared) must be greater than 0 and recorded. Recorded alongside: the image-only negative control must still measure 0 moved tensors for the tower (`docs/VERIFICATION_MATRIX.md:122-123`), and `derive_find_unused_parameters` must report `False` for the audio run (expectation shape at `tests/families/test_family_towers.py:55-57`).

**G5 WerCerMetric with Coverage**

- MUST_FIRE: every eval on a split that declares both audio and reference text. 0 references scored is VACUOUS and fails.
- MUST_PASS fixture: a set of K reference-and-hypothesis pairs scored with the Phase 2 winner (kaldialign or jiwer). Output carries WER, CER and Coverage rows (references scored over references expected). Exactness fixture: a 3-word reference with 1 wrong word must measure 1/3.
- Refusal fixture: an eval split with audio but no reference text must refuse (exit 96) rather than report 0.0 WER.

Additional pre-run gates in the same format, landing with the ADAPT decisions:

**G6 AudioParallelPreflightGate** (adapted from `nemo/collections/speechlm2/parts/parallel.py`, `validate_parallelism_compatibility`)

- MUST_FIRE: before GPU allocation for any audio run that sets context-parallel size greater than 1 or selects a THD or fused-attention attention path.
- MUST_PASS fixture: a legal audio config (no CP with unpacked padding; non-THD attention) passes and records the checks evaluated. Refusal fixtures (each exit 96): BSHD-style padding with CP greater than 1; THD with a non-TE attention backend; THD with TE and fused attention enabled on a device capability the upstream guard marks known-bad. Note: upstream keys its hard error on `_SM120` and warns otherwise, so a GB200 tray sits in the upstream warn-only branch; FS extends the capability coverage or refuses by rule (UNMEASURED = refuse).

**G7 AudioFreezeScopeGate** (adapted from `nemo/collections/speechlm2/parts/optim_setup.py`, `freeze_and_subset`)

- MUST_FIRE: every run that installs an adapter scope or freeze and keep regexes while audio is present.
- MUST_PASS fixture: every declared regex matches at least 1 named parameter; the resulting trainable set is recorded as checked over expected counts. Refusal fixture: any unmatched pattern is exit 96 (stricter than the upstream "bad regexp?" warning).

**G8 AudioLengthPrecondition** (adapted from icefall `egs/librispeech/ASR/zipformer/train.py`, `remove_short_and_long_utt`)

- MUST_FIRE: every audio row, before sampling and batching.
- MUST_PASS fixture: rows within the duration bounds whose subsampled frame count `((num_frames - 7) // 2 + 1) // 2` is at least the target token count minus 2 pass with rows-checked counts recorded. Refusal fixtures: a row designed to exceed the family's audio sequence cap must be refused (exit 96) or excluded with a counted drop entry, never silent; a row with fewer frames than tokens is excluded and counted for transducer-class families. 0 rows checked is VACUOUS.

---

## 8. Risks and open questions

Risks.

1. **Transformers version split.** Native Gemma-4 audio classes are measured in transformers 5.12.1 in the Megatron-plane container. The thin-plane container image and its transformers version are UNVERIFIED. If the thin plane carries a pre-5.x version there is no Gemma-4 audio family and P0 blocks. Mitigation: P0 measurement; the gate rule stays fail-closed if the classes are absent.
2. **Causal-LM versus multimodal class on Gemma-4.** The causal-LM build never instantiates the audio layer (`docs/STATUS.md:42`). If FoundationScale loads the causal-LM class today, the audio tower is not a real submodule and the movement gate can only fail. P0 records the loaded class.
3. **FSDP wrapping.** With Gemma-4 audio inside `_no_split_modules` class trees, unit wrapping may be wrong for the audio block: memory blow-up, prefetch misbehavior, or a movement measurement that counts whole units instead of tensors. Mitigation: measure wrap granularity in P3 and record it with the movement numbers.
4. **Long audio truncation.** Gemma-4 caps audio at `audio_seq_length=750` tokens. Longer input is truncated by the processor. Silent truncation would break the Coverage doctrine; G8 and G2 must make it a counted event or a refusal.
5. **Data I/O throughput.** Decoding wav with soundfile on Grace-class CPUs on a GB200 tray may starve the GPU. Mitigation: the post-training P3 bucketing on/off throughput measurement, and precomputed features are not in scope, so decode cost must be measured honestly.
6. **DDP `find_unused_parameters` flip.** It is forced `True` (`train/loop.py:4998-4999`) and must derive `False` once audio is exercised, or the movement gate is masked by unused-parameter tolerance. This changes an existing tested expectation (`tests/families/test_family_towers.py:55-57`).
7. **Refusal removal process.** Enabling audio in default runs requires removing the `FOUNDATIONSCALE_TRAIN_AUDIO_COLUMN` refusal row (`train/loop.py:360-363`). This is a refusal change and needs human approval per `AGENTS.md`; it is requested from the maintainer before P2 enablement.
8. **WER tooling choice is open until measured.** kaldialign is compiled but ships aarch64 manylinux cp310-313 wheels; jiwer is pure python via rapidfuzz. If neither measures well on the fixture set, the P2 metric slice stalls and the phase cannot close.
9. **Optional dependency drift.** lhotse 1.33.0 is pure python, but its audio backends (audioread, SoundFile) introduce OS-level codec behavior. The thin path must keep working with no lhotse installed.
10. **Upstream parallelism guard coverage.** The adapted pre-run refusal keys upstream on `_SM120`; a GB200 tray lands in the warn-only branch upstream. FS must either extend the capability guard with measurement or refuse the configuration outright.

Open questions.

1. Which model class FoundationScale loads for Gemma-4 today (causal-LM or conditional-generation), and what changes if it is the wrong one. (P0.)
2. Exact transformers version carried by the thin-plane container image, and whether Lane A needs a container update. (P0.)
3. Is a LibriSpeech dev-clean subset (and its license and location provenance) already available to the training containers. (P0.)
4. Pre-declared acceptable WER improvement margin for P3, and the precise held-out dev split definition. (Before the P3 run.)
5. kaldialign or jiwer for the WER metric, plus which text normalization applies before alignment (no decision made).
6. Audio sequence caps for the non-Gemma-4 families and, where a family has no native processor, which family is the first user of the SALM-style fail-closed splice.
7. Whether the audio tower and the audio projector need separate learning-rate groups (the `build_param_groups` shape supports it) and what the default should be for full fine-tune.
8. Whether the transducer loss resolves to torchaudio `rnnt_loss` in Lane B, and whether Whisper encoder-decoder plus CTC covering heads is enough for P4 before any transducer work.

---

## 9. Non-goals

Not goals of this cycle, by decision:

1. TP, PP or EP for speech models. The thin plane refuses these, and the Megatron lane stays text-only.
2. Streaming ASR. No chunked, causal or online decoding path.
3. TTS and speech-to-speech duplex. Lane C receives a P5 design note only (NeMo `speechlm2` duplex_s2s and EAR-TTS as REFERENCE).
4. Audio augmentation. No noise mixing, speed perturbation, RIR, compression, clipping or lowpass chains until measured before and after counts exist. This includes lhotse augmentation chains.

Also excluded, consistent with the REJECT list in section 4: no NeMo framework import (ModelPT, Lightning, Hydra, `exp_manager`, `.nemo`), no `nemo_automodel` or TE compiled extras, no k2 or Kaldi-derived decoders, no DeepSpeed configs, no Whisper encoder monkey-patches, no ONNX or TorchScript export, no serving plane work, and no packed-sequence or THD training path in the thin plane.
