# Text-to-speech (TTS) on GB200: zero-shot voice cloning and Seed-TTS-eval

This example runs the Seed-TTS-eval benchmark on GB200 (aarch64): three zero-shot voice-cloning synthesizers, the upstream Seed-TTS-eval scoring code run unmodified as subprocesses, and a Vietnamese judge for the languages `run_wer.py` does not cover. The governing rule is upstream: **"integrated" means "reproduced"** — the harness must reproduce the published scores of the real recordings ("Human" row) before it scores a synthesizer. Everything with a number below was measured and is **VERIFIED**; parts that have not run are marked **PLANNED** and carry no commands. Per-row output is kept schema-compatible with `speech_p7/paired_bootstrap.py`; the plan of record is `docs/research/tts_omni_plan.md`. Related example: [speech-2xH200](../speech-2xH200/).

All script paths below are repo-relative. Scripts take their environment/container and weight directories as arguments; these placeholders are written `<env>`, `<weights>` and `<path>`.

---

## What you have at the end

- Three pinned environments, one per upstream: **tts-judge**, **tts-qwen**, **tts-voxcpm**.
- Downloaded models and test sets with `sha256` manifests (public repos only, no token).
- Synthesized wav sets (`<filename>.wav` per row, plus a `synth_record.json` provenance record) for the Seed-TTS-eval `en` and `zh` splits from `Qwen3-TTS-12Hz-1.7B-Base`, `Qwen3-TTS-12Hz-0.6B-Base` and `VoxCPM2`.
- Scoring manifests in a JSON/JSONL schema readable by `speech_p7/paired_bootstrap.py`.
- The measured-vs-published tables in Step 5, including acceptance run T0.

---

## The TTS techniques

| model | architecture / inference technique | languages | licence | what FoundationScale verified |
|---|---|---|---|---|
| `Qwen/Qwen3-TTS-12Hz-1.7B-Base` | inference-time zero-shot voice cloning, "voice clone per row", `generate_voice_clone(...)`, `language="auto"`, bfloat16, `max_new_tokens=2048`, other sampling parameters from the checkpoint's `generate_config.json` ("its README protocol") | en, zh | not stated | test-zh CER `0.959` (published `0.77`) reproduced; test-en WER `1.561` (published `1.24`) **not reproduced** (+`0.32`) |
| `Qwen/Qwen3-TTS-12Hz-0.6B-Base` | same technique and protocol as the 1.7B model, smaller checkpoint | en, zh | not stated | test-zh `1.132` (published `0.92`) reproduced; test-en `4.224` (published `1.32`) **not reproduced** (one runaway row, WER `27.18`) |
| `openbmb/VoxCPM2` | inference-time zero-shot "ultimate cloning": `generate(text=…, prompt_wav_path=…, prompt_text=…, reference_wav_path=…)` — prompt clip **and its transcript** are passed in; `cfg_value=2.0`, `inference_timesteps=10` | en, zh | not stated | test-en `1.706` (published `1.84`), test-zh `0.946` (published `0.97`), SIM `75.0` / `79.22` (published `75.3` / `79.5`) — reproduced |

How to pick:

- **Qwen3-TTS 1.7B Base** — clones a voice from a prompt wav alone, no reference transcript. Larger of the two Qwen3-TTS sizes, with the lower published Chinese score.
- **Qwen3-TTS 0.6B Base** — the same cloning interface in a smaller checkpoint; the cheap baseline against the 1.7B model.
- **VoxCPM2** — "ultimate cloning" conditions on the prompt clip **plus its transcript** (`prompt_text`); pick it when you have the prompt's text and want the lower published Seed-TTS numbers. Its published numbers come with no stated inference settings (see Troubleshooting).

Judges (scoring only, not synthesizers):

- **Whisper-large-v3** — ASR inference inside the upstream `run_wer.py`, the English WER judge. The published Seed-TTS protocol pins it.
- **`funasr/paraformer-zh`** — ASR inference inside `run_wer.py`, the Chinese CER judge, pinned the same way.
- **WavLM-large speaker verification** (`--model_name wavlm_large`, checkpoint `wavlm_large_finetune.pth`) — a speaker-verification checkpoint run at inference only through UniSpeech `verification_pair_list_v2.py`. The pinned SIM judge.
- **`Qwen3-ASR-1.7B`** — greedy ASR inference. Pinned as the **Vietnamese** judge because it scored a tighter floor on FLEURS-vi ground truth than Whisper (WER `7.49` vs `10.25`).

---

## Step 1 — Environments (one per upstream)

One environment per upstream, because each upstream pins its own stack and the stacks collide. All three run jobs in the container passed as `<env>`, via `enroot start -e PYTHONNOUSERSITE=1`, `-e CUDA_VISIBLE_DEVICES=$G`, `-e TORCH_HOME=<path>/.cache/torch_tts`, `-e HF_HOME=<path>/.cache/hf_tts_judge`, `-e MODELSCOPE_CACHE=<path>/.cache/modelscope_tts_judge` (judge runs) and `-e HF_HUB_OFFLINE=1` (synthesis runs). Each build script takes the target `<env>` directory as argument.

| env | build script | pinned versions (verbatim) | why separate |
|---|---|---|---|
| **tts-judge** | `validation_campaigns/tts_eval/cluster/build_tts_judge.sh` | `"torch==2.11.0" "torchaudio==2.11.0"` from `--index-url https://download.pytorch.org/whl/cu130`; constraint file `torch==2.11.0` / `torchaudio==2.11.0`; `"transformers==4.57.3" "jiwer==3.0.5" "fire==0.7.1" funasr modelscope zhconv zhon scipy soundfile librosa tqdm easydict joblib omegaconf`; `pip freeze > $V/requirements.lock`; SIM side: "s3prl is pinned in `TORCH_HOME/hub` (commit `ec8064b`) with hubconf narrowed to `wavlm_large`; see FS_PIN.txt there." | hosts the upstream Seed-TTS-eval judges (Whisper-large-v3 via transformers, Paraformer-zh via funasr, WavLM-large SV via UniSpeech + s3prl hub). `run_wer.py` imports `compute_measures`, removed in jiwer `4`, so jiwer stays at `3.0.5` — that alone forces a dedicated env. |
| **tts-qwen** | `validation_campaigns/tts_eval/cluster/build_tts_qwen.sh` | `"torch==2.11.0" "torchaudio==2.11.0"` (cu130 index, aarch64 wheels), `"qwen-tts==0.1.1"` | `qwen-tts` pins transformers `4.57.3`, so it cannot share an env with other stacks. The import check prints versions of `torch,torchaudio,transformers,accelerate,qwen-tts,onnxruntime`. |
| **tts-voxcpm** | `validation_campaigns/tts_eval/cluster/build_tts_voxcpm.sh` | `"torch==2.11.0" "torchaudio==2.11.0"` constraints file; `pip install -q -c $V/constraints.txt "voxcpm==2.0.3"` | `voxcpm==2.0.3` with its own dependency set (import check prints `torch,torchaudio,torchcodec,transformers,voxcpm,funasr`). |

Download helper: `validation_campaigns/tts_eval/cluster/fetch_t0.sh` takes `<env>` (a python env) and installs `gdown` with `python3 -m pip install --quiet --target $E/.gdown gdown` (gdown version not pinned).

Check it worked: each build script prints its import check lines (torch `2.11.0` in all three; `torch,torchaudio,transformers,accelerate,qwen-tts,onnxruntime` for tts-qwen; `torch,torchaudio,torchcodec,transformers,voxcpm,funasr` for tts-voxcpm).

---

## Step 2 — Data

```bash
bash validation_campaigns/tts_eval/cluster/fetch_t0.sh
```

Public repos only; no token. Writes sha256 manifests. It performs `snapshot_download` of `Qwen/Qwen3-TTS-12Hz-1.7B-Base`, `Qwen/Qwen3-TTS-12Hz-0.6B-Base`, `openbmb/VoxCPM2` and the companion artefacts `Qwen/Qwen3-TTS-Tokenizer-12Hz` and `Qwen/Qwen3-Omni-30B-A3B-Instruct`, then gdowns `seedtts_testset.zip` and `wavlm_large_finetune.pth`, and writes `sha256sum $E/gd/* > $E/gd/SHA256SUMS`. The two companion artefacts produce **no TTS result row** here.

| dataset | as stated | used for |
|---|---|---|
| Seed-TTS-eval test set (gdown id `1GlSjVfSHkW3-leKKBlfrjuuTGqQ_xaLP`) | splits as read: en `1,088`, zh `2,020`, zh-hard `400` (the README says `~1,000` / `~2,000`) | T0 ground truth and all T1 scoring |
| seed-tts-eval checkout, commit `752f429` | pinned by commit | the upstream scoring code passed as `--seed-code` |
| SV checkpoint `wavlm_large_finetune.pth` (gdown id `1-aE1NfzpRCLxA4GUxX9ITI3F9LlbtEGP`) | pinned by sha256 | SIM judge |
| FLEURS-vi test recordings | CC BY 4.0, `857` utterances | Vietnamese judge selection (T0b) |
| `MiniMaxAI/TTS-Multilingual-Test-Set` | CC BY-SA 4.0; `99` Vietnamese sentences, two reference voices; "has no ground-truth target audio" | Vietnamese synthesis set (**PLANNED** to be scored against the `7.49` floor) |

Meta list rows are `utt|prompt_text|prompt_wav|target[|gt]` (4 or 5 fields).

---

## Step 3 — Synthesize

Both synth scripts are resumable (already-existing output wavs are skipped), write `<filename>.wav` in `--out` plus a `synth_record.json`, and run under `HF_HUB_OFFLINE=1`.

### `synth_qwen3tts.py` — Qwen3-TTS voice cloning

Synthesizes Seed-TTS-eval wavs with Qwen3-TTS voice cloning using `generate_voice_clone`, bfloat16, `max_new_tokens=2048`, every other sampling parameter from the checkpoint's `generate_config.json`, and `language="auto"` (Qwen's published protocol). Runs in the tts-qwen environment.

Flags:

| flag | meaning | default / required |
|---|---|---|
| `--model` | model path | required |
| `--testset-dir` | Seed-TTS-eval test-set directory | required |
| `--lang` | language split, choices `en` or `zh` | required |
| `--list` | meta list file name | `meta.lst` |
| `--out` | output directory for wavs | required |
| `--batch-size` | synthesis batch size | `8` |
| `--max-new-tokens` | maximum new tokens | `2048` |
| `--seed` | random seed | `0` |
| `--limit` | cap number of rows | `None` |

Example:

```bash
<path>/envs/tts-qwen/bin/python validation_campaigns/tts_eval/synth_qwen3tts.py --model <path>/Speech-Models/TTS/Qwen3-TTS-12Hz-1.7B-Base --testset-dir <path>/seedtts_testset --lang zh --list meta.lst --out <path>/synth/qwen_1.7b_zh
```

Cluster wrapper (resolves the model directory to `<weights>/<MODEL_DIRNAME>` and runs `synth_qwen3tts.py` in the tts-qwen environment):

```
bash validation_campaigns/tts_eval/cluster/run_synth_qwen.sh <GPU> <MODEL_DIRNAME> <LANG> <LIST> <TAG>
```

### `synth_voxcpm2.py` — VoxCPM2 "ultimate cloning"

Synthesizes Seed-TTS-eval wavs with VoxCPM2 "ultimate cloning" using `generate(text=…, prompt_wav_path=…, prompt_text=…, reference_wav_path=…)` with `cfg_value=2.0` and `inference_timesteps=10`; the prompt clip is also passed as `reference_wav_path`.

Flags:

| flag | meaning | default / required |
|---|---|---|
| `--model` | model path | required |
| `--testset-dir` | Seed-TTS-eval test-set directory | required |
| `--lang` | language split | required |
| `--list` | meta list file name | `meta.lst` |
| `--out` | output directory for wavs | required |
| `--cfg-value` | classifier-free guidance value | `2.0` |
| `--inference-timesteps` | inference timesteps | `10` |
| `--seed` | random seed | `0` |
| `--lora-ckpt` | VoxCPM LoRA checkpoint dir (T2b) | `None` |
| `--limit` | cap number of rows | `None` |

Example:

```bash
<path>/envs/tts-voxcpm/bin/python validation_campaigns/tts_eval/synth_voxcpm2.py --model <path>/Speech-Models/TTS/VoxCPM2 --testset-dir <path>/seedtts_testset --lang en --list meta.lst --out <path>/synth/voxcpm2_en
```

Cluster wrapper (resolves the VoxCPM2 model directory from `<weights>` and runs `synth_voxcpm2.py` in the tts-voxcpm environment):

```bash
bash validation_campaigns/tts_eval/cluster/run_synth_voxcpm.sh <GPU> <LANG> <LIST> <TAG>
```

### Full chain (both splits)

```bash
bash validation_campaigns/tts_eval/cluster/run_t1_chain.sh <GPU_EN> <GPU_ZH>
```

It runs the per-language lane for `en` and `zh` as two backgrounded jobs across the tts-judge, tts-qwen and tts-voxcpm environments (GT, then `Qwen3-TTS-12Hz-1.7B-Base` + `Qwen3-TTS-12Hz-0.6B-Base` synth+score, then `voxcpm2_$l` synth+score).

### GPU idle gate

Every runner refuses with `REFUSE gpu $G busy $U` and exit `96` when the target GPU holds at least `1000` MB.

---

## Step 4 — Score

### What WER/CER and SIM mean

- **WER (en) / CER (zh)** — the error rate of the recognized transcript of the synthesized wav against the reference text, computed by the upstream `run_wer.py` (Whisper-large-v3 for `en`, `funasr/paraformer-zh` for `zh`). `en` is word-level (words = tokens, truth lowercased first), `zh` is character-level (space-joined per char). Corpus WER mirrors `average_wer.py`: the unweighted mean of per-utterance WER values, × `100`. Lower is better.
- **SIM** — the speaker-verification score between two wavs (pairs `"<wav1>|<wav2>"`: wav1 = synth wav, wav2 = prompt wav) from WavLM-large SV through UniSpeech `verification_pair_list_v2.py`. SER = mean of per-pair scores × `100`, matching `average.py`. Higher is better.

### `seedtts_score.py` — Seed-TTS-eval scorer

Scores Seed-TTS wavs (synth or ground truth) and emits a WER/SIM JSON manifest by running the upstream scripts **unmodified, as subprocesses** (seed-tts-eval commit `752f429`). FoundationScale owns only list building, coverage (missing audio is counted and exits `95`, never dropped), refusals (`96`) and the per-row output. Upstream invocations: WER via `run_wer.py`, SIM via `verification_pair_list_v2.py`.

Flags:

| flag | meaning | default / required |
|---|---|---|
| `--testset-dir` | Seed-TTS-eval test-set directory | required |
| `--lang` | language split, choices `zh` or `en` | required |
| `--list` | list file, relative to `<testset-dir>/<lang>` | required |
| `--synth-dir` | directory of synthesized wavs | provide exactly one of `--synth-dir` or `--gt` |
| `--gt` | use `<testset-dir>/<lang>/wavs` as ground truth | provide exactly one of `--synth-dir` or `--gt` |
| `--seed-code` | root of a seed-tts-eval checkout | required |
| `--wavlm-ckpt` | WavLM speaker-verification checkpoint | required |
| `--python` | python executable for upstream calls | `sys.executable` |
| `--device` | CUDA device | `cuda:0` |
| `--limit` | cap number of rows | `None` |
| `--workdir` | working directory for upstream artefacts | optional |
| `--out` | output JSON manifest | required |
| `--no-sim` | skip SIM scoring | `False` |

Example:

```bash
<path>/envs/tts-judge/bin/python validation_campaigns/tts_eval/seedtts_score.py --testset-dir <path>/seedtts_testset --lang zh --list meta.lst --synth-dir <path>/synth/voxcpm2_zh --seed-code <path>/seed-tts-eval-code --wavlm-ckpt <path>/wavlm_large_finetune.pth --device cuda:0 --out <path>/runs/score_voxcpm2_zh.json
```

Cluster wrappers (`run_gt.sh` calls `seedtts_score.py` with `--gt` in the tts-judge environment; `run_score.sh` calls it with `--synth-dir` pointing at the matching synth output, also in the tts-judge environment):

```bash
bash validation_campaigns/tts_eval/cluster/run_gt.sh <GPU> <LANG> <LIST> <OUTTAG>
bash validation_campaigns/tts_eval/cluster/run_score.sh <GPU> <LANG> <LIST> <SYNTH_TAG>
```

### `judge_multilingual.py` — languages `run_wer.py` does not cover

Scores speech for languages the upstream Seed-TTS scorer does not cover, reading a JSONL manifest of rows with string `id`, `audio`, `text` and optional string `prompt_audio` (enables SIM). Malformed rows are refused with exit `96`, missing audio counted with exit `95`. It reuses `seedtts_score.py` helpers `_run_sim` and `_run` and refuses at runtime with a clear `NotImplementedError` if they are absent.

Flags:

| flag | meaning | default / required |
|---|---|---|
| `--manifest` | JSONL manifest path | required |
| `--asr` | ASR backend, choices `whisper` or `qwen3asr` | required |
| `--language` | language code, e.g. `"vi"` for Whisper/Qwen3-ASR | required |
| `--whisper` | local path to Whisper-large-v3 | required for `--asr whisper` |
| `--qwen3asr` | local path to Qwen3-ASR | required for `--asr qwen3asr` |
| `--wavlm-ckpt` | WavLM speaker-verification checkpoint | required when rows carry `prompt_audio` and `--no-sim` is not given |
| `--seed-code` | root of a seed-tts-eval checkout | required when rows carry `prompt_audio` and `--no-sim` is not given |
| `--python` | python executable | `sys.executable` |
| `--device` | CUDA device | `cuda:0` |
| `--batch-size` | scoring batch size | `8` |
| `--limit` | cap number of rows | `None` |
| `--out` | output JSON manifest | required |
| `--no-sim` | skip SIM scoring | `False` |

Audio is decoded to float32 at `16` kHz, Whisper's multilingual `BasicTextNormalizer` (keeps diacritics) is applied on both sides, and the word-level edit distance goes through `jiwer.compute_measures`. Whisper path: greedy, `language` forced, task transcribe; clips over `30` s use long-form mode (`return_timestamps=True`, `do_sample=False`). Qwen3-ASR path: `apply_transcription_request(audio=waves, language=language)`, `model.generate(**inputs, max_new_tokens=1024, do_sample=False)`, per-row `processor.decode(..., return_format="transcription_only")`. Corpus WER here is **pooled** (sum(errors) / sum(reference_length) × `100`).

Example:

```bash
<path>/envs/tts-judge/bin/python validation_campaigns/tts_eval/judge_multilingual.py --manifest <path>/runs/vi_manifest.jsonl --asr qwen3asr --language vi --qwen3asr <path>/models/Qwen3-ASR-1.7B --wavlm-ckpt <path>/wavlm_large_finetune.pth --seed-code <path>/seed-tts-eval-code --out <path>/runs/judge_vi.json
```

---

## Step 5 — Measured results vs published targets (VERIFIED)

Acceptance criterion, tolerance fixed in advance: ground truth scored with `--gt` must match the published "Human" row within `±0.3` WER/CER and `±1.0` SIM (reference: CosyVoice README, Seed-TTS-eval "Human").

### T0 — judges reproduce the published "Human" row (2026-10-12)

| split | rows scored | WER / CER (ours) | published | SIM (ours) | published | verdict |
|---|---|---|---|---|---|---|
| test-en | `1,088` / `1,088` | `2.143` | `2.14` | `73.38` | `73.4` | "all four numbers match the published values to rounding" |
| test-zh | `2,020` / `2,020` | `1.254` | `1.26` | `75.50` | `75.5` | as above |

Every row scored, none missing. The harness is accepted for T1.

### T1 — synthesizers vs published Seed-TTS numbers (decode-only, 2026-10-12)

| model | split | ours WER/CER | published | SIM (ours / published) | verdict |
|---|---|---|---|---|---|
| VoxCPM2 | test-en | `1.706` | `1.84` | `75.0` / `75.3` | reproduced |
| VoxCPM2 | test-zh | `0.946` | `0.97` | `79.22` / `79.5` | reproduced |
| Qwen3-TTS-12Hz-1.7B-Base | test-zh | `0.959` | `0.77` | `76.91` / `-` (not stated) | reproduced |
| Qwen3-TTS-12Hz-0.6B-Base | test-zh | `1.132` | `0.92` | `76.33` / `-` (not stated) | reproduced |
| Qwen3-TTS-12Hz-1.7B-Base | test-en | `1.561` | `1.24` | `71.65` / `-` (not stated) | **not reproduced (+`0.32`)** |
| Qwen3-TTS-12Hz-0.6B-Base | test-en | `4.224` | `1.32` | `70.3` / `-` (not stated) | **not reproduced** |
| VoxCPM2 | test-zh-hard | `7.826` | `8.13` | `75.11` / `75.3` | reproduced |
| Qwen3-TTS-12Hz-1.7B-Base | test-zh-hard | `5.914` | `-` | `74.65` / `-` | measured (no published value) |

### Follow-ups on the two English misses (VERIFIED)

| measurement | value(s) |
|---|---|
| Apostrophe folding (curly→ASCII), Qwen 1.7B en WER | `1.561` → `1.441` |
| Same folding, ground-truth (test-en) WER | `2.143` → `2.069` |
| Qwen 0.6B en runaway row | row WER `27.18`; adds `~2.5` points, `~1.7` without it |
| Qwen3-TTS-1.7B test-en variance (2026-10-14) | seed `0` batch `8` = `1.561`; seed `1` batch `8` = `1.446`; seed `0` unbatched = `1.691` (spread `±0.12`) |
| Distance to the card's `1.24` | `0.2`–`0.45` above `1.24`, about `0.12` of which is the apostrophe effect |

### T0b — Vietnamese judge selection on FLEURS-vi ground truth (2026-10-13)

| judge | rows scored | WER on ground truth |
|---|---|---|
| **Qwen3-ASR-1.7B** (language forced to vi) | `857` / `857` | **`7.49`** (pinned as the Vietnamese judge and the floor) |
| Whisper-large-v3 (vi, long-form timestamps) | `857` / `857` | `10.25` (not selected) |

**Not measured in this campaign:** MiniMax Vietnamese synthesis scored against the `7.49` floor.

---

## Step 6 — Fine-tune: Vietnamese speaker adaptation with LoRA (VoxCPM2)

Qwen3-TTS and Qwen3-Omni cannot speak Vietnamese, but VoxCPM2 (Apache-2.0) can, and its official fine-tuning recipe keeps zero-shot cloning. This step adapts VoxCPM2 to Vietnamese speech with LoRA so it reads Vietnamese text accurately with a cloned voice, and measures how much intelligibility was gained and how much speaker similarity was lost.

### Data

All training data is CC BY 4.0, decoded to 16 kHz mono WAV (VoxCPM2's AudioVAE input rate):

- FLEURS-vi train+dev: 10.0 h, cased and punctuated text.
- InfoRe-25h: 24.8 h, one paid reader.
- FPT-FOSD: 30.1 h (its dataset card says 100 h).

In total 64.9 h and 43,907 train rows plus a 300-row dev slice. There are no empty transcripts and no duplicate audio, and the epoch fraction in the trainer log implies all 43,907 rows were used (no silent drops by `max_batch_tokens`). InfoRe and FOSD transcripts are lowercase without punctuation, which is a prosody risk on punctuated input. FLEURS-vi test is never trained on — it is the eval set.

Evaluation sets: MiniMax-vi (CC BY-SA 4.0), 100 sentences, two Common Voice voices (`build_minimax_vi.py`); FLEURS-vi cloning set (`build_fleurs_vi_tts.py`), the 857 test sentences, each cloned from a different test utterance of the same gender (537 distinct prompts). Both are scored by `tts_eval/judge_multilingual.py` (Qwen3-ASR-1.7B judge, WER; WavLM SIM against the prompt clip).

### Commands

Build the Vietnamese training manifest (CPU):

```bash
<path>/envs/tts-voxcpm/bin/python validation_campaigns/tts_vi/cluster/build_voxcpm_vi.py
```

Generate the LoRA fine-tune config by copying the upstream VoxCPM 2.0.3 `conf/voxcpm_v2/voxcpm_finetune_lora.yaml` and filling only paths and steps:

```bash
bash validation_campaigns/tts_vi/cluster/make_t2b_config.sh
```

Train on one GPU using the upstream `scripts/train_voxcpm_finetune.py` under `torch.distributed.run` in the tts-voxcpm venv:

```bash
bash validation_campaigns/tts_vi/cluster/run_t2b.sh <GPU> [t2b_lora]
```

Synthesize one eval set with the LoRA checkpoint (or base) and judge WER + SIM:

```bash
VISET=<minimax_vi|fleurs_vi_tts> LORA=<ckpt dir> bash validation_campaigns/tts_vi/cluster/run_vi_eval.sh <GPU> voxcpm2 <tag>
```

### What the training does

LoRA, not full fine-tuning: the corpus transcripts (InfoRe, FOSD) are lowercase without punctuation, so freezing the base weights keeps its punctuation and prosody handling while the adapters learn the target language/reader acoustics. The config is the upstream `conf/voxcpm_v2/voxcpm_finetune_lora.yaml` at VoxCPM tag 2.0.3 with only paths and steps filled in by `make_t2b_config.sh`: LoRA r32/alpha 32 on LM and DiT, lr 1e-4, effective batch 16, 2,000 steps (0.73 epoch). One GB200 GPU at about 10 s/step, so roughly 5.5 h for 2,000 steps.

### Check it worked

| checkpoint | MiniMax-vi WER / SIM | FLEURS-vi WER / SIM |
|---|---|---|
| base VoxCPM2 | 5.75 / 79.50 | 8.59 / 82.56 |
| step 2000 (reported) | 1.44 / 77.01 | 6.79 / 81.49 |

Paired bootstrap, base vs step 2000:

- FLEURS-vi WER -1.80 points, 95% CI [-3.39, -0.58], n=857; MiniMax-vi WER -4.31 [-8.59, -1.10].
- FLEURS-vi SIM -1.07 points, 95% CI [-1.47, -0.68].

LoRA on 65 h of Vietnamese makes VoxCPM2 significantly more intelligible at a small but significant cost in speaker similarity, consistent with the training mix (one dominant reader, unpunctuated text). For reference, the same judge reads real FLEURS-vi recordings at 7.49 WER: the fine-tuned output is at the judge's floor on read speech.

Reporting rules: MiniMax-vi is small (100 rows) and swings about 1 point from sampling alone, so the 857-row FLEURS-vi set is the anchor and the paired intervals carry the claim. The reported checkpoint is the final one (step 2000), fixed in advance; intermediate checkpoints can be inspected but not reported as the result.

### Full fine-tuning vs LoRA (VERIFIED, 2026-10-15)

Same data, steps and effective batch, but all weights except the frozen AudioVAE at lr 1e-5 (upstream `voxcpm_finetune_all.yaml`): build the config with `MODE=full bash validation_campaigns/tts_vi/cluster/make_t2b_config.sh`, then train and evaluate exactly as above. Final step 2000, fixed in advance.

| run | FLEURS-vi WER / SIM | MiniMax-vi WER / SIM |
|---|---|---|
| base | 8.59 / 82.56 | 5.75 / 79.50 |
| LoRA step 2000 | 6.79 / 81.49 | 1.44 / 77.01 |
| full FT step 2000 | 6.895 / 81.79 | 1.303 / 73.93 |

LoRA vs full on FLEURS-vi (n=857): WER +0.10 [-0.32, +0.53], SIM +0.30 [-0.07, +0.70] — no significant difference in intelligibility. Full fine-tuning loses more speaker similarity on the two-voice MiniMax set (73.93 vs 77.01). **LoRA stays the recipe:** equal quality, a small adapter instead of a full checkpoint to ship, and cloning similarity stays closer to the base model.

### Qwen3-TTS single-speaker fine-tuning: null result (VERIFIED)

Upstream `finetuning/sft_12hz.py` (qwen-tts 0.1.1) trains one fixed new voice per run; it does not support multi-speaker or cloning fine-tuning. On 28 minutes of one LibriTTS-R speaker (398 utterances, 60 held out; `validation_campaigns/tts_train/`):

| run (README recipe: batch 32, 10 epochs) | held-out WER | SIM |
|---|---|---|
| base 1.7B, zero-shot clone | 2.309 | 44.75 |
| SFT lr 2e-6 (README value) | 2.165 | 44.47 |
| SFT lr 2e-5 (upstream script default) | 217.17 | 4.49 |

At the README learning rate the fine-tuned voice is no better than zero-shot cloning (WER [-0.50, 0.00], SIM [-2.5, +1.8], n=60); at 10x it diverges into runaway generation. For a new language, use the VoxCPM2 LoRA path above; for a new voice, read on.

### Speaker adaptation with LoRA: VoxCPM2 on the same speaker (VERIFIED, 2026-10-15)

The same 28-minute LibriTTS-R speaker (60 held-out utterances), trained as a VoxCPM2 LoRA adapter with the Step 6 tooling (`validation_campaigns/tts_train/`):

| | held-out SIM | held-out WER |
|---|---|---|
| base VoxCPM2, zero-shot clone | 44.48 | 2.597 |
| speaker LoRA | 47.43 | 2.886 |

Speaker similarity rises significantly: paired +2.95 [+1.44, +4.49]. The WER change is unresolved at n=60 (CI [-0.50, +1.47]). With the adapter loaded, zero-shot cloning of other voices does not regress in intelligibility: Seed-TTS test-en WER 1.624 vs 1.706 (paired [-0.27, +0.10]), with a small similarity cost (SIM 74.22 vs 75.0, paired -0.78 [-1.03, -0.53]). Use it as an optional per-speaker adapter.

### Not measured yet

- Naturalness (UTMOS): its licence is unrecorded, so there is no naturalness score.

---

## Troubleshooting

- **Judge jobs hang or die on imports.** Three compatibility fixes are needed to run 2023-era judge code, none touching how a score is computed: jiwer pinned to `3.0.5` (`run_wer.py` imports `compute_measures`, removed in jiwer `4`); s3prl pinned at commit `ec8064b` in the torch hub cache and marked trusted (an untrusted hub repo waits on an interactive prompt); that pinned s3prl's top-level `hubconf.py` narrowed to its own `wavlm_large` function, because the full hub imports `~60` upstreams and some call torchaudio APIs removed in `2.9` (the SIM path uses only `wavlm_large`).
- **Curly apostrophes inflate WER.** Upstream `run_wer.py` strips curly quotes (zhon's punctuation set) but keeps the ASCII apostrophe, so "aren't" against "aren't" is an error. Folding curly to ASCII changes WER (see the follow-ups table); the official comparison keeps the unmodified scorer.
- **One huge row skews a corpus WER.** The Qwen 0.6B en split contains one runaway repetition loop row; row WER `27.18` adds `~2.5` points (`~1.7` without it). This is exactly what FoundationScale's runaway / repetition-loop gates catch on the ASR side.
- **Qwen3-TTS English does not reach the card number.** The spread (`±0.12`) is sampling noise; all three runs sit `0.2`–`0.45` above the card's `1.24`, about `0.12` of that the apostrophe effect. The remaining gap is real: "our harness reads Qwen3-TTS English about `0.3` above the card".
- **A VoxCPM2 gap to its published numbers is settings first.** VoxCPM2's README publishes Seed-TTS results (en `1.84` WER / `75.3` SIM, zh `0.97` / `79.5`) but not the inference settings behind them, so `synth_record.json` records `"published_settings": "UNVERIFIED (README states none for its Seed-TTS table)"`.
- **Averaging differs between scorers.** `judge_multilingual.py` reports pooled WER (sum over rows), while Seed-TTS averages per-row values in `average_wer.py`. Per-row numbers stay in a schema compatible with `speech_p7/paired_bootstrap.py` for pairwise comparisons; do not compare the two families of corpus numbers directly.
- **`run_wer.py` covers en and zh only**, and the MiniMax Vietnamese set has no ground-truth target audio, so Vietnamese is measured against a proxy floor (`7.49`) rather than a published target.
- **Exit codes are deliberate.** Missing audio is counted and exits `95`, never dropped. `96` means refuse: bad meta rows (not `4` or `5` `|` fields), giving both or neither of `--synth-dir` and `--gt`, an upstream command failure ("we abort rather than invent numbers"), or a busy GPU (`REFUSE gpu $G busy $U` when the GPU holds ≥ `1000` MB).
- **Error counts in the WER manifest are derived, not read directly.** `run_wer.py` TSV error columns are rates; raw edit counts are backed out as `rate * reference_length` with a fallback `ref_len = round(errors_rate_sum / wer_val)`; for `zh`, `run_wer.py` counts spaces in `ref_list = normalised truth.split(" ")` and that same length is used for count recovery when available.
- **Qwen3-ASR transcripts get an echo prefix.** `batch_decode` ignores the return format and echoes `language English<asr_text>...` verbatim — use per-row `processor.decode(..., return_format="transcription_only")`, which is what `judge_multilingual.py` does.
