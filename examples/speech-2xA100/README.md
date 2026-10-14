# Run the 2x H200 speech tutorial on 2x A100-80GB (no script changes)

The tutorial in [`../speech-2xH200/`](../speech-2xH200/) runs **unchanged** on two A100-80GB GPUs: same scripts,
same configs, same commands. This file is only a portability note: how to pin two free GPUs, what was
**MEASURED** (2026-10-12, 2x A100-SXM4-80GB) versus the 2x H200 reference numbers in
the H200 README, and what does not fit in 80 GB yet. All numbers below are **MEASURED (2026-10-12, 2x A100-SXM4-80GB)**
unless marked **UNMEASURED** or labelled as the H200 reference. Chapter details — installation, data,
training recipes, evaluation semantics — live in
[01_installation.md](../speech-2xH200/01_installation.md),
[02_data_preparation.md](../speech-2xH200/02_data_preparation.md),
[03_training.md](../speech-2xH200/03_training.md), [04_evaluation.md](../speech-2xH200/04_evaluation.md);
we link, not repeat.

## Environment (delta versus the H200 VM)

- Baseline stack: Python 3.12, torch 2.13.0+cu129, transformers 5.18.0, peft 0.21.2, accelerate 1.15.0,
  datasets 5.1.0, liger-kernel 0.8.4, av 19.0.1, librosa 1.0.0, scipy 1.18.1. cu129 instead of the H200's
  cu132: both run on driver >= 580 (ours: 580.95.05 / CUDA 13.0); on A100 (sm80) there is no FP8/FA3 path
  anyway — the example uses sdpa.
- Install = [`../vlm-2xH200/install/requirements-vlm.txt`](../vlm-2xH200/install/requirements-vlm.txt)
  **without its three torch lines** (`--extra-index-url` cu132, torch, torchvision), with a constraints file
  pinning the torch/torchvision already installed, then `pip install -e ".[train]" librosa "scipy>=1.16"`.
  Chapter [01](../speech-2xH200/01_installation.md) covers the rest.
- If your index is slow (we saw ~20–200 KB/s to PyPI/PyTorch): `pip install --dry-run --report report.json
  -c constraints.txt -r ...` on the node (metadata only), fetch the listed URLs on any fast machine, then
  `pip install --no-index --find-links WHEELS --no-build-isolation ...`. 21 wheels, 152 MB.
- Starting from a pre-existing env: `pip uninstall` stale `deepspeed` (0.14.0 breaks transformers' Trainer
  import on torch 2.13 -> FS REFUSE 96) and `bitsandbytes` (imports `triton.ops`, breaks peft). Then
  `pip check` and import peft, transformers.trainer, liger_kernel.
- Put `PIP_CACHE_DIR`, `HF_HOME`, `TMPDIR` on a big filesystem ($FS_ROOT); export
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
- MFU is off unless you opt in (FS has no default device peak):
  `export FS_DEVICE_PEAK_TFLOPS=312 FS_DEVICE_PEAK_SOURCE="NVIDIA A100-SXM4-80GB datasheet, dense bf16"`.
  Sanity, warm bf16 8192^3 matmul on one A100: 257 TFLOPS (82% of the 312 peak).

## Running it: GPUS=2 + two pinned GPUs, nothing else changes

Shared-node hygiene (see [../A100_2GPU.md](../A100_2GPU.md)): pick two free GPU ids, pin them, and abort if
either holds memory (our wrapper aborts above 500 MiB used).

```bash
nvidia-smi --query-compute-apps=pid,used_memory --format=csv
export CUDA_VISIBLE_DEVICES=<two free gpu ids>
GPUS=2 bash examples/speech-2xH200/scripts/train_speech.sh examples/speech-2xH200/configs/parakeet-ctc-1.1b.env
GPUS=2 bash examples/speech-2xH200/scripts/train_speech.sh examples/speech-2xH200/configs/whisper-large-v3.env
```

Same config files, same 500-step recipe: full train-clean-100 (28,539 rows / 100.59 h, 0 skipped) and full
dev-clean (2,683 of 2,703 rows / 5.22 h, 20 `too_long`) — identical row counts to the H200 tutorial
(**MEASURED** again here). Profile `local-single-node` (scheduler `local`) works on any hostname; run from
inside the git checkout so the manifest is attributable. **MEASURED**: EXIT 0, `[fs:train:done]` PASS for both.

## Measured results: A100 versus the H200 2-GPU reference

Peak = highest 5 s `nvidia-smi` sample (same semantics as
[the H200 chapter](../speech-2xH200/03_training.md)); WER/CI base-vs-tuned on identical held-out rows.

| model | exit | peak GB/GPU | s/step | wall 500 steps | train_loss | WER base -> tuned | 95% CI (pts) |
|---|---|---|---|---|---|---|---|
| parakeet-ctc-1.1b full FT | 0 | 13.6 | 0.415 | 279 s | 0.4169 | 1.89% -> 1.78% | [-0.16, -0.06] |
| whisper-large-v3 full FT, batch 4/GPU, sentence-cased | 0 | 26.1 | 0.545 | 329 s | 0.1225 | 2.25% -> 1.72% | [-0.65, -0.41] |

H200 2-GPU reference ([../speech-2xH200/](../speech-2xH200/)): parakeet 14.3 GB, 0.44 s/step, 245 s,
WER 1.87% -> 1.77% [-0.14, -0.05]; whisper 27.6 GB, 0.41 s/step, 230 s, WER 2.25% -> 1.72% [-0.65, -0.41]
(the H200 README's whisper WER row is the 1-GPU batch-8 run). Per step, parakeet is slightly faster on A100 (0.415 vs 0.44 s) and whisper is slower (0.545 vs 0.41 s);
wall time is longer for both (startup and the final checkpoint write). Peaks are slightly lower on A100
(13.6 vs 14.3 GB, 26.1 vs 27.6 GB). WER lands on the same answer: parakeet within 0.02 pts, whisper exactly.
Both fine-tunes improved — but as the H200 README says, a 500-step run is a plumbing check, not a quality claim.

## Evaluation: one GPU each, base in parallel with tuned

```bash
M=parakeet-ctc-1.1b   # or whisper-large-v3
CUDA_VISIBLE_DEVICES=<gpu a> python validation_campaigns/speech_p4/eval_wer.py --model models/$M \
    --manifest data/librispeech/manifests/eval.jsonl --out eval_logs/${M}_base.json &
CUDA_VISIBLE_DEVICES=<gpu b> python validation_campaigns/speech_p4/eval_wer.py --model runs/$M/final \
    --processor models/$M --manifest data/librispeech/manifests/eval.jsonl --out eval_logs/${M}_tuned.json &
wait
python validation_campaigns/speech_p7/paired_bootstrap.py eval_logs/${M}_base.json eval_logs/${M}_tuned.json
```

See [04_evaluation.md](../speech-2xH200/04_evaluation.md) for the exact invocations. **MEASURED**: each
`eval_wer.py` pass takes ~58 s (parakeet) or ~305 s (whisper); `paired_bootstrap.py` over n=2683 returns the
CI rows in the table above. Running base on one GPU and tuned on the other keeps the wall time at one pass.

## What fits in 80 GB — and what does not yet

| model | recipe | A100-80GB status | H200 2-GPU peak (reference) |
|---|---|---|---|
| parakeet-ctc-1.1b full FT | as-is | fits (13.6 GB) | 14.3 GB |
| whisper-large-v3 full FT | batch 4/GPU | fits (26.1 GB) | 27.6 GB |
| gemma-4-E4B-it LoRA | as-is | **UNMEASURED** — does not fit at the H200 batch size | 86.2 GB |
| qwen2-audio-7b LoRA | as-is | **UNMEASURED** — does not fit at the H200 batch size | 82.0 GB |
| qwen2-audio-7b full FT | FSDP, `full` state dict | **UNMEASURED** — does not fit at the H200 batch size | 92.9 GB |

All three unmeasured rows exceed 80 GB at the H200 global-batch sizes, so to try them you must lower `BATCH`
and compensate with `GRAD_ACCUM` (the configs set tunables with `: "${BATCH:=...}"`, so a shell
`BATCH`/`LR`/`WARMUP`/`STEPS`/`GRAD_ACCUM` wins; `MODEL`/`DATASET`/`OUTPUT_DIR` come from the config):
`STEPS=500 BATCH=... GRAD_ACCUM=... GPUS=2 bash examples/speech-2xH200/scripts/train_speech.sh
examples/speech-2xH200/configs/gemma-4-E4B-it.env`. No A100 measurement exists for any of them — treat the
numbers tagged **UNMEASURED** as unverified.

## Check it worked

- Data: train-clean-100 manifest shows 28,539 rows / 100.59 h, 0 skipped; dev-clean 2,683 rows kept of 2,703,
  20 `too_long` — identical to [chapter 02](../speech-2xH200/02_data_preparation.md).
- Both training runs printed `[fs:train:done]` with a PASS verdict and exit 0; checkpoint save gates all PASS.
- `nvidia-smi --query-gpu=memory.used` during the runs stayed under the table's peaks (13.6 / 26.1 GB per GPU).
- Eval: base and tuned passes used the two pinned GPUs concurrently (each ~58 s for parakeet, ~305 s for whisper),
  on identical 2,683 held-out rows.
- `paired_bootstrap.py` printed a CI of [-0.16, -0.06] (parakeet) and [-0.65, -0.41] (whisper, n=2683) —
  both exclude zero, both significant on this hardware.

See also: [../speech-2xH200/](../speech-2xH200/) (the main tutorial), [../vlm-2xH200/](../vlm-2xH200/)
(shared install/eval scripts), [../A100_2GPU.md](../A100_2GPU.md) (A100 portability notes)
and `examples/train_tiny.py` for a 2-GPU DDP smoke test (PASS: 9 save gates,
`checkpoint.save_complete` 28/28, NCCL all-reduce probe 1.3 s — **MEASURED**).
