# NVFP4 (4-bit) training on Blackwell GPUs, with a bf16 control

One end-to-end recipe for training a model with **NVFP4**, NVIDIA's 4-bit floating-point format,
and **proving** that the run really used it. You fine-tune the same model twice, in bf16 and in
NVFP4, from the same checkpoint, data, seed and batch, and a verifier reads both logs and tells you
whether FP4 was active and what it cost.

The model is **`Qwen/Qwen2.5-7B-Instruct`**, the data is SQuAD (Megatron-Bridge's built-in `squad`
preset, downloaded on first use), and training runs through **Megatron-Bridge**, NVIDIA's training
library for Megatron-Core. FoundationScale's own trainer (`python -m foundationscale.train`) does
**not** train in NVFP4 yet: `--precision nvfp4` is declarable but refused with exit 96, because it
has no NVFP4 backend and must never train in bf16 while claiming NVFP4. Sibling tutorials:
[LLM LoRA](../llm-2xA100/), [VLM on GB200](../vlm-2xGB200/).

Every number below is **MEASURED** on GB200 with the container named in Prerequisites. Read them for
what they are: short runs that prove the pipeline and show the cost, not a quality claim.

## What NVFP4 is, and what it is not

* **What it changes.** The matrix multiplies (GEMMs) inside Linear layers run in 4-bit
  floating point (E2M1) with a scale per block of 16 values. Megatron-Bridge's recipe
  `bf16_with_nvfp4_mixed` turns this on (`fp4: e2m1`, `fp4_recipe: nvfp4`); everything else
  stays bf16.
* **What it does not change.** The fp32 master weights, gradients and optimizer state stay in
  higher precision. Checkpoints are the **same size** as bf16 ones (measured: 101,682 MiB for both
  arms of the smoke run). NVFP4 is a compute precision, not a storage format.
* **One setting decides whether fine-tuning works: `fp4_param`.** Megatron-Bridge's stock recipe
  sets `fp4_param: true`, which also keeps the model's working copy of the weights in FP4 between
  steps (NVIDIA's Nemotron-3 NVFP4 pretraining recipes do the same). At a fine-tuning learning rate
  most weight updates are smaller than one FP4 rounding step, and the model barely learned (table
  below). With `fp4_param: false` the weights stay bf16, the GEMMs still run in FP4, and the loss
  tracked bf16. This example defaults to `FP4_PARAM=false`; set `FP4_PARAM=true` in the config to
  get the stock recipe.
* **Hardware.** NVFP4 GEMMs exist only on **Blackwell** GPUs (compute capability 10.x: B200, GB200,
  B300). `train_nvfp4.sh` refuses the NVFP4 arm on anything else (exit 96) instead of crashing
  inside TransformerEngine.
* **The failure to guard against is silent.** A run whose FP4 setting never took effect trains in
  bf16 and looks perfectly healthy. Step time cannot tell you: it is noisy, and NVFP4 is not
  always faster. That is why this example always runs a bf16 control and checks the logs.

## What you get: measured results

**The example config, 2 GPUs (MEASURED, 2026-10-16, 2x GB200).** TP=2, global batch 32, 50 steps,
sequence length 2048, SQuAD, lr 5e-6, all three arms on the same two GPUs. `verify_nvfp4.py`
exits 0 (all five checks PASS) for each NVFP4 arm against the bf16 control:

| arm | median s/step | loss @ 1 | @ 10 | @ 20 | @ 30 | @ 50 | grad norm @ 50 |
|---|---|---|---|---|---|---|---|
| bf16 control | 3.72 | 4.19 | 1.20 | 0.31 | 0.24 | 0.38 | 18.3 |
| **nvfp4, `FP4_PARAM=false` (default)** | 5.00 (**0.74x**) | 3.71 | 1.08 | 0.32 | 0.23 | **0.41** | 17.7 |
| nvfp4, `FP4_PARAM=true` (stock recipe) | 5.17 (0.72x) | 3.71 | 3.07 | 3.86 | 2.80 | **2.76** | 178.7 |

The default NVFP4 arm differs from the bf16 loss by 0.08 on average over the 50 steps (5.8% higher
at step 50). Its run is deterministic: rerunning it reproduced the step-50 loss to the last
printed digit. The stock recipe never got going: its grad norm stayed at 150-180 while the other
two fell to about 15.

**Smoke run, 1 GPU (MEASURED, 2026-10-16, 1x GB200).** TP=1, global batch 8, 10 steps, sequence
length 2048, both arms on the same GPU, `verify_nvfp4.py` exit 0 (all five checks PASS):

| arm | median s/step (steps 4-10) | loss @ step 1 | loss @ step 10 | NaN / skipped | exit |
|---|---|---|---|---|---|
| bf16 | 0.92 | 2.94 | 2.04 | 0 / 0 | 0 |
| nvfp4 (stock recipe) | 1.50 (**0.61x** bf16 speed) | 2.66 | 3.59 | 0 / 0 | 0 |

Three things to take from this:

1. **Use `fp4_param: false` for fine-tuning.** See the 2-GPU table: with the stock recipe the
   model barely learns at a fine-tuning learning rate.
2. **NVFP4 was slower than bf16 here.** At this model size and batch, the cost of quantizing
   activations and weights every step outweighs the faster 4-bit GEMMs. The same was measured on
   other models in this project (Gemma-4 31B dense about 1.7x slower, Gemma-4 E4B about 23% slower,
   TransformerEngine 2.11). A microbenchmark of one GEMM in isolation predicted the opposite; only
   an end-to-end control pair answers the question.
3. **Check quality on your own task.** Fifty steps of training loss shows the run learns; it is not
   an evaluation. Per-step SFT loss is also noisy at small batch (in the 1-GPU smoke run the bf16
   arm alone moved between 1.3 and 5.6), so compare trends, not single steps.

## Prerequisites

* **GPU.** At least one Blackwell GPU (2 for the default config). For bf16-only you can use any
  recent NVIDIA GPU.
* **Container.** The NVIDIA NeMo Framework container that ships Megatron-Bridge. Tested with
  Megatron-Bridge **0.6.1** and TransformerEngine **2.18.0**. Everything below runs **inside** it.
* **Disk.** About 15 GB for the HF model, 15 GB for the converted checkpoint, and about 100 GB
  per arm for the final training checkpoint (it includes optimizer state).

Start the container with this example's directory mounted, for example:

```bash
# Docker
docker run --gpus all --ipc=host --rm -it -v $PWD:/workspace -w /workspace <nemo-container> bash
# enroot (shared clusters): mount your home, keep ~/.local out of Python
enroot start --mount $HOME:$HOME --env PYTHONNOUSERSITE=1 <nemo-container-name> bash
```

Check what you got: `python -c "import megatron.bridge, transformer_engine as te; print(te.__version__)"`.

## Step 1 — get the model

```bash
cd examples/nvfp4-GB200
huggingface-cli download Qwen/Qwen2.5-7B-Instruct --local-dir models/Qwen2.5-7B-Instruct
```

## Step 2 — convert it to a Megatron checkpoint

```bash
python scripts/convert_hf_to_megatron.py models/Qwen2.5-7B-Instruct checkpoints/qwen2.5-7b
```

A few minutes on CPU; writes `checkpoints/qwen2.5-7b/iter_0000000` (15 GB). This calls
Megatron-Bridge's `AutoBridge.import_ckpt` directly. Upstream's `scripts/conversion/convert.sh` does
the same through `uv run`, which needs a writable uv cache that the container does not provide.

## Step 3 — train the bf16 control, then the NVFP4 arm

```bash
bash scripts/train_nvfp4.sh configs/qwen2.5-7b-instruct_sft.env bf16
bash scripts/train_nvfp4.sh configs/qwen2.5-7b-instruct_sft.env nvfp4
```

Both arms read the **same** config file; the precision is the only argument that differs, so the
comparison is fair by construction. With the default `FP4_PARAM=false` the NVFP4 arm uses the stock
recipe with `fp4_param` and `fp4_param_gather` turned off (the script spells out the full
`mixed_precision` block, because a recipe name cannot be edited field by field from the command
line). Each arm writes `runs/qwen2.5-7b_sft/<arm>/train.log`,
TensorBoard logs and a final checkpoint. Under the hood this is NVIDIA's own runner:

```bash
torchrun --nproc_per_node 2 /opt/Megatron-Bridge/scripts/training/run_recipe.py \
  --recipe qwen25_7b_sft_config --dataset squad -tp 2 ... \
  mixed_precision=bf16_with_nvfp4_mixed      # bf16 arm: mixed_precision=bf16_mixed
```

`mixed_precision=` is the whole NVFP4 switch, so you can add it to any Megatron-Bridge recipe the
same way; take the `FP4_PARAM=false` block from `train_nvfp4.sh` if you are fine-tuning.

## Step 4 — verify the pair

```bash
python scripts/verify_nvfp4.py runs/qwen2.5-7b_sft
```

```
bf16-arm   PASS  mixed_precision=bf16_mixed, fp4 e2m1 printed=False
nvfp4-arm  PASS  mixed_precision=block(fp4=e2m1,fp4_recipe=nvfp4), fp4 e2m1 printed=True
completed  PASS  bf16 50/50, nvfp4 50/50
finite     PASS  skipped 0/0, nan 0/0, non-finite losses none
diverged   PASS  0 of 50 common steps print the same loss text

step time (median, steps > 3): bf16 3717 ms, nvfp4 4995 ms  -> nvfp4 is 0.74x the bf16 speed
loss at step 50: bf16 0.3839, nvfp4 0.4064 (+5.8%); mean |gap| over 50 steps 0.0837
```

(the 2-GPU run above). What each check means:

| check | passes when | why |
|---|---|---|
| `bf16-arm` | the control resolved to `bf16_mixed` and printed no `fp4: e2m1` | the control must really be bf16 |
| `nvfp4-arm` | the treatment asked for NVFP4 (the recipe name, or a `mixed_precision` block with `fp4: e2m1` and `fp4_recipe: nvfp4`) **and** the resolved model config printed `fp4: e2m1` | the request reached the model, not just the command line |
| `completed` | both arms logged every planned step | a crashed arm proves nothing |
| `finite` | no NaN or skipped steps, every loss finite | 4-bit training can diverge |
| `diverged` | the two arms' losses are **not** identical at every step | same data and seed: if FP4 were off, the losses would match to the last digit. Agreement is the failure here |

`fp4_recipe` is deliberately not checked: the bf16 arm prints `fp4_recipe: nvfp4` too, as an
inactive default. Exit codes: **0** all pass, **5** a check failed, **95** nothing to check (a log
is missing or has no iteration lines). We checked that the verifier can fail: pointing it at a
pair whose "NVFP4" log is a copy of the bf16 log gives `nvfp4-arm FAIL`, `diverged FAIL`, exit 5.

## Using your own data or another model

* **Your own data.** Set `DATASET=local-jsonl` and `DATA_ROOT=<dir>` in the config. The directory
  holds `training.jsonl` and `validation.jsonl`, one `{"input": "...", "output": "..."}` object
  per line (prompt and completion; only the completion is trained on).
* **Another model.** Set `RECIPE` to any Megatron-Bridge SFT recipe (for example
  `qwen3_8b_sft_config`), convert that model in Step 2, and adjust `TP`. Check that the model's
  hidden and MLP widths are multiples of 128: on TransformerEngine 2.11 we measured an NVFP4 MoE
  model whose expert width was not (Gemma-4 26B-A4B, 704) failing on its first forward pass.
  We have not re-tested that on 2.18.

## Troubleshooting

| symptom | cause and fix |
|---|---|
| NVFP4 loss stays high (near its starting value) and grad norm stays large while bf16 learns | the run used `fp4_param: true` (the stock recipe). Set `FP4_PARAM=false`. |
| `cublas_gemm ... CUBLAS_STATUS_NOT_SUPPORTED` in `wgrad_gemm` | NVFP4's backward weight-gradient GEMM uses the batch's token count as its inner dimension, and SFT batches are only padded to a multiple of 16 by default. `train_nvfp4.sh` pads both arms to `PAD_TO_MULTIPLE` (default 128); do not lower it for NVFP4. |
| `CUDA out of memory` naming another process with most of the GPU's memory | on a shared machine, someone else's job took the GPU between your two arms. Pick idle GPUs with `CUDA_VISIBLE_DEVICES` and rerun the arm; the verifier reports the pair as UNMEASURED (exit 95) until both logs have steps. |
| `refused: NVFP4 needs a Blackwell GPU` (exit 96) | the GPU is not compute capability 10.x. Run the bf16 arm only, or move to Blackwell. |
| `Read-only file system: '/opt/Megatron-Bridge/...'` | the runner changes directory into the Megatron-Bridge repo and writes its defaults there. The script sends checkpoints and TensorBoard logs to `OUTPUT_DIR`; set any other output path the same way. |
| `Pretrained checkpoint ... does not exist` with a relative path | same cause; the script converts the config's paths to absolute ones. |
| `EADDRINUSE` while converting | the checkpoint writer starts worker processes that re-import the main script; keep the `if __name__ == "__main__":` guard in `convert_hf_to_megatron.py`. |
| `Key '...' is not in struct` for an override | to **add** a key that the recipe does not define, prefix it with `+` (Hydra syntax), as the script does for `+dataset.dataset_kwargs.pad_seq_length_to_mult`. |

## Files

| file | what it does |
|---|---|
| `configs/qwen2.5-7b-instruct_sft.env` | model, recipe, data, parallelism, batch and steps, shared by both arms |
| `scripts/convert_hf_to_megatron.py` | Hugging Face checkpoint to Megatron checkpoint (Step 2) |
| `scripts/train_nvfp4.sh` | one arm, `bf16` or `nvfp4`, through Megatron-Bridge's `run_recipe.py` (Step 3) |
| `scripts/verify_nvfp4.py` | the five checks and the cost report (Step 4) |
