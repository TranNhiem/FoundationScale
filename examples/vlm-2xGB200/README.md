# VLM LoRA on 2x GB200: the vlm-2xH200 tutorial on a shared GB200 tray

This example takes everything from [chapter 01](../vlm-2xH200/01_installation.md)–[chapter 05](../vlm-2xH200/05_evaluation.md) and repeats it on two GPUs of a shared GB200 tray (aarch64). By the end you have a trained adapter on GB200 and the held-out loss of its base model and adapter, exactly as in [chapter 05](../vlm-2xH200/05_evaluation.md). All numbers here were measured on 2026-10-14 on one shared GB200 tray, using 2 of its 4 GPUs.

**What you have:** a shared GB200 tray (4 GPUs, 186 GB HBM each, compute capability 10.0, aarch64 Grace CPU, 956 GB RAM), other people included. We used 2 GPUs. Other users start jobs on GPUs that look idle within minutes — plan for that in Step 2.

---

## Same vs different

| | 2x H200 (chapters 01–05) | 2x GB200 (this example) |
|---|---|---|
| Install commands | [chapter 01](../vlm-2xH200/01_installation.md) Option A | unchanged |
| Dependency pins | torch 2.13.0+cu132 etc. | unchanged; aarch64 wheels exist for every pin |
| Architecture | x86_64 | aarch64 |
| GPU memory | 141 GB each | 186 GB each |
| Job scheduling | dedicated VM | shared tray, ssh + nohup, no Slurm |
| GPU selection | both GPUs of the VM | two idle GPUs picked by `scripts/train_sft_gb200.sh` (Step 2) |
| Data commands | [chapter 02](../vlm-2xH200/02_data_preparation.md) | unchanged |
| Training launcher | `vlm-2xH200/scripts/train_sft.sh` | `scripts/train_sft_gb200.sh`: shared-tray settings, then the same `train_sft.sh` |
| Configs | `vlm-2xH200/configs/` | `configs/` here: same models, measured GB200 peaks, Qwen3.6 at batch 2 |
| `eval_heldout_loss.py` | [chapter 05](../vlm-2xH200/05_evaluation.md) | unchanged |

---

## Step 1 — Install on aarch64

The install commands from [chapter 01](../vlm-2xH200/01_installation.md) Option A run unchanged. Every pin has an aarch64 wheel: torch 2.13.0+cu132, torchvision 0.28.0, transformers 5.18.0, peft 0.21.2, liger-kernel, flash-linear-attention, av 19.0.1.

MEASURED on GB200: `conda env create -f environment.yml` takes 4–5 min and `pip install -e ".[train]"` 5–6 s (`[train]` brings peft, av, liger-kernel, flash-linear-attention).

**TRAP (shared accounts):** `~/.local/lib/python3.12/site-packages` leaks into every env and pip treats it as installed. We hit this: numpy 1.26.4 came from `~/.local`, and then every run refused with `pyarrow requires NumPy 2.0 or newer, found 1.26.4` (exit 96). Set `PYTHONNOUSERSITE=1` **before** `conda env create` and in **every training shell**:

```bash
export PYTHONNOUSERSITE=1
cd examples/vlm-2xH200/install && conda env create -f environment.yml && conda activate fs-vlm && cd ../../..
pip install -e ".[train]"
```

### Check it worked

First confirm numpy comes from the env, not `~/.local`:

```bash
python -c "import numpy; print(numpy.__file__)"
```

Expected output: a path **inside** the env (contains `envs/fs-vlm`). If it prints `~/.local`, `PYTHONNOUSERSITE=1` is not set in this shell.

Then the import check from [chapter 01](../vlm-2xH200/01_installation.md):

```bash
python -c "import torch,transformers,peft,liger_kernel,av,foundationscale;print(torch.__version__,torch.cuda.is_available(),transformers.__version__,peft.__version__)"
```

Expected output (exact):

```
2.13.0+cu132 True 5.18.0 0.21.2
```

The device check prints `device NVIDIA GB200 (10, 0)`.

---

## Step 2 — Pick 2 GPUs on the shared tray

There is no Slurm here: you run over ssh with nohup, and other people use the other GPUs. `scripts/train_sft_gb200.sh` handles the shared-tray part before it calls the unchanged [`train_sft.sh`](../vlm-2xH200/scripts/train_sft.sh):

- **Two idle GPUs.** If `CUDA_VISIBLE_DEVICES` is not set, it picks two GPUs with less than 1000 MB in use and exits 96 if there are fewer than two. To choose them yourself: `export CUDA_VISIBLE_DEVICES=1,2`.
- **`PYTHONNOUSERSITE=1`**, so packages in `~/.local` cannot leak in (Step 1).
- **Your own `TMPDIR`** (`./tmp`): `torchrun` otherwise writes `torchelastic_*` directories into `/tmp`.
- **A free port:** `MASTER_PORT` defaults to 29871 because the usual 29500 is often taken. Override it if that one is busy too.

It prints what it chose, for example:

```
GB200: GPUs 1,2, master port 29871, TMPDIR /path/to/FoundationScale/tmp
```

Name the job so other users can tell whose it is. `exec -a` on a shebang script loses the name, so wrap it in `bash` on both sides:

```bash
bash -c 'exec -a foxbrain-vlm-sft bash examples/vlm-2xGB200/scripts/train_sft_gb200.sh <config>'
```

(On Slurm use `--job-name` instead.)

### Check it worked

Run `nvidia-smi` or `nvidia-smi --query-compute-apps` — your two processes show up under the job name `foxbrain-vlm-sft`. Remember: **read per-process memory, not per-GPU totals.** Another user's job on the same GPU inflated one reading to 132 GB while our process used 47 GB.

---

## Step 3 — Data

The data commands from [chapter 02](../vlm-2xH200/02_data_preparation.md) are unchanged. Counts are identical to chapter 02: image 11,000 written / 53,516 skipped (shard 0), video 2,985, text 209, mix 2,000 — 34 s total (image download 19 s).

**Frame cache:** the first run decodes clips; the pre-pass took 545 s for 300 rows (114 clips, 520 MB of JPEGs) on the shared filesystem with the ARM CPU. Later runs reach the same pass in 43 s. The default cache is `<dataset dir>/.fs_video_frames`; `FOUNDATIONSCALE_TRAIN_VIDEO_CACHE_DIR` overrides it. Warming all 949 video rows of `mixed.jsonl` + the held-out file took 4.5 min with 16 threads.

### Check it worked

The prepared files exist (`data/prepared/…`) and match the counts above. The first training run's pre-pass decodes and caches the frames; later runs start step 1 much sooner (43 s instead of 545 s for 300 rows).

---

## Step 4 — Train

Run the end-to-end command from the repository root:

```bash
STEPS=40 SAVE_EVERY=40 \
  bash -c 'exec -a foxbrain-vlm-sft bash examples/vlm-2xGB200/scripts/train_sft_gb200.sh examples/vlm-2xGB200/configs/Qwen3.6-35B-A3B_16k.env'
```

The Qwen3.6 16K configs set `BATCH=2 GRAD_ACCUM=4`, which keeps the H200 default effective batch of `BATCH=1 GRAD_ACCUM=8` (see [chapter 03](../vlm-2xH200/03_training_configuration.md)); the extra 45 GB per GB200 GPU buys the larger micro-batch. The gemma-4 configs keep `BATCH=1` (batch 2 gained little: 1,823 vs 1,670 tokens/s). Drop `STEPS=40 SAVE_EVERY=40` for a full run (1,000 steps, a checkpoint every 200).

Expect a pre-pass over the 2,000-row `mixed.jsonl` with a warm frame cache in ~9.5 min (3.5 rows/s on the ARM CPU) before step 1.

### Check it worked

These are the measured end-to-end numbers for this exact configuration (`configs/<model>_16k.env`, full 2,000-row `mixed.jsonl`, `STEPS=40 SAVE_EVERY=40`):

- **Qwen3.6-35B-A3B, `BATCH=2 GRAD_ACCUM=4`:** exit 0, save gates PASS, 27 min including the ~9.5 min pre-pass; train loss 0.676 -> 0.432.
- **gemma-4-12B-it, `BATCH=1 GRAD_ACCUM=8`:** exit 0, save gates PASS, 26 min; train loss 1.089 -> 0.597.

---

## Step 5 — Evaluate

Same procedure as [chapter 05](../vlm-2xH200/05_evaluation.md). To make the held-out file mixed, shuffle `heldout.jsonl` once:

```bash
python3 -c "import random; r=open('data/prepared/heldout.jsonl').readlines(); random.Random(0).shuffle(r); open('data/prepared/heldout_mixed.jsonl','w').writelines(r)"
```

Then evaluate the base model and the adapter with `eval_heldout_loss.py` on 240 held-out rows (194 image / 44 video / 2 text), `--video-frames 16`, max length 16384:

```bash
python examples/vlm-2xH200/scripts/eval_heldout_loss.py --model models/<model> --data data/prepared/heldout_mixed.jsonl --max-length 16384 --video-frames 16 --limit 240 --out results/<name>_base.json
python examples/vlm-2xH200/scripts/eval_heldout_loss.py --model models/<model> --adapter runs/<run>/checkpoint-40 --data data/prepared/heldout_mixed.jsonl --max-length 16384 --video-frames 16 --limit 240 --out results/<name>_adapter.json
```

### Check it worked

Measured results of the Step 4 runs:

| Model | Held-out NLL (base -> adapter) | Perplexity (base -> adapter) | NLL change |
|---|---|---|---|
| Qwen3.6-35B-A3B | 0.994 -> 0.458 | 2.70 -> 1.58 | -54% |
| gemma-4-12B-it | 1.577 -> 0.672 | 4.84 -> 1.96 | -57% |

---

## Step 6 — Measured fit matrix

LoRA r16/a32, FSDP 2 GPUs, bf16, gradient checkpointing, sdpa, liger, batch 1, padded, 6 steps, image+video+text 300 rows, 16 frames. **GB = per-GPU peak of our processes.** Every cell is `GB / tokens/s` on GB200 with an H200 tokens/s reference column from the README.

| Model | 16K GB200 | 16K H200 ref | 32K GB200 | 32K H200 ref | 64K GB200 | 64K H200 ref | 128K GB200 | 128K H200 ref |
|---|---|---|---|---|---|---|---|---|
| gemma-4-12B-it | 31.8 / 1,670 | 2,596 | 48.8 / 1,021 | 1,582 | 88.4 / 468 | 861 | OOM | OOM |
| gemma-4-26B-A4B-it | 48.3 / 2,532 | 3,626 | 64.4 / 1,570 | 2,526 | 105.3 / 853 | 1,379 | OOM | OOM |
| gemma-4-31B-it | 61.8 / 734 | 1,088 | 89.3 / 415 | 650 | OOM | OOM (max 32K) | OOM | OOM |
| Qwen3.6-27B (with fix) | 51.6 / 2,748 | 3,372 | 70.4 / 1,926 | 3,024 | 112.2 / 1,140 | 2,042 | OOM | OOM |
| Qwen3.6-35B-A3B (with fix) | 50.0 / 5,688 | 8,009 | 58.7 / 4,271 | 8,071 | 81.3 / 2,720 | 4,830 | 144.6 / 1,470 fits | OOM |

**Note:** the H200 Qwen numbers predate the FSDP fix described below.

Micro-batch sweep on GB200 at 16K: Qwen3.6-35B-A3B batch 2: 6,276 tokens/s (58.0 GB); batch 4: 6,651 (74.0 GB); text-only 7,075. Qwen3.6-27B batch 2: 2,872 (70.0 GB).

Memory matches H200 within a few GB; the extra 45 GB per GPU is what lets Qwen3.6-35B-A3B run at 128K here (144.6 GB peak) and take larger micro-batches.

---

## Training algorithms on GB200: DPO, ORPO, GRPO with LoRA

The preference and RL scripts from [chapter 04](../vlm-2xH200/04_training_algorithms.md) run unchanged on 2x GB200. Data comes from chapter 04 Step 1 (`prepare_rl_data.py preference`, `preference_mm`, `scienceqa`). Every run below picked two idle GPUs and used its own `MASTER_PORT`. Measured 2026-10-15:

| run | command difference vs chapter 04 | exit | steps | s/step | peak GB (rank 0) | check |
|---|---|---|---|---|---|---|
| DPO, text, gemma-4-12B-it | none (`--max-length 4096`) | 0 | 20/20 | 11.8 | 32.0 | step-0 loss = ln 2 exactly |
| DPO, image+video+text, Qwen3.6-27B | `--max-length 16384` | 0 | 20/20 | 20.9 (H200: 20.6) | 91.6 | step-0 loss = ln 2 exactly |
| ORPO, image+video+text, Qwen3.6-27B | `--max-length 16384` | 0 | 10/10 | 21.1 | 90.2 | ORPO adds an NLL term, so step 0 is not ln 2 |
| GRPO, images, gemma-4-12B-it | none | 0 | 8 measured, 12 abstained | 202 | — | another user's job shared one GPU, so the time is not clean |
| GRPO, images, Qwen3.6-27B | `--max-new-tokens 1024` | 96 (refused) | 0 measured | — | — | see below |

**Why `--max-length 16384` for multimodal preference data.** Built from the full 2,985-row video set, the mixed preference file has a row longer than 8,192 tokens after media expansion. With chapter 04's `--max-length 8192` the trainer refuses (exit 96) instead of truncating, because truncation could cut a media placeholder. 16,384 fits with room to spare on GB200.

**Why Qwen3.6-27B GRPO is refused on this recipe.** At `--max-new-tokens 400` and again at 1,024, almost every completion reasoned without ending in `Answer: X` (typically 0 of 8 completions per step earned a score). The few scored rows gave the advantage estimator nothing to use, so all 20 steps were UNMEASURED. FoundationScale refuses a run that trained nothing (exit 96) rather than report it. This is a prompt and answer-format problem of this recipe with a reasoning model, not a GB200 or memory limit (66 GB peak, no errors). gemma-4-12B-it on the same data trains (8 of 20 steps measured).

**Dense vs MoE.** On the two MoE models LoRA trains attention and the shared (Qwen) or dense (gemma) MLP, not the routed experts or the router; see [chapter 03, "Dense vs MoE"](../vlm-2xH200/03_training_configuration.md#dense-vs-moe-what-training-actually-changes) for what that means for speed, memory and quality.

## Longer run: 300 steps on every model

The Step 4 command with `STEPS=300 SAVE_EVERY=300`, one model per pair of GPUs, five pairs in parallel through the shared GPU pool. Same 2,000-row image+video+text mix, the 16K configs from `configs/` (gemma-4: batch 1 x 8 accumulation; Qwen3.6: batch 2 x 4, the same effective batch), then [chapter 05](../vlm-2xH200/05_evaluation.md)'s held-out loss on the 240 mixed held-out rows (194 image / 44 video / 2 text) for the base model and the 300-step adapter. Measured 2026-10-18:

| model | held-out NLL, base | after 300 steps | change | training time |
|---|---|---|---|---|
| gemma-4-12B-it | 1.577 | 0.515 | -67% | 2.9 h |
| gemma-4-26B-A4B-it | 5.048 | 0.581 | -88% | 2.0 h |
| gemma-4-31B-it | 2.831 | 0.509 | -82% | 2.8 h |
| Qwen3.6-27B | 0.904 | 0.363 | -60% | 4.0 h |
| Qwen3.6-35B-A3B | 0.994 | 0.400 | -60% | 1.9 h |

All five finished with exit 0 and save gates PASS; every evaluation scored all 240 rows. gemma-4-12B-it needed one re-run: its first attempt stopped at the first collective (the timeout described in Troubleshooting), and the identical re-run on another tray passed. The held-out set here is not the one in the H200 300-step table, so compare the change, not the absolute NLL.

## Why GB200 is not faster here

At batch 1, gemma-4 runs at about 0.65–0.7x the H200 tokens/s across lengths, and Qwen3.6 at 0.5–0.8x of the published H200 numbers (which predate the FSDP fix below, so they understate H200 too). We ruled out the usual suspects: not data loading (collate is 0.03–0.3 s/row against ~20 s steps), not clocks (2,062 MHz, no throttle), not interconnect (NCCL uses P2P/CUMEM, direct GPU-to-GPU over NVLink).

What remains is attention. All five models use 256-wide attention heads in the language model (gemma-4, and Qwen3.6's full-attention layers: `head_dim` 256 in both configs). At that head size PyTorch's SDPA has only its FlashAttention kernel on GB200; the cuDNN and memory-efficient kernels have none. An attention-only benchmark with gemma-4's shapes (forward + backward) measured 23.3 ms at 16K and 354.8 ms at 64K, roughly 330 TFLOPS: Hopper-class speed, not Blackwell's. For reference, the cuDNN kernel at head size 128 ran about 4x faster than FlashAttention on the same GPU (16K: 5.0 vs 21.2 ms), but none of these five models has 128-wide heads, so `--sdp-backend cudnn` does not help them. gemma batch 2 gains little (1,823 tokens/s); Qwen3.6 gains from batch 2 (Step 4).

---

## Two bugs this hardware found

Both are fixed in the repository now.

1. **`prepare_data.py` did not create the output directory.** Fixed. The Step 3 commands work on a fresh clone.
2. **`train/loop.py` FSDP unit split.** `Qwen3_5VisionBlock` was listed in `_no_split_modules`, which made every vision block its own FSDP unit. With batch > 1 and image+video rows, the run **hung to the 30-min NCCL timeout**; at batch 1 it "only" cost speed. Tower-internal block classes now stay in the root unit. Measured effect on Qwen3.6-35B-A3B at 16K batch 1: 1,472 -> **5,688 tokens/s**.

---

## Troubleshooting

**`pyarrow requires NumPy 2.0 or newer, found 1.26.4` (exit 96)** — the `~/.local` leak from a shared account. Set `export PYTHONNOUSERSITE=1` in this shell and re-check `python -c "import numpy; print(numpy.__file__)"` points inside the env. It must be set before `conda env create` and in every training shell.

**The run starts but another job appears on your GPU mid-run and it slows down** — expected on a shared tray. Measured: gemma-4-31B-it at 32K dropped from 415 to 301 tokens/s clean vs shared. Re-run the Step 2 one-liner next time and re-check with `nvidia-smi --query-compute-apps`.

**Per-GPU memory looks far higher than the matrix above** — read per-process memory (`nvidia-smi --query-compute-apps`), not per-GPU totals; another user's job inflated one reading to 132 GB while our process used 47 GB.

**`MASTER_PORT=29500` fails to bind** — default port is taken. Set `MASTER_PORT=<free port>` (we used 29871 in Step 4).

**`torchelastic_*` files in `/tmp`** — `TMPDIR` not set. `mkdir -p tmp && export TMPDIR=$PWD/tmp` before every run.

**The job name in `nvidia-smi` is not `foxbrain-vlm-sft`** — `exec -a` on a shebang script loses the name. Wrap it: `bash -c 'exec -a foxbrain-vlm-sft bash … script.sh <config>'`.

**Batch > 1 with Qwen image+video rows hangs to the NCCL timeout** — you are running stale code without the `train/loop.py` fix. Update and re-run (see "Two bugs this hardware found" above).

**The first collective times out after 120 s (`WorkNCCL(SeqNum=1, OpType=ALLREDUCE, NumelIn=1 ...)` on both ranks), before step 1.** Seen in 2 of about 35 launches, both on GPUs 0 and 1 of the same tray; rerunning the identical command passed both times, and NCCL then used `P2P/CUMEM`. Check `nvidia-smi` for another job on your GPUs and rerun. To see where setup stalls, add `NCCL_DEBUG=INFO`.

**A run shorter than 10 steps ends `UNMEASURED` (exit 95) although it trained and saved.** `train_sft.sh` logs every 10 steps, so no loss was ever logged and the objective gates had nothing to read. Use `STEPS` of 10 or more for a smoke test.
