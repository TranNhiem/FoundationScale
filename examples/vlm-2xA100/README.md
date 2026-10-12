# Run the vlm-2xH200 VLM SFT on 2x A100-80GB (image-only, 16K)

LoRA SFT of `gemma-4-26B-A4B-it` at 16K context on the VDoc image shard, `examples/vlm-2xH200/` scripts
+ one new `VIDEO_COLUMN=none` knob. All numbers are **MEASURED (2026-10-12, 2x A100-SXM4-80GB)** unless
marked **UNMEASURED**. Environment (cu129 vs cu132, slow-index install, stale-package cleanup) is in
[../A100_2GPU.md](../A100_2GPU.md); data/training/eval detail lives in
[../vlm-2xH200/](../vlm-2xH200/) — we link, not repeat.

## Measured results (gemma-4-26B-A4B-it, 16K, VDoc images, 100 steps)

| model | exit | peak GB/GPU | s/step | tokens/s | held-out NLL base -> tuned | ppl base -> tuned |
|---|---|---|---|---|---|---|
| gemma-4-26B-A4B-it LoRA r16/a32 | 0 | 41.9 | 25.4 | ~1,390 | 3.957 -> 0.623 | 52.3 -> 1.87 |

Also **MEASURED**: `[fs:train:done]` PASS, save gate PASS on checkpoint-100; train_runtime 2544 s,
wall 2936 s, 3.53 M input tokens, mean train_loss 1.046 (last logged step 0.605).
Held-out = first 200 of the 550 eval rows (identical rows for both passes), 319,957 assistant tokens,
0 dropped. **In-distribution caveat**: the eval rows come from the same VDoc set as training — most of
the drop is the model learning VDoc's answer/reasoning-marker format. It proves the pipeline learns;
it is **not** a general-quality claim.

## Why the H200 table is not comparable

The H200 throughput/peak rows in the H200 README are a different workload in three exact ways: they
(1) **padded every batch to the full 16K context**, (2) ran a **mixed image+video+text set**, and
(3) ran **6 steps**. This run is **image-only, unpadded, 100 steps**. Do not compare the 41.9 GB peak
or the 25.4 s/step against the H200 rows.

## 1. Data (VDoc images, shard 0)

```bash
hf download trannhiem/TranNhiem-Vietnamese-DocumentImage-Reasoning --repo-type dataset --include 'data/vdoc.jsonl' 'shards/images-00000.tar' --local-dir data/raw/image
cd data/raw/image && tar xf shards/images-00000.tar && cd ../..
python examples/vlm-2xH200/data/prepare_data.py image --input data/raw/image/data/vdoc.jsonl --image-root data/raw/image --output data/prepared/image.jsonl
python examples/vlm-2xH200/data/prepare_data.py split --input data/prepared/image.jsonl --train-out data/prepared/image.train.jsonl --eval-out data/prepared/image.eval.jsonl --eval-fraction 0.05 --seed 0
```

Expected **MEASURED** counts: 11,000 rows kept, 53,516 skipped `missing_media` (same as H200); split
10,450 train / 550 eval (seed 0, 5%).

## 2. Model

```bash
hf download google/gemma-4-26B-A4B-it --local-dir models/gemma-4-26B-A4B-it
```

Or symlink an existing checkout into `models/gemma-4-26B-A4B-it`. Expected: 2 safetensors shards,
49 GB. Pair an old model copy with the H200 install verbatim; an A100 env misses FA3/FP8 paths only.

## 3. Train (16K, image-only)

```bash
nvidia-smi --query-compute-apps=pid,used_memory --format=csv
CUDA_VISIBLE_DEVICES=<two free gpu ids> STEPS=100 SAVE_EVERY=100 \
  bash examples/vlm-2xH200/scripts/train_sft.sh examples/vlm-2xA100/configs/gemma-4-26B-A4B-it_16k_image.env
```

`VIDEO_COLUMN=none` is set by the config: the launcher then **skips declaring the `video` column**, so
image-only/text-only data (which has no `video` column) is not refused. The default is unchanged —
video datasets keep working as in [../vlm-2xH200/](../vlm-2xH200/).
PASS looks like: exit 0, a `[fs:train:done]` verdict of PASS, and the checkpoint-100 save gate PASS.
Recipe baked into the config/launcher: LoRA r16/a32, `context 16384`, GRAD_ACCUM 8 (global 16). peak
memory = highest 5 s `nvidia-smi` sample ([../A100_2GPU.md](../A100_2GPU.md)); MFU **UNMEASURED**
(MoE: experts-per-token not stated in the config).

## 4. Evaluate (held-out NLL, base then tuned)

```bash
python examples/vlm-2xH200/scripts/eval_heldout_loss.py --model models/gemma-4-26B-A4B-it \
  --data data/prepared/image.eval.jsonl --max-length 16384 --video-column none --limit 200 \
  --out eval_logs/gemma-4-26B-A4B-it_16k_image_base.json
python examples/vlm-2xH200/scripts/eval_heldout_loss.py --model models/gemma-4-26B-A4B-it \
  --adapter runs/gemma-4-26B-A4B-it_16k_image/final \
  --data data/prepared/image.eval.jsonl --max-length 16384 --video-column none --limit 200 \
  --out eval_logs/gemma-4-26B-A4B-it_16k_image_tuned.json
```

The `[--adapter ...]` argument is what switches base to tuned; drop it for the base pass. Expected
**MEASURED** output: base mean NLL 3.957 (ppl 52.3) → tuned 0.623 (ppl 1.87), 319,957 assistant
tokens, 0 dropped, on identical 200 rows. ~11 min (~660 s) per pass on 2 GPUs — given one A100's
257 TFLOPS warm matmul, expect roughly 2x H200 wall time.

## 5. Not run on A100 yet

- **32K context** — **UNMEASURED**. (The H200 16K peak is 48.2 GB; no H200 32K figure is available here.)
- The other four VLM models of the H200 tutorial — **UNMEASURED**.
- Video input — **UNMEASURED** (this run is image-only, `VIDEO_COLUMN=none`).
- DPO / GRPO — **UNMEASURED**.

See also: [../vlm-2xH200/](../vlm-2xH200/) (chapters for installation, data, training, evaluation),
[../A100_2GPU.md](../A100_2GPU.md) (A100 env + shared-node hygiene), and `examples/train_tiny.py`
for a 2-GPU DDP smoke test.
