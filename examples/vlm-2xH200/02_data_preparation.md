# 02 — Data preparation: image-text, video-text and text-only, and mixing them

In this chapter you will get three sample datasets onto the VM, convert every one of them into the single JSONL format FoundationScale trains on, split each into a train/held-out pair, mix the three with weighted sampling, and read the resulting statistics. Along the way you will set the few environment variables that control video frame sampling and the handling of over-long conversations.

**Time needed:** about 30–45 min for the image and text datasets combined (measured by a fresh user: the convert, split, mix and stats commands themselves take **< 1 min** on the samples — the time is spent downloading and unpacking); the video set is ~116 GB and takes hours to download.

All commands run from the repository root (`FoundationScale/`). Everything the converter does is pure Python stdlib; there is nothing to `pip install` here.

---

## 0. Know the canonical format

FoundationScale trains on **one JSON object per line**. A row looks like this (long fields truncated):

```json
{"id": "96220", "source": "image", "conversations": [{"from": "human", "value": "<image>\nTheo văn bản, những tác phẩm nào được liệt kê là của tác giả Vũ đình Long?"}, {"from": "gpt", "value": "🥵Okay, let's see. The user is asking which works are listed as being by the author Vũ Đình Long according to the text.\n\n…🥵/Theo văn bản, các tác phẩm được liệt kê là của tác giả **Vũ Đình Long** bao gồm: …"}]}
```

```json
{"id": "50969", "source": "video", "conversations": [{"from": "human", "value": "<video>\nDescribe what is happening in the video clip from 03:55.13 to 04:01.53."}, {"from": "gpt", "value": "🥵The user wants me to describe the video segment from 03:55.13 to 04:01.53.\n\n1.  **Identify the timeframe:** …"}]}
```

Field by field:

| Field | Required | Meaning |
|---|---|---|
| `id` | yes | Row identifier. |
| `source` | yes | `"image"`, `"video"` or `"text"`. |
| `conversations` | yes | List of `{"from", "value"}` turns; `from` is `"human"`, `"gpt"` or `"system"`. |
| `image` | optional | List of image paths. Human turns carry **one `<image>` marker per image**. |
| `video` | optional | Path to a video file. Human turns carry **one `<video>` marker**. |
| `start`, `end` | optional | Clip segment in seconds inside the full video. Video rows must carry **both**. |

Things that are worth internalising now, because they affect what your data must look like:

- **Reasoning is written inline** as `🥵…🥵/…answer`. You do **not** hand-convert it: FoundationScale translates it into the target model's native format. **Gemma 4 uses its own thought channel and does *not* understand a literal `🥵` tag** — `Qwen3.6` uses `🥵`. Never tape one family's tags onto the other by hand.
- **Reasoning is kept only on the *final* assistant turn.** Both model families' chat templates drop reasoning from earlier turns. So in the 10-turn Vietnamese document rows and the 6-turn video rows (turn counts are typical, not guaranteed — see below), only the last answer's reasoning is actually trained.
- **Loss is computed on assistant turns only.** System and human turns are context, not targets.
- The converter **does not require you to write `<image>` yourself**: the image-text sample has no markers, and the converter inserts them. If you build custom data, write markers explicitly and expect them to be kept as-is.

---

## 1. Get the converter

```bash
python examples/vlm-2xH200/data/prepare_data.py --help
```

Check it worked:

```
usage: prepare_data.py [-h] {image,video,text,mix,stats,split} ...

Convert raw datasets into one canonical JSONL format and mix them.

positional arguments:
  {image,video,text,mix,stats,split}
    image               Convert a VDoc-style image dataset.
    video               Convert an Action-100M-style video dataset.
    text                Convert a text-only QA dataset.
    mix                 Mix canonical JSONL files with weighted sampling.
    stats               Print statistics on a canonical JSONL file.
    split               Split a canonical JSONL file into disjoint train/eval files.

options:
  -h, --help            show this help message and exit
```

Each of `image`, `video`, `text` takes `--max-rows N` (stop after N *valid* rows) and `--no-check-files` (skip media existence checks). `split` has its own flags. Full help per subcommand is in the details blocks below if you want to look up flags later.

<details>
<summary><code>image</code> / <code>video</code> / <code>text</code> flag reference</summary>

```
usage: prepare_data.py image [-h] --input INPUT --output OUTPUT
                             [--max-rows MAX_ROWS] [--no-check-files]
                             --image-root IMAGE_ROOT

options:
  -h, --help            show this help message and exit
  --input INPUT         Path to the input JSONL file.
  --output OUTPUT       Path to the output JSONL file.
  --max-rows MAX_ROWS   Stop after writing N valid rows (default: unlimited).
  --no-check-files      Skip media file-existence checks during validation.
  --image-root IMAGE_ROOT
                        Directory that relative image paths are resolved
                        against.
```

```
usage: prepare_data.py video [-h] --input INPUT --output OUTPUT
                             [--max-rows MAX_ROWS] [--no-check-files]
                             --video-root VIDEO_ROOT

options:
  -h, --help            show this help message and exit
  --input INPUT         Path to the input JSONL file.
  --output OUTPUT       Path to the output JSONL file.
  --max-rows MAX_ROWS   Stop after writing N valid rows (default: unlimited).
  --no-check-files      Skip media file-existence checks during validation.
  --video-root VIDEO_ROOT
                        Directory that video basenames are resolved against.
```

```
usage: prepare_data.py text [-h] --input INPUT --output OUTPUT
                            [--max-rows MAX_ROWS] [--no-check-files]

options:
  -h, --help            show this help message and exit
  --input INPUT         Path to the input JSONL file.
  --output OUTPUT       Path to the output JSONL file.
  --max-rows MAX_ROWS   Stop after writing N valid rows (default: unlimited).
  --no-check-files      Skip media file-existence checks during validation.
```

</details>
<details>
<summary><code>mix</code> / <code>split</code> / <code>stats</code> flag reference</summary>

```
usage: prepare_data.py mix [-h] --inputs INPUTS [INPUTS ...] --weights WEIGHTS
                           [WEIGHTS ...] --total TOTAL --output OUTPUT
                           [--seed SEED]

options:
  -h, --help            show this help message and exit
  --inputs INPUTS [INPUTS ...]
                        Two or more canonical JSONL files to mix.
  --weights WEIGHTS [WEIGHTS ...]
                        Sampling weight for each input (same count as
                        --inputs).
  --total TOTAL         Total number of rows in the output.
  --output OUTPUT       Path to the output JSONL file.
  --seed SEED           Random seed for deterministic sampling and shuffling
                        (default: 0).
```

```
usage: prepare_data.py split [-h] --input INPUT --train-out TRAIN_OUT
                             --eval-out EVAL_OUT [--eval-fraction EVAL_FRACTION]
                             [--seed SEED]

options:
  -h, --help            show this help message and exit
  --input INPUT         Canonical JSONL file to split.
  --train-out TRAIN_OUT Where the training rows are written.
  --eval-out EVAL_OUT   Where the held-out rows are written.
  --eval-fraction EVAL_FRACTION
                        Fraction of rows sent to --eval-out.
  --seed SEED           Random seed for the split.
```

```
usage: prepare_data.py stats [-h] --input INPUT

options:
  -h, --help     show this help message and exit
  --input INPUT  Canonical JSONL file to analyse.
```

</details>

---

## 2. Image-text: Vietnamese document images with reasoning

**What it is:** `trannhiem/TranNhiem-Vietnamese-DocumentImage-Reasoning` on the Hub — 64,516 records of Vietnamese document images with 10-turn conversations. The repository contains a `data/vdoc.jsonl` metadata file plus **6 image tar shards under `shards/`**, named `shards/images-00000.tar` … `shards/images-00005.tar`. The `gpt` turns are `{"reasoning","answer"}` dicts, which the converter unfolds into the `🥵…🥵/…answer` form for you.

Note: 64,516 is the metadata count. The measured baseline below uses **only shard 0**, which is why you get 11,000 rows and 53,516 skips.

### Download

```bash
hf download trannhiem/TranNhiem-Vietnamese-DocumentImage-Reasoning \
  --repo-type dataset \
  --local-dir data/raw/image
```

### Extract the image shards

Each shard is a tar whose members are `images/…`. Extract them **inside `data/raw/image/`** so that the relative image paths inside `data/vdoc.jsonl` resolve. For a fast start, extract shard 0 only:

```bash
cd data/raw/image
tar xf shards/images-00000.tar
cd ../..
```

All six shards at once:

```bash
cd data/raw/image
for s in shards/*.tar; do tar xf "$s"; done   # images-00000.tar ... images-00005.tar
cd ../..
```

Partial extraction is fine on purpose: the converter checks that each image file exists and **skips rows whose image is missing**, so a half-downloaded dataset still yields a usable training file.

### Convert

```bash
python examples/vlm-2xH200/data/prepare_data.py image \
  --input data/raw/image/data/vdoc.jsonl \
  --image-root data/raw/image \
  --output data/prepared/image.jsonl
```

Check it worked (measured with shard 0 extracted):

```
rows written: 11000
rows skipped: 53516
missing_media: 53516
```

The converter always prints one `rows written: N` line and one `rows skipped: M` line, and *then* one line per skip reason (like `missing_media: …`) whenever that reason's count is greater than zero.

If you later extract the remaining shards and re-run, `rows written` goes up and `missing_media` goes down. TODO(verify) the exact counts with all 6 shards extracted.

---

## 3. Video-text: human activity clips with segment annotations

**What it is:** `trannhiem/TranNhiem-Action100M-Human-Activities` — 2,985 records over 908 YouTube source videos, ~116 GB in total. Each record describes a `[start, end]` clip inside a full video (median 8.7 s), with a typical 6-turn English conversation and `🥵…🥵/` reasoning. Turn counts are **typical, not guaranteed** — measured on a 120-row sample: 113 of 120 rows have 6 turns, 7 have 2. The layout on the Hub is:

```
data/action100m_split1.jsonl              all 2,985 records
data/action100m_split1_shortvideo.jsonl   short-video subset
data/action100m_split1_longvideo.jsonl    long-video subset
videos/<youtube_id>.mp4                   908 source videos
```

**Publishing in progress:** if the dataset is not on the Hub yet, ask the organisers for the files and place them in that layout.

**Read the licence first:** the dataset is under **FAIR Noncommercial Research** terms and the videos are third-party YouTube content. Noncommercial research use only.

### Download (full)

```bash
hf download trannhiem/TranNhiem-Action100M-Human-Activities \
  --repo-type dataset \
  --local-dir data/raw/video
```

There are no tars to extract here — `hf download` leaves the layout above on disk. Expect hours: it is ~116 GB across 908 videos.

### Or download only what you need

The dataset is too large for a first end-to-end test. Download `data/` first, then a subset of the videos with `--include` patterns (the selection is up to you — see `hf download --help`):

```bash
hf download trannhiem/TranNhiem-Action100M-Human-Activities \
  --repo-type dataset \
  --local-dir data/raw/video \
  --include <patterns covering data/ and the videos you want>
```

The converter **skips rows whose video file is missing**, exactly like the image converter. Measured with 300 of the 908 videos present:

```
rows written: 671
rows skipped: 2314
missing_media: 2314
```

Download more videos and re-run to grow the file.

### Convert

`--video-root` is the directory that the **basenames** in the metadata are resolved against — that is `videos/` here:

```bash
python examples/vlm-2xH200/data/prepare_data.py video \
  --input data/raw/video/data/action100m_split1.jsonl \
  --video-root data/raw/video/videos \
  --output data/prepared/video.jsonl
```

Check it worked (full download: every record resolves):

```
rows written: 2985
rows skipped: 0
```

Each converted row carries `start`/`end` in seconds, so the trainer knows which segment to sample frames from.

---

## 4. Text-only: ShareGPT-style QA

**What it is:** any ShareGPT jsonl works. The competition sample is a small Vietnamese persona reasoning file: 209 rows, 2-turn conversations (typical, not guaranteed), `🥵…🥵/` reasoning. It is supplied by the organisers and is **not** on the Hub.

```bash
python examples/vlm-2xH200/data/prepare_data.py text \
  --input data/raw/text/persona_vi_reasoning.jsonl \
  --output data/prepared/text.jsonl
```

Check it worked:

```
rows written: 209
rows skipped: 0
```

---

## 5. Split, then mix the three datasets

`mix` does weighted sampling over the inputs to produce exactly `--total` rows, then shuffles. Weights are relative — `0.4 0.4 0.2` means 40 % / 40 % / 20 % of the output.

### First: keep a held-out split

Do not evaluate on the rows you trained on. Split **each converted file before mixing**, so that train and eval are exactly disjoint and every row lands in precisely one of the two files:

```bash
python3 examples/vlm-2xH200/data/prepare_data.py split \
  --input data/prepared/image.jsonl \
  --train-out data/prepared/image.train.jsonl \
  --eval-out data/prepared/image.eval.jsonl \
  --eval-fraction 0.05 \
  --seed 0
```

It prints one line of the form:

```
split data/prepared/image.jsonl: N train -> data/prepared/image.train.jsonl, M eval -> data/prepared/image.eval.jsonl (seed 0)
```

with `N + M` equal to the number of rows in `--input`. Run the same command for `video.jsonl` and `text.jsonl`. Then **mix only the `*.train.jsonl` files** and keep the `*.eval.jsonl` files untouched — chapter 05 evaluates on them. Concatenate the three eval files into one held-out file, which is exactly what chapter 05 scores:

```bash
cat data/prepared/image.eval.jsonl data/prepared/video.eval.jsonl data/prepared/text.eval.jsonl > data/prepared/heldout.jsonl
```

### Mix

```bash
python examples/vlm-2xH200/data/prepare_data.py mix \
  --inputs data/prepared/image.train.jsonl data/prepared/video.train.jsonl data/prepared/text.train.jsonl \
  --weights 0.4 0.4 0.2 \
  --total 2000 \
  --output data/prepared/mixed.jsonl \
  --seed 0
```

The mix output is `data/prepared/mixed.jsonl` — that is the file every `configs/*.env` points at through `DATASET`.

Check it worked (measured):

- `image` 800 rows / `video` 800 rows / `text` 400 rows
- a warning telling you that `text` **was repeated to fill 400**

### Reading the repeat warning

When a pool cannot fill its quota the converter says so, in so many words. The exact wording (measured here with a `text` pool of 209 rows and a 400 quota):

```
⚠  source 'data/prepared/text.jsonl' had 209 rows but quota is 400.  Repeated 191 rows (each original row appears at least once).
```

The source name and numbers in your run describe the file and quota you actually passed (e.g. `data/prepared/text.train.jsonl` after the split); the wording is always this shape.

The text pool only has ~209 rows and you asked for 400, so the sampler has to reuse rows. That is not an error, but it deserves attention:

- Rows copied more than once add no information; they just weight those conversations harder. The model starts to memorise 209 rows instead of seeing 400 different ones.
- Every repeat is a signal that your `--weights` are asking for more of a pool than the pool holds. Prefer lowering the text weight (or `--total`) over silently doubling rows.
- Keep an eye on the same thing for the other pools once you use the full image shard set.

### Choosing weights

Start from what is scarce and what you want the evaluated models to do well at. Image-text document reasoning and video understanding dominate this sample, so 40 % / 40 % is the given baseline; text is the anchor preventing the model from degenerating into pure VQA. Then rerun `stats` (next section) and adjust. There is no single correct split — just don't let the weights drift away from what the competition actually evaluates. TODO(verify) what the evaluation emphasises.

---

## 6. Sanity-check the mix

```bash
python examples/vlm-2xH200/data/prepare_data.py stats --input data/prepared/mixed.jsonl
```

Check it worked (measured for the 2,000-row mix above):

- turns per row: `2: 428`, `6: 772`, `10: 800`
- rows with reasoning: `2000`
- median characters per row: `8932`

Two things to notice: the three sources contribute distinct conversation lengths (text = 2, video = 6, image = 10 turns — **typical, not guaranteed**: measured on 120 video rows, 113 have 6 turns and 7 have 2), and reasoning rows are guaranteed only on the **final** assistant turn of each conversation regardless of those lengths.

---

## 7. Declare a video frame budget

Video frames are sampled **per clip, inside `[start, end]`**, and the segment **is** honoured: sampling is uniform over exactly that interval (a video row must carry both `start` and `end`), and the segment is part of the decode-cache key. The frames are handed to the model as **native video** — with timestamps, for Qwen — not as separate images.

The frame budget is **declared by you — there is no hidden default**. Set these before training:

```bash
export FOUNDATIONSCALE_TRAIN_VIDEO_COLUMN=video       # required whenever your rows carry <video>
export FOUNDATIONSCALE_TRAIN_VIDEO_FRAMES=16          # required — frame budget per clip
export FOUNDATIONSCALE_TRAIN_VIDEO_SAMPLING=uniform   # the only option
export FOUNDATIONSCALE_TRAIN_VIDEO_MAX_SIDE=448       # optional — longest-side pixel cap per frame
export FOUNDATIONSCALE_TRAIN_VIDEO_CACHE_DIR=...      # optional — decoded-frame cache directory
```

- `FOUNDATIONSCALE_TRAIN_VIDEO_COLUMN` names the field holding the video path (the canonical field is `video`). Required when your rows carry `<video>`.
- `FOUNDATIONSCALE_TRAIN_VIDEO_FRAMES` is required and has no default. Forget it and training is **refused with exit 96**, along with a message asking you to declare the frame budget.
- `FOUNDATIONSCALE_TRAIN_VIDEO_SAMPLING=uniform` is the only option.
- `FOUNDATIONSCALE_TRAIN_VIDEO_MAX_SIDE` is optional: it caps the longer side of each decoded frame to N pixels and keeps the aspect ratio. Unset, the decoded resolution is kept and the processor resizes by its own rule.
- `FOUNDATIONSCALE_TRAIN_VIDEO_CACHE_DIR` is optional: decoded frames are cached per clip, budget and segment in a `.fs_video_frames` directory next to the dataset by default. The first run pays the decode cost (measured pre-pass: 197 s over 300 rows with a cold cache, ~30–60 s once warm).

### What a clip costs in tokens

Measured at the default resolution with 16 frames per clip:

| Model | Tokens per frame | Tokens per 16-frame clip |
|---|---|---|
| `gemma-4-12B-it` | 63 | 1008 |
| `Qwen3.6-27B` | 40 | 640 |

A 16-frame clip together with its 6-turn conversation is about 1.8–2.3 K tokens — far under a 16 K context. Budget your `--total` and frame count against the context limit in Section 8.

**Qwen caveat:** Qwen's video processor in `transformers` 5.18 does **not** apply the per-frame pixel cap that the reference `qwen-vl-utils` applies, so a high-resolution video can cost many more tokens than the table says. If rows are being dropped as overlong, set `FOUNDATIONSCALE_TRAIN_VIDEO_MAX_SIDE` (e.g. `448`) to bring the per-frame token cost back down.

---

## 8. Decide what happens to over-long conversations

Once images and frames are expanded into tokens, some conversations will exceed the maximum sequence length. Training **never truncates**: it either drops those rows or refuses to start. You must pick one explicitly:

```bash
export FOUNDATIONSCALE_TRAIN_OVERLONG=drop    # or: refuse
```

The measurement happens in a **pre-pass before training, on every rank** — after images/frames have been expanded into tokens — so you know the answer before any GPU work.

| Value | Behaviour | When to use |
|---|---|---|
| `drop` | Over-long rows are removed and counted; the count is recorded in `run_manifest.json` under `conversation_prepass dropped_overlong`. | You have plenty of rows and a few long ones are not worth shrinking the dataset. |
| `refuse` | Training exits with status 96, naming the first offending row and its token count. | You want to catch a mis-configured frame budget or context length before burning GPU time. |

Use `refuse` for your first run so a bad setting shows up immediately; switch to `drop` once you know your longest rows.

---

## Troubleshooting

**My image conversion says `rows skipped: …` and `missing_media: …`.** Expected when you extracted fewer than all 6 shards. Each counted row points at an image file that is not on disk yet. Extract more shards (`shards/images-00001.tar` … `images-00005.tar`) and re-run — the skipped rows come back. Only use `--no-check-files` if you are sure the files exist at training time; otherwise the failure moves (lazily) into the training run.

**My video conversion writes far fewer than 2,985 rows.** Same mechanism: you downloaded `data/` but not all 908 videos. With 300 videos present you get 671 rows; download more videos and re-run.

**`mix` warns that a source was repeated.** You asked for more rows than that pool contains. Lower that source's weight, lower `--total`, or widen the pool. See Section 5.

**Training exits 96 and talks about a frame budget.** `FOUNDATIONSCALE_TRAIN_VIDEO_FRAMES` has no default. Set it (and `FOUNDATIONSCALE_TRAIN_VIDEO_COLUMN`) and start again — nothing was trained and nothing was silently zeroed.

**Training exits 96 and names a row with a token count.** `FOUNDATIONSCALE_TRAIN_OVERLONG=refuse` found an over-long row in the pre-pass. Lower `FOUNDATIONSCALE_TRAIN_VIDEO_FRAMES`, set `FOUNDATIONSCALE_TRAIN_VIDEO_MAX_SIDE`, or switch to `=drop`.

**A training row is much longer than expected.** Frames count as tokens and the cost is model-dependent — 63 tokens/frame for `gemma-4-12B-it` versus 40 for `Qwen3.6-27B` at the default resolution. With Qwen in `transformers` 5.18 a high-resolution clip can cost far more than that (no `qwen-vl-utils` pixel cap); set `FOUNDATIONSCALE_TRAIN_VIDEO_MAX_SIDE=448`. Rows are not truncated; with `FOUNDATIONSCALE_TRAIN_OVERLONG=drop` they silently disappear from the run (the count is in `run_manifest.json`).

**The pre-pass is slow the first time and fast afterwards.** That is the frame decode cache (`FOUNDATIONSCALE_TRAIN_VIDEO_CACHE_DIR`, or `.fs_video_frames` next to the dataset). Measured: 197 s over 300 rows with a cold cache, ~30–60 s warm. The cache keys on the clip, the frame budget and the segment.

**The video download is enormous.** It is ~116 GB across 908 source videos by design. If you only need an end-to-end smoke test, download `data/` plus a subset of videos with `--include` patterns, and use `--total 200` and `--max-rows` on each converter.

**Can I fine-tune on the video data for my product demo?** No — FAIR Noncommercial Research, third-party YouTube content.

**`gpt` values look like objects, not strings.** That is the raw image dataset layout (`{"reasoning","answer"}` dicts). The converter is what turns them into `🥵…🥵/` strings and then into the per-model native format; don't pre-transform them.

**I pasted Qwen `🥵` tags into a Gemma prompt and it output tags as text.** Gemma 4 has its own thought channel and does not understand a literal `🥵`. Keep tags in the canonical file and let FoundationScale convert; never bake a model's tags into the JSONL.

---

## Next

**Next:** [03 — Training configuration](03_training_configuration.md)