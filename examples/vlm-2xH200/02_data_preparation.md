# 02 — Data preparation: image-text, video-text and text-only, and mixing them

In this chapter you will get three sample datasets onto the VM, convert every one of them into the single JSONL format FoundationScale trains on, mix the three with weighted sampling, and read the resulting statistics. Along the way you will set the few environment variables that control video frame sampling and the handling of over-long conversations.

**Time needed:** TODO(verify) — the video dataset alone is ~116 GB to download.

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
| `start`, `end` | optional | Clip segment in seconds inside the full video. |

Things that are worth internalising now, because they affect what your data must look like:

- **Reasoning is written inline** as `🥵…🥵/…answer`. You do **not** hand-convert it: FoundationScale translates it into the target model's native format. **Gemma 4 uses its own thought channel and does *not* understand a literal `🥵` tag** — `Qwen3.6` uses `🥵`. Never tape one family's tags onto the other by hand.
- **Reasoning is kept only on the *final* assistant turn.** Both model families' chat templates drop reasoning from earlier turns. So in the 10-turn Vietnamese document rows and the 6-turn video rows, only the last answer's reasoning is actually trained.
- **Loss is computed on assistant turns only.** System and human turns are context, not targets.
- The converter **does not require you to write `<image>` yourself**: the image-text sample has no markers, and the converter inserts them. If you build custom data, write markers explicitly and expect them to be kept as-is.

---

## 1. Get the converter

```bash
python examples/vlm-2xH200/data/prepare_data.py --help
```

Check it worked:

```
usage: prepare_data.py [-h] {image,video,text,mix,stats} ...

Convert raw datasets into one canonical JSONL format and mix them.

positional arguments:
  {image,video,text,mix,stats}
    image               Convert a VDoc-style image dataset.
    video               Convert an Action-100M-style video dataset.
    text                Convert a text-only QA dataset.
    mix                 Mix canonical JSONL files with weighted sampling.
    stats               Print statistics on a canonical JSONL file.

options:
  -h, --help            show this help message and exit
```

Each subcommand takes `--max-rows N` (stop after N *valid* rows) and `--no-check-files` (skip media existence checks). Full help for each subcommand is in the details blocks below if you want to look up flags later.

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
<summary><code>mix</code> / <code>stats</code> flag reference</summary>

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
usage: prepare_data.py stats [-h] --input INPUT

options:
  -h, --help     show this help message and exit
  --input INPUT  Canonical JSONL file to analyse.
```

</details>

---

## 2. Image-text: Vietnamese document images with reasoning

**What it is:** `trannhiem/TranNhiem-Vietnamese-DocumentImage-Reasoning` on the Hub — 64,516 records of Vietnamese document images with 10-turn conversations. The repository contains a `data/vdoc.jsonl` metadata file plus **6 image tar shards under `shards/`**. The `gpt` turns are `{"reasoning","answer"}` dicts, which the converter unfolds into the `🥵…🥵/…answer` form for you.

Note: 64,516 is the metadata count. The measured baseline below uses **only shard 0**, which is why you get 11,000 rows and 53,516 skips.

### Download

```bash
hf download trannhiem/TranNhiem-Vietnamese-DocumentImage-Reasoning \
  --repo-type dataset \
  --local-dir data/raw/image
```

### Extract the image shards

Shards are tars; extract one per shard **inside `data/raw/image/`** so that the relative image paths inside `data/vdoc.jsonl` resolve. For a fast start, extract shard 0 only (TODO(verify) exact shard filenames inside `shards/`):

```bash
cd data/raw/image
for s in shards/*.tar; do tar xf "$s"; done   # all 6 shards
cd ../..
```

or just the first one while you iterate:

```bash
cd data/raw/image
tar xf shards/shard_0.tar   # TODO(verify) exact filename
cd ../..
```

Partial extraction is fine on purpose: the converter checks that each image file exists and **skips rows whose image is missing**, so a half-downloaded dataset still yields a usable training file.

### Convert

```bash
python examples/vlm-2xH200/data/prepare_data.py image \
  --input data/raw/image/data/vdoc.jsonl \
  --image-root data/raw/image \
  --output data/canon/image.jsonl
```

Check it worked (measured with shard 0 extracted):

```
rows written: 11000, rows skipped: 53516 (missing_media: 53516)
```

If you later extract the remaining shards and re-run, `rows written` goes up and `missing_media` goes down. TODO(verify) exact numbers with all 6 shards.

---

## 3. Video-text: human activity clips with segment annotations

**What it is:** `trannhiem/TranNhiem-Action100M-Human-Activities` — 2,985 records over 908 YouTube source videos, ~116 GB in total. Each record describes a `[start, end]` clip inside a full video (median 8.7 s), with a 6-turn English conversation and `🥵…🥵/` reasoning. TODO(verify) the final repo layout.

**Read the licence first:** the dataset is under **FAIR Noncommercial Research** terms and the videos are third-party YouTube content. Noncommercial research use only.

```bash
hf download trannhiem/TranNhiem-Action100M-Human-Activities \
  --repo-type dataset \
  --local-dir data/raw/video     # TODO(verify) + layout: metadata file name, video files location
```

TODO(verify) extraction/layout command for this dataset (mirror the image flow once the layout is fixed).

Then convert — `--video-root` is the directory that the **basenames** in the metadata are resolved against:

```bash
python examples/vlm-2xH200/data/prepare_data.py video \
  --input data/raw/video/TODO.jsonl \
  --video-root data/raw/video \
  --output data/canon/video.jsonl
```

Check it worked: expect 2,985 written rows.

```
rows written: 2985, rows skipped: …   # TODO(verify) exact line wording when nothing is skipped
```

Each converted row carries `start`/`end` in seconds, so the trainer knows which segment to sample frames from.

---

## 4. Text-only: ShareGPT-style QA

**What it is:** any ShareGPT jsonl works. The competition sample is a small Vietnamese persona logic/math reasoning file: 209 rows, 2-turn conversations, `🥵…🥵/` reasoning, median ~5,700 characters per row. It is supplied by the organisers and is **not** on the Hub.

```bash
python examples/vlm-2xH200/data/prepare_data.py text \
  --input data/raw/text/persona_logic_math.jsonl \   # TODO(verify) organiser-supplied filename and path
  --output data/canon/text.jsonl
```

Check it worked: 209 written rows.

```
rows written: 209, rows skipped: …   # TODO(verify) exact line wording
```

---

## 5. Mix the three datasets

`mix` does weighted sampling over the inputs to produce exactly `--total` rows, then shuffles. Weights are relative — `0.4 0.4 0.2` means 40 % / 40 % / 20 % of the output.

```bash
python examples/vlm-2xH200/data/prepare_data.py mix \
  --inputs data/canon/image.jsonl data/canon/video.jsonl data/canon/text.jsonl \
  --weights 0.4 0.4 0.2 \
  --total 2000 \
  --output data/canon/mix.jsonl \
  --seed 0
```

Check it worked (measured):

- `image` 800 rows / `video` 800 rows / `text` 400 rows
- a warning telling you that `text` (209 rows) **was repeated to fill 400** (TODO(verify) the exact warning wording)

### Reading the repeat warning

The text pool only has 209 rows and you asked for 400, so the sampler has to reuse rows. That is not an error, but it deserves attention:

- Rows copied more than once add no information; they just weight those conversations harder. The model starts to memorise 209 rows instead of seeing 400 different ones.
- Every repeat is a signal that your `--weights` are asking for more of a pool than the pool holds. Prefer lowering the text weight (or `--total`) over silently doubling rows.
- Keep an eye on the same thing for the other pools once you use the full image shard set.

### Choosing weights

Start from what is scarce and what you want the evaluated models to do well at. Image-text document reasoning and video understanding dominate this sample, so 40 % / 40 % is the given baseline; text is the anchor preventing the model from degenerating into pure VQA. Then rerun `stats` (next section) and adjust. There is no single correct split — just don't let the weights drift away from what the competition actually evaluates. TODO(verify) what the evaluation emphasises.

### Keep a held-out split

Do not evaluate on the rows you trained on. Produce a second mix with a different `--seed` (or use a split step; TODO(verify) the exact command — e.g. a dedicated `split` subcommand or a documented convention) and keep that untouched:

```bash
python examples/vlm-2xH200/data/prepare_data.py mix \
  --inputs data/canon/image.jsonl data/canon/video.jsonl data/canon/text.jsonl \
  --weights 0.4 0.4 0.2 \
  --total 2000 \
  --output data/canon/mix_eval.jsonl \
  --seed 1     # TODO(verify) confirm non-overlap with seed 0 before trusting this as "held out"
```

Two seeds sample from the *same* pools, so they can overlap. If exact disjointness matters, split each canonical file into train/eval first. TODO(verify).

---

## 6. Sanity-check the mix

```bash
python examples/vlm-2xH200/data/prepare_data.py stats --input data/canon/mix.jsonl
```

Check it worked (measured for the 2,000-row mix above):

- turns per row: `2: 428`, `6: 772`, `10: 800`
- rows with reasoning: `2000`
- median characters per row: `8932`

Two things to notice: the three sources contribute distinct conversation lengths (text = 2, video = 6, image = 10 turns), and reasoning rows are guaranteed only on the **final** assistant turn of each conversation regardless of those lengths.

---

## 7. Declare a video frame budget

Video frames are sampled **per clip, inside `[start, end]`**, and the frame budget is **declared by you — there is no hidden default**. Set these before training:

```bash
export FOUNDATIONSCALE_TRAIN_VIDEO_FRAMES=16             # frame budget per clip; use 16 to match the sample setup
export FOUNDATIONSCALE_TRAIN_VIDEO_SAMPLING=uniform      # sampling pattern across the clip
export FOUNDATIONSCALE_TRAIN_VIDEO_MAX_SIDE=…            # TODO(verify) value and meaning (longest side in pixels)
export FOUNDATIONSCALE_TRAIN_VIDEO_CACHE_DIR=…           # TODO(verify) path; frames are cached here across runs
export FOUNDATIONSCALE_TRAIN_VIDEO_FRAMES FOUNDATIONSCALE_TRAIN_VIDEO_SAMPLING
```

Token cost per frame depends on the target model (TODO(verify) the per-model numbers) — budget your `--total` and frame count against the context limit in Section 8.

Segment support (`start`/`end`) is being finalised: TODO(verify) what is honoured at training time.

---

## 8. Decide what happens to over-long conversations

Once images and frames are expanded into tokens, some conversations will exceed the maximum sequence length. Training **never truncates**: it either drops those rows or refuses to start. You must pick one explicitly:

```bash
export FOUNDATIONSCALE_TRAIN_OVERLONG=drop    # or: refuse
```

| Value | Behaviour | When to use |
|---|---|---|
| `drop` | Over-long rows are removed from training. | You have plenty of rows and a few long ones are not worth shrinking the dataset. |
| `refuse` | Training refuses to start if any row is over-long. | You want to catch a mis-configured frame budget or context length before burning GPU time. |

Use `refuse` for your first run so a bad setting shows up immediately; switch to `drop` once you know your longest rows.

---

## Troubleshooting

**My image conversion says `rows skipped: … (missing_media: …)`.** Expected when you extracted fewer than all 6 shards. Each counted row points at an image file that is not on disk yet. Extract more shards and re-run — the skipped rows come back. Only use `--no-check-files` if you are sure the files exist at training time; otherwise the failure moves (lazily) into the training run.

**`mix` warns that a source was repeated.** You asked for more rows than that pool contains. Lower that source's weight, lower `--total`, or widen the pool. See Section 5.

**Nothing seems to happen with videos / frames are empty.** Check `FOUNDATIONSCALE_TRAIN_VIDEO_FRAMES` — the frame budget has no default. TODO(verify) whether it errors or silently produces zero frames.

**A training row is much longer than expected.** Remember frames count as tokens and the token cost per frame is model-dependent (TODO(verify)). Rows are not truncated; with `FOUNDATIONSCALE_TRAIN_OVERLONG=drop` they silently disappear from the run.

**The video download is enormous.** It is ~116 GB across 908 source videos by design. If you only need an end-to-end smoke test, start `--total 200` and `--max-rows` on each converter.

**Can I fine-tune on the video data for my product demo?** No — FAIR Noncommercial Research, third-party YouTube content.

**`gpt` values look like objects, not strings.** That is the raw image dataset layout (`{"reasoning","answer"}` dicts). The converter is what turns them into `🥵…🥵/` strings and then into the per-model native format; don't pre-transform them.

**I pasted Qwen `🥵` tags into a Gemma prompt and it output tags as text.** Gemma 4 has its own thought channel and does not understand a literal `🥵`. Keep tags in the canonical file and let FoundationScale convert; never bake a model's tags into the JSONL.

---

## Next

[03_training.md](./03_training.md) — TODO(verify) chapter 03's title and file name (data manifest handed to the trainer / first training run on 2x H200).
</parameter>