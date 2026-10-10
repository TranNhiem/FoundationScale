# 02 — Data preparation: LibriSpeech to FLAC + JSONL, and why the manifest looks like it does

This chapter turns raw LibriSpeech into the exact format FoundationScale's speech trainer consumes: one
16 kHz mono FLAC file per utterance plus one JSONL manifest row per utterance. The converter is
`scripts/prepare_librispeech.py`, and every number below was measured end-to-end on the real 2x H200 VM
this tutorial targets.

**Time needed:** ~25 minutes (the HuggingFace download is the slow part at ~2.5 minutes for 14 shards;
converting the full 28,539-row train split takes a few minutes).

All commands are run **from the repository root**.

## 0. Know the canonical format

One row per utterance, one JSON object per line:

```json
{"id": "1272-128104-0000", "text": "Transcribe this audio exactly.", "answer": "<transcript>", "audio": "/abs/path/to/file.flac", "duration": 5.855}
```

| Field | Type | Meaning |
| --- | --- | --- |
| `id` | string | Globally unique utterance id. LibriSpeech's own `{speaker}-{chapter}-{utterance}` ids already are, so we keep them verbatim. |
| `text` | string | The fixed literal prompt `"Transcribe this audio exactly."` — the training instruction. It is **never** derived from the data. |
| `answer` | string | The ground-truth transcript, **sentence-cased**: whitespace collapsed, lower-cased, then only the first character upper-cased (`" ".join(text.split()).lower().capitalize()`) — **no** punctuation added. |
| `audio` | string | **Absolute** path to a 16 kHz mono FLAC file on disk. |
| `duration` | float | The real decoded duration in seconds, recomputed from the audio samples — never trusted from dataset metadata. |

Things worth internalising now:

- **Audio is NEVER resampled and NEVER channel-mixed by this pipeline.** A clip that is not already
  16 kHz mono is skipped and counted, not silently converted. This mirrors FoundationScale's own
  training-time "never resample" policy in `src/foundationscale/train/audio.py`: the model should never be
  quietly handed audio whose provenance the manifest does not describe.
- `text` is a constant. If you ever see a manifest where `text` varies per row, something upstream has
  leaked the transcript into the prompt.
- **`answer` is sentence-cased, not LibriSpeech's raw upper case** (`"Marguerite to be unable..."`, not
  `"MARGUERITE TO BE UNABLE..."`). This is a MEASURED fix, not a style choice: training Whisper-large-v3 on
  raw upper-case targets pushes it onto upper-case BPE pieces it rarely uses, fragmenting words it already
  transcribes correctly zero-shot (`"Sweetwater"` -> `"SWEET WATER"`) and regressing dev-clean WER from
  2.25% to 3.32%. Switching the TRAINING target to sentence case alone took it to **1.72%** — see
  chapter 03 §6 for the full before/after. This repo's own `speech_p3`/`speech_p6` campaigns always used
  sentence-cased targets; this tutorial's early rounds did not, and paid the measured cost above.
- The **eval** manifest (§3 below) is intentionally left with LibriSpeech's raw upper-case reference text
  — it is NOT rebuilt with this change. `eval_wer.py`'s scorer casefolds both the reference and the
  hypothesis before computing WER, so an upper-case reference scores identically to a sentence-cased one;
  only the training target's case mattered.
- `audio` is absolute at write time. Move the audio directory and the manifest goes stale (see
  Troubleshooting).

## 1. Get the data (already on disk for this tutorial's VM)

This tutorial uses LibriSpeech (`openslr/librispeech_asr`, CC-BY-4.0) from HuggingFace, config `"clean"`,
as **parquet shards** — not the classic `.flac` + `.trans.txt` directory layout. The parquet export embeds
each utterance's audio as a struct column with `bytes` + `path` keys, which
`scripts/prepare_librispeech.py` decodes directly with `soundfile` + `pyarrow`. There is no `datasets`
library dependency.

The full `train-clean-100` split is **14 parquet shards** (`0000.parquet` … `0013.parquet`). Fetch all of
them plus the validation split:

```bash
hf download openslr/librispeech_asr \
    --repo-type dataset \
    --include "clean/train.100/*.parquet" \
    --include "clean/validation/*" \
    --local-dir data/librispeech
```

Measured on this VM: all 14 shards fetched in **~2.5 min**.

The parquet schema, introspected on the real downloaded files:

```
file: string
audio: struct<bytes: binary, path: string>
text: string
speaker_id: int64
chapter_id: int64
id: string
```

All 14 `train.100` shards together give **28,539 raw rows** (100.59 h of audio). The `validation` split
(HF config `"clean"`) **is** dev-clean: **2,703 raw rows** before filtering.

## 2. Convert: build the training manifest

The primary recipe builds from **all** of `clean/train.100` — all 14 shards — into
`data/librispeech/manifests/train_full.jsonl`. This is the manifest `configs/*.env`'s `DATASET` points at.

```bash
python examples/speech-2xH200/scripts/prepare_librispeech.py \
    --input data/librispeech/clean/train.100 \
    --audio-out-dir data/librispeech/flac/train_full \
    --manifest-out data/librispeech/manifests/train_full.jsonl \
    --max-duration 25
```

**Check it worked.** The script prints one JSON summary line; on this VM it is exactly:

```
{"shards": 14, "rows_seen": 28539, "rows_written": 28539, "skipped": {"bad_sample_rate": 0, "decode_error": 0, "duplicate_id": 0, "not_mono": 0, "too_long": 0, "too_short": 0}, "total_hours": 100.59, "min_duration": 1.41, "max_duration": 24.525}
```

Every one of the 28,539 `train.100` rows survives the `<= 25 s` filter untouched — train-clean-100 is
already short-form by construction, so `rows_written == rows_seen` here. That is **not** generally true for
longer corpora, and the `skipped` reason counters exist for exactly that case: they tell you whether rows
were dropped for duration, sample rate, channel count, decode failure or duplicate id, rather than leaving
you to guess.

The totals are a useful independent cross-check: 28,539 rows / 100.59 h with durations in 1.41–24.525 s
matches this dataset's known "100.6 h, 1–25 s" statistic exactly.

### Optional: a 2-shard quick-smoke manifest

For a **faster first pass** — less disk, less decode time — you can point `--input` at just two of the 14
shards instead of the whole directory:

```
data/librispeech/clean/train.100/0000.parquet data/librispeech/clean/train.100/0001.parquet
```

```bash
python examples/speech-2xH200/scripts/prepare_librispeech.py \
    --input data/librispeech/clean/train.100/0000.parquet data/librispeech/clean/train.100/0001.parquet \
    --audio-out-dir data/librispeech/flac/train \
    --manifest-out data/librispeech/manifests/train.jsonl \
    --max-duration 25
```

That gives **4,078 rows / 14.75 h** instead (2,039 rows per shard):

```
{"shards": 2, "rows_seen": 4078, "rows_written": 4078, "skipped": {"bad_sample_rate": 0, "decode_error": 0, "duplicate_id": 0, "not_mono": 0, "too_long": 0, "too_short": 0}, "total_hours": 14.75, "min_duration": 1.775, "max_duration": 17.275}
```

Note the **different manifest path** (`data/librispeech/manifests/train.jsonl`, not `train_full.jsonl`) and
the matching `--audio-out-dir` (`flac/train`, not `flac/train_full`), so the quick-smoke build cannot
collide with the full one. If you use this variant, remember to point `DATASET` at `train.jsonl` yourself —
`configs/*.env` ships pointing at `train_full.jsonl`.

This quick-smoke manifest is **for experimentation only**. The README's measured numbers come from the
**full** manifest.

## 3. Convert: build the full dev-clean eval manifest

Same command shape, pointed at `clean/validation`, with the **same** `--max-duration 25` policy. The eval
set is filtered identically to train, which is why train-vs-eval comparisons are apples-to-apples.

```bash
python examples/speech-2xH200/scripts/prepare_librispeech.py \
    --input data/librispeech/clean/validation \
    --audio-out-dir data/librispeech/flac/eval \
    --manifest-out data/librispeech/manifests/eval.jsonl \
    --max-duration 25
```

**Check it worked.** Measured on this VM: of 2,703 raw dev-clean rows, **2,683 survive the filter** (20 are
skipped for exceeding 25 s, counted under `too_long`), totalling **5.22 h** over 2,683 rows:

```
{"shards": 1, "rows_seen": 2703, "rows_written": 2683, "skipped": {"bad_sample_rate": 0, "decode_error": 0, "duplicate_id": 0, "not_mono": 0, "too_long": 20, "too_short": 0}, "total_hours": 5.22, "min_duration": 1.445, "max_duration": 24.8}
```

That 2,683 is a useful cross-check: it exactly matches `validation_campaigns/speech_p7`'s independently
built "dev-clean full" eval set size (2,683 rows). Two different builders of the same public dataset
landing on the same row count is reassuring evidence that neither is silently dropping or double-counting
utterances.

## 4. Why >= 2000-row eval sets

This repo's own measured finding: 300-row held-out sets cannot resolve WER shifts smaller than about 1
point — there are too few utterances for the paired-bootstrap confidence interval to narrow enough to
separate a real regression from noise. That is why this tutorial always evaluates on the full 2,683-row
dev-clean set rather than a convenient 300-row slice. Chapter 04 (evaluation) elaborates this with the
actual paired-bootstrap numbers.

## 5. Dataset license and scope

LibriSpeech is CC-BY-4.0 (`openslr/librispeech_asr`). This tutorial trains on the full train-clean-100
split (all 14 of its parquet shards, 28,539 utterances / 100.59 h) and evaluates on the full dev-clean
split. `validation_campaigns/speech_p5`..`p8` in this repository already explore AMI (CC-BY-4.0, a hard
out-of-domain test) and Earnings-22 at much larger scale if you want a harder benchmark; this tutorial does
not re-run those — they are out of scope for a quick-start.

## Troubleshooting

- **`rows_written` is 0 / exit code 2** → wrong `--input` path, or a schema the script's
  `_decode_audio_cell` does not recognise. It raises `RuntimeError: unsupported audio cell schema` on the
  **first** bad row rather than silently writing nothing — check stderr before assuming the data is empty.
- **Manifest references files that don't exist** → `--audio-out-dir` was moved after the manifest was
  built. Paths are absolute at write time and do not update; rebuild the manifest (or move the audio back).
- **Training refuses with a missing dataset file** → the `DATASET` path in `configs/*.env` is relative to
  wherever you run `train_speech.sh` from. Run it from the repository root, the same convention as
  `../vlm-2xH200`, and confirm you are in the right directory. Note that `configs/*.env` points at
  `manifests/train_full.jsonl`; if you built only the quick-smoke manifest, point `DATASET` at
  `manifests/train.jsonl` instead.

**Next:** [03 — Training](03_training.md)
