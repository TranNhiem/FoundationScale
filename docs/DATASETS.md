# Datasets in FoundationScale

README §13 states the shape of dataset integration in one sentence: datasets reach
training through the `[train]` extra's `datasets` library via `transformers.Trainer`,
and FoundationScale adds **no dataset abstraction of its own today**. This document is
the chapter behind that sentence — what is accepted, where it is loaded, what the thin
path demands of your data, what a run records about it, and what is explicitly not
implemented.

---

## 1. There is no dataset layer

This is the load-bearing fact and it is stated first so it is not missed:

**FoundationScale does not ship a dataset class, registry, loader abstraction, sampler,
or data-parallel input pipeline of its own.** There is no `FoundationScaleDataset`, no
`register_dataset`, no dataset config schema. Data is loaded by `datasets` (pulled in by
the `train` extra alongside torch, transformers and accelerate) and consumed by
`transformers.Trainer`. The thin training loop sits in front of that machinery: it loads
your data with plain `datasets.load_dataset` calls, tokenizes it, hands it to the
Trainer, and wraps the run in gates.

A first-class dataset layer **is** designed — the audit's "one data contract" in
`docs/deliverables/B2_scaling.md` — and it is **unimplemented**. The current behaviour is
the thin path described here. Anything you read in this repository's examples about
datasets is that thin path; do not look for an entry point that does not exist.

---

## 2. What `--dataset` / `TrainConfig.dataset` accepts

`TrainConfig.dataset` (the CLI's `--dataset` flag) accepts three reference forms, all
resolved by `_load_raw_dataset` in `src/foundationscale/train/loop.py`:

| reference form | resolution |
|---|---|
| a `.json` or `.jsonl` file | `datasets.load_dataset("json", data_files=<path>)` |
| a directory | every `*.json*` file in it, sorted, loaded the same way |
| anything else | treated as a Hugging Face dataset id and passed to `datasets.load_dataset(<ref>)` |

The dispatch is by filesystem shape: a path that ends in `.json`/`.jsonl` or names a
directory takes the local route; everything else falls through to the Hub. Whatever it
resolves to, **the dataset must expose a `text` column**. That is the whole contract.

A 16-row toy corpus ships in the repository at `examples/data/toy_text.jsonl` purely so
the offline recipe needs no dataset download. Each line is one JSON object with one
`text` field — it doubles as a minimal specification of what the thin path consumes:

```json
{"text": "Exit codes are 0 PASS, 5 RED, 95 UNMEASURED, 96 REFUSE."}
```

---

## 3. Loading in practice

### Path B: the offline CLI against the toy corpus

The canonical local-file invocation is the offline one from docs/TRAINING.md §4:

```bash
HF_HUB_OFFLINE=1 python -m foundationscale.train.cli \
  --model sshleifer/tiny-gpt2 \
  --dataset examples/data/toy_text.jsonl \
  --output-dir /tmp/fs_train_demo \
  --profile-name local-single-node \
  --nodes 1 --gpus-per-node 1 --dp 1 \
  --max-steps 8 --save-interval 4
```

`HF_HUB_OFFLINE=1` covers the dataset half outright: the local `json` builder needs no
network. The model is still an HF id, so the first run needs the Hub once; after that it
is served from the local cache and the command is fully offline. Point `--model` at a
local directory to remove the Hub entirely. This run was measured at `PASS`, `rc=0`,
with gated checkpoints at `checkpoint-4` and `checkpoint-8`.

### Path A: an HF id from the Hub

`examples/train_tiny.py` trains on `fancyzhx/ag_news`, referenced by id:

```
[fs:train:data]          120000 examples tokenized (split=train, max_length=128)
```

One version-related property matters here: `datasets` ≥ 5 rejects bare legacy ids with
`HfUriError: Repository id must be 'namespace/name'`. Use `fancyzhx/ag_news`, not
`ag_news`.

---

## 4. Inside the loop: what actually happens to your data

Once resolved by `_load_raw_dataset`, the loaded object goes through three explicit
steps inside `train()`:

1. **Split selection.** The loop prefers the `train` split; if the dataset has no
   `train` split, it takes the first available split.
2. **The `text` column check.** Before any tokenization, the selected split's column
   names are inspected. If `text` is absent, this is not a training failure — it is a
   **refusal**, before anything has started:

   ```
   dataset 'examples/data/toy_text.jsonl' split 'train' has columns ['sentence'];
   the thin path requires a 'text' column
   ```

   The run exits `96 REFUSE` with that message. The fix is on your side: rename or map
   your column to `text`. There is no flag to name a different column; the thin path
   has exactly one input contract.
3. **Tokenization.** The selected split is mapped in batches through the model's
   tokenizer with `truncation=True` and `max_length=128`, with the original columns
   removed. The tokenized count is reported on the `[fs:train:data]` line shown above.

If model or dataset *construction* itself raises — a failed download, a malformed file
— that is caught separately and reported as `5 RED`
(`model/dataset construction failed: ...`). The distinction is deliberate: a dataset
that exists but violates the contract is a refusal; a dataset that cannot be built at
all is a caught failure.

The tokenizer's pad token, if unset, is filled from its EOS token before training.

---

## 5. What the run records about the dataset

Provenance about data is not kept in your head. The run manifest records the dataset
reference as validated, in the `config` payload written to `run_manifest.json`:

```json
"config": {
    "model": "sshleifer/tiny-gpt2",
    "dataset": "examples/data/toy_text.jsonl",
    "output_dir": "...",
    "max_steps": 8, "...": "..."
}
```

That means the manifest's `argv` and recorded `config.dataset` are sufficient to
repoint a rerun at the identical data reference — the reproducibility claim the toy
corpus's own rows repeat: a run manifest can rebuild the argv that produced it. Note
what is and is not recorded: the manifest holds the *reference* and the validated
configuration, not the dataset contents. A local file that changes between runs is not
detected by this field.

---

## 6. Offsets and limits of the thin path

| question | answer |
|---|---|
| Streaming datasets | Not implemented; the current behaviour is an in-memory map over the selected split. |
| Column mapping to `text` | Not implemented; the loop refuses (`96 REFUSE`) if `text` is absent. Rename or map beforehand. |
| Multiple splits / eval split selection | The loop picks `train`, else the first split. There is no flag for it today. |
| Non-text tasks (vision, audio) | Not supported by the thin path; the `text` contract is unconditional. |
| The audit's "one data contract" | Designed in `docs/deliverables/B2_scaling.md`, unimplemented. |
| Sharding input across DDP ranks | Handled downstream by `transformers.Trainer`, not by any FoundationScale layer. |

The last row deserves emphasis in both directions. FoundationScale adding no dataset
abstraction means it also adds no *reimplementation*: the tokenized dataset reaches the
Trainer unwrapped, and whatever the Trainer does with it — batching, collation via
`DataCollatorForLanguageModeling`, distribution — is `transformers` behaviour, not
framework behaviour. Dataset integration is honest rather than absent: unmeasured
claims, if they arose about data, would report `95 UNMEASURED`, never clean.

---

## 7. When it does not work

| symptom | cause |
|---|---|
| `HfUriError: Repository id must be 'namespace/name'` | `datasets` ≥ 5 rejects bare legacy ids — use `fancyzhx/ag_news`, not `ag_news` |
| `dataset ... has columns [...]; the thin path requires a 'text' column` → `96 REFUSE` | rename or map your column to `text` |
| `model/dataset construction failed: ...` → `5 RED` | download or file construction failed |
| `96 REFUSE`, message names `pip install 'foundationscale[train]'` | the `train` extra (which brings `datasets`) is not installed |

Construction failures are RED rather than REFUSE because the request was well-formed and
the operation failed; a missing `text` column is REFUSE because the request itself
cannot be satisfied as typed.

---

## 8. Checking data coherence without an allocation

You do not need to start training to find out whether a request is coherent. The same
CLI with `--dry-run` runs the whole prologue — profile resolution, topology arithmetic,
validation — and stops before importing torch. Dataset loading happens after that
prologue, so `--dry-run` will *not* prove your `text` column exists; it proves the
launch request is coherent without holding an allocation. Validating the data column
itself costs one actual run, which is cheap against `examples/data/toy_text.jsonl` and
the reason that file exists.
