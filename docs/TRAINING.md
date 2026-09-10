# Training with FoundationScale

The worked end-to-end recipe README §10 defers to. Two paths are shown, both run and
measured on the machine that wrote this document, both exiting `0 PASS`:

1. **The example** — `python3 examples/train_tiny.py`, model and dataset from the Hub.
2. **The CLI, offline** — `python3 -m foundationscale.train.cli` against a toy `.jsonl`
   that ships in this repository, with `HF_HUB_OFFLINE=1`.

Everything below is transcript, not illustration. Where output is quoted it was copied
from the run, not written by hand.

---

## 1. Install

The base install in README §8 is deliberately CPU-torch and has **no** `train` extra: it
is what CI runs to exercise the gate contract. To train you need the extra, and on a GPU
box you need a torch build for your accelerator rather than the CPU wheel:

```bash
git clone https://github.com/TranNhiem/FoundationScale && cd FoundationScale
python3 -m pip install -e ".[train]"
```

`train` pulls torch, transformers, datasets and accelerate. Without it the loop refuses
before importing anything — `96 REFUSE` with the install command in the message, not a
half-started run and an `ImportError` three minutes in.

Verify the entry point resolves before going further:

```bash
python3 -m foundationscale.train.cli --help
```

---

## 2. The exit-code contract

Every path below returns one of four codes. They are the reason a FoundationScale run is
worth wrapping in a script:

| code | meaning |
|---|---|
| `0` | PASS — training ran and every gate that could adjudicate did, and cleared |
| `5` | RED — something is wrong and was caught. Usually *before* an allocation is burned |
| `95` | UNMEASURED — a claim could not be measured. Explicitly not the same as clean |
| `96` | REFUSE — the request is malformed or a dependency is absent; nothing was started |

`95` is the one worth internalising. A check that could not run reports `UNMEASURED`; it
never reports success. See the `[fs:train:partition]` line in the transcript below for
what that looks like in practice.

---

## 3. Path A — the example, model and dataset from the Hub

```bash
python3 examples/train_tiny.py
```

That is the whole command. `examples/train_tiny.py` is one screen: a `ClusterProfile`
describing the machine, and a `TrainConfig` naming a model, a dataset and a topology.
It trains `sshleifer/tiny-gpt2` (a few MB) on `fancyzhx/ag_news` for 20 steps.

### To run it on more than one GPU

Edit **one** value at the top of the file:

```python
GPUS = 1   # -> 4 on a 4-GPU box, 8 on an 8-GPU node
```

then launch under torchrun with a matching process count:

```bash
torchrun --nproc_per_node=4 examples/train_tiny.py
```

The profile's `gpus_per_node`, the declared `gpus_per_node` and the DDP degree `dp` all
read that one constant, so they cannot disagree with each other. They *can* disagree with
reality — and that is deliberate. `train()` compares the topology you **declared** against
the one torchrun actually built, and a declaration copied out of `WORLD_SIZE` would be a
comparator that can never disagree with itself. Declare it; let the framework check you.

Mismatch is caught in the prologue, before a GPU is touched:

```
[fs:train:consistency]   no torchrun runtime (WORLD_SIZE unset); declared-vs-effective comparison skipped on the driver
```

### What a clean run prints

```
[fs:train:start]         model=sshleifer/tiny-gpt2 dataset=fancyzhx/ag_news output_dir=out/train_tiny dry_run=False
[fs:train:topology]      topology: 1 GPUs = 1 nodes x 1 GPUs/node
  decomposition : dp=1 x tp=1 x pp=1 x ep=1 x cp=1 = 1
  model replica : tp x pp x ep x cp = 1 GPUs -> dp=1 independent replicas
  per node      : 1 GPUs, 1 tasks (must be equal; a gap is the Duplicate-GPU crash, caught here, not 2m10s in)
[fs:train:profile]       example-single-node: scheduler=slurm gpus_per_node=1
[fs:train:partition]     UNMEASURED: no launch corpus supplied (--launch-corpus), so partition spelling was not compared...
[fs:train:validated]     [   ok] topology.validate_summary: validate_against ran 5 checks against profile 'example-single-node': 0 blocking, 0 warnings...
[fs:train:deps]          importing torch/transformers/datasets (optional extra 'foundationscale[train]')
[fs:train:data]          120000 examples tokenized (split=train, max_length=128)
[fs:train:manifest]      declared checkpoint: dense: config declares none of ['num_experts', ...], and 0 of 28 declared tensors carry an expert path segment
[fs:train:trainer]       transformers.Trainer constructed (single-node DDP is automatic under torchrun); FoundationScaleSaveGate attached
[fs:train:manifest]      run manifest (train) -> out/train_tiny/run_manifest.json
[fs:train:run]           training starts
[fs:train:save_gate]     PASS 4/4 gates over out/train_tiny/checkpoint-10
[fs:train:save_gate]     PASS 3/3 gates over out/train_tiny/checkpoint-20
[fs:train:saved]         final checkpoint -> out/train_tiny/final (1 safetensors shard(s))
[fs:train:adjudicate]    gates @ save: 3 run — all clear (1 of 3 verified; 2 declared SKIP)
  [   skip] checkpoint.expert_distinctness: 0 units — context declares no experts and none are present (dense model)
  [   skip] checkpoint.expert_bytes: 0 units — context declares no experts and none are present (dense model)
  [     ok] checkpoint.save_complete: 28/28 tensors — all 28 declared tensors present (excluding 0 _extra_state metadata blobs)
  — 3 gates ran of 3 registered for save
[fs:train:manifest]      run manifest (done) -> out/train_tiny/run_manifest.json
[fs:train:done]          PASS
```

Read the adjudication tail closely, because its shape is the whole argument of this
project. Three gates ran; **one** verified, **two** declared SKIP and said why. A
framework that printed "all clear" over three skips would be telling you nothing while
sounding like it told you something. `1 of 3 verified` is the honest denominator.

Output lands in `out/train_tiny/` (git-ignored).

---

## 4. Path B — the CLI, offline, on a toy dataset

`TrainConfig.dataset` accepts an HF id, a `.json`/`.jsonl` file, or a directory of them.
Any of them must expose a `text` column. A 16-row toy corpus ships at
`examples/data/toy_text.jsonl` so the recipe below needs no dataset download:

```bash
HF_HUB_OFFLINE=1 python3 -m foundationscale.train.cli \
  --model sshleifer/tiny-gpt2 \
  --dataset examples/data/toy_text.jsonl \
  --output-dir /tmp/fs_train_demo \
  --profile-name local-single-node \
  --nodes 1 --gpus-per-node 1 --dp 1 \
  --max-steps 8 --save-interval 4
```

Measured result: `PASS`, `rc=0`, checkpoints at `checkpoint-4` and `checkpoint-8`
(`PASS 4/4` and `PASS 3/3` gates), a final save, and a run manifest.

`HF_HUB_OFFLINE=1` covers the dataset half outright. The model is still an HF id, so the
first run needs the Hub once; after that it is served from the local cache and the command
above is fully offline. Point `--model` at a local directory to remove the Hub entirely.

### Choosing a profile

`--profile-name` takes a built-in; `--profile-path` takes a JSON file; in Python you can
pass a `ClusterProfile` directly. Exactly one is required — a cluster profile is a machine
fact and has no default. The built-ins:

| name | gpus/node | runtime | filesystem roots |
|---|---|---|---|
| `local-single-node` | 8 | none | `/` |
| `slurm-generic` | 8 | enroot | `/home`, `/scratch` |

Neither will match your estate. Describe yours as **data** — one dict, no code changes —
as `examples/train_tiny.py` does inline, or as a JSON file for `--profile-path`.

One field is worth calling out because it has already caused a bug in this repository:

```python
"node_pattern": r"compute-0[1-8]",   # a REGEX
```

It is a Python regular expression, not a Slurm hostlist. Writing the hostlist spelling of
that same range puts a `1`-to-`0` range inside a character class, `re.compile` refuses it,
and the profile fails to construct. `ClusterProfile` catches it in `__post_init__` with a
named error — but it catches it at *your* first run, so it is worth knowing.

### Validate without training

```bash
python3 -m foundationscale.train.cli ... --dry-run
```

Runs the whole prologue — profile resolution, topology arithmetic, validation — and stops
before importing torch. This is the thing to put in front of a scheduler submission: it
answers "is this request coherent?" without holding an allocation while it finds out.

---

## 5. What a run leaves behind

```
out/train_tiny/
├── checkpoint-10/       gated at save time
├── checkpoint-20/       gated at save time
├── final/               final save
└── run_manifest.json    provenance for the whole run
```

The manifest is the artifact that makes a run auditable after the fact:

| field | what it records |
|---|---|
| `schema_version`, `run_id`, `attempt` | identity; `FS_RUN_ID` / `FS_ATTEMPT` override |
| `created_at`, `job_id` | when, and the scheduler job if there was one |
| `fingerprint` | a hash over the recorded configuration |
| `code` | the framework version and interpreter that produced this |
| `config`, `topology` | the request as validated, not as typed |
| `environment` | the environment variables that steered the run |
| `artifact_paths` | where the checkpoints went |
| `declared` | what the model *claimed* to contain — the denominator the gates checked against |
| `findings` | every gate verdict, including the SKIPs and their reasons |

`declared` and `findings` are the pair that matters. `declared` is what was expected;
`findings` is what was observed and, for anything not observed, why not.

---

## 6. Save gates

`FoundationScaleSaveGate` is attached to `transformers.Trainer` as a callback, so every
checkpoint the trainer writes is adjudicated as it is written — not discovered broken at
load time, weeks later. The gates registered for the `save` event today are checkpoint
integrity gates:

- `checkpoint.save_complete` — every declared tensor is present in the written shards
- `checkpoint.expert_distinctness` — MoE experts are not aliases of one another
- `checkpoint.expert_bytes` — expert tensors carry the bytes they claim

On a dense model the two expert gates declare SKIP and say so, with their zero denominator
stated. They do not silently pass.

To add your own, see the gate registry in `foundationscale.gates.core`; a gate that
registers for `save` is picked up with no change to the training loop.

---

## 7. When it does not work

| symptom | cause |
|---|---|
| `96 REFUSE`, message names `pip install 'foundationscale[train]'` | the `train` extra is not installed |
| `ValueError: node_pattern is not a valid regex` | `node_pattern` was written as a Slurm hostlist (§4) |
| `HfUriError: Repository id must be 'namespace/name'` | `datasets` ≥ 5 rejects bare legacy ids — use `fancyzhx/ag_news`, not `ag_news` |
| `dataset ... has columns [...]; the thin path requires a 'text' column` | rename or map your column to `text` |
| `5 RED` at `[fs:train:topology]` | the declared decomposition does not multiply out to the declared GPU count |
| declared-vs-effective mismatch under torchrun | `GPUS` in the example, or `--dp`/`--gpus-per-node`, disagrees with `--nproc_per_node` |
| `95 UNMEASURED` on partition spelling | expected on the driver; pass `--launch-corpus` to measure it |

---

## 8. Scope — what this path is and is not

The thin training path is `transformers.Trainer` DDP with the FoundationScale validation
prologue and save gates wired in. It is deliberately thin: it exists to prove the gate
plane works against a real trainer, end to end, on any model the Hub can load.

It is **not** the multi-node production launch plane. That is
`validation_campaigns/h100_validation/` (H100, Singularity/enroot, Slurm) and the GB200
launchers — those carry the tensor/pipeline/expert-parallel work, resume, and the fabric
tripwires. `--tp/--pp/--ep/--cp` are accepted and validated here, but the thin loop
executes DDP.

[-> ../README.md §14 Distributed training, §15 Multi-node training]
