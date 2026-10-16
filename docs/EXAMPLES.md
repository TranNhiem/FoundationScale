# Examples

The catalogue of `examples/` — the surface a new user runs first — with the hello-world
gate and the exit-code contract shared by every training path.

## What `examples/` contains

| path | what it is |
|---|---|
| `examples/train_tiny.py` | the thinnest real FoundationScale training path, one screen |
| `examples/data/toy_text.jsonl` | a 16-row toy corpus exposing a `text` column, so training needs no dataset download |
| `examples/vlm-2xH200/`, `examples/speech-2xH200/` | end-to-end VLM and speech fine-tuning tutorials measured on two H200s |
| `examples/SPEECH.md` | entry page for all speech work: ASR, TTS and omni speech-LLM, with the technique each example covers |
| `examples/tts-GB200/` | TTS on GB200: zero-shot voice cloning scored with Seed-TTS-eval, and VoxCPM2 Vietnamese LoRA fine-tuning |
| `examples/omni-GB200/` | omni speech-LLM on GB200: cascade (ASR then LLM) versus end-to-end Qwen3-Omni on VoiceBench |
| `examples/A100_2GPU.md` | entry page for two GPUs of an A100-80GB node: environment, GPU hygiene, measured results |
| `examples/llm-2xA100/` | text-only LLM LoRA SFT tutorial (Qwen2.5-7B-Instruct), measured on two A100s |
| `examples/speech-2xA100/`, `examples/vlm-2xA100/` | the H200 speech and VLM tutorials re-measured on two A100s |
| `examples/vlm-2xGB200/` | the H200 VLM tutorial on two GPUs of a shared GB200 tray (aarch64): launcher, configs, measured fit matrix and end-to-end run |

The runnable gate example from README §10 is not a file under `examples/`; it lives in
the README itself and in `foundationscale.gates.example`. The launchers and the harness
evidence live under `validation_campaigns/`, not `examples/` (see "What is not in
`examples/`" below).

New files are covered the day they land: the glob-tested sweep in
`tests/train/test_examples_runnable.py` keeps this catalogue current.

## The gate example — the honest hello-world

Before any training example, the artifact that exists end-to-end is a gate. This runs
verbatim against the installed package:

```python
from foundationscale.gates.core import REGISTRY, Verdict
from foundationscale.gates.example import ExpertCheckContext   # importing registers the gate
from foundationscale.gates.fixtures import make_empty_experts

gate = REGISTRY.get("checkpoint.expert_alias")
ctx = ExpertCheckContext.from_expert_set(make_empty_experts(declared_expert_count=128))
result = gate.run(ctx)
print(result.render())
assert result.verdict is Verdict.VACUOUS and result.blocking
```

The fixture declares **128** experts and provides zero of them; the gate examines zero
units and reports `VACUOUS`, **blocking** — not `PASS`. This is the founding rule in
miniature: a claim must carry its own coverage. It is told once, at length, in
[docs/ARCHITECTURE.md](ARCHITECTURE.md). The training examples below show the same shape
as SKIP verdicts over a dense model.

## `examples/train_tiny.py` — the one-screen training example

The example is deliberately one file and one screen: a `ClusterProfile` describing the
machine as **data**, and a `TrainConfig` naming a model, a dataset and a topology.

```bash
python3 -m pip install -e ".[train]"
python3 examples/train_tiny.py
```

It trains `sshleifer/tiny-gpt2` on `fancyzhx/ag_news` for 20 steps, adjudicates the
intermediate checkpoint (step 10) and the final save (step 20) via
`FoundationScaleSaveGate`, and exits `0 PASS`. Without the `train` extra the loop
refuses (`96 REFUSE`) and prints the install remedy rather than half-starting.

### One knob, three readers

The GPU count is written **once**, at the top of the file:

```python
GPUS = 1
```

The profile's `gpus_per_node`, the `TrainConfig.gpus_per_node`, and the data-parallel
degree `dp` all read it. To run on four GPUs:

```python
GPUS = 1   # -> 4 on a 4-GPU box
```

```bash
torchrun --nproc_per_node=4 examples/train_tiny.py
```

`GPUS` is deliberately **not** derived from `WORLD_SIZE`. `train()` compares the
topology you declared against the one torchrun actually built; a declaration copied out
of the runtime would be a comparator that can never disagree with itself. Mismatch is
caught in the prologue, before a GPU is touched. The one-source rule is pinned by
`test_declared_gpu_count_has_exactly_one_source` below.

### `node_pattern` is a regex

A regular expression, not a Slurm hostlist:

```python
"node_pattern": r"compute-0[1-8]",   # a REGEX, not a Slurm hostlist
```

The hostlist spelling of the same range — `"compute-[01-08]"` — contains a `1`-to-`0`
range inside a character class, which `re.compile` refuses. `ClusterProfile` validates
`node_pattern` in `__post_init__` and raises
`ValueError: node_pattern is not a valid regex`.

### Validate only, zero GPUs

Set `dry_run=True` in the config, or use the CLI with `--dry-run`:

```bash
python3 -m foundationscale.train.cli ... --dry-run
```

`--dry-run` runs the full validation prologue — profile resolution, topology arithmetic,
validation — without constructing a model. Put it in front of a scheduler submission, so
an incoherent request is rejected without holding an allocation while it finds out.

## The offline variant

For a corpus that needs no network, the shipped toy dataset drives through the console
script `foundationscale-train` (equivalently `python3 -m foundationscale.train.cli`):

```bash
HF_HUB_OFFLINE=1 python3 -m foundationscale.train.cli \
  --model sshleifer/tiny-gpt2 --dataset examples/data/toy_text.jsonl \
  --output-dir /tmp/fs_train_demo --profile-name local-single-node \
  --nodes 1 --gpus-per-node 1 --dp 1 --max-steps 8 --save-interval 4
```

`HF_HUB_OFFLINE=1` covers the dataset half; the first run still fetches the model from
the Hub once, and afterwards the command is fully offline. The loop reads `FS_RUN_ID`
and `FS_ATTEMPT` from the environment. The full recipe — transcripts, the run manifest's
fields, the save gates, the symptom→cause table — is [docs/TRAINING.md](TRAINING.md).

## Keeping `examples/` runnable

`tests/train/test_examples_runnable.py` exists so the shipped copy of an example is the
copy under test — a profile or value that diverges from the tested one breaks the sweep.

| test | what it guarantees |
|---|---|
| `test_examples_directory_is_not_empty` | the denominator is real. Every other test iterates the discovered set; if `examples/` were renamed or emptied, parametrisation would collapse to zero cases and pass green over nothing. This is the one assertion that cannot be satisfied vacuously. |
| `test_example_guards_its_entry_point` | every example keeps its work behind `if __name__ == "__main__"` — asserted, not assumed — so importing one is safe. |
| `test_example_imports_cleanly` | each example is imported as a throwaway module, executing its declarations. Constructing a `ClusterProfile` runs `__post_init__`, so this is a real check of the values, not a syntax check. |
| `test_declared_gpu_count_has_exactly_one_source` | `train_tiny.GPUS` is an `int ≥ 1` and `PROFILE.gpus_per_node` equals it — the GPU count still has one source. |
| `test_the_shipped_defect_would_now_be_caught` | MUST_FIRE control: plants the original hostlist spelling into a real copy of the example and asserts the import machinery raises `ValueError: node_pattern is not a valid regex`. Only then does it assert the shipped file is clean. |

Discovery is by filesystem glob (`*.py`, excluding names starting with `_`), so a new
example is covered the day it lands, without anyone remembering to add a test. The
imports are isolated — `sys.modules` is restored in `finally` — so a failed import
cannot leave a half-built module behind.

## Exit codes across the examples

Every training path returns one of four codes; that is what makes the examples worth
wrapping in a script:

| code | meaning |
|---|---|
| `0` | PASS — training ran and every gate that could adjudicate did, and cleared |
| `5` | RED — something is wrong and was caught, usually before an allocation is burned |
| `95` | UNMEASURED — a claim could not be measured; explicitly not the same as clean |
| `96` | REFUSE — the request is malformed or a dependency is absent; nothing was started |

The save-time adjudication keeps the honest denominator visible. On a dense model,
`checkpoint.expert_distinctness` and `checkpoint.expert_bytes` declare SKIP with their
zero denominators stated, while `checkpoint.save_complete` verifies. "1 of 3 verified"
is reported as such — not "all clear".

## What is not in `examples/`

- **The launchers.** The trainable, fabric-tripwired production launch plane is
  `validation_campaigns/h100_validation/` and the GB200 launchers. `examples/train_tiny.py`
  is explicitly the *thin* path: `transformers.Trainer` DDP with the validation prologue
  and save gates. `--tp/--pp/--ep/--cp` are accepted and validated, but the thin loop
  executes DDP.
- **The harness evidence.** Gates firing against real launches are at
  `validation_campaigns/h100_validation/h100/EVIDENCE.md`.

## Related

- [docs/TRAINING.md](TRAINING.md) — the full recipe: transcripts, run manifest fields,
  save gates, symptom→cause table, and the exit-code contract.
- `foundationscale.gates.core` — the gate registry behind the hello-world example.
- `tests/train/test_examples_runnable.py` — the denominator that keeps this directory
  runnable.
