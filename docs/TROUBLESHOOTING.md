# Troubleshooting

This document is the chapter behind README §26. Its organising principle is the
founding rule: a claim must carry its own coverage. `all([])` is `True`, and in
FoundationScale a check that examined zero units reports VACUOUS — or, where the
check could not run at all, UNMEASURED — never PASS. Most of the symptoms below are
cases where something *looked* fine because nothing was actually examined. The fix,
in each case, is to look at the denominator before trusting the verdict.

## 1. The exit-code contract, because it frames everything else

The training loop returns one of four codes (`EXIT_PASS`, `EXIT_RED`,
`EXIT_UNMEASURED`, `EXIT_REFUSE` in `foundationscale.train.loop`):

| code | name | meaning |
|---|---|---|
| `0` | PASS | training ran and every gate that could adjudicate did, and cleared |
| `5` | RED | something is wrong and was caught — usually before an allocation is burned |
| `95` | UNMEASURED | a claim could not be measured; explicitly not the same as clean |
| `96` | REFUSE | the request is malformed or a dependency is absent; nothing was started |

When a run "fails", the first diagnostic act is to read the exit code and the last
`[fs:train:*]` marker before it. Every line the entry emits starts with one, and the
markers are enumerated in `Step` (`fs:train:start` through `fs:train:done`,
`fs:train:red`, `fs:train:unmeasured`). `fs:train:unmeasured` exists precisely so
that an abstaining step is visible in the log rather than inferable only from the
exit code: a step that abstains silently is indistinguishable from one that ran.

## 2. Many tests "skip" instead of failing

**Cause worth checking (README §26):** torch is missing — the project was installed
with the `[dev]` extra but not `[checkpoint]`.

The general mechanism is the one this project is built to fight: a suite full of
skips *reports success over an empty numerator*. Treat a surprising skip count as a
finding, not a convenience. The parallel failure in the training path is documented
in `docs/TRAINING.md`: on a dense model, the gates `checkpoint.expert_distinctness`
and `checkpoint.expert_bytes` declare SKIP with their zero denominator stated
(`0 units — context declares no experts and none are present (dense model)`), and the
adjudication tail prints `1 of 3 verified` rather than "all clear". A framework that
printed "all clear" over three skips would be telling you nothing while sounding like
it told you something.

**What the material does not say:** the exact name of the test-suite skip reason
string, and how to enumerate which tests skipped for the missing torch. That
mechanism is not shown in the source supplied here; the current behaviour is only
that `[dev]` alone leaves torch absent and the skips follow from it.

## 3. `foundationscale-train` refuses at startup

**Cause:** the `train` extra is absent.

This is a designed refusal, not a crash. `import foundationscale.train.loop` is
torch-free by contract; torch, transformers and datasets are imported *inside*
`train()`, and their absence is a `96 REFUSE` naming the remedy — never a bare
`ImportError` traceback three minutes into a half-started run. The message names:

```bash
pip install 'foundationscale[train]'
```

(loop.py builds this string as `EXTRA_HINT` from `EXTRA = "foundationscale[train]"`.)
`[fs:train:deps]` is the marker you would see immediately before the import attempt.

**One historical reason to trust the refusal's remedy:** for one release the refusal
recommended an extra that did not exist, because `pyproject.toml` carried the console
script and the extra as a note rather than a declaration (finding #224). Both are now
declared in `pyproject.toml`, and `checks/packaging_reachability.py` asks the
installed distribution whether every advertised name resolves.

**Verify before proceeding:**

```bash
python -m foundationscale.train.cli --help
```

## 4. `make ...` says `python: command not found`

**Cause (README §26):** the Makefile uses `python3` deliberately; bare `python`
does not exist on modern macOS.

The failing command is something *you* typed or an alias you configured, not the
Makefile itself. Use `python3` (or activate an environment that provides `python`)
for any command you compose by hand; the Makefile needs no changes.

**What the material does not say:** the full list of Makefile targets is not in the
source supplied here, so none is listed. The one target the material names is
`make countables` (§7).

## 5. Launcher contract suite fails with "0/8 launcher unreadable"

**Cause (README §26):** the suite is CWD-sensitive by measured behaviour; run it from
the repository root.

**What the material does not say:** the name of the suite, the path it resolves its
launchers against, and the meaning of "0/8" beyond the symptom string are not in the
supplied source. The current behaviour is exactly what the README states: run it from
the repository root. A related, fully documented instance of the same class is the
partition-spelling scan: `TrainConfig.launch_corpus` points at a directory of
launcher scripts (`.sh/.sbatch/.slurm`); it defaults to `None` because `train()` is
invoked *inside* an allocation, not as the thing that requests one, and when it is
`None` the scan reports UNMEASURED at `[fs:train:partition]` — never silently skipped
and never faked clean. If you want the scan measured, pass `--launch-corpus` with a
directory that is actually readable from where you launch.

## 6. The mutation battery exits 2

**Cause (README §26):** deliberate — nothing was measured. The candidate causes
listed are a stale anchor, a red suite, or a skip. Fix the cause; do not re-run
hoping for 0.

Exit 2 here is the battery refusing to report a score with no coverage behind it —
the same doctrine as `95 UNMEASURED` in the training loop, at a different lever. A
re-run without a changed cause is a re-run of a vacuous measurement.

**What the material does not say:** the command name of the battery and the meanings
of its other exit codes are not in the supplied source. Only the deliberate exit 2
and its three candidate causes are stated.

## 7. A number in a doc looks wrong

```bash
make countables
```

The drift gate compares shipped wording against a freshly measured census. If the
number in the doc is stale, the gate says so; if the gate is clean and the number
still looks wrong, the census itself is the next thing to read. Either way, the
comparison is automated so that wording cannot silently drift from measurement —
the same rationale as the typechecking pin in `loop.py`'s docstring: a gate whose
verdict moves without a code change is a defect (the unpinned-formatter class,
finding #111).

**What the material does not say:** which documents and which census fields
`make countables` covers are not enumerated in the supplied source.

## 8. Training-path failures, indexed by marker and message

These are the causes `docs/TRAINING.md` §7 states, with the loop.py mechanism behind
each.

### `96 REFUSE`, message names `pip install 'foundationscale[train]'`

The `train` extra is not installed. See §3.

### `ValueError: node_pattern is not a valid regex`

`node_pattern` in a `ClusterProfile` is a Python regular expression, not a Slurm
hostlist:

```python
"node_pattern": r"compute-0[1-8]",   # a REGEX
```

The hostlist spelling of that range puts a `1`-to-`0` range inside a character
class, `re.compile` refuses it, and the profile fails to construct. `ClusterProfile`
catches it in `__post_init__` with a named error — but at *your* first run. If you
are writing or editing a profile for `--profile-path`, check the field against
`re.compile` before submitting.

### `HfUriError: Repository id must be 'namespace/name'`

`datasets` ≥ 5 rejects bare legacy ids. Use `fancyzhx/ag_news`, not `ag_news`. This
applies anywhere `--dataset` (or `TrainConfig.dataset`) takes an HF id; it does not
apply to a local `.json`/`.jsonl` file or directory of them.

### `dataset ... has columns [...]; the thin path requires a 'text' column`

Any dataset source — HF id, `.json`/`.jsonl` file, or directory — must expose a
`text` column. Rename or map your column to `text`.

### `5 RED` at `[fs:train:topology]`

The declared decomposition does not multiply out to the declared GPU count: the
prologue requires `dp x tp x pp x ep x cp` to equal the declared total, and the
per-node accounting requires GPUs per node to equal tasks per node (a gap there is
the Duplicate-GPU crash — caught here rather than two minutes into a run). Fix the
`--dp/--tp/--pp/--ep/--cp` and `--nodes`/`--gpus-per-node` numbers to agree with
each other. Note that the CLI-side degrees (`--tp/--pp/--ep/--cp`) are validated but
the thin loop executes DDP; see `docs/TRAINING.md` §8.

### Declared-vs-effective mismatch under torchrun

Deliberate design: `train()` compares the topology you *declared* against the one
torchrun actually built. The effective side is derived from `WORLD_SIZE` — see
`_effective_topology` in `loop.py`; a declaration copied out of `WORLD_SIZE` would
be a comparator that can never disagree with itself. Declare the topology; let the
framework check you. If the comparison complains, the disagreeing values are `GPUS`
in `examples/train_tiny.py`, or `--dp`/`--gpus-per-node` versus
`--nproc_per_node`. Related runtime-evidence failures surface as findings coded
`train.world_size` (`WORLD_SIZE` is not an integer) and `train.effective_topology`
(`WORLD_SIZE` cannot form a valid topology). On the driver, with `WORLD_SIZE`
unset, the comparison is skipped by announcement at `[fs:train:consistency]`, not
by silence.

### `95 UNMEASURED` on partition spelling

Expected on the driver; pass `--launch-corpus` to measure it. See §5.

## 9. Declared-vs-carried manifest failures: the VACUOUS save gate

If every save-gate verdict on every real run comes back looking empty, check the
manifest filename before anything else. `checkpoint.dcp_meta.load_manifest` — the
reader every checkpoint gate goes through — searches a fixed tuple of basenames, and
the training loop writes `run_manifest.json` (`MANIFEST_NAME` in `loop.py`) for
exactly that reason. For one release the name was `foundationscale.run.json`, which
shares not one character with anything the reader looks for, so every save-gate
verdict came back VACUOUS on every real run (finding #225): a producer and a
consumer each internally consistent, and never introduced. Of the accepted names,
`run_manifest.json` is the reserved, strict one: a malformed file there raises
`CheckpointFormatError` rather than being skipped as absent — under either of the
other two names, corruption is indistinguishable from a run that wrote nothing.

**What the material does not say:** the other two accepted basenames are not listed
in the supplied source.

## 10. Reading what a run left behind

When the symptom is "the run finished but I don't trust it", the audit artifact is
`run_manifest.json` in the output directory (`out/train_tiny/run_manifest.json` for
the example). Two fields answer trust questions directly: `declared` records what
the model *claimed* to contain — the denominator the gates checked against — and
`findings` records every gate verdict, including the SKIPs and their reasons.
`config` and `topology` are recorded *as validated, not as typed*, so they are the
values to diff when a mismatch is suspected. Identity and provenance live in
`run_id`, `attempt` (overridable via `FS_RUN_ID` / `FS_ATTEMPT`), `fingerprint`,
`code`, and `environment`.

If the whole prologue is what you doubt, re-run it without training:

```bash
python -m foundationscale.train.cli ... --dry-run
```

`--dry-run` runs profile resolution, topology arithmetic and validation, and stops
before importing torch — the coherent-request check that does not hold an allocation
while it finds out.

## 11. Offline and repeat-run notes

A recipe that worked once and now fails differently than expected on a second run is
often a download difference, not a code difference. `HF_HUB_OFFLINE=1` covers the
dataset half outright when the dataset is a local file such as
`examples/data/toy_text.jsonl`; a model given as an HF id still needs the Hub on the
first run and is served from the local cache afterwards. Point `--model` at a local
directory to remove the Hub entirely.
