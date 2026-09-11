# Getting Started with FoundationScale

This document walks a first-time contributor from a clean clone to a full local run of the gate contract, the test suite, and the training example. Everything here is what CI itself executes; the Makefile targets are convenience wrappers that mirror the CI steps exactly. The source of truth for every check is `pyproject.toml` plus `.github/workflows/ci.yml` — CI does not run `make`, so if a target and a workflow ever disagree, the workflow wins.

FoundationScale's founding rule applies to your first session as much as to any gate: a claim must carry its own coverage. `all([])` is True, and a check that examined zero units reports VACUOUS, not PASS. The install layout below exists precisely so that the verifier can run before anything heavy is installed, and so that a missing optional dependency shrinks the suite loudly (via skips) rather than silently.

## Requirements

- Python ≥ 3.10 (`requires-python = ">=3.10"` in `pyproject.toml`; classifiers cover 3.10, 3.11, and 3.12).
- `git` for the clone.
- `make` for the convenience targets. It is not required — every target is a single obvious command, and the commands are given below — but it is how the repository expects a developer to drive the checks before pushing.

Windows support is not stated in the source material; the documented invocations are POSIX (`bash`, `--extra-index-url`, `torchrun`). If another platform is needed, that is a declared gap, not an omission this document can paper over.

## Installation

From a clean clone:

```bash
git clone https://github.com/TranNhiem/FoundationScale && cd FoundationScale
python3 -m pip install -e ".[checkpoint,dev]" "pytest-cov>=5" \
    --extra-index-url https://download.pytorch.org/whl/cpu
```

This is the **contributor** install — the same install `make install` and CI run. It deliberately cannot train. The CPU torch build is what CI needs to exercise the gate contract; if you want accelerator support, the training install below replaces it.

### The extras, and what each one buys

FoundationScale ships **no core runtime dependencies** — the gate contract (`foundationscale.gates.core`) is pure stdlib, so the verification plane runs on a login node with nothing installed. Heavy dependencies live in optional extras declared in `pyproject.toml`:

| extra | contents | when to install |
|---|---|---|
| *(none)* | stdlib only | running the gate contract itself |
| `checkpoint` | `torch>=2.1`, `safetensors>=0.4`, `numpy>=1.24` | checkpoint gates and the real-checkpoint tests |
| `dev` | `pytest>=8.0`, `ruff>=0.5`, `mypy>=1.7`, `types-PyYAML>=6.0` | development |
| `train` | `torch>=2.1`, `transformers>=4.40`, `datasets>=2.18`, `accelerate>=1.1` | the training entry point |
| `all` | `foundationscale[checkpoint,dev,train]` | exists so `all` means all; CI deliberately does **not** install it |

Two properties of this table matter in daily use:

- **`checkpoint` is not optional in practice for contributors.** Omitting it does not fail loudly — it shrinks the suite through skips. That is precisely why CI forbids skips entirely with `FS_FORBID_SKIPS=1`, and why the contributor install carries it. A green run over a shrunken suite is the repository's forbidden shape, not a convenience.
- **CI pins `.[checkpoint,dev]` on purpose** — so `transformers` stays absent and the training loop's guarded-import refusal path is exercised on a host without it. The `numpy` entry in `checkpoint` and the `accelerate` entry in `train` are there because the corresponding upstream packages leave those deps undeclared and the gap only shows up on a clean runner. An undeclared dependency is a defect of the same class as the one that made CI green without torch; declaring it is what keeps the failure loud.

### The training install

To run the training example instead, add the `train` extra and drop the CPU index, so pip resolves a torch build for your own accelerator:

```bash
python3 -m pip install -e ".[train]"
python3 examples/train_tiny.py          # measured: exits 0 PASS
```

## The interpreter question

The Makefile does not assume your tools are on `PATH`. Logged measurements on the machine this file serves:

```
python3           -> 3.14.6 (Homebrew)  pip yes, pytest/ruff/mypy/coverage NO
/usr/bin/python3  -> 3.9.6  (system)    pip yes, pytest/ruff/mypy/coverage NO
.venv/bin/python3 -> 3.14.6             all five present
```

The tools live in the project venv, which is where `make install` puts them. Accordingly, every tool in the Makefile is invoked as `$(PY) -m <tool>`, never as a bare name on `PATH`, and `PY` resolves in this order:

1. `.venv/bin/python3` in the checkout, when it exists and is executable.
2. `python3` as a fallback.
3. An explicit `PY` from the environment or command line, which beats both:

```bash
make check                      # venv if present, else python3
make check PY=python3.11        # explicit wins
PY=/usr/bin/python3 make lint   # environment wins
```

This is not ceremony. A bare `pip`, `pytest`, `ruff`, or `mypy` resolves only if it happens to be on `PATH`; on a laptop with no venv activated it is not, and `make lint` has died with `ruff: command not found` for exactly that reason. `-m` also binds each tool to the same interpreter as the rest of the file — a `mypy` from one environment checking a tree installed in another is a different program. The guard `checks/makefile_tooling.py` makes the bare-name class un-reintroducible by refusing any recipe line that invokes `pip`, `pytest`, `ruff`, `mypy`, `coverage`, or a bare interpreter by name instead of through `$(PY)`.

Run the typecheck in the same environment `install` creates — one with `[checkpoint]`. mypy without torch checks a different program: `MetadataIndex` becomes `Any`, and an `Any | None` assigned over a variable already bound to `str` stops being an error. A torch-free typecheck has passed this tree while CI, which installs torch, failed it on all three Pythons. Same command, same source, different answer, because the environment differed.

## The quick-start commands

After installing per above, these are the commands CI executes, in order of increasing cost.

### 1. Test suite, coverage floor included

```bash
python3 -m pytest --cov=foundationscale --cov-report=term-missing --cov-fail-under=90
```

The `make test` target adds two things to this shape: `--cov=tools` (so the Makefile measures the two adjudicating `tools/` modules the way CI does) and `--cov-report=json` (which writes the `coverage.json` that the `coverage-floor` gate below adjudicates):

```bash
make test
```

`--cov-fail-under=90` is still a **total**, and a total can be subsidised. That is why a second gate exists.

### 2. Per-module coverage floor

```bash
make coverage-floor
```

This runs `checks/coverage_floor.py --self-test` and then `checks/coverage_floor.py`, consuming the `coverage.json` that `make test` writes. Ordering is load-bearing — `coverage-floor` comes after `test` in the `check` aggregate and must stay there. If `coverage.json` is missing, the gate exits 95 (UNMEASURED), never 0: "the report was not there" is not "every module passed".

This is the general pattern for every gate in the repository: **self-test first, then the real run**. A detector whose controls misbehave has no licence to report a verdict.

### 3. Gate controls

```bash
python3 -m foundationscale.gates.controls
```

Also reachable as the console script:

```bash
foundationscale-controls
```

This executes every registered gate's MUST_FIRE / MUST_PASS fixtures and exits nonzero if a gate fails to block its defective input, declares no MUST_FIRE control at all, or the registry is empty. The last failure is the vacuous pass one level up: an empty registry, like an empty test suite, cannot be read as a pass.

### 4. Mutation battery

List the module shards, then run one exactly as CI runs it:

```bash
python3 tools/mutate.py --list
FS_FORBID_SKIPS=1 python3 tools/mutate.py --module checkpoint_gates
```

Or through the Makefile:

```bash
make mutation-module MODULE=checkpoint_gates
make mutation                        # the whole corpus; a surviving mutant fails it
```

What the corpus is, and what it costs, are stated because an understated cost is not harmless — it is the reason a developer reaches for `make test` and skips the one target that certifies the detectors:

- The corpus is **78 rows over 9 modules**: 69 MUST_FIRE mutants and 9 MUST_PASS controls. CI shards per module across a job matrix; `mutation-module` is the mirror of one shard.
- Every module carries its own `must_survive` control, so a shard is a whole detector rather than half of one. Run without a control, `tools/mutate.py` exits 2 (never measured) rather than printing a `caught=` figure it cannot support.
- Each scoreable row runs the whole suite. The measured anchor: `--module checkpoint_gates`, 5 rows, 3m19s on an M-series laptop, i.e. ~40s per row. Extrapolated, **not measured**, the full corpus is near 50 minutes.
- Install the same extras everywhere. All 9 CI shards have died at collection with `No module named 'tokenizers'` because a job that ran the whole suite per mutant lacked `[train]` — and the battery reported that as a wrong verdict rather than a missing dependency.

### 5. Everything at once

```bash
make check
```

The aggregate is, in order:

`lint` · `typecheck` · `typecheck-checks` · `skip-guard-probe` · `test` · `coverage-floor` · `ci-suite-extras` · `controls` · `packaging` · `training-plane` · `makefile-tooling` · `countables` · `launcher-contracts` · `checks-gates` · `mutation`

Every entry is a gate someone once had to type by name; the aggregate exists because a gate reachable only by typing its name is reachable by nobody. Two entries deserve more introduction than their names suggest.

#### `skip-guard-probe`

This mirrors CI's probe step. It generates a deliberately skipped test (`tests/test__skip_guard_probe.py`, which it deletes afterwards), runs it under `FS_FORBID_SKIPS=1`, and asserts two things: the armed guard fails the run, **and** the guard names the probe. A failure without a name is not good enough — an operator must be able to find the offending skip from the output. The recipe ends with the verdict CI requires:

```
skip-guard-probe: guard fired and named its probe, as CI requires
```

#### `launcher-contracts` and `checks-gates`

```bash
make launcher-contracts   # bash launchers/test_launcher_contracts.sh — 149 controls
make checks-gates         # bash launchers/test_checks_gates.sh      — 31 controls
```

The first is the largest gate in the repository (measured 2026-09-11 on an otherwise-idle developer machine: 28.9s wall / 4.1s user — the wall half moves with load, see `docs/TESTING.md`). Both must run: the anti-orphan leg's corpus is the two suites concatenated, so running only one indicts every helper called solely from the other. The anti-orphan leg scans every `launchers/*.py` and `checks/*.py` for call sites and refuses a file with none — a new gate file has been indicted as an orphan after a fully green `make check` more than once, which is why `launcher-contracts` is in the aggregate at all. None of these targets is prefix-suppressed: a recipe written `-...` prints "Error (ignored)" and returns 0, which is a gate that reports and cannot fail.

Both suites resolve one interpreter before their first leg — the repo `.venv` if it exists, else your `python3`, overridable with `FS_SUITE_PY` — and print it on their first line. If it is below the `requires-python` floor, they print one named abstention and exit 95 (UNMEASURED) rather than reporting the gates as broken. If you see that, create the checkout venv (`python3 -m venv .venv && make install`) or point `FS_SUITE_PY` at a conforming interpreter, and re-run; see `docs/TESTING.md`.

### Skips: `FS_FORBID_SKIPS`

`FS_FORBID_SKIPS` is the one declared divergence between `make` and CI. The CI `check` job sets it job-wide so that **any skip fails the build**. `make test` leaves it unset so a laptop may skip — but `tests/conftest.py` still names every skip and its reason in the summary.

To run the suite byte-for-byte as CI sees it:

```bash
FS_FORBID_SKIPS=1 make test
```

The rule skips enforce is the point of the `[checkpoint]` extra. Without torch, 41 tests skip rather than fail, and the suite reads green over a smaller denominator. Forbidding skips turns that difference into a red build.

### Console scripts

The install registers two entry points from `pyproject.toml`:

| script | target |
|---|---|
| `foundationscale-controls` | `foundationscale.gates.controls:main` |
| `foundationscale-train` | `foundationscale.train.cli:main` |

The `packaging` gate (`make packaging`, backed by `checks/packaging_reachability.py`) resolves both names against the interpreter's own script directory and the install record. `PATH` is reported as operator convenience that can never be red — running `.venv/bin/python3` without sourcing `activate` must not turn a packaging gate false-red.

## Your first gate

The honest hello-world for this repository is a gate, because that is the artifact that exists end-to-end. This runs verbatim against the installed package:

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

Note what the assertion is. The fixture declares 128 experts and supplies none, so the gate examined zero units. It is **blocking**, and its verdict is **VACUOUS**, not PASS. That is the founding rule in one screen: a check that found nothing to check has not passed.

## The training path

Real training is one command, and it runs:

```bash
python3 -m pip install -e ".[train]"
python3 examples/train_tiny.py
```

`examples/train_tiny.py` is one screen — a `ClusterProfile` describing the machine and a `TrainConfig` naming a model, a dataset, and a topology. It trains `sshleifer/tiny-gpt2` on `fancyzhx/ag_news` for 20 steps and exits `0 PASS`, having adjudicated both intermediate checkpoints and the final save.

To run it on four GPUs, edit the single `GPUS` constant and launch with:

```bash
torchrun --nproc_per_node=4
```

The profile, the declared topology, and the DDP degree all read that one value, and the declared topology is then checked against the one torchrun actually built.

For a corpus that needs no network, a 16-row toy dataset ships at `examples/data/toy_text.jsonl`, driven through the console script `foundationscale-train` (equivalently `python3 -m foundationscale.train.cli`):

```bash
HF_HUB_OFFLINE=1 python3 -m foundationscale.train.cli \
  --model sshleifer/tiny-gpt2 --dataset examples/data/toy_text.jsonl \
  --output-dir /tmp/fs_train_demo --profile-name local-single-node \
  --nodes 1 --gpus-per-node 1 --dp 1 --max-steps 8 --save-interval 4
```

Three operational details from the loop:

- **Without the `train` extra, the loop refuses.** It exits 96 and prints the install remedy — `pip install 'foundationscale[train]'` — rather than half-starting. A refusal with a remedy is a red build; a half-started run is an unaudited export.
- **`--dry-run` runs the whole validation prologue and stops before importing torch.** Put it in front of a scheduler submission, so an incoherent request is rejected without holding an allocation while it finds out.
- **The loop reads `FS_RUN_ID` and `FS_ATTEMPT` from the environment.**

The full recipe — transcripts, the run manifest's fields, the save gates, and a symptom→cause table — is in [docs/TRAINING.md](TRAINING.md).

## Exit codes you will see

The gates publish a small exit-code namespace rather than the generic pytest code:

| code | meaning |
|---|---|
| `0` | PASS — the check ran and held |
| `2` | never measured — the mutation battery run had no MUST_PASS control to stand on |
| `95` | UNMEASURED — the evidence (for example `coverage.json`) was absent; never a pass |
| `96` | dependency refusal — for example the training loop without the `[train]` extra, or the CI suite-extras gate over zero qualifying jobs |

Zero qualifying jobs is not a pass either: agreement across an empty set is `all([])`, so `ci_suite_extras` exits 95 rather than 0 when it has nothing to examine.

## Environment variables

| variable | effect |
|---|---|
| `FS_FORBID_SKIPS=1` | any skipped test fails the build; set job-wide in CI, unset in `make test` |
| `PY` | overrides the Makefile's interpreter selection (venv, then `python3`) |
| `HF_HUB_OFFLINE=1` | keeps the training CLI on local files for network-free runs |
| `FS_RUN_ID` | run identifier read by the training loop |
| `FS_ATTEMPT` | attempt counter read by the training loop |

## Cleaning up

```bash
make clean
```

This removes build and cache directories plus `.coverage`, `.coverage.*`, `coverage.xml`, `coverage.json`, and `.countables_census.json`. The census is measured, never committed: `make countables` writes it to the working tree, `clean` removes it, and `.gitignore` keeps it out of a commit — a frozen oracle is worse than no oracle.

Uninstallation, virtualenv creation, and pin-level dependency management are **not implemented** as documented commands in the source material; the current behaviour is that you construct the venv and run `pip install` yourself with the lines in the Installation section above, and remove the environment the way you created it.
