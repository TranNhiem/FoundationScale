# Development guide

The Makefile is convenience only. The source of truth for every check is
`pyproject.toml` plus `.github/workflows/ci.yml`, and CI does not run `make` —
each target is a single obvious command that mirrors the corresponding CI step
exactly. This chapter is the expansion of README §24: what each target runs,
the two decisions behind interpreter selection, where `make test` deliberately
diverges from CI, and which parts of the tree each tool does and does not see.

Every tool is invoked as `$(PY) -m <tool>`, never as a bare name on PATH. That
convention is load-bearing and is enforced by a gate, not by review discipline;
see [The interpreter is two decisions](#the-interpreter-is-two-decisions) and
[`makefile-tooling`](#the-gate-targets).

## Target reference

| target | what runs | notes |
|---|---|---|
| `make install` | `pip install -e ".[checkpoint,dev]" "pytest-cov>=5"` with the PyTorch CPU extra index | creates the environment every other target assumes |
| `make test` | pytest with `--cov=foundationscale --cov=tools --cov-report=term-missing --cov-report=json --cov-fail-under=90` | writes the `coverage.json` that `coverage-floor` consumes |
| `make coverage-floor` | `checks/coverage_floor.py --self-test`, then the real run | exits 95 (UNMEASURED) if `coverage.json` is missing, never 0 |
| `make ci-suite-extras` | `checks/ci_suite_extras.py --self-test`, then the real run | every CI job that runs pytest or `tools/mutate.py` must install the same extras |
| `make lint` | `ruff check` + `ruff format --check` over `src tests tools checks` | read-only |
| `make fmt` | `ruff check --fix` + `ruff format` over the same four directories | rewrites in place |
| `make typecheck` | mypy over `src` plus three adjudicating CLIs | see [typecheck](#typecheck-and-typecheck-checks) |
| `make typecheck-checks` | `mypy checks` | a gate, in `check`, and not error-suppressed |
| `make controls` | `python3 -m foundationscale.gates.controls` | the gate controls entry point |
| `make packaging` | `checks/packaging_reachability.py --self-test`, then the real run | self-test first, deliberately two lines |
| `make training-plane` | `checks/training_plane_probe.py --self-test`, then the real run | reports training primitives and delegation as separate axes |
| `make makefile-tooling` | `checks/makefile_tooling.py --self-test`, then the real run | forbids bare tool names in recipes |
| `make mirror` | `checks/makefile_ci_mirror.py --self-test`, then the real run | this file and `ci.yml` must run the same check scripts |
| `make countables` | census self-test, census run, then `checks/countables_drift.py` over the corpus | census is measured, never committed |
| `make launcher-contracts` | `bash launchers/test_launcher_contracts.sh` | 149 controls; 28.9s wall / 4.1s user, measured 2026-09-11 on an idle developer machine |
| `make checks-gates` | `bash launchers/test_checks_gates.sh` | 36 controls; 8.9s wall / 5.5s user, same conditions |
| `make mutation` | `FS_FORBID_SKIPS=1 tools/mutate.py` — the whole corpus | a surviving mutant fails it |
| `make mutation-module MODULE=x` | one mutation shard, as CI runs it | `tools/mutate.py --list` names the modules |
| `make skip-guard-probe` | generates a skipped test, asserts the armed guard fails the run and names it | creates and deletes `tests/test__skip_guard_probe.py` |
| `make check` | the aggregate: all of the above, in a fixed order | the one command to run before pushing |
| `make clean` | removes build artefacts, caches, coverage output, and the census file | also prunes every `__pycache__` directory |

## The interpreter is two decisions

Interpreter selection was reopened as a class by #247 (after #232 fixed bare
`python` in four targets) and is two decisions, not one.

**Decision 1: `-m`, always.** Every recipe calls `$(PY) -m pip`,
`$(PY) -m pytest`, `$(PY) -m ruff`, `$(PY) -m mypy`. A bare name resolves only
if the tool happens to be on PATH, and on the machine this file exists to serve
— a laptop with no venv activated — it is not. During the review campaign
`make lint` died with `ruff: command not found` and `make packaging` reported
UNMEASURED purely because the project venv was inactive. CI never sees this:
`actions/setup-python` puts everything on PATH. `-m` additionally binds each
tool to the same interpreter as the rest of the file.

**Decision 2: which interpreter.** `PY ?= python3` was written first and
measured, then rejected: under a bare `python3`, `$(PY) -m ruff` turns
`command not found` into `No module named ruff` — a better error, but not a
fix. The tools live in the project venv, which is where `make install` puts
them. So the Makefile prefers the in-tree venv and falls back to `python3`:

```make
FS_VENV_PY := $(CURDIR)/.venv/bin/python3
ifeq ($(origin PY),undefined)
PY := $(shell test -x '$(FS_VENV_PY)' && printf %s '$(FS_VENV_PY)' || printf %s python3)
endif
```

`?=` is not used because a recursively-expanded `$(shell ...)` would re-run
the probe at every reference; `origin` expands once and still yields to an
explicit `PY`:

```bash
make check                      # venv if present, else python3
make check PY=python3.11        # explicit wins
PY=/usr/bin/python3 make lint   # environment wins
```

`checks/makefile_tooling.py` makes the bare-name class un-reintroducible, but
it deliberately does NOT gate which interpreter you chose or what that
interpreter has installed — that is a property of the machine, and a gate that
reddens on it would be reporting the developer's environment as a repository
defect.

## `make test` and the one declared divergence

`make install` installs `.[checkpoint,dev]` because the `[checkpoint]` extra
carries torch, and without it 41 tests skip: the suite is 387 passed with
torch, 346 passed / 41 skipped without — and that skip-heavy run was CI's old
green. `pytest-cov` rides along because `make test` carries coverage flags,
exactly like CI's step.

`FS_FORBID_SKIPS` is the one declared divergence between Makefile and CI. The
CI `check` job sets it job-wide, so any skip fails the build. `make test`
leaves it unset, so a laptop may skip — but `tests/conftest.py` still names
every skip and its reason in the summary; a skip is never silent. To run the
suite byte-for-byte as CI sees it:

```bash
FS_FORBID_SKIPS=1 make test
```

The `skip-guard-probe` target proves the guard is not decorative. It writes
`tests/test__skip_guard_probe.py` containing a single test that calls
`pytest.skip(...)`, runs pytest on it with `FS_FORBID_SKIPS=1`, then asserts
two things: the run's exit code is non-zero, and the output names
`test_skip_guard_probe`. A guard that fails the run without naming the probe
fails the probe target itself. It is a compound recipe because the CI step is
a script and the faithful mirror of a script is the same script; dropping it
from `make check` would make `check` weaker than CI, which is its own vacuous
pass.

## The coverage pair

`test` and `coverage-floor` are a pair and their order in `check` is fixed.
The `--cov=tools` flag on the pytest line matters as much as
`--cov=foundationscale`: without it a developer running `make check` got a
green over a denominator two adjudicating modules smaller than the one the
build enforces. `--cov-report=json` writes the `coverage.json` that
`coverage-floor` adjudicates. `--cov-fail-under=90` on the pytest line is a
TOTAL, and a total can be subsidised — one heavily-covered module can carry an
unmeasured one — which is why `checks/coverage_floor.py` exists as a separate
per-module gate.

If `coverage.json` is missing, `coverage-floor` exits 95 (UNMEASURED), never
0: "the report was not there" is not "every module passed". This is the
founding rule applied to tooling — a check that examined zero units reports
VACUOUS, not PASS.

## `typecheck` and `typecheck-checks`

`make typecheck` runs:

```bash
$(PY) -m mypy src tools/emit_run_manifest.py tools/live_save_gate.py tools/real_checkpoint_probe.py
```

Run it in the same environment `install` creates — one with `[checkpoint]`.
mypy without torch checks a different program: `MetadataIndex` becomes `Any`,
and an `Any | None` assigned over a variable already bound to `str` stops
being an error. A torch-free typecheck passed this tree while CI, which
installs torch, failed it on all three Pythons.

The three `tools/` files are named individually because they are the
adjudicating CLIs: `real_checkpoint_probe.py` is a thin CLI over
`foundationscale.gates.probe`, and `live_save_gate.py` has the same shape over
`gates/adjudication.py`. A boundary wrapper that is not typechecked is a
re-export list nobody reads; mypy is what notices when the library's signature
moves out from under the CLI that forwards to it.

`make typecheck-checks` runs `mypy checks` and is in `check` WITHOUT a `-`
prefix. It was once `-$(PY) -m mypy checks`, which made it the one recipe that
could not fail: make prints "Error 1 (ignored)" and returns 0, so the target
read green while reporting 22 errors. That historical shape is why the current
command must be read as written — the exclusion is visible only because the
suppression was removed.

### Stated typecheck exclusions

Stated so silence does not read as coverage:

- mypy does not check `tools/preflight/`. That directory carries an explicit
  exemption; it is not typed and no target typechecks it.
- `checks/` **is** typechecked, by `typecheck-checks`. README §24's line
  naming it "deliberately not a gate" no longer holds: the target gates, is
  in `check`, and the exclusion note that once stood in the Makefile was
  itself wrong (it excused a `Distribution.entry_points` fallback as a
  pre-3.10 shape; `entry_points` is a list before 3.10 and has never been the
  mapping `.get()` addresses, so the arm was dead on 3.10+ and an
  `AttributeError` on exactly the interpreters it was written for).

ruff, by contrast, covers everything listed on the command line:
`src tests tools checks`, including everything under `tools/`.

## The gate targets

Six targets follow the same shape: run the gate's self-test first, then the
real run, as two separate recipe lines (never `&&`-joined, because CI runs
them as two steps and a combined recipe would hide which half failed). A
detector whose controls misbehave has no licence to report a verdict.

| gate script | target | what it adjudicates |
|---|---|---|
| `checks/packaging_reachability.py` | `packaging` | both console scripts reachable, resolved against the interpreter's script directory and the install record — never against PATH, which is operator convenience and can never be red |
| `checks/training_plane_probe.py` | `training-plane` | reports "no training primitives" and "delegates to `transformers.Trainer`" as two axes; scans every git-tracked `*.md` for the retired phrasings |
| `checks/makefile_tooling.py` | `makefile-tooling` | no recipe line may invoke pip, pytest, ruff, mypy, coverage or a bare interpreter by name |
| `checks/makefile_ci_mirror.py` | `mirror` | the `check` tree and `.github/workflows/ci.yml` run the same set of `checks/` scripts — it compares WHICH scripts, not their argv |
| `checks/countables_drift.py` | `countables` | drift between the measured census and what the shipped documents claim |
| `checks/coverage_floor.py` | `coverage-floor` | per-module coverage, against `coverage.json` |
| `checks/ci_suite_extras.py` | `ci-suite-extras` | CI jobs that execute pytest or `tools/mutate.py` all install the same extras |

`ci-suite-extras` denominates on jobs that RUN pytest or `tools/mutate.py`,
not jobs that merely mention them — a job installing pytest-cov without
invoking pytest is a MUST_PASS control, because widening the denominator to
"mentions pytest" is how a scanner starts reddening jobs it has no claim over.
Zero or one such job exits 95, never 0: agreement across an empty set is
`all([])`.

### `countables`

The census is measured, never committed. The full sequence:

```bash
$(PY) checks/countables_drift.py --self-test
$(PY) tools/countables_census.py --self-test
$(PY) tools/countables_census.py --no-coverage --out $(CENSUS)
$(PY) checks/countables_drift.py --census $(CENSUS) $(COUNTABLES_CORPUS)
```

The scan set is `docs README.md Makefile .github/workflows/ci.yml`. A
directory argument is walked for `*.md`; a file named outright is scanned
whatever its suffix, so the two files that decide what CI measures are inside
the denominator. The census counts `tools/countables_census.py` itself — it is
inside its own denominator, and editing the producer moves `tools_loc`, which
is exactly the drift this gate exists to catch. The census lands in the
working tree as `.countables_census.json`; `make clean` removes it and
`.gitignore` keeps it out of a commit.

## The launcher suites

Two bash suites sit beside the Python gates:

- `make launcher-contracts` runs `launchers/test_launcher_contracts.sh` — 149 controls,
  the largest gate in the repository. Measured 2026-09-11 on an idle
  developer machine at 28.9s wall / 4.1s user; user time is a seventh of
  wall because the watchdog legs run their budgets concurrently. Its
  anti-orphan leg scans every `launchers/*.py` plus `checks/*.py` for call
  sites and refuses a file that has none.
- `make checks-gates` runs `launchers/test_checks_gates.sh` — 36 controls,
  the gate self-tests split out of the launcher suite, 8.9s wall under the
  same conditions.

Take the wall figures as dated observations, not budgets. The launcher
suite measured 63.6s wall on this same machine under concurrent load —
2.2x — while its user time held at 4.0s. That asymmetry is why the two
halves are treated differently: a control count is a property of the
repository and is worth anchoring, whereas a wall time is a property of
whatever else the machine happens to be running, and gating it would
manufacture a red that says nothing about the code. Re-measure the wall
figures when you care about them; do not treat a slower run as a
regression.

Both must run: the anti-orphan leg's corpus is the two suites concatenated,
so running only one indicts every helper called solely from the other. That
is why both targets are in `check`, not merely available to type.

## Mutation testing

The corpus is 78 rows over 9 modules: 69 MUST_FIRE mutants and 9 MUST_PASS
controls. All four counts are anchored in the countables census
(`total_rows`, `mut_modules`, `must_fire`, `must_pass`), each on its own line
in the Makefile comment, because the gate reads a clause, not a paragraph. A
surviving mutant fails `make mutation`.

Each scoreable row runs the whole suite, so wall time is one full pytest run
per MUST_FIRE row. The wall time is NOT anchored — it is a property of the
machine. What is measured is one module: `--module checkpoint_gates`, 5 rows,
3m19s on an M-series laptop (~40s per row). Extrapolated, not measured, the
full corpus runs near 50 minutes. That cost is stated because an understated
cost is the reason a developer reaches for `make test` and skips the one
target that certifies the detectors.

CI shards the corpus per module across a job matrix; `make mutation-module` is
the mirror of one shard:

```bash
make mutation-module MODULE=dcp
```

Run `$(PY) tools/mutate.py --list` to see the module names. Every module
carries its own `must_survive` row precisely so a shard is a whole detector —
run without a control, `tools/mutate.py` exits 2 (never measured) rather than
printing a `caught=` figure it cannot support.

Both mutation targets set `FS_FORBID_SKIPS=1`. The measured reason: a shard
whose environment lacks a suite dependency dies at collection (the historical
failure was "No module named 'tokenizers'", reported by the battery as
`assert 96 == 5`), and an unmeasured mutant read as a wrong verdict rather
than as a missing dependency.

## `make check` and its ordering

```make
check: lint typecheck typecheck-checks skip-guard-probe test coverage-floor \
       ci-suite-extras controls packaging training-plane makefile-tooling \
       countables launcher-contracts checks-gates mutation
```

The order carries meaning:

- `coverage-floor` follows `test` and must stay there — it consumes the
  `coverage.json` that `test` writes.
- `skip-guard-probe` precedes `test`, proving the guard is armed before the
  suite it protects runs.
- Every gate target that exists must appear on this line. A gate reachable
  only by typing its name is reachable by nobody — the aggregate is the one
  command a developer runs before pushing, and a target absent from it makes
  `check` weaker than the CI it claims to mirror.

## Exit-code conventions

The gates in `checks/` and `tools/` publish a small namespace of exit codes,
and `make` recipes stay inside it:

| code | meaning | example |
|---|---|---|
| 0 | measured pass | — |
| 95 | UNMEASURED — the report or input was not there | `coverage-floor` with no `coverage.json`; `ci-suite-extras` over an empty job set |
| 2 | never measured / usage | `mutate.py` with no control row; `mutation-module` with no `MODULE` |
| 1 | assertion failure inside the probe | `skip-guard-probe` when the guard passed a skipped test |

Codes 5 and 96 also belong to the published gate namespace; the Makefile
comments reference "the 0/5/95/96 namespace" without assigning 5 and 96 to
specific make-visible conditions in this file, so this document does not assign
them either. What is established: a gate that cannot start at all (an import
error, a missing interpreter module) exits 1, which is OUTSIDE the namespace
the gates claim to publish — that shape occurred when `checks/` files lacking
`from __future__ import annotations` died at import on Python 3.9, and it is
one reason `checks/` is linted and typechecked like everything else.

## What this file does not provide

There is no `make` target for building the H100 validation harness,
publishing deliverables, or running anything under `validation_campaigns/`;
the estate launch plane beyond the two contract suites is likewise not
reachable from the Makefile. Those trees are documented in README §23 as
lab notebooks and launch code, not framework code, and the absence of a
target is deliberate rather than an omission to patch ad hoc — but if you add
one, `checks/makefile_tooling.py` will require it to respect the `$(PY) -m`
convention, and CI will need its own mirror, because the Makefile is the
mirror, never the source.
