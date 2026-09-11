# Testing

FoundationScale's test suite is built on one rule: a check that examined zero units reports VACUOUS, not PASS. `all([])` is `True`, and every mechanism described below exists to stop an absent check from buying confidence. This document covers how the suite is configured, how the skip guard closes the vacuous-pass hole, how mutation testing validates the detectors themselves, and how the local Makefile is kept byte-faithful to CI.

## How pytest is configured

pytest is configured in `pyproject.toml`:

```toml
testpaths = ["tests"]
```

Two markers are registered, `slow` and `integration`, and the configuration carries `--strict-markers`, so an unregistered marker is an error rather than a silent typo. Shared fixtures and hooks live in `tests/conftest.py`.

The conftest is deliberately minimal. The package is expected to be importable via `pip install -e .` (src layout) or a `pythonpath = ["src"]` entry in the pytest configuration. There is intentionally no `sys.path` manipulation: a test suite that can only find the framework through path hacks would silently pass against a stale checkout — the same class of silent success the framework exists to prevent.

### Isolated registries

Tests must never register against the global `REGISTRY`. The conftest provides a `fresh_registry` fixture returning an isolated `GateRegistry` per test, imported from `foundationscale.gates.core`:

```python
from foundationscale.gates.core import GateRegistry

@pytest.fixture
def fresh_registry() -> GateRegistry:
    return GateRegistry()
```

Gates registered against the global registry would leak between tests and could mask a `required=` assertion by making a gate present that a later test expects to be missing — the registry-level analogue of `all([]) is True`.

## The skip guard: skips are failures in CI

This repository's founding incident was reproduced by a test CI never ran: the extra that provides torch was never installed, and 41 checkpoint tests skipped while CI stayed green (387 passed with torch; 346 passed / 41 skipped without — the old green). A tolerated skip is the same shape as `all([]) is True`: the check was absent *and* it bought confidence.

### Arming the guard

Set `FS_FORBID_SKIPS=1` and the hooks in `tests/conftest.py` turn any skip into a build failure. The variable is set job-wide in CI's `check` job and stays unset on a developer laptop, where skips are merely named:

```bash
FS_FORBID_SKIPS=1 make test   # byte-for-byte what CI sees
make test                     # laptop mode: skips listed, tolerated
```

The guard is armed by exactly one comparison, `os.environ.get("FS_FORBID_SKIPS") == "1"` — it is never armed by default.

### What the guard records

Every skip, from every phase, lands in a per-process ledger `_SKIPPED: list[tuple[str, str]]` of `(nodeid, reason)` pairs:

| Hook | Catches |
|---|---|
| `pytest_runtest_logreport` | Skips from setup (`skipif`, fixture skip), call, and teardown — each arrives as its own report with `skipped` set |
| `pytest_collectreport` | Module-level skips (`pytest.skip(..., allow_module_level=True)`), which never reach the runtest phase |

The reason is always recorded, because the reason is the actionable half: "could not import 'torch'" calls for installing an extra, while a platform guard calls for fixing the CI matrix. Skips carry `(path, lineno, reason)` tuples; anything unexpected is stringified whole.

One scope note: CI runs pytest in a single process, so a module-level list is the whole ledger. Under pytest-xdist each worker would keep its own ledger and the controller would need to merge them. **This merge is not implemented; the current behaviour is correct only for single-process runs, and xdist support is out of scope until CI adopts it.**

### The terminal summary

`pytest_terminal_summary` (with `trylast=True`, so it lands at the end of the summary where a failing build's last screenful lives) names every skipped test and its reason — armed or not. "N tests skipped" is not actionable:

- **Armed:** the section header reads `FS_FORBID_SKIPS=1 — N skip(s): every one is a failure here` in red, followed by each nodeid, its reason, and remediation guidance (install the missing extra, or fix the CI matrix).
- **Unarmed:** the section reads `N skip(s) — tolerated locally, FAILED by CI (FS_FORBID_SKIPS=1)` in yellow — a visible reminder of exactly which tests CI would fail.
- **Armed with zero skips:** the guard says so explicitly (`skip guard armed — zero skips observed`), so an armed run states its own coverage either way.

### Flipping the exit status

`pytest_sessionfinish` (also `trylast=True`) flips an otherwise-green run to failure when the guard is armed and skips were observed:

```python
if _guard_armed() and _SKIPPED and session.exitstatus == pytest.ExitCode.OK:
    session.exitstatus = pytest.ExitCode.TESTS_FAILED
```

`session.exitstatus` is read by pytest after all `pytest_sessionfinish` hooks complete — pytest's own `--suppress-no-test-exit-code` option is implemented by assigning to it from this same hook — so assignment here is the supported way for a plugin to change the process exit code. An already-red status is never touched: the hook can only make a run fail, never make one pass.

### Keeping the guard honest: the probe

A guard whose probe exits 0 is a guard that has rotted into a no-op. The `check` job in `.github/workflows/ci.yml` (and `make skip-guard-probe`) generates a deliberately skipped test — `test_skip_guard_probe` in `tests/test__skip_guard_probe.py`, which calls `pytest.skip("deliberate skip; an armed skip guard must fail this run")` — runs pytest against it with the guard armed, and fails unless the run **both fails and names the probe**:

```bash
make skip-guard-probe
```

The probe file is generated and deleted by the recipe. The probe fails the build in either of two ways: the run exits 0 despite the armed guard, or the run fails without the summary naming `test_skip_guard_probe`. On success it prints: `skip-guard-probe: guard fired and named its probe, as CI requires`.

## Mutation testing asks the other question

The skip guard asks "did every check run?". Mutation testing asks the converse: "would the checks catch a real defect?" A suite that passes over broken code is vacuous in the other direction.

### The corpus

The mutation corpus is **78 rows over 9 modules**: 69 MUST_FIRE mutants and 9 MUST_PASS controls. Each scoreable row runs the whole suite, so wall time is one full pytest run per MUST_FIRE row. Every module carries its own `must_survive` control row precisely so a shard is a whole detector rather than half of one.

### Exit codes separate survival from silence

`tools/mutate.py` distinguishes "a mutant survived" from "nothing was measured":

| Exit code | Meaning |
|---|---|
| 0 | All MUST_FIRE mutants caught, all MUST_PASS controls surviving |
| 1 | A mutant survived — a real detector gap |
| 2 | Never measured — a red suite, any skipped test, a stale anchor, or a shard run without its control |

The 1/2 separation is the founding rule applied to the battery itself: a red suite, any skipped test, or a stale anchor all read as never-measured, never as caught. Run without its control row, `mutate.py` exits 2 rather than printing a `caught=` figure it cannot support.

### Running it

The whole corpus (a surviving mutant fails the run):

```bash
make mutation
```

One shard, the way CI runs it since #242:

```bash
make mutation-module MODULE=checkpoint_gates
make mutation-module MODULE=dcp
```

`make mutation-module` with no `MODULE` exits 2 with usage text; `$(PY) tools/mutate.py --list` names the modules. Both targets run with `FS_FORBID_SKIPS=1` in the recipe — a skipped test inside a mutation run would otherwise read as a caught mutant.

Measured cost, stated so nobody underestimates it: one module (`--module checkpoint_gates`, 5 rows) took 3m19s on an M-series laptop, roughly 40s per row. Extrapolated — not measured — the full corpus runs near 50 minutes. The wall time is deliberately *not* anchored in the countables census, because no gate can hold it: it is a property of the machine, not of the repository. The row counts (78, 9, 69, 9) **are** anchored — `total_rows`, `mut_modules`, `must_fire`, and `must_pass` in the countables census — each on its own line in the Makefile comment, because the gate reads a clause, not a paragraph: a number wrapped across a comment continuation is a number in no denominator.

### Suite-extras parity

Every CI job that executes the pytest suite must install the same extras. The measured failure behind this: a test module needing `[train]` was added, the `check` job got the extra, the `mutation` job — which runs the whole suite once per mutant — did not, and all 9 shards died at collection with "No module named 'tokenizers'". The battery reported that as `assert 96 == 5`, so an unmeasured mutant read as a wrong verdict rather than a missing dependency. `checks/ci_suite_extras.py` (reachable locally as `make ci-suite-extras`) gates this, with the denominator scoped to jobs that *run* pytest or `tools/mutate.py`, not jobs that merely mention them — and zero or one such job exits 95 (UNMEASURED), never 0, because agreement across an empty set is `all([])`.

## CI's four jobs

CI has four jobs on purpose, each covering a distinct failure axis:

| Job | Purpose | Python |
|---|---|---|
| `check` | Hygiene gates across the suite | 3.10 / 3.11 / 3.12 |
| `controls` | Gate fixtures (`make controls`, `make packaging`) | matrix |
| `launchers` | The bash contract suites plus the workflow-YAML and bash-`lc` standing legs | — |
| `mutation` | The mutation battery, sharded per module, enumerated from the mutation table itself | matrix |

The `mutation` job shards per module across a matrix rather than running the whole corpus in one job; `make mutation-module MODULE=<name>` is the local mirror of one shard.

## The Makefile is a mirror, not a source

The source of truth for every check is `pyproject.toml` plus `.github/workflows/ci.yml` — CI does not run `make`. The Makefile exists so the one command a developer runs before pushing is exactly what CI runs. Its self-described principle: a gate reachable only by typing its name is reachable by nobody, and a mirror weaker than CI is its own vacuous pass. Every target below is therefore in the `check` aggregate:

```bash
make check
```

The aggregate: `lint typecheck typecheck-checks skip-guard-probe test coverage-floor ci-suite-extras controls packaging training-plane makefile-tooling countables launcher-contracts checks-gates mutation`.

### The one declared divergence

`FS_FORBID_SKIPS` is the single intentional difference: CI's `check` job sets it job-wide; `make test` leaves it unset so a laptop may skip — but the conftest still names every skip and its reason in the summary. For the exact CI behaviour locally, `FS_FORBID_SKIPS=1 make test`.

### Interpreter selection

Every tool is invoked as `$(PY) -m <tool>`, never as a bare name on PATH — a bare name resolves only if the tool happens to be on PATH, and on a laptop with no venv activated it is not. `PY` prefers the in-tree venv (`.venv/bin/python3`) when one exists, falls back to `python3` when it does not, and yields to an explicit `PY`:

```bash
make check                      # venv if present, else python3
make check PY=python3.11        # explicit wins
PY=/usr/bin/python3 make lint   # environment wins
```

`PY=` under an interpreter without the dev tools turns `ruff: command not found` into `No module named ruff` — a strictly better error, because it makes the environment question the visible one. `checks/makefile_tooling.py` (`make makefile-tooling`) makes the bare-name class un-reintroducible. It does not gate which interpreter you chose or what that interpreter has installed — that is a property of the machine, and a gate that reddens on it would report the developer's environment as a repository defect.

### Install, then test

```bash
make install
make test
```

`install` runs `pip install -e ".[checkpoint,dev]" "pytest-cov>=5"` against the PyTorch CPU index (`--extra-index-url https://download.pytorch.org/whl/cpu`). The `[checkpoint]` extra carries torch; without it 41 tests skip — the founding incident, reproduced on demand. `test` mirrors CI's invocation, coverage included:

```bash
$(PY) -m pytest --cov=foundationscale --cov=tools --cov-report=term-missing --cov-report=json --cov-fail-under=90
```

`--cov-report=json` writes the `coverage.json` that `coverage-floor` adjudicates. Note that `--cov-fail-under=90` on that line is still a *total*, and a total can be subsidised by strong modules covering weak ones — which is why the next gate exists.

### The gates in `check`

| Target | Script / command | What it certifies |
|---|---|---|
| `lint` | `ruff check` / `ruff format --check` over `src tests tools checks` | Style, including the gate scripts themselves |
| `fmt` | `ruff check --fix` / `ruff format` | Auto-fix (not in `check`) |
| `typecheck` | `mypy src tools/emit_run_manifest.py tools/live_save_gate.py tools/real_checkpoint_probe.py` | Types — must run in the `[checkpoint]` environment; mypy without torch checks a different program (`MetadataIndex` becomes `Any`) |
| `typecheck-checks` | `mypy checks` | The gate scripts, and deliberately not `-`-prefixed — a recipe that cannot fail reports green while reporting errors |
| `test` | pytest with coverage (above) | The suite |
| `coverage-floor` | `checks/coverage_floor.py --self-test`, then the real run | Per-module coverage floors; missing `coverage.json` exits 95 (UNMEASURED), never 0 |
| `ci-suite-extras` | `checks/ci_suite_extras.py --self-test`, then the real run | Parity of installed extras across pytest-executing CI jobs |
| `controls` | `python3 -m foundationscale.gates.controls` | The gate fixtures |
| `packaging` | `checks/packaging_reachability.py --self-test`, then the real run | Console scripts reachable, resolved against the interpreter's own script directory and install record |
| `training-plane` | `checks/training_plane_probe.py --self-test`, then the real run | Reports "no training primitives" and "package delegates to `transformers.Trainer`" as separate axes, and scans every git-tracked `*.md` for retired phrasings |
| `makefile-tooling` | `checks/makefile_tooling.py --self-test`, then the real run | No recipe invokes a bare tool name |
| `mirror` | `checks/makefile_ci_mirror.py --self-test`, then the real run | The Makefile `check` tree and the CI workflow run the same check scripts; drift in either direction is RED |
| `countables` | `checks/countables_drift.py --self-test`; `tools/countables_census.py`; drift check against the census | Fixed numbers in docs match the measured census |
| `launcher-contracts` | `bash launchers/test_launcher_contracts.sh` | The launcher bash contract suites — 149 controls, 4 named abstentions |
| `checks-gates` | `bash launchers/test_checks_gates.sh` | The gate-script self-test legs — 40 controls, 1 named abstention |
| `skip-guard-probe` | compound recipe | The armed skip guard fires and names its probe |
| `mutation` | `FS_FORBID_SKIPS=1 python3 tools/mutate.py` | The detectors catch their MUST_FIRE mutants |

Two ordering rules are load-bearing:

- **`coverage-floor` stays after `test`.** It consumes the `coverage.json` that `test` writes.
- **Self-test first, then the real run**, for every `checks/` gate. A detector whose controls misbehave has no licence to report a verdict, and the real run must not be reachable without it. The pairing is two recipe lines rather than one `&&`-joined line, because CI runs them as two steps and a combined recipe would hide which half failed.

### Exit-code namespace of the checks/ gates

The gate scripts publish 0/5/95/96; 95 is UNMEASURED. A gate that cannot start states no verdict — a bare exit 1 from an import failure is outside the namespace the gates claim to publish, which is why `checks/` is linted, formatted, and typechecked like everything else.

### The countables census

`make countables` measures the census fresh every run — it is never committed (`clean` removes `.countables_census.json`; `.gitignore` keeps it out of commits), because a frozen oracle drifts silently while a measured one cannot. The scan set is `docs README.md Makefile .github/workflows/ci.yml`:

```bash
$(PY) checks/countables_drift.py --self-test
$(PY) tools/countables_census.py --self-test
$(PY) tools/countables_census.py --no-coverage --out $(CENSUS)
$(PY) checks/countables_drift.py --census $(CENSUS) $(COUNTABLES_CORPUS)
```

A directory argument is walked for `*.md`; a file named outright is scanned whatever its suffix — so the two files that *decide* what CI measures are inside the denominator. The census counts `tools/countables_census.py` itself: it lives under `tools/`, and `tools_loc` means LOC under `tools/`. Editing the producer moves its own denominator, which is exactly the drift this gate exists to catch.

### Launcher suites come in two halves

`launcher-contracts` and `checks-gates` must both run. The anti-orphan leg's corpus is the two suites concatenated — it scans every `launchers/*.py` and `checks/*.py` for call sites and refuses a file that has none — so running only one half indicts every helper called solely from the other. Measured on the developer machine, 2026-09-11, with nothing else running: 28.9s wall / 4.1s user for the launcher suite (the watchdog legs run their wall budgets concurrently, which is why user time is a fraction of wall), and 8.3s wall / 4.9s user for the checks-gates half (that half is dominated by `campaign_self_tests`, which spawns one subprocess per enrolled `--self-test` module, and by the discrimination leg that runs the whole gate twice to prove it can go red). Read those as dated observations, not as budgets, and note the asymmetry between the two numbers: **wall time is not a property of this repository.** The same launcher suite on the same machine measured 63.6s wall while a concurrent job held the CPU — 2.2x, with user time unmoved at 4.0s. Wall time is therefore deliberately not a gated countable; gating it would manufacture a red on a busy laptop and would say nothing about the code. User time is the stable half, and it drifts only when the suites actually grow.

### The suites resolve their own interpreter

Both suites shell out to bare `python3` in roughly 150 places, including sites a leg writes at run time. `launchers/_suite_prelude.sh` therefore resolves one interpreter before any leg runs — `$FS_SUITE_PY` if you set it, else the repo `.venv`, else whatever `python3` your PATH gives — and binds it by putting a one-entry shim directory in front of PATH. The shim holds `python3` and nothing else, so `ruff`, `mypy`, `pytest` and `torchrun` still resolve exactly as they did; it is an exec wrapper rather than a symlink so the interpreter reports the environment it belongs to. Every run prints the interpreter it used on its first line, which is the same rule the `checks/` gates follow: a verdict that does not name its interpreter is unattributable.

If that interpreter is below the `requires-python` floor in `pyproject.toml`, the suite prints one named abstention and exits **95 (UNMEASURED)** without running a leg. This replaces a measurement that was actively misleading: under this machine's `/usr/bin/python3` (3.9.6), `launchers/test_checks_gates.sh` reported 22 passed and 9 failed, and every one of those nine — six gate self-tests, three MUST_FIRE discrimination legs — announced its gate as broken. None was. A suite that cannot run its gates has not found them defective, and one unmet host precondition should read as one line, not as nine indictments. Exit 5 would claim the gates are broken; exit 0 would be a green over legs that never ran.

Set `FS_SUITE_PY` to a below-floor interpreter to see the abstention arm fire — that is the positive control, and it is how the arm was verified.

### Clean

```bash
make clean
```

Removes build and packaging output, tool caches (`.pytest_cache`, `.mypy_cache`, `.ruff_cache`), coverage artefacts (`.coverage`, `coverage.json`, `coverage.xml`, `htmlcov`), the census file, and every `__pycache__` directory.

## What this document does not answer

Two items a reader may want are not resolvable from the sources this chapter is written against:

- **The full pyproject.toml pytest configuration.** Only `testpaths`, the two markers, and `--strict-markers` are attested here. Additional options (coverage defaults, filter warnings, `pythonpath`) are not quoted in the available material; consult `pyproject.toml` directly.
- **The shard list for the mutation matrix.** `make mutation-module MODULE=<name>` takes a module name, and `python3 tools/mutate.py --list` enumerates them; only `checkpoint_gates` and `dcp` are named in the sources above. The matrix is enumerated from the mutation table itself, so the Makefile and CI cannot drift apart on it.
