# Testing

FoundationScale's test suite is built on one rule: a check that examined zero units reports VACUOUS, not PASS. `all([])` is `True`, and every mechanism below exists to stop an absent check from buying confidence. This document covers pytest configuration, the skip guard, the mutation battery, and how the Makefile mirrors CI.

## How pytest is configured

pytest is configured in `pyproject.toml`:

```toml
testpaths = ["tests"]
```

Two markers are registered, `slow` and `integration`, and the configuration carries `--strict-markers`, so an unregistered marker is an error rather than a silent typo. Anything not set here lives in `pyproject.toml`. Shared fixtures and hooks live in `tests/conftest.py`.

The conftest is deliberately minimal. The package must be importable via `pip install -e .` (src layout) or a `pythonpath = ["src"]` entry in the pytest configuration. There is intentionally no `sys.path` manipulation: a suite that can only find the framework through path hacks would silently pass against a stale checkout.

### Isolated registries

Tests must never register against the global `REGISTRY`. The conftest provides a `fresh_registry` fixture returning an isolated `GateRegistry` per test, imported from `foundationscale.gates.core`:

```python
from foundationscale.gates.core import GateRegistry

@pytest.fixture
def fresh_registry() -> GateRegistry:
    return GateRegistry()
```

Gates registered against the global registry leak between tests and could mask a `required=` assertion by making a gate present that a later test expects to be missing — the registry-level analogue of `all([]) is True`.

## The skip guard: skips are failures in CI

A tolerated skip is the same shape as `all([]) is True`: the check was absent *and* it bought confidence. That is exactly the founding incident — a check CI never ran stayed green — retold once in [ARCHITECTURE.md](ARCHITECTURE.md).

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

CI runs pytest in a single process, so a module-level list is the whole ledger; under pytest-xdist each worker would keep its own ledger and the controller would need to merge them. **This merge is not implemented; the current behaviour is correct only for single-process runs, and xdist support is out of scope until CI adopts it.**

### The terminal summary

`pytest_terminal_summary` (with `trylast=True`, so it lands at the end of the summary) names every skipped test and its reason — armed or not. "N tests skipped" is not actionable:

- **Armed:** the section header reads `FS_FORBID_SKIPS=1 — N skip(s): every one is a failure here` in red, followed by each nodeid, its reason, and remediation guidance (install the missing extra, or fix the CI matrix).
- **Unarmed:** the section reads `N skip(s) — tolerated locally, FAILED by CI (FS_FORBID_SKIPS=1)` in yellow, naming exactly which tests CI would fail.
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

The mutation corpus is **78 rows over 9 modules**: 69 MUST_FIRE mutants and 9 MUST_PASS controls. Each scoreable row runs the whole suite. Every module carries its own `must_survive` control row precisely so a shard is a whole detector rather than half of one.

### Exit codes separate survival from silence

`tools/mutate.py` distinguishes "a mutant survived" from "nothing was measured":

| Exit code | Meaning |
|---|---|
| 0 | All MUST_FIRE mutants caught, all MUST_PASS controls surviving |
| 1 | A mutant survived — a real detector gap |
| 2 | Never measured — a red suite, any skipped test, a stale anchor, or a shard run without its control |

Empty or invalid input must read as never-measured, never as caught. Run without its control row, `mutate.py` exits 2 rather than printing a `caught=` figure it cannot support.

### Running it

The whole corpus (a surviving mutant fails the run):

```bash
make mutation
```

One shard, the way CI runs it:

```bash
make mutation-module MODULE=checkpoint_gates
make mutation-module MODULE=dcp
```

`make mutation-module` with no `MODULE` exits 2 with usage text; `$(PY) tools/mutate.py --list` names the modules. Both targets run with `FS_FORBID_SKIPS=1` in the recipe — a skipped test inside a mutation run would otherwise read as a caught mutant.

The row counts (78, 9, 69, 9) **are** anchored — `total_rows`, `mut_modules`, `must_fire`, and `must_pass` in the countables census — each on its own line in the Makefile comment, because the gate reads a clause, not a paragraph: a number wrapped across a comment continuation is a number in no denominator. Wall time is deliberately never anchored in any countable: it is a property of the machine, not of the repository.

### Suite-extras parity

Every CI job that executes the pytest suite must install the same extras. The failure mode: a job missing an extra dies at collection, and the battery must read that as never-measured, not as a caught mutant. `checks/ci_suite_extras.py` (reachable locally as `make ci-suite-extras`) gates this, with the denominator scoped to jobs that *run* pytest or `tools/mutate.py`, not jobs that merely mention them — and zero or one such job exits 95 (UNMEASURED), never 0, because agreement across an empty set is `all([])`.

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

The source of truth for every check is `pyproject.toml` plus `.github/workflows/ci.yml` — CI does not run `make`. The Makefile exists so the one command a developer runs before pushing is exactly what CI runs. A gate reachable only by typing its name is reachable by nobody, and a mirror weaker than CI is its own vacuous pass. Every target is therefore in the `check` aggregate:

```bash
make check
```

The aggregate: `lint typecheck typecheck-checks skip-guard-probe test coverage-floor ci-suite-extras controls packaging training-plane makefile-tooling countables launcher-contracts checks-gates mutation`. Each target's script and what it certifies is listed in [DEVELOPMENT.md](DEVELOPMENT.md).

### The one declared divergence

`FS_FORBID_SKIPS` is the single intentional difference: CI's `check` job sets it job-wide; `make test` leaves it unset so a laptop may skip — but the conftest still names every skip and its reason in the summary. For the exact CI behaviour locally, `FS_FORBID_SKIPS=1 make test`.

### Interpreter selection

Every tool is invoked as `$(PY) -m <tool>`, never as a bare name on PATH. `PY` prefers the in-tree venv (`.venv/bin/python3`) when one exists, falls back to `python3` when it does not, and yields to an explicit `PY` (see [DEVELOPMENT.md](DEVELOPMENT.md) for the full conventions):

```bash
make check                      # venv if present, else python3
make check PY=python3.11        # explicit wins
PY=/usr/bin/python3 make lint   # environment wins
```

`PY=` under an interpreter without the dev tools turns `ruff: command not found` into `No module named ruff` — a strictly better error, because it makes the environment question the visible one. `checks/makefile_tooling.py` (`make makefile-tooling`) makes the bare-name class un-reintroducible. It does not gate which interpreter you chose or what that interpreter has installed — a gate that reddened on a developer's environment would report a machine as a repository defect.

### Install, then test

```bash
make install
make test
```

`install` runs `pip install -e ".[checkpoint,dev]" "pytest-cov>=5"` against the PyTorch CPU index (`--extra-index-url https://download.pytorch.org/whl/cpu`). The `[checkpoint]` extra carries torch; without it 41 tests skip. `test` mirrors CI's invocation, coverage included:

```bash
$(PY) -m pytest --cov=foundationscale --cov=tools --cov-report=term-missing --cov-report=json --cov-fail-under=90
```

`--cov-report=json` writes the `coverage.json` that `coverage-floor` adjudicates. Note that `--cov-fail-under=90` is a *total*, and a total can be subsidised by strong modules covering weak ones — which is why the per-module floors exist.

### Ordering rules in `check`

Two ordering rules are load-bearing:

- **`coverage-floor` stays after `test`.** It consumes the `coverage.json` that `test` writes.
- **Self-test first, then the real run**, for every `checks/` gate. A detector whose controls misbehave has no licence to report a verdict, and the real run must not be reachable without it. The pairing is two recipe lines rather than one `&&`-joined line, because CI runs them as two steps and a combined recipe would hide which half failed.

### Exit-code namespace of the checks/ gates

The gate scripts publish 0/5/95/96; 95 is UNMEASURED. A gate that cannot start states no verdict — a bare exit 1 from an import failure is outside the namespace the gates claim to publish, which is why `checks/` is linted, formatted, and typechecked like everything else.

That namespace is measured, not merely asserted: `bash_lc_sweep.py` and `wf_yaml_audit.py` once published 0/1, giving a crash and a finding the same answer; both now answer 5/95/96, both MUST_FIRE probes demand exactly 5, and every tracked gate is declared in [`checks/exit_contract_scope.py`](../checks/exit_contract_scope.py) so the drift cannot return unseen.

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

`launcher-contracts` (`bash launchers/test_launcher_contracts.sh`) and `checks-gates` (`bash launchers/test_checks_gates.sh`) must both run: the bash contract suites are 156 controls, 4 named abstentions, and the gate-script self-test legs are 47 controls, 1 named abstention. The anti-orphan leg's corpus is the two suites concatenated — it scans every `launchers/*.py` and `checks/*.py` for call sites and refuses a file that has none — so running only one half indicts every helper called solely from the other.

### The suites resolve their own interpreter

Both suites shell out to bare `python3` in roughly 150 places, including sites a leg writes at run time. `launchers/_suite_prelude.sh` therefore resolves one interpreter before any leg runs — `$FS_SUITE_PY` if you set it, else the repo `.venv`, else whatever `python3` your PATH gives — and binds it by putting a one-entry shim directory in front of PATH. The shim holds `python3` and nothing else, so `ruff`, `mypy`, `pytest` and `torchrun` still resolve exactly as they did; it is an exec wrapper rather than a symlink so the interpreter reports the environment it belongs to. Every run prints the interpreter it used on its first line: a verdict that does not name its interpreter is unattributable.

If that interpreter is below the `requires-python` floor in `pyproject.toml`, the suite prints one named abstention and exits **95 (UNMEASURED)** without running a leg. A suite that cannot run its gates has not found them defective, and one unmet host precondition should read as one line, not as indictments of the gates: exit 5 would claim the gates are broken; exit 0 would be a green over legs that never ran.

Set `FS_SUITE_PY` to a below-floor interpreter to see the abstention arm fire — that is the positive control.

### Clean

```bash
make clean
```

Removes build and packaging output, tool caches (`.pytest_cache`, `.mypy_cache`, `.ruff_cache`), coverage artefacts (`.coverage`, `coverage.json`, `coverage.xml`, `htmlcov`), the census file, and every `__pycache__` directory.
