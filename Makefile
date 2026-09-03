# Convenience only. The source of truth for every check is pyproject.toml plus
# .github/workflows/ci.yml — CI does not run `make`, so these targets are kept to
# single obvious commands that mirror the CI steps exactly.
#
# What the skip-hole fix changed here (full story in the ci.yml header):
#
#   * `install` now installs .[checkpoint,dev]: the [checkpoint] extra carries
#     torch, and without it 41 tests skip (387 passed with torch; 346 passed /
#     41 skipped without — CI's old green). pytest-cov rides along because the
#     `test` command carries coverage flags, exactly like CI's step.
#   * `test` mirrors CI's pytest invocation, coverage floor included.
#   * FS_FORBID_SKIPS is the ONE declared divergence from CI: the `check` job
#     sets it job-wide so any skip fails the build; `make test` leaves it unset
#     so a laptop may skip — but tests/conftest.py still names every skip and
#     its reason in the summary. To run the suite byte-for-byte as CI sees it:
#
#         FS_FORBID_SKIPS=1 make test
#
#   * `skip-guard-probe` mirrors CI's probe step: it generates a deliberately
#     skipped test and asserts the armed guard fails the run AND names it. It is
#     a compound recipe rather than "one obvious command" — the CI step is a
#     script, and the faithful mirror of a script is the same script. Dropping
#     it from `make check` would make `check` weaker than CI, which is its own
#     vacuous pass, so fidelity won over brevity here.
#   * `mutation` runs the WHOLE corpus (a surviving mutant fails it). CI no
#     longer does: as of #242 it shards per module across a job matrix, and
#     `mutation-module` below is the mirror of one shard.
#
#     The mutation corpus is 78 rows over 9 modules.
#     Of those, 69 are MUST_FIRE mutants and 9 are MUST_PASS controls.
#
#     Each scoreable row runs the whole suite, so the wall time is one full
#     pytest run per MUST_FIRE row rather than one in total. All four counts
#     above are anchored (total_rows, mut_modules, must_fire, must_pass in the
#     countables census) and each sits on its own line, because the gate reads
#     a clause, not a paragraph: a number wrapped across a comment continuation
#     is a number in no denominator. The line read "73 rows" until #242 gave a
#     control row to each of the five modules that had none.
#
#     The WALL TIME is not anchored, because no gate can hold it: it is a
#     property of the machine, which is the #83/#111 shape. What is measured is
#     one module — `--module checkpoint_gates`, 5 rows, 3m19s on an M-series
#     laptop, i.e. ~40s per row. Extrapolated (not measured) that puts the full
#     corpus near 50 minutes. It is stated because an understated cost is not a
#     harmless comment: it is the reason a developer reaches for `make test`
#     and skips the one target that certifies the detectors. It said "~2 min"
#     until #234 and "~35 min" until #242, both of them guesses.

.PHONY: install test lint fmt typecheck typecheck-checks controls packaging countables mutation mutation-module skip-guard-probe check clean

install:
	pip install -e ".[checkpoint,dev]" "pytest-cov>=5" --extra-index-url https://download.pytorch.org/whl/cpu

test:
	pytest --cov=foundationscale --cov-report=term-missing --cov-fail-under=90

# checks/ joined this list with #231. It was the one directory of executable
# gates that no linter, no formatter and no typechecker saw, and the exclusion
# read as coverage: on the day it was added, 2 of its 3 files were unformatted
# and all 3 lacked the `from __future__ import annotations` that every tools/
# gate carries. That last omission was not cosmetic -- a PEP 604 annotation in
# a class body is evaluated at definition time, so all three died at IMPORT
# with a TypeError and exit 1 on Python 3.9, which is the interpreter class the
# login node actually has (#138). A gate that cannot start states no verdict,
# and exit 1 is outside the 0/5/95/96 namespace it claims to publish.
lint:
	ruff check src tests tools checks
	ruff format --check src tests tools checks

fmt:
	ruff check --fix src tests tools checks
	ruff format src tests tools checks

# Run this in the SAME environment as `install` creates — one with [checkpoint].
# mypy without torch checks a different program: MetadataIndex becomes Any, and an
# `Any | None` assigned over a variable already bound to `str` stops being an error.
# A torch-free typecheck passed this tree; CI, which installs torch, failed it on all
# three Pythons. Same command, same source, different answer, because the environment
# differed. That is the repository's own thesis pointed at its Makefile.
# tools/real_checkpoint_probe.py joined this list with finding #219. It is now a
# thin CLI over foundationscale.gates.probe, the same shape live_save_gate.py has
# over gates/adjudication.py, and a boundary wrapper that is not typechecked is a
# re-export list nobody reads: mypy is what notices when the library's signature
# moves out from under the CLI that forwards to it.
#
# checks/ is NOT in this list, and that is a stated exclusion rather than an
# implied one (#231): all 4 files under checks/ are unchecked by mypy. That
# count is anchored -- checks_files in the countables census, bound by
# checks/countables_drift.py -- because the clause it replaces read "3 of 3"
# in the same commit that ADDED the fourth file (#241). It had copied mypy's
# own "Found 10 errors in 3 files (checked 4 source files)" and taken the
# error-bearing count for the denominator: the numerator, printed as the whole.
#
# The error count is deliberately NOT stated here. It was written once as 10
# and measured 19 the next time anyone ran it, and a number that needs mypy --
# and one particular mypy version -- to verify is the #83/#111 shape: a claim
# whose truth depends on the environment that reads it. `make typecheck-checks`
# prints it on demand instead. Two of what it reports are deliberate
# back-compat shapes in packaging_reachability.py that want silencing rather
# than fixing: an EntryPoints.get fallback that only executes on the pre-3.10
# shape mypy cannot see, and an ArgumentParser.exit override that always raises
# and so wants NoReturn.
# Ruff and ruff format DO cover checks/ as of #231; mypy is the tracked half.
typecheck:
	mypy src tools/emit_run_manifest.py tools/live_save_gate.py tools/real_checkpoint_probe.py

# Not part of `check`, and that is the point: this REPORTS, it does not gate.
# It exists so the count above can stay unstated and still be knowable.
typecheck-checks:
	-mypy checks

# python3, not python: bare `python` does not exist on modern macOS or most
# Linux distributions, so `make check` died with command-not-found for any
# developer who had not activated a virtualenv. CI never saw it — setup-python
# provides `python` — so the only machine this file exists to serve was the
# only machine it did not run on (#232).
controls:
	python3 -m foundationscale.gates.controls

# Mirrors the two steps in CI's `controls` job. Self-test first: a detector whose
# controls misbehave has no licence to report a verdict, so the real run must not
# be reachable without it. Deliberately NOT `&&`-joined into one line — CI runs
# them as two steps and a combined recipe would hide which half failed.
#
# This gate is why the target exists at all: it caught a false RED on itself.
# Its first version asked `shutil.which(ep.name)` and reported both console
# scripts unreachable on a tree where pip had installed both correctly — the
# only fact it had measured was that the developer ran `.venv/bin/python3`
# instead of sourcing `activate`. Existence is now resolved against the
# interpreter's own script directory and the install record, and PATH is
# reported as operator convenience that can never be red.
packaging:
	python3 checks/packaging_reachability.py --self-test
	python3 checks/packaging_reachability.py

# Mirrors the three steps in CI's countables leg (#220). The census is measured,
# never committed -- see the ci.yml comment for why a frozen oracle is worse than
# no oracle. It lands in the working tree, so `clean` removes it and .gitignore
# keeps it out of a commit.
#
# Note the census counts tools/countables_census.py itself: it is inside its own
# denominator. That is correct, not a bug -- tools_loc means "LOC under tools/",
# and the producer lives there. Editing the producer moves tools_loc, which is
# exactly the kind of drift this gate exists to catch.
#
# The scan set is `docs README.md Makefile .github/workflows/ci.yml`. The last
# two joined with #241: a directory argument is walked for *.md, but a file
# named outright is scanned whatever its suffix, so the build configuration --
# the two files that DECIDE what CI measures -- is now inside the denominator
# it was previously outside of. It had "3 of 3 files under checks/" in both,
# written in the commit that added the fourth.
CENSUS := $(CURDIR)/.countables_census.json
COUNTABLES_CORPUS := docs README.md Makefile .github/workflows/ci.yml

countables:
	python3 checks/countables_drift.py --self-test
	python3 tools/countables_census.py --self-test
	python3 tools/countables_census.py --no-coverage --out $(CENSUS)
	python3 checks/countables_drift.py --census $(CENSUS) $(COUNTABLES_CORPUS)

mutation:
	FS_FORBID_SKIPS=1 python3 tools/mutate.py

# One shard, the way CI runs it after #242. `make mutation-module MODULE=dcp`.
# Every module carries its own "must_survive" row precisely so a shard is a
# whole detector rather than half of one -- run without a control, mutate.py
# exits 2 (never measured) rather than printing a caught= figure it cannot
# support, and before #242 that is what 5 of the 9 shards would have done.
mutation-module:
	@test -n "$(MODULE)" || { echo 'usage: make mutation-module MODULE=<name>   ("python3 tools/mutate.py --list" names them)'; exit 2; }
	FS_FORBID_SKIPS=1 python3 tools/mutate.py --module $(MODULE)

skip-guard-probe:
	@printf '%s\n' \
		'"""Throwaway MUST_FIRE probe for the FS_FORBID_SKIPS guard (tests/conftest.py).' \
		'' \
		'Generated by the skip-guard-probe target and deleted by it. If this file ever' \
		'survives into the real suite run, the armed guard fails the build on its skip.' \
		'"""' \
		'' \
		'import pytest' \
		'' \
		'' \
		'def test_skip_guard_probe():' \
		'    pytest.skip("deliberate skip; an armed skip guard must fail this run")' \
		> tests/test__skip_guard_probe.py
	@set +e; \
	output=$$(FS_FORBID_SKIPS=1 pytest tests/test__skip_guard_probe.py 2>&1); \
	rc=$$?; \
	set -e; \
	printf '%s\n' "$$output"; \
	rm -f tests/test__skip_guard_probe.py; \
	if [ $$rc -eq 0 ]; then \
		echo "skip-guard-probe: FAILED - guard armed but a skipped test exited 0"; \
		exit 1; \
	fi; \
	printf '%s\n' "$$output" | grep -q test_skip_guard_probe || { \
		echo "skip-guard-probe: FAILED - run failed without the guard naming the probe"; \
		exit 1; \
	}; \
	echo "skip-guard-probe: guard fired and named its probe, as CI requires"

check: lint typecheck skip-guard-probe test controls packaging countables mutation

clean:
	rm -rf build dist .eggs src/*.egg-info *.egg-info \
		.pytest_cache .mypy_cache .ruff_cache .cache \
		.coverage .coverage.* coverage.xml htmlcov .countables_census.json
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
