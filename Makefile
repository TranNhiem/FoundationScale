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

# ---------------------------------------------------------------------------
# Interpreter selection, and why it is two decisions rather than one (#247).
# ---------------------------------------------------------------------------
# Every tool in this file is invoked as `$(PY) -m <tool>`, never as a bare name
# on PATH. #247 reopened #232 as a class: #232 fixed bare `python` in four
# targets, and the same defect survived in the TOOL invocations -- 9 recipe
# lines across 7 targets calling `pip`, `pytest`, `ruff` and `mypy` by name. A
# bare name resolves only if the tool happens to be on PATH, and on the machine
# this file exists to serve -- a laptop with no venv activated -- it is not.
# That is not hypothetical: during the review campaign `make lint` died with
# `ruff: command not found` and `make packaging` reported UNMEASURED, both
# purely because the project venv was inactive. CI never sees it, because
# actions/setup-python puts everything on PATH, so the only machine this file
# exists for is the only machine it failed on.
#
# `-m` also binds each tool to the SAME interpreter as the rest of the file
# rather than to whatever PATH offers, which is the #83/#111 shape closed one
# more time: `mypy` from one environment checking a tree installed in another
# is a different program, and the typecheck comment below says exactly that
# about torch.
#
# THE SECOND DECISION, and the reason the obvious one-liner is not the fix.
# `PY ?= python3` was written first and MEASURED, on the machine in question:
#
#     python3          -> 3.14.6 (Homebrew)  pip yes, pytest/ruff/mypy/coverage NO
#     /usr/bin/python3 -> 3.9.6  (system)    pip yes, pytest/ruff/mypy/coverage NO
#     .venv/bin/python3-> 3.14.6             all five present
#
# So `$(PY) -m ruff` under a bare `python3` does not fix `make lint`; it turns
# `ruff: command not found` into `No module named ruff`. That is a strictly
# better error -- it names the interpreter and makes the environment question
# the visible one instead of a PATH riddle -- but shipping it as the fix would
# have been the campaign's own recurring defect: declaring a class closed on
# the strength of the edit rather than of a re-measurement. The tools are not
# on PATH and they are not in the default interpreter either; they are in the
# project venv, which is where `make install` puts them.
#
# Hence: prefer the in-tree venv when one exists, fall back to `python3` when
# it does not, and let an explicit PY beat both. `?=` is not used, because a
# recursively-expanded `$(shell ...)` would re-run the probe at every one of
# this file's ~20 references; `origin` is the equivalent that expands once and
# still yields to an environment or command-line PY.
#
#     make check                      # venv if present, else python3
#     make check PY=python3.11        # explicit wins
#     PY=/usr/bin/python3 make lint   # environment wins
#
# `checks/makefile_tooling.py` makes the bare-name class un-reintroducible. It
# does NOT gate which interpreter you chose or what that interpreter has
# installed: that is the #83/#111 axis, a property of the machine reading the
# file rather than of the file, and a gate that reddens on it would be
# reporting the developer's environment as a repository defect.
FS_VENV_PY := $(CURDIR)/.venv/bin/python3
ifeq ($(origin PY),undefined)
PY := $(shell test -x '$(FS_VENV_PY)' && printf %s '$(FS_VENV_PY)' || printf %s python3)
endif

.PHONY: install test coverage-floor ci-suite-extras lint fmt typecheck typecheck-checks controls packaging training-plane makefile-tooling countables launcher-contracts checks-gates mutation mutation-module skip-guard-probe check clean

install:
	$(PY) -m pip install -e ".[checkpoint,dev]" "pytest-cov>=5" --extra-index-url https://download.pytorch.org/whl/cpu

# --cov=tools joined this line with #251, so the Makefile measures what CI measures.
# It had drifted the other way from #230's case: there CI was the weaker of the two,
# here the Makefile was, and a developer running `make check` got a green over a
# denominator two adjudicating modules smaller than the one the build enforces.
#
# --cov-report=json is what makes the per-module claim possible at all: it writes the
# coverage.json that the `coverage-floor` target below adjudicates. Note that
# --cov-fail-under=90 on this line is still a TOTAL, and a total can be subsidised --
# that is #228, and the reason the next target exists.
test:
	$(PY) -m pytest --cov=foundationscale --cov=tools --cov-report=term-missing --cov-report=json --cov-fail-under=90

# Ordered AFTER `test` in the `check` aggregate, and it must stay there: it consumes
# the coverage.json that `test` writes. Self-test first, then the real run -- the same
# pairing every other checks/ gate uses, because a detector whose controls misbehave
# has no licence to report a verdict. If coverage.json is missing the gate exits 95
# (UNMEASURED), never 0: "the report was not there" is not "every module passed".
coverage-floor:
	$(PY) checks/coverage_floor.py --self-test
	$(PY) checks/coverage_floor.py

# Finding #253. Every CI job that EXECUTES the pytest suite must install the same
# extras. The measured failure: #228 added tests/test_train_execution.py, the
# `check` job got `[train]`, the `mutation` job -- which runs the WHOLE suite once
# per mutant -- did not, and all 9 shards died at COLLECTION with "No module named
# 'tokenizers'". The battery reported that as `assert 96 == 5`, so an unmeasured
# mutant read as a WRONG VERDICT rather than as a missing dependency.
#
# The gate's denominator is jobs that run pytest or tools/mutate.py, NOT jobs that
# merely mention them; a job installing pytest-cov without ever invoking pytest is
# a MUST_PASS control, because widening the denominator to "mentions pytest" is how
# this kind of scanner starts reddening jobs it has no claim over. Zero or one such
# job exits 95, never 0: agreement across an empty set is `all([])`.
ci-suite-extras:
	$(PY) checks/ci_suite_extras.py --self-test
	$(PY) checks/ci_suite_extras.py

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
	$(PY) -m ruff check src tests tools checks
	$(PY) -m ruff format --check src tests tools checks

fmt:
	$(PY) -m ruff check --fix src tests tools checks
	$(PY) -m ruff format src tests tools checks

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
# implied one (#231): all 8 files under checks/ are unchecked by mypy. That
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
	$(PY) -m mypy src tools/emit_run_manifest.py tools/live_save_gate.py tools/real_checkpoint_probe.py

# Not part of `check`, and that is the point: this REPORTS, it does not gate.
# It exists so the count above can stay unstated and still be knowable.
typecheck-checks:
	-$(PY) -m mypy checks

# $(PY), not bare `python`: bare `python` does not exist on modern macOS or most
# Linux distributions, so `make check` died with command-not-found for any
# developer who had not activated a virtualenv. CI never saw it — setup-python
# provides `python` — so the only machine this file exists to serve was the
# only machine it did not run on (#232). #247 closed the same defect one layer
# out, for the TOOLS: see the $(PY) block at the top of this file.
controls:
	$(PY) -m foundationscale.gates.controls

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
	$(PY) checks/packaging_reachability.py --self-test
	$(PY) checks/packaging_reachability.py

# Finding #245. Four review documents asserted the package "contains no
# training code" -- true of training PRIMITIVES, false of the package, which
# delegates to transformers.Trainer through function-scope imports. This probe
# reports the two axes separately so neither can be quoted alone, and scans
# every git-tracked *.md (not just docs/: README.md carries the claim too) for
# the retired phrasings.
#
# Self-test first, same order as packaging: an instrument whose controls have
# not run has not earned the right to have its verdict read.
training-plane:
	$(PY) checks/training_plane_probe.py --self-test
	$(PY) checks/training_plane_probe.py

# Finding #247. The gate that makes the bare-tool class un-reintroducible: it
# reads THIS file and refuses any recipe line that invokes pip, pytest, ruff,
# mypy, coverage or a bare interpreter by name instead of through $(PY).
#
# It exists because #232 was fixed by editing four lines, and editing lines does
# not close a class -- five more instances of the same defect were sitting in
# the same file the whole time, and the next one would have arrived the next
# time someone added a target. A fix with no detector is a fix with a half-life.
#
# Self-test first, same order and same reason as packaging and training-plane.
makefile-tooling:
	$(PY) checks/makefile_tooling.py --self-test
	$(PY) checks/makefile_tooling.py

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
	$(PY) checks/countables_drift.py --self-test
	$(PY) tools/countables_census.py --self-test
	$(PY) tools/countables_census.py --no-coverage --out $(CENSUS)
	$(PY) checks/countables_drift.py --census $(CENSUS) $(COUNTABLES_CORPUS)

# Finding #254. CI runs launchers/test_launcher_contracts.sh; until this target
# existed, no `make` goal did, so the largest gate in the repository was one a
# developer could not run before pushing. That is the #230 asymmetry with the
# arrow reversed -- there the Makefile was the stronger of the two mirrors and
# CI the weaker; here CI enforced a contract the mirror could not even reach.
#
# It was not a theoretical gap. The suite's anti-orphan leg scans every
# launchers/*.py + checks/*.py for call sites and refuses a file that has none.
# #228 committed checks/coverage_floor.py with `make check` fully green, and CI
# indicted it as an orphan on the next push -- the second time (#238 was the
# first) that a new gate file was discoverable only after the commit existed.
#
# Cost, measured rather than assumed: 27.8s wall / 4.1s user on the developer
# machine, 146 controls. The first draft of this comment said "minutes, because
# it runs real watchdog legs with wall budgets" -- plausible, and wrong; the
# watchdog legs run their budgets concurrently, which is why user time is a
# sixth of wall. It is cheap enough that there is no argument for keeping it
# out of `check`. (#93 is the reason the legs are not made cheaper still: one
# was, and went load-sensitive -- red at 7/8 under parallel load, 8/8 alone.)
launcher-contracts:
	bash launchers/test_launcher_contracts.sh

# Finding #257. The checks/*.py gate self-tests were split out of the launcher
# suite: they certify the repository's own gate scripts, not the launchers, and
# appending each new one to a file named for the launchers is what grew that
# file to 5,562 lines. Both halves must run -- the anti-orphan leg's corpus is
# the two suites concatenated, so running only one indicts every helper called
# solely from the other. That is why this target is in `check` beside its
# sibling and not merely available to type.
#
# Cost: 1.5s wall / 0.9s user, 19 controls. It is fast because the wall-budget
# watchdog legs all stayed on the launcher side.
checks-gates:
	bash launchers/test_checks_gates.sh

mutation:
	FS_FORBID_SKIPS=1 $(PY) tools/mutate.py

# One shard, the way CI runs it after #242. `make mutation-module MODULE=dcp`.
# Every module carries its own "must_survive" row precisely so a shard is a
# whole detector rather than half of one -- run without a control, mutate.py
# exits 2 (never measured) rather than printing a caught= figure it cannot
# support, and before #242 that is what 5 of the 9 shards would have done.
mutation-module:
	@test -n "$(MODULE)" || { echo 'usage: make mutation-module MODULE=<name>   ("$(PY) tools/mutate.py --list" names them)'; exit 2; }
	FS_FORBID_SKIPS=1 $(PY) tools/mutate.py --module $(MODULE)

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
	output=$$(FS_FORBID_SKIPS=1 $(PY) -m pytest tests/test__skip_guard_probe.py 2>&1); \
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

# Finding #247, second half. `training-plane` had a target, a .PHONY entry and
# two CI steps (ci.yml:276, ci.yml:278) and was absent from this line, so the
# one command a developer runs before pushing was WEAKER than the CI it claims
# to mirror -- #230's shape, in the file that states the mirror as its purpose.
# A gate reachable only by typing its name is reachable by nobody: #238's
# orphan class, one layer up from the gate files it was written about.
check: lint typecheck skip-guard-probe test coverage-floor ci-suite-extras controls packaging training-plane makefile-tooling countables launcher-contracts checks-gates mutation

clean:
	rm -rf build dist .eggs src/*.egg-info *.egg-info \
		.pytest_cache .mypy_cache .ruff_cache .cache \
		.coverage .coverage.* coverage.xml coverage.json htmlcov .countables_census.json
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
