# Convenience only; source of truth is pyproject.toml plus .github/workflows/ci.yml — CI does not run `make`, so targets mirror the CI steps exactly.
#
#   * `install` installs .[checkpoint,dev]: [checkpoint] carries torch, without which 41 tests skip. pytest-cov rides along because `test` carries coverage flags.
#   * `test` mirrors CI's pytest invocation, coverage floor included.
#   * FS_FORBID_SKIPS is the ONE declared divergence from CI: CI's `check` job sets it job-wide so any skip fails; `make test` leaves it unset so a laptop may skip (tests/conftest.py still names every skip). CI's exact view: `FS_FORBID_SKIPS=1 make test`.
#   * `skip-guard-probe` mirrors CI's probe step: it generates a deliberately skipped test and asserts the armed guard fails the run AND names it. It is a compound recipe because the CI step is a script; dropping it from `make check` would make `check` weaker than CI.
#   * `mutation` runs the WHOLE corpus (a surviving mutant fails it). CI now shards per module (#242); `mutation-module` mirrors one shard.
#
#     The mutation corpus is 144 rows over 19 modules.
#     Of those, 125 are MUST_FIRE mutants and 19 are MUST_PASS controls.
#
#     Each scoreable row runs the whole suite, so wall time is one full pytest run per MUST_FIRE row. All four counts above are anchored (total_rows, mut_modules, must_fire, must_pass in the countables census) and each sits on its own line, because the gate reads a clause, not a paragraph.
#
#     WALL TIME is not anchored (it is a property of the machine — the #83/#111 shape). Measured on one module (`--module checkpoint_gates`, 5 rows, 3m19s on an M-series laptop, ~40s per row); extrapolated, the full corpus is near 50 minutes. Stated deliberately: an understated cost is the reason a developer skips the one target that certifies the detectors.

# ---------------------------------------------------------------------------
# Interpreter selection: two decisions (#247).
# ---------------------------------------------------------------------------
# Every tool is invoked as `$(PY) -m <tool>`, never as a bare name on PATH: a
# bare name resolves only if the tool happens to be on PATH, which on the
# machine this file serves (a laptop with no venv activated) it is not. `-m`
# also binds each tool to the SAME interpreter as the rest of the file.
#
# Second decision: `PY ?= python3` does not work, because a bare `python3` has
# none of the tools installed either (measured: Homebrew and system python3
# lack ruff/pytest/mypy/coverage; only .venv/bin/python3 has all five). The
# tools live in the project venv, which is where `make install` puts them.
#
# Hence: prefer the in-tree venv when one exists, fall back to `python3`, and
# let an explicit PY beat both. `?=` is not used because a recursively-expanded
# `$(shell ...)` would re-run the probe at every reference; `origin` expands
# once and still yields to an environment or command-line PY.
#
#     make check                      # venv if present, else python3
#     make check PY=python3.11        # explicit wins
#     PY=/usr/bin/python3 make lint   # environment wins
#
# `checks/makefile_tooling.py` makes the bare-name class un-reintroducible. It
# does NOT gate which interpreter you chose or what it has installed: that is
# the #83/#111 machine axis, not a repository defect.
FS_VENV_PY := $(CURDIR)/.venv/bin/python3
ifeq ($(origin PY),undefined)
PY := $(shell test -x '$(FS_VENV_PY)' && printf %s '$(FS_VENV_PY)' || printf %s python3)
endif

.PHONY: install test coverage-floor ci-suite-extras lint fmt typecheck typecheck-checks controls packaging training-plane makefile-tooling mirror countables doc-pointers citation-lines mutation-scope rl-static-subtypes verification-matrix exit-contract-scope launcher-contracts checks-gates standing-gates control-scratch-restore campaign-self-tests mutation mutation-module skip-guard-probe check clean

# [train] is here because CI's suite jobs install it and this target is the
# developer's mirror of them. Without it `make install` provisions a WEAKER
# environment than CI, so `make check` can pass locally on a machine CI would
# fail — the PATH-dependent-verdict class (#83/#111/#229), and the same
# missing-extra that #253 found in the mutation job.
install:
	$(PY) -m pip install -e ".[checkpoint,train,dev]" "pytest-cov>=5" --extra-index-url https://download.pytorch.org/whl/cpu

# --cov=tools so the Makefile measures what CI measures.
#
# --cov-report=json writes the coverage.json that the `coverage-floor` target
# adjudicates per module. Note --cov-fail-under=90 on this line is still a
# TOTAL, and a total can be subsidised (#228) — the reason the next target
# exists.
test:
	$(PY) -m pytest --cov=foundationscale --cov=tools --cov-report=term-missing --cov-report=json --cov-fail-under=90

# Ordered AFTER `test` in the `check` aggregate and must stay there: it consumes
# the coverage.json `test` writes. Self-test first, then the real run. A missing
# coverage.json exits 95 (UNMEASURED), never 0: "the report was not there" is
# not "every module passed".
coverage-floor:
	$(PY) checks/coverage_floor.py --self-test
	$(PY) checks/coverage_floor.py

# Finding #253. Every CI job that EXECUTES the pytest suite must install the same
# extras. The measured failure: #228 added tests/train/test_train_execution.py, the
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

# checks/ must be linted: a PEP 604 annotation in a class body is evaluated at
# definition time, so without `from __future__ import annotations` a gate dies
# at IMPORT with exit 1 on Python 3.9 — outside the 0/5/95/96 namespace it
# claims to publish. `examples` must match CI's lint set exactly: a four-word
# gap means `make lint` green and push, CI red.
#
# ONE VARIABLE, both recipes. Stating the denominator twice is how the gap
# opened; the Makefile<->CI pairing is still unpinned by any gate.
LINT_PATHS := src tests tools checks examples

lint:
	$(PY) -m ruff check $(LINT_PATHS)
	$(PY) -m ruff format --check $(LINT_PATHS)

fmt:
	$(PY) -m ruff check --fix $(LINT_PATHS)
	$(PY) -m ruff format $(LINT_PATHS)

# Run this in the SAME environment as `install` creates — one with [checkpoint].
# mypy without torch checks a different program: MetadataIndex becomes Any and
# `Any | None` over `str` stops being an error. A torch-free typecheck passed
# this tree; CI (with torch) failed it on all three Pythons.
#
# tools/real_checkpoint_probe.py is a thin CLI over foundationscale.gates.probe:
# a boundary wrapper that is not typechecked is a re-export list nobody reads.
#
# checks/ is no longer excluded from here — `typecheck-checks` below GATES it.
# Note Distribution.entry_points is a LIST before 3.10 and an EntryPoints after,
# never the mapping that .get() addresses (that is the module-level
# entry_points() FUNCTION): the former back-compat arm in
# packaging_reachability.py was dead on 3.10+ and an AttributeError on 3.9.
typecheck:
	$(PY) -m mypy src tools/emit_run_manifest.py tools/live_save_gate.py tools/real_checkpoint_probe.py

# In `check`, and NOT prefixed with `-`: an ignored error makes the target green
# while reporting failures — a gate that cannot fail reports no verdict.
typecheck-checks:
	$(PY) -m mypy checks

# $(PY), not bare `python`: bare `python` does not exist on modern macOS or most
# Linux distributions, so only a developer without an activated venv runs here
# and `make check` would die with command-not-found. See the $(PY) block above.
controls:
	$(PY) -m foundationscale.gates.controls

# Mirrors the two steps in CI's `controls` job. Self-test first: a detector whose
# controls misbehave has no licence to report a verdict. Deliberately NOT
# `&&`-joined: CI runs them as two steps and a combined recipe would hide which
# half failed.
#
# Existence is resolved against the interpreter's own script directory and the
# install record — `shutil.which` reports false REDs whenever PATH is not
# sourced. PATH is reported as operator convenience and can never be red.
packaging:
	$(PY) checks/packaging_reachability.py --self-test
	$(PY) checks/packaging_reachability.py

# The package "contains no training code" is true of training PRIMITIVES but
# false of the package, which delegates to transformers.Trainer through
# function-scope imports. This probe reports the two axes separately and scans
# every git-tracked *.md (README.md carries the claim too) for the retired
# phrasings.
#
# Self-test first, same order as packaging: an instrument whose controls have
# not run has not earned the right to have its verdict read.
training-plane:
	$(PY) checks/training_plane_probe.py --self-test
	$(PY) checks/training_plane_probe.py

# Reads THIS file and refuses any recipe line that invokes pip, pytest, ruff,
# mypy, coverage or a bare interpreter by name instead of through $(PY) —
# editing lines does not close the bare-tool class; only a detector does.
#
# Self-test first, same order and same reason as packaging and training-plane.
makefile-tooling:
	$(PY) checks/makefile_tooling.py --self-test
	$(PY) checks/makefile_tooling.py

# The Makefile and CI workflow must run the same checks. This gate compares WHICH
# check scripts run on each side, not their argv: the sides legitimately differ
# in output paths, sharding, and whether `--self-test` is direct or via
# campaign_self_tests.py. Its printed banner states that narrowing and both of
# its blind spots.
#
# Self-test first, same order and same reason as makefile-tooling above.
mirror:
	$(PY) checks/makefile_ci_mirror.py --self-test
	$(PY) checks/makefile_ci_mirror.py

# Mirrors the three steps in CI's countables leg (#220). The census is measured,
# never committed (see ci.yml: a frozen oracle is worse than no oracle); it lands
# in the working tree, so `clean` removes it and .gitignore keeps it out of a
# commit. It counts tools/countables_census.py itself (tools_loc means "LOC
# under tools/", and the producer lives there) — editing the producer moves
# tools_loc, exactly the drift this gate exists to catch.
#
# The scan set is `docs README.md Makefile .github/workflows/ci.yml`: a directory
# argument is walked for *.md, but a file named outright is scanned whatever its
# suffix. The build configuration is inside the denominator it decides.
CENSUS := $(CURDIR)/.countables_census.json
COUNTABLES_CORPUS := docs README.md Makefile .github/workflows/ci.yml

countables:
	$(PY) checks/countables_drift.py --self-test
	$(PY) tools/countables_census.py --self-test
	$(PY) tools/countables_census.py --no-coverage --out $(CENSUS)
	$(PY) checks/countables_drift.py --census $(CENSUS) $(COUNTABLES_CORPUS)

# The README is a contents page, and its chapter pointers declare documents that
# must exist. A pointer is a DECLARATION with no denominator — the count can
# only grow while nothing reads it.
#
# Reports each extractor's sub-count separately, so an extractor that goes inert
# shows a zero instead of a green. Two notations are supported (arrow and
# bracket); one alone leaves pointers in no denominator.
#
# Fenced blocks and inline-code spans are masked before extraction (a path in
# code formatting is SHOWN, not followed). The masked-span count is printed so
# the narrowing stays measured rather than silent.
#
# Self-test first, same order and same reason as packaging and makefile-tooling.
doc-pointers:
	$(PY) checks/doc_pointers.py --self-test
	$(PY) checks/doc_pointers.py

# The repository cites its own source as `path.ext:N` and `path.ext:N-M`; a
# citation rots silently whenever the file it names is edited unless those
# tokens sit in a denominator.
#
# The claim is narrow: it proves the cited LINE EXISTS, not that the line says
# what the prose claims. Resolve counts are printed next to the verified count so
# they are never read as "citations are correct".
#
# No allowlist: the `name:rc` arguments to make_stub_campaign and `site` literals
# in fs_required_knobs.py cannot be told apart from citations BY SHAPE — their
# paths name no tracked file, so they resolve to UNMEASURED alongside genuine
# out-of-repo citations.
#
# Self-test first (same reason as doc-pointers) and it carries a control on
# itself: with the range comparison neutered, every MUST_FIRE control must go
# silent, or the harness refuses 96.
citation-lines:
	$(PY) checks/citation_lines.py --self-test
	$(PY) checks/citation_lines.py

# The mutation battery's module scope is the set of files whose RULES are checked
# against mutants — distinct from coverage-floor, whose LINES execute. A file
# added to the tree and to no mutation map is certified by nothing; a pinned
# MODULE_PATHS count stops the map SHRINKING and says nothing about the tree it
# is meant to reach.
#
# The gate partitions the tracked index into covered + pending + out-of-scope and
# requires every file in exactly one. PENDING_ENROLMENT is an UNMEASURED
# population COUNTED, not hidden — its size prints on every CLEAR line.
#
# Self-test first, same reason as doc-pointers and citation-lines. Mirrored by a
# CI step held in the same commit by hand (#286, no Makefile<->CI gate, still
# open).
mutation-scope:
	$(PY) checks/mutation_scope.py --self-test
	$(PY) checks/mutation_scope.py

# The RL plane ships claims of the shape "this concrete binding IS a `Foo`".
# runtime isinstance() against a @runtime_checkable Protocol only checks that the
# named methods EXIST — a binding whose `step` has wrong argument/return types
# passes. This gate adjudicates nothing itself: it discovers registry._REGISTRY
# claims statically with `ast` (walked, not read top-level, because registry.py
# imports every binding INSIDE _install_default_algorithms() to break a cycle),
# writes a probe module asserting each one, and hands the verdict to mypy.
#
# A committed baseline guards the COMPLETENESS axis a subtype checker cannot see:
# a deleted or renamed binding makes the remaining claims MORE true, so
# discovery shrinking is RED.
#
# Self-test first (same reason as citation-lines and mutation-scope). mypy absent
# is 95 UNMEASURED, never a silent green.
rl-static-subtypes:
	$(PY) checks/rl_static_subtypes.py --self-test
	$(PY) checks/rl_static_subtypes.py

# validation_campaigns/verification_matrix/matrix.json is the ledger of what this
# repository claims to have verified; docs/VERIFICATION_MATRIX.md is its
# rendering. Two failure modes: the two drift (hand-edited, either direction),
# and a row rots in place — its control_arm slides from a real arm to a
# placeholder while presence checks stay green. A row with no arm that must come
# out DIFFERENT is a row that cannot fail.
#
# C5 makes the doc DERIVED rather than mirrored: table rows are compared
# cell-for-cell against the JSON, with markdown ticks normalised on BOTH sides
# (one-sided normalisation manufactures drift). C4 reads the honest-count
# paragraph POSITIONALLY, so every integer there is a countable — which is why
# the two tiers are named in words, not by numeric labels.
#
# Self-test first (same reason as doc-pointers, citation-lines, mutation-scope).
# SC6 (unmodified files must be CLEAR) makes SC1-SC5 mean anything; SC8 pins RED
# over UNMEASURED precedence.
verification-matrix:
	$(PY) checks/verification_matrix.py --self-test
	$(PY) checks/verification_matrix.py

# Everything cli.main does BEFORE `return train(cfg)` must be guarded over the
# 0/5/95/96 contract (a TypeError in parser construction exits 1 — the one code
# the contract has no room for). tests/train/test_train_boundary_is_total.py pins
# this behaviourally; this gate pins it STATICALLY over three axes (RETURN,
# MODULE-EXIT, ESCAPE), because the defect class is "a statement was added and
# nobody wrapped it" — a test covers only the sites it was written for.
#
# The predecessor (validation_campaigns/h100_validation/
# patch_trainer_exit_contract.py) scopes a GENERATED artifact, leaving SHIPPED
# entry points in no exit-contract denominator. This one takes the repository
# root as a positional and reads src/, which also lets its MUST_FIRE leg plant
# pre-fix shapes into a COPY rather than the tree it guards.
#
# Self-test first, same order and same reason as the targets above.
exit-contract-scope:
	$(PY) checks/exit_contract_scope.py --self-test
	$(PY) checks/exit_contract_scope.py

# Runs launchers/test_launcher_contracts.sh, which CI runs but no other make goal
# does — the largest gate in the repository must be runnable before pushing.
# Its anti-orphan leg scans every launchers/*.py + checks/*.py for call sites and
# refuses a file with none, so a new gate file is discoverable before it is
# committed.
#
# Cost: 27.8s wall / 4.1s user, 146 controls. Cheap enough for `check`. The wall-
# budget watchdog legs run concurrently; they must not be made cheaper still
# (#93: one was, and went load-sensitive — red at 7/8 under parallel load, 8/8
# alone).
launcher-contracts:
	bash launchers/test_launcher_contracts.sh

# The checks/*.py gate self-tests, split out of the launcher suite: they certify
# the repository's own gate scripts. BOTH halves must run — the anti-orphan
# leg's corpus is the two suites concatenated, so running only one indicts every
# helper called solely from the other.
#
# Cost: 9.4s wall / 5.8s user, 39 controls. Cheaper than its sibling (no wall-
# budget watchdog legs); each new checks/*.py gate lands its MUST_PASS/MUST_FIRE
# pair here.
checks-gates:
	bash launchers/test_checks_gates.sh

# Runs the nine standing gate_*.py scripts in the H100 campaign, which only
# build_h100_plane.sh invokes and no make target or CI step runs. Executed
# directly, read-only, in seconds — NOT the build, which without the private
# upstreams refuses at stage one and says nothing about these gates.
#
# --self-test first, and its rc gates the real run (`&&` is deliberate): a runner
# with broken controls cannot certify anything.
#
# --allow-refused NAMES the one gate that cannot measure without the estate
# environment (absent here by design and never in a public checkout). A named set
# still reds when anything ELSE goes quiet — a bare `|| true` would swallow a new
# refusal too.
#
# Cost: 6.5s wall, nine gates.
standing-gates:
	bash validation_campaigns/h100_validation/run_standing_gates.sh --self-test && \
		FS_ESTATE_IDENT_PAT=NONE bash validation_campaigns/h100_validation/run_standing_gates.sh \
			--allow-refused gate_launch_doc.py

# The build snapshots eight tracked artifacts before its `rm -f` and restores
# whatever no stage regenerated, so a refusing stage no longer leaves a clone
# with fewer files than it started with. This is the two-arm measurement of that
# claim, separate from campaign-self-tests: the self-tests use fixtures; this
# runs the real build script twice.
#
# Arm A is the tree as committed. Arm B is the same build with the two restore
# anchors stripped (pre-fix). Both arms must REACH the rm site (L1) — an earlier
# refusal would leave 8/8 intact and look like a pass. L5 must show arm B losing
# 8/8: if both survive, the fix is not what saved arm A.
#
# Cost: 6s wall (both arms refuse at stage one without private upstreams). The
# only target here that EXECUTES the build script rather than reading it, so it
# stays a named target.
control-scratch-restore:
	$(PY) validation_campaigns/h100_validation/control_scratch_restore.py --self-test && \
		$(PY) validation_campaigns/h100_validation/control_scratch_restore.py

# pyproject.toml sets testpaths = ["tests"], so pytest collects NOTHING under
# validation_campaigns/. Campaign modules ship a --self-test flag (plant
# controls, return 0 only if every control passes) that no pytest collection
# reaches — this target invokes them. The population is measured, not numbered
# here.
#
# --self-test first, and its rc gates the real run.
#
# Scope is MEASURED (an ast '--self-test' literal), not hand-listed. The rules are
# two-way so the denominator cannot shrink silently: R1 reds a measured file that
# is undeclared, R6 reds a declared file no longer measured. Campaign files with
# no self-test need no reason strings — they are outside the claim by
# measurement.
#
# Cost: ~2s wall spawning each self-test as a subprocess.
campaign-self-tests:
	$(PY) checks/campaign_self_tests.py --self-test && $(PY) checks/campaign_self_tests.py

# `campaign-self-tests` runs control_boundary_classification's --self-test
# against SYNTHETIC subjects. The entry that runs it against the REAL rows is
# `--expect postfix`, and until this target existed no make goal or CI step
# invoked it — the rule was certified in the abstract and never on the shipped
# code. The real-row mode needs the rows importable and nothing else (every leg
# patches the measurement delegate out before calling main), so it runs in
# seconds on a laptop.
boundary-control:
	cd validation_campaigns/verification_matrix && \
		$(PY) control_boundary_classification.py --self-test && \
		$(PY) control_boundary_classification.py --expect postfix

mutation:
	FS_FORBID_SKIPS=1 $(PY) tools/mutate.py

# One shard, the way CI runs it after #242. `make mutation-module MODULE=dcp`.
# Every module carries its own "must_survive" row so a shard is a whole detector
# — run without a control, mutate.py exits 2 (never measured) rather than
# printing a caught= figure it cannot support.
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

# The `check` aggregate must list every make target CI runs: a gate missing here
# is WEAKER than the CI the mirror claims to reflect (#230's shape). A gate
# reachable only by typing its name is reachable by nobody: #238's orphan class,
# one layer up.
check: lint typecheck typecheck-checks skip-guard-probe test coverage-floor ci-suite-extras controls packaging training-plane makefile-tooling mirror countables doc-pointers citation-lines mutation-scope rl-static-subtypes verification-matrix exit-contract-scope launcher-contracts checks-gates standing-gates control-scratch-restore campaign-self-tests boundary-control mutation

clean:
	rm -rf build dist .eggs src/*.egg-info *.egg-info \
		.pytest_cache .mypy_cache .ruff_cache .cache \
		.coverage .coverage.* coverage.xml coverage.json htmlcov .countables_census.json
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
