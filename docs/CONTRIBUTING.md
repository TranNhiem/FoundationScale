# Contributing

FoundationScale accepts contributions on two conditions: the change runs the same
battery CI runs, and any claim it makes carries its own coverage. Both are tested,
not requested. This document explains the workflow and the two standing review
rules, and states plainly where the machinery does not reach.

## Scope of this document

This page is the contributor-facing companion to
[docs/DEVELOPMENT.md](DEVELOPMENT.md), which documents each `make` target in
detail. Here the emphasis is on what a contributor owes a PR, why the review
rules exist, and how the drift gate polices numbers in shipped documents.

## Before opening a PR

Run:

```bash
make check
```

The aggregate runs every target CI runs: `lint`, `typecheck`, `typecheck-checks`,
`skip-guard-probe`, `test`, `coverage-floor`, `ci-suite-extras`, `controls`,
`packaging`, `training-plane`, `makefile-tooling`, `countables`,
`launcher-contracts`, `checks-gates`, and `mutation`. CI runs the same steps
across its matrix. The Makefile is convenience only — CI mirrors it, it does not
consume it — but `make check` is kept at parity on purpose. A gate reachable
only by typing its name on a machine where the developer forgets it exists is an
orphan; that is why the aggregate is required to stay complete rather than
merely available.

| Situation | What to run |
|---|---|
| Routine change before pushing | `make check` |
| Confirming laptop vs. CI skip behaviour | `FS_FORBID_SKIPS=1 make test` |
| One mutation shard while iterating | `make mutation-module MODULE=<name>` |
| Cleaning generated artefacts, including the census | `make clean` |

`make test` deliberately leaves `FS_FORBID_SKIPS` unset so a laptop may skip,
but `tests/conftest.py` still names every skip and its reason in the summary.
CI's `check` job sets the variable job-wide so any skip fails the build. To see
the suite exactly as CI sees it, export `FS_FORBID_SKIPS=1`.

### Interpreter selection

Every recipe invokes tools as `$(PY) -m <tool>`, never as a bare name on PATH.
Bare tool names resolve only when the project venv is activated, and on a
developer laptop they may not be. The Makefile resolves `PY` in this order:

1. An explicit `PY` from the command line or environment wins.
2. Otherwise the in-tree venv `.venv/bin/python3` is used if it exists.
3. Otherwise it falls back to `python3`.

Run `make install` first; that is what installs the tools into the project venv:

```bash
make install                      # editable install with [checkpoint,dev] + pytest-cov
make check PY=python3.11          # explicit interpreter wins
PY=/usr/bin/python3 make lint     # environment wins
```

Note that `$(PY) -m ruff` under an interpreter that lacks ruff fails with
`No module named ruff` rather than `ruff: command not found`; both errors mean
"wrong environment", and the fix is `make install`, not a PATH edit.

`make typecheck` must run in the environment `make install` creates, one with the
`[checkpoint]` extra installed. mypy without torch checks a different program:
`MetadataIndex` becomes `Any`, and an `Any | None` assigned over a variable
bound to `str` stops being an error. Same command, same source, different answer,
because the environment differed.

## Standing review rule 1: every count carries its denominator

Founding rule of the project: `all([])` is True, and a check that examined zero
units reports VACUOUS (or its stated non-zero exit), not PASS. Every count in a
shipped document must state what it is a count of.

The drift gate `checks/countables_drift.py` enforces this on shipped numbers:
state counts only in wordings it anchors. The workflow is:

```bash
make countables
```

which runs, in order:

```bash
python3 checks/countables_drift.py --self-test
python3 tools/countables_census.py --self-test
python3 tools/countables_census.py --no-coverage --out .countables_census.json
python3 checks/countables_drift.py --census .countables_census.json docs README.md Makefile .github/workflows/ci.yml
```

Practical consequences for contributors:

* **The census is measured, never committed.** It lands as
  `.countables_census.json` in the working tree; `.gitignore` keeps it out of
  commits and `make clean` removes it. Do not add it to a PR.
* **The scan set is** `docs README.md Makefile .github/workflows/ci.yml`. Note
  that the README, the Makefile, and the CI workflow are inside the denominator:
  the two files that decide what CI measures are counted like any prose
  document. A directory argument is walked for `*.md`; a file named outright is
  scanned whatever its suffix.
* **The census counts its own producer.** `tools/countables_census.py` is inside
  `tools_loc`. Editing the producer moves the number; that is intended, because
  that drift is exactly what the gate exists to catch.
* **One anchored number per line.** The gate reads a clause, not a paragraph: a
  number wrapped across a comment continuation is a number in no denominator.
  Keep each count on its own line beside the wording the census anchors.

The same rule shapes verdict gates: when a denominator is empty the gates exit
95 (UNMEASURED) rather than 0. `checks/coverage_floor.py` exits 95 if
`coverage.json` is missing, because "the report was not there" is not "every
module passed". `checks/ci_suite_extras.py` exits 95 on zero or one
pytest-executing job, because agreement across an empty set is `all([])`.
`tools/mutate.py` run on a shard with no MUST_PASS control exits 2 (never
measured) rather than printing a `caught=` figure it cannot support.

## Standing review rule 2: name the control proving absence

Every claim that something does not exist names the control proving its
detector could have fired. Write the MUST_FIRE control with the gate. A check
that has never been observed going red is not evidence.

In practice this means new gates follow the same self-test pairing the rest of
`checks/` uses:

```bash
python3 checks/<your_gate>.py --self-test
python3 checks/<your_gate>.py
```

Self-test first, then the real run — a detector whose controls misbehave has no
licence to report a verdict, so the real run must not be reachable without it.
The pair is kept as two recipe lines (not `&&`-joined) so a failure names which
half failed. The Makefile's `skip-guard-probe` target demonstrates the standard
at its sharpest: it generates a deliberately skipped test, asserts the armed
`FS_FORBID_SKIPS` guard fails the run, and asserts the failure names the probe
(`test_skip_guard_probe`). A red that cannot name what fired is not a
satisfying control either.

Controls must be real. Two failure modes the repository has already shipped and
fixed are worth internalising:

* A control whose premise was never checked. An early packaging-reachability
  gate asked `shutil.which(ep.name)` and reported both console scripts
  unreachable when the only fact it had measured was an unactivated venv. The
  resolved shape: existence is checked against the interpreter's own script
  directory and the install record; PATH is reported as operator convenience
  and can never be red.
* A control written off as "deliberate back-compat". An exclusion note called a
  `Distribution.entry_points.get` fallback a silenced-by-design shape; mypy was
  right and the note was wrong — `entry_points` has on every release been a
  list (pre-3.10) or an `EntryPoints` (3.10+), never the mapping `.get()`
  addresses. An excuse written into an exclusion is how a real defect keeps an
  unfailing gate.

## Landmarks and exclusions for new checks

| Rule | Consequence |
|---|---|
| Exit-code namespace | Gates publish 0 / 5 / 95 / 96. A gate dying at import with exit 1 is outside the namespace it claims to publish — this actually happened on Python 3.9 when `checks/` files lacked `from __future__ import annotations`. |
| Linting | ruff covers `src tests tools checks`; `checks/` was added precisely because its exclusion read as coverage. |
| Typechecking | `make typecheck-checks` gates `checks/` and is in `check`, not prefixed with `-`. A recipe that cannot fail while printing 22 errors is a suppression, not a report. |
| Call sites | The launcher suite's anti-orphan leg scans every `launchers/*.py` and `checks/*.py` for call sites and refuses a file that has none — a new gate committed without a call site is indicted by CI even with `make check` green on the authoring branch. |
| CI extras parity | If your change adds tests importing a new extra (the `#253` case was `[train]` / `tokenizers`), every job executing pytest needs the same extras; `make ci-suite-extras` checks this. Jobs that merely mention pytest without executing it are MUST_PASS controls, deliberately. |

## Editing the audit documents

The audit documents carry their own grading scheme — `[M]`, `[V]`, `[A]`,
`[K]`, `[U]` — and are governed by [docs/DECISIONS.md](DECISIONS.md). Read it
before editing them.

The source material for this document does not include docs/DECISIONS.md, so
the meanings of the five grades and the rules for transitioning between them
cannot be stated here. What can be stated from the README: state counts only in
wordings `checks/countables_drift.py` anchors, because the Makefile,
README.md, and `docs/` are all inside the drift corpus. If you retire a
phrasing in an audit document, expect `checks/training_plane_probe.py` to scan
every git-tracked `*.md` (not only `docs/`) for retired phrasings — README.md
carries claims too.

## License

FoundationScale is licensed under the [MIT License](../LICENSE). Contributions
are accepted under the same terms.
