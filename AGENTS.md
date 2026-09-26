# AGENTS.md — working agreement for AI agents in this repository

FoundationScale's doctrine is load-bearing: claims are MEASURED or REFUSED;
ambiguity is an ABSTENTION, never a guess; skips are failures
(`FS_FORBID_SKIPS=1`); the verdict taxonomy is 0/5/95/96 and nothing else on the
training and check surfaces.

## Non-negotiables
- Exit codes on train/check/gate CLIs: 0 pass, 5 red-measured, 95 unmeasured, 96
  refused. A new path returning anything else (incl. 1) is a defect.
- Refusals name the missing input, never retry with a guessed value.
- Manifest/model/write paths are rank-scoped and atomic (see train/loop.py
  `_emit_manifest`, `_effective_rank`); per-rank returns must go through the
  collective agreement helpers.
- Every gate ships controls: >=1 MUST_FIRE (must block broken input) and
  typically MUST_PASS. A detector without a firing control is unproven.
- Prose numbers come from the measured census (`tools/countables_census.py` +
  `checks/countables_drift.py --fix`), never from memory.

## Standard workflow for a change
1. Understand: read the module's own docstring contracts first (they carry the
   measured history); use the coverage floors and mutation corpus as the map of
   what "tested" means here.
2. Research before building (see §Research rule below).
3. Implement with refusal surfaces explicit; run focused tests, then the battery.
4. Review before "done": the Open Code Review deck (`.opencodereview/rule.json`)
   runs in delegation mode; every finding needs a disposition in
   `artifacts/evidence/<commit>/review.json` on the estate copy of this repo.
5. Documentation sync in the SAME commit: docstrings that describe old behavior,
   countables numbers, manifest schema prose.
6. Commit only after `make check` is green on the exact tree committed.

## Battery invocation
```
make check PY=<python-with-extras>     # PY override required outside a repo .venv
```
Hidden-order matters: `test` must precede `coverage-floor` (the latter consumes
coverage.json); `mutation` is last and mutates the **live tree** -- during a
mutation window nobody reads/edits/commits this checkout, and two batteries never
overlap.

## Research rule (rule-of-reuse)
Before building non-trivial machinery, survey existing solutions and decide
explicitly integrate / adapt / reference-only / build-custom:
- triggers: unfamiliar domain, mature OSS territory, perf/kernel/parallelism
  mechanisms, anything whose reference implementation you have not read;
- on this estate, vetted stacks exist locally (e.g. Megatron-Bridge under the
  training workspace; NeMo sqsh images under SQSH-env): check them first;
- the survey record (problem, candidates, comparison, decision) lands in
  `artifacts/research/<topic>.md` beside the estate; a decision without the
  record is an unmeasured claim about ourselves. "Not invented here" is not a
  defense, and neither is reinventing a measured wheel.
- Licenses are part of the decision, stated, not footnoted.

## What agents may not do alone (human-required)
Doctrine/verdict changes, public capability claims, removal of refusal surfaces,
accepting P0/P1 review findings as won't-fix, dependency additions, release
tagging.
