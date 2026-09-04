# Comprehensive framework review

A step-back review of the whole codebase and its architecture, conducted after the initial
build was validated on H100 and GB200. It asks a different question from the audit that
produced `docs/deliverables/`: not *"is this correct?"* but *"is this the right thing, and
would an AI engineer be able to use it?"*

## Status

This set is being written incrementally. Each deliverable lands only after its claims have
been re-measured against the current tree — several early drafts asserted a state that the
T2 boundary move had already changed, and were corrected rather than published.

| # | Deliverable | File | Status |
|---|---|---|---|
| 1 | Complete codebase review | [`D1_codebase_review.md`](D1_codebase_review.md) | **Landed** |
| 2 | Current architecture diagram | [`D2_current_architecture.md`](D2_current_architecture.md) | **Landed** |
| 3 | Problems and weaknesses | [`D3_problems_weaknesses.md`](D3_problems_weaknesses.md) | **Landed** |
| 4 | Feature evaluation (Keep / Simplify / Redesign / Remove / Missing) | [`D4_feature_evaluation.md`](D4_feature_evaluation.md) | **Landed** |
| 5 | Proposed architecture | [`D5_proposed_architecture.md`](D5_proposed_architecture.md) | **Landed** |
| 6 | Developer-experience review | [`D6_developer_experience.md`](D6_developer_experience.md) | **Landed** |
| 7 | Prioritized roadmap (P0–P3) | [`D7_roadmap.md`](D7_roadmap.md) | **Landed** |
| 8 | Implementation of highest-priority improvements | tracked in the issue log | **In progress** |
| 9 | Rewritten README | [`/README.md`](../../README.md) | **Landed** |
| 10 | Validation that H100 and GB200 still work | [`D10_validation_protocol.md`](D10_validation_protocol.md) | **Landed** |

Deliverable 8 is the only one that is not a document, and it is deliberately open: the
highest-priority improvements are being implemented as numbered findings, each landing with
its own gate and controls rather than as a single refactor commit.

## The headline finding

`src/foundationscale/` measures 18,915 lines and implements **no training primitives of its
own**. Across all 25 git-tracked `src/*.py` files, an AST probe finds zero that define an
`nn.Module`, call `backward()`, construct a `DataLoader`, define a `forward`, call
`optimizer.step()`, or import torch at module scope.

**That zero is literally true and, read alone, materially misleading** — and it was read alone
when D2, D3 and D4 were first drafted. The package *does* ship a training entry point:
`src/foundationscale/train/` (`loop.py` 1,168 lines, `cli.py` 108, `__main__.py` 55, `__init__.py` 29) builds a
`transformers.Trainer` and calls `trainer.train()`, `trainer.save_model()`,
`AutoModelForCausalLM.from_pretrained` and `DataCollatorForLanguageModeling`. It imports torch
and transformers at *function* scope, so every module-scope marker the probe looks for stays at
zero. The package **delegates** training rather than implementing it, and six primitive markers
cannot tell delegation apart from not training at all (finding #223).

The accurate statement has two axes, not one:

| Axis | Measured |
|---|---|
| Training **primitives** implemented in `src/` | 0 of 25 files, on all six markers |
| Training **entry point** delegating to a third-party trainer | present — `train/`, 1,360 lines across 4 files |
| Gate seam into that trainer | present — `FoundationScaleSaveGate.on_save` runs the registry for `FIRST_SAVE`/`SAVE` and fails closed |
| Tests exercising the entry point | `train/loop.py` 62%, `train/__init__.py` 50%; every other module ≥81% (#228) |

The rest of the package is a *verification plane* — gates, checkpoint readers, a parity
comparator, topology validation, provenance manifests — and it is a good one, better tested
than the training path the framework is named for. That asymmetry, not an absence, is the
organising fact of D2, D3 and D4, and it is why the roadmap leads with naming the product
honestly rather than with refactoring.

**How the error survived.** `train/` arrived with #224/#225/#226, *after* those documents were
drafted, and no committed probe existed to go red when the repo moved — the "measured `src/`
probe" they cite was an ad-hoc campaign command, not an instrument. The countables gate could
not catch it either: it anchors numbers, not claims. Both defects are tracked as #245; the
passages are corrected in place below, and the probe now ships as
[`checks/training_plane_probe.py`](../../checks/training_plane_probe.py) — a real 0/5/95/96
instrument with nine controls across both axes, wired into the launcher suite, the `Makefile`
and CI in the same commit. It scans every git-tracked `*.md`, this file included, so the
retired phrasing cannot come back anywhere in the repository without turning a gate red.

Two further things about the 18,761 are worth stating plainly, because both were caught during
the review rather than after it:

- It moved *during* the review. These documents were written against a census of 13,667 lines.
  The T2 boundary move then relocated the 2,546-line checkpoint decision API out of
  `tools/live_save_gate.py` and into `src/foundationscale/gates/adjudication.py`, and the fixes
  landed since have added the rest: `src/foundationscale/` now measures 18,915 lines. The
  structural finding survived re-measurement in re-scoped form: the package now holds real
  decision logic where it previously held none, and a delegating trainer where it previously
  held nothing at all.
- The move did not finish the job it started. The library's decision path imports an
  unpackaged `tools/` module, so on a clean install it loads and then declines to decide
  (finding #219, measured with a positive control in D3 Theme 3).

## Relationship to `docs/deliverables/`

`docs/deliverables/` is the output of the earlier correctness audit — lettered A1/A2/A3,
B1/B2, C, D. It established *what the system does and where it is wrong*, and its roadmap
is scoped to closing findings.

This directory supersedes that roadmap where the two disagree. Where the audit asked whether
a gate fires correctly, this review asks whether the gate should be reachable from an
installable API at all. The audit's findings remain the evidence base; several are cited
directly here by their finding IDs.

## How to read a claim in here

Every claim carries its denominator, and a claim that could not be measured says so rather
than resolving to a pass. Where a supplied finding overstated its scope, the corrected form
is used and the correction is noted inline. Line numbers are deliberately omitted: paths are
checkable and stay true across edits, line numbers drift silently and had already drifted in
an earlier document.
