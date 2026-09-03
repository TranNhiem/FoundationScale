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
| 1 | Complete codebase review | `D1_codebase_review.md` | **Pending** — regenerating |
| 2 | Current architecture diagram | [`D2_current_architecture.md`](D2_current_architecture.md) | **Landed** |
| 3 | Problems and weaknesses | [`D3_problems_weaknesses.md`](D3_problems_weaknesses.md) | **Landed** |
| 4 | Feature evaluation (Keep / Simplify / Redesign / Remove / Missing) | [`D4_feature_evaluation.md`](D4_feature_evaluation.md) | **Landed** |
| 5 | Proposed architecture | `D5_proposed_architecture.md` | **Pending** — regenerating |
| 6 | Developer-experience review | `D6_developer_experience.md` | **Pending** — regenerating |
| 7 | Prioritized roadmap (P0–P3) | `D7_roadmap.md` | **Pending** — regenerating |
| 8 | Implementation of highest-priority improvements | tracked in the issue log | **In progress** |
| 9 | Rewritten README | `/README.md` | **Pending** |
| 10 | Validation that H100 and GB200 still work | `D10_validation_protocol.md` | **Pending** — regenerating |

## The headline finding

`src/foundationscale/` measures 18,706 lines and contains no training code. Measured across the
whole package: zero files define an `nn.Module`, call `backward()`, construct a `DataLoader`,
or define a `forward`; zero import torch at module scope. What is shipped is a *verification
plane* — gates, checkpoint readers, a parity comparator, topology validation, provenance
manifests — and it is a good one. What is documented is a training framework.

That gap is the organising fact of D2, D3 and D4, and it is the reason the roadmap leads with
naming the product honestly rather than with refactoring.

Two things about that number are worth stating plainly, because both were caught during the
review rather than after it:

- It moved *during* the review. These documents were written against a census of 13,667 lines.
  The T2 boundary move then relocated the 2,546-line checkpoint decision API out of
  `tools/live_save_gate.py` and into `src/foundationscale/gates/adjudication.py`, and the fixes
  landed since have added the rest: `src/foundationscale/` now measures 18,706 lines. The
  structural finding survived re-measurement unchanged: the package now holds real decision
  logic where it previously held none, and still holds no training code.
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
