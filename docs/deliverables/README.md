# FoundationScale — Codebase Analysis & Framework Architecture

> **Note on redaction.** This is the public copy. Cluster-internal identifiers —
> account names, home paths, node and partition names, Slurm job numbers — have been
> replaced with stable pseudonyms (`<user>`, `<CLUSTER_HOME>`, `<compute-node>`, `J07`),
> and the two audited codebases are referred to under one neutral family name as
> **`omni-accel`** and **`omni-bridge`**. The two are deliberately *not* merged into a
> single name: the difference between them carries evidence — `omni-accel` is not a git
> repository at all, which is why the recommendation is quarantine rather than deletion.
> Every pseudonym is consistent across all documents, so cross-references still resolve
> and every claim keeps its evidence. Nothing else has been altered, removed or
> softened; the failure record, the numbers and the retractions are as written.


Deliverables A–D for the two working training codebases
(`omni-accel`, `omni-bridge`), targeting the
Late-Fusion VLM ladder (B1–B4) and the advanced LLM ladder (A0–A4).

| Doc | Deliverable | What it answers |
|---|---|---|
| [A1](A1_existing_system.md) | Existing System Analysis — structure | The two repos, workflows end-to-end (SFT / RL / export), infrastructure, dependency map, operational reality |
| [A2](A2_algorithms.md) | Existing System Analysis — algorithms | Every objective (SDPO/POLAR/GSPO/DPO/ODPO), the reward cascade, the three teacher mechanisms, and the **pipeline coverage matrix** |
| [A3](A3_debt.md) | Existing System Analysis — debt & risk | Duplication census, the **silent-failure catalogue**, blast-radius accounting, reproducibility, risk register |
| [B1](B1_architecture.md) | Architecture Proposal — core | The L0–L6 layered design, gate catalogue, run manifest, directory structure, and falsifiable claims |
| [B2](B2_scaling.md) | Architecture Proposal — unified LLM/VLM + scaling | One data contract, one Stage abstraction, the topology object, 1 GPU → 100+ nodes as a **ladder with measured rungs**, H100→GB200 abstraction |
| [C](C_mapping.md) | Codebase Mapping | Per-component **Existing → Reusable → Refactor → Wrap → Replace → Build-New**, plus the crosswalk from the brief's 15 boundaries to the L0–L6 planes |
| [D](D_roadmap.md) | Development Roadmap | Phases 1–10 with scope, deliverables, exit criteria, dependencies, risks and a **falsification condition** each |

The reasoning spine behind all seven — the architectural decisions, the evidence
grading, and the honest list of what remains unknown — is [`../DECISIONS.md`](../DECISIONS.md).
Read that first if you only read one.

## Evidence discipline

Every claim carries a tag, and the tags are load-bearing:

**[M]** measured · **[V]** verified by source inspection · **[A]** artefact census ·
**[K]** asserted by Kimi-K3, not independently confirmed · **[U]** explicitly unverified ·
**[V·post]** verified after the drafting prompt was frozen, applied by the tech lead

Two rules the documents apply to themselves:

1. **An unqualified count is not a fact.** Every duplication figure names its file set.
   The same file is "3 variants / 8 copies" or "18 md5s / 23 copies" depending on the
   question asked; both are correct and neither may be quoted bare.
2. **Every claim that something does not exist must name the positive control that
   proves its detector could have fired.** This rule exists because the audit violated
   it four times and was wrong each time — a file inventory that excluded `.jsonl`
   produced a confident retraction of a grader statistic that turned out to exist and
   reproduce exactly (κ = 0.9824); a `find` abandoned on a 23 TB tree declared a launcher
   nonexistent when it sits at `mb/sdpo_gemma4/smoke_refit_e4b.sh`; a grep for a literal
   path missed what the launcher writes as `$WORKSPACE/gspo`; and, most instructively of
   all, **the verification probe written to detect silent success silently succeeded** —
   it passed a corrupt artefact because the tensors it meant to compare were absent and
   `all([])` is `True` (A3 §2, S19). Three of the four were caught only by applying this
   rule; the fourth was caught by a negative control. None were caught by review.

## The headline findings

- **5 of the 10 target stages have never executed.** VLM B1/B2 never; B4 is a smoke test
  with an empty checkpoint dir; LLM A0 is a 12-iteration mock-data run (val PPL 4.3e5);
  A1 has zero logs. The system is more scaffolding than its directory structure suggests,
  so the roadmap is sequenced against *executed* capability.
- **Most of the GSPO campaign ran with no trust region at all.** `--old_logp_source`
  defaults to `self`, making the importance ratio identically 1.0 — 7 of 10 official arms.
- **The dominant failure mode is silent, and it recurs across every subsystem** —
  checkpointing, export, rewards, the trust region, and throughput. The most vivid single
  artefact: 1,876 steps of training in which 472 steps had `grad_norm` exactly 0.000 while
  the run logged `reward/mean=0.794, success=1.00`.
- **The 8-node ceiling is a property of this codebase, not the hardware.** Other users on
  the same cluster run 18 nodes / 72 GPUs.
- **The bytes are right; whether the models run is still unknown.** A post-draft sweep
  verified 200 of 205 export directories tensor-by-tensor against Megatron at **0 DIFFER**,
  and proved the expert fix numerically correct (3,840/3,840 experts bitwise identical,
  controls firing at 128 and 112) — on CPU, for free. **No exported artefact has ever been
  asked to produce a token.** That gap is one GPU and twenty minutes, and it is the last
  thing standing between the estate and a defensible correctness claim.

## Still open

Items listed as **[U]** in `../DECISIONS.md §5` are stated as unknown wherever they
appear. **The two that were said to gate Phase 1 are now closed, and were closed with no
GPU at all** — the DCP `.metadata` offset table makes individual tensor chunks
byte-range-readable, so a login node was enough. The expert fix is numerically correct
(3,840/3,840 experts bitwise identical across 30 layers; injected controls fired at 128
and 112), and **200 of 205 export directories** — the denominator 23 was also wrong —
are verified tensor-by-tensor against Megatron at **0 DIFFER**, with LoRA merges
reconstructing at max error exactly 0.0.

What remains is one thing, stated precisely: **no exported artefact has ever been asked
to produce a token.** Bitwise-identical weights survive a wrong rope base, a mismatched
attention kernel, a tokenizer drift and a bad generation config. That probe is one node,
one GPU, ~20 minutes, and it should land as the export path's first semantic gate rather
than as a one-off script.

One footnote deserves to outlive the numbers. The verification tool **shipped the bug it
was written to find**: on a corrupt artefact it reported `all_identity: true` because the
expert tensors were absent and `all([])` is `True`. A negative control caught it. Every
gate in B1 is specified to assert *positive work* — a non-zero comparison count, an
explicit vacuous verdict — for exactly this reason.
