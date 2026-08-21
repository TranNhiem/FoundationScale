<p align="center">
  <img src="assets/hero.png" alt="FoundationScale — a scalable distributed training framework for foundation models, from single-GPU experiments to large-scale multi-node training" width="100%">
</p>

<h1 align="center">FoundationScale</h1>

<p align="center">
  <b>Efficient &middot; Scalable &middot; Distributed &middot; Foundation Model &middot; Multimodal &middot; Small-to-Large Scale</b>
</p>

<p align="center">
  A scalable distributed training framework for foundation models,<br>
  from single-GPU experiments to large-scale multi-node training.
</p>

---

## Why this exists

FoundationScale is being built from a forensic audit of two production training
codebases — 1,774 Slurm jobs, 205 export artefacts, ten target training stages, MoE and
dense model families across H100 and GB200 — rather than from a blank page. The audit is
published here in full, because the architecture is only defensible if the evidence
behind it is legible.

The finding that shapes the design:

> **The dominant failure mode in large-scale training is not a crash. It is a run that
> reports success.**

A checkpoint whose expert tensors are aliased still saves. A policy-gradient run whose
importance ratio is identically 1.0 still logs a healthy reward curve. 472 steps with
`grad_norm` exactly 0.000 still reported `reward/mean=0.794, success=1.00`. Every one of
these passed every check that existed, because the checks asked *"did anything go
wrong?"* — a question a broken system answers "no" just as fluently as a working one.

So FoundationScale's central abstraction is not a trainer or a parallelism strategy.
It is a **gate whose verdict includes its coverage.**

```python
class Verdict(str, Enum):
    PASS          # examined N things, all correct
    FAIL          # examined N things, found a defect
    VACUOUS       # examined ZERO things, and therefore proves nothing
    UNDERCOVERED  # examined fewer than expected, without declaring a sample
    SKIP
    ERROR         # the gate itself broke — fail closed
```

`VACUOUS` and `UNDERCOVERED` are blocking verdicts, and a gate author **cannot** return
`PASS` while reporting zero examined units — the coverage rule lives in the base class,
not in the author's discipline.

This is not a hypothetical. The audit's own verification tool shipped exactly this bug:
it reported `expert_perm_all_identity: true` on a known-corrupt checkpoint, because the
expert tensors were *absent*, the comparison set was empty, and `all([])` is `True` in
Python. It was caught in minutes by a negative control — after the same class of error
had gone undetected for months, four separate times, in the systems it was auditing. The
engineer who wrote the detector for that bug wrote the bug into the detector. That is the
argument for putting the rule in the type system instead of in a code-review checklist.

## The analysis

Seven deliverables and a decision log, written as an evidence chain rather than a
report. Start with
[`docs/DECISIONS.md`](docs/DECISIONS.md) — the reasoning spine — or jump to the
[deliverables index](docs/deliverables/README.md).

| | Document | What it answers |
|---|---|---|
| **A1** | [Existing system — structure](docs/deliverables/A1_existing_system.md) | Two repos, workflows end-to-end (SFT / RL / export), infrastructure, dependency map |
| **A2** | [Existing system — algorithms](docs/deliverables/A2_algorithms.md) | Every objective (SDPO/POLAR/GSPO/DPO/ODPO), the reward cascade, pipeline coverage matrix |
| **A3** | [Existing system — debt & risk](docs/deliverables/A3_debt.md) | Duplication census, the **silent-failure catalogue**, blast-radius accounting, risk register |
| **B1** | [Architecture — core](docs/deliverables/B1_architecture.md) | The L0–L6 layered design, gate catalogue, run manifest, falsifiable claims |
| **B2** | [Architecture — unified LLM/VLM + scaling](docs/deliverables/B2_scaling.md) | One data contract, one Stage abstraction, 1 GPU → 100+ nodes as a ladder with measured rungs |
| **C** | [Codebase mapping](docs/deliverables/C_mapping.md) | Per-component Reusable / Refactor / Wrap / Replace / Build-New |
| **D** | [Development roadmap](docs/deliverables/D_roadmap.md) | Phases 1–10, each with exit criteria and a **falsification condition** |

Every claim carries an evidence tag — **[M]** measured, **[V]** verified by source
inspection, **[A]** artefact census, **[K]** asserted by a model and not independently
confirmed, **[U]** explicitly unverified. The documents apply two rules to themselves:
an unqualified count is not a fact, and *every claim that something does not exist must
name the positive control proving its detector could have fired.* That second rule exists
because the audit broke it four times and was wrong each time.

The documents are published with cluster-internal identifiers replaced by stable
pseudonyms. Nothing else is altered — the failure record, the numbers and the retractions
are as written.

## What the diagram shows

**Model stack** — curated recipes and framework libraries covering the full model
development path, from pre-training and SFT/LoRA (VFM, LLM &amp; VLM) through the
framework layer (Megatron-LM, Megatron-Bridge, AutoModel) to post-training
alignment and RL.

**Parallelism strategies** — the four ways work is distributed across GPUs:

| | strategy | what is split |
|---|---|---|
| **DP** | Data Parallelism | Same model replicated on each GPU, different data per GPU |
| **TP** | Tensor Parallelism | Each GPU holds a slice of every layer (tensor / hidden dimension split) |
| **PP** | Pipeline Parallelism | Different layers (stages) on different GPUs, executed sequentially |
| **EP** | Expert Parallelism | Different experts on different GPUs, a router selects which experts run |

**Distributed infrastructure** — the same stack runs on local machines, SLURM
clusters, DGX Cloud / Lepton, and Kubernetes clusters.

## Status

Early. The gate contract (`src/foundationscale/gates/core.py`) is frozen; the layers
above it are being built against the roadmap in
[`docs/deliverables/D_roadmap.md`](docs/deliverables/D_roadmap.md). The analysis is
complete and is the more useful artefact today.

## Assets

| file | description |
|---|---|
| [`assets/hero.png`](assets/hero.png) | 2000 &times; 2242 — the image used above |
| [`assets/hero@2x.png`](assets/hero%402x.png) | 4000 &times; 4484 — retina / print master |
| [`assets/hero.html`](assets/hero.html) | Self-contained source (system fonts, inline SVG, no external requests) |

The poster is rendered from `hero.html`, which has no network dependencies — open
it in any browser, or re-render it headlessly at any resolution:

```bash
chrome --headless=new --disable-gpu --hide-scrollbars \
       --window-size=2000,2242 --force-device-scale-factor=2 \
       --screenshot=hero@2x.png file://$PWD/assets/hero.html
```
