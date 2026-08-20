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

## Assets

| file | description |
|---|---|
| [`assets/hero.png`](assets/hero.png) | 2000 &times; 1560 — the image used above |
| [`assets/hero@2x.png`](assets/hero%402x.png) | 4000 &times; 3120 — retina / print master |
| [`assets/hero.html`](assets/hero.html) | Self-contained source (system fonts, inline SVG, no external requests) |

The poster is rendered from `hero.html`, which has no network dependencies — open
it in any browser, or re-render it headlessly at any resolution:

```bash
chrome --headless=new --disable-gpu --hide-scrollbars \
       --window-size=2000,1560 --force-device-scale-factor=2 \
       --screenshot=hero@2x.png file://$PWD/assets/hero.html
```
