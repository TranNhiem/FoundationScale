# A.1 — Existing System Analysis (Part 1 of 3)

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


> **Post-draft corrections.** Four investigations returned after this document's
> evidence prompt was frozen. The tech lead applied their results in place; the
> corrected claims are marked **[V·post]** and each names its evidence. Corrections
> applied here: the `reward_m1m4` census (24/6 → **59/8**), run provenance
> (**not** unrecoverable — 35 runs carry `repro/` bundles; the gap is environment
> variables specifically), the co-located NCCL hang (a control run **refutes** the
> explanation recorded in the launcher), the 8-node ceiling (a **choice**, not a
> hardware limit), and `exports/fullft_iter2400_1tray_hf` (**empty**, so never served). **The fourth landed last and is the largest:** a Megatron↔HF weight comparison closed both items that were said to gate Phase 1 — the expert fix is numerically correct (3,840/3,840 experts bitwise identical) and **200 of 205** export dirs are weight-verified at **0 DIFFER** (the denominator 23 was also wrong). It ran on a login node with no GPU. It also produced S19: the probe passed a corrupt artefact because `all([])` is `True` — the audit's own thesis, live in the audit's own tool.


**Scope.** This document analyses the two Omni codebases as they exist on the
cluster as of 2026-08-20, and draws the operational conclusions FoundationScale's
architecture must be designed against. It is a description of what *is*, including
what only *appears* to be. Where an earlier analysis pass made claims that direct
source inspection has since refuted, the corrected statement is used and the
retraction is flagged inline, because three of those refutations change the
architecture conclusions materially.

Confidence markers: **[M]** deterministic measurement · **[V]** direct source
inspection · **[A]** exhaustive artefact census (2026-08-20) · **[U]** unverified.

---

## 1. The two codebases

Two sibling directory trees hold all first-party training code:

- `omni-accel` — 3,010 files / 681,440 LOC **[M]**. The more active tree.
  Holds all SFT launchers for the Gemma-4 line (`g4_sft/`), the pretraining-adjacent
  assets, the vendored HF architecture sources (`hf_src/`), the `rl-accel/` async-RL
  design work, and the `exports/` tree. **It is not a git repository at all** [M] —
  there is no `.git` directory. The more active half of the program is entirely
  unversioned.
- `omni-bridge` — a git repo at HEAD `5744a25`, last
  commit 2026-07-11 ("Skip videoqa in during-training eval"), **7 commits in 90 days
  and 892 dirty files** [M]. It holds the currently-live RL drivers
  (`sdpo_gemma4/run_gspo.py`, `polar/`, `sdpo_odpo/`), the judge registry stack, and
  the off-Slurm production scripts.

First-party source across both, excluding vendored trees, benches and checkpoint
dirs: **2,286 files / 361,339 LOC** (1,344 `.py` / 241,803 LOC) [M]. Including the
vendored Megatron-Bridge trees the whole tree is 9,905 files / 1,488,286 LOC.

### 1.1 Duplication: two censuses, and the number that actually matters

Because the two censuses scanned different file sets, both must be quoted with their
set named **[M]**:

| Census | File set | Redundant LOC | Share | Clusters |
|---|---|---:|---:|---:|
| Strict (`fs_clones.py`; excludes vendored Bridge, benches, checkpoints, lmms-eval) | 1,319 `.py` / 227,357 LOC | **137,568** | 60.5% | 98 |
| Loose (all first-party) | 241,803 LOC Python | **149,647** | 61.9% | 112 |

Deduplication by normalised AST hash collapses 1,296 on-disk Python files to **326
unique** [M]. The worst families by wasted LOC: `omni_sdpo_reward.py`
19×565 = 10,170; `omni_sdpo_dataset.py` 19×414 = 7,452; `dpo_dataset.py`
23×250 = 5,500 (25 copies across sets); `run_dpo.py` 21×261 = 5,220 [M].
`accel/sdpo_r` and `bridge/sdpo_r` are Jaccard 1.000 over 46 files [M].

Raw duplication is not the load-bearing fact. The load-bearing fact is **diverged
clone families**: 11 filenames exist in more than one AST-normalised variant — same
name, materially different code [M]. `rule_checks.py` has 5 variants over 24 copies
(379/318/387/397/265 LOC); `sdpo_loss.py` 4 variants over 21 copies; `run_sdpo.py`
3 variants over 8 copies on the strict AST set (and **18 distinct md5s over 23
copies** counting vendored trees byte-wise — same question, different answer, which
is why no fork count may be quoted without its file set). At least some of the
divergence is capability, not style: the 171-LOC variant of `modeling_gemma4_vl.py`
has **no `freeze()` method** while the 217-LOC variant does **[V]** — a build that
resolves the wrong copy trains every parameter and reports success.

**Operational cost.** The copying is deliberate — `PRETRAINING_PIPELINE.md` defends
it as clean A/B isolation: a self-contained fork per algorithm, toggled by a
default-off flag. The measured cost **[V]**:

- The DP>1 reward-broadcast fix (`src=tp_src, group=tp_group`, dated 2026-08-12 in
  `omni-bridge/sdpo_gemma4/run_gspo.py:815,916`) is
  present in 14 driver trees and absent from 13, which still carry the global
  `src=0` form. A correctness fix had to be landed ~27 times and demonstrably was
  not — a ~48% propagation miss rate.
- Reward semantics differ across four coexisting near-identical clone families of
  `reward_m1m4` (§5). Because modules are imported from the CWD by bare name
  (`from omni_sdpo_reward import build_reward_fn`, `run_gspo.py:40`), **the
  objective RL optimises is selected by which of ~20 sibling directories you `cd`
  into**, with no version marker. Objective provenance is recoverable only from
  judge audit logs, not from code **[V]**.

### 1.2 Which repo is live, and the asymmetry that matters

Both are live, for different things. The live RL production path is in the bridge
repo (the DP>1-safe GSPO driver and the 31 `offslurm_*` scripts, §5); the SFT /
export / Gemma-4 bring-up path is in accel. Two asymmetries must be stated plainly:

1. **Every pretraining-adjacent asset lives only in `omni-accel`** **[V]**:
   the CPT launcher (`launch_omni_cpt_omni.sh`), `PRETRAINING_PIPELINE.md`, and
   the random-init smoke driver (`train_resume_test_e4b.py`). The bridge repo has
   none of them. (This corrects the retracted claim "absent from both repos" — the
   full from-scratch machinery is *vendored* in both, unexercised by first-party
   code in either.)
2. **Names are not identity.** `sdpo_gemma4` exists in both repos with different
   contents: `accel/sdpo_gemma4` is gspo-family reward with the **unfixed** reward
   broadcast (`run_sdpo.py:635`); `omni-bridge/
   sdpo_gemma4` is grader-primary reward with the **fixed** broadcast **[V]**. Every
   reference in this document is repo-qualified for that reason. Likewise "FACTS"
   names two unrelated things: a reward variant (`sdpo_facts`) and the POLAR_A
   anchored co-teaching *loss*.

One further hazard, because it has already burned this analysis twice **[A]**:
superseded design docs are retired by prepending a banner, not by deletion.
`exports/EXPORT_STATUS.md` (2026-08-03) and `g4_sft/EXPERT_SAVE_BUG.md` both
describe long-fixed states; `EXPERT_SAVE_BUG.md` is banner-marked SUPERSEDED at
line 1 (its "the fix was reverted" lines describe six failed *intermediate probes*).
They are cited here only as history. The live truth is in
`g4_sft/EXPERT_ALIASING_VERDICT.md` and the artefact census (§4.2, §5.3).

---

## 2. Current training workflows, end to end

### 2.1 SFT (the best-documented, production path)

A developer running Gemma-4 SFT today **[V]**:

1. **Prepare the corpus.** Format raw sources to ShareGPT-style jsonl
   (`g4_sft/format_taiwan_aiec.py`), validate read-only with
   `g4_sft/validate_v3_corpus.py` (13 ERROR-class invariants on the Gemma-4
   channel-marker format). The v3 corpus is 8 files / 20,769 rows.
2. **Drive a wrapper.** `g4_sft/run_26b_fullft_v3.sh` (or `run_31b_fullft_v3.sh`)
   exports `OMNI_SFT_JSONLS`, computes `ITERS=ceil(rows*epochs/GBS)` in bash
   (2,467 for the v3 schedule), and submits `g4_sft/launch_g4moe26b_sft_32k.sh` /
   `launch_g4dense31b_sft.sh`.
3. **The launcher** validates geometry by hand (TP must divide
   `num_global_key_value_heads=4` on the 31B; EP/ETP==1 on dense; `RECOMPUTE=full`
   refused), detects fresh-vs-resume from `latest_checkpointed_iteration.txt`,
   toggles `load_optim`/`load_rng`, grep-preflights that the vendored
   Megatron-Bridge still carries the Omni patches (§4), hard-refuses a MoE base
   checkpoint not ending in `-expertfix` (`launch_g4moe26b_sft_32k.sh:164-167`), and
   runs `srun --container-image=<sqsh> bash -lc 'python3
   scripts/training/run_recipe.py --recipe gemma4_vl_{26b,31b}_omni_sft_config
   --step_func vlm_step'` with ~30 dotted CLI overrides.
4. **Batch construction** happens inside vendored Megatron-Bridge
   (`.../data/vlm_datasets/collate.py`) with two live, default-on, Gemma-4-only
   patches: the thinking-preserving chat-template patch (`collate.py:1141`, gate
   `OMNI_GEMMA4_KEEP_COT` — without it the stock Jinja template strips CoT and
   *every sample has zero supervised tokens* under a healthy-looking loss curve) and
   the stop-token fix (`collate.py:1260-1300`, extends supervision over `<turn|>`,
   id 106). The loss mask is built by *string search over rendered text*
   (`create_multiturn_loss_mask_by_search`). `g4_sft/check_loss_mask.py`, a
   tensor-level mask checker, **was written and never run** **[V]**.
5. **Health is verified by acceptance greps** on logs ("thinking-preserving chat
   template active" > 0; "ZERO supervised tokens" == 0), not by any structured
   launch gate. There is no automated trainer-vs-server template/tokenization
   parity check — only a manual md5 fingerprint of `chat_template.jinja`
   (`b1d40a45...`) compared by hand **[V]**.

Only **3 real recipe names** exist in the entire tree (`gemma4_vl_26b` /
`gemma4_vl_31b` / `nemotron_omni_..._peft` configs) [M]; everything else is
launcher-level variation over them.

### 2.2 RL (two paths: Slurm torchrun, and the off-Slurm production path)

The canonical on-policy RL (GSPO, in the bridge repo) **[V]**:

1. Stand up sidecar fleets *by hand, outside Slurm*: vLLM rollout servers and judge
   pools on borrowed "tray" nodes via ssh + tmux + raw enroot
   (`ODPO_TRAY_STANDUP_RUNBOOK.md`, `start_judge_pool.sh`, `vllm_serve_sidecar.sh`).
   Trays are node-holds from another project's sbatch jobs with hard 48-hour limits.
2. Submit a launcher, e.g. `sdpo_gemma4/gspo_v2/launch_gspo_g4dense31b.sh`:
   preflights every `VLLM_URLS` endpoint with `curl /v1/models` **plus a served-name
   match** (an endpoint can be up but serving the wrong model — that produced silent
   reward 0 in production), validates the reward name against a Python registry
   before paying the ~50 GB model load, screens out `-it` instruct checkpoints on
   both HF and Megatron paths, implements tri-state `RESUME=auto|never|always`, then
   `srun --ntasks-per-node=1 torchrun --rdzv_backend=c10d ... run_gspo.py`.
3. Per step, `run_gspo.py` draws a modality-homogeneous prompt batch, renders the
   chat template, POSTs to the rollout fleet (only the TP-group leader POSTs;
   results broadcast within the TP group), computes reward **on the TP-source rank
   and broadcasts with `src=tp_src, group=tp_group`** (the 2026-08-12 fix),
   captures `pi_old` from its own forward at inner epoch 0, and steps.

**The off-Slurm path is currently the production path** **[V]** — this corrects the
retracted "no non-Slurm path" claim. 31 `offslurm_*` scripts in
`omni-bridge/sdpo_gemma4/` run training via ssh +
`enroot start --rw` + bare torchrun with no `#SBATCH`, no `sbatch`, no `srun`
(`offslurm_run.sh`, `offslurm_multinode.sh`, `offslurm_gspo_official1..10_cmd.sh`);
`TRAYS_IN_USE.txt` documents four trays training this way, and 6-tray GSPO at DP=12
(`GSPO_STATE.md:26`) is the largest live configuration — the driver is a 300-second
ssh watchdog (`gspo_watchdog.sh`). The proximate cause is operational: a
cluster-wide broken IMEX prolog makes every multi-node Slurm submit drain its tray
(`offslurm_run.sh` header). But it is a **bolt-on, not an abstraction**: the
off-Slurm command is *string-grepped out of `DRYRUN=1` output* of the Slurm
launcher, and exactly **1 of ~240 launchers** supports `DRYRUN` **[V]**.

**Two caveats that must travel with any description of the RL path.** First, the
reward object itself is fork-dependent (§1.1, §5.1). Second —
the single most consequential audit finding **[V]**: `run_gspo.py
--old_logp_source` defaults to `self`, meaning `pi_old = pi_theta.detach()` from
the *current* forward, so the GSPO importance ratio is identically 1.0 and the clip
band can never bind. **7 of the 10 official GSPO arms ran with no trust region at
all** — structurally healthy jobs executing an algorithm other than the one named.
One arm (`gspo_official4`) additionally *changed gold policy mid-run* (steps 0–890
under one reward semantics, resumed at step 800 under another, in one W&B run and
one checkpoint dir) — recoverable only because every launcher passes
`--judge_audit_path` and the audit `decided_by` vocabulary fingerprints the policy
**[V]**.

### 2.3 Export (Megatron → HF)

The historical claim "export is broken end to end; only LoRA works" is **retracted**
— it reproduced the superseded `EXPORT_STATUS.md` (2026-08-03). The adjudicated
census of all 23 export directories **[A]** (a later, wider sweep finds **205** index-bearing export dirs — the 23 was the `results/` subset **[V·post]**): 19 PASS byte-vs-index (MoE-26B:
51,612,009,852 actual vs 51,611,872,412 declared bytes, 1013 tensors incl. all 60
expert tensors; dense-31B: 62,546,338,184 B / 1188 tensors); 3 FAIL, all pre-fix
and already renamed `BAD_INCOMPLETE*`; 1 with no index at all
(`exports/fullft_iter2400_1tray_hf` — **resolved [V·post]: it is empty — 0 files, 0 bytes**, created by job J07, which FAILED 2m10s later with `ncclInvalidUsage — Duplicate GPU detected: rank 3 and rank 7 both on CUDA device` (the "1-tray" trick packs an 8-rank EP=8 world onto a 4-GPU tray). It died inside `read_run_config` → `broadcast_object_list`, **before a single weight was read**, so `save_hf_pretrained()` was never reached. *Method note, which matters more than the answer:* the investigator first checked whether the detector could fire, and it could not — known-good exports also score 0–1 log references and one positive control failed outright. The verdict therefore rests on the dispositive physical fact (zero bytes), not on absence of mentions.)

The working path is deliberately boring and is the right design seed: the shipped
exporter runs `WORLD_SIZE=1`, EP=1, on a single GPU (~4 min for 51.6 GB), letting
the DCP checkpoint format — not the exporter — carry the parallelism. That became
possible only when the root-cause save bug was fixed (§4.2): the elaborate EP=8
multi-rank exporter (`export_moe_ckpt.py`, `export_moe_1tray.py`) was workaround
machinery for a one-word bug in a class declaration, and was dropped with it.

**Mandatory caveat (must appear wherever export health is claimed):**
byte-completeness is not weight correctness. `BAD_INCOMPLETE_..._1723` carries a
*perfect-looking* 1013-tensor / 60-expert-tensor `index.json` over 5.94 GB of the
required 51.6 GB; and no byte sum can detect a permuted expert axis at all.

**Both halves of that caveat have now been discharged at the weight level, and the
second half is why it matters [V·post].** A direct Megatron↔HF tensor comparison —
run on a login node with no GPU, by byte-range-reading individual DCP chunks via the
`.metadata` offset table rather than loading checkpoints — covers **200 of 205 export
directories**: Gemma4 same-step 3,861 EXACT / **0 DIFFER** / 0 SHAPE_MISMATCH (24
MISSING, all LoRA-only sources), Nemotron **1,200 / 1,200 EXACT**, 91 cross-step
refits at expert-permutation identity, and 23 LoRA merges reconstructing
`merged == round_bf16(base + 2·B·A)` at **max error exactly 0.0 across 480/480
modules**. The permuted-expert question specifically: full N×N cosine crossmatch on
`g4moe26b_twaiec_BASE_fullft_v3/hf_iter_0002467` puts **3,840 / 3,840 experts across
all 30 layers** at `argmax == identity`, and the detector was proven live by injected
controls that caught a random shuffle 128/128 and the bug's own `i → i%16` alias map
112/112. A header-only integrity sweep of all 205 gives 202 OK / 3 BROKEN, all three
already quarantined by filename. The coverage figure carries its own positive control:
the batch runner was interrupted, so the 205 denominator was re-enumerated independently
of its exit status and reconciled per family — a coverage number sourced from the job
that produced the coverage would not be evidence of coverage.

**What is still true, and is now the only gap: no export has ever been asked to
produce a token [U].** Bitwise-identical weights do not survive a wrong `rope_theta`,
a mismatched attention implementation, a tokenizer drift or a bad generation config —
every one of which passes a tensor comparison perfectly. Correctness is established
at the level of bytes and expert assignment; it is not established at the level of
behaviour.
The live byte-gate exists inline in `export_v2_fullft_sbatch.sh:58-70` ("NEVER
trust rc=0") but is copy-pasted heredoc, and `export_many.sh` — which passes the
dangerous `--not-strict` — verifies only with `du -sh` [V].

---

## 3. Infrastructure

- **Cluster:** one Bright-Computing Slurm cluster (`<login-node>`), partition `<partition>`
  (inconsistently spelled `<partition_alt>` in 4 launchers) [M], GB200 trays at 4 GPUs/node,
  node names `<compute-nodes>`, `<compute-node>*` hard-coded throughout — in launchers, in
  judge-registry YAMLs, in Python module bodies (rollout endpoints are *module-level
  constants* in the training driver), and in design docs. Absolute paths under
  `<CLUSTER_HOME>/` everywhere, including inside containers (the container
  mount is literally the same home directory).
- **Containers:** two pinned `.sqsh` images (training:
  `nemo-automodel-26-04_compute.sqsh`; serving: a vLLM image whose filename contains
  an en-dash), launched three different ways: pyxis `srun --container-image` for
  recipe training, bare `torchrun` on host python for GSPO, and raw `enroot
  start --rw` for everything off-Slurm. Raw-enroot sidecars require hand-built rc
  scripts (mktemp heredoc fixing `/etc/passwd`, pre-created cache dirs) and —
  because non-pyxis enroot does not isolate GPUs — **absolute physical GPU indices**
  in `CUDA_VISIBLE_DEVICES` matching `NVIDIA_VISIBLE_DEVICES`. Orphaned vLLM
  `EngineCore`/`Worker_TP*` processes survive tmux kills and must be reaped by
  `nvidia-smi --query-compute-apps=pid`; if not, they keep serving **stale weights**
  on the reclaimed GPU **[V]**.
- **The tray model.** RL jobs are multi-job constellations: a Slurm training job
  plus rollout/judge fleets on *borrowed* nodes held by foreign 48-hour hold-jobs,
  stood up by hand in tmux. Because the services are invisible to the scheduler,
  `sinfo` reports the tray idle and other users collide with them; the workaround
  is pinning the training job's `--nodelist`. Sidecar lifecycle (start, health,
  reload, reap) is a runbook, not a system **[V]**.
- **Rollout/judge fleets.** vLLM over OpenAI-compatible HTTP (`/v1/completions` for
  text, `/v1/chat/completions` for multimodal with base64 images / `file://`
  videos). Judge routing is genuinely the most architecturally mature piece in the
  tree: `judge_registry.yaml` maps task family → pool → endpoints with glob
  fallback and heterogeneous multi-model pools — but plural endpoints are
  round-robin *load balancing* (`judge_pool.py:44-58`), nothing is
  content-matched, and the registry schema has zero validation (a wrong key or
  stale endpoint is discovered mid-run). vLLM appears in exactly **1** first-party
  Python file [M]; everything else reaches it over HTTP.
- **Secrets.** `<secrets-env-file>` (chmod 600) sourced on the worker for
  `WANDB_API_KEY` / `KIMI_K3_API_KEY`; behaviour varies per launcher (hard-fail,
  offline fallback, preserve-preset) — three policies in flight. A missing secrets
  file **silently disables W&B**; the run's only metrics record is conditional on
  an uncommitted dotfile **[V]**. (The sane consequence: environment is for secrets
  and paths only. The insane current state: 166 behaviour-bearing env vars, §4.3.)
- **Time limits.** The "7-day job max" in launcher headers is stale folklore:
  launchers request `--time=14-00:00:00` and even `UNLIMITED`; g4_sft records
  partition MaxTime=UNLIMITED with 10-day user directives; tray holds cap at 48h
  regardless. Long runs chain across limits via `sbatch
  --dependency=afterany:<prev>` resume-chains whose surplus links exit 0 once the
  run is complete (`g4_sft/revised_corpus_run.sh`) **[V]**.

---

## 4. Framework dependency map

### 4.1 What the stack actually stands on

First-party import counts [M]:

| Dependency | First-party importers |
|---|---:|
| PyTorch | 393 |
| HF transformers | 161 |
| torch.distributed | 113 |
| Megatron-Bridge | 97 |
| Pillow | 67 |
| Megatron(-Core/-LM) | 62 |
| W&B | 40 |
| vLLM (library) | **1** |

The zeros are as load-bearing as the counts: **no** NeMo, NeMo-RL, DeepSpeed, FSDP,
TransformerEngine, Triton, TensorRT, Hydra, OmegaConf, or Ray imports anywhere
first-party [M]. There is one Accelerate import and one PEFT import. Everything
trainable is Megatron: 748 of 2,888 non-vendored `.py` import it; there are zero
TRL, zero HF `Trainer`, zero standalone-PEFT training entrypoints **[V]**. Even the
single-GPU path (§5.2) is Megatron degenerated to TP=1/EP=1/DP=1. The project's own
hero diagram advertising "NeMo Framework Libraries & Collections" contradicts the
code and should be corrected.

### 4.2 Megatron-Bridge is forked, not depended on

Three vendored Bridge trees coexist — `Megatron-Bridge/`,
`_vendor_bridge_expertfix`, `_vendor_bridge_g4fix` — selected by `PYTHONPATH`
ordering and guarded by **grep preflights**: launchers grep the vendored sources
for literal patch strings (`OMNI_MAX_VIDEO_FRAMES` hooks, the modality bucket
sampler, the video-embedder backstop) and abort if absent **[V]**. This detects
deletion but not mutation — a patch edited to a no-op still passes a text match.

Which tree is live is settled **[V]**: training launchers put
`$ACCEL/Megatron-Bridge` on `PYTHONPATH`, and that tree carries the merged expert
fix at `.../models/gemma4_vl/gemma4_provider.py:445,457`
(`class Gemma4DenseMoE(MegatronModule)` + `super().__init__(config)`, applied
2026-08-05 13:23; pre-fix backups remain on disk). `_vendor_bridge_expertfix` is
used only by `convert_g4_expertfix_sbatch.sh`.

That one-line base-class fix is the highest-stakes line in the codebase, and its
history is the sharpest argument in this document, so it is stated precisely and
with the retraction explicit **[V][A]**: the earlier claim "the expert fix was
reverted and never landed" is **false** — it came from reading the SUPERSEDED
`EXPERT_SAVE_BUG.md`. The bug was real and worse than reported (it was a *load-side*
aliasing bug too: at EP=8 every rank loaded global experts 0–15, so pre-fix runs
trained 16 distinct experts replicated 8×). The fix landed 2026-08-05, production
checkpoints have carried the correct signature (45.70 GB expert bytes, 0 indexed
keys, `(128,1408,2816)` chunks=256) since, and the launcher hard-refuses non-
`-expertfix` bases. But the census **enlarged the blast radius beyond any bug
report**: 9 of 20 MoE runs trained from the aliased base `gemma-4-26b-it` — 2
full-FT *and 7 LoRA runs*, whose adapters are structurally clean (0 expert bytes)
and therefore **pass every check while sitting on a frozen base that was 87.5%
wrong** — including the entire published muon-vs-AdamW LoRA comparison. That
comparison did not merely become invalid; it **inverted** on the fixed base (AdamW
0.7412 beats Muon 0.7828; jobs J09/J10, matched, already complete) **[V]**. The
fix is *structurally* verified; numerical correctness (right experts vs
rightly-shaped wrong ones) is **[U]**.

### 4.3 The configuration surface

There is no unified configuration system. A run's effective config is the union of:
vendored HF `@strict` config dataclasses; recipe functions plus untyped `OMNI_*`
env overlays; launcher `${VAR:-default}` env layers; argparse flags on drivers;
hand-edited judge-registry YAMLs (~20 near-identical copies, differing mostly in
hardcoded node hostnames, some carrying *incompatible pool schemas*); sourced
sidecar settings files; and filename-suffix conventions selecting data-lineage
stage. Measured surface [M]: **166 distinct environment variables** (37
`OMNI_*`, 26 `SDPO_*`, 16 `GSPO_*`, 16 `VLLM_*`, 15 `JUDGE_*`, ...), 205–247
launcher scripts carrying the actual run configuration, and 3 real recipes.

Two reward-semantics flags are read at **module import time**
(`OMNI_GOLD_MISS_IS_BAD`, `OMNI_GOLD_GRADER`), so setting them after import
silently no-ops — and a dead duplicate `_GOLD_MISS_IS_BAD` at
`mb/sdpo_gemma4:787` shows the copy-paste-patch workflow already generating
unreachable code **[V]**. Worse: `OMNI_GOLD_MISS_IS_BAD` was active for ≥12
ODPO runs while **nothing in the repo sets it** — it was exported by hand in an
interactive shell and propagated by `sbatch --export=ALL`, leaving zero on-disk
trace **[V]**. Reproducibility is attempted via manual tree copies into
`code_snapshots/`, which freeze stale node endpoints and (being machine
re-serialized) strip exactly the comment-level incident knowledge that made the
live files valuable.

---

## 5. Operational reality

### 5.1 How failures are actually noticed

Detection is the strongest layer; recovery is the weakest. Detection, all of it
standing between a typo and a 50 GB model load **[V]**: pre-flight `curl` +
served-name identity on every endpoint, reward-registry import checks, grep patch
preflights, instruct-model screening, jsonl non-empty checks, geometry asserts,
canary-fingerprint greps on 2-iteration probe jobs ("parameters per rank"), and
`DIST_TIMEOUT_MIN=30` on GSPO to bound losses from a dead rollout server. On top of
that sit hand-built supervisors: cron/tmux autopilots (`sdpo_autopilot*.sh`,
`autopilot_grid_32k.sh`) that classify crashes (novel-fatal device-side asserts →
halt-and-alert; known-benign → capped resubmit), detect rollout outages by
*rate-of-empty-completions*, gate checkpoint promotion through staging-probe
batteries and long-generation collapse gates, and chain resubmits via `afterany`.

Recovery has **no in-job component whatsoever**: no elastic torchrun, no
`--requeue`, no signal-triggered checkpointing in any launcher. Recovery is
resubmit-into-a-stable-`OUT_DIR` plus `latest_checkpointed_iteration.txt`
auto-resume, with a known, unfixed resume livelock in `ModalityBucketBatchSampler`
(fast-forwards a full epoch instead of the consumed prefix, spinning at 0% GPU —
`g4_sft/RELAUNCH.md`) sitting on the default auto-resume path of the 8-tray
launchers.

And the deepest operational fact: **the failures that mattered were invisible to
every gate above.** An 87.5%-wrong MoE checkpoint passed rc=0, resume, loss curves,
tensor counts and dtypes for two full runs; only a hand-computed byte sum caught it
**[A]**. 472 steps of `grad_norm` exactly 0.000 with `reward/mean=0.794,
success=1.00` logged straight through — the pre-fix DP=12 driver was, for a quarter
of its steps, not training at all, and said it was fine **[V]**. A stock chat
template deleted CoT from targets yielding zero supervised tokens under a healthy
loss. 23.42% of rows silently losing vision supervision. Every one was caught late,
by a human, reading bytes by hand.

### 5.2 The real scale distribution (including what the sbatch histogram misses)

Directive-based counting gives [M]: nodes `{"1": 110, "2": 71, "8": 24, "4": 19,
"6": 15}`, GPUs/node `{"4": 166, "1": 14}`. **Do not quote this as a topology
census** — it misses the single-GPU path (configured by an internal `TRAIN_GPUS`
variable that contradicts the directives) and all 31 off-Slurm scripts.

The corrected picture, verified against logs **[V]**:

- **Single-GPU training exists and has run.** Six launchers default
  `TRAIN_GPUS=${TRAIN_GPUS:-1}` and emit `torchrun --nproc_per_node=1
  --master_addr=127.0.0.1`
  (`accel/sdpo_e4b/launch_sdpo_e4b_arm.sh:96`, `launch_sft_e4b_smoke.sh:53`,
  `mb/sdpo_gemma4/launch_omni_sdpo_gemma4.sh:119`, et al.). Fourteen job logs
  (`logs/e4b_arm_1458..1473.out`) print `train GPUs=1(DP=1) TP=1 EP=1`; two reach
  step 300. This is Megatron degenerated to one rank, and DP1 is the *default*,
  not a fallback — the cited reason is unresolved co-located multi-GPU NCCL hangs
  **[U: no root cause named — but the launcher's own explanation is now refuted
  and the cause is bounded to three confounded factors; see `DECISIONS.md` §5]**
  **[V·post]**. A genuinely valuable single-tray colocated RL recipe also exists:
  `mb/sdpo_gemma4/smoke_refit_e4b.sh:94` (GPU0 train / GPU1 rollout / GPU2 export),
  re-verified on disk during review — 7,586 bytes, line 94 is the cited `torchrun`
  **[V·post]**.

  > **A false negative caught during review, recorded because it is the third of
  > four of its kind** — the fourth being S19, where the export-verification probe
  > passed a corrupt artefact because `all([])` is `True`, caught by a negative
  > control rather than by review. The NCCL investigation reported that `smoke_refit_e4b.sh` "does not
  > exist anywhere under `<CLUSTER_HOME>`" and that only two launchers co-locate vLLM
  > with training. Both claims are wrong: the file is exactly where this document
  > cites it, and it is a *third* co-location launcher — in the other repo, which
  > is why a search scoped to `omni-accel` missed it. The negative came
  > from a `find` that stalled on the 23 TB tree and was abandoned, then read as
  > absence. The tech lead nearly propagated the "correction" into this document
  > and caught it only by applying the rule stated in the README: *an absence claim
  > must name the positive control that proves its detector could have fired.*
  > The rule earns its place. One part of that report does survive and is worth
  > keeping — `smoke_refit_e4b.sh:110`'s `srun --export=` allowlist passes
  > `NCCL_SOCKET_IFNAME`, `NCCL_IB_HCA` and `NCCL_NVLS_ENABLE` but **not**
  > `NCCL_DEBUG`, which corroborates why no NCCL diagnostics ever reached a log
  > **[V·post]**.
- **The off-Slurm fleet (§2.2) is the production path** and does not appear in any
  sbatch histogram.
- **The ceiling is now an execution fact, and it is a *choice* [V·post].** Slurm accounting over all **1,774** Omni jobs gives a node histogram `{1:1242, 2:299, 3:2, 4:134, 5:2, 6:2, 8:93}` topping out at **8 nodes / 32 GPUs**; max degrees are TP=8, EP=32, ETP=1, **PP=1 — pipeline parallelism was never >1 in any log** — and CP peaked at 2. The off-Slurm path tops out *lower* (6 nodes / 24 GPUs), so it does not raise the ceiling, and 213 jobs allocated exactly 1 GPU, confirming the single-GPU floor. The decisive nuance for planning: **other users on this same cluster have run 18 nodes / 72 GPUs**, against a capacity of 23 nodes / 92 GPUs. **8 nodes is ~35% of what was available — the ceiling belongs to this codebase, not the hardware.** Also: W&B is a dead channel for topology (35,001 local run dirs are login-node backfill; 238 sampled configs contain zero `world_size`/TP/PP/EP keys).
- **The ceiling is 8 nodes / 32 GPUs / EP=32** (the tp8ep32 omni full-FT launchers)
  and 6 nodes / 24 GPUs / DP=12 for the live GSPO run [M][V]. The only >8-node
  scripts in the tree are vendored NVIDIA examples for GLM-4.5V and Qwen3.5-VL —
  64 and 16 nodes — with **no Omni logs or results behind them** [V]. The
  stated 100+-node target is unvalidated by three orders of magnitude, and those
  vendored launchers are false evidence of scale waiting to mislead a reviewer;
  they should be quarantined or labelled.
- **The two lower-end stories do not compose** [V]: the random-init path
  (`train_resume_test_e4b.py`) ran only at 4 GPUs on mock data (12 steps, val PPL
  4.3e5 = random), and the single-GPU path has only run from a pretrained
  checkpoint. There is no proven 1-GPU from-scratch path.

### 5.3 What runs and what has never run

Across both stage ladders, **5 of 10 stages have never executed** **[V]**. VLM: B1
(projector init) and B2 (VL pretraining) have zero logs, zero result dirs, zero
checkpoints; B4 (RL) ran a 5-iteration smoke on 2026-06-24 whose checkpoint dir is
empty; only B3 (visual instruction tuning) is real (~24 multimodal jobs,
`iter_0002400`). LLM: A0 has only the 12-step mock-data smoke; A1 (CPT) has zero
logs in either repo and its launcher is the SFT recipe relabelled
(`RECIPE=..._sft_config`); A2/A3 ran for real; A4 ran with exactly one teacher.
`pretraining_data/` contains **SFT jsonl**, not a pretraining corpus; the
documented `Taiwan-formosa-VLM-caption-V1/data/` directory is empty (0 files) while
23 unreferenced parquet shards sit in `Formosa-Vision/data/` **[V]**. Any roadmap
that treats these stages as "mostly built" is mis-scoped — scaffolding produces
checkpoints on disk that look identical to real ones.

### 5.4 Fairness note

It would be easy to read this document as an indictment. The same team shipped
full-FT and RL runs of a 30B-A3B MoE VLM on GB200 trays, root-caused a
distributed-checkpoint aliasing bug by reading DCP `.metadata` by hand (six probe
jobs in one morning, then the correct one-line fix the same afternoon), built a
genuinely sophisticated reward/distillation stack (union-top-K log-space anchor
mixtures with per-token adaptive λ and a written record of four adversarial failure
modes), re-executed a grader agreement study to κ=0.9824, and maintains a
measured-report culture with explicit MEASURED/INFERRED/UNVERIFIED evidence flags.
The algorithm layer and the debugging discipline are strong. What the system lacks
is not talent or rigor in the small — it is a unit of reuse, a semantic gate, and
a run record; every structural failure in this analysis is one of those three
absences expressing itself at scale. That is precisely the gap FoundationScale
exists to close, and Parts 2 and 3 take up the component-level and gap analyses
accordingly.