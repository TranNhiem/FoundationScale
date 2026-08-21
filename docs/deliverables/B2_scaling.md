# Deliverable B.2 — Unified LLM/VLM Architecture, Distributed Design, Hardware Abstraction

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


**FoundationScale · Architecture Series B · System Design**

Evidence convention: **[M]** measured · **[V]** verified by source inspection · **[A]** cluster artefact census (2026-08-20) · **[U]** unverified. All paths are repo-qualified; `sdpo_gemma4` names two different components and is never written unqualified.

---

## 0. Scope and reading discipline

This document specifies the model/data/objective contract common to LLM and VLM training (§1), the Execution plane and its topology planner (§2), the declarative topology object that replaces three contradictory launch mechanisms (§3), the honest scaling ladder from 1 GPU to 100+ nodes (§4), the hardware abstraction over H100–GB200 (§5), and the RL service topology (§6).

Three things the evidence forbids:

1. **Claiming scale we do not have.** The measured ceiling is **8 nodes / 32 GPUs / EP=32** [M][V]. The only >8-node scripts in the tree are vendored NVIDIA examples (`Megatron-Bridge/examples/models/vlm/glm_45v/slurm_sft.sh` at `--nodes=64`, `qwen35_vl/slurm_sft.sh` at `--nnodes=16`) with no Omni logs behind them. They must be quarantined in the new tree, because they read as false evidence of scale.
2. **Claiming a missing lower end.** A single-GPU Megatron path exists and ran (six launchers defaulting `TRAIN_GPUS=${TRAIN_GPUS:-1}`; 14 logs `omni-accel/logs/e4b_arm_1458..1473.out` print `train GPUs=1(DP=1) TP=1 EP=1`, two reaching step 300) [V]. FSDP2 is therefore *not* the lower-end rescue (see §2).
3. **Treating scaffolding as capability.** Of the ten target pipeline stages, **five have never executed** [V] (§1.5). Mock-data smoke runs produce checkpoints indistinguishable on disk from real ones; all capability claims below are keyed to *executed* evidence.

---

## 1. The LLM/VLM common architecture

### 1.1 Where the paths must differ — and where they measurably must not

The two training codebases already assert the answer, negatively: everything that *should* be shared is cloned, and the clones drifted at exactly the correctness-critical seams. `accel/sdpo_r` vs `bridge/sdpo_r` have Jaccard 1.000 over 46 files [M]; `dpo_loss.py` exists in 25 copies, `dpo_dataset.py` and `verifiers.py` in 23 each [M]; and 11 filenames exist in *multiple AST-normalised variants* of materially different code — `rule_checks.py` in 5 variants over 24 copies, `sdpo_loss.py` in 4 over 21, `omni_sdpo_reward.py` in 2 over 21 [M].

Three convergent conclusions:

**(a) The objective must be shared.** Four distinct gold-handling code paths exist across **59** `reward_m1m4` definers collapsing to **8** md5s **[V·post]** (23 live-tree copies + 36 `repro/code_snapshot/` provenance copies; the earlier 24/6 inherited an inventory extension filter) — gold→1.0 short-circuit (2 copies, `sdpo`), gold-as-post-judge-floor to `threshold` default 0.6 (19 copies, `gspo` family), opt-in `OMNI_GOLD_MISS_IS_BAD` (1 copy), deterministic-grader-primary (1 copy, `omni-bridge/sdpo_gemma4` only) [V]. Only **three** distinct default behaviours result, because the third flag defaults off and no launcher exports it [V]. Reward modules are imported by bare name from CWD (`from omni_sdpo_reward import build_reward_fn`, `.../sdpo_gemma4/run_gspo.py:40`), so the objective is a property of which directory you `cd` into [V]. This caused a real failure: `gspo_official4` changed objective mid-run (steps 0–890 under policy P2, resumed at step 800 under P4, in one W&B run / one checkpoint dir / one audit file) [V]. Whether LLM or VLM, there is precisely one reward implementation: an installed, versioned `omni.reward` package with the gold policy as an explicit config enum (`short_circuit | post_judge_floor | miss_is_bad | grader_primary`), its version + md5 stamped into every run record (decision D6). One correction to folklore: the grader's agreement harness exists and was re-executed at κ = 0.9824 (`omni-bridge/sdpo_gemma4/shadow_grade/`), but "certified" remains the wrong word because the oracle (Kimi-K3) was prompted with gold_extract's *own rulebook* — the raters are not independent, and 100% of `fill_in_blank`/`short_answer` strata abstain [V].

**(b) The trainer loop must be shared.** ~27 near-identical RL drivers were produced by directory copy; the DP>1 reward-broadcast fix (`src=tp_src, group=tp_group`) is landed in 14 and absent from 13 [V]. One driver, one `RolloutScope(group, src)` derived once from `mpu` and handed to both the rollout client and the reward broadcast, with a startup assertion that the two are object-identical (D14).

**(c) What must genuinely differ is small, and today it is scattered.** Per-architecture knowledge is smeared across ~13 hand-written touch points per model family: chat-template quirks (`enable_thinking` rejected by the Gemma4 template), collate dialects (Gemma4 `image_position_ids` vs Qwen `image_grid_thw` vs Omni `num_image_tiles`), vocab-limit resolution (a TP-local shard bound once silently clamped 53.66% of CJK tokens across four "official" runs), softcapping (see §6.2), EOS sets ({1,106,50} for Gemma-4), and freeze targets [V]. The fix is a **ModelDescriptor** — one typed, registered record per architecture carrying all of the above — consumed identically by the LLM and VLM stage machinery.

The most dangerous duplicate in the tree is precisely a capability-rename: `modeling_gemma4_vl.py` exists as a 217-LOC variant *with* `freeze()` and a 171-LOC variant *without* one [V]. Resolving the wrong copy does not fail — it trains every parameter and reports success. Names are not identity (F7); FoundationScale identity is repo-qualified path plus content hash (P8).

### 1.2 The single data contract

One sample schema carries text, image, video, and audio; one pipeline (`source → sample → render/template/mask → sampler → collate`) serves both modalities. The contract that exists today is ShareGPT-style jsonl, and it is adequate as an *input* format; what failed was never the schema but the absence of an enforceable contract downstream of it:

- The stock chat template's `strip_thinking()` deleted CoT from targets, yielding **zero supervised tokens** under a healthy-looking loss curve; the fix is a live, default-on, env-gated patch `_gemma4_training_chat_template` (`Megatron-Bridge/src/megatron/bridge/data/vlm_datasets/collate.py:1138-1229`, gate `OMNI_GEMMA4_KEEP_COT`) [V].
- 23.42% of sampled rows silently lost vision supervision until an offline tag fixer (`scripts/fix_jsonl_tags.py`) and runtime injector existed [V].
- The loss mask is built by **string search over rendered text** (`create_multiturn_loss_mask_by_search`) with ~3% baseline all-masked rate [V] — fragile by construction.
- Over-length image microbatches on the B4 path are **skipped, not truncated**, dropping samples silently [V].

FoundationScale's data contract therefore has three moving parts, each replacing a named failure:

1. **A typed sample record** — conversation turns, lazy media references, `gold_answer`/`reference_answer`, `task_type`, and *provenance fields* (source file, row id, preprocessing-cache key). Two genuine corpus assets are currently wasted by the absence of this contract: 9,499 agentic trajectories (275 MB; 25,307 `<tool_call>` turns with tool-error recovery) that the Gemma-4 line has never seen because `OMNI_SFT_JSONLS` overrides the corpus to Taiwan-AIEC files [V]; and a 152 MB converted tool corpus orphaned purely because it speaks HF `messages` while the loader requires `conversations` [V]. Schema is a first-class adapter, not an orphaning mechanism.
2. **Template/mask as a first-class, contract-bearing stage** (promoted out of Dataset per the spine). The render is span-tracking, not string-search. The **template-parity launch gate** (D7) renders a fixed probe set through both trainer and serving path and asserts token-id equality plus non-empty supervision — replacing a manual md5 fingerprint and a `check_loss_mask.py` that was written and never run [V].
3. **The stratified sampler, lifted.** `StratifiedTemperatureBatchSampler` (stratification by (modality, task_type, source_dir), Hamilton largest-remainder allocation, caps/floors fixed point, WFQ interleave, pure function of (seed, epoch), `strata_signature` in `state_dict`) is the strongest reusable data component in the tree [V]. Its modality-homogeneity guarantee is load-bearing for *distributed correctness*, not data science: mixed-modality batches across DP ranks produce per-rank conditional compute graphs and NCCL deadlock [V]. The samples-per-modality contract becomes part of the batch, and its known defect — the resume livelock that fast-forwards a full epoch instead of the consumed prefix (`g4_sft/RELAUNCH.md`) — must be fixed before promotion, not inherited. Preprocessing nose-counts (video sidecar frame counts) are hashed into the cache key, because a `OMNI_MAX_VIDEO_FRAMES` / `--nframe` mismatch once silently disabled the entire sidecar cache [V].

This contract is deliberately stage-agnostic: A0 corpus rows, A2 SFT conversations, A3 RL prompts, and B1 caption rows are all instances of it. The *pretraining* half of the data plane does not exist — 0 of 1,344 first-party `.py` touch `GPTDataset`/`BlendedMegatronDatasetBuilder`/mmap corpora; 0 of 532 first-party `.sh` reference `.bin`/`.idx`/`--data-path` [V] — so the contract's text branch gets an indexed-corpus sibling (§4, rung 5 prerequisite), and `pretraining_data/` (which contains SFT jsonl under a pretraining name) stops lying about its contents.

### 1.3 The single Stage abstraction and freeze policy

A **Stage** is: `{model_handle, freeze_policy, objective_plugin, data_contract_binding, topology_ref, gates}`. The decisive fact is that the freeze *primitives already exist* and are production-proven: `freeze_vision_model` / `freeze_language_model` / `freeze_vision_projection` / `freeze_sound_encoder` / `freeze_sound_projection` are provider fields in `Megatron-Bridge/src/megatron/bridge/models/nemotron_omni/nemotron_omni_provider.py`, referenced first-party in `omni-bridge/sdpo_gemma4/run_sdpo.py` and `run_gspo.py`, and present in real checkpoint `run_config.yaml` files [V]. VLM Stage 1 (projector init) is therefore expressible *today* as a stage config — `model.freeze_language_model=True`, `model.freeze_vision_projection=False` — plus a caption adapter. It has simply never been run (§1.5).

Three defects sit exactly on this path and must be named in any design:

- `VLMLoRA` hardcodes `model.vision_model` / `model.vision_projection` (`Megatron-Bridge/.../peft/lora.py:203-208`), but `Gemma4VLModel` exposes `vision_tower` / `multi_modal_projector` — it `AttributeError`s as written. This one at least fails loudly.
- `_peft_common_vlm` ships *plain* LoRA, not VLMLoRA (`Megatron-Bridge/.../recipes/common.py:544`): a "B1 via PEFT" stage trains LLM adapters, never touches the projector, and **reports success**. This one is silent.
- B4's freeze policy is hardcoded at `omni-bridge/sdpo_gemma4/run_sdpo.py:131-145` with no override channel, and a "text-only smoke" comment sits on flags set *unconditionally* — the image run also trained with a frozen projector [V].

So `freeze_policy` is a *declared* field of every Stage, resolved against the ModelDescriptor's module names (not path heuristics), and — the cheapest high-value addition in the whole system — **every run logs its effective trainable-parameter partition** (~10 lines). Today the freeze flags appear in no config dump in any log, so even for the ~24 B3 jobs that ran, the trainable partition is recoverable only by reading recipe source and guessing which duplicate resolved [V].

### 1.4 The single objective interface

Objectives are plugins behind one signature: `loss(student_ctx, teacher_ctxs[0..K], batch, config) → (loss, metrics)`. The plugin registry admits CE (SFT/pretrain), DPO-variants, SDPO/POLAR self-distillation, GSPO, FACTS/POLAR_A anchored co-teaching, and chunked-logit distillation — all of which exist as working, CPU-tested code (`omni-bridge/sdpo_gemma4/{sdpo_loss.py, gspo_loss.py, chunked_logit_loss.py}`, `polar/polar_a_loss.py`) [V]. Two consolidations are preconditions, not options:

- `polar_a_loss.py` and `facts_loss.py` are **byte-identical twins** differing only in import source and metric prefix [V] — unify before anyone generalizes a fork.
- "FACTS" denotes two unrelated things in the tree (a reward variant, `sdpo_facts`; and the POLAR_A anchored loss). The registry keys disambiguate (F7).

The objective interface carries one mandatory behaviour that the audit discovered the hard way: **the objective asserts its own identity at step 0.** `run_gspo.py`'s `--old_logp_source` defaults to `"self"`, making `pi_old = pi_theta.detach()` from the current forward — the importance ratio is identically 1.0, the clip band can never bind, and **7 of the 10 official GSPO arms ran with no trust region at all** while logging structurally healthy curves [V]. Framework default: `old_logp_source=frozen`, and the step-0 invariant gate asserts the executed objective equals the declared one (`seq_ratio_mean=1.0` exactly + `clip_fraction=0.0` exactly is a *failure signature*, not a health signal). The 7 affected runs are not invalid, but they are not GSPO runs either, and nothing currently labels them — the run record must carry objective identity.

Generalizing the 2-teacher `compute_polar_a_loss(student, selfT, anchor, anchor_bare)` to K teachers is a contained change: union-top-K index sets, arithmetic log-space mixture, per-token adaptive λ with a student-ahead guard, and EMA gap tracking all already exist, and the in-file FIX-1..FIX-4 comments are a written record of the four failure modes an N-teacher rewrite would otherwise rediscover [V]. What is genuinely new engineering is K-teacher *residency*: nine co-resident 26B teachers do not fit, so A4 needs a remote-logprob teacher service — built on the judge-registry substrate (`judge_registry.py` family→pool router) but with `judge_pool.py`'s round-robin dispatch (`itertools.cycle`) replaced by content-aware selection, because round-robin is load balancing and is actively wrong for "matching teacher" semantics, and `run_odpo.py's` teacher path currently collapses plural endpoints to `models[0]` [V]. Weight fusion is categorically absent (zero slerp/TIES/task-arithmetic/model-soup/DARE hits across 9,905 files) [V] and is **out of scope** for FoundationScale — there is no prior art, no merged-artifact verification harness, and no budgeted risk.

### 1.5 How A0–A4 and B1–B4 become instances — and which have ever run

The Stage machinery is justified by an asymmetry the roadmap must respect: **5 of the 10 stages have never executed** [V].

| Stage | Same machinery? | Executed? | Blocking gap to run it as a Stage config |
|---|---|---|---|
| A0 pretrain-from-scratch | yes — CE plugin + corpus data plane | **No** — 12-iter mock-data smoke only (job J01, val PPL 4.3e5 = random) | the entire `.bin`/`.idx` data plane (D12) |
| A1 continued-PT | yes | **No** — zero CPT logs; its launcher is the SFT recipe relabelled (`launch_omni_cpt_omni.sh`, aborts at line 114 without `iter_0000000`) | real CPT objective + weighted replay sampler |
| A2 SFT cold start | yes | **Yes** — richest executed surface | corpus routing (the A2 agentic corpus never reached Gemma-4) |
| A3 verifiable RL | yes | **Yes** — most exercised stage; 10 GSPO runs | objective-identity gate (7 ran with no trust region) |
| A3 subjective RL | yes | **Yes** — judge-decided rewards; ODPO job J02, J24 real steps | fail-closed reward gates (†) |
| A4 multi-teacher distill | yes — K-teacher objective plugin | **Partially** — single frozen teacher ran 4,169 logged steps; the "multi" part has never existed | remote-logprob teacher service |
| B1 projector init | yes — this *is* the Stage/freeze design point | **Never** — 0 logs, 0 checkpoints | a caption corpus (the documented `Taiwan-formosa-VLM-caption-V1/data/` is **empty**, 0 files; `Formosa-Vision/data/` holds 23 unreferenced parquet shards) [V] |
| B2 VL pre-training | yes | **Never** | no interleaved loader, no document-level loss path; *every* task encoder masks loss to assistant spans [V] |
| B3 visual instruction tuning | yes | **Yes, at scale** — ~24 multimodal jobs, `iter_0002400` | none blocking |
| B4 VLM RL | yes | **Smoke only** — 5 iterations 2026-06-24; checkpoint dir **empty** | hardcoded freeze policy; skip-not-truncate media; reward cascade is HTTP-LLM-judge, not a reward model |

(† The reward gates currently *fail open*: a `rule_checks` import failure silently disables the degeneracy veto, a verifier exception counts as a pass, the gold floor can promote a *wrong* answer via substring fallback (gold `答案：25` vs answer 15 → reward 0.9412), and 2 of 6 live task types (`fill_in_blank`, `short_answer`) are hardcoded to always abstain with an in-source `# BUG 3` comment [V]. Fail-closed gating is a contract-level fix: gates raise; abstention is an explicit, counted outcome.)

The load-bearing reuse proof: `omni-accel/train_resume_test_e4b.py` drove the real `megatron.bridge.training.pretrain` entrypoint with a random-init `Gemma4E4BModelProvider()` and `checkpoint.load=None` end-to-end on this cluster [V]. FoundationScale reuses that entrypoint (D12); nothing in the trainer is rebuilt. Note the asymmetry: every pretraining-adjacent asset lives *only* in `omni-accel`; the sibling repo has none. And a capability inventory must never count mock-data smoke checkpoints as pretraining capability — identical on disk, different in kind.

---

## 2. Distributed training architecture

### 2.1 The Execution-plane interface

One interface; three backends, in order of reality:

```
interface ExecutionBackend:
    def build_mesh(topology: Topology) -> Mesh                      # TP/PP/EP/CP/DP groups
    def build_model(descriptor: ModelDescriptor, mesh) -> Module   # MegatronModule check §2.3
    def train_step(policy: StepPolicy) -> StepResult
    def save(state, path) -> CheckpointRef                         # reshardable by construction
    def load(ref: CheckpointRef, mesh) -> state                    # layout-agnostic (DCP)
```

- **Megatron-Core is the production backend, today and tomorrow** (D1/P7): 97 first-party files import Megatron-Bridge, 62 import Megatron [M]; TP/EP/CP, the distributed optimizer, and DCP work at 30B-A3B MoE on GB200. Reimplementing is indefensible.
- **Single-device is a degenerate mesh of the same backend**, not a separate stack. It exists and ran (§0). "1 GPU" in FoundationScale means Megatron at TP=1/EP=1/DP=1 with no NCCL — which is what the fourteen `e4b_arm` logs already demonstrate.
- **FSDP2/DTensor is gated behind a Phase-4 measurement** (D2). The honest justification is narrow: add it only if it measurably improves laptop/single-node iteration speed (plain `state_dict`, no DCP resharding, no EP bookkeeping) or unblocks non-NVIDIA CI. Current first-party FSDP imports: **zero** [M] — nothing is at risk in deferring it. The first pass's justification ("the honest answer to 1 GPU") was refuted by the discovery of the existing Megatron single-GPU path; do not resurrect that argument.

Everything else in the ecosystem is declined deliberately: zero first-party imports of NeMo, NeMo-RL, DeepSpeed, Transformers-Trainer, standalone PEFT entrypoints, or Ray [M]. Revisit Ray only if the rollout fleet needs dynamic autoscaling. Both repos remain Megatron-only: 748 of 2,888 non-vendored `.py` import megatron [V].

### 2.2 The topology planner

Today, parallelism geometry is hand-written GiB arithmetic in launcher comments ("~80–105 GiB / 184 GiB GB200"), and the derived-quantity math hardcodes `world=32` (`DP_lm=$((32/TP/CP))`), so banners silently misreport on any other node count [V]. The exercised envelope is narrow and measured: TP ∈ {2,4,8}, EP ∈ {16,32}, ETP=1, sequence-parallel always on, **PP=1 always, CP=1 always** (plumbed, never exercised) [V]. Any FoundationScale claim of CP or PP support cannot cite this codebase as evidence.

The planner replaces comment arithmetic with a solver over a declared objective:

```
plan(model_params_bp, active_params_bp, seq_len, mbs, gbs,
     device_mem_gib, nodes, gpus_per_node, nvlink_domain_size,
     requires: {ep_sharded_experts, video? -> mbs == 1}) ->
    Topology(tp, pp, ep, cp, dp, sp, etp, mbs, ga)
```

Rules the evidence forces into the planner as *constraints with citations*: video implies MBS=1 (`collate.py:423` assert; job J25 hung 32 ranks) [V]; `moe_shared_expert_overlap=False` is mandatory when EP < world (job J13 deadlocked) [V]; TP must divide the KV-head count (31B dense ⇒ TP ≤ 4); global vocab bounds must be computed as `local_embedding_shard × TP` (the 53.66% regression) [V]; EP-sharded checkpoints must be reshardable by construction (§2.3); TP groups spanning node boundaries (TP=8 on 4-GPU nodes) carry an unmeasured inter-node all-reduce cost — the launcher comment asserting TP=4 is preferable "for comms" is asserted, not measured [V], so the planner must *measure* this rather than inherit folklore. Pretraining at 100B+ will force PP>1 and CP>1 — both are zero-prior-art here and enter the ladder as gated rungs (§4), not as claims.

### 2.3 The conformance test and the checkpoint contract

The interface means nothing without the conformance test (D2): a tiny model trained to step N on each backend must produce matching loss curves within tolerance, and matching weights after checkpoint round-trip through DCP at a different mesh. DCP layout-agnostic load is a *proven* capability of this stack (training at TP=4–8/EP=16–32 loaded back at TP=1 on a single node for export) — the single-source export design (WORLD_SIZE=1, EP=1, one GPU, ~4 min for 51.6 GB) exists *because* the checkpoint carries the parallelism [V].

A conformance suite also needs the three checkpoint invariants from D8, because this program's empirical record is that structural gates pass on broken models and only semantic gates catch it:

1. **Build-time (structural):** refuse to construct a model containing any parallel-aware submodule whose nearest ancestor is not a `MegatronModule`. The entire expert-aliasing catastrophe was one wrong base class — `Gemma4DenseMoE(torch.nn.Module)` — silently downgrading `sharded_state_dict` to a plain-torch flatten [V]. The one-line fix is live at `Megatron-Bridge/.../gemma4_provider.py:445,457` [V]; **do not cite `EXPERT_SAVE_BUG.md` as current** — it is banner-marked SUPERSEDED and describes six failed intermediate probes, not the fix.
2. **Save-time (structural):** on the *first* checkpoint of every run, assert expected parameter bytes per EP rank, EP-reshardability, and zero locally-indexed expert keys. This exact arithmetic (45.70 GB vs 5.73 GB, 0 vs 960 indexed keys) is what eventually caught the bug — two full runs too late [A].
3. **Post-export (semantic):** a logit-parity / generation probe before any artefact is promotable. Byte-completeness cannot detect a permuted expert axis; `BAD_INCOMPLETE_..._1723` carries a flawless 1013-tensor/60-expert index over 11.5% of the required bytes [A]. **[U]: no existing export has ever had this check** — though the weight-level worry underneath it is now retired, 200 of 205 dirs verified against Megatron at 0 DIFFER with no permuted expert axis anywhere [V·post]. It is one GPU and a handful of prompts; retrofitting it is a Phase-1 task and now the single cheapest correctness win remaining. **Specify it to assert positive work:** `top-1 agreement == 1.0`, `KL < 1e-3`, *and* a non-zero compared-tensor count — the sweep that closed the weight-level question initially reported `all_identity: true` on a corrupt artefact because the expert tensors were absent and `all([])` is `True`. A gate that can pass vacuously is not a gate.

The structural-bug blast radius is the sharpest argument in this document for semantic gating: 9 Gemma4-MoE runs trained from the aliased base (2 full-FT, 7 LoRA whose adapters are structurally clean while sitting on an 87.5%-wrong frozen base) [A], and the aliased-base Muon-vs-AdamW comparison **inverted** on the fixed base (1.4288/1.4847 became AdamW 0.7412 beating Muon 0.7828, matched configs, already complete) [V]. A structural defect silently reversed the conclusion of a controlled experiment. No structural gate can see that class of failure; the conformance plan cannot either, which is why gate 3 is non-negotiable.

---

## 3. The declarative topology object

### 3.1 What it replaces

Topology currently has three contradictory sources of truth [V]:

1. `#SBATCH` directives — the basis of every census, and systematically misleading;
2. an internal `TRAIN_GPUS`/`NTRAIN` variable that *contradicts* them — the single-GPU trainers request a full exclusive 4-GPU tray and self-restrict via `CUDA_VISIBLE_DEVICES`, which is how a directive-based analysis missed them entirely;
3. 31 `offslurm_*` scripts whose torchrun command is **string-grepped out of `DRYRUN=1` output**, supported by exactly **1 of ~240 launchers** — and yet off-Slurm is the *current production path* (`omni-bridge/sdpo_gemma4/offslurm_run.sh:2-7`, `offslurm_gspo_official1..10_cmd.sh`; `TRAYS_IN_USE.txt` lists four live trays) [V].

Consequence: the small-scale path was invisible, off-Slurm is a bolt-on, and a `#SBATCH`-histogram reads as a topology census while missing both. (Also: these histograms miss the partition-name split — `<partition>` 188 vs `<partition_alt>` 4 [M] — a telling detail about string-typed cluster identity.)

### 3.2 The object

```yaml
topology:
  nodes: 6
  gpus_per_node: 4
  mesh: {tp: 2, pp: 1, ep: 1, cp: 1}   # dp derived: 24/2 = 12 — validated, never hand-asserted
  device: {type: auto, mem_gib: auto, nvlink_domain: auto}
  services:                            # empty for pure training; see §6
    rollout: {replicas: 4, placement: separate_nodes, weight_sync: refit_manager}
  launch: {backend: slurm | offslurm-enroot | local,
           container: "nemo-automodel-26-04_compute.sqsh",
           entrypoint: "foundationscale.train", module_args: [...]}
  invariants:                          # executable preflights, declared not grepped
    - tp_divides_kv_heads
    - gbs_divisible_by: [dp, mbs]
    - vocab_global_bound: local_shard * tp
    - objective_identity_step0
```

One declaration; three emitters:

- **`emit_slurm()`** — renders the `#SBATCH` block *from the object* (not vice versa), the container mount, the NCCL/cluster-profile env (§5), and `srun`/torchrun stage assembly with a job-id-derived rendezvous port (`29100 + jobid % 800`, as in `gspo_v2/launch_gspo_g4dense31b.sh`) [V]. Resume policy (tri-state auto/never/always, `latest_checkpointed_iteration.txt` detection, run-complete exit-0) is generated, not per-script folklore.
- **`emit_offslurm()`** — renders the ssh + `enroot start --rw` + bare-torchrun constellation directly, making off-Slurm *universal rather than a 1-of-240 hack*, because the launcher no longer needs to know which path it is on. The existing `offslurm_multinode.sh` watchdog discipline (two-strikes dead-rank detection over ssh ps-grep, full constellation relaunch) is preserved as a backend policy [V].
- **`emit_local()`** — renders `torchrun --nproc_per_node=$NTRAIN --master_addr=127.0.0.1`, generalizing the existing `TRAIN_GPUS` knob that has 14 real runs behind it [V].

All three are rendered from one resolved artefact, so P3's provenance discipline (resolved config + code hash + data hash + seed + topology + objective version) cannot drift between launch modes — today, `OMNI_GOLD_MISS_IS_BAD` was active for ≥12 ODPO runs and *nothing in either repo sets it*: it was exported by hand in an interactive shell and propagated by `sbatch --export=ALL`, leaving zero on-disk trace [V]. Environment is for secrets and cluster paths only (D5); 166 distinct env vars (37 `OMNI_*`, 26 `SDPO_*`, 16 `GSPO_*`, 16 `VLLM_*`...) [M] retire into the typed config. Only three real recipe names exist (`gemma4_vl_26b/31b_omni_sft_config`, `nemotron_omni_omni_accel_peft_config`) [M] against 205–247 launchers — the launcher layer was already config-starved.

This simultaneously solves both ends of the scale story: the small-scale path becomes *discoverable* because single-GPU is a first-class render (`nodes: 1, gpus_per_node: 1`), and off-Slurm becomes universal because nothing about it is special.

---

## 4. Scaling from 1 GPU to 100+ nodes

**Honest baseline.** **The ceiling is now an execution fact, and it is a *choice* [V·post].** Slurm accounting over all **1,774** Omni jobs gives a node histogram `{1:1242, 2:299, 3:2, 4:134, 5:2, 6:2, 8:93}` topping out at **8 nodes / 32 GPUs**; max degrees are TP=8, EP=32, ETP=1, **PP=1 — pipeline parallelism was never >1 in any log** — and CP peaked at 2. The off-Slurm path tops out *lower* (6 nodes / 24 GPUs), so it does not raise the ceiling, and 213 jobs allocated exactly 1 GPU, confirming the single-GPU floor. The decisive nuance for planning: **other users on this same cluster have run 18 nodes / 72 GPUs**, against a capacity of 23 nodes / 92 GPUs. **8 nodes is ~35% of what was available — the ceiling belongs to this codebase, not the hardware.** Also: W&B is a dead channel for topology (35,001 local run dirs are login-node backfill; 238 sampled configs contain zero `world_size`/TP/PP/EP keys).

Maximum ever executed: 8 nodes / 32 GPUs / EP=32 (the tp8ep32 full-FT launchers), and the largest live RL configuration is 6 nodes × 4 GPUs, TP=2 ⇒ DP=12 [M][V]. Two lower-end stories exist but **do not compose**: the random-init path has only run at 4 GPUs on mock data; the single-GPU path has only run from a pretrained checkpoint [V]. **There is no proven 1-GPU from-scratch path.** The 100+-node target is unmeasured by three orders of magnitude, full stop. Node histograms are reported here only with their caveat: they count `#SBATCH` directives, miss the internal-GPU path, and miss the 31 off-Slurm scripts [M].

**The validation ladder.** Each rung names what breaks first (from the incident record, where known) and what must exist before attempting it.

| Rung | What we know | What breaks first | Must exist before attempting |
|---|---|---|---|
| **1: 1 GPU** | Proven — 14 runs, 2 to step 300, Megatron degenerate mesh [V] | — | done; conformance baseline established here |
| **2: 1 node (4 GPU, DP>1)** | **Proven at DP=4 by job J08 (1500/1500 steps) — but *not* in the co-located configuration**, where DP=1 remains the default because of a hang whose recorded explanation is refuted and whose cause is bounded to 3 confounded factors **[V·post]** | the co-located NCCL hang only | **One short diagnostic run** (no vLLM, `TRAIN_GPUS=1,2`, DP=2, 1 iter, `NCCL_DEBUG=INFO` in the `srun --export=` allowlist); plus a DP>1 smoke asserting two DP groups never receive identical reward vectors for different completions (D14) |
| **3: 2 nodes / 8 GPUs** | Proven class: TP=8 spanning nodes, EP=8 (production GSPO/SDPO) [V] | TP groups crossing nodes pay inter-node all-reduce — unmeasured | planner measure-not-folklore rule (§2.2); quantified NVLink-vs-IB split |
| **4: 8 nodes / 32 GPUs** | Proven — the TP8/EP32 full-FT ceiling [V]; the DP=12 GSPO campaign also lives here | EP=32 exhausted (4 experts/rank); save-time invariants now exist for this mesh | — this is current production; harden it while proceeding |
| **5: 32 nodes / 128 GPUs** | **Unmeasured [U]** | PP>1 or CP>1 become *necessary* for 100B+; both are zero-prior-art here; EP=64 untried; NCCL topology (§5) | pretraining data plane (D12) if from-scratch; PP/CP conformance runs stolen from *upstream* Megatron evidence plus our own step-0 invariants; straggler/timeout budgets derived from the 600s/480s NCCL limits |
| **6: 100+ nodes** | **Unmeasured [U]** | everything above, plus elastic/requeue failure modes (no in-job fault tolerance exists anywhere in the current tree — no elastic torchrun, no `--requeue`, no signal traps [V]) | failure-injection harness; cost-model validation; a *measured* scaling study before any claim above 32 GPUs |

**On the co-located NCCL hang.** **The recorded explanation is refuted; the cause is bounded but not named [V·post].** The stall is at `Megatron-Bridge/.../training/initialize.py:640` — `init_process_group` is called with no `device_id=`, so that barrier is where `ncclCommInitRank` actually runs. Both occurrences gave **~24 minutes of total silence: no NCCL WARN, no watchdog abort, no traceback**, because **`NCCL_DEBUG` was never set in any run in either repo** (it exists only inside a comment). **Job J08 is a control the team did not know it had**: same node, same container, same nested `srun`+`torchrun`, *identical* `NCCL_IB_DISABLE=1` / `NCCL_NVLS_ENABLE=1` / `MASTER_ADDR=127.0.0.1` — and it ran **1500/1500 steps at DP=4**. That single artefact kills the IB flags, NVLS, the loopback master, nested torchrun, a bad node, cross-socket NVLink and IMEX-domain absence — including the launcher's own claim that *"vLLM on GPU0 blocks GPU1↔GPU2 NVLink"*, which also contradicts its own line 181. Three factors remain **perfectly confounded** — co-resident vLLM, a `CUDA_VISIBLE_DEVICES=1,2` subset excluding the occupied GPU, and `srun --overlap` — since the trainer excludes GPU0 *because* vLLM is there. **The decisive experiment is one short run**: no vLLM, `TRAIN_GPUS=1,2`, DP=2, 1 iteration, `NCCL_DEBUG=INFO` exported **and** added to the `srun --export=` allowlist (variables off that list never reach the container — likely why no NCCL output ever existed). Unrelated but fix it anyway: the launcher's EXIT trap `kill -9`s every compute process on every GPU on the node, safe only by `--exclusive`.

---

## 5. Hardware abstraction: H100 → H200 → B200 → GB200 → next

**What the evidence says actually differs.** The current tree is welded to one cluster generation: node names `<compute-nodes>`/`<compute-node>*` in code and YAMLs, partition `<partition>` (and its `<partition_alt>` spelling), Bright-Computing Slurm paths, `bond0`/`mlx5` NCCL env, one `.sqsh` image, absolute `<CLUSTER_HOME>/...` paths [V]. Everything hardware-dependent that this program *learned expensively* maps onto a small capability set:

| Capability | Evidence it bites | Policy |
|---|---|---|
| **Device memory** — H100/H200 (80/141 GB) → B200/GB200 (~184 GiB per launcher arithmetic) | every geometry choice in §2.2 was computed by hand against 184 GiB [V] | capability query feeding the topology planner; never a comment |
| **NVLink domain size / MNNVL+IMEX** | the cluster-wide IMEX prolog outage drained every Slurm submission and is the *reason the off-Slurm production path exists* [V]; TP=8-across-nodes sends all-reduce over IB | query (domain size; IMEX up/down) driving TP placement; off-Slurm emitter as engineered fallback, not accident |
| **FP8 / NVFP4** | the CPT launcher exposes an `FP4` env override [V]; NVTE_* vars in the env census (5) [M] | declared precision policy per platform; softly gated conformance run before a precision is allowed to differ by backend |
| **TransformerEngine version coupling** | a pip install into the first-on-PYTHONPATH overlay once shadowed container torch and broke transformer_engine for *all* launchers; quarantined manually into `_quarantine_eo_dep*` [V] | container pins; preflight asserts TE/torch pair, not a grep |
| **NIC topology** | IB HCA prefix-matching including BlueField ports (above) | capability query enumerating compute-eligible HCAs; pinning generated, not `mlx5` |

Two rules generalize the audit's lessons:

1. **Capability queries, not assumptions.** Anything a launcher currently hardcodes (device GiB, NVLink domain, HCA names, container image, partition/QOS limits — which in the tree are already contradictory: headers assert 7-day limits while scripts request 14 days and UNLIMITED [V]) becomes a `ClusterProfile`: data, versioned, diff-checked. The `<partition>`/`<partition_alt>` partition split [M] is what happens when cluster identity lives in strings.
2. **Precision is selected per platform by the registry, and asserted at run time.** The vendored trees are *selected by PYTHONPATH ordering* (three coexist: `Megatron-Bridge`, `_vendor_bridge_expertfix`, `_vendor_bridge_g4fix`) and validated by grep preflights [V]. FoundationScale replaces this with patch-as-plugin against a pinned upstream (D1): a test that fails when an extension point moves, and a run-record field naming the exact tree and commit — because today no artefact records which of the three trees a run used.

---

## 6. The RL service topology

### 6.1 Colocated vs disaggregated is a policy flag, not an architecture (D10)

Today the RL constellation is: a Slurm training job plus vLLM rollout/judge fleets stood up by hand in tmux/enroot **outside Slurm's control**, on nodes held by borrowable 48-hour hold-jobs [V]. Two measured costs follow: the SDPO README concedes training GPUs idle during rollout/judge phases [V]; and the sidecar tier is invisible to `sinfo`, so reservation collisions are handled by node-pinning conventions [V]. Meanwhile `smoke_refit_e4b.sh:94` demonstrates the colocated single-tray recipe (GPU0 train / GPU1 rollout / GPU2 export) and it works [V].

FoundationScale: `services: {rollout: {...placement: colocated | separate}, judge: {...}}` on the topology object, emitted as first-class scheduler allocations (Slurm heterogeneous jobs where available), managed lifecycle (GPU-PID reaping before restart — killing tmux does not reap detached `EngineCore` workers [V]), and served-name identity verification (`make_judge_registry.py --check` exists because an endpoint can be up serving the *wrong* model → silent reward 0 [V]).

### 6.2 The weight-refit protocol and its enemies

Refit is checkpoint-export-based: `SDPORefitManager` (blue-green pooled variant in `.../sdpo_gemma4/refit_manager.py`) exports Megatron→HF on a separate host, reloads the sidecar pool with readiness polling against `/v1/models`, tracks per-replica weight ages, and fires tripwires at p99 weight-age > 12 steps [V]. It is inert unless both `export_fn` and `reload_fn` are wired — and the completed 200-step run served a *fixed* iter_7200 policy start-to-end while the trainer drifted, defended only by IS-clip [V]. The policy flag must therefore never allow staleness to default to unbounded (jobs J20/J21/J22/J23 collapsed behind `REFIT_EVERY=0` [V]); K is a mandatory first-class field with the deterministic K=0 mode reserved for A/B reproduction.

The refit protocol is also where the two nastiest silent-objective findings land:

- **Trainer/sampler numerical parity.** Megatron does not apply Gemma-4's `final_logit_softcapping=30.0`; vLLM does. Measured consequence: rollout logprobs unusable as `pi_old` (seq_ratio 0.062 at step 0, clip_fraction pinned at 1.0, all negative-advantage gradients zeroed) [V]. `gspo_loss.py` re-applies the softcap trainer-side for exactly this reason. Generalize: a step-0 parity probe comparing trainer and sampler logprobs on the same batch, as a launch gate.
- **Objective identity (§1.4).** The services plane makes the 7-of-10-arms failure *impossible by construction*: the declared objective in the resolved config is what the refit manager, the rollout client, and the loss plugin all read — there is no second channel.

### 6.3 The admission currency: KV tokens, not requests

Measured on this cluster's 8-engine vLLM K3 endpoint during this analysis: 16 concurrent requests at ~250K-token context decoded at **10 tok/s aggregate**; 24 concurrent at ~37K-token context decoded at **270–438 tok/s aggregate** — a 27–44× swing from per-request context length alone, at comparable concurrency, prefill finished, `num_requests_waiting` = 0 throughout [M]. Every structural health signal read normal while throughput had collapsed.

Two consequences for the services plane: **(1)** admission is budgeted by *aggregate KV tokens* (concurrency × context length), exposed as `services.rollout.kv_budget_tokens`, because a scheduler that admits by request count is choosing a point on a cliff blindly — and long-context high-concurrency decode is the *normal* regime for RL rollouts, not an edge case; **(2)** service health requires a *throughput* gate (tokens/s against an expected floor per policy), in addition to liveness — the same F3 pattern (structurally healthy, semantically broken) observed in checkpoints, appearing in serving. That repetition across unrelated subsystems is the strongest evidence that the missing semantic gate is a systemic property of this stack, and it is why Contracts (≥1 semantic gate per boundary) is a crosscutting plane rather than a lint rule.

---

## Closing position

The framework's claim to elegance is this: B1 is a *config* (freeze fields exist today); B3 is *production* (recipe + data contract); A2/A3 are *proven objectives* behind one plugin interface; the trainer is *reused* (`megatron.bridge.training.pretrain` drove random-init end-to-end from first-party code [V]); the launcher is *one object* rendering three modes; and the scale claim is *a ladder with measured rungs*, of which rungs 1, 3, and 4 are done and rungs 2, 5, and 6 each have a named, gated blocker. Nothing in this design requires a component the codebase has not already demonstrated at least once — except the pretraining data plane, the remote-logprob teacher service, and measured evidence above 32 GPUs. Those three are the honest scope of new engineering.

**Dissent / open risks.** (1) D2's FSDP2 deferral is correct on evidence but carries a coherence cost: the Execution interface cannot be validated by the conformance test with only one working backend; the single-device path must be promoted to a full second backend in Phase 1, or the interface is an aspiration. (2) §3's universal off-Slurm emission inherits a hazard the bolt-on version did not have: when off-Slurm is one flag away, nothing stops two operators from double-allocating a tray; the emitter must refuse to render without a claim token, not merely warn. (3) All claims about what runs at 100+ nodes are **[U]** and should remain in this document phrased exactly that way until the ladder reaches them.