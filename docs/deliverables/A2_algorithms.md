# FoundationScale — Deliverable A.2: Algorithms and Techniques (Part 2 of 3)

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


**Audience.** Senior engineers building FoundationScale. Every claim cites the source it came from; confidence is marked [M] measured, [V] verified by direct inspection, [A] artefact census, [U] unverified. Superseded documents (`EXPORT_STATUS.md` 2026-08-03, `EXPERT_SAVE_BUG.md`) are not cited as current state anywhere below.

---

## 1. The objective layer: what RL actually optimises here

The codebase is not short of algorithms — it is short of a *unit of reuse* for them. The algorithm layer below is genuinely strong, but it exists as ~20–27 forked directory trees, and the single most important caveat for everything in §1 is applied at the end: **the exact objective any given run optimised is a property of the directory you `cd` into, not of the run config** [V]. Read each subsection as "the best-known variant of X", with the divergence note attached.

### 1.1 SDPO — on-policy self-distillation

**What.** A port of TRL's `trl.experimental.self_distillation`, VLM-adapted. Per step: sample N rollouts on one prompt, score them with a reward cascade (§2), mine the best above-threshold sibling as an in-context demonstration, re-prompt the *same* model with that demo plus optional environment feedback, and distill the re-prompted forward (teacher, `no_grad`) into the original forward (student) over completion tokens [V].

**The loss** (`omni-bridge/sdpo_gemma4/sdpo_loss.py`, 4 AST-normalised variants over 21 copies [M]) supports three distillation modes:
1. **Top-K α-divergence** — student top-K index set, teacher gathered at those indices, renormalised over the K bucket with an optional appended tail bucket carrying residual log-mass (`_add_tail`).
2. **Full-vocab α-divergence**.
3. **Token-level reverse-KL score-function surrogate** — `(log π_s − log π_t).detach() · log π_s`, restricted to α=1.0.

Optional IS-clipping against rollout-time logprobs (`exp(clamp(Δlogp, ±20)).clamp(max=clip)`); four aggregation rules (grpo / bnpo / dr_grpo / dapo).

**Why it exists.** Reward is a *selector*, never a loss weight: the binary reward picks the teacher sibling, and the distillation loss does the learning. This side-steps reward hacking of a weighted objective but makes verifier correctness (§2) load-bearing.

**Dependencies.** vLLM sidecar rollout over HTTP, `sdpo_teacher_context.py` for mining, TP-aware completion-span logit extraction via `gather_from_tensor_model_parallel_region` (`sdpo_step.py`), and `finalize_model_grads` called manually because the driver bypasses `forward_backward_func`.

**FoundationScale disposition: REFACTOR into the Objective plane as the canonical loss plugin.** The multimodal adaptation in `sdpo_teacher_context.py` (teacher prompt re-tokenised through the processor *with the same images/videos*, fixing an upstream TRL misalignment bug) is real, reusable knowledge [V]. One known live divergence to adjudicate during the port: the sdpo_e4b/polar variant removed a shorter-demo tie-break after it caused premature-EOS collapse; the 13-copy gspo-lineage family **still carries the buggy tie-break** (`key=lambda s: (..., -len(...))`) [V].

### 1.2 POLAR — a rename with real extensions, and one byte-identical twin

`polar/` is nominally a rename of `sdpo/` — `polar/sdpo_teacher_context.py` is a two-line wildcard re-export shim, and `omni_polar_dataset.py` is a name-swapped clone [V]. POLAR's substantive deltas are layered extensions on the SDPO loss:

**POLAR-R (repulsion).** A bounded, span-localised unlikelihood term on rule-flagged degenerate tokens: `coef · log p` with `coef = clamp(p/(1−p), max=g_cap).detach()`, IS-corrected, normalised per repel span. Every no-op path returns a graph-connected zero (`student_logits.sum()*0.0`) so DP/reduce-scatter collectives stay rank-symmetric [V]. fp32 upcast on the completion span because bf16 rounds `1−p` to 0 for p>0.99. **Reuse the pattern; unify the denominator conventions** (repulsion normalises by span count, the distill term by `aggregate_weights` — easy to mis-replicate) [V].

**POLAR-RT (teacher-regret token localiser).** Per-token regret = teacher top-1 logp − teacher logp(sampled token); flag tokens above `tregret_min`, keep top-k per row capped at `tregret_max_frac`, drop the row if >50% of tokens look wrong. Merged into the repel mask via `torch.maximum`. An in-comment claim of 87% top-3 / ~0.1% FP cross-sibling validation is [U] (no artefact cited in-source).

**POLAR-A (frozen-anchor mixture) — the crown jewel.** `polar_a_loss.py` (referenced from `polar/run_polar.py`) mixes a **frozen anchor model** *into the distillation target* in log space: `z_T* = (1−λ)·z_selfT + λ·z_anchor`, arithmetic mixture renormalised over a **union top-K bucket** of student and anchor logits — the union is load-bearing because a collapsed student's own top-K excludes the anchor's coherent tokens and yields ~zero pull [V]. Per-token λ switches lo/hi on tregret/repeat/script-flagged tokens, forced lo when the student is ahead (student-ahead guard). A robust anchor-gap statistic (95%-quantile-trimmed median, EMA-tracked in `cfg.anchor_gap_ema`) sets the routing threshold `A_hi`; tokens with gap ≥ A_hi route to JS divergence instead of forward KL. An always-on bare-prompt mass-covering floor (`D_α(student ‖ anchor_bare)`) prevents collapse on prompts without demonstrations.

The FIX-1..FIX-4 comments in this file are a written record of four adversarial failure modes (sub-normalisation, geometric mixture pinning P(EOS)≈1, student-only top-K starving the anchor of gradient, unstable gap threshold) that any N-teacher rewrite would otherwise rediscover the hard way [V].

**Two required corrections before reuse:**
1. **[A] `polar_a_loss.py` and `facts_loss.py` are byte-identical twins** differing only in import source and metric prefix. A4's core objective is forked before anyone has extended it. **Unify first.**
2. **Disambiguation (mandatory): "FACTS" names two unrelated things [V].** (i) The POLAR-A anchored co-teaching *loss* above. (ii) `sdpo_facts/` — a *reward* variant adding the opt-in `OMNI_GOLD_MISS_IS_BAD` env flag (§2). Any document that says "FACTS" unqualified is ambiguous.

**FoundationScale disposition: EXTEND, don't rewrite.** Generalising `compute_polar_a_loss(student, selfT, anchor, anchor_bare)` from 2 to N mixture members is a contained change; see D13 of the decision spine.

### 1.3 GSPO — sequence-level importance ratios

**What.** A pure-PyTorch implementation of GSPO (arXiv:2507.18071) in `omni-bridge/sdpo_gemma4/gspo_loss.py`: a length-normalised sequence log-ratio `s_i = exp(mean_t(log π_θ − log π_old))`, PPO-style pessimistic clip in a deliberately tiny band `[1−3e-4, 1+4e-4]`, plus a `gspo_token` surrogate (`sg[s_i]·exp(logp − sg[logp])`) so per-token gradients flow while the ratio stays sequence-level. Rich self-diagnostics: `seq_ratio_mean/absdev`, `clip_fraction`, ESS fractions, a token-multiplication probability error probe [V].

**The π_old contract, and the most consequential finding in the audit.** π_old is *always trainer-computed*, because vLLM rollout logprobs were measured unusable on Gemma-4: Megatron does not apply `final_logit_softcapping=30.0`, so step-0 `seq_ratio` was 0.062 with `clip_fraction` pinned at 1.0 — the served policy and the training policy were numerically two different models [V]. The fix re-implements softcapping in `gspo_loss.py` before log_softmax.

**[V] But: 7 of the 10 official GSPO arms ran with no trust region at all.** `--old_logp_source` defaults to `"self"`, meaning `π_old = π_θ.detach()` from the *current* forward → ratio identically 1.0 → the clip band can never bind → no importance correction. The flag's own help text states this and gives the measured signature (`seq_ratio_mean=1.0` exactly, `clip_fraction=0.0` exactly). The jobs were structurally healthy while the algorithm executed was not the algorithm named. **FoundationScale must default `old_logp_source=frozen`, and the objective must assert its own identity at step 0** (e.g. refuse to start if the configured and effective π_old sources differ, or if `clip_fraction==0.0` for >N consecutive steps under a non-trivial clip band).

**[V·post] This finding now has independent corroboration from a completely different measurement channel**, which matters because the original evidence was a flag default plus a self-reported diagnostic — i.e. the run's own telemetry, the thing under suspicion. A checkpoint-weight comparison run for an unrelated purpose (export parity) measured how far the `gspo_official*` refits actually moved the model: **roughly one bf16 ULP over hundreds of steps** — `official10` iter 25→125 at max |Δ| 6.1e-05, `official4` iter 100→800 at 4.9e-04. That is the displacement profile of a policy taking gradient steps with no importance correction and nothing binding, observed in the weights themselves rather than in the metrics. It also has a practical consequence for anyone verifying these artefacts: an exact-step weight match on a GSPO export is nearly uninformative, because adjacent steps are almost identical anyway — a step-discrimination control is required before such a match may be reported as a pass.

**Reward-broadcast correctness (retracted-bug hygiene).** The DP>1 defect "rewards for rank-0's completions applied to other DP groups" is **fixed in the driver that actually runs**: `mb/sdpo_gemma4/run_gspo.py:815,916` scores on `tp_src` and broadcasts with `src=tp_src, group=tp_group` under an explicit "BUG FIX (2026-08-12)" comment [V]. The fix form is present in 14 trees and absent from 13 — the 48% fix-miss rate measured on a live correctness bug [V]. The residue in `omni-accel/gspo/run_gspo.py:417` (`src=0`) is **dead code, not a live hazard**: its sole launcher is 1 node / TP=4 ⇒ DP=1, and at TP>1 the fork hard-crashes at step 0 before its silent TP==1 mistrain path is reachable [V; the crash is confirmed, the unreachability rests on surviving launchers/logs — no surviving launcher or log configures DP>1]. **Delete the tree; do not fix it.** The forensic artefact is worth keeping in mind: the pre-fix sibling at DP=12 ran 1,876 steps with **472 steps of `grad_norm` exactly 0.000 while logging `reward/mean=0.794, success=1.00`** [V]. A quarter of a training run that was not training, and said it was fine.

**FoundationScale disposition: REFACTOR to one driver.** A single RL driver that obtains one `RolloutScope(group, src)` from `mpu` and hands it to *both* the rollout client and the reward broadcast; startup assertion that the two are object-identical; a DP>1 smoke test that fails if two DP groups receive identical reward vectors for different completions (D14).

### 1.4 DPO / ODPO — preference pairs, online, judge-repaired

Classical DPO exists only as heavily cloned library code (`dpo_loss.py` ×25, `dpo_dataset.py` ×23 [M, strict set: see the counting caveat]). The production-relevant variant is **ODPO** (`omni-bridge/sdpo_odpo/`): per prompt, k rollouts are scored by the reward cascade, and a deterministic decision table maps the spread to exactly one training decision — mixed → on-policy DPO pair; all-good-with-spread → contrast best vs worst good; all-good-flat → NLL-only (or skip); all-bad → **teacher fallback**: an external HTTP teacher (Kimi-K3 via `--teacher_endpoints`, or the judge model by default) *authors* the `chosen` completion, which must then pass the degeneracy rule gate, a deterministic gold check, and a judge score threshold before admission (`judge_gen.py`, `odpo_decide.py`, `odpo_pairs.py`, CPU-tested in `test_teacher_gold_cpu.py`) [V]. Budget machinery is real: per-prompt teacher slots, per-step budget charging, fallback fraction caps, wall-clock deadlines.

Two documented failure lessons worth institutionalising: "gold could only ADMIT, never REJECT" is called out in-source as a fixed bug [V]; and a run where only 19/333 steps yielded gradient shows a *config failure masquerading as health* [V]. The reference model is either a co-resident frozen anchor or — neatly — **the policy with its LoRA adapter toggled off** (`odpo_lora.py`, `make_adapter_toggle`), so π_ref costs a forward pass, not a residency slot.

**FoundationScale disposition: REFACTOR into the Objective plane**; keep the decision-table/pearl pattern (deterministic, broadcastable, CPU-testable) and the adapter-off-as-reference idiom.

---

## 2. The reward stack: a four-tier cascade with 4 divergent gold policies

**What.** `reward_m1m4` is the orchestrator; per completion the cascade is [V]:

1. **Rule gate** (`rule_checks.py`): pure-regex degeneracy detectors — emoji spam, word-span loops, CJK char loops, digit runaway, identical lines, unfinished CoT, premature-EOS tail. Any hard hit → 0.0, judge skipped.
2. **Deterministic verifiers** (`verifiers.py` ×23 [M]): per-family checks — `json_valid` (gated per-sample on the question actually demanding JSON), `language` script check, `tool_schema`. Failure → 0.0.
3. **Gold handling** — *this tier is where the objective diverges; see below.*
4. **Generative judge** (`m1m4_judge.py`, 2 variants ×21 copies [M]): ONE batched multimodal judge call per row scoring a rubric tree of atomic Yes/No/Abstain criteria, graded *against the evidence* (images as base64, videos as `file://`), with the reference treated as a fallible hint (`reference_trust` scales reference-based weights). Abstain removes a criterion from the denominator; confidence-gated vetoes (C_no_hallucination, C_factually_correct at confidence ≥ 0.6) can zero the score; a separate CoT-quality channel Q enters the *selection* score `R_cont = A_eff·(0.5 + 0.5·has_cot·Q·Q_veto)` used for teacher re-ranking. `_budget_candidate` always preserves the final answer and head/tail-elides the CoT middle against the judge window. Judge calls are injected (`judge_call_fn`) so the whole stack is CPU-testable — a pattern to keep.

**The divergence, precisely [V].** **59** files define `reward_m1m4`, collapsing to **8** md5s **[V·post]** — of which **23 are live-tree copies (the shadowing surface, unchanged) and 36 are `results/<run>/repro/code_snapshot/` provenance copies**, i.e. the extra files are the audit trail, not more shadowing. (The earlier 24/6 count inherited `fs_inventory.json`'s extension filter — 9,905 indexed entries against **19,912** real `.py` files — the same filter that caused the κ miss. F7 again, on this document's own method.) They implement **4 gold policy code paths but only 3 distinct default behaviours**:

| Policy | Behaviour | Copies | Repo-qualified location |
|---|---|---:|---|
| P1 short-circuit | gold match → reward 1.0, judge skipped | 2 | `accel/sdpo`, `bridge/sdpo` |
| P2 post-judge floor | judge runs; `A = max(A, threshold)`, threshold default **0.6 not 1.0** | 19 | the gspo family incl. `accel/sdpo_gemma4` |
| P3 miss-is-bad | P2 + opt-in `OMNI_GOLD_MISS_IS_BAD` → gold miss forces 0.0 | 1 | `bridge/sdpo_facts`; **defaults off, no launcher exports it** |
| P4 grader-primary | `gold_extract.grade()` first: True→1.0, False→0.0, None→judge | 1 | **`mb/sdpo_gemma4` only** |

Reward modules are imported from the CWD by bare name (`from omni_sdpo_reward import build_reward_fn`, `run_gspo.py:40`), so the objective is selected by directory. Note the name collision: `sdpo_gemma4` means P2-in-`accel/`, P4-in-`mb/` [V]. Both P3/P4 flags are read **at import time** (`mb/sdpo_gemma4:787` carries a dead duplicate `_GOLD_MISS_IS_BAD`) — mid-run flips silently no-op [V].

**Objective provenance is recoverable — from audit logs, not code [V].** Every RL launcher passes `--judge_audit_path`, and the per-sample `decided_by` vocabulary fingerprints the policy (`gold_fastpath`→P1, `gold+judge`→P2, `gold_miss`→P3, `gold_rule_pass|fail`→P4). Two consequences are worse than the original unknown: (a) `gspo_official4` **changed objective mid-run** (steps 0–890 P2, resumed at step 800 under P4, one W&B run, one audit file — not a single-objective curve); (b) P3 was active for ≥12 ODPO runs with **zero on-disk trace** — exported by hand in a shell, propagated by `sbatch --export=ALL`.

**Three live silent-objective defects, all in the same family [V]:** the gold floor can promote a wrong answer via substring fallback (production audit row: gold `答案：25`, answer 15, reward **0.9412**); `fill_in_blank` and `short_answer` are hardcoded to always abstain (`# BUG 3`) — 2 of 6 live task types can never be rule-decided; and the gates **fail open** — a `rule_checks` import failure silently disables the degeneracy veto and a verifier exception counts as a pass.

**On "certified."** The grader-agreement statistic **exists** and was re-executed during this audit: `sdpo_gemma4/shadow_grade/` reproduces κ=0.9824 (99.13% agreement, 1,724 decided pairs, 95.8% coverage; judge-vs-oracle κ=0.160) [V — an earlier retraction of this was itself retracted; the harness was missed because the audit's file inventory excludes `.jsonl`, where the evidence lives]. "Certified" remains the wrong word for a better reason: the oracle is Kimi-K3 prompted with *gold_extract's own rulebook* — the raters are not independent — and the 76 excluded rows are the hardest strata (100% of `fill_in_blank`/`short_answer` abstain). Defensible wording: *a faithful, reproducible implementation of a shared rulebook*.

**FoundationScale disposition (D6): REBUILD as one installed, versioned `omni.reward` package.** Gold policy = explicit config enum (`short_circuit | post_judge_floor | miss_is_bad | grader_primary`); reward version + md5 stamped into every run record; gates fail *closed*; `gold_extract.py` is lifted as the default verifier with a human-labelled kappa artefact checked in beside it. The substrate to adjudicate first: `rule_checks.py` is the most-forked file in the tree (5 variants / 24 copies), `judge_registry.py` 2×23, `judge_pool.py` 2×22, `m1m4_judge.py` 2×21 [M].

---

## 3. The three teacher mechanisms — and what A4 actually lacks

There are **three** teacher mechanisms, not two [V; an earlier pass said two]:

1. **Self-distillation** (§1.1) — same policy re-prompted with its own mined best sibling. On-policy, per-token, runs.
2. **One frozen co-resident anchor** (§1.2 POLAR-A) — `--anchor_ckpt` is a *single optional string* in `run_polar.py:379`, `sdpo_gemma4/run_sdpo.py:586`, `sdpo_e4b/run_sdpo.py:497`. A second full model, weights-only, `requires_grad_(False)`, eval mode, on the same TP/EP/DP mesh (`run_polar.py:235-250`). Ran for 4,169 logged steps [V].
3. **An HTTP frozen teacher** (`sdpo_odpo`, §1.4) — authors the `chosen` completion. Teacher-as-data-generator, not logit distillation. Plural `--teacher_endpoints` are failover replicas of one model, and the teacher path collapses any pool to `models[0]` (`run_odpo.py:396`) [V].

**Categorically absent** [V, 9,905-file census; adjudicator footnote: keyword census + 31 full reads establishes *no evidence of*, not provable absence]: N>1 teacher residency; per-rollout routing to a *matching* teacher by content; expert/weight **fusion** (zero hits for slerp / TIES / task-arithmetic / model-soup / DARE / mergekit / weight-averaging; every `*merge*` artefact is LoRA-into-its-own-base).

**The near-miss that matters.** `judge_registry.py:50-116` is a live family→pool router with glob matching, lazy pool construction and heterogeneous multi-model pools (`judge_registry_rt_r04dp2.yaml` declares a `dual` pool mixing qwen36_35b and gemma4_31b). But it routes *graders*, dispatches round-robin for load balancing (`judge_pool.py:44-58`) — actively wrong for "matching teacher" semantics — and its own comment states the limitation (`run_odpo.py:190-192`). **Right shape, wrong axis.** And per §D13's caveat: this substrate is itself forked (2 divergent variants each of `judge_registry.py`/`judge_pool.py`), so extension begins with *adjudication*, not coding.

**The wall is memory, not code [V].** Nine co-resident 26B teachers will not fit on any mesh; a multi-teacher design almost certainly must serve experts over HTTP (a remote-logprob teacher service) — which is another argument for reusing the judge registry/pool substrate rather than the co-resident anchor substrate. Guard: keep an N=1 configuration byte-identical to today's single-anchor path (`anchor_enabled=False` already dispatches to the untouched `compute_polar_loss` — preserve that discipline, or the ablation proving multi-teacher was worth it becomes unmeasurable).

---

## 4. The chunked logit loss — the memory primitive under §1

`sdpo_gemma4/chunked_logit_loss.py` re-implements the full SDPO/POLAR/FACTS loss surface as a chunked-over-completion backward: slice hidden states to the completion span, project through the real output layer in chunks, `.backward()` per chunk, accumulate into the detached hidden-state leaf and output weight, reattach via `hidden.backward(gradient=...)`. Peak logit memory drops from `[S,V]` (~16 GiB at S=32768, V=262144) to `[chunk,V]` [V].

**Why it is trustworthy:** `test_chunked_logit_loss_cpu.py` proves loss *and* both gradient tensors agree with the reference across chunk sizes, dtypes, loss types, and feature combinations [V]. **Why it is fragile:** it claims to reproduce `sdpo_loss.py` "exactly", but it is a *re-implementation*, and one drift is already recorded (fp32 upcast vs the reference's deliberate bf16) [V]. Reduction schemes live in two places (`aggregate_loss` / `aggregate_weights`); the test is the only guard.

**FoundationScale disposition: REUSE as the standard distillation memory primitive, with the CPU equivalence test promoted to CI** and the duplicated reduction algebra collapsed to one implementation.

## 5. Data-side algorithms

**StratifiedTemperatureBatchSampler** (`omni_sdpo_dataset.py`, gemma4/facts/e4b copies): strata = (modality, task_type, source_dir); weights `n_s^tau_task · override`; modality frequencies `n_m^tau_mod` clamped by caps/floors via fixed-point redistribution; Hamilton largest-remainder integer allocation; oversample ceiling with spill; weighted-fair-queue interleave with consecutive-run caps; rotating-window cross-epoch coverage; `state_dict` carries a `strata_signature` (logs, not crashes, on corpus change). Pure function of (seed, epoch, strata) → all DP ranks compute identical global batches → DP-correct by construction. Opt-in difficulty weighting (`OMNI_DIFFICULTY_JSON`, Efraimidis-Spirakis keyed sampling) exists only in the gemma4 copy [V]. **REUSE — this is the strongest single reusable component in the data layer.**

**ModalityBucketBatchSampler:** modality-homogeneous global batches so every DP rank takes the same `vision_tower` branch — a *distributed-deadlock* fix, not an efficiency device (jobs J11–J12) [V]. **REUSE the invariant; note the known unfixed resume livelock** (fast-forwards a full epoch instead of the consumed prefix, `g4_sft/RELAUNCH.md`) [V].

## 6. LoRA / PEFT

`odpo_lora.py::build_megatron_lora` encodes the known-correct build ordering (pre-wrap hook registered *and* passed to `get_model`; optimizer built after wrap; adapter-count and trainable-fraction assertions; `finetune=False` flip for adapter resume), r32/α64 on qkv/proj/fc1/fc2, MoE experts provably never adapted (merged `experts.gate_up_proj max|Δ|=0.0`) [V]. Two B1-blocking defects: `VLMLoRA` hardcodes `model.vision_model`/`vision_projection` but `Gemma4VLModel` exposes `vision_tower`/`multi_modal_projector` — **AttributeError as written** (`MB/peft/lora.py:203-208`); and `_peft_common_vlm` ships plain LoRA instead, so "B1 via PEFT" trains LLM adapters, never touches the projector, and reports success — silently (`MB/recipes/common.py:544`) [V].

## 7. MoE parallelism and precision specifics

**MoE [V]:** alltoall token dispatcher, grouped GEMM, permute fusion, ETP=1 always, EP∈{16,32} (4–8 experts/rank of 128); `moe_shared_expert_overlap=False` is a mandatory deadlock workaround when EP<world (job J13). TP∈{2,4,8} with sequence-parallel always on; CP plumbed but never >1; PP never used; recompute configured off everywhere (the "selective recompute on [moe]" OOM recipe is untested comment folklore). The expert-save/load correctness story is told in full in A.3's checkpoint chapter; the salient fact for algorithms is that the entire EP=8 silent-corruption incident came from *one wrong base class* (`Gemma4DenseMoE(torch.nn.Module)` instead of `MegatronModule`) — and that Huang-Law consequence propagates into FoundationScale as a build-time invariant, not a training-time hope [V, A].

**Precision [V]:** bf16 training with fp32 distributed-optimizer main params; the interesting precision work is defensive: fp32 upcast on the repulsion completion span; a bf16-stable `_bf16_logsumexp` that exists **only in some forks** (polar still upcasts — clone drift in a numerics primitive); chunked logits capped into `[chunk,V]` to avoid the fp32 `[S,V]` spike. No FP4/FP8 training path in first-party code (NVTE appears in 5 env vars; TransformerEngine imports are vendored-only) [M].

---

## 8. The pipeline coverage matrix — the decision-relevant table

"Executed" means runtime evidence exists (logs, checkpoints, result dirs), not merely that code exists. Both ladders: **5 of 10 stages have never executed** [V]. Any roadmap treating these as "mostly built" is mis-scoped.

| Stage | Status | Executed? | Evidence | The one blocking gap |
|---|---|---|---|---|
| **A0** Pretrain from scratch | **Partial** (path yes, program no) | 12-iter **mock-data** smoke only | `omni-accel/train_resume_test_e4b.py` (random-init `Gemma4E4BModelProvider`, `checkpoint.load=None`, `MockGPTDataset`, real `megatron.bridge.training.pretrain` entrypoint; job J01, 7.52B params, val PPL 4.3e5 = random init) [V] | **The data plane**: 0 of 1,344 first-party `.py` touch GPTDataset/BlendedMegatronDatasetBuilder/mmap corpora; 0 of 532 `.sh` reference `.bin`/`.idx`/`--data-path`/`blend`; no tokenizer training. Guard: mock-data smoke checkpoints are on-disk indistinguishable from real ones — do not count scaffolding as capability [V] |
| **A1** Continued pre-training | **Partial** | **No** — zero cpt/pretrain logs in either repo [V] | `launch_omni_cpt_omni.sh` hard-aborts without `$MEGATRON_CKPT/iter_0000000` (`:114`) and passes `checkpoint.pretrained_checkpoint` (`:117`) [V] | The launcher reuses the **SFT recipe** (`RECIPE=..._sft_config`) — masked instruction data, not packed raw text; replay is proportion-by-file-size ("Exact per-source weighting is a follow-up"); forgetting gate documented, eval disabled |
| **A2** SFT cold start | **Implemented** | **Yes**, with a routing accident | `g4_sft/` v3 corpus + `launch_g4{moe26b,dense31b}_sft*.sh` [V] | 9,499 genuine multi-turn agentic trajectories (275 MB; 38,588 `<think>` / 25,307 `<tool_call>` / 28,882 `<tool_response>` turns) exist but the Gemma-4 line **overrides the corpus via `OMNI_SFT_JSONLS` and has never seen them**; a 152 MB converted tool corpus is orphaned purely by schema (`messages` vs required `conversations`) [V] |
| **A3** Per-domain RL | **Implemented** | **Yes** — most-exercised stage; 10 official GSPO arms, ODPO job J02 (254 real steps), one live GSPO run at audit time [V] | §1.3, §1.4, §2 above | Objective identity is not trustworthy: 7/10 GSPO arms ran with no trust region (`old_logp_source=self`); 4 gold policies selected by CWD; gates fail open; `official4` changed objective mid-run. The stage *runs*; what it ran is in question |
| **A4** Multi-teacher on-policy distillation | **~60%, and the missing 40% is exactly the word "multi"** | **Partially** — 4,169 logged steps of the single-anchor objective [V] | per-token rKL core + chunked loss CPU-proven (`chunked_logit_loss.py`); POLAR-A 2-source log-space mixture with FIX-1..4 scar tissue (`polar_a_loss.py`); co-resident frozen anchor with correct stop-gradient (`run_polar.py:235-250`) | Per-rollout teacher *selection* (the routing key), N>1 teacher *residency* (memory wall ⇒ remote-logprob teacher service), and any fusion step (zero evidence in 9,905 files). Build routing on the judge registry; unify the byte-identical `polar_a_loss.py`/`facts_loss.py` twins first; fusion needs its own certification before any number from it is trusted [V, A] |

| Stage | Status | Executed? | Evidence | The one blocking gap |
|---|---|---|---|---|
| **B1** Projector init (encoder+LLM frozen) | **Partial — primitives yes, recipe no** | **Never** — 0 logs, 0 result dirs, 0 checkpoints [V] | Freeze primitives exist and are live: `freeze_vision_model` / `freeze_language_model` / `freeze_vision_projection` / `freeze_sound_encoder` / `freeze_sound_projection` in `Megatron-Bridge/.../nemotron_omni/nemotron_omni_provider.py` + `conversion/auto_bridge.py`; B1 is expressible today as `model.freeze_language_model=True model.freeze_vision_projection=False` [V] | A recipe and a corpus. The documented `Taiwan-formosa-VLM-caption-V1/data/` is **empty (0 files)**; `Formosa-Vision/data/` has 23 parquet shards **referenced by nothing**. Two defects sit on the path: `VLMLoRA` AttributeError (`MB/peft/lora.py:203-208`) and `_peft_common_vlm` shipping plain LoRA (`MB/recipes/common.py:544`, silent projector-skip) [V] |
| **B2** Interleaved VL pre-training | **Absent** | **Never** [V] | Energon is wired for `qwen3_vl`/`nemotron_omni`; multi-image scatter and per-image bidirectional masking work | The objective, not the plumbing: every task encoder masks loss to assistant spans; full-document NTP over an interleaved stream has no code path. Compounding: `_inject_missing_markers` fires on **23.42%** of sampled rows, destroying the document ordering B2 exists to learn [V] |
| **B3** Visual instruction tuning | **Implemented** | **Yes, at scale** — ~24 multimodal jobs, `iter_0002400` [V] | `omni_maker` → `nemotron_omni_collate_fn`; KEEP_COT template patch + stop-token-106 supervision (both live, default-on, Gemma4-only) [V] | None blocking. Cheapest high-value gap: **no executed run records which modules were frozen** — logging the effective trainable-parameter partition is ~10 lines and unblocks everything above |
| **B4** RL with a reward model | **Partial — mislabelled** | **5-iteration smoke only** (2026-06-24); **checkpoint dir empty** [V] | End-to-end code exists (§1–§2) | There **is no reward model** — reward is an HTTP LLM-judge. Vision-token invariant tests exist (Gemma4/Qwen collate CPU tests) but B4's freeze policy is hardcoded at `run_sdpo.py:131-145` with no override channel, and its "text-only smoke" comment sits on flags set **unconditionally** — the image run also trained with a frozen projector. Over-length image microbatches are **skipped, not truncated** [V]. Note correction: Gemma-4 VL multimodal forward parity is done and measured (image cosine 0.999661, video 0.998345, `g4_parity_mm_1643.out`) — do not re-assert earlier "parity remaining" claims |

Two cross-cutting rows the matrix cannot hold: `polar/run_sdpo.py` is a **3-line stub** — variant counts for `run_sdpo.py` are 3 AST-variants/8 copies (strict set) vs 18 distinct md5s/23 copies (byte-wise incl. vendored); never quote either bare [M]. And **the two lower-end stories do not compose**: the random-init path has only run at 4 GPUs on mock data; the single-GPU path (`TRAIN_GPUS=1`, 14 logs, two to step 300) has only run from a pretrained checkpoint. There is no proven 1-GPU from-scratch path [V].

---

## 9. Disposition summary

| Component | Verdict | Why (measured) |
|---|---|---|
| SDPO core loss + teacher-context mining | **Refactor** into Objective plane | 4 variants × 21 copies; one correctness fix present in one lineage, absent from the 13-copy family [V] |
| POLAR-R / -RT repulsion stack | **Reuse code, unify conventions** | Graph-connected-zero and fp32-upcast patterns are hard-won; denominators already drifted [V] |
| POLAR-A anchored mixture | **Extend to N teachers** | Union-top-K + adaptive λ + FIX-1..4 record; but `polar_a_loss.py` ≡ `facts_loss.py` byte-identical — unify first [A] |
| GSPO loss + π_old contract | **Refactor; change the default** | 7/10 arms ran ratio≡1.0; softcap parity fix is correct and must be contract-enforced [V] |
| ODPO decision table + judge-authored pairs | **Refactor** | Deterministic, CPU-testable, budget-bounded; gold-as-gate-fixed lesson embedded [V] |
| Reward cascade | **Rebuild as versioned package** | 4 gold policies by CWD; fail-open gates; substrate files are the most-forked in the tree [V, M] |
| `gold_extract.py` grader | **Reuse** as default verifier, ship the harness | κ=0.9824 reproducible but rater-not-independent; human-label artefact ~8 person-hours [V] |
| Chunked logit loss | **Reuse; equivalence test → CI** | Exact-gradient proof exists; drift already observed [V] |
| Stratified temperature sampler | **Reuse as-is** | DP-correct by construction; invariant-tested [V] |
| Modality bucket sampler | **Reuse invariant, fix resume livelock** | Deadlock prevention is mandatory; the fast-forward bug is live and unfixed [V] |
| LoRA/PEFT build path | **Reuse ordering discipline; fix VLMLoRA/plain-LoRA confusion** | Adapter-off-as-π_ref is elegant; B1 path AttributeErrors as written [V] |
| Multi-teacher routing / expert fusion | **Build on judge-registry substrate / Do not build fusion** | Registry exists with the right shape and the wrong axis; fusion has zero prior art and needs certification before trust [V, A] |

The net: FoundationScale inherits a strong, battle-scarred algorithm core inside a reuse boundary that has measurably failed (48% fix-miss; mid-run objective swaps with no on-disk trace; 5 of 10 stages unexecuted). Part 3 takes up the gates and invariants these algorithms require to be trustworthy.