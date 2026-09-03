# FoundationScale — Architecture Decision Spine

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


Working document. This is the *decision layer*: the judgments that deliverables A–D
are written against. Every decision cites the measured evidence that forces it.

Status legend: **[M]** measured deterministically · **[V]** verified by direct source
inspection · **[A]** established by exhaustive artefact census on the cluster
(2026-08-20) · **[K]** from the Kimi codebase read, not independently verified ·
**[U]** explicitly unverified — stated as unknown, not as fact.

> **Revision note (2026-08-20).** Seven high-stakes claims from the first pass were put
> through adversarial verification against the source, plus an independent census of 224
> MoE checkpoints, 23 HF export directories and 20 run configs. **Three were refuted and
> two partially refuted.** Those corrections are folded in below and the retracted
> statements are named explicitly in §5, because a deliverable that repeats a refuted
> bug report is worse than one that omits it.

---

## 0. The diagnosis

The system is not short of algorithms. The algorithm layer is genuinely strong: an
SDPO/POLAR family with three distillation modes and IS clipping, GSPO with a
length-normalised sequence ratio, a FACTS anchor mix with EMA routing and JS gating,
a chunked logit loss with exact gradients and a CPU equivalence proof, a four-tier
reward cascade with per-criterion audit records, and a stratified temperature sampler
with largest-remainder allocation and DP-correctness by construction.

Nor is it short of engineering discipline in the small. The bugs listed below were all
found and fixed, several within hours, by people reading DCP metadata by hand.

What is missing is a **unit of reuse**, and a **semantic gate**.

The team's unit of experimentation is the *directory*. So the unit of reuse became the
directory too, and every fix is a copy. **[M]** ~61% of first-party Python is redundant
duplication: 137,568 of 227,357 LOC across 98 AST-normalised clone clusters on the
strict file set (1,319 files), or 149,647 of 241,803 across 112 clusters on the looser
one (1,344). *Both numbers are stated because they measure different file sets — which
is itself the lesson of F7: never quote an N without defining the set.* `dpo_loss.py`
exists in 25 copies, `verifiers.py` and `dpo_dataset.py` in 23 each; `accel/sdpo_r` and
`bridge/sdpo_r` have Jaccard 1.000 over 46 files.

**And the copies are not interchangeable.** **[M]** 11 filenames exist in *multiple*
AST-normalised variants — same name, different code:

| File | Variants | Total copies |
|---|---:|---:|
| `rule_checks.py` | **5** (379/318/387/397/265 LOC) | 24 |
| `sdpo_loss.py` | **4** (365/477/405/309 LOC) | 21 |
| `sdpo_rollout_vllm.py` | 3 | 20 |
| `sdpo_teacher_context.py` | 3 | 22 |
| `run_sdpo.py` | 3 (840/1081/787 LOC) | 8 |
| `modeling_gemma4_vl.py` | 2 (**217 / 171** LOC) | 4 |
| `judge_registry.py`, `judge_pool.py`, `m1m4_judge.py`, `omni_sdpo_reward.py` | 2 each | 23 / 22 / 21 / 21 |

This is the precise mechanism behind the 48% fix-miss rate. A change described as
"fixing `rule_checks.py`" lands in one of five mutually-incompatible versions and
silently misses the other four, and nothing in the tree records which version a run
used. The unit of reuse is not merely a directory — it is a *directory whose contents
have drifted apart unobserved*.

Two riders, both **[V]**, that make this sharper than a maintenance complaint:

*The divergence removes capability, not just style.* The 171-LOC variant of
`modeling_gemma4_vl.py` **has no `freeze()` method at all**; the 217-LOC variant
(`MB/models/gemma4_vl/modeling_gemma4_vl.py:207-217`) does. `freeze()` is the single
primitive the entire VLM stage ladder is built on. A build that resolves the wrong copy
does not fail — it trains every parameter and reports success. That is the most
dangerous duplicate found so far, and it is dangerous *specifically* because the two
copies share a name and differ by a capability no caller checks for.

*And the counts themselves depend on the file set, again.* `run_sdpo.py` is 3 variants /
8 copies under AST-normalisation of the strict first-party set, and **18 distinct md5s
across 23 copies** counting vendored trees byte-wise (`BRIDGE/sdpo_gemma4/` is 1730 LOC;
`ACCEL/sdpo_gemma4_mm/` is 1161; `polar/run_sdpo.py` is a **3-line stub**). Same file,
same tree, 3 vs 18 depending on the question asked. This is F7 restated: an unqualified
fork count is not a fact.

This matters because the copying is **deliberate**. `PRETRAINING_PIPELINE.md` defends
it: *"each new algorithm is a self-contained fork that toggles on via a default-off
config flag, giving a clean A/B against the base."* That is a real benefit — perfect
experiment isolation, zero risk of breaking a running arm. The cost is that
correctness-critical semantics drift silently between copies, and that every fix has an
N-copy propagation cost with a **[V]** demonstrated ~48% miss rate.

> **Therefore FoundationScale's first job is not "a better trainer". It is to make
> experiment isolation cheap without copying code** — because that is the only reason
> the copying is tolerated. Any design that does not replace the *benefit* of cloning
> will simply be cloned around.

### The structural failures the evidence forces us to design against

**F1 — No unit of reuse, so a fix is an N-copy chore that is measurably not completed.**
**[V]** The DP>1 reward-broadcast fix (`src=tp_src, group=tp_group`) is present in 14
driver trees and **absent from 13**, which still carry the global `src=0` form. **[V]**
24 files define `reward_m1m4`; they collapse to 6 distinct md5s implementing **4 distinct
gold policies** across ~20 sibling directories. Reward modules are imported by bare name
from the current working directory, so **the objective RL optimises is selected by which
directory you `cd` into.**

**F2 — No provenance.** **[M]** `omni-accel` — the *more active* repo, 3,010
files / 681,440 LOC — **is not a git repository at all**. Its sibling is, with HEAD
`5744a25`, last commit 2026-07-11, 7 commits in 90 days, and **892 dirty files**.
Configuration is spread across **[M]** 247 launchers with sbatch directives, **166
distinct environment variables** (37 `OMNI_*`, 26 `SDPO_*`, 16 `GSPO_*`, 16
`VLLM_*`, 15 `JUDGE_*`), ~20 cloned judge-registry YAMLs, and only **4 distinct
recipes**. "Reproducibility" is manual tree copies into `code_snapshots/`, which freeze
stale node endpoints. **A past run cannot be reproduced from committed artefacts,
because most of the code was never committed** — and, per F1, **[A]** which reward
semantics a historical W&B run optimised is not recoverable from the code at all.

**F3 — Every gate in the system is structural; there is no semantic gate anywhere.**
This is the sharpest finding of the verification pass, and it replaces the looser
"silent correctness loss" framing. The canonical case: a Gemma4-MoE checkpoint that was
**87.5% wrong** (16 of 128 distinct experts, replicated 8×, from a wrong module base
class) passed `rc=0`, successful resume, a healthy loss curve, correct tensor counts and
correct dtypes, for two complete training runs. **The only detector that ever fired was a
byte sum.**

And a byte sum is itself insufficient. **[A]** `BAD_INCOMPLETE_..._1723` carries a
*perfect-looking* `index.json` — 1013 tensors including all 60 expert tensors — over
5.94 GB of a required 51.6 GB. No byte count can detect a permuted expert axis at all.
**[U]** No exported model in the program's history has ever been checked by logit parity
or a generation probe against its Megatron source — and that remains true today, even
though the *weight-level* question underneath it was closed post-draft (§5: 200 of 205
dirs, 0 DIFFER, no permuted expert axis, on CPU). The distinction is the whole point of
this section: a tensor comparison cannot see a wrong `rope_theta`, a missing
`final_logit_softcapping`, a mismatched attention implementation, a tokenizer drift or a
bad generation config. **Every artefact this program has ever served was served on the
strength of byte accounting alone.**

The same pattern recurs across the record: a stock chat template that deleted CoT from
targets, yielding *zero* supervised tokens under a healthy-looking loss curve; 23.4% of
rows silently losing vision supervision; rewards computed for one DP group applied to
another. Every one was caught late, by a human, by hand, by reading bytes.

**F3a — The blast radius is wider than the bug reports say.** **[A]** A census of all 20
Gemma4-MoE runs shows **9 trained from the aliased base `gemma-4-26b-it`**: 2 full-FT
(`official_fullft_16k`, `muon_fullft_16k` — 20 checkpoints, all carrying the 5.73 GB /
960-indexed-key bug signature) and **7 LoRA runs** (`lora_matched_16k`, `lora_recompA_16k`,
`muon_lora_16k_lr5e5`, `muonperhead_lora_16k_lr5e5`, `muon_lora_16k_UNSTABLE`,
`lora_fullrecomp_DIVERGED`, `probe_lora_4096`, plus `probe_lora_16384` at EP=4). **LoRA
adapters are structurally clean — 0 expert bytes — and therefore pass every check while
sitting on top of a frozen base that was 87.5% wrong.** This includes the entire
muon-vs-AdamW LoRA comparison. Whether any of those runs fed a reported number is
**[U]** human archaeology; the artefacts cannot say.

**F4 — Topology is not an abstraction: the floor is real but accidental, the ceiling is
unvalidated, and the from-scratch end has no data plane.**

The first pass claimed there was no small-scale path. **That was wrong.** **[V]** Six
launchers default `TRAIN_GPUS=1` and emit `torchrun --nproc_per_node=1
--master_addr=127.0.0.1`; 14 job logs (`e4b_arm_1458..1473`) print
`train GPUs=1(DP=1) TP=1 EP=1`, two reaching step 300. DP=1 is the *default*, not a
fallback, because co-located multi-GPU NCCL hangs — **[U]** a workaround never diagnosed.
**[V]** A non-Slurm path also exists and is the *current production path*: 31
`offslurm_*` scripts run via ssh + `enroot start --rw` + bare torchrun.

What is actually true is worse in a more interesting way:

- **[V]** Both lower-end stories are bolt-ons. The off-Slurm command is *string-grepped
  out of `DRYRUN=1` output*, and exactly **1 of ~240 launchers** supports `DRYRUN`.
  Topology is smeared across `#SBATCH` directives, an internal `TRAIN_GPUS` that
  contradicts them, and that grep — which is precisely why the single-GPU path was
  invisible to a directive-based analysis.
- **[M][V]** The ceiling is **8 nodes / 32 GPUs / EP=32**. The only >8-node scripts in
  the tree are vendored NVIDIA examples for GLM-4.5V and Qwen3.5-VL with **no Omni
  logs behind them**. The stated 100+-node target is unmeasured by three orders of
  magnitude.
- **[V]** Everything is Megatron: 748 of 2,888 non-vendored `.py` import it; zero TRL,
  zero HF `Trainer`, zero standalone-PEFT training entrypoints. The low end is "Megatron
  degenerated to one rank", not a lightweight stack.
- **[V]** The two lower-end stories **do not compose**. The random-init path has only run
  at 4 GPUs on mock data; the single-GPU path has only run from a pretrained checkpoint.
  **There is no proven 1-GPU from-scratch path.**

**F5 — Megatron-Bridge is forked, not depended on.** Three vendored trees coexist
(`Megatron-Bridge`, `_vendor_bridge_expertfix`, `_vendor_bridge_g4fix`), selected by
`PYTHONPATH` ordering and validated by *grep preflights that check the patches are
still applied*. **[V]** The live training path is `$ACCEL/Megatron-Bridge`, which has the
expert fix merged in; the `_vendor_bridge_expertfix` tree is used only by one conversion
script. This is the worst of both worlds: all the maintenance cost of a fork, none of the
upgrade safety of a dependency — and no way to tell from a run which tree it used.

**F6 — Disaggregation is manual and idle-costly.** RL is a Slurm training job plus
vLLM rollout/judge fleets stood up by hand in tmux/enroot outside Slurm's control. The
SDPO README concedes: *"During the rollout/judge phase the training GPUs idle… this
alternation is the cost of the disaggregated design."*

**F7 — Names are not identity, and the docs actively lie.** **[V]** `sdpo_gemma4` denotes
two *different* components: `accel/sdpo_gemma4` is gspo-family reward with the **unfixed**
broadcast, `mb/sdpo_gemma4` is grader-primary reward with the **fixed** broadcast. **[V]**
"FACTS" denotes two unrelated things: a reward variant (`sdpo_facts`, a gold-miss policy)
and the POLAR_A anchored co-teaching *loss*. **[A]** Superseded design docs are retired by
prepending a banner rather than by deletion — `EXPORT_STATUS.md` (2026-08-03) and
`EXPERT_SAVE_BUG.md` both describe long-fixed states and **both produced false findings in
this very review**. A doc that is wrong and present is a defect, not documentation.

---

## 1. Design principles, in priority order

| # | Principle | Answers |
|---|---|---|
| **P1** | Experiment isolation without code copying — composition, registries, immutable resolved config | F1 |
| **P2** | Every boundary carries an **executable contract**, and at least one contract per boundary must be *semantic*, not structural | F3 |
| **P3** | Provenance by construction — a run emits resolved config + code/data hashes + seed + topology + objective version | F2, F1 |
| **P4** | One declarative topology object; one interface, multiple execution backends and multiple launch backends | F4 |
| **P5** | Stages are first-class and declarative — a new pipeline stage is a config, not a directory | F1, F4 |
| **P6** | Services, not trays — rollout/judge/export are addressable, health-checked, weight-versioned | F6 |
| **P7** | Do not reinvent the kernel layer — Megatron-Core keeps TP/EP/CP/dist-opt/DCP | F5 |
| **P8** | Identity is a repo-qualified path plus a content hash, never a directory name | F7 |

---

## 2. Challenging the proposed layering

The brief proposed fifteen flat components: Model, Dataset, Data Pipeline, Trainer,
Distributed Strategy, Parallelism, Optimizer, Scheduler, Checkpoint, Evaluation,
Logger, Launcher, Cluster, Configuration, Runtime.

That is a sound *inventory* but a weak *architecture*: it is flat, it splits things that
must move together, it elevates things that are merely configuration, and — most
importantly — it has no home for the components whose absence actually caused the
failures above. Specific critique:

**Merge — "Distributed Strategy" + "Parallelism" + "Runtime" → one Execution plane.**
These are one concern: how a step is executed across devices. Splitting them is exactly
the conceptual blur that put Data Parallelism under a heading called "Model Parallelism
Techniques" in the project's own hero diagram.

**Demote — Optimizer, Scheduler.** These are Trainer configuration, not architectural
planes. Giving them top-level boxes adds ceremony without creating a decision point.

**Split out — Tokenizer / Template / Masking** from Dataset. This single surface caused
the highest-severity silent bugs in the record (CoT stripped → zero supervised tokens;
think-mode prefix misalignment at char 12; vision-token count mismatch; loss mask built
by *string search over rendered text*). It earns first-class, contract-bearing status.

**Rename — Logger → Telemetry & Run Registry.** Logging was never the problem. *Identity*
was: joining W&B ↔ checkpoint dir ↔ scalar JSONL ↔ manifest is a manual, regex-scraping,
JobID-keyed ritual, and `build_manifest.py` warns in its own header that some runs are
silently non-comparable.

**Elevate — Launcher/Cluster → a declarative Topology object, not a plane.** F4 shows the
defect is not that launching lacks a box; it is that topology has *three contradictory
sources of truth*. One object must emit Slurm, off-Slurm-enroot and bare-local from the
same declaration.

**Add — six planes the inventory is missing**, each justified by a measured failure:

| Added component | Justified by |
|---|---|
| **Objective/Algorithm** (loss · reward · teacher as plugins) | the crown jewels are also the most-cloned artefacts (F1) |
| **Reward & Verifier** contract, versioned | 4 gold policies across 25 files, selected by CWD (F1) |
| **Weight Bridge** (HF ↔ Megatron mapping) | dropped 7/8 of experts; produced structurally-perfect garbage (F3) |
| **Inference / Service plane** | RL needs rollout+judge fleets with weight-age tracking (F6) |
| **Stage / Pipeline orchestration** | A0–A4 and B1–B4 must compose declaratively (F4) |
| **Contracts & Gates** (crosscutting) | F3: the dominant failure mode has no home in the original list |

### The proposed layering — 7 planes + 2 crosscutting concerns

```
L6  CONTROL        typed config · run registry · provenance · telemetry · topology & launch backends · CLI
L5  ORCHESTRATION  Stage graph (A0-A4, B1-B4) · services (rollout/judge/export) · checkpoint · eval gates
L4  OBJECTIVE      loss plugins (CE/DPO/SDPO/POLAR/GSPO/FACTS/distill) · reward stack · teacher providers
L3  DATA           source -> sample -> render/template/mask -> sampler -> collate   (text + multimodal, one contract)
L2  MODEL          architecture registry · provider/config · freeze & adapter policy · weight bridge
L1  EXECUTION      backends: single-device | FSDP2/DTensor | Megatron-Core · precision · memory
L0  SUBSTRATE      torch · CUDA/NCCL · container · cluster

crosscutting:  CONTRACTS (executable invariants at every boundary; >=1 semantic per boundary)
               PROVENANCE (everything hashed, resolved and recorded)
```

---

## 3. Decisions with teeth

**D1 — Extend Megatron-Core; un-fork Megatron-Bridge.** **[M]** 97 first-party files
import Bridge, 62 import Megatron; TP/EP/CP, the distributed optimizer and DCP work
today at 30B-A3B MoE on GB200. Reimplementing that is not defensible. But replace the
three vendored patched trees with a *patch-as-plugin* mechanism against a pinned
upstream, plus a test that fails when an extension point moves, plus a run-record field
naming the exact tree and commit used. Rationale: the fork is what makes every upgrade
terrifying, and grep-preflights are not a dependency strategy.

**D2 — Adopt FSDP2/DTensor as a *developer-ergonomics* backend, on evidence, not as the
lower-end rescue.** The first pass justified this as "the honest answer to 1 GPU"; the
verification pass refuted the premise — **[V]** a Megatron TP=1/EP=1/DP=1 single-GPU path
exists and has run to step 300. So the correct justification is narrower and must be
earned: FSDP2 is worth adding only if it measurably improves iteration speed on a
laptop/single-node (no DCP resharding, no EP bookkeeping, plain `state_dict`) or unblocks
non-NVIDIA/CI environments. **Decision: build the Execution-plane *interface* now with
single-device and Megatron-Core behind it; gate the FSDP2 backend behind a Phase-4
measurement.** **[M]** zero current FSDP imports, so nothing is at risk in deferring it.
The conformance test (matching loss curves on a tiny model across backends) is required
regardless — it is what makes the interface meaningful.

**D3 — Do not adopt NeMo, NeMo-RL, DeepSpeed, or Ray.** **[M]** zero first-party imports
of any of them ⇒ zero migration debt in declining. Each would add a second opinionated
stack over the one that already works. Revisit Ray *only* if the rollout fleet needs
dynamic autoscaling. Corollary: the project's hero diagram currently advertises "NeMo
Framework Libraries & Collections" — that legend contradicts the code and should be
corrected.

**D4 — Typed, composable config with a resolved artefact; reject Hydra.** The need here
is *validation and provenance*, not sweep syntax. Hydra's interpolation and struct-mode
make "what actually ran" harder to answer — the exact failure mode F2. Use typed
(Pydantic-style) config objects, explicit overlay composition, and require every run to
write `resolved.yaml` plus content hashes.

**D5 — Ban environment variables as behaviour switches.** **[M]** 166 distinct env vars.
**[V]** `OMNI_GOLD_MISS_IS_BAD` and `OMNI_GOLD_GRADER` are read at *import time*
and change **reward semantics** invisibly, without appearing in any config diff — and
because the read is at module scope, **setting the flag after import silently no-ops**,
which is a worse failure than not having the flag. **[V]** A dead duplicate
`_GOLD_MISS_IS_BAD` at `mb/sdpo_gemma4:787` shows the copy-paste-patch workflow already
generating unreachable code. All behaviour moves into the resolved config; environment is
for secrets and cluster paths only; enforced by a lint gate in CI.

**D6 — Reward is a single installed, versioned package — and "certified" must be earned,
not asserted.** Ship one `omni.reward` imported by module path, never picked up from
CWD. Gold policy becomes an explicit config enum
(`short_circuit | post_judge_floor | miss_is_bad | grader_primary`), not four source
files. Every run stamps the reward package version + md5 into its run record, so a W&B
curve is traceable to the objective that produced it. **[U] Retraction:** the earlier
draft called the deterministic grader "certified". **No kappa or agreement artefact for
it exists in-source.** Either produce that artefact or strike the adjective — including
from the paper. Deduplication alone would not have prevented the four-way divergence,
because each semantic was intentional at the time; only a config enum plus a recorded
version does.

Scope of the consolidation, measured: **[M]** `rule_checks.py` — the verifiable-reward
substrate for A3's rule-based branch — is the *most* forked file in the tree, 5 variants
across 24 copies (379/318/387/397/265 LOC). `omni_sdpo_reward.py` is 2 variants over
21 copies, `m1m4_judge.py` 2 over 21. Adopting D6 therefore means resolving three
independent divergence sets, not deleting duplicates; budget it as a semantics review
with sign-off, and note that the 5-way split of `rule_checks.py` makes "the rule reward"
an undefined term in any existing result until the variant used is recovered per run.

**D7 — Template/mask parity is a launch gate.** Render a fixed probe set through both the
trainer and the serving path; assert token-id equality and non-empty supervision; fail
the launch otherwise. Replaces a manual md5 fingerprint, post-hoc log greps, and a
`check_loss_mask.py` that was written and never run.

**D8 — Three checkpoint/export invariants, one of them semantic.** F3 shows structural
checks are necessary and demonstrably insufficient, so specify all three levels:

1. **Build-time (structural).** Refuse to construct a model containing any
   parallel-aware submodule (grouped experts, TP linears) whose nearest ancestor is not
   a `MegatronModule`. **[V]** The entire expert bug was one wrong base class —
   `Gemma4DenseMoE(torch.nn.Module)` — which silently downgraded `sharded_state_dict` to
   a plain-torch flatten. This is a one-line assertion that would have fired at import.
2. **Save-time (structural).** On the *first* checkpoint of every run — not at export —
   assert expected parameter bytes per EP rank, assert EP-reshardability, and assert zero
   locally-indexed expert keys. **[A]** This is exactly the check that eventually caught
   the bug (45.70 GB vs 5.73 GB, 0 vs 960 indexed keys); it just ran two runs too late.
3. **Post-export (semantic) — new, and non-negotiable.** A logit-parity / generation probe
   against the Megatron source before any artefact is promotable. **[A]** Byte-completeness
   cannot detect a permuted expert axis, and `BAD_INCOMPLETE_..._1723` proves a
   complete-looking index can sit over 11.5% of the bytes. **[U]** No existing export has had this
   run. The weight-level retrofit is **done** (§5: 200/205 at 0 DIFFER, no GPU required —
   DCP chunks are byte-range-readable through the `.metadata` offset table, and that
   technique belongs in `omni.checkpoint` as a supported API). The semantic retrofit
   remains a Phase-1 task and is cheap: one GPU, ~20 minutes, `top-1 agreement == 1.0` and
   `KL < 1e-3` on a fixed 4×512 batch under the bridge revision that produced the export.
   **Specify it to assert positive work.** The weight sweep initially passed a corrupt
   artefact because the tensors it meant to compare were absent and `all([])` is `True`;
   a gate that can pass vacuously is not a gate. Emit the compared-element count, and make
   a zero count its own failure verdict.

**D9 — Stages are declarative, with a freeze policy field.** **[V]** the freeze primitives
already exist: `freeze_vision_model`, `freeze_language_model`, `freeze_vision_projection`,
`freeze_sound_encoder`, `freeze_sound_projection` are provider fields in
`Megatron-Bridge/.../nemotron_omni/nemotron_omni_provider.py`, referenced first-party in
`sdpo_gemma4/run_sdpo.py` and `run_gspo.py`, and present in real checkpoint
`run_config.yaml` files. So **VLM Stage 1 (projector init) costs a stage config plus a
caption data adapter — not a new subsystem.** This corrects the round-1 reading, which
saw 18 of 481 candidate files and concluded the machinery was absent.

**D10 — Services plane with an explicit colocation policy.** Make
colocated-vs-disaggregated a *policy flag* rather than an architectural given, so the
idle-GPU cost during rollout/judge (F6) becomes a tunable with a measured number
attached, not a permanent tax. **[V]** `smoke_refit_e4b.sh` already demonstrates a
single-tray colocated topology (GPU0 train / GPU1 rollout / GPU2 export) worth
generalising.

*The policy needs a second axis: request shape, not just placement.* **[M]** Measured on
this cluster's K3 endpoint (8 vLLM engines) while running this very analysis: 16
concurrent requests at ~250K-token context decoded at **10 tok/s aggregate**
(0.63 tok/s per sequence); 24 concurrent requests at ~37K-token context decoded at
**270–438 tok/s aggregate** — a **27–44× swing** driven purely by per-request context
length at comparable concurrency. Prefill was complete in both cases, so this is KV-cache
pressure in the decode loop, not admission queueing (`num_requests_waiting` was 0
throughout; the server never signalled distress).

Two consequences for the RL plane, where long-context rollouts at high concurrency are
the *normal* regime, not an edge case. First, rollout throughput is a **cliff, not a
slope**: a scheduler that admits by request count is choosing a point on that cliff
blindly, so the services plane must budget by *aggregate KV tokens* (concurrency ×
context) and expose that budget as config. Second — and this is the part that generalises
beyond serving — **the failure was silent and inverted**: every structural signal read
healthy (requests running, none waiting, no preemption counter, no error) while the job
was on track to deliver nothing before its timeout. It took an explicit throughput
measurement to see it. That is F3's thesis appearing in a completely different subsystem,
which is the strongest evidence yet that the missing semantic gate is a systemic property
of this stack and not an artefact of checkpointing.

**D11 — An experiment is a config delta plus a registered plugin, over immutable run
dirs.** This is the direct replacement for clone-per-experiment: it *preserves* the
benefit (the delta is a cleaner A/B than a directory diff ever was) while removing the
cost. Without this, P1 fails and the framework gets cloned around.

**D12 — Build the pretraining data plane; reuse the pretraining engine.** **[V]**
`accel/train_resume_test_e4b.py` is an existence proof that first-party code can drive
`megatron.bridge.training.pretrain` with a random-init `Gemma4E4BModelProvider()`,
`checkpoint.load=None` and `MockGPTDataset` — it ran 12 steps at 7.52B params (Slurm job
J01). So **do not rebuild a pretraining trainer.** What is genuinely absent is the data
plane: **[V]** 0 of 1,344 first-party `.py` touch `GPTDataset` /
`BlendedMegatronDatasetBuilder` / mmap corpora, 0 of 532 first-party `.sh` reference
`.bin` / `.idx` / `--data-path` / `blend`, and there is no tokenizer training anywhere.
Every first-party dataset path is jsonl chat data. Corollary guard: **mock-data smoke runs
produce checkpoints indistinguishable on disk from real ones** — a capability inventory
must never count scaffolding as capability.

**D13 — Build teacher routing on the judge registry; do not build fusion.** **[V]** Model
and expert *fusion* is categorically absent — zero hits for slerp / TIES / task-arithmetic
/ model-soup / DARE / mergekit / weight-averaging across 9,905 files (every `*merge*`
artefact is LoRA-into-its-own-base). A *pool* of frozen teachers is absent: `--anchor_ckpt`
is a single optional string. But two thirds of the substrate exists: `judge_registry.py`
is already a family→pool router with glob matching and heterogeneous multi-model pools,
and `polar_a_loss.py` already implements the hard part — union-top-K index sets, log-space
mixture, per-token adaptive λ with a student-ahead guard, EMA gap tracking. **Extend both
rather than rewriting.** Two concrete defects to fix in the move: `judge_pool.py`
dispatches round-robin (load balancing, which is *actively wrong* for "matching teacher"
semantics), and `run_odpo.py`'s teacher path collapses plural `--teacher_endpoints` to
`models[0]`. Note there is a **third** teacher mechanism the first pass missed: an HTTP
frozen teacher in `sdpo_odpo` that *authors* the `chosen` completion for online DPO.

*Caveat on "extend both", and it is not a small one:* **[M]** the substrate D13 leans on
is itself forked. `judge_registry.py` exists in 2 divergent variants across 23 copies
(13×@126L, 10×@132L) and `judge_pool.py` in 2 across 22 (12×@108L, 10×@138L). So D13's
first task is not extension but **adjudication** — pick the canonical variant of each by
reading the diff, not by copy count, and record why. Until that is done, "fix the
round-robin dispatch" is ambiguous: there are two `judge_pool.py`s to fix, and the tree
does not record which one any given run loaded.

**D14 — One RL driver, with rollout scope and reward scope derived from one object.**
**[V]** The DP>1 reward-broadcast defect is fixed in the driver that actually runs
(`mb/sdpo_gemma4/run_gspo.py`, `src=tp_src, group=tp_group`, "BUG FIX (2026-08-12)"), but
13 sibling drivers still carry `src=0`, and `accel/gspo/run_gspo.py` is dead code kept
alive by a 1-node launcher. The architectural defect is not the src rank; it is that ~27
near-identical RL drivers were produced by directory-copy. Obtain a single
`RolloutScope(group, src)` from `mpu` once and hand it to *both* the rollout client and
the reward broadcast; assert at startup that they are object-identical; add a DP>1 smoke
test that fails if two DP groups receive identical reward vectors for different
completions. **Delete `accel/gspo` rather than fixing it.**

---

## 4. What this buys, stated as falsifiable claims

1. A correctness bug fixed once is fixed everywhere — because there is one copy.
2. Any run can be re-executed from its recorded manifest alone, and the objective it
   optimised is recoverable from that manifest.
3. The failure modes in F3 become launch-time or save-time failures, not post-mortem
   discoveries — including at least one *semantic* check per artefact class.
4. A new pipeline stage (e.g. B1 projector init) is a config file, not a directory.
5. The same training code runs on 1 GPU and on 32+ GPUs from one topology declaration,
   verified by a conformance test — and emits Slurm, off-Slurm and local launches from
   that same declaration.

---

## 4a. Stage-execution evidence: the VLM ladder B1–B4

The target diagram draws B1→B4 as four equal boxes. **[V]** The evidence does not
support that symmetry, and the asymmetry should drive the roadmap:

| Stage | Code | **Ever executed?** | The single blocking gap |
|---|---|---|---|
| **B1** Projector init | primitives yes, recipe no | **never** — 0 logs, 0 result dirs, 0 checkpoints | every Gemma-4 recipe pins the *exact inverse* freeze triple |
| **B2** VL pre-training | model yes, data no | **never** | no interleaved loader, no corpus, and no document-level loss anywhere |
| **B3** Visual instruction tuning | production | **yes, at scale** — ~24 multimodal jobs, `iter_0002400` | none blocking |
| **B4** RL | end-to-end | **smoke only** — 5 iters, 2026-06-24, **checkpoint dir empty** | reward is an HTTP LLM-judge, computed on rank 0 → DP=1 only |

**This is the most important structural fact about the VLM side: only B3 has ever run.**
B1 and B2 are scaffolding that has never executed once, and B4 started five iterations
and saved nothing. Any roadmap that treats the four stages as "mostly built, needs
integration" is mis-scoped. Three of the four are unproven.

Four consequences that change decisions, each **[V]**:

**(i) B1 needs no new code — it needs a recipe and a corpus.** `run_recipe.py` applies
dotted overrides *after* the recipe builder, and the provider already reads
`freeze_language_model` / `freeze_vision_model` / `freeze_vision_projection`
(`gemma4_vl_provider.py:36-38`, applied at `:42-47`). So B1 is expressible today as
`model.freeze_language_model=True model.freeze_vision_projection=False`. This is direct
confirmation of §4 claim 4 — *the stage really is a config file* — and it is the
strongest single piece of evidence that the layering proposed in §2 is achievable rather
than aspirational. What is genuinely missing is a caption corpus: the documented
`Taiwan-formosa-VLM-caption-V1/data/` directory is **empty, 0 files**, while
`Formosa-Vision/data/` holds 23 parquet shards that **nothing in either repo
references**.

**(ii) Two unreported defects sit directly on the B1 path**, and both fail in the mode
this whole document is about — silently, or at a distance from the cause:
- `VLMLoRA` hardcodes `model.vision_model` / `model.vision_projection` (`MB/peft/lora.py:203-208`),
  but `Gemma4VLModel` exposes `vision_tower` / `multi_modal_projector`. **It will
  `AttributeError` on Gemma-4 as written.** Loud, at least — but it means the one PEFT
  class designed for staged VLM freezing has never been run against this model family.
- `_peft_common_vlm` ships plain `LoRA`, not `VLMLoRA` (`MB/recipes/common.py:544`), and
  hardcodes the CORD-v2 receipt-OCR dataset (`:512`). A "B1 via PEFT" built on it trains
  LLM adapters, never touches the projector, and **reports success**. This one is silent.

**(iii) B2's blocker is the objective, not the plumbing.** Multi-image scatter and
per-image bidirectional masking exist and work; energon is real and already wired for
`qwen3_vl` and `nemotron_omni`. But *every* task encoder in the tree masks loss to
assistant spans only. Full-document next-token prediction over an interleaved stream —
B2's actual objective — has no code path at all. Compounding it, `_inject_missing_markers`
collapses unreferenced media to the front of the first user turn, firing on **23.42%** of
sampled rows, which destroys exactly the document ordering B2 exists to learn.

**(iv) B4 is architecturally mislabelled.** The diagram says "RL with Reward Model";
there is no reward model. Reward is an HTTP LLM-judge with a yes-rate threshold, and the
freeze policy is hardcoded at `run_sdpo.py:131-145` with no override channel — including
a comment reading *"text-only smoke: keep the vision tower + projection frozen"* attached
to flags that are set **unconditionally**, so the image run also trained with a frozen
projector. No `chosen`/`rejected` keys exist anywhere in either repo. And over-length
image microbatches are **skipped, not truncated**, dropping samples silently.

*Two findings here correct earlier claims and must not be re-asserted.* Gemma-4 VL
multimodal forward parity is **done and measured** — `g4_parity_mm_1643.out`, image
cosine 0.999661, video 0.998345 — so "P2 vision remaining" is refuted for this model.
And B4 *does* set freeze policy in code, contradicting an earlier "FREEZE: NONE" reading
that came from `run_sdpo.py` simply not being in that analysis pass's input. The second
is the more instructive error: an absence in a *sampled* corpus was reported as an
absence in the *codebase*. Deliverables A–D must never make that inference.

One gap deserves its own line because it is cheap and blocks everything above: **no
executed run records which modules were frozen.** The freeze flags appear in no config
dump in any log, so even for the ~24 B3 jobs that did run, the trainable partition is
recoverable only by reading recipe source and hoping the right copy was resolved — which
§0/F1 just established cannot be assumed. *Logging the effective trainable-parameter
partition is ten lines and is the highest-value single addition to the current system.*

---

## 4b. Stage-execution evidence: the LLM ladder A0–A4

Same audit applied to the advanced-LLM diagram, same asymmetry, **[V]** throughout:

| Stage | Code | **Ever executed?** |
|---|---|---|
| **A0** Pretrain from scratch | entry point + vendored data plane | **NO** — a 12-iteration *mock-data* smoke test, val PPL 4.3e5 (random init) |
| **A1** Continued pretraining | one launcher | **NO** — zero `cpt`/`pretrain` logs in either repo |
| **A2** SFT cold start | yes, richly | **YES** — but see the corpus split below |
| **A3** verifiable RL | yes | **YES** — most exercised stage in the tree; 10 official GSPO runs, one live during this audit |
| **A3** subjective RL | yes | **YES** — judge-decided rewards in every audit; ODPO job J02 trained 254 real steps |
| **A4** multi-teacher distillation | ~60% | **PARTIALLY** — single frozen teacher + per-token objective ran (4,169 logged steps); the *multi* part has never existed |

So across **both** ladders, 5 of 10 stages have never executed. The system is
substantially more scaffolding than its directory structure suggests, and the roadmap
must be sequenced against *executed* capability, not against files that exist.

**A0's blocker is one thing: there is no pretraining data plane.** No
`preprocess_data.py`, no `indexed_dataset`, no `.bin`/`.idx` corpus anywhere in the
9,905-file inventory, and `cfg.dataset.blend = None`. `PRETRAINING_PIPELINE.md` says so
itself — Stage-0 data is "the largest new build." Note the trap in the name:
`pretraining_data/` contains **SFT jsonl**, not a pretraining corpus. A1 is worse than
missing — its launcher sets `RECIPE=..._sft_config`, so it is *the SFT recipe wearing a
CPT hat*: masked instruction data, not raw-text packed continued pretraining.

**A2's asset is real and its gap is a routing accident.** There are 9,499 genuine
multi-turn agentic trajectories (275 MB) with 38,588 `<think>` turns, 25,307
`<tool_call>`, 28,882 `<tool_response>`, including tool-error recovery. But the Gemma-4
line — the line that feeds A3 and A4 — overrides the corpus via `OMNI_SFT_JSONLS` to
Taiwan-AIEC files, so **the Gemma-4 models have never seen the agentic data.** The
cold-start happened on the Nemotron-Omni line, which itself only reached 28% of its
schedule. Separately, a 152 MB converted tool corpus is orphaned purely by schema: it
speaks HF `messages`, the loader requires `conversations`, and it is referenced by
nothing.

**A4 is ~60% real, and the missing 40% is precisely the word "multi".** What exists and
runs: on-policy self-sampling, refit/weight-sync with staleness p99 tripwires, one
genuinely frozen co-resident teacher (`requires_grad=False`, verified on the training
mesh), and a sophisticated per-token clipped α-divergence objective with correct
stop-gradient discipline. What does not exist *at all*: N>1 teacher residency
(`--anchor_ckpt` is a single optional string), content-based per-rollout teacher routing
(the existing router keys on dataset-declared `task_type` and load-balances round-robin —
right shape, wrong axis, and it routes *graders*, not teachers), and any form of weight
fusion (no slerp/TIES/task-arithmetic/DARE anywhere). The honest framing for the roadmap:
A4 is not "build a distillation stage," it is "generalize a working 2-source log-space
mixture to K sources and solve teacher residency." **Memory, not code, is the wall** —
nine co-resident 26B teachers will not fit, so this needs a remote-logprob teacher
service. That is the one genuinely new engineering item in A4.

### The finding that should change what we do first

**[V] Most of the GSPO campaign ran with no trust region at all.** `run_gspo.py`'s
`--old_logp_source` defaults to `self`, meaning `pi_old = pi_theta.detach()` from the
*current* forward — so the importance ratio is identically 1.0, the clip band can never
bind, and there is no importance correction whatsoever. This is not an inference: the
flag's own help text states it, including the measured signature
`seq_ratio_mean=1.0 exactly, clip_fraction=0.0 exactly`. **7 of the 10 official arms ran
that way.**

This is the single most consequential finding in the audit, and it is a *fifth* instance
of the F3 thesis: the system was structurally healthy — jobs ran, losses logged, curves
looked plausible — while the algorithm being executed was not the algorithm named in the
experiment. FoundationScale must default `old_logp_source` to `frozen`, and P2's
"one semantic contract per boundary" must include *the objective asserting its own
identity at step 0*.

Three more silent-objective defects, all **[V]**, all in the same family:

- **The gold floor can promote a wrong answer.** For non-digit non-label golds the match
  degrades to a raw substring test; a hit then floors the judge score to the pass
  threshold. The in-file comment documents gold `答案：(B)(C)(D)` matching answer
  `答案：(B)(C)(D)(E)`, and a production audit row was observed where gold was `答案：25`,
  the model answered 15, and the reward returned **0.9412**.
- **Two of six live task types can never be rule-decided** — `fill_in_blank` and
  `short_answer` are hardcoded to always abstain, with the source comment `# BUG 3`.
- **The reward gates fail open.** If `rule_checks` fails to import, the degeneracy veto
  is *silently disabled*; a verifier exception counts as a pass. A fail-open gate on a
  reward path is a semantic gate that has been quietly converted into no gate.

**[V] `polar_a_loss.py` and `facts_loss.py` are byte-identical twins** differing only in
import source and metric prefix — so A4's core objective is already forked before anyone
has extended it. Unify before generalizing.

---

## 5. Retractions, and what remains unverified

Deliverables A–D must not repeat these. Named here so they cannot leak back in.

**Retracted — refuted by direct inspection:**

- ~~"The EP=8 expert fix was reverted and never landed; all Gemma4-MoE full-FT checkpoints
  are invalid."~~ **[V]** The fix landed 2026-08-05 (`Gemma4DenseMoE(MegatronModule)`,
  `gemma4_provider.py:445,457`) and was never reverted; the launcher hard-refuses a
  non-`-expertfix` base. Only pre-fix artefacts are affected — but see F3a, the affected
  set is *larger* than the bug reports say (9 runs, 7 of them LoRA).
- ~~"The DP>1 reward-broadcast bug is still live in the gspo fork."~~ **[V]** Fixed
  2026-08-12 in the driver every DP>1 launcher invokes. The residue is dead code.
- ~~"Megatron→HF export is broken end to end; only the LoRA path works."~~ **[A]** 19 of
  23 export directories are byte-complete against their own index; the 3 failures are
  pre-fix and already renamed `BAD_INCOMPLETE*`; two MoE exports were written 2026-08-19.
  The claim reproduced `EXPORT_STATUS.md` (2026-08-03), which is superseded.
- ~~"There is no single-GPU path and no non-Slurm path."~~ **[V]** Both exist (F4).

**Corrected, not retracted:**

- The gspo gold *floor* is to the family pass `threshold` (default 0.6), **not** to 1.0.
- Four gold *code paths* but only **three distinct default behaviours** —
  `OMNI_GOLD_MISS_IS_BAD` defaults off and no launcher exports it.
- "Absent from both repos" is wrong for pretraining: every pretraining-adjacent asset
  lives **only** in `omni-accel`, and the full machinery is *vendored* in both.
- Always repo-qualify `sdpo_gemma4` and disambiguate "FACTS" (F7).
- Do not quote a single "N forks" figure without defining the file set: 27 RL drivers,
  24 reward modules, ~20 directories are three different counts of three different things.

**Closed since the first draft — four items moved from [U] to [V] by direct
investigation. Each is recorded with what actually settled it:**

- **[U]3 → the grader agreement statistic *exists*, and my retraction of it was wrong.**
  `sdpo_gemma4/shadow_grade/` holds the harness, its 1,800-row reference set and its
  per-chunk checkpoints. It was **re-executed during this audit and reproduced
  κ = 0.9824 exactly** (99.13% agreement over 1,724 decided pairs, 95.8% coverage;
  judge-vs-oracle κ = 0.160 for contrast). *Why the first pass missed it is the
  embarrassing part and the useful part:* the report is written to **stdout only**, and
  `fs_inventory.json` — the file set the whole audit was built on — indexes
  `.py/.yaml/.json/.sh/.md/.toml/.sbatch/.yml` and therefore **excludes `.jsonl`**, which
  is what the entire evidence set is stored as. F7's rule ("never quote an N without
  defining the set") turned out to apply to *this document's own method*. Any future
  "X does not exist" claim must state the file set it searched, and that set must include
  data extensions.
  **"Certified" is still the wrong word, for a better reason than absence.** The oracle
  is Kimi-K3 prompted with *gold_extract's own rulebook* — same NFKC folding, same
  last-restatement rule, same token map — and both were tuned on the same 3.3k census.
  The two raters are not independent, and the 76 excluded rows are exactly the hardest
  strata (100% of `fill_in_blank` and `short_answer` abstain). Defensible wording:
  *a faithful, reproducible implementation of a shared rulebook*, not correctness against
  human labels. A human-grounded artefact is ~8 person-hours, and the harness crashes on
  a 3-hunk v1/v2 API drift that should be fixed first.
- **[U]4 → answered, and it is two defects, not one.** At TP>1 the fork hard-crashes at
  step 0 (`broadcast_object_list` with a hardcoded global `src=0` on a TP subgroup); at
  TP==1 it silently mistrains via the known reward-scope bug. **It is unreachable at DP>1
  from any launcher as shipped**, which *lowers* its severity — the deletion argument is
  "duplicate carrying two defects already fixed elsewhere," not "live hazard."
  That reachability claim was re-checked across the full inventory (5,632 `.sh`/`.py`
  files, both repos): 36 files mention `run_gspo.py`, and only **two** sit under
  `omni-accel` — `launch_omni_gspo.sh`, the single invoker, which hardcodes
  `--nodes=1 --ntasks-per-node=4` with `TP=${TP:-4}` (→ DP=1) and whose documented "full"
  mode is `TP=8` on 2 nodes (also DP=1). The other 34 drive the fixed production driver.
  *And the first attempt at this check was itself a false negative*: a recursive grep for
  the literal `omni-accel/gspo` returned nothing, because the launcher writes
  `$WORKSPACE/gspo`. A zero from a search that could not have matched — the same failure
  shape as the κ retraction and the "never served" claim below. Every absence assertion
  in A–D is required to name the positive control that proves its detector fires.
- **Run provenance is *better* than I claimed, and its one gap is the whole finding.**
  I wrote that reward semantics were unrecoverable. Wrong: **35 RL runs carry a full
  `results/<run>/repro/` bundle** — `MANIFEST.json` with job id, `git_commit` and the
  resolved `code_tree`; `launch_cmd.sh` with the complete argv; `uncommitted.patch`; and
  `code_snapshot/`, **a frozen copy of the reward module actually used**. That is a
  genuinely good provenance system, which is exactly what makes its blind spot decisive:
  **no `repro/` artefact records environment variables.** So the bundle pins the commit,
  the source bytes and every flag — and still cannot tell you which objective was
  optimised.
  The demonstration is the sharpest single artefact in the audit. **24 runs (jobs
  J03–J04) share identical source bytes, identical directory, identical git commit and
  identical argv — and 12 ran one objective while 12 ran another**, split only by
  `OMNI_GOLD_MISS_IS_BAD`, visible only in the judge-audit log. Job J03 is named
  `odpo_g4_31b_goldmiss`: **a deliberate ablation whose treatment variable was written
  down nowhere but a directory name.** This is F3 in its purest form and it is why
  "capture the resolved config, not the inputs to it" is a Phase-1 requirement rather
  than a nicety.
  Two corrections fall out. The variant census is **59 definers / 8 md5s**, not 23/5 —
  my count inherited `fs_inventory.json`'s extension filter (9,905 indexed vs **19,912**
  real `.py` files), the *same* filter that caused the κ retraction. The 36 extra files
  are all `repro/code_snapshot/` copies, so live-tree shadowing is unchanged at 23. And
  ODPO jobs J05/J02/J06 resolve to `sdpo_facts`, not `sdpo_gemma4` (policy verdict
  unchanged) — **the snapshot overrides my launcher inference**, which is the general
  rule: prefer the frozen bytes to any reconstruction from the launch script.
  The corroboration is the most vivid single artefact in the audit: the sibling pre-fix
  driver at DP=12 ran **1,876 steps with 472 steps of `grad_norm` exactly 0.000 while
  logging `reward/mean=0.794, success=1.00`.** Post-fix, that correlation disappears
  (89 zero-advantage steps, 0 zero-gradient). A clean before/after of a training run that
  was, for a quarter of its steps, not training at all — and said it was fine.
- **[U]5 → yes, on both counts.** The aliased runs fed a published optimizer comparison
  (`CHECKPOINT_INVENTORY.md`, `BENCHMARK_GUIDE.md`), a *methodological* conclusion in
  `MUON_RUNBOOK.md`, and a model decision: 10 merged HF exports declared ready to serve.
  Retraction hygiene is good (banner-prepend + a verdict doc marking every arm INVALID),
  but per F7 the wrong numbers remain present and greppable.
  **And the comparison did not merely become invalid — it inverted.** On the aliased base
  Muon beat AdamW (1.4288 vs 1.4847); on the fixed base **AdamW beats Muon** (0.7412 vs
  0.7828). Mechanistically consistent: the aliased base is a 16-expert model replicated
  8×, so Muon's edge was an edge on a crippled model. This is the sharpest possible
  argument for D8's semantic checkpoint gate — a structural bug silently reversed the
  conclusion of a controlled experiment.
- **[U]6 → recoverable after all.** Every RL launcher passes `--judge_audit_path`, and the
  per-sample `decided_by` vocabulary is a **unique fingerprint of the gold policy**
  (`gold_fastpath` → P1, `gold+judge` → P2, `gold_miss` → P3, `gold_rule_pass|fail` → P4).
  Histogramming the audits reconstructs the objective for essentially every logged run.
  Two consequences are worse than the original unknown, though:
  **(a) `gspo_official4` changed objective mid-run** — steps 0→890 under P2, then resumed
  at step 800 under P4, in one W&B run, one checkpoint dir, one audit file. Its curve is
  not a single-objective curve and must not be plotted as one.
  **(b) `OMNI_GOLD_MISS_IS_BAD` was active for ≥12 ODPO runs, and nothing in the repo
  sets it.** It was exported by hand in an interactive shell and propagated by sbatch
  `--export=ALL`, leaving **zero on-disk trace**. The objective for those runs is
  recoverable *only* from the audit logs. This is F2 in its purest form, and it is the
  strongest argument in the document for P3: environment is part of the objective, so a
  run record that does not capture the environment does not identify the run.

- **The 8-node ceiling is now an execution fact, and it is a *choice*.** Slurm accounting
  (available all along, but hidden — the binaries are off `PATH` and `sacct` needs an
  explicit `SLURM_CONF`) gives all 1,774 Omni jobs: node histogram
  `{1:1242, 2:299, 3:2, 4:134, 5:2, 6:2, 8:93}` and GPU histogram topping out at 32.
  **Nothing above 8 nodes / 32 GPUs has ever run.** Max observed degrees are TP=8, EP=32,
  ETP=1, **PP=1 — pipeline parallelism was never >1 in any log**, and CP peaked at 2. The
  off-Slurm path tops out *lower* (6 nodes / 24 GPUs), so it does not raise the ceiling;
  and 213 jobs allocated exactly 1 GPU, confirming F4's single-GPU floor is real.
  The decisive nuance: **other users on this same cluster have run 18 nodes / 72 GPUs**,
  and capacity is 23 nodes / 92 GPUs. So 8 nodes is ~35% of what was available — the
  ceiling is a property of this codebase, not the hardware. That is exactly the claim
  D-roadmap's scaling section needs, and it could not be made before.
  Also worth recording so nobody re-runs it: **W&B is a dead channel for topology.** The
  35,001 local run dirs are login-node backfill jobs; 238 sampled configs contained zero
  `world_size`/TP/PP/EP keys.
- **`exports/fullft_iter2400_1tray_hf` was never served, and could not have been — it is
  empty.** Not "missing an index": **0 files, 0 bytes.** It was created by job J07,
  which FAILED 2m10s later with `ncclInvalidUsage — Duplicate GPU detected: rank 3 and
  rank 7 both on CUDA device`: the "1-tray" trick packs an 8-rank EP=8 world onto a
  4-GPU tray. It died inside `read_run_config` → `broadcast_object_list`, **before a
  single weight was read**, so `save_hf_pretrained()` was never reached.
  *The method note matters more than the answer.* The investigator checked whether the
  detector could fire before trusting a zero — and it could not: known-good exports also
  score 0–1 log references, and one positive control failed outright, invalidating that
  search scope. The verdict therefore rests on the dispositive physical fact (zero bytes),
  not on absence of mentions. **This is the standard every "X does not exist" claim in
  A–D must meet**, and it is the same discipline whose absence produced the κ retraction
  above.

- **[U]1 and [U]2 are CLOSED at weight level — and they were closed on a login node with
  no GPU.** The blocker was assumed to be "you cannot read a 50 GB `torch_dist` checkpoint
  without loading it." That assumption was wrong: the DCP `.metadata` is plain pickle and
  its `storage_data` maps each FQN to `(relative_path, offset, length)`, so individual
  chunks are byte-range-readable as standalone `torch.save` blobs.

  **Correction, and it is one of ours [V·post].** This document originally said "peak RSS
  stayed in the low hundreds of MB," full stop. A ground-truth probe measured it properly
  and the unqualified form is wrong: **198 MB is the chunk-level figure** (metadata parse
  alone is 180 MB; 128 expert chunks of ~4 MB add 18 MB and take 1.9 s), but reading a
  full 262144x2816 bf16 embedding and calling `.float()` on it peaks at **11.5 GB**. Both
  numbers are real; only one was quoted, and the quoted one flattered the technique. This
  is precisely the failure the first method rule of this audit names — *an unqualified
  count is not a fact; every count names its file set* — committed in the sentence
  advertising how cheap our verification was. The technique is still free at chunk
  granularity, which is the granularity that matters; the API must simply not promise a
  memory profile it cannot keep for whole-tensor reads.

  **A property worth stating, because it is why this is safe rather than clever [V·post]:**
  each byte range is a *complete, self-describing* `torch.save` ZIP archive (magic
  `PK\x03\x04`). A wrong offset raises `UnpicklingError` and a truncated length raises
  `PytorchStreamReader ... failed finding central directory`. It cannot return plausible
  garbage. And it is not a hack: torch's own loader slices the same file the same way
  (`_create_file_view(file, sinfo.offset, sinfo.length)` then `torch.load(...,
  weights_only=True)`).

  **A question we had priced at "one GPU, schedule it" was answered in
  an afternoon for free** — worth remembering the next time a verification is deferred on
  a cost estimate nobody tested.
  **[U]1 — the expert fix is numerically correct, not merely structurally plausible.** On
  `g4moe26b_twaiec_BASE_fullft_v3/hf_iter_0002467`, **all 30 layers, 3,840 / 3,840 experts
  bitwise identical** to the Megatron source, with full N×N cosine crossmatch requiring
  `argmax == identity` per expert (min margin 0.804). The `-expertfix` base is bitwise
  EXACT against upstream HF `gemma-4-26b-it` across experts, router, `o_proj`, dense MLP,
  layernorms and embeddings; the alias signature `cos(expert i, expert i%16)` reads ≈0.13
  where the bug would force ≈1.0. Crucially the detector was proven able to fire: injected
  positive controls caught a random shuffle on **128/128** and the bug's own `i → i%16`
  alias map on **112/112**, and the pre-fix checkpoint still shows the local-name signature
  (`...linear_fc1.weight0..15`). Trained post-fix weights also moved off base (max |Δ|
  0.0011–0.161), so this is not a "nothing trained" false pass.
  **[U]2 — 200 of 205 export directories now carry weight-level verification, up from
  0 of 23.** Note the denominator was also wrong: there are **205** index-bearing dirs, not
  23. Results: Gemma4 same-step 3,885 checks → 3,861 EXACT, **0 DIFFER, 0 SHAPE_MISMATCH**
  (24 MISSING, all LoRA-only sources); Nemotron **1,200/1,200 EXACT**; 91 cross-step refits
  at permutation identity with 91/91 positive controls firing; 23 LoRA merges reconstructing
  `merged == round_bf16(base + 2·B·A)` with **max error exactly 0.0 across 480/480 modules**.
  A header-only integrity sweep over all 205 gives 202 OK / 3 BROKEN — and all three broken
  ones were already quarantined by filename. The 5 unverified dirs are each individually
  explained (one quarantined-broken, two HF-native with no Megatron source, one duplicate
  of another under a second name, and `results/odpo_lora_run1/hf_merged_iter_0000400`,
  whose Megatron ckpt holds 456 adapter keys and **zero base weights**, making it
  unverifiable from what exists on disk rather than merely unchecked).
  **The coverage figure itself was given a positive control, which is why it can be
  quoted.** The Nemotron/LoRA batch runner was interrupted before it exited cleanly —
  precisely the condition that turns "200 verified" into an unnoticed undercount. The
  denominator was therefore re-derived by enumerating all 205 directories independently
  of the runner's exit status, and each family reconciled against its own result files
  (Nemotron: 25 files / 1,200 checks / 75 layers permutation-tested; LoRA: 24 files
  covering 23 merged dirs at 480/480 modules and 288/288 experts bitwise equal to base,
  plus 2 adapter-less orphans checked against base). The one LoRA dir in the unverified
  five is unverifiable by construction, not by interruption. **A coverage number sourced
  from the job that produced the coverage is not evidence of coverage** — the same
  circularity as S19, checked this time before the number was believed.
  **The finding inside the finding — and it is the best single piece of evidence in this
  audit.** The verification probe *itself* shipped a vacuous-truth false negative: on a
  known-corrupt artefact it reported `expert_perm_all_identity: true` and 33/33 EXACT,
  because the expert tensors were **absent** and `all()` over an empty set is `True`. A
  negative control caught it. **A tool written specifically to detect silent success
  silently succeeded** — same bug class as F3, live, in the detector. This is why gates
  must assert *positive work* (coverage counters, explicit `VACUOUS_*` verdicts) rather
  than the absence of a mismatch, and it is why G1–G12 are specified that way.
  **Independent corroboration of the trust-region finding, from a different channel.**
  The `gspo_official*` refits move weights by roughly **one bf16 ULP over hundreds of
  steps** (official10 iter25→125: 6.1e-05; official4 iter100→800: 4.9e-04) — exactly what
  a run with no effective trust region and `clip_fraction = 1.0` would produce. Two
  unrelated measurements, one conclusion. (It also means an exact-step match on those
  particular exports is nearly uninformative, which the investigator handled with a
  step-discrimination control rather than by claiming the pass.) One more F7 instance
  fell out: the dirs named `gemma4_e4b_gspo_*` **contain 26B MoE artefacts, not E4B**.
  **What is still not closed, precisely stated.** Bitwise-equal weights plus correct
  expert assignment prove the *conversion* is faithful. They do **not** prove the exported
  model *runs* correctly — config/rope/dtype/attention-implementation mismatch, tokenizer
  drift and generation-config errors all survive a perfect weight comparison. Depth is
  also uneven: one export was checked exhaustively, the other 199 sampled at ~3 layers.
  So the semantic gate (D8.3 / Phase 5) is unchanged and is now the *only* thing standing
  between "the bytes are right" and "the model is right."

**[U] Still unverified — must be stated as unknown wherever it appears:**

1. **Semantic** correctness of the exports — that they *generate* correctly, not merely
   that they convert faithfully. Weight level is closed (above); this is not. One node,
   one GPU, ~20 minutes: one forward pass on an identical fixed token batch through the
   exported HF checkpoint and its Megatron source under the bridge revision that produced
   it, asserting top-1 agreement == 1.0 and KL < 1e-3. Run it once for the MoE family and
   once for Nemotron-Omni. This should become the export path's first semantic gate.
2. Full-depth coverage: 199 of the 200 verified exports were layer-sampled, so a defect
   confined to an unsampled layer would not have been caught. Cheap to close incrementally.
3. Root cause of the co-located multi-GPU NCCL hang that forces DP1 as the default.
   **This one stays [U] deliberately, and the investigation is still the most valuable of
   the eight** — because it converted an unexamined belief into a bounded, testable one.
   What is now [V]: the stall is at `initialize.py:640`, the first *lazy* NCCL collective
   (`init_process_group` is called with no `device_id=`, so the barrier is where
   `ncclCommInitRank` really runs); both occurrences produced **~24 minutes of total
   silence — no NCCL WARN, no watchdog abort, no traceback**; and **`NCCL_DEBUG` was never
   set in any run in either repo** (it exists only inside a comment). That single omission
   is why a defect that reshaped the framework's default topology was never diagnosed.
   What is now **refuted**: the explanation written into the launcher itself
   (`launch_sdpo_e4b_smoke.sh:61-70`, *"P2P/NVLink between GPU1↔GPU2 won't establish while
   GPU0 runs vLLM"*) — which also contradicts its own line 181. **Job J08 is a control
   the team did not know it had:** same node, same container, same nested `srun`+`torchrun`,
   *identical* `NCCL_IB_DISABLE=1` / `NCCL_NVLS_ENABLE=1` / `MASTER_ADDR=127.0.0.1`, and it
   ran **1500/1500 steps at DP=4**. That kills IB flags, NVLS, loopback master, nested
   torchrun, bad node, cross-socket NVLink and IMEX-domain absence in one artefact.
   Three factors remain **perfectly confounded** — co-resident vLLM, a
   `CUDA_VISIBLE_DEVICES=1,2` subset that excludes the occupied GPU, and `srun --overlap`
   — because the trainer excludes GPU0 *precisely because* vLLM is there. No single cause
   can be named above ~35% confidence, and naming one would be exactly the failure this
   document is about. **The decisive experiment is one short run** (no vLLM, `TRAIN_GPUS=1,2`,
   DP=2, 1 iteration, `NCCL_DEBUG=INFO` added to both the exports *and* the
   `srun --export=` allowlist). The allowlist point is independently corroborated:
   `mb/sdpo_gemma4/smoke_refit_e4b.sh:110` passes `NCCL_SOCKET_IFNAME`, `NCCL_IB_HCA` and
   `NCCL_NVLS_ENABLE` — and **not** `NCCL_DEBUG`. Also worth fixing regardless: the
   launcher's EXIT trap `kill -9`s **every compute process on every GPU on the node**,
   safe only by `--exclusive`.
   **One claim from that investigation is retracted, and it is instructive.** It reported
   that `smoke_refit_e4b.sh` "does not exist anywhere under `<CLUSTER_HOME>`" and that only two
   launchers co-locate vLLM with training. Both are false: the file is
   `mb/sdpo_gemma4/smoke_refit_e4b.sh` (7,586 bytes; line 94 is the cited `torchrun`; it
   runs GPU0 train / GPU1 rollout / GPU2 export under `srun --overlap`), making it a
   **third** co-location launcher — in the *other* repo, which a search scoped to
   `omni-accel` cannot see. The negative came from a `find` that stalled on the
   23 TB tree and was abandoned, then read as absence. This is the **third** false
   negative of exactly this shape in this audit (after the κ retraction and the
   `$WORKSPACE/gspo` grep; a **fourth** has since landed — S19, the export probe that
   passed a corrupt artefact because `all([])` is `True`), and the first one caught before
   it shipped — by applying the
   rule the other two produced. That is the strongest available argument that the rule is
   worth its cost, and it is also the argument for why "the detector could not have fired"
   must be a *review gate*, not a habit.
4. Whether Omni ever exceeded 8 nodes **on a different cluster**. The GB200
   accounting DB starts 2026-05-14 but no log in either repo predates it, so the window
   covers these repos completely. The H100 cluster is a separate DB that was not queried;
   if such a run exists it left no artefact here.
5. Reward semantics for the runs that have **neither** a `repro/` bundle **nor** a
   non-empty judge-audit log — the POLAR grid and the GSPO official runs predate the
   bundle. For those, both evidence channels are absent and the objective is genuinely
   unrecoverable. (For the 35 runs that *do* have a bundle, this is now closed — above.)
6. Whether `official4`'s reported metrics were taken before or after its mid-run
   objective switch. Missing artefact: a step-range annotation on anything quoting them.
7. Muon vs AdamW at **full-FT** and **per-head** on a corrected base. Unlike the LoRA arm,
   no clean-base run exists — this is the one open item that genuinely needs GPU time
   rather than more forensics. Related: the `MUON_RUNBOOK` LR-stability claim was argued,
   never measured, and is now *weaker* than when written, since the loss ranking inverted.

### Human decisions required before Phase 1

1. **Run the *semantic* export probe, or accept the exports as behaviourally unverified?**
   *(Rewritten — this decision got smaller and sharper while the document was being
   reviewed.)* It used to be the expensive one. The weight-level half settled itself for
   free: 200 of 205 export dirs verified at 0 DIFFER and the expert fix confirmed
   numerically correct, on a login node, because the assumption that you must load a 50 GB
   checkpoint to read one tensor out of it was simply false. What is left is one node, one
   GPU, ~20 minutes, run once for the MoE family and once for Nemotron-Omni. It is the only
   thing that settles item 1 above, and it is the only correctness question in this entire
   audit that no amount of further reading can answer — the model has to actually run.
2. **The muon-vs-AdamW LoRA arm needs no GPU time — adopt the reversed result.** Clean,
   matched, completed replacements already exist (jobs J09/J10). Discard the aliased
   pair and the `CHECKPOINT_INVENTORY` ranking. Still genuinely requiring GPU time:
   Muon-vs-AdamW at **full-FT** (no Muon full-FT v2 exists at all), `muonperhead` at 16K,
   and the `lora_recompA` LR-stability claim.
3. **Strike "certified", keep the artefact.** The number is real and reproducible; the
   claim it licenses is narrower than the word implies. Zero-cost first step: apply the
   3-hunk fix and re-run with stdout captured to a checked-in report, so the evidence
   stops living in a terminal scrollback.
   **One check nobody in this audit could run:** whether "certified" or `0.982` ever
   reached the paper. macOS TCC denies `<local-workspace>` to every process used here
   (`Operation not permitted` on `ls`), and no `.tex` mirror exists outside it, so this
   is unverified for an *access* reason, not an evidence reason — the distinction
   matters, because it is the one open item that a single command settles:
   `grep -rin 'certif\|kappa\|0\.982' <local-workspace>/<paper-dir>/`. Indirect transcript
   evidence suggests the claim never reached the paper; that is [K], not [V].
4. **Set `old_logp_source=frozen` as the framework default, and decide what to do with
   the 7 official GSPO arms that ran without a trust region.** They are not invalid runs,
   but they are not GSPO runs either, and nothing currently labels them.
