# Deliverable D — FoundationScale Development Roadmap

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


**Status:** implementation roadmap  
**Source basis:** Omni evidence audit, adversarial verification reports, artifact census of 2026-08-20, and the architecture decision spine  
**Planning assumption:** 5–7 engineers, divided into approximately two platform/runtime engineers, two ML-systems/engine owners, one data/infrastructure engineer, and one evaluation/release engineer. Estimates assume part-time availability of at least two people for production support.

---

## 0. Roadmap thesis and sequencing

FoundationScale should be delivered as a **progressive consolidation of the existing production system**, not as a rewrite followed by a migration. The evidence argues for that ordering:

- The most active repository, `omni-accel`, has **no `.git` directory at all** [M]. Any refactor before capture would make both rollback and attribution worse.
- Reward semantics differ across **four code paths and three default behaviours**, imported by bare name from the current working directory [V]. Before deduplication, every run must therefore record which objective it executed.
- **Phase 1's provenance work is smaller and sharper than it first appeared [V·post].** 35 RL runs already carry a `results/<run>/repro/` bundle — job id, `git_commit`, resolved `code_tree`, full argv, working-tree patch, and a frozen `code_snapshot/` of the reward module itself. That is most of a run manifest, already built. **The one thing no `repro/` artefact records is environment variables** — and that single omission is the whole gap: jobs J03–J04 are **24 runs with identical source bytes, directory, commit and argv that split 12/12 across two different objectives**, separable only by the judge-audit log. So Phase 1 should *extend the existing bundle* rather than introduce a parallel mechanism, and the acceptance test is environment capture, not manifest existence.
- The DP>1 reward-broadcast fix is present in 14 trees and absent from 13 others [V]. `accel/gspo` and `bridge/sdpo_gemma4` cannot be treated as interchangeable by name.
- The strongest validated ceiling is **8 nodes / 32 GPUs / EP=32** [M]. Nothing should be scheduled assuming “100-node readiness” until measured. **And that ceiling is a choice, not a hardware limit [V·post]:** Slurm accounting over all 1,774 Omni jobs tops out at 8 nodes / 32 GPUs with **PP never >1 in any log**, while *other users on the same cluster have run 18 nodes / 72 GPUs* against a capacity of 23 nodes / 92 GPUs. Phase 9 is therefore removing a software constraint, not waiting on procurement — which is what makes it schedulable at all. The off-Slurm path tops out *lower* (6 nodes / 24 GPUs), so it is not the escape hatch.
- A0’s data plane, B1’s operational recipe, and B2’s document-level objective are absent as executed capabilities [V].
- Megatron-Bridge is represented by three trees selected through `PYTHONPATH`; the live training tree is `$ACCEL/Megatron-Bridge`, not the fix-only vendor tree [V].
- Export correctness splits in two, and the halves now have different answers [V·post]. **Weight level: closed.** 200 of 205 export directories are verified tensor-by-tensor against their Megatron source — 0 DIFFER, 0 SHAPE_MISMATCH, no permuted expert axis, LoRA merges reconstructing at max error exactly 0.0 — done on CPU by byte-range-reading DCP chunks. **Semantic level: untouched.** Zero exports have ever been asked to generate a token [U]. Phase 1's export item therefore shrinks from a sweep to a single 20-minute forward-pass probe, and Phase 5 gains the assertion that makes it permanent.

The first phase is therefore deliberately forensic rather than architectural. Every subsequent phase ends in a working system with measurable improvement. No phase requires existing training jobs to stop or cut over to a new stack on the same day.

### Phase map

| Phase | Prime outcome | Estimated effort |
|---|---|---:|
| 1 | Version control, run identity, forensic baseline, binary gates | 4–5 engineer-weeks |
| 2 | Reward/objective SKU consolidation and semantic repair | 6–8 |
| 3 | One topology/launch source of truth | 6–8 |
| **Checkpoint** | Decide full migration scope from measured results | — |
| 4 | Single RL driver and rollout/reward scope contract | 8–10 |
| 5 | Checkpoint/export semantic certification | 5–7 |
| 6 | Data plane and template/mask contracts | 8–10 |
| 7 | Stage system: SFT declarative, B1 recipe, A0 prototype | 8–10 |
| 8 | First-class services plane and teacher routing | 9–12 |
| 9 | Scale validation and B2 from-scratch program | 9–12 |
| 10 | Deprecation, certification and production handoff | 6–8 |

The ranges are not additive linearly because platform, reward, and production engineers work in parallel. Calendared elapsed time is expected to be 9–12 months with the stated team.

---

## Phase 1 — Provenance and gates before refactor

**Objective.** Make every current and future run attributable, and turn the two highest-frequency post-mortem discoveries into automatic failures.

### Scope

1. Put `omni-accel` under version control and bring `omni-bridge` back to a clean or explicitly snapshotted state.
2. Add one immutable run manifest to all training, export, conversion and rollout jobs.
3. Retrofit objective provenance from judge audit logs.
4. Run the *semantic* export probe (weight-level parity is already closed at 200/205 [V·post]; what is missing is one forward pass per model family).
5. Land build-time and first-checkpoint assertions.
6. Audit the nine MoE runs trained from the aliased `gemma-4-26b-it` base.

### Concrete deliverables

- A commit-pinned mirror or imported repository for `omni-accel`, preserving the existing filesystem layout where practical. The current absence of any `.git` directory [M] means import must not demand perfect history.
- A reconciliation plan for the sibling repository’s **892 dirty files** [M], with a large-file checkpoint/data exclusion policy.
- `omni.run_manifest`, emitted at process start and again after config resolution, containing:
  - resolved config;
  - repository path and content hash;
  - import resolution for reward, dataset, collate, driver and reward-package modules;
  - reward module MD5 and policy enumeration;
  - topology (`nodes`, GPUs/node, TP/EP/ETP/CP/PP/DP);
  - environment capture for every whitelisted/allowed variable;
  - checkpoint/base-model path and content metadata;
  - launch command, scheduler type and JobID.
- A loader that resolves imports explicitly instead of relying on CWD. This is mandatory because `run_gspo.py` imports `omni_sdpo_reward` by bare name (`omni-bridge/sdpo_gemma4/run_gspo.py:40`) [V].
- An audit-log objective classifier using the `decided_by` vocabulary:
  - `gold_fastpath` → P1;
  - `gold+judge` → P2;
  - `gold_miss` → P3;
  - `gold_rule_pass|gold_rule_fail` → P4.
- A per-run objective report covering every RL run with an audit file. This recoverability route is verified; however, it identifies policy class, not the exact module hash [V].
  - The report must flag `gspo_official4`, which changed from P2 to P4 on restart/step 800 within one W&B run, checkpoint directory and audit file [V].
  - It must flag at least 12 ODPO runs for which `OMNI_GOLD_MISS_IS_BAD` was active despite no repository launcher setting it [V].
- A parity probe harness comparing Megatron and HF logits/generations on:
  - at least one dense-31B export;
  - at least one MoE-26B export;
  - one deliberately selected old `BAD_INCOMPLETE*` export as a negative control.
- Build-time assertion: refuse to construct a model with a parallel-aware submodule whose nearest ancestor is not a `MegatronModule`. This directly operationalizes the fix behind `Gemma4DenseMoE(MegatronModule)` at `gemma4_provider.py:445,457` [V].
- Save-time assertion on the **first checkpoint of every run**:
  - expected expert/parameter bytes;
  - zero locally indexed `weight<N>` expert keys;
  - expected expert tensor shape (`(128,1408,2816)` for Gemma4-MoE);
  - EP-reshardability metadata.
- Export-byte gate as a library function, replacing copy-pasted shell logic. It must be callable from `export_v2_fullft_sbatch.sh` and, critically, from `g4_sft/export/export_many.sh`, which currently passes `--not-strict` and verifies only with `du -sh` [V].
- Aliased-base inventory report covering all 20 MoE run configs. It must list the nine affected runs and distinguish full-FT checkpoints with broken expert-byte signatures from structurally clean LoRA artifacts [A].
- Deletion/retirement package for stale authoritative-sounding documents:
  - `exports/EXPORT_STATUS.md`;
  - `EXPERT_SAVE_BUG.md`;
  - any regenerated status page must be derived from artifacts or carry an expiry.
- A quarantine marker for `exports/fullft_iter2400_1tray_hf`, which lacks an `index.json` [A].

### Exit criteria

- 100% of first-party training and conversion code imported into or represented by version control.
- Both repositories have zero undisclosed uncommitted source deltas, or each remaining delta has a recorded snapshot and owner.
- Every new training/export/rollout job writes a complete manifest before model loading begins.
- For a sample of at least 10 new or resumed runs, a replay tool can reconstruct resolved config, topology and reward-policy class without opening logs.
- 100% of judge-audit-bearing runs are classified into P1–P4 or explicitly marked unrecoverable.
- The logit-parity report covers at least one representative export of each model family, with numerical pass/fail recorded — `top-1 agreement == 1.0` and `KL < 1e-3` on a fixed 4×512 batch, `use_cache=False`, under the bridge revision that produced the export. **The report must state how many tensors/logits were actually compared**; a pass with a zero comparison count is a failure, because that is precisely how the weight-level sweep first fooled itself.
- Both structural gates are unit-tested and demonstrated:
  - build gate catches a deliberately misbased subclass;
  - save gate catches the measured 5.73-GB expert-byte signature and accepts the measured 45.70-GB signature.
- The nine aliased-base runs have explicit disposition: invalidated, recoverable by equivalence argument, or requiring rerun.

### Ordering rationale

Phase 1 precedes refactoring because the measured failure mode is not “the code cannot train”; existing systems train and export at production scale. The failure is that **structure can look right while objective, identity or semantics are wrong**. Version control, manifests and gates preserve that distinction during all later moves.

It also avoids gambling on a full architecture rewrite before answering cheap decisive questions: parity on exports, the 9-run blast radius, and exact objective provenance.

### Dependencies

None, except access to existing checkpoints, exports, W&B records and GPU time for parity.

### Risks

- The repository import may surface undocumented live patches. Mitigation: import byte-for-byte first; lint later.
- Import-order changes can alter behavior. Manifest capture must precede any `PYTHONPATH` cleanup.
- Objective classification resolves policy family but not always a unique file bytes identity [U/V]. Runs must therefore be labelled by the evidence actually available, not over-attributed.
- Parity failures will require export scrubbing and possibly rerun decisions.

### Falsification condition

Refactor-first becomes preferable if Phase 1 finds that manifests and import capture cannot be attached non-invasively, or if parity/provenance repair is blocked by missing runtime data at a materially greater rate than the existing audit-log coverage suggests. Otherwise, deferring a rewrite remains justified.

### Keep-the-lights-on track

- Existing training jobs remain on their current launch script and directory.
- Additive manifest emission and assertions only.
- No required migration of in-flight runs.
- Any assertion that could kill a live production run is initially `WARN`; it becomes blocking only after one observed successful checkpoint cycle.

---

## Phase 2 — One versioned reward and objective package

**Objective.** Make the objective trained by RL a selected, versioned configuration value instead of a property of `[REPO]/[directory]/cwd`.

### Scope

The four observed gold policies:

1. `short_circuit` — gold match directly yields 1.0;
2. `post_judge_floor` — gold match floors a judge score to the task threshold, default 0.6;
3. `miss_is_bad` — opt-in gold miss returns 0.0;
4. `grader_primary` — deterministic grading before judge fallback.

Only three distinct default behaviours exist because `OMNI_GOLD_MISS_IS_BAD` defaults off and no launcher exports it [V].

### Deliverables

- One installed `omni.reward` package, imported by qualified module path.
- A reward `semver`, source content hash and API version.
- A single config enum:
  - `short_circuit`;
  - `post_judge_floor`;
  - `miss_is_bad`;
  - `grader_primary`.
- One canonical variant selection for each forked substrate:
  - `rule_checks.py` — five AST variants across 24 copies;
  - `omni_sdpo_reward.py` — two AST variants across 21 copies;
  - `m1m4_judge.py` — two variants across 21 copies [M].
- Promotion of `gold_extract.py` into the package.
- Removal of import-time environment reads for behavior. The existing flags silently no-op if set after import, and `OMNI_GOLD_MISS_IS_BAD` has at least one dead duplicate read at `mb/sdpo_gemma4` line 787 [V].
- Fail-closed gates:
  - a verifier exception cannot count as a pass;
  - failure to import `rule_checks` cannot silently disable degeneracy protection.
- A decomposition of `polar_a_loss.py` and `facts_loss.py`, which are byte-identical apart from import source and metric prefix [V].
- Certification discipline around grading:
  - keep the re-executed κ=0.9824 report as a rulebook-consistency artifact;
  - do not call it independent human certification, because the Kimi-K3 oracle used gold_extract’s own rulebook and the two raters are not independent [V].
- Rejection-test suite:
  - wrong-answer substring case (`gold=答案：25`, answer `15`) must not receive reward 0.9412;
  - `fill_in_blank` and `short_answer` must be treated according to a named policy instead of unconditional abstention caused by the existing `# BUG 3` path [V].

### Exit criteria

- Exactly one package exports reward selection.
- 100% of new runs record package version, reward enum and md5.
- Known historical policy differences are reproduced by explicit compatibility fixtures.
- `fill_in_blank`, `short_answer`, wrong-answer substring matching and verifier exceptions have regression tests.
- Existing unmodified runs show byte-identical N=1 behavior where byte identity was promised; deviations require an explicit compatibility mode.

### Dependencies

Phase 1 manifests and versioned source control.

### Risks

- The decision to reuse judge registries is complicated by forked code: `judge_registry.py` has two variants over 23 files and `judge_pool.py` two over 22 [M]. Phase 2 must adjudicate variants by diffs, not copy counts.
- The same directory name has different meanings across repositories; reviews must use repo-qualified paths everywhere.
- Label or task taxonomy changes can alter historical comparability.

### Falsification

If a semantic review proves the four policies depend on irreducibly different interfaces but tests cannot characterize them, Phase 2 splits: immediate release of identity/logging only, then policy consolidation. It does not default to another directory fork.

### Production safety

Legacy wrappers may remain for one release but resolve through the one package to the same object. Existing scripts do not need same-day rewrites.

---

## Phase 3 — Declarative topology and launch abstraction

**Objective.** Replace three contradictory sources of topology truth with one object that can render Slurm, off-Slurm enroot and bare local launches.

### Scope

Today’s topology is split across:

- `#SBATCH` directives;
- internal `TRAIN_GPUS`/`NTRAIN` variables that can contradict those directives;
- `DRYRUN=1` string extraction, supported by exactly one of approximately 240 launchers [V].

### Deliverables

- A typed `TopologySpec`:
  - nodes;
  - GPUs/node;
  - TP/TP context/EP/ETP/CP/PP;
  - DP derivation;
  - memory model;
  - process-group naming;
  - local vs multi-node vs off-Slurm execution backend.
- Renderers for:
  - Slurm;
  - off-Slurm `ssh + enroot + torchrun`;
  - bare-local/torchrun.
- A generated environment export list. Specific inconsistencies such as `<partition>` vs `<partition_alt>` partitions [M] become validation errors.
- Every launcher replaced by either:
  - a thin generated script; or
  - an explicit legacy exception.
- Preservation and formalization of:
  - six launchers’ `TRAIN_GPUS=1` single-GPU path [V];
  - 14 observed logs with `train GPUs=1(DP=1) TP=1 EP=1` [V];
  - the co-located `smoke_refit_e4b.sh` topology (GPU0 train / GPU1 rollout / GPU2 export) [V];
  - the 31 off-Slurm production scripts [V].
- A topology record in every Phase 1 run manifest.
- Resolution pass for the unresolved co-located multi-GPU NCCL hang currently motivating DP=1 as default [U]. Phase 3 does not require the hang fixed, but it does require a deterministic reproducer and a tracking issue.

### Exit criteria

- 100% of newly submitted jobs render from `TopologySpec`.
- The same smoke experiment can launch locally on one GPU, inside Slurm, and off-Slurm without source modification.
- No new generated launcher reads `TRAIN_GPUS` separately from the declared topology.
- A conformance run demonstrates the same loss curve (within a stated tolerance) on:
  - one GPU;
  - four GPUs on one node;
  - eight GPUs across two nodes.
- The largest existing topology — 8 nodes/32 GPUs/EP=32 — can be launched through the abstraction.

### Dependencies

Phase 1 for run recording; Phase 2 is not strictly required and may proceed in parallel after manifests stabilize.

### Risks

- Re-implementing the launch shell could lose battle-tested preflights. Mitigation: convert the strongest launcher gates into executable contract checks, not comments.
- Some launcher behavior is undocumented and contradictory. Any discrepancy is resolved by probing the actual command rather than rewriting prose.
- Multi-node DRYRUN or shell inspection remains brittle if kept. It is removed rather than emulated.

### Falsification

Adopt an external job framework only if the internal renderer cannot reproduce all three production launch modes with fewer special cases than today within this estimated effort. Do not bring in Ray by default: there are zero first-party imports and no migration debt [M].

---

## Decision checkpoint after Phase 3

This checkpoint decides whether FoundationScale proceeds as the main production path or remains a hardened wrapper.

### Review measurements

- Fraction of active runs emitting complete manifests.
- Fraction of new runs using one reward package.
- Difference between legacy and new-driver smoke losses.
- Number of launchers still hand-written.
- Frequency of `WARN`-to-`BLOCK` gate escalations.
- Measured wall-clock and throughput parity at 1, 4, 8 and 32 GPUs.

### Possible decisions

| Evidence | Decision |
|---|---|
| New launch path reproduces existing jobs and reduces special cases | Make FoundationScale default for new runs |
| Manifest or objective capture is incomplete | Fix control plane first |
| Legacy drivers are materially faster or numerically different | Keep dual-track; investigate before driver consolidation |
| Phase 1 found parity failure | Redirect next quarter to export/certification and rerun remediation before new features |

A declaration of framework success is not evidence of success. The checkpoint requires observed runs.

---

## Phase 4 — One RL driver and one scope object

**Objective.** Collapse the clone-family RL drivers into one driver, while preserving algorithm variants as registered plugins/config deltas.

### Scope and deliverables

- One RL driver capable of the validated behaviours:
  - GSPO;
  - SDPO;
  - POLAR;
  - ONLINE DPO;
  - FACTS/POLAR_A.
- One `RolloutScope(group, src)` object derived once from Megatron parallel state and consumed by:
  - rollout client;
  - scoring;
  - reward broadcast;
  - diagnostics.
- Start-up assertion that rollout and reward scopes are object-identical.
- A broadcast validator rejecting a global source rank not present in the passed group.
- DP>1 regression test that fails if two DP groups receive identical reward vectors despite distinct completions.
- Old-policy capture defaults to frozen, not the current `self` behavior. The measured signature of the existing bad default is sequence ratio exactly 1.0 and clip fraction exactly 0.0; 7 of 10 official arms ran that way [V]. Existing affected runs must be labelled.
- Compatibility plugins for existing behavior, but with visible naming: a no-trust-region run may not be labelled GSPO without qualification.
- Formal adoption of the already-fixed driver behavior from `omni-bridge/sdpo_gemma4/run_gspo.py:815,916` [V].
- Deletion or archival policy for `omni-accel/gspo/run_gspo.py:417`, which is unreachable at DP>1 from shipped launchers but carries both a TP>1 hard-crash and a TP==1 silent mistraining class [V]. Archive it as superseded; do not market it as corrected.

### Exit criteria

- One active driver source appears in all new runs.
- 100% of DP>1 runs pass scope validation at startup.
- Legacy, fixed and single-process smoke tests agree on reward routing.
- DP>1 synthetic test demonstrates independently computed rewards and independent gradients.
- New no-trust-region compatibility mode is recorded explicitly and not mislabeled.

### Dependencies

Phases 1–3. Phase 2’s reward enum is strongly preferred but not a hard blocker if the driver receives an injected reward-callable with version metadata.

### Risks

- There are 27 near-identical RL drivers and hidden behavior differences [V/M]. A compatibility matrix and a live-run inventory are mandatory before deletion.
- Byte-identical prior runs may depend on quirks that correctness fixes intentionally eliminate. Preserve experiment semantics only when scientifically intentional.

### Falsification

If the plugin interface cannot express the four most-used production algorithms without changing the driver, the interface is wrong and Phase 4 is redesigned around a smaller core algorithm registry rather than forcing variants through hooks.

---

## Phase 5 — Checkpoint and export certification

> **Scope correction after the weight-level sweep [V·post].** This phase was written
> assuming export correctness was wholly unverified. The *weight* half is now closed —
> 200 of 205 export directories compared tensor-by-tensor against their Megatron source,
> 0 DIFFER, 0 SHAPE_MISMATCH, no permuted expert axis, LoRA merges reconstructing at max
> error exactly 0.0 — so the phase does not need to *establish* export correctness, it
> needs to make it **continuous and semantic**. Two concrete consequences:
>
> 1. **The retroactive sweep becomes a library function, not a project.** The technique
>    that made it free — reading individual DCP chunks through the `.metadata`
>    `storage_data` offset table, `(relative_path, offset, length)` per FQN, **198 MB peak
>    RSS at chunk granularity [M]** — belongs in `omni.checkpoint` as a supported API.
>    It turns "verify every export" from a GPU request into a cron job. **Scope the promise
>    honestly:** 198 MB is the chunk-level figure, and a full-embedding read with a float32
>    cast peaks at 11.5 GB [M]. The API must expose block-wise comparison rather than
>    implying whole-tensor reads are cheap.
>
>    Two defects in the throwaway probe must not survive into the library [V·post]:
>    `read_full()` discards its coverage counter, so a region no chunk covered returns
>    **zeros that look like data** — the vacuous-truth bug again, in the reader this time;
>    and `cmp()`'s cosine/rms underflow in float32, with the reproduced symptom being a
>    tensor's cosine *with itself* returning **1.80**. Any parity API must accumulate in
>    float64 and ship a self-comparison guard.
>
>    One more metadata fact that breaks naive implementations: `state_dict_metadata` is not
>    a tensor list. A real 26B checkpoint has **8,970 keys of which only 928 are tensors**;
>    the other 8,042 are `_extra_state` byte blobs. A completeness gate that counts keys
>    rather than tensors passes trivially [M].
> 2. **The gate that remains is the one no byte comparison can reach.** One node, one
>    GPU, ~20 minutes: load the exported HF checkpoint and its Megatron source under the
>    bridge revision that produced it, run one forward pass on an identical fixed batch
>    (4×512, `use_cache=False`), and assert **top-1 agreement == 1.0** and **KL < 1e-3**
>    on next-token logits. This catches what weight parity provably cannot: `rope_theta`,
>    `final_logit_softcapping`, attention implementation, dtype policy, tokenizer drift,
>    generation config. Run it once per model family (MoE-26B, Nemotron-Omni) to close
>    the backlog, then on every export forever.
>
> **Both gates must report coverage, not just verdict.** The sweep that closed the weight
> question initially passed a corrupt artefact because the tensors it meant to compare
> were absent and `all([])` is `True` (S19). Every assertion in this phase emits the
> number of elements actually compared, and a zero count is a failure with its own
> verdict name — never silence, and never a pass.


**Objective.** Make artifact acceptance semantic, reproducible and centrally enforced.

### Deliverables

- A `verify_checkpoint()` service:
  - parameter-byte accounting;
  - module-base-class traversal;
  - EP resharding metadata check;
  - run-manifest binding.
- A `verify_export()` library invoked by the exporter itself, not by optional shell wrappers.
- Mandatory byte-vs-index and tensor-count checks.
- Mandatory semantic promotion gate:
  - logit parity;
  - a generation probe;
  - expert-order sanity checks for MoE.
- A narrowly-scoped architectural exception list instead of broad `strict=False`. The existing `OMNI_DROP_REDUNDANT_VPROJ=1` pattern is preferable to global non-strict export because it recognizes only Gemma4’s known K==V redundancy [V].
- Elimination or hard failure of the multi-rank `idx % num_savers` save path whenever the number of output files is less than the number of saver ranks.
- Retirement of the heuristic that two shards imply correctness. Two shards can simply be the base HF layout in a complete export [A].
- A periodic export census generated from disk rather than manually maintained status documents.

### Exit criteria

- 100% of new checkpoints receive first-checkpoint structural validation.
- 100% of new exports carry byte, tensor and parity verdicts.
- A deliberately incomplete model cannot return success with a valid-looking index.
- A deliberately permuted MoE expert axis is detected by the semantic probe unless its computed parity tolerance is scientifically justified.
- Historical exports are labelled:
  - byte-complete;
  - byte-invalid;
  - no-index/unlabeled;
  - semantically verified/unverified.

### Dependencies

Phase 1 parity harness and structural gates. Runs on all exported models may be batched on one GPU.

### Risks

- Byte sums cannot detect wrong expert permutations [A]. This is why semantic parity is non-negotiable.
- Export repair can force old base models to be re-converted because the pre-fix layout is not safely resumable; the existing launcher already hard-refuses non-`-expertfix` bases at `launch_g4moe26b_sft_32k.sh:164-167` [V].

### Falsification

If a single-GPU EP=1 export is no longer possible for new checkpoint formats, Phase 5 cannot require that format alone. The save format must then guarantee mesh-independent resharding before export; otherwise the roadmap pauses scale claims.

---

## Phase 6 — Token/template/mask and data contracts

**Objective.** Move the most dangerous correctness surface — rendering and supervision masks — behind executable contracts.

### Deliverables

- A first-class tokenizer/template/masking module:
  - canonical chat-template resolution;
  - answer-span tracking rather than string masking;
  - explicit stop-token policy;
  - modality-token accounting;
  - loss-mask construction with token identity, not text search.
- Template parity probe comparing trainer and serving token IDs on a fixed probe corpus.
- Launch gate requiring:
  - non-empty supervision;
  - expected stop-token inclusion;
  - no media sentinel in supervised span;
  - token-count invariants by model family;
  - identical rendered token IDs across trainer and server where applicable.
- One ShareGPT/prompt-sample converter with an explicit schema for:
  - prompt conversation;
  - privileged context;
  - gold and reference answer;
  - image/video assets;
  - task family;
  - modality.
- Consolidated sampler:
  - modality-bucket behavior;
  - stratified temperature mixing;
  - largest-remainder allocation;
  - optional difficulty weighting;
  - resumable deterministic state.
- A data plane for pretraining:
  - `.bin` + `.idx` indexed corpora or an explicitly chosen equivalent;
  - blended/weighted sampling;
  - document-level loss spans;
  - tokenizer-training or tokenizer-selection workflow.
- Corpus reconciliation:
  - stop calling `pretraining_data/` SFT JSONL a pretraining corpus;
  - register rather than orphan the 152 MB HF-`messages` tool corpus;
  - fix or remove the empty `Taiwan-formosa-VLM-caption-V1/data/` reference;
  - ingest the 23 currently unreferenced `Formosa-Vision` parquet shards or delete the stale path.

### Exit criteria

- A serving/trainer parity harness produces identical token IDs on the probe corpus.
- A run without render/mask parity cannot start.
- The “stock template strips CoT and leaves zero supervised tokens” and “stop token not supervised” classes have fixed regression tests.
- One corpus spec can reproduce an SFT stream with content hashes.
- At least one indexed pretraining corpus is loaded end-to-end by the data plane at smoke scale.

### Dependencies

Phases 1–3. Production compatibility with existing SFT recipes is required.

### Risks

- Existing production relies on precise template behavior. The default transition path must be compatibility against token-ID outputs, not merely qualitatively equivalent rendering.
- Some existing corpus damage, including 23.42% missing-media marker repairs, may have historically been patched in one tree only; schema unification must not silently reintroduce it.

### Falsification

If exact trainer/serving parity cannot be achieved because current serving and training genuinely require different rendering, the roadmap should abandon universal parity and instead require per-mode declared differences with a golden reference. Do not hide the difference in a tolerance.

---

## Phase 7 — Declarative stages: SFT, B1 and A0 prototype

**Objective.** Convert executed capabilities into stage configs and stop representing unexecuted stages as built.

### Deliverables

- A stage graph with explicit fields:
  - model/provider;
  - optimizer;
  - data plane;
  - freeze policy;
  - objective;
  - evaluation gates;
  - export policy.
- SFT declarative recipe migration for existing 26B MoE and 31B dense production paths.
- A B1 recipe built from the existing provider freeze fields:
  - `freeze_vision_model`;
  - `freeze_language_model`;
  - `freeze_vision_projection`;
  - `freeze_sound_encoder`;
  - `freeze_sound_projection` [V].
- Correction of B1-adjacent defects:
  - `VLMLoRA` must target Gemma4’s `vision_tower` and `multi_modal_projector`, not `vision_model` / `vision_projection` (`Megatron-Bridge/src/megatron/bridge/peft/lora.py:203-208`) [V];
  - `_peft_common_vlm` must not silently ship plain LLM-only LoRA in place of VLMLoRA (`recipes/common.py:544`) [V].
- Trainable-partition manifest logging the exact modules frozen and trainable. Today no run records the effective partition; this is the cheapest high-value correction in the system [V].
- One canonical `modeling_gemma4_vl.py` with a `freeze()` contract; the 171-LOC variant lacks it while the 217-LOC variant has it [M/V].
- A0 prototype reusing `megatron.bridge.training.pretrain`, as proven by `omni-accel/train_resume_test_e4b.py`, but replacing `MockGPTDataset` with the Phase 6 data plane [V].

### Exit criteria

- An existing SFT job is reproduced through stage config with matching loss and effective trainable partition.
- B1 executes and saves non-empty checkpoints, with exactly projector-trainable state.
- The trainable parameter counts match the manifest within zero tolerance.
- A0 reaches a real-corpus smoke checkpoint from random init using indexed data, without falling back to SFT masking.
- Every pipeline inventory is annotated by execution evidence, not source-file existence.

### Dependencies

Phases 1–4 and 6. Phase 5 is recommended before exported B1/A0 artifacts are promoted.

### Risks

- Mock-data smoke checkpoints look like real ones on disk. Any A0 claim must report dataset signature, corpus hash and steps.
- B2 depends on document-level interleaving; B1 success must not be used as evidence that VL pretraining works.

### Falsification

If B1 cannot be expressed without invasive model edits, the stage abstraction is too shallow and Phase 7 stops at SFT until the freeze/provider interface is redesigned.

---

## Phase 8 — Services plane and teacher routing

**Objective.** Turn rollout, judge, export and teacher services into schedulable, health-checked, versioned parts of a run rather than tmux/enroot processes outside Slurm.

### Deliverables

- Service descriptors for rollout, judge, teacher and export pools:
  - replicas;
  - TP/EP;
  - endpoint declarations;
  - served model identity;
  - health contract;
  - weight version;
  - KV-token budget.
- Slurm-native co-scheduling equivalent, replacing services held alive by borrowed node reservations and manual `ssh` workflows.
- Blue-green weight update and identity verification, generalizing existing `refit` machinery.
- Colocated vs disaggregated placement as a policy choice, measured rather than assumed.
- Rollout schedulers budget by **aggregate KV tokens**, not only request count. The serving measurement showed 10 tok/s aggregate at 16×~250K-token context versus 270–438 tok/s at 24×~37K-token context — a 27–44× swing invisible to request-count and waiting-request health [M].
- Teacher registry built on the judge-registry substrate after canonical-variant adjudication:
  - family→pool;
  - heterogeneous members;
  - content-aware selection;
  - explicit scoring vs generation roles.
- Replace round-robin `judge_pool.py` selection when “matching teacher” semantics are required.
- Remove the teacher-path `models[0]` collapse in `run_odpo.py`.
- Remote teacher-logprob service where co-residency does not fit. N > 1 co-resident 26B teachers are memory-impractical; today `--anchor_ckpt` is one optional string [V].
- Generalize `polar_a_loss.py` / `facts_loss.py` from a two-source mixture to N sources only after their unification.

### Exit criteria

- An RL run can declare trainer and rollout/judge services in one manifest.
- Services are health-checked by identity and version.
- A blue-green update changes rollout weights without corrupting the served version.
- A synthetic long-context load test produces a KV-budget violation before throughput collapse.
- N=1 teacher mode is byte/benchmark-compatible with today’s single-anchor behavior; N>1 works in a controlled smoke run.
- Teacher selection emits a run-recorded routing decision per rollout.

### Dependencies

Phases 2–5. Also requires Phase 6’s multimodal schema for teacher evidence alignment.

### Risks

- Extending forked judge substrate without choosing canonical variants would recreate the existing fix-miss problem.
- There is no prior art for expert fusion in the repositories: zero slerp/TIES/model-soup/task-arithmetic/DARE hits across 9,905 files [V]. Fusion is therefore out of Phase 8 scope unless separately prototyped and certified.

### Falsification

If a remote teacher service cannot serve logits within the RL latency budget, N>1 routing pauses and the system remains at N=1 while optimization work is isolated to the service contract. Do not revert to an undeclared co-residency shortcut.

---

## Phase 9 — Scale validation and B2

**Objective.** Prove scale claims and build the missing B2 objective.

### Scale validation

- Run staged scaling at:
  - 1 GPU;
  - 4 GPUs / one node;
  - 32 GPUs / 8 nodes;
  - 64 GPUs;
  - 128 GPUs;
  - larger only after 128-GPU evidence.
- Track:
  - throughput;
  - loss parity;
  - NCCL stability;
  - checkpoint bytes and runtime;
  - memory per rank;
  - rollback behavior.
- Quarantine vendored GLM-4.5V and Qwen3.5-VL launchers requesting 16 or 64 nodes, because they are not Omni evidence [M].
- Fix the co-located NCCL hang that currently makes DP=1 the practical default [U].

### B2

- Implement interleaved-document VL pretraining:
  - document-level loss spans;
  - media ordering preserved;
  - no unconditional assistant-only mask;
  - document-level interleave loader;
  - image/video token accounting.
- Prevent `_inject_missing_markers` from collapsing unreferenced media to the first user turn; the observed incidence is 23.42% [V].
- Produce checkpoints and a measured quality/parity report before advertising B2 as built.

### Exit criteria

- No stage is declared complete without at least one executed run and artifact report.
- Scaling curves are published against measured maxima, including failure boundaries.
- Everything above 32 GPUs is labelled **validated** or **unvalidated**, not estimated.
- B2 runs real interleaved data to a non-empty checkpoint.
- B1/B2/B3/B4 coverage table in the repository reflects executed reality.

### Dependencies

Phases 1–8. B2 specifically depends on Phase 6.

### Risks

- Development scope expands from framework work to cluster or vendor debugging.
- Measured scale may be constrained by hardware access rather than software.
- B2 may expose new masking or data-loader semantics that SFT assumptions cannot express.

### Falsification

If 64-GPU validation cannot exceed the existing 32-GPU state because of the unresolved NCCL issue or launch instability, Phase 9 narrows to fixing those blockers. No schedule may claim 100-node readiness without evidence.

---

## Phase 10 — Deprecation, certification and production handoff

**Objective.** Make the consolidated path the only supported path and retire dangerous ambiguity.

### Deliverables

- Archive or delete forked drivers, reward copies and stale exported paths after inventory and migration are complete.
- ci lint:
  - no behavior-controlling environment variable unless declared and recorded;
  - no CWD bare import of reward/dataset modules;
  - no duplicated canonical file name across active packages;
  - no provider module without a MegatronModule-compatible contract.
- Documentation generated from:
  - manifests;
  - registry entries;
  - checkpoint/export artifacts.
- A release notebook joining:
  - W&B run;
  - code content hash;
  - reward version;
  - topology;
  - checkpoint;
  - export;
  - parity result;
  - evaluation result.
- A support runbook converting the current failure-signature archaeology into executable checks.
- Publish corrected public claims:
  - remove unqualified “certified grader”;
  - correct any optimizer-comparison result based on the aliased base;
  - label the seven no-trust-region official GSPO arms accurately.

### Exit criteria

- New engineers can bootstrap a supported run without copying a directory.
- 100% of active production recipes are registered entry points.
- The checker fails builds that import a legacy reward path.
- Old directories are read-only archives or deleted.
- Every public capability statement names executed evidence and its scale.

### Dependencies

All previous phases. Production managers must approve deletion timing.

### Risks

- The strongest risk is cultural: “isolation by directory copy” is attractive because it guarantees no interference. FoundationScale must demonstrate equal or better isolation through immutable config and registered plugins, not merely ask teams to stop.

### Falsification

If migration reveals that “experiment isolation” still requires source overrides more often than configuration/plugin registration, the plugin abstraction failed and must be revised rather than hidden behind a banner.

---

## Parallel “keep the lights on” track

A dedicated production-support lane remains active throughout:

1. **No forced material migration.** Existing SFT/RL runs continue under their known launch scripts.
2. **Additive-only observability.** Manifests, run hashes and objective classifiers can ship before new training internals.
3. **Gates graduate gradually.** Structural checks run as warnings until proven stable, then become blocking for new checkpoints.
4. **Legacy wrappers survive.** Drivers and reward modules delegate into consolidated packages without changing command lines immediately.
5. **Known-invalid artifacts stay isolated.** Aliased-base runs and bad exports are marked by machine-readable status, not prose banners alone.
6. **Production rollback is the current workflow.** Stable OUT_DIR resumption stays supported through Phase 10.
7. **Production-support SLO.** No more than one working day of blocked live training due solely to framework migration; if exceeded, the migration change is rolled back and the phase is re-scoped.

---

## Assumptions and sizing notes

- Estimates assume 5–7 engineers with direct repository and cluster access.
- Engineer-weeks are focused work; production firefighting is not “free.”
- GPU time for smoke and parity is available in small allocations.
- Large multi-node time is not available continuously; Phase 9 therefore includes its own measurement campaign.
- Existing artifacts, logs and audit JSONL files remain accessible.
- FoundationScale is not committing to NeMo, NeMo-RL, DeepSpeed, Ray, FSDP or Hydra by default. All have zero migration debt given current first-party imports [M]. FSDP2 remains an optionally measured execution backend, not the assumed low-end rescue; a Megatron TP=1/EP=1/DP=1 path already exists and has run to step 300 [V].

---

## First two weeks: day-level plan

### Week 1

| Day | Workstream | Concrete output |
|---|---|---|
| **1** | Repository capture | Inventory remotes/mounts; import `omni-accel` into a private Git repo byte-for-byte; freeze source-ref descriptions; add checkpoint/data exclusion policy. |
| **1** | Sibling repo reconciliation | List all 892 dirty files; classify source vs artifacts; commit source or snapshot unknown patches with owner. |
| **2** | Run inventory | Build the definitive live-run table from Slurm, `squeue`/`sacct`, results dirs, audit JSONL and W&B. |
| **2** | Objective provenance prototype | Write audit-log parser and classify at least one run from each P1–P4 policy class. |
| **3** | Manifest design and shim | Add first-run manifest writer to the highest-traffic launcher/driver without changing behavior. |
| **3** | Reward import inventory | Trace reward import resolution across active directories; map each live run to actual repository path where possible. |
| **4** | Build-time gate | Implement `MegatronModule` ancestor assertion with a synthetic negative test. |
| **4** | Save-time gate prototype | Read DCP metadata and implement expert-byte/indexed-key assertions matching the measured 5.73 GB vs 45.70 GB signatures. |
| **5** | Parity harness setup | Select model/prompt set; run Megatron logit extraction and HF logit extraction for one dense export and one MoE export. |
| **5** | Weekly forensic review | Publish discrepancy report: which live runs lack manifests, which exports lack parity, which audit runs are objective-unresolved. |

### Week 2

| Day | Workstream | Concrete output |
|---|---|---|
| **6** | Export parity execution | Run controlled parity on selected exports; include one `BAD_INCOMPLETE*` negative control. |
| **6** | Aliased-base census UI/report | Generate machine-readable disposition table for all 20 MoE runs and bind it to their checkpoint paths. |
| **7** | Export verification library | Move byte-vs-index verification into shared code; wire it into `export_v2_fullft_sbatch.sh` and `export_many.sh`, initially warn-only for the latter. |
| **7** | Topological inventory | Complete list of active launch modes: Slurm, off-Slurm, bare-one-GPU, and all AD HOC special cases. |
| **8** | Objective provenance coverage | Classify all judge-audit-bearing runs; flag `gspo_official4` and ODPO `GOLD_MISS_IS_BAD` runs explicitly. |
| **9** | Reward package skeleton | Create `foxtalk`/FoundationScale reward package layout, policy enum and legacy adapter interfaces; no semantic migration yet. |
| **9** | Canonical-variant review starts | Diff the five `rule_checks.py` variants and the two judge registry/pool variants; record where each is used. |
| **10** | NCCL reproduction | Create minimal reproducer for the co-located multi-GPU hang and separate it from all full-training dependencies. |
| **11** | Gate dry-run on production checkpoint | Run build/save/export gates against a current run and confirm no production interruption. |
| **11** | Decision dashboard | Produce measurable Phase 1 progress view: manifests coverage, provenance coverage, parity coverage, aliased-base coverage. |
| **12** | Phase-1 close review | Decide which gates can become hard failures, which runs require rerun, and which launch paths Phase 3 must support first. |

Day numbering is elapsed working days, not a promise that GPU jobs finish on schedule. Parity and scaling tasks may complete asynchronously.

---

## Decisions required before Phase 1 starts

The spine’s final “Human decisions required before Phase 1” block contains **four numbered decisions**, despite the task asking for three. All four are blocking and are listed without collapsing distinct operational commitments.

1. **Require *semantic* probing on current exports, or accept them as behaviourally unverified?**  
   The recommended answer is **require it**, and the decision is now cheaper and narrower than when it was written. Weight-level correctness was the expensive half and it has been settled for free — 200 of 205 export dirs, 0 DIFFER, no permuted expert axis, on CPU [V·post]. What is left is one node, one GPU, ~20 minutes: one forward pass on an identical fixed batch through the exported HF checkpoint and its Megatron source, asserting top-1 agreement == 1.0 and KL < 1e-3. Run it once for the MoE family and once for Nemotron-Omni. Byte completeness does not establish weight correctness (now independently established), and **weight correctness does not establish that the model runs** — rope, dtype, attention implementation, tokenizer and generation config all pass a perfect tensor comparison [A/U].

2. **Adopt the reversed optimizer result and dispose of the aliased-base arms.**  
   The recommended answer is to **discard the aliased Muon/AdamW comparison and its downstream rankings**. On the aliased base Muon reported 1.4288 versus AdamW 1.4847; on the fixed base matched runs report AdamW 0.7412 beating Muon 0.7828. The structural base bug changed the conclusion of the controlled experiment [A/V]. Full-FT Muon and other arms still need scheduled GPU work.

3. **Strike or narrow “certified” for the deterministic grader.**  
   The recommended answer is **strike the unqualified adjective and preserve the reproducible kappa artifact**. κ=0.9824 is real and re-derived from the 1,800-row reference set, but the oracle is not independent: Kimi-K3 was prompted with the grader’s own rulebook, and all `fill_in_blank`/`short_answer` rows abstain [V]. Produce a human-grounded agreement artifact before restoring a stronger claim.

4. **Set `old_logp_source=frozen` as the framework default and label the affected official arms.**  
   The recommended answer is **yes**. The existing `self` default makes `pi_old = pi_theta.detach()`, yielding sequence ratio exactly 1.0 and clip fraction 0.0; 7 of 10 official arms ran without the named trust-region algorithm [V]. Those runs should not be called invalid, but they must not be represented as ordinary GSPO runs either.

---

## Closing criterion

FoundationScale is successful only when the following stops being a workflow of exceptions:

- a new experiment is a config delta over registered code;
- its objective is in its manifest;
- its freeze policy is recorded;
- its launch topology has one source of truth;
- its first checkpoint is automatically structural-validated;
- its export is semantic-validated;
- its services are health- and version-checked;
- and its scale claim is bounded by an executed run.

Until those conditions hold, every capability statement should carry an evidence level, a repo-qualified source path, and a measured ceiling.