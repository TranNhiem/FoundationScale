# Deliverable A — FoundationScale Architecture Review

**Question under review:** is this a framework, or is it one estate's launch script with
a framework's vocabulary?

**Answer, up front:** partly. The layer stack is real, most seams exist and most of them
are load-bearing, and the reporting contract is enforced in code rather than in prose.
But several seams were retrofitted onto hard-coded estate facts after measured failures,
two seams remain names over hard-coded values, and the cross-model evidence has a
denominator of 2. Section A.7 gives the verdict and the unearned claims.

**Elision notice.** This document is published to a public repository. Estate identifiers
are elided throughout: `<PARTITION>` (Slurm submit partition), `<NODE>` (compute-node
hostname), `<HOME>` (account home path), `<IMAGE>` (container image path). A build gate
scans this file case-insensitively and refuses the build over a single hit.

**Sources and denominators.** The only sources of truth for numbers and job outcomes are
`h100/DELIVERABLE_E_matrix.md` and `PHASE4_FACTS_QWEN_CHAIN.md`. Claims derived by reading
code say so. Anything else is written UNMEASURED. Files in scope for every code-reading
claim below: the six generated artifacts (`launch_fs_h100.fixed.sh`,
`fs_container_backend.bound.sh`, `fs_train.fixed.py`, `fs_model_root.py`,
`fs_ckpt_adjudicator.py`, and the build plane `build_h100_plane.sh`), plus `h100/LAUNCH.md`
as the operator-facing claim set. Where this review reports "no occurrence found", the
method was a full read of those files as supplied, not a grep with a recorded pattern
count; that is a weaker instrument and is stated as such.

---

## A.1 What the framework is, in one page

Six layers, top to bottom. Line counts: the build plane prints `wc -l` per artifact at
the end of every build, but no build output is among the files supplied to this review,
so exact counts are UNMEASURED except where LAUNCH.md records one.

| # | Layer | File | Lines | Abstracts over | Hands to the next layer |
|---|---|---|---|---|---|
| 1 | Build plane | `build_h100_plane.sh` | UNMEASURED in these sources | Upstream estate text plus an ordered, gated stage list; nothing under `h100/gen/` is hand-maintained | The generated artifacts below, rebuilt from scratch each run, plus standing gates (env drift, input partition, exit contract, blocklist, parse, naming agreement, linkage, launch-doc) |
| 2 | Launch plane | `h100/gen/launch_fs_h100.fixed.sh` | ~800 per LAUNCH.md §4 (approximate; not re-measured here) | Slurm submit chain (probe → production → resume → post-mortem), required-env contract, path-root containment, walltime proof against a live `sinfo` probe, bind-set derivation, the adjudication walk | A validated environment, a declared `FS_BIND_PATHS` array, and a composed engine command, to the backend |
| 3 | Runtime/allocation backend | `h100/gen/fs_container_backend.bound.sh` | UNMEASURED | Two orthogonal axes — container runtime (`singularity`\|`enroot`) and allocation (`slurm`\|`local`) — plus the env allowlist, bind materialization (`--bind` vs `--mount` from one declaration), drain gate, and in-container probes (interpreter tripwire, torch provenance, NVML/device count) | A verified in-container execution of the engine command; every in-container step routes through `run_in_container` |
| 4 | Model-root resolution | `h100/gen/fs_model_root.py` | UNMEASURED | Three measured meanings of "the model root" (self-contained, config-overlay, commit-SHA nested); config searched per depth, shallowest populated depth wins | `config_dir` plus `bind_closure` to the trainer; `ModelRootError` on zero, ambiguous, or unreadable candidates |
| 5 | Trainer | `h100/gen/fs_train.fixed.py` | UNMEASURED | Model class (dispatched on `config.architectures`, never on a name table), dataset mode (`real`\|`synthetic`), FSDP wrap policy (from the model's own `_no_split_modules`), checkpoint write and resume proof | A checkpoint tree (`rank-local-sharded-v1`) and a JSONL evidence stream (`PHASE_JSON`, `TRAIN_JSON`, `RUN_SUMMARY_JSON`) |
| 6 | Checkpoint adjudication | `h100/gen/fs_ckpt_adjudicator.py` | UNMEASURED | Checkpoint format, isolated behind the `FORMAT_READERS` registry (current size: 1) | A per-checkpoint verdict in the launcher's exit vocabulary (0/95/96), consumed by `adjudicate_tree` in layer 2 |

The shape in one sentence: a generated plane in which the launcher validates and declares,
the backend executes inside a container it has measured, the trainer trains and proves
resume, and the adjudicator reads what the trainer wrote — with the model-root resolver
as the trainer's mandatory front door and the build plane regenerating all of it from
stages.

A reader who stops here should also know: the launcher and backend are shell; the
resolver, trainer, and adjudicator are Python; the trainer's exit vocabulary (0/1/2/3)
is a different namespace from the plane's (0/5/95/96), and the two meet only at
`run_in_container`, which propagates the trainer's code unchanged.

---

## A.2 The seams — where model-specific and estate-specific knowledge is allowed to live

Summary table, then per-seam notes. "Retrofitted" means the seam was a hard-coded value
until a measured finding parameterised it; the evidence is the patch-stage names in
`build_h100_plane.sh` and the ledger rows in `DELIVERABLE_E_matrix.md` §E.4.

| Seam | What varies | How expressed | Validated by | When absent | Real seam? | Designed or retrofitted |
|---|---|---|---|---|---|---|
| Model identity / model root | architecture class, root layout, bind closure | `MODEL_DIR` env + `config.json`'s `architectures` field; `fs_model_root.resolve_model_root` | resolver per-depth candidate counts; `load_artifacts` requires exactly one `architectures` entry and the class to exist in `transformers` | trainer `OperationFailure`, exit 3; resolver `ModelRootError` with denominators | Real — dispatch is config-declared; no model-name table exists in the code | Retrofitted (#133, three stages) but now structural and mandatory (no fallback import) |
| Dataset | corpus, text field, synthetic vs real | `--dataset-mode`, `--dataset-path`, `--text-field`; `DATASET_DIR` at the launcher | trainer contract (mutually exclusive branches), field membership against `column_names`, partition-size checks | `ContractError`, trainer exit 2 | Real | Designed in the trainer; executed denominator is 1 corpus |
| Container runtime | `singularity` vs `enroot` | `FS_CONTAINER_RUNTIME`, required, no default, never auto-detected | `fs_backend_init` case; `run_in_container` re-validates at the arm boundary | exit 96 | Real — one `fs_bind_spec` declaration materialised per arm | Retrofitted (the fix43 R1/R2 rework of the conflated `FS_BACKEND`) |
| Allocation | `slurm` vs `local` | `FS_ALLOCATION`, required, no default, never inferred from `SLURM_JOB_ID` | `fs_backend_init`; local arm mints `SLURM_*` only after the kernel ground-truth guard | exit 96 | Real | Retrofitted (same rework; the measured estate has `srun` but no pyxis, so the conflation was a live bug) |
| Interconnect / communication | NCCL net plugin, socket interface, IB HCA, fabric tripwire | `FS_NCCL_NET_PLUGIN` (`none` or absolute `.so`), `FS_NCCL_SOCKET_IFNAME`, `FS_NCCL_IB_HCA` (forbidden-if-set), `FS_FABRIC_TRIPWIRE` (`host:port` or sentinel `none`) | value validation; the fs129 collective probe measures an 8-rank all_reduce rather than trusting any of it | plugin unset → image default, which measured SIGSEGV on this estate (E.3); tripwire unset → exit 96 | **Mixed.** Plugin seam is real and measured. `FS_NCCL_SOCKET_IFNAME` is validated then discarded (#131) — a name over a hard-coded value, seam not yet real | Plugin seam designed after measurement (#129); tripwire retrofitted (#163 — it was one estate's hard-coded endpoint and killed job 37280 on another) |
| Slurm partition / walltime | submit partition, max walltime | `FS_PARTITION`, required, no default, carried on the `sbatch` invocation (the `#SBATCH --partition=` directive was deleted, not parameterised — fs152) | launcher guard; walltime proven against a live `sinfo` probe of the named partition | exit 96 | Partition: real. Walltime: **partially** — a hard-coded 7-day literal guard runs before the live probe and refuses legitimately shorter values (LAUNCH.md §7 known limitation) | Retrofitted (#152; the partition was hard-coded 13×, 2 functional) |
| Checkpoint format / adjudicators | checkpoint layout, which adjudicators run | `FS_CHECKPOINT_ADJUDICATORS` list; `FORMAT_READERS` registry keyed on the manifest `format` field | containment check per spec (fs146a), dirname auto-bound (fs146b); writer/adjudicator naming agreement gate (#150) | empty list → exit 96 (`all([]) != PASS`, #68); unknown format → ABSTAIN → exit 95 | Real by construction (registry), but registry size is 1 | Retrofitted in stages: #68 (zero call sites), #141 (adjudicator shipped), #146 (binding), #150 (naming drift) |

Per-seam notes:

* **Model identity.** The strongest seam in the plane. The trainer resolves the class by
  `getattr(transformers, architecture_name)` on the config's single declared entry, and
  the wrap policy comes from the model's own `_no_split_modules` (see A.5). The registry
  is the installed `transformers` package itself, which is also the seam's limit:
  `gemma4_unified` and `gemma4` are refused by the package (E.1.4: 3 rows, 0 executed, 2
  measured NO), and there is no architecture registration seam. That absence is named in
  E.1.4 as the standing gap, not papered over.
* **Dataset.** The contract is genuinely exclusive — supplying a flag from the branch not
  selected is a `ContractError`, same as an omission. But the executed evidence is one
  corpus (`phase3_real_4k`) across both model arms; cross-dataset generality has a
  denominator of 1.
* **Runtime and allocation.** Two axes, both required-no-default, and the refusal text
  states why: auto-detection makes the runtime an accident of `$PATH`, and `SLURM_JOB_ID`
  says a job is running, nothing about which runtime the node has. This is the correct
  shape. It is nonetheless a retrofit: the measured estate has `srun` without pyxis, and
  the old single-axis code inferred a pyxis arm from Slurm's presence. The enroot arm is
  UNMEASURED on this estate (E.6); the seam is real in code and unproven in execution for
  its second value.
* **Interconnect.** The fs129 probe is the right pattern: the framework does not trust
  the knob, it measures the collective (`world=8 got=28.0 expected=28.0 spread=0.0`,
  E.3), with a MUST_FIRE control (`FS_PROBE_CORRUPT=1`, observed `got=29.0`). Against
  that, `FS_NCCL_SOCKET_IFNAME` is validated and then has no reader (#131, listed in
  LAUNCH.md §8 as UNMEASURED effect) — a knob that claims configurability it does not
  have, and the review should not count it as a seam.
* **Partition and walltime.** The partition seam is real and well-built (required, no
  default, deleted directive rather than a parameterised comment). The walltime is the
  residual estate fact: oracle 1 is the literal `7-00:00:00`, oracle 2 is the live probe,
  and oracle 1 runs first. Fail-closed in both directions, but a framework that refuses
  `FS_WALLTIME=0-02:00:00` on a partition it never asked is carrying one estate's maximum
  in its vocabulary.
* **Adjudicators.** The seam is real and the registry is the right shape — a second
  format is one reader plus one entry, and the driver's comment says why no
  format-specific branch belongs in a generic gate. The denominator is 1 format
  (`rank-local-sharded-v1`), which is the framework's own writer format; no foreign
  checkpoint format has ever been adjudicated.

**Designed vs retrofitted, in aggregate.** Of the seven seams, five carry retrofit
evidence in the stage list or the E.4 ledger (#122, #129, #133, #146, #150, #152, #163).
A retrofitted seam is still a seam — the guards are real, required, and fail closed —
but its generality is less trustworthy than a designed one, because the parameterisation
was driven by one measured failure and may encode exactly that failure's shape. The two
clear instances are the walltime literal (one estate's maximum, kept deliberately as a
stale-rule guard) and `FS_NCCL_SOCKET_IFNAME` (a validated non-reader). The review treats
both as open, not as seams.

---

## A.3 The build plane as an architectural choice

Every artifact under `h100/gen/` is generated. `build_h100_plane.sh` runs an ordered
`STAGES` list over upstream text; each stage refuses to write while its own gates are
red; the tree is removed and rebuilt on every invocation; standing gates run over the
result (env drift, input/output partition, exit contract, blocklist, parse, generated
suites, naming agreement, linkage, launch-doc).

**What it buys.**

* Every fix is a stage, so every fix is reproducible and auditable; a hand edit puts the
  fix in the file you read and leaves it out of the file that runs, and the build
  prohibits that by construction.
* The estate's own text is an input, not a dependency: `h100/upstream/` holds the base
  texts with recorded sha256 (#137), and `gate_build_inputs.py` partitions every touched
  file into declared-artifact or documented-upstream.
* Determinism makes the gates meaningful: E.5 measured two consecutive full rebuilds
  byte-identical by sha256.

**What it costs, with the measured failure mode.**

* **An authored fix that is not in the stage list never runs.** This has happened twice.
  #136: the backend's 73 KB base text was produced by `apply_splice.py`, which was in no
  stage list, so it survived every "from scratch" rebuild (measured: moving it aside
  turns the build red at stage 1). #188: `patch_adjudicate_denominator.py` was authored,
  verified, and committed under #175, then wired into neither `STAGES` nor
  `PUBLISH_SET.txt` — and the build kept regenerating the defective one-liner whose loop
  body drains its own process substitution. Measured on jobs 37341, 37342, 37344: a tree
  holding 3 checkpoint directories adjudicated 1 deep, three times, under the banner
  `ADJUDICATE complete`, every job recorded COMPLETED 0:0. Two occurrences is a class,
  and E.6 records it as one: how many other `patch_*.py` files sit in no execution list
  is UNMEASURED, because no gate cross-checks the two sets.
* **Line-number citations rot on every launcher edit.** #154 measured 19 of 19 citations
  in the operator document pushed stale by one resolver insertion. `gate_launch_doc.py`
  now re-anchors them, which is why this review cites file-plus-identifier and no line
  numbers.
* **The measured record lags the stage list.** *(Restated 2026-09-01: the lag named here
  was real and has since been closed twice over — E.5 now reports 36/36 and the count is
  derived from the build script rather than retyped, `gate_doc_stage_count.py`. The
  original observation is kept because the class it names is the point.)* E.5 reported 17/17 stages green, 5
  artifacts, blocklist 0 hits over 5 files — measured 2026-08-31. The build script
  supplied to this review contains 32 entries in the `STAGES` array and 7 generated
  artifacts (both counted from the array literals in the supplied text). A green run of
  the supplied revision is UNMEASURED in these sources. The same lag is visible in
  LAUNCH.md §8, which states Phase 3 has never been executed; the Phase 4 record (E.1.1,
  `PHASE4_FACTS_QWEN_CHAIN.md`) supersedes it. Document staleness is the cost side of
  generated code beside hand-written docs, observed live in the supplied set.

**The invariant that would close the orphan-stage class.** Every `patch_*.py` in the
repository appears in `STAGES` or in `PUBLISH_SET.txt`, enforced by a gate that
enumerates both sets and refuses over any file in neither — the same derivation-over-
ordering pattern the build already applies to its scan denominators (#157) and to the
second, deliberate re-application of `patch_list_separators.py`. Status: owed, listed in
E.6, not yet built.

---

## A.4 What is genuinely model-agnostic, and what the evidence for that is

Two different claims must be kept apart.

**(a) A property of the code, readable in front of you.** The supplied artifacts contain
no conditional on model identity. The method was a full read of the six files in scope;
the dispatch points that exist are all config-declared: `load_artifacts` resolves the
class from `config.architectures` via `getattr(transformers, ...)`, `_resolve_wrap_policy`
reads the model's own `_no_split_modules`, and `_check_tied_parameters` groups by storage
pointer or object identity, not by name. Model names appear in comments (the fs172 block
narrates the Qwen3 measurement) but not in executable branches. This is a real property
of the code as supplied.

**(b) A property of the measurements, with a denominator of 2.** Two models —
`qwen3-0.6b` and `gemma-3-1b-it` — ran the same framework revision, the same argv shape,
and the same corpus (`phase3_real_4k`), end to end, on 8 ranks of one node, runtime
`singularity`, `FS_NCCL_NET_PLUGIN=none` (E.1.1; `PHASE4_FACTS_QWEN_CHAIN.md` §1, §4).
Outcomes: Qwen chain 37340-37343 all COMPLETED 0:0, resume `restore_delta = 0.0`,
cross-rank spread `0.0` at steps 5 and 50; Gemma 37319 COMPLETED, restore bit-identical
(`after_resume = before_save = 0.5986318588256836`), cross-rank spread `0.294` at step 50
with one declared abstention (`resume.fixed_eval_rank_invariance`, UNMEASURED, not a
pass). The one behavioural divergence is attributable to the model because the identical
code path produced exact cross-rank agreement on the control; E.1.2 lists the seven
hypotheses ruled out by measurement (data, weights, restore fidelity, attention backend,
sliding window, dropout/softcapping, missing keys), each with its denominator, and names
the mechanism inside Gemma UNMEASURED.

**The licensed claim is exactly:** on this revision, the framework's code path is
model-name-free (claim a), and it executed two small models from two families with one
characterized, model-attributable divergence (claim b).

**The denominator, stated as a denominator:** 2 models (both small: 0.6B and 1B), 1 node,
8 ranks, 1 container runtime (`singularity`; enroot UNMEASURED here), 1 dataset, 1
framework revision, 1 engine (transformers + FSDP; Megatron/NeMo measured absent or
unusable in the container). E.1.3 adds 4 larger models declared by config search and
never executed (4B, 7B, 8B, 27B — "Builds" means meta-device instantiation, weights never
materialised). E.1.4 adds 3 Gemma-4 rows, 0 executed, 2 measured NO at build.

**What is NOT licensed:** "FoundationScale is model-agnostic" as an unqualified claim.
The evidence does not cover a second runtime, a second dataset, a second node, a model
above 1B executed end to end, or any architecture the installed `transformers` does not
already register.

---

## A.5 Where model-specific knowledge still leaks

Read from the code, not from the ledger. For each: location, the assumption, the
classification, and the abstraction that would remove it.

| Location | Assumption | Defensible generic default, or model assumption wearing a default's clothes | Removing abstraction |
|---|---|---|---|
| `fs_train.fixed.py::_build_model` | `dtype=torch.bfloat16` hard-coded in both `from_pretrained` call forms | Model assumption. bf16 is an estate-and-model property; a model or part that needs fp32/fp16 gets no say and no refusal names the choice | Read dtype from the resolved config (`torch_dtype`) with a refusal when absent, or a required engine-config knob with provenance |
| `fs_train.fixed.py::_resolve_wrap_policy` | the model declares `_no_split_modules` as a non-empty list of class-name strings | Defensible — the model-specific fact stays in the model, and refusal replaces the measured size-based failure (job 37300: tied embedding sharded, `vec (48619520)` = 388,956,160/8 on all 8 ranks). But it is a HuggingFace-convention dependency: a model without the declaration cannot run at all | A declared wrap-policy override in the engine config, with provenance, refused silently never; never a size-based fallback |
| `fs_train.fixed.py::_check_tied_parameters` | none model-specific — groups by `data_ptr()`/object identity on the pre-wrap tree | Defensible generic mechanism; zero tied groups is recorded as a measured zero, and the detector's own vacuity modes (`remove_duplicate` default, post-wrap null pointers) are named and avoided in code | None needed |
| `fs_train.fixed.py::_make_batch` | causal-LM interface: tokenizer yields `input_ids`/`attention_mask`; labels = input_ids with -100 on padding; the model returns a scalar `loss` over `labels`. No chat template is applied | Defensible as an infrastructure-proof default; it is a model-interface assumption for training semantics. The executed Gemma arm is instruction-tuned (`gemma-3-1b-it`) and was trained on raw text — correct for a continuity proof, not a statement about instruction tuning | A declared collator/template seam in the engine config; the trainer refuses when the model's forward does not emit a scalar loss (it already does, via `OperationFailure`) |
| `fs_train.fixed.py::load_artifacts` | `AutoProcessor` iff `config.processor_class` is set, else `AutoTokenizer`; a pad token id must exist | Defensible inside the one measured engine (transformers). It is an engine assumption, not a model assumption; Megatron/NeMo are measured absent, so no second engine interface exists to generalise to | The engine seam itself (`FS_ENGINE_LAUNCH_CMD`/`FS_ENGINE_LAUNCH_MODE`) already exists at the launcher; the trainer is one engine adapter. Naming it as such in the artifact map would be honest; no code change owed |
| `fs_train.fixed.py` contract / `--sequence-length` | operator-supplied, required, no default — but nothing checks it against the model's declared context or sliding window | Leak by omission. E.1.2 measured the 512 sliding window inert at seqlen 512; the framework learned that from forensics, not from a guard | A load-phase check reading the resolved config (`max_position_embeddings`, `sliding_window` where declared) and refusing, or declaring UNMEASURED, when `sequence_length` exceeds a model-declared bound |
| `fs_train.fixed.py::_initialized_distributed`, launcher `fs129_collective_probe` | backend `"nccl"` hard-coded in both | Estate assumption (NVIDIA), defensible today; on a non-NCCL fabric it is a seam in name only | Backend selection as a required knob, validated by the collective probe it already runs |
| `fs_train.fixed.py::build_runtime`, `::_optimizer_steps` | optimizer is AdamW, constructed inline; the step-continuity proof assumes an AdamW-style `step` entry in optimizer state | Generic default, defensible for the proof's purpose; not configurable | Optimizer factory in the engine config; the continuity proof reads the optimizer's declared step key rather than assuming `"step"` |
| `fs_train.fixed.py::build_runtime` | `ShardingStrategy.FULL_SHARD`, `use_orig_params=True`, fixed | Defensible at the measured shape (8 ranks, 1 node); multi-node is UNMEASURED (E.6), so the strategy's generality is unproven, not wrong | Strategy selection in the engine config once multi-node evidence exists; parameterising it now would be a knob with no measured reader |
| `fs_ckpt_adjudicator.py::FORMAT_READERS` | registry size 1; the key vocabulary (`rank-%05d.pt`, `world_size`, `rank_payload_count`, `global_step`, `fixed_loss_before_save`) is the framework's own writer format | Not model-specific — correctly so. The seam is real by construction (one reader + one entry per format) and cross-checked by `gate_ckpt_naming_agreement.py` (#150: writer and adjudicator drifted while both suites stayed green, 31 green tests, leg A7b abstaining on 2/2 real shapes) | None owed for models; the registry is the abstraction. A second format has never been registered — denominator 1 |
| `fs_train.fixed.py::load_artifacts` → `getattr(transformers, architecture_name)` | the architecture is registered in the installed `transformers` | The measured limit of the model seam: `gemma4_unified` and `gemma4` are refused by the package (E.1.4, 2 NO rows, 0 executed) | The architecture registration seam E.1.4 already names: a declared, verified extension point for out-of-tree architectures, rather than a core edit per vendor fork |

Net: the leaks that remain are mostly **defaults that have never had to move** (dtype,
optimizer, sharding strategy, NCCL) plus one **omission** (sequence length vs model
bounds) and one **absent seam** (architecture registration). None of them is an
`if model == ...` branch; all of them are places where a second model family, engine, or
fabric would currently have to edit core code or be refused.

---

## A.6 The reporting contract, and why the architecture has one

The plane publishes four exit states — 0 PASS, 5 RED, 95 UNMEASURED, 96 REFUSE — and the
trainer publishes its own four — 0 measured, 1 selftest mismatch, 2 contract refused,
3 ran-but-unmeasured. The adjudicator emits only 0/95/96 and maps any internal exception
to 96 so the launcher's composition never sees an undeclared code; the launcher's
`run_adjudicators` maps any undeclared adjudicator code to 96 with the original printed,
because a laundered silent pass is, in its own words, `all([])` wearing green.

**Why UNMEASURED is a declared state with its own exit code, rather than a pass.**
Because `all([])` is True. Any sweep, walk, or gate whose body executes zero times
returns vacuous success unless the framework makes "zero units measured" a separately
reportable outcome. The code enforces this in three independent places:
`DenominatedCount` distinguishes `measured_zero` from `unmeasured` and the
`MeasurementLedger` degrades the run verdict to UNMEASURED (exit 3) if any payload
carries the status; the adjudicator's `_finish` backstops exit 0 with
`measured == 0 → UNMEASURED`; the launcher refuses `checkpoint_observed == 0` with 95 and
compares `observed` against an independently measured `found`, exiting 5 on partial
coverage.

**Receipts — the two failures the state exists to prevent.**

* **#187 — a pass over zero adjudications.** The post-mortem `afterany` link exists for
  the case production died. Job 37343: the entire log was one line, no `END` line, exit
  0, three checkpoints on disk, 0 of 3 adjudicated, denominator never printed, `sacct`
  records COMPLETED 0:0 — indistinguishable from a clean sweep. The fs187 fix did not add
  a verdict; it stopped exiting before the existing verdict could run. Job 37344:
  COMPLETED 0:0 through the shared adjudication tail.
* **#188 — a denominator derived from the stream it measured.** The shipped
  `adjudicate_tree` invoked the per-checkpoint runner inside the streaming read loop; the
  runner inherited the loop's stdin and drained the `find` stream, so only the first
  directory was ever adjudicated — and the count was incremented inside the same
  truncated loop, so numerator and denominator were cut together and the report stayed
  self-consistent at one third of the truth. Observed on three jobs (37341, 37342,
  37344), all COMPLETED 0:0. No internal consistency check could have caught it; only a
  count taken independently of the walk exposes it. The installed fix collects first,
  feeds every runner `</dev/null`, counts `found` before any adjudicator runs, and
  refuses 96 when `processed != found`. Job 37345: `adjudicated=3 of 3`, three
  `VERDICT 0 PASS` lines, `checks_measured=10 checks_green=10 checks_red=0` each.

Two supporting facts from the ledger. The contract itself was once collapsed: `fs_die`
exited 1 at all 72 call sites until #163 made it code-aware (`patch_fabric_tripwire.py`),
and twelve Python exit sites used `raise SystemExit("text")` — exit 1 — until
`gate_exit_contract.py` (#161). And the abstention is honoured where it occurs: Gemma
37319 reports `verdict:"MEASURED"` with `unmeasured:["resume.fixed_eval_rank_invariance"]`
— a declared abstention carried in the output, not upgraded to a pass and not laundered
into a resume failure (fs178: restore fidelity and cross-rank agreement are two
statistics, and RED outranks divergence so a real restore defect cannot be renamed).

The design principle, stated once: **a measurement framework's most dangerous output is a
self-consistent report of a partial measurement.** Exit 95 exists so that "we could not
measure" is a first-class, greppable, non-zero outcome that no caller can mistake for
success — and every denominator is printed so that a shrunken sweep reads as shrunken.

---

## A.7 Verdict, with the parts that are not yet earned

**Verdict: partly a framework.**

Earned: a named layer stack with one executor (`run_in_container`) and one mandatory
model-root resolver; seams that are mostly real, required-no-default, and fail closed; a
build plane that makes every fix reproducible and auditable; a reporting contract in
which UNMEASURED is a declared state with receipts showing why; and execution evidence
that the code path is model-name-free and ran two model families end to end with one
model-attributable divergence.

Not earned: the unqualified claim "FoundationScale is model-agnostic" (denominator 2
models, 1 node, 1 runtime, 1 dataset); any multi-node claim; any second-runtime claim on
this estate; any claim over models larger than 1B executed; the Gemma-4 family (refused
at build, registration seam absent); and the orphan-stage class is open — two measured
occurrences (#136, #188), no gate.

What would have to be measured before the stronger claim is available, with current
status taken from E.6:

| Claim sought | Required measurement | Current status (E.6) |
|---|---|---|
| Multi-node generality | a multi-node launch, any two nodes; E.3's socket row is a control, not a pass | UNMEASURED — every executed row is one node, 8 ranks |
| Scale generality | load/train/checkpoint/resume/eval on the declared 4B, 7B, 8B, 27B rows | UNMEASURED — E.1.3 denominator: 4 declared, 2 executed; "Builds" is meta-device only |
| Runtime generality | an enroot run on this estate (the #122 var-crossing fix is in code) | UNMEASURED here — enroot exists on another estate; no enroot run is in the matrix |
| Dataset generality | a second corpus through the same argv shape | UNMEASURED — no E.6 row exists for this; all executed rows used one corpus (`phase3_real_4k`), so the cross-dataset denominator is 1 |
| Architecture extensibility | the registration seam, then a Gemma-4 row executed through it | REFUSED at build / UNMEASURED — E.1.4: 3 rows, 0 executed, 2 measured NO; no framework pass may be reported until the seam exists |
| Gemma divergence understood | root-cause the cross-rank forward spread | UNMEASURED — E.1.2 lists what 37320-37323 ruled out; the mechanism is not guessed |
| Gemma abstention closed | re-run the Gemma tree under the fixed adjudicator; resolve `resume.fixed_eval_rank_invariance` | UNMEASURED — only the Qwen tree has been re-walked (job 37345, 3 of 3); the 37319 abstention stands as declared |
| Orphan-stage class closed | a gate enumerating every `patch_*.py` against `STAGES ∪ PUBLISH_SET` | UNMEASURED as a class — two occurrences measured, the sweep has never been run |
| Invocation self-record | the run log records its own `FS_ENGINE_LAUNCH_CMD` and key flags | UNMEASURED as a shipped property — #180: on 37319's 936-line log the launch command and four key flags occur 0 times combined |

Until those rows move, the honest one-line summary is the one A.1's question deserves:
the framework's *structure* is model-agnostic by construction and estate-agnostic by
required-no-default knobs; its *evidence* is two small models, one node, one runtime, one
dataset — and the framework itself insists, in its own exit codes, that the difference
between those two sentences is the whole point.
