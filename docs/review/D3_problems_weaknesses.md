# Deliverable 3 — Problems & Weaknesses

Ranked by blast radius: how much of the intended audience (dense + MoE, 4B–100B+, H100/H200/GB200/GB300, pre-train/SFT/RL/multimodal) is actually blocked, and at what point in their day. All claims trace to the measured census or the adjudicated findings; where a finding shipped an overstated claim, the corrected form is used.

## Rank summary

| # | Theme | Verdict classes | Evidence base |
|---|-------|----------------|---------------|
| 1 | The shipped package is not the advertised product | P0 Missing, P1 Redesign | Census §3 probe; A_front_door#0/#1; F1_docs#0/#1 |
| 2 | The verification engine fires on one seam only, via a path that bypasses its own public helper | P1 Missing (re-scoped post-`train/`) | Census §6 import/call-site graph; B_gate_engine#4; re-measured #245 |
| 3 | The library/script boundary — **closed during this review**; a 98-name compatibility shim is the residual | ~~P0/P1 Redesign~~ → **P2 Simplify** (fix landed) | T2_lib_script_boundary#0–#15, plus post-move re-measurement |
| 4 | Documentation describes a system that does not exist | P0/P1 Redesign (CONFIRMED) | F1_docs#2; F2_docs#1; F2_docs#5 |
| 5 | Model- and campaign-specific semantics are hardcoded into "model-agnostic" infrastructure | P0/P1/P2 (mixed) | T3_skeptic#0/#3/#5; C_gates_domain#1/#2; T2#6/#10 |
| 6 | Single-geometry lock-in: the topology registry has no H100/GB200/MNNVL profile | P2 Missing | A_front_door#5 |
| 7 | Save-side-only verification: no resume/load-side contract | P1 Missing (CONFIRMED) | D_checkpoint_verify#0/#2 |
| 8 | Boundary duplication with demonstrated drift | P1/P2 (mixed) | T2#5; D_checkpoint_verify#4; C_gates_domain#4/#6 |
| 9 | Docs root is an audit ledger, and its own doctrine calls this a defect | P1/P2 (mixed) | F1_docs#3; F2_docs#2/#7; F2_docs#3 |

---

## Theme 1 — The shipped package is not the advertised product

**What it is.** FoundationScale presents as a foundation-model training framework. Most of the measured content of `src/foundationscale` (18,915 LOC) is a verification plane: gates, checkpoint readers, a parity comparator, topology validation, provenance manifests. The training-primitive probe reads zero on every marker:

| Probe (across all 25 git-tracked `src/*.py`) | Match count |
|---|---|
| `nn.Module` | **0** |
| optimizer step | **0** |
| backward | **0** |
| dataloader | **0** |
| forward pass | **0** |
| dist collective | **0** |
| module-scope `torch` import | **0** |
| any-scope `torch` import | 3 files: read-side `checkpoint/dcp.py` and `verify/parity.py` — plus `train/loop.py` |
| **delegated trainer construction** | **1 file: `train/loop.py`** |

**The zero is real; the conclusion the first draft drew from it was not.** There *is* a `train/` package under `src/` — `loop.py` (1,168 lines), `cli.py` (108), `__init__.py` (29) — and it builds a `transformers.Trainer`, calls `trainer.train()` and `trainer.save_model()`, and gates every checkpoint it writes. It imports torch at function scope, so a marker set built from primitives cannot see it (#223, #245). The honest Theme-1 statement is therefore **narrower and still a problem**: the package trains by *delegation to one `transformers.Trainer` path*, while the estate the launchers actually drive is Megatron/NeMo in a container — so the shipped trainer is not the trainer that runs the H100 and GB200 workloads, and it is the least-tested module in the package (`train/loop.py` 62% vs ≥81% everywhere else, #228).

The README itself admits the trainer "is early", and its three-step Quickstart exercises only `pytest`, `foundationscale.gates.controls`, and `tools/mutate.py` — a quickstart that never reaches the trainer the package now ships (F1_docs#0, corrected: a Quickstart *does* exist; what is absent is one that trains). The top-level package ships a 3-line `__init__.py` that exports nothing (A_front_door#2), so even the `train` entry point is invisible from `import foundationscale`, and the front-door messaging claims a distributed-training product the packaged path does not deliver (A_front_door#1).

**Who it hurts and when.** Every new adopter, in the first ten minutes. An engineer who installs `foundationscale` following the stated goal (make distributed training *easier*) finds a top-level package that exports nothing, a Quickstart that runs pytest, and — only if they already know the name — a `foundationscale-train` console script that fine-tunes a causal LM through `transformers`. Nothing in the front door leads them there, and nothing in the front door tells them that the distributed path the docs describe is the launcher estate, not the package.

**Failure mode in practice.** The first contact is not a training failure mid-run; it is silent disbelief at the install prompt. The repo burns its credibility budget before a single GPU is touched, because the README's framing promises a framework and the disk contains a (genuinely good) gate engine.

---

## Theme 2 — The verification engine fires on one seam, through a path that bypasses its own public helper

**What it is.** The gate engine is the crown jewel of the current slice — multi-way-blocking `Verdict`, frozen `Coverage` carried per result, ERROR-on-exception, MUST_FIRE controls. The measured call-site graph for `run_event`, the public helper written to be *the* way to invoke it:

| Layer | `run_event(...)` call sites |
|---|---|
| `src/` | 2 — the `def` in `gates/core.py` and one call in `integrate.py`, the wrapper around it |
| `tests/` | 18 |
| `tools/` | 0 (one comment mention in `preflight.py`) |
| `h100_validation/` | 0 |
| **Production** | **0** |

**Re-measured after `train/` landed, and the finding narrows rather than dissolves.** One production seam now exists: `FoundationScaleSaveGate.on_save` in `train/loop.py` runs the registry on every checkpoint (`FIRST_SAVE` first, `SAVE` after) and fails closed. But it calls `registry.run(event, ctx)` directly, so the public helper is still dead code sitting beside a hand-rolled equivalent — two ways to do one thing, one of them untested in production and one of them undiscoverable. And the seam covers exactly one of six lifecycle points: `LAUNCH`, `STEP_ZERO`, `EXPORT` and `PROMOTE` have no in-process caller, and there is no RESUME/LOAD member at all (Theme 7). B_gate_engine#4's correction stands in re-scoped form: the *general* integration seam is missing; what exists is one bespoke callback for one trainer.

**Who it hurts and when.** The framework's own maintainers, continuously: they are maintaining a defect-injection control framework whose only adversary is its own test suite. And every operator of a trainer other than the packaged one. The save seam is now wired for exactly one path — `FoundationScaleSaveGate.on_save` in `train/loop.py` — so an operator on the Megatron/NeMo estate the launchers actually drive still gets zero runtime protection from the gate plane until someone hand-wires the equivalent callback.

**Failure mode in practice.** The gate plane can pass CI green indefinitely while the pipelines it nominally protects run outside it. The one wired seam narrows the gap without closing it: the gates have now seen checkpoints from the packaged `transformers` trainer, and still none from the H100 or GB200 jobs this framework was built for. "Our gates have never seen a checkpoint from a cluster job" remains true, and that is the sentence that matters.

---

## Theme 3 — The library/script boundary, closed during this review

> **Status: FIXED.** This theme described the single largest structural defect the review
> found. It was repaired while the review was being written (T2_lib_script_boundary#0), so
> the text below states the original problem, the delivered fix, and the measured residual.
> Everything after "What remains" is a live finding; everything before it is history.

**What it was.** The most consequential production logic — the save-gate adjudication API
`adjudicate_checkpoint`, with `GateDecision`, `TrainSpec`, `resolve_train_spec`,
`derive_declared_block` and the controls registry — lived in `tools/live_save_gate.py`, a
script, not in the installable package. The consequences cascaded: a 95-line defensive
script-to-script import contract reaching into `real_checkpoint_probe.py` (T2#1); underscore-private
imports from package internals into tools, with `F401` drift tripwires, because the package's
public surface was too thin to import from (T2#2); a reimplemented safetensors header parser and
HF base-model loader inside the gate script (T2#3); MUST_FIRE defect-injection controls trapped in
two scripts, unreachable from the package (T2#4). An adopter who wanted the one thing here that
would help them — the actual decision logic — could not `pip install` it.

**What was delivered.** The decision plane moved into
`src/foundationscale/gates/adjudication.py` (2,546 lines). `tools/live_save_gate.py` is now a
450-line argparse wrapper. The move is documented in-file, including the non-obvious hazard it
creates for test authors: re-export binds a *name*, not the defining module's globals, so
monkeypatching `live_save_gate._probe_derive_declared` no longer changes what the moved functions
read. The suites that relied on that were migrated with the move.

**What remains — the residual finding.** The compatibility shim re-exports **98 names, 63 of them
private**, each carrying a `noqa: F401`. Two things follow:

- **The public contract is still not the public surface.** Tests, the mutation table's anchors and
  the launchers' call sites all still bind to `live_save_gate.<name>`. The extraction moved the
  code without yet moving the *dependents*, so `foundationscale.gates.adjudication`'s real API is
  still defined by what a script chose to re-export. 35 of the 98 are legitimately public
  (`adjudicate_checkpoint`, `GateDecision`, `TrainSpec`, `Verdict`, the gate classes); the other 63
  are implementation detail with an import statement pinning it in place.
- **`Any`, `Path`, `Protocol`, `TypedDict`, `dataclass` and `field` are among the re-exports** —
  stdlib names crossing a package boundary, which is a reliable signal that the shim was generated
  from "what did this file use" rather than "what does a caller need".

- **The move inverted T2#1 rather than retiring it — and this is the serious residual (finding
  #219).** The defensive script-to-script import contract still exists, but it now runs *inside the
  installable library*: `foundationscale.gates.adjudication` imports `real_checkpoint_probe` — a
  `tools/` module — at module-import time, through a dual-arm `try/except` that degrades to `None`
  on failure. `pyproject.toml` sets `packages.find where = ["src"]`, so `tools/` is **not part of
  the distribution**. Measured directly, with a positive control:

  | sys.path | `_PROBE_IMPORT_ERROR` | `derive_declared` bound |
  |---|---|---|
  | `src/` + `tools/` (repo checkout — positive control) | `None` | **True** |
  | `src/` only (what `pip install` gives you) | `ModuleNotFoundError('real_checkpoint_probe')` | **False** |

  With the helper unbound, `derive_declared_block` refuses: *"probe helpers unimportable … refusing
  to …"*. So the headline benefit of the extraction — an adopter can now `pip install` the decision
  logic — **is not yet realised**. The API imports, and then declines to decide. It fails closed,
  which is correct and is the reason this is P1 and not a blocker; but the refusal is indistinguishable
  from a legitimate abstention unless the operator reads the error string.

Safetensors parsing also still appears in four `tools/` modules alongside the package's own
readers, so T2#3's duplication is only partly retired.

**Priority after the fix: P1 for #219, P2 for the shim.** The blocking form of the original problem
is gone — the API imports. Two pieces of work remain, in order:

1. **P1 — make the decision path self-contained.** Move `derive_declared` and `run_alias_control`
   into the package (they are library functions living in a probe script), or declare `tools/` a
   packaged subpackage. Ship a test that imports `foundationscale.gates.adjudication` with only
   `src/` on `sys.path` and asserts `_probe_derive_declared is not None` — the negative leg above,
   inverted into a control, so this cannot regress silently.
2. **P2 — retire the shim.** Migrate dependents to the package path, delete the 63 private
   re-exports, keep the 35 public names as a deprecation surface.

---

## Theme 4 — Documentation describes a system that does not exist

**What it is.** This is not stale docs; it is anticipatory docs presented as fact.

- **`B2_scaling.md` presents unimplemented interfaces and the wrong entrypoint as the system.** Measured: `class ExecutionBackend` is defined in zero source files while appearing 3 times in docs, and the YAML "working config" it shows is loaded by nothing (F1_docs#2, CONFIRMED on the `ExecutionBackend` leg). The entrypoint leg has been **overtaken by events twice**, and is restated here for the second time (#245, then #246). `B2_scaling.md:196` declares `entrypoint: "foundationscale.train"`. When this bullet was drafted that module path did not exist; #224/#225/#226 made it import; and until #246 it imported but could not be executed — `python -m foundationscale.train` died with *no module named `foundationscale.train.__main__`*, which is the most misleading of the three errors, because it says the package cannot be EXECUTED rather than that the documented target does not exist. Two other forms ran throughout, and an earlier draft of this bullet named only one of them: the `foundationscale-train` console script from `[project.scripts]` (#224), **and** the submodule form `python -m foundationscale.train.cli`, which works because `cli.py` carries its own `if __name__ == "__main__"` guard. Calling the console script "the only runnable form" was itself half a measurement. #246 shipped the four-line `train/__main__.py` forwarding to `cli.main`, together with a four-leg control module (`tests/test_train_module_entrypoint.py`) whose MUST_FIRE leg deletes that file and demands red — so the documented form and the shipped form are now the same form by construction, not by convention, and the identity leg compares the two `--help` texts byte-for-byte. **What survives in this bullet is the shape**: B2 still presents a single-backend `transformers` CLI as the backend protocol its surrounding YAML implies, and `class ExecutionBackend` is still defined in zero source files.
- **`B1_architecture.md` §12 worked examples** are polished YAML with no not-implemented disclaimer, naming an entrypoint that does not exist (F2_docs#1, CONFIRMED).
- **`F2_docs#5`** (CONFIRMED): zero API-reference/module-reference headings across all 19 markdown files, and exactly one markdown file names `from foundationscale`.
- **`F2_docs#6`** (PARTLY_TRUE, corrected): the finding claimed no troubleshooting page; in fact one *does* exist, at `h100_validation/h100/LAUNCH.md`, scoped to the H100 launch plane. What is absent is one scoped to the *package*.

**Who it hurts and when.** Anyone evaluating the repo for adoption, and any contributor trying to locate the true API. The docs' own doctrine (F2_docs#2) states that "a doc that is wrong and present is a defect," and this doctrine is kept — but the shipped docs currently violate it.

**Failure mode in practice.** A reader internalises an architecture (`ExecutionBackend`, `train.py`) and writes plans, wrappers, and mental models against it. When they grep the tree they find no `ExecutionBackend` at all and a `train/` package with a different shape than the one documented — and now *every* document they have read is retroactively suspect. This review is not exempt: its own load-bearing "no training code" claim went stale the moment `train/` landed, and no instrument caught it (#245).

---

## Theme 5 — Model- and campaign-specific semantics are hardcoded into "model-agnostic" infrastructure

**What it is.** The explicit user constraint is a model-agnostic core with model-specific components isolated. The shipped tools and gates embed one campaign.

- **MoE classification is Gemma-config semantics hardcoded into the core manifest emitter** (T3_skeptic#0, PARTLY_TRUE, corrected): the affirmative flag is `text_config.enable_moe_block` (plus a top-level fallback), and the accepted routed-count keys are a closed three-name set inside the tool. A model family whose config uses any other key name or nesting gets a refusal whose only remedy is to patch the tool. The correction: the tool is *not* limited to two Gemma modes.
- **Expert-layout regexes and projection-family tables hardcode a closed census of Megatron/Mixtral/Qwen/Gemma-4/GPT-OSS projection spellings into library source** (C_gates_domain#1, PARTLY_TRUE, corrected): any other MoE projection spelling fails or is mis-priced, and the only fix path is editing the library. The fail-closed `UNKNOWN` behaviour is good doctrine; the *closed table* is the lock-in.
- **`tensors_per_expert_layer=2`, megatron-core FQN space, `iter_*` glob** frozen as architecture constants (T3_skeptic#5).
- **Campaign env prefixes (`FOXBRAIN_`) baked into CLI help and recording helpers** (T3_skeptic#3).
- **A hardcoded literal base-model directory as the default fallback in the general adjudication callable** (T2_lib_script_boundary#6, corrected form): `adjudicate_checkpoint` defaults `base_model_dir` to `$HF_MODEL` and then to a hardcoded estate path. A missed flag silently loads one specific base model in an otherwise model-agnostic tool.
- **RL-flavoured objective gates** (GSPO/GRPO vocabulary, kl_coef, trust-region) sit in the model-agnostic core package (C_gates_domain#2, PARTLY_TRUE, corrected): the correction is that the claim they are globally `@register`-ed with no RL-vs-pretraining scoping is *not* supported; the RL-specific fixtures and incident narratives are.

**Who it hurts and when.** The next model family, at first contact. The first team that wires in a non-Gemma MoE model, or a projection spelling outside the closed census, hits a refusal whose error text names the *extension point* but the extension point is "edit `src/foundationscale/gates/checkpoint_gates.py` and `tools/emit_run_manifest.py`" — i.e. fork the framework.

**Failure mode in practice.** Model-agnosticism is a directory layout, not a fact. The framework forces new-model teams to become co-owners of the gate codebase before they have saved a single weight.

---

## Theme 6 — Single-geometry lock-in: the topology registry has no H100/GB200/MNNVL profile

**What it is.** `src/foundationscale/topology.py` is one of the few modules actually built for an end user (the construction-time product check is explicitly flagged Keep — `A_front_door#4`). But the shipped profile table is thin:

- `PROFILES` is built from a `_PROFILE_DATA` table containing exactly **two** profiles — `slurm-generic` and `local-single-node` — **both with `mnnvl_available=False`** (A_front_door#5, PARTLY_TRUE, corrected). The correction matters: the strictness of `from_dict` validation and how central H100/GB200 is to the module's stated target are *not* established by the cited lines; what is measured is the absence of any MNNVL-capable or H100/GB200 profile.
- Deriving `dp` by hand is the only entry path (A_front_door#6); no convenience constructor.
- `partition_consistency` is exported at top level but is a corpus-scanning check with generic semantics (A_front_door#7, corrected: it has generic inputs/findings, though motivated by an estate measurement).

**Who it hurts and when.** The first user targeting actual production hardware — H100 SXM nodes, GB200 NVL72 trays — finds no named profile, so they must hand-model the cluster, derive `dp` by hand, and only then get the strict product-check validation.

**Failure mode in practice.** The one module that most directly serves an end user is the one that greets a user on the framework's headline hardware with a blank slot where their cluster should be.

**What this is *not*.** The absence is of *data*, not of mechanism. `ClusterProfile.from_dict`
already carries an `mnnvl_available` field, rejects unknown keys, and accepts a profile from a JSON
document or path; `profile_by_name`'s own error text says so — *"New clusters are added as data: one
dict in `_PROFILE_DATA`, no code changes."* So this is the cheapest item in the review that a user
actually feels: shipping H100-SXM, GB200-NVL72 and H200 profiles is adding dicts to a table and the
measurements to fill them, not a redesign. It is ranked as Missing rather than Redesign for exactly
that reason, and it is why M9 sits low in effort and high in perceived value.

---

## Theme 7 — Save-side-only verification: no resume/load-side contract

**What it is.** Verification exists only *after* save, never *before* consume. Measured: **no `RESUME`, `LOAD`, `BEFORE_LOAD`, or `RESTORE` member exists in the `Lifecycle` enum at all** (D_checkpoint_verify#0, CONFIRMED). The lifecycle bindings recorded in the census are `FIRST_SAVE` (7), `SAVE` (7), `STEP_ZERO` (4), `LAUNCH` (1), `EXPORT` (1), `PROMOTE` (1) — every gate fires on the write path. The `CheckpointMetadata` dataclass is defined with exactly three fields — tensors, format, origin — and itself carries no training-step/iteration binding (D_checkpoint_verify#2, PARTLY_TRUE, corrected: whether step binding exists elsewhere in the module is not established by the cited lines).

The readers exist. `open_weights`, `read_metadata`, `read_chunk`, and the self-validating archive machinery are all present and flagged Keep-worthy (`D_checkpoint_verify#10`, `D_checkpoint_verify#9`). The gap is in *when they are wired in*, not in capability.

**Who it hurts and when.** Anyone resuming from a checkpoint produced by an untrusted or partially-verified producer, and any export/promote consumer. The failure is late and asymmetric: the gate plane can declare a checkpoint good *at write time*, and there is no structural gate to check what is about to be *loaded*.

**Failure mode in practice.** A corrupt, aliased, or truncated checkpoint sails through save-side verification (or through no verification, because the engine never fires — Theme 2), lands in storage, and is consumed on resume with no load-side check to stop it.

---

## Theme 8 — Boundary duplication with demonstrated drift

**What it is.** Several critical invariants are stated in *two* places by *two* implementations that must agree, and in at least one case they are already observed to drift.

- **Three deliberate duplications between `dcp.py` and `dcp_meta.py`** must stay byte-identical or pre-flight and read diverge (D_checkpoint_verify#4, CONFIRMED): one sniff→`Format` enum path, one safetensors weight-map builder, one dtype-name table, each written twice with different torch-import constraints.
- **Duplicated measurement/context plumbing between the probe and the gate tools with one demonstrated drift** (T2_lib_script_boundary#5, PARTLY_TRUE, corrected): the gate normalises dtype via `str(tm.dtype).removeprefix('torch.')` while the probe's `build_context` passes `tm.dtype` unchanged. This is not hypothetical drift risk; it is a live behavioural difference between two tools that claim the same contract.
- **`coerce_context` bodies duplicated verbatim across all four checkpoint gates** (C_gates_domain#4), plus a duplicated docstring paragraph in `ExpertByteVolumeGate`.
- **Malformed-denominator guards** (`bool`-zero / non-int declared counts) reimplemented per gate (B_gate_engine#7, PARTLY_TRUE, corrected: established for `ExpertAliasGate`; duplication across other gates is not determinable from the cited lines, but the structural absence of a shared helper is).
- **Six-places-to-add-a-field serialization** across the manifest (E_provenance#3).

**Who it hurts and when.** The maintainer editing *one* of the two copies, and the user whose two tools now disagree on the answer by silent design. The dtype drift between probe and gate means an artifact can be measured "same" by one tool and "different" by the other, with neither one obviously wrong.

**Failure mode in practice.** Invariants that must be binary-symmetric (pre-flight vs. read, probe vs. gate) become left-asymmetric over time, and the failure surfaces as a *user-visible contradiction between two of the framework's own tools*, not as a clean gate refusal.

---

## Theme 9 — The docs root is an audit ledger, and its own doctrine keeps it that way

**What it is.** The user-facing documentation surface is occupied by engineering-governance material.

- **`SELF_AUDIT.md` is an engineering-governance ledger at the docs root** (F1_docs#3, CONFIRMED). Its evidentiary rigour is real (explicitly flagged Keep — F1_docs#7); the problem is its *location*.
- **README routes every newcomer into the 982-line `DECISIONS.md`** with a single "Read that first if you only read one" instruction and no persona-based routing (F2_docs#7, PARTLY_TRUE, corrected: the redaction/legend front matter is ~30 lines, ~3%, not the first 20%).
- **Identical ~10-line redaction preamble and a second identical post-draft-corrections block pasted across all files** (F2_docs#3).
- **A1/A2 audit a private legacy estate, not the shipped framework** (F1_docs#4); **D_roadmap.md is an internal project plan** (F1_docs#5).

**Who it hurts and when.** Any outside reader of this public repository, on first open. The README described by F2_docs#7 is the entry point of a training framework, and it hands every visitor a forensic audit of another estate.

**Failure mode in practice.** The project's evidentiary discipline is genuinely good work, but the audience targeting is inverted: the governance ledger is presented as user documentation while no installed-package API reference, capability statement, or quickstart exists at the same level of prominence.

---

## Cross-cutting observation

The damage ranking above is not a ranking of engineering quality. The inverse is closer to true: the highest-damage theme (1) coexists with the highest-quality internal code (the gates), and the inverse-quality themes (8, 9) concern material that is mostly sound. The structural problem is not that the wrong code was written, but that the *right* code — gates, controls, checkpoint readers, topology — was written and then packaged, documented, and bound into a lifecycle in a way that prevents any of it from being exercised by the audience the project says it is for. Fixing the top four themes (ship a training path, wire the engine in, promote `adjudicate_checkpoint` into the package, reflag the fictional docs) would already convert most of the existing investment into user-visible value without redesigning any working gate behaviour.

> **Census correction (applied post-draft).** This document was written against a census of 13,667 lines in `src/foundationscale/`. The T2 library/script boundary move has since relocated the 2,546-line checkpoint-decision API from `tools/live_save_gate.py` into `src/foundationscale/gates/adjudication.py`, and the fixes landed since have added the rest; `src/foundationscale/` now measures **18,915 lines**. Re-measured after the move, the structural finding is UNCHANGED: 0 files define `nn.Module`, call `backward()`, construct a `DataLoader`, or define `forward`, and 0 files import torch at module scope. The three `optimizer` hits and three `broadcast`/`all_*` hits are gate vocabulary (checkpoint optimizer-state fields; the registry broadcasting a context to gates), not NCCL collectives, and the single `torch.distributed` reference is a read-only DCP reader. What changed is that `src/` now holds real decision logic where it previously held none.
