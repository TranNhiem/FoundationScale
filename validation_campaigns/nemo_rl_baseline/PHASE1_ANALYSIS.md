# PHASE1_ANALYSIS.md — NeMo-RL Phase 1 Analysis Record

> This document records a **code reading** of an operator-supplied NeMo-RL checkout at
> `<estate home>/NeMo-RL`, performed at **API-skeleton granularity**: for each source file the
> reader saw a header `=== <path> (<N> LOC) ===`, every class and function signature, and the
> first line of each docstring. It saw no function bodies and ran nothing.
> The corpus covers 276 file entries and 140,865 declared LOC of the 297 non-`__init__` files
> (148,493 lines) under `nemo_rl/` — about 94.9 percent by line count, from the skeleton-header
> count and `find` + `wc -l` on the checkout.
> Every claim about a name, a signature, a file, or a LOC count is grounded. Every claim about
> behaviour, performance, or runtime is inference from a signature or docstring, and is marked
> as such where it matters. The reader is an LLM whose output is a draft to be checked, not
> evidence; in Phase 2 the same model produced a corrupted spliced string in a digest.
> This is not a measurement and not a benchmark.

## 1. Scope and denominator

The corpus is per-file API skeletons produced by a skeleton extractor, bundled into 12
component bundles. Sizes below are files / declared LOC, counted from the `=== <path> (<N>
LOC) ===` headers.

| Bundle | Files | Declared LOC |
|---|---|---|
| `models` | 91 | 52,015 |
| `algorithms` | 29 | 35,537 |
| `data` | 54 | 10,500 |
| `utils` | 24 | 8,969 |
| `distributed` | 12 | 7,821 |
| `experience` | 8 | 6,108 |
| `environments` | 15 | 5,545 |
| `weight_sync` | 14 | 4,902 |
| `data_plane` | 13 | 4,617 |
| `modelopt` | 9 | 2,833 |
| `telemetry` | 5 | 1,345 |
| `evals` | 2 | 673 |

Whole-package denominator, from `find` + `wc -l` on the checkout: `nemo_rl/` is 343 `.py`
files, 150,857 lines; excluding 46 `__init__.py` files, 297 files and 148,493 lines. Line
coverage is 140,865 of 148,493 lines, about 94.9 percent. The skeleton's declared LOC agreed
with `wc -l` on 2 of 2 files spot-checked (manual check).

**Outside the corpus: 26 files.** 26 of the 297 non-`__init__` files matched no corpus entry
under an anchored path-segment-suffix test. They were not analysed and nothing in this
document covers them. Grouped: the whole `data/energon/` multimodal SFT path (12 files,
including `data/energon/sft_dataloader.py`, `data/energon/sft_worker.py`, and the
`task_encoders/` modules); four `experience/` rollout-routing files (`rollout_reassembler.py`,
`rollout_reassembler_actor.py`, `route_assembly.py`, `route_plan.py`); two `data_plane/` files
(`adapters/local.py`, `tq_token_sink.py`); three `models/generation/` support files; plus
`algorithms/sft_v2.py`, `environments/nemo_gym_request.py`, `modelopt/registry.py`,
`package_info.py`, and `transformers_compat.py`. Three were re-checked directly: the strings
`sft_v2`, `energon`, and `transformers_compat` appear zero times anywhere in the corpus, so
the gap is real, not a matcher artifact. The `energon` absence matters most: it is NeMo-RL's
multimodal SFT data path, and multimodal training is on FoundationScale's roadmap. This is a
finding about the INSTRUMENT's coverage, not about the SYSTEM.

**How the analyses were produced.** Eight tasks, one component scope each, against a
self-hosted Kimi-K3 endpoint; JSON-schema-constrained output; effort `high`; `--max-tokens
24000`; one invocation per task with its own output file. Six tasks returned on 2026-09-05;
tasks 6 (quantization/precision) and 7 (recipe/config) returned nothing on that run and were
re-run to completion on 2026-09-08. All eight report `finish_reason = stop` with no error.
Every analysis answers the same schema: `component`, `maturity`, `key_abstractions`,
`port_to_foundationscale`, `do_not_port`, `hard_dependencies`, `unknowns`, `summary`.

## 2. What NeMo-RL is, structurally

One page, drawn from the eight analyses. Each subsection names the abstractions the skeletons
actually show.

### 2.1 Algorithm surface (`algorithms`)

Seven first-class trainers, each a script-like trio of per-file `MasterConfig`, a `setup()`
returning a typed resource tuple (policy, interfaces, clusters, dataloaders, `loss_fn`,
logger, checkpointer, save-state), and a top-level `<algo>_train()` function: SFT, DPO, RM,
same-tokenizer Distillation, cross-tokenizer Off-Policy Distillation, GRPO (`grpo.py`, 5,976
LOC, the largest file in the bundle), and PPO. There is no shared base trainer, no algorithm
protocol, and no registry; code sharing between trainers is literal duplication (the analysis
reports an identical `dynamic_sampling()` in both `grpo.py` and `ppo.py`). The genuine shared
abstractions sit deeper: a `LossFunction` Protocol in `loss/interfaces.py`, per-loss config +
TypedDict data contracts in `loss/loss_functions.py`, a six-member advantage-estimator family
with a uniform `compute_advantage()` signature in `advantage_estimator.py`, and a
protocol-typed async staleness-sampler hierarchy in `async_utils/`. `single_controller.py` is
a runtime, not an algorithm, and its dispatch is an if/else over GRPO|PPO.

### 2.2 Model and policy backends (`models`)

A driver-side `PolicyInterface` ABC (`train(data, loss_fn, ...)`, `get_logprobs(...)`) over
`BatchedDataDict[OutputSpec]` typed dicts, with `loss_fn` injected from the algorithm layer; a
`ColocatablePolicyInterface` extension for train/generate colocation; a per-rank
`AbstractPolicyWorker` base; a parallel `ValueInterface` for PPO; and a separate
`GenerationInterface` with five named backends (vLLM, SGLang, TRT-LLM, Dynamo, Megatron) per
the models analysis — note the weight_sync analysis reports direct evidence only for vLLM,
SGLang, and Megatron-native generation and flags its own digest as truncated in the vLLM
section, so the TRT-LLM and Dynamo backends rest on one reader's signature list. Two training
backend families exist: DTensor/FSDP2 (`automodel/`, `dtensor/`) and Megatron (`megatron/`),
with near-duplicate fwd/bwd pipelines in `automodel/train.py` and `megatron/train.py`.

### 2.3 Distributed layer (`distributed` + `utils`)

Single-controller Ray-actor execution: `RayWorkerGroup` / `RayWorkerBuilder` dispatch with
axis-aware sharded-data methods, `RayVirtualCluster` over placement groups with
NVLink-segment-aware placement, and `NamedSharding` — a roughly 245-LOC pure-NumPy named-axis
rank layout with no Ray or CUDA dependency, per the analysis. `BatchedDataDict` is the
controller-to-worker data contract. `StatelessProcessGroup`, `HeldPortReservation`, and
`RefitAbortWatchdog` provide cross-world NCCL bootstrap, port safety, and planned-abort
semantics. `utils/` adds checkpoint async-finalization, a `LoggerInterface` fan-out, timers,
and a per-model-family FLOPs formula table.

### 2.4 Weight sync and generation (`weight_sync` + `models/generation`)

A `WeightSynchronizer` strategy seam (`init_communicator`, `sync_weights`, `is_stale`,
`reconcile_communicator`) with eight named implementations spanning IPC, collective,
NCCL-reshard, checkpoint-engine, and remote-sparse transports, plus a
`create_weight_synchronizer` factory. The `nccl_reshard` path carries the strongest
parameter-translation pattern: a canonical HF-name manifest mapped to per-side
`LocalParamSpec` entries, with `xferdtensor` (`xferdtensor.py` 342 LOC plus
`xferdtensor_python.py` 1,106 LOC, from the skeleton headers) for arbitrary-layout reshard. `RefitMembership` and `GenerationFleetHealth`
model the refit cohort: serving, stale, partial-weight, and dead shard states.

### 2.5 Data, experience, and data plane (`data` + `experience` + `data_plane`)

`DatumSpec` / `PreferenceDatumSpec` / `TaskDataSpec` TypedDicts form the sample boundary; a
`TaskDataProcessFnCallable` Protocol plus `register_processor` gives a named processor
registry; a message-log intermediate (`LLMMessageLogType`) serves SFT, preference, and
multi-turn RL. Rollout produces `Completion` objects grouped into `PromptGroupRecord`.
`SequencePacker` is an abstract length-to-bin interface with five subclasses, four of them
concrete algorithms; the fifth, `FirstFitPacker`, is described by its first docstring line as
a base class for the first-fit family, and a `get_packer` factory selects by name. The
`DataPlaneClient` ABC separates rollout producers from training consumers behind metadata plus
named columns, with a jagged wire codec and a `NoOpDataPlaneClient` for local operation.
`RolloutRecoveryLedger` tracks in-flight prompt ownership across checkpointing.

### 2.6 Environments and evals (`environments` + `evals`)

A compact `EnvironmentInterface` ABC — batched `step(message_log_batch, metadata)` plus
`global_post_process_and_metrics` — with a string-keyed `register_env` / `create_env`
registry. Implementations span math verifiers, a stateful code-execution environment, a
reward-model environment, a BM25 RAG environment, a sliding-puzzle game, and a NeMo-Gym
bridge. The analysis notes that concrete implementations diverge from the base signature
(e.g., an optional `return_extracted_answer` flag), so the contract is loose in practice.
`evals/eval.py` reuses environments for validation and provides `eval_pass_k` and
`eval_cons_k` estimators, but its setup returns a `VllmGeneration` directly.

### 2.7 Quantization and precision (`modelopt` + `telemetry`)

Precision is layered on by subclassing: `DTensorQuantPolicyWorker` / `MegatronQuantPolicyWorker`
add quantizer lifecycle (hide, disable, stats, refit hooks) to the standard workers, and
quantized serving registers ModelOpt NVFP4 configs through vLLM's public API. FP8 is a
backend-scoped config block (`fp8_cfg` inside `megatron_cfg`). Whether the path performs true
QAT with gradients during RL or only calibration PTQ is an explicit unknown — the skeletons
show quantizer toggling and calibration loops, not backward behaviour.

### 2.8 Recipe and config system (`examples/configs` + runners)

One self-contained YAML per run: algorithm-namespaced top-level sections, OmegaConf
interpolation, dual train-backend blocks (`dtensor_cfg` / `megatron_cfg`) toggled by an
`enabled` flag, generation backend selected by a string, and named processor/env indirection.
Composition is by filename (`grpo_math_70B_megatron_fp8.yaml`), not inheritance; algorithm
selection is by which runner script consumes the file.

## 3. Maturity assessment

Maturity labels are the reader's judgement from skeletons, not a test result. Evidence cited
is signature- or docstring-level.

| Component | Maturity (as reported) | One line of evidence |
|---|---|---|
| `algorithms` | production | Seven complete trainer trios plus a loss Protocol and sampler protocols; but no trainer-level contract exists. |
| `models` | production | Three generation backends confirmed behind one interface, two more (TRT-LLM, Dynamo) reported by one reader only; two training backend families; refit handshake on both sides. |
| `distributed` + `utils` | solid | Full dispatch/placement/refit/watchdog surface; the reader audited it as a reference, not a target. |
| `weight_sync` + generation | production | Eight synchronizer implementations plus fleet-health ledger; contract drift on `sync_weights` return types noted. |
| `data` + `experience` + `data_plane` | production | Feature-gated backends, no-op and metrics decorators, typed retry classes, checkpoint recovery. |
| `environments` + `evals` | solid | Small interface with real verifiers, but diverging implementation signatures and toy-grade members (Jaccard, sliding puzzle). |
| `modelopt` + `telemetry` | solid | Quant layered via subclassing on both worker sides; QAT-vs-PTQ behaviour unresolved. |
| recipe/config | production | Large config corpus in active use; but composition-by-filename and vendor-shaped schema. |

Disagreements and soft spots: the models analysis claims five generation backends while the
weight_sync analysis evidences three and reports its own digest truncated — treat TRT-LLM and
Dynamo as unconfirmed. The `algorithms` "production" label rests partly on docstrings (the
`LossFunction` Protocol shows no method members, so its call signature is unknown). The
`environments` "solid" label rests on interface shape; sandbox security of the code
environment is undetermined. The modelopt summary contains a self-corrected splice ("— see
json payload. — Wait, correcting:"), which is an INSTRUMENT finding: this reader's prose is a
draft, and its `summary` fields in particular should not be quoted as fact.

## 4. What is worth taking

Cross-cutting patterns that recur across several analyses, ranked. Everything in this section
is a **Phase 3 proposal**, not a decision.

1. **Injected-loss policy contract.** `PolicyInterface.train(data, loss_fn, ...)` with the
   loss supplied by the algorithm layer, over typed output dicts (models; echoed by the
   `LossFunction` Protocol and per-loss TypedDict data contracts in algorithms). Suits a
   model-agnostic framework because the backend never owns algorithm math. Proposal:
   reimplement natively with one parameterized inference call rather than NeMo-RL's
   method-per-output-type surface.
2. **Canonical manifest + per-side layout adapters for weight refit.** HF parameter names as
   the lingua franca, `prepare_refit_info()` metadata-first handshake, `HFToLocalParamMap` on
   both sides (models, weight_sync). Backend-neutral in spirit; proven across two training
   backends and at least three generation engines. Proposal: one `refit(transport, ...)`
   entry with pluggable transports, not eight interface methods.
3. **Named-axis rank layout as the single source of truth for group membership.**
   `NamedSharding` (distributed): ~245 LOC, pure NumPy, no Ray/CUDA coupling, per the
   analysis. Directly portable to a torchrun world; anchors axis-aware dispatch.
4. **Columnar batch contract with declared schema.** `BatchedDataDict` slicing/sharding/
   microbatch iteration (distributed, data, experience). Proposal: keep the semantics, add
   runtime schema validation — the data analysis notes TypedDict alone enforces nothing.
5. **Registry indirection by name.** Processor registry (data), environment registry
   (environments), generation-class factory (models), synchronizer factory (weight_sync),
   FQN-loadable custom samplers (algorithms). The recurring pattern is config names a string,
   one factory constructs it. Proposal: one typed, versioned registry mechanism reused across
   all these slots.
6. **Pluggable advantage estimation and staleness sampling.** Uniform
   `compute_advantage(prompt_ids, rewards, mask, **aux)` convention and the
   `PromptGroupSampler` protocol hierarchy with capacity validation (algorithms). Proposal:
   declare real Protocols where NeMo-RL has only convention.
7. **Failure taxonomy and planned abort.** `FailureClass` infra-vs-data retry classification
   (experience), `RefitAbortWatchdog` abort-vs-context-lost distinction (distributed),
   partial-weight quarantine in fleet health (weight_sync). Proposal: port the taxonomy; it
   matters more under torchrun, where one dead rank kills the job.
8. **Backend-scoped precision config.** `fp8_cfg` as a nested block inside a backend section,
   quant-worker-as-subclass, one resolver emitting trainer-side and serve-side quant configs
   (modelopt, recipes). Proposal: adopt the shape with vendor-neutral key names.

## 5. What must not be copied

Couplings, each with the FoundationScale principle it would violate. These are Phase 3
constraints on proposals, not decisions.

* **Ray as the execution substrate.** Every dispatch API in `distributed` returns or consumes
  Ray objects; `RayVirtualCluster`, `runtime_env`, and Ray-based monitoring are structural,
  not incidental (distributed analysis). Taking it would hard-depend FoundationScale on one
  orchestration vendor and conflicts with the Slurm/enroot torchrun plane. The analysis
  enumerates what a torchrun port loses: elasticity, per-actor Python environments, the Ray
  object store for large payloads, and actor-granular fault isolation.
* **Megatron plumbing in generic code.** `Megatron*Config` TypedDicts inside
  `policy/__init__.py`, a 2,500-LOC `megatron/setup.py`, an HF-to-MCore conversion cache, and
  `vocab_parallel_group` / `context_parallel_group` parameters threaded through the loss seam
  (models, algorithms). Violates vendor neutrality and hardware agnosticism; the seam
  locations are right, the parameter types are not.
* **vLLM-shaped interfaces.** SGLang engine names inside `ColocatablePolicyInterface`
  signatures, `evals/eval.py` returning `VllmGeneration`, vLLM source-compat patch walls
  (`generation/vllm/patches.py`, 694 LOC from the skeleton header; `policy/workers/patches.py`;
  a `monkey_patch_vllm_ray_executor` in quantization). Violates vendor neutrality; monkey
  patches are the quantified fragility tax of a hard engine dependency.
* **`uv`-managed per-worker virtual environments.** `venvs.py`, `prefetch_venvs.py`,
  `ray_actor_environment_registry.py` exist to colocate conflicting Python stacks in one Ray
  cluster (distributed analysis). Under per-component enroot containers the problem vanishes
  structurally; porting the machinery would import dependency-version roulette. Keep only the
  idea of a worker-group-scoped environment spec.
* **Model-family hard-wiring in core.** Per-family FLOPs formula table in
  `utils/flops_formulas.py`; `is_gemma_model` / rotary-embedding special cases in generic
  model code; Nemotron-specific preprocessing in `environments/nemotron_utils.py`;
  name-regex expert/TP classification in weight_sync. Violates model agnosticism. Adopt the
  flag/registry mechanism (`ModelFlag` enum is the contained version), never the scattered
  if-model-X pattern.
* **Trainer-level god-functions and two-algorithm dispatch.** The seven `<algo>_train()`
  functions and `algo_config(...) -> GRPOConfig | PPOConfig` (algorithms). Copying their shape
  imports the exact pluggability failure FoundationScale exists to avoid.

## 6. Hard dependencies

Dependencies the analyses identify, what each buys, and what taking it would cost. "Cost" is
stated against FoundationScale's model-agnostic, vendor-neutral, modular constraints.

| Dependency | What it buys NeMo-RL | Cost if taken |
|---|---|---|
| Ray | Single-controller actor orchestration, placement groups, object store, runtime envs | Structural vendor/framework lock; conflicts with the Slurm/enroot plane |
| PyTorch (+ DCP) | Tensors, autograd functions, process groups, checkpoint Statefulness | Low; already FoundationScale's plane |
| Megatron-Core / Transformer Engine / Megatron-Bridge | Large-model training backend, FP8 recipes, HF conversion | NVIDIA vendor stack; config dialect leaks into generic schema |
| NeMo-Automodel | DTensor-path checkpointing | NVIDIA vendor stack behind the DTensor backend |
| vLLM | Primary generation engine, refit endpoints, quant registration API | Hard engine dependency; patch walls evidence version fragility |
| SGLang, TensorRT-LLM, Dynamo | Additional generation backends | Each is a vendor-shaped adapter; TRT-LLM/Dynamo presence is single-reader evidence |
| NVIDIA ModelOpt | NVFP4 calibration and real-quant export | NVIDIA-only; must sit behind a FoundationScale quant interface |
| NCCL / NVSHMEM / NIXL / CUDA IPC / ZMQ | All refit transports | NVIDIA-gated interconnect; no non-NVIDIA transport evidenced |
| HuggingFace tokenizers / datasets / checkpoint naming | Canonical weight namespace, data ingestion | De facto standard; reasonable, but must enter via adapters |
| NeMo-Gym | External async environment servers | Heavyweight vendor bridge; adapter-only at most |
| OmegaConf/Hydra, `uv`, Pydantic v2 | Config interpolation, venvs, config models | Framework marriages; choose independently |
| tensordict, Pillow, NumPy | Data-plane wire container, media handling | Modest; wire container choice should be FoundationScale's own |

## 7. Open questions the skeletons cannot answer

Merged and deduplicated from the eight `unknowns` lists, ranked by how much a Phase 3 design
would depend on the answer. Each is answerable by a body-level read or a running system; this
instrument could not answer it.

1. The call signature of the `LossFunction` Protocol — `loss/interfaces.py` shows no methods
   beyond its docstring, so the loss contract cannot be assumed.
2. Whether the two advantage-computation bindings (`advantage_estimator.py` vs the
   SingleController `AdvantageConfig` field mapping) share math or duplicate it.
3. Whether quantized training is true QAT with gradients or calibration PTQ followed by
   fake-to-real export; and whether `fp8_cfg` works on the DTensor path or is Megatron-only.
4. How backend dispatch actually works: `Policy.__init__` worker selection, the
   `PolicyConfig` backend key, and the `create_weight_synchronizer` dispatch matrix are all
   outside the skeletons.
5. Refit cadence, `is_stale` semantics under failed syncs, and partial-transfer rollback
   guarantees — `mark_weights_partial` proves partial state is tracked, not that rollback is
   atomic.
6. `BatchedDataDict` internals: required keys, row alignment, padding conventions, and
   whether it hides cross-backend layout differences.
7. `EnvironmentReturn` fields and metadata-TypedDict lifecycles; the exact environment output
   contract cannot be quoted.
8. Code-execution sandbox isolation: process, filesystem, network, and resource limits are
   not determinable from `safe_open` / `safe_import` names.
9. Worker-group failure behaviour: whether actor loss mid-run is tolerated or fail-stop, and
   the memory/isolation policy behind `max_colocated_worker_groups`.
10. Whether the async GRPO path in `grpo.py` and the SingleController `AsyncRLConfig` path are
    one system or two coexisting generations.
11. Checkpoint atomicity across dataloader state, recovery sidecar, data-plane state, and
    model weights — each subsystem exposes save/load, but their joint atomicity is invisible.
12. Whether any cross-file YAML composition exists, and how runner scripts validate configs
    (only `TelemetryConfig` is shown as a BaseModel).
13. MoE expert-parallel placement: MoE-aware helpers exist, but no `ep` axis in
    `NamedSharding` is evidenced; EP may be Megatron-inherited.
14. Which refit transport is the production-hardened default at GB200 scale; the skeletons
    show full implementations of several but no selection evidence.
15. Provenance of specific files for design-derivative reimplementation (e.g.,
    `StatelessProcessGroup`'s unique-id protocol may carry upstream vLLM provenance).

## 8. What this does NOT establish

Stated as flatly as the results. The denominator of each claim in this document is 276 of 297
non-`__init__` files, read at signature granularity, by one LLM reader.

* **No function bodies were read.** Every behavioural statement above is inference from a
  signature or a first docstring line.
* **No runtime behaviour was measured.** Nothing here is a benchmark; no throughput, memory,
  convergence, or latency number appears or should be derived.
* **26 files are outside the corpus**, including the entire `data/energon/` multimodal path,
  `algorithms/sft_v2.py`, and `transformers_compat.py`. Any claim about multimodal data,
  second-generation SFT, or transformers compatibility is out of scope by construction.
* **The analyser is an LLM with a demonstrated splice failure.** In Phase 2 the same model
  produced a corrupted spliced string in a digest, and the modelopt summary in this evidence
  base contains a visible self-correction splice. Its prose is a draft to be checked, not
  evidence; only the schema fields grounded in signatures are load-bearing.
* **One analysis fabricated a number while citing the instrument.** A draft of Section 2.3
  stated that `xferdtensor` is 451 LOC "from the skeleton headers". The header reads 342, and
  451 appears in no header and in no task prompt. It was caught by auditing every three-or-more
  digit number in the draft against the 276-row header table and corrected before this document
  was written. This is the same splice class as the Phase 2 corruption, and it is the second
  occurrence: a number carrying an instrument citation is not thereby grounded.
* **Maturity labels are the reader's judgement**, not a test result. Two analyses disagree on
  the generation-backend inventory, and at least one maturity claim rests on docstrings.
* **Nothing here compares NeMo-RL to FoundationScale on any axis.** "Worth taking" sections
  are proposals about patterns, not findings about FoundationScale's needs measured against
  an alternative.
* **The only estate fact stated is that the hardware is GB200.** No claim in this document
  depends on any other property of the estate.

## 9. Hand-off to Phase 3

**Design first.** (a) The algorithm interface and registry NeMo-RL lacks — its seven
`setup()` signatures agree on their inputs (policy, clusters, dataloaders, `loss_fn`, logger,
checkpointer, save-state), which is evidence for the interface's shape, but the contract
itself must be invented. (b) The canonical weight manifest plus per-side layout adapters,
since dense-and-MoE refit correctness depends on it. (c) The named-axis rank layout and the
schema-validated columnar batch contract, which anchor everything else.

**Measure before committing.** Refit cost per transport on GB200 (full copy vs collective vs
reshard) before choosing a default synchronizer; sequence-packing utilization of one strong
default algorithm before porting the four-algorithm catalog; serialized-bytes crossing the
controller/worker boundary before designing the data plane; and whether quantized serving is
needed in-training at all before building the quant-worker layer.

**Re-verify against source bodies before they carry a design decision.** The `LossFunction`
Protocol call signature (unknown 1); the advantage-estimator duplication question (unknown
2); the QAT-vs-PTQ question (unknown 3); the backend dispatch seams (unknown 4); rollback and
staleness semantics (unknown 5); and the TRT-LLM/Dynamo backend claim, which rests on one
reader's truncated digest. Until these are confirmed against bodies, Phase 3 must treat them
as open, and must specify its own minimal interfaces rather than claiming equivalence with
NeMo-RL's.
