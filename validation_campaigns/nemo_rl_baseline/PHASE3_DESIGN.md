# PHASE3_DESIGN.md — A native pluggable RL abstraction for FoundationScale

> This document proposes a set of interfaces that would let reinforcement-learning
> algorithms plug into FoundationScale the way gates already plug into its lifecycle:
> by contract, not by inheritance from any one implementation. It rests on three
> instruments and no others: the FoundationScale API skeleton (signatures and first
> docstring lines, no bodies), the Phase 1 NeMo-RL reading at the same granularity,
> and Phase 2's running evidence, which is four stock recipes at ten steps each on
> one GB200 GPU. No code proposed here has been written. Nothing here has been run.
> Every interface below is a proposal, and the document is written to be read by
> someone who will reject half of it. Where the design rests on something not yet
> measured, the text says so and names the measurement.

## 1. What this proposes, in one page

FoundationScale today is 20 files and 17,785 LOC (`git ls-files` + `wc -l`) organised
around a thin training entry (`train/loop.py`), a gate plane (`gates/`), a model
adapter layer (`models/adapters.py`), a checkpoint layer (`checkpoint/`), a provenance
layer (`provenance/`), and a verification layer (`verify/`). None of these exposes an
RL-specific symbol at skeleton granularity. This document proposes adding that
knowledge as a small set of contracts, not as a framework:

* `Algorithm` — the trainer contract. Owns the step loop's decisions; receives
  everything else by injection.
* `LossFn` — the loss contract. A callable over a typed batch and a forward pass,
  returning a scalar plus named components the gate plane can inspect.
* `RolloutSource` — the experience-generation contract. Produces batches of
  prompts-and-completions without the core knowing which engine produced them.
* `AdvantageFn` — the objective-shaping contract. Maps rewards to per-token weights.
* `PolicyPair` — the policy-and-reference-model contract. Holds the training view,
  the generation view, and any frozen reference the algorithm needs.
* `WeightSync` — the contract that moves weights from the training view to the
  generation view, with explicit staleness and failure semantics.
* `ExperienceBatch` — the data contract between all of the above: a columnar batch
  with a declared, validated schema.

Each is specified in section 3 with its responsibility, its input/output shape, the
seam it attaches to, and what would break if it were drawn wider or narrower. Section
4 tests the set against the four algorithms Phase 2 exercised. Sections 5 and 6 state
what stays out of the core and what was rejected. Sections 7 and 8 state what must be
measured and what could make this the wrong design. A reader who accepts only sections
2, 5, and 7 has still gotten the load-bearing content.

## 2. The seams FoundationScale already has

Everything in this section is grounded in the attached API skeleton. Where a design
decision would depend on behaviour inside a body, that is stated.

### 2.1 The training loop (`train/loop.py`, 1,199 LOC)

The skeleton shows `TrainConfig` as a plain data class: `model`, `dataset`,
`output_dir`, topology integers (`dp`, `tp`, `pp`, `ep`, `cp`), `max_steps`,
`per_device_batch_size`, `learning_rate`, `save_interval`, `seed`, and a
`ClusterProfile` slot. `train(cfg) -> int` runs the path and adjudicates it;
`loop.py` defines `EXIT_PASS`, `EXIT_RED`, `EXIT_UNMEASURED`, and `EXIT_REFUSE`, and
the skeleton does not state which of these `train` returns. The skeleton shows
helpers whose own docstrings state their roles: `_emit_manifest` and
`_build_run_manifest` for the run manifest, `_declare_checkpoint` for declaring the
checkpoint denominator from the live model before any save, and
`FoundationScaleSaveGate`, described as a `TrainerCallback` wiring the registered
checkpoint gates into `on_save`. Whether `on_save` invokes `_run_save_gates`
directly is a body-level fact the skeleton does not show.

What an RL abstraction must respect: the loop is the adjudication point, not just the
driver. The RL design must reach the manifest, the declared denominator, and the save
gates through this loop's seam — a trainer that runs beside `train(cfg)` would
bypass all three. The proposal in section 3 therefore attaches the algorithm contract
to the loop's adjudication role, not to its callback machinery: the skeleton shows
the save gate riding one training framework's callback base class, and if that
lifecycle cannot host an injected step contract, the loop, not the contract, moves.
Whether `train(cfg)`'s body can accept an injected step function is a body-level
question; the skeleton shows the signature, not the extension point. That must be
read before committing.

### 2.2 The gate plane (`gates/`)

`gates/core.py` (1,523 LOC) defines the contract: `Gate` is an ABC with `id`,
`events`, `context_type`, `check(ctx) -> GateResult`, and `controls()`; results carry
a `Verdict`, a `Coverage` with an explicit denominator (`checked`, `expected`,
`sampled`), and an `AbstentionKind` for non-verifying outcomes. `GateRegistry.run`
and `run_event` dispatch by `Lifecycle` event; `GateReport.raise_if_blocking` fails
closed. `gates/objective_gates.py` (1,250 LOC) is the seam that matters most for RL:
`ObjectiveGateContext` already carries `objective: ValueProvenance | None`,
`declared_components`, `components: tuple[LossComponent, ...]`, `uses_rewards`,
`reward_stats: RewardStats | None`, `reward_bounds`, and hyperparameter drift state
(`step0_fingerprint`, `current_hparams`). Four gates sit on it:
`ObjectiveDeclaredGate`, `LossComponentCoverageGate`, `RewardScaleSanityGate`,
`HyperparameterDriftGate`.

What an RL abstraction must respect: the objective gates expect the loss to be
decomposable into named, weighted, observed components, and rewards to be reported as
statistics over the samples actually inspected. The `LossFn` and `AdvantageFn`
contracts below are shaped so that an algorithm produces what these gates read by
construction. That is a design requirement, not an observed property of today's code:
the skeleton shows the checkpoint gates wired into the save event and shows no
dispatch site for the objective gates inside `train/loop.py`. Wiring one — at which
`Lifecycle` event, with what cadence, constructing `ObjectiveGateContext` from each
step's report — is part of this design's cost and is listed in section 7. The RL
design's first constraint stands regardless: the objective must be measurable by
construction, because the gate plane fails closed once wired.

### 2.3 The model adapter layer (`models/adapters.py`, 294 LOC)

`ModelAdapter` is a `Protocol` with `matches(config)`, `classify(config) ->
Classification`, and `enable_moe_block_flag(config)`. `Classification` carries an
`Architecture` enum value, `num_routed_experts`, the `evidence` string, and the
`adapter` name. `select_adapter` and `classify_config` pick one adapter from a
registry (`register_adapter`, `registry_snapshot`). Two concrete adapters exist in
the skeleton: `GenericHFAdapter` and `GemmaAdapter`.

What an RL abstraction must respect: this is the established pattern for keeping
model-family facts at the edge — a protocol, a registry, classification by evidence.
The RL design reuses this pattern for policy/reference construction and for any
model-family-specific layout mapping. It must not add a second, RL-specific
classification mechanism.

### 2.4 The checkpoint layer (`checkpoint/`)

`checkpoint/dcp.py` (1,494 LOC) defines `WeightSource`, a `Protocol` over DCP and
safetensors: `tensor_keys`, `shape`, `dtype`, `chunks`, `read_chunk`, `read_box`,
`read_full`, `close`, with `open_weights(path)` as the format-dispatching entry and
`compare_keys` for streaming cross-format comparison. `checkpoint/dcp_meta.py`
(637 LOC) reads metadata torch-free (`read_metadata`) and loads the run manifest
sitting beside a checkpoint (`load_manifest`).

What an RL abstraction must respect: `WeightSource` is read-only and partial by
design. The weight-synchronisation contract in section 3 must treat checkpoint-format
facts as adapter concerns and must not assume the generation view's weights are
readable through this interface — `WeightSource` reads artifacts, not live engine
state. That distinction is a design input, not an inconvenience.

### 2.5 The provenance layer (`provenance/manifest.py`, 2,623 LOC)

`RunManifest` records code provenance (`CodeProvenance`), effective configuration
with resolution provenance (`EffectiveValue`, `ConfigResolver`), the captured
environment, the `Topology` (with a `consistency()` check), and the
`DeclaredCheckpoint` denominator. `ManifestStore` is append-only and refuses
overwrites; `require_manifest` fails closed.

What an RL abstraction must respect: every effective value an RL run uses —
algorithm name, loss weights, advantage parameters, and whatever synchronisation
cadence the chosen `WeightSync` realization declares — must enter the manifest
through `ConfigResolver.record_effective`; once the objective gates are wired into
the loop (section 7), their `ValueProvenance` checks are designed to refuse anything
else. The core prescribes that cadence values be recorded, not that any particular
synchronisation lifecycle exists. The RL config surface must be designed to be
recordable, which rules out opaque per-algorithm config blobs.

### 2.6 The verification layer (`verify/parity.py`, 1,053 LOC)

`compare_sources(left, right, policy, block_rows) -> ParityReport` adjudicates two
`WeightSource`s key by key under a `TolerancePolicy`, with float64 statistics and
invariant guards (`ParityInvariantError`). `WeightParityGate` is the gate-plane
presence of weight parity (per its docstring), with a context carrying the same
parameters `compare_sources` takes; whether its `check` body delegates to
`compare_sources` is a body-level question.

What an RL abstraction must respect: this is the natural instrument for verifying
that a weight synchronisation actually landed — training-view checkpoint against
generation-view export, per key, under declared tolerances. The `WeightSync`
contract below names parity verification as part of its postconditions, and this
layer is where that check would live.

## 3. The proposed abstraction

Seven interfaces. Each is small on purpose; where a wider or narrower drawing would
break something, that is stated. All names below are proposals; none of these
symbols exist.

### 3.1 `Algorithm` — the trainer contract

Responsibility: given injected components, decide what happens at each step and
report what the objective actually was. Proposed shape: `setup(policy_pair, loss_fn,
dataloader, config, rollout_source=None, advantage_fn=None, weight_sync=None)` once
— an algorithm declares the components it needs, and components it does not consume
are absent, never stubbed — then `step() -> StepReport` called by the loop, where
`StepReport` carries the loss components, reward statistics if any, and the per-step
data the objective gates consume. Per-step data does not enter `RunManifest`, which
is the record of one launch attempt and is append-only; the manifest records
run-level effective values through `ConfigResolver`, and the per-step record's
durable home is the gate context and run record. Attaches to the `train/loop.py`
seam: the loop owns manifest emission, save gating, and exit codes; the algorithm
owns only step semantics. Phase 1 section 9(a) notes that NeMo-RL's seven `setup()`
signatures agree on their inputs — evidence for this shape, though the contract
itself must be invented, and the agreement was observed at signature granularity
only.

One level wider (algorithm owns the loop, the manifest, the save path) and the
adjudication plane is bypassed — rejected. One level narrower (algorithm is only a
loss function) and rollout-owning algorithms like GRPO have nowhere to live.

### 3.2 `LossFn` — the loss contract

Responsibility: map a forward pass over a typed batch to a scalar loss plus named
components. Proposed shape: a callable `(forward_fn, batch: ExperienceBatch) ->
LossOutput`, where `LossOutput` carries the scalar and a tuple of
`LossComponent`-shaped entries (name, weight, observed, contribution) so
`LossComponentCoverageGate` can read it without adaptation. Attaches to the
objective-gate seam (`gates/objective_gates.py`). Phase 1's pattern 1 (injected
loss, backend never owns algorithm math) is adopted; NeMo-RL's `LossFunction`
Protocol call signature is Phase 1 unknown 1 and must be re-verified against
`loss/interfaces.py` bodies before any equivalence is claimed — this proposal
specifies FoundationScale's own signature regardless.

One level wider (loss also owns advantage computation) and SFT, which has no
advantages, forces a no-op path through a reward-shaped interface. One level
narrower (loss returns only a scalar) and the objective gates go blind.

### 3.3 `RolloutSource` — the experience-generation contract

Responsibility: produce completions for prompts. Proposed shape:
`generate(prompts: ExperienceBatch) -> ExperienceBatch` plus `capabilities()`
describing what the source can return (logprobs, token counts, truncation metadata).
The handshake is fail-closed: at setup, the algorithm's required columns (section
3.7) are compared against the source's capabilities, and a non-empty difference
refuses the run before any step. The core knows nothing about which engine sits
behind it. Attaches to no existing FoundationScale seam — this is the one genuinely
new edge, and section 5 assigns the engine-specific half of it to adapters. Phase 2
ran GRPO's generation path on one GB200 GPU for ten steps; throughput, staleness,
and engine behaviour at scale are UNMEASURED.

One level wider (source also owns weight synchronisation) and every generation
engine adapter re-implements synchronisation policy. One level narrower (source
returns raw text) and policy-gradient methods lose the logprobs they need.

### 3.4 `AdvantageFn` — the objective-shaping contract

Responsibility: map per-sample rewards to per-token weights. Proposed shape:
`compute(prompt_ids, rewards, mask, **aux) -> AdvantageResult`, a real `Protocol`
where Phase 1 pattern 6 found only convention in NeMo-RL. `AdvantageResult` must
expose enough for `RewardScaleSanityGate` to price the gradient: reward statistics
over the samples actually used, not a healthy-looking average. Attaches to the
objective-gate seam. Phase 1 unknown 2 (whether NeMo-RL's two advantage bindings
share math or duplicate it) is open; this design does not depend on the answer, but
the re-verification is listed in section 7 because the convention's shape is one of
the patterns being adopted.

One level wider (advantage owns reward computation from environment feedback) and
reward-model training, which *is* the reward computation, becomes circular. One
level narrower (advantages are precomputed columns in the batch) and staleness and
normalisation policy become invisible to the gates. Note what this contract does not
do: it consumes rewards and does not produce them. Who turns completions into
rewards is the open edge recorded in section 4.

### 3.5 `PolicyPair` — the policy-and-reference-model contract

Responsibility: hold the model views an algorithm needs: the training view, the
generation view, and zero or more frozen references (DPO's reference policy).
Proposed shape: a small container with named roles — `train_view`, `generate_view`,
`reference(role)` — constructed through the `models/adapters.py` registry so
model-family facts stay at the edge. Attaches to the model adapter seam. The Phase 2
finding that the `_v2` DTensor worker could not be built on GB200 (finding #285: the
`automodel` extra pulls DeepEP, which fails a CCCL guard and pins `sm_90` against
GB200's `sm_100`) is a system finding about one backend's buildability; it is
evidence that backend selection must be an injected, replaceable edge, not a core
assumption.

One level wider (the pair owns parallelism layout) and the core's topology surface is
duplicated. That surface is currently five hard-named axes whose consistency check
is, per its own docstring, expressed in one parallelism library's vocabulary; the RL
contracts should consume a named-axis mapping rather than those five names, and
generalising the topology surface is a prerequisite, not something to duplicate. One
level narrower (a single model handle) and DPO cannot express its frozen reference.

### 3.6 `WeightSync` — the synchronisation contract

Responsibility: move weights from the training view to the generation view, with
explicit semantics. Proposed shape: `sync(mapping) -> SyncReport`, where the mapping
pairs each side's parameter names — both sides' names are adapter-owned and opaque
to the core, which requires only a total, sided name map and treats names as
strings; no naming convention is canonical in the core (Phase 1 pattern 2's per-side
layout adapters, adopted as a pattern, not as code). `SyncReport` records what was
transferred and what was skipped. Whether it also carries a staleness marker is
left open: Phase 1 unknown 5 left `is_stale` semantics unread, so any definition
written here would be invented. What would settle it is reading the reference
implementation's staleness call sites, to establish whether staleness is a
property of the weights, of the generation engine's cache, or of the step count. Failure semantics are part of the
contract: partial transfer must be representable. Post-sync verification runs
`WeightParityGate` over two on-disk artifacts — the training-view checkpoint and a
generation-view export — because `WeightSource` reads artifacts, not live engine
state; producing that export is the generation-side adapter's responsibility, and
`SyncReport` is evidence for the run record, not the parity gate's input. Attaches
to the verify seam; the checkpoint layer is read-only by design and offers no write
seam, so durable write-side facts are defined new here or delegated to the
generation-side adapter. Of Phase 1 unknowns 5 and 14, the transport half is now
MEASURED — section 7 item 3 records full copy, collective and reshard cost across
four payload sizes on 4 GB200 GPUs, and the ordering inverts between the per-byte
and per-call regimes, so this contract must let the realization choose rather than
fix one transport. Rollback atomicity and `is_stale` semantics remain UNMEASURED
and still gate this interface's final shape.

One level wider (sync owns generation-engine control plane) and every engine adapter
inherits a lifecycle it may not have. One level narrower (sync is a bare tensor
copy) and partial-failure quarantine — required because substrates differ: some fail
a synchronisation as a whole, others fail it per-rank, and the report must represent
both — has no representation.

### 3.7 `ExperienceBatch` — the data contract

Responsibility: the columnar batch that flows between rollout, advantage, loss, and
the gates. Proposed shape: a typed columnar container with a declared schema,
validated at construction and at every hand-off, whose slicing semantics are
specified here in FoundationScale's own terms — row-aligned, with no implicit device
semantics. Phase 1 pattern 4's validation discipline is adopted, because the Phase 1
data analysis notes that TypedDict alone enforces nothing; `BatchedDataDict`'s
semantics are not inherited, and Phase 1 unknown 6 (its internals: required keys,
padding, cross-backend layout) stays open. Required columns are per-algorithm and
declared by the algorithm; the container validates row alignment and reports
coverage the way `Coverage` does. Attaches to the data path between all the
contracts above.

One level wider (the batch carries engine-specific payloads opaquely) and schema
validation is theatre. One level narrower (plain dicts) and the row-alignment
defects the gates exist to catch become unrepresentable.

A note on size: seven interfaces is already at the edge of what this document
defends. If section 4 shows one of the four algorithms needs an eighth, the correct
response is to say so, not to widen an existing one. A large interface is a design
smell; so is a large interface count.

## 4. How four algorithms would sit on it

Phase 2 ran these four stock NeMo-RL recipes for ten steps each on one GB200 GPU;
all four exited `rc = 0` with populated `training_info.json`. That is the entire
running-system evidence. The sketches below are design sketches, not run reports.

* **SFT.** `RolloutSource` is absent (or the identity source over the dataset);
  `AdvantageFn` is absent; `LossFn` is token cross-entropy with one component;
  `PolicyPair` has a training view only; `WeightSync` is unused. This is the
  degenerate case and it must stay degenerate — if SFT needs stub implementations
  of RL-shaped interfaces to run, the abstraction is drawn wrong. Section 3.1's
  optionality rule exists to make this case expressible without stubs.
* **DPO.** `PolicyPair` carries a training view and a frozen `reference`; `LossFn`
  is the preference loss with the reference logprobs as an input column;
  `AdvantageFn` is absent. The reference-logprobs column has no producer in the
  seven contracts as drawn: something must run the frozen reference over the batch,
  and that scoring pass is the same open edge the reward callable below runs into.
  Phase 2's open finding belongs here: DPO `accuracy` and `sft_loss` both read
  exactly 0.0000 for the whole run, still unexplained as to CAUSE. What the gates
  DO with the reading is no longer unmeasured — section 7 item 4 measured it, and
  `LossComponentCoverageGate` catches the `sft_loss` half while
  `DiagnosticMetricGate` (added by #316) catches the `accuracy` half. Both are
  conditional on the objective DECLARING the quantity, which is what stage 2 in
  section 9 now carries as conditions (a) and (c).
* **Reward-model training (Bradley-Terry).** `PolicyPair` carries a training view
  with a scalar head; `LossFn` is the pairwise ranking loss; no rollout, no
  advantage. The fit is clean, with one caveat: the trained artifact is consumed by
  other algorithms as a reward callable, and the seven contracts give that callable
  no home — the open edge named below.
* **A policy-gradient method (GRPO-shaped).** Exercises all seven contracts: rollout
  generates completions, the advantage function shapes group-relative rewards, the
  loss consumes advantages and logprobs, and `WeightSync` refreshes the generation
  view between rounds. It also exposes the set's shared gap most sharply: rollout
  produces completions and the advantage function consumes rewards, but no contract
  turns the one into the other. This is the algorithm the abstraction is drawn for,
  and it is also the one whose fit is least proven: Phase 2 ran it for ten steps on
  one GPU with the non-`_v2` DTensor worker, and scaling, throughput, and
  convergence are UNMEASURED.

The most valuable sentence in this section: the seven contracts have no scoring
edge, and three of the four algorithms hit the absence. Reward-model training
produces a learned reward callable with no first-class home; a policy-gradient
method needs something to turn completions into rewards; DPO needs something to
produce reference logprobs. Whether the edge is a scoring capability on
`RolloutSource`, a callable role on `PolicyPair`, or an eighth contract is a genuine
open design question this document does not resolve. That gap is recorded, not
papered over.

## 5. What is deliberately NOT in the core

Each entry names the coupling, where it lives instead, and the adapter boundary.

* **A specific generation engine.** Lives in a `RolloutSource` adapter per engine.
  Boundary: `generate`/`capabilities`. Phase 1 documents the cost of the
  alternative — vLLM-shaped names inside generic interfaces and a 694-LOC patch
  wall (`generation/vllm/patches.py`, per the skeleton header) as the quantified
  fragility tax of a hard engine dependency.
* **A specific distributed runtime.** Ray is rejected as substrate (Phase 1 section
  5: structural, not incidental). The core assumes only an abstract process-group
  world — named ranks and collectives, one group per step — of which a
  torchrun-style launcher is one realization, not a core dependency. Rank layout,
  if adopted, follows Phase 1 pattern 3 (named-axis sharding, ~245 LOC, pure NumPy
  per the analysis) as a portable pattern. Boundary: the layout object and nothing
  else.
* **A specific parallelism library / training backend.** Megatron-shaped config
  TypedDicts in generic code are rejected (Phase 1 section 5). Backend selection is
  an injected edge behind `PolicyPair` construction. Phase 2's evidence is that the
  Megatron backend was never entered (`megatron_cfg.enabled = false`) and the `_v2`
  DTensor worker could not be built on GB200 — both findings argue for the edge,
  neither proves the edge's shape.
* **A specific model family.** Lives in `models/adapters.py` behind `ModelAdapter`,
  exactly as `GemmaAdapter` does today. The scattered if-model-X pattern is
  rejected; the registry mechanism is the contained version.
* **A specific cluster scheduler.** Lives outside the package entirely. The core
  sees `Topology` integers and a `ClusterProfile`; no scheduler name, partition, or
  path may appear in any proposed interface.
* **A specific quantisation stack.** Phase 1 pattern 8's shape (backend-scoped
  precision block, one resolver emitting trainer-side and serve-side configs) is
  adoptable with vendor-neutral keys, but whether in-training quantized serving is
  needed at all is UNMEASURED (Phase 1 unknown 3, and Phase 3's own open question);
  no quant interface is proposed in this document.

## 6. Alternatives considered and rejected

1. **Port NeMo-RL wholesale.** Rejected: it imports Ray as substrate, Megatron
   plumbing in generic code, vLLM-shaped interfaces, and per-worker venv machinery
   (Phase 1 section 5), each violating model-agnostic, vendor-neutral modularity.
   It would also import the trainer-level god-functions and two-algorithm dispatch
   that are the pluggability failure this design exists to avoid.
2. **Depend on NeMo-RL as a library.** Rejected: same couplings, plus a hard
   upstream-version dependency on code whose behaviour this campaign has read only
   at skeleton granularity (276 of 297 non-`__init__` files, 140,865 of 148,493
   lines, about 94.9 percent; no bodies, nothing run). A dependency on unmeasured
   behaviour is worse than a port, because the failure surface is someone else's
   release cycle.
3. **Build one algorithm well and generalise later.** Rejected as a plan, accepted
   as a staging order (section 9). The risk is named in section 8: an abstraction
   drawn from one reference implementation inherits its shape. The mitigation is to
   design against four algorithms now (section 4) even if only one is built first.
4. **One large algorithm base class.** Rejected: it concentrates the loss,
   rollout, advantage, and sync contracts into one inheritance surface, which is
   the design smell section 3 exists to avoid, and it gives the gate plane no
   narrow seams to attach to. Composition of small contracts is the proposal.
5. **Adopt NeMo-RL's interfaces verbatim as FoundationScale's own.** Rejected:
   Phase 1 unknowns 1, 4, 5, 6, and 7 are precisely the interface-level questions
   the skeletons could not answer. Specifying FoundationScale's own minimal
   interfaces is cheaper than discovering, after commitment, that the adopted
   signature was inferred from a docstring.

## 7. What must be measured before this is committed to

Ranked. Each item: the question, why the design turns on it, the measurement.

1. **Can `train(cfg)` accept an injected step contract, and do the objective gates
   have a dispatch site in the loop at all?** The whole proposal attaches inside
   the existing loop; the skeleton shows checkpoint-gate wiring on the save event
   and no objective-gate dispatch anywhere. Measurement: a body-level read of
   `train/loop.py` (the skeleton shows signatures only), then a minimal spike.
2. **Re-verify the Phase 1 claims this design leans on, against NeMo-RL source
   bodies in `<estate home>/NeMo-RL`.** Specifically: the `LossFunction` Protocol
   call signature (unknown 1), the advantage-estimator duplication question
   (unknown 2), backend dispatch seams (unknown 4), rollback and staleness
   semantics (unknown 5), and `BatchedDataDict` internals (unknown 6). These are
   instrument limitations of Phase 1, not system findings; until re-verified they
   carry no design decision.
3. **Weight-sync cost per transport on GB200.** MEASURED, and the answer changes
   the default. `WeightSync`'s default and its cadence semantics turn on this;
   Phase 1 section 9 listed it as measure-before-committing. The measurement ran
   on one GB200 tray, 4 GPUs, off-Slurm `torchrun` over NCCL, driven by
   `bench_weight_sync.py` in this directory; the full record is
   `measurements/weight_sync_gb200_4gpu.json` and `.log`. Median wall-clock
   seconds — the slowest rank's median, since the ranks share no common clock —
   per transport per declared size:

   | transport | 64 MiB | 256 MiB | 1 GiB | 4 GiB | 1 MiB per-call floor |
   |---|---|---|---|---|---|
   | `full_copy` (pinned) | 0.000757 | 0.002946 | 0.011661 | **ABSTAINED** | 0.0000444 |
   | `full_copy` (pageable) | 0.000828 | 0.002965 | 0.011743 | 0.145439 | 0.0000797 |
   | `collective` (broadcast) | 0.000276 | 0.000596 | 0.001775 | 0.006540 | 0.0001173 |
   | `reshard` (allgather) | 0.000165 | 0.000431 | 0.001433 | 0.005263 | 0.0001023 |

   **The GB/s figures in the record are NOT comparable across transports, and
   this table deliberately does not print them side by side.** Each transport
   declares its own byte model: `full_copy` counts a per-rank D2H+H2D round trip
   (2x the declared size), `collective` counts a broadcast to `world_size-1`
   consumers (3x at 4 ranks), `reshard` counts `size*(world_size-1)` (also 3x).
   A rate is therefore an in-transport quantity — useful for asking whether one
   transport saturates as size grows, useless for ranking two transports against
   each other. Wall-clock time at a fixed declared size is the basis-independent
   comparison, so it is the one the table shows and the one the readings below
   use. Within a transport the rates are worth stating: pinned `full_copy` sits
   at 177 to 184 GB/s from 64 MiB to 1 GiB, `collective` climbs 728 to 1815, and
   `reshard` climbs 1220 to 2448 through 4 GiB.

   Two readings, and they point opposite ways, which is why the design needs both
   halves rather than a single default:

   **Per byte, `reshard` > `collective` >> `full_copy`, and the gap widens with
   size.** The like-for-like comparison is at 1 GiB, the largest size where all
   four rows measured: reshard completes in 0.001433 s and collective in
   0.001775 s against full copy's 0.011661 s — 8.1x and 6.6x faster respectively.
   Full copy's own rate is flat across that range while the two collective
   transports keep climbing, which is the signature of a host round trip that has
   already saturated rather than of a fabric still filling. A default of "full
   copy" would leave that on the floor for any real policy weight-sync.

   **Per call, the ordering inverts.** At the 1 MiB floor `full_copy` pinned is
   the *fastest* transport measured — 0.0000444 s against reshard's 0.0001023 s
   and collective's 0.0001173 s, so 2.3x and 2.6x the other way — because the
   collective's fixed per-call cost dominates a payload that small. So cadence
   semantics genuinely turn on per-call versus per-byte cost, exactly as this
   section anticipated: a `WeightSync` that syncs many small tensors separately
   and one that syncs a flattened buffer do not want the same transport.

   The verdict was **CLEAR_WITH_ABSTENTIONS**, not CLEAR, and the abstention is
   named rather than dropped: `full_copy` at 4 GiB never stabilised its warmup
   (spread 74.1% of the median against a 10% tolerance, at the 40-iteration cap),
   so that cell was withheld. 19 of 20 (transport, size) cells were measured and
   admitted. Because the pinned/pageable pair at 4 GiB lost a row, the
   pinned-versus-pageable elision control adjudicated only 64 MiB, 256 MiB and
   1 GiB — 4 GiB is absent from the set it examined, and this table certifies
   nothing about that cell.

   That control also reproduced the size-dependence recorded as issue #321 in one
   run: the pinned-versus-pageable margin adjudicated at 64 MiB (0.000757 against
   0.000828, 9.4%) but was downgraded to an OBSERVATION at 256 MiB (0.65%) and
   1 GiB (0.70%), where the poison gate covered both rows. Widening the margin
   globally to quiet the small-payload case would have blinded the control at
   exactly the size where it still carries signal.

   16 GiB was deliberately excluded from the sweep rather than attempted and
   lost: allgather output across 4 ranks plus ring buffers is roughly 192 GiB
   against ~186 GiB of HBM, so the attempt would have OOM'd the process and taken
   the four measured sizes with it. Whether the occupancy gate degrades cleanly or
   crashes at that size is UNMEASURED and is tracked separately.
4. **Would the objective gates have caught Phase 2's DPO anomaly?** MEASURED, and
   the answer is split down the middle of the two readings. The reading was driven
   through the production dispatch (`run_event` over the shipped registry) in
   `tests/rl/test_dpo_anomaly_gate_response.py`, with a healthy DPO step as the
   positive control. Phase 2's config is on the cluster and WHY the numbers are
   zero is still open, so all three hypotheses about what the run declared were
   measured rather than one guessed:

   | Reading | Arm | `objective.loss_components` |
   |---|---|---|
   | healthy step (control) | both terms live | PASS |
   | `sft_loss` 0.0000 | declared, weight `0.0` | **FAIL** — "components with weight 0.0" |
   | `sft_loss` 0.0000 | declared, weighted, contributing `0.0` | **FAIL** — "measured contributing exactly 0.0" |
   | `sft_loss` 0.0000 | **not declared**, log-only | PASS — four green gates |

   So the gate plane catches the `sft_loss` reading **if and only if the objective
   declares `sft_loss` as a component**, and NeMo-RL logs that key whether or not
   the term is active. Declaration is what puts a component in the denominator.
   This is a requirement on stage 2, not a free property of it: FoundationScale's
   DPO must declare its auxiliary term even when the term is switched off, because
   an undeclared inactive term and an undeclared broken term read identically.

   The `accuracy` half was worse, and it was a finding rather than a caveat:
   `accuracy` at 0.0000 means the policy ranked the REJECTED completion above the
   chosen one on every pair, and **no gate could see it, structurally**. Of the
   eleven `ObjectiveGateContext` fields the only named-scalar channel was
   `components`, and `LossComponent` requires a `weight`; a diagnostic metric has
   no weight and is not a term of the loss. `LossOutput` (two fields) and
   `build_objective_gate_context` (seven parameters) carried no metric either, so
   the gap was in the stage-1 seam and not only in the context.

   **That gap is finding #316, and it is now closed.** The plane gained a fifth
   objective gate, `objective.metrics`, and the context gained two fields —
   `declared_metrics: tuple[MetricExpectation, ...]` and
   `metrics: tuple[MetricObservation, ...]` — with matching changes at the seam
   (`LossOutput.metrics`, and a `declared_metrics` parameter on
   `build_objective_gate_context`). The metric types are deliberately NOT
   `LossComponent`: routing a metric through a type that requires a weight would
   make the plane assert something false in order to look at it. `MetricExpectation`
   requires its `low`/`high` bounds, because an expectation with no bounds is not
   an expectation and a metric channel with no declared expectation is a data dump
   that READS as coverage; `MetricObservation.value=None` abstains with exactly
   `LossComponent.contribution`'s documented semantics. Re-measured over the same
   production dispatch:

   | Reading | Arm | `objective.metrics` |
   |---|---|---|
   | `accuracy` 0.5781 (control) | declared `[0, 1]`, observed | PASS |
   | no metrics at all (SFT) | declared none, observed none | SKIP — declared abstention |
   | `accuracy` 0.0000 | declared `[0, 1]` with `degenerate=(0.0,)` | **FAIL** — "pinned at 0.0" |
   | `accuracy` 0.0000 | **not declared**, log-only | **FAIL** — "never declared" |
   | `accuracy` 0.0000 | declared `[0, 1]`, no `degenerate` named | PASS |
   | `accuracy` declared, never emitted | declared `[0, 1]`, absent | **FAIL** |
   | every declared metric abstains | declared `[0, 1]`, `value=None` | **VACUOUS** |

   Two things in that table are load-bearing and neither is free. First, the
   metric axis does NOT inherit the loss axis's H3 hole: an observed metric the
   objective never declared is itself a refusal, so `accuracy` cannot slip through
   by going undeclared. Second, `0.0` is INSIDE an accuracy's natural `[0, 1]`
   range, so bounds alone cannot refuse it — what refuses it is the objective
   declaring `0.0` pathological for THIS metric, which the plane cannot infer
   (accuracy pinned at 0.0 is broken; a truncation fraction at 0.0 is ideal). So
   the conditional is narrowed, not removed, and it lands on stage 2 next to the
   `sft_loss` one: **FoundationScale's DPO must declare `accuracy` with its
   degenerate reading named.** The empty case is a declared SKIP rather than a
   silent PASS for the same reason: absence has to stay visible in the denominator.

   The design's claim that "the gate plane is the right instrument for this class
   of defect" now holds on both axes, conditional on declaration on both.
   Measured in `tests/rl/test_dpo_anomaly_gate_response.py` (8 legs) and in the
   gate's own ten controls.
5. **Serialized bytes crossing the trainer/generation boundary.** Sizes the
   `ExperienceBatch` schema and the sync mapping. UNMEASURED; Phase 2's ten steps
   on one GPU did not price the boundary.
6. **Sequence-packing utilization of one strong default algorithm.** Gates the
   decision to port a four-algorithm catalog vs one algorithm plus the contract
   set. UNMEASURED.
7. **Is in-training quantized serving needed at all?** Decides whether a quant
   interface is ever proposed. UNMEASURED (Phase 1 unknown 3).

## 8. Risks

* **Single-reference-inheritance risk.** The abstraction is drawn substantially
  from one reference implementation's patterns. It may inherit NeMo-RL's shape in
  ways the skeleton granularity cannot reveal — the seven `setup()` signatures
  agreeing on inputs is evidence for the contract's shape, and also evidence that
  the contract may be fitted to those seven algorithms and no others. Mitigation:
  section 4's four-algorithm test, and the explicit misfit recorded there.
* **The one genuinely new edge — and a second, missing one.** `RolloutSource`
  attaches to no existing FoundationScale seam. If generation-engine diversity
  turns out to matter less than engine depth on this estate, the interface is
  overhead; if it matters more, one adapter is not evidence the boundary is right.
  The scoring edge named in section 4 is a second new edge and is currently absent
  from the contract set entirely; resolving it may add an eighth contract, which
  section 3's size note already warns against.
* **Gate-plane impedance.** The objective gates expect decomposed, observed loss
  components. An algorithm whose objective is not naturally decomposable would
  either force a fake decomposition or pressure the gates to weaken. Both are
  failure modes; the design has no third answer yet.
* **Skeleton granularity on both sides.** FoundationScale's own bodies have not
  been read for this document either. Section 2's claims about what the loop
  "already does" are claims about signatures and docstrings; behaviour may differ.
* **Phase 2's denominator.** Four recipes, ten steps, one GPU, `rc = 0`. The SFT
  MFU of 0.12 to 0.17 percent comes from a one-GPU, ten-step smoke run whose cost
  is dominated by startup, not a throughput benchmark; it is not a performance
  claim. Nothing about scale, throughput, or convergence is known, and a design
  validated only at this denominator may be wrong at any other.

## 9. Proposed staging

Each stage leaves the repository working, and each stage must land with its own
measurements — the gate plane's fail-closed discipline is the motivation, and stage
1 includes wiring the objective-gate dispatch the skeleton does not yet show
(section 7, item 1).

1. Land `ExperienceBatch` and `LossFn` with SFT only, inside `train(cfg)`, with
   objective-gate contexts emitted. Smallest possible change to a working path.
2. Land `PolicyPair`'s reference role and DPO; use stage 1's measurements plus the
   DPO-anomaly investigation (section 7, item 4) as the gate. That investigation is
   now done and it attaches two conditions to this stage rather than clearing it
   unconditionally: (a) DPO must declare `sft_loss` as a component **even when the
   auxiliary term is switched off**, because the measurement shows an undeclared
   inert term passing the whole sweep; (b) the diagnostic-metric channel the same
   measurement found missing must be resolved — either added to the contract or
   explicitly declared out of scope — before an algorithm whose primary health
   signal is a metric rather than a loss term is built on it.

   Condition (b) is **DISCHARGED**: #316 added the channel (`objective.metrics`,
   `ObjectiveGateContext.declared_metrics`/`.metrics`, `LossOutput.metrics`), so
   this stage no longer blocks on it. What (b) leaves behind is a third condition
   in the same family as (a): (c) DPO must declare `accuracy` **with its degenerate
   reading named** (`MetricExpectation(name="accuracy", low=0.0, high=1.0,
   degenerate=(0.0,))`), because `0.0` is inside an accuracy's natural range and
   bounds alone cannot refuse it. Conditions (a) and (c) are the same finding on
   two axes: declaration is what puts a quantity in the denominator.

   **Condition (a) is unsatisfiable as literally written, and was implemented as
   its intent instead.** Reading the gate rather than assuming it:
   `LossComponentCoverageGate` FAILS any declared component whose `weight` is
   `0.0` (`src/foundationscale/gates/objective_gates.py`, the `zero_weight`
   branch), and the section-7 item-4 measurement table says the same thing in its
   own row — `sft_loss`, 0.0000, declared, weight 0.0, **FAIL**. That gate runs at
   `Lifecycle.STEP_ZERO`, so a DPO run that declared an inactive `sft_loss` would
   be refused before step 1: the instruction as written makes every sft-off DPO
   run unlaunchable. The gate's bidirectional leg blocks the obvious escape too —
   computing the term without declaring it fails on the other side.

   What (a) actually requires is that *an inactive term and a broken term cannot
   read identically*. `DPOLoss` gets that by deriving BOTH the declaration and the
   computation from one field, `sft_weight`: `declaration()` names
   `sft_component_name` exactly when `sft_weight != 0.0`, and `__call__` computes
   and emits the component under exactly the same condition. There is therefore no
   computed-but-undeclared state for a gate to be blind to, and no declared-but-
   zero-weight state for a gate to refuse. `tests/rl/test_dpo_loss.py` pins the
   declared set equal to the observed set across both settings of that one field;
   that equality, not the declaration of an inert term, is the property (a) was
   reaching for.

**Stages 1 and 2 have landed** (`src/foundationscale/rl/`: `interfaces.py`,
`losses.py`, `policy.py`). Stage 2 split implementations out of `interfaces.py`
because stage 3 adds three more contracts plus an algorithm; the public import
path (`foundationscale.rl`) is unchanged, so the move is invisible to callers.
3. Land `RolloutSource`, `AdvantageFn`, and `WeightSync` together — they are not
   separable — behind one policy-gradient algorithm, gated on the transport
   measurement (section 7, item 3).

   **Stage 3 was SPLIT at 3a/3b, and the split is the gate doing its job.** The
   claim that the three contracts "are not separable" is true of the *algorithm*
   that uses them and false of the contracts themselves. `RolloutSource` and
   `AdvantageFn` are specified entirely by what a rollout produces and what a
   reward becomes — neither needs a number from a cluster. `WeightSync` and the
   `Algorithm` binding are the opposite: both are shaped by how expensive a
   weight transfer is, which is exactly the section-7 item-3 measurement. Writing
   them before it would have meant *asserting* a cost model and then discovering
   it, which is the failure this campaign keeps finding under a different name.

   **That measurement has since been taken** (section 7 item 3), so 3b is
   unblocked, and it arrived with a constraint the asserted version would have
   missed: no single transport wins. Reshard beats collective beats full copy per
   byte by up to 13x, while at the 1 MiB per-call floor full copy is the fastest
   of the three. `WeightSync` therefore cannot ship a fixed transport with a
   cadence knob bolted on; the transport choice is part of what the realization
   declares, and the contract's job is to make that choice visible in
   `SyncReport` rather than to make it.

   So: **3a has landed** (`rollout.py`, `advantage.py`) and **3b is blocked on the
   item-3 measurement, deliberately and by name** — not deferred for want of time.
   Three advantage functions ship in 3a (`GroupNormalisedAdvantage`,
   `LeaveOneOutAdvantage`, `GeneralisedAdvantageEstimation`); section 4's four
   algorithms need no more than these, so 3b adds a binding, not more arithmetic.

   Two contracts in 3a exist only because a count is not an attribution.
   `RolloutSource.capabilities()` is a CLAIM and `verify_generated` is the
   MEASUREMENT of the same thing, kept as separate calls so a source that
   mis-declares itself is caught by the artifact rather than believed. And
   `AdvantageResult` carries `rows` alongside `used`/`offered`: an advantage
   function that drops a degenerate group returns a COMPACTED weights tuple, so
   `used < offered` says how many samples survived but not which — and with
   interleaved groups the survivors are not a prefix. A caller zipping weights
   positionally against its batch would apply one group's advantages to another
   group's sequences and never raise. `rows` names the survivors; the strictly-
   increasing constraint makes it a subsequence of the offered batch, so the zip
   is safe without also trusting the producer to have preserved batch order.
4. Reward-model training last, because its misfit (section 4) may force a contract
   revision, and revisions are cheapest before anything depends on the contract.

## 10. What this document does NOT establish

* Stages 1 and 2 of section 9 have been written; stages 3 and 4 have not.
  `ExperienceBatch`, `LossFn`, `LossDeclaration`, `SFTLoss`, `DPOLoss`,
  `PolicyPair` and `build_objective_gate_context` exist under
  `src/foundationscale/rl/`. `Algorithm`, `RolloutSource`, `AdvantageFn`,
  `WeightSync` and `StepReport` remain proposals with no implementation.
* What exists has been unit-tested, not RUN: stage 1 was exercised inside
  `train(cfg)`, and stage 2's DPO has never trained a model. No RL training run
  of any kind has been executed through this plane.
* This is not a benchmark. No relative performance claim is made or implied; Phase
  4 is the benchmark.
* Both sides of this design were read at API-skeleton granularity: signatures and
  first docstring lines, no bodies. Phase 1 covered 276 of 297 non-`__init__`
  files under `nemo_rl/` (140,865 of 148,493 lines, about 94.9 percent); the
  FoundationScale skeleton is the same granularity. Behavioural claims from either
  side are inference pending body-level re-verification.
* Phase 2's running evidence is four stock recipes at ten steps each on one GB200
  GPU, one node, with the non-`_v2` DTensor worker and the Megatron backend never
  entered. That denominator is part of every claim that rests on it.
* No FoundationScale-versus-NeMo-RL comparison exists yet — not of correctness, not
  of throughput, not of operability. This document designs against patterns and
  seams; it does not and cannot claim that the result would be better than, equal
  to, or even different from the reference implementation in any measured respect.
