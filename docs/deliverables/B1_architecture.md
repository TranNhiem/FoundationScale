# B.1 — FoundationScale Architecture Proposal (core)

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


**Status:** design proposal against the decision spine. Every "should" below cites the measured fact that forces it. Confidence markers: **[M]** measured · **[V]** verified in source · **[A]** census-backed · **[U]** unverified.

---

## 0. Ground rules this design inherits

Three findings bound everything that follows, and the architecture exists to make them structurally impossible to repeat:

1. **The objective is currently a property of the filesystem path.** **59** files define `reward_m1m4`, collapsing to **8** md5s **[V·post]** — of which **23 are live-tree copies (the shadowing surface, unchanged) and 36 are `results/<run>/repro/code_snapshot/` provenance copies**, i.e. the extra files are the audit trail, not more shadowing. (The earlier 24/6 count inherited `fs_inventory.json`'s extension filter — 9,905 indexed entries against **19,912** real `.py` files — the same filter that caused the κ miss. F7 again, on this document's own method.) They implement 4 gold policies with 3 distinct default behaviours, imported by bare name from CWD (`run_gspo.py:40`) across ~20 sibling directories **[V]**. Worse than divergence: `gspo_official4` changed objective *mid-run* (steps 0–890 under P2, resumed at step 800 under P4) inside one W&B run, one checkpoint dir, one audit file **[V]**.
2. **Every gate is structural; nothing is semantic.** A checkpoint that was 87.5% wrong (16 of 128 distinct experts, replicated 8×) passed rc=0, resume, healthy loss, tensor counts and dtypes for two full training runs; only a byte sum fired, and a byte sum cannot catch a permuted expert axis **[A]**. A post-draft sweep did catch it a different way — 200 of 205 exports are now verified tensor-by-tensor against their Megatron source, 0 DIFFER, no permutation anywhere **[V·post]** — but that sweep is a one-off script, not a gate, and **no HF export has ever been asked to emit a token [U]**. The claim of this section is unchanged and is if anything stronger: the program's assurance still stops at the byte level.
3. **Topology has three contradictory sources of truth** — `#SBATCH` directives, an internal `TRAIN_GPUS` variable that contradicts them, and a `DRYRUN` string-grep — which is why the genuinely-working single-GPU path (six launchers defaulting `TRAIN_GPUS=1`, 14 job logs printing `train GPUs=1(DP=1) TP=1 EP=1`, two reaching step 300 **[V]**) was invisible to directive-based analysis, and why off-Slurm works for exactly 1 of ~240 launchers **[V]**.

Two corrections this document must not regress on: the EP=8 expert fix **landed** 2026-08-05 (`Gemma4DenseMoE(MegatronModule)`, `gemma4_provider.py:445,457` **[V]**), and the DP>1 broadcast fix **landed** 2026-08-12 in `omni-bridge/sdpo_gemma4/run_gspo.py` (lines 815, 916 **[V]**). Both appear here as *gates*, not as open bugs.

---

## 1. Architecture overview

```
L6  CONTROL        typed config · run registry · provenance · telemetry · topology & launch backends · CLI
L5  ORCHESTRATION  stage graph (A0-A4, B1-B4) · services (rollout/judge/teacher/export) · checkpoint · eval gates
L4  OBJECTIVE      loss plugins (CE/DPO/SDPO/POLAR/GSPO/POLAR_A) · reward stack · teacher providers
L3  DATA           source -> render/template/mask -> sampler -> collate   (chat jsonl AND tokenized corpus, one contract)
L2  MODEL          architecture registry · providers · freeze & adapter policy · weight bridge (HF <-> Megatron)
L1  EXECUTION      backends: single-device | Megatron-Core (FSDP2 gated on Phase-4 measurement) · RolloutScope
L0  SUBSTRATE      torch · CUDA/NCCL · container · cluster profile

crosscutting:  CONTRACTS   (executable invariants; >=1 SEMANTIC gate per artefact class)
               PROVENANCE  (run manifest; identity = repo-qualified path + content hash, never a directory name)
```

Dependency rule: a plane may only import downward. In particular, **L4 must not import L6 config machinery**, because today's reward modules reading `os.environ` at import time (module-scope reads of `OMNI_GOLD_MISS_IS_BAD`, `OMNI_GOLD_GRADER` — set-after-import silently no-ops **[V]**) is precisely inversion of that rule.

---

## 2. L0 — Substrate

**Responsibility.** Describe the machine, nothing more: cluster topology parameters (partition names — the tree contains both `<partition>` and `<partition_alt>` **[M]**, counts of `#SBATCH` only), IB HCA lists, socket interfaces (the `mlx5` prefix-matching-all-eight-devices incident is recorded in `launch_gspo_g4moe26b_v2.sh` **[V]**), container image identity, GPU count per node.

**Non-responsibility.** No knowledge of models, objectives, or run semantics. L0 never sees a checkpoint.

**Interface.**

```python
class ClusterProfile(BaseModel):
    name: str                                   # not a hostname; identity is content
    partition: str
    gpus_per_node: int
    ib_hcas: list[str] = []                     # explicit list, never a prefix glob
    socket_ifname: str
    container_image: str                        # registry path + sha256
    max_wallclock: str                          # per-QOS, measured, not folklore
    paths: dict[str, str]                       # no /home/<user> literals below this line
```

**Extension mechanism.** Add a YAML under `substrate/profiles/`, validate with a probe job. **Reuse:** the NCCL/env preset blocks from the 8-tray launchers are real, verified operational knowledge; lift them into profile defaults. **Build new:** everything above the env block — which constants are data versus code is currently decided per-launcher **[V]**.

**Boundary contract.** At launch, emit the profile's content hash into the run manifest; assert the allocated world size equals what the profile and topology jointly imply — because today every launcher's DP/GA maths hardcodes `world=32` and silently misreports anywhere else **[V]**.

---

## 3. L1 — Execution

**Responsibility.** How a step is executed across devices: mesh construction, distributed optimizer, checkpoint save/load symmetry, and **one object that owns group semantics** so that rollout scope and reward scope cannot disagree.

**Non-responsibility.** Loss content, reward content, data content. L1 produces tensors and scopes; it does not know what rewards mean.

**Core interfaces.**

```python
class ParallelSpec(BaseModel):
    tp: int = 1; ep: int = 1; etp: int = 1; cp: int = 1; pp: int = 1
    # dp is DERIVED at launch from world/(tp*cp) etc. and is never user-stated,
    # because stated-and-derived disagreeing is how src=0 shipped.

@dataclass(frozen=True)
class RolloutScope:
    """Obtained ONCE from the backend, handed to BOTH the rollout client and the
       reward broadcast. The DP>1 defect (mb/sdpo_gemma4 'BUG FIX (2026-08-12)',
       lines 815/916) was two callsites computing this independently."""
    group: ProcessGroup
    src_global_rank: int

class ExecutionBackend(Protocol):
    name: str
    def scope(self) -> RolloutScope: ...
    def build(self, provider: "ModelProvider", spec: ParallelSpec) -> "ModelHandle": ...
    def optimizer(self, model: ModelHandle, spec: ParallelSpec) -> "OptimizerHandle": ...
    def save(self, h: ModelHandle, opt: OptimizerHandle, path: Path, step: int) -> None: ...
    def load(self, h: ModelHandle, opt: OptimizerHandle | None, path: Path) -> int: ...
```

**Extension mechanism.** A new backend registers via entry point (`fs.execution.backends`). Per D2, exactly two backends ship initially — `single_device` and `megatron_core` — because a Megatron TP=1/EP=1/DP=1 single-GPU path already exists and has run to step 300 (`sdpo_e4b/launch_sdpo_e4b_arm.sh:96`, logs `e4b_arm_1458..1473` **[V]**); FSDP2 is gated on a Phase-4 measurement showing it measurably improves laptop/single-node iteration. **[M]** zero current FSDP imports means deferral risks nothing. The cross-backend conformance test (matching loss curves on a tiny model) is required *regardless* — it is what makes the interface meaningful.

**Reuse.** Megatron-Core keeps TP/EP/CP, the distributed optimizer and torch_dist checkpointing — 97 first-party files import Megatron-Bridge and the stack works today at 30B-A3B MoE **[M][V]**; reimplementing it is not defensible (D1). The single-axis `TRAIN_GPUS`/`torchrun --nproc_per_node=$N --master_addr=127.0.0.1` knob becomes the canonical small-scale entry. **Build new:** the backend interface itself, and the un-forking of Megatron-Bridge — three vendored trees (`Megatron-Bridge`, `_vendor_bridge_expertfix`, `_vendor_bridge_g4fix`) selected by PYTHONPATH ordering with grep-preflights verifying the patches still exist **[V]** become a patch-as-plugin layer against a pinned upstream, with a test that fails when an extension point moves.

**Boundary contract (runs at build, STRUCTURAL).**

```python
def assert_parallel_aware_ancestry(model: torch.nn.Module, backend: ExecutionBackend) -> None:
    """D8.1. The entire expert bug was one wrong base class: Gemma4DenseMoE was
    torch.nn.Module, silently downgrading sharded_state_dict to a plain-torch
    flatten. Measured signature: 5.73 GB vs 45.70 GB expert bytes, 960 vs 0
    locally-indexed weight<N> keys [A]. This assertion would have fired at import."""
    base = backend.parallel_module_base  # MegatronModule for megatron_core
    for name, m in model.named_modules():
        if is_parallel_aware(m) and not isinstance(nearest_framework_ancestor(m), base.__class__ if isinstance(base, type) else type(base)):
            raise ContractViolation(f"{name}: parallel-aware submodule without {base} ancestor")
```

Plus the D14 startup invariant: assert the reward broadcast's `(group, src)` is object-identical to the rollout client's scope, and reject any broadcast whose src is a global rank not in the passed group.

> **Dissent (explicit).** The spine says "delete `accel/gspo/` rather than fixing it" (D14). I agree it is dead as shipped — unreachable at DP>1, two defects (hard crash at TP>1 via the global `src=0` broadcast on a TP subgroup; silent mistrain at TP==1) **[V]** — but I recommend **quarantine-with-tombstone, not deletion**: `omni-accel` has no `.git` at all **[M]**, so `rm` is unrecoverable, and the only surviving interpretation artefacts for jobs J18/J19 have meaning only against that tree. Move it to `attic/accel_gspo/` with a README stating the two defects and where the fixed code lives.

---

## 4. L2 — Model

**Responsibility.** Turning an architecture name into a buildable, freezable, bridgeable object; owning the per-architecture quirk catalogue.

**Non-responsibility.** Training loops, datasets, checkpoints-on-disk formats (L1 owns those). L2 must not know which loss will consume the model — because `VLMLoRA` hardcoding `model.vision_model`/`model.vision_projection` while `Gemma4VLModel` exposes `vision_tower`/`multi_modal_projector` (`MB/peft/lora.py:203-208`, AttributeError as written **[V]**) is what happens when adapter policy encodes anatomy by string.

**Core interfaces.**

```python
class ModelDescriptor(BaseModel):
    """One typed entry per architecture. Replaces knowledge currently scattered
    across collate patches, reward constants and export heredocs."""
    name: str                                   # repo-qualified, e.g. "fs://model/gemma4_vl_26b"
    eos_token_ids: list[int]                    # gemma4: [1, 106, 50]
    thinking_delimiters: tuple[str, str]
    hf_to_megatron_map: MappingSpec             # declarative, includes exceptions:
    key_exceptions: list[KeyException]          # e.g. gemma4 K==V: exactly 5 layers lack
                                                # hf v_proj; drop-only-those-keys, raise
                                                # on anything else (state.py:756 pattern)
    vision_token_invariant: str                 # e.g. "count(<|image|>) == prod(grid)/merge**2"

class FreezePolicy(BaseModel):
    vision_model: bool = True; vision_projection: bool = True
    language_model: bool = False; sound_encoder: bool = True; sound_projection: bool = True
    # These primitives exist TODAY: nemotron_omni_provider.py, consumed by
    # mb/sdpo_gemma4/run_sdpo.py and run_gspo.py, present in checkpoint run_config.yaml [V].

class WeightBridge(Protocol):
    def import_hf(self, hf_dir: Path, desc: ModelDescriptor) -> CheckpointRef: ...
    def export_hf(self, ckpt: CheckpointRef, desc: ModelDescriptor) -> ExportArtifact: ...
        # export_hf MUST run the byte-vs-index gate and the logit-parity probe and
        # REFUSE to return a promotable artifact until both pass. (See Contracts §10.)
```

**Extension mechanism.** Register a descriptor + provider + mapping spec; the bridge is generated from the spec rather than hand-maintained — because the EP=8 multi-rank exporter family (`export_moe_ckpt.py`, `export_moe_1tray.py`) was elaborate machinery built to route around a one-word bug, and once checkpoints were reshardable at save time, the shipped exporter collapsed to `WORLD_SIZE=1`, EP=1, one GPU, ~4 min for 51.6 GB **[V][A]**.

**Reuse.** The HF model definitions under `hf_src/` (Gemma4, Qwen3.5-MoE) and the Megatron-Bridge providers; the working single-GPU export mechanism wholesale; the freeze primitives. `modeling_gemma4_vl.py`'s 217-LOC variant — the one with `freeze()` at `:207-217` — becomes canonical. **Build new:** the descriptor/registry, the declarative mapping, and an explicit fork-adjudication step before any reuse of `judge_registry.py`-adjacent substrate (that file exists in 2 divergent variants over 23 copies, and `judge_pool.py` in 2 over 22 **[M]** — L4 inherits this problem).

**Boundary contract.** No `ModelDescriptor` may enter the registry without: (a) a `freeze()` capability test — because the 171-LOC `modeling_gemma4_vl.py` variant has **no `freeze()` method at all**, and resolving it does not fail, it trains everything and reports success **[V]**; (b) a recorded effective trainable-parameter partition per run — today *no executed run records which modules were frozen* **[V]**, which is ten lines and the highest-value single instrumentation addition in the system.

---

## 5. L3 — Data

**Responsibility.** `source → render/template/mask → sampler → collate`, for text and multimodal, chat-formatted and raw-corpus, under one sample contract.

**Non-responsibility.** Storage layout of checkpoints, service endpoints, reward semantics (it supplies fields; L4 decides policy — because today's `infer_task_type` path-substring heuristics silently steer *reward routing*, a cross-plane coupling **[V]**).

**Core interfaces.**

```python
class Rendered(BaseModel):
    token_ids: list[int]; loss_mask: list[int]
    supervision_spans: list[Span]                # constructed spans, not re-found by
                                                 # string search over rendered text
    media: MediaRefs | None

class Renderer(Protocol):
    """Tokenizer + chat template + loss-mask construction, ONE object, versioned,
    content-hashed. Search-based re-discovery of assistant spans
    (create_multiturn_loss_mask_by_search) is what produced 'ZERO supervised tokens'
    under a healthy loss curve [V]; rendering must construct spans directly."""
    def render(self, sample: Sample, mode: Literal["train", "serve"]) -> Rendered: ...
    def fingerprint(self) -> str: ...

class BatchSampler(Protocol):
    def __iter__(self) -> Iterator[list[int]]: ...
    def set_epoch(self, epoch: int) -> None: ...
    def state_dict(self) -> SamplerState: ...   # includes strata_signature + consumed-prefix
    def load_state_dict(self, s: SamplerState) -> None: ...

class TokenizedCorpus(Protocol):
    """The pretraining data plane. 0 of 1,344 first-party .py touch GPTDataset /
    BlendedMegatronDatasetBuilder / mmap corpora; 0 of 532 first-party .sh reference
    .bin/.idx/--data-path/blend [V]. This must be BUILT; the engine that consumes it
    is already vendored and proven driveable end-to-end by
    accel/train_resume_test_e4b.py (random init, MockGPTDataset, 12 steps at 7.52B,
    Slurm job J01) [V]."""
    def shards(self) -> list[ShardRef]: ...      # .bin/.idx with content hashes
```

**Extension mechanism.** New source = a `SampleSource` plugin emitting the schema (support both `conversations` and HF `messages` — a 152 MB converted tool corpus is orphaned today purely because it speaks `messages` while the loader requires `conversations` **[V]**). New mixing strategy = a sampler plugin.

**Reuse.** The stratified-temperature sampler design (largest-remainder allocation, caps/floors fixed point, DP-correctness by construction) and modality-homogeneous batching — both are the documented fix for cross-modality NCCL hangs (jobs J11–J12) **[V]**; the offline QC fleet (`fix_jsonl_tags.py` etc.) as orchestrated prep stages. **Build new:** the corpus blend path, span-tracking rendering, and a data-manifest recorder — because `OMNI_SFT_JSONLS` is an env-only corpus override whose absence silently falls back to a hardcoded 135k multimodal default **[V]**, and under it the entire Gemma-4 line has never seen the 9,499 agentic trajectories that exist on disk **[V]**.

**Boundary contracts.** Template/mask parity (D7) at launch: render a fixed probe set through `render(mode="train")` and `render(mode="serve")`, assert token-id equality and non-empty supervision; fail the launch otherwise. Media alignment per batch: assert the descriptor's `vision_token_invariant` and that no media sentinel sits in a supervised span — 23.42% of rows once silently lost vision supervision from missing markers **[V]**. Resume-prefix correctness: `load_state_dict` must restore *consumed prefix*, not epoch roll-over — the full-epoch fast-forward livelock is documented in `g4_sft/RELAUNCH.md` (doc-sourced, treat as **[U]** until reproduced, then gate it).

---

## 6. L4 — Objective

**Responsibility.** Losses, rewards, teachers. This plane owns the quantity being optimized and must be able to *assert its own identity*.

**Non-responsibility.** How teachers are served (L5), how gradients are reduced (L1), which directory code lives in (none — P8).

**Core interfaces.**

```python
class GoldPolicy(StrEnum):                       # D6: four source files become one enum
    SHORT_CIRCUIT = "short_circuit"              # gold -> 1.0        (2 copies today)
    POST_JUDGE_FLOOR = "post_judge_floor"        # A = max(A, threshold), default 0.6
    MISS_IS_BAD = "miss_is_bad"                  # opt-in, defaults OFF; no launcher exports it
    GRADER_PRIMARY = "grader_primary"            # mb/sdpo_gemma4 only; uses gold_extract.grade

class RewardProvider(Protocol):
    version: str                                 # semver of the ONE installed package
    def content_md5(self) -> str: ...
    def score(self, batch: RewardBatch, policy: GoldPolicy) -> list[RewardRecord]: ...

class TeacherProvider(Protocol):
    """Self-teacher, frozen co-resident anchor, or remote-logprob service.
    Memory, not code, is the wall for N>1 teachers: today's anchor is co-resident
    on the training mesh (run_polar.py:235-250) [V], and 9 co-resident 26B teachers
    do not fit. So the DEFAULT N>1 implementation is HTTP logprob service."""
    def logprobs(self, ctx: TeacherContext) -> LogprobBatch: ...

class ObjectivePlugin(Protocol):
    name: str; objective_version: str
    def configure(self, cfg: ObjectiveConfig) -> None: ...
    def loss(self, batch: TrainBatch) -> LossOut: ...
    def assert_identity(self, step0_metrics: dict[str, float]) -> None: ...
        # e.g. GSPO with a configured trust region must observe seq_ratio != 1.0
        # exactly; ratio identically 1.0 means pi_old = pi_theta.detach() — the
        # configuration under which 7 of 10 official arms ran [V]. Default
        # old_logp_source is FROZEN; "no trust region" must be an explicit named mode.
```

**Extension mechanism.** Entry-point-registered plugins; an experiment selects `objective: gspo@1.3.0` + a yaml stanza. Registry entries are immutable; changing semantics = a new version.

**Reuse.** `gold_extract.py` (pure-stdlib, precision-first, explicit abstain) becomes the default verifier in the shared package; its shadow-grade harness reproduces κ = 0.9824 (99.13% agreement over 1,724 decided pairs, 95.8% coverage **[V]**) *but the oracle is Kimi-K3 prompted with gold_extract's own rulebook — the raters are not independent, and the 76 excluded rows are the hardest strata* — so "certified" stays struck until a human-grounded agreement artefact (~8 person-hours) exists. `polar_a_loss.py`'s union-top-K / log-space-mixture / per-token-λ machinery (with its FIX-1..FIX-4 adversarial-failure comments) is extended from 2-member to N-member rather than rewritten — but first `polar_a_loss.py` and `facts_loss.py`, byte-identical twins **[V]**, are unified, and the canonical variants of `judge_registry.py`/`judge_pool.py` are adjudicated by reading diffs, not by copy count. `judge_pool.py`'s round-robin (`itertools.cycle`, load balancing, actively wrong for "matching teacher") is replaced by content-aware `pick()`, and `run_odpo.py`'s `models[0]` collapse (line 396) is deleted. Alignment-of-effort note: "FACTS" denotes two unrelated things today (the POLAR_A loss and the `sdpo_facts` reward variant **`[V]`**); the registry forbids name reuse.

**Boundary contracts.** (1) Gates fail **closed**: a verifier exception is a reward-0 fail, a `rule_checks` import failure is a launch failure — today both fail open **[V]**. (2) The gold floor must not promote a wrong answer via substring fallback — measured incident: gold `答案：25`, answer `15`, reward 0.9412 **[V]**. (3) `fill_in_blank`/`short_answer` hardcoded-abstain (`# BUG 3`) is surfaced as a declared routing decision in config, not an invisible default. (4) DP>1 reward-distinctness smoke at step 0: two DP groups with different completions must not receive identical reward vectors — the pre-fix driver ran 1,876 steps with 472 steps of `grad_norm` exactly 0.000 while logging `reward/mean=0.794, success=1.00` **[V]**; post-fix: 0 zero-gradient steps.

---

## 7. L5 — Orchestration

**Responsibility.** Stage graphs (A0–A4, B1–B4 as declarative configs), service fleets (rollout/judge/teacher/export), checkpoint lifecycle, promotion gates.

**Non-responsibility.** Loss math, sampler internals, NCCL. L5 composes planes; it implements nothing they own. **A new stage is a config, not a directory** — this is already nearly true: B1 projector init is expressible today as `freeze_language_model=True freeze_vision_projection=False` because `run_recipe.py` applies dotted overrides after the recipe builder and the provider reads the freeze flags **[V]**; what's missing is the caption corpus (the documented `Taiwan-formosa-VLM-caption-V1/data/` directory is *empty*; `Formosa-Vision/data/` holds 23 parquet shards nothing references **[V]**). Honest scope: 5 of 10 target stages have never executed; B1/B2 have zero logs, A0 has only a 12-iteration mock-data smoke, A1 zero logs, B4 a 5-iteration smoke with an empty checkpoint dir **[V]**. The framework must never count scaffolding as capability — mock-data checkpoints are byte-indistinguishable on disk from real ones.

**Core interfaces.**

```python
class Stage(Protocol):
    config_type: ClassVar[type[BaseModel]]
    produces: ArtifactClass                      # e.g. "megatron_ckpt", "hf_export"
    def run(self, ctx: RunContext) -> StageResult: ...

class ServicePool(Protocol):
    """Rollout/judge/teacher fleets as addressable, health-checked,
    weight-versioned services. Today they are ssh+tmux+enroot processes outside
    Slurm, on borrowed trays, invisible to sinfo [V]."""
    def endpoints(self) -> list[Endpoint]: ...
    def health(self) -> PoolHealth: ...          # includes served-identity check, not
                                                 # just liveness — an endpoint can be UP
                                                 # serving the WRONG model -> silent reward 0 [V]
    def weight_version(self) -> int: ...
    def refit(self, artifact: ExportArtifact) -> RefitResult: ...

class AdmissionPolicy(Protocol):
    def admit(self, req: RolloutRequest, pool: ServicePool) -> Decision: ...
        # Budgets by AGGREGATE KV TOKENS (concurrency x context), never request count.
        # Measured on this cluster's K3 endpoint during this audit: 16 concurrent
        # requests at ~250K context decoded at 10 tok/s aggregate; 24 concurrent at
        # ~37K context at 270-438 tok/s — a 27-44x swing from context length alone,
        # with num_requests_waiting == 0 throughout [M]. Request-count scheduling
        # is choosing a point on that cliff blindly.
```

**Reuse.** The verify-before-commit gate semantics from `sdpo_gate_r.sh` (staging replica, generation probes, advisory-vs-block split); the colocated single-tray topology from `smoke_refit_e4b.sh` (GPU0 train / GPU1 rollout / GPU2 export) generalized into the colocation *policy flag* (D10); the blue-green refit design. **Build new:** the service lifecycle manager (allocation-scoped, GPU-PID reaping built in — orphaned `EngineCore` workers surviving tmux kills to serve stale weights is a documented incident **[V]**), the stage graph, and the KV-budget admission.

**Boundary contracts.** Export artifacts are promotable only after the post-export semantic gate (§10). Every refit asserts the pool never drops below one live replica and stamps `weight_version` onto every rollout's audit record —

---

## 8. L6 — Control

**Responsibility.** Typed config, run registry, provenance, telemetry, topology + launch backends, CLI.

**Non-responsibility.** Numerics of any kind. L6 must not import torch — because the plane that computes and the plane that records have, historically, disagreed silently (a config can say `vllm` while the GSPO code keys behaviour on tensor presence, not the flag **[V]**).

**Core interfaces.**

```python
class TopologySpec(BaseModel):
    nodes: int; gpus_per_node: int
    parallel: ParallelSpec
    backend: Literal["megatron_core", "single_device"]
    wallclock: str
    # ONE declaration. Emits Slurm, off-Slurm-enroot, or bare-local from the same
    # source of truth — replacing #SBATCH directives, the contradicting TRAIN_GPUS
    # variable, and the DRYRUN string-grep that (by design) reconstituted today's
    # only off-Slurm launcher [V]. gpus_per_node here covers the real single-GPU
    # path's trick: request a full tray, restrict internally — which is exactly why
    # a directive census missed it [V].

class LaunchBackend(Protocol):
    def emit(self, job: JobSpec) -> LaunchPlan: ...      # slurm / enroot_offslurm / local
    def submit(self, plan: LaunchPlan) -> JobHandle: ...

class RunConfig(BaseModel):
    model: ModelRef; data: DataConfig; objective: ObjectiveConfig
    topology: TopologySpec; stage: StageRef
    # Composition is explicit overlay; Hydra is rejected (D4) because struct-mode
    # interpolation makes "what actually ran" harder to answer — the F2 failure mode.
    model_config = ConfigDict(extra="forbid")
```

**Reuse.** The `preflight_accel_config.py` config-as-delta assertion pattern, generalized. **Build new:** everything else. 166 distinct environment variables **[M]** collapse per D5: env carries secrets and cluster paths only; a CI lint gate fails any `os.environ` read outside `fs.control.env`. This is non-negotiable because `OMNI_GOLD_MISS_IS_BAD` was active for ≥12 ODPO runs while *nothing in the repo sets it* — exported by hand, propagated by `sbatch --export=ALL`, zero on-disk trace **[V]**.

**Boundary contract.** Before any GPU is allocated, L6 writes `resolved.yaml` + content hashes to the run dir; if that write fails, the launch fails. Sequencing deliberate: the resolved config must precede allocation, because today correctness-critical config (judge YAMLs keyed to literal hostnames) is discovered mid-run **[V]**.

---

## 9. Contracts — the gate catalogue

Rule: **every artefact class carries at least one SEMANTIC gate**, because the system's defining failure mode is structural-green/semantic-red [A][V], and it generalizes beyond checkpointing — the decode cliff ran with every structural signal healthy **[M]**.

| Gate | Asserts | Class | Runs at | Cost | Because |
|---|---|---|---|---|---|
| G1 module-ancestry | every parallel-aware submodule has framework-module ancestor | STRUCTURAL | build (import of model) | ms | one wrong base class silently stored 16 of 128 experts, save *and* load aliased **[A]** |
| G2 topology validity | TP divides KV heads; DP derived not stated; world == nodes×gpus; GBS % (DP×MBS) == 0 | STRUCTURAL | launch, pre-allocation | s | launchers hardcode world=32; geometry asserts live in per-launcher bash and have drifted **[V]** |
| G3 template/mask parity | trainer vs serving render equal token-ids, non-empty supervision | SEMANTIC | launch | s (CPU) | stock template stripped CoT → zero supervised tokens under healthy loss **[V]** |
| G4 objective identity | configured trust region actually binds; reward gates fail closed | SEMANTIC | step 0 | free (reads metrics) | 7/10 official GSPO arms ran ratio ≡ 1.0 exactly **[V]** |
| G5 DP>1 reward distinctness | different completions ⇏ identical reward vectors; broadcast src ∈ group; scope object-identical | SEMANTIC | step 0 + periodic | one extra group today | 472 steps of grad_norm == 0.000 logging success **[V]** |
| G6 first-checkpoint invariant | expected param bytes per EP rank; zero locally-indexed expert keys; EP-reshardability | STRUCTURAL | first save of every run | s (DCP .metadata) | the byte-sum check that caught the expert bug ran two runs too late **[A]** |
| G7 export byte-vs-index | sum(safetensors) vs index `total_size`; tensor count | STRUCTURAL | export | s | `BAD_INCOMPLETE_..._1723` carries a *perfect* 1013-tensor/60-expert index over 11.5% of required bytes **[A]**; gate exists inline in `export_v2_fullft_sbatch.sh:58-70` but is absent from `export_many.sh`, which passes `--not-strict` and checks only `du -sh` (line 49) **[V]** → promote to a library `verify_export()` the exporter calls itself |
| G8 post-export logit parity | HF vs Megatron logits + generation probe, few prompts, one GPU | SEMANTIC | export AND promote | ~5 min | Weight level is now closed retroactively (200/205, 0 DIFFER, no permutation **[V·post]**) — this gate is the *semantic* half, and **no export has ever had it [U]**. Assert `top-1 agreement == 1.0` and `KL < 1e-3` on a fixed batch, and emit a coverage count so an empty comparison cannot pass as agreement |
| G9 media/supervision alignment | vision-token invariant per batch; no sentinel in supervised span | SEMANTIC | every collate (cheap mode) + launch (full probe) | <1% step | 23.42% silent vision-supervision loss **[V]** |
| G10 resume-prefix | sampler restores consumed prefix; first post-resume batch == reference | SEMANTIC | resume | one batch re-render | consumption-unit drift documented in `MUON_RUNBOOK.md` #10 **[V]** |
| G11 KV-budget admission | admits by aggregate KV tokens; emits throughput-per-KV-token telemetry | SEMANTIC | rollout admission, continuous | free | 27–44× throughput cliff invisible to all structural signals **[M]** |
| G12 reward certification | publishing any grader claim requires a human-agreement artefact with independent raters | SEMANTIC | publish | ~8 person-hours once | κ = 0.9824 exists but raters share a rulebook; "certified" remains unearned **[V]** |

Audit-method rider folded into contract tooling: the κ artefact was missed by a full audit pass because the evidence inventory excluded `.jsonl` **[V]**. Gate evidence must be written to indexed formats *and* the index must include data extensions.

---

## 10. Provenance — the run manifest

Identity is a content hash over the resolved run record, never a directory name (P8). The manifest is written by L6 before allocation and follows the run everywhere.

```json
{
  "run_id": "fsrun_01J...",
  "created_utc": "...", "user": "...", "cluster_profile_sha": "...",
  "config": {"resolved_yaml_sha": "...", "overlays": ["base/g4moe26b", "arms/gspo_tr"]},
  "code": {
    "repo": "foundation-scale",
    "vcs": {"git_sha": "...", "dirty_files": 0},
    "source_content_sha": "...",
    "imported_module_map_sha": "...",
    "bridge": {"tree": "...", "pinned_upstream_sha": "...", "patchset_sha": "..."}
  },
  "objective": {
    "plugin": "fs.objective.gspo", "version": "1.3.0",
    "reward": {"package": "fs.reward", "version": "2.1.0", "md5": "...",
               "gold_policy": "grader_primary"},
    "old_logp_source": "frozen", "trust_region": {"eps_lo": 3e-4, "eps_hi": 4e-4}
  },
  "data": {"corpus": [{"path": "...", "md5": "...", "rows": 20769}],
           "renderer_fingerprint": "...", "sampler": {"seed": 42, "strata_signature": "..."}},
  "model": {"base_ckpt_content_sha": "...", "descriptor": "fs://model/gemma4_vl_26b@3"},
  "freeze_partition": {"trainable_param_count": 0, "frozen": ["vision_tower", "multi_modal_projector"]},
  "topology": {"nodes": 6, "gpus_per_node": 4, "tp": 2, "ep": 1, "dp": 12,
               "launch_backend": "enroot_offslurm"},
  "environment": {"captured_env_allowlist_sha": "...", "nccl": {...}},
  "gates": {"passed": ["G1","G2","G3","G4"], "waived": []}
}
```

Why each field exists is a measured incident: `bridge.patchset_sha` because three vendored trees are PYTHONPATH-selected with no record of which a run used **[V]**; `environment` because the objective once lived only in an interactive shell **[V]**; `freeze_partition` because the trainable partition is currently unrecoverable even for executed runs **[V]**; `base_ckpt_content_sha` because base-checkpoint validity is today enforced by a *name-suffix* guard (`launch_g4moe26b_sft_32k.sh:164-167` requires `-expertfix` **[V]**) — names are not identity (F7). `code.vcs` is optional-by-design: `omni-accel` is not a git repository at all (**[M]**, 3,010 files/681,440 LOC), so `source_content_sha` + the imported-module map is the fallback identity, and migration of first-party code into one git repo is a Phase-0 prerequisite, not a choice.

**Propagation.** `run_id` is embedded deterministically in: the W&B run name *and* id (hash-derived, resumable), the checkpoint dir name, the scalar-JSONL header (not just the filename), the judge-audit JSONL header (today the audit's `decided_by` vocabulary is the only objective fingerprint — keep it, but as corroboration, not sole record **[V]**), and the export manifest. A W&B curve is then traceable to the exact objective that produced it, replacing the JobID-keyed log-regex reconstruction in `build_manifest.py` — whose own header warns some runs are silently non-comparable **[V]**.

---

## 11. Directory structure

```
foundation-scale/
  pyproject.toml                      # ONE installed package; nothing imported from CWD
  fs/
    substrate/     profiles/*.yaml (cluster as data)
    execution/     backend.py, megatron.py, single_device.py, scope.py
    model/         registry.py, descriptors/, bridges/, freeze.py, adapters/
    data/          sources/, render/, samplers/, collate/, pretrain/ (bin/idx plane), qc/
    objective/     losses/{ce,dpo,sdpo,polar,gspo,polar_a}.py
                   reward/  (versioned; gold_extract.py lives here)
                   teachers/{self,anchor,remote}.py
    orchestration/ stages/, services/, checkpoint.py, promotion.py
    control/       config schema, run_registry/, topology/, launch/{slurm,enroot,local}.py, cli.py
    contracts/     gates.py  (every gate is an importable, unit-tested function)
    provenance/    manifest.py
  recipes/         experiments as config deltas (see below)
  tests/           conformance/, gates/, smoke/ (DP>1 distinctness test lives here)
  attic/           quarantined clone trees, tombstoned (see Dissent, §3)
  docs/            # policy: superseded docs are MOVED to attic, not banner-prepended —
                   # EXPERT_SAVE_BUG.md and EXPORT_STATUS.md both produced false
                   # findings in this very review by remaining present and stale [A]
```

Deliberate aggregates: `objective/polar_a.py` absorbs both byte-identical twins; `data/qc/` absorbs the `fix_jsonl_tags`/`scan_*`/`predecode` fleet as orchestrated stages; there is **no** `sdpo_gemma4/` name anywhere, because that name denotes two different reward semantics *and* two different broadcast-corrections across the two repos **[V]**.

---

## 12. Worked examples — experiment as config delta

**SFT run.** `recipes/sft/g4moe26b_16k.yaml`:

```yaml
stage: sft.vlm
model: {ref: "fs://model/gemma4_vl_26b@3", base_ckpt: "artifact://gemma-4-26b-it-expertfix"}
freeze: {vision_model: true, vision_projection: true, language_model: false}
data:
  source: {type: chat_jsonl, manifest: "data/taiwan_aiec_v3.manifest.yaml"}   # hashed file list
  renderer: {family: gemma4, keep_cot: true, supervise_turn_end: 106}          # collate.py:1141/1260 semantics, as config
  sampler: {type: stratified_temperature, tau_task: 0.3, seed: 42}
objective: {plugin: fs.objective.ce, label_smoothing: 0.0}
topology:
  nodes: 8; gpus_per_node: 4
  parallel: {tp: 8, ep: 32, etp: 1, cp: 1, pp: 1}
  launch_backend: slurm
train: {lr: 5.0e-5, min_lr: 5.0e-6, schedule: cosine, gbs: 16, mbs: 1, iters: 2467}
gates: {require: [G1, G2, G3, G6]}
```

Note what this replaces: a ~350-line launcher whose 31B sibling *silently ignored* the `LR=` knob and trained 10× colder **[V]**, plus a `OMNI_SFT_JSONLS` env override whose absence silently loads a different corpus **[V]**. Here every behaviour-affecting value is in one typed document, validated pre-allocation.

**RL run.** `recipes/rl/g4moe26b_gspo_v4.yaml` — the *entire experiment*, as a delta:

```yaml
extends: recipes/sft/g4moe26b_16k.yaml          # same model family, same renderer hash REQUIRED
stage: rl.online
objective:
  plugin: fs.objective.gspo@1.3.0
  old_logp_source: frozen                        # explicit; the 2026 default was 'self' -> ratio==1.0 [V]
  trust_region: {eps_lo: 3.0e-4, eps_hi: 4.0e-4}
  reward:
    package: fs.reward@2.1.0                     # md5 stamped into manifest
    gold_policy: grader_primary                  # enum, not an import-time env var [V]
    judge:
      registry: judges/prod_registry.yaml        # endpoints resolved at launch, not hostnames in YAML
services:
  rollout: {replicas: 8, engine: vllm, admission: {budget: aggregate_kv_tokens}}
  refit: {policy: blue_green, every_steps: 200}  # never 0 by default: REFIT_EVERY=0 drove
                                                 # off-policy collapse jobs J20/J21/J22/J23 [V]
topology: {nodes: 6, gpus_per_node: 4, parallel: {tp: 2, ep: 1}, launch_backend: enroot_offslurm}
gates: {require: [G1, G2, G3, G4, G5], covenant: byte_identical_baseline_at_N1_teacher}
```

The last line encodes the repo's own best discipline: today's `anchor_enabled=False` dispatches to the untouched base loss so the ablation is provably unchanged **[V]** — any multi-teacher extension must keep an N=1 configuration byte-identical to the single-anchor path, or the ablation that would prove multi-teacher worth it becomes unmeasurable.

A/B isolation — the one genuine benefit of clone-per-experiment (`PRETRAINING_PIPELINE.md` defends it **[V]**) — is preserved as a *diffable YAML pair*, which is a cleaner A/B than a directory diff ever was. This is the load-bearing claim of P1: the framework must replace the **benefit** of cloning, or it will be cloned around.

---

## 13. What this buys, as falsifiable claims

1. A correctness fix lands once, because there is one copy — and the ~48% fix-miss rate mechanism (11 filenames in multiple AST variants, e.g. `rule_checks.py` at 5 variants/24 copies **[M]**, strict first-party set) is designed out, not just deduplicated.
2. Any run re-executes from its manifest alone; the objective is recoverable from the manifest, including for runs launched on the off-Slurm backend.
3. The F3 failure modes become build/launch/save-time failures, with ≥1 semantic gate per artefact class (§9).
4. B1 projector init is a stage config + caption data adapter, exactly as D9 predicts — and the framework records this honestly against the fact that B1 has **zero logs to date** **[V]**.
5. The same code runs on 1 GPU (proven today via degenerate Megatron **[V]**) and at today's measured ceiling of 8 nodes / 32 GPUs / EP=32 **[M][V]** from one topology declaration. **Nothing above 32 GPUs is claimed**; >8 nodes remains [U] until a measured scaling run exists, and the vendored 64-node NVIDIA example launchers must be quarantined before they read as false evidence again.

**Explicitly unresolved and stated as unknown:** ~~numerical correctness of the expert fix~~ and ~~weight-level correctness of the exports~~ — **both closed [V·post]**: 3,840/3,840 experts bitwise identical with controls firing at 128 and 112, and 200 of 205 export dirs verified at 0 DIFFER. In their place stands one narrower unknown, **semantic** correctness — no exported artefact has ever generated a token **[U]** — which G8 exists to close and which no amount of byte checking can reach; the co-located multi-GPU NCCL hang's root cause **[U]** — which the framework must not inherit as "DP1 by default"; and — **now closed** — whether `exports/fullft_iter2400_1tray_hf` was ever served: **resolved [V·post]: it is empty — 0 files, 0 bytes**, created by job J07, which FAILED 2m10s later with `ncclInvalidUsage — Duplicate GPU detected: rank 3 and rank 7 both on CUDA device` (the "1-tray" trick packs an 8-rank EP=8 world onto a 4-GPU tray). It died inside `read_run_config` → `broadcast_object_list`, **before a single weight was read**, so `save_hf_pretrained()` was never reached. *Method note, which matters more than the answer:* the investigator first checked whether the detector could fire, and it could not — known-good exports also score 0–1 log references and one positive control failed outright. The verdict therefore rests on the dispositive physical fact (zero bytes), not on absence of mentions. The one-GPU/handful-of-prompts Phase-1 gate now settles only the semantic question; the weight-level questions were settled with no GPU at all, by reading DCP chunks through the `.metadata` offset table. **That is worth recording as a design lesson:** the verification had been deferred on a cost estimate nobody had tested, and the estimate was wrong by the entire cost.