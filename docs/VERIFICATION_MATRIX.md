# Verification Matrix — FoundationScale

## The measurement that shapes everything

This matrix exists because of one measurement, taken before any of it was built, and the paragraphs below state that measurement in the past tense on purpose: #373 has since changed most of its numbers, and the next section says exactly which. The zeroes are kept rather than overwritten because they are the reason this matrix exists at all.

The instrument was a grep over `src/foundationscale` (excluding `rl/`), read against the code. The measurement: FoundationScale's training plane does NOT implement an inner loop. It builds a `transformers.Trainer` (`src/foundationscale/train/loop.py:2223`) out of `TrainingArguments`, from a kwargs dict (`src/foundationscale/train/loop.py:2103`) whose ENTIRE content was:

    output_dir, max_steps, per_device_train_batch_size, learning_rate,
    save_strategy, save_steps, seed, logging_steps, report_to,
    ddp_find_unused_parameters, (bf16|fp16) from cfg.precision, and
    save_safetensors when the installed transformers accepts it

TrainConfig's full field set was: model, dataset, output_dir, nodes, gpus_per_node, profile*, objective, precision, adapter*, max_steps, per_device_batch_size, learning_rate, save_interval, seed, dp/tp/pp/ep/cp, dry_run, launch_corpus.

The grep counted, at that time:

- `attn_implementation|flash_attention|sdpa` → 0 occurrences
- `torch.compile` → 0
- `get_scheduler|lr_scheduler|warmup` → 0
- `AdamW|Adafactor|SGD(` → 0
- `deepspeed|FSDP|fully_shard|ZeRO` → 2, both in `gates/` + `provenance/` (adjudicated/recorded, never built)
- `clip_grad|max_grad_norm` → 0

The consequence, and it is the whole point: every named axis — optimizer, CPU offload, DeepSpeed, gradient checkpointing, gradient accumulation, flash attention, LR schedule — was NOT absent from the RUN. HF Trainer supplies a default for each. It was absent from the DECLARATION. The run used `adamw_torch_fused`, accum=1, max_grad_norm=1.0, no recompute, linear LR with zero warmup, and whatever attention implementation the model config picks — and the manifest could not say so, because there was no config key to resolve and record.

Those six values are measured, not assumed: they are the `TrainingArguments` defaults read out of transformers 5.16.1. Stating the version is not pedantry — it is the whole shape of the problem. An undeclared axis takes its value from the installed engine, so the same FoundationScale config trains differently under a different transformers release, and no artifact of either run would record that anything changed. Note in particular that the optimizer default is the *fused* AdamW, not the reference one: T1-9 proposes to measure step time across optimizers, and it would be measuring against a fused baseline without ever saying so.

That is the #346 class at scale: unmeasured reads as green. It is why, when this matrix was written, 14 of its 33 rows were unrunnable.

## What #373 changed, and why the grep counts above no longer hold

#373 shipped tier 0, so **every count in the list above is now stale by design.** They are corrected here rather than edited in place because a matrix that quietly rewrites its own premise cannot be audited against the tree it described.

Re-measured on the same instrument, over the same scope, on the same day this section was written: `attn_implementation|flash_attention|sdpa` → 31 occurrences, `get_scheduler|lr_scheduler|warmup` → 26, `clip_grad|max_grad_norm` → 14, `deepspeed|FSDP|fully_shard|ZeRO` → 9 (the original two in `gates/` and `provenance/`, plus seven in `train/`), `AdamW|Adafactor|SGD(` → 3. Only `torch.compile` is still zero: nothing declares it, and it remains T1-15's business.

`TrainConfig` now carries nine tier-0 axes — `optimizer`, `gradient_accumulation_steps`, `max_grad_norm`, `gradient_checkpointing`, `attn_implementation`, `lr_scheduler_type`, `warmup_steps`, `sharding_strategy`, `cpu_optimizer_offload` — each resolved through `ConfigResolver` and recorded. Each is bound into the kwargs dict only when the operator declared it, so an undeclared axis still takes the engine default; what changed is that the ABSENCE is itself recorded, which is precisely the distinction the original measurement said no artifact could make.

Two of the nine are declarable without being executable, and the matrix says so rather than counting them as capability: `sharding_strategy=fsdp|zero3` and `cpu_optimizer_offload=true` REFUSE with 96 rather than accept a declaration this plane cannot honour. That is why T1-13 and T1-14 read REFUSED and not UNBLOCKED — an unrunnable arm is not the same fact as a missing one.

## Two tiers, and tier 0 is not optional

**Tier 0** (no GPU, blocks everything below): make each axis DECLARABLE and RECORDED. A knob that is exposed but unrecorded is worse than absent, because a run then makes a claim no artifact can check. Every tier-0 row is:

- (a) a TrainConfig field,
- (b) bound into the TrainingArguments kwargs,
- (c) resolved through ConfigResolver so `provenance/manifest.py` records key/value/source,
- (d) a control proving an off-declaration run REFUSES rather than silently training on the engine default.

**Tier 1** (GPU, a 4-GPU tray): one row per claim. Every row carries a CONTROL ARM — an arm that must come out DIFFERENT.

Tier 0 blocks tier 1 because a tier-1 row that toggles an undeclared axis measures nothing attributably: the run would train on the engine default either way, and no artifact could distinguish "the knob worked" from "the knob was never wired." Rows T1-9 through T1-14 each named the tier-0 row they waited on. #373 shipped all of those, so each of the six now names what released it: T1-9 through T1-12 are UNBLOCKED and wait only on a tray, while T1-13 and T1-14 are REFUSED — the dependency shipped as a refusal, so their run arms have nothing to execute.

## The matrix

### T0 — declaration and recording (CPU, minutes each)

| ID | Claim | Run arm | Control arm (must differ) | Cost | Status today |
|---|---|---|---|---|---|
| T0-1 | optimizer name is declared and recorded | build with optimizer=adamw_torch | declaring an unsupported optimizer REFUSES 96, does not fall back | 0 GPU | SHIPPED (#373): declared optimizer reaches TrainingArguments.optim; an unsupported name REFUSES 96 before a Trainer exists |
| T0-2 | grad accumulation steps are declared and recorded | accum=4 | kwargs actually carries gradient_accumulation_steps=4 (introspect args) | 0 GPU | SHIPPED (#373): reaches gradient_accumulation_steps in the kwargs; an omitted axis adds no kwarg and is recorded as an explicit absence |
| T0-3 | max_grad_norm is declared and recorded | 1.0 vs 0.0 | args.max_grad_norm reads back what was declared | 0 GPU | SHIPPED (#373): reaches max_grad_norm in the kwargs; a non-positive value REFUSES 96 at the CLI boundary (#384) |
| T0-4 | gradient checkpointing (recompute) is declared and recorded | on | model.is_gradient_checkpointing is True (not just the flag set) | 0 GPU | SHIPPED (#373): model.is_gradient_checkpointing reads True on the loaded model, not merely the flag |
| T0-5 | attention implementation is declared and recorded | sdpa | model.config._attn_implementation == declared; an impl the build cannot supply REFUSES | 0 GPU | SHIPPED (#373): reaches model.config._attn_implementation; an impl the build cannot supply and one the loader silently overrides both REFUSE 96 |
| T0-6 | LR schedule and warmup are declared and recorded | cosine, warmup=2 | the LR at step 1 != LR at step 10 (read the log) | 0 GPU | SHIPPED (#373): the declared schedule and warmup move the optimizer LR -- early below half the base rate, late above 0.85 of it |
| T0-7 | sharding strategy (ddp\|fsdp\|deepspeed) is declared and recorded | ddp | a strategy with no wiring REFUSES 96 rather than silently running DDP | 0 GPU | SHIPPED (#373): declared and recorded; fsdp/zero3 REFUSE 96 rather than silently running DDP, while ddp and undeclared proceed |
| T0-8 | CPU optimizer offload is declared and recorded | off | declaring on with no backend REFUSES | 0 GPU | SHIPPED (#373): declared and recorded; offload=true REFUSES 96 with no backend, while false and undeclared proceed |
| T0-9 | code.status is a real commit | run from a git checkout | #346: a COPY must record status=not_a_repository AND the run must say so loudly, not proceed silently | 0 GPU | RED (#346) |
| T0-10 | the effective topology is read from runtime evidence, never echoed from the config | WORLD_SIZE=8 LOCAL_WORLD_SIZE=8 against a config declaring tp=pp=ep=cp=4 and gpus_per_node=4 | #375: the pre-fix `_effective_topology`, which sourced 5 of the 7 fields from cfg -- 11 of the 14 legs in `tests/train/test_effective_topology.py` go RED against it and 14 of 14 pass after | 0 GPU | SHIPPED (#375): 14 of 14 legs green, 11 RED on the pre-fix tree |

### T1 — precision (GPU)

| ID | Claim | Run arm | Control arm (must differ) | Cost | Status today |
|---|---|---|---|---|---|
| T1-1 | fp32 masters change the update magnitude relative to bf16 | fp32 masters | bf16 masters, same seed/lr/steps | ~15 min x2 | MEASURED (#367/#368: bf16 discards ~99.99%) — re-take under the new manifest |
| T1-2 | fp16 grad scaler skips an overflowing step | fp16 | inject an overflow; scaler must SKIP, loss must not become nan | ~15 min | never run |
| T1-3 | a 4-bit (nvfp4) declaration REFUSES, never falls back to bf16 | precision=nvfp4 | assert rc==96 AND no checkpoint written | ~2 min | REFUSES by design (#345) |

### T1 — checkpoint lifecycle (GPU)

| ID | Claim | Run arm | Control arm (must differ) | Cost | Status today |
|---|---|---|---|---|---|
| T1-4 | reloaded weights are bit-identical (save->load parity) | save then load, compare_keys EXACT | perturb one tensor -> parity must go RED | ~15 min | verify/parity.py SHIPPED, no run |
| T1-5 | 10 steps + resume 10 == 20 steps straight (RESUME fidelity) | resume arm | straight-through arm; final weights must match within tolerance | ~40 min | THE gap — No optimizer/RNG/dataloader restore is proven |
| T1-6 | a checkpoint whose shard is a symlink to the base is CAUGHT | the #343 artifact | a real shard must pass the same detector | ~5 min | SHIPPED (#343): is_symlink() hoisted ahead of exists(), so a linked shard draws no link-following stat at all; GREEN on a real two-shard ~51.6 GB safetensors checkpoint on GB200 with zero byte credit; harness 11/11 here, 8/11 pre-fix (the 3 that flip are the detector arms; the 8 that hold include two proving the src-layout import bootstrap both finds this checkout and declines a tree with no package, added after the launcher-contracts job showed the self-test assumed an installed distribution) |
| T1-7 | converted weights are parity-equal to source (DCP<->HF) | convert both ways | a truncated shard must be caught | ~20 min | ABSENT |
| T1-8 | resharding: save on 4 GPUs, load on 2 | 4->2 | 4->3 (non-divisor) must REFUSE, not silently drop experts | ~30 min | ABSENT |

### T1 — optimizer / memory / kernels (GPU, each needs its T0 row first)

| ID | Claim | Run arm | Control arm (must differ) | Cost | Status today |
|---|---|---|---|---|---|
| T1-9 | adamw vs fused vs adafactor vs sgd differ in step time and loss curve | 4 arms, same seed | identical curves across optimizers = the knob did nothing | ~15 min x4 | UNBLOCKED by #373 (T0-1 shipped): needs a GPU tray, never run |
| T1-10 | accum=4,bs=1 and accum=1,bs=4 give the SAME loss curve | both arms | if they differ beyond tolerance, accumulation is wrong | ~15 min x2 | UNBLOCKED by #373 (T0-2 shipped): needs a GPU tray, never run |
| T1-11 | grad checkpointing: peak memory DROPS and step time RISES | on vs off | neither moving = the flag is a no-op (the silent-fallback shape) | ~15 min x2 | UNBLOCKED by #373 (T0-4 shipped): needs a GPU tray, never run |
| T1-12 | eager/sdpa/flash agree on logits within tolerance, differ in step time | 3 arms | identical step times = the impl never changed | ~10 min x3 | UNBLOCKED by #373 (T0-5 shipped): needs a GPU tray, never run |
| T1-13 | CPU offload: optimizer state leaves GPU memory | offload on vs off | GPU memory must drop by ~optimizer-state size; if not, offload did nothing | ~20 min x2 | REFUSED by the package plane (#373): offload=true refuses before load, so there is no offload arm to weigh against off |
| T1-14 | ZeRO-1/2/3 reduce per-rank memory monotonically | 3 arms + DDP baseline | flat memory across stages = no sharding happened | ~30 min x4 | REFUSED by the package plane (#373): fsdp/zero3 refuse before load, so there are no ZeRO stages to compare |

### T1 — architecture (GPU)

| ID | Claim | Run arm | Control arm (must differ) | Cost | Status today |
|---|---|---|---|---|---|
| T1-15 | a dense model trains end-to-end and the bytes move | Qwen dense, lr>0 | an `lr=0` arm through the same path: weight parity must show ZERO movement | ~15 min x2 | done (#162/#176, #366) — re-take under new manifest, control arm never run |
| T1-16 | an MoE model trains and experts receive gradient | MoE arm | a router that never routes = zero grad on some experts; must be detected | ~40 min | classification SHIPPED, runtime ABSENT |
| T1-17 | ep=2 shards experts across ranks | ep=2 vs ep=1 | identical per-rank memory = no expert sharding | ~40 min | REFUSED by the package plane (#375): ep>1 refuses before load |
| T1-18 | LoRA through the package: adapter weights train, base frozen | LoRA arm | base-weight parity must show ZERO movement; adapter parity must show movement | ~20 min | ABSENT from the package plane (#329) |
| T1-19 | tied embeddings: tying survives save/load and sharding | a tied-embedding model | untied model as control | ~20 min | ABSENT (#172/#202 shape) |

### T1 — modality (GPU) — the stated core purpose

| ID | Claim | Run arm | Control arm (must differ) | Cost | Status today |
|---|---|---|---|---|---|
| T1-20 | text: a text corpus trains end-to-end and the bytes move | text arm, lr>0 | an `lr=0` arm through the same path: weight parity must show ZERO movement | ~15 min x2 | done (#162/#176, #366), control arm never run |
| T1-21 | an image-bearing corpus trains and the vision tower's weights MOVE | image arm | a text-only arm through the same path: tower weights must NOT move | ~30 min | #371 routed images, never trained on them |
| T1-22 | declaring video REFUSES cleanly today | video corpus | must be 96, not a silent text-only train | ~5 min | refuses by design |
| T1-23 | declaring audio REFUSES cleanly today | audio corpus | must be 96, not a silent drop of the audio field | ~5 min | ABSENT — may silently drop |

## Rows that cannot fail

The control-arm rule: a row with no arm that must come out differently is a row that cannot fail, and a row that cannot fail is not a measurement. This campaign has now produced several of those — #291, #294, and #372 — which is why every row in this matrix carries an explicit control arm, and why the JSON rendering of this matrix enforces, as a gate, that no `control_arm` field is empty.

## Ordering rule

T0-9 is still first and still RED (a run whose commit is unrecorded is unattributable — every row below inherits it). T0-1..T0-8 are done: #373 shipped them, and their CPU-only proofs now run in the suite on every commit, so they no longer sit in this queue. What remains is tier 1, cheapest-first: T1-3, T1-22, T1-23 (minutes), then T1-4, T1-1, T1-12, T1-10, T1-11, then the long ones (T1-5, T1-16, T1-21). T1-6 has left the queue — #343 shipped it — and it is worth saying why it was cheap, because the reason is not that the claim was small. The whole defect is the ORDER of two stdlib calls, so proving it needs one real checkpoint and no tray: only shard headers are read, and a fifty-gigabyte checkpoint costs the same as a fifty-kilobyte one. T1-14 has left the queue with T1-13: the plane refuses both run arms, so there is nothing to schedule until a ZeRO or offload backend exists to schedule it against.

## The honest count
33 rows: 10 that need no GPU and 23 that do. Of those, 0 now name an axis the training plane does not expose at all, and 0 are blocked by another row, so 0 are unrunnable for want of a declaration — the lower tier has shipped, and what stands between this matrix and the rest of its rows is a tray, not a missing knob. 3 are already measured and need only a re-take under a manifest that records them. 2 are refusals that must be PROVEN to be refusals rather than silent fallbacks.

The expert-parallel row belongs to none of those buckets and is still not runnable through this plane. It is refused before a model is loaded, and that refusal is already proven by the topology legs, so it is neither blocked by another row nor awaiting a proof — the axis is simply not wired, and the plane now says so instead of training pure DDP under the label.

Every number in that paragraph is derived from `matrix.json` by `checks/verification_matrix.py`, which reads them back positionally and refuses a row whose control arm is a placeholder. They are not restated by hand, and the table above is a rendering of the same ledger. The two tiers are described in words here rather than by their numeric labels because the countables check reads every integer in this section positionally, so a digit inside a label would itself be adjudicated as a countable.
