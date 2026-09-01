# Deliverable E — Compatibility Matrix

**Status: IN PROGRESS.** Every cell is either a measurement with a date, or the literal
token `UNMEASURED`. No cell is blank, and no cell is inferred from a neighbouring cell.
An empty matrix would satisfy "all rows pass" vacuously; that is the failure this project
exists to prevent, so absence of evidence is written down as absence, not as a pass.
A count of zero is never a pass; it is `UNMEASURED` unless a refusal is measured.

Estate identifiers are elided in this public document: `<PARTITION>`, `<NODE>`, `<HOME>`,
`<IMAGE>`. Estate shape, without identifiers: 8× H100 SXM per node, one node used for the
measured runs below, Slurm time cap 7 days, container runtime `singularity`; enroot is
present on another estate and is not assumed here.

## E.1 Models

"Builds" means: config resolved by search, `AutoConfig` accepted it, and the architecture
instantiated on a meta device inside the container (transformers 4.57.1,
`PYTHONNOUSERSITE=1`). It does **not** mean weights were materialised. The executed rows
below are the two models that actually ran the validation chain; every later stage cell is
anchored to a job id and, where a number exists, the number.

### E.1.1 Executed here — 2 models

| Model | Stage | Status | Measurement |
|---|---|---|---|
| `qwen3-0.6b` | Environment | MEASURED | probe link 37340 COMPLETED 0:0; chain 37340-37343 all COMPLETED 0:0; runtime `singularity`, `FS_GPUS_PER_NODE=8`, one node, `FS_NCCL_NET_PLUGIN=none` |
| `qwen3-0.6b` | Model Loading | MEASURED | production 37341 and resume 37342 end `rc=0`; checkpoint payloads present `8/8` in 37342 |
| `qwen3-0.6b` | Dataset | MEASURED | shared `phase3_real_4k` corpus, same corpus as the Gemma arm; held-out tokens scored `3709 / 3709` in 37342 |
| `qwen3-0.6b` | Distributed Training | MEASURED | 37341 terminal line `END rc=0 phase=train`; 8 ranks, one node. Its `checkpoint_saves_adjudicated=1` is the #188 truncated denominator — see E.4; the training result does not rest on it |
| `qwen3-0.6b` | Checkpointing | MEASURED | chain left 3 checkpoints — `checkpoint-step-00000002`, `checkpoint-step-00000005`, `checkpoint-step-00000020` — each with `manifest.json`; optimizer-state mismatches `0 of 513` in 37342. **Adjudication denominator corrected:** 37341/37342 each printed `checkpoint_saves_adjudicated=1` over those 3 dirs (#188); the corrected walk over the same tree is job 37345 — `adjudicated=3 of 3 checkpoint dir(s) ok=3 abstain=0 refuse=0`, three `VERDICT 0 PASS` lines, `checks_measured=10 checks_green=10 checks_red=0` each |
| `qwen3-0.6b` | Resume | MEASURED | 37342 `restore_delta = 0.0`; `cross_rank_spread_after_resume = 0.0` on all 8 ranks; `verdict:"MEASURED"` with empty `unmeasured` list |
| `qwen3-0.6b` | Evaluation | MEASURED | 37342 held-out eval loss `2.0253733`; held-out tokens `3709 / 3709`; peak GPU memory `3.535 GiB` |
| `gemma-3-1b-it` | Environment | MEASURED | **two Gemma runs, not one** — 37319 (`phase3.json`, `--resume-tolerance 10.0`) COMPLETED 0:0 in 00:06:12, and 37336 (`phase4_gemma.json`, `--resume-tolerance 0.0005`, the Qwen arm's threshold) reached the resume proof and then died on the #174 self-kill path: `srun: error: task 0: Exited with exit code 1`, 1163 lines, **0** `END` lines, **0** `FATAL` lines, `sacct` FAILED `0:15`. Forensics 37320-37323 ran zero training steps on 8 of 8 ranks through the trainer's own construction path |
| `gemma-3-1b-it` | Model Loading | MEASURED | forensics reuse `load_artifacts`; attention backend/model class/dtype identical on 8 of 8 ranks: `sdpa`, `Gemma3ForCausalLM`, bf16; weights `0 of 340` divergent across 8 of 8 ranks in 37322-37323 |
| `gemma-3-1b-it` | Dataset | MEASURED | same `phase3_real_4k` corpus as Qwen; `input_ids` sha256 identical on 8 of 8 ranks, shape `[2, seqlen]`, attention mask fully packed in 37321-37323 |
| `gemma-3-1b-it` | Distributed Training | MEASURED | 37319 `rc=0 phase=train`; train throughput `5517.637` tokens/s segment 1 and `4527.572` tokens/s segment 2 post-resume; optimizer-step mismatches `0 of 529` |
| `gemma-3-1b-it` | Checkpointing | MEASURED | `checkpoint-step-00000050` has 8 of 8 rank payloads present; missing checkpoint keys `341 of 341` present; final save at step 200 records `fixed_loss` on `0 of 8` ranks, a declared absence, not agreement. 37319's `checkpoint_saves_adjudicated=1` is the #188 truncated denominator. The re-walk was attempted first on job 37347 and returned nothing — `ADJUDICATE` 0 times in a 51-line log ending `END rc=0 mapped_rc=95`, the #193 dead branch, not a property of the tree. With #193 landed, **job 37348** walked the same tree: `adjudicated=2 of 2 … ok=2 abstain=0 refuse=0`, `END rc=0 phase=post-mortem checkpoint_saves_adjudicated=2 checkpoint_saves_found=2`, `COMPLETED 0:0`. Both checkpoints green on all of A1–A7b at `checks_measured=10 checks_green=10 checks_red=0 legs_abstained=0`, `world_size=8`, `present 8/8 expected ranks`, `unexpected=[]`, `regular non-empty shards 8/8`, directory step agreeing with `manifest global_step` at 5 and 20 |
| `gemma-3-1b-it` | Resume — restore fidelity | MEASURED | 37336 `restore_delta = 0.0` at tolerance `0.0005`; 37319 `after_resume = 0.5986318588256836` and `before_save = 0.5986318588256836`, bit-identical at tolerance `10.0`; forensics within-rank restore delta `0.0`, fingerprint keys changed `0 of 341` in 37320-37323. **Before #177**, job 37312 at tolerance `0.0005` read the same divergence as a continuity failure — `RUN_SUMMARY_JSON ... "reason":"fixed loss changed by 0.16057962, exceeding the stated tolerance 0.0005","verdict":"UNMEASURED"` with `dataset_origin:"UNKNOWN_AFTER_FAILURE"` — which is the defect #177 closed, not a restore result |
| `gemma-3-1b-it` | Resume — `fixed_eval_rank_invariance` | UNMEASURED | **declared abstention in job 37336** — `unmeasured:["resume.fixed_eval_rank_invariance"]` at `--resume-tolerance 0.0005`, `restore_delta 0.0`, `cross_rank_spread_before_save` and `cross_rank_spread_after_resume` both `0.2940967082977295`; not a pass. **Corrected 2026-09-01:** this row previously cited job 37319. 37319 contains the string `fixed_eval_rank_invariance` **0 times** and reports `unmeasured:[]` — it ran at `--resume-tolerance 10.0`, where the same divergence passes. See #192 in E.4. **Stage 35 (#192) landed 2026-09-01:** the
absolute claim now has its own knob, and an unset knob makes the abstention self-describing
(`cross_rank_spread_delta`, `rank_agreement_preserved`). Certified at build time; this row stays
UNMEASURED until a job runs it. |
| `gemma-3-1b-it` | Evaluation | MEASURED | fixed eval executed; cross-rank spread at step 50 `0.294`; forensics cross-rank loss spread `1.1025075912475586` at seqlen 1024 and `0.7679` at seqlen 512; mechanism inside Gemma is UNMEASURED — see E.1.2 |

### E.1.2 Characterized model-level limitation — Gemma cross-rank forward divergence

Restore is exact on both executed models: Gemma 37336 reports `restore_delta = 0.0` and
37319's `after_resume` and `before_save` are bit-identical, and Qwen 37342 reports
`restore_delta = 0.0`. The statistic that differs does **not** measure restore; it measures
cross-rank spread on a fixed eval batch. On the same framework revision, the same corpus,
the same 8 ranks and — this is what makes the comparison admissible — **the same
`--resume-tolerance 0.0005`**, the two env files being character-identical in trainer argv
apart from the model path: Qwen 37342 measures `cross_rank_spread_before_save 0.0` and
`cross_rank_spread_after_resume 0.0`, Gemma 37336 measures `0.2940967082977295` for both. A framework defect would have to explain
exact rank agreement on one model and `0.294` on the other over the identical code path; it
does not. The divergence is attributable to the model/stack and is recorded here as a
characterized limitation, not as a framework blocker. The mechanism inside Gemma is
**UNMEASURED** and is labelled `UNMEASURED`; no mechanism is asserted.

Ruled out by measurement, each with its denominator, from forensics jobs 37320-37323 and
the checkpoint-scalar reader:

* data / row indexing — `input_ids` sha256 identical on 8 of 8 ranks, shape `[2, seqlen]`,
  attention mask fully packed.
* weights — `340 of 340` parameters byte-equal across 8 of 8 ranks under
  `summon_full_params(rank0_only=False)`; `named_parameters()` reports 340 because the tied
  `lm_head.weight` is deduped, while the state-dict fingerprint reports 341.
* restore fidelity — `0 of 341` fingerprint keys changed; within-rank delta exactly `0.0`.
* attention backend / model class / dtype — `sdpa`, `Gemma3ForCausalLM`, bf16, identical on
  8 of 8 ranks.
* sequence length and the 512 sliding window — spread persists at seqlen 512, where the
  window is inert; this falsified the sliding-window hypothesis stated before the run.
* dropout and logit softcapping — `attention_dropout: 0.0`, both softcappings null.
* missing checkpoint keys — `341 of 341` present.
* wrap or load divergence — the load-phase census is byte-identical between the two models.

### E.1.3 Declared by config search, never executed here — 4 declared, 2 executed

These rows were not run through load → distributed → checkpoint → resume → eval in this
campaign. They are kept separate so a declaration cannot be mistaken for a measurement.
Denominator for this block: **4 models declared, 2 executed** (the executed two are in
E.1.1). Every downstream stage is `UNMEASURED`, not blocked and not passed.

| Model | model_type | Architecture | Params | Root layout | Binds | Builds | Distributed | Checkpoint | Resume | Eval | Status |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Qwen3-4B-Instruct-2507 | `qwen3` | Qwen3ForCausalLM | 4.02B | self-contained | 1 | **YES** | UNMEASURED | UNMEASURED | UNMEASURED | UNMEASURED | declared, never executed here |
| qwen25_7B_Instruct | `qwen2` | Qwen2ForCausalLM | 7.62B | self-contained | 1 | **YES** | UNMEASURED | UNMEASURED | UNMEASURED | UNMEASURED | declared, never executed here |
| llama31_8B_Instruct | `llama` | LlamaForCausalLM | 8.03B | commit-SHA nested | 1 | **YES** | UNMEASURED | UNMEASURED | UNMEASURED | UNMEASURED | declared, never executed here |
| gemma3_27B | `gemma3` | Gemma3ForConditionalGeneration | 27.43B | self-contained | 1 | **YES** | UNMEASURED | UNMEASURED | UNMEASURED | UNMEASURED | declared, never executed here |

### E.1.4 Not declared supported — build-time refusals, 3 rows, 0 executed

| Model | model_type | Architecture | Params | Root layout | Binds | Builds | Distributed | Checkpoint | Resume | Eval | Status |
|---|---|---|---|---|---|---|---|---|---|---|---|
| gemma-4-12B-IT | `gemma4_unified` | Gemma4UnifiedForConditionalGeneration | — | config-overlay + symlink | **2** | **NO** — unrecognised | blocked | blocked | blocked | blocked | needs registration seam |
| gemma-4-31b-it | `gemma4` | — | — | self-contained | 1 | **NO** — unrecognised | blocked | blocked | blocked | blocked | needs registration seam |
| gemma-4-26B-A4B | UNMEASURED | UNMEASURED | — | config-overlay + symlink | **2** | UNMEASURED | UNMEASURED | UNMEASURED | UNMEASURED | UNMEASURED | stretch |

Notes on the two NO rows: transformers 4.57.1 in the container rejects both with
"does not recognize this architecture" for `gemma4_unified` and `gemma4`. These are
out-of-tree checkpoints (ref #119). They are the concrete instance of the standing question
"is this a one-time workaround or does it belong in the abstraction?" — the answer is that
FoundationScale needs an **architecture registration seam** (a declared, verified extension
point) rather than a core edit teaching it about one vendor's fork. Until that seam exists
these rows stay NO, and no Gemma-4 result may be reported as a framework pass.

**Retracted 2026-08-31 — the `Med-Gemma` row.** An earlier revision of this matrix carried
a row reading "Med-Gemma | ambiguous | 3 config candidates | REFUSED". It has been removed
rather than corrected, because it cannot be reproduced: the path it was measured at was
never recorded, and no directory of that name is now findable on the estate. The number 3
therefore has no denominator behind it and no command that regenerates it. A row that
cannot be re-run is not a measurement, and leaving it in place would have made the refusal
detector look certified by evidence that does not exist. This is the same defect the
matrix exists to catch, found in the matrix itself.

The designed refusal behaviour is nonetheless real, and E.2 below records the rows that
substantiate it — each one re-measured per depth with `find -maxdepth`, independently of
the resolver, and restated as a prediction the resolver has to reproduce in
`h100/gate133_estate.py`.

## E.2 Root-layout topologies (measured 2026-08-31, ref #133)

Three incompatible meanings of "the model root" on ONE estate:

| Layout | `config.json` at root | Symlinks | Bind closure | Breaks what |
|---|---|---|---|---|
| self-contained | yes | 0 | model root only | — |
| config-overlay | yes (patched) | 7–8 | model root **+ the tree the symlinks resolve into** | binding only the root ⇒ R4 green, weights dangling (#132) |
| commit-SHA nested | **no** — one level down under a 40-hex dir | 0 | model root only | `join(model_root, "config.json")` ⇒ file not found |

Consequence for the framework: config location must be **searched and counted**, and the
bind set must be the **closure**, not the declared root. Neither may be hard-coded per
model. This is the standing instruction "inspect the actual directory structure and model
configuration before writing any hard-coded assumptions" made concrete.

**The counting rule is per-depth, not flat.** "Refuse on more than one candidate anywhere
in the subtree" was the first rule written, and it was wrong: it refuses stock upstream
layouts. `gpt-oss-20b` ships an `original/` directory and `qwen2_7B_embedding` ships
`1_Pooling/`, each with its own `config.json` beside an unambiguous one at the root. The
rule is therefore **shallowest populated depth wins** — ambiguity is a property of that
depth alone, and the message reports both denominators so the narrower claim is auditable.

| Root | d0 | d1 | Verdict | Why |
|---|---|---|---|---|
| `OpenAI/gpt-oss-20b` | 1 | 1 (`original/`) | **resolves** | depth 0 is unambiguous; `original/` never consulted |
| `embedding_models/qwen2_7B_embedding` | 1 | 1 (`1_Pooling/`) | **resolves** | same shape, different vendor convention |
| `Vision-Language-Models/Google` | **0** | **2** | **REFUSED** | "found 2 config.json candidates at depth 1" |
| `Alibaba-Qwen/qwen3` | **0** | **7** | **REFUSED** | a family directory, not a checkpoint |

Per-depth counts measured with `find -maxdepth` independently of the resolver. The last two
are the observed instances of the refusal firing on real data, and the first two are its
controls: without a MUST_PASS the refusal could be a resolver that refuses everything.

**Status of the rule as code.** The resolver and its suite are generated by three build stages
(`extract_fs_model_root.py`, `patch_fs_model_root.py`, `patch_fs_train_model_root.py`) and the
suite runs in the build: **12 passed**. The `gpt-oss-20b` MUST_PASS was additionally reproduced
off-estate on a synthetic root of the same shape (a `config.json` at depth 0 beside
`original/config.json` at depth 1), which resolves self-contained with `binds=1`.

**Re-run on the estate, 2026-08-31: 8/8 measured rows reproduce the independent measurement,
0/8 UNMEASURED.** Until that run this table's estate rows were a prediction; `gate133_estate.py`
abstains with rc=3 off the estate rather than reporting eight failures, because "root not
present on this host" and "prediction not reproduced" are different verdicts and it used to
print them identically. The two refusal rows fired with their denominators intact — 2 candidates
at depth 1 (6 in the bounded subtree) for the vendor directory, 7 at depth 1 (8 in the subtree)
for the model-family directory — so the refusal is measured, not merely defined.

Getting that run to happen produced a finding of its own, filed as #138: the login node has
**Python 3.6.8** and nothing newer, so the gate could not parse, let alone abstain. It ran
inside the declared container instead. A host-side gate that assumes a modern host interpreter
is the same defect as importing the host's torch (#107) or inheriting the host's PATH (#67) —
the framework using an execution environment it never declared.

**The resolver is no longer an orphan.** Stage C (`patch_fs_train_model_root.py`, 11/11 gates,
idempotence verified by sha256 across a re-run) binds `load_artifacts` to
`resolve_model_root`, so `AutoConfig`, `AutoProcessor`, `AutoTokenizer` and both downstream
`model_class.from_pretrained` readers see the resolved config directory rather than the
operator-declared root. Before it landed the plane had 12 green tests and zero callers, which
is the #86 shape: a suite that passes about code nothing runs.

## E.3 Communication plane (measured 2026-08-31, ref #129)

| Configuration | 8-rank all_reduce | Notes |
|---|---|---|
| image default (HPC-X plugin auto-loaded) | **SIGSEGV** | faults inside the first collective, after `init_process_group` succeeds |
| `NCCL_NET_PLUGIN=none` | **PASS** | `world=8 got=28.0 expected=28.0 spread=0.0`; NVLink unaffected (see below) |
| `NCCL_NET=Socket` | PASS | forces socket inter-node — wrong fix for a multi-node framework |
| `NCCL_IB_DISABLE=1` | no effect | disables a **transport**, not a **plugin load** |
| `NCCL_NET_PLUGIN=""` (empty) | **disables the plugin** | empty is *not* unset — an unconditional export would silently disable a working plugin on another estate |

**"NVLink unaffected" — what was actually counted.** The fix disables a *network plugin*,
so the obvious worry is that it also demotes intra-node transport. Re-measured under
`NCCL_DEBUG=INFO` on 8× H100, counting lines in the combined 8-rank stderr:
`24 coll channels`, 17 lines matching `NVLS`, 24 matching `via P2P/CUMEM`. NVLS is
therefore still selected and P2P/CUMEM is still the intra-node path — the plugin is gone
and NVLink is not.

An earlier revision of this row read "24 NVLS channels, 192 P2P/CUMEM". The 192 is
withdrawn: it is 8 × 24, an inference from rank count, not a count of anything observed.
The figures above are log-line counts with the grep patterns stated so they can be
re-derived. This distinction matters more than the numbers do — the claim ships in a
comment in the generated launcher, where a reader has no way to tell a measurement from
an arithmetic guess.

Detector controls: MUST_PASS green, MUST_FIRE observed red on both a numerical fault
(poisoned rank-0 contribution) and a crash. 3/3 control rows correct.

## E.4 Gates that are green but were green for a weaker reason than claimed

Recorded here because a matrix of passes is worthless without the list of passes that were
narrowed by measurement.

| Gate | Claimed | Actually verified | Ref |
|---|---|---|---|
| fs117 R4 declared-mount verification | "declared mounts materialised inside" | the declared **path** is readable; nothing beneath it | #132 |
| 8-GPU launch | "8 GPUs" | measured count and actual launch were decoupled | #124 (fixed) |
| resume contract | runtime-agnostic | singularity-only; enroot never crossed the vars | #122 (fixed) |
| E.1 `Med-Gemma` row | "3 config candidates → REFUSED", a measured refusal | **nothing** — path never recorded, directory not findable; row retracted, not corrected | this doc, 2026-08-31 |
| E.3 / launcher comment | "192 P2P/CUMEM" | 24 log lines match `via P2P/CUMEM`; 192 was 8 × 24 inferred from rank count | #129, 2026-08-31 |
| build plane header | "the whole plane is rebuilt from scratch on every invocation" | 3 of 4 layers. The backend's 73 KB base text came from a stage that was not in `STAGES` and was never removed | #136 (fixed) |
| Deliverable D generator | — (assumed producible) | had **never** produced `LAUNCH.md`; red on L2 and correctly refusing to write | #127 |
| post-mortem adjudication link | `afterany` link COMPLETED 0:0 read as a clean sweep | **Before, job 37343:** entire log one line — `POST-MORTEM afterany adjudication link reached; reporting only, no training launched. OUT_DIR=<HOME>/...`; no `END` line, exit 0, 3 checkpoints present, **0 of 3 adjudicated**, denominator never printed. **After, job 37344 / fs187:** no early exit; sets launcher-internal skip flag, falls through backend init, runtime setup, bind plane, in-container GPU census and `fs_compose_launch`, skips only the training `run_in_container`, then reuses the existing adjudication tail verbatim — same `adjudicate_tree`, same `checkpoint_observed` denominator, same `[[ "$checkpoint_observed" -gt 0 ]] || fail 95` guard, same `END` line printing `phase=post-mortem` | #187, jobs 37343 → 37344 |
| checkpoint-tree adjudication | `ADJUDICATE complete`, every checkpoint in the tree read | **1 of 3.** The shipped `adjudicate_tree` was a one-liner whose loop body drains the process substitution the loop reads from: `run_adjudicators` inherits the loop's stdin, consumes the rest of the `find` stream, and the loop ends after the first entry. It then printed its count **from the truncated stream it was measuring**, so numerator and denominator were cut by the same defect and the report stayed self-consistent at one third of the truth. Observed on **three** jobs — 37341, 37342, 37344 — each recorded COMPLETED 0:0. The fix (`patch_adjudicate_denominator.py`) had been authored under #175, verified and committed, then wired into **neither `STAGES` nor `PUBLISH_SET.txt`**; the build kept regenerating the defective one-liner. **After, job 37345 / fs188:** `adjudicated=3 of 3 ... ok=3 abstain=0 refuse=0`, `END rc=0 ... checkpoint_saves_adjudicated=3 checkpoint_saves_found=3`, three `VERDICT 0 PASS` lines. The installed version collects into an array before iterating, redirects `run_adjudicators` from `/dev/null`, counts `found` and `processed` independently and **refuses 96 when they disagree** | #188, jobs 37341/37342/37344 → 37345 |
| launcher failure path / exit contract | #174, #169 and #171 all recorded **fixed** | **none of the three was in the artifact.** Measured on the shipped build: `fs175:` occurred **0 times** in both generated files, the launcher carried **0 traps**, and its failure block handed the backend's hard-stop helper `"$$"` — the launcher's own pid. The helper's first act is `kill -TERM "$pid"`, so with no trap bash took SIGTERM's default action and the `END` line, the `exit` below it and the helper's own container-cleanup arm were all unreachable; `\|\| true` cannot help, the shell dies from the signal and not from a status. `patch_launcher_exit_discipline.py` (676 lines) was in neither `STAGES` nor `PUBLISH_SET.txt`. **Before, job 37304:** 452-line log, **0** `END` lines, **0** `FATAL` lines, `sacct` `0:15` — signal 15, which an orchestrator reads as a human cancel. **After, job 37346 / fs175, a deliberately-failing run:** `FAILED` **5:0** in `00:05:05`, 278 lines, terminating in `END rc=1 mapped_rc=5 phase=train FAILED (fs_map_run_verdict: verdict=NONE rc=1 mapped=5 -- no verdict line; an undeclared death is a FAILURE, not an abstention)` | #189 → reopens #174/#169/#171, jobs 37304 → 37346 |
| resume proof, Gemma arm | "the Gemma arm ended MEASURED with one declared abstention", cited to job 37319 | **assembled from two runs.** 37319 contains the string `fixed_eval_rank_invariance` **0 times** and reports `unmeasured:[]`; it ran at `--resume-tolerance 10.0`. The abstention is job **37336**, at `0.0005`. Underneath the citation error is a live defect: `_resume_continuity_verdict` takes **one** `tolerance` and spends it on two questions — `restore_delta > tolerance → RED` and `spread <= tolerance → the ranks agree`. #177 separated the two verdicts; it did not separate the two thresholds, so the only way to stop a model's forward nondeterminism from abstaining is to raise the number that also guards restore fidelity. Measured on one model: spread `0.2940967082977295` abstains at `0.0005` (37336) and passes at `10.0` (37319). At `10.0` the RED needs a delta of 10.0 in a quantity observed at `0.0` whose scale is a loss of `0.5986` — ~17× the loss — while the run still prints `why_tolerance: "continuity is therefore bounded, not claimed bit-for-bit"` | #192 (OPEN), jobs 37336 / 37319 |
| publish set vs. stage list | coverage gate green | **one direction only.** The gate refuses when a line in `PUBLISH_SET.txt` falls in no scan category, and never asked the mirror question — whether a stage that RUNS is shipped. Five stages were in that state (`patch_fabric_tripwire`, `patch_fsdp_wrap_policy`, `patch_master_addr`, `patch_nv_runtime`, `patch_train_phase_balance`), all real BLOCKER fixes, so the published `build_h100_plane.sh` referenced five files a clone would not have. Same shape as #157, mirrored | #190 (fixed; the gate that would keep it fixed, #191, does not exist) |
| post-mortem adjudication | #187 recorded **fixed** — the link now routes through the shared adjudication tail instead of exiting 0 over zero adjudications | **the tail is unreachable.** #187 sets a launcher-internal `FS_SKIP_TRAIN=1` and `rc=0` so the post-mortem arm falls into the shared tail; #171, added afterwards, put a trainer-verdict guard directly above that tail — `rc==0` is not PASS until the trainer's declaration is checked. On the post-mortem arm no trainer ran, so there is no declaration **by construction**, `fs_map_run_verdict` correctly returns 95, and the launcher exits above `adjudicate_tree` on every invocation. Measured on job **37347**, 8×H100 over the Gemma `out_probe` tree (2 checkpoints on disk): all 8 `srun` steps `COMPLETED`, `FS_COLLECTIVE world=8 ... verdict=OK`, `fs129: collective probe PASS`, and then a 51-line log containing `ADJUDICATE` **0 times**, ending `END rc=0 mapped_rc=95`; `sacct` `FAILED 95:0` in `00:02:36`. The launcher's own banner on that arm promises the tree `is adjudicated below`. **This is a regression, with a control on both sides:** job **37344** ran the same link on the pre-#189 artifact, where `fs175` occurs **0 times** — 71 lines, `ADJUDICATE complete root=... phase=post-mortem`, `END rc=0 phase=post-mortem checkpoint_saves_adjudicated=1`, `COMPLETED 0:0`. The tail was reachable until #189's fix landed above it. **CLOSED by job 37348**: with #193's stage landed (build stage 34, 5/5 controls fired, including `MUST_FIRE/EMPTY_TREE_STILL_95` so the fix cannot reopen #187), the identical submit returned `COMPLETED 0:0` in `00:05:13`, 91 lines, `ADJUDICATE complete … adjudicated=2 of 2 … ok=2 abstain=0 refuse=0`, `END rc=0 phase=post-mortem checkpoint_saves_adjudicated=2 checkpoint_saves_found=2`, and a logged line recording that the trainer-verdict mapper was deliberately not applied on this arm | #193 (CLOSED), jobs 37344 → 37347 → 37348 |

The #187 row is the cleanest instance of this section's thesis: the link exists for the
case where production died, and in that case it reported success having measured nothing —
`all([])` in a Slurm job costume. The repair did not add a new verdict; it stopped exiting
before the existing verdict could run.

#188 is what that verdict said once it could run, and it is the sharper lesson of the two.
A denominator derived from the stream it measures cannot detect its own truncation — no
amount of internal consistency checking would have caught it, because the report was
perfectly consistent. Only a count taken independently of the walk exposes it, which is why
the fix compares `processed` against `found` and refuses rather than reporting. #188 is also
the **second** orphan-stage occurrence after #136: a fix that exists in the repository, is
correct, is committed, and is in no execution list is indistinguishable from a fix that was
never written. That makes it a class, and the matrix records it as one — a gate that
cross-checks every `patch_*.py` against `STAGES ∪ PUBLISH_SET` is owed and is listed in E.6.

#189 is the third instance and the one that settles the class. Asking the #188 question of the
FAILURE path — *what does this actually do when it runs?* — found three findings recorded closed
and none of them present, held out of the build by the same mechanism. #190 is the mirror image:
five stages that ran and never shipped. The four together say that this build had **two**
independent membership lists and no gate reading either one against the other, so a fix could be
absent from execution or absent from publication and the build was green in both cases. All six
files are now wired in and the plane builds green at 37 stages. **How many other stages are in
this state is UNMEASURED**: a one-off audit today found two remaining files in neither list, of
which `patch_gate_launch_contract.py` is superseded — its sentinel is absent from
`gate_launch_contract.py`, but the hard-coded build-host `ROOT` it existed to remove is already
gone, so the effect is present by another route. An audit is not a gate. The number is true today
and nothing keeps it true; #191 is the gate that would, and it is listed in E.6.

#189 also demonstrates the difference between verifying a failure path and asserting one. The
fix was checked by *causing* a failure: `env_fs189.sh` sources the passing Phase 4 Qwen
environment verbatim and changes exactly two things — a scratch output directory and one
unrecognised flag on the trainer argv. The positive control is in the log, `fs_train.fixed.py:
error: unrecognized arguments: --fs189-deliberate-bad-flag` once per rank across all 8, so the
injected fault is what the run died of. Three claims are visible in that single terminal line and
none was before it: the launcher survived its own failure path and reached its own exit (#174);
the raw srun rc and the mapped code are both recorded, torchrun having flattened the trainer's
argparse rc 2 to 1, so the plane no longer republishes that flattened number as if it meant
something (#171); and the mapper refuses to read a missing verdict as an abstention — 5 RED, not
95 UNMEASURED, because nothing declared itself (#169).

#192 is the one row in this section whose fix is an **abstraction** question rather than a repair,
and it is worth stating as such because it is the shape the rest of this framework is trying to
have. The two quantities being compared against one scalar have different natural scales and
different provenance: restore fidelity is a property of serialization, observed at exactly `0.0`
on both models, and can hold a tight threshold forever; cross-rank agreement is a property of
kernels, scheduling and the model's own numerics, and its scale is not knowable in advance for a
model nobody has run yet. An operator-supplied constant cannot serve both, and asking the operator
to pick one is asking them to trade a real guarantee for a nuisance. The framework already
computes the number that resolves it — the **before-save** spread, which the trainer's own comment
calls "the measured noise floor of the instrument" — and then throws it away by comparing it
against the external tolerance instead of using it as the floor. Two knobs, one of them reading that
floor, generalizes to the next model; a widened scalar generalizes to nothing and silently disarms
a check that was working.

**Landed 2026-09-01 as build stage 35, `patch_resume_tolerance_split.py`** — and the landed rule
departs from the one argued for above, on the strength of this campaign's own doctrine. The
paragraph proposes a cross-rank knob *self-calibrating against the before-save floor*. Implemented
literally, that mints `rank_invariant = True` from a threshold derived from the run it judges,
which is a denominator derived from the stream it measures wearing a different hat. So the split
ships with self-calibration REPORTING and never CERTIFYING: `--resume-tolerance` governs restore
fidelity alone — `restore_delta > tolerance` stays RED and final, unreachable by any value of the
other knob — while the new optional `--rank-agreement-tolerance` (flag only, no env fallback,
matching its neighbour) carries the absolute cross-rank claim. Unset, the verdict states
`cross_rank_spread_before_save`, `cross_rank_spread_after_resume`, their signed
`cross_rank_spread_delta`, `rank_agreement_preserved` and
`rank_agreement_tolerance_source: "self-calibrated"`, and returns `unmeasured_cross_rank`.
`rank_invariant` can go True only under an explicit operator-supplied tolerance. That is the
generalization the paragraph was reaching for: the framework now *computes and publishes* the floor
for a model nobody has run yet, and still refuses to certify against it. Certified 10/10 static
gates and 8/8 controls, byte-idempotent, with the controls executing the real patched function.
`MUST_FIRE/RESTORE_RED_SURVIVES_WIDE_RANK_KNOB` — pre `[1.0]x8`, post `[1.5]x8`, restore tolerance
0.0005, rank tolerance 1000.0 — reads `status=red restore_delta=0.5`, where the pre-image function
on the identical vector reads `status=pass`; that pair is the #192 regression, pinned. The real
37336 vector reads `cross_rank_spread_delta=0.0`, the resume-attributable term the Gemma arm had no
field to state. **The claim is build-time only: no cluster job has run the split trainer.**

#193 is the first defect in this campaign that belongs to **neither** of the two fixes that
produce it. #187 is right: a link that reports success over zero adjudications is `all([])` in a
Slurm job costume. #171 is right: a run that exits clean having declared nothing is the
vacuous-pass hole, and it must be refused *before* anything downstream reads the tree. Each was
authored with controls, each fires exactly as designed, and composed they make the post-mortem
link incapable of the one thing it exists to do. The mechanism is a **precondition that was never
written down**: #171's guard assumes a trainer ran and was supposed to declare, and #187 built an
arm where nothing ran and nothing was supposed to declare. Because the guard's refusal is *honest*
on that arm — 95 is the correct code for a run that measured nothing — the composition produces no
contradiction anywhere for a gate to catch. Job 37347 exits with the right number for the wrong
reason, and the only tell is a banner promising an `ADJUDICATE` line that occurs zero times.

It is also, precisely, a **regression**, and the ordering is measurable rather than inferred.
`patch_launcher_exit_discipline.py` — the stage carrying the guard — was itself the #189 orphan, so
every build before #189 landed produced a launcher with nothing above the adjudication tail. Job
**37344** ran the post-mortem link on exactly that artifact and reached the tail: 71 lines,
`fs175` 0 times, one `ADJUDICATE complete`, `END rc=0 phase=post-mortem`, `COMPLETED 0:0`. Job
37347 ran it after and did not. Two jobs, one arm, one stage between them.

That is the part worth generalizing. #189 shipped with controls, and they were good ones — a
deliberately-failing training run (37346) for the failure arm and a passing training run for the
success arm. What it had no drill for was `FS_PHASE=post-mortem`, the third arm of a path it
inserted a guard into. The arms are not hidden: the launcher enumerates them in one variable. So
the rule this section should carry forward is narrower and more actionable than "compose
carefully" — **a fix that inserts a guard on a shared path owes a drill on every arm that path
serves**, and the arm list is a fact the artifact already states. #189 drilled one of three.

#193's fix therefore ships five drills, and the load-bearing one is not the drill that proves
adjudication now happens — it is the drill that proves a post-mortem over an **empty** tree still
exits 95, because the cheapest way to close this branch is to widen the hole #187 closed. All
five fired at build time against real bash, over fragments lifted verbatim from the pre- and
post-image rather than hand-copied: the pre-image reproduces 37347 (`exit=95`, zero
`adjudicate_tree` calls); the post-image reaches the tail (`exit=0`, one call, `END rc=0`); an
empty tree still exits 95; a tree covered 1 of 2 still exits 5 under fs176; and with
`FS_SKIP_TRAIN` unset the train arm is byte-for-byte the arm #171 was written for
(`exit=95`, zero calls).

The close is a measurement, not a build-time claim: the fixed launcher was shipped to the
cluster, checksum-verified, and the 37347 submit re-run verbatim as **job 37348** —
`COMPLETED 0:0`, 91 lines, `adjudicated=2 of 2 … ok=2 abstain=0 refuse=0`,
`checkpoint_saves_adjudicated=2 checkpoint_saves_found=2`. The before-and-after now has three
points on one arm: 37344 reached the tail, 37347 could not, 37348 does again — and the third
is the one that also proves the fix did not simply delete the guard, because the drills that
would have caught that are in the build the artifact came from.

Two earlier rows were found **in this matrix**, by applying its own rule to itself. That is
the point of keeping the section: a document that audits a framework and never finds a defect
in its own claims is not being audited. Neither was a code defect — the resolver refuses
correctly and the plugin fix is real — which is exactly the failure mode worth naming, because
a claim broader than its evidence is a defect even when the code underneath is right.

## E.5 Build plane (Deliverable C) — measured properties, 2026-08-31

`build_h100_plane.sh` regenerates every shipped artifact from stages that each refuse to
write while any of their own gates is red.

| Property | Status | How it was measured |
|---|---|---|
| stages green | **37/37** | full run, red stage aborts the build. Restated 2026-09-01 from a stale `17/17`, and again from a stale `36/36` when #182 landed as stage 37 -- that second staleness survived the gate's first pass because the gate only read `N/D stages`, with the noun AFTER the numbers, and this cell puts it before. The pattern now reads both orders: the count is now derived by `gate_doc_stage_count.py` from the build script's own `STAGES` array on every build, because a denominator that is retyped by hand is a denominator that lags (#194). |
| bidirectional env drift | **green** | `gate_env_drift.py`, all 3 detectors drilled with planted violations |
| public-repo blocklist | **0 hits / 5 files** | plus a planted-string control proving the pattern is live (1/1) |
| parse | **5/5** | `bash -n` ×2 and `py_compile` ×3 — each artifact checked by its own language's parser |
| generated unit suite | **12 passed** | the build RUNS it; missing pytest is an UNMEASURED that fails the build |
| input/output partition | **4/4** | every file the build touches is a declared artifact or a documented upstream; I1 drilled with a planted file each run |
| **from scratch** | **true** (since #136) | the un-generated intermediate is now removed and rebuilt every run |
| **deterministic** | **true** | two consecutive full rebuilds → byte-identical sha256 on all 5 artifacts |

Determinism is the property that makes the rest auditable: without it "the gates were green"
refers to a build nobody can reproduce. It was only checkable after #136, because until then
one input to every build was a file no stage produced.

Two of those rows are recent and worth naming. The suite row exists because a generated test
the build never executes is an orphan — #86 was exactly that, eight passing legs nobody ran —
so the build runs it, and an absent pytest is an UNMEASURED that turns the build red rather
than a skip that reads like a pass. `FS_SKIP_SUITE=1` waives it, but the waiver has to be said
out loud and is printed in the summary line. The blocklist and parse denominators moved from
3 to 5 in the same change, which is the point of writing denominators down: a "0 hits" that
silently covers fewer files than last week is indistinguishable from a clean result.

**#137, fixed the same day it was found.** Three files under `h100/gen/` were inputs, not
outputs. One of them, `launchers__launch_fs_h100.sh`, is the entire base text of the shipped
550-line launcher, and `find` over `fs-repo` returns no upstream for it — so unlike #136 it
cannot be re-derived at all, and it was sitting three lines from an `rm -f` over its own
directory. MUST_FIRE: moving it aside turns the build red at `apply_113.py`; restoring it
turns it green. The three now live in `h100/upstream/` with a README recording each one's
sha256, its reader, and whether an upstream exists; relocating them changed no output byte.

The generalizable half is the row above. #136 and #137 were the same defect twice — a file
the build *reads* sitting in the directory the build *writes*, so nothing distinguished it
from an artifact — and both were found by accident. `gate_build_inputs.py` is the on-purpose
version: it partitions every file into declared-artifact or documented-upstream and refuses
on a third category. It does not try to infer what the stages read by parsing them, because
static reading of arbitrary path construction is precisely what produced this project's worst
false readings; the set comparison is exact.

## E.6 Not yet exercised

Phase 3 and the Phase 4 arms have now run for the two small models in E.1.1, on one node
with 8 ranks. What remains unexercised, with the denominator that keeps it honest:

| Gap | Status | What is known |
|---|---|---|
| multi-node | UNMEASURED | every executed row in E.1.1 is one node, 8 ranks; E.3's socket row is a control, not a multi-node pass |
| larger declared models | UNMEASURED | E.1.3 denominator: 4 declared, 2 executed; the 4B, 7B, 8B and 27B rows have no load/train/checkpoint/resume/eval measurement here |
| Gemma-4 family | REFUSED at build / UNMEASURED | E.1.4 denominator: 3 rows, 0 executed; two are `NO — unrecognised`, one is UNMEASURED; no framework pass may be reported until the registration seam exists |
| non-`singularity` runtime | UNMEASURED here | enroot is present on another estate; #122 fixed the var-crossing defect, but no enroot run is in this matrix |
| Gemma divergence mechanism | UNMEASURED | E.1.2 lists what 37320-37323 ruled out; the mechanism inside Gemma is not root-caused and is not guessed |
| Gemma `resume.fixed_eval_rank_invariance` | UNMEASURED | declared abstention in **37336** at `--resume-tolerance 0.0005` (`unmeasured:["resume.fixed_eval_rank_invariance"]`); shown as abstention, not pass. Job 37319 passes the same divergence at tolerance `10.0` — see #192, closed at build time by stage 35, unrun on hardware |
| ~~orphan build stages~~ | **CLOSED — now MEASURED (#191)** | #136, #188, #189 and #190 were four instances of one class: a fix authored, verified and committed while appearing in neither `STAGES` nor `PUBLISH_SET.txt`. `gate_stage_orphans.py` now cross-reads the two sets in both directions on every build and partitions **54 files** into four declared states: `RUN_AND_SHIPPED 33`, **`RUN_NOT_SHIPPED 0`** — the direction that produced #189 and #190, now empty by measurement rather than by assumption — `SHIPPED_NOT_RUN 18` (build-driver 2, gate 8, library 1, runtime-artifact 4, test 3), `NEITHER 4` (developer-tool 2, superseded 2). Roles are declared in `h100/STAGE_ROLES.tsv`; 4/4 drills fire each run and the MUST_PASS output is identical. Scope, stated so it is not over-read: `runtime-artifact` is an **attestation** that a file executes on the cluster rather than in the build, and the declaration is not a measurement of that. The arm-coverage failure #193 exposes is a different mode that happens to share a stage — this gate does not address it |
| ~~Gemma tree under the fixed adjudicator~~ | **CLOSED — now MEASURED** | the re-walk was attempted first as job 37347 and returned nothing: 51-line log, `ADJUDICATE` 0 times, `END rc=0 mapped_rc=95`, sacct `FAILED 95:0`. That was **#193**, not the tree. After #193's stage landed as build stage 34 (5/5 controls fired), the identical submit ran as **job 37348**: `COMPLETED 0:0` in 5m13s, 91 lines, `ADJUDICATE complete … adjudicated=2 of 2 checkpoint dir(s) ok=2 abstain=0 refuse=0`, `END rc=0 phase=post-mortem checkpoint_saves_adjudicated=2 checkpoint_saves_found=2`. Both Gemma checkpoints green on all of A1–A7b (`checks_measured=10 checks_green=10 checks_red=0 legs_abstained=0`, `world_size=8`, `present 8/8 expected ranks`, `unexpected=[]`, `shards 8/8`, directory step agreeing with `manifest global_step` at 5 and 20). Numerator equals denominator, which is what the fs176 refusal requires. Both trees have now been re-walked: Qwen 3 of 3 (37345), Gemma 2 of 2 (37348) |
| invocation self-record | **CLOSED at build time (#180)** — UNMEASURED on hardware | The finding: on job 37319's own 936-line log, `FS_ENGINE_LAUNCH_CMD`, `--resume-tolerance`, `--model-path`, `--dataset-path` and `--sequence-length` occur 0 times combined; `tolerance = 10.0` was known only from the proof's PHASE_JSON echo — an accidental witness. The fix: build stage 36, `patch_launch_provenance.py`, writes `${RUN_LOG%.log}.provenance.json` immediately before the sole trainer exec and only on the arm that execs one, carrying the raw and composed command, world size and its source, the four directory knobs, phase, probe, job id, node count, and every `FS_` name the shell held — `compgen -v`, not `compgen -e`, because the exported-only oracle omitted the two bare knobs that define a resume segment. `10/10` static gates, `8/8` controls, including a census drill that goes red on the pre-image (`FS_BARE_KNOB present: False`). **Residual: no job has written one**, so the number this matrix can still cite is the pre-image's 0 occurrences |
| operator-facing text naming a file nothing ships | **CLOSED (#182)** | The launcher's `PROBE denominator:` line told operators that `FS_ITERATION_BUDGET` and `FS_EARLY_SAVE_STEPS` are read by `tools/fs_train.py` — a file with **0 consumers**, shipped by no stage, present in neither the publish set nor the repository. Seven sites across three shipped files repeated it, one of them naming a second phantom, `run_recipe.py`, that the finding never mentioned. All seven now name **the engine entrypoint given in `FS_ENGINE_LAUNCH_CMD`**, which is the real seam and is operator-supplied, so no filename is asserted at all. Build stage 37, `patch_engine_entrypoint_naming.py`: a **census** of every `*.py` token in operator-facing text (comments plus quoted `printf`/`echo`/`fail` strings) against `PUBLISH_SET.txt` ∪ the repository's `git ls-files`, `17/17` static gates, `10/10` controls. Its first run taught the denominator rule that now governs the plane's detectors: scoped to the publish set alone it produced **6 false REDs out of 9**, because that list enumerates only the `h100_validation/` subtree; and where the repository half cannot be resolved a not-found token is **95 UNMEASURED, never 5** |
| absolute line citations into a generated launcher | OPEN as a maintenance cost, detected not prevented | LAUNCH.md cites the launcher by absolute line number (`L:543`). Stage 37 added two comment lines at :481, and **32 citation numbers at or after that point shifted by +2** — 9 of them past the point where the L2 gate could still resolve them, which is how the shift was caught. The gate is the right shape (it refused a build over stale pointers, and its MUST_FIRE drills a renumbered citation red), but the repair is manual re-anchoring on every stage that changes the launcher's length. The structural fix would be symbolic anchors resolved at gate time instead of typed line numbers; not attempted here, and recorded so the next stage author expects the churn |
| stale status lines in shipped docs | **CLOSED for the stage count (#194)** — open as a class | LAUNCH.md led operators with "Phase 3 … has **never** been executed" for a full campaign after 37310 executed 8/8 legs, and four documents carried three different stage counts (`17/17`, `33`, `36`) for one build. Understatement is the same defect as overstatement: a status line that lags its evidence is unreliable in both directions. The stage count is now derived, not retyped — `gate_doc_stage_count.py` reads the build script's `STAGES` array and refuses on disagreement, with a planted-mismatch control. The narrative status lines were corrected by hand against the job record and are **not** machine-checked; that residual is stated rather than closed |
