# Deliverable E — Compatibility Matrix

**Status: IN PROGRESS.** Every cell is either a measurement with a date, or the literal
token `UNMEASURED`. No cell is blank, and no cell is inferred from a neighbouring cell.
An empty matrix would satisfy "all rows pass" vacuously; that is the failure this project
exists to prevent, so absence of evidence is written down as absence, not as a pass.

Estate: 8× H100 SXM per node, Slurm partition capped at 7 days, container runtime
`singularity` (enroot present on the other estate — the framework must not assume either).

## E.1 Models

"Builds" below means: config resolved by search, `AutoConfig` accepted it, and the
architecture instantiated on a meta device inside the container (transformers 4.57.1,
`PYTHONNOUSERSITE=1`). It does **not** mean weights were materialised — that is Phase 3/4.

| Model | model_type | Architecture | Params | Root layout | Binds | Builds | Distributed | Checkpoint | Resume | Eval | Status |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Qwen3-4B-Instruct-2507 | `qwen3` | Qwen3ForCausalLM | 4.02B | self-contained | 1 | **YES** | UNMEASURED | UNMEASURED | UNMEASURED | UNMEASURED | **Phase 3 pick** |
| qwen25_7B_Instruct | `qwen2` | Qwen2ForCausalLM | 7.62B | self-contained | 1 | **YES** | UNMEASURED | UNMEASURED | UNMEASURED | UNMEASURED | Phase 4 (Qwen leg) |
| llama31_8B_Instruct | `llama` | LlamaForCausalLM | 8.03B | commit-SHA nested | 1 | **YES** | UNMEASURED | UNMEASURED | UNMEASURED | UNMEASURED | Phase 4 (3rd arch) |
| gemma3_27B | `gemma3` | Gemma3ForConditionalGeneration | 27.43B | self-contained | 1 | **YES** | UNMEASURED | UNMEASURED | UNMEASURED | UNMEASURED | **Phase 4 (Gemma leg)** |
| gemma-4-12B-IT | `gemma4_unified` | Gemma4UnifiedForConditionalGeneration | — | config-overlay + symlink | **2** | **NO** — unrecognised | blocked | blocked | blocked | blocked | needs registration seam |
| gemma-4-31b-it | `gemma4` | — | — | self-contained | 1 | **NO** — unrecognised | blocked | blocked | blocked | blocked | needs registration seam |
| gemma-4-26B-A4B | UNMEASURED | UNMEASURED | — | config-overlay + symlink | **2** | UNMEASURED | UNMEASURED | UNMEASURED | UNMEASURED | UNMEASURED | stretch |

Notes on the two NO rows: transformers 4.57.1 in the container rejects both with
"does not recognize this architecture" for `gemma4_unified` and `gemma4`. These are the
estate's out-of-tree checkpoints (ref #119). They are the concrete instance of the standing
question "is this a one-time workaround or does it belong in the abstraction?" — the answer
is that FoundationScale needs an **architecture registration seam** (a declared, verified
extension point) rather than a core edit teaching it about one vendor's fork. Until that
seam exists these rows stay NO, and no Gemma-4 result may be reported as a framework pass.

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

The last two rows were found **in this matrix**, by applying its own rule to itself. That is
the point of keeping the section: a document that audits a framework and never finds a defect
in its own claims is not being audited. Neither was a code defect — the resolver refuses
correctly and the plugin fix is real — which is exactly the failure mode worth naming, because
a claim broader than its evidence is a defect even when the code underneath is right.

## E.5 Build plane (Deliverable C) — measured properties, 2026-08-31

`build_h100_plane.sh` regenerates every shipped artifact from stages that each refuse to
write while any of their own gates is red.

| Property | Status | How it was measured |
|---|---|---|
| stages green | **17/17** | full run, red stage aborts the build |
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

Phase 3 (minimal genuine 8×H100 run: load → distributed → checkpoint → resume → eval) has
not run. Until it does, every per-model column in E.1 stays UNMEASURED. Phase 4 (Gemma +
Qwen) follows Phase 3.
