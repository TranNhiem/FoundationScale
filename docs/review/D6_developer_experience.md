# D6 — Developer-experience review

**Scope and method.** This review was written as an AI engineer arriving cold with
one instruction: "train a model with this framework this week." Every workflow below
was attempted against the shipped tree only — what is in `src/foundationscale/`,
`launchers/`, `tools/`, `checks/`, `h100_validation/`, the README, the Makefile,
`pyproject.toml`, and the CI workflow. Where a workflow has no documented path in
that evidence, it is rated **Absent** rather than assigned a plausible invented one.
`src/foundationscale/` measures 18849 lines; `launchers/` contains 9500 shell LOC
plus 1615 Python LOC; `h100_validation/` (31313 .py LOC, 63 files) is a validation
harness, not a training product. The framework's own README says it plainly: "The
trainer itself is early." This review measures what "early" feels like from the
operator's chair.

**Ratings:** Works / Painful / Blocked / Absent. "Blocked" means a path exists but
a first-timer cannot complete it without estate-specific knowledge; "Absent" means
no shipped path exists.

---

## 1. Start a training job — **Painful**

**Today.** Two unrelated paths exist, and neither greets you.

Path A, the package: `pip install -e ".[checkpoint,dev,train]"`, then
`foundationscale-train` (registered in `pyproject.toml` under `[project.scripts]`,
target `foundationscale.train.cli:main`) or `python -m foundationscale.train.cli`.
What flags `src/foundationscale/train/cli.py` accepts is **unmeasured** — the
evidence contains no `--help` output for it. The trainer delegates:
`src/foundationscale/train/loop.py` validates the declared topology and profile,
then hands the step to `transformers.Trainer`. If `[train]` is missing, a guarded
import refuses with the remedy string `pip install 'foundationscale[train]'`, and
`checks/packaging_reachability.py` keeps that string honest. That refusal message
is the best piece of operator UX in the package.

Path B, the launchers: `sbatch launchers/launch_g4e4b_lora_1tray.sh` — except
sbatch no longer exists on the estate's login nodes (finding #51, cited in the
launcher header comment), the `#SBATCH` block is vestigial, and the only
executable backend is enroot via `launchers/fs_container_backend.sh`
(`FS_BACKEND=auto|slurm|enroot`). Both shipped launchers are hardwired to one
model (Gemma-4-E4B) on one tray of one estate.

**Single best change:** one documented, minimal `foundationscale-train` invocation
in the README Quickstart — model, dataset, one GPU — so path A exists in prose,
not just in packaging.

## 2. Configure a model — **Blocked**

**Today.** There is no model-configuration schema in evidence. `src/foundationscale/models/adapters.py`
and `src/foundationscale/models/__init__.py` exist; what they expose is unmeasured
from the evidence slice. `src/foundationscale/topology.py` validates a "declared
topology and profile," but no profile format is shown anywhere. In the launchers,
"configuring the model" means editing paths and `HF_MODEL` in a bash header —
and `HF_MODEL` had to be *exported* or an in-container probe died `KeyError:
'HF_MODEL'` on every launch, a defect found only by running (recorded in the
full-FT launcher comment, fix45-A2/#82).

**Single best change:** a single dataclass or JSON profile (name racing:
`topology.py` already validates one) with one worked example committed under
`examples/` — a path that exists at top level but whose contents are unmeasured.

## 3. Configure datasets — **Absent**

**Today.** Nothing in the package tree is a dataset abstraction. The `[train]`
extra pulls `datasets>=2.18`, so the intended path is "HuggingFace datasets via
`transformers.Trainer`," but no shipped code or doc says that. The launchers reach
data through an estate-specific env var (`FOXBRAIN_SFT_JSONLS`, exported by name in
the full-FT launcher because container-side python reads it from `os.environ`).
`h100_validation/estate.env.example` is the closest thing to a data-configuration
document, and it is a validation-harness artifact.

**Single best change:** one sentence in `train/cli.py`'s help plus one committed
example JSONL wired end-to-end. The launcher contract suite's census rule —
"every variable a container-side python body reads must be exported by the
launcher" — is the right instinct; it needs a Python-side twin.

## 4. Select GPUs — **Painful**

**Today.** In the package: unmeasured (nothing in evidence shows a `--gpus` flag on
`train/cli.py`). In the launchers: GPU selection is split across a static
`#SBATCH --gpus-per-node=4` header (vestigial), `FS_GPUS_PER_NODE` (108
occurrences across the launch plane and validation docs), and
`FS_ENGINE_PROCS_PER_NODE`. World-size arithmetic is done by hand in launcher
header comments ("world 16 -> 4, DP 8 -> 4, ga = 16/(4/1)/1 = 4"). The LoRA
launcher gets one thing very right: `EP=world` is refused on a measured-dense
base, with the measurement cited. But the operator computes gradient accumulation
in their head.

**Single best change:** one knob — total GPUs — from which the launcher derives
and *prints* DP/GA/world before launching. The arithmetic already exists in the
comments; it is executing in the wrong medium.

## 5. Single-node training — **Works, narrowly**

**Today.** This is the one training workflow with an enforced happy path, and it
lives in the LoRA launcher. The run order is gated, not suggested: `PROBE=1`
first (20 iterations, saves at 10 and 20, writes to a `_probe` dir, verifies
adapter attach counts, trainable-parameter census, chain-of-thought survival, and
the save path), then production into a stable `OUT_DIR`, then a
`--dependency=afterany` resume chain. `PROBE` unset reads as 0; any other value
is refused, and the launcher says why in prose. As single-node-operated-by-the-
author goes, this is genuinely good.

The narrowness: it works for Gemma-4-E4B LoRA/full-FT on that one tray. A
single-GPU laptop run through `train/cli.py` is unmeasured — the reference
environment is torch CPU (`pyproject.toml` pins the pytorch CPU wheel index), and
no evidence shows a smoke train on CPU.

**Single best change:** port the `PROBE=1` contract into the Python trainer so
`foundationscale-train --probe`-shaped smoke runs exist off the estate. (Flag name
here is a proposal, not an assertion of an existing flag.)

## 6. Multi-node distributed training — **Blocked**

**Today.** Both shipped launchers are single-node by construction: `--nodes=1`,
"1tray" in the filename, world = 4. `launchers/fs_container_backend.sh` contains
the machinery multi-node needs — torchrun in the enroot arm, NCCL pins
(`NCCL_SOCKET_IFNAME=bond0`, `NCCL_IB_HCA` deliberately unset, measured
multi-tray notes about `NCCL_MNNVL_ENABLE`) — and the README poster advertises
"1 GPU → 100+ nodes as a ladder with measured rungs," but no shipped launcher
crosses a node boundary. The 4-tray reference launchers cited in the LoRA header
are estate-side, not in this tree.

**Single best change:** ship one 2-node launcher, even estate-shaped, so the
multi-node arm of `fs_container_backend.sh` has an executable caller in-tree.

## 7. Change parallelism strategies — **Painful**

**Today.** The full-FT launcher header lists env overrides: `TP CP EP ETP
SEQ_LENGTH GBS MBS EPOCHS TRAIN_ITERS`. Defaults are stated with justification
(`TP=1 CP=1 PP=1 DP=4; EP=4 only if config.text_config.enable_moe_block=true`).
The knob surface exists and is honestly documented *in that one file*. What is
missing: any validation of a bad combination before GPU-seconds are spent, and
any presence of these knobs in the Python package — `topology.py` is presumably
their future home, but no evidence connects them yet. The README's DP/TP/PP/EP
table is marketing-poster prose, not configuration documentation.

**Single best change:** a preflight leg (the pattern already exists —
`tools/preflight.py` at 3593 LOC is the largest tool in `tools/`) that refuses
impossible TP/PP/EP shapes against the measured model config, the way EP=world is
already refused on a dense base.

## 8. Customize the training workload — **Painful**

**Today.** The package trainer deliberately owns no `nn.Module`: `train/loop.py`
delegates the step to `transformers.Trainer`, so customization today means
"whatever `TrainingArguments` exposes," reached through an undocumented CLI. The
full-FT launcher lists `EXTRA_OVERRIDES` as an escape hatch — a string spliced
into a CLI, which is the estate pattern this repository elsewhere audits as debt.
`FS_ITERATION_BUDGET`, `FS_EARLY_SAVE_STEPS`, `FS_WALLTIME` shape runs from the
environment.

**Single best change:** document the delegation. One section — "the trainer is
`transformers.Trainer`; here is how arguments reach it" — converts an
architectural decision the reader must reverse-engineer into a feature.

## 9. Add a new model — **Absent**

**Today.** There is no documented model-onboarding path. `src/foundationscale/models/adapters.py`
(294 LOC) is the only plausible seam and nothing in evidence describes its
contract. Empirically, "add a new model" has been done exactly once in this
lineage: copy a launcher, measure the base `config.json` by hand (the LoRA
launcher cites `enable_moe_block=False and num_experts=null, both measured off
the base config.json`), and rewrite the header. That works for an expert with
cluster access and is not reproducible by anyone else.

**Single best change:** the checklist that already exists implicitly in the LoRA
launcher comment (measure MoE-ness, census tensor names, choose targets) written
as `docs/` content with the census tooling named —
`launchers/lora_target_census.py` and `launchers/peft_override_replay.py` are
real, and no prose points at them.

## 10. Add a new dataset — **Absent**

**Today.** See workflow 3. The one dataset-specific behavior with evidence is
negative-space: the v3 corpus needed no inference-side template patch because the
`-it` checkpoint ships `chat_template.jinja` — knowledge recorded in a launcher
comment, discoverable only by reading it.

**Single best change:** one documented data contract (columns, template
expectations) in the package, of the shape B2's "one data contract" promises.

## 11. Modify training logic — **Painful but honest**

**Today.** Edit `src/foundationscale/train/loop.py` (1168 LOC) or the gates.
Then the gauntlet, which is this repository's pride and its toll booth:
`make check` = lint, typecheck, skip-guard-probe, test (coverage floor 90),
controls, packaging, countables, mutation. Frictions I hit:

- `make test` skips differently on a laptop than CI (`FS_FORBID_SKIPS` unset by
  design; skips are *named* but allowed locally) — same command, two verdicts.
- The full mutation battery is ~40s per row, and the Makefile states an
  *extrapolated, not measured* ~50 minutes for the corpus; the shard mirror is
  `make mutation-module MODULE=<name>`, discoverable only by reading the
  Makefile. CI shards it across a job matrix derived from the mutation table;
  locally you get one big bill.
- mypy checks `src` plus three named `tools/` files; all 7 files under `checks/`
  are unchecked by mypy (stated, with the stale-count retraction attached).
- `checks/countables_drift.py` will fail your docs edit if a countable wording
  drifts — correct, and a surprise the first time it happens.

**Single best change:** a `CONTRIBUTING` section naming the fast inner loop
(`pytest`, `ruff`, one mutation shard) versus the full `make check`, with the
wall-clock costs the Makefile already knows.

## 12. Debug a failed job — **Works**

**Today.** This is the framework's home turf, and it shows. Gates fail closed
(`ERROR` blocks; `VACUOUS` blocks; exceptions in a gate never read as passes).
Verdicts carry coverage denominators. The launch plane has a failure vocabulary:
`gate_exit_contract.py`, exit codes outside a published namespace treated as
defects, refusal messages that name the knob and the remedy (`FS_ALLOWED_NODE`'s
unset refusal prints an example value). `h100_validation/` adds EVIDENCE.md,
LAUNCH.md, checkpoint scalar extractors (`fs_ckpt_scalars.py`), and an
adjudicator. The culture — "a number wrapped across a comment continuation is a
number in no denominator" — means failures tend to arrive with their denominator
attached. The strongest operator experience in the repository, by a wide margin.

**Single best change:** nothing structural. Lift LAUNCH.md's debugging content
into the top-level docs so a non-H100 reader finds it.

## 13. Resume from a checkpoint — **Painful**

**Today.** The checkpoint machinery is the deepest module in the package
(`checkpoint/dcp.py`, `checkpoint/dcp_meta.py`, `verify/parity.py`), and the
launch plane has a real resume vocabulary: `FS_RESUME_CKPT` (21 occurrences),
`FS_RESUME_STEP`, `FS_RESUME_CKPT_EXTRA`, an estate-proven no-op resume chain
(`sbatch --dependency=afterany`, exits 0 once `last_iter >= train_iters`), plus
dedicated patching history (`patch_resume_env.py`,
`patch_resume_tolerance_split.py`, `patch_resume_proof_attribution.py`) that
testifies to how hard resume was to get right. What is missing is the seam: no
evidence shows a `--resume` flag on `foundationscale-train`, so package-side
resume behavior is unmeasured.

**Single best change:** one resume knob in the Python trainer whose semantics are
the ones the launch plane already proved out.

## 14. Monitor training performance — **Painful**

**Today.** What exists is *correctness* monitoring: the live save gate with a
wall-clock watchdog (`FS_GATE_TIMEOUT_S`), first-save adjudication
(`tools/live_save_gate.py` over `gates/adjudication.py`), checkpoint scalar
extraction, early-save cadence (`FS_EARLY_SAVE_STEPS`). What does not exist in
any evidence: loss/metric logging integration — no wandb, no tensorboard, no
metrics sink named anywhere. The README's own incident (`grad_norm` exactly
0.000 for 472 steps under `success=1.00`) argues for a metric-plausibility gate,
and it is specified but not landed.

**Single best change:** the metric-plausibility gate the audit already specified —
it is the one monitor this framework is uniquely positioned to ship.

## 15. Compare experiments — **Absent**

**Today.** Provenance exists: `src/foundationscale/provenance/manifest.py` (2623
LOC), `tools/emit_run_manifest.py`, which "refuses dishonest emissions," and
run-identity knobs (`FS_RUN_ID`, `FS_ATTEMPT`, read by `train/loop.py`). But no
tool in evidence compares two manifests, diffs two runs' scalars, or tabulates
experiments. The material for comparison is collected; the comparison is not.

**Single best change:** a `diff` verb over two run manifests. The manifest schema
is the hard part and it exists.

## 16. Scale from H100 to GB200 — **Absent (in-tree)**

**Today.** The audit covered both platforms, and GB200-specific operational
knowledge is recorded where it was earned — the `NCCL_MNNVL_ENABLE=0` note in
`fs_container_backend.sh` with its honest denominator ("measured on 4 trays, not
1"). But the shipped validation harness is H100-only by name
(`h100_validation/`, `h100_validation/h100/`, `launch_fs_h100.fixed.sh` under
`gen/`), and no GB200 launcher, plane, or gate set appears in the tree. Scaling
guidance for GB200 is unmeasured.

**Single best change:** state on the README Status list that GB200 is
audit-knowledge only, so nobody reads the poster's hardware span as shipped
capability.

## 17. Scale from a small experiment to large-scale training — **Painful, with the right shape**

**Today.** The *intended* ladder exists and is enforced at the single-node rung:
`PROBE=1` → production → resume chain (workflow 5). Above that rung, the ladder
is documentation: B2 promises "1 GPU → 100+ nodes as a ladder with measured
rungs," but rungs 2 through 100 have no shipped launcher (workflow 6). The
scaling story's strong half is that promotion between rungs is gated rather than
assumed; its weak half is that there is currently one rung.

**Single best change:** make rung two real (workflow 6's fix), then write the
ladder table with one executable artifact per rung.

---

## Personas: what each hits first

**Beginner.** Opens the README and meets an audit, not a quickstart. The
Quickstart that exists teaches you to *verify the framework* (pytest, controls,
mutation battery), never to train a model. First stuck point: "where is train?"
The answer (`foundationscale-train`, `[train]` extra) is reachable only from
`pyproject.toml`. Everything a beginner touches first is polished — which sets an
expectation the trainer cannot yet meet.

**Intermediate.** Installs `.[checkpoint,dev,train]`, runs
`foundationscale-train`, and immediately needs a model profile, a dataset
contract, and GPU selection — three consecutive Absents/Painfuls (workflows 2,
3, 4). The guarded-import remedy string is their best friend; it is also the
last helpful message they will see.

**Advanced.** Modifies gates or `train/loop.py` and meets `make check`. The
checks are legible and every one has a comment explaining the incident that
motivated it — excellent — but the wall-clock shape (mutation extrapolated ~50
minutes full; shards via a Makefile variable) and the local/CI divergence on
skips bite early. First red is likely `countables_drift.py` over a docs number,
which feels adversarial until the comment explains it.

**Expert.** Works on the launch plane. Meets measured estate knowledge of a
density rarely written down (the `fs_container_backend.sh` header alone encodes
a dozen measured failure modes), but also the sharp edges: the contract suite is
CWD-sensitive by necessity ("0/8 launcher unreadable" from the wrong directory),
bash 3.2 vs 5 portability scars (`mapfile` does not exist on macOS; counters in
piped loops die in subshells), and a knob surface of dozens of `FS_*` variables
whose only registry is a grep.

## What would make me confused, slow, or afraid to modify the framework?

- **Confused:** the repository is two things — an audit and a framework — and the
  seams between `src/`, `launchers/`, and `h100_validation/` are organizational,
  not architectural. It took real reading to learn that the launchers do not call
  the package trainer at all; they run an estate engine (`Megatron-Bridge`)
  through a container backend.
- **Slow:** the mutation battery's cost profile, and the fact that several checks
  exist precisely to be slow (one full pytest run per mutation row, measured
  ~40s/row on one machine). Also: `python` vs `python3` has already bitten once
  (#232 — `make check` died command-not-found outside CI), so every doc snippet
  is a small portability negotiation.
- **Afraid:** the gates' own self-discipline cuts both ways. `countables_drift.py`
  anchors *wordings*; the launcher contract floor is a hand-maintained number
  (`FLOOR=154`) that has already gone stale once and produced a confounded red
  that sent a reader after the wrong defect. The comment trail is honest about
  both, which reassures — but a contributor learns that touching almost anything
  moves a denominator somewhere, and the framework will notice. That is the
  point, and it is also a real activation energy.

## Simple defaults that do NOT limit advanced customization

**Where it gets this right:**

- `Verdict` semantics default to blocking: `Gate.ok()` cannot mint a PASS over
  zero coverage no matter what the author writes; "nothing to check" requires an
  explicit `skip()` with a reason. The safe path is the default path, and the
  advanced path (a declared sample, a scoped skip) is fully open.
- `FS_BACKEND=auto` resolves slurm-inside-allocation vs enroot-otherwise;
  `CLUSTER_HOME` defaults to `$HOME` and documents why that default is
  behavior-preserving on the measured tray. Both are overridable.
- GPU-drain guard defaults (`FS_GPU_DRAIN_MAX_MIB` 2048, `FS_GPU_DRAIN_TIMEOUT_S`
  1800) — tuned from a measured segfault-and-hang, overridable, and the timeout
  refuses rather than warns.
- The guarded import in `train/loop.py` refuses with the remedy printed, and a
  packaging gate keeps the printed command true. Defaults as telemetry.
- `PROBE=1` rejects every value except 1 and unset — a boolean that cannot be
  half-set, because "silence about this knob was finding #81."

**Where it fails it:**

- `FS_ALLOWED_NODE` has no default *by design* — correct for a standing safety
  rule, but there is no est ate-level example env in the package; the only
  template is `h100_validation/estate.env.example`, one directory too deep to be
  discovered.
- Dozens of `FS_*` knobs, no registry. Occurrence counts in the hundreds
  (`FS_ALLOCATION` 155, `FS_ALLOWED_NODE` 142) say the launch plane *is* the
  configuration system, and its documentation is the union of launcher header
  comments and h100_validation's LAUNCH.md.
- The Python package has almost no configuration surface at all — the opposite
  failure. Defaults cannot be simple where there is nothing to default.

## The five knobs a first-time user must set and cannot discover

All five come from the launcher path, because the package path does not expose
knobs at all (itself the sixth finding).

1. **`FS_ALLOWED_NODE`** — required, no default, unset refuses. Discoverable only
   by *failing* — the refusal message is exemplary, but no doc front-loads it.
2. **`CLUSTER_HOME`** — every path default reroots on it; lives in a bash
   parameter expansion in the launcher preamble.
3. **`FS_CONTAINER_SQSH`** — the container image; the enroot store semantics
   (`ENROOT_DATA_PATH=$HOME/.enroot`, unrecorded-provenance reuse is a hard
   error) make a wrong image an expensive mistake, and the knob is named only in
   backend code and validation docs.
4. **`FS_BACKEND`** — `auto|slurm|enroot`, where the estate's truth is "enroot is
   the only executable arm today" — documented in `fs_container_backend.sh`'s
   header, which is a library file a first-timer has no reason to open.
5. **`HF_MODEL`** — must be set *and exported*; the measured consequence of
   missing the export was a `KeyError` from an in-container probe before any
   training GPU-seconds. Found, per the launcher's own comment, only by running
   it. A discoverable-in-advance config schema would have caught it statically.

---

## Summary

The developer experience is bimodal, and the boundary is exactly the one the
README declares: the **verification plane** (gates, controls, contracts,
provenance, debugging) is the best-instrumented surface I have been asked to
operate; the **training plane** above it is one frozen-excellent estate launcher
pair, one early delegation to `transformers.Trainer`, and a gap where
configuration, datasets, multi-node, and experiment comparison will live. The
seventeen workflows score: 2 Works, 9 Painful, 3 Blocked, 3 Absent (workflow 5's
"Works" is narrow and workflow 12's is genuine). The fastest improvements all
have the same shape: the knowledge already exists, in comments with measured
denominators, one directory away from the person who needs it. Move the prose to
where the operator stands, ship rung two of the ladder, and D6 halves its reds.

*Unmeasured here (stated, not passed): the actual CLI surface of
`foundationscale-train` (would be measured by running `--help` against
`src/foundationscale/train/cli.py`); the contents of `examples/` and `docs/`;
behavior of `train/loop.py` on a real single-node run outside the estate (one
`foundationscale-train` invocation with the `[train]` extra installed).*
