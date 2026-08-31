# H100 validation — measured evidence ledger

Only things actually executed. Each row states what was run and what came back.
Inferences are marked INFERRED and kept separate from measurements. Anything not
measured is absent from this file rather than assumed.

Account: backup (<backup-account>). Primary (<primary-account>) is Permission denied on this cluster —
see CONFOUND-2.

---

## M1 — target node

    sinfo/scontrol -> <h100-node>  gres gpu:H100:8(S:0-1)  112 CPUs  2 TB RAM  state=idle

## M2 — container runtimes

    singularity  /usr/bin/singularity   PRESENT
    apptainer                           ABSENT
    enroot                              ABSENT
    srun, sbatch                        PRESENT

Consequence: on this estate the singularity arm is not one option of two, it is
the ONLY arm. The enroot arm — which is the arm the existing framework actually
implements (58 enroot references, 0 singularity) — cannot run here at all.

## M3 — L1 confirmed by execution, not inference   [BLOCKER]

The launcher's GPU probe (`launchers__launch_fs_h100.sh:159-167`) was executed
verbatim on the cluster:

    host python3                 -> Python 3.6.8
    python3 -c "import torch"    -> ModuleNotFoundError: No module named 'torch'
    python3.12 on host           -> ABSENT
    probe's $visible             -> ""   (empty)
    [[ "$visible" =~ ^[0-9]+$ ]] -> false
    result                       -> fail 96 "could not measure visible CUDA devices"

**The generated launcher is unrunnable on this estate.** Every job dies at :165,
before `fs_backend_runtime_setup` at :171 ever runs, so the container is never
even entered. This is not node-dependent: the alternative branch ("host torch
exists and the probe silently measures the wrong CUDA stack") is unreachable
here because no host torch exists at any version.

Note the failure is *correctly directed* — it fails closed, refusing to guess a
device count. The defect is that the measurement is taken in the wrong place:
on the host interpreter, before the runtime that will do the training exists.

Superseded diagnosis, recorded so it is not re-derived: this was originally
called a "host user-site torch leak". That is REFUTED — `PYTHONNOUSERSITE=1` is
exported at :83, well before the probe, and closes the user-site path. The real
defect is broader and does not involve user-site at all.

## M4 — L2 confirmed by executing the pattern

`:67` claims to prove the submitted TimeLimit is within the partition max. Its
last alternative is `TimeLimit=*:*`, which matches any value containing a colon.
Executed:

    TimeLimit=8-00:00:00   -> ADMITTED
    TimeLimit=99-00:00:00  -> ADMITTED

The disjunction cannot fail, so it states no coverage — `all([])` in glob form.
The guard's own error string promises a proof it does not perform.

## M5 — L4 confirmed by search

`FS_ITERATION_BUDGET` and `FS_EARLY_SAVE_STEPS` are exported at :107 and read in
NONE of the three generated artifacts. The probe phase's "20 iterations, save at
5" is decorative; the probe runs at whatever the engine defaults to, which is
the one thing a probe phase exists to bound.

## M6 — the container boundary drops 9 of 10 required variables

The launcher exports **14** variables (an earlier count of 12 was wrong; :107
exports three names, not one). Before the fix, `FS_ENV_ALLOWLIST` forwarded
exactly one of them (`PYTHONNOUSERSITE`). Nine consequential variables were
dropped silently, including `HF_HUB_OFFLINE`/`TRANSFORMERS_OFFLINE` (container
reaches for the network on an offline estate), `OUT_DIR` (trainer loses its
output path), and `TORCH_NCCL_ASYNC_ERROR_HANDLING` (hangs instead of failing
fast on an NCCL error).

Post-fix, executed against the real spliced file:

    must-cross forwarded      10 / 10
    host-side-only held        4 / 4
    SLURM_JOB_ID_EXTRA         held (no prefix-match leak)

## M7 — the torch-provenance detector is certified, with its control observed

    fs_selftest_torch_provenance          -> "4 of 4", rc 0
    detector sabotaged to always-accept   -> rc 1   (MUST_FIRE went red)

The drill discriminates in both directions: always-accept leaves the MUST_FIRE
unfired, always-reject fails the MUST_PASS. A drill that cannot fail proves
nothing, so this control is the reason the "4 of 4" is worth stating.

## M8 — splice integrity

    functions      10 -> 13   (none lost)
    enroot refs     59 -> 79   (rose; the enroot arm is not damaged)
    singularity     46 refs present
    fs_* calls      12 distinct, 13 defined, 0 undefined
    bash -n         clean

## M9 — the two Phase-4 models, read from their real config.json

| | Gemma-4-12B-IT | Qwen2.5-VL-3B-Instruct |
|---|---|---|
| architectures | `Gemma4UnifiedForConditionalGeneration` | `Qwen2_5_VLForConditionalGeneration` |
| model_type | `gemma4_unified` | `qwen2_5_vl` |
| hidden / layers | 3840 / 48 | 2048 / 36 |
| attn heads / KV heads | 16 / 8  (GQA 2:1) | 16 / 2  (GQA 8:1) |
| vocab | 262,144 | 151,936 |
| torch_dtype in config | **absent (None)** | `bfloat16` |
| MoE | dense | dense |
| vision tower | yes | yes |
| on disk | 1 shard, 23 G | 2 shards, 7.1 G |

Good pair for the model-agnosticism claim: different GQA ratios (2:1 vs 8:1),
1.7x vocab difference, different shard counts, two unrelated architecture
classes.

**TRAP (model-agnosticism, concrete).** Gemma's config declares no
`torch_dtype`. Any code that does the obvious `getattr(torch, config.torch_dtype)`
or passes it straight to `from_pretrained(torch_dtype=...)` gets `None` and
either crashes or silently loads fp32 — a 2x memory difference that would be
read as "Gemma needs more memory than Qwen" rather than as a config defect.
Precision must be resolved from the framework's own config with an explicit
default, never inherited from the checkpoint and assumed present.

**COVERAGE HOLE — must appear in the compatibility matrix as a hole, not a
pass.** Both available models are DENSE and both are MULTIMODAL. So a Phase-4
run over this pair:
  * cannot exercise the MoE expert-FQN logic, which is where most of
    `checkpoint_gates.py`'s complexity and most of its prior defects live;
  * cannot exercise a text-only (no vision tower) path.
Two green rows here would certify the dense-VLM case and nothing else. Claiming
"model-agnostic: verified" off this pair would be a claim broader than its
evidence.

## M11 — partition and scheduler, measured

    partitions visible to this account : exactly ONE -- <h100-partition>
    <h100-partition> max walltime                  : 7-00:00:00
    <h100-partition> nodes                         : 1  (<h100-node>)
    queue                              : EMPTY (0 jobs, mine or anyone's)
    <h100-node>                             : idle, 0/112 CPUs allocated, 8 H100 free

L2's proposed oracle works: `sinfo -h -p <h100-partition> -o '%l'` returns exactly
`7-00:00:00` — a parseable value, so parse-and-compare has something real to
read and the literal need not be hard-coded.

**COVERAGE HOLE — the estate is a SINGLE NODE.** Multi-*node* distributed
training, inter-node NCCL, and node-failure recovery cannot be validated here at
all. Only single-node 8-GPU is testable. The compatibility matrix must say
"untestable on this estate", not leave the column blank or imply a pass.

## M12 — Phase 3 distributed layer: PASSES on real hardware

One `srun -p <h100-partition> -N1 --gres=gpu:8`, in-container `torchrun --nproc_per_node=8`:

    torch          2.11.0a0+eb65b36914.nv26.02
    torch.__file__ /usr/local/lib/python3.12/dist-packages/torch/__init__.py
    device_count   8        world_size 8
    GPU            NVIDIA H100 80GB HBM3
    NCCL           2.29.2
    all_reduce correctness   got 28.0, want 28.0  -> PASS
    all_reduce 256 MiB       1.12 ms, 420.8 GB/s  -> NVLink-class
    memory alloc/reserved    0.25 / 0.27 GiB

The correctness check is a real one, not a smoke test: the reduction sums ranks
0..7, so the expected 28.0 is only produced if every rank contributed. A no-op
all-reduce, a partial reduce, or a wrong-group reduce all yield a different
number.

Covers three objective-list items — GPU allocation, multi-GPU communication,
container reproducibility — and confirms the M10 leak fix holds inside a real
Slurm allocation, not only in a login-node `singularity exec`.

Still NOT covered by this: model loading, dataset, checkpoint, resume, eval.
M12 is the transport layer only.

## M13 — the container is a VENV with EIGHT site dirs; the provenance prefix is 1 of them

    sys.executable  /opt/venv/bin/python3
    sys.prefix      /opt/venv        base_prefix  /usr
    purelib         /opt/venv/lib/python3.12/site-packages     <- pip's default target
    site.getsitepackages() -> 8 directories

    torch        -> /usr/local/lib/python3.12/dist-packages/torch/__init__.py
    transformers -> /opt/venv/lib/python3.12/site-packages/transformers/__init__.py

**CORRECTION to M10.** M10 recorded `/usr/local/lib/python3.12/dist-packages` as
"the container prefix, now measured on this image". Measured for TORCH — yes.
"The container prefix" — no: it is 1 of 8 legitimate site dirs, and it excludes
purelib, which is where pip installs by default. The claim was broader than its
evidence, which is the exact defect shape catalogued throughout this estate.
`fs_assert_torch_provenance` pins that single literal, so an image rebuild that
pip-installs torch into the venv would make it reject legitimate container torch
and hard-block every run via fs_die — fail-closed in the wrong direction, with a
confidently wrong message. Tracked as its own task.

## M14 — the image does NOT contain what its name says

    nemo                ABSENT        transformer_engine  present
    nemo_automodel      ABSENT        peft                present
    megatron            ABSENT        accelerate          present
    megatron.core       ABSENT        datasets            present
    megatron.bridge     ABSENT        flash_attn          present
                                      transformers        5.5.0
                                      torch               2.11.0a0+...nv26.02

`nemo-automodel-26-04.sif` is an **HF-stack image**, not a NeMo/Megatron image.
Anyone reasoning from the filename will be wrong about the available engine.

Consequences, and they are the load-bearing ones for the architecture review:

  * The framework is **engine-bound as well as runtime-bound.** It is written
    against Megatron-Bridge; this estate has no Megatron in the only image
    available. Combined with M2 (enroot absent, singularity only), the H100
    estate supplies NEITHER the runtime NOR the engine FoundationScale assumes.
  * This is direct evidence for the `naming_convention` defect: a manifest that
    defaults to `"megatron-core"` would assert Megatron provenance on a node
    where Megatron does not exist. The default asserts something never observed.
  * Any H100 training path must go through the HF/accelerate stack (or raw
    torch FSDP), not Megatron — which is a different model-loading seam than
    the one the framework currently implements.

## M15 — architecture support in the container, measured

    Qwen2.5-VL-3B-Instruct   AutoConfig OK -> Qwen2_5_VLConfig, qwen2_5_vl
    Gemma-4-12B-IT           AutoConfig FAILS: model type `gemma4_unified`
                             not recognized by transformers 5.5.0

**SUPERSEDED BY M17 — do not cite the Gemma row as a family verdict.** I probed
ONE Gemma-4 checkpoint and wrote the result up as "transformers 5.5.0 does not
recognize Gemma-4", a Phase 4 blocker. M17 measures five checkpoints and finds
stock Gemma-4 loads fine. The failing artifact declares a NON-STOCK model_type
(`gemma4_unified`); that is a property of that one checkpoint, not of the model
family and not of the toolchain. Retracted rather than quietly amended, because
the blocker claim had already propagated into the task ledger.

Phase 4 needs one Gemma and one Qwen. Both are available — see M17.

**REFUTES my own earlier prediction (M9's TRAP).** I predicted that Gemma's
absent `torch_dtype` would make naive dtype resolution crash or silently load
fp32. Measured: transformers 5.5.0 normalises `torch_dtype` into a real
`torch.dtype` object (Qwen yields `torch.bfloat16`, not the string), so the
naive `getattr(torch, dt)` path does not arise on loading. The Gemma half stays
untested because AutoConfig fails first. The prediction was made from reading
config.json alone; running it refuted it. Recording so it is not re-derived.

Still real from that area: transformers 5.5.0 warns "`torch_dtype` is
deprecated! Use `dtype` instead!", so framework code reading `config.torch_dtype`
is on a deprecation path.

## M16 — the mount plane: measured on the cluster AND in the source

Two-arm control against the real image, no other change between arms:

    ARM A  singularity exec <sif> ls /work
             -> ls: cannot access '/work': No such file or directory
    ARM B  singularity exec --bind /work:/work <sif> ls /work
             -> 0001  0097  0098

    Default in-container mounts with NO --bind:
      /  /etc/group  /etc/hosts  /etc/passwd  /etc/resolv.conf
      /home/<user>  /tmp  /usr/share/zoneinfo/Etc/UTC  /var/tmp

(Honesty note: the `rc=0` in that transcript is an artifact of my own quoting —
`$?` expanded host-side, not in-container. The `ls: cannot access` line is the
evidence; the rc is not.)

And in the source (`fs_container_backend.spliced.sh`):

    enroot arm       :982   local -a mounts=("$HOME:$HOME" /dev:/dev /sys:/sys)
    singularity arm  :1039  local -a sargs=(exec --no-home --pwd "$spwd")

So **neither** runtime mounts the filesystem root holding every model, dataset
and the framework tree. This is not a singularity-only defect, which is how I
first logged it.

**The default-bind policy causes BOTH container defects on this estate, in
opposite directions.** It mounts what must not be present — host `$HOME`, hence
`~/.local/.../torch` reaching the container (M10/#107) — and omits what cannot
be absent — `/work`, hence no assets (#117). One policy, two bugs, pointing
opposite ways. `--no-home` is therefore load-bearing and must not be dropped.

**The arms also disagree about `$HOME`:** enroot mounts it, singularity
suppresses it. Same code, two runtimes, opposite visibility of one directory —
the DIRECTION-FLIP hazard on the filesystem axis, mirroring the environment axis
(enroot forwards nothing by default; singularity forwards the host environment).

The compounding problem is that the set is **undeclared**, so nothing can check
it. There is no artifact stating which paths a run requires, so nothing can fail
closed when one is missing. The failure instead surfaces much later as a
file-not-found from the model loader — which reads as a bad path ARGUMENT rather
than a missing MOUNT. One cause, two misleading diagnoses.

## M17 — what this image can actually load (supersedes M15's Gemma row)

Five checkpoints, `AutoConfig.from_pretrained`, in-container, no trust_remote_code:

    gemma-3-12b-it      OK  gemma3  Gemma3ForConditionalGeneration   5 shards
    gemma-3-27b-it      OK  gemma3  Gemma3ForConditionalGeneration  12 shards
    gemma-4-31b-it      OK  gemma4  Gemma4ForConditionalGeneration   2 shards, 59G, index present
    Qwen3-4B-Instruct   OK  qwen3   Qwen3ForCausalLM                 3 shards
    qwen25_7B_Instruct  OK  qwen2   Qwen2ForCausalLM                 4 shards
    -- 5 loadable, 0 not, of 5 candidates

**Stock Gemma-4 loads.** M15's blocker was generalised from a single non-stock
checkpoint; see the retraction there.

This is also better news for the architecture review than the original Phase 4
plan. "One Gemma and one Qwen" is satisfiable today by FOUR distinct
architectures — gemma3, gemma4, qwen2, qwen3 — two of them
ConditionalGeneration (multimodal-capable) and two CausalLM. A model-agnostic
abstraction that holds across that spread is meaningfully tested; one that holds
across two checkpoints of adjacent families is not.

---

## CONFOUNDS — state these in the report; do not let them read as clean results

## M10 — #107, the original H100 blocker: leak reproduced, then fixed, both observed

Two-arm controlled experiment on the real image
(`<estate-root>/.../singularity_envs/nemo-automodel-26-04.sif`). Because this account
cannot leak naturally (CONFOUND-1), the leak was SYNTHESIZED against a
bind-mounted fake HOME — a python3.12 user-site `torch` stub announcing itself
as `0.0.0-HOST-LEAK-STUB`. The real home was never touched; the fake was removed
and its removal verified.

    ARM 1  leak planted, PYTHONNOUSERSITE unset      [MUST_FIRE]
           torch.__file__ = /home/<uid>/.local/lib/python3.12/site-packages/torch/__init__.py
           version        = 0.0.0-HOST-LEAK-STUB
           -> the container imported HOST torch. Leak reproduced.

    ARM 2  same leak still planted, PYTHONNOUSERSITE=1   [the fix]
           torch.__file__ = /usr/local/lib/python3.12/dist-packages/torch/__init__.py
           version        = 2.11.0a0+eb65b36914.nv26.02
           -> container torch. Leak suppressed with the leak still present.

This is the whole point of a positive control: the detector was observed firing
in ARM 1, so its silence in ARM 2 is evidence rather than absence of evidence.
Had only ARM 2 been run, "no leak" would have been unattributable — this account
cannot leak anyway.

Two further facts fall out, both previously assumed:
  * container python prefix = `/usr/local/lib/python3.12/dist-packages`,
    which is exactly the value `fs_assert_torch_provenance` hard-codes as its
    measured fallback. That constant is now measured on this image.
  * container torch = `2.11.0a0+eb65b36914.nv26.02`, confirming the version
    split that #107 originally reported against host 2.9.0+cu128.

**CONFOUND-1 — the torch-leak MUST_FIRE cannot fire naturally under this
account.** (Worked around in M10 by synthesis; still true, and still means any
future leak drill on <backup-account> must synthesize rather than observe.) Measured for <backup-account>: torch is present at `~/.local/lib/python3.10/
site-packages` but ABSENT at python3.12. The container is python3.12, so only
python3.12 user-site is consulted, and this account therefore *cannot* leak host
torch into the container. "No leak observed" under <backup-account> is consequently NOT
evidence that the leak is fixed — it is evidence that the account cannot produce
it. Any leak drill run here must SYNTHESIZE the condition (plant a python3.12
user-site torch) or it is measuring nothing. The original blocker (#107) was
observed under a different account and remains real.

**CONFOUND-2 — results are from the backup account.** Primary <primary-account> is
Permission denied. Anything account-scoped (user-site contents, quotas, group
membership, module availability) may differ on <primary-account> and is not transferable.

**CONFOUND-3 — no GPU work has run yet.** Everything above is static analysis,
shell execution, and login-node measurement. No claim about throughput, memory,
NCCL behaviour, loss, or checkpoint correctness is supported by any of it.
Phase 3 has not started.

---

## Two exit-status traps hit while doing this work

Both produced a green status over work that did not happen. Recording them
because the same shape will recur in CI.

1. `kimi_code.py ... | tail -6` — the codegen timed out after 3060s and wrote
   zero bytes, but the pipeline's exit status is `tail`'s, so it reported 0.
   A phantom artifact would have been recorded as delivered.
2. A backgrounded dispatch (`nohup ... &`) reported "completed, exit code 0"
   the instant the wrapper shell exited, while the real work ran on for minutes.

Rule both point at: never accept an exit status as evidence that an artifact
exists. Check for the artifact.

---

## M18 — #117 closed on both halves, with the second half nearly lost

The mount plane is now DECLARED (`FS_BIND_PATHS`) and materialised by both
runtime adapters from the same array: `--mount` under enroot, `--bind` under
singularity. Host-side validation refuses an absent source before launch;
in-container verification reports `k of N declared bind paths readable`.
`--no-home` is retained, so the #107 containment fix stays closed.

Gates: `apply_117.py` B1–B8 green, including B7, which locates each arm by its
own marker and requires BOTH to reference the shared array. B7 exists because a
patch that declares a shared plane and then materialises it under one runtime
only reproduces the defect while looking like the cure, and would pass every
generic gate.

**The worker refused to populate the array, and was right to.** Its `gaps`:
inventing a default mount plane inside `run_in_container` would recreate the
defect being fixed — a runtime deciding for itself what a run needs.

Population therefore went to the layer that knows the run. It is DERIVED from
the four paths the launcher already requires — `MODEL_DIR`, `DATASET_DIR`,
`dirname CONFIG_FILE`, `OUT_DIR` — plus an `FS_EXTRA_BIND_PATHS` escape hatch
for what cannot be inferred. **No estate path is hard-coded**, and the bind set
cannot drift from the paths the run actually references. Binds are identity
(`HOST:HOST`) so nothing needs rewriting at the boundary; remapping would force
every path variable to carry a host form and a container form, which is how the
two arms drifted apart to begin with.

    P5a MUST_FIRE  empty derivation EXECUTED -> REFUSED rc 96
    P5b MUST_PASS  4 inputs -> 3 paths (OUT_DIR==MODEL_DIR duplicate collapsed)

Both were run, not read. The `[[ ... ]] || fail` line proves only that it was
typed.

**A near-miss worth recording.** The population edit first went in by hand — to
a GENERATED file. `apply_113.py` rebuilds `launch_fs_h100.fixed.sh` from source
on every run and silently erased it; the regeneration was re-run deliberately
and confirmed `FS_BIND_PATHS` back to 0 references. The fix now lives in
`patch_bindpop.py`, and the pipeline is
`python3 apply_113.py && python3 patch_bindpop.py`. A fix present in the file
you are reading and absent from the file that runs is the worst available
outcome, and nothing in the gate suite would have caught it.

---

## M19 — a parser blind spot that lied in both directions at once

`_function_bodies()` recognised only `name() {` … column-0 `}`. Its docstring
defended the gap: anything else "simply does not enter the table — it is left
to A5". Both halves of that were wrong, from one cause:

  * **Over-reporting.** A5/B5 do not tolerate an unparsed definition, they
    mis-report it as *undefined*. `fs_die` — defined as a one-liner at
    `fs_container_backend.spliced.sh:76`, called 58 times — came back as the
    sole red on an otherwise clean B-gate run. A false blocker on a good fix.
  * **Under-reporting.** A8/B6 silently SKIP the contract of anything absent
    from the table. The three launcher helpers that dereference `$1` — `fail`,
    `require_cmd`, `req_env`, lines 30–32 — are *all* one-liners. Their
    contracts were never checked. A8's #113 green covered 15 of 19 definitions
    and reported no denominator for the shortfall.

One omission that shouts, one that whispers. Measured: backend 12 multi-line +
1 one-line, launcher 3 + 3; after the fix 13 and 6 parsed, **0 unparsed**.

No real defect was hiding there — `fail` is never called with a non-numeric
first argument (0 matches). So this was a detector repair, not a code repair,
and the widened gate is now load-bearing rather than decorative. Both suites
re-run green afterwards.

**Rule, extending #120:** a detector must be able to state what it failed to
parse. `_unparsed_definitions()` now supplies that denominator. "Left to another
gate" is not a coverage argument unless that gate is shown to cover it.

---

## M20 — #122: the resume proof could pass by not resuming

Measured before the fix, four independent facts:

| probe | result |
|---|---|
| `launch_fs_h100.fixed.sh:291-292` | exports **only** `SINGULARITYENV_FS_RESUME_CKPT` / `_STEP` |
| `grep -c RESUME fs_container_backend.bound.sh` | **0** |
| `FS_RESUME_CKPT` on `FS_ENV_ALLOWLIST` | **0** hits |
| `FS_RESUME_STEP` on `FS_ENV_ALLOWLIST` | **0** hits |

Controls for that last pair, so the zero is a measurement and not a dead grep:
`FS_ITERATION_BUDGET` 1, `FS_EARLY_SAVE_STEPS` 1, `OUT_DIR` 1.

`SINGULARITYENV_X` is singularity's private mechanism for injecting `X`. Under
enroot it is an ordinary host variable with an odd name — not allowlisted, so
not forwarded, so **`FS_RESUME_CKPT` does not exist in-container**. The trainer
finds no checkpoint, starts from step 0, trains its bounded budget and exits 0.
R5 records PASS. The gate is green *because* the work was skipped.

**Third instance of one pattern** (#109 runtime conflation → #117 mounts → #122
environment): a capability wired to one runtime's private mechanism works there
and vanishes silently on the other. So the fix is not "also export an enroot
equivalent". The forwarding path is already *shared* — one `forward_env` array
filtered through `fs_env_forward_allowlisted` (`:1035`), emitted as `--env` by
enroot (`:1112`) and as an `env K=V` prefix by singularity (`:1215`). Putting
the two names on the allowlist and exporting the **plain** names fixes both arms
from one place.

The `SINGULARITYENV_` forms were **deleted, not kept as belt-and-braces**: a
surviving runtime-specific duplicate keeps one arm working for a reason
unrelated to the allowlist, which is exactly how the defect stayed invisible.

`patch_resume_env.py`, 8/8. Q7 executes the patched allowlist rather than
reading it — MUST_PASS both names forwarded; MUST_FIRE `LD_PRELOAD`,
`FS_RESUME_CKPT_EXTRA` and `FS_RESUME` all refused, the middle row proving the
match is exact and not a prefix. Q8 gives the harness a denominator (23 names
lifted, bash agrees) so an empty extraction cannot satisfy the MUST_FIRE rows
for the wrong reason.

**Scope, stated so it is not over-read:** this proves the names are FORWARDED.
Whether `tools/fs_train.py` READS them and asserts restored == recorded is C5's
job and is UNMEASURED until that gate runs.

## M21 — #116: the drift gate, and what the reverse direction found

#115 (7 of 12 exports dropped) and #122 (two names never allowlisted) are the
same defect found twice by a person reading two files side by side. That is not
a control. `gate_env_drift.py` makes the class mechanical and checks **both**
directions, because they fail differently:

* **forward** — an exported name that does not cross ⇒ the container silently
  defaults;
* **reverse** — an allowlisted name that nothing produces ⇒ dead weight, or a
  consumer that will fail closed on a variable no one ever sets.

The reverse direction is the one nobody runs by hand. It immediately found
**`MASTER_PORT`: on the allowlist, consumed by `fs_launch_python`'s
`${MASTER_PORT:?...}`, produced by nothing in either file.** Latent only because
`fs_launch_python` has zero call sites (#124) — fixing #124 without this would
have detonated it.

Run against the real pair the gate was **RED on D1 and D3**, with all three
detectors independently observed going red on planted violations. Real reds
first, then the fix, then planted controls to keep it honest.

`patch_env_drift.py` (6/6) cleared both. F5 is the integration proof: it imports
the gate and requires GREEN before declaring success. F6 executes the minting
arithmetic — including a row that pins the literal port rather than asserting
mere inequality, because ids differing only above the modulus (12345 vs 13345)
*must* collide and an inequality-only row would hide that. That collision is
recorded as a **known limit**, not fixed: Slurm ids on one tray are
near-consecutive, which is the convention's stated scope.

One self-inflicted finding worth recording: D3's first breakdown printed
"12 launcher, 10 backend, 6 workload manager" against a denominator of 23,
because a name with two producers was counted twice. A parts-exceed-the-whole
breakdown is the same class of claim this gate exists to catch. Now disjoint by
precedence with an assertion that the parts sum to the whole.

## M22 — #123: one estate's filesystem compiled into the framework

Blocklist scan of the generated pair: backend **0**, launcher **16 matches over
4 lines** — `:53` `:54` `:140` validating against a literal `<estate-root>/*`,
and `:342` a `mounts=(...)` array that is **write-only** (zero readers; the
backend's `local -a mounts` at `:1069` is a different, arm-local variable).

Two standing requirements fail on those four lines at once. The path guards are
*right* — refusing a path unreachable from inside the container prevents a
failure that blames the model. What is wrong is that the **policy is a literal
instead of an input**: correct on one filesystem, silently wrong on the next.
And `/work/` plus the estate segment are both on the public-repo blocklist, so
all four would ship.

`FS_ALLOWED_PATH_ROOTS` follows the **existing** precedent rather than inventing
one: `FS_ALLOWED_NODE` and `FS_CONTAINER_RUNTIME` are already
required-with-no-default and fail closed at point of use, on the stated grounds
that an unconfigured guard is a disabled standing rule. It is a **list** because
estates keep assets and scratch on different filesystems, and a single-root
assumption is how a one-estate literal returns wearing a variable's name.

`patch_estate_roots.py`, 10/10. E8 executes the emitted matcher: MUST_FIRE on a
path outside every root **and** on `/rootfoo` against root `/root` — the
original glob got the path-boundary right and a naive `== "$root"*` rewrite
would lose it; MUST_PASS on the *second* of two roots, so a list silently
truncated to its first element cannot pass. E7 reports 16 → 0 with the pre-patch
count as the denominator.

**Breaking change, stated rather than glossed:** every launch must now export
`FS_ALLOWED_PATH_ROOTS`. The previous behaviour was not "no configuration
needed", it was "one estate's configuration compiled in".

## Build state

Six stages, reproducible from scratch, 41 gates, 0 red:

```
apply_113 → patch_bindpop → patch_estate_roots → apply_117 → patch_resume_env → patch_env_drift
   6P           5P                10P               6P             8P                6P
```

Standing gate `gate_env_drift.py` green (3 rules + 3 drills). Both artifacts
`bash -n` clean; blocklist 0/0 — the pair is publishable for the first time.

## M23 — #124: the eight-GPU run that was one process

Four measurements, and the conclusion is arithmetic rather than inference:

| probe | result |
|---|---|
| launcher sbatch header | `--gpus-per-node=8`, **`--ntasks-per-node=1`** |
| training launch | `run_in_container --workdir "$OUT_DIR" --` — **no** `--slurm-ntasks` |
| backend srun arm `:1275` | `srun ${ntasks:+--ntasks="$ntasks"} "${cmd[@]}"` → inherits **1** |
| `fs_launch_python` call sites | **0** (the single launcher mention is a comment) |

So the launcher proves `visible CUDA devices == 8`, hands an opaque
`FS_ENGINE_LAUNCH_CMD` to exactly one process, prints `gpus=8`, and exits 0.
The measurement was **checked and never load-bearing**. Phase 3's deliverable
would have recorded "8×H100, distributed training PASS" for a single-process
job, with every gate green — the vacuous-truth failure on the critical path,
in the artifact the whole exercise exists to produce.

`fs_launch_python` — the one function whose job is to couple gpus to command —
was dead, and its non-enroot branch returned bare `python3`, correct only under
a GB200 sbatch that supplied ranks via `--ntasks-per-node=4`.

**Fourth instance of the runtime-divergence pattern** (#109 → #117 → #122 →
here), and the clearest statement of it: rank multiplicity is decided by **who
forks the ranks**, which is orthogonal to **which container runtime is in use**.
Branching on `FS_BACKEND == enroot` conflated the two axes. The general fix has
the same shape it had all four times — declare the need in FoundationScale's
own vocabulary and let each arm materialise it:

`FS_ENGINE_LAUNCH_MODE`, required with no default, three values that exhaust
how ranks come into existence:

* `torchrun` — FoundationScale composes `--nproc_per_node` **from the measured
  count**, so the measurement becomes load-bearing rather than decorative;
* `wlm` — the workload manager forks them; refuse unless its per-node task count
  is numeric and equals the measured count, **and pass that count to srun** —
  asserting without passing leaves one process starting and the assertion true;
* `self` — the engine forks its own (deepspeed/accelerate/custom). FoundationScale
  cannot compose or count those, so it requires `FS_ENGINE_PROCS_PER_NODE` and
  stamps `world_size_source=engine-declared`, so **no report may call it
  measured**.

`patch_launch_topology.py`, 9/9. G7 executes the composer in a clean environment
across 11 rows (4 MUST_PASS, 7 MUST_FIRE). The load-bearing row is
`wlm gpus=8 ntasks_pn=1 → REFUSED`: that is the live configuration, and if it
does not go red the patch has fixed nothing. G9 is the gate that would have
caught #124 itself — it counts CALL SITES, requiring ≥1 for the composer and 0
remaining definitions of `fs_launch_python`, so an orphaned composer reads as red
instead of as clean code.

### The gate corrected the spec, not the implementation

The first draft appended both new names to `FS_ENV_ALLOWLIST`, because **I told
it to** — "launcher-produced policy the trainer will read". `gate_env_drift.py`
went red on D3: two allowlisted names with no producer. It was right. The
draft's own comment conceded the case against itself — *"Nothing in-container
consumes these yet"* — which is the dead-weight shape D3 exists to catch, the
same shape as `MASTER_PORT` before it was minted and the write-only `mounts`
array deleted in #123. Both names are host-side control-plane inputs read by
`fs_compose_launch` before any container starts, exactly like
`FS_CONTAINER_RUNTIME` / `FS_ALLOCATION` / `FS_BACKEND`. The allowlist is now
gated as **unchanged**, so a later helpful addition goes red rather than quietly
re-creating the violation.

**Rule:** a standing gate earns its keep when it refuses the author's own
instruction. This is the second time #116 has caught something no reader
proposed to look for.

### Parser postscript

Reconciling G6's "14 names" against D3's "25" mattered more than it looked: the
allowlist mixes bare entries with entries carrying trailing comments, and
`^\s+NAME\s*$` silently drops the commented 11 and then reports the remainder as
the whole list. `gate_env_drift.allowlist()` strips the comment first and is
correct; the earlier "23 of 23" verdict stands. Same blind spot as #121 and M19,
found again by two detectors disagreeing — **which is the only reason it was
found at all.** A denominator that is not the denominator is invisible until
something else counts the same thing.

## M24 — the build is one command

`build_h100_plane.sh` rebuilds the plane from scratch on every invocation:

```
apply_113 → patch_bindpop → patch_estate_roots → apply_117
          → patch_resume_env → patch_env_drift → patch_launch_topology
```

then runs `gate_env_drift.py`, the public-repo blocklist, and `bash -n`. The
blocklist ships its own MUST_FIRE (a planted `ghp_` string must match 1/1),
because a dead regex reports zero exactly as loudly as a clean file. Stages stop
at the first red rather than continuing, since each stage's anchors assume the
previous one applied.

Current: **7 stages green, drift gate green, 0 blocklist hits, 2/2 parse clean**
— launcher 450 lines, backend 1337 lines.

---

## M25 — #150: two artifacts, both green, disagreeing with each other

The checkpoint WRITER (`fs_train.fixed.py`) and the checkpoint ADJUDICATOR
(`fs_ckpt_adjudicator.py`) are emitted by two different build stages. Each passes
its own generated suite. They disagreed about the one thing they both touch.

Measured by importing the adjudicator and calling its parser directly:

```
checkpoint-step-00000010  -> None      <-- what the writer actually emits
checkpoint-step-00000200  -> None      <-- what the writer actually emits
step_10                   -> 10
checkpoint-10             -> 10
ckpt_10                   -> 10
```

`fullmatch` against `(?:step|checkpoint|ckpt)[_-](\d+)` read `checkpoint`, then
`-`, then a tail of `step-00000010` that is not all digits. So the parser
accepted three shapes the writer never produces and rejected both shapes it does.

Consequence: leg **A7b** — the only leg that cross-validates the directory name
against the manifest, i.e. the one that would catch a save landing in the wrong
directory or a resume reading the wrong checkpoint — **abstained on 2/2 of this
framework's own checkpoint formats.** Because ABSTAIN is correctly distinguished
from FAIL, the verdict still reached 0. **31 tests green, nothing surfaced.**

**The fix is the gate, not the regex.** Widening the pattern repairs today's
mismatch and leaves the next one undetected, and the next one is likely: nothing
holds two independently-generated stages in step. `gate_ckpt_naming_agreement.py`
now parses the writer's source (ast), extracts its format strings, renders them
at 5 step values including 0, and asserts the adjudicator's parser accepts each
one AND returns the right integer. Zero writer sites extracted is UNMEASURED
(exit 3), not agreement — that is the single most important line in it.

Only then was the parser repaired, in the EMITTER rather than the generated file
(the generated-vs-source asymmetry again; editing the artifact would survive
exactly until the next build).

| | before fix | after fix |
|---|---|---|
| gate exit | 5 DISAGREEMENT | 0 |
| rendered names parsed | 0/5 | 5/5 |
| parsed to the correct step | 0/5 | 5/5 |
| parser not vacuously permissive | 2/2 | 2/2 |
| gate's own controls behaving | 4/4 | 4/4 |

Wired into the build as a standing gate, and the wiring is itself certified:
reverting the parser to its broken form was observed taking the whole build to
`rc=5 NAMING GATE RED`, and restoring it to `rc=0`. A gate whose red has never
been observed through the build is a gate nobody knows is connected.

**A defect only a CROSS-artifact check could see.** Per-artifact suites are
structurally blind to it — both suites were green throughout.

---

## M26 — #151: the stage that removes the estate root hard-coded the estate root

`patch_estate_roots.py` exists to delete one estate's filesystem root from the
launcher (#123). It contained that root, hard-coded, **10 times**.

The build reported `0 blocklist hits` the entire time, and was not lying about
what it measured — it scanned **exactly the 7 generated artifacts** and nothing
else. The generators that produce them were never scanned. This is the same
question as #142 (launcher cannot find its siblings) and #146 (container cannot
find what the launcher points at), asked of a fourth edge: **the build scans its
outputs but never its inputs.** Generators are published too.

Fixed both halves:

* `FS_ESTATE_ROOT` is now a REQUIRED-WITH-NO-DEFAULT input to the stage. Measured:
  unset -> `rc 96`, relative path -> `rc 96`, correct absolute root -> `rc 0` with
  gate E0 observed RED first (10 literals at L11/L18/L32) and then green.
* The build now scans its **26 generators** in addition to its 7 artifacts, with a
  denominator floor: fewer than 20 inputs resolved exits **95 UNMEASURED**, because
  a shrunken denominator reads exactly like a clean scan.

**The discriminator took three attempts, and the two failures are the finding.**

1. *Exclude lines containing `BLOCKLIST`* — on the theory that a redaction list is
   vocabulary, not disclosure (#145). This immediately excused
   `BLOCKLIST = ("/work/<real-segment>/public/weights",)` in `extract_stage.py`:
   **a real estate path hidden inside a fake blocklist tuple** — precisely the case
   that fixture exists to prove is fatal. Excluding by keyword let the disclosure
   through by agreeing with it.
2. *Match the bare identifier* — went red on 10 hits, all of them redactors being
   flagged for containing the vocabulary they redact. #145 with no escape: the
   pattern MUST contain the token.
3. **Path adjacency**, which holds. The structural difference between the two uses
   is a slash: in an alternation the token sits between pipes; in a disclosure it
   sits against a separator. So a hit counts only when the identifier touches `/`.
   The K3 case stays fatal — a real path inside a fake blocklist still has its
   slashes — without pressuring generators to drop their redaction lists.

The adjacency rule has a real blind spot, **declared rather than discovered later**:
a bare identifier in generator PROSE is not path-adjacent. That declaration is
what motivated M27.

---

## M27 — #152: a promise in a comment, and what honouring it found

M26's blind-spot note said published DOCUMENTS were "scanned separately with the
full pattern and no adjacency rule". **No such scan existed.** A claim broader
than its evidence — doctrine point 6, turned on the build itself.

Writing it took ~20 lines and found disclosures in **6 documents** that neither
existing scan could reach, because neither reads `.md`: the node name in 3 places,
the partition name in 5, both account ids in 7, and a full model weights path.
All were redacted deterministically (substitution, not paraphrase, so the result
is provable by re-scan).

Then the same scan implicated the launcher, and this is the part worth keeping:

> **`<partition>` — the Slurm partition name — appears 13 times in the shipped launcher,
> and NO artifact scan could ever have seen it.**

Not through an oversight. The artifact pattern matches identifiers shaped like
hostnames (`dgpn0[0-9]`, `r0\d+dgx\d+`) or paths (`/work/…`). A bare lowercase
partition name has neither shape. It was reachable only from the third scan, over
prose, with no adjacency requirement — and once that scan existed it pointed back
at code.

**Filed as #152, and it is two defects wearing one coat:**

* *Publishability* — a cluster identifier in a public repository.
* *Generalizability*, which is the brief. Two of the 13 sites are functional
  (`#SBATCH --partition=<partition>`, `sinfo -h -p <partition>`); the other 11 are message text.
  A launcher that can only submit to one site's partition is not a foundation-model
  training framework, it is one site's script. Exact twin of #123, and the fix is
  deliberately the same shape — `FS_PARTITION`, required with no default — so the
  two read as one policy rather than two patches.

There is a trap in the functional half, recorded because it is easy to get wrong
in a way that looks right: `#SBATCH` is a **comment** to the shell, so
`#SBATCH --partition=$FS_PARTITION` does not expand — Slurm parses the literal
text. The directive cannot be parameterised in place; the flag has to move onto
the real submit command line and the dead directive has to go.

**Three scans, three patterns, and the differences are load-bearing:**

| scope | pattern | why it differs |
|---|---|---|
| 7 generated artifacts | full: identifiers + `/work/` + token-shaped `ghp_` | a generated file has no legitimate reason to contain either class |
| 26 build inputs | identifiers only, **path-adjacent** | generators must be able to name what they redact (#145) |
| 8 published documents | identifiers, **no adjacency**; `/work/` and `ghp_` stay token-class | prose is where bare identifiers live — but the docs must still be able to *describe* the blocklist |

The document scan's controls include the one that matters: a real estate path
smuggled into a sentence *about* redaction must still fire (1/1 observed). Without
it, "we allow prose about the blocklist" silently becomes "we allow anything on a
line that mentions the blocklist" — attempt 1's mistake, one layer up.

## M28 -- #142: the launcher sourced a file the build never produced

The shipped launcher contains a `source` line naming a sibling file. Every build
stage parsed clean; every per-file gate was green. The file named on that line is
one no build stage produces, so the shipped launcher reads a path that does not
exist.

Every gate that existed at the time checked artifacts one at a time, and the
artifact is fine one at a time. The defect lives on an edge between two files --
the launcher's belief that a sibling exists, and the build's belief about what it
emits. Per-file verification is structurally blind to edges, which is why nothing
already in the build could have caught this one.

**The fix is a standing linkage gate, run every build, not a one-off patch.**
Each declared cross-artifact reference is resolved against the set of files the
build actually emits; an edge whose target is not in the emit set is broken.
Measured this build: **6/6 linkage checks green.**

The gate ships three controls, all run on every build, none behind a flag:

* **C1 MUST_FIRE** -- a synthetic plane whose `source` names a sibling the build
  never produces: expected 5, observed 5 (1 edge, 1 broken). Without this, a gate
  that can never go red is decoration; a broken edge must be observed producing
  the refusal, not just presumed to.
* **C2 MUST_PASS** -- a fully linked synthetic plane: expected 0, observed 0
  (2 edges, 0 broken). This keeps the resolver honest in the other direction --
  a gate that fails everything it touches is indistinguishable from one that
  fails the right things.
* **C3 MUST_ABSTAIN** -- an edge-free synthetic plane: expected 95, observed 95
  (0 edges). This is the load-bearing control. `all([])` is True: a resolver
  that finds no edges to check has checked nothing, and a zero-edge plane
  reporting GREEN would be THE CLEAN CASE folded onto the unmeasured case. An
  edge-free plane must report UNMEASURED, not GREEN.

C3's absence would have allowed the most plausible failure mode of the gate it
guards: the reference-extraction pattern silently matching nothing, every build,
forever, reporting 100 percent linked on a denominator of zero.

## M29 -- #146: a required knob whose target could not be reached

`FS_CHECKPOINT_ADJUDICATORS` was already a required-no-default knob: the launcher
refused to run without it, which made the knob look done. It was not done. The
values it carried were neither containment-checked against the allowed roots nor
bound into the container. The knob was enforced with full ceremony; the thing it
named could not be reached from inside the container where enforcement happened.
The existing gates were blind to this because enforcement and reachability were
checked by nobody -- the refusal proved someone SET the knob, and nothing proved
the container could honour it.

The fix is in three parts, all in the generated launcher:

* **(a) Every adjudicator spec now gets a containment check** against
  `FS_ALLOWED_PATH_ROOTS` -- the same check four other executed paths already
  had. The report carries its denominator, printed as `containment %d of %d
  adjudicator spec(s) under a declared root`, because a sweep that short-circuits
  or an enumeration that returns empty would otherwise print the same "0
  violations" as a real pass.
* **(b) Each spec's dirname joins the container bind inference**, and an
  adjudicator sitting outside the allowed roots is REFUSED, not silently
  skipped. A skip converts a refusing knob into a silently weaker one; a refusal
  converts it into an operator-visible fact.
* **(c) Invocation captures the adjudicator's output and classifies it** rather
  than propagating its exit status blind, so a container-level failure to reach
  the adjudicator can no longer masquerade as the adjudicator's verdict.

Two ordering and reporting decisions are load-bearing. The parse was moved above
the bind inference, because a value parsed after the mounts are computed cannot
influence them -- the old order made even a correct refusal too late to matter.
And the refusal text names both the offending spec AND the declared roots: blame
aimed only at the value sends the operator to edit the wrong half of the
contract, and they will keep editing it, because either side may be the half
that is wrong.

## M30 -- #154: nineteen citations, none of them resolving

The operator runbook `LAUNCH.md` cited launcher behavior as `L:<n>` line
references. A resolver inserted near the top of the launcher had pushed every
later line down, and the document was not regenerated or re-checked, so all
**19/19 citations pointed at unrelated shell**. The runbook was not partly
stale; every single navigational sentence in it led the operator to the wrong
place. The existing gates were blind in the obvious way: the launcher was
fully gated, the document was scanned for estate literals, and no gate asked
whether the document still described the launcher.

The same sweep surfaced a second, independent break: `FS_PARTITION`, a
required-no-default knob that refuses at launcher line 28, is used **18 times
in the launcher and 0 times in the document**. An operator following the
runbook verbatim was refused on their first submit, from a prerequisite the
runbook never mentions.

**Fix: the document was rewritten against the current launcher, and a standing
gate now compares the two on every build** so the document cannot drift from the
launcher silently again. Measured this build (launcher 798 lines, doc 331 lines):

* **L1** -- 2/2 required-no-default knobs named in the document, enumerated FROM
  the launcher's 3 guard lines rather than from a hard-coded list in the gate;
  the 1 `SLURM_*` guard line is skipped as workload-manager supplied, not an
  operator input. Enumeration from the launcher matters: a hard-coded list
  drifts exactly the way `LAUNCH.md` did.
* **L2** -- 59/59 citations resolve: **38 on the first line of their own range**
  (the strict reading) and **21 elsewhere inside a declared range**.
* **L3** -- **0 estate-literal hits**; 3/3 required patterns loaded and scanned
  over all 331 document lines.

The gate ships three MUST_FIRE drills, each observed going red on a corrupted
copy of the document: a covered knob deleted (L1), a resolving citation
renumbered onto a line that does not mention its knob (L2), and the partition
literal planted (L3) -- plus a MUST_PASS that re-derives all three rules
identically afterwards, proving the drills did not damage the gate they
exercised.

The gate also **prints its drill population** -- `59/59 currently resolve, 38
corruptible` -- because a MUST_FIRE with nothing to corrupt is an unplantable
drill and proves nothing. A drill that passes on an empty denominator is a
clean scan of zero files, which is to say no scan at all.

## M31 -- #155: the redaction list that published the estate it protected

Every stage and gate in this build redacts estate identifiers, and each of them
redacts by naming what it removes. The alternation of identifiers was compiled
into the source in many places -- once per stage that needed it. Which means the
repository that exists to keep the estate out of the repository contained the
estate, once per copy, in the vocabulary of its own defenders. A blocklist scan
for estate literals was already running, and it read these as vocabulary, which
they are; what it could not decide was whether the vocabulary should have been
in the tree at all.

**The prior fix was right in shape and wrong in criterion, and the wrong
criterion is the finding.** #151b had already externalised some literals --
exactly the ones that could not be written as a safe regex. A bare five-digit
account id is inexpressible, because `[0-9]{5}` matches line counts and byte
sizes and half the corpus, so that went to the environment. Anything that COULD
be expressed as a pattern stayed compiled in. That criterion sounds technical
and is simply wrong: whether a token is expressible as a regex has nothing to do
with whether publishing it names somebody's estate. A corporate name is trivially
expressible -- it is a literal string -- and names the owner outright. The
question was never "can I write a pattern for this". It is "does this name
somebody's estate". Those are different tests, and the wrong one had already
shipped.

**Fix: one shared module, `fs_estate_pat.py`, and the vocabulary split into two
tiers that differ in kind, not degree:**

* The **IDENTIFIER tier** -- every string that names an estate -- is a build
  input: `FS_ESTATE_IDENT_PAT`, required with no default. Nothing in this tier
  lives in the tree anywhere.
* The **TOKEN tier** -- `/work/`, the `ghp_` prefix -- stays compiled in,
  deliberately, and this is #144's logic rather than a compromise: a secret
  PREFIX is not a secret. `ghp_` is documented by GitHub; the secret is the
  suffix the prefix announces, and every consumer of the scan needs the prefix
  shape to find instances.

**The empty case is declared, not assumed.** Setting the knob to the literal
`NONE` returns `""` rather than a bare `|`, because an empty alternation branch
matches every line -- the blocklist would flip from redacting an estate to
rejecting the universe, silently, on the exact configuration an operator picks
when they have no estate to redact. Conversely, unset is UNMEASURED, not clean,
and refuses `96`: a scan with no vocabulary has measured nothing.

**The denominator moved, and that is the real lesson.** The first scan covered
**19 candidate files and found 18 hits** -- a bad number for a redaction repo,
but a bounded one, and bounded problems invite targeted fixes. But the build
enumerates its generators FROM DISK, and the honest publish set is the whole
pipeline: **44 files**. Rescanning the true denominator found **24 further hits
in 13 more files**. Had the fix stopped at the narrow set, I would have shipped
those 24 hits while reporting a clean scan -- the scan genuinely was clean over
the denominator it saw, and the denominator was a decision I had made
implicitly instead of from the build. Every claim carries its denominator; a
clean number over the wrong denominator is a defect in the claim, not in the
scan.

**Measured, final:** 44 candidates, 0 hits. Five files refuse `96` with a named
message when the vocabulary is unset, and the build itself refuses. The declared-
empty case is verified not to degrade into a match-everything alternation: a
neutral line gives no match; `/work/` still matches. Build green, rc=0, 24
stages.

**A second hole, found while fixing the first:** the build's required-no-default
knobs had no home. They had only ever existed as inline shell arguments -- a
build advertised as reproducible was reproducible only from my scrollback. The
values were recovered by parsing the session's **5,514 recorded shell
commands**, and now live in one file outside the repository, mode 600. It is
deliberately not in the tree: a checked-in estate root is a published estate
root, which is this ticket in one file. What ships is a documented list of the
knobs and no values.

**Three mistakes worth recording, because each cost a cycle:**

1. **bash's `${VAR:?word}` cannot contain an apostrophe**, even when the whole
   expansion is double-quoted -- the parser opens a single-quoted region at the
   apostrophe and swallows text to the next `'`, so `bash -n` blamed a line 35
   lines away from the real fault. Confirmed with a two-case probe rather than
   guessed.
2. **Locating a module's import block with a regex matched prose inside a
   docstring** and injected a helper INTO the docstring. The prose happened to
   contain an import-shaped line, and the regex could not tell prose from code.
   The correct locator is the AST:
   `max(n.end_lineno for n in ast.parse(t).body if isinstance(n, (Import, ImportFrom)))`.
3. **The same class of anchor regex missed twice for the same reason** -- it was
   written from memory of the file's formatting instead of from the formatting.
   Continuation lines ending `",` and a flag that closes on the same line both
   defeated it.

None of the three turned into damage, for one structural reason: every edit in
the final pass ran through a wrapper that writes ONLY if its callback returns
without raising, so all seven resultant misses left their files untouched and
reportable. A missed edit is a fact; a half-applied one is damage. The mistakes
and the fix are held to the same standard here as anywhere else in the ledger:
a criterion broader or narrower than its evidence is a defect, whoever wrote it.

## M32 -- #156: the build was reproducible on one machine, and the proof of that was a clean copy

**The claim under test was "this plane builds from a clean clone."** Nothing had
ever tested it. Every green build had run in `fs-build`, which happens to sit
BESIDE the framework repo, in a directory that happens to be named `fs-repo`.
`apply_splice.py` resolved its upstream as `_ROOT.parent / "fs-repo"` -- one
guess encoding two separate assumptions, one about somebody's directory layout
and one about somebody's naming.

**Measured by staging the publish set into the repo and building there** -- the
arrangement a reader actually gets from `git clone`. The guess resolved to
`<repo>/fs-repo`, which does not exist, and the build went red at the first
stage. It had never been reproducible; it had been adjacent.

**The fix is a measured resolver, and the shape is now house style.** Candidates
are tried most-explicit-first (`--repo`, `$FS_UPSTREAM_REPO`, every ancestor of
`--root`, then the legacy named sibling), a candidate is accepted only if the
file the stage actually needs is present under it, and **every candidate tried is
named in the refusal**. A resolver that fails without saying where it looked
sends the reader to guess in turn. `apply_113.py` got the same treatment for the
launcher snapshot.

**The clean copy paid for itself three more times.** Building the staged tree is
a positive control on the publish set -- it can only go green if the set is
self-sufficient -- and it found, in order: `fs_estate_pat.py` missing
(`ModuleNotFoundError`), `h100_backend_splice.py` missing behind it, and the
suite gate declaring UNMEASURED because no interpreter with pytest resolved. The
third is not a missing file but an environment input, and the gate said so by
name, which is the difference between a refusal and a failure.

**One method note.** Discovering the missing modules one red build at a time is
O(depth) and stops the moment a build goes green for an unrelated reason. The
correct instrument is an **AST-based transitive import closure** over the publish
set. A `sitecustomize.py` `open`-audit trace was tried first and under-counts:
Python imports do not raise `open` audit events, so the trace reports a smaller
module set with complete confidence -- a detector that cannot see reports zero
and reads exactly like clean.

**Final:** in-repo build rc=0, 24 stages, suite 31 passed.

## M33 -- #157: the publish set and the scan denominator were different sets, and nothing compared them

**Seven estate identifiers were sitting in the published tree, in files that
three consecutive scans had reported clean.** Not one scan was buggy. Each was
clean over the denominator it was given, and no two denominators were the same
set:

| scan | denominator | derived from |
|---|---|---|
| artifacts | 7 generated files | a literal list in the build |
| generators | 31 files | "things that look like a stage" (`STAGES` + `gate_*.py` + 2 names) |
| documents | 9 files | `find h100 docs -maxdepth 2 -name '*.md'` |
| **what ships** | **52 files** | **nothing -- it was implicit in a `cp` loop** |

Three rules produced three sets; the fourth set was the only one that mattered
and it existed nowhere. `fs_estate_pat.py`, `h100_backend_splice.py`, five
`h100/*.json` generator envelopes and one generated intermediate were published
and scanned by nothing at all.

**Two independent mechanisms, one root.**

*Mechanism 1 -- an exemption that outlived its premise.* The generator scan
applied a PATH-ADJACENCY rule: an identifier counted only when it touched a `/`.
That rule was forced by a real constraint -- a generator that redacts these
tokens has to CONTAIN them, so matching the bare token flagged every redactor for
redacting. **#155 removed the constraint** by making the whole identifier tier an
environment input. Nothing re-derived the narrowing, and it survived exactly one
ticket past its reason. Measured over the same 31 generators: **adjacency 0 hits,
full identifier tier 3** -- two comments naming an identifier outright, and a
MUST_FIRE drill that assembled its org segment across quotes (`"HH""RI-AI"`) and
then wrote a node name literally in the same `printf`. *That is the general
lesson: a narrowing is only as sound as the constraint that forced it.*

*Mechanism 2 -- a third reason an estate literal enters source.* Not redaction
vocabulary (#155) and not a hard-coded default (#123), but **patch anchors**. A
stage locates its edit site by matching before-text exactly, and when that
before-text names the estate the anchor must name it too. Three stages did.
`h100/fix_113.json` -- the recorded generator response whose `fixes` are
before/after pairs cut from that same launcher -- carried the literal **11 more
times**.

**This is simultaneously a disclosure and a non-generality, which is why it
belongs in the framework and not in a redaction pass.** An anchor carrying one
estate's name can only ever match one estate's launcher, so all three stages were
silently single-site. The framework's *output* is estate-agnostic -- the shipped
launcher reads `FS_PARTITION` and the phrase is already stripped -- while the
anchors that find the estate's text were not. Supplying the name makes the same
stage work anywhere.

**A retraction inside the fix.** The first version introduced
`FS_ESTATE_SHORTNAME` to carry the name. `FS_PARTITION_LITERAL` already held
exactly that string, is already required-with-no-default, and
`patch_partition_knob.py` already enumerates all 13 of its sites -- two of which
are these anchors. A second knob for one fact is #153's defect, introduced by the
very ticket that exists to remove duplicated estate literals. Retracted on the
build host before it shipped; the accessor is `estate_partition_literal()` and
there is one oracle.

**The envelope was redacted losslessly, not edited.** Withholding it was not
available -- unlike `h100/upstream/`, it is REQUIRED for a build from a clean
clone, and a repository whose build cannot run is not a deliverable. Rewriting a
recorded response quietly is not provenance either. So the 11 literals became
`@FS_PARTITION_LITERAL@` and `apply_113.py` expands them at load, and **the
reversal is checkable by anyone holding the literal**:

    sed 's/@FS_PARTITION_LITERAL@/<literal>/g' h100/fix_113.json | shasum -a 256
      -> 4ec3cf083a0ad28b04de785f3ec32e0b529558e7836f81cad724db5461bb50d9

That hash is the original response, recorded before the substitution and proven
to round-trip at the moment it was made. A declared-empty (`NONE`) estate is
REFUSED here rather than expanded to nothing: `NONE` is a claim about the
operator's launcher, not about a response recorded from an estate whose text
demonstrably names a partition, and substituting `""` would leave 11 anchors
matching nothing and the stage reporting a clean no-op.

**The fix is construction, not detection.** A coverage gate was written first and
did find the gap -- 52 published, 43 covered, 9 unowned, with its own MUST_FIRE.
But a detector for a class of defect that construction can make impossible is a
detector somebody has to keep answering. So `h100/PUBLISH_SET.txt` now declares
what ships and **the scan denominator is derived from it**: everything published
that is not a generated artifact and not a document is a generator. The stage
glob is unioned in rather than replaced, because it covers files scanned without
being published, which is the safe direction to over-scan in. The gate is demoted
from discovery to a MUST_FIRE control on that derivation.

**Measured, final:** generators 31 -> **40 files scanned**, coverage **52
published / 52 covered / 0 unowned**, identifier tier **0/52**. Token tier 43,
all of it vocabulary -- bare `/work/` and `ghp_`, `printf` format strings,
`<org>` placeholders and a fabricated `acme-org` -- with no real segment
anywhere. Both trees rc=0, 24 stages, suite 31 passed.

**One more zsh trap, third occurrence.** The first version of the verification
loop above reported `identifier-tier 0 | token-tier 0` and was **vacuous**: in zsh
an unquoted `$var` is NOT word-split, so `for f in $files` runs one iteration on a
53-line blob as a filename, every `grep` fails, and `|| true` turns the failure
into a clean zero. The file-count line printed `52` by coincidence -- the blob's
own newlines. Rewritten with `while IFS= read -r`, the same probe reports 43
token-tier hits. A harness with no positive control is not a measurement, and
that applies to the harness I write to check the gate exactly as much as to the
gate.

## M34 -- #158: a producer that exited nonzero truncated the union that measured coverage

**The build measured a 55-file publish set but only nine documents, and the
coverage gate correctly refused to call that a pass.** Three files had just
been added to `h100/PUBLISH_SET.txt` -- `README.md`, `.gitignore` and
`estate.env.example` -- taking the set from 52 to 55 entries. The next build
returned RC=95, not because `README.md` was missing and not because it
contained a hit, but because it was published and had no measuring scan:

    [documents]                        9 file(s) scanned, 0 hit(s)
    [publish-set coverage]             55 published, 54 covered, 1 unowned
    COVERAGE UNMEASURED: README.md

The file existed and was 6028 bytes. It was line 18 of the publish set. The
publish-set branch of the document producer emitted it correctly when that
branch was run on its own. So the visible contradiction was not between a
missing file and a missing scan entry: the file, the declaration and one arm
of the producer were each correct, and the union of correct parts was missing
exactly the declared file.

The union had been introduced by #157 to derive denominators instead of
merely detecting their gaps. For documents it combined the filesystem glob
with the `.md` entries declared for publication:

    DOCS=()
    while IFS= read -r _d; do [[ -f "$_d" ]] && DOCS+=("$_d"); done < <(
      { find h100 docs -maxdepth 2 -name '*.md' 2>/dev/null
        if [[ -r h100/PUBLISH_SET.txt ]]; then
          grep -vE '^[[:space:]]*(#|$)' h100/PUBLISH_SET.txt | sed 's|^\./||' | grep -E '\.md$'
        fi; } | sort -u)

**The failing producer was `find`, and its failure did not look like a
failure from the caller.** The root `docs/` does not exist in this tree, so
`find h100 docs ...` exits 1 after reporting what it can. `set -euo pipefail`
is active for the build, and that behaviour is inherited by the
process-substitution subshell. The failing `find` therefore killed the whole
brace group before the publish-set branch could run. The consumer did not see
an error; it received the nine entries already emitted by the `find` arm and
treated that truncated stream as the complete denominator.

The `2>/dev/null` completed the failure mode. It did not cause `find` to miss
`README.md`, but it suppressed the one diagnostic line that would have said
why the producer stopped. From outside, the arithmetic consequently looked
ordinary: nine documents were scanned, zero hit, and only the independent
coverage control knew that a published tenth document had gone unmeasured.

**The failure was isolated by changing only the shell discipline around the
same pipeline.** With `set -euo pipefail`, the producer yielded nine entries
and omitted `README.md`. Without `set -e`, the same pipeline yielded ten and
included it:

    bash -c 'set -euo pipefail; ...'  -> n=9   (README.md absent)
    bash -c '...'                     -> n=10  (README.md present)

That is the whole defect, measured before it was named: one producer's exit
status terminated unrelated producers sharing its brace group. The entry that
#157 had added specifically so that no published document could escape the
scan was removed by a filesystem producer that did not know the publish set
existed.

The generator union carried the same latent shape. Its shell side used
`ls gate_*.py 2>/dev/null`; on an estate with no gate files, that `ls` would
exit nonzero and truncate the union in exactly the same way. It did not fire
here only because this tree has gate files. Correct output on this host was
therefore not evidence that the construction was safe.

**The fix removes the producer failure instead of teaching the build to
ignore it.** `find` is now given only roots that `[[ -d ]]` has already
confirmed to exist, so a nonexistent root cannot terminate the brace group.
The generator glob is expanded by the shell under `nullglob` rather than by
`ls`, turning “no matches” from a command error into an empty list. The two
producers now have absence as data, not as a nonzero exit.

`|| true` was considered and rejected for the same reason it would have
looked attractive: it fixes this instance by making all failures look alike.
It would have allowed the missing `docs/` root, and it would equally have
allowed a genuine permission error on a real root, again producing a partial
denominator under a clean-looking zero. Removing a known impossible state is
different from suppressing every possible error, and only the former leaves
the build able to fail closed on the next one.

**The derivation also gained a control on the limb, not only on the final
union.** If `h100/PUBLISH_SET.txt` is readable but the branch reading it
contributes zero entries, the build exits 95 and names that branch as dead.
The outer coverage gate could not have supplied that control by itself: it
catches a truncation whose victim is published, but a dead limb that happens
not to drop anything currently published would leave every denominator
balanced and every scan clean. The control has to be placed on the producer
because that is where the loss occurs.

After the change, the document scan measured ten files instead of nine and
coverage measured **55 published / 55 covered / 0 unowned**. The important
part of that result is not the extra file alone; it is that all three claims
now refer to the same set. The publish declaration says what ships, the
producer derives the document denominator from it, and the outer control
checks that derivation instead of being asked to discover its underlying
shape.

**The defect was introduced by the same edit that added the control which
caught it.** #157 replaced a coverage detector with a derivation because a
constructed denominator removes the whole class “published but unscanned.”
That remains true, and this ticket does not reverse it. But deriving the
denominator does not make the derivation itself immune to error; it moves the
place where an estate-specific failure can enter from the scan list into the
producer. A MUST_FIRE control on a construction believed airtight is not
redundant insurance. It is the only measured answer to the possibility that
the construction's author wrote a defect while removing one.

This is now the second time in this build that a coverage control has earned
its keep against its own author. It is also the fourth recorded
shell-semantics trap: `set -e` crossed into process substitution, and the
suppressed stderr turned the resulting truncation into a plausible number.
The number was not noisy and the build was not randomly red. It was precise,
clean and one entry short -- exactly the failure a positive denominator
control exists to make visible.

## M35 -- #159: the pattern that named the partition could not match its own source text

**A fixed-string sweep of all 55 published files found the partition literal
in exactly one file, `build_h100_plane.sh`, while the generator scan was
reporting 0 hits on that same file.** Line 351 of the build script assembled
the document-scan pattern in this form:

    DOC_PAT="$IDENT_CORE|/work/[A-Za-z0-9]|ghp_[A-Za-z0-9]{8}|\b<PARTITION>\b"

Where the placeholder appears here, the published source carried the estate's
actual four-character partition name. That made the build script a fourteenth
site of the literal that #152 had removed from the launcher's thirteen sites
and #157 had removed from the patch anchors. The value was not operating
vocabulary and was not needed at compile time; it had been written directly
into a file in the publish set.

The generator denominator includes `build_h100_plane.sh`, and the publish set
includes it as well. The generator scan reads every such file with the
identifier pattern, so the literal was inside a measured file and the
pattern's own document-scan branch was visible to the sweep. Yet the reported
measurement over that file was zero. That was not a false statement about an
unmatched source. It was a precise statement from a pattern that could not
reach the text carrying its value.

**The cause was the word boundary that made the expression precise in
running prose.** `FS_REDACT_EXTRA` carries this token in the form
`\b<PARTITION>\b`. In prose, a boundary before the value is the correct
narrowing: it prevents an accidental match inside a longer word. In the
source line, however, the character immediately before the value was the `b`
of the preceding `\b` escape sequence. `b` is itself a word character, so
there was no word boundary at that position and the expression did not match
the very line that wrote the token down.

The narrowing was measured on both sides before the conclusion was drawn:

    printf '<PARTITION>\n' | grep -cEi "$IDENT_CORE"   -> 1
    printf 'b<PARTITION>b\n' | grep -cEi "$IDENT_CORE" -> 0

A clean boundary around an isolated value matches; a word character before
and after the same value makes the boundary expression blind. The source line
had the second shape. A detector tuned so that prose could mention a token
without incorrectly flagging the token's surrounding syntax therefore passed
exactly at the place where the token had become the secret.

**The fix removes the value rather than adding a cleverer regex for finding
it.** `DOC_PAT` is now assembled from `$FS_PARTITION_LITERAL` at runtime.
When that input is the declared-empty `NONE`, it contributes no partition
branch at all. The value does not appear anywhere in the tree, so publication
no longer depends on a matcher being broad enough to recognise its own
definition.

A fourth scan was added anyway, and it is deliberately not a regular
expression. It performs a `grep -F` search for the literal value across the
publish set itself -- not across the document category, generator category or
any other intermediate classification, but across the shipment. The
denominator floor is 20 files, so an empty or accidentally tiny search set
cannot report a clean zero.

That fixed-string scan carries a MUST_FIRE control that plants the value in a
temporary file and asserts that the sweep finds it. This control is not
ornamental. A fixed-string search whose needle is absent and a fixed-string
search whose needle was never made both print the same convincing result:
zero. The control separates “the shipment does not contain the value” from
“the probe did not contain the value it claimed to seek.”

The needle is written to a temporary file rather than assembled inline in the
build script. An inline representation would place the literal back into this
file, after which the sweep would be correct to fail on the file defining the
sweep. The clean result must therefore be produced by a needle that exists
for the measurement without becoming part of the measured text.

**The control fired on the commit that introduced it.** The first draft of
the explanatory comment above `DOC_PAT` quoted the offending source text
verbatim, twice. It did so to explain why publishing the value was wrong, and
in doing so published it twice more. The pattern line itself had already been
fixed by then, so the sweep's two hits in `build_h100_plane.sh` were both in
the explanation: the fix was in, and the file still shipped the value. The
comment now uses a placeholder.

That self-hit was the control working before it had a chance to become
background noise. A detector that checks only code while leaving prose to
judgement would still have missed the disclosure, because the disclosure was
the explanation. The scan therefore checks the publish set for the value
itself, and the prose carries the reason without carrying the secret it is
about.

There are two lessons here, and the second is the one that generalises.

The first is #157's shape again: **a narrowing is only as sound as the
constraint that forced it.** The word boundary was right when the pattern was
matching prose and wrong when the same pattern was pointed at source text
containing an escaped representation of itself. Nothing re-derived the
boundary when the pattern crossed that boundary into source. Precision in one
context became blindness in another, while the clean zero preserved the
appearance of generality.

The second is that **the use-versus-mention licence has a limit.** #144
established that a redactor may name the vocabulary it redacts: `ghp_` is a
public prefix, so publishing that prefix discloses the mechanism without
disclosing a secret. That licence was granted to a token class, not to every
token merely because it appears in a pattern. A partition name is itself the
secret; mentioning it in a comment discloses it exactly as much as mentioning
it in code.

The habit being corrected was therefore not merely “a regex failed.” It was a
rule transferred past the reason that made it safe. Source may need to
construct a pattern for a supplied value, but it does not need to write the
value down, and a comment explaining the rule has no stronger need than the
code does. The runtime input carries the fact; the shipment carries the
mechanism; and the fixed-string sweep now measures the difference.
## M36 -- #160: the gates computed the right verdict and discarded it at the exit

**An audit of all nine UNMEASURED sites in the build found two that computed the correct verdict and then collapsed it into RED at the last step.** The contract itself was never ambiguous. The build publishes four codes used by every stage and gate: 0 PASS, 5 RED (a real defect), 95 UNMEASURED (the check could not run at all, and is explicitly not a pass), and 96 REFUSE (a required input is unset, with no default by design). The distinction between 5 and 95 is the reason 95 exists. Two call sites erased it.

The first was the generated-unit-suite gate. When no interpreter with pytest resolves -- neither `$FS_PYTEST` nor `./.venv/bin/python` -- the gate cannot measure anything. It printed the right word and then reported the wrong state:

    print UNMEASURED
    exit 5

The second was the checkpoint-naming-agreement gate. That gate is more careful than most: it returns four distinct codes of its own, where 3 is UNMEASURED (zero writer sites, or the adjudicator would not import), 4 is CONTROLS FAILED, and 5 is RED (writer and adjudicator disagree). The calling `case` arm printed the right words for all four codes and then fell through to a single `exit 5` for every one of them. In both cases the vocabulary survived the gate and died at the translation into an exit status.

**The defects were found by running the build in the published tree rather than on the build host.** The published tree has no `.venv`, because a virtualenv is not a publishable artifact, so the suite gate took its no-pytest branch there for the first time. Nothing on the build host would ever have surfaced either defect: 5 and 95 are both nonzero, both fail the build, and a red build with working inputs was not a build anyone ran. The contract was being honoured everywhere it was observed and violated everywhere it was not.

The collapse also made a shipped document false. `README.md`, written in the same batch, states that the pytest-suite gate "declares 95 rather than skipping if $FS_PYTEST and ./.venv/bin/python both fail to resolve". At the moment that sentence shipped, the code exited 5. The document described the contract; the code broke it; and the two were published together, each apparently certifying the other.

**The fix restores the codes and records the verdict in the printed line.** The suite gate now runs `exit 95`, and the line it prints names the code: "UNMEASURED (95)". In the `case` arm, rc=3 now maps to `exit 95`. rc=4 deliberately stays RED: a gate that fails its own controls is not unmeasured, it is untrustworthy, and the build must not offer it the softer word. rc=5 stays RED, because a disagreement between writer and adjudicator is exactly what RED means. The mapping preserves the gate's vocabulary instead of assigning one comfortable code to everything nonzero.

The defect was then drilled to order rather than assumed repaired:

    FS_PYTEST=/nonexistent/python        -> exit 95; exactly one "UNMEASURED (95)" line
    inputs supplied, build-host tree     -> exit 0
    inputs supplied, published tree      -> exit 0

A missing interpreter now produces the softer code and a single line carrying it, with no RED printed anywhere. With genuine inputs, both trees pass.

**The remaining seven sites were already correct, and the audit names them rather than assuming them.** The two denominator-branch controls, the generator scan floor, the document scan floor, the partition sweep floor, the publish-set size floor, the unowned-file refusal and the absent-manifest refusal all exit 95 when they cannot measure. Denominator: 9 sites checked, 2 defective, 7 correct. A clean statement about the sites that were wrong, drawn from a count of all the sites there were.

The whole reason 95 exists as a separate code is that a consumer must be able to tell "a gate found a defect" from "a gate could not run". Every gate in this build was careful to compute that distinction, and then two call sites threw it away at the last step -- which is where a four-state contract usually dies: not in the detector, in the translation. And the reason it survived is that both codes are nonzero: the defect was only observable from an environment nobody routinely built in. That is the argument for building in the published tree as a standing step, rather than treating the build host as representative.
## M37 -- #161: the trap was documented beside the code and violated nine times

The plane publishes a four-state exit contract, and every stage and gate is required to honor it. Exit code 0 means PASS. Exit code 5 means RED, a real measured defect. Exit code 95 means UNMEASURED: the check could not run, which is explicitly not a pass. Exit code 96 means REFUSE: a required input was unset, with no default by design. Consumers of these stages — gates, launchers, aggregate summaries — branch on which of the four states a stage reported. The contract only carries information if the distinction between the states survives the exit.

In Python it did not. `raise SystemExit("some message")` prints the message to stderr and exits with status **1**. `sys.exit("message")` does the same. So a stage that intended REFUSE and was written as:

```python
raise SystemExit("...required input missing...")
```

exited 1, not 96. And a consumer receiving 1 could not tell REFUSE from RED from an unhandled crash — all three funnel into the same undifferentiated nonzero. An audit of the build's stages and gates found **nine sites** of this shape, in six files.

### The trap was already written down

The sharpest part of the finding is that this was not an unknown failure mode. Two files in the same tree already carried a docstring warning describing exactly it, written when three earlier sites were fixed. The text read:

```text
`raise SystemExit("text")` prints the text but exits 1, silently breaking the
contract. Print to stderr, then raise SystemExit(<number>).
```

That warning lived in the module docstring of `patch_partition_knob.py` at line 91 and as an inline comment at `patch_estate_roots.py:123`. Both of those files were correct. Their siblings were not. Nine further violations accumulated in the same tree, beside prose that described them precisely. Nothing re-derived the rule; the knowledge existed only as commentary, read by whoever already suspected the problem and invisible to everyone else. This was the fourth recurrence in this build of the same overall shape: a narrowing or a lesson that exists in prose and is enforced by nothing mechanical.

### The audit's own count was wrong, in the way this build keeps getting things wrong

The first audit did not report nine. It reported **12**, from a text grep for
`raise SystemExit(` followed by a string. Three of those twelve were not
violations at all. They were the warning:

```text
patch_partition_knob.py:91   `raise SystemExit("text")` prints the text but exits 1, ...
patch_partition_knob.py:132  # `raise SystemExit("text")` prints the text but exits 1, ...
patch_estate_roots.py:123    # `raise SystemExit("text")` prints the text but exits 1, ...
```

9 real sites + 3 prose mentions = the 12 that was nearly published. The count was
recovered only because the `ast` gate re-derived it independently and disagreed:
5 rejections on the pre-fix published tree, plus 4 hand-fixed sites in the three
unpublished files, is 9.

This is the third time in four tickets that a text scanner has confused a mention
of a token with an instance of it. #144 was a redactor's pattern read as the
secret it redacts. #159 was the reverse -- a pattern that named the partition and
then could not see the partition written inside its own escape. Here the miscount
ran the other way again and would have overstated the defect by a third. The
common cause is that a grep sees characters and a claim is about meaning, and the
only reliable repair is a second detector of a different class: `ast` here, a
fixed-string sweep in #159. A number restated in a report is a claim like any
other, and it needed its own denominator.

### Each site was classified, not blanket-fixed

The nine sites did not all deserve the same exit code, and giving them one would have repeated the original error in a different costume. Distinguishing 95 from 5 from 96 is the entire reason the contract has four states; "it exits nonzero, pick one" is the reasoning that caused the bug. Each site was examined for what its situation actually was:

| Site | Situation | Code |
|---|---|---|
| `apply_113.py:94` | upstream launcher snapshot absent | 96 — the build input is deliberately not published; there is nothing to patch |
| `apply_113.py:171` | envelope is a template and `FS_PARTITION_LITERAL` unset or declared NONE | 96 — note that this site was authored in the previous ticket (#157), the very work that fixed three other instances |
| `apply_splice.py:112` | upstream backend not found | 96 |
| `pre71e_mutate.py:1518` | cannot write the attribution artifact | 95 — the run happened; its result is unknowable, which is UNMEASURED, not RED |
| `merge_splice_parts.py:29` | a fan-out part errored or came back empty | 95 — the spec is absent, not wrong |
| `merge_splice_parts.py:39` | two parts replace the same function | 5 — a measured contradiction |
| `patch_resume_env.py:153` | cannot locate the block a control is built from | 5 |
| `patch_resume_env.py:161` | unterminated block | 5 |
| `h100/gate133_estate.py:25` | generated module absent | 96 |

`merge_splice_parts.py` did not import `sys`; the import was added as part of its fix.

Three of these assignments deserve a sentence each. `pre71e_mutate.py:1518` is the case most likely to be misclassified as RED: the mutation ran and produced a result, and what is lost is the ability to know that result, which is precisely what UNMEASURED exists to express. `merge_splice_parts.py:39` is the genuine RED of the set — two parts asserting replacement of the same function is a measured contradiction, observable and attributable. And `apply_113.py:171` is the uncomfortable one: the engineer who fixed three instances of this trap wrote a fourth while doing so, in the same ticket. The docstring did not protect its own author.

### The control that makes the class impossible to reintroduce

Fixes without enforcement had already been tried — twice, by the docstring's own history. The remedy this time was a new gate, `gate_exit_contract.py`, which runs on every build. It parses each published Python file with `ast` and judges every `raise SystemExit(...)`, `sys.exit(...)`, and bare `exit(...)` call site according to a static rule:

- a `str` constant, an f-string (`ast.JoinedStr`), a `str.format` call, or a `+`/`%` chain whose leftmost operand is any of those: REJECT
- an int literal in {0, 5, 95, 96}, or its unary-minus form: ACCEPT
- an int literal outside that set, or a bool: REJECT, with the rejection naming the contract
- a Name, a Call, or anything not statically decidable: UNJUDGED — printed and counted, never silently folded into "clean"
- a file that does not parse: REJECT (fail closed)

The denominator is fixed by the `.py` entries of `h100/PUBLISH_SET.txt`, with **no glob fallback**. If the publish set is unreadable, or if fewer than 8 Python files resolve from it, the gate exits 95 rather than reporting a clean scan of a self-selected subset. Its own exit codes are 0 clean, 5 violation, 4 controls failed, 95 unmeasured.

The gate's last line is deliberately `raise SystemExit(main())`, which its own rule classifies as UNJUDGED. That is the correct bucket: the exit code of the gate is decided by `main()`'s returns, not by any literal at the call site, and pretending otherwise would be exactly the kind of silent folding the gate exists to prevent.

### The gate's own first draft was 60% blind, and the real tree proved it

The first draft handled `ast.Constant` strings and lone f-strings, but its leftmost-operand walk tested only `isinstance(node, ast.Constant)` and never tested `ast.JoinedStr`. Run against the pre-fix staged tree, it reported:

```text
scanned 37 files, 44 exit sites, 2 rejected, 36 unjudged   RC=5
```

Two of five published violations caught. The other three — `apply_113.py:94`, `apply_113.py:168`, `apply_splice.py:112` — were written as an f-string, or as an implicitly concatenated `"str" f"{x}"` that Python folds into a single `ast.JoinedStr`, followed by `+ "".join(...)`. All three landed in UNJUDGED. The gate was red, but for the wrong two reasons, and silent on the majority of the defect.

It fired at all only because the staged publish tree still held the pre-fix text, which made a genuine before/after MUST_FIRE available on real source rather than on a synthetic fixture. Had the fixes been applied to both trees before the gate was validated, the gate would have gone straight to `0 rejected`, looked healthy, and been believed — while three of five live violations sat in its blind spot.

The fix was that `_leftmost_is_string` now returns True for `ast.JoinedStr`. Two controls were added covering the two shapes the tree actually contained — an f-string followed by `+`, and an implicit `str`+f-string followed by `+` — alongside the pre-existing must-fire (`raise SystemExit("...")`), must-fire 2 (`sys.exit(1)`), and must-pass (`SystemExit(96)` / `sys.exit(0)` / `SystemExit(rc)`, which must yield zero rejections and exactly one unjudged). Measured after that fix:

```text
pre-fix staged tree : scanned 37 files, 44 exit sites, 5 rejected, 33 unjudged   RC=5
fixed build tree    : scanned 38 files, 45 exit sites, 0 rejected, 34 unjudged   RC=0
```

The denominator is 38 rather than 37 because the gate is itself published and scans itself.

### Scope, stated rather than implied

Because the denominator is the publish set, three of the nine sites — `merge_splice_parts.py`, `pre71e_mutate.py`, and `h100/gate133_estate.py` — fall outside it and were fixed by hand without a standing control over them. This is a declared limit, not an oversight: the gate enforces the contract that the shipment publishes. An unpublished build-internal file violating the contract is still a defect, but it is not one that any published claim depends on, and conflating the two would overstate what the gate guarantees.

### Build result

The whole build, after the fixes and with the new gate in place, ran to **RC=0** across 24 stages. The publish set held 56 entries, with publish-set coverage of 56 published / 56 covered / 0 unowned, the partition-literal sweep reporting 56 files scanned and 0 hits, and the exit-contract gate reporting 0 rejected.

### The lesson to land

Two sentences of prose, sitting beside the code, did not prevent nine further violations of the rule those sentences described — and one of the nine was written by the person who had just fixed three others while reading that very warning. Documentation is not a control. A control is something that runs.

The second lesson is narrower and sharper. The gate that enforces a rule needs its control set drawn from the shapes the tree actually contains, not from the textbook statement of the rule. The textbook shape was `SystemExit("text")`. The tree's dominant shape was `SystemExit(f"..." + "...")`. A gate built only from the textbook would have shipped green while three-fifths of the defect remained, and it would have been trusted precisely because it was red on a minority that happened to match the textbook. A control that fires on the easy cases and folds the hard ones into an unexamined bucket is worse than no control, because it manufactures the confidence the class of bug feeds on.

## M38 -- #162: the scans were tuned for the estate and blind to the build host

`h100_backend_splice.py`, a published build stage, hard-coded two absolute paths under one developer's home directory:

```
39: SRC = pathlib.Path("/Users/<dev>/fs-repo/launchers/fs_container_backend.sh")
40: OUT = pathlib.Path("/Users/<dev>/fs-build/h100")
```

This is a public repository, and the paths are a small disclosure on their own: a username and a local layout. The worse half was structural. A published stage with both of its paths hard-coded only runs on one machine. A checkout on any other host resolves neither path, and the stage does not fail in any way recognisable as a portability defect -- it simply cannot find its input.

### Why every in-build scan said clean

At the time, the build ran four content scans, and all four were green:

```
generators        43 file(s) scanned, 0 hit(s)
documents         10 file(s) scanned, 0 hit(s)
partition literal 56 file(s) swept,   0 hit(s)
publish coverage  56 published, 56 covered, 0 unowned
```

None of them could have caught this. Their vocabulary is the estate's: node names, account ids, organisational hostnames, management IPs (`FS_ESTATE_IDENT_PAT`), the Slurm partition literal (`FS_PARTITION_LITERAL`), the estate filesystem root (`FS_ESTATE_ROOT`), plus token shapes such as `ghp_`. Every one of those patterns describes the machine the build *targets*. Not one describes the machine the build *runs on*. `/Users/<dev>/...` was invisible to all four vocabulary sets, and it passed cleanly through each.

The defect was caught instead by `prepush_gate.py`, an unpublished local gate with a wider vocabulary, which returned RC=1 with the two hits named.

### One defect, three appearances

This was the third recurrence of a single defect class in this build:

- **#123** -- the estate launcher hard-coded an organisational filesystem root four times.
- **#151** -- the stage written to de-hard-code that root hard-coded it itself.
- **#162** -- one layer further down: not the estate's root this time, the developer's laptop.

The lesson was the same each time. A path that happens to be correct on the machine where the code was written is not a resolved path; it is an unstated assumption that has not yet been contradicted.

### The fix

Both constants were replaced with a lazy resolver. The resolution order is deliberately identical to `apply_splice.py`'s -- the same environment variable, the same ancestor walk, the same legacy-sibling fallback -- because "where is the upstream repo" must have exactly one answer in this build. Two resolvers that disagree would constitute a second oracle.

```
_HERE = pathlib.Path(__file__).resolve().parent
_BACKEND_REL = pathlib.Path("launchers/fs_container_backend.sh")

def _resolve_src() -> pathlib.Path:
    # $FS_UPSTREAM_REPO, then every ancestor of this file, then the legacy
    # sibling; each candidate accepted only if the file is really there;
    # every candidate named in the refusal.
    ...
    raise SystemExit(96)

OUT = _HERE / "h100"
```

The resolver is a function rather than module-level for a specific reason. `h100_backend_splice.functions()` is imported by three other stages, and a missing upstream repo must not make importing a brace matcher fail. Module-level resolution would have converted a stage-scoped refusal into an import-time crash in unrelated stages. The refusal is 96 (REFUSE), reached through `print(..., file=sys.stderr)` followed by `raise SystemExit(96)` -- the shape #161 had just standardised.

### The first control was vacuous

The obvious control -- point `FS_UPSTREAM_REPO` at a nonexistent path and check that the stage refuses -- returned RC=0. It passed, because the stage found the repo anyway: the ancestor walk and the legacy-sibling candidate both still resolved to the real checkout. Setting one candidate to nonsense proves nothing about a resolver with ten candidates.

A genuine control required an environment in which no candidate can resolve, so it was re-run from an isolated `mktemp -d` with the module copied in:

```
MUST_REFUSE (isolated tmpdir, no repo on any candidate path):
    RC=96, refusal names all 10 candidates tried
MUST_PASS  (same tmpdir, FS_UPSTREAM_REPO -> synthetic repo with the file):
    RC=0
```

The general form is worth stating: a control over a fallback chain is vacuous unless every arm of the chain is disarmed. The chain's entire purpose is to succeed when one arm fails, so a control that breaks one arm tests the chain's success path and reports it as the failure path.

### Closing the class by construction

Fixing two lines does not stop the next occurrence, so a fifth scan was added to the build, immediately before the publish-set coverage gate. It sweeps the publish set for build-host home paths:

```
HOME_PAT='/(Users|home)/[A-Za-z0-9_-][A-Za-z0-9._-]*/'
```

The sweep refuses UNMEASURED (95) if fewer than 20 files resolve from the publish set, so a broken denominator cannot present as a clean sweep. Measured on the fixed tree:

```
[build-host home]  56 file(s) swept, 0 hit(s)
control: planted home path 1/1 caught, elided citation 0/1 (not a disclosure)
```

The MUST_FIRE control plants two lines in a temp file and asserts exactly one hit, not at-least-one: a real home-rooted path, which must be caught, and an elided citation of the form `/home/.../probe_sub/x.sh`, which must not be. Two files in the publish set legitimately cite paths in that elided form, and a sweep that flagged them would be disabled as noise within a week. Asserting `== 1` tests the discrimination; asserting `>= 1` would not.

The sweep was also run against the pre-fix copy of the stage still present in the staged publish tree -- a must-fire on real source rather than a fixture:

```
grep -cE "$HOME_PAT" h100_backend_splice.py   ->   2
    39: SRC = pathlib.Path("/Users/.../fs-repo/launchers/fs_container_backend.sh")
    40: OUT = pathlib.Path("/Users/.../fs-build/h100")
```

### The sweep was red on itself three times running

`build_h100_plane.sh` is itself in the publish set, so the sweep sweeps the file that defines it. Three consecutive drafts failed, each for a subtler reason than the last:

1. The probe wrote `/Users/<name>/fs-repo/x.sh` as a literal -- caught, correctly.
2. Rewritten to assemble the needle with `printf` arguments, but the comment explaining why still spelled the shape out -- caught, correctly.
3. The comment's literal was replaced with the placeholder `</Users/<name>/...>` -- still caught, because the placeholder preserved the shape the pattern matches.

Draft 3 is the one worth recording. #159 had hit the first two of these with the partition literal, and the repair there was a placeholder. That repair does not transfer: #159's pattern was a fixed string, and a placeholder that changes the characters escapes it, while this pattern is a shape, and a placeholder that keeps the shape does not. A placeholder is only a placeholder if it fails the pattern. The final form assembles the needle (`printf 'SRC = "/%s/%s/..."' Users somebody`) and describes the shape in prose containing no path.

### A fourth time, in this entry

`h100/EVIDENCE.md` is in the publish set, so the sweep sweeps this entry too.
The first draft of the section you are reading carried five hits: the two
hard-coded lines quoted verbatim above, one inline reference to them in the
paragraph about scan vocabulary, and the two probe literals in the list of
failed drafts — including the very placeholder that draft 3 had already been
red for. Writing about a shape reproduces the shape. Two of those five were also
a genuine disclosure: the entry documenting a leaked username leaked it four
more times.

They were elided to a form the pattern cannot match before the entry was
appended, which is why the quotations above read `/Users/<dev>` rather than the
literal text. That is not cosmetic tidying — an evidence log that can only be
published by exempting itself from the check it documents is not evidence, it is
an exception.

### What was not fixed, stated plainly

The same pre-push gate reported 64 further hits, all of them one project-name
literal (elided here in both its casings, for the reason the next paragraph
gives), in six files committed to this repository before this work began:

```
launchers/launch_g4e4b_fullft_1tray.sh   26
tools/preflight.py                       16
launchers/launch_g4e4b_lora_1tray.sh     15
launchers/test_launcher_contracts.sh      4
tools/emit_run_manifest.py                2
tests/test_preflight.py                   1
```

None was introduced by this change, and none is in the h100_validation publish set. They are recorded here rather than silently rewritten, because quietly editing six unrelated committed files under cover of an unrelated ticket is how a diff stops being reviewable.

The first draft of the paragraph above quoted that literal in both casings, so
the gate reported it twice more — from the sentence describing it. That is
#144's use-versus-mention confusion for the fourth time in this build, and the
resolution here is deliberately the boring one rather than an acknowledgement
exception. The count and the six filenames carry the whole finding; the literal
itself carries none of it, and the entry loses nothing by not spelling it. An
exception in a blocklist is a permanent widening bought to save one word.

A seventh hit on the same file, `launch_g4e4b_fullft_1tray.sh:985`, is a false
positive of the same family in the other direction. The line exports the W&B
credential variable, assigning it an empty-default expansion of itself and
setting the project name alongside — the correct way to pass a credential
through a launcher without embedding one. The gate's key rule fired on the
variable's NAME. The detector cannot distinguish that from an embedded
credential because it is matching characters, so the line is left alone.

Quoting that line here made the gate report an eighth hit, from the sentence
exonerating the seventh. Five separate hits in this one entry, each from prose
about the thing rather than the thing: two home paths, two casings of a project
name, and a credential variable named only to say it was harmless. A document
that describes a detector is inside that detector's denominator, and every time
that has been forgotten in this build it has been forgotten while writing about
having forgotten it.

### Build result

RC=0 across 24 stages with the fifth scan in place: generators 43/0, documents 10/0, partition literal 56 swept / 0 hits, build-host home 56 swept / 0 hits with its control 1/1, publish-set coverage 56 published / 56 covered / 0 unowned, exit-contract gate 0 rejected, suite 31 passed.

### The lesson to land

A scan's denominator is not only how many files it reads; it is also how many kinds of secret it knows about. Four scans reported clean over the exact file that carried the defect, and every one of those reports was accurate about the thing it measured. The gap was never in coverage. It was that the whole vocabulary described one of the two machines involved in a build, and nothing had ever asked what the build host itself leaks into its output.
