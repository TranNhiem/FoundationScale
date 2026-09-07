# Distributed training

FoundationScale's distributed story has exactly two shipped layers, and this chapter
draws the boundary between them precisely:

1. **`src/foundationscale/topology.py`** — a library module that validates the declared
   parallel geometry *before any process group exists*. A nonsense layout fails on the
   login node, not at NCCL init.
2. **`launchers/fs_container_backend.sh`** — the launch plane. It routes every
   in-container step — preflight probes and the training run alike — through one
   backend function, with `FS_USE_TORCHRUN` selecting torchrun on the enroot arm.

A library-level distributed runtime — the DP/TP/PP/EP strategies as runtime
implementations — is design-stage. **Do not read this document as an API for one.**
There is no process-group management, no collective wrapper, and no sharding strategy
in the package. What exists is a way to prove a decomposition is coherent, against
the cluster it will land on, before a single GPU is touched.

## Why a validator instead of a runtime

A ground-truth probe of the audited estate measured 526 shell files, of which 189 are
launchers. Every check in `topology.py` encodes one of the measured failures:

| # | Measured failure | Where it is caught now |
|---|---|---|
| 1 | Pipeline parallelism absent: `pipeline_model_parallel_size` in **zero** of 189 launchers | `validate_against` demands evidence before trusting `pp > 1` |
| 2 | Context parallelism dead: `CP=${CP:-1}` in 44 places, four entrypoints hardcode `context_parallel_size = 1` | `declared_vs_effective`, `runtime_overrides` |
| 3 | Two dialects for the same degrees (Hydra `model.*_parallel_size=`, argparse `--tp/--etp/--ep`) | one `Topology` object both paths build |
| 4 | World size welded: `--nodes` only ever in {1,2,4,8} × `--gpus-per-node=4` | `ClusterProfile.max_nodes` carries the real limit as data |
| 5 | Rendezvous wrong two ways: `MASTER_ADDR=127.0.0.1` in 27 launchers, a hardcoded node name in 22 | `topology.master_addr_loopback`, `topology.master_addr_hardcoded_node` |
| 6 | `--ntasks-per-node=8` against `--gpus-per-node=4` → `ncclInvalidUsage / Duplicate GPU detected`, 2m10s into the job | `Topology.__post_init__` rejects `tasks_per_node != gpus_per_node` at construction |
| 7 | Two partition spellings, 188 files vs 4, never compared | `partition_consistency` |

The governing rule of the whole module: **a claim must carry its own coverage.**
`all([])` is `True`, and a check that examined zero units reports VACUOUS, not PASS.
Every validator below therefore either always emits a summary finding or refuses to
return a clean verdict on an empty input.

## ClusterProfile: the cluster as data

`ClusterProfile` is everything a launcher needs to know about a cluster — name,
scheduler, `partitions`, `node_pattern`, `gpus_per_node`, `nccl_socket_ifname`,
`ib_hca_pattern`, `mnnvl_available`, `container_runtime`, `container_image`,
`filesystem_roots`, and an optional `max_nodes` — as one validated, loadable
structure. Two ship as literal dicts in `PROFILES` (`"slurm-generic"` with
`max_nodes: 64`, `"local-single-node"` with `max_nodes: 1`); adding a third is one
more dict, no code changes. `profile_by_name` raises `KeyError` on an unknown name
with a message saying exactly that.

Two design points are load-bearing:

- **`max_nodes` is cluster truth, not launcher habit.** The estate's 8-node ceiling
  was a script artifact — other users on the same cluster ran 18 nodes. The profile
  states the real limit so the ceiling stops being inherited by copy-paste.
- **`from_dict` refuses unknown and missing keys.** A misspelled key in a profile
  file must not silently reset a field to its default. Silent config drift is the
  incident class this module exists to kill.

`from_json` accepts JSON text or a path; `to_dict` round-trips. Construction
validates: `gpus_per_node >= 1`, `node_pattern` must compile, and a pinned
`container_image` is mandatory whenever `container_runtime` is not `"none"` — an
unpinned image is the unrecorded-config failure.

## Topology: validated arithmetic at construction

`Topology` carries the five parallelism degrees — `dp`, `tp`, `pp`, `ep`, `cp` —
plus `nodes`, `gpus_per_node`, and an optional `tasks_per_node`. It is frozen, and
its `__post_init__` is the launch gate:

- every degree and layout field must be a positive `int` (booleans rejected);
- the product `dp × tp × pp × ep × cp` must equal `nodes × gpus_per_node`, and the
  `ValueError` **shows the arithmetic** on mismatch;
- `tasks_per_node` defaults to `gpus_per_node`, and any explicit unequal value is
  rejected, because one-process-per-GPU is the only mapping this stack supports.
  This is failure (6) — the Duplicate-GPU crash — caught at launch instead of 2m10s
  into the job.

```python
from foundationscale.topology import Topology, profile_by_name

topo = Topology(dp=2, tp=4, pp=1, ep=1, cp=1, nodes=2, gpus_per_node=4)
profile = profile_by_name("slurm-generic")
findings = topo.validate_against(profile, master_addr="127.0.0.1")
print(topo.describe())
```

Derived quantities come free: `total_gpus`, `model_parallel_width`
(`tp × pp × ep × cp`), `degree_dict()`, and `describe()` — a one-screen launch
summary whose first line carries the total and whose remaining lines show the
arithmetic producing it. If `pp > 1` or `cp > 1`, `describe()` adds a scrutiny
line: in the measured estate, 0/189 launchers ever exercised pipeline parallelism
and CP was hardcoded to 1 in every RL/SFT entrypoint.

## Findings and severity

All consistency checks return `Finding` objects — a stable `code` under
`topology.*`, a `Severity`, a human message, JSON-safe `details`, and a `control`
naming the positive control proving the check can fire. Severities:

| Severity | Meaning |
|---|---|
| `OK` | Coverage reporting — "0 defects in N units", not silence |
| `WARN` | Something demands evidence or confirmation; the launch proceeds |
| `BLOCK` | Stops the launch. Only BLOCK blocks. |

`blocking(findings)` filters to BLOCKs; `render_findings` renders for a log and is
explicitly non-empty on no findings — an empty result must never be mistaken for a
clean one.

## validate_against: topology vs. the real world

`topo.validate_against(profile, master_addr=..., num_experts=...,
runtime_overrides=...)` runs the measured failure classes and **always appends a
coverage summary** (`topology.validate_summary`) stating how many checks ran:

- **`topology.gpus_per_node_exceeds_profile` (BLOCK)** — requested GPU shape exceeds
  what the profile says the scheduler offers.
- **`topology.nodes_exceed_profile_limit` (BLOCK)** — past `max_nodes`; the message
  says to fix the *profile data*, not the launcher.
- **`topology.tp_crosses_node_boundary` (BLOCK, or WARN if the profile reports
  MNNVL)** — a TP group silently spanning nodes rides the network for every
  all-reduce; with `mnnvl_available` it downgrades to "confirm this was intended."
- **`topology.ep1_replicates_experts` (WARN)** — `ep=1` on an MoE replicates every
  expert per model-parallel rank; measured at 88.7% of parameters replicated where
  the fix was `ep=2`, not a reshard. **`topology.ep_uneven_expert_shard` (BLOCK)** —
  `num_experts` not divisible by `ep`.
- **Rendezvous, only when `nodes > 1`:** `master_addr=None` →
  `topology.master_addr_unrecorded` (WARN — unrecorded config is how runs silently
  diverge); a loopback host → `topology.master_addr_loopback` (BLOCK, measured in 27
  launchers); a concrete hostname matching `profile.node_pattern` →
  `topology.master_addr_hardcoded_node` (BLOCK, measured in 22 — resolve at batch
  time, e.g. via `scontrol`).
- **PP/CP honesty:** with `runtime_overrides={"pp": 1, "cp": 1}` (the shape of the
  four measured entrypoints) a declared/forced mismatch is `topology.runtime_overrides_pp/cp`
  (BLOCK); without overrides, `pp > 1` or `cp > 1` draws `topology.pp_unverified` /
  `topology.cp_unverified` (WARN) demanding `declared_vs_effective` evidence.

## declared_vs_effective: the check nobody had

`declared_vs_effective(declared, effective)` compares two `Topology` objects —
one built from the launch configuration, one read back from the constructed model.
Every differing field among the seven (`dp`, `tp`, `pp`, `ep`, `cp`, `nodes`,
`gpus_per_node`) produces `topology.effective_overrides_<field>` (BLOCK), followed
by an `OK` coverage summary stating all 7 fields were compared; on a full match it
returns `topology.effective_matches_declared`. This is the only structure that
catches failure (2): 44 launchers exporting `CP=${CP:-1}` while four entrypoints
hardcoded `cp=1` — a knob that appears configurable and is not is worse than a
missing knob.

## partition_consistency: corpus-level truth

`partition_consistency(files)` takes a `path → contents` mapping and returns a
single finding:

| Code | Severity | Condition |
|---|---|---|
| `topology.partition_scan_empty` | BLOCK | zero files supplied — a scan of nothing cannot assert consistency |
| `topology.partition_not_found` | BLOCK | files scanned, zero partition declarations extracted |
| `topology.partition_spelling_variants` | BLOCK | one partition under multiple spellings; details carry per-spelling counts |
| `topology.partition_consistent` | OK | one spelling per partition, with files scanned reported |

Extraction recognises `--partition=x`, `partition = "x"`, and `-p x` **only on
sbatch/srun lines** (`-p` is overloaded everywhere else — `mkdir`, `cp`, `tee` —
and overmatching would manufacture variants out of nothing). Normalisation ignores
case and separators: `gpu-h100`, `gpu_h100` and `GPU.H100` are one partition. The
check **does not assume the majority spelling is correct** — the estate split 188/4
— it reports the split and refuses to resolve it for you.

## The launch plane

Multi-process orchestration lives outside the library, in
`launchers/fs_container_backend.sh`. Per the README: every in-container step —
preflight probes and the training run alike — goes through one backend function,
and `FS_USE_TORCHRUN` selects torchrun on the enroot arm. The shipped launchers are
single-tray (one node) by construction, and their node guards — `FS_ALLOWED_NODE`
and `FS_FORBIDDEN_NODES` — exist precisely so a script cannot wander onto another
team's hardware by accident. The backend script's full surface is documented by the
launcher tree, not here; this chapter covers the geometry gate it should run before
dispatch.

## Multi-node training

**Multi-node training is not implemented in the package.** The launchers above are
single-node, deliberately. The multi-node story today is
`validation_campaigns/h100_validation/` — an experimental harness carrying the
hardening patches, launch gates and evidence documents for scaling one H100 estate.
It has its own README at `validation_campaigns/h100_validation/README.md` and
published deliverables under `validation_campaigns/h100_validation/h100/`. Treat it
as a lab notebook that CI gates, not as supported surface.

Concretely, the gaps a reader should not paper over:

- **No multi-process runtime exists.** `Topology.dp/tp/pp/ep/cp` are validated
  numbers, not a process group. Nothing in the package initialises NCCL, assigns
  ranks, or shards a model.
- **`validate_against` checks the *geometry* of a multi-node run** (rendezvous
  address, node ceiling, TP placement) and will catch the measured disasters, but
  it does not make the run possible.
- **Pipeline parallelism is unexercised anywhere**: 0 of 189 measured launchers set
  it, and the module's posture is to demand `declared_vs_effective` evidence rather
  than to bless it.
