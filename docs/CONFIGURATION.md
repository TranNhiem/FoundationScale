# Configuration

FoundationScale's configuration has no global config file. There are three layers, and each has its own mechanism, its own failure modes, and its own enforcement point. This chapter covers all three: the package layer (Python objects), the train-loop layer (environment variables), and the launch plane (environment variables only).

## The package layer: configuration as Python objects

The package takes configuration as Python objects — context objects per gate, and topology/profile declarations for the train loop. Anything that would traditionally live in a YAML block lives in a dataclass whose constructor refuses invalid states.

### The doctrine in one sentence

A configuration error should be a `ValueError` raised at construction time, showing the arithmetic, rather than a crash 2m10s into a job. Both `ClusterProfile` and `Topology` validate themselves in `__post_init__`.

## `ClusterProfile`: the cluster as data

`ClusterProfile` is **data, not code**. It carries every machine fact a launcher previously hardcoded: scheduler, partitions, hostname pattern, GPUs per node, interconnect, container, filesystem roots, and the cluster's real node ceiling.

| Field | Meaning |
|---|---|
| `name` | Short identifier, e.g. `"slurm-generic"` |
| `scheduler` | `"slurm"`, `"pbs"`, `"local"`, …; informational except where a validator keys on it |
| `partitions` | Valid scheduler partition names |
| `node_pattern` | Regex matching any concrete schedulable hostname; used to detect a hardcoded node name masquerading as `MASTER_ADDR` |
| `gpus_per_node` | Accelerators per node |
| `nccl_socket_ifname` | Value for `NCCL_SOCKET_IFNAME` |
| `ib_hca_pattern` | Glob/regex for InfiniBand HCAs; `""` if none |
| `mnnvl_available` | Whether NVLink spans multiple nodes |
| `container_runtime` | `"enroot"`, `"docker"`, `"apptainer"`, or `"none"` |
| `container_image` | Fully-pinned image reference; `""` only when runtime is `"none"` |
| `filesystem_roots` | Mounts that must exist for artifacts/logs/data |
| `max_nodes` | The cluster's real node ceiling, or `None` if unconstrained |

`max_nodes` is deliberately *not* inherited from launcher habit. The measured estate's 8-node ceiling was a script artifact (`--nodes` only ever `{1,2,4,8}` against `--gpus-per-node=4`) while other users on the same cluster ran 18 nodes. If the real limit differs from the profile, fix the profile data, not the launcher.

### Loading profiles

Profiles load from JSON via `ClusterProfile.from_json` (a path, or JSON text starting with `{`) or from a parsed mapping via `ClusterProfile.from_dict`. `to_dict` serialises back and round-trips through `from_dict`.

`from_dict` **refuses unknown keys**. This is load-bearing: a misspelled key in a profile file must not silently reset that field to its default. It also refuses missing required keys.

### Built-in profiles

Two profiles ship as literal dicts in `_PROFILE_DATA` and are exposed through `PROFILES` — a read-only mapping keyed by name. Look one up with `profile_by_name(name)`; an unknown name raises `KeyError` with a message that itself documents the extension mechanism: new clusters are added as data, one dict, no code changes.

### Construction-time validation

`ClusterProfile.__post_init__` rejects, among other cases:

- empty `name`, `scheduler`, or `partitions`;
- `gpus_per_node < 1` (booleans are rejected as non-integers);
- a `node_pattern` that does not compile as a regex;
- `container_image` set when the runtime is `"none"`;
- an unpinned `container_image` when a runtime is used — an unpinned image is exactly the unrecorded-config failure;
- `max_nodes < 1` when set.

## `Topology`: the parallelism decomposition, validated at construction

```python
from foundationscale.topology import Topology

topo = Topology(dp=1, tp=8, pp=1, ep=1, cp=1, nodes=1, gpus_per_node=8)
```

Construction **is** the launch gate. `__post_init__` requires every degree (`dp`, `tp`, `pp`, `ep`, `cp`) and layout value (`nodes`, `gpus_per_node`) to be a positive integer, then verifies the product rule: `dp x tp x pp x ep x cp` must equal `nodes x gpus_per_node`, or the object never exists and the error message shows both sides of the arithmetic.

`tasks_per_node` defaults to `gpus_per_node`. Any other explicit value is rejected: one launcher slot per GPU is the only mapping the stack supports. This encodes a measured crash — `--ntasks-per-node=8` against `--gpus-per-node=4` died with `ncclInvalidUsage / Duplicate GPU detected` 2m10s into the job, before a single weight was read. Checking at launch costs milliseconds.

### Derived quantities

| Member | Value |
|---|---|
| `total_gpus` | `nodes * gpus_per_node` |
| `model_parallel_width` | `tp * pp * ep * cp` — GPUs holding exactly one model replica |
| `degree_dict()` | The five degrees as a plain mapping |
| `describe()` | One-screen launch summary showing the total first, then the arithmetic that produces it |

`describe()` appends a scrutiny line whenever `pp > 1` or `cp > 1`, because zero of 189 measured launchers exercised pipeline parallelism end-to-end and `cp` was hardcoded to 1 in every RL/SFT entrypoint.

### The two dialects this replaces

The estate had two incompatible ways to type the same degrees — Hydra-style `model.*_parallel_size=` on the SFT path and argparse `--tp/--etp/--ep` on the RL path, with literals like `--tp 8 --etp 1 --ep 8` re-typed by hand inside command strings. `Topology` is the single object both paths build and both paths validate.

## `Topology.validate_against`: checking config against the real world

```python
findings = topo.validate_against(
    profile, master_addr=args.master_addr, num_experts=8,
    runtime_overrides={"pp": 1, "cp": 1},
)
```

`validate_against` returns a list of `Finding` objects — one per defect, plus a coverage summary that is **always** appended. A clean result is distinguishable from a validator that examined nothing.

| Code | Severity | What it encodes |
|---|---|---|
| `topology.gpus_per_node_exceeds_profile` | BLOCK | Requested GPUs/node exceeds the profile |
| `topology.nodes_exceed_profile_limit` | BLOCK | `nodes` beyond `profile.max_nodes` |
| `topology.tp_crosses_node_boundary` | BLOCK (WARN if `mnnvl_available`) | `tp > gpus_per_node`: every TP all-reduce rides the network |
| `topology.ep1_replicates_experts` | WARN | `ep=1` on an MoE replicates every expert on every model-parallel rank |
| `topology.ep_uneven_expert_shard` | BLOCK | `num_experts` not divisible by `ep` |
| `topology.master_addr_unrecorded` | WARN | Multi-node run with no `master_addr` supplied to check |
| `topology.master_addr_loopback` | BLOCK | `MASTER_ADDR` is loopback with `nodes > 1`; measured verbatim in 27 of 189 launchers |
| `topology.master_addr_hardcoded_node` | BLOCK | `MASTER_ADDR` matches the profile's `node_pattern`; the launcher is capped at one machine |
| `topology.runtime_overrides_{pp,cp}` | BLOCK | Config requests a degree the runtime will silently force elsewhere |
| `topology.{pp,cp}_unverified` | WARN | `pp`/`cp > 1` with no override evidence; demands `declared_vs_effective` |
| `topology.validate_summary` | OK | Checks run, blocking count, warning count — always emitted |

### `Finding` and `Severity`

Every finding is a frozen `Finding`: `code` (a stable `topology.*` identifier), `severity`, `message`, `details` (JSON-safe evidence), and `control` — the positive control proving the check can fire. An `OK` finding with no control is indistinguishable from a broken detector, which is the `all([]) is True` incident class: `all([])` is True, so a check that reports nothing and names no control is reporting **vacuity, not success**.

`Severity` has three values: `OK`, `WARN`, `BLOCK`. Only `BLOCK` stops a launch. `OK` exists precisely so a check can report *coverage* when it found nothing. `blocking(findings)` filters to the findings that must stop the job; `render_findings(findings)` renders for a launch log and is explicitly non-empty on an empty list — `(no findings — this is itself suspicious; validators never return empty)`.

## `declared_vs_effective`: the comparison nobody had

The RL entrypoints hardcoded `m.context_parallel_size = 1` and `m.pipeline_model_parallel_size = 1` while 44 launchers set `CP=${CP:-1}`; the shell variable was never seen by the code and nothing ever compared the two. `declared_vs_effective(declared, effective)` compares all seven fields (five degrees plus `nodes` and `gpus_per_node`) between the topology built from the launch manifest and the one read back from the constructed model:

- every mismatch emits a BLOCK `topology.effective_overrides_{field}`;
- mismatches are followed by an OK `topology.effective_comparison_summary` stating how many fields were compared;
- a full match emits OK `topology.effective_matches_declared` — again with the comparison count in `details`.

## `partition_consistency`: scanning a launcher corpus

`partition_consistency(files)` takes a mapping of path → file contents and returns a single `Finding`:

| Code | Severity | Condition |
|---|---|---|
| `topology.partition_scan_empty` | BLOCK | Zero files supplied — a scan of nothing cannot assert consistency |
| `topology.partition_not_found` | BLOCK | Files scanned, zero partition declarations extracted |
| `topology.partition_spelling_variants` | BLOCK | One partition appears under multiple spellings (case and separators ignored: `gpu-h100`, `gpu_h100`, `GPU.H100` are one partition) |
| `topology.partition_consistent` | OK | One spelling per partition, with scanned-file counts in `details` |

The check does **not** assume the majority spelling is correct. The measured estate split 188 files one way and 4 the other on a single partition; the finding reports the split and refuses to resolve it. Extraction keys on `--partition`, `partition=`, and the short `-p` form only on `#SBATCH`/`sbatch`/`srun` lines, because `-p` is overloaded everywhere else.

## The train-loop layer: `foundationscale-train`

The CLI is installed by `[project.scripts]` as `foundationscale-train`; the heavy dependencies are the `train` extra (`pip install 'foundationscale[train]'`). Both facts are declarations in pyproject.toml, verified by `checks/packaging_reachability.py` against the installed distribution — for one release the packaging stanza lived only in a docstring, so `prog="foundationscale-train"` advertised a command `pip install` never created (finding #224).

### The three-layer boundary as flags

`main(argv)` builds a `TrainConfig` from parsed arguments and hands it to `train(cfg)`:

- **Machine facts fail closed.** `--nodes` and `--gpus-per-node` are `required=True`; they have no defaults.
- **The profile is mandatory and exclusive.** Exactly one of `--profile-name` (a key into `PROFILES`) or `--profile-path` (a JSON file loaded through `ClusterProfile.from_json`) is required.
- **Degrees default to 1.** `--dp --tp --pp --ep --cp` each default to 1; they are composed into a `Topology` and validated against the chosen profile **before** touching a GPU.
- **`--dry-run`** runs the entire validation prologue and exits without touching a GPU.
- **`--launch-corpus`** points at a directory of launcher scripts scanned for partition spelling variants via `partition_consistency`. Omit it and the scan reports UNMEASURED — it is never assumed clean.

Model and data configuration is equally explicit: `--model`, `--dataset`, `--output-dir` are required; `--max-steps`, `--per-device-batch-size`, `--learning-rate`, `--save-interval`, and `--seed` carry defaults.

### Environment variables

The train loop reads `FS_RUN_ID` and `FS_ATTEMPT` from the environment.

### Exit codes

| Code | Meaning |
|---|---|
| 0 | PASS |
| 5 | RED |
| 95 | UNMEASURED |
| 96 | REFUSE |

UNMEASURED (95) is not a soft pass: it is the CLI-level analogue of VACUOUS — the run reported nothing because it measured nothing, e.g. `--launch-corpus` omitted.

## The launch plane: environment variables only

The launch plane is configured entirely through environment variables, so that no account name, hostname or path is committed to the repository:

| Variable | Role |
|---|---|
| `CLUSTER_HOME` | Estate root |
| `FS_ALLOWED_NODE` | Required, **no default** — an unset guard is a disabled guard |
| `FS_FORBIDDEN_NODES` | Forbidden-node list |
| `FS_BACKEND` | Backend selection |
| `FS_USE_TORCHRUN` | Whether to launch via torchrun |
| `FS_RUN_ID` | Run identity (read by the train loop) |
| `FS_ATTEMPT` | Attempt counter (read by the train loop) |

**No single reference enumerates the full set of launch-plane variables.** They are documented in the header comments of the launchers themselves, which is where they are enforced; variables beyond those in the table above are not catalogued here because they have not been collected into this document yet. If you need a variable not listed, read the launcher header that consumes it — do not guess a name and trust it, because a misspelled environment variable is unset, and an unset guard is a disabled guard.

The same catalogue gap applies to the gates' per-gate context objects: the package layer is stated as "context objects per gate" in the package contract, but the individual context-object field shapes are not part of this chapter's source material.
