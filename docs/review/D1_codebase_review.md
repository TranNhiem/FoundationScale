# D1 — Complete codebase review

FoundationScale describes itself, in `src/foundationscale/__init__.py`, as "verifiable distributed training" at version 0.1.0. This review was produced from a measured census (`tools/countables_census.py`, git-tracked method), the package tree with per-file sizes, the public API surface of the installable package, module docstrings, the intra-package import graph, and the argparse surfaces of the `tools/` entry points. Where evidence ends, this document says so rather than filling the gap.

## 1. What the repository physically contains

Repository-wide, the census records 118342 git-tracked .py/.sh/.md lines. The trees, from largest to smallest:

- **`h100_validation/`** — an off-package validation-and-repair plane for an H100 estate launch. `h100_validation/` (31313 .py LOC, 63 files) plus `h100_validation/*.sh` (4986 LOC, 4 files). It is the single largest tree in the repository, bigger than the installable package. It contains a large family of `patch_*.py` scripts, a second family of `gate_*.py` scripts, executor `apply_*.py` scripts, a generated subtree at `h100_validation/h100/gen/`, deliverable documents under `h100_validation/h100/` (architecture review, validation report, matrix, EVIDENCE.md, LAUNCH.md), and two pytest files.
- **`tests/`** — `tests/` (28690 .py LOC, 55 files; 96 import statements) in the census's wording. Test file names are not in the evidence slice, so this review cannot enumerate what the suite covers by file; section 9 reasons from what is visible elsewhere.
- **`src/`** — the installable package. `src/ = 18761 LOC across 25 files`. Only three files exceed 2000 lines: `src/foundationscale/provenance/manifest.py`, `src/foundationscale/gates/adjudication.py`, and `src/foundationscale/gates/checkpoint_gates.py`.
- **`launchers/`** — `launchers/` contains 9857 shell LOC plus 1615 Python LOC, according to the census. The two largest files are `launchers/test_launcher_contracts.sh` (4891) and `launchers/launch_g4e4b_lora_1tray.sh` (1913). Notably, `launchers/__pycache__/` with two `.pyc` files appears in this listing; the pycache directory also appears in the census's exclusion list, so those artifacts do not count in any anchored number, but their presence in the tree listing is a hygiene smell.
- **`tools/`** — script-level adjudicators. `tools/` contains 8916 Python LOC. Dominated by `tools/preflight.py` (3593) and `tools/mutate.py` (1745).
- **`checks/`** — four drift-blocking scripts the repository runs against itself: `checks/countables_drift.py` (which anchors the wordings under which countable claims are allowed to exist), `checks/bash_lc_sweep.py`, `checks/packaging_reachability.py`, and `checks/wf_yaml_audit.py`.
- **Top level** — the usual root files (LICENSE, Makefile, README.md, pyproject.toml) plus `artifacts/`, `assets/`, `build/`, `container_tests/`, `docs/`, and `examples/`. Two of these deserve attention: `build/` exists on disk and is in the census exclusion list — it is exactly the stale tree that double-counted the old repo-wide total before the census moved to the git index — and `container_tests/` is present but has no content in the evidence slice (unmeasured; listing it would require a tree walk of that directory).

The census itself flags that the previously circulating repo-wide total cannot be reproduced by any counting rule and must not be reused; the git-tracked figures above are the only defensible totals.

## 2. Module structure and core abstractions of the installable package

`src/foundationscale/` measures 18849 lines in the census's wording and is organised into six subpackages plus two top-level modules:

- **`checkpoint/`** — weight reading. `checkpoint/dcp.py` defines the central abstraction, `WeightSource` (a `Protocol` with `tensor_keys`, `nontensor_keys`, `shape`, `dtype`, `chunks`, `read_chunk`, `read_box`, `read_full`, `close`), plus two implementations: `DcpReader` (torch DCP directories) and `SafetensorsReader` (with a bounded `_HandleCache` for shard files). `open_weights` dispatches on format and refuses zero-tensor sources. `checkpoint/dcp_meta.py` is the torch-free metadata layer: `read_metadata` returns a `CheckpointMetadata` of `StoredTensorMeta` records, whose load-bearing field is `storage_id` — identity of bytes on disk, not of names.
- **`gates/`** — the verification engine. `gates/core.py` holds the contract: `Gate` (ABC, with `ok`/`fail`/`skip` result constructors, abstract `check`, optional `coerce_context`), `GateRegistry` / module-level `REGISTRY` / `register`, `Verdict`, `AbstentionKind`, `Coverage` (checked vs. declared, with `none()` marked vacuous), `GateResult`, `GateReport`, `GateBlocked`, and `Control` with `ControlKind` (MUST_FIRE / MUST_PASS). Concrete gate families live in `gates/checkpoint_gates.py` (`ExpertDistinctnessGate`, `ExpertByteVolumeGate`, `SaveCompletenessGate`, `FirstSaveGate`), `gates/objective_gates.py` (`ObjectiveDeclaredGate`, `LossComponentCoverageGate`, `RewardScaleSanityGate`, `HyperparameterDriftGate`), and the single parity gate in `verify/parity.py`. `gates/example.py` (`ExpertAliasGate`) is explicitly teaching material. `gates/fixtures.py` builds the deterministic synthetic expert sets consumed by controls. `gates/adjudication.py` is the production decision layer (see section 5). `gates/probe.py` holds pure measurement helpers. `gates/controls.py` is a CLI self-test (see section 4).
- **`provenance/`** — run manifests. `provenance/manifest.py` is the largest file in the package: `RunManifest`, `ManifestStore` (append-only attempts keyed by run id), `DeclaredCheckpoint`, `Topology`, `CodeProvenance` (via hermetically-env'd git subprocess calls), `capture_environment`, `ConfigResolver` / `EffectiveValue`, `require_manifest`, `_upgrade_to_current` (versioned manifest migrations), and `declared_from_hf_config` / `declared_from_megatron_args`.
- **`verify/`** — `verify/parity.py` compares two `WeightSource`s key-wise under a `TolerancePolicy` of `ToleranceRule`-selected `Tolerances`, producing a `ParityReport` of `KeyParity` entries, and registers `WeightParityGate`.
- **`train/`** — see section 6.
- **`models/`** — `ModelAdapter` protocol, `GenericHFAdapter` and `GemmaAdapter` implementations, `classify_config`, `select_adapter`, `register_adapter`, `registry_snapshot`. Classification is signal-based (`_Signal`, `_scopes`, `_verdict`) rather than name-match-only.
- **`topology.py`** — `ClusterProfile` (with `profile_by_name` and JSON loading), a second, distinct `Topology` dataclass than the one in `provenance/manifest.py` (section 11), `declared_vs_effective`, `partition_consistency`, `Finding`/`Severity`.
- **`integrate.py`** — a thin re-export layer that multi-family-dispatches `run_event` over several context types.

Two structural notes: the root `__init__.py` is three lines and exports nothing, so consumers must import submodules directly; and most package `__init__` files (`gates`, `verify`) are docstring-only.

## 3. How the major components interact — the real call paths

Path A — **training to adjudication** (the production path):

1. `train/cli.py:main` builds an `argparse` parser and constructs `TrainConfig`, then calls `train(cfg)` in `train/loop.py`.
2. `train()` resolves a `ClusterProfile` via `_resolve_profile` (backed by `topology.profile_by_name`/JSON) and computes an effective `Topology` via `_effective_topology`, whose findings can block before any GPU work.
3. The heavy imports (torch, transformers, datasets) happen *inside* `train()` — `train/__init__.py` is PEP 562 lazy and `train/loop.py` is importable torch-free.
4. The loop installs `FoundationScaleSaveGate` (a transformers `_CallbackBase` subclass) whose `on_save` invokes `_run_save_gates`, which builds a context via `_default_context_builder` and runs the registered save-event gates.
5. Manifest emission runs through `_build_run_manifest` → `_emit_manifest`, which composes `provenance.manifest.RunManifest`, `DeclaredCheckpoint` (from `_declare_checkpoint`), `ConfigResolver.record_effective`, and `ManifestStore`.
6. Final adjudication goes to `gates/adjudication.adjudicate_checkpoint`.

Path B — **adjudication internals**:

`adjudicate_checkpoint` resolves a `TrainSpec` (`resolve_train_spec`, reading launcher-side config), derives a `Declared` block (`derive_declared_block`, with the LoRA-aware census `_load_adapter_modules` and the attach rule `_lora_target_attaches`), measures the artifact (`_measure` → `dcp_meta.read_metadata`), builds a `CheckpointGateContext` (`_context`), runs the registered gates, and then exercises the MUST_FIRE controls `control_drop`, `control_alias`, `control_underfill` — the latter three prove the gate suite can still fire on this population, not merely pass. `adjudication.py` imports `derive_declared` and `run_alias_control` from `gates/probe.py`, so probe stays the pure-measurement half and adjudication is the decision half.

Path C — **gate dispatch**:

`gates/core.run_event` (re-exported by `integrate.py`) resolves each gate's context type from a `{type: object}` mapping via `_resolve_context`, catching `_AmbiguousMatch`; `coerce_context` allows per-gate conversion, and `_run_dispatched_gate` reports *unwired* contexts as explicit results instead of TypeErrors deep in gate code. Dispatch errors become ERROR `GateResult`s via `_dispatch_error`.

Path D — **parity**:

`verify/parity.py:compare_sources` coerces paths or `WeightSource`s (`_ensure_source`) — meaning it consumes the same `checkpoint/dcp.py` abstraction as everything else — computes `_guard_stats` and `_classify` per key, and lifts the result into `WeightParityGate.check`. The parity gate imports `Control`/`Coverage`/`Gate`/`register` from `gates/core`, so it registers on the same `REGISTRY` as the checkpoint and objective gates.

Named seams: `WeightSource` (artifact → verification), `CheckpointGateContext.from_path` (the boundary where a filesystem path becomes a judgment object), `coerce_context`/`_resolve_context` (multi-family dispatch), `ManifestStore` (provenance persistence), and the `provenance.manifest` private keys (`_EXPERT_COUNT_KEYS`, `_NESTED_LM_SCOPE_KEY`, `_ENABLE_MOE_BLOCK_KEY`) consumed cross-module by `models/adapters.py` and `gates/probe.py` — a deliberate but real private-symbol seam.

## 4. The gate/verification engine

The contract, as written in `gates/core.py`:

- **`Gate`** is abstract; concrete gates override `check`, optionally `controls`, and reach a result **only** through `ok`/`fail`/`skip`. `Gate.run` is the funnel point.
- **`Coverage`** requires a declared denominator; a gate checked against zero items is `vacuous` rather than pass. `is_short`/`is_over` expose mismatch relative to declared.
- **`GateReport`** aggregates results and can assert `is_vacuous`/`is_unverified`; `raise_if_blocking` throws `GateBlocked`.

**Registry**: module-level `REGISTRY` plus the `@register` class decorator; `GateRegistry.for_event(Lifecycle...)` selects gates by lifecycle, `GateRegistry.run` executes them.

**Controls**: every ship-worthy gate lists `Control` fixtures in `controls()` — MUST_FIRE inputs it must block and MUST_PASS inputs it must not. `verify_controls` executes them. The census records a large MUST_FIRE : MUST_PASS ratio across the suite (the census keys are `must_fire` and `must_pass`), consistent with the doctrine that a gate which cannot be observed firing is not evidence; the denominator pairing per gate is not visible from this slice.

**Executable self-test**: `gates/controls.py:main` walks the gate subpackages with `pkgutil`/`importlib` (`_walk_gate_packages`, `_import_gate_modules`), records import failures as findings rather than swallowing them, counts controls per gate (`_count_controls`), and flags *uncertified provenance* (`_uncertified_provenance_findings`) and *unclassified packages* (`_unclassified_package_findings`). A gate module that cannot be imported contributes nothing to the registry and the tool says so — the F1 blind spot, named in the source.

**Exit-code contract** (house doctrine, stated verbatim in `train/loop.py` and implemented in `gates/adjudication.py`): `0` PASS, `5` RED (a gate blocked), `95` UNMEASURED (a determination could not be made), `96` REFUSE (the verifier declines to answer — e.g. missing torch extra, interpreter mismatch). Adjudication additionally has `_refuse_on_interpreter_mismatch` and `GateUnmeasured`, with interpreter provenance entries (`_interpreter_report_entry`) written into the record. The distinction between RED and UNMEASURED is the engine's central design statement: absence of evidence is never a verdict.

## 5. The checkpoint plane

Three layers, bottom up.

**Readers** (`checkpoint/dcp.py`): `DcpReader` and `SafetensorsReader` both implement `WeightSource`. Read granularity is deliberately partial — `read_chunk`, `read_box` (validated by `_validate_box` and `_boxes_overlap`), `read_full`. Errors are a typed lattice: `CheckpointError` → `CheckpointFormatError`, `TensorNotFoundError`, `ChunkReadError`, `IncompleteCoverageError` (raised when a full read cannot prove coverage — the structural kill of the `all([]) is True` defect). `compare_keys` is in `__all__`.

**Metadata** (`checkpoint/dcp_meta.py`): `read_metadata` sniffs the format (`_sniff`) and dispatches to `_read_dcp_tensor_storage_id`) and per-shard safetensors headers (`_st_weight_map`). The stated purpose in the docstring is the cheap pre-flight — shapes, dtypes, storage identity — before anyone commits to a full-byte comparison. `load_manifest(path)` lives here, searching beside the checkpoint for a `RunManifest`.

**Adjudication** (`gates/adjudication.py`): covered functionally in sections 3 and 4; structurally it is the composition root for the checkpoint plane — it wires metadata, declared expectations, gate execution, control firing, and interpreter/provenance checks into one `GateDecision` with `ok`/`raise_if_blocking`. The `DeclaredBasis` TypedDict and the refusal-recording `_record_refusal` round out the record side.

**Manifests** (`provenance/manifest.py`): `RunManifest` versions are migrated by `_upgrade_to_current`; `ManifestStore.allocate_attempt`/`save`/`attempts`/`latest` provide append-only run history; `require_manifest` inverts the historical fail-open default — if provenance was not written, the consumer blocks rather than proceeds. `DeclaredCheckpoint` construction routes through `declared_from_hf_config` and `declared_from_megatron_args`, so DCP and Megatron launches both end in one declaration type.

## 6. The training entry point and what it actually delegates to

`foundationscale-train` is a console script (declared in `pyproject.toml`'s `[project.scripts]`, per the docstring in `train/cli.py` — which also documents the incident where the entry point was advertised but never installed, and gates that incident behind the willingness to edit the manifest).

`train/cli.py:build_parser`/`main` is argparse glue only. Everything real is `train/loop.py:train`:

- The actual trainer is `transformers.Trainer`, deliberately. FoundationScale contributes the verification plane around it: topology validation, save-gate callback (`FoundationScaleSaveGate`), manifest emission, adjudication, and the exit-code translation.
- Torch/transformers/datasets are optional extras; their import happens inside `train()` and their absence is REFUSE (exit 96) naming the extra, not an ImportError.
- Loop-side helpers visible in the public-surface slice: `_tied_aliases` and `_declare_checkpoint` (declaring what will be saved), `_run_id`, `_manifest_payload`, `_load_raw_dataset`.

## 7. The out-of-package planes

- **`tools/`** — `tools/` contains 8916 Python LOC. The heavy hitters: `tools/preflight.py` (login-node, torch-free pre-launch CLEAR/NOT-CLEARED blocklist whose clearance predicate is strictly stronger than `Verdict.blocking` — SKIP/VACUOUS/INAPPLICABLE all fail clearance); `tools/mutate.py` (a mutation battery over the verification framework itself, whose hard preconditions are a green suite and *zero skips*); `tools/emit_run_manifest.py` (the producer side for `RunManifest.declared` — "nothing at launch time ever populated it" is the stated motivation); `tools/real_checkpoint_probe.py` (points shipped gates at real static artifacts; denominator sourced from the adjacent HF `config.json` only); `tools/live_save_gate.py` (argument-level wrapper over `gates/adjudication.py`, re-exporting its names — see sections 10 and 11); `tools/count_census_modules.py` + `tools/census_denominator_control.py` (the LoRA census denominator and its MUST_FIRE/MUST_PASS/WIRING control); `tools/countables_census.py` (the census that produced this review's numbers, with built-in self-check keys).
- **`checks/`** — repository self-policing: `checks/countables_drift.py` anchors the countable wordings used everywhere including this document; `checks/packaging_reachability.py` and `checks/bash_lc_sweep.py` and `checks/wf_yaml_audit.py` cover packaging reachability, bash LOC drift, and workflow YAML audits respectively (inferred from docstring-level names only in the header comments of the tree; the bodies were not reviewed).
- **`launchers/`** — the launchers (`launchers/` contains 9857 shell LOC plus 1615 Python LOC, in the census's wording) across the shell files, plus the Python helpers `lora_target_census.py`, `peft_override_replay.py`, and the 769-LOC `f78_census_writer_driver.py`. Two contract test suites live here as shell (`test_launcher_contracts.sh`, `test_fs_live_gate_watchdog_contracts.sh`) — the former is larger than several source files and is the closest thing the bash plane has to a test harness.
- **`h100_validation/`** — `h100_validation/` harness (31313 .py LOC) per the census form; laid out as (a) `patch_*.py` remediation scripts (roughly thirty, from `patch_master_addr.py` through `patch_resume_proof_attribution.py`), (b) `gate_*.py` estate gates (`gate_launch_doc.py` at 1857 is the largest), (c) `apply_*.py` executors, (d) an `h100/` deliverables directory with EVIDENCE/LAUNCH/deliverable documents and a `gen/` subtree of generated ("fixed"/"spliced"/"bound") artifacts, and (e) two pytest files. This tree is a one-off estate-renovation plane that grew into the repository's largest body of Python.

## 8. Configuration and how a run is parameterised

A run is parameterised by, in order of authority:

1. **`TrainConfig`** (`train/loop.py`, frozen kw-only dataclass with `__post_init__` validation) — the in-process configuration object.
2. **`ClusterProfile`** (`topology.py`) — named profiles via `profile_by_name` or JSON via `from_json`; degree arithmetic (`total_gpus`, `model_parallel_width`) with `validate_against` producing `Finding`s.
3. **Launcher shell environment** — the real parameterisation for production runs; `h100_validation/estate.env.example` is the template of record for the H100 estate.
4. **Manifest capture** — `provenance/manifest.py`'s `ConfigResolver.record_effective(key, value, source)` + `capture_environment` value, not of a flag's default) is the mechanism that makes env-switched behaviour visible post hoc.
5. **`tools/emit_run_manifest.py`** — the CLI that assembles the launch-time manifest (`--run-id`, `--nodes`, `--gpus-per-node`, `--tp/--pp/--dp`, mode selectors, entrypoint).

## 9. Testing: what is covered, and name what is NOT

Covered, with evidence:

- The gate engine's controls themselves are executed by `gates/controls.py:main`, and `tools/mutate.py` mutates the verification framework to prove the suite can kill mutants — this is the strongest coverage claim in the repository but it is a *meta* claim: it certifies the suite's lethality, not semantic coverage of any particular defect class beyond the mutations defined there.
- `tools/preflight.py` and the bash launchers carry shell contract suites (`launchers/test_launcher_contracts.sh` is effectively a launchers-plane test runner).
- `h100_validation/` has two measured pytest files (`test_fs_argv_preflight.py`, `test_fs_ckpt_scalars.py`).
- The test suite (28550 .py LOC) over `tests/` (54 tracked test files in the census's wording) — the individual test module names are not in this slice, so per-area coverage inside `tests/` is **unmeasured** from here; enumerating it requires a file listing of `tests/`.

NOT covered, with evidence for the absence or the gap:

- The `h100_validation/patch_*.py` family: files the tree shows. Their only test presence visible in the slice is incidental. Patching scripts of several hundred lines apiece with no tests is the largest untested mass in the repository.
- `checks/` scripts and `tools/census_denominator_control.py` are exercised by themselves-as-their-own-control, not by `tests/` — the countables drift gate's self-check lives inside its own run.
- `launchers/__pycache__/*.pyc` committed artifacts imply local execution of helper scripts outside any harness.

## 10. Duplicated logic, tight coupling, unnecessary complexity

- **Duplicate files, literally**: `h100_validation/fs_argv_preflight.py` and `h100_validation/h100/gen/fs_argv_preflight.py` — both 685 lines. The `gen/` tree also re-homes copies of `fs_model_root.py`, `test_fs_model_root.py`, and "fixed" variants (`fs_train.fixed.py` at 2843 lines vs. the launcher-origin train extraction scripts). Some of this is presumably generated output, but `gen/` is tracked and human-browsable, so which file is the source of truth is unclear in the tree.
- **Two `Topology` types**: `src/foundationscale/topology.py:Topology` and `src/foundationscale/provenance/manifest.py:Topology` share a name with different shapes and no visible bridge. Importing both in one module forces aliasing gymnastics.
- **Declared-expectation derivation, thrice**: `manifest.declared_from_hf_config` / `declared_from_megatron_args`, `gates/probe.py:derive_declared`, and `adjudication.derive_declared_block` all derive "what the run said it would write" from overlapping inputs (HF config, Megatron args, train config). The seams are intentional per docstrings but the coupling (via `_EXPERT_COUNT_KEYS`, `_NESTED_LM_SCOPE_KEY`, `_ENABLE_MOE_BLOCK_KEY` — private symbol imports across subpackage boundaries) makes refactor ripples wide.
- **`adjudication.py` ← `live_save_gate.py`**: the census's `shim_names`/`shim_private` keys confirm `tools/live_save_gate.py` is now a 493-LOC shim re-exporting 94 names (60 of them private — `_`-prefixed module internals) from the package module. The extraction is documented (review finding T2_lib_script_boundary#0) but a 60-name private re-export surface is a coupling batch, not a seam.
- **`gate_launch_doc.py`** (1857 lines, H100 plane) vs. the package gates: the estate-plane gate scripts reimplement exit-doc, stage-doc, and naming closures that the package's exit-code contract already encodes — doctrine drift between the estate plane and the library plane, visible in the parallel `gate_*`/`patch_*` naming vs. the `CaseGate`-style class structure.

## 11. What surprised you / what a newcomer would misread

- **Docstring-first authoring**: the module docstrings are unusually long and read as incident reports. They are genuine documentation of *why* (Incidents #1/#6, finding numbers like #83, #219, #224), and a newcomer who skims code and skips prose will miss the rationale that justifies every structural choice.
- **`__init__.py` files export almost nothing**: the root package is three lines; `gates/__init__.py` and `verify/__init__.py` carry docstrings and a one-line sentinel. All consumption is by explicit submodule path. This is deliberate hygiene (registry side effects) but reads as "no API."
- **Exit codes 0/5/95/96**: unusual nonzero values are doctrine, not accidents — 95 UNMEASURED and 96 REFUSE exist so that a tool declining to answer is distinguishable from a tool detecting corruption.
- **The largest tree is not the product**: `h100_validation/` outweighs `src/`. It is an estate-renovation workshop, not user-facing code, and most files there are one-time patches.
- **`build/` on disk**: exists in the top-level listing and in the census exclusion list; it is the historical double-count source and should be deleted, not documented around.
- **`.pyc` files in the tracked tree listing for `launchers/` and `h100_validation/`**: excluded from all census numbers but present where they shouldn't be; a newcomer will wonder which tree is authoritative.
- **Caveat on this review's denominators**: the `tests/` breakdown by module, `container_tests/` contents, and the per-file coverage of the gate suite's controls could not be measured from the supplied slice; those remain unmeasured rather than passed.
