# Observability: Monitoring and Debugging Gate Verdicts

FoundationScale's debugging surface is built on one rule: a claim must carry its own coverage. `all([])` is `True`, so a check that examined zero units is never allowed to report success — it reports **VACUOUS**, or refuses to run at all (**UNMEASURED**). This chapter documents where that rule becomes visible to an operator: the verdict strings, the exit codes, the on-disk refusal records, and the manifest that separates what a run claimed from what was measured.

## Verdict vocabulary

The decision layer (`src/foundationscale/gates/adjudication.py`) emits exactly three verdicts at the top level:

| Verdict | Exit code | Meaning |
|---|---|---|
| `CLEAR` | 0 (`EXIT_CLEAR`) | Every gate ran, every blocking check passed, at least one MUST_FIRE control fired |
| `BLOCKED` | 1 (`EXIT_BLOCKED`) | A measured property of the checkpoint fails, or a detector could not be proven live |
| `UNMEASURED` | 3 (`EXIT_UNMEASURED`) | The tool itself could not measure — a refused run, **not** a checkpoint verdict |

Exit code 2 is argparse's and is deliberately left alone; the comment at `EXIT_UNMEASURED` notes it is "kept distinct from the probe's contract". The distinction between 1 and 3 is the debugging affordance that matters most: **exit 3 never accuses the checkpoint.** `GateUnmeasured` is documented as "The tool could not measure. Distinct from 'measured, and it blocks'", and every refusal message states this explicitly — e.g. the `--adapter-prefix` refusal opens with "exit 3 -- a refused measurement, not a checkpoint verdict".

## Coverage renders inline, always

Counters travel with the verdict, on every path, not just failures:

* Gate output: `[VACUOUS] … examined 0 of 128` — the denominator is part of the statement.
* Auto kind inference: `auto: {marked}/{len(judged)} adapter-namespace tensors carry an adapter marker ({frac:.2f}), with {excluded} non-adapter checkpoint namespace entries excluded from the denominator`.
* LoRA structural sweep: `lora: 0 of {len(real)} real tensors could be bound to a base parent … That is a vacuous detector, and it blocks.`
* Artifact overlap in the declared basis: `artifact overlap {overlap}/{len(artifact_real_fqns)}`.
* Green-path exclusions are counted too: when `_is_non_adapter_namespace` sets `optimizer.*` / `rng_state` entries aside from LoRA adjudication, a note records exactly how many were excluded (`set {len(non_adapter)} non-adapter checkpoint namespace entry(ies) aside …`), because "an exclusion that silently shrinks a population is indistinguishable from a detector that stopped working".

There is also a framework invariant enforced in code: any `GateResult` with `verdict is Verdict.PASS and coverage.checked == 0` appends the blocking reason `PASS over 0 checked units -- framework invariant breach` regardless of what the gate thought it was doing.

## The exit-3 class is demultiplexed on disk

An UNMEASURED exit is multiplexed — many causes share one exit code — so `_record_refusal` writes a JSON refusal record to `--json` *at the point of refusal*, before any gate runs. The record honestly states its own denominators: `"gates_exercised": "0 of 3"`, `"controls_exercised": "0 of 3"`, because the refusal precedes any verdict by definition.

The launcher distinguishes causes via `refusal_class`, computed by `_refusal_class`:

| `refusal_class` | Trigger | Launcher treatment |
|---|---|---|
| `adapter_prefix_unpinned` | message starts with `--adapter-prefix was not pinned` | the one chosen abstention; calibrated to rc 0 |
| `adapter_census_unavailable` | message contains `--adapter-modules` | rc-92 wiring class |
| `checkpoint_unreadable` | message contains `checkpoint unreadable:` | rc-92 wiring class |
| `other_unmeasured` | anything else (tool bug, missing base files, …) | rc-92 wiring class |

Two operational notes from the source:

* **Matching order is load-bearing.** The prefix refusal's guidance text itself names `--adapter-modules`, so the prefix arm is tested first; inverted, every prefix refusal would misclassify as a census refusal and land on the wrong launcher arm. Error-message wording here is a contract — reclassifying a cause is "a deliberate act, never a side effect of rewording an error message downstream".
* **A refusal that cannot record itself is loud.** If the `--json` write fails, `_record_refusal` prints to stderr and the launcher maps a claimed-but-absent record to its rc-92 class. If no `--json` was given at all, stderr says so: "the caller waived the record".

## The CLEAR/BLOCKED report

`adjudicate_checkpoint` writes a JSON report (via `json_out`) whose observability-relevant blocks are:

* `inventory` — `origin`, `format`, `entries_total`, `real_tensors`, `base_tensors`, `base_source`. What was actually on disk, next to the independent base.
* `declared_basis` — a `DeclaredBasis` with five string basis statements (`run_kind`, `fqns`, `num_experts`, `num_moe_layers`, `expected_expert_bytes`) plus the `notes` list. Each denominator names where it came from; e.g. the expert census note reports `N of M base-header tensor names match the expert classifiers` and states it was computed over the *unfiltered* base key set.
* `gates` — every `GateResult.to_dict()`. The always-run gates are `ExpertDistinctnessGate`, `ExpertByteVolumeGate`, `SaveCompletenessGate`; `FirstSaveGate` is added only for `event == "first_save"`, because its contract is scoped to that lifecycle event and "running it on save #347 would be a claim outside its own declared events".
* `controls` — the MUST_FIRE control reports (below).
* `blocking_reasons` — every reason, each prefixed `gate_id=VERDICT: detail-first-line`.
* `interpreter` — the #83 provenance block (below).

Programmatic callers get the same information as a `GateDecision`: `.ok`, `.exit_code`, `.verdict`, and `raise_if_blocking()`, which raises `RuntimeError` naming the verdict, exit code, checkpoint, and up to eight blocking reasons.

## MUST_FIRE control statuses

Every adjudication injects defects into copies of *this artifact's* metadata to prove the gates can still fire. The control vocabulary, and how the consume loop grades it:

| Status | Meaning | Effect |
|---|---|---|
| `fired` | the gate answered the injection with FAIL, attributable to the injection | required: at least one control must fire |
| `not_fired` | clean baseline, injection produced no block — a true negative | blocking: "the detector cannot be trusted on this artifact" |
| `unconstructable` | the control could not be built on this artifact | blocking: "an unexercised detector proves nothing" |
| `inconclusive` | baseline absent/already blocking, or the detector answered with ERROR/VACUOUS | blocking: "an exercised-but-unattributable control proves nothing" |
| `inapplicable` / `skipped` | the claim genuinely does not exist (e.g. dense artifact, no aliasing claim) | recorded, non-blocking — but only tolerable because the `any_fired` floor still requires a *different* control to have fired |
| anything else | unrecognized status string | blocking: "a control vocabulary this loop cannot read is an unproven detector" |

If nothing fired at all, the decision blocks with `no MUST_FIRE control fired on this artifact -- the run proved nothing about the detectors`. Attribution is strict: only `Verdict.FAIL` with a clean baseline counts as a fire; a blocking-but-not-FAIL answer is a malfunction or coverage failure, "the verifier-exception-counted-as-pass fallacy". The builders are `control_drop`, `control_alias`, and `control_underfill`, dispatched through `_CONTROL_BUILDERS`; an unknown control name is recorded as `unconstructable`.

One divergence is deliberately surfaced rather than smoothed over: if the shared `_split_expert_layouts` router finds an aliasable shard group but the probe's alias control answers `skipped`, the record is graded `unconstructable` with `probe_status: "skipped"` preserved — a classifier divergence for a human to settle, "an auditable divergence, not a laundered one".

## Interpreter provenance (#83)

Every verdict — and every refusal — is attributed to a measured interpreter. `_interpreter_provenance()` records `python_executable`, `python_version`, and a `torch_record` (`detected`, `origin`, `dist_version`, `basis`), located via `importlib.util.find_spec` and `importlib.metadata.version`; torch is *never imported* for this record, "because importing just to record it would let the measurement change the measured".

The caller's expectation arrives as the `expected_interpreter` kwarg or, failing that, the environment variable `LIVE_SAVE_GATE_EXPECT_INTERPRETER`, with vocabulary `host` (no torch) or `container` (torch + DCP). Handling rules, all fail-closed to UNMEASURED:

* A value outside `{host, container}` is a refusal, never a silent normalization.
* The record is re-probed in-process before use: if the record's executable, version, or torch detection disagrees with a fresh probe, it is an authored or foreign record — "a record that lies about the interpreter that wrote it attributes nothing".
* Expected `container` with no torch detected (or vice versa) refuses *before any gate runs*: "host and container verdicts are not interchangeable".
* With no expectation, the run is recorded but uncontested, and the report entry says exactly that.

On the refusal path the raw (possibly malformed) expectation is transcribed verbatim — "if it is malformed, that malformed value is very possibly WHY the refusal exists".

## Tools at the edge

The decision logic lives in the package; the CLIs are thin wrappers over it:

* `tools/live_save_gate.py` — an argparse wrapper over `foundationscale.gates.adjudication`, re-exporting every name so callers resolve unchanged after the extraction.
* `tools/real_checkpoint_probe.py` — a thin CLI over `foundationscale.gates.probe`. The probe's declaration machinery is imported by the adjudication module as `derive_declared` and `run_alias_control`; the controls contract requires controls to exercise the gate's *own* selection logic and dtype table, so private members (`_SHARD_SUFFIX_RE`, `_expert_weights`) are imported as deliberate drift tripwires — a rename or deletion in `checkpoint_gates.py` fails loudly at import time.
* `tools/emit_run_manifest.py` — refuses dishonest emissions; the run manifest at `src/foundationscale/provenance/manifest.py` is where what-was-claimed vs what-was-measured is made explicit for a run. The manifest shares vocabulary with the gate report: the interpreter record's keys mirror the manifest's training-stack entries (`python_executable`, `python_version`, `torch_record`) so a manifest-side directive resolves to a field the gate actually writes.

The detailed surfaces of `emit_run_manifest.py` and `manifest.py` — their flags, record schema beyond the training-stack entries named above, and rejection conditions — are not implemented in the source material this chapter documents; the observable contract stated here (dishonest emissions are refused; claimed-vs-measured is explicit) is all that can be asserted. Similarly, `tools/real_checkpoint_probe.py` and `tools/live_save_gate.py` are described only through their package counterparts; their exact command-line flags are argparse wiring in `tools/` and are not reproduced here.

## Debugging doctrine, applied to CI

The same rule the gates enforce applies to the wiring around them: exit-code-only wiring is distrusted everywhere. A summary line with a denominator is required before a green is credited — in a gate verdict, in a control report, in a launcher mapping, and in CI output. When you are debugging, read the record in that order: exit code (which class am I in?), then `refusal_class` or `blocking_reasons` (which member?), then the inline denominators (what was actually examined?), then `declared_basis.notes` and `controls` (was anything silently shrunk, excused, or left unexercised?). A green without a denominator is not evidence; in this framework it is the failure shape.
