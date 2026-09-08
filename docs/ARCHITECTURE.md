# Architecture

This document expands the layout sketched in README §5. It describes what each layer
is, why the ordering is load-bearing, and which invariants hold at each level. The
full L0–L6 design lives in `docs/deliverables/B1_architecture.md`; this chapter is the
orientation to what is in `src/foundationscale/` today.

## The stack, bottom to top

```
gate contract (stdlib-only core: Verdict, Gate, REGISTRY)
   ├── gate domains        checkpoint / objective / probe / example + fixtures
   ├── adjudication        composes gate verdicts into a run-level judgment
   ├── checkpoint + verify DCP metadata, parity checks
   ├── provenance          the run manifest — what was claimed, measured, emitted
   ├── topology            validates the declared parallel geometry
   ├── rl/                 the post-training contracts: batch, loss, policy pair
   └── train/              cli + loop: validate → delegate to Trainer → save gates
```

The arrows point upward from the most frozen surface to the thinnest. Everything
above the gate contract earns its position by *consuming* the contract, not by
redefining it.

## The gate contract: `src/foundationscale/gates/core.py`

A *gate* is a correctness check that runs at a defined point in a job's lifecycle and
can block. The founding rule of the framework lives here:

> A gate that inspected nothing did not pass. It returns VACUOUS, and VACUOUS blocks.

This is not a style preference. It is the literal shape of the incident the framework
is named after: a verification tool asked whether every expert tensor matched reported
`all_identity: True` on a corrupt artifact, because the expert tensors were absent, the
comparison set was empty, and `all([])` is `True`.

`core.py` is **pure stdlib, with zero dependencies**, and that is deliberate: the gate
plane must run on a bare login node, on hosts where torch is absent by design. The
same reason explains why `import foundationscale` stays torch-free at the package root
— every public name in `foundationscale/__init__.py` resolves lazily via a
module-level `__getattr__`, so nothing heavy loads until a name that needs it is
touched.

### Verdicts

A gate returns a `GateResult` carrying a `Verdict`, a `Coverage`, and evidence. The
asymmetry of the verdict set is intentional: it is much easier to accidentally
produce a meaningless success than a meaningless failure, so there are exactly two
non-blocking outcomes and five blocking ones.

| Verdict | Blocks | Meaning |
|---|---|---|
| `PASS` | no | Checked a non-vacuous, sufficient set of units and found no defect. |
| `FAIL` | yes | Found a defect. |
| `VACUOUS` | yes | Reported no defect while inspecting nothing. |
| `UNDERCOVERED` | yes | Examined fewer units than `expected`, without declaring a sample. |
| `OVERCOVERED` | yes | Examined MORE units than the denominator declares — the numerator contradicts the denominator. |
| `SKIP` | no | Explicitly declined to verify. Requires a reason; is reported. |
| `ERROR` | yes | The gate itself raised. Gates fail closed. |

Every `Verdict` exposes `blocking` (the decision used to stop a job) and `symbol`
(the short rendering used in report lines, e.g. `UNDER`, `OVER`).

### Coverage: a verdict is a claim about a denominator

Every result carries a `Coverage(checked, unit, expected=None, sampled=False,
sample_reason="")`. The denominator binds in **both directions**:

* `checked == 0` → `VACUOUS`. Always. `Coverage.none(unit)` is the explicit
  constructor for this state.
* `checked < expected` → `UNDERCOVERED`, unless the gate declared
  `sampled=True` **with** a non-empty `sample_reason` (`Coverage.__post_init__`
  raises `ValueError` otherwise). A sample is a declared, reasoned choice to
  examine fewer.
* `checked > expected` → `OVERCOVERED`, with **no** author pardon. "500 examined
  out of 256 expected" is a contradiction — double-counted units, a superset
  sweep, or a stale `expected`. No declaration makes 500 a subset of 256, so
  there is nothing to bless; the remediations are to fix the count, correct
  `expected`, or pass `expected=None` when no denominator is knowable.

`Coverage.fraction` may return a value above 1.0 — that overage **is** the signal,
and it is deliberately not capped. `Coverage.__str__` annotates an over-covered
ratio in place ("500/256 reward samples" printed bare reads as a typo, not a
defect), because the coverage column may render where no verdict sits beside it.

### The `Gate` base class and the coverage rule

Authors subclass `Gate`, set `id`, `description`, and `events` (enforced by
`__init_subclass__`; a missing or malformed attribute is a `TypeError` at class
definition time), implement `check(ctx)`, and return results **only** through the
constructors `ok()`, `fail()`, and `skip()`. The rule is enforced by the
framework, not by the author:

* `Gate.ok()` downgrades to `VACUOUS`, `UNDERCOVERED`, or `OVERCOVERED` based on
  the `Coverage` it is handed — no matter what the author writes. This is the
  point.
* `Gate.skip(reason)` requires a non-blank reason and records zero coverage.
* `Gate.run(ctx)` wraps `check` with timing and converts **any** exception into
  `ERROR` — deliberately broad, because a verifier exception must never count as
  a pass. A `check()` that returns something that is not a `GateResult` (a bare
  `True` type-checks clean) is converted to `ERROR` the same way. Keep that
  branch.

### Abstentions are data, not prose

`SKIP` is two different statements wearing one verdict, and composites must price
them differently. The split lives in `GateResult.abstention` as an
`AbstentionKind`:

| Kind | Meaning | Priced as |
|---|---|---|
| `NOT_APPLICABLE` | The property does not exist in this run's **declared** scope (a positive declaration like `num_experts == 0`, never an absence of evidence). | May be removed from a composite's applicable denominator; must be named; never counts as verified. |
| `NOT_ESTABLISHED` | The property exists (or may); the evidence cannot settle it — e.g. per-expert identity inside a stacked MoE tensor. | Stays in **every** denominator. "Could not check" is charged against the sweep. |

`abstention=None` on a SKIP is **not** a synonym for either member: it means the
call site was never audited for the distinction and prices exactly like
`NOT_ESTABLISHED`. `GateResult.__post_init__` refuses an abstention kind on any
non-SKIP verdict — a pass cannot smuggle an inapplicability claim. Aggregators
read that field, never the prose `detail` reason.

### Controls: every gate must prove its detector fires

A gate declares `controls()` — executable fixtures, not documentation — of
**both** kinds, enforced in CI by `verify_controls` (exposed at the package root
as `foundationscale.verify_controls`):

* `ControlKind.MUST_FIRE` — a deliberately defective input the gate *must*
  block on. Proof the detector can fire.
* `ControlKind.MUST_PASS` — a known-good input that must produce its **declared**
  outcome: `PASS` by default, or an explicit `Control.expect_skip` reason when
  the healthy fixture is genuinely unadjudicable. `expect_skip` is **illegal**
  on a `MUST_FIRE` control (refused at control construction): a positive
  control the detector is expected to abstain on proves nothing.

`verify_controls` applies the framework's doctrine to itself, so that "all
controls held" can never be asserted over zero work. It reports a named failure
when:

1. A gate declares no `MUST_FIRE` control, or no `MUST_PASS` control — the
   zero-trip guard. A check nested in a data-driven loop cannot see the absence
   of its own trip; a gate shipping only `MUST_FIRE` controls had its
   healthy-input behaviour verified zero times while `verify_controls` returned
   "all controls held" for what could be a detector that blocks on everything.
2. A `MUST_FIRE` control does not block, or a `MUST_PASS` control blocks.
3. A `MUST_PASS` control abstains without an `expect_skip` declaration — or
   declares `expect_skip` and then reaches `PASS` anyway (the declaration is
   stale; delete it).
4. A gate's entire `MUST_PASS` set reaches zero real `PASS` verdicts. Every
   abstention may be individually honest and declared while the set certifies
   nothing — the zero-trip guard for the abstention lane.
5. `gate.controls()` raises, a `gate_ids` entry is not registered, or **zero
   gates are targeted at all** — the verifier-layer `all([])`.

An empty return list means every control produced its declared outcome and every
gate proved at least one affirmative healthy-input pass.

### Lifecycle events

`Lifecycle` names the points where a defect either gets caught or gets baked into
an artifact: `LAUNCH`, `BUILD`, `DATA`, `STEP_ZERO`, `FIRST_SAVE`, `SAVE`,
`EXPORT`, `PROMOTE`. `PROMOTE` is the last gate before the blast radius.

### Registries and reports: the sweep-level `all([])`

`GateRegistry` holds gates and runs them by event — invocation is a property of
the *event*, not of whichever launcher script happens to be running. The module
exposes the process-wide default `REGISTRY` and a `register` class decorator.

The empty-comparison bug is closed at the sweep level too, not just per gate:

| Shape | Mechanism |
|---|---|
| Zero gates ran for an event | `GateRegistry.run` appends a synthesized blocking `VACUOUS` result under the reserved id prefix `registry.empty_sweep.<event>`, with the registered-gate population in evidence. `register()` refuses author gates under that prefix, so the marker cannot be spoofed. |
| Required gates did not run | `required=` ids that never ran land in `GateReport.missing` and block the whole report. |
| A report over zero results | `GateReport.is_vacuous` is true over an empty results tuple or an all-marker report; `GateReport.ok` blocks on it. The only pardon is `GateRegistry(event_allow_empty=...)` — a declaration that this event is legitimately gateless, which sets `GateReport.allow_empty=True`. |
| Every executed gate abstained | `GateReport.is_unverified` wedges `ok` even when nothing blocked and nothing is missing: `PASS` and `FAIL` are the only post-examination verdicts, and a sweep of nothing but declared SKIPs examined zero units. `allow_empty` does **not** pardon this clause — the declaration is "this event is gateless", and a report carrying SKIP results is not gateless. |
| A hand-built report | `allow_empty` defaults to `False`, so a filtered, merged, or never-invoked report fails closed on the type itself. |

`GateReport.render()` computes its run count once so the headline and footer
cannot disagree ("1 run" above "0 gates ran" is the audit's complaint in
miniature), and an `ok` headline over a gateless-by-declaration sweep says so on
its face. `raise_if_blocking()` raises `GateBlocked` — call it at every gate
site that is allowed to stop the job.

### Multi-context dispatch: `run_event`

A training job registers gates from several context families — checkpoint,
objective, parity — whose contexts are not interchangeable. `run_event(registry,
event, contexts, ...)` dispatches on the gate class attribute `context_type`:

* A gate declares `context_type = SomeCtx` and `run_event` is handed
  `{SomeCtx: ctx}` (or a subclass key, uniquely — ambiguity is a blocking ERROR,
  because dispatch never picks arbitrarily).
* No matching context is a **blocking ERROR** — "unwired, not healthy" — unless
  the caller passes `missing_ctx="report-skip"`, an explicit declaration. Any
  other `missing_ctx` value raises `ValueError`: a misspelled abstention mode
  must not read as consent.
* A legacy gate (`context_type=None`) meeting a typed map refuses to guess which
  entry it wants: blocking ERROR naming the fix. With a bare single context it
  behaves exactly like `GateRegistry.run`.
* `coerce_context(ctx)` lets a gate accept more than its declared type; it must
  return `None` to refuse, never raise `TypeError`.
* Mapping values may be zero-argument factories, invoked lazily once per
  consuming gate, inside the same ERROR conversion as `check()`. A raising
  factory is an ERROR verdict with a traceback; a factory no gate consumes is
  never invoked.
* `gate_ids=` and `exclude=` select within the sweep; a selected-or-required id
  that does not run lands in `missing`. A selection that empties the sweep
  inherits the blocking-VACUOUS empty-sweep rule.

Nothing in `run_event` can produce `PASS` without the gate's `check` having run.

## The gate plane above the contract

Reference gates and fixtures live in `src/foundationscale/gates/example.py`,
`src/foundationscale/gates/fixtures.py`,
`src/foundationscale/gates/checkpoint_gates.py`, and
`src/foundationscale/gates/objective_gates.py`; domains are checkpoint,
objective, probe, and example-plus-fixtures per the layer diagram. The
adjudication layer (`src/foundationscale/gates/adjudication.py`) composes gate
verdicts into a run-level judgment — this is where `AbstentionKind` earns its
keep, because only a machine-readable kind lets a composite price a verified /
applicable denominator correctly. The load-bearing rule for every layer above
the contract: **a layer may only emit a verdict it earned** — a composite
propagates its children's coverage rather than minting a pass they never
produced.

Around that sit the support modules named in README §1:

* `src/foundationscale/checkpoint/dcp.py` — checkpoint I/O over DCP metadata;
  its `CheckpointFormatError` is exposed at the package root.
* `src/foundationscale/verify/parity.py` — parity checking.
* `src/foundationscale/provenance/manifest.py` — the run manifest: what was
  claimed, measured, and emitted. `Topology` and `TopologyConsistency` are
  re-exported from `foundationscale.provenance` at the package root.
* `src/foundationscale/models/adapters.py` — `select_adapter` and
  `AdapterRefusal` (re-exported at the root).

## The RL plane: `src/foundationscale/rl/`

The post-training contracts. Three modules, split so that a contract and an
implementation of it are never the same file:

* `interfaces.py` — the contracts only. `ExperienceBatch` (the data contract),
  `LossFn`/`ForwardFn` (the protocols), `LossOutput` (what one measured step
  produced), `LossDeclaration` (what a loss *says* it will produce), the three
  refusals `BatchRefusal`/`SupervisionRefusal`/`LossConfigRefusal`, and
  `build_objective_gate_context`.
* `losses.py` — `SFTLoss` and `DPOLoss`.
* `policy.py` — `PolicyPair` and `PolicyRoleRefusal`: the policy/reference role
  split DPO needs.

`build_objective_gate_context` is the whole reason this plane sits under the
gate contract rather than beside it. It turns one measured step into an
`ObjectiveGateContext`, so the objective gates
(`src/foundationscale/gates/objective_gates.py`) adjudicate an RL step the same
way they adjudicate any other — `LossComponentCoverageGate` over the declared
components, `DiagnosticMetricGate` over the declared metrics. That is what makes
a loss's `declaration()` load-bearing instead of documentation: **a component
that is computed and not declared, or declared and not computed, is a RED.**
`SFTLoss` declares one component and no metric, and the empty metric tuple is a
declared abstention — `DiagnosticMetricGate` answers "nothing declared, nothing
observed" with a SKIP, which keeps the absence inside the sweep's denominator
instead of reading as coverage.

Two consequences of that rule are visible in the code. `DPOLoss` derives both
its declaration and its computation of the auxiliary SFT term from the one field
`sft_weight`, because any second source of truth could produce exactly the
computed-but-undeclared state the gate blocks on. And `declaration()` is for the
run manifest, not for the gate context: building the context *from* the
declaration would compare what is in force against what is in force.

The public import path is the subpackage — `from foundationscale.rl import
DPOLoss` — not the package root, so these names are not in `_EXPORTS` and sit
outside `tests/packaging/test_front_door.py`'s denominator. They are in the
coverage-floor and mutation-scope denominators (`checks/coverage_floor.py`,
`checks/mutation_scope.py`).

The contracts named in the Phase 3 design that are *not* here — `Algorithm`,
`RolloutSource`, `AdvantageFn`, `WeightSync` — are later stages and are
deliberately absent. Nothing in this plane has been benchmarked against the
reference implementation; `validation_campaigns/nemo_rl_baseline/PHASE3_DESIGN.md`
section 10 states what is not established.

## The training plane: `src/foundationscale/train/`

The entry point is deliberately thin. `cli.py` and `loop.py` validate the
declared topology (`src/foundationscale/topology.py`, with `ClusterProfile`
re-exported at the root) and profile, delegate the step to
`transformers.Trainer`, and run the save gates over the result. Training types
surfaced at the root are `TrainConfig` and `FoundationScaleSaveGate`; the
function `train` itself is imported as `from foundationscale.train import
train`, not from the root (the collision class is gated by
`tests/packaging/test_front_door.py`).

### The exit-code contract

`foundationscale/__init__.py` re-exports the four exit codes as the public
contract: `EXIT_PASS` (0), `EXIT_RED` (5), `EXIT_UNMEASURED` (95),
`EXIT_REFUSE` (96). Doctrine in one line, from the package docstring: *a
verdict is a claim about a DENOMINATOR. Gates fail closed, and "nothing was
measured" is a declared state (`EXIT_UNMEASURED`), never a pass.*

## The import surface itself is gated

`foundationscale/__init__.py` holds a single `_EXPORTS` mapping of public name
to home module, exposing `Coverage`, `GateRegistry`, `GateReport`, `REGISTRY`,
`Verdict`, `verify_controls`, the four exit codes, `FoundationScaleSaveGate`,
`TrainConfig`, `ClusterProfile`, `Topology`, `TopologyConsistency`,
`AdapterRefusal`, `CheckpointFormatError`, `select_adapter`, and
`__version__`. The mapping *is* the public surface:
`tests/packaging/test_front_door.py` enumerates it, so a name added without a
working home fails a test rather than an import at a user's site.

## Around the package

`src/` is 22776 LOC across 33 files; `tests/` adds 33569 `.py` LOC (its
conftest carries the skip guard). Beside the package:

| Tree | Contents |
|---|---|
| `tools/` | 9533 Python LOC of CLIs over the package: `emit_run_manifest`, `live_save_gate`, `real_checkpoint_probe`, `preflight/`, `mutate`, `census`. |
| `checks/` | Standalone repository gates: countables drift, packaging reachability, `bash -lc` sweep, workflow YAML audit. |
| `launchers/` | The estate launch plane (10716 shell LOC) plus two bash contract suites and Python helpers (lora target census, peft override replay) — 1615 Python LOC. |
| `validation_campaigns/h100_validation/` | The experimental H100 harness (33385 Python, 6414 shell LOC): build script, `gate_*.py`, `patch_*.py`, its own tests, and the published `h100/` deliverables. **Evidence campaigns, not framework code — read as lab notebooks.** |
| `docs/` | `DECISIONS.md`, `deliverables/` (A1–D, including `B1_architecture.md`), `SELF_AUDIT.md`. |
| `.github/workflows/` | CI: check / controls / launchers / mutation shards. |

Repo-wide: 144885 git-tracked `.py`/`.sh`/`.md` lines.

## Known gaps

This document expands only what the README sections and `core.py` source
support. The following are named in the project layout but their internal APIs
are not documented here, and no names are invented to cover them:

* The adjudication composition rule — beyond the README's statement that
  `adjudication.py` composes gate verdicts into a run-level judgment — is not
  specified in the available material; the function or class names it exposes
  are not stated anywhere in the sources at hand.
* The same applies to the concrete gate ids inside `checkpoint_gates.py`,
  `objective_gates.py`, `example.py`, and `fixtures.py`, and to the public
  names of `checkpoint/dcp.py` beyond `CheckpointFormatError`,
  `verify/parity.py`, `provenance/manifest.py`, and `topology.py` beyond
  `ClusterProfile`. Consult each module directly.
* The `tools/` CLIs are enumerated by directory in the README but their
  command-line spellings are not given; no invocation examples appear here
  because none exist in the source material.

Where the full design intent matters beyond what ships in `src/`, read
`docs/deliverables/B1_architecture.md`. Where a decision needs a rationale, read
`docs/DECISIONS.md`.
