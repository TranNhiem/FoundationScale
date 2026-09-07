# Supported training workflows

This document expands README section 7. It covers what FoundationScale can put under
the gate plane today, what the launchers in `launchers/` actually are, and — with equal
prominence — what is not implemented. The ordering is deliberate: the framework's
founding rule is that a claim must carry its own coverage, and a workflow document that
quietly implied RL alignment or pre-training were supported would be the documentation
equivalent of `all([])` returning True.

## What "supported" means here

FoundationScale splits into two planes, and a workflow is "supported" to different
degrees on each:

| Plane | What it is | Portability |
|---|---|---|
| **Gate plane** | `foundationscale.gates`: the `Gate` contract, verdicts, coverage accounting, controls, registries | Python 3.10/3.11/3.12 (the CI matrix), OS-independent, stdlib-only at the core; checkpoint I/O needs the `[checkpoint]` extra (torch, safetensors, numpy) |
| **Launch plane** | The scripts under `launchers/`, including `launchers/fs_container_backend.sh` | Exercised on exactly one estate: Slurm and enroot container backends (`FS_BACKEND=auto\|slurm\|enroot`), one H100 tray. A hardened example, **not** a portable launcher |

A workflow is fully supported when it runs on the launch plane *with* the gate plane
wired into its lifecycle. A workflow the gate plane can verify but no launcher drives is
supported "via the package": you own the loop, and the gates run where you call them.

## Workflows via the package

Any workflow `transformers.Trainer` supports can run under the gate plane. This is a
structural statement, not a compatibility list: the gates operate on artifacts and
lifecycle events, not on trainer internals, so the support is model-agnostic by
construction. There is no per-architecture adapter layer to enumerate because there is
nothing architecture-specific in the contract a gate sees.

### The lifecycle a workflow attaches to

Gates run at defined points in a job's lifecycle, declared as members of
`Lifecycle`. These are the moments where a defect either gets caught or gets baked into
an artifact; each corresponds to a real incident class in the audited estate:

| `Lifecycle` member | Value | When gates fire |
|---|---|---|
| `Lifecycle.LAUNCH` | `"launch"` | Before any process starts: topology validity, config resolution, manifest write |
| `Lifecycle.BUILD` | `"build"` | After the model object is constructed, before weights load or training starts |
| `Lifecycle.DATA` | `"data"` | After the data pipeline renders a batch: supervision masks, template parity |
| `Lifecycle.STEP_ZERO` | `"step_zero"` | After the first optimizer step: objective identity, trust region, trainable set |
| `Lifecycle.FIRST_SAVE` | `"first_save"` | The first checkpoint of a run — the cheapest place to catch a save defect |
| `Lifecycle.SAVE` | `"save"` | Every subsequent checkpoint |
| `Lifecycle.EXPORT` | `"export"` | After a checkpoint is converted to a serving format |
| `Lifecycle.PROMOTE` | `"promote"` | Before an artifact is declared servable; the last gate before the blast radius |

A "workflow under the gate plane" means: the loop validates topology and profile at
`LAUNCH` first, and save gates fire on the result at `FIRST_SAVE` and `SAVE`. Export
and promotion are gated the same way when the workflow reaches them.

### Wiring gates into your own loop: single-context sweeps

The single-context path is `GateRegistry.run`: every gate registered for an event
receives the same context object unchanged.

```python
from foundationscale.gates.core import GateRegistry, Lifecycle

registry = GateRegistry()
registry.register(my_checkpoint_gate)

report = registry.run(
    Lifecycle.FIRST_SAVE,
    ctx,
    required=["checkpoint.expert_bytes"],   # ids that MUST run; absent ones block
)
report.raise_if_blocking()                  # raises GateBlocked if not ok
```

The `required` argument exists because a missing gate is itself the bug at most gate
sites — on the audited estate, the export byte check lived as a copy-pasted heredoc in
one launcher script and was simply absent from the other, which is how a truncated
export reached `rc=0`. Registry-based invocation makes gate coverage a property of the
*event*, not of whichever script happens to be running.

Three structural guarantees apply to any sweep:

* **An empty sweep blocks.** If no gate is registered for the event, `run` appends a
  framework-synthesized, blocking `VACUOUS` result reporting how many gates were
  registered in total. A sweep that ran nothing is not an all-clear. The sole opt-out
  is declaring the event gateless up front: `GateRegistry(event_allow_empty=[...])`.
  Even then the report remains `GateReport.is_vacuous`; only the block is lifted. The
  marker gate-id prefix (`registry.empty_sweep.`) is reserved — `register` refuses
  author gates under it, so the marker cannot be spoofed.
* **`required` adds the named-gate leg.** Ids in `required` that did not run land in
  `GateReport.missing` and block the whole report.
* **Nothing blocks silently.** `GateReport.ok` is true only when nothing blocks,
  nothing required is missing, the sweep is not vacuous (or was declared gateless),
  and at least one executed gate verified something. A report whose every result is
  `SKIP` — gates ran, and every one declined — is *not* ok: `allow_empty` does not
  pardon an all-SKIP sweep, because gates that were registered, wired and individually
  declined are not a gateless event.

For a process-wide default, `REGISTRY` is a shared `GateRegistry`, and the `register`
class decorator instantiates a gate and adds it:

```python
from foundationscale.gates.core import REGISTRY, register, Lifecycle

@register
class ExpertBytesGate(Gate):
    id = "checkpoint.expert_bytes"
    description = "Expert parameter bytes match the model's declared shape"
    events = (Lifecycle.FIRST_SAVE, Lifecycle.SAVE)
    ...
```

### Wiring gates into your own loop: typed, multi-context sweeps

A real training job registers gates from several context families — checkpoint,
objective, parity — whose contexts are not interchangeable. Broadcasting one object to
all of them dies inside `check` as a raw `TypeError`. `run_event` is the multi-context
counterpart: each gate declares a `context_type` class attribute, and the sweep
dispatches on it.

```python
from foundationscale.gates.core import REGISTRY, Lifecycle, run_event

report = run_event(
    REGISTRY,
    Lifecycle.SAVE,
    {CheckpointContext: ckpt_ctx, ObjectiveContext: obj_ctx},
    required=["checkpoint.expert_bytes"],
)
report.raise_if_blocking()
```

`run_event` turns wiring facts into verdicts — nothing in the dispatch path can return
PASS without the gate's `check` having run:

| Situation | Outcome |
|---|---|
| No supplied context matches the gate's declared type | Blocking `ERROR`: *"no context of type X supplied for gate Y — unwired, not healthy"* |
| Same, with `missing_ctx="report-skip"` | `SKIP` — an explicit, declared abstention; still never PASS. Any other `missing_ctx` value raises `ValueError` |
| Two supplied contexts both satisfy the declared type | Blocking `ERROR` — dispatch refuses to pick arbitrarily |
| Legacy gate (`context_type is None`) meets a typed context map | Blocking `ERROR` naming the fix: declare `context_type`, or use `GateRegistry.run` with a single context |
| A mapping value is a zero-argument callable | Invoked lazily, once per consuming gate, inside the sweep's `ERROR` conversion; a factory for an unconsumed type is never invoked |
| `gate_ids` selects an id not registered for this event | Lands in `missing` and blocks — a typo must not read as "all selected gates clear" |
| An id appears in both `exclude` and `required`/selection | Contradiction: lands in `missing` and blocks |
| Selection runs zero gates | Inherits the registry's empty-sweep rule: blocking `VACUOUS` unless the event was declared `event_allow_empty` |

A gate that accepts more than its declared type (a path, an adapter object) overrides
`coerce_context`. It must return `None` for unrecognized input — never raise
`TypeError` — so the dispatcher can produce the named "unwired, not healthy" ERROR
instead of an opaque crash. Other exceptions during adaptation propagate into the
sweep's ERROR conversion exactly like failures inside `check`.

Context resolution is exact-then-subclass: an exact type-key hit wins; a single
subclass hit is honoured; two or more are the ambiguous case above. A `Mapping` passed
to `run_event` is always read as a typed context map — to broadcast a mapping-shaped
context itself, use `GateRegistry.run`.

Gate failures never raise from `run_event`; they are verdicts. Bad sweep arguments do
raise.

## What a gate verdict does to your workflow

A gate returns one of seven verdicts. Two do not block; five do. The asymmetry is
intentional: it is much easier to accidentally produce a meaningless success than a
meaningless failure.

| Verdict | Blocks? | Meaning |
|---|---|---|
| `PASS` | no | Checked a non-vacuous, sufficient set of units, found no defect |
| `SKIP` | no | Explicitly declined to verify; requires a reason; reported |
| `FAIL` | yes | Found a defect |
| `VACUOUS` | yes | Reported no defect while inspecting **nothing** |
| `UNDERCOVERED` | yes | Examined fewer units than `expected`, without declaring a sample |
| `OVERCOVERED` | yes | Examined *more* units than the denominator declares exist |
| `ERROR` | yes | The gate itself raised — gates fail closed |

The consequences for a workflow author:

* **VACUOUS is the founding rule.** `all([])` is `True`, and a check that examined
  zero units must not pass. `Gate.ok` cannot return `PASS` with zero coverage,
  regardless of what the gate author writes inside `check` — it downgrades to
  `VACUOUS`, and `VACUOUS` blocks. An author asserting "nothing to check here" must
  use `Gate.skip` with a reason, which is recorded and surfaced.
* **The denominator binds in both directions.** "3 of 205 layers checked" blocks as
  `UNDERCOVERED` unless the gate declares itself a sample (`Coverage(sampled=True,
  sample_reason=...)`, surfaced in every rendering). "500 of 256 examined" blocks as
  `OVERCOVERED` with no declaration available: a count above the denominator is a
  contradiction (double-counted units, a superset sweep, or a stale `expected`), not
  a partial anything. The honest remediations are to correct the count, correct
  `expected`, or pass `expected=None` when no denominator is knowable. Do not
  fabricate a denominator to make a ratio look complete.
* **Exceptions are verdicts.** `Gate.run` converts any exception inside `check` —
  including a `check` that returns something other than a `GateResult` — into a
  blocking `ERROR` with a truncated traceback in evidence. An exception inside a gate
  is an incident: in the audited record, a reward-module import failure silently
  disabled a degeneracy veto, and a verifier exception counted as a pass.
* **Skips are priced by machines, not prose.** A `SKIP` carries an optional
  `abstention` field: `AbstentionKind.NOT_APPLICABLE` (the property does not exist in
  this run's declared scope — legitimate only behind a *positive* declaration such as
  an explicit `num_experts == 0`, never an absence of evidence) versus
  `AbstentionKind.NOT_ESTABLISHED` (the property may exist; the evidence could not
  settle it). Composites pricing a verified/applicable denominator must read that
  field, never the skip reason string. An undeclared kind (`None`) is charged exactly
  like `NOT_ESTABLISHED`: it never leaves a denominator. An abstention kind on a
  non-SKIP verdict is refused at construction — a pass cannot smuggle an
  inapplicability claim.
* **Blocking is enforced at the type level.** `GateReport.raise_if_blocking` raises
  `GateBlocked`; call it at every gate site that is allowed to stop the job.

### Reading a report

`GateReport.render()` produces a head line, per-gate lines, missing-gate lines, and a
"N gates ran of M registered" footer when the runner set the denominator. Two
rendering details matter when you read one in a log:

* The empty-sweep marker is excluded from the run count — one report never emits two
  disagreeing counts.
* An over-covered claim is never rendered neutrally: `"500/256 reward samples"` is
  suffixed `(over: checked exceeds expected — one of them is wrong)`, because the bare
  string reads as a typo rather than a defect. `"3/205"` stays bare: it is a believable
  statement that needs the verdict beside it.
* An ok report with skips says so in the head: `all clear (X of Y verified; Z declared
  SKIP)`. A genuinely gateless-but-declared report says `all clear (gateless by
  declaration: event_allow_empty)`. `GateReport.to_json()` carries the same state on
  the wire, including the top-level `unverified` flag, so a downstream consumer never
  sees a bare `"ok": true` over a partial sweep with the skips invisible.

## Controls: proof the gates themselves work

Every gate declares controls of both kinds, and `verify_controls` enforces this in CI:

* At least one `ControlKind.MUST_FIRE` — a deliberately defective input the gate must
  flag. Proof the detector *can* fire.
* At least one `ControlKind.MUST_PASS` — a known-good input producing its declared
  outcome (PASS by default, or an explicitly declared `Control.expect_skip` reason
  when the healthy input is genuinely unadjudicable — the live case is per-expert
  identity inside a stacked MoE tensor, which metadata cannot see). `expect_skip` is
  illegal on `MUST_FIRE`: a positive control the detector abstains on proves nothing.
* The MUST_PASS set must reach at least one real PASS in total — a gate that has never
  affirmatively accepted any input could block (or abstain) on everything, and
  detectors like that are the ones operators learn to route around.

```python
from foundationscale.gates.core import verify_controls

failures = verify_controls()          # over REGISTRY; pass gate_ids=[...] to scope
assert not failures, "\n".join(failures)
```

`verify_controls` also fails closed about itself — the same rule one level up:

| Self-check | Behaviour |
|---|---|
| A gate's `controls()` raises | Reported as a failure naming the gate and exception; never re-raised, never silently skipped |
| An id in `gate_ids` is not registered | Reported as a failure naming the id — a dropped typo must not read as "all controls held" |
| Zero gates targeted (empty registry, gate modules never imported, unmatched selection) | Reported as a failure: returning `[]` would be success over zero work |

An empty return list means every control produced its declared outcome *and* every
gate proved at least one affirmative healthy-input pass.

## Workflows via the launchers

`launchers/` ships two estate-parameterized reference workflows, exercised on the one
estate described above (Slurm and enroot backends, one H100 tray):

| Script | Workflow |
|---|---|
| `launchers/launch_g4e4b_fullft_1tray.sh` | One full fine-tune |
| `launchers/launch_g4e4b_lora_1tray.sh` | One LoRA fine-tune |

Both are parameterized through environment variables rather than hard-coded paths, and
both select their container backend through `FS_BACKEND=auto|slurm|enroot` in
`launchers/fs_container_backend.sh`.

These scripts are **reference material**: a concrete, hardened example of how a gated
job is launched on one real estate, not a portable launcher you point at an arbitrary
cluster. Treat them as the worked example to read when wiring the gate plane into your
own loop — the parts that generalize are the lifecycle wiring and the gate invocation,
not the estate assumptions.

## Hardware and platform reach of these workflows

The gate plane has been exercised on the CI matrix (Python 3.10, 3.11, 3.12) and is
OS-independent beyond its stdlib core and the `[checkpoint]` extra. The launch plane
has been exercised on the single estate named above.

Local machines, DGX Cloud / Lepton and Kubernetes appear on the project poster as
design targets from the audit's scaling work (`docs/deliverables/B2_scaling.md`). In
this repository they are **unmeasured**: nothing here has run on them. What would
measure them is the roadmap's structured-harness phases, not a claim in a README — and
that holds even if the launcher scripts *look* portable, because "parameterized by
environment variables" is not evidence of execution.

## Not implemented

Nothing in this section is hedged; these are absences in the tree, stated flatly:

* **RL / post-training alignment pipelines are not implemented.** There is no gate,
  launcher, loop or data path in this repository that runs one. They are roadmap items
  with falsification conditions in `docs/deliverables/`.
* **Pre-training recipes are not implemented.** Same status: roadmap, falsification
  conditions in `docs/deliverables/`, nothing in the tree.
* **The unified data contract is not implemented.** `Lifecycle.DATA` exists as an
  event you can hang your own data gates on, but there is no shipped contract defining
  a canonical batch/supervision representation across workflows.
* **Portable launching beyond the one H100 Slurm/enroot estate is not implemented.**
  The launchers are a worked example with `FS_BACKEND=auto|slurm|enroot`; extending the
  backend set is your integration work.
* **No shipped gate library is enumerated here.** The source material for this
  document specifies the gate *contract* (`Gate`, `Coverage`, `GateResult`,
  `GateReport`, `GateRegistry`, `run_event`, `verify_controls`) and the launchers; the
  concrete inventory of gates registered by default in `REGISTRY` for a given workflow
  is not stated in it and is therefore not claimed here. Discover it against the
  installed package — `verify_controls()`'s zero-target failure mode exists precisely
  because "gate modules never imported" is a shape of verified-nothing.
* **Exit codes and CLI entry points for gated workflows are not specified** in the
  source material. The blocking mechanism this document can truthfully describe is the
  Python-level one — `GateBlocked` from `GateReport.raise_if_blocking()` — and how a
  launcher maps a blocked report to a process exit status is not stated; do not infer
  one.
