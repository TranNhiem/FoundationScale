# Writing a Custom Gate

This is the chapter for the extension point the README can only gesture at: a gate
over your own workload, written against the base class in
`foundationscale.gates.core`, proven by its own fixtures, and exercised by
`foundationscale-controls` exactly like a built-in gate. The reference implementation
is `src/foundationscale/gates/example.py`; read it alongside this document.

## What a gate is

A gate is a correctness check that runs at a defined point in a job's lifecycle and
can block. Three things make a class into a gate:

1. **A verdict** — and `PASS` is only one of two non-blocking outcomes.
2. **Coverage** — how many units the gate actually examined, and out of how many.
3. **Controls** — at least one deliberately broken input the gate *must* flag, plus
   at least one known-good input it must accept. `verify_controls` enforces both in
   CI.

The founding rule of the framework is that a claim must carry its own coverage:
`all([])` is `True`, and a check that examined zero units reports `VACUOUS`, not
`PASS`. The rule is enforced by the framework, not by you. `Gate.ok` cannot return
`PASS` with zero coverage no matter what you write; it downgrades. Your job as an
author is a smaller one than it looks: count honestly, route every result through
the provided constructors, and ship fixtures that could prove you wrong.

## The three class attributes

A concrete gate sets three class attributes. `__init_subclass__` rejects any
subclass missing a non-empty `id`, `description`, or `events`, so these are not
conventions — the class will not defined at all without them.

| Attribute | Type | Purpose |
|---|---|---|
| `id` | `str` | Stable identifier. Appears in every rendered and serialized result. The prefix `registry.empty_sweep.` is reserved for the framework's own empty-sweep markers; `GateRegistry.register` refuses author gates under it. |
| `description` | `str` | What defect class this gate exists to catch, stated as a claim about the artifact. |
| `events` | `tuple[Lifecycle, ...]` | The lifecycle points where this gate runs. Must literally be a tuple of `Lifecycle` members. |

A fourth attribute is optional but load-bearing:

| Attribute | Type | Purpose |
|---|---|---|
| `context_type` | `type \| None` | The concrete context this gate consumes. Default `None` keeps the legacy single-context broadcast of `GateRegistry.run`. Declaring a real type lets `run_event` mix your gate into a multi-context sweep — and turns "the integrator never wired my context" from a raw `TypeError` inside `check` into a named, blocking ERROR identifying the missing type. |

The `Lifecycle` enum defines the points at which gates run — `LAUNCH`, `BUILD`,
`DATA`, `STEP_ZERO`, `FIRST_SAVE`, `SAVE`, `EXPORT`, `PROMOTE`. Each corresponds to
a moment where a defect either gets caught or gets baked into an artifact. Pick the
cheapest event where your defect class is still catchable: the README's example
gate runs at `FIRST_SAVE` precisely because the first checkpoint of a run is the
cheapest place to catch a save defect — and again at every subsequent `SAVE`.

## Writing `check()`

`check(self, ctx)` is the only method that runs against real inputs. The contract
on its return value is absolute: return through `self.ok(...)`, `self.fail(...)`,
or `self.skip(...)`. Constructing `GateResult` by hand bypasses the coverage rule
that the class exists to close.

Every result carries a `Coverage`:

```python
Coverage(checked=len(units_examined), unit="expert tensors",
         expected=declared_count_from_config)
```

The fields, per the `Coverage` docstring:

| Field | Rules |
|---|---|
| `checked` | Units actually examined. Zero means vacuous, *always*. Must not be negative — `__post_init__` raises `ValueError`. |
| `unit` | What is being counted: plural, lowercase, reads naturally in rendered output (`"experts"`, `"tensors"`, `"export dirs"`). |
| `expected` | Units that *should* have been examined, when knowable. Leave `None` when the denominator genuinely is not known; do not fabricate one. When given, it binds in **both** directions: short of it blocks as `UNDERCOVERED` (unless declared a sample), above it blocks as `OVERCOVERED`. If the true population moved, update `expected` — do not ship a contradicted ratio. |
| `sampled` | Declares deliberately partial coverage, converting would-be `UNDERCOVERED` into a pass. Requires `sample_reason`, and is surfaced in every rendering. Pardons undercoverage only — it has no effect on overage. |
| `sample_reason` | Why partial coverage is acceptable. A blank value with `sampled=True` raises `ValueError` at construction. |

`expected` must come from outside the artifact being checked — a model config, a
manifest — never by counting units in the artifact itself. The reference gate's
`ExpertCheckContext` docstring is blunt about why: the corrupt artifact in the
audited incident "contained" exactly as many expert keys as its bogus index claimed
it should.

### What `ok()` can actually return

`self.ok(detail, coverage, evidence=...)` does not always produce `PASS`. Read the
downgrade chain as the framework's half of the contract:

| Coverage state | `ok()` returns | Pardon available |
|---|---|---|
| `checked == 0` | `VACUOUS` | None |
| `checked < expected`, not sampled | `UNDERCOVERED` | `sampled=True` + `sample_reason` |
| `checked > expected` | `OVERCOVERED` | **None** — a count above the denominator is a contradiction, not a partial anything |
| otherwise | `PASS` | — |

The `OVERCOVERED` branch exists because "500 of 256 examined" is as false a claim
as "3 of 205 checked": the claim contradicts its own denominator. There is
deliberately no author pardon. The honest remediations are: correct the count,
correct `expected` to the true population, or pass `expected=None` when no
denominator is knowable.

`self.fail(detail, coverage, evidence=...)` always blocks; coverage is recorded
but never softens it.

`self.skip(reason, kind=...)` declares the gate non-verifying on this input. The
reason is mandatory — an empty reason raises `ValueError`. The `kind` must be an
`AbstentionKind` or `None`:

| Kind | Meaning | Pricing |
|---|---|---|
| `AbstentionKind.NOT_APPLICABLE` | The property does not exist in this run's declared scope — legitimate only behind a **positive** declaration (an explicit `num_experts == 0`), never an absence of evidence. | A composite may remove the gate from its applicable denominator, must name it, must never count it as verified. |
| `AbstentionKind.NOT_ESTABLISHED` | The property may exist; the evidence could not settle it (e.g. per-expert identity inside a stacked tensor is metadata-invisible). | Stays in every denominator. |
| `None` | Not a synonym for either member: records that the call site was never audited for the distinction. Consumers price it exactly like `NOT_ESTABLISHED`. | Stays in every denominator. |

### Exceptions are verdicts

You do not need to defensively catch your own bugs. `Gate.run` invokes `check` with
timing and converts any raised exception into a blocking `ERROR` verdict carrying
the exception type and a traceback in evidence. Gates fail closed; an exception
inside a gate can never read as success. The same conversion applies if `check`
returns something that is not a `GateResult` — returning `True` type-checks no
better in practice than a crash, and both become ERROR.

## Structuring the context

The reference gate wraps everything it needs from a checkpoint into a small frozen
dataclass, `ExpertCheckContext`: the gate never opens a file or imports a
framework. Whoever owns the reader adapts their format into the context shape. This
indirection is deliberate: checks that knew their file format got copy-pasted per
script in the audited estate and quietly omitted from one of them; checks that know
an interface run identically at `FIRST_SAVE`, `SAVE`, and anywhere else the context
can be built.

Follow that pattern:

- Put exactly what `check` needs in the context — and nothing more.
- Never derive denominators from the artifact itself (see above).
- Declare `context_type = YourContext` on the gate class. Without it the gate is
  legacy: broadcast whatever single context `GateRegistry.run` is holding, and in a
  typed `run_event` sweep it refuses to guess, producing a blocking ERROR that
  names the fix.

## Declaring fixtures: `controls()`

A gate with no control is not a gate; `verify_controls` fails it in CI. Controls
are executable, not documentation.

Each `Control` names a fixture, a kind, and a zero-argument `make_ctx` callable
(called fresh per run; it may create tmp files), plus a one-line `note` describing
the defect injected. The kinds:

- **`ControlKind.MUST_FIRE`** — a deliberately defective input. If the gate does
  not block on it, the gate is broken and must not be trusted on real inputs. This
  is the executable form of the review rule: *every claim that something does not
  exist must name the positive control proving its detector could have fired.*
- **`ControlKind.MUST_PASS`** — a known-good input. Guards against a gate that
  blocks on everything, which is just as useless and tends to get disabled. The
  default expectation is an affirmative `PASS`, not merely "does not block".

The README's summary — one `MUST_FIRE` your gate must block, one `MUST_PASS` so a
gate that blocks everything is also caught — is the floor, not the ceiling. The
reference gate ships one `MUST_FIRE` per defect class (three of them), plus
`MUST_PASS` fixtures in both verdict directions it can honestly produce.

### `expect_skip`: the declared abstention lane

A known-healthy fixture that the gate genuinely cannot adjudicate — the canonical
instance is per-expert identity inside a stacked MoE tensor, which is
metadata-invisible — declares its expected abstention up front:

```python
Control(
    name="dense-model",
    kind=ControlKind.MUST_PASS,
    make_ctx=...,
    note="...the gate must take its stated NOT_APPLICABLE door...",
    expect_skip="<why this fixture is genuinely outside the gate's scope>",
)
```

The declaration is checked against reality in **both directions**:

- Abstain without declaring → the control fails (`verify_controls` names it).
- Declare and then reach `PASS` anyway → the control also fails, because the gate
  demonstrably can adjudicate the fixture and the declaration has become a stale
  claim.

`expect_skip` is illegal on `MUST_FIRE`: a deliberately defective input the
detector is expected to abstain on never fired, so nothing is proven. The
combination is refused at `Control` construction, not priced at run time. A blank
`expect_skip` string is also a construction error — an unreasoned exemption is the
exact defect the field exists to remove.

### What `verify_controls` enforces

Run it — via `foundationscale-controls`, as the README notes — and it enforces:

1. Every gate declares controls of **both** kinds. A gate with only `MUST_FIRE`
   controls had its healthy-input behaviour verified zero times while every
   control "held".
2. Each `MUST_FIRE` control actually makes the gate block.
3. Each `MUST_PASS` control produces its declared outcome (`PASS` by default, or
   the declared `expect_skip` abstention).
4. Each gate's `MUST_PASS` set reaches **at least one real PASS** in total. Every
   individual abstention may be honest and declared while the set still certifies
   nothing: a gate that has never affirmatively accepted any input could block —
   or abstain — on everything and return green.

It also enforces three things about itself:

- A gate whose `controls()` raises is reported as a failure naming the gate and
  the exception — not re-raised, not skipped.
- An id in `gate_ids` that is not registered is reported as a failure naming the
  id — a dropped id would verify nothing and return "all controls held".
- A call targeting zero gates (empty registry, gate modules never imported, a
  selection that matched nothing) is itself reported as a failure. Returning an
  empty list there would be "all controls held" over zero controls.

An empty return means every control produced its declared outcome and every gate
proved at least one affirmative healthy-input pass.

## Registration

Importing registers the gate. The mechanism is the `register` class decorator,
which instantiates the class and adds it to the process-wide `REGISTRY`:

```python
from .core import Gate, Lifecycle, register

@register
class MyGate(Gate):
    id = "checkpoint.my_gate"
    description = "..."
    events = (Lifecycle.FIRST_SAVE, Lifecycle.SAVE)
```

Registration at import time is doctrine, stated in the reference gate's
docstring: a gate that requires a manual registration step is a gate absent from
exactly the run where it would have fired. `foundationscale-controls` exercises
whatever import brought into `REGISTRY`, like any built-in gate.

Registry behaviour worth knowing:

- Duplicate `id` → `ValueError` from `GateRegistry.register`.
- A gate `id` starting with `registry.empty_sweep.` → `ValueError`; the prefix is
  reserved so the framework's empty-sweep marker cannot be spoofed.

There is no setuptools entry-point discovery mechanism. Registration is by import,
full stop — your module must actually be imported by the harness or job that runs
gates. Whether that should grow into an entry-point group is an open design
question (see `docs/EXTENSION_POINTS.md`). Anything advertised as a plugin API
wider than the gate contract does not exist today.

## How your gate runs

Two runners exist, and which one reads your gate depends on `context_type`:

- `GateRegistry.run(event, ctx, required=...)` broadcasts one context object to
  every gate registered for the event. `required=[...]` names gate ids the caller
  asserts must run; any not registered for the event land in `GateReport.missing`
  and block.
- `run_event(registry, event, contexts, ...)` is the multi-context counterpart:
  it takes a `{context_type: context}` mapping and hands each gate the context it
  declared.

Dispatch facts under `run_event`, each a blocking ERROR unless stated:

| Situation | Outcome |
|---|---|
| No context of the declared type supplied | Blocking ERROR: "unwired, not healthy" — unless the caller declared `missing_ctx="report-skip"`, which surfaces the gate as SKIP instead. Any other `missing_ctx` value raises `ValueError`. |
| Two supplied contexts both satisfy the declared type | Blocking ERROR: dispatch never picks arbitrarily. |
| A legacy gate (no `context_type`) meets a typed context map | Blocking ERROR naming the fix: declare `context_type`, or use `GateRegistry.run`. |
| Mapping value is a zero-argument callable | Invoked lazily, once per consuming gate, inside the sweep's ERROR conversion — a raising factory is an ERROR verdict, never an escape. A factory for a type no gate consumes is never invoked. |

Your gate cannot be handed a context it cannot read, but there is one path where
gate-author code joins the adaptation: override `coerce_context(ctx)` to adapt a
foreign (bare, non-mapping) context to your declared type. Return `None` for
inputs you do not recognise — the dispatcher turns that into the named, blocking
"unwired, not healthy" ERROR. Never raise `TypeError` from `coerce_context`; that
reproduces the opaque failure the hook exists to replace. Other exceptions (I/O
while adapting, a lazy import) may propagate; they land in the sweep's ERROR
conversion exactly like a failure inside `check`.

Both runners apply the sweep-level vacuity rule: a sweep that executes zero gates
produces a blocking `VACUOUS` report (a synthesized result whose id carries the
`registry.empty_sweep.` prefix), with your event and the registered-gate count in
its evidence. The single opt-out is constructing the registry with
`GateRegistry(event_allow_empty=[Lifecycle.X, ...])`. An allowed-empty report is
still `GateReport.is_vacuous`; only the block is lifted — and the lifted headline
says so ("gateless by declaration: event_allow_empty").

## The report you shipped into

A caller asserting success goes through `GateReport.ok`, and there is no path from
your gate's PASS to the report's all-clear that does not re-audit the sweep:

- `report.ok` is true only when nothing blocks, no required gate is missing, the
  report is not vacuous (or the event was declared `allow_empty`), **and** the
  report is not unverified (`is_unverified`: gates ran, and not one of them
  returned `PASS` or `FAIL` — a sweep whose every result is a declared, reasoned
  SKIP examined zero units and must say so).
- `report.raise_if_blocking()` converts a not-ok report into `GateBlocked`. Its
  docstring is an instruction, not a description: call this at every gate site
  that is allowed to stop the job.
- `report.registered` carries the sweep-level denominator; `report.render()` and
  `report.to_json(...)` keep `unverified`, `missing`, `registered`, and
  `allow_empty` visible on the wire so a downstream consumer never has to re-derive
  why a zero-verification sweep was not ok.

## Porting the reference gate

`src/foundationscale/gates/example.py` is the worked example to copy, and its
structural choices each answer an author question before it is asked:

- **Fingerprinting is injectable.** `ExpertAliasGate` takes a `FingerprintFn`
  (default: sha256 of raw bytes) so the gate and, more importantly, its controls
  never need torch or a real reader. Your controls should be cheap to construct;
  a fixture that takes minutes gets bypassed.
- **The fingerprint must compare content, not identity.** Comparing pointers, ids
  or tensor objects is the exact bug (a replicated view compares equal to itself)
  the gate exists to catch. The reference went further and nearly shipped a
  subtler version: folding the expert index into the hash would make every expert
  trivially unique and pass the exact artifact the gate exists to catch. The
  `MUST_FIRE` control caught it on the first run. Except controls to find the bugs
  in your gate, not to bless it.
- **Cheap signature before expensive evidence.** The gate checks the name
  signature (keys with a trailing local index) from the key set alone, before a
  single tensor byte is read — and when it fires, content comparison is impossible
  anyway.
- **Denominator hygiene before verdict.** A `declared_expert_count` that merely
  *compares* equal to zero (`False`, `0.0`, `0j` — Python laundered `False == 0`
  into truth) is a malformed denominator. The gate blocks it *before* any `== 0`
  test, routed through `self.ok(...)` over zero coverage so the framework renders
  it `VACUOUS` with the raw value named. The `malformed-dense-count-bool` control
  (`MUST_FIRE`) proves a boolean zero cannot buy the dense abstention.
- **The empty case is deliberately *not* special-cased.** On a declared-MoE model,
  an empty expert set calls `self.ok` with zero coverage and the contract
  downgrades it to `VACUOUS`. The `empty-expert-set` control exists to prove
  nobody "helpfully" bypasses that. The one exception — the positive dense
  declaration, corroborated by the artifact census (0 declared and 0 present) —
  takes the declared `NOT_APPLICABLE` skip, and the `dense-model` control proves
  that door is taken.

Copy the shape: context dataclass, injectable primitives, both-kinds fixtures,
registration by decorator, no special case for emptiness.

## What this document does not cover

The material above is the complete boundary of the extension point. The following
things are not specified anywhere in it:

- **How `foundationscale-controls` is invoked** (its command-line surface), beyond
  that it runs your registered gates' fixtures like any built-in gate: not stated
  in the sources behind this document.
- **How built-in fixtures are constructed** beyond what `example.py` uses:
  `make_aliased_experts`, `make_empty_experts`, `make_healthy_experts`, and
  `make_local_name_experts` are imported from `.fixtures` and are checkpoint-gate
  machinery, not a public fixture-authoring API. Write your own `make_ctx`
  callables.
- **Adapter plumbing from real artifacts** to your context type.
  `ExpertCheckContext.from_expert_set` exists only to adapt synthetic control
  fixtures; the docstring states plainly that production callers write their own
  adapter. There is no framework-supplied reader interface.
- **Any plugin discovery mechanism beyond import**: there is no setuptools
  entry-point group, and whether one should exist is an open design question. The
  boundary is the gate contract and nothing wider.
