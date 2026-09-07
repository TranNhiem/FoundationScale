# Design Principles

This document is the chapter behind the five bullets in the README's "Core design
principles" section. Each principle below states the rule, the machinery in
`src/foundationscale/gates/core.py` that enforces it, and the failure in the audited
estate that the rule exists to prevent. The ordering is deliberate: every later
principle is the same rule restated one level of abstraction up.

## 1. A verdict is a claim about coverage

`PASS` means exactly one thing: *examined N units, found no defect*. A `GateResult`
never carries a verdict without a `Coverage`, and an unqualified count is not a fact.
Three properties define a `Coverage`:

| Field | Meaning |
|---|---|
| `checked` | Units actually examined. Zero is vacuous, always. |
| `expected` | Units that should have been examined, or `None` when the denominator is genuinely not knowable in advance. |
| `sampled` / `sample_reason` | An explicit declaration of deliberately partial coverage. |

The denominator binds in **both directions**:

- `checked == 0` → `Verdict.VACUOUS`. Blocks.
- `checked < expected`, undeclared → `Verdict.UNDERCOVERED`. Blocks. "19 of 23
  checked" is a different claim from "checked", and only one of them is what a green
  check mark communicates.
- `checked > expected` → `Verdict.OVERCOVERED`. Blocks, and uniquely has **no sample
  pardon**: sampling declares a partial sweep, and a count above the denominator is
  not a partial anything. The honest remediations are to correct the count, correct
  `expected` to the true population, or pass `expected=None` — never to ship a
  contradicted ratio.
- `checked < expected` with `sampled=True` and a non-empty `sample_reason` → the
  undercoverage is converted to a pass, and the reason is surfaced in every
  rendering. `Coverage.__post_init__` rejects `sampled=True` without a reason.

The verdict set is asymmetric on purpose: two non-blocking outcomes (`PASS`, `SKIP`)
and five blocking ones (`FAIL`, `VACUOUS`, `UNDERCOVERED`, `OVERCOVERED`, `ERROR`).
It is much easier to accidentally produce a meaningless success than a meaningless
failure.

The enforcement lives in `Gate.ok()`, not in gate authors' discipline. A gate that
calls `self.ok(...)` over vacuous coverage gets `VACUOUS` back; short-undeclared
coverage gets `UNDERCOVERED`; over-covered gets `OVERCOVERED`. The author cannot
override this, which is the point. Constructing `GateResult` directly bypasses the
rule, so gates return through `ok`/`fail`/`skip`.

This is the `all([]) is True` incident made structural: a checkpoint-verification
tool reported `all_identity: True` on a corrupt artifact because the expert tensors
were absent, the comparison set was empty, and `all([])` is `True`.

## 2. Declared abstentions, priced as data

`SKIP` is the second non-blocking verdict and is two different statements wearing one
outcome:

| `AbstentionKind` | Meaning | How composites must price it |
|---|---|---|
| `NOT_APPLICABLE` | The property does not exist in this run's **declared** scope (an explicit `num_experts == 0`, never an absence of evidence). | May leave the applicable denominator, must be named, never counts as verified. |
| `NOT_ESTABLISHED` | The property may exist; the available evidence could not settle it (per-expert identity inside a stacked MoE tensor). | Stays inside every denominator — "could not check" is charged against the sweep. |
| `None` | The call site was never audited for the distinction. | Priced exactly like `NOT_ESTABLISHED`; never launders legacy code into an audited kind. |

The split lives on `GateResult.abstention` as machine-readable data — never in the
skip reason string, which prose-sniffing aggregators would paraphrase. `GateResult`
serialization emits `"abstention"` for every result so a downstream consumer reading
JSON never re-parses prose. And `GateResult.__post_init__` refuses an abstention kind
on any non-`SKIP` verdict: a pass cannot carry an inapplicability claim past an
aggregator.

"The property does not exist here" is legitimate only behind a positive declaration.
"I found no experts" is not "this model has no experts"; that inference is the
founding incident.

## 3. Controls are executable

Every gate declares `controls()` containing fixtures of **both** kinds, and
`verify_controls()` runs them — intended to be wired into CI:

- **`ControlKind.MUST_FIRE`** — a deliberately defective input the gate must block
  on. Proof the detector can fire; the executable form of the audit's review rule
  *name the positive control proving your detector could have fired.*
- **`ControlKind.MUST_PASS`** — a known-good input that must produce its *declared*
  outcome. The default declaration is `Verdict.PASS`; a healthy fixture the gate
  genuinely cannot adjudicate declares `Control.expect_skip` with a reason.
  `expect_skip` on a `MUST_FIRE` fixture is refused at construction — a positive
  control the detector is expected to abstain on proved nothing.

A gate with no `MUST_FIRE` control fails the build. So does a gate with no
`MUST_PASS` control, and so does a gate whose `MUST_PASS` set never reaches one real
`PASS` — each a zero-trip guard outside a data-driven loop that cannot see its own
absence of trips. `verify_controls` also applies the doctrine to itself: a
`gate_ids` entry that matches nothing, a `controls()` that raises, and a call
targeting zero gates are each reported as named failures, because an empty return
value would be "all controls held" over zero verified gates.

## 4. Gates fail closed

An exception inside `check()` is converted by `Gate.run()` into `Verdict.ERROR`,
which blocks. The same conversion applies one frame up: `run_event` turns exceptions
from `coerce_context` and from lazy context factories into `ERROR` results with
tracebacks, never escapes. A `check()` that returns something other than a
`GateResult` (it type-checks clean; the framework's premise is that promises about
verification are the thing most worth checking) is likewise an `ERROR`.

This is drawn from the record: a reward-module import failure silently disabled a
degeneracy veto, and a verifier exception counted as a pass.

The dispatcher enforces unwired-not-healthy: no supplied context of a gate's
declared `context_type` is a blocking `ERROR` naming the type, unless the caller
explicitly passes `missing_ctx="report-skip"` to `run_event`. Two supplied contexts
satisfying the declared type is ambiguous and blocks; dispatch never picks
arbitrarily. A value of `missing_ctx` other than `"block"` or `"report-skip"`
raises `ValueError` — a misspelled abstention mode must not read as consent.

## 5. Absence blocks, one level up

The report layer restates the same rule over gates instead of units:

- Callers declare `required` gate ids; any that never ran land in
  `GateReport.missing` and block the report.
- A sweep that ran zero gates gets a framework-synthesized `VACUOUS` result (id
  prefix `registry.empty_sweep.`, reserved so author gates cannot spoof it) unless
  the event was declared in `GateRegistry(event_allow_empty=...)` — the single
  opt-out, which lifts the block but leaves `GateReport.is_vacuous` true, and
  renders as "gateless by declaration".
- `GateReport.is_unverified` names the all-SKIP sweep: gates ran, every one
  abstained, zero units were examined. `allow_empty` does **not** pardon this
  clause — an environment where no gate applies is expressed by not selecting the
  gates, not by a wall of declared skips.
- `GateReport.ok` is the conjunction: nothing blocking, nothing missing, not
  vacuous (unless declared gateless), and not unverified. `raise_if_blocking()`
  raises `GateBlocked` otherwise.

Lifecyle placement matters for the same reason: `Lifecycle.FIRST_SAVE` exists
because the first checkpoint of a run is the cheapest place to catch a save defect,
and `Lifecycle.PROMOTE` is the last gate before the blast radius.

## Not yet specified

The following are not answered by the material at hand:

- **Concrete gate implementations** beyond the `ExpertBytesGate` example embedded in
  the `Gate` docstring (whose `make_aliased_ckpt` / `make_intact_ckpt` fixtures are
  illustrative, not shipped symbols shown here). Individual production gates and
  their contexts live outside this module.
- **The CLI, exit codes, and environment variables**: none appear in
  `gates/core.py`. How `verify_controls` is wired into CI (entry point, non-zero
  exit behaviour) is not specified in this source.
- **Aggregators/composites** that price `abstention` and the verified/applicable
  denominator are specified contractually here but their implementations are not in
  this file.
