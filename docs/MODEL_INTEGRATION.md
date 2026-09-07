# Model integration

## Scope of this document

FoundationScale validates the declared topology around a model rather than wrapping
the model class. The train path delegates to `transformers.Trainer`, so any
architecture Trainer can construct is usable; the package never subclasses or
requires a particular model. This document describes the two modules that do touch
model-adjacent state — `src/foundationscale/models/adapters.py` and
`src/foundationscale/integrate.py` — both thin by measurement, and states plainly
what the package does not provide.

The founding rule applies throughout: a claim must carry its own coverage. When no
MoE dialect key is present in a config, classifying the model as "dense" would be a
claim without evidence, so the adapter's evidence string says exactly that:
`absence is unmeasured, not dense`.

## Model-agnostic framing

No example in the documentation should imply one model family. The launchers
demonstrate a vision-language model end to end, but every estate value — paths,
model root, node identity — enters via environment, not source. Nothing in the
adapter layer keys behaviour off a file location.

## The adapter layer

The public surface is re-exported from `foundationscale.models`:

| Name | Kind | Role |
|---|---|---|
| `AdapterRefusal` | exception (`ValueError`) | Raised when a stated config fact has the wrong type |
| `Architecture` | `str, Enum` | `DENSE`, `MOE`, `UNDETERMINED` |
| `Classification` | frozen dataclass | `architecture`, `num_routed_experts`, `evidence`, `adapter` |
| `ModelAdapter` | `Protocol` | `name`, `matches`, `classify`, `enable_moe_block_flag` |
| `DIALECT_TABLE` | tuple of dicts | The single auditable dialect table |
| `GenericHFAdapter` | class | The declared explicit fallback; never registered |
| `GemmaAdapter` | class | Owns `enable_moe_block` affirmative semantics |
| `classify_config` | function | Classify a config dict |
| `select_adapter` | function | Choose the first registered adapter whose `matches` accepts the config |
| `register_adapter` | function | Append to the registry; rejects duplicates |
| `registry_snapshot` | function | Immutable view of the registry |

### Fail closed, never coerce

`AdapterRefusal` exists because a stated declaration of the wrong type is refused,
not coerced:

* A flag key holding anything other than a JSON boolean is refused. Coercing a
  quoted `"false"` would classify it as MoE while looking like a config fact.
* A count key holding a boolean or non-integer is refused. The bool check comes
  first because `isinstance(True, int)` is `True` in Python; without it
  `num_experts: true` would read as a one-expert MoE.
* A negative count is refused; expert counts are unsigned.
* `classify_config` refuses a config that is not a JSON object.

Refusals are raises, not asserts — `python -O` strips asserts, and a guard that
vanishes under a flag is not a guard.

### `classify_config` and `Architecture`

```python
from foundationscale.models import classify_config

classification = classify_config(config)
```

`select_adapter` walks the registry (or an explicitly supplied sequence) in order,
returns the first adapter with a matching `matches(config)`, and otherwise returns
`GenericHFAdapter`. Registration order is deterministic: built-ins first
(`GemmaAdapter`), then `register_adapter` call order. Registering an adapter
named `generic`, or any duplicate name, raises `AdapterRefusal`; the generic
fallback is selected explicitly and must never appear inside the registry.

A `Classification` carries its evidence string, which names the actual dotted keys
and values observed — e.g. `conflict: dense flag (...) vs positive expert count
(...)` — so a reader never has to trust a bare enum value.

### The dialect table

`DIALECT_TABLE` is the one auditable dialect table; key names do not appear in
branches elsewhere. Each row has a `kind`:

* `flag` — an affirmative MoE declaration (`enable_moe_block`).
* `count` — a routed-expert count.

The count set has two tiers with different reach:

* `_MEASURED_COUNT_KEYS` — pinned by import to the manifest's
  `_EXPERT_COUNT_KEYS`, never restated. These are the names the repo has observed
  in production configs (a Gemma-4 26B-A4B declared `text_config.num_experts`; a
  DeepSeek-family config declared `n_routed_experts`). A second, narrower copy in
  the probe is what once made one config MoE to the library and dense to the
  probe; that failure mode is why the keys are imported, not duplicated.
* `_EXTENSION_COUNT_KEYS` — `num_routed_experts`, `moe_num_experts`. Published
  HF dialects that extend `classify_config` only. `classify_config` is a
  standalone classifier with no second side to disagree with; the extension keys
  are deliberately *not* reachable from the emitter seam, which reads only the
  affirmative flag.

An import-time structural guard raises `ImportError` if any measured key is
missing from the table: a measured key the producer honours and the table omits
would silently classify a real MoE as "no dialect keys present".

### Scope order and conflict resolution

Nested scopes are checked before top level. `NESTED_SCOPES` is pinned by import to
the manifest's `_NESTED_LM_SCOPE_KEY` rather than restated. Within the collected
signals the verdict rules are:

* Conflicts yield `Architecture.UNDETERMINED` with a `conflict:` evidence string:
  a dense flag beside a positive count, an MoE flag beside a zero count, divergent
  positive counts, or contradictory flags. In particular, `GemmaAdapter` treats a
  false flag beside a live routed count as `UNDETERMINED`, never a silent win.
* One or more positive counts with no conflict yield `Architecture.MOE` with the
  count populated, corroborating flags noted in the evidence.
* A lone affirmative flag yields `Architecture.MOE` with `num_routed_experts`
  as `None`.
* Negative evidence (a false flag or a zero count) yields `Architecture.DENSE`.
* No dialect keys anywhere yields `UNDETERMINED`; the evidence names the scopes
  actually searched, so the string cannot go stale if `NESTED_SCOPES` changes.

### The two-sided emitter seam

`enable_moe_block_flag` is one half of a two-sided comparison in
`tools/emit_run_manifest.py`: the emitter reads the affirmative flag here and the
routed count from `declared_from_hf_config`, then refuses when the two disagree.
Widening only this side is therefore not a safe superset. If a config nested under
`llm_config` were read by the adapter while the producer traversed only
`text_config` and top level, the emitter would see `flag=True` beside no count and
refuse with "no routed-expert count was found under the keys this tool
understands" — a reason that is false, because the count exists in a scope the
producer never looked at. A claim mismatched to its evidence is a defect even when
the verdict is safe.

Consequently, supporting another nesting requires widening
`_NESTED_LM_SCOPE_KEY`'s consumers in the same commit; `test_model_adapters.py`
pins the two sides together.

### Writing a custom adapter

Implement the `ModelAdapter` Protocol and call `register_adapter`:

```python
from foundationscale.models import register_adapter

register_adapter(my_adapter)
```

Deterministic precedence is call order of `register_adapter` after the built-ins.
Duplicate or reserved names are refused, so attribution cannot drift. `matches`
must claim a config verifiably — `GemmaAdapter` checks that `model_type` is a
string whose lowercase form starts with `gemma`; the base implementation claims
nothing.

## `integrate.py`

`src/foundationscale/integrate.py` is the package's other model-touching module.
The README describes it as thin by measurement, but its source is not provided
with this documentation's reference material, so its individual functions are not
enumerated here. Where its behaviour matters, it is covered through the seam
described above: `tools/emit_run_manifest.py` and the dialect table contract. If
its exact public surface is needed, that section is a gap in this document's
source, not an omission of intent — no API from it is quoted here rather than
invented.

## Environment-entered estate values

Paths, model root, and node identity all enter through environment variables, not
source edits. The launcher examples therefore remain architecture-neutral; nothing
in the documentation or the launchers hardcodes an estate.

## What does not exist

* **A curated recipe library.** There is no catalogue of per-family training
  recipes in this tree. The catalogue on the poster is the audit's design output,
  not code in this tree.
* **A parallelism-aware model wrapper.** The framework validates the declared
  topology around the model; it does not wrap the model class to manage
  parallelism strategies.
* **A MoE detection path wider than the dialect table.** Published HF key names
  beyond those listed above are not consulted; an unrecognised declaration is
  `UNDETERMINED` with evidence stating which scopes were searched, not a guessed
  verdict.
