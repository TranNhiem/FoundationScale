# Extension and plugin points

FoundationScale's plugin surface is deliberately narrow. There are exactly three seams you can extend today — gates registered by import, model adapters registered by a call, and the adjudication layer that composes gate verdicts — plus the fixture/controls protocol documented alongside them. There is **no** setuptools entry-point mechanism for third-party plugins, and whether one should exist is an open design question. Anything advertised as a plugin API beyond this chapter does not exist in the codebase.

## What exists, at a glance

| Seam | Where | How a third party extends it |
|---|---|---|
| Gate contract | `foundationscale.gates.core` (re-exported via `foundationscale.integrate`) | Implement a gate, register it in `REGISTRY` by import |
| Model adapters | `src/foundationscale/models/adapters.py` | Implement `ModelAdapter`, call `register_adapter` |
| Adjudication | `src/foundationscale/gates/adjudication.py` | Composes per-gate verdicts into a run-level judgment |
| Dialect table | `DIALECT_TABLE` in `adapters.py` | Extension expert-count keys, subject to the structural guard |

Two things you will **not** find, so do not document them as if they exist:

* **Entry-point discovery.** There is no `entry_points` group, no `importlib.metadata` scan. Registration is by import: the module that calls `register_adapter` (or that installs a gate) must be imported for the registration to happen. An unimported plugin is not a plugin; it is a file on disk.
* **A gate base-class reference in this chapter.** The README names the gate base class as an extension point, but the source defining it (`foundationscale.gates.core`) is not part of this document's material. `src/foundationscale/integrate.py` re-exports `REGISTRY`, `GateRegistry`, `GateReport`, `GateBlocked`, `Lifecycle`, and `run_event` from it, so those names are safe to rely on via `foundationscale.integrate`. Anything more specific about the base class's constructor or abstract methods cannot be stated here.

## The gate registry and `run_event`

Registration populates `REGISTRY` by import. Once a gate is registered, callers run lifecycle events through `run_event`, re-exported from `foundationscale.integrate`, which dispatches each gate to the context it declares via its `Gate.context_type` attribute. This matters because a single training job hands its gates different context families: checkpoint gates want a `CheckpointGateContext`, the parity gate a `ParityGateContext`. `GateRegistry.run` broadcasts one object to every gate in an event; `run_event` exists so a multi-family sweep degrades into a *named, blocking* failure instead of a raw `TypeError`/`AttributeError` named for nothing:

```python
report = run_event(
    REGISTRY,
    "save",
    {
        CheckpointGateContext: CheckpointGateContext.from_path(ckpt_dir),
        ParityGateContext: ParityGateContext(ckpt_dir, reference_dir),
    },
    required=["checkpoint.expert_distinctness", "checkpoint.weight_parity"],
)
report.raise_if_blocking()
```

Two extension-relevant rules follow from the founding vacuity rule, one level up from a single gate:

* **Required gates.** The `required` argument declares which gates must run at that lifecycle point. Any required gate that never ran renders as `MISSING` and blocks the report. A caller that lists a gate name in `required` but never imports the module that registers it gets a blocking report, not a green one.
* **Missing contexts.** A gate whose `context_type` was never supplied is a blocking ERROR — *unwired, not healthy* — unless the caller explicitly passes `missing_ctx="report-skip"`. A sweep that ends up running zero gates is VACUOUS and blocks. `all([])` is `True`; FoundationScale refuses to inherit that.

Callers retrieve the run as a `GateReport`, and `GateBlocked` is what `report.raise_if_blocking()` raises.

## The model adapter seam

`src/foundationscale/models/adapters.py` is a full, real registration API. The contract is the `ModelAdapter` protocol:

```python
class ModelAdapter(Protocol):
    name: str

    def matches(self, config: dict[str, Any]) -> bool: ...

    def classify(self, config: dict[str, Any]) -> Classification: ...

    def enable_moe_block_flag(self, config: dict[str, Any]) -> tuple[bool | None, str]: ...
```

`register_adapter` adds an adapter to `_REGISTERED` (initialised from `_BUILTINS`, which contains only `GemmaAdapter()`), and precedence is deterministic: call order after the built-ins. Use `registry_snapshot()` to inspect the current registered tuple.

```python
register_adapter(MyAdapter())
classification = classify_config(config)
```

`select_adapter` walks adapters in order and returns the first whose `matches` returns true; if none match, it returns the generic fallback `_GENERIC` (`GenericHFAdapter`). The generic heuristic is **never registered** and `register_adapter` refuses it.

### What `register_adapter` refuses

Registration can fail, and the failure type is `AdapterRefusal` (a `ValueError`):

| Condition | Behaviour |
|---|---|
| `name` equal to a registered adapter's name | Refused: duplicate attribution |
| `name` equal to `"generic"` | Refused: the fallback is reserved and selected explicitly, never via the registry |

Inside classification itself, stated config facts of the wrong type are refused, never coerced:

* A flag key present but not a JSON boolean (e.g. the string `"false"`) raises `AdapterRefusal`.
* A count key present but not a JSON integer raises `AdapterRefusal` — and because `isinstance(True, int)` is `True` in Python, the boolean check deliberately comes first, so `num_experts: true` does not classify as a one-expert MoE.
* A negative count raises `AdapterRefusal`.

Contradicting signals (dense flag against a positive count, an MoE flag against a zero count, divergent positive counts, contradictory flags) do not raise; they classify as `Architecture.UNDETERMINED` with a `conflict:` evidence string naming every signal.

### `Classification`

`classify` returns a frozen `Classification`:

* `architecture` — an `Architecture` enum value: `DENSE`, `MOE`, or `UNDETERMINED`.
* `num_routed_experts` — `int | None`; set only from a positive count signal.
* `evidence` — a string that names the actual dotted keys and values consulted (e.g. `text_config.enable_moe_block=True`), or, when nothing was found, names the scopes actually searched.
* `adapter` — the `name` of the adapter that produced the classification.

The no-evidence case is explicit: `no MoE dialect keys present in <scopes>; absence is unmeasured, not dense`. Absence of a declaration classifies as `UNDETERMINED`, not `DENSE`.

### The dialect table and what you may safely extend

All recognised key names live in one auditable `DIALECT_TABLE`, never in branches. Rows are `kind: "flag"` (affirmative MoE declaration) or `kind: "count"` (routed-expert count):

* `_MEASURED_COUNT_KEYS` is pinned by import to the manifest's `_EXPERT_COUNT_KEYS` — names this repository has observed in the wild (`text_config.num_experts` on a production Gemma-4 26B-A4B; `n_routed_experts` in the DeepSeek family).
* `_EXTENSION_COUNT_KEYS` (`"num_routed_experts"`, `"moe_num_experts"`) extend `classify_config` only.
* `NESTED_SCOPES` is pinned to the manifest's `_NESTED_LM_SCOPE_KEY`; nested scopes are searched before top level.

#### Two pinned invariants you break at your own risk

1. **The emitter seam is two-sided.** `enable_moe_block_flag` is one half of a comparison in `tools/emit_run_manifest.py`: the emitter reads the affirmative flag *here* and the routed count from `declared_from_hf_config`, then refuses when the two disagree. Widening only `NESTED_SCOPES` on this side would make a config nested under `llm_config` read `flag=True` while the producer found no count, producing a refusal whose stated reason is *false* — the count exists, in a scope the producer never looked at. A claim mismatched to its evidence is a defect even when the verdict is safe. Widen `_NESTED_LM_SCOPE_KEY`'s consumers in the **same commit**; `test_model_adapters.py` pins the two sides together.

2. **The table may be wider than the producer's vocabulary, never narrower.** Extension keys reach the standalone classifier only, which is safe because `classify_config` has no second side to disagree with. But an import-time guard raises `ImportError` if any measured key from `_EXPERT_COUNT_KEYS` is missing from `DIALECT_TABLE` — and it is a `raise`, not an `assert`, because `python3 -O` strips asserts and a guard that vanishes under a flag is not a guard.

Do not add key names in code branches. The table is the single place dialect knowledge lives; the comment at its head says so and means it.

### A concrete subclass

`GemmaAdapter` shows the intended shape: `matches` claims any config whose `model_type` is a string starting with `"gemma"`; `classify` keeps Gemma's ownership of `enable_moe_block` affirmative semantics while still collecting counts, so a false flag beside a live routed count is `UNDETERMINED`, never a silent win. `enable_moe_block_flag` scans only the adapter's `_flag_keys` and returns `(value, scope_name)` or `(None, "")`.

Note also a style rule written into the source: `GenericHFAdapter` does not override `classify` because the override would be byte-identical to `_Base`'s — a duplicated method that cannot differ is a second place to fix a bug in. Extension code should inherit before it duplicates.

## The controls runner as an entry point

The controls runner is itself installed as a console script:

```
foundationscale-controls = foundationscale.gates.controls:main
```

The practical consequence for anyone adding a gate: because CI runs a `controls` job through this entry point over `REGISTRY`, a gate added to the registry is automatically audited by CI. You do not register your new gate with the controls job separately; importing it into the registry is sufficient. (The fixture/controls protocol that the runner enforces is named as an extension point by the README, but its defining source is not part of this chapter's material — the console script and the `controls:main` target above are the only fixtures-side facts that can be stated here.)

## The adjudication layer

The adjudication layer lives at `src/foundationscale/gates/adjudication.py`. Its role is stated by the README — it composes per-gate verdicts into a run-level judgment, propagating coverage instead of assent — but its source is not included in the material for this chapter, so no function names, signatures, or invocation examples from that module can be given here. Adjudication is not implemented in any file shown in this chapter; the current behaviour that *can* be stated is that `run_event` returns a `GateReport`, `raise_if_blocking()` raises `GateBlocked`, and the run-level judgment above that is adjudication's to define. Extend it only against the module itself, not against an imagined API synthesized from this text — a claim must carry its own coverage, and so must this document.
