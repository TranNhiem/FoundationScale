"""Scoped adapter-target selection.

The mechanism this module depends on, and the reason it can be this small: peft
matches a LIST ``target_modules`` with ``key.endswith(target)``. So a
fully-qualified module name matches exactly one module -- itself. Tower scoping
therefore needs no regex, no ``exclude_modules``, and no peft version bump; it
needs the selector to return qualified names instead of bare leaf names.

What went wrong without it, measured 2026-09-20: ``--adapter-target q_proj`` on
``gemma-4-31B`` selected 60 ``torch.nn.Linear`` under ``model.language_model``
AND 27 ``Gemma4ClippableLinear`` under ``model.vision_tower``, and peft raised
``Target module Gemma4ClippableLinear(...) is not supported``. The declaration
was not wrong -- ``q_proj`` is a real and reasonable thing to adapt. The
selector was wrong, because it had no way to say "the language tower's q_proj".

The second measured defect, same date, is why selection also announces
LAYER-POSITION COVERAGE. On Qwen3.5 -- a 3:1 hybrid where only one layer in
four carries ``self_attn.{q,k,v,o}_proj`` -- the conventional declaration
selected exactly ``full_attention * 4`` modules on three model sizes, printed a
plausible module count, and silently adapted one quarter of the network. A
module count cannot distinguish 64 modules spread over 16 layers from 64 spread
over 64; a layer-position coverage line can, and it is computed from data this
function already holds. Coverage is announced, never enforced: partial-depth
LoRA is a legitimate choice, and an unannounced one is not.

Nothing here imports torch at module scope. Selection is duck-typed over an
iterable of ``(qualified_name, module)`` pairs and an ``is_adaptable``
predicate, which keeps it testable on a machine with no torch and, more
usefully, keeps the scoping rule readable as a rule rather than as framework
plumbing.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

from foundationscale.families.registry import (
    FamilySpec,
    resolve_family,
    unregistered_family_reason,
)

__all__ = [
    "AdapterPlan",
    "plan_adapter_targets",
    "select_adapter_modules",
    "torch_linear_predicate",
]


def _under(name: str, prefixes: tuple[str, ...]) -> str | None:
    """Return the prefix that contains ``name``, or None.

    Containment means equality or a dotted descendant. ``model.layers`` must not
    match ``model.layers_extra``, which a bare ``startswith`` would.
    """
    for prefix in prefixes:
        if name == prefix or name.startswith(prefix + "."):
            return prefix
    return None


def _layer_key(name: str) -> str | None:
    """Return the layer-position key of a qualified module name, or None.

    The key is the prefix up to and INCLUDING the first integer path segment:
    ``model.layers.3.mlp.experts.7.gate_proj`` belongs to position
    ``model.layers.3``, not 7. First-integer-wins because the first integer is
    the depth coordinate the coverage statement is about; integers deeper in
    the path index experts, heads, or shards within a single layer, and
    charging a declaration for reaching layer 3 "seven times" would measure
    width as depth. A name with no integer segment -- embeddings, final norms
    -- has no layer position and returns None rather than being forced into a
    bucket where it would corrupt the count.
    """
    segments = name.split(".")
    for index, segment in enumerate(segments):
        if segment.isdigit():
            return ".".join(segments[: index + 1])
    return None


def select_adapter_modules(
    named_modules: Iterable[tuple[str, object]],
    spec: FamilySpec,
    is_adaptable: Callable[[object], bool],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Choose the modules an adapter may wrap, and say what was left out.

    Returns ``(selected_qualified_names, announcement_lines)``.

    The announcements are not logging. They are the part of this function that
    would have explained the ``Gemma4ClippableLinear`` failure at the moment it
    happened instead of a week later: every exclusion is attributed to a tower
    and a type, and the two distinguishable ways of selecting nothing -- no name
    matched, versus names matched but nothing was adaptable -- are reported
    differently, because they have different fixes.

    Layer-position coverage is announced for every language prefix, every time,
    including when it is complete. The measured Qwen3.5 defect was not a wrong
    selection, it was a complete-in-module-count selection over one quarter of
    the depth, and only a coverage line states that. A full-coverage case that
    said nothing would be indistinguishable from a run where coverage was never
    measured, which in this repository is the difference between PASS and
    VACUOUS. Coverage is derived over ALL modules the iterable yields -- not
    only leaf-name matches -- because the whole defect is that layers whose
    adaptable modules never match a declared leaf are invisible to the
    selection loop and must be counted anyway. The iterable is consumed exactly
    once; callers may hand a generator.
    """
    selected: list[str] = []
    # tower prefix -> type names excluded under it, and how many
    excluded_by_tower: dict[str, dict[str, int]] = {}
    # type names that matched by name and scope but failed is_adaptable
    rejected_types: dict[str, int] = {}
    matched_leaf_count = 0
    selected_under: dict[str, int] = {}
    # language prefix -> layer positions holding at least one adaptable module
    # (the coverage denominator; computed over every module, leaf match or not)
    adaptable_positions: dict[str, set[str]] = {}
    # language prefix -> layer positions holding at least one selected module
    selected_positions: dict[str, set[str]] = {}

    for name, module in named_modules:
        leaf = name.rsplit(".", 1)[-1]
        if leaf not in spec.adapter_leaf_modules:
            # Never a selection candidate, but it still witnesses that its layer
            # position holds something an adapter COULD have wrapped. This is the
            # population the Qwen3.5 defect lived in: linear-attention layers
            # match no declared leaf, so the selection loop above them is blind
            # to their existence unless the denominator is computed here.
            # No tower test here, unlike the selection path below, and that
            # asymmetry is safe only because FamilySpec refuses a language prefix
            # and a tower prefix where one contains the other. Without that
            # invariant a nested tower would be excluded from the numerator and
            # counted in the denominator, i.e. a gap no declaration could close.
            language = _under(name, spec.language_prefixes)
            if language is not None:
                key = _layer_key(name)
                if key is not None and is_adaptable(module):
                    adaptable_positions.setdefault(language, set()).add(key)
            continue
        matched_leaf_count += 1

        tower = _under(name, spec.tower_prefixes)
        if tower is not None:
            bucket = excluded_by_tower.setdefault(tower, {})
            type_name = type(module).__name__
            bucket[type_name] = bucket.get(type_name, 0) + 1
            continue

        language = _under(name, spec.language_prefixes)
        if language is None:
            # Matched the leaf name but sits outside every declared scope. This
            # is its own population: it means the family's prefixes are
            # incomplete, which is a registration gap, not a model defect.
            bucket = excluded_by_tower.setdefault("<outside every declared prefix>", {})
            type_name = type(module).__name__
            bucket[type_name] = bucket.get(type_name, 0) + 1
            continue

        if not is_adaptable(module):
            type_name = type(module).__name__
            rejected_types[type_name] = rejected_types.get(type_name, 0) + 1
            continue

        selected.append(name)
        selected_under[language] = selected_under.get(language, 0) + 1
        key = _layer_key(name)
        if key is not None:
            # Already known adaptable, so it belongs in the denominator without
            # asking the predicate a second time.
            adaptable_positions.setdefault(language, set()).add(key)
            selected_positions.setdefault(language, set()).add(key)

    lines: list[str] = []
    for tower in sorted(excluded_by_tower):
        types = excluded_by_tower[tower]
        total = sum(types.values())
        detail = ", ".join(f"{n}x{c}" for n, c in sorted(types.items()))
        lines.append(
            f"adapter scope: EXCLUDED {total} module(s) under {tower!r} ({detail}). "
            f"They carry a declared target leaf name but are not the language tower of "
            f"family {spec.name!r}, so adapting them would train a tower the declaration "
            "never asked for"
        )
    if rejected_types:
        detail = ", ".join(f"{n}x{c}" for n, c in sorted(rejected_types.items()))
        lines.append(
            f"adapter scope: {sum(rejected_types.values())} module(s) matched by name and "
            f"scope were NOT adaptable ({detail}). This is not the same as nothing "
            "matching: the names and the scope are right and the module type is the "
            "obstacle, so the fix is a wrapper for that type, not a different --adapter-target"
        )
    for language in sorted(selected_under):
        lines.append(
            f"adapter scope: selected {selected_under[language]} module(s) under "
            f"{language!r} for family {spec.name!r}"
        )
    for language in sorted(spec.language_prefixes):
        positions = adaptable_positions.get(language, set())
        if not positions:
            # "0 of 0" would print a vacuous completeness claim; an unindexed
            # stack is UNMEASURED coverage, and unmeasured must look different
            # from measured-and-complete or nobody will ever notice the swap.
            lines.append(
                f"adapter scope: layer-position coverage is UNMEASURABLE under "
                f"{language!r}: the model exposes no integer-indexed layer positions "
                "there, so there is no denominator to measure the declaration "
                "against. This is absence of a measurement, not evidence of coverage"
            )
            continue
        reached = selected_positions.get(language, set()) & positions
        if len(reached) == len(positions):
            lines.append(
                f"adapter scope: declared target leaves reached all {len(positions)} "
                f"indexed layer position(s) under {language!r}"
            )
        else:
            gap = len(positions) - len(reached)
            lines.append(
                f"adapter scope: declared target leaves reached {len(reached)} of "
                f"{len(positions)} indexed layer position(s) under {language!r}; "
                f"{gap} position(s) hold adaptable modules that no declared leaf "
                "name matched, so those layers train no adapter"
            )
    if not selected:
        lines.append(
            "adapter scope: SELECTED NOTHING. An adapter with no target modules trains "
            "no parameters while reporting a successful configuration, which is the one "
            f"outcome that must never be silent. {matched_leaf_count} module(s) matched a "
            f"declared leaf name out of {tuple(spec.adapter_leaf_modules)}; the caller must "
            "refuse rather than proceed"
        )

    return tuple(selected), tuple(lines)


@dataclass(frozen=True)
class AdapterPlan:
    """What an adapter should target, or why it must not run.

    Exactly one of ``targets`` and ``refusal`` is meaningful: a plan with a
    refusal has empty targets and must not be handed to peft. ``announcements``
    is populated in EVERY case, including the cases that change nothing --
    "family scoping was not applied" is a fact about the run that has to reach
    the log, because its absence is what made the measured failure confusing.
    """

    targets: tuple[str, ...]
    announcements: tuple[str, ...]
    refusal: str | None
    family: str | None

    @property
    def refused(self) -> bool:
        return self.refusal is not None


def plan_adapter_targets(
    config: Mapping[str, Any],
    declared_targets: Sequence[str] | None,
    named_modules: Iterable[tuple[str, object]],
    is_adaptable: Callable[[object], bool],
) -> AdapterPlan:
    """Decide the adapter's target modules from a declaration and a family.

    This is the whole policy, kept out of the training loop on purpose: adding a
    model family must be a registration, and if a new family ever required an
    edit to the loop then the loop, not the registry, would be where families
    live. The four cases and why each is what it is:

    * **Declared names are already qualified** -- passed through untouched.
      Scoping a fully-specified declaration would be the framework overriding a
      statement it was given.
    * **No family and no declaration** -- REFUSE. There is nothing to derive
      targets from, and peft's own inference is exactly the thing that failed on
      six of six runs here.
    * **No family but names were declared** -- pass them through, and say
      plainly that no scoping was possible. This is runnable but is the shape
      that selected a vision tower on ``gemma-4-31B``, so it does not get to be
      quiet.
    * **Family resolved** -- scope the declaration (or the family's own leaves)
      to the language tower and return qualified names.
    """
    spec = resolve_family(config)
    declared = list(declared_targets) if declared_targets is not None else None

    if declared is not None and any("." in target for target in declared):
        return AdapterPlan(
            targets=tuple(declared),
            announcements=(
                f"adapter scope: {len(declared)} target(s) were declared as qualified "
                "module names and are passed through verbatim, NOT family-scoped, "
                "because a qualified declaration already states its own scope",
            ),
            refusal=None,
            family=spec.name if spec is not None else None,
        )

    if spec is None:
        if declared is None:
            return AdapterPlan(
                targets=(),
                announcements=(),
                refusal=(
                    "an adapter is declared with no target modules, and "
                    + unregistered_family_reason(config)
                ),
                family=None,
            )
        return AdapterPlan(
            targets=tuple(declared),
            announcements=(
                f"adapter scope: targets {declared!r} are bare leaf names and "
                + unregistered_family_reason(config)
                + ". They are passed to the adapter UNSCOPED: if this model carries a "
                "vision or audio tower that reuses those leaf names, the adapter will "
                "attach to it",
            ),
            refusal=None,
            family=None,
        )

    leaves = tuple(declared) if declared is not None else spec.adapter_leaf_modules
    selected, lines = select_adapter_modules(
        named_modules,
        replace(spec, adapter_leaf_modules=leaves),
        is_adaptable,
    )
    if not selected:
        return AdapterPlan(
            targets=(),
            announcements=lines,
            refusal=(
                f"an adapter selected 0 modules in family {spec.name!r} for leaf names "
                f"{list(leaves)!r}. Refusing rather than handing peft an empty target "
                "set, which would send it back to the inference that failed here"
            ),
            family=spec.name,
        )
    return AdapterPlan(
        targets=selected,
        announcements=(
            *lines,
            f"adapter scope: family {spec.name!r} resolved from model_type; "
            f"{len(selected)} qualified target module(s) will be handed to the adapter",
        ),
        refusal=None,
        family=spec.name,
    )


def torch_linear_predicate() -> Callable[[object], bool]:
    """A predicate that is True for ``torch.nn.Linear`` and False for subclasses.

    Deliberately ``type(m) is Linear`` and not ``isinstance``. The measured
    failure was a Linear SUBCLASS -- ``Gemma4ClippableLinear`` wraps a Linear and
    peft refuses it -- so an ``isinstance`` test would admit exactly the module
    that caused the defect this package exists to fix. When a subclass does
    become adaptable, that is a change to this predicate made on purpose, with a
    test, rather than a change that happens by inheritance.

    torch is imported inside the function so that importing the family registry
    does not require torch, and so that a missing torch raises here with a name
    rather than returning a predicate that quietly answers False for everything
    and reports "selected nothing" for a reason that has nothing to do with the
    model.
    """
    try:
        import torch
    except ImportError as exc:  # pragma: no cover -- exercised by the negative control
        raise RuntimeError(
            "torch is required to decide which modules an adapter can wrap; refusing to "
            "return a predicate that answers False for every module, because that would "
            "report an empty selection as a property of the MODEL when it is a property "
            f"of the machine. Underlying: {exc!r}"
        ) from exc

    linear = torch.nn.Linear

    def _is_plain_linear(module: object) -> bool:
        return type(module) is linear

    return _is_plain_linear
