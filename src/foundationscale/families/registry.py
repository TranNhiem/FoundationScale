"""The model-family registry.

A family is DECLARED here, never guessed. Nothing in this module consults a
filesystem path: a checkpoint directory called ``gemma-4-31B`` is a naming
convention, not a contract, and a framework that keys behaviour off it will be
wrong the first time someone renames a directory or vendors a fork. The only
thing consulted is ``config.json``'s ``model_type``, which the checkpoint's own
author wrote and which the modelling code itself dispatches on.

Why the registry exists at all, measured 2026-09-20 on GB200:

    peft 0.18.1 ships a table of 38 model_types it can infer LoRA targets for.
    It knows ``gemma``, ``gemma2``, ``gemma3_text``, ``qwen2``, ``qwen3``. It
    knows none of ``gemma4``, ``gemma4_text``, ``gemma4_unified``, ``qwen3_5``,
    ``qwen3_5_moe`` -- that is, none of the families actually on this estate. Six
    LoRA runs across two vendors and both densities died inside peft with
    ``Please specify `target_modules```.

    Inheriting a third party's family table means the framework works for last
    year's models and hard-fails on this year's. So the table is ours.

The second measured reason is subtler and is why a bare list of leaf-module
names is not enough. On ``gemma-4-31B`` the instantiated graph carries
``{q,k,v,o,gate,up,down}_proj`` in two places: 60 of each under
``model.language_model`` as ``torch.nn.Linear``, and 27 of each under
``model.vision_tower`` as ``Gemma4ClippableLinear``. A selector that matches on
leaf name alone is TOWER-BLIND -- it selects the vision tower too, and peft
cannot wrap a Linear subclass. So a family must declare where its language
tower lives and where its other towers live, and selection must be scoped.
``qwen3_5`` makes the same point from the other direction: it has no audio
tower at all, and it adds an ``mtp`` multi-token-prediction head that Gemma4
does not have. There is no global answer.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

__all__ = [
    "REGISTRY",
    "FamilySpec",
    "resolve_family",
    "unregistered_family_reason",
]


@dataclass(frozen=True)
class FamilySpec:
    """Everything the training plane needs to know about one model family.

    Every field is data. There is deliberately no behaviour here and no hook
    for behaviour: the moment a family can supply code, adding a family stops
    being a registration and becomes a code review, which is the coupling this
    package exists to remove.
    """

    name: str
    """Human label used in announcements and refusals. Not a lookup key."""

    model_types: tuple[str, ...]
    """``config.json`` ``model_type`` values this spec claims, including the
    ``text_config`` sub-types, because a composite VLM config states one at each
    level and either may be the one present."""

    language_prefixes: tuple[str, ...]
    """Qualified-name prefixes under which the language tower lives."""

    tower_prefixes: tuple[str, ...]
    """Qualified-name prefixes of every NON-language tower. May be empty, which
    means the family declares none -- not that none was looked for."""

    adapter_leaf_modules: tuple[str, ...]
    """Leaf module names an adapter may wrap, scoped by the prefixes above."""

    expert_count_path: tuple[str, ...]
    """Path into the config mapping at which an expert count is found, e.g.
    ``("text_config", "num_experts")``. Composite configs state this on the text
    sub-config, so a flat ``config.get("num_experts")`` reads ``None`` on a
    model that has 128 experts and silently concludes the model is dense. May be
    empty for a family that never declares experts."""

    def __post_init__(self) -> None:
        for field in ("name",):
            if not getattr(self, field):
                raise ValueError(f"FamilySpec.{field} must be non-empty")
        for field in ("model_types", "language_prefixes", "adapter_leaf_modules"):
            if not getattr(self, field):
                raise ValueError(
                    f"FamilySpec.{field} must be non-empty: a family that declares no "
                    f"{field} cannot be resolved or scoped, and an empty tuple would "
                    "make every lookup silently miss rather than loudly refuse"
                )
        # A language prefix that is a prefix of a tower prefix (or the reverse)
        # makes scoping ambiguous: a module could be simultaneously in and out of
        # scope depending on which rule ran first. That is a registration bug and
        # it must be caught here, at import, not at the first LoRA run.
        for lang in self.language_prefixes:
            for tower in self.tower_prefixes:
                if lang == tower or lang.startswith(tower + ".") or tower.startswith(lang + "."):
                    raise ValueError(
                        f"FamilySpec {self.name!r} declares overlapping scopes: language "
                        f"prefix {lang!r} and tower prefix {tower!r}. One contains the "
                        "other, so whether a module inside it is adaptable depends on "
                        "evaluation order rather than on the declaration"
                    )


REGISTRY: tuple[FamilySpec, ...] = (
    FamilySpec(
        name="gemma4",
        # Four keys for one vendor family, all measured from real checkpoints:
        # E4B/26B-A4B/31B state `gemma4`, while 12B states `gemma4_unified`. A
        # family is a SET of model_types, which is exactly why the registry keys
        # on the set and not on a vendor name.
        model_types=("gemma4", "gemma4_text", "gemma4_unified", "gemma4_unified_text"),
        language_prefixes=("model.language_model",),
        tower_prefixes=("model.vision_tower", "model.audio_tower", "model.embed_vision"),
        adapter_leaf_modules=(
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ),
        expert_count_path=("text_config", "num_experts"),
    ),
    FamilySpec(
        name="qwen3.5",
        model_types=("qwen3_5", "qwen3_5_text", "qwen3_5_moe", "qwen3_5_moe_text"),
        # Two language prefixes because the auto-class matters: loaded as a
        # conditional-generation VLM the language tower is `model.language_model`,
        # loaded as a plain causal LM it is `model.layers`. Both were observed on
        # the same checkpoint, so declaring only one would make the selector
        # silently empty under the other load path.
        language_prefixes=("model.language_model", "model.layers"),
        # No audio tower -- Qwen3.5 carries `vision_config` only -- and an `mtp`
        # multi-token-prediction head that Gemma4 has no analogue for. Listing it
        # here is what keeps an adapter out of it.
        tower_prefixes=("model.visual", "mtp"),
        adapter_leaf_modules=(
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ),
        expert_count_path=("text_config", "num_experts"),
    ),
)


def _assert_no_duplicate_model_types(registry: tuple[FamilySpec, ...]) -> None:
    """Two specs claiming one model_type would make resolution order-dependent.

    Checked at import so a bad registration cannot reach a training run.
    """
    seen: dict[str, str] = {}
    for spec in registry:
        for model_type in spec.model_types:
            if model_type in seen:
                raise ValueError(
                    f"model_type {model_type!r} is claimed by both {seen[model_type]!r} "
                    f"and {spec.name!r}; resolution would depend on registry order"
                )
            seen[model_type] = spec.name


_assert_no_duplicate_model_types(REGISTRY)


def _declared_model_types(config: Mapping[str, Any]) -> tuple[str, ...]:
    """The model_type values a composite config states, outermost first.

    A VLM config states one at the top level and another on ``text_config``.
    Either may be the one the registry knows, so both are tried -- and both are
    reported in the refusal when neither is.
    """
    found: list[str] = []
    top = config.get("model_type")
    if isinstance(top, str) and top:
        found.append(top)
    text_config = config.get("text_config")
    if isinstance(text_config, Mapping):
        nested = text_config.get("model_type")
        if isinstance(nested, str) and nested and nested not in found:
            found.append(nested)
    return tuple(found)


def resolve_family(config: Mapping[str, Any]) -> FamilySpec | None:
    """Return the spec claiming this config's model_type, or None.

    None is not a failure signal to be papered over with a default. There is no
    default: a family nobody declared is a family nobody has measured, and
    training it under another family's assumptions produces a number that looks
    exactly like a real one. Callers must turn None into a refusal, using
    :func:`unregistered_family_reason` for the text.
    """
    declared = _declared_model_types(config)
    for model_type in declared:
        for spec in REGISTRY:
            if model_type in spec.model_types:
                return spec
    return None


def unregistered_family_reason(config: Mapping[str, Any]) -> str:
    """Refusal text for a config no spec claims.

    This is REFUSE (96) text, not an exception message: the machine is fine and
    the declaration is fine, the framework simply has nothing measured to say
    about this family. It names what was tried, what is known, and both ways
    forward, because a refusal that does not say how to proceed is just a stop.
    """
    declared = _declared_model_types(config)
    tried = (
        ", ".join(repr(m) for m in declared) if declared else "<none: config states no model_type>"
    )
    known = ", ".join(sorted(m for spec in REGISTRY for m in spec.model_types))
    return (
        f"no registered model family claims model_type {tried}. Registered model_types "
        f"are: {known}. Adapter target selection needs to know where this family's "
        "language tower lives and which submodules are separate towers, and guessing "
        "produces a run rather than an error -- which is worse. Two ways forward, both "
        "supported today: pass --adapter-target with fully-qualified module names, which "
        "needs no registration because a qualified name already states its own scope; or "
        "add a FamilySpec for this family to foundationscale.families.registry, which is "
        "a registration and not an edit to the training loop"
    )
