"""Tower resolution and DDP dormancy, kept inside the families package.

The old loop.py heuristic keyed on LEAF attribute names ("vision_tower") with a
one-level ``.model`` fallback guess. FamilySpec declares full DOTTED paths, and
qwen3.5's tower is ``model.visual`` -- so the leaf table never matched, the
dormant list came back empty, ``find_unused_parameters`` stayed False, and every
multi-rank qwen3.5 run aborted on first backward. Resolution is therefore an
exact dotted-path walk over declared prefixes, and the prefix set itself lives
in the registry. The callers modality vocabulary is passed IN; nothing here
imports from the train package.
"""

from __future__ import annotations

from typing import Any

from foundationscale.families.registry import FamilySpec

__all__ = [
    "FamilyRefusal",
    "derive_find_unused_parameters",
    "resolve_module_path",
]

# Exit-contract wording for the REFUSE path: callers translate this exception
# to exit 96 (CANNOT-MEASURE/REFUSE); an unmet family precondition is never
# RED (5), because nothing failed -- the framework simply declines to guess.
REFUSAL_EXIT_STATUS = 96


class FamilyRefusal(Exception):
    """REFUSE (96): the family is unregistered, so dormancy is unknowable.

    Returning a default False here would silently reproduce the original qwen3.5
    abort for every future unregistered family; returning True would tune DDP
    for towers nobody has measured. Neither guess is acceptable.
    """

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason
        self.exit_status = REFUSAL_EXIT_STATUS


def resolve_module_path(model: Any, dotted_path: str) -> Any | None:
    """Resolve a full dotted attribute path on ``model``, or return None.

    Pure getattr per segment: "model.vision_tower" and "model.visual" both
    resolve if and only if every segment exists. The old one-level ``.model``
    fallback is gone on purpose -- a guess that happens to be right for one
    family's nesting depth and wrong for the next is exactly the bug being
    removed. None means ABSENT, which callers must treat as a declared-but-
    unmeasurable tower, not as a vacuous pass.
    """
    current = model
    for segment in dotted_path.split("."):
        current = getattr(current, segment, None)
        if current is None:
            return None
    return current


def derive_find_unused_parameters(
    family: FamilySpec | None,
    declared_modalities: frozenset[str] | tuple[str, ...] | set[str],
) -> bool:
    """True iff this family has a modality tower no declaration exercises.

    Floor is False: a tower-less family returns False, so a genuinely unused
    parameter still aborts DDP rather than being swallowed. Only
    ``tower_modalities`` participates -- a ``None``-modality prefix like
    ``mtp`` is out of adapter scope but is not a tower any declaration could
    exercise, and must not force the flag on its own. Declared-but-unresolvable
    towers are the caller's concern (resolve_module_path reports absence
    separately); this function answers the factual question the registry CAN
    answer.

    ``family=None`` REFUSES: dormancy for an unregistered family is unknowable,
    and guessing produced the multi-rank abort this function exists to end.
    """
    if family is None:
        raise FamilyRefusal(
            "cannot derive find_unused_parameters: the model family is not "
            "registered, so which submodules are towers (and which modalities "
            "exercise them) has never been declared. Guessing -- in either "
            "direction -- is what made every multi-rank qwen3.5 run abort on "
            "first backward. Two ways forward: pass --adapter-target with "
            "fully-qualified module names (a qualified name states its own "
            "scope, no registration needed), or add a FamilySpec for this "
            "family to foundationscale.families.registry -- a registration, "
            "not an edit to the training loop. Exit status 96 (REFUSE)."
        )
    declared = set(declared_modalities)
    # A prefix nobody declared in `towers` does not exist; an empty
    # tower_modalities is therefore a correct False, not a vacuous one --
    # the measurement happened at registration time.
    return any(modality not in declared for _, modality in family.tower_modalities)
