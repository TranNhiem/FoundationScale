"""Policy-and-reference-model contract (design section 3.5).

This module holds :class:`PolicyPair`, the container for the model VIEWS an RL
algorithm needs, and nothing else. The container deliberately knows nothing
about what a view IS: a train view is whatever the training loop can step, a
generate view is whatever rollouts are sampled from, a reference is whatever a
frozen KL or DPO margin is scored against. Typing those views any tighter
would import model-family facts into the core, and those facts belong at the
``foundationscale.models.adapters`` edge.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

__all__ = ("PolicyPair", "PolicyRoleRefusal")


class PolicyRoleRefusal(ValueError):
    """The policy container was asked to hold or resolve an incoherent role."""


@dataclass(frozen=True)
class PolicyPair:
    """The model views one algorithm run needs: train, maybe generate, references.

    A view is opaque and duck-typed. This container asserts NOTHING about model
    family, parameter layout, or device placement: those facts live at the
    ``foundationscale.models.adapters`` edge, and re-deriving them here would
    duplicate the edge and make the core model-specific.

    The pair owns NO parallelism or topology surface, on purpose. One level
    wider and the core's topology surface is duplicated; that surface is
    currently five hard-named axes expressed in one parallelism library's
    vocabulary, and generalising it is a prerequisite for widening, not
    something to copy into this container.

    Optionality is the point (design section 3.1). SFT is
    ``PolicyPair(train_view=model)`` -- no generation view, no references, and
    NO stub implementations standing in for either. If a degenerate algorithm
    needed stubs to satisfy this container, the abstraction would be drawn
    wrong. DPO is ``PolicyPair(train_view=policy,
    references={"reference": frozen})``: the frozen policy arrives as a named
    reference and this container never asks who loaded it or how it stays
    frozen.
    """

    train_view: Any
    generate_view: Any | None = None
    references: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.train_view is None:
            raise PolicyRoleRefusal(
                "PolicyPair requires a train view and got None; a pair with "
                "no training view has nothing to optimise, and absence is "
                "not a view"
            )
        for role, view in self.references.items():
            if not isinstance(role, str) or not role:
                raise PolicyRoleRefusal(
                    f"reference role {role!r} is not a non-empty str; "
                    f"reference roles name run-record entries and must be "
                    f"non-empty strings"
                )
            if role in ("train", "generate"):
                raise PolicyRoleRefusal(
                    f"reference role {role!r} shadows the container's own "
                    f"{role!r} role; roles() would become ambiguous between "
                    f"a view and a reference"
                )
            if view is None:
                raise PolicyRoleRefusal(
                    f"reference {role!r} is None; absent is not a frozen "
                    f"reference. A caller with no reference passes no entry -- "
                    f"this refusal is what stops a run from training against "
                    f"a reference that was never loaded"
                )
        # Copy into a plain dict BEFORE the object is observable elsewhere:
        # the dataclass is frozen, but the caller's own mapping is not, and a
        # caller mutating it afterwards must not change this container.
        object.__setattr__(self, "references", dict(self.references))

    def reference(self, role: str) -> Any:
        """Return the frozen reference registered under ``role``.

        Absence is refused rather than answered with None: None means
        UNMEASURED in this codebase, and silently reading a missing reference
        as anything is how a run scores against a model that does not exist.
        """
        if role not in self.references:
            raise PolicyRoleRefusal(
                f"no reference role {role!r}; this pair carries reference "
                f"roles {tuple(sorted(self.references))}"
            )
        return self.references[role]

    def roles(self) -> tuple[str, ...]:
        """The roles this pair carries, in a stable, reproducible order.

        ``"train"`` first because it is mandatory, ``"generate"`` second when
        present, then reference roles sorted. A run record built from this
        tuple must not depend on a caller's mapping iteration order.
        """
        roles = ["train"]
        if self.generate_view is not None:
            roles.append("generate")
        roles.extend(sorted(self.references))
        return tuple(roles)

    @property
    def has_generate_view(self) -> bool:
        """Whether a generation view is present (SFT's honest answer is False)."""
        return self.generate_view is not None

    def with_reference(self, role: str, view: Any) -> PolicyPair:
        """Return a NEW pair with ``role`` added; the container is frozen.

        A duplicate role is refused and named: silently replacing a frozen
        reference is how a run ends up scoring against the wrong model.
        """
        if role in self.references:
            raise PolicyRoleRefusal(
                f"reference role {role!r} is already present; replacing a "
                f"frozen reference in place would change what this run "
                f"scores against, so build a new pair explicitly instead"
            )
        return PolicyPair(
            train_view=self.train_view,
            generate_view=self.generate_view,
            references={**self.references, role: view},
        )
