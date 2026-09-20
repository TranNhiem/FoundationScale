"""Model families, declared rather than assumed.

FoundationScale exists to train LLMs, VLMs and multimodal models -- plural, and
including ones that do not exist yet. That makes "which family is this?" a
first-class question, and this package is where it is answered.

The contract is one sentence: a family is DECLARED, and a family nobody declared
is a refusal, not a default. There is no fallback spec and there will not be
one. A fallback would let an unmeasured family train under another family's
assumptions, and the result of that is not an error -- it is a loss curve, a
checkpoint and a throughput number, all of them wrong in a way no gate can see.

Adding a family is adding a :class:`FamilySpec` and a test. It is not an edit to
the training loop, and if it ever becomes one, that is the defect.
"""

from __future__ import annotations

from foundationscale.families.adapters import (
    AdapterPlan,
    plan_adapter_targets,
    select_adapter_modules,
    torch_linear_predicate,
)
from foundationscale.families.registry import (
    REGISTRY,
    FamilySpec,
    resolve_family,
    unregistered_family_reason,
)

__all__ = [
    "REGISTRY",
    "AdapterPlan",
    "FamilySpec",
    "plan_adapter_targets",
    "resolve_family",
    "select_adapter_modules",
    "torch_linear_predicate",
    "unregistered_family_reason",
]
