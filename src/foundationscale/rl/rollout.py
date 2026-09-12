"""The ``RolloutSource`` contract: experience generation behind one edge.

This module holds the stage-3 contract for experience generation (design
section 3.3) and its fail-closed handshake, and nothing else. No engine
adapter lives here: ``RolloutSource`` attaches to no existing
FoundationScale seam -- it is the one genuinely new edge in the design --
so the core deliberately knows nothing about which engine sits behind it,
and whichever engine does is an adapter concern per design section 5. No
algorithm lives here either: the policy-gradient algorithm that is this
contract's first consumer is stage 3's other half. The transport
measurement that gated it (design section 7, item 3) has since been taken
and returned CLEAR_WITH_ABSTENTIONS: no single transport wins, so weight
movement is specified as a sibling contract that declares the transport it
actually ran, and not as a capability of this one.

The handshake is split in two on purpose. ``SourceCapabilities`` and
:func:`check_capabilities` are the CLAIM half, run once at setup: the
source states which columns it can return, and a non-empty difference
against the algorithm's requirement refuses the run before any step.
:func:`verify_generated` is the MEASUREMENT half, run after every
``generate``: it reads the returned batch and confirms the claimed columns
actually arrived, aligned, with rows in them. A declared capability and a
delivered column are different things -- the design section 7 item 4
measurement that drives this plane found an undeclared ``sft_loss``
reading 0.0000 for a whole run passing four green gates -- and a source
that claims a column it never delivers is the rollout-side mirror of that
reading. A quantity that is not present sits in no denominator, and
nothing downstream may read as though it were.

WHAT THIS MODULE DOES NOT CLAIM: anything about generation throughput,
staleness, or engine behaviour at scale (design section 3.3 lists all three
UNMEASURED off Phase 2's ten-step, one-GPU run -- the section-7 item-3
measurement named above is weight transfer, a different quantity, and it
does not carry over here); anything about the scoring edge --
who turns completions into rewards is the open edge recorded in design
section 4, and this contract neither closes it nor papers over it; and any
device or dtype semantics -- a batch is a plain columnar container and no
accelerator is assumed anywhere in this file.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from foundationscale.rl.interfaces import ExperienceBatch

__all__ = (
    "CapabilityRefusal",
    "RolloutSource",
    "SourceCapabilities",
    "check_capabilities",
    "verify_generated",
)


class CapabilityRefusal(ValueError):
    # Raised when the fail-closed handshake fails: a required column the
    # source does not declare, a returned batch that does not deliver what
    # was declared, or a requirement so empty the check would answer
    # vacuously. Fail closed; never treat a claim as a delivery.
    pass


@dataclass(frozen=True, slots=True)
class SourceCapabilities:
    """What a rollout source CLAIMS it can put in a returned batch.

    ``columns`` names the columns ``generate`` can return;
    ``truncation_reported`` states whether the source reports truncation
    metadata -- whether it tells its consumer a completion was cut off.
    WHAT IS CLAIMED here is only the SHAPE of a claim: every column name a
    non-empty ``str`` (a source cannot claim to deliver something it
    cannot name), and no name twice (one column, one claim -- a duplicated
    capability names nothing new and usually hides the column the source
    meant to declare). WHAT IS NOT CLAIMED is delivery: construction never
    sees a batch, a source may hold a perfectly well-formed and perfectly
    dishonest capabilities object, and honesty is enforced by
    :func:`verify_generated` at the batch, not here. ``truncation_reported``
    is likewise an unmeasured claim; no check in this module corroborates
    it -- it is recorded so a run manifest can state what the source
    asserted.
    """

    columns: tuple[str, ...]
    truncation_reported: bool = False

    def __post_init__(self) -> None:
        columns = tuple(self.columns)
        seen: dict[str, int] = {}
        for index, name in enumerate(columns):
            if not isinstance(name, str) or not name:
                raise CapabilityRefusal(
                    f"columns[{index}]={name!r}: every declared capability "
                    f"column must be a non-empty str; absence of a name is "
                    f"not a name, and a source cannot claim to deliver "
                    f"something it cannot name"
                )
            if name in seen:
                raise CapabilityRefusal(
                    f"columns[{index}] repeats {name!r}, first declared at "
                    f"columns[{seen[name]}]: one column, one claim -- a "
                    f"duplicated capability names nothing new and usually "
                    f"hides the column the source meant to declare"
                )
            seen[name] = index
        object.__setattr__(self, "columns", columns)

    @classmethod
    def of(cls, *names: str, truncation_reported: bool = False) -> SourceCapabilities:
        """Ergonomic construction: ``SourceCapabilities.of("tokens", "logprobs")``.

        Routes through the real constructor, so every refusal above fires
        identically -- a convenience path that validated less would be a
        hole in the handshake, not a convenience.
        """
        return cls(columns=tuple(names), truncation_reported=truncation_reported)


@runtime_checkable
class RolloutSource(Protocol):
    """Produces completions for prompts without the core knowing the engine.

    Design section 3.3: one level wider (the source also owns weight
    synchronisation) and every generation-engine adapter re-implements
    synchronisation policy; one level narrower (the source returns raw
    text) and policy-gradient methods lose the logprobs they need. The
    signature is FoundationScale's own; nothing about any particular
    engine's batching or layout is claimed.

    ``capabilities`` is the CLAIM half of the fail-closed handshake and
    the batch ``generate`` returns is what :func:`verify_generated`
    MEASURES; an implementation whose ``generate`` delivers fewer columns
    than its ``capabilities`` declares is not a smaller source, it is a
    broken one. The protocol is runtime-checkable, so ``isinstance``
    refuses an object carrying neither ``generate`` nor ``capabilities``
    before any step rather than at the first ``generate`` call.

    It refuses nothing narrower than that, and the scope word is
    load-bearing (#397, MEASURED): ``@runtime_checkable`` checks method
    PRESENCE only and never signatures, so a source whose ``generate``
    takes a different keyword argument entirely passes
    ``isinstance(obj, RolloutSource)`` and fails later as a
    ``TypeError``. Setup tooling that needs the mis-signature question
    answered must ask
    :func:`foundationscale.rl.structural.structural_report`.
    """

    def generate(self, prompts: ExperienceBatch) -> ExperienceBatch: ...

    def capabilities(self) -> SourceCapabilities: ...


def _checked_required(required: Sequence[str], *, caller: str) -> tuple[str, ...]:
    # Shared shape validation for BOTH halves of the handshake. An empty
    # requirement is refused rather than passed, because an empty
    # requirement makes the check VACUOUS: every requirement in it is
    # trivially met because there are none, the check passes having
    # measured nothing, and a vacuous pass is the exact failure this
    # framework exists to prevent. A non-name entry is refused with its
    # index, for the reason every name-bearing field in this plane refuses
    # non-names: absence of a name is not a name.
    names = tuple(required)
    if not names:
        raise CapabilityRefusal(
            f"{caller} got an EMPTY required set: an empty requirement set "
            f"makes the check vacuously pass -- every requirement in it is "
            f"trivially met because there are none -- and a vacuous pass "
            f"is the exact failure this framework exists to prevent"
        )
    for index, name in enumerate(names):
        if not isinstance(name, str) or not name:
            raise CapabilityRefusal(
                f"required[{index}]={name!r}: every required column must "
                f"be a non-empty str; absence of a name is not a name"
            )
    return names


def check_capabilities(
    *,
    required: Sequence[str],
    capabilities: SourceCapabilities,
    origin: str = "<rollout-source>",
) -> tuple[str, ...]:
    """The fail-closed setup handshake: requirement against claim (design 3.3).

    WHAT IS CLAIMED on success: every name in ``required`` appears in the
    source's declared capabilities. The function returns exactly the
    validated required set, so the caller's denominator is the precise set
    that was checked -- never a superset it inferred.

    WHAT IS NOT CLAIMED: that the source will DELIVER any of it. This
    function compares two declarations -- the algorithm's requirement and
    the source's capabilities -- and never sees a batch. Declaring a
    capability is not the same as delivering it, which is why
    :func:`verify_generated` exists and why a run wires both halves.

    On failure the refusal lists EVERY missing name against the full
    requirement, with the denominator stated ("3 of 5"), rather than
    stopping at the first gap: a setup refusal costs nothing, and keeping
    the requirement's denominator visible in the refusal is what stops a
    half-met requirement from reading as a met one. An empty ``required``
    is refused outright -- see ``_checked_required``.
    """
    names = _checked_required(required, caller="check_capabilities")
    available = frozenset(capabilities.columns)
    missing = tuple(name for name in names if name not in available)
    if missing:
        raise CapabilityRefusal(
            f"{len(missing)} of {len(names)} required columns are "
            f"unavailable from {origin}: {', '.join(missing)}; the source "
            f"declares {capabilities.columns}. Refusing at setup costs "
            f"nothing; discovering the gap at step zero costs an "
            f"allocation"
        )
    return names


def verify_generated(
    batch: ExperienceBatch,
    *,
    required: Sequence[str],
    origin: str = "<rollout-source>",
) -> int:
    """The MEASUREMENT half of the handshake, run after ``generate``.

    ``capabilities()`` is a CLAIM and this function is the measurement
    that tests it. A source can declare a capability it never delivers --
    declaring a capability is not the same as delivering it -- which is
    exactly why both halves exist: the design section 7 item 4 measurement
    found trusted-but-absent quantities passing green gate sweeps, and a
    batch missing a declared column is the rollout-side mirror of that
    reading.

    WHAT IS CLAIMED on success: the returned ``int`` is the row count of
    every REQUIRED column, verified to be one shared count. That count --
    not ``len(batch)``, not a capabilities claim -- is the denominator any
    downstream verdict about this generation must use.

    WHAT IS NOT CLAIMED: anything about columns outside ``required`` (they
    sit in no denominator this check reports, so their alignment is not
    this function's business); anything about row CONTENT (a column of
    empty strings is present and aligned; quality is not presence); and
    anything about ``truncation_reported``, which no measurement in this
    module corroborates.

    Three refusals, in order. A required column the batch does not carry
    is named, in full, against the requirement. Two required columns with
    different row counts are named with BOTH counts. And a zero-row batch
    is refused rather than passed: with zero rows nothing was generated,
    so nothing was measured, and a pass would read as a healthy generation
    over an empty denominator.
    """
    names = _checked_required(required, caller="verify_generated")
    missing = tuple(name for name in names if name not in batch.columns)
    if missing:
        raise CapabilityRefusal(
            f"{len(missing)} of {len(names)} required columns are absent "
            f"from the batch returned by {origin}: {', '.join(missing)}; "
            f"the batch carries {tuple(batch.columns)}. A declared "
            f"capability that generate() does not deliver is a broken "
            f"source, not a smaller batch"
        )
    # Row alignment is re-derived here rather than inherited from
    # ExperienceBatch's construction-time invariant: the object crossing
    # this seam was produced by adapter code the core does not control,
    # and nothing guarantees that constructor ever ran on it.
    reference = names[0]
    row_count = len(batch.columns[reference])
    for name in names[1:]:
        length = len(batch.columns[name])
        if length != row_count:
            raise CapabilityRefusal(
                f"column {name!r} has {length} rows but column "
                f"{reference!r} has {row_count} rows in the batch returned "
                f"by {origin}; the count this verification returns must "
                f"be ONE row count, and a misaligned delivered batch has "
                f"none"
            )
    if row_count == 0:
        raise CapabilityRefusal(
            f"the batch returned by {origin} has 0 rows in the required "
            f"columns: nothing was generated, so nothing was measured, "
            f"and passing would read as a healthy generation over an "
            f"empty denominator"
        )
    return row_count
