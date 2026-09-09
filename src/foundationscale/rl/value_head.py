"""The ``ValueHead`` contract: value estimation as a callable role, not a column.

This module is the third sibling of the capability family, after
rollout.py and weightsync.py: the same two-half fail-closed handshake,
run against a different edge. The reason that edge is a ROLE and not a
column is PPO's defining feature -- the inner epoch loop. GRPO reads the
current policy log-probs out of a batch column named in config, which
keeps that module torch-free. PPO cannot do the same for values: the
value estimate must be recomputed with the CURRENT parameters on every
inner epoch, and a frozen column would price stale parameters while
reading as a fresh estimate. The current-value estimate is therefore
specified here as a callable -- ``estimate`` -- and never as batch data.
The OLD value estimate is the other half of the split: it is the PPO2
clipping reference, fixed at rollout time and static for the whole
rollout, so that one stays a batch column exactly like GRPO's log-prob
column. A column holds what does not move; a role prices what must move
with the parameters.

Like its siblings, the handshake is in two halves. ``ValueCapabilities``
and :func:`check_value_capabilities` are the CLAIM half, run once at
setup; :func:`verify_estimates` is the MEASUREMENT half, run on the
returned estimates, grading only the SUPERVISED positions the mask keeps
and returning their count -- the denominator any downstream verdict
about this estimate must use.

WHAT THIS MODULE DOES NOT CLAIM: anything about estimate QUALITY -- a
finite number at a supervised position is present, not correct;
anything about masked positions -- they are unmeasured and their
contents are not graded; anything about ``shares_policy_trunk`` -- it
is declared here and consumed by NEITHER function in this module, on
purpose: the weight-sync plane reads it to move a separate value head's
parameters without double-moving a shared trunk; anything about the
model behind ``estimate`` -- architecture, device, and dtype are
adapter concerns; and no torch import appears anywhere in this file.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from math import isfinite
from typing import Any, Literal, Protocol, cast, runtime_checkable

from foundationscale.rl.interfaces import ExperienceBatch

__all__ = (
    "ValueCapabilities",
    "ValueCapabilityRefusal",
    "ValueEstimateRefusal",
    "ValueHead",
    "check_value_capabilities",
    "verify_estimates",
)


class ValueCapabilityRefusal(ValueError):
    # Raised when the setup handshake fails: a capability value outside
    # the contract's literals, a truthy int standing in for a bool flag,
    # a granularity the requirement cannot be served from, or an
    # old-value need the head does not report. CONFIGURATION-side only.
    # Fail closed; never treat a claim as a delivery.
    pass


class ValueEstimateRefusal(ValueError):
    # Raised when returned estimates cannot be counted: a non-iterable
    # return, a row count or row length that disagrees with the mask, a
    # non-finite or non-coercible value at a supervised position, or a
    # batch with zero supervised positions. CALL-TIME DATA-side only.
    # Abstention is None; UNMEASURED is not zero.
    pass


_GRANULARITIES = ("token", "sequence")


@dataclass(frozen=True, slots=True)
class ValueCapabilities:
    """What a value head CLAIMS about itself at setup.

    ``granularity`` declares the unit each estimate prices -- exactly
    ``"token"`` (one number per position) or ``"sequence"`` (one number
    per row). ``shares_policy_trunk`` declares whether the head's
    parameters are shared with the policy trunk or separate.
    ``reports_old_values`` declares whether the head records the
    rollout-time estimate that PPO2 value clipping needs.

    WHAT IS CLAIMED: only the SHAPE of a claim -- granularity is one of
    the two literals and both flags are real bools. Python's 1 is not
    True: it is truthy and isinstance(1, int) is True, so a truthy int
    read as a capability would launder an unmeasured claim into a
    declared one, and it is refused.

    WHAT IS NOT CLAIMED: delivery or correctness of any estimate --
    construction never sees a batch and a head may hold a well-formed,
    dishonest capabilities object; honesty is measured by
    :func:`verify_estimates`, not here. ``shares_policy_trunk`` in
    particular is declared and consumed by NEITHER function in this
    module: the weight-sync plane reads it, because it must move a
    separate value head's parameters and must not double-move a shared
    trunk. It is a declaration for that consumer, not dead config.
    """

    granularity: Literal["token", "sequence"]
    shares_policy_trunk: bool
    reports_old_values: bool

    def __post_init__(self) -> None:
        if not isinstance(self.granularity, str) or self.granularity not in _GRANULARITIES:
            raise ValueCapabilityRefusal(
                f"granularity={self.granularity!r}: granularity is "
                f"exactly 'token' or 'sequence', one declaration of "
                f"the unit the head prices; a value outside the two "
                f"literals declares neither, and no consumer can size "
                f"GAE against a granularity it cannot name"
            )
        if not isinstance(self.shares_policy_trunk, bool):
            raise ValueCapabilityRefusal(
                f"shares_policy_trunk={self.shares_policy_trunk!r}: a "
                f"capability flag must be a real bool -- Python's 1 is "
                f"truthy, so a truthy int read as a capability would "
                f"launder an unmeasured claim into a declared one; 1 "
                f"is not True"
            )
        if not isinstance(self.reports_old_values, bool):
            raise ValueCapabilityRefusal(
                f"reports_old_values={self.reports_old_values!r}: a "
                f"capability flag must be a real bool -- Python's 1 is "
                f"truthy, so a truthy int read as a capability would "
                f"launder an unmeasured claim into a declared one; 1 "
                f"is not True"
            )


@runtime_checkable
class ValueHead(Protocol):
    """Recomputes value estimates with the current parameters, and nothing else.

    ``estimate`` is the call PPO's inner epoch loop makes on every inner
    epoch -- the reason this contract is a role and not a column, per
    the module docstring. ``capabilities`` is the CLAIM half, checked
    once at setup by :func:`check_value_capabilities`; what ``estimate``
    returns is MEASURED by :func:`verify_estimates`.

    WHAT IS CLAIMED: the signature is FoundationScale's own, and the
    protocol is runtime-checkable so setup tooling can refuse a
    non-head before any step rather than at the first ``estimate``
    call. WHAT IS NOT CLAIMED: anything about the model behind
    ``estimate`` -- architecture, device, dtype, and how estimates are
    produced are adapter concerns, and none is assumed in this file.
    """

    def estimate(self, batch: ExperienceBatch) -> Sequence[Sequence[float]]: ...

    def capabilities(self) -> ValueCapabilities: ...


def check_value_capabilities(
    *,
    required_granularity: str,
    needs_old_values: bool,
    capabilities: ValueCapabilities,
    origin: str = "<value-head>",
) -> tuple[str, ...]:
    """The CLAIM half of the handshake, run once at setup.

    WHAT IS CLAIMED on success: the head's declared granularity can
    serve ``required_granularity``, and if ``needs_old_values`` is True
    the head declares old-value reporting. The returned sorted tuple is
    exactly the capability names this check CONSUMED -- ``granularity``
    always, ``reports_old_values`` only when ``needs_old_values`` is
    True -- so the caller's denominator is precisely what was read.

    WHAT IS NOT CLAIMED: that the head will DELIVER any of it -- this
    function compares declarations and never sees an estimate, which is
    why :func:`verify_estimates` exists; anything about how a consumer
    aggregates token-level estimates when only a sequence value is
    required; and anything about ``shares_policy_trunk``, which is
    declared for the weight-sync plane and deliberately never consumed
    here.

    Four refusals. A ``required_granularity`` outside the two literals
    is named and refused -- a requirement naming neither checks against
    nothing and would answer vacuously. A ``needs_old_values`` that is
    not a real bool is refused: a truthy int read as a requirement
    launders an unmeasured need into a declared one. A head offering
    ``"sequence"`` where ``"token"`` is required is refused: GAE over
    per-token values cannot be computed from one number per row, and
    silently broadcasting the sequence value across tokens would be a
    fabricated per-token measurement. And ``needs_old_values=True``
    against a head that does not report old values is refused: the PPO2
    value clip needs the rollout-time estimate as its reference, and
    silently disabling the clip is not a smaller algorithm, it is a
    different one running under the clipped name.
    """
    if not isinstance(required_granularity, str) or required_granularity not in _GRANULARITIES:
        raise ValueCapabilityRefusal(
            f"required_granularity={required_granularity!r}: the "
            f"required granularity is exactly 'token' or 'sequence'; "
            f"a requirement naming neither is checked against nothing "
            f"and would answer vacuously -- a vacuous pass is the "
            f"exact failure this framework exists to prevent"
        )
    if not isinstance(needs_old_values, bool):
        raise ValueCapabilityRefusal(
            f"needs_old_values={needs_old_values!r}: the requirement "
            f"must be a real bool -- Python's 1 is truthy, so a truthy "
            f"int read as a requirement would launder an unmeasured "
            f"need into a declared one; 1 is not True"
        )
    consumed = ["granularity"]
    if required_granularity == "token" and capabilities.granularity == "sequence":
        raise ValueCapabilityRefusal(
            f"{origin} offers sequence-granularity values but "
            f"required_granularity='token': GAE over per-token values "
            f"cannot be computed from one number per row, and silently "
            f"broadcasting the sequence value across tokens would be a "
            f"fabricated per-token measurement -- refusing at setup "
            f"costs nothing; discovering the gap inside the inner "
            f"epoch loop costs a run"
        )
    if needs_old_values:
        consumed.append("reports_old_values")
        if not capabilities.reports_old_values:
            raise ValueCapabilityRefusal(
                f"{origin} does not report old values but "
                f"needs_old_values=True: the PPO2 value clip needs "
                f"the rollout-time estimate as its clipping "
                f"reference, and silently disabling the clip is not "
                f"a smaller algorithm, it is a different one running "
                f"under the clipped name -- refuse instead"
            )
    return tuple(sorted(consumed))


def verify_estimates(
    estimates: object,
    batch: ExperienceBatch,
    *,
    mask_column: str,
    origin: str = "<value-head>",
) -> int:
    """The MEASUREMENT half of the handshake, run on returned estimates.

    ``capabilities()`` is a CLAIM and this function is the measurement
    that tests its delivery, one row of estimates per batch row, one
    estimate per mask position.

    WHAT IS CLAIMED on success: the returned ``int`` is the number of
    SUPERVISED positions -- positions the mask keeps -- that the
    estimates cover with a coercible, finite value. That count, not
    the row count and not a capabilities claim, is the denominator any
    downstream verdict about this estimate must use.

    WHAT IS NOT CLAIMED: anything about estimate QUALITY -- a finite
    number at a supervised position is present, not correct; anything
    about MASKED positions -- a masked position is not measured, so
    its value is not graded: it is never coerced, never compared, and
    its contents sit in no denominator this check reports; and
    anything about the mask itself, whose entries are read by
    truthiness and not otherwise graded.

    Five refusals. A ``mask_column`` the batch does not carry is
    named: without the mask no position can be told measured from
    unmeasured. A non-iterable ``estimates`` is named with its type.
    A row count that disagrees with the mask column is named with
    BOTH counts. A row whose length disagrees with its mask row is
    named with its row index and both lengths. A non-finite or
    non-coercible value at a SUPERVISED position is named with its
    row index, its position, and its repr. And a batch with zero
    supervised positions is refused rather than returned as 0:
    nothing was measured, and returning 0 would hand the caller an
    empty denominator to divide by -- UNMEASURED is not zero.
    """
    if mask_column not in batch.columns:
        raise ValueEstimateRefusal(
            f"mask_column={mask_column!r} is absent from the batch "
            f"passed with estimates from {origin}: the batch carries "
            f"{tuple(batch.columns)}; the mask separates measured "
            f"positions from unmeasured ones, and without it no "
            f"position can be graded"
        )
    mask_rows = batch.columns[mask_column]
    row_count = len(mask_rows)
    # `estimates` is typed `object` deliberately: this function GRADES an
    # untrusted return value, so it must not presuppose the shape it is
    # checking for. The cast states that presupposition for the typechecker
    # only, and the TypeError arm is what actually enforces it at runtime.
    rows: tuple[object, ...]
    try:
        rows = tuple(cast(Iterable[object], estimates))
    except TypeError as exc:
        raise ValueEstimateRefusal(
            f"estimates from {origin} is of type "
            f"{type(estimates).__name__!r}, which is not iterable: a "
            f"value head must return one row of estimates per batch "
            f"row, and a non-iterable delivers none"
        ) from exc
    if len(rows) != row_count:
        raise ValueEstimateRefusal(
            f"{origin} returned {len(rows)} estimate rows but mask "
            f"column {mask_column!r} has {row_count} rows: one "
            f"estimate row per batch row, and a count that disagrees "
            f"with the batch is an estimate of a different batch"
        )
    supervised = 0
    for row_index, (row, mask_row) in enumerate(zip(rows, mask_rows, strict=True)):
        mask_length = len(mask_row)
        # Same shape as the cast above: the annotation states for the
        # typechecker what the guard below is about to enforce at runtime. A
        # row that is not a sized, indexable sequence raises TypeError here,
        # and that arm refuses rather than counting it.
        sized_row = cast(Sequence[Any], row)
        try:
            row_length = len(sized_row)
        except TypeError as exc:
            raise ValueEstimateRefusal(
                f"estimates[{row_index}] from {origin} is not a sized "
                f"row while mask column {mask_column!r} row "
                f"{row_index} has {mask_length} positions: an "
                f"estimate row with no length covers nothing"
            ) from exc
        if row_length != mask_length:
            raise ValueEstimateRefusal(
                f"estimates[{row_index}] has {row_length} positions "
                f"but mask column {mask_column!r} row {row_index} "
                f"has {mask_length}: one estimate per mask position "
                f"is the shape this check counts over, and a row of "
                f"a different length covers different positions"
            )
        for position in range(mask_length):
            if not mask_row[position]:
                continue
            raw = sized_row[position]
            if isinstance(raw, bool):
                raise ValueEstimateRefusal(
                    f"estimates[{row_index}][{position}]={raw!r}: "
                    f"True is not a value estimate -- bool is "
                    f"checked first because isinstance(True, int) "
                    f"would let 1.0 stand in for a measured value"
                )
            try:
                value = float(raw)
            except (TypeError, ValueError) as exc:
                raise ValueEstimateRefusal(
                    f"estimates[{row_index}][{position}]={raw!r}: "
                    f"the estimate at a supervised position must be "
                    f"readable via float(raw), the settled scalar "
                    f"idiom; a value the idiom cannot read is not "
                    f"a measurement"
                ) from exc
            if not isfinite(value):
                raise ValueEstimateRefusal(
                    f"estimates[{row_index}][{position}]={value!r}: "
                    f"NaN and inf are not value estimates -- every "
                    f"comparison against NaN is False, so a NaN "
                    f"estimate at a supervised position would pass "
                    f"any alarm gate unchecked"
                )
            supervised += 1
    if supervised == 0:
        raise ValueEstimateRefusal(
            f"the batch passed with estimates from {origin} has 0 "
            f"supervised positions in mask column {mask_column!r} "
            f"across {row_count} rows: nothing was measured, and "
            f"returning 0 would hand the caller an empty "
            f"denominator to divide by -- UNMEASURED is not zero"
        )
    return supervised
