"""Stage-3 advantage functions (design section 3.4): rewards to per-token weights.

This module holds the ``AdvantageFn`` protocol and its three implementations;
the batch and loss contracts live in ``interfaces.py``/``losses.py`` and the
model views in ``policy.py``. An advantage function CONSUMES rewards and does
not produce them: who turns completions into rewards is the open scoring edge
recorded in design section 4, and this module deliberately does not invent a
producer.

WHAT THIS MODULE CLAIMS: every implementation reports the denominator it
ACTUALLY used. ``AdvantageResult`` carries ``used`` and ``offered`` side by
side, and ``RewardStats`` summarises the used samples only -- design section
3.4 requires statistics over the samples actually used, not a healthy-looking
average over the set offered.

WHAT THIS MODULE DOES NOT CLAIM: any equivalence with the reference
implementation's advantage bindings. Phase 1 unknown 2 (whether its two
bindings share math) is unread, and this contract is FoundationScale's own.
GAE's zero value baseline is an assumption of THIS binding, not of GAE; the
:class:`GeneralisedAdvantageEstimation` docstring says so again where the
recursion is written down.

No torch, no randomness, no clocks, no I/O: grouping, normalisation and the
backward recursion are pure stdlib arithmetic, and module scope stays
importable by torch-free host tooling.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

__all__ = (
    "AdvantageConfigRefusal",
    "AdvantageFn",
    "AdvantageRefusal",
    "AdvantageResult",
    "GeneralisedAdvantageEstimation",
    "GroupNormalisedAdvantage",
    "LeaveOneOutAdvantage",
    "RewardStats",
)


class AdvantageRefusal(ValueError):
    # Raised when advantage inputs violate the stated contract: ragged
    # lengths, an unconvertible reward or mask entry, a sample with no
    # supervised token. Fail closed; never pad, never coerce.
    pass


class AdvantageConfigRefusal(ValueError):
    # Raised when an advantage function's OWN configuration is incoherent --
    # a group size below 2, a discount outside [0, 1], an empty method name.
    # Distinct from AdvantageRefusal because the offending value belongs to
    # the algorithm, not the data, and is knowable at construction: refusing
    # there costs nothing, while refusing at step zero costs an allocation.
    pass


@dataclass(frozen=True, slots=True)
class RewardStats:
    """Reward statistics over the samples an advantage function ACTUALLY USED.

    WHAT IS CLAIMED: ``count`` is the number of samples these four numbers
    summarise, and it is the denominator the gates price the gradient
    against. WHAT IS NOT CLAIMED: anything about the samples offered but not
    used -- the used/offered gap stays on :class:`AdvantageResult`, beside
    this object, where a reader can see it instead of finding it averaged
    away.

    There is NO zero-sample ``RewardStats``: nothing used means nothing
    measured, an UNMEASURED state that must refuse rather than report zeros
    as if they were a measurement. Construction with ``count < 1`` is refused
    here and :meth:`over` refuses an empty sequence, so the state cannot
    exist through either door.

    NOT :class:`foundationscale.gates.objective_gates.RewardStats`, which
    shares this name and is a DIFFERENT contract. The two diverge on three
    axes and the divergence is deliberate (finding #327, which is #222's
    shape):

    * DENOMINATOR SCOPE. This record summarises the samples ACTUALLY USED.
      The gate one summarises whatever the caller inspected at the gate
      point, a denominator this class cannot see.
    * ZERO-COUNT ADMISSIBILITY. This record refuses ``count < 1``. The gate
      one validates nothing on purpose, because the gate above it exists to
      catch impossible aggregates and its MUST_FIRE fixtures must be able to
      construct them. Unifying the two in EITHER direction breaks one of the
      two contracts: adopting this validation disarms that gate's fixtures,
      and dropping it lets a zeroed summary claim a denominator of nothing.
    * DIVISOR PROVENANCE. :meth:`over` computes ``std`` and so names it the
      POPULATION deviation. The gate one receives ``std`` from a caller and
      makes no such claim.

    The field vocabularies are disjoint (``count``/``minimum``/``maximum``
    here, ``n``/``min``/``max`` there) so substituting one for the other
    fails loudly at construction. That looseness is the protection, not an
    inconsistency to tidy away; the controls that pin it live at the foot of
    ``tests/rl/test_advantage.py``.
    """

    count: int
    mean: float
    std: float
    minimum: float
    maximum: float

    def __post_init__(self) -> None:
        # A zero-count RewardStats would report a healthy-looking summary
        # over a denominator of nothing; refuse rather than let the value
        # object carry a claim about samples that do not exist.
        if self.count < 1:
            raise AdvantageRefusal(
                f"count={self.count}: reward statistics over zero samples "
                f"claim a denominator of nothing; nothing used is an "
                f"UNMEASURED state, never a zeroed pass"
            )

    @classmethod
    def over(cls, values: Sequence[float]) -> RewardStats:
        """Summarise the samples ACTUALLY USED into one statistics record.

        ``std`` is the POPULATION standard deviation (divisor ``n``, not
        ``n - 1``). The two differ and a reader will assume one, so it is
        named here: these numbers describe the measured set itself, they are
        not an estimate of a wider population, and the unbiased divisor would
        be a claim about samples this record never saw.

        An EMPTY sequence is refused: nothing used means nothing measured.
        """
        cleaned: list[float] = []
        for position, raw in enumerate(values):
            try:
                value = float(raw)
            except (TypeError, ValueError) as exc:
                raise AdvantageRefusal(
                    f"value at position {position} is {raw!r}, which does "
                    f"not convert to a scalar float ({type(raw).__name__}); "
                    f"reward values must be finite scalars"
                ) from exc
            if not math.isfinite(value):
                raise AdvantageRefusal(
                    f"value at position {position} is {raw!r}, which is not "
                    f"finite; one non-finite reading would poison the mean "
                    f"and std the gate prices the gradient against"
                )
            cleaned.append(value)
        if not cleaned:
            raise AdvantageRefusal(
                "RewardStats.over got an empty sequence: nothing used means "
                "nothing measured -- a mean over zero samples is the all([]) "
                "shape, an UNMEASURED state, never a zeroed pass"
            )
        count = len(cleaned)
        mean = sum(cleaned) / count
        variance = sum((value - mean) ** 2 for value in cleaned) / count
        return cls(
            count=count,
            mean=mean,
            std=math.sqrt(variance),
            minimum=min(cleaned),
            maximum=max(cleaned),
        )


@dataclass(frozen=True, slots=True)
class AdvantageResult:
    """The per-token weights plus the honest used/offered denominator.

    WHAT IS CLAIMED: exactly ``used`` samples carry weight rows, ``rows``
    names WHICH of the ``offered`` samples those are, the reward statistics
    summarise exactly those ``used`` samples, and ``method`` names the
    advantage function that produced them. WHAT IS NOT CLAIMED: that the
    advantage function used everything it was offered. An excluded sample --
    a degenerate GRPO group, a singleton RLOO group -- leaves NO row behind:
    if it left a row of zeros, that row would sit in this result reading as
    a measured 0.0 and the used/offered gap would be laundered into the
    weights. The checks below make the laundering unrepresentable: a result
    cannot claim a denominator it did not use.

    ``rows`` is what makes the exclusion actionable rather than merely
    honest. Because an excluded sample leaves no row, ``weights`` is
    COMPACTED: with ``prompt_ids = ("a", "b", "a", "b")`` and group ``b``
    degenerate, the two surviving rows are offered-rows 0 and 2, not 0 and
    1. A caller that zipped ``weights`` against its batch positionally would
    apply group ``a``'s advantages to rows 0 and 1 and never raise -- the
    gradient would land on the wrong sequence, silently. A count alone
    cannot prevent that; naming the members can. So a result reports both
    how many samples it used and which ones.
    """

    weights: tuple[tuple[float, ...], ...]
    rewards: RewardStats
    used: int
    offered: int
    method: str
    rows: tuple[int, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.method, str) or not self.method:
            raise AdvantageRefusal(
                f"method={self.method!r}: the producing method must be a "
                f"non-empty string; absence of a name is not a name"
            )
        if self.used > self.offered:
            raise AdvantageRefusal(
                f"used={self.used} but offered={self.offered}: a result "
                f"cannot report using more samples than it was offered"
            )
        if len(self.weights) != self.used:
            raise AdvantageRefusal(
                f"len(weights)={len(self.weights)} but used={self.used}: "
                f"every used sample must carry exactly one weight row, and "
                f"no excluded sample may leave one behind"
            )
        if self.rewards.count != self.used:
            raise AdvantageRefusal(
                f"rewards.count={self.rewards.count} but used={self.used}: "
                f"the reward statistics must summarise exactly the used "
                f"samples, never the offered ones"
            )
        rows = tuple(self.rows)
        if len(rows) != self.used:
            raise AdvantageRefusal(
                f"len(rows)={len(rows)} but used={self.used}: a result must "
                f"name exactly as many offered-row indices as it claims to "
                f"have used, or the weights cannot be attributed to a batch"
            )
        for position, index in enumerate(rows):
            if not isinstance(index, int) or isinstance(index, bool):
                raise AdvantageRefusal(
                    f"rows[{position}]={index!r}: an offered-row index must "
                    f"be an int, not {type(index).__name__}"
                )
            if not 0 <= index < self.offered:
                raise AdvantageRefusal(
                    f"rows[{position}]={index} is outside the offered range "
                    f"[0, {self.offered}): a result cannot name a sample it "
                    f"was never offered"
                )
            if position and index <= rows[position - 1]:
                # Strictly increasing, so `rows` is a SUBSEQUENCE of the
                # offered batch rather than a permutation of it. A caller may
                # therefore zip weights against rows without also having to
                # trust that the producer kept batch order.
                raise AdvantageRefusal(
                    f"rows[{position}]={index} does not follow "
                    f"rows[{position - 1}]={rows[position - 1]}: offered-row "
                    f"indices must be strictly increasing and distinct"
                )
        object.__setattr__(self, "rows", rows)
        object.__setattr__(self, "weights", tuple(tuple(row) for row in self.weights))


@runtime_checkable
class AdvantageFn(Protocol):
    """Maps per-sample rewards to per-token weights (design section 3.4).

    WHAT IS CLAIMED: ``compute`` receives ``prompt_ids`` (the grouping key),
    one scalar ``rewards`` entry per sample, and a per-row supervision
    ``mask``, and returns an :class:`AdvantageResult` whose ``used`` /
    ``offered`` pair reports the denominator the gradient is priced against.
    WHAT IS NOT CLAIMED: any equivalence with the reference implementation's
    advantage shape. This is a real Protocol where Phase 1 pattern 6 found
    only convention in NeMo-RL; how its two bindings relate (unknown 2) is
    unverified, and FoundationScale specifies its own signature regardless.

    The arguments are keyword-only so a caller cannot swap ``rewards`` and
    ``mask`` positionally and have the error surface three samples later as
    a fractional-mask refusal against the wrong row.
    """

    def compute(
        self,
        *,
        prompt_ids: Sequence[str],
        rewards: Sequence[float],
        mask: Sequence[Sequence[int]],
    ) -> AdvantageResult: ...


def _require_name(field_name: str, value: Any) -> None:
    # Every name-ish config field refuses a non-name at CONSTRUCTION, with
    # the field name in the message: refusing here names the field that
    # produced it rather than the gate or manifest that saw it first.
    if not isinstance(value, str) or not value:
        raise AdvantageConfigRefusal(
            f"{field_name}={value!r}: the method name must be a non-empty "
            f"string; absence of a name is not a name"
        )


def _require_min_group_size(value: Any) -> None:
    # Below 2 there is no baseline to normalise or leave out, so the field
    # is checked at construction where the refusal costs nothing. A bool is
    # an int in Python, and True == 1 fails the size check on its own.
    if not isinstance(value, int) or value < 2:
        raise AdvantageConfigRefusal(
            f"min_group_size={value!r}: a group must contain at least 2 "
            f"samples before it yields a baseline; below that there is "
            f"nothing to compare a sample against"
        )


def _as_tuple(value: Any, name: str) -> tuple[Any, ...]:
    try:
        return tuple(value)
    except TypeError as exc:
        raise AdvantageRefusal(
            f"{name} is {value!r}, which is not iterable "
            f"({type(value).__name__}); a sequence of per-sample values is "
            f"required"
        ) from exc


def _coerce_reward(raw: Any, row: int) -> float:
    # A reward is one scalar per sample, so its only coordinates are the row
    # it belongs to: the message names row and position alike, both being
    # that row. A non-finite reading is refused outright for the same reason
    # losses.py refuses non-finite reference scores: it would poison exactly
    # one group while the run reads as healthy everywhere else.
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise AdvantageRefusal(
            f"reward at row {row}, position {row} is {raw!r}, which does "
            f"not convert to a scalar float ({type(raw).__name__}); rewards "
            f"must be one finite scalar per sample"
        ) from exc
    if not math.isfinite(value):
        raise AdvantageRefusal(
            f"reward at row {row}, position {row} is {raw!r}, which is not "
            f"finite; rewards must be one finite scalar per sample"
        )
    return value


def _mask_entry(raw: Any, row: int, position: int) -> bool:
    # The bool branch comes FIRST, because isinstance(True, int) is True in
    # Python. Everything after it is admitted by VALUE, via float(), matching
    # the losses.py discipline: a fractional entry is refused rather than
    # read as a partial weight, because admitting one would silently change
    # which positions carry gradient.
    if isinstance(raw, bool):
        return raw
    try:
        value = float(raw)
    except (TypeError, ValueError):
        pass
    else:
        if value in (0.0, 1.0):
            return value == 1.0
    raise AdvantageRefusal(
        f"mask entry at row {row}, position {position} is {raw!r}; "
        f"supervision mask entries must be 0 or 1"
    )


def _checked_rows(
    prompt_ids: Sequence[str],
    rewards: Sequence[float],
    mask: Sequence[Sequence[int]],
) -> tuple[tuple[Any, ...], tuple[float, ...], tuple[tuple[bool, ...], ...]]:
    ids = _as_tuple(prompt_ids, "prompt_ids")
    raw_rewards = _as_tuple(rewards, "rewards")
    raw_masks = _as_tuple(mask, "mask")
    if not (len(ids) == len(raw_rewards) == len(raw_masks)):
        raise AdvantageRefusal(
            f"prompt_ids, rewards and mask disagree on the sample count: "
            f"{len(ids)} prompt ids, {len(raw_rewards)} rewards, "
            f"{len(raw_masks)} mask rows; the three lengths must agree, "
            f"because sample i's reward must land on row i's tokens and a "
            f"silent mismatch would train the wrong row"
        )
    if not ids:
        raise AdvantageRefusal(
            "compute got 0 samples: advantage statistics over zero used "
            "samples claim a denominator of nothing -- the all([]) shape -- "
            "so the result is UNMEASURED, not 0.0"
        )
    cleaned_rewards = tuple(_coerce_reward(raw, row) for row, raw in enumerate(raw_rewards))
    cleaned_masks: list[tuple[bool, ...]] = []
    for row, raw_row in enumerate(raw_masks):
        entries = tuple(
            _mask_entry(raw, row, position)
            for position, raw in enumerate(_as_tuple(raw_row, f"mask row {row}"))
        )
        if not entries:
            raise AdvantageRefusal(
                f"mask row {row} is empty; a sample with no token positions "
                f"has no supervised token to land an advantage on, so it "
                f"contributes no gradient and cannot be counted as used"
            )
        if not any(entries):
            raise AdvantageRefusal(
                f"mask row {row} supervises 0 of {len(entries)} positions; "
                f"a sample with no supervised token contributes no gradient "
                f"and must be refused, not counted as used with zero weights"
            )
        cleaned_masks.append(entries)
    return ids, cleaned_rewards, tuple(cleaned_masks)


def _group_indices(ids: tuple[Any, ...]) -> list[tuple[int, ...]]:
    # First-appearance order is preserved, so two runs over the same batch
    # produce the same grouping and the same result row order.
    groups: dict[Any, list[int]] = {}
    for row, prompt_id in enumerate(ids):
        try:
            bucket = groups.setdefault(prompt_id, [])
        except TypeError as exc:
            raise AdvantageRefusal(
                f"prompt id at row {row} is {prompt_id!r}, which is not "
                f"hashable ({type(prompt_id).__name__}); prompt ids group "
                f"samples, so each must be usable as a grouping key"
            ) from exc
        bucket.append(row)
    return [tuple(indices) for indices in groups.values()]


def _broadcast(advantage: float, mask_row: tuple[bool, ...]) -> tuple[float, ...]:
    # Masked positions get a literal 0.0: they carry no gradient by
    # construction of the mask, which is a different statement from a
    # measured zero advantage at a supervised position.
    return tuple(advantage if entry else 0.0 for entry in mask_row)


def _build_result(
    weights: list[tuple[float, ...]],
    used_rewards: list[float],
    rows: list[int],
    offered: int,
    method: str,
) -> AdvantageResult:
    # RewardStats.over([]) refuses, so an advantage function that excluded
    # EVERY offered sample fails closed here rather than returning a result
    # whose denominator is zero.
    return AdvantageResult(
        weights=tuple(weights),
        rewards=RewardStats.over(tuple(used_rewards)),
        used=len(weights),
        offered=offered,
        method=method,
        rows=tuple(rows),
    )


def _emit_used_rows(
    per_sample: list[float | None],
    cleaned_rewards: tuple[float, ...],
    masks: tuple[tuple[bool, ...], ...],
    method: str,
) -> AdvantageResult:
    weights: list[tuple[float, ...]] = []
    used_rewards: list[float] = []
    rows: list[int] = []
    for row, (advantage, reward) in enumerate(zip(per_sample, cleaned_rewards, strict=True)):
        if advantage is None:
            continue
        weights.append(_broadcast(advantage, masks[row]))
        used_rewards.append(reward)
        # The offered-row index, recorded at the one point where the
        # compaction happens. Recovering it downstream from `weights` alone
        # is impossible, which is the whole reason it is carried.
        rows.append(row)
    return _build_result(weights, used_rewards, rows, len(per_sample), method)


@dataclass(frozen=True)
class GroupNormalisedAdvantage:
    """GRPO-style group normalisation: centre on the group, scale by its std.

    Each sample's group is the set of samples sharing its prompt id. Within
    a group of at least ``min_group_size`` samples with non-zero spread,
    ``advantage = (reward - group_mean) / group_std`` where ``group_std`` is
    the POPULATION std -- the same divisor choice as :meth:`RewardStats.over`,
    named here for the same reason.

    WHAT IS CLAIMED: used samples carry a normalised advantage broadcast
    across their unmasked positions. WHAT IS NOT CLAIMED: anything for the
    samples that leave the result. A group whose std is 0.0 has NO signal:
    its weights would be 0.0 by cancellation, and a zero row is not a
    measurement, so the group is EXCLUDED from ``used`` instead of being
    written as zeros -- the degenerate group stays visible in the
    used/offered gap rather than being laundered into the mean. A group
    smaller than ``min_group_size`` cannot be normalised at all and is
    excluded the same way.
    """

    min_group_size: int = 2
    method_name: str = "GroupNormalisedAdvantage"

    def __post_init__(self) -> None:
        _require_min_group_size(self.min_group_size)
        _require_name("method_name", self.method_name)

    def compute(
        self,
        *,
        prompt_ids: Sequence[str],
        rewards: Sequence[float],
        mask: Sequence[Sequence[int]],
    ) -> AdvantageResult:
        ids, cleaned_rewards, masks = _checked_rows(prompt_ids, rewards, mask)
        per_sample: list[float | None] = [None] * len(ids)
        for indices in _group_indices(ids):
            if len(indices) < self.min_group_size:
                # Too few samples for a baseline. The samples leave the
                # result entirely -- no weight row -- so used < offered is
                # the visible record of the exclusion.
                continue
            group = [cleaned_rewards[i] for i in indices]
            mean = sum(group) / len(group)
            variance = sum((reward - mean) ** 2 for reward in group) / len(group)
            std = math.sqrt(variance)
            if std == 0.0:
                # Zero variance means no relative signal anywhere in the
                # group. The would-be weights are 0.0 by cancellation, which
                # is NOT a measurement, so the samples leave the result
                # entirely and the gap stays in used < offered.
                continue
            for i in indices:
                per_sample[i] = (cleaned_rewards[i] - mean) / std
        return _emit_used_rows(per_sample, cleaned_rewards, masks, self.method_name)


@dataclass(frozen=True)
class LeaveOneOutAdvantage:
    """RLOO-style advantage: each sample's baseline is the mean of the OTHERS.

    Within a group of at least ``min_group_size`` samples sharing a prompt
    id, ``baseline_i = mean(rewards without i)`` and
    ``advantage_i = rewards_i - baseline_i``.

    WHAT IS CLAIMED: used samples carry a leave-one-out advantage broadcast
    across their unmasked positions, and a group too small to have a
    baseline leaves the result entirely (a leave-one-out mean over zero
    other samples is not a number). WHAT IS NOT CLAIMED: the GRPO exclusion.
    Unlike group normalisation, a zero-variance group is NOT excluded here:
    each sample is exactly equal to its leave-one-out baseline, so its
    advantage of 0.0 is a genuinely measured reading -- "this sample is
    exactly average among its peers" -- and measured zeros stay in the
    denominator.
    """

    min_group_size: int = 2
    method_name: str = "LeaveOneOutAdvantage"

    def __post_init__(self) -> None:
        _require_min_group_size(self.min_group_size)
        _require_name("method_name", self.method_name)

    def compute(
        self,
        *,
        prompt_ids: Sequence[str],
        rewards: Sequence[float],
        mask: Sequence[Sequence[int]],
    ) -> AdvantageResult:
        ids, cleaned_rewards, masks = _checked_rows(prompt_ids, rewards, mask)
        per_sample: list[float | None] = [None] * len(ids)
        for indices in _group_indices(ids):
            if len(indices) < self.min_group_size:
                # One sample has no OTHERS to average into a baseline; it
                # leaves the result rather than dividing by group_size - 1.
                continue
            total = sum(cleaned_rewards[i] for i in indices)
            group_size = len(indices)
            for i in indices:
                baseline = (total - cleaned_rewards[i]) / (group_size - 1)
                per_sample[i] = cleaned_rewards[i] - baseline
        return _emit_used_rows(per_sample, cleaned_rewards, masks, self.method_name)


@dataclass(frozen=True)
class GeneralisedAdvantageEstimation:
    """GAE over a terminal per-sample reward, with a ZERO VALUE BASELINE.

    WHAT IS CLAIMED: the mask is the token layout; the scalar ``rewards[i]``
    is a TERMINAL reward landing on the last unmasked position of row ``i``;
    deltas are computed as ``delta_t = r_t + gamma * V(t+1) - V(t)`` with
    ``V == 0`` everywhere, so ``delta_t == r_t`` and the backward recursion
    over each row's unmasked positions is
    ``A_t = r_t + gamma * lam * A_{t+1}``, with a literal 0.0 written at
    masked positions. WHAT IS NOT CLAIMED: that the zero value baseline is a
    property of GAE. It is an assumption of THIS binding: a binding with a
    learned value function would need the value estimates as input, and this
    contract does not carry them. ``prompt_ids`` participates only in the
    length check -- the recursion needs no grouping.

    Every offered sample is used: there is no grouping and no exclusion, so
    ``used == offered`` by construction, not by luck.
    """

    gamma: float = 1.0
    lam: float = 0.95
    method_name: str = "GeneralisedAdvantageEstimation"

    def __post_init__(self) -> None:
        _require_name("method_name", self.method_name)
        for field_name, value in (("gamma", self.gamma), ("lam", self.lam)):
            # Outside [0, 1] a "discount" amplifies instead of discounting
            # (or is not a number at all), and the recursion carries the
            # error into every position's weight. The range check also
            # refuses NaN, because every comparison against NaN is False.
            if not isinstance(value, (int, float)) or not 0.0 <= value <= 1.0:
                raise AdvantageConfigRefusal(
                    f"{field_name}={value!r}: a discount factor must be a "
                    f"finite number in [0.0, 1.0]"
                )

    def compute(
        self,
        *,
        prompt_ids: Sequence[str],
        rewards: Sequence[float],
        mask: Sequence[Sequence[int]],
    ) -> AdvantageResult:
        ids, cleaned_rewards, masks = _checked_rows(prompt_ids, rewards, mask)
        factor = float(self.gamma) * float(self.lam)
        weights: list[tuple[float, ...]] = []
        for row, mask_row in enumerate(masks):
            row_weights = [0.0] * len(mask_row)
            unmasked = [position for position, entry in enumerate(mask_row) if entry]
            next_advantage = 0.0
            for step, position in enumerate(reversed(unmasked)):
                # The terminal reward lands on the LAST unmasked position,
                # which reversed() visits FIRST; every earlier position sees
                # r_t == 0.0. The chain runs over unmasked positions only:
                # a masked position is skipped by the recursion AND written
                # 0.0, because it carries no gradient.
                r_t = cleaned_rewards[row] if step == 0 else 0.0
                advantage_t = r_t + factor * next_advantage
                row_weights[position] = advantage_t
                next_advantage = advantage_t
            weights.append(tuple(row_weights))
        # GAE excludes nothing: every offered row gets a weight row, so the
        # index list is the identity and used == offered. It is still stated
        # rather than defaulted, because a caller must not have to know which
        # advantage functions compact and which do not.
        return _build_result(
            weights,
            list(cleaned_rewards),
            list(range(len(ids))),
            len(ids),
            self.method_name,
        )
