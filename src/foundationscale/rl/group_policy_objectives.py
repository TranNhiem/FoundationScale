"""Group-relative policy objectives: GSPO, Dr.GRPO, and DAPO arithmetic.

This module holds the three LOSS OBJECTS; the protocol they satisfy
(``SequenceObjective``), the family binding (``SequencePolicyAlgorithm``), and
the three factories live in the sibling ``group_policy.py``. The arithmetic
here is recognisably the same family as :class:`GRPOPolicyLoss`: the same
guards, the same abstention behaviour, and the same treatment of an empty or
fully-masked batch -- what differs is exactly the two declarations the axes
matrix measures as load-bearing, ``ratio_scope`` and ``reduction``.

WHAT IS CLAIMED: each objective computes precisely the ratio scope, clip
bounds, advantage estimator, KL weighting, and reduction its class docstring
states, over exactly the rows its estimator returns; each refuses the same
mis-sized, non-finite, or unsupervised batches GRPO refuses, with both
sides' counts named; and each declares at least one component by
construction, so a ``declaration()`` that decomposes into zero components is
unreachable.

WHAT IS NOT CLAIMED: DAPO's dynamic sampling -- the refill half of the
zero-variance-group filter -- is NOT implemented. The drop half already
happens in :class:`GroupNormalisedAdvantage` (``used < offered``); the
refill half is a rollout-loop behaviour and this package contains no rollout
loop to host it. ``DAPOLoss.expects_dynamic_sampling`` declares the
expectation; NOTHING in this module enforces it, and treating a wired DAPO
run as "dynamic sampling applied" is a misreading, not a feature. Overlong
reward shaping is likewise rollout-side and unimplemented. No convergence,
benchmark, or equivalence-to-the-paper claim is made: none of these
objectives has ever been trained here.

On the standing open question (docs/RL_ALGORITHMS.md:137) -- this module's
answer, stated once and as an ARGUMENT, not a measurement: no third
``ratio_scope`` member and no companion flag. A non-length-normalised
sequence ratio is ``exp(SUM_t log_ratio_t)``; over any response long enough
to matter that value leaves every usable floating-point range for any
non-trivial policy shift, so it is not a geometry anything binds. Length
normalisation is what makes a sequence ratio EXIST, not an option on one.
The quantity that genuinely separates these three objectives is the
denominator, and that is carried by the separate ``reduction`` property.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Literal

from foundationscale.gates.objective_gates import LossComponent
from foundationscale.rl.advantage import (
    AdvantageFn,
    AdvantageResult,
    CentredAdvantage,
    GroupNormalisedAdvantage,
)
from foundationscale.rl.algorithm import AlgorithmSemantics
from foundationscale.rl.interfaces import (
    BatchRefusal,
    ExperienceBatch,
    ForwardFn,
    LossConfigRefusal,
    LossDeclaration,
    LossOutput,
    SupervisionRefusal,
)

__all__ = (
    "DAPOLoss",
    "DrGRPOLoss",
    "GSPOLoss",
)


def _checked_name(field_name: str, value: Any) -> None:
    if not isinstance(value, str) or not value:
        raise LossConfigRefusal(
            f"{field_name}={value!r}: column and component names must be "
            f"non-empty strings; absence of a name is not a name"
        )


def _outer_sequence(
    value: Any,
    *,
    field_name: str,
    refusal: type[ValueError],
) -> tuple[Any, ...]:
    try:
        return tuple(value)
    except TypeError as exc:
        raise refusal(
            f"field {field_name}={value!r}: 1 of 1 required inputs was not "
            f"iterable ({type(value).__name__}); one outer entry per batch "
            f"row is required"
        ) from exc


def _row_sequence(
    value: Any,
    *,
    field_name: str,
    row: int,
) -> tuple[Any, ...]:
    try:
        return tuple(value)
    except TypeError as exc:
        raise BatchRefusal(
            f"field {field_name} row {row}={value!r}: 1 of 1 rows was not "
            f"iterable ({type(value).__name__}); a group-relative "
            f"objective requires one entry per token position"
        ) from exc


def _mask_entry(raw: Any, row: int, position: int) -> bool:
    if isinstance(raw, bool):
        return raw
    try:
        value = float(raw)
    except (TypeError, ValueError):
        pass
    else:
        if value in (0.0, 1.0):
            return value == 1.0
    raise BatchRefusal(
        f"mask entry at row {row}, position {position} is {raw!r}; "
        f"supervision mask entries must be 0 or 1"
    )


def _target_logprob(raw: Any, field_name: str, row: int, position: int) -> float:
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise BatchRefusal(
            f"field {field_name} at row {row}, position {position} does "
            f"not convert to a scalar float ({type(raw).__name__}); a "
            f"group-relative objective requires one finite target "
            f"log-probability per token"
        ) from exc
    if not math.isfinite(value):
        raise BatchRefusal(
            f"field {field_name} at row {row}, position {position} is "
            f"{raw!r}, which is not finite; a non-finite token reading "
            f"would poison the importance ratio for its whole response"
        )
    return value


def _advantage_weight(raw: Any, row: int, position: int) -> float:
    if isinstance(raw, bool):
        raise BatchRefusal(
            f"advantage weight at row {row}, position {position} is "
            f"{raw!r}; a bool is not a measured token weight"
        )
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise BatchRefusal(
            f"advantage weight at row {row}, position {position} does "
            f"not convert to a scalar float ({type(raw).__name__})"
        ) from exc
    if not math.isfinite(value):
        raise BatchRefusal(
            f"advantage weight at row {row}, position {position} is "
            f"{raw!r}, which is not finite; the objective cannot price a "
            f"non-finite advantage as gradient"
        )
    return value


@dataclass(frozen=True, slots=True)
class _PricedRows:
    """Validated per-position readings over exactly the estimator's used rows.

    WHAT IS CLAIMED: every tuple here is indexed by BATCH row (not result
    row), covers exactly ``advantage.rows``, and every leaf is a finite
    float or a bool mask entry; ``reference_rows`` is ``None`` iff the
    objective is configured reference-free, never as an empty tuple.

    WHAT IS NOT CLAIMED: that any token is supervised, or that the
    resulting loss is finite for every conceivable batch.
    """

    advantage: AdvantageResult
    batch_rows: tuple[int, ...]
    mask_rows: tuple[tuple[bool, ...], ...]
    weight_rows: tuple[tuple[float, ...], ...]
    current_rows: tuple[tuple[float, ...], ...]
    old_rows: tuple[tuple[float, ...], ...]
    reference_rows: tuple[tuple[float, ...], ...] | None


def _price_rows(
    *,
    forward_fn: ForwardFn,
    batch: ExperienceBatch,
    advantage_fn: AdvantageFn,
    group_size: int,
    prompt_id_column: str,
    reward_column: str,
    mask_column: str,
    old_logprob_column: str,
    reference_logprob_column: str | None,
    origin: str,
) -> _PricedRows:
    """Validate one batch and price its token readings for one objective.

    This is the shared GRPO pipeline minus the reduction: schema presence,
    exact-K group composition, estimator invocation, forward row alignment,
    and per-position conversion are identical for all three objectives --
    differing pipelines would make the axes matrix's "same family" claim a
    statement about code the code does not contain.
    """
    required = [prompt_id_column, reward_column, mask_column, old_logprob_column]
    if reference_logprob_column is not None:
        required.append(reference_logprob_column)
    missing = tuple(name for name in required if name not in batch.columns)
    if missing:
        raise BatchRefusal(
            f"field columns: {len(missing)} of {len(required)} required "
            f"inputs absent for {origin}: {{{', '.join(missing)}}}"
        )
    batch_rows = len(batch)
    if batch_rows == 0:
        raise BatchRefusal(
            f"field rows: 0 of at least 1 required batch rows were "
            f"supplied for {origin}; an empty batch is an unmeasured "
            f"objective, never 0.0"
        )

    prompt_ids = _outer_sequence(
        batch.column(prompt_id_column),
        field_name=prompt_id_column,
        refusal=BatchRefusal,
    )
    rewards = _outer_sequence(
        batch.column(reward_column),
        field_name=reward_column,
        refusal=BatchRefusal,
    )
    mask_outer = _outer_sequence(
        batch.column(mask_column),
        field_name=mask_column,
        refusal=BatchRefusal,
    )
    old_outer = _outer_sequence(
        batch.column(old_logprob_column),
        field_name=old_logprob_column,
        refusal=BatchRefusal,
    )
    reference_outer: tuple[Any, ...] | None = None
    if reference_logprob_column is not None:
        reference_outer = _outer_sequence(
            batch.column(reference_logprob_column),
            field_name=reference_logprob_column,
            refusal=BatchRefusal,
        )

    counts = {
        prompt_id_column: len(prompt_ids),
        reward_column: len(rewards),
        mask_column: len(mask_outer),
        old_logprob_column: len(old_outer),
    }
    if reference_logprob_column is not None and reference_outer is not None:
        counts[reference_logprob_column] = len(reference_outer)
    if len(set(counts.values())) != 1 or next(iter(counts.values())) != batch_rows:
        raise BatchRefusal(
            f"field row counts {tuple(counts.items())} disagree with "
            f"{batch_rows} batch rows for {origin}; all supplied columns "
            f"must carry one value per offered row"
        )

    groups: dict[Any, list[int]] = {}
    for row, prompt_id in enumerate(prompt_ids):
        try:
            groups.setdefault(prompt_id, []).append(row)
        except TypeError as exc:
            raise BatchRefusal(
                f"prompt id at row {row} is {prompt_id!r}, which is "
                f"not hashable ({type(prompt_id).__name__}); prompt ids "
                f"must support the same grouping the advantage function "
                f"performs"
            ) from exc
    for prompt_id, rows in groups.items():
        if len(rows) != group_size:
            raise BatchRefusal(
                f"field {prompt_id_column}: prompt {prompt_id!r} carries "
                f"{len(rows)} of {batch_rows} batch rows but {origin} "
                f"declares group_size={group_size}; 1 prompt group "
                f"disagrees with 1 declared K"
            )

    advantage = advantage_fn.compute(
        prompt_ids=prompt_ids,
        rewards=rewards,
        mask=mask_outer,
    )
    if advantage.offered != batch_rows:
        raise BatchRefusal(
            f"AdvantageResult.offered={advantage.offered} but the batch "
            f"contains {batch_rows} rows; {advantage.offered} of "
            f"{batch_rows} offered rows cannot anchor a {origin} step"
        )

    current_outer = _outer_sequence(
        forward_fn(batch),
        field_name="forward_fn(batch)",
        refusal=BatchRefusal,
    )
    if len(current_outer) != batch_rows:
        raise BatchRefusal(
            f"forward_fn returned {len(current_outer)} rows for a batch "
            f"of {batch_rows} rows; one token row per batch row is "
            f"required"
        )

    used_batch_rows: list[int] = []
    mask_rows: list[tuple[bool, ...]] = []
    weight_rows: list[tuple[float, ...]] = []
    current_rows: list[tuple[float, ...]] = []
    old_rows: list[tuple[float, ...]] = []
    reference_rows: list[tuple[float, ...]] = []
    for result_row, batch_row in enumerate(advantage.rows):
        mask = tuple(
            _mask_entry(raw, batch_row, position)
            for position, raw in enumerate(
                _row_sequence(mask_outer[batch_row], field_name=mask_column, row=batch_row)
            )
        )
        weights = tuple(
            _advantage_weight(raw, batch_row, position)
            for position, raw in enumerate(
                _row_sequence(
                    advantage.weights[result_row],
                    field_name="advantage weights",
                    row=batch_row,
                )
            )
        )
        current = tuple(
            _target_logprob(raw, "current_logprobs", batch_row, position)
            for position, raw in enumerate(
                _row_sequence(
                    current_outer[batch_row],
                    field_name="forward_fn output",
                    row=batch_row,
                )
            )
        )
        old = tuple(
            _target_logprob(raw, old_logprob_column, batch_row, position)
            for position, raw in enumerate(
                _row_sequence(
                    old_outer[batch_row],
                    field_name=old_logprob_column,
                    row=batch_row,
                )
            )
        )
        lengths = [len(mask), len(weights), len(current), len(old)]
        reference: tuple[float, ...] = ()
        if reference_outer is not None:
            reference = tuple(
                _target_logprob(
                    raw,
                    reference_logprob_column or "reference_logprobs",
                    batch_row,
                    position,
                )
                for position, raw in enumerate(
                    _row_sequence(
                        reference_outer[batch_row],
                        field_name=reference_logprob_column or "reference_logprobs",
                        row=batch_row,
                    )
                )
            )
            lengths.append(len(reference))
        if len(set(lengths)) != 1:
            demand = len(lengths)
            raise BatchRefusal(
                f"row {batch_row}: mask/advantage/current/old(/reference) "
                f"lengths are {tuple(lengths)}; all {demand} token "
                f"denominators must be equal before one token ratio can "
                f"be attributed"
            )
        used_batch_rows.append(batch_row)
        mask_rows.append(mask)
        weight_rows.append(weights)
        current_rows.append(current)
        old_rows.append(old)
        reference_rows.append(reference)

    return _PricedRows(
        advantage=advantage,
        batch_rows=tuple(used_batch_rows),
        mask_rows=tuple(mask_rows),
        weight_rows=tuple(weight_rows),
        current_rows=tuple(current_rows),
        old_rows=tuple(old_rows),
        reference_rows=(tuple(reference_rows) if reference_outer is not None else None),
    )


def _ratio(log_ratio: float, batch_row: int, position: int | None, origin: str) -> float:
    """Exponentiate one log-ratio with the shard's shared guards.

    ``position is None`` marks the single sequence-level reading (GSPO's
    geometric mean): the guard is the same, the message names the response
    rather than the token.
    """
    try:
        ratio = math.exp(log_ratio)
    except OverflowError as exc:
        where = f"position {position}" if position is not None else "the response"
        raise BatchRefusal(
            f"row {batch_row}, {where}: ratio exp({log_ratio!r}) is not "
            f"representable for {origin}; 1 of 1 ratios failed before "
            f"clipping"
        ) from exc
    if ratio <= 0.0 or not math.isfinite(ratio):
        where = f"position {position}" if position is not None else "the response"
        raise BatchRefusal(
            f"row {batch_row}, {where}: ratio {ratio!r} is not a positive "
            f"finite value for {origin}; a silently underflowed ratio "
            f"would manufacture a measured-looking zero contribution"
        )
    return ratio


def _clipped_term(ratio: float, weight: float, low: float, high: float) -> float:
    clipped = min(max(ratio, low), high)
    return min(ratio * weight, clipped * weight)


def _kl_component_terms(
    priced: _PricedRows,
    *,
    origin: str,
) -> tuple[float, int]:
    """Sum the k3 reference term over supervised tokens of the used rows.

    The k3 estimator is token-relative wherever these three objectives
    carry it; only RATIO scope and the policy-side denominator distinguish
    the objectives, and carrying a whole-sequence KL beside GSPO's
    sequence ratio would introduce a third unmeasured axis this framework
    has no axes-matrix cell for.
    """
    reference_rows = priced.reference_rows
    assert reference_rows is not None  # callers gate on configuration
    total = 0.0
    supervised = 0
    for index, batch_row in enumerate(priced.batch_rows):
        mask = priced.mask_rows[index]
        current = priced.current_rows[index]
        reference = reference_rows[index]
        for position, is_supervised in enumerate(mask):
            if not is_supervised:
                continue
            log_reference_ratio = reference[position] - current[position]
            try:
                k3 = math.expm1(log_reference_ratio) - log_reference_ratio
            except OverflowError as exc:
                raise BatchRefusal(
                    f"row {batch_row}, position {position}: k3 "
                    f"projection exp({log_reference_ratio!r}) is not "
                    f"representable for {origin}; 1 of 1 reference-token "
                    f"readings failed"
                ) from exc
            if -1.0e-15 < k3 < 0.0:
                k3 = 0.0
            if k3 < 0.0 or not math.isfinite(k3):
                raise BatchRefusal(
                    f"row {batch_row}, position {position}: k3 value "
                    f"{k3!r} is not a finite nonnegative measurement"
                )
            total += k3
            supervised += 1
    return total, supervised


def _validate_common(
    *,
    group_size: int,
    clip_low: float,
    clip_high: float,
    weight: float,
    kl_weight: float | None,
    reference_logprob_column: str | None,
    origin: str,
) -> None:
    """Shared construction validation for all three objectives.

    ``kl_weight is None`` means the objective carries no KL term at all
    (DAPO, structurally); a caller-visible float means the term is
    optional and gated on ``kl_weight != 0.0`` -- both readings of
    Dr.GRPO's KL are then representable with the standard recipe as the
    default.
    """
    if isinstance(group_size, bool):
        raise LossConfigRefusal(
            f"group_size={group_size!r}: True is not a {origin} group "
            f"size; a group size is an integer of at least 2"
        )
    if not isinstance(group_size, int) or group_size < 2:
        raise LossConfigRefusal(
            f"group_size={group_size!r}: {origin} requires a group of at "
            f"least 2 samples; below that there is no within-group "
            f"baseline to estimate"
        )
    for field_name, value in (("clip_low", clip_low), ("clip_high", clip_high)):
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise LossConfigRefusal(
                f"{field_name}={value!r}: 1 of 2 clip bounds for {origin} "
                f"is not a number; a clip bound must be a finite float"
            )
        if not math.isfinite(value):
            raise LossConfigRefusal(
                f"{field_name}={value!r}: 1 of 2 clip bounds for {origin} is not finite"
            )
    if clip_low <= 0.0 or not clip_low < clip_high:
        raise LossConfigRefusal(
            f"clip_bounds=({clip_low!r}, {clip_high!r}): the lower bound "
            f"must be positive and the two declared bounds must be "
            f"strictly ordered for {origin}; a degenerate interval cannot "
            f"state ratio clipping"
        )
    if (
        not isinstance(weight, (int, float))
        or isinstance(weight, bool)
        or not math.isfinite(weight)
        or weight <= 0.0
    ):
        raise LossConfigRefusal(
            f"weight={weight!r}: the one mandatory component of {origin} "
            f"requires a positive finite weight; a zero or non-finite "
            f"weight would take the whole objective outside the coverage "
            f"denominator"
        )
    if kl_weight is not None:
        if (
            not isinstance(kl_weight, (int, float))
            or isinstance(kl_weight, bool)
            or not math.isfinite(kl_weight)
            or kl_weight < 0.0
        ):
            raise LossConfigRefusal(
                f"kl_weight={kl_weight!r}: the optional k3 term of "
                f"{origin} must be finite and non-negative; 0.0 carries "
                f"no reference column and deactivates the term, which is "
                f"the standard recipe's default"
            )
        if kl_weight != 0.0 and (
            not isinstance(reference_logprob_column, str) or not reference_logprob_column
        ):
            raise LossConfigRefusal(
                f"kl_weight={kl_weight!r} with reference_logprob_column="
                f"{reference_logprob_column!r}: an active k3 term for "
                f"{origin} requires a named reference column; 2 declared "
                f"fields disagree about whether a reference exists"
            )


@dataclass(frozen=True, slots=True)
class GSPOLoss:
    """GSPO: sequence-level ratio, sequence-mean reduction, near-1 clip.

    Per response, over its supervised tokens::

        seq_log_ratio = mean(current_logprob - old_logprob)
        ratio         = exp(seq_log_ratio)          (a geometric mean)
        score         = min(ratio * advantage,
                            clip(ratio, low, high) * advantage)
        loss          = -weight * mean over responses of score

    Token count NEVER enters the policy denominator: the ratio is already a
    whole-sequence quantity, so there is exactly one scalar surrogate per
    response and the objective is their mean (``sequence_mean``). Length
    enters only through ``seq_log_ratio`` itself.

    On the clip interval: GRPO's (0.8, 1.2) applied to a sequence ratio
    would essentially never bind, because a geometric mean over tokens
    concentrates far closer to 1 than a token ratio does. The DEFAULT
    ``(0.9997, 1.0003)`` is a default, not a measurement -- the measured,
    load-bearing claim is the SCALE (strictly narrower than any token-level
    interval), and the three readings of the published constant all survive
    that structural claim. Respondents are encouraged to treat the exact
    value as configuration, and the tests pin only the narrowness.

    ``kl_weight`` defaults to 0.0 (the measured GSPO recipe carries no
    reference term); setting it positive adds GRPO's token-relative k3 term
    and names a reference column in ``required_columns``. The k3
    denominator is the supervised-token mean regardless of ratio scope:
    only the policy ratio moved to sequence scope, and a whole-sequence KL
    would be a third unmeasured axis (see the module docstring's argument).

    WHAT IS CLAIMED: the ratio scope is sequence, the clip interval is the
    configured near-1 (default narrower than token intervals), the
    advantage estimator is the concrete
    :class:`GroupNormalisedAdvantage`, and the policy reduction divides by
    the count of PRICED RESPONSES only.

    WHAT IS NOT CLAIMED: that a response with zero supervised tokens is
    priced (it is refused -- ``exp`` of an undefined mean is not a reading),
    that dropping zero-variance groups is waived (it is GRPO's estimator,
    unchanged), or that any training-quality property holds for the chosen
    default interval.
    """

    group_size: int = 2
    clip_low: float = 0.9997
    clip_high: float = 1.0003
    kl_weight: float = 0.0
    weight: float = 1.0
    prompt_id_column: str = "prompt_ids"
    reward_column: str = "rewards"
    mask_column: str = "loss_mask"
    old_logprob_column: str = "old_logprobs"
    reference_logprob_column: str = "reference_logprobs"
    component_name: str = "gspo_policy_loss"
    kl_component_name: str = "gspo_kl"
    advantage_fn: GroupNormalisedAdvantage = field(default_factory=GroupNormalisedAdvantage)

    def __post_init__(self) -> None:
        _validate_common(
            group_size=self.group_size,
            clip_low=self.clip_low,
            clip_high=self.clip_high,
            weight=self.weight,
            kl_weight=self.kl_weight,
            reference_logprob_column=self.reference_logprob_column,
            origin="gspo",
        )
        for field_name, name_value in (
            ("prompt_id_column", self.prompt_id_column),
            ("reward_column", self.reward_column),
            ("mask_column", self.mask_column),
            ("old_logprob_column", self.old_logprob_column),
            ("reference_logprob_column", self.reference_logprob_column),
            ("component_name", self.component_name),
            ("kl_component_name", self.kl_component_name),
        ):
            _checked_name(field_name, name_value)
        if self.component_name == self.kl_component_name:
            raise LossConfigRefusal(
                f"component_name and kl_component_name are both "
                f"{self.component_name!r}: 2 components would share 1 "
                f"name and collapse the objective denominator"
            )
        if not isinstance(self.advantage_fn, GroupNormalisedAdvantage):
            raise LossConfigRefusal(
                f"advantage_fn={self.advantage_fn!r}: GSPO binds "
                f"GroupNormalisedAdvantage; 1 of 1 estimator slots must "
                f"carry that concrete estimator, not an undeclared "
                f"substitute"
            )
        if self.advantage_fn.min_group_size != self.group_size:
            raise LossConfigRefusal(
                f"group_size={self.group_size} and "
                f"advantage_fn.min_group_size="
                f"{self.advantage_fn.min_group_size}: 1 configured GSPO "
                f"group size disagrees with 1 estimator minimum"
            )

    @property
    def expects_dynamic_sampling(self) -> bool:
        """False: GSPO declares no refill expectation on the rollout loop."""
        return False

    @property
    def ratio_scope(self) -> Literal["token", "sequence"]:
        return "sequence"

    @property
    def reduction(self) -> Literal["token_mean", "sequence_mean", "constant"]:
        return "sequence_mean"

    @property
    def clip_bounds(self) -> tuple[float, float]:
        return (float(self.clip_low), float(self.clip_high))

    @property
    def required_columns(self) -> tuple[str, ...]:
        columns = [
            self.prompt_id_column,
            self.reward_column,
            self.mask_column,
            self.old_logprob_column,
        ]
        if self.kl_weight != 0.0:
            columns.append(self.reference_logprob_column)
        return tuple(columns)

    def declaration(self) -> LossDeclaration:
        components: tuple[str, ...] = (self.component_name,)
        if self.kl_weight != 0.0:
            components = components + (self.kl_component_name,)
        # The policy component is declared unconditionally and its weight is
        # validated strictly positive, so declaration() can never decompose
        # into zero components.
        return LossDeclaration(components=components)

    def semantics(self) -> AlgorithmSemantics:
        return AlgorithmSemantics(
            group_size=self.group_size,
            ratio_scope="sequence",
            kl_estimator=("k3" if self.kl_weight != 0.0 else None),
            clip_bounds=self.clip_bounds,
            reference_free=self.kl_weight == 0.0,
        )

    def __call__(self, forward_fn: ForwardFn, batch: ExperienceBatch) -> LossOutput:
        output, _advantage = self.compute_with_report(forward_fn, batch)
        return output

    def compute_with_report(
        self, forward_fn: ForwardFn, batch: ExperienceBatch
    ) -> tuple[LossOutput, AdvantageResult]:
        prices = _price_rows(
            forward_fn=forward_fn,
            batch=batch,
            advantage_fn=self.advantage_fn,
            group_size=self.group_size,
            prompt_id_column=self.prompt_id_column,
            reward_column=self.reward_column,
            mask_column=self.mask_column,
            old_logprob_column=self.old_logprob_column,
            reference_logprob_column=(
                self.reference_logprob_column if self.kl_weight != 0.0 else None
            ),
            origin="gspo",
        )
        response_scores: list[float] = []
        for index, batch_row in enumerate(prices.batch_rows):
            mask = prices.mask_rows[index]
            weights = prices.weight_rows[index]
            current = prices.current_rows[index]
            old = prices.old_rows[index]
            log_ratio_total = 0.0
            weight_total = 0.0
            supervised = 0
            for position, is_supervised in enumerate(mask):
                if not is_supervised:
                    continue
                log_ratio_total += current[position] - old[position]
                weight_total += weights[position]
                supervised += 1
            if supervised == 0:
                raise SupervisionRefusal(
                    f"row {batch_row}: 0 supervised tokens remain in 1 of "
                    f"{len(prices.batch_rows)} priced responses; a "
                    f"sequence-level ratio over an empty token set is "
                    f"unmeasured, not 1.0, and a mean including it would "
                    f"silently reweight its group"
                )
            # The response's scalar advantage is the mean of its supervised
            # token weights: GroupNormalisedAdvantage emits one constant
            # weight per (group, position) slice, so the mean is the
            # estimator's advantage exactly; honestly averaging guards
            # against a future estimator that returns positional values.
            response_advantage = weight_total / supervised
            sequence_ratio = _ratio(log_ratio_total / supervised, batch_row, None, "gspo")
            response_scores.append(
                _clipped_term(
                    sequence_ratio,
                    response_advantage,
                    self.clip_low,
                    self.clip_high,
                )
            )
        if not response_scores:
            raise SupervisionRefusal(
                f"0 of {prices.advantage.offered} offered rows survived "
                f"group normalisation for gspo; a sequence mean over zero "
                f"responses is unmeasured, never 0.0"
            )
        policy_contribution = -self.weight * (sum(response_scores) / len(response_scores))
        components: list[LossComponent] = [
            LossComponent(
                name=self.component_name,
                weight=self.weight,
                observed=True,
                contribution=policy_contribution,
            )
        ]
        kl_contribution = 0.0
        if self.kl_weight != 0.0:
            kl_total, kl_supervised = _kl_component_terms(prices, origin="gspo")
            if kl_supervised == 0:
                raise SupervisionRefusal(
                    f"kl_weight={self.kl_weight!r} is active but 0 "
                    f"supervised tokens remain across {prices.advantage.used} "
                    f"used rows for gspo; the k3 term is unmeasured"
                )
            kl_contribution = self.kl_weight * (kl_total / kl_supervised)
            components.append(
                LossComponent(
                    name=self.kl_component_name,
                    weight=self.kl_weight,
                    observed=True,
                    contribution=kl_contribution,
                )
            )
        return (
            LossOutput(
                loss=policy_contribution + kl_contribution,
                components=tuple(components),
            ),
            prices.advantage,
        )


@dataclass(frozen=True, slots=True)
class DrGRPOLoss:
    """Dr.GRPO: token ratio, centred advantage, constant-length reduction.

    Per supervised token, over each priced row::

        ratio = exp(current_logprob - old_logprob)
        score = min(ratio * advantage, clip(ratio, 0.8, 1.2) * advantage)
        loss  = -weight * (SUM of scores) / (rows_used * constant_length)

    Two deliberate departures from GRPO, both measured:

    * The advantage is CENTRED ONLY -- :class:`CentredAdvantage`, reward
      minus group mean, with NO division by group standard deviation. A
      zero-variance group therefore yields all-zero weights and is KEPT
      (``used == offered`` for it), where GroupNormalisedAdvantage must
      drop it. That estimator-level difference is pinned by the advantage
      tests, not restated here.
    * The length-bias fix: the denominator uses the CONFIGURED
      ``constant_length``, never an observed length. A response twice as
      long contributes twice the token surrogates against the same
      denominator as every other row, so gradient-per-token stays flat in
      length; dividing by observed lengths (GRPO's ``token_mean``) is
      exactly the bias being removed.

    ``constant_length`` has no honest default -- it is a property of the
    run's maximum response length, not of this module -- but dataclass
    fields after defaulted ones require an ordering, so it defaults to the
    sentinel-concrete 1024 and its overspecified nature is stated here: any
    real run MUST configure it from its rollout maximum, and the value is a
    configured constant, not a measurement of any batch.

    ``kl_weight`` defaults to 0.0, matching the standard recipe; setting it
    positive adds GRPO's token-relative k3 term (denominated by the same
    supervised-token mean as GRPO's). Both readings of the Dr.GRPO record
    ("KL dropped" and "KL optional") are representable.

    WHAT IS CLAIMED: token ratio scope, symmetric clip, the concrete
    :class:`CentredAdvantage`, and a denominator of
    ``rows_used * constant_length`` in which no observed length appears.

    WHAT IS NOT CLAIMED: that ``constant_length`` matches any run's real
    maximum unless the caller sets it, that zero-variance groups are
    dropped (they are kept, with zero weight), or that removing the
    measured length bias improves training -- nothing here has been
    trained.
    """

    group_size: int = 2
    clip_low: float = 0.8
    clip_high: float = 1.2
    kl_weight: float = 0.0
    weight: float = 1.0
    constant_length: int = 1024
    prompt_id_column: str = "prompt_ids"
    reward_column: str = "rewards"
    mask_column: str = "loss_mask"
    old_logprob_column: str = "old_logprobs"
    reference_logprob_column: str = "reference_logprobs"
    component_name: str = "drgrpo_policy_loss"
    kl_component_name: str = "drgrpo_kl"
    advantage_fn: CentredAdvantage = field(default_factory=CentredAdvantage)

    def __post_init__(self) -> None:
        _validate_common(
            group_size=self.group_size,
            clip_low=self.clip_low,
            clip_high=self.clip_high,
            weight=self.weight,
            kl_weight=self.kl_weight,
            reference_logprob_column=self.reference_logprob_column,
            origin="dr_grpo",
        )
        if isinstance(self.constant_length, bool):
            raise LossConfigRefusal(
                f"constant_length={self.constant_length!r}: True is not a "
                f"Dr.GRPO length constant; the configured maximum "
                f"response length is a positive integer"
            )
        if not isinstance(self.constant_length, int) or self.constant_length <= 0:
            raise LossConfigRefusal(
                f"constant_length={self.constant_length!r}: the "
                f"constant-reduction denominator requires a positive "
                f"integer length; a zero or non-integer denominator "
                f"would divide the whole step by nothing measurable"
            )
        for field_name, name_value in (
            ("prompt_id_column", self.prompt_id_column),
            ("reward_column", self.reward_column),
            ("mask_column", self.mask_column),
            ("old_logprob_column", self.old_logprob_column),
            ("reference_logprob_column", self.reference_logprob_column),
            ("component_name", self.component_name),
            ("kl_component_name", self.kl_component_name),
        ):
            _checked_name(field_name, name_value)
        if self.component_name == self.kl_component_name:
            raise LossConfigRefusal(
                f"component_name and kl_component_name are both "
                f"{self.component_name!r}: 2 components would share 1 "
                f"name and collapse the objective denominator"
            )
        if not isinstance(self.advantage_fn, CentredAdvantage):
            raise LossConfigRefusal(
                f"advantage_fn={self.advantage_fn!r}: Dr.GRPO binds "
                f"CentredAdvantage; 1 of 1 estimator slots must carry "
                f"the centred-only estimator -- the measured recipe has "
                f"NO std normalisation -- and a normalised substitute "
                f"would silently become GRPO-with-constant-reduction"
            )
        if self.advantage_fn.min_group_size != self.group_size:
            raise LossConfigRefusal(
                f"group_size={self.group_size} and "
                f"advantage_fn.min_group_size="
                f"{self.advantage_fn.min_group_size}: 1 configured "
                f"Dr.GRPO group size disagrees with 1 estimator minimum"
            )

    @property
    def expects_dynamic_sampling(self) -> bool:
        """False: Dr.GRPO declares no refill expectation on the rollout loop."""
        return False

    @property
    def ratio_scope(self) -> Literal["token", "sequence"]:
        return "token"

    @property
    def reduction(self) -> Literal["token_mean", "sequence_mean", "constant"]:
        return "constant"

    @property
    def clip_bounds(self) -> tuple[float, float]:
        return (float(self.clip_low), float(self.clip_high))

    @property
    def required_columns(self) -> tuple[str, ...]:
        columns = [
            self.prompt_id_column,
            self.reward_column,
            self.mask_column,
            self.old_logprob_column,
        ]
        if self.kl_weight != 0.0:
            columns.append(self.reference_logprob_column)
        return tuple(columns)

    def declaration(self) -> LossDeclaration:
        components: tuple[str, ...] = (self.component_name,)
        if self.kl_weight != 0.0:
            components = components + (self.kl_component_name,)
        return LossDeclaration(components=components)

    def semantics(self) -> AlgorithmSemantics:
        return AlgorithmSemantics(
            group_size=self.group_size,
            ratio_scope="token",
            kl_estimator=("k3" if self.kl_weight != 0.0 else None),
            clip_bounds=self.clip_bounds,
            reference_free=self.kl_weight == 0.0,
        )

    def __call__(self, forward_fn: ForwardFn, batch: ExperienceBatch) -> LossOutput:
        output, _advantage = self.compute_with_report(forward_fn, batch)
        return output

    def compute_with_report(
        self, forward_fn: ForwardFn, batch: ExperienceBatch
    ) -> tuple[LossOutput, AdvantageResult]:
        prices = _price_rows(
            forward_fn=forward_fn,
            batch=batch,
            advantage_fn=self.advantage_fn,
            group_size=self.group_size,
            prompt_id_column=self.prompt_id_column,
            reward_column=self.reward_column,
            mask_column=self.mask_column,
            old_logprob_column=self.old_logprob_column,
            reference_logprob_column=(
                self.reference_logprob_column if self.kl_weight != 0.0 else None
            ),
            origin="dr_grpo",
        )
        surrogate_total = 0.0
        supervised = 0
        for index, batch_row in enumerate(prices.batch_rows):
            mask = prices.mask_rows[index]
            weights = prices.weight_rows[index]
            current = prices.current_rows[index]
            old = prices.old_rows[index]
            for position, is_supervised in enumerate(mask):
                if not is_supervised:
                    continue
                ratio = _ratio(
                    current[position] - old[position],
                    batch_row,
                    position,
                    "dr_grpo",
                )
                surrogate_total += _clipped_term(
                    ratio,
                    weights[position],
                    self.clip_low,
                    self.clip_high,
                )
                supervised += 1
        rows_used = len(prices.batch_rows)
        if supervised == 0:
            raise SupervisionRefusal(
                f"0 supervised tokens remain across {rows_used} used rows "
                f"for dr_grpo (of {prices.advantage.offered} offered); the "
                f"loss is unmeasured and returning 0.0 would report a "
                f"perfect loss over no gradient"
            )
        denominator = rows_used * self.constant_length
        policy_contribution = -self.weight * (surrogate_total / denominator)
        components: list[LossComponent] = [
            LossComponent(
                name=self.component_name,
                weight=self.weight,
                observed=True,
                contribution=policy_contribution,
            )
        ]
        kl_contribution = 0.0
        if self.kl_weight != 0.0:
            kl_total, kl_supervised = _kl_component_terms(prices, origin="dr_grpo")
            if kl_supervised == 0:
                raise SupervisionRefusal(
                    f"kl_weight={self.kl_weight!r} is active but 0 "
                    f"supervised tokens remain across {rows_used} used rows "
                    f"for dr_grpo; the k3 term is unmeasured"
                )
            kl_contribution = self.kl_weight * (kl_total / kl_supervised)
            components.append(
                LossComponent(
                    name=self.kl_component_name,
                    weight=self.kl_weight,
                    observed=True,
                    contribution=kl_contribution,
                )
            )
        return (
            LossOutput(
                loss=policy_contribution + kl_contribution,
                components=tuple(components),
            ),
            prices.advantage,
        )


@dataclass(frozen=True, slots=True)
class DAPOLoss:
    """DAPO: token ratio, ASYMMETRIC clip, token-mean reduction, NO kl field.

    Per supervised token, over each priced row::

        ratio = exp(current_logprob - old_logprob)
        score = min(ratio * advantage,
                    clip(ratio, clip_low, clip_high) * advantage)
        loss  = -weight * (SUM of scores) / (count of supervised tokens)

    The reduction is ``token_mean`` -- byte-for-byte GRPOPolicyLoss's
    ``policy_objective_total / supervised`` -- and the clip defaults to the
    measured ASYMMETRIC ``(0.8, 1.28)`` ("clip-higher"), the only clip
    interval in this family whose upper bound is not the mirror of its
    lower.

    There is no ``kl_weight`` field at all -- deliberately absent, not
    defaulted to zero. A present-but-zero field invites a caller to set it
    positive and thereby build a DAPO-with-k3 configuration nothing in the
    measured record describes; omitting the field makes that configuration
    unrepresentable. Dr.GRPO's KL is "dropped or optional", so its field
    exists with a zero default; DAPO's KL is absent from the record, so the
    field is absent with it.

    DYNAMIC SAMPLING IS DECLARED, NOT IMPLEMENTED.
    ``expects_dynamic_sampling`` is True, and NOTHING in this module -- or
    anywhere in this package -- enforces it. Zero-variance groups are
    already dropped by :class:`GroupNormalisedAdvantage` (visible as
    ``used < offered``), which is the FILTER half; the REFILL half --
    regenerating completions until the batch is full of mixed-outcome
    groups -- is a rollout-loop behaviour, and this package has no rollout
    loop to put it in. Overlong reward shaping is rollout-side for the same
    reason and is likewise NOT implemented here. A declared, unenforced
    expectation is honest; silently omitting the declaration would be the
    defect, because the loss alone cannot detect its absence.

    WHAT IS CLAIMED: token ratio scope, the configured asymmetric clip, the
    concrete :class:`GroupNormalisedAdvantage`, a token-mean denominator
    over exactly the supervised tokens of the used rows, and no KL term
    under any configuration.

    WHAT IS NOT CLAIMED: dynamic sampling, overlong reward shaping, or any
    rollout-side DAPO behaviour; and no convergence, benchmark, or
    equivalence-to-the-paper claim -- this objective has never been
    trained here.
    """

    group_size: int = 2
    clip_low: float = 0.8
    clip_high: float = 1.28
    weight: float = 1.0
    prompt_id_column: str = "prompt_ids"
    reward_column: str = "rewards"
    mask_column: str = "loss_mask"
    old_logprob_column: str = "old_logprobs"
    component_name: str = "dapo_policy_loss"
    advantage_fn: GroupNormalisedAdvantage = field(default_factory=GroupNormalisedAdvantage)

    def __post_init__(self) -> None:
        _validate_common(
            group_size=self.group_size,
            clip_low=self.clip_low,
            clip_high=self.clip_high,
            weight=self.weight,
            kl_weight=None,
            reference_logprob_column=None,
            origin="dapo",
        )
        for field_name, name_value in (
            ("prompt_id_column", self.prompt_id_column),
            ("reward_column", self.reward_column),
            ("mask_column", self.mask_column),
            ("old_logprob_column", self.old_logprob_column),
            ("component_name", self.component_name),
        ):
            _checked_name(field_name, name_value)
        if not isinstance(self.advantage_fn, GroupNormalisedAdvantage):
            raise LossConfigRefusal(
                f"advantage_fn={self.advantage_fn!r}: DAPO binds "
                f"GroupNormalisedAdvantage; 1 of 1 estimator slots must "
                f"carry that concrete estimator, not an undeclared "
                f"substitute"
            )
        if self.advantage_fn.min_group_size != self.group_size:
            raise LossConfigRefusal(
                f"group_size={self.group_size} and "
                f"advantage_fn.min_group_size="
                f"{self.advantage_fn.min_group_size}: 1 configured DAPO "
                f"group size disagrees with 1 estimator minimum"
            )

    @property
    def expects_dynamic_sampling(self) -> bool:
        """True -- declared, and unenforced here; see the class docstring.

        The expectation names a ROLLOUT-side refill behaviour this
        torch-free loss module cannot host; the binding surfaces this to
        whatever caller owns the rollout loop.
        """
        return True

    @property
    def ratio_scope(self) -> Literal["token", "sequence"]:
        return "token"

    @property
    def reduction(self) -> Literal["token_mean", "sequence_mean", "constant"]:
        return "token_mean"

    @property
    def clip_bounds(self) -> tuple[float, float]:
        return (float(self.clip_low), float(self.clip_high))

    @property
    def required_columns(self) -> tuple[str, ...]:
        return (
            self.prompt_id_column,
            self.reward_column,
            self.mask_column,
            self.old_logprob_column,
        )

    def declaration(self) -> LossDeclaration:
        # One unconditional component whose weight is validated strictly
        # positive: a zero-component declaration is unreachable.
        return LossDeclaration(components=(self.component_name,))

    def semantics(self) -> AlgorithmSemantics:
        # kl_estimator is None -- abstention, in the house sense: DAPO
        # carries no KL term structurally, so "which KL estimator" is a
        # question the objective refuses to pose.
        return AlgorithmSemantics(
            group_size=self.group_size,
            ratio_scope="token",
            kl_estimator=None,
            clip_bounds=self.clip_bounds,
            reference_free=True,
        )

    def __call__(self, forward_fn: ForwardFn, batch: ExperienceBatch) -> LossOutput:
        output, _advantage = self.compute_with_report(forward_fn, batch)
        return output

    def compute_with_report(
        self, forward_fn: ForwardFn, batch: ExperienceBatch
    ) -> tuple[LossOutput, AdvantageResult]:
        prices = _price_rows(
            forward_fn=forward_fn,
            batch=batch,
            advantage_fn=self.advantage_fn,
            group_size=self.group_size,
            prompt_id_column=self.prompt_id_column,
            reward_column=self.reward_column,
            mask_column=self.mask_column,
            old_logprob_column=self.old_logprob_column,
            reference_logprob_column=None,
            origin="dapo",
        )
        surrogate_total = 0.0
        supervised = 0
        for index, batch_row in enumerate(prices.batch_rows):
            mask = prices.mask_rows[index]
            weights = prices.weight_rows[index]
            current = prices.current_rows[index]
            old = prices.old_rows[index]
            for position, is_supervised in enumerate(mask):
                if not is_supervised:
                    continue
                ratio = _ratio(
                    current[position] - old[position],
                    batch_row,
                    position,
                    "dapo",
                )
                surrogate_total += _clipped_term(
                    ratio,
                    weights[position],
                    self.clip_low,
                    self.clip_high,
                )
                supervised += 1
        if supervised == 0:
            raise SupervisionRefusal(
                f"0 supervised tokens remain across {prices.advantage.used} "
                f"used rows (of {prices.advantage.offered} offered) for "
                f"dapo; the loss is unmeasured and returning 0.0 would "
                f"report a perfect loss over no gradient"
            )
        policy_contribution = -self.weight * (surrogate_total / supervised)
        components = (
            LossComponent(
                name=self.component_name,
                weight=self.weight,
                observed=True,
                contribution=policy_contribution,
            ),
        )
        return (
            LossOutput(loss=policy_contribution, components=components),
            prices.advantage,
        )
