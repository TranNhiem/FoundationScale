"""Stage-3e policy-gradient family: RLOO, REINFORCE-with-baseline, Reinforce++.

Three concrete bindings that sit beside the landed GRPO binding and plug into
the same registry seam. Stage 3d's registry docstring sets the shape of that
seam: a family adds a module and a registration, not an edit to the contracts
above. So this module imports ``AlgorithmRequirements``,
``check_algorithm_wiring`` and ``verify_step`` unchanged, reuses
:class:`LeaveOneOutAdvantage` from ``advantage.py`` for RLOO rather than
restating the estimator, and registers its three factories at import.

WHAT THIS MODULE CLAIMS: a wired RLOO step prices exactly the compacted rows
of :class:`LeaveOneOutAdvantage` through an unclipped token-level ratio; a
wired REINFORCE-with-baseline step subtracts exactly the EMA baseline it
reports as a declared metric, and the first step's baseline is the batch's own
mean return, stated rather than hidden; a wired Reinforce++ step folds a
token-level k1 penalty into each sample's return, normalises the penalised
returns GLOBALLY (population statistics, never per-group), and applies
PPO-style clipping on top. Each binding declares its ``AlgorithmSemantics``
for the seams it constrains and abstains (``None``) on the seams it does not.

WHAT THIS MODULE DOES NOT CLAIM: any equivalence with a reference
implementation of any of the three algorithms -- nothing here has been
benchmarked against one. The estimator and parameterisation choices (k1 for
the Reinforce++ penalty, population statistics for its global normalisation,
the mean-return-seeded EMA for REINFORCE, the unclipped token ratio for RLOO)
are declared binding choices, not verified reproductions. No rollout
production, model forward, optimizer, or weight transport lives here.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from foundationscale.gates.objective_gates import (
    LossComponent,
    MetricExpectation,
    MetricObservation,
)
from foundationscale.rl.advantage import (
    AdvantageFn,
    AdvantageRefusal,
    AdvantageResult,
    LeaveOneOutAdvantage,
    RewardStats,
)
from foundationscale.rl.algorithm import (
    AlgorithmRequirements,
    AlgorithmSemantics,
    AlgorithmWiringRefusal,
    StepReport,
    StepReportRefusal,
    check_algorithm_wiring,
)
from foundationscale.rl.interfaces import (
    BatchRefusal,
    ExperienceBatch,
    ForwardFn,
    LossConfigRefusal,
    LossDeclaration,
    LossOutput,
    SupervisionRefusal,
)
from foundationscale.rl.policy import PolicyPair
from foundationscale.rl.rollout import RolloutSource
from foundationscale.rl.weightsync import WeightSync

__all__ = (
    "RLOOAlgorithm",
    "RLOOPolicyLoss",
    "ReinforceBaselineAlgorithm",
    "ReinforceBaselineLoss",
    "ReinforcePlusPlusAlgorithm",
    "ReinforcePlusPlusLoss",
    "check_reinforce_baseline_requirements",
    "check_reinforce_pp_requirements",
    "check_rloo_requirements",
)


def _checked_name(field_name: str, value: Any) -> None:
    # Every name-bearing config field refuses a non-name at construction:
    # the refusal names the field that produced it rather than the gate or
    # manifest that would otherwise see it first.
    if not isinstance(value, str) or not value:
        raise LossConfigRefusal(
            f"{field_name}={value!r}: column and component names must be "
            f"non-empty strings; absence of a name is not a name"
        )


def _positive_finite(field_name: str, value: Any, mechanism: str) -> None:
    # bool is checked BEFORE the numeric test because isinstance(True, int)
    # is True and True > 0.0 would otherwise read as a measured weight of 1.
    if isinstance(value, bool):
        raise LossConfigRefusal(
            f"{field_name}={value!r}: True is not a {mechanism}; 1 of 1 "
            f"{mechanism} fields must be a positive finite number"
        )
    if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0.0:
        raise LossConfigRefusal(
            f"{field_name}={value!r}: 1 of 1 {mechanism} fields must be a "
            f"positive finite number; a zero or non-finite {mechanism} would "
            f"sit in the objective silently erasing its own denominator"
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


def _row_sequence(value: Any, *, field_name: str, row: int) -> tuple[Any, ...]:
    try:
        return tuple(value)
    except TypeError as exc:
        raise BatchRefusal(
            f"field {field_name} row {row}={value!r}: 1 of 1 rows was not "
            f"iterable ({type(value).__name__}); one entry per token "
            f"position is required"
        ) from exc


def _mask_entry(raw: Any, row: int, position: int) -> bool:
    # The bool branch comes FIRST, because isinstance(True, int) is True.
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


def _logprob_value(raw: Any, field_name: str, row: int, position: int) -> float:
    # One finite log-probability per token; a non-finite reading would
    # poison both the ratio and any penalty projection while the rest of
    # the batch reads as healthy.
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise BatchRefusal(
            f"field {field_name} at row {row}, position {position} does "
            f"not convert to a scalar float ({type(raw).__name__}); policy "
            f"gradients require one finite log-probability per token"
        ) from exc
    if not math.isfinite(value):
        raise BatchRefusal(
            f"field {field_name} at row {row}, position {position} is "
            f"{raw!r}, which is not finite; a non-finite token reading "
            f"would poison the ratio before any clipping could refuse it"
        )
    return value


def _reward_scalar(raw: Any, row: int) -> float:
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise BatchRefusal(
            f"reward at row {row} is {raw!r}, which does not convert to a "
            f"scalar float ({type(raw).__name__}); rewards must be one "
            f"finite scalar per sample"
        ) from exc
    if not math.isfinite(value):
        raise BatchRefusal(
            f"reward at row {row} is {raw!r}, which is not finite; rewards "
            f"must be one finite scalar per sample"
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


def _token_ratio(current_value: float, old_value: float, row: int, position: int) -> float:
    log_ratio = current_value - old_value
    try:
        ratio = math.exp(log_ratio)
    except OverflowError as exc:
        raise BatchRefusal(
            f"row {row}, position {position}: token ratio "
            f"exp({log_ratio!r}) is not representable; 1 of 1 token "
            f"ratios failed before any use of it"
        ) from exc
    if ratio <= 0.0 or not math.isfinite(ratio):
        raise BatchRefusal(
            f"row {row}, position {position}: token ratio {ratio!r} is "
            f"not a positive finite value; a silently underflowed ratio "
            f"would manufacture a measured-looking zero contribution"
        )
    return ratio


def _supplied_is_present(value: Any) -> bool:
    # Absence is None or a missing key, never a boolean: True read as
    # presence would launder a declaration into wiring.
    if value is None:
        return False
    if isinstance(value, bool):
        raise AlgorithmWiringRefusal(
            f"field supplied contains the boolean {value!r}: True and "
            f"False are not supplied components; presence must be the actual "
            f"wired object and absence must be represented by no key or by "
            f"None"
        )
    return True


def _check_policy_gradient_roles(
    *,
    requires: Mapping[str, bool],
    supplied: Mapping[str, Any],
    origin: str,
) -> tuple[str, ...]:
    # The one implementation behind all three public check_<name>_requirements
    # helpers. The role-map check is ONE countable (the findings this
    # repository keeps filing are restated countables drifting apart), so the
    # family states it once and the three bindings differ only in origin.
    if not isinstance(requires, Mapping):
        raise AlgorithmWiringRefusal(
            f"field requires={requires!r}: 1 of 1 requirement mappings must implement Mapping"
        )
    if not isinstance(supplied, Mapping):
        raise AlgorithmWiringRefusal(
            f"field supplied={supplied!r}: 1 of 1 supplied mappings must implement Mapping"
        )
    requirement_map = dict(requires)
    supplied_map = dict(supplied)
    bad_requirement_names = tuple(
        name for name in requirement_map if not isinstance(name, str) or not name
    )
    bad_supplied_names = tuple(
        name for name in supplied_map if not isinstance(name, str) or not name
    )
    if bad_requirement_names:
        raise AlgorithmWiringRefusal(
            f"field requires: {len(bad_requirement_names)} of "
            f"{len(requirement_map)} names are not non-empty strings: "
            f"{bad_requirement_names!r}"
        )
    if bad_supplied_names:
        raise AlgorithmWiringRefusal(
            f"field supplied: {len(bad_supplied_names)} of "
            f"{len(supplied_map)} names are not non-empty strings: "
            f"{bad_supplied_names!r}"
        )
    bad_values = tuple(
        name for name, required in requirement_map.items() if not isinstance(required, bool)
    )
    if bad_values:
        raise AlgorithmWiringRefusal(
            f"field requires: {len(bad_values)} of "
            f"{len(requirement_map)} requirement values are not bool: "
            f"{bad_values!r}; a required role must be measured True or "
            f"declared unrequired False"
        )
    union = tuple(sorted(set(requirement_map) | set(supplied_map)))
    if not union:
        raise AlgorithmWiringRefusal(
            f"fields requires and supplied: 0 roles appear in either map "
            f"for {origin}; a wiring check over an empty union would pass "
            f"vacuously by measuring nothing"
        )
    required = tuple(name for name in union if requirement_map.get(name) is True)
    if not required:
        raise AlgorithmWiringRefusal(
            f"field requires: 0 of {len(requirement_map)} declared roles "
            f"are required for {origin}; an empty required-set is refused "
            f"as vacuous"
        )
    present = {name: entry for name, entry in supplied_map.items() if _supplied_is_present(entry)}
    absent_required = tuple(name for name in required if name not in present)
    if absent_required:
        raise AlgorithmWiringRefusal(
            f"field supplied: {len(absent_required)} of {len(required)} "
            f"required inputs absent for {origin}: {absent_required!r}; "
            f"the checked denominator was {len(union)} roles from "
            f"{len(requirement_map)} requires keys and {len(supplied_map)} "
            f"supplied keys"
        )
    unrequired_supplied = tuple(
        name for name in union if name in present and requirement_map.get(name) is not True
    )
    if unrequired_supplied:
        raise AlgorithmWiringRefusal(
            f"field supplied: {len(unrequired_supplied)} of "
            f"{len(present)} supplied inputs are not required by "
            f"{origin}: {unrequired_supplied!r}; the checked denominator "
            f"was {len(union)} roles from {len(requirement_map)} requires "
            f"keys and {len(supplied_map)} supplied keys"
        )
    return tuple(sorted(name for name in required if name in present))


def check_rloo_requirements(
    *,
    requires: Mapping[str, bool],
    supplied: Mapping[str, Any],
    origin: str = "rloo",
) -> tuple[str, ...]:
    """Check RLOO's role map over the union of both mapping key sets.

    WHAT IS CLAIMED on success: every required role names a supplied object,
    the required set is non-empty (an empty one is refused as vacuous before
    any agreement could read as coverage), and the returned tuple is the
    sorted set of roles actually consumed.

    WHAT IS NOT CLAIMED: that any role object is a valid model, dataloader,
    estimator, or loss -- presence only. The implementation is shared with
    the family's other two checks because a restated role-map check is one
    countable in three copies; only the origin differs per binding.
    """
    return _check_policy_gradient_roles(requires=requires, supplied=supplied, origin=origin)


def check_reinforce_baseline_requirements(
    *,
    requires: Mapping[str, bool],
    supplied: Mapping[str, Any],
    origin: str = "reinforce_baseline",
) -> tuple[str, ...]:
    """Check REINFORCE-with-baseline's role map over both key sets' union.

    WHAT IS CLAIMED on success: as for :func:`check_rloo_requirements` --
    presence and non-vacuity of the required set, with the consumed roles
    returned sorted. REINFORCE's own ``_requires`` map marks only the three
    mandatory setup roles consumed, so those three are the smallest passing
    required set.

    WHAT IS NOT CLAIMED: that the baseline the algorithm carries agrees with
    any wiring; the baseline is step-time state, not a wired role, and this
    check sees only roles.
    """
    return _check_policy_gradient_roles(requires=requires, supplied=supplied, origin=origin)


def check_reinforce_pp_requirements(
    *,
    requires: Mapping[str, bool],
    supplied: Mapping[str, Any],
    origin: str = "reinforce_pp",
) -> tuple[str, ...]:
    """Check Reinforce++'s role map over the union of both mapping key sets.

    WHAT IS CLAIMED on success: as for :func:`check_rloo_requirements`.

    WHAT IS NOT CLAIMED: that the policy pair attesting the reference role
    carries a reference that produces finite log-probabilities; reference
    presence is the pair-half of this check and is measured by
    ``check_algorithm_wiring``, not here.
    """
    return _check_policy_gradient_roles(requires=requires, supplied=supplied, origin=origin)


# ---------------------------------------------------------------------------
# RLOO
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RLOOPolicyLoss:
    """Group-relative REINFORCE leave-one-out objective, unclipped token ratio.

    For each group of exactly ``group_size`` samples sharing a prompt id, this
    loss calls the existing :class:`LeaveOneOutAdvantage` -- reused, never
    restated -- and then for each used token computes::

        ratio = exp(current_logprob - old_logprob)
        contribution_token = ratio * advantage

    The reported policy component is the negative supervised-token mean over
    the estimator's surviving rows. There is NO clipping and NO reference
    term: unlike GRPO, nothing here reads a reference distribution.

    WHAT IS CLAIMED: the loss uses the existing estimator, preserves its
    used-row denominator, and declares exactly the one component it returns.
    A group of equal-reward samples keeps its measured 0.0 advantages --
    leave-one-out measures those, it does not manufacture them.

    WHAT IS NOT CLAIMED: that old log-probabilities came from any particular
    (or recent) policy, that the batch is on-policy, or that the unclipped
    ratio stays small; an unclipped ratio is a declared binding choice, not
    a verified property.
    """

    group_size: int = 2
    weight: float = 1.0
    prompt_id_column: str = "prompt_ids"
    reward_column: str = "rewards"
    mask_column: str = "loss_mask"
    old_logprob_column: str = "old_logprobs"
    component_name: str = "rloo_policy_loss"
    advantage_fn: LeaveOneOutAdvantage = field(default_factory=LeaveOneOutAdvantage)

    def __post_init__(self) -> None:
        if isinstance(self.group_size, bool):
            raise LossConfigRefusal(
                f"group_size={self.group_size!r}: True is not an RLOO group "
                f"size; a group size is an integer of at least 2"
            )
        if not isinstance(self.group_size, int) or self.group_size < 2:
            raise LossConfigRefusal(
                f"group_size={self.group_size!r}: RLOO requires a group of "
                f"at least 2 samples; below that there is no leave-one-out "
                f"baseline to subtract"
            )
        _positive_finite("weight", self.weight, "declared RLOO component weight")
        # `name_value` is DISTINCT from any numeric loop variable: reusing one
        # variable for floats and for names would make the two element types
        # one variable, and mypy strict reads that as an error.
        for field_name, name_value in (
            ("prompt_id_column", self.prompt_id_column),
            ("reward_column", self.reward_column),
            ("mask_column", self.mask_column),
            ("old_logprob_column", self.old_logprob_column),
            ("component_name", self.component_name),
        ):
            _checked_name(field_name, name_value)
        if not isinstance(self.advantage_fn, LeaveOneOutAdvantage):
            raise LossConfigRefusal(
                f"advantage_fn={self.advantage_fn!r}: RLOO binds "
                f"LeaveOneOutAdvantage; 1 of 1 estimator slots must "
                f"carry that concrete estimator, not an undeclared "
                f"substitute"
            )
        if self.advantage_fn.min_group_size != self.group_size:
            raise LossConfigRefusal(
                f"group_size={self.group_size} and "
                f"advantage_fn.min_group_size="
                f"{self.advantage_fn.min_group_size}: 1 configured RLOO "
                f"group size disagrees with 1 estimator minimum; RLOO "
                f"treats group_size as the exact K, not merely as a lower "
                f"bound"
            )

    @property
    def required_columns(self) -> tuple[str, ...]:
        """Return the batch columns consumed by this loss.

        WHAT IS CLAIMED: the tuple has a stable order and contains exactly
        the four inputs needed before forward log-probabilities are read.

        WHAT IS NOT CLAIMED: that the columns' values are shaped or finite;
        values are measured by ``compute_with_report``.
        """
        return (
            self.prompt_id_column,
            self.reward_column,
            self.mask_column,
            self.old_logprob_column,
        )

    def declaration(self) -> LossDeclaration:
        """Declare the single objective component this loss always emits.

        WHAT IS CLAIMED: the declared component name is exactly the name
        present in a successful ``LossOutput``.

        WHAT IS NOT CLAIMED: that any step uses every offered row; a prompt
        group of the wrong size is refused, and a leave-one-out group too
        small for a baseline leaves the estimator's denominator as
        ``used < offered``.
        """
        return LossDeclaration(components=(self.component_name,))

    def semantics(self) -> AlgorithmSemantics:
        """Return this loss's independently declared RLOO semantics.

        WHAT IS CLAIMED: exact group size, token ratio scope, reference-free.

        WHAT IS NOT CLAIMED: any KL estimator or clip bounds -- RLOO here has
        neither, and both abstain as ``None`` rather than posing defaults the
        mathematics never stated. Agreement with an independently constructed
        algorithm is measured by ``RLOOAlgorithm.setup``.
        """
        return AlgorithmSemantics(
            group_size=self.group_size,
            ratio_scope="token",
            kl_estimator=None,
            clip_bounds=None,
            reference_free=True,
        )

    def __call__(self, forward_fn: ForwardFn, batch: ExperienceBatch) -> LossOutput:
        """Price one batch through the existing ``LossFn`` surface.

        WHAT IS CLAIMED: the result is the first value returned by
        ``compute_with_report`` and carries exactly the declared component.

        WHAT IS NOT CLAIMED: that the accompanying ``AdvantageResult`` was
        preserved; callers needing its denominator call
        ``compute_with_report``.
        """
        output, _advantage = self.compute_with_report(forward_fn, batch)
        return output

    def compute_with_report(
        self, forward_fn: ForwardFn, batch: ExperienceBatch
    ) -> tuple[LossOutput, AdvantageResult]:
        """Compute the RLOO loss and retain its advantage denominator.

        WHAT IS CLAIMED: the returned ``AdvantageResult`` names the exact
        offered rows represented in the loss.

        WHAT IS NOT CLAIMED: that every offered row survived; groups of the
        wrong size are refused outright before the estimator runs, and the
        estimator itself may still leave a too-small group out with its
        absence on ``used < offered`` and ``rows``.
        """
        missing = tuple(name for name in self.required_columns if name not in batch.columns)
        if missing:
            raise BatchRefusal(
                f"field columns: {len(missing)} of "
                f"{len(self.required_columns)} required inputs absent for "
                f"rloo: {{{', '.join(missing)}}}"
            )
        batch_rows = len(batch)
        if batch_rows == 0:
            raise BatchRefusal(
                "field rows: 0 of at least 1 required batch rows were "
                "supplied for rloo; a token mean over zero rows is "
                "unmeasured, never 0.0"
            )
        prompt_ids = _outer_sequence(
            batch.column(self.prompt_id_column),
            field_name=self.prompt_id_column,
            refusal=BatchRefusal,
        )
        rewards = _outer_sequence(
            batch.column(self.reward_column),
            field_name=self.reward_column,
            refusal=BatchRefusal,
        )
        mask_rows = _outer_sequence(
            batch.column(self.mask_column),
            field_name=self.mask_column,
            refusal=BatchRefusal,
        )
        old_rows = _outer_sequence(
            batch.column(self.old_logprob_column),
            field_name=self.old_logprob_column,
            refusal=BatchRefusal,
        )
        for field_name, entries in (
            (self.prompt_id_column, prompt_ids),
            (self.reward_column, rewards),
            (self.mask_column, mask_rows),
            (self.old_logprob_column, old_rows),
        ):
            if len(entries) != batch_rows:
                raise BatchRefusal(
                    f"field {field_name}: {len(entries)} entries were "
                    f"supplied for {batch_rows} batch rows; the two counts "
                    f"must be the one offered denominator"
                )
        groups: dict[Any, list[int]] = {}
        for row, prompt_id in enumerate(prompt_ids):
            try:
                groups.setdefault(prompt_id, []).append(row)
            except TypeError as exc:
                raise BatchRefusal(
                    f"prompt id at row {row} is {prompt_id!r}, which is "
                    f"not hashable ({type(prompt_id).__name__}); prompt ids "
                    f"must support the same grouping the estimator performs"
                ) from exc
        for prompt_id, rows in groups.items():
            if len(rows) != self.group_size:
                raise BatchRefusal(
                    f"field {self.prompt_id_column}: prompt {prompt_id!r} "
                    f"carries {len(rows)} of {batch_rows} batch rows but "
                    f"RLOO declares group_size={self.group_size}; 1 prompt "
                    f"group disagrees with 1 declared K"
                )
        advantage = self.advantage_fn.compute(
            prompt_ids=prompt_ids,
            rewards=rewards,
            mask=mask_rows,
        )
        if advantage.offered != batch_rows:
            raise BatchRefusal(
                f"AdvantageResult.offered={advantage.offered} but the batch "
                f"contains {batch_rows} rows; {advantage.offered} of "
                f"{batch_rows} offered rows cannot anchor an RLOO step"
            )
        current_outer = _outer_sequence(
            forward_fn(batch),
            field_name="forward_fn(batch)",
            refusal=BatchRefusal,
        )
        if len(current_outer) != batch_rows:
            raise BatchRefusal(
                f"forward_fn returned {len(current_outer)} rows for a "
                f"batch of {batch_rows} rows; one token row per batch row "
                f"is required"
            )
        policy_objective_total = 0.0
        supervised = 0
        for result_row, batch_row in enumerate(advantage.rows):
            mask = tuple(
                _mask_entry(raw_mask_entry, batch_row, position)
                for position, raw_mask_entry in enumerate(
                    _row_sequence(
                        mask_rows[batch_row],
                        field_name=self.mask_column,
                        row=batch_row,
                    )
                )
            )
            weights = tuple(advantage.weights[result_row])
            current = _row_sequence(
                current_outer[batch_row],
                field_name="forward_fn output",
                row=batch_row,
            )
            old = _row_sequence(
                old_rows[batch_row],
                field_name=self.old_logprob_column,
                row=batch_row,
            )
            lengths = (len(mask), len(weights), len(current), len(old))
            if len(set(lengths)) != 1:
                raise BatchRefusal(
                    f"row {batch_row}: mask/advantage/current/old lengths "
                    f"are {lengths}; all 4 token denominators must be "
                    f"equal before one token ratio can be attributed"
                )
            for position, is_supervised in enumerate(mask):
                if not is_supervised:
                    continue
                weight = _advantage_weight(weights[position], batch_row, position)
                current_value = _logprob_value(
                    current[position], "current_logprobs", batch_row, position
                )
                old_value = _logprob_value(
                    old[position], self.old_logprob_column, batch_row, position
                )
                ratio = _token_ratio(current_value, old_value, batch_row, position)
                policy_objective_total += ratio * weight
                supervised += 1
        if supervised == 0:
            raise SupervisionRefusal(
                f"0 supervised tokens remain across {advantage.used} used "
                f"rows; the RLOO loss is unmeasured and returning 0.0 "
                f"would report a perfect loss over no gradient"
            )
        contribution = -self.weight * (policy_objective_total / supervised)
        return (
            LossOutput(
                loss=contribution,
                components=(
                    LossComponent(
                        name=self.component_name,
                        weight=self.weight,
                        observed=True,
                        contribution=contribution,
                    ),
                ),
            ),
            advantage,
        )


@dataclass(frozen=True, slots=True)
class _RlooWiring:
    loss_fn: RLOOPolicyLoss
    dataloader: Iterator[ExperienceBatch]
    policy_logprob_column: str


class RLOOAlgorithm:
    """Concrete batch-step RLOO algorithm bound to the existing estimator.

    WHAT IS CLAIMED: one successful step calls the wired
    :class:`RLOOPolicyLoss`, keeps its returned ``AdvantageResult``
    denominator, and returns a ``StepReport`` whose reward statistics cover
    exactly the priced rows.

    WHAT IS NOT CLAIMED: that the dataloader rows are fresh/on-policy, that
    rollout or weight synchronisation is unnecessary outside this
    batch-consuming binding (both roles are declared False, graded, not
    silently absent), or that a reference-free objective needs no reference
    comparison anywhere else in the run.
    """

    __slots__ = (
        "_next_step",
        "_requirements",
        "_requires",
        "_semantics",
        "_supplied",
        "_wiring",
    )

    def __init__(self, *, group_size: int = 2) -> None:
        """Construct RLOO's declarations independently of any supplied loss.

        WHAT IS CLAIMED: invalid semantic configuration is refused at
        construction via a temporary default-parameter loss, mirroring GRPO.

        WHAT IS NOT CLAIMED: that a later loss agrees; the setup handshake
        compares this declaration against the loss's own declaration. Note
        the inherited GRPO quirk: with the default estimator's
        ``min_group_size`` of 2, only ``group_size=2`` passes here.
        """
        RLOOPolicyLoss(group_size=group_size)
        self._semantics = AlgorithmSemantics(
            group_size=group_size,
            ratio_scope="token",
            kl_estimator=None,
            clip_bounds=None,
            reference_free=True,
        )
        self._requirements = AlgorithmRequirements(
            name="rloo",
            requires={
                "rollout_source": False,
                "advantage_fn": True,
                "weight_sync": False,
                "reference_policy": False,
            },
            declared_components=("rloo_policy_loss",),
            declared_metrics=(),
            semantics=self._semantics,
        )
        self._requires: Mapping[str, bool] = MappingProxyType(
            {
                "policy_pair": True,
                "loss_fn": True,
                "dataloader": True,
                "advantage_fn": True,
                "reference_policy": False,
                "rollout_source": False,
                "weight_sync": False,
            }
        )
        self._supplied: Mapping[str, Any] | None = None
        self._wiring: _RlooWiring | None = None
        self._next_step = 0

    def requirements(self) -> AlgorithmRequirements:
        """Return the role and objective declaration measured by gates.

        WHAT IS CLAIMED: the record is this binding's declaration.

        WHAT IS NOT CLAIMED: that the declaration is evidence the mathematics
        are correct; it is the denominator the gates measure.
        """
        return self._requirements

    def semantics(self) -> AlgorithmSemantics:
        """Return RLOO's algorithm-side semantics declaration.

        WHAT IS CLAIMED: values came from this algorithm's constructor,
        independent of any later loss object.

        WHAT IS NOT CLAIMED: algorithm/loss agreement; ``setup`` measures it.
        """
        return self._semantics

    def requires(self) -> Mapping[str, bool]:
        """Return the static role-requirement denominator.

        WHAT IS CLAIMED: an immutable string-to-bool declaration with at
        least one required role.

        WHAT IS NOT CLAIMED: that required roles are currently present.
        """
        return self._requires

    def supplied(self) -> Mapping[str, Any] | None:
        """Return the actual post-setup wiring, or abstain before setup.

        WHAT IS CLAIMED: after setup, the immutable mapping names exactly
        what this algorithm was handed.

        WHAT IS NOT CLAIMED before setup: any wiring measurement. The
        pre-setup value is ``None`` because the supplied set was not
        measured; an empty map would falsely read as a completed check.
        """
        return self._supplied

    def setup(
        self,
        *,
        policy_pair: PolicyPair,
        loss_fn: Any,
        dataloader: Iterable[ExperienceBatch],
        config: Mapping[str, Any],
        rollout_source: RolloutSource | None = None,
        advantage_fn: AdvantageFn | None = None,
        weight_sync: WeightSync | None = None,
    ) -> None:
        """Wire RLOO and perform its declaration-to-loss handshake.

        WHAT IS CLAIMED on success: the concrete loss is ``RLOOPolicyLoss``,
        its semantics equal this algorithm's, the supplied estimator IS the
        loss-owned estimator object (identity, not equality -- two
        leave-one-out estimators that agree everywhere still double-count the
        one declared role), and both role-map checks name the same consumed
        set.

        WHAT IS NOT CLAIMED: that any batch exists, is grouped to the
        declared K, or is finite; data failures are step-time refusals.
        """
        if self._wiring is not None:
            raise AlgorithmWiringRefusal(
                "field setup: 1 of 1 RLOO algorithm objects was already "
                "wired; calling setup twice would silently replace the "
                "measured supplied mapping"
            )
        if not isinstance(loss_fn, RLOOPolicyLoss):
            raise AlgorithmWiringRefusal(
                f"field loss_fn={loss_fn!r}: RLOO requires RLOOPolicyLoss; "
                f"1 of 1 loss slots carries {type(loss_fn).__name__}"
            )
        if advantage_fn is None:
            raise AlgorithmWiringRefusal(
                "field advantage_fn: 1 of 1 required inputs absent for rloo: {'advantage_fn'}"
            )
        if advantage_fn is not loss_fn.advantage_fn:
            raise AlgorithmWiringRefusal(
                "field advantage_fn: the setup object and the loss-owned "
                "estimator are 2 distinct objects for 1 declared role; "
                "the concrete RLOO binding must wire the same "
                "LeaveOneOutAdvantage instance through both sides"
            )
        loss_semantics = loss_fn.semantics()
        for field_name in (
            "group_size",
            "ratio_scope",
            "kl_estimator",
            "clip_bounds",
            "reference_free",
        ):
            algorithm_value = getattr(self._semantics, field_name)
            loss_value = getattr(loss_semantics, field_name)
            if algorithm_value != loss_value:
                raise AlgorithmWiringRefusal(
                    f"field {field_name}: algorithm declares "
                    f"{algorithm_value!r} but loss declares {loss_value!r}; "
                    f"1 of 5 semantics fields disagrees between the 2 "
                    f"declarations"
                )
        consumed = check_algorithm_wiring(
            self._requirements,
            policy_pair=policy_pair,
            loss_fn=loss_fn,
            supplied={
                "rollout_source": rollout_source,
                "advantage_fn": advantage_fn,
                "weight_sync": weight_sync,
            },
            origin="RLOOAlgorithm.setup",
        )
        try:
            dataloader_iterator = iter(dataloader)
        except TypeError as exc:
            raise AlgorithmWiringRefusal(
                f"field dataloader={dataloader!r}: 1 of 1 mandatory "
                f"step inputs was not iterable "
                f"({type(dataloader).__name__})"
            ) from exc
        if not isinstance(config, Mapping):
            raise AlgorithmWiringRefusal(
                f"field config={config!r}: 1 of 1 setup configurations must implement Mapping"
            )
        column = config.get("policy_logprob_column")
        if not isinstance(column, str) or not column:
            raise AlgorithmWiringRefusal(
                "field policy_logprob_column: 0 of 1 required config "
                "values names a non-empty batch column for rloo; current "
                "token log-probabilities cannot be attributed to the "
                "existing LossFn surface without that column"
            )
        supplied = MappingProxyType(
            {
                "policy_pair": policy_pair,
                "loss_fn": loss_fn,
                "dataloader": dataloader,
                "advantage_fn": advantage_fn,
            }
        )
        mapped_consumed = check_rloo_requirements(
            requires=self._requires,
            supplied=supplied,
            origin="rloo",
        )
        if mapped_consumed != tuple(
            sorted(
                consumed
                + (
                    "policy_pair",
                    "loss_fn",
                    "dataloader",
                )
            )
        ):
            raise AlgorithmWiringRefusal(
                f"mapped roles {mapped_consumed!r} disagree with the "
                f"existing handshake roles {consumed!r} plus the 3 "
                f"mandatory setup roles; 2 independently checked role "
                f"denominators must name the same consumed set"
            )
        self._supplied = supplied
        self._wiring = _RlooWiring(
            loss_fn=loss_fn,
            dataloader=dataloader_iterator,
            policy_logprob_column=column,
        )

    def step(self) -> StepReport:
        """Price the next batch and return its honest reward denominator.

        WHAT IS CLAIMED: ``rows`` is the estimator's compacted ``used``
        count rather than the offered batch length.

        WHAT IS NOT CLAIMED: that another batch exists after this one, or
        that the result is finite.
        """
        wiring = self._wiring
        if wiring is None:
            raise AlgorithmWiringRefusal(
                "field supplied: 0 of 4 required inputs are present for "
                "rloo because setup has not completed"
            )
        try:
            batch = next(wiring.dataloader)
        except StopIteration as exc:
            raise StepReportRefusal(
                "0 of at least 1 required batches remain for rloo; the "
                "step is unmeasured rather than a step with zero rows"
            ) from exc
        column = wiring.policy_logprob_column

        def forward_fn(
            current_batch: ExperienceBatch,
        ) -> Sequence[Sequence[float]]:
            return current_batch.column(column)

        output, advantage = wiring.loss_fn.compute_with_report(forward_fn, batch)
        report = StepReport(
            step=self._next_step,
            loss=output,
            rows=advantage.used,
            reward_stats=advantage.rewards,
            sync=None,
        )
        self._next_step += 1
        return report


# ---------------------------------------------------------------------------
# REINFORCE-with-baseline
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ReinforceBaselineLoss:
    """REINFORCE with a scalar EMA baseline subtracted from each return.

    For each sample, ``advantage = reward - baseline`` is broadcast over its
    supervised tokens, and the reported component is the negative
    supervised-token mean of ``advantage * current_logprob``. The baseline
    itself is STATE the algorithm carries; this loss never invents one. When
    called with ``baseline=None`` the effective baseline is the batch's own
    mean return -- derived from data and then REPORTED as the baseline metric,
    never silently defaulted to 0.0. The metric is emitted on every step so
    declaration and observation stay in agreement under ``verify_step``.

    WHAT IS CLAIMED: the metric channel carries exactly one reading, the
    baseline value actually subtracted, and a diverging baseline is visible
    in the run manifest's declared metric set rather than hidden in state.

    WHAT IS NOT CLAIMED: that the EMA parameterisation or the mean-return
    seeding matches any reference implementation (unverified binding choice),
    that the batch is on-policy, or that the baseline reduces variance for
    any particular reward distribution.
    """

    baseline_momentum: float = 0.99
    weight: float = 1.0
    reward_column: str = "rewards"
    mask_column: str = "loss_mask"
    component_name: str = "reinforce_loss"
    baseline_metric_name: str = "reinforce_baseline_frac_above"

    def __post_init__(self) -> None:
        if isinstance(self.baseline_momentum, bool):
            raise LossConfigRefusal(
                f"baseline_momentum={self.baseline_momentum!r}: True is not "
                f"a momentum; 1 of 1 EMA momentum fields must be a real "
                f"number in (0.0, 1.0]"
            )
        if (
            not isinstance(self.baseline_momentum, (int, float))
            or not math.isfinite(self.baseline_momentum)
            or not 0.0 < self.baseline_momentum <= 1.0
        ):
            raise LossConfigRefusal(
                f"baseline_momentum={self.baseline_momentum!r}: 1 of 1 EMA "
                f"momentum fields must be a real number in (0.0, 1.0]; "
                f"outside that range the 'average' either erases its own "
                f"memory (0), forgets nothing and never adapts (above 1 "
                f"inverts it), or is not a number at all"
            )
        _positive_finite("weight", self.weight, "declared REINFORCE component weight")
        for field_name, name_value in (
            ("reward_column", self.reward_column),
            ("mask_column", self.mask_column),
            ("component_name", self.component_name),
            ("baseline_metric_name", self.baseline_metric_name),
        ):
            _checked_name(field_name, name_value)
        if self.component_name == self.baseline_metric_name:
            raise LossConfigRefusal(
                f"component_name and baseline_metric_name are both "
                f"{self.component_name!r}: 2 declared names would be 1 name "
                f"in 2 denominators -- a component is a term OF the loss "
                f"and the baseline metric is a reading of its state"
            )

    @property
    def required_columns(self) -> tuple[str, ...]:
        """Return the batch columns consumed by this loss.

        WHAT IS CLAIMED: exactly the two inputs besides the forward current
        log-probabilities the ``LossFn`` surface already supplies.

        WHAT IS NOT CLAIMED: anything about the columns' contents.
        """
        return (self.reward_column, self.mask_column)

    def declaration(self) -> LossDeclaration:
        """Declare the one component and the one state reading this loss emits.

        WHAT IS CLAIMED: the diagnostic is DECLARED, by name and with real
        bounds, on every declaration this loss produces -- the baseline state
        is visible to the metric channel and hidden in no branch. The reading
        is the FRACTION of the batch whose return is strictly above the
        carried baseline, so it is bounded on [0, 1] by construction. Both
        endpoints are declared degenerate: at 0.0 the baseline sits above
        every return and at 1.0 below every one, and in both cases it shifts
        the sign of every advantage without reducing any variance -- which is
        the only thing a baseline is there to do. A zero-variance batch reads
        0.0 for the same honest reason.

        WHAT IS NOT CLAIMED: that the raw baseline VALUE travels this channel.
        :class:`MetricExpectation` requires ``low``/``high`` deliberately --
        "an expectation with no bounds is not an expectation" -- and an EMA
        over returns is unbounded on the reward scale, which no algorithm can
        state in advance. Publishing it here would mean inventing bounds, so
        the bounded derived reading is declared instead and the raw value is
        left to the run's own logging.
        """
        return LossDeclaration(
            components=(self.component_name,),
            metrics=(
                MetricExpectation(
                    name=self.baseline_metric_name,
                    low=0.0,
                    high=1.0,
                    degenerate=(0.0, 1.0),
                ),
            ),
        )

    def semantics(self) -> AlgorithmSemantics:
        """Return this loss's independently declared REINFORCE semantics.

        WHAT IS CLAIMED: reference-freeness -- the single seam this binding
        constrains.

        WHAT IS NOT CLAIMED: any group size, ratio geometry, KL estimator,
        or clip bounds; REINFORCE-with-baseline as bound here constrains none
        of those seams, and all four abstain as ``None``.
        """
        return AlgorithmSemantics(
            group_size=None,
            ratio_scope=None,
            kl_estimator=None,
            clip_bounds=None,
            reference_free=True,
        )

    def __call__(self, forward_fn: ForwardFn, batch: ExperienceBatch) -> LossOutput:
        """Price one batch through the existing ``LossFn`` surface.

        WHAT IS CLAIMED: the result is the first value of
        ``compute_with_report`` called with ``baseline=None`` -- the
        abstained-baseline case, where the batch's mean return is used and
        reported.

        WHAT IS NOT CLAIMED: that any carried state informed the price; the
        protocol call shape cannot carry the baseline, so stateful callers
        must use ``compute_with_report``.
        """
        output, _returns_mean = self.compute_with_report(forward_fn, batch, baseline=None)
        return output

    def compute_with_report(
        self,
        forward_fn: ForwardFn,
        batch: ExperienceBatch,
        *,
        baseline: float | None,
    ) -> tuple[LossOutput, float]:
        """Compute the REINFORCE loss against one declared baseline value.

        WHAT IS CLAIMED: the emitted metric value is exactly the number
        subtracted from every return this call (the batch's mean return
        when ``baseline`` abstains), and the returned float is the batch's
        mean return -- the measurement an EMA-carrying caller needs to
        update its state with.

        WHAT IS NOT CLAIMED: that the baseline was OPTIMAL, merely that it
        was the one used and the one reported; and no grouping -- REINFORCE
        here has none, so every offered row is priced and rows equal the
        offered count by construction.
        """
        if baseline is not None:
            if isinstance(baseline, bool):
                raise LossConfigRefusal(
                    f"baseline={baseline!r}: True is not a baseline; 1 of 1 "
                    f"baseline values must be None (abstain) or a finite "
                    f"real number"
                )
            if not isinstance(baseline, (int, float)) or not math.isfinite(baseline):
                raise LossConfigRefusal(
                    f"baseline={baseline!r}: 1 of 1 baseline values must be "
                    f"None (abstain) or a finite real number; a non-finite "
                    f"baseline would poison every advantage in the batch"
                )
        missing = tuple(name for name in self.required_columns if name not in batch.columns)
        if missing:
            raise BatchRefusal(
                f"field columns: {len(missing)} of "
                f"{len(self.required_columns)} required inputs absent for "
                f"reinforce_baseline: {{{', '.join(missing)}}}"
            )
        batch_rows = len(batch)
        if batch_rows == 0:
            raise BatchRefusal(
                "field rows: 0 of at least 1 required batch rows were "
                "supplied for reinforce_baseline; a baseline over zero "
                "returns is unmeasured, never 0.0"
            )
        raw_rewards = _outer_sequence(
            batch.column(self.reward_column),
            field_name=self.reward_column,
            refusal=BatchRefusal,
        )
        mask_rows = _outer_sequence(
            batch.column(self.mask_column),
            field_name=self.mask_column,
            refusal=BatchRefusal,
        )
        if len(raw_rewards) != batch_rows or len(mask_rows) != batch_rows:
            raise BatchRefusal(
                f"fields {self.reward_column}/{self.mask_column}: "
                f"{len(raw_rewards)}/{len(mask_rows)} entries were supplied "
                f"for {batch_rows} batch rows; both counts must be the one "
                f"offered denominator"
            )
        rewards = tuple(
            _reward_scalar(raw_reward, row) for row, raw_reward in enumerate(raw_rewards)
        )
        returns_mean = sum(rewards) / batch_rows
        # Abstention resolves to the batch's own mean return, which is then
        # REPORTED like any other baseline value: the state is hidden in no
        # branch.
        used_baseline = returns_mean if baseline is None else float(baseline)
        current_outer = _outer_sequence(
            forward_fn(batch),
            field_name="forward_fn(batch)",
            refusal=BatchRefusal,
        )
        if len(current_outer) != batch_rows:
            raise BatchRefusal(
                f"forward_fn returned {len(current_outer)} rows for a "
                f"batch of {batch_rows} rows; one token row per batch row "
                f"is required"
            )
        objective_total = 0.0
        supervised = 0
        for batch_row in range(batch_rows):
            mask = tuple(
                _mask_entry(raw_mask_entry, batch_row, position)
                for position, raw_mask_entry in enumerate(
                    _row_sequence(
                        mask_rows[batch_row],
                        field_name=self.mask_column,
                        row=batch_row,
                    )
                )
            )
            current = _row_sequence(
                current_outer[batch_row],
                field_name="forward_fn output",
                row=batch_row,
            )
            if len(mask) != len(current):
                raise BatchRefusal(
                    f"row {batch_row}: mask/current lengths are "
                    f"{(len(mask), len(current))}; the 2 token denominators "
                    f"must be equal before one sample return can be "
                    f"attributed to its tokens"
                )
            sample_advantage = rewards[batch_row] - used_baseline
            for position, is_supervised in enumerate(mask):
                if not is_supervised:
                    continue
                current_value = _logprob_value(
                    current[position], "current_logprobs", batch_row, position
                )
                objective_total += sample_advantage * current_value
                supervised += 1
        if supervised == 0:
            raise SupervisionRefusal(
                f"0 supervised tokens remain across {batch_rows} rows; the "
                f"REINFORCE loss is unmeasured and returning 0.0 would "
                f"report a perfect loss over no gradient"
            )
        contribution = -self.weight * (objective_total / supervised)
        # The bounded diagnostic declared above: strictly-above, so a batch
        # whose returns all EQUAL the baseline reads 0.0 -- degenerate, and
        # truthfully so, because such a baseline reduces no variance.
        frac_above = sum(1 for reward in rewards if reward > used_baseline) / batch_rows
        return (
            LossOutput(
                loss=contribution,
                components=(
                    LossComponent(
                        name=self.component_name,
                        weight=self.weight,
                        observed=True,
                        contribution=contribution,
                    ),
                ),
                metrics=(
                    MetricObservation(
                        name=self.baseline_metric_name,
                        value=frac_above,
                    ),
                ),
            ),
            returns_mean,
        )


@dataclass(frozen=True, slots=True)
class _ReinforceBaselineWiring:
    loss_fn: ReinforceBaselineLoss
    dataloader: Iterator[ExperienceBatch]
    policy_logprob_column: str
    baseline_momentum: float


class ReinforceBaselineAlgorithm:
    """REINFORCE-with-baseline: the binding that CARRIES the baseline state.

    The baseline is updated after every step as
    ``momentum * baseline + (1 - momentum) * batch_mean_return``, seeded from
    the first batch's mean return -- never from a defaulted 0.0, because 0.0
    is a measurement and the first step has none. The value actually
    subtracted each step is declared as a metric by
    :meth:`requirements` and observed by the loss on every step, so
    declaration and observation cannot drift.

    WHAT IS CLAIMED: every successful step reports the baseline it used, and
    :meth:`baseline` abstains (``None``) exactly until the first step seeds
    the state.

    WHAT IS NOT CLAIMED: that the EMA tracks any optimum, that its momentum
    matches a reference implementation, or that reporting a baseline makes
    the gradient estimate unbiased for batches that are not on-policy.
    """

    __slots__ = (
        "_baseline_momentum",
        "_baseline_state",
        "_next_step",
        "_requirements",
        "_requires",
        "_semantics",
        "_supplied",
        "_wiring",
    )

    def __init__(self, *, baseline_momentum: float = 0.99) -> None:
        """Construct the declarations and validate the momentum up front.

        WHAT IS CLAIMED: an invalid momentum is refused at construction via
        a temporary default-parameter loss, mirroring GRPO's constructor
        discipline -- refusing here costs nothing.

        WHAT IS NOT CLAIMED: that a later loss carries the same momentum;
        ``setup`` refuses a disagreement between the two declarations.
        """
        ReinforceBaselineLoss(baseline_momentum=baseline_momentum)
        # Stored because setup() must compare the loss's declaration against
        # the value THIS object validated. Re-deriving it from a second
        # temporary loss would make one countable have two editions.
        self._baseline_momentum = float(baseline_momentum)
        self._semantics = AlgorithmSemantics(
            group_size=None,
            ratio_scope=None,
            kl_estimator=None,
            clip_bounds=None,
            reference_free=True,
        )
        self._requirements = AlgorithmRequirements(
            name="reinforce_baseline",
            # All four OPTIONAL roles are declared False: REINFORCE here
            # consumes none of them, and a role named False is graded where
            # a role left unnamed is invisible to the wiring check.
            requires={
                "rollout_source": False,
                "advantage_fn": False,
                "weight_sync": False,
                "reference_policy": False,
            },
            declared_components=("reinforce_loss",),
            # The baseline is a reading the loss observes but does not
            # optimise -- exactly what the #316 metric channel exists for --
            # so it is declared here and never hidden in algorithm state.
            declared_metrics=("reinforce_baseline_frac_above",),
            semantics=self._semantics,
        )
        self._requires: Mapping[str, bool] = MappingProxyType(
            {
                "policy_pair": True,
                "loss_fn": True,
                "dataloader": True,
                "advantage_fn": False,
                "reference_policy": False,
                "rollout_source": False,
                "weight_sync": False,
            }
        )
        self._supplied: Mapping[str, Any] | None = None
        self._wiring: _ReinforceBaselineWiring | None = None
        self._baseline_state: float | None = None
        self._next_step = 0

    def requirements(self) -> AlgorithmRequirements:
        """Return the role, objective, and metric declaration measured by gates.

        WHAT IS CLAIMED: the declared metric set names the baseline reading.

        WHAT IS NOT CLAIMED: that the declaration is evidence of correctness;
        it is the denominator the gates measure.
        """
        return self._requirements

    def semantics(self) -> AlgorithmSemantics:
        """Return the algorithm-side semantics declaration.

        WHAT IS CLAIMED: reference_freeness, independently of any loss.

        WHAT IS NOT CLAIMED: any value on the other four seams -- they
        abstain as ``None`` because this binding constrains none of them.
        """
        return self._semantics

    def requires(self) -> Mapping[str, bool]:
        """Return the static role-requirement denominator.

        WHAT IS CLAIMED: an immutable declaration; only the three mandatory
        setup roles are required.

        WHAT IS NOT CLAIMED: that required roles are currently present.
        """
        return self._requires

    def supplied(self) -> Mapping[str, Any] | None:
        """Return the post-setup wiring, or abstain (``None``) before setup.

        WHAT IS CLAIMED: after setup, exactly what this algorithm was handed.

        WHAT IS NOT CLAIMED before setup: any wiring measurement.
        """
        return self._supplied

    def baseline(self) -> float | None:
        """Return the carried EMA baseline, or abstain before the first step.

        WHAT IS CLAIMED: ``None`` means UNSEEDED -- no batch has been priced
        yet -- and never means a baseline of 0.0. After the first step the
        value is a real measurement-derived number.

        WHAT IS NOT CLAIMED: that the returned value was the one any
        particular step used; per-step values are on the steps' reports,
        this accessor answers the current state.
        """
        return self._baseline_state

    def setup(
        self,
        *,
        policy_pair: PolicyPair,
        loss_fn: Any,
        dataloader: Iterable[ExperienceBatch],
        config: Mapping[str, Any],
        rollout_source: RolloutSource | None = None,
        advantage_fn: AdvantageFn | None = None,
        weight_sync: WeightSync | None = None,
    ) -> None:
        """Wire REINFORCE-with-baseline and perform its handshake.

        WHAT IS CLAIMED on success: the concrete loss is
        ``ReinforceBaselineLoss``, its momentum equals this algorithm's (the
        momentum is declared twice, in two objects, exactly so a disagreement
        can be refused), the five semantics fields agree for every seam
        either side constrains, and both role-map checks name the same
        consumed set.

        WHAT IS NOT CLAIMED: that any batch exists; and note the wiring
        handshake will refuse a wired advantage function or reference policy,
        because this binding declares both False.
        """
        if self._wiring is not None:
            raise AlgorithmWiringRefusal(
                "field setup: 1 of 1 ReinforceBaselineAlgorithm objects was "
                "already wired; calling setup twice would silently replace "
                "the measured supplied mapping"
            )
        if not isinstance(loss_fn, ReinforceBaselineLoss):
            raise AlgorithmWiringRefusal(
                f"field loss_fn={loss_fn!r}: REINFORCE-with-baseline "
                f"requires ReinforceBaselineLoss; 1 of 1 loss slots carries "
                f"{type(loss_fn).__name__}"
            )
        if loss_fn.baseline_momentum != self._baseline_momentum:
            raise AlgorithmWiringRefusal(
                f"field baseline_momentum: algorithm declares "
                f"{self._baseline_momentum!r} but loss declares "
                f"{loss_fn.baseline_momentum!r}; 1 of 1 baseline-momentum "
                f"declarations disagrees between the 2 objects, and the "
                f"state update would answer to whichever it happened to read"
            )
        loss_semantics = loss_fn.semantics()
        for field_name in (
            "group_size",
            "ratio_scope",
            "kl_estimator",
            "clip_bounds",
            "reference_free",
        ):
            algorithm_value = getattr(self._semantics, field_name)
            loss_value = getattr(loss_semantics, field_name)
            if algorithm_value != loss_value:
                raise AlgorithmWiringRefusal(
                    f"field {field_name}: algorithm declares "
                    f"{algorithm_value!r} but loss declares {loss_value!r}; "
                    f"1 of 5 semantics fields disagrees between the 2 "
                    f"declarations"
                )
        consumed = check_algorithm_wiring(
            self._requirements,
            policy_pair=policy_pair,
            loss_fn=loss_fn,
            supplied={
                "rollout_source": rollout_source,
                "advantage_fn": advantage_fn,
                "weight_sync": weight_sync,
            },
            origin="ReinforceBaselineAlgorithm.setup",
        )
        try:
            dataloader_iterator = iter(dataloader)
        except TypeError as exc:
            raise AlgorithmWiringRefusal(
                f"field dataloader={dataloader!r}: 1 of 1 mandatory "
                f"step inputs was not iterable "
                f"({type(dataloader).__name__})"
            ) from exc
        if not isinstance(config, Mapping):
            raise AlgorithmWiringRefusal(
                f"field config={config!r}: 1 of 1 setup configurations must implement Mapping"
            )
        column = config.get("policy_logprob_column")
        if not isinstance(column, str) or not column:
            raise AlgorithmWiringRefusal(
                "field policy_logprob_column: 0 of 1 required config "
                "values names a non-empty batch column for "
                "reinforce_baseline; current token log-probabilities "
                "cannot be attributed to the existing LossFn surface "
                "without that column"
            )
        supplied = MappingProxyType(
            {
                "policy_pair": policy_pair,
                "loss_fn": loss_fn,
                "dataloader": dataloader,
            }
        )
        mapped_consumed = check_reinforce_baseline_requirements(
            requires=self._requires,
            supplied=supplied,
            origin="reinforce_baseline",
        )
        if mapped_consumed != tuple(
            sorted(
                consumed
                + (
                    "policy_pair",
                    "loss_fn",
                    "dataloader",
                )
            )
        ):
            raise AlgorithmWiringRefusal(
                f"mapped roles {mapped_consumed!r} disagree with the "
                f"existing handshake roles {consumed!r} plus the 3 "
                f"mandatory setup roles; 2 independently checked role "
                f"denominators must name the same consumed set"
            )
        self._supplied = supplied
        self._wiring = _ReinforceBaselineWiring(
            loss_fn=loss_fn,
            dataloader=dataloader_iterator,
            policy_logprob_column=column,
            baseline_momentum=loss_fn.baseline_momentum,
        )

    def step(self) -> StepReport:
        """Price the next batch against the carried baseline, then update it.

        WHAT IS CLAIMED: the step's metric reports the baseline USED (the
        batch's own mean return on the unseeded first step), the EMA update
        happens only AFTER the price, and ``rows`` is the offered batch
        length -- REINFORCE here compacts nothing.

        WHAT IS NOT CLAIMED: that another batch exists, or that the updated
        baseline is finite for every conceivable batch -- rewards are
        refused non-finite before either.
        """
        wiring = self._wiring
        if wiring is None:
            raise AlgorithmWiringRefusal(
                "field supplied: 0 of 3 required inputs are present for "
                "reinforce_baseline because setup has not completed"
            )
        try:
            batch = next(wiring.dataloader)
        except StopIteration as exc:
            raise StepReportRefusal(
                "0 of at least 1 required batches remain for "
                "reinforce_baseline; the step is unmeasured rather than a "
                "step with zero rows"
            ) from exc
        column = wiring.policy_logprob_column

        def forward_fn(
            current_batch: ExperienceBatch,
        ) -> Sequence[Sequence[float]]:
            return current_batch.column(column)

        output, returns_mean = wiring.loss_fn.compute_with_report(
            forward_fn, batch, baseline=self._baseline_state
        )
        state = self._baseline_state
        if state is None:
            # Seeding from the first measured mean return, not from a
            # defaulted 0.0: abstention resolves into a measurement.
            state = returns_mean
        else:
            momentum = wiring.baseline_momentum
            state = momentum * state + (1.0 - momentum) * returns_mean
        self._baseline_state = state
        report = StepReport(
            step=self._next_step,
            loss=output,
            rows=len(batch),
            reward_stats=None,
            sync=None,
        )
        self._next_step += 1
        return report


# ---------------------------------------------------------------------------
# Reinforce++
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ReinforcePlusPlusLoss:
    """Reinforce++: token-level KL penalty folded into the reward, GLOBAL
    advantage normalisation, PPO-style ratio clipping.

    For each sample, the penalised return is::

        G_i = reward_i - kl_beta * sum_{supervised t} (current_it - reference_it)

    A SINGLE scalar advantage per sample is then the GLOBAL z-score::

        A_i = (G_i - mean(G)) / std(G)   (population statistics over the batch)

    and each supervised token prices the PPO-clipped surrogate::

        min(ratio_t * A_i, clip(ratio_t, 1 - eps, 1 + eps) * A_i)

    WHAT IS CLAIMED: the normalisation is global -- never per-group, so no
    prompt ids are consumed -- with population mean and std via
    :class:`RewardStats`; zero global spread REFUSES, because every z-score
    would be a manufactured 0.0, not a measurement; and the declared
    component is the single folded objective, since the penalty lives inside
    the reward rather than as a separable loss term.

    WHAT IS NOT CLAIMED: any equivalence with a reference Reinforce++
    implementation. The k1 penalty estimator, the population statistics, and
    the scalar-per-sample normalisation are unverified binding choices.
    """

    clip_epsilon: float = 0.2
    kl_beta: float = 0.04
    weight: float = 1.0
    reward_column: str = "rewards"
    mask_column: str = "loss_mask"
    old_logprob_column: str = "old_logprobs"
    reference_logprob_column: str = "reference_logprobs"
    component_name: str = "reinforce_pp_loss"

    def __post_init__(self) -> None:
        if isinstance(self.clip_epsilon, bool):
            raise LossConfigRefusal(
                f"clip_epsilon={self.clip_epsilon!r}: True is not a clip "
                f"epsilon; 1 of 1 clip-epsilon fields must be a real "
                f"number in (0.0, 1.0)"
            )
        if (
            not isinstance(self.clip_epsilon, (int, float))
            or not math.isfinite(self.clip_epsilon)
            or not 0.0 < self.clip_epsilon < 1.0
        ):
            raise LossConfigRefusal(
                f"clip_epsilon={self.clip_epsilon!r}: 1 of 1 clip-epsilon "
                f"fields must be a real number in (0.0, 1.0); the declared "
                f"clip bounds (1 - eps, 1 + eps) must be positive and "
                f"non-degenerate, and eps <= 0 or eps >= 1 fails one side "
                f"of that"
            )
        _positive_finite("kl_beta", self.kl_beta, "reward-penalty weight")
        _positive_finite("weight", self.weight, "declared Reinforce++ component weight")
        for field_name, name_value in (
            ("reward_column", self.reward_column),
            ("mask_column", self.mask_column),
            ("old_logprob_column", self.old_logprob_column),
            ("reference_logprob_column", self.reference_logprob_column),
            ("component_name", self.component_name),
        ):
            _checked_name(field_name, name_value)

    @property
    def clip_bounds(self) -> tuple[float, float]:
        """The declared clip interval, derived once and claimed as declared.

        WHAT IS CLAIMED: (1 - eps, 1 + eps) with the constructor's epsilon.

        WHAT IS NOT CLAIMED: that any other Reinforce++ binding uses a
        symmetric interval; symmetry is this binding's declaration.
        """
        return (1.0 - float(self.clip_epsilon), 1.0 + float(self.clip_epsilon))

    @property
    def required_columns(self) -> tuple[str, ...]:
        """Return the batch columns consumed by this loss.

        WHAT IS CLAIMED: exactly four inputs, and deliberately NOT a prompt
        id column -- normalisation here is global and needs no grouping key.

        WHAT IS NOT CLAIMED: anything about column content.
        """
        return (
            self.reward_column,
            self.mask_column,
            self.old_logprob_column,
            self.reference_logprob_column,
        )

    def declaration(self) -> LossDeclaration:
        """Declare the single folded objective component.

        WHAT IS CLAIMED: one component, because the KL penalty is folded
        INTO the reward and is not a separable term of the loss.

        WHAT IS NOT CLAIMED: that the penalty is invisible -- its effect is
        observable in the returned ``RewardStats`` over penalised returns.
        """
        return LossDeclaration(components=(self.component_name,))

    def semantics(self) -> AlgorithmSemantics:
        """Return this loss's independently declared Reinforce++ semantics.

        WHAT IS CLAIMED: token ratio scope, the k1 penalty estimator, both
        clip bounds, and reference dependence.

        WHAT IS NOT CLAIMED: any group size -- the normalisation is global,
        so that seam abstains as ``None``.
        """
        return AlgorithmSemantics(
            group_size=None,
            ratio_scope="token",
            kl_estimator="k1",
            clip_bounds=self.clip_bounds,
            reference_free=False,
        )

    def __call__(self, forward_fn: ForwardFn, batch: ExperienceBatch) -> LossOutput:
        """Price one batch through the existing ``LossFn`` surface.

        WHAT IS CLAIMED: the first value of ``compute_with_report``.

        WHAT IS NOT CLAIMED: that the penalised-return statistics were
        preserved; callers needing them call ``compute_with_report``.
        """
        output, _stats = self.compute_with_report(forward_fn, batch)
        return output

    def compute_with_report(
        self, forward_fn: ForwardFn, batch: ExperienceBatch
    ) -> tuple[LossOutput, RewardStats]:
        """Compute the Reinforce++ loss and retain the global statistics.

        WHAT IS CLAIMED: the returned ``RewardStats`` summarises exactly the
        ``len(batch)`` penalised returns G_i -- the set the z-scores ranged
        over, not a superset and not a subset.

        WHAT IS NOT CLAIMED: that raw (unpenalised) rewards are summarised;
        they are not, and a reader wanting them must measure separately.
        """
        missing = tuple(name for name in self.required_columns if name not in batch.columns)
        if missing:
            raise BatchRefusal(
                f"field columns: {len(missing)} of "
                f"{len(self.required_columns)} required inputs absent for "
                f"reinforce_pp: {{{', '.join(missing)}}}"
            )
        batch_rows = len(batch)
        if batch_rows == 0:
            raise BatchRefusal(
                "field rows: 0 of at least 2 required batch rows were "
                "supplied for reinforce_pp; global normalisation over zero "
                "samples is the all([]) shape -- unmeasured, never 0.0"
            )
        if batch_rows < 2:
            raise BatchRefusal(
                f"field rows: {batch_rows} of at least 2 required batch "
                f"rows were supplied for reinforce_pp; a global standard "
                f"deviation over 1 sample is 0 by construction, so the "
                f"normalised advantage would be manufactured, not measured"
            )
        raw_rewards = _outer_sequence(
            batch.column(self.reward_column),
            field_name=self.reward_column,
            refusal=BatchRefusal,
        )
        mask_rows = _outer_sequence(
            batch.column(self.mask_column),
            field_name=self.mask_column,
            refusal=BatchRefusal,
        )
        old_rows = _outer_sequence(
            batch.column(self.old_logprob_column),
            field_name=self.old_logprob_column,
            refusal=BatchRefusal,
        )
        reference_rows = _outer_sequence(
            batch.column(self.reference_logprob_column),
            field_name=self.reference_logprob_column,
            refusal=BatchRefusal,
        )
        for field_name, entries in (
            (self.reward_column, raw_rewards),
            (self.mask_column, mask_rows),
            (self.old_logprob_column, old_rows),
            (self.reference_logprob_column, reference_rows),
        ):
            if len(entries) != batch_rows:
                raise BatchRefusal(
                    f"field {field_name}: {len(entries)} entries were "
                    f"supplied for {batch_rows} batch rows; the two counts "
                    f"must be the one offered denominator"
                )
        rewards = tuple(
            _reward_scalar(raw_reward, row) for row, raw_reward in enumerate(raw_rewards)
        )
        current_outer = _outer_sequence(
            forward_fn(batch),
            field_name="forward_fn(batch)",
            refusal=BatchRefusal,
        )
        if len(current_outer) != batch_rows:
            raise BatchRefusal(
                f"forward_fn returned {len(current_outer)} rows for a "
                f"batch of {batch_rows} rows; one token row per batch row "
                f"is required"
            )
        cleaned_masks: list[tuple[bool, ...]] = []
        cleaned_currents: list[tuple[Any, ...]] = []
        cleaned_olds: list[tuple[Any, ...]] = []
        cleaned_references: list[tuple[Any, ...]] = []
        penalised: list[float] = []
        for batch_row in range(batch_rows):
            mask = tuple(
                _mask_entry(raw_mask_entry, batch_row, position)
                for position, raw_mask_entry in enumerate(
                    _row_sequence(
                        mask_rows[batch_row],
                        field_name=self.mask_column,
                        row=batch_row,
                    )
                )
            )
            current = _row_sequence(
                current_outer[batch_row],
                field_name="forward_fn output",
                row=batch_row,
            )
            old = _row_sequence(
                old_rows[batch_row],
                field_name=self.old_logprob_column,
                row=batch_row,
            )
            reference = _row_sequence(
                reference_rows[batch_row],
                field_name=self.reference_logprob_column,
                row=batch_row,
            )
            lengths = (len(mask), len(current), len(old), len(reference))
            if len(set(lengths)) != 1:
                raise BatchRefusal(
                    f"row {batch_row}: mask/current/old/reference lengths "
                    f"are {lengths}; all 4 token denominators must be "
                    f"equal before one token penalty or ratio can be "
                    f"attributed"
                )
            kl_sum = 0.0
            for penalty_position, is_penalised in enumerate(mask):
                if not is_penalised:
                    continue
                current_value = _logprob_value(
                    current[penalty_position], "current_logprobs", batch_row, penalty_position
                )
                reference_value = _logprob_value(
                    reference[penalty_position],
                    self.reference_logprob_column,
                    batch_row,
                    penalty_position,
                )
                kl_sum += current_value - reference_value
            cleaned_masks.append(mask)
            cleaned_currents.append(current)
            cleaned_olds.append(old)
            cleaned_references.append(reference)
            # The penalty is folded into the RETURN, not emitted as a
            # separable loss term: one folded component is what the
            # declaration states, and two terms would double the denominator.
            penalised.append(rewards[batch_row] - self.kl_beta * kl_sum)
        stats = RewardStats.over(tuple(penalised))
        if stats.std == 0.0:
            raise AdvantageRefusal(
                f"global advantage normalisation over {batch_rows} samples "
                f"found zero spread in the penalised returns: every "
                f"z-score would be a manufactured 0.0, and a manufactured "
                f"zero gradient is not a measurement -- the step refuses "
                f"instead of reporting a perfect-looking flat batch"
            )
        clip_low, clip_high = self.clip_bounds
        policy_objective_total = 0.0
        supervised = 0
        for batch_row in range(batch_rows):
            sample_advantage = (penalised[batch_row] - stats.mean) / stats.std
            for position, is_supervised in enumerate(cleaned_masks[batch_row]):
                if not is_supervised:
                    continue
                current_value = _logprob_value(
                    cleaned_currents[batch_row][position],
                    "current_logprobs",
                    batch_row,
                    position,
                )
                old_value = _logprob_value(
                    cleaned_olds[batch_row][position],
                    self.old_logprob_column,
                    batch_row,
                    position,
                )
                ratio = _token_ratio(current_value, old_value, batch_row, position)
                clipped_ratio = min(max(ratio, clip_low), clip_high)
                policy_objective_total += min(
                    ratio * sample_advantage,
                    clipped_ratio * sample_advantage,
                )
                supervised += 1
        if supervised == 0:
            raise SupervisionRefusal(
                f"0 supervised tokens remain across {batch_rows} rows; the "
                f"Reinforce++ loss is unmeasured and returning 0.0 would "
                f"report a perfect loss over no gradient"
            )
        contribution = -self.weight * (policy_objective_total / supervised)
        return (
            LossOutput(
                loss=contribution,
                components=(
                    LossComponent(
                        name=self.component_name,
                        weight=self.weight,
                        observed=True,
                        contribution=contribution,
                    ),
                ),
            ),
            stats,
        )


@dataclass(frozen=True, slots=True)
class _ReinforcePlusPlusWiring:
    loss_fn: ReinforcePlusPlusLoss
    dataloader: Iterator[ExperienceBatch]
    policy_logprob_column: str


class ReinforcePlusPlusAlgorithm:
    """Concrete batch-step Reinforce++ binding.

    WHAT IS CLAIMED: one successful step calls the wired
    :class:`ReinforcePlusPlusLoss` and returns a ``StepReport`` over the
    full offered batch -- global normalisation compacts nothing, so no
    advantage-function role is declared and no reward statistics ride the
    report (verify_step grades that pairing both ways).

    WHAT IS NOT CLAIMED: that the reference policy produces the penalty's
    log-probabilities (the pair only ATTESTS a reference exists), that the
    batch is on-policy, or anything about convergence.
    """

    __slots__ = (
        "_next_step",
        "_requirements",
        "_requires",
        "_semantics",
        "_supplied",
        "_wiring",
    )

    def __init__(self, *, clip_epsilon: float = 0.2) -> None:
        """Construct Reinforce++'s declarations independently of any loss.

        WHAT IS CLAIMED: an invalid epsilon is refused at construction via a
        temporary default-parameter loss, mirroring GRPO; ``kl_beta`` is
        loss-local configuration, exactly as GRPO's ``kl_weight`` is.

        WHAT IS NOT CLAIMED: that a later loss agrees on epsilon; ``setup``
        compares the two semantics declarations field by field.
        """
        ReinforcePlusPlusLoss(clip_epsilon=clip_epsilon)
        self._semantics = AlgorithmSemantics(
            group_size=None,
            ratio_scope="token",
            kl_estimator="k1",
            clip_bounds=(1.0 - float(clip_epsilon), 1.0 + float(clip_epsilon)),
            reference_free=False,
        )
        self._requirements = AlgorithmRequirements(
            name="reinforce_pp",
            requires={
                "rollout_source": False,
                "advantage_fn": False,
                "weight_sync": False,
                "reference_policy": True,
            },
            declared_components=("reinforce_pp_loss",),
            declared_metrics=(),
            semantics=self._semantics,
        )
        self._requires: Mapping[str, bool] = MappingProxyType(
            {
                "policy_pair": True,
                "loss_fn": True,
                "dataloader": True,
                "advantage_fn": False,
                "reference_policy": True,
                "rollout_source": False,
                "weight_sync": False,
            }
        )
        self._supplied: Mapping[str, Any] | None = None
        self._wiring: _ReinforcePlusPlusWiring | None = None
        self._next_step = 0

    def requirements(self) -> AlgorithmRequirements:
        """Return the role and objective declaration measured by gates.

        WHAT IS CLAIMED: the record is this binding's declaration, including
        the reference-policy requirement.

        WHAT IS NOT CLAIMED: evidentiary correctness of the mathematics.
        """
        return self._requirements

    def semantics(self) -> AlgorithmSemantics:
        """Return Reinforce++'s algorithm-side semantics declaration.

        WHAT IS CLAIMED: token scope, k1, both clip bounds, reference need,
        each from this constructor.

        WHAT IS NOT CLAIMED: algorithm/loss agreement; ``setup`` measures it.
        """
        return self._semantics

    def requires(self) -> Mapping[str, bool]:
        """Return the static role-requirement denominator.

        WHAT IS CLAIMED: an immutable declaration with its required roles.

        WHAT IS NOT CLAIMED: that required roles are currently present.
        """
        return self._requires

    def supplied(self) -> Mapping[str, Any] | None:
        """Return the post-setup wiring, or abstain (``None``) before setup.

        WHAT IS CLAIMED: after setup, exactly what this algorithm was handed.

        WHAT IS NOT CLAIMED before setup: any wiring measurement.
        """
        return self._supplied

    def setup(
        self,
        *,
        policy_pair: PolicyPair,
        loss_fn: Any,
        dataloader: Iterable[ExperienceBatch],
        config: Mapping[str, Any],
        rollout_source: RolloutSource | None = None,
        advantage_fn: AdvantageFn | None = None,
        weight_sync: WeightSync | None = None,
    ) -> None:
        """Wire Reinforce++ and perform its declaration-to-loss handshake.

        WHAT IS CLAIMED on success: the concrete loss is
        ``ReinforcePlusPlusLoss``, the 5-fields semantics agree, the policy
        pair attested the reference role through ``check_algorithm_wiring``,
        and both role-map checks name the same consumed set.

        WHAT IS NOT CLAIMED: that any batch exists or has nonzero global
        reward spread; those are step-time refusals.
        """
        if self._wiring is not None:
            raise AlgorithmWiringRefusal(
                "field setup: 1 of 1 ReinforcePlusPlusAlgorithm objects was "
                "already wired; calling setup twice would silently replace "
                "the measured supplied mapping"
            )
        if not isinstance(loss_fn, ReinforcePlusPlusLoss):
            raise AlgorithmWiringRefusal(
                f"field loss_fn={loss_fn!r}: Reinforce++ requires "
                f"ReinforcePlusPlusLoss; 1 of 1 loss slots carries "
                f"{type(loss_fn).__name__}"
            )
        loss_semantics = loss_fn.semantics()
        for field_name in (
            "group_size",
            "ratio_scope",
            "kl_estimator",
            "clip_bounds",
            "reference_free",
        ):
            algorithm_value = getattr(self._semantics, field_name)
            loss_value = getattr(loss_semantics, field_name)
            if algorithm_value != loss_value:
                raise AlgorithmWiringRefusal(
                    f"field {field_name}: algorithm declares "
                    f"{algorithm_value!r} but loss declares {loss_value!r}; "
                    f"1 of 5 semantics fields disagrees between the 2 "
                    f"declarations"
                )
        consumed = check_algorithm_wiring(
            self._requirements,
            policy_pair=policy_pair,
            loss_fn=loss_fn,
            # reference_policy stays OUT of the supplied mapping: the pair
            # attests it, and two statements of one countable get refused.
            supplied={
                "rollout_source": rollout_source,
                "advantage_fn": advantage_fn,
                "weight_sync": weight_sync,
            },
            origin="ReinforcePlusPlusAlgorithm.setup",
        )
        try:
            dataloader_iterator = iter(dataloader)
        except TypeError as exc:
            raise AlgorithmWiringRefusal(
                f"field dataloader={dataloader!r}: 1 of 1 mandatory "
                f"step inputs was not iterable "
                f"({type(dataloader).__name__})"
            ) from exc
        if not isinstance(config, Mapping):
            raise AlgorithmWiringRefusal(
                f"field config={config!r}: 1 of 1 setup configurations must implement Mapping"
            )
        column = config.get("policy_logprob_column")
        if not isinstance(column, str) or not column:
            raise AlgorithmWiringRefusal(
                "field policy_logprob_column: 0 of 1 required config "
                "values names a non-empty batch column for reinforce_pp; "
                "current token log-probabilities cannot be attributed to "
                "the existing LossFn surface without that column"
            )
        supplied = MappingProxyType(
            {
                "policy_pair": policy_pair,
                "loss_fn": loss_fn,
                "dataloader": dataloader,
                "reference_policy": policy_pair.references,
            }
        )
        mapped_consumed = check_reinforce_pp_requirements(
            requires=self._requires,
            supplied=supplied,
            origin="reinforce_pp",
        )
        if mapped_consumed != tuple(
            sorted(
                consumed
                + (
                    "policy_pair",
                    "loss_fn",
                    "dataloader",
                )
            )
        ):
            raise AlgorithmWiringRefusal(
                f"mapped roles {mapped_consumed!r} disagree with the "
                f"existing handshake roles {consumed!r} plus the 3 "
                f"mandatory setup roles; 2 independently checked role "
                f"denominators must name the same consumed set"
            )
        self._supplied = supplied
        self._wiring = _ReinforcePlusPlusWiring(
            loss_fn=loss_fn,
            dataloader=dataloader_iterator,
            policy_logprob_column=column,
        )

    def step(self) -> StepReport:
        """Price the next batch and return the step record.

        WHAT IS CLAIMED: ``rows`` is the offered batch length (global
        normalisation uses every row) and no reward statistics ride the
        report -- the declared ``advantage_fn`` role is False, so their
        presence would be refused as an undescribed path.

        WHAT IS NOT CLAIMED: that another batch exists, or that the batch
        had nonzero spread -- the loss refuses that before this record is
        built.
        """
        wiring = self._wiring
        if wiring is None:
            raise AlgorithmWiringRefusal(
                "field supplied: 0 of 4 required inputs are present for "
                "reinforce_pp because setup has not completed"
            )
        try:
            batch = next(wiring.dataloader)
        except StopIteration as exc:
            raise StepReportRefusal(
                "0 of at least 1 required batches remain for reinforce_pp; "
                "the step is unmeasured rather than a step with zero rows"
            ) from exc
        column = wiring.policy_logprob_column

        def forward_fn(
            current_batch: ExperienceBatch,
        ) -> Sequence[Sequence[float]]:
            return current_batch.column(column)

        output, _stats = wiring.loss_fn.compute_with_report(forward_fn, batch)
        report = StepReport(
            step=self._next_step,
            loss=output,
            rows=len(batch),
            reward_stats=None,
            sync=None,
        )
        self._next_step += 1
        return report
