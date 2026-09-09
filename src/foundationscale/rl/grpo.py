"""The first concrete FoundationScale RL binding: token-level GRPO.

This module binds :class:`GroupNormalisedAdvantage` to the existing
``LossFn``/``ExperienceBatch``/``LossOutput`` loss surface. It does not widen
``AdvantageFn.compute``: the estimator continues to read exactly
``prompt_ids``, ``rewards``, and ``mask``. Current, old, and reference token
log-probabilities are loss geometry and remain outside the advantage surface.

WHAT THIS MODULE CLAIMS: a wired GRPO step prices exactly the compacted
advantage rows returned by the estimator, declares a token-scope clipped
ratio and k3 reference term, and reports reward statistics over exactly the
samples whose gradients the step contains.

WHAT THIS MODULE DOES NOT CLAIM: that GRPO's rollout producer, model forward
implementation, reference-model execution, optimizer, or weight transport is
implemented here. Those are producer or loop responsibilities; this binding
measures the pure torch-free objective over supplied log-probabilities.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Literal

from foundationscale.gates.objective_gates import LossComponent
from foundationscale.rl.advantage import (
    AdvantageFn,
    AdvantageResult,
    GroupNormalisedAdvantage,
)
from foundationscale.rl.algorithm import (
    AlgorithmRequirements,
    AlgorithmSemantics,
    AlgorithmWiringRefusal,
    StepReport,
    StepReportRefusal,
    check_algorithm_wiring,
    check_role_map,
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
    "GRPOAlgorithm",
    "GRPOPolicyLoss",
    "check_grpo_requirements",
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
            f"iterable ({type(value).__name__}); GRPO requires one entry "
            f"per token position"
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
            f"not convert to a scalar float ({type(raw).__name__}); GRPO "
            f"requires one finite target log-probability per token"
        ) from exc
    if not math.isfinite(value):
        raise BatchRefusal(
            f"field {field_name} at row {row}, position {position} is "
            f"{raw!r}, which is not finite; a non-finite token reading "
            f"would poison both the ratio and the k3 reference projection"
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
class GRPOPolicyLoss:
    """Group-relative clipped policy objective with a k3 reference term.

    For each group of exactly ``group_size`` samples sharing a prompt id,
    this loss first calls :meth:`GroupNormalisedAdvantage.compute`. For each
    used token it then computes::

        ratio = exp(current_logprob - old_logprob)
        policy_score = min(ratio * advantage,
                           clip(ratio, low, high) * advantage)
        k3 = exp(reference_logprob - current_logprob)
             - (reference_logprob - current_logprob) - 1

    The reported policy component is the negative supervised-token mean of
    ``policy_score``. The reported KL component is ``kl_weight`` times the
    same-denominator supervised-token mean of ``k3``.

    WHAT IS CLAIMED: the loss uses the existing
    :class:`GroupNormalisedAdvantage`, excludes no degenerate-group gradient
    silently, computes token-scope PPO/GRPO clipping over the estimator's
    surviving rows, and declares exactly the two components it returns.

    WHAT IS NOT CLAIMED: that old or reference log-probabilities were
    produced by any particular model, that the batch is on-policy, that a
    forward implementation exists in this torch-free module, or that the
    floating-point loss is finite for every conceivable future batch.
    """

    group_size: int = 2
    clip_low: float = 0.8
    clip_high: float = 1.2
    kl_weight: float = 0.04
    weight: float = 1.0
    prompt_id_column: str = "prompt_ids"
    reward_column: str = "rewards"
    mask_column: str = "loss_mask"
    old_logprob_column: str = "old_logprobs"
    reference_logprob_column: str = "reference_logprobs"
    component_name: str = "grpo_policy_loss"
    kl_component_name: str = "grpo_kl"
    advantage_fn: GroupNormalisedAdvantage = field(default_factory=GroupNormalisedAdvantage)

    def __post_init__(self) -> None:
        if isinstance(self.group_size, bool):
            raise LossConfigRefusal(
                f"group_size={self.group_size!r}: True is not a GRPO group "
                f"size; a group size is an integer of at least 2"
            )
        if not isinstance(self.group_size, int) or self.group_size < 2:
            raise LossConfigRefusal(
                f"group_size={self.group_size!r}: GRPO requires a group of "
                f"at least 2 samples; below that there is no within-group "
                f"baseline to estimate"
            )
        for field_name, value in (
            ("clip_low", self.clip_low),
            ("clip_high", self.clip_high),
        ):
            if not isinstance(value, (int, float)) or not math.isfinite(value):
                raise LossConfigRefusal(
                    f"{field_name}={value!r}: a GRPO ratio clip bound must be finite"
                )
        if self.clip_low <= 0.0 or not self.clip_low < self.clip_high:
            raise LossConfigRefusal(
                f"clip_bounds=({self.clip_low!r}, {self.clip_high!r}): the "
                f"lower bound must be positive and the two declared bounds "
                f"must be strictly ordered; a degenerate interval cannot "
                f"state GRPO clipping"
            )
        for field_name, value in (
            ("kl_weight", self.kl_weight),
            ("weight", self.weight),
        ):
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0.0:
                raise LossConfigRefusal(
                    f"{field_name}={value!r}: a declared GRPO component "
                    f"requires a positive finite weight; a zero or non-finite "
                    f"weight would put its contribution outside the coverage "
                    f"denominator"
                )
        # A DISTINCT loop variable, not a second binding of `value`: the weight
        # loop above is over floats and this one is over names, and reusing the
        # name makes the two element types one variable -- which is a type error
        # here and, worse, would let a numeric field silently join the name list.
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
                f"{self.component_name!r}: 2 components would share 1 name "
                f"and collapse the objective denominator"
            )
        if not isinstance(self.advantage_fn, GroupNormalisedAdvantage):
            raise LossConfigRefusal(
                f"advantage_fn={self.advantage_fn!r}: GRPO binds "
                f"GroupNormalisedAdvantage; 1 of 1 estimator slots must "
                f"carry that concrete estimator, not an undeclared "
                f"substitute"
            )
        if self.advantage_fn.min_group_size != self.group_size:
            raise LossConfigRefusal(
                f"group_size={self.group_size} and "
                f"advantage_fn.min_group_size="
                f"{self.advantage_fn.min_group_size}: 1 configured GRPO "
                f"group size disagrees with 1 estimator minimum; GRPO "
                f"treats group_size as the exact K, not merely as a lower "
                f"bound"
            )

    # The three axis properties below state, in the shared axes vocabulary,
    # facts this class already asserted in prose and in ``semantics()``. They
    # add no field, no configuration and no behaviour: ``ratio_scope`` repeats
    # ``semantics().ratio_scope``, ``clip_bounds`` pairs the two bounds
    # ``semantics()`` already pairs, and ``reduction`` names the
    # supervised-token mean the class docstring describes. They exist because
    # the generic tensor kernel reads an objective's axes rather than its
    # class, so an objective that does not speak the vocabulary is invisible
    # to it. Restating a declaration is the alternative to a per-algorithm
    # branch, and this module stays torch-free either way.
    @property
    def ratio_scope(self) -> Literal["token", "sequence"]:
        """Declare GRPO's token-scope importance ratio."""
        return "token"

    @property
    def reduction(self) -> Literal["token_mean", "sequence_mean", "constant"]:
        """Declare GRPO's supervised-token-mean policy denominator."""
        return "token_mean"

    @property
    def clip_bounds(self) -> tuple[float, float]:
        """Pair the two separately configured ratio clip bounds."""
        return (float(self.clip_low), float(self.clip_high))

    @property
    def required_columns(self) -> tuple[str, ...]:
        """Return the batch columns consumed by this loss.

        WHAT IS CLAIMED: the tuple has a stable order and contains exactly
        the five inputs needed before forward log-probabilities are read.

        WHAT IS NOT CLAIMED: that the columns contain values of the right
        shape; shapes and values are measured by ``compute_with_report``.
        """
        return (
            self.prompt_id_column,
            self.reward_column,
            self.mask_column,
            self.old_logprob_column,
            self.reference_logprob_column,
        )

    def declaration(self) -> LossDeclaration:
        """Declare the two objective components this loss always emits.

        WHAT IS CLAIMED: the declared component names are exactly the names
        present in a successful ``LossOutput``.

        WHAT IS NOT CLAIMED: that any particular step will be finite, use
        every offered row, or produce diagnostic metrics beyond the declared
        empty metric set.
        """
        return LossDeclaration(components=(self.component_name, self.kl_component_name))

    def semantics(self) -> AlgorithmSemantics:
        """Return this loss's independently declared GRPO semantics.

        WHAT IS CLAIMED: the declaration states exact group size, token ratio
        scope, the k3 estimator, both clip bounds, and the need for a
        reference distribution.

        WHAT IS NOT CLAIMED: that an independently constructed algorithm
        agrees; agreement is measured by ``GRPOAlgorithm.setup``.
        """
        return AlgorithmSemantics(
            group_size=self.group_size,
            ratio_scope="token",
            kl_estimator="k3",
            clip_bounds=(float(self.clip_low), float(self.clip_high)),
            reference_free=False,
        )

    def __call__(self, forward_fn: ForwardFn, batch: ExperienceBatch) -> LossOutput:
        """Price one batch through the existing ``LossFn`` surface.

        WHAT IS CLAIMED: the result is the first value returned by
        ``compute_with_report`` and therefore carries exactly the declared
        objective components.

        WHAT IS NOT CLAIMED: that the accompanying ``AdvantageResult`` was
        preserved; callers needing its used-row denominator should call
        ``compute_with_report``.
        """
        output, _advantage = self.compute_with_report(forward_fn, batch)
        return output

    def compute_with_report(
        self, forward_fn: ForwardFn, batch: ExperienceBatch
    ) -> tuple[LossOutput, AdvantageResult]:
        """Compute the GRPO loss and retain its advantage denominator.

        WHAT IS CLAIMED: the returned ``AdvantageResult`` names the exact
        offered rows represented in the loss, and its reward statistics are
        over the same usable samples.

        WHAT IS NOT CLAIMED: that every offered row survived group
        normalisation; a degenerate prompt group can leave the result, with
        its absence represented by ``used < offered`` and ``rows``.
        """
        missing = tuple(name for name in self.required_columns if name not in batch.columns)
        if missing:
            raise BatchRefusal(
                f"field columns: {len(missing)} of "
                f"{len(self.required_columns)} required inputs absent for "
                f"grpo: {{{', '.join(missing)}}}"
            )
        batch_rows = len(batch)
        if batch_rows == 0:
            raise BatchRefusal(
                "field rows: 0 of at least 1 required batch rows were "
                "supplied for grpo; a token mean over zero rows is "
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
        reference_rows = _outer_sequence(
            batch.column(self.reference_logprob_column),
            field_name=self.reference_logprob_column,
            refusal=BatchRefusal,
        )
        if len(prompt_ids) != batch_rows:
            raise BatchRefusal(
                f"field {self.prompt_id_column}: {len(prompt_ids)} values "
                f"were supplied for {batch_rows} batch rows; the two counts "
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
                    f"must support the same grouping the advantage function "
                    f"performs"
                ) from exc
        for prompt_id, rows in groups.items():
            if len(rows) != self.group_size:
                raise BatchRefusal(
                    f"field {self.prompt_id_column}: prompt {prompt_id!r} "
                    f"carries {len(rows)} of {batch_rows} batch rows but "
                    f"GRPO declares group_size={self.group_size}; 1 prompt "
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
                f"{batch_rows} offered rows cannot anchor a GRPO step"
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

        cleaned_mask_rows = tuple(
            tuple(
                _mask_entry(raw, batch_row, position)
                for position, raw in enumerate(
                    _row_sequence(
                        mask_rows[batch_row],
                        field_name=self.mask_column,
                        row=batch_row,
                    )
                )
            )
            for batch_row in range(batch_rows)
        )
        if len(mask_rows) != batch_rows:
            raise BatchRefusal(
                f"field {self.mask_column}: {len(mask_rows)} mask rows "
                f"were supplied for {batch_rows} batch rows"
            )
        if len(old_rows) != batch_rows:
            raise BatchRefusal(
                f"field {self.old_logprob_column}: {len(old_rows)} old "
                f"log-probability rows were supplied for {batch_rows} "
                f"batch rows"
            )
        if len(reference_rows) != batch_rows:
            raise BatchRefusal(
                f"field {self.reference_logprob_column}: "
                f"{len(reference_rows)} reference rows were supplied for "
                f"{batch_rows} batch rows"
            )

        policy_objective_total = 0.0
        kl_total = 0.0
        supervised = 0
        for result_row, batch_row in enumerate(advantage.rows):
            mask = cleaned_mask_rows[batch_row]
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
            reference = _row_sequence(
                reference_rows[batch_row],
                field_name=self.reference_logprob_column,
                row=batch_row,
            )
            lengths = (
                len(mask),
                len(weights),
                len(current),
                len(old),
                len(reference),
            )
            if len(set(lengths)) != 1:
                raise BatchRefusal(
                    f"row {batch_row}: mask/advantage/current/old/"
                    f"reference lengths are {lengths}; all 5 token "
                    f"denominators must be equal before one token ratio "
                    f"can be attributed"
                )
            for position, is_supervised in enumerate(mask):
                weight = _advantage_weight(weights[position], batch_row, position)
                current_value = _target_logprob(
                    current[position],
                    "current_logprobs",
                    batch_row,
                    position,
                )
                old_value = _target_logprob(
                    old[position],
                    self.old_logprob_column,
                    batch_row,
                    position,
                )
                reference_value = _target_logprob(
                    reference[position],
                    self.reference_logprob_column,
                    batch_row,
                    position,
                )
                if not is_supervised:
                    continue
                log_ratio = current_value - old_value
                try:
                    ratio = math.exp(log_ratio)
                except OverflowError as exc:
                    raise BatchRefusal(
                        f"row {batch_row}, position {position}: token "
                        f"ratio exp({log_ratio!r}) is not representable; "
                        f"1 of 1 token ratios failed before clipping"
                    ) from exc
                if ratio <= 0.0 or not math.isfinite(ratio):
                    raise BatchRefusal(
                        f"row {batch_row}, position {position}: token "
                        f"ratio {ratio!r} is not a positive finite value; "
                        f"a silently underflowed ratio would manufacture "
                        f"a measured-looking zero advantage contribution"
                    )
                clipped_ratio = min(max(ratio, self.clip_low), self.clip_high)
                policy_objective_total += min(
                    ratio * weight,
                    clipped_ratio * weight,
                )
                log_reference_ratio = reference_value - current_value
                try:
                    k3 = math.expm1(log_reference_ratio) - log_reference_ratio
                except OverflowError as exc:
                    raise BatchRefusal(
                        f"row {batch_row}, position {position}: k3 "
                        f"projection exp({log_reference_ratio!r}) is not "
                        f"representable; 1 of 1 reference-token readings "
                        f"failed"
                    ) from exc
                if -1.0e-15 < k3 < 0.0:
                    k3 = 0.0
                if k3 < 0.0 or not math.isfinite(k3):
                    raise BatchRefusal(
                        f"row {batch_row}, position {position}: k3 value "
                        f"{k3!r} is not a finite nonnegative measurement"
                    )
                kl_total += k3
                supervised += 1

        if supervised == 0:
            raise SupervisionRefusal(
                f"0 supervised tokens remain across {advantage.used} used "
                f"rows; the GRPO loss is unmeasured and returning 0.0 "
                f"would report a perfect loss over no gradient"
            )
        policy_contribution = -self.weight * (policy_objective_total / supervised)
        kl_contribution = self.kl_weight * (kl_total / supervised)
        components = (
            LossComponent(
                name=self.component_name,
                weight=self.weight,
                observed=True,
                contribution=policy_contribution,
            ),
            LossComponent(
                name=self.kl_component_name,
                weight=self.kl_weight,
                observed=True,
                contribution=kl_contribution,
            ),
        )
        return (
            LossOutput(
                loss=policy_contribution + kl_contribution,
                components=components,
            ),
            advantage,
        )


def check_grpo_requirements(
    *,
    requires: Mapping[str, bool],
    supplied: Mapping[str, Any],
    origin: str = "grpo",
) -> tuple[str, ...]:
    """Check GRPO's role map over the union of both mapping key sets.

    WHAT IS CLAIMED on success: every required role names a supplied object,
    no required set was emptily satisfied, no supplied required-role value is
    ``None``, and the returned tuple is the sorted set of roles actually
    consumed.

    WHAT IS NOT CLAIMED: that any role object is a valid model, dataloader,
    estimator, or loss. Type and semantic checks remain with the component
    handshakes; this helper checks the declared denominator only. The
    implementation is shared with every other family's role-map check --
    only the ``origin`` this binding names differs.
    """
    return check_role_map(requires=requires, supplied=supplied, origin=origin)


@dataclass(frozen=True, slots=True)
class _Wiring:
    loss_fn: GRPOPolicyLoss
    dataloader: Iterator[ExperienceBatch]
    policy_logprob_column: str


class GRPOAlgorithm:
    """Concrete batch-step GRPO algorithm bound to the existing estimator.

    This binding consumes completed ``ExperienceBatch`` objects from its
    mandatory dataloader; rollout production is upstream of the step. It
    therefore declares no rollout-source or weight-sync object. It does
    require an advantage function and a reference distribution because the
    implemented objective contains group normalisation and k3.

    WHAT IS CLAIMED: one successful step calls the wired
    :class:`GRPOPolicyLoss`, keeps its returned ``AdvantageResult``
    denominator, and returns a ``StepReport`` whose reward statistics cover
    exactly the priced rows.

    WHAT IS NOT CLAIMED: that the dataloader rows are fresh/on-policy, that
    the policy pair's forward implementation is invoked by this torch-free
    object, or that rollout/weight synchronization is unnecessary outside
    this batch-consuming binding.
    """

    __slots__ = (
        "_next_step",
        "_requirements",
        "_requires",
        "_semantics",
        "_supplied",
        "_wiring",
    )

    def __init__(
        self,
        *,
        group_size: int = 2,
        clip_low: float = 0.8,
        clip_high: float = 1.2,
    ) -> None:
        """Construct GRPO's declarations independently of any supplied loss.

        WHAT IS CLAIMED: invalid semantic configuration is refused at
        construction, and the resulting declarations name the configured
        group size and clip bounds.

        WHAT IS NOT CLAIMED: that a later loss agrees; the setup handshake
        compares this declaration against the loss's own declaration.
        """
        # The temporary default-bound loss performs all shared construction
        # validation without becoming the run's loss. Setup still requires a
        # loss supplied by the caller.
        GRPOPolicyLoss(
            group_size=group_size,
            clip_low=clip_low,
            clip_high=clip_high,
        )
        self._semantics = AlgorithmSemantics(
            group_size=group_size,
            ratio_scope="token",
            kl_estimator="k3",
            clip_bounds=(float(clip_low), float(clip_high)),
            reference_free=False,
        )
        self._requirements = AlgorithmRequirements(
            name="grpo",
            # Every role this surface grades is named, including the three
            # it does NOT consume: a role declared with False is graded,
            # a role left unnamed is invisible to the check.
            requires={
                "rollout_source": False,
                "advantage_fn": True,
                "weight_sync": False,
                "reference_policy": True,
            },
            declared_components=("grpo_policy_loss", "grpo_kl"),
            declared_metrics=(),
            semantics=self._semantics,
        )
        self._requires: Mapping[str, bool] = MappingProxyType(
            {
                "policy_pair": True,
                "loss_fn": True,
                "dataloader": True,
                "advantage_fn": True,
                "reference_policy": True,
                "rollout_source": False,
                "weight_sync": False,
            }
        )
        self._supplied: Mapping[str, Any] | None = None
        self._wiring: _Wiring | None = None
        self._next_step = 0

    def requirements(self) -> AlgorithmRequirements:
        """Return the role and objective declaration measured by gates.

        WHAT IS CLAIMED: the record is this binding's declaration, including
        the two loss components and the requirement for a reference policy.

        WHAT IS NOT CLAIMED: that the declaration is sufficient evidence the
        mathematics are correct; it is the denominator the gates measure.
        """
        return self._requirements

    def semantics(self) -> AlgorithmSemantics:
        """Return GRPO's algorithm-side semantics declaration.

        WHAT IS CLAIMED: the returned values came from this algorithm's
        constructor, independent of any later loss object.

        WHAT IS NOT CLAIMED: that algorithm and loss currently agree;
        agreement is refused or established by ``setup``.
        """
        return self._semantics

    def requires(self) -> Mapping[str, bool]:
        """Return the static role-requirement denominator.

        WHAT IS CLAIMED: every entry is an immutable string-to-bool
        declaration, and the mapping contains at least one required role.

        WHAT IS NOT CLAIMED: that required roles are currently present;
        presence is represented only by ``supplied()`` after setup.
        """
        return self._requires

    def supplied(self) -> Mapping[str, Any] | None:
        """Return the actual post-setup wiring, or abstain before setup.

        WHAT IS CLAIMED: after successful setup, the immutable mapping names
        exactly the objects handed to this algorithm during that setup.

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
        """Wire GRPO and perform its declaration-to-loss handshake.

        ``config`` must contain ``policy_logprob_column``. At each step the
        column becomes the ``ForwardFn`` result for the existing ``LossFn``
        surface, keeping this module torch-free while retaining the standard
        two-argument loss call shape.

        WHAT IS CLAIMED on success: the concrete loss is ``GRPOPolicyLoss``,
        its independently computed semantics equal this algorithm's, the
        supplied estimator is the exact estimator object owned by the loss,
        the existing wiring handshake accepted the policy/reference roles,
        and the D2 role maps describe the same required-and-supplied set.

        WHAT IS NOT CLAIMED: that any batch is available, on-policy, finite,
        or grouped correctly; data failures are step-time ``BatchRefusal`` or
        ``StepReportRefusal`` facts.
        """
        if self._wiring is not None:
            raise AlgorithmWiringRefusal(
                "field setup: 1 of 1 GRPO algorithm objects was already "
                "wired; calling setup twice would silently replace the "
                "measured supplied mapping"
            )
        if not isinstance(loss_fn, GRPOPolicyLoss):
            raise AlgorithmWiringRefusal(
                f"field loss_fn={loss_fn!r}: GRPO requires "
                f"GRPOPolicyLoss; 1 of 1 loss slots carries "
                f"{type(loss_fn).__name__}"
            )
        if advantage_fn is None:
            raise AlgorithmWiringRefusal(
                "field advantage_fn: 1 of 1 required inputs absent for grpo: {'advantage_fn'}"
            )
        if advantage_fn is not loss_fn.advantage_fn:
            raise AlgorithmWiringRefusal(
                "field advantage_fn: the setup object and the loss-owned "
                "estimator are 2 distinct objects for 1 declared role; "
                "the concrete GRPO binding must wire the same "
                "GroupNormalisedAdvantage instance through both sides"
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
            # reference_policy is deliberately absent from the supplied
            # mapping: the pair attests it, and naming it here would be
            # one countable declared in two objects (the surface refuses).
            supplied={
                "rollout_source": rollout_source,
                "advantage_fn": advantage_fn,
                "weight_sync": weight_sync,
            },
            origin="GRPOAlgorithm.setup",
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
                "values names a non-empty batch column for grpo; current "
                "token log-probabilities cannot be attributed to the "
                "existing LossFn surface without that column"
            )
        supplied = MappingProxyType(
            {
                "policy_pair": policy_pair,
                "loss_fn": loss_fn,
                "dataloader": dataloader,
                "advantage_fn": advantage_fn,
                "reference_policy": policy_pair.references,
            }
        )
        mapped_consumed = check_grpo_requirements(
            requires=self._requires,
            supplied=supplied,
            origin="grpo",
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
        self._wiring = _Wiring(
            loss_fn=loss_fn,
            dataloader=dataloader_iterator,
            policy_logprob_column=column,
        )

    def step(self) -> StepReport:
        """Price the next batch and return its honest reward denominator.

        WHAT IS CLAIMED: the current-policy log-probabilities come from the
        configured batch column, the loss uses the same object supplied as
        the advantage role, and ``rows`` is the estimator's compacted ``used``
        count rather than the offered batch length.

        WHAT IS NOT CLAIMED: that another batch exists after this one, that
        the result is finite, or that rollout freshness was measured here.
        """
        wiring = self._wiring
        if wiring is None:
            raise AlgorithmWiringRefusal(
                "field supplied: 0 of 5 required inputs are present for "
                "grpo because setup has not completed"
            )
        try:
            batch = next(wiring.dataloader)
        except StopIteration as exc:
            raise StepReportRefusal(
                "0 of at least 1 required batches remain for grpo; the "
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
