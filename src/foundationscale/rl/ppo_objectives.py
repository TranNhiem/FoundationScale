"""PPO objectives: clipped surrogate policy loss, value loss, and KL control.

This module implements the classic PPO objective family on the existing
``LossFn``/``ExperienceBatch``/``LossOutput`` surface from ``interfaces.py``:
``PPOClippedPolicyLoss`` prices a batch whose advantages arrive per token,
``ValueFunctionLoss`` regresses value predictions onto returns (with optional
PPO2 value clipping), and the KL pair -- ``FixedKLCoefficient`` and
``AdaptiveKLController`` behind the ``KLCoefficientController`` protocol --
supplies the coefficient that ``KLPenaltyLoss`` applies to the k3 reference
projection.

Unlike GRPO, nothing here normalises advantages: advantage estimation (a
``GeneralisedAdvantageEstimation`` or any other estimator) is a producer-stage
decision and arrives as a batch column. There is therefore no estimator
denominator to retain after the loss, and no ``compute_with_report`` twin is
offered: every diagnostic these objectives produce fits the existing
``LossOutput.metrics`` channel declared through ``LossDeclaration.metrics``.

WHAT THIS MODULE CLAIMS: the losses price exactly the supervised tokens of a
supplied batch, refuse degenerate or non-finite batches rather than reporting
a number, and declare exactly the components and metrics they emit.

WHAT THIS MODULE DOES NOT CLAIM: that advantage or return production, old or
reference distribution execution, a model forward implementation, or the
optimizer exists here. This is a torch-free module measuring pure objectives
over supplied per-token readings.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Any, Protocol, runtime_checkable

from foundationscale.gates.objective_gates import (
    LossComponent,
    MetricExpectation,
    MetricObservation,
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

__all__ = (
    "AdaptiveKLController",
    "FixedKLCoefficient",
    "KLCoefficientController",
    "KLPenaltyLoss",
    "PPOClippedPolicyLoss",
    "ValueFunctionLoss",
)


def _checked_name(field_name: str, value: Any) -> None:
    if not isinstance(value, str) or not value:
        raise LossConfigRefusal(
            f"{field_name}={value!r}: column, component and metric names "
            f"must be non-empty strings; absence of a name is not a name"
        )


def _outer_rows(value: Any, *, field_name: str) -> tuple[Any, ...]:
    try:
        return tuple(value)
    except TypeError as exc:
        raise BatchRefusal(
            f"field {field_name}={value!r}: 1 of 1 required inputs was not "
            f"iterable ({type(value).__name__}); one outer entry per batch "
            f"row is required"
        ) from exc


def _token_row(value: Any, *, field_name: str, row: int) -> tuple[Any, ...]:
    try:
        return tuple(value)
    except TypeError as exc:
        raise BatchRefusal(
            f"field {field_name} row {row}={value!r}: 1 of 1 rows was not "
            f"iterable ({type(value).__name__}); PPO requires one entry "
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


def _token_scalar(raw: Any, field_name: str, row: int, position: int) -> float:
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise BatchRefusal(
            f"field {field_name} at row {row}, position {position} does "
            f"not convert to a scalar float ({type(raw).__name__}); PPO "
            f"requires one finite scalar per token"
        ) from exc
    if not math.isfinite(value):
        raise BatchRefusal(
            f"field {field_name} at row {row}, position {position} is "
            f"{raw!r}, which is not finite; a non-finite token reading "
            f"would poison the supervised-token mean this loss divides by"
        )
    return value


def _advantage_scalar(raw: Any, row: int, position: int) -> float:
    if isinstance(raw, bool):
        raise BatchRefusal(
            f"advantage at row {row}, position {position} is {raw!r}; "
            f"a bool is not a measured token advantage"
        )
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise BatchRefusal(
            f"advantage at row {row}, position {position} does not "
            f"convert to a scalar float ({type(raw).__name__})"
        ) from exc
    if not math.isfinite(value):
        raise BatchRefusal(
            f"advantage at row {row}, position {position} is {raw!r}, "
            f"which is not finite; the objective cannot price a "
            f"non-finite advantage as gradient"
        )
    return value


def _measured_kl(raw: Any) -> float:
    if isinstance(raw, bool):
        raise BatchRefusal(
            f"measured KL {raw!r} is a bool, not a reading; a KL "
            f"controller can only absorb an honestly measured mean KL"
        )
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise BatchRefusal(
            f"measured KL {raw!r} does not convert to a scalar float "
            f"({type(raw).__name__}); a KL controller can only absorb "
            f"an honestly measured mean KL"
        ) from exc
    if not math.isfinite(value) or value < 0.0:
        raise BatchRefusal(
            f"measured KL {raw!r} is not a finite nonnegative reading; "
            f"a k3 KL mean cannot be negative, so {raw!r} cannot be a "
            f"measured step reading"
        )
    return value


def _controller_coefficient(controller: KLCoefficientController) -> float:
    observed = controller.coefficient()
    if (
        isinstance(observed, bool)
        or not isinstance(observed, (int, float))
        or not math.isfinite(observed)
        or observed <= 0.0
    ):
        raise LossConfigRefusal(
            f"controller.coefficient() returned {observed!r}: a declared "
            f"KL component requires a positive finite coefficient; a zero "
            f"or non-finite weight would put its contribution outside the "
            f"coverage denominator"
        )
    return float(observed)


@dataclass(frozen=True, slots=True)
class PPOClippedPolicyLoss:
    """Token-level PPO clipped surrogate policy objective.

    For each supervised token the batch supplies the behaviour-policy
    log-probability and the advantage, and ``forward_fn`` supplies the
    current-policy log-probability::

        ratio  = exp(current_logprob - old_logprob)
        score  = min(ratio * advantage,
                     clip(ratio, ratio_low, ratio_high) * advantage)

    The reported component is the negative weighted supervised-token mean of
    ``score``.

    ``clip_epsilon`` alone selects the symmetric interval
    ``[1 - clip_epsilon, 1 + clip_epsilon]`` and remains the simple default.
    Setting ``clip_epsilon_low`` / ``clip_epsilon_high`` overrides the
    respective side, which is what DAPO's clip-higher recipe
    (``clip_epsilon_high > clip_epsilon_low``) will need; this object states
    the bounds and does not know DAPO. Each width must be finite and strictly
    positive, and the resolved lower ratio bound must stay positive.

    The ``clip_fraction`` metric is DIAGNOSTIC and is not a term of the
    objective: it is the proportion of supervised tokens whose ratio fell
    outside the resolved interval. A clip fraction of 0.0 on the first inner
    epoch is CORRECT -- current and old log-probabilities coincide there, so
    the ratio is identically 1 -- not evidence of a broken clip.

    WHAT IS CLAIMED: the loss prices exactly the supervised tokens of the
    supplied batch, computes token-scope PPO clipping against the supplied
    old log-probabilities, measures the clip fraction over the same
    supervised-token denominator, and REFUSES an unrepresentable ``exp``
    rather than clamping it -- this loss never silently substitutes a
    clamped ratio for a measured one, and that refusal is a stated property
    of the estimator.

    WHAT IS NOT CLAIMED: that the advantages were produced by any particular
    estimator, that the old log-probabilities came from any particular
    behaviour policy, that the batch is on-policy, that any forward
    implementation exists in this torch-free module, or that the resolved
    clip interval is a sensible width for any particular run -- only that it
    is strictly positive and well ordered.
    """

    clip_epsilon: float = 0.2
    clip_epsilon_low: float | None = None
    clip_epsilon_high: float | None = None
    weight: float = 1.0
    mask_column: str = "loss_mask"
    old_logprob_column: str = "old_logprobs"
    advantage_column: str = "advantages"
    component_name: str = "ppo_policy_loss"
    metric_name: str = "clip_fraction"
    _ratio_low: float = field(init=False, repr=False)
    _ratio_high: float = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if (
            isinstance(self.clip_epsilon, bool)
            or not isinstance(self.clip_epsilon, (int, float))
            or not math.isfinite(self.clip_epsilon)
            or self.clip_epsilon <= 0.0
        ):
            raise LossConfigRefusal(
                f"clip_epsilon={self.clip_epsilon!r}: a PPO clip width "
                f"must be finite and strictly positive; a zero or "
                f"negative width collapses the trust region to a step "
                f"function"
            )
        for field_name, override in (
            ("clip_epsilon_low", self.clip_epsilon_low),
            ("clip_epsilon_high", self.clip_epsilon_high),
        ):
            if override is None:
                continue
            if (
                isinstance(override, bool)
                or not isinstance(override, (int, float))
                or not math.isfinite(override)
                or override <= 0.0
            ):
                raise LossConfigRefusal(
                    f"{field_name}={override!r}: an asymmetric clip "
                    f"width must be finite and strictly positive; "
                    f"absence of an override is None, not 0.0"
                )
        resolved_low = float(
            self.clip_epsilon if self.clip_epsilon_low is None else self.clip_epsilon_low
        )
        resolved_high = float(
            self.clip_epsilon if self.clip_epsilon_high is None else self.clip_epsilon_high
        )
        ratio_low = 1.0 - resolved_low
        ratio_high = 1.0 + resolved_high
        if ratio_low <= 0.0:
            raise LossConfigRefusal(
                f"resolved clip interval is [{ratio_low!r}, "
                f"{ratio_high!r}] from clip widths ({resolved_low!r}, "
                f"{resolved_high!r}); the lower bound must stay "
                f"positive, so the low-side width must be strictly "
                f"below 1.0"
            )
        object.__setattr__(self, "_ratio_low", ratio_low)
        object.__setattr__(self, "_ratio_high", ratio_high)
        if (
            isinstance(self.weight, bool)
            or not isinstance(self.weight, (int, float))
            or not math.isfinite(self.weight)
            or self.weight <= 0.0
        ):
            raise LossConfigRefusal(
                f"weight={self.weight!r}: a declared PPO component "
                f"requires a positive finite weight; a zero or "
                f"non-finite weight would put its contribution "
                f"outside the coverage denominator"
            )
        for field_name, name_value in (
            ("mask_column", self.mask_column),
            ("old_logprob_column", self.old_logprob_column),
            ("advantage_column", self.advantage_column),
            ("component_name", self.component_name),
            ("metric_name", self.metric_name),
        ):
            _checked_name(field_name, name_value)

    @property
    def required_columns(self) -> tuple[str, ...]:
        """Return the batch columns consumed by this loss, in a stable order.

        WHAT IS CLAIMED: the tuple contains exactly the three inputs needed
        before forward log-probabilities are read, and its order does not
        depend on the construction site.

        WHAT IS NOT CLAIMED: that the columns carry values of the right
        shape or finiteness; shapes and values are measured by ``__call__``.
        """
        return (
            self.mask_column,
            self.old_logprob_column,
            self.advantage_column,
        )

    def declaration(self) -> LossDeclaration:
        """Declare the policy component and the clip-fraction diagnostic.

        WHAT IS CLAIMED: the metric is declared with bounds [0.0, 1.0] and
        an EMPTY degenerate set -- deliberately: 0.0 is the correct first
        inner epoch reading (the ratio is identically 1 there) and 1.0 is a
        real measured reading of a batch whose ratios all left the
        interval, so neither may be refused as degenerate.

        WHAT IS NOT CLAIMED: that this declaration is suitable input to
        ``build_objective_gate_context``; it is the run-manifest statement,
        per the ``LossDeclaration`` contract.
        """
        metric = MetricExpectation(
            name=self.metric_name,
            low=0.0,
            high=1.0,
            degenerate=(),
        )
        return LossDeclaration(components=(self.component_name,), metrics=(metric,))

    def __call__(self, forward_fn: ForwardFn, batch: ExperienceBatch) -> LossOutput:
        """Price one batch and report the surrogate plus the clip fraction.

        WHAT IS CLAIMED: the returned scalar is the negative weighted
        supervised-token mean of the clipped surrogate, computed with the
        resolved (possibly asymmetric) clip interval, and the accompanying
        metric counts exactly the supervised tokens whose ratio fell
        outside that interval.

        WHAT IS NOT CLAIMED: that the clip fraction enters the objective --
        it is diagnostic only -- or that a low clip fraction means the step
        was safe; it means only that the measured ratios stayed inside the
        stated interval.
        """
        missing = tuple(name for name in self.required_columns if name not in batch.columns)
        if missing:
            raise BatchRefusal(
                f"field columns: {len(missing)} of "
                f"{len(self.required_columns)} required inputs absent "
                f"for the ppo policy loss: {{{', '.join(missing)}}}"
            )
        batch_rows = len(batch)
        if batch_rows == 0:
            raise BatchRefusal(
                "field rows: 0 of at least 1 required batch rows were "
                "supplied for the ppo policy loss; a token mean over "
                "zero rows is unmeasured, never 0.0"
            )
        mask_rows = _outer_rows(batch.column(self.mask_column), field_name=self.mask_column)
        old_rows = _outer_rows(
            batch.column(self.old_logprob_column),
            field_name=self.old_logprob_column,
        )
        advantage_rows = _outer_rows(
            batch.column(self.advantage_column),
            field_name=self.advantage_column,
        )
        current_outer = _outer_rows(forward_fn(batch), field_name="forward_fn(batch)")
        if len(current_outer) != batch_rows:
            raise BatchRefusal(
                f"forward_fn returned {len(current_outer)} rows for a "
                f"batch of {batch_rows} rows; one token row per batch "
                f"row is required"
            )
        cleaned_masks = tuple(
            tuple(
                _mask_entry(raw, row, position)
                for position, raw in enumerate(
                    _token_row(mask_rows[row], field_name=self.mask_column, row=row)
                )
            )
            for row in range(batch_rows)
        )
        objective_total = 0.0
        clipped_tokens = 0
        supervised = 0
        for row, mask in enumerate(cleaned_masks):
            old = _token_row(old_rows[row], field_name=self.old_logprob_column, row=row)
            advantages = _token_row(advantage_rows[row], field_name=self.advantage_column, row=row)
            current = _token_row(current_outer[row], field_name="forward_fn output", row=row)
            lengths = (len(mask), len(old), len(advantages), len(current))
            if len(set(lengths)) != 1:
                raise BatchRefusal(
                    f"row {row}: mask/old/advantage/current lengths are "
                    f"{lengths}; all 4 token denominators must be equal "
                    f"before one token ratio can be attributed"
                )
            for position, is_supervised in enumerate(mask):
                if not is_supervised:
                    continue
                advantage_value = _advantage_scalar(advantages[position], row, position)
                old_value = _token_scalar(old[position], self.old_logprob_column, row, position)
                current_value = _token_scalar(current[position], "current_logprobs", row, position)
                log_ratio = current_value - old_value
                try:
                    ratio = math.exp(log_ratio)
                except OverflowError as exc:
                    raise BatchRefusal(
                        f"row {row}, position {position}: token ratio "
                        f"exp({log_ratio!r}) is not representable; 1 of 1 "
                        f"token ratios failed before clipping, and this "
                        f"loss refuses rather than clamps an "
                        f"unrepresentable ratio"
                    ) from exc
                if ratio <= 0.0 or not math.isfinite(ratio):
                    raise BatchRefusal(
                        f"row {row}, position {position}: token ratio "
                        f"{ratio!r} is not a positive finite value; a "
                        f"silently underflowed ratio would manufacture "
                        f"a measured-looking zero advantage contribution"
                    )
                if ratio < self._ratio_low or ratio > self._ratio_high:
                    clipped_tokens += 1
                clipped_ratio = min(max(ratio, self._ratio_low), self._ratio_high)
                objective_total += min(
                    ratio * advantage_value,
                    clipped_ratio * advantage_value,
                )
                supervised += 1
        if supervised == 0:
            raise SupervisionRefusal(
                f"0 supervised tokens remain across {batch_rows} offered "
                f"rows; the PPO clipped surrogate is unmeasured over "
                f"them, and returning 0.0 would report a perfect loss "
                f"for a batch that taught the policy nothing"
            )
        contribution = -self.weight * (objective_total / supervised)
        components = (
            LossComponent(
                name=self.component_name,
                weight=self.weight,
                observed=True,
                contribution=contribution,
            ),
        )
        metrics = (
            MetricObservation(
                name=self.metric_name,
                value=clipped_tokens / supervised,
            ),
        )
        return LossOutput(loss=contribution, components=components, metrics=metrics)


@dataclass(frozen=True, slots=True)
class ValueFunctionLoss:
    """Token-level value regression against returns, optionally PPO2-clipped.

    With ``clip_epsilon=None`` the loss is plain MSE. With ``clip_epsilon``
    set, each supervised token pays the LARGER of the unclipped and the
    clipped squared error (the PPO2 form)::

        unclipped = (prediction - target) ** 2
        shifted   = old_value + clamp(prediction - old_value,
                                      -clip_epsilon, clip_epsilon)
        clipped   = (shifted - target) ** 2
        token_error = max(unclipped, clipped)

    Whether clipping was APPLIED is visible in the report, not inferable by
    the caller: when clipping is configured, the loss emits the
    ``value_clip_fraction`` metric -- the proportion of supervised tokens on
    which the clipped branch strictly exceeded the unclipped squared error --
    and declares that metric in ``declaration()``. When clipping is NOT
    configured the metric is ABSENT (unmeasured), never reported as 0.0, and
    the declaration omits it: both the declaration and the computation read
    the same field (``clip_epsilon is not None``), so the two cannot diverge.

    WHAT IS CLAIMED: the mean squared error is taken over exactly the
    supervised tokens of the supplied batch, the max-of-two form above is
    precisely the PPO2 value-clipping construction, the applied count shares
    the supervised-token denominator, and a non-finite return, prediction,
    or old value is refused before it can poison the mean.

    WHAT IS NOT CLAIMED: that the returns were produced by any particular
    estimator, that the old values came from any particular checkpoint, that
    bootstrapping was done correctly upstream, or that value clipping helps
    any particular run -- only that when it is configured it is applied and
    reported as stated.
    """

    clip_epsilon: float | None = None
    weight: float = 1.0
    mask_column: str = "loss_mask"
    return_column: str = "returns"
    old_value_column: str = "old_values"
    component_name: str = "value_loss"
    metric_name: str = "value_clip_fraction"

    def __post_init__(self) -> None:
        if self.clip_epsilon is not None and (
            isinstance(self.clip_epsilon, bool)
            or not isinstance(self.clip_epsilon, (int, float))
            or not math.isfinite(self.clip_epsilon)
            or self.clip_epsilon <= 0.0
        ):
            raise LossConfigRefusal(
                f"clip_epsilon={self.clip_epsilon!r}: a value clip "
                f"width must be finite and strictly positive; "
                f"absence of clipping is None, not 0.0"
            )
        if (
            isinstance(self.weight, bool)
            or not isinstance(self.weight, (int, float))
            or not math.isfinite(self.weight)
            or self.weight <= 0.0
        ):
            raise LossConfigRefusal(
                f"weight={self.weight!r}: a declared PPO component "
                f"requires a positive finite weight; a zero or "
                f"non-finite weight would put its contribution "
                f"outside the coverage denominator"
            )
        for field_name, name_value in (
            ("mask_column", self.mask_column),
            ("return_column", self.return_column),
            ("old_value_column", self.old_value_column),
            ("component_name", self.component_name),
            ("metric_name", self.metric_name),
        ):
            _checked_name(field_name, name_value)

    @property
    def required_columns(self) -> tuple[str, ...]:
        """Return the batch columns consumed by this loss, in a stable order.

        WHAT IS CLAIMED: the tuple contains the mask and return columns,
        plus the old-value column if and only if clipping is configured --
        derived from the same field the computation reads, so schema and
        computation cannot diverge.

        WHAT IS NOT CLAIMED: that the columns carry values of the right
        shape or finiteness; shapes and values are measured by ``__call__``.
        """
        columns = [self.mask_column, self.return_column]
        if self.clip_epsilon is not None:
            columns.append(self.old_value_column)
        return tuple(columns)

    def declaration(self) -> LossDeclaration:
        """Declare the value component and, only when clipping, its metric.

        WHAT IS CLAIMED: the clip-fraction metric is declared with bounds
        [0.0, 1.0] and an empty degenerate set, and is declared if and only
        if ``clip_epsilon is not None`` -- a pure-MSE run declares no
        clipping metric and emits none, and an absent metric is unmeasured,
        not zero.

        WHAT IS NOT CLAIMED: that this declaration is suitable input to
        ``build_objective_gate_context``; it is the run-manifest statement,
        per the ``LossDeclaration`` contract.
        """
        metrics: tuple[MetricExpectation, ...] = ()
        if self.clip_epsilon is not None:
            metrics = (
                MetricExpectation(
                    name=self.metric_name,
                    low=0.0,
                    high=1.0,
                    degenerate=(),
                ),
            )
        return LossDeclaration(components=(self.component_name,), metrics=metrics)

    def __call__(self, forward_fn: ForwardFn, batch: ExperienceBatch) -> LossOutput:
        """Price one batch of value predictions against returns.

        WHAT IS CLAIMED: the returned scalar is the weighted
        supervised-token mean of the per-token error (max of clipped and
        unclipped squared errors when clipping is configured, plain squared
        error otherwise), and the applied-reading metric is emitted exactly
        when the declaration promised it.

        WHAT IS NOT CLAIMED: that clipping was necessarily active on any
        token -- the metric reports the measured proportion, which may
        correctly be 0.0 on a batch whose predictions stayed inside the
        band.
        """
        missing = tuple(name for name in self.required_columns if name not in batch.columns)
        if missing:
            raise BatchRefusal(
                f"field columns: {len(missing)} of "
                f"{len(self.required_columns)} required inputs absent "
                f"for the ppo value loss: {{{', '.join(missing)}}}"
            )
        batch_rows = len(batch)
        if batch_rows == 0:
            raise BatchRefusal(
                "field rows: 0 of at least 1 required batch rows were "
                "supplied for the ppo value loss; a token mean over "
                "zero rows is unmeasured, never 0.0"
            )
        mask_rows = _outer_rows(batch.column(self.mask_column), field_name=self.mask_column)
        return_rows = _outer_rows(batch.column(self.return_column), field_name=self.return_column)
        old_rows: tuple[Any, ...] = ()
        if self.clip_epsilon is not None:
            old_rows = _outer_rows(
                batch.column(self.old_value_column),
                field_name=self.old_value_column,
            )
        current_outer = _outer_rows(forward_fn(batch), field_name="forward_fn(batch)")
        if len(current_outer) != batch_rows:
            raise BatchRefusal(
                f"forward_fn returned {len(current_outer)} rows for a "
                f"batch of {batch_rows} rows; one token row per batch "
                f"row is required"
            )
        cleaned_masks = tuple(
            tuple(
                _mask_entry(raw, row, position)
                for position, raw in enumerate(
                    _token_row(mask_rows[row], field_name=self.mask_column, row=row)
                )
            )
            for row in range(batch_rows)
        )
        error_total = 0.0
        applied_tokens = 0
        supervised = 0
        for row, mask in enumerate(cleaned_masks):
            targets = _token_row(return_rows[row], field_name=self.return_column, row=row)
            predictions = _token_row(current_outer[row], field_name="forward_fn output", row=row)
            lengths: tuple[int, ...]
            old_predictions: tuple[Any, ...] = ()
            if self.clip_epsilon is None:
                role_names = "mask/returns/predictions"
                lengths = (len(mask), len(targets), len(predictions))
            else:
                old_predictions = _token_row(
                    old_rows[row], field_name=self.old_value_column, row=row
                )
                role_names = "mask/returns/old-values/predictions"
                lengths = (
                    len(mask),
                    len(targets),
                    len(old_predictions),
                    len(predictions),
                )
            if len(set(lengths)) != 1:
                raise BatchRefusal(
                    f"row {row}: {role_names} lengths are {lengths}; "
                    f"all token denominators must be equal before one "
                    f"token squared error can be attributed"
                )
            for position, is_supervised in enumerate(mask):
                if not is_supervised:
                    continue
                prediction = _token_scalar(
                    predictions[position], "value predictions", row, position
                )
                target = _token_scalar(targets[position], self.return_column, row, position)
                token_error = (prediction - target) ** 2
                if self.clip_epsilon is not None:
                    old_prediction = _token_scalar(
                        old_predictions[position],
                        self.old_value_column,
                        row,
                        position,
                    )
                    shift = min(
                        max(prediction - old_prediction, -self.clip_epsilon),
                        self.clip_epsilon,
                    )
                    clipped_prediction = old_prediction + shift
                    clipped_error = (clipped_prediction - target) ** 2
                    if clipped_error > token_error:
                        applied_tokens += 1
                        token_error = clipped_error
                error_total += token_error
                supervised += 1
        if supervised == 0:
            raise SupervisionRefusal(
                f"0 supervised tokens remain across {batch_rows} offered "
                f"rows; the value loss is unmeasured over them, and "
                f"returning 0.0 would report a perfect regression "
                f"against no returns"
            )
        contribution = self.weight * (error_total / supervised)
        components = (
            LossComponent(
                name=self.component_name,
                weight=self.weight,
                observed=True,
                contribution=contribution,
            ),
        )
        metrics: tuple[MetricObservation, ...] = ()
        if self.clip_epsilon is not None:
            metrics = (
                MetricObservation(
                    name=self.metric_name,
                    value=applied_tokens / supervised,
                ),
            )
        return LossOutput(loss=contribution, components=components, metrics=metrics)


@runtime_checkable
class KLCoefficientController(Protocol):
    """Supplies the KL penalty coefficient for one step and absorbs its reading.

    WHAT IS CLAIMED: ``coefficient`` returns the coefficient in force for the
    current step, and ``update`` measures the observed mean KL and returns
    the controller whose coefficient governs the NEXT step. No mutation
    occurs anywhere in this protocol: the returned object IS the new state.

    WHAT IS NOT CLAIMED: that any state is exchanged between steps -- there
    is none inside the objects -- or that the reading handed to ``update``
    came from the step this coefficient governed; the caller states that
    pairing.
    """

    def coefficient(self) -> float: ...

    def update(self, measured_kl: float) -> KLCoefficientController: ...


@dataclass(frozen=True, slots=True)
class FixedKLCoefficient:
    """A KL penalty coefficient that never moves.

    WHAT IS CLAIMED: ``coefficient`` always returns the ``value`` validated
    at construction, and ``update`` validates that it was handed an honest
    finite nonnegative reading and returns ``self`` -- a fixed coefficient
    does not respond to the reading, but it still refuses to absorb a
    corrupt one, because a corrupt reading reaching this point is a step
    failure, not a control decision.

    WHAT IS NOT CLAIMED: that any particular fixed value suits any
    particular run, or that refusing to adapt is the right control law;
    choosing fixed versus adaptive is the caller's stated decision.
    """

    value: float = 0.04

    def __post_init__(self) -> None:
        if (
            isinstance(self.value, bool)
            or not isinstance(self.value, (int, float))
            or not math.isfinite(self.value)
            or self.value <= 0.0
        ):
            raise LossConfigRefusal(
                f"value={self.value!r}: a fixed KL coefficient must be "
                f"finite and strictly positive; a zero weight would "
                f"put its declared component outside the coverage "
                f"denominator"
            )

    def coefficient(self) -> float:
        """Return the coefficient in force -- the same value at every step.

        WHAT IS CLAIMED: the returned float is the ``value`` validated at
        construction.

        WHAT IS NOT CLAIMED: any response to observed KL; there is none,
        by construction.
        """
        return float(self.value)

    def update(self, measured_kl: float) -> FixedKLCoefficient:
        """Validate the reading and return this same object unchanged.

        WHAT IS CLAIMED: a non-finite or negative reading is refused even
        though this control law discards it, and the returned object is
        ``self``.

        WHAT IS NOT CLAIMED: that the coefficient changed; it did not, and
        nothing here pretends otherwise.
        """
        _measured_kl(measured_kl)
        return self


@dataclass(frozen=True, slots=True)
class AdaptiveKLController:
    """KL controller targeting a KL value, stateful ACROSS steps.

    The object itself is frozen, as every value object in this package is;
    the STATE cannot live in a mutable field without breaking that package
    rule, so it lives in the returned object. ``update`` returns a NEW
    ``AdaptiveKLController`` carrying the updated coefficient, and the
    caller threads the state by rebinding::

        controller = controller.update(measured_kl)

    making the coefficient sequence a chain of immutable, independently
    inspectable and replayable values rather than a hidden mutation.

    The update law is multiplicative with a dead band::

        measured_kl > target_kl * tolerance_band  -> value * growth
        measured_kl < target_kl / tolerance_band  -> value * decay
        otherwise                                 -> value unchanged

    and the result is clamped into [minimum, maximum] when ``maximum`` is
    set.

    WHAT IS CLAIMED: the returned controller's ``value`` differs from this
    one's only by the stated law and clamps, no other state is introduced,
    and an unchanged reading returns ``self``.

    WHAT IS NOT CLAIMED: that this control law converges for any particular
    run, that the target was chosen well, or that the measurement fed to
    ``update`` came from the step this coefficient governed -- the caller
    states that pairing.
    """

    value: float = 0.04
    target_kl: float = 0.02
    tolerance_band: float = 1.5
    growth: float = 2.0
    decay: float = 0.5
    minimum: float = 1.0e-8
    maximum: float | None = None

    def __post_init__(self) -> None:
        for field_name, positive in (
            ("value", self.value),
            ("target_kl", self.target_kl),
            ("minimum", self.minimum),
        ):
            if (
                isinstance(positive, bool)
                or not isinstance(positive, (int, float))
                or not math.isfinite(positive)
                or positive <= 0.0
            ):
                raise LossConfigRefusal(
                    f"{field_name}={positive!r}: an adaptive KL "
                    f"controller parameter must be finite and "
                    f"strictly positive"
                )
        if (
            isinstance(self.tolerance_band, bool)
            or not isinstance(self.tolerance_band, (int, float))
            or not math.isfinite(self.tolerance_band)
            or self.tolerance_band <= 1.0
        ):
            raise LossConfigRefusal(
                f"tolerance_band={self.tolerance_band!r}: the dead band "
                f"must exceed 1.0; at 1.0 or below the band has no "
                f"interior and the controller adjusts on every reading"
            )
        if (
            isinstance(self.growth, bool)
            or not isinstance(self.growth, (int, float))
            or not math.isfinite(self.growth)
            or self.growth <= 1.0
        ):
            raise LossConfigRefusal(
                f"growth={self.growth!r}: the up-multiplier must be "
                f"finite and exceed 1.0; a value at or below 1.0 "
                f"cannot strengthen the penalty"
            )
        if (
            isinstance(self.decay, bool)
            or not isinstance(self.decay, (int, float))
            or not math.isfinite(self.decay)
            or not 0.0 < self.decay < 1.0
        ):
            raise LossConfigRefusal(
                f"decay={self.decay!r}: the down-multiplier must lie "
                f"strictly inside (0.0, 1.0); outside it the penalty "
                f"cannot weaken and remain positive"
            )
        if self.maximum is not None:
            if (
                isinstance(self.maximum, bool)
                or not isinstance(self.maximum, (int, float))
                or not math.isfinite(self.maximum)
                or self.maximum <= self.minimum
            ):
                raise LossConfigRefusal(
                    f"maximum={self.maximum!r}: the coefficient ceiling "
                    f"must be finite and strictly above "
                    f"minimum={self.minimum!r}"
                )
            if self.value > self.maximum:
                raise LossConfigRefusal(
                    f"value={self.value!r} exceeds the stated ceiling "
                    f"maximum={self.maximum!r}; the coefficient in "
                    f"force must begin inside its own clamp range"
                )
        if self.value < self.minimum:
            raise LossConfigRefusal(
                f"value={self.value!r} lies below the stated floor "
                f"minimum={self.minimum!r}; the coefficient in force "
                f"must begin inside its own clamp range"
            )

    def coefficient(self) -> float:
        """Return the coefficient in force for the current step.

        WHAT IS CLAIMED: the returned float is this object's ``value``,
        which construction guaranteed lies inside the stated clamp range.

        WHAT IS NOT CLAIMED: that the value equals any other step's; the
        chain of returned controllers is the only record of adaptation.
        """
        return float(self.value)

    def update(self, measured_kl: float) -> AdaptiveKLController:
        """Measure the mean KL and return the controller for the next step.

        WHAT IS CLAIMED: a corrupt reading is refused, the stated
        multiplicative dead-band law is the only adjustment applied, the
        result is clamped to [minimum, maximum], and a reading inside the
        dead band returns ``self`` unchanged.

        WHAT IS NOT CLAIMED: that the returned coefficient will move the
        next batch's KL towards the target; adaptation is a control
        heuristic, not a measured guarantee.
        """
        observed = _measured_kl(measured_kl)
        updated = float(self.value)
        if observed > self.target_kl * self.tolerance_band:
            updated *= self.growth
        elif observed < self.target_kl / self.tolerance_band:
            updated *= self.decay
        updated = max(float(self.minimum), updated)
        if self.maximum is not None:
            updated = min(float(self.maximum), updated)
        if updated == float(self.value):
            return self
        return replace(self, value=updated)


@dataclass(frozen=True, slots=True)
class KLPenaltyLoss:
    """KL penalty against a reference distribution, coefficient-controlled.

    For each supervised token, the k3 projection (the same estimator GRPO
    declares for its reference term)::

        k3 = exp(reference_logprob - current_logprob)
             - (reference_logprob - current_logprob) - 1

    The component is ``controller.coefficient()`` times the supervised-token
    mean of k3. A diagnostic metric carries the UNWEIGHTED mean k3: that is
    the reading an :class:`AdaptiveKLController` consumes, and the adaptive
    loop closes OUTSIDE this frozen loss as
    ``controller = controller.update(kl_estimate)`` followed by constructing
    the next step's ``KLPenaltyLoss(controller=controller)``.

    WHAT IS CLAIMED: the k3 value is nonnegative for every admitted token
    (tiny negative artefacts below 1e-15 are floored to 0.0, matching the
    GRPO handling of the same estimator), the mean shares the
    supervised-token denominator, and an unrepresentable ``exp`` is refused
    rather than clamped.

    WHAT IS NOT CLAIMED: that the k3 mean is an unbiased estimate of the
    true KL divergence -- unbiasedness is a distributional property this
    torch-free module does not measure -- that the reference came from any
    particular model, or that the controller's coefficient is well chosen;
    only that it is positive, finite, and applied as declared.
    """

    controller: KLCoefficientController = field(default_factory=FixedKLCoefficient)
    mask_column: str = "loss_mask"
    reference_logprob_column: str = "reference_logprobs"
    component_name: str = "kl_penalty"
    metric_name: str = "kl_estimate"
    metric_ceiling: float = 10.0

    def __post_init__(self) -> None:
        if not isinstance(self.controller, KLCoefficientController):
            raise LossConfigRefusal(
                f"controller={self.controller!r}: the KL coefficient "
                f"slot must carry a KLCoefficientController (a fixed or "
                f"adaptive coefficient), not "
                f"{type(self.controller).__name__}"
            )
        _controller_coefficient(self.controller)
        for field_name, name_value in (
            ("mask_column", self.mask_column),
            ("reference_logprob_column", self.reference_logprob_column),
            ("component_name", self.component_name),
            ("metric_name", self.metric_name),
        ):
            _checked_name(field_name, name_value)
        if (
            isinstance(self.metric_ceiling, bool)
            or not isinstance(self.metric_ceiling, (int, float))
            or not math.isfinite(self.metric_ceiling)
            or self.metric_ceiling <= 0.0
        ):
            raise LossConfigRefusal(
                f"metric_ceiling={self.metric_ceiling!r}: the declared upper "
                f"bound on the k3 reading must be a finite number strictly "
                f"above 0.0. An infinite bound would make every gate "
                f"comparison against it silently True, and a bounds check "
                f"that examined nothing reads as coverage"
            )

    @property
    def required_columns(self) -> tuple[str, ...]:
        """Return the batch columns consumed by this loss, in a stable order.

        WHAT IS CLAIMED: the tuple contains exactly the mask and reference
        log-probability columns.

        WHAT IS NOT CLAIMED: that the columns carry values of the right
        shape or finiteness; shapes and values are measured by ``__call__``.
        """
        return (self.mask_column, self.reference_logprob_column)

    def declaration(self) -> LossDeclaration:
        """Declare the KL component and the raw k3 estimate diagnostic.

        WHAT IS CLAIMED: the metric is declared with a lower bound of 0.0,
        an upper bound of ``metric_ceiling``, and an empty degenerate set:
        a 0.0 reading is correct for a first inner epoch on-policy batch,
        where current and reference coincide, so it is not degenerate.

        WHAT IS NOT CLAIMED: that ``metric_ceiling`` is a property of the
        k3 estimator. k3 is unbounded above; the ceiling is a DECLARED
        tripwire, set by whoever configures the loss, above which the run
        is asserted to be broken rather than merely drifting. It is stated
        as a finite number because the gate that reads this declaration
        refuses a non-finite bound outright -- an infinite bound makes
        every comparison against it silently True, and a bounds check that
        examined nothing reads as coverage. A run that legitimately
        exceeds the default should RAISE the field, not read the resulting
        gate problem as noise.

        WHAT IS NOT CLAIMED: that this declaration is suitable input to
        ``build_objective_gate_context``; it is the run-manifest statement,
        per the ``LossDeclaration`` contract.
        """
        metric = MetricExpectation(
            name=self.metric_name,
            low=0.0,
            high=float(self.metric_ceiling),
            degenerate=(),
        )
        return LossDeclaration(components=(self.component_name,), metrics=(metric,))

    def __call__(self, forward_fn: ForwardFn, batch: ExperienceBatch) -> LossOutput:
        """Price one batch's KL against the reference distribution.

        WHAT IS CLAIMED: the returned scalar is the controller's coefficient
        times the supervised-token mean of k3, the accompanying metric is
        the same mean UNWEIGHTED, and the coefficient is re-validated at
        call time so a controller whose reading decayed is refused here.

        WHAT IS NOT CLAIMED: that the adaptive loop was closed -- the
        update step is the caller's responsibility, and this loss measures
        whatever coefficient the supplied controller currently states.
        """
        missing = tuple(name for name in self.required_columns if name not in batch.columns)
        if missing:
            raise BatchRefusal(
                f"field columns: {len(missing)} of "
                f"{len(self.required_columns)} required inputs absent "
                f"for the ppo kl penalty: {{{', '.join(missing)}}}"
            )
        batch_rows = len(batch)
        if batch_rows == 0:
            raise BatchRefusal(
                "field rows: 0 of at least 1 required batch rows were "
                "supplied for the ppo kl penalty; a token mean over "
                "zero rows is unmeasured, never 0.0"
            )
        mask_rows = _outer_rows(batch.column(self.mask_column), field_name=self.mask_column)
        reference_rows = _outer_rows(
            batch.column(self.reference_logprob_column),
            field_name=self.reference_logprob_column,
        )
        current_outer = _outer_rows(forward_fn(batch), field_name="forward_fn(batch)")
        if len(current_outer) != batch_rows:
            raise BatchRefusal(
                f"forward_fn returned {len(current_outer)} rows for a "
                f"batch of {batch_rows} rows; one token row per batch "
                f"row is required"
            )
        cleaned_masks = tuple(
            tuple(
                _mask_entry(raw, row, position)
                for position, raw in enumerate(
                    _token_row(mask_rows[row], field_name=self.mask_column, row=row)
                )
            )
            for row in range(batch_rows)
        )
        k3_total = 0.0
        supervised = 0
        for row, mask in enumerate(cleaned_masks):
            reference = _token_row(
                reference_rows[row],
                field_name=self.reference_logprob_column,
                row=row,
            )
            current = _token_row(current_outer[row], field_name="forward_fn output", row=row)
            lengths = (len(mask), len(reference), len(current))
            if len(set(lengths)) != 1:
                raise BatchRefusal(
                    f"row {row}: mask/reference/current lengths are "
                    f"{lengths}; all 3 token denominators must be equal "
                    f"before one token reference projection can be "
                    f"attributed"
                )
            for position, is_supervised in enumerate(mask):
                if not is_supervised:
                    continue
                reference_value = _token_scalar(
                    reference[position],
                    self.reference_logprob_column,
                    row,
                    position,
                )
                current_value = _token_scalar(current[position], "current_logprobs", row, position)
                log_reference_ratio = reference_value - current_value
                try:
                    k3 = math.expm1(log_reference_ratio) - log_reference_ratio
                except OverflowError as exc:
                    raise BatchRefusal(
                        f"row {row}, position {position}: k3 projection "
                        f"exp({log_reference_ratio!r}) is not "
                        f"representable; 1 of 1 reference-token readings "
                        f"failed"
                    ) from exc
                # expm1(d) rounds to at least d for every finite d, because
                # the true value exceeds d by d**2/2, so this difference is
                # never strictly negative on a correctly-rounded libm and the
                # floor never fires there. It is KEPT rather than deleted for
                # the libm whose expm1 carries more than half an ulp of error:
                # without it, that platform would REFUSE a correct on-policy
                # batch at the check below. A test could reach the line only
                # by monkeypatching math.expm1, and a test that can only fire
                # against a fake is measuring the fake.
                if -1.0e-15 < k3 < 0.0:  # pragma: no cover -- correct libm cannot reach it
                    k3 = 0.0
                if k3 < 0.0 or not math.isfinite(k3):
                    raise BatchRefusal(
                        f"row {row}, position {position}: k3 value "
                        f"{k3!r} is not a finite nonnegative measurement"
                    )
                k3_total += k3
                supervised += 1
        if supervised == 0:
            raise SupervisionRefusal(
                f"0 supervised tokens remain across {batch_rows} offered "
                f"rows; the KL penalty is unmeasured over them, and "
                f"returning 0.0 would report the policy exactly on the "
                f"reference over no evidence"
            )
        estimate = k3_total / supervised
        coefficient_value = _controller_coefficient(self.controller)
        contribution = coefficient_value * estimate
        components = (
            LossComponent(
                name=self.component_name,
                weight=coefficient_value,
                observed=True,
                contribution=contribution,
            ),
        )
        metrics = (MetricObservation(name=self.metric_name, value=estimate),)
        return LossOutput(loss=contribution, components=components, metrics=metrics)
