"""Stage-3 reward-model training path for the FoundationScale RL loop.

This module holds the REWARD-MODEL TRAINING side of the preference-data seam:
``RewardModelLoss``, a ``LossFn`` implementation that trains a scalar reward
head on preference pairs under the Bradley-Terry objective, and
``check_reward_model_requirements``, the wiring gate that refuses a batch
whose column schema does not carry what the loss declared it needs. The
reward-model INFERENCE path (scoring fresh rollouts inside the RL loop) is
deliberately not here: training consumes logged preference pairs with a
chosen/rejected label structure, while inference consumes unpaired rollouts,
and folding both into one module would let a training-side column schema
silently gate an inference-side caller.

WHAT IS CLAIMED: given a ``forward_fn`` that scores each member of each
preference pair, this loss computes the Bradley-Terry objective
``-log sigmoid(r_chosen - r_rejected)`` exactly, declares honestly what it
consumes (two named score columns via ``required_columns``), abstains with
``None`` on every semantics axis it does not constrain, and refuses --
never guesses -- when the column schema is half-declared.

WHAT IS NOT CLAIMED: that the reward head's architecture, normalisation, or
calibration is specified here (who owns the head is the host's concern and
this module never reaches into the model); that the trained reward is a
faithful measure of human preference (Bradley-Terry optimises pairwise
consistency, not truth); or that this objective forms an importance ratio,
reads a sampling group, or constrains any other algorithm-semantics axis.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from foundationscale.gates.objective_gates import (
    LossComponent,
    MetricExpectation,
    MetricObservation,
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

__all__ = ("RewardModelLoss", "check_reward_model_requirements")


def _pair_score(raw: object, member: str, row: int) -> float:
    # A non-finite reward is refused rather than clamped: an inf or NaN from
    # the head in exactly one pair would saturate that pair's sigmoid and let
    # the run read as healthy everywhere else while one corrupted gradient
    # dominates the mean. Rewards are head OUTPUTS, not logged references, so
    # no finiteness guarantee comes in from upstream; the check happens here.
    try:
        value = float(raw)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise BatchRefusal(
            f"{member} reward at row {row} is {raw!r}, which does not "
            f"convert to a scalar float; forward_fn must return one finite "
            f"scalar reward per member of each preference pair"
        ) from exc
    if not math.isfinite(value):
        raise BatchRefusal(
            f"{member} reward at row {row} is {raw!r}, which is not finite; "
            f"forward_fn must return one finite scalar reward per member of "
            f"each preference pair"
        )
    return value


@dataclass(frozen=True, slots=True)
class RewardModelLoss:
    """Bradley-Terry preference loss over paired rewards, one pair per batch row.

    Per batch row = ONE preference pair. ``forward_fn`` is invoked ONCE with
    the batch and must return one row per batch row, each row a 2-element
    sequence ``(chosen_reward, rejected_reward)`` of finite scalars -- the
    reward head's score for each member of the pair::

        pair_loss = -log sigmoid(chosen_reward - rejected_reward)
        loss      = weight * mean(pair_loss)

    Unlike ``DPOLoss`` this loss reads REFERENCE-FREE head outputs: there is
    no frozen-reference anchor and no beta, because the reward model is the
    object being trained rather than a policy being kept near a reference.

    ``required_columns`` names the chosen/rejected SCORE columns a host
    typically logs beside the pair (pre-computed scores from a previous
    scoring pass). The loss does not read them -- the live scores come from
    ``forward_fn`` -- but it declares them so the wiring gate can refuse a
    half-declared schema at batch construction instead of discovering the
    gap after scoring.

    ``accuracy`` (fraction of pairs with ``r_chosen > r_rejected``) is
    declared with ``low=0.0, high=1.0, degenerate=(0.0,)``: a reward model
    that prefers the rejected member on EVERY pair has inverted the
    preference signal, and bounds alone cannot refuse 0.0 because it sits
    inside the natural range.
    """

    weight: float = 1.0
    chosen_score_column: str = "chosen_reward"
    rejected_score_column: str = "rejected_reward"
    component_name: str = "reward_model_loss"
    metric_name: str = "accuracy"

    def __post_init__(self) -> None:
        # Refuse at CONSTRUCTION the configurations the objective gates would
        # refuse at step zero, naming the field rather than the gate that
        # happened to see it first.
        if (
            not isinstance(self.weight, (int, float))
            or isinstance(self.weight, bool)
            or not math.isfinite(self.weight)
            or self.weight == 0.0
        ):
            raise LossConfigRefusal(
                f"weight={self.weight!r}: a zero-weight or non-finite "
                f"declared component is a blocking FAIL at "
                f"LossComponentCoverageGate, so the loss refuses to be "
                f"built that way"
            )
        name_fields = (
            ("chosen_score_column", self.chosen_score_column),
            ("rejected_score_column", self.rejected_score_column),
            ("component_name", self.component_name),
            ("metric_name", self.metric_name),
        )
        for field_name, value in name_fields:
            if not isinstance(value, str) or not value:
                raise LossConfigRefusal(
                    f"{field_name}={value!r}: column, component and metric "
                    f"names must be non-empty strings; absence of a name is "
                    f"not a name"
                )
        if self.chosen_score_column == self.rejected_score_column:
            raise LossConfigRefusal(
                f"chosen_score_column and rejected_score_column are both "
                f"{self.chosen_score_column!r}: one column cannot hold both "
                f"members of a preference pair, and a schema that claims so "
                f"would score the same value twice and call it a comparison"
            )

    @property
    def required_columns(self) -> tuple[str, ...]:
        """The two score columns this loss declares, in a stable order.

        The order must not depend on construction site or dict iteration:
        callers build ``ExperienceBatch(required=...)`` from this, and a
        schema that wobbles between runs would make two identical batches
        read as different shapes.
        """
        return (self.chosen_score_column, self.rejected_score_column)

    @property
    def reference_free(self) -> bool:
        """Whether the objective consumes a frozen reference model's scores.

        Reward-model training reads no reference columns -- the head's own
        outputs are the objective -- so this is True, and ``semantics()``
        derives from this one field so the two statements cannot drift.
        """
        return True

    def semantics(self) -> AlgorithmSemantics:
        """The loss side of the two-declaration seam check.

        Only ``reference_free`` is constrained. Every other axis --
        importance-ratio formation, group reads, clipping -- is abstained as
        ``None``: a Bradley-Terry trainer forms no ratio and reads no group,
        and stating ``False`` would CLAIM a negative this loss has no
        standing to make about whatever algorithm hosts it.
        """
        # A reward model speaks exactly ONE of the five declared axes. It scores
        # sequences; it forms no importance ratio, reads no group and clips
        # nothing, so those axes are None -- ABSTENTION, not a claim of "no".
        # The three fields this originally passed (forms_importance_ratio,
        # reads_group, clips_ratio) do not exist on AlgorithmSemantics, so
        # semantics() raised TypeError on every call.
        return AlgorithmSemantics(
            group_size=None,
            ratio_scope=None,
            kl_estimator=None,
            clip_bounds=None,
            reference_free=self.reference_free,
        )

    def __call__(self, forward_fn: ForwardFn, batch: ExperienceBatch) -> LossOutput:
        scores = forward_fn(batch)
        rows = [tuple(row) for row in scores]
        if len(rows) != len(batch):
            raise BatchRefusal(
                f"forward_fn returned {len(rows)} score rows for a batch of "
                f"{len(batch)} rows; one (chosen, rejected) pair per batch "
                f"row is required"
            )
        total = 0.0
        correct = 0.0
        observed = 0
        for row_index, pair in enumerate(rows):
            if len(pair) != 2:
                raise BatchRefusal(
                    f"row {row_index}: forward_fn returned {len(pair)} "
                    f"scores; each batch row is ONE preference pair and "
                    f"must produce exactly 2, (chosen, rejected)"
                )
            chosen = _pair_score(pair[0], "chosen", row_index)
            rejected = _pair_score(pair[1], "rejected", row_index)
            # -log sigmoid(margin) in the numerically stable form
            # softplus(-margin): naive log(sigmoid()) underflows to inf for
            # large negative margins, reporting an untrainable loss for what
            # is merely a very wrong pair.
            margin = chosen - rejected
            total += math.log1p(math.exp(-abs(margin))) + max(-margin, 0.0)
            correct += 1.0 if chosen > rejected else 0.0
            observed += 1
        if observed == 0:
            raise SupervisionRefusal(
                f"0 preference pairs across {len(batch)} batch rows; the "
                f"reward-model loss is unmeasurable on this batch, and "
                f"returning 0.0 would report a perfect loss for a batch "
                f"that taught the model nothing"
            )
        loss = self.weight * (total / observed)
        component = LossComponent(
            name=self.component_name,
            weight=self.weight,
            observed=True,
            contribution=loss,
        )
        metric = MetricObservation(name=self.metric_name, value=correct / observed)
        return LossOutput(loss=loss, components=(component,), metrics=(metric,))

    def declaration(self) -> LossDeclaration:
        return LossDeclaration(
            components=(self.component_name,),
            metrics=(
                MetricExpectation(
                    name=self.metric_name,
                    low=0.0,
                    high=1.0,
                    degenerate=(0.0,),
                ),
            ),
        )


def check_reward_model_requirements(
    loss: RewardModelLoss,
    present_columns: tuple[str, ...],
) -> None:
    """Refuse a half-declared score-column schema before any scoring happens.

    ``present_columns`` is what the host batch actually carries. The gate
    compares it against the loss's full ``required_columns`` and REFUSES on
    any gap, naming both sides of the count -- it does not guess which
    column the host "meant", because a mis-bound score column inverts the
    preference signal silently. An empty required set is refused as vacuous:
    a reward-model loss that declares no score columns would pass this gate
    over every conceivable batch, which is exactly the false green the gate
    exists to prevent.
    """
    required = loss.required_columns
    if not required:
        raise LossConfigRefusal(
            f"{type(loss).__name__} declares 0 required score columns; an "
            f"empty required set is vacuous -- it would pass this gate over "
            f"every batch and grade a wiring gap as coverage"
        )
    missing = tuple(name for name in required if name not in present_columns)
    if missing:
        raise BatchRefusal(
            f"{len(missing)} of {len(required)} required score column(s) "
            f"absent: {missing}; this batch carries {tuple(present_columns)}. "
            f"A half-declared preference schema is refused, not guessed at: "
            f"binding one score column to the wrong pair member inverts the "
            f"preference signal while every dashboard reads green"
        )
