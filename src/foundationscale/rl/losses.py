"""Stage-2 loss implementations for the FoundationScale RL loop.

This module holds the ``LossFn`` implementations; ``interfaces.py`` holds the
contracts (``ExperienceBatch``, ``LossOutput``, ``LossFn``, the refusal types
and ``LossDeclaration``). ``SFTLoss`` MOVED here unchanged from
``interfaces.py`` -- the public import path ``foundationscale.rl.SFTLoss`` is
unchanged -- and the split happened because stage 2 adds a second
implementation (``DPOLoss``) and stage 3 adds more. A contracts module that
also grows one implementation per stage would force every host-side reader of
the protocol to import numerical code it never calls; keeping contracts and
implementations in separate modules is what lets torch-free tooling import the
protocols alone.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

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

__all__ = ("DPOLoss", "SFTLoss")


def _mask_value(raw: Any, row: int, position: int) -> float:
    # `isinstance(True, int)` is True in Python, so the bool branch comes
    # FIRST. Everything after it is admitted by VALUE, via float(), rather
    # than by an isinstance test against int/float: a mask arriving as a
    # foreign scalar (one framework's 0-d array element) is the ordinary
    # case at the loop seam, and a type test would refuse a well-formed
    # mask for carrying the wrong Python class. The value check is what
    # holds the contract -- entries outside {0, 1} are refused rather than
    # read as fractional weights, because admitting one would silently
    # change the supervised-token count this loss divides by.
    if isinstance(raw, bool):
        return 1.0 if raw else 0.0
    try:
        value = float(raw)
    except (TypeError, ValueError):
        pass
    else:
        if value in (0.0, 1.0):
            return value
    raise BatchRefusal(
        f"mask entry at row {row}, position {position} is {raw!r}; "
        f"supervision mask entries must be 0 or 1"
    )


def _as_float(raw: Any, row: int, position: int) -> float:
    try:
        return float(raw)
    except (TypeError, ValueError) as exc:
        raise BatchRefusal(
            f"log-probability at row {row}, position {position} does not "
            f"convert to a scalar float ({type(raw).__name__}); forward_fn "
            f"must return one scalar target log-probability per token"
        ) from exc


def _reference_score(raw: Any, column: str, row: int) -> float:
    # Reference scores arrive pre-summed from batch columns, so (unlike
    # per-token log-probs) a non-finite value here is also refused: an inf or
    # NaN in a frozen reference is a data failure, and letting it through
    # would poison the margin for exactly one pair while the run reads as
    # healthy everywhere else.
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise BatchRefusal(
            f"reference column {column!r} row {row} is {raw!r}, which does "
            f"not convert to a scalar float; reference scores must be one "
            f"finite sequence-level log-probability sum per batch row"
        ) from exc
    if not math.isfinite(value):
        raise BatchRefusal(
            f"reference column {column!r} row {row} is {raw!r}, which is not "
            f"finite; reference scores must be one finite sequence-level "
            f"log-probability sum per batch row"
        )
    return value


@dataclass(frozen=True)
class SFTLoss:
    """Supervised fine-tuning loss: masked mean of negative target log-probs.

    ``forward_fn`` is invoked once with the batch and must return per-token
    TARGET log-probabilities: one row per batch row, one scalar per token
    position, as any iterable of iterables whose leaves support ``float()``
    (a plain nested list satisfies this; so does a row-iterable of scalar
    leaves). Raw logits are NOT accepted: who owns the log-softmax and the
    label gather is an open question (Phase 1 unknown 1, design section
    3.2), and this stage takes the narrowest option -- the caller
    normalises -- rather than inventing a mechanism here.

    The mask column must contain only 0 or 1 entries. A batch whose mask
    selects zero supervised tokens is REFUSED with the batch's row count
    and the measured supervised-token count in the message; the loss is
    unmeasurable on that batch.
    """

    mask_column: str = "loss_mask"
    component_name: str = "sft_loss"
    weight: float = 1.0

    def __call__(self, forward_fn: ForwardFn, batch: ExperienceBatch) -> LossOutput:
        if self.mask_column not in batch.columns:
            raise BatchRefusal(
                f"SFTLoss requires mask column {self.mask_column!r}; this "
                f"batch carries {tuple(batch.columns)}"
            )
        mask = batch.column(self.mask_column)
        log_probs = forward_fn(batch)
        rows = [tuple(row) for row in log_probs]
        if len(rows) != len(batch):
            raise BatchRefusal(
                f"forward_fn returned {len(rows)} per-token rows for a batch "
                f"of {len(batch)} rows; one row of log-probabilities per "
                f"batch row is required"
            )
        total = 0.0
        supervised = 0
        for row_index, (lp_row, mask_row) in enumerate(zip(rows, mask, strict=True)):
            mask_entries = tuple(mask_row)
            if len(lp_row) != len(mask_entries):
                raise BatchRefusal(
                    f"row {row_index}: forward_fn returned {len(lp_row)} "
                    f"per-token log-probabilities but the mask has "
                    f"{len(mask_entries)} entries"
                )
            for position, (lp, raw_mask) in enumerate(zip(lp_row, mask_entries, strict=True)):
                if _mask_value(raw_mask, row_index, position):
                    supervised += 1
                    total += _as_float(lp, row_index, position)
        if supervised == 0:
            raise SupervisionRefusal(
                f"supervision mask {self.mask_column!r} selected 0 supervised "
                f"tokens across {len(batch)} batch rows; the loss is "
                f"unmeasurable on this batch, and returning 0.0 would report "
                f"a perfect loss for a batch that taught the model nothing"
            )
        loss = -self.weight * (total / supervised)
        component = LossComponent(
            name=self.component_name,
            weight=self.weight,
            observed=True,
            contribution=loss,
        )
        return LossOutput(loss=loss, components=(component,))

    def declaration(self) -> LossDeclaration:
        # SFT declares one component and no diagnostic metric. The empty metric
        # tuple is the honest state, not an oversight: DiagnosticMetricGate
        # answers "nothing declared, nothing observed" with a DECLARED SKIP, so
        # the absence stays in the sweep's denominator instead of reading as
        # coverage.
        return LossDeclaration(components=(self.component_name,))


@dataclass(frozen=True)
class DPOLoss:
    """Direct preference optimisation over paired completions, one pair per batch row.

    Per batch row = ONE preference pair. With per-row policy sequence scores
    ``pi_chosen``/``pi_rejected`` and reference scores ``ref_chosen``/
    ``ref_rejected`` (all sequence-level sums of masked target
    log-probabilities)::

        margin    = beta * ((pi_chosen - ref_chosen) - (pi_rejected - ref_rejected))
        pair_loss = -log sigmoid(margin)
        loss      = weight * mean(pair_loss)

    ``forward_fn`` is invoked ONCE with the batch and must return one row per
    batch row, each row a 2-element sequence ``(chosen_token_logprobs,
    rejected_token_logprobs)`` of per-token TARGET log-probabilities. Raw
    logits are NOT accepted, for the same reason ``SFTLoss`` gives: who owns
    the log-softmax and the label gather is an open question, and the caller
    normalises.

    The REFERENCE log-probabilities are read from batch COLUMNS as
    sequence-level sums, one scalar per row. Who runs the frozen reference
    over the batch to produce them is the OPEN EDGE recorded in design
    section 3.4/4 -- this loss consumes the column and deliberately does not
    invent a producer.

    Condition (a) of design section 9 stage 2 is discharged by construction:
    the auxiliary SFT term is declared if and only if ``sft_weight != 0.0``,
    and BOTH the declaration and the computation read that one field, so the
    two cannot diverge. The literal form of the condition -- declare the sft
    component unconditionally -- cannot be implemented, for a measured
    reason: ``LossComponentCoverageGate`` FAILS a declared component whose
    ``weight == 0.0``, so declaring an inactive ``sft_loss`` would refuse
    every sft-off DPO run at step zero. What the condition actually requires
    -- that an inactive term and a broken term cannot read identically -- is
    met by there being no computed-but-undeclared state at all.

    Condition (c): ``accuracy`` is declared with ``low=0.0, high=1.0,
    degenerate=(0.0,)``, because ``0.0`` is inside an accuracy's natural
    range and bounds alone cannot refuse it. Accuracy pinned at ``0.0`` means
    the policy ranked the REJECTED completion above the chosen one on every
    pair -- the Phase-2 anomaly this metric exists to catch.
    """

    beta: float = 0.1
    weight: float = 1.0
    sft_weight: float = 0.0
    chosen_mask_column: str = "chosen_loss_mask"
    rejected_mask_column: str = "rejected_loss_mask"
    reference_chosen_column: str = "reference_chosen_logprob"
    reference_rejected_column: str = "reference_rejected_logprob"
    component_name: str = "preference_loss"
    sft_component_name: str = "sft_loss"
    metric_name: str = "accuracy"

    def __post_init__(self) -> None:
        # Fail at CONSTRUCTION, not at step zero: every refusal below names a
        # configuration the objective gates would later refuse anyway, and
        # refusing here names the field that produced it rather than the gate
        # that happened to see it first.
        if (
            not isinstance(self.beta, (int, float))
            or not math.isfinite(self.beta)
            or self.beta <= 0.0
        ):
            raise LossConfigRefusal(
                f"beta={self.beta!r}: a non-positive or non-finite beta "
                f"inverts or annihilates the preference signal"
            )
        if (
            not isinstance(self.weight, (int, float))
            or not math.isfinite(self.weight)
            or self.weight == 0.0
        ):
            raise LossConfigRefusal(
                f"weight={self.weight!r}: a zero-weight or non-finite "
                f"declared component is a blocking FAIL at "
                f"LossComponentCoverageGate, so the loss refuses to be "
                f"built that way"
            )
        if (
            not isinstance(self.sft_weight, (int, float))
            or not math.isfinite(self.sft_weight)
            or self.sft_weight < 0.0
        ):
            raise LossConfigRefusal(
                f"sft_weight={self.sft_weight!r}: the auxiliary SFT weight "
                f"must be finite and non-negative"
            )
        name_fields = (
            ("chosen_mask_column", self.chosen_mask_column),
            ("rejected_mask_column", self.rejected_mask_column),
            ("reference_chosen_column", self.reference_chosen_column),
            ("reference_rejected_column", self.reference_rejected_column),
            ("component_name", self.component_name),
            ("sft_component_name", self.sft_component_name),
            ("metric_name", self.metric_name),
        )
        for field_name, value in name_fields:
            if not isinstance(value, str) or not value:
                raise LossConfigRefusal(
                    f"{field_name}={value!r}: column, component and metric "
                    f"names must be non-empty strings; absence of a name is "
                    f"not a name"
                )
        if self.component_name == self.sft_component_name:
            raise LossConfigRefusal(
                f"component_name and sft_component_name are both "
                f"{self.component_name!r}: two components sharing one name "
                f"collapse the coverage denominator"
            )

    @property
    def required_columns(self) -> tuple[str, ...]:
        """The four batch columns this loss consumes, in a stable order.

        Callers use this to build the ``ExperienceBatch(required=...)``
        schema, so the order must not depend on construction site or dict
        iteration: a schema that wobbles between runs would make two
        identical batches read as different shapes.
        """
        return (
            self.chosen_mask_column,
            self.rejected_mask_column,
            self.reference_chosen_column,
            self.reference_rejected_column,
        )

    @property
    def reference_free(self) -> bool:
        """Whether the objective consumes a frozen reference model's scores.

        DPO is reference-ANCHORED -- it reads two reference columns -- so
        this is False. The property is the SOURCE that ``semantics()``
        derives from, so the two statements about this seam cannot drift
        apart.
        """
        return False

    def semantics(self) -> AlgorithmSemantics:
        """The loss side of the two-declaration seam check.

        Exactly one of the five cross-checked fields -- ``reference_free``
        -- is constrained by a preference objective, and it is derived from
        the ``reference_free`` property rather than restated. The other four
        abstain with ``None``, which the record defines as "this does not
        constrain that seam": DPO forms no importance ratio, so it has no
        ratio geometry, no clip bounds and no KL estimator to state, and it
        reads no group, so it has no group size. Declaring a number there
        would be a measured value posing as an unconstrained seam.

        What is NOT claimed: that reference-freeness is CHECKED against a
        wired reference model. This record states what the loss consumes;
        whether a reference policy is actually present is the role-presence
        layer's question, answered by ``AlgorithmRequirements``.
        """
        return AlgorithmSemantics(reference_free=self.reference_free)

    def declaration(self) -> LossDeclaration:
        """The manifest's statement of what this loss is configured to emit.

        This is for the RUN MANIFEST, not for the gate context. The gate
        compares what the run RECORDED against what the live step OBSERVED;
        feeding ``declaration()`` straight into the gate context would
        compare what is in force against what is in force -- vacuous, and
        exactly the reading the gate exists to refuse.

        The sft entry is derived from the same field the computation reads
        (``self.sft_weight != 0.0``), so declaration and observation cannot
        diverge: there is no computed-but-undeclared state for the gate to
        be blind to.
        """
        components: tuple[str, ...] = (self.component_name,)
        if self.sft_weight != 0.0:
            components = components + (self.sft_component_name,)
        # 0.0 is inside an accuracy's natural range, so bounds alone cannot
        # refuse it; the degenerate tuple is what keeps a fully-inverted run
        # in the denominator.
        metric = MetricExpectation(
            name=self.metric_name,
            low=0.0,
            high=1.0,
            degenerate=(0.0,),
        )
        return LossDeclaration(components=components, metrics=(metric,))

    def __call__(self, forward_fn: ForwardFn, batch: ExperienceBatch) -> LossOutput:
        missing = tuple(name for name in self.required_columns if name not in batch.columns)
        if missing:
            raise BatchRefusal(
                f"DPOLoss requires columns {missing!r}, which are absent; "
                f"this batch carries {tuple(batch.columns)}"
            )
        if len(batch) == 0:
            raise BatchRefusal(
                "DPOLoss got a batch of 0 rows: a mean over zero pairs is "
                "the all([]) shape -- the loss is unmeasurable, and "
                "returning 0.0 would report a perfect preference loss for "
                "a batch containing no preferences"
            )
        chosen_masks = batch.column(self.chosen_mask_column)
        rejected_masks = batch.column(self.rejected_mask_column)
        reference_chosen = batch.column(self.reference_chosen_column)
        reference_rejected = batch.column(self.reference_rejected_column)

        rows = list(forward_fn(batch))
        if len(rows) != len(batch):
            raise BatchRefusal(
                f"forward_fn returned {len(rows)} pair rows for a batch of "
                f"{len(batch)} rows; one (chosen, rejected) row per batch "
                f"row is required"
            )

        pair_losses: list[float] = []
        margins: list[float] = []
        chosen_sum_total = 0.0
        chosen_supervised_total = 0
        for row_index, row in enumerate(rows):
            try:
                row_length = len(row)
            except TypeError as exc:
                raise BatchRefusal(
                    f"row {row_index} of forward_fn's return is not a sized "
                    f"sequence ({type(row).__name__}); each row must be a "
                    f"2-element sequence whose two elements are the chosen "
                    f"and rejected per-token log-probability sequences"
                ) from exc
            if row_length != 2:
                raise BatchRefusal(
                    f"row {row_index} of forward_fn's return has {row_length} "
                    f"elements, expected 2; the two elements are the chosen "
                    f"and rejected per-token log-probability sequences"
                )
            side_rows = (
                ("chosen", tuple(row[0]), chosen_masks[row_index]),
                ("rejected", tuple(row[1]), rejected_masks[row_index]),
            )
            side_scores: dict[str, float] = {}
            for side, token_lps, mask_row in side_rows:
                mask_entries = tuple(mask_row)
                if len(token_lps) != len(mask_entries):
                    raise BatchRefusal(
                        f"row {row_index}, side {side!r}: forward_fn "
                        f"returned {len(token_lps)} per-token "
                        f"log-probabilities but the mask has "
                        f"{len(mask_entries)} entries"
                    )
                # The side's policy score is the SUM (not the mean) of the
                # masked token log-probabilities: DPO compares sequence
                # likelihoods, and a mean would make the comparison
                # length-normalised, which is a different algorithm.
                score = 0.0
                supervised = 0
                for position, (lp, raw_mask) in enumerate(
                    zip(token_lps, mask_entries, strict=True)
                ):
                    if _mask_value(raw_mask, row_index, position):
                        supervised += 1
                        score += _as_float(lp, row_index, position)
                if supervised == 0:
                    raise SupervisionRefusal(
                        f"row {row_index}, side {side!r}: supervision mask "
                        f"selected 0 supervised tokens; a pair in which one "
                        f"completion supervises nothing is unmeasurable, and "
                        f"scoring it 0.0 would make an empty completion "
                        f"look maximally likely"
                    )
                side_scores[side] = score
                if side == "chosen":
                    chosen_sum_total += score
                    chosen_supervised_total += supervised

            pi_c = side_scores["chosen"]
            pi_r = side_scores["rejected"]
            ref_c = _reference_score(
                reference_chosen[row_index], self.reference_chosen_column, row_index
            )
            ref_r = _reference_score(
                reference_rejected[row_index], self.reference_rejected_column, row_index
            )

            margin = self.beta * ((pi_c - ref_c) - (pi_r - ref_r))
            # Numerically stable -log sigmoid(margin) as softplus(-margin).
            # The naive -math.log(1 / (1 + math.exp(-margin))) overflows
            # math.exp for large negative margins -- exactly the regime a
            # diverging run enters -- so both branches route through log1p.
            if margin >= 0.0:
                pair_loss = math.log1p(math.exp(-margin))
            else:
                pair_loss = -margin + math.log1p(math.exp(margin))
            margins.append(margin)
            pair_losses.append(pair_loss)

        preference = self.weight * (sum(pair_losses) / len(pair_losses))
        # margin == 0.0 counts as NOT correct: a tie is not a correct
        # ranking, and counting it as one would lift a fully-tied run off the
        # declared degenerate 0.0.
        correct = sum(1 for margin in margins if margin > 0.0)
        accuracy = correct / len(margins)

        sft_term = 0.0
        sft_component: LossComponent | None = None
        if self.sft_weight != 0.0:
            # Reuse the chosen-side masked sums and supervised counts already
            # computed above: a second forward_fn call would be both wasteful
            # and a second chance for the two readings to disagree.
            sft_term = self.sft_weight * (-(chosen_sum_total / chosen_supervised_total))
            sft_component = LossComponent(
                name=self.sft_component_name,
                weight=self.sft_weight,
                observed=True,
                contribution=sft_term,
            )

        loss = preference + sft_term
        components: list[LossComponent] = [
            LossComponent(
                name=self.component_name,
                weight=self.weight,
                observed=True,
                contribution=preference,
            )
        ]
        if sft_component is not None:
            components.append(sft_component)
        metrics = (MetricObservation(name=self.metric_name, value=accuracy),)
        return LossOutput(loss=loss, components=tuple(components), metrics=metrics)
