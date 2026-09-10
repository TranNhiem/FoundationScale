"""Online objectives: OnlineDPO, IterativeDPO, RAFT and Best-of-N SFT.

This module carries four objectives that differ from the preference family
in ``preference_objectives.py`` in ONE respect: the supervision is drawn
from the CURRENT policy's own samples, not from a fixed preference corpus.
The idiom is otherwise borrowed wholesale from DPOLoss: per-token TARGET
log-probabilities returned by ``forward_fn``, supervision masks read from
batch columns, sequence-level reference scores read as pre-summed batch
columns, refusals that name the offending row, field or count rather than
clamping or skipping, and abstention with ``None`` on every seam the
family does not constrain.

The four split along two axes. ``OnlineDPOLoss`` and ``IterativeDPOLoss``
are preference losses over pairs sampled from the current policy: the
former reads its reference scores from batch columns refreshed however
the producer chooses, the latter explicitly models the periodically
refreshed reference SNAPSHOT and refuses a batch whose declared snapshot
identity is not uniform, because a pair priced against two different
snapshots is not one measurement. ``RAFTLoss`` and ``BestOfNLoss`` are
rejection-sampling fine-tuning objectives: RAFT consumes an already
filtered best-of-N batch and prices SFT on it, while ``BestOfNLoss``
receives whole groups plus per-row rewards and itself selects the
argmax-reward member of each group before pricing SFT on the winners.

WHAT THIS MODULE CLAIMS: each loss prices exactly the rows (and, for
``BestOfNLoss``, exactly the selected group winners) of the supplied
batch, declares exactly the components and metrics it emits, derives its
``reference_free`` statement from the property of the same name, and
refuses degenerate, mis-shaped, mixed-snapshot or non-finite input
rather than reporting a number.

WHAT THIS MODULE DOES NOT CLAIM: that any sampler, any reward model, any
snapshot store, or any producer of the consumed batch columns exists
here. Pair construction, reward ranking, filtering and snapshot refresh
all happen upstream of the loss; this is a torch-free module measuring
pure objectives over supplied per-token readings, enrolled in the
torch-free gate.
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

__all__ = (
    "BestOfNLoss",
    "IterativeDPOLoss",
    "OnlineDPOLoss",
    "RAFTLoss",
)


def _mask_value(raw: Any, row: int, position: int) -> float:
    # Copied from losses.py verbatim (private names are not imported across
    # the module boundary). The bool branch comes FIRST because
    # isinstance(True, int) is True; everything after is admitted by VALUE
    # via float(), and entries outside {0, 1} are refused rather than read
    # as fractional weights, because admitting one would silently change
    # the supervised-token count the losses divide by.
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
    # The finiteness refusal ppo_objectives.py established: a non-finite
    # token reading is named with row and position rather than clamped or
    # skipped, because a silently clamped row would change the objective
    # the run reports and a silently dropped row would change the mean's
    # denominator.
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise BatchRefusal(
            f"log-probability at row {row}, position {position} does not "
            f"convert to a scalar float ({type(raw).__name__}); forward_fn "
            f"must return one scalar target log-probability per token"
        ) from exc
    if not math.isfinite(value):
        raise BatchRefusal(
            f"log-probability at row {row}, position {position} is "
            f"{raw!r}, which is not finite; a non-finite token reading "
            f"would poison the objective this loss reports, and clamping "
            f"it would silently rewrite the row the model actually produced"
        )
    return value


def _reference_score(raw: Any, column: str, row: int) -> float:
    # Verbatim from losses.py: reference scores arrive pre-summed, so a
    # non-finite value is a data failure and is refused rather than let
    # through to poison exactly one margin while the run reads healthy.
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


def _reward_value(raw: Any, column: str, row: int) -> float:
    # Rewards are ranked, not merely summed, so a non-finite entry is worse
    # than useless: float('nan') breaks every comparison and float('inf')
    # wins every group it touches. Both are refused with the row named.
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise BatchRefusal(
            f"reward column {column!r} row {row} is {raw!r}, which does "
            f"not convert to a scalar float; rewards must be one finite "
            f"scalar per batch row"
        ) from exc
    if not math.isfinite(value):
        raise BatchRefusal(
            f"reward column {column!r} row {row} is {raw!r}, which is not "
            f"finite; a non-finite reward cannot be ranked, and ranking it "
            f"anyway would silently pick one completion as 'best'"
        )
    return value


def _neg_log_sigmoid(value: float) -> float:
    # Numerically stable -log sigmoid(value) as softplus(-value), verbatim
    # the branch pair DPOLoss writes inline: the naive form overflows
    # math.exp for large negative values -- exactly the regime a diverging
    # run enters -- so both branches route through log1p.
    if value >= 0.0:
        return math.log1p(math.exp(-value))
    return -value + math.log1p(math.exp(value))


def _masked_side_score(
    token_lps: tuple[Any, ...],
    mask_entries: tuple[Any, ...],
    row_index: int,
    side: str,
) -> tuple[float, int]:
    """Return the masked log-probability SUM and supervised count for one side.

    The reduction is a SUM, matching DPOLoss: whether (and by what) the
    caller normalises is the caller's stated choice -- the DPO variants
    compare raw sequence likelihoods, the SFT variants divide by the
    supervised-token count -- and baking the mean in here would silently
    impose one of those choices on the other.
    """
    if len(token_lps) != len(mask_entries):
        raise BatchRefusal(
            f"row {row_index}, side {side!r}: forward_fn returned "
            f"{len(token_lps)} per-token log-probabilities but the mask "
            f"has {len(mask_entries)} entries"
        )
    score = 0.0
    supervised = 0
    for position, (lp, raw_mask) in enumerate(zip(token_lps, mask_entries, strict=True)):
        if _mask_value(raw_mask, row_index, position):
            supervised += 1
            score += _as_float(lp, row_index, position)
    return score, supervised


def _paired_forward_rows(
    forward_fn: ForwardFn,
    batch: ExperienceBatch,
    family: str,
) -> tuple[tuple[tuple[Any, ...], tuple[Any, ...]], ...]:
    """Validate the paired (chosen, rejected) forward_fn return shape.

    This is DPOLoss's contract exactly: one outer entry per batch row,
    each a 2-element sequence whose two elements are the chosen and
    rejected per-token log-probability sequences, with the same refusal
    messages DPO emits for the two mis-shape cases.
    """
    rows = list(forward_fn(batch))
    if len(rows) != len(batch):
        raise BatchRefusal(
            f"forward_fn returned {len(rows)} pair rows for a batch of "
            f"{len(batch)} rows; {family} requires one (chosen, rejected) "
            f"row per batch row"
        )
    cleaned: list[tuple[tuple[Any, ...], tuple[Any, ...]]] = []
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
        cleaned.append((tuple(row[0]), tuple(row[1])))
    return tuple(cleaned)


def _per_row_forward(
    forward_fn: ForwardFn,
    batch: ExperienceBatch,
    family: str,
) -> tuple[tuple[Any, ...], ...]:
    """Validate the single-completion forward_fn return shape.

    The SFT variants consume ONE per-token log-probability sequence per
    batch row: the completion to be imitated. A count mismatch is refused
    rather than padded, because a row the model never read cannot be
    priced, and a silently dropped row would change the mean's
    denominator.
    """
    rows = list(forward_fn(batch))
    if len(rows) != len(batch):
        raise BatchRefusal(
            f"forward_fn returned {len(rows)} rows for a batch of "
            f"{len(batch)} rows; {family} requires one per-token "
            f"log-probability sequence per batch row"
        )
    return tuple(tuple(row) for row in rows)


def _check_name_fields(name_fields: tuple[tuple[str, Any], ...]) -> None:
    # DPOLoss's name-field check, factored out; the message is DPO's
    # verbatim.
    for field_name, value in name_fields:
        if not isinstance(value, str) or not value:
            raise LossConfigRefusal(
                f"{field_name}={value!r}: column, component and metric "
                f"names must be non-empty strings; absence of a name is "
                f"not a name"
            )


def _check_positive_finite(name: str, value: Any) -> None:
    # Shared beta/ceiling check: a non-finite bound makes every gate
    # comparison against it silently True, and a non-positive temperature
    # inverts or annihilates the preference signal.
    if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0.0:
        raise LossConfigRefusal(f"{name}={value!r}: the value must be finite and strictly positive")


def _check_weight(value: Any) -> None:
    # DPO's weight check verbatim: a zero-weight declared component is a
    # blocking FAIL at LossComponentCoverageGate, so the loss refuses to
    # be built that way.
    if not isinstance(value, (int, float)) or not math.isfinite(value) or value == 0.0:
        raise LossConfigRefusal(
            f"weight={value!r}: a zero-weight or non-finite declared "
            f"component is a blocking FAIL at LossComponentCoverageGate, "
            f"so the loss refuses to be built that way"
        )


@dataclass(frozen=True, slots=True)
class OnlineDPOLoss:
    """Direct preference optimisation over pairs sampled from the CURRENT policy.

    Per batch row = ONE preference pair whose chosen/rejected sides are
    completions the CURRENT policy produced this step, ranked upstream by
    a reward signal this module neither sees nor assumes. Pricing is
    DPO's exactly: ``forward_fn`` is invoked ONCE and must return one row
    per batch row, each a 2-element sequence ``(chosen_token_logprobs,
    rejected_token_logprobs)`` of per-token TARGET log-probabilities, and
    the reference scores arrive as batch columns of sequence-level sums::

        h         = (pi_chosen - ref_chosen) - (pi_rejected - ref_rejected)
        pair_loss = -log sigmoid(beta * h)
        loss      = weight * mean(pair_loss)

    "Online" here concerns WHERE THE PAIRS COME FROM, not the objective:
    the loss is reference-anchored like DPO, and it expects the producer
    to keep the reference columns and the sampled pairs mutually
    consistent. Because pairs are freshly sampled every step, the margin
    distribution drifts with the policy; that drift is a property of the
    data stream, not something this objective smooths over or corrects.

    WHAT IS CLAIMED: the loss is the log-sigmoid form above over exactly
    the supplied pairs, with ``h`` built from masked SUMS of token
    log-probabilities; preference accuracy (fraction of rows with
    ``h > 0``) and the mean margin are reported through the metrics
    channel; a zero-supervision side, a missing column or a non-finite
    reading is refused rather than measured.

    WHAT IS NOT CLAIMED: that the pairs were sampled from the current
    policy -- only the producer knows that, and this module measures the
    batch it is handed -- or that the reference scores track any
    particular snapshot cadence; that is ``IterativeDPOLoss``'s stated
    concern.
    """

    beta: float = 0.1
    weight: float = 1.0
    chosen_mask_column: str = "chosen_loss_mask"
    rejected_mask_column: str = "rejected_loss_mask"
    reference_chosen_column: str = "reference_chosen_logprob"
    reference_rejected_column: str = "reference_rejected_logprob"
    component_name: str = "online_dpo_loss"
    accuracy_metric_name: str = "accuracy"
    margin_metric_name: str = "online_dpo_margin_mean"
    margin_metric_ceiling: float = 100.0

    def __post_init__(self) -> None:
        _check_positive_finite("beta", self.beta)
        _check_weight(self.weight)
        _check_name_fields(
            (
                ("chosen_mask_column", self.chosen_mask_column),
                ("rejected_mask_column", self.rejected_mask_column),
                ("reference_chosen_column", self.reference_chosen_column),
                ("reference_rejected_column", self.reference_rejected_column),
                ("component_name", self.component_name),
                ("accuracy_metric_name", self.accuracy_metric_name),
                ("margin_metric_name", self.margin_metric_name),
            )
        )
        _check_positive_finite("margin_metric_ceiling", self.margin_metric_ceiling)

    @property
    def reference_free(self) -> bool:
        """Whether the objective consumes a frozen reference model's scores.

        WHAT IS CLAIMED: OnlineDPO is reference-ANCHORED, so this is
        False. The property is the SOURCE that ``semantics()`` derives
        from, so the two statements about this seam cannot drift apart.

        WHAT IS NOT CLAIMED: any gate behaviour attached to this value.
        """
        return False

    @property
    def required_columns(self) -> tuple[str, ...]:
        """The four batch columns this loss consumes, in a stable order.

        WHAT IS CLAIMED: the tuple is byte-identical in shape to
        ``DPOLoss.required_columns`` -- two masks and two reference-score
        columns -- so callers can build the ``ExperienceBatch`` schema
        the same way.

        WHAT IS NOT CLAIMED: that the columns carry well-typed values;
        shapes and values are measured by ``__call__``.
        """
        return (
            self.chosen_mask_column,
            self.rejected_mask_column,
            self.reference_chosen_column,
            self.reference_rejected_column,
        )

    def semantics(self) -> AlgorithmSemantics:
        """The loss side of the two-declaration seam check.

        WHAT IS CLAIMED: exactly one field is constrained --
        ``reference_free``, derived from the property rather than
        restated. The other fields abstain with ``None``: a preference
        objective forms no importance ratio, has no clip bounds, reads no
        group and declares no KL estimator, and stating a number there
        would be a measured value posing as an unconstrained seam.

        WHAT IS NOT CLAIMED: that reference-freeness is CHECKED against a
        wired reference model; that is the role-presence layer's
        question.
        """
        return AlgorithmSemantics(reference_free=self.reference_free)

    def declaration(self) -> LossDeclaration:
        """The manifest's statement of what this loss is configured to emit.

        WHAT IS CLAIMED: one component plus two metrics, in DPO's idiom:
        accuracy ``low=0.0, high=1.0, degenerate=(0.0,)``, and the margin
        mean bounded symmetrically by ``margin_metric_ceiling`` with an
        EMPTY degenerate set -- a zero mean margin is a real reading of a
        run at the decision boundary, not a broken metric.

        WHAT IS NOT CLAIMED: that this declaration is suitable input to
        ``build_objective_gate_context``; it is the run-manifest
        statement, per the ``LossDeclaration`` contract.
        """
        accuracy = MetricExpectation(
            name=self.accuracy_metric_name,
            low=0.0,
            high=1.0,
            degenerate=(0.0,),
        )
        margin = MetricExpectation(
            name=self.margin_metric_name,
            low=-float(self.margin_metric_ceiling),
            high=float(self.margin_metric_ceiling),
            degenerate=(),
        )
        return LossDeclaration(
            components=(self.component_name,),
            metrics=(accuracy, margin),
        )

    def __call__(self, forward_fn: ForwardFn, batch: ExperienceBatch) -> LossOutput:
        """Price one batch of on-policy preference pairs with the DPO loss.

        WHAT IS CLAIMED: the returned scalar is ``weight`` times the mean
        of ``-log sigmoid(beta * h)`` over the batch's pairs; accuracy
        counts rows with ``h > 0.0`` (a tie counts as NOT correct, as in
        DPO, so a fully-tied run stays on the declared degenerate 0.0),
        and the margin metric is the mean of ``h``.

        WHAT IS NOT CLAIMED: that the chosen side is genuinely preferred
        by any reward model; the ranking happened upstream and this loss
        measures the batch it is handed.
        """
        missing = tuple(name for name in self.required_columns if name not in batch.columns)
        if missing:
            raise BatchRefusal(
                f"OnlineDPOLoss requires {len(missing)} of "
                f"{len(self.required_columns)} required columns that are "
                f"absent: {missing!r}; this batch carries "
                f"{tuple(batch.columns)}"
            )
        if len(batch) == 0:
            raise BatchRefusal(
                "OnlineDPOLoss got a batch of 0 rows: a mean over zero "
                "pairs is the all([]) shape -- the loss is unmeasurable, "
                "and returning 0.0 would report a perfect preference loss "
                "for a batch containing no preferences"
            )
        chosen_masks = batch.column(self.chosen_mask_column)
        rejected_masks = batch.column(self.rejected_mask_column)
        reference_chosen = batch.column(self.reference_chosen_column)
        reference_rejected = batch.column(self.reference_rejected_column)

        pairs = _paired_forward_rows(forward_fn, batch, "OnlineDPOLoss")
        pair_losses: list[float] = []
        margins: list[float] = []
        for row_index, (chosen_lps, rejected_lps) in enumerate(pairs):
            pi_c, supervised_c = _masked_side_score(
                chosen_lps, tuple(chosen_masks[row_index]), row_index, "chosen"
            )
            pi_r, supervised_r = _masked_side_score(
                rejected_lps, tuple(rejected_masks[row_index]), row_index, "rejected"
            )
            for side, supervised in (("chosen", supervised_c), ("rejected", supervised_r)):
                if supervised == 0:
                    raise SupervisionRefusal(
                        f"row {row_index}, side {side!r}: supervision mask "
                        f"selected 0 supervised tokens; a pair in which one "
                        f"completion supervises nothing is unmeasurable, and "
                        f"scoring it 0.0 would make an empty completion "
                        f"look maximally likely"
                    )
            ref_c = _reference_score(
                reference_chosen[row_index], self.reference_chosen_column, row_index
            )
            ref_r = _reference_score(
                reference_rejected[row_index], self.reference_rejected_column, row_index
            )
            margin = (pi_c - ref_c) - (pi_r - ref_r)
            margins.append(margin)
            pair_losses.append(_neg_log_sigmoid(float(self.beta) * margin))

        mean_loss = math.fsum(pair_losses) / len(pair_losses)
        mean_margin = math.fsum(margins) / len(margins)
        accuracy = sum(1 for margin in margins if margin > 0.0) / len(margins)
        return LossOutput(
            loss=float(self.weight) * mean_loss,
            components=(
                LossComponent(
                    name=self.component_name,
                    weight=float(self.weight),
                    observed=True,
                    contribution=mean_loss,
                ),
            ),
            metrics=(
                MetricObservation(name=self.accuracy_metric_name, value=accuracy),
                MetricObservation(name=self.margin_metric_name, value=mean_margin),
            ),
        )


@dataclass(frozen=True, slots=True)
class IterativeDPOLoss:
    """DPO against a periodically refreshed reference SNAPSHOT.

    The pricing is identical to ``OnlineDPOLoss`` -- the same log-sigmoid
    DPO margin over masked SUMS -- but the batch carries one extra column,
    ``snapshot_column``, naming the reference snapshot each row's reference
    scores were computed against. Iterative DPO refreshes that snapshot
    only periodically, so a single batch must be UNIFORM in it: a row
    priced against snapshot A beside a row priced against snapshot B is
    two different objectives averaged together, and the mean margin would
    describe neither. This loss therefore refuses a mixed-snapshot batch
    and reports the observed snapshot identity through the metrics
    channel as ``snapshot_metric_name`` is NOT usable for a string -- the
    identity is validated, not emitted, because the metrics channel
    carries finite floats only.

    WHAT IS CLAIMED: the loss is the log-sigmoid DPO form over exactly
    the supplied pairs, measured only when every row names the SAME
    non-empty snapshot identifier; accuracy and the mean margin are
    reported as in ``OnlineDPOLoss``; a zero-supervision side, a missing
    column, an empty snapshot name, a mixed snapshot set or a non-finite
    reading is refused rather than measured.

    WHAT IS NOT CLAIMED: that any refresh SCHEDULE is enforced here. This
    loss sees one batch at a time and can only state that THIS batch is
    snapshot-uniform; when snapshots rotate, and whether the cadence the
    run manifests matches what the store actually served, is the
    producer's and the gates' question, not the objective's.
    """

    beta: float = 0.1
    weight: float = 1.0
    chosen_mask_column: str = "chosen_loss_mask"
    rejected_mask_column: str = "rejected_loss_mask"
    reference_chosen_column: str = "reference_chosen_logprob"
    reference_rejected_column: str = "reference_rejected_logprob"
    snapshot_column: str = "reference_snapshot_id"
    component_name: str = "iterative_dpo_loss"
    accuracy_metric_name: str = "accuracy"
    margin_metric_name: str = "iterative_dpo_margin_mean"
    margin_metric_ceiling: float = 100.0

    def __post_init__(self) -> None:
        _check_positive_finite("beta", self.beta)
        _check_weight(self.weight)
        _check_name_fields(
            (
                ("chosen_mask_column", self.chosen_mask_column),
                ("rejected_mask_column", self.rejected_mask_column),
                ("reference_chosen_column", self.reference_chosen_column),
                ("reference_rejected_column", self.reference_rejected_column),
                ("snapshot_column", self.snapshot_column),
                ("component_name", self.component_name),
                ("accuracy_metric_name", self.accuracy_metric_name),
                ("margin_metric_name", self.margin_metric_name),
            )
        )
        _check_positive_finite("margin_metric_ceiling", self.margin_metric_ceiling)

    @property
    def reference_free(self) -> bool:
        """Whether the objective consumes a frozen reference model's scores.

        WHAT IS CLAIMED: IterativeDPO is reference-anchored -- against a
        snapshot, refreshed periodically -- so this is False. The
        property is the SOURCE that ``semantics()`` derives from, so the
        two statements about this seam cannot drift apart.

        WHAT IS NOT CLAIMED: any gate behaviour attached to this value,
        or any statement about WHEN the snapshot refreshes.
        """
        return False

    @property
    def required_columns(self) -> tuple[str, ...]:
        """The five batch columns this loss consumes, in a stable order.

        WHAT IS CLAIMED: DPO's four columns plus the snapshot-identity
        column; emptiness types and per-row values are measured by
        ``__call__``, not here.

        WHAT IS NOT CLAIMED: that the snapshot identity matches any
        particular store state.
        """
        return (
            self.chosen_mask_column,
            self.rejected_mask_column,
            self.reference_chosen_column,
            self.reference_rejected_column,
            self.snapshot_column,
        )

    def semantics(self) -> AlgorithmSemantics:
        """The loss side of the two-declaration seam check.

        WHAT IS CLAIMED: exactly one field is constrained --
        ``reference_free``, derived from the property rather than
        restated. All other fields abstain with ``None``.

        WHAT IS NOT CLAIMED: that the snapshot UNIFORMITY requirement
        appears in this record; it is a batch-level refusal, checked by
        ``__call__`` where the data is visible.
        """
        return AlgorithmSemantics(reference_free=self.reference_free)

    def declaration(self) -> LossDeclaration:
        """The manifest's statement of what this loss is configured to emit.

        WHAT IS CLAIMED: one component plus the accuracy and margin-mean
        metrics, with the same bounds and degenerate sets
        ``OnlineDPOLoss`` declares and for the same reasons.

        WHAT IS NOT CLAIMED: that the snapshot identity is declared as a
        metric; the metrics channel carries finite floats, and a snapshot
        identifier is validated, not numerically observed.
        """
        accuracy = MetricExpectation(
            name=self.accuracy_metric_name,
            low=0.0,
            high=1.0,
            degenerate=(0.0,),
        )
        margin = MetricExpectation(
            name=self.margin_metric_name,
            low=-float(self.margin_metric_ceiling),
            high=float(self.margin_metric_ceiling),
            degenerate=(),
        )
        return LossDeclaration(
            components=(self.component_name,),
            metrics=(accuracy, margin),
        )

    def __call__(self, forward_fn: ForwardFn, batch: ExperienceBatch) -> LossOutput:
        """Price one snapshot-uniform batch of pairs with the DPO loss.

        WHAT IS CLAIMED: the returned scalar is ``weight`` times the mean
        log-sigmoid loss, computed ONLY after the batch is shown to name
        exactly one non-empty snapshot identifier across all rows;
        accuracy and the margin mean are computed exactly as in
        ``OnlineDPOLoss``.

        WHAT IS NOT CLAIMED: that a refresh happened "recently"; recency
        is invisible to a single-batch objective and is deliberately not
        inferred from anything in the columns.
        """
        missing = tuple(name for name in self.required_columns if name not in batch.columns)
        if missing:
            raise BatchRefusal(
                f"IterativeDPOLoss requires {len(missing)} of "
                f"{len(self.required_columns)} required columns that are "
                f"absent: {missing!r}; this batch carries "
                f"{tuple(batch.columns)}"
            )
        if len(batch) == 0:
            raise BatchRefusal(
                "IterativeDPOLoss got a batch of 0 rows: a mean over zero "
                "pairs is the all([]) shape -- the loss is unmeasurable, "
                "and returning 0.0 would report a perfect preference loss "
                "for a batch containing no preferences"
            )
        snapshots = batch.column(self.snapshot_column)
        snapshot_names: set[str] = set()
        for row_index, raw in enumerate(snapshots):
            if not isinstance(raw, str) or not raw:
                raise BatchRefusal(
                    f"snapshot column {self.snapshot_column!r} row "
                    f"{row_index} is {raw!r}; each row must name the "
                    f"reference snapshot its scores were computed "
                    f"against, and absence of a name is not a name"
                )
            snapshot_names.add(raw)
        if len(snapshot_names) != 1:
            raise BatchRefusal(
                f"IterativeDPOLoss got {len(snapshot_names)} distinct "
                f"snapshot identifiers {sorted(snapshot_names)!r} in one "
                f"batch of {len(batch)} rows; a row priced against one "
                f"snapshot cannot be averaged with a row priced against "
                f"another, because the mean margin would describe neither"
            )

        chosen_masks = batch.column(self.chosen_mask_column)
        rejected_masks = batch.column(self.rejected_mask_column)
        reference_chosen = batch.column(self.reference_chosen_column)
        reference_rejected = batch.column(self.reference_rejected_column)

        pairs = _paired_forward_rows(forward_fn, batch, "IterativeDPOLoss")
        pair_losses: list[float] = []
        margins: list[float] = []
        for row_index, (chosen_lps, rejected_lps) in enumerate(pairs):
            pi_c, supervised_c = _masked_side_score(
                chosen_lps, tuple(chosen_masks[row_index]), row_index, "chosen"
            )
            pi_r, supervised_r = _masked_side_score(
                rejected_lps, tuple(rejected_masks[row_index]), row_index, "rejected"
            )
            for side, supervised in (("chosen", supervised_c), ("rejected", supervised_r)):
                if supervised == 0:
                    raise SupervisionRefusal(
                        f"row {row_index}, side {side!r}: supervision mask "
                        f"selected 0 supervised tokens; a pair in which one "
                        f"completion supervises nothing is unmeasurable, and "
                        f"scoring it 0.0 would make an empty completion "
                        f"look maximally likely"
                    )
            ref_c = _reference_score(
                reference_chosen[row_index], self.reference_chosen_column, row_index
            )
            ref_r = _reference_score(
                reference_rejected[row_index], self.reference_rejected_column, row_index
            )
            margin = (pi_c - ref_c) - (pi_r - ref_r)
            margins.append(margin)
            pair_losses.append(_neg_log_sigmoid(float(self.beta) * margin))

        mean_loss = math.fsum(pair_losses) / len(pair_losses)
        mean_margin = math.fsum(margins) / len(margins)
        accuracy = sum(1 for margin in margins if margin > 0.0) / len(margins)
        return LossOutput(
            loss=float(self.weight) * mean_loss,
            components=(
                LossComponent(
                    name=self.component_name,
                    weight=float(self.weight),
                    observed=True,
                    contribution=mean_loss,
                ),
            ),
            metrics=(
                MetricObservation(name=self.accuracy_metric_name, value=accuracy),
                MetricObservation(name=self.margin_metric_name, value=mean_margin),
            ),
        )


@dataclass(frozen=True, slots=True)
class RAFTLoss:
    """Reward-ranked fine-tuning: SFT on a pre-filtered best-of-N batch.

    RAFT samples N completions per prompt upstream, keeps the
    highest-reward one (or top fraction), and fine-tunes on the
    survivors. THIS loss sees only the survivors: the sampling, ranking
    and filtering all happened before the batch arrived. ``forward_fn``
    must return ONE per-token TARGET log-probability sequence per batch
    row -- the surviving completion -- and the loss is plain masked
    negative log-likelihood, token-normalised within each row and then
    averaged over rows::

        row_nll = -(masked sum of token log-probs) / (supervised tokens)
        loss    = weight * mean(row_nll)

    The token-mean within a row -- rather than DPO's raw SUM -- keeps a
    long survivor from dominating the batch mean purely by length; the
    row mean is then averaged unweighted over rows, so every surviving
    prompt counts once.

    WHAT IS CLAIMED: the loss is the masked token-mean NLL above over
    exactly the supplied rows; the mean per-row NLL is also reported
    through the metrics channel; a row with no supervised tokens, a
    missing mask column or a non-finite reading is refused rather than
    measured. The objective is reference-free: no reference column is
    required and none is read.

    WHAT IS NOT CLAIMED: that the survivors are actually the best of
    anything -- the reward column is not even required, because the
    ranking is the producer's deed and re-checking it here would demand a
    trust this single batch cannot vouch for -- or that optimising this
    objective is stable under repeated self-training; RAFT's known
    distribution-narrowing behaviour is a property of the LOOP, not of
    this per-batch pricing.
    """

    weight: float = 1.0
    loss_mask_column: str = "completion_loss_mask"
    component_name: str = "raft_loss"
    nll_metric_name: str = "raft_nll_mean"
    nll_metric_ceiling: float = 100.0

    def __post_init__(self) -> None:
        _check_weight(self.weight)
        _check_name_fields(
            (
                ("loss_mask_column", self.loss_mask_column),
                ("component_name", self.component_name),
                ("nll_metric_name", self.nll_metric_name),
            )
        )
        _check_positive_finite("nll_metric_ceiling", self.nll_metric_ceiling)

    @property
    def reference_free(self) -> bool:
        """Whether the objective consumes a frozen reference model's scores.

        WHAT IS CLAIMED: RAFT reads no reference column, so this is
        True. The property is the SOURCE that ``semantics()`` derives
        from, so the two statements about this seam cannot drift apart.

        WHAT IS NOT CLAIMED: that reference-freeness is safe here; the
        absence of a KL anchor is the method's construction, and whether
        the run's other machinery compensates is not this objective's
        question.
        """
        return True

    @property
    def required_columns(self) -> tuple[str, ...]:
        """The single batch column this loss consumes.

        WHAT IS CLAIMED: exactly one name -- the supervision mask.

        WHAT IS NOT CLAIMED: that the column's entries are well-typed;
        values are measured by ``__call__``.
        """
        return (self.loss_mask_column,)

    def semantics(self) -> AlgorithmSemantics:
        """The loss side of the two-declaration seam check.

        WHAT IS CLAIMED: exactly one field is constrained --
        ``reference_free``, derived from the property rather than
        restated. The other fields abstain with ``None``: an SFT
        objective forms no ratio, clips nothing, reads no group and
        estimates no KL.

        WHAT IS NOT CLAIMED: any statement about the upstream best-of-N
        width N; the filtered batch carries no trace of it.
        """
        return AlgorithmSemantics(reference_free=self.reference_free)

    def declaration(self) -> LossDeclaration:
        """The manifest's statement of what this loss is configured to emit.

        WHAT IS CLAIMED: one component plus one metric, the mean per-row
        NLL, bounded ``[0.0, nll_metric_ceiling]`` because a masked NLL
        of finite token readings is non-negative, and with degenerate
        ``(0.0,)``: a mean NLL of exactly zero means the model placed ALL
        probability on every supervised token of every survivor, which is
        a frozen or memorised reading, not a healthy one.

        WHAT IS NOT CLAIMED: that this declaration is suitable input to
        ``build_objective_gate_context``; it is the run-manifest
        statement, per the ``LossDeclaration`` contract.
        """
        nll = MetricExpectation(
            name=self.nll_metric_name,
            low=0.0,
            high=float(self.nll_metric_ceiling),
            degenerate=(0.0,),
        )
        return LossDeclaration(
            components=(self.component_name,),
            metrics=(nll,),
        )

    def __call__(self, forward_fn: ForwardFn, batch: ExperienceBatch) -> LossOutput:
        """Price one batch of best-of-N survivors with masked SFT.

        WHAT IS CLAIMED: the returned scalar is ``weight`` times the mean
        over rows of ``-(masked lp sum) / (supervised tokens)``, and the
        unweighted mean of the same per-row NLLs is the reported metric.

        WHAT IS NOT CLAIMED: that any row was the best of its group; the
        filter happened upstream and this loss measures the survivors it
        is handed.
        """
        missing = tuple(name for name in self.required_columns if name not in batch.columns)
        if missing:
            raise BatchRefusal(
                f"RAFTLoss requires {len(missing)} of "
                f"{len(self.required_columns)} required columns that are "
                f"absent: {missing!r}; this batch carries "
                f"{tuple(batch.columns)}"
            )
        if len(batch) == 0:
            raise BatchRefusal(
                "RAFTLoss got a batch of 0 rows: a mean over zero "
                "survivors is the all([]) shape -- the loss is "
                "unmeasurable, and returning 0.0 would report a perfect "
                "SFT loss for a batch that taught the model nothing"
            )
        masks = batch.column(self.loss_mask_column)
        rows = _per_row_forward(forward_fn, batch, "RAFTLoss")

        row_nlls: list[float] = []
        for row_index, token_lps in enumerate(rows):
            score, supervised = _masked_side_score(
                token_lps, tuple(masks[row_index]), row_index, "survivor"
            )
            if supervised == 0:
                raise SupervisionRefusal(
                    f"row {row_index}: supervision mask selected 0 "
                    f"supervised tokens; a survivor that supervises "
                    f"nothing is unmeasurable, and scoring it 0.0 would "
                    f"report perfect imitation of a completion the model "
                    f"was never asked to imitate"
                )
            row_nlls.append(-score / supervised)

        mean_nll = math.fsum(row_nlls) / len(row_nlls)
        return LossOutput(
            loss=float(self.weight) * mean_nll,
            components=(
                LossComponent(
                    name=self.component_name,
                    weight=float(self.weight),
                    observed=True,
                    contribution=mean_nll,
                ),
            ),
            metrics=(MetricObservation(name=self.nll_metric_name, value=mean_nll),),
        )


@dataclass(frozen=True, slots=True)
class BestOfNLoss:
    """SFT on the argmax-reward sample of each group in the batch.

    Unlike ``RAFTLoss``, this loss receives the WHOLE groups: every batch
    row is one sampled completion, grouped by ``group_column`` and scored
    by ``reward_column``. The loss itself selects, per group, the row
    with the highest finite reward -- ties are refused rather than broken
    silently, because a deterministic first-row tiebreak would smuggle an
    ordering dependency into what is claimed to be a reward measurement --
    and prices masked token-mean NLL on exactly the winners, exactly as
    ``RAFTLoss`` prices its pre-filtered survivors.

    doing the selection IN the loss is the point of the type: the metric
    ``reward_metric_name`` reports the mean reward of the selected
    winners, which is a real reading of what the run actually imitated,
    rather than a producer-supplied claim about what it THINKS it imitated.

    WHAT IS CLAIMED: the loss is the masked token-mean NLL over exactly
    one argmax-reward winner per group present in the batch; every group
    must carry at least two rows, because a "best of 1" group selects
    nothing and would report as selection; rewards must be finite scalars;
    tied maxima, empty groups, missing columns and non-finite readings
    are refused rather than measured. The objective is reference-free.

    WHAT IS NOT CLAIMED: that the reward is a good one, that the
    argmax-over-N estimator is unbiased for any downstream quantity, or
    that N is the same across groups -- group sizes are read from the
    batch, not legislated.
    """

    weight: float = 1.0
    loss_mask_column: str = "completion_loss_mask"
    reward_column: str = "reward"
    group_column: str = "group_id"
    component_name: str = "best_of_n_loss"
    nll_metric_name: str = "best_of_n_nll_mean"
    reward_metric_name: str = "best_of_n_winner_reward_mean"
    nll_metric_ceiling: float = 100.0
    reward_metric_ceiling: float = 1000.0

    def __post_init__(self) -> None:
        _check_weight(self.weight)
        _check_name_fields(
            (
                ("loss_mask_column", self.loss_mask_column),
                ("reward_column", self.reward_column),
                ("group_column", self.group_column),
                ("component_name", self.component_name),
                ("nll_metric_name", self.nll_metric_name),
                ("reward_metric_name", self.reward_metric_name),
            )
        )
        _check_positive_finite("nll_metric_ceiling", self.nll_metric_ceiling)
        _check_positive_finite("reward_metric_ceiling", self.reward_metric_ceiling)

    @property
    def reference_free(self) -> bool:
        """Whether the objective consumes a frozen reference model's scores.

        WHAT IS CLAIMED: Best-of-N SFT reads no reference column, so this
        is True. The property is the SOURCE that ``semantics()`` derives
        from, so the two statements about this seam cannot drift apart.

        WHAT IS NOT CLAIMED: any gate behaviour attached to this value.
        """
        return True

    @property
    def required_columns(self) -> tuple[str, ...]:
        """The three batch columns this loss consumes, in a stable order.

        WHAT IS CLAIMED: the supervision mask, the per-row reward and the
        per-row group identifier.

        WHAT IS NOT CLAIMED: that the columns carry well-typed values;
        values are measured by ``__call__``.
        """
        return (
            self.loss_mask_column,
            self.reward_column,
            self.group_column,
        )

    def semantics(self) -> AlgorithmSemantics:
        """The loss side of the two-declaration seam check.

        WHAT IS CLAIMED: exactly one field is constrained --
        ``reference_free``, derived from the property rather than
        restated. All other fields abstain with ``None``; in particular
        the group SIZE is a property of the batch, measured at call time,
        not a configuration constant this record could honestly state.

        WHAT IS NOT CLAIMED: that the group structure of any particular
        batch satisfies the at-least-two-per-group refusal.
        """
        return AlgorithmSemantics(reference_free=self.reference_free)

    def declaration(self) -> LossDeclaration:
        """The manifest's statement of what this loss is configured to emit.

        WHAT IS CLAIMED: one component plus two metrics. The winner NLL
        mean is declared ``[0.0, nll_metric_ceiling]`` with degenerate
        ``(0.0,)``, as in ``RAFTLoss``; the winner reward mean is
        declared with symmetric bounds ``+/- reward_metric_ceiling`` as a
        stated tripwire and an EMPTY degenerate set, because a zero mean
        winner reward is a real reading of the reward model's output, not
        a broken metric.

        WHAT IS NOT CLAIMED: that this declaration is suitable input to
        ``build_objective_gate_context``; it is the run-manifest
        statement, per the ``LossDeclaration`` contract.
        """
        nll = MetricExpectation(
            name=self.nll_metric_name,
            low=0.0,
            high=float(self.nll_metric_ceiling),
            degenerate=(0.0,),
        )
        reward = MetricExpectation(
            name=self.reward_metric_name,
            low=-float(self.reward_metric_ceiling),
            high=float(self.reward_metric_ceiling),
            degenerate=(),
        )
        return LossDeclaration(
            components=(self.component_name,),
            metrics=(nll, reward),
        )

    def __call__(self, forward_fn: ForwardFn, batch: ExperienceBatch) -> LossOutput:
        """Select the argmax-reward row per group and price masked SFT on it.

        WHAT IS CLAIMED: the returned scalar is ``weight`` times the mean
        over the selected winners of ``-(masked lp sum) / (supervised
        tokens)``; the metrics channel carries the mean winner NLL and
        the mean winner reward across groups.

        WHAT IS NOT CLAIMED: that the winners are unique by construction
        -- tied maxima are refused, not resolved -- or that the reward
        ordering matches any preference a human would state.
        """
        missing = tuple(name for name in self.required_columns if name not in batch.columns)
        if missing:
            raise BatchRefusal(
                f"BestOfNLoss requires {len(missing)} of "
                f"{len(self.required_columns)} required columns that are "
                f"absent: {missing!r}; this batch carries "
                f"{tuple(batch.columns)}"
            )
        if len(batch) == 0:
            raise BatchRefusal(
                "BestOfNLoss got a batch of 0 rows: a mean over zero "
                "groups is the all([]) shape -- the loss is unmeasurable, "
                "and returning 0.0 would report a perfect SFT loss for a "
                "batch that selected nothing"
            )
        masks = batch.column(self.loss_mask_column)
        rewards = batch.column(self.reward_column)
        groups = batch.column(self.group_column)

        group_rows: dict[Any, list[int]] = {}
        for row_index, group_id in enumerate(groups):
            if group_id is None:
                raise BatchRefusal(
                    f"group column {self.group_column!r} row {row_index} is "
                    f"None; every row must belong to a named group, and "
                    f"absence of a group is not a group"
                )
            group_rows.setdefault(group_id, []).append(row_index)

        forward_rows = _per_row_forward(forward_fn, batch, "BestOfNLoss")

        winner_rows: list[int] = []
        winner_rewards: list[float] = []
        for group_id, row_indices in group_rows.items():
            if len(row_indices) < 2:
                raise BatchRefusal(
                    f"group {group_id!r} has {len(row_indices)} of the "
                    f"minimum 2 rows required for a best-of-N selection; "
                    f"a best-of-1 group selects nothing, and reporting it "
                    f"as a selection would overstate what was measured"
                )
            scored = [
                (row_index, _reward_value(rewards[row_index], self.reward_column, row_index))
                for row_index in row_indices
            ]
            best_reward = max(reward for _, reward in scored)
            winners = [row_index for row_index, reward in scored if reward == best_reward]
            if len(winners) != 1:
                raise BatchRefusal(
                    f"group {group_id!r} has {len(winners)} rows tied at "
                    f"the maximum reward {best_reward!r} (rows {winners}); "
                    f"a tiebreak would smuggle row order into a quantity "
                    f"claimed to be a reward measurement, so the tie is "
                    f"refused rather than resolved"
                )
            winner_rows.append(winners[0])
            winner_rewards.append(best_reward)

        row_nlls: list[float] = []
        for row_index in winner_rows:
            score, supervised = _masked_side_score(
                forward_rows[row_index], tuple(masks[row_index]), row_index, "winner"
            )
            if supervised == 0:
                raise SupervisionRefusal(
                    f"row {row_index}: the selected winner's supervision "
                    f"mask selected 0 supervised tokens; a winner that "
                    f"supervises nothing is unmeasurable, and scoring it "
                    f"0.0 would report perfect imitation of a completion "
                    f"the model was never asked to imitate"
                )
            row_nlls.append(-score / supervised)

        mean_nll = math.fsum(row_nlls) / len(row_nlls)
        mean_reward = math.fsum(winner_rewards) / len(winner_rewards)
        return LossOutput(
            loss=float(self.weight) * mean_nll,
            components=(
                LossComponent(
                    name=self.component_name,
                    weight=float(self.weight),
                    observed=True,
                    contribution=mean_nll,
                ),
            ),
            metrics=(
                MetricObservation(name=self.nll_metric_name, value=mean_nll),
                MetricObservation(name=self.reward_metric_name, value=mean_reward),
            ),
        )
