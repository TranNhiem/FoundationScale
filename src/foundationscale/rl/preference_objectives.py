"""Preference objectives: IPO, KTO, ORPO, SimPO and CPO, beside DPOLoss.

This module carries five preference-loss implementations on the same
``LossFn`` / ``ExperienceBatch`` / ``LossOutput`` surface that ``DPOLoss``
occupies in ``losses.py``, and it borrows DPO's idiom throughout: one
forward pass per batch, per-token TARGET log-probabilities returned by
``forward_fn``, supervision masks read from batch columns, reference scores
read from batch columns as per-row sequence-level sums, and refusals that
name the offending row, field or count rather than clamping or skipping.

Where the papers and the popular reference implementations disagree, the
PAPER is implemented and the disagreement is stated: IPO is the SQUARED
loss, not a log-sigmoid; KTO reads its KL reference point from a required
batch column rather than estimating it by an in-batch shuffle whose result
depends on batch composition; ORPO refuses, with a named row, at the
float64 wall where a confident model makes ``1 - p`` exactly ``0.0``.

WHAT THIS MODULE CLAIMS: each loss prices exactly the rows and supervised
tokens of the supplied batch, declares exactly the components and metrics
it emits, and refuses degenerate, mis-shaped or non-finite input rather
than reporting a number.

WHAT THIS MODULE DOES NOT CLAIM: that any model forward, any reference
model, or any producer of the consumed batch columns exists here. This is
a torch-free module measuring pure objectives over supplied per-token
readings, enrolled in the torch-free gate.
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
    "CPOLoss",
    "IPOLoss",
    "KTOLoss",
    "ORPOLoss",
    "SimPOLoss",
)


def _mask_value(raw: Any, row: int, position: int) -> float:
    # Copied from losses.py verbatim (private names are not imported across
    # the module boundary). The bool branch comes FIRST because
    # isinstance(True, int) is True; everything after is admitted by VALUE
    # via float(), not by type, and entries outside {0, 1} are refused
    # rather than read as fractional weights, because admitting one would
    # silently change the supervised-token count the losses divide by.
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
    # As losses.py's helper, plus the finiteness refusal ppo_objectives.py
    # established: a non-finite token reading is named with row and
    # position rather than clamped or skipped, because a silently clamped
    # row would change the margin the run reports, and a silently dropped
    # row would change the mean's denominator.
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
            f"would poison the margin this loss reports, and clamping it "
            f"would silently rewrite the row the model actually produced"
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


def _kl_reference_point(raw: Any, column: str, row: int) -> float:
    # KTO's z0: a KL divergence is finite and non-negative, so anything
    # else arriving in the column cannot be a measured reference point.
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise BatchRefusal(
            f"KL reference-point column {column!r} row {row} is {raw!r}, "
            f"which does not convert to a scalar float; the reference "
            f"point must be one finite non-negative KL value per batch row"
        ) from exc
    if not math.isfinite(value) or value < 0.0:
        raise BatchRefusal(
            f"KL reference-point column {column!r} row {row} is {raw!r}, "
            f"which is not a finite non-negative value; a KL divergence "
            f"is non-negative, so {raw!r} cannot be a measured "
            f"reference point"
        )
    return value


def _desirable_flag(raw: Any, column: str, row: int) -> bool:
    # bools first (isinstance(True, int) is True), then 0/1 admitted by
    # value, mirroring _mask_value; anything else is refused, because KTO's
    # row shape gives no paired side to disambiguate a fractional "flag".
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
        f"desirable-flag column {column!r} row {row} is {raw!r}; each row "
        f"must carry one boolean (or 0/1) saying whether the completion "
        f"is desirable"
    )


def _neg_log_sigmoid(value: float) -> float:
    # Numerically stable -log sigmoid(value) as softplus(-value), verbatim
    # the branch pair DPOLoss writes inline: the naive
    # -math.log(1 / (1 + math.exp(-value))) overflows math.exp for large
    # negative values -- exactly the regime a diverging run enters -- so
    # both branches route through log1p.
    if value >= 0.0:
        return math.log1p(math.exp(-value))
    return -value + math.log1p(math.exp(value))


def _sigmoid(value: float) -> float:
    # Branch on sign so math.exp is never handed a large positive argument.
    # KTO's 1 - sigmoid(x) terms are computed as sigmoid(-x), which is the
    # same quantity without the cancellation at sigmoid(x) == 1.0.
    if value >= 0.0:
        return 1.0 / (1.0 + math.exp(-value))
    exp_value = math.exp(value)
    return exp_value / (1.0 + exp_value)


def _masked_side_score(
    token_lps: tuple[Any, ...],
    mask_entries: tuple[Any, ...],
    row_index: int,
    side: str,
) -> tuple[float, int]:
    """Return the masked log-probability SUM and supervised count for one side.

    The reduction is a SUM, not a mean, matching DPOLoss: whether (and by
    what) the caller normalises it is the caller's stated choice -- DPO,
    IPO and CPO compare raw sequence likelihoods, SimPO and ORPO divide by
    the count -- and baking the mean in here would silently impose one of
    those choices on the other.
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
            f"{len(batch)} rows; one (chosen, rejected) row per batch "
            f"row is required"
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


def _check_name_fields(name_fields: tuple[tuple[str, Any], ...]) -> None:
    # DPOLoss's name-field check, factored out because all five losses need
    # it; the message is DPO's verbatim.
    for field_name, value in name_fields:
        if not isinstance(value, str) or not value:
            raise LossConfigRefusal(
                f"{field_name}={value!r}: column, component and metric "
                f"names must be non-empty strings; absence of a name is "
                f"not a name"
            )


@dataclass(frozen=True)
class IPOLoss:
    """Identity preference optimisation (Azar et al. 2023) over paired completions.

    Per batch row = ONE preference pair, scored exactly as ``DPOLoss``
    scores it: ``forward_fn`` is invoked ONCE and must return one row per
    batch row, each a 2-element sequence ``(chosen_token_logprobs,
    rejected_token_logprobs)`` of per-token TARGET log-probabilities, and
    the reference scores arrive as batch columns of sequence-level sums.
    With ``pi``/``ref`` the policy and reference masked sums::

        h         = (pi_chosen - ref_chosen) - (pi_rejected - ref_rejected)
        pair_loss = (h - 1 / (2 * tau)) ** 2
        loss      = weight * mean(pair_loss)

    This is a SQUARED loss, deliberately, and that is the entire point of
    the method: a log-sigmoid saturates, so a handful of already-separated
    pairs can dominate it, while the squared hinge at ``1 / (2 * tau)``
    keeps pulling once the margin is large. A reference implementation that
    wraps this in a sigmoid implements a different objective, and nothing
    here "helpfully" adds one.

    The knob is named ``tau`` and not ``beta`` on purpose: the two are not
    the same quantity, and a config written for DPO pasted into IPO under
    the name ``beta`` would silently mean something else.

    WHAT IS CLAIMED: the loss is the squared form above over exactly the
    supplied pairs, with ``h`` built from masked SUMS of token
    log-probabilities; preference accuracy (fraction of rows with
    ``h > 0``) and the mean margin ``h`` are reported through the metrics
    channel; a zero-supervision side, a missing column, or a non-finite
    reading is refused rather than measured.

    WHAT IS NOT CLAIMED: that the objective saturates -- it does not, and
    cannot be driven by a few already-separated pairs -- or that ``tau``
    interchangeable with DPO's ``beta``; it is not.
    """

    tau: float = 1.0
    weight: float = 1.0
    chosen_mask_column: str = "chosen_loss_mask"
    rejected_mask_column: str = "rejected_loss_mask"
    reference_chosen_column: str = "reference_chosen_logprob"
    reference_rejected_column: str = "reference_rejected_logprob"
    component_name: str = "ipo_loss"
    accuracy_metric_name: str = "accuracy"
    margin_metric_name: str = "ipo_margin_mean"
    margin_metric_ceiling: float = 100.0

    def __post_init__(self) -> None:
        if not isinstance(self.tau, (int, float)) or not math.isfinite(self.tau) or self.tau <= 0.0:
            raise LossConfigRefusal(
                f"tau={self.tau!r}: a non-positive or non-finite tau "
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
        if (
            not isinstance(self.margin_metric_ceiling, (int, float))
            or not math.isfinite(self.margin_metric_ceiling)
            or self.margin_metric_ceiling <= 0.0
        ):
            raise LossConfigRefusal(
                f"margin_metric_ceiling={self.margin_metric_ceiling!r}: the "
                f"declared bound on the margin-mean reading must be finite "
                f"and strictly positive; a non-finite bound makes every "
                f"gate comparison against it silently True, and a bounds "
                f"check that examined nothing reads as coverage"
            )

    @property
    def reference_free(self) -> bool:
        """Whether the objective consumes a frozen reference model's scores.

        WHAT IS CLAIMED: IPO is reference-ANCHORED, so this is False. The
        property is the SOURCE that ``semantics()`` derives from, so the
        two statements about this seam cannot drift apart.

        WHAT IS NOT CLAIMED: any gate behaviour attached to this value.
        """
        return False

    @property
    def required_columns(self) -> tuple[str, ...]:
        """The four batch columns this loss consumes, in a stable order.

        WHAT IS CLAIMED: the tuple is byte-identical in shape to
        ``DPOLoss.required_columns`` -- two masks and two reference-score
        columns -- so callers can build the ``ExperienceBatch(required=...)``
        schema the same way.

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

        WHAT IS CLAIMED: exactly one of the five cross-checked fields --
        ``reference_free`` -- is constrained by a preference objective, and
        it is derived from the ``reference_free`` property rather than
        restated, so the two cannot disagree. The other four abstain with
        ``None``, which the record defines as "this does not constrain that
        seam": a preference objective forms no importance ratio, so it has
        no ratio geometry, no clip bounds and no KL estimator to state, and
        it reads no group, so it has no group size. Declaring a number
        there would be a measured value posing as an unconstrained seam.

        WHAT IS NOT CLAIMED: that reference-freeness is CHECKED against a
        wired reference model. This record states what the loss consumes;
        whether a reference policy is actually present is the role-presence
        layer's question, answered by ``AlgorithmRequirements``.
        """
        return AlgorithmSemantics(reference_free=self.reference_free)

    def declaration(self) -> LossDeclaration:
        """The manifest's statement of what this loss is configured to emit.

        WHAT IS CLAIMED: one component plus two metrics. Accuracy is
        declared ``low=0.0, high=1.0, degenerate=(0.0,)`` exactly as DPO
        declares it -- accuracy pinned at 0.0 means the policy ranked the
        rejected completion above the chosen one on every pair. The margin
        mean is declared with symmetric bounds ``+/- margin_metric_ceiling``
        as a stated tripwire in the KLPenaltyLoss ``metric_ceiling`` idiom,
        and an EMPTY degenerate set: a zero mean margin is a real reading
        of a run sitting at the decision boundary, not a broken metric.

        WHAT IS NOT CLAIMED: that this declaration is suitable input to
        ``build_objective_gate_context``; it is the run-manifest statement,
        per the ``LossDeclaration`` contract.
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
        """Price one batch of preference pairs with the squared IPO loss.

        WHAT IS CLAIMED: the returned scalar is ``weight`` times the mean
        of ``(h - 1 / (2 * tau)) ** 2`` over the batch's pairs, accuracy
        counts rows with ``h > 0.0`` (a tie counts as NOT correct, as in
        DPO, so a fully-tied run stays on the declared degenerate 0.0),
        and the margin metric is the mean of ``h``.

        WHAT IS NOT CLAIMED: that a large negative ``h`` costs less than
        under DPO -- it costs quadratically MORE; that asymmetry is the
        paper's construction, not a defect.
        """
        missing = tuple(name for name in self.required_columns if name not in batch.columns)
        if missing:
            raise BatchRefusal(
                f"IPOLoss requires columns {missing!r}, which are absent; "
                f"this batch carries {tuple(batch.columns)}"
            )
        if len(batch) == 0:
            raise BatchRefusal(
                "IPOLoss got a batch of 0 rows: a mean over zero pairs is "
                "the all([]) shape -- the loss is unmeasurable, and "
                "returning 0.0 would report a perfect preference loss for "
                "a batch containing no preferences"
            )
        chosen_masks = batch.column(self.chosen_mask_column)
        rejected_masks = batch.column(self.rejected_mask_column)
        reference_chosen = batch.column(self.reference_chosen_column)
        reference_rejected = batch.column(self.reference_rejected_column)

        pairs = _paired_forward_rows(forward_fn, batch)
        target = 1.0 / (2.0 * float(self.tau))
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
            pair_losses.append((margin - target) ** 2)

        contribution = self.weight * (sum(pair_losses) / len(pair_losses))
        correct = sum(1 for margin in margins if margin > 0.0)
        accuracy = correct / len(margins)
        mean_margin = sum(margins) / len(margins)
        components = (
            LossComponent(
                name=self.component_name,
                weight=self.weight,
                observed=True,
                contribution=contribution,
            ),
        )
        metrics = (
            MetricObservation(name=self.accuracy_metric_name, value=accuracy),
            MetricObservation(name=self.margin_metric_name, value=mean_margin),
        )
        return LossOutput(loss=contribution, components=components, metrics=metrics)


@dataclass(frozen=True)
class KTOLoss:
    """Kahneman-Tversky optimisation (Ethayarajh et al. 2024), UNPAIRED.

    Each batch row carries ONE completion: its per-token policy
    log-probabilities from ``forward_fn`` (invoked once, one row per batch
    row, SFTLoss's contract), a supervision mask column, a reference
    log-probability-sum column, a KL reference-point column ``z0``, and a
    boolean saying whether the completion is desirable. With
    ``r = beta * (pi - ref)``::

        desirable   loss = lambda_desirable   * sigmoid(z0 - r)
        undesirable loss = lambda_undesirable * sigmoid(r - z0)

    written as ``sigmoid`` of the NEGATED argument rather than
    ``1 - sigmoid(...)`` so no cancellation is taken, and stabilised by the
    sign branch in ``_sigmoid``.

    ``z0`` is read from a REQUIRED batch column. The popular reference
    implementation estimates it by shuffling completions within the
    microbatch and averaging the resulting KL; that is deliberately NOT
    done here. An in-batch shuffle makes a row's loss a function of which
    other rows happened to share its microbatch, so the same row scores
    differently under a different batch order, world size, or
    gradient-accumulation factor -- and nothing in the run's record would
    say so. An unrecorded, composition-dependent estimate is not a
    measurement: if the column is absent this loss REFUSES and names it.

    WHAT IS CLAIMED: the per-row losses above are computed over exactly
    the batch's rows, averaged, scaled by ``weight``, and reported with
    the measured desirable and undesirable COUNTS.

    WHAT IS NOT CLAIMED: any pair accuracy -- there are no pairs in this
    batch shape, and fabricating one (for instance by pairing rows within
    the batch) would be a metric whose value depends on batch composition,
    the same defect the required ``z0`` column exists to remove. Nor is
    any in-batch or cross-batch estimation of ``z0`` claimed; the recorded
    column is the only reference point this loss will read. Finally, no
    choice of lambda values is endorsed; they are the caller's stated
    weighting of the two row families.
    """

    beta: float = 0.1
    weight: float = 1.0
    lambda_desirable: float = 1.0
    lambda_undesirable: float = 1.0
    mask_column: str = "loss_mask"
    reference_column: str = "reference_logprob"
    kl_reference_point_column: str = "kl_reference_point"
    desirable_column: str = "is_desirable"
    component_name: str = "kto_loss"
    desirable_count_metric_name: str = "desirable_count"
    undesirable_count_metric_name: str = "undesirable_count"
    count_metric_ceiling: float = 1.0e6
    paired_batch_columns: tuple[str, ...] = (
        "chosen_loss_mask",
        "rejected_loss_mask",
        "reference_chosen_logprob",
        "reference_rejected_logprob",
    )

    def __post_init__(self) -> None:
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
        for field_name, value in (
            ("lambda_desirable", self.lambda_desirable),
            ("lambda_undesirable", self.lambda_undesirable),
        ):
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0.0:
                raise LossConfigRefusal(
                    f"{field_name}={value!r}: each row-family weight must "
                    f"be finite and strictly positive; a zero weight would "
                    f"silently drop one family of rows from the objective"
                )
        _check_name_fields(
            (
                ("mask_column", self.mask_column),
                ("reference_column", self.reference_column),
                ("kl_reference_point_column", self.kl_reference_point_column),
                ("desirable_column", self.desirable_column),
                ("component_name", self.component_name),
                ("desirable_count_metric_name", self.desirable_count_metric_name),
                ("undesirable_count_metric_name", self.undesirable_count_metric_name),
            )
        )
        if (
            not isinstance(self.count_metric_ceiling, (int, float))
            or not math.isfinite(self.count_metric_ceiling)
            or self.count_metric_ceiling <= 0.0
        ):
            raise LossConfigRefusal(
                f"count_metric_ceiling={self.count_metric_ceiling!r}: the "
                f"declared upper bound on the row-family count readings "
                f"must be finite and strictly positive; a non-finite bound "
                f"makes every gate comparison against it silently True"
            )
        if not isinstance(self.paired_batch_columns, tuple) or not self.paired_batch_columns:
            raise LossConfigRefusal(
                f"paired_batch_columns={self.paired_batch_columns!r}: the "
                f"paired-marker set is refused empty or non-tuple, because "
                f"an empty set would admit every paired batch unchecked "
                f"and a paired batch is a silently halved denominator"
            )
        for index, marker in enumerate(self.paired_batch_columns):
            if not isinstance(marker, str) or not marker:
                raise LossConfigRefusal(
                    f"paired_batch_columns[{index}]={marker!r}: paired "
                    f"marker column names must be non-empty strings; "
                    f"absence of a name is not a name"
                )

    @property
    def reference_free(self) -> bool:
        """Whether the objective consumes a frozen reference model's scores.

        WHAT IS CLAIMED: KTO is reference-ANCHORED, so this is False; the
        property is the SOURCE that ``semantics()`` derives from, so the
        two statements about this seam cannot drift apart.

        WHAT IS NOT CLAIMED: any gate behaviour attached to this value.
        """
        return False

    @property
    def required_columns(self) -> tuple[str, ...]:
        """The four batch columns this loss consumes, in a stable order.

        WHAT IS CLAIMED: the tuple names the mask, the reference score,
        the KL reference point, and the desirable flag -- and deliberately
        does NOT name any chosen/rejected column, because KTO's row shape
        is one completion per row and its schema must say so.

        WHAT IS NOT CLAIMED: that the columns carry well-typed values, or
        that the batch carries no paired columns on top; the latter is
        refused separately by ``__call__``.
        """
        return (
            self.mask_column,
            self.reference_column,
            self.kl_reference_point_column,
            self.desirable_column,
        )

    def semantics(self) -> AlgorithmSemantics:
        """The loss side of the two-declaration seam check.

        WHAT IS CLAIMED: exactly one of the five cross-checked fields --
        ``reference_free`` -- is constrained by a preference objective, and
        it is derived from the ``reference_free`` property rather than
        restated, so the two cannot disagree. The other four abstain with
        ``None``, which the record defines as "this does not constrain that
        seam": a preference objective forms no importance ratio, so it has
        no ratio geometry, no clip bounds and no KL estimator to state, and
        it reads no group, so it has no group size. Declaring a number
        there would be a measured value posing as an unconstrained seam.

        WHAT IS NOT CLAIMED: that reference-freeness is CHECKED against a
        wired reference model. This record states what the loss consumes;
        whether a reference policy is actually present is the role-presence
        layer's question, answered by ``AlgorithmRequirements``.
        """
        return AlgorithmSemantics(reference_free=self.reference_free)

    def declaration(self) -> LossDeclaration:
        """The manifest's statement of what this loss is configured to emit.

        WHAT IS CLAIMED: one component and two count metrics, each declared
        ``low=0.0, high=count_metric_ceiling`` with an EMPTY degenerate set.
        The ceiling is a stated tripwire in the KLPenaltyLoss
        ``metric_ceiling`` idiom, not a property of the counts; a run whose
        batches legitimately exceed it should RAISE the field, not read the
        gate finding as noise. The degenerate sets are empty because a batch
        that is all-desirable or all-undesirable is a data property a run
        may legitimately produce, not a broken metric.

        WHAT IS NOT CLAIMED: that this declaration is suitable input to
        ``build_objective_gate_context``; it is the run-manifest statement,
        per the ``LossDeclaration`` contract.
        """
        desirable = MetricExpectation(
            name=self.desirable_count_metric_name,
            low=0.0,
            high=float(self.count_metric_ceiling),
            degenerate=(),
        )
        undesirable = MetricExpectation(
            name=self.undesirable_count_metric_name,
            low=0.0,
            high=float(self.count_metric_ceiling),
            degenerate=(),
        )
        return LossDeclaration(
            components=(self.component_name,),
            metrics=(desirable, undesirable),
        )

    def __call__(self, forward_fn: ForwardFn, batch: ExperienceBatch) -> LossOutput:
        """Price one batch of unpaired completions against the reference point.

        WHAT IS CLAIMED: the returned scalar is ``weight`` times the mean
        of the per-row KTO losses, and the desirable/undesirable counts are
        emitted with the same denominator the mean used.

        WHAT IS NOT CLAIMED: that ``z0`` was produced by any particular
        estimator -- only that it arrived recorded, finite, and
        non-negative in the named column -- or that a row's loss is
        independent of its batch; it is, by construction, but only because
        the reference point is a recorded input rather than an in-batch
        estimate.
        """
        if self.kl_reference_point_column not in batch.columns:
            raise BatchRefusal(
                f"KTOLoss requires KL reference-point column "
                f"{self.kl_reference_point_column!r}, which is absent; "
                f"this batch carries {tuple(batch.columns)}. Reading the "
                f"reference point from a recorded column is mandatory: the "
                f"alternative, estimating it by shuffling completions "
                f"within the microbatch, makes a row's loss a function of "
                f"which other rows happened to share its microbatch, and "
                f"an unrecorded, composition-dependent estimate is not a "
                f"measurement"
            )
        missing = tuple(name for name in self.required_columns if name not in batch.columns)
        if missing:
            raise BatchRefusal(
                f"KTOLoss requires columns {missing!r}, which are absent; "
                f"this batch carries {tuple(batch.columns)}"
            )
        present_paired = tuple(name for name in self.paired_batch_columns if name in batch.columns)
        if present_paired:
            raise BatchRefusal(
                f"KTOLoss got a batch carrying {len(present_paired)} of "
                f"{len(self.paired_batch_columns)} paired marker columns: "
                f"{present_paired!r}; KTO prices ONE completion per row, "
                f"so a paired batch handed to it is a silently halved "
                f"denominator, because half of each pair's completions "
                f"would never enter the mean"
            )
        if len(batch) == 0:
            raise BatchRefusal(
                "KTOLoss got a batch of 0 rows: a mean over zero rows is "
                "the all([]) shape -- the loss is unmeasurable, and "
                "returning 0.0 would report a perfect KTO loss for a "
                "batch containing no completions"
            )
        masks = batch.column(self.mask_column)
        reference = batch.column(self.reference_column)
        kl_points = batch.column(self.kl_reference_point_column)
        desirable_flags = batch.column(self.desirable_column)

        token_rows = [tuple(row) for row in forward_fn(batch)]
        if len(token_rows) != len(batch):
            raise BatchRefusal(
                f"forward_fn returned {len(token_rows)} per-token rows for "
                f"a batch of {len(batch)} rows; one row of "
                f"log-probabilities per batch row is required"
            )

        row_losses: list[float] = []
        desirable_count = 0
        undesirable_count = 0
        for row_index, token_lps in enumerate(token_rows):
            pi, supervised = _masked_side_score(
                token_lps, tuple(masks[row_index]), row_index, "completion"
            )
            if supervised == 0:
                raise SupervisionRefusal(
                    f"row {row_index}: supervision mask selected 0 "
                    f"supervised tokens; a row that supervises nothing is "
                    f"unmeasurable, and scoring it 0.0 would make an "
                    f"empty completion look maximally likely"
                )
            ref = _reference_score(reference[row_index], self.reference_column, row_index)
            z0 = _kl_reference_point(
                kl_points[row_index], self.kl_reference_point_column, row_index
            )
            desirable = _desirable_flag(
                desirable_flags[row_index], self.desirable_column, row_index
            )
            r = self.beta * (pi - ref)
            if desirable:
                desirable_count += 1
                row_losses.append(self.lambda_desirable * _sigmoid(z0 - r))
            else:
                undesirable_count += 1
                row_losses.append(self.lambda_undesirable * _sigmoid(r - z0))

        contribution = self.weight * (sum(row_losses) / len(row_losses))
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
                name=self.desirable_count_metric_name,
                value=float(desirable_count),
            ),
            MetricObservation(
                name=self.undesirable_count_metric_name,
                value=float(undesirable_count),
            ),
        )
        return LossOutput(loss=contribution, components=components, metrics=metrics)


@dataclass(frozen=True)
class ORPOLoss:
    """Odds ratio preference optimisation (Hong et al. 2024), reference-FREE.

    Per batch row = ONE preference pair under DPOLoss's forward contract.
    TWO components, with the paper's total ``L_SFT + lambda_ * L_OR``: a
    supervised NLL term over the chosen completion (masked mean, the
    ``SFTLoss`` form, fixed weight 1.0) and the odds-ratio term::

        p_side  = exp(masked_sum / supervised_count)   # length-normalised
        L_OR    = -log sigmoid(log(odds_w / odds_l))   # odds = p / (1 - p)

    The odds are computed as ``log_odds = average - log(-expm1(average))``
    so that ``1 - p`` is evaluated through ``expm1`` rather than by
    subtracting near-1.0 floats; even so, a completion whose
    length-normalised log-probability is indistinguishable from 0.0 has
    probability exactly 1.0 in float64, ``1 - p`` is exactly 0.0, and the
    odds are undefined. That is a real numerical wall: the offending row
    and side are NAMED and the loss REFUSES, rather than emitting an
    infinite margin.

    ``lambda_`` weights the odds-ratio component and must be strictly
    positive: both components are always declared, and a declared
    component at weight 0.0 is a blocking FAIL at
    ``LossComponentCoverageGate``.

    WHAT IS CLAIMED: both terms are computed over exactly the batch's
    supervised tokens, the declaration names both components, accuracy
    (fraction of rows with ``log_odds_w > log_odds_l``) and the mean
    log-odds margin are reported, and the float64 wall above is a refusal,
    never a clamp.

    WHAT IS NOT CLAIMED: that the objective is defined where ``1 - p``
    underflows to exactly ``0.0`` -- it is not, hence the refusal -- that
    any reference model was consulted (none is), or that the SFT term can
    be switched off; removing it is a different objective, not a
    configuration of this one.
    """

    lambda_: float = 1.0
    chosen_mask_column: str = "chosen_loss_mask"
    rejected_mask_column: str = "rejected_loss_mask"
    component_name: str = "orpo_odds_loss"
    sft_component_name: str = "sft_loss"
    accuracy_metric_name: str = "accuracy"
    margin_metric_name: str = "orpo_margin_mean"
    margin_metric_ceiling: float = 100.0

    def __post_init__(self) -> None:
        if (
            not isinstance(self.lambda_, (int, float))
            or not math.isfinite(self.lambda_)
            or self.lambda_ <= 0.0
        ):
            raise LossConfigRefusal(
                f"lambda_={self.lambda_!r}: the declared odds-ratio "
                f"component must carry a positive finite weight; a zero "
                f"weight is a blocking FAIL at LossComponentCoverageGate, "
                f"and ORPO always declares both of its terms, so the loss "
                f"refuses to be built that way"
            )
        _check_name_fields(
            (
                ("chosen_mask_column", self.chosen_mask_column),
                ("rejected_mask_column", self.rejected_mask_column),
                ("component_name", self.component_name),
                ("sft_component_name", self.sft_component_name),
                ("accuracy_metric_name", self.accuracy_metric_name),
                ("margin_metric_name", self.margin_metric_name),
            )
        )
        if self.component_name == self.sft_component_name:
            raise LossConfigRefusal(
                f"component_name and sft_component_name are both "
                f"{self.component_name!r}: two components sharing one name "
                f"collapse the coverage denominator"
            )
        if (
            not isinstance(self.margin_metric_ceiling, (int, float))
            or not math.isfinite(self.margin_metric_ceiling)
            or self.margin_metric_ceiling <= 0.0
        ):
            raise LossConfigRefusal(
                f"margin_metric_ceiling={self.margin_metric_ceiling!r}: the "
                f"declared bound on the margin-mean reading must be finite "
                f"and strictly positive; a non-finite bound makes every "
                f"gate comparison against it silently True, and a bounds "
                f"check that examined nothing reads as coverage"
            )

    @property
    def reference_free(self) -> bool:
        """Whether the objective consumes a frozen reference model's scores.

        WHAT IS CLAIMED: ORPO is reference-FREE, so this is True; the
        property is the SOURCE that ``semantics()`` derives from, so the
        two statements about this seam cannot drift apart.

        WHAT IS NOT CLAIMED: any gate behaviour attached to this value.
        """
        return True

    @property
    def required_columns(self) -> tuple[str, ...]:
        """The two batch columns this loss consumes, in a stable order.

        WHAT IS CLAIMED: the tuple names only the two supervision masks;
        ORPO reads no reference columns, and its schema must say so.

        WHAT IS NOT CLAIMED: that the columns carry well-typed values;
        shapes and values are measured by ``__call__``.
        """
        return (self.chosen_mask_column, self.rejected_mask_column)

    def semantics(self) -> AlgorithmSemantics:
        """The loss side of the two-declaration seam check.

        WHAT IS CLAIMED: exactly one of the five cross-checked fields --
        ``reference_free`` -- is constrained by a preference objective, and
        it is derived from the ``reference_free`` property rather than
        restated, so the two cannot disagree. The other four abstain with
        ``None``, which the record defines as "this does not constrain that
        seam": a preference objective forms no importance ratio, so it has
        no ratio geometry, no clip bounds and no KL estimator to state, and
        it reads no group, so it has no group size. Declaring a number
        there would be a measured value posing as an unconstrained seam.

        WHAT IS NOT CLAIMED: that reference-freeness is CHECKED against a
        wired reference model. This record states what the loss consumes;
        whether a reference policy is actually present is the role-presence
        layer's question, answered by ``AlgorithmRequirements``.
        """
        return AlgorithmSemantics(reference_free=self.reference_free)

    def declaration(self) -> LossDeclaration:
        """The manifest's statement of what this loss is configured to emit.

        WHAT IS CLAIMED: TWO components, always -- the SFT term and the
        odds-ratio term, in the formula's order -- plus the accuracy metric
        declared exactly as DPO declares it (``degenerate=(0.0,)``: accuracy
        pinned at 0.0 means the rejected completion's odds exceeded the
        chosen one's on every pair) and the margin mean with symmetric
        tripwire bounds and an empty degenerate set.

        WHAT IS NOT CLAIMED: that this declaration is suitable input to
        ``build_objective_gate_context``; it is the run-manifest statement,
        per the ``LossDeclaration`` contract.
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
            components=(self.sft_component_name, self.component_name),
            metrics=(accuracy, margin),
        )

    def __call__(self, forward_fn: ForwardFn, batch: ExperienceBatch) -> LossOutput:
        """Price one batch as supervised NLL plus ``lambda_`` times odds ratio.

        WHAT IS CLAIMED: the returned scalar is the sum of the two
        components' contributions as declared, computed over exactly the
        batch's supervised tokens; the per-side length-normalised odds use
        the ``expm1`` form and the exact-1.0-probability wall is refused
        with row and side named.

        WHAT IS NOT CLAIMED: that the margin is finite on every possible
        batch -- at the documented wall it is not defined at all, and the
        refusal is the claim.
        """
        missing = tuple(name for name in self.required_columns if name not in batch.columns)
        if missing:
            raise BatchRefusal(
                f"ORPOLoss requires columns {missing!r}, which are absent; "
                f"this batch carries {tuple(batch.columns)}"
            )
        if len(batch) == 0:
            raise BatchRefusal(
                "ORPOLoss got a batch of 0 rows: a mean over zero pairs is "
                "the all([]) shape -- the loss is unmeasurable, and "
                "returning 0.0 would report a perfect preference loss for "
                "a batch containing no preferences"
            )
        chosen_masks = batch.column(self.chosen_mask_column)
        rejected_masks = batch.column(self.rejected_mask_column)

        pairs = _paired_forward_rows(forward_fn, batch)
        odds_losses: list[float] = []
        margins: list[float] = []
        chosen_sum_total = 0.0
        chosen_supervised_total = 0
        for row_index, (chosen_lps, rejected_lps) in enumerate(pairs):
            side_rows = (
                ("chosen", chosen_lps, tuple(chosen_masks[row_index])),
                ("rejected", rejected_lps, tuple(rejected_masks[row_index])),
            )
            log_odds: dict[str, float] = {}
            for side, token_lps, mask_entries in side_rows:
                score, supervised = _masked_side_score(token_lps, mask_entries, row_index, side)
                if supervised == 0:
                    raise SupervisionRefusal(
                        f"row {row_index}, side {side!r}: supervision mask "
                        f"selected 0 supervised tokens; a pair in which one "
                        f"completion supervises nothing is unmeasurable, and "
                        f"normalising to 0.0 would make an empty completion "
                        f"look maximally likely"
                    )
                average = score / supervised
                one_minus_p = -math.expm1(average)
                if one_minus_p <= 0.0:
                    raise BatchRefusal(
                        f"row {row_index}, side {side!r}: the "
                        f"length-normalised log-probability {average!r} "
                        f"rounds to a completion probability of exactly "
                        f"1.0 in float64, so 1 - p is exactly 0.0 and the "
                        f"odds p / (1 - p) are undefined; ORPO is "
                        f"undefined on this row, and the loss refuses "
                        f"rather than emitting an infinite margin"
                    )
                log_odds[side] = average - math.log(one_minus_p)
                if side == "chosen":
                    chosen_sum_total += score
                    chosen_supervised_total += supervised
            margin = log_odds["chosen"] - log_odds["rejected"]
            margins.append(margin)
            odds_losses.append(_neg_log_sigmoid(margin))

        sft_contribution = -(chosen_sum_total / chosen_supervised_total)
        odds_contribution = self.lambda_ * (sum(odds_losses) / len(odds_losses))
        correct = sum(1 for margin in margins if margin > 0.0)
        accuracy = correct / len(margins)
        mean_margin = sum(margins) / len(margins)
        loss = sft_contribution + odds_contribution
        components = (
            LossComponent(
                name=self.sft_component_name,
                weight=1.0,
                observed=True,
                contribution=sft_contribution,
            ),
            LossComponent(
                name=self.component_name,
                weight=self.lambda_,
                observed=True,
                contribution=odds_contribution,
            ),
        )
        metrics = (
            MetricObservation(name=self.accuracy_metric_name, value=accuracy),
            MetricObservation(name=self.margin_metric_name, value=mean_margin),
        )
        return LossOutput(loss=loss, components=components, metrics=metrics)


@dataclass(frozen=True)
class SimPOLoss:
    """Simple preference optimisation (Meng et al. 2024), reference-FREE.

    Per batch row = ONE preference pair under DPOLoss's forward contract.
    The objective is LENGTH-NORMALISED, which is what makes it work
    without a reference::

        r(y)      = (beta / count(y)) * masked_sum(y)
        margin    = r(chosen) - r(rejected) - gamma
        pair_loss = -log sigmoid(margin)

    where ``count(y)`` is the number of SUPERVISED tokens in the
    completion, DERIVED from the batch's mask -- never accepted as a
    column, because a column could disagree with the mask it claims to
    summarise. A row whose supervised-token count is 0 is REFUSED rather
    than divided by. ``gamma`` is the target reward margin; it defaults to
    0.0 because the corpus idiom defaults every knob (DPO's ``beta=0.1``,
    PPO's ``clip_epsilon=0.2``), and tuned runs are expected to set it.

    The negative log-sigmoid is evaluated through ``_neg_log_sigmoid``'s
    stable softplus branches, as everywhere in this module.

    WHAT IS CLAIMED: the normalised rewards and margin above are computed
    over exactly the batch's supervised tokens, accuracy counts rows whose
    FULL margin (gamma included) is positive, the mean margin is reported,
    and a zero-count row is refused before any division by it.

    WHAT IS NOT CLAIMED: that accuracy means what DPO's accuracy means --
    it measures the fraction of pairs already PAST the target margin, so a
    freshly initialised model below ``gamma`` honestly reads near 0.0 --
    or that any reference model was consulted (none was).
    """

    beta: float = 1.0
    gamma: float = 0.0
    weight: float = 1.0
    chosen_mask_column: str = "chosen_loss_mask"
    rejected_mask_column: str = "rejected_loss_mask"
    component_name: str = "simpo_loss"
    accuracy_metric_name: str = "accuracy"
    margin_metric_name: str = "simpo_margin_mean"
    margin_metric_ceiling: float = 100.0

    def __post_init__(self) -> None:
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
            not isinstance(self.gamma, (int, float))
            or not math.isfinite(self.gamma)
            or self.gamma < 0.0
        ):
            raise LossConfigRefusal(
                f"gamma={self.gamma!r}: the target reward margin must be "
                f"finite and non-negative; a negative target supervises "
                f"the policy towards ranking the rejected completion "
                f"above the chosen one"
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
        _check_name_fields(
            (
                ("chosen_mask_column", self.chosen_mask_column),
                ("rejected_mask_column", self.rejected_mask_column),
                ("component_name", self.component_name),
                ("accuracy_metric_name", self.accuracy_metric_name),
                ("margin_metric_name", self.margin_metric_name),
            )
        )
        if (
            not isinstance(self.margin_metric_ceiling, (int, float))
            or not math.isfinite(self.margin_metric_ceiling)
            or self.margin_metric_ceiling <= 0.0
        ):
            raise LossConfigRefusal(
                f"margin_metric_ceiling={self.margin_metric_ceiling!r}: the "
                f"declared bound on the margin-mean reading must be finite "
                f"and strictly positive; a non-finite bound makes every "
                f"gate comparison against it silently True, and a bounds "
                f"check that examined nothing reads as coverage"
            )

    @property
    def reference_free(self) -> bool:
        """Whether the objective consumes a frozen reference model's scores.

        WHAT IS CLAIMED: SimPO is reference-FREE, so this is True; the
        property is the SOURCE that ``semantics()`` derives from, so the
        two statements about this seam cannot drift apart.

        WHAT IS NOT CLAIMED: any gate behaviour attached to this value.
        """
        return True

    @property
    def required_columns(self) -> tuple[str, ...]:
        """The two batch columns this loss consumes, in a stable order.

        WHAT IS CLAIMED: the tuple names only the two supervision masks;
        SimPO reads no reference columns, and its schema must say so.

        WHAT IS NOT CLAIMED: that the columns carry well-typed values;
        shapes and values are measured by ``__call__``.
        """
        return (self.chosen_mask_column, self.rejected_mask_column)

    def semantics(self) -> AlgorithmSemantics:
        """The loss side of the two-declaration seam check.

        WHAT IS CLAIMED: exactly one of the five cross-checked fields --
        ``reference_free`` -- is constrained by a preference objective, and
        it is derived from the ``reference_free`` property rather than
        restated, so the two cannot disagree. The other four abstain with
        ``None``, which the record defines as "this does not constrain that
        seam": a preference objective forms no importance ratio, so it has
        no ratio geometry, no clip bounds and no KL estimator to state, and
        it reads no group, so it has no group size. Declaring a number
        there would be a measured value posing as an unconstrained seam.

        WHAT IS NOT CLAIMED: that reference-freeness is CHECKED against a
        wired reference model. This record states what the loss consumes;
        whether a reference policy is actually present is the role-presence
        layer's question, answered by ``AlgorithmRequirements``.
        """
        return AlgorithmSemantics(reference_free=self.reference_free)

    def declaration(self) -> LossDeclaration:
        """The manifest's statement of what this loss is configured to emit.

        WHAT IS CLAIMED: one component and two metrics. Accuracy is
        declared ``low=0.0, high=1.0`` with an EMPTY degenerate set --
        deliberately, unlike the other paired objectives: because accuracy
        here counts pairs past the TARGET margin, a fresh model honestly
        reads 0.0, the diagnostic gates run at STEP_ZERO, and refusing the
        honest starting reading would fail every legitimate SimPO run at
        launch. The margin mean carries symmetric tripwire bounds and an
        empty degenerate set.

        WHAT IS NOT CLAIMED: that this declaration is suitable input to
        ``build_objective_gate_context``; it is the run-manifest statement,
        per the ``LossDeclaration`` contract.
        """
        accuracy = MetricExpectation(
            name=self.accuracy_metric_name,
            low=0.0,
            high=1.0,
            degenerate=(),
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
        """Price one batch with the length-normalised margin loss.

        WHAT IS CLAIMED: the returned scalar is ``weight`` times the mean
        of the stable ``-log sigmoid`` of the full margin (gamma
        included), over exactly the batch's pairs; each side's reward is
        its masked sum divided by its DERIVED supervised-token count.

        WHAT IS NOT CLAIMED: that the normalisation makes margins
        comparable to DPO's -- they are in different units, which is why
        the margin metric carries a SimPO-specific name.
        """
        missing = tuple(name for name in self.required_columns if name not in batch.columns)
        if missing:
            raise BatchRefusal(
                f"SimPOLoss requires columns {missing!r}, which are absent; "
                f"this batch carries {tuple(batch.columns)}"
            )
        if len(batch) == 0:
            raise BatchRefusal(
                "SimPOLoss got a batch of 0 rows: a mean over zero pairs is "
                "the all([]) shape -- the loss is unmeasurable, and "
                "returning 0.0 would report a perfect preference loss for "
                "a batch containing no preferences"
            )
        chosen_masks = batch.column(self.chosen_mask_column)
        rejected_masks = batch.column(self.rejected_mask_column)

        pairs = _paired_forward_rows(forward_fn, batch)
        pair_losses: list[float] = []
        margins: list[float] = []
        for row_index, (chosen_lps, rejected_lps) in enumerate(pairs):
            side_rows = (
                ("chosen", chosen_lps, tuple(chosen_masks[row_index])),
                ("rejected", rejected_lps, tuple(rejected_masks[row_index])),
            )
            rewards: dict[str, float] = {}
            for side, token_lps, mask_entries in side_rows:
                score, supervised = _masked_side_score(token_lps, mask_entries, row_index, side)
                if supervised == 0:
                    raise SupervisionRefusal(
                        f"row {row_index}, side {side!r}: supervision mask "
                        f"selected 0 supervised tokens; SimPO divides the "
                        f"masked log-probability sum by that count, so this "
                        f"row would force a division by zero, and skipping "
                        f"it would silently change the mean's denominator"
                    )
                rewards[side] = (self.beta / supervised) * score
            margin = rewards["chosen"] - rewards["rejected"] - float(self.gamma)
            margins.append(margin)
            pair_losses.append(_neg_log_sigmoid(margin))

        contribution = self.weight * (sum(pair_losses) / len(pair_losses))
        correct = sum(1 for margin in margins if margin > 0.0)
        accuracy = correct / len(margins)
        mean_margin = sum(margins) / len(margins)
        components = (
            LossComponent(
                name=self.component_name,
                weight=self.weight,
                observed=True,
                contribution=contribution,
            ),
        )
        metrics = (
            MetricObservation(name=self.accuracy_metric_name, value=accuracy),
            MetricObservation(name=self.margin_metric_name, value=mean_margin),
        )
        return LossOutput(loss=contribution, components=components, metrics=metrics)


@dataclass(frozen=True)
class CPOLoss:
    """Contrastive preference optimisation (Xu et al. 2024), reference-FREE.

    Per batch row = ONE preference pair under DPOLoss's forward contract.
    The objective is DPO's with the reference term dropped, plus an SFT
    term on the chosen completion::

        L = -log sigmoid(beta * pi_chosen - beta * pi_rejected)
            + lambda_ * NLL(chosen)

    with ``pi`` the masked SUM of target log-probabilities and the
    log-sigmoid evaluated through the stable softplus branches. TWO
    components are declared and computed, always; ``lambda_`` must be
    strictly positive for the same reason ORPO gives: a declared component
    at weight 0.0 is a blocking FAIL at ``LossComponentCoverageGate``.

    WHAT IS CLAIMED: both terms are computed over exactly the batch's
    supervised sequence-level scores, accuracy (fraction of rows with a
    positive beta-scaled margin) and the mean margin are reported with
    DPO's degenerate-zero accuracy declaration, and a zero-supervision
    side or non-finite reading is refused rather than measured.

    WHAT IS NOT CLAIMED: that dropping the reference is exact. It is an
    APPROXIMATION that holds under the paper's uniform-reference assumption
    -- a uniform reference contributes identically to both sides and
    cancels in the difference. This implementation does not and CANNOT
    check that assumption against anything: there is no reference here to
    check against. A run whose true reference is materially non-uniform is
    optimising a different objective than DPO would, and nothing in this
    loss will say so.
    """

    beta: float = 0.1
    lambda_: float = 1.0
    chosen_mask_column: str = "chosen_loss_mask"
    rejected_mask_column: str = "rejected_loss_mask"
    component_name: str = "cpo_loss"
    sft_component_name: str = "sft_loss"
    accuracy_metric_name: str = "accuracy"
    margin_metric_name: str = "cpo_margin_mean"
    margin_metric_ceiling: float = 100.0

    def __post_init__(self) -> None:
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
            not isinstance(self.lambda_, (int, float))
            or not math.isfinite(self.lambda_)
            or self.lambda_ <= 0.0
        ):
            raise LossConfigRefusal(
                f"lambda_={self.lambda_!r}: the declared SFT component "
                f"must carry a positive finite weight; a zero weight is a "
                f"blocking FAIL at LossComponentCoverageGate, and CPO "
                f"always declares both of its terms, so the loss refuses "
                f"to be built that way"
            )
        _check_name_fields(
            (
                ("chosen_mask_column", self.chosen_mask_column),
                ("rejected_mask_column", self.rejected_mask_column),
                ("component_name", self.component_name),
                ("sft_component_name", self.sft_component_name),
                ("accuracy_metric_name", self.accuracy_metric_name),
                ("margin_metric_name", self.margin_metric_name),
            )
        )
        if self.component_name == self.sft_component_name:
            raise LossConfigRefusal(
                f"component_name and sft_component_name are both "
                f"{self.component_name!r}: two components sharing one name "
                f"collapse the coverage denominator"
            )
        if (
            not isinstance(self.margin_metric_ceiling, (int, float))
            or not math.isfinite(self.margin_metric_ceiling)
            or self.margin_metric_ceiling <= 0.0
        ):
            raise LossConfigRefusal(
                f"margin_metric_ceiling={self.margin_metric_ceiling!r}: the "
                f"declared bound on the margin-mean reading must be finite "
                f"and strictly positive; a non-finite bound makes every "
                f"gate comparison against it silently True, and a bounds "
                f"check that examined nothing reads as coverage"
            )

    @property
    def reference_free(self) -> bool:
        """Whether the objective consumes a frozen reference model's scores.

        WHAT IS CLAIMED: CPO is reference-FREE, so this is True; the
        property is the SOURCE that ``semantics()`` derives from, so the
        two statements about this seam cannot drift apart.

        WHAT IS NOT CLAIMED: any gate behaviour attached to this value --
        and, per the class docstring, no verification that the
        uniform-reference assumption this objective relies on actually
        holds.
        """
        return True

    @property
    def required_columns(self) -> tuple[str, ...]:
        """The two batch columns this loss consumes, in a stable order.

        WHAT IS CLAIMED: the tuple names only the two supervision masks;
        CPO reads no reference columns, and its schema must say so.

        WHAT IS NOT CLAIMED: that the columns carry well-typed values;
        shapes and values are measured by ``__call__``.
        """
        return (self.chosen_mask_column, self.rejected_mask_column)

    def semantics(self) -> AlgorithmSemantics:
        """The loss side of the two-declaration seam check.

        WHAT IS CLAIMED: exactly one of the five cross-checked fields --
        ``reference_free`` -- is constrained by a preference objective, and
        it is derived from the ``reference_free`` property rather than
        restated, so the two cannot disagree. The other four abstain with
        ``None``, which the record defines as "this does not constrain that
        seam": a preference objective forms no importance ratio, so it has
        no ratio geometry, no clip bounds and no KL estimator to state, and
        it reads no group, so it has no group size. Declaring a number
        there would be a measured value posing as an unconstrained seam.

        WHAT IS NOT CLAIMED: that reference-freeness is CHECKED against a
        wired reference model. This record states what the loss consumes;
        whether a reference policy is actually present is the role-presence
        layer's question, answered by ``AlgorithmRequirements``.
        """
        return AlgorithmSemantics(reference_free=self.reference_free)

    def declaration(self) -> LossDeclaration:
        """The manifest's statement of what this loss is configured to emit.

        WHAT IS CLAIMED: TWO components, always -- the preference term and
        the SFT term, in that order -- plus accuracy declared exactly as
        DPO declares it (``degenerate=(0.0,)``: pinned-at-zero accuracy
        means the rejected completion's scaled score exceeded the chosen
        one's on every pair) and the margin mean with symmetric tripwire
        bounds and an empty degenerate set.

        WHAT IS NOT CLAIMED: that this declaration is suitable input to
        ``build_objective_gate_context``; it is the run-manifest statement,
        per the ``LossDeclaration`` contract.
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
            components=(self.component_name, self.sft_component_name),
            metrics=(accuracy, margin),
        )

    def __call__(self, forward_fn: ForwardFn, batch: ExperienceBatch) -> LossOutput:
        """Price one batch as the reference-free preference term plus SFT.

        WHAT IS CLAIMED: the returned scalar is the sum of the two declared
        components' contributions: the mean stable ``-log sigmoid`` of the
        beta-scaled sum-score margin, and ``lambda_`` times the chosen-side
        masked-mean NLL (reusing the sums already computed, as DPO reuses
        its chosen-side sums for its auxiliary term).

        WHAT IS NOT CLAIMED: that the dropped reference term was small --
        per the class docstring that assumption is unverifiable here -- or
        that the margin is length-normalised; it is not, matching DPO's
        treatment of sequence scores as sums.
        """
        missing = tuple(name for name in self.required_columns if name not in batch.columns)
        if missing:
            raise BatchRefusal(
                f"CPOLoss requires columns {missing!r}, which are absent; "
                f"this batch carries {tuple(batch.columns)}"
            )
        if len(batch) == 0:
            raise BatchRefusal(
                "CPOLoss got a batch of 0 rows: a mean over zero pairs is "
                "the all([]) shape -- the loss is unmeasurable, and "
                "returning 0.0 would report a perfect preference loss for "
                "a batch containing no preferences"
            )
        chosen_masks = batch.column(self.chosen_mask_column)
        rejected_masks = batch.column(self.rejected_mask_column)

        pairs = _paired_forward_rows(forward_fn, batch)
        pair_losses: list[float] = []
        margins: list[float] = []
        chosen_sum_total = 0.0
        chosen_supervised_total = 0
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
            margin = self.beta * pi_c - self.beta * pi_r
            margins.append(margin)
            pair_losses.append(_neg_log_sigmoid(margin))
            chosen_sum_total += pi_c
            chosen_supervised_total += supervised_c

        preference_contribution = sum(pair_losses) / len(pair_losses)
        sft_contribution = self.lambda_ * (-(chosen_sum_total / chosen_supervised_total))
        correct = sum(1 for margin in margins if margin > 0.0)
        accuracy = correct / len(margins)
        mean_margin = sum(margins) / len(margins)
        loss = preference_contribution + sft_contribution
        components = (
            LossComponent(
                name=self.component_name,
                weight=1.0,
                observed=True,
                contribution=preference_contribution,
            ),
            LossComponent(
                name=self.sft_component_name,
                weight=self.lambda_,
                observed=True,
                contribution=sft_contribution,
            ),
        )
        metrics = (
            MetricObservation(name=self.accuracy_metric_name, value=accuracy),
            MetricObservation(name=self.margin_metric_name, value=mean_margin),
        )
        return LossOutput(loss=loss, components=components, metrics=metrics)
