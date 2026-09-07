"""Phase 3 stage-1 contracts: the batch, the loss, and the gate bridge.

Design section 9 stage 1: ``ExperienceBatch``, ``LossFn`` with supervised
fine-tuning only, and the objective-gate bridge. ``Algorithm``,
``RolloutSource``, ``AdvantageFn``, ``PolicyPair``, ``WeightSync`` and
``StepReport`` are later stages and are deliberately absent -- shipping them
here would be wrong, not ahead.

No torch at module scope: the gate plane is imported by torch-free host
tooling, so every tensor interaction in this file is duck-typed and no
accelerator is assumed.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from foundationscale.gates.objective_gates import (
    LossComponent,
    ObjectiveGateContext,
    ValueProvenance,
)

__all__ = (
    "BatchRefusal",
    "ExperienceBatch",
    "ForwardFn",
    "LossFn",
    "LossOutput",
    "SFTLoss",
    "SupervisionRefusal",
    "build_objective_gate_context",
)


class BatchRefusal(ValueError):
    # Raised when a batch violates its declared schema or a caller's value
    # breaks the stated contract. Fail closed; never pad, never coerce.
    pass


class SupervisionRefusal(ValueError):
    # Raised when the supervision mask selects nothing. The loss is
    # UNMEASURABLE on such a batch; returning 0.0 would report a perfect
    # loss for a batch that taught the model nothing.
    pass


@dataclass(frozen=True)
class ExperienceBatch:
    """Typed columnar batch flowing between rollout, loss, and the gates.

    The schema is declared by the caller: ``required`` names the columns the
    consuming algorithm needs, and construction refuses when any is absent.
    Every column must carry exactly one value per batch row; a misaligned
    column is refused with a message naming the column and both row counts.
    Slicing is row-aligned: ``batch[2:5]`` takes the same rows from every
    column and re-validates. There are no implicit device semantics and no
    engine-specific opaque payload -- a column is a plain sequence of
    per-row values.

    Phase 1 pattern 4's validation discipline is adopted because TypedDict
    alone enforces nothing. This container's semantics are FoundationScale's
    own (design section 3.7); Phase 1 unknown 6 -- the internals of the
    reference implementation's batch dict -- stays open and is not closed
    here by inheritance or imitation.
    """

    columns: Mapping[str, Sequence[Any]]
    required: tuple[str, ...] = ()
    _row_count: int = field(init=False, repr=False)

    def __post_init__(self) -> None:
        required = tuple(self.required)
        missing = tuple(name for name in required if name not in self.columns)
        if missing:
            raise BatchRefusal(
                f"ExperienceBatch is missing {len(missing)} of "
                f"{len(required)} declared required column(s): {missing}; "
                f"present columns: {tuple(self.columns)}"
            )
        frozen: dict[str, Sequence[Any]] = {}
        row_count: int | None = None
        reference: str | None = None
        for name, column in self.columns.items():
            try:
                length = len(column)
            except TypeError as exc:
                raise BatchRefusal(
                    f"column {name!r} has no row count ({type(column).__name__}); "
                    f"a column must be a sized sequence of per-row values"
                ) from exc
            if row_count is None:
                row_count = length
                reference = name
            elif length != row_count:
                raise BatchRefusal(
                    f"column {name!r} has {length} rows but column "
                    f"{reference!r} has {row_count} rows; every column must "
                    f"carry one value per batch row"
                )
            frozen[name] = column if isinstance(column, tuple) else tuple(column)
        object.__setattr__(self, "columns", frozen)
        object.__setattr__(self, "required", required)
        object.__setattr__(self, "_row_count", row_count if row_count is not None else 0)

    def __len__(self) -> int:
        return self._row_count

    def column(self, name: str) -> Sequence[Any]:
        try:
            return self.columns[name]
        except KeyError:
            raise KeyError(
                f"no column {name!r}; this batch carries {tuple(self.columns)}"
            ) from None

    def __getitem__(self, index: slice) -> ExperienceBatch:
        if not isinstance(index, slice):
            raise TypeError(
                f"ExperienceBatch slicing is row-aligned; expected a slice, got "
                f"{type(index).__name__}. Use .column(name) for column access"
            )
        return ExperienceBatch(
            columns={name: column[index] for name, column in self.columns.items()},
            required=self.required,
        )


@dataclass(frozen=True)
class LossOutput:
    """Return type of a :class:`LossFn`: the scalar plus named components.

    ``loss`` is the scalar the optimizer sees. ``components`` carries the
    same quantity decomposed into real :class:`LossComponent` entries so
    ``LossComponentCoverageGate`` reads it without adaptation. An unmeasured
    contribution is ``None``, never ``0.0`` -- absent is not zero, and a
    measured ``0.0`` is a real and different observation.
    """

    loss: float
    components: tuple[LossComponent, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "components", tuple(self.components))


# The training loop's opaque forward callable. The loss invokes it with the
# batch and treats the return as duck-typed; the loss never reaches into the
# model. Each LossFn documents the exact return contract it requires.
ForwardFn = Callable[[ExperienceBatch], Any]


class LossFn(Protocol):
    """Maps one forward pass over a typed batch to a scalar plus components.

    Design section 3.2: one level wider and SFT forces a no-op path through
    a reward-shaped interface; one level narrower and the objective gates go
    blind. The call signature is FoundationScale's own; the reference
    implementation's signature (Phase 1 unknown 1) is unverified and no
    equivalence with it is claimed.
    """

    def __call__(self, forward_fn: ForwardFn, batch: ExperienceBatch) -> LossOutput: ...


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


def build_objective_gate_context(
    loss: LossOutput,
    *,
    objective: ValueProvenance | None,
    declared_components: Sequence[str],
    current_hparams: Mapping[str, Any],
    step0_fingerprint: str | None,
    step0_hparams: Mapping[str, Any] | None,
    origin: str = "<rl-loss>",
) -> ObjectiveGateContext:
    """Assemble the gate context for one measured step (design section 7, item 1).

    The caller supplies the run's declared objective information; the bridge
    passes absence through as absence. A ``None`` objective stays ``None``
    and the declared component names are taken as given -- nothing here is
    defaulted to something plausible, because a gate comparing what is in
    force against what is in force is vacuous.

    ``step0_fingerprint`` and ``step0_hparams`` are keyword-required and have NO
    default, so a caller with no step-0 record must say so by passing ``None``.
    This was measured: with the fingerprint absent, ``objective.hparam_drift``
    returns FAIL over a ``STEP_ZERO`` sweep, which is the gate behaving
    correctly -- hyperparameters with nothing recorded to compare against are
    unfalsifiable. Defaulting the parameters would have made that blocking
    verdict a property of this bridge rather than a stated property of the call
    site, so the loop is required to state which it is.

    Both are passed through verbatim and neither is derived here. The gate
    compares a fingerprint of the LIVE hyperparameters against the one the run
    RECORDED at step 0, so re-deriving the recorded side at call time from
    whatever the caller has to hand would compare what is in force against what
    is in force -- vacuous, and precisely the reading the gate exists to refuse.
    Producing the step-0 fingerprint belongs to the loop/manifest seam, where
    the value is written once and read back.
    """
    # Reward fields stay at their defaults: reward-bearing algorithms are a
    # later stage (design section 9), and `uses_rewards=False` with no stats is
    # the state the reward gate skips on, not a claim that rewards were checked.
    return ObjectiveGateContext(
        objective=objective,
        declared_components=tuple(declared_components),
        components=loss.components,
        uses_rewards=False,
        step0_fingerprint=step0_fingerprint,
        step0_hparams=step0_hparams,
        current_hparams=current_hparams,
        origin=origin,
    )
