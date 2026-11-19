"""ARL-G01 process shaping: ``ProcessShaping``, its ``AgenticGRPOLoss`` binding,
the ``agentic_process_grpo`` algorithm and ``RLTrainer._shaping_plane_from_batch``.

WHAT IS CLAIMED: ``ProcessShaping`` refuses a weight that is not a finite real
(a bool is not one), a negative weight, a weight above ``max_weight``, a pair
that would shape nothing (both 0.0) and a ``max_weight`` that is not > 0, and
accepts ``(error_weight=0.1, repetition_weight=0.0)``; ``token_shaping`` is the
hand-computed ``-error_weight*err - repetition_weight*rep`` per response token
and refuses ragged columns or a mask entry that is not the int 0/1;
``AgenticGRPOLoss.required_columns`` appends the two declared mask columns ONLY
behind ``process_shaping``; ``agentic_process_grpo`` carries exactly
``PROCESS_ERROR_WEIGHT``/``PROCESS_REPETITION_WEIGHT`` and nothing else moved
relative to ``agentic_grpo``; the reference kernel prices an all-zero mask batch
exactly as the unshaped objective does and moves the loss by exactly
``row_weight * error_weight`` when one supervised token carries an error at
ratio 1 (``current == old`` logprobs); and ``_shaping_plane_from_batch`` is
``None`` without ``process_shaping`` and otherwise lands each row's values on
that row's OWN shifted response targets (``prompt_len-1 .. prompt_len+resp_len-2``)
and 0.0 everywhere else.

WHAT IS NOT CLAIMED: that the declared weights are tuned, that shaping helps any
benchmark, or that any run has been trained.

The batch/loss/config builders are the ones the neighbouring tests already pin:
``_objective`` (test_agentic_grpo_binding), ``_config`` (test_trainer_group_family)
and ``_batch`` (test_trainer_rollout_rows).
"""

from __future__ import annotations

import dataclasses
from typing import Any

import pytest

torch = pytest.importorskip("torch")

from test_agentic_grpo_binding import _objective  # noqa: E402
from test_trainer_group_family import _config  # noqa: E402
from test_trainer_rollout_rows import _batch  # noqa: E402

from foundationscale.rl.group_policy import (  # noqa: E402
    PROCESS_ERROR_WEIGHT,
    PROCESS_REPETITION_WEIGHT,
)
from foundationscale.rl.group_policy_objectives import (  # noqa: E402
    AgenticGRPOLoss,
    BatchRefusal,
    ProcessShaping,
    prompt_mean_row_weights,
)
from foundationscale.rl.interfaces import LossConfigRefusal  # noqa: E402
from foundationscale.rl.registry import lookup_algorithm, reset_algorithm_registry  # noqa: E402
from foundationscale.rl.trainer import RLTrainer  # noqa: E402

_PAD = 0
_BASE_COLUMNS = ("prompt_ids", "rewards", "loss_mask", "old_logprobs")


def _with_shaping(shaping: ProcessShaping) -> AgenticGRPOLoss:
    """The shipped ``_objective`` with process shaping turned on and NOTHING else
    moved, so every shaper-vs-plain comparison below prices one change only."""
    return dataclasses.replace(_objective(), process_shaping=shaping)


def _kernel_loss(objective: AgenticGRPOLoss, batch: Any, current_rows: Any) -> float:
    """The objective's own pricing kernel over one batch: mask, advantages and
    the ``old_logprobs`` column all come from the batch's columns; the caller
    supplies the CURRENT logprob rows, so feeding the ``old_logprobs`` column
    back in holds every token ratio at exactly 1.0."""
    return float(objective(lambda _batch: current_rows, batch).loss)


# --- 1) ProcessShaping validation -------------------------------------------


@pytest.mark.parametrize(
    ("weights", "match"),
    (
        ({"error_weight": True, "repetition_weight": 0.1}, "a shaping weight must be a finite"),
        (
            {"error_weight": 0.1, "repetition_weight": float("nan")},
            "a shaping weight must be a finite",
        ),
        ({"error_weight": -0.1, "repetition_weight": 0.1}, r"error_weight=-0\.1: it must lie in"),
        ({"error_weight": 1.5, "repetition_weight": 0.1}, r"error_weight=1\.5: it must lie in"),
        (
            {"error_weight": 0.1, "repetition_weight": 0.2, "max_weight": 0.15},
            r"repetition_weight=0\.2",
        ),
        ({"error_weight": 0.0, "repetition_weight": 0.0}, "shapes nothing"),
        ({"error_weight": 0.1, "repetition_weight": 0.0, "max_weight": 0.0}, "bound must be > 0"),
        ({"error_weight": 0.1, "repetition_weight": 0.0, "max_weight": -1.0}, "bound must be > 0"),
    ),
)
def test_process_shaping_refuses_every_weight_outside_the_declared_bounds(
    weights: dict[str, Any], match: str
) -> None:
    with pytest.raises(LossConfigRefusal, match=match):
        ProcessShaping(**weights)


def test_process_shaping_accepts_a_zero_repetition_weight() -> None:
    shaping = ProcessShaping(error_weight=0.1, repetition_weight=0.0)
    assert shaping.columns == ("tool_call_error_mask", "repetition_mask")
    # repetition_weight 0.0 zeroes the repetition term, nothing else.
    assert shaping.token_shaping((0, 1, 0), (0, 0, 1)) == pytest.approx([0.0, -0.1, 0.0])


# --- 2) token_shaping -------------------------------------------------------


def test_token_shaping_is_the_hand_computed_additive_penalty() -> None:
    shaping = ProcessShaping(error_weight=0.1, repetition_weight=0.2)
    # err=[0,1,1,0] priced at 0.1 and rep=[0,0,1,1] at 0.2: [0, -0.1, -0.3, -0.2].
    assert shaping.token_shaping((0, 1, 1, 0), (0, 0, 1, 1)) == pytest.approx(
        [0.0, -0.1, -0.3, -0.2]
    )


def test_token_shaping_refuses_ragged_columns_and_any_non_bit_entry() -> None:
    shaping = ProcessShaping(error_weight=0.1, repetition_weight=0.2)
    with pytest.raises(BatchRefusal, match="one per response token"):
        shaping.token_shaping((0, 1, 0), (0, 0))
    with pytest.raises(BatchRefusal, match="a mask entry must"):
        shaping.token_shaping((0, True), (0, 0))  # a bool is not a mask bit
    with pytest.raises(BatchRefusal, match=r"tool_call_error_mask\[1\]=2"):
        shaping.token_shaping((0, 2), (0, 0))


# --- 3) required_columns ----------------------------------------------------


def test_required_columns_gain_the_two_masks_only_behind_process_shaping() -> None:
    assert _objective().required_columns == _BASE_COLUMNS
    shaped = _with_shaping(ProcessShaping(error_weight=0.1, repetition_weight=0.2))
    assert shaped.required_columns == _BASE_COLUMNS + ("tool_call_error_mask", "repetition_mask")
    renamed = _with_shaping(
        ProcessShaping(
            error_weight=0.1,
            repetition_weight=0.0,
            error_column="tool_err",
            repetition_column="rep",
        )
    )
    assert renamed.required_columns == _BASE_COLUMNS + ("tool_err", "rep")


# --- 4) the agentic_process_grpo binding ------------------------------------


def test_agentic_process_grpo_is_agentic_grpo_plus_the_declared_shaping_weights() -> None:
    reset_algorithm_registry()
    try:
        agentic = lookup_algorithm("agentic_process_grpo")
        plain = lookup_algorithm("agentic_grpo")
    finally:
        reset_algorithm_registry()
    assert agentic.requirements().name == "agentic_process_grpo"
    shaping = agentic._objective.process_shaping
    assert isinstance(shaping, ProcessShaping)
    assert (shaping.error_weight, shaping.repetition_weight) == (
        PROCESS_ERROR_WEIGHT,
        PROCESS_REPETITION_WEIGHT,
    )
    assert shaping.columns == ("tool_call_error_mask", "repetition_mask")
    ahead, base = agentic._objective, plain._objective
    assert base.process_shaping is None
    assert dataclasses.asdict(ahead.advantage_fn) == dataclasses.asdict(base.advantage_fn)
    assert ahead.clip_bounds == base.clip_bounds == (0.8, 1.2)
    assert ahead.reduction_mode == base.reduction_mode == "prompt_mean"
    assert ahead.reduction == base.reduction
    assert ahead.group_size == base.group_size
    assert ahead.required_columns == base.required_columns + shaping.columns


# --- 5) the reference kernel: identical on zeros, delta priced exactly -------


def test_the_reference_kernel_is_unchanged_when_the_masks_are_all_zero() -> None:
    # Asymmetric supervised mass ((3), (2) tokens of reward-advantage +0.5 / -0.5):
    # the surrogate is non-zero, so the equality below is a real comparison.
    logprobs = ((-0.5, -0.5, -0.5), (-0.5, -0.5))
    batch = _batch(
        prompt_ids=("g0", "g0"),
        rewards=(1.0, 0.0),
        loss_mask=((1, 1, 1), (1, 1)),
        old_logprobs=logprobs,
        tool_call_error_mask=((0, 0, 0), (0, 0)),
        repetition_mask=((0, 0, 0), (0, 0)),
    )
    unshaped = _kernel_loss(_objective(), batch, logprobs)
    shaped = _kernel_loss(
        _with_shaping(ProcessShaping(error_weight=0.1, repetition_weight=0.2)), batch, logprobs
    )
    assert unshaped != 0.0
    assert shaped == pytest.approx(unshaped)


def test_one_error_on_one_supervised_token_moves_the_loss_by_row_weight_times_the_weight() -> None:
    logprobs = ((-0.5, -0.5, -0.5), (-0.5, -0.5))
    batch = _batch(
        prompt_ids=("g0", "g0"),
        rewards=(1.0, 0.0),
        loss_mask=((1, 1, 1), (1, 1)),
        old_logprobs=logprobs,
        # One error, on row 1's SUPERVISED token 1; repetition stays silent.
        tool_call_error_mask=((0, 0, 0), (0, 1)),
        repetition_mask=((0, 0, 0), (0, 0)),
    )
    delta = _kernel_loss(
        _with_shaping(ProcessShaping(error_weight=0.1, repetition_weight=0.2)), batch, logprobs
    ) - _kernel_loss(_objective(), batch, logprobs)
    # ratio == 1 at that token and its mask is 1, so the -0.1 advantage shift is
    # lost at exactly row 1's own prompt-mean row weight (the declaration's single
    # component weight is 1.0): delta_loss = +row_weight * error_weight.
    row_weight = prompt_mean_row_weights(["g0", "g0"], [3, 2])[1]
    assert delta > 0.0
    assert delta == pytest.approx(row_weight * 0.1)


# --- 6) trainer._shaping_plane_from_batch -----------------------------------


def _plane_batch() -> Any:
    # Prompt widths [2, 3] and response widths [3, 2]: row 0's response targets
    # are t = 1..3, row 1's are t = 2..3 and no other position is addressable.
    return _batch(
        prompt_ids=("g0", "g0"),
        prompt_token_ids=((1, 2), (3, 4, 5)),
        response_ids=((10, 11, 12), (20, 21)),
        loss_mask=((1, 1, 1), (1, 1)),
        reward=(1.0, 0.0),
        tool_call_error_mask=((1, 0, 1), (0, 1)),
        repetition_mask=((0, 0, 1), (0, 0)),
    )


def test_shaping_plane_is_none_for_an_objective_without_process_shaping() -> None:
    trainer = RLTrainer(_config("agentic_process_grpo"))
    assert (
        trainer._shaping_plane_from_batch(
            _plane_batch(), _objective(), kept_row_indices=[0, 1], response_mask=torch.zeros(2, 4)
        )
        is None
    )


def test_shaping_plane_lands_each_row_on_its_own_shifted_response_positions() -> None:
    trainer = RLTrainer(_config("agentic_process_grpo"))
    objective = trainer._resolve_objective()
    assert objective.process_shaping is not None
    assert (
        objective.process_shaping.error_weight,
        objective.process_shaping.repetition_weight,
    ) == (
        PROCESS_ERROR_WEIGHT,
        PROCESS_REPETITION_WEIGHT,
    )
    batch = _plane_batch()
    _kept_sequences, response_mask, _attention, _rewards, _groups, kept = (
        trainer._planes_from_batch(batch, device="cpu", pad_token_id=_PAD)
    )
    # width = max_len - 1 = 4; row 0 (prompt 2) targets t = 1..3, row 1 (prompt 3)
    # targets t = 2..3 -- the same shifted alignment the plane must obey.
    assert response_mask.tolist() == [[0.0, 1.0, 1.0, 1.0], [0.0, 0.0, 1.0, 1.0]]
    plane = trainer._shaping_plane_from_batch(
        batch, objective, kept_row_indices=kept, response_mask=response_mask
    )
    assert plane is not None
    assert plane.shape == response_mask.shape
    # row 0: err=(1,0,1), rep=(0,0,1) at weights 0.1/0.1 -> [-0.1, 0.0, -0.2] on
    # t = 1..3 (t = 0 left at 0.0); row 1: err=(0,1), rep=(0,0) -> [0.0, -0.1]
    # on t = 2..3 (t = 0, 1 left at 0.0).
    row = plane.tolist()
    assert row[0] == pytest.approx([0.0, -0.1, 0.0, -0.2])
    assert row[1] == pytest.approx([0.0, 0.0, 0.0, -0.1])
