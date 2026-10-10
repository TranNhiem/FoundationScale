"""The temporal contract: which frames of an episode one training sample draws.

GR00T names those frames as ``delta_indices``: offsets from an anchor frame, one
tuple per group -- the state frames fed to the policy, the action frames it must
predict (the horizon ``H``), and the video frames decoded for it. This module
holds that contract as :class:`ChunkSpec` and answers the one question every
sampler must answer the same way: which anchors of an episode may be drawn at
all, :func:`valid_anchor_indices`.

Nothing here is defaulted and nothing is padded silently. A spec with an empty
tuple, a duplicated offset, offsets that do not increase strictly, a negative
action offset, or a gap in ``action_delta`` is refused with :class:`ChunkError`
naming the tuple, the index and the observed versus expected value: a gap would
hand the policy an action horizon GR00T's ``PolicyHorizonSpec`` refuses, and a
duplicate would read one frame twice into one vector. ``state_delta`` and
``video_delta`` may be negative -- a policy may read history frames -- but
``action_delta`` may not, because an action before the anchor is one the policy
is never asked to predict.

The pad mode has no default either: the caller declares ``"refuse"`` (only
anchors whose every offset lands inside ``[0, length)``, so nothing is invented)
or ``"edge"`` (every anchor in ``[0, length)``, out-of-range offsets clamped to
the episode's edge, and the sample carrying an ``action_valid`` mask so the loss
can exclude the padded steps -- GR00T and openpi pad the tail the same way).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypeGuard

__all__ = ["ChunkError", "ChunkSpec", "valid_anchor_indices"]

# The two pad modes a caller may declare; there is no default among them.
_PAD_MODES: tuple[str, ...] = ("refuse", "edge")


class ChunkError(ValueError):
    """A chunk spec cannot name frames, or an anchor request cannot describe an episode."""


@dataclass(frozen=True)
class ChunkSpec:
    """Frame offsets from the anchor, per group, as GR00T's ``delta_indices`` names them.

    ``state_delta`` are the state frames fed to the policy, ``action_delta`` the
    action frames it must predict -- its horizon ``H`` is
    :attr:`action_horizon` -- and ``video_delta`` the video frames decoded for
    it. Offsets are relative to the anchor frame: ``0`` is the anchor itself and
    a positive offset a later frame. Every tuple is strictly increasing and free
    of duplicates, so each offset names its own frame exactly once.
    """

    state_delta: tuple[int, ...]
    action_delta: tuple[int, ...]
    video_delta: tuple[int, ...]

    def __post_init__(self) -> None:
        """Refuse a delta tuple that cannot name frames, and an action horizon GR00T refuses.

        Raises :class:`ChunkError` for an empty tuple, a non-int offset, a
        duplicate, offsets that do not increase strictly, a negative action
        offset, or a gap in ``action_delta`` -- each of those would silently
        reshape the vector the policy consumes or the horizon it is scored on.
        """
        _check_offsets("state_delta", self.state_delta)
        _check_offsets("action_delta", self.action_delta)
        _check_offsets("video_delta", self.video_delta)
        _check_action_delta(self.action_delta)

    @property
    def action_horizon(self) -> int:
        """``H``: how many action steps one sample carries."""
        return len(self.action_delta)


# -- fail-closed validation ----------------------------------------------------------


def _is_int(value: object) -> TypeGuard[int]:
    return isinstance(value, int) and not isinstance(value, bool)


def _check_offsets(name: str, values: tuple[int, ...]) -> None:
    """Refuse a delta tuple that does not name frames exactly once, in increasing order."""
    if not isinstance(values, tuple):
        raise ChunkError(
            f"ChunkSpec.{name} is {values!r} ({type(values).__name__}); expected a tuple of ints"
        )
    if not values:
        raise ChunkError(f"ChunkSpec.{name} is empty; expected at least one frame offset")
    for index, value in enumerate(values):
        if not _is_int(value):
            raise ChunkError(
                f"ChunkSpec.{name}[{index}] is {value!r} ({type(value).__name__}); "
                "expected an int frame offset"
            )
    for index in range(1, len(values)):
        previous = values[index - 1]
        current = values[index]
        if current == previous:
            raise ChunkError(
                f"ChunkSpec.{name}[{index}] is {current}, the same as index {index - 1}; "
                f"expected each offset exactly once in {values} "
                "(a duplicate would read one frame twice into one vector)"
            )
        if current < previous:
            raise ChunkError(
                f"ChunkSpec.{name}[{index}] is {current}, less than index {index - 1} value "
                f"{previous}; expected strictly increasing offsets in {values}"
            )


def _check_action_delta(values: tuple[int, ...]) -> None:
    """Refuse a negative or gapped action horizon: GR00T's PolicyHorizonSpec refuses gaps."""
    for index, value in enumerate(values):
        if value < 0:
            raise ChunkError(
                f"ChunkSpec.action_delta[{index}] is {value} (negative); expected >= 0 -- "
                "an action before "
                "the anchor is one the policy is never asked to predict"
            )
    for index in range(1, len(values)):
        previous = values[index - 1]
        current = values[index]
        if current != previous + 1:
            raise ChunkError(
                f"ChunkSpec.action_delta is {values}; expected contiguous offsets stepping by 1 "
                f"(GR00T's PolicyHorizonSpec refuses gaps): index {index} is {current}, "
                f"{current - previous} after index {index - 1} value {previous}"
            )


# -- anchors -------------------------------------------------------------------------


def valid_anchor_indices(length: int, spec: ChunkSpec, *, pad: Literal["refuse", "edge"]) -> range:
    """The anchors of an episode of ``length`` frames that ``spec`` can draw from.

    ``pad="refuse"`` returns only the anchors ``t`` whose every offset lands
    inside ``[0, length)``: nothing is padded and nothing is invented, so the
    range starts ``-min(delta)`` after the beginning when an offset is negative
    and stops ``max(delta)`` frames before the end. ``pad="edge"`` returns every
    ``t`` in ``[0, length)``: an out-of-range offset clamps to the episode's
    first or last frame, and the sample built from it carries an
    ``action_valid`` mask marking the padded steps so the loss can exclude them
    (GR00T and openpi pad the tail the same way).

    There is no default pad mode: the caller declares how an out-of-range frame
    is handled. A ``length`` that is not an integer > 0 (an episode with no
    frames has no anchor) or a ``pad`` other than ``"refuse"``/``"edge"`` is
    refused with :class:`ChunkError` naming the value -- silently picking a pad
    mode would change which steps the loss counts.
    """
    if not _is_int(length) or length <= 0:
        raise ChunkError(
            f"valid_anchor_indices: length is {length!r}; expected an integer > 0 "
            "(an episode with no frames has no anchor)"
        )
    if pad not in _PAD_MODES:
        raise ChunkError(
            f"valid_anchor_indices: pad is {pad!r}; expected one of {list(_PAD_MODES)} -- there "
            "is no default, the caller declares how an out-of-range frame is handled"
        )
    if pad == "edge":
        return range(0, length)
    deltas = (*spec.state_delta, *spec.action_delta, *spec.video_delta)
    start = max(0, -min(deltas))
    stop = length - max(deltas)
    if stop <= start:
        return range(0, 0)
    return range(start, stop)
