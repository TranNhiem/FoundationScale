"""Tests for :mod:`foundationscale.vla.chunking`, the temporal contract of spec S2.

Every refusal listed in the spec gets its own test that asserts the exception type and a
message substring naming the offending delta tuple; ``valid_anchor_indices`` is checked for
exact ranges in both pad modes, since the anchor set is the contract the sampler relies on.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest

from foundationscale.vla.chunking import ChunkError, ChunkSpec, valid_anchor_indices

STATE_DELTA: tuple[int, ...] = (0,)
ACTION_DELTA: tuple[int, ...] = (0, 1, 2, 3)
VIDEO_DELTA: tuple[int, ...] = (0,)


def _spec(
    state_delta: tuple[int, ...] = STATE_DELTA,
    action_delta: tuple[int, ...] = ACTION_DELTA,
    video_delta: tuple[int, ...] = VIDEO_DELTA,
) -> ChunkSpec:
    """A ChunkSpec with overridable delta tuples."""
    return ChunkSpec(state_delta=state_delta, action_delta=action_delta, video_delta=video_delta)


def _refusal(expected: type[Exception], fn: Callable[[], object], *needles: str) -> Exception:
    """Run ``fn`` and assert it raises ``expected`` naming every ``needle`` in the message."""
    with pytest.raises(expected) as excinfo:
        fn()
    message = str(excinfo.value)
    for needle in needles:
        assert needle in message, f"expected {needle!r} in error message {message!r}"
    return excinfo.value


def test_chunk_error_is_a_value_error() -> None:
    assert issubclass(ChunkError, ValueError)


def test_chunk_spec_accepts_the_documented_horizons() -> None:
    spec = ChunkSpec(state_delta=(0,), action_delta=tuple(range(16)), video_delta=(0,))
    assert spec.action_horizon == 16
    assert spec.state_delta == (0,)
    assert spec.video_delta == (0,)


def test_chunk_spec_refuses_an_empty_state_delta() -> None:
    _refusal(ChunkError, lambda: _spec(state_delta=()), "state_delta", "empty")


def test_chunk_spec_refuses_an_empty_action_delta() -> None:
    _refusal(ChunkError, lambda: _spec(action_delta=()), "action_delta", "empty")


def test_chunk_spec_refuses_an_empty_video_delta() -> None:
    _refusal(ChunkError, lambda: _spec(video_delta=()), "video_delta", "empty")


def test_chunk_spec_refuses_duplicate_state_delta() -> None:
    _refusal(ChunkError, lambda: _spec(state_delta=(0, 2, 2)), "state_delta", "duplicate")


def test_chunk_spec_refuses_duplicate_action_delta() -> None:
    _refusal(ChunkError, lambda: _spec(action_delta=(0, 1, 1, 2)), "action_delta", "duplicate")


def test_chunk_spec_refuses_duplicate_video_delta() -> None:
    _refusal(ChunkError, lambda: _spec(video_delta=(0, 0)), "video_delta", "duplicate")


def test_chunk_spec_refuses_non_increasing_state_delta() -> None:
    _refusal(ChunkError, lambda: _spec(state_delta=(2, 1, 0)), "state_delta", "increasing")


def test_chunk_spec_refuses_non_increasing_action_delta() -> None:
    _refusal(ChunkError, lambda: _spec(action_delta=(2, 1, 0)), "action_delta", "increasing")


def test_chunk_spec_refuses_non_increasing_video_delta() -> None:
    _refusal(ChunkError, lambda: _spec(video_delta=(1, 0)), "video_delta", "increasing")


def test_chunk_spec_refuses_negative_action_delta() -> None:
    _refusal(ChunkError, lambda: _spec(action_delta=(-1, 0, 1)), "action_delta", "negative")


def test_chunk_spec_refuses_gaps_in_action_delta() -> None:
    _refusal(ChunkError, lambda: _spec(action_delta=(0, 1, 3)), "action_delta", "contiguous")


def test_chunk_spec_allows_gaps_outside_action_delta() -> None:
    spec = _spec(state_delta=(0, 2), action_delta=(0, 1), video_delta=(0, 3))
    assert spec.state_delta == (0, 2)
    assert spec.video_delta == (0, 3)
    assert spec.action_horizon == 2


def test_valid_anchor_indices_refuse_mode_exact_range() -> None:
    result = valid_anchor_indices(10, _spec(), pad="refuse")
    assert result == range(0, 7)
    assert (result.start, result.stop, result.step) == (0, 7, 1)
    assert tuple(result) == (0, 1, 2, 3, 4, 5, 6)


def test_valid_anchor_indices_refuse_mode_uses_the_largest_delta() -> None:
    spec = _spec(state_delta=(0, 5), action_delta=(0, 1), video_delta=(0,))
    result = valid_anchor_indices(12, spec, pad="refuse")
    assert result == range(0, 7)
    assert tuple(result) == (0, 1, 2, 3, 4, 5, 6)


def test_valid_anchor_indices_refuse_mode_keeps_the_boundary_anchor() -> None:
    result = valid_anchor_indices(4, _spec(), pad="refuse")
    assert result == range(0, 1)
    assert tuple(result) == (0,)


def test_valid_anchor_indices_refuse_mode_is_empty_for_a_short_episode() -> None:
    result = valid_anchor_indices(3, _spec(), pad="refuse")
    assert result == range(0, 0)
    assert tuple(result) == ()


def test_valid_anchor_indices_edge_mode_exact_range() -> None:
    result = valid_anchor_indices(10, _spec(), pad="edge")
    assert result == range(0, 10)
    assert (result.start, result.stop, result.step) == (0, 10, 1)
    assert tuple(result) == tuple(range(10))


def test_valid_anchor_indices_edge_mode_covers_a_short_episode() -> None:
    result = valid_anchor_indices(3, _spec(), pad="edge")
    assert result == range(0, 3)
    assert tuple(result) == (0, 1, 2)


def test_valid_anchor_indices_edge_mode_covers_a_single_frame_episode() -> None:
    assert valid_anchor_indices(1, _spec(), pad="edge") == range(0, 1)


def test_valid_anchor_indices_refuses_a_call_without_a_pad_mode() -> None:
    _refusal(TypeError, lambda: valid_anchor_indices(10, _spec()), "pad")


def test_valid_anchor_indices_refuses_an_unknown_pad_mode() -> None:
    _refusal(ValueError, lambda: valid_anchor_indices(10, _spec(), pad="zero"), "pad")
