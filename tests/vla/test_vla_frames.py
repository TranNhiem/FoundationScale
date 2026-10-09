"""Tests for :mod:`foundationscale.vla.frames`, decoding frames at exact frame indices.

A fake decoder returning solid-colour frames is injected through ``DECODERS`` and the
``_decoder`` resolver, so nothing is really decoded on CPU; every refusal listed in spec S2
gets its own test asserting the exception type and a message substring naming the file,
the counts or the index.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

import foundationscale.vla.frames as frames_module
from foundationscale.vla.frames import DECODERS, FrameDecodeError, decode_frames

HEIGHT = 4
WIDTH = 5
VIDEO_NAME = "clip.mp4"


def _solid(index: int) -> np.ndarray:
    """A solid-colour uint8 RGB frame whose pixel value identifies ``index``."""
    return np.full((HEIGHT, WIDTH, 3), (index * 17) % 256, dtype=np.uint8)


class _FakeDecoder:
    """Stand-in decoder: ``count`` solid-colour frames and a record of every call."""

    def __init__(self, count: int) -> None:
        self.count = count
        self.calls: list[Path] = []

    def __call__(self, path: Path) -> list[np.ndarray]:
        self.calls.append(Path(path))
        return [_solid(index) for index in range(self.count)]


def _write_video(root: Path, name: str = VIDEO_NAME, size: int = 14) -> Path:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"0" * size)
    return path


def _inject(monkeypatch: pytest.MonkeyPatch, decoder: Callable[[Path], list[np.ndarray]]) -> None:
    """Inject a stand-in decoder exactly as the spec allows: DECODERS plus the resolver."""
    monkeypatch.setitem(DECODERS, "pyav", decoder)
    monkeypatch.setattr(frames_module, "_decoder", lambda *args, **kwargs: DECODERS["pyav"])


def _refusal(expected: type[Exception], fn: Callable[[], object], *needles: str) -> Exception:
    """Run ``fn`` and assert it raises ``expected`` naming every ``needle`` in the message."""
    with pytest.raises(expected) as excinfo:
        fn()
    message = str(excinfo.value)
    for needle in needles:
        assert needle in message, f"expected {needle!r} in error message {message!r}"
    return excinfo.value


def test_decode_frames_returns_the_requested_frames_in_request_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _write_video(tmp_path)
    fake = _FakeDecoder(count=6)
    _inject(monkeypatch, fake)
    decoded = decode_frames(path, [3, 1, 4], expected_length=6)
    assert len(decoded) == 3
    for frame, index in zip(decoded, [3, 1, 4], strict=True):
        assert frame.dtype == np.uint8
        assert frame.shape == (HEIGHT, WIDTH, 3)
        np.testing.assert_array_equal(frame, _solid(index))
    assert fake.calls == [path]


def test_decode_frames_returns_duplicate_indices_in_request_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _write_video(tmp_path)
    fake = _FakeDecoder(count=5)
    _inject(monkeypatch, fake)
    decoded = decode_frames(path, [2, 2, 0], expected_length=5)
    assert len(decoded) == 3
    np.testing.assert_array_equal(decoded[0], _solid(2))
    np.testing.assert_array_equal(decoded[1], _solid(2))
    np.testing.assert_array_equal(decoded[2], _solid(0))


def test_decode_frames_decodes_the_stream_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _write_video(tmp_path)
    fake = _FakeDecoder(count=5)
    _inject(monkeypatch, fake)
    decode_frames(path, [0, 1, 2, 4], expected_length=5)
    assert fake.calls == [path]


def test_decode_frames_refuses_a_missing_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "gone.mp4"
    fake = _FakeDecoder(count=3)
    _inject(monkeypatch, fake)
    _refusal(
        FrameDecodeError,
        lambda: decode_frames(path, [0], expected_length=3),
        "gone.mp4",
        "missing",
    )


def test_decode_frames_refuses_a_video_shorter_than_the_episode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _write_video(tmp_path)
    fake = _FakeDecoder(count=5)
    _inject(monkeypatch, fake)
    _refusal(
        FrameDecodeError,
        lambda: decode_frames(path, [0], expected_length=11),
        VIDEO_NAME,
        "5",
        "11",
    )


def test_decode_frames_refuses_a_video_longer_than_the_episode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _write_video(tmp_path)
    fake = _FakeDecoder(count=11)
    _inject(monkeypatch, fake)
    _refusal(
        FrameDecodeError,
        lambda: decode_frames(path, [0], expected_length=5),
        VIDEO_NAME,
        "11",
        "5",
    )


def test_decode_frames_refuses_an_index_beyond_the_episode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _write_video(tmp_path)
    fake = _FakeDecoder(count=4)
    _inject(monkeypatch, fake)
    _refusal(
        FrameDecodeError,
        lambda: decode_frames(path, [0, 7], expected_length=4),
        VIDEO_NAME,
        "7",
    )


def test_decode_frames_refuses_a_negative_index(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _write_video(tmp_path)
    fake = _FakeDecoder(count=4)
    _inject(monkeypatch, fake)
    _refusal(
        FrameDecodeError,
        lambda: decode_frames(path, [-1], expected_length=4),
        VIDEO_NAME,
        "-1",
    )


def test_decode_frames_refuses_when_no_decoder_is_importable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _write_video(tmp_path)

    def _no_decoder(*args: object, **kwargs: object) -> list[np.ndarray]:
        raise ImportError("No module named 'av'")

    monkeypatch.setattr(frames_module, "_decoder", _no_decoder)
    _refusal(
        FrameDecodeError,
        lambda: decode_frames(path, [0], expected_length=3),
        "pip install av",
    )


def test_decode_frames_cache_hit_avoids_a_second_decode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _write_video(tmp_path)
    fake = _FakeDecoder(count=6)
    _inject(monkeypatch, fake)
    first = decode_frames(path, [1, 4], expected_length=6)
    second = decode_frames(path, [4, 1, 1], expected_length=6)
    assert fake.calls == [path]
    assert len(second) == 3
    np.testing.assert_array_equal(second[0], first[1])
    np.testing.assert_array_equal(second[1], first[0])
    np.testing.assert_array_equal(second[2], first[0])


def test_decode_frames_cache_is_keyed_by_mtime_and_size(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _write_video(tmp_path)
    fake = _FakeDecoder(count=4)
    _inject(monkeypatch, fake)
    decode_frames(path, [0], expected_length=4)
    path.write_bytes(b"rewritten video bytes")
    decode_frames(path, [0], expected_length=4)
    assert fake.calls == [path, path]
