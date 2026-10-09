"""Tests for :mod:`foundationscale.vla.frames`, decoding frames at exact frame indices.

A fake decoder returning solid-colour frames is injected through ``DECODERS`` and the
``_decoder`` resolver, so nothing is really decoded on CPU; every refusal listed in spec S2
gets its own test asserting the exception type and a message substring naming the file,
the counts or the index.
"""

from __future__ import annotations

import sys
import types
from collections.abc import Callable
from pathlib import Path
from typing import Any

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


# -- the real PyAV decoder, through a stand-in ``av`` module ---------------------------
# PyAV is optional (not a declared dependency), so the decoder's own logic is pinned
# against a fake module in sys.modules, the way tests/train/test_video.py pins
# foundationscale/video.py's decoders. Real AV1 decoding is measured on hardware.


class _FakeFFmpegError(Exception):
    pass


class _FakeFrame:
    def __init__(self, index: int) -> None:
        self.index = index

    def to_ndarray(self, *, format: str) -> np.ndarray:  # noqa: A002 -- PyAV's keyword
        assert format == "rgb24"
        return _solid(self.index)


class _FakeContainer:
    def __init__(self, count: int, *, streams: int = 1, fail_at: int | None = None) -> None:
        self.count = count
        self.fail_at = fail_at
        self.streams = type("Streams", (), {"video": [object()] * streams})()

    def __enter__(self) -> _FakeContainer:
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def decode(self, *, video: int) -> Any:
        assert video == 0
        for index in range(self.count):
            if self.fail_at is not None and index == self.fail_at:
                raise _FakeFFmpegError("corrupt packet")
            yield _FakeFrame(index)


def _fake_av(monkeypatch: pytest.MonkeyPatch, opener: Callable[[str], Any]) -> None:
    module = types.ModuleType("av")
    module.open = opener  # type: ignore[attr-defined]
    module.error = types.SimpleNamespace(FFmpegError=_FakeFFmpegError)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "av", module)


def test_pyav_decoder_returns_every_frame_as_contiguous_uint8_rgb(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _fake_av(monkeypatch, lambda path: _FakeContainer(4))
    frames = frames_module._decode_all_pyav(_write_video(tmp_path))
    assert len(frames) == 4
    for index, frame in enumerate(frames):
        assert frame.dtype == np.uint8 and frame.flags["C_CONTIGUOUS"]
        np.testing.assert_array_equal(frame, _solid(index))


def test_pyav_decoder_refuses_a_file_it_cannot_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _opener(path: str) -> Any:
        raise _FakeFFmpegError("moov atom not found")

    _fake_av(monkeypatch, _opener)
    _refusal(
        FrameDecodeError,
        lambda: frames_module._decode_all_pyav(_write_video(tmp_path)),
        "could not open it",
        "moov atom not found",
    )


def test_pyav_decoder_refuses_a_container_with_no_video_stream(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _fake_av(monkeypatch, lambda path: _FakeContainer(4, streams=0))
    _refusal(
        FrameDecodeError,
        lambda: frames_module._decode_all_pyav(_write_video(tmp_path)),
        "no video stream",
    )


def test_pyav_decoder_refuses_a_stream_that_fails_mid_decode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _fake_av(monkeypatch, lambda path: _FakeContainer(4, fail_at=2))
    _refusal(
        FrameDecodeError,
        lambda: frames_module._decode_all_pyav(_write_video(tmp_path)),
        "could not decode it",
        "corrupt packet",
    )


def test_decode_error_types_finds_both_pyav_error_names() -> None:
    class _AVError(Exception):
        pass

    av = types.SimpleNamespace(error=types.SimpleNamespace(FFmpegError=_FakeFFmpegError))
    av.AVError = _AVError  # type: ignore[attr-defined]
    found = frames_module._decode_error_types(av)
    assert _FakeFFmpegError in found and _AVError in found and OSError in found
    assert frames_module._decode_error_types(types.SimpleNamespace()) == (
        OSError,
        ValueError,
        RuntimeError,
    )


def test_decode_frames_without_av_installed_names_the_install(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(sys.modules, "av", None)  # `import av` now raises ImportError
    video = _write_video(tmp_path, size=23)  # unique size: never a cache hit
    _refusal(
        FrameDecodeError,
        lambda: decode_frames(video, [0], expected_length=1),
        "no decoder is importable",
        "pip install av",
    )


def test_resolver_uses_a_lone_stand_in_and_refuses_an_ambiguous_table(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lone = _FakeDecoder(1)
    monkeypatch.setattr(frames_module, "DECODERS", {"stand-in": lone})
    assert frames_module._decoder() is lone
    monkeypatch.setattr(frames_module, "DECODERS", {"a": lone, "b": lone})
    with pytest.raises(ImportError, match="no decoder 'pyav' is registered"):
        frames_module._decoder()


@pytest.mark.parametrize("length", [0, True, 2.0])
def test_decode_frames_refuses_a_bad_expected_length(tmp_path: Path, length: Any) -> None:
    _refusal(
        FrameDecodeError,
        lambda: decode_frames(_write_video(tmp_path), [0], expected_length=length),
        "expected_length",
    )


@pytest.mark.parametrize("index", [1.5, True, "0"])
def test_decode_frames_refuses_a_non_integer_index(tmp_path: Path, index: Any) -> None:
    _refusal(
        FrameDecodeError,
        lambda: decode_frames(_write_video(tmp_path), [index], expected_length=3),
        "expected an integer frame index",
    )


def test_cached_frames_are_read_only(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _inject(monkeypatch, _FakeDecoder(2))
    first = decode_frames(_write_video(tmp_path, size=31), [0], expected_length=2)[0]
    with pytest.raises(ValueError, match="read-only"):
        first[0, 0, 0] = 7
