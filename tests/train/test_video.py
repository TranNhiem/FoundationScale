"""CPU legs for foundationscale.video: the declared frame budget and its cache.

Decoding is driven through an injected stand-in backend so these legs measure the
module's own contract -- centred-uniform instants, one decode per clip and budget,
a refusal that names every backend tried -- on any runner, with or without
ffmpeg or a decoder library installed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from foundationscale import video
from foundationscale.video import FrameBudget, VideoDecodeError, sample_times, video_frame_paths


def test_sample_times_are_centred_uniform_over_the_clip() -> None:
    assert sample_times(8.0, 4) == [1.0, 3.0, 5.0, 7.0]
    assert sample_times(1.0, 1) == [0.5]


def test_sample_times_refuse_a_non_positive_duration() -> None:
    with pytest.raises(VideoDecodeError, match="not positive"):
        sample_times(0.0, 16)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"frames": 0}, "frames must be >= 1"),
        ({"frames": 4, "sampling": "random"}, "sampling"),
        ({"frames": 4, "max_side": 8}, "max_side"),
    ],
)
def test_frame_budget_refuses_an_undeclarable_budget(kwargs: dict[str, Any], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        FrameBudget(**kwargs)


def test_frame_budget_key_distinguishes_every_axis() -> None:
    keys = {
        FrameBudget(16).key,
        FrameBudget(8).key,
        FrameBudget(16, max_side=448).key,
    }
    assert len(keys) == 3


def _fake_decoder(calls: list[tuple[str, int]], size: tuple[int, int] = (64, 32)) -> Any:
    from PIL import Image

    def decode(path: str, frames: int) -> list[Any]:
        calls.append((path, frames))
        return [Image.new("RGB", size, (index * 10, 0, 0)) for index in range(frames)]

    return decode


@pytest.fixture
def clip(tmp_path: Path) -> Path:
    path = tmp_path / "clip.mp4"
    path.write_bytes(b"not really a video; the stand-in decoder never reads it")
    return path


def test_frames_are_decoded_once_then_served_from_cache(
    clip: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[str, int]] = []
    monkeypatch.setitem(video._DECODERS, "fake", _fake_decoder(calls))
    budget = FrameBudget(4)
    first = video_frame_paths(clip, budget, tmp_path / "cache", backends=["fake"])
    second = video_frame_paths(clip, budget, tmp_path / "cache", backends=["fake"])
    assert first == second and len(first) == 4
    assert calls == [(str(clip), 4)]  # the second call never decoded
    record = json.loads((Path(first[0]).parent / "frames.json").read_text())
    assert record["backend"] == "fake" and record["budget"] == "f4-uniform-s0"


def test_a_different_budget_decodes_its_own_frames(
    clip: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[str, int]] = []
    monkeypatch.setitem(video._DECODERS, "fake", _fake_decoder(calls))
    video_frame_paths(clip, FrameBudget(2), tmp_path / "c", backends=["fake"])
    paths = video_frame_paths(clip, FrameBudget(3), tmp_path / "c", backends=["fake"])
    assert len(paths) == 3 and len(calls) == 2


def test_max_side_caps_the_longer_side_and_keeps_aspect(
    clip: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from PIL import Image

    monkeypatch.setitem(video._DECODERS, "fake", _fake_decoder([], size=(400, 200)))
    paths = video_frame_paths(clip, FrameBudget(1, max_side=100), tmp_path / "c", backends=["fake"])
    assert Image.open(paths[0]).size == (100, 50)


def test_every_failing_backend_is_named_in_the_refusal(
    clip: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def broken(path: str, frames: int) -> list[Any]:
        raise RuntimeError("codec missing")

    monkeypatch.setitem(video._DECODERS, "b1", broken)
    monkeypatch.setitem(video._DECODERS, "b2", broken)
    with pytest.raises(VideoDecodeError, match=r"b1: codec missing.*b2: codec missing"):
        video_frame_paths(clip, FrameBudget(2), tmp_path / "c", backends=["b1", "b2"])


def test_a_backend_returning_the_wrong_count_is_refused(
    clip: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(video._DECODERS, "short", lambda path, frames: [])
    with pytest.raises(VideoDecodeError, match="got 0"):
        video_frame_paths(clip, FrameBudget(2), tmp_path / "c", backends=["short"])


def test_no_backend_at_all_is_refused_with_the_install_hint(clip: Path, tmp_path: Path) -> None:
    with pytest.raises(VideoDecodeError, match="no video decoder available"):
        video_frame_paths(clip, FrameBudget(2), tmp_path / "c", backends=[])


def test_a_missing_clip_is_refused(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        video_frame_paths(tmp_path / "absent.mp4", FrameBudget(2), tmp_path / "c", backends=[])


def test_backend_order_can_be_pinned_and_unknown_names_refuse(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FOUNDATIONSCALE_VIDEO_BACKEND", "nope")
    with pytest.raises(ValueError, match="unknown backend"):
        video.available_backends()
    monkeypatch.setenv("FOUNDATIONSCALE_VIDEO_BACKEND", "ffmpeg")
    monkeypatch.setenv("FOUNDATIONSCALE_FFMPEG", "/x/ffmpeg")
    monkeypatch.setenv("FOUNDATIONSCALE_FFPROBE", "/x/ffprobe")
    assert video.available_backends() == ["ffmpeg"]


def test_module_imports_without_decoders() -> None:
    import importlib

    assert importlib.reload(video).BACKENDS == ("torchcodec", "av", "cv2", "ffmpeg")


# -- the SFT plane's declaration and row fold -----------------------------------------


def test_no_frame_count_is_no_budget() -> None:
    assert video.budget_from_env({}) is None
    assert video.budget_from_env({video.VIDEO_FRAMES_ENV: "  "}) is None


def test_budget_from_env_reads_every_axis() -> None:
    budget = video.budget_from_env(
        {
            video.VIDEO_FRAMES_ENV: "16",
            video.VIDEO_MAX_SIDE_ENV: "448",
            video.VIDEO_SAMPLING_ENV: "uniform",
        }
    )
    # Fields, not ==: another leg reloads the module, minting a second FrameBudget class.
    assert budget is not None
    assert (budget.frames, budget.sampling, budget.max_side) == (16, "uniform", 448)


@pytest.mark.parametrize(
    "environ",
    [
        {video.VIDEO_FRAMES_ENV: "sixteen"},
        {video.VIDEO_FRAMES_ENV: "0"},
        {video.VIDEO_FRAMES_ENV: "4", video.VIDEO_SAMPLING_ENV: "random"},
        {video.VIDEO_FRAMES_ENV: "4", video.VIDEO_MAX_SIDE_ENV: "big"},
    ],
)
def test_an_undeclarable_budget_raises(environ: dict[str, str]) -> None:
    with pytest.raises(ValueError):
        video.budget_from_env(environ)


def test_fold_appends_frames_after_existing_images_and_strips_the_marker(
    clip: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[str, int]] = []
    monkeypatch.setattr(video, "_DECODERS", {"fake": _fake_decoder(calls)})
    monkeypatch.setattr(video, "available_backends", lambda: ["fake"])
    row = {"text": "<video> what happens?", "img": "a.png", "clip": clip.name}
    out = video.fold_video_row(
        row,
        video_column="clip",
        image_column="img",
        budget=FrameBudget(3),
        cache_dir=tmp_path / "cache",
        base_dir=clip.parent,
    )
    assert out["img"][0] == "a.png" and len(out["img"]) == 4
    assert all(Path(p).is_file() for p in out["img"][1:])
    assert out["text"] == "what happens?"
    assert row["img"] == "a.png", "the input row must not be mutated"


def test_fold_passes_a_row_without_a_clip_through(tmp_path: Path) -> None:
    out = video.fold_video_row(
        {"text": "plain <video>", "clip": None},
        video_column="clip",
        image_column=video.VIDEO_FRAMES_COLUMN,
        budget=FrameBudget(3),
        cache_dir=tmp_path,
        base_dir=tmp_path,
    )
    assert out[video.VIDEO_FRAMES_COLUMN] == []
    assert out["text"] == "plain <video>", "no frames, so nothing resolved the marker"


def test_fold_refuses_a_missing_clip(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        video.fold_video_row(
            {"text": "t", "clip": ["gone.mp4"]},
            video_column="clip",
            image_column="img",
            budget=FrameBudget(2),
            cache_dir=tmp_path,
            base_dir=tmp_path,
        )


def test_a_stale_record_is_decoded_afresh(
    clip: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[str, int]] = []
    monkeypatch.setattr(video, "_DECODERS", {"fake": _fake_decoder(calls)})
    first = video_frame_paths(clip, FrameBudget(2), tmp_path, backends=["fake"])
    Path(first[0]).unlink()
    second = video_frame_paths(clip, FrameBudget(2), tmp_path, backends=["fake"])
    assert len(calls) == 2 and all(Path(p).is_file() for p in second)


def test_a_concurrent_publisher_wins_and_its_frames_are_kept(
    clip: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two ranks decoding one clip: the loser keeps the winner's directory."""
    calls: list[tuple[str, int]] = []
    budget = FrameBudget(2)
    inner = _fake_decoder(calls)

    def racing(path: str, frames: int) -> list[Any]:
        images = inner(path, frames)
        if len(calls) == 1:  # the other rank publishes while this one decodes
            video_frame_paths(clip, budget, tmp_path, backends=["plain"])
        return images

    monkeypatch.setattr(video, "_DECODERS", {"race": racing, "plain": inner})
    paths = video_frame_paths(clip, budget, tmp_path, backends=["race"])
    assert len(calls) == 2 and all(Path(p).is_file() for p in paths)
    assert not list(Path(paths[0]).parent.parent.glob(".frames-*")), "staging left behind"


# -- each decoder backend, driven through a stand-in library ---------------------------
# The cluster images differ in which decoder they ship, so every backend's own
# instant arithmetic is pinned here against a fake of its library: all four must
# ask for the SAME centred-uniform instants of an 8-second clip.

_WANT = [1.0, 3.0, 5.0, 7.0]


def _rgb(frames: int) -> Any:
    import numpy as np

    return np.zeros((frames, 6, 8, 3), dtype=np.uint8)


def test_torchcodec_backend_asks_for_the_centred_instants(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import sys
    import types

    asked: list[list[float]] = []

    class _Batch:
        def __init__(self, n: int) -> None:
            self.n = n

        @property
        def data(self) -> _Batch:
            return self

        def permute(self, *_: int) -> _Batch:
            return self

        def cpu(self) -> _Batch:
            return self

        def numpy(self) -> Any:
            return _rgb(self.n)

    class VideoDecoder:
        metadata = types.SimpleNamespace(duration_seconds=8.0)

        def __init__(self, path: str) -> None:
            self.path = path

        def get_frames_played_at(self, seconds: list[float]) -> _Batch:
            asked.append(seconds)
            return _Batch(len(seconds))

    decoders = types.ModuleType("torchcodec.decoders")
    decoders.VideoDecoder = VideoDecoder  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "torchcodec", types.ModuleType("torchcodec"))
    monkeypatch.setitem(sys.modules, "torchcodec.decoders", decoders)
    images = video._decode_torchcodec("c.mp4", 4)
    assert asked == [_WANT] and len(images) == 4


def test_av_backend_seeks_each_instant_and_keeps_the_first_frame_at_or_after_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import sys
    import types
    from fractions import Fraction

    from PIL import Image

    seeks: list[int] = []

    class _Frame:
        def __init__(self, time: float) -> None:
            self.time = time

        def to_image(self) -> Any:
            return Image.new("RGB", (8, 6), (int(self.time * 10), 0, 0))

    class _Container:
        duration = None

        def __init__(self) -> None:
            stream = types.SimpleNamespace(duration=8000, time_base=Fraction(1, 1000))
            self.streams = types.SimpleNamespace(video=[stream])
            self.at = 0.0

        def __enter__(self) -> _Container:
            return self

        def __exit__(self, *_: Any) -> None:
            return None

        def seek(self, offset: int, **_: Any) -> None:
            seeks.append(offset)
            self.at = max(0.0, offset / 1000 - 0.5)

        def decode(self, _stream: Any) -> Any:
            t = self.at
            while t < 8.0:
                yield _Frame(t)
                t += 0.25

    fake = types.ModuleType("av")
    fake.open = lambda _path: _Container()  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "av", fake)
    images = video._decode_av("c.mp4", 4)
    assert seeks == [1000, 3000, 5000, 7000]
    assert [im.getpixel((0, 0))[0] for im in images] == [10, 30, 50, 70]


def test_av_backend_refuses_a_clip_with_no_frame_at_an_instant(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import sys
    import types

    class _Empty:
        duration = 2_000_000
        streams = types.SimpleNamespace(
            video=[types.SimpleNamespace(duration=None, time_base=0.001)]
        )

        def __enter__(self) -> _Empty:
            return self

        def __exit__(self, *_: Any) -> None:
            return None

        def seek(self, *_: Any, **__: Any) -> None:
            return None

        def decode(self, _stream: Any) -> Any:
            return iter(())

    fake = types.ModuleType("av")
    fake.open = lambda _path: _Empty()  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "av", fake)
    with pytest.raises(video.VideoDecodeError, match="no frame decoded"):
        video._decode_av("c.mp4", 2)


def _fake_cv2(opened: bool = True, readable: bool = True) -> tuple[Any, list[float]]:
    import types

    positions: list[float] = []

    class VideoCapture:
        def __init__(self, path: str) -> None:
            self.path = path

        def isOpened(self) -> bool:  # noqa: N802 -- cv2's spelling
            return opened

        def get(self, prop: int) -> float:
            return {1: 25.0, 2: 200.0}[prop]

        def set(self, prop: int, value: float) -> None:
            positions.append(value)

        def read(self) -> tuple[bool, Any]:
            return readable, _rgb(1)[0]

        def release(self) -> None:
            return None

    fake = types.ModuleType("cv2")
    fake.CAP_PROP_FPS = 1  # type: ignore[attr-defined]
    fake.CAP_PROP_FRAME_COUNT = 2  # type: ignore[attr-defined]
    fake.CAP_PROP_POS_MSEC = 3  # type: ignore[attr-defined]
    fake.COLOR_BGR2RGB = 4  # type: ignore[attr-defined]
    fake.VideoCapture = VideoCapture  # type: ignore[attr-defined]
    fake.cvtColor = lambda array, _code: array  # type: ignore[attr-defined]
    return fake, positions


def test_cv2_backend_positions_by_milliseconds(monkeypatch: pytest.MonkeyPatch) -> None:
    import sys

    fake, positions = _fake_cv2()
    monkeypatch.setitem(sys.modules, "cv2", fake)
    images = video._decode_cv2("c.mp4", 4)
    assert positions == [t * 1000.0 for t in _WANT] and len(images) == 4


@pytest.mark.parametrize(
    ("opened", "readable", "message"),
    [(False, True, "could not open"), (True, False, "read failed")],
)
def test_cv2_backend_refusals(
    monkeypatch: pytest.MonkeyPatch, opened: bool, readable: bool, message: str
) -> None:
    import sys

    fake, _ = _fake_cv2(opened=opened, readable=readable)
    monkeypatch.setitem(sys.modules, "cv2", fake)
    with pytest.raises(video.VideoDecodeError, match=message):
        video._decode_cv2("c.mp4", 2)


def _png() -> bytes:
    import io

    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", (8, 6)).save(buffer, format="PNG")
    return buffer.getvalue()


def test_ffmpeg_backend_grabs_each_instant(monkeypatch: pytest.MonkeyPatch) -> None:
    import subprocess
    import types

    grabs: list[str] = []

    def run(cmd: list[str], **_: Any) -> Any:
        if cmd[0] == "ffprobe":
            return types.SimpleNamespace(stdout="8.0\n", stderr="", returncode=0)
        grabs.append(cmd[cmd.index("-ss") + 1])
        return types.SimpleNamespace(stdout=_png(), stderr=b"", returncode=0)

    monkeypatch.setattr(video, "_ffmpeg_bins", lambda: ("ffmpeg", "ffprobe"))
    monkeypatch.setattr(subprocess, "run", run)
    images = video._decode_ffmpeg("c.mp4", 4)
    assert [float(g) for g in grabs] == _WANT and len(images) == 4


@pytest.mark.parametrize(
    ("probe_out", "grab_rc", "message"),
    [("N/A", 0, "no duration"), ("8.0", 1, "grab failed")],
)
def test_ffmpeg_backend_refusals(
    monkeypatch: pytest.MonkeyPatch, probe_out: str, grab_rc: int, message: str
) -> None:
    import subprocess
    import types

    def run(cmd: list[str], **_: Any) -> Any:
        if cmd[0] == "ffprobe":
            return types.SimpleNamespace(stdout=probe_out, stderr="bad", returncode=0)
        return types.SimpleNamespace(stdout=b"", stderr=b"", returncode=grab_rc)

    monkeypatch.setattr(video, "_ffmpeg_bins", lambda: ("ffmpeg", "ffprobe"))
    monkeypatch.setattr(subprocess, "run", run)
    with pytest.raises(video.VideoDecodeError, match=message):
        video._decode_ffmpeg("c.mp4", 2)


def test_ffmpeg_backend_without_binaries_refuses(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(video, "_ffmpeg_bins", lambda: None)
    with pytest.raises(video.VideoDecodeError, match="not found"):
        video._decode_ffmpeg("c.mp4", 2)


def test_ffmpeg_bins_honour_the_override_variables(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FOUNDATIONSCALE_FFMPEG", "/x/ffmpeg")
    monkeypatch.setenv("FOUNDATIONSCALE_FFPROBE", "/x/ffprobe")
    assert video._ffmpeg_bins() == ("/x/ffmpeg", "/x/ffprobe")
