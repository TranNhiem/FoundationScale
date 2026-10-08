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
