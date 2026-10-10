"""The SFT plane's video fold: a declared column plus a frame budget rides the image arm."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from foundationscale import video
from foundationscale.train import loop
from foundationscale.video import FrameBudget


class _Split:
    """The two members of a datasets split the fold touches."""

    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.rows = rows
        self.column_names = sorted({key for row in rows for key in row})

    def map(self, fn: Any) -> _Split:
        return _Split([fn(row) for row in self.rows])


def _frames(monkeypatch: pytest.MonkeyPatch, seen: list[str]) -> None:
    def fake(path: Any, budget: FrameBudget, cache_dir: Any, **_: Any) -> list[str]:
        seen.append(str(path))
        return [f"{cache_dir}/f{index}.jpg" for index in range(budget.frames)]

    monkeypatch.setattr(video, "video_frame_paths", fake)


def _fold(split: _Split, dataset: str, **overrides: Any) -> Any:
    kwargs: dict[str, Any] = {
        "dataset": dataset,
        "video_column": "clip",
        "image_column": video.VIDEO_FRAMES_COLUMN,
        "budget": FrameBudget(16),
        "cache_dir": None,
    }
    kwargs.update(overrides)
    return loop._fold_video_split(split, **kwargs)


def test_clips_resolve_beside_a_local_dataset_and_cache_there(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[str] = []
    _frames(monkeypatch, seen)
    data = tmp_path / "train.jsonl"
    data.write_text("{}\n")
    out = _fold(_Split([{"text": "<video> q", "clip": "a.mp4"}]), str(data))
    assert not isinstance(out, str)
    row = out.rows[0]
    assert seen == [str(tmp_path / "a.mp4")]
    assert len(row[video.VIDEO_FRAMES_COLUMN]) == 16
    assert row[video.VIDEO_FRAMES_COLUMN][0].startswith(str(tmp_path / ".fs_video_frames"))
    assert video.VIDEO_FRAMES_COLUMN in out.column_names


def test_an_explicit_cache_dir_wins(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _frames(monkeypatch, [])
    out = _fold(_Split([{"text": "q", "clip": "a.mp4"}]), str(tmp_path), cache_dir="/c")
    assert out.rows[0][video.VIDEO_FRAMES_COLUMN][0].startswith("/c/")


def test_an_absent_video_column_refuses(tmp_path: Path) -> None:
    out = _fold(_Split([{"text": "q"}]), str(tmp_path))
    assert isinstance(out, str) and "'clip'" in out and "silent-drop" in out


def test_an_undecodable_clip_refuses(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def broken(*_: Any, **__: Any) -> list[str]:
        raise video.VideoDecodeError("no decoder")

    monkeypatch.setattr(video, "video_frame_paths", broken)
    out = _fold(_Split([{"text": "q", "clip": "a.mp4"}]), "hub/dataset-id")
    assert isinstance(out, str) and "no decoder" in out


def test_an_undeclared_column_pair_refuses() -> None:
    out = _fold(_Split([{"text": "q"}]), "x", image_column=None)
    assert isinstance(out, str)
