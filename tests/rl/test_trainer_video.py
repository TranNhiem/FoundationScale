"""The RL trainer's video expansion: a declared budget turns clips into images."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from foundationscale.rl import trainer
from foundationscale.rl.corpus import Sample


def _sample(sample_id: str, video: str | None = None, images: tuple[str, ...] = ()) -> Sample:
    return Sample(
        sample_id=sample_id,
        prompt_turns=(("user", "Which?"),),
        response="A",
        gold="A",
        images=images,
        video=video,
    )


def _config(tmp_path: Path, frames: int) -> Any:
    return SimpleNamespace(
        dataset=str(tmp_path / "data.jsonl"),
        video_frames=frames,
        video_sampling="uniform",
        video_max_side=None,
        video_cache_dir=str(tmp_path / "cache"),
    )


def test_text_only_samples_pass_through_untouched(tmp_path: Path) -> None:
    samples = [_sample("t0"), _sample("t1")]
    assert trainer._expand_video_samples(samples, _config(tmp_path, 0)) == tuple(samples)


def test_video_without_a_declared_budget_refuses_naming_the_sample(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit) as excinfo:
        trainer._expand_video_samples([_sample("v7", video="a.mp4")], _config(tmp_path, 0))
    assert excinfo.value.code == 96
    assert "v7 (video)" in capsys.readouterr().err


def test_a_declared_budget_appends_the_frames_as_images(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[tuple[Path, int, str]] = []

    def fake_paths(clip: Path, budget: Any, cache: str) -> list[str]:
        seen.append((clip, budget.frames, cache))
        return [f"{clip.stem}_f{index}.jpg" for index in range(budget.frames)]

    monkeypatch.setattr("foundationscale.video.video_frame_paths", fake_paths)
    out = trainer._expand_video_samples(
        [_sample("v0", video="clips/a.mp4", images=("lead.png",)), _sample("t0")],
        _config(tmp_path, 3),
    )
    assert out[0].video is None
    assert out[0].images == ("lead.png", "a_f0.jpg", "a_f1.jpg", "a_f2.jpg")
    assert out[1].images == ()
    clip, frames, cache = seen[0]
    assert clip == tmp_path / "clips" / "a.mp4"  # resolved beside the dataset
    assert frames == 3 and cache == str(tmp_path / "cache")


def test_an_undecodable_clip_refuses_naming_the_sample(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from foundationscale.video import VideoDecodeError

    def broken(clip: Path, budget: Any, cache: str) -> list[str]:
        raise VideoDecodeError("no decoder")

    monkeypatch.setattr("foundationscale.video.video_frame_paths", broken)
    with pytest.raises(SystemExit) as excinfo:
        trainer._expand_video_samples([_sample("v3", video="/abs/x.mp4")], _config(tmp_path, 2))
    assert excinfo.value.code == 96
    assert "v3" in capsys.readouterr().err


def test_an_invalid_budget_refuses(tmp_path: Path) -> None:
    config = _config(tmp_path, 2)
    config.video_sampling = "random"
    with pytest.raises(SystemExit) as excinfo:
        trainer._expand_video_samples([_sample("v0", video="a.mp4")], config)
    assert excinfo.value.code == 96
