"""CPU legs for the conversation video-frame path: frames_for and its refusals.

video.py's ``conversation_frames_for`` feeds frames into the conversation
collator. When loop.py builds that callable is pinned through ``train()`` in
tests/train/test_conversation_fused_loss_fsdp_wiring.py. Decoding runs through an
injected stand-in backend exactly as
tests/train/test_video.py does (monkeypatching ``_DECODERS``/``available_backends``),
so these legs pass on any runner with no decoder library and no real clip.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from foundationscale import video
from foundationscale.video import FrameBudget, conversation_frames_for


def _fake_decoder(calls: list[tuple[str, int]]) -> Any:
    from PIL import Image

    def decode(path: str, frames: int) -> list[Any]:
        calls.append((path, frames))
        return [Image.new("L", (16, 8), index * 20) for index in range(frames)]

    return decode


@pytest.fixture
def clip(tmp_path: Path) -> Path:
    path = tmp_path / "clips" / "clip0.mp4"
    path.parent.mkdir()
    path.write_bytes(b"stand-in bytes; the fake decoder never reads them")
    return path


def _wire(monkeypatch: pytest.MonkeyPatch, calls: list[tuple[str, int]]) -> None:
    monkeypatch.setitem(video._DECODERS, "fake", _fake_decoder(calls))
    monkeypatch.setattr(video, "available_backends", lambda: ["fake"])


def test_frames_for_returns_budget_frames_rgb_then_cache(
    clip: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from PIL import Image

    calls: list[tuple[str, int]] = []
    _wire(monkeypatch, calls)
    budget = FrameBudget(4)
    frames_for = conversation_frames_for(
        budget,
        tmp_path / "cache",
        base_dir=tmp_path,
        video_column="clips",
    )
    first = frames_for({"clips": "clips/clip0.mp4"})
    assert first["frames_indices"] == list(range(4))
    assert len(first["frames"]) == 4
    assert all(isinstance(f, Image.Image) and f.mode == "RGB" for f in first["frames"])
    second = frames_for({"clips": "clips/clip0.mp4"})
    assert second["frames_indices"] == list(range(4))
    assert len(second["frames"]) == 4
    assert calls == [(str(clip), 4)]  # the second call decoded zero times


def test_a_relative_clip_that_does_not_exist_is_file_not_found(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[str, int]] = []
    _wire(monkeypatch, calls)
    frames_for = conversation_frames_for(
        FrameBudget(2), tmp_path / "cache", base_dir=tmp_path, video_column="clips"
    )
    with pytest.raises(FileNotFoundError):
        frames_for({"clips": "absent/clip9.mp4"})
    assert calls == []


def test_a_row_naming_two_clips_is_a_value_error(
    clip: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[str, int]] = []
    _wire(monkeypatch, calls)
    frames_for = conversation_frames_for(
        FrameBudget(2), tmp_path / "cache", base_dir=tmp_path, video_column="clips"
    )
    with pytest.raises(ValueError, match="one clip"):
        frames_for({"clips": ["clips/clip0.mp4", "clips/clip1.mp4"]})
    assert calls == []


def test_the_collator_refuses_96_naming_row_zero_on_frame_failure(
    clip: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from foundationscale.train.conversation import train_conversation_collator_or_refuse

    def frames_for(_data: Any) -> dict[str, Any]:
        raise FileNotFoundError("video clips/nope.mp4 does not exist")

    class _Proc:
        def apply_chat_template(self, *a: Any, **k: Any) -> str:
            raise AssertionError("the processor must not be reached before the frames")

    collate = train_conversation_collator_or_refuse(
        _Proc(),
        conversations_column="conversations",
        image_column=None,
        video_column="clips",
        max_length=64,
        overlong="drop",
        frames_for=frames_for,
    )
    row = {
        "conversations": [
            {"from": "human", "value": "<video> what happens?"},
            {"from": "gpt", "value": "something"},
        ],
        "clips": "clips/clip0.mp4",
    }
    with pytest.raises(SystemExit) as excinfo:
        collate([row])
    assert excinfo.value.code == 96
    message = getattr(excinfo.value, "fs_refusal_message", "")
    assert "row 0" in message


# ---------------------------------------------------------------------------
# loop wiring: conversation_prepass_or_refuse / train_conversation_collator_or_refuse
# ---------------------------------------------------------------------------
