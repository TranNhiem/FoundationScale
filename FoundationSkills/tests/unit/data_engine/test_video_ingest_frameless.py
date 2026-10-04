"""Tests for ASR segments left behind in frameless chunk windows (``video_ingest``).

Same fakes/monkeypatching as :mod:`tests.unit.data_engine.test_video_ingest` -- a
synthetic ffmpeg clip plus an injected ``_ASR_FACTORY`` -- focused on the chunk
path: a window ``[lo, hi)`` with a segment but no sampled frame emits nothing,
so its segments must be counted instead of vanishing uncounted.
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from foundationskills.core.schema import validate
from foundationskills.skills.data_engine.ops.base import OPS, OpStats
from foundationskills.skills.data_engine.ops import video_ingest as vi_mod
from foundationskills.skills.data_engine.ops.video_ingest import CONFIG_SCHEMA


SEGS = [
    {"start": 0.5, "end": 1.5, "text": "alpha"},  # midpoint 1.0 -> window [0, 2.5) keeps the frame
    {"start": 3.0, "end": 3.5, "text": "gamma"},  # midpoint 3.25 -> frameless window [2.5, 5.0)
]


def _make_clip(directory: Path) -> Path:
    """4 s testsrc (160x120 @ 10 fps) muxed with a 440 Hz sine; requires ffmpeg."""
    ffmpeg = shutil.which("ffmpeg")
    assert ffmpeg is not None, "ffmpeg must be resolvable to build the synthetic clip"
    out = Path(directory) / "clip.mp4"
    if out.is_file() and out.stat().st_size > 0:
        return out
    base = [
        ffmpeg, "-v", "error", "-y",
        "-f", "lavfi", "-i", "testsrc=duration=4:size=160x120:rate=10",
        "-f", "lavfi", "-i", "sine=frequency=440:duration=4",
        "-pix_fmt", "yuv420p", "-shortest",
    ]
    last = ""
    for audio_args in (["-c:a", "aac"], ["-c:a", "libmp3lame"], []):
        proc = subprocess.run(base + audio_args + [str(out)], capture_output=True)
        if proc.returncode == 0 and out.is_file() and out.stat().st_size > 0:
            return out
        last = proc.stderr.decode("utf-8", errors="replace")
        if out.is_file():
            out.unlink()
    raise AssertionError(f"ffmpeg cannot build the synthetic clip: {last}")


@pytest.fixture(scope="module")
def clip(tmp_path_factory) -> Path:
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg not on PATH; the synthetic clip cannot be generated")
    return _make_clip(tmp_path_factory.mktemp("video-frameless"))


def _cfg(tmp_path: Path, **overrides) -> dict:
    cfg = {
        "frames_dir": str(tmp_path / "frames"),
        "fps": 0.5,
        "max_frames": 16,
        "frame_width": 160,
        "asr": "whisper",
        "asr_model": "tests/fake-whisper",
    }
    cfg.update(overrides)
    return cfg


def _run(records, cfg):
    errors = validate(cfg, CONFIG_SCHEMA)
    assert not errors, errors
    stats = OpStats(name="video_ingest")
    out = list(OPS["video_ingest"](iter(records), cfg, stats))
    dropped = sum(stats.dropped.values())
    assert stats.records_in + stats.extra["chunks_extra"] == stats.records_out + dropped, stats.to_dict()
    return out, stats


def _fake_asr(monkeypatch, segments, calls=None, error=None):
    seen = [] if calls is None else calls

    def factory(cfg):
        def run(audio, sr):
            seen.append((audio, sr))
            if error is not None:
                raise error
            return [dict(seg) for seg in segments]

        return run

    monkeypatch.setattr(vi_mod, "_ASR_FACTORY", factory)
    return seen


def test_asr_segments_in_frameless_window_are_counted(tmp_path, monkeypatch, clip):
    # one sampled frame at t = 2.0 s -> window [0, 2.5) has it, window [2.5, 5.0) has none
    _fake_asr(monkeypatch, SEGS)
    out, stats = _run([{"id": "vidF", "video": str(clip)}], _cfg(tmp_path, chunk_seconds=2.5, max_frames=1))
    assert [r["id"] for r in out] == ["vidF#c0"]
    assert [s["text"] for s in out[0]["meta"]["transcript"]] == ["alpha"]
    assert stats.extra["asr_segments_frameless_window"] == 1


def test_frameless_window_counter_present_when_nothing_is_lost(tmp_path, monkeypatch, clip):
    _fake_asr(monkeypatch, SEGS)
    out, stats = _run([{"id": "vidN", "video": str(clip)}], _cfg(tmp_path))
    assert stat_key_present(stats)
    assert len(out[0]["meta"]["transcript"]) == 2
    assert stats.extra["asr_segments_frameless_window"] == 0


def test_frameless_window_counter_present_with_full_chunk_windows(tmp_path, monkeypatch, clip):
    # two frames (t = 1.0/3.0 s) -> every chunk window keeps one frame
    _fake_asr(monkeypatch, SEGS)
    out, stats = _run(
        [{"id": "vidC", "video": str(clip)}],
        _cfg(tmp_path, chunk_seconds=2.5, max_frames=2),
    )
    assert stat_key_present(stats)
    assert [r["id"] for r in out] == ["vidC#c0", "vidC#c1"]
    assert stats.extra["asr_segments_frameless_window"] == 0


def stat_key_present(stats: OpStats) -> bool:
    return "asr_segments_frameless_window" in stats.extra
