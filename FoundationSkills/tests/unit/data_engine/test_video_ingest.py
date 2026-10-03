"""Tests for the ``video_ingest`` op: ffmpeg frames + injected ASR backend (CPU, fast).

Heavy parts are injected -- ``video_ingest._ASR_FACTORY`` is monkeypatched with a
fake so no model weights are needed -- except for one ``gpu``-marked test that runs
the real whisper-tiny path whenever :func:`preflight` reports no problems (its skip
is therefore always visible in the junit xml, never a silent pass).
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from foundationskills.core.schema import validate
from foundationskills.skills.data_engine.ops.base import OPS, OpStats, OpUnavailable
from foundationskills.skills.data_engine.ops import video_ingest as vi_mod
from foundationskills.skills.data_engine.ops.video_ingest import CONFIG_SCHEMA, preflight


SEGS = [
    {"start": 0.5, "end": 1.5, "text": "alpha"},
    {"start": 2.5, "end": 3.5, "text": "beta"},
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
    # rv-note: an unencodable clip is a broken toolchain, not a reason to pass silently
    raise AssertionError(f"ffmpeg cannot build the synthetic clip: {last}")


@pytest.fixture(scope="module")
def clip(tmp_path_factory) -> Path:
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg not on PATH; the synthetic clip cannot be generated")
    return _make_clip(tmp_path_factory.mktemp("video"))


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


def test_frames_times_alignment_and_text_layout(tmp_path, monkeypatch, clip):
    calls = _fake_asr(monkeypatch, SEGS)
    out, stats = _run([{"id": "vid1", "video": str(clip), "meta": {"source": "unit"}}], _cfg(tmp_path))
    assert [r["id"] for r in out] == ["vid1"]
    rec = out[0]
    meta = rec["meta"]
    times = meta["frame_times_s"]
    duration = meta["duration_s"]
    assert duration == pytest.approx(4.0, abs=0.25)
    # K = min(max_frames=16, floor(duration * 0.5)) = 2 and t_k = (k + 0.5) * duration / K
    assert times == pytest.approx([0.25 * duration, 0.75 * duration], abs=1e-9)
    assert len(rec["images"]) == 2 and all(Path(p).is_file() for p in rec["images"])
    expected = "\n".join([
        f"<frame 1 @ {times[0]:.1f}s>",
        "alpha",  # midpoint 1.0 -> first frame interval [0, duration/2)
        f"<frame 2 @ {times[1]:.1f}s>",
        "beta",
    ])
    assert rec["text"] == expected
    assert meta["source"] == "unit"
    assert meta["video"] == str(clip)
    assert meta["asr"] == {"model": "tests/fake-whisper", "language": None, "segments": 2}
    assert meta["transcript"] == SEGS
    audio, sr = calls[0]
    assert sr == 16000
    assert len(audio) > 1000
    assert getattr(audio.dtype, "name", str(audio.dtype)) == "float32"
    assert stats.backend == "ffmpeg+transformers-asr:tests/fake-whisper"
    assert stats.extra["frames_written"] == 2
    assert stats.extra["no_audio"] == 0
    assert stats.extra["total_duration_s"] == pytest.approx(duration)


def test_frame_cache_reused_on_second_run(tmp_path, monkeypatch, clip):
    cfg = _cfg(tmp_path, asr="none")
    _out1, stats1 = _run([{"id": "a", "video": str(clip)}], cfg)
    out2, stats2 = _run([{"id": "a", "video": str(clip)}], cfg)
    assert stats1.extra["frames_written"] == 2 and stats1.extra["frames_cached"] == 0
    assert stats2.extra["frames_written"] == 0 and stats2.extra["frames_cached"] == 2
    assert len(out2[0]["images"]) == 2


def test_frame_cache_never_relabels_frames_across_sampling_plans(tmp_path, clip):
    # plan A samples t = 1.0/3.0 s, plan B t = 0.5/1.5/2.5/3.5 s: no cached frame may be reused
    _out_a, stats_a = _run([{"id": "a", "video": str(clip)}], _cfg(tmp_path, asr="none", fps=2.0, max_frames=2))
    out_b, stats_b = _run([{"id": "a", "video": str(clip)}], _cfg(tmp_path, asr="none", fps=2.0, max_frames=4))
    assert stats_a.extra["frames_written"] == 2
    assert stats_b.extra["frames_cached"] == 0 and stats_b.extra["frames_written"] == 4
    times = out_b[0]["meta"]["frame_times_s"]
    assert [Path(p).name for p in out_b[0]["images"]] == [f"frame_{t:010.3f}s_w160.jpg" for t in times]


def test_broken_audio_stream_is_not_counted_as_no_audio(tmp_path, monkeypatch, clip):
    monkeypatch.setattr(vi_mod, "_has_audio", lambda ffprobe, path: True)
    real_run = subprocess.run

    def fake_run(cmd, *args, **kwargs):
        if "f32le" in cmd:
            return subprocess.CompletedProcess(cmd, 1, b"", b"Invalid data found when processing input\n")
        return real_run(cmd, *args, **kwargs)

    monkeypatch.setattr(vi_mod.subprocess, "run", fake_run)
    calls = _fake_asr(monkeypatch, SEGS)
    out, stats = _run([{"id": "v", "video": str(clip)}], _cfg(tmp_path))
    assert out == [] and calls == []
    assert stats.dropped["audio_extract_failed"] == 1
    assert stats.extra["no_audio"] == 0
    assert "Invalid data" in stats.extra["asr_errors"][0]


def test_chunking_ids_windows_and_transcripts(tmp_path, monkeypatch, clip):
    _fake_asr(monkeypatch, SEGS)
    out, stats = _run([{"id": "vid9", "video": str(clip)}], _cfg(tmp_path, chunk_seconds=2.5))
    assert [r["id"] for r in out] == ["vid9#c0", "vid9#c1"]
    assert stats.records_out == 2
    assert stats.extra["chunks_emitted"] == 1
    assert stats.extra["chunks_extra"] == 1
    assert [s["text"] for s in out[0]["meta"]["transcript"]] == ["alpha"]
    assert [s["text"] for s in out[1]["meta"]["transcript"]] == ["beta"]
    assert len(out[0]["images"]) == 1 and len(out[1]["images"]) == 1
    assert out[1]["meta"]["duration_s"] == out[0]["meta"]["duration_s"]


def test_chunk_windows_without_frames_are_skipped(tmp_path, monkeypatch, clip):
    # single frame near t = 2.0 s -> only window [0, 2.5) has a frame
    out, stats = _run(
        [{"id": "vidS", "video": str(clip)}],
        _cfg(tmp_path, chunk_seconds=2.5, max_frames=1, asr="none"),
    )
    assert [r["id"] for r in out] == ["vidS#c0"]
    assert len(out[0]["images"]) == 1
    assert stats.extra["chunks_emitted"] == 0


def test_drops_video_missing_probe_failed_too_long(tmp_path, monkeypatch, clip):
    _fake_asr(monkeypatch, SEGS)
    junk = tmp_path / "not-a-video.bin"
    junk.write_bytes(b"definitely not a movie")
    cfg = _cfg(tmp_path, max_duration_s=2.0)
    out, stats = _run(
        [
            {"id": "missing", "video": str(tmp_path / "nope.mp4")},
            {"id": "junk", "video": str(junk)},
            {"id": "long", "video": str(clip)},
        ],
        cfg,
    )
    assert out == []
    assert stats.dropped["video_missing"] == 1
    assert stats.dropped["probe_failed"] == 1
    assert stats.dropped["too_long"] == 1
    assert stats.extra["asr_errors"] == []


def test_non_video_passthrough_and_require_video(tmp_path):
    text_only = {"id": "note", "text": "just text"}
    out, stats = _run([text_only], _cfg(tmp_path, asr="none"))
    assert out[0] is text_only  # untouched pass-through
    assert stats.extra["passthrough_non_video"] == 1
    out, stats = _run([{"id": "note", "text": "x"}], _cfg(tmp_path, asr="none", require_video=True))
    assert out == []
    assert stats.dropped["no_video"] == 1


def test_asr_failed_records_first_error(tmp_path, monkeypatch, clip):
    _fake_asr(monkeypatch, [], error=RuntimeError("boom: model exploded"))
    out, stats = _run([{"id": "v", "video": str(clip)}], _cfg(tmp_path))
    assert out == []
    assert stats.dropped["asr_failed"] == 1
    assert len(stats.extra["asr_errors"]) == 1
    assert "boom" in stats.extra["asr_errors"][0]


def test_repetition_loop_hallucination_is_suppressed(tmp_path, monkeypatch, clip):
    # whisper-small really emitted this on a 440 Hz sine tone (GPU smoke, 2026-10-03)
    loop = {"start": 0.5, "end": 1.5, "text": "B" + "e" * 120}
    speech = {"start": 2.0, "end": 3.0, "text": "the robot picks up the red cube"}
    _fake_asr(monkeypatch, [loop, speech])
    out, stats = _run([{"id": "v", "video": str(clip)}], _cfg(tmp_path))
    assert stats.extra["asr_segments_suppressed"] == 1
    assert [seg["text"] for seg in out[0]["meta"]["transcript"]] == [speech["text"]]
    _fake_asr(monkeypatch, [loop, speech])
    out, stats = _run([{"id": "v", "video": str(clip)}], _cfg(tmp_path, max_compression_ratio=1000.0))
    assert stats.extra["asr_segments_suppressed"] == 0
    assert len(out[0]["meta"]["transcript"]) == 2


def test_missing_asr_dependency_raises_op_unavailable(tmp_path, monkeypatch, clip):
    def factory(cfg):
        raise OpUnavailable("video_ingest: fake model weights missing")

    monkeypatch.setattr(vi_mod, "_ASR_FACTORY", factory)
    stats = OpStats(name="video_ingest")
    with pytest.raises(OpUnavailable, match="fake model weights missing"):
        list(OPS["video_ingest"](iter([{"id": "v", "video": str(clip)}]), _cfg(tmp_path), stats))


def test_clip_without_audio_stream_counts_no_audio(tmp_path, monkeypatch):
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        pytest.skip("ffmpeg not on PATH; the silent clip cannot be generated")
    silent = tmp_path / "silent.mp4"
    proc = subprocess.run(
        [ffmpeg, "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc=duration=4:size=160x120:rate=10",
         "-pix_fmt", "yuv420p", str(silent)],
        capture_output=True,
    )
    assert proc.returncode == 0 and silent.is_file(), proc.stderr.decode("utf-8", errors="replace")
    calls = _fake_asr(monkeypatch, SEGS)
    out, stats = _run([{"id": "s", "video": str(silent)}], _cfg(tmp_path))
    assert stats.extra["no_audio"] == 1
    assert calls == []  # never ask ASR about a video without audio
    assert out[0]["meta"]["transcript"] == []
    assert out[0]["meta"]["asr"]["segments"] == 0
    assert stats.dropped.get("asr_failed", 0) == 0


def test_emit_messages_and_empty_transcript_drop(tmp_path, monkeypatch, clip):
    _fake_asr(monkeypatch, SEGS)
    cfg = _cfg(tmp_path, emit="messages", prompt="What happens?")
    out, stats = _run([{"id": "m", "video": str(clip), "meta": {"domain": "video"}}], cfg)
    assert "text" not in out[0]
    assert out[0]["messages"] == [
        {"role": "user", "content": "<image><image>What happens?"},
        {"role": "assistant", "content": "alpha\nbeta"},
    ]
    assert out[0]["meta"]["domain"] == "video"
    _fake_asr(monkeypatch, [])
    out, stats = _run([{"id": "e", "video": str(clip)}], cfg)
    assert out == []
    assert stats.dropped["empty_transcript"] == 1


def test_accounting_invariant_mixed_run_with_chunks(tmp_path, monkeypatch, clip):
    _fake_asr(monkeypatch, SEGS)
    out, stats = _run(
        [
            {"id": "keep", "video": str(clip)},
            {"id": "text-only", "text": "no video here"},
            {"id": "gone", "video": str(tmp_path / "nope.mp4")},
        ],
        _cfg(tmp_path, chunk_seconds=2.5, require_video=True),
    )
    assert stats.records_in == 3
    assert stats.records_out == 2
    assert sum(stats.dropped.values()) == 2
    assert stats.extra["chunks_extra"] == 1
    assert stats.records_in + stats.extra["chunks_extra"] == stats.records_out + sum(stats.dropped.values())


def test_preflight_reports_missing_binaries_and_relative_frames_dir(tmp_path):
    problems = preflight({
        "frames_dir": "relative/frames",
        "ffmpeg": "/nonexistent/ffmpeg",
        "ffprobe": "/nonexistent/ffprobe",
        "asr": "none",
    })
    assert any("/nonexistent/ffmpeg" in p for p in problems)
    assert any("/nonexistent/ffprobe" in p for p in problems)
    assert any("absolute" in p for p in problems)


def test_preflight_clean_when_dependencies_present(tmp_path):
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        pytest.skip("ffmpeg/ffprobe not on PATH; cannot assert a clean preflight")
    assert preflight({"frames_dir": str(tmp_path), "asr": "none"}) == []


def test_preflight_reports_uncached_whisper_model(tmp_path):
    problems = preflight({
        "frames_dir": str(tmp_path),
        "asr": "whisper",
        "asr_model": "org/definitely-not-cached-xyz",
        "local_files_only": True,
    })
    assert any("definitely-not-cached-xyz" in p for p in problems)


def test_config_schema_rejects_unknown_key_and_bad_values():
    assert validate({"frames_dir": "/abs-frames"}, CONFIG_SCHEMA) == []
    assert any("additional property" in e for e in validate({"frames_dir": "/abs-frames", "frames": 1}, CONFIG_SCHEMA))
    assert validate({"frames_dir": "/abs-frames", "fps": 0.0}, CONFIG_SCHEMA)  # exclusiveMinimum > 0
    assert validate({"fps": 0.5}, CONFIG_SCHEMA)  # frames_dir is required


@pytest.mark.gpu
def test_real_whisper_tiny_on_synthetic_clip(tmp_path):
    cfg = {
        "frames_dir": str(tmp_path / "frames"),
        "fps": 0.5,
        "max_frames": 4,
        "frame_width": 160,
        "asr": "whisper",
        "asr_model": "openai/whisper-tiny",
        "asr_device": "cpu",
        "local_files_only": True,
    }
    problems = preflight(cfg)
    if problems:
        pytest.skip("real whisper-tiny run unavailable: " + "; ".join(problems))
    clip_dir = tmp_path / "video"
    clip_dir.mkdir()
    clip_path = _make_clip(clip_dir)  # ffmpeg is resolvable: preflight above passed
    out, stats = _run([{"id": "real", "video": str(clip_path)}], cfg)
    assert stats.dropped.get("asr_failed", 0) == 0
    assert isinstance(out[0]["meta"]["transcript"], list)  # sine has no speech: [] is the honest answer
    assert out[0]["meta"]["asr"]["model"] == "openai/whisper-tiny"
