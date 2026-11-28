"""Safe fixes from the 2026-11-28 FoundationScale-wide review (each pins one verified defect)."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from foundationscale.checkpoint.dcp import CheckpointFormatError, checked_member_path
from foundationscale.train.audio import AudioLoadError, _mono_float32
from foundationscale.upstream.nemo._io import write_text_atomic
from foundationscale.verify import parity


@pytest.mark.parametrize(
    "name", ["/etc/passwd", "../outside.safetensors", "a/../../b", "x\x00y", "", 7]
)
def test_checkpoint_member_paths_must_stay_inside_the_checkpoint(name) -> None:
    with pytest.raises(CheckpointFormatError, match="member path"):
        checked_member_path(name, path="/ckpt", what="weight_map['w']")


def test_plain_relative_member_paths_are_kept() -> None:
    assert checked_member_path("model-00001-of-00002.safetensors", path="/c", what="w") == (
        "model-00001-of-00002.safetensors"
    )
    assert checked_member_path("__0_0.distcp", path="/c", what="w") == "__0_0.distcp"


def test_a_nan_cosine_never_satisfies_the_cosine_tolerance() -> None:
    tol = parity._DEFAULT_TOLERANCES
    common = dict(bitwise=False, mismatched=1, max_abs_diff=0.0, rel_frob=0.0, tolerances=tol)
    assert parity._classify(cosine=math.nan, **common) is parity.ParityStatus.DIFFER
    assert parity._classify(cosine=1.0, **common) is parity.ParityStatus.CLOSE
    assert (
        parity._classify(cosine=None, **common) is parity.ParityStatus.CLOSE
    )  # no angle: abs/frob decide


def test_stereo_int16_is_scaled_before_channels_are_averaged() -> None:
    stereo = np.array([[32767, 32767], [-32768, -32768]], dtype=np.int16)
    out = _mono_float32(stereo, "r")
    assert out.dtype == np.float32 and float(np.abs(out).max()) <= 1.0 + 1e-6


def test_unsigned_pcm_is_centred_like_soundfile() -> None:
    out = _mono_float32(np.array([0, 128, 255], dtype=np.uint8), "r")
    assert out[1] == pytest.approx(0.0) and out[0] == pytest.approx(-1.0)


@pytest.mark.parametrize("bad", [math.nan, math.inf])
def test_non_finite_audio_is_refused(bad) -> None:
    with pytest.raises(AudioLoadError):
        _mono_float32(np.array([0.0, bad], dtype=np.float32), "r")


def test_atomic_write_replaces_whole_file_and_leaves_no_temp(tmp_path: Path) -> None:
    target = tmp_path / "result.json"
    target.write_text("old")
    write_text_atomic(target, '{"new": 1}')
    assert target.read_text() == '{"new": 1}'
    assert [p.name for p in tmp_path.iterdir()] == ["result.json"]


@pytest.mark.parametrize("bad", [math.nan, math.inf])
def test_level3_judge_refuses_a_non_finite_measurement(bad) -> None:
    from foundationscale.upstream.levels import judge, level3_plan

    step = next(s for s in level3_plan() if s.model_id == "canary-qwen-2.5b")
    with pytest.raises(ValueError, match="unmeasured"):
        judge(step, bad)


def test_unknown_sdp_backend_is_a_named_refusal_not_a_keyerror() -> None:
    from foundationscale.train.sdp_backend import sdp_pin_refusal_reason

    reason = sdp_pin_refusal_reason("no-such-backend", object(), "2.9.0")
    assert reason is not None and "unknown" in reason and "96" in reason


def test_atomic_write_failure_leaves_the_old_file_and_no_temp(tmp_path: Path, monkeypatch) -> None:
    import foundationscale.upstream.nemo._io as io_mod

    target = tmp_path / "result.json"
    target.write_text("old")

    def boom(self, dst):
        raise OSError("replace failed")

    monkeypatch.setattr(io_mod.Path, "replace", boom)
    with pytest.raises(OSError, match="replace failed"):
        write_text_atomic(target, "new")
    assert target.read_text() == "old"
    assert [p.name for p in tmp_path.iterdir()] == ["result.json"]


def test_manifest_dir_fsync_tolerates_an_unopenable_directory(tmp_path: Path) -> None:
    from foundationscale.provenance.manifest import _fsync_dir

    _fsync_dir(tmp_path / "does-not-exist")  # best effort: never raises
    _fsync_dir(tmp_path)
