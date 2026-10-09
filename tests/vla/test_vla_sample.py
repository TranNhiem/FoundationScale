"""Tests for :mod:`foundationscale.vla.sample`, one training sample of spec S2.

``Normalizer`` is checked for its exact affine maps, its clipping and every refusal;
``build_sample`` is checked end to end on a synthetic LeRobot v2.1 dataset whose videos are
"decoded" by an injected fake decoder returning solid-colour frames, so nothing is really
decoded on CPU.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import foundationscale.vla.frames as frames_module
from foundationscale.vla.frames import DECODERS
from foundationscale.vla.lerobot import LeRobotDataset, open_lerobot
from foundationscale.vla.modality import ModalitySpec, load_modality
from foundationscale.vla.norm import DEGENERATE_STD, NormStats
from foundationscale.vla.sample import Normalizer, SampleError, VlaSample, build_sample

LENGTHS: tuple[int, ...] = (4, 5, 6)
STATE_DIM = 4
EXTRA_DIM = 2
ACTION_DIM = 3
GROUP_STATE_DIM = STATE_DIM + EXTRA_DIM
FPS = 30.0
CHUNKS_SIZE = 1000
TASK = "stack the blocks"
STATE_KEY = "observation.state"
EXTRA_KEY = "observation.extra"
ACTION_KEY = "action"
VIDEO_KEY = "observation.images.image"
WRIST_KEY = "observation.images.wrist"
DATA_PATH = "data/chunk-{episode_chunk}/episode_{episode_index}.parquet"
VIDEO_PATH = "videos/chunk-{episode_chunk}/{video_key}/episode_{episode_index}.mp4"
HEIGHT = 4
WIDTH = 5
TINTS: Mapping[str, int] = {VIDEO_KEY: 0, WRIST_KEY: 64}

# The slices in modality.json file order; the expected sample concatenates them in this order.
STATE_PARTS: tuple[tuple[str, int, int], ...] = (
    (STATE_KEY, 0, 3),
    (STATE_KEY, 3, 4),
    (EXTRA_KEY, 0, 2),
)
ACTION_PARTS: tuple[tuple[str, int, int], ...] = ((ACTION_KEY, 0, 2), (ACTION_KEY, 2, 3))

STATE_MEAN = (0.0, 1.0, 2.0, 3.0)
STATE_STD = (1.0, 2.0, 4.0, 8.0)
EXTRA_MEAN = (10.0, 20.0)
EXTRA_STD = (2.0, 5.0)
ACTION_MEAN = (-1.0, 0.5, 2.0)
ACTION_STD = (0.5, 1.5, 3.0)
GROUP_STATE_MEAN = STATE_MEAN + EXTRA_MEAN
GROUP_STATE_STD = STATE_STD + EXTRA_STD

NORM_KEY = "observation.velocity"
NORM_MEAN = (1.0, -2.0, 0.5)
NORM_STD = (2.0, 0.5, 4.0)
NORM_Q01 = (0.0, -3.0, -1.5)
NORM_Q99 = (4.0, -1.0, 6.5)
NORM_MIN = (-3.0, -4.0, -7.5)
NORM_MAX = (5.0, 0.0, 8.5)


def _state(episode: int, length: int) -> np.ndarray:
    """Deterministic float32 state column of dim 4."""
    rows = np.arange(length, dtype=np.float32)[:, None]
    dims = np.arange(STATE_DIM, dtype=np.float32)[None, :]
    return rows * np.float32(0.5) + dims * np.float32(0.25) + np.float32(episode)


def _extra(episode: int, length: int) -> np.ndarray:
    """Deterministic float32 state column of dim 2."""
    rows = np.arange(length, dtype=np.float32)[:, None]
    dims = np.arange(EXTRA_DIM, dtype=np.float32)[None, :]
    return rows * np.float32(-0.25) + dims * np.float32(1.5) + np.float32(episode) * np.float32(0.5)


def _action(episode: int, length: int) -> np.ndarray:
    """Deterministic float32 action column of dim 3."""
    rows = np.arange(length, dtype=np.float32)[:, None]
    dims = np.arange(ACTION_DIM, dtype=np.float32)[None, :]
    return (
        rows * np.float32(0.75) - dims * np.float32(0.5) + np.float32(episode) * np.float32(0.125)
    )


def _features() -> dict[str, Any]:
    return {
        STATE_KEY: {"dtype": "float32", "shape": [STATE_DIM], "names": ["a", "b", "c", "d"]},
        EXTRA_KEY: {"dtype": "float32", "shape": [EXTRA_DIM], "names": ["a", "b"]},
        ACTION_KEY: {"dtype": "float32", "shape": [ACTION_DIM], "names": ["a", "b", "c"]},
        "frame_index": {"dtype": "int64", "shape": [1], "names": ["frame_index"]},
        "timestamp": {"dtype": "float64", "shape": [1], "names": ["timestamp"]},
        "episode_index": {"dtype": "int64", "shape": [1], "names": ["episode_index"]},
        "index": {"dtype": "int64", "shape": [1], "names": ["index"]},
        "task_index": {"dtype": "int64", "shape": [1], "names": ["task_index"]},
        VIDEO_KEY: {
            "dtype": "video",
            "shape": [3, 224, 224],
            "names": ["channel", "height", "width"],
            "info": {"video.height": 224, "video.width": 224},
        },
        WRIST_KEY: {
            "dtype": "video",
            "shape": [3, 224, 224],
            "names": ["channel", "height", "width"],
            "info": {"video.height": 224, "video.width": 224},
        },
    }


def _info() -> dict[str, Any]:
    return {
        "codebase_version": "v2.1",
        "fps": FPS,
        "chunks_size": CHUNKS_SIZE,
        "data_path": DATA_PATH,
        "video_path": VIDEO_PATH,
        "features": _features(),
        "total_episodes": len(LENGTHS),
        "total_frames": sum(LENGTHS),
    }


def _episodes() -> list[dict[str, Any]]:
    return [
        {"episode_index": episode, "length": length, "tasks": [TASK]}
        for episode, length in enumerate(LENGTHS)
    ]


def _modality() -> dict[str, Any]:
    """GR00T modality.json: slices in file order, two videos, the task annotation."""
    return {
        "state": {
            "pos": {"original_key": STATE_KEY, "start": 0, "end": 3},
            "grip": {"original_key": STATE_KEY, "start": 3, "end": 4},
            "extra": {"original_key": EXTRA_KEY, "start": 0, "end": 2},
        },
        "action": {
            "target": {"original_key": ACTION_KEY, "start": 0, "end": 2},
            "grip": {"original_key": ACTION_KEY, "start": 2, "end": 3},
        },
        "video": {
            "image": {"original_key": VIDEO_KEY},
            "wrist": {"original_key": WRIST_KEY},
        },
        "annotation": {"human.action.task_description": {"original_key": "task_index"}},
    }


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _parquet_table(episode: int, length: int) -> pa.Table:
    frame_index = np.arange(length, dtype=np.int64)
    return pa.table(
        {
            STATE_KEY: pa.array(_state(episode, length).tolist(), type=pa.list_(pa.float32())),
            EXTRA_KEY: pa.array(_extra(episode, length).tolist(), type=pa.list_(pa.float32())),
            ACTION_KEY: pa.array(_action(episode, length).tolist(), type=pa.list_(pa.float32())),
            "frame_index": pa.array(frame_index),
            "timestamp": pa.array(frame_index.astype(np.float64) / FPS),
            "episode_index": pa.array(np.full(length, episode, dtype=np.int64)),
            "index": pa.array(np.arange(length, dtype=np.int64) + sum(LENGTHS[:episode])),
            "task_index": pa.array(np.zeros(length, dtype=np.int64)),
        }
    )


def _build(root: Path, modality: Mapping[str, Any] | None = None) -> None:
    """Write a tiny LeRobot v2.1 dataset: 3 episodes of lengths 4, 5 and 6 plus dummy videos."""
    _write_json(root / "meta" / "info.json", _info())
    _write_jsonl(root / "meta" / "episodes.jsonl", _episodes())
    _write_jsonl(root / "meta" / "tasks.jsonl", [{"task_index": 0, "task": TASK}])
    payload = _modality() if modality is None else dict(modality)
    _write_json(root / "meta" / "modality.json", payload)
    for episode, length in enumerate(LENGTHS):
        path = root / "data" / "chunk-0" / f"episode_{episode}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(_parquet_table(episode, length), path)
        for video_key in (VIDEO_KEY, WRIST_KEY):
            video = root / "videos" / "chunk-0" / video_key / f"episode_{episode}.mp4"
            video.parent.mkdir(parents=True, exist_ok=True)
            video.write_bytes(b"not a real mp4")


def _setup(
    tmp_path: Path, modality: Mapping[str, Any] | None = None
) -> tuple[LeRobotDataset, ModalitySpec]:
    _build(tmp_path, modality)
    return open_lerobot(tmp_path), load_modality(tmp_path / "meta" / "modality.json")


def _solid(index: int, tint: int = 0) -> np.ndarray:
    """A solid-colour uint8 RGB frame identifying ``index`` and the video it came from."""
    return np.full((HEIGHT, WIDTH, 3), (index * 17 + tint) % 256, dtype=np.uint8)


class _FakeDecoder:
    """Stand-in decoder: one solid-colour frame per episode frame, no decoding at all."""

    def __init__(self) -> None:
        self.calls: list[Path] = []

    def __call__(self, path: Path) -> list[np.ndarray]:
        self.calls.append(Path(path))
        episode = int(Path(path).stem.split("_")[1])
        tint = TINTS.get(Path(path).parent.name, 0)
        return [_solid(index, tint) for index in range(LENGTHS[episode])]


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


def _norm_stats(
    key: str = NORM_KEY,
    *,
    mean: tuple[float, ...] = NORM_MEAN,
    std: tuple[float, ...] = NORM_STD,
    q01: tuple[float, ...] = NORM_Q01,
    q99: tuple[float, ...] = NORM_Q99,
    minimum: tuple[float, ...] = NORM_MIN,
    maximum: tuple[float, ...] = NORM_MAX,
) -> NormStats:
    """Hand-built NormStats so every normalisation mode has a distinct affine map."""
    return NormStats(
        key=key,
        frames=15,
        episodes=3,
        mean=mean,
        std=std,
        min=minimum,
        max=maximum,
        q01=q01,
        q99=q99,
    )


def _sample_stats(key: str, mean: Sequence[float], std: Sequence[float]) -> NormStats:
    """NormStats with q01/q99 at mean -/+ std and min/max at mean -/+ 2 std."""
    pairs = [(float(m), float(s)) for m, s in zip(mean, std, strict=True)]
    return NormStats(
        key=key,
        frames=sum(LENGTHS),
        episodes=len(LENGTHS),
        mean=tuple(m for m, _ in pairs),
        std=tuple(s for _, s in pairs),
        min=tuple(m - 2.0 * s for m, s in pairs),
        max=tuple(m + 2.0 * s for m, s in pairs),
        q01=tuple(m - s for m, s in pairs),
        q99=tuple(m + s for m, s in pairs),
    )


def _normalizer(mode: Literal["mean_std", "q01_q99", "min_max"] = "mean_std") -> Normalizer:
    """Normalizer with consistent statistics for the columns and the assembled groups.

    ``build_sample`` may normalise each ``original_key`` column or the assembled ``state`` /
    ``action`` groups; the statistics agree per output dim, so the expected sample values do
    not depend on which of the two keys the sample is normalised under.
    """
    stats = {
        STATE_KEY: _sample_stats(STATE_KEY, STATE_MEAN, STATE_STD),
        EXTRA_KEY: _sample_stats(EXTRA_KEY, EXTRA_MEAN, EXTRA_STD),
        ACTION_KEY: _sample_stats(ACTION_KEY, ACTION_MEAN, ACTION_STD),
        "state": _sample_stats("state", GROUP_STATE_MEAN, GROUP_STATE_STD),
    }
    return Normalizer(stats, mode=mode)


def _raw(episode: int, indices: Sequence[int], parts: Sequence[tuple[str, int, int]]) -> np.ndarray:
    """Gather the ``parts`` columns at ``indices`` and concatenate them in file order."""
    length = LENGTHS[episode]
    columns: dict[str, np.ndarray] = {
        STATE_KEY: _state(episode, length),
        EXTRA_KEY: _extra(episode, length),
        ACTION_KEY: _action(episode, length),
    }
    rows = [columns[key][list(indices), start:end] for key, start, end in parts]
    return np.concatenate(rows, axis=1).astype(np.float32)


def _expected_mean_std(raw: np.ndarray, mean: Sequence[float], std: Sequence[float]) -> np.ndarray:
    return (raw.astype(np.float64) - np.array(mean)) / np.array(std)


def _expected_min_max(raw: np.ndarray, mean: Sequence[float], std: Sequence[float]) -> np.ndarray:
    low = np.array(mean) - 2.0 * np.array(std)
    high = np.array(mean) + 2.0 * np.array(std)
    # min_max clips to [-1, 1] (spec S2): the synthetic stats here are not the data's
    # own extremes, so some frames land outside the range and saturate.
    return np.clip(2.0 * (raw.astype(np.float64) - low) / (high - low) - 1.0, -1.0, 1.0)


def test_mean_std_apply_matches_the_formula() -> None:
    norm = Normalizer({NORM_KEY: _norm_stats()}, mode="mean_std")
    x = np.array([[0.0, -2.0, 4.5], [2.0, -1.0, -3.5]], dtype=np.float32)
    expected = (x.astype(np.float64) - np.array(NORM_MEAN)) / np.array(NORM_STD)
    np.testing.assert_allclose(norm.apply(NORM_KEY, x), expected, rtol=1e-6, atol=1e-6)


def test_mean_std_round_trips_within_1e_6() -> None:
    norm = Normalizer({NORM_KEY: _norm_stats()}, mode="mean_std")
    x = np.array([[0.0, -2.0, 4.5], [2.0, -1.0, -3.5], [-7.25, 3.5, 0.0]], dtype=np.float32)
    np.testing.assert_allclose(
        norm.unapply(NORM_KEY, norm.apply(NORM_KEY, x)), x, rtol=0, atol=1e-6
    )


def test_mean_std_round_trips_a_single_frame() -> None:
    norm = Normalizer({NORM_KEY: _norm_stats()}, mode="mean_std")
    x = np.array([0.25, -1.5, 2.0], dtype=np.float32)
    np.testing.assert_allclose(
        norm.unapply(NORM_KEY, norm.apply(NORM_KEY, x)), x, rtol=0, atol=1e-6
    )


def test_q01_q99_maps_the_quantile_range_to_the_unit_interval() -> None:
    norm = Normalizer({NORM_KEY: _norm_stats()}, mode="q01_q99")
    x = np.array([[0.0, -3.0, -1.5], [4.0, -1.0, 6.5], [2.0, -2.0, 2.5]], dtype=np.float32)
    expected = np.array([[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0], [0.0, 0.0, 0.0]])
    np.testing.assert_allclose(norm.apply(NORM_KEY, x), expected, rtol=0, atol=1e-6)


def test_q01_q99_clips_out_of_range_values() -> None:
    norm = Normalizer({NORM_KEY: _norm_stats()}, mode="q01_q99")
    x = np.array([[-100.0, 5.0, -10.0], [100.0, -50.0, 50.0]], dtype=np.float32)
    expected = np.array([[-1.0, 1.0, -1.0], [1.0, -1.0, 1.0]])
    np.testing.assert_array_equal(norm.apply(NORM_KEY, x), expected)


def test_q01_q99_round_trips_inside_the_quantile_range() -> None:
    norm = Normalizer({NORM_KEY: _norm_stats()}, mode="q01_q99")
    x = np.array([[0.0, -3.0, -1.5], [2.0, -2.0, 2.5], [4.0, -1.0, 6.5]], dtype=np.float32)
    np.testing.assert_allclose(
        norm.unapply(NORM_KEY, norm.apply(NORM_KEY, x)), x, rtol=0, atol=1e-6
    )


def test_min_max_maps_the_range_to_the_unit_interval_and_round_trips() -> None:
    norm = Normalizer({NORM_KEY: _norm_stats()}, mode="min_max")
    x = np.array([[-3.0, -4.0, -7.5], [5.0, 0.0, 8.5], [1.0, -2.0, 0.5]], dtype=np.float32)
    expected = np.array([[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0], [0.0, 0.0, 0.0]])
    np.testing.assert_allclose(norm.apply(NORM_KEY, x), expected, rtol=0, atol=1e-6)
    np.testing.assert_allclose(
        norm.unapply(NORM_KEY, norm.apply(NORM_KEY, x)), x, rtol=0, atol=1e-6
    )


def test_unapply_returns_the_clipped_boundary_for_clipped_values() -> None:
    norm = Normalizer({NORM_KEY: _norm_stats()}, mode="q01_q99")
    applied = norm.apply(NORM_KEY, np.array([[-100.0, 5.0, -10.0]], dtype=np.float32))
    restored = norm.unapply(NORM_KEY, applied)
    expected = np.array([[NORM_Q01[0], NORM_Q99[1], NORM_Q01[2]]])
    np.testing.assert_allclose(restored, expected, rtol=0, atol=1e-6)


def test_apply_refuses_an_unknown_key() -> None:
    norm = Normalizer({NORM_KEY: _norm_stats()}, mode="mean_std")
    _refusal(
        ValueError,
        lambda: norm.apply("observation.unknown", np.zeros((1, 3))),
        "observation.unknown",
    )


def test_unapply_refuses_an_unknown_key() -> None:
    norm = Normalizer({NORM_KEY: _norm_stats()}, mode="mean_std")
    _refusal(
        ValueError,
        lambda: norm.unapply("observation.unknown", np.zeros((1, 3))),
        "observation.unknown",
    )


def test_apply_refuses_a_dim_mismatch() -> None:
    norm = Normalizer({NORM_KEY: _norm_stats()}, mode="mean_std")
    _refusal(
        ValueError,
        lambda: norm.apply(NORM_KEY, np.zeros((2, 2), dtype=np.float32)),
        NORM_KEY,
        "dim",
        "3",
        "2",
    )


def test_unapply_refuses_a_dim_mismatch() -> None:
    norm = Normalizer({NORM_KEY: _norm_stats()}, mode="mean_std")
    _refusal(
        ValueError,
        lambda: norm.unapply(NORM_KEY, np.zeros((2, 2), dtype=np.float32)),
        NORM_KEY,
        "dim",
        "3",
        "2",
    )


def test_mean_std_refuses_a_degenerate_std() -> None:
    def _construct_and_apply() -> None:
        stats = _norm_stats(std=(1.0, DEGENERATE_STD / 2.0, 4.0))
        norm = Normalizer({NORM_KEY: stats}, mode="mean_std")
        norm.apply(NORM_KEY, np.zeros((1, 3), dtype=np.float32))

    _refusal(ValueError, _construct_and_apply, NORM_KEY, "std", "1")


def test_mean_std_accepts_a_std_at_the_degenerate_boundary() -> None:
    stats = _norm_stats(std=(1.0, DEGENERATE_STD, 4.0))
    norm = Normalizer({NORM_KEY: stats}, mode="mean_std")
    applied = norm.apply(NORM_KEY, np.zeros((1, 3), dtype=np.float32))
    assert applied.shape == (1, 3)


def test_q01_q99_refuses_a_degenerate_range_naming_the_dims() -> None:
    def _construct_and_apply() -> None:
        stats = _norm_stats(q01=(0.0, -3.0, -1.5), q99=(4.0, -3.0, -1.5))
        norm = Normalizer({NORM_KEY: stats}, mode="q01_q99")
        norm.apply(NORM_KEY, np.zeros((1, 3), dtype=np.float32))

    _refusal(ValueError, _construct_and_apply, NORM_KEY, "1", "2")


def test_min_max_refuses_a_degenerate_range_naming_the_dims() -> None:
    def _construct_and_apply() -> None:
        stats = _norm_stats(minimum=(-3.0, 1.0, -7.5), maximum=(5.0, 1.0, 8.5))
        norm = Normalizer({NORM_KEY: stats}, mode="min_max")
        norm.apply(NORM_KEY, np.zeros((1, 3), dtype=np.float32))

    _refusal(ValueError, _construct_and_apply, NORM_KEY, "1")


def test_build_sample_end_to_end_with_refuse_padding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ds, spec = _setup(tmp_path)
    fake = _FakeDecoder()
    _inject(monkeypatch, fake)
    chunks = _chunks = None  # noqa: F841
    del chunks, _chunks
    from foundationscale.vla.chunking import ChunkSpec

    chunks = ChunkSpec(state_delta=(0, 1), action_delta=(0, 1, 2), video_delta=(0, 1))
    sample = build_sample(ds, spec, chunks, _normalizer(), 1, 2, pad="refuse")
    assert isinstance(sample, VlaSample)
    assert sample.episode_index == 1
    assert sample.anchor == 2
    assert sample.task == TASK
    assert set(sample.images) == {"image", "wrist"}
    for name, key in (("image", VIDEO_KEY), ("wrist", WRIST_KEY)):
        frames = sample.images[name]
        assert isinstance(frames, tuple)
        assert len(frames) == 2
        np.testing.assert_array_equal(frames[0], _solid(2, TINTS[key]))
        np.testing.assert_array_equal(frames[1], _solid(3, TINTS[key]))
    assert sample.state.shape == (2, GROUP_STATE_DIM)
    assert sample.state.dtype == np.float32
    raw_state = _raw(1, [2, 3], STATE_PARTS)
    np.testing.assert_allclose(
        sample.state,
        _expected_mean_std(raw_state, GROUP_STATE_MEAN, GROUP_STATE_STD),
        rtol=0,
        atol=1e-6,
    )
    assert sample.action.shape == (3, ACTION_DIM)
    assert sample.action.dtype == np.float32
    raw_action = _raw(1, [2, 3, 4], ACTION_PARTS)
    np.testing.assert_allclose(
        sample.action,
        _expected_mean_std(raw_action, ACTION_MEAN, ACTION_STD),
        rtol=0,
        atol=1e-6,
    )
    assert sample.action_valid.shape == (3,)
    assert sample.action_valid.dtype == bool
    assert sample.action_valid.tolist() == [True, True, True]


def test_build_sample_marks_padded_action_steps_with_action_valid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from foundationscale.vla.chunking import ChunkSpec

    ds, spec = _setup(tmp_path)
    _inject(monkeypatch, _FakeDecoder())
    chunks = ChunkSpec(state_delta=(0,), action_delta=(0, 1, 2), video_delta=(0,))
    sample = build_sample(ds, spec, chunks, _normalizer(), 0, 2, pad="edge")
    assert sample.action_valid.tolist() == [True, True, False]
    raw = _raw(0, [2, 3, 3], ACTION_PARTS)
    np.testing.assert_allclose(
        sample.action,
        _expected_mean_std(raw, ACTION_MEAN, ACTION_STD),
        rtol=0,
        atol=1e-6,
    )
    tail = build_sample(ds, spec, chunks, _normalizer(), 0, 3, pad="edge")
    assert tail.action_valid.tolist() == [True, False, False]
    np.testing.assert_allclose(
        tail.action,
        _expected_mean_std(_raw(0, [3, 3, 3], ACTION_PARTS), ACTION_MEAN, ACTION_STD),
        rtol=0,
        atol=1e-6,
    )


def test_build_sample_edge_padding_clamps_state_and_images(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from foundationscale.vla.chunking import ChunkSpec

    ds, spec = _setup(tmp_path)
    _inject(monkeypatch, _FakeDecoder())
    chunks = ChunkSpec(state_delta=(0, 1), action_delta=(0, 1), video_delta=(0, 1))
    sample = build_sample(ds, spec, chunks, _normalizer(), 0, 3, pad="edge")
    np.testing.assert_array_equal(sample.state[0], sample.state[1])
    np.testing.assert_array_equal(sample.images["image"][0], sample.images["image"][1])
    np.testing.assert_array_equal(sample.images["wrist"][0], sample.images["wrist"][1])
    np.testing.assert_array_equal(sample.images["image"][0], _solid(3, TINTS[VIDEO_KEY]))
    np.testing.assert_array_equal(sample.images["wrist"][0], _solid(3, TINTS[WRIST_KEY]))
    assert sample.action_valid.tolist() == [True, False]


def test_build_sample_refuses_an_anchor_outside_the_refuse_range(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from foundationscale.vla.chunking import ChunkSpec, valid_anchor_indices

    ds, spec = _setup(tmp_path)
    _inject(monkeypatch, _FakeDecoder())
    chunks = ChunkSpec(state_delta=(0,), action_delta=(0, 1, 2), video_delta=(0,))
    assert tuple(valid_anchor_indices(LENGTHS[0], chunks, pad="refuse")) == (0, 1)
    _refusal(
        SampleError,
        lambda: build_sample(ds, spec, chunks, _normalizer(), 0, 2, pad="refuse"),
        "anchor",
        "2",
    )
    allowed = build_sample(ds, spec, chunks, _normalizer(), 0, 2, pad="edge")
    assert allowed.anchor == 2


def test_build_sample_refuses_a_missing_task_annotation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from foundationscale.vla.chunking import ChunkSpec

    modality = _modality()
    modality["annotation"] = {"human.action.task_description": {"original_key": "episode_index"}}
    ds, spec = _setup(tmp_path, modality)
    _inject(monkeypatch, _FakeDecoder())
    chunks = ChunkSpec(state_delta=(0,), action_delta=(0, 1), video_delta=(0,))
    _refusal(
        SampleError,
        lambda: build_sample(ds, spec, chunks, _normalizer(), 0, 0, pad="refuse"),
        "task_index",
    )


def test_build_sample_normalises_with_the_declared_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from foundationscale.vla.chunking import ChunkSpec

    ds, spec = _setup(tmp_path)
    _inject(monkeypatch, _FakeDecoder())
    chunks = ChunkSpec(state_delta=(0,), action_delta=(0, 1, 2), video_delta=(0,))
    sample = build_sample(ds, spec, chunks, _normalizer("min_max"), 2, 1, pad="refuse")
    raw = _raw(2, [1, 2, 3], ACTION_PARTS)
    np.testing.assert_allclose(
        sample.action,
        _expected_min_max(raw, ACTION_MEAN, ACTION_STD),
        rtol=0,
        atol=1e-6,
    )
    raw_state = _raw(2, [1], STATE_PARTS)
    np.testing.assert_allclose(
        sample.state,
        _expected_min_max(raw_state, GROUP_STATE_MEAN, GROUP_STATE_STD),
        rtol=0,
        atol=1e-6,
    )
