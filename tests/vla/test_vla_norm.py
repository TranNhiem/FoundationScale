"""Tests for :mod:`foundationscale.vla.norm`, the normalisation statistics module.

Every refusal listed in spec S1 gets its own test that asserts the exception type and a
message substring naming the offending key, stat name or observed value; the statistics
themselves are checked for exact equality against numpy on the concatenated frames.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from foundationscale.vla.lerobot import LeRobotFormatError, open_lerobot
from foundationscale.vla.norm import (
    DEGENERATE_STD,
    STAT_NAMES,
    NormStats,
    compute_norm_stats,
    stats_digest,
    stats_from_json,
    stats_to_json,
)

LENGTHS: tuple[int, ...] = (4, 5, 6)
STATE_DIM = 4
ACTION_DIM = 3
FPS = 30.0
CHUNKS_SIZE = 1000
TASK = "stack the blocks"
STATE_KEY = "observation.state"
ACTION_KEY = "action"
VIDEO_KEY = "observation.images.image"
DATA_PATH = "data/chunk-{episode_chunk}/episode_{episode_index}.parquet"
VIDEO_PATH = "videos/chunk-{episode_chunk}/{video_key}/episode_{episode_index}.mp4"


def _state(episode: int, length: int) -> np.ndarray:
    """Deterministic float32 state; dim 2 is constant so it is a degenerate dim."""
    rows = np.arange(length, dtype=np.float32)[:, None]
    dims = np.arange(STATE_DIM, dtype=np.float32)[None, :]
    values = rows * np.float32(0.5) + dims * np.float32(0.25) + np.float32(episode)
    values[:, 2] = np.float32(1.0)
    return values


def _action(episode: int, length: int) -> np.ndarray:
    """Deterministic float32 action with no degenerate dims."""
    rows = np.arange(length, dtype=np.float32)[:, None]
    dims = np.arange(ACTION_DIM, dtype=np.float32)[None, :]
    values = rows * np.float32(0.75) - dims * np.float32(0.5)
    return values + np.float32(episode) * np.float32(0.125)


def _features() -> dict[str, Any]:
    return {
        STATE_KEY: {"dtype": "float32", "shape": [STATE_DIM], "names": ["a", "b", "c", "d"]},
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


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _parquet_table(episode: int, length: int) -> pa.Table:
    frame_index = np.arange(length, dtype=np.int64)
    state = pa.array(_state(episode, length).tolist(), type=pa.list_(pa.float32()))
    action = pa.array(_action(episode, length).tolist(), type=pa.list_(pa.float32()))
    return pa.table(
        {
            STATE_KEY: state,
            ACTION_KEY: action,
            "frame_index": pa.array(frame_index),
            "timestamp": pa.array(frame_index.astype(np.float64) / FPS),
            "episode_index": pa.array(np.full(length, episode, dtype=np.int64)),
            "index": pa.array(np.arange(length, dtype=np.int64) + sum(LENGTHS[:episode])),
            "task_index": pa.array(np.zeros(length, dtype=np.int64)),
        }
    )


def _build(root: Path) -> None:
    """Write a tiny LeRobot v2.1 dataset: 3 episodes of lengths 4, 5 and 6."""
    _write_json(root / "meta" / "info.json", _info())
    _write_jsonl(root / "meta" / "episodes.jsonl", _episodes())
    _write_jsonl(root / "meta" / "tasks.jsonl", [{"task_index": 0, "task": TASK}])
    for episode, length in enumerate(LENGTHS):
        path = root / "data" / "chunk-0" / f"episode_{episode}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(_parquet_table(episode, length), path)
        video = root / "videos" / "chunk-0" / VIDEO_KEY / f"episode_{episode}.mp4"
        video.parent.mkdir(parents=True, exist_ok=True)
        video.write_bytes(b"not a real mp4")


def _edit_info(root: Path, mutate: Callable[[dict[str, Any]], None]) -> None:
    path = root / "meta" / "info.json"
    info: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    mutate(info)
    _write_json(path, info)


def _refusal(expected: type[Exception], fn: Callable[[], object], *needles: str) -> Exception:
    """Run ``fn`` and assert it raises ``expected`` naming every ``needle`` in the message."""
    with pytest.raises(expected) as excinfo:
        fn()
    error = excinfo.value
    message = str(error)
    for needle in needles:
        assert needle in message, f"expected {needle!r} in error message {message!r}"
    return error


def _frames(key: str, episodes: tuple[int, ...] = (0, 1, 2)) -> np.ndarray:
    """Concatenate the float32 frames of ``key`` for ``episodes`` in episode order."""
    builder = _state if key == STATE_KEY else _action
    return np.concatenate([builder(episode, LENGTHS[episode]) for episode in episodes], axis=0)


def _expected(values: np.ndarray) -> dict[str, tuple[float, ...]]:
    """Reference statistics computed with numpy on the concatenated float32 frames."""
    return {
        "mean": tuple(float(value) for value in values.mean(axis=0)),
        "std": tuple(float(value) for value in values.std(axis=0)),
        "min": tuple(float(value) for value in values.min(axis=0)),
        "max": tuple(float(value) for value in values.max(axis=0)),
        "q01": tuple(float(value) for value in np.quantile(values, 0.01, axis=0)),
        "q99": tuple(float(value) for value in np.quantile(values, 0.99, axis=0)),
    }


def _assert_matches_numpy(stats: NormStats, values: np.ndarray) -> None:
    expected = _expected(values)
    for name in STAT_NAMES:
        assert getattr(stats, name) == expected[name], name
    assert stats.frames == values.shape[0]
    assert stats.dim == values.shape[1]


def _stats_payload() -> dict[str, Any]:
    """A minimal valid stats document in the flat NormStats schema."""
    return {
        ACTION_KEY: {
            "key": ACTION_KEY,
            "frames": 15,
            "episodes": 3,
            "mean": [0.0, 0.0, 0.0],
            "std": [1.0, 1.0, 1.0],
            "min": [-1.0, -1.0, -1.0],
            "max": [1.0, 1.0, 1.0],
            "q01": [-1.0, -1.0, -1.0],
            "q99": [1.0, 1.0, 1.0],
        }
    }


def test_compute_norm_stats_matches_numpy(tmp_path: Path) -> None:
    _build(tmp_path)
    ds = open_lerobot(tmp_path)
    stats = compute_norm_stats(ds, [STATE_KEY, ACTION_KEY])
    assert set(stats) == {STATE_KEY, ACTION_KEY}
    for key in (STATE_KEY, ACTION_KEY):
        assert stats[key].key == key
        assert stats[key].episodes == 3
        _assert_matches_numpy(stats[key], _frames(key))


def test_compute_norm_stats_episode_subset(tmp_path: Path) -> None:
    _build(tmp_path)
    ds = open_lerobot(tmp_path)
    stats = compute_norm_stats(ds, [STATE_KEY, ACTION_KEY], episodes=[1, 2])
    for key in (STATE_KEY, ACTION_KEY):
        assert stats[key].frames == 11
        assert stats[key].episodes == 2
        _assert_matches_numpy(stats[key], _frames(key, (1, 2)))


def test_degenerate_dims_detect_constant_column(tmp_path: Path) -> None:
    _build(tmp_path)
    ds = open_lerobot(tmp_path)
    stats = compute_norm_stats(ds, [STATE_KEY, ACTION_KEY])
    state = stats[STATE_KEY]
    assert state.degenerate_dims() == (2,)
    assert state.std[2] < DEGENERATE_STD
    assert state.std[2] == 0.0
    assert all(state.std[dim] >= DEGENERATE_STD for dim in (0, 1, 3))
    assert stats[ACTION_KEY].degenerate_dims() == ()


def test_stats_json_round_trip_is_exact(tmp_path: Path) -> None:
    _build(tmp_path)
    ds = open_lerobot(tmp_path)
    stats = compute_norm_stats(ds, [STATE_KEY, ACTION_KEY])
    assert stats_from_json(stats_to_json(stats)) == stats


def test_stats_from_json_reads_document() -> None:
    stats = stats_from_json(_stats_payload())
    assert set(stats) == {ACTION_KEY}
    assert stats[ACTION_KEY].key == ACTION_KEY
    assert stats[ACTION_KEY].frames == 15
    assert stats[ACTION_KEY].episodes == 3
    assert stats[ACTION_KEY].dim == 3
    assert stats[ACTION_KEY].mean == (0.0, 0.0, 0.0)
    assert stats[ACTION_KEY].q99 == (1.0, 1.0, 1.0)


def test_stats_digest_stable_and_sensitive(tmp_path: Path) -> None:
    _build(tmp_path)
    ds = open_lerobot(tmp_path)
    first = compute_norm_stats(ds, [ACTION_KEY])
    second = compute_norm_stats(ds, [ACTION_KEY])
    digest = stats_digest(first)
    assert digest == stats_digest(second)
    assert re.fullmatch(r"[0-9a-f]{64}", digest) is not None
    mean = first[ACTION_KEY].mean
    changed = dict(first)
    changed[ACTION_KEY] = replace(first[ACTION_KEY], mean=(mean[0] + 1e-3, *mean[1:]))
    assert stats_digest(changed) != digest


def test_compute_norm_stats_refuses_non_float_key(tmp_path: Path) -> None:
    _build(tmp_path)
    ds = open_lerobot(tmp_path)
    _refusal(LeRobotFormatError, lambda: compute_norm_stats(ds, ["frame_index"]), "frame_index")


def test_compute_norm_stats_refuses_non_1d_key(tmp_path: Path) -> None:
    _build(tmp_path)

    def _flatten_action(info: dict[str, Any]) -> None:
        info["features"][ACTION_KEY]["shape"] = [3, 1]

    _edit_info(tmp_path, _flatten_action)
    ds = open_lerobot(tmp_path)
    _refusal(LeRobotFormatError, lambda: compute_norm_stats(ds, [ACTION_KEY]), ACTION_KEY)


def test_compute_norm_stats_refuses_empty_keys(tmp_path: Path) -> None:
    _build(tmp_path)
    ds = open_lerobot(tmp_path)
    _refusal(ValueError, lambda: compute_norm_stats(ds, []), "keys")


def test_compute_norm_stats_refuses_empty_episode_selection(tmp_path: Path) -> None:
    _build(tmp_path)
    ds = open_lerobot(tmp_path)
    _refusal(ValueError, lambda: compute_norm_stats(ds, [ACTION_KEY], episodes=[]), "episodes")


def test_compute_norm_stats_refuses_unknown_episode(tmp_path: Path) -> None:
    _build(tmp_path)
    ds = open_lerobot(tmp_path)
    _refusal(ValueError, lambda: compute_norm_stats(ds, [ACTION_KEY], episodes=[99]), "99")


def test_stats_from_json_refuses_missing_stat_name() -> None:
    payload = _stats_payload()
    payload[ACTION_KEY].pop("q01")
    _refusal(ValueError, lambda: stats_from_json(payload), "q01")


def test_stats_from_json_refuses_length_mismatch() -> None:
    payload = _stats_payload()
    payload[ACTION_KEY]["std"] = [1.0, 1.0]
    _refusal(ValueError, lambda: stats_from_json(payload), "std", "2", "3")


def test_stats_from_json_refuses_non_finite_value() -> None:
    payload = _stats_payload()
    payload[ACTION_KEY]["mean"][1] = float("nan")
    error = _refusal(ValueError, lambda: stats_from_json(payload), "mean")
    assert "nan" in str(error).lower()


# Review-found refusals: shapes and dtypes that are not a 1-D float declaration.


def test_a_bare_int_shape_is_not_a_1d_feature(tmp_path: Path) -> None:
    _build(tmp_path)
    _edit_info(tmp_path, lambda info: info["features"][STATE_KEY].update(shape=STATE_DIM))
    ds = open_lerobot(tmp_path)
    _refusal(LeRobotFormatError, lambda: compute_norm_stats(ds, [STATE_KEY]), "1-D float")


def test_a_non_string_dtype_is_refused_not_a_type_error(tmp_path: Path) -> None:
    _build(tmp_path)
    _edit_info(tmp_path, lambda info: info["features"][STATE_KEY].update(dtype=["float32"]))
    ds = open_lerobot(tmp_path)
    _refusal(LeRobotFormatError, lambda: compute_norm_stats(ds, [STATE_KEY]), "1-D float")
