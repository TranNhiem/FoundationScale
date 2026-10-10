"""Tests for :mod:`foundationscale.vla.lerobot`, the LeRobot v2 reader.

Every refusal listed in spec S1 gets its own test that asserts the exception type and a
message substring naming the offending file, key, episode or observed value.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from foundationscale.vla.lerobot import (
    LeRobotFormatError,
    open_lerobot,
    read_episode_columns,
)

LENGTHS: tuple[int, ...] = (4, 5, 6)
STATE_DIM = 4
ACTION_DIM = 3
FPS = 30.0
CHUNKS_SIZE = 1000
TASK = "stack the blocks"
UNKNOWN_TASK = "wave hello"
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


def _parquet_table(
    episode: int,
    length: int,
    *,
    state: pa.Array | None = None,
    frame_index: np.ndarray | None = None,
    timestamp: np.ndarray | None = None,
) -> pa.Table:
    if frame_index is None:
        frame_index = np.arange(length, dtype=np.int64)
    if timestamp is None:
        timestamp = frame_index.astype(np.float64) / FPS
    if state is None:
        state = pa.array(_state(episode, length).tolist(), type=pa.list_(pa.float32()))
    action = pa.array(_action(episode, length).tolist(), type=pa.list_(pa.float32()))
    return pa.table(
        {
            STATE_KEY: state,
            ACTION_KEY: action,
            "frame_index": pa.array(frame_index),
            "timestamp": pa.array(timestamp),
            "episode_index": pa.array(np.full(length, episode, dtype=np.int64)),
            "index": pa.array(np.arange(length, dtype=np.int64) + sum(LENGTHS[:episode])),
            "task_index": pa.array(np.zeros(length, dtype=np.int64)),
        }
    )


def _write_parquet(root: Path, episode: int, table: pa.Table) -> Path:
    path = root / "data" / "chunk-0" / f"episode_{episode}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)
    return path


def _build(root: Path) -> None:
    """Write a tiny LeRobot v2.1 dataset: 3 episodes of lengths 4, 5 and 6."""
    _write_json(root / "meta" / "info.json", _info())
    _write_jsonl(root / "meta" / "episodes.jsonl", _episodes())
    _write_jsonl(root / "meta" / "tasks.jsonl", [{"task_index": 0, "task": TASK}])
    for episode, length in enumerate(LENGTHS):
        _write_parquet(root, episode, _parquet_table(episode, length))
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


def test_open_lerobot_reads_metadata(tmp_path: Path) -> None:
    _build(tmp_path)
    ds = open_lerobot(tmp_path)
    assert ds.root == tmp_path
    assert ds.codebase_version == "v2.1"
    assert ds.fps == FPS
    assert ds.chunks_size == CHUNKS_SIZE
    assert ds.data_path == DATA_PATH
    assert ds.video_path == VIDEO_PATH
    assert ds.video_keys() == (VIDEO_KEY,)
    assert [episode.index for episode in ds.episodes] == [0, 1, 2]
    assert [episode.length for episode in ds.episodes] == list(LENGTHS)
    assert ds.episodes[0].tasks == (TASK,)
    assert dict(ds.tasks) == {0: TASK}
    assert ds.total_frames == 15
    assert ds.episode(2).length == 6
    assert ds.parquet_path(1) == tmp_path / "data" / "chunk-0" / "episode_1.parquet"
    assert ds.video_file(1, VIDEO_KEY) == (
        tmp_path / "videos" / "chunk-0" / VIDEO_KEY / "episode_1.mp4"
    )


def test_open_lerobot_without_video_features_has_no_video_path(tmp_path: Path) -> None:
    _build(tmp_path)

    def _drop_video(info: dict[str, Any]) -> None:
        info["features"].pop(VIDEO_KEY)
        info.pop("video_path")

    _edit_info(tmp_path, _drop_video)
    ds = open_lerobot(tmp_path)
    assert ds.video_path is None
    assert ds.video_keys() == ()


@pytest.mark.parametrize(
    "relative",
    ["meta/info.json", "meta/episodes.jsonl", "meta/tasks.jsonl"],
)
def test_open_lerobot_refuses_missing_meta_file(tmp_path: Path, relative: str) -> None:
    _build(tmp_path)
    (tmp_path / relative).unlink()
    _refusal(LeRobotFormatError, lambda: open_lerobot(tmp_path), relative)


def test_open_lerobot_refuses_unsupported_codebase_version(tmp_path: Path) -> None:
    _build(tmp_path)
    _edit_info(tmp_path, lambda info: info.update(codebase_version="v3.0"))
    _refusal(
        LeRobotFormatError,
        lambda: open_lerobot(tmp_path),
        "v3.0",
        "v3 stores many episodes per file",
    )


def test_open_lerobot_refuses_missing_fps(tmp_path: Path) -> None:
    _build(tmp_path)
    _edit_info(tmp_path, lambda info: info.pop("fps"))
    _refusal(LeRobotFormatError, lambda: open_lerobot(tmp_path), "fps")


def test_open_lerobot_refuses_non_positive_fps(tmp_path: Path) -> None:
    _build(tmp_path)
    _edit_info(tmp_path, lambda info: info.update(fps=0.0))
    _refusal(LeRobotFormatError, lambda: open_lerobot(tmp_path), "fps")


def test_open_lerobot_refuses_missing_chunks_size(tmp_path: Path) -> None:
    _build(tmp_path)
    _edit_info(tmp_path, lambda info: info.pop("chunks_size"))
    _refusal(LeRobotFormatError, lambda: open_lerobot(tmp_path), "chunks_size")


def test_open_lerobot_refuses_non_positive_chunks_size(tmp_path: Path) -> None:
    _build(tmp_path)
    _edit_info(tmp_path, lambda info: info.update(chunks_size=-1))
    _refusal(LeRobotFormatError, lambda: open_lerobot(tmp_path), "chunks_size")


def test_open_lerobot_refuses_missing_data_path(tmp_path: Path) -> None:
    _build(tmp_path)
    _edit_info(tmp_path, lambda info: info.pop("data_path"))
    _refusal(LeRobotFormatError, lambda: open_lerobot(tmp_path), "data_path")


def test_open_lerobot_refuses_data_path_without_episode_chunk(tmp_path: Path) -> None:
    _build(tmp_path)
    _edit_info(tmp_path, lambda info: info.update(data_path="data/episode_{episode_index}.parquet"))
    _refusal(LeRobotFormatError, lambda: open_lerobot(tmp_path), "episode_chunk")


def test_open_lerobot_refuses_data_path_without_episode_index(tmp_path: Path) -> None:
    _build(tmp_path)
    _edit_info(tmp_path, lambda info: info.update(data_path="data/chunk-{episode_chunk}/e.parquet"))
    _refusal(LeRobotFormatError, lambda: open_lerobot(tmp_path), "episode_index")


def test_open_lerobot_refuses_missing_video_path_with_video_features(tmp_path: Path) -> None:
    _build(tmp_path)
    _edit_info(tmp_path, lambda info: info.pop("video_path"))
    _refusal(LeRobotFormatError, lambda: open_lerobot(tmp_path), "video_path")


def test_open_lerobot_refuses_video_path_without_video_key(tmp_path: Path) -> None:
    _build(tmp_path)
    _edit_info(
        tmp_path,
        lambda info: info.update(video_path="videos/chunk-{episode_chunk}/episode_{episode_index}"),
    )
    _refusal(LeRobotFormatError, lambda: open_lerobot(tmp_path), "video_key")


def test_open_lerobot_refuses_empty_episodes_jsonl(tmp_path: Path) -> None:
    _build(tmp_path)
    _write_jsonl(tmp_path / "meta" / "episodes.jsonl", [])
    _refusal(LeRobotFormatError, lambda: open_lerobot(tmp_path), "meta/episodes.jsonl")


def test_open_lerobot_refuses_duplicate_episode_index(tmp_path: Path) -> None:
    _build(tmp_path)
    rows = _episodes()
    rows[2]["episode_index"] = 1
    _write_jsonl(tmp_path / "meta" / "episodes.jsonl", rows)
    _refusal(LeRobotFormatError, lambda: open_lerobot(tmp_path), "episode_index", "1")


def test_open_lerobot_refuses_non_positive_episode_length(tmp_path: Path) -> None:
    _build(tmp_path)
    rows = _episodes()
    rows[1]["length"] = 0
    _write_jsonl(tmp_path / "meta" / "episodes.jsonl", rows)
    _refusal(LeRobotFormatError, lambda: open_lerobot(tmp_path), "length", "0")


def test_open_lerobot_refuses_total_episodes_mismatch(tmp_path: Path) -> None:
    _build(tmp_path)
    _edit_info(tmp_path, lambda info: info.update(total_episodes=4))
    _refusal(LeRobotFormatError, lambda: open_lerobot(tmp_path), "total_episodes", "4", "3")


def test_open_lerobot_refuses_total_frames_mismatch(tmp_path: Path) -> None:
    _build(tmp_path)
    _edit_info(tmp_path, lambda info: info.update(total_frames=16))
    _refusal(LeRobotFormatError, lambda: open_lerobot(tmp_path), "total_frames", "16", "15")


def test_open_lerobot_refuses_task_missing_from_tasks_jsonl(tmp_path: Path) -> None:
    _build(tmp_path)
    rows = _episodes()
    rows[1]["tasks"] = [UNKNOWN_TASK]
    _write_jsonl(tmp_path / "meta" / "episodes.jsonl", rows)
    _refusal(
        LeRobotFormatError,
        lambda: open_lerobot(tmp_path),
        UNKNOWN_TASK,
        "meta/tasks.jsonl",
    )


def test_episode_refuses_unknown_index(tmp_path: Path) -> None:
    _build(tmp_path)
    ds = open_lerobot(tmp_path)
    _refusal(LeRobotFormatError, lambda: ds.episode(99), "99")


def test_video_file_refuses_non_video_key(tmp_path: Path) -> None:
    _build(tmp_path)
    ds = open_lerobot(tmp_path)
    _refusal(LeRobotFormatError, lambda: ds.video_file(0, STATE_KEY), STATE_KEY)


def test_read_episode_columns_returns_only_requested_columns(tmp_path: Path) -> None:
    _build(tmp_path)
    ds = open_lerobot(tmp_path)
    columns = [STATE_KEY, ACTION_KEY, "frame_index", "timestamp"]
    out = read_episode_columns(ds, 1, columns)
    assert set(out) == set(columns)
    assert out[STATE_KEY].shape == (5, STATE_DIM)
    assert out[STATE_KEY].dtype == np.float32
    assert np.array_equal(out[STATE_KEY], _state(1, 5))
    assert out[ACTION_KEY].shape == (5, ACTION_DIM)
    assert np.array_equal(out[ACTION_KEY], _action(1, 5))
    assert out["frame_index"].shape == (5,)
    assert out["frame_index"].dtype == np.int64
    assert np.array_equal(out["frame_index"], np.arange(5, dtype=np.int64))
    assert out["timestamp"].dtype == np.float64


def test_read_episode_columns_refuses_missing_parquet(tmp_path: Path) -> None:
    _build(tmp_path)
    (tmp_path / "data" / "chunk-0" / "episode_0.parquet").unlink()
    ds = open_lerobot(tmp_path)
    _refusal(LeRobotFormatError, lambda: read_episode_columns(ds, 0, [STATE_KEY]), "episode_0")


def test_read_episode_columns_refuses_absent_column(tmp_path: Path) -> None:
    _build(tmp_path)
    ds = open_lerobot(tmp_path)
    _refusal(
        LeRobotFormatError,
        lambda: read_episode_columns(ds, 0, ["observation.missing"]),
        "observation.missing",
        STATE_KEY,
    )


def test_read_episode_columns_refuses_row_count_mismatch(tmp_path: Path) -> None:
    _build(tmp_path)
    _write_parquet(tmp_path, 0, _parquet_table(0, 3))
    ds = open_lerobot(tmp_path)
    _refusal(LeRobotFormatError, lambda: read_episode_columns(ds, 0, [STATE_KEY]), "3", "4")


def test_read_episode_columns_refuses_wrong_frame_index(tmp_path: Path) -> None:
    _build(tmp_path)
    table = _parquet_table(0, 4, frame_index=np.array([0, 1, 2, 4], dtype=np.int64))
    _write_parquet(tmp_path, 0, table)
    ds = open_lerobot(tmp_path)
    _refusal(LeRobotFormatError, lambda: read_episode_columns(ds, 0, [STATE_KEY]), "frame_index")


def test_read_episode_columns_refuses_non_increasing_timestamp(tmp_path: Path) -> None:
    _build(tmp_path)
    stamps = np.array([0.0, 1.0 / FPS, 1.0 / FPS, 3.0 / FPS])
    _write_parquet(tmp_path, 0, _parquet_table(0, 4, timestamp=stamps))
    ds = open_lerobot(tmp_path)
    _refusal(LeRobotFormatError, lambda: read_episode_columns(ds, 0, [STATE_KEY]), "timestamp")


def test_read_episode_columns_refuses_timestamp_drift(tmp_path: Path) -> None:
    _build(tmp_path)
    stamps = np.arange(4, dtype=np.float64) / FPS
    stamps[2] += 2.0 / FPS
    _write_parquet(tmp_path, 0, _parquet_table(0, 4, timestamp=stamps))
    ds = open_lerobot(tmp_path)
    _refusal(
        LeRobotFormatError,
        lambda: read_episode_columns(ds, 0, [STATE_KEY]),
        "timestamp",
        "2",
    )


def test_read_episode_columns_refuses_ragged_rows(tmp_path: Path) -> None:
    _build(tmp_path)
    ragged = pa.array(
        [[0.0, 0.25, 1.0, 0.5], [0.5, 0.75, 1.0], [1.0, 1.25, 1.0, 1.5], [1.5, 1.75, 1.0, 2.0]],
        type=pa.list_(pa.float32()),
    )
    _write_parquet(tmp_path, 0, _parquet_table(0, 4, state=ragged))
    ds = open_lerobot(tmp_path)
    _refusal(LeRobotFormatError, lambda: read_episode_columns(ds, 0, [STATE_KEY]), STATE_KEY)


# Review-found refusals: malformed values that must raise the named error, never a
# bare TypeError/AttributeError/UnicodeDecodeError, and present-but-null totals.


def test_open_lerobot_refuses_an_unhashable_codebase_version(tmp_path: Path) -> None:
    _build(tmp_path)
    _edit_info(tmp_path, lambda info: info.update(codebase_version=["v3.0"]))
    _refusal(LeRobotFormatError, lambda: open_lerobot(tmp_path), "codebase_version")


def test_a_template_with_attribute_access_is_a_format_error(tmp_path: Path) -> None:
    _build(tmp_path)
    _edit_info(
        tmp_path,
        lambda info: info.update(data_path="data/{episode_chunk}/{episode_index.real.x}.parquet"),
    )
    ds = open_lerobot(tmp_path)
    _refusal(LeRobotFormatError, lambda: ds.parquet_path(0), "could not be formatted")


def test_info_json_that_is_not_utf8_is_a_format_error(tmp_path: Path) -> None:
    _build(tmp_path)
    (tmp_path / "meta" / "info.json").write_bytes(b'{"fps": \xe9}')
    _refusal(LeRobotFormatError, lambda: open_lerobot(tmp_path), "info.json")


def test_episodes_jsonl_that_is_not_utf8_is_a_format_error(tmp_path: Path) -> None:
    _build(tmp_path)
    (tmp_path / "meta" / "episodes.jsonl").write_bytes(b'{"episode_index": \xe9}\n')
    _refusal(LeRobotFormatError, lambda: open_lerobot(tmp_path), "not UTF-8")


@pytest.mark.parametrize("key", ["total_episodes", "total_frames"])
def test_a_null_declared_total_is_refused_not_skipped(tmp_path: Path, key: str) -> None:
    _build(tmp_path)
    _edit_info(tmp_path, lambda info: info.update({key: None}))
    _refusal(LeRobotFormatError, lambda: open_lerobot(tmp_path), key)
