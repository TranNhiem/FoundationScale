"""Tests for :mod:`foundationscale.vla.modality`, the GR00T modality.json parser.

Every refusal listed in spec S1 gets its own test that asserts the exception type and a
message substring naming the offending key, slice name or observed value.
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

from foundationscale.vla.lerobot import open_lerobot
from foundationscale.vla.modality import (
    ModalityError,
    SliceSpec,
    check_against_dataset,
    load_modality,
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


def _modality() -> dict[str, Any]:
    """GR00T-shaped modality.json matching the synthetic dataset."""
    return {
        "state": {
            "x": {"start": 0, "end": 2},
            "y": {
                "start": 2,
                "end": 4,
                "original_key": STATE_KEY,
                "absolute": False,
                "rotation_type": "axis_angle",
                "dtype": "float32",
            },
        },
        "action": {"a": {"start": 0, "end": 3}},
        "video": {"image": {"original_key": VIDEO_KEY}},
        "annotation": {"human.action.task_description": {"original_key": "task_index"}},
    }


def _write_modality(root: Path, payload: dict[str, Any]) -> Path:
    path = root / "meta" / "modality.json"
    _write_json(path, payload)
    return path


def _refusal(expected: type[Exception], fn: Callable[[], object], *needles: str) -> Exception:
    """Run ``fn`` and assert it raises ``expected`` naming every ``needle`` in the message."""
    with pytest.raises(expected) as excinfo:
        fn()
    error = excinfo.value
    message = str(error)
    for needle in needles:
        assert needle in message, f"expected {needle!r} in error message {message!r}"
    return error


def test_load_modality_parses_gr00t_shape(tmp_path: Path) -> None:
    path = _write_modality(tmp_path, _modality())
    spec = load_modality(path)
    assert [slice_.name for slice_ in spec.state] == ["x", "y"]
    assert spec.state[0] == SliceSpec(
        name="x", original_key=STATE_KEY, start=0, end=2, absolute=True, rotation_type=None
    )
    assert spec.state[1] == SliceSpec(
        name="y",
        original_key=STATE_KEY,
        start=2,
        end=4,
        absolute=False,
        rotation_type="axis_angle",
    )
    assert spec.state[0].dim == 2
    assert spec.state_dim() == 4
    assert [slice_.name for slice_ in spec.action] == ["a"]
    assert spec.action[0] == SliceSpec(
        name="a", original_key=ACTION_KEY, start=0, end=3, absolute=True, rotation_type=None
    )
    assert spec.action_dim() == 3
    assert dict(spec.video) == {"image": VIDEO_KEY}
    assert dict(spec.annotation) == {"human.action.task_description": "task_index"}


def test_check_against_dataset_accepts_matching_spec(tmp_path: Path) -> None:
    _build(tmp_path)
    ds = open_lerobot(tmp_path)
    spec = load_modality(_write_modality(tmp_path, _modality()))
    assert check_against_dataset(spec, ds) is None


def test_load_modality_refuses_unknown_group(tmp_path: Path) -> None:
    payload = _modality()
    payload["other"] = {"x": {"start": 0, "end": 1}}
    path = _write_modality(tmp_path, payload)
    _refusal(ModalityError, lambda: load_modality(path), "other")


def test_load_modality_refuses_missing_state(tmp_path: Path) -> None:
    payload = _modality()
    payload.pop("state")
    path = _write_modality(tmp_path, payload)
    _refusal(ModalityError, lambda: load_modality(path), "state")


def test_load_modality_refuses_missing_action(tmp_path: Path) -> None:
    payload = _modality()
    payload.pop("action")
    path = _write_modality(tmp_path, payload)
    _refusal(ModalityError, lambda: load_modality(path), "action")


def test_load_modality_refuses_non_int_start(tmp_path: Path) -> None:
    payload = _modality()
    payload["state"]["x"]["start"] = 0.5
    path = _write_modality(tmp_path, payload)
    _refusal(ModalityError, lambda: load_modality(path), "start")


def test_load_modality_refuses_non_int_end(tmp_path: Path) -> None:
    payload = _modality()
    payload["state"]["x"]["end"] = "2"
    path = _write_modality(tmp_path, payload)
    _refusal(ModalityError, lambda: load_modality(path), "end")


def test_load_modality_refuses_negative_start(tmp_path: Path) -> None:
    payload = _modality()
    payload["state"]["x"]["start"] = -1
    path = _write_modality(tmp_path, payload)
    _refusal(ModalityError, lambda: load_modality(path), "start", "-1")


def test_load_modality_refuses_end_not_greater_than_start(tmp_path: Path) -> None:
    payload = _modality()
    payload["state"]["x"]["end"] = 0
    path = _write_modality(tmp_path, payload)
    _refusal(ModalityError, lambda: load_modality(path), "end", "0")


def test_load_modality_refuses_overlapping_slices(tmp_path: Path) -> None:
    payload = _modality()
    payload["state"] = {
        "x0": {"start": 0, "end": 3},
        "x1": {"start": 2, "end": 4},
    }
    path = _write_modality(tmp_path, payload)
    _refusal(ModalityError, lambda: load_modality(path), "x0", "x1")


def test_check_against_dataset_refuses_unknown_original_key(tmp_path: Path) -> None:
    _build(tmp_path)
    ds = open_lerobot(tmp_path)
    payload = _modality()
    payload["state"]["x"]["original_key"] = "observation.nope"
    spec = load_modality(_write_modality(tmp_path, payload))
    _refusal(
        ModalityError,
        lambda: check_against_dataset(spec, ds),
        "observation.nope",
    )


def test_check_against_dataset_refuses_non_1d_feature(tmp_path: Path) -> None:
    _build(tmp_path)

    def _flatten_action(info: dict[str, Any]) -> None:
        info["features"][ACTION_KEY]["shape"] = [3, 1]

    _edit_info(tmp_path, _flatten_action)
    ds = open_lerobot(tmp_path)
    spec = load_modality(_write_modality(tmp_path, _modality()))
    _refusal(ModalityError, lambda: check_against_dataset(spec, ds), ACTION_KEY)


def test_check_against_dataset_refuses_coverage_gap(tmp_path: Path) -> None:
    _build(tmp_path)
    ds = open_lerobot(tmp_path)
    payload = _modality()
    payload["state"] = {
        "x": {"start": 0, "end": 2},
        "y": {"start": 3, "end": 4},
    }
    spec = load_modality(_write_modality(tmp_path, payload))
    _refusal(ModalityError, lambda: check_against_dataset(spec, ds), STATE_KEY, "2")


def test_check_against_dataset_refuses_coverage_overrun(tmp_path: Path) -> None:
    _build(tmp_path)
    ds = open_lerobot(tmp_path)
    payload = _modality()
    payload["state"] = {"x": {"start": 0, "end": 5}}
    spec = load_modality(_write_modality(tmp_path, payload))
    _refusal(ModalityError, lambda: check_against_dataset(spec, ds), STATE_KEY, "4")


def test_check_against_dataset_refuses_non_video_feature(tmp_path: Path) -> None:
    _build(tmp_path)
    ds = open_lerobot(tmp_path)
    payload = _modality()
    payload["video"] = {"image": {"original_key": STATE_KEY}}
    spec = load_modality(_write_modality(tmp_path, payload))
    _refusal(ModalityError, lambda: check_against_dataset(spec, ds), STATE_KEY, "video")


def test_check_against_dataset_refuses_absent_annotation_key(tmp_path: Path) -> None:
    _build(tmp_path)
    ds = open_lerobot(tmp_path)
    payload = _modality()
    payload["annotation"] = {"human.action.task_description": {"original_key": "observation.nope"}}
    spec = load_modality(_write_modality(tmp_path, payload))
    _refusal(ModalityError, lambda: check_against_dataset(spec, ds), "observation.nope")


def test_a_modality_file_that_is_not_utf8_is_a_modality_error(tmp_path: Path) -> None:
    path = tmp_path / "modality.json"
    path.write_bytes(b'{"state": \xe9}')
    with pytest.raises(ModalityError, match="not valid JSON"):
        load_modality(path)
