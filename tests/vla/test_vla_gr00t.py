"""Tests for :mod:`foundationscale.vla.adapters.gr00t.contract`, the GR00T adapter (spec S4).

``load_gr00t_contract`` is checked against a synthetic ``processor_config.json`` written in both
of the locations GR00T ships it in, with a second embodiment beside ``libero_sim`` and every
refusal it promises; ``chunk_spec_from_contract`` and ``check_contract_against`` are checked for
their mappings and refusals; ``build_gr00t_observation`` is checked end to end on a synthetic
LeRobot v2.1 dataset whose videos are "decoded" by an injected fake decoder returning solid-colour
frames, so nothing is really decoded on CPU.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import foundationscale.vla.frames as frames_module
from foundationscale.vla.adapters.gr00t.contract import (
    ACTION_REPS,
    Gr00tContract,
    Gr00tContractError,
    Gr00tGroup,
    _find_action_configs,
    _state_block,
    _task_text,
    build_gr00t_observation,
    check_contract_against,
    chunk_spec_from_contract,
    load_gr00t_contract,
)
from foundationscale.vla.chunking import ChunkSpec
from foundationscale.vla.frames import DECODERS
from foundationscale.vla.lerobot import LeRobotDataset, open_lerobot
from foundationscale.vla.modality import ModalitySpec, load_modality

LENGTHS: tuple[int, ...] = (4, 5, 6)
STATE_DIM = 4
EXTRA_DIM = 2
ACTION_DIM = 3
FPS = 30.0
CHUNKS_SIZE = 1000
TASK = "stack the blocks"
TASK_2 = "pick up the cup"
TASKS: tuple[str, ...] = (TASK, TASK_2)
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

# The slices in modality.json file order; the contract names them, the blocks read their columns.
STATE_SLICES: Mapping[str, tuple[str, int, int]] = {
    "pos": (STATE_KEY, 0, 3),
    "grip": (STATE_KEY, 3, 4),
    "extra": (EXTRA_KEY, 0, 2),
}
VIDEO_ORIGINALS: Mapping[str, str] = {"image": VIDEO_KEY, "wrist": WRIST_KEY}

# The GR00T side: one embodiment shaped like libero_sim plus a second one to select between.
EMBODIMENT = "libero_sim"
OTHER_EMBODIMENT = "aloha_sim"
STATE_KEYS: tuple[str, ...] = ("grip", "pos", "extra")  # file order, deliberately not sorted
ACTION_KEYS: tuple[str, ...] = ("target", "grip")
VIDEO_KEYS: tuple[str, ...] = ("image", "wrist")
LANGUAGE_KEYS: tuple[str, ...] = ("annotation.human.action.task_description",)
STATE_DELTA: tuple[int, ...] = (0,)
ACTION_DELTA: tuple[int, ...] = (0, 1, 2)
VIDEO_DELTA: tuple[int, ...] = (0, 1)
LANGUAGE_DELTA: tuple[int, ...] = (0,)
ABSOLUTE_REPS: tuple[str, ...] = ("ABSOLUTE",) * len(ACTION_KEYS)
LANGUAGE_KEY = "annotation.human.action.task_description"

MISSING = object()


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


def _task_index(length: int) -> np.ndarray:
    """Deterministic per-frame task index, so the task text depends on the anchor."""
    return np.arange(length, dtype=np.int64) % len(TASKS)


def _task_at(anchor: int) -> str:
    """The task text the synthetic ``task_index`` column points at for ``anchor``."""
    return TASKS[anchor % len(TASKS)]


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
        {"episode_index": episode, "length": length, "tasks": list(TASKS)}
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
            "task_index": pa.array(_task_index(length)),
        }
    )


def _build(root: Path, modality: Mapping[str, Any] | None = None) -> None:
    """Write a tiny LeRobot v2.1 dataset: 3 episodes of lengths 4, 5 and 6 plus dummy videos."""
    _write_json(root / "meta" / "info.json", _info())
    _write_jsonl(root / "meta" / "episodes.jsonl", _episodes())
    _write_jsonl(
        root / "meta" / "tasks.jsonl",
        [{"task_index": index, "task": task} for index, task in enumerate(TASKS)],
    )
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


def _modality_config(
    *,
    state_delta: Sequence[int] = STATE_DELTA,
    action_delta: Sequence[int] = ACTION_DELTA,
    video_delta: Sequence[int] = VIDEO_DELTA,
    language_delta: Sequence[int] = LANGUAGE_DELTA,
    state_keys: Sequence[str] = STATE_KEYS,
    action_keys: Sequence[str] = ACTION_KEYS,
    video_keys: Sequence[str] = VIDEO_KEYS,
    language_keys: Sequence[str] = LANGUAGE_KEYS,
    action_reps: Sequence[str] | None = None,
) -> dict[str, Any]:
    """One embodiment's modality contract, shaped like GR00T's ``libero_sim`` entry."""
    reps = tuple(action_reps) if action_reps is not None else ("ABSOLUTE",) * len(action_keys)
    return {
        "video": {"delta_indices": list(video_delta), "modality_keys": list(video_keys)},
        "state": {"delta_indices": list(state_delta), "modality_keys": list(state_keys)},
        "action": {
            "delta_indices": list(action_delta),
            "modality_keys": list(action_keys),
            "action_configs": [{"rep": rep} for rep in reps],
        },
        "language": {"delta_indices": list(language_delta), "modality_keys": list(language_keys)},
    }


def _other_modality_config() -> dict[str, Any]:
    """A second embodiment with its own keys and deltas, so the two can be told apart."""
    return _modality_config(
        state_delta=(0, 1),
        action_delta=(0,),
        video_delta=(0,),
        language_delta=(0,),
        state_keys=("joint",),
        action_keys=("cmd",),
        video_keys=("front",),
        language_keys=("annotation.human.action.task_description",),
    )


def _processor_config(**kwargs: Any) -> dict[str, Any]:
    """A synthetic ``processor_config.json`` payload holding two embodiments."""
    return {
        "processor_kwargs": {
            "modality_configs": {
                EMBODIMENT: _modality_config(**kwargs),
                OTHER_EMBODIMENT: _other_modality_config(),
            }
        }
    }


def _group_of(
    payload: Mapping[str, Any], group: str, embodiment: str = EMBODIMENT
) -> dict[str, Any]:
    """The named group of ``embodiment``, for tests that corrupt one field at a time."""
    return payload["processor_kwargs"]["modality_configs"][embodiment][group]


def _write_config(tmp_path: Path, payload: Any, *, location: str = "root") -> Path:
    """Write ``payload`` where GR00T ships ``processor_config.json``: the root or ``processor/``."""
    path = tmp_path / "processor_config.json"
    if location == "processor":
        path = tmp_path / "processor" / "processor_config.json"
    _write_json(path, payload)
    return path


def _contract(
    tmp_path: Path,
    payload: Any | None = None,
    *,
    location: str = "root",
    embodiment: str = EMBODIMENT,
) -> Gr00tContract:
    """Write a processor config and load the contract it declares."""
    _write_config(tmp_path, _processor_config() if payload is None else payload, location=location)
    return load_gr00t_contract(tmp_path, embodiment)


def _expected_state(episode: int, indices: Sequence[int], key: str) -> np.ndarray:
    """The ``state.<key>`` block: the slice's columns of its ``original_key`` at ``indices``."""
    length = LENGTHS[episode]
    columns: Mapping[str, np.ndarray] = {
        STATE_KEY: _state(episode, length),
        EXTRA_KEY: _extra(episode, length),
    }
    original_key, start, end = STATE_SLICES[key]
    return columns[original_key][list(indices), start:end].astype(np.float32)


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


# -- the exception and the representation vocabulary -------------------------------------


def test_gr00t_contract_error_is_a_value_error() -> None:
    assert issubclass(Gr00tContractError, ValueError)


def test_action_reps_covers_the_three_gr00t_representations() -> None:
    assert frozenset({"ABSOLUTE", "RELATIVE", "DELTA"}) == ACTION_REPS


# -- load_gr00t_contract -----------------------------------------------------------------


def test_load_gr00t_contract_reads_the_top_level_config(tmp_path: Path) -> None:
    source = _write_config(tmp_path, _processor_config())
    contract = load_gr00t_contract(tmp_path, EMBODIMENT)
    assert isinstance(contract, Gr00tContract)
    assert contract.embodiment == EMBODIMENT
    assert contract.source == source
    assert isinstance(contract.state, Gr00tGroup)
    assert contract.state.delta_indices == STATE_DELTA
    assert contract.state.keys == STATE_KEYS
    assert contract.action.delta_indices == ACTION_DELTA
    assert contract.action.keys == ACTION_KEYS
    assert contract.video.delta_indices == VIDEO_DELTA
    assert contract.video.keys == VIDEO_KEYS
    assert contract.language.delta_indices == LANGUAGE_DELTA
    assert contract.language.keys == LANGUAGE_KEYS
    assert contract.action_reps == ABSOLUTE_REPS


def test_load_gr00t_contract_reads_the_training_layout_config(tmp_path: Path) -> None:
    source = _write_config(tmp_path, _processor_config(), location="processor")
    contract = load_gr00t_contract(tmp_path, EMBODIMENT)
    assert contract.source == source
    assert contract.state.keys == STATE_KEYS
    assert contract.action_reps == ABSOLUTE_REPS


def test_load_gr00t_contract_prefers_the_top_level_config(tmp_path: Path) -> None:
    top = _processor_config()
    _group_of(top, "state")["modality_keys"] = ["pos"]
    _write_config(tmp_path, top, location="root")
    _write_config(tmp_path, _processor_config(), location="processor")
    contract = load_gr00t_contract(tmp_path, EMBODIMENT)
    assert contract.source == tmp_path / "processor_config.json"
    assert contract.state.keys == ("pos",)


def test_load_gr00t_contract_selects_the_requested_embodiment(tmp_path: Path) -> None:
    _write_config(tmp_path, _processor_config())
    contract = load_gr00t_contract(tmp_path, OTHER_EMBODIMENT)
    assert contract.embodiment == OTHER_EMBODIMENT
    assert contract.state.keys == ("joint",)
    assert contract.action.keys == ("cmd",)
    assert contract.video.keys == ("front",)
    assert contract.state.delta_indices == (0, 1)
    assert contract.action.delta_indices == (0,)
    assert contract.video.delta_indices == (0,)


def test_load_gr00t_contract_keeps_the_group_keys_in_file_order(tmp_path: Path) -> None:
    payload = _processor_config()
    _group_of(payload, "state")["modality_keys"] = ["pos", "extra", "grip"]
    _write_config(tmp_path, payload)
    contract = load_gr00t_contract(tmp_path, EMBODIMENT)
    assert contract.state.keys == ("pos", "extra", "grip")
    assert contract.state.keys != tuple(sorted(contract.state.keys))


def test_load_gr00t_contract_pairs_action_reps_with_action_keys_in_file_order(
    tmp_path: Path,
) -> None:
    _write_config(tmp_path, _processor_config(action_reps=("DELTA", "ABSOLUTE")))
    contract = load_gr00t_contract(tmp_path, EMBODIMENT)
    assert contract.action.keys == ACTION_KEYS
    assert contract.action_reps == ("DELTA", "ABSOLUTE")
    assert set(contract.action_reps) <= ACTION_REPS


def test_load_gr00t_contract_refuses_a_missing_config(tmp_path: Path) -> None:
    _refusal(
        Gr00tContractError,
        lambda: load_gr00t_contract(tmp_path, EMBODIMENT),
        "processor_config.json",
    )


def test_load_gr00t_contract_refuses_a_config_that_is_not_a_directory(tmp_path: Path) -> None:
    path = tmp_path / "checkpoint"
    path.write_text("not a directory", encoding="utf-8")
    _refusal(
        Gr00tContractError,
        lambda: load_gr00t_contract(path, EMBODIMENT),
        "processor_config.json",
    )


def test_load_gr00t_contract_refuses_invalid_json(tmp_path: Path) -> None:
    path = tmp_path / "processor_config.json"
    path.write_text("{not json", encoding="utf-8")
    _refusal(
        Gr00tContractError,
        lambda: load_gr00t_contract(tmp_path, EMBODIMENT),
        "processor_config.json",
        "JSON",
    )


@pytest.mark.parametrize(
    "payload",
    [
        pytest.param({}, id="empty"),
        pytest.param({"processor_kwargs": {}}, id="no_processor_kwargs"),
        pytest.param({"processor_kwargs": {"modality_configs": None}}, id="null_modality_configs"),
    ],
)
def test_load_gr00t_contract_refuses_a_config_without_modality_configs(
    tmp_path: Path, payload: dict[str, Any]
) -> None:
    _write_config(tmp_path, payload)
    _refusal(
        Gr00tContractError,
        lambda: load_gr00t_contract(tmp_path, EMBODIMENT),
        "modality_configs",
    )


def test_load_gr00t_contract_refuses_an_absent_embodiment_naming_the_available_ones(
    tmp_path: Path,
) -> None:
    _write_config(tmp_path, _processor_config())
    error = _refusal(
        Gr00tContractError,
        lambda: load_gr00t_contract(tmp_path, "new_embodiment"),
        "new_embodiment",
        OTHER_EMBODIMENT,
        EMBODIMENT,
    )
    message = str(error)
    assert message.index(OTHER_EMBODIMENT) < message.index(EMBODIMENT), (
        f"expected the available embodiments sorted in {message!r}"
    )


@pytest.mark.parametrize("group", ["video", "state", "action", "language"])
def test_load_gr00t_contract_refuses_a_missing_group(tmp_path: Path, group: str) -> None:
    payload = _processor_config()
    del payload["processor_kwargs"]["modality_configs"][EMBODIMENT][group]
    _write_config(tmp_path, payload)
    _refusal(Gr00tContractError, lambda: load_gr00t_contract(tmp_path, EMBODIMENT), group)


def test_load_gr00t_contract_refuses_a_group_without_delta_indices(tmp_path: Path) -> None:
    payload = _processor_config()
    del _group_of(payload, "state")["delta_indices"]
    _write_config(tmp_path, payload)
    _refusal(
        Gr00tContractError,
        lambda: load_gr00t_contract(tmp_path, EMBODIMENT),
        "delta_indices",
    )


@pytest.mark.parametrize(
    ("group", "delta"),
    [
        pytest.param("state", None, id="state-null"),
        pytest.param("state", [], id="state-empty"),
        pytest.param("state", 0, id="state-not-a-list"),
        pytest.param("state", "01", id="state-string"),
        pytest.param("state", [0, 1.5], id="state-float"),
        pytest.param("state", [0, "1"], id="state-string-entry"),
        pytest.param("video", [0, None], id="video-null-entry"),
        pytest.param("action", [0, [1]], id="action-nested-entry"),
        pytest.param("language", [0, 0.0], id="language-float-entry"),
    ],
)
def test_load_gr00t_contract_refuses_delta_indices_that_are_not_a_non_empty_list_of_ints(
    tmp_path: Path, group: str, delta: Any
) -> None:
    payload = _processor_config()
    _group_of(payload, group)["delta_indices"] = delta
    _write_config(tmp_path, payload)
    _refusal(
        Gr00tContractError,
        lambda: load_gr00t_contract(tmp_path, EMBODIMENT),
        "delta_indices",
    )


def test_load_gr00t_contract_refuses_a_group_without_modality_keys(tmp_path: Path) -> None:
    payload = _processor_config()
    del _group_of(payload, "video")["modality_keys"]
    _write_config(tmp_path, payload)
    _refusal(
        Gr00tContractError,
        lambda: load_gr00t_contract(tmp_path, EMBODIMENT),
        "modality_keys",
    )


@pytest.mark.parametrize(
    ("group", "keys"),
    [
        pytest.param("state", None, id="state-null"),
        pytest.param("state", [], id="state-empty"),
        pytest.param("state", 0, id="state-not-a-list"),
        pytest.param("state", ["pos", "pos"], id="state-duplicate"),
        pytest.param("state", ["pos", 3], id="state-int-entry"),
        pytest.param("action", [None], id="action-null-entry"),
        pytest.param("video", ["image", "wrist", "image"], id="video-duplicate"),
    ],
)
def test_load_gr00t_contract_refuses_modality_keys_that_are_not_unique_strings(
    tmp_path: Path, group: str, keys: Any
) -> None:
    payload = _processor_config()
    _group_of(payload, group)["modality_keys"] = keys
    _write_config(tmp_path, payload)
    _refusal(
        Gr00tContractError,
        lambda: load_gr00t_contract(tmp_path, EMBODIMENT),
        "modality_keys",
    )


@pytest.mark.parametrize(
    "configs",
    [pytest.param(MISSING, id="absent"), pytest.param(None, id="null")],
)
def test_load_gr00t_contract_refuses_absent_or_null_action_configs(
    tmp_path: Path, configs: Any
) -> None:
    payload = _processor_config()
    group = _group_of(payload, "action")
    if configs is MISSING:
        del group["action_configs"]
    else:
        group["action_configs"] = configs
    _write_config(tmp_path, payload)
    _refusal(
        Gr00tContractError,
        lambda: load_gr00t_contract(tmp_path, EMBODIMENT),
        "action_configs",
    )


@pytest.mark.parametrize(
    "configs",
    [
        pytest.param([{"rep": "ABSOLUTE"}], id="too-few"),
        pytest.param([{"rep": "ABSOLUTE"}] * 3, id="too-many"),
    ],
)
def test_load_gr00t_contract_refuses_action_configs_that_do_not_match_the_action_keys(
    tmp_path: Path, configs: list[dict[str, str]]
) -> None:
    payload = _processor_config()
    _group_of(payload, "action")["action_configs"] = configs
    _write_config(tmp_path, payload)
    _refusal(
        Gr00tContractError,
        lambda: load_gr00t_contract(tmp_path, EMBODIMENT),
        "action_configs",
    )


def test_load_gr00t_contract_refuses_a_rep_outside_action_reps(tmp_path: Path) -> None:
    payload = _processor_config(action_reps=("ABSOLUTE", "SCALE"))
    _write_config(tmp_path, payload)
    _refusal(
        Gr00tContractError,
        lambda: load_gr00t_contract(tmp_path, EMBODIMENT),
        "SCALE",
        "ABSOLUTE",
    )


# -- chunk_spec_from_contract ------------------------------------------------------------


def test_chunk_spec_from_contract_maps_the_delta_indices(tmp_path: Path) -> None:
    _write_config(tmp_path, _processor_config())
    contract = load_gr00t_contract(tmp_path, EMBODIMENT)
    chunks = chunk_spec_from_contract(contract)
    assert isinstance(chunks, ChunkSpec)
    assert chunks.state_delta == STATE_DELTA
    assert chunks.action_delta == ACTION_DELTA
    assert chunks.video_delta == VIDEO_DELTA


def test_chunk_spec_from_contract_re_raises_chunk_spec_refusals_with_the_source_path(
    tmp_path: Path,
) -> None:
    source = tmp_path / "processor" / "processor_config.json"
    empty: Any = ()  # no steps at all
    unsorted: Any = (1, 0, 1)  # out of order and repeated
    non_int: Any = ("a", "b")  # not ints
    contract = Gr00tContract(
        embodiment=EMBODIMENT,
        video=Gr00tGroup(non_int, VIDEO_KEYS),
        state=Gr00tGroup(empty, STATE_KEYS),
        action=Gr00tGroup(unsorted, ACTION_KEYS),
        language=Gr00tGroup(LANGUAGE_DELTA, LANGUAGE_KEYS),
        action_reps=ABSOLUTE_REPS,
        source=source,
    )
    _refusal(Gr00tContractError, lambda: chunk_spec_from_contract(contract), str(source))


# -- check_contract_against --------------------------------------------------------------


def test_check_contract_against_accepts_a_matching_spec(tmp_path: Path) -> None:
    _, spec = _setup(tmp_path)
    contract = _contract(tmp_path)
    assert check_contract_against(contract, spec) is None


def test_check_contract_against_refuses_a_state_key_missing_from_the_spec(tmp_path: Path) -> None:
    _, spec = _setup(tmp_path)
    payload = _processor_config()
    _group_of(payload, "state")["modality_keys"] = ["pos", "nope"]
    contract = _contract(tmp_path, payload)
    _refusal(
        Gr00tContractError,
        lambda: check_contract_against(contract, spec),
        "nope",
        "state",
    )


def test_check_contract_against_refuses_an_action_key_missing_from_the_spec(tmp_path: Path) -> None:
    _, spec = _setup(tmp_path)
    payload = _processor_config(action_keys=("target", "nope"), action_reps=("ABSOLUTE",) * 2)
    contract = _contract(tmp_path, payload)
    _refusal(
        Gr00tContractError,
        lambda: check_contract_against(contract, spec),
        "nope",
        "action",
    )


def test_check_contract_against_refuses_a_video_key_missing_from_the_spec(tmp_path: Path) -> None:
    _, spec = _setup(tmp_path)
    payload = _processor_config(video_keys=("image", "nope"))
    contract = _contract(tmp_path, payload)
    _refusal(
        Gr00tContractError,
        lambda: check_contract_against(contract, spec),
        "nope",
        "video",
    )


@pytest.mark.parametrize("rep", ["RELATIVE", "DELTA"])
def test_check_contract_against_refuses_a_non_absolute_action_rep(tmp_path: Path, rep: str) -> None:
    _, spec = _setup(tmp_path)
    contract = _contract(tmp_path, _processor_config(action_reps=(rep, "ABSOLUTE")))
    _refusal(
        Gr00tContractError,
        lambda: check_contract_against(contract, spec),
        "target",
        "RELATIVE/DELTA actions need the state_key reference and are not supported yet",
    )


def test_check_contract_against_refuses_an_annotation_without_task_index(tmp_path: Path) -> None:
    modality = _modality()
    modality["annotation"] = {"human.action.task_description": {"original_key": "episode_index"}}
    _, spec = _setup(tmp_path, modality)
    contract = _contract(tmp_path)
    _refusal(Gr00tContractError, lambda: check_contract_against(contract, spec), "task_index")


# -- build_gr00t_observation ------------------------------------------------------------


def test_build_gr00t_observation_builds_the_observation_dict(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ds, spec = _setup(tmp_path)
    fake = _FakeDecoder()
    _inject(monkeypatch, fake)
    contract = _contract(tmp_path)
    observation = build_gr00t_observation(ds, spec, contract, 1, 2)
    assert set(observation) == {
        "state.grip",
        "state.pos",
        "state.extra",
        "video.image",
        "video.wrist",
        LANGUAGE_KEY,
    }
    for key in STATE_KEYS:
        block = observation[f"state.{key}"]
        expected = _expected_state(1, [2], key)
        assert isinstance(block, np.ndarray)
        assert block.shape == (len(STATE_DELTA), expected.shape[1])
        assert block.dtype == np.float32
        np.testing.assert_allclose(block, expected, rtol=0, atol=1e-6)
    for key in VIDEO_KEYS:
        frames = observation[f"video.{key}"]
        tint = TINTS[VIDEO_ORIGINALS[key]]
        assert isinstance(frames, np.ndarray)
        assert frames.shape == (len(VIDEO_DELTA), HEIGHT, WIDTH, 3)
        assert frames.dtype == np.uint8
        np.testing.assert_array_equal(frames[0], _solid(2, tint))
        np.testing.assert_array_equal(frames[1], _solid(3, tint))
    assert observation[LANGUAGE_KEY] == _task_at(2)
    assert len(fake.calls) == 2
    assert {path.parent.name for path in fake.calls} == {VIDEO_KEY, WRIST_KEY}


def test_build_gr00t_observation_reads_the_task_text_at_the_anchor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ds, spec = _setup(tmp_path)
    _inject(monkeypatch, _FakeDecoder())
    contract = _contract(tmp_path)
    first = build_gr00t_observation(ds, spec, contract, 1, 0)
    second = build_gr00t_observation(ds, spec, contract, 1, 1)
    assert first[LANGUAGE_KEY] == TASK
    assert second[LANGUAGE_KEY] == TASK_2


def test_build_gr00t_observation_gives_every_language_key_the_task_text(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ds, spec = _setup(tmp_path)
    _inject(monkeypatch, _FakeDecoder())
    contract = _contract(
        tmp_path,
        _processor_config(
            language_keys=(
                "annotation.human.action.task_description",
                "annotation.human.action.comment",
            )
        ),
    )
    observation = build_gr00t_observation(ds, spec, contract, 2, 3)
    assert set(observation) == {
        "state.grip",
        "state.pos",
        "state.extra",
        "video.image",
        "video.wrist",
        LANGUAGE_KEY,
        "annotation.human.action.comment",
    }
    assert observation["annotation.human.action.comment"] == _task_at(3)
    assert observation["annotation.human.action.comment"] == observation[LANGUAGE_KEY]


def test_build_gr00t_observation_clamps_a_negative_delta_at_the_first_frame(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ds, spec = _setup(tmp_path)
    _inject(monkeypatch, _FakeDecoder())
    contract = _contract(
        tmp_path,
        _processor_config(state_delta=(-2, -1, 0), action_delta=(-1, 0), video_delta=(-1, 0)),
    )
    observation = build_gr00t_observation(ds, spec, contract, 0, 0)
    np.testing.assert_allclose(
        observation["state.pos"], _expected_state(0, [0, 0, 0], "pos"), rtol=0, atol=1e-6
    )
    np.testing.assert_allclose(
        observation["state.extra"], _expected_state(0, [0, 0, 0], "extra"), rtol=0, atol=1e-6
    )
    frames = observation["video.image"]
    assert frames.shape == (2, HEIGHT, WIDTH, 3)
    np.testing.assert_array_equal(frames[0], _solid(0, TINTS[VIDEO_KEY]))
    np.testing.assert_array_equal(frames[1], _solid(0, TINTS[VIDEO_KEY]))
    assert observation[LANGUAGE_KEY] == _task_at(0)


def test_build_gr00t_observation_clamps_a_positive_delta_at_the_last_frame(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ds, spec = _setup(tmp_path)
    _inject(monkeypatch, _FakeDecoder())
    contract = _contract(
        tmp_path,
        _processor_config(state_delta=(0, 1, 2), action_delta=(0, 1), video_delta=(0, 1)),
    )
    last = LENGTHS[2] - 1
    observation = build_gr00t_observation(ds, spec, contract, 2, last)
    np.testing.assert_allclose(
        observation["state.pos"],
        _expected_state(2, [last, last, last], "pos"),
        rtol=0,
        atol=1e-6,
    )
    frames = observation["video.wrist"]
    assert frames.shape == (2, HEIGHT, WIDTH, 3)
    np.testing.assert_array_equal(frames[0], _solid(last, TINTS[WRIST_KEY]))
    np.testing.assert_array_equal(frames[1], _solid(last, TINTS[WRIST_KEY]))
    assert observation[LANGUAGE_KEY] == _task_at(last)


def test_build_gr00t_observation_refuses_an_anchor_before_the_first_frame(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ds, spec = _setup(tmp_path)
    _inject(monkeypatch, _FakeDecoder())
    contract = _contract(tmp_path)
    _refusal(
        Gr00tContractError,
        lambda: build_gr00t_observation(ds, spec, contract, 0, -1),
        "anchor",
        "-1",
    )


def test_build_gr00t_observation_refuses_an_anchor_past_the_last_frame(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ds, spec = _setup(tmp_path)
    _inject(monkeypatch, _FakeDecoder())
    contract = _contract(tmp_path)
    _refusal(
        Gr00tContractError,
        lambda: build_gr00t_observation(ds, spec, contract, 0, LENGTHS[0]),
        "anchor",
        str(LENGTHS[0]),
    )


def test_build_gr00t_observation_refuses_an_unknown_episode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ds, spec = _setup(tmp_path)
    _inject(monkeypatch, _FakeDecoder())
    contract = _contract(tmp_path)
    _refusal(
        Gr00tContractError,
        lambda: build_gr00t_observation(ds, spec, contract, 99, 0),
        "episode",
        "99",
    )


def test_build_gr00t_observation_checks_the_contract_before_decoding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ds, spec = _setup(tmp_path)
    fake = _FakeDecoder()
    _inject(monkeypatch, fake)
    payload = _processor_config()
    _group_of(payload, "state")["modality_keys"] = ["pos", "nope"]
    contract = _contract(tmp_path, payload)
    _refusal(
        Gr00tContractError,
        lambda: build_gr00t_observation(ds, spec, contract, 0, 0),
        "nope",
    )
    assert fake.calls == []


# -- coverage of the remaining refusal branches ------------------------------------------


def test_load_gr00t_contract_refuses_an_embodiment_entry_that_is_not_a_mapping(
    tmp_path: Path,
) -> None:
    """gr00t.py:160 -- the embodiment's own entry must be a mapping of groups."""
    payload = _processor_config()
    payload["processor_kwargs"]["modality_configs"][EMBODIMENT] = ["not", "a", "mapping"]
    _write_config(tmp_path, payload)
    _refusal(
        Gr00tContractError,
        lambda: load_gr00t_contract(tmp_path, EMBODIMENT),
        "expected a mapping of groups",
    )


def test_check_contract_against_refuses_action_reps_that_do_not_match_the_action_keys(
    tmp_path: Path,
) -> None:
    """gr00t.py:238 -- one rep per action key, no more and no fewer."""
    _, spec = _setup(tmp_path)
    contract = _contract(tmp_path)
    mismatched = Gr00tContract(
        embodiment=contract.embodiment,
        video=contract.video,
        state=contract.state,
        action=contract.action,
        language=contract.language,
        action_reps=("ABSOLUTE",),
        source=contract.source,
    )
    _refusal(
        Gr00tContractError,
        lambda: check_contract_against(mismatched, spec),
        "contract action_reps has",
        "rep per action key",
    )


def test_build_gr00t_observation_refuses_a_non_integer_episode_index(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """gr00t.py:292 -- ``episode_index`` must be a real int."""
    ds, spec = _setup(tmp_path)
    _inject(monkeypatch, _FakeDecoder())
    contract = _contract(tmp_path)
    _refusal(
        Gr00tContractError,
        lambda: build_gr00t_observation(ds, spec, contract, 1.5, 0),
        "episode_index is",
        "expected an int",
    )


def test_build_gr00t_observation_refuses_a_non_integer_anchor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """gr00t.py:301 -- ``anchor`` must be a real int."""
    ds, spec = _setup(tmp_path)
    _inject(monkeypatch, _FakeDecoder())
    contract = _contract(tmp_path)
    _refusal(
        Gr00tContractError,
        lambda: build_gr00t_observation(ds, spec, contract, 0, 1.5),
        "anchor is",
        "expected an int",
    )


def test_load_gr00t_contract_refuses_a_group_that_is_not_a_mapping(tmp_path: Path) -> None:
    """gr00t.py:351 -- every group must be a mapping of its two fields."""
    payload = _processor_config()
    payload["processor_kwargs"]["modality_configs"][EMBODIMENT]["state"] = 7
    _write_config(tmp_path, payload)
    _refusal(
        Gr00tContractError,
        lambda: load_gr00t_contract(tmp_path, EMBODIMENT),
        "'delta_indices' and 'modality_keys'",
    )


def test_find_action_configs_yields_nothing_when_the_action_group_is_not_a_mapping() -> None:
    """gr00t.py:412 -- no action group, so no ``action_configs`` and a key that names it."""
    configs, key = _find_action_configs({"action": 7}, base_key="cfg")
    assert configs is None
    assert key == "cfg.action.action_configs"


def test_load_gr00t_contract_reads_action_configs_as_a_mapping_of_key_to_entry(
    tmp_path: Path,
) -> None:
    """gr00t.py:432-444 -- the mapping form of ``action_configs``, keyed by action key."""
    # One entry per action key, keyed by that key: read in the action keys' order.
    payload = _processor_config()
    _group_of(payload, "action")["action_configs"] = {
        "target": {"rep": "ABSOLUTE"},
        "grip": {"rep": "DELTA"},
    }
    contract = _contract(tmp_path, payload)
    assert contract.action.keys == ACTION_KEYS
    assert contract.action_reps == ("ABSOLUTE", "DELTA")

    # A mapping with the wrong number of entries is refused.
    payload = _processor_config()
    _group_of(payload, "action")["action_configs"] = {"target": {"rep": "ABSOLUTE"}}
    _write_config(tmp_path, payload)
    _refusal(
        Gr00tContractError,
        lambda: load_gr00t_contract(tmp_path, EMBODIMENT),
        "has length",
        "expected one entry per action key",
    )

    # A mapping of the right size that omits an action key is refused.
    payload = _processor_config()
    _group_of(payload, "action")["action_configs"] = {
        "target": {"rep": "ABSOLUTE"},
        "nope": {"rep": "ABSOLUTE"},
    }
    _write_config(tmp_path, payload)
    _refusal(
        Gr00tContractError,
        lambda: load_gr00t_contract(tmp_path, EMBODIMENT),
        "expected an entry for action key",
        "grip",
    )


def test_load_gr00t_contract_refuses_action_configs_that_are_not_a_list_or_mapping(
    tmp_path: Path,
) -> None:
    """gr00t.py:453 -- ``action_configs`` must be a list or a mapping of entries."""
    payload = _processor_config()
    _group_of(payload, "action")["action_configs"] = 7
    _write_config(tmp_path, payload)
    _refusal(
        Gr00tContractError,
        lambda: load_gr00t_contract(tmp_path, EMBODIMENT),
        "mapping of one entry per action key",
    )


def test_load_gr00t_contract_refuses_an_action_configs_entry_that_is_not_a_mapping(
    tmp_path: Path,
) -> None:
    """gr00t.py:460 -- every ``action_configs`` entry must carry a ``rep``."""
    payload = _processor_config()
    _group_of(payload, "action")["action_configs"] = [{"rep": "ABSOLUTE"}, 7]
    _write_config(tmp_path, payload)
    _refusal(
        Gr00tContractError,
        lambda: load_gr00t_contract(tmp_path, EMBODIMENT),
        "expected a mapping with a 'rep'",
    )


def test_state_block_reshapes_a_one_dimensional_column(tmp_path: Path) -> None:
    """gr00t.py:492 -- a scalar column is one value per frame, so it gains its dim."""
    modality = _modality()
    modality["state"]["tick"] = {"original_key": "frame_index", "start": 0, "end": 1}
    ds, spec = _setup(tmp_path, modality)
    item = next(item for item in spec.state if item.name == "tick")
    columns = {item.original_key: np.arange(LENGTHS[0], dtype=np.float32)}
    block = _state_block(columns, item, [1, 2], ds=ds, episode=0)
    assert isinstance(block, np.ndarray)
    assert block.shape == (2, 1)
    assert block.dtype == np.float32
    np.testing.assert_allclose(block, [[1.0], [2.0]], rtol=0, atol=1e-6)


def test_state_block_refuses_a_column_with_unexpected_dimensions(tmp_path: Path) -> None:
    """gr00t.py:494 -- one value per frame per dim is the only column shape served."""
    ds, spec = _setup(tmp_path)
    item = next(item for item in spec.state if item.name == "pos")
    columns = {STATE_KEY: np.zeros((LENGTHS[0], 2, 2), dtype=np.float32)}
    _refusal(
        Gr00tContractError,
        lambda: _state_block(columns, item, [0], ds=ds, episode=0),
        "expected one value per frame per dim",
    )


def test_state_block_refuses_a_slice_wider_than_the_column(tmp_path: Path) -> None:
    """gr00t.py:500 -- the slice's end may not run past the dims the column carries."""
    ds, spec = _setup(tmp_path)
    item = next(item for item in spec.state if item.name == "pos")
    columns = {STATE_KEY: np.zeros((LENGTHS[0], 2), dtype=np.float32)}
    _refusal(
        Gr00tContractError,
        lambda: _state_block(columns, item, [0], ds=ds, episode=0),
        "the column carries",
        "dim(s) per frame",
    )


def test_task_text_refuses_a_task_index_column_that_is_not_one_dimensional(
    tmp_path: Path,
) -> None:
    """gr00t.py:519 -- one task index per frame, never a per-dim column."""
    ds, _ = _setup(tmp_path)
    columns = {"task_index": np.zeros((LENGTHS[0], 2), dtype=np.int64)}
    _refusal(
        Gr00tContractError,
        lambda: _task_text(ds, columns, 0, 0),
        "expected one task index per frame",
    )


def test_task_text_refuses_a_task_index_that_is_not_a_number(tmp_path: Path) -> None:
    """gr00t.py:526-527 -- a task index that will not parse as a number is refused."""
    ds, _ = _setup(tmp_path)
    columns = {"task_index": np.array(["nope"] * LENGTHS[0], dtype=object)}
    _refusal(
        Gr00tContractError,
        lambda: _task_text(ds, columns, 0, 0),
        "is 'nope'",
        "integer task index (",
    )


def test_task_text_refuses_a_task_index_that_is_not_an_integer(tmp_path: Path) -> None:
    """gr00t.py:532 -- a fractional task index names no task in ``meta/tasks.jsonl``."""
    ds, _ = _setup(tmp_path)
    columns = {"task_index": np.array([1.5] * LENGTHS[0], dtype=object)}
    _refusal(
        Gr00tContractError,
        lambda: _task_text(ds, columns, 0, 0),
        "1.5",
        "expected an integer task index",
    )


def test_task_text_refuses_a_task_index_absent_from_the_tasks(tmp_path: Path) -> None:
    """gr00t.py:538 -- the task index must name a row of ``meta/tasks.jsonl``."""
    ds, _ = _setup(tmp_path)
    columns = {"task_index": np.array([7] * LENGTHS[0])}
    _refusal(
        Gr00tContractError,
        lambda: _task_text(ds, columns, 0, 0),
        "is absent from",
        "meta/tasks.jsonl",
    )
