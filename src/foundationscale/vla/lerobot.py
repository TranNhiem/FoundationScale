"""LeRobot v2 metadata as declared, plus one episode's columns as numpy arrays.

LeRobot v3 is refused here, and deliberately: v3 stores many episodes per file,
so the one-file-per-episode paths built from ``info.json``'s templates do not
describe a v3 corpus -- opening one would resolve the wrong file, or none at
all. Convert the corpus to v2, or add a v3 reader with its own layout rules.

Everything is read exactly as the dataset declares it: ``meta/info.json``
(codebase version, ``fps``, ``chunks_size``, path templates, features),
``meta/episodes.jsonl`` (each episode's length and tasks) and ``meta/tasks.jsonl``
(task index -> task text). Nothing is defaulted: a missing ``fps`` or
``chunks_size``, a path template without its placeholders, an episode whose rows
disagree with its declared length, or a task text that no task file declares are
all :class:`LeRobotFormatError`, because a guessed value would silently
mis-resolve every path and every timestamp check built on top of it.

:func:`read_episode_columns` returns only the requested columns, and only after
the episode's rows agree with its declared length and the dataset's ``fps``:
``frame_index`` must be exactly ``0..length-1``, ``timestamp`` must increase
strictly and sit within half a frame of ``frame_index / fps``. A list column
becomes a 2-D ``float32`` block; ragged rows are refused rather than padded,
since padding would invent values the corpus never stored.

Imports are lazy: the module loads without numpy or pyarrow.
"""

from __future__ import annotations

import json
import math
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeGuard

if TYPE_CHECKING:
    import numpy as np

__all__ = [
    "SUPPORTED_CODEBASE_VERSIONS",
    "EpisodeMeta",
    "LeRobotDataset",
    "LeRobotFormatError",
    "open_lerobot",
    "read_episode_columns",
]

SUPPORTED_CODEBASE_VERSIONS: frozenset[str] = frozenset({"v2.0", "v2.1"})

_INFO = "meta/info.json"
_EPISODES = "meta/episodes.jsonl"
_TASKS = "meta/tasks.jsonl"


class LeRobotFormatError(ValueError):
    """A LeRobot v2 file is missing, unsupported, or inconsistent with the rest."""


@dataclass(frozen=True)
class EpisodeMeta:
    """One episode as ``meta/episodes.jsonl`` declares it."""

    index: int
    length: int
    tasks: tuple[str, ...]


@dataclass(frozen=True)
class LeRobotDataset:
    """A LeRobot v2 dataset's declared metadata, resolved against ``root``.

    ``data_path`` and ``video_path`` are ``info.json``'s own templates;
    ``video_path`` is ``None`` when no feature is a video, because then no
    template can describe a file the corpus does not have.
    """

    root: Path
    codebase_version: str
    fps: float
    chunks_size: int
    data_path: str
    video_path: str | None
    features: Mapping[str, Mapping[str, Any]]
    episodes: tuple[EpisodeMeta, ...]
    tasks: Mapping[int, str]

    def video_keys(self) -> tuple[str, ...]:
        """The features declared with ``dtype == "video"``, sorted."""
        return tuple(
            sorted(
                name for name, feature in self.features.items() if feature.get("dtype") == "video"
            )
        )

    def episode(self, index: int) -> EpisodeMeta:
        """Episode ``index``; an undeclared one is a format error, not a ``KeyError``."""
        for episode in self.episodes:
            if episode.index == index:
                return episode
        raise LeRobotFormatError(
            f"{self.root / _EPISODES}: episode {index} is not declared; expected one of "
            f"{[episode.index for episode in self.episodes]}"
        )

    def parquet_path(self, episode_index: int) -> Path:
        """The parquet file ``info.json``'s ``data_path`` places episode in."""
        self.episode(episode_index)
        chunk = episode_index // self.chunks_size
        name = _format_template(
            self.data_path,
            source=self.root / _INFO,
            key="data_path",
            episode_chunk=chunk,
            episode_index=episode_index,
        )
        return self.root / name

    def video_file(self, episode_index: int, video_key: str) -> Path:
        """The video file for ``episode_index``/``video_key``.

        Refuses a ``video_key`` that is not a video feature: a name like
        ``observation.state`` has no video file, and inventing one for it would
        hide a caller that mistyped its modality config.
        """
        if video_key not in self.video_keys():
            raise LeRobotFormatError(
                f"{self.root / _INFO}: episode {episode_index} video key {video_key!r} is not a "
                f"video feature; expected one of {list(self.video_keys())}"
            )
        self.episode(episode_index)
        if self.video_path is None:
            raise LeRobotFormatError(
                f"{self.root / _INFO}: episode {episode_index} video key {video_key!r} has no "
                "video_path template; expected one because video features are declared"
            )
        chunk = episode_index // self.chunks_size
        name = _format_template(
            self.video_path,
            source=self.root / _INFO,
            key="video_path",
            episode_chunk=chunk,
            episode_index=episode_index,
            video_key=video_key,
        )
        return self.root / name

    @property
    def total_frames(self) -> int:
        """Every episode's declared length, summed."""
        return sum(episode.length for episode in self.episodes)


def open_lerobot(root: str | os.PathLike[str]) -> LeRobotDataset:
    """The dataset ``root`` declares, or a :class:`LeRobotFormatError`.

    Refuses a missing ``meta/info.json``, ``meta/episodes.jsonl`` or
    ``meta/tasks.jsonl``; a ``codebase_version`` outside
    :data:`SUPPORTED_CODEBASE_VERSIONS` (v3.0 stores many episodes per file, so
    this layout cannot read it -- convert or add a v3 reader); a missing or
    non-positive ``fps`` or ``chunks_size``; a ``data_path`` without both
    ``{episode_chunk`` and ``{episode_index``; video features without a
    ``video_path`` carrying ``{video_key``; an empty ``episodes.jsonl``, a
    duplicate ``episode_index``, a non-positive ``length``; ``total_episodes`` or
    ``total_frames`` in ``info.json`` that disagree with the episodes declared;
    and a task text in ``episodes.jsonl`` that ``tasks.jsonl`` never declares.
    Every message names the file, key, episode and observed versus expected
    value -- a guessed ``fps`` or chunk size would silently mis-resolve every
    path and every timestamp check underneath it.
    """
    base = Path(root)
    info_path = base / _INFO
    episodes_path = base / _EPISODES
    tasks_path = base / _TASKS
    for path in (info_path, episodes_path, tasks_path):
        if not path.is_file():
            raise LeRobotFormatError(f"{path}: is missing; a LeRobot v2 dataset declares it")

    raw_info = _read_json(info_path)
    if not isinstance(raw_info, dict):
        raise LeRobotFormatError(
            f"{info_path}: is a {type(raw_info).__name__}; expected a JSON object"
        )
    info: Mapping[str, Any] = raw_info

    version = info.get("codebase_version")
    if not isinstance(version, str) or version not in SUPPORTED_CODEBASE_VERSIONS:
        raise LeRobotFormatError(
            f"{info_path}: key 'codebase_version' is {version!r}; expected one of "
            f"{sorted(SUPPORTED_CODEBASE_VERSIONS)} -- v3 stores many episodes per file; "
            "convert or add a v3 reader"
        )

    fps = _positive_number(info, "fps", info_path)
    chunks_size = _positive_int(info, "chunks_size", info_path)

    data_path = info.get("data_path")
    if not isinstance(data_path, str) or not data_path:
        raise LeRobotFormatError(
            f"{info_path}: key 'data_path' is {data_path!r}; expected a non-empty path template"
        )
    for placeholder in ("{episode_chunk", "{episode_index"):
        if placeholder not in data_path:
            raise LeRobotFormatError(
                f"{info_path}: key 'data_path' is {data_path!r}; expected the "
                f"{placeholder!r} placeholder so every episode resolves to its own file"
            )

    raw_features = info.get("features")
    if not isinstance(raw_features, dict):
        raise LeRobotFormatError(
            f"{info_path}: key 'features' is {raw_features!r}; expected a JSON object of features"
        )
    features: dict[str, Mapping[str, Any]] = {}
    for feature_key, feature in raw_features.items():
        if not isinstance(feature_key, str) or not isinstance(feature, dict):
            raise LeRobotFormatError(
                f"{info_path}: key 'features' entry {feature_key!r} is {feature!r}; expected a "
                "JSON object per feature"
            )
        features[feature_key] = feature

    video_features = sorted(
        name for name, feature in features.items() if feature.get("dtype") == "video"
    )
    video_path: str | None = None
    if video_features:
        candidate = info.get("video_path")
        if not isinstance(candidate, str) or not candidate:
            raise LeRobotFormatError(
                f"{info_path}: key 'video_path' is {candidate!r}; expected a path template "
                f"because features {video_features} are dtype 'video'"
            )
        if "{video_key" not in candidate:
            raise LeRobotFormatError(
                f"{info_path}: key 'video_path' is {candidate!r}; expected the "
                "'{video_key' placeholder so every video feature resolves to its own file"
            )
        video_path = candidate

    episodes: list[EpisodeMeta] = []
    seen: set[int] = set()
    for lineno, entry in _read_jsonl(episodes_path):
        if not isinstance(entry, dict):
            raise LeRobotFormatError(
                f"{episodes_path}:{lineno}: is a {type(entry).__name__}; expected a JSON object"
            )
        index = entry.get("episode_index")
        if not _is_int(index) or index < 0:
            raise LeRobotFormatError(
                f"{episodes_path}:{lineno}: key 'episode_index' is {index!r}; expected an "
                "integer >= 0"
            )
        if index in seen:
            raise LeRobotFormatError(
                f"{episodes_path}:{lineno}: episode {index} is declared twice; expected each "
                "episode_index exactly once"
            )
        length = entry.get("length")
        if not _is_int(length) or length <= 0:
            raise LeRobotFormatError(
                f"{episodes_path}:{lineno}: episode {index} length is {length!r}; expected an "
                "integer > 0"
            )
        episode_tasks = entry.get("tasks")
        if not isinstance(episode_tasks, list) or not all(
            isinstance(task, str) for task in episode_tasks
        ):
            raise LeRobotFormatError(
                f"{episodes_path}:{lineno}: episode {index} key 'tasks' is {episode_tasks!r}; "
                "expected a list of task texts"
            )
        seen.add(index)
        episodes.append(EpisodeMeta(index=index, length=length, tasks=tuple(episode_tasks)))
    if not episodes:
        raise LeRobotFormatError(f"{episodes_path}: declares no episodes; expected at least one")
    episodes.sort(key=lambda episode: episode.index)

    # A key that is present is checked even when null: a declared corpus size
    # that cannot be compared is refused, the same as a null fps.
    total_episodes = info.get("total_episodes")
    if "total_episodes" in info:
        if not _is_int(total_episodes):
            raise LeRobotFormatError(
                f"{info_path}: key 'total_episodes' is {total_episodes!r}; expected an integer "
                f"equal to the {len(episodes)} episodes {episodes_path} declares"
            )
        if total_episodes != len(episodes):
            raise LeRobotFormatError(
                f"{info_path}: key 'total_episodes' is {total_episodes}; {episodes_path} "
                f"declares {len(episodes)} episodes; expected the two to agree"
            )

    total_frames = info.get("total_frames")
    if "total_frames" in info:
        declared_frames = sum(episode.length for episode in episodes)
        if not _is_int(total_frames):
            raise LeRobotFormatError(
                f"{info_path}: key 'total_frames' is {total_frames!r}; expected an integer equal "
                f"to the {declared_frames} frames {episodes_path} declares"
            )
        if total_frames != declared_frames:
            raise LeRobotFormatError(
                f"{info_path}: key 'total_frames' is {total_frames}; the lengths in "
                f"{episodes_path} sum to {declared_frames}; expected the two to agree"
            )

    tasks: dict[int, str] = {}
    for lineno, entry in _read_jsonl(tasks_path):
        if not isinstance(entry, dict):
            raise LeRobotFormatError(
                f"{tasks_path}:{lineno}: is a {type(entry).__name__}; expected a JSON object"
            )
        task_index = entry.get("task_index")
        if not _is_int(task_index):
            raise LeRobotFormatError(
                f"{tasks_path}:{lineno}: key 'task_index' is {task_index!r}; expected an integer"
            )
        text = entry.get("task")
        if not isinstance(text, str):
            raise LeRobotFormatError(
                f"{tasks_path}:{lineno}: task {task_index} text is {text!r}; expected a string"
            )
        if task_index in tasks:
            raise LeRobotFormatError(
                f"{tasks_path}:{lineno}: task_index {task_index} is declared twice "
                f"({tasks[task_index]!r} then {text!r}); expected each task_index exactly once"
            )
        tasks[task_index] = text

    known_tasks = set(tasks.values())
    for episode in episodes:
        for text in episode.tasks:
            if text not in known_tasks:
                raise LeRobotFormatError(
                    f"{episodes_path}: episode {episode.index} task {text!r} is absent from "
                    f"{tasks_path}; expected one of {sorted(known_tasks)}"
                )

    return LeRobotDataset(
        root=base,
        codebase_version=str(version),
        fps=fps,
        chunks_size=chunks_size,
        data_path=data_path,
        video_path=video_path,
        features=features,
        episodes=tuple(episodes),
        tasks=tasks,
    )


def read_episode_columns(
    ds: LeRobotDataset,
    episode_index: int,
    columns: Sequence[str],
) -> dict[str, np.ndarray]:
    """The requested ``columns`` of episode ``episode_index``, as numpy arrays.

    Only the requested columns are returned. The episode is refused -- as
    :class:`LeRobotFormatError` -- when its parquet file is missing, a requested
    column is absent (the available ones are named), the row count disagrees with
    the episode's declared length, ``frame_index`` is not exactly
    ``0..length-1``, ``timestamp`` does not increase strictly, or a timestamp
    sits further than half a frame from ``frame_index / fps`` (the worst row is
    reported). A list column becomes a 2-D ``float32`` array of its own width and
    a numeric scalar column a 1-D array of its own dtype; ragged rows and
    non-numeric columns are refused, since padding or casting would invent
    values the corpus never stored.
    """
    import numpy as np  # noqa: PLC0415
    import pyarrow.parquet as pq  # noqa: PLC0415

    episode = ds.episode(episode_index)
    requested: list[str] = []
    for column in columns:
        if not isinstance(column, str) or not column:
            raise LeRobotFormatError(
                f"{ds.root / _INFO}: episode {episode_index} column name {column!r} is not a "
                "non-empty string; expected a feature name"
            )
        if column not in requested:
            requested.append(column)

    path = ds.parquet_path(episode_index)
    if not path.is_file():
        raise LeRobotFormatError(
            f"{path}: is missing; episode {episode_index} of {ds.root} must store its "
            f"{episode.length} rows there"
        )

    available = list(pq.read_schema(path).names)
    for name in [*requested, "frame_index", "timestamp"]:
        if name not in available:
            raise LeRobotFormatError(
                f"{path}: episode {episode_index} column {name!r} is absent; available columns "
                f"are {available}"
            )

    read_columns = list(dict.fromkeys([*requested, "frame_index", "timestamp"]))
    table = pq.read_table(path, columns=read_columns)

    rows = table.num_rows
    if rows != episode.length:
        raise LeRobotFormatError(
            f"{path}: episode {episode_index} has {rows} rows; {ds.root / _EPISODES} declares "
            f"length {episode.length}; expected the two to agree"
        )

    frame_index = np.asarray(_as_numpy(table.column("frame_index")))
    expected_index = np.arange(episode.length)
    mismatch = np.nonzero(frame_index != expected_index)[0]
    if mismatch.size:
        row = int(mismatch[0])
        raise LeRobotFormatError(
            f"{path}: episode {episode_index} column 'frame_index' row {row} is "
            f"{frame_index[row].item()!r}; expected {row} (frame_index must be exactly "
            f"0..{episode.length - 1})"
        )

    timestamp = np.asarray(_as_numpy(table.column("timestamp")), dtype=np.float64)
    flat = np.nonzero(~(timestamp[1:] > timestamp[:-1]))[0]
    if flat.size:
        row = int(flat[0]) + 1
        raise LeRobotFormatError(
            f"{path}: episode {episode_index} column 'timestamp' row {row} is "
            f"{timestamp[row].item()!r}; expected strictly greater than row {row - 1} value "
            f"{timestamp[row - 1].item()!r}"
        )

    tolerance = 0.5 / ds.fps
    expected_time = frame_index.astype(np.float64) / ds.fps
    delta = np.abs(timestamp - expected_time)
    worst = int(np.argmax(delta))
    worst_delta = float(delta[worst])
    if not math.isfinite(worst_delta) or worst_delta > tolerance:
        raise LeRobotFormatError(
            f"{path}: episode {episode_index} column 'timestamp' row {worst} is "
            f"{timestamp[worst].item()!r}; expected {expected_time[worst].item()!r} "
            f"(frame_index {frame_index[worst].item()!r} / fps {ds.fps}) within {tolerance!r}s, "
            f"observed {worst_delta!r}s off (worst row {worst})"
        )

    out: dict[str, np.ndarray] = {}
    for name in requested:
        out[name] = _column_array(table.column(name), name, path, episode_index)
    return out


# -- fail-closed readers and converters ---------------------------------------------


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise LeRobotFormatError(f"{path}: is not valid JSON ({exc})") from exc


def _read_jsonl(path: Path) -> list[tuple[int, Any]]:
    entries: list[tuple[int, Any]] = []
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise LeRobotFormatError(f"{path}: is not UTF-8 text ({exc})") from exc
    for lineno, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            entries.append((lineno, json.loads(line)))
        except json.JSONDecodeError as exc:
            raise LeRobotFormatError(f"{path}:{lineno}: is not valid JSON ({exc})") from exc
    return entries


def _is_int(value: Any) -> TypeGuard[int]:
    return isinstance(value, int) and not isinstance(value, bool)


def _positive_number(info: Mapping[str, Any], key: str, path: Path) -> float:
    if key not in info:
        raise LeRobotFormatError(f"{path}: key {key!r} is missing; expected a number > 0")
    value = info[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise LeRobotFormatError(
            f"{path}: key {key!r} is {value!r} ({type(value).__name__}); expected a number > 0"
        )
    if not math.isfinite(value) or value <= 0:
        raise LeRobotFormatError(f"{path}: key {key!r} is {value!r}; expected a number > 0")
    return float(value)


def _positive_int(info: Mapping[str, Any], key: str, path: Path) -> int:
    if key not in info:
        raise LeRobotFormatError(f"{path}: key {key!r} is missing; expected an integer > 0")
    value = info[key]
    if not _is_int(value) or value <= 0:
        raise LeRobotFormatError(f"{path}: key {key!r} is {value!r}; expected an integer > 0")
    return value


def _format_template(template: str, *, source: Path, key: str, **fields: Any) -> str:
    try:
        return template.format(**fields)
    except (KeyError, IndexError, ValueError, AttributeError, TypeError) as exc:
        raise LeRobotFormatError(
            f"{source}: key {key!r} template {template!r} could not be formatted with "
            f"{fields} ({exc})"
        ) from exc


def _as_numpy(column: Any) -> Any:
    """A pyarrow column as one numpy array, whatever its chunking."""
    import numpy as np  # noqa: PLC0415
    import pyarrow as pa  # noqa: PLC0415

    chunks = list(column.iterchunks()) if isinstance(column, pa.ChunkedArray) else [column]
    if not chunks:
        return np.empty(0, dtype=np.float64)
    if len(chunks) == 1:
        return chunks[0].to_numpy(zero_copy_only=False)
    return pa.concat_arrays(chunks).to_numpy(zero_copy_only=False)


def _column_array(column: Any, name: str, path: Path, episode_index: int) -> np.ndarray:
    """One parquet column as an array: lists 2-D ``float32``, numeric scalars 1-D."""
    import numpy as np  # noqa: PLC0415
    import pyarrow as pa  # noqa: PLC0415

    column_type = column.type
    is_list_type = (
        pa.types.is_fixed_size_list(column_type)
        or pa.types.is_list(column_type)
        or pa.types.is_large_list(column_type)
    )
    if is_list_type:
        value_type = column_type.value_type
        if not (pa.types.is_integer(value_type) or pa.types.is_floating(value_type)):
            raise LeRobotFormatError(
                f"{path}: episode {episode_index} column {name!r} has type {column_type}; "
                "expected a list of numbers (only numeric values become an array)"
            )
        declared = column_type.list_size if pa.types.is_fixed_size_list(column_type) else None
        width = declared
        rows: list[list[Any]] = []
        for row_index, row in enumerate(column.to_pylist()):
            if not isinstance(row, (list, tuple)):
                raise LeRobotFormatError(
                    f"{path}: episode {episode_index} column {name!r} row {row_index} is "
                    f"{row!r}; expected a list of {declared if declared is not None else 'numbers'}"
                )
            if width is None:
                width = len(row)
            elif len(row) != width:
                raise LeRobotFormatError(
                    f"{path}: episode {episode_index} column {name!r} row {row_index} has "
                    f"{len(row)} values; expected {width} (ragged rows are refused: padding "
                    "would invent values the corpus never stored)"
                )
            rows.append(list(row))
        return np.asarray(rows, dtype=np.float32).reshape(len(rows), width or 0)

    if not (pa.types.is_integer(column_type) or pa.types.is_floating(column_type)):
        raise LeRobotFormatError(
            f"{path}: episode {episode_index} column {name!r} has type {column_type}; expected "
            "a numeric or list-of-numbers column (strings, booleans and structs are refused: "
            "they are not numbers to read as arrays)"
        )
    return np.asarray(_as_numpy(column))
