"""Norm statistics over every frame of every episode, in GR00T's exact semantics.

Isaac-GR00T normalises with ``gr00t/data/stats.py::calculate_dataset_statistics``:
over EVERY frame of every episode, on float32 values, per dimension the mean, the
population standard deviation (``ddof=0``), the min, the max and the 0.01 and 0.99
quantiles (numpy's default method). This module computes exactly those six numbers
per key and keeps them as plain JSON, so a policy can be checked against the
normalisation it was trained under instead of trusting a file nobody can read.

Nothing is guessed and nothing is filled in. A key that is not a 1-D float feature
is refused -- GR00T normalises per-dim float vectors, and a video, an int column or
a matrix is not one. An empty key list or an empty episode selection is refused,
because "no frames" is not a statistic. An unknown episode index is refused rather
than silently skipped, and a column whose width disagrees with the feature shape is
refused, because the two together describe a different dataset than the one named.
``stats_from_json`` refuses a record missing any of the six statistics, statistics
of differing lengths and non-finite values: a NaN in the stats would silently poison
every normalised frame.

numpy is imported function-locally: the core install has none of it.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from foundationscale.vla.lerobot import LeRobotDataset, LeRobotFormatError, read_episode_columns

__all__ = [
    "DEGENERATE_STD",
    "STAT_NAMES",
    "NormStats",
    "compute_norm_stats",
    "stats_digest",
    "stats_from_json",
    "stats_to_json",
]

STAT_NAMES: tuple[str, ...] = ("mean", "std", "min", "max", "q01", "q99")
DEGENERATE_STD = 1e-6
# The dtypes GR00T normalises; anything else (int, bool, video, image) is refused.
_FLOAT_DTYPES: frozenset[str] = frozenset({"float16", "float32", "float64"})


@dataclass(frozen=True)
class NormStats:
    """The six per-dimension statistics of one key, over ``frames`` frames.

    Every tuple is one value per dimension and they all share ``dim``; a dimension
    whose ``std`` falls below :data:`DEGENERATE_STD` is reported by
    :meth:`degenerate_dims` because dividing by it would blow the frame up.
    """

    key: str
    frames: int
    episodes: int
    mean: tuple[float, ...]
    std: tuple[float, ...]
    min: tuple[float, ...]
    max: tuple[float, ...]
    q01: tuple[float, ...]
    q99: tuple[float, ...]

    @property
    def dim(self) -> int:
        """The number of dimensions these statistics cover."""
        return len(self.mean)

    def degenerate_dims(self) -> tuple[int, ...]:
        """Dimensions whose std is below :data:`DEGENERATE_STD` (constant columns)."""
        return tuple(index for index, value in enumerate(self.std) if value < DEGENERATE_STD)


def _shape_dim(shape: Any) -> int | None:
    """The single dimension of a 1-D feature shape, or ``None`` when it is not one."""
    # Only a one-element list is a 1-D shape: LeRobot writes ``[n]``, and a bare
    # int is a malformed declaration, refused rather than read as one.
    if isinstance(shape, Sequence) and not isinstance(shape, (str, bytes)):
        if len(shape) != 1:
            return None
        first = shape[0]
        if isinstance(first, bool) or not isinstance(first, int):
            return None
        return first if first >= 1 else None
    return None


def _feature_dim(ds: LeRobotDataset, key: str) -> int:
    """The per-frame dimension of ``key``, refused unless it is a 1-D float feature."""
    feature = ds.features.get(key)
    if feature is None:
        raise LeRobotFormatError(
            f"{ds.root}: norm stats key {key!r} is not a feature of the dataset "
            f"(expected one of {sorted(ds.features)})"
        )
    dtype = feature.get("dtype")
    shape = feature.get("shape")
    dim = _shape_dim(shape)
    if not isinstance(dtype, str) or dtype not in _FLOAT_DTYPES or dim is None:
        raise LeRobotFormatError(
            f"{ds.root}: norm stats key {key!r} must be a 1-D float feature "
            f"(dtype in {sorted(_FLOAT_DTYPES)}, shape [n] with n >= 1); "
            f"got dtype={dtype!r} shape={shape!r}"
        )
    return dim


def _frame_matrix(array: Any, *, key: str, dim: int, episode: int, root: Path) -> Any:
    """``array`` as a ``[frames, dim]`` matrix, refusing a shape the feature denies."""
    if array.ndim == 1:
        matrix = array.reshape(-1, 1)
    elif array.ndim == 2:
        matrix = array
    else:
        raise LeRobotFormatError(
            f"{root}: episode {episode}, column {key!r}: read a {array.ndim}-D array, "
            f"expected one value per frame per dim of the feature shape [{dim}]"
        )
    if matrix.shape[1] != dim:
        raise LeRobotFormatError(
            f"{root}: episode {episode}, column {key!r}: read {matrix.shape[1]} dim(s) per frame, "
            f"but the feature shape is [{dim}] (observed {matrix.shape[1]}, expected {dim})"
        )
    return matrix


def _vector(values: Any) -> tuple[float, ...]:
    """A numpy per-dim result as plain floats (a float32 widens to float exactly)."""
    import numpy as np  # noqa: PLC0415

    return tuple(float(value) for value in np.asarray(values).reshape(-1))


def compute_norm_stats(
    ds: LeRobotDataset,
    keys: Sequence[str],
    *,
    episodes: Sequence[int] | None = None,
) -> dict[str, NormStats]:
    """GR00T's statistics for each of ``keys`` over the selected episodes' frames.

    ``episodes=None`` means every episode; otherwise exactly the listed episodes,
    read in the order given. Each key is measured over the concatenated frames as
    float32: mean, population std (``ddof=0``), min, max and the 0.01/0.99 quantiles
    with numpy's default method -- the same calls
    ``gr00t/data/stats.py::calculate_dataset_statistics`` makes.

    Raises :class:`LeRobotFormatError` when a key is not a 1-D float feature of the
    dataset, or when a read column does not carry the feature's dimensions -- the
    numbers would then describe something other than the declared feature. Raises
    ``ValueError`` when ``keys`` or the episode selection is empty (there would be
    nothing to measure) or when an episode index is unknown (skipping it silently
    would change the corpus the statistics claim to describe).
    """
    import numpy as np  # noqa: PLC0415

    wanted = list(keys)
    if not wanted:
        raise ValueError("compute_norm_stats: keys is empty; there is nothing to measure")
    if episodes is not None and len(episodes) == 0:
        raise ValueError(
            "compute_norm_stats: episodes=[] selects no episode; there is nothing to measure"
        )
    dims = {key: _feature_dim(ds, key) for key in wanted}
    known = {meta.index: meta for meta in ds.episodes}
    if episodes is None:
        selection = tuple(sorted(known))
    else:
        selection = tuple(episodes)
        for index in selection:
            if index not in known:
                raise ValueError(
                    f"{ds.root}: episode {index} is unknown; the dataset holds episodes "
                    f"{sorted(known)}"
                )
    collected: dict[str, list[Any]] = {key: [] for key in wanted}
    for index in selection:
        columns = read_episode_columns(ds, index, wanted)
        for key in wanted:
            collected[key].append(
                _frame_matrix(
                    np.asarray(columns[key]),
                    key=key,
                    dim=dims[key],
                    episode=index,
                    root=ds.root,
                )
            )
    stats: dict[str, NormStats] = {}
    for key in wanted:
        values = np.concatenate(collected[key], axis=0).astype(np.float32, copy=False)
        stats[key] = NormStats(
            key=key,
            frames=int(values.shape[0]),
            episodes=len(selection),
            mean=_vector(values.mean(axis=0)),
            std=_vector(values.std(axis=0)),
            min=_vector(values.min(axis=0)),
            max=_vector(values.max(axis=0)),
            q01=_vector(np.quantile(values, 0.01, axis=0)),
            q99=_vector(np.quantile(values, 0.99, axis=0)),
        )
    return stats


def stats_to_json(stats: Mapping[str, NormStats]) -> dict[str, Any]:
    """``stats`` as JSON-ready ``key -> {frames, episodes, <stat> -> [float, ...]}``.

    The values are plain Python floats and lists, so the record survives a JSON
    round trip unchanged and :func:`stats_from_json` rebuilds the same objects.
    """
    out: dict[str, Any] = {}
    for key, entry in stats.items():
        record: dict[str, Any] = {"frames": entry.frames, "episodes": entry.episodes}
        for name in STAT_NAMES:
            record[name] = [float(value) for value in getattr(entry, name)]
        out[key] = record
    return out


def _positive_int(value: Any, *, key: str, field: str) -> int:
    """A count from a JSON record, refused unless it is a positive int."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"norm stats {key!r}.{field}: expected an int, got {value!r}")
    if value < 1:
        raise ValueError(f"norm stats {key!r}.{field}: expected >= 1, got {value}")
    return value


def stats_from_json(data: Mapping[str, Any]) -> dict[str, NormStats]:
    """The :class:`NormStats` behind a :func:`stats_to_json` record.

    A record missing any of ``STAT_NAMES`` is refused (half a record would normalise
    half the pipeline), statistics of differing lengths are refused (every statistic
    must cover the same dimensions), and a non-finite value is refused (a NaN there
    would poison every normalised frame). ``frames`` and ``episodes`` must be
    positive ints. The result of a round trip equals the input exactly.
    """
    out: dict[str, NormStats] = {}
    for key, record in data.items():
        if not isinstance(record, Mapping):
            raise ValueError(
                f"norm stats {key!r}: expected an object of statistics, got {type(record).__name__}"
            )
        required = ("frames", "episodes", *STAT_NAMES)
        missing = [name for name in required if name not in record]
        if missing:
            raise ValueError(f"norm stats {key!r}: missing {missing}; expected {list(required)}")
        counts: dict[str, int] = {}
        vectors: dict[str, tuple[float, ...]] = {}
        for name in STAT_NAMES:
            raw = record[name]
            if isinstance(raw, (str, bytes)) or not isinstance(raw, Sequence):
                raise ValueError(
                    f"norm stats {key!r}.{name}: expected a list of numbers, got {raw!r}"
                )
            values: list[float] = []
            for position, item in enumerate(raw):
                if isinstance(item, bool) or not isinstance(item, (int, float)):
                    raise ValueError(
                        f"norm stats {key!r}.{name}[{position}]: expected a number, got {item!r}"
                    )
                number = float(item)
                if not math.isfinite(number):
                    raise ValueError(
                        f"norm stats {key!r}.{name}[{position}] = {number!r} is not finite"
                    )
                values.append(number)
            counts[name] = len(values)
            vectors[name] = tuple(values)
        if len(set(counts.values())) != 1:
            raise ValueError(
                f"norm stats {key!r}: the statistics have different lengths {counts}; "
                "every statistic must cover the same dimensions"
            )
        if counts["mean"] < 1:
            raise ValueError(
                f"norm stats {key!r}: the statistics are empty; expected at least one dim"
            )
        out[key] = NormStats(
            key=key,
            frames=_positive_int(record["frames"], key=key, field="frames"),
            episodes=_positive_int(record["episodes"], key=key, field="episodes"),
            mean=vectors["mean"],
            std=vectors["std"],
            min=vectors["min"],
            max=vectors["max"],
            q01=vectors["q01"],
            q99=vectors["q99"],
        )
    return out


def stats_digest(stats: Mapping[str, NormStats]) -> str:
    """The sha256 hex digest of ``stats`` in canonical JSON (sorted, no spaces).

    Canonical form makes the digest a property of the numbers alone: the same
    statistics digest the same however the mapping was built or ordered.
    """
    payload = json.dumps(stats_to_json(stats), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
