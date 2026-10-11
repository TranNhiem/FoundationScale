"""One VLA training sample: the frames at an anchor, the state there, and the action chunk to learn.

A sample is what a policy trains on. At ``anchor`` of one episode it carries the decoded frames at
the anchor's ``video_delta`` indices (``images``), the normalised state rows at its ``state_delta``
indices (``state``), the normalised action rows at its ``action_delta`` indices (``action``) and
``action_valid``, which marks the action steps that came from real frames. ``action_valid`` exists
because ``pad="edge"`` repeats the last frame past the end of an episode: the repeated tail is
convenient batching, but a loss that counts it teaches the policy to repeat a frame forever.

Nothing here guesses a value. ``pad`` has no default -- "refuse" and "edge" hand the policy
different data, so the caller declares which one it wants -- and an anchor outside the range
:func:`~foundationscale.vla.chunking.valid_anchor_indices` allows for that mode is a
:class:`SampleError`. So is a modality whose annotation group carries no ``original_key``
``task_index`` (there would be no task text to attach), a ``task_index`` that
``meta/tasks.jsonl`` does not declare (the task would be invented), a slice that runs past its
column, and a normalisation key or width the statistics do not carry.

:class:`Normalizer` implements GR00T's three normalisation modes over the per-column statistics of
:mod:`foundationscale.vla.norm`, and refuses the scales that would destroy the data rather than
normalise it: a degenerate ``std`` under ``mean_std`` and a degenerate ``hi - lo`` under
``q01_q99``/``min_max`` both divide by (almost) zero, and the offending dims are named.

numpy is imported function-locally: the core install has none of it.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, TypeGuard

from foundationscale.vla.chunking import ChunkSpec, valid_anchor_indices
from foundationscale.vla.frames import decode_frames
from foundationscale.vla.lerobot import LeRobotDataset, read_episode_columns
from foundationscale.vla.modality import ModalitySpec, SliceSpec
from foundationscale.vla.norm import DEGENERATE_STD, STAT_NAMES, NormStats

if TYPE_CHECKING:
    import numpy as np

__all__ = [
    "MODES",
    "PAD_MODES",
    "Normalizer",
    "SampleError",
    "VlaSample",
    "build_sample",
]

MODES: tuple[str, ...] = ("mean_std", "q01_q99", "min_max")
PAD_MODES: tuple[str, ...] = ("refuse", "edge")
# The annotation ``original_key`` whose column carries the task index into ``meta/tasks.jsonl``.
_TASK_KEY = "task_index"


@dataclass(frozen=True)
class VlaSample:
    """One training sample at ``anchor`` of episode ``episode_index``.

    ``images`` maps each modality video name to its decoded frames, one per ``video_delta``;
    ``state`` is ``float32 [len(state_delta), state_dim]`` and ``action`` is
    ``float32 [action_horizon, action_dim]``, both normalised; ``action_valid`` is a bool mask over
    the action steps, ``False`` where ``pad="edge"`` clamped the step onto the episode's last frame.
    """

    episode_index: int
    anchor: int
    task: str
    images: Mapping[str, tuple[np.ndarray, ...]]
    state: np.ndarray
    action: np.ndarray
    action_valid: np.ndarray


class SampleError(ValueError):
    """A training sample, or its normalisation, is not the one the declared configs describe."""


class Normalizer:
    """Per-key normalisation of state/action vectors under one declared ``mode``.

    ``mean_std`` maps ``x`` to ``(x - mean) / std``. ``q01_q99`` and ``min_max`` map the
    ``[lo, hi]`` of the chosen statistics onto ``[-1, 1]`` and clip to that range, the way
    a frame beyond the range the corpus actually showed is saturated. :meth:`unapply` is
    the exact inverse for ``mean_std`` and the inverse of the affine map for the other two:
    a clipped value has no exact inverse, and inventing one would claim to recover data the
    normalisation destroyed.

    :meth:`apply` and :meth:`unapply` refuse a key the statistics do not carry (there is no
    scale to use), a frame whose last dim is not the statistics' dim (reshaping it would mix
    dims), and degenerate statistics for the mode -- a ``std`` below :data:`DEGENERATE_STD`
    under ``mean_std``, or a ``hi - lo`` below it under ``q01_q99``/``min_max``, with the
    offending dims named. Either degenerate scale divides by (almost) zero and would blow
    every frame up. Results are ``float32``, the dtype the sample carries.
    """

    def __init__(
        self,
        stats: Mapping[str, NormStats],
        *,
        mode: Literal["mean_std", "q01_q99", "min_max"],
    ) -> None:
        """Store ``stats`` under ``mode``; an unknown ``mode`` is refused, never defaulted."""
        if mode not in MODES:
            raise SampleError(f"normalizer mode {mode!r} is unknown; expected one of {list(MODES)}")
        entries: dict[str, NormStats] = {}
        for key, entry in stats.items():
            if not isinstance(key, str) or not key:
                raise SampleError(
                    f"normalizer stats key {key!r} is not a non-empty string; expected a dataset "
                    "column name"
                )
            if not isinstance(entry, NormStats):
                raise SampleError(
                    f"normalizer stats {key!r} is a {type(entry).__name__}; expected a NormStats"
                )
            entries[key] = entry
        self._stats: Mapping[str, NormStats] = entries
        self._mode = mode

    @property
    def mode(self) -> str:
        """The declared normalisation mode."""
        return self._mode

    @property
    def stats(self) -> Mapping[str, NormStats]:
        """The statistics this normalizer was built from, keyed by dataset column."""
        return self._stats

    def apply(self, key: str, x: Any) -> np.ndarray:
        """``x`` mapped into the normalised range under ``key``'s statistics.

        ``mean_std`` returns ``(x - mean) / std``; ``q01_q99`` and ``min_max`` map ``[lo, hi]`` onto
        ``[-1, 1]`` and clip to it. Refuses an unknown ``key``, a frame whose last dim is not the
        statistics' dim, and degenerate statistics for this mode (see the class docstring).
        """
        import numpy as np  # noqa: PLC0415

        values, entry = self._frame(key, x)
        if self._mode == "mean_std":
            mean, std = self._mean_std_of(entry)
            out = (values - mean) / std
        else:
            lo, hi = self._range_of(entry)
            out = np.clip(2.0 * (values - lo) / (hi - lo) - 1.0, -1.0, 1.0)
        return np.asarray(out, dtype=np.float32)

    def unapply(self, key: str, y: Any) -> np.ndarray:
        """``y`` mapped back to data space under ``key``'s statistics.

        ``mean_std`` returns ``y * std + mean``, the exact inverse of :meth:`apply`; the other modes
        invert the affine map ``[lo, hi] -> [-1, 1]`` without unclipping, because a clipped value
        does not say how far outside the range it came from. Refuses the same cases as
        :meth:`apply`.
        """
        import numpy as np  # noqa: PLC0415

        values, entry = self._frame(key, y)
        if self._mode == "mean_std":
            mean, std = self._mean_std_of(entry)
            out = values * std + mean
        else:
            lo, hi = self._range_of(entry)
            out = (values + 1.0) / 2.0 * (hi - lo) + lo
        return np.asarray(out, dtype=np.float32)

    def _frame(self, key: str, x: Any) -> tuple[Any, NormStats]:
        """``x`` as a float array whose last dim is ``key``'s declared dim, plus its statistics."""
        import numpy as np  # noqa: PLC0415

        entry = self._checked(key)
        try:
            values = np.asarray(x, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise SampleError(
                f"normalizer key {key!r}: x of type {type(x).__name__} is not a numeric array "
                f"({exc})"
            ) from exc
        observed = values.shape[-1] if values.ndim >= 1 else None
        if values.ndim < 1 or values.shape[-1] != entry.dim:
            raise SampleError(
                f"normalizer key {key!r}: x has shape {values.shape}; expected the last dim to be "
                f"{entry.dim} (the dims the statistics declare), observed {observed}"
            )
        return values, entry

    def _checked(self, key: str) -> NormStats:
        """``key``'s statistics, refused when unknown, ragged or degenerate for this mode."""
        entry = self._stats.get(key)
        if entry is None:
            raise SampleError(
                f"normalizer key {key!r} is unknown; the statistics declare {sorted(self._stats)}"
            )
        lengths = {name: len(getattr(entry, name)) for name in STAT_NAMES}
        if len(set(lengths.values())) != 1 or lengths["mean"] < 1:
            raise SampleError(
                f"normalizer key {key!r}: the statistics have dims {lengths}; expected every "
                f"statistic in {list(STAT_NAMES)} to cover the same, non-empty dims"
            )
        if self._mode == "mean_std":
            bad = [index for index, value in enumerate(entry.std) if value < DEGENERATE_STD]
            if bad:
                raise SampleError(
                    f"normalizer key {key!r} mode 'mean_std' has a degenerate std below "
                    f"DEGENERATE_STD ({DEGENERATE_STD}) on dims {bad} (std "
                    f"{[entry.std[index] for index in bad]}); dividing by it would blow every "
                    "frame up"
                )
        else:
            lo, hi = self._range_of(entry)
            bad = [
                index
                for index, (low, high) in enumerate(zip(lo, hi, strict=True))
                if high - low < DEGENERATE_STD
            ]
            if bad:
                raise SampleError(
                    f"normalizer key {key!r} mode {self._mode!r} has a degenerate hi - lo below "
                    f"DEGENERATE_STD ({DEGENERATE_STD}) on dims {bad} (lo "
                    f"{[float(lo[index]) for index in bad]}, hi "
                    f"{[float(hi[index]) for index in bad]}); mapping it onto [-1, 1] would blow "
                    "every frame up"
                )
        return entry

    def _mean_std_of(self, entry: NormStats) -> tuple[Any, Any]:
        """The ``mean``/``std`` of ``entry`` as float64 vectors."""
        import numpy as np  # noqa: PLC0415

        return np.asarray(entry.mean, dtype=np.float64), np.asarray(entry.std, dtype=np.float64)

    def _range_of(self, entry: NormStats) -> tuple[Any, Any]:
        """The ``[lo, hi]`` this mode maps onto ``[-1, 1]``, as float64 vectors."""
        import numpy as np  # noqa: PLC0415

        if self._mode == "q01_q99":
            lo, hi = entry.q01, entry.q99
        else:
            lo, hi = entry.min, entry.max
        return np.asarray(lo, dtype=np.float64), np.asarray(hi, dtype=np.float64)


def build_sample(
    ds: LeRobotDataset,
    spec: ModalitySpec,
    chunks: ChunkSpec,
    norm: Normalizer,
    episode_index: int,
    anchor: int,
    *,
    pad: Literal["refuse", "edge"],
) -> VlaSample:
    """The training sample at ``anchor`` of ``episode_index``, or a :class:`SampleError`.

    ``state`` and ``action`` are the ``ModalitySpec`` slices of their ``original_key`` columns, read
    with :func:`~foundationscale.vla.lerobot.read_episode_columns` and concatenated in file order.
    Each column is normalised as a whole before it is sliced: the statistics describe the whole
    column, so normalising a slice against them would mix one vector's scale into another's dims.
    ``images`` decodes each modality video at the anchor's ``video_delta`` frames, and ``task`` is
    the ``task_index`` at the anchor mapped through ``ds.tasks``.

    Refuses a non-int ``episode_index`` or ``anchor``, a ``pad`` outside :data:`PAD_MODES` (there is
    no default), an anchor :func:`~foundationscale.vla.chunking.valid_anchor_indices` does not allow
    for that ``pad``, a modality whose annotation group has no ``task_index`` column, a
    ``task_index`` absent from ``meta/tasks.jsonl``, a slice past its column's width, and a
    normalisation key or width the statistics do not carry -- each would hand the policy a silently
    wrong task, vector or scale. An unknown episode is refused by
    :meth:`~foundationscale.vla.lerobot.LeRobotDataset.episode` as a
    :class:`~foundationscale.vla.lerobot.LeRobotFormatError`.
    """
    import numpy as np  # noqa: PLC0415

    if not _is_int(episode_index):
        raise SampleError(f"{ds.root}: episode_index is {episode_index!r}; expected an int")
    if not _is_int(anchor):
        raise SampleError(
            f"{ds.root}: episode {episode_index} anchor is {anchor!r}; expected an int"
        )
    if pad not in PAD_MODES:
        raise SampleError(
            f"{ds.root}: episode {episode_index} anchor {anchor} pad is {pad!r}; expected one of "
            f"{list(PAD_MODES)} -- there is no default, the caller declares the pad mode"
        )
    episode = ds.episode(episode_index)
    allowed = valid_anchor_indices(episode.length, chunks, pad=pad)
    if anchor not in allowed:
        raise SampleError(
            f"{ds.root}: episode {episode_index} anchor {anchor} is not allowed for pad {pad!r}; "
            f"expected an anchor in [{allowed.start}, {allowed.stop}) of the episode's "
            f"{episode.length} frames"
        )

    task_column = _task_column(spec)
    wanted: list[str] = []
    for key in (
        *[item.original_key for item in spec.state],
        *[item.original_key for item in spec.action],
        task_column,
    ):
        if key not in wanted:
            wanted.append(key)
    columns = read_episode_columns(ds, episode_index, wanted)

    state_rows = _rows(
        anchor, chunks.state_delta, episode.length, pad, group="state", ds=ds, episode=episode_index
    )
    action_rows = _rows(
        anchor,
        chunks.action_delta,
        episode.length,
        pad,
        group="action",
        ds=ds,
        episode=episode_index,
    )
    video_rows = _rows(
        anchor, chunks.video_delta, episode.length, pad, group="video", ds=ds, episode=episode_index
    )

    images: dict[str, tuple[np.ndarray, ...]] = {}
    for name, key in spec.video.items():
        frames = decode_frames(
            ds.video_file(episode_index, key), video_rows, expected_length=episode.length
        )
        images[name] = tuple(frames)

    action_valid = np.asarray(
        [0 <= anchor + delta < episode.length for delta in chunks.action_delta], dtype=bool
    )
    return VlaSample(
        episode_index=episode_index,
        anchor=anchor,
        task=_task_text(ds, columns, anchor, episode_index),
        images=images,
        state=_assemble(
            "state",
            spec.state,
            columns,
            state_rows,
            norm,
            ds=ds,
            episode=episode_index,
        ),
        action=_assemble(
            "action",
            spec.action,
            columns,
            action_rows,
            norm,
            ds=ds,
            episode=episode_index,
        ),
        action_valid=action_valid,
    )


# -- fail-closed helpers ---------------------------------------------------------------


def _is_int(value: Any) -> TypeGuard[int]:
    """``True`` for a real int: ``bool`` is refused, it is not an episode or a frame index."""
    return isinstance(value, int) and not isinstance(value, bool)


def _rows(
    anchor: int,
    deltas: Sequence[int],
    length: int,
    pad: str,
    *,
    group: str,
    ds: LeRobotDataset,
    episode: int,
) -> list[int]:
    """The frame index of each ``anchor + delta``, clamped to the episode only under ``edge``."""
    rows: list[int] = []
    for delta in deltas:
        index = anchor + delta
        if 0 <= index < length:
            rows.append(index)
        elif pad == "edge":
            rows.append(min(max(index, 0), length - 1))
        else:
            raise SampleError(
                f"{ds.root}: episode {episode} anchor {anchor} {group} delta {delta} lands on "
                f"frame {index} outside [0, {length}); pad 'refuse' never pads, so this anchor "
                "should have been refused"
            )
    return rows


def _assemble(
    group: str,
    slices: Sequence[SliceSpec],
    columns: Mapping[str, np.ndarray],
    rows: Sequence[int],
    norm: Normalizer,
    *,
    ds: LeRobotDataset,
    episode: int,
) -> np.ndarray:
    """The ``[len(rows), dim]`` block ``slices`` build, in file order.

    Each source column is normalised whole, then sliced.
    """
    import numpy as np  # noqa: PLC0415

    matrices: dict[str, np.ndarray] = {}
    for item in slices:
        key = item.original_key
        if key not in matrices:
            column = np.asarray(columns[key])
            if column.ndim == 1:
                column = column.reshape(-1, 1)
            elif column.ndim != 2:
                raise SampleError(
                    f"{ds.root}: episode {episode} column {key!r} is a {column.ndim}-D array; "
                    "expected one value per frame per dim"
                )
            matrices[key] = column
        width = int(matrices[key].shape[1])
        if item.end > width:
            raise SampleError(
                f"{ds.root}: episode {episode} modality {group} slice {item.name!r} spans "
                f"[{item.start}, {item.end}) of original_key {key!r}; the column carries {width} "
                f"dim(s) per frame (observed end {item.end}, expected <= {width})"
            )
    if not slices:
        return np.zeros((len(rows), 0), dtype=np.float32)
    normalized = {key: norm.apply(key, matrix[list(rows)]) for key, matrix in matrices.items()}
    blocks = [normalized[item.original_key][:, item.start : item.end] for item in slices]
    return np.asarray(np.concatenate(blocks, axis=1), dtype=np.float32)


def _task_column(spec: ModalitySpec) -> str:
    """The ``original_key`` of the annotation entry that carries the task index column."""
    for key in spec.annotation.values():
        if key == _TASK_KEY:
            return key
    raise SampleError(
        f"modality annotation {dict(spec.annotation)!r} has no entry with original_key "
        f"{_TASK_KEY!r}; the task text is read from that column, so a modality without it cannot "
        "name the task it trains on"
    )


def _task_text(
    ds: LeRobotDataset,
    columns: Mapping[str, np.ndarray],
    anchor: int,
    episode: int,
) -> str:
    """The task text at ``anchor``, through the ``task_index`` column and ``meta/tasks.jsonl``."""
    import numpy as np  # noqa: PLC0415

    column = np.asarray(columns[_TASK_KEY])
    if column.ndim != 1:
        raise SampleError(
            f"{ds.root}: episode {episode} column {_TASK_KEY!r} is a {column.ndim}-D array; "
            "expected one task index per frame"
        )
    raw = column[anchor]
    try:
        number = float(raw)
    except (TypeError, ValueError) as exc:
        raise SampleError(
            f"{ds.root}: episode {episode} anchor {anchor} {_TASK_KEY} is {raw!r}; expected an "
            f"integer task index ({exc})"
        ) from exc
    if not math.isfinite(number) or not number.is_integer():
        raise SampleError(
            f"{ds.root}: episode {episode} anchor {anchor} {_TASK_KEY} is {raw!r}; expected an "
            "integer task index"
        )
    index = int(number)
    if index not in ds.tasks:
        raise SampleError(
            f"{ds.root}: episode {episode} anchor {anchor} {_TASK_KEY} {index} is absent from "
            f"meta/tasks.jsonl; expected one of {sorted(ds.tasks)}"
        )
    return ds.tasks[index]
