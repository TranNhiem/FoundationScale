"""GR00T's own observation contract, read from its checkpoint, and the dict it builds from our data.

A GR00T N1.x checkpoint ships ``processor_config.json``: the modality contract the policy trained
with -- the ``delta_indices`` each group draws from an anchor frame, the modality keys feeding its
state, action, video and language inputs, and the representation each action key is in. This module
reads that file (JSON only; the ``gr00t`` package is never imported) into a :class:`Gr00tContract`,
maps its frame offsets onto :class:`~foundationscale.vla.chunking.ChunkSpec`, checks it against a
:class:`~foundationscale.vla.modality.ModalitySpec`, and builds GR00T's observation dict --
``state.<key>``, ``video.<key>`` and each language key verbatim -- from FoundationScale's data path.
A new GR00T release therefore drops in by reading its checkpoint instead of hand-copying configs: on
GB200 (evidence S3) the observations built here equal GR00T's ``LeRobotEpisodeLoader`` +
``extract_step_data`` bit for bit, and the same seed gives identical actions.

Nothing here is guessed and nothing is defaulted. A checkpoint with no processor config at either of
GR00T's layouts, a config that is not valid JSON, one without ``processor_kwargs.modality_configs``,
an embodiment it does not name (the available ones are listed, sorted), a modality group it omits,
``delta_indices`` that are not a non-empty list of ints, ``modality_keys`` that are not a non-empty
list of unique strings, an ``action_configs`` that is null/absent or does not carry exactly one
entry per action key, or a ``rep`` outside :data:`ACTION_REPS` is refused with
:class:`Gr00tContractError` naming the file, the key and the value observed: a guessed offset or
key would feed the policy frames and dims it never trained on. ``ChunkSpec``'s own refusals -- a
duplicated or decreasing offset, a negative or gapped action horizon -- are re-raised as
:class:`Gr00tContractError` with the source path prefixed, so the message names the config that
declared the offsets.

Against a dataset the same rule holds: a contract key the modality spec has no slice for, an action
rep other than ``"ABSOLUTE"`` (``RELATIVE``/``DELTA`` need the ``state_key`` reference and are not
supported yet -- reading them as absolute would silently change the action space), a modality whose
annotation group carries no ``task_index`` (there would be no task text for the language keys), an
anchor outside the episode, and an episode the dataset does not declare are all refused, before any
frame is decoded. Frame indices clamp at the episode edge exactly the way GR00T clamps them, so
anchor 0 with a negative delta repeats the first frame rather than inventing one.

numpy is imported function-locally: the core install has none of it.
"""

from __future__ import annotations

import json
import math
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeGuard

from foundationscale.vla.chunking import ChunkError, ChunkSpec
from foundationscale.vla.frames import decode_frames
from foundationscale.vla.lerobot import LeRobotDataset, read_episode_columns
from foundationscale.vla.modality import ModalitySpec, SliceSpec

if TYPE_CHECKING:
    import numpy as np

__all__ = [
    "ACTION_REPS",
    "Gr00tContract",
    "Gr00tContractError",
    "Gr00tGroup",
    "build_gr00t_observation",
    "check_contract_against",
    "chunk_spec_from_contract",
    "load_gr00t_contract",
]

# The action representations ``action_configs`` may declare; only ABSOLUTE is served so far.
ACTION_REPS: frozenset[str] = frozenset({"ABSOLUTE", "RELATIVE", "DELTA"})

# The groups GR00T's modality config names, in the order its files declare them.
_GROUPS: tuple[str, ...] = ("video", "state", "action", "language")

# The annotation ``original_key`` whose column carries the task index into ``meta/tasks.jsonl``.
_TASK_KEY = "task_index"


class Gr00tContractError(ValueError):
    """A GR00T processor config is malformed, or its contract cannot be served from the data."""


@dataclass(frozen=True)
class Gr00tGroup:
    """One group of GR00T's modality config: the frame offsets it draws and the keys it reads.

    ``delta_indices`` keep the file's order -- that order is the layout of the frames in the tensor
    the policy consumes -- as do ``keys``, the order of the blocks in its vector.
    """

    delta_indices: tuple[int, ...]
    keys: tuple[str, ...]


@dataclass(frozen=True)
class Gr00tContract:
    """One embodiment's modality contract as ``processor_config.json`` declares it.

    ``action_reps`` carries one representation per :attr:`action` key, from
    ``action_configs[i]["rep"]``; ``source`` is the config file the contract was read from, and it
    prefixes every refusal this contract causes later.
    """

    embodiment: str
    video: Gr00tGroup
    state: Gr00tGroup
    action: Gr00tGroup
    language: Gr00tGroup
    action_reps: tuple[str, ...]
    source: Path


def load_gr00t_contract(checkpoint_dir: str | os.PathLike[str], embodiment: str) -> Gr00tContract:
    """The modality contract ``embodiment`` declares in the GR00T checkpoint at ``checkpoint_dir``.

    The config is read from ``<dir>/processor_config.json`` or, GR00T's training layout,
    ``<dir>/processor/processor_config.json``. Raises :class:`Gr00tContractError` when neither
    exists, the file is not valid JSON, ``processor_kwargs.modality_configs`` is absent, the config
    has no ``embodiment`` (the available embodiments are named, sorted), a ``video``, ``state``,
    ``action`` or ``language`` group is missing, a group's ``delta_indices`` is not a non-empty list
    of ints or its ``modality_keys`` not a non-empty list of unique strings, ``action_configs`` is
    null/absent or does not carry exactly one entry per action key, or an entry's ``rep`` is outside
    :data:`ACTION_REPS`. Every message names the file, the key and the value observed: a guessed
    offset or key would hand the policy frames and dims it never trained on. Groups and their keys
    keep the file's order, because that order is the layout of the tensors the policy consumes.
    """
    base = Path(checkpoint_dir)
    candidates = (base / "processor_config.json", base / "processor" / "processor_config.json")
    source = next((path for path in candidates if path.is_file()), None)
    if source is None:
        raise Gr00tContractError(
            f"{base}: no GR00T processor config; expected {candidates[0]} or {candidates[1]} "
            "(the second is GR00T's training layout) -- the checkpoint's own modality contract is "
            "the only authority this module reads"
        )
    try:
        raw = json.loads(source.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise Gr00tContractError(f"{source}: processor config is not valid JSON ({exc})") from exc
    if not isinstance(raw, Mapping):
        raise Gr00tContractError(
            f"{source}: processor config is a {type(raw).__name__}; expected a JSON object with "
            "'processor_kwargs'"
        )
    processor_kwargs = raw.get("processor_kwargs")
    modality_configs = (
        processor_kwargs.get("modality_configs") if isinstance(processor_kwargs, Mapping) else None
    )
    if not isinstance(modality_configs, Mapping):
        raise Gr00tContractError(
            f"{source}: key 'processor_kwargs.modality_configs' is {modality_configs!r} "
            f"(processor_kwargs is {processor_kwargs!r}); expected a mapping of embodiment -> "
            "modality config"
        )
    available = sorted(str(name) for name in modality_configs)
    if not isinstance(embodiment, str) or embodiment not in modality_configs:
        raise Gr00tContractError(
            f"{source}: key 'processor_kwargs.modality_configs' has no embodiment {embodiment!r}; "
            f"available embodiments are {available}"
        )
    config = modality_configs[embodiment]
    if not isinstance(config, Mapping):
        raise Gr00tContractError(
            f"{source}: key 'processor_kwargs.modality_configs' embodiment {embodiment!r} is "
            f"{config!r} ({type(config).__name__}); expected a mapping of groups {list(_GROUPS)}"
        )
    base_key = f"processor_kwargs.modality_configs[{embodiment!r}]"
    groups: dict[str, Gr00tGroup] = {}
    for group in _GROUPS:
        if group not in config:
            raise Gr00tContractError(
                f"{source}: key {base_key!r} is missing group {group!r}; expected every one of "
                f"{list(_GROUPS)} -- a group the config does not name has no frames or keys for "
                "the policy to read"
            )
        groups[group] = _parse_group(source, f"{base_key}.{group}", config[group])

    action_keys = groups["action"].keys
    action_configs, action_key = _find_action_configs(config, base_key=base_key)
    action_reps = _action_reps(source, action_key, action_configs, action_keys)
    return Gr00tContract(
        embodiment=embodiment,
        video=groups["video"],
        state=groups["state"],
        action=groups["action"],
        language=groups["language"],
        action_reps=action_reps,
        source=source,
    )


def chunk_spec_from_contract(contract: Gr00tContract) -> ChunkSpec:
    """``contract``'s frame offsets as the :class:`ChunkSpec` every sampler draws with.

    ``ChunkSpec`` refuses an empty, duplicated or non-increasing delta tuple and a negative or
    gapped action horizon; those refusals are re-raised as :class:`Gr00tContractError` with
    ``contract.source`` prefixed, so the message names the processor config that declared the
    offsets rather than only the tuple that carries them.
    """
    try:
        return ChunkSpec(
            state_delta=contract.state.delta_indices,
            action_delta=contract.action.delta_indices,
            video_delta=contract.video.delta_indices,
        )
    except ChunkError as exc:
        raise Gr00tContractError(f"{contract.source}: {exc}") from exc


def check_contract_against(contract: Gr00tContract, spec: ModalitySpec) -> None:
    """Refuse a contract the modality spec ``spec`` cannot serve, naming the key that cannot.

    Raises :class:`Gr00tContractError` when a contract state key is not a ``spec.state`` slice name,
    an action key is not a ``spec.action`` slice name, a video key is not in ``spec.video``, an
    action rep is not ``"ABSOLUTE"`` (``RELATIVE``/``DELTA`` need the ``state_key`` reference and
    are not supported yet -- reading them as absolute would silently change the action space), or
    the annotation group has no entry whose ``original_key`` is ``task_index`` (there would be no
    task text to attach to the language keys).
    """
    state_names = {item.name for item in spec.state}
    action_names = {item.name for item in spec.action}
    for key in contract.state.keys:
        if key not in state_names:
            raise Gr00tContractError(
                f"{contract.source}: contract state key {key!r} is not a modality state slice "
                f"name; expected one of {sorted(state_names)}"
            )
    for key in contract.action.keys:
        if key not in action_names:
            raise Gr00tContractError(
                f"{contract.source}: contract action key {key!r} is not a modality action slice "
                f"name; expected one of {sorted(action_names)}"
            )
    for key in contract.video.keys:
        if key not in spec.video:
            raise Gr00tContractError(
                f"{contract.source}: contract video key {key!r} is not a modality video name; "
                f"expected one of {sorted(spec.video)}"
            )
    if len(contract.action_reps) != len(contract.action.keys):
        raise Gr00tContractError(
            f"{contract.source}: contract action_reps has {len(contract.action_reps)} entries for "
            f"{len(contract.action.keys)} action keys {list(contract.action.keys)}; expected one "
            "rep per action key"
        )
    refused = [
        (key, rep)
        for key, rep in zip(contract.action.keys, contract.action_reps, strict=True)
        if rep != "ABSOLUTE"
    ]
    if refused:
        raise Gr00tContractError(
            f"{contract.source}: action keys {[key for key, _ in refused]} have reps "
            f"{[rep for _, rep in refused]} instead of 'ABSOLUTE'; RELATIVE/DELTA actions need "
            "the state_key reference and are not supported yet"
        )
    if _TASK_KEY not in spec.annotation.values():
        raise Gr00tContractError(
            f"{contract.source}: modality annotation {dict(spec.annotation)!r} has no entry with "
            f"original_key {_TASK_KEY!r}; the task text is read from that column, so the language "
            "keys would carry no task"
        )


def build_gr00t_observation(
    ds: LeRobotDataset,
    spec: ModalitySpec,
    contract: Gr00tContract,
    episode_index: int,
    anchor: int,
) -> dict[str, Any]:
    """GR00T's observation dict at ``anchor`` of episode ``episode_index``, unbatched.

    GR00T adds the batch dimension itself. ``state.<key>`` is
    ``float32 [len(state.delta_indices), dim]``: dims ``[start, end)`` of the slice's
    ``original_key`` column at the state frames, read with
    :func:`~foundationscale.vla.lerobot.read_episode_columns`. ``video.<key>`` is
    ``uint8 [len(video.delta_indices), H, W, 3]``, the frames
    :func:`~foundationscale.vla.frames.decode_frames` yields at the video indices, stacked. Every
    language key carries the task text: ``ds.tasks`` at the ``task_index`` of the anchor frame, the
    annotation column whose ``original_key`` is ``task_index``. Each group's indices are
    ``min(max(anchor + delta, 0), length - 1)`` -- GR00T clamps at the episode edge, so a negative
    delta at the first frame repeats it instead of inventing one.

    :func:`check_contract_against` runs first, so a contract the modality spec cannot serve is
    refused before any frame is decoded. A non-int ``episode_index`` or ``anchor``, an anchor
    outside ``[0, length)``, and an episode the dataset does not declare are refused with
    :class:`Gr00tContractError` naming the value: an anchor outside the episode has no frame to
    clamp onto, and an undeclared episode has no columns to read.
    """
    import numpy as np  # noqa: PLC0415

    check_contract_against(contract, spec)
    if not _is_int(episode_index):
        raise Gr00tContractError(f"{ds.root}: episode_index is {episode_index!r}; expected an int")
    episode = next((item for item in ds.episodes if item.index == episode_index), None)
    if episode is None:
        raise Gr00tContractError(
            f"{ds.root}: episode {episode_index} is not declared; expected one of "
            f"{[item.index for item in ds.episodes]}"
        )
    length = episode.length
    if not _is_int(anchor):
        raise Gr00tContractError(
            f"{ds.root}: episode {episode_index} anchor is {anchor!r}; expected an int"
        )
    if not 0 <= anchor < length:
        raise Gr00tContractError(
            f"{ds.root}: episode {episode_index} anchor {anchor} is outside [0, {length}); GR00T "
            "clamps delta_indices at the episode edge, but the anchor itself must name a frame"
        )
    state_rows = _indices(anchor, contract.state.delta_indices, length)
    video_rows = _indices(anchor, contract.video.delta_indices, length)

    state_items = {item.name: item for item in spec.state}
    wanted: list[str] = []
    for key in contract.state.keys:
        original_key = state_items[key].original_key
        if original_key not in wanted:
            wanted.append(original_key)
    if _TASK_KEY not in wanted:
        wanted.append(_TASK_KEY)
    columns = read_episode_columns(ds, episode_index, wanted)

    observation: dict[str, Any] = {}
    for key in contract.state.keys:
        observation[f"state.{key}"] = _state_block(
            columns, state_items[key], state_rows, ds=ds, episode=episode_index
        )
    for key in contract.video.keys:
        frames = decode_frames(
            ds.video_file(episode_index, spec.video[key]), video_rows, expected_length=length
        )
        observation[f"video.{key}"] = np.asarray(np.stack(list(frames)), dtype=np.uint8)
    task = _task_text(ds, columns, anchor, episode_index)
    # Language keys are used verbatim: GR00T's observation names them exactly as the contract
    # does (measured: "annotation.human.action.task_description"), with no group prefix.
    for key in contract.language.keys:
        observation[key] = task
    return observation


# -- fail-closed helpers ---------------------------------------------------------------


def _is_int(value: Any) -> TypeGuard[int]:
    """``True`` for a real int: ``bool`` is refused, it is not an episode or a frame index."""
    return isinstance(value, int) and not isinstance(value, bool)


def _parse_group(source: Path, key: str, value: Any) -> Gr00tGroup:
    """One modality group as GR00T declares it: its ``delta_indices`` and ``modality_keys``."""
    if not isinstance(value, Mapping):
        raise Gr00tContractError(
            f"{source}: key {key!r} is {value!r} ({type(value).__name__}); expected a mapping with "
            "'delta_indices' and 'modality_keys'"
        )
    return Gr00tGroup(
        delta_indices=_delta_indices(source, f"{key}.delta_indices", value.get("delta_indices")),
        keys=_modality_keys(source, f"{key}.modality_keys", value.get("modality_keys")),
    )


def _delta_indices(source: Path, key: str, value: Any) -> tuple[int, ...]:
    """``delta_indices`` in file order; a non-empty list of ints is the only shape accepted."""
    if not isinstance(value, list) or not value:
        raise Gr00tContractError(
            f"{source}: key {key!r} is {value!r}; expected a non-empty list of int frame offsets"
        )
    for index, item in enumerate(value):
        if not _is_int(item):
            raise Gr00tContractError(
                f"{source}: key {key!r}[{index}] is {item!r} ({type(item).__name__}); expected an "
                "int frame offset"
            )
    return tuple(value)


def _modality_keys(source: Path, key: str, value: Any) -> tuple[str, ...]:
    """``modality_keys`` in file order; a non-empty list of unique strings is the only shape."""
    if not isinstance(value, list) or not value:
        raise Gr00tContractError(
            f"{source}: key {key!r} is {value!r}; expected a non-empty list of unique strings"
        )
    keys: list[str] = []
    seen: dict[str, int] = {}
    for index, item in enumerate(value):
        if not isinstance(item, str) or not item:
            raise Gr00tContractError(
                f"{source}: key {key!r}[{index}] is {item!r}; expected a non-empty string naming "
                "one modality key"
            )
        if item in seen:
            raise Gr00tContractError(
                f"{source}: key {key!r}[{index}] is {item!r}, the same as index {seen[item]}; "
                "expected each modality key exactly once (a duplicate would build one block of "
                "the vector twice)"
            )
        seen[item] = index
        keys.append(item)
    return tuple(keys)


def _find_action_configs(config: Mapping[str, Any], *, base_key: str) -> tuple[Any, str]:
    """``action_configs`` and its dotted key: GR00T keeps it inside the embodiment's action group.

    Measured on nvidia/GR00T-N1.7-LIBERO: ``modality_configs.<embodiment>.action.action_configs``.
    Only that location is read; a config that does not carry it there yields ``(None, key)`` so the
    refusal names the key instead of inventing an entry.
    """
    action = config.get("action")
    key = f"{base_key}.action.action_configs"
    if isinstance(action, Mapping):
        return action.get("action_configs"), key
    return None, key


def _action_reps(
    source: Path, key: str, value: Any, action_keys: tuple[str, ...]
) -> tuple[str, ...]:
    """One ``rep`` per action key, in the action keys' order, from ``action_configs``.

    Refuses a null/absent ``action_configs``, one whose entry count differs from the action keys
    (the entries line up with the keys, so a mismatch would read one key's representation as
    another's), an entry that is not a mapping, and a ``rep`` outside :data:`ACTION_REPS` -- each
    named with the file, the key and the value observed.
    """
    if value is None:
        raise Gr00tContractError(
            f"{source}: key {key!r} is {value!r}; expected one entry per action key "
            f"{list(action_keys)}, each a mapping whose 'rep' is one of {sorted(ACTION_REPS)}"
        )
    entries: list[Any]
    if isinstance(value, Mapping):
        if len(value) != len(action_keys):
            raise Gr00tContractError(
                f"{source}: key {key!r} has length {len(value)}; expected one entry per action key "
                f"{list(action_keys)} (length {len(action_keys)})"
            )
        entries = []
        for action_key in action_keys:
            if action_key not in value:
                raise Gr00tContractError(
                    f"{source}: key {key!r} is {value!r}; expected an entry for action key "
                    f"{action_key!r} (one per action key {list(action_keys)})"
                )
            entries.append(value[action_key])
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        if len(value) != len(action_keys):
            raise Gr00tContractError(
                f"{source}: key {key!r} has length {len(value)}; expected one entry per action key "
                f"{list(action_keys)} (length {len(action_keys)})"
            )
        entries = list(value)
    else:
        raise Gr00tContractError(
            f"{source}: key {key!r} is {value!r} ({type(value).__name__}); expected a list or "
            f"mapping of one entry per action key {list(action_keys)}"
        )
    reps: list[str] = []
    for index, (action_key, entry) in enumerate(zip(action_keys, entries, strict=True)):
        if not isinstance(entry, Mapping):
            raise Gr00tContractError(
                f"{source}: key {key!r} entry [{index}] for action key {action_key!r} is "
                f"{entry!r} ({type(entry).__name__}); expected a mapping with a 'rep'"
            )
        rep = entry.get("rep")
        if not isinstance(rep, str) or rep not in ACTION_REPS:
            raise Gr00tContractError(
                f"{source}: key {key!r} entry [{index}] for action key {action_key!r} 'rep' is "
                f"{rep!r}; expected one of {sorted(ACTION_REPS)}"
            )
        reps.append(rep)
    return tuple(reps)


def _indices(anchor: int, deltas: Sequence[int], length: int) -> list[int]:
    """``anchor + delta`` per offset, clamped to ``[0, length)`` as GR00T clamps at the edge."""
    return [min(max(anchor + delta, 0), length - 1) for delta in deltas]


def _state_block(
    columns: Mapping[str, np.ndarray],
    item: SliceSpec,
    rows: Sequence[int],
    *,
    ds: LeRobotDataset,
    episode: int,
) -> np.ndarray:
    """``float32 [len(rows), item.dim]``: dims ``[start, end)`` of ``item``'s column at ``rows``."""
    import numpy as np  # noqa: PLC0415

    column = np.asarray(columns[item.original_key])
    if column.ndim == 1:
        column = column.reshape(-1, 1)
    elif column.ndim != 2:
        raise Gr00tContractError(
            f"{ds.root}: episode {episode} column {item.original_key!r} is a {column.ndim}-D "
            "array; expected one value per frame per dim"
        )
    width = int(column.shape[1])
    if item.end > width:
        raise Gr00tContractError(
            f"{ds.root}: episode {episode} modality state slice {item.name!r} spans "
            f"[{item.start}, {item.end}) of original_key {item.original_key!r}; the column carries "
            f"{width} dim(s) per frame (observed end {item.end}, expected <= {width})"
        )
    return np.asarray(column[list(rows)][:, item.start : item.end], dtype=np.float32)


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
        raise Gr00tContractError(
            f"{ds.root}: episode {episode} column {_TASK_KEY!r} is a {column.ndim}-D array; "
            "expected one task index per frame"
        )
    raw = column[anchor]
    try:
        number = float(raw)
    except (TypeError, ValueError) as exc:
        raise Gr00tContractError(
            f"{ds.root}: episode {episode} anchor {anchor} {_TASK_KEY} is {raw!r}; expected an "
            f"integer task index ({exc})"
        ) from exc
    if not math.isfinite(number) or not number.is_integer():
        raise Gr00tContractError(
            f"{ds.root}: episode {episode} anchor {anchor} {_TASK_KEY} is {raw!r}; expected an "
            "integer task index"
        )
    index = int(number)
    if index not in ds.tasks:
        raise Gr00tContractError(
            f"{ds.root}: episode {episode} anchor {anchor} {_TASK_KEY} {index} is absent from "
            f"meta/tasks.jsonl; expected one of {sorted(ds.tasks)}"
        )
    return ds.tasks[index]
