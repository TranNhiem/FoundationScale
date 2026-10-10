"""GR00T ``meta/modality.json``: which dataset columns the policy sees, and how.

A modality file names, per group, the slices of a dataset column that make up the
policy's state and action vectors, the video columns it reads, and the annotation
columns that carry text. This module parses that file into a
:class:`ModalitySpec` and checks it against a
:class:`~foundationscale.vla.lerobot.LeRobotDataset`, so a mismatch is refused
before training instead of silently reshaping the policy or dropping a dim.

Nothing here fills in a missing decision. A file with an unknown top-level group,
a missing ``state`` or ``action`` group, a slice whose ``start``/``end`` are not
ints or do not span at least one dim, or two slices of the same column that
overlap is refused with :class:`ModalityError` naming the file, the group and the
slices involved -- an overlap would count a dim twice and a gap would hand the
policy a vector nobody declared. ``dtype`` inside a slice entry is GR00T's own
hint and is IGNORED: the dataset's feature metadata is the authority. ``absolute``
defaults to ``True`` and ``rotation_type`` to ``None`` because GR00T documents
those defaults; a slice's ``original_key`` defaults to ``observation.state``
(state) or ``action`` (action), again as documented.

Against a dataset the same rule holds: a column the modality file names but the
dataset does not carry, a state/action column whose feature is not 1-D ``[n]``,
slices of one column that do not cover dims ``0..n-1`` exactly (a gap or an
overrun), a video column whose dtype is not ``video``, and an absent annotation
column are all refused, because each would silently drop or misread data the
policy was declared to use.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from foundationscale.vla.lerobot import LeRobotDataset

__all__ = [
    "GROUPS",
    "ModalityError",
    "ModalitySpec",
    "SliceSpec",
    "check_against_dataset",
    "load_modality",
]

GROUPS: tuple[str, ...] = ("state", "action", "video", "annotation")
# GR00T's documented defaults for a slice entry that names no ``original_key``.
_DEFAULT_KEYS: Mapping[str, str] = {"state": "observation.state", "action": "action"}


class ModalityError(ValueError):
    """A modality file is malformed, or disagrees with the dataset it names."""


@dataclass(frozen=True)
class SliceSpec:
    """Dims ``[start, end)`` of ``original_key`` under one policy name.

    ``absolute`` and ``rotation_type`` are carried through for the policy; this
    module only validates them. ``dim`` is the width the slice contributes.
    """

    name: str
    original_key: str
    start: int
    end: int
    absolute: bool = True
    rotation_type: str | None = None

    @property
    def dim(self) -> int:
        """How many dims this slice contributes to the vector."""
        return self.end - self.start


@dataclass(frozen=True)
class ModalitySpec:
    """One parsed modality file: state/action slices plus video and annotation names.

    ``state`` and ``action`` keep the file's order, because that order is the
    layout of the vector the policy consumes.
    """

    state: tuple[SliceSpec, ...]
    action: tuple[SliceSpec, ...]
    video: Mapping[str, str]
    annotation: Mapping[str, str]

    def state_dim(self) -> int:
        """Total dims of the state vector the slices build."""
        return sum(spec.dim for spec in self.state)

    def action_dim(self) -> int:
        """Total dims of the action vector the slices build."""
        return sum(spec.dim for spec in self.action)


# -- parsing ------------------------------------------------------------------------


def _entry(file: Path, group: str, name: str, value: Any) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ModalityError(
            f"{file}: modality {group} entry {name!r} is a {type(value).__name__}, "
            "expected a mapping with 'start' and 'end'"
        )
    return value


def _int_field(file: Path, group: str, name: str, entry: Mapping[str, Any], field: str) -> int:
    value = entry.get(field)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ModalityError(
            f"{file}: modality {group} slice {name!r} {field} is {value!r}, expected an int"
        )
    return value


def _original_key(file: Path, group: str, name: str, entry: Mapping[str, Any]) -> str:
    default = _DEFAULT_KEYS[group]
    key = entry.get("original_key", default)
    if not isinstance(key, str) or not key:
        raise ModalityError(
            f"{file}: modality {group} slice {name!r} original_key is {key!r}, "
            f"expected a non-empty string (GR00T default {default!r})"
        )
    return key


def _absolute(file: Path, group: str, name: str, entry: Mapping[str, Any]) -> bool:
    value = entry.get("absolute", True)
    if not isinstance(value, bool):
        raise ModalityError(
            f"{file}: modality {group} slice {name!r} absolute is {value!r}, "
            "expected a bool (GR00T default True)"
        )
    return value


def _rotation_type(file: Path, group: str, name: str, entry: Mapping[str, Any]) -> str | None:
    value = entry.get("rotation_type")
    if value is not None and not isinstance(value, str):
        raise ModalityError(
            f"{file}: modality {group} slice {name!r} rotation_type is {value!r}, "
            "expected a string or null (GR00T default null)"
        )
    return value


def _check_overlaps(file: Path, group: str, slices: Sequence[SliceSpec]) -> None:
    """Refuse two slices of one column that share a dim, naming both."""
    by_key: dict[str, list[SliceSpec]] = {}
    for spec in slices:
        by_key.setdefault(spec.original_key, []).append(spec)
    for key, specs in by_key.items():
        previous: SliceSpec | None = None
        for spec in sorted(specs, key=lambda item: (item.start, item.end)):
            if previous is not None and spec.start < previous.end:
                raise ModalityError(
                    f"{file}: modality {group} slices {previous.name!r} "
                    f"[{previous.start}, {previous.end}) and {spec.name!r} "
                    f"[{spec.start}, {spec.end}) of original_key {key!r} overlap; "
                    "a dim covered twice would be counted twice in the vector"
                )
            if previous is None or spec.end > previous.end:
                previous = spec


def _parse_slices(file: Path, group: str, value: Any) -> tuple[SliceSpec, ...]:
    if not isinstance(value, Mapping):
        raise ModalityError(
            f"{file}: modality group {group!r} is a {type(value).__name__}, "
            "expected a mapping of slice name -> slice entry"
        )
    slices: list[SliceSpec] = []
    for name, raw_entry in value.items():
        entry = _entry(file, group, name, raw_entry)
        start = _int_field(file, group, name, entry, "start")
        end = _int_field(file, group, name, entry, "end")
        if start < 0:
            raise ModalityError(
                f"{file}: modality {group} slice {name!r} start is {start}, expected >= 0"
            )
        if end <= start:
            raise ModalityError(
                f"{file}: modality {group} slice {name!r} spans [{start}, {end}), "
                f"expected end ({end}) > start ({start})"
            )
        slices.append(
            SliceSpec(
                name=name,
                original_key=_original_key(file, group, name, entry),
                start=start,
                end=end,
                absolute=_absolute(file, group, name, entry),
                rotation_type=_rotation_type(file, group, name, entry),
            )
        )
    _check_overlaps(file, group, slices)
    return tuple(slices)


def _parse_named(file: Path, group: str, value: Any) -> dict[str, str]:
    """The ``video``/``annotation`` groups: name -> ``original_key``."""
    if not isinstance(value, Mapping):
        raise ModalityError(
            f"{file}: modality group {group!r} is a {type(value).__name__}, "
            "expected a mapping of name -> {'original_key': ...}"
        )
    named: dict[str, str] = {}
    for name, raw_entry in value.items():
        entry = _entry(file, group, name, raw_entry)
        key = entry.get("original_key")
        if not isinstance(key, str) or not key:
            raise ModalityError(
                f"{file}: modality {group} entry {name!r} original_key is {key!r}, "
                "expected a non-empty string"
            )
        named[name] = key
    return named


def load_modality(path: str | os.PathLike[str]) -> ModalitySpec:
    """Parse GR00T's ``meta/modality.json`` at ``path``.

    Raises :class:`ModalityError` for a missing or unparseable file, an unknown
    top-level group (only ``state``, ``action``, ``video`` and ``annotation``
    exist), a missing ``state`` or ``action`` group, a slice whose ``start`` or
    ``end`` is not an int or whose range is empty or negative, and two slices of
    one column that overlap -- each of those would otherwise invent a layout the
    file never declared.
    """
    file = Path(path)
    if not file.is_file():
        raise ModalityError(f"modality file {file} does not exist")
    try:
        raw = json.loads(file.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ModalityError(f"{file}: modality file is not valid JSON: {exc}") from exc
    if not isinstance(raw, Mapping):
        raise ModalityError(
            f"{file}: modality file is a {type(raw).__name__}, expected a mapping of groups"
        )
    unknown = sorted(str(group) for group in raw if group not in GROUPS)
    if unknown:
        raise ModalityError(
            f"{file}: modality file has unknown top-level group(s) {unknown}; "
            f"only {list(GROUPS)} are recognised"
        )
    for required in ("state", "action"):
        if required not in raw:
            raise ModalityError(
                f"{file}: modality file is missing the {required!r} group; "
                "state and action slices are required"
            )
    return ModalitySpec(
        state=_parse_slices(file, "state", raw["state"]),
        action=_parse_slices(file, "action", raw["action"]),
        video=_parse_named(file, "video", raw.get("video", {})),
        annotation=_parse_named(file, "annotation", raw.get("annotation", {})),
    )


# -- checking against a dataset ------------------------------------------------------


def _feature(
    ds: LeRobotDataset, features: Mapping[str, Mapping[str, Any]], group: str, name: str, key: str
) -> Mapping[str, Any]:
    feature = features.get(key)
    if feature is None:
        raise ModalityError(
            f"{ds.root}: modality {group} {name!r} original_key {key!r} is absent from the "
            f"dataset features (have {sorted(features)})"
        )
    return feature


def _feature_dim(ds: LeRobotDataset, group: str, key: str, feature: Mapping[str, Any]) -> int:
    """The ``n`` of a 1-D ``[n]`` feature; anything else is refused."""
    shape = feature.get("shape")
    if (
        isinstance(shape, Sequence)
        and not isinstance(shape, (str, bytes))
        and len(shape) == 1
        and isinstance(shape[0], int)
        and not isinstance(shape[0], bool)
        and shape[0] > 0
    ):
        return int(shape[0])
    raise ModalityError(
        f"{ds.root}: modality {group} original_key {key!r} has shape {shape!r}, "
        "expected a 1-D feature shape [n] with n >= 1"
    )


def _check_coverage(
    ds: LeRobotDataset, group: str, key: str, dim: int, slices: Sequence[SliceSpec]
) -> None:
    """Refuse slices that do not tile dims ``0..dim-1`` of ``key`` exactly."""
    counts = [0] * dim
    overrun: list[int] = []
    for spec in slices:
        for index in range(spec.start, spec.end):
            if 0 <= index < dim:
                counts[index] += 1
            else:
                overrun.append(index)
    uncovered = [index for index, count in enumerate(counts) if count == 0]
    doubled = [index for index, count in enumerate(counts) if count > 1]
    over = sorted(set(overrun))
    if not uncovered and not doubled and not over:
        return
    parts = []
    if uncovered:
        parts.append(f"uncovered dims {uncovered}")
    if over:
        parts.append(f"overrun dims {over}")
    if doubled:
        parts.append(f"dims covered more than once {doubled}")
    raise ModalityError(
        f"{ds.root}: modality {group} slices for original_key {key!r} do not cover dims "
        f"0..{dim - 1} exactly: {'; '.join(parts)}; "
        "a dim the policy never sees is a silent drop"
    )


def check_against_dataset(spec: ModalitySpec, ds: LeRobotDataset) -> None:
    """Refuse a modality spec the dataset ``ds`` cannot satisfy.

    Raises :class:`ModalityError` when an ``original_key`` is absent from
    ``ds.features``, a state/action column is not 1-D ``[n]``, the slices of one
    column do not cover dims ``0..n-1`` exactly (a gap or an overrun is named), a
    video column's dtype is not ``video``, or an annotation column is absent --
    every one of those would silently drop or misread data the policy was
    declared to use.
    """
    features = ds.features
    for group, slices in (("state", spec.state), ("action", spec.action)):
        by_key: dict[str, list[SliceSpec]] = {}
        for item in slices:
            feature = _feature(ds, features, group, item.name, item.original_key)
            _feature_dim(ds, group, item.original_key, feature)
            by_key.setdefault(item.original_key, []).append(item)
        for key, key_slices in by_key.items():
            dim = _feature_dim(ds, group, key, _feature(ds, features, group, key, key))
            _check_coverage(ds, group, key, dim, key_slices)
    for name, key in spec.video.items():
        feature = _feature(ds, features, "video", name, key)
        dtype = feature.get("dtype")
        if dtype != "video":
            raise ModalityError(
                f"{ds.root}: modality video {name!r} original_key {key!r} has dtype {dtype!r}, "
                "expected 'video'"
            )
    for name, key in spec.annotation.items():
        _feature(ds, features, "annotation", name, key)
