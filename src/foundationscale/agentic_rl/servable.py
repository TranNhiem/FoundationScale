"""``complete_for_serving``: turn a text-only-looking agentic-RL checkpoint
into one a fresh vLLM/SGLang process can actually load.

THE PROBLEM (measured on real hardware): the agentic RL trainer loads a
multimodal base model (``architectures=["Qwen3_5ForConditionalGeneration"]``,
a config with both ``text_config`` and ``vision_config``) as a text-only
``CausalLM`` -- only the language tower is ever instantiated, trained and
gathered. ``rl.distributed.save_checkpoint`` therefore writes, via
``save_pretrained``, a checkpoint whose ``config.json`` names a TEXT-ONLY
architecture (e.g. ``model_type="qwen3_5_text"``,
``architectures=["Qwen3_5ForCausalLM"]``) and whose safetensors hold ONLY the
language tensors (``model.language_model.*``, ``lm_head.weight``, etc.),
using exactly the base model's own tensor names. The base model's other
tensors (e.g. ``model.visual.*``) are simply absent -- never renamed, never
guessed at. A fresh ``from_pretrained``/vLLM load of that directory fails
(``"Invalid type of HuggingFace config"``): nothing on disk claims the
multimodal architecture the missing tensors belong to, and even if it did,
the tensors themselves are not there. ``reload_weights(weights_path=dir)``
into an ALREADY-RUNNING engine process started from the base model still
works, because that engine never re-reads ``config.json`` or expects the
missing tensors to exist on disk -- it only overwrites the language tensors
it already holds in memory. A FRESH serve has no such running engine to
reuse, so it needs a directory that is honestly, structurally complete.

THE FIX: :func:`complete_for_serving` makes exactly one published checkpoint
directory loadable from cold, by completing it with what the base model
declares and the trained checkpoint is missing -- never by inventing a tensor
or a name mapping. It refuses outright (:class:`ServableRefusal`) the moment
a published tensor name is not a base tensor name, or a published tensor's
shape disagrees with the base's: both would mean this function's "every
published key is a base key, just a subset" assumption is false, and GUESSING
a mapping in that case would silently serve a model wearing the wrong
weights. It extracts, into a dedicated extras file, ONLY the tensors the base
model has that the published checkpoint does not -- and that extras file is
cached by a hash of ``(base model dir, the exact missing-key list)``, so N
published checkpoints trained from the same base pay the (base-model-sized)
extraction cost once.

The one invariant everything else here serves: a fresh load of the completed
directory must see the TRAINED language tensors, never the base model's
original ones. That is why the base model's own shard files are never linked
or copied into ``publish_dir`` -- they still carry the base's untrained
language weights, and a ``model.safetensors.index.json`` that pointed any
language key back at one of them would silently serve a stale policy out of
a trained-looking checkpoint. Only a purpose-built extras file, holding
exactly the tensors the published checkpoint lacks, is ever placed there.
"""

from __future__ import annotations

import errno
import hashlib
import json
import os
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from safetensors import safe_open
from safetensors.torch import save_file

__all__ = (
    "ServableRefusal",
    "ServableReport",
    "complete_for_serving",
)

_INDEX_FILENAME = "model.safetensors.index.json"
_SINGLE_SHARD_FILENAME = "model.safetensors"
_EXTRAS_FILENAME = "model-serving-extras.safetensors"
_PROVENANCE_FILENAME = "fs_servable.json"
_BACKUP_CONFIG_FILENAME = "config.trained_text_only.json"

# Copied from the base model INTO publish_dir, overwriting any published copy:
# config.json always; the rest only "if present" in the base model dir.
_OPTIONAL_COPY_FILENAMES = (
    "preprocessor_config.json",
    "processor_config.json",
    "video_preprocessor_config.json",
    "chat_template.jinja",
)

# safetensors' own fixed dtype-string vocabulary (not a guess: this is the
# format's declared enum, unlike a tensor-NAME mapping). Used only to compute
# a declared total_size for the merged index; an unrecognised dtype string
# refuses rather than assumes a width.
_DTYPE_BYTES: dict[str, int] = {
    "F64": 8,
    "F32": 4,
    "F16": 2,
    "BF16": 2,
    "I64": 8,
    "I32": 4,
    "I16": 2,
    "I8": 1,
    "U8": 1,
    "BOOL": 1,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
}


def _described(value: object) -> str:
    return f"{type(value).__name__} {value!r}"


class ServableRefusal(ValueError):
    """A config/shape error preventing :func:`complete_for_serving` from safely
    completing a published checkpoint -- never raised for an I/O fault moving
    bytes it has already decided are safe to move (that propagates as
    whatever ``OSError`` the filesystem raised).
    """


@dataclass(frozen=True)
class ServableReport:
    """What :func:`complete_for_serving` did to ``publish_dir``.

    ``status="already_servable"`` means NOTHING was changed -- the published
    config's ``architectures`` already equalled the base's -- and
    ``extras_keys``/``extras_bytes``/``copied_files`` are all the honest
    empty/zero of "no work was needed", never a stale measurement of a PRIOR
    call. ``status="completed"`` means the directory was just made loadable;
    ``extras_keys`` is the count of base tensors the published checkpoint was
    missing, ``extras_bytes`` is the on-disk size of the single extras file
    holding them, and ``copied_files`` names exactly which base files were
    copied into ``publish_dir`` (``config.json`` always, the rest only when
    the base model declared them).
    """

    status: Literal["already_servable", "completed"]
    extras_keys: int
    extras_bytes: int
    copied_files: tuple[str, ...]


@dataclass(frozen=True)
class _TensorInfo:
    file: str
    shape: tuple[int, ...]
    dtype: str


def _read_json_object(path: Path, *, where: str) -> dict[str, object]:
    if not path.is_file():
        raise ServableRefusal(f"{where}: {path} does not exist or is not a file")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ServableRefusal(f"{where}: {path} is not valid JSON: {exc}") from exc
    if not isinstance(raw, dict):
        raise ServableRefusal(
            f"{where}: {path} top level is {_described(raw)}, expected a JSON object"
        )
    return raw


def _weight_map(directory: Path, *, where: str) -> dict[str, _TensorInfo]:
    """Every tensor name directly inside ``directory``'s safetensors checkpoint,
    with its shape and dtype -- read from ``model.safetensors.index.json``
    (sharded) if present, else a single ``model.safetensors``. Refuses if
    neither exists. Shapes/dtypes come from safetensors' own slice metadata
    (``get_slice``), never by loading a tensor's data -- cheap even for a huge
    checkpoint.
    """
    index_path = directory / _INDEX_FILENAME
    if index_path.is_file():
        index_raw = _read_json_object(index_path, where=where)
        weight_map_raw = index_raw.get("weight_map")
        if not isinstance(weight_map_raw, dict) or not weight_map_raw:
            raise ServableRefusal(
                f"{where}: {index_path} field 'weight_map' is "
                f"{_described(weight_map_raw)}: it must be a non-empty JSON object "
                f"of {{tensor_name: shard_filename}}"
            )
        filenames = sorted({str(value) for value in weight_map_raw.values()})
        named_by = str(index_path)
    else:
        single_path = directory / _SINGLE_SHARD_FILENAME
        if not single_path.is_file():
            raise ServableRefusal(
                f"{where}: neither {index_path} nor {single_path} exists -- no "
                f"safetensors checkpoint found to read tensor names/shapes from"
            )
        filenames = [_SINGLE_SHARD_FILENAME]
        named_by = str(single_path)

    info: dict[str, _TensorInfo] = {}
    for filename in filenames:
        shard_path = directory / filename
        if not shard_path.is_file():
            raise ServableRefusal(f"{where}: shard {shard_path} named by {named_by} does not exist")
        with safe_open(str(shard_path), framework="pt") as handle:
            tensor_names = handle.keys()  # a plain list[str], not a dict -- not SIM118
            for key in tensor_names:
                tensor_slice = handle.get_slice(key)
                info[key] = _TensorInfo(
                    file=filename,
                    shape=tuple(tensor_slice.get_shape()),
                    dtype=tensor_slice.get_dtype(),
                )
    return info


def _tensor_bytes(info: _TensorInfo) -> int:
    itemsize = _DTYPE_BYTES.get(info.dtype)
    if itemsize is None:
        raise ServableRefusal(
            f"complete_for_serving: tensor dtype {info.dtype!r} is not one of the "
            f"declared safetensors dtypes {sorted(_DTYPE_BYTES)} -- refusing to "
            f"guess its byte width for the merged index's total_size"
        )
    size = itemsize
    for dim in info.shape:
        size *= dim
    return size


def _extras_cache_path(cache_dir: Path, base: Path, extras_keys: list[str]) -> Path:
    digest_input = "\n".join([str(base.resolve()), *extras_keys]).encode("utf-8")
    digest = hashlib.sha256(digest_input).hexdigest()
    return cache_dir / f"extras-{digest}.safetensors"


def _write_extras_cache(
    *, base: Path, base_map: dict[str, _TensorInfo], extras_keys: list[str], cache_file: Path
) -> None:
    """Extract exactly ``extras_keys``'s tensors from ``base``'s shards into ONE
    safetensors file at ``cache_file``, atomically (write to a tmp file in the
    same directory, then ``Path.replace``) so a concurrent reader of
    ``cache_file`` never observes a partial write.
    """
    by_file: dict[str, list[str]] = {}
    for key in extras_keys:
        by_file.setdefault(base_map[key].file, []).append(key)
    tensors = {}
    for filename, keys in by_file.items():
        shard_path = base / filename
        with safe_open(str(shard_path), framework="pt") as handle:
            for key in keys:
                tensors[key] = handle.get_tensor(key)
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = cache_file.with_name(f"{cache_file.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}")
    save_file(tensors, str(tmp_path))
    tmp_path.replace(cache_file)


_FALLBACK_ERRNOS = (errno.EXDEV, errno.EPERM, errno.EACCES)


def _link_or_copy(src: Path, dst: Path) -> tuple[Literal["hardlink", "copy"], str | None]:
    """Hard-link ``src`` to ``dst``; fall back to a byte copy ONLY on a
    cross-device or permission ``OSError`` (``errno`` in ``_FALLBACK_ERRNOS``),
    returning which happened and -- for a fallback -- why. Any other
    ``OSError`` propagates: this function decides when a fallback is SAFE, it
    never papers over an unrelated filesystem fault.
    """
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError as exc:
        if exc.errno not in _FALLBACK_ERRNOS:
            raise
        shutil.copy2(src, dst)
        return "copy", f"{type(exc).__name__}: {exc}"
    return "hardlink", None


def complete_for_serving(
    publish_dir: str, base_model_dir: str, *, cache_dir: str
) -> ServableReport:
    """Make ``publish_dir`` (a just-saved agentic-RL checkpoint) loadable by a
    fresh inference engine, completing it from ``base_model_dir`` (the
    multimodal model the trainer actually loaded) when it is not already.

    See the module docstring for the problem this solves and the one
    invariant it keeps: a loader reading the result afterwards must see the
    TRAINED language tensors, never the base model's original ones.
    """
    where = "complete_for_serving"
    publish = Path(publish_dir)
    base = Path(base_model_dir)
    if not publish.is_dir():
        raise ServableRefusal(f"{where}: publish_dir {publish_dir!r} is not a directory")
    if not base.is_dir():
        raise ServableRefusal(f"{where}: base_model_dir {base_model_dir!r} is not a directory")

    published_config = _read_json_object(publish / "config.json", where=f"{where}: publish_dir")
    base_config = _read_json_object(base / "config.json", where=f"{where}: base_model_dir")

    base_architectures = base_config.get("architectures")
    if not isinstance(base_architectures, list) or not base_architectures:
        raise ServableRefusal(
            f"{where}: base_model_dir's config.json field 'architectures' is "
            f"{_described(base_architectures)}: it must be a non-empty list"
        )
    published_architectures = published_config.get("architectures")
    if published_architectures == base_architectures:
        return ServableReport(
            status="already_servable", extras_keys=0, extras_bytes=0, copied_files=()
        )

    base_map = _weight_map(base, where=f"{where}: base_model_dir")
    published_map = _weight_map(publish, where=f"{where}: publish_dir")

    unknown = sorted(set(published_map) - set(base_map))
    if unknown:
        shown = ", ".join(unknown[:10])
        more = f" (+{len(unknown) - 10} more)" if len(unknown) > 10 else ""
        raise ServableRefusal(
            f"{where}: {len(unknown)} of {len(published_map)} published tensor name(s) "
            f"are not among the {len(base_map)} base model tensor name(s) at "
            f"{base_model_dir!r}: {shown}{more} -- refusing rather than guess a mapping"
        )
    mismatched = sorted(
        name for name in published_map if published_map[name].shape != base_map[name].shape
    )
    if mismatched:
        details = "; ".join(
            f"{name}: published {published_map[name].shape} vs base {base_map[name].shape}"
            for name in mismatched[:5]
        )
        more = f"; (+{len(mismatched) - 5} more)" if len(mismatched) > 5 else ""
        raise ServableRefusal(
            f"{where}: {len(mismatched)} of {len(published_map)} published tensor(s) "
            f"have a shape differing from the base model's: {details}{more}"
        )

    extras_keys = sorted(set(base_map) - set(published_map))

    extras_bytes = 0
    link_method: Literal["hardlink", "copy"] | None = None
    link_fallback_reason: str | None = None
    if extras_keys:
        cache_root = Path(cache_dir)
        cache_root.mkdir(parents=True, exist_ok=True)
        cache_file = _extras_cache_path(cache_root, base, extras_keys)
        if not cache_file.is_file():
            _write_extras_cache(
                base=base, base_map=base_map, extras_keys=extras_keys, cache_file=cache_file
            )
        extras_bytes = cache_file.stat().st_size
        link_method, link_fallback_reason = _link_or_copy(cache_file, publish / _EXTRAS_FILENAME)

    backup_path = publish / _BACKUP_CONFIG_FILENAME
    shutil.copy2(publish / "config.json", backup_path)
    shutil.copy2(base / "config.json", publish / "config.json")
    copied_files = ["config.json"]
    for filename in _OPTIONAL_COPY_FILENAMES:
        src = base / filename
        if src.is_file():
            shutil.copy2(src, publish / filename)
            copied_files.append(filename)

    weight_map_out: dict[str, str] = {name: info.file for name, info in published_map.items()}
    total_size = sum(_tensor_bytes(info) for info in published_map.values())
    if extras_keys:
        weight_map_out.update(dict.fromkeys(extras_keys, _EXTRAS_FILENAME))
        total_size += sum(_tensor_bytes(base_map[name]) for name in extras_keys)
    index_payload = {"metadata": {"total_size": total_size}, "weight_map": weight_map_out}
    (publish / _INDEX_FILENAME).write_text(
        json.dumps(index_payload, indent=2, sort_keys=True), encoding="utf-8"
    )

    provenance_payload: dict[str, object] = {
        "base_model_dir": str(base.resolve()),
        "extras_file": _EXTRAS_FILENAME if extras_keys else None,
        "extras_key_count": len(extras_keys),
        "published_key_count": len(published_map),
        "extras_link_method": link_method,
    }
    if link_fallback_reason is not None:
        provenance_payload["extras_link_fallback_reason"] = link_fallback_reason
    (publish / _PROVENANCE_FILENAME).write_text(
        json.dumps(provenance_payload, indent=2, sort_keys=True), encoding="utf-8"
    )

    return ServableReport(
        status="completed",
        extras_keys=len(extras_keys),
        extras_bytes=extras_bytes,
        copied_files=tuple(copied_files),
    )
