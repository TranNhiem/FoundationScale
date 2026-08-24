"""Torch-free checkpoint metadata: what is on disk, stated before anyone reads it.

Why this module exists
----------------------
:class:`~foundationscale.gates.checkpoint_gates.CheckpointGateContext.from_path`
needs two things from this package: a metadata summary of a checkpoint that costs a
header parse instead of a tensor read, and the run manifest sitting beside it. This
module is that summary. It is the cheap pre-flight that runs *before* anyone commits
to an 11.5 GB ``.float()`` comparison: shapes, dtypes, and — the field that justifies
the module — :attr:`StoredTensorMeta.storage_id`.

``storage_id`` is the identity of the **bytes on disk**, not of the name. Incident #1
passed tensor counts, shapes and dtypes for two full training runs because all of
those were right: 128 expert FQNs pointed at 16 stored blobs, aliased 8 ways. The only
metadata-level observable was that many names resolved to the same storage. So:

* DCP derives the id from the storage record ``(relative_path, offset, length)`` —
  that triple *is* the byte identity (it is exactly what
  :class:`~foundationscale.checkpoint.dcp.DcpReader` uses to slice the bytes back out).
* safetensors derives it from ``(shard filename, data_offsets start, length)`` from
  the header — the same idea, one span per tensor by format design.
* Anything unknowable is ``None``. ``None`` means "unknown", and the gates already
  treat it that way (``ExpertDistinctnessGate`` falls back to the unique FQN, i.e.
  aliasing becomes *undetectable*, never "proven distinct"). Fabricating a per-FQN id
  would make every checkpoint look perfectly distinct and silently disarm the one gate
  built for this incident. That is why there is no fallback id anywhere in this file.

Refusal behaviour mirrors :func:`~foundationscale.checkpoint.dcp.open_weights`
deliberately: a metadata read that returns zero tensors and calls it success is the
``all([])`` bug relocated. Non-checkpoints, ambiguous layouts and zero-key sources
raise :class:`CheckpointFormatError`, with the same messages where the situation is
the same.

Metadata **only**: nothing here materialises tensor data. The DCP path pays the
measured ~180 MB RSS ``.metadata`` parse; the safetensors path parses header JSON and
needs neither torch nor the safetensors package. Both lazy rules of the subpackage are
honoured: this module imports no ML stack at module level, and the provenance import
inside :func:`load_manifest` is function-local so a checkpoint read never pays for
manifest machinery it may not need.
"""

from __future__ import annotations

import json
import os
import struct
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .dcp import (
    DCP_METADATA_FILENAME,
    SAFETENSORS_INDEX_FILENAME,
    CheckpointFormatError,
    DcpReader,
)

if TYPE_CHECKING:
    # Function-local at runtime (see module docstring): typing-only here.
    from ..provenance.manifest import RunManifest

__all__ = [
    "CheckpointMetadata",
    "StoredTensorMeta",
    "load_manifest",
    "read_metadata",
]


# ---------------------------------------------------------------------------
# Value types (the frozen contract consumed by checkpoint_gates.from_path)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StoredTensorMeta:
    """Metadata for one named entry of a checkpoint.

    ``dtype`` is the bare torch dtype name ("bfloat16", not "torch.bfloat16") so the
    gates' byte table can look it up without importing torch. ``is_extra_state``
    marks non-tensor entries (DCP ``BytesStorageMetadata`` blobs): they are *surfaced
    here* — the 8,042-of-8,970 entry lesson is that silently dropping metadata keys
    is how "the checkpoint contains X" claims drift from the checkpoint — and the
    gates filter them via this flag plus the FQN, never counting them as tensors.
    """

    shape: tuple[int, ...]
    dtype: str
    storage_id: str | None
    """Identity of the stored bytes: same span => same id, different span => different
    id. ``None`` means unknowable, which gates must read as "unknown", never
    "distinct". See the module docstring; there is deliberately no fabricated
    fallback."""

    is_extra_state: bool


@dataclass(frozen=True)
class CheckpointMetadata:
    """A complete metadata summary of one checkpoint source.

    ``tensors`` maps *every* named entry — real tensors and flagged byte blobs — so
    that counting rules downstream can make the real/blob distinction themselves with
    full information, instead of trusting a pre-filtered denominator.
    """

    tensors: Mapping[str, StoredTensorMeta]
    format: str  # "dcp" | "safetensors"
    origin: str


# ---------------------------------------------------------------------------
# Format sniffing — mirrors open_weights, but returns a format name instead of
# constructing a reader, because the readers' dtype paths import torch and a
# metadata pre-flight must not.
# ---------------------------------------------------------------------------


def _sniff(path: str) -> str:
    """Classify ``path`` as "dcp" or "safetensors", refusing everything else.

    The refusal branches duplicate :func:`open_weights` message-for-message on
    purpose: whether a caller opens weights or pre-flights metadata, "this is not a
    checkpoint" should read identically, because the ambiguity/.bin/unrecognized
    diagnostics were written so an operator never re-runs a sniffer by hand.
    """
    p = Path(path)
    if not p.exists():
        raise CheckpointFormatError("path does not exist", path=path)

    if p.is_file():
        if path.endswith(".safetensors"):
            return "safetensors"
        if path.endswith(".bin"):
            raise CheckpointFormatError(
                "plain torch pickle (.bin) cannot be partially read or "
                "verified chunk-wise; pass a DCP directory or a safetensors export",
                path=path,
            )
        raise CheckpointFormatError(
            f"unrecognized checkpoint file (name: {p.name!r})",
            path=path,
        )

    names = {n.name for n in p.iterdir()}
    has_dcp = DCP_METADATA_FILENAME in names
    st_shards = sorted(n for n in names if n.endswith(".safetensors"))
    has_st = bool(st_shards) or SAFETENSORS_INDEX_FILENAME in names

    if has_dcp and has_st:
        raise CheckpointFormatError(
            f"ambiguous: both a DCP {DCP_METADATA_FILENAME!r} and safetensors "
            f"shards {st_shards[:3]} are present; open the intended one "
            f"explicitly — a metadata read that guesses would pre-flight the "
            f"wrong artifact with total confidence",
            path=path,
        )
    if has_dcp:
        return "dcp"
    if has_st:
        return "safetensors"

    bins = sorted(n for n in names if n.endswith(".bin"))
    if bins:
        raise CheckpointFormatError(
            f"directory holds only plain .bin pickles {bins[:4]}; these cannot "
            f"be partially read or verified chunk-wise — provide the DCP or "
            f"safetensors form of this artifact",
            path=path,
        )
    found = ", ".join(sorted(names)[:10]) or "<empty directory>"
    raise CheckpointFormatError(
        f"no recognizable checkpoint layout; found: {found}",
        path=path,
    )


def read_metadata(path: str | os.PathLike[str]) -> CheckpointMetadata:
    """Summarize a checkpoint's metadata without materializing any tensor data.

    Works for both formats :func:`open_weights` dispatches on, through the same
    layout rules. A path that is not a checkpoint — or a checkpoint-shaped source
    declaring zero keys — raises :class:`CheckpointFormatError` rather than
    returning an empty summary, because an empty key set is ``all([])`` raw
    material: every downstream "does the checkpoint contain X" loop over it
    vacuously passes (the ``open_weights`` zero-key refusal exists for exactly
    this reason; this function matches it).
    """
    p = os.fspath(path)
    if _sniff(p) == "dcp":
        return _read_dcp_metadata(p)
    return _read_safetensors_metadata(p)


# ---------------------------------------------------------------------------
# DCP
# ---------------------------------------------------------------------------


def _span_id(records: list[tuple[str, int, int]]) -> str:
    """Format (relative_path, offset, length) spans as one storage id.

    Same spans in same order => same string; any difference in file, offset or
    length => different string. That is the entire contract
    :attr:`StoredTensorMeta.storage_id` needs, and nothing more is claimed: ids
    are comparable *within one CheckpointMetadata*, not across checkpoints.
    """
    return ";".join(f"{rel}:{off}:{length}" for rel, off, length in records)


def _dcp_tensor_storage_id(reader: DcpReader, fqn: str) -> str | None:
    """Byte identity of one DCP tensor: the storage records of all its chunks.

    A multi-chunk tensor's identity is the *set* of its chunk spans, canonicalized
    by chunk offsets. If metadata declares a chunk with no storage record (the
    orphan-chunk case — the same disagreement read_box refuses on read), byte
    identity is genuinely unknowable and the answer is ``None``. Inventing an id
    here would convert "we cannot tell whether these experts alias" into "they
    are provably distinct", which is the audit's incident worn as a metadata field.
    """
    meta = reader._tensor_md[fqn]  # TensorStorageMetadata, typed Any in dcp.py
    per_key = reader._storage.get(fqn, {})
    records: list[tuple[str, int, int]] = []
    for chunk in sorted(meta.chunks, key=lambda c: tuple(int(o) for o in c.offsets)):
        sinfo = per_key.get(tuple(int(o) for o in chunk.offsets))
        if sinfo is None:
            return None
        records.append((str(sinfo.relative_path), int(sinfo.offset), int(sinfo.length)))
    if not records:
        # Zero-chunk tensor: no bytes on disk, hence no byte identity.
        return None
    return _span_id(records)


def _read_dcp_metadata(dirpath: str) -> CheckpointMetadata:
    """Build the summary from what DcpReader already parsed — single-sourced parse.

    DcpReader exposes keys/shapes/dtypes publicly but keeps the storage-record
    index private; this module is its package sibling and the *only* place that
    reaches in (``_tensor_md``, ``_storage``, ``_metadata``). If DcpReader's
    internals ever change, this is the single seam to update — the alternative,
    re-parsing ``.metadata`` here, would fork the byte-blob/vacuity filtering
    rules that took two incidents to get right.
    """
    reader = DcpReader(dirpath)

    # Byte-blob storage records are indexed by MetadataIndex(offset=None) and were
    # deliberately kept out of the reader's tensor offset index. They still live
    # in the raw metadata, so their byte identity IS knowable — build it in one
    # pass rather than rescanning storage_data once per blob (8k blobs x 8k
    # records is a real cost on the audited 26B checkpoint).
    blob_records: dict[str, list[tuple[str, int, int]]] = defaultdict(list)
    for index, sinfo in reader._metadata.storage_data.items():
        fqn = getattr(index, "fqn", None)
        if fqn is None or getattr(index, "offset", None) is not None:
            continue
        blob_records[str(fqn)].append(
            (str(sinfo.relative_path), int(sinfo.offset), int(sinfo.length))
        )

    tensors: dict[str, StoredTensorMeta] = {}
    for fqn in reader.tensor_keys():
        meta = reader._tensor_md[fqn]
        tensors[fqn] = StoredTensorMeta(
            shape=tuple(int(d) for d in meta.size),
            dtype=str(meta.properties.dtype).removeprefix("torch."),
            storage_id=_dcp_tensor_storage_id(reader, fqn),
            is_extra_state="_extra_state" in fqn,
        )
    for fqn in reader.nontensor_keys():
        records = sorted(blob_records.get(fqn, []))
        tensors[fqn] = StoredTensorMeta(
            shape=(),
            # Not a torch dtype, deliberately: a BytesStorageMetadata payload is
            # raw bytes with no element type. Gates never byte-count blobs (the
            # is_extra_state flag excludes them upstream of nbytes), and the
            # string keeps the "what is this entry" answer honest.
            dtype="bytes",
            storage_id=_span_id(records) if records else None,
            is_extra_state=True,
        )
    return CheckpointMetadata(tensors=tensors, format="dcp", origin=dirpath)


# ---------------------------------------------------------------------------
# safetensors. The header (8-byte LE length + JSON) IS the metadata: shapes,
# dtype names and the data_offsets that give each tensor's byte span. Reading it
# needs no safetensors package and no torch — unlike SafetensorsReader.dtype(),
# which validates against torch attributes. The dtype-name table below mirrors
# dcp._safetensors_dtype's mapping; keep them in sync, and prefer editing both.
# ---------------------------------------------------------------------------

_ST_DTYPE_NAMES: dict[str, str] = {
    "F64": "float64",
    "F32": "float32",
    "F16": "float16",
    "BF16": "bfloat16",
    "I64": "int64",
    "U64": "uint64",
    "I32": "int32",
    "U32": "uint32",
    "I16": "int16",
    "U16": "uint16",
    "I8": "int8",
    "U8": "uint8",
    "BOOL": "bool",
    "F8_E4M3": "float8_e4m3fn",
    "F8_E5M2": "float8_e5m2",
}

_StHeader = dict[str, tuple[str, tuple[int, ...], tuple[int, int]]]
"""key -> (dtype name, shape, (data_offset start, end)); "``__metadata__`` excluded."""


def _read_st_header(shard_path: str) -> _StHeader:
    """Parse one shard's header, failing closed on anything malformed or truncated."""
    try:
        file_size = Path(shard_path).stat().st_size
    except OSError as exc:
        raise CheckpointFormatError(
            f"cannot stat shard ({type(exc).__name__}: {exc})", path=shard_path
        ) from exc
    try:
        with Path(shard_path).open("rb") as fh:
            raw = fh.read(8)
            if len(raw) != 8:
                raise CheckpointFormatError(
                    "shard too short for the 8-byte safetensors header prefix",
                    path=shard_path,
                )
            (header_len,) = struct.unpack("<Q", raw)
            # Bound the read against the actual file size so a junk prefix
            # cannot request a 2^63-byte allocation.
            if header_len == 0 or 8 + header_len > file_size:
                raise CheckpointFormatError(
                    f"declared header length {header_len} does not fit shard size "
                    f"{file_size}; the shard is truncated or not safetensors",
                    path=shard_path,
                )
            blob = fh.read(header_len)
    except OSError as exc:
        raise CheckpointFormatError(
            f"cannot read shard header ({type(exc).__name__}: {exc})", path=shard_path
        ) from exc
    if len(blob) != header_len:
        raise CheckpointFormatError(
            f"short read: got {len(blob)} of {header_len} header bytes; shard is truncated",
            path=shard_path,
        )
    try:
        header: Any = json.loads(blob.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise CheckpointFormatError(
            f"shard header is not valid JSON ({type(exc).__name__}: {exc})",
            path=shard_path,
        ) from exc
    if not isinstance(header, dict):
        raise CheckpointFormatError("shard header is not a JSON object", path=shard_path)

    out: _StHeader = {}
    for key, entry in header.items():
        if key == "__metadata__":
            continue  # format-level free-form metadata, not a tensor
        if not isinstance(entry, dict):
            raise CheckpointFormatError(
                f"header entry for {key!r} is not an object", path=shard_path, key=key
            )
        dtype_name = _ST_DTYPE_NAMES.get(str(entry.get("dtype")))
        if dtype_name is None:
            raise CheckpointFormatError(
                f"unsupported safetensors dtype {entry.get('dtype')!r}",
                path=shard_path,
                key=key,
            )
        shape_raw = entry.get("shape")
        offsets_raw = entry.get("data_offsets")
        if (
            not isinstance(shape_raw, list)
            or not isinstance(offsets_raw, list)
            or len(offsets_raw) != 2
        ):
            raise CheckpointFormatError(
                f"header entry for {key!r} lacks a valid shape/data_offsets pair",
                path=shard_path,
                key=key,
            )
        try:
            start, end = int(offsets_raw[0]), int(offsets_raw[1])
            shape = tuple(int(d) for d in shape_raw)
        except (TypeError, ValueError) as exc:
            raise CheckpointFormatError(
                f"non-integer shape or data_offsets for {key!r}",
                path=shard_path,
                key=key,
            ) from exc
        if end < start:
            raise CheckpointFormatError(
                f"data_offsets for {key!r} run backwards ({start} > {end})",
                path=shard_path,
                key=key,
            )
        out[str(key)] = (dtype_name, shape, (start, end))
    return out


def _read_safetensors_metadata(path: str) -> CheckpointMetadata:
    """Summarize an HF-style safetensors export from header JSON alone.

    Mirrors :class:`SafetensorsReader`'s structure exactly — index file if
    present, otherwise shard enumeration with duplicate-key detection — so a
    metadata pre-flight and a real weight read can never disagree about which
    keys exist while claiming to describe the same artifact.
    """
    p = Path(path)
    cache: dict[str, _StHeader] = {}

    def header_of(shard: Path) -> _StHeader:
        sp = os.fspath(shard)
        if sp not in cache:
            cache[sp] = _read_st_header(sp)
        return cache[sp]

    if p.is_file():
        base = p.parent
        weight_map = {k: p.name for k in header_of(p)}
    else:
        base = p
        weight_map = _st_weight_map(base, header_of)

    if not weight_map:
        # Same refusal as SafetensorsReader's: zero keys is a vacuous-truth
        # machine for every all() downstream, so it is a format error, not an
        # empty summary.
        raise CheckpointFormatError(
            "export declares 0 tensors; an empty key set turns every "
            "downstream all(...) into the vacuous-truth bug, so this "
            "source is refused",
            path=os.fspath(p if p.is_dir() else base),
        )

    tensors: dict[str, StoredTensorMeta] = {}
    for key in sorted(weight_map):
        shard = weight_map[key]
        shard_path = base / shard
        if not shard_path.exists():
            raise CheckpointFormatError(
                f"weight_map points at missing shard {shard!r}",
                path=os.fspath(base),
                key=key,
            )
        entry = header_of(shard_path).get(key)
        if entry is None:
            raise CheckpointFormatError(
                f"weight_map names {key!r} but shard {shard!r} does not contain it",
                path=os.fspath(base),
                key=key,
            )
        dtype_name, shape, (start, end) = entry
        # safetensors stores each tensor contiguously in one shard, so
        # (file, offset, length) is a complete byte identity — the same shape
        # of triple DCP uses, and for the same reason.
        tensors[key] = StoredTensorMeta(
            shape=shape,
            dtype=dtype_name,
            storage_id=f"{shard}:{start}:{end - start}",
            is_extra_state="_extra_state" in key,
        )
    return CheckpointMetadata(tensors=tensors, format="safetensors", origin=os.fspath(p))


def _st_weight_map(base: Path, header_of: Any) -> dict[str, str]:
    """key -> shard filename, from the index when present, else from the shards."""
    index_path = base / SAFETENSORS_INDEX_FILENAME
    if index_path.exists():
        try:
            with index_path.open(encoding="utf-8") as handle:
                raw_map: Any = json.load(handle)["weight_map"]
        except (OSError, json.JSONDecodeError, KeyError) as exc:
            raise CheckpointFormatError(
                f"malformed {SAFETENSORS_INDEX_FILENAME} ({type(exc).__name__}: {exc})",
                path=os.fspath(base),
            ) from exc
        if not isinstance(raw_map, dict):
            raise CheckpointFormatError(
                f"{SAFETENSORS_INDEX_FILENAME}: 'weight_map' is not an object",
                path=os.fspath(base),
            )
        return {str(k): str(v) for k, v in raw_map.items()}

    shards = sorted(n.name for n in base.iterdir() if n.name.endswith(".safetensors"))
    if not shards:
        raise CheckpointFormatError(
            "directory contains neither an index nor any .safetensors shards",
            path=os.fspath(base),
        )
    weight_map: dict[str, str] = {}
    for shard in shards:
        for key in header_of(base / shard):
            if key in weight_map:
                # A tensor in two shards means two byte spans claim one name;
                # storage identity would be ambiguous, so refuse like the reader.
                raise CheckpointFormatError(
                    "tensor key appears in two shards; the export is corrupt",
                    path=os.fspath(base),
                    key=key,
                )
            weight_map[key] = shard
    return weight_map


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------

_MANIFEST_BASENAMES = ("run_manifest.json", "manifest.json", "provenance.json")
_MANIFEST_REQUIRED_KEYS = ("run_id", "code", "environment", "topology")

_MANIFEST_RESERVED_BASENAME = _MANIFEST_BASENAMES[0]
"""The basename this framework alone writes beside a checkpoint.

Strictness in :func:`load_manifest` scales with how certainly a file is *ours*.
``manifest.json`` and ``provenance.json`` are shared names — export tooling and
monitoring write files under them too — so an unparseable or unshaped file
bearing one of those names is skipped as foreign. But nothing except a
FoundationScale launcher writes ``run_manifest.json`` into a checkpoint tree, so
for that name "exists but unreadable/unshaped" can only mean the manifest write
failed: corrupt provenance, which raises, never the benign "no manifest" that
``|| true`` made of it.
"""


def load_manifest(path: str | os.PathLike[str]) -> RunManifest | None:
    """Return the run manifest sitting beside the checkpoint, or ``None``.

    Searches, in order: fixed manifest filenames (``run_manifest.json`` /
    ``manifest.json`` / ``provenance.json``) in the checkpoint directory, then
    attempt-keyed records (``attempt-NNNN.json``, newest first) there, then the
    same in the parent directory — checkpoints commonly live at
    ``<run_dir>/checkpoints/step_0001``, two levels shy of where the launcher
    writes. Resolving a whole :class:`ManifestStore` by run-id is deliberately
    out of scope: that is the launcher's job, and guessing a store root from a
    checkpoint path would invent provenance rather than find it.

    ``None`` means *absent* — a checkpoint with no manifest is a normal case.
    A file that is shaped like a manifest but fails validation raises
    :class:`CheckpointFormatError` instead of quietly counting as absent: the
    predecessor system's ``|| true`` turned "provenance failed" into
    "provenance missing", and 77 result dirs accumulated zero bundles without
    anyone being paged. Corrupt-unreadable and corrupt-unloadable are not the
    same state as never-written.

    Under the reserved name ``run_manifest.json`` (see
    :data:`_MANIFEST_RESERVED_BASENAME`) the strictness starts earlier: a file
    under that exact name that cannot be read as JSON at all, or that parses
    but lacks the required manifest keys, is corrupt or partial provenance and
    raises. Under the shared names and the ``attempt-*.json`` glob the search
    stays lenient — those names legitimately belong to other tools, and
    blocking a healthy checkpoint on some export tool's ``manifest.json``
    would be a false failure minted to fix a false pass.
    """
    from ..provenance.manifest import ManifestError
    from ..provenance.manifest import load as _load_run_manifest

    p = Path(os.fspath(path))
    base = p if p.is_dir() else p.parent

    candidates: list[Path] = []
    seen: set[Path] = set()

    def _add(candidate: Path) -> None:
        if candidate in seen or not candidate.is_file():
            return
        seen.add(candidate)
        candidates.append(candidate)

    for directory in (base, base.parent):
        for name in _MANIFEST_BASENAMES:
            _add(directory / name)
    # Newest attempt first: names are zero-padded, so reverse lexical order is
    # reverse attempt order, and the manifest that matters for a checkpoint is
    # the one from the launch that wrote it.
    for directory in (base, base.parent):
        for candidate in sorted(directory.glob("attempt-*.json"), reverse=True):
            _add(candidate)

    for candidate in candidates:
        # Ownership decides what `continue` is allowed to mean. For a shared
        # name, skipping an unreadable file is correct: it probably is not
        # ours. For the reserved name, skipping relabels "provenance write
        # failed" as "provenance never existed" — the exact || true collapse,
        # and the gates then treat corrupt provenance as a legacy checkpoint.
        reserved = candidate.name == _MANIFEST_RESERVED_BASENAME
        try:
            data: Any = json.loads(candidate.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
            if reserved:
                raise CheckpointFormatError(
                    f"{candidate.name!r} is the reserved run-manifest name but "
                    f"cannot be read as JSON ({type(exc).__name__}: {exc}); a "
                    f"corrupt manifest is not 'no manifest', and treating it "
                    f"as one is the || true failure",
                    path=os.fspath(path),
                ) from exc
            continue  # a neighbouring file that simply is not a manifest
        if not isinstance(data, dict) or not all(k in data for k in _MANIFEST_REQUIRED_KEYS):
            if reserved:
                # Same ownership argument one layer up: the bytes parsed, but a
                # run_manifest.json that lost run_id/code/environment/topology
                # (a torn write, an ancient writer) is a *partial* manifest,
                # and partial provenance must surface, not pass as absence.
                found: object = (
                    sorted(str(key) for key in data)[:8]
                    if isinstance(data, dict)
                    else f"<{type(data).__name__}>"
                )
                raise CheckpointFormatError(
                    f"{candidate.name!r} is the reserved run-manifest name but "
                    f"lacks the required manifest keys "
                    f"{list(_MANIFEST_REQUIRED_KEYS)!r} (content keys: {found!r}); "
                    f"a partial manifest is not 'no manifest'",
                    path=os.fspath(path),
                )
            continue  # JSON, but not manifest-shaped (a config, an export index...)
        # Shaped like a run manifest: from here strictness is the feature, and
        # failures raise rather than degrade to "no manifest".
        try:
            return _load_run_manifest(candidate)
        except ManifestError as exc:
            raise CheckpointFormatError(
                f"manifest-shaped file {candidate.name!r} fails validation "
                f"({type(exc).__name__}: {exc}); a corrupt manifest is not "
                f"'no manifest', and treating it as one is the || true failure",
                path=os.fspath(path),
            ) from exc
    return None
