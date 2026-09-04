"""Artifact readers: safetensors headers, JSON, hashes -- stdlib only, no torch."""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import struct
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ._base import (
    _CHUNK,
    _SAFETENSORS_DTYPE_BYTES,
)
from ._errors import (
    ArtifactError,
)

# ---------------------------------------------------------------------------
# Artifact helpers
# ---------------------------------------------------------------------------


def _sha256_and_lines(path: Path) -> tuple[str, int]:
    """Streamed sha256 and a wc -l line count (count of b'\\n'), one pass.

    wc -l counts newline characters, not "lines"; a trailing unterminated line
    is invisible to both wc and to this function — the contract states so rather
    than disagreeing with the reference tool it replaces.
    """
    h = hashlib.sha256()
    lines = 0
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(_CHUNK), b""):
            h.update(chunk)
            lines += chunk.count(b"\n")
    return h.hexdigest(), lines


def _read_safetensors_header(path: Path) -> dict[str, dict[str, Any]]:
    """Parse a safetensors header with stdlib only: 8 LE length bytes + JSON.

    Returns {tensor_name: {dtype, shape, numel}} with __metadata__ excluded.
    Any deviation raises ArtifactError -> the caller BLOCKS; a shard whose
    format we cannot price is not a shard we clear.

    Chaining contract, load-bearing for _check_frozen_manifest: an OS-level
    refusal (open/read) is re-raised with the originating OSError chained as
    __cause__; a bytes-level format defect (truncated length prefix, bad JSON,
    malformed shape, unpriced dtype) raises bare, with __cause__ None. That is
    the only reliable separator between "the environment refused the read"
    (ERROR, fail closed — the operator goes to the machine) and "the artifact
    is corrupt" (a FAIL the check exists to name — the operator goes to the
    checkpoint), and the two demand opposite responses.
    """
    try:
        with path.open("rb") as fh:
            raw = fh.read(8)
            if len(raw) != 8:
                raise ArtifactError(f"{path}: shorter than a safetensors length prefix")
            (n,) = struct.unpack("<Q", raw)
            if n > 512 * 1024 * 1024:
                raise ArtifactError(
                    f"{path}: header claims {n} bytes — implausible, refusing to buffer it"
                )
            payload = fh.read(n)
            if len(payload) != n:
                raise ArtifactError(f"{path}: truncated header ({len(payload)}/{n} bytes)")
    except OSError as exc:
        raise ArtifactError(f"{path}: unreadable: {exc}") from exc
    try:
        meta = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        # `from None`, not chained: this function's docstring chaining contract
        # reserves an OSError __cause__ for environmental refusals (frozen_manifest
        # classifies on exactly that), and the decode error's text already rides
        # inside the message, so discarding the exception object costs no evidence.
        raise ArtifactError(
            f"{path}: header is not JSON ({exc}); cannot count tensors — BLOCK, not guess"
        ) from None
    if not isinstance(meta, dict):
        raise ArtifactError(f"{path}: header JSON is not an object")
    out: dict[str, dict[str, Any]] = {}
    for name, entry in meta.items():
        if name == "__metadata__":
            continue
        if not isinstance(entry, dict):
            raise ArtifactError(f"{path}: tensor {name!r} entry is not an object")
        shape = entry.get("shape")
        dtype = entry.get("dtype")
        if not isinstance(shape, list) or not all(
            isinstance(d, int) and not isinstance(d, bool) and d >= 0 for d in shape
        ):
            raise ArtifactError(f"{path}: tensor {name!r} has a malformed shape {shape!r}")
        numel = 1
        for d in shape:
            numel *= d
        if dtype not in _SAFETENSORS_DTYPE_BYTES:
            raise ArtifactError(
                f"{path}: dtype {dtype!r} has no known byte width; extend "
                f"_SAFETENSORS_DTYPE_BYTES — arithmetic over an unpriced dtype is a false number"
            )
        out[str(name)] = {"dtype": str(dtype), "shape": shape, "numel": numel}
    return out


def _canonical_sample_sha256(path: Path) -> str:
    """Hash of the batch-0 sample as a human would decode it: first JSONL row, canonicalized."""

    try:
        with path.open("r", encoding="utf-8") as fh:
            line = fh.readline()
    except OSError as exc:
        raise ArtifactError(f"{path}: unreadable while decoding batch-0 sample: {exc}") from exc
    if not line.strip():
        raise ArtifactError(f"{path}: first line is empty — there is no batch-0 sample to read")
    try:
        obj = json.loads(line)
    except json.JSONDecodeError as exc:
        raise ArtifactError(f"{path}: first line does not decode as JSON: {exc}") from exc
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def _parse_iso(text: str) -> _dt.datetime | None:
    try:
        dt = _dt.datetime.fromisoformat(str(text).replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=_dt.timezone.utc)
    return dt


def _manifest_hash_for(frozen: Mapping[str, Any], cfg_sha: str) -> tuple[str, dict[str, Any]]:
    """One canonical payload + hash, used by BOTH the check and the fixture world.

    A single source of truth is load-bearing: launch_provenance ties checkpoint
    provenance records to *this* hash, and the self-test world must compute the
    identical value or the fixture would prove nothing about the real equality.
    """
    payload = {
        "schema": 1,
        "config_sha256": cfg_sha,
        "model": {
            "files": list(frozen["model"]["files"]),
            "tensor_count": frozen["model"]["tensor_count"],
            "total_bytes": frozen["model"]["total_bytes"],
        },
        "corpus": [
            {"path": f["path"], "sha256": f["sha256"], "lines": f["lines"]}
            for f in frozen["corpus"]["files"]
        ],
        "run_config": dict(frozen["run_config"]),
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest(), payload
