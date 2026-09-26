
"""Typed artifacts, references, and atomic on-disk JSON persistence."""
from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from enum import Enum
from importlib import resources
from pathlib import Path
from typing import Any

from foundationskills.core.provenance import Provenance, canonical_json, sha256_bytes, sha256_file, sha256_json
from foundationskills.core.schema import SchemaError, validate


class ArtifactType(str, Enum):
    GOAL_SPEC = "goal_spec"
    RAW_DATA_REF = "raw_data_ref"
    DATA_PIPELINE_SPEC = "data_pipeline_spec"
    DATASET = "dataset"
    READINESS_REPORT = "readiness_report"
    TRAINING_PLAN = "training_plan"
    FS_LAUNCH_SPEC = "fs_launch_spec"
    CHECKPOINT = "checkpoint"
    EVAL_REPORT = "eval_report"

    def __str__(self) -> str:
        return self.value


_ARTIFACT_SCHEMAS: dict[str, str] = {member.value: member.value for member in ArtifactType}
_schema_cache: dict[str, dict[str, Any]] = {}


def register_artifact_type(name: str, schema_name: str) -> None:
    """Register/override mapping from an artifact type name to a payload schema name."""
    if not name or not name.strip():
        raise ValueError("artifact type name must be non-empty")
    if not schema_name or not schema_name.strip():
        raise ValueError("schema_name must be non-empty")
    _ARTIFACT_SCHEMAS[str(name)] = str(schema_name)
    _schema_cache.pop(str(schema_name), None)


def _load_artifact_schema(schema_name: str) -> dict[str, Any]:
    if schema_name in _schema_cache:
        return _schema_cache[schema_name]
    rel = resources.files("foundationskills").joinpath("schemas", "artifacts", f"{schema_name}.json")
    try:
        with rel.open("r", encoding="utf-8") as handle:
            loaded = json.load(handle)
    except FileNotFoundError as exc:
        raise SchemaError(f"artifact schema not found: {schema_name}") from exc
    if not isinstance(loaded, dict):
        raise SchemaError(f"artifact schema {schema_name} is not a JSON object")
    _schema_cache[schema_name] = loaded
    return loaded


@dataclass(frozen=True)
class Artifact:
    """A typed payload plus provenance. Immutable and hash-addressable by content."""

    type: str
    id: str
    payload: dict[str, Any]
    provenance: Provenance
    schema_version: int = 1

    def content_hash(self) -> str:
        return sha256_json(
            {
                "type": self.type,
                "id": self.id,
                "payload": self.payload,
                "schema_version": self.schema_version,
            }
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "id": self.id,
            "payload": self.payload,
            "provenance": self.provenance.to_dict(),
            "schema_version": self.schema_version,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Artifact":
        return cls(
            type=str(data["type"]),
            id=str(data["id"]),
            payload=dict(data["payload"]),
            provenance=Provenance.from_dict(dict(data["provenance"])),
            schema_version=int(data.get("schema_version", 1)),
        )

    def validate(self) -> list[str]:
        """Validate payload against the registered artifact payload schema."""
        schema_name = _ARTIFACT_SCHEMAS.get(self.type)
        if schema_name is None:
            return [f"{self.type}: no schema registered (unmeasured)"]
        schema = _load_artifact_schema(schema_name)
        return validate(self.payload, schema)


@dataclass(frozen=True)
class ArtifactRef:
    type: str
    id: str
    path: str
    sha256: str

    def to_dict(self) -> dict[str, str]:
        return {"type": self.type, "id": self.id, "path": self.path, "sha256": self.sha256}


def _atomic_write_json(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(canonical_json(obj))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
    except BaseException:
        try:
            tmp_path.unlink(missing_ok=True)
        finally:
            raise


_SAFE_PART = __import__("re").compile(r"^[A-Za-z0-9_-][A-Za-z0-9._-]*$")


def write_artifact(artifact: Artifact, directory: str | Path) -> ArtifactRef:
    """Validate and atomically write an artifact as <type>.<id>.json."""
    for label, value in (("type", artifact.type), ("id", artifact.id)):
        if not _SAFE_PART.match(str(value)):  # both name the file: no separators, no traversal
            raise ValueError(f"artifact {label} {value!r} must match [A-Za-z0-9._-]+ and not start with '.'")
    errors = artifact.validate()
    if errors:
        raise SchemaError(f"artifact {artifact.type}/{artifact.id} invalid: " + "; ".join(errors))
    target = Path(directory) / f"{artifact.type}.{artifact.id}.json"
    _atomic_write_json(target, artifact.to_dict())
    return ArtifactRef(type=artifact.type, id=artifact.id, path=str(target), sha256=sha256_file(target))


def read_artifact(path: str | Path, expect_type: str | None = None) -> Artifact:
    """Read an artifact, optionally asserting its type and validating its payload."""
    source = Path(path)
    with source.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, dict):
        raise SchemaError(f"artifact file {source} is not a JSON object")
    artifact = Artifact.from_dict(raw)
    if expect_type is not None and artifact.type != str(expect_type):
        raise SchemaError(f"artifact type mismatch: expected {expect_type!r}, got {artifact.type!r}")
    errors = artifact.validate()
    if errors:
        raise SchemaError(f"artifact {source} invalid: " + "; ".join(errors))
    return artifact


def verify_ref(ref: ArtifactRef) -> bool:
    """Recompute the referenced file sha256; False on mismatch or unreadable file."""
    try:
        return sha256_file(ref.path) == ref.sha256
    except OSError:
        return False
