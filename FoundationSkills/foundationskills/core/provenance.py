
"""Hashing and provenance records for skill inputs and outputs."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json(obj: Any) -> str:
    """Canonical JSON string used for content hashes."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_json(obj: Any) -> str:
    return sha256_bytes(canonical_json(obj).encode("utf-8"))


@dataclass(frozen=True)
class Provenance:
    """Who produced an artifact, from which inputs, when, and where."""

    skill: str
    skill_version: str
    fskills_version: str
    fs_version: str | None
    fs_commit: str | None
    inputs: dict[str, str]
    created_at: str
    host: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "skill": self.skill,
            "skill_version": self.skill_version,
            "fskills_version": self.fskills_version,
            "fs_version": self.fs_version,
            "fs_commit": self.fs_commit,
            "inputs": dict(self.inputs),
            "created_at": self.created_at,
            "host": self.host,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Provenance":
        inputs = data.get("inputs", {})
        return cls(
            skill=str(data["skill"]),
            skill_version=str(data["skill_version"]),
            fskills_version=str(data["fskills_version"]),
            fs_version=None if data.get("fs_version") is None else str(data.get("fs_version")),
            fs_commit=None if data.get("fs_commit") is None else str(data.get("fs_commit")),
            inputs={str(k): str(v) for k, v in dict(inputs).items()},
            created_at=str(data["created_at"]),
            host=str(data["host"]),
        )


def _detect_fs_version() -> str | None:
    try:
        return str(importlib.metadata.version("foundationscale"))
    except Exception:
        pass
    try:
        import foundationscale  # type: ignore

        return str(getattr(foundationscale, "__version__")) if hasattr(foundationscale, "__version__") else None
    except Exception:
        return None


def make_provenance(skill: str, skill_version: str, inputs: dict[str, str] | None = None) -> Provenance:
    """Build provenance best-effort; this helper never raises."""
    from foundationskills import __version__ as fskills_version

    try:
        fs_version = _detect_fs_version()
    except Exception:
        fs_version = None
    try:
        created = datetime.now(timezone.utc).isoformat()
        host = platform.node() or "unknown-host"
    except Exception:
        created = "1970-01-01T00:00:00+00:00"
        host = "unknown-host"
    return Provenance(
        skill=str(skill),
        skill_version=str(skill_version),
        fskills_version=str(fskills_version),
        fs_version=fs_version,
        fs_commit=os.environ.get("FS_COMMIT"),
        inputs=dict(inputs or {}),
        created_at=created,
        host=host,
    )
