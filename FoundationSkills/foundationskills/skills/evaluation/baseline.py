"""Baseline score cache keyed by a fingerprint of everything that changes a score.

A cached baseline is only reused when its stored fingerprint equals the one
recomputed now; any drift (harness version, task, few-shot, seed, gen kwargs,
chat-template mode, dtype, limit, backend, base-model identity) is a miss, so a
stale number can never be compared against a fresh checkpoint score.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from foundationskills.core.artifacts import _atomic_write_json
from foundationskills.core.provenance import sha256_file, sha256_json

_WEIGHT_SUFFIXES = (".safetensors", ".bin", ".pt", ".json")


def model_identity(model_dir: Path) -> dict[str, Any]:
    """Cheap identity: resolved path, config.json hash, and top-level weight file names+sizes."""
    root = Path(model_dir).resolve()
    config = root / "config.json"
    files = sorted(
        (p.name, p.stat().st_size) for p in root.iterdir() if p.is_file() and p.suffix in _WEIGHT_SUFFIXES
    )
    return {
        "path": str(root),
        "config_sha256": sha256_file(config) if config.is_file() else None,
        "files": [list(f) for f in files],
    }


def fingerprint(fields: dict[str, Any]) -> str:
    return sha256_json(fields)


class BaselineCache:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def path(self, base_identity: dict[str, Any], fp: str) -> Path:
        name = re.sub(r"[^A-Za-z0-9._-]", "_", Path(base_identity["path"]).name) or "base"
        return self.root / f"{name}-{sha256_json(base_identity['path'])[:8]}" / f"{fp}.json"

    def get(self, base_identity: dict[str, Any], fp: str) -> dict[str, Any] | None:
        target = self.path(base_identity, fp)
        try:
            record = json.loads(target.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        if not isinstance(record, dict) or record.get("fingerprint") != fp:
            return None  # a record whose stored fingerprint disagrees is stale, never reused
        return record

    def put(self, base_identity: dict[str, Any], fp: str, record: dict[str, Any]) -> Path:
        target = self.path(base_identity, fp)
        _atomic_write_json(target, {**record, "fingerprint": fp})
        return target
