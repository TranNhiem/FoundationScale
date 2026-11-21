"""Collect the HarnessManifest of a launch spec at the moment it is written (C2.3).

Every ``fs_launch_spec`` passes through ``core.artifacts.write_artifact``, which calls ``attach`` here, so the
typed-back plan hash (``fskills hash``) commits to the model-harness pair whichever skill emitted the spec. Values
that cannot be measured on this host are recorded as ``"unmeasured"``, never guessed.
"""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any

from foundationskills.core.harness import UNMEASURED, HarnessManifest, manifest_payload, skill_pin, validate_manifest
from foundationskills.core.provenance import Provenance, sha256_file

POOL_REGISTRY = Path.home() / ".fskfan" / "pool" / "models.yaml"


def _skill_md(skill_name: str) -> Path | None:
    """The SKILL.md beside the registered skill's module, or None (unregistered skill, no file)."""
    try:
        from foundationskills.core import get_skill
        from foundationskills.skills import register_builtin_skills

        register_builtin_skills()
        path = Path(inspect.getfile(type(get_skill(skill_name)))).with_name("SKILL.md")
    except Exception:  # noqa: BLE001 - an unknown skill only loses its pin hash, never the launch
        return None
    return path if path.is_file() else None


def _routing_registry_version() -> str:
    try:
        from foundationskills.agent import routing_eval

        return routing_eval._index_sha256(routing_eval.load_index())
    except Exception:  # noqa: BLE001
        return UNMEASURED


def _substrate(spec: dict[str, Any]) -> tuple[str, str | None]:
    """(substrate image, interpreter sha256): the container when the spec names one, else the interpreter."""
    container = (spec.get("serve") or {}).get("container") or spec.get("container")
    argv = spec.get("argv") or []
    exe = Path(str(argv[0])) if argv else None
    exe_sha = sha256_file(exe) if exe is not None and exe.is_file() else None
    if container:
        return str(container), exe_sha
    return (str(exe) if exe is not None else UNMEASURED), exe_sha


def collect(spec: dict[str, Any], provenance: Provenance, pool_registry: Path = POOL_REGISTRY) -> dict[str, Any]:
    """The schema-shaped manifest payload for ``spec`` as emitted by ``provenance.skill``."""
    image, exe_sha = _substrate(spec)
    out_dir = spec.get("output_dir")
    seed = spec.get("seed")
    manifest = HarnessManifest(
        skills=(skill_pin(provenance.skill, provenance.skill_version, _skill_md(provenance.skill)),),
        routing_registry_version=_routing_registry_version(),
        foundationskills_version=provenance.fskills_version,
        source_commit=provenance.fs_commit or "not_a_repository",
        pool_registry_sha256=sha256_file(pool_registry) if pool_registry.is_file() else UNMEASURED,
        pool_entries=tuple(sorted({str(e) for e in spec.get("pool_entries") or ()})),
        substrate_image=image,
        seed=seed if isinstance(seed, int) and not isinstance(seed, bool) else None,
        reset_semantics=("fresh process per launch; outputs to a new location that is never overwritten: " + str(out_dir))
        if out_dir else "fresh process per launch",
        workspace_snapshot=f"cwd={spec.get('cwd')}; output_dir={out_dir}",
        interpreter_sha256=exe_sha,
    )
    return manifest_payload(manifest)


def attach(spec: dict[str, Any], provenance: Provenance) -> dict[str, Any]:
    """A copy of ``spec`` carrying a valid ``harness_manifest`` (an existing valid one is kept as is)."""
    if "harness_manifest" in spec and not validate_manifest(spec["harness_manifest"]):
        return dict(spec)
    return {**spec, "harness_manifest": collect(spec, provenance)}
