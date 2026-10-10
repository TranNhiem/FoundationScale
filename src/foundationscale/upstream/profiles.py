"""Upstream profiles: the exact framework versions each backend container provides.

Rule 3 of the upstream design (docs/research/upstream_integration.md) isolates backends, and the
two stacks really do differ (captured on GB200, 2026-10-10): the NeMo worker runs torch 2.13 /
transformers 5.12 / peft 0.21, while the HF lane runs torch 2.11 / transformers 5.5 / peft 0.18.
A profile pins what a backend was verified on; :func:`installed_versions` measures what a running
process actually has, and :func:`profile_mismatches` names every difference. A worker writes the
measured versions next to its outputs (:func:`write_profile_record`), so every result records the
upstream it ran on.

Image digests are not visible from the job environment (the containers are enroot-imported by
name), so ``image_digest`` says "unrecorded" rather than guessing; recording it is part of the
profile-build step of the upgrade process.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path

__all__ = [
    "PROFILES",
    "TRACKED_PACKAGES",
    "UpstreamProfile",
    "get_profile",
    "installed_versions",
    "profile_mismatches",
    "write_profile_record",
]

# The packages whose versions decide speech behaviour on either backend.
TRACKED_PACKAGES: tuple[str, ...] = (
    "torch",
    "transformers",
    "peft",
    "accelerate",
    "nemo_toolkit",
    "lhotse",
    "lightning",
    "omegaconf",
    "soundfile",
    "numpy",
)


@dataclass(frozen=True)
class UpstreamProfile:
    name: str
    backend: str  # "hf" | "nemo"
    container: str  # enroot container name the backend runs in
    python: str
    packages: Mapping[str, str | None] = field(default_factory=dict)  # None: must be absent
    image_digest: str = "unrecorded"
    captured: str = ""  # date the versions were measured


PROFILES: tuple[UpstreamProfile, ...] = (
    UpstreamProfile(
        name="nemo-26.08",
        backend="nemo",
        container="fs-nemo-26-08 + venv nemo-speech-asr",
        python="3.12.3",
        packages={
            "torch": "2.13.0a0+8145d630e8.nv26.6.54250401",
            "transformers": "5.12.1",
            "peft": "0.21.2",
            "accelerate": "1.15.0",
            "nemo_toolkit": "3.1.0+912d96b",
            "lhotse": "2.0.0a6",
            "lightning": "2.4.0",
            "omegaconf": "2.3.0",
            "soundfile": "0.14.0",
            "numpy": "1.26.4",
        },
        captured="2026-10-10",
    ),
    UpstreamProfile(
        name="hf-26.04",
        backend="hf",
        container="fs-g4e4b-nemo-automodel-26-04_compute",
        python="3.12.3",
        packages={
            "torch": "2.11.0a0+eb65b36914.nv26.2",
            "transformers": "5.5.0",
            "peft": "0.18.1",
            "accelerate": "1.11.0",
            "nemo_toolkit": None,
            "lhotse": None,
            "lightning": None,
            "omegaconf": None,
            "soundfile": "0.13.1",
            "numpy": "1.26.4",
        },
        captured="2026-10-10",
    ),
    # Candidate for the HF lane: the hf-26.04 container plus a venv carrying transformers 5.19
    # (with its tokenizers and huggingface_hub); every other package is hf-26.04's own, reached
    # through a .pth to the container's /opt/venv placed after the venv's site-packages.
    # Parakeet-CTC reproduces its card under it (Level 3), and the FS trainer runs: a seeded
    # 30-step fine-tune is bit-deterministic within each profile and differs across them by
    # at most 0.007 loss (transformers forward numerics) -- validation_campaigns/speech_repro.
    UpstreamProfile(
        name="hf-cand-519",
        backend="hf",
        container="fs-g4e4b-nemo-automodel-26-04_compute + venv hf-cand-519",
        python="3.12.3",
        packages={
            "torch": "2.11.0a0+eb65b36914.nv26.2",
            "transformers": "5.19.0",
            "peft": "0.18.1",
            "accelerate": "1.11.0",
            "nemo_toolkit": None,
            "lhotse": None,
            "lightning": None,
            "omegaconf": None,
            "soundfile": "0.13.1",
            "numpy": "1.26.4",
        },
        captured="2026-10-10",
    ),
)


def get_profile(name: str) -> UpstreamProfile:
    for profile in PROFILES:
        if profile.name == name:
            return profile
    known = ", ".join(p.name for p in PROFILES)
    raise ValueError(f"unknown upstream profile {name!r}; known: {known}")


def installed_versions(packages: Iterable[str] = TRACKED_PACKAGES) -> dict[str, str | None]:
    """The installed version of each package in THIS process (None when not installed)."""
    import importlib.metadata as metadata

    out: dict[str, str | None] = {}
    for name in packages:
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            out[name] = None
    return out


def profile_mismatches(profile: UpstreamProfile, installed: Mapping[str, str | None]) -> list[str]:
    """Every package whose installed version differs from the profile (``[]`` when it matches).

    Codes: ``version:<pkg>:<pinned>!=<installed>``, ``missing:<pkg>``, ``unexpected:<pkg>``.
    """
    problems: list[str] = []
    for pkg, pinned in profile.packages.items():
        got = installed.get(pkg)
        if pinned is None and got is not None:
            problems.append(f"unexpected:{pkg}")
        elif pinned is not None and got is None:
            problems.append(f"missing:{pkg}")
        elif pinned != got:
            problems.append(f"version:{pkg}:{pinned}!={got}")
    return problems


def write_profile_record(out_dir: Path, profile_name: str | None = None) -> dict[str, object]:
    """Write ``upstream_profile.json`` (measured versions, and mismatches vs a named profile)."""
    import sys

    record: dict[str, object] = {
        "python": sys.version.split()[0],
        "packages": installed_versions(),
        "profile": profile_name,
    }
    if profile_name is not None:
        record["mismatches"] = profile_mismatches(get_profile(profile_name), record["packages"])  # type: ignore[arg-type]
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "upstream_profile.json").write_text(json.dumps(record, indent=2, sort_keys=True))
    return record
