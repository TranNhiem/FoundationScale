"""Spec-compliant export tree + plugin manifests generated from the skill packages.

The Agent Skills spec requires a skill's directory name to equal its frontmatter
``name`` (``fskills-auto-research``) and ships only ``SKILL.md`` plus the skill's
own bundle (``references/``, ``scripts/``, ``assets/``).  A Python package
directory cannot satisfy that naming rule, so the packages stay the single
source of truth and this module *generates* the export tree that Claude Code
plugins, ``npx skills add``, OpenCode, Codex and Gemini CLI discover:

* ``<plugin root>/skills/<name>/SKILL.md`` -- byte-identical to the package's --
  plus the bundle directories already in the package, without ``__pycache__``
  or ``*.pyc``,
* ``<plugin root>/.claude-plugin/plugin.json``,
* ``<repo root>/.claude-plugin/marketplace.json`` (the repo root being the
  plugin root's parent).

Nothing else is exported: no ``skill.py``, no ``evals/``, no ``BENCHMARK.md`` or
``BEHAVIOUR*.md``.  ``expected_tree`` is the plan (repo-root-relative posix
path -> exact bytes), ``write_tree`` materialises it atomically and prunes stale
``fskills-*`` export directories, ``check_tree`` reports the drift of a tree
that was never regenerated.  An invalid ``SKILL.md`` is never exported: its
problems refuse the whole export.
"""
from __future__ import annotations

import importlib.resources
import json
import os
import re
import shutil
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import yaml

from foundationskills.agent.routing_eval import (
    BUNDLE_DIRS,
    RoutingEvalRefused,
    load_index,
)

PLUGIN_NAME = "foundationskills"
PLUGIN_DESCRIPTION = (
    "FoundationSkills: probe-first, refusal-honest agent skills for planning, launching, "
    "evaluating and researching FoundationScale training runs (statuses PASS/RED/UNMEASURED/REFUSED)."
)
MARKETPLACE_NAME = "foundationscale"
MARKETPLACE_SCHEMA = "https://json.schemastore.org/claude-code-marketplace.json"
AUTHOR_NAME = "TranNhiem"
AUTHOR_URL = "https://github.com/TranNhiem"
HOMEPAGE = "https://github.com/TranNhiem/FoundationScale"
REPOSITORY = "https://github.com/TranNhiem/FoundationScale"
LICENSE = "MIT"
EXPORT_DIR_PREFIX = "fskills-"

ALLOWED_KEYS = frozenset({"name", "description", "license", "compatibility", "metadata", "allowed-tools"})
NAME_RE = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$")
NAME_CHARS = (1, 64)
DESCRIPTION_CHARS = (1, 1024)
COMPATIBILITY_CHARS = 500
_VERSION_RE = re.compile(r'^version\s*=\s*"([^"]+)"\s*$', re.MULTILINE)
_UNVERSIONED = "0.0.0"


class ExportRefused(Exception):
    """A declaration that must not be exported (invalid SKILL.md, no source checkout)."""


def spec_problems(text: str, dirname: str) -> list[str]:
    """The Agent Skills spec checks for one ``SKILL.md``; ``[]`` means exportable.

    Frontmatter block present, closed and a YAML mapping; only the allowed
    top-level keys; ``name`` 1-64 chars matching ``^[a-z0-9]+(-[a-z0-9]+)*$`` and
    equal to the skill directory name; ``description`` 1-1024 chars;
    ``compatibility`` (optional) a string of at most 500 chars; ``metadata``
    (optional) a string -> string mapping.
    """
    problems: list[str] = []
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return [f"{dirname}: SKILL.md has no frontmatter block (the file must open with `---`)"]
    closing = next((i for i in range(1, len(lines)) if lines[i].strip() == "---"), None)
    if closing is None:
        return [f"{dirname}: the frontmatter block is not closed (missing its `---`)"]
    try:
        data = yaml.safe_load("\n".join(lines[1:closing]))
    except yaml.YAMLError as exc:
        return [f"{dirname}: the frontmatter is not valid YAML: {exc}"]
    if not isinstance(data, dict):
        return [f"{dirname}: the frontmatter must be a YAML mapping"]
    for key in data:
        if not isinstance(key, str) or key not in ALLOWED_KEYS:
            problems.append(
                f"{dirname}: frontmatter key {key!r} is not allowed (allowed: {', '.join(sorted(ALLOWED_KEYS))})"
            )
    name = data.get("name")
    if name is None:
        problems.append(f"{dirname}: the frontmatter key `name` is required")
    elif not isinstance(name, str):
        problems.append(f"{dirname}: `name` must be a string, got {type(name).__name__}")
    else:
        low, high = NAME_CHARS
        if not low <= len(name) <= high:
            problems.append(f"{dirname}: `name` must be {low}..{high} characters, got {len(name)}")
        elif not NAME_RE.match(name):
            problems.append(f"{dirname}: `name` must match ^[a-z0-9]+(-[a-z0-9]+)*$, got {name!r}")
        if name != dirname:
            problems.append(f"{dirname}: `name` {name!r} must equal the skill directory name {dirname!r}")
    description = data.get("description")
    if description is None:
        problems.append(f"{dirname}: the frontmatter key `description` is required")
    elif not isinstance(description, str):
        problems.append(f"{dirname}: `description` must be a string, got {type(description).__name__}")
    else:
        low, high = DESCRIPTION_CHARS
        if not low <= len(description) <= high:
            problems.append(
                f"{dirname}: `description` must be {low}..{high} characters, got {len(description)}"
            )
    compatibility = data.get("compatibility")
    if compatibility is not None:
        if not isinstance(compatibility, str):
            problems.append(f"{dirname}: `compatibility` must be a string, got {type(compatibility).__name__}")
        elif len(compatibility) > COMPATIBILITY_CHARS:
            problems.append(
                f"{dirname}: `compatibility` must be at most {COMPATIBILITY_CHARS} characters, "
                f"got {len(compatibility)}"
            )
    metadata = data.get("metadata")
    if metadata is not None:
        if not isinstance(metadata, dict):
            problems.append(f"{dirname}: `metadata` must be a mapping of strings to strings")
        else:
            for key, value in metadata.items():
                if not isinstance(key, str) or not isinstance(value, str):
                    problems.append(
                        f"{dirname}: `metadata` must map strings to strings, got {key!r} -> {value!r}"
                    )
    return problems


def _index() -> list[dict]:
    """``routing_eval.load_index`` with every refusal taken to an ``ExportRefused``."""
    try:
        return load_index()
    except RoutingEvalRefused as exc:
        raise ExportRefused(f"export-skills: {exc}") from exc


def _bundle_dirs() -> list[tuple[str, str]]:
    """``routing_eval.BUNDLE_DIRS`` as (package directory, exported directory) pairs."""
    pairs: list[tuple[str, str]] = []
    for entry in BUNDLE_DIRS:
        parts = entry if isinstance(entry, (tuple, list)) else (entry,)
        pairs.append((str(parts[0]), str(parts[-1])))
    return pairs


def _walk_files(base: Any) -> Iterator[tuple[str, bytes]]:
    """(path relative to *base*, exact bytes) for every file under *base*.

    ``__pycache__`` directories and ``*.pyc`` are never part of a bundle.
    """
    if not base.is_dir():
        return
    for child in sorted(base.iterdir(), key=lambda part: part.name):
        if child.is_dir():
            if child.name == "__pycache__":
                continue
            for rel, data in _walk_files(child):
                yield f"{child.name}/{rel}", data
        elif not child.name.endswith(".pyc"):
            yield child.name, child.read_bytes()


def _json_bytes(payload: dict) -> bytes:
    return (json.dumps(payload, indent=2, ensure_ascii=False) + "\n").encode("utf-8")


def _target(repo_root: Path, rel: str) -> Path:
    return repo_root.joinpath(*rel.split("/"))


def _version(plugin_root: Path) -> str:
    """The version the two manifests carry: ``__version__`` else ``[project].version``."""
    import foundationskills

    declared = getattr(foundationskills, "__version__", None)
    if isinstance(declared, str) and declared.strip():
        return declared.strip()
    here = Path(foundationskills.__file__).resolve()
    candidates = (
        plugin_root / "pyproject.toml",
        plugin_root.parent / "pyproject.toml",
        here.parent.parent / "pyproject.toml",
    )
    for candidate in candidates:
        if candidate.is_file():
            match = _VERSION_RE.search(candidate.read_text(encoding="utf-8"))
            if match:
                return match.group(1)
    return _UNVERSIONED  # a checkout that declares no version exports "0.0.0", not a lie


def expected_tree(plugin_root: Path) -> dict[str, bytes]:
    """Every exported file -- posix path relative to the *repo root* -> its exact bytes.

    The sources come from ``routing_eval.load_index()`` (package, name) and are
    read through ``importlib.resources.files``.  Every ``SKILL.md`` is validated
    by ``spec_problems`` first: one problem refuses the whole export, listing
    every problem.
    """
    skills_prefix = f"{plugin_root.name}/skills/"
    version = _version(plugin_root)
    tree: dict[str, bytes] = {}
    problems: list[str] = []
    names: list[str] = []
    for entry in _index():
        package, name = str(entry["package"]), str(entry["name"])
        if name in names:
            problems.append(f"{name}: two skill packages declare the same skill name ({package})")
            continue
        names.append(name)
        root = importlib.resources.files(package)
        md_bytes = root.joinpath("SKILL.md").read_bytes()
        problems.extend(spec_problems(md_bytes.decode("utf-8"), name))
        tree[f"{skills_prefix}{name}/SKILL.md"] = md_bytes
        for source, destination in _bundle_dirs():
            for rel, data in _walk_files(root.joinpath(source)):
                tree[f"{skills_prefix}{name}/{destination}/{rel}"] = data
    if problems:
        raise ExportRefused("export-skills: " + "; ".join(problems))
    plugin = {
        "name": PLUGIN_NAME,
        "version": version,
        "description": PLUGIN_DESCRIPTION,
        "author": {"name": AUTHOR_NAME},
        "homepage": HOMEPAGE,
        "repository": REPOSITORY,
        "license": LICENSE,
        "skills": "./skills",
    }
    marketplace = {
        "$schema": MARKETPLACE_SCHEMA,
        "name": MARKETPLACE_NAME,
        "owner": {"name": AUTHOR_NAME, "url": AUTHOR_URL},
        "plugins": [
            {
                "name": PLUGIN_NAME,
                "source": f"./{plugin_root.name}",
                "version": version,
                "description": PLUGIN_DESCRIPTION,
                "license": LICENSE,
                "skills": [f"./skills/{name}" for name in sorted(names)],
            }
        ],
    }
    tree[f"{plugin_root.name}/.claude-plugin/plugin.json"] = _json_bytes(plugin)
    tree[".claude-plugin/marketplace.json"] = _json_bytes(marketplace)
    return tree


def write_tree(repo_root: Path, plugin_root: Path) -> list[str]:
    """Write the expected tree and prune stale ``fskills-*`` export directories.

    Returns the repo-root-relative paths written or removed.  Every file is
    staged as ``<name>.tmp`` and moved into place with ``os.replace``.
    """
    tree = expected_tree(plugin_root)
    done: list[str] = []
    for rel in sorted(tree):
        target = _target(repo_root, rel)
        target.parent.mkdir(parents=True, exist_ok=True)
        staged = target.with_name(f"{target.name}.tmp")
        staged.write_bytes(tree[rel])
        os.replace(staged, target)
        done.append(rel)
    done.extend(_prune_exports(repo_root, plugin_root, tree))
    return done


def _prune_exports(repo_root: Path, plugin_root: Path, tree: dict[str, bytes]) -> list[str]:
    """Remove stale export directories -- only ``fskills-*`` dirs under ``skills/``."""
    skills_prefix = f"{plugin_root.name}/skills/"
    exported = {rel[len(skills_prefix):].split("/")[0] for rel in tree if rel.startswith(skills_prefix)}
    skills_root = plugin_root / "skills"
    removed: list[str] = []
    if not skills_root.is_dir():
        return removed
    for child in sorted(skills_root.iterdir(), key=lambda part: part.name):
        if not child.is_dir():
            continue
        if not child.name.startswith(EXPORT_DIR_PREFIX) or child.name in exported:
            continue
        shutil.rmtree(child)
        removed.append(f"{skills_prefix}{child.name}")
    return removed


def check_tree(repo_root: Path, plugin_root: Path) -> list[str]:
    """Drift problems of the export tree on disk; ``[]`` means it is in sync.

    Missing (or byte-differing) expected files and files that no registered
    package would export are all drift.  ``skill.py``, ``evals/`` and friends
    are never expected, so their absence is correct and their presence outside a
    ``fskills-*`` directory is somebody else's business.
    """
    tree = expected_tree(plugin_root)
    skills_prefix = f"{plugin_root.name}/skills/"
    problems: list[str] = []
    for rel in sorted(tree):
        target = _target(repo_root, rel)
        if not target.is_file():
            problems.append(f"missing: {rel}")
        elif target.read_bytes() != tree[rel]:
            problems.append(f"differs: {rel}")
    skills_root = plugin_root / "skills"
    if skills_root.is_dir():
        for child in sorted(skills_root.iterdir(), key=lambda part: part.name):
            if not (child.is_dir() and child.name.startswith(EXPORT_DIR_PREFIX)):
                continue
            for rel, _data in _walk_files(child):
                extra = f"{skills_prefix}{child.name}/{rel}"
                if extra not in tree:
                    problems.append(f"extra: {extra}")
    return problems


def _roots() -> tuple[Path, Path]:
    """``(repo_root, plugin_root)`` of the FoundationSkills source checkout.

    An installed wheel has no ``pyproject.toml`` under the plugin root, so its
    version is unverifiable: that checkout refuses to export.
    """
    import foundationskills

    plugin_root = Path(foundationskills.__file__).resolve().parent.parent
    repo_root = plugin_root.parent
    if not (plugin_root / "pyproject.toml").is_file():
        raise ExportRefused("export-skills needs a FoundationSkills source checkout")
    return repo_root, plugin_root
