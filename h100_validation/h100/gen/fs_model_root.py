"""Model-agnostic model-root resolution for FoundationScale.

Measured on a real 8xH100 estate (2026-08-31), "the model root" has three
shapes: self-contained (L1), config-overlay with symlinked payloads (L2),
and commit-SHA nested with no root config (L3). Code that assumes any one
of them is silently wrong on the others. Zero candidates examined is
UNMEASURED, never a pass; ambiguous, missing, or unreadable is an error.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

LAYOUT_SELF_CONTAINED = "self-contained"
LAYOUT_OVERLAY = "config-overlay"
LAYOUT_NESTED = "commit-nested"
LAYOUT_NESTED_OVERLAY = "commit-nested-overlay"


class ModelRootError(Exception):
    """Ambiguous, missing, or unreadable model root: fail closed."""


@dataclass(frozen=True)
class ModelRoot:
    declared_root: str          # what the operator named, absolute, normalised
    config_path: str            # the single config actually found
    config_dir: str             # the directory a loader should be pointed at
    layout: str                 # one of the LAYOUT_* constants
    bind_closure: tuple[str, ...]  # roots that MUST be mounted, root first
    symlinks_total: int         # symlinks seen beneath declared_root
    symlinks_escaping: int      # of those, resolving OUTSIDE declared_root
    config_candidates: int      # denominator behind the choice (1 on success)


def _is_within(path: str, ancestor: str) -> bool:
    try:
        return os.path.commonpath((path, ancestor)) == ancestor
    except ValueError:  # different drives; treat as outside
        return False


def _on_walk_error(err: OSError) -> None:
    raise ModelRootError(f"cannot walk {err.filename!r}: {err.strerror}")


def _depth_of(config_path: str, root: str) -> int:
    """Directory depth of a config below root; 0 means the config sits at the root."""
    rel = os.path.relpath(os.path.dirname(config_path), root)
    return 0 if rel == os.curdir else rel.count(os.sep) + 1


def _find_configs(root: str, config_name: str, max_depth: int) -> list[str]:
    candidates: list[str] = []
    for dirpath, dirnames, filenames in os.walk(root, onerror=_on_walk_error):
        rel = os.path.relpath(dirpath, root)
        depth = 0 if rel == os.curdir else rel.count(os.sep) + 1
        if depth > max_depth:
            dirnames[:] = []
            continue
        if config_name in filenames:
            candidates.append(os.path.join(dirpath, config_name))
    candidates.sort()
    return candidates


def _scan_symlinks(root: str) -> tuple[int, int, list[str]]:
    total, escaping = 0, 0
    escaping_targets: list[str] = []
    for dirpath, dirnames, filenames in os.walk(root, onerror=_on_walk_error):
        for name in (*dirnames, *filenames):
            path = os.path.join(dirpath, name)
            if not os.path.islink(path):
                continue
            total += 1
            try:
                raw = os.readlink(path)
            except OSError as exc:
                raise ModelRootError(f"cannot readlink {path!r}: {exc}") from exc
            raw_abs = raw if os.path.isabs(raw) else os.path.normpath(
                os.path.join(dirpath, raw))
            if os.path.exists(path):
                resolved = os.path.realpath(path)
            else:
                resolved = raw_abs
                if _is_within(raw_abs, root):
                    raise ModelRootError(
                        f"broken symlink {path!r} -> {raw!r} resolves INSIDE the "
                        "declared root: the tree is corrupt and no bind can fix it")
            if not _is_within(resolved, root):
                escaping += 1
                escaping_targets.append(resolved)
    return total, escaping, escaping_targets


def _bind_closure(root: str, escaping_targets: list[str]) -> tuple[str, ...]:
    closure: list[str] = [root]
    if escaping_targets:
        parents = sorted({os.path.dirname(t) for t in escaping_targets})
        common = os.path.commonpath(parents)
        if common == os.sep or (common != root and _is_within(root, common)):
            raise ModelRootError(
                f"{len(escaping_targets)} escaping symlink target(s) share only "
                f"{common!r} as a common mount; widening a mount silently is how "
                "a narrow bug becomes an estate-wide one -- declare "
                "FS_BIND_PATHS explicitly instead")
        closure.append(common)
    out: list[str] = []
    for entry in closure:
        if entry in out:
            continue
        if entry != root and any(
                other != entry and _is_within(entry, other) for other in closure):
            continue  # already contained in another entry
        out.append(entry)
    return tuple(out)


def resolve_model_root(declared_root: str | os.PathLike[str], *,
                       config_name: str = "config.json",
                       max_depth: int = 2) -> ModelRoot:
    try:
        root = os.path.realpath(os.fspath(declared_root))
    except (OSError, TypeError) as exc:
        raise ModelRootError(
            f"cannot resolve model root {declared_root!r}: {exc}") from exc
    if not os.path.isdir(root):
        raise ModelRootError(
            f"model root {root!r} does not exist or is not a directory")

    candidates = _find_configs(root, config_name, max_depth)
    found = len(candidates)
    if found == 0:
        raise ModelRootError(
            f"found 0 {config_name} under {root} at depth <={max_depth}; "
            "a model root with no config is UNMEASURED, not empty")

    # Ambiguity is a property of the SHALLOWEST populated depth, not of the whole
    # subtree. MEASURED on a real estate: stock upstream checkpoints ship a nested
    # variant directory beside an unambiguous root config -- gpt-oss-20b has
    # original/config.json and sentence-transformers models have 1_Pooling/config.json,
    # both d0=1 d1=1. Counting flat calls those ambiguous and refuses a perfectly
    # well-formed model root. Genuine ambiguity measures differently: a vendor family
    # directory has NO root config and several one level down (d0=0 d1=2, d0=0 d1=7).
    # Both denominators are reported, so narrowing to the shallowest depth stays
    # visible rather than quietly discarding the rest of the subtree.
    by_depth = [(_depth_of(c, root), c) for c in candidates]
    shallowest = min(depth for depth, _ in by_depth)
    contenders = [c for depth, c in by_depth if depth == shallowest]
    n = len(contenders)
    if n > 1:
        shown = ", ".join(repr(os.path.relpath(c, root)) for c in contenders[:5])
        raise ModelRootError(
            f"found {n} {config_name} candidates at depth {shallowest} under {root} "
            f"({found} in the subtree at depth <={max_depth}) "
            f"(showing up to 5: {shown}); choosing among them "
            "would be a guess, not a measurement")

    config_path = contenders[0]
    config_dir = os.path.dirname(config_path)
    nested = os.path.relpath(config_dir, root) != os.curdir
    total, escaping, targets = _scan_symlinks(root)
    if nested:
        layout = LAYOUT_NESTED_OVERLAY if escaping else LAYOUT_NESTED
    else:
        layout = LAYOUT_OVERLAY if escaping else LAYOUT_SELF_CONTAINED
    return ModelRoot(
        declared_root=root,
        config_path=config_path,
        config_dir=config_dir,
        layout=layout,
        bind_closure=_bind_closure(root, targets),
        symlinks_total=total,
        symlinks_escaping=escaping,
        config_candidates=1,
    )


def describe(root: ModelRoot) -> str:
    """One log-safe line; absolute paths are estate identifiers, never printed."""
    rel = os.path.relpath(root.config_path, root.declared_root)
    return (f"model-root: layout={root.layout} config={rel} "
            f"binds={len(root.bind_closure)} "
            f"symlinks={root.symlinks_escaping}/{root.symlinks_total} escaping")
