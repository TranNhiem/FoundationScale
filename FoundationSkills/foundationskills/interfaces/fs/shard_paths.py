"""Resolve a dataset's declared shard list to the ONE path a consumer is handed.

Both FS corpus consumers take a single path and fill it by globbing the directory
they were given: ``rl_driver`` merges every ``*.jsonl`` in a dataset directory
and the FS SFT trainer loads every ``*.json*`` in it. Passing a directory is
therefore a claim about that directory's contents, and a wrong claim used to
train on wrong data (an eval/test split beside a train shard became training
data) or drop data (shards in other directories silently vanished) -- both
silently and without a refusal.

``resolve_shard_ref`` turns a shard list into one of three honest outcomes: the
one directory (ONLY when its glob is exactly the declared shards), the one
shard FILE (when that directory would also load foreign files), or a refusal
naming its reason. When the directory cannot be seen on this host (planning
off-node is legitimate) it is passed as before and REPORTED as unverified --
never stamped fine. Never raises, never mutates the shard list.
"""
from __future__ import annotations

from pathlib import Path


__all__ = ["resolve_shard_ref"]


def _fmt_names(names: list[str], limit: int | None = None) -> str:
    """Names sorted (every message is deterministic); the first ``limit`` of them."""
    ordered = sorted(names)
    if limit is not None:
        ordered = ordered[:limit]
    return ", ".join(ordered)


def _pattern_files(directory: Path, pattern: str) -> set[Path] | None:
    """The resolved files ``directory.glob(pattern)`` selects, or None when this host
    cannot list ``directory`` at all: an OSError there means UNVERIFIABLE, never
    "the directory holds nothing".
    """
    try:
        # Force the listing: pathlib's own glob swallows OSError and yields nothing.
        list(directory.iterdir())
        return {p.resolve() for p in directory.glob(pattern) if p.is_file()}
    except (OSError, RuntimeError):
        return None


def resolve_shard_ref(shards: list[dict], pattern: str) -> tuple[str | None, str | None, list[str]]:
    """(path, refusal, notes) for the declared ``shards`` and the consumer's ``pattern``.

    Exactly one of ``path`` and ``refusal`` is set; ``notes`` never carries a
    refusal, and every message is deterministic (names sorted). Rules, in order:

    1. No shards, or no shard with a non-empty ``path`` -> refusal "dataset
       payload has no shards to pass to --dataset". A pathless shard among
       declared ones -> "dataset shard <i> has no path" per offender (in order).
    2. Shards in more than one parent directory -> refusal naming the sorted
       directories: the consumer takes one file or one directory.
    3. One parent directory D, read as the files matching ``D.glob(pattern)``:

       a. D is listable on this host:
          - the glob is EXACTLY the declared shards -> D (it loads the declared
            data and nothing else).
          - one shard plus foreign files -> that shard's FILE + a note naming the
            files D would also load (first 5, sorted).
          - several shards plus foreign files -> refusal naming them.
          - a declared shard the glob does not match -> refusal naming it.
       b. D is absent/unreadable here, or listing raised OSError: one shard -> D
          as before (see the branch comment); several shards -> D + a note that
          it could NOT be verified (never reported as fine).
    """
    declared: list[str] = []
    pathless: list[str] = []
    for i, shard in enumerate(list(shards or [])):
        raw = shard.get("path") if isinstance(shard, dict) else None
        if isinstance(raw, str) and raw.strip():
            declared.append(raw)
        else:
            pathless.append(f"dataset shard {i} has no path")
    if not declared:
        return None, "dataset payload has no shards to pass to --dataset", []
    if pathless:
        return None, "; ".join(pathless), []

    by_dir: dict[str, list[str]] = {}
    for raw in declared:
        by_dir.setdefault(str(Path(raw).parent), []).append(raw)
    if len(by_dir) > 1:
        dirs = ", ".join(sorted(by_dir))
        return None, (
            f"dataset shards span {len(by_dir)} directories ({dirs}); the trainer takes one file or one "
            f"directory - co-locate the shards in one directory"
        ), []

    dir_str, dir_shards = next(iter(by_dir.items()))
    directory = Path(dir_str)
    on_disk = _pattern_files(directory, pattern) if directory.is_dir() else None
    if on_disk is None:
        if len(dir_shards) == 1:
            # Rule 3b, one shard: an unverifiable single-shard directory is passed as before
            # (narrowing it to the one file would rewrite a declaration this host cannot check).
            return dir_str, None, []
        return dir_str, None, [
            f"could not verify {dir_str} holds only the declared shards: directory absent on this host"
        ]

    # Compare by resolved path: the glob and the declaration may spell one file two ways.
    declared_on_disk = {Path(raw).resolve() for raw in dir_shards}
    extras = sorted(p.name for p in on_disk - declared_on_disk)
    absent = sorted(Path(raw).name for raw in dir_shards if Path(raw).resolve() not in on_disk)
    if not extras and not absent:
        return dir_str, None, []
    if len(dir_shards) == 1:
        only = dir_shards[0]
        if absent:
            return None, f"dataset shard {only} is not in {dir_str} (or does not match {pattern})", []
        note = (
            f"passing the shard file, not {dir_str}: the directory also holds {len(extras)} "
            f"other file(s) the trainer would load: {_fmt_names(extras, 5)}"
        )
        return only, None, [note]
    if extras:
        return None, (
            f"shard directory {dir_str} also holds {len(extras)} non-shard file(s) the trainer would "
            f"load: {_fmt_names(extras, 5)}; move the shards into their own directory"
        ), []
    return None, f"dataset shard(s) missing from {dir_str}: {_fmt_names(absent)}", []
