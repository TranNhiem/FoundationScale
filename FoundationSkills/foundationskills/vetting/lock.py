"""Skills.lock (``fskills.skills_lock/1``) and the vetting gate's structural checks (A6.1).

A vetting run compares three things: the bytes on disk (:func:`read_tree`), the pins a human recorded in
``skills.lock``, and the reviews a human signed over exactly those bytes. This module loads and digests the
first two and runs the four structural (non-textual) checks of the gate -- VG-01 source, VG-02 pins,
VG-03 licence, VG-10 scripts policy -- returning :class:`CheckResult` rows shaped like the scanner's
(:mod:`foundationskills.vetting.scan`).

Digests
-------
:func:`file_sha256` is the hex sha256 of one file's bytes. :func:`subset_digest` is the digest a review must
match and the digest compared against ``files_sha256``: sha256 of the utf-8 concatenation of
``"<relpath>\\0<sha256 hex of bytes>\\n"`` over the reviewed paths sorted, the sha256 of nothing for no paths;
:func:`tree_digest` is that over the whole tree. A review therefore only counts while its ``files_sha256``
still equals the digest of the current bytes of exactly the paths it covers -- a scripts review covers every
``scripts/`` path, a github review every ``.github/`` path, a network review one file, a divergence review
``SKILL.md`` -- which is what forces re-vetting the moment a file moves (:func:`approved_review` returns the
latest matching approval and nothing else).

Refusals
--------
:func:`load_lock` and :func:`lock_entry` refuse rather than guess: a missing lock file, unparseable JSON, the
wrong ``schema``, a lock without ``allowlist`` or ``skills``, or a skill the lock does not pin. The checks
refuse unverifiable input too: VG-01 a source or publisher that is not on the allowlist; VG-02 a commit that is
not 40 lowercase hex, a file the lock forgot (extra), a file the tree forgot (missing) or a file whose sha256
moved (mismatch), each path named; VG-03 a licence the lock and the ``SKILL.md`` frontmatter disagree about,
any ``EPL-*`` (weak copyleft excluded) or an SPDX id outside ``allowlist.spdx``. Absence of evidence is never
PASS: a tree with no licence at all is UNMEASURED, and a ``scripts/`` directory no reviewer signed is REFUSED --
whatever a signed scripts review approves is never executed by the gate, only recorded.

Determinism
-----------
Nothing here touches a network: it reads one JSON file and the bytes of one tree (``files: dict[str, bytes]``,
relative posix paths). Digests sort their paths and problem notes sort their paths, reviews break same-day ties
by lock order, and :func:`read_tree` refuses a symlink whose target leaves the root instead of silently
recording a tree a human cannot reproduce.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Mapping
from pathlib import Path

from .errors import VettingRefused
from .scan import CheckResult, Hit

LOCK_SCHEMA = "fskills.skills_lock/1"
__all__ = [
    "LOCK_SCHEMA",
    "CheckResult",
    "Hit",
    "VettingRefused",
    "approved_review",
    "check_license",
    "check_pins",
    "check_scripts_policy",
    "check_source",
    "file_sha256",
    "load_lock",
    "lock_entry",
    "read_tree",
    "subset_digest",
    "tree_digest",
]

_COMMIT = re.compile(r"[0-9a-f]{40}")
_NO_HITS: tuple[Hit, ...] = ()


def read_tree(root: Path) -> dict[str, bytes]:
    """Return every regular file under *root* as ``{relative posix path: bytes}``, refusing escapes."""
    root = Path(root).resolve()
    if not root.is_dir():
        raise VettingRefused(f"skill root is not a directory: {root}")
    tree: dict[str, bytes] = {}
    for path in sorted(root.rglob("*")):
        rel = path.relative_to(root).as_posix()
        if path.is_symlink():
            if not path.resolve().is_relative_to(root):
                raise VettingRefused(f"symlink escapes the skill root: {rel}")
            if not path.is_file():
                continue
        elif not path.is_file():
            continue   # directories, sockets and pipes are not tree members
        tree[rel] = path.read_bytes()
    return tree


def file_sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def subset_digest(files: Mapping[str, bytes], paths: Iterable[str]) -> str:
    """Digest of ``"<relpath>\\0<sha256>\\n"`` over *paths* sorted (the review ``files_sha256``)."""
    chunks: list[str] = []
    for rel in sorted({str(p) for p in paths}):
        if rel not in files:
            raise VettingRefused(f"reviewed path is not in the tree: {rel}")
        chunks.append(f"{rel}\0{file_sha256(files[rel])}\n")
    return file_sha256("".join(chunks).encode("utf-8"))


def tree_digest(files: Mapping[str, bytes]) -> str:
    return subset_digest(files, list(files))


def load_lock(path: Path) -> dict:
    """Load ``skills.lock``; refuse a missing file, bad JSON, a wrong schema or missing keys."""
    path = Path(path)
    if not path.is_file():
        raise VettingRefused(f"lock file not found: {path}")
    try:
        lock = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        raise VettingRefused(f"lock file unreadable or not valid JSON: {path} ({exc})") from exc
    if not isinstance(lock, dict):
        raise VettingRefused(f"lock file is not a mapping: {path}")
    if lock.get("schema") != LOCK_SCHEMA:
        raise VettingRefused(f"lock schema is {lock.get('schema')!r}, expected {LOCK_SCHEMA!r}")
    for key in ("allowlist", "skills"):
        if not isinstance(lock.get(key), Mapping):
            raise VettingRefused(f"lock file is missing the {key!r} mapping: {path}")
    for key in ("sources", "publishers", "spdx"):
        if key not in lock["allowlist"]:
            raise VettingRefused(f"lock allowlist is missing {key!r}: {path}")
    return lock


def lock_entry(lock: Mapping, name: str) -> dict:
    """Return the pinned entry for *name*; refuse a skill the lock does not pin."""
    skills = lock.get("skills")
    entry = skills.get(name) if isinstance(skills, Mapping) else None
    if not isinstance(entry, Mapping):
        raise VettingRefused(f"skill {name!r} is not pinned in the lock")
    return dict(entry)


def _latest(entry: Mapping, kind: str, digest: str, decision: str) -> dict | None:
    """Latest review of *kind* over *digest* with *decision*, ties broken by lock order."""
    best: dict | None = None
    best_key = ("", -1)
    for index, review in enumerate(entry.get("reviews") or ()):
        if not isinstance(review, Mapping):
            continue
        if review.get("kind") != kind or review.get("files_sha256") != digest:
            continue
        if review.get("decision") != decision:
            continue
        key = (str(review.get("date", "")), index)
        if best is None or key >= best_key:
            best, best_key = dict(review), key
    return best


def approved_review(entry: Mapping, kind: str, digest: str) -> dict | None:
    """Latest ``"approve"`` review of *kind* whose ``files_sha256`` equals *digest*, else ``None``."""
    return _latest(entry, kind, digest, "approve")


def review_decision(entry: Mapping, kind: str, digest: str) -> dict | None:
    """The deciding review of *kind* over *digest*: any matching ``"reject"`` wins over an approval (fail closed)."""
    return _latest(entry, kind, digest, "reject") or _latest(entry, kind, digest, "approve")


def _describe(review: Mapping) -> str:
    return f"by {review.get('reviewer', '<unknown>')} on {review.get('date', '<undated>')}"


def check_source(lock: Mapping, entry: Mapping) -> CheckResult:
    """VG-01: pinned source and publisher must both be on the allowlist."""
    allowlist = lock.get("allowlist") or {}
    sources = {str(item) for item in allowlist.get("sources", ())}
    publishers = {str(item) for item in allowlist.get("publishers", ())}
    source = str(entry.get("source", "") or "")
    publisher = str(entry.get("publisher", "") or "")
    problems = []
    if source not in sources:
        problems.append(f"source {source or '<none>'} is not on allowlist.sources")
    if publisher not in publishers:
        problems.append(f"publisher {publisher or '<none>'} is not on allowlist.publishers")
    if problems:
        return CheckResult("VG-01", "source allowlist", "REFUSED", _NO_HITS, "; ".join(problems))
    return CheckResult(
        "VG-01", "source allowlist", "PASS", _NO_HITS,
        f"source {source} and publisher {publisher} are both on the allowlist",
    )


def check_pins(entry: Mapping, files: Mapping[str, bytes]) -> CheckResult:
    """VG-02: 40 lowercase hex commit, and the pinned per-file sha256 set exactly equal to the tree."""
    commit = str(entry.get("commit", "") or "")
    problems: list[str] = []
    if not _COMMIT.fullmatch(commit):
        problems.append(f"commit {commit or '<none>'} is not 40 lowercase hex")
    pinned = entry.get("files")
    if not isinstance(pinned, Mapping):
        problems.append("entry.files is missing (no per-file sha256 pins)")
        pinned = {}
    pins = {str(path): str(digest) for path, digest in pinned.items()}
    tree_paths, pin_paths = set(files), set(pins)
    problems += [f"extra path (in the tree, not pinned): {rel}" for rel in sorted(tree_paths - pin_paths)]
    problems += [f"missing path (pinned, not in the tree): {rel}" for rel in sorted(pin_paths - tree_paths)]
    for rel in sorted(tree_paths & pin_paths):
        want, got = pins[rel], file_sha256(files[rel])
        if want != got:
            problems.append(f"sha256 mismatch: {rel} pinned {want} found {got}")
    if problems:
        return CheckResult(
            "VG-02", "pinned commit and per-file sha256", "REFUSED", _NO_HITS,
            f"{len(problems)} problem(s): " + "; ".join(problems),
        )
    return CheckResult(
        "VG-02", "pinned commit and per-file sha256", "PASS", _NO_HITS,
        f"commit {commit} pinned; {len(pins)} file sha256 match the tree exactly",
    )


def check_license(lock: Mapping, entry: Mapping, frontmatter: Mapping) -> CheckResult:
    """VG-03: one SPDX id, agreed between lock and frontmatter, and on the allowlist."""
    allowlist = lock.get("allowlist") or {}
    spdx_ids = {str(item) for item in allowlist.get("spdx", ())}
    fm = frontmatter if isinstance(frontmatter, Mapping) else {}
    entry_id = str(entry.get("license", "") or "").strip()
    fm_id = str(fm.get("license", "") or "").strip()
    if entry_id and fm_id and entry_id != fm_id:
        return CheckResult(
            "VG-03", "SPDX licence", "REFUSED", _NO_HITS,
            f"licence disagreement: lock pins {entry_id!r}, frontmatter claims {fm_id!r}",
        )
    spdx = entry_id or fm_id
    if not spdx:
        return CheckResult(
            "VG-03", "SPDX licence", "UNMEASURED", _NO_HITS,
            "no licence in the lock entry and none in the SKILL.md frontmatter",
        )
    if spdx.upper().startswith("EPL-"):
        return CheckResult("VG-03", "SPDX licence", "REFUSED", _NO_HITS, f"{spdx}: weak copyleft excluded")
    if spdx not in spdx_ids:
        return CheckResult("VG-03", "SPDX licence", "REFUSED", _NO_HITS, f"licence {spdx} is not on allowlist.spdx")
    return CheckResult(
        "VG-03", "SPDX licence", "PASS", _NO_HITS,
        f"licence {spdx} is on the allowlist ({'lock and frontmatter' if entry_id and fm_id else entry_id and 'lock' or 'frontmatter'})",
    )


def check_scripts_policy(entry: Mapping, files: Mapping[str, bytes]) -> CheckResult:
    """VG-10: ``scripts/`` runs only when one signed review covers every scripts path."""
    scripts = sorted(rel for rel in files if rel.startswith("scripts/"))
    if not scripts:
        return CheckResult("VG-10", "scripts policy", "PASS", _NO_HITS, "no scripts/ files in the tree")
    digest = subset_digest(files, scripts)
    review = approved_review(entry, "scripts", digest)
    if review is not None:
        return CheckResult(
            "VG-10", "scripts policy", "PASS", _NO_HITS,
            f"scripts review {_describe(review)} covers all {len(scripts)} scripts/ file(s) "
            f"(files_sha256 {digest[:12]}...) and is approved; never executed by the gate",
        )
    reject = _latest(entry, "scripts", digest, "reject")
    if reject is not None:
        return CheckResult(
            "VG-10", "scripts policy", "REFUSED", _NO_HITS,
            f"scripts review {_describe(reject)} rejected {len(scripts)} scripts/ file(s) (files_sha256 {digest[:12]}...)",
        )
    return CheckResult(
        "VG-10", "scripts policy", "REFUSED", _NO_HITS,
        f"no scripts review matches the current {len(scripts)} scripts/ file(s) (files_sha256 {digest[:12]}...); "
        f"unsigned scripts are never run: {', '.join(scripts)}",
    )
