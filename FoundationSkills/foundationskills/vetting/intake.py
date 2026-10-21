"""Controlled intake of vetted third-party skills (A6.2): only a skill that PASSes the gate at its pin is vendored.

``intake`` pins every requested skill of one upstream checkout at one commit into ``skills.lock`` (per-file sha256,
source, publisher, SPDX licence from the frontmatter or the repository's root LICENSE) and runs the full gate (``run_gate``, VG-01..VG-15) with the live
advisory. Then:

- **PASS:** the bytes are copied, unchanged, to ``<dest>/<name>/``. The advisory's verdict is recorded in the lock as
  a review of kind ``advisory`` bound to the digest of exactly that ``SKILL.md``.
- **Anything else:** nothing is vendored, and the skill is red-cased in the intake ledger with every failing check.
  Nothing is ever waived.

``revet`` re-runs the gate over a vendored skill offline. The advisory is replayed from the recorded review only while
``SKILL.md`` still has the reviewed digest. One changed byte voids the review (VG-11 UNMEASURED) and the pins (VG-02
REFUSED), so an update is re-vetted, never inherited.

Licence policy (plan decision 8): EPL is excluded (VG-03 refuses EPL-* by itself); CC-BY-4.0 needs an attribution entry
in THIRD_PARTY_NOTICES; ``write_notices`` writes one for every vendored skill, and ``write_ledger`` the intake ledger.
"""

from __future__ import annotations

import datetime as _dt
import json
import shutil
from pathlib import Path
from typing import Any, Callable

from .gate import run_gate
from .lock import LOCK_SCHEMA, file_sha256, read_tree, subset_digest
from .scan import parse_frontmatter

DEFAULT_ALLOWLIST = {
    "sources": ["https://github.com/mattpocock/skills"],
    "publishers": ["mattpocock"],
    "spdx": ["MIT", "Apache-2.0", "BSD-2-Clause", "BSD-3-Clause", "ISC", "0BSD", "CC0-1.0", "CC-BY-4.0"],
}


def _load_or_new(lock_path: Path) -> dict[str, Any]:
    if lock_path.is_file():
        return json.loads(lock_path.read_text(encoding="utf-8"))
    return {"schema": LOCK_SCHEMA, "allowlist": DEFAULT_ALLOWLIST, "skills": {}}


def _write(lock_path: Path, lock: dict[str, Any]) -> None:
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n", encoding="utf-8")


# The first line of a repository LICENSE file -> SPDX id. Only these exact headings count; anything else stays unknown.
_LICENSE_HEADINGS = {"MIT License": "MIT", "Apache License": "Apache-2.0", "ISC License": "ISC",
                     "BSD 3-Clause License": "BSD-3-Clause", "BSD 2-Clause License": "BSD-2-Clause"}


def repo_license(repo_root: Path) -> tuple[str, str]:
    """(SPDX id, evidence) from the repository's root LICENSE heading; ("", reason) when it cannot be identified."""
    for name in ("LICENSE", "LICENSE.md", "LICENSE.txt"):
        path = repo_root / name
        if path.is_file():
            first = next((line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()), "")
            for heading, spdx in _LICENSE_HEADINGS.items():
                if first.startswith(heading):
                    return spdx, f"{name} (repository root) heading: {first!r}"
            return "", f"{name} heading {first!r} is not a recognised licence heading"
    return "", "no LICENSE file at the repository root"


def pin_entry(skill_dir: Path, relpath: str, commit: str, source: str, publisher: str,
              repo_root: Path | None = None) -> dict[str, Any]:
    """The lock entry for one upstream skill tree at ``commit``.

    The licence comes from the SKILL.md frontmatter, else from the repository's root LICENSE heading (the evidence is
    recorded beside it). It is never guessed: an unidentified licence stays empty and VG-03 reports it.
    """
    files = read_tree(skill_dir)
    frontmatter = parse_frontmatter(files.get("SKILL.md", b"")) if "SKILL.md" in files else {}
    license_id = str(frontmatter.get("license") or "")
    evidence = "SKILL.md frontmatter" if license_id else ""
    if not license_id and repo_root is not None:
        license_id, evidence = repo_license(repo_root)
    return {"source": source, "publisher": publisher, "path": relpath, "commit": commit,
            "license": license_id, "license_evidence": evidence,
            "files": {rel: file_sha256(data) for rel, data in sorted(files.items())}, "reviews": []}


def recorded_advisory(entry: dict[str, Any], files: dict[str, bytes]) -> Callable[[str, str], dict[str, Any]]:
    """An advisory that replays the recorded verdict for exactly these SKILL.md bytes, and fails otherwise."""
    digest = subset_digest(files, ["SKILL.md"])

    def advisory(description: str, body: str) -> dict[str, Any]:  # noqa: ARG001 - the gate's callback signature
        for review in reversed(entry.get("reviews") or []):
            if review.get("kind") == "advisory" and review.get("files_sha256") == digest:
                return {"consistent": review.get("decision") == "approve",
                        "reason": f"recorded advisory ({review.get('reviewer')}, {review.get('date')}): "
                                  f"{review.get('reason', '')}"}
        raise RuntimeError("no recorded advisory for these SKILL.md bytes: re-vet with a live advisory")

    return advisory


def intake(source_dir: Path, commit: str, skills: list[str], lock_path: Path, dest: Path,
           advisory: Callable[[str, str], dict[str, Any]], reviewer: str,
           source: str = DEFAULT_ALLOWLIST["sources"][0], publisher: str = DEFAULT_ALLOWLIST["publishers"][0]
           ) -> list[dict[str, Any]]:
    """Pin, vet and (on PASS only) vendor each skill; return one ledger row per skill."""
    lock = _load_or_new(lock_path)
    rows = []
    today = _dt.date.today().isoformat()
    for relpath in skills:
        skill_dir = source_dir / relpath
        name = Path(relpath).name
        entry = pin_entry(skill_dir, relpath, commit, source, publisher, repo_root=source_dir)
        lock["skills"][name] = entry
        _write(lock_path, lock)
        verdicts: dict[str, Any] = {}

        def capture(description: str, body: str) -> dict[str, Any]:
            verdicts["advisory"] = advisory(description, body)
            return verdicts["advisory"]

        report = run_gate(skill_dir, lock_path, name, advisory=capture)
        status = str(report.get("verdict") or report.get("status"))
        failing = [f"{c['id']} {c['status']}: {c.get('note', '')[:160]}" for c in report["checks"]
                   if c["status"] != "PASS"]
        if status == "PASS":
            target = dest / name
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(skill_dir, target)
            verdict = verdicts.get("advisory") or {}
            entry["reviews"].append({"kind": "advisory", "decision": "approve",
                                     "files_sha256": subset_digest(read_tree(skill_dir), ["SKILL.md"]),
                                     "reviewer": reviewer, "date": today, "reason": str(verdict.get("reason", ""))})
            _write(lock_path, lock)
        else:
            del lock["skills"][name]  # never pin what was not vendored
            _write(lock_path, lock)
        rows.append({"skill": name, "path": relpath, "commit": commit, "license": entry["license"], "verdict": status,
                     "vendored": status == "PASS", "failing": failing})
    return rows


def revet(vendored_dir: Path, lock_path: Path, name: str) -> dict[str, Any]:
    """Re-run the gate over a vendored skill offline, replaying only the advisory recorded for these exact bytes."""
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    entry = lock["skills"][name]
    return run_gate(vendored_dir, lock_path, name, advisory=recorded_advisory(entry, read_tree(vendored_dir)))


def write_ledger(rows: list[dict[str, Any]], path: Path, source: str = DEFAULT_ALLOWLIST["sources"][0]) -> None:
    """The intake ledger: one row per requested skill; every non-PASS skill red-cased with each failing check."""
    lines = ["# Third-party skill intake ledger (A6.2)", "",
             f"Source `{source}`. Only a skill that PASSes the gate (VG-01..VG-15) at its pin is vendored. "
             "Failing checks are red-cased here and never waived.", "",
             "| skill | upstream path | commit | licence | verdict | vendored |", "|---|---|---|---|---|---|"]
    for row in rows:
        lines.append(f"| {row['skill']} | `{row['path']}` | `{row['commit'][:12]}` | {row['license'] or '(none)'} "
                     f"| {row['verdict']} | {'yes' if row['vendored'] else '**no**'} |")
    red = [row for row in rows if not row["vendored"]]
    if red:
        lines += ["", "## Red-cased (not vendored)", ""]
        for row in red:
            lines.append(f"- **{row['skill']}** ({row['verdict']}):")
            lines += [f"  - {failure}" for failure in row["failing"]]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_notices(lock_path: Path, path: Path, holders: dict[str, str]) -> None:
    """THIRD_PARTY_NOTICES: one attribution per vendored skill (``holders`` maps a source URL to its copyright line)."""
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    lines = ["Third-party skills vendored under third_party/skills/ (see skills.lock for the pinned bytes).", ""]
    for name, entry in sorted(lock["skills"].items()):
        lines += [f"{name}", f"  Source:    {entry['source']} ({entry['path']} @ {entry['commit']})",
                  f"  Licence:   {entry['license']} ({entry.get('license_evidence') or 'lock entry'})",
                  f"  Copyright: {holders.get(entry['source'], '(not recorded)')}", ""]
    path.write_text("\n".join(lines), encoding="utf-8")
