"""The vetting gate for FoundationSkills (A6.1): fifteen checks over one skill tree.

This module answers one question per skill: can the tree be trusted?  It takes a skill
directory, a ``skills.lock`` (``fskills.skills_lock/1``), the skill name and -- optionally --
an advisory callback, an install target and a project root, runs all fifteen checks
VG-01..VG-15 and returns the ``fskills.vetting_report/1`` report that :func:`write_report`
stores as deterministic JSON.

Checks
------
Every row is one :class:`CheckResult` (see :mod:`foundationskills.vetting.scan`) and is always
present in the report in the fixed order VG-01..VG-15.  Each check is wrapped so that *any*
exception -- including :class:`VettingRefused` from a scanner -- turns that row into
``REFUSED`` with a ``"scanner failure: <ExcType>: <msg>"`` note and also makes VG-15
(``scanner integrity``) REFUSED.  A failure to read the tree, the lock or the frontmatter is
recorded the same way on every dependent row.  The advisory check (VG-11) is the sole
exception to the scanner-failure rule: an advisory failure yields ``UNMEASURED`` and never
fires VG-15.

Statuses and aggregation
------------------------
The four statuses are ``PASS``, ``RED``, ``UNMEASURED`` and ``REFUSED``
(:data:`~foundationskills.vetting.scan.STATUSES`).  The gate result is the worst status
found: any ``REFUSED`` wins, then ``RED``, then ``UNMEASURED``, else ``PASS``.  Exit codes
follow the repo doctrine: ``0``/``5``/``95``/``96``.  Absence of evidence is never PASS.

Installation
------------
Installation is attempted only when ``install_to`` is set and every row is ``PASS``.  Files
are copied byte-for-byte into the target (directories are created; nothing is ever
executed).  If the target already exists with different content the report is amended
``REFUSED``/``96`` and the VG-12 row notes the clash; if the content is identical the
install is an idempotent no-op with ``installed`` set.

Determinism
-----------
The report contains no timestamps and is rendered by :func:`write_report` with
``json.dumps(..., indent=2, sort_keys=True)`` so one skill tree and one lock always give
byte-identical JSON.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from . import lock as _lock_mod
from . import scan as _scan_mod
from .scan import CheckResult

REPORT_SCHEMA = "fskills.vetting_report/1"

_TITLES: dict[str, str] = {
    "VG-01": "source allowlist",
    "VG-02": "pinned commit and per-file sha256",
    "VG-03": "SPDX licence",
    "VG-04": "network primitives",
    "VG-05": "credential paths and env tokens",
    "VG-06": "dynamic execution",
    "VG-07": "hidden HTML comments",
    "VG-08": "invisible Unicode tags and bidi controls",
    "VG-09": "least-privilege allowed-tools",
    "VG-10": "scripts policy",
    "VG-11": "description-vs-body advisory",
    "VG-12": "project-scope install",
    "VG-13": "re-vet on update",
    "VG-14": ".github review",
    "VG-15": "scanner integrity",
}

CHECK_IDS: tuple[str, ...] = tuple(_TITLES.keys())

_EXIT_CODES: dict[str, int] = {
    "PASS": 0,
    "RED": 5,
    "UNMEASURED": 95,
    "REFUSED": 96,
}

_GLOBAL_SCOPES: tuple[str, ...] = (".claude", ".agents", ".config", ".codex")


def _failure_note(exc: Exception) -> str:
    """A human-readable note naming the exception that blocked a step."""
    return f"scanner failure: {type(exc).__name__}: {exc}"


def run_gate(
    skill_dir: Path,
    lock_path: Path,
    name: str,
    *,
    advisory: Callable[[str, str], dict[str, Any]] | None = None,
    install_to: Path | None = None,
    project_root: Path | None = None,
) -> dict[str, Any]:
    """Run every vetting check over one skill tree and return the report dict.

    Each of the fifteen rows VG-01..VG-15 is always present in ``report["checks"]``.
    The advisory (VG-11) is optional; when it is ``None`` the row is ``UNMEASURED``.
    Installation is attempted only when *install_to* is supplied and every row
    passes on the first evaluation.
    """
    scanner_failed = [False]

    # ---- pre-read: skill tree ------------------------------------------------
    files: dict[str, bytes] | None = None
    tree_err: Exception | None = None
    try:
        files = _lock_mod.read_tree(skill_dir)
    except Exception as exc:
        tree_err = exc
        scanner_failed[0] = True

    # ---- pre-read: lock and entry --------------------------------------------
    lock_data: Any = None
    entry: dict[str, Any] | None = None
    lock_err: Exception | None = None
    try:
        lock_data = _lock_mod.load_lock(lock_path)
        entry = _lock_mod.lock_entry(lock_data, name)
    except Exception as exc:
        lock_err = exc
        scanner_failed[0] = True

    # ---- pre-read: frontmatter ------------------------------------------------
    frontmatter: dict[str, Any] | None = None
    fm_err: Exception | None = None
    skill_text = ""
    try:
        if files is None:
            raise tree_err if tree_err else RuntimeError("tree not readable")
        skill_md_bytes = files.get("SKILL.md", b"")
        frontmatter = _scan_mod.parse_frontmatter(skill_md_bytes)
        skill_text = skill_md_bytes.decode("utf-8")
    except Exception as exc:
        fm_err = exc
        scanner_failed[0] = True

    if files is not None and tree_err is None:
        digest = _lock_mod.tree_digest(files)
    else:
        digest = "0" * 64

    # ---- helper to run one check with dependency & exception wrapping --------
    def _guard(
        cid: str,
        fn: Callable[[], CheckResult],
        *dep_errs: Exception | None,
    ) -> CheckResult:
        for err in dep_errs:
            if err is not None:
                return CheckResult(
                    id=cid,
                    title=_TITLES[cid],
                    status="REFUSED",
                    note=_failure_note(err),
                )
        try:
            return fn()
        except Exception as exc:
            scanner_failed[0] = True
            return CheckResult(
                id=cid,
                title=_TITLES[cid],
                status="REFUSED",
                note=_failure_note(exc),
            )

    rows: list[CheckResult] = []

    # VG-01 source allowlist
    rows.append(_guard("VG-01", lambda: _lock_mod.check_source(lock_data, entry), lock_err))

    # VG-02 pinned commit and per-file sha256
    rows.append(_guard("VG-02", lambda: _lock_mod.check_pins(entry, files), tree_err, lock_err))

    # VG-03 SPDX licence
    rows.append(
        _guard("VG-03", lambda: _lock_mod.check_license(lock_data, entry, frontmatter), lock_err, fm_err)
    )

    # VG-04 network primitives
    rows.append(_guard("VG-04", lambda: _scan_mod.check_network(entry, files), tree_err, lock_err))

    # VG-05 credential paths and env tokens
    rows.append(_guard("VG-05", lambda: _scan_mod.check_credentials(files), tree_err))

    # VG-06 dynamic execution
    rows.append(_guard("VG-06", lambda: _scan_mod.check_dynamic_exec(files), tree_err))

    # VG-07 hidden HTML comments
    rows.append(_guard("VG-07", lambda: _scan_mod.check_html_comments(files), tree_err))

    # VG-08 invisible Unicode tags and bidi controls
    rows.append(_guard("VG-08", lambda: _scan_mod.check_invisible_unicode(files), tree_err))

    # VG-09 least-privilege allowed-tools
    rows.append(
        _guard("VG-09", lambda: _scan_mod.check_allowed_tools(entry, frontmatter), lock_err, fm_err)
    )

    # VG-10 scripts policy
    rows.append(_guard("VG-10", lambda: _lock_mod.check_scripts_policy(entry, files), tree_err, lock_err))

    # ---- VG-11 description-vs-body advisory (special handling) ---------------
    if fm_err is not None or tree_err is not None:
        block_err = tree_err if tree_err else fm_err
        rows.append(
            CheckResult(
                id="VG-11",
                title=_TITLES["VG-11"],
                status="REFUSED",
                note=_failure_note(block_err),  # type: ignore[arg-type]
            )
        )
    else:
        try:
            if advisory is None:
                rows.append(
                    CheckResult(
                        id="VG-11",
                        title=_TITLES["VG-11"],
                        status="UNMEASURED",
                        note="no advisory backend: description-vs-body consistency unmeasured",
                    )
                )
            else:
                description = str(frontmatter.get("description", "")) if frontmatter else ""
                result = advisory(description, skill_text)
                if not isinstance(result, dict) or "consistent" not in result:
                    raise ValueError("malformed advisory response")
                consistent = bool(result["consistent"])
                advisory_reason = str(result.get("reason", ""))
                if consistent:
                    rows.append(
                        CheckResult(
                            id="VG-11",
                            title=_TITLES["VG-11"],
                            status="PASS",
                            note=advisory_reason,
                        )
                    )
                else:
                    rev: dict[str, Any] | None = None
                    if entry is not None and files is not None:
                        div_digest = _lock_mod.subset_digest(files, ["SKILL.md"])
                        rev = _lock_mod.review_decision(entry, "divergence", div_digest)
                    if rev and rev.get("decision") == "reject":
                        rows.append(
                            CheckResult(
                                id="VG-11",
                                title=_TITLES["VG-11"],
                                status="RED",
                                note=advisory_reason,
                            )
                        )
                    elif rev and rev.get("decision") == "approve":
                        rows.append(
                            CheckResult(
                                id="VG-11",
                                title=_TITLES["VG-11"],
                                status="PASS",
                                note=f"human cleared divergence: {advisory_reason}",
                            )
                        )
                    else:
                        rows.append(
                            CheckResult(
                                id="VG-11",
                                title=_TITLES["VG-11"],
                                status="UNMEASURED",
                                note=f"held for human review: {advisory_reason}",
                            )
                        )
        except Exception as exc:
            rows.append(
                CheckResult(
                    id="VG-11",
                    title=_TITLES["VG-11"],
                    status="UNMEASURED",
                    note=f"advisory failure: {type(exc).__name__}: {exc}",
                )
            )

    # ---- VG-12 project-scope install -----------------------------------------

    def _check_vg12() -> CheckResult:
        if install_to is not None and project_root is None:
            return CheckResult(
                id="VG-12",
                title=_TITLES["VG-12"],
                status="REFUSED",
                note="project_root is None while install_to is given",
            )
        if install_to is not None:
            target = Path(install_to)
        else:
            pr_base = Path(project_root) if project_root is not None else Path.cwd()
            target = pr_base / ".agents" / "skills" / name
        resolved_target = target.resolve()
        pr_check = Path(project_root) if project_root is not None else Path.cwd()
        resolved_pr = pr_check.resolve()
        home_scope_dirs = [(Path.home() / scope).resolve() for scope in _GLOBAL_SCOPES]
        if not resolved_target.is_relative_to(resolved_pr):
            return CheckResult(
                id="VG-12",
                title=_TITLES["VG-12"],
                status="REFUSED",
                note="target is not inside project_root",
            )
        elif any(resolved_target.is_relative_to(h) for h in home_scope_dirs):
            return CheckResult(
                id="VG-12",
                title=_TITLES["VG-12"],
                status="REFUSED",
                note="target is inside a user/global scope",
            )
        else:
            return CheckResult(
                id="VG-12",
                title=_TITLES["VG-12"],
                status="PASS",
                note="install target is project-scoped",
            )

    rows.append(_guard("VG-12", _check_vg12))

    # ---- VG-13 re-vet on update -----------------------------------------------
    rows.append(
        _guard(
            "VG-13",
            lambda: CheckResult(
                id="VG-13",
                title=_TITLES["VG-13"],
                status="PASS",
                note=f"checks 1-12 re-ran against tree {digest[:12]}",
            ),
            tree_err,
        )
    )

    # VG-14 .github review
    rows.append(_guard("VG-14", lambda: _scan_mod.check_github(entry, files), tree_err, lock_err))

    # ---- VG-15 scanner integrity ----------------------------------------------
    if scanner_failed[0]:
        rows.append(
            CheckResult(
                id="VG-15",
                title=_TITLES["VG-15"],
                status="REFUSED",
                note="scanner failure: one or more checks raised",
            )
        )
    else:
        rows.append(
            CheckResult(
                id="VG-15",
                title=_TITLES["VG-15"],
                status="PASS",
                note="all checks ran without exception",
            )
        )

    # ---- aggregation ---------------------------------------------------------
    statuses = {r.status for r in rows}
    if "REFUSED" in statuses:
        gate_status = "REFUSED"
    elif "RED" in statuses:
        gate_status = "RED"
    elif "UNMEASURED" in statuses:
        gate_status = "UNMEASURED"
    else:
        gate_status = "PASS"
    exit_code = _EXIT_CODES[gate_status]

    # ---- report ---------------------------------------------------------------
    report: dict[str, Any] = {
        "schema": REPORT_SCHEMA,
        "name": name,
        "skill_dir": str(skill_dir),
        "tree_sha256": digest,
        "lock_entry": entry,
        "checks": [r.to_dict() for r in rows],
        "status": gate_status,
        "exit_code": exit_code,
        "installed": None,
    }

    # ---- install (only when everything is PASS and VG-12 is PASS) --------------
    vg12_idx = 11  # rows[11] is VG-12
    can_install = install_to is not None and gate_status == "PASS" and rows[vg12_idx].status == "PASS"

    if can_install and files is not None:
        target = Path(install_to)  # type: ignore[arg-type]
        if target.exists():
            try:
                existing = _lock_mod.read_tree(target)
                existing_digest = _lock_mod.tree_digest(existing)
            except Exception:
                existing_digest = ""
            if existing_digest != digest:
                rows[vg12_idx] = CheckResult(
                    id="VG-12",
                    title=_TITLES["VG-12"],
                    status="REFUSED",
                    note=rows[vg12_idx].note + "; target exists with different content",
                )
                report["status"] = "REFUSED"
                report["exit_code"] = 96
                report["checks"] = [r.to_dict() for r in rows]
            else:
                report["installed"] = str(target)
        else:
            target.mkdir(parents=True, exist_ok=True)
            for rel_path in sorted(files):
                dest = target / rel_path
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(files[rel_path])
            report["installed"] = str(target)

    return report


def write_report(report: dict[str, Any], path: Path) -> None:
    """Write *report* as deterministic JSON to *path* (atomic tmp-then-replace)."""
    data = json.dumps(report, indent=2, sort_keys=True) + "\n"
    tmp = path.parent / f"{path.name}.tmp"
    tmp.write_text(data, encoding="utf-8")
    tmp.replace(path)
