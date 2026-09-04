"""Item 2 -- template audit (CONTRACT-BOUND)."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any

from .._base import (
    Coverage,
    Verdict,
)
from .._core import (
    CheckResult,
    _finalize,
)

# ---------------------------------------------------------------------------
# Item 2 — template audit (CONTRACT-BOUND)
# ---------------------------------------------------------------------------


def _check_template_audit(cfg, s, env, shared, registry=None) -> CheckResult:
    """Item 2 (CPU): the chat template must keep CoT inside the masked span.

    INPUT CONTRACT (unverified against the FoxBrain repo — see 'What this fix
    does NOT close'):
      * template.probe_command: argv; the substrings '{file}' and '{rows}' are
        substituted. Ran once per declared file via subprocess. It must print
        EXACTLY rows_per_file JSON lines, each:
            {"row": int, "tokens_stock": int, "tokens_patched": int,
             "cot_span": [start, end], "masked_span": [start, end]}
        spans are token-index half-open ranges into that row's encoding.
      * template.files: the corpus JSONL files probed (8 rows each in the
        design; rows_per_file makes the count explicit).
      * env[template.keep_cot_env] (design: FOXBRAIN_GEMMA4_KEEP_COT): must be
        set and != "0".
      * template.chat_template_path: chat_template.jinja; md5 always recorded,
        compared when template.chat_template_md5 is pinned.

    Assertions per design: masked_span ⊇ cot_span for EVERY row examined;
    stock-vs-patched token-count diff computed for every row (both numbers
    required — a row missing a variant means the diff was never measured over
    it); row count exactly rows_per_file × len(files) (a probe that returns
    fewer rows than asked produced an under-denominator audit); KEEP_COT pin.
    """
    cov_unit = "rows"
    expected = int(s["rows_per_file"]) * len(s["files"])
    evidence: dict[str, Any] = {"files": [], "env_var": s["keep_cot_env"]}

    keep = env.get(s["keep_cot_env"])
    if keep is None:
        return _finalize(
            "template_audit",
            "Template audit",
            Verdict.ERROR,
            Coverage.none(cov_unit),
            f"environment variable {s['keep_cot_env']} is not set — "
            f"the design asserts it ≠ 0; absent is not ≠ 0",
            evidence,
        )
    evidence["keep_cot_value"] = keep
    if keep == "0":
        return _finalize(
            "template_audit",
            "Template audit",
            Verdict.FAIL,
            Coverage.none(cov_unit),
            f"{s['keep_cot_env']}=0: CoT is being dropped from supervision",
            evidence,
        )

    tpl = Path(s["chat_template_path"])
    try:
        md5 = hashlib.md5(tpl.read_bytes()).hexdigest()
    except OSError as exc:
        return _finalize(
            "template_audit",
            "Template audit",
            Verdict.ERROR,
            Coverage.none(cov_unit),
            f"chat template unreadable: {tpl}: {exc}",
            evidence,
        )
    evidence["chat_template"] = {
        "path": str(tpl),
        "md5": md5,
        "pinned_md5": s.get("chat_template_md5") or None,
    }
    if s.get("chat_template_md5") and md5 != s["chat_template_md5"]:
        return _finalize(
            "template_audit",
            "Template audit",
            Verdict.FAIL,
            Coverage.none(cov_unit),
            f"chat_template.jinja md5 {md5} != pinned {s['chat_template_md5']} "
            f"— the template under test is not the template that was reviewed",
            evidence,
        )

    checked = 0
    containment_violations: list[str] = []
    rows_with_diff = 0
    for file_path in s["files"]:
        argv = [
            part.replace("{file}", str(file_path)).replace("{rows}", str(s["rows_per_file"]))
            for part in s["probe_command"]
        ]
        try:
            proc = subprocess.run(
                argv, capture_output=True, text=True, timeout=300, env={**os.environ, **dict(env)}
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            return _finalize(
                "template_audit",
                "Template audit",
                Verdict.ERROR,
                Coverage(checked, cov_unit, expected=expected),
                f"template probe could not run ({exc}); the audit was not performed",
                evidence,
            )
        if proc.returncode != 0:
            return _finalize(
                "template_audit",
                "Template audit",
                Verdict.ERROR,
                Coverage(checked, cov_unit, expected=expected),
                f"template probe exited {proc.returncode}: {proc.stderr.strip()[:200]}",
                evidence,
            )
        rows = []
        for line_no, line in enumerate(proc.stdout.splitlines()):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                return _finalize(
                    "template_audit",
                    "Template audit",
                    Verdict.ERROR,
                    Coverage(checked, cov_unit, expected=expected),
                    f"probe row {line_no} of {file_path} is not JSON: {exc}",
                    evidence,
                )
        file_rec = {"path": file_path, "rows_returned": len(rows)}
        if len(rows) != int(s["rows_per_file"]):
            return _finalize(
                "template_audit",
                "Template audit",
                Verdict.FAIL,
                Coverage(checked, cov_unit, expected=expected),
                f"probe returned {len(rows)} rows for {file_path}; "
                f"{s['rows_per_file']} were asked for — a short audit is an under-covered claim",
                {**evidence, "files": evidence["files"] + [file_rec]},
            )
        for row in rows:
            try:
                m_lo, m_hi = row["masked_span"]
                c_lo, c_hi = row["cot_span"]
                t_stock = int(row["tokens_stock"])
                t_patched = int(row["tokens_patched"])
            except (KeyError, TypeError, ValueError) as exc:
                return _finalize(
                    "template_audit",
                    "Template audit",
                    Verdict.ERROR,
                    Coverage(checked, cov_unit, expected=expected),
                    f"probe row missing contract fields ({exc!r}) in {file_path}",
                    evidence,
                )
            checked += 1
            if t_stock != t_patched:
                rows_with_diff += 1
            if not (m_lo <= c_lo and c_hi <= m_hi):
                containment_violations.append(
                    f"{file_path} row {row.get('row', '?')}: cot_span "
                    f"[{c_lo},{c_hi}) escapes masked_span [{m_lo},{m_hi})"
                )
        evidence["files"].append(file_rec)

    evidence["rows_with_stock_vs_patched_diff"] = rows_with_diff
    evidence["rows_examined"] = checked
    cov = Coverage(checked, cov_unit, expected=expected)
    if containment_violations:
        return _finalize(
            "template_audit",
            "Template audit",
            Verdict.FAIL,
            cov,
            "; ".join(containment_violations[:3])
            + (
                f" (+{len(containment_violations) - 3} more)"
                if len(containment_violations) > 3
                else ""
            ),
            evidence,
        )
    return _finalize(
        "template_audit",
        "Template audit",
        Verdict.PASS,
        cov,
        f"CoT span inside masked span for {checked}/{expected} rows; "
        f"KEEP_COT={keep}; {rows_with_diff}/{checked} rows show a stock-vs-patched token diff",
        evidence,
    )
