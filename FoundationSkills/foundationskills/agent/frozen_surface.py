from __future__ import annotations
"""C9 increment 1: what a self-improvement candidate may NOT change.
Declares the frozen surface of a skill document as comparable plain data.
Standard library only; consumed by the admission gate in module B."""

import hashlib
import json
import re

FROZEN_FRONTMATTER = ("name", "license", "compatibility", "allowed-tools")
USAGE_BEGIN, USAGE_END, FENCE = "<!-- FS-USAGE:BEGIN", "<!-- FS-USAGE:END -->", "`" * 3
RULE_ID = re.compile(r"\b[A-Z]{2,4}-[A-Z0-9]+(?:-[A-Z0-9]+)*-\d{3}\b|\bCORE-[A-Z0-9-]+\b")
EXIT_CODE = re.compile(r"\b(?:0|5|95|96|exit(?: code)? (?:0|5|95|96))\b")
URL, _INV = re.compile(r"https?://\S+"), re.compile(r"^(?:metadata\.|\s+)invocation\s*:")


def split_frontmatter(text: str) -> tuple[str, str]:
    """Split a skill document into (frontmatter, body) decoded from text.
    Frontmatter sits between the leading delimiter lines.
    Returns ("", text) when no leading frontmatter block is present."""
    lines = text.split("\n")
    if not lines or lines[0].strip() != "---":
        return "", text
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            return "\n".join(lines[1:i]), "\n".join(lines[i + 1:])
    return "", text


def _key_lines(front: str, key: str) -> list[str]:
    """Return the raw value lines of one frontmatter key, if present."""
    lines = front.split("\n")
    pat = _INV if key == "metadata.invocation" else re.compile(r"^" + re.escape(key) + r":")
    for i, ln in enumerate(lines):
        if not pat.match(ln):
            continue
        out, ind = [ln.rstrip()], len(ln) - len(ln.lstrip())
        for nxt in lines[i + 1:]:
            if not nxt.strip() or len(nxt) - len(nxt.lstrip()) <= ind:
                break
            out.append(nxt.rstrip())
        return out
    return []


def _usage_block(body: str) -> str | None:
    """Return the generated usage block text, or None when absent."""
    i = body.find(USAGE_BEGIN)
    if i < 0:
        return None
    j = body.find(USAGE_END, i)
    return body[i: j + len(USAGE_END)] if j >= 0 else body[i:]


def _code_blocks(body: str) -> list[str]:
    """Collect fenced code blocks verbatim, in order of appearance."""
    blocks: list[str] = []
    cur: list[str] = []
    opened = False
    for ln in body.split("\n"):
        if not ln.lstrip().startswith(FENCE):
            if opened:
                cur.append(ln)
        elif opened:
            cur.append(ln)
            blocks.append("\n".join(cur))
            cur, opened = [], False
        else:
            cur.append(ln)
            opened = True
    return blocks


def frozen_parts(text: str) -> dict:
    """Return every frozen region of a skill document as plain data.
    Covers frozen frontmatter, the generated usage block, code blocks,
    rule ids, exit-code text and URLs for byte-for-byte comparison."""
    front, body = split_frontmatter(text)
    fm = {}
    for key in FROZEN_FRONTMATTER + ("metadata.invocation",):
        got = _key_lines(front, key)
        if got:
            fm[key] = "\n".join(got)
    return {
        "frontmatter": fm,
        "usage_block": _usage_block(body),
        "code_blocks": _code_blocks(body),
        "rule_ids": sorted(set(RULE_ID.findall(body))),
        "exit_codes": sorted(set(EXIT_CODE.findall(body))),
        "urls": sorted(set(URL.findall(body))),
    }


def fingerprint(text: str) -> str:
    """Return the sha256 hex digest of the canonical frozen-surface payload.
    Identical frozen parts always share one fingerprint across runs and dicts."""
    data = json.dumps(frozen_parts(text), sort_keys=True)
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def _delta(label: str, old: set, new: set) -> str:
    """Describe a set difference as one human-readable change message."""
    bits = ["+" + ",".join(sorted(new - old)) if new - old else "",
            "-" + ",".join(sorted(old - new)) if old - new else ""]
    return f"{label} changed: " + " ".join(x for x in bits if x)


def assert_editable(old: str, new: str) -> list[str]:
    """Compare two revisions and list violations of the frozen surface.
    Returns an empty list when every frozen part is identical.
    Never raises; one problem string is returned per differing part."""
    problems: list[str] = []
    try:
        a = frozen_parts(old)
        b = frozen_parts(new)
    except Exception:
        return ["frozen surface comparison failed"]
    for key in FROZEN_FRONTMATTER + ("metadata.invocation",):
        if a["frontmatter"].get(key) != b["frontmatter"].get(key):
            problems.append("frozen frontmatter key changed: " + key)
    if a["usage_block"] != b["usage_block"]:
        problems.append("generated usage block changed")
    ca, cb = a["code_blocks"], b["code_blocks"]
    for i in range(max(len(ca), len(cb))):
        if ca[i:i + 1] != cb[i:i + 1]:
            problems.append(f"code block {i + 1} changed")
    if set(a["rule_ids"]) != set(b["rule_ids"]):
        problems.append(_delta("rule ids", set(a["rule_ids"]), set(b["rule_ids"])))
    if a["exit_codes"] != b["exit_codes"]:
        problems.append("exit-code text changed")
    if set(a["urls"]) != set(b["urls"]):
        problems.append(_delta("urls", set(a["urls"]), set(b["urls"])))
    return problems
