"""Static scans for the FoundationSkills vetting gate (A6.1): what is in the tree, line by line.

The gate (:mod:`foundationskills.vetting.gate`) runs one fixed battery of checks over one skill
tree -- the same ``files: dict[str, bytes]`` (relative posix path -> raw bytes) every part of the
package passes around. This module owns the scans: the fixed pattern lists, the YAML frontmatter
reader, the UTF-8 text walker and the checks that reduce a tree to :class:`Hit` findings (VG-04 to
VG-09, and the review-backed VG-14). Tree, pin, digest and review state belongs to
:mod:`foundationskills.vetting.lock`; the two checks that must consult a human review import from it
lazily, so this file stays a scanner that runs on its own.

Statuses and refusals
---------------------
A :class:`CheckResult` reports exactly one of :data:`STATUSES` -- ``PASS``, ``RED``, ``UNMEASURED``
or ``REFUSED`` -- and absence of evidence is never ``PASS``. ``PASS`` is what a check returns when it
ran and found nothing, including when the attack surface is absent (no ``scripts/``, no
``.github/``); when a current human review covers the evidence, the check passes and names that
review in its note; ``RED`` is a finding that blocks; ``REFUSED`` is a gate that will not decide --
a reject review, or bytes it cannot vet. :class:`VettingRefused`
(from :mod:`foundationskills.vetting.errors`) is raised when the input itself is unusable: a
scanner-suffix file that is not UTF-8, frontmatter that is absent, invalid or not a mapping.

Findings
--------
Every hit names ``path:line`` (1-based), the pattern that matched, and at most 120 characters of the
line it matched on. Invisible characters are escaped in excerpts as ``\\uXXXX`` / ``\\U000XXXXX`` (a
backslash, ``u``, four lowercase hex digits -- eight after ``U``, for the tag range at U+E0000), so
an excerpt is always printable. Patterns are word-bounded about their stem and literal about their
suffix: the scans mean ``eval(`` and never the prose words ``fetches``, ``request``, ``evaluation``,
``executive`` or ``environment``. Scans are deterministic -- sorted paths, sorted hits -- and every
result renders itself with :meth:`CheckResult.to_dict`.
"""
from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import yaml

from .errors import VettingRefused

STATUSES: tuple[str, ...] = ("PASS", "RED", "UNMEASURED", "REFUSED")
CHECK_TITLES: dict[str, str] = {
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
EXCERPT_LIMIT = 120
FRONTMATTER_DELIM = "---"
FRONTMATTER_PATH = "SKILL.md"
MARKDOWN_SUFFIX = ".md"
SCANNABLE_SUFFIXES = (".md", ".py", ".sh", ".js", ".ts")
SCRIPTS_PREFIX = "scripts/"
GITHUB_PREFIX = ".github/"
WILDCARD_TOOL = "*"
BARE_BASH = "Bash"
TOOLS_KEY = "allowed-tools"
NEED_KEY = "allowed_tools_need"

NETWORK_PATTERNS: tuple[str, ...] = (
    r"\bcurl\b",
    r"\bwget\b",
    r"\brequests\.(?:get|post|put|patch|delete|head|options|request|Session)\b",
    r"(?m)^\s*(?:import|from)\s+requests\b",
    r"\burllib\b",
    r"\bfetch\(",
    r"\bhttp\.client\b",
    r"\bsocket\.",
    r"\bnc\b",
    r"\bInvoke-WebRequest\b",
)
CREDENTIAL_PATTERNS: tuple[str, ...] = (
    r"~/\.ssh\b",
    r"~/\.aws\b",
    r"~/\.codex\b",
    r"~/\.config\b",
    r"~/\.gnupg\b",
    r"~/\.netrc\b",
    r"\bid_rsa\b",
    r"\bid_ed25519\b",
    r"\bid_ecdsa\b",
    r"\b[A-Za-z][A-Za-z0-9]*_(?:API_KEY|API_KEYS|TOKEN|TOKENS|SECRET|SECRETS)\b",
    r"\bANTHROPIC_BASE_URL\b",
    r"\bOPENAI_BASE_URL\b",
    r"\bos\.environ\b",
    r"\bgetenv\b",
    r"\bprintenv\b",
    r"\benv\s*\|",
)
DYNAMIC_EXEC_PATTERNS: tuple[str, ...] = (
    r"\beval\s*\(",
    r"\bexec\s*\(",
    r"\bcompile\s*\(",
    r"__import__\s*\(",
    r"\bbase64\b[^|\n]*(?:-d|--decode)\b[^|\n]*\|\s*(?:sh|bash|python)\b",
    r"\bb64decode\b.*(?:eval|exec)\s*\(|(?:eval|exec)\s*\(.*\bb64decode\b",
    r"\b(?:curl|wget)\b[^|\n]*\|\s*(?:sh|bash)\b",
    r"\bsubprocess\b[\s\S]{0,200}?shell\s*=\s*True",
)
HTML_COMMENT_OPEN = "<!--"
COMMENT_OPEN_RE = re.compile(re.escape(HTML_COMMENT_OPEN))
COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)
PARTIAL_ESCAPE_RE = re.compile(r"\\[uU][0-9a-fA-F]*$")
INVISIBLE_RANGES: tuple[tuple[int, int], ...] = (
    (0xE0000, 0xE007F),
    (0x202A, 0x202E),
    (0x2066, 0x2069),
    (0x200E, 0x200F),
    (0x061C, 0x061C),
)
INVISIBLE_CHARS: tuple[str, ...] = tuple(
    chr(code) for low, high in INVISIBLE_RANGES for code in range(low, high + 1)
)
INVISIBLE_ESCAPES: dict[str, str] = {
    char: ("\\u%04x" if ord(char) < 0x10000 else "\\U%08x") % ord(char) for char in INVISIBLE_CHARS
}


@dataclass(frozen=True)
class Hit:
    """One finding: ``pattern`` matched at ``path:line`` (1-based) with printable context.

    ``line`` is 0 only for a frontmatter finding (VG-09), whose position the YAML mapping cannot
    resolve; ``excerpt`` is capped at :data:`EXCERPT_LIMIT` with invisible characters escaped.
    """

    path: str
    line: int
    pattern: str
    excerpt: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "excerpt", _excerpt(self.excerpt))


@dataclass(frozen=True)
class CheckResult:
    """One vetting check: fixed ``id``/``title``, a :data:`STATUSES` verdict, hits and a note.

    ``hits`` is empty exactly when the check found nothing or named its evidence in ``note`` instead
    (a missing or rejecting review has no line to point at). :meth:`to_dict` is the wire form.
    """

    id: str
    title: str
    status: str
    hits: tuple[Hit, ...] = ()
    note: str = ""

    def __post_init__(self) -> None:
        if self.status not in STATUSES:
            raise VettingRefused(f"{self.id}: {self.status!r} is not one of {'/'.join(STATUSES)}")

    def to_dict(self) -> dict:
        """The wire form: hits as plain dicts sorted by ``(path, line, pattern)``."""
        return {
            "id": self.id,
            "title": self.title,
            "status": self.status,
            "hits": [
                {"path": hit.path, "line": hit.line, "pattern": hit.pattern, "excerpt": hit.excerpt}
                for hit in sorted(self.hits, key=_sort_key)
            ],
            "note": self.note,
        }


def parse_frontmatter(skill_md: bytes) -> dict:
    """The YAML mapping between the first two ``---`` lines of ``SKILL.md``.

    Refuses a file with no frontmatter or no closing delimiter, YAML that does not parse, and YAML
    that parses to anything but a mapping: a skill that cannot state a licence, a description or its
    tool need may not be read downstream as if it had declared none.
    """
    lines = _utf8(skill_md, FRONTMATTER_PATH).splitlines()
    if not lines or lines[0].strip() != FRONTMATTER_DELIM:
        raise VettingRefused(f"{FRONTMATTER_PATH}: no YAML frontmatter")
    close = next((index for index in range(1, len(lines)) if lines[index].strip() == FRONTMATTER_DELIM), None)
    if close is None:
        raise VettingRefused(f"{FRONTMATTER_PATH}: frontmatter is never closed")
    try:
        document = yaml.safe_load("\n".join(lines[1:close]))
    except yaml.YAMLError as exc:
        raise VettingRefused(f"{FRONTMATTER_PATH}: frontmatter is not valid YAML ({exc})") from exc
    if not isinstance(document, dict):
        raise VettingRefused(f"{FRONTMATTER_PATH}: frontmatter is not a YAML mapping")
    return document


def text_files(files: Mapping[str, bytes]) -> dict[str, str]:
    """Every file of ``files`` that decodes as UTF-8, as text keyed by path (sorted insertion).

    A file that does not decode is skipped unless the scanner must read it -- anything under
    ``scripts/`` or ending in :data:`SCANNABLE_SUFFIXES` -- in which case :class:`VettingRefused`
    names it: an unscanned file the scans cover is a hole in the gate.
    """
    texts: dict[str, str] = {}
    for path in sorted(files):
        try:
            texts[path] = _utf8(files[path], path)
        except VettingRefused:
            if path.startswith(SCRIPTS_PREFIX) or path.endswith(SCANNABLE_SUFFIXES):
                raise
    return texts


def check_network(entry: Mapping, files: Mapping[str, bytes]) -> CheckResult:
    """VG-04: network primitives (:data:`NETWORK_PATTERNS`) anywhere in the tree.

    ``RED`` on any hit, but a file whose hits are covered by a current ``network`` review of exactly
    that file is excused (review and path named in the note); a review that rejects the file refuses
    the check. :mod:`foundationskills.vetting.lock` is imported lazily.
    """
    from .lock import review_decision, subset_digest

    hits = _find_hits(text_files(files), NETWORK_PATTERNS)
    kept: list[Hit] = []
    excused: list[str] = []
    rejected: list[str] = []
    for path in sorted({hit.path for hit in hits}):
        review = review_decision(entry, "network", subset_digest(files, (path,)))
        decision = "none" if review is None else str(review.get("decision", "none"))
        if decision == "approve":
            excused.append(path)
            continue
        kept.extend(hit for hit in hits if hit.path == path)
        if decision == "reject":
            rejected.append(path)
    note = "; ".join(
        part
        for part in (
            f"approved network review excuses {', '.join(excused)}" if excused else "",
            f"network review refused {', '.join(rejected)}" if rejected else "",
            f"{len(kept)} network primitive hit(s)" if kept else "no network primitive found",
        )
        if part
    )
    status = "REFUSED" if rejected else ("RED" if kept else "PASS")
    return _result("VG-04", status, kept, note)


def check_credentials(files: Mapping[str, bytes]) -> CheckResult:
    """VG-05: credential paths and env tokens (:data:`CREDENTIAL_PATTERNS`) anywhere in the tree."""
    hits = _find_hits(text_files(files), CREDENTIAL_PATTERNS)
    return _result(
        "VG-05",
        "RED" if hits else "PASS",
        hits,
        f"{len(hits)} credential path or env token hit(s)" if hits else "no credential path or env token found",
    )


def check_dynamic_exec(files: Mapping[str, bytes]) -> CheckResult:
    """VG-06: dynamic execution (:data:`DYNAMIC_EXEC_PATTERNS`) anywhere in the tree."""
    hits = _find_hits(text_files(files), DYNAMIC_EXEC_PATTERNS)
    return _result(
        "VG-06",
        "RED" if hits else "PASS",
        hits,
        f"{len(hits)} dynamic execution hit(s)" if hits else "no dynamic execution found",
    )


def check_html_comments(files: Mapping[str, bytes]) -> CheckResult:
    """VG-07: a hidden ``<!--`` comment in a markdown file (excerpt: the comment's first 120 chars)."""
    hits: list[Hit] = []
    texts = text_files(files)
    for path in sorted(texts):
        if not path.endswith(MARKDOWN_SUFFIX):
            continue
        text = texts[path]
        for match in COMMENT_OPEN_RE.finditer(text):
            span = COMMENT_RE.search(text, match.start())
            comment = span.group(0) if span is not None else text[match.start() :]
            hits.append(Hit(path, _line_number(text, match.start()), HTML_COMMENT_OPEN, _head(comment)))
    return _result(
        "VG-07",
        "RED" if hits else "PASS",
        hits,
        f"{len(hits)} hidden HTML comment hit(s)" if hits else "no hidden HTML comment found",
    )


def check_invisible_unicode(files: Mapping[str, bytes]) -> CheckResult:
    """VG-08: tag characters and bidi controls (:data:`INVISIBLE_RANGES`) in any text file."""
    hits: list[Hit] = []
    for path, text in sorted(text_files(files).items()):
        for index, char in enumerate(text):
            shape = INVISIBLE_ESCAPES.get(char)
            if shape is None:
                continue
            hits.append(Hit(path, _line_number(text, index), shape, _excerpt(_line_at(text, index))))
    return _result(
        "VG-08",
        "RED" if hits else "PASS",
        hits,
        f"{len(hits)} invisible character hit(s)" if hits else "no invisible character found",
    )


def check_allowed_tools(entry: Mapping, frontmatter: Mapping) -> CheckResult:
    """VG-09: ``allowed-tools`` against the entry's ``allowed_tools_need``.

    No declaration is ``PASS`` (nothing is claimed). Otherwise the tokens -- a space/comma separated
    string or a list -- must all be in the entry's need list: a ``*`` wildcard or a bare unscoped
    ``Bash`` is ``RED`` unless the need list asks for exactly that, and a token outside the need list
    is ``RED``. Findings name ``SKILL.md`` at line 0 (a frontmatter finding has no resolvable line).
    """
    if TOOLS_KEY not in frontmatter:
        return _result("VG-09", "PASS", (), "no allowed-tools declaration to vet")
    raw = frontmatter[TOOLS_KEY]
    tokens = _tool_tokens(raw)
    if not tokens:
        return _result("VG-09", "PASS", (), "allowed-tools declares no tool at all")
    need = [str(item) for item in (entry.get(NEED_KEY) or ())]
    hits = [
        Hit(FRONTMATTER_PATH, 0, token, f"{TOOLS_KEY}: {_tool_text(raw)}")
        for token in tokens
        if token not in need
    ]
    broad = sorted({token for token in tokens if token in (WILDCARD_TOOL, BARE_BASH)})
    note = "; ".join(
        part
        for part in (
            f"{len(hits)} token(s) outside {NEED_KEY}" if hits else "every token is in the declared need",
            f"wildcard or unscoped tool request: {', '.join(broad)}" if broad else "",
            f"{NEED_KEY}: {', '.join(need) if need else 'unstated'}",
        )
        if part
    )
    return _result("VG-09", "RED" if hits else "PASS", hits, note)


def check_github(entry: Mapping, files: Mapping[str, bytes]) -> CheckResult:
    """VG-14: the tree's ``.github/`` policy is human-reviewed or the check refuses.

    No ``.github/`` path is ``PASS`` (nothing to review). Otherwise one current ``github`` review
    must cover every ``.github/`` path at once and decide ``approve``; a reject, or no review at all,
    is ``REFUSED`` with the paths named in the note. :mod:`foundationskills.vetting.lock` is
    imported lazily.
    """
    from .lock import review_decision, subset_digest

    paths = sorted(path for path in files if path.startswith(GITHUB_PREFIX))
    if not paths:
        return _result("VG-14", "PASS", (), "no .github/ policy in the tree")
    review = review_decision(entry, "github", subset_digest(files, paths))
    decision = "none" if review is None else str(review.get("decision", "none"))
    if decision == "approve":
        return _result("VG-14", "PASS", (), f"approved github review covers {', '.join(paths)}")
    if decision == "reject":
        return _result("VG-14", "REFUSED", (), f"github review refused {', '.join(paths)}")
    return _result("VG-14", "REFUSED", (), f"no approved github review covers {', '.join(paths)}")


def _utf8(data: bytes, name: str) -> str:
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise VettingRefused(f"{name}: not valid UTF-8 and cannot be scanned") from exc


def _line_number(text: str, offset: int) -> int:
    return text.count("\n", 0, offset) + 1


def _line_at(text: str, offset: int) -> str:
    start = text.rfind("\n", 0, offset) + 1
    end = text.find("\n", offset)
    return text[start:end] if end >= 0 else text[start:]


def _escape(text: str) -> str:
    return "".join(INVISIBLE_ESCAPES.get(char, char) for char in text)


def _head(text: str) -> str:
    """The first :data:`EXCERPT_LIMIT` chars of ``text``, invisibles escaped, no escape cut in half."""
    return PARTIAL_ESCAPE_RE.sub("", _escape(text)[:EXCERPT_LIMIT])


def _excerpt(text: str) -> str:
    """``text`` with invisibles escaped and capped at 120 chars, elided with ``...`` when cut."""
    escaped = _escape(text)
    if len(escaped) <= EXCERPT_LIMIT:
        return escaped
    return PARTIAL_ESCAPE_RE.sub("", escaped[: EXCERPT_LIMIT - 3]).rstrip() + "..."


def _sort_key(hit: Hit) -> tuple[str, int, str, str]:
    return (hit.path, hit.line, hit.pattern, hit.excerpt)


def _find_hits(texts: Mapping[str, str], patterns: Sequence[str]) -> list[Hit]:
    hits: list[Hit] = []
    for path in sorted(texts):
        text = texts[path]
        for pattern in patterns:
            for match in re.finditer(pattern, text):
                start = match.start()
                hits.append(Hit(path, _line_number(text, start), pattern, _excerpt(_line_at(text, start))))
    return hits


def _result(check_id: str, status: str, hits: Iterable[Hit], note: str) -> CheckResult:
    return CheckResult(check_id, CHECK_TITLES[check_id], status, tuple(sorted(set(hits), key=_sort_key)), note)


def _tool_tokens(raw: Any) -> list[str]:
    """``allowed-tools`` as tokens: a space/comma separated string, or a list of token strings."""
    if isinstance(raw, str):
        parts = [raw]
    elif isinstance(raw, (list, tuple)):
        parts = [str(item) for item in raw]
    else:
        parts = [str(raw)]
    return [token for part in parts for token in re.split(r"[,\s]+", part) if token]


def _tool_text(raw: Any) -> str:
    if isinstance(raw, str):
        return raw
    if isinstance(raw, (list, tuple)):
        return ", ".join(str(item) for item in raw)
    return str(raw)
