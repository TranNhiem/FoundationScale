
"""``clean`` op: text normalization, mojibake repair, HTML stripping, language ID and PII redaction.

Applies to ``text`` and to every message ``content`` / ``prompt`` / ``chosen``
/ ``rejected`` / ``answer`` field. Also exports helpers reused by sibling ops
and the readiness report: :func:`html_to_text`, :func:`pii_scan`,
:func:`primary_text`, :func:`mostly_cjk`.

Defaults where the spec is silent:

- ``normalize_unicode``: NFKC, default True.
- ``fix_mojibake``: default True. Uses ``ftfy`` when importable, else a
  builtin UTF-8-as-latin-1 repair that only accepts the repair when it lowers
  the non-ASCII ratio. The builtin is an approximation; ``stats.extra``
  records which backend ran.
- ``strip_html``: ``"auto"`` (strip when HTML tags are detected), True/False
  to force.
- ``langid``: ``fasttext`` only when ``cfg.langid.model_path`` is set (and
  fasttext is importable); otherwise a builtin Unicode-script + diacritic
  heuristic distinguishing en/zh/ja/ko/vi/unknown with
  ``stats.extra["approximate_langid"] = True``. Filtering is only applied
  when ``langid.allowed`` is a non-empty list.
- ``min_chars``/``max_chars``/the ratio thresholds default to None (filter
  disabled). Empty primary text is always dropped (``empty``).
- ``pii.redact`` default True with builtin regex types
  ``[email, phone, ipv4, credit_card, national_id, ssn]`` (``url`` opt-in).
  ``pii.backend == "presidio"`` uses Presidio when importable, else falls back
  to the builtin regexes with ``stats.extra["approximate_pii"] = True``.
  Phone candidates are validated to 10-15 digits so plain numbers and SSNs are
  not mis-redacted; credit cards require a Luhn check.
"""
from __future__ import annotations

import re
import string
import unicodedata
from html.parser import HTMLParser
from typing import Any, Callable, Iterable, Iterator

from foundationskills.skills.data_engine.ops.base import FunctionOp, OpStats, counted, register_op


CONFIG_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "normalize_unicode": {"type": "boolean"},
        "fix_mojibake": {"type": "boolean"},
        "strip_html": {"enum": ["auto", True, False]},
        "langid": {
            "type": "object",
            "properties": {
                "allowed": {"type": ["array", "null"], "items": {"type": "string"}},
                "min_confidence": {"type": "number"},
                "model_path": {"type": ["string", "null"]},
            },
        },
        "min_chars": {"type": ["integer", "null"]},
        "max_chars": {"type": ["integer", "null"]},
        "max_non_alnum_ratio": {"type": ["number", "null"]},
        "max_line_repeat_ratio": {"type": ["number", "null"]},
        "max_char_repeat_ratio": {"type": ["number", "null"]},
        "pii": {
            "type": "object",
            "properties": {
                "redact": {"type": "boolean"},
                "types": {"type": "array", "items": {"type": "string"}},
                "backend": {"enum": ["builtin", "presidio"]},
            },
        },
    },
}


class CleanError(ValueError):
    """An explicitly requested cleaning backend is unavailable (a refusal)."""


DEFAULT_PII_TYPES = ("email", "phone", "ipv4", "credit_card", "national_id", "ssn")

_PII_TOKENS = {
    "email": "<PII_EMAIL>",
    "phone": "<PII_PHONE>",
    "ipv4": "<PII_IPV4>",
    "credit_card": "<PII_CREDIT_CARD>",
    "national_id": "<PII_NATIONAL_ID>",
    "ssn": "<PII_SSN>",
    "url": "<PII_URL>",
}

_EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}\b")
# Number boundaries matter: real data had binary strings and long decimals
# ("0.0000000000000004648") redacted as cards. A PII number must not sit inside
# a larger number (no adjacent digit, '.', ',', '^'), and IPs are exactly 4 octets.
_IPV4_RE = re.compile(r"(?<![\d.])(?:\d{1,3}\.){3}\d{1,3}(?![\d]|\.\d)")
_CREDIT_RE = re.compile(
    r"(?<![\d.,^-])(?:[2-6]\d{12,18}|[2-6]\d{3}(?:( |-)\d{4})(?:\1\d{4}){1,2}(?:\1\d{1,4})?)(?![\d]|\.\d|-\d)"
)
_NATID_RE = re.compile(r"\b[A-Z][12]\d{8}\b")
_SSN_RE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
_PHONE_RE = re.compile(r"(?<![\w.+-])(?:\+\d{1,3}[\s.-]?)?(?:\(\d{1,4}\)[\s.-]?|\d{1,4}[\s.-]){1,4}\d{3,4}(?![\w-]|\.\d)")
_URL_RE = re.compile(r"\b(?:https?://|www\.)\S+", re.IGNORECASE)

# redaction order matters: credit cards before phones, SSN before phones
_PII_ORDER = ("email", "url", "credit_card", "ssn", "national_id", "ipv4", "phone")


def _ipv4_ok(match: re.Match) -> bool:
    try:
        return all(0 <= int(part) <= 255 for part in match.group(0).split("."))
    except ValueError:
        return False


def _luhn_ok(digits: str) -> bool:
    total = 0
    for i, ch in enumerate(reversed(digits)):
        n = int(ch)
        if i % 2 == 1:
            n *= 2
            if n > 9:
                n -= 9
        total += n
    return total % 10 == 0


def _credit_card_ok(match: re.Match) -> bool:
    digits = re.sub(r"\D", "", match.group(0))
    if set(digits) <= {"0", "1"}:  # binary strings pass Luhn often enough to matter
        return False
    return 13 <= len(digits) <= 19 and _luhn_ok(digits)


# Taiwan national ID: letter -> two-digit area code, then weights 1,9,8,...,1,1.
_TW_LETTER_CODES = {c: n for c, n in zip("ABCDEFGHJKLMNPQRSTUVXYWZIO", range(10, 36))}


def _natid_ok(match: re.Match) -> bool:
    text = match.group(0)
    code = _TW_LETTER_CODES.get(text[0])
    if code is None:
        return False
    digits = [code // 10, code % 10] + [int(c) for c in text[1:]]
    weights = [1, 9, 8, 7, 6, 5, 4, 3, 2, 1, 1]
    return sum(d * w for d, w in zip(digits, weights)) % 10 == 0


def _phone_ok(match: re.Match) -> bool:
    text = match.group(0)
    digits = re.sub(r"\D", "", text)
    # A bare space-separated digit run is as likely a table row as a phone, so
    # a phone must carry a country code, parentheses, or -/. separators.
    return 10 <= len(digits) <= 15 and (text.startswith("+") or "(" in text or "-" in text or "." in text)


_PII_SCANNERS: dict[str, tuple[re.Pattern, Callable[[re.Match], bool] | None]] = {
    "email": (_EMAIL_RE, None),
    "phone": (_PHONE_RE, _phone_ok),
    "ipv4": (_IPV4_RE, _ipv4_ok),
    "credit_card": (_CREDIT_RE, _credit_card_ok),
    "national_id": (_NATID_RE, _natid_ok),
    "ssn": (_SSN_RE, None),
    "url": (_URL_RE, None),
}


def pii_scan(text: str, types: Iterable[str] | None = None) -> dict[str, int]:
    """Count PII matches per type with the builtin regexes.

    ``url`` is only counted when explicitly requested. Returns zeros so the
    readiness report can assert "remaining == 0" against empty results.
    """
    requested = tuple(types) if types is not None else DEFAULT_PII_TYPES
    out: dict[str, int] = {}
    for ptype in requested:
        pattern, validator = _PII_SCANNERS[ptype]
        count = 0
        for match in pattern.finditer(text):
            if validator is None or validator(match):
                count += 1
        out[ptype] = count
    return out


def _redact_pii_builtin(text: str, types: set[str]) -> tuple[str, dict[str, int]]:
    counts: dict[str, int] = {}
    unknown = sorted(set(types) - set(_PII_SCANNERS))
    if unknown:  # a typo in the redaction list must not become "zero redactions"
        raise ValueError(f"unknown PII type(s) {unknown}; known: {sorted(_PII_SCANNERS)}")
    for ptype in _PII_ORDER:
        if ptype not in types:
            continue
        pattern, validator = _PII_SCANNERS[ptype]
        token = _PII_TOKENS[ptype]

        def repl(match: re.Match, validator: Callable[[re.Match], bool] | None = validator,
                 token: str = token, ptype: str = ptype) -> str:
            if validator is not None and not validator(match):
                return match.group(0)
            counts[ptype] = counts.get(ptype, 0) + 1
            return token

        text = pattern.sub(repl, text)
    return text, counts


_PRESIDIO_MAP = {
    "EMAIL_ADDRESS": "email",
    "PHONE_NUMBER": "phone",
    "IP_ADDRESS": "ipv4",
    "IPV4_ADDRESS": "ipv4",
    "CREDIT_CARD": "credit_card",
    "US_SSN": "ssn",
    "TW_NATIONAL_ID": "national_id",
    "URL": "url",
}


def _redact_pii_presidio(text: str, analyzer: Any) -> tuple[str, dict[str, int]]:
    results = analyzer.analyze(text=text, language="en")
    counts: dict[str, int] = {}
    # apply spans right-to-left so offsets stay valid
    for result in sorted(results, key=lambda r: int(r.start), reverse=True):
        ptype = _PRESIDIO_MAP.get(str(result.entity_type))
        if ptype is None:
            continue
        text = text[: int(result.start)] + _PII_TOKENS[ptype] + text[int(result.end):]
        counts[ptype] = counts.get(ptype, 0) + 1
    return text, counts


# --------------------------------------------------------------------------
# HTML stripping (builtin; also used by ingest as the resiliparse fallback)
# --------------------------------------------------------------------------

class _HTMLToText(HTMLParser):
    _BLOCK = {
        "p", "br", "div", "li", "tr", "td", "th", "table", "ul", "ol",
        "h1", "h2", "h3", "h4", "h5", "h6", "section", "article", "blockquote", "pre",
    }
    _SKIP = {"script", "style", "noscript", "head", "title", "template"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._skip: list[str] = []  # open skip tags; only a MATCHING close pops

    def handle_starttag(self, tag: str, attrs: Any) -> None:
        if tag in self._SKIP:
            self._skip.append(tag)
        elif self._skip:
            return
        elif tag in self._BLOCK:
            self.parts.append("\n")
        elif tag not in _KNOWN_HTML_TAGS:
            # Not HTML (e.g. a <think> delimiter in reasoning data): keep verbatim.
            self.parts.append(self.get_starttag_text() or f"<{tag}>")

    def handle_endtag(self, tag: str) -> None:
        if tag in self._SKIP:
            if tag in self._skip:
                while self._skip and self._skip.pop() != tag:
                    pass
        elif self._skip:
            return
        elif tag in self._BLOCK:
            self.parts.append("\n")
        elif tag not in _KNOWN_HTML_TAGS:
            self.parts.append(f"</{tag}>")

    def handle_data(self, data: str) -> None:
        if not self._skip:
            self.parts.append(data)


def html_to_text(html_text: str) -> str:
    """Builtin HTML to text: drops scripts/styles/tags, keeps block structure.

    Approximation compared to resiliparse (no boilerplate/main-content model).
    """
    parser = _HTMLToText()
    parser.feed(html_text)
    raw = "".join(parser.parts)
    lines = [line.strip() for line in raw.splitlines()]
    lines = [line for line in lines if line]
    return "\n".join(lines)


# Only these names count as HTML. Real data showed the old "any <word>" rule
# stripping <think> delimiters from 22k reasoning records.
_KNOWN_HTML_TAGS = frozenset("""html head body div span p br a img table tr td th thead tbody ul ol li
h1 h2 h3 h4 h5 h6 script style meta link title section article header footer nav pre code em strong b i
blockquote iframe form input button noscript template hr sup sub small label select option""".split())
_HTML_TAG_RE = re.compile(
    r"</?(?:" + "|".join(sorted(_KNOWN_HTML_TAGS, key=len, reverse=True)) + r")\b[^<>]{0,300}/?>", re.IGNORECASE
)


# --------------------------------------------------------------------------
# mojibake repair
# --------------------------------------------------------------------------

_MOJIBAKE_MARKERS = ("Ã", "Â", "â")


def _non_ascii_ratio(text: str) -> float:
    if not text:
        return 0.0
    return sum(1 for ch in text if ord(ch) > 127) / len(text)


def fix_mojibake(text: str) -> tuple[str, bool]:
    """Repair UTF-8-decoded-as-latin-1 mojibake (e.g. "cafÃ©" -> "café").

    Uses ftfy when importable; otherwise the builtin repair is applied only
    when it strictly lowers the non-ASCII ratio. Returns (text, changed).
    """
    if not any(marker in text for marker in _MOJIBAKE_MARKERS):
        return text, False
    try:
        import ftfy  # type: ignore
    except ImportError:
        ftfy = None
    if ftfy is not None:
        fixed = str(ftfy.fix_text(text))
        return fixed, fixed != text
    try:
        fixed = text.encode("latin-1").decode("utf-8")
    except UnicodeError:
        return text, False
    if _non_ascii_ratio(fixed) < _non_ascii_ratio(text):
        return fixed, True
    return text, False


# --------------------------------------------------------------------------
# language ID
# --------------------------------------------------------------------------

_VIET_CHARS = frozenset("ăâđêôơưĂÂĐÊÔƠƯ")
_FT_MODELS: dict[str, Any] = {}


def builtin_langid(text: str) -> tuple[str, float]:
    """Builtin Unicode-script heuristic -> (lang, confidence).

    Covers en, zh, ja, ko, vi; everything else is "unknown". This is an
    approximation of fasttext lid.176: op stats must flag it.
    """
    cjk = hira_kata = hangul = latin = viet = 0
    for ch in text:
        cp = ord(ch)
        if 0x4E00 <= cp <= 0x9FFF or 0x3400 <= cp <= 0x4DBF:
            cjk += 1
        elif 0x3040 <= cp <= 0x30FF:
            hira_kata += 1
        elif 0xAC00 <= cp <= 0xD7AF or 0x1100 <= cp <= 0x11FF:
            hangul += 1
        elif ch.isalpha():
            latin += 1
            if ch in _VIET_CHARS or 0x1EA0 <= cp <= 0x1EFF:
                viet += 1
    letters = cjk + hira_kata + hangul + latin
    if letters < 3:
        return "unknown", 0.2
    if hangul and hangul / letters >= 0.2:
        return "ko", 0.9
    if hira_kata and hira_kata / letters >= 0.05:
        return "ja", 0.9
    if cjk and cjk / letters >= 0.2:
        return "zh", 0.9
    if viet and viet / max(1, latin) >= 0.02:
        return "vi", 0.7
    if latin:
        return "en", 0.6
    return "unknown", 0.2


def _fasttext_langid(text: str, model_path: str) -> tuple[str, float]:
    import fasttext  # type: ignore

    model = _FT_MODELS.get(model_path)
    if model is None:
        model = fasttext.load_model(model_path)
        _FT_MODELS[model_path] = model
    labels, probs = model.predict(text.replace("\n", " ")[:4000], k=1)
    label = str(labels[0]).replace("__label__", "") if labels else "unknown"
    return label, float(probs[0]) if len(probs) else 0.0


# --------------------------------------------------------------------------
# shared helpers
# --------------------------------------------------------------------------

_TEXT_FIELDS = ("text", "prompt", "chosen", "rejected", "answer")


def primary_text(rec: dict) -> str:
    """The record's main textual content: text, else joined message contents,
    else prompt+chosen/answer."""
    text = rec.get("text")
    if isinstance(text, str) and text.strip():
        return text
    messages = rec.get("messages")
    if isinstance(messages, list):
        parts = [str(m.get("content", "")) for m in messages if isinstance(m, dict)]
        joined = "\n".join(p for p in parts if p.strip())
        if joined.strip():
            return joined
    parts = [rec.get(k) for k in ("prompt", "chosen", "answer")]
    return "\n".join(p for p in parts if isinstance(p, str))


def mostly_cjk(text: str) -> bool:
    """True when more than half of alphabetic characters are CJK ideographs."""
    cjk = letters = 0
    for ch in text:
        if ch.isalpha():
            letters += 1
            cp = ord(ch)
            if 0x4E00 <= cp <= 0x9FFF or 0x3400 <= cp <= 0x4DBF:
                cjk += 1
    return letters > 0 and cjk / letters > 0.5


def _longest_run_fraction(text: str) -> float:
    if not text:
        return 0.0
    best = run = 1
    for i in range(1, len(text)):
        if text[i] == text[i - 1]:
            run += 1
            best = max(best, run)
        else:
            run = 1
    return best / len(text)


def _line_repeat_fraction(text: str) -> float:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return 0.0
    return 1.0 - len(set(lines)) / len(lines)


def _non_alnum_ratio(text: str) -> float:
    if not text:
        return 0.0
    bad = sum(1 for ch in text if not (ch.isalnum() or ch.isspace()))
    return bad / len(text)


def _clean_text(
    text: str,
    *,
    normalize: bool,
    mojibake: bool,
    strip_html: str | bool,
    pii_redact: bool,
    pii_types: set[str],
    presidio_analyzer: Any,
    stats: OpStats,
) -> str:
    if normalize:
        fixed = unicodedata.normalize("NFKC", text)
        if fixed != text:
            stats.modified["unicode_normalized"] += 1
            text = fixed
    do_strip = strip_html is True or (strip_html == "auto" and _HTML_TAG_RE.search(text) is not None)
    if do_strip and _HTML_TAG_RE.search(text):
        stripped_text = html_to_text(text)
        if stripped_text != text:
            stats.modified["html_stripped"] += 1
            text = stripped_text
    if mojibake:
        fixed, changed = fix_mojibake(text)
        if changed:
            stats.modified["mojibake_fixed"] += 1
            text = fixed
    if pii_redact:
        if presidio_analyzer is not None:
            text, counts = _redact_pii_presidio(text, presidio_analyzer)
        else:
            text, counts = _redact_pii_builtin(text, pii_types)
        for ptype, count in counts.items():
            stats.modified[f"pii:{ptype}"] += count
    return text


def _clean_op(records: Iterable[dict], cfg: dict, stats: OpStats) -> Iterator[dict]:
    normalize = bool(cfg.get("normalize_unicode", True))
    mojibake = bool(cfg.get("fix_mojibake", True))
    strip_html: str | bool = cfg.get("strip_html", "auto")
    langid_cfg = cfg.get("langid") or {}
    allowed = set(langid_cfg["allowed"]) if langid_cfg.get("allowed") else None
    min_confidence = float(langid_cfg.get("min_confidence", 0.5))
    model_path = langid_cfg.get("model_path")
    pii_cfg = cfg.get("pii") or {}
    pii_redact = bool(pii_cfg.get("redact", True))
    pii_types = set(pii_cfg.get("types") or DEFAULT_PII_TYPES)
    pii_backend = str(pii_cfg.get("backend", "builtin"))

    min_chars = cfg.get("min_chars")
    max_chars = cfg.get("max_chars")
    max_non_alnum = cfg.get("max_non_alnum_ratio")
    max_line_repeat = cfg.get("max_line_repeat_ratio")
    max_char_repeat = cfg.get("max_char_repeat_ratio")

    presidio_analyzer = None
    if pii_redact and pii_backend == "presidio":
        try:
            from presidio_analyzer import AnalyzerEngine  # type: ignore

            presidio_analyzer = AnalyzerEngine()
            stats.backend = "presidio"
            stats.extra["pii_backend"] = "presidio"
        except ImportError as exc:
            # explicitly requested: a refusal naming the dependency, never a silent downgrade
            raise CleanError("pii.backend='presidio' requires optional dependency 'presidio-analyzer'") from exc
    else:
        stats.extra["pii_backend"] = "builtin"

    langid_backend = "builtin"
    if model_path:
        try:
            import fasttext  # noqa: F401  type: ignore

            langid_backend = "fasttext"
        except ImportError as exc:
            raise CleanError("langid.model_path is set but optional dependency 'fasttext' is missing") from exc
    stats.extra["langid_backend"] = langid_backend
    if langid_backend == "builtin":
        stats.extra["approximate_langid"] = True

    for rec in counted(records, stats):
        if not isinstance(rec, dict):
            stats.drop("non_dict_record")
            continue

        for field in _TEXT_FIELDS:
            value = rec.get(field)
            if isinstance(value, str):
                rec[field] = _clean_text(
                    value, normalize=normalize, mojibake=mojibake, strip_html=strip_html,
                    pii_redact=pii_redact, pii_types=pii_types,
                    presidio_analyzer=presidio_analyzer, stats=stats,
                )
        messages = rec.get("messages")
        if isinstance(messages, list):
            for turn in messages:
                if not isinstance(turn, dict):
                    continue
                for key in ("content", "reasoning_content"):
                    if isinstance(turn.get(key), str):
                        turn[key] = _clean_text(
                            turn[key], normalize=normalize, mojibake=mojibake, strip_html=strip_html,
                            pii_redact=pii_redact, pii_types=pii_types,
                            presidio_analyzer=presidio_analyzer, stats=stats,
                        )

        text = primary_text(rec)
        stripped = text.strip()
        if not stripped:
            stats.drop("empty")
            continue
        if min_chars is not None and len(stripped) < int(min_chars):
            stats.drop("too_short")
            continue
        if max_chars is not None and len(stripped) > int(max_chars):
            stats.drop("too_long")
            continue

        # language ID always runs so meta.lang exists for coverage analysis;
        # it only filters when langid.allowed is configured.
        if langid_backend == "fasttext":
            lang, confidence = _fasttext_langid(stripped, str(model_path))
        else:
            lang, confidence = builtin_langid(stripped)
        meta = rec.setdefault("meta", {})
        if isinstance(meta, dict):
            meta["lang"] = lang
            meta["lang_confidence"] = round(confidence, 4)
        if allowed is not None and (lang not in allowed or confidence < min_confidence):
            stats.drop("lang")
            stats.extra.setdefault("lang_dropped", {}).setdefault(lang, 0)
            stats.extra["lang_dropped"][lang] += 1
            continue

        if max_non_alnum is not None and _non_alnum_ratio(stripped) > float(max_non_alnum):
            stats.drop("non_alnum")
            continue
        if max_line_repeat is not None and _line_repeat_fraction(stripped) > float(max_line_repeat):
            stats.drop("line_repeat")
            continue
        if max_char_repeat is not None and _longest_run_fraction(stripped) > float(max_char_repeat):
            stats.drop("char_repeat")
            continue

        # What redaction left behind: the readiness report needs a MEASURED
        # count here (absent key -> its PII check is UNMEASURED, not PASS).
        scan_types = tuple(t for t in pii_types if t in _PII_SCANNERS)
        remaining = sum(sum(pii_scan(t, scan_types).values()) for t in _record_texts(rec))
        stats.extra["pii_remaining"] = int(stats.extra.get("pii_remaining", 0)) + remaining

        stats.records_out += 1
        yield rec
    stats.extra.setdefault("pii_remaining", 0)


def _record_texts(rec: dict) -> Iterator[str]:
    for field in _TEXT_FIELDS:
        if isinstance(rec.get(field), str):
            yield rec[field]
    for turn in rec.get("messages") or []:
        if isinstance(turn, dict):
            for key in ("content", "reasoning_content"):
                if isinstance(turn.get(key), str):
                    yield turn[key]


register_op(FunctionOp("clean", _clean_op, CONFIG_SCHEMA))
