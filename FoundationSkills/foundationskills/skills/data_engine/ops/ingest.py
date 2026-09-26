
"""``ingest`` op: read raw sources and map them into canonical records.

The op is a *source*: it ignores its input iterator (counted/dropped so a
mis-wired pipeline is visible in stats) and yields one canonical record per
raw item. Supported kinds (``cfg.sources[i].kind``):

- ``local_dir``  recurse a directory and dispatch per extension
- ``jsonl``      one JSON object per line (strict: bad lines are dropped with
  reason ``parse_error``, never silently rewritten)
- ``json``       a JSON list, or an object with a ``data`` list, or one object
- ``parquet``    via ``pyarrow``; missing dep raises :class:`IngestError`
  naming ``pyarrow``
- ``csv``        stdlib ``csv.DictReader``
- ``hf_dataset`` via ``datasets.load_dataset(uri, split=..., streaming=...)``;
  missing dep raises :class:`IngestError` naming ``datasets``
- ``documents``  ``.txt/.md`` read as-is; ``.html/.htm`` via ``resiliparse``
  when importable, else the builtin tag stripper from ``clean.py``;
  ``.pdf/.docx`` via ``docling``, else counted under
  ``stats.dropped["unsupported_doc:<ext>"]``; :class:`IngestError` is raised
  only when *no* file produced a record (no silent skip-and-succeed)
- ``image_text`` a ``jsonl``/``parquet`` file with an image column plus a
  caption or messages column (``options.image_key`` / ``options.text_key``)

Field mapping: ``options.field_map`` maps *canonical* field -> *source*
field and wins over auto-detection. Auto-detection covers common names
(text/content/document/body, messages/conversations incl. ShareGPT
from/value, Alpaca instruction/input/output, prompt/question +
chosen/rejected, answer/solution/gold, image/images/image_path). A ``prompt``
is only auto-mapped when chosen/rejected/answer-style companions exist, so
plain QA corpora are not re-labelled. Missing ids are deterministic:
``sha1("<source>::<index>")[:16]``. ``meta.source`` is the individual file
path for file-derived records (the request uri otherwise), ``meta.license``
comes from ``options.license``.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Iterator

from foundationskills.skills.data_engine.ops.base import FunctionOp, OpStats, counted, register_op
from foundationskills.skills.data_engine.ops.clean import html_to_text


class IngestError(ValueError):
    """Raised when a source cannot be ingested; the message names the missing input/dependency."""


CONFIG_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["sources"],
    "properties": {
        "sources": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "required": ["uri", "kind"],
                "properties": {
                    "uri": {"type": "string", "minLength": 1},
                    "kind": {
                        "enum": ["local_dir", "jsonl", "json", "parquet", "csv", "hf_dataset", "documents", "image_text"]
                    },
                    "options": {"type": "object"},
                },
            },
        },
        "max_records": {"type": "integer", "minimum": 1},
    },
}

_TEXT_KEYS = ("text", "content", "document", "body")
_MESSAGES_KEYS = ("messages", "conversations", "conversation")
_IMAGE_KEYS = ("image", "images", "image_path", "image_url")
_ANSWER_KEYS = ("answer", "solution", "gold", "gold_answer", "reference")
_PROMPT_KEYS = ("prompt", "question")

_SHAREGPT_ROLES = {
    "human": "user",
    "gpt": "assistant",
    "system": "system",
    "user": "user",
    "assistant": "assistant",
}

_DOC_EXTS = (".txt", ".md", ".html", ".htm", ".pdf", ".docx")
_FILE_KIND_BY_EXT = {
    ".jsonl": "jsonl",
    ".ndjson": "jsonl",
    ".json": "json",
    ".csv": "csv",
    ".parquet": "parquet",
}


# --------------------------------------------------------------------------
# raw readers (yield plain source dicts)
# --------------------------------------------------------------------------

def _read_jsonl(path: Path, stats: OpStats) -> Iterator[dict]:
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                stats.drop("parse_error")
                continue
            if isinstance(obj, dict):
                yield obj
            else:
                stats.drop("non_object_record")


def _read_json(path: Path, stats: OpStats) -> Iterator[dict]:
    try:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            obj = json.load(handle)
    except json.JSONDecodeError as exc:
        raise IngestError(f"invalid JSON in {path}: {exc}") from exc
    if isinstance(obj, dict) and isinstance(obj.get("data"), list):
        items = obj["data"]
    elif isinstance(obj, list):
        items = obj
    elif isinstance(obj, dict):
        items = [obj]
    else:
        raise IngestError(f"unsupported JSON top-level type in {path}: {type(obj).__name__}")
    for item in items:
        if isinstance(item, dict):
            yield item
        else:
            stats.drop("non_object_record")


def _read_csv(path: Path, stats: OpStats) -> Iterator[dict]:
    with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row and any(v not in (None, "") for v in row.values()):
                yield dict(row)
            else:
                stats.drop("empty_row")


def _read_parquet(path: Path) -> Iterator[dict]:
    try:
        import pyarrow.parquet as pq  # type: ignore
    except ImportError as exc:
        raise IngestError(f"parquet source {path} requires optional dependency 'pyarrow'") from exc
    table = pq.read_table(str(path))
    for row in table.to_pylist():
        if isinstance(row, dict):
            yield row


def _read_hf_dataset(uri: str, options: dict) -> Iterator[dict]:
    try:
        from datasets import load_dataset  # type: ignore
    except ImportError as exc:
        raise IngestError(f"hf_dataset source {uri!r} requires optional dependency 'datasets'") from exc
    split = str(options.get("split") or "train")
    streaming = bool(options.get("streaming", False))
    name = options.get("name")
    if name is None:
        dataset = load_dataset(uri, split=split, streaming=streaming)
    else:
        dataset = load_dataset(uri, name, split=split, streaming=streaming)
    for row in dataset:
        if isinstance(row, dict):
            yield row


def _read_html_text(path: Path, stats: OpStats) -> str:
    raw = path.read_text(encoding="utf-8", errors="replace")
    try:
        from resiliparse.extract.html2text import extract_plain_text  # type: ignore
    except ImportError:
        stats.extra.setdefault("html_backend", "builtin")
        return html_to_text(raw)
    stats.extra.setdefault("html_backend", "resiliparse")
    return str(extract_plain_text(raw))


def _read_rich_document(path: Path, stats: OpStats) -> str | None:
    """Read .pdf/.docx via docling; None (counted) when the dep/conversion fails."""
    ext = path.suffix.lower()
    try:
        from docling.document_converter import DocumentConverter  # type: ignore
    except ImportError:
        stats.drop(f"unsupported_doc:{ext}")
        return None
    try:
        converter = DocumentConverter()
        result = converter.convert(str(path))
        return str(result.document.export_to_markdown())
    except Exception as exc:  # noqa: BLE001 - document conversion failures are counted, not fatal
        stats.drop(f"doc_error:{ext}")
        stats.extra.setdefault("doc_errors", []).append(f"{path}: {type(exc).__name__}")
        return None


def _read_document(path: Path, stats: OpStats) -> str | None:
    ext = path.suffix.lower()
    if ext in (".txt", ".md"):
        return path.read_text(encoding="utf-8", errors="replace")
    if ext in (".html", ".htm"):
        return _read_html_text(path, stats)
    if ext in (".pdf", ".docx"):
        return _read_rich_document(path, stats)
    stats.drop(f"unsupported_ext:{ext}")
    return None


def _iter_documents(uri: str, stats: OpStats) -> Iterator[tuple[str, dict]]:
    root = Path(uri)
    if root.is_file():
        files = [root]
    elif root.is_dir():
        files = sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in _DOC_EXTS)
    else:
        raise IngestError(f"missing input: documents source uri {uri!r}")
    if not files:
        raise IngestError(f"missing input: no supported documents under {uri!r}")
    produced = 0
    for file in files:
        text = _read_document(file, stats)
        if text is None:
            continue
        produced += 1
        yield str(file), {"text": text}
    if produced == 0:
        raise IngestError(
            f"all {len(files)} document file(s) under {uri!r} failed to ingest; "
            ".pdf/.docx require optional dependency 'docling'"
        )


def _iter_local_dir(uri: str, stats: OpStats) -> Iterator[tuple[str, dict]]:
    root = Path(uri)
    if not root.is_dir():
        raise IngestError(f"missing input: local_dir uri {uri!r} is not a directory")
    files = sorted(p for p in root.rglob("*") if p.is_file())
    if not files:
        raise IngestError(f"missing input: no files under {uri!r}")
    for file in files:
        ext = file.suffix.lower()
        file_kind = _FILE_KIND_BY_EXT.get(ext)
        if file_kind == "jsonl":
            for raw in _read_jsonl(file, stats):
                yield str(file), raw
        elif file_kind == "json":
            for raw in _read_json(file, stats):
                yield str(file), raw
        elif file_kind == "csv":
            for raw in _read_csv(file, stats):
                yield str(file), raw
        elif file_kind == "parquet":
            for raw in _read_parquet(file):
                yield str(file), raw
        elif ext in _DOC_EXTS:
            text = _read_document(file, stats)
            if text is not None:
                yield str(file), {"text": text}
        else:
            stats.drop(f"unsupported_ext:{ext}")


# --------------------------------------------------------------------------
# canonical record mapping
# --------------------------------------------------------------------------

def _first_present(raw: dict, keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key in raw and raw[key] is not None:
            return raw[key]
    return None


def _normalize_messages(value: Any) -> list[dict]:
    if not isinstance(value, list):
        return []
    out: list[dict] = []
    for turn in value:
        if not isinstance(turn, dict):
            continue
        if "role" in turn and "content" in turn:
            out.append({"role": str(turn["role"]), "content": str(turn["content"])})
            continue
        frm = str(turn.get("from", turn.get("speaker", "user")))
        role = _SHAREGPT_ROLES.get(frm.lower(), frm)
        content = turn.get("value", turn.get("text", turn.get("content", "")))
        out.append({"role": role, "content": str(content)})
    return out


def _normalize_images(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [str(v) for v in value if v is not None]
    return [str(value)]


def _map_record(raw: dict, *, source_uri: str, index: int, options: dict) -> dict:
    rec: dict[str, Any] = {}
    field_map = options.get("field_map") or {}
    for canonical, src in dict(field_map).items():
        if isinstance(src, str) and src in raw:
            rec[canonical] = raw[src]

    # messages (explicit, ShareGPT from/value, or Alpaca instruction/input/output)
    if "messages" not in rec:
        text_key = options.get("text_key")
        if text_key and isinstance(raw.get(text_key), list):
            rec["messages"] = raw[text_key]
        else:
            mv = _first_present(raw, _MESSAGES_KEYS)
            if mv is not None:
                rec["messages"] = mv
    if "messages" not in rec and isinstance(raw.get("instruction"), str):
        out = raw.get("output", raw.get("response"))
        if isinstance(out, str):
            inp = raw.get("input")
            user = raw["instruction"] if not isinstance(inp, str) or not inp.strip() else f"{raw['instruction']}\n{inp}"
            rec["messages"] = [
                {"role": "user", "content": user},
                {"role": "assistant", "content": out},
            ]
    if "messages" in rec:
        rec["messages"] = _normalize_messages(rec["messages"])

    # preference / RL companions first, so prompt mapping sees the shape
    if "chosen" not in rec and isinstance(raw.get("chosen"), str):
        rec["chosen"] = raw["chosen"]
    if "rejected" not in rec and isinstance(raw.get("rejected"), str):
        rec["rejected"] = raw["rejected"]
    if "answer" not in rec:
        answer = _first_present(raw, _ANSWER_KEYS)
        if isinstance(answer, str):
            rec["answer"] = answer

    if "prompt" not in rec and ("chosen" in rec or "rejected" in rec or "answer" in rec):
        prompt = _first_present(raw, _PROMPT_KEYS)
        if isinstance(prompt, str):
            rec["prompt"] = prompt

    if "text" not in rec:
        text_key = options.get("text_key")
        if text_key and isinstance(raw.get(text_key), str):
            rec["text"] = raw[text_key]
        else:
            text = _first_present(raw, _TEXT_KEYS)
            if isinstance(text, str):
                rec["text"] = text

    if "images" not in rec:
        image_key = options.get("image_key")
        images = raw.get(image_key) if isinstance(image_key, str) else _first_present(raw, _IMAGE_KEYS)
        if images is not None:
            rec["images"] = _normalize_images(images)

    rid = rec.get("id", raw.get("id"))
    if rid is None:
        rid = hashlib.sha1(f"{source_uri}::{index}".encode("utf-8")).hexdigest()[:16]
    rec["id"] = str(rid)

    meta_raw = raw.get("meta")
    meta = dict(meta_raw) if isinstance(meta_raw, dict) else {}
    meta.setdefault("source", source_uri)
    if options.get("license") is not None:
        meta.setdefault("license", str(options["license"]))
    for extra_key in ("lang", "domain"):
        if options.get(extra_key) is not None:
            meta.setdefault(extra_key, str(options[extra_key]))
    rec["meta"] = meta
    return rec


def _iter_source(uri: str, kind: str, options: dict, stats: OpStats, counter: list[int]) -> Iterator[dict]:
    kind = kind.lower()

    def mapped(raw: dict, source: str) -> dict:
        counter[0] += 1
        return _map_record(raw, source_uri=source, index=counter[0], options=options)

    if kind == "local_dir":
        for path, raw in _iter_local_dir(uri, stats):
            yield mapped(raw, path)
    elif kind in ("jsonl", "json", "csv", "parquet", "image_text"):
        path = Path(uri)
        if not path.is_file():
            raise IngestError(f"missing input: {kind} uri {uri!r} is not a file")
        use_kind = kind
        if kind == "image_text":
            ext = path.suffix.lower()
            if ext in (".jsonl", ".ndjson"):
                use_kind = "jsonl"
            elif ext == ".parquet":
                use_kind = "parquet"
            else:
                raise IngestError(f"image_text source {uri!r} must be .jsonl or .parquet")
        if use_kind == "jsonl":
            for raw in _read_jsonl(path, stats):
                yield mapped(raw, str(path))
        elif use_kind == "json":
            for raw in _read_json(path, stats):
                yield mapped(raw, str(path))
        elif use_kind == "csv":
            for raw in _read_csv(path, stats):
                yield mapped(raw, str(path))
        else:
            for raw in _read_parquet(path):
                yield mapped(raw, str(path))
    elif kind == "hf_dataset":
        for raw in _read_hf_dataset(uri, options):
            yield mapped(raw, uri)
    elif kind == "documents":
        for path, raw in _iter_documents(uri, stats):
            yield mapped(raw, path)
    else:
        raise IngestError(f"unknown ingest kind {kind!r} for uri {uri!r}")


def _ingest_op(records: Iterable[dict], cfg: dict, stats: OpStats) -> Iterator[dict]:
    # ingest is a source; count anything mis-wired upstream instead of ignoring it silently
    for _ in counted(records, stats):
        stats.drop("ingest_ignores_upstream")
    sources = list(cfg.get("sources") or [])
    if not sources:
        raise IngestError("missing input: cfg.sources (list of {uri, kind, options?})")
    max_records = cfg.get("max_records")
    counter = [0]
    emitted = 0
    stats.extra["sources"] = len(sources)
    for spec in sources:
        if not isinstance(spec, dict):
            raise IngestError(f"each source must be an object with uri/kind, got {type(spec).__name__}")
        uri = str(spec.get("uri") or "")
        if not uri:
            raise IngestError("missing input: source.uri is empty")
        kind = str(spec.get("kind") or "")
        options = dict(spec.get("options") or {})
        for rec in _iter_source(uri, kind, options, stats, counter):
            if max_records is not None and emitted >= int(max_records):
                stats.extra["truncated_at_max_records"] = int(max_records)
                stats.extra["records_emitted"] = emitted
                return
            emitted += 1
            stats.records_out += 1
            yield rec
    stats.extra["records_emitted"] = emitted


register_op(FunctionOp("ingest", _ingest_op, CONFIG_SCHEMA))
