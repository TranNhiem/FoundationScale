"""``toolcall_format`` op: normalise tool-calling traces to one canonical shape.

Seven published dialects (openai, sharegpt, hermes, glaive, xlam, llama3,
mistral) are detected per record and converted to OpenAI chat messages with
canonical tool calls (deterministic ids, canonical JSON argument strings) and
a JSON-schema ``tools`` list. Detection order is fixed (openai first) so a
record that satisfies several shapes always lands in the same bucket.

Decisions:

- Marker strings are assembled once from fragments (module constants) so the
  literals never appear verbatim in source or tests; every fixture is built
  from the exported constants.
- Missing call ids are derived from ``sha256(id_salt|rec_id|turn|k)`` so the
  same input yields the same ids across runs and machines (idempotence).
- xlam/glaive parameter dicts are promoted to JSON-schema objects; "str"-style
  types and ", optional" suffixes are mapped, ``optional``/``default`` marks a
  property as not required.
- Validation is strict by default: a record that fails any of the structural
  checks is dropped with a short slug. ``strict: false`` keeps it unchanged
  with ``meta.toolcall_status = "invalid:<slug>"`` so a caller can inspect the
  failure instead of losing the row (counted in ``stats.extra["kept_invalid"]``).
- Argument payloads are checked against the declared parameter schema with
  :mod:`foundationskills.core.schema` (the same validator the skill uses), so
  a schema violation is a data error, not a crash.

``stats.extra``: ``formats_detected``, ``calls``, ``tools_declared``,
``kept_invalid``, ``validator_backend``. Pure Python: :func:`preflight` is
always ``[]``.
"""
from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Callable, Iterable, Iterator

from foundationskills.core import schema as core_schema
from foundationskills.skills.data_engine.ops.base import FunctionOp, OpStats, counted, register_op

# ---------------------------------------------------------------------------
# marker constants (assembled from fragments: the literals must not appear
# verbatim in source or tests -- the serving stack parses them as tool calls)
# ---------------------------------------------------------------------------
TC_OPEN = "<" + "tool_call>"
TC_CLOSE = "</" + "tool_call>"
TR_OPEN = "<" + "tool_response>"
TR_CLOSE = "</" + "tool_response>"
TOOLS_OPEN = "<" + "tools>"
TOOLS_CLOSE = "</" + "tools>"
FUNCTIONCALL_TAG = "<" + "functioncall>"
MISTRAL_CALLS = "[" + "TOOL_CALLS]"
MISTRAL_RESULTS = "[" + "TOOL_RESULTS]"
PY_TAG = "<|" + "python_tag|>"
EOT_TAG = "<|" + "endoftext|>"

RESIDUAL_MARKERS = (TC_OPEN, FUNCTIONCALL_TAG, MISTRAL_CALLS, PY_TAG)

CONFIG_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,  # unknown keys refused: a silently ignored key measures nothing
    "properties": {
        "formats": {
            "type": "array",
            "minItems": 1,
            "items": {"enum": ["openai", "sharegpt", "hermes", "glaive", "xlam", "llama3", "mistral"]},
        },
        "require_results": {"type": "boolean"},
        "require_tools": {"type": "boolean"},
        "strict": {"type": "boolean"},
        "max_calls": {"type": "integer", "minimum": 1},
        "id_salt": {"type": "string"},
    },
}

_TYPE_MAP = {
    "str": "string",
    "string": "string",
    "int": "integer",
    "integer": "integer",
    "float": "number",
    "number": "number",
    "bool": "boolean",
    "boolean": "boolean",
    "list": "array",
    "array": "array",
    "dict": "object",
    "object": "object",
}

_SHAREGPT_ROLES = {
    "human": "user",
    "user": "user",
    "gpt": "assistant",
    "assistant": "assistant",
    "system": "system",
    "function_call": "function_call",
    "observation": "observation",
    "tool": "observation",
}

_GLAIVE_SPLIT_RE = re.compile(r"(?:\s*)\|\|EOT_TAG\|\|(?:\s*)".replace("||EOT_TAG||", EOT_TAG))
_GLAIVE_MARKER_RE = re.compile(
    r"(USER:|ASSISTANT:|" + re.escape(FUNCTIONCALL_TAG) + r"|FUNCTION RESPONSE:)"
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _canon_args(obj: Any) -> str:
    """Canonical JSON for tool arguments: sorted keys, compact separators."""
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def _parse_json(value: Any) -> Any:
    """Parse a JSON string (tolerating Python-style single quotes); pass dicts/lists through."""
    if isinstance(value, (dict, list)):
        return value
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        try:
            import ast

            return ast.literal_eval(text)
        except (ValueError, SyntaxError):
            return None


def _call_id(rec_id: str, turn: int, k: int, salt: str) -> str:
    digest = hashlib.sha256(f"{salt}|{rec_id}|{turn}|{k}".encode("utf-8")).hexdigest()
    return "call_" + digest[:12]


def _norm_params(raw: Any) -> dict:
    """Normalise a parameter declaration into a JSON-schema object."""
    if isinstance(raw, dict) and raw.get("type") == "object" and isinstance(raw.get("properties"), dict):
        out = dict(raw)
        out.setdefault("required", [])
        return out
    props: dict[str, Any] = {}
    required: list[str] = []
    if isinstance(raw, dict):
        items = list(raw.items())
    elif isinstance(raw, list):
        items = []
        for entry in raw:
            if isinstance(entry, dict) and "name" in entry:
                items.append((str(entry["name"]), entry))
    else:
        items = []
    for name, decl in items:
        name = str(name)
        if isinstance(decl, str):
            decl = {"type": decl}
        if not isinstance(decl, dict):
            decl = {}
        type_raw = str(decl.get("type", "string")).strip()
        type_raw = type_raw.replace(", optional", "").strip()
        json_type = _TYPE_MAP.get(type_raw.lower(), "string")
        prop: dict[str, Any] = {"type": json_type}
        desc = decl.get("description")
        if isinstance(desc, str):
            prop["description"] = desc
        if "enum" in decl and isinstance(decl["enum"], list):
            prop["enum"] = decl["enum"]
        if "default" in decl:
            prop["default"] = decl["default"]
        props[name] = prop
        optional = bool(decl.get("optional")) or "default" in decl or type_raw.lower().endswith("optional")
        if not optional:
            required.append(name)
    return {"type": "object", "properties": props, "required": required}


def _norm_tool_def(raw: Any) -> dict | None:
    """Normalise one tool declaration to ``{"type","function":{name,description,parameters}}``."""
    if not isinstance(raw, dict):
        return None
    fn = raw.get("function") if isinstance(raw.get("function"), dict) else raw
    name = fn.get("name")
    if not isinstance(name, str) or not name:
        return None
    desc = fn.get("description", "")
    params = _norm_params(fn.get("parameters", fn.get("arguments", {})))
    return {"type": "function", "function": {"name": name, "description": str(desc), "parameters": params}}


def _norm_tools(raw: Any) -> list[dict]:
    parsed = _parse_json(raw) if isinstance(raw, str) else raw
    if isinstance(parsed, dict):
        parsed = [parsed]
    if not isinstance(parsed, list):
        return []
    out: list[dict] = []
    for entry in parsed:
        tool = _norm_tool_def(entry)
        if tool is not None:
            out.append(tool)
    return out


def _make_call(name: Any, args: Any, call_id: str, ctx: dict | None = None) -> dict:
    obj = _parse_json(args)
    if obj is None:
        if ctx is not None and isinstance(args, str) and args.strip():
            ctx["errors"].append("bad_arguments_json")  # never launder unparsable arguments into {}
        obj = {}
    if not isinstance(obj, dict):
        obj = {"value": obj}
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": str(name), "arguments": _canon_args(obj)},
    }


# ---------------------------------------------------------------------------
# dialect detection + conversion
# ---------------------------------------------------------------------------

def _detect(rec: dict) -> str | None:
    """First dialect (spec order) that matches ANYWHERE in the record, not the first message that matches."""
    msgs = rec.get("messages")
    if isinstance(msgs, list):
        ms = [m for m in msgs if isinstance(m, dict)]
        texts = [m["content"] for m in ms if isinstance(m.get("content"), str)]
        if any(
            (m.get("role") == "assistant" and ("tool_calls" in m or "function_call" in m))
            or (m.get("role") == "tool" and "tool_call_id" in m)
            for m in ms
        ):
            return "openai"
        if any("from" in m and str(m.get("from", "")).lower() in _SHAREGPT_ROLES for m in ms):
            return "sharegpt"
        if any(TC_OPEN in t for t in texts):
            return "hermes"
        if any(m.get("role") == "ipython" for m in ms) or any(
            m.get("role") == "assistant"
            and isinstance(m.get("content"), str)
            and (m["content"].startswith(PY_TAG) or (m["content"].strip().startswith("{") and '"name"' in m["content"]))
            for m in ms
        ):
            return "llama3"
        if any(MISTRAL_CALLS in t for t in texts):
            return "mistral"
        # A messages list with no tool markers is still a chat record: treat it
        # as openai-shaped (the canonical form is a no-op for plain chat).
        return "openai"
    if isinstance(rec.get("chat"), list):
        return "glaive"
    if isinstance(rec.get("query"), str) and ("tools" in rec or "answers" in rec):
        return "xlam"
    if isinstance(rec.get("system"), str) and "functions:" in rec["system"]:
        return "glaive"
    return None


def _conv_openai(rec: dict, ctx: dict) -> tuple[list[dict], list[dict]]:
    tools = _norm_tools(rec.get("tools", rec.get("functions", [])))
    out: list[dict] = []
    for turn_i, m in enumerate(rec.get("messages") or []):
        if not isinstance(m, dict):
            continue
        role = str(m.get("role", "user"))
        content = m.get("content")
        content = content if isinstance(content, str) else ("" if content is None else str(content))
        if role == "assistant":
            calls_raw = m.get("tool_calls")
            if not calls_raw and isinstance(m.get("function_call"), dict):
                calls_raw = [{"id": m["function_call"].get("id"), "function": m["function_call"]}]
            if calls_raw:
                calls = []
                for k, c in enumerate(calls_raw):
                    if not isinstance(c, dict):
                        continue
                    fn = c.get("function") if isinstance(c.get("function"), dict) else c
                    cid = str(c.get("id") or _call_id(ctx["rec_id"], turn_i, k, ctx["salt"]))
                    calls.append(_make_call(fn.get("name", ""), fn.get("arguments", "{}"), cid, ctx=ctx))
                out.append({"role": "assistant", "content": content or None, "tool_calls": calls})
                continue
            out.append({"role": "assistant", "content": content})
        elif role == "tool":
            out.append(
                {
                    "role": "tool",
                    "tool_call_id": str(m.get("tool_call_id") or ""),
                    "content": content,
                }
            )
        elif role == "function":
            out.append({"role": "tool", "tool_call_id": str(m.get("tool_call_id") or ""), "content": content})
        else:
            out.append({"role": role, "content": content})
    return out, tools


def _conv_sharegpt(rec: dict, ctx: dict) -> tuple[list[dict], list[dict]]:
    tools = _norm_tools(rec.get("tools", []))
    out: list[dict] = []
    pending: list[dict] = []
    turn = 0
    for m in rec.get("messages") or []:
        if not isinstance(m, dict):
            continue
        frm = str(m.get("from", "human")).lower()
        role = _SHAREGPT_ROLES.get(frm, frm)
        value = m.get("value", m.get("content", ""))
        value = value if isinstance(value, str) else str(value)
        if role == "function_call":
            parsed = _parse_json(value) or {}
            name = parsed.get("name", "")
            args = parsed.get("arguments", "{}")
            cid = _call_id(ctx["rec_id"], turn, len(pending), ctx["salt"])
            pending.append(_make_call(name, args, cid, ctx=ctx))
        elif role == "observation":
            if pending:
                out.append({"role": "assistant", "content": None, "tool_calls": pending})
                for call in pending:
                    out.append({"role": "tool", "tool_call_id": call["id"], "content": value})
                pending = []
            else:
                out.append({"role": "tool", "tool_call_id": "", "content": value})
        else:
            if pending:
                out.append({"role": "assistant", "content": None, "tool_calls": pending})
                pending = []
            out.append({"role": role, "content": value})
        turn += 1
    if pending:
        out.append({"role": "assistant", "content": None, "tool_calls": pending})
    return out, tools


def _conv_hermes(rec: dict, ctx: dict) -> tuple[list[dict], list[dict]]:
    tools: list[dict] = _norm_tools(rec.get("tools", []))
    out: list[dict] = []
    turn = 0
    for m in rec.get("messages") or []:
        if not isinstance(m, dict):
            continue
        role = str(m.get("role", "user"))
        content = m.get("content", "")
        content = content if isinstance(content, str) else str(content)
        if not tools and TOOLS_OPEN in content:
            start = content.index(TOOLS_OPEN) + len(TOOLS_OPEN)
            end = content.index(TOOLS_CLOSE, start) if TOOLS_CLOSE in content[start:] else len(content)
            tools = _norm_tools(content[start:end])
            content = (content[: content.index(TOOLS_OPEN)] + content[end + len(TOOLS_CLOSE) :]).strip()
        calls: list[dict] = []
        if TC_OPEN in content:
            k = 0
            while TC_OPEN in content:
                s = content.index(TC_OPEN) + len(TC_OPEN)
                e = content.index(TC_CLOSE, s) if TC_CLOSE in content[s:] else len(content)
                blob = content[s:e]
                parsed = _parse_json(blob)
                if parsed is None:
                    ctx["errors"].append("malformed_tool_marker")
                else:
                    entries = parsed if isinstance(parsed, list) else [parsed]
                    for entry in entries:
                        if not isinstance(entry, dict):
                            ctx["errors"].append("malformed_tool_marker")
                            continue
                        cid = str(entry.get("id") or _call_id(ctx["rec_id"], turn, k, ctx["salt"]))
                        calls.append(_make_call(entry.get("name", ""), entry.get("arguments", entry.get("parameters", "{}")), cid, ctx=ctx))
                        k += 1
                content = content[: content.index(TC_OPEN)] + content[e + len(TC_CLOSE) :]
            content = content.strip()
        if calls:
            out.append({"role": "assistant", "content": content or None, "tool_calls": calls})
        elif TR_OPEN in content:
            k = 0
            while TR_OPEN in content:
                s = content.index(TR_OPEN) + len(TR_OPEN)
                e = content.index(TR_CLOSE, s) if TR_CLOSE in content[s:] else len(content)
                blob = content[s:e]
                parsed = _parse_json(blob)
                if isinstance(parsed, dict) and "tool_call_id" in parsed:
                    out.append({"role": "tool", "tool_call_id": str(parsed["tool_call_id"]), "content": str(parsed.get("content", ""))})
                else:
                    out.append({"role": "tool", "tool_call_id": "", "content": blob})
                k += 1
                content = content[: content.index(TR_OPEN)] + content[e + len(TR_CLOSE) :]
            content = content.strip()
            if content:
                out.append({"role": role, "content": content})
        else:
            out.append({"role": role, "content": content})
        turn += 1
    return out, tools


def _conv_glaive(rec: dict, ctx: dict) -> tuple[list[dict], list[dict]]:
    system_text = str(rec.get("system", ""))
    tools: list[dict] = []
    if "functions:" in system_text:
        blob = system_text.split("functions:", 1)[1].strip()
        parsed = _parse_json(blob)
        if parsed is not None:
            tools = _norm_tools(parsed)
    out: list[dict] = []
    if system_text:
        head = system_text.split("functions:", 1)[0].strip()
        if head:
            out.append({"role": "system", "content": head})
    turn = 0
    pending: list[dict] = []
    for chunk in _GLAIVE_SPLIT_RE.split(str(rec.get("chat", ""))):
        chunk = chunk.strip()
        if not chunk:
            continue
        parts = [p for p in _GLAIVE_MARKER_RE.split(chunk) if p]
        idx = 0
        while idx < len(parts):
            marker = parts[idx]
            body = parts[idx + 1] if idx + 1 < len(parts) else ""
            idx += 2
            if marker == "USER:":
                if pending:
                    out.append({"role": "assistant", "content": None, "tool_calls": pending})
                    pending = []
                out.append({"role": "user", "content": body.strip()})
            elif marker == "ASSISTANT:":
                if pending:
                    out.append({"role": "assistant", "content": None, "tool_calls": pending})
                    pending = []
                text = body.strip()
                if text:
                    out.append({"role": "assistant", "content": text})
            elif marker == FUNCTIONCALL_TAG:
                parsed = _parse_json(body.strip()) or {}
                name = parsed.get("name", "")
                args = parsed.get("arguments", parsed.get("parameters", "{}"))
                if isinstance(args, dict):
                    args = _canon_args(args)
                cid = _call_id(ctx["rec_id"], turn, len(pending), ctx["salt"])
                pending.append(_make_call(name, args, cid, ctx=ctx))
            elif marker == "FUNCTION RESPONSE:":
                if pending:
                    out.append({"role": "assistant", "content": None, "tool_calls": pending})
                    for call in pending:
                        out.append({"role": "tool", "tool_call_id": call["id"], "content": body.strip()})
                    pending = []
                else:
                    out.append({"role": "tool", "tool_call_id": "", "content": body.strip()})
            turn += 1
    if pending:
        out.append({"role": "assistant", "content": None, "tool_calls": pending})
    return out, tools


def _conv_xlam(rec: dict, ctx: dict) -> tuple[list[dict], list[dict]]:
    tools = _norm_tools(rec.get("tools", []))
    out: list[dict] = [{"role": "user", "content": str(rec.get("query", ""))}]
    answers = _parse_json(rec.get("answers", [])) or []
    if isinstance(answers, dict):
        answers = [answers]
    calls: list[dict] = []
    for k, entry in enumerate(answers):
        if not isinstance(entry, dict):
            continue
        cid = _call_id(ctx["rec_id"], 1, k, ctx["salt"])
        calls.append(_make_call(entry.get("name", ""), entry.get("arguments", "{}"), cid, ctx=ctx))
    if calls:
        out.append({"role": "assistant", "content": None, "tool_calls": calls})
    else:
        out.append({"role": "assistant", "content": ""})
    return out, tools


def _conv_llama3(rec: dict, ctx: dict) -> tuple[list[dict], list[dict]]:
    tools = _norm_tools(rec.get("tools", []))
    out: list[dict] = []
    turn = 0
    for m in rec.get("messages") or []:
        if not isinstance(m, dict):
            continue
        role = str(m.get("role", "user"))
        content = m.get("content", "")
        content = content if isinstance(content, str) else str(content)
        if role == "ipython":
            out.append({"role": "tool", "tool_call_id": "", "content": content})
            turn += 1
            continue
        if role == "assistant":
            body = content
            if body.startswith(PY_TAG):
                body = body[len(PY_TAG) :].strip()
            parsed = _parse_json(body) if body.strip().startswith("{") else None
            if isinstance(parsed, dict) and "name" in parsed:
                cid = _call_id(ctx["rec_id"], turn, 0, ctx["salt"])
                out.append(
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [_make_call(parsed.get("name", ""), parsed.get("parameters", parsed.get("arguments", "{}")), cid, ctx=ctx)],
                    }
                )
                turn += 1
                continue
        out.append({"role": role, "content": content})
        turn += 1
    return out, tools


def _conv_mistral(rec: dict, ctx: dict) -> tuple[list[dict], list[dict]]:
    tools = _norm_tools(rec.get("tools", []))
    out: list[dict] = []
    turn = 0
    for m in rec.get("messages") or []:
        if not isinstance(m, dict):
            continue
        role = str(m.get("role", "user"))
        content = m.get("content", "")
        content = content if isinstance(content, str) else str(content)
        if role == "tool":
            out.append({"role": "tool", "tool_call_id": str(m.get("tool_call_id") or ""), "content": content})
            turn += 1
            continue
        if MISTRAL_CALLS in content:
            s = content.index(MISTRAL_CALLS) + len(MISTRAL_CALLS)
            e = content.index(MISTRAL_RESULTS, s) if MISTRAL_RESULTS in content[s:] else len(content)
            blob = content[s:e]
            parsed = _parse_json(blob) or []
            if isinstance(parsed, dict):
                parsed = [parsed]
            calls: list[dict] = []
            for k, entry in enumerate(parsed):
                if not isinstance(entry, dict):
                    continue
                cid = str(entry.get("id") or _call_id(ctx["rec_id"], turn, k, ctx["salt"]))
                calls.append(_make_call(entry.get("name", ""), entry.get("arguments", entry.get("parameters", "{}")), cid, ctx=ctx))
            rest = content[: content.index(MISTRAL_CALLS)] + content[e:]
            if MISTRAL_RESULTS in rest:
                rs = rest.index(MISTRAL_RESULTS) + len(MISTRAL_RESULTS)
                results_blob = rest[rs:]
                rest = rest[: rest.index(MISTRAL_RESULTS)]
                out.append({"role": "assistant", "content": rest.strip() or None, "tool_calls": calls})
                r_parsed = _parse_json(results_blob) or []
                if isinstance(r_parsed, dict):
                    r_parsed = [r_parsed]
                for entry in r_parsed:
                    if isinstance(entry, dict):
                        out.append({"role": "tool", "tool_call_id": str(entry.get("id", entry.get("tool_call_id", ""))), "content": str(entry.get("content", entry.get("result", "")))})
                turn += 1
                continue
            out.append({"role": "assistant", "content": rest.strip() or None, "tool_calls": calls})
            turn += 1
            continue
        out.append({"role": role, "content": content})
        turn += 1
    return out, tools


_CONVERTERS: dict[str, Callable[[dict, dict], tuple[list[dict], list[dict]]]] = {
    "openai": _conv_openai,
    "sharegpt": _conv_sharegpt,
    "hermes": _conv_hermes,
    "glaive": _conv_glaive,
    "xlam": _conv_xlam,
    "llama3": _conv_llama3,
    "mistral": _conv_mistral,
}


# ---------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------

def _pair_results(messages: list[dict]) -> None:
    """Link tool results to calls in order (dialects whose wire format carries no call ids).

    In hermes/llama3/mistral/sharegpt/glaive traces a call usually has no id (one
    was minted) while its result has none or one of its own. A result whose id
    matches no call is paired with the oldest unanswered call: an empty result id
    takes the call's id, an explicit one is kept and the call adopts it (source
    ids survive). With no unanswered call left the result is untouched and later
    dropped as ``result_without_call``. openai traces are never repaired.
    """
    calls: dict[str, dict] = {}
    pending: list[dict] = []
    for m in messages:
        if m.get("role") == "assistant":
            for call in m.get("tool_calls") or []:
                calls[call["id"]] = call
                pending.append(call)
        elif m.get("role") == "tool":
            cid = m.get("tool_call_id", "")
            if cid in calls:
                if calls[cid] in pending:
                    pending.remove(calls[cid])
            elif pending:
                call = pending.pop(0)
                if cid:
                    del calls[call["id"]]
                    call["id"] = cid
                    calls[cid] = call
                else:
                    m["tool_call_id"] = call["id"]


def _validate_record(messages: list[dict], tools: list[dict], cfg: dict) -> str | None:
    if not messages:
        return "no_messages"
    if messages[0].get("role") == "assistant":
        return "roles_out_of_order"
    seen_ids: set[str] = set()
    call_ids: set[str] = set()
    tool_names: set[str] = set()
    for tool in tools:
        name = tool.get("function", {}).get("name")
        if name in tool_names:
            return "duplicate_tool_def"
        tool_names.add(name)
    calls_total = 0
    for m in messages:
        role = m.get("role")
        if role == "assistant":
            for call in m.get("tool_calls", []) or []:
                calls_total += 1
                cid = call.get("id", "")
                if not cid or cid in seen_ids:
                    return "tool_call_id_collision"
                seen_ids.add(cid)
                call_ids.add(cid)
                name = call.get("function", {}).get("name", "")
                if tool_names and name not in tool_names:
                    return "undeclared_tool_call"
                try:
                    args = json.loads(call.get("function", {}).get("arguments", "{}"))
                except json.JSONDecodeError:
                    return "bad_arguments_json"
                params = {}
                for tool in tools:
                    if tool.get("function", {}).get("name") == name:
                        params = tool["function"].get("parameters", {})
                if params:
                    try:
                        errors = core_schema.validate(args, params)
                    except Exception:  # noqa: BLE001 - schema errors are data errors
                        return "args_schema_violation"
                    if errors:
                        return "args_schema_violation"
        elif role == "tool":
            cid = m.get("tool_call_id", "")
            if cid not in call_ids:
                return "result_without_call"
    if cfg.get("require_tools", True) and not tools:
        return "missing_tools_schema"
    if cfg.get("require_results", False):
        for cid in call_ids:
            if not any(m.get("role") == "tool" and m.get("tool_call_id") == cid for m in messages):
                return "call_without_result"
    if calls_total > int(cfg.get("max_calls", 64)):
        return "too_many_calls"
    for m in messages:
        content = m.get("content")
        if isinstance(content, str):
            for marker in RESIDUAL_MARKERS:
                if marker in content:
                    return "residual_tool_markers"
    return None


# ---------------------------------------------------------------------------
# op
# ---------------------------------------------------------------------------

def _toolcall_format_op(records: Iterable[dict], cfg: dict, stats: OpStats) -> Iterator[dict]:
    formats = cfg.get("formats") or list(_CONVERTERS)
    strict = bool(cfg.get("strict", True))
    salt = str(cfg.get("id_salt", "fs"))
    stats.extra["formats_detected"] = {}
    stats.extra["kept_invalid"] = {}
    stats.extra["calls"] = 0
    stats.extra["tools_declared"] = 0
    stats.extra["validator_backend"] = "core.schema"

    for rec in counted(records, stats):
        if not isinstance(rec, dict):
            stats.drop("no_messages")
            continue
        dialect = _detect(rec)
        if dialect is None or dialect not in formats:
            slug = "unknown_tool_format"
            if strict:
                stats.drop(slug)
                continue
            stats.extra["kept_invalid"].setdefault(slug, 0)
            stats.extra["kept_invalid"][slug] += 1
            rec.setdefault("meta", {})["toolcall_status"] = f"invalid:{slug}"
            stats.records_out += 1
            yield rec
            continue
        ctx = {"rec_id": str(rec.get("id", "")), "salt": salt, "errors": []}
        try:
            messages, tools = _CONVERTERS[dialect](rec, ctx)
        except Exception:  # noqa: BLE001 - a converter crash is a malformed record
            messages, tools = [], []
            ctx["errors"].append("malformed_tool_marker")
        if dialect != "openai" and not ctx["errors"]:
            _pair_results(messages)
        if ctx["errors"]:
            slug = ctx["errors"][0]
            if strict:
                stats.drop(slug)
                continue
            stats.extra["kept_invalid"].setdefault(slug, 0)
            stats.extra["kept_invalid"][slug] += 1
            rec.setdefault("meta", {})["toolcall_status"] = f"invalid:{slug}"
            stats.records_out += 1
            yield rec
            continue
        slug = _validate_record(messages, tools, cfg)
        if slug is not None:
            if strict:
                stats.drop(slug)
                continue
            stats.extra["kept_invalid"].setdefault(slug, 0)
            stats.extra["kept_invalid"][slug] += 1
            rec.setdefault("meta", {})["toolcall_status"] = f"invalid:{slug}"
            stats.records_out += 1
            yield rec
            continue
        stats.extra["formats_detected"][dialect] = stats.extra["formats_detected"].get(dialect, 0) + 1
        stats.extra["calls"] += sum(len(m.get("tool_calls", []) or []) for m in messages)
        stats.extra["tools_declared"] += len(tools)
        new_rec = dict(rec)
        new_rec["messages"] = messages
        new_rec["tools"] = tools
        if new_rec != rec:
            stats.modified["reformatted"] += 1
        stats.records_out += 1
        yield new_rec


def preflight(cfg: dict) -> list[str]:
    """Pure Python: no runtime dependencies to check."""
    return []


register_op(FunctionOp("toolcall_format", _toolcall_format_op, CONFIG_SCHEMA))
