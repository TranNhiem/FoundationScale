"""Wire-format translation for the agentic-RL gateway (PART 1: parsing).

Pure, stdlib-only, torch-free, I/O-free translation between four client wire
formats and ONE canonical form (OpenAI Chat Completions messages + tools).
Translation happens BEFORE tokenization; nothing here touches a tokenizer or
an engine.

Supported client formats
------------------------
* ``chat_completions``   -- OpenAI Chat Completions style requests
* ``responses``          -- OpenAI Responses style requests
* ``anthropic_messages`` -- Anthropic Messages style requests
* ``gemini``             -- Google Gemini ``generateContent`` style requests

Contract
--------
* All canonical dataclasses are frozen; sequences are tuples, never lists.
* ``bool`` is checked with ``type(x) is bool`` BEFORE any ``int`` check (in
  Python ``bool`` subclasses ``int`` and would otherwise be accepted).
* Any input that cannot be represented canonically raises
  :class:`WireRefusal` (a ``ValueError`` subclass) with a message naming what
  was expected and what arrived (counts where relevant).
* Data is never invented. A field that cannot be represented is refused,
  unless it is explicitly listed as "recorded and dropped", in which case its
  name is recorded in ``CanonicalRequest.extras_dropped``.

PART 1 provides :func:`detect_format` and :func:`parse_request` (with one
private parser per format). ``render_response`` / ``render_stream`` live in
PART 2.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any

__all__ = [
    "WireRefusal",
    "WireFormat",
    "CanonicalToolCall",
    "CanonicalMessage",
    "CanonicalTool",
    "SamplingRequest",
    "CanonicalRequest",
    "CanonicalResult",
    "detect_format",
    "parse_request",
]


class WireRefusal(ValueError):
    """Raised when a client payload cannot be represented canonically.

    Messages always name what was expected and what arrived (with counts where
    relevant) so that a client can see exactly which field was rejected.
    """


class WireFormat(str, Enum):
    """The four client wire formats the gateway can translate."""

    CHAT_COMPLETIONS = "chat_completions"
    RESPONSES = "responses"
    ANTHROPIC_MESSAGES = "anthropic_messages"
    GEMINI = "gemini"


@dataclass(frozen=True)
class CanonicalToolCall:
    """One tool (function) call as requested/produced.

    ``arguments_raw`` is the raw JSON text of the arguments and is NEVER
    re-serialised by us. ``call_id_origin`` records whether the id came from
    the client/model (``"model"``) or had to be synthesized (``"synthesized"``,
    e.g. Gemini, which has no call ids).
    """

    call_id: str
    name: str
    arguments_raw: str
    call_id_origin: str = "model"


@dataclass(frozen=True)
class CanonicalMessage:
    """One message in the canonical conversation.

    ``content`` is text only in P0 (image/file parts are refused as content and
    only recorded via ``CanonicalRequest.has_images``). ``tool_calls`` is
    assistant-only; ``tool_call_id`` is required for role ``"tool"``.
    ``reasoning`` carries assistant reasoning text when the client sent it back.
    """

    role: str
    content: str | None
    tool_calls: tuple[CanonicalToolCall, ...] = ()
    tool_call_id: str | None = None
    name: str | None = None
    reasoning: str | None = None


@dataclass(frozen=True)
class CanonicalTool:
    """One tool declaration: a JSON-schema object plus name/description."""

    name: str
    description: str | None
    parameters: Mapping[str, Any]


@dataclass(frozen=True)
class SamplingRequest:
    """What the CLIENT asked for, recorded verbatim (never defaulted/invented)."""

    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    max_tokens: int | None = None
    stop: tuple[str, ...] = ()
    n: int | None = None
    seed: int | None = None


@dataclass(frozen=True)
class CanonicalRequest:
    """A fully parsed client request in canonical form."""

    format: WireFormat
    model: (
        str | None
        # None when the format carries it in the URL (Gemini) or the client omits it; the gateway
        # fills it from the path/session
    )
    messages: tuple[CanonicalMessage, ...]
    tools: tuple[CanonicalTool, ...]
    tool_choice: str | Mapping[str, Any] | None
    sampling: SamplingRequest
    stream: bool
    has_images: bool
    extras_dropped: tuple[str, ...]


@dataclass(frozen=True)
class CanonicalResult:
    """What the ENGINE produced for one call (rendered back per format)."""

    text: str | None
    tool_calls: tuple[CanonicalToolCall, ...]
    reasoning: str | None
    finish_reason: str
    prompt_tokens: int
    completion_tokens: int
    response_id: str
    created: int


# ---------------------------------------------------------------------------
# Small shared validation helpers
# ---------------------------------------------------------------------------


def _refuse(expected: str, got: Any) -> WireRefusal:
    """Build a refusal naming what was expected and what arrived."""

    return WireRefusal(f"expected {expected}, got {got!r}")


def _require_mapping(value: Any, expected: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise _refuse(expected, value)
    return value


def _require_str(value: Any, expected: str) -> str:
    if not isinstance(value, str):
        raise _refuse(expected, value)
    return value


def _optional_str(value: Any, expected: str) -> str | None:
    if value is None:
        return None
    return _require_str(value, expected)


def _require_int(value: Any, expected: str) -> int:
    # bool is a subclass of int in Python: check it FIRST.
    if type(value) is bool or not isinstance(value, int):
        raise _refuse(expected, value)
    return value


def _optional_int(value: Any, expected: str) -> int | None:
    if value is None:
        return None
    return _require_int(value, expected)


def _require_float(value: Any, expected: str) -> float:
    if type(value) is bool or not isinstance(value, (int, float)):
        raise _refuse(expected, value)
    return float(value)


def _optional_float(value: Any, expected: str) -> float | None:
    if value is None:
        return None
    return _require_float(value, expected)


def _require_bool(value: Any, expected: str) -> bool:
    if type(value) is not bool:
        raise _refuse(expected, value)
    return value


def _optional_bool(value: Any, expected: str) -> bool | None:
    if value is None:
        return None
    return _require_bool(value, expected)


def _require_sequence(value: Any, expected: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise _refuse(expected, value)
    return value


def _stop_tuple(value: Any) -> tuple[str, ...]:
    """``stop`` may be a single string or a list of strings in every format."""

    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    seq = _require_sequence(value, "stop to be a string or a list of strings")
    out: list[str] = []
    for item in seq:
        out.append(_require_str(item, "each stop entry to be a string"))
    return tuple(out)


def _check_call_id_origin(origin: str) -> str:
    if origin not in ("model", "synthesized"):
        raise _refuse("call_id_origin to be 'model' or 'synthesized'", origin)
    return origin


def _check_role(role: str) -> str:
    if role not in ("system", "user", "assistant", "tool"):
        raise _refuse("role to be one of 'system', 'user', 'assistant', 'tool'", role)
    return role


def _check_finish_reason(reason: str) -> str:
    if reason not in ("stop", "length", "tool_calls"):
        raise _refuse("finish_reason to be one of 'stop', 'length', 'tool_calls'", reason)
    return reason


def _validate_message(msg: CanonicalMessage) -> CanonicalMessage:
    """Validate cross-field invariants of a canonical message."""

    _check_role(msg.role)
    _check_call_id_origin  # noqa: B018  (kept for symmetry; ids validated below)
    for call in msg.tool_calls:
        _check_call_id_origin(call.call_id_origin)
    if msg.role == "tool":
        if msg.tool_call_id is None:
            raise _refuse("tool_call_id to be present for role 'tool'", msg.tool_call_id)
        if msg.tool_calls:
            raise _refuse("no tool_calls on a role 'tool' message", len(msg.tool_calls))
    elif msg.tool_call_id is not None:
        raise _refuse(f"tool_call_id to be absent for role {msg.role!r}", msg.tool_call_id)
    if msg.tool_calls and msg.role != "assistant":
        raise _refuse("tool_calls only on role 'assistant' messages", msg.role)
    return msg


def _validate_request(req: CanonicalRequest) -> CanonicalRequest:
    if not req.messages:
        raise _refuse("a non-empty message list after parsing", 0)
    for msg in req.messages:
        _validate_message(msg)
    return req


# ---------------------------------------------------------------------------
# detect_format
# ---------------------------------------------------------------------------


def _optional_model(value: Any) -> str | None:
    """``model`` is optional on the wire (Gemini puts it in the URL); a present value must be a
    str.
    """
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise WireRefusal(f"expected model to be a non-empty string or absent, got {value!r}")
    return value


def detect_format(
    path: str,
    headers: Mapping[str, str],
    body: Mapping[str, Any],
) -> WireFormat:
    """Detect the client wire format from path, headers and body shape.

    Order of detection:

    1. ``path`` suffix: ``/chat/completions``, ``/responses``, ``/messages``,
       ``:generateContent``, ``:streamGenerateContent``.
    2. Header ``anthropic-version`` (case-insensitive) -> ANTHROPIC_MESSAGES.
    3. Body shape: ``contents`` -> GEMINI; ``input`` plus (``instructions`` or
       no ``messages``) -> RESPONSES; ``messages`` -> CHAT_COMPLETIONS.

    Undetectable input raises :class:`WireRefusal`.
    """

    if not isinstance(path, str):
        raise _refuse("path to be a string", path)
    _require_mapping(headers, "headers to be a mapping")
    _require_mapping(body, "body to be a mapping")

    # 1. Path first.
    if path.endswith("/chat/completions"):
        return WireFormat.CHAT_COMPLETIONS
    if path.endswith("/responses"):
        return WireFormat.RESPONSES
    if path.endswith("/messages"):
        return WireFormat.ANTHROPIC_MESSAGES
    if path.endswith(":generateContent") or path.endswith(":streamGenerateContent"):
        return WireFormat.GEMINI

    # 2. anthropic-version header (case-insensitive lookup).
    lowered: dict[str, str] = {str(k).lower(): v for k, v in headers.items()}
    if "anthropic-version" in lowered:
        return WireFormat.ANTHROPIC_MESSAGES

    # 3. Body shape.
    if "contents" in body:
        return WireFormat.GEMINI
    if "input" in body and ("instructions" in body or "messages" not in body):
        return WireFormat.RESPONSES
    if "messages" in body:
        return WireFormat.CHAT_COMPLETIONS

    raise WireRefusal(
        "expected a request body shaped like chat_completions ('messages'), responses "
        "('input' with 'instructions' or without 'messages'), anthropic_messages "
        "(path '/messages' or 'anthropic-version' header) or gemini ('contents'), "
        f"got body keys {sorted(body.keys())!r} for path {path!r}"
    )


# ---------------------------------------------------------------------------
# parse_request
# ---------------------------------------------------------------------------


def parse_request(fmt: WireFormat, body: Mapping[str, Any]) -> CanonicalRequest:
    """Parse a client request body into a :class:`CanonicalRequest`.

    Exactly one private parser per format is dispatched here. Anything that
    cannot be represented canonically raises :class:`WireRefusal`.
    """

    if not isinstance(fmt, WireFormat):
        raise _refuse("fmt to be a WireFormat", fmt)
    body = _require_mapping(body, "body to be a mapping")

    if fmt is WireFormat.CHAT_COMPLETIONS:
        return _parse_chat(body)
    if fmt is WireFormat.RESPONSES:
        return _parse_responses(body)
    if fmt is WireFormat.ANTHROPIC_MESSAGES:
        return _parse_anthropic(body)
    return _parse_gemini(body)


# ---------------------------------------------------------------------------
# CHAT COMPLETIONS
# ---------------------------------------------------------------------------


def _parse_chat(body: Mapping[str, Any]) -> CanonicalRequest:
    extras: list[str] = []
    has_images = False

    model = _optional_model(body.get("model"))

    raw_messages = body.get("messages")
    raw_messages = _require_sequence(raw_messages, "messages to be a list")
    messages: list[CanonicalMessage] = []
    for raw in raw_messages:
        item = _require_mapping(raw, "each message to be an object")
        role = _require_str(item.get("role"), "message 'role' to be a string")
        _check_role(role)

        content_raw = item.get("content")
        content: str | None
        if content_raw is None:
            content = None
        elif isinstance(content_raw, str):
            content = content_raw
        elif isinstance(content_raw, Sequence) and not isinstance(content_raw, (str, bytes)):
            parts: list[str] = []
            for part in content_raw:
                p = _require_mapping(part, "each content part to be an object")
                ptype = p.get("type")
                if ptype == "text":
                    parts.append(_require_str(p.get("text"), "text part 'text' to be a string"))
                elif ptype in ("image_url", "image", "input_image", "file"):
                    has_images = True
                else:
                    raise _refuse(
                        "content part 'type' to be 'text' or an image/file type "
                        "('image_url', 'image', 'input_image', 'file')",
                        ptype,
                    )
            content = "".join(parts) if parts else None
        else:
            raise _refuse(
                "message 'content' to be a string, a list of content parts, or null",
                content_raw,
            )

        tool_calls_raw = item.get("tool_calls")
        tool_calls: list[CanonicalToolCall] = []
        if tool_calls_raw is not None:
            if role != "assistant":
                raise _refuse("tool_calls only on assistant messages", role)
            seq = _require_sequence(tool_calls_raw, "tool_calls to be a list")
            for tc in seq:
                t = _require_mapping(tc, "each tool_call to be an object")
                call_id = _require_str(t.get("id"), "tool_call 'id' to be a string")
                fn = _require_mapping(t.get("function"), "tool_call 'function' to be an object")
                name = _require_str(fn.get("name"), "tool_call function 'name' to be a string")
                arguments_raw = fn.get("arguments")
                if arguments_raw is None:
                    arguments_raw = ""
                arguments_raw = _require_str(
                    arguments_raw, "tool_call function 'arguments' to be a JSON string"
                )
                tool_calls.append(
                    CanonicalToolCall(
                        call_id=call_id,
                        name=name,
                        arguments_raw=arguments_raw,
                        call_id_origin="model",
                    )
                )

        tool_call_id = _optional_str(item.get("tool_call_id"), "tool_call_id to be a string")
        message_name = _optional_str(item.get("name"), "message 'name' to be a string")
        reasoning = _optional_str(item.get("reasoning_content"), "reasoning_content to be a string")
        if reasoning is None:
            reasoning = _optional_str(item.get("reasoning"), "reasoning to be a string")

        messages.append(
            _validate_message(
                CanonicalMessage(
                    role=role,
                    content=content,
                    tool_calls=tuple(tool_calls),
                    tool_call_id=tool_call_id,
                    name=message_name,
                    reasoning=reasoning,
                )
            )
        )

    tools, tool_choice = _parse_chat_tools(body.get("tools"), body.get("tool_choice"))

    # Sampling: max_tokens OR max_completion_tokens.
    max_tokens_raw = body.get("max_tokens")
    max_completion_raw = body.get("max_completion_tokens")
    if max_tokens_raw is not None and max_completion_raw is not None:
        raise WireRefusal("expected only one of 'max_tokens' or 'max_completion_tokens', got both")
    max_tokens = _optional_int(
        max_tokens_raw if max_tokens_raw is not None else max_completion_raw,
        "max_tokens/max_completion_tokens to be an integer",
    )

    sampling = SamplingRequest(
        temperature=_optional_float(body.get("temperature"), "temperature to be a number"),
        top_p=_optional_float(body.get("top_p"), "top_p to be a number"),
        top_k=_optional_int(body.get("top_k"), "top_k to be an integer"),
        max_tokens=max_tokens,
        stop=_stop_tuple(body.get("stop")),
        n=_optional_int(body.get("n"), "n to be an integer"),
        seed=_optional_int(body.get("seed"), "seed to be an integer"),
    )

    stream = _optional_bool(body.get("stream"), "stream to be a boolean") or False

    # Recorded-and-dropped fields (documented as droppable).
    for key in (
        "user",
        "metadata",
        "logprobs",
        "top_logprobs",
        "presence_penalty",
        "frequency_penalty",
        "logit_bias",
        "response_format",
        "store",
    ):
        if key in body:
            extras.append(key)

    return _validate_request(
        CanonicalRequest(
            format=WireFormat.CHAT_COMPLETIONS,
            model=model,
            messages=tuple(messages),
            tools=tools,
            tool_choice=tool_choice,
            sampling=sampling,
            stream=stream,
            has_images=has_images,
            extras_dropped=tuple(extras),
        )
    )


def _parse_chat_tools(
    tools_raw: Any, tool_choice_raw: Any
) -> tuple[tuple[CanonicalTool, ...], str | Mapping[str, Any] | None]:
    tools: list[CanonicalTool] = []
    if tools_raw is not None:
        seq = _require_sequence(tools_raw, "tools to be a list")
        for raw in seq:
            t = _require_mapping(raw, "each tool to be an object")
            ttype = t.get("type")
            if ttype is not None and ttype != "function":
                raise _refuse("tool 'type' to be 'function'", ttype)
            fn = _require_mapping(t.get("function"), "tool 'function' to be an object")
            name = _require_str(fn.get("name"), "tool function 'name' to be a string")
            description = _optional_str(
                fn.get("description"), "tool function 'description' to be a string"
            )
            parameters = fn.get("parameters")
            if parameters is None:
                raise _refuse("tool function 'parameters' to be a JSON schema object", None)
            parameters = _require_mapping(
                parameters, "tool function 'parameters' to be a JSON schema object"
            )
            tools.append(CanonicalTool(name=name, description=description, parameters=parameters))

    tool_choice: str | Mapping[str, Any] | None
    if tool_choice_raw is None:
        tool_choice = None
    elif isinstance(tool_choice_raw, (str, Mapping)):
        tool_choice = tool_choice_raw
    else:
        raise _refuse("tool_choice to be a string, an object, or null", tool_choice_raw)

    return tuple(tools), tool_choice


# ---------------------------------------------------------------------------
# RESPONSES
# ---------------------------------------------------------------------------


def _parse_responses(body: Mapping[str, Any]) -> CanonicalRequest:
    extras: list[str] = []
    has_images = False

    model = _optional_model(body.get("model"))

    messages: list[CanonicalMessage] = []

    instructions = body.get("instructions")
    if instructions is not None:
        instructions = _require_str(instructions, "instructions to be a string")
        messages.append(_validate_message(CanonicalMessage(role="system", content=instructions)))

    input_raw = body.get("input")
    if input_raw is None:
        raise _refuse("input to be a string or a list of input items", None)

    if isinstance(input_raw, str):
        messages.append(_validate_message(CanonicalMessage(role="user", content=input_raw)))
    else:
        items = _require_sequence(input_raw, "input to be a string or a list of input items")
        pending_reasoning: str | None = None
        for raw in items:
            item = _require_mapping(raw, "each input item to be an object")
            itype = item.get("type")

            if itype == "message":
                role = _require_str(item.get("role"), "input message 'role' to be a string")
                _check_role(role)
                content_raw = item.get("content")
                content_raw = _require_sequence(
                    content_raw, "input message 'content' to be a list of parts"
                )
                parts: list[str] = []
                for part in content_raw:
                    p = _require_mapping(part, "each input message content part to be an object")
                    ptype = p.get("type")
                    if ptype in ("input_text", "output_text"):
                        parts.append(_require_str(p.get("text"), "text part 'text' to be a string"))
                    elif ptype in ("input_image", "input_file"):
                        has_images = True
                    else:
                        raise _refuse(
                            "input message content part 'type' to be 'input_text', "
                            "'output_text', 'input_image' or 'input_file'",
                            ptype,
                        )
                content = "".join(parts) if parts else None
                reasoning = pending_reasoning
                pending_reasoning = None
                messages.append(
                    _validate_message(
                        CanonicalMessage(role=role, content=content, reasoning=reasoning)
                    )
                )

            elif itype == "function_call":
                call_id = _require_str(
                    item.get("call_id"), "function_call 'call_id' to be a string"
                )
                name = _require_str(item.get("name"), "function_call 'name' to be a string")
                arguments_raw = item.get("arguments")
                if arguments_raw is None:
                    arguments_raw = ""
                arguments_raw = _require_str(
                    arguments_raw, "function_call 'arguments' to be a JSON string"
                )
                call = CanonicalToolCall(
                    call_id=call_id,
                    name=name,
                    arguments_raw=arguments_raw,
                    call_id_origin="model",
                )
                reasoning = pending_reasoning
                pending_reasoning = None
                # Consecutive function_call items merge into one assistant message.
                if messages and messages[-1].role == "assistant" and messages[-1].tool_calls:
                    last = messages[-1]
                    messages[-1] = _validate_message(
                        CanonicalMessage(
                            role="assistant",
                            content=last.content,
                            tool_calls=last.tool_calls + (call,),
                            reasoning=last.reasoning,
                        )
                    )
                else:
                    messages.append(
                        _validate_message(
                            CanonicalMessage(
                                role="assistant",
                                content=None,
                                tool_calls=(call,),
                                reasoning=reasoning,
                            )
                        )
                    )

            elif itype == "function_call_output":
                call_id = _require_str(
                    item.get("call_id"), "function_call_output 'call_id' to be a string"
                )
                output_raw = item.get("output")
                output = _require_str(output_raw, "function_call_output 'output' to be a string")
                pending_reasoning = None
                messages.append(
                    _validate_message(
                        CanonicalMessage(role="tool", content=output, tool_call_id=call_id)
                    )
                )

            elif itype == "reasoning":
                summary_raw = item.get("summary")
                summary_raw = _require_sequence(
                    summary_raw, "reasoning 'summary' to be a list of {text} objects"
                )
                summary_parts: list[str] = []
                for s in summary_raw:
                    sm = _require_mapping(s, "each reasoning summary entry to be an object")
                    summary_parts.append(
                        _require_str(sm.get("text"), "summary entry 'text' to be a string")
                    )
                pending_reasoning = "".join(summary_parts) if summary_parts else None

            else:
                raise _refuse(
                    "input item 'type' to be 'message', 'function_call', "
                    "'function_call_output' or 'reasoning'",
                    itype,
                )

        if pending_reasoning is not None:
            raise WireRefusal(
                "expected a trailing 'reasoning' item to be followed by an assistant "
                "'message' item to attach its reasoning to, got end of input"
            )

    tools_raw = body.get("tools")
    tools: list[CanonicalTool] = []
    if tools_raw is not None:
        seq = _require_sequence(tools_raw, "tools to be a list")
        for raw in seq:
            t = _require_mapping(raw, "each tool to be an object")
            ttype = t.get("type")
            if ttype is not None and ttype != "function":
                raise _refuse("tool 'type' to be 'function'", ttype)
            name = _require_str(t.get("name"), "tool 'name' to be a string")
            description = _optional_str(t.get("description"), "tool 'description' to be a string")
            parameters = t.get("parameters")
            if parameters is None:
                raise _refuse("tool 'parameters' to be a JSON schema object", None)
            parameters = _require_mapping(
                parameters, "tool 'parameters' to be a JSON schema object"
            )
            tools.append(CanonicalTool(name=name, description=description, parameters=parameters))

    tool_choice_raw = body.get("tool_choice")
    tool_choice: str | Mapping[str, Any] | None
    if tool_choice_raw is None:
        tool_choice = None
    elif isinstance(tool_choice_raw, (str, Mapping)):
        tool_choice = tool_choice_raw
    else:
        raise _refuse("tool_choice to be a string, an object, or null", tool_choice_raw)

    sampling = SamplingRequest(
        temperature=_optional_float(body.get("temperature"), "temperature to be a number"),
        top_p=_optional_float(body.get("top_p"), "top_p to be a number"),
        top_k=_optional_int(body.get("top_k"), "top_k to be an integer"),
        max_tokens=_optional_int(
            body.get("max_output_tokens"), "max_output_tokens to be an integer"
        ),
        stop=_stop_tuple(body.get("stop")),
        n=_optional_int(body.get("n"), "n to be an integer"),
        seed=_optional_int(body.get("seed"), "seed to be an integer"),
    )

    stream = _optional_bool(body.get("stream"), "stream to be a boolean") or False

    for key in (
        "metadata",
        "user",
        "store",
        "previous_response_id",
        "background",
        "include",
        "parallel_tool_calls",
        "reasoning",
        "text",
        "truncation",
    ):
        if key in body:
            extras.append(key)

    return _validate_request(
        CanonicalRequest(
            format=WireFormat.RESPONSES,
            model=model,
            messages=tuple(messages),
            tools=tuple(tools),
            tool_choice=tool_choice,
            sampling=sampling,
            stream=stream,
            has_images=has_images,
            extras_dropped=tuple(extras),
        )
    )


# ---------------------------------------------------------------------------
# ANTHROPIC MESSAGES
# ---------------------------------------------------------------------------


def _parse_anthropic(body: Mapping[str, Any]) -> CanonicalRequest:
    extras: list[str] = []
    has_images = False

    model = _optional_model(body.get("model"))

    messages: list[CanonicalMessage] = []

    system_raw = body.get("system")
    if system_raw is not None:
        if isinstance(system_raw, str):
            messages.append(_validate_message(CanonicalMessage(role="system", content=system_raw)))
        else:
            blocks = _require_sequence(system_raw, "system to be a string or a list of text blocks")
            parts: list[str] = []
            for b in blocks:
                blk = _require_mapping(b, "each system block to be an object")
                btype = blk.get("type")
                if btype != "text":
                    raise _refuse("system block 'type' to be 'text'", btype)
                parts.append(_require_str(blk.get("text"), "system block 'text' to be a string"))
            messages.append(
                _validate_message(
                    CanonicalMessage(role="system", content="".join(parts) if parts else None)
                )
            )

    raw_messages = body.get("messages")
    raw_messages = _require_sequence(raw_messages, "messages to be a list")
    for raw in raw_messages:
        item = _require_mapping(raw, "each message to be an object")
        role = _require_str(item.get("role"), "message 'role' to be a string")
        if role not in ("user", "assistant"):
            raise _refuse("message 'role' to be 'user' or 'assistant'", role)

        content_raw = item.get("content")
        if isinstance(content_raw, str):
            messages.append(_validate_message(CanonicalMessage(role=role, content=content_raw)))
            continue

        blocks = _require_sequence(
            content_raw, "message 'content' to be a string or a list of blocks"
        )
        text_parts: list[str] = []
        reasoning_parts: list[str] = []
        tool_calls: list[CanonicalToolCall] = []
        tool_messages: list[CanonicalMessage] = []
        for b in blocks:
            blk = _require_mapping(b, "each content block to be an object")
            btype = blk.get("type")

            if btype == "text":
                text_parts.append(_require_str(blk.get("text"), "text block 'text' to be a string"))

            elif btype == "tool_use":
                if role != "assistant":
                    raise _refuse("tool_use blocks only on assistant messages", role)
                call_id = _require_str(blk.get("id"), "tool_use 'id' to be a string")
                name = _require_str(blk.get("name"), "tool_use 'name' to be a string")
                input_obj = blk.get("input")
                if input_obj is None:
                    raise _refuse("tool_use 'input' to be a JSON object", None)
                input_obj = _require_mapping(input_obj, "tool_use 'input' to be a JSON object")
                arguments_raw = json.dumps(input_obj, separators=(",", ":"), ensure_ascii=False)
                tool_calls.append(
                    CanonicalToolCall(
                        call_id=call_id,
                        name=name,
                        arguments_raw=arguments_raw,
                        call_id_origin="model",
                    )
                )

            elif btype == "tool_result":
                tool_use_id = _require_str(
                    blk.get("tool_use_id"), "tool_result 'tool_use_id' to be a string"
                )
                result_raw = blk.get("content")
                if result_raw is None:
                    result_content: str | None = None
                elif isinstance(result_raw, str):
                    result_content = result_raw
                else:
                    rblocks = _require_sequence(
                        result_raw,
                        "tool_result 'content' to be a string or a list of text blocks",
                    )
                    rparts: list[str] = []
                    for rb in rblocks:
                        rblk = _require_mapping(
                            rb, "each tool_result content block to be an object"
                        )
                        rbtype = rblk.get("type")
                        if rbtype != "text":
                            raise _refuse("tool_result content block 'type' to be 'text'", rbtype)
                        rparts.append(
                            _require_str(
                                rblk.get("text"), "tool_result block 'text' to be a string"
                            )
                        )
                    result_content = "".join(rparts) if rparts else None
                tool_messages.append(
                    _validate_message(
                        CanonicalMessage(
                            role="tool",
                            content=result_content,
                            tool_call_id=tool_use_id,
                        )
                    )
                )

            elif btype == "thinking":
                thinking_raw = blk.get("thinking")
                if thinking_raw is None:
                    thinking_raw = blk.get("text")
                reasoning_parts.append(
                    _require_str(thinking_raw, "thinking block 'thinking' to be a string")
                )

            elif btype == "image":
                has_images = True

            else:
                raise _refuse(
                    "content block 'type' to be 'text', 'tool_use', 'tool_result', "
                    "'thinking' or 'image'",
                    btype,
                )

        content = "".join(text_parts) if text_parts else None
        reasoning = "".join(reasoning_parts) if reasoning_parts else None
        # tool_result blocks answer the PREVIOUS assistant turn's tool_use, so they come
        # before any text the user wrote in the same message (OpenAI ordering).
        messages.extend(tool_messages)
        if tool_calls or content is not None or reasoning is not None:
            messages.append(
                _validate_message(
                    CanonicalMessage(
                        role=role,
                        content=content,
                        tool_calls=tuple(tool_calls),
                        reasoning=reasoning,
                    )
                )
            )

    tools_raw = body.get("tools")
    tools: list[CanonicalTool] = []
    if tools_raw is not None:
        seq = _require_sequence(tools_raw, "tools to be a list")
        for raw in seq:
            t = _require_mapping(raw, "each tool to be an object")
            name = _require_str(t.get("name"), "tool 'name' to be a string")
            description = _optional_str(t.get("description"), "tool 'description' to be a string")
            schema = t.get("input_schema")
            if schema is None:
                raise _refuse("tool 'input_schema' to be a JSON schema object", None)
            schema = _require_mapping(schema, "tool 'input_schema' to be a JSON schema object")
            tools.append(CanonicalTool(name=name, description=description, parameters=schema))

    tool_choice_raw = body.get("tool_choice")
    tool_choice: str | Mapping[str, Any] | None
    if tool_choice_raw is None:
        tool_choice = None
    elif isinstance(tool_choice_raw, (str, Mapping)):
        tool_choice = tool_choice_raw
    else:
        raise _refuse("tool_choice to be a string, an object, or null", tool_choice_raw)

    sampling = SamplingRequest(
        temperature=_optional_float(body.get("temperature"), "temperature to be a number"),
        top_p=_optional_float(body.get("top_p"), "top_p to be a number"),
        top_k=_optional_int(body.get("top_k"), "top_k to be an integer"),
        max_tokens=_optional_int(body.get("max_tokens"), "max_tokens to be an integer"),
        stop=_stop_tuple(body.get("stop_sequences")),
        n=None,
        seed=_optional_int(body.get("seed"), "seed to be an integer"),
    )

    stream = _optional_bool(body.get("stream"), "stream to be a boolean") or False

    for key in ("metadata", "user"):
        if key in body:
            extras.append(key)

    return _validate_request(
        CanonicalRequest(
            format=WireFormat.ANTHROPIC_MESSAGES,
            model=model,
            messages=tuple(messages),
            tools=tuple(tools),
            tool_choice=tool_choice,
            sampling=sampling,
            stream=stream,
            has_images=has_images,
            extras_dropped=tuple(extras),
        )
    )


# ---------------------------------------------------------------------------
# GEMINI
# ---------------------------------------------------------------------------


def _parse_gemini(body: Mapping[str, Any]) -> CanonicalRequest:
    extras: list[str] = []
    has_images = False

    model = _optional_model(body.get("model"))

    messages: list[CanonicalMessage] = []

    system_instruction = body.get("systemInstruction")
    if system_instruction is not None:
        si = _require_mapping(system_instruction, "systemInstruction to be an object")
        parts_raw = si.get("parts")
        parts_raw = _require_sequence(parts_raw, "systemInstruction 'parts' to be a list")
        sparts: list[str] = []
        for p in parts_raw:
            part = _require_mapping(p, "each systemInstruction part to be an object")
            sparts.append(
                _require_str(part.get("text"), "systemInstruction part 'text' to be a string")
            )
        messages.append(
            _validate_message(
                CanonicalMessage(role="system", content="".join(sparts) if sparts else None)
            )
        )

    contents_raw = body.get("contents")
    contents_raw = _require_sequence(contents_raw, "contents to be a list")

    # Running index for synthesized call ids, and a stack of unmatched calls
    # (call_id, name) used to pair functionResponse with its functionCall.
    call_index = 0
    unmatched: list[tuple[str, str]] = []

    for raw in contents_raw:
        item = _require_mapping(raw, "each content item to be an object")
        role_raw = item.get("role")
        if role_raw is None:
            raise _refuse("content 'role' to be 'user' or 'model'", None)
        role_raw = _require_str(role_raw, "content 'role' to be a string")
        if role_raw == "user":
            role = "user"
        elif role_raw == "model":
            role = "assistant"
        else:
            raise _refuse("content 'role' to be 'user' or 'model'", role_raw)

        parts_raw = item.get("parts")
        parts_raw = _require_sequence(parts_raw, "content 'parts' to be a list")

        text_parts: list[str] = []
        tool_calls: list[CanonicalToolCall] = []
        tool_messages: list[CanonicalMessage] = []
        for p in parts_raw:
            part = _require_mapping(p, "each content part to be an object")

            if "text" in part:
                text_parts.append(_require_str(part.get("text"), "part 'text' to be a string"))
                continue

            if "functionCall" in part:
                if role != "assistant":
                    raise _refuse("functionCall parts only on 'model' contents", role_raw)
                fc = _require_mapping(part.get("functionCall"), "functionCall to be an object")
                name = _require_str(fc.get("name"), "functionCall 'name' to be a string")
                args_obj = fc.get("args")
                if args_obj is None:
                    args_obj = {}
                args_obj = _require_mapping(args_obj, "functionCall 'args' to be an object")
                arguments_raw = json.dumps(args_obj, separators=(",", ":"), ensure_ascii=False)
                call_id = f"gemini:{call_index}"
                call_index += 1
                unmatched.append((call_id, name))
                tool_calls.append(
                    CanonicalToolCall(
                        call_id=call_id,
                        name=name,
                        arguments_raw=arguments_raw,
                        call_id_origin="synthesized",
                    )
                )
                continue

            if "functionResponse" in part:
                fr = _require_mapping(
                    part.get("functionResponse"), "functionResponse to be an object"
                )
                name = _require_str(fr.get("name"), "functionResponse 'name' to be a string")
                response_obj = fr.get("response")
                if response_obj is None:
                    raise _refuse("functionResponse 'response' to be a JSON object", None)
                response_obj = _require_mapping(
                    response_obj, "functionResponse 'response' to be a JSON object"
                )
                matched_id: str | None = None
                for idx in range(len(unmatched) - 1, -1, -1):
                    if unmatched[idx][1] == name:
                        matched_id = unmatched[idx][0]
                        del unmatched[idx]
                        break
                if matched_id is None:
                    raise WireRefusal(
                        "expected a preceding unmatched functionCall with name "
                        f"{name!r} to pair this functionResponse with, got none "
                        f"(unmatched call names: {[n for _, n in unmatched]!r})"
                    )
                tool_messages.append(
                    _validate_message(
                        CanonicalMessage(
                            role="tool",
                            content=json.dumps(
                                response_obj, separators=(",", ":"), ensure_ascii=False
                            ),
                            tool_call_id=matched_id,
                            name=name,
                        )
                    )
                )
                continue

            if "inlineData" in part or "fileData" in part:
                has_images = True
                continue

            raise _refuse(
                "content part to contain 'text', 'functionCall', 'functionResponse', "
                "'inlineData' or 'fileData'",
                sorted(part.keys()),
            )

        content = "".join(text_parts) if text_parts else None
        # functionResponse parts answer the previous model turn's calls: emit them first.
        messages.extend(tool_messages)
        if tool_calls or content is not None:
            messages.append(
                _validate_message(
                    CanonicalMessage(role=role, content=content, tool_calls=tuple(tool_calls))
                )
            )

    tools_raw = body.get("tools")
    tools: list[CanonicalTool] = []
    if tools_raw is not None:
        seq = _require_sequence(tools_raw, "tools to be a list")
        for raw in seq:
            t = _require_mapping(raw, "each tool to be an object")
            decls = t.get("functionDeclarations")
            decls = _require_sequence(
                decls, "tool 'functionDeclarations' to be a list of declarations"
            )
            for d in decls:
                decl = _require_mapping(d, "each function declaration to be an object")
                name = _require_str(decl.get("name"), "function declaration 'name' to be a string")
                description = _optional_str(
                    decl.get("description"), "function declaration 'description' to be a string"
                )
                parameters = decl.get("parameters")
                if parameters is None:
                    raise _refuse(
                        "function declaration 'parameters' to be a JSON schema object", None
                    )
                parameters = _require_mapping(
                    parameters, "function declaration 'parameters' to be a JSON schema object"
                )
                tools.append(
                    CanonicalTool(name=name, description=description, parameters=parameters)
                )

    tool_choice_raw = body.get("tool_choice")
    tool_choice: str | Mapping[str, Any] | None
    if tool_choice_raw is None:
        tool_choice = None
    elif isinstance(tool_choice_raw, (str, Mapping)):
        tool_choice = tool_choice_raw
    else:
        raise _refuse("tool_choice to be a string, an object, or null", tool_choice_raw)

    gen = body.get("generationConfig")
    if gen is None:
        gen = {}
    gen = _require_mapping(gen, "generationConfig to be an object")

    sampling = SamplingRequest(
        temperature=_optional_float(gen.get("temperature"), "temperature to be a number"),
        top_p=_optional_float(gen.get("topP"), "topP to be a number"),
        top_k=_optional_int(gen.get("topK"), "topK to be an integer"),
        max_tokens=_optional_int(gen.get("maxOutputTokens"), "maxOutputTokens to be an integer"),
        stop=_stop_tuple(gen.get("stopSequences")),
        n=_optional_int(gen.get("candidateCount"), "candidateCount to be an integer"),
        seed=_optional_int(gen.get("seed"), "seed to be an integer"),
    )

    stream = _optional_bool(body.get("stream"), "stream to be a boolean") or False

    for key in ("safetySettings", "cachedContent"):
        if key in body:
            extras.append(key)

    return _validate_request(
        CanonicalRequest(
            format=WireFormat.GEMINI,
            model=model,
            messages=tuple(messages),
            tools=tuple(tools),
            tool_choice=tool_choice,
            sampling=sampling,
            stream=stream,
            has_images=has_images,
            extras_dropped=tuple(extras),
        )
    )


# ---------------------------------------------------------------------------
# render_response / render_stream (PART 2)
# ---------------------------------------------------------------------------


def _compact_json(value: Any) -> str:
    """Serialise ``value`` as compact JSON with non-ASCII kept literal."""

    return json.dumps(value, separators=(",", ":"), ensure_ascii=False)


def _sse_data(payload: Any) -> str:
    """One SSE ``data:`` frame carrying a JSON payload."""

    return f"data: {_compact_json(payload)}\n\n"


def _sse_event(name: str, payload: Any) -> str:
    """One SSE named-event frame carrying a JSON payload."""

    # Anthropic and OpenAI Responses both repeat the event name as "type" inside the
    # JSON payload; SDK clients dispatch on that field, not on the SSE event line.
    if isinstance(payload, dict) and "type" not in payload:
        payload = {"type": name, **payload}
    return f"event: {name}\ndata: {_compact_json(payload)}\n\n"


def _check_result(res: CanonicalResult) -> CanonicalResult:
    """Validate the engine result before rendering it."""

    if not isinstance(res, CanonicalResult):
        raise _refuse("res to be a CanonicalResult", res)
    _check_finish_reason(res.finish_reason)
    if res.text is not None and not isinstance(res.text, str):
        raise _refuse("text to be a string or None", res.text)
    if res.reasoning is not None and not isinstance(res.reasoning, str):
        raise _refuse("reasoning to be a string or None", res.reasoning)
    for call in res.tool_calls:
        _check_call_id_origin(call.call_id_origin)
    return res


def _check_request(req: CanonicalRequest) -> CanonicalRequest:
    if not isinstance(req, CanonicalRequest):
        raise _refuse("req to be a CanonicalRequest", req)
    return _validate_request(req)


def _anthropic_stop_reason(finish_reason: str) -> str:
    return {"stop": "end_turn", "length": "max_tokens", "tool_calls": "tool_use"}[finish_reason]


def _gemini_finish_reason(finish_reason: str) -> str:
    return {"stop": "STOP", "length": "MAX_TOKENS", "tool_calls": "STOP"}[finish_reason]


def _responses_status(finish_reason: str) -> str:
    return "incomplete" if finish_reason == "length" else "completed"


def _anthropic_tool_input(arguments_raw: str) -> Mapping[str, Any]:
    """Parse ``arguments_raw`` into the JSON object Anthropic's ``input`` needs."""

    try:
        parsed = json.loads(arguments_raw) if arguments_raw else {}
    except json.JSONDecodeError as exc:
        raise WireRefusal(
            "expected tool call 'arguments_raw' to be valid JSON so it can be "
            f"rendered as an Anthropic tool_use 'input' object, got {arguments_raw!r} "
            f"(json error: {exc})"
        ) from exc
    if not isinstance(parsed, Mapping):
        raise WireRefusal(
            "expected tool call 'arguments_raw' to decode to a JSON object for "
            f"Anthropic tool_use 'input', got {type(parsed).__name__}"
        )
    return parsed


def _gemini_function_args(arguments_raw: str) -> Mapping[str, Any]:
    """Parse ``arguments_raw`` into the JSON object Gemini's ``args`` needs."""

    try:
        parsed = json.loads(arguments_raw) if arguments_raw else {}
    except json.JSONDecodeError as exc:
        raise WireRefusal(
            "expected tool call 'arguments_raw' to be valid JSON so it can be "
            f"rendered as a Gemini functionCall 'args' object, got {arguments_raw!r} "
            f"(json error: {exc})"
        ) from exc
    if not isinstance(parsed, Mapping):
        raise WireRefusal(
            "expected tool call 'arguments_raw' to decode to a JSON object for "
            f"Gemini functionCall 'args', got {type(parsed).__name__}"
        )
    return parsed


def render_response(fmt: WireFormat, req: CanonicalRequest, res: CanonicalResult) -> dict[str, Any]:
    """Render one finished engine result in the client's wire format."""

    if not isinstance(fmt, WireFormat):
        raise _refuse("fmt to be a WireFormat", fmt)
    _check_request(req)
    _check_result(res)

    if fmt is WireFormat.CHAT_COMPLETIONS:
        return _render_chat_response(req, res)
    if fmt is WireFormat.RESPONSES:
        return _render_responses_response(req, res)
    if fmt is WireFormat.ANTHROPIC_MESSAGES:
        return _render_anthropic_response(req, res)
    return _render_gemini_response(req, res)


def _render_chat_response(req: CanonicalRequest, res: CanonicalResult) -> dict[str, Any]:
    message: dict[str, Any] = {"role": "assistant", "content": res.text}
    if res.tool_calls:
        message["tool_calls"] = [
            {
                "id": call.call_id,
                "type": "function",
                "function": {"name": call.name, "arguments": call.arguments_raw},
            }
            for call in res.tool_calls
        ]
    if res.reasoning is not None:
        message["reasoning_content"] = res.reasoning

    return {
        "id": res.response_id,
        "object": "chat.completion",
        "created": res.created,
        "model": req.model,
        "choices": [{"index": 0, "message": message, "finish_reason": res.finish_reason}],
        "usage": {
            "prompt_tokens": res.prompt_tokens,
            "completion_tokens": res.completion_tokens,
            "total_tokens": res.prompt_tokens + res.completion_tokens,
        },
    }


def _render_responses_response(req: CanonicalRequest, res: CanonicalResult) -> dict[str, Any]:
    output: list[dict[str, Any]] = []
    if res.reasoning is not None:
        output.append(
            {"type": "reasoning", "summary": [{"type": "summary_text", "text": res.reasoning}]}
        )
    if res.text is not None:
        output.append(
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": res.text}],
            }
        )
    for call in res.tool_calls:
        output.append(
            {
                "type": "function_call",
                "call_id": call.call_id,
                "name": call.name,
                "arguments": call.arguments_raw,
            }
        )

    body: dict[str, Any] = {
        "id": res.response_id,
        "object": "response",
        "created_at": res.created,
        "model": req.model,
        "status": _responses_status(res.finish_reason),
        "output": output,
        "usage": {
            "input_tokens": res.prompt_tokens,
            "output_tokens": res.completion_tokens,
            "total_tokens": res.prompt_tokens + res.completion_tokens,
        },
    }
    if res.finish_reason == "length":
        body["incomplete_details"] = {"reason": "max_output_tokens"}
    return body


def _render_anthropic_response(req: CanonicalRequest, res: CanonicalResult) -> dict[str, Any]:
    content: list[dict[str, Any]] = []
    if res.reasoning is not None:
        content.append({"type": "thinking", "thinking": res.reasoning})
    if res.text is not None:
        content.append({"type": "text", "text": res.text})
    for call in res.tool_calls:
        content.append(
            {
                "type": "tool_use",
                "id": call.call_id,
                "name": call.name,
                "input": _anthropic_tool_input(call.arguments_raw),
            }
        )

    return {
        "id": res.response_id,
        "type": "message",
        "role": "assistant",
        "model": req.model,
        "content": content,
        "stop_reason": _anthropic_stop_reason(res.finish_reason),
        "stop_sequence": None,
        "usage": {
            "input_tokens": res.prompt_tokens,
            "output_tokens": res.completion_tokens,
        },
    }


def _render_gemini_response(req: CanonicalRequest, res: CanonicalResult) -> dict[str, Any]:
    parts: list[dict[str, Any]] = []
    if res.text is not None:
        parts.append({"text": res.text})
    for call in res.tool_calls:
        parts.append(
            {
                "functionCall": {
                    "name": call.name,
                    "args": _gemini_function_args(call.arguments_raw),
                }
            }
        )

    return {
        "candidates": [
            {
                "content": {"role": "model", "parts": parts},
                "finishReason": _gemini_finish_reason(res.finish_reason),
                "index": 0,
            }
        ],
        "usageMetadata": {
            "promptTokenCount": res.prompt_tokens,
            "candidatesTokenCount": res.completion_tokens,
            "totalTokenCount": res.prompt_tokens + res.completion_tokens,
        },
        "modelVersion": req.model,
    }


def render_stream(fmt: WireFormat, req: CanonicalRequest, res: CanonicalResult) -> list[str]:
    """Render the complete SSE event sequence for one finished result.

    Each list item is one complete SSE frame (``"data: ...\\n\\n"`` or
    ``"event: X\\ndata: ...\\n\\n"``). A standard client reassembles the
    sequence into exactly :func:`render_response`'s content. Chunking
    granularity is one text chunk plus one chunk per tool call.
    """

    if not isinstance(fmt, WireFormat):
        raise _refuse("fmt to be a WireFormat", fmt)
    _check_request(req)
    _check_result(res)

    if fmt is WireFormat.CHAT_COMPLETIONS:
        return _render_chat_stream(req, res)
    if fmt is WireFormat.RESPONSES:
        return _render_responses_stream(req, res)
    if fmt is WireFormat.ANTHROPIC_MESSAGES:
        return _render_anthropic_stream(req, res)
    return _render_gemini_stream(req, res)


def _chat_chunk(
    req: CanonicalRequest,
    res: CanonicalResult,
    delta: dict[str, Any],
    finish_reason: str | None = None,
) -> dict[str, Any]:
    return {
        "id": res.response_id,
        "object": "chat.completion.chunk",
        "created": res.created,
        "model": req.model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }


def _render_chat_stream(req: CanonicalRequest, res: CanonicalResult) -> list[str]:
    events: list[str] = [_sse_data(_chat_chunk(req, res, {"role": "assistant"}))]

    if res.reasoning is not None:
        events.append(_sse_data(_chat_chunk(req, res, {"reasoning_content": res.reasoning})))
    if res.text is not None:
        events.append(_sse_data(_chat_chunk(req, res, {"content": res.text})))
    for call in res.tool_calls:
        events.append(
            _sse_data(
                _chat_chunk(
                    req,
                    res,
                    {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": call.call_id,
                                "type": "function",
                                "function": {
                                    "name": call.name,
                                    "arguments": call.arguments_raw,
                                },
                            }
                        ]
                    },
                )
            )
        )

    events.append(_sse_data(_chat_chunk(req, res, {}, finish_reason=res.finish_reason)))
    events.append("data: [DONE]\n\n")
    return events


def _render_responses_stream(req: CanonicalRequest, res: CanonicalResult) -> list[str]:
    events: list[str] = []

    created = {
        "id": res.response_id,
        "object": "response",
        "created_at": res.created,
        "model": req.model,
        "status": "in_progress",
    }
    events.append(_sse_event("response.created", created))

    item_index = 0

    if res.reasoning is not None:
        item = {
            "type": "reasoning",
            "id": f"{res.response_id}_reasoning_{item_index}",
            "summary": [{"type": "summary_text", "text": res.reasoning}],
        }
        events.append(
            _sse_event(
                "response.output_item.added",
                {
                    "type": "response.output_item.added",
                    "output_index": item_index,
                    "item": {"type": "reasoning", "id": item["id"], "summary": []},
                },
            )
        )
        events.append(
            _sse_event(
                "response.output_item.done",
                {"type": "response.output_item.done", "output_index": item_index, "item": item},
            )
        )
        item_index += 1

    if res.text is not None:
        item = {
            "type": "message",
            "id": f"{res.response_id}_message_{item_index}",
            "role": "assistant",
            "content": [{"type": "output_text", "text": res.text}],
        }
        events.append(
            _sse_event(
                "response.output_item.added",
                {
                    "type": "response.output_item.added",
                    "output_index": item_index,
                    "item": {
                        "type": "message",
                        "id": item["id"],
                        "role": "assistant",
                        "content": [],
                    },
                },
            )
        )
        events.append(
            _sse_event(
                "response.output_text.delta",
                {
                    "type": "response.output_text.delta",
                    "output_index": item_index,
                    "content_index": 0,
                    "delta": res.text,
                },
            )
        )
        events.append(
            _sse_event(
                "response.output_text.done",
                {
                    "type": "response.output_text.done",
                    "output_index": item_index,
                    "content_index": 0,
                    "text": res.text,
                },
            )
        )
        events.append(
            _sse_event(
                "response.output_item.done",
                {"type": "response.output_item.done", "output_index": item_index, "item": item},
            )
        )
        item_index += 1

    for call in res.tool_calls:
        item = {
            "type": "function_call",
            "id": f"{res.response_id}_function_call_{item_index}",
            "call_id": call.call_id,
            "name": call.name,
            "arguments": call.arguments_raw,
        }
        events.append(
            _sse_event(
                "response.output_item.added",
                {
                    "type": "response.output_item.added",
                    "output_index": item_index,
                    "item": {
                        "type": "function_call",
                        "id": item["id"],
                        "call_id": call.call_id,
                        "name": call.name,
                        "arguments": "",
                    },
                },
            )
        )
        events.append(
            _sse_event(
                "response.function_call_arguments.delta",
                {
                    "type": "response.function_call_arguments.delta",
                    "output_index": item_index,
                    "delta": call.arguments_raw,
                },
            )
        )
        events.append(
            _sse_event(
                "response.function_call_arguments.done",
                {
                    "type": "response.function_call_arguments.done",
                    "output_index": item_index,
                    "arguments": call.arguments_raw,
                },
            )
        )
        events.append(
            _sse_event(
                "response.output_item.done",
                {"type": "response.output_item.done", "output_index": item_index, "item": item},
            )
        )
        item_index += 1

    completed = {
        "id": res.response_id,
        "object": "response",
        "created_at": res.created,
        "model": req.model,
        "status": _responses_status(res.finish_reason),
        "output": _render_responses_response(req, res)["output"],
        "usage": {
            "input_tokens": res.prompt_tokens,
            "output_tokens": res.completion_tokens,
            "total_tokens": res.prompt_tokens + res.completion_tokens,
        },
    }
    if res.finish_reason == "length":
        completed["incomplete_details"] = {"reason": "max_output_tokens"}
    events.append(_sse_event("response.completed", completed))
    return events


def _render_anthropic_stream(req: CanonicalRequest, res: CanonicalResult) -> list[str]:
    events: list[str] = []

    message_start = {
        "type": "message_start",
        "message": {
            "id": res.response_id,
            "type": "message",
            "role": "assistant",
            "model": req.model,
            "content": [],
            "stop_reason": None,
            "stop_sequence": None,
            "usage": {"input_tokens": res.prompt_tokens, "output_tokens": 0},
        },
    }
    events.append(_sse_event("message_start", message_start))

    block_index = 0

    if res.reasoning is not None:
        events.append(
            _sse_event(
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": block_index,
                    "content_block": {"type": "thinking", "thinking": ""},
                },
            )
        )
        events.append(
            _sse_event(
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": block_index,
                    "delta": {"type": "thinking_delta", "thinking": res.reasoning},
                },
            )
        )
        events.append(
            _sse_event("content_block_stop", {"type": "content_block_stop", "index": block_index})
        )
        block_index += 1

    if res.text is not None:
        events.append(
            _sse_event(
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": block_index,
                    "content_block": {"type": "text", "text": ""},
                },
            )
        )
        events.append(
            _sse_event(
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": block_index,
                    "delta": {"type": "text_delta", "text": res.text},
                },
            )
        )
        events.append(
            _sse_event("content_block_stop", {"type": "content_block_stop", "index": block_index})
        )
        block_index += 1

    for call in res.tool_calls:
        events.append(
            _sse_event(
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": block_index,
                    "content_block": {
                        "type": "tool_use",
                        "id": call.call_id,
                        "name": call.name,
                        "input": {},
                    },
                },
            )
        )
        events.append(
            _sse_event(
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": block_index,
                    "delta": {"type": "input_json_delta", "partial_json": call.arguments_raw},
                },
            )
        )
        events.append(
            _sse_event("content_block_stop", {"type": "content_block_stop", "index": block_index})
        )
        block_index += 1

    events.append(
        _sse_event(
            "message_delta",
            {
                "type": "message_delta",
                "delta": {
                    "stop_reason": _anthropic_stop_reason(res.finish_reason),
                    "stop_sequence": None,
                },
                "usage": {"output_tokens": res.completion_tokens},
            },
        )
    )
    events.append(_sse_event("message_stop", {"type": "message_stop"}))
    return events


def _render_gemini_stream(req: CanonicalRequest, res: CanonicalResult) -> list[str]:
    """Gemini streaming: one or more GenerateContentResponse JSON objects.

    Chunking granularity is one text chunk plus one chunk per tool call; each
    chunk carries the same cumulative ``usageMetadata`` and ``modelVersion``.
    """

    chunks: list[dict[str, Any]] = []

    if res.text is not None:
        chunks.append({"text": res.text})
    for call in res.tool_calls:
        chunks.append(
            {
                "functionCall": {
                    "name": call.name,
                    "args": _gemini_function_args(call.arguments_raw),
                }
            }
        )
    if not chunks:
        chunks.append({})

    events: list[str] = []
    for parts in chunks:
        events.append(
            _sse_data(
                {
                    "candidates": [
                        {
                            "content": {"role": "model", "parts": [parts]},
                            "finishReason": _gemini_finish_reason(res.finish_reason),
                            "index": 0,
                        }
                    ],
                    "usageMetadata": {
                        "promptTokenCount": res.prompt_tokens,
                        "candidatesTokenCount": res.completion_tokens,
                        "totalTokenCount": res.prompt_tokens + res.completion_tokens,
                    },
                    "modelVersion": req.model,
                }
            )
        )
    return events
