from __future__ import annotations

import copy
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Protocol

from foundationscale.agentic_rl import markup

__all__ = [
    "ToolSpec",
    "ToolSpecRefusal",
    "ParsedCall",
    "ToolCallParser",
    "QwenXmlToolCallParser",
    "HermesJsonToolCallParser",
    "validate_call",
    "default_tools",
]


class ToolSpecRefusal(ValueError):
    """Raised when a ToolSpec cannot be constructed from the given fields."""


_TOOL_NAME_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _described(value: object) -> str:
    return f"{type(value).__name__} {value!r}"


def _freeze_json(value: Any) -> Any:
    """Recursively freeze a JSON-shaped value: dicts -> MappingProxyType, lists/tuples -> tuple.

    Scalars are left as-is. Best-effort: any other nested container type is left as-is
    rather than refused, so this never raises.
    """
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze_json(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json(item) for item in value)
    return value


def _unfreeze_json(value: Any) -> Any:
    """Inverse of _freeze_json: produce plain dict/list JSON-able structures."""
    if isinstance(value, Mapping):
        return {key: _unfreeze_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_unfreeze_json(item) for item in value]
    return copy.deepcopy(value)


@dataclass(frozen=True)
class ToolSpec:
    """A tool contract: name, description, JSON-schema parameters, required keys.

    Construction refuses a name that is not identifier-like (so any template can route
    it), an empty description, a non-Mapping parameters schema, or duplicate/empty
    required parameter names.
    """

    name: str
    description: str
    parameters: Mapping[str, Any]
    required: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not _TOOL_NAME_PATTERN.match(self.name):
            raise ToolSpecRefusal(
                f"ToolSpec: field 'name' is {_described(self.name)}: every tool name must "
                f"match the pattern {_TOOL_NAME_PATTERN.pattern!r} -- an unparseable name "
                f"cannot be routed by any template"
            )
        if not isinstance(self.description, str) or self.description == "":
            raise ToolSpecRefusal(
                f"ToolSpec({self.name!r}): field 'description' is "
                f"{_described(self.description)}: every tool carries a non-empty str "
                f"description -- an undescribed tool cannot be chosen meaningfully by a model"
            )
        if not isinstance(self.parameters, Mapping):
            raise ToolSpecRefusal(
                f"ToolSpec({self.name!r}): field 'parameters' is "
                f"{_described(self.parameters)}: parameters must be a Mapping holding a "
                f"JSON-schema object -- anything else cannot describe call arguments"
            )
        object.__setattr__(self, "parameters", _freeze_json(dict(self.parameters)))

        declared_required: object = self.required
        if isinstance(declared_required, (str, bytes)) or not isinstance(
            declared_required, Sequence
        ):
            raise ToolSpecRefusal(
                f"ToolSpec({self.name!r}): field 'required' is {_described(declared_required)}: "
                f"required must be a Sequence[str] of parameter names -- a non-sequence "
                f"cannot enumerate which arguments a call must supply"
            )
        seen: set[str] = set()
        coerced: list[str] = []
        for item in declared_required:
            if not isinstance(item, str) or item == "":
                raise ToolSpecRefusal(
                    f"ToolSpec({self.name!r}): field 'required' contains "
                    f"{_described(item)}: every required entry is a non-empty str naming a "
                    f"parameter -- an unnamed requirement cannot be checked against a call"
                )
            if item in seen:
                raise ToolSpecRefusal(
                    f"ToolSpec({self.name!r}): field 'required' repeats parameter "
                    f"{item!r}: each required parameter is listed exactly once -- a duplicate "
                    f"would report the same missing argument twice"
                )
            seen.add(item)
            coerced.append(item)
        object.__setattr__(self, "required", tuple(coerced))

    def to_openai(self) -> dict[str, Any]:
        """Return the standard OpenAI tools-array entry for this spec.

        The frozen parameters mapping is unfrozen back to plain dict/list so callers
        receive ordinary JSON-able Python objects.
        """
        parameters: dict[str, Any] = _unfreeze_json(self.parameters)
        if self.required:
            parameters = {**parameters, "required": list(self.required)}
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": parameters,
            },
        }


@dataclass(frozen=True)
class ParsedCall:
    """One tool call as the model wrote it, plus a parse verdict.

    ``name`` is the tool name the model wrote and MAY be the empty string when the
    parser could not even find a name: this is the one place in this module that
    tolerates an empty name, because ParsedCall reports what the model wrote rather
    than asserting a valid call. ``arguments`` is a fresh plain dict of decoded
    parameters, or None when parsing failed too badly to produce any arguments.
    ``error`` is None when the call parsed and validated syntactically fine, else a
    non-empty str naming a model-attributable parse error. This type never raises.
    """

    name: str
    arguments: Mapping[str, Any] | None
    raw: str
    error: str | None


class ToolCallParser(Protocol):
    """Parse tool-call markup out of assistant text.

    Implementations MUST NEVER RAISE for malformed input: a malformed tool call is
    recorded as a ParsedCall with ``error`` set, never an exception, because the
    model's own output is adversarial-by-construction input that the harness must
    always be able to report on.
    """

    def parse(self, text: str) -> tuple[list[ParsedCall], str]: ...


def _strip_one_newline(value: str) -> str:
    """Strip exactly one leading and one trailing newline if present, nothing else."""
    if value.startswith("\n"):
        value = value[1:]
    if value.endswith("\n"):
        value = value[:-1]
    return value


def _decode_parameter_value(value: str) -> Any:
    """Decode one parameter value body.

    The body has already had one leading/trailing newline stripped. If it parses as
    JSON and the decoded result is a bool, int, float, dict or list, the decoded value
    is used; otherwise (JSON parse failure, or a decoded ``str``) the ORIGINAL stripped
    text is kept as a plain str. A bare JSON string is deliberately NOT decoded: the
    raw text itself IS the string representation the model intended.
    """
    try:
        decoded = json.loads(value)
    except (json.JSONDecodeError, ValueError):
        return value
    if type(decoded) in (bool, int, float, dict, list):
        return decoded
    return value


def _unterminated_call(raw: str) -> ParsedCall:
    return ParsedCall(
        name="",
        arguments=None,
        raw=raw,
        error=(
            f"unterminated tool_call: found {_described(markup.TOOL_CALL_OPEN)} with no "
            f"matching {_described(markup.TOOL_CALL_CLOSE)} after it -- an unclosed call "
            f"span cannot be delimited, so its body is reported as unparseable"
        ),
    )


class QwenXmlToolCallParser:
    """Parse the Qwen-XML tool-call wire layout out of assistant text.

    Never raises: every malformed call is reported as a ParsedCall with ``error`` set.
    """

    def __init__(self) -> None:
        open_re = re.escape(markup.TOOL_CALL_OPEN)
        close_re = re.escape(markup.TOOL_CALL_CLOSE)
        self._span_re = re.compile(open_re + r"(.*?)" + close_re, re.DOTALL)
        self._open_only_re = re.compile(open_re, re.DOTALL)
        fn_open = re.escape(markup.FUNCTION_OPEN_PREFIX)
        fn_close = re.escape(markup.FUNCTION_CLOSE)
        tag_close = re.escape(markup.TAG_CLOSE)
        self._function_re = re.compile(
            r"^\s*" + fn_open + r"(.*?)" + tag_close + r"(.*?)" + fn_close + r"\s*$",
            re.DOTALL,
        )
        param_open = re.escape(markup.PARAMETER_OPEN_PREFIX)
        param_close = re.escape(markup.PARAMETER_CLOSE)
        self._parameter_re = re.compile(
            param_open + r"(.*?)" + tag_close + r"(.*?)" + param_close,
            re.DOTALL,
        )

    def parse(self, text: str) -> tuple[list[ParsedCall], str]:
        """Return parsed calls in order and ``text`` with every call span removed."""
        calls: list[ParsedCall] = []
        pieces: list[str] = []
        cursor = 0
        for match in self._span_re.finditer(text):
            start, end = match.span()
            pieces.append(text[cursor:start])
            calls.append(self._parse_span(match.group(0)))
            cursor = end
        pieces.append(text[cursor:])
        cleaned = "".join(pieces)

        # Any TOOL_CALL_OPEN left in the cleaned tail has no matching close: report it
        # as one unterminated call and remove that trailing span from the visible text.
        # `cleaned` already holds the fully-joined text at this point, so the tail is
        # cut directly from it -- re-using the (already-consumed) `pieces` list here
        # would duplicate everything before the dangling open tag.
        tail_open = self._open_only_re.search(cleaned)
        if tail_open is not None:
            start = tail_open.start()
            calls.append(_unterminated_call(cleaned[start:]))
            cleaned = cleaned[:start]
        return calls, cleaned

    def _parse_span(self, raw: str) -> ParsedCall:
        inner = raw[len(markup.TOOL_CALL_OPEN) : -len(markup.TOOL_CALL_CLOSE)]
        fn_match = self._function_re.match(inner)
        if fn_match is None:
            return ParsedCall(
                name="",
                arguments=None,
                raw=raw,
                error=(
                    f"malformed tool_call body: expected "
                    f"{_described(markup.FUNCTION_OPEN_PREFIX)} + name + "
                    f"{_described(markup.TAG_CLOSE)} ... {_described(markup.FUNCTION_CLOSE)} "
                    f"but found {_described(inner)} -- without the function wrapper the call "
                    f"has neither a routable name nor delimited arguments"
                ),
            )
        name = fn_match.group(1)
        body = fn_match.group(2)
        error: str | None = None
        if name == "":
            error = (
                f"missing function name: found {_described(markup.FUNCTION_OPEN_PREFIX)} "
                f"immediately followed by {_described(markup.TAG_CLOSE)} -- an unnamed call "
                f"cannot be routed to any tool"
            )
        arguments: dict[str, Any] = {}
        cursor = 0
        for param_match in self._parameter_re.finditer(body):
            stray = body[cursor : param_match.start()]
            if stray.strip() != "":
                # Whitespace/newlines between parameter tags are tolerated (the wire
                # layout allows them); only non-whitespace stray text is an error.
                error = error or (
                    f"unclosed parameter tag: stray text {_described(stray)} "
                    f"before a {_described(markup.PARAMETER_OPEN_PREFIX)} group -- an unclosed "
                    f"parameter cannot be delimited from the rest of the body"
                )
            key = param_match.group(1)
            value = _strip_one_newline(param_match.group(2))
            if key in arguments:
                error = error or (
                    f"duplicate parameter key {key!r}: the same parameter appears more than "
                    f"once in one function body -- last-wins would silently discard an argument"
                )
            else:
                arguments[key] = _decode_parameter_value(value)
            cursor = param_match.end()
        trailing = body[cursor:]
        if trailing.strip() != "":
            # Trailing whitespace/newlines before </function> are tolerated too.
            error = error or (
                f"unclosed parameter tag: trailing text {_described(trailing)} after the last "
                f"complete parameter -- an unterminated parameter body cannot be delimited"
            )
        return ParsedCall(name=name, arguments=arguments, raw=raw, error=error)


class HermesJsonToolCallParser:
    """Parse the Hermes-JSON tool-call wire layout (a JSON object inside the call tags).

    Never raises: every malformed call is reported as a ParsedCall with ``error`` set.
    """

    def __init__(self) -> None:
        open_re = re.escape(markup.TOOL_CALL_OPEN)
        close_re = re.escape(markup.TOOL_CALL_CLOSE)
        self._span_re = re.compile(open_re + r"(.*?)" + close_re, re.DOTALL)
        self._open_only_re = re.compile(open_re, re.DOTALL)

    def parse(self, text: str) -> tuple[list[ParsedCall], str]:
        """Return parsed calls in order and ``text`` with every call span removed."""
        calls: list[ParsedCall] = []
        pieces: list[str] = []
        cursor = 0
        for match in self._span_re.finditer(text):
            start, end = match.span()
            pieces.append(text[cursor:start])
            calls.append(self._parse_span(match.group(0)))
            cursor = end
        pieces.append(text[cursor:])
        cleaned = "".join(pieces)

        # `cleaned` already holds the fully-joined text at this point, so the tail is
        # cut directly from it -- re-using the (already-consumed) `pieces` list here
        # would duplicate everything before the dangling open tag.
        tail_open = self._open_only_re.search(cleaned)
        if tail_open is not None:
            start = tail_open.start()
            calls.append(_unterminated_call(cleaned[start:]))
            cleaned = cleaned[:start]
        return calls, cleaned

    def _parse_span(self, raw: str) -> ParsedCall:
        body = raw[len(markup.TOOL_CALL_OPEN) : -len(markup.TOOL_CALL_CLOSE)]
        try:
            decoded = json.loads(body)
        except json.JSONDecodeError as exc:
            return ParsedCall(
                name="",
                arguments=None,
                raw=raw,
                error=(
                    f"malformed tool_call JSON: {exc.msg} at position {exc.pos} -- the call "
                    f"body must be a JSON object naming a tool and its arguments"
                ),
            )
        if not isinstance(decoded, dict):
            return ParsedCall(
                name="",
                arguments=None,
                raw=raw,
                error=(
                    f"malformed tool_call JSON: decoded body is {_described(decoded)}: the "
                    f"call body must be a JSON object -- a non-object names no tool"
                ),
            )
        if "name" not in decoded:
            return ParsedCall(
                name="",
                arguments=None,
                raw=raw,
                error=(
                    "missing function name: decoded body has no 'name' key -- an unnamed call "
                    "cannot be routed to any tool"
                ),
            )
        raw_name = decoded["name"]
        if not isinstance(raw_name, str) or raw_name == "":
            return ParsedCall(
                name=str(raw_name),
                arguments=None,
                raw=raw,
                error=(
                    f"malformed function name: 'name' is {_described(raw_name)}: the name must "
                    f"be a non-empty str -- a non-string name cannot be routed to any tool"
                ),
            )
        if "arguments" not in decoded:
            return ParsedCall(name=raw_name, arguments={}, raw=raw, error=None)
        arguments = decoded["arguments"]
        if not isinstance(arguments, dict):
            return ParsedCall(
                name=raw_name,
                arguments=None,
                raw=raw,
                error=(
                    f"malformed arguments: 'arguments' is {_described(arguments)}: arguments "
                    f"must be a JSON object -- a non-object cannot name call parameters"
                ),
            )
        return ParsedCall(name=raw_name, arguments=arguments, raw=raw, error=None)


def validate_call(call: ParsedCall, tools: Mapping[str, ToolSpec]) -> str | None:
    """Return None if ``call`` is acceptable to execute, else a short error string.

    Never raises. Checks, short-circuiting at the first failure: parse error, unknown
    tool, missing/invalid arguments, missing required parameters, unexpected parameters
    when the schema forbids them, and per-property JSON-schema type mismatches.
    """
    if call.error is not None:
        return call.error
    if call.name not in tools:
        known = ", ".join(sorted(tools)) or "<none>"
        return (
            f"unknown tool {call.name!r}: known tools are {known} -- an unroutable call "
            f"cannot be executed"
        )
    if not isinstance(call.arguments, Mapping):
        return (
            f"tool {call.name!r}: arguments are {_described(call.arguments)}: a call supplies "
            f"a Mapping of parameters -- no/invalid arguments cannot be dispatched"
        )
    spec = tools[call.name]
    for required_name in spec.required:
        if required_name not in call.arguments:
            return (
                f"tool {call.name!r}: missing required parameter {required_name!r} -- the "
                f"schema declares it required, so a call without it cannot be executed"
            )
    parameters = dict(spec.parameters)
    properties = parameters.get("properties", {})
    if not isinstance(properties, Mapping):
        properties = {}
    if parameters.get("additionalProperties") is False:
        for key in call.arguments:
            if key not in properties:
                return (
                    f"tool {call.name!r}: unexpected parameter {key!r}: the schema sets "
                    f"'additionalProperties' to False -- an undeclared argument cannot be "
                    f"interpreted"
                )
    for key, value in call.arguments.items():
        if key not in properties:
            continue
        schema = properties[key]
        if not isinstance(schema, Mapping):
            continue
        expected = schema.get("type")
        if expected == "string":
            ok = isinstance(value, str)
        elif expected == "integer":
            ok = type(value) is int
        elif expected == "number":
            ok = type(value) in (int, float)
        elif expected == "boolean":
            ok = type(value) is bool
        elif expected == "object":
            ok = isinstance(value, dict)
        elif expected == "array":
            ok = isinstance(value, list)
        else:
            continue
        if not ok:
            return (
                f"tool {call.name!r}: parameter {key!r} is {_described(value)}: the schema "
                f"declares type {expected!r} -- a mismatched argument type cannot be used by "
                f"the tool"
            )
    return None


def default_tools() -> dict[str, ToolSpec]:
    """Return the built-in bash/submit tool specs keyed by name."""
    return {
        "bash": ToolSpec(
            name="bash",
            description="Execute a shell command and return its output.",
            parameters={
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The shell command to execute.",
                    }
                },
                "additionalProperties": False,
            },
            required=("command",),
        ),
        "submit": ToolSpec(
            name="submit",
            description="Submit the final answer and end the episode.",
            parameters={
                "type": "object",
                "properties": {
                    "answer": {
                        "type": "string",
                        "description": "The final answer.",
                    }
                },
                "additionalProperties": False,
            },
            required=("answer",),
        ),
    }
