"""Parse Gemma-4 tool calls: ``<|tool_call>call:NAME{key:value,...}<tool_call|>``.

The layout is the one Gemma-4's chat template renders for an assistant ``tool_calls`` entry:
keys are bare, strings sit between ``<|"|>`` delimiters with no escaping, and numbers,
``true``/``false``/``null``, objects and lists are written bare. Because a string may contain
anything -- braces, commas, even the literal close marker -- every scan here is string-aware: a
close marker inside ``<|"|>...<|"|>`` does not end the call.

:class:`Gemma4ToolCallParser` satisfies ``tools.ToolCallParser`` and never raises: a malformed
call is a :class:`~foundationscale.agentic_rl.tools.ParsedCall` whose ``error`` names what was
wrong. Visible text drops every call span plus the turn markers Gemma-4 emits around tool use
(``<|tool_response>``, ``<turn|>``, ``<eos>``, ``<|turn>model``) so a reward reading the final
answer never sees protocol tokens.
"""

from __future__ import annotations

import json
import re
from typing import Any

from foundationscale.agentic_rl.tools import ParsedCall

TOOL_CALL_OPEN = "<|tool_call>"
TOOL_CALL_CLOSE = "<tool_call|>"
STR_DELIM = '<|"|>'
_CALL_PREFIX = "call:"
_MAX_DEPTH = 32
_TURN_MARKERS = ("<|turn>model\n", "<|tool_response>", "<turn|>", "<eos>")
_NAME = re.compile(r"[A-Za-z0-9_.\-]+")
_KEY = re.compile(r"[A-Za-z_][A-Za-z0-9_.\-]*")
_NUMBER = re.compile(r"-?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+-]?\d+)?")


class _Malformed(Exception):
    """Internal: a parse error with a message; never escapes this module."""


def _close_index(text: str, start: int) -> int:
    """Index of the first close marker at or after ``start`` outside any string, or -1."""
    i = start
    while i < len(text):
        if text.startswith(STR_DELIM, i):
            end = text.find(STR_DELIM, i + len(STR_DELIM))
            if end == -1:
                return -1
            i = end + len(STR_DELIM)
        elif text.startswith(TOOL_CALL_CLOSE, i):
            return i
        else:
            i += 1
    return -1


class _Values:
    """Recursive-descent reader over one call body."""

    def __init__(self, text: str) -> None:
        self.text = text
        self.pos = 0

    def _skip_ws(self) -> None:
        while self.pos < len(self.text) and self.text[self.pos] in " \t\r\n":
            self.pos += 1

    def _expect(self, char: str) -> None:
        self._skip_ws()
        if not self.text.startswith(char, self.pos):
            found = self.text[self.pos : self.pos + 20]
            raise _Malformed(f"expected {char!r} at offset {self.pos}, found {found!r}")
        self.pos += 1

    def _string(self) -> str:
        start = self.pos + len(STR_DELIM)
        end = self.text.find(STR_DELIM, start)
        if end == -1:
            raise _Malformed(f"unterminated string starting at offset {self.pos}")
        self.pos = end + len(STR_DELIM)
        return self.text[start:end]

    def _key(self) -> str:
        self._skip_ws()
        if self.text.startswith(STR_DELIM, self.pos):
            return self._string()
        match = _KEY.match(self.text, self.pos)
        if match is None:
            raise _Malformed(f"expected a key at offset {self.pos}")
        self.pos = match.end()
        return match.group(0)

    def obj(self, depth: int) -> dict[str, Any]:
        if depth > _MAX_DEPTH:
            raise _Malformed(f"nesting deeper than {_MAX_DEPTH}")
        self._expect("{")
        out: dict[str, Any] = {}
        self._skip_ws()
        if self.text.startswith("}", self.pos):
            self.pos += 1
            return out
        while True:
            key = self._key()
            self._expect(":")
            value = self.value(depth + 1)
            if key in out:
                raise _Malformed(f"duplicate key {key!r}: last-wins would discard an argument")
            out[key] = value
            self._skip_ws()
            if self.text.startswith(",", self.pos):
                self.pos += 1
                continue
            self._expect("}")
            return out

    def _list(self, depth: int) -> list[Any]:
        if depth > _MAX_DEPTH:
            raise _Malformed(f"nesting deeper than {_MAX_DEPTH}")
        self._expect("[")
        out: list[Any] = []
        self._skip_ws()
        if self.text.startswith("]", self.pos):
            self.pos += 1
            return out
        while True:
            out.append(self.value(depth + 1))
            self._skip_ws()
            if self.text.startswith(",", self.pos):
                self.pos += 1
                continue
            self._expect("]")
            return out

    def value(self, depth: int) -> Any:
        self._skip_ws()
        rest = self.text[self.pos :]
        if rest.startswith(STR_DELIM):
            return self._string()
        if rest.startswith("{"):
            return self.obj(depth)
        if rest.startswith("["):
            return self._list(depth)
        for word, literal in (("true", True), ("false", False), ("null", None)):
            if rest.startswith(word):
                self.pos += len(word)
                return literal
        match = _NUMBER.match(self.text, self.pos)
        if match is None:
            raise _Malformed(f"bad value at offset {self.pos}: {rest[:20]!r}")
        self.pos = match.end()
        return json.loads(match.group(0))


def _parse_span(raw: str) -> ParsedCall:
    inner = raw[len(TOOL_CALL_OPEN) : -len(TOOL_CALL_CLOSE)]
    if not inner.startswith(_CALL_PREFIX):
        return ParsedCall(
            name="",
            arguments=None,
            raw=raw,
            error=f"malformed tool_call body: expected 'call:NAME{{...}}', found {inner[:40]!r}",
        )
    match = _NAME.match(inner, len(_CALL_PREFIX))
    if match is None:
        return ParsedCall(
            name="",
            arguments=None,
            raw=raw,
            error="missing function name after 'call:' -- an unnamed call cannot be routed",
        )
    name = match.group(0)
    reader = _Values(inner[match.end() :])
    try:
        arguments = reader.obj(depth=1)
        reader._skip_ws()
        if reader.pos != len(reader.text):
            trailing = reader.text[reader.pos : reader.pos + 20]
            raise _Malformed(f"trailing text {trailing!r} after the argument object")
    except _Malformed as exc:
        return ParsedCall(name=name, arguments=None, raw=raw, error=f"malformed arguments: {exc}")
    except (RecursionError, ValueError) as exc:  # json.loads of a matched number cannot fail
        return ParsedCall(name=name, arguments=None, raw=raw, error=f"malformed arguments: {exc}")
    return ParsedCall(name=name, arguments=arguments, raw=raw, error=None)


class Gemma4ToolCallParser:
    """Parse Gemma-4 tool calls out of assistant text. Never raises."""

    def parse(self, text: str) -> tuple[list[ParsedCall], str]:
        """Return parsed calls in order and ``text`` without call spans or turn markers."""
        calls: list[ParsedCall] = []
        pieces: list[str] = []
        cursor = 0
        while True:
            start = text.find(TOOL_CALL_OPEN, cursor)
            if start == -1:
                pieces.append(text[cursor:])
                break
            pieces.append(text[cursor:start])
            close = _close_index(text, start + len(TOOL_CALL_OPEN))
            if close == -1:
                calls.append(
                    ParsedCall(
                        name="",
                        arguments=None,
                        raw=text[start:],
                        error=(
                            f"unterminated tool_call: {TOOL_CALL_OPEN!r} with no "
                            f"{TOOL_CALL_CLOSE!r} outside a string after it -- the call span "
                            f"cannot be delimited"
                        ),
                    )
                )
                break
            end = close + len(TOOL_CALL_CLOSE)
            calls.append(_parse_span(text[start:end]))
            cursor = end
        visible = "".join(pieces)
        for marker in _TURN_MARKERS:
            visible = visible.replace(marker, "")
        return calls, visible.strip()
