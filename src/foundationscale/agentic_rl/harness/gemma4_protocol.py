"""Gemma-4 inline tool observations: tool results continue the model's own turn.

Gemma-4 does not answer a tool call with a new chat message. The model writes
``<|tool_call>call:NAME{...}<tool_call|>`` and stops on ``<|tool_response>`` (one of its
end-of-generation ids). The result is then written INLINE, in the same model turn, as
``response:NAME{value:<|"|>TEXT<|"|>}<tool_response|>``, and the model continues writing with
no new turn header. This is the layout its chat template renders for an assistant message with
``tool_calls`` followed by ``tool`` messages.

The template cannot be used to produce these tokens by diffing renders: a ``tool`` message after
a plain assistant message renders to NOTHING (measured with the real E4B tokenizer), so
:class:`~foundationscale.agentic_rl.harness.native_tool_loop.NativeToolLoop` would silently drop
every tool result. :class:`Gemma4InlineObservations` builds the observation ids directly:
the marker tokens by id, and the tool text through an encoder that never turns text into special
tokens. A tool output that contains a literal ``<|"|>`` therefore stays plain text and cannot close
the string early or forge a marker. The ids are fed to the engine as ids, so the trained tokens
are exactly what the model saw.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

TOOL_RESPONSE_OPEN = "<|tool_response>"
TOOL_RESPONSE_CLOSE = "<tool_response|>"
STRING_DELIMITER = '<|"|>'
_MARKERS = (TOOL_RESPONSE_OPEN, TOOL_RESPONSE_CLOSE, STRING_DELIMITER)


class Gemma4ProtocolRefusal(ValueError):
    """A tokenizer or call that cannot express the Gemma-4 inline tool protocol."""


def _described(value: object) -> str:
    return f"{type(value).__name__} {value!r}"


@dataclass(frozen=True)
class Gemma4InlineObservations:
    """Build the token ids of one turn's tool results in Gemma-4's inline layout.

    ``encode_plain`` must tokenize text WITHOUT recognising special tokens (see
    :func:`from_hf_tokenizer`). ``open_id``/``close_id``/``string_id`` are the single ids of
    ``<|tool_response>``, ``<tool_response|>`` and ``<|"|>``.
    """

    encode_plain: Callable[[str], Sequence[int]]
    open_id: int
    close_id: int
    string_id: int

    def __post_init__(self) -> None:
        if not callable(self.encode_plain):
            raise Gemma4ProtocolRefusal(
                f"Gemma4InlineObservations: field 'encode_plain' is "
                f"{_described(self.encode_plain)}: it must be callable"
            )
        ids = {"open_id": self.open_id, "close_id": self.close_id, "string_id": self.string_id}
        for field, value in ids.items():
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise Gemma4ProtocolRefusal(
                    f"Gemma4InlineObservations: field {field!r} is {_described(value)}: "
                    f"expected a non-negative token id"
                )
        if len(set(ids.values())) != 3:
            raise Gemma4ProtocolRefusal(
                f"Gemma4InlineObservations: marker ids {ids} are not distinct: the tokenizer "
                f"does not separate the tool-response markers"
            )

    def _plain(self, text: str) -> list[int]:
        ids = list(self.encode_plain(text))
        forged = {self.open_id, self.close_id, self.string_id}.intersection(ids)
        if forged:
            raise Gemma4ProtocolRefusal(
                f"Gemma4InlineObservations: encode_plain turned text into marker id(s) "
                f"{sorted(forged)}: tool text must never become protocol tokens"
            )
        return ids

    def encode(self, results: Sequence[tuple[str, str]], *, opened: bool) -> list[int]:
        """Ids of ``results`` -- ``(tool_name, text)`` per call, in call order.

        ``opened=True`` when the generation already ended with ``<|tool_response>`` (the model's
        stop id), so the first block's opening marker is not repeated.
        """
        if not results:
            raise Gemma4ProtocolRefusal(
                "Gemma4InlineObservations.encode: no results: an inline observation needs at "
                "least one tool result"
            )
        ids: list[int] = []
        for index, (name, text) in enumerate(results):
            if not isinstance(name, str) or not name or not isinstance(text, str):
                raise Gemma4ProtocolRefusal(
                    f"Gemma4InlineObservations.encode: result {index} is "
                    f"{_described((name, text))}: expected (non-empty tool name, text)"
                )
            if index > 0 or not opened:
                ids.append(self.open_id)
            ids += self._plain(f"response:{name}{{value:")
            ids.append(self.string_id)
            ids += self._plain(text)
            ids.append(self.string_id)
            ids += self._plain("}")
            ids.append(self.close_id)
        return ids


def from_hf_tokenizer(tokenizer: Any) -> Gemma4InlineObservations:
    """Resolve the marker ids from a Hugging Face tokenizer and refuse if it lacks any."""
    ids: list[int] = []
    for marker in _MARKERS:
        encoded = tokenizer.encode(marker, add_special_tokens=False)
        if len(encoded) != 1:
            raise Gemma4ProtocolRefusal(
                f"from_hf_tokenizer: {marker!r} encodes to {encoded}: the tokenizer has no single "
                f"token for it, so it does not speak the Gemma-4 tool protocol"
            )
        ids.append(int(encoded[0]))

    def encode_plain(text: str) -> list[int]:
        return [
            int(i)
            for i in tokenizer.encode(text, add_special_tokens=False, split_special_tokens=True)
        ]

    return Gemma4InlineObservations(
        encode_plain=encode_plain, open_id=ids[0], close_id=ids[1], string_id=ids[2]
    )
