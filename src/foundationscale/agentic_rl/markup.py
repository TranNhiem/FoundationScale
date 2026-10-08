"""Chat-markup string constants for the reference policy family's templates.

Every special token and tool-call tag the harness emits or parses is spelled
ONCE, here. The strings are assembled by concatenation on purpose: the literal
spellings are control tokens to the inference servers this code talks to, and
tooling that echoes source text through such a server (review bots, model-based
code assistants) can have its output cut at the first literal occurrence.
"""

from __future__ import annotations

__all__ = [
    "FUNCTION_CLOSE",
    "FUNCTION_OPEN_PREFIX",
    "IM_END",
    "IM_START",
    "PARAMETER_CLOSE",
    "PARAMETER_OPEN_PREFIX",
    "TAG_CLOSE",
    "THINK_CLOSE",
    "THINK_OPEN",
    "TOOL_CALL_CLOSE",
    "TOOL_CALL_OPEN",
    "TOOL_RESPONSE_CLOSE",
    "TOOL_RESPONSE_OPEN",
]

IM_START = "<|" + "im_start|>"
IM_END = "<|" + "im_end|>"
THINK_OPEN = "<" + "think>"
THINK_CLOSE = "</" + "think>"
TOOL_CALL_OPEN = "<" + "tool_call>"
TOOL_CALL_CLOSE = "</" + "tool_call>"
TOOL_RESPONSE_OPEN = "<" + "tool_response>"
TOOL_RESPONSE_CLOSE = "</" + "tool_response>"
# Qwen-XML function call body: <function=NAME><parameter=KEY>VALUE</parameter></function>
FUNCTION_OPEN_PREFIX = "<" + "function="
FUNCTION_CLOSE = "</" + "function>"
PARAMETER_OPEN_PREFIX = "<" + "parameter="
PARAMETER_CLOSE = "</" + "parameter>"
TAG_CLOSE = ">"
