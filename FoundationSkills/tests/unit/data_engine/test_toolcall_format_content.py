"""Tests for ``toolcall_format`` message content normalisation and optional tool params."""
from __future__ import annotations

import json

from foundationskills.skills.data_engine.ops.base import OPS, OpStats
from foundationskills.skills.data_engine.ops.toolcall_format import (
    FUNCTIONCALL_TAG,
    TC_CLOSE,
    TC_OPEN,
    TR_CLOSE,
    TR_OPEN,
    _norm_params,
)


def run(records, cfg=None):
    stats = OpStats(name="toolcall_format")
    out = list(OPS["toolcall_format"](iter(records), cfg or {}, stats))
    return out, stats


PARAMS = {
    "type": "object",
    "properties": {"city": {"type": "string", "description": "City name"}},
    "required": ["city"],
}
TOOLS = [{"type": "function", "function": {"name": "get_weather", "description": "Get weather", "parameters": PARAMS}}]
CALL = {"id": "call_1", "type": "function", "function": {"name": "get_weather", "arguments": '{"city": "x"}'}}
XSTYLE_PARAMS = {"city": {"type": "str", "description": "c"}, "unit": {"type": "str, optional"}}
XSTYLE_TOOLS = [{"name": "get_weather", "description": "w", "parameters": XSTYLE_PARAMS}]


def _openai_rec(content):
    """Minimal valid openai record whose user message content is ``content``."""
    return {
        "id": "rc",
        "messages": [
            {"role": "user", "content": content},
            {"role": "assistant", "content": None, "tool_calls": [CALL]},
            {"role": "tool", "tool_call_id": "call_1", "content": "ok"},
        ],
        "tools": TOOLS,
    }


def test_text_parts_content_joined():
    rec = _openai_rec([{"type": "text", "text": "weather in "}, {"type": "text", "text": "Paris?"}])
    out, stats = run([rec])
    assert sum(stats.dropped.values()) == 0
    assert out[0]["messages"][0]["content"] == "weather in Paris?"


def test_content_parts_with_image_dropped_as_malformed_message():
    img = {"type": "image_url", "image_url": {"url": "http://x/y.png"}}
    rec = _openai_rec([{"type": "text", "text": "see "}, img])
    out, stats = run([rec])
    assert out == []
    assert stats.dropped["malformed_message"] == 1
    assert stats.records_in == stats.records_out + sum(stats.dropped.values())


def test_non_string_content_dropped_as_malformed_message():
    rec = _openai_rec(42)
    out, stats = run([rec])
    assert out == []
    assert stats.dropped["malformed_message"] == 1


def test_list_of_plain_strings_dropped_as_malformed_message():
    rec = _openai_rec(["weather in ", "Paris?"])
    out, stats = run([rec])
    assert out == []
    assert stats.dropped["malformed_message"] == 1


def test_text_part_without_string_text_dropped_as_malformed_message():
    rec = _openai_rec([{"type": "text", "text": 7}])
    out, stats = run([rec])
    assert out == []
    assert stats.dropped["malformed_message"] == 1


def test_none_content_still_accepted():
    rec = _openai_rec(None)
    out, stats = run([rec])
    assert sum(stats.dropped.values()) == 0
    assert out[0]["messages"][0]["content"] == ""


def test_strict_false_keeps_malformed_message_with_meta_status():
    rec = _openai_rec(123)
    out, stats = run([rec], {"strict": False})
    assert len(out) == 1
    assert out[0]["meta"]["toolcall_status"] == "invalid:malformed_message"
    assert stats.extra["kept_invalid"]["malformed_message"] == 1
    assert stats.records_in == stats.records_out + sum(stats.dropped.values())


def test_hermes_text_parts_content_joined():
    call_blob = json.dumps({"name": "get_weather", "arguments": {"city": "Rome"}})
    res_blob = json.dumps({"tool_call_id": "call_x", "content": "hot"})
    rec = {
        "id": "h1",
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": "weather in "}, {"type": "text", "text": "Rome?"}]},
            {"role": "assistant", "content": TC_OPEN + call_blob + TC_CLOSE},
            {"role": "tool", "content": TR_OPEN + res_blob + TR_CLOSE},
        ],
        "tools": TOOLS,
    }
    out, stats = run([rec])
    assert sum(stats.dropped.values()) == 0
    msgs = out[0]["messages"]
    assert msgs[0] == {"role": "user", "content": "weather in Rome?"}
    assert msgs[2] == {"role": "tool", "tool_call_id": "call_x", "content": "hot"}


def test_sharegpt_non_string_value_dropped_as_malformed_message():
    rec = {"id": "s1", "messages": [{"from": "human", "value": 7}], "tools": json.dumps(TOOLS)}
    out, stats = run([rec])
    assert out == []
    assert stats.dropped["malformed_message"] == 1


def test_norm_params_type_optional_suffix_marks_param_optional():
    decl = {"city": {"type": "str"}, "unit": {"type": "str, optional"}, "limit": {"type": "int, optional"}}
    params = _norm_params(decl)
    assert params == {
        "type": "object",
        "properties": {"city": {"type": "string"}, "unit": {"type": "string"}, "limit": {"type": "integer"}},
        "required": ["city"],
    }


def test_xlam_optional_type_param_may_be_absent_from_args():
    rec = {
        "id": "t7x",
        "query": "weather?",
        "tools": json.dumps(XSTYLE_TOOLS),
        "answers": json.dumps([{"name": "get_weather", "arguments": {"city": "Lima"}}]),
    }
    out, stats = run([rec])
    assert sum(stats.dropped.values()) == 0
    assert out[0]["tools"][0]["function"]["parameters"]["required"] == ["city"]


def test_glaive_optional_type_param_may_be_absent_from_args():
    chat = (
        "USER: weather?\n"
        f"{FUNCTIONCALL_TAG} {json.dumps({'name': 'get_weather', 'arguments': {'city': 'Berlin'}})}\n"
        "FUNCTION RESPONSE: rainy\n"
        "ASSISTANT: Bring an umbrella."
    )
    rec = {"id": "t7g", "system": "assistant bot\nfunctions: " + json.dumps(XSTYLE_TOOLS), "chat": chat}
    out, stats = run([rec])
    assert sum(stats.dropped.values()) == 0
    assert out[0]["tools"][0]["function"]["parameters"]["required"] == ["city"]
