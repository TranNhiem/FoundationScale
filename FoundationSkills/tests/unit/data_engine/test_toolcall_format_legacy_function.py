"""Tests for legacy OpenAI ``function_call``/``function`` traces in the ``toolcall_format`` op."""
from __future__ import annotations

import hashlib
import json

from foundationskills.skills.data_engine.ops.base import OPS, OpStats


def run(records, cfg=None):
    stats = OpStats(name="toolcall_format")
    out = list(OPS["toolcall_format"](iter(records), cfg or {}, stats))
    return out, stats


def canon(obj):
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


PARAMS = {
    "type": "object",
    "properties": {"city": {"type": "string", "description": "City name"}},
    "required": ["city"],
}
TOOLS = [{"type": "function", "function": {"name": "get_weather", "description": "Get weather", "parameters": PARAMS}}]


def test_legacy_function_result_pairs_with_generated_call_id():
    rec = {
        "id": "lf1",
        "messages": [
            {"role": "user", "content": "weather in Paris?"},
            {
                "role": "assistant",
                "content": None,
                "function_call": {"name": "get_weather", "arguments": '{"city": "Paris"}'},
            },
            {"role": "function", "name": "get_weather", "content": "sunny"},
        ],
        "tools": TOOLS,
    }
    out, stats = run([rec])
    assert len(out) == 1
    assert not stats.dropped
    assert stats.records_in == stats.records_out + sum(stats.dropped.values())
    msgs = out[0]["messages"]
    expected = "call_" + hashlib.sha256(b"fs|lf1|1|0").hexdigest()[:12]
    assert msgs[1] == {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": expected,
                "type": "function",
                "function": {"name": "get_weather", "arguments": canon({"city": "Paris"})},
            }
        ],
    }
    assert msgs[2] == {"role": "tool", "tool_call_id": expected, "content": "sunny"}


def test_legacy_function_results_pair_with_oldest_pending_call():
    rec = {
        "id": "lf2",
        "messages": [
            {"role": "user", "content": "weather?"},
            {"role": "assistant", "content": None, "function_call": {"name": "get_weather", "arguments": '{"city": "Paris"}'}},
            {"role": "assistant", "content": None, "function_call": {"name": "get_weather", "arguments": '{"city": "Oslo"}'}},
            {"role": "function", "name": "get_weather", "content": "sunny"},
            {"role": "function", "name": "get_weather", "content": "cold"},
        ],
        "tools": TOOLS,
    }
    out, _ = run([rec])
    msgs = out[0]["messages"]
    first = msgs[1]["tool_calls"][0]["id"]
    second = msgs[2]["tool_calls"][0]["id"]
    assert first != second
    assert msgs[3] == {"role": "tool", "tool_call_id": first, "content": "sunny"}
    assert msgs[4] == {"role": "tool", "tool_call_id": second, "content": "cold"}


def test_function_result_without_any_call_is_still_dropped():
    rec = {
        "id": "lf3",
        "messages": [
            {"role": "user", "content": "q"},
            {"role": "function", "name": "get_weather", "content": "sunny"},
        ],
        "tools": TOOLS,
    }
    out, stats = run([rec])
    assert out == []
    assert stats.dropped["result_without_call"] == 1


def test_openai_result_with_unmatched_explicit_id_is_not_repaired():
    rec = {
        "id": "lf4",
        "messages": [
            {"role": "user", "content": "q"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {"id": "call_a", "type": "function", "function": {"name": "get_weather", "arguments": '{"city": "Paris"}'}}
                ],
            },
            {"role": "tool", "tool_call_id": "call_b", "content": "sunny"},
        ],
        "tools": TOOLS,
    }
    out, stats = run([rec])
    assert out == []
    assert stats.dropped["result_without_call"] == 1
