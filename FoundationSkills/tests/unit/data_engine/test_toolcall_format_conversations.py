"""Tests for the ``toolcall_format`` op: the ShareGPT ``conversations`` field is accepted."""
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


def minted_id(rec_id, turn, k):
    return "call_" + hashlib.sha256(f"fs|{rec_id}|{turn}|{k}".encode("utf-8")).hexdigest()[:12]


def test_conversations_sharegpt_detected_and_converted():
    rec = {
        "id": "c1",
        "conversations": [
            {"from": "human", "value": "weather?"},
            {"from": "function_call", "value": json.dumps({"name": "get_weather", "arguments": {"city": "Oslo"}})},
            {"from": "observation", "value": "cold"},
            {"from": "gpt", "value": "It is cold."},
        ],
        "tools": json.dumps(TOOLS),
    }
    out, stats = run([rec])
    assert sum(stats.dropped.values()) == 0
    msgs = out[0]["messages"]
    assert msgs[0] == {"role": "user", "content": "weather?"}
    call = msgs[1]["tool_calls"][0]
    assert call["id"] == minted_id("c1", 1, 0)
    assert call["function"]["arguments"] == canon({"city": "Oslo"})
    assert msgs[2] == {"role": "tool", "tool_call_id": call["id"], "content": "cold"}
    assert msgs[3] == {"role": "assistant", "content": "It is cold."}
    assert stats.extra["formats_detected"] == {"sharegpt": 1}
    assert stats.records_in == stats.records_out + sum(stats.dropped.values())


def test_conversations_openai_shape_plain_chat():
    rec = {
        "id": "c2",
        "conversations": [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ],
        "tools": TOOLS,
    }
    out, stats = run([rec])
    assert out[0]["messages"] == [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ]
    assert out[0]["tools"] == TOOLS
    assert stats.extra["formats_detected"] == {"openai": 1}


def test_conversations_with_tool_call_and_result():
    cid = "call_xyz"
    rec = {
        "id": "c4",
        "conversations": [
            {"role": "user", "content": "weather in Paris?"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {"id": cid, "type": "function", "function": {"name": "get_weather", "arguments": '{"city":"Paris"}'}}
                ],
            },
            {"role": "tool", "tool_call_id": cid, "content": "sunny"},
        ],
        "tools": TOOLS,
    }
    out, stats = run([rec])
    msgs = out[0]["messages"]
    assert msgs[1]["tool_calls"][0]["function"]["arguments"] == canon({"city": "Paris"})
    assert msgs[2] == {"role": "tool", "tool_call_id": cid, "content": "sunny"}
    assert stats.extra["formats_detected"] == {"openai": 1}


def test_messages_takes_precedence_over_conversations():
    rec = {
        "id": "c3",
        "messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}],
        "conversations": [{"from": "human", "value": "ignored"}],
        "tools": TOOLS,
    }
    out, stats = run([rec])
    assert out[0]["messages"] == [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]
    assert stats.extra["formats_detected"] == {"openai": 1}


def test_accounting_invariant_conversations_mixed():
    good = {
        "id": "g",
        "conversations": [
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "a"},
        ],
        "tools": TOOLS,
    }
    bad = {"id": "b", "text": "nope"}
    out, stats = run([good, bad])
    assert len(out) == 1
    assert stats.dropped.get("unknown_tool_format", 0) == 1
    assert stats.records_in == stats.records_out + sum(stats.dropped.values())
