"""FIFO pairing of results with parallel tool calls in the sharegpt/glaive converters."""
from __future__ import annotations

import json

from foundationskills.skills.data_engine.ops.base import OPS, OpStats
from foundationskills.skills.data_engine.ops.toolcall_format import FUNCTIONCALL_TAG


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


def test_sharegpt_parallel_calls_each_observation_answers_one_call_fifo():
    rec = {
        "id": "pf1",
        "messages": [
            {"from": "human", "value": "weather in Oslo and Rome?"},
            {"from": "function_call", "value": json.dumps({"name": "get_weather", "arguments": {"city": "Oslo"}})},
            {"from": "function_call", "value": json.dumps({"name": "get_weather", "arguments": {"city": "Rome"}})},
            {"from": "observation", "value": "cold"},
            {"from": "observation", "value": "hot"},
            {"from": "gpt", "value": "Oslo is cold, Rome is hot."},
        ],
        "tools": json.dumps(TOOLS),
    }
    out, stats = run([rec])
    assert out and stats.records_out == 1 and sum(stats.dropped.values()) == 0
    msgs = out[0]["messages"]
    calls = msgs[1]["tool_calls"]
    assert [c["function"]["arguments"] for c in calls] == [canon({"city": "Oslo"}), canon({"city": "Rome"})]
    results = [m for m in msgs if m["role"] == "tool"]
    assert [r["content"] for r in results] == ["cold", "hot"]
    assert [r["tool_call_id"] for r in results] == [calls[0]["id"], calls[1]["id"]]
    assert msgs[-1] == {"role": "assistant", "content": "Oslo is cold, Rome is hot."}


def test_glaive_parallel_function_calls_each_response_answers_one_call_fifo():
    chat = (
        "USER: weather in Oslo and Rome?\n"
        f'{FUNCTIONCALL_TAG} {{"name": "get_weather", "arguments": {{"city": "Oslo"}}}}\n'
        f'{FUNCTIONCALL_TAG} {{"name": "get_weather", "arguments": {{"city": "Rome"}}}}\n'
        "FUNCTION RESPONSE: cold\n"
        "FUNCTION RESPONSE: hot\n"
        "ASSISTANT: Oslo is cold, Rome is hot."
    )
    rec = {"id": "pf2", "system": "assistant bot\nfunctions: " + json.dumps(TOOLS), "chat": chat}
    out, stats = run([rec])
    assert out and stats.records_out == 1 and sum(stats.dropped.values()) == 0
    msgs = out[0]["messages"]
    calls = msgs[2]["tool_calls"]
    assert [c["function"]["arguments"] for c in calls] == [canon({"city": "Oslo"}), canon({"city": "Rome"})]
    results = [m for m in msgs if m["role"] == "tool"]
    assert [r["content"] for r in results] == ["cold", "hot"]
    assert [r["tool_call_id"] for r in results] == [calls[0]["id"], calls[1]["id"]]
    assert msgs[-1] == {"role": "assistant", "content": "Oslo is cold, Rome is hot."}
