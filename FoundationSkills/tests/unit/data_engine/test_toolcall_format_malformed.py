"""Tests for ``toolcall_format`` drops on unparsable payloads and non-dict entries."""
from __future__ import annotations

import json

from foundationskills.skills.data_engine.ops.base import OPS, OpStats
from foundationskills.skills.data_engine.ops.toolcall_format import MISTRAL_CALLS, MISTRAL_RESULTS


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

CALL = {"id": "c1", "type": "function", "function": {"name": "get_weather", "arguments": '{"city": "Oslo"}'}}


def _mistral_rec(calls_blob, results_blob="[]"):
    return {
        "id": "m",
        "messages": [
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": f"{MISTRAL_CALLS} {calls_blob} {MISTRAL_RESULTS} {results_blob}"},
        ],
        "tools": TOOLS,
    }


def _openai_rec(messages=None, tools=TOOLS):
    return {
        "id": "r",
        "messages": messages
        if messages is not None
        else [
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": None, "tool_calls": [CALL]},
            {"role": "tool", "tool_call_id": "c1", "content": "dry"},
        ],
        "tools": tools,
    }


def test_mistral_unparsable_call_payload_counts_malformed_tool_marker():
    out, stats = run([_mistral_rec("notjson", json.dumps([]))])
    assert out == []
    assert stats.dropped["malformed_tool_marker"] == 1
    assert stats.records_in == stats.records_out + sum(stats.dropped.values())


def test_mistral_non_dict_call_entry_dropped_malformed_message():
    calls_blob = json.dumps([{"name": "get_weather", "arguments": {"city": "Oslo"}, "id": "c1"}, "oops"])
    results_blob = json.dumps([{"id": "c1", "content": "dry"}])
    out, stats = run([_mistral_rec(calls_blob, results_blob)])
    assert out == []
    assert stats.dropped["malformed_message"] == 1


def test_xlam_unparsable_answers_dropped_malformed_message():
    rec = {"id": "x1", "query": "weather?", "tools": json.dumps(TOOLS), "answers": "[{bad"}
    out, stats = run([rec])
    assert out == []
    assert stats.dropped["malformed_message"] == 1
    assert stats.records_in == stats.records_out + sum(stats.dropped.values())


def test_xlam_single_dict_answer_still_accepted():
    rec = {
        "id": "x2",
        "query": "weather?",
        "tools": json.dumps(TOOLS),
        "answers": json.dumps({"name": "get_weather", "arguments": {"city": "Oslo"}}),
    }
    out, stats = run([rec])
    assert len(out) == 1 and not stats.dropped
    assert out[0]["messages"][1]["tool_calls"][0]["function"]["name"] == "get_weather"


def test_xlam_scalar_answers_dropped_malformed_message():
    rec = {"id": "x2s", "query": "weather?", "tools": json.dumps(TOOLS), "answers": "5"}
    out, stats = run([rec])
    assert out == []
    assert stats.dropped["malformed_message"] == 1


def test_xlam_non_dict_answer_entry_dropped_malformed_message():
    rec = {
        "id": "x3",
        "query": "weather?",
        "tools": json.dumps(TOOLS),
        "answers": json.dumps([{"name": "get_weather", "arguments": {"city": "Oslo"}}, "oops"]),
    }
    out, stats = run([rec])
    assert out == []
    assert stats.dropped["malformed_message"] == 1


def test_non_dict_message_entry_dropped_malformed_message():
    messages = [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": None, "tool_calls": [CALL]},
        {"role": "tool", "tool_call_id": "c1", "content": "dry"},
        "oops",
    ]
    out, stats = run([_openai_rec(messages)])
    assert out == []
    assert stats.dropped["malformed_message"] == 1
    assert stats.records_in == stats.records_out + sum(stats.dropped.values())


def test_non_dict_tool_call_entry_dropped_malformed_message():
    messages = [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": None, "tool_calls": [CALL, "oops"]},
        {"role": "tool", "tool_call_id": "c1", "content": "dry"},
    ]
    out, stats = run([_openai_rec(messages)])
    assert out == []
    assert stats.dropped["malformed_message"] == 1


def test_non_dict_tool_definition_dropped_malformed_message():
    out, stats = run([_openai_rec(tools=TOOLS + ["oops"])])
    assert out == []
    assert stats.dropped["malformed_message"] == 1
    assert stats.records_in == stats.records_out + sum(stats.dropped.values())
