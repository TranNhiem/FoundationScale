"""Tests for the ``toolcall_format`` op (all seven dialects, validation slugs)."""
from __future__ import annotations

import json

from foundationskills.core.schema import validate
from foundationskills.skills.data_engine.ops.base import OPS, OpStats
from foundationskills.skills.data_engine.ops.toolcall_format import (
    FUNCTIONCALL_TAG,
    MISTRAL_CALLS,
    MISTRAL_RESULTS,
    PY_TAG,
    TC_CLOSE,
    TC_OPEN,
    TOOLS_CLOSE,
    TOOLS_OPEN,
    TR_CLOSE,
    TR_OPEN,
    CONFIG_SCHEMA,
    preflight,
)


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


def test_openai_dialect_exact_canonical_shape():
    rec = {
        "id": "r1",
        "messages": [
            {"role": "user", "content": "weather in Paris?"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {"id": "call_abc", "type": "function", "function": {"name": "get_weather", "arguments": '{"city":"Paris"}'}}
                ],
            },
            {"role": "tool", "tool_call_id": "call_abc", "content": "sunny"},
        ],
        "tools": TOOLS,
    }
    out, stats = run([rec])
    assert out == [
        {
            "id": "r1",
            "messages": [
                {"role": "user", "content": "weather in Paris?"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{"id": "call_abc", "type": "function", "function": {"name": "get_weather", "arguments": canon({"city": "Paris"})}}],
                },
                {"role": "tool", "tool_call_id": "call_abc", "content": "sunny"},
            ],
            "tools": TOOLS,
        }
    ]
    assert stats.records_in == stats.records_out + sum(stats.dropped.values())


def test_sharegpt_dialect_deterministic_ids():
    rec = {
        "id": "r2",
        "messages": [
            {"from": "human", "value": "weather?"},
            {"from": "function_call", "value": json.dumps({"name": "get_weather", "arguments": {"city": "Oslo"}})},
            {"from": "observation", "value": "cold"},
            {"from": "gpt", "value": "It is cold."},
        ],
        "tools": json.dumps(TOOLS),
    }
    out, _ = run([rec])
    msgs = out[0]["messages"]
    assert msgs[0] == {"role": "user", "content": "weather?"}
    call = msgs[1]["tool_calls"][0]
    assert call["id"] == "call_" + __import__("hashlib").sha256(b"fs|r2|1|0").hexdigest()[:12]
    assert call["function"]["arguments"] == canon({"city": "Oslo"})
    assert msgs[2] == {"role": "tool", "tool_call_id": call["id"], "content": "cold"}
    assert msgs[3] == {"role": "assistant", "content": "It is cold."}


def test_hermes_dialect_markers():
    rec = {
        "id": "r3",
        "messages": [
            {"role": "system", "content": f"helper {TOOLS_OPEN}{json.dumps(TOOLS)}{TOOLS_CLOSE}"},
            {"role": "user", "content": "weather?"},
            {"role": "assistant", "content": f"checking {TC_OPEN}{json.dumps({'name': 'get_weather', 'arguments': {'city': 'Rome'}})}{TC_CLOSE}"},
            {"role": "tool", "content": f"{TR_OPEN}{json.dumps({'tool_call_id': 'call_x', 'content': 'hot'})}{TR_CLOSE}"},
        ],
    }
    out, _ = run([rec])
    msgs = out[0]["messages"]
    assert msgs[0]["role"] == "system" and TOOLS_OPEN not in msgs[0]["content"]
    assert msgs[2]["tool_calls"][0]["function"]["name"] == "get_weather"
    assert msgs[3] == {"role": "tool", "tool_call_id": "call_x", "content": "hot"}


def test_glaive_dialect_single_quoted_args():
    chat = (
        "USER: weather?\n"
        f"{FUNCTIONCALL_TAG} {{'name': 'get_weather', 'arguments': \"{{\\\"city\\\": \\\"Berlin\\\"}}\"}}\n"
        "FUNCTION RESPONSE: rainy\n"
        "ASSISTANT: Bring an umbrella."
    )
    rec = {"id": "r4", "system": "assistant bot\nfunctions: " + json.dumps(TOOLS), "chat": chat}
    out, _ = run([rec])
    msgs = out[0]["messages"]
    assert msgs[0] == {"role": "system", "content": "assistant bot"}
    assert msgs[1] == {"role": "user", "content": "weather?"}
    assert msgs[2]["tool_calls"][0]["function"]["arguments"] == canon({"city": "Berlin"})
    assert msgs[3]["role"] == "tool" and msgs[3]["content"] == "rainy"
    assert msgs[4] == {"role": "assistant", "content": "Bring an umbrella."}


def test_xlam_dialect_param_promotion():
    tools = [{"name": "get_weather", "description": "w", "parameters": {"city": {"type": "str", "description": "c"}, "unit": {"type": "str", "optional": True}}}]
    rec = {"id": "r5", "query": "weather?", "tools": json.dumps(tools), "answers": json.dumps([{"name": "get_weather", "arguments": {"city": "Lima"}}])}
    out, _ = run([rec])
    assert out[0]["tools"][0]["function"]["parameters"] == {
        "type": "object",
        "properties": {"city": {"type": "string", "description": "c"}, "unit": {"type": "string"}},
        "required": ["city"],
    }
    assert out[0]["messages"][1]["tool_calls"][0]["function"]["arguments"] == canon({"city": "Lima"})


def test_llama3_dialect_python_tag_and_ipython():
    rec = {
        "id": "r6",
        "messages": [
            {"role": "user", "content": "weather?"},
            {"role": "assistant", "content": PY_TAG + ' {"name": "get_weather", "parameters": {"city": "Kyoto"}}'},
            {"role": "ipython", "content": "humid"},
            {"role": "assistant", "content": "Humid in Kyoto."},
        ],
        "tools": TOOLS,
    }
    out, _ = run([rec])
    msgs = out[0]["messages"]
    assert msgs[1]["tool_calls"][0]["function"]["arguments"] == canon({"city": "Kyoto"})
    assert msgs[2] == {"role": "tool", "tool_call_id": msgs[1]["tool_calls"][0]["id"], "content": "humid"}


def test_mistral_dialect_calls_and_results():
    rec = {
        "id": "r7",
        "messages": [
            {"role": "user", "content": "weather?"},
            {
                "role": "assistant",
                "content": f"{MISTRAL_CALLS} [{{\"name\": \"get_weather\", \"arguments\": {{\"city\": \"Cairo\"}}, \"id\": \"call_m\"}}] {MISTRAL_RESULTS} [{{\"id\": \"call_m\", \"content\": \"dry\"}}]",
            },
        ],
        "tools": TOOLS,
    }
    out, _ = run([rec])
    msgs = out[0]["messages"]
    assert msgs[1]["tool_calls"][0]["id"] == "call_m"
    assert msgs[2] == {"role": "tool", "tool_call_id": "call_m", "content": "dry"}


def _bad_openai(**overrides):
    base = {
        "id": "bad",
        "messages": [
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": None, "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "get_weather", "arguments": '{"city": "x"}'}}]},
        ],
        "tools": TOOLS,
    }
    base.update(overrides)
    return base


def test_drop_slugs():
    cases = {
        "no_messages": {"id": "x", "messages": []},
        "unknown_tool_format": {"id": "x", "text": "nothing here"},
        "malformed_tool_marker": {"id": "x", "messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": TC_OPEN + "notjson" + TC_CLOSE}]},
        "bad_arguments_json": _bad_openai(messages=[{"role": "user", "content": "q"}, {"role": "assistant", "content": None, "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "get_weather", "arguments": "{bad"}}]}]),
        "undeclared_tool_call": _bad_openai(messages=[{"role": "user", "content": "q"}, {"role": "assistant", "content": None, "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "nope", "arguments": '{"city": "x"}'}}]}]),
        "missing_tools_schema": _bad_openai(tools=[]),
        "duplicate_tool_def": _bad_openai(tools=TOOLS + TOOLS),
        "args_schema_violation": _bad_openai(messages=[{"role": "user", "content": "q"}, {"role": "assistant", "content": None, "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "get_weather", "arguments": "{}"}}]}]),
        "tool_call_id_collision": _bad_openai(messages=[{"role": "user", "content": "q"}, {"role": "assistant", "content": None, "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "get_weather", "arguments": '{"city": "x"}'}}, {"id": "c1", "type": "function", "function": {"name": "get_weather", "arguments": '{"city": "x"}'}}]}]),
        "result_without_call": _bad_openai(messages=[{"role": "user", "content": "q"}, {"role": "tool", "tool_call_id": "ghost", "content": "z"}]),
        "call_without_result": _bad_openai(),
        "roles_out_of_order": _bad_openai(messages=[{"role": "assistant", "content": "hi"}, {"role": "user", "content": "q"}]),
        "residual_tool_markers": _bad_openai(messages=[{"role": "user", "content": "q" + TC_OPEN}, {"role": "assistant", "content": None, "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "get_weather", "arguments": '{"city": "x"}'}}]}, {"role": "tool", "tool_call_id": "c1", "content": "ok"}]),
        "too_many_calls": _bad_openai(messages=[{"role": "user", "content": "q"}, {"role": "assistant", "content": None, "tool_calls": [{"id": f"c{i}", "type": "function", "function": {"name": "get_weather", "arguments": '{"city": "x"}'}} for i in range(3)]}, {"role": "tool", "tool_call_id": "c0", "content": "ok"}, {"role": "tool", "tool_call_id": "c1", "content": "ok"}, {"role": "tool", "tool_call_id": "c2", "content": "ok"}]),
    }
    cfgs = {
        "call_without_result": {"require_results": True},
        "too_many_calls": {"max_calls": 2, "require_results": False},
    }
    for slug, rec in cases.items():
        cfg = cfgs.get(slug, {})
        if slug == "call_without_result":
            cfg = {"require_results": True}
        out, stats = run([rec], cfg)
        assert out == [], slug
        assert stats.dropped[slug] == 1, (slug, stats.dropped)
        assert stats.records_in == stats.records_out + sum(stats.dropped.values())


def test_strict_false_keeps_with_meta_status():
    rec = {"id": "x", "text": "nothing"}
    out, stats = run([rec], {"strict": False})
    assert len(out) == 1
    assert out[0]["meta"]["toolcall_status"] == "invalid:unknown_tool_format"
    assert stats.extra["kept_invalid"]["unknown_tool_format"] == 1
    assert stats.records_in == stats.records_out + sum(stats.dropped.values())


def test_accounting_invariant_mixed():
    good = {
        "id": "g",
        "messages": [
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": None, "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "get_weather", "arguments": '{"city": "x"}'}}]},
            {"role": "tool", "tool_call_id": "c1", "content": "ok"},
        ],
        "tools": TOOLS,
    }
    bad = {"id": "b", "text": "nope"}
    out, stats = run([good, bad])
    assert len(out) == 1
    assert stats.records_in == stats.records_out + sum(stats.dropped.values())


def test_idempotent_second_pass_no_modifications():
    rec = {
        "id": "r8",
        "messages": [
            {"role": "user", "content": "weather?"},
            {"role": "assistant", "content": None, "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "get_weather", "arguments": '{"city":"Paris"}'}}]},
            {"role": "tool", "tool_call_id": "c1", "content": "sunny"},
        ],
        "tools": TOOLS,
    }
    out1, stats1 = run([rec])
    out2, stats2 = run(out1)
    assert out2 == out1
    assert sum(stats2.modified.values()) == 0
    assert stats2.records_in == stats2.records_out + sum(stats2.dropped.values())


def test_config_schema_rejects_unknown_key():
    errors = validate({"nope": 1}, CONFIG_SCHEMA)
    assert errors and "additional property" in errors[0]


def test_preflight_always_empty():
    assert preflight({}) == []
