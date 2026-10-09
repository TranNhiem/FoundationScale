"""Tests for slice 3's tool-spec/parsing module (``foundationscale.agentic_rl.tools``).

Claimed here: ``ToolSpec``'s frozen/deep-frozen parameter handling and naming rules,
both parsers' happy and malformed paths (``QwenXmlToolCallParser`` and
``HermesJsonToolCallParser``), and ``validate_call``'s full check order (parse error,
unknown tool, non-mapping/missing arguments, missing required params, unexpected
params under ``additionalProperties: False``, then JSON-schema simple-type
mismatches). Also ``default_tools``'s shape.

Not claimed: no harness or environment is exercised here -- see
``test_native_tool_loop.py``.

All tool-call markup in this file is built exclusively from
``foundationscale.agentic_rl.markup`` constants, concatenated; no tag text is ever
spelled literally (the project hygiene rule forbids it).
"""

from __future__ import annotations

import json

import pytest

from foundationscale.agentic_rl import markup
from foundationscale.agentic_rl.tools import (
    HermesJsonToolCallParser,
    ParsedCall,
    QwenXmlToolCallParser,
    ToolSpec,
    ToolSpecRefusal,
    default_tools,
    validate_call,
)


def _qwen_call(name: str, params: dict[str, str]) -> str:
    body = "".join(
        markup.PARAMETER_OPEN_PREFIX + key + markup.TAG_CLOSE + value + markup.PARAMETER_CLOSE
        for key, value in params.items()
    )
    return (
        markup.TOOL_CALL_OPEN
        + markup.FUNCTION_OPEN_PREFIX
        + name
        + markup.TAG_CLOSE
        + body
        + markup.FUNCTION_CLOSE
        + markup.TOOL_CALL_CLOSE
    )


def _hermes_call(payload: object) -> str:
    return markup.TOOL_CALL_OPEN + json.dumps(payload) + markup.TOOL_CALL_CLOSE


# --------------------------------------------------------------------------- ToolSpec


def test_tool_spec_freezes_nested_parameters() -> None:
    original: dict[str, object] = {
        "type": "object",
        "properties": {"x": {"type": "string", "enum": ["a", "b"]}},
    }
    spec = ToolSpec(
        name="probe",
        description="probe tool",
        parameters=original,
        required=(),
    )
    assert type(spec.parameters) is not dict

    original["properties"]["x"]["enum"].append("c")  # type: ignore[index]
    original["type"] = "array"

    frozen_props = spec.parameters["properties"]  # type: ignore[index]
    assert frozen_props["x"]["enum"] == ("a", "b")  # type: ignore[index]
    assert spec.parameters["type"] == "object"


@pytest.mark.parametrize(
    "bad_name",
    ["", "1abc", "bad name", "bad-name"],
)
def test_tool_spec_rejects_bad_name(bad_name: str) -> None:
    with pytest.raises(ToolSpecRefusal):
        ToolSpec(name=bad_name, description="d", parameters={}, required=())


def test_tool_spec_rejects_duplicate_required() -> None:
    with pytest.raises(ToolSpecRefusal):
        ToolSpec(name="dup", description="d", parameters={}, required=("a", "a"))


def test_to_openai_shape() -> None:
    entry = default_tools()["bash"].to_openai()
    assert entry == {
        "type": "function",
        "function": {
            "name": "bash",
            "description": entry["function"]["description"],
            "parameters": entry["function"]["parameters"],
        },
    }
    assert entry["function"]["name"] == "bash"
    assert entry["function"]["parameters"]["type"] == "object"
    assert entry["function"]["parameters"]["required"] == ["command"]
    assert entry["function"]["parameters"]["additionalProperties"] is False
    json.dumps(entry)


# ------------------------------------------------------------------- QwenXmlToolCallParser


def test_qwen_parses_single_call_with_two_params() -> None:
    # "slow" (not bare JSON) is used rather than a numeric-looking string like "30":
    # a value that parses as JSON is deliberately decoded (see
    # test_qwen_decodes_json_typed_values), so a numeric-looking string here would
    # not exercise "two plain string params" as intended.
    text = "before " + _qwen_call("bash", {"command": "ls -la", "mode": "slow"}) + " after"
    calls, cleaned = QwenXmlToolCallParser().parse(text)
    assert len(calls) == 1
    call = calls[0]
    assert call.error is None
    assert call.name == "bash"
    assert call.arguments == {"command": "ls -la", "mode": "slow"}
    assert markup.TOOL_CALL_OPEN not in cleaned
    assert markup.FUNCTION_OPEN_PREFIX not in cleaned
    assert "before " in cleaned
    assert " after" in cleaned


def test_qwen_decodes_json_typed_values() -> None:
    text = _qwen_call(
        "mixed",
        {
            "count": "42",
            "flag": "true",
            "items": "[1, 2, 3]",
            "obj": '{"a": 1}',
        },
    )
    calls, _ = QwenXmlToolCallParser().parse(text)
    assert len(calls) == 1
    args = calls[0].arguments
    assert args["count"] == 42
    assert type(args["count"]) is int
    assert args["flag"] is True
    assert type(args["flag"]) is bool
    assert args["items"] == [1, 2, 3]
    assert type(args["items"]) is list
    assert args["obj"] == {"a": 1}
    assert type(args["obj"]) is dict


def test_qwen_keeps_bare_json_string_as_plain_text() -> None:
    # Observed behavior: a JSON string literal decodes to a bare Python str, which the
    # documented rule keeps as the raw text (NOT unwrapped to `quoted`).
    raw = '"quoted"'
    text = _qwen_call("strcall", {"value": raw})
    calls, _ = QwenXmlToolCallParser().parse(text)
    assert len(calls) == 1
    value = calls[0].arguments["value"]
    assert value == raw
    assert type(value) is str


def test_qwen_strips_one_leading_and_trailing_newline_only() -> None:
    text = _qwen_call("nl", {"value": "\n\nindented\n\n"})
    calls, _ = QwenXmlToolCallParser().parse(text)
    assert len(calls) == 1
    assert calls[0].arguments["value"] == "\nindented\n"


def test_qwen_multiple_calls_in_order() -> None:
    first = _qwen_call("first", {"a": "1"})
    second = _qwen_call("second", {"b": "2"})
    text = "start " + first + " middle " + second + " end"
    calls, cleaned = QwenXmlToolCallParser().parse(text)
    assert [c.name for c in calls] == ["first", "second"]
    assert all(c.error is None for c in calls)
    assert "middle" in cleaned
    assert "start" in cleaned
    assert "end" in cleaned
    assert markup.TOOL_CALL_OPEN not in cleaned
    assert markup.FUNCTION_OPEN_PREFIX not in cleaned
    assert markup.FUNCTION_CLOSE not in cleaned


def test_qwen_duplicate_parameter_key_is_an_error() -> None:
    body = (
        markup.PARAMETER_OPEN_PREFIX
        + "k"
        + markup.TAG_CLOSE
        + "1"
        + markup.PARAMETER_CLOSE
        + markup.PARAMETER_OPEN_PREFIX
        + "k"
        + markup.TAG_CLOSE
        + "2"
        + markup.PARAMETER_CLOSE
    )
    text = (
        markup.TOOL_CALL_OPEN
        + markup.FUNCTION_OPEN_PREFIX
        + "dup"
        + markup.TAG_CLOSE
        + body
        + markup.FUNCTION_CLOSE
        + markup.TOOL_CALL_CLOSE
    )
    calls, _ = QwenXmlToolCallParser().parse(text)
    assert len(calls) == 1
    assert calls[0].error is not None
    assert "k" in calls[0].error


def test_qwen_missing_function_name_is_an_error() -> None:
    text = (
        markup.TOOL_CALL_OPEN
        + markup.FUNCTION_OPEN_PREFIX
        + markup.TAG_CLOSE
        + markup.FUNCTION_CLOSE
        + markup.TOOL_CALL_CLOSE
    )
    calls, _ = QwenXmlToolCallParser().parse(text)
    assert len(calls) == 1
    assert calls[0].error is not None
    assert calls[0].name == ""


def test_qwen_unterminated_tool_call_is_an_error() -> None:
    dangling = (
        markup.TOOL_CALL_OPEN
        + markup.FUNCTION_OPEN_PREFIX
        + "bash"
        + markup.TAG_CLOSE
        + markup.PARAMETER_OPEN_PREFIX
        + "command"
        + markup.TAG_CLOSE
        + "ls"
        + markup.PARAMETER_CLOSE
        + markup.FUNCTION_CLOSE
    )
    text = "prefix " + dangling + " suffix"
    calls, cleaned = QwenXmlToolCallParser().parse(text)
    assert len(calls) == 1
    assert calls[0].error is not None
    assert markup.TOOL_CALL_OPEN not in cleaned
    assert markup.FUNCTION_OPEN_PREFIX not in cleaned


# ------------------------------------------------------------------ HermesJsonToolCallParser


def test_hermes_parses_well_formed_call() -> None:
    text = _hermes_call({"name": "bash", "arguments": {"command": "ls"}})
    calls, cleaned = HermesJsonToolCallParser().parse(text)
    assert len(calls) == 1
    assert calls[0].error is None
    assert calls[0].name == "bash"
    assert calls[0].arguments == {"command": "ls"}
    assert cleaned == ""


def test_hermes_missing_arguments_key_defaults_to_empty_dict() -> None:
    text = _hermes_call({"name": "submit"})
    calls, _ = HermesJsonToolCallParser().parse(text)
    assert len(calls) == 1
    assert calls[0].error is None
    assert calls[0].arguments == {}


def test_hermes_malformed_json_is_an_error() -> None:
    text = markup.TOOL_CALL_OPEN + "{not json" + markup.TOOL_CALL_CLOSE
    calls, _ = HermesJsonToolCallParser().parse(text)
    assert len(calls) == 1
    assert calls[0].error is not None


def test_hermes_missing_name_is_an_error() -> None:
    text = _hermes_call({"arguments": {}})
    calls, _ = HermesJsonToolCallParser().parse(text)
    assert len(calls) == 1
    assert calls[0].error is not None


def test_hermes_arguments_not_a_dict_is_an_error() -> None:
    text = _hermes_call({"name": "bash", "arguments": "oops"})
    calls, _ = HermesJsonToolCallParser().parse(text)
    assert len(calls) == 1
    assert calls[0].error is not None


# ------------------------------------------------------------------------ validate_call


def _typed_spec(
    name: str,
    props: dict[str, dict[str, object]],
    required: tuple[str, ...] = (),
    additional_properties: bool = True,
) -> ToolSpec:
    parameters: dict[str, object] = {
        "type": "object",
        "properties": props,
        "required": list(required),
    }
    if not additional_properties:
        parameters["additionalProperties"] = False
    return ToolSpec(name=name, description="d", parameters=parameters, required=required)


_TYPED = _typed_spec(
    "typed",
    {
        "s": {"type": "string"},
        "i": {"type": "integer"},
        "n": {"type": "number"},
        "b": {"type": "boolean"},
        "o": {"type": "object"},
        "a": {"type": "array"},
    },
    required=(),
    additional_properties=False,
)
_OPEN = _typed_spec("open", {"x": {"type": "string"}}, required=("x",))


@pytest.mark.parametrize(
    ("call", "expected_ok"),
    [
        (ParsedCall(name="nope", arguments={}, raw="<test>", error=None), False),
        (ParsedCall(name="bash", arguments={}, raw="<test>", error=None), False),
        (
            ParsedCall(
                name="bash", arguments={"command": "ls", "extra": 1}, raw="<test>", error=None
            ),
            False,
        ),
        (ParsedCall(name="typed", arguments={"s": 1}, raw="<test>", error=None), False),
        (ParsedCall(name="typed", arguments={"i": True}, raw="<test>", error=None), False),
        (ParsedCall(name="typed", arguments={"i": 1.5}, raw="<test>", error=None), False),
        (ParsedCall(name="typed", arguments={"n": 3}, raw="<test>", error=None), True),
        (
            ParsedCall(name="typed", arguments={"s": "ok", "b": True}, raw="<test>", error=None),
            True,
        ),
        (ParsedCall(name="submit", arguments={"answer": "42"}, raw="<test>", error=None), True),
        (ParsedCall(name="open", arguments={"x": "y"}, raw="<test>", error=None), True),
    ],
)
def test_validate_call_matrix(call: ParsedCall, expected_ok: bool) -> None:
    tools = default_tools()
    tools["typed"] = _TYPED
    tools["open"] = _OPEN
    result = validate_call(call, tools)
    if expected_ok:
        assert result is None
    else:
        assert isinstance(result, str)
        assert result


def test_validate_call_passes_through_parse_error_unchanged() -> None:
    message = "parse exploded"
    call = ParsedCall(name="bash", arguments={}, raw="<test>", error=message)
    sentinel = {"boom": object()}

    class ExplodingTools(dict):
        def __getitem__(self, key: object) -> object:
            raise AssertionError("tools must not be inspected")

    assert validate_call(call, ExplodingTools(sentinel)) == message


# ------------------------------------------------------------------------ default_tools


def test_default_tools_shape() -> None:
    tools = default_tools()
    assert set(tools) == {"bash", "submit"}
    assert tools["bash"].required == ("command",)
    assert tools["submit"].required == ("answer",)
    for spec in tools.values():
        assert spec.parameters["additionalProperties"] is False


# --------------------------------------------------- coverage additions below


def test_tool_spec_rejects_non_str_description() -> None:
    with pytest.raises(ToolSpecRefusal):
        ToolSpec(name="desc", description=7, parameters={}, required=())  # type: ignore[arg-type]


def test_tool_spec_rejects_empty_description() -> None:
    with pytest.raises(ToolSpecRefusal):
        ToolSpec(name="desc", description="", parameters={}, required=())


def test_tool_spec_rejects_non_mapping_parameters() -> None:
    with pytest.raises(ToolSpecRefusal):
        ToolSpec(name="params", description="d", parameters=["x"], required=())  # type: ignore[arg-type]


def test_tool_spec_rejects_bare_str_required() -> None:
    with pytest.raises(ToolSpecRefusal):
        ToolSpec(name="req", description="d", parameters={}, required="a")  # type: ignore[arg-type]


def test_tool_spec_rejects_non_sequence_required() -> None:
    with pytest.raises(ToolSpecRefusal):
        ToolSpec(name="req", description="d", parameters={}, required=5)  # type: ignore[arg-type]


def test_tool_spec_rejects_non_str_required_entry() -> None:
    with pytest.raises(ToolSpecRefusal):
        ToolSpec(name="req", description="d", parameters={}, required=(3,))  # type: ignore[arg-type]


def test_tool_spec_rejects_empty_required_entry() -> None:
    with pytest.raises(ToolSpecRefusal):
        ToolSpec(name="req", description="d", parameters={}, required=("",))


def test_qwen_stray_text_before_parameter_is_an_error() -> None:
    body = (
        "junk"
        + markup.PARAMETER_OPEN_PREFIX
        + "k"
        + markup.TAG_CLOSE
        + "v"
        + markup.PARAMETER_CLOSE
    )
    text = (
        markup.TOOL_CALL_OPEN
        + markup.FUNCTION_OPEN_PREFIX
        + "stray"
        + markup.TAG_CLOSE
        + body
        + markup.FUNCTION_CLOSE
        + markup.TOOL_CALL_CLOSE
    )
    calls, _ = QwenXmlToolCallParser().parse(text)
    assert len(calls) == 1
    assert calls[0].error is not None
    assert "unclosed parameter" in calls[0].error


def test_qwen_trailing_text_after_last_parameter_is_an_error() -> None:
    body = (
        markup.PARAMETER_OPEN_PREFIX
        + "k"
        + markup.TAG_CLOSE
        + "v"
        + markup.PARAMETER_CLOSE
        + "junk"
    )
    text = (
        markup.TOOL_CALL_OPEN
        + markup.FUNCTION_OPEN_PREFIX
        + "trail"
        + markup.TAG_CLOSE
        + body
        + markup.FUNCTION_CLOSE
        + markup.TOOL_CALL_CLOSE
    )
    calls, _ = QwenXmlToolCallParser().parse(text)
    assert len(calls) == 1
    assert calls[0].error is not None
    assert "unclosed parameter" in calls[0].error


def test_qwen_whitespace_between_parameter_tags_is_tolerated() -> None:
    body = (
        "\n  "
        + markup.PARAMETER_OPEN_PREFIX
        + "k"
        + markup.TAG_CLOSE
        + "v"
        + markup.PARAMETER_CLOSE
        + "\n  "
    )
    text = (
        markup.TOOL_CALL_OPEN
        + markup.FUNCTION_OPEN_PREFIX
        + "ws"
        + markup.TAG_CLOSE
        + body
        + markup.FUNCTION_CLOSE
        + markup.TOOL_CALL_CLOSE
    )
    calls, _ = QwenXmlToolCallParser().parse(text)
    assert len(calls) == 1
    assert calls[0].error is None
    assert calls[0].arguments == {"k": "v"}


def test_qwen_function_open_without_function_close_is_malformed() -> None:
    inner = markup.FUNCTION_OPEN_PREFIX + "bash" + markup.TAG_CLOSE + "body"
    text = markup.TOOL_CALL_OPEN + inner + markup.TOOL_CALL_CLOSE
    calls, _ = QwenXmlToolCallParser().parse(text)
    assert len(calls) == 1
    assert calls[0].error is not None
    assert calls[0].arguments is None


def test_qwen_empty_function_body_parses_to_no_arguments() -> None:
    text = (
        markup.TOOL_CALL_OPEN
        + markup.FUNCTION_OPEN_PREFIX
        + "empty"
        + markup.TAG_CLOSE
        + markup.FUNCTION_CLOSE
        + markup.TOOL_CALL_CLOSE
    )
    calls, _ = QwenXmlToolCallParser().parse(text)
    assert len(calls) == 1
    assert calls[0].error is None
    assert calls[0].name == "empty"
    assert calls[0].arguments == {}


def test_qwen_empty_parameter_key_is_kept() -> None:
    text = _qwen_call("keyless", {"": "v"})
    calls, _ = QwenXmlToolCallParser().parse(text)
    assert len(calls) == 1
    assert calls[0].error is None
    assert calls[0].arguments == {"": "v"}


def test_hermes_decoded_body_not_a_dict_is_an_error() -> None:
    text = markup.TOOL_CALL_OPEN + "[1, 2, 3]" + markup.TOOL_CALL_CLOSE
    calls, _ = HermesJsonToolCallParser().parse(text)
    assert len(calls) == 1
    assert calls[0].error is not None
    assert calls[0].arguments is None


def test_hermes_non_str_name_is_an_error() -> None:
    text = _hermes_call({"name": 5, "arguments": {}})
    calls, _ = HermesJsonToolCallParser().parse(text)
    assert len(calls) == 1
    assert calls[0].error is not None


def test_hermes_empty_name_is_an_error() -> None:
    text = _hermes_call({"name": "", "arguments": {}})
    calls, _ = HermesJsonToolCallParser().parse(text)
    assert len(calls) == 1
    assert calls[0].error is not None


def test_validate_call_arguments_not_a_mapping() -> None:
    call = ParsedCall(name="bash", arguments=None, raw="<test>", error=None)
    result = validate_call(call, default_tools())
    assert isinstance(result, str)
    assert "arguments" in result


def test_validate_call_rejects_unexpected_parameter_when_forbidden() -> None:
    spec = _typed_spec("closed", {"x": {"type": "string"}}, additional_properties=False)
    call = ParsedCall(name="closed", arguments={"y": "1"}, raw="<test>", error=None)
    result = validate_call(call, {"closed": spec})
    assert isinstance(result, str)
    assert "unexpected parameter" in result


def test_validate_call_skips_non_mapping_property_schema() -> None:
    spec = ToolSpec(
        name="weird",
        description="d",
        parameters={"type": "object", "properties": {"x": "oops"}},
        required=(),
    )
    call = ParsedCall(name="weird", arguments={"x": 123}, raw="<test>", error=None)
    assert validate_call(call, {"weird": spec}) is None


def test_validate_call_skips_unknown_property_type() -> None:
    spec = ToolSpec(
        name="unknown_type",
        description="d",
        parameters={
            "type": "object",
            "properties": {"x": {"type": "something-unknown"}},
        },
        required=(),
    )
    call = ParsedCall(name="unknown_type", arguments={"x": object()}, raw="<test>", error=None)
    assert validate_call(call, {"unknown_type": spec}) is None


def test_validate_call_skips_non_mapping_properties_block() -> None:
    spec = ToolSpec(
        name="badprops",
        description="d",
        parameters={"type": "object", "properties": ["not", "a", "mapping"]},
        required=(),
    )
    call = ParsedCall(name="badprops", arguments={"x": 1}, raw="<test>", error=None)
    assert validate_call(call, {"badprops": spec}) is None


def test_validate_call_rejects_missing_required_parameter() -> None:
    call = ParsedCall(name="bash", arguments={}, raw="<test>", error=None)
    result = validate_call(call, default_tools())
    assert isinstance(result, str)
    assert "missing required parameter" in result


def test_validate_call_rejects_unknown_tool_with_no_tools() -> None:
    call = ParsedCall(name="ghost", arguments={}, raw="<test>", error=None)
    result = validate_call(call, {})
    assert isinstance(result, str)
    assert "<none>" in result


def test_validate_call_rejects_mismatched_object_and_array_types() -> None:
    spec = _typed_spec("oa", {"o": {"type": "object"}, "a": {"type": "array"}})
    call = ParsedCall(name="oa", arguments={"o": [], "a": {}}, raw="<test>", error=None)
    result = validate_call(call, {"oa": spec})
    assert isinstance(result, str)
    assert "parameter" in result


def test_validate_call_rejects_mismatched_string_and_boolean_types() -> None:
    spec = _typed_spec("sb", {"s": {"type": "string"}, "b": {"type": "boolean"}})
    call = ParsedCall(name="sb", arguments={"s": 1, "b": 1}, raw="<test>", error=None)
    result = validate_call(call, {"sb": spec})
    assert isinstance(result, str)
    assert "parameter" in result


def test_validate_call_rejects_mismatched_number_type() -> None:
    spec = _typed_spec("num", {"n": {"type": "number"}})
    call = ParsedCall(name="num", arguments={"n": "x"}, raw="<test>", error=None)
    result = validate_call(call, {"num": spec})
    assert isinstance(result, str)
    assert "parameter" in result


def test_validate_call_accepts_float_for_number() -> None:
    spec = _typed_spec("numok", {"n": {"type": "number"}})
    call = ParsedCall(name="numok", arguments={"n": 1.5}, raw="<test>", error=None)
    assert validate_call(call, {"numok": spec}) is None


def test_validate_call_skips_argument_absent_from_properties() -> None:
    spec = _typed_spec("partial", {"x": {"type": "string"}})
    call = ParsedCall(name="partial", arguments={"y": 1}, raw="<test>", error=None)
    assert validate_call(call, {"partial": spec}) is None


def test_hermes_unterminated_tool_call_is_an_error() -> None:
    dangling = markup.TOOL_CALL_OPEN + '{"name": "bash", "arguments": {}}'
    text = "prefix " + dangling + " suffix"
    calls, cleaned = HermesJsonToolCallParser().parse(text)
    assert len(calls) == 1
    assert calls[0].error is not None
    assert cleaned == "prefix "
    assert markup.TOOL_CALL_OPEN not in cleaned


def test_to_openai_unfreezes_nested_list_values() -> None:
    spec = ToolSpec(
        name="probe",
        description="probe tool",
        parameters={
            "type": "object",
            "properties": {"x": {"type": "string", "enum": ["a", "b", "c"]}},
        },
        required=(),
    )
    entry = spec.to_openai()
    enum_value = entry["function"]["parameters"]["properties"]["x"]["enum"]
    assert enum_value == ["a", "b", "c"]
    assert type(enum_value) is list
    json.dumps(entry)


def test_validate_call_accepts_list_for_array_type() -> None:
    spec = _typed_spec("arrok", {"items": {"type": "array"}})
    call = ParsedCall(name="arrok", arguments={"items": [1, 2, 3]}, raw="<test>", error=None)
    assert validate_call(call, {"arrok": spec}) is None
