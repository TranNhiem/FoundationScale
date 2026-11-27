"""MM-0: the Gemma-4 tool-call parser (``<|tool_call>call:NAME{...}<tool_call|>``)."""

from __future__ import annotations

import random

from foundationscale.agentic_rl.gemma4_tools import Gemma4ToolCallParser

Q = '<|"|>'
P = Gemma4ToolCallParser()


def _call(name: str, body: str) -> str:
    return f"<|tool_call>call:{name}{{{body}}}<tool_call|>"


def test_a_bash_call_and_the_visible_text() -> None:
    text = "Let me check." + _call("bash", f"command:{Q}ls -la{Q}") + "<|tool_response>"
    calls, visible = P.parse(text)
    assert [(c.name, c.arguments, c.error) for c in calls] == [
        ("bash", {"command": "ls -la"}, None)
    ]
    assert visible == "Let me check."


def test_two_calls_and_typed_values() -> None:
    body = f"n:-3,x:2.5e-1,ok:true,no:false,z:null,o:{{a:[1,{Q}b{Q}],{Q}k k{Q}:{{}}}}"
    calls, _ = P.parse(_call("bash", f"command:{Q}a{Q}") + _call("f", body))
    assert calls[0].arguments == {"command": "a"}
    assert calls[1].arguments == {
        "n": -3,
        "x": 0.25,
        "ok": True,
        "no": False,
        "z": None,
        "o": {"a": [1, "b"], "k k": {}},
    }


def test_strings_hold_anything_including_the_close_marker() -> None:
    nasty = 'a, b} {c: "d"}\nnew line <tool_call|> still string'
    calls, visible = P.parse(_call("bash", f"command:{Q}{nasty}{Q}") + " tail")
    assert calls[0].error is None and calls[0].arguments == {"command": nasty}
    assert visible == "tail"


def test_empty_arguments_and_final_answer_markers() -> None:
    calls, _ = P.parse(_call("submit", ""))
    assert calls[0].arguments == {} and calls[0].error is None
    calls, visible = P.parse("The answer is \\boxed{4}.<turn|><eos>")
    assert calls == [] and visible == "The answer is \\boxed{4}."


def test_malformed_calls_are_reported_not_raised() -> None:
    cases = {
        "<|tool_call>call:bash{command:" + Q + "ls": "unterminated tool_call",
        "<|tool_call>bash{}<tool_call|>": "expected 'call:NAME",
        "<|tool_call>call:{}<tool_call|>": "missing function name",
        _call("bash", f"command:{Q}a{Q},command:{Q}b{Q}"): "duplicate key 'command'",
        _call("bash", "command:ls"): "bad value",
        _call("bash", f"command:{Q}a{Q}") + "": "",
        "<|tool_call>call:bash{a:1} junk<tool_call|>": "trailing text",
        _call("bash", f"a:{Q}x"): "unterminated",
    }
    for text, fragment in cases.items():
        calls, _ = P.parse(text)
        assert len(calls) == 1
        if fragment:
            assert calls[0].error is not None and fragment in calls[0].error, (text, calls)
    calls, visible = P.parse("before <|tool_call>call:bash{")
    assert calls[0].name == "" and visible == "before"


def test_nesting_is_capped() -> None:
    deep = "a:" + "[" * 40 + "]" * 40
    calls, _ = P.parse(_call("f", deep))
    assert calls[0].error is not None and "nesting deeper" in calls[0].error


def test_random_garbage_never_raises() -> None:
    pieces = ["<|tool_call>", "<tool_call|>", "call:", "bash", "{", "}", "[", "]", ":", ",", Q,
              "x", "1", "-", "true", " ", "\n", "<turn|>", "<|tool_response>"]  # fmt: skip
    rng = random.Random(0)
    for _ in range(300):
        text = "".join(rng.choice(pieces) for _ in range(rng.randint(0, 30)))
        calls, visible = P.parse(text)
        assert isinstance(visible, str)
        for call in calls:
            assert call.error is None or call.error


def test_config_accepts_gemma4_and_the_cli_builds_its_parser() -> None:
    import json
    from pathlib import Path

    import pytest

    from foundationscale.agentic_rl.cli import _make_parser
    from foundationscale.agentic_rl.config import AgenticConfigRefusal, build_config

    repo = Path(__file__).resolve().parents[2]
    raw = json.loads((repo / "examples" / "agentic_rl" / "music_s0.json").read_text())
    raw["rollout"]["tasks_path"] = str(repo / raw["rollout"]["tasks_path"])
    raw["harness"]["parser"] = "gemma4"
    assert build_config(raw, {}).harness.parser == "gemma4"
    assert isinstance(_make_parser("gemma4"), Gemma4ToolCallParser)
    raw["harness"]["parser"] = "gemma3"
    with pytest.raises(AgenticConfigRefusal, match="or 'gemma4'"):
        build_config(raw, {})
