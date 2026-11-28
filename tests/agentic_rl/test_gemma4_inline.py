"""MM-0: Gemma-4 inline tool observations (tool results continue the model's own turn)."""

from __future__ import annotations

import dataclasses
import importlib.util
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from foundationscale.agentic_rl.contracts import SegmentKind
from foundationscale.agentic_rl.harness.base import Generation, SamplingParams
from foundationscale.agentic_rl.harness.gemma4_protocol import (
    Gemma4InlineObservations,
    Gemma4ProtocolRefusal,
    from_hf_tokenizer,
)
from foundationscale.agentic_rl.harness.native_tool_loop import (
    CompactionPolicy,
    NativeToolLoopRefusal,
)

_SPEC = importlib.util.spec_from_file_location(
    "_mm0_native_tool_loop_helpers", Path(__file__).with_name("test_native_tool_loop.py")
)
assert _SPEC is not None and _SPEC.loader is not None
_H = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _H
_SPEC.loader.exec_module(_H)

OPEN, CLOSE, STRING = 900_001, 900_002, 900_003


def _encoder(**overrides: Any) -> Gemma4InlineObservations:
    fields: dict[str, Any] = {
        "encode_plain": lambda text: [800_000 + ord(ch) for ch in text],
        "open_id": OPEN,
        "close_id": CLOSE,
        "string_id": STRING,
    }
    fields.update(overrides)
    return Gemma4InlineObservations(**fields)


class _Recording:
    def __init__(self, generations: Sequence[Generation]) -> None:
        self._client = _H.ScriptedClient(generations)
        self.contexts: list[tuple[int, ...]] = []

    async def generate(self, prompt_ids: Sequence[int], sampling: SamplingParams) -> Generation:
        self.contexts.append(tuple(prompt_ids))
        result: Generation = await self._client.generate(prompt_ids, sampling)
        return result


def _plain(text: str) -> list[int]:
    return [800_000 + ord(ch) for ch in text]


def test_encode_matches_the_inline_layout_and_skips_an_already_open_block() -> None:
    enc = _encoder()
    one = enc.encode([("bash", "hi")], opened=False)
    body = _plain("response:bash{value:") + [STRING] + _plain("hi") + [STRING] + _plain("}")
    assert one == [OPEN, *body, CLOSE]
    assert enc.encode([("bash", "hi")], opened=True) == [*body, CLOSE]
    two = enc.encode([("bash", "a"), ("bash", "b")], opened=True)
    assert two.count(OPEN) == 1 and two.count(CLOSE) == 2


def test_inline_observation_continues_the_turn_without_a_header() -> None:
    first = _H._generation(_H._bash_call_text("echo hi"))
    first = dataclasses.replace(first, token_ids=(*first.token_ids, OPEN))  # stopped on open
    client = _Recording([first, _H._generation(_H._submit_call_text("done"))])
    loop = _H._loop(inline_observations=_encoder())
    env = _H.FakeEnvironment(execute_fn=lambda command: _H._exec_result("hi\n"))
    outcome = _H._run(loop, _H._task(), client, env, _H._budget())
    observation = next(t for t in outcome.trajectory_turns if t.kind == SegmentKind.TOOL_RESULT)
    assert observation.generated is False and observation.token_ids[0] != OPEN
    assert observation.token_ids[-1] == CLOSE
    # The second generation saw exactly: everything before + the generated call + the result.
    second_context = client.contexts[1]
    assert second_context == client.contexts[0] + first.token_ids + observation.token_ids
    assert outcome.submitted_answer == "done"


def test_a_generation_that_did_not_emit_the_open_marker_gets_it_from_the_observation() -> None:
    client = _Recording(
        [_H._generation(_H._bash_call_text("echo hi")), _H._generation(_H._submit_call_text("x"))]
    )
    loop = _H._loop(inline_observations=_encoder())
    env = _H.FakeEnvironment(execute_fn=lambda command: _H._exec_result("hi\n"))
    outcome = _H._run(loop, _H._task(), client, env, _H._budget())
    observation = next(t for t in outcome.trajectory_turns if t.kind == SegmentKind.TOOL_RESULT)
    assert observation.token_ids[0] == OPEN


def test_inline_observations_refuse_compaction_and_checkpoints() -> None:
    with pytest.raises(NativeToolLoopRefusal, match="'inline_observations' and 'compaction'"):
        _H._loop(
            inline_observations=_encoder(),
            compaction=CompactionPolicy(mode="external", trigger_tokens=10),
        )
    with pytest.raises(NativeToolLoopRefusal, match="expected None or a Gemma4InlineObservations"):
        _H._loop(inline_observations=object())


def test_encoder_refusals() -> None:
    with pytest.raises(Gemma4ProtocolRefusal, match="not distinct"):
        _encoder(close_id=OPEN)
    with pytest.raises(Gemma4ProtocolRefusal, match="non-negative token id"):
        _encoder(string_id=True)
    with pytest.raises(Gemma4ProtocolRefusal, match="must be callable"):
        _encoder(encode_plain="x")
    with pytest.raises(Gemma4ProtocolRefusal, match="no results"):
        _encoder().encode([], opened=False)
    with pytest.raises(Gemma4ProtocolRefusal, match="expected \\(non-empty tool name"):
        _encoder().encode([("", "x")], opened=False)
    forging = _encoder(encode_plain=lambda text: [STRING] if "q" in text else [1])
    with pytest.raises(Gemma4ProtocolRefusal, match="must never become protocol tokens"):
        forging.encode([("bash", "q")], opened=False)


class _FakeHF:
    def __init__(self, vocab: dict[str, list[int]]) -> None:
        self.vocab = vocab

    def encode(self, text: str, **kwargs: Any) -> list[int]:
        return self.vocab.get(text, [7, 7])


def test_from_hf_tokenizer_resolves_markers_and_refuses_a_foreign_tokenizer() -> None:
    hf = _FakeHF({"<|tool_response>": [50], "<tool_response|>": [51], '<|"|>': [52]})
    enc = from_hf_tokenizer(hf)
    assert (enc.open_id, enc.close_id, enc.string_id) == (50, 51, 52)
    assert enc.encode_plain("zz") == [7, 7]
    with pytest.raises(Gemma4ProtocolRefusal, match="does not speak the Gemma-4 tool protocol"):
        from_hf_tokenizer(_FakeHF({}))


def test_inline_observations_refuse_a_checkpoint_hook() -> None:
    import asyncio

    async def hook(checkpoint: Any) -> None:
        return None

    loop = _H._loop(inline_observations=_encoder())
    env = _H.FakeEnvironment(execute_fn=lambda command: _H._exec_result("hi\n"))
    client = _H.ScriptedClient([_H._generation(_H._submit_call_text("x"))])
    with pytest.raises(NativeToolLoopRefusal, match="with inline_observations"):
        asyncio.run(
            loop.run_episode(
                _H._task(), client=client, env=env, budget=_H._budget(), checkpoint_hook=hook
            )
        )


def test_an_inline_observation_past_the_response_budget_ends_the_episode_as_length() -> None:
    from foundationscale.agentic_rl.contracts import Termination

    first = _H._generation(_H._bash_call_text("echo hi"), n_tokens=3)
    client = _Recording([first, _H._generation(_H._submit_call_text("done"))])
    loop = _H._loop(inline_observations=_encoder())
    env = _H.FakeEnvironment(execute_fn=lambda command: _H._exec_result("x" * 200))
    outcome = _H._run(loop, _H._task(), client, env, _H._budget(max_response_tokens=40))
    assert outcome.termination is Termination.LENGTH
    assert all(t.kind != SegmentKind.TOOL_RESULT for t in outcome.trajectory_turns)
