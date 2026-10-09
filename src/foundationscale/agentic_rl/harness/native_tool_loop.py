"""One multi-turn tool-calling episode over a token-in/token-out GenerationClient.

The loop attributes exact per-turn tokens via cumulative-render diffs so a chat
template's rewriting of tool-call text never desyncs the loss mask from what was
actually sampled.
"""

from __future__ import annotations

import dataclasses
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from foundationscale.agentic_rl.contracts import (
    SegmentKind,
    Termination,
    ToolCall,
    Turn,
)
from foundationscale.agentic_rl.engines import EngineInfraError
from foundationscale.agentic_rl.envs.base import (
    EnvInfraError,
    Environment,
    ExecResult,
)
from foundationscale.agentic_rl.harness.base import (
    ChatTokenizer,
    EpisodeBudget,
    EpisodeOutcome,
    EpisodeTask,
    GenerationClient,
    HarnessAdapter,
    SamplingParams,
)
from foundationscale.agentic_rl.token_trace import (
    ResponseBudgetExhausted,
    TokenTrace,
    select_delta_messages,
)
from foundationscale.agentic_rl.tools import (
    ParsedCall,
    ToolCallParser,
    ToolSpec,
    validate_call,
)

__all__ = ("NativeToolLoop", "NativeToolLoopRefusal")


class NativeToolLoopRefusal(ValueError):
    """A named refusal raised by this harness's own construction or rendering rules."""


def _described(value: object) -> str:
    """Short type-and-value description used in refusal messages."""
    return f"{type(value).__name__} {value!r}"


def _render(
    tokenizer: ChatTokenizer,
    messages: Sequence[Mapping[str, Any]],
    tools_schema: Sequence[Mapping[str, Any]],
    *,
    add_generation_prompt: bool,
) -> list[int]:
    """Render `messages` to token ids, passing the tool schema only when non-empty."""
    return list(
        tokenizer.render(
            list(messages),
            tools=tools_schema or None,
            add_generation_prompt=add_generation_prompt,
        )
    )


def _prefix_delta(before_ids: Sequence[int], after_ids: Sequence[int], *, where: str) -> list[int]:
    """Return the ids `after_ids` adds beyond `before_ids`, requiring a strict prefix.

    Refuses when the later render does not start with the earlier one's ids: this
    harness attributes exact per-message tokens by diffing cumulative renders, which
    requires every render to extend its predecessor's ids as a strict prefix.
    """
    if len(after_ids) < len(before_ids) or after_ids[: len(before_ids)] != before_ids:
        raise NativeToolLoopRefusal(
            f"{where}: tokenizer render is not prefix-stable: the earlier render has "
            f"{len(before_ids)} token(s) and the later render has {len(after_ids)}, "
            f"but the later render does not start with the earlier one's ids -- this "
            f"harness attributes exact per-message tokens by diffing cumulative "
            f"renders, which requires every render to extend its predecessor's ids "
            f"as a strict prefix"
        )
    return list(after_ids[len(before_ids) :])


def _format_exec_result(name: str, result: ExecResult) -> str:
    """One tool call's result, formatted for the model to read: exit code, stdout,
    stderr, a timed-out marker, and a truncation marker -- all measured, never hidden.
    """
    lines = [f"[{name}] exit_code={result.exit_code}"]
    if result.timed_out:
        lines.append("[timed out]")
    lines.append(f"stdout:\n{result.stdout}")
    lines.append(f"stderr:\n{result.stderr}")
    if result.truncated_bytes:
        lines.append(f"[... {result.truncated_bytes} bytes elided ...]")
    return "\n".join(lines)


_MANDATORY_ARGUMENTS: Mapping[str, str] = {"submit": "answer", "bash": "command"}


def _missing_mandatory_argument(call: ParsedCall) -> str | None:
    """A model-attributable error for the two tool names this harness executes
    directly (``"submit"``/``"bash"``) when the ONE argument each handler below
    reads out of ``call.arguments`` by key is missing.

    This is deliberately independent of whether the configured ``ToolSpec``
    declares that argument ``required``: a spec that forgets to do so would
    otherwise let ``validate_call`` report no error at all, and the submit/bash
    handling below would then index a missing key directly -- a bare ``KeyError``
    crashing ``run_episode`` instead of a reported, model-attributable parse
    error. Returns ``None`` for any other tool name, or when the mandatory
    argument is present.
    """
    required_key = _MANDATORY_ARGUMENTS.get(call.name)
    if required_key is None:
        return None
    if not isinstance(call.arguments, Mapping) or required_key in call.arguments:
        return None
    return (
        f"tool {call.name!r}: missing required parameter {required_key!r} -- this "
        f"harness reads it directly to execute {call.name!r} and cannot proceed "
        f"without it"
    )


def _truncate_observation(text: str, max_chars: int) -> str:
    """Middle-truncate to max_chars, keeping the first and last halves and an
    explicit elided-count marker -- never a silent cut, so the model (and any
    downstream reader) can see exactly how much was removed.
    """
    if len(text) <= max_chars:
        return text
    marker_budget = max_chars // 2
    head = text[:marker_budget]
    tail = text[len(text) - marker_budget :]
    elided = len(text) - len(head) - len(tail)
    return f"{head}[... {elided} chars elided ...]{tail}"


@dataclass(frozen=True)
class NativeToolLoop(HarnessAdapter):
    """Runs one multi-turn tool-calling episode against a GenerationClient + Environment.

    Recognises exactly two tool names by string value: "submit" ends the episode and
    "bash" is forwarded to the environment; every other schema-valid tool name has no
    execution binding here and is reported as an execution error without touching its
    parse verdict.
    """

    tools: Mapping[str, ToolSpec]
    parser: ToolCallParser
    tokenizer: ChatTokenizer
    sampling: SamplingParams
    name: str = "native_tool_loop"
    observation_role: str = "tool"

    def __post_init__(self) -> None:
        """Refuse an empty tool registry or empty identity/role strings.

        Prevents an episode that can never route a call or label its observations.
        """
        if not self.tools:
            raise NativeToolLoopRefusal(
                f"NativeToolLoop: field 'tools' is {_described(self.tools)}: the tool "
                f"registry must hold at least one entry -- an empty registry leaves "
                f"every parsed call unvalidatable and unrouteable"
            )
        if not isinstance(self.name, str) or not self.name:
            raise NativeToolLoopRefusal(
                f"NativeToolLoop: field 'name' is {_described(self.name)}: the harness "
                f"name must be a non-empty str -- an unnamed harness cannot be "
                f"identified in recorded trajectories"
            )
        if not isinstance(self.observation_role, str) or not self.observation_role:
            raise NativeToolLoopRefusal(
                f"NativeToolLoop: field 'observation_role' is "
                f"{_described(self.observation_role)}: the observation role must be a "
                f"non-empty str -- an unlabelled observation message cannot be "
                f"attributed to the tool results it carries"
            )

    async def run_episode(
        self,
        task: EpisodeTask,
        *,
        client: GenerationClient,
        env: Environment,
        budget: EpisodeBudget,
    ) -> EpisodeOutcome:
        """Run one episode and return its turns, prompt turns, and termination.

        Ends on a valid submit call, a plain-text answer, a response/step/observation
        budget exhaustion, the episode timeout, an aborted generation, an environment
        infrastructure error (``EnvInfraError`` from ``env.execute``), or a generation
        engine infrastructure error (``EngineInfraError`` from ``client.generate``,
        reported as ``abstention_reason=f"infra:engine_{exc.kind}"``) -- never by
        raising for model-attributable outcomes.
        """
        episode_start = time.monotonic()
        tools_schema: tuple[Mapping[str, Any], ...] = tuple(
            t.to_openai() for t in self.tools.values()
        )
        messages: list[dict[str, Any]] = [dict(m) for m in task.messages]
        trace = TokenTrace(response_length=budget.max_response_tokens)

        # 1. Build the initial prompt + prompt_turns via cumulative-render diffs.
        prompt_turn_list: list[Turn] = []
        before_ids: list[int] = []
        for position in range(len(task.messages)):
            after_ids = _render(
                self.tokenizer,
                messages[: position + 1],
                tools_schema,
                add_generation_prompt=False,
            )
            delta = _prefix_delta(
                before_ids,
                after_ids,
                where=f"prompt render at position {position}",
            )
            role = messages[position]["role"]
            # Any role other than "system"/"user" falls back to USER: SegmentKind has
            # no kind for other prompt-side roles, and these turns are never generated.
            kind = SegmentKind.SYSTEM if role == "system" else SegmentKind.USER
            prompt_turn_list.append(
                Turn(
                    index=position,
                    kind=kind,
                    token_ids=tuple(delta),
                    generated=False,
                    logprobs=None,
                )
            )
            before_ids = after_ids

        full_prompt_ids = _render(
            self.tokenizer,
            messages,
            tools_schema,
            add_generation_prompt=True,
        )
        header_delta = _prefix_delta(
            before_ids,
            full_prompt_ids,
            where="generation-prompt header render",
        )
        if prompt_turn_list:
            # SegmentKind has no dedicated kind for template scaffolding, and
            # prompt_turns' concatenated token ids must equal exactly what gets
            # installed as the model's prompt, so the header delta joins the last turn.
            last_turn = prompt_turn_list[-1]
            prompt_turn_list[-1] = dataclasses.replace(
                last_turn,
                token_ids=last_turn.token_ids + tuple(header_delta),
            )
        prompt_turns: tuple[Turn, ...] = tuple(prompt_turn_list)
        trace.append_prompt(full_prompt_ids)
        turn_index = len(prompt_turns)
        cursor = len(messages)

        # 2. The step loop.
        turns: list[Turn] = []
        termination: Termination | None = None
        submitted_answer: str | None = None
        abstention_reason: str | None = None

        for _step in range(budget.step_limit):
            if time.monotonic() - episode_start > env.episode_timeout_s:
                termination = Termination.TIMEOUT
                break

            try:
                trace.check_budget(1)
            except ResponseBudgetExhausted:
                termination = Termination.LENGTH
                break

            effective_max_new = min(self.sampling.max_new_tokens, trace.remaining_budget())
            step_sampling = dataclasses.replace(self.sampling, max_new_tokens=effective_max_new)
            try:
                generation = await client.generate(tuple(trace.token_ids), step_sampling)
            except EngineInfraError as exc:
                termination = Termination.INFRA
                abstention_reason = f"infra:engine_{exc.kind}"
                break

            parsed_calls, visible_text = self.parser.parse(generation.text)
            tool_calls = tuple(
                ToolCall(
                    name=pc.name or None,
                    arguments=pc.raw,
                    call_id=None,
                    parse_error=(
                        pc.error
                        if pc.error is not None
                        else (validate_call(pc, self.tools) or _missing_mandatory_argument(pc))
                    ),
                )
                for pc in parsed_calls
            )

            trace.append_generated(
                list(generation.token_ids),
                list(generation.logprobs) if generation.logprobs is not None else None,
            )
            assistant_turn = Turn(
                index=turn_index,
                kind=SegmentKind.ASSISTANT,
                token_ids=tuple(generation.token_ids),
                generated=True,
                logprobs=generation.logprobs,
                tool_calls=tool_calls,
            )
            turns.append(assistant_turn)
            turn_index += 1
            messages.append({"role": "assistant", "content": visible_text})
            cursor = len(messages)

            if generation.finish_reason == "abort":
                termination = Termination.ABORTED
                break
            if generation.finish_reason == "length":
                termination = Termination.LENGTH
                break

            # finish_reason == "stop" from here on
            valid_submit = next(
                (
                    pc
                    for pc, tc in zip(parsed_calls, tool_calls, strict=True)
                    if tc.parse_error is None and pc.name == "submit"
                ),
                None,
            )
            if valid_submit is not None:
                answer = (
                    valid_submit.arguments["answer"] if valid_submit.arguments is not None else ""
                )
                submitted_answer = str(answer)
                termination = Termination.STOP
                break

            if not parsed_calls:
                submitted_answer = visible_text
                termination = Termination.STOP
                break

            # One or more real (non-submit) tool calls: execute each in order, build ONE
            # combined observation message for the whole turn.
            observation_parts: list[str] = []
            infra_error: EnvInfraError | None = None
            for pc, tc in zip(parsed_calls, tool_calls, strict=True):
                if tc.parse_error is not None:
                    observation_parts.append(f"[{pc.name or '<unknown>'}] error: {tc.parse_error}")
                    continue
                if pc.name == "bash":
                    command = pc.arguments["command"] if pc.arguments is not None else ""
                    try:
                        result = await env.execute(str(command))
                    except EnvInfraError as exc:
                        infra_error = exc
                        break
                    observation_parts.append(_format_exec_result(pc.name, result))
                else:
                    observation_parts.append(
                        f"[{pc.name}] error: tool {pc.name!r} has no execution binding"
                    )

            if infra_error is not None:
                termination = Termination.INFRA
                abstention_reason = f"infra:{infra_error.kind}"
                break

            observation_text = _truncate_observation(
                "\n".join(observation_parts), budget.max_observation_chars
            )
            messages.append({"role": self.observation_role, "content": observation_text})
            delta_msgs, cursor = select_delta_messages(messages, cursor)
            before_ids = _render(
                self.tokenizer,
                messages[: len(messages) - len(delta_msgs)],
                tools_schema,
                add_generation_prompt=False,
            )
            after_no_gen = _render(
                self.tokenizer, messages, tools_schema, add_generation_prompt=False
            )
            obs_ids = _prefix_delta(before_ids, after_no_gen, where="observation render")
            after_gen = _render(self.tokenizer, messages, tools_schema, add_generation_prompt=True)
            header_ids = _prefix_delta(
                after_no_gen, after_gen, where="generation-prompt header render"
            )
            full_obs_ids = obs_ids + header_ids

            try:
                trace.check_budget(len(full_obs_ids))
            except ResponseBudgetExhausted:
                termination = Termination.LENGTH
                break
            trace.append_observation(full_obs_ids)
            turns.append(
                Turn(
                    index=turn_index,
                    kind=SegmentKind.TOOL_RESULT,
                    token_ids=tuple(full_obs_ids),
                    generated=False,
                    logprobs=None,
                )
            )
            turn_index += 1
        else:
            termination = Termination.STEP_LIMIT
        assert termination is not None  # every break/the for-else above sets it

        # 4. Build the return value.
        return EpisodeOutcome(
            trajectory_turns=tuple(turns),
            prompt_turns=prompt_turns,
            termination=termination,
            submitted_answer=submitted_answer,
            abstention_reason=abstention_reason,
            metadata={},
        )
