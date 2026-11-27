"""One multi-turn tool-calling episode over a token-in/token-out GenerationClient.

The loop attributes exact per-turn tokens via cumulative-render diffs so a chat
template's rewriting of tool-call text never desyncs the loss mask from what was
actually sampled.
"""

from __future__ import annotations

import copy
import dataclasses
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from foundationscale.agentic_rl.contracts import (
    SegmentKind,
    Termination,
    ToolCall,
    Turn,
)
from foundationscale.agentic_rl.engines import EngineInfraError

if TYPE_CHECKING:
    from foundationscale.agentic_rl.fs_trace import LoopObserver
from foundationscale.agentic_rl.envs.base import (
    EnvInfraError,
    Environment,
    ExecResult,
)
from foundationscale.agentic_rl.harness.base import (
    ChatTokenizer,
    EpisodeBudget,
    EpisodeCheckpoint,
    EpisodeOutcome,
    EpisodeTask,
    GenerationClient,
    HarnessAdapter,
    SamplingParams,
    SegmentOutcome,
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


COMPACTION_MODES = ("external", "policy")


def _response_tokens(trace: TokenTrace) -> int:
    """Tokens in the trace's response region (generated + observations) so far."""
    return len(trace.token_ids) - trace.prompt_len


SUMMARIZE_INSTRUCTION = (
    "[context limit] Summarize everything needed to finish the task: the goal, what you have "
    "done and found, files changed, and the next step. Reply with the summary only, no tool calls."
)
SUMMARY_PREFIX = "Summary of your earlier work on this task:\n"


@dataclass(frozen=True)
class CompactionPolicy:
    """ARL-17: when the context reaches ``trigger_tokens`` at a response boundary, close the
    segment and continue from [system, task + summary, last ``keep_turns`` messages] rendered
    as a FRESH prompt. ``mode="external"``: the summary is generated off the record (never a
    turn, never trained) and enters only as injected text. ``mode="policy"``: the summarize
    instruction rides on the last observation and the policy's reply is recorded as a
    generated turn of the closing segment, trained like any other (CompactionRL).
    ``max_segments`` counts the final segment too."""

    mode: str
    trigger_tokens: int
    keep_turns: int = 2
    max_segments: int = 4
    summary_max_tokens: int = 1024
    # Episode-wide cap on response-region tokens across ALL segments plus external summary
    # tokens (None = uncapped). Set it to the control arm's max_response_tokens so compaction
    # is compared at the SAME total token budget, not handed a fresh budget per segment.
    episode_tokens: int | None = None

    def __post_init__(self) -> None:
        if self.mode not in COMPACTION_MODES:
            raise NativeToolLoopRefusal(
                f"CompactionPolicy: field 'mode' is {_described(self.mode)}: the legal set is "
                f"{COMPACTION_MODES}"
            )
        for name, low in (
            ("trigger_tokens", 1),
            ("keep_turns", 0),
            ("max_segments", 2),
            ("summary_max_tokens", 1),
        ):
            value = getattr(self, name)
            if type(value) is not int or value < low:
                raise NativeToolLoopRefusal(
                    f"CompactionPolicy: field {name!r} is {_described(value)}: an int >= {low}"
                )
        if self.episode_tokens is not None and (
            type(self.episode_tokens) is not int or self.episode_tokens < 1
        ):
            raise NativeToolLoopRefusal(
                f"CompactionPolicy: field 'episode_tokens' is {_described(self.episode_tokens)}: "
                f"None or an int >= 1"
            )


def _compacted_messages(
    history: Sequence[Mapping[str, Any]], summary: str, keep_turns: int
) -> list[dict[str, Any]]:
    """The next segment's chat: the system message (if any), the first user message with the
    summary appended (one user turn, so no template sees two in a row), then the last
    ``keep_turns`` messages, trimmed to start at an assistant turn."""
    system = [dict(m) for m in history[:1] if m.get("role") == "system"]
    rest = list(history[len(system) :])
    first = next((i for i, m in enumerate(rest) if m.get("role") == "user"), None)
    if first is None:
        raise NativeToolLoopRefusal("compaction: the episode has no user message to restate")
    task_message = dict(rest[first])
    task_message["content"] = f"{task_message.get('content', '')}\n\n{SUMMARY_PREFIX}{summary}"
    tail = [dict(m) for m in rest[first + 1 :]][-keep_turns:] if keep_turns else []
    while tail and tail[0].get("role") != "assistant":
        tail = tail[1:]
    return [*system, task_message, *tail]


def _resume_into(
    resume: EpisodeCheckpoint,
    task: EpisodeTask,
    messages: list[dict[str, Any]],
    trace: TokenTrace,
    turns: list[Turn],
) -> tuple[tuple[Turn, ...], int, int]:
    """ARL-18: load a checkpoint's boundary state into the loop's buffers. The TokenTrace is
    rebuilt by replaying the turns -- never by re-rendering or decoding -- so its masks and
    logprobs are the ones originally measured."""
    if resume.task_uid != task.uid:
        raise NativeToolLoopRefusal(
            f"run_episode: checkpoint is for task {resume.task_uid!r}, not {task.uid!r}"
        )
    messages[:] = [copy.deepcopy(dict(m)) for m in resume.messages]
    trace.append_prompt([t for turn in resume.prompt_turns for t in turn.token_ids])
    for turn in resume.turns:
        if turn.generated:
            trace.append_generated(
                list(turn.token_ids),
                list(turn.logprobs) if turn.logprobs is not None else None,
            )
        else:
            trace.append_observation(list(turn.token_ids))
    # ARL-19: a resumed episode's kept turns were sampled by an EARLIER attempt, so they never
    # earn credit in this one -- enforced here, whoever calls run_episode(resume=...).
    turns.extend(
        dataclasses.replace(turn, credited=False) if turn.generated else turn
        for turn in resume.turns
    )
    return resume.prompt_turns, len(resume.prompt_turns) + len(resume.turns), resume.cursor


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
    # False renders prompts WITHOUT the tool schema (single-turn tasks such as music): the
    # registry still routes any call the model emits, but the model is not told tools exist.
    # Measured: Gemma-4-E4B-it shown the schema wrote its music via bash `echo ... > f.abc`
    # tool calls, which the reward (plain-text ABC) cannot score.
    advertise_tools: bool = True
    # ARL-17 context compaction (off by default); see CompactionPolicy.
    compaction: CompactionPolicy | None = None

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

    def _build_prompt(
        self,
        messages: list[dict[str, Any]],
        tools_schema: tuple[Mapping[str, Any], ...],
        trace: TokenTrace,
    ) -> tuple[tuple[Turn, ...], int, int]:
        """Render the initial prompt into ``trace`` via cumulative-render diffs; return the
        prompt turns, the next turn index and the message cursor."""
        prompt_turn_list: list[Turn] = []
        before_ids: list[int] = []
        for position in range(len(messages)):
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
        return prompt_turns, turn_index, cursor

    async def run_episode(
        self,
        task: EpisodeTask,
        *,
        client: GenerationClient,
        env: Environment,
        budget: EpisodeBudget,
        observer: LoopObserver | None = None,
        policy_version: int | None = None,
        checkpoint_hook: Callable[[EpisodeCheckpoint], Awaitable[None]] | None = None,
        resume: EpisodeCheckpoint | None = None,
    ) -> EpisodeOutcome:
        """Run one episode and return its turns, prompt turns, and termination.

        ``observer`` (ARL-21 crash evidence, optional) is told immediately before and
        after every generation and every tool execution; a started call with no finished
        call is the record of what a hung or killed episode was doing. ``policy_version``
        is only forwarded to it. With ``observer=None`` nothing extra happens.

        Ends on a valid submit call, a plain-text answer, a response/step/observation
        budget exhaustion, the episode timeout, an aborted generation, an environment
        infrastructure error (``EnvInfraError`` from ``env.execute``), or a generation
        engine infrastructure error (``EngineInfraError`` from ``client.generate``,
        reported as ``abstention_reason=f"infra:engine_{exc.kind}"``) -- never by
        raising for model-attributable outcomes.

        ARL-18 forks (both optional; with neither, nothing changes): ``checkpoint_hook`` is
        awaited with an :class:`EpisodeCheckpoint` at every response boundary, immediately
        before a generation (not at the boundary a resumed run starts from); its exceptions
        propagate. ``resume`` continues from such a checkpoint instead of rendering the
        prompt: the TokenTrace is rebuilt by replaying the checkpoint's turns (no
        tokenizer call, no tool replay) and the step and wall-clock budgets continue from it.
        """
        episode_start = time.monotonic()
        tools_schema: tuple[Mapping[str, Any], ...] = (
            tuple(t.to_openai() for t in self.tools.values()) if self.advertise_tools else ()
        )
        messages: list[dict[str, Any]] = [dict(m) for m in task.messages]
        trace = TokenTrace(response_length=budget.max_response_tokens)

        # 1. Build the initial prompt (or, for an ARL-18 fork, rebuild the boundary state).
        turns: list[Turn] = []
        first_step = 0
        if resume is None:
            prompt_turns, turn_index, cursor = self._build_prompt(messages, tools_schema, trace)
        else:
            prompt_turns, turn_index, cursor = _resume_into(resume, task, messages, trace, turns)
            first_step = resume.steps_used
            episode_start -= resume.elapsed_s

        # 2. The step loop.
        termination: Termination | None = None
        submitted_answer: str | None = None
        abstention_reason: str | None = None
        policy = self.compaction
        if policy is not None and (resume is not None or checkpoint_hook is not None):
            raise NativeToolLoopRefusal(
                "run_episode: checkpoints (resume / checkpoint_hook) with compaction are not "
                "supported -- a checkpoint does not carry the closed segments, so resuming one "
                "would silently drop them from training"
            )
        closed: list[SegmentOutcome] = []
        spent = 0  # response tokens of closed segments + external summary tokens (episode cap)
        # Set when a compaction failed to shrink the context: compacting again would only loop,
        # so compaction stops for the rest of the episode.
        compaction_spent = False
        summarizing = False  # policy mode: the next generation is the summary turn
        observation_before_instruction = ""

        for _step in range(first_step, budget.step_limit):
            if time.monotonic() - episode_start > env.episode_timeout_s:
                termination = Termination.TIMEOUT
                break

            try:
                trace.check_budget(1)
            except ResponseBudgetExhausted:
                termination = Termination.LENGTH
                break
            episode_left = (
                None
                if policy is None or policy.episode_tokens is None
                else policy.episode_tokens - spent - _response_tokens(trace)
            )
            if episode_left is not None and episode_left < 1:
                termination = Termination.LENGTH  # the episode-wide token budget is spent
                break

            if (
                policy is not None
                and policy.mode == "external"
                and not compaction_spent
                and turns
                and len(closed) < policy.max_segments - 1
                and _step < budget.step_limit - 1
                and len(trace.token_ids) >= policy.trigger_tokens
            ):
                try:
                    summary, summary_tokens = await self._external_summary(
                        client, messages, tools_schema, episode_left
                    )
                except EngineInfraError as exc:
                    termination = Termination.INFRA
                    abstention_reason = f"infra:engine_{exc.kind}"
                    break
                spent += _response_tokens(trace) + summary_tokens
                closed.append(
                    SegmentOutcome(prompt_turns=prompt_turns, trajectory_turns=tuple(turns))
                )
                messages[:] = _compacted_messages(messages, summary, policy.keep_turns)
                before_compaction = len(trace.token_ids)
                trace = TokenTrace(response_length=budget.max_response_tokens)
                turns = []
                prompt_turns, turn_index, cursor = self._build_prompt(messages, tools_schema, trace)
                # A compaction that did not SHRINK the context (summary + kept turns at least as
                # long as what they replace) cannot help; stop compacting rather than loop.
                compaction_spent = len(trace.token_ids) >= before_compaction
                if policy.episode_tokens is not None:
                    episode_left = policy.episode_tokens - spent
                    if episode_left < 1:
                        termination = Termination.LENGTH
                        break

            if checkpoint_hook is not None and not (resume is not None and _step == first_step):
                await checkpoint_hook(
                    EpisodeCheckpoint(
                        task_uid=task.uid,
                        messages=tuple(copy.deepcopy(m) for m in messages),
                        prompt_turns=prompt_turns,
                        turns=tuple(turns),
                        cursor=cursor,
                        steps_used=_step,
                        elapsed_s=time.monotonic() - episode_start,
                    )
                )

            effective_max_new = min(self.sampling.max_new_tokens, trace.remaining_budget())
            if episode_left is not None:
                effective_max_new = min(effective_max_new, episode_left)
            step_sampling = dataclasses.replace(self.sampling, max_new_tokens=effective_max_new)
            gen_index = turn_index  # the index the assistant turn of this generation gets
            if observer is not None:
                observer.generation_started(gen_index, policy_version)
            try:
                generation = await client.generate(tuple(trace.token_ids), step_sampling)
            except EngineInfraError as exc:
                termination = Termination.INFRA
                abstention_reason = f"infra:engine_{exc.kind}"
                break

            if observer is not None:
                observer.generation_finished(
                    gen_index, generation.finish_reason, len(generation.token_ids)
                )
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
            if summarizing:
                # Policy-mode compaction: this reply IS the summary -- recorded and trained in
                # the closing segment; any tool call in it is not executed.
                summarizing = False
                assert policy is not None  # summarizing is only ever set under a policy
                spent += _response_tokens(trace)
                closed.append(
                    SegmentOutcome(prompt_turns=prompt_turns, trajectory_turns=tuple(turns))
                )
                history = messages[:-1]
                history[-1] = {**history[-1], "content": observation_before_instruction}
                messages[:] = _compacted_messages(history, visible_text, policy.keep_turns)
                before_compaction = len(trace.token_ids)
                trace = TokenTrace(response_length=budget.max_response_tokens)
                turns = []
                prompt_turns, turn_index, cursor = self._build_prompt(messages, tools_schema, trace)
                # A compaction that did not SHRINK the context (summary + kept turns at least as
                # long as what they replace) cannot help; stop compacting rather than loop.
                compaction_spent = len(trace.token_ids) >= before_compaction
                continue

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
            for call_index, (pc, tc) in enumerate(zip(parsed_calls, tool_calls, strict=True)):
                if tc.parse_error is not None:
                    observation_parts.append(f"[{pc.name or '<unknown>'}] error: {tc.parse_error}")
                    continue
                if pc.name == "bash":
                    command = pc.arguments["command"] if pc.arguments is not None else ""
                    if observer is not None:
                        observer.tool_started(gen_index, call_index, pc.name)
                    try:
                        result = await env.execute(str(command))
                    except EnvInfraError as exc:
                        infra_error = exc
                        break
                    if observer is not None:
                        observer.tool_finished(
                            gen_index, call_index, result.exit_code, result.timed_out
                        )
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
            if (
                policy is not None
                and policy.mode == "policy"
                and not compaction_spent
                and len(closed) < policy.max_segments - 1
                and _step < budget.step_limit - 2
                and len(trace.token_ids) >= policy.trigger_tokens
            ):
                summarizing = True
                observation_before_instruction = observation_text
                observation_text = f"{observation_text}\n\n{SUMMARIZE_INSTRUCTION}"
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
        if closed and not any(turn.generated for turn in turns):
            # The episode ended right after a compaction: nothing was generated in the new
            # segment, so the last closed segment is the final one after all.
            reopened = closed.pop()
            prompt_turns, turns = reopened.prompt_turns, list(reopened.trajectory_turns)
        return EpisodeOutcome(
            trajectory_turns=tuple(turns),
            prompt_turns=prompt_turns,
            termination=termination,
            submitted_answer=submitted_answer,
            abstention_reason=abstention_reason,
            metadata={},
            segments=tuple(closed),
        )

    async def _external_summary(
        self,
        client: GenerationClient,
        messages: Sequence[Mapping[str, Any]],
        tools_schema: tuple[Mapping[str, Any], ...],
        episode_left: int | None = None,
    ) -> tuple[str, int]:
        """ARL-17 external mode: one off-the-record generation over the current chat plus the
        summarize instruction. Nothing here enters the trace or a turn."""
        assert self.compaction is not None
        context = _render(
            self.tokenizer,
            [*messages, {"role": "user", "content": SUMMARIZE_INSTRUCTION}],
            tools_schema,
            add_generation_prompt=True,
        )
        limit = self.compaction.summary_max_tokens
        if episode_left is not None:
            limit = max(1, min(limit, episode_left))  # summary tokens come out of the same budget
        sampling = dataclasses.replace(self.sampling, max_new_tokens=limit)
        generation = await client.generate(tuple(context), sampling)
        return self.parser.parse(generation.text)[1], len(generation.token_ids)
