"""Torch-free trajectory contracts for the FoundationScale agentic RL plane.

This module is the CONTRACT PLANE of the agentic RL feature: the tool call, the
turn, the trajectory, and the flat batch :func:`flatten` emits from them. It
depends on the standard library and ``foundationscale.rl.interfaces`` only -- a
batch is that module's :class:`~foundationscale.rl.interfaces.ExperienceBatch`,
constructed here and claimed nowhere. The rollout host, the harness adapter,
the environment and the engine adapters arrive in later slices and import THIS
plane through their own submodules, never the other way round (the same
one-way rule ``foundationscale.rl`` applies to its torch-bearing modules).

Three rules run through every type here, and each names the failure it
prevents:

* A value is stated or refused. ``Turn.logprobs`` and ``ToolCall.call_id`` /
  ``ToolCall.parse_error`` carry no default: a caller that means "the engine
  reported none" or "parsed fine" must pass ``None`` explicitly, because a
  default would assert one of those on a caller who merely forgot. There is no
  padded logprob list and no default reward -- an unmeasured score is ``None``
  and never ``0.0``, which is a real and different observation.
* Refusals name their subject. Every ``TrajectoryRefusal`` carries the failed
  field, the trajectory (``uid`` and ``session_id``) wherever construction knows
  it, and BOTH sides of any count compared. A turn or tool call built
  standalone is refused under a label that states no trajectory is attached
  yet -- honest, where naming a uid would invent one.
* Nothing is coerced to shape. Sequences are frozen to tuples and metadata to a
  read-only mapping -- representation, not repair -- but ``True`` is never read
  as ``1`` (``type(x) is int``), a mismatched logprob list is never padded or
  truncated, and a boolean-in-place-of-a-count is refused.

WHAT IS NOT CLAIMED: how a harness tokenises, scores or renders (later slices);
anything behind ``metadata``, which carries harness-private strings and never
reaches the batch; and any verdict on the MODEL's tool-call text --
:class:`ToolCall` records the raw text and the harness's parse verdict, and
neither is re-parsed nor re-serialised here, so there is exactly one opinion
about model output and it is the harness's.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from math import isfinite
from types import MappingProxyType
from typing import Any, TypeGuard

from foundationscale.rl.interfaces import ExperienceBatch

__all__ = (
    "DECLARED_COLUMNS",
    "SegmentKind",
    "Termination",
    "ToolCall",
    "Trajectory",
    "TrajectoryRefusal",
    "Turn",
    "flatten",
    "group_by_uid",
)


class TrajectoryRefusal(ValueError):
    # Raised when a tool call, a turn, a trajectory or a batch of them violates
    # its declared contract. Every message names the failed field, the
    # trajectory (uid and session_id) wherever the construction knows it, and
    # both sides of any count it compares. Fail closed: never pad, never
    # coerce, never default a field into a value that would assert something
    # the caller did not state.
    pass


class SegmentKind(str, Enum):
    """What a rendered segment IS, used per token for supervision and diagnosis.

    There is deliberately NO PAD member: padding is not a segment, it carries no
    text, and a padded position would need an empty ``token_ids`` which
    :class:`Turn` refuses. The values are lower-case strings and the members are
    ``str`` subclasses, so a row dumps to text with no adapter while typed
    consumers keep the enum.
    """

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    OBSERVATION = "observation"


class Termination(str, Enum):
    """How the attempt ended, which decides whether it counts at all.

    ``INFRA`` is the harness failing (never the model's, never scored);
    ``ABORTED`` is the attempt cancelled before it produced a sample;
    ``LENGTH`` is the response budget ending it; ``STEP_LIMIT`` and ``TIMEOUT``
    are the harness's step and wall-clock budgets -- the attempt finished but
    the model never declared completion and no refusal is claimed;
    ``STOP`` is the model declaring the task done.
    """

    STOP = "stop"
    LENGTH = "length"
    STEP_LIMIT = "step_limit"
    TIMEOUT = "timeout"
    ABORTED = "aborted"
    INFRA = "infra"


_SUPERVISED_KINDS = (SegmentKind.ASSISTANT, SegmentKind.TOOL_CALL)


def _is_int(value: object) -> TypeGuard[int]:
    # `type(value) is int`, deliberately NOT isinstance: `True` is an `int`
    # under isinstance, and a bool silently read as a count or a token id is
    # exactly the coercion this plane refuses ("1 is not True").
    return type(value) is int


def _is_float(value: object) -> TypeGuard[float]:
    return type(value) is float


def _is_real(value: object) -> TypeGuard[float]:
    # Narrowed to float for typing only: an int entry is stored VERBATIM, and
    # a bool is refused because `type(value) is float`/`is int` excludes it.
    return type(value) in (int, float)


def _described(value: object) -> str:
    # A refusal names what it refused: value first, concrete type second, so a
    # bool never reads as an int nor text as a token id.
    return f"{value!r} ({type(value).__name__})"


def _items(value: object, *, field_name: str, where: str) -> Sequence[object]:
    # Expose a tuple/sequence's entries; refuse text and bare scalars. Freezing
    # list to tuple happens where the field is stored -- representation, not
    # repair -- but there is no path in which a str is read as a sequence of
    # one-character values.
    if isinstance(value, tuple):
        return value
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray, memoryview)):
        return value
    raise TrajectoryRefusal(
        f"{where}: field {field_name!r} is {_described(value)}: this field is a sequence, "
        f"and text or a bare scalar is not a sequence of values"
    )


def _token_ids(value: object, *, field_name: str, where: str) -> tuple[int, ...]:
    items = _items(value, field_name=field_name, where=where)
    if not items:
        raise TrajectoryRefusal(
            f"{where}: field {field_name!r} is empty: a turn with no tokens has no mask "
            f"entry and no count, and would make every per-token column lie by one"
        )
    tokens: list[int] = []
    for position, item in enumerate(items):
        if not _is_int(item):
            raise TrajectoryRefusal(
                f"{where}: field {field_name!r} entry {position} is {_described(item)}: "
                f"token ids are real ints -- 1 is not True -- and a bool here would decode "
                f"as whatever id sits underneath it"
            )
        if item < 0:
            raise TrajectoryRefusal(
                f"{where}: field {field_name!r} entry {position} is {item}: token ids are >= 0"
            )
        tokens.append(item)
    return tuple(tokens)


def _logprob_values(value: object, *, token_count: int, where: str) -> tuple[float, ...]:
    items = _items(value, field_name="logprobs", where=where)
    if len(items) != token_count:
        raise TrajectoryRefusal(
            f"{where}: field 'logprobs' carries {len(items)} entries but field 'token_ids' "
            f"carries {token_count}: one logprob per sampled token, exactly -- padding or "
            f"truncating either side would invent probability where the engine reported none "
            f"(0.0 is a real logprob meaning p=1)"
        )
    logprobs: list[float] = []
    for position, item in enumerate(items):
        if not _is_real(item):
            raise TrajectoryRefusal(
                f"{where}: field 'logprobs' entry {position} is {_described(item)}: a logprob "
                f"is a real number"
            )
        if _is_float(item) and not isfinite(item):
            raise TrajectoryRefusal(
                f"{where}: field 'logprobs' entry {position} is {item}: logprobs are finite -- "
                f"a nan or an inf would poison every mean it enters"
            )
        logprobs.append(item)
    return tuple(logprobs)


def _tool_calls(value: object, *, where: str) -> tuple[ToolCall, ...]:
    items = _items(value, field_name="tool_calls", where=where)
    calls: list[ToolCall] = []
    for position, item in enumerate(items):
        if not isinstance(item, ToolCall):
            raise TrajectoryRefusal(
                f"{where}: field 'tool_calls' entry {position} is {_described(item)}, not a "
                f"ToolCall"
            )
        calls.append(item)
    return tuple(calls)


def _turn_list(value: object, *, field_name: str, where: str) -> tuple[Turn, ...]:
    items = _items(value, field_name=field_name, where=where)
    if not items:
        raise TrajectoryRefusal(
            f"{where}: field {field_name!r} is empty: a trajectory is prompt turns (the "
            f"shared context) plus response turns (the attempt), and one side missing leaves "
            f"nothing the attempt can be attributed to"
        )
    turns: list[Turn] = []
    for position, item in enumerate(items):
        if not isinstance(item, Turn):
            raise TrajectoryRefusal(
                f"{where}: field {field_name!r} entry {position} is {_described(item)}, not a Turn"
            )
        turns.append(item)
    return tuple(turns)


def _unattached(carrier: str) -> str:
    # Label for a carrier that exists without a trajectory yet. Saying so is the
    # honest form: a refusal can only name uid/session_id once the Trajectory
    # exists, and inventing one here would name a trajectory that is not there.
    return f"{carrier} (unattached: no trajectory uid/session_id to name)"


@dataclass(frozen=True)
class ToolCall:
    """One tool invocation exactly as the model produced it, plus the harness's verdict.

    ``arguments`` is the raw JSON TEXT the model emitted and is never
    re-serialised, re-ordered or re-indented here: a model's bytes are evidence
    and this plane keeps them intact. ``parse_error`` is the harness's verdict
    on that evidence -- ``None`` meaning "parsed fine" and a non-empty string
    naming the model-attributable error (unknown tool, invalid JSON, schema
    mismatch). There is no third opinion: re-parsing here would let a caller
    juggle two verdicts about the same sample, and ``this module never
    inspecting arguments`` is what keeps exactly one.

    ``call_id`` and ``parse_error`` carry no default. Absence must be stated as
    ``None``: a default of ``None`` on ``parse_error`` would record "parsed
    fine" for a call whose caller simply forgot to report.
    """

    name: str
    arguments: str
    call_id: str | None
    parse_error: str | None

    def __post_init__(self) -> None:
        where = _unattached(f"tool call name={self.name!r}")
        if not isinstance(self.name, str) or not self.name:
            raise TrajectoryRefusal(
                f"{where}: field 'name' is {_described(self.name)}: every tool call names its "
                f"tool with a non-empty str -- absence of a name is not a name, and an unnamed "
                f"call can be neither routed, refused nor reported"
            )
        if not isinstance(self.arguments, str):
            raise TrajectoryRefusal(
                f"{where}: field 'arguments' is {_described(self.arguments)}: arguments are the "
                f"raw JSON text the model produced -- text that fails to parse is still text, "
                f"and its verdict lives in parse_error -- and never a re-serialised structure"
            )
        if self.call_id is not None and (not isinstance(self.call_id, str) or not self.call_id):
            raise TrajectoryRefusal(
                f"{where}: field 'call_id' is {_described(self.call_id)}: call_id is None "
                f"(nothing was traced) or a non-empty str identifying the invocation"
            )
        if self.parse_error is not None and (
            not isinstance(self.parse_error, str) or not self.parse_error
        ):
            raise TrajectoryRefusal(
                f"{where}: field 'parse_error' is {_described(self.parse_error)}: None means "
                f"parsed fine and a non-empty string names the model-attributable error; an "
                f"empty string asserts neither and would read as an error with no reason"
            )


@dataclass(frozen=True)
class Turn:
    """One rendered segment of a trajectory: its kind, its exact tokens, their provenance.

    ``index`` is the turn's position in the trajectory's ``prompt_turns + turns``
    sequence and must be strictly increasing across it -- checked by
    :class:`Trajectory`, which can see the whole sequence.
    ``generated`` is the provenance claim the whole supervision plane rests on:
    True iff the rollout engine SAMPLED these tokens. Everything the loss trains
    on derives from it, which is why it is a strict bool and why a truthy
    non-bool is refused rather than read.

    ``logprobs`` has no default on purpose. A turn whose engine reported none and
    a turn whose caller forgot to pass them look identical if absence is the
    default, and ``rollout_logprobs`` would then report an absence the engine
    never reported. The caller passes ``None`` when it means it. When present the
    list is exactly one value per token -- never padded, never truncated.

    ``tool_calls`` is lawful only on a generated ASSISTANT/TOOL_CALL turn: a tool
    call is part of the text the model sampled, and recording one on injected
    text would attribute the harness's rendering to the policy.
    """

    index: int
    kind: SegmentKind
    token_ids: tuple[int, ...]
    generated: bool
    logprobs: tuple[float, ...] | None
    tool_calls: tuple[ToolCall, ...] = ()
    repetition_hit: bool = False

    def __post_init__(self) -> None:
        where = _unattached(f"turn index={self.index!r}")
        if not _is_int(self.index):
            raise TrajectoryRefusal(
                f"{where}: field 'index' is {_described(self.index)}: a turn index is a real int "
                f"(type(x) is int) -- 1 is not True -- and a bool would alias the turn it "
                f"numerically equals"
            )
        if self.index < 0:
            raise TrajectoryRefusal(
                f"{where}: field 'index' is {self.index}: turn indices are >= 0 and order the "
                f"sequence they index"
            )
        if not isinstance(self.kind, SegmentKind):
            raise TrajectoryRefusal(
                f"{where}: field 'kind' is {_described(self.kind)}, not a SegmentKind: the kind "
                f"decides alone whether these tokens are supervised"
            )
        object.__setattr__(
            self, "token_ids", _token_ids(self.token_ids, field_name="token_ids", where=where)
        )
        if type(self.generated) is not bool:
            raise TrajectoryRefusal(
                f"{where}: field 'generated' is {_described(self.generated)}: True or False only "
                f"-- the loss mask is derived from this flag alone, and a truthy non-bool "
                f"(the text 'false' is truthy) would supervise injected text"
            )
        if self.logprobs is not None:
            if not self.generated:
                raise TrajectoryRefusal(
                    f"{where}: field 'logprobs' is not None on a non-generated turn: a logprob "
                    f"is the sampling probability of a token the POLICY generated, and injected "
                    f"text has no sampling distribution to report"
                )
            object.__setattr__(
                self,
                "logprobs",
                _logprob_values(self.logprobs, token_count=len(self.token_ids), where=where),
            )
        calls = _tool_calls(self.tool_calls, where=where)
        if calls and not (self.generated and self.kind in _SUPERVISED_KINDS):
            raise TrajectoryRefusal(
                f"{where}: field 'tool_calls' carries {len(calls)} call(s) on a "
                f"{self.kind.value!r} turn with generated={self.generated!r}: a tool call is "
                f"recorded only on a generated ASSISTANT/TOOL_CALL turn, because a call is part "
                f"of the sampled text and nothing else has text the model chose"
            )
        object.__setattr__(self, "tool_calls", calls)
        if type(self.repetition_hit) is not bool:
            raise TrajectoryRefusal(
                f"{where}: field 'repetition_hit' is {_described(self.repetition_hit)}: True or "
                f"False only -- repetition_mask() is derived from this flag"
            )


@dataclass(frozen=True)
class Trajectory:
    """One attempt: the shared prompt, the response turns, and what the run made of it.

    ``uid`` names the prompt group and ``session_id`` one attempt inside it --
    together the identity ``flatten`` requires unique. ``prompt_turns`` are the
    context the whole group shares and are never generated: crediting them to one
    attempt's policy sample would supervise the same tokens once per attempt.
    ``turns`` are the response side in order, every turn carrying its own
    provenance.

    ``reward`` and ``abstention_reason`` form a biconditional:
    ``reward is None`` exactly when ``abstention_reason`` is a non-empty str.
    A trajectory either stands a score or carries the reason it abstained; a
    missing reward is ``None`` and never ``0.0``, because an unmeasured score
    that reads as a measured zero is the exact reading the RL gates exist to
    refuse. Under ``Termination.INFRA`` the run -- never the model -- ended the
    attempt, so the reward MUST stay ``None`` and the reason MUST carry the
    ``infra:`` prefix that separates harness faults from model abstentions.

    ``policy_version_min``/``policy_version_max`` bound the policies that could
    have sampled this batch and are declared together or not at all.
    ``metadata`` is harness-private ``str`` text carried read-only and never
    exported to a batch.

    Validation lives where the context lives: per-field shape is refused at
    :class:`Turn`/class:`ToolCall` construction -- where no trajectory exists and
    the message says so -- and every sequence-level rule (strict index order,
    the reward/abstention biconditional, the infra prefix, the sampled-at-least-
    once rule) is refused HERE, where the refusal can name uid and session_id.
    """

    uid: str
    session_id: str
    harness: str
    prompt_turns: tuple[Turn, ...]
    turns: tuple[Turn, ...]
    reward: float | None
    abstention_reason: str | None
    termination: Termination
    policy_version_min: int | None
    policy_version_max: int | None
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        where = f"trajectory uid={self.uid!r} session_id={self.session_id!r}"
        for name in ("uid", "session_id", "harness"):
            declared: object = getattr(self, name)
            if not isinstance(declared, str) or not declared:
                raise TrajectoryRefusal(
                    f"{where}: field {name!r} is {_described(declared)}: uid names the prompt "
                    f"group, session_id one attempt inside it and harness which robot recorded "
                    f"the turns, and each is a non-empty str -- absence of a name is not a name"
                )
        object.__setattr__(
            self,
            "prompt_turns",
            _turn_list(self.prompt_turns, field_name="prompt_turns", where=where),
        )
        object.__setattr__(self, "turns", _turn_list(self.turns, field_name="turns", where=where))
        for position, turn in enumerate(self.prompt_turns):
            if turn.generated:
                raise TrajectoryRefusal(
                    f"{where}: field 'prompt_turns' entry {position} has generated=True: the "
                    f"prompt is the context the whole prompt group shares and no one attempt "
                    f"sampled it -- crediting it to a policy would supervise shared tokens once "
                    f"per attempt in the group"
                )
        sequence = self.prompt_turns + self.turns
        for position in range(1, len(sequence)):
            previous, current = sequence[position - 1], sequence[position]
            if current.index <= previous.index:
                raise TrajectoryRefusal(
                    f"{where}: field 'prompt_turns'+'turns' breaks strict index order at "
                    f"position {position}: index {current.index} follows index {previous.index}; "
                    f"indices must be strictly increasing from prompt into response or no "
                    f"per-token column can say which turn a token belongs to"
                )
        if self.reward is not None:
            if not _is_real(self.reward):
                raise TrajectoryRefusal(
                    f"{where}: field 'reward' is {_described(self.reward)}: a reward is a finite "
                    f"scalar or None -- None is how an unmeasured score is stated and it is "
                    f"never 0.0, a measured zero"
                )
            if _is_float(self.reward) and not isfinite(self.reward):
                raise TrajectoryRefusal(
                    f"{where}: field 'reward' is {self.reward}: rewards are finite -- a nan or "
                    f"an inf would poison every group mean it enters"
                )
        if self.abstention_reason is not None and (
            not isinstance(self.abstention_reason, str) or not self.abstention_reason
        ):
            raise TrajectoryRefusal(
                f"{where}: field 'abstention_reason' is {_described(self.abstention_reason)}: "
                f"it is None or a non-empty str -- an empty reason names no cause and the "
                f"abstention would read as a score of nothing"
            )
        if self.reward is None and self.abstention_reason is None:
            raise TrajectoryRefusal(
                f"{where}: field 'reward' is None together with field 'abstention_reason' "
                f"None: a trajectory either stands a score or carries the reason it abstained "
                f"(reward is None exactly when abstention_reason is a non-empty str), and "
                f"neither leaves a run unable to tell an unmeasured attempt from a discarded one"
            )
        if self.reward is not None and self.abstention_reason is not None:
            raise TrajectoryRefusal(
                f"{where}: field 'reward' is {self.reward!r} together with field "
                f"'abstention_reason' {self.abstention_reason!r}: a scored attempt has nothing "
                f"to abstain about -- the score counts or the reason does, never both"
            )
        if not isinstance(self.termination, Termination):
            raise TrajectoryRefusal(
                f"{where}: field 'termination' is {_described(self.termination)}, not a "
                f"Termination: the stop reason decides whether the attempt counts at all"
            )
        if self.termination is Termination.INFRA:
            if self.reward is not None:
                raise TrajectoryRefusal(
                    f"{where}: field 'reward' is {self.reward!r} on an INFRA termination: an "
                    f"infra failure is the harness's and never the model's, and scoring it "
                    f"would train the policy on a pod it never saw"
                )
            reason: str = self.abstention_reason or ""
            if not reason.startswith("infra:"):
                raise TrajectoryRefusal(
                    f"{where}: field 'abstention_reason' is {self.abstention_reason!r} and must "
                    f"start with 'infra:' under an INFRA termination: the prefix is what lets a "
                    f"reader separate harness faults (never scored) from model abstentions"
                )
        minimum = self.policy_version_min
        maximum = self.policy_version_max
        if (minimum is None) != (maximum is None):
            raise TrajectoryRefusal(
                f"{where}: field 'policy_version_min' is {minimum!r} and field "
                f"'policy_version_max' is {maximum!r}: the version pair is declared together or "
                f"not at all -- half an off-policy range supports every correction and none"
            )
        if minimum is not None and maximum is not None:
            if not _is_int(minimum) or not _is_int(maximum):
                raise TrajectoryRefusal(
                    f"{where}: field 'policy_version_min' is {_described(minimum)} and field "
                    f"'policy_version_max' is {_described(maximum)}: policy versions are real "
                    f"ints -- 1 is not True"
                )
            if minimum < 0:
                raise TrajectoryRefusal(
                    f"{where}: field 'policy_version_min' is {minimum}: policy versions start "
                    f"at 0, the step-0 record every drift is measured against"
                )
            if minimum > maximum:
                raise TrajectoryRefusal(
                    f"{where}: field 'policy_version_min' is {minimum} but field "
                    f"'policy_version_max' is {maximum}: the range is ordered 0 <= min <= max "
                    f"and this one names no policy at all"
                )
        if self.termination not in (Termination.INFRA, Termination.ABORTED) and not any(
            turn.generated for turn in self.turns
        ):
            raise TrajectoryRefusal(
                f"{where}: field 'turns' has 0 generated turns of {len(self.turns)} under a "
                f"{self.termination.value!r} termination: a completed attempt that sampled "
                f"nothing is not attributable to the policy -- record Termination.INFRA or "
                f"Termination.ABORTED, which say exactly that, instead of a finished attempt "
                f"carrying no sampled token"
            )
        if not isinstance(self.metadata, Mapping):
            raise TrajectoryRefusal(
                f"{where}: field 'metadata' is {_described(self.metadata)}: metadata is a "
                f"mapping of str to str that this plane carries without reading"
            )
        entries: dict[str, str] = {}
        for key, entry_value in self.metadata.items():
            if not isinstance(key, str) or not key:
                raise TrajectoryRefusal(
                    f"{where}: field 'metadata' key {key!r} at position {len(entries)}: keys "
                    f"are non-empty strs -- absence of a name is not a name"
                )
            if not isinstance(entry_value, str):
                raise TrajectoryRefusal(
                    f"{where}: field 'metadata' entry {key!r} is {_described(entry_value)}: "
                    f"values are strings and this plane coerces none of them"
                )
            entries[key] = entry_value
        object.__setattr__(self, "metadata", MappingProxyType(entries))

    def __hash__(self) -> int:
        # Frozen+eq dataclasses generate a hash over every field, which would
        # reach `metadata` -- a read-only proxy and unhashable. Hash on the
        # identity pair instead: `__eq__` compares every field so equal
        # trajectories necessarily share uid and session_id (equal implies
        # equal hash), and the pair is what `flatten` treats as one row.
        return hash((self.uid, self.session_id))

    @property
    def response_token_ids(self) -> tuple[int, ...]:
        """The response side's token ids in order, ``flatten``'s ``response_ids`` row.

        Prompt tokens are deliberately absent: the prompt is carried whole by the
        ``prompt_ids`` column and is never per-token aligned, because it holds no
        sampled token to supervise or count.
        """
        return tuple(token for turn in self.turns for token in turn.token_ids)

    def loss_mask(self) -> tuple[int, ...]:
        """1 exactly on generated ASSISTANT/TOOL_CALL tokens; 0 on every other token.

        This is what the loss trains on. Tool results, observations, user and
        system text are 0 because the harness rendered them and the model never
        sampled them: supervising injected text rewards tokens the policy did not
        choose. One entry per response token, aligned with ``response_token_ids``.

        An INFRA trajectory is all-0 even where it holds generated tokens: the run,
        not the model, ended it, so none of its tokens is a choice the policy can be
        held to (the row also carries ``reward=None`` and leaves the batch).
        """
        if self.is_infra:
            return (0,) * sum(len(turn.token_ids) for turn in self.turns)
        mask: list[int] = []
        for turn in self.turns:
            supervised = turn.generated and turn.kind in _SUPERVISED_KINDS
            mask.extend([int(supervised)] * len(turn.token_ids))
        return tuple(mask)

    def turn_index_per_token(self) -> tuple[int, ...]:
        """The owning ``Turn.index`` repeated for each of its response tokens.

        Carried per token (rather than as turn offsets) so a downstream slice can
        group sampled tokens by turn without re-deriving boundaries.
        """
        indices: list[int] = []
        for turn in self.turns:
            indices.extend([turn.index] * len(turn.token_ids))
        return tuple(indices)

    def segment_kind_per_token(self) -> tuple[SegmentKind, ...]:
        """The owning turn's :class:`SegmentKind` repeated for each response token.

        Rebuilt on every call from the frozen turns, so the caller owns the tuple
        it receives and no cached copy can drift from the trajectory.
        """
        kinds: list[SegmentKind] = []
        for turn in self.turns:
            kinds.extend([turn.kind] * len(turn.token_ids))
        return tuple(kinds)

    def tool_call_error_mask(self) -> tuple[int, ...]:
        """1 on every token of a generated turn carrying any failed :class:`ToolCall`.

        TURN granularity, deliberately: a model's bad tool-call syntax corrupts
        the turn it was written in, the attribution for a single broken token is
        an inference no measurement supports, and a per-turn flag is what a gate
        can count without guessing where the malformed text began.
        """
        mask: list[int] = []
        for turn in self.turns:
            flagged = turn.generated and any(
                call.parse_error is not None for call in turn.tool_calls
            )
            mask.extend([int(flagged)] * len(turn.token_ids))
        return tuple(mask)

    def repetition_mask(self) -> tuple[int, ...]:
        """1 on every token of a generated turn the harness flagged ``repetition_hit``.

        Like ``tool_call_error_mask`` this is per turn: the flag is asserted of the
        turn and no finer attribution exists to record.
        """
        mask: list[int] = []
        for turn in self.turns:
            mask.extend([int(turn.generated and turn.repetition_hit)] * len(turn.token_ids))
        return tuple(mask)

    def rollout_logprobs(self) -> tuple[float | None, ...]:
        """Per response token the logprob the engine reported, or ``None`` for absence.

        ``None`` on every token the policy did not sample and on sampled tokens
        whose turn carried no logprob list. Never ``0.0`` as a filler: 0.0 is a
        real logprob meaning p=1, and an unreported quantity that reads as a
        reported one is the absent-is-not-zero failure the RL plane refuses.
        These values exist ONLY for a declared off-policy correction -- see the
        ``rollout_logprobs`` row of the column table above ``DECLARED_COLUMNS``.
        """
        values: list[float | None] = []
        for turn in self.turns:
            if not turn.generated or turn.logprobs is None:
                values.extend([None] * len(turn.token_ids))
            else:
                values.extend(turn.logprobs)
        return tuple(values)

    @property
    def supervised_token_count(self) -> int:
        """How many response tokens this trajectory would train on.

        The denominator for any per-token supervision rate. ``0`` is reachable
        (an INFRA attempt is all-0 by construction; an ABORTED one may hold no
        sampled token) and must be read
        as "nothing to train on", never as a healthy zero: a batch summing to 0
        is unmeasurable and the loss owes the caller the RL plane's
        SupervisionRefusal rather than a 0.0 loss.
        """
        return sum(self.loss_mask())

    @property
    def is_infra(self) -> bool:
        """True when the run -- never the model -- ended this attempt.

        Such a trajectory records an ``infra:`` reason and no reward, and its
        :meth:`loss_mask` is all-0 whatever tokens it holds.
        """
        return self.termination is Termination.INFRA


# ---------------------------------------------------------------------------
# DECLARED_COLUMNS -- the exact schema :func:`flatten` emits, one row per
# trajectory, in this order. The emitted batch passes these as
# ``required=DECLARED_COLUMNS``, so a consumer that renames a column learns at
# construction and not at step zero. Per-token columns cover the RESPONSE side
# and carry exactly one entry per ``response_ids`` token; the prompt is carried
# whole by ``prompt_token_ids`` and is never per-token aligned, because the prompt is
# shared context and holds no sampled token to supervise or count.
#
#   column                contents (one value per row)
#   --------------------- ------------------------------------------------------------
#   prompt_ids            str -- the GROUP key the RL plane's advantage estimators
#                         and the prompt_mean reduction read (``rl/advantage.py``
#                         AdvantageFn.compute(prompt_ids=...)): ``uid``, or
#                         ``f"{uid}::{harness}"`` under group_by_harness=True so
#                         each harness gets its own baseline. The name is the RL
#                         plane's, kept so the batch plugs in without a rename.
#   prompt_token_ids      tuple[int] -- the prompt turns' token ids in turn order,
#                         concatenated: the context every attempt in the group
#                         shares.
#   response_ids          tuple[int] -- the response turns' token ids in turn order.
#   loss_mask             tuple[int] -- 1 exactly on tokens of generated
#                         ASSISTANT/TOOL_CALL turns. Tool results, observations, user
#                         and system text are never supervised.
#   rollout_logprobs      tuple[float | None] -- the logprob the rollout engine
#                         reported for that sampled token, None wherever it reported
#                         none (environment-injected tokens included). THE RULE THAT
#                         GOVERNS THIS COLUMN: these exist ONLY for a declared
#                         off-policy correction (an importance weighting that states
#                         which policy sampled the batch) and are NEVER used as
#                         old_logprobs -- the ``rl/trainer.py`` rule is that the
#                         policy's own logprobs are recomputed under ``no_grad`` at the
#                         step that needs them, because the rollout engine's scoring
#                         and the trainer's scoring are not one measurement and mixing
#                         them would silently move the correction's numerator under it.
#   turn_index            tuple[int] -- the owning Turn.index per response token.
#   segment_kind          tuple[SegmentKind] -- the owning turn's kind per response
#                         token (a str subclass carrying the lower-case name).
#   tool_call_error_mask  tuple[int] -- 1 on every token of a generated turn carrying
#                         any ToolCall with parse_error set. TURN granularity; see
#                         Trajectory.tool_call_error_mask.
#   repetition_mask       tuple[int] -- 1 on every token of a generated turn the
#                         harness flagged repetition_hit.
#   uid                   str -- prompt group id (prompt + scoring reference).
#   session_id            str -- one attempt within the group. Together with uid this
#                         is the row's identity and is unique across a batch.
#   harness               str -- which harness recorded the turns.
#   reward                float | None -- the attempt's score; an unmeasured score
#                         stays None and is NEVER substituted with 0.0.
#   abstention_reason     str | None -- non-empty exactly when reward is None.
#   termination           Termination -- how the attempt ended (str enum values).
#   policy_version_min    int | None -- lower bound of the policies that could have
#                         sampled this batch; pair-declared with max.
#   policy_version_max    int | None -- upper bound of the same range.
# ---------------------------------------------------------------------------

DECLARED_COLUMNS: tuple[str, ...] = (
    "prompt_ids",
    "prompt_token_ids",
    "response_ids",
    "loss_mask",
    "rollout_logprobs",
    "turn_index",
    "segment_kind",
    "tool_call_error_mask",
    "repetition_mask",
    "uid",
    "session_id",
    "harness",
    "reward",
    "abstention_reason",
    "termination",
    "policy_version_min",
    "policy_version_max",
)


def flatten(
    trajectories: Sequence[Trajectory], *, group_by_harness: bool = False
) -> ExperienceBatch:
    """Pack trajectories into one row per trajectory, columns exactly ``DECLARED_COLUMNS``.

        WHAT IS CLAIMED on success: a batch carrying precisely the declared seventeen
    columns in the declared order, one row per trajectory in caller order, every
    per-token column one entry per response token of its row, and a ``None`` reward
    kept as ``None``. ``metadata`` stays behind: harness-private strings never
    reach the batch, so nothing downstream can mistake them for a measurement.

        WHAT IS NOT CLAIMED: any ordering or grouping beyond the caller's row order
    (use :func:`group_by_uid`), and any row CONTENT check -- a column of empty
    strings is present and aligned.

        Two refusals, both against empty denominators. An empty input is the
    all([]) shape: a zero-row batch would carry every declared column, satisfy
    required-column presence, and have every duplicate/alignment check pass having
    measured nothing -- and green over an empty denominator is exactly the reading
    the RL gates exist to refuse. A repeated ``(uid, session_id)`` is the same
    failure at row scale: one attempt is one row, and a second row for it would
    count twice in every rate and its reward twice in every mean.

        The per-token equal-length check is a SEAM check and says so: the objects
    crossing :func:`flatten` come from adapter code this module does not control,
    and nothing guarantees one of them measured its masks against the ids it
    returned. It refuses naming the column and both lengths rather than emitting
    two tables pasted into one.
    """
    rows = tuple(trajectories)
    if not rows:
        raise TrajectoryRefusal(
            "flatten(): trajectories is empty: this is the all([]) shape -- every "
            "check this plane runs (duplicate attempts, per-token alignment, "
            "required-column presence) evaluates True over it having measured nothing, "
            "and a batch that grades green over an empty denominator is the exact "
            "reading the RL gates exist to refuse"
        )
    seen: dict[tuple[str, str], int] = {}
    for row_index, trajectory in enumerate(rows):
        if not isinstance(trajectory, Trajectory):
            raise TrajectoryRefusal(
                f"flatten(): element {row_index} is {_described(trajectory)}, not a Trajectory"
            )
        identity = (trajectory.uid, trajectory.session_id)
        first = seen.get(identity)
        if first is not None:
            raise TrajectoryRefusal(
                f"flatten(): rows {first} and {row_index} share trajectory uid={identity[0]!r} "
                f"session_id={identity[1]!r}: one attempt is one row -- a repeated attempt "
                f"would count twice in every rate and its reward twice in every mean"
            )
        seen[identity] = row_index

    if type(group_by_harness) is not bool:
        raise TrajectoryRefusal(
            f"flatten(): field 'group_by_harness' is {group_by_harness!r}: it is a real "
            f"bool -- 1 is not True"
        )
    prompt_ids: list[str] = []
    prompt_token_ids: list[tuple[int, ...]] = []
    response_ids: list[tuple[int, ...]] = []
    loss_mask: list[tuple[int, ...]] = []
    rollout_logprobs: list[tuple[float | None, ...]] = []
    turn_index: list[tuple[int, ...]] = []
    segment_kind: list[tuple[SegmentKind, ...]] = []
    tool_call_error_mask: list[tuple[int, ...]] = []
    repetition_mask: list[tuple[int, ...]] = []
    uid: list[str] = []
    session_id: list[str] = []
    harness: list[str] = []
    reward: list[float | None] = []
    abstention_reason: list[str | None] = []
    termination: list[Termination] = []
    policy_version_min: list[int | None] = []
    policy_version_max: list[int | None] = []

    for trajectory in rows:
        where = f"trajectory uid={trajectory.uid!r} session_id={trajectory.session_id!r}"
        ids = trajectory.response_token_ids
        response_is_reference = ids
        response_ids.append(response_is_reference)
        loss_mask.append(trajectory.loss_mask())
        rollout_logprobs.append(trajectory.rollout_logprobs())
        turn_index.append(trajectory.turn_index_per_token())
        segment_kind.append(trajectory.segment_kind_per_token())
        tool_call_error_mask.append(trajectory.tool_call_error_mask())
        repetition_mask.append(trajectory.repetition_mask())
        prompt_ids.append(
            f"{trajectory.uid}::{trajectory.harness}" if group_by_harness else trajectory.uid
        )
        prompt_token_ids.append(
            tuple(token for turn in trajectory.prompt_turns for token in turn.token_ids)
        )
        uid.append(trajectory.uid)
        session_id.append(trajectory.session_id)
        harness.append(trajectory.harness)
        reward.append(trajectory.reward)
        abstention_reason.append(trajectory.abstention_reason)
        termination.append(trajectory.termination)
        policy_version_min.append(trajectory.policy_version_min)
        policy_version_max.append(trajectory.policy_version_max)
        for name, values in (
            ("loss_mask", loss_mask[-1]),
            ("rollout_logprobs", rollout_logprobs[-1]),
            ("turn_index", turn_index[-1]),
            ("segment_kind", segment_kind[-1]),
            ("tool_call_error_mask", tool_call_error_mask[-1]),
            ("repetition_mask", repetition_mask[-1]),
        ):
            if len(values) != len(response_is_reference):
                raise TrajectoryRefusal(
                    f"flatten(): {where}: column {name!r} has {len(values)} entries for a "
                    f"response of {len(response_is_reference)} token(s) in column "
                    f"'response_ids': per-token columns line up one entry per response token "
                    f"or the row is refused -- a column that runs its own length is not "
                    f"columnar data, it is two tables pasted together"
                )

    columns: dict[str, Sequence[Any]] = {}
    columns["prompt_ids"] = tuple(prompt_ids)
    columns["prompt_token_ids"] = tuple(prompt_token_ids)
    columns["response_ids"] = tuple(response_ids)
    columns["loss_mask"] = tuple(loss_mask)
    columns["rollout_logprobs"] = tuple(rollout_logprobs)
    columns["turn_index"] = tuple(turn_index)
    columns["segment_kind"] = tuple(segment_kind)
    columns["tool_call_error_mask"] = tuple(tool_call_error_mask)
    columns["repetition_mask"] = tuple(repetition_mask)
    columns["uid"] = tuple(uid)
    columns["session_id"] = tuple(session_id)
    columns["harness"] = tuple(harness)
    columns["reward"] = tuple(reward)
    columns["abstention_reason"] = tuple(abstention_reason)
    columns["termination"] = tuple(termination)
    columns["policy_version_min"] = tuple(policy_version_min)
    columns["policy_version_max"] = tuple(policy_version_max)
    return ExperienceBatch(columns=columns, required=DECLARED_COLUMNS)


def group_by_uid(trajectories: Sequence[Trajectory]) -> Mapping[str, tuple[Trajectory, ...]]:
    """Group trajectories by ``uid``, first-seen order preserved for keys and rows.

        A read-only view: a caller that could re-group this mapping could also
        forget a row out of it, and a grouping with a missing attempt undercounts
    every rate computed over the group.

        An empty input returns an empty mapping rather than refusing. Grouping is a
    RE-ARRANGEMENT with nothing to measure and no verdict to grade; the vacuous
    pass this plane refuses lives in :func:`flatten`, where a zero-row batch would
    sit in a denominator. .
    """
    groups: dict[str, list[Trajectory]] = {}
    for position, trajectory in enumerate(trajectories):
        if not isinstance(trajectory, Trajectory):
            raise TrajectoryRefusal(
                f"group_by_uid(): element {position} is {_described(trajectory)}, not a Trajectory"
            )
        groups.setdefault(trajectory.uid, []).append(trajectory)
    grouped: dict[str, tuple[Trajectory, ...]] = {
        uid_key: tuple(group) for uid_key, group in groups.items()
    }
    return MappingProxyType(grouped)
