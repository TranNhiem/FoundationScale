"""Harness-facing contracts for the FoundationScale Agentic RL package.

This module is the CONTRACT plane between a rollout host and a concrete episode
harness: it declares the sampling knobs, the token-in/token-out generation client,
the chat renderer, the episode task/budget/outcome shapes, and the adapter
protocol. It deliberately imports nothing from ``envs`` -- a concrete adapter
narrows ``env: Any`` to the real environment protocol in its own signature.

``to_trajectory`` is the single assembly point that turns one harness attempt
into a validated ``contracts.Trajectory``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from math import isfinite
from types import MappingProxyType
from typing import Any, Literal, Protocol, runtime_checkable

from foundationscale.agentic_rl.contracts import (
    Termination,
    Trajectory,
    Turn,
)

__all__ = [
    "ChatTokenizer",
    "EpisodeBudget",
    "EpisodeOutcome",
    "EpisodeTask",
    "Generation",
    "GenerationClient",
    "HarnessAdapter",
    "HarnessRefusal",
    "SamplingParams",
    "to_trajectory",
]


class HarnessRefusal(ValueError):
    """Raised when a harness-plane argument violates its declared contract.

    Every message names the offending field, what was given, the rule, and the
    concrete failure the rule prevents.
    """


def _described(value: object) -> str:
    """Return a short ``type + repr`` description of ``value`` for refusal messages."""
    return f"{type(value).__name__} {value!r}"


def _real_number(value: object, *, where: str, name: str) -> float:
    """Return ``value`` as a float after proving it is a real, finite number.

    Refuses bool (which aliases int in Python) and non-finite values, so a
    sampler knob can never silently become ``True``/``False`` or NaN/inf.
    """
    if type(value) is bool or not isinstance(value, (int, float)):
        raise HarnessRefusal(
            f"{where}: field '{name}' is {_described(value)}: it must be a real number "
            f"(type(x) in (int, float), bool excluded) -- a bool or non-number would "
            f"silently change the sampler's behaviour instead of being an error"
        )
    number = float(value)
    if not isfinite(number):
        raise HarnessRefusal(
            f"{where}: field '{name}' is {_described(value)}: it must be finite -- "
            f"NaN or infinity cannot be honoured by a sampler and would make every "
            f"downstream comparison meaningless"
        )
    return number


def _real_int(value: object, *, where: str, name: str) -> int:
    """Return ``value`` as an int after proving it is a real int (bool refused)."""
    if type(value) is not int:
        raise HarnessRefusal(
            f"{where}: field '{name}' is {_described(value)}: it must be a real int "
            f"(type(x) is int, bool excluded) -- a bool or float would silently "
            f"reinterpret a count instead of being an error"
        )
    return value


def _non_empty_str(value: object, *, where: str, name: str) -> str:
    """Return ``value`` after proving it is a non-empty ``str``."""
    if not isinstance(value, str):
        raise HarnessRefusal(
            f"{where}: field '{name}' is {_described(value)}: it must be a str -- "
            f"a non-str identifier cannot name the object it labels"
        )
    if value == "":
        raise HarnessRefusal(
            f"{where}: field '{name}' is {_described(value)}: it must be a non-empty str -- "
            f"an empty identifier names nothing and cannot distinguish one object from another"
        )
    return value


def _token_ids(value: object, *, where: str) -> tuple[int, ...]:
    """Coerce a ``Sequence[int]`` to a tuple of real ints.

    Refuses non-sequences and bool/str/bytes elements (naming the element and its
    position) so a token id can never be an alias for something that is not an id.
    """
    if isinstance(value, (str, bytes, bytearray, memoryview)) or not isinstance(value, Sequence):
        raise HarnessRefusal(
            f"{where}: field 'token_ids' is {_described(value)}: it must be a Sequence[int] -- "
            f"a non-sequence carries no ordered token ids to sample from or append to"
        )
    ids: list[int] = []
    for position, element in enumerate(value):
        if type(element) is not int:
            raise HarnessRefusal(
                f"{where}: field 'token_ids' element at position {position} is "
                f"{_described(element)}: every token id must be a real int (type(x) is int, "
                f"bool excluded) -- a non-int id cannot address a vocabulary entry"
            )
        ids.append(element)
    return tuple(ids)


def _float_tuple(value: object, *, where: str, name: str, count: int) -> tuple[float, ...]:
    """Coerce a sequence of reals to a tuple of finite floats, refusing bad elements."""
    if isinstance(value, (str, bytes, bytearray, memoryview)) or not isinstance(value, Sequence):
        raise HarnessRefusal(
            f"{where}: field '{name}' is {_described(value)}: it must be a Sequence[float] -- "
            f"a non-sequence carries no per-token values to align with the tokens"
        )
    values: list[float] = []
    for position, element in enumerate(value):
        if type(element) is bool or type(element) not in (int, float):
            raise HarnessRefusal(
                f"{where}: field '{name}' element at position {position} is "
                f"{_described(element)}: every value must be a real finite number "
                f"(type(x) in (int, float), bool excluded) -- a non-number cannot be a "
                f"per-token score"
            )
        number = float(element)
        if not isfinite(number):
            raise HarnessRefusal(
                f"{where}: field '{name}' element at position {position} is "
                f"{_described(element)}: every value must be finite -- NaN or infinity "
                f"cannot be compared or averaged downstream"
            )
        values.append(number)
    if len(values) != count:
        raise HarnessRefusal(
            f"{where}: field '{name}' has {len(values)} values but field 'token_ids' has "
            f"{count} tokens: there must be exactly one value per token -- a misaligned "
            f"score would be attributed to the wrong token"
        )
    return tuple(values)


def _str_mapping(value: object, *, where: str, name: str) -> Mapping[str, str]:
    """Freeze a ``Mapping[str, str]`` into a read-only ``MappingProxyType`` copy.

    Refuses non-mappings and non-str keys/values so metadata can never carry a
    value that a later consumer would have to guess how to stringify.
    """
    if not isinstance(value, Mapping):
        raise HarnessRefusal(
            f"{where}: field '{name}' is {_described(value)}: it must be a Mapping[str, str] -- "
            f"a non-mapping carries no key/value metadata to record"
        )
    frozen: dict[str, str] = {}
    for key, val in value.items():
        if not isinstance(key, str):
            raise HarnessRefusal(
                f"{where}: field '{name}' key is {_described(key)}: every key must be a str -- "
                f"a non-str key cannot be looked up or reported consistently"
            )
        if not isinstance(val, str):
            raise HarnessRefusal(
                f"{where}: field '{name}' value for key {key!r} is {_described(val)}: every "
                f"value must be a str -- a non-str value would need silent stringification "
                f"before it could be stored or compared"
            )
        frozen[key] = val
    return MappingProxyType(frozen)


def _turn_list(value: object, *, where: str, name: str) -> tuple[Turn, ...]:
    """Coerce a sequence of ``Turn`` objects to a tuple, refusing non-``Turn`` elements."""
    if isinstance(value, (str, bytes, bytearray, memoryview)) or not isinstance(value, Sequence):
        raise HarnessRefusal(
            f"{where}: field '{name}' is {_described(value)}: it must be a Sequence[Turn] -- "
            f"a non-sequence carries no ordered turns to record"
        )
    turns: list[Turn] = []
    for position, element in enumerate(value):
        if not isinstance(element, Turn):
            raise HarnessRefusal(
                f"{where}: field '{name}' element at position {position} is "
                f"{_described(element)}: every element must be a Turn -- a non-Turn element "
                f"cannot be indexed, masked or scored as part of a trajectory"
            )
        turns.append(element)
    return tuple(turns)


@dataclass(frozen=True)
class SamplingParams:
    """Fully declared sampling knobs for one generation call.

    Every field is validated here as a real, finite number or a real int, but no
    field inherits a value from anywhere else: each sampler is declared in full by
    its caller. ``temperature`` is only required to be ``> 0`` when GROUP sampling
    (multiple samples per prompt) is used -- that rule is enforced by the caller
    (the rollout host in a later slice), NOT here; this dataclass only enforces
    "it is a real finite number".
    """

    temperature: float
    top_p: float
    top_k: int
    min_p: float
    presence_penalty: float
    repetition_penalty: float
    max_new_tokens: int

    def __post_init__(self) -> None:
        where = "SamplingParams"
        temperature = _real_number(self.temperature, where=where, name="temperature")
        top_p = _real_number(self.top_p, where=where, name="top_p")
        if not 0.0 < top_p <= 1.0:
            raise HarnessRefusal(
                f"{where}: field 'top_p' is {_described(self.top_p)}: it must satisfy "
                f"0 < top_p <= 1 -- a nucleus mass outside that open-closed range either "
                f"selects no token at all or keeps an unnormalised tail"
            )
        top_k = _real_int(self.top_k, where=where, name="top_k")
        if top_k < 0:
            raise HarnessRefusal(
                f"{where}: field 'top_k' is {_described(self.top_k)}: it must be >= 0 -- "
                f"a negative k would ask the sampler to keep fewer than zero candidates"
            )
        min_p = _real_number(self.min_p, where=where, name="min_p")
        if not 0.0 <= min_p < 1.0:
            raise HarnessRefusal(
                f"{where}: field 'min_p' is {_described(self.min_p)}: it must satisfy "
                f"0 <= min_p < 1 -- a threshold outside that half-open range would either "
                f"drop every candidate or keep the full unfiltered tail"
            )
        presence_penalty = _real_number(self.presence_penalty, where=where, name="presence_penalty")
        repetition_penalty = _real_number(
            self.repetition_penalty, where=where, name="repetition_penalty"
        )
        if repetition_penalty <= 0.0:
            raise HarnessRefusal(
                f"{where}: field 'repetition_penalty' is {_described(self.repetition_penalty)}: "
                f"it must be > 0 -- a non-positive penalty would divide token logits by zero "
                f"or flip their sign"
            )
        max_new_tokens = _real_int(self.max_new_tokens, where=where, name="max_new_tokens")
        if max_new_tokens < 1:
            raise HarnessRefusal(
                f"{where}: field 'max_new_tokens' is {_described(self.max_new_tokens)}: it must "
                f"be >= 1 -- a zero or negative token budget would end the generation before "
                f"the model could emit anything"
            )
        object.__setattr__(self, "temperature", temperature)
        object.__setattr__(self, "top_p", top_p)
        object.__setattr__(self, "top_k", top_k)
        object.__setattr__(self, "min_p", min_p)
        object.__setattr__(self, "presence_penalty", presence_penalty)
        object.__setattr__(self, "repetition_penalty", repetition_penalty)
        object.__setattr__(self, "max_new_tokens", max_new_tokens)


@dataclass(frozen=True)
class Generation:
    """One sampled continuation: the exact token ids, optional per-token logprobs,
    the stop reason, and the decoded text.

    ``logprobs`` must be exactly one real finite float per token when present. A
    ``Generation`` is always a sampled output by construction, so -- unlike
    ``Turn`` -- there is no ``generated`` flag to gate this against.
    """

    token_ids: tuple[int, ...]
    logprobs: tuple[float, ...] | None
    finish_reason: Literal["stop", "length", "abort"]
    text: str

    def __post_init__(self) -> None:
        where = "Generation"
        token_ids = _token_ids(self.token_ids, where=where)
        if self.logprobs is None:
            logprobs: tuple[float, ...] | None = None
        else:
            logprobs = _float_tuple(
                self.logprobs, where=where, name="logprobs", count=len(token_ids)
            )
        if self.finish_reason not in ("stop", "length", "abort"):
            raise HarnessRefusal(
                f"{where}: field 'finish_reason' is {_described(self.finish_reason)}: it must "
                f"be one of 'stop', 'length', 'abort' -- an unknown reason cannot be mapped to "
                f"a termination or told apart from a truncated generation"
            )
        if not isinstance(self.text, str):
            raise HarnessRefusal(
                f"{where}: field 'text' is {_described(self.text)}: it must be a str -- the "
                f"decoded text of the sampled ids must be text, not some other object"
            )
        object.__setattr__(self, "token_ids", token_ids)
        object.__setattr__(self, "logprobs", logprobs)


@runtime_checkable
class GenerationClient(Protocol):
    """Token-in/token-out generation client.

    Implementations must never re-tokenise ``prompt_ids`` or treat them as
    anything but opaque ids to continue from; the returned
    ``Generation.token_ids`` are the exact ids the policy sampled, appended
    verbatim by the caller to its trace.
    """

    async def generate(self, prompt_ids: Sequence[int], sampling: SamplingParams) -> Generation: ...


@runtime_checkable
class ChatTokenizer(Protocol):
    """Stateless chat-template renderer and decoder (token ids only, no sampling)."""

    def render(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        tools: Sequence[Mapping[str, Any]] | None,
        add_generation_prompt: bool,
    ) -> list[int]:
        """Render the WHOLE message list fresh every call and return its token ids.

        Callers needing a turn's exact tokens must diff two such renders (see
        ``token_trace.select_delta_messages``) rather than trust that a later
        render of an earlier prefix stays byte-identical to what was actually
        sampled for it.
        """
        ...

    def decode(self, ids: Sequence[int]) -> str:
        """Return the decoded text for ``ids``."""
        ...


@dataclass(frozen=True)
class EpisodeTask:
    """The system/user messages that seed one episode -- never an assistant message
    (the harness generates those).

    An assistant-authored seed message would be indistinguishable from a sampled
    turn once the episode starts, so it is refused here.
    """

    uid: str
    session_id: str
    messages: tuple[Mapping[str, str], ...]
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        where = "EpisodeTask"
        uid = _non_empty_str(self.uid, where=where, name="uid")
        session_id = _non_empty_str(self.session_id, where=where, name="session_id")
        declared_messages: object = self.messages
        if isinstance(declared_messages, (str, bytes, bytearray, memoryview)) or not isinstance(
            declared_messages, Sequence
        ):
            raise HarnessRefusal(
                f"{where}: field 'messages' is {_described(declared_messages)}: it must be a "
                f"Sequence[Mapping[str, str]] -- a non-sequence carries no seed messages to "
                f"start the episode from"
            )
        if len(declared_messages) == 0:
            raise HarnessRefusal(
                f"{where}: field 'messages' is {_described(declared_messages)}: it must be "
                f"non-empty -- an episode with no seed messages has no prompt to answer"
            )
        frozen_messages: list[Mapping[str, str]] = []
        for position, message in enumerate(declared_messages):
            if not isinstance(message, Mapping):
                raise HarnessRefusal(
                    f"{where}: field 'messages' element at position {position} is "
                    f"{_described(message)}: every message must be a Mapping[str, str] -- a "
                    f"non-mapping message carries no role or content to render"
                )
            if "role" not in message:
                raise HarnessRefusal(
                    f"{where}: field 'messages' element at position {position} is missing "
                    f"key 'role': every message must name its role -- an unroled message "
                    f"cannot be rendered into the chat template"
                )
            role = message["role"]
            if not isinstance(role, str) or role == "":
                raise HarnessRefusal(
                    f"{where}: field 'messages' element at position {position} has 'role' "
                    f"{_described(role)}: it must be a non-empty str -- a non-str or empty "
                    f"role cannot be rendered into the chat template"
                )
            if role == "assistant":
                raise HarnessRefusal(
                    f"{where}: field 'messages' element at position {position} has 'role' "
                    f"{_described(role)}: seed messages must never be assistant messages -- "
                    f"an assistant-authored seed would be indistinguishable from a sampled "
                    f"turn once the episode starts"
                )
            if "content" not in message:
                raise HarnessRefusal(
                    f"{where}: field 'messages' element at position {position} is missing "
                    f"key 'content': every message must carry its content -- a message without "
                    f"content cannot be rendered into the chat template"
                )
            content = message["content"]
            if not isinstance(content, str):
                raise HarnessRefusal(
                    f"{where}: field 'messages' element at position {position} has 'content' "
                    f"{_described(content)}: it must be a str -- a non-str content cannot be "
                    f"rendered into the chat template"
                )
            frozen_messages.append(MappingProxyType({"role": role, "content": content}))
        metadata = _str_mapping(self.metadata, where=where, name="metadata")
        object.__setattr__(self, "uid", uid)
        object.__setattr__(self, "session_id", session_id)
        object.__setattr__(self, "messages", tuple(frozen_messages))
        object.__setattr__(self, "metadata", metadata)


@dataclass(frozen=True)
class EpisodeBudget:
    """Hard resource ceilings for one episode attempt."""

    step_limit: int
    max_response_tokens: int
    max_observation_chars: int

    def __post_init__(self) -> None:
        where = "EpisodeBudget"
        step_limit = _real_int(self.step_limit, where=where, name="step_limit")
        if step_limit < 1:
            raise HarnessRefusal(
                f"{where}: field 'step_limit' is {_described(self.step_limit)}: it must be "
                f">= 1 -- a zero or negative step budget would end the episode before the "
                f"model could act"
            )
        max_response_tokens = _real_int(
            self.max_response_tokens, where=where, name="max_response_tokens"
        )
        if max_response_tokens < 1:
            raise HarnessRefusal(
                f"{where}: field 'max_response_tokens' is {_described(self.max_response_tokens)}: "
                f"it must be >= 1 -- a zero or negative token budget would end every response "
                f"before the model could emit anything"
            )
        max_observation_chars = _real_int(
            self.max_observation_chars, where=where, name="max_observation_chars"
        )
        if max_observation_chars < 64:
            raise HarnessRefusal(
                f"{where}: field 'max_observation_chars' is "
                f"{_described(self.max_observation_chars)}: it must be >= 64 -- an observation "
                f"budget below one short line can't carry even a minimal error message"
            )
        object.__setattr__(self, "step_limit", step_limit)
        object.__setattr__(self, "max_response_tokens", max_response_tokens)
        object.__setattr__(self, "max_observation_chars", max_observation_chars)


@dataclass(frozen=True)
class EpisodeOutcome:
    """What one ``HarnessAdapter.run_episode()`` call reports about ONE attempt,
    BEFORE any reward is known.

    Reward is a RewardService's job in a later slice, not this dataclass's or any
    harness's. ``abstention_reason`` is asserted here ONLY for a harness fault the
    harness is CERTAIN of (``Termination.INFRA``); any other abstention verdict is
    a scoring decision made later, outside this dataclass, by the code that builds
    the final ``contracts.Trajectory``.
    """

    trajectory_turns: tuple[Turn, ...]
    prompt_turns: tuple[Turn, ...]
    termination: Termination
    submitted_answer: str | None
    abstention_reason: str | None
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        where = "EpisodeOutcome"
        trajectory_turns = _turn_list(self.trajectory_turns, where=where, name="trajectory_turns")
        prompt_turns = _turn_list(self.prompt_turns, where=where, name="prompt_turns")
        for position, turn in enumerate(prompt_turns):
            if turn.generated is not False:
                raise HarnessRefusal(
                    f"{where}: field 'prompt_turns' element at position {position} has "
                    f"generated={turn.generated!r}: every prompt turn must have "
                    f"generated=False -- a generated prompt turn would be scored as model "
                    f"output it never produced"
                )
        if not isinstance(self.termination, Termination):
            raise HarnessRefusal(
                f"{where}: field 'termination' is {_described(self.termination)}: it must be "
                f"a Termination -- an unknown termination cannot be mapped to a trajectory "
                f"ending"
            )
        if self.submitted_answer is not None and not isinstance(self.submitted_answer, str):
            raise HarnessRefusal(
                f"{where}: field 'submitted_answer' is {_described(self.submitted_answer)}: "
                f"it must be None or a str -- a non-str answer cannot be recorded as the "
                f"model's final answer"
            )
        if self.termination is Termination.INFRA:
            if (
                not isinstance(self.abstention_reason, str)
                or self.abstention_reason == ""
                or not self.abstention_reason.startswith("infra:")
            ):
                raise HarnessRefusal(
                    f"{where}: field 'abstention_reason' is {_described(self.abstention_reason)}: "
                    f"under Termination.INFRA it must be a non-empty str starting with "
                    f"'infra:' -- an INFRA outcome must name the harness fault it is certain "
                    f"of, or downstream consumers cannot tell a real abstention from a bug"
                )
        elif self.abstention_reason is not None:
            raise HarnessRefusal(
                f"{where}: field 'abstention_reason' is {_described(self.abstention_reason)}: "
                f"outside Termination.INFRA it must be None -- this outcome only asserts an "
                f"abstention for a harness fault it is CERTAIN of; any other abstention "
                f"verdict is a scoring decision made later, outside this dataclass"
            )
        metadata = _str_mapping(self.metadata, where=where, name="metadata")
        object.__setattr__(self, "trajectory_turns", trajectory_turns)
        object.__setattr__(self, "prompt_turns", prompt_turns)
        object.__setattr__(self, "metadata", metadata)


@runtime_checkable
class HarnessAdapter(Protocol):
    """Protocol for one episode harness.

    ``env`` is typed ``Any`` here so this contract plane does not depend on
    ``envs.base.Environment``; a concrete adapter (e.g. ``NativeToolLoop``) narrows
    it to the real ``Environment`` protocol type in its own signature, which is
    allowed since Protocol method signatures are structurally, not nominally,
    checked.
    """

    name: str

    async def run_episode(
        self,
        task: EpisodeTask,
        *,
        client: GenerationClient,
        env: Any,
        budget: EpisodeBudget,
    ) -> EpisodeOutcome: ...


def to_trajectory(
    outcome: EpisodeOutcome,
    *,
    uid: str,
    session_id: str,
    harness: str,
    reward: float | None,
    abstention_reason: str | None,
    versions: tuple[int, int] | None = None,
    extra_metadata: Mapping[str, str] | None = None,
) -> Trajectory:
    """Assemble a ``contracts.Trajectory`` from one harness ``EpisodeOutcome``.

    ``reward`` and ``abstention_reason`` here are the CALLER's final verdict (after
    any RewardService scoring), not re-derived from ``outcome``. For an INFRA
    outcome the caller is expected to pass ``reward=None`` and
    ``abstention_reason=outcome.abstention_reason`` to propagate the harness's own
    verdict, but this function does not enforce that propagation itself: it
    constructs a ``Trajectory`` from exactly the arguments given and lets
    ``Trajectory.__post_init__`` refuse an inconsistent combination (e.g. a reward
    supplied under ``Termination.INFRA``), the same way every other ``Trajectory``
    caller is held to that contract. ``versions=None`` means "declare neither
    bound"; ``versions=(lo, hi)`` declares both. ``extra_metadata``, when given, is
    merged on top of ``outcome.metadata`` (its keys win on overlap) -- the one way
    a caller may record something the reward verdict learned (e.g. an
    informational note on a MEASURED score, which must never be smuggled into
    ``abstention_reason``) without it reading as an abstention. Any
    ``TrajectoryRefusal`` raised by the ``Trajectory(...)`` construction
    propagates unchanged.
    """
    metadata = (
        outcome.metadata if extra_metadata is None else {**outcome.metadata, **extra_metadata}
    )
    return Trajectory(
        uid=uid,
        session_id=session_id,
        harness=harness,
        prompt_turns=outcome.prompt_turns,
        turns=outcome.trajectory_turns,
        reward=reward,
        abstention_reason=abstention_reason,
        termination=outcome.termination,
        policy_version_min=versions[0] if versions is not None else None,
        policy_version_max=versions[1] if versions is not None else None,
        metadata=metadata,
    )
