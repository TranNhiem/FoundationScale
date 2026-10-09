# Portions adapted from XiaomiMiMo/verl recipes/arvo/token_trace.py (commit a2ad9f61),
# Copyright the original authors, licensed under the Apache License, Version 2.0.
# Modifications Copyright (c) 2026 TranNhiem, licensed under the MIT License (see LICENSE).
# See THIRD_PARTY_NOTICES.md.
"""Incremental token bookkeeping for one rollout: a flat trace and its response mask.

The agent harness keeps its conversation as an append-only message list and
re-sends the whole list to the model every turn. A policy-gradient batch needs
the opposite shape: one flat token sequence plus a mask marking which tokens the
policy sampled (1) and which the environment injected (0). Re-rendering the full
message list each turn cannot produce it -- a chat template REWRITES tool-call
text and can strip earlier reasoning, so the ids a template renders for a turn
are NOT the ids the engine sampled for it -- so the trace is built
incrementally and a sampled turn is never re-rendered: its exact ids are already
in the buffer. Only environment-injected turns (tool results, error text) are
tokenised, together, through the delta renderer the later-slice harness module
owns.

This module imports the standard library only, so the offset arithmetic is
unit-testable on a laptop (see tests/agentic_rl/test_token_trace.py).

How this differs from the upstream file named in the attribution header:

* Upstream PADDED a short ``log_probs`` list with ``0.0`` and TRUNCATED a long
  one to the token count. Both are silent repairs -- ``0.0`` is a real logprob
  (p=1.0) and would read as the engine's report -- so here a length mismatch is
  a :class:`TokenTraceRefusal` naming both counts, and absence is recorded as
  ``None``, never ``0.0``.
* Upstream's ``finalize`` CLIPPED the response at ``response_length`` whenever
  the trace had run past it. A clip can split a turn mid-text and desync the
  mask from every message boundary, so here :meth:`TokenTrace.finalize` refuses
  and leaves the boundary where :meth:`TokenTrace.check_budget` put it.
* Upstream ASSERTED its preconditions (gone under ``-O``, and unnameable in a
  message). Here they are named refusals: empty or malformed token ids, an append
  before the prompt, a twice-installed prompt, a pre-filled buffer, and a
  message with no role -- which upstream silently counted as "not assistant" and
  would have tokenised a sampled turn a second time.
* ``fits()`` IS carried over, from the upstream tree's sibling design variant
  (the arvo variant lacks it): the pure predicate :meth:`check_budget` consults,
  so a caller can plan a message boundary without provoking the exception. That
  variant's image-placeholder span bookkeeping is NOT carried -- placeholder
  spans and image payloads are a model-modality concern and a generic token
  trace records tokens and masks only.
* :meth:`TokenTrace.check_budget` keeps upstream's boundary rule verbatim
  (``used + incoming >= response_length`` exhausts it), including its
  consequence that a guarded rollout fills at most ``response_length - 1``
  response tokens, while ``finalize`` accepts exactly ``response_length`` and
  refuses beyond. The two rules meet at the boundary and nothing here silently
  reconciles them.
* ``truncated`` stays as upstream's explicit record that the BUDGET stopped the
  rollout (nothing is ever cut here), and :class:`ResponseBudgetExhausted` stays
  outside the refusal hierarchy for the reason its docstring gives.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from math import isfinite
from typing import Any, TypeGuard

__all__ = (
    "ResponseBudgetExhausted",
    "TokenTrace",
    "TokenTraceRefusal",
    "select_delta_messages",
)


class TokenTraceRefusal(ValueError):
    # Raised when an append, a cursor, a budget request or a finalize would
    # violate the trace's stated contract. Where upstream padded a short
    # logprob list, truncated a long one or clipped an over-budget response,
    # this port refuses with the field named and BOTH counts stated. Fail
    # closed: the caller owns the boundary, so nothing here is repaired into
    # shape.
    pass


class ResponseBudgetExhausted(Exception):
    """The response budget cannot hold what is about to be appended.

        Deliberately NOT a :class:`TokenTraceRefusal`: refusals are fail-closed
    contract breaks that must propagate to a bug report, while this is EXPECTED
    control flow -- the rollout bridge catches it and exits cleanly with the partial
    trajectory intact, at a message boundary where ``check_budget`` was consulted
    first. Making it a refusal would let a caller catch budget exhaustion where it
    expected a contract violation and swallow the violation as a stop.

        Raised only from :meth:`TokenTrace.check_budget`, and always with
    ``truncated`` set: the trace records that the budget stopped the rollout and
    nothing was cut to fit it.
    """

    pass


def _is_int(value: object) -> TypeGuard[int]:
    # `type(value) is int`, not isinstance: a bool is refused as a count
    # ("1 is not True"), in this module exactly as in the contracts plane.
    return type(value) is int


def _is_float(value: object) -> TypeGuard[float]:
    return type(value) is float


def _is_real(value: object) -> TypeGuard[float]:
    # Narrowed to float for typing only; an int entry is stored VERBATIM.
    return type(value) in (int, float)


def _described(value: object) -> str:
    return f"{value!r} ({type(value).__name__})"


def _items(value: object, *, field_name: str, where: str) -> Sequence[object]:
    if isinstance(value, tuple):
        return value
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray, memoryview)):
        return value
    raise TokenTraceRefusal(
        f"{where}: field {field_name!r} is {_described(value)}: this field is a sequence, "
        f"and text or a bare scalar is not a sequence of values"
    )


def _token_ids(value: object, *, field_name: str, where: str) -> list[int]:
    # Frozen to a list of real ints (upstream copied a list too). These shape
    # checks duplicate the contracts plane's rather than importing them, so this
    # port keeps its upstream-stable property: standard library only, offset
    # arithmetic testable with no engine and no package state.
    items = _items(value, field_name=field_name, where=where)
    if not items:
        raise TokenTraceRefusal(
            f"{where}: field {field_name!r} is empty: appending nothing advances nothing and "
            f"almost always means a renderer returned no tokens for a message it was handed"
        )
    tokens: list[int] = []
    for position, item in enumerate(items):
        if not _is_int(item):
            raise TokenTraceRefusal(
                f"{where}: field {field_name!r} entry {position} is {_described(item)}: "
                f"token ids are real ints -- 1 is not True"
            )
        if item < 0:
            raise TokenTraceRefusal(
                f"{where}: field {field_name!r} entry {position} is {item}: token ids are >= 0"
            )
        tokens.append(item)
    return tokens


def _logprob_values(value: object, *, token_count: int, where: str) -> list[float]:
    items = _items(value, field_name="logprobs", where=where)
    if len(items) != token_count:
        raise TokenTraceRefusal(
            f"{where}: field 'logprobs' carries {len(items)} entries for {token_count} "
            f"token(s) in field 'token_ids': one logprob per sampled token, exactly -- "
            f"upstream padded a short list with 0.0 and truncated a long one, and both invent "
            f"probability where the engine reported none (0.0 is a real logprob meaning p=1)"
        )
    logprobs: list[float] = []
    for position, item in enumerate(items):
        if not _is_real(item):
            raise TokenTraceRefusal(
                f"{where}: field 'logprobs' entry {position} is {_described(item)}: a "
                f"logprob is a real number"
            )
        if _is_float(item) and not isfinite(item):
            raise TokenTraceRefusal(
                f"{where}: field 'logprobs' entry {position} is {item}: logprobs are "
                f"finite -- a nan or an inf would poison every mean it enters"
            )
        logprobs.append(item)
    return logprobs


def _checked_incoming(value: object, *, where: str) -> int:
    if not _is_int(value):
        raise TokenTraceRefusal(
            f"{where}: field 'incoming' is {_described(value)}: a token count is a real "
            f"int -- 1 is not True"
        )
    if value < 0:
        raise TokenTraceRefusal(f"{where}: field 'incoming' is {value}: a token count is >= 0")
    return value


def select_delta_messages(
    messages: Sequence[Mapping[str, Any]], cursor: int
) -> tuple[list[Mapping[str, Any]], int]:
    """The environment's messages appended since ``cursor``, and the cursor to store.

        Assistant turns are dropped: the trace already holds the exact token ids the
    policy emitted for them and re-rendering would produce different ids (the chat
    template reformats ``tool_calls`` and may strip reasoning from earlier turns).
    Every other role is returned in message order and must be rendered in ONE chat
    template call afterwards -- templates group consecutive tool messages into a
    single turn, so rendering them one at a time emits spurious turn headers.

        Refusals here have upstream counterparts that guessed instead: a cursor out
    of range was an ``assert`` (gone under ``-O`` and silent as an empty delta
    afterwards) and a message without a role fell through ``.get("role")`` into the
    "not assistant" branch, which would have tokenised a sampled assistant turn a
    second time and split the trace from the buffer. Both now refuse naming the
    cursor or the message with BOTH counts stated.

        WHAT IS NOT CLAIMED: any rendering. This function selects which messages need
    tokenising and never produces ids.
    """
    where = "select_delta_messages()"
    if not _is_int(cursor):
        raise TokenTraceRefusal(
            f"{where}: field 'cursor' is {_described(cursor)}: the cursor is a real int -- "
            f"1 is not True"
        )
    if not 0 <= cursor <= len(messages):
        raise TokenTraceRefusal(
            f"{where}: field 'cursor' is {cursor} with {len(messages)} message(s): the "
            f"cursor satisfies 0 <= cursor <= len(messages) -- past the end it would return "
            f"an empty delta and the rollout would stop tokenising messages it still renders"
        )
    delta: list[Mapping[str, Any]] = []
    for position in range(cursor, len(messages)):
        message = messages[position]
        if not isinstance(message, Mapping):
            raise TokenTraceRefusal(
                f"{where}: message {position} is {_described(message)}, not a Mapping: every "
                f"message carries at least the role it is rendered with"
            )
        role = message.get("role")
        if not isinstance(role, str) or not role:
            raise TokenTraceRefusal(
                f"{where}: message {position} has role {role!r}: every message names its "
                f"role with a non-empty str -- without one an injected turn cannot be told "
                f"from a sampled one, and a sampled turn rendered twice desyncs the trace "
                f"from the buffer"
            )
        if role != "assistant":
            delta.append(message)
    return delta, len(messages)


@dataclass
class TokenTrace:
    """The flat token sequence of one rollout, plus its response mask.

        Layout mirrors what the rollout's output container expects: ``token_ids`` is
    the whole conversation and the trailing ``len(response_mask)`` entries of it are
    the response; everything before is the prompt (the first rendered
    system+instance turn), installed once by :meth:`append_prompt`.

        Deliberately NOT frozen: this is a builder whose buffers grow one message at
    a time, and a frozen builder would hand back copies each turn, hiding exactly
    the mask/token desync :meth:`finalize` refuses. Every growth goes through a
    method that validates first; construction refuses a pre-filled buffer because a
    trace whose buffers arrived from somewhere else carries a mask no append here
    measured.

        ``response_logprobs`` is aligned 1:1 with ``response_mask`` at all times and
    carries ``None`` -- never ``0.0`` -- where no logprob was reported.
    :meth:`finalize` returns that alignment as a whole only when at least one
    sampled token actually carries a logprob, and ``None`` otherwise, which is how a
    consumer learns the engine reported none anywhere.
    """

    response_length: int
    """Token budget for the response region (the rollout's response length)."""

    token_ids: list[int] = field(default_factory=list)
    """Prompt + response token ids, in order."""

    response_mask: list[int] = field(default_factory=list)
    """1 for policy-sampled tokens, 0 for environment-injected ones."""

    response_logprobs: list[float | None] = field(default_factory=list)
    """Aligned with response_mask; None where the engine reported no logprob."""

    prompt_len: int = 0
    """Length of the leading prompt region. Set once by ``append_prompt``."""

    truncated: bool = False
    """True once the token BUDGET stopped the rollout (nothing was ever cut)."""

    _logprobs_supplied: bool = False
    """True once any append supplied logprobs (what finalize reports as a whole)."""

    def __post_init__(self) -> None:
        where = "TokenTrace()"
        if not _is_int(self.response_length):
            raise TokenTraceRefusal(
                f"{where}: field 'response_length' is {_described(self.response_length)}: "
                f"the budget is a real int -- 1 is not True"
            )
        if self.response_length < 1:
            raise TokenTraceRefusal(
                f"{where}: field 'response_length' is {self.response_length}: the budget is "
                f">= 1 -- a budget of 0 can hold no sampled token and would stop the rollout "
                f"before the first generate"
            )
        if self.token_ids or self.response_mask or self.response_logprobs or self.prompt_len:
            raise TokenTraceRefusal(
                f"{where}: buffer fields are pre-filled (token_ids={len(self.token_ids)}, "
                f"response_mask={len(self.response_mask)}, "
                f"response_logprobs={len(self.response_logprobs)}, "
                f"prompt_len={self.prompt_len}): this trace is built by append_prompt, "
                f"append_observation and append_generated in order, and a pre-filled buffer "
                f"carries a mask no append here measured"
            )

    def _require_prompt(self, where: str) -> None:
        if self.prompt_len == 0:
            raise TokenTraceRefusal(
                f"{where}: no prompt installed (prompt_len=0): append_prompt() runs first and "
                f"exactly once -- appending before it would leave the prompt/response offset "
                f"at 0 and every prompt token would read as sampled"
            )

    def append_prompt(self, token_ids: Sequence[int]) -> None:
        """Install the initial prompt. Exactly once, before anything else."""
        where = "append_prompt()"
        if self.prompt_len > 0:
            raise TokenTraceRefusal(
                f"{where}: field 'token_ids' already holds a prompt of {self.prompt_len} "
                f"token(s): append_prompt runs exactly once and a second prompt would move "
                f"the prompt/response boundary under tokens already recorded"
            )
        self.token_ids = _token_ids(token_ids, field_name="token_ids", where=where)
        self.prompt_len = len(self.token_ids)

    def append_observation(self, token_ids: Sequence[int]) -> None:
        """Append environment-injected tokens (tool results, error text). Masked out.

                These carry no logprob because nothing sampled them, and the aligned
        ``response_logprobs`` entries are ``None`` -- never ``0.0``.
        """
        where = "append_observation()"
        self._require_prompt(where)
        new_tokens = _token_ids(token_ids, field_name="token_ids", where=where)
        self.token_ids.extend(new_tokens)
        self.response_mask.extend([0] * len(new_tokens))
        self.response_logprobs.extend([None] * len(new_tokens))

    def append_generated(
        self, token_ids: Sequence[int], logprobs: Sequence[float] | None = None
    ) -> None:
        """Append policy-sampled tokens. Masked in; these are what the loss trains on.

        ``logprobs`` is optional and its absence is recorded per token as ``None``. When
        supplied it must be exactly one value per token -- where
        upstream padded and truncated to fit, this port names both counts and refuses,
        because a repair there invents probability for tokens the engine never scored.
        """
        where = "append_generated()"
        self._require_prompt(where)
        new_tokens = _token_ids(token_ids, field_name="token_ids", where=where)
        row: list[float | None]
        if logprobs is None:
            row = [None] * len(new_tokens)
        else:
            row = list(_logprob_values(logprobs, token_count=len(new_tokens), where=where))
            self._logprobs_supplied = True
        self.token_ids.extend(new_tokens)
        self.response_mask.extend([1] * len(new_tokens))
        self.response_logprobs.extend(row)

    def remaining_budget(self) -> int:
        """Response tokens still available in the budget. Never negative."""
        return max(0, self.response_length - len(self.response_mask))

    def fits(self, incoming: int) -> bool:
        """Whether ``incoming`` more response tokens stay strictly inside the budget.

                The pure predicate :meth:`check_budget` consults (carried from the
        design variant of the upstream tree), so a caller can plan a message boundary
        without provoking the exception. Upstream's boundary rule is kept verbatim: a
        response that lands exactly on ``response_length`` does not fit.
        """
        return (
            len(self.response_mask) + _checked_incoming(incoming, where="fits()")
            < self.response_length
        )

    def check_budget(self, incoming: int) -> None:
        """Raise :class:`ResponseBudgetExhausted` if ``incoming`` more tokens blow the budget.

        Called before rendering a tool observation and before each generation so the
        rollout stops at a MESSAGE boundary instead of mid-turn. On exhaustion
        ``truncated`` is set: the explicit record that the budget stopped the rollout,
        since nothing in this module ever cuts tokens to fit.
        """
        if not self.fits(_checked_incoming(incoming, where="check_budget()")):
            self.truncated = True
            raise ResponseBudgetExhausted(
                f"response budget exhausted: {len(self.response_mask)} used + {incoming} "
                f"incoming >= response_length {self.response_length}"
            )

    @property
    def prompt_ids(self) -> list[int]:
        """The prompt region's token ids (a copy: the buffer stays the owner)."""
        return self.token_ids[: self.prompt_len]

    @property
    def response_ids(self) -> list[int]:
        """The response region's token ids (a copy)."""
        return self.token_ids[self.prompt_len :]

    def finalize(self) -> tuple[list[int], list[int], list[int], list[float | None] | None]:
        """Return ``(prompt, response, mask, logprobs)`` for the container that wants them.

        The consumer requires response, mask and logprobs to be one length, so the
        alignment is MEASURED here and a desync is refused rather than returned for a
        later stage to
        hand, in which case nothing else would notice.

        Where upstream clipped the response at ``response_length``, this refuses: a clip
        can split a turn mid-text and desync the mask from every message boundary that
        :meth:`check_budget` placed; the caller owns an over-long response.

        ``logprobs`` is ``None`` when no sampled token anywhere carries one (the engine
        reported none) and otherwise the per-token alignment in which absence is
        ``None`` on every token the engine did not score.
        """
        where = "finalize()"
        self._require_prompt(where)
        response_ids = self.response_ids
        mask = list(self.response_mask)
        logprob_row = list(self.response_logprobs)
        if len(response_ids) != len(mask):
            raise TokenTraceRefusal(
                f"{where}: field 'response_mask' has {len(mask)} entries but field "
                f"'response_ids' has {len(response_ids)}: the mask is one entry per response "
                f"token and a desync is refused rather than returned -- the lists are public "
                f"and a caller may have grown them by hand"
            )
        if len(logprob_row) != len(mask):
            raise TokenTraceRefusal(
                f"{where}: field 'response_logprobs' has {len(logprob_row)} entries but "
                f"field 'response_mask' has {len(mask)}: one entry per response token, "
                f"``None`` where the engine reported none"
            )
        if len(response_ids) > self.response_length:
            raise TokenTraceRefusal(
                f"{where}: the response region holds {len(response_ids)} token(s) but field "
                f"'response_length' is {self.response_length}: this trace does not clip -- "
                f"upstream cut at the budget and returned the prefix, which can split a turn "
                f"mid-text; check_budget() stops the rollout at a message boundary and the "
                f"caller owns that boundary"
            )
        logprobs: list[float | None] | None = logprob_row if self._logprobs_supplied else None
        return (self.prompt_ids, response_ids, mask, logprobs)
