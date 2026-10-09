"""``RewardFn`` contract: one episode outcome scored into a verdict.

Torch-free and harness-agnostic: ``RewardFn`` never imports an engine, an
environment, or a harness implementation, only ``harness.base``'s task/outcome
carriers. ``MusicRewardFn`` is this slice's one adapter, wrapping
``rewards.music.MusicReward`` behind the protocol.
"""

from __future__ import annotations

import asyncio
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from foundationscale.agentic_rl.contracts import SegmentKind
from foundationscale.agentic_rl.harness.base import EpisodeOutcome, EpisodeTask
from foundationscale.agentic_rl.rewards.music import MusicReward

__all__ = (
    "MusicRewardFn",
    "RewardFn",
    "RewardFnRefusal",
    "RewardVerdict",
)


def _described(value: object) -> str:
    return f"{type(value).__name__} {value!r}"


class RewardFnRefusal(ValueError):
    """A config error constructing a ``RewardFn`` adapter, or a malformed
    ``RewardVerdict`` -- not a scoring-time infra fault, which is the adapter's
    own job to report THROUGH ``RewardVerdict`` (see ``MusicRewardFn``).
    """


@dataclass(frozen=True)
class RewardVerdict:
    """One episode's reward verdict.

    ``value`` is ``None`` (unmeasured) or a real finite score; ``reason`` is
    REQUIRED (a non-empty str) whenever ``value`` is ``None`` -- an unmeasured
    verdict must always name why, the same None-is-never-a-silent-zero rule
    ``contracts.Trajectory.reward`` uses. Unlike that biconditional, a MEASURED
    ``value`` may ALSO carry a ``reason``: an informational note (e.g. "no_abc",
    a model-attributable zero) does not make the score any less measured, so
    ``reason`` is optional, not forbidden, whenever ``value`` is not ``None``.
    """

    value: float | None
    reason: str | None

    def __post_init__(self) -> None:
        where = "RewardVerdict"
        if self.value is not None:
            if isinstance(self.value, bool) or not isinstance(self.value, (int, float)):
                raise RewardFnRefusal(
                    f"{where}: field 'value' is {_described(self.value)}: it must be None "
                    f"or a real number (bool excluded) -- a bool would silently stand in "
                    f"for a measured score"
                )
            if not math.isfinite(float(self.value)):
                raise RewardFnRefusal(
                    f"{where}: field 'value' is {self.value!r}: it must be finite -- a nan "
                    f"or inf score would poison every mean it enters"
                )
        if self.reason is not None and (not isinstance(self.reason, str) or not self.reason):
            raise RewardFnRefusal(
                f"{where}: field 'reason' is {_described(self.reason)}: it must be None or "
                f"a non-empty str -- an empty reason names no cause"
            )
        if self.value is None and self.reason is None:
            raise RewardFnRefusal(
                f"{where}: field 'value' is None and field 'reason' is None: reason is "
                f"required whenever value is None -- an unmeasured verdict must always "
                f"name why, never read as a silent, unexplained abstention"
            )


@runtime_checkable
class RewardFn(Protocol):
    """Scores one episode's outcome without knowing which harness or engine
    produced it.
    """

    async def score(self, task: EpisodeTask, outcome: EpisodeOutcome) -> RewardVerdict: ...


def _final_assistant_text(
    outcome: EpisodeOutcome, *, decode: Callable[[Sequence[int]], str]
) -> str | None:
    """The last generated ASSISTANT turn's decoded text, or ``None`` if the episode
    never produced one (e.g. every turn was INFRA/environment-injected).
    """
    for turn in reversed(outcome.trajectory_turns):
        if turn.generated and turn.kind is SegmentKind.ASSISTANT:
            return decode(turn.token_ids)
    return None


@dataclass(frozen=True)
class MusicRewardFn:
    """Adapter wrapping ``rewards.music.MusicReward`` behind ``RewardFn``.

    Scores ``outcome.submitted_answer`` when the episode submitted one, else the
    final generated ASSISTANT turn's text (decoded via ``decode``, supplied by
    the caller since this plane carries no tokenizer of its own). Runs the
    blocking ``MusicReward.score`` call in ``asyncio.to_thread`` so it never
    blocks the rollout host's event loop.
    """

    scorer: MusicReward
    decode: Callable[[Sequence[int]], str]

    def __post_init__(self) -> None:
        if not isinstance(self.scorer, MusicReward):
            raise RewardFnRefusal(
                f"MusicRewardFn: field 'scorer' is {_described(self.scorer)}, not a MusicReward"
            )
        if not callable(self.decode):
            raise RewardFnRefusal(
                f"MusicRewardFn: field 'decode' is {_described(self.decode)}: it must be "
                f"callable -- a non-callable cannot decode a turn's token ids"
            )

    async def score(self, _task: EpisodeTask, outcome: EpisodeOutcome) -> RewardVerdict:
        """Score the episode; ``task`` is unused by this domain's scorer (the music
        reward reads only the response text) and kept for ``RewardFn`` conformance.
        """
        text: str | None
        if outcome.submitted_answer is not None:
            text = outcome.submitted_answer
        else:
            text = _final_assistant_text(outcome, decode=self.decode)
        if text is None:
            # No assistant turn/answer exists to score at all -- this is UNMEASURED,
            # never a measured zero: value must stay None so a trainer's RL gates
            # never read "nothing to score" as "scored, and scored badly".
            return RewardVerdict(None, "no_assistant_turn")
        music_score = await asyncio.to_thread(self.scorer.score, text)
        return RewardVerdict(music_score.value, music_score.reason)
