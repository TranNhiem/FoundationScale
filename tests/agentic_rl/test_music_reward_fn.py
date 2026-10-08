"""Tests for ``rewards.base.MusicRewardFn``, the ``RewardFn`` adapter over
``rewards.music.MusicReward``.

No real ``abc2midi`` binary: ``abc2midi_bin="definitely-not-a-real-binary-xyz"``
(never on PATH) is used throughout, which is enough to exercise BOTH of
``MusicReward.score``'s outcomes without any fake binary script --
``"infra:abc2midi_missing"`` (ABC notation present) and ``"no_abc"`` (no ABC
notation at all, which never even reaches the binary).

WHAT IS CLAIMED: ``outcome.submitted_answer`` is scored when present, even
over a present ASSISTANT turn; otherwise the LAST generated ASSISTANT turn's
text (decoded via the supplied ``decode``) is scored; an episode with neither
scores ``RewardVerdict(None, "no_assistant_turn")`` -- UNMEASURED, since there
is nothing to score, never a measured zero that would read as "scored, and
scored badly"; ``RewardVerdict.value``/``.reason`` pass the underlying
``MusicScore`` through unchanged, including the ``None``-valued abstention
case; construction refuses a non-``MusicReward`` scorer or a non-callable
``decode``.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence

import pytest

from foundationscale.agentic_rl.contracts import SegmentKind, Termination, Turn
from foundationscale.agentic_rl.harness.base import EpisodeOutcome, EpisodeTask
from foundationscale.agentic_rl.rewards.base import MusicRewardFn, RewardFnRefusal, RewardVerdict
from foundationscale.agentic_rl.rewards.music import MusicReward

_MISSING_BIN = "definitely-not-a-real-binary-xyz"
_ABC_TEXT = "X:1\nT:Test\nK:C\nCDEF|"


def _decode(ids: Sequence[int]) -> str:
    # The tiny fixed vocabulary this test suite's fake tokenizers share:
    # even ids decode to ABC_TEXT, odd ids decode to plain prose.
    return _ABC_TEXT if ids and ids[0] % 2 == 0 else "just some prose, no notation here"


def _task() -> EpisodeTask:
    return EpisodeTask(uid="u1", session_id="s1", messages=({"role": "user", "content": "go"},))


def _assistant_turn(index: int, token_ids: tuple[int, ...]) -> Turn:
    return Turn(
        index=index, kind=SegmentKind.ASSISTANT, token_ids=token_ids, generated=True, logprobs=None
    )


def _tool_result_turn(index: int, token_ids: tuple[int, ...]) -> Turn:
    return Turn(
        index=index,
        kind=SegmentKind.TOOL_RESULT,
        token_ids=token_ids,
        generated=False,
        logprobs=None,
    )


def _outcome(**overrides: object) -> EpisodeOutcome:
    fields: dict[str, object] = {
        "trajectory_turns": (_assistant_turn(0, (10,)),),
        "prompt_turns": (
            Turn(index=0, kind=SegmentKind.USER, token_ids=(1,), generated=False, logprobs=None),
        ),
        "termination": Termination.STOP,
        "submitted_answer": None,
        "abstention_reason": None,
        "metadata": {},
    }
    fields.update(overrides)
    return EpisodeOutcome(**fields)  # type: ignore[arg-type]


def _reward_fn(**overrides: object) -> MusicRewardFn:
    fields: dict[str, object] = {
        "scorer": MusicReward(abc2midi_bin=_MISSING_BIN),
        "decode": _decode,
    }
    fields.update(overrides)
    return MusicRewardFn(**fields)  # type: ignore[arg-type]


def _score(reward_fn: MusicRewardFn, task: EpisodeTask, outcome: EpisodeOutcome) -> RewardVerdict:
    return asyncio.run(reward_fn.score(task, outcome))


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_rejects_non_music_reward_scorer() -> None:
    with pytest.raises(RewardFnRefusal):
        MusicRewardFn(scorer="not a scorer", decode=_decode)  # type: ignore[arg-type]


def test_rejects_non_callable_decode() -> None:
    with pytest.raises(RewardFnRefusal):
        MusicRewardFn(scorer=MusicReward(abc2midi_bin=_MISSING_BIN), decode="not callable")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Text source selection: submitted_answer > final assistant turn > neither
# ---------------------------------------------------------------------------


def test_scores_submitted_answer_when_present() -> None:
    reward_fn = _reward_fn()
    outcome = _outcome(submitted_answer=_ABC_TEXT, trajectory_turns=(_assistant_turn(0, (11,)),))
    verdict = _score(reward_fn, _task(), outcome)
    assert verdict.value is None
    assert verdict.reason == "infra:abc2midi_missing"


def test_submitted_answer_takes_priority_over_assistant_turn_text() -> None:
    # decode(11) -> prose (odd id), but submitted_answer is ABC: submitted_answer wins.
    reward_fn = _reward_fn()
    outcome = _outcome(submitted_answer=_ABC_TEXT, trajectory_turns=(_assistant_turn(0, (11,)),))
    verdict = _score(reward_fn, _task(), outcome)
    assert verdict.reason == "infra:abc2midi_missing"


def test_falls_back_to_final_assistant_turn_when_no_submitted_answer() -> None:
    reward_fn = _reward_fn()
    outcome = _outcome(
        submitted_answer=None,
        trajectory_turns=(
            _tool_result_turn(0, (99,)),
            _assistant_turn(1, (10,)),  # even -> decodes to ABC_TEXT
        ),
    )
    verdict = _score(reward_fn, _task(), outcome)
    assert verdict.reason == "infra:abc2midi_missing"


def test_uses_the_last_generated_assistant_turn_not_the_first() -> None:
    reward_fn = _reward_fn()
    outcome = _outcome(
        submitted_answer=None,
        trajectory_turns=(
            _assistant_turn(0, (11,)),  # odd -> prose (would score "no_abc" if used)
            _tool_result_turn(1, (99,)),
            _assistant_turn(2, (10,)),  # even -> ABC_TEXT, and it is the LAST assistant turn
        ),
    )
    verdict = _score(reward_fn, _task(), outcome)
    assert verdict.reason == "infra:abc2midi_missing"


def test_no_submitted_answer_and_no_assistant_turn_is_unmeasured() -> None:
    # Regression: an episode with nothing to score (no submitted_answer and no
    # generated ASSISTANT turn) is UNMEASURED, not a measured zero -- value must
    # be None, never 0.0, or a trainer's RL gates would read "nothing to score"
    # as "scored, and scored badly".
    reward_fn = _reward_fn()
    outcome = _outcome(
        submitted_answer=None,
        trajectory_turns=(_tool_result_turn(0, (99,)),),
        termination=Termination.INFRA,
        abstention_reason="infra:engine_transport",
    )
    verdict = _score(reward_fn, _task(), outcome)
    assert verdict == RewardVerdict(None, "no_assistant_turn")


# ---------------------------------------------------------------------------
# RewardVerdict passthrough of the underlying MusicScore
# ---------------------------------------------------------------------------


def test_no_abc_in_text_is_a_measured_zero_with_reason() -> None:
    reward_fn = _reward_fn()
    outcome = _outcome(submitted_answer="just prose, no notation")
    verdict = _score(reward_fn, _task(), outcome)
    assert verdict.value == 0.0
    assert verdict.reason == "no_abc"


def test_missing_binary_with_abc_present_is_an_unmeasured_infra_abstention() -> None:
    reward_fn = _reward_fn()
    outcome = _outcome(submitted_answer=_ABC_TEXT)
    verdict = _score(reward_fn, _task(), outcome)
    assert verdict.value is None
    assert verdict.reason == "infra:abc2midi_missing"


# ---------------------------------------------------------------------------
# RewardVerdict's own construction-time validation
# ---------------------------------------------------------------------------


def test_reward_verdict_rejects_bool_value() -> None:
    with pytest.raises(RewardFnRefusal):
        RewardVerdict(True, None)  # type: ignore[arg-type]


def test_reward_verdict_rejects_non_finite_value() -> None:
    with pytest.raises(RewardFnRefusal):
        RewardVerdict(float("nan"), "measured")


def test_reward_verdict_rejects_empty_reason() -> None:
    with pytest.raises(RewardFnRefusal):
        RewardVerdict(1.0, "")


def test_reward_verdict_requires_a_reason_when_value_is_none() -> None:
    with pytest.raises(RewardFnRefusal):
        RewardVerdict(None, None)


def test_reward_verdict_allows_a_measured_value_with_an_informational_reason() -> None:
    verdict = RewardVerdict(0.0, "no_abc")
    assert verdict.value == 0.0
    assert verdict.reason == "no_abc"
