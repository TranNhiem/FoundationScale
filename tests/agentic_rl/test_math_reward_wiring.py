"""reward.kind="math": config parsing, the RewardFn adapter, and the CLI dry run."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

from foundationscale.agentic_rl.cli import EXIT_PASS, main
from foundationscale.agentic_rl.config import AgenticConfigRefusal, build_config
from foundationscale.agentic_rl.contracts import Termination
from foundationscale.agentic_rl.harness.base import EpisodeOutcome, EpisodeTask
from foundationscale.agentic_rl.rewards.base import MathRewardFn, RewardFnRefusal
from foundationscale.agentic_rl.rewards.math import MathAnswerReward
from foundationscale.agentic_rl.rewards.swe import SweReward

REPO = Path(__file__).resolve().parents[2]
MUSIC_CONFIG = REPO / "examples" / "agentic_rl" / "music_s0.json"


def _raw(reward: dict[str, Any]) -> dict[str, Any]:
    raw = json.loads(MUSIC_CONFIG.read_text())
    raw["rollout"]["tasks_path"] = str(REPO / raw["rollout"]["tasks_path"])
    raw["reward"] = reward
    return raw


def test_math_reward_needs_no_abc2midi() -> None:
    config = build_config(_raw({"kind": "math"}), {})
    assert config.reward.kind == "math"
    assert config.reward.abc2midi_bin is None


def test_math_reward_refuses_an_unused_abc2midi() -> None:
    with pytest.raises(AgenticConfigRefusal, match="never uses it"):
        build_config(_raw({"kind": "math", "abc2midi_bin": "abc2midi"}), {})


def test_music_reward_still_requires_abc2midi() -> None:
    with pytest.raises(AgenticConfigRefusal, match="abc2midi_bin"):
        build_config(_raw({"kind": "music"}), {})


def test_unknown_reward_kind_is_refused() -> None:
    with pytest.raises(AgenticConfigRefusal, match="only 'music', 'math'"):
        build_config(_raw({"kind": "code"}), {})


def _outcome(submitted: str | None) -> EpisodeOutcome:
    return EpisodeOutcome(
        trajectory_turns=(),
        prompt_turns=(),
        termination=Termination.STOP,
        submitted_answer=submitted,
        abstention_reason=None,
    )


def _task(metadata: dict[str, str]) -> EpisodeTask:
    return EpisodeTask(
        uid="m-1", session_id="m-1#0", messages=({"role": "user", "content": "1+1?"},),
        metadata=metadata,
    )  # fmt: skip


def _score(task: EpisodeTask, outcome: EpisodeOutcome) -> tuple[float | None, str | None]:
    fn = MathRewardFn(scorer=MathAnswerReward(), decode=lambda ids: "")
    verdict = asyncio.run(fn.score(task, outcome))
    return verdict.value, verdict.reason


def test_adapter_reads_the_reference_from_task_metadata() -> None:
    assert _score(_task({"answer": "2"}), _outcome(r"\boxed{2}")) == (1.0, "correct")
    assert _score(_task({"answer": "3"}), _outcome(r"\boxed{2}")) == (0.0, "wrong")
    assert _score(_task({}), _outcome(r"\boxed{2}")) == (0.0, "no_reference")


def test_adapter_without_any_assistant_text_is_unmeasured() -> None:
    assert _score(_task({"answer": "2"}), _outcome(None)) == (None, "no_assistant_turn")


def test_adapter_refuses_a_wrong_scorer() -> None:
    with pytest.raises(RewardFnRefusal, match="not a MathAnswerReward"):
        MathRewardFn(scorer=object(), decode=lambda ids: "")  # type: ignore[arg-type]


def test_cli_dry_run_accepts_a_math_config(tmp_path: Path) -> None:
    path = tmp_path / "math.json"
    path.write_text(json.dumps(_raw({"kind": "math"})))
    assert main(["--config", str(path), "--dry-run"]) == EXIT_PASS


def test_adapter_refuses_a_non_callable_decode() -> None:
    with pytest.raises(RewardFnRefusal, match="must be callable"):
        MathRewardFn(scorer=MathAnswerReward(), decode="not callable")  # type: ignore[arg-type]


def test_real_run_builds_the_math_reward_adapter() -> None:
    import foundationscale.agentic_rl.cli as cli_module

    host = cli_module._build_real_run(build_config(_raw({"kind": "math"}), {}))
    assert isinstance(host.reward, MathRewardFn)


def test_adapter_accepts_a_bare_submitted_answer() -> None:
    assert _score(_task({"answer": "1/2"}), _outcome("0.5")) == (1.0, "correct")


def test_swe_reward_kind_builds_the_swe_reward() -> None:
    import foundationscale.agentic_rl.cli as cli_module

    config = build_config(_raw({"kind": "swe"}), {})
    assert config.reward.kind == "swe"
    assert isinstance(cli_module._build_real_run(config).reward, SweReward)
    with pytest.raises(AgenticConfigRefusal, match="kind='swe' never uses it"):
        build_config(_raw({"kind": "swe", "abc2midi_bin": "abc2midi"}), {})
