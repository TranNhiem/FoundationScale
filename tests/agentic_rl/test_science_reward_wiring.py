"""reward.kind="science": config, CLI and the RewardFn adapter (atlas science domains, slice 1)."""

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
from foundationscale.agentic_rl.rewards.base import RewardFnRefusal, ScienceRewardFn
from foundationscale.agentic_rl.rewards.science import ScienceAnswerReward

REPO = Path(__file__).resolve().parents[2]
MUSIC_CONFIG = REPO / "examples" / "agentic_rl" / "music_s0.json"
PUBMED = {"checker": "label", "answer": "yes", "choices": '["yes", "no", "maybe"]'}


def _raw(reward: dict[str, Any]) -> dict[str, Any]:
    raw = json.loads(MUSIC_CONFIG.read_text())
    raw["rollout"]["tasks_path"] = str(REPO / raw["rollout"]["tasks_path"])
    raw["reward"] = reward
    return raw


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
        uid="s-1", session_id="s-1#0", messages=({"role": "user", "content": "q?"},),
        metadata=metadata,
    )  # fmt: skip


def _score(task: EpisodeTask, outcome: EpisodeOutcome) -> tuple[float | None, str | None]:
    fn = ScienceRewardFn(scorer=ScienceAnswerReward(), decode=lambda ids: "")
    verdict = asyncio.run(fn.score(task, outcome))
    return verdict.value, verdict.reason


def test_science_kind_needs_no_abc2midi_and_refuses_an_unused_one() -> None:
    assert build_config(_raw({"kind": "science"}), {}).reward.kind == "science"
    with pytest.raises(AgenticConfigRefusal, match="kind='science' never uses it"):
        build_config(_raw({"kind": "science", "abc2midi_bin": "abc2midi"}), {})


def test_unknown_kind_refusal_names_science() -> None:
    with pytest.raises(AgenticConfigRefusal, match="'judge' and 'science'"):
        build_config(_raw({"kind": "chemistry"}), {})


def test_adapter_scores_from_task_metadata() -> None:
    assert _score(_task(PUBMED), _outcome(r"\boxed{yes}")) == (1.0, "correct")
    assert _score(_task(PUBMED), _outcome(r"\boxed{no}")) == (0.0, "wrong")
    assert _score(_task(PUBMED), _outcome("C")) == (0.0, "wrong")  # bare submitted letter
    value, note = _score(_task({}), _outcome(r"\boxed{yes}"))
    assert value == 0.0 and note is not None and note.startswith("bad_task:checker")


def test_adapter_without_any_assistant_text_is_unmeasured() -> None:
    assert _score(_task(PUBMED), _outcome(None)) == (None, "no_assistant_turn")


def test_adapter_refusals() -> None:
    with pytest.raises(RewardFnRefusal, match="not a ScienceAnswerReward"):
        ScienceRewardFn(scorer=object(), decode=lambda ids: "")  # type: ignore[arg-type]
    with pytest.raises(RewardFnRefusal, match="must be callable"):
        ScienceRewardFn(scorer=ScienceAnswerReward(), decode="x")  # type: ignore[arg-type]


def test_cli_dry_run_and_real_run_build_the_science_adapter(tmp_path: Path) -> None:
    import foundationscale.agentic_rl.cli as cli_module

    path = tmp_path / "science.json"
    path.write_text(json.dumps(_raw({"kind": "science"})))
    assert main(["--config", str(path), "--dry-run"]) == EXIT_PASS
    host = cli_module._build_real_run(build_config(_raw({"kind": "science"}), {}))
    assert isinstance(host.reward, ScienceRewardFn)
