"""ARL-G02 wiring: ``reward.kind="judge"`` config keys and the CLI's startup gate.

The judge reward itself is pinned by ``test_judge_reward.py``; this file pins that
the config plane resolves/refuses the ``reward.judge_*`` keys and that the CLI
builds a judge only behind the calibration gate and a present key variable.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest

from foundationscale.agentic_rl import cli
from foundationscale.agentic_rl.config import AgenticConfigRefusal, build_config
from foundationscale.agentic_rl.rewards.judge import (
    CalibrationReport,
    JudgeCalibrationRefusal,
    OpenAIJudgeClient,
)


def _raw_config() -> dict[str, Any]:
    spec = importlib.util.spec_from_file_location(
        "_test_config_raw_judge", Path(__file__).with_name("test_config.py")
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    raw: dict[str, Any] = module._raw_config()
    return raw


def _rubric(tmp_path: Path) -> str:
    path = tmp_path / "rubric.json"
    path.write_text(
        json.dumps(
            {
                "rubric_id": "report_v1",
                "instructions": "Grade the report.",
                "criteria": [
                    {"name": "correct", "description": "Facts are right.", "weight": 2.0},
                    {"name": "complete", "description": "Covers the task.", "weight": 1.0},
                ],
            }
        ),
        encoding="utf-8",
    )
    return str(path)


def _judge_raw(tmp_path: Path, **extra: Any) -> dict[str, Any]:
    raw = _raw_config()
    raw["reward"] = {
        "kind": "judge",
        "judge_base_url": "http://judge.invalid:8000",
        "judge_model": "judge-model",
        "judge_rubric_path": _rubric(tmp_path),
        "judge_calibration_path": str(tmp_path / "calibration.json"),
        **extra,
    }
    return raw


def test_judge_keys_resolve_with_declared_defaults(tmp_path: Path) -> None:
    config = build_config(_judge_raw(tmp_path), {})
    judge = config.reward.judge
    assert config.reward.kind == "judge" and judge is not None
    assert (judge.votes, judge.min_n, judge.min_agreement, judge.api_key_env) == (1, 30, 0.8, None)


def test_a_judge_key_under_another_reward_kind_is_refused(tmp_path: Path) -> None:
    raw = _raw_config()
    raw["reward"] = {**raw["reward"], "judge_votes": 3}
    with pytest.raises(AgenticConfigRefusal, match="reward.judge_votes"):
        build_config(raw, {})


@pytest.mark.parametrize(
    ("extra", "match"),
    [
        ({"judge_min_agreement": 0.0}, "judge_min_agreement"),
        ({"judge_votes": 0}, "judge_votes"),
        ({"judge_max_spread": 1.5}, "judge_max_spread"),
    ],
)
def test_out_of_range_judge_settings_are_refused(
    tmp_path: Path, extra: dict[str, Any], match: str
) -> None:
    with pytest.raises(AgenticConfigRefusal, match=match):
        build_config(_judge_raw(tmp_path, **extra), {})


def test_a_missing_required_judge_key_is_refused(tmp_path: Path) -> None:
    raw = _judge_raw(tmp_path)
    del raw["reward"]["judge_model"]
    with pytest.raises(AgenticConfigRefusal, match="reward.judge_model"):
        build_config(raw, {})


def test_cli_refuses_an_uncalibrated_judge(tmp_path: Path) -> None:
    config = build_config(_judge_raw(tmp_path), {})
    with pytest.raises(JudgeCalibrationRefusal):  # no calibration file at all
        cli._reward_scorer(config)
    CalibrationReport(rubric_id="report_v1", n=50, agreement=0.6, threshold=0.5).to_json(
        tmp_path / "calibration.json"
    )
    with pytest.raises(JudgeCalibrationRefusal, match="agreement"):
        cli._reward_scorer(config)


def test_cli_refuses_an_unset_key_variable_naming_only_the_variable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("FS_TEST_JUDGE_KEY", raising=False)
    config = build_config(_judge_raw(tmp_path, judge_api_key_env="FS_TEST_JUDGE_KEY"), {})
    with pytest.raises(AgenticConfigRefusal, match="FS_TEST_JUDGE_KEY"):
        cli._reward_scorer(config)


def test_cli_builds_a_calibrated_judge_without_holding_the_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FS_TEST_JUDGE_KEY", "s3cr3t-value")
    CalibrationReport(rubric_id="report_v1", n=50, agreement=0.9, threshold=0.5).to_json(
        tmp_path / "calibration.json"
    )
    config = build_config(_judge_raw(tmp_path, judge_api_key_env="FS_TEST_JUDGE_KEY"), {})
    setup = cli._reward_scorer(config)
    assert isinstance(setup, cli._JudgeSetup)
    assert isinstance(setup.client, OpenAIJudgeClient)
    assert [c.name for c in setup.rubric.criteria] == ["correct", "complete"]
    assert "s3cr3t-value" not in repr(vars(setup.client))
