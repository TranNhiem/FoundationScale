from __future__ import annotations

import dataclasses

import pytest

from foundationscale.upstream.levels import judge, level3_plan
from foundationscale.upstream.models import MODELS, ReferenceResult, SupportStatus


def test_plan_covers_exactly_the_supported_models() -> None:
    steps = level3_plan()
    assert {s.model_id for s in steps} == {
        e.id for e in MODELS if e.status is SupportStatus.SUPPORTED
    }
    by_id = {s.model_id: s for s in steps}
    assert by_id["canary-1b-flash"].entry_point == "foundationscale.upstream.nemo.decode"
    assert by_id["canary-1b-flash"].manifest == "librispeech_test_other.jsonl"
    assert by_id["canary-qwen-2.5b"].entry_point == "foundationscale.upstream.nemo.salm_decode"
    assert "--max-new-tokens" in by_id["canary-qwen-2.5b"].args
    assert by_id["parakeet-ctc-1.1b"].backend == "hf"
    assert by_id["parakeet-ctc-1.1b"].card_value == 1.83
    for step in steps:
        assert {"{model}", "{manifest}", "{out}"} <= set(step.args)


def test_a_supported_model_without_a_reference_value_is_refused() -> None:
    entry = next(e for e in MODELS if e.id == "canary-1b-flash")
    broken = dataclasses.replace(
        entry,
        reference=dataclasses.replace(entry.reference, upstream_value=None),  # type: ignore[arg-type]
    )
    with pytest.raises(ValueError, match="no reference value"):
        level3_plan((broken,))


def test_an_unknown_reference_task_is_refused() -> None:
    entry = next(e for e in MODELS if e.id == "canary-1b-flash")
    ref = ReferenceResult(
        task="common_voice_en",
        metric="wer",
        upstream_value=7.0,
        source="",
        tolerance=0.3,
        measured=7.0,
    )
    with pytest.raises(ValueError, match="no manifest known"):
        level3_plan((dataclasses.replace(entry, reference=ref),))


def test_judge_boundary_and_line() -> None:
    step = next(s for s in level3_plan() if s.model_id == "canary-qwen-2.5b")
    assert judge(step, 1.61 + 0.3).passed
    regressed = judge(step, 1.61 + 0.31)
    assert not regressed.passed
    assert regressed.line().startswith("[REGRESSION] canary-qwen-2.5b: 1.920 vs card 1.610")
    assert judge(step, 1.624).line().startswith("[PASS]")


def test_steps_carry_the_registry_upstream_ref() -> None:
    refs = {s.model_id: s.upstream_ref for s in level3_plan()}
    assert refs["canary-qwen-2.5b"] == "nvidia/canary-qwen-2.5b"
    assert refs["parakeet-ctc-1.1b"] == "nvidia/parakeet-ctc-1.1b"
