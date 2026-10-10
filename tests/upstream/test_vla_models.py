"""The VLA model registry: self-consistent, honest about tolerance, and wired into Level 3."""

from __future__ import annotations

import dataclasses

import pytest

from foundationscale.upstream.levels import judge, level3_plan
from foundationscale.upstream.models import (
    MODELS,
    Backend,
    ModelKind,
    SupportStatus,
    registry_problems,
)
from foundationscale.upstream.profiles import get_profile
from foundationscale.upstream.vla_models import VLA_MODELS, binomial_tolerance


def test_vla_registry_passes_registry_problems() -> None:
    assert registry_problems(VLA_MODELS) == []


def test_vla_entries_are_kept_out_of_the_speech_table() -> None:
    assert not {e.id for e in VLA_MODELS} & {e.id for e in MODELS}
    assert all(e.kind is ModelKind.VLA for e in VLA_MODELS)
    assert {e.backend for e in VLA_MODELS} == {Backend.GR00T, Backend.OPENPI}


@pytest.mark.parametrize(
    ("model_id", "n_published", "n_measured"),
    [("gr00t-n1.7-libero-spatial", 200, 606), ("openpi-pi05-libero", 500, 500)],
)
def test_tolerance_comes_from_the_published_rate_and_trial_counts(
    model_id: str, n_published: int, n_measured: int
) -> None:
    """Fixed from the card, not from our gap: the registered value covers the binomial margin."""
    ref = next(e for e in VLA_MODELS if e.id == model_id).reference
    assert ref is not None and ref.upstream_value is not None
    margin = binomial_tolerance(ref.upstream_value / 100.0, n_published, n_measured)
    assert margin <= ref.tolerance < margin + 0.1


def test_every_supported_vla_entry_actually_reproduced() -> None:
    for entry in VLA_MODELS:
        assert entry.status is SupportStatus.SUPPORTED
        assert entry.reference is not None and entry.reference.reproduced


def test_gr00t_card_value_is_the_counts_not_the_readme_label() -> None:
    """The README says '195/200 (97.65%)'; 195/200 is 97.5, and the counts are transcribed."""
    ref = next(e for e in VLA_MODELS if e.id.startswith("gr00t")).reference
    assert ref is not None and ref.upstream_value == pytest.approx(100.0 * 195 / 200)


def test_level3_plan_schedules_each_vla_model_through_the_fs_harness() -> None:
    steps = level3_plan(VLA_MODELS)
    assert [s.model_id for s in steps] == [e.id for e in VLA_MODELS]
    for step in steps:
        assert step.entry_point == "foundationscale.vla.eval"
        assert step.manifest == "libero_spatial"
        assert step.args[:2] == ("--backend", step.backend)
        assert {"{model}", "{manifest}", "{out}"} <= set(step.args)
        # each VLA step runs in its own backend's recorded environment (rule 3)
        assert get_profile(step.profile).backend == step.backend


def test_speech_level3_plan_is_unchanged_by_the_vla_branch() -> None:
    assert all(s.entry_point != "foundationscale.vla.eval" for s in level3_plan())


def test_judge_flags_a_vla_regression_beyond_tolerance() -> None:
    step = level3_plan(VLA_MODELS)[0]
    assert judge(step, step.card_value - step.tolerance).passed
    assert not judge(step, step.card_value - step.tolerance - 0.5).passed


def test_a_vla_entry_without_a_reproduction_cannot_claim_support() -> None:
    entry = dataclasses.replace(
        VLA_MODELS[0],
        reference=dataclasses.replace(VLA_MODELS[0].reference, measured=None),  # type: ignore[arg-type]
    )
    assert registry_problems((entry,)) == [f"supported_without_reproduction:{entry.id}"]


@pytest.mark.parametrize(("rate", "n1", "n2"), [(0.0, 10, 10), (1.0, 10, 10), (0.5, 0, 10)])
def test_binomial_tolerance_refuses_degenerate_inputs(rate: float, n1: int, n2: int) -> None:
    with pytest.raises(ValueError):
        binomial_tolerance(rate, n1, n2)
