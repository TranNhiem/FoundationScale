"""The RL trainer's sampling parameters are DECLARED, not inherited (#370).

Before this, `run()` called generate(do_sample=True) with no temperature, no
top_p and no top_k, so the sampling distribution came from the checkpoint's
generation_config.json -- a file the training config never mentions. That is
not a missing convenience: group-relative objectives (GRPO, GSPO, Dr.GRPO,
DAPO) are DEFINED by within-group reward variance, and temperature is the
primary lever on it, so the trainer could not influence the one quantity its
objective family depends on.

Measured consequence before the fix: an end-to-end run on a slice selected FOR
having reward variance refused every step with "advantage is identically zero
over 4 of 4 used row(s) ... distinct rewards: [1.0]". The slice had been
selected at temperature 1.3 while the trainer sampled at the checkpoint
default, so the property did not transfer -- a variance slice is only valid
for the sampling configuration that produced it.
"""

from __future__ import annotations

import dataclasses

import pytest

from foundationscale.rl.trainer import RLTrainConfig, RLTrainer, TrainerRefusal


def test_sampling_parameters_are_declared_fields() -> None:
    # The point of #370: these must exist ON THE CONFIG. If they are absent,
    # the value silently comes from the model directory and no run can say what
    # it sampled with.
    names = {f.name for f in dataclasses.fields(RLTrainConfig)}
    assert {"temperature", "top_p", "top_k"} <= names


def test_sampling_defaults_are_explicit_not_none() -> None:
    # A None default would be handed straight to generate(), which restores the
    # checkpoint's value -- reintroducing the exact inheritance #370 removed.
    defaults = {
        f.name: f.default
        for f in dataclasses.fields(RLTrainConfig)
        if f.name in {"temperature", "top_p", "top_k"}
    }
    assert all(value is not None for value in defaults.values()), defaults
    assert defaults["temperature"] > 0.0


def test_greedy_decoding_with_a_group_is_refused_before_any_allocation() -> None:
    # The failing input. Greedy decoding makes every completion in a group
    # byte-identical, so their rewards are equal, so the group-relative
    # advantage is identically zero and no gradient exists. That is not a bad
    # hyperparameter -- it is a configuration in which training cannot happen,
    # and it must be refused loudly rather than abstaining once per step
    # forever. Exit 96 is the REFUSE code; the guard runs before the model is
    # loaded, so a bogus model path is never reached.
    config = RLTrainConfig(
        model="never-loaded",
        dataset="never-read",
        group_size=4,
        temperature=0.0,
    )
    with pytest.raises(SystemExit) as excinfo:
        RLTrainer(config=config).run()
    assert excinfo.value.code == 96


def test_group_size_one_is_refused_by_the_GROUP_guard_not_the_greedy_one() -> None:
    # Scope control, corrected by measurement. My first version of this leg
    # assumed group_size=1 was a legal way to opt out of group-relative
    # training; it is not -- an EARLIER guard refuses it outright, because a
    # group-relative objective needs at least 2 samples per group. Asserting
    # on the message keeps the two refusals distinguishable: if the greedy
    # guard ever swallowed this case, the leg would catch it.
    config = RLTrainConfig(
        model="never-loaded",
        dataset="never-read",
        group_size=1,
        temperature=0.0,
    )
    with pytest.raises(TrainerRefusal) as excinfo:
        RLTrainer(config=config)
    assert "at least 2 samples per group" in str(excinfo.value)
    assert "greedy decoding" not in str(excinfo.value)


def test_a_positive_temperature_does_not_trip_the_greedy_guard() -> None:
    # The other side of the failing input: a legal, sampling configuration must
    # get PAST the #370 guard. Without this the guard could refuse everything
    # and the RED leg above would still pass, which is the vacuous-gate shape.
    config = RLTrainConfig(
        model="never-loaded",
        dataset="never-read",
        group_size=4,
        temperature=1.0,
    )
    with pytest.raises(SystemExit) as excinfo:
        RLTrainer(config=config).run()
    # It exits on the bogus model path, NOT on sampling.
    assert "greedy decoding" not in str(excinfo.value)
