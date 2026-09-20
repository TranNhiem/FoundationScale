"""``RLTrainer`` must REFUSE an objective that declares no advantage estimator.

WHAT IS CLAIMED: ``_resolve_objective`` admits an algorithm only when its
objective declares the ``advantage_fn`` the loop goes on to read. An objective
without one is refused by name, before any model is loaded, rather than
surfacing as ``AttributeError`` inside ``_one_step`` after a full group has
already been generated.

WHAT IS NOT CLAIMED: nothing about loss values, gradients, or whether a
pairwise/rank-based objective COULD be driven by some other loop. The claim is
about where the boundary is announced, not about where it should ultimately be.

The MUST_FIRE arms are the preference and online families, and they are
enumerated from the registry rather than hard-coded, so an algorithm added
later is covered the day it is registered. A hard-coded pair would keep passing
while a new family silently reopened the hole.
"""

from __future__ import annotations

import pytest

from foundationscale.rl.registry import available_algorithm_names, lookup_algorithm
from foundationscale.rl.trainer import RLTrainConfig, RLTrainer, TrainerRefusal

__all__ = (
    "test_every_objective_bearing_algorithm_is_admitted_or_refused_by_name",
    "test_objective_without_an_advantage_estimator_is_refused_by_name",
    "test_the_group_relative_family_still_resolves",
)


def _objective_of(name: str) -> object | None:
    """The objective a registry entry exposes, or ``None`` if it exposes none."""
    return getattr(lookup_algorithm(name), "_objective", None)


def _split_by_estimator() -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Partition the objective-bearing registry entries on ``advantage_fn``."""
    with_fn: list[str] = []
    without_fn: list[str] = []
    for name in available_algorithm_names():
        objective = _objective_of(name)
        if objective is None:
            continue
        (with_fn if hasattr(objective, "advantage_fn") else without_fn).append(name)
    return tuple(with_fn), tuple(without_fn)


def _config(algorithm: str) -> RLTrainConfig:
    """A config that reaches ``_resolve_objective`` and nothing further."""
    return RLTrainConfig(model="unused/never-loaded", dataset="unused.jsonl", algorithm=algorithm)


def test_objective_without_an_advantage_estimator_is_refused_by_name() -> None:
    """MUST_FIRE: every estimator-less objective refuses, naming itself."""
    _, without_fn = _split_by_estimator()
    assert without_fn, (
        "this control is VACUOUS unless at least one registered objective lacks "
        "an advantage_fn; if the families were wired up, delete the control"
    )
    for name in without_fn:
        trainer = RLTrainer(_config(name))
        with pytest.raises(TrainerRefusal) as exc:
            trainer._resolve_objective()
        message = str(exc.value)
        assert repr(name) in message, f"{name}: refusal does not name the algorithm"
        assert type(_objective_of(name)).__name__ in message, (
            f"{name}: refusal does not name the objective class that lacks the estimator"
        )
        assert "advantage" in message


def test_the_group_relative_family_still_resolves() -> None:
    """MUST_PASS: the new guard must not refuse an objective that HAS one."""
    with_fn, _ = _split_by_estimator()
    assert with_fn, "the guard cannot be shown harmless if nothing resolves"
    for name in with_fn:
        objective = RLTrainer(_config(name))._resolve_objective()
        assert hasattr(objective, "advantage_fn")


def test_every_objective_bearing_algorithm_is_admitted_or_refused_by_name() -> None:
    """The two arms above must together cover every objective-bearing entry.

    Without this, a family that resolved by neither path -- raising something
    other than ``TrainerRefusal`` -- would be counted by no arm and the suite
    would report clean coverage over a hole.
    """
    with_fn, without_fn = _split_by_estimator()
    covered = set(with_fn) | set(without_fn)
    bearing = {name for name in available_algorithm_names() if _objective_of(name) is not None}
    assert covered == bearing
