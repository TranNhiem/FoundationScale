from __future__ import annotations

import pytest

from foundationskills.skills.evaluation.policy import (
    PolicyError,
    TaskPolicy,
    compare,
    load_policy,
    parse_policy,
)


def _task(**kw) -> TaskPolicy:
    base = dict(name="t", metric="acc,none", k=2.0, abs_epsilon=0.01)
    base.update(kw)
    return TaskPolicy(**base)


def test_drop_exactly_at_threshold_is_not_a_breach():
    # threshold = max(0.01, 2*sqrt(0.03^2+0.04^2)) = 0.1
    c = compare(_task(), score=0.40, baseline=0.50, se_c=0.03, se_b=0.04)
    assert c.threshold == pytest.approx(0.1)
    assert c.drop == pytest.approx(0.1)
    assert c.breach is False
    assert c.band == "stderr"


def test_drop_just_past_threshold_breaches():
    assert compare(_task(), score=0.399, baseline=0.50, se_c=0.03, se_b=0.04).breach is True


def test_abs_epsilon_floor_wins_over_tiny_stderr():
    c = compare(_task(abs_epsilon=0.05), score=0.46, baseline=0.50, se_c=0.001, se_b=0.001)
    assert c.threshold == pytest.approx(0.05)
    assert c.band == "abs_epsilon"
    assert c.breach is False


def test_improvement_never_fails():
    assert compare(_task(), score=0.90, baseline=0.10, se_c=0.0, se_b=0.0).breach is False


def test_lower_is_better_flips_the_sign():
    t = _task(higher_is_better=False)
    assert compare(t, score=0.30, baseline=0.10, se_c=0.01, se_b=0.01).breach is True
    assert compare(t, score=0.05, baseline=0.10, se_c=0.01, se_b=0.01).breach is False


@pytest.mark.parametrize("se", [None, "N/A", float("nan")])
def test_missing_stderr_without_explicit_epsilon_is_a_policy_error(se):
    with pytest.raises(PolicyError, match="no measured stderr"):
        compare(_task(), score=0.4, baseline=0.5, se_c=se, se_b=0.01)


def test_missing_stderr_with_explicit_epsilon_uses_it():
    c = compare(_task(abs_epsilon=0.02, explicit_epsilon=True), score=0.47, baseline=0.50)
    assert c.breach is True and c.band == "abs_epsilon"


def test_parse_policy_marks_explicit_epsilon_and_inherits_defaults():
    p = parse_policy({
        "policy_version": 1,
        "default": {"k": 3, "abs_epsilon": 0.02},
        "tasks": {"a": {"metric": "acc,none"}, "b": {"metric": "em", "abs_epsilon": 0.05, "num_fewshot": 5}},
    })
    assert p.task("a").k == 3.0 and p.task("a").explicit_epsilon is False
    assert p.task("b").abs_epsilon == 0.05 and p.task("b").explicit_epsilon is True
    assert p.task("b").num_fewshot == 5
    assert len(p.fingerprint) == 64


def test_unknown_benchmark_names_the_missing_policy_entry():
    p = parse_policy({"policy_version": 1, "tasks": {"a": {"metric": "acc,none"}}})
    with pytest.raises(PolicyError, match="missing input: policy entry for benchmark 'zzz'"):
        p.task("zzz")


@pytest.mark.parametrize("data", [
    [],
    {"tasks": {"a": {"metric": "m"}}},
    {"policy_version": 1, "tasks": {}},
    {"policy_version": 1, "tasks": {"a": {}}},
    {"policy_version": 1, "default": {"k": -1}, "tasks": {"a": {"metric": "m"}}},
    {"policy_version": 1, "tasks": {"a": {"metric": "m", "num_fewshot": -2}}},
])
def test_invalid_policies_are_rejected(data):
    with pytest.raises(PolicyError, match="precondition failed"):
        parse_policy(data)


def test_load_policy_missing_file_names_it(tmp_path):
    with pytest.raises(PolicyError, match="missing input: eval policy file"):
        load_policy(tmp_path / "nope.yaml")


def test_shipped_default_policy_is_valid():
    from importlib import resources

    path = resources.files("foundationskills.skills.evaluation").joinpath("eval_policy.yaml")
    p = load_policy(str(path))
    assert p.version >= 1
    assert all(not t.judge for t in p.tasks.values())
