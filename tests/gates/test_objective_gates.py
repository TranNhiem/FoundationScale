"""Adversarial tests for ``foundationscale.gates.objective_gates``.

Why this suite exists
---------------------
These four gates encode the lessons of three concrete incidents: 24 runs that split
12/12 between two objectives because an environment variable was read and never
recorded; a trust region listed and instantiated in seven of ten arms while its
importance ratio sat identically at 1.0; and 472 steps of ``grad_norm == 0.000``
beneath a healthy-looking ``reward/mean=0.794``. Each incident survived precisely
because some checker reported success without having checked the thing that was
wrong — or without having checked anything at all.

The tests here attack the gates the way the estate actually failed: they try to get
PASS out of an empty component list, out of an env-shadowed default, out of an
all-identical reward batch, out of hyperparameters nobody fingerprinted, and out of
malformed context objects. Every "X is rejected" test sits next to a positive
control proving the detector could have fired. If any of these paths ever yields a
non-blocking verdict again, this suite — not a resurrected training run — must be
where it surfaces.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import pytest

from foundationscale.gates.core import (
    REGISTRY,
    Control,
    ControlKind,
    GateRegistry,
    Lifecycle,
    Verdict,
    verify_controls,
)
from foundationscale.gates.objective_gates import (
    HyperparameterDriftGate,
    LossComponent,
    LossComponentCoverageGate,
    ObjectiveDeclaredGate,
    ObjectiveGateContext,
    RewardScaleSanityGate,
    RewardStats,
    ValueProvenance,
    fingerprint_hparams,
)

if TYPE_CHECKING:
    from foundationscale.gates.core import Gate


# ---------------------------------------------------------------------------
# Local context builders. Deliberately independent of the module's private
# ``_healthy_ctx``: tests that reuse the module's own fixtures can rot in
# lockstep with it. These mirror the same healthy geometry from first principles.
# ---------------------------------------------------------------------------


def _hparams() -> dict[str, Any]:
    return {
        "objective": "gspo",
        "clip_epsilon": 0.2,
        "kl_coef": 0.04,
        "entropy_coef": 0.001,
        "group_size": 8,
    }


def _provenance(**overrides: Any) -> ValueProvenance:
    kwargs: dict[str, Any] = {
        "name": "objective",
        "value": "gspo",
        "source": "cli",
        "recorded": True,
        "env_var": "FOUNDATIONSCALE_OBJECTIVE",
    }
    kwargs.update(overrides)
    return ValueProvenance(**kwargs)


def _components() -> tuple[LossComponent, ...]:
    return (
        LossComponent(name="policy_loss", weight=1.0, observed=True, contribution=0.83),
        LossComponent(name="kl_penalty", weight=0.04, observed=True, contribution=0.011),
        LossComponent(name="entropy_bonus", weight=0.001, observed=True, contribution=0.0021),
    )


def _healthy_stats() -> RewardStats:
    return RewardStats(n=256, mean=0.794, std=0.31, min=-1.25, max=2.4)


def _healthy_ctx(**overrides: Any) -> ObjectiveGateContext:
    hparams = _hparams()
    kwargs: dict[str, Any] = {
        "objective": _provenance(),
        "declared_components": ("policy_loss", "kl_penalty", "entropy_bonus"),
        "components": _components(),
        "uses_rewards": True,
        "reward_stats": _healthy_stats(),
        "reward_bounds": (-10.0, 10.0),
        "expected_sample_count": 256,
        "step0_fingerprint": fingerprint_hparams(hparams),
        "step0_hparams": hparams,
        "current_hparams": dict(hparams),
        "origin": "<test>",
    }
    kwargs.update(overrides)
    return ObjectiveGateContext(**kwargs)


def _control_by_name(gate: Gate, name: str) -> Control:
    return next(c for c in gate.controls() if c.name == name)


@pytest.fixture
def declared_gate() -> ObjectiveDeclaredGate:
    return ObjectiveDeclaredGate()


@pytest.fixture
def component_gate() -> LossComponentCoverageGate:
    return LossComponentCoverageGate()


@pytest.fixture
def reward_gate() -> RewardScaleSanityGate:
    return RewardScaleSanityGate()


@pytest.fixture
def drift_gate() -> HyperparameterDriftGate:
    return HyperparameterDriftGate()


class TestFingerprintHparams:
    """The fingerprint is the verdict the drift gate stands on. Two mappings that
    mean the same thing must fingerprint identically, and two that differ must
    not — a fingerprint that cannot distinguish drift makes the gate decorative."""

    def test_key_insertion_order_does_not_change_the_fingerprint(self) -> None:
        forward = {"a": 1, "b": 2.5, "c": "x"}
        backward = {"c": "x", "b": 2.5, "a": 1}
        assert fingerprint_hparams(forward) == fingerprint_hparams(backward)

    def test_any_value_difference_changes_the_fingerprint(self) -> None:
        # Positive control for the order-invariance test above: prove the
        # fingerprint is not constant by showing a one-value change moves it.
        base = {"a": 1, "b": 2.5}
        assert fingerprint_hparams(base) != fingerprint_hparams({"a": 1, "b": 2.6})

    def test_fingerprint_is_a_stable_sha256_hex_digest(self) -> None:
        fp = fingerprint_hparams(_hparams())
        assert fingerprint_hparams(_hparams()) == fp
        assert len(fp) == 64
        assert all(c in "0123456789abcdef" for c in fp)

    def test_non_json_values_fingerprint_via_repr_instead_of_raising(self) -> None:
        # A hyperparameter value that json cannot serialise must not take the
        # gate down with it — the function is documented to total-ise. If it
        # raised, the drift gate would return ERROR and block every healthy run
        # whose config carries one exotic value, which is how gates get bypassed.

        class _Odd:
            def __init__(self, label: str) -> None:
                self._label = label

            def __repr__(self) -> str:
                return f"<odd {self._label}>"

        fp_one = fingerprint_hparams({"coef": _Odd("alpha")})
        assert fingerprint_hparams({"coef": _Odd("alpha")}) == fp_one
        assert fp_one != fingerprint_hparams({"coef": _Odd("beta")})


class TestObjectiveDeclaredGate:
    """An env-shadowed default is the 24-run incident verbatim. A gate that
    passes it — at any severity below FAIL — reproduces the incident on demand."""

    def test_healthy_recorded_objective_passes_with_full_coverage(
        self, declared_gate: ObjectiveDeclaredGate
    ) -> None:
        result = declared_gate.run(_healthy_ctx())
        assert result.verdict is Verdict.PASS
        assert not result.blocking
        assert result.coverage.checked == 1
        assert result.coverage.expected == 1
        assert result.coverage.unit == "objective selections"
        assert "'gspo'" in result.detail
        assert "'cli'" in result.detail

    def test_no_objective_record_at_all_is_vacuous_never_pass(
        self, declared_gate: ObjectiveDeclaredGate
    ) -> None:
        # There is no objective claim in the artefacts. A green tick here tells
        # the operator the objective is recoverable from the record; it is not.
        result = declared_gate.run(_healthy_ctx(objective=None))
        assert result.verdict is Verdict.VACUOUS
        assert result.verdict is not Verdict.PASS
        assert result.blocking
        assert "unrecoverable" in result.detail
        assert result.evidence["origin"] == "<test>"

    @pytest.mark.parametrize("source", ["cli", "config", "default", "env"])
    def test_recorded_objective_from_any_known_source_passes(
        self, declared_gate: ObjectiveDeclaredGate, source: str
    ) -> None:
        # This is the positive-control battery: it proves the failures below
        # are caused by the specific defect each injects, not by a gate that
        # has learned to block every provenance shape. Note that source="env"
        # itself is legitimate — what the incident killed was env *unrecorded*
        # or *disagreeing* with the effective value.
        result = declared_gate.run(_healthy_ctx(objective=_provenance(source=source)))
        assert result.verdict is Verdict.PASS

    @pytest.mark.parametrize("source", ["cli", "env"])
    def test_unrecorded_effective_objective_fails_from_any_source(
        self, declared_gate: ObjectiveDeclaredGate, source: str
    ) -> None:
        # The rule is "recorded", not "from a nice source". A CLI flag nobody
        # wrote down is just as unrecoverable as an env switch nobody wrote down.
        result = declared_gate.run(
            _healthy_ctx(objective=_provenance(source=source, recorded=False)),
        )
        assert result.verdict is Verdict.FAIL
        assert result.blocking
        assert "never recorded" in result.detail
        assert result.evidence["objective"] == "gspo"
        assert result.evidence["source"] == source
        # Coverage of 1/1 must be recorded even on failure: something was inspected.
        assert result.coverage.checked == 1

    def test_unknown_provenance_source_fails(self, declared_gate: ObjectiveDeclaredGate) -> None:
        result = declared_gate.run(
            _healthy_ctx(objective=_provenance(source="makefile")),
        )
        assert result.verdict is Verdict.FAIL
        assert "'makefile'" in result.detail
        assert result.evidence["source"] == "makefile"

    def test_env_shadowed_default_fails_with_the_incident_signature(
        self, declared_gate: ObjectiveDeclaredGate
    ) -> None:
        # Found the objective said "grpo" from defaults while the environment
        # asked for "gspo" and lost silently. This is byte-for-byte the shape
        # that split 24 runs 12/12; the gate must name all three participants
        # (effective value, env var, env value) or a 3am operator cannot act.
        prov = _provenance(
            value="grpo",
            source="default",
            shadowed_by_env=True,
            env_value="gspo",
        )
        result = declared_gate.run(_healthy_ctx(objective=prov))
        assert result.verdict is Verdict.FAIL
        assert "FOUNDATIONSCALE_OBJECTIVE" in result.detail
        assert "'gspo'" in result.detail
        assert "'grpo'" in result.detail
        assert result.evidence["effective"] == "grpo"
        assert result.evidence["env_var"] == "FOUNDATIONSCALE_OBJECTIVE"
        assert result.evidence["env_value"] == "gspo"

    def test_recorded_default_without_env_disagreement_passes(
        self, declared_gate: ObjectiveDeclaredGate
    ) -> None:
        # Positive control one line away from the incident: same source, same
        # env var present in the record, but no shadowing — must pass, or the
        # env-shadow rule is secretly "defaults are bad", which it is not.
        prov = _provenance(value="grpo", source="default", shadowed_by_env=False)
        result = declared_gate.run(_healthy_ctx(objective=prov))
        assert result.verdict is Verdict.PASS


class TestLossComponentCoverageGate:
    """Two incident shapes: the component that is listed, instantiated, and inert;
    and the empty component list that ``all([])`` waves through. Both must block."""

    def test_healthy_components_pass_with_full_coverage(
        self, component_gate: LossComponentCoverageGate
    ) -> None:
        result = component_gate.run(_healthy_ctx())
        assert result.verdict is Verdict.PASS
        assert result.coverage.checked == 3
        assert result.coverage.expected == 3
        assert "all 3 declared components" in result.detail

    def test_empty_declared_component_list_is_vacuous_never_pass(
        self, component_gate: LossComponentCoverageGate
    ) -> None:
        # The ``gspo._zero_metrics`` twin: an objective that supervises nothing
        # must not emerge as "no missing components". VACUOUS, blocking.
        result = component_gate.run(_healthy_ctx(declared_components=(), components=()))
        assert result.verdict is Verdict.VACUOUS
        assert result.verdict is not Verdict.PASS
        assert result.blocking

    def test_empty_declaration_with_observed_components_still_blocks(
        self, component_gate: LossComponentCoverageGate
    ) -> None:
        # Components are measurably contributing yet the objective claims to
        # optimise nothing. The gate must not call this clean; its evidence
        # must at least surface the observed names so the contradiction is
        # audible rather than silently averaged away.
        result = component_gate.run(_healthy_ctx(declared_components=()))
        assert result.blocking
        assert result.verdict is Verdict.VACUOUS
        assert "policy_loss" in result.evidence["observed"]

    def test_declared_component_absent_from_the_step_fails(
        self, component_gate: LossComponentCoverageGate
    ) -> None:
        components = tuple(c for c in _components() if c.name != "kl_penalty")
        result = component_gate.run(_healthy_ctx(components=components))
        assert result.verdict is Verdict.FAIL
        assert "kl_penalty" in result.detail
        assert result.evidence["declared"] == ["policy_loss", "kl_penalty", "entropy_bonus"]
        assert any("kl_penalty" in p for p in result.evidence["problems"])
        # Two of three declared components were actually verified; the FAIL
        # verdict must dominate the coverage shortfall, not replace it.
        assert result.coverage.checked == 2
        assert result.coverage.expected == 3

    def test_declared_component_present_but_unobserved_counts_as_missing(
        self, component_gate: LossComponentCoverageGate
    ) -> None:
        components = tuple(
            LossComponent(c.name, c.weight, observed=False, contribution=c.contribution)
            if c.name == "kl_penalty"
            else c
            for c in _components()
        )
        result = component_gate.run(_healthy_ctx(components=components))
        assert result.verdict is Verdict.FAIL
        assert "kl_penalty" in result.detail

    def test_zero_weight_component_fails_as_listed_but_inert(
        self, component_gate: LossComponentCoverageGate
    ) -> None:
        # The trust-region-off-by-default signature: declared, observed,
        # weighted 0.0, optimising nothing by construction.
        components = tuple(
            LossComponent(c.name, 0.0, c.observed, contribution=0.0)
            if c.name == "kl_penalty"
            else c
            for c in _components()
        )
        result = component_gate.run(_healthy_ctx(components=components))
        assert result.verdict is Verdict.FAIL
        assert "weight 0.0" in result.detail
        assert "kl_penalty" in result.detail

    def test_zero_weight_is_reported_once_not_also_as_inert(
        self, component_gate: LossComponentCoverageGate
    ) -> None:
        # A component with weight 0.0 and contribution 0.0 has one defect with
        # two descriptions; double-counting it inflates the problem list and
        # dilutes the other reports around it.
        components = tuple(
            LossComponent(c.name, 0.0, c.observed, contribution=0.0)
            if c.name == "kl_penalty"
            else c
            for c in _components()
        )
        result = component_gate.run(_healthy_ctx(components=components))
        assert result.verdict is Verdict.FAIL
        assert len(result.evidence["problems"]) == 1

    def test_measured_zero_contribution_with_nonzero_weight_fails(
        self, component_gate: LossComponentCoverageGate
    ) -> None:
        # The ratio-identically-1.0 shape: weight is real, the measured
        # contribution is exactly zero. Present, weighted, inert.
        components = tuple(
            LossComponent(c.name, c.weight, c.observed, contribution=0.0)
            if c.name == "kl_penalty"
            else c
            for c in _components()
        )
        result = component_gate.run(_healthy_ctx(components=components))
        assert result.verdict is Verdict.FAIL
        assert "kl_penalty" in result.detail
        assert any("exactly 0.0" in p for p in result.evidence["problems"])

    def test_unmeasured_contribution_makes_no_claim_and_passes(
        self, component_gate: LossComponentCoverageGate
    ) -> None:
        # Positive control for the inert test above: ``contribution=None`` is
        # documented as "no contributing claim either way". If this ever fails,
        # the gate has started inventing defects the measurement never made.
        components = tuple(
            LossComponent(c.name, c.weight, c.observed, contribution=None)
            if c.name == "kl_penalty"
            else c
            for c in _components()
        )
        result = component_gate.run(_healthy_ctx(components=components))
        assert result.verdict is Verdict.PASS

    def test_observed_component_never_declared_fails(
        self, component_gate: LossComponentCoverageGate
    ) -> None:
        components = (
            *_components(),
            LossComponent(name="shadow_term", weight=0.5, observed=True, contribution=0.3),
        )
        result = component_gate.run(_healthy_ctx(components=components))
        assert result.verdict is Verdict.FAIL
        assert "shadow_term" in result.detail
        assert "exceeds its own record" in result.detail

    def test_duplicate_declared_names_are_counted_once(
        self, component_gate: LossComponentCoverageGate
    ) -> None:
        # A resolver that repeats a component name must not inflate coverage
        # into "2 declared, 2 checked" when only one real component exists.
        result = component_gate.run(
            _healthy_ctx(
                declared_components=("kl_penalty", "kl_penalty"),
                components=(
                    LossComponent(
                        name="kl_penalty", weight=0.04, observed=True, contribution=0.011
                    ),
                ),
            ),
        )
        assert result.verdict is Verdict.PASS
        assert result.coverage.expected == 1
        assert "all 1 declared components" in result.detail

    # Was a strict xfail: the gate compared weight and contribution to 0.0 with ==, and
    # NaN is equal to nothing, so a component whose weight had gone NaN took the false
    # branch of every check and fell through to PASS. Fixed with math.isfinite ahead of
    # the zero checks. The same audit found a second site — a NaN reward bound silently
    # disabling the bounds check — which is now covered too.
    @pytest.mark.parametrize("defect", ["nan_weight", "nan_contribution"])
    def test_non_finite_component_measurements_are_rejected(
        self, component_gate: LossComponentCoverageGate, defect: str
    ) -> None:
        # Positive control for this detector lives two tests up: measured
        # contribution 0.0 with non-zero weight already FAILs, so the gate's
        # arithmetic path demonstrably fires — it just goes blind at NaN.
        if defect == "nan_weight":
            broken = LossComponent(
                name="kl_penalty", weight=math.nan, observed=True, contribution=0.011
            )
        else:
            broken = LossComponent(
                name="kl_penalty", weight=0.04, observed=True, contribution=math.nan
            )
        components = tuple(broken if c.name == "kl_penalty" else c for c in _components())
        result = component_gate.run(_healthy_ctx(components=components))
        assert result.verdict is Verdict.FAIL

    def test_malformed_component_entry_fails_closed_as_error(
        self, component_gate: LossComponentCoverageGate
    ) -> None:
        # A component record that isn't a LossComponent must become ERROR
        # (blocking), not an exception escaping into launcher code where the
        # audited estate would have logged it and continued training.

        class _NotAComponent:
            name = "kl_penalty"

        ctx = _healthy_ctx(
            declared_components=("kl_penalty",),
            components=(_NotAComponent(),),
        )
        result = component_gate.run(ctx)
        assert result.verdict is Verdict.ERROR
        assert result.blocking

    def test_zero_measured_contributions_is_vacuous_and_names_zero(
        self, component_gate: LossComponentCoverageGate
    ) -> None:
        # F1: declared, observed, non-zero-weighted — and not one contribution
        # measured. Presence/weight examination is real, but the relation this
        # gate exists to check ran over zero units, so the pass grade is
        # all([]) with a covered numerator.
        # FAILS TODAY: current tree returns PASS with checked=3 and a detail
        # asserting "present, weighted, contributing".
        components = tuple(
            LossComponent(c.name, c.weight, c.observed, contribution=None) for c in _components()
        )
        result = component_gate.run(_healthy_ctx(components=components))
        assert result.verdict is Verdict.VACUOUS  # today: PASS
        assert result.verdict is not Verdict.PASS
        assert result.blocking
        assert result.coverage.checked == 0  # today: 3
        assert result.coverage.unit == "contribution measurements"
        assert "0 of 3" in result.detail  # the 0 must be named
        assert "contributing" not in result.detail
        assert result.evidence["contributions_measured"] == 0
        assert result.evidence["contributions_unmeasured"] == [
            "policy_loss",
            "kl_penalty",
            "entropy_bonus",
        ]

    def test_partially_measured_contributions_pass_as_a_declared_sample(
        self, component_gate: LossComponentCoverageGate
    ) -> None:
        # F1's sibling arm, constrained by the pinned None contract (see
        # test_unmeasured_contribution_makes_no_claim_and_passes — verdict PASS
        # is preserved): the pass must carry the measured/total denominator,
        # count only fully-examined components, and name the abstainer. This is
        # doctrine 2 applied to the numerator, not a minted defect.
        # FAILS TODAY: checked==3, sampled==False, no denominator in the
        # detail, and the evidence keys do not exist.
        components = tuple(
            LossComponent(c.name, c.weight, c.observed, contribution=None)
            if c.name == "kl_penalty"
            else c
            for c in _components()
        )
        result = component_gate.run(_healthy_ctx(components=components))
        assert result.verdict is Verdict.PASS
        assert result.coverage.checked == 2  # today: 3 — credit for an unexamined leg
        assert result.coverage.expected == 3
        assert result.coverage.sampled is True
        assert "kl_penalty" in result.coverage.sample_reason
        assert "2 of 3" in result.detail
        assert "present, weighted, contributing" not in result.detail
        assert result.evidence["contributions_measured"] == 2
        assert result.evidence["contributions_unmeasured"] == ["kl_penalty"]


class TestRewardScaleSanityGate:
    """The gate exists because of a batch where every reward was 0.794, every
    advantage was 0, the gradient vanished, and every average stayed healthy.
    Means are exactly what the degenerate case preserves; this gate must not."""

    def test_healthy_rewards_pass_with_sample_coverage(
        self, reward_gate: RewardScaleSanityGate
    ) -> None:
        result = reward_gate.run(_healthy_ctx())
        assert result.verdict is Verdict.PASS
        assert result.coverage.checked == 256
        assert result.coverage.expected == 256
        assert result.coverage.unit == "reward samples"

    def test_no_reward_objective_and_no_stats_is_an_explained_skip(
        self, reward_gate: RewardScaleSanityGate
    ) -> None:
        result = reward_gate.run(
            _healthy_ctx(uses_rewards=False, reward_stats=None, expected_sample_count=None),
        )
        assert result.verdict is Verdict.SKIP
        assert not result.blocking
        assert result.detail.strip() != ""

    def test_reward_objective_with_no_samples_is_vacuous_never_pass(
        self, reward_gate: RewardScaleSanityGate
    ) -> None:
        # Zero samples examined over a reward-bearing objective is the
        # "the check ran green because the checker read nothing" shape.
        result = reward_gate.run(
            _healthy_ctx(reward_stats=None, expected_sample_count=None),
        )
        assert result.verdict is Verdict.VACUOUS
        assert result.blocking
        assert "no reward samples" in result.detail

    def test_stats_object_over_zero_samples_is_still_vacuous(
        self, reward_gate: RewardScaleSanityGate
    ) -> None:
        # The statistics object existing does not mean anything was measured.
        result = reward_gate.run(
            _healthy_ctx(
                reward_stats=RewardStats(n=0, mean=0.0, std=0.0, min=0.0, max=0.0),
                expected_sample_count=0,
            ),
        )
        assert result.verdict is Verdict.VACUOUS
        assert result.blocking

    def test_all_identical_rewards_fail_despite_a_healthy_mean(
        self, reward_gate: RewardScaleSanityGate
    ) -> None:
        # The incident batch, restated: n=472, mean 0.794, spread exactly 0.
        # The detail must carry the numbers or nobody can find the batch.
        result = reward_gate.run(
            _healthy_ctx(
                reward_stats=RewardStats(n=472, mean=0.794, std=0.0, min=0.794, max=0.794),
                expected_sample_count=472,
            ),
        )
        assert result.verdict is Verdict.FAIL
        assert result.blocking
        assert "472" in result.detail
        assert "0.794" in result.detail
        assert result.evidence["stats"]["std"] == 0.0

    def test_single_sample_with_zero_spread_passes(
        self, reward_gate: RewardScaleSanityGate
    ) -> None:
        # Boundary AT the degeneracy rule: one sample cannot exhibit variance.
        # If this fails, the gate forbids every single-sample sanity check,
        # which is not the incident shape and would get the gate muted.
        result = reward_gate.run(
            _healthy_ctx(
                reward_stats=RewardStats(n=1, mean=0.794, std=0.0, min=0.794, max=0.794),
                expected_sample_count=1,
            ),
        )
        assert result.verdict is Verdict.PASS

    def test_two_identical_samples_fail(self, reward_gate: RewardScaleSanityGate) -> None:
        # Boundary just OUTSIDE: the smallest batch on which the claim
        # "every advantage is identically zero" can be made. Must fail.
        result = reward_gate.run(
            _healthy_ctx(
                reward_stats=RewardStats(n=2, mean=0.5, std=0.0, min=0.5, max=0.5),
                expected_sample_count=2,
            ),
        )
        assert result.verdict is Verdict.FAIL

    @pytest.mark.parametrize("stat_field", ["mean", "std", "min", "max"])
    @pytest.mark.parametrize("bad_value", [math.nan, math.inf, -math.inf])
    def test_non_finite_reward_statistics_fail(
        self, reward_gate: RewardScaleSanityGate, stat_field: str, bad_value: float
    ) -> None:
        # Every statistic field, every non-finite flavour. A NaN mean compared
        # against bounds is silently "within bounds" under IEEE ordering — the
        # finiteness check is the only thing standing there, so test it per
        # field, not once.
        stats_kwargs: dict[str, Any] = {
            "n": 32,
            "mean": 0.5,
            "std": 0.2,
            "min": -1.0,
            "max": 1.5,
        }
        stats_kwargs[stat_field] = bad_value
        result = reward_gate.run(
            _healthy_ctx(
                reward_stats=RewardStats(**stats_kwargs),
                expected_sample_count=32,
            ),
        )
        assert result.verdict is Verdict.FAIL
        assert "non-finite" in result.detail

    @pytest.mark.parametrize(
        ("min_v", "max_v"),
        [(-10.0, 9.0), (-9.0, 10.0)],
        ids=["at-lower-bound", "at-upper-bound"],
    )
    def test_rewards_exactly_at_declared_bounds_pass(
        self, reward_gate: RewardScaleSanityGate, min_v: float, max_v: float
    ) -> None:
        # Boundary AT: touching the declared bound is legal. An off-by-one in
        # the comparison operator would turn these into FAILs and train users
        # to widen the bounds until the gate says nothing.
        result = reward_gate.run(
            _healthy_ctx(
                reward_stats=RewardStats(n=32, mean=0.0, std=3.0, min=min_v, max=max_v),
                expected_sample_count=None,
            ),
        )
        assert result.verdict is Verdict.PASS

    @pytest.mark.parametrize(
        ("min_v", "max_v"),
        [(-10.000_001, 0.0), (0.0, 10.000_001)],
        ids=["just-below-lower-bound", "just-above-upper-bound"],
    )
    def test_rewards_just_outside_declared_bounds_fail_with_the_numbers(
        self, reward_gate: RewardScaleSanityGate, min_v: float, max_v: float
    ) -> None:
        # Boundary just OUTSIDE: the smallest representable breach must fail,
        # and the failure must quote both the observed span and the declared
        # bounds — "out of bounds" without numbers is not actionable.
        result = reward_gate.run(
            _healthy_ctx(
                reward_stats=RewardStats(n=32, mean=0.0, std=3.0, min=min_v, max=max_v),
                expected_sample_count=None,
            ),
        )
        assert result.verdict is Verdict.FAIL
        assert f"[{min_v}, {max_v}]" in result.detail
        assert "[-10.0, 10.0]" in result.detail

    def test_degenerate_and_out_of_bounds_reports_both_defects(
        self, reward_gate: RewardScaleSanityGate
    ) -> None:
        # Two independent problems in one batch must both be in the evidence,
        # even though the headline detail can only quote the first.
        result = reward_gate.run(
            _healthy_ctx(
                reward_stats=RewardStats(n=8, mean=50.0, std=0.0, min=50.0, max=50.0),
                expected_sample_count=None,
            ),
        )
        assert result.verdict is Verdict.FAIL
        assert "(+1 more)" in result.detail
        assert len(result.evidence["problems"]) == 2

    def test_fewer_samples_than_promised_is_undercovered(
        self, reward_gate: RewardScaleSanityGate
    ) -> None:
        # 200 of 256 samples bounded is not the claim "256 samples bounded".
        result = reward_gate.run(
            _healthy_ctx(
                reward_stats=RewardStats(n=200, mean=0.5, std=0.3, min=-1.0, max=2.0),
                expected_sample_count=256,
            ),
        )
        assert result.verdict is Verdict.UNDERCOVERED
        assert result.blocking
        assert "200 of 256 reward samples" in result.detail

    def test_genuine_defect_stays_fail_even_under_short_coverage(
        self, reward_gate: RewardScaleSanityGate
    ) -> None:
        # A found defect must not be re-framed as UNDERCOVERED just because
        # the batch was also short: FAIL is the actionable verdict.
        result = reward_gate.run(
            _healthy_ctx(
                reward_stats=RewardStats(n=100, mean=0.5, std=0.0, min=0.5, max=0.5),
                expected_sample_count=256,
            ),
        )
        assert result.verdict is Verdict.FAIL

    def test_undeclared_bounds_still_check_degeneracy(
        self, reward_gate: RewardScaleSanityGate
    ) -> None:
        result = reward_gate.run(
            _healthy_ctx(
                reward_stats=RewardStats(n=64, mean=1.0, std=0.0, min=1.0, max=1.0),
                reward_bounds=None,
                expected_sample_count=None,
            ),
        )
        assert result.verdict is Verdict.FAIL

    def test_undeclared_bounds_with_genuine_variance_pass(
        self, reward_gate: RewardScaleSanityGate
    ) -> None:
        result = reward_gate.run(
            _healthy_ctx(
                reward_stats=RewardStats(n=64, mean=0.5, std=0.4, min=-0.5, max=1.5),
                reward_bounds=None,
                expected_sample_count=None,
            ),
        )
        assert result.verdict is Verdict.PASS
        assert "no declared bounds" in result.detail

    def test_malformed_statistics_fail_closed_as_error(
        self, reward_gate: RewardScaleSanityGate
    ) -> None:
        # A string where a float belongs must become a blocking ERROR via the
        # framework's fail-closed path — never an uncaught TypeError in a
        # launcher, and CERTAINLY never a comparison that silently "passes".
        stats = RewardStats(
            n=4,
            mean="high",  # type: ignore[arg-type]
            std=0.1,
            min=0.0,
            max=1.0,
        )
        result = reward_gate.run(_healthy_ctx(reward_stats=stats, expected_sample_count=None))
        assert result.verdict is Verdict.ERROR
        assert result.blocking
        assert "TypeError" in result.detail

    def test_supplied_stats_for_a_reward_less_objective_are_examined_and_noted(
        self, reward_gate: RewardScaleSanityGate
    ) -> None:
        # F3 re-verified: with stats populated the skip branch is unreachable —
        # the samples ARE examined (256-strong coverage proves it). The residue
        # is that the verdict said nothing about the declaration contradiction.
        # The fix surfaces it as data; the verdict still judges the statistics,
        # because minting a failure nobody observed is doctrine 5's symmetric sin.
        # FAILS TODAY on the note assertion only (the note text is absent);
        # every other assertion holds on both trees and pins "no false failure".
        result = reward_gate.run(_healthy_ctx(uses_rewards=False))
        assert result.verdict is Verdict.PASS
        assert not result.blocking
        assert result.coverage.checked == 256
        assert "nothing to bound" not in result.detail
        assert "declares no reward term" in result.detail  # today: absent


class TestRewardStatsInternalCoherence:
    """W-obj-8: the gate adjudicates a caller-supplied aggregate and never sees
    raw samples, so a RewardStats no sample set can produce is fiction — and
    before the fix the gate passed exactly such a summary (std=0.0 over
    min=-1.0, max=2.0) while its own detail line asserted ``std > 0``."""

    def test_zero_std_over_a_nonzero_range_fails_with_the_ledger_reproduction(
        self, reward_gate: RewardScaleSanityGate
    ) -> None:
        # The ledger reproduction, verbatim: 256 samples, mean 0.5, std exactly
        # 0.0, range [-1.0, 2.0]. Zero spread forces min == max; one of the
        # numbers is a lie, so this can never be a passing summary again.
        result = reward_gate.run(
            _healthy_ctx(
                reward_stats=RewardStats(n=256, mean=0.5, std=0.0, min=-1.0, max=2.0),
            ),
        )
        assert result.verdict is Verdict.FAIL  # flips: this batch PASSes today
        assert result.blocking
        assert any("std is exactly 0.0" in p for p in result.evidence["problems"])
        assert result.evidence["stats"]["std"] == 0.0

    @pytest.mark.parametrize(
        ("stats", "marker"),
        [
            (RewardStats(n=64, mean=0.0, std=-0.25, min=-1.0, max=2.0), "negative"),
            (RewardStats(n=64, mean=0.0, std=0.3, min=2.0, max=-1.0), "inverted"),
            (
                RewardStats(n=128, mean=5.0, std=0.4, min=-1.0, max=2.0),
                "lies outside the reported range",
            ),
            (
                RewardStats(n=128, mean=-5.0, std=0.4, min=-1.0, max=2.0),
                "lies outside the reported range",
            ),
            (RewardStats(n=1, mean=0.0, std=0.0, min=-1.0, max=2.0), "cannot span a range"),
            (RewardStats(n=-3, mean=0.5, std=0.3, min=-1.0, max=2.0), "negative number"),
            (RewardStats(n=0, mean=0.0, std=0.0, min=0.0, max=1.0), "cannot span a range"),
        ],
        ids=[
            "negative-std",
            "inverted-range",
            "mean-above-max",
            "mean-below-min",
            "one-sample-with-range",
            "negative-count",
            "zero-samples-with-range",
        ],
    )
    def test_internally_impossible_summaries_fail_and_say_why(
        self, reward_gate: RewardScaleSanityGate, stats: RewardStats, marker: str
    ) -> None:
        # One class per row of the impossible-summary taxonomy. The marker is
        # asserted against evidence so a FAIL from the wrong branch (e.g. the
        # bounds check) cannot masquerade as coherence coverage.
        result = reward_gate.run(
            _healthy_ctx(reward_stats=stats, expected_sample_count=None),
        )
        # Flips on every row: each summary currently walks to the ok path
        # (PASS, or VACUOUS for the n=0 row — neither is FAIL). The negative-count
        # row additionally pins the hoist above Coverage: left in the cascade it
        # died as "ValueError: coverage cannot be negative", an ERROR traceback
        # rather than the stated reason doctrine 5 asks for.
        assert result.verdict is Verdict.FAIL
        assert result.blocking
        assert any(marker in p for p in result.evidence["problems"])

    def test_std_ceiling_admits_the_maximum_and_rejects_anything_above(
        self, reward_gate: RewardScaleSanityGate
    ) -> None:
        # 32 samples at -1.0 and 32 at 2.0: mean 0.5 with Bessel-corrected std
        # exactly 1.5*sqrt(64/63). A summary AT the ceiling is physically real
        # and must pass — otherwise the gate false-fires on honest extremes and
        # gets muted, which is how limits checks die in this estate.
        ceiling_std = 1.5 * math.sqrt(64 / 63)
        at = RewardStats(n=64, mean=0.5, std=ceiling_std, min=-1.0, max=2.0)
        above = RewardStats(n=64, mean=0.5, std=ceiling_std * 1.05, min=-1.0, max=2.0)
        result_at = reward_gate.run(
            _healthy_ctx(reward_stats=at, expected_sample_count=64),
        )
        assert result_at.verdict is Verdict.PASS
        result_above = reward_gate.run(
            _healthy_ctx(reward_stats=above, expected_sample_count=None),
        )
        # The assertion that flips: nothing on the current tree compares std
        # to the range, so the impossible batch passes.
        assert result_above.verdict is Verdict.FAIL
        assert "ceiling" in result_above.detail

    def test_single_sample_pass_claims_no_spread_positivity(
        self, reward_gate: RewardScaleSanityGate
    ) -> None:
        # Boundary AT the n>1 positivity guarantee: n=1 passes, but the verdict
        # text must not assert "std > 0" the way the pre-fix literal did — the
        # gate only earns that claim from the coherence cascade at n>1.
        result = reward_gate.run(
            _healthy_ctx(
                reward_stats=RewardStats(n=1, mean=0.794, std=0.0, min=0.794, max=0.794),
                expected_sample_count=1,
            ),
        )
        assert result.verdict is Verdict.PASS
        # Flips: the current detail ends in the literal "std=0 > 0".
        assert "> 0" not in result.detail

    _COHERENCE_MUST_FIRE = {
        "zero-std-with-nonzero-range",
        "negative-std",
        "inverted-range",
        "mean-outside-range",
        "std-above-analytic-ceiling",
        "single-sample-with-range",
        "negative-sample-count",
    }

    def test_coherence_controls_exist_and_each_blocks_with_fail(self) -> None:
        gate = RewardScaleSanityGate()
        by_name = {control.name: control for control in gate.controls()}
        # Flips: none of the coherence controls exist on the current tree.
        assert set(by_name) >= self._COHERENCE_MUST_FIRE
        for name in sorted(self._COHERENCE_MUST_FIRE):
            control = by_name[name]
            assert control.kind is ControlKind.MUST_FIRE
            result = gate.run(control.make_ctx())
            # verify_controls only demands blocking; pin FAIL so a severity
            # downgrade to VACUOUS — or an ERROR traceback, which the
            # negative-count control produced before the Coverage hoist — cannot
            # ride through as "still a control".
            assert result.verdict is Verdict.FAIL, name
            assert result.blocking, name


class TestHyperparameterDriftGate:
    """The gate's whole claim: what is being optimised now is what the run
    declared at step 0 — compared against the recorded fingerprint, never a
    re-read of a file the environment may have moved underneath."""

    def test_live_hparams_matching_the_step0_record_pass(
        self, drift_gate: HyperparameterDriftGate
    ) -> None:
        result = drift_gate.run(_healthy_ctx())
        assert result.verdict is Verdict.PASS
        assert result.coverage.checked == 5
        assert result.coverage.expected == 5
        assert result.evidence["fingerprint"] == fingerprint_hparams(_hparams())

    def test_key_insertion_order_in_the_live_view_is_not_drift(
        self, drift_gate: HyperparameterDriftGate
    ) -> None:
        # Config round-trips through YAML/JSON routinely reorder keys. If that
        # counted as drift, every run would block at SAVE and the gate would
        # be disabled within a week — detector rot by false alarm.
        reordered = dict(reversed(list(_hparams().items())))
        result = drift_gate.run(_healthy_ctx(current_hparams=reordered))
        assert result.verdict is Verdict.PASS

    def test_kl_coef_sliding_to_zero_fails_and_names_the_key(
        self, drift_gate: HyperparameterDriftGate
    ) -> None:
        # The incident: the trust region evaporated between saves and nothing
        # compared against the step-0 record. The evidence must name the key
        # and carry both fingerprints.
        current = _hparams()
        current["kl_coef"] = 0.0
        ctx = _healthy_ctx(current_hparams=current)
        result = drift_gate.run(ctx)
        assert result.verdict is Verdict.FAIL
        assert result.blocking
        assert "kl_coef" in result.detail
        assert result.evidence["changed_keys"] == ["kl_coef"]
        assert result.evidence["recorded_fingerprint"] == ctx.step0_fingerprint
        assert result.evidence["current_fingerprint"] == fingerprint_hparams(current)
        assert result.evidence["recorded_fingerprint"] != result.evidence["current_fingerprint"]

    def test_newly_appearing_hyperparameter_fails_and_is_named(
        self, drift_gate: HyperparameterDriftGate
    ) -> None:
        current = _hparams()
        current["mystery_coef"] = 1.0
        result = drift_gate.run(_healthy_ctx(current_hparams=current))
        assert result.verdict is Verdict.FAIL
        assert result.evidence["changed_keys"] == ["mystery_coef"]

    def test_disappearing_hyperparameter_fails_and_is_named(
        self, drift_gate: HyperparameterDriftGate
    ) -> None:
        current = _hparams()
        del current["entropy_coef"]
        result = drift_gate.run(_healthy_ctx(current_hparams=current))
        assert result.verdict is Verdict.FAIL
        assert result.evidence["changed_keys"] == ["entropy_coef"]

    def test_value_type_change_is_drift(self, drift_gate: HyperparameterDriftGate) -> None:
        # 0.2 → "0.2" survives a surprising number of dashboards. The
        # fingerprint must not.
        current = _hparams()
        current["clip_epsilon"] = "0.2"
        result = drift_gate.run(_healthy_ctx(current_hparams=current))
        assert result.verdict is Verdict.FAIL
        assert result.evidence["changed_keys"] == ["clip_epsilon"]

    def test_present_but_none_key_is_distinguished_from_absent(
        self, drift_gate: HyperparameterDriftGate
    ) -> None:
        # ``{a: None}`` and ``{}`` are not the same hyperparameter set, and a
        # diff that treats them as equal is how a config row gets silently
        # dropped. This is why the module keeps the _ABSENT sentinel.
        step0 = {"a": None, "x": 1}
        result = drift_gate.run(
            _healthy_ctx(
                step0_hparams=step0,
                step0_fingerprint=fingerprint_hparams({"a": 0.0, "x": 1}),
                current_hparams={"x": 1},
            ),
        )
        assert result.verdict is Verdict.FAIL
        assert result.evidence["changed_keys"] == ["a"]
        assert "a" in result.detail

    def test_missing_step0_fingerprint_fails_as_unfalsifiable(
        self, drift_gate: HyperparameterDriftGate
    ) -> None:
        # Nothing recorded at step 0 means the hyperparameters in force can
        # never be contradicted by any artefact. Unfalsifiable is the incident,
        # so this is FAIL, not SKIP.
        result = drift_gate.run(_healthy_ctx(step0_fingerprint=None, step0_hparams=None))
        assert result.verdict is Verdict.FAIL
        assert result.blocking
        assert "no step-0 hyperparameter fingerprint" in result.detail

    def test_zero_live_hyperparameters_is_vacuous_never_pass(
        self, drift_gate: HyperparameterDriftGate
    ) -> None:
        # An empty live view compared against anything (including a valid
        # step-0 fingerprint!) proves nothing. ``all([])`` does not run here.
        result = drift_gate.run(_healthy_ctx(current_hparams={}))
        assert result.verdict is Verdict.VACUOUS
        assert result.blocking

    def test_drift_without_a_step0_snapshot_still_blocks(
        self, drift_gate: HyperparameterDriftGate
    ) -> None:
        # The fingerprint is the verdict; the key-by-key diff is evidence that
        # only exists when the step-0 snapshot was kept. A mismatch with no
        # snapshot must still FAIL, with the evidence honestly saying so.
        result = drift_gate.run(
            _healthy_ctx(
                step0_hparams=None,
                step0_fingerprint="0" * 64,
            ),
        )
        assert result.verdict is Verdict.FAIL
        assert result.evidence["changed_keys"] is None
        assert "changed:" not in result.detail


class TestContextCoercion:
    """Gates accept the frozen context or anything carrying the same fields.
    Anything else must fail closed as ERROR, because in the audited estate a
    verifier exception counted as a pass."""

    def test_duck_typed_context_with_the_fields_is_accepted(
        self, declared_gate: ObjectiveDeclaredGate
    ) -> None:
        class _DuckContext:
            def __init__(self, objective: ValueProvenance | None) -> None:
                self.objective = objective
                self.declared_components: tuple[str, ...] = ()
                self.components: tuple[LossComponent, ...] = ()

        result = declared_gate.run(_DuckContext(_provenance()))
        assert result.verdict is Verdict.PASS

    def test_duck_context_missing_optional_fields_gets_documented_defaults(
        self, reward_gate: RewardScaleSanityGate
    ) -> None:
        # uses_rewards defaults to False and reward_stats to None, so a minimal
        # duck context takes the declared-not-applicable path. If the defaults
        # silently changed to uses_rewards=True, this would become VACUOUS and
        # every minimal integration would start blocking.
        class _MinimalDuck:
            objective = _provenance()
            declared_components: tuple[str, ...] = ()
            components: tuple[LossComponent, ...] = ()

        result = reward_gate.run(_MinimalDuck())
        assert result.verdict is Verdict.SKIP

    def test_context_without_objective_field_fails_closed_as_error(
        self, declared_gate: ObjectiveDeclaredGate
    ) -> None:
        result = declared_gate.run(object())
        assert result.verdict is Verdict.ERROR
        assert result.blocking
        assert "TypeError" in result.detail
        assert "ObjectiveGateContext" in result.detail


class TestControlsArePrecise:
    """``verify_controls`` only demands that MUST_FIRE fixtures *block*. A gate
    that reported VACUOUS where the incident demands FAIL would still satisfy
    it. These tests pin the exact verdict the module's own fixtures produce, so
    a severity downgrade cannot ride through as "still a positive control"."""

    def test_every_gate_declares_both_control_kinds(self) -> None:
        for gate in (
            ObjectiveDeclaredGate(),
            LossComponentCoverageGate(),
            RewardScaleSanityGate(),
            HyperparameterDriftGate(),
        ):
            kinds = {c.kind for c in gate.controls()}
            assert ControlKind.MUST_FIRE in kinds, gate.id
            assert ControlKind.MUST_PASS in kinds, gate.id

    def test_all_module_controls_hold_under_verify_controls(self) -> None:
        # The end-to-end property the whole framework rests on: every fixture
        # the gates ship actually discriminates, in one CI-callable assertion.
        registry = GateRegistry()
        registry.register(ObjectiveDeclaredGate())
        registry.register(LossComponentCoverageGate())
        registry.register(RewardScaleSanityGate())
        registry.register(HyperparameterDriftGate())
        assert verify_controls(registry) == []

    @pytest.mark.parametrize(
        ("gate", "control_name", "expected"),
        [
            (ObjectiveDeclaredGate(), "env-shadowed-default", Verdict.FAIL),
            (ObjectiveDeclaredGate(), "unrecorded-env-switch", Verdict.FAIL),
            (ObjectiveDeclaredGate(), "declared-and-recorded", Verdict.PASS),
            (LossComponentCoverageGate(), "empty-component-list", Verdict.VACUOUS),
            (LossComponentCoverageGate(), "zero-weight-trust-region", Verdict.FAIL),
            (LossComponentCoverageGate(), "absent-declared-component", Verdict.FAIL),
            (LossComponentCoverageGate(), "no-contributions-measured", Verdict.VACUOUS),
            (LossComponentCoverageGate(), "all-components-contributing", Verdict.PASS),
            (LossComponentCoverageGate(), "partially-measured-contributions", Verdict.PASS),
            (RewardScaleSanityGate(), "all-identical-rewards", Verdict.FAIL),
            (RewardScaleSanityGate(), "out-of-bounds-rewards", Verdict.FAIL),
            (RewardScaleSanityGate(), "no-samples-on-reward-objective", Verdict.VACUOUS),
            (RewardScaleSanityGate(), "healthy-rewards", Verdict.PASS),
            (RewardScaleSanityGate(), "stats-supplied-without-reward-term", Verdict.PASS),
            (HyperparameterDriftGate(), "kl-coef-slid-to-zero", Verdict.FAIL),
            (HyperparameterDriftGate(), "no-step0-fingerprint", Verdict.FAIL),
            (HyperparameterDriftGate(), "no-hyperparameters-at-all", Verdict.VACUOUS),
            (HyperparameterDriftGate(), "matches-step0-record", Verdict.PASS),
        ],
    )
    def test_control_produces_its_exact_verdict(
        self, gate: Gate, control_name: str, expected: Verdict
    ) -> None:
        control = _control_by_name(gate, control_name)
        result = gate.run(control.make_ctx())
        assert result.verdict is expected


class TestRegistration:
    """A gate that exists on disk but was never registered for its lifecycle
    events is the estate's classic failure: a picket fence with a gate drawn
    on it. These pin the ids and the events each gate must answer for."""

    _IDS = {
        "objective.declared",
        "objective.loss_components",
        "objective.reward_scale",
        "objective.hparam_drift",
    }

    def test_all_four_gates_are_registered(self) -> None:
        registered = {gate.id for gate in REGISTRY}
        assert registered >= self._IDS

    def test_each_gate_runs_at_its_declared_lifecycle_events(self) -> None:
        launch_ids = {g.id for g in REGISTRY.for_event(Lifecycle.LAUNCH)}
        step_zero_ids = {g.id for g in REGISTRY.for_event(Lifecycle.STEP_ZERO)}
        save_ids = {g.id for g in REGISTRY.for_event(Lifecycle.SAVE)}
        assert "objective.declared" in launch_ids
        assert step_zero_ids >= self._IDS
        assert "objective.hparam_drift" in save_ids
        # The declared gate does NOT re-run at every save: provenance is a
        # launch-time fact, and re-auditing it per save against a live env is
        # exactly the re-read-the-config mistake the drift gate exists to refuse.
        assert "objective.declared" not in save_ids
