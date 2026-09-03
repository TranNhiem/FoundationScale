"""Objective gates: proving that the objective a run *is* optimising is the one it *claims*\nto be.

Why this module exists
----------------------
A 24-run family, launched from byte-identical argv, split 12/12 between two different
objectives. The switch was an environment variable: read by the launcher, defaulted
silently when absent, and recorded in no manifest, log, or checkpoint. When the reward
curves disagreed months later, the objective actually in force was unrecoverable — the
artefacts did not contain the fact.

The same shape recurred inside the objectives themselves. A trust region whose
--old_logp_source defaulted to ``self`` made the importance ratio identically 1.0 in
seven of ten arms: the component was listed, instantiated, and contributing nothing.
The runs moved about one bf16 ULP over hundreds of steps; nothing failed because
"the loss went down somewhere". And 472 of 1,876 logged steps had ``grad_norm``
exactly 0.000 while ``reward/mean=0.794`` sat in declared-normal range — every reward
in those batches was identical, so every advantage was identically zero and the
gradient vanished. The averages looked healthy because averages are exactly what the
degenerate case preserves.

So the gates here enforce three claims with counted coverage:

1. **Declared** — the objective in force is an effective value with a named source,
   recorded before any process starts. An env-shadowed default is FAIL, not a warning.
2. **Contributing** — every loss component the objective claims is present, weighted
   non-zero, and measured contributing. An empty component list is VACUOUS: this is
   the ``gspo._zero_metrics`` twin, where a step that supervised nothing passed the
   only semantic gate because the checker read fabricated keys and ignored the flag
   beside them. ``all([]) is True`` does not run here.
3. **Stable** — reward statistics are finite, in-bounds, and non-degenerate, and the
   hyperparameters at step N fingerprint-match the step-0 record. The comparison is
   against the recorded fingerprint, never a re-read of a config file the environment
   may have moved underneath.

Contexts are built by the launcher/trainer from the *effective* configuration view —
after CLI, config file, environment and defaults have merged — and from observation of
an actual step (measured component contributions, sampled rewards). Building from any
earlier view is how an env-set switch ends up audited against the wrong objective.
Controls build synthetic contexts in this module, so :func:`verify_controls` runs with
no trainer, no torch, and no I/O.
"""

from __future__ import annotations

import hashlib
import json
import math
from abc import ABC
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, ClassVar

from .core import (
    Control,
    ControlKind,
    Coverage,
    Gate,
    GateResult,
    Lifecycle,
    register,
)

__all__ = [
    "ValueProvenance",
    "LossComponent",
    "RewardStats",
    "ObjectiveGateContext",
    "fingerprint_hparams",
    "ObjectiveDeclaredGate",
    "LossComponentCoverageGate",
    "RewardScaleSanityGate",
    "HyperparameterDriftGate",
]

# Provenance a resolver is allowed to claim. "env" is a legitimate source when it is
# recorded with its value; what the incident killed was env-as-source *unrecorded*.
_KNOWN_SOURCES = frozenset({"cli", "config", "default", "env"})

_ABSENT = object()
"""Sentinel for key-diffing so a present-but-``None`` value is distinguishable from absent."""

# Relative slack for the RewardStats coherence rules in RewardScaleSanityGate.
# The rules are exact theorems of real arithmetic (a mean of values bounded by
# [min, max] cannot leave [min, max]; a std over that range is analytically
# bounded), but the numbers reach the gate after fp32/fp64 accumulation and
# sometimes rounded serialisation, so an honest summary can land a hair past
# the exact bound. The slack is orders of magnitude below any violation a
# fabrication produces in practice — the incident estate's std values were
# comparable to the entire range, ~2x the ceiling. It exists so the gate fires
# on fiction, not rounding dust: a limits gate that false-fires on float noise
# gets muted within a week, and the estate has already demonstrated what it
# learns from that.
_MEAN_RANGE_SLACK = 1e-6
_STD_CEILING_SLACK = 1e-6


@dataclass(frozen=True)
class ValueProvenance:
    """One effective configuration value and where it came from.

    ``shadowed_by_env`` is set by the resolver when the named environment variable was
    defined in the launch environment with a value *different* from the effective one
    — i.e. the environment tried to steer the objective and silently lost. That is the
    exact 12/12-split signature, and surfacing it as data (rather than a log line) is
    what lets :class:`ObjectiveDeclaredGate` make it a FAIL.
    """

    name: str
    value: Any
    source: str
    recorded: bool
    env_var: str | None = None
    shadowed_by_env: bool = False
    env_value: Any | None = None


@dataclass(frozen=True)
class LossComponent:
    """One term of the objective as observed at the inspected step.

    ``contribution`` is the measured magnitude of this component's term at that step;
    ``None`` means unmeasured, in which case no contributing/not-contributing claim is
    made either way. A measured contribution of exactly 0.0 with a declared non-zero
    weight is the ratio-identically-1.0 signature: listed, instantiated, inert.

    ``None`` is a per-component abstention, not a licence at aggregate scope: a step
    in which EVERY declared component carries ``None`` leaves
    :class:`LossComponentCoverageGate`'s inert-component detector with zero examined
    units, and the gate answers that sweep with a blocking VACUOUS that names 0
    measured — never with ``all([])`` and a full-looking numerator.
    """

    name: str
    weight: float
    observed: bool
    contribution: float | None = None


@dataclass(frozen=True)
class RewardStats:
    """Reward statistics over the samples actually inspected at the gate point."""

    n: int
    mean: float
    std: float
    min: float
    max: float


@dataclass(frozen=True)
class ObjectiveGateContext:
    """Everything the objective gates need.

    ``objective``, ``declared_components``, ``step0_fingerprint`` come from the run
    manifest — the run's own statement of what it set out to optimise. ``components``,
    ``reward_stats`` and ``current_hparams`` come from observing the live step. The
    audit rule applies directly: comparing what is in force against what is in force
    is vacuous; comparing it against what the run declared is the check.
    """

    objective: ValueProvenance | None
    declared_components: tuple[str, ...]
    components: tuple[LossComponent, ...]
    uses_rewards: bool = False
    reward_stats: RewardStats | None = None
    reward_bounds: tuple[float, float] | None = None
    expected_sample_count: int | None = None
    step0_fingerprint: str | None = None
    step0_hparams: Mapping[str, Any] | None = None
    current_hparams: Mapping[str, Any] = field(default_factory=dict)
    origin: str = "<context>"


def fingerprint_hparams(hparams: Mapping[str, Any]) -> str:
    """Canonical, stable fingerprint of a hyperparameter mapping.

    Key order is normalised before serialisation so two semantically equal mappings
    fingerprint identically regardless of insertion order; non-JSON values fall back
    to ``repr`` so the function total-ises instead of raising inside a gate.

    Args:
        hparams: The mapping to fingerprint. Values are expected to be JSON-ish.

    Returns:
        A hex SHA-256 digest over the canonical serialisation.
    """
    ordered = {key: hparams[key] for key in sorted(hparams, key=str)}
    canonical = json.dumps(ordered, sort_keys=True, separators=(",", ":"), default=repr)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _changed_keys(step0: Mapping[str, Any], current: Mapping[str, Any]) -> list[str]:
    """Keys whose effective value differs between two hyperparameter snapshots."""
    keys = set(step0) | set(current)
    return sorted(k for k in keys if step0.get(k, _ABSENT) != current.get(k, _ABSENT))


def _coerce(ctx: Any) -> ObjectiveGateContext:
    """Accept a context or any duck-typed object carrying an ``.objective``."""
    if isinstance(ctx, ObjectiveGateContext):
        return ctx
    if hasattr(ctx, "objective"):
        return ObjectiveGateContext(
            objective=ctx.objective,
            declared_components=tuple(ctx.declared_components),
            components=tuple(ctx.components),
            uses_rewards=bool(getattr(ctx, "uses_rewards", False)),
            reward_stats=getattr(ctx, "reward_stats", None),
            reward_bounds=getattr(ctx, "reward_bounds", None),
            expected_sample_count=getattr(ctx, "expected_sample_count", None),
            step0_fingerprint=getattr(ctx, "step0_fingerprint", None),
            step0_hparams=getattr(ctx, "step0_hparams", None),
            current_hparams=getattr(ctx, "current_hparams", {}),
            origin=getattr(ctx, "origin", repr(ctx)),
        )
    raise TypeError(f"objective gates need an ObjectiveGateContext, got {type(ctx).__name__}")


# ---------------------------------------------------------------------------
# Control fixtures. These build synthetic contexts directly so verify_controls
# exercises every gate with no trainer, torch, or I/O. The healthy baseline is
# constructed once here and perturbed per fixture, so a fixture's defect is the
# only difference from a passing context — a fixture that *also* broke something
# unrelated would be visible immediately in MUST_PASS runs.
# ---------------------------------------------------------------------------


def _healthy_hparams() -> dict[str, Any]:
    return {
        "objective": "gspo",
        "clip_epsilon": 0.2,
        "kl_coef": 0.04,
        "entropy_coef": 0.001,
        "group_size": 8,
    }


def _healthy_ctx(**overrides: Any) -> ObjectiveGateContext:
    kwargs: dict[str, Any] = {
        "objective": ValueProvenance(
            name="objective",
            value="gspo",
            source="cli",
            recorded=True,
            env_var="FOUNDATIONSCALE_OBJECTIVE",
        ),
        "declared_components": ("policy_loss", "kl_penalty", "entropy_bonus"),
        "components": (
            LossComponent(name="policy_loss", weight=1.0, observed=True, contribution=0.83),
            LossComponent(name="kl_penalty", weight=0.04, observed=True, contribution=0.011),
            LossComponent(name="entropy_bonus", weight=0.001, observed=True, contribution=0.0021),
        ),
        "uses_rewards": True,
        "reward_stats": RewardStats(n=256, mean=0.794, std=0.31, min=-1.25, max=2.4),
        "reward_bounds": (-10.0, 10.0),
        "expected_sample_count": 256,
        "step0_fingerprint": fingerprint_hparams(_healthy_hparams()),
        "step0_hparams": _healthy_hparams(),
        "current_hparams": _healthy_hparams(),
        "origin": "<fixture>",
    }
    kwargs.update(overrides)
    return ObjectiveGateContext(**kwargs)


def _healthy_objective_ctx() -> ObjectiveGateContext:
    return _healthy_ctx()


def _env_shadowed_default_ctx() -> ObjectiveGateContext:
    # The 24-run incident: env var set to one objective, default supplied another,
    # and the record said nothing about the disagreement until now.
    return _healthy_ctx(
        objective=ValueProvenance(
            name="objective",
            value="grpo",
            source="default",
            recorded=True,
            env_var="FOUNDATIONSCALE_OBJECTIVE",
            shadowed_by_env=True,
            env_value="gspo",
        ),
    )


def _unrecorded_env_switch_ctx() -> ObjectiveGateContext:
    return _healthy_ctx(
        objective=ValueProvenance(
            name="objective",
            value="gspo",
            source="env",
            recorded=False,
            env_var="FOUNDATIONSCALE_OBJECTIVE",
        ),
    )


def _empty_component_list_ctx() -> ObjectiveGateContext:
    return _healthy_ctx(declared_components=(), components=())


def _zero_weight_component_ctx() -> ObjectiveGateContext:
    return _healthy_ctx(
        components=(
            LossComponent(name="policy_loss", weight=1.0, observed=True, contribution=0.83),
            LossComponent(name="kl_penalty", weight=0.0, observed=True, contribution=0.0),
            LossComponent(name="entropy_bonus", weight=0.001, observed=True, contribution=0.0021),
        ),
    )


def _absent_component_ctx() -> ObjectiveGateContext:
    return _healthy_ctx(
        components=(
            LossComponent(name="policy_loss", weight=1.0, observed=True, contribution=0.83),
            LossComponent(name="entropy_bonus", weight=0.001, observed=True, contribution=0.0021),
        ),
    )


def _no_contributions_measured_ctx() -> ObjectiveGateContext:
    # Every component present, observed, non-zero-weighted — and not one
    # contribution measured. Presence and weight come back clean while the
    # inert-component detector, the part of its gate that exists for the
    # ratio-identically-1.0 incident, examines zero units. A pass grade here is
    # all([]) wearing a covered numerator.
    return _healthy_ctx(
        components=(
            LossComponent(name="policy_loss", weight=1.0, observed=True, contribution=None),
            LossComponent(name="kl_penalty", weight=0.04, observed=True, contribution=None),
            LossComponent(name="entropy_bonus", weight=0.001, observed=True, contribution=None),
        ),
    )


def _partially_measured_components_ctx() -> ObjectiveGateContext:
    # kl_penalty declines the contribution leg while the other two components
    # carry measurements. This pins the documented None contract — abstention,
    # not accusation — while forcing the verdict to carry the measured/total
    # denominator and the abstainer's name.
    return _healthy_ctx(
        components=(
            LossComponent(name="policy_loss", weight=1.0, observed=True, contribution=0.83),
            LossComponent(name="kl_penalty", weight=0.04, observed=True, contribution=None),
            LossComponent(name="entropy_bonus", weight=0.001, observed=True, contribution=0.0021),
        ),
    )


def _stats_without_reward_term_ctx() -> ObjectiveGateContext:
    # Healthy reward statistics attached to an objective that declares no reward
    # term: the data exists, so it is examined — but the declaration and the
    # data contradict each other, and the verdict must surface that rather than
    # read as a clean bill for a term the objective says it does not have.
    return _healthy_ctx(uses_rewards=False)


def _identical_rewards_ctx() -> ObjectiveGateContext:
    # reward/mean=0.794 with zero spread: the 472-zero-grad-steps signature.
    return _healthy_ctx(
        reward_stats=RewardStats(n=472, mean=0.794, std=0.0, min=0.794, max=0.794),
    )


def _out_of_bounds_rewards_ctx() -> ObjectiveGateContext:
    return _healthy_ctx(
        reward_stats=RewardStats(n=256, mean=11.5, std=3.0, min=-2.0, max=31.0),
    )


def _missing_reward_stats_ctx() -> ObjectiveGateContext:
    return _healthy_ctx(reward_stats=None, expected_sample_count=None)


def _zero_std_with_range_ctx() -> ObjectiveGateContext:
    # W-obj-8 verbatim: std reported as exactly 0.0 across 256 samples that
    # supposedly span [-1.0, 2.0]. Zero std forces min == max, so one of the
    # two reported numbers is a lie. Whichever it is, the gate once PASSED it
    # while its own detail line asserted "std > 0".
    return _healthy_ctx(
        reward_stats=RewardStats(n=256, mean=0.5, std=0.0, min=-1.0, max=2.0),
    )


def _negative_std_ctx() -> ObjectiveGateContext:
    return _healthy_ctx(
        reward_stats=RewardStats(n=256, mean=0.5, std=-0.25, min=-1.0, max=2.0),
    )


def _inverted_range_ctx() -> ObjectiveGateContext:
    return _healthy_ctx(
        reward_stats=RewardStats(n=64, mean=0.0, std=0.3, min=2.0, max=-1.0),
    )


def _mean_outside_range_ctx() -> ObjectiveGateContext:
    return _healthy_ctx(
        reward_stats=RewardStats(n=128, mean=5.0, std=0.4, min=-1.0, max=2.0),
    )


def _std_above_ceiling_ctx() -> ObjectiveGateContext:
    # A range of 3.0 over 64 samples caps the sample std at 1.5*sqrt(64/63)
    # ~= 1.5119 and the population std at 1.5. Reporting std == the full range
    # width is the classic careless fabrication: it sounds plausible and is
    # ~2x what any real batch could reach.
    return _healthy_ctx(
        reward_stats=RewardStats(n=64, mean=0.5, std=3.0, min=-1.0, max=2.0),
    )


def _single_sample_with_spread_ctx() -> ObjectiveGateContext:
    return _healthy_ctx(
        reward_stats=RewardStats(n=1, mean=0.0, std=0.0, min=-1.0, max=2.0),
    )


def _negative_sample_count_ctx() -> ObjectiveGateContext:
    return _healthy_ctx(
        reward_stats=RewardStats(n=-3, mean=0.5, std=0.3, min=-1.0, max=2.0),
        expected_sample_count=None,
    )


def _drifted_hparams_ctx() -> ObjectiveGateContext:
    current = _healthy_hparams()
    current["kl_coef"] = 0.0  # the trust region silently slid to nothing between saves
    return _healthy_ctx(current_hparams=current)


def _no_step0_fingerprint_ctx() -> ObjectiveGateContext:
    return _healthy_ctx(step0_fingerprint=None, step0_hparams=None)


def _no_hparams_at_all_ctx() -> ObjectiveGateContext:
    return _healthy_ctx(current_hparams={}, step0_fingerprint=None, step0_hparams=None)


class _ObjectiveGate(Gate, ABC):
    """Base for the objective gates: declares the context these gates actually read.

    ``ABC`` is in the bases deliberately: :meth:`Gate.__init_subclass__` requires a
    non-empty ``id``/``description``/``events`` on every concrete subclass and
    exempts only classes that name ``ABC`` directly, which is the framework's way of
    spelling "shared behaviour, not a gate". Without it this base would have to
    invent an id and an event tuple, and a gate with a placeholder id is a gate that
    can be registered by accident.

    Every gate below opens with ``_coerce(ctx)``, which raises ``TypeError`` on any
    object without an ``.objective``. Left undeclared, a gate is LEGACY, and the
    legacy path hands it whatever context the sweep happens to be holding — so that
    ``TypeError`` surfaces as a blocking ERROR in a sweep that was never about
    objectives. ``objective.hparam_drift`` is registered for SAVE, which is the
    event the training loop's end-of-run adjudicator uses: a run that merely
    imported this module died there on a ``CheckpointGateContext`` (#250). Because
    registration is an import side effect, *which* runs died was decided by an
    import graph rather than by anything about the run.

    Declaring ``context_type`` turns that from a traceback into a dispatch fact.
    :func:`run_event` routes by the declared type, so a caller holding a foreign
    context gets a named "unwired" ERROR, or a SKIP where it declared the
    abstention — visible in the denominator, never a PASS, and never a stopped run.

    The base is applied to all four gates, not only the one with a save event. The
    others are STEP_ZERO/LAUNCH today, so they collide with nothing; that is a fact
    about the current event map, not a property of the gates, and a gate whose
    correctness depends on nobody registering it for another event is the same
    defect waiting for a different import.

    ``coerce_context`` preserves the duck-typing :func:`_coerce` documents — any
    object carrying an ``.objective``, which is what the trainer-side adapters
    supply — and refuses by returning ``None`` rather than raising, per the
    base-class contract: raising here would reproduce the opaque failure the hook
    exists to replace.
    """

    context_type = ObjectiveGateContext

    def coerce_context(self, ctx: Any) -> ObjectiveGateContext | None:
        return _coerce(ctx) if hasattr(ctx, "objective") else None


@register
class ObjectiveDeclaredGate(_ObjectiveGate):
    """Prevents the 24-run unrecorded-objective incident.

    The objective in force must be an *effective* value (post-merge of CLI, config,
    env, defaults) with a named source, recorded to an artefact before launch. Three
    failure shapes, all binary:

    1. Nothing recorded at all — VACUOUS, not "nothing to check".
    2. Effective value never recorded — FAIL, however healthy the source looks.
    3. Env-shadowed default — FAIL. The environment tried to steer the objective and
       lost silently; that is precisely how 24 byte-identical runs ran two objectives.
       In the incident estate this was, at best, a warning in a log nobody aggregated.
    """

    id: ClassVar[str] = "objective.declared"
    description: ClassVar[str] = (
        "The objective in force is recorded as an effective value with a named "
        "source; an env-shadowed default is a defect, not a warning"
    )
    events: ClassVar[tuple[Lifecycle, ...]] = (Lifecycle.LAUNCH, Lifecycle.STEP_ZERO)

    def check(self, ctx: Any) -> GateResult:
        c = _coerce(ctx)
        prov = c.objective
        if prov is None:
            # The record contains no objective claim at all. That is an absence of a
            # fact, not a verified fact — ok() downgrades it to VACUOUS.
            return self.ok(
                "no objective was recorded — the run's objective is unrecoverable "
                "from its artefacts",
                Coverage.none("objective selections"),
                evidence={"origin": c.origin},
            )
        cov = Coverage(1, "objective selections", expected=1)
        if not prov.recorded:
            return self.fail(
                f"objective {prov.value!r} is in force (source={prov.source!r}) but was "
                f"never recorded to any artefact — the 12/12 split was undiagnosable "
                f"for exactly this reason",
                cov,
                evidence={
                    "objective": prov.value,
                    "source": prov.source,
                    "env_var": prov.env_var,
                    "origin": c.origin,
                },
            )
        if prov.source not in _KNOWN_SOURCES:
            return self.fail(
                f"objective {prov.value!r} claims source={prov.source!r}, which is not "
                f"a provenance class this framework can trust {sorted(_KNOWN_SOURCES)}",
                cov,
                evidence={"objective": prov.value, "source": prov.source},
            )
        if prov.shadowed_by_env:
            return self.fail(
                f"objective {prov.value!r} came from source={prov.source!r} while "
                f"{prov.env_var}={prov.env_value!r} was set in the environment — an "
                f"env-shadowed default is how 24 runs split 12/12 between two "
                f"objectives with identical argv",
                cov,
                evidence={
                    "effective": prov.value,
                    "source": prov.source,
                    "env_var": prov.env_var,
                    "env_value": prov.env_value,
                    "origin": c.origin,
                },
            )
        return self.ok(
            f"objective {prov.value!r} in force, source={prov.source!r}, recorded with provenance",
            cov,
            evidence={
                "objective": prov.value,
                "source": prov.source,
                "env_var": prov.env_var,
                "origin": c.origin,
            },
        )

    def controls(self) -> list[Control]:
        return [
            Control(
                "env-shadowed-default",
                ControlKind.MUST_FIRE,
                _env_shadowed_default_ctx,
                note="env var says gspo, default says grpo, record silent — the incident",
            ),
            Control(
                "unrecorded-env-switch",
                ControlKind.MUST_FIRE,
                _unrecorded_env_switch_ctx,
                note="objective came from the environment and was never recorded",
            ),
            Control(
                "declared-and-recorded",
                ControlKind.MUST_PASS,
                _healthy_objective_ctx,
                note="effective objective recorded with named source, no env shadowing",
            ),
        ]


@register
class LossComponentCoverageGate(_ObjectiveGate):
    """Prevents listed-but-inert objective components, and the empty-list pass.

    Two incident shapes share one gate because they share one root cause — trusting
    the objective's *description* of itself:

    * The trust region that optimised nothing: ``old_logp_source=self`` made the
      ratio identically 1.0, so the KL/trust term was declared, instantiated, and
      contributing exactly 0.0 for seven of ten arms. A zero weight or a measured
      zero contribution on a declared component is that signature, surfaced.
    * The ``gspo._zero_metrics`` twin: a step supervising nothing passed the gate
      because "no missing components" over an empty component set is ``all([])``.
      Here an empty declared-component list is VACUOUS and blocks — an objective
      that claims to optimise nothing is not a passable state.

    The check is bidirectional: components observed contributing but never declared
    also fail, because the objective in force exceeding its own record is the same
    unrecoverability with the signs flipped.

    A third, quieter shape sits between those two: every declared component
    present and weighted, and not one contribution measured. ``contribution=None``
    is a legitimate per-component abstention (see :class:`LossComponent`), but a
    step where EVERY component abstains leaves this gate's inert-component
    detector — the part that exists for the trust-region incident — having
    examined zero units. That sweep returns VACUOUS naming 0 measured, never
    PASS. Partial abstention passes as a declared sample: the numerator counts
    only components verified on every leg, and the abstainers are named in the
    sample reason rather than averaged into an unqualified "N/N checked".
    """

    id: ClassVar[str] = "objective.loss_components"
    description: ClassVar[str] = (
        "Every declared loss component is present, non-zero-weighted, and measured "
        "contributing; an empty component list is vacuous, not clean"
    )
    events: ClassVar[tuple[Lifecycle, ...]] = (Lifecycle.STEP_ZERO,)

    def check(self, ctx: Any) -> GateResult:
        c = _coerce(ctx)
        declared = tuple(dict.fromkeys(c.declared_components))
        if not declared:
            return self.ok(
                "objective declares zero loss components — 'no missing components' "
                "over an empty set is not a pass",
                Coverage.none("loss components"),
                evidence={
                    "observed": [comp.name for comp in c.components],
                    "origin": c.origin,
                },
            )
        declared_set = set(declared)
        by_name = {comp.name: comp for comp in c.components}

        missing = [n for n in declared if n not in by_name or not by_name[n].observed]
        zero_weight: list[str] = []
        inert: list[str] = []
        non_finite: list[str] = []
        measured_names: list[str] = []
        unmeasured_names: list[str] = []
        for name in declared:
            comp = by_name.get(name)
            if comp is None or not comp.observed:
                continue  # already counted in `missing`; report each component once
            # Track the contribution leg separately from presence/weight: it is
            # the only conditional leg of the check, and the verdict below may
            # only claim it — or count a component toward it — when it ran.
            if comp.contribution is not None:
                measured_names.append(name)
            else:
                unmeasured_names.append(name)
            if not math.isfinite(comp.weight):
                non_finite.append(f"{name} (weight={comp.weight!r})")
            elif comp.weight == 0.0:
                zero_weight.append(name)
            elif comp.contribution is not None and not math.isfinite(comp.contribution):
                non_finite.append(f"{name} (contribution={comp.contribution!r})")
            elif comp.contribution is not None and comp.contribution == 0.0:
                inert.append(name)
        undeclared = sorted(
            n for n, comp in by_name.items() if n not in declared_set and comp.observed
        )

        problems: list[str] = []
        if missing:
            problems.append(
                f"declared components absent at this step: {missing} — the objective "
                f"claims to optimise terms that never entered the loss"
            )
        if non_finite:
            problems.append(
                f"components with non-finite values: {non_finite} — a NaN or infinite "
                f"weight or contribution has poisoned the objective and compares equal "
                f"to nothing, so it slips past every zero check"
            )
        if zero_weight:
            problems.append(
                f"components with weight 0.0: {zero_weight} — a listed component that "
                f"optimises nothing is the trust-region-off-by-default signature"
            )
        if inert:
            problems.append(
                f"components measured contributing exactly 0.0: {inert} — present, "
                f"weighted, and inert, the ratio-identically-1.0 shape"
            )
        if undeclared:
            problems.append(
                f"observed components never declared: {undeclared} — the objective in "
                f"force exceeds its own record"
            )

        examined = len(declared) - len(missing)
        measured = len(measured_names)
        cov = Coverage(
            checked=examined,
            unit="loss components",
            expected=len(declared),
        )
        if problems:
            return self.fail(
                problems[0] + (f" (+{len(problems) - 1} more)" if len(problems) > 1 else ""),
                cov,
                evidence={
                    "problems": problems,
                    "declared": list(declared),
                    "contributions_measured": measured,
                    "contributions_unmeasured": unmeasured_names,
                    "origin": c.origin,
                },
            )
        if measured == 0:
            # Presence and weight came back clean, but not one contribution
            # carried a measurement — so "no component is inert" over this step
            # is all([]) with a covered numerator: 3/3 components examined for
            # presence, 0/3 examined for the relation this gate exists to check.
            # ok() over empty coverage downgrades to a blocking VACUOUS that
            # names the 0; the claimed clause records what WAS established so
            # the downgrade reads as scoped abstention, not malfunction.
            # (examined >= 1 here: declared is non-empty and any missing
            # component would have produced a FAIL above.)
            return self.ok(
                f"all {len(declared)} declared components present and "
                f"non-zero-weighted, but contribution was measured for 0 of "
                f"{examined} — the inert-component check examined nothing this step",
                Coverage.none("contribution measurements"),
                evidence={
                    "declared": list(declared),
                    "contributions_measured": 0,
                    "contributions_unmeasured": unmeasured_names,
                    "origin": c.origin,
                },
            )
        if unmeasured_names:
            # Partial measurement: some components declined the contribution leg
            # (the LossComponent contract makes that an abstention, not a
            # defect), so the numerator counts only components verified on every
            # leg and the shortfall enters through the framework's declared-sample
            # lane — Coverage(sampled=True) exists precisely so partial coverage
            # is admitted in the open, with the abstainers named, instead of
            # being averaged into an unqualified "N/N checked". The verdict stays
            # PASS because no defect was found; the claim now stops at the
            # evidence.
            return self.ok(
                f"all {len(declared)} declared components present, weighted; "
                f"contribution established for {measured} of {examined} "
                f"(unmeasured make no claim either way): {unmeasured_names}",
                Coverage(
                    checked=measured,
                    unit="loss components",
                    expected=len(declared),
                    sampled=True,
                    sample_reason=(
                        f"contribution unmeasured for {len(unmeasured_names)} "
                        f"of {examined} present components: {unmeasured_names}; "
                        f"per the LossComponent contract an unmeasured "
                        f"contribution carries no claim either way"
                    ),
                ),
                evidence={
                    "declared": list(declared),
                    "contributions_measured": measured,
                    "contributions_unmeasured": unmeasured_names,
                    "origin": c.origin,
                },
            )
        return self.ok(
            f"all {len(declared)} declared components present, weighted, contributing",
            cov,
        )

    def controls(self) -> list[Control]:
        return [
            Control(
                "empty-component-list",
                ControlKind.MUST_FIRE,
                _empty_component_list_ctx,
                note="supervises nothing and must not pass — the all([]) / _zero_metrics trap",
            ),
            Control(
                "zero-weight-trust-region",
                ControlKind.MUST_FIRE,
                _zero_weight_component_ctx,
                note="kl_penalty declared at weight 0.0: listed, instantiated, inert",
            ),
            Control(
                "absent-declared-component",
                ControlKind.MUST_FIRE,
                _absent_component_ctx,
                note="kl_penalty declared but never entered the loss at this step",
            ),
            Control(
                "no-contributions-measured",
                ControlKind.MUST_FIRE,
                _no_contributions_measured_ctx,
                note="all three components present and weighted, zero contributions "
                "measured — the inert-component detector examined nothing and "
                "must block (VACUOUS naming 0), not issue a pass grade",
            ),
            Control(
                "all-components-contributing",
                ControlKind.MUST_PASS,
                _healthy_objective_ctx,
                note="three declared components, all observed with non-zero contribution",
            ),
            Control(
                "partially-measured-contributions",
                ControlKind.MUST_PASS,
                _partially_measured_components_ctx,
                note="kl_penalty unmeasured: abstains that leg as a declared sample "
                "without blocking — pins the documented None contract so the "
                "zero-measurement fix cannot grow into 'unmeasured is a defect'",
            ),
        ]


@register
class RewardScaleSanityGate(_ObjectiveGate):
    """Prevents healthy-looking averages on a dead policy gradient.

    The logged evidence from the incident: 1,876 steps, 472 with ``grad_norm``
    exactly 0.000, while ``reward/mean=0.794`` and ``success=1.00`` sat comfortably
    in range. Every reward in those batches was identical; identical rewards make
    every advantage identically zero, and the gradient vanishes *while every average
    a dashboard shows stays healthy*, because means are precisely what the degenerate
    case preserves. This gate therefore treats zero reward variance across a
    multi-sample batch as FAIL, not as a curiosity — alongside the conventional
    checks (finite statistics, declared bounds) that alone would have waved it through.

    Second incident shape (W-obj-8): the gate adjudicates a CALLER-SUPPLIED
    aggregate and never sees the raw samples, so any RewardStats that is
    internally impossible — std exactly 0 over a nonzero reported range, std
    beyond the analytic ceiling of its range, a mean outside [min, max], an
    inverted range, spread over fewer than two samples, a negative sample
    count — is fabrication or corruption, and accepting it means auditing
    fiction. Those summaries FAIL here, and the PASS detail states only
    relations the checks actually evaluated: the pre-fix "std={...} > 0"
    wording was a hardcoded literal that shipped a verdict claiming positive
    spread about a batch whose attached evidence said std=0.0.
    """

    id: ClassVar[str] = "objective.reward_scale"
    description: ClassVar[str] = (
        "Reward statistics are finite and inside declared bounds, the "
        "all-identical-rewards degenerate case (zero variance, healthy mean) is "
        "caught explicitly, and caller-supplied aggregates that are internally "
        "impossible (std = 0 over a nonzero range; std beyond the analytic "
        "ceiling; mean outside [min, max]; inverted range; spread over n < 2; "
        "negative n) fail as fabricated summaries"
    )
    events: ClassVar[tuple[Lifecycle, ...]] = (Lifecycle.STEP_ZERO,)

    def check(self, ctx: Any) -> GateResult:
        c = _coerce(ctx)
        stats = c.reward_stats
        # Order is load-bearing: "are there statistics?" is asked BEFORE "does
        # the objective declare rewards?" because the answers compose
        # asymmetrically. Supplied statistics are always examined below — even
        # when uses_rewards is False, where the final verdict surfaces the
        # declaration contradiction as data. The skip is therefore reachable
        # only when no statistics exist at all, the one state where its reason
        # ("nothing to bound") is a verified fact rather than a claim about
        # data nobody looked at. Reversing these tests would let an abstention
        # wave past populated samples — the defect shape this framework exists
        # to catch, committed by the abstainer itself.
        if stats is None:
            if not c.uses_rewards:
                return self.skip("objective declares no reward term; nothing to bound")
            # A reward-bearing objective with zero samples examined has not passed a
            # sanity check; ok() downgrades this to VACUOUS.
            return self.ok(
                "objective uses rewards but no reward samples were collected",
                Coverage.none("reward samples"),
                evidence={"origin": c.origin},
            )
        if stats.n < 0:
            # Hoisted ABOVE the Coverage construction on purpose. Coverage rejects a
            # negative count in __post_init__, so leaving this in the cascade below
            # made the gate die with "ValueError: coverage cannot be negative: -3"
            # — blocking, so it failed closed, but an ERROR traceback is a worse
            # artifact than a stated reason. A negative count was never "collected":
            # the summary describes no batch that ran. n == 0 deliberately does not
            # join this branch — zero samples is doctrine-1 VACUOUS via empty
            # coverage, not fabrication. Reported alone and before the finiteness
            # check, because a fantasy count invalidates the denominator itself.
            return self.fail(
                f"reward sample count is {stats.n} — no batch ever contained a "
                f"negative number of rewards; the summary is fabricated or corrupted",
                Coverage.none("reward samples"),
                evidence={
                    "problems": [
                        f"reward sample count is {stats.n} — no batch ever contained a "
                        f"negative number of rewards; the summary is fabricated or "
                        f"corrupted"
                    ],
                    "stats": {
                        "n": stats.n,
                        "mean": stats.mean,
                        "std": stats.std,
                        "min": stats.min,
                        "max": stats.max,
                    },
                    "origin": c.origin,
                },
            )
        cov = Coverage(stats.n, "reward samples", expected=c.expected_sample_count)

        problems: list[str] = []
        values = (stats.mean, stats.std, stats.min, stats.max)
        finite = all(math.isfinite(v) for v in values)
        if not finite:
            problems.append(
                f"non-finite reward statistics (mean={stats.mean}, std={stats.std}, "
                f"min={stats.min}, max={stats.max})"
            )
        elif stats.min > stats.max:
            # An inverted range poisons every comparison below that assumes
            # min <= max, so it too is reported alone rather than cascaded over.
            problems.append(
                f"reported reward range is inverted: min={stats.min} > "
                f"max={stats.max} — no set of samples produces a minimum above "
                f"its maximum; the summary is self-contradictory"
            )
        elif stats.n > 1 and stats.min == stats.max:
            problems.append(
                f"all {stats.n} sampled rewards are identical ({stats.mean!r}): every "
                f"advantage is identically 0 and the policy gradient vanishes, while "
                f"reward/mean still logs a healthy-looking {stats.mean!r} — the "
                f"472-steps-of-grad_norm-0.000 signature"
            )
        else:
            # Finite statistics, n >= 0, min <= max, and a nonzero range whenever
            # n > 1 — the comparisons below now tell the truth. This is the
            # RewardStats coherence cascade (W-obj-8): the gate only ever sees a
            # caller-supplied aggregate, so a summary that no sample set could
            # produce is fabrication, and before this cascade the gate PASSED one
            # (std=0.0 over min=-1.0, max=2.0) while claiming "std > 0" in its
            # own verdict line.
            spread = stats.max - stats.min
            if stats.n < 2 and spread > 0.0:
                problems.append(
                    f"n={stats.n} sample(s) cannot span a range, yet min={stats.min} "
                    f"and max={stats.max} differ — a summary of fewer than two "
                    f"samples must have min == max"
                )
            slack = _MEAN_RANGE_SLACK * max(abs(stats.min), abs(stats.max))
            if stats.mean < stats.min - slack or stats.mean > stats.max + slack:
                problems.append(
                    f"reported mean {stats.mean!r} lies outside the reported range "
                    f"[{stats.min}, {stats.max}] — samples bounded by that range "
                    f"cannot average beyond it; the summary is self-contradictory"
                )
            if stats.std < 0.0:
                problems.append(
                    f"reported std {stats.std!r} is negative — a standard deviation "
                    f"is the square root of a mean of squares and cannot be "
                    f"negative under any convention"
                )
            elif stats.std == 0.0 and stats.n > 1 and spread > 0.0:
                # The exact W-obj-8 batch. std vanishes iff every sample is equal,
                # which forces min == max; a nonzero reported range contradicts it.
                # Either the rewards really varied and the std is fabricated, or
                # they did not and the range is — the second reading is the
                # 472-zero-grad-steps incident arriving through the aggregate.
                problems.append(
                    f"reported std is exactly 0.0 while rewards span "
                    f"[{stats.min}, {stats.max}] over {stats.n} samples — zero "
                    f"spread with a nonzero range is arithmetically impossible"
                )
            else:
                # Analytic ceiling: a population std over [min, max] is at most
                # spread/2 (Popoviciu; half the mass at each endpoint), and the
                # Bessel-corrected sample std at most spread/2*sqrt(n/(n-1)).
                # The sample bound is the looser, so enforcing it is legal under
                # either convention — the gate never sees which one the caller
                # used. For n < 2 no spread is observable at all: ceiling 0.
                ceiling = 0.0 if stats.n < 2 else spread / 2.0 * math.sqrt(stats.n / (stats.n - 1))
                if stats.std > ceiling * (1.0 + _STD_CEILING_SLACK):
                    problems.append(
                        f"reported std {stats.std!r} exceeds the analytic ceiling "
                        f"{ceiling:.6g} for {stats.n} sample(s) bounded by "
                        f"[{stats.min}, {stats.max}] (max sample std = "
                        f"range/2*sqrt(n/(n-1)); max population std = range/2) — "
                        f"no real batch can produce it"
                    )
        if c.reward_bounds is not None and math.isfinite(stats.min) and math.isfinite(stats.max):
            lo, hi = c.reward_bounds
            if not (math.isfinite(lo) and math.isfinite(hi)):
                problems.append(
                    f"declared reward bounds [{lo}, {hi}] are non-finite — every "
                    f"comparison against them is silently True, so the bounds check "
                    f"examined nothing while looking like it ran"
                )
            elif stats.min < lo or stats.max > hi:
                problems.append(
                    f"rewards span [{stats.min}, {stats.max}], outside declared bounds "
                    f"[{lo}, {hi}] — the reward scale has left the region the "
                    f"objective's coefficients were tuned for"
                )

        if problems:
            return self.fail(
                problems[0] + (f" (+{len(problems) - 1} more)" if len(problems) > 1 else ""),
                cov,
                evidence={
                    "problems": problems,
                    "stats": {
                        "n": stats.n,
                        "mean": stats.mean,
                        "std": stats.std,
                        "min": stats.min,
                        "max": stats.max,
                    },
                    "origin": c.origin,
                },
            )
        if c.reward_bounds is not None:
            lo, hi = c.reward_bounds
            bound_note = f"inside declared bounds [{lo}, {hi}]"
        else:
            bound_note = "with no declared bounds (degeneracy still checked)"
        if not c.uses_rewards:
            # Reachable only via populated reward_stats alongside
            # uses_rewards=False (the stats-None case skipped above): the
            # samples exist and were examined on their own merits, so the
            # statistics earn the verdict — but a verdict silent about
            # "objective declares no reward term, yet N reward statistics were
            # supplied" would hide a wiring contradiction. It is surfaced as
            # data and NOT made a defect: nothing here observed the statistics
            # to be wrong, and a divergence claim nobody observed is as
            # counterfeit as a pass nobody earned.
            declared_note = (
                "; note: the objective declares no reward term, yet these "
                "statistics were supplied — examined and reported, the "
                "contradiction surfaced rather than skipped or failed"
            )
        else:
            declared_note = ""
        # State only relations the checks above actually evaluated (W-obj-8: the
        # pre-fix line ended in a literal "std=... > 0" that was never computed).
        # At this point the cascade guarantees std > 0 whenever n > 1 — a zero std
        # with a nonzero range failed, and a collapsed range failed earlier — so
        # the positivity claim is earned on exactly that branch and no other.
        if stats.n > 1:
            spread_note = f"std={stats.std:.4g} > 0 over [{stats.min:.4g}, {stats.max:.4g}]"
        elif stats.n == 1:
            spread_note = (
                f"single sample at mean={stats.mean:.4g}; spread is unobservable "
                f"at n=1 and no positivity claim is made"
            )
        else:
            # n == 0 reaches ok() only to be downgraded to VACUOUS by the empty
            # coverage; the detail must not describe a measurement that never ran.
            spread_note = "no samples were measured"
        return self.ok(
            f"{stats.n} reward samples {bound_note}; {spread_note}{declared_note}",
            cov,
            evidence={
                "std": stats.std,
                "min": stats.min,
                "max": stats.max,
                "origin": c.origin,
            },
        )

    def controls(self) -> list[Control]:
        return [
            Control(
                "all-identical-rewards",
                ControlKind.MUST_FIRE,
                _identical_rewards_ctx,
                note="472 samples, mean 0.794, spread exactly 0 — the incident batches",
            ),
            Control(
                "out-of-bounds-rewards",
                ControlKind.MUST_FIRE,
                _out_of_bounds_rewards_ctx,
                note="rewards reach 31.0 against declared bounds of ±10",
            ),
            Control(
                "no-samples-on-reward-objective",
                ControlKind.MUST_FIRE,
                _missing_reward_stats_ctx,
                note="reward-bearing objective, zero samples examined — vacuous, not clean",
            ),
            Control(
                "zero-std-with-nonzero-range",
                ControlKind.MUST_FIRE,
                _zero_std_with_range_ctx,
                note="W-obj-8 verbatim: std 0.0 over [-1.0, 2.0] — impossible, once passed",
            ),
            Control(
                "negative-std",
                ControlKind.MUST_FIRE,
                _negative_std_ctx,
                note="std is a root-mean-square; -0.25 under it describes no batch",
            ),
            Control(
                "inverted-range",
                ControlKind.MUST_FIRE,
                _inverted_range_ctx,
                note="min=2.0 > max=-1.0: the range itself contradicts itself",
            ),
            Control(
                "mean-outside-range",
                ControlKind.MUST_FIRE,
                _mean_outside_range_ctx,
                note="mean 5.0 reported over samples bounded by [-1.0, 2.0]",
            ),
            Control(
                "std-above-analytic-ceiling",
                ControlKind.MUST_FIRE,
                _std_above_ceiling_ctx,
                note="std 3.0 vs ceiling ~1.512 for range 3.0 over n=64: fabrication",
            ),
            Control(
                "single-sample-with-range",
                ControlKind.MUST_FIRE,
                _single_sample_with_spread_ctx,
                note="one sample cannot span [-1.0, 2.0]; min must equal max at n=1",
            ),
            Control(
                "negative-sample-count",
                ControlKind.MUST_FIRE,
                _negative_sample_count_ctx,
                note="n=-3 finite stats and all: the summary describes no batch that ran",
            ),
            Control(
                "healthy-rewards",
                ControlKind.MUST_PASS,
                _healthy_objective_ctx,
                note="256 samples in bounds with genuine variance",
            ),
            Control(
                "stats-supplied-without-reward-term",
                ControlKind.MUST_PASS,
                _stats_without_reward_term_ctx,
                note="objective declares no reward term, yet 256 sane samples "
                "exist — examined and surfaced, never auto-failed: a "
                "divergence claim nobody observed would itself be a defect",
            ),
        ]


@register
class HyperparameterDriftGate(_ObjectiveGate):
    """Prevents the objective quietly becoming a different objective mid-run.

    In the audited estate the trust-region coefficient that should have bound the
    updates was effectively absent for hundreds of steps; the only corroboration was
    that the weights had moved about one bf16 ULP. Nothing compared the
    hyperparameters in force at step N against those the run declared at step 0 —
    and re-reading the config file at check time is not a fix, because the
    environment that selects parts of that config is exactly what the 24-run
    incident showed cannot be trusted twice. So this gate compares a canonical
    fingerprint of the *live* hyperparameters against the fingerprint recorded in
    the manifest at step 0. The fingerprint is the verdict; key-by-key diffing
    (when the step-0 snapshot is available) is evidence only.

    A missing step-0 fingerprint is FAIL, not SKIP: hyperparameters with nothing
    recorded to compare against are unfalsifiable, and unfalsifiable is the incident.
    Zero hyperparameters at all is VACUOUS for the usual reason.
    """

    id: ClassVar[str] = "objective.hparam_drift"
    description: ClassVar[str] = (
        "Objective hyperparameters at this step fingerprint-match the step-0 "
        "manifest record — compared against the record, never a re-read config"
    )
    events: ClassVar[tuple[Lifecycle, ...]] = (Lifecycle.STEP_ZERO, Lifecycle.SAVE)

    def check(self, ctx: Any) -> GateResult:
        c = _coerce(ctx)
        if not c.current_hparams:
            return self.ok(
                "objective exposes zero hyperparameters — nothing was compared",
                Coverage.none("hyperparameters"),
                evidence={"origin": c.origin},
            )
        expected = len(c.step0_hparams) if c.step0_hparams is not None else None
        cov = Coverage(len(c.current_hparams), "hyperparameters", expected=expected)

        if c.step0_fingerprint is None:
            return self.fail(
                "no step-0 hyperparameter fingerprint is recorded — the "
                "hyperparameters now in force are unfalsifiable, which is how the "
                "objective became unrecoverable from every artefact",
                cov,
                evidence={"origin": c.origin},
            )
        current_fp = fingerprint_hparams(c.current_hparams)
        if current_fp != c.step0_fingerprint:
            changed = (
                _changed_keys(c.step0_hparams, c.current_hparams)
                if c.step0_hparams is not None
                else None
            )
            suffix = f" (changed: {', '.join(changed)})" if changed else ""
            return self.fail(
                f"objective hyperparameters have drifted from the step-0 manifest "
                f"record{suffix} — what is being optimised now is not what the run "
                f"declared",
                cov,
                evidence={
                    "recorded_fingerprint": c.step0_fingerprint,
                    "current_fingerprint": current_fp,
                    "changed_keys": changed,
                    "origin": c.origin,
                },
            )
        return self.ok(
            f"fingerprint {current_fp[:12]}… matches the step-0 record over "
            f"{len(c.current_hparams)} hyperparameters",
            cov,
            evidence={"fingerprint": current_fp},
        )

    def controls(self) -> list[Control]:
        return [
            Control(
                "kl-coef-slid-to-zero",
                ControlKind.MUST_FIRE,
                _drifted_hparams_ctx,
                note="kl_coef 0.04 at step 0, 0.0 now — the trust region evaporated",
            ),
            Control(
                "no-step0-fingerprint",
                ControlKind.MUST_FIRE,
                _no_step0_fingerprint_ctx,
                note="nothing recorded at step 0: currently-in-force is unfalsifiable",
            ),
            Control(
                "no-hyperparameters-at-all",
                ControlKind.MUST_FIRE,
                _no_hparams_at_all_ctx,
                note="an empty hparam set compared against itself is all([]) again",
            ),
            Control(
                "matches-step0-record",
                ControlKind.MUST_PASS,
                _healthy_objective_ctx,
                note="live hyperparameters fingerprint-identical to the manifest",
            ),
        ]
