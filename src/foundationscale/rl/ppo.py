"""The PPO binding: one composite loss wiring the three landed PPO objectives.

``PPOCompositeLoss`` exists because ``Algorithm.setup`` takes exactly one
``loss_fn`` and :func:`check_algorithm_wiring` compares
``requirements.declared_components`` against ``loss_fn.declaration()``: the
three PPO objectives must arrive as ONE declared objective denominator or the
gate has three unconcatenated declarations it cannot grade. The composite
re-implements none of their arithmetic.

THE VALUE-HEAD SEAM, stated once and in both halves. GRPO reads the current
policy log-probabilities out of a batch COLUMN named in config, which keeps
that module torch-free. PPO cannot do the same for value estimates, because
PPO's defining feature is the inner-epoch loop: the value estimate must be
recomputed with the CURRENT parameters on every inner epoch, so the CURRENT
value estimate is a CALLABLE role (a ``ValueHead``), not a static column. The
OLD value estimate is different: the PPO2 clipping reference is computed once
at rollout time and is fixed for the whole rollout by construction, so the
old values stay a batch column (``ValueFunctionLoss.old_value_column``). A
callable would be a lie for the old estimate and a column would be a lie for
the current one; this binding holds both halves of that split.

THE TEMPORAL ESTIMATOR'S PER-ROW INPUTS are likewise READ columns, never
synthesised: ``terminated`` marks trajectory ends and a fabricated flag --
even an innocent-looking all-True -- changes every bootstrapped advantage
while claiming to be measured, so a missing termination column is refused,
never defaulted.

THE INNER-EPOCH LOOP IS NOT HERE, deliberately: one ``step()`` is one
optimiser step over one batch, exactly as every other binding in this family
defines it; the epoch loop belongs to the trainer. KL controller state is
also not here: an ``AdaptiveKLController.update`` returns a NEW controller,
and the owner of that chain is the trainer, which closes the loop as
``controller = controller.update(kl_estimate)`` and constructs the next
KLPenaltyLoss -- never by mutating a frozen object and never by rebinding
this binding's measured supplied map behind its own back.

WHAT THIS MODULE CLAIMS: a wired PPO step prices exactly the supervised
tokens of a supplied batch through the three landed objectives, re-estimates
values through the role on every call, reads the estimator's prompt-id and
termination columns from the batch under configured names, and reports the
temporal estimator's compacted ``used`` count as the step denominator.

WHAT THIS MODULE DOES NOT CLAIM: that the inner-epoch loop, the optimiser,
any model forward, reference-model execution, rollout production, or KL
controller threading exists here; that the value head's estimates are
verified anywhere but ``PPOAlgorithm.step`` (a bare ``__call__`` estimates
unverified); that any termination flag is inferred or defaulted -- it is
read or refused; or that PPO is complete without the trainer-owned epoch
loop.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from foundationscale.rl.advantage import (
    AdvantageFn,
    AdvantageResult,
    LearnedValueAdvantageEstimation,
)
from foundationscale.rl.algorithm import (
    AlgorithmRequirements,
    AlgorithmSemantics,
    AlgorithmWiringRefusal,
    StepReport,
    StepReportRefusal,
    check_algorithm_wiring,
    check_role_map,
)
from foundationscale.rl.interfaces import (
    BatchRefusal,
    ExperienceBatch,
    ForwardFn,
    LossConfigRefusal,
    LossDeclaration,
    LossOutput,
)
from foundationscale.rl.policy import PolicyPair
from foundationscale.rl.ppo_objectives import (
    KLPenaltyLoss,
    PPOClippedPolicyLoss,
    ValueFunctionLoss,
)
from foundationscale.rl.rollout import RolloutSource
from foundationscale.rl.value_head import (
    ValueHead,
    check_value_capabilities,
    verify_estimates,
)
from foundationscale.rl.weightsync import WeightSync

__all__ = (
    "PPOAlgorithm",
    "PPOCompositeLoss",
    "check_ppo_requirements",
)


def _checked_name(field_name: str, value: Any) -> None:
    if not isinstance(value, str) or not value:
        raise LossConfigRefusal(
            f"{field_name}={value!r}: column and component names must be "
            f"non-empty strings; absence of a name is not a name"
        )


@dataclass(frozen=True, slots=True)
class PPOCompositeLoss:
    """One ``LossFn`` composing the clipped policy, value, and KL objectives.

    ``Algorithm.setup`` accepts exactly one ``loss_fn`` and the wiring gate
    grades ``requirements.declared_components`` against
    ``loss_fn.declaration()``, so the three landed PPO objectives are wired
    here as ONE loss. ``kl_loss=None`` means the KL term is ABSENT -- no
    component, no metric, no reference role -- and is NOT a zero-weighted
    KL: a zero coefficient is refused inside ``KLPenaltyLoss`` itself, so
    the only honest representation of "no KL" is no KL object.

    The temporal estimator's per-row inputs are batch columns named here by
    config: ``prompt_id_column`` carries the prompt ids,
    ``terminated_column`` carries the per-row trajectory-end flags the
    bootstrapped estimate requires, and ``reward_column`` carries the
    per-row rewards. ``terminated`` is READ, never invented: a fabricated
    termination flag changes every bootstrapped advantage, so a missing
    termination column is refused, never defaulted to all-True or
    all-False. ``value_head`` and ``advantage_fn`` are required keyword
    roles: the annotation states the contract and the construction refusal
    enforces it against callers the typechecker cannot see.

    Weights come from the sub-losses' own declared component weights. The
    composite adds NO second weighting layer: the reported scalar is the sum
    of the sub-losses' own contributions, and one countable stated twice is
    refused here at construction -- two sub-losses contributing the same
    component name would make the gate's denominator ambiguous, and are
    refused with both contributors named.

    WHAT IS CLAIMED: ``declaration()`` is exactly the concatenation of the
    sub-declarations' components and of their ``MetricExpectation``s in
    (policy, value, kl) order, ``__call__`` returns one ``LossOutput`` whose
    ``loss`` is the sum of the sub-contributions and whose components and
    metrics are the union of the sub-outputs, and ``compute_with_report``
    retains the temporal estimator's ``AdvantageResult`` using the SAME
    measured value estimates the value loss priced and the batch's own
    prompt-id and termination columns.

    WHAT IS NOT CLAIMED: that a bare ``__call__``'s estimates were verified
    -- verification against the batch is ``verify_estimates`` and lives in
    ``PPOAlgorithm.step``; that the estimator's output and the batch's
    advantage/return columns were CROSS-CHECKED token by token -- the
    producer declares those columns came from this estimator and the
    binding anchors only the offered/used denominators; that any
    termination flag was inferred rather than read -- it was not; or that
    the inner-epoch loop or the KL controller chain is threaded here --
    both are trainer-owned.
    """

    policy_loss: PPOClippedPolicyLoss = field(default_factory=PPOClippedPolicyLoss)
    value_loss: ValueFunctionLoss = field(default_factory=ValueFunctionLoss)
    kl_loss: KLPenaltyLoss | None = None
    value_head: ValueHead = field(kw_only=True)
    advantage_fn: LearnedValueAdvantageEstimation = field(kw_only=True)
    reward_column: str = "rewards"
    prompt_id_column: str = "prompt_ids"
    terminated_column: str = "terminated"

    def __post_init__(self) -> None:
        if not isinstance(self.policy_loss, PPOClippedPolicyLoss):
            raise LossConfigRefusal(
                f"policy_loss={self.policy_loss!r}: 1 of 1 policy slots "
                f"must carry PPOClippedPolicyLoss, not "
                f"{type(self.policy_loss).__name__}"
            )
        if not isinstance(self.value_loss, ValueFunctionLoss):
            raise LossConfigRefusal(
                f"value_loss={self.value_loss!r}: 1 of 1 value slots "
                f"must carry ValueFunctionLoss, not "
                f"{type(self.value_loss).__name__}"
            )
        if self.kl_loss is not None and not isinstance(self.kl_loss, KLPenaltyLoss):
            raise LossConfigRefusal(
                f"kl_loss={self.kl_loss!r}: None (the KL term is absent) "
                f"or a KLPenaltyLoss; {type(self.kl_loss).__name__} is not "
                f"an absent term -- absence is None, never a substitute"
            )
        if self.value_head is None:
            raise LossConfigRefusal(
                "field value_head=None: 1 of 1 required inputs absent for "
                "ppo; the current-parameter value estimate is a callable "
                "role, not an absent column"
            )
        if not isinstance(self.value_head, ValueHead):
            raise LossConfigRefusal(
                f"field value_head={self.value_head!r}: 1 of 1 value "
                f"slots must carry a ValueHead (estimate plus "
                f"capabilities), not {type(self.value_head).__name__}"
            )
        if self.advantage_fn is None:
            raise LossConfigRefusal(
                "field advantage_fn=None: 1 of 1 required inputs absent "
                "for ppo advantage estimation; the temporal estimator is a "
                "declared role, not an optional extra"
            )
        if not isinstance(self.advantage_fn, LearnedValueAdvantageEstimation):
            raise LossConfigRefusal(
                f"advantage_fn={self.advantage_fn!r}: PPO binds "
                f"LearnedValueAdvantageEstimation; 1 of 1 estimator slots "
                f"must carry that concrete temporal estimator, not an "
                f"undeclared substitute"
            )
        # One loop over names, not three separate calls: a column-name field
        # added later must join this denial surface, not bypass it.
        for field_name, name_value in (
            ("reward_column", self.reward_column),
            ("prompt_id_column", self.prompt_id_column),
            ("terminated_column", self.terminated_column),
        ):
            _checked_name(field_name, name_value)
        contributors: tuple[tuple[str, Any], ...] = (
            ("policy_loss", self.policy_loss),
            ("value_loss", self.value_loss),
            *(() if self.kl_loss is None else (("kl_loss", self.kl_loss),)),
        )
        seen: dict[str, str] = {}
        for slot, sub_loss in contributors:
            for name in sub_loss.declaration().components:
                if name in seen:
                    raise LossConfigRefusal(
                        f"component {name!r} is contributed by both "
                        f"{seen[name]!r} and {slot!r}: 2 of "
                        f"{len(contributors)} sub-losses declare 1 "
                        f"component name, which would make the composite "
                        f"coverage denominator ambiguous"
                    )
                seen[name] = slot

    def _sub_losses(self) -> tuple[Any, ...]:
        kl: tuple[Any, ...] = () if self.kl_loss is None else (self.kl_loss,)
        return (self.policy_loss, self.value_loss, *kl)

    def declaration(self) -> LossDeclaration:
        """Declare the concatenated components and metrics of the sub-losses.

        WHAT IS CLAIMED: the component tuple is the concatenation in
        (policy, value, kl) order of each sub-declaration's components, and
        the metric tuple is the same concatenation of their
        ``MetricExpectation``s; an absent KL term declares nothing.

        WHAT IS NOT CLAIMED: that metric NAMES are unique across sub-losses
        -- construction refuses duplicate components only, and a duplicated
        declared metric is refused downstream by the wiring gate; or that
        this declaration is suitable input to any gate context -- it is the
        run-manifest statement, per the ``LossDeclaration`` contract.
        """
        declared = tuple(sub.declaration() for sub in self._sub_losses())
        components = tuple(name for sub in declared for name in sub.components)
        metrics = tuple(metric for sub in declared for metric in sub.metrics)
        return LossDeclaration(components=components, metrics=metrics)

    def semantics(self) -> AlgorithmSemantics:
        """Return this composite's independently declared PPO semantics.

        WHAT IS CLAIMED: the declaration states token ratio scope, the k3
        estimator when -- and only when -- a KL term is wired, the policy
        clip interval resolved from ``clip_epsilon`` and its overrides, and
        ``reference_free=True`` exactly when the KL term is absent.

        WHAT IS NOT CLAIMED: that an independently constructed algorithm
        agrees; agreement is measured by ``PPOAlgorithm.setup``.
        """
        low_width = float(
            self.policy_loss.clip_epsilon
            if self.policy_loss.clip_epsilon_low is None
            else self.policy_loss.clip_epsilon_low
        )
        high_width = float(
            self.policy_loss.clip_epsilon
            if self.policy_loss.clip_epsilon_high is None
            else self.policy_loss.clip_epsilon_high
        )
        return AlgorithmSemantics(
            group_size=None,
            ratio_scope="token",
            kl_estimator=None if self.kl_loss is None else "k3",
            clip_bounds=(1.0 - low_width, 1.0 + high_width),
            reference_free=self.kl_loss is None,
        )

    def __call__(self, forward_fn: ForwardFn, batch: ExperienceBatch) -> LossOutput:
        """Price one batch through the existing ``LossFn`` surface.

        WHAT IS CLAIMED: the result is the first value returned by
        ``compute_with_report`` and therefore carries exactly the declared
        objective components.

        WHAT IS NOT CLAIMED: that the estimates used for the value
        regression were verified against the batch -- a bare call estimates
        fresh but unverified; callers needing verification and the
        estimator's denominator should go through ``PPOAlgorithm.step``.
        """
        output, _advantage = self.compute_with_report(forward_fn, batch)
        return output

    def compute_with_report(
        self,
        forward_fn: ForwardFn,
        batch: ExperienceBatch,
        *,
        values: Sequence[Sequence[float]] | None = None,
    ) -> tuple[LossOutput, AdvantageResult]:
        """Compute the composite loss and retain its estimator denominator.

        ``values`` are the current-parameter value estimates. When omitted,
        they are measured here through the wired value head; when supplied --
        as ``PPOAlgorithm.step`` does after ``verify_estimates`` -- the SAME
        reading feeds both the value regression and the temporal estimator:
        one measurement, two consumers, never two divergent estimations of
        the same countable.

        WHAT IS CLAIMED: the returned ``AdvantageResult`` names the exact
        offered rows represented on the estimator's side, the estimator was
        handed this module's prompt-id, reward, termination, and mask
        columns plus ``values`` -- the termination flags come from the
        batch, because an invented flag would change every bootstrapped
        advantage -- and the loss is the unweighted-by-the-composite sum of
        the sub-losses' own weighted contributions.

        WHAT IS NOT CLAIMED: that the batch's advantage and return columns
        agree token by token with this estimator's output -- the producer
        states that pairing and this binding anchors only the offered
        denominator; or that supplied ``values`` were verified -- this
        method prices what it was handed.
        """
        required_columns = (
            self.reward_column,
            self.prompt_id_column,
            self.terminated_column,
        )
        missing = tuple(name for name in required_columns if name not in batch.columns)
        if missing:
            raise BatchRefusal(
                f"field columns: {len(missing)} of {len(required_columns)} "
                f"required inputs absent for ppo advantage estimation: "
                f"{{{', '.join(missing)}}}"
            )
        observed_values = values if values is not None else self.value_head.estimate(batch)
        advantage = self.advantage_fn.compute(
            prompt_ids=batch.column(self.prompt_id_column),
            rewards=batch.column(self.reward_column),
            mask=batch.column(self.policy_loss.mask_column),
            values=observed_values,
            terminated=batch.column(self.terminated_column),
        )
        batch_rows = len(batch)
        if advantage.offered != batch_rows:
            raise BatchRefusal(
                f"AdvantageResult.offered={advantage.offered} but the "
                f"batch contains {batch_rows} rows; {advantage.offered} "
                f"of {batch_rows} offered rows cannot anchor a PPO step"
            )

        def value_forward(_batch: ExperienceBatch) -> Sequence[Sequence[float]]:
            return observed_values

        outputs = (
            self.policy_loss(forward_fn, batch),
            self.value_loss(value_forward, batch),
            *(() if self.kl_loss is None else (self.kl_loss(forward_fn, batch),)),
        )
        combined = LossOutput(
            loss=sum(output.loss for output in outputs),
            components=tuple(component for output in outputs for component in output.components),
            metrics=tuple(metric for output in outputs for metric in output.metrics),
        )
        return combined, advantage


def check_ppo_requirements(
    *,
    requires: Mapping[str, bool],
    supplied: Mapping[str, Any],
    origin: str = "ppo",
) -> tuple[str, ...]:
    """Check PPO's role map over the union of both mapping key sets.

    WHAT IS CLAIMED on success: every required role names a supplied object,
    no required set was emptily satisfied, no supplied required-role value is
    ``None``, and the returned tuple is the sorted set of roles actually
    consumed.

    WHAT IS NOT CLAIMED: that any role object is a valid model, dataloader,
    estimator, value head, or loss. Type and semantic checks remain with the
    component handshakes; this helper checks the declared denominator only.
    The implementation is shared with every other family's role-map check --
    only the ``origin`` this binding names differs.
    """
    return check_role_map(requires=requires, supplied=supplied, origin=origin)


@dataclass(frozen=True, slots=True)
class _Wiring:
    loss_fn: PPOCompositeLoss
    dataloader: Iterator[ExperienceBatch]
    policy_logprob_column: str


class PPOAlgorithm:
    """Concrete batch-step PPO algorithm bound to the landed objectives.

    This binding consumes completed ``ExperienceBatch`` objects from its
    mandatory dataloader; rollout production is upstream of the step. It
    therefore declares no rollout-source or weight-sync object. It does
    require a temporal advantage function and a value head because the
    implemented objective contains learned-baseline advantage estimation
    and a current-parameter value regression, and it requires a reference
    policy exactly when a KL term is wired.

    THE INNER-EPOCH LOOP IS NOT IMPLEMENTED HERE and that is deliberate:
    one ``step()`` is one optimiser step over one batch, exactly as every
    other binding in this family defines it; the epoch loop belongs to the
    trainer. PPO without that loop is still PPO's objective surface, but a
    run reading this binding alone has not run PPO's defining repetition.

    WHAT IS CLAIMED: one successful step calls the wired
    :class:`PPOCompositeLoss` on values the wired value head measured and
    this module verified against the batch, keeps the returned
    ``AdvantageResult`` denominator, and returns a ``StepReport`` whose
    ``rows`` is the estimator's compacted ``used`` count.

    WHAT IS NOT CLAIMED: that the inner-epoch loop exists here (it does
    not), that the KL controller chain is threaded here (the trainer owns
    it; this binding carries no controller state and rebinds nothing), that
    the dataloader rows are fresh or on-policy, or that any forward
    implementation is invoked by this torch-free object.
    """

    __slots__ = (
        "_next_step",
        "_requirements",
        "_requires",
        "_semantics",
        "_supplied",
        "_wiring",
    )

    def __init__(
        self,
        *,
        clip_epsilon: float = 0.2,
        clip_epsilon_low: float | None = None,
        clip_epsilon_high: float | None = None,
        value_clip_epsilon: float | None = None,
        with_kl: bool = True,
    ) -> None:
        """Construct PPO's declarations independently of any supplied loss.

        ``with_kl`` is this algorithm's own statement that a reference-KL
        term is part of the objective; it fixes the ``reference_policy``
        role and the declared k3 estimator BEFORE any loss is seen, and a
        later composite that disagrees is REFUSED in the semantics
        handshake, never silently adjusted.

        WHAT IS CLAIMED: invalid clip configuration is refused at
        construction, and the resulting declarations name the configured
        policy clip interval, value-clip metric stance, and KL stance.

        WHAT IS NOT CLAIMED: that a later loss agrees; the setup handshake
        compares this declaration against the loss's own declaration field
        by field.
        """
        # The temporary default-column objectives perform all shared
        # construction validation without becoming the run's losses. Setup
        # still requires a composite supplied by the caller.
        PPOClippedPolicyLoss(
            clip_epsilon=clip_epsilon,
            clip_epsilon_low=clip_epsilon_low,
            clip_epsilon_high=clip_epsilon_high,
        )
        ValueFunctionLoss(clip_epsilon=value_clip_epsilon)
        low_width = float(clip_epsilon if clip_epsilon_low is None else clip_epsilon_low)
        high_width = float(clip_epsilon if clip_epsilon_high is None else clip_epsilon_high)
        self._semantics = AlgorithmSemantics(
            group_size=None,
            ratio_scope="token",
            kl_estimator="k3" if with_kl else None,
            clip_bounds=(1.0 - low_width, 1.0 + high_width),
            reference_free=not with_kl,
        )
        self._requirements = AlgorithmRequirements(
            name="ppo",
            # Every role this surface grades is named, including the two it
            # does NOT consume: a role declared with False is graded, a role
            # left unnamed is invisible to the check.
            requires={
                "rollout_source": False,
                "advantage_fn": True,
                "weight_sync": False,
                "reference_policy": with_kl,
                "value_head": True,
            },
            declared_components=(
                ("ppo_policy_loss", "value_loss") + (("kl_penalty",) if with_kl else ())
            ),
            declared_metrics=(
                ("clip_fraction",)
                + (("value_clip_fraction",) if value_clip_epsilon is not None else ())
                + (("kl_estimate",) if with_kl else ())
            ),
            semantics=self._semantics,
        )
        self._requires: Mapping[str, bool] = MappingProxyType(
            {
                "policy_pair": True,
                "loss_fn": True,
                "dataloader": True,
                "advantage_fn": True,
                "value_head": True,
                "reference_policy": with_kl,
                "rollout_source": False,
                "weight_sync": False,
            }
        )
        self._supplied: Mapping[str, Any] | None = None
        self._wiring: _Wiring | None = None
        self._next_step = 0

    def requirements(self) -> AlgorithmRequirements:
        """Return the role and objective declaration measured by gates.

        WHAT IS CLAIMED: the record is this binding's declaration, including
        the composite components, the diagnostic metrics those components
        emit, and the requirement for a value head and -- only under a
        wired KL term -- a reference policy.

        WHAT IS NOT CLAIMED: that the declaration is sufficient evidence the
        mathematics are correct; it is the denominator the gates measure.
        """
        return self._requirements

    def semantics(self) -> AlgorithmSemantics:
        """Return PPO's algorithm-side semantics declaration.

        WHAT IS CLAIMED: the returned values came from this algorithm's
        constructor, independent of any later loss object.

        WHAT IS NOT CLAIMED: that algorithm and loss currently agree;
        agreement is refused or established by ``setup``.
        """
        return self._semantics

    def requires(self) -> Mapping[str, bool]:
        """Return the static role-requirement denominator.

        WHAT IS CLAIMED: every entry is an immutable string-to-bool
        declaration, and the mapping contains at least one required role.

        WHAT IS NOT CLAIMED: that required roles are currently present;
        presence is represented only by ``supplied()`` after setup.
        """
        return self._requires

    def supplied(self) -> Mapping[str, Any] | None:
        """Return the actual post-setup wiring, or abstain before setup.

        WHAT IS CLAIMED: after successful setup, the immutable mapping names
        exactly the objects handed to this algorithm during that setup.

        WHAT IS NOT CLAIMED before setup: any wiring measurement. The
        pre-setup value is ``None`` because the supplied set was not
        measured; an empty map would falsely read as a completed check.
        """
        return self._supplied

    def setup(
        self,
        *,
        policy_pair: PolicyPair,
        loss_fn: Any,
        dataloader: Iterable[ExperienceBatch],
        config: Mapping[str, Any],
        rollout_source: RolloutSource | None = None,
        # #359/#395: declared at least as WIDE as the protocol's, never
        # narrower. A narrower parameter type on an implementer is a
        # contravariance violation -- it makes PPOAlgorithm unsubstitutable
        # for Algorithm, which is what kept it out of the registry (whose
        # value type is Callable[[], Algorithm]) while it was otherwise
        # shipped and working.
        #
        # The union is not decoration. MEASURED (#396), and the scope word
        # matters: the two arms are disjoint STATICALLY, not at runtime.
        # LearnedValueAdvantageEstimation.compute requires two further
        # keyword-only arguments (`values`, `terminated`) that
        # AdvantageFn.compute does not declare, so mypy refuses the
        # substitution and names the conflicting member. At RUNTIME the same
        # pair is NOT disjoint: `@runtime_checkable` checks method PRESENCE
        # only and never signatures, so `isinstance(lv, AdvantageFn)` returns
        # True and the mismatch surfaces later as a TypeError at the call
        # rather than as a refusal at the seam. Annotating this slot as
        # AdvantageFn alone would therefore be false in the OTHER direction:
        # it would name a width PPO can never accept.
        # Stating both arms keeps the parameter a supertype of the
        # protocol's (substitutable) while remaining true about what PPO
        # can be handed.
        #
        # The narrowing PPO genuinely needs is not lost: it is RE-STATED
        # BELOW as a runtime refusal, which is the stronger of the two
        # claims. A static annotation is a promise about callers; an
        # isinstance refusal is a measurement of the object actually handed
        # over, and only the second survives a dynamic wiring path.
        #
        # That refusal is written against the CONCRETE class, not against a
        # Protocol, and #396 is why. An isinstance against AdvantageFn would
        # admit anything carrying a method named `compute` -- including this
        # very class, whose signature does not fit -- so the Protocol arm
        # cannot carry a runtime gate. Where a seam must actually REFUSE, it
        # names the class.
        advantage_fn: AdvantageFn | LearnedValueAdvantageEstimation | None = None,
        value_head: ValueHead | None = None,
        weight_sync: WeightSync | None = None,
    ) -> None:
        """Wire PPO and perform its declaration-to-loss handshake.

        ``config`` must contain ``policy_logprob_column`` naming the batch
        column of current-policy token log-probabilities, and -- when the
        wired value loss clips -- ``old_value_column`` naming the static
        PPO2 reference column; the config statement must agree with the
        column the wired value loss reads, because one seam stated two ways
        is refused, never arbitrated.

        WHAT IS CLAIMED on success: the concrete loss is
        ``PPOCompositeLoss``, its independently computed semantics equal
        this algorithm's, the supplied estimator and value head are the
        exact objects owned by the loss, the value head attests token
        granularity and the old-value reporting the wired value loss needs,
        the existing wiring handshake accepted the policy/reference roles,
        and the two independent role denominators name the same consumed
        set.

        WHAT IS NOT CLAIMED: that any batch is available, on-policy, or
        finite; that the estimates the head will produce are verified until
        ``step`` runs ``verify_estimates``; or that the inner-epoch loop
        exists -- it does not, here.
        """
        if self._wiring is not None:
            raise AlgorithmWiringRefusal(
                "field setup: 1 of 1 PPO algorithm objects was already "
                "wired; calling setup twice would silently replace the "
                "measured supplied mapping"
            )
        if not isinstance(loss_fn, PPOCompositeLoss):
            raise AlgorithmWiringRefusal(
                f"field loss_fn={loss_fn!r}: PPO requires "
                f"PPOCompositeLoss; 1 of 1 loss slots carries "
                f"{type(loss_fn).__name__}"
            )
        if advantage_fn is None:
            raise AlgorithmWiringRefusal(
                "field advantage_fn: 1 of 1 required inputs absent for ppo: {'advantage_fn'}"
            )
        if not isinstance(advantage_fn, LearnedValueAdvantageEstimation):
            raise AlgorithmWiringRefusal(
                f"field advantage_fn={advantage_fn!r}: PPO binds "
                f"learned-baseline advantage estimation, so 1 of 1 "
                f"advantage slots must carry a "
                f"LearnedValueAdvantageEstimation; it carries "
                f"{type(advantage_fn).__name__}. The parameter is annotated "
                f"at the Algorithm protocol's width so this binding stays "
                f"substitutable; the width is narrowed HERE, by measurement"
            )
        if advantage_fn is not loss_fn.advantage_fn:
            raise AlgorithmWiringRefusal(
                "field advantage_fn: the setup object and the loss-owned "
                "estimator are 2 distinct objects for 1 declared role; "
                "the concrete PPO binding must wire the same "
                "LearnedValueAdvantageEstimation instance through both sides"
            )
        if value_head is None:
            raise AlgorithmWiringRefusal(
                "field value_head: 1 of 1 required inputs absent for ppo: {'value_head'}"
            )
        if value_head is not loss_fn.value_head:
            raise AlgorithmWiringRefusal(
                "field value_head: the setup object and the loss-owned "
                "value head are 2 distinct objects for 1 declared role; "
                "the concrete PPO binding must wire the same ValueHead "
                "instance through both sides"
            )
        needs_old_values = loss_fn.value_loss.clip_epsilon is not None
        check_value_capabilities(
            required_granularity="token",
            needs_old_values=needs_old_values,
            capabilities=value_head.capabilities(),
            origin="PPOAlgorithm.setup",
        )
        loss_semantics = loss_fn.semantics()
        for field_name in (
            "group_size",
            "ratio_scope",
            "kl_estimator",
            "clip_bounds",
            "reference_free",
        ):
            algorithm_value = getattr(self._semantics, field_name)
            loss_value = getattr(loss_semantics, field_name)
            if algorithm_value != loss_value:
                raise AlgorithmWiringRefusal(
                    f"field {field_name}: algorithm declares "
                    f"{algorithm_value!r} but loss declares {loss_value!r}; "
                    f"1 of 5 semantics fields disagrees between the 2 "
                    f"declarations"
                )
        consumed = check_algorithm_wiring(
            self._requirements,
            policy_pair=policy_pair,
            loss_fn=loss_fn,
            # reference_policy is deliberately absent from the supplied
            # mapping: the pair attests it, and naming it here would be
            # one countable declared in two objects (the surface refuses).
            supplied={
                "rollout_source": rollout_source,
                "advantage_fn": advantage_fn,
                "value_head": value_head,
                "weight_sync": weight_sync,
            },
            origin="PPOAlgorithm.setup",
        )
        try:
            dataloader_iterator = iter(dataloader)
        except TypeError as exc:
            raise AlgorithmWiringRefusal(
                f"field dataloader={dataloader!r}: 1 of 1 mandatory "
                f"step inputs was not iterable "
                f"({type(dataloader).__name__})"
            ) from exc
        if not isinstance(config, Mapping):
            raise AlgorithmWiringRefusal(
                f"field config={config!r}: 1 of 1 setup configurations must implement Mapping"
            )
        column = config.get("policy_logprob_column")
        if not isinstance(column, str) or not column:
            raise AlgorithmWiringRefusal(
                "field policy_logprob_column: 0 of 1 required config "
                "values names a non-empty batch column for ppo; current "
                "token log-probabilities cannot be attributed to the "
                "existing LossFn surface without that column"
            )
        if needs_old_values:
            old_column = config.get("old_value_column")
            if not isinstance(old_column, str) or not old_column:
                raise AlgorithmWiringRefusal(
                    "field old_value_column: 0 of 1 required config "
                    "values names a non-empty batch column for ppo; the "
                    "PPO2 value-clipping reference is a static column and "
                    "cannot be attributed to a batch without that name"
                )
            if old_column != loss_fn.value_loss.old_value_column:
                raise AlgorithmWiringRefusal(
                    f"field old_value_column: config names {old_column!r} "
                    f"but the wired value loss reads "
                    f"{loss_fn.value_loss.old_value_column!r}; 1 old-value "
                    f"seam was stated 2 ways and PPO refuses to arbitrate "
                    f"between them"
                )
        supply: dict[str, Any] = {
            "policy_pair": policy_pair,
            "loss_fn": loss_fn,
            "dataloader": dataloader,
            "advantage_fn": advantage_fn,
            "value_head": value_head,
        }
        if self._requires["reference_policy"]:
            supply["reference_policy"] = policy_pair.references
        supplied = MappingProxyType(supply)
        mapped_consumed = check_ppo_requirements(
            requires=self._requires,
            supplied=supplied,
            origin="ppo",
        )
        handshake_roles = tuple(sorted(consumed + ("policy_pair", "loss_fn", "dataloader")))
        # This comparison is a MAINTENANCE tripwire, not a runtime one, and it
        # is marked as such rather than left to read like a covered branch. The
        # two denominators are literal mappings written side by side in
        # __init__ (``_requirements.requires`` and ``_requires``), and both
        # supplied maps are built here from the same validated arguments, so no
        # construction the public API permits can make them disagree. What it
        # catches is an EDIT that changes one mirror and not the other -- the
        # #222 shape, one countable declared in two objects -- and that failure
        # arrives through the source, not through a caller. Reaching it from a
        # test would mean assigning the private mirror directly, which proves
        # the assignment worked and nothing about the guard.
        if mapped_consumed != handshake_roles:  # pragma: no cover -- mirrors agree by construction
            raise AlgorithmWiringRefusal(
                f"mapped roles {mapped_consumed!r} disagree with the "
                f"existing handshake roles {consumed!r} plus the 3 "
                f"mandatory setup roles; 2 independently checked role "
                f"denominators must name the same consumed set"
            )
        self._supplied = supplied
        self._wiring = _Wiring(
            loss_fn=loss_fn,
            dataloader=dataloader_iterator,
            policy_logprob_column=column,
        )

    def step(self) -> StepReport:
        """Price the next batch and return its honest estimator denominator.

        WHAT IS CLAIMED: the current-policy log-probabilities come from the
        configured batch column, the current value estimates come from the
        wired head and were verified against the batch before pricing, the
        SAME estimates fed the value regression and the temporal estimator,
        and ``rows`` is the estimator's compacted ``used`` count rather than
        the offered batch length.

        WHAT IS NOT CLAIMED: that this step was repeated under the
        inner-epoch loop -- one step is one optimiser step over one batch
        and the loop is the trainer's; that the KL controller was updated --
        this binding holds no controller state and freezes the coefficient
        in force for the step; that another batch exists after this one; or
        that the result is finite.
        """
        wiring = self._wiring
        if wiring is None:
            required_count = sum(1 for needed in self._requires.values() if needed)
            raise AlgorithmWiringRefusal(
                f"field supplied: 0 of {required_count} required inputs "
                f"are present for ppo because setup has not completed"
            )
        try:
            batch = next(wiring.dataloader)
        except StopIteration as exc:
            raise StepReportRefusal(
                "0 of at least 1 required batches remain for ppo; the "
                "step is unmeasured rather than a step with zero rows"
            ) from exc
        estimates = wiring.loss_fn.value_head.estimate(batch)
        verify_estimates(
            estimates,
            batch,
            mask_column=wiring.loss_fn.value_loss.mask_column,
            origin="PPOAlgorithm.step",
        )
        column = wiring.policy_logprob_column

        def forward_fn(
            current_batch: ExperienceBatch,
        ) -> Sequence[Sequence[float]]:
            return current_batch.column(column)

        output, advantage = wiring.loss_fn.compute_with_report(
            forward_fn,
            batch,
            values=estimates,
        )
        report = StepReport(
            step=self._next_step,
            loss=output,
            rows=advantage.used,
            reward_stats=advantage.rewards,
            sync=None,
        )
        self._next_step += 1
        return report
