"""Differentiable tensor plane for the six preference oracles.

This module is the preference family's counterpart to
``torch_backend.TensorPolicyLoss``. The group-relative objectives decompose
into shared ratio, clipping and reduction axes; the six preference
objectives do not, because their differences are semantic families instead:
margin scaling and sides, length normalisation, SFT terms, odds and the KTO
reference point. Dispatch therefore keys one small recipe name on the exact
oracle class while reading every numerical knob from the objective instance.
That keeps the six torch-free implementations in ``losses.py`` and
``preference_objectives.py`` the only statements of their mathematics.

The public kernel accepts per-token target log-probabilities, never logits.
For the five paired objectives, chosen and rejected rows are interleaved as
``[chosen_0, rejected_0, chosen_1, rejected_1, ...]`` so the full input has
``R = 2B`` rows. KTO accepts one completion per row (``R = B``), one boolean
per row, and a caller-recorded finite non-negative ``kl_reference_point``.

torch is imported only under ``TYPE_CHECKING`` at module scope and inside
runtime functions, matching ``torch_backend.py``: importing this module on a
host without torch stays possible, while calling the kernel fails honestly at
the first function that needs the tensor runtime.

WHAT IS CLAIMED: for equivalent finite inputs, the returned scalar is a
float64 evaluation of the corresponding oracle loss to floating-point
tolerance; the returned tensor is 0-dimensional; and its gradient is routed
back to ``policy_logprobs``. Sequence sums and recipe arithmetic run in
float64 because the oracles convert every reading to a Python float.

WHAT IS NOT CLAIMED: that this kernel performs tokenisation, model
forwarding, batch loading, reference-model production, or trainer-side
record keeping; that masked positions carry gradients; or that this module
is torch-free. It is deliberately the tensor plane for the audited oracle
mathematics, not a second statement of that mathematics.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, NoReturn

if TYPE_CHECKING:  # pragma: no cover - typing only, never executed at runtime
    import torch

from foundationscale.rl.interfaces import BatchRefusal, SupervisionRefusal
from foundationscale.rl.losses import DPOLoss
from foundationscale.rl.preference_objectives import (
    CPOLoss,
    IPOLoss,
    KTOLoss,
    ORPOLoss,
    SimPOLoss,
)
from foundationscale.rl.torch_backend import (
    _require_2d,
    _require_finite,
    _require_same_shape,
)

__all__ = (
    "TensorPreferenceLoss",
    "is_paired",
    "needs_reference",
    "preference_metrics",
)


# Exact classes, not structural families: class identity selects the recipe,
# while every recipe reads its knobs from the supplied instance. A subclass
# could preserve type sufficiently to pass isinstance while changing the
# formula, so accepting it would quietly claim oracle parity for different
# mathematics.
_RECIPE: dict[type, str] = {
    DPOLoss: "logsig_margin",
    CPOLoss: "logsig_margin+sft",
    IPOLoss: "squared",
    SimPOLoss: "logsig_margin_lenorm",
    ORPOLoss: "odds+sft",
    KTOLoss: "kto",
}


def _recipe_name(objective: Any) -> str:
    """Select the named recipe, refusing an object outside the six oracles.

    WHAT IS CLAIMED: an accepted objective is exactly one of the six audited
    classes whose tensor forms this module contains.

    WHAT IS NOT CLAIMED: that attribute-compatible third-party objectives
    accidentally compute any oracle's formula.
    """
    recipe = _RECIPE.get(type(objective))
    if recipe is None:
        expected = ", ".join(objective_type.__name__ for objective_type in _RECIPE)
        raise BatchRefusal(
            f"objective={objective!r} ({type(objective).__name__}): 0 of "
            f"{len(_RECIPE)} supported preference oracle classes matched; "
            f"expected one of: {expected}. A structural duck is not a "
            f"parity contract, and its mathematics cannot be inferred"
        )
    return recipe


def needs_reference(objective: Any) -> bool:
    """Return whether the objective consumes reference log-probabilities.

    The answer is derived from the oracle's own ``reference_free``
    declaration, not from a restated table: DPO, IPO and KTO declare
    reference anchoring; SimPO, ORPO and CPO declare reference freedom.

    WHAT IS CLAIMED: the result is the inverse of the selected oracle's own
    ``reference_free`` property.

    WHAT IS NOT CLAIMED: that a caller has supplied the reference inputs;
    presence and shape are measured by the kernel.
    """
    _recipe_name(objective)
    return not bool(objective.reference_free)


def is_paired(objective: Any) -> bool:
    """Return whether the objective expects interleaved chosen/rejected rows.

    WHAT IS CLAIMED: this is True for DPO, CPO, IPO, SimPO and ORPO and
    False for KTO, as fixed by the explicit recipe table.

    WHAT IS NOT CLAIMED: that paired classification was inferred from column
    names, which need not exist at this tensor seam.
    """
    return _recipe_name(objective) != "kto"


def _required_tensor(name: str, value: Any) -> torch.Tensor:
    """Return ``value`` as a tensor, naming a wrong-plane input.

    WHAT IS CLAIMED: the returned object is a ``torch.Tensor``.

    WHAT IS NOT CLAIMED: anything about its shape, values or gradient.
    """
    import torch

    if not isinstance(value, torch.Tensor):
        raise BatchRefusal(
            f"{name} is {type(value).__name__}, not torch.Tensor; this "
            f"kernel consumes tensors so every mask entry and token "
            f"reading is attributable to one tensor position"
        )
    return value


def _checked_inputs(
    *,
    objective: Any,
    policy_logprobs: torch.Tensor,
    completion_mask: torch.Tensor,
    reference_logprobs: torch.Tensor | None,
    require_policy_grad: bool,
) -> tuple[str, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Measure the common shape, value, mask and reference contracts.

    WHAT IS CLAIMED on success: policy and mask are finite, equally shaped,
    two-dimensional tensors; mask entries are exactly 0 or 1; every input
    row has supervision; paired inputs have an even, non-zero row count;
    KTO inputs are non-zero row-count; and reference presence agrees with
    the objective's own ``reference_free`` declaration. Policy requires a
    gradient when and only when ``require_policy_grad`` is true.

    WHAT IS NOT CLAIMED: that recipe-specific tensors such as KTO's flag
    have been measured, that any reference tensor came from a reference
    model, or that policy readings came from any particular model.
    """
    import torch

    recipe = _recipe_name(objective)
    origin = type(objective).__name__
    policy = _required_tensor("policy_logprobs", policy_logprobs)
    raw_mask = _required_tensor("completion_mask", completion_mask)

    _require_2d("policy_logprobs", policy)
    _require_finite("policy_logprobs", policy)
    if require_policy_grad and not policy.requires_grad:
        raise BatchRefusal(
            f"policy_logprobs.requires_grad is False for {origin}; 1 of 1 "
            f"live inputs carries no gradient, and a loss computed from it "
            f"can never produce a weight update"
        )

    _require_2d("completion_mask", raw_mask)
    _require_finite("completion_mask", raw_mask)
    _require_same_shape("policy_logprobs", policy, "completion_mask", raw_mask)
    mask = raw_mask.detach().to(dtype=policy.dtype)

    if objective.reference_free:
        if reference_logprobs is not None:
            raise BatchRefusal(
                f"{origin} declares reference_free=True but 1 unexpected "
                f"reference_logprobs input was supplied; those readings "
                f"would be silently unused by a recipe that declares no "
                f"reference seam"
            )
        reference: torch.Tensor | None = None
    else:
        if reference_logprobs is None:
            raise BatchRefusal(
                f"{origin} declares reference_free=False, so 1 of 1 "
                f"required reference_logprobs inputs is absent; the "
                f"anchored margin cannot be attributed without it"
            )
        reference_input = _required_tensor("reference_logprobs", reference_logprobs)
        _require_2d("reference_logprobs", reference_input)
        _require_finite("reference_logprobs", reference_input)
        _require_same_shape("policy_logprobs", policy, "reference_logprobs", reference_input)
        reference = reference_input.detach().to(dtype=policy.dtype)

    row_count = int(policy.shape[0])
    if recipe == "kto":
        if row_count == 0:
            raise BatchRefusal(
                f"{origin} got 0 rows: a mean over zero completions is "
                f"unmeasurable, and returning 0.0 would report a perfect "
                f"KTO loss for an input containing no completions"
            )
    else:
        if row_count == 0:
            raise BatchRefusal(
                f"{origin} got 0 rows: a mean over zero preference pairs "
                f"is unmeasurable, and returning 0.0 would report a "
                f"perfect loss for an input containing no preferences"
            )
        if row_count % 2 != 0:
            raise BatchRefusal(
                f"policy_logprobs has {row_count} rows, but a paired "
                f"objective requires R == 2B with rows interleaved as "
                f"[c0, r0, c1, r1, ...]; 1 unmatched row cannot be "
                f"attributed to a preference pair"
            )

    bad_mask_count = int(((mask != 0.0) & (mask != 1.0)).sum().item())
    if bad_mask_count:
        raise BatchRefusal(
            f"{bad_mask_count} of {mask.numel()} completion_mask entries "
            f"are neither 0 nor 1; supervision mask entries must be "
            f"0 or 1, and a fractional entry cannot state which positions "
            f"carry gradient"
        )
    row_supervision = mask.sum(dim=-1)
    empty_rows = torch.nonzero((row_supervision == 0.0).reshape(-1), as_tuple=False).reshape(-1)
    if empty_rows.numel():
        first = int(empty_rows[0].item())
        if recipe == "kto":
            raise SupervisionRefusal(
                f"row {first}: supervision mask selected 0 supervised "
                f"tokens; a row that supervises nothing is unmeasurable, "
                f"and scoring it 0.0 would make an empty completion look "
                f"maximally likely"
            )
        pair_row = first // 2
        side = "chosen" if first % 2 == 0 else "rejected"
        raise SupervisionRefusal(
            f"row {pair_row}, side {side!r}: supervision mask selected "
            f"0 supervised tokens; a pair in which one completion "
            f"supervises nothing is unmeasurable, and scoring it 0.0 "
            f"would make an empty completion look maximally likely"
        )
    return recipe, policy, mask, reference


def _masked_sequence_sums(
    logprobs: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Return one masked log-probability sum per input row.

    The reduction is a SUM, exactly as the paired and unpaired float
    oracles define it. Both operands are upcast to float64 before the
    reduction because the oracle helpers convert each reading with Python's
    ``float()`` and perform Python's float64 arithmetic.
    """
    import torch

    return (logprobs.to(dtype=torch.float64) * mask.to(dtype=torch.float64)).sum(dim=-1)


def _paired_terms(
    *,
    policy: torch.Tensor,
    mask: torch.Tensor,
    reference: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
    """Shape interleaved rows into ``(B, 2)`` scores and supervised counts.

    WHAT IS CLAIMED: the first inner axis is chosen and the second rejected,
    by the fixed interleaving contract; sums and counts are float64.

    WHAT IS NOT CLAIMED: that the caller supplied complete rows in any pair;
    ``_checked_inputs`` has already refused every form of incompleteness
    visible at the tensor seam.
    """
    import torch

    pair_count = int(policy.shape[0]) // 2
    policy_scores = _masked_sequence_sums(policy, mask).reshape(pair_count, 2)
    reference_scores = (
        None if reference is None else _masked_sequence_sums(reference, mask).reshape(pair_count, 2)
    )
    supervised = mask.to(dtype=torch.float64).sum(dim=-1).reshape(pair_count, 2)
    return policy_scores, reference_scores, supervised


def _orpo_odds_margin_and_sft(
    policy_scores: torch.Tensor,
    supervised: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute ORPO's exact log-odds margin and chosen-side NLL.

    The arithmetic follows the oracle side by side: the side average is
    ``score / supervised``, its complement is ``-expm1(average)``, and its
    log-odds is ``average - log(one_minus_p)``. The exact float64 wall is
    checked on a CPU copy of one scalar per row side rather than clamped,
    because clamping would report a different objective precisely where the
    oracle refuses.
    """
    import torch

    averages = policy_scores.to(dtype=torch.float64) / supervised.to(dtype=torch.float64)
    one_minus_p = -torch.expm1(averages)
    one_minus_p_cpu = one_minus_p.detach().cpu()
    invalid = torch.nonzero((one_minus_p_cpu <= 0.0).reshape(-1), as_tuple=False).reshape(-1)
    if invalid.numel():
        flat_row = int(invalid[0].item())
        pair_row = flat_row // 2
        side = "chosen" if flat_row % 2 == 0 else "rejected"
        average = float(averages.detach().cpu().reshape(-1)[flat_row].item())
        raise BatchRefusal(
            f"row {pair_row}, side {side!r}: the length-normalised "
            f"log-probability {average!r} rounds to a completion "
            f"probability of exactly 1.0 in float64, so 1 - p is exactly "
            f"0.0 and the odds p / (1 - p) are undefined; ORPO is "
            f"undefined on this row, and the loss refuses rather than "
            f"emitting an infinite margin"
        )
    log_odds = averages - torch.log(one_minus_p)
    margin = log_odds[:, 0] - log_odds[:, 1]
    sft = -(policy_scores[:, 0].sum() / supervised[:, 0].sum())
    return margin, sft


def _checked_kto_inputs(
    *,
    kl_reference_point: float | None,
    desirable: torch.Tensor | None,
    row_count: int,
) -> tuple[float, torch.Tensor]:
    """Measure KTO's two recipe-specific inputs.

    WHAT IS CLAIMED: ``z0`` is a finite, non-negative recorded KL value and
    ``desirable`` is one boolean per completion.

    WHAT IS NOT CLAIMED: that ``z0`` was produced by an in-batch estimator;
    this family deliberately accepts only the recorded value.
    """
    import torch

    if kl_reference_point is None:
        raise BatchRefusal(
            "kl_reference_point is None; KTOLoss requires 1 recorded "
            "finite non-negative KL reference point. Inferring it from the "
            "current batch would make a row's loss depend on which other "
            "rows shared the batch"
        )
    try:
        z0 = float(kl_reference_point)
    except (TypeError, ValueError, OverflowError) as exc:
        raise BatchRefusal(
            f"kl_reference_point is {kl_reference_point!r}, which does "
            f"not convert to a scalar float; the reference point must be "
            f"one finite non-negative KL value"
        ) from exc
    if not math.isfinite(z0) or z0 < 0.0:
        raise BatchRefusal(
            f"kl_reference_point={kl_reference_point!r} is not a finite "
            f"non-negative value; a KL divergence is non-negative, so "
            f"it cannot be a measured reference point"
        )
    if desirable is None:
        raise BatchRefusal(
            f"desirable is None; 1 boolean label per KTO completion is "
            f"required, and {row_count} rows cannot be assigned the two "
            f"KTO row families without one"
        )
    flags = _required_tensor("desirable", desirable).detach()
    if flags.dtype != torch.bool:
        raise BatchRefusal(
            f"desirable has dtype {flags.dtype}, expected torch.bool; "
            f"the tensor interface requires one explicit boolean per row "
            f"rather than inferring a fractional label"
        )
    if flags.dim() != 1 or int(flags.shape[0]) != row_count:
        raise BatchRefusal(
            f"desirable has shape {tuple(flags.shape)} but the batch has "
            f"{row_count} rows; exactly one boolean flag per completion "
            f"is required"
        )
    return z0, flags


def _missing_reference_refusal(origin: str) -> NoReturn:
    # This helper is annotated NoReturn because it always raises: callers
    # place it under ``if reference is None:`` guards precisely so that the
    # common checks' failure to narrow an Optional becomes a live, typed
    # narrowing point, and an annotated ``-> None`` here would instead leave
    # every guarded reference typed ``Tensor | None`` downstream.
    # The branch itself is unreachable through the public checks, which
    # already refuse a missing reference for a reference-anchored objective.
    # It is kept fail-closed rather than asserted so an edited property
    # cannot convert an incoherent wiring into an AttributeError.
    raise BatchRefusal(  # pragma: no cover - caught by the common checks
        f"{origin} declared a reference seam but supplied no masked "
        f"reference scores; the tensor recipe cannot form its anchored "
        f"margin"
    )


@dataclass(frozen=True, slots=True)
class TensorPreferenceLoss:
    """One differentiable kernel for the six fixed preference recipes.

    The kernel forms masked sequence sums from caller-supplied target
    log-probabilities and then evaluates the selected oracle's exact
    formula:

    * DPO:
      ``weight * mean(softplus(-beta * ((pi_c-ref_c)-(pi_r-ref_r))))``,
      plus ``sft_weight * NLL(chosen)`` only when ``sft_weight`` is
      non-zero.
    * CPO:
      ``mean(softplus(-(beta*pi_c - beta*pi_r))) + lambda_ * NLL(chosen)``.
    * IPO:
      ``weight * mean((((pi_c-ref_c)-(pi_r-ref_r)) - 1/(2*tau))**2)``,
      with each side's anchored sequence sum divided by that side's
      supervised-token count when ``length_normalise`` is True (TRL's
      form; the default stays the paper's sums)..
    * SimPO:
      ``weight * mean(softplus(-((beta/count_c)*pi_c - (beta/count_r)*pi_r
      - gamma)))``.
    * ORPO:
      ``NLL(chosen) + lambda_ * mean(softplus(-(log_odds_c-log_odds_r)))``,
      with the oracle's exact float64 odds-refusal wall.
    * KTO: ``weight * mean(lambda_d * sigmoid(z0-r))`` on desirable rows
      and ``weight * mean(lambda_u * sigmoid(r-z0))`` on undesirable rows,
      where ``r = beta * (pi-ref)``.

    ``policy_logprobs`` must require gradient. The mask and reference are
    detached, as in ``TensorPolicyLoss``. Sequence arithmetic is performed
    in float64 to mirror the Python-float oracles; the dtype conversion is
    differentiable, so the statement remains that gradient lands on
    ``policy_logprobs``.

    WHAT IS CLAIMED: the same validation classes used by the oracle plane
    distinguish malformed batches (``BatchRefusal``) from absent
    supervision (``SupervisionRefusal``), and the returned tensor is finite,
    scalar and differentiable.

    WHAT IS NOT CLAIMED: that reference rows came from a frozen model; that
    KTO estimated its reference point; that any optional reference-free
    input is accepted and ignored; or that a non-finite overflow is clamped.
    """

    objective: Any

    def __call__(
        self,
        *,
        policy_logprobs: torch.Tensor,
        completion_mask: torch.Tensor,
        reference_logprobs: torch.Tensor | None = None,
        kl_reference_point: float | None = None,
        desirable: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Evaluate one complete preference batch as a differentiable scalar.

        WHAT IS CLAIMED: the row layout, reference presence and KTO-specific
        inputs are measured before any formula branch; every scale and SFT
        term is read from the objective instance; and the final scalar is
        finite and carries a gradient path to ``policy_logprobs``.

        WHAT IS NOT CLAIMED: that missing optional KTO inputs have defaults,
        that a reference passed to a reference-free objective is ignored, or
        that ORPO's exact-probability wall is replaced by a clamp.
        """
        import torch

        origin = type(self.objective).__name__
        softplus = torch.nn.functional.softplus
        recipe, policy, mask, reference = _checked_inputs(
            objective=self.objective,
            policy_logprobs=policy_logprobs,
            completion_mask=completion_mask,
            reference_logprobs=reference_logprobs,
            require_policy_grad=True,
        )

        if recipe == "kto":
            if reference is None:
                _missing_reference_refusal(origin)
            z0, flags = _checked_kto_inputs(
                kl_reference_point=kl_reference_point,
                desirable=desirable,
                row_count=int(policy.shape[0]),
            )
            # The kto_* names are deliberate: the paired branch below binds
            # policy_scores/reference_scores with a wider Optional type from
            # _paired_terms, and reusing the same names here would pin them
            # to Tensor, making that branch's reference guard read as dead
            # code under warn_unreachable in exchange for nothing.
            kto_policy_scores = _masked_sequence_sums(policy, mask)
            kto_reference_scores = _masked_sequence_sums(reference, mask)
            r = float(self.objective.beta) * (kto_policy_scores - kto_reference_scores)
            desirable_losses = float(self.objective.lambda_desirable) * torch.sigmoid(z0 - r)
            undesirable_losses = float(self.objective.lambda_undesirable) * torch.sigmoid(r - z0)
            row_losses = torch.where(flags, desirable_losses, undesirable_losses)
            loss = float(self.objective.weight) * row_losses.mean()
        else:
            if kl_reference_point is not None:
                raise BatchRefusal(
                    f"{origin} is paired but received 1 unexpected "
                    f"kl_reference_point; that input is meaningful only "
                    f"for KTO and would otherwise be silently unused"
                )
            if desirable is not None:
                raise BatchRefusal(
                    f"{origin} is paired but received 1 unexpected "
                    f"desirable tensor; per-row labels are meaningful "
                    f"only for KTO and would otherwise be silently unused"
                )
            policy_scores, reference_scores, supervised = _paired_terms(
                policy=policy,
                mask=mask,
                reference=reference,
            )

            if recipe == "logsig_margin":
                if reference_scores is None:
                    _missing_reference_refusal(origin)
                # DPOLoss: beta multiplies the difference of the two
                # anchored margins, exactly in this placement.
                margin = float(self.objective.beta) * (
                    (policy_scores[:, 0] - reference_scores[:, 0])
                    - (policy_scores[:, 1] - reference_scores[:, 1])
                )
                loss = float(self.objective.weight) * softplus(-margin).mean()
                sft_weight = float(self.objective.sft_weight)
                if sft_weight != 0.0:
                    sft = -(policy_scores[:, 0].sum() / supervised[:, 0].sum())
                    loss = loss + sft_weight * sft
            elif recipe == "logsig_margin+sft":
                # CPOLoss multiplies each side by beta before subtraction,
                # rather than algebraically rewriting beta*(pi_c-pi_r), to
                # preserve the oracle's float-order parity as closely as
                # tensor reduction permits.
                margin = (
                    float(self.objective.beta) * policy_scores[:, 0]
                    - float(self.objective.beta) * policy_scores[:, 1]
                )
                preference = softplus(-margin).mean()
                sft = float(self.objective.lambda_) * -(
                    policy_scores[:, 0].sum() / supervised[:, 0].sum()
                )
                loss = preference + sft
            elif recipe == "squared":
                if reference_scores is None:
                    _missing_reference_refusal(origin)
                # The IPOLoss margin is the anchored h; the paper's form
                # divides by nothing, while length_normalise=True selects
                # TRL's per-sequence-mean form, dividing each side's
                # anchored sum by that side's supervised-token count --
                # the same mask-derived counts the oracle uses. The hinge
                # is the paper's squared target either way.
                if self.objective.length_normalise:
                    anchored = policy_scores - reference_scores
                    margin = anchored[:, 0] / supervised[:, 0] - anchored[:, 1] / supervised[:, 1]
                else:
                    margin = (policy_scores[:, 0] - reference_scores[:, 0]) - (
                        policy_scores[:, 1] - reference_scores[:, 1]
                    )
                target = 1.0 / (2.0 * float(self.objective.tau))
                loss = float(self.objective.weight) * ((margin - target) ** 2).mean()
            elif recipe == "logsig_margin_lenorm":
                # SimPOLoss evaluates (beta / count) * score per side; the
                # division by the derived mask count precedes the score
                # multiplication exactly as the oracle writes it.
                rewards = (float(self.objective.beta) / supervised) * policy_scores
                margin = rewards[:, 0] - rewards[:, 1] - float(self.objective.gamma)
                loss = float(self.objective.weight) * softplus(-margin).mean()
            else:
                odds_margin, sft = _orpo_odds_margin_and_sft(policy_scores, supervised)
                odds = float(self.objective.lambda_) * softplus(-odds_margin).mean()
                loss = sft + odds

        if not bool(torch.isfinite(loss)):
            raise BatchRefusal(
                f"{origin} produced a non-finite loss ({loss.detach()!r}); "
                f"the recipe's guarded inputs were finite, so the overflow "
                f"arose inside the objective and clamping it would hide "
                f"a divergent margin"
            )
        if loss.dim() != 0 or not loss.requires_grad:
            raise BatchRefusal(  # pragma: no cover - guarded path above
                f"{origin} produced a loss with dim={loss.dim()} and "
                f"requires_grad={loss.requires_grad}; the optimiser "
                f"scalar must be 0-dimensional and differentiable"
            )
        return loss


def preference_metrics(
    objective: Any,
    policy_logprobs: torch.Tensor,
    completion_mask: torch.Tensor,
    reference_logprobs: torch.Tensor | None = None,
) -> dict[str, float]:
    """Return detached oracle-compatible ``accuracy`` and ``margin`` metrics.

    For DPO, CPO and SimPO the margin is the same full, scaled margin used
    in the loss. IPO's reported margin is its unscaled anchored ``h``, and
    ORPO's is the log-odds difference. Accuracy is always the fraction whose
    oracle-defined margin is strictly positive, so ties do not count as
    correct. KTO has no paired comparison and returns an empty mapping,
    just as its oracle deliberately emits counts rather than inventing a
    batch-composition-dependent pair accuracy.

    WHAT IS CLAIMED: outputs are Python floats computed without retaining
    gradients and follow the same margin definitions as the matching float
    oracles.

    WHAT IS NOT CLAIMED: that every objective's ``margin`` uses identical
    units; IPO, ORPO and the beta-scaled objectives deliberately do not.
    """
    recipe = _recipe_name(objective)
    if recipe == "kto":
        return {}

    origin = type(objective).__name__
    _, checked_policy, mask, reference = _checked_inputs(
        objective=objective,
        policy_logprobs=policy_logprobs,
        completion_mask=completion_mask,
        reference_logprobs=reference_logprobs,
        require_policy_grad=False,
    )
    policy = checked_policy.detach()
    policy_scores, reference_scores, supervised = _paired_terms(
        policy=policy,
        mask=mask,
        reference=reference,
    )

    margin: torch.Tensor
    if recipe == "logsig_margin":
        if reference_scores is None:
            _missing_reference_refusal(origin)
        margin = float(objective.beta) * (
            (policy_scores[:, 0] - reference_scores[:, 0])
            - (policy_scores[:, 1] - reference_scores[:, 1])
        )
    elif recipe == "logsig_margin+sft":
        margin = (
            float(objective.beta) * policy_scores[:, 0]
            - float(objective.beta) * policy_scores[:, 1]
        )
    elif recipe == "squared":
        if reference_scores is None:
            _missing_reference_refusal(origin)
        if objective.length_normalise:
            anchored = policy_scores - reference_scores
            margin = anchored[:, 0] / supervised[:, 0] - anchored[:, 1] / supervised[:, 1]
        else:
            margin = (policy_scores[:, 0] - reference_scores[:, 0]) - (
                policy_scores[:, 1] - reference_scores[:, 1]
            )
    elif recipe == "logsig_margin_lenorm":
        rewards = (float(objective.beta) / supervised) * policy_scores
        margin = rewards[:, 0] - rewards[:, 1] - float(objective.gamma)
    else:
        margin, _ = _orpo_odds_margin_and_sft(policy_scores, supervised)

    correct = (margin > 0.0).to(dtype=margin.dtype)
    return {
        "accuracy": float(correct.mean().item()),
        "margin": float(margin.mean().item()),
    }
