
"""Continued-pretraining (CPT) policy.

Where the interface spec is silent, this module chooses simple, documented
defaults:

- LR comes from a size band table (coarser is unmeasured, so we do not invent
  per-family LRs).
- The recommended token budget is ~20x the domain corpus, clamped by a
  size-dependent ceiling; the executed budget is
  ``min(domain_tokens * EPOCHS_CAP, recommended)``.
- The replay ratio is taken from the Data Engine mixing rules when that module
  is importable; otherwise a literature default of 0.25 is used and clamped to
  [0.05, 0.30] (the commonly cited CPT replay range). ``preserve_general=False``
  disables replay entirely (0.0).
"""
from __future__ import annotations

from typing import Any

EPOCHS_CAP = 4
RECOMMENDED_MULTIPLIER = 20
WARMUP_RATIO = 0.01
MIN_LR_RATIO = 0.10

# (size_b upper bound, peak LR) bands; the unbounded tail uses _LR_DEFAULT.
_LR_TABLE: tuple[tuple[float, float], ...] = ((3.0, 3e-5), (10.0, 2e-5), (40.0, 1.5e-5))
_LR_DEFAULT = 1e-5

# Size-dependent ceiling (tokens) for the 20x-domain recommended budget.
_TOKEN_CAP_TABLE: tuple[tuple[float, float], ...] = ((3.0, 100e9), (10.0, 200e9), (40.0, 500e9))
_TOKEN_CAP_DEFAULT = 1.0e12

_REPLAY_DEFAULT = 0.25
_REPLAY_MIN = 0.05
_REPLAY_MAX = 0.30


def _lr_for_size(size_b: float) -> tuple[float, str]:
    for bound, lr in _LR_TABLE:
        if size_b <= bound:
            return lr, f"<= {bound:g}B band"
    return _LR_DEFAULT, "> 40B band"


def _token_cap(size_b: float) -> float:
    for bound, cap in _TOKEN_CAP_TABLE:
        if size_b <= bound:
            return cap
    return _TOKEN_CAP_DEFAULT


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _replay_from_mixing_rules(domain_tokens: int, goal: Any, preserve_general: bool, because: list[str]) -> float | None:
    """Ask the Data Engine mixture designer for the non-domain share.

    Returns None when the module/opinion is unavailable so the caller can fall
    back to the literature default. Any mixture failure degrades to a stated
    default, never to silence.
    """
    if not preserve_general:
        return 0.0
    try:
        from foundationskills.skills.data_engine.mix import design_mixture
    except Exception:
        because.append(
            "replay: data_engine mixing rules unavailable (import failed); "
            f"fell back to the literature default {_REPLAY_DEFAULT} (approximate)"
        )
        return None
    try:
        mixture = design_mixture(
            goal=str(goal),
            stage="cpt",
            domain_tokens=int(domain_tokens),
            total_tokens=None,
            preserve_general=True,
            available=[],
        )
    except Exception as exc:  # noqa: BLE001 - recorded, fallback used
        because.append(
            f"replay: design_mixture failed ({type(exc).__name__}: {exc}); "
            f"fell back to the literature default {_REPLAY_DEFAULT} (approximate)"
        )
        return None
    components = mixture.get("components", []) if isinstance(mixture, dict) else []
    domainish = {"domain", "target", "in_domain", "domain_corpus", str(goal).lower()}
    replay_ratio = 0.0
    for comp in components:
        if not isinstance(comp, dict):
            continue
        name = str(comp.get("name", "")).lower()
        try:
            ratio = float(comp.get("ratio") or 0.0)
        except (TypeError, ValueError):
            ratio = 0.0
        if name and name not in domainish and "domain" not in name:
            replay_ratio += ratio
    if replay_ratio <= 0.0:
        because.append(
            "replay: mixing rules returned no general/replay component; "
            f"fell back to the literature default {_REPLAY_DEFAULT} (approximate)"
        )
        return None
    clamped = _clamp(replay_ratio, _REPLAY_MIN, _REPLAY_MAX)
    because.append(
        f"replay {clamped:.3f} from data_engine mixing rules "
        f"(raw {replay_ratio:.3f}, clamped to [{_REPLAY_MIN}, {_REPLAY_MAX}])"
    )
    return clamped


def cpt_policy(variant: Any, domain_tokens: int | float, goal: Any, preserve_general: bool) -> dict[str, Any]:
    """CPT hyperparameters for ``variant`` over a domain corpus.

    Contract: returns a dict with keys ``lr`` (float), ``schedule`` (dict),
    ``token_budget`` (int), ``replay_ratio`` (float in [0, 0.30]),
    ``epochs_cap`` (int) and ``because`` (list[str]) explaining every choice.
    """
    size_b = float(getattr(variant, "size_b", 0.0) or 0.0)
    if size_b <= 0:
        # rv41: an unknown size must not fall into the <=3B band (highest LR,
        # smallest token cap) -- exactly wrong for a large model.
        raise ValueError("missing input: base_model.size_b (CPT learning rate and token budget depend on it)")
    size_label = f"{size_b:g}B"
    domain_tokens = int(max(0, domain_tokens))
    because: list[str] = []

    lr, band = _lr_for_size(size_b)
    because.append(f"lr {lr:g}: {size_label} falls in the {band} CPT LR band")

    cap = _token_cap(size_b)
    recommended = int(min(RECOMMENDED_MULTIPLIER * domain_tokens, cap))
    budget_epochs_limited = int(domain_tokens * EPOCHS_CAP)
    token_budget = int(min(budget_epochs_limited, recommended))
    because.append(
        f"token budget {token_budget:,} = min(domain x {EPOCHS_CAP} epochs = {budget_epochs_limited:,}, "
        f"recommended ~{RECOMMENDED_MULTIPLIER}x domain capped at {int(cap):,} = {recommended:,})"
    )

    replay = _replay_from_mixing_rules(domain_tokens, goal, preserve_general, because)
    if replay is None:
        replay = _clamp(_REPLAY_DEFAULT, _REPLAY_MIN, _REPLAY_MAX)
    if not preserve_general:
        because.append("replay 0.0: preserve_general is false")

    epochs_planned = (token_budget / domain_tokens) if domain_tokens > 0 else 0.0
    schedule = {
        "type": "cosine",
        "warmup_ratio": WARMUP_RATIO,
        "min_lr_ratio": MIN_LR_RATIO,
        "note": (
            "warmup ~1% of steps, cosine decay to 10% of peak. For long CPT runs prefer "
            "rewarm+redecay: rewarm to the max LR then re-decay per cycle "
            "(Ibrahim et al. 2024, 'Simple and Scalable Strategies to Continually Pre-train "
            "Large Language Models')."
        ),
    }
    because.append("schedule: warmup ~1% + cosine to 10%, with the Ibrahim et al. 2024 rewarm+redecay note")

    return {
        "lr": lr,
        "schedule": schedule,
        "token_budget": token_budget,
        "recommended_tokens": recommended,
        "replay_ratio": float(replay),
        "epochs_cap": EPOCHS_CAP,
        "epochs_planned": round(epochs_planned, 3),
        "because": because,
    }
