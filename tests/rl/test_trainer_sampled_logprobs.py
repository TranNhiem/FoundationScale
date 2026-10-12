"""Scoring tokens under the distribution the engine sampled from (temperature, top-k, top-p).

Every expected value is computed independently from a plain softmax over an explicitly
listed support, never through the code under test.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from foundationscale.rl import trainer as trainer_mod  # noqa: E402
from foundationscale.rl.trainer import (  # noqa: E402
    RLTrainConfig,
    RLTrainer,
    TrainerRefusal,
    _sampling_support,
    _token_logprobs,
)

# One position, vocabulary of 5. Probabilities at temperature 1 (descending):
# token 0: logit 3.0, token 1: 2.0, token 2: 1.0, token 3: 0.0, token 4: -1.0
LOGITS = torch.tensor([[[3.0, 2.0, 1.0, 0.0, -1.0]]])


def _expected(target: int, support: list[int], temperature: float = 1.0) -> float:
    z = LOGITS[0, 0] / temperature
    return float(z[target] - torch.logsumexp(z[support], dim=-1))


def _score(target: int, **kw: float) -> float:
    return float(_token_logprobs(LOGITS, torch.tensor([[target]]), **kw)[0, 0])


def test_defaults_are_bitwise_the_raw_logit_score() -> None:
    """With no sampling transform the result equals the historical formula exactly."""
    g = torch.Generator().manual_seed(0)
    logits = torch.randn(2, 7, 11, generator=g)
    targets = torch.randint(0, 11, (2, 7), generator=g)
    raw = logits.gather(-1, targets.unsqueeze(-1)).squeeze(-1) - torch.logsumexp(logits, dim=-1)
    assert torch.equal(_token_logprobs(logits, targets), raw)


def test_temperature_scales_logits_over_the_full_vocabulary() -> None:
    """temperature T scores log softmax(z / T)."""
    assert _score(2, temperature=0.7) == pytest.approx(_expected(2, [0, 1, 2, 3, 4], 0.7))


def test_top_k_renormalises_over_the_k_largest() -> None:
    """top_k=2 keeps tokens 0 and 1 only."""
    assert _score(1, top_k=2) == pytest.approx(_expected(1, [0, 1]))


def test_top_p_renormalises_over_the_nucleus() -> None:
    """At T=1 token 0 holds 0.636 and tokens 0+1 hold 0.870, so top_p=0.8 keeps {0, 1}."""
    p = torch.softmax(LOGITS[0, 0], dim=-1)
    assert float(p[0]) < 0.8 < float(p[0] + p[1])
    assert _score(0, top_p=0.8) == pytest.approx(_expected(0, [0, 1]))


def test_top_k_then_top_p_matches_the_vllm_order() -> None:
    """top_k=3 first, then top_p over the renormalised three: token 0 alone holds 0.665
    of {0,1,2}, so top_p=0.6 keeps token 0 only."""
    q = torch.softmax(LOGITS[0, 0, :3], dim=-1)
    assert float(q[0]) > 0.6
    keep = _sampling_support(LOGITS, top_p=0.6, top_k=3)
    assert keep[0, 0].tolist() == [True, False, False, False, False]


def test_a_target_outside_the_support_is_scored_finitely() -> None:
    """An engine/trainer disagreement at the nucleus edge must not produce -inf."""
    score = _score(4, top_k=2)
    assert score == pytest.approx(_expected(4, [0, 1, 4]))
    assert score > float("-inf")


def test_chunking_does_not_change_the_result(monkeypatch: pytest.MonkeyPatch) -> None:
    """A budget of 2 positions per chunk gives the same scores as one chunk."""
    g = torch.Generator().manual_seed(1)
    logits = torch.randn(1, 9, 13, generator=g)
    targets = torch.randint(0, 13, (1, 9), generator=g)
    whole = _token_logprobs(logits, targets, temperature=0.8, top_p=0.9, top_k=5)
    monkeypatch.setattr(trainer_mod, "_SCORE_CHUNK_ENTRIES", 2 * 13)
    chunked = _token_logprobs(logits, targets, temperature=0.8, top_p=0.9, top_k=5)
    assert torch.allclose(whole, chunked)


def test_gradient_flows_through_the_kept_logits_only() -> None:
    """Backward works; logits outside the support receive zero gradient."""
    logits = LOGITS.clone().requires_grad_(True)
    _token_logprobs(logits, torch.tensor([[0]]), top_k=2).sum().backward()
    assert logits.grad is not None
    assert torch.all(torch.isfinite(logits.grad))
    assert logits.grad[0, 0, 2:].abs().sum() == 0


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("score_temperature", 0.0),
        ("score_temperature", float("nan")),
        ("score_temperature", True),
        ("score_top_p", 0.0),
        ("score_top_p", 1.5),
        ("score_top_k", -1),
        ("score_top_k", 2.0),
    ],
)
def test_bad_score_parameters_are_refused(field: str, value: object) -> None:
    """Scoring parameters are validated before any model is loaded."""
    config = RLTrainConfig(model="unused", **{field: value})  # type: ignore[arg-type]
    with pytest.raises(TrainerRefusal, match=field):
        RLTrainer(config)
