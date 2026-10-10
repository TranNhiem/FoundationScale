"""CPU tests for ``_generation_mode``: the eval()+use_cache bracket EVERY
``.generate()`` call site on the online path (``trainer.py``'s built-in leg,
``ppo_step``, ``online_pref_step``) now uses.

MEASURED BUG this fixes (gemma-4-12B-it, GRPO rollout, 2026-10-09): rollouts
ran with the model still in ``.train()`` mode and
``model.config.use_cache=False`` (left that way by
``gradient_checkpointing_enable()``). ``Gemma4UnifiedTextDecoderLayer``
drops its KV cache whenever ``self.training and self.gradient_checkpointing``,
which silently corrupts ``generate()``'s incremental decode loop: the first
generated token is correct (a full-prompt forward needs no cache) and every
token after it is near-random. ``generate()`` raises nothing -- a caller
that does not inspect the decoded text never learns the rollout was
garbage. These tests pin the three invariants that make the fix correct:
eval() for the duration, use_cache restored exactly (never fabricated for
a family that never had it), and training mode restored exactly (never
unconditionally forced to True).

FS_FORBID_SKIPS=1 clean -- nothing here may skip.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
import torch
from torch import nn

from foundationscale.rl.trainer import _generation_mode


class _FakeModel(nn.Module):
    """Records whether it was in eval mode and what use_cache read, per call."""

    def __init__(self, *, with_use_cache: bool = True) -> None:
        super().__init__()
        self.linear = nn.Linear(2, 2)
        self.config: Any = SimpleNamespace(use_cache=False) if with_use_cache else object()
        self.generate_calls: list[dict[str, Any]] = []

    def generate(self, **kwargs: Any) -> torch.Tensor:
        self.generate_calls.append(
            {
                "was_eval": not self.training,
                "use_cache": getattr(self.config, "use_cache", "<absent>"),
            }
        )
        return torch.zeros(1, 1, dtype=torch.long)


def test_generation_mode_sets_eval_and_use_cache_true() -> None:
    model = _FakeModel()
    model.train()
    with _generation_mode(model) as gen_model:
        gen_model.generate()
    assert model.generate_calls == [{"was_eval": True, "use_cache": True}]


@pytest.mark.parametrize("start_training", [True, False])
def test_generation_mode_restores_training_mode_exactly(start_training: bool) -> None:
    model = _FakeModel()
    model.train(start_training)
    with _generation_mode(model):
        assert model.training is False
    assert model.training is start_training


def test_generation_mode_restores_prior_use_cache_value() -> None:
    model = _FakeModel()
    model.config.use_cache = False
    with _generation_mode(model):
        assert model.config.use_cache is True
    assert model.config.use_cache is False


def test_generation_mode_leaves_a_family_with_no_use_cache_attribute_alone() -> None:
    # A family whose config never had use_cache must not GAIN the
    # attribute: fabricating it would be a claim about the family's own
    # generation config this function has no basis to make.
    model = _FakeModel(with_use_cache=False)
    with _generation_mode(model):
        assert not hasattr(model.config, "use_cache")
    assert not hasattr(model.config, "use_cache")


def test_generation_mode_unwraps_a_ddp_style_module_attribute() -> None:
    class _DDPLike:
        def __init__(self, module: _FakeModel) -> None:
            self.module = module

    inner = _FakeModel()
    inner.train()
    wrapper = _DDPLike(inner)
    with _generation_mode(wrapper) as gen_model:
        assert gen_model is inner
        gen_model.generate()
    assert inner.generate_calls == [{"was_eval": True, "use_cache": True}]
    assert inner.training is True


def test_generation_mode_unshards_before_and_reshards_after(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Integration wiring check: _generation_mode must call
    # distributed.unshard_for_generation(target) and invoke its returned
    # resharder in finally -- the fix for the SECOND measured bug (the
    # root's own first forward still saw embed_tokens un-materialised even
    # after the vision tower got its own fully_shard unit).
    calls: list[str] = []

    def _fake_unshard_for_generation(target: Any):
        calls.append(f"unshard:{target is model}")
        return lambda: calls.append("reshard")

    import foundationscale.rl.distributed as dist_mod

    monkeypatch.setattr(dist_mod, "unshard_for_generation", _fake_unshard_for_generation)

    model = _FakeModel()
    with _generation_mode(model) as gen_model:
        assert calls == ["unshard:True"]
        gen_model.generate()
    assert calls == ["unshard:True", "reshard"]


def test_generation_mode_reshards_even_if_generate_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def _fake_unshard_for_generation(target: Any):
        return lambda: calls.append("reshard")

    import foundationscale.rl.distributed as dist_mod

    monkeypatch.setattr(dist_mod, "unshard_for_generation", _fake_unshard_for_generation)

    model = _FakeModel()
    with pytest.raises(RuntimeError), _generation_mode(model):
        raise RuntimeError("boom")
    assert calls == ["reshard"]


def test_generation_mode_restores_state_even_if_generate_raises() -> None:
    model = _FakeModel()
    model.train()

    class _Boom(RuntimeError):
        pass

    with pytest.raises(_Boom), _generation_mode(model):
        raise _Boom("generation failed mid-call")
    assert model.training is True
    assert model.config.use_cache is False
