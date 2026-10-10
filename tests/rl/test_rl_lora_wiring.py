"""CPU tests for the LoRA wiring shared by RLTrainer and PreferenceTrainer:

* ``_apply_lora_adapter`` -- target selection reuse (the SAME
  ``foundationscale.families.plan_adapter_targets`` the SFT plane uses),
  the peft-missing / zero-attached-modules refusals, and the
  construction-failure-propagates-uncaught classification.
* ``_trainable_parameters`` -- the optimizer-only-gets-trainable-params filter.
* ``_reference_plan`` + ``_AdapterDisabledReference`` -- reference-path
  dispatch: adapter -> disable_adapter (no second model), no adapter ->
  the historical second_copy path, byte-identical selection.

peft is genuinely NOT installed in this CPU environment (measured), so
every peft-dependent case here drives ``_apply_lora_adapter`` against a
``sys.modules["peft"]`` stand-in -- the SAME technique
``tests/train/test_train_precision_adapter.py`` uses for the SFT plane's
peft wrap, never against real peft. The model under test is a REAL small
``torch.nn.Module`` (torch IS installed), so the target-selection calls
exercise the real, unmocked ``foundationscale.families`` registry code.

FS_FORBID_SKIPS=1 clean -- nothing here may skip.
"""

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest
import torch
from torch import nn

from foundationscale.rl.trainer import (
    TrainerRefusal,
    _AdapterDisabledReference,
    _apply_lora_adapter,
    _reference_plan,
    _trainable_parameters,
)

# ---------------------------------------------------------------------------
# A tiny real model + a tiny fake peft, shared by the _apply_lora_adapter tests
# ---------------------------------------------------------------------------


class _TinyModel(nn.Module):
    """Two named Linear leaves and a config no registered family matches.

    The unmatched ``model_type`` is deliberate: it routes
    ``plan_adapter_targets`` through its "no family but names were
    declared -- pass them through" branch, which is the one this test suite
    can assert on WITHOUT depending on which families are registered today.
    """

    def __init__(self) -> None:
        super().__init__()
        self.q_proj = nn.Linear(4, 4)
        self.k_proj = nn.Linear(4, 4)
        self.config = SimpleNamespace(to_dict=lambda: {"model_type": "totally_fake_test_model"})


class _FakePeftModel:
    """Stand-in for peft's ``PeftModel``: wraps a real module, adds lora_A/B."""

    def __init__(self, base: nn.Module, config: Any, attached: list[str]) -> None:
        self._base = base
        self.peft_config = {"default": config}
        self._attached = attached
        for _, p in base.named_parameters():
            p.requires_grad_(False)
        self._lora_params: dict[str, nn.Parameter] = {}
        for name in attached:
            self._lora_params[f"{name}.lora_A.weight"] = nn.Parameter(torch.zeros(2, 2))
            self._lora_params[f"{name}.lora_B.weight"] = nn.Parameter(torch.zeros(2, 2))

    def named_parameters(self):
        yield from self._base.named_parameters()
        yield from self._lora_params.items()

    def named_modules(self):
        return self._base.named_modules()


def _install_fake_peft(monkeypatch: pytest.MonkeyPatch, *, on_wrap: Any = None) -> list[Any]:
    """Install a fake ``peft`` module; returns the sink of LoraConfig objects built."""
    sink: list[Any] = []

    def fake_get_peft_model(model: Any, config: Any) -> Any:
        if on_wrap is not None:
            on_wrap(model, config)
        sink.append(config)
        model_names = {name for name, _ in model.named_modules() if name}
        attached = [name for name in config.target_modules if name in model_names]
        return _FakePeftModel(model, config, attached)

    fake_module = ModuleType("peft")
    fake_module.LoraConfig = lambda **kw: SimpleNamespace(**kw)  # type: ignore[attr-defined]
    fake_module.get_peft_model = fake_get_peft_model  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "peft", fake_module)
    return sink


# ---------------------------------------------------------------------------
# _apply_lora_adapter
# ---------------------------------------------------------------------------


def test_adapter_none_returns_model_unchanged_and_empty_notes() -> None:
    model = _TinyModel()
    returned, notes = _apply_lora_adapter(
        model,
        adapter=None,
        adapter_rank=None,
        adapter_alpha=None,
        adapter_targets=None,
        adapter_dropout=None,
        log_prefix="[test]",
    )
    assert returned is model
    assert notes == {}


def test_apply_lora_adapter_refuses_96_when_peft_missing(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setitem(sys.modules, "peft", None)
    with pytest.raises(SystemExit) as excinfo:
        _apply_lora_adapter(
            _TinyModel(),
            adapter="lora",
            adapter_rank=8,
            adapter_alpha=None,
            adapter_targets=("q_proj",),
            adapter_dropout=None,
            log_prefix="[test]",
        )
    assert excinfo.value.code == 96
    assert "'peft'" in capsys.readouterr().err


def test_apply_lora_adapter_reuses_family_target_selection_pass_through(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # No registered family matches "totally_fake_test_model", so
    # plan_adapter_targets passes the DECLARED targets through literally --
    # this is the real foundationscale.families code, not a stand-in.
    sink = _install_fake_peft(monkeypatch)
    model = _TinyModel()
    wrapped, notes = _apply_lora_adapter(
        model,
        adapter="lora",
        adapter_rank=8,
        adapter_alpha=16.0,
        adapter_targets=("q_proj",),
        adapter_dropout=0.1,
        log_prefix="[test]",
    )
    assert len(sink) == 1
    built_config = sink[0]
    assert built_config.r == 8
    assert built_config.lora_alpha == 16.0
    assert built_config.lora_dropout == 0.1
    assert list(built_config.target_modules) == ["q_proj"]
    assert notes["adapter.mode"] == "lora"
    assert notes["adapter.rank"] == "8"
    assert notes["adapter.attached_modules"] == "1"
    assert notes["adapter.resolved_targets"] == "q_proj"
    assert wrapped.peft_config == {"default": built_config}
    # k_proj was never declared, so it must not have been swept in.
    lora_names = [name for name, _ in wrapped.named_parameters() if ".lora_" in name]
    assert all(name.startswith("q_proj.") for name in lora_names)


def test_apply_lora_adapter_omits_unset_knobs_from_lora_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # alpha/dropout stay UNSET (not defaulted to some value) when the caller
    # did not declare them -- peft's own defaults then apply.
    sink = _install_fake_peft(monkeypatch)
    _apply_lora_adapter(
        _TinyModel(),
        adapter="lora",
        adapter_rank=4,
        adapter_alpha=None,
        adapter_targets=("q_proj",),
        adapter_dropout=None,
        log_prefix="[test]",
    )
    built_config = sink[0]
    assert not hasattr(built_config, "lora_alpha")
    assert not hasattr(built_config, "lora_dropout")


def test_apply_lora_adapter_refuses_96_on_zero_attached_modules(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _install_fake_peft(monkeypatch)
    with pytest.raises(SystemExit) as excinfo:
        _apply_lora_adapter(
            _TinyModel(),
            adapter="lora",
            adapter_rank=8,
            adapter_alpha=None,
            adapter_targets=("nonexistent_proj",),
            adapter_dropout=None,
            log_prefix="[test]",
        )
    assert excinfo.value.code == 96
    assert "0 modules" in capsys.readouterr().err


def test_apply_lora_adapter_lets_a_construction_failure_propagate_uncaught(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # SAME classification train/loop.py's SFT plane gives this: a
    # get_peft_model() crash is RED, not a named refusal, and must not be
    # silently swallowed.
    def _boom(model: Any, config: Any) -> Any:
        raise RuntimeError("boom: wrap construction failed")

    fake_module = ModuleType("peft")
    fake_module.LoraConfig = lambda **kw: SimpleNamespace(**kw)  # type: ignore[attr-defined]
    fake_module.get_peft_model = _boom  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "peft", fake_module)
    with pytest.raises(RuntimeError, match="boom: wrap construction failed"):
        _apply_lora_adapter(
            _TinyModel(),
            adapter="lora",
            adapter_rank=8,
            adapter_alpha=None,
            adapter_targets=("q_proj",),
            adapter_dropout=None,
            log_prefix="[test]",
        )


# ---------------------------------------------------------------------------
# _trainable_parameters
# ---------------------------------------------------------------------------


def test_trainable_parameters_filters_frozen_params() -> None:
    model = _TinyModel()
    for p in model.k_proj.parameters():
        p.requires_grad_(False)
    trainable = list(_trainable_parameters(model))
    assert all(p.requires_grad for p in trainable)
    assert set(trainable) == set(model.q_proj.parameters())


def test_trainable_parameters_is_the_identity_set_for_full_finetune() -> None:
    # Every parameter requires grad (the historical, non-adapter case): the
    # filtered set must equal model.parameters() exactly, same order.
    model = _TinyModel()
    assert list(_trainable_parameters(model)) == list(model.parameters())


# ---------------------------------------------------------------------------
# _reference_plan
# ---------------------------------------------------------------------------


def test_reference_plan_no_adapter_is_always_second_copy() -> None:
    assert (
        _reference_plan(adapter=None, reference_model=None, model="m", refresh_every=0)
        == "second_copy"
    )
    assert (
        _reference_plan(adapter=None, reference_model="other", model="m", refresh_every=5)
        == "second_copy"
    )


def test_reference_plan_adapter_same_model_is_disable_adapter() -> None:
    assert (
        _reference_plan(adapter="lora", reference_model=None, model="m", refresh_every=0)
        == "disable_adapter"
    )
    assert (
        _reference_plan(adapter="lora", reference_model="m", model="m", refresh_every=0)
        == "disable_adapter"
    )


def test_reference_plan_adapter_distinct_reference_model_is_second_copy() -> None:
    assert (
        _reference_plan(
            adapter="lora", reference_model="other/checkpoint", model="m", refresh_every=0
        )
        == "second_copy"
    )


def test_reference_plan_adapter_same_model_with_refresh_refuses() -> None:
    with pytest.raises(TrainerRefusal, match="ref_refresh_steps"):
        _reference_plan(adapter="lora", reference_model=None, model="m", refresh_every=5)


# ---------------------------------------------------------------------------
# _AdapterDisabledReference
# ---------------------------------------------------------------------------


class _FakePolicy(nn.Module):
    """A real nn.Module with a fake peft-shaped ``disable_adapter()``.

    Records, at call time, whether the policy was in eval mode, whether the
    adapter was disabled, and whether autograd was off -- the three
    invariants _AdapterDisabledReference exists to guarantee.
    """

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(2, 2)
        self.adapter_disabled = False
        self.calls: list[dict[str, bool]] = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.calls.append(
            {
                "was_eval": not self.training,
                "adapter_disabled": self.adapter_disabled,
                "grad_disabled": not torch.is_grad_enabled(),
            }
        )
        return self.linear(x)

    def disable_adapter(self):
        from contextlib import contextmanager

        @contextmanager
        def _ctx():
            self.adapter_disabled = True
            try:
                yield
            finally:
                self.adapter_disabled = False

        return _ctx()


def test_adapter_disabled_reference_forwards_under_eval_no_grad_disabled_adapter() -> None:
    policy = _FakePolicy()
    policy.train()
    ref = _AdapterDisabledReference(policy)
    out = ref(torch.randn(1, 2))
    assert isinstance(out, torch.Tensor)
    assert len(policy.calls) == 1
    assert policy.calls[0] == {"was_eval": True, "adapter_disabled": True, "grad_disabled": True}


@pytest.mark.parametrize("start_training", [True, False])
def test_adapter_disabled_reference_restores_prior_training_mode(start_training: bool) -> None:
    policy = _FakePolicy()
    policy.train(start_training)
    ref = _AdapterDisabledReference(policy)
    ref(torch.randn(1, 2))
    assert policy.training is start_training
    assert policy.adapter_disabled is False  # restored, not left disabled


def test_adapter_disabled_reference_eval_and_parameters_are_inert() -> None:
    policy = _FakePolicy()
    ref = _AdapterDisabledReference(policy)
    assert ref.eval() is ref
    assert list(ref.parameters()) == []


def test_adapter_disabled_reference_unwraps_a_ddp_style_module_attribute() -> None:
    # Measured on 2x H200: sharding='ddp' hands _AdapterDisabledReference a
    # DistributedDataParallel instance, which does not proxy .disable_adapter()
    # to the wrapped module -- calling it on the wrapper raises AttributeError.
    # Unwrapping .module (the same idiom save_checkpoint already uses) fixes it.
    class _DDPLike:
        def __init__(self, module: _FakePolicy) -> None:
            self.module = module

    inner = _FakePolicy()
    inner.train()
    wrapper = _DDPLike(inner)
    ref = _AdapterDisabledReference(wrapper)
    ref(torch.randn(1, 2))
    assert len(inner.calls) == 1
    assert inner.calls[0] == {"was_eval": True, "adapter_disabled": True, "grad_disabled": True}
    assert inner.training is True  # restored on the UNWRAPPED module


def test_adapter_disabled_reference_matches_disabled_forward_at_zero_delta() -> None:
    """Step-0 invariant at the CPU unit level: a LoRA delta initialised to
    zero (peft's real init: A random, B zero) makes the adapter-enabled and
    adapter-disabled forwards IDENTICAL. _AdapterDisabledReference's job is
    only to read the disabled forward faithfully; this checks it does.
    """

    class _ZeroDeltaPolicy(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = nn.Linear(3, 3)
            self._adapter_on = True

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            base = self.linear(x)
            delta = torch.zeros_like(base) if not self._adapter_on else torch.zeros_like(base)
            return base + delta

        def disable_adapter(self):
            from contextlib import contextmanager

            @contextmanager
            def _ctx():
                previous = self._adapter_on
                self._adapter_on = False
                try:
                    yield
                finally:
                    self._adapter_on = previous

            return _ctx()

    policy = _ZeroDeltaPolicy()
    policy.eval()
    x = torch.randn(2, 3)
    with torch.no_grad():
        policy_out = policy(x)
    ref = _AdapterDisabledReference(policy)
    reference_out = ref(x)
    assert torch.equal(policy_out, reference_out)
