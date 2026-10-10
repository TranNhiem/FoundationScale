"""CPU tests: LoRA adapter config fields on RLTrainConfig / PreferenceTrainConfig.

No torch, no peft, no model load: every case here raises (or does not raise)
``TrainerRefusal`` from pure config validation, in ``RLTrainer.__init__`` /
``PreferenceTrainer.__init__``, before any dependency is imported. Mirrors
the SFT plane's own adapter validation (train/loop.py's RLTrainConfig
__post_init__ equivalent), proven here against the SAME five refusal shapes:
unknown adapter name, partial declaration (adapter_* set while adapter is
None), adapter set without a positive adapter_rank, and adapter_targets=().
FS_FORBID_SKIPS=1 clean -- nothing here may skip.
"""

from __future__ import annotations

import pytest

from foundationscale.rl.preference_trainer import PreferenceTrainConfig, PreferenceTrainer
from foundationscale.rl.trainer import RLTrainConfig, RLTrainer, TrainerRefusal

# ---------------------------------------------------------------------------
# RLTrainConfig / RLTrainer
# ---------------------------------------------------------------------------


def test_rl_adapter_none_is_the_default_and_constructs() -> None:
    cfg = RLTrainConfig(model="m")
    assert cfg.adapter is None
    assert cfg.adapter_rank is None
    assert cfg.adapter_alpha is None
    assert cfg.adapter_targets is None
    assert cfg.adapter_dropout is None
    RLTrainer(cfg)  # does not raise


def test_rl_adapter_unknown_name_refuses() -> None:
    cfg = RLTrainConfig(model="m", adapter="qlora", adapter_rank=8)
    with pytest.raises(TrainerRefusal, match="qlora"):
        RLTrainer(cfg)


@pytest.mark.parametrize(
    "field_name,value",
    [
        ("adapter_rank", 8),
        ("adapter_alpha", 16.0),
        ("adapter_targets", ("q_proj",)),
        ("adapter_dropout", 0.05),
    ],
)
def test_rl_partial_adapter_declaration_refuses(field_name: str, value: object) -> None:
    cfg = RLTrainConfig(model="m", **{field_name: value})
    with pytest.raises(TrainerRefusal, match=field_name):
        RLTrainer(cfg)


@pytest.mark.parametrize("bad_rank", [None, 0, -1, True, 2.0, "8"])
def test_rl_adapter_lora_requires_positive_int_rank(bad_rank: object) -> None:
    cfg = RLTrainConfig(model="m", adapter="lora", adapter_rank=bad_rank)
    with pytest.raises(TrainerRefusal, match="adapter_rank"):
        RLTrainer(cfg)


def test_rl_adapter_targets_empty_tuple_refuses_as_vacuous() -> None:
    cfg = RLTrainConfig(model="m", adapter="lora", adapter_rank=8, adapter_targets=())
    with pytest.raises(TrainerRefusal, match="vacuous"):
        RLTrainer(cfg)


def test_rl_adapter_targets_list_is_coerced_to_tuple() -> None:
    cfg = RLTrainConfig(model="m", adapter="lora", adapter_rank=8, adapter_targets=["q_proj"])
    RLTrainer(cfg)
    assert cfg.adapter_targets == ("q_proj",)
    assert isinstance(cfg.adapter_targets, tuple)


def test_rl_adapter_lora_minimal_valid_config_constructs() -> None:
    cfg = RLTrainConfig(
        model="m",
        adapter="lora",
        adapter_rank=16,
        adapter_alpha=32.0,
        adapter_targets=("q_proj", "v_proj"),
        adapter_dropout=0.0,
    )
    trainer = RLTrainer(cfg)
    assert trainer.config.adapter == "lora"


# ---------------------------------------------------------------------------
# PreferenceTrainConfig / PreferenceTrainer
# ---------------------------------------------------------------------------


def test_pref_adapter_none_is_the_default_and_constructs() -> None:
    cfg = PreferenceTrainConfig(model="m", dataset="d")
    assert cfg.adapter is None
    PreferenceTrainer(cfg)  # does not raise


def test_pref_adapter_unknown_name_refuses() -> None:
    cfg = PreferenceTrainConfig(model="m", dataset="d", adapter="qlora", adapter_rank=8)
    with pytest.raises(TrainerRefusal, match="qlora"):
        PreferenceTrainer(cfg)


@pytest.mark.parametrize(
    "field_name,value",
    [
        ("adapter_rank", 8),
        ("adapter_alpha", 16.0),
        ("adapter_targets", ("q_proj",)),
        ("adapter_dropout", 0.05),
    ],
)
def test_pref_partial_adapter_declaration_refuses(field_name: str, value: object) -> None:
    cfg = PreferenceTrainConfig(model="m", dataset="d", **{field_name: value})
    with pytest.raises(TrainerRefusal, match=field_name):
        PreferenceTrainer(cfg)


@pytest.mark.parametrize("bad_rank", [None, 0, -1, True, 2.0, "8"])
def test_pref_adapter_lora_requires_positive_int_rank(bad_rank: object) -> None:
    cfg = PreferenceTrainConfig(model="m", dataset="d", adapter="lora", adapter_rank=bad_rank)
    with pytest.raises(TrainerRefusal, match="adapter_rank"):
        PreferenceTrainer(cfg)


def test_pref_adapter_targets_empty_tuple_refuses_as_vacuous() -> None:
    cfg = PreferenceTrainConfig(
        model="m", dataset="d", adapter="lora", adapter_rank=8, adapter_targets=()
    )
    with pytest.raises(TrainerRefusal, match="vacuous"):
        PreferenceTrainer(cfg)


def test_pref_adapter_targets_list_is_coerced_to_tuple() -> None:
    cfg = PreferenceTrainConfig(
        model="m", dataset="d", adapter="lora", adapter_rank=8, adapter_targets=["q_proj"]
    )
    PreferenceTrainer(cfg)
    assert cfg.adapter_targets == ("q_proj",)
    assert isinstance(cfg.adapter_targets, tuple)


def test_pref_adapter_lora_minimal_valid_config_constructs() -> None:
    cfg = PreferenceTrainConfig(
        model="m",
        dataset="d",
        adapter="lora",
        adapter_rank=16,
        adapter_alpha=32.0,
        adapter_targets=("q_proj", "v_proj"),
        adapter_dropout=0.0,
    )
    trainer = PreferenceTrainer(cfg)
    assert trainer.config.adapter == "lora"
