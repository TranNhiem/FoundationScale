"""Tests for the runtime-evidence effective topology and the unwired-degree refusal.

House rules honoured throughout: every test asserts REACHED-SITE (the returned
object's type, or the finding's ``code``) separately from OUTCOME (the value);
every arm can come out different; ``os.environ`` is restored by monkeypatch,
never by hand; nothing imports torch at module scope.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Any

import pytest

from foundationscale.topology import (
    Finding,
    Severity,
    Topology,
    blocking,
    declared_vs_effective,
)
from foundationscale.train.loop import (
    EXIT_PASS,
    EXIT_REFUSE,
    TrainConfig,
    _effective_topology,
    train,
)


def _cfg(tmp_path: Path, **overrides: Any) -> TrainConfig:
    """A valid all-degrees-1 config; overrides name the single axis under test."""
    fields: dict[str, Any] = {
        "model": "fake-model",
        "dataset": "fake-dataset",
        "output_dir": tmp_path,
        "nodes": 1,
        "gpus_per_node": 8,
        "profile_name": "local-single-node",
        "dp": 8,
    }
    fields.update(overrides)
    return TrainConfig(**fields)


def test_honest_path_effective_matches_declared(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """WORLD_SIZE=8 + LOCAL_WORLD_SIZE=8 + cfg all-1: zero BLOCK findings."""
    monkeypatch.setenv("WORLD_SIZE", "8")
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "8")
    cfg = _cfg(tmp_path)
    declared = Topology(
        dp=cfg.dp,
        tp=cfg.tp,
        pp=cfg.pp,
        ep=cfg.ep,
        cp=cfg.cp,
        nodes=cfg.nodes,
        gpus_per_node=cfg.gpus_per_node,
    )
    effective = _effective_topology(cfg)
    # REACHED-SITE: the runtime evidence formed a Topology, not a Finding.
    assert isinstance(effective, Topology)
    # OUTCOME: dp=8, nodes=1, gpus_per_node=8, and the comparison is clean.
    assert (effective.dp, effective.nodes, effective.gpus_per_node) == (8, 1, 8)
    findings = declared_vs_effective(declared, effective)
    assert findings  # the comparison ran; an empty list would be UNMEASURED
    assert blocking(findings) == []
    assert any(f.code == "topology.effective_matches_declared" for f in findings)


class _TrainerTripwire:
    """A transformers.Trainer double that fails the test if constructed.

    This is what distinguishes "train() refused" from "train() ran and
    happened to return 96": any regression that lets execution reach the
    Trainer raises here instead of passing.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise AssertionError(
            "transformers.Trainer was constructed: train() ran the model "
            "path instead of refusing the unwired degree"
        )


class _FakeTrainingArguments:
    """Explicit signature: train() introspects it and refuses unknown kwargs."""

    def __init__(
        self,
        output_dir: str | None = None,
        max_steps: int | None = None,
        per_device_train_batch_size: int | None = None,
        learning_rate: float | None = None,
        save_strategy: str | None = None,
        save_steps: int | None = None,
        seed: int | None = None,
        logging_steps: int | None = None,
        report_to: list[Any] | None = None,
        ddp_find_unused_parameters: bool | None = None,
        bf16: bool | None = None,
        fp16: bool | None = None,
        save_safetensors: bool | None = None,
    ) -> None:
        self.output_dir = output_dir
        self.max_steps = max_steps
        self.per_device_train_batch_size = per_device_train_batch_size
        self.learning_rate = learning_rate
        self.save_strategy = save_strategy
        self.save_steps = save_steps
        self.seed = seed
        self.logging_steps = logging_steps
        self.report_to = report_to
        self.ddp_find_unused_parameters = ddp_find_unused_parameters
        self.bf16 = bf16
        self.fp16 = fp16
        self.save_safetensors = save_safetensors


class _FakeModel:
    """The smallest model the thin path can derive a checkpoint declaration from."""

    config: None = None  # no tie_word_embeddings, no expert-count keys

    def state_dict(self) -> dict[str, Any]:
        return {}


class _FakeAutoModel:
    @classmethod
    def from_pretrained(cls, _model: str) -> _FakeModel:
        return _FakeModel()


class _FakeTokenizer:
    pad_token: str | None = None
    eos_token = "</s>"


class _FakeAutoTokenizer:
    @classmethod
    def from_pretrained(cls, _model: str) -> _FakeTokenizer:
        return _FakeTokenizer()


class _FakeDataset:
    column_names = ["text"]

    def map(self, *args: Any, **kwargs: Any) -> _FakeDataset:
        return self

    def __len__(self) -> int:
        return 1


class _FakeCollator:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass


def _install_fake_training_stack(monkeypatch: pytest.MonkeyPatch) -> None:
    """Shadow torch/datasets/transformers in sys.modules for one test.

    The doubles exist so that a REGRESSED guard cannot pass by accident: with
    them installed, a train() that fails to refuse flows through the fake
    model path all the way to the Trainer tripwire, instead of returning 96
    for an unrelated reason (a missing optional dependency on the test host).
    """

    def _manual_seed(_seed: int) -> None:
        return None

    torch = types.ModuleType("torch")
    torch.manual_seed = _manual_seed
    datasets = types.ModuleType("datasets")
    datasets.load_dataset = lambda _ref: {"train": _FakeDataset()}
    transformers = types.ModuleType("transformers")
    transformers.Trainer = _TrainerTripwire
    transformers.TrainingArguments = _FakeTrainingArguments
    transformers.AutoModelForCausalLM = _FakeAutoModel
    transformers.AutoTokenizer = _FakeAutoTokenizer
    transformers.DataCollatorForLanguageModeling = _FakeCollator
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "datasets", datasets)
    monkeypatch.setitem(sys.modules, "transformers", transformers)


@pytest.mark.parametrize("degree", ["tp", "pp", "ep", "cp"])
def test_unwired_degree_refuses_before_any_trainer(
    degree: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Each degree named INDEPENDENTLY: tp=2 alone refuses, ep=2 alone refuses."""
    # WORLD_SIZE unset so that, without the refusal, the declared-vs-effective
    # comparison is skipped on the driver and execution would flow all the way
    # to the Trainer tripwire -- the leg can come out different.
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    monkeypatch.delenv("LOCAL_WORLD_SIZE", raising=False)
    _install_fake_training_stack(monkeypatch)
    # dp=4 keeps the DECLARED topology constructible (4 x 2 = 1 node x 8
    # GPUs), so the refusal under test -- not Topology.__post_init__ -- is
    # what stops the run.
    cfg = _cfg(tmp_path, dp=4, **{degree: 2})
    rc = train(cfg)
    out = capsys.readouterr().out
    # REACHED-SITE: the refusal marker fired and names THIS degree by value.
    assert "fs:train:refuse" in out
    assert f"{degree}=2" in out
    # OUTCOME: exit 96, the message says WHY (no Trainer kwarg carries the
    # degree), and the tripwire above proves no Trainer was constructed.
    assert rc == EXIT_REFUSE
    assert "transformers.Trainer" in out


def test_all_degrees_one_does_not_refuse(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The all-degrees-1 arm must NOT refuse, or the legs prove nothing."""
    monkeypatch.setenv("WORLD_SIZE", "8")
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "8")
    cfg = _cfg(tmp_path, dry_run=True)
    rc = train(cfg)
    # REACHED-SITE and OUTCOME: the full validation prologue ran (dry_run)
    # and passed -- no refusal, no block. Without this pairing the legs
    # cannot tell a working guard from a guard that refuses everything.
    assert rc == EXIT_PASS


@pytest.mark.parametrize("axis", ["tp", "pp", "ep", "cp", "gpus_per_node"])
def test_effective_topology_never_echoes_cfg(
    axis: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Pre-fix, ALL FIVE legs are RED: these axes were sourced from cfg."""
    # These five legs are the reason the entry-point refusal is not
    # sufficient on its own: gpus_per_node is not a degree, no refusal
    # covers it, and it was cfg-sourced too. _effective_topology is called
    # DIRECTLY here, below the entry-point refusal, so the unit stays
    # reachable.
    monkeypatch.setenv("WORLD_SIZE", "8")
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "8")
    cfg = _cfg(tmp_path, **{axis: 4})
    effective = _effective_topology(cfg)
    # REACHED-SITE: a Topology came back, not a Finding and not None.
    assert isinstance(effective, Topology)
    # OUTCOME: the runtime value (LOCAL_WORLD_SIZE=8, degrees pinned to 1),
    # never the cfg echo of 4.
    assert getattr(cfg, axis) == 4  # the cfg axis really did disagree
    expected = 8 if axis == "gpus_per_node" else 1
    assert getattr(effective, axis) == expected


def test_missing_local_world_size_is_unmeasured_not_fabricated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """LOCAL_WORLD_SIZE unset: an UNMEASURED Finding, never a cfg default."""
    monkeypatch.setenv("WORLD_SIZE", "4")
    monkeypatch.delenv("LOCAL_WORLD_SIZE", raising=False)
    # cfg carries the pre-fix fabrication inputs (tp=8, gpus_per_node=8
    # against a 4-rank world) so the regression arm is the measured one.
    cfg = _cfg(tmp_path, tp=8)
    effective = _effective_topology(cfg)
    # REACHED-SITE: a Finding, identified by its code.
    assert isinstance(effective, Finding)
    assert effective.code == "train.local_world_size_unset"
    assert effective.severity is Severity.BLOCK
    # OUTCOME: no Topology is returned. Pre-fix, these exact inputs returned
    # Topology(dp=1, tp=8, nodes=1, gpus_per_node=8) -- a 4-rank run recorded
    # as an 8-GPU topology. That object is gone, not relabelled.
    assert not isinstance(effective, Topology)


def test_ragged_world_is_a_finding_not_a_topology(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """WORLD_SIZE=6 % LOCAL_WORLD_SIZE=4 != 0: ragged evidence, not a Topology."""
    monkeypatch.setenv("WORLD_SIZE", "6")
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "4")
    effective = _effective_topology(_cfg(tmp_path))
    # REACHED-SITE: a Finding, identified by its code.
    assert isinstance(effective, Finding)
    assert effective.code == "train.ragged_world"
    # OUTCOME: no ceil-padded Topology is minted over the ragged evidence.
    assert not isinstance(effective, Topology)


def test_driver_process_returns_none(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """WORLD_SIZE unset: the driver gets None -- specifically None, not falsy."""
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    effective = _effective_topology(_cfg(tmp_path))
    # REACHED-SITE and OUTCOME in one: `is None`, not `not effective` -- a
    # Finding or a Topology must not satisfy this assertion by accident.
    assert effective is None
