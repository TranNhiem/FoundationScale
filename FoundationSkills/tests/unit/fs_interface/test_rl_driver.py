
"""Tests for the fskills-rl driver.

Dry-run tests for a runnable (dr_grpo) and a non-runnable (dpo) algorithm use
the REAL installed foundationscale -- objective resolution needs no model, no
tokenizer, no GPU. The full-run paths are driven through a fake
foundationscale.rl.trainer injected into sys.modules (a real run needs a GPU).
"""
from __future__ import annotations

import dataclasses
import json
import sys
import types
from pathlib import Path

import pytest

from foundationskills.interfaces.fs import rl_driver


def write_config(tmp_path: Path, **cfg) -> Path:
    base = {"model": "probe/none", "dataset": "probe.jsonl", "algorithm": "dr_grpo"}
    base.update(cfg)
    path = tmp_path / "rl_config.json"
    path.write_text(json.dumps(base), encoding="utf-8")
    return path


def test_unknown_config_key_refused(tmp_path):
    cfg = write_config(tmp_path, bogus_key=1)
    assert rl_driver.main(["--config", str(cfg)]) == 96


def test_dry_run_runnable_real_fs(tmp_path):
    cfg = write_config(tmp_path, algorithm="dr_grpo", output_dir=str(tmp_path / "out"))
    assert rl_driver.main(["--config", str(cfg), "--dry-run"]) == 0


def test_dry_run_non_runnable_real_fs(tmp_path):
    cfg = write_config(tmp_path, algorithm="dpo", output_dir=str(tmp_path / "out"))
    assert rl_driver.main(["--config", str(cfg), "--dry-run"]) == 96


# --------------------------------------------------------------------------
# Fake foundationscale.rl.trainer for full-run paths (no model, no GPU).
# --------------------------------------------------------------------------


@dataclasses.dataclass
class _FakeStepReport:
    step: int
    loss: float = 0.5


class _FakeTrainerRefusal(RuntimeError):
    pass


def _install_fake(monkeypatch, *, reports=(), refuse=None, expose_model=False):
    module = types.ModuleType("foundationscale.rl.trainer")

    class RLTrainConfig:
        def __init__(self, **kw):
            self.__dict__.update(kw)
            self.algorithm = kw.get("algorithm", "dr_grpo")
            self.max_steps = kw.get("max_steps", 10)

    class RLTrainer:
        def __init__(self, config):
            self.config = config
            if expose_model:
                self.model = _FakeModel()

        def _resolve_objective(self):
            if self.config.algorithm == "fake-refuse":
                raise _FakeTrainerRefusal("fake objective refusal")
            return object()

        def run(self):
            if refuse == "trainer_refusal":
                raise _FakeTrainerRefusal("fake run refusal")
            if refuse == "system_exit_96":
                raise SystemExit(96)
            return list(reports)

    class _FakeModel:
        def __init__(self):
            self.saved_to = None

        def save_pretrained(self, path):
            Path(path).mkdir(parents=True, exist_ok=True)
            (Path(path) / "model.safetensors").write_text("fake", encoding="utf-8")

    module.RLTrainConfig = RLTrainConfig
    module.RLTrainer = RLTrainer
    module.TrainerRefusal = _FakeTrainerRefusal
    for name in ("foundationscale", "foundationscale.rl", "foundationscale.rl.trainer"):
        if name == "foundationscale.rl.trainer":
            monkeypatch.setitem(sys.modules, name, module)
        else:
            parent = types.ModuleType(name)
            monkeypatch.setitem(sys.modules, name, parent)


def test_run_pass_writes_reports_and_manifest(tmp_path, monkeypatch):
    _install_fake(monkeypatch, reports=[_FakeStepReport(0), _FakeStepReport(1)])
    out = tmp_path / "out"
    cfg = write_config(tmp_path, max_steps=2, output_dir=str(out))
    assert rl_driver.main(["--config", str(cfg)]) == 0
    reports = json.loads((out / "rl_reports.json").read_text(encoding="utf-8"))
    assert len(reports["reports"]) == 2
    assert reports["reports"][0]["step"] == 0
    manifest = json.loads((out / "fskills_rl_manifest.json").read_text(encoding="utf-8"))
    assert manifest["steps_measured"] == 2
    assert manifest["steps_unmeasured"] == 0
    assert manifest["config"]["algorithm"] == "dr_grpo"
    assert manifest["status"] == "PASS"


def test_run_zero_measured_steps_unmeasured(tmp_path, monkeypatch):
    _install_fake(monkeypatch, reports=[])
    out = tmp_path / "out"
    cfg = write_config(tmp_path, max_steps=4, output_dir=str(out))
    assert rl_driver.main(["--config", str(cfg)]) == 95
    manifest = json.loads((out / "fskills_rl_manifest.json").read_text(encoding="utf-8"))
    assert manifest["steps_measured"] == 0
    assert manifest["steps_unmeasured"] == 4


def test_run_trainer_refusal_is_96(tmp_path, monkeypatch):
    _install_fake(monkeypatch, refuse="trainer_refusal")
    cfg = write_config(tmp_path, output_dir=str(tmp_path / "out"))
    assert rl_driver.main(["--config", str(cfg)]) == 96


def test_run_system_exit_96_passthrough(tmp_path, monkeypatch):
    _install_fake(monkeypatch, refuse="system_exit_96")
    cfg = write_config(tmp_path, output_dir=str(tmp_path / "out"))
    assert rl_driver.main(["--config", str(cfg)]) == 96


def test_save_final_without_model_attribute_is_unmeasured(tmp_path, monkeypatch):
    _install_fake(monkeypatch, reports=[_FakeStepReport(0)], expose_model=False)
    out = tmp_path / "out"
    cfg = write_config(tmp_path, max_steps=1, output_dir=str(out), save_final=True)
    assert rl_driver.main(["--config", str(cfg)]) == 95
    manifest = json.loads((out / "fskills_rl_manifest.json").read_text(encoding="utf-8"))
    assert manifest["checkpoint"] == "checkpoint: UNMEASURED (RLTrainer exposes no model)"
    assert not (out / "final").exists()


def test_save_final_with_exposed_model_saves(tmp_path, monkeypatch):
    _install_fake(monkeypatch, reports=[_FakeStepReport(0)], expose_model=True)
    out = tmp_path / "out"
    cfg = write_config(tmp_path, max_steps=1, output_dir=str(out), save_final=True)
    assert rl_driver.main(["--config", str(cfg)]) == 0
    assert (out / "final" / "model.safetensors").exists()
    manifest = json.loads((out / "fskills_rl_manifest.json").read_text(encoding="utf-8"))
    assert manifest["checkpoint"].startswith("saved:")


def test_dry_run_fake_refusal(tmp_path, monkeypatch):
    _install_fake(monkeypatch)
    cfg = write_config(tmp_path, algorithm="fake-refuse", output_dir=str(tmp_path / "out"))
    assert rl_driver.main(["--config", str(cfg), "--dry-run"]) == 96
