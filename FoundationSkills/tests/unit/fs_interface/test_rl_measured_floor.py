"""min_measured_fraction: a declared floor on measured/max_steps turns a saturated RL run into 95.

Measured 2026-10-09 (gemma-4-E4B-it, ARC-C, dr_grpo): 5 of 32 steps measured -- every other group
was all-correct, so FS marked it UNMEASURED -- and the driver still exited 0.
"""
from __future__ import annotations

import json

import pytest

from foundationskills.interfaces.fs import rl_driver
from foundationskills.interfaces.fs.emit_rl import emit_rl

try:  # the relative-import approach test_skill_m5b_flow.py uses
    from .test_emit_rl import make_caps, rl_dataset, rl_stage
    from .test_rl_driver import _FakeStepReport, _install_fake, write_config
except ImportError:  # rootdir-less layout: pytest prepends this directory to sys.path
    from test_emit_rl import make_caps, rl_dataset, rl_stage  # type: ignore[no-redef]
    from test_rl_driver import (  # type: ignore[no-redef]
        _FakeStepReport, _install_fake, write_config)


def _run(tmp_path, monkeypatch, measured, max_steps, **cfg):
    _install_fake(monkeypatch, reports=[_FakeStepReport(i) for i in range(measured)])
    out = tmp_path / "out"
    rc = rl_driver.main(["--config", str(write_config(tmp_path, max_steps=max_steps, output_dir=str(out), **cfg))])
    manifest_path = out / "fskills_rl_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else None
    return rc, manifest


def test_below_the_floor_is_unmeasured(tmp_path, monkeypatch, capsys):
    rc, manifest = _run(tmp_path, monkeypatch, 5, 32, min_measured_fraction=0.5)
    assert rc == 95
    assert manifest["status"].startswith("UNMEASURED: measured fraction 5/32")
    assert manifest["measured_fraction"] == round(5 / 32, 4)
    assert manifest["min_measured_fraction"] == 0.5
    assert "difficulty_filter" in capsys.readouterr().out


def test_at_the_floor_passes(tmp_path, monkeypatch):
    rc, manifest = _run(tmp_path, monkeypatch, 16, 32, min_measured_fraction=0.5)
    assert rc == 0
    assert manifest["status"] == "PASS"


def test_no_floor_keeps_the_old_verdict(tmp_path, monkeypatch):
    rc, manifest = _run(tmp_path, monkeypatch, 5, 32)
    assert rc == 0
    assert manifest["min_measured_fraction"] is None
    assert manifest["measured_fraction"] == round(5 / 32, 4)


@pytest.mark.parametrize("bad", [0, 1.5, -0.1, "half"])
def test_floor_outside_unit_interval_refused(tmp_path, monkeypatch, bad):
    rc, manifest = _run(tmp_path, monkeypatch, 5, 32, min_measured_fraction=bad)
    assert rc == 96
    assert manifest is None


def test_emit_rl_passes_a_declared_floor(tmp_path):
    spec = emit_rl(rl_stage(min_measured_fraction=0.25), dataset=rl_dataset(tmp_path), model="models/g4",
                   output_dir=str(tmp_path / "out"), caps=make_caps(), run_name="fskills-rl1")
    assert spec["rl_config"]["min_measured_fraction"] == 0.25
    assert not any("min_measured_fraction" in n for n in spec["notes"])


def test_emit_rl_notes_an_undeclared_floor(tmp_path):
    spec = emit_rl(rl_stage(), dataset=rl_dataset(tmp_path), model="models/g4",
                   output_dir=str(tmp_path / "out"), caps=make_caps(), run_name="fskills-rl1")
    assert "min_measured_fraction" not in spec["rl_config"]
    assert any("min_measured_fraction not declared" in n for n in spec["notes"])


@pytest.mark.parametrize("bad", [True, "0.5"])
def test_floor_must_be_a_declared_number_not_a_coercible_one(tmp_path, monkeypatch, bad):
    rc, manifest = _run(tmp_path, monkeypatch, 32, 32, min_measured_fraction=bad)
    assert rc == 96  # float(True) == 1.0 and float("0.5") == 0.5 must not slip through
    assert manifest is None


def test_floor_without_a_denominator_is_unmeasured(tmp_path, monkeypatch):
    rc, manifest = _run(tmp_path, monkeypatch, 3, 0, min_measured_fraction=0.5)
    assert rc == 95
    assert manifest["status"].startswith("UNMEASURED: declared floor 0.5 cannot be evaluated")
