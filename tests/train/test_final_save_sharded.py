"""#544: the final save under SHARDED_STATE_DICT goes through accelerate, and a missing
final directory is reported, not crashed on."""

from __future__ import annotations

import types
from pathlib import Path

import accelerate.utils

from foundationscale.train import loop


def _trainer(should_save: bool, saved: list[str]):
    config = types.SimpleNamespace(save_pretrained=lambda d: saved.append("config"))
    model = types.SimpleNamespace(config=config)
    accelerator = types.SimpleNamespace(
        state=types.SimpleNamespace(fsdp_plugin="plugin"), unwrap_model=lambda m: m
    )
    return types.SimpleNamespace(
        accelerator=accelerator,
        model=model,
        args=types.SimpleNamespace(should_save=should_save),
        processing_class=types.SimpleNamespace(save_pretrained=lambda d: saved.append("tok")),
    )


def test_sharded_final_save_calls_accelerate_on_every_rank(monkeypatch, tmp_path: Path) -> None:
    calls = []
    monkeypatch.setattr(
        accelerate.utils,
        "save_fsdp_model",
        lambda plugin, acc, model, out: calls.append((plugin, out)),
    )
    for should_save in (True, False):
        saved: list[str] = []
        final = tmp_path / f"final_{should_save}"
        loop._save_final_fsdp_sharded(_trainer(should_save, saved), final)
        assert final.is_dir()
        assert saved == (["config", "tok"] if should_save else [])
    assert [c[0] for c in calls] == ["plugin", "plugin"]


def test_dir_listing_reports_absent_directory(tmp_path: Path) -> None:
    assert loop._dir_listing(tmp_path / "final") == "<directory absent>"
    (tmp_path / "a").write_text("x")
    assert loop._dir_listing(tmp_path) == ["a"]
