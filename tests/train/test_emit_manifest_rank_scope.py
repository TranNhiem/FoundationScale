"""Run-manifest emission: one writer per run, and the write is atomic.

The run manifest lives at run scope -- one file per run, not per rank. The
pre-fix emitter ran on every rank with a plain ``Path.write_text``: under
torchrun every rank truncated and rewrote the same ``run_manifest.json``, so
a save gate reading beside a checkpoint could open the file mid-truncate on
one rank while another rank was mid-rewrite. A torn parse RAISES in the gate
reader by design, which is loud -- but the torn read itself was a property of
the producer, not of the reader's strictness.

Two guarantees, each pinned by its own test below:

- rank 0 alone writes; every other rank abstains loudly (a mark line, never
  a file)
- the write is temp-file + fsync + ``os.replace``: successive stages
  (``train`` then the terminal stage) legitimately overwrite, so no-clobber
  is the wrong policy, but a reader must never catch a half-written manifest
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from foundationscale.train import loop


def _cfg(tmp_path: Path) -> loop.TrainConfig:
    """A config sufficient for emission; no stage of ``train()`` itself runs."""
    return loop.TrainConfig(
        model="fake-model",
        dataset="fake-dataset",
        output_dir=tmp_path / "out",
        nodes=1,
        gpus_per_node=1,
        profile_name="synthetic-profile",
        max_steps=2,
        save_interval=1,
    )


@pytest.fixture(autouse=True)
def _no_env_rank(monkeypatch: pytest.MonkeyPatch) -> None:
    """Tests start unranked: torchrun's stamp enters per-test, explicitly."""
    monkeypatch.delenv("RANK", raising=False)


def test_rank_zero_writes_parseable_manifest(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)

    path = loop._emit_manifest(cfg, stage="train")

    assert path == Path(cfg.output_dir) / loop.MANIFEST_NAME
    assert path.exists()
    json.loads(path.read_text(encoding="utf-8"))
    # The write protocol leaves no scratch file behind on success.
    assert [p.name for p in path.parent.iterdir()] == [loop.MANIFEST_NAME]


def test_nonzero_rank_abstains_and_writes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("RANK", "1")
    cfg = _cfg(tmp_path)

    path = loop._emit_manifest(cfg, stage="train")

    out = capsys.readouterr().out
    assert "[fs:train:manifest]" in out
    assert "abstains as a non-writer" in out
    # Not even the output directory may appear: the file is not this rank's.
    assert not Path(cfg.output_dir).exists()
    assert not path.exists()


def test_nonzero_rank_abstention_names_writer_and_stage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("RANK", "3")
    cfg = _cfg(tmp_path)

    path = loop._emit_manifest(cfg, stage="done")

    out = capsys.readouterr().out
    assert "rank 0 writes this file" in out
    assert "rank 3 abstains as a non-writer" in out
    assert "run manifest (done)" in out
    assert str(path) in out


def test_non_numeric_rank_stamp_falls_back_to_writer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("RANK", "unparseable")

    path = loop._emit_manifest(_cfg(tmp_path), stage="train")

    assert path.exists()


def test_live_collective_overrides_env_stamp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An initialised collective outranks the launcher's RANK stamp."""
    import torch.distributed as real_dist

    monkeypatch.setenv("RANK", "0")
    monkeypatch.setattr(real_dist, "is_available", lambda: True)
    monkeypatch.setattr(real_dist, "is_initialized", lambda: True)
    monkeypatch.setattr(real_dist, "get_rank", lambda: 2)

    path = loop._emit_manifest(_cfg(tmp_path), stage="train")

    assert not path.exists()


def test_every_write_goes_through_atomic_replace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Stage overwrites stay atomic: temp file, fsync, os.replace."""
    cfg = _cfg(tmp_path)
    replaces: list[tuple[Path, Path]] = []
    real_replace = Path.replace

    def _spy_replace(self: Path, target: Path) -> Path:
        replaces.append((self, Path(target)))
        return real_replace(self, target)

    monkeypatch.setattr(Path, "replace", _spy_replace)

    first = loop._emit_manifest(cfg, stage="train").read_bytes()
    second = loop._emit_manifest(cfg, stage="done").read_bytes()

    # Both stages published through Path.replace with a pid-marked temp name.
    assert len(replaces) == 2
    for src, dst in replaces:
        assert src.name.startswith(f".{loop.MANIFEST_NAME}.tmp.")
        assert dst.name == loop.MANIFEST_NAME
    # The terminal stage legitimately overwrote the interim one, atomically.
    assert first != second
    json.loads(second.decode("utf-8"))
