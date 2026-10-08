"""CPU legs for the resumable-state tracker of the Megatron RL lane.

The collective save/load runs only under megatron-core on GPUs; what CAN be pinned
on CPU is the part a resume decision rests on: which directory is the newest
COMPLETE state, and that a half-written or mislabelled one is never resumed.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from foundationscale.rl.megatron.resume import (
    LATEST_NAME,
    latest_training_state,
    state_dir_for_step,
)


def test_no_tracker_means_nothing_to_resume(tmp_path: Path) -> None:
    state_dir_for_step(tmp_path, 3).mkdir(parents=True)  # a save that never finished
    assert latest_training_state(tmp_path) is None


def test_tracker_names_the_latest_complete_step(tmp_path: Path) -> None:
    for step in (1, 3):
        state_dir_for_step(tmp_path, step).mkdir(parents=True)
    (tmp_path / LATEST_NAME).write_text("3\n", encoding="utf-8")
    assert latest_training_state(tmp_path) == tmp_path / "step_000003"


def test_tracker_pointing_at_a_missing_directory_is_refused(tmp_path: Path) -> None:
    (tmp_path / LATEST_NAME).write_text("7\n", encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="step 7"):
        latest_training_state(tmp_path)


def test_a_garbled_tracker_is_refused_not_ignored(tmp_path: Path) -> None:
    (tmp_path / LATEST_NAME).write_text("seven\n", encoding="utf-8")
    with pytest.raises(ValueError, match="expected a step number"):
        latest_training_state(tmp_path)


def test_module_imports_without_megatron() -> None:
    import foundationscale.rl.megatron.resume as resume

    assert callable(resume.save_training_state) and callable(resume.load_training_state)


# -- the collective save/load path, driven through a stand-in dist_checkpointing ----


class _Chunk:
    """A bare mcore-like chunk: keys under a prefix, strict load records its input."""

    def __init__(self, prefix: str) -> None:
        self.prefix = prefix
        self.loaded: dict[str, object] | None = None
        self.strict: bool | None = None

    def sharded_state_dict(self, metadata: dict[str, object]) -> dict[str, object]:
        assert metadata["distrib_optim_sharding_type"] == "fully_reshardable"
        return {f"{self.prefix}.weight": f"{self.prefix}-shard"}

    def load_state_dict(self, state: dict[str, object], strict: bool) -> None:
        self.loaded, self.strict = state, strict


class _DDP:
    def __init__(self, module: object) -> None:
        self.module = module


class _Optimizer:
    def __init__(self) -> None:
        self.loaded: object = None
        self.calls: list[dict[str, object]] = []

    def sharded_state_dict(self, model_state: dict[str, object], **kwargs: object) -> object:
        self.calls.append({"keys": sorted(model_state), **kwargs})
        return {"exp_avg": "moments"}

    def load_state_dict(self, state: object) -> None:
        self.loaded = state


class _PG:
    dp_cp = "dp-cp-group"


@pytest.fixture
def fake_dcp(monkeypatch: pytest.MonkeyPatch) -> dict[str, dict[str, object]]:
    """Install a megatron.core.dist_checkpointing stand-in that stores by path."""
    import sys
    import types

    store: dict[str, dict[str, object]] = {}

    def save(state: dict[str, object], path: str, content_metadata: object = None) -> None:
        assert "dp_cp_group" not in (content_metadata or {})
        store[path] = dict(state)

    def load(state: dict[str, object], path: str) -> dict[str, object]:
        saved = store[path]
        assert set(saved) == set(state)
        return saved

    dcp = types.ModuleType("megatron.core.dist_checkpointing")
    dcp.save = save  # type: ignore[attr-defined]
    dcp.load = load  # type: ignore[attr-defined]
    core = types.ModuleType("megatron.core")
    core.dist_checkpointing = dcp  # type: ignore[attr-defined]
    meg = types.ModuleType("megatron")
    meg.core = core  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "megatron", meg)
    monkeypatch.setitem(sys.modules, "megatron.core", core)
    monkeypatch.setitem(sys.modules, "megatron.core.dist_checkpointing", dcp)
    return store


def test_save_then_resume_restores_model_optimizer_and_step(
    tmp_path: Path, fake_dcp: dict[str, dict[str, object]]
) -> None:
    from foundationscale.rl.megatron.resume import resume_if_present, save_training_state

    chunk, optimizer = _Chunk("decoder"), _Optimizer()
    path = save_training_state(tmp_path, _DDP(chunk), optimizer, _PG(), step=3)
    assert path == tmp_path / "step_000003"
    assert (tmp_path / LATEST_NAME).read_text(encoding="utf-8").strip() == "3"
    assert optimizer.calls[0]["keys"] == ["model"] and "is_loading" not in optimizer.calls[0]

    fresh_chunk, fresh_opt = _Chunk("decoder"), _Optimizer()
    start, latest = resume_if_present(tmp_path, _DDP(fresh_chunk), fresh_opt, _PG())
    assert (start, latest) == (4, path)
    assert fresh_chunk.loaded == {"decoder.weight": "decoder-shard"}
    assert fresh_chunk.strict is True
    assert fresh_opt.loaded == {"exp_avg": "moments"}
    assert fresh_opt.calls[0]["is_loading"] is True


def test_virtual_pipeline_chunks_save_under_indexed_model_keys(
    tmp_path: Path, fake_dcp: dict[str, dict[str, object]]
) -> None:
    from foundationscale.rl.megatron.resume import load_training_state, save_training_state

    chunks = [_Chunk("a"), _Chunk("b")]
    save_training_state(tmp_path, chunks, _Optimizer(), _PG(), step=0, extra={"note": "x"})
    assert {"model0", "model1"} <= set(fake_dcp[str(tmp_path / "step_000000")])
    fresh = [_Chunk("a"), _Chunk("b")]
    restored = load_training_state(tmp_path / "step_000000", fresh, _Optimizer(), _PG())
    assert restored == {"step": 0, "note": "x"}
    assert fresh[1].loaded == {"b.weight": "b-shard"}


def test_resaving_the_latest_step_is_refused(
    tmp_path: Path, fake_dcp: dict[str, dict[str, object]]
) -> None:
    from foundationscale.rl.megatron.resume import save_training_state

    save_training_state(tmp_path, _Chunk("m"), _Optimizer(), _PG(), step=2)
    with pytest.raises(FileExistsError, match="already the latest"):
        save_training_state(tmp_path, _Chunk("m"), _Optimizer(), _PG(), step=2)


def test_a_checkpoint_without_loop_state_is_not_a_training_state(
    tmp_path: Path, fake_dcp: dict[str, dict[str, object]]
) -> None:
    from foundationscale.rl.megatron.resume import load_training_state, save_training_state

    save_training_state(tmp_path, _Chunk("m"), _Optimizer(), _PG(), step=1)
    fake_dcp[str(tmp_path / "step_000001")]["fs_rl_state"] = None
    with pytest.raises(ValueError, match="not a training-state checkpoint"):
        load_training_state(tmp_path / "step_000001", _Chunk("m"), _Optimizer(), _PG())


def test_no_root_or_no_state_starts_at_zero(tmp_path: Path) -> None:
    from foundationscale.rl.megatron.resume import resume_if_present

    assert resume_if_present("", None, None, None) == (0, None)
    assert resume_if_present(tmp_path, None, None, None) == (0, None)


def test_state_save_due_is_periodic_and_leaves_the_final_save_alone() -> None:
    from foundationscale.rl.megatron.resume import state_save_due

    assert [s for s in range(10) if state_save_due(s, 3, 10)] == [2, 5, 8]
    assert not state_save_due(9, 5, 10)  # step 9 is the final save, written separately
    assert not any(state_save_due(s, 0, 10) for s in range(10))
