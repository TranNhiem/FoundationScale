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
