#!/usr/bin/env python3
"""Executable controls for the reference FoundationScale checkpoint adjudicator."""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

# The suite uses @pytest.mark.parametrize but did not import pytest, so all 14 tests died at
# COLLECTION with NameError -- zero tests ran while the emitter reported 9/9 gates green. The
# emitter's C5 counted 14 test FUNCTIONS, which is a static property of the text; nothing
# executed them. That is the all([]) shape in its most literal form: a suite that cannot be
# collected has measured nothing, and counting its functions measures the counting.
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import fs_ckpt_adjudicator as adjudicator

FORMAT = "rank-local-sharded-v1"


def build_checkpoint(
    directory: Path,
    *,
    world_size: int = 8,
    global_step: int = 120,
    omitted_ranks: tuple[int, ...] = (),
    zero_length_rank: int | None = None,
    extra_ranks: tuple[int, ...] = (),
    overrides: dict[str, object] | None = None,
) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, object] = {
        "format": FORMAT,
        "global_step": global_step,
        "world_size": world_size,
        "rank_payload_count": world_size,
        "expected_rank_payload_count": world_size,
        "fixed_loss_before_save": 1.25,
    }
    if overrides:
        manifest.update(overrides)
    (directory / "manifest.json").write_text(
        json.dumps(manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )

    for rank in range(world_size):
        if rank in omitted_ranks:
            continue
        shard = directory / f"rank-{rank:05d}.pt"
        if rank == zero_length_rank:
            shard.write_bytes(b"")
        else:
            shard.write_bytes(bytes([(rank % 254) + 1]) * 16)

    for rank in extra_ranks:
        (directory / f"rank-{rank:05d}.pt").write_bytes(b"unexpected-rank-payload")
    return directory


def invoke_checkpoint(directory: Path, tmp_path: Path) -> int:
    out_dir = tmp_path / "launcher-out"
    out_dir.mkdir(exist_ok=True)
    return adjudicator.main([str(directory), "after-save", str(out_dir)])


def test_must_pass_well_formed_eight_rank_checkpoint_returns_zero(tmp_path: Path) -> None:
    checkpoint = build_checkpoint(tmp_path / "step_120")

    assert invoke_checkpoint(checkpoint, tmp_path) == 0


def test_must_pass_reports_measured_denominators(tmp_path: Path, capsys) -> None:
    checkpoint = build_checkpoint(tmp_path / "checkpoint_120")

    assert invoke_checkpoint(checkpoint, tmp_path) == 0
    output = capsys.readouterr().out
    assert "A5a expected rank shard set is complete: PASS (present 8/8 expected ranks" in output
    assert "A6 every expected rank shard is a non-empty regular file: PASS " in output
    assert "regular non-empty shards 8/8" in output
    assert "VERDICT 0 PASS" in output


def test_must_fire_missing_rank_shard_returns_96(tmp_path: Path) -> None:
    checkpoint = build_checkpoint(tmp_path / "step_120", omitted_ranks=(3,))

    assert invoke_checkpoint(checkpoint, tmp_path) == 96


def test_must_fire_zero_length_rank_shard_returns_96(tmp_path: Path) -> None:
    checkpoint = build_checkpoint(tmp_path / "step_120", zero_length_rank=4)

    assert invoke_checkpoint(checkpoint, tmp_path) == 96


def test_must_fire_extra_unexpected_rank_shard_returns_96(tmp_path: Path) -> None:
    checkpoint = build_checkpoint(tmp_path / "step_120", extra_ranks=(8,))

    assert invoke_checkpoint(checkpoint, tmp_path) == 96


def test_must_fire_rank_payload_count_less_than_expected_returns_96(tmp_path: Path) -> None:
    # Every file is present, so only the writer's collective save count can expose this loss.
    checkpoint = build_checkpoint(
        tmp_path / "step_120",
        overrides={"rank_payload_count": 7},
    )

    assert invoke_checkpoint(checkpoint, tmp_path) == 96


def test_must_fire_directory_step_disagreeing_with_manifest_returns_96(tmp_path: Path) -> None:
    checkpoint = build_checkpoint(tmp_path / "checkpoint-121", global_step=120)

    assert invoke_checkpoint(checkpoint, tmp_path) == 96


def test_abstains_without_manifest_and_returns_95(tmp_path: Path) -> None:
    checkpoint = tmp_path / "step_120"
    checkpoint.mkdir()
    (checkpoint / "rank-00000.pt").write_bytes(b"payload")

    assert invoke_checkpoint(checkpoint, tmp_path) == 95


def test_abstains_on_unknown_format_and_returns_95(tmp_path: Path) -> None:
    checkpoint = build_checkpoint(
        tmp_path / "step_120", overrides={"format": "unknown-layout-v0"}
    )

    assert invoke_checkpoint(checkpoint, tmp_path) == 95


def test_directory_without_step_abstains_only_the_a7_leg(
    tmp_path: Path, capsys
) -> None:
    checkpoint = build_checkpoint(tmp_path / "ordinary-checkpoint-name")

    assert invoke_checkpoint(checkpoint, tmp_path) == 0
    output = capsys.readouterr().out
    assert "A7b directory step agrees with manifest global_step: ABSTAIN" in output
    assert "0/1 step values encoded in directory name" in output
    assert "VERDICT 0 PASS" in output


@pytest.mark.parametrize(
    "suffix",
    ("step_120", "step-120", "checkpoint_120", "checkpoint-120", "ckpt_120", "ckpt-120"),
)
def test_supported_step_encoding_forms_agree_and_return_zero(
    tmp_path: Path, suffix: str
) -> None:
    checkpoint = build_checkpoint(tmp_path / suffix, global_step=120)

    assert invoke_checkpoint(checkpoint, tmp_path) == 0


def test_exit_95_and_exit_96_are_never_confused(tmp_path: Path) -> None:
    abstaining_missing = tmp_path / "abstain-missing"
    abstaining_missing.mkdir()
    abstaining_unknown = build_checkpoint(
        tmp_path / "step_120", overrides={"format": "not-a-registered-format"}
    )

    # These assertions guard the control pair directly: UNMEASURED must not be relabelled as
    # a detected defect, and a detected defect must not be laundered into abstention merely
    # because both are launch-refusing non-zero results.
    assert invoke_checkpoint(abstaining_missing, tmp_path) == 95
    assert invoke_checkpoint(abstaining_unknown, tmp_path) == 95
    assert invoke_checkpoint(abstaining_missing, tmp_path) != 96
    assert invoke_checkpoint(abstaining_unknown, tmp_path) != 96

    bad_cases = (
        build_checkpoint(
            tmp_path / "bad-missing" / "step_120", omitted_ranks=(0,), global_step=120
        ),
        build_checkpoint(
            tmp_path / "bad-empty" / "step_120", zero_length_rank=1, global_step=120
        ),
        build_checkpoint(
            tmp_path / "bad-extra" / "step_120", extra_ranks=(8,), global_step=120
        ),
        build_checkpoint(
            tmp_path / "bad-count" / "step_120",
            overrides={"rank_payload_count": 7},
            global_step=120,
        ),
        build_checkpoint(tmp_path / "bad-step" / "step_999", global_step=120),
    )
    for checkpoint in bad_cases:
        assert invoke_checkpoint(checkpoint, tmp_path) == 96
        assert invoke_checkpoint(checkpoint, tmp_path) != 95


def test_missing_checkpoint_directory_is_a_measured_refusal(tmp_path: Path) -> None:
    missing = tmp_path / "step_120"

    assert invoke_checkpoint(missing, tmp_path) == 96


def test_module_imports_no_tensor_library() -> None:
    # The adjudicator may run on a login node with no model stack installed. Reading the AST
    # tests the actual artifact rather than this test environment's coincidental imports.
    module_path = Path(adjudicator.__file__).resolve()
    tree = ast.parse(module_path.read_text(encoding="utf-8"), filename=str(module_path))
    imported_roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_roots.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_roots.add(node.module.split(".", 1)[0])

    assert "torch" not in imported_roots
