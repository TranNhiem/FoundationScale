"""Tests for ``tasks.TaskSource``: reading a JSONL task file and sampling
deterministic per-step batches.

WHAT IS CLAIMED: valid rows parse into ``EpisodeTask`` with ``session_id`` set
to the row's ``uid``; malformed rows (bad JSON, non-object, missing keys, a
duplicate uid, a row that ``EpisodeTask`` itself refuses) raise
``TaskSourceRefusal`` naming the source path and the 1-based line number; blank
lines are skipped silently; an empty file (or a file with only blank lines)
refuses; ``batch(step, n)`` is deterministic given ``(seed, step, n)`` and
independent of call order/history; it samples without replacement when
``n <= len(pool)`` and with replacement otherwise; bad ``step``/``n`` arguments
refuse.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from foundationscale.agentic_rl.harness.base import EpisodeTask
from foundationscale.agentic_rl.tasks import TaskSource, TaskSourceRefusal


def _write(path: Path, rows: list[object]) -> str:
    lines = []
    for row in rows:
        lines.append(row if isinstance(row, str) else json.dumps(row))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(path)


def _row(uid: str, **overrides: object) -> dict[str, object]:
    fields: dict[str, object] = {
        "uid": uid,
        "messages": [{"role": "user", "content": f"hello {uid}"}],
    }
    fields.update(overrides)
    return fields


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def test_reads_valid_rows_into_episode_tasks(tmp_path: Path) -> None:
    path = _write(
        tmp_path / "tasks.jsonl",
        [_row("a"), _row("b", metadata={"difficulty": "easy"})],
    )
    source = TaskSource(path=path)
    assert len(source) == 2
    batch = source.batch(0, 2)
    assert {task.uid for task in batch} == {"a", "b"}
    for task in batch:
        assert isinstance(task, EpisodeTask)
        assert task.session_id == task.uid


def test_metadata_defaults_to_empty_mapping(tmp_path: Path) -> None:
    path = _write(tmp_path / "tasks.jsonl", [_row("a")])
    source = TaskSource(path=path)
    (task,) = source.batch(0, 1)
    assert dict(task.metadata) == {}


def test_blank_lines_are_skipped(tmp_path: Path) -> None:
    path = tmp_path / "tasks.jsonl"
    path.write_text(f"\n{json.dumps(_row('a'))}\n\n\n", encoding="utf-8")
    source = TaskSource(path=str(path))
    assert len(source) == 1


def test_empty_file_refuses(tmp_path: Path) -> None:
    path = tmp_path / "tasks.jsonl"
    path.write_text("", encoding="utf-8")
    with pytest.raises(TaskSourceRefusal, match=r"0 usable rows"):
        TaskSource(path=str(path))


def test_file_of_only_blank_lines_refuses(tmp_path: Path) -> None:
    path = tmp_path / "tasks.jsonl"
    path.write_text("\n\n\n", encoding="utf-8")
    with pytest.raises(TaskSourceRefusal, match=r"0 usable rows"):
        TaskSource(path=str(path))


# ---------------------------------------------------------------------------
# Malformed rows: refusals naming the path and the 1-based line number
# ---------------------------------------------------------------------------


def test_invalid_json_names_line_number(tmp_path: Path) -> None:
    path = _write(tmp_path / "tasks.jsonl", [_row("a"), "not { valid json"])
    with pytest.raises(TaskSourceRefusal, match=r"line 2"):
        TaskSource(path=path)


def test_non_object_row_names_line_number(tmp_path: Path) -> None:
    path = _write(tmp_path / "tasks.jsonl", [_row("a"), "[1, 2, 3]"])
    with pytest.raises(TaskSourceRefusal, match=r"line 2"):
        TaskSource(path=path)


def test_missing_uid_names_line_number(tmp_path: Path) -> None:
    row = {"messages": [{"role": "user", "content": "hi"}]}
    path = _write(tmp_path / "tasks.jsonl", [row])
    with pytest.raises(TaskSourceRefusal, match=r"line 1.*'uid'"):
        TaskSource(path=path)


def test_missing_messages_names_line_number(tmp_path: Path) -> None:
    row = {"uid": "a"}
    path = _write(tmp_path / "tasks.jsonl", [row])
    with pytest.raises(TaskSourceRefusal, match=r"line 1.*'messages'"):
        TaskSource(path=path)


def test_episode_task_refusal_is_wrapped_with_line_number(tmp_path: Path) -> None:
    # An assistant-role seed message is a HarnessRefusal at EpisodeTask construction
    # (see harness.base.EpisodeTask); TaskSource must wrap it, not let it escape raw.
    row = _row("a", messages=[{"role": "assistant", "content": "hi"}])
    path = _write(tmp_path / "tasks.jsonl", [row])
    with pytest.raises(TaskSourceRefusal, match=r"line 1"):
        TaskSource(path=path)


def test_duplicate_uid_names_both_lines(tmp_path: Path) -> None:
    path = _write(tmp_path / "tasks.jsonl", [_row("a"), _row("a")])
    with pytest.raises(TaskSourceRefusal, match=r"line 2.*already read at line 1"):
        TaskSource(path=path)


# ---------------------------------------------------------------------------
# TaskSource construction validation
# ---------------------------------------------------------------------------


def test_rejects_empty_path() -> None:
    with pytest.raises(TaskSourceRefusal):
        TaskSource(path="")


def test_rejects_bool_seed(tmp_path: Path) -> None:
    path = _write(tmp_path / "tasks.jsonl", [_row("a")])
    with pytest.raises(TaskSourceRefusal):
        TaskSource(path=path, seed=True)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# batch(step, n): determinism and sampling
# ---------------------------------------------------------------------------


def _source(tmp_path: Path, n_rows: int = 10, *, seed: int = 0) -> TaskSource:
    rows = [_row(f"t{i}") for i in range(n_rows)]
    path = _write(tmp_path / "tasks.jsonl", rows)
    return TaskSource(path=path, seed=seed)


def test_batch_is_deterministic_for_the_same_step(tmp_path: Path) -> None:
    source = _source(tmp_path)
    first = [task.uid for task in source.batch(5, 4)]
    second = [task.uid for task in source.batch(5, 4)]
    assert first == second


def test_batch_does_not_depend_on_call_order_or_history(tmp_path: Path) -> None:
    source_a = _source(tmp_path)
    for step in range(5):
        source_a.batch(step, 3)
    direct = [task.uid for task in source_a.batch(5, 3)]

    source_b = _source(tmp_path)
    fresh = [task.uid for task in source_b.batch(5, 3)]
    assert direct == fresh


def test_batch_without_replacement_has_unique_uids_when_n_fits(tmp_path: Path) -> None:
    source = _source(tmp_path, n_rows=10)
    batch = source.batch(0, 5)
    uids = [task.uid for task in batch]
    assert len(uids) == 5
    assert len(set(uids)) == 5


def test_batch_with_replacement_when_n_exceeds_pool(tmp_path: Path) -> None:
    source = _source(tmp_path, n_rows=2)
    batch = source.batch(0, 20)
    assert len(batch) == 20
    assert set(task.uid for task in batch) <= {"t0", "t1"}


def test_different_steps_can_draw_different_batches(tmp_path: Path) -> None:
    source = _source(tmp_path, n_rows=50)
    batches = {tuple(task.uid for task in source.batch(step, 5)) for step in range(10)}
    assert len(batches) > 1


def test_different_seeds_can_draw_different_batches(tmp_path: Path) -> None:
    rows = [_row(f"t{i}") for i in range(50)]
    path = _write(tmp_path / "tasks.jsonl", rows)
    source_a = TaskSource(path=path, seed=0)
    source_b = TaskSource(path=path, seed=1)
    assert [t.uid for t in source_a.batch(0, 5)] != [t.uid for t in source_b.batch(0, 5)]


@pytest.mark.parametrize("step", [-1, 1.5, "0"])
def test_batch_rejects_bad_step(tmp_path: Path, step: object) -> None:
    source = _source(tmp_path)
    with pytest.raises(TaskSourceRefusal):
        source.batch(step, 1)  # type: ignore[arg-type]


@pytest.mark.parametrize("n", [0, -1, 1.5, "1"])
def test_batch_rejects_bad_n(tmp_path: Path, n: object) -> None:
    source = _source(tmp_path)
    with pytest.raises(TaskSourceRefusal):
        source.batch(0, n)  # type: ignore[arg-type]
