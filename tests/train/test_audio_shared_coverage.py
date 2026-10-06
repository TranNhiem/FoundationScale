"""Shared audio coverage pins: the counts survive DataLoader worker processes.

MEASURED (GB200, 2026-10-06): with ``--dataloader-num-workers 4`` the collator
counted into per-worker copies of ``AudioCoverage``; the parent read
rows_checked=0, the speech gates went VACUOUS and the run ended RED. Zero
workers passed only because there was one copy to count in.

torch imports freely here: a REAL DataLoader is the claim under test. The
bare-interpreter contract binds ``foundationscale/train/audio.py``, not its
tests.
"""

from __future__ import annotations

import multiprocessing
from collections.abc import Sequence
from typing import Any

import pytest
from torch.utils.data import DataLoader, Dataset

from foundationscale.train.audio import AudioCoverage, SharedAudioCoverage

# torch's import can leave background threads alive on some builds, and a 3.12+
# fork out of a threaded process warns. The claim under test is the COUNTS
# reaching the parent -- never a skip, only the warning is silenced.
pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


def _exercise(coverage: AudioCoverage | SharedAudioCoverage) -> None:
    """One identical operation sequence on either class; the manifests must agree.

    record_ok x2, record_refused x1, then one write per add_* method -- the
    shared interface spelled out on both backends.
    """
    coverage.record_ok(1.5)
    coverage.record_ok(0.5)
    coverage.record_refused("too_long")
    coverage.add_expected(4)
    coverage.add_placeholder_verified(2)
    coverage.add_placeholder_unmeasured(1)


def _count_in_child(coverage: SharedAudioCoverage) -> None:
    """What a worker does to the counters -- run here from a process that forked them."""
    coverage.add_expected()
    coverage.record_ok(1.0)


class _RowDataset(Dataset[int]):
    """Indifferent rows: what is under test is the COUNTING, not the content."""

    def __init__(self, count: int) -> None:
        self.count = count

    def __len__(self) -> int:
        return self.count

    def __getitem__(self, index: int) -> int:
        return index


class _CountingCollator:
    """A collate_fn that counts into the shared coverage, picklable on purpose.

    Picklable because a closure cannot travel to the workers under spawn and the
    test would then fail on the pickle rather than on the counts. It calls
    exactly ``cov.add_expected(len(batch))`` and ``cov.record_ok(1.0)`` per item.
    """

    def __init__(self, coverage: SharedAudioCoverage) -> None:
        self.coverage = coverage

    def __call__(self, batch: Sequence[int]) -> list[int]:
        self.coverage.add_expected(len(batch))
        for _ in batch:
            self.coverage.record_ok(1.0)
        return list(batch)


def test_shared_manifest_equals_plain_manifest() -> None:
    # MUST_PASS: same keys, same values -- one contract, two storage backends.
    plain = AudioCoverage(rows_expected=0)
    shared = SharedAudioCoverage(rows_expected=0)
    _exercise(plain)
    _exercise(shared)

    assert set(shared.as_manifest()) == set(plain.as_manifest())
    assert shared.as_manifest() == plain.as_manifest()
    # and the readable attributes are plain builtins, never ctypes handles
    assert type(shared.rows_expected) is int
    assert type(shared.rows_checked) is int
    assert type(shared.seconds_total) is float
    assert type(shared.refused) is dict
    assert type(shared.placeholder_rows_verified) is int
    assert type(shared.placeholder_rows_unmeasured) is int


def test_child_process_counts_are_visible_in_the_parent() -> None:
    # MUST_PASS: three forked children, each record_ok + add_expected, counted
    # in the parent. Plain attributes would show 0 here -- that WAS the defect.
    ctx: Any = multiprocessing.get_context("fork")
    coverage = SharedAudioCoverage(rows_expected=0)
    children = [ctx.Process(target=_count_in_child, args=(coverage,)) for _ in range(3)]
    for child in children:
        child.start()
    for child in children:
        child.join(timeout=60)
        assert child.exitcode == 0

    assert coverage.rows_expected == 3
    assert coverage.rows_checked == 3
    assert coverage.seconds_total == 3.0
    assert coverage.verdict() == "COVERED"


def test_dataloader_workers_count_into_the_parent() -> None:
    # MUST_PASS: the measured failure's shape for real -- num_workers=2 over 8
    # rows, counting in the workers, read in the parent after iteration.
    coverage = SharedAudioCoverage(rows_expected=0)
    loader = DataLoader(
        _RowDataset(8),
        batch_size=4,
        num_workers=2,
        collate_fn=_CountingCollator(coverage),
    )
    batches = list(loader)

    assert sum(len(batch) for batch in batches) == 8
    assert coverage.rows_expected == 8
    assert coverage.rows_checked == 8
    assert coverage.seconds_total == 8.0
    assert coverage.verdict() == "COVERED"


def test_unknown_refusal_reason_raises() -> None:
    # MUST_FIRE: the reason vocabulary is closed at the write, exactly like
    # AudioLoadError's construction-time check -- and the row is counted NOWHERE.
    coverage = SharedAudioCoverage(rows_expected=0)
    with pytest.raises(ValueError):
        coverage.record_refused("not-a-reason")

    assert coverage.rows_checked == 0
    assert coverage.as_manifest()["rows_refused"] == 0


def test_reset_zeroes_every_counter_like_a_fresh_instance() -> None:
    coverage = SharedAudioCoverage(rows_expected=0)
    _exercise(coverage)
    coverage.reset()

    assert coverage.as_manifest() == SharedAudioCoverage(rows_expected=0).as_manifest()
    assert coverage.as_manifest()["verdict"] == "VACUOUS"


def test_sampling_rate_none_round_trips_the_sentinel_never_leaks() -> None:
    # -1 is the None sentinel in the shared c_long; a manifest that leaked it
    # would claim the run trained at a sampling rate no measurement produced.
    coverage = SharedAudioCoverage(rows_expected=0)
    assert coverage.sampling_rate is None
    assert coverage.as_manifest()["sampling_rate"] is None

    _exercise(coverage)
    coverage.reset()

    assert coverage.sampling_rate is None
    assert coverage.as_manifest()["sampling_rate"] is None
    assert coverage.sampling_rate != -1
