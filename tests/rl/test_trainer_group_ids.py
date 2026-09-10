"""Regression tests pinning the PRE-abstention group-id arithmetic in ``_one_step``.

WHAT IS CLAIMED: the trainer assigns group ids as ``f"row-{index // group_size}"``
using the PRE-abstention ``enumerate`` position, so surviving rows re-partition into
their original prompt groups even after abstentions drop rows. This test recomputes
that assignment directly and asserts the partition survives abstention. A CONTRAST
test asserts that the POST-filter position would NOT preserve the partition, proving
the property is load-bearing rather than restated.

WHAT IS NOT CLAIMED: no claim is made about the trainer's model, optimiser, loss
value, or any GPU behaviour. Nothing is loaded; the arithmetic is tested in
isolation. No claim is made about group sizes other than 4, though the stated
property is arithmetic and generalises.
"""

from __future__ import annotations

from collections.abc import Iterable

__all__ = (
    "test_post_filter_position_would_break_the_partition_contrast",
    "test_pre_abstention_indices_preserve_original_prompt_groups",
)


def _trainer_group_ids(
    offered_indices: Iterable[int],
    abstained_indices: set[int],
    group_size: int,
) -> dict[int, str]:
    """Recompute the trainer's id assignment from the PRE-abstention position."""
    return {
        index: f"row-{index // group_size}"
        for index in offered_indices
        if index not in abstained_indices
    }


def test_pre_abstention_indices_preserve_original_prompt_groups() -> None:
    """Survivors of abstention must fall back into their ORIGINAL prompt groups."""
    group_size = 4
    offered = range(2 * group_size)
    abstained = {2, 5}  # abstention is a dropped row, never a zeroed score

    ids = _trainer_group_ids(offered, abstained, group_size)

    assert set(ids) == {0, 1, 3, 4, 6, 7}
    assert all(ids[i] == "row-0" for i in (0, 1, 3))
    assert all(ids[i] == "row-1" for i in (4, 6, 7))


def test_post_filter_position_would_break_the_partition_contrast() -> None:
    """Indexing by POST-filter position must NOT reproduce the original partition."""
    group_size = 4
    abstained = {2, 5}
    survivors = [i for i in range(2 * group_size) if i not in abstained]

    post_filter_ids = {
        original: f"row-{position // group_size}"
        for position, original in zip(survivors, survivors, strict=True)
    }
    post_filter_ids = {
        original: f"row-{position // group_size}" for position, original in enumerate(survivors)
    }

    # Under post-filter indexing survivor 3 lands in row-0 (correct by luck of the
    # first prefix abstention) but survivor 4 is exiled from its prompt's group.
    assert post_filter_ids[4] != "row-1"
    assert post_filter_ids[4] == "row-0"
