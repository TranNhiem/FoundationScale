# SPDX-License-Identifier: Apache-2.0
"""Controls for #497: a saturated step's evidence must be per-group.

The saturated-step line claims the reward has no variance WITHIN a group, and
then prints the distinct rewards it observed. Those were pooled across every
used row, so the evidence contradicted the claim whenever two groups saturated
at different values -- which is the common case, not a corner. MEASURED on
GB200, two adjacent lines of one run:

    UNMEASURED step 17: ... no within-group variance (distinct rewards: [0.0, 1.0])
    UNMEASURED step 18: ... no within-group variance (distinct rewards: [1.0])

Step 18 is honest. Step 17 reads as self-refuting, and a reader has no way to
tell from the line which of the two it is looking at. The diagnosis was correct
both times; only the evidence was reported on the wrong axis.
"""

from __future__ import annotations

from foundationscale.rl.trainer import _MAX_GROUPS_REPORTED, _per_group_reward_summary


def test_two_groups_saturated_at_different_values_are_not_pooled() -> None:
    """The GB200 observation, as a control.

    Every group is internally uniform, so the claim "no within-group variance"
    is true. The pooled report said `[0.0, 1.0]`, which denies it.
    """
    kept_rows = [0, 1, 2, 3]
    prompt_ids = ["row-0", "row-0", "row-1", "row-1"]
    rewards = [0.0, 0.0, 1.0, 1.0]

    summary = _per_group_reward_summary(kept_rows, prompt_ids, rewards)

    assert summary == "row-0: [0.0]; row-1: [1.0]"
    # The precise regression: no group may be shown carrying two values, because
    # a group that carried two values would not have reached this message.
    assert "[0.0, 1.0]" not in summary


def test_a_uniformly_saturated_step_reads_the_same_as_before() -> None:
    """The case that was already honest must not change."""
    summary = _per_group_reward_summary([0, 1], ["row-0", "row-0"], [1.0, 1.0])
    assert summary == "row-0: [1.0]"


def test_every_group_is_named_not_just_the_first() -> None:
    kept_rows = list(range(6))
    prompt_ids = [f"row-{index // 2}" for index in kept_rows]
    rewards = [0.0, 0.0, 1.0, 1.0, 0.0, 0.0]

    summary = _per_group_reward_summary(kept_rows, prompt_ids, rewards)

    for name in ("row-0", "row-1", "row-2"):
        assert name in summary


def test_many_groups_are_truncated_and_the_truncation_is_stated() -> None:
    """A truncated list that does not say it was truncated is its own untruth.

    This message exists to be read; a run with many prompts per step would
    otherwise emit a line long enough that nobody reads any of it. What is NOT
    acceptable is silently dropping groups, so the remainder is counted.
    """
    count = _MAX_GROUPS_REPORTED + 3
    kept_rows = list(range(count))
    prompt_ids = [f"row-{index:02d}" for index in kept_rows]
    rewards = [1.0] * count

    summary = _per_group_reward_summary(kept_rows, prompt_ids, rewards)

    assert summary.count("row-") == _MAX_GROUPS_REPORTED
    assert "(+3 more group(s))" in summary
