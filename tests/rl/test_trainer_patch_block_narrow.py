"""Tests for ``trainer._narrow_modality_kwargs_by_row``.

MEASURED defect (GRPO with images, Qwen3.6-27B/35B-A3B, ``logprob_micro_batch``
slicing): ``forward_logprob_slice`` row-narrows every modality tensor with a
plain ``value.narrow(0, start, end - start)``. After
``_expand_modality_kwargs_for_group`` fixed the GROUP-expansion step,
``image_grid_thw`` has exactly one row per kept training row, so narrowing
it that way is correct -- but its paired flattened-patch value
(``pixel_values``) is still the concatenation of each row's own patch BLOCK
in row order, and row-wise narrowing truncates to the first
``end - start`` raw PATCHES instead of the patches belonging to rows
``[start, end)``. MEASURED on GPU: a 1-row logprob slice over a 320-patch
image produced a 1-patch ``hidden_states`` against a 320-patch
``pos_embeds`` inside the vision tower (``RuntimeError: size of tensor a
(1) must match b (320)``; a 2-row slice gave ``(2)`` vs ``(640)``).

``_narrow_modality_kwargs_by_row`` detects a flattened-patch key the same
way ``_expand_modality_kwargs_for_group`` does (its width equals its paired
count key's ``prod(-1).sum()``) and slices the single contiguous span of
patches covering rows ``[start, end)`` instead.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch", reason="trainer.py itself refuses torch-free hosts")

import foundationscale.rl.trainer as trainer_module  # noqa: E402

_narrow = trainer_module._narrow_modality_kwargs_by_row


def test_a_single_row_slice_takes_that_rows_full_patch_block_not_one_patch() -> None:
    # 2 rows: row 0 has 3 patches, row 1 has 2 patches, laid out in order.
    pixel_values = torch.tensor([[0.0], [0.1], [0.2], [1.0], [1.1]])
    image_grid_thw = torch.tensor([[1, 1, 3], [1, 1, 2]], dtype=torch.long)
    modality_kwargs = {"pixel_values": pixel_values, "image_grid_thw": image_grid_thw}

    out = _narrow(modality_kwargs, start=0, end=1)

    assert torch.equal(out["pixel_values"], pixel_values[0:3])  # row 0's FULL block
    assert torch.equal(out["image_grid_thw"], image_grid_thw[0:1])


def test_a_two_row_slice_takes_both_rows_blocks_contiguously() -> None:
    pixel_values = torch.tensor([[0.0], [0.1], [0.2], [1.0], [1.1]])
    image_grid_thw = torch.tensor([[1, 1, 3], [1, 1, 2]], dtype=torch.long)
    modality_kwargs = {"pixel_values": pixel_values, "image_grid_thw": image_grid_thw}

    out = _narrow(modality_kwargs, start=0, end=2)

    assert torch.equal(out["pixel_values"], pixel_values)  # the whole 5-patch span
    assert torch.equal(out["image_grid_thw"], image_grid_thw)


def test_a_slice_starting_mid_batch_offsets_correctly() -> None:
    pixel_values = torch.tensor([[0.0], [0.1], [0.2], [1.0], [1.1], [2.0]])
    image_grid_thw = torch.tensor([[1, 1, 3], [1, 1, 2], [1, 1, 1]], dtype=torch.long)
    modality_kwargs = {"pixel_values": pixel_values, "image_grid_thw": image_grid_thw}

    out = _narrow(modality_kwargs, start=1, end=3)

    assert torch.equal(out["pixel_values"], pixel_values[3:6])  # rows 1 and 2's blocks
    assert torch.equal(out["image_grid_thw"], image_grid_thw[1:3])


def test_gemma_style_per_row_pixel_values_uses_plain_row_narrow() -> None:
    """No *_grid_thw key at all: the historical plain-narrow path."""
    pixel_values = torch.arange(3 * 2 * 4, dtype=torch.float32).reshape(3, 2, 4)
    out = _narrow({"pixel_values": pixel_values}, start=1, end=3)
    assert torch.equal(out["pixel_values"], pixel_values.narrow(0, 1, 2))


def test_a_value_not_actually_flattened_uses_plain_row_narrow() -> None:
    pixel_values = torch.arange(2 * 4, dtype=torch.float32).reshape(2, 4)  # already per-row
    image_grid_thw = torch.tensor([[1, 2, 5], [1, 1, 1]], dtype=torch.long)  # prod sum 11 != 2
    out = _narrow({"pixel_values": pixel_values, "image_grid_thw": image_grid_thw}, start=0, end=1)
    assert torch.equal(out["pixel_values"], pixel_values.narrow(0, 0, 1))
    assert torch.equal(out["image_grid_thw"], image_grid_thw.narrow(0, 0, 1))


def test_mixed_flattened_and_per_token_keys_are_each_narrowed_correctly() -> None:
    pixel_values = torch.tensor([[0.0], [0.1], [0.2], [1.0], [1.1]])
    image_grid_thw = torch.tensor([[1, 1, 3], [1, 1, 2]], dtype=torch.long)
    mm_token_type_ids = torch.tensor([[0, 0, 1, 1], [1, 1, 0, 0]], dtype=torch.long)

    out = _narrow(
        {
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
            "mm_token_type_ids": mm_token_type_ids,
        },
        start=1,
        end=2,
    )

    assert torch.equal(out["pixel_values"], pixel_values[3:5])
    assert torch.equal(out["mm_token_type_ids"], mm_token_type_ids[1:2])


def test_whole_batch_slice_matches_the_unsliced_tensor_exactly() -> None:
    """start=0, end=n_rows -- the non-micro-batched path -- must reproduce the
    full tensor byte for byte, the same invariant the old plain-narrow code
    trivially held."""
    pixel_values = torch.arange(6, dtype=torch.float32).reshape(6, 1)
    image_grid_thw = torch.tensor([[1, 1, 2], [1, 1, 4]], dtype=torch.long)

    out = _narrow({"pixel_values": pixel_values, "image_grid_thw": image_grid_thw}, start=0, end=2)

    assert torch.equal(out["pixel_values"], pixel_values)
    assert torch.equal(out["image_grid_thw"], image_grid_thw)


def test_video_pair_is_narrowed_the_same_way_as_the_image_pair() -> None:
    pixel_values_videos = torch.tensor([[9.0], [9.1], [8.0]])
    video_grid_thw = torch.tensor([[1, 1, 2], [1, 1, 1]], dtype=torch.long)

    out = _narrow(
        {"pixel_values_videos": pixel_values_videos, "video_grid_thw": video_grid_thw},
        start=1,
        end=2,
    )

    assert torch.equal(out["pixel_values_videos"], pixel_values_videos[2:3])
    assert torch.equal(out["video_grid_thw"], video_grid_thw[1:2])
