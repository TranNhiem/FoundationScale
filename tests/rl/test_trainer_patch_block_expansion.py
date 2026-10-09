"""Tests for ``trainer._expand_modality_kwargs_for_group``.

MEASURED defect (GRPO with images, Qwen3.6-27B/35B-A3B): Qwen-VL-family
``pixel_values`` is FLATTENED PATCHES across the whole encoded chunk --
shape ``(sum_i patches_i, patch_dim)`` -- with ``image_grid_thw`` (one row
per image, ``(t, h, w)``) giving each image's own ``patches_i = t*h*w``
contiguous block length. The trainer's row-expansion
(``repeat_interleave(group, dim=0).index_select(0, kept_indices)``) treats
every modality key as one row per PROMPT, which is correct for gemma-4's
``pixel_values`` (shape ``(n_images, ...)``) but slices Qwen's flattened
tensor by raw PATCH position instead of by image. MEASURED on GPU by
printing shapes at the call site: two images with patch counts
``[320, 288]`` (608 total rows) and ``kept_indices=[0, 1]`` (both
group-expanded rows mapping to prompt 0) produced a 2-row ``pixel_values``
against an ``image_grid_thw`` still correctly claiming 640 patches --
exactly the ``RuntimeError: size of tensor a (2) must match b (640)`` the
vision tower's position-embedding add raised.

``_expand_modality_kwargs_for_group`` detects a flattened-patch key by
shape alone (its width equals its paired count key's ``prod(-1).sum()``)
and block-expands it instead: each kept, group-expanded row's own image
block (``kept_indices // group`` recovers the original image index) is
concatenated in order. Every other key -- including gemma-4's per-image
``pixel_values``, which has no ``*_grid_thw`` pairing at all -- keeps the
original per-row ``repeat_interleave``/``index_select`` path, unchanged.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch", reason="trainer.py itself refuses torch-free hosts")

import foundationscale.rl.trainer as trainer_module  # noqa: E402

_expand = trainer_module._expand_modality_kwargs_for_group


def test_qwen_style_flattened_patches_are_expanded_block_wise() -> None:
    # 2 images: image 0 has 3 patches, image 1 has 2 patches -- distinct
    # per-patch values so a wrong slice is visibly wrong, not coincidentally
    # right.
    pixel_values = torch.tensor(
        [[0.0], [0.1], [0.2], [1.0], [1.1]]  # image 0 block  # image 1 block
    )
    image_grid_thw = torch.tensor([[1, 1, 3], [1, 1, 2]], dtype=torch.long)
    group = 2
    # Rows 0,1 are prompt 0's group copies; row 2 is prompt 1's first copy.
    kept_indices = torch.tensor([0, 1, 2], dtype=torch.long)

    out = _expand(
        {"pixel_values": pixel_values, "image_grid_thw": image_grid_thw},
        group=group,
        kept_indices=kept_indices,
    )

    # Rows 0 and 1 both come from prompt 0 -> image 0's FULL 3-patch block,
    # twice. Row 2 comes from prompt 1 -> image 1's full 2-patch block.
    expected_pixels = torch.cat([pixel_values[0:3], pixel_values[0:3], pixel_values[3:5]], dim=0)
    assert torch.equal(out["pixel_values"], expected_pixels)
    assert torch.equal(
        out["image_grid_thw"], torch.tensor([[1, 1, 3], [1, 1, 3], [1, 1, 2]], dtype=torch.long)
    )
    # The count key's own expansion must agree row-for-row with the patch
    # blocks actually produced: the gate this bug would trip downstream.
    assert out["image_grid_thw"].prod(dim=-1).sum().item() == out["pixel_values"].shape[0]


def test_video_pair_is_expanded_the_same_way_as_the_image_pair() -> None:
    pixel_values_videos = torch.tensor([[9.0], [9.1], [8.0]])
    video_grid_thw = torch.tensor([[1, 1, 2], [1, 1, 1]], dtype=torch.long)
    group = 1
    kept_indices = torch.tensor([1], dtype=torch.long)  # prompt 1's video

    out = _expand(
        {"pixel_values_videos": pixel_values_videos, "video_grid_thw": video_grid_thw},
        group=group,
        kept_indices=kept_indices,
    )

    assert torch.equal(out["pixel_values_videos"], pixel_values_videos[2:3])
    assert torch.equal(out["video_grid_thw"], torch.tensor([[1, 1, 1]], dtype=torch.long))


def test_gemma_style_per_image_pixel_values_is_left_on_the_per_row_path() -> None:
    """No *_grid_thw key at all: pixel_values is already (n_images, ...), one
    row per prompt -- the historical repeat_interleave/index_select path."""
    pixel_values = torch.arange(2 * 3 * 4, dtype=torch.float32).reshape(2, 3, 4)
    group = 2
    kept_indices = torch.tensor([0, 2, 3], dtype=torch.long)

    out = _expand({"pixel_values": pixel_values}, group=group, kept_indices=kept_indices)

    expected = pixel_values.repeat_interleave(group, dim=0).index_select(0, kept_indices)
    assert torch.equal(out["pixel_values"], expected)


def test_a_value_not_actually_flattened_falls_back_to_per_row() -> None:
    """A (value, count) pair present but value.shape[0] does NOT equal
    counts.prod(-1).sum(): not genuinely flattened, so it must not be
    block-expanded on a guess."""
    pixel_values = torch.arange(2 * 4, dtype=torch.float32).reshape(2, 4)  # already per-row
    # 2 rows, matching pixel_values row-for-row, but prod(-1).sum() == 11 !=
    # pixel_values.shape[0] == 2 -- NOT flattened patches of this value.
    image_grid_thw = torch.tensor([[1, 2, 5], [1, 1, 1]], dtype=torch.long)
    group = 1
    kept_indices = torch.tensor([0, 1], dtype=torch.long)

    out = _expand(
        {"pixel_values": pixel_values, "image_grid_thw": image_grid_thw},
        group=group,
        kept_indices=kept_indices,
    )

    assert torch.equal(out["pixel_values"], pixel_values.index_select(0, kept_indices))
    assert torch.equal(out["image_grid_thw"], image_grid_thw.index_select(0, kept_indices))


def test_single_patch_per_image_reduces_to_the_per_row_result() -> None:
    """Degenerate case measured on the real corpus's small images: every
    image is exactly 1 patch, so sum(prod) coincidentally equals n_images
    too -- the block-wise path must still agree with the simple per-row one."""
    pixel_values = torch.tensor([[1.0], [2.0], [3.0]])
    image_grid_thw = torch.tensor([[1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype=torch.long)
    group = 2
    kept_indices = torch.tensor([0, 1, 4], dtype=torch.long)

    out = _expand(
        {"pixel_values": pixel_values, "image_grid_thw": image_grid_thw},
        group=group,
        kept_indices=kept_indices,
    )

    expected = pixel_values.repeat_interleave(group, dim=0).index_select(0, kept_indices)
    assert torch.equal(out["pixel_values"], expected)


def test_mixed_flattened_and_per_row_keys_in_one_call() -> None:
    pixel_values = torch.tensor([[0.0], [0.1], [1.0]])
    image_grid_thw = torch.tensor([[1, 1, 2], [1, 1, 1]], dtype=torch.long)
    mm_token_type_ids = torch.tensor([[0, 0], [1, 1]], dtype=torch.long)
    group = 1
    kept_indices = torch.tensor([0, 1], dtype=torch.long)

    out = _expand(
        {
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
            "mm_token_type_ids": mm_token_type_ids,
        },
        group=group,
        kept_indices=kept_indices,
    )

    assert torch.equal(out["pixel_values"], pixel_values)  # unchanged: already in block order
    assert torch.equal(out["mm_token_type_ids"], mm_token_type_ids)


def test_empty_modality_kwargs_is_a_no_op() -> None:
    out = _expand({}, group=4, kept_indices=torch.tensor([0, 1], dtype=torch.long))
    assert out == {}


def test_repeated_rows_duplicate_the_same_image_block_not_overlapping_patches() -> None:
    """Three kept rows all mapping to the SAME original prompt (a group with
    3 of its 4 rollouts kept) must each get the IDENTICAL full block, not a
    sliding/overlapping window."""
    pixel_values = torch.tensor([[1.0], [2.0], [3.0]])  # one image, 3 patches
    image_grid_thw = torch.tensor([[1, 1, 3]], dtype=torch.long)
    group = 4
    kept_indices = torch.tensor([0, 1, 2], dtype=torch.long)  # all from prompt 0

    out = _expand(
        {"pixel_values": pixel_values, "image_grid_thw": image_grid_thw},
        group=group,
        kept_indices=kept_indices,
    )

    assert torch.equal(out["pixel_values"], torch.cat([pixel_values] * 3, dim=0))
    assert torch.equal(out["image_grid_thw"], image_grid_thw.repeat(3, 1))
