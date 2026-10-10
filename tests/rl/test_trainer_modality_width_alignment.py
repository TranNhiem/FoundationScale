"""Tests for ``trainer._align_modality_keys_to_scored_width``.

MEASURED defect (GRPO with images, Qwen3.6-27B/35B-A3B): the scorer forward
in ``_one_step`` runs over ``kept_sequences`` -- prompt_width +
max_new_tokens columns wide, since #371 scores the GENERATED completion
under the same image conditioning it was generated under -- but
``modality_kwargs`` was only ever repeat_interleaved/index_selected along
the ROW axis. Per-TOKEN keys (``mm_token_type_ids`` on Qwen3VL,
``image_position_ids`` on gemma-4) stay at PROMPT width, so Qwen3.5's own
``get_rope_index`` indexes a full-width attention_mask against a
prompt-width ``mm_token_type_ids`` and raises an ``IndexError``: MEASURED on
GPU, mask shape ``[927]`` vs tensor shape ``[527]`` (prompt_width=527,
max_new_tokens=400) on one rank and ``[787]`` vs ``[387]``
(prompt_width=387) on another, same step. Per-IMAGE keys (``pixel_values``,
``image_grid_thw``) are untouched: their non-batch dims are patch features /
``(t, h, w)``, nothing to do with sequence length.

``_align_modality_keys_to_scored_width`` tells the two apart by shape alone
(dim 1 == prompt_width) and zero-pads only the per-token keys out to the
scored width -- 0 is plain text's own value in Qwen3VL's token-type
convention, and the newly generated columns are always text, never a second
image.
"""

from __future__ import annotations

from typing import Any

import pytest

torch = pytest.importorskip("torch", reason="trainer.py itself refuses torch-free hosts")

import foundationscale.rl.trainer as trainer_module  # noqa: E402

_align = trainer_module._align_modality_keys_to_scored_width


def test_per_token_key_is_zero_padded_to_the_scored_width() -> None:
    rows, prompt_width, new_tokens = 2, 5, 3
    mm_token_type_ids = torch.ones((rows, prompt_width), dtype=torch.long)

    out = _align(
        {"mm_token_type_ids": mm_token_type_ids},
        prompt_width=prompt_width,
        sequence_width=prompt_width + new_tokens,
    )

    aligned = out["mm_token_type_ids"]
    assert aligned.shape == (rows, prompt_width + new_tokens)
    assert torch.equal(aligned[:, :prompt_width], mm_token_type_ids)
    assert torch.equal(aligned[:, prompt_width:], torch.zeros((rows, new_tokens), dtype=torch.long))


def test_per_image_key_with_a_different_width_is_untouched() -> None:
    rows, prompt_width, new_tokens = 2, 5, 3
    # pixel_values: per-image patch features, dim 1 has nothing to do with
    # the sequence length -- here deliberately NOT equal to prompt_width.
    pixel_values = torch.arange(rows * 7, dtype=torch.float32).reshape(rows, 7)
    image_grid_thw = torch.tensor([[1, 2, 2], [1, 2, 2]], dtype=torch.long)

    out = _align(
        {"pixel_values": pixel_values, "image_grid_thw": image_grid_thw},
        prompt_width=prompt_width,
        sequence_width=prompt_width + new_tokens,
    )

    assert torch.equal(out["pixel_values"], pixel_values)
    assert torch.equal(out["image_grid_thw"], image_grid_thw)


def test_mixed_per_token_and_per_image_keys_are_each_handled_correctly() -> None:
    rows, prompt_width, new_tokens = 1, 4, 2
    mm_token_type_ids = torch.tensor([[0, 1, 1, 0]], dtype=torch.long)
    pixel_values = torch.ones((rows, 6), dtype=torch.float32)

    out = _align(
        {"mm_token_type_ids": mm_token_type_ids, "pixel_values": pixel_values},
        prompt_width=prompt_width,
        sequence_width=prompt_width + new_tokens,
    )

    assert out["mm_token_type_ids"].shape == (rows, prompt_width + new_tokens)
    assert torch.equal(out["mm_token_type_ids"][:, :prompt_width], mm_token_type_ids)
    assert out["pixel_values"].shape == (rows, 6)
    assert torch.equal(out["pixel_values"], pixel_values)


def test_no_completion_tokens_is_a_no_op() -> None:
    """sequence_width == prompt_width (e.g. max_new_tokens=0): nothing to pad."""
    mm_token_type_ids = torch.ones((2, 5), dtype=torch.long)

    out = _align({"mm_token_type_ids": mm_token_type_ids}, prompt_width=5, sequence_width=5)

    assert out["mm_token_type_ids"] is mm_token_type_ids


def test_dtype_is_preserved_through_the_pad() -> None:
    rows, prompt_width, new_tokens = 1, 3, 2
    for dtype in (torch.long, torch.float32, torch.bool):
        value = torch.ones((rows, prompt_width), dtype=dtype)
        out = _align(
            {"k": value}, prompt_width=prompt_width, sequence_width=prompt_width + new_tokens
        )
        assert out["k"].dtype == dtype
        assert out["k"].shape == (rows, prompt_width + new_tokens)


def test_a_per_token_key_with_trailing_feature_dims_pads_only_the_sequence_axis() -> None:
    """Defensive: today's measured keys are 2D, but the pad shape must still
    be correct if a future per-token key carries extra trailing dims."""
    rows, prompt_width, new_tokens, feat = 2, 3, 2, 4
    value = torch.ones((rows, prompt_width, feat), dtype=torch.float32)

    out = _align({"k": value}, prompt_width=prompt_width, sequence_width=prompt_width + new_tokens)

    aligned = out["k"]
    assert aligned.shape == (rows, prompt_width + new_tokens, feat)
    assert torch.equal(aligned[:, :prompt_width, :], value)
    assert torch.equal(aligned[:, prompt_width:, :], torch.zeros((rows, new_tokens, feat)))


def test_empty_modality_kwargs_is_a_no_op() -> None:
    assert _align({}, prompt_width=5, sequence_width=9) == {}


def test_returns_a_new_dict_without_mutating_the_input() -> None:
    mm_token_type_ids = torch.ones((1, 3), dtype=torch.long)
    original: dict[str, Any] = {"mm_token_type_ids": mm_token_type_ids}

    out = _align(original, prompt_width=3, sequence_width=5)

    assert out is not original
    assert original["mm_token_type_ids"].shape == (1, 3)
