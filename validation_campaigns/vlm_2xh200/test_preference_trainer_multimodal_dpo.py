"""GPU-only proof that the offline preference trainer's image declaration
(``PreferenceTrainConfig.image_column``) actually reaches the model, not just
the processor's tensors.

Two claims, each needing real hardware and a real checkpoint, hence living
here rather than in ``tests/`` (see this directory's ``conftest.py``):

  1. An image-declared DPO run on a real model (``gemma-4-12B-it``, LoRA
     r16) reproduces the SAME step-0 invariant the text-only smoke test
     proves on CPU with a toy model (``tests/rl/test_preference_trainer.py``):
     policy and reference are the same checkpoint with LoRA disabled, so
     every margin is zero and the loss is exactly ``softplus(0) = ln(2)``.
  2. THE CONTROL: two independent 1-step DPO runs on the SAME 8 synthetic
     image pairs, same model, same seed (so LoRA's own random init is
     reproduced identically -- ``PreferenceTrainer.run()`` seeds ``torch``
     before loading), once with the REAL per-row image and once with every
     image replaced by a solid BLACK image of identical size (same marker,
     same declared column, only the PIXEL CONTENT differs). If the model's
     forward pass ignored the vision tower's output, the two step-0 losses
     would be identical; they must differ.

Skipped (this directory's own convention, NOT ``tests/``'s "no skips" rule --
see ``test_video_reaches_the_model.py`` for the same shape) unless CUDA is
available and ``FS_TEST_MODELS_DIR`` has full weights for ``gemma-4-12B-it``::

    FS_TEST_MODELS_DIR=/workspace/fs-vlm/models \
        /workspace/fs-vlm/venv/bin/python -m pytest -s \
        validation_campaigns/vlm_2xh200/test_preference_trainer_multimodal_dpo.py
"""

from __future__ import annotations

import gc
import math
import os
from pathlib import Path
from typing import Any

import pytest

MODELS_DIR = os.environ.get("FS_TEST_MODELS_DIR")
_FAMILY = "gemma-4-12B-it"


def _cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except ImportError:
        return False


def _family_dir() -> Path | None:
    if not MODELS_DIR:
        return None
    candidate = Path(MODELS_DIR) / _FAMILY
    return candidate if candidate.is_dir() else None


def _available() -> bool:
    return _cuda_available() and _family_dir() is not None


requires_gpu_model = pytest.mark.skipif(
    not _available(),
    reason=(
        f"needs CUDA and FS_TEST_MODELS_DIR containing full weights for {_FAMILY!r}"
    ),
)


_PALETTE = [
    (200, 30, 30),
    (30, 180, 30),
    (30, 60, 200),
    (220, 200, 30),
    (180, 30, 180),
    (30, 180, 180),
    (120, 90, 40),
    (90, 90, 90),
]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    import json

    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _make_image_pairs(tmp_path: Path, *, black: bool) -> Path:
    """8 synthetic DPO image pairs: a distinctly-coloured (or solid black)
    8x8 PNG per row, same prompt/chosen/rejected text in both conditions --
    only the pixel content differs between ``black=False``/``True``."""
    from PIL import Image

    img_dir = tmp_path / ("images_black" if black else "images_real")
    img_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, color in enumerate(_PALETTE):
        fill = (0, 0, 0) if black else color
        image_path = img_dir / f"pair-{index}.png"
        Image.new("RGB", (8, 8), color=fill).save(image_path)
        rows.append(
            {
                "prompt": "<image> What is the dominant colour of this image?",
                "image": str(image_path),
                "chosen": "The image is mostly one solid colour.",
                "rejected": "The image shows a busy crowded city street.",
            }
        )
    out_path = tmp_path / ("pairs_black.jsonl" if black else "pairs_real.jsonl")
    _write_jsonl(out_path, rows)
    return out_path


def _run_steps(dataset: Path, *, model_dir: str, max_steps: int) -> tuple[list[float], float]:
    """Run ``max_steps`` LoRA DPO steps; return (per-step losses, peak GB)."""
    import torch

    from foundationscale.rl.preference_trainer import PreferenceTrainConfig, PreferenceTrainer

    torch.cuda.reset_peak_memory_stats()
    cfg = PreferenceTrainConfig(
        model=model_dir,
        dataset=str(dataset),
        algorithm="dpo",
        pairs_per_step=8,
        max_steps=max_steps,
        max_length=512,
        logprob_micro_batch=2,
        learning_rate=1e-4,
        device="cuda",
        gradient_checkpointing=True,
        adapter="lora",
        adapter_rank=16,
        adapter_alpha=32.0,
        seed=0,
        image_column="image",
    )
    trainer = PreferenceTrainer(cfg)
    reports = trainer.run()
    assert len(reports) == max_steps, f"expected {max_steps} measured step(s), got {len(reports)}"
    peak_gb = torch.cuda.max_memory_allocated() / (1024**3)
    return [entry["loss"] for entry in trainer.history], peak_gb


@requires_gpu_model
def test_image_declared_dpo_step0_loss_is_ln2(tmp_path: Path) -> None:
    import torch

    dataset = _make_image_pairs(tmp_path, black=False)
    try:
        losses, peak_gb = _run_steps(dataset, model_dir=str(_family_dir()), max_steps=1)
    finally:
        gc.collect()
        torch.cuda.empty_cache()

    print(
        f"\n[preference_mm_dpo] family={_FAMILY!r} image-declared DPO step-0 "
        f"loss={losses[0]!r} peak_gb={peak_gb:.1f}"
    )
    assert abs(losses[0] - math.log(2)) < 1e-3, (
        f"image-declared DPO step-0 loss {losses[0]!r} is not ln(2) within 1e-3 "
        f"on a policy==reference step -- the reference-disable-adapter path "
        f"may not be reached for a media-declared run"
    )


@requires_gpu_model
def test_image_pixels_change_the_loss_real_vs_black_control(tmp_path: Path) -> None:
    """The control: the SAME 8 rows, SAME model, SAME seed (so LoRA's own
    random init reproduces identically) -- real pixels vs black pixels must
    give DIFFERENT loss AFTER a gradient step.

    NOT step-0: at step 0 policy == reference EXACTLY (peft's LoRA B matrix
    initialises to zero, so the adapter-disabled reference and the
    adapter-enabled policy compute the IDENTICAL forward for ANY input), so
    DPO's loss collapses to softplus(0) = ln(2) regardless of what the image
    pixels are -- that invariant is exactly what the test above measures,
    and it would make this control vacuously "pass" (identical losses) on a
    CORRECT trainer too. Running 2 steps and comparing the SECOND step's
    loss compares a policy that has already taken one gradient step shaped
    by the actual pixel content it was shown -- real vs black pixels then
    diverge the policy differently, and the comparison is meaningful.
    """
    import torch

    model_dir = str(_family_dir())
    real_dataset = _make_image_pairs(tmp_path, black=False)
    black_dataset = _make_image_pairs(tmp_path, black=True)

    try:
        losses_real, peak_real = _run_steps(real_dataset, model_dir=model_dir, max_steps=2)
        gc.collect()
        torch.cuda.empty_cache()
        losses_black, peak_black = _run_steps(black_dataset, model_dir=model_dir, max_steps=2)
    finally:
        gc.collect()
        torch.cuda.empty_cache()

    print(
        f"\n[preference_mm_control] family={_FAMILY!r} "
        f"losses_real={losses_real!r} (peak {peak_real:.1f} GB) "
        f"losses_black={losses_black!r} (peak {peak_black:.1f} GB)"
    )
    assert losses_real[0] == pytest.approx(losses_black[0], abs=1e-6), (
        f"step-0 losses differ ({losses_real[0]!r} vs {losses_black[0]!r}) "
        "before any gradient step was taken -- policy and reference should "
        "be identical regardless of pixel content at step 0, so a mismatch "
        "here means something OTHER than the image is not held fixed "
        "between the two conditions"
    )
    assert losses_real[1] != losses_black[1], (
        f"{_FAMILY}: step-1 loss (after one gradient step) is IDENTICAL for "
        f"real vs black images ({losses_real[1]!r} == {losses_black[1]!r}) "
        "from the same seed and the same text -- the model's forward pass "
        "does not appear to depend on the declared image pixels at all"
    )
