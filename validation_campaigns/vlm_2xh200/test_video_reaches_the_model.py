"""GPU-only proof that NATIVE video (train/conversation.py's ``frames_for`` path,
foundationscale.video.frames_for_row) actually reaches the model, not just the
collator's tensors.

Two claims, each needing real hardware and real checkpoints, hence living here
rather than in ``tests/`` (see this directory's ``conftest.py``):

  1. A real video row produces a non-zero, countable number of video tokens
     (``processor.video_token_id`` occurrences in ``input_ids``) -- tokens per
     frame and the row's total length are printed as evidence, not just
     asserted nonzero.
  2. The CONTROL: training 2 steps on the SAME 8 video rows, from the SAME
     initial LoRA weights, once with the REAL decoded frames and once with
     those frames replaced by solid BLACK images of identical size (same
     token count, same position/metadata, only the PIXEL CONTENT differs) --
     if the model's forward pass ignored the video tower's output, the two
     step-0 losses would be identical; they must differ.

Skipped (this directory's own convention, NOT tests/'s "no skips" rule --
see test_conversation_real_processors.py and test_fused_loss_parity.py for
the same shape) unless CUDA is available, FS_TEST_MODELS_DIR has full weights
for both families, and a real video dataset is reachable (FS_TEST_VIDEO_JSONL,
default the 8-row estate control set under /workspace/fs-vlm)::

    FS_TEST_MODELS_DIR=/workspace/fs-vlm/models \
        /workspace/fs-vlm/venv/bin/python -m pytest -s \
        validation_campaigns/vlm_2xh200/test_video_reaches_the_model.py
"""

from __future__ import annotations

import functools
import json
import os
from pathlib import Path
from typing import Any

import pytest

MODELS_DIR = os.environ.get("FS_TEST_MODELS_DIR")
_FAMILIES = ("gemma-4-12B-it", "Qwen3.6-27B")
_FRAMES = 16
_VIDEO_JSONL = os.environ.get(
    "FS_TEST_VIDEO_JSONL", "/workspace/fs-vlm/data/prepared/video_control8.jsonl"
)
_VIDEO_CACHE_DIR = os.environ.get(
    "FS_TEST_VIDEO_CACHE_DIR", "/workspace/fs-vlm/data/raw/video/.cache"
)
_VIDEO_BASE_DIR = os.environ.get("FS_TEST_VIDEO_BASE_DIR", "/workspace/fs-vlm")


def _cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except ImportError:
        return False


def _family_dir(name: str) -> Path | None:
    if not MODELS_DIR:
        return None
    candidate = Path(MODELS_DIR) / name
    return candidate if candidate.is_dir() else None


def _video_rows() -> list[dict[str, Any]]:
    path = Path(_VIDEO_JSONL)
    if not path.is_file():
        return []
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _video_available() -> bool:
    return bool(_video_rows())


def _available() -> bool:
    return (
        _cuda_available()
        and all(_family_dir(name) is not None for name in _FAMILIES)
        and _video_available()
    )


requires_gpu_models_and_video = pytest.mark.skipif(
    not _available(),
    reason=(
        "needs CUDA, FS_TEST_MODELS_DIR containing full weights for "
        f"{_FAMILIES!r}, and a real video dataset at {_VIDEO_JSONL!r} "
        "(override with FS_TEST_VIDEO_JSONL/_CACHE_DIR/_BASE_DIR)"
    ),
)


def _frames_for_real(budget: Any) -> Any:
    from foundationscale import video

    return functools.partial(
        video.frames_for_row,
        video_column="video",
        budget=budget,
        cache_dir=_VIDEO_CACHE_DIR,
        base_dir=_VIDEO_BASE_DIR,
    )


def _frames_for_black(budget: Any) -> Any:
    """The real decode's own metadata (fps/duration/frames_indices unchanged),
    frames replaced by solid black images of the SAME size -- same token
    count and same position information, only the pixel content differs."""
    real = _frames_for_real(budget)

    def control(row: Any) -> dict[str, Any]:
        from PIL import Image

        info = real(row)
        black = [Image.new("RGB", frame.size, (0, 0, 0)) for frame in info["frames"]]
        return {**info, "frames": black}

    return control


@requires_gpu_models_and_video
@pytest.mark.parametrize("family", _FAMILIES)
def test_video_tokens_per_frame_and_row_length(family: str) -> None:
    from transformers import AutoProcessor

    from foundationscale.train.conversation import train_conversation_collator_or_refuse
    from foundationscale.video import FrameBudget

    model_dir = str(_family_dir(family))
    processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)
    budget = FrameBudget(_FRAMES)

    rows = _video_rows()
    assert rows, f"no rows found at {_VIDEO_JSONL}"
    row = rows[0]

    collate = train_conversation_collator_or_refuse(
        processor,
        max_length=16384,
        inject_dummy_media=False,
        frames_for=_frames_for_real(budget),
        video_column="video",
    )
    batch = collate([row])

    video_token_id = getattr(processor, "video_token_id", None)
    assert video_token_id is not None, f"{family}: processor has no video_token_id"
    video_token_count = int((batch["input_ids"] == video_token_id).sum())
    total_length = int(batch["attention_mask"][0].sum())
    tokens_per_frame = video_token_count / _FRAMES

    print(
        f"\n[video_reaches_model] family={family!r} frames={_FRAMES} "
        f"video_tokens={video_token_count} tokens_per_frame={tokens_per_frame:.2f} "
        f"total_row_length={total_length}"
    )
    assert video_token_count > 0, (
        f"{family}: zero video tokens in the collated row -- the native video path "
        "did not reach the processor"
    )
    # Every video token is masked out of the labels (never supervised).
    assert not bool((batch["labels"] == video_token_id).any())


def _build_lora_model(model_dir: str) -> Any:
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForImageTextToText

    from foundationscale.families.adapters import plan_adapter_targets, torch_linear_predicate
    from foundationscale.train.fused_loss import apply_fused_loss, fused_loss_refusal_reason

    # AutoModelForImageTextToText, NOT AutoModelForCausalLM: MEASURED, for the
    # "qwen3_5"/"qwen3_5_moe" families (Qwen3.6-27B, Qwen3.6-35B-A3B),
    # AutoModelForCausalLM resolves to Qwen3_5ForCausalLM / Qwen3_5MoeForCausalLM
    # -- transformers' own "VLM compatibility" text-only wrapper, which
    # explicitly SKIPS the vision tower's weights on load
    # (`_keys_to_ignore_on_load_unexpected = [r"^mtp.*", r"^model.visual.*"]`)
    # and silently absorbs pixel_values_videos/video_grid_thw through its
    # **kwargs without ever reading them -- a forward pass through it produces
    # a loss that is BIT-IDENTICAL for real vs black video frames, which is
    # exactly the failure this test exists to catch, and is how this gap was
    # found. gemma4_unified has no such split (AutoModelForCausalLM already
    # resolves it to the one Gemma4UnifiedForConditionalGeneration class), so
    # this swap is a no-op there and a required fix here.
    model = AutoModelForImageTextToText.from_pretrained(model_dir, dtype=torch.bfloat16).to("cuda")
    model.gradient_checkpointing_enable()
    # Un-fused cross-entropy materializes (seq x vocab) fp32 logits -- for an
    # 8-row video batch at gemma-4-12B-it's 262144 vocab that OOMs on a
    # single GPU outright (MEASURED here; train/fused_loss.py's own docstring
    # records the identical shape at 32768 context). liger is applied BEFORE
    # peft wraps the model, the same order train/loop.py uses.
    model_type = getattr(model.config, "model_type", None)
    assert fused_loss_refusal_reason("liger", model_type) is None, (
        f"model_type={model_type!r} is not covered by --fused-loss liger; this "
        "proof's family list assumed it was"
    )
    apply_fused_loss(model, model_type)
    config_to_dict = getattr(getattr(model, "config", None), "to_dict", None)
    family_config = config_to_dict() if callable(config_to_dict) else {}
    plan = plan_adapter_targets(
        family_config, None, model.named_modules(), torch_linear_predicate()
    )
    assert plan.refusal is None, f"LoRA target plan refused: {plan.refusal}"
    return get_peft_model(model, LoraConfig(r=16, lora_alpha=32, target_modules=list(plan.targets)))


@requires_gpu_models_and_video
@pytest.mark.parametrize("family", _FAMILIES)
def test_video_pixels_change_the_loss_real_vs_black_control(family: str) -> None:
    """The control: 2 training steps on the SAME 8 rows, from the SAME initial
    LoRA weights, real frames vs black frames -- step-0 losses must differ."""
    import torch
    from transformers import AutoProcessor

    from foundationscale.train.conversation import train_conversation_collator_or_refuse
    from foundationscale.video import FrameBudget

    model_dir = str(_family_dir(family))
    processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)
    budget = FrameBudget(_FRAMES)
    rows = _video_rows()
    assert len(rows) >= 8, f"need >= 8 video rows, got {len(rows)} at {_VIDEO_JSONL}"
    rows = rows[:8]

    model = _build_lora_model(model_dir)
    model.train()
    initial_lora_state = {
        name: param.detach().clone() for name, param in model.named_parameters() if "lora_" in name
    }

    # Gradient accumulation over MICRO_BATCH-sized sub-batches: a single
    # 8-row forward (no FSDP sharding here, deliberately -- see
    # _build_lora_model's docstring note) OOMs Qwen3.6-27B on one H200 even
    # under LoRA + gradient checkpointing + the fused loss (MEASURED). This
    # still trains on all 8 rows per "step", only the peak activation memory
    # shrinks; the step's reported loss is the mean over its micro-batches,
    # the standard gradient-accumulation convention.
    micro_batch = 2

    def run(net: Any, frames_for: Any, label: str) -> list[float]:
        with torch.no_grad():
            live = dict(net.named_parameters())
            for name, snapshot in initial_lora_state.items():
                live[name].copy_(snapshot)
        optimizer = torch.optim.AdamW([p for p in net.parameters() if p.requires_grad], lr=1e-4)
        collate = train_conversation_collator_or_refuse(
            processor,
            max_length=16384,
            pad_to_max_length=False,
            inject_dummy_media=False,
            frames_for=frames_for,
            video_column="video",
        )
        chunks = [rows[i : i + micro_batch] for i in range(0, len(rows), micro_batch)]
        losses: list[float] = []
        for _step in range(2):
            optimizer.zero_grad()
            step_losses: list[float] = []
            for chunk in chunks:
                batch = collate(chunk)
                batch = {
                    key: value.to(net.device) if hasattr(value, "to") else value
                    for key, value in batch.items()
                }
                loss = net(**batch).loss / len(chunks)
                loss.backward()
                step_losses.append(float(loss.detach().cpu()) * len(chunks))
            optimizer.step()
            losses.append(sum(step_losses) / len(step_losses))
        print(f"\n[video_control] family={family!r} condition={label} losses={losses}")
        return losses

    try:
        losses_real = run(model, _frames_for_real(budget), "real")
        losses_black = run(model, _frames_for_black(budget), "black")
    finally:
        # ALWAYS freed, even on assertion/OOM failure: a leftover reference
        # from a failed parametrization must not starve the NEXT family's
        # model load on this same (shared, single-GPU) process.
        del model
        import gc

        gc.collect()
        torch.cuda.empty_cache()

    print(
        f"\n[video_control] family={family!r} summary: "
        f"real={losses_real} black={losses_black}"
    )
    assert losses_real[0] != losses_black[0], (
        f"{family}: step-0 loss is IDENTICAL for real vs black video frames "
        f"({losses_real[0]!r} == {losses_black[0]!r}), starting from the SAME LoRA "
        "weights on the SAME text -- the model's forward pass does not appear to "
        "depend on the video pixels at all"
    )
