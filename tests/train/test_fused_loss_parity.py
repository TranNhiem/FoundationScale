"""GPU-only parity check for --fused-loss liger (train/fused_loss.py).

Skipped entirely unless CUDA is available AND FS_TEST_MODELS_DIR points at a
directory containing full weights for the three families this flag covers on
this estate. Mirrors test_conversation_collator.py's FS_TEST_MODELS_DIR
convention, but needs full model WEIGHTS here (a real forward pass), not just
processor/tokenizer config::

    FS_TEST_MODELS_DIR=/workspace/fs-vlm/models \
        /workspace/fs-vlm/venv/bin/python -m pytest -s \
        tests/train/test_fused_loss_parity.py

For each covered family, the SAME model and the SAME real-text batch
(~2k tokens) are run twice -- once with the stock forward, once after
apply_fused_loss patches the instance -- and the two losses are compared.
|diff| <= 1e-2 RELATIVE is the tolerance the task specifies; both losses are
printed (pytest -s) so the numbers are part of the run's own evidence, not
just its verdict.

Deliberately single-GPU and no_grad for all three families, including the
larger gemma4 arm: only a FORWARD pass and a loss scalar are needed for this
comparison (no optimizer state, no backward graph to retain), so the
"2-rank FSDP" alternative the task allows is unnecessary here -- loading one
set of bf16 weights plus a short/medium sequence's activations fits on a
single measured-free H200.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest

from foundationscale.train.fused_loss import apply_fused_loss, fused_loss_refusal_reason

MODELS_DIR = os.environ.get("FS_TEST_MODELS_DIR")

# One family per LIGER_GENERIC_MODEL_TYPES entry that actually has weights on
# this estate, plus the LIGER_CUSTOM_MODEL_TYPES family (gemma4_unified).
# gemma-4-26B-A4B-it is preferred over -31B-it for the second gemma4 arm
# (smaller download); either satisfies the task's "one gemma4" requirement.
_FAMILIES: tuple[str, ...] = ("gemma-4-12B-it", "gemma-4-26B-A4B-it", "Qwen3.6-27B")

_PARITY_TEXT = (
    "The history of distributed training systems traces a path from simple "
    "data parallelism, where every worker holds a full replica of the model "
    "and only gradients are synchronized, toward increasingly disaggregated "
    "designs that shard parameters, optimizer state, and activations across "
    "memory domains that were never meant to behave like one machine. "
) * 40  # ~2k tokens after tokenization for every family measured here


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


def _parity_available() -> bool:
    return _cuda_available() and all(_family_dir(name) is not None for name in _FAMILIES)


requires_gpu_and_models = pytest.mark.skipif(
    not _parity_available(),
    reason=(
        "needs CUDA and FS_TEST_MODELS_DIR containing full weights for "
        f"{_FAMILIES!r}; set FS_TEST_MODELS_DIR and run on a GPU host"
    ),
)


def _build_batch(tokenizer: Any, device: Any) -> dict[str, Any]:
    encoded = tokenizer(
        _PARITY_TEXT,
        return_tensors="pt",
        truncation=True,
        max_length=2048,
    )
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    labels = input_ids.clone()
    labels[attention_mask == 0] = -100
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def _measure_loss(model: Any, batch: dict[str, Any]) -> float:
    import torch

    model.train()
    with torch.no_grad():
        outputs = model(**batch)
    return float(outputs.loss.detach().cpu())


@requires_gpu_and_models
@pytest.mark.parametrize("family", _FAMILIES)
def test_fused_loss_parity(family: str) -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_dir = str(_family_dir(family))
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.bfloat16).to("cuda")
    model_type = getattr(model.config, "model_type", None)
    assert fused_loss_refusal_reason("liger", model_type) is None, (
        f"{family} (model_type={model_type!r}) is expected to be covered by "
        "--fused-loss liger; this test's family list is stale"
    )

    batch = _build_batch(tokenizer, model.device)
    loss_before = _measure_loss(model, batch)

    apply_fused_loss(model, model_type)
    loss_after = _measure_loss(model, batch)

    rel_diff = abs(loss_after - loss_before) / max(abs(loss_before), 1e-8)
    print(
        f"\n[fused_loss_parity] family={family} model_type={model_type!r} "
        f"loss_before={loss_before:.6f} loss_after={loss_after:.6f} "
        f"rel_diff={rel_diff:.6f}"
    )
    assert rel_diff <= 1e-2, (
        f"{family}: fused-loss relative difference {rel_diff:.6f} exceeds the "
        f"1e-2 tolerance (loss_before={loss_before!r}, loss_after={loss_after!r})"
    )

    del model
    torch.cuda.empty_cache()


@requires_gpu_and_models
def test_gemma4_unified_fused_loss_eval_mode_and_return_dict() -> None:
    """The two behaviours only gemma4_unified's OWN forward controls (train/fused_loss.py item 6).

    (a) eval mode (``model.eval()``) WITH labels still takes the fused path
    (``outputs.logits is None``, matching the skip-logits convention) --
    the whole point being that an eval pass at long context must not
    materialize the full vocab-sized logits tensor either.
    (b) ``return_dict=False`` returns a plain tuple, not the dataclass, and
    does not crash on the kwarg collision the stock call's own
    ``return_dict=True`` would otherwise cause.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_dir = str(_family_dir("gemma-4-12B-it"))
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.bfloat16).to("cuda")
    model_type = getattr(model.config, "model_type", None)
    assert model_type == "gemma4_unified"
    apply_fused_loss(model, model_type)

    batch = _build_batch(tokenizer, model.device)
    model.eval()
    with torch.no_grad():
        outputs = model(**batch)
    assert outputs.loss is not None
    assert outputs.logits is None, (
        "eval mode with labels must still take the fused path and skip "
        "materializing full logits -- this is the exact behaviour item 6(b) adds"
    )

    with torch.no_grad():
        tuple_outputs = model(**batch, return_dict=False)
    assert isinstance(tuple_outputs, tuple)
    assert torch.allclose(tuple_outputs[0], outputs.loss)

    del model
    torch.cuda.empty_cache()
