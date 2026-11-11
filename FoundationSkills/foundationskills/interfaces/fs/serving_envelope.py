"""The MEASURED serving envelope (A4.9): the engine x model family x mode x modality cells actually served on this estate.

A ledger of measured runs, never a capability guess. A deployment outside it is UNMEASURED, never PASS. Widen it only
with a new measured row and its evidence.

- ``mode``: ``full`` (base or merged full weights) or ``lora`` (base + adapter unmerged, ``--enable-lora``).
- ``quantization``: ``None`` for unquantized weights, else ``quant_method:format`` from the checkpoint's
  quantization_config. A quantized checkpoint never inherits an unquantized row's evidence.
- ``fidelity``: whether the served output was measured to reproduce the trained model.
  - ``measured`` only where it was.
  - Served-LoRA fidelity is measured at the task level (held-out NLL), never from per-token logprobs, which bf16 noise
    dominates (floor 0.40). gemma4 LoRA keeps 99.5% of the adapter's gain.
"""

from __future__ import annotations

from typing import Any

ENGINE = "vllm-0.31.0-arm64"  # the vllm031 enroot container (vllm/vllm-openai:v0.31.0, arm64)

MEASURED_SERVING: tuple[dict[str, Any], ...] = (
    {"engine": ENGINE, "family": "qwen3_5", "mode": "full", "modality": "text", "fidelity": "unmeasured",
     "evidence": "B2.2 + B6.1: Qwen3.5-4B and Qwen3.8-27B served as pi agent endpoints on GB200 (tool parser qwen3_coder)"},
    {"engine": ENGINE, "family": "gemma4_unified", "mode": "full", "modality": "text", "fidelity": "unmeasured",
     "evidence": "A4.9 a49/FINDINGS.md: gemma-4-12B-it served on GB200, prompt logprobs returned"},
    {"engine": ENGINE, "family": "gemma4_unified", "mode": "lora", "modality": "text", "fidelity": "measured",
     "evidence": "A4.9 a49/FINDINGS.md: gemma-4-12B-it + A4.4 LoRA unmerged (--enable-lora, rank 16); task-level held-out "
                 "NLL over 1,080 unseen answer tokens: served LoRA keeps 99.5% of the local adapter's gain (pre-registered "
                 "band 0.95-1.05; engine sanity 0.004 nats); served merged checkpoints keep 95.4%"},
    {"engine": ENGINE, "family": "gemma4", "mode": "full", "modality": "text", "fidelity": "measured",
     "evidence": "A4.8 v2 serving step: gemma-4-E4B-it served on GB200 (r04dgx04, leased UUID), names listed, smoke ok; "
                 "served wikitext-2 ppl 64.539 vs local 64.707 (rel diff 0.26%, 65,504 tokens, 0 missing)"},
    {"engine": ENGINE, "family": "gemma4", "mode": "full", "modality": "text", "fidelity": "measured",
     "quantization": "compressed-tensors:pack-quantized",
     "evidence": "A4.8 v2 serving step: GPTQ W4A16/g128 gemma-4-E4B-it (written by compressed-tensors 0.18) loads in the "
                 "container's compressed-tensors 0.17 on GB200; served ppl 68.076 vs local quantized 68.333 (0.38%)"},
)


def lookup(family: str, mode: str, modality: str, engine: str = ENGINE,
           quantization: str | None = None) -> dict[str, Any] | None:
    """The measured cell for this deployment, or None when it has never been served here."""
    for cell in MEASURED_SERVING:
        if ((cell["engine"], cell["family"], cell["mode"], cell["modality"], cell.get("quantization"))
                == (engine, family, mode, modality, quantization)):
            return dict(cell)
    return None
