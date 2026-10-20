"""The MEASURED serving envelope (A4.9): the engine x model family x mode x modality cells actually served on this estate.

A ledger of measured runs, never a capability guess. A deployment outside it is UNMEASURED, never PASS. Widen it only
with a new measured row and its evidence.

- ``mode``: ``full`` (base or merged full weights) or ``lora`` (base + adapter unmerged, ``--enable-lora``).
- ``fidelity``: whether the served output was measured to reproduce the trained model.
  - ``measured`` only where it was.
  - Served-LoRA fidelity is not measured yet: at the token level, bf16 noise (0.40 floor) dominates the comparison
    (served LoRA sits at 0.63), so it needs a task-level held-out check.
"""

from __future__ import annotations

from typing import Any

ENGINE = "vllm-0.31.0-arm64"  # the vllm031 enroot container (vllm/vllm-openai:v0.31.0, arm64)

MEASURED_SERVING: tuple[dict[str, Any], ...] = (
    {"engine": ENGINE, "family": "qwen3_5", "mode": "full", "modality": "text", "fidelity": "unmeasured",
     "evidence": "B2.2 + B6.1: Qwen3.5-4B and Qwen3.8-27B served as pi agent endpoints on GB200 (tool parser qwen3_coder)"},
    {"engine": ENGINE, "family": "gemma4_unified", "mode": "full", "modality": "text", "fidelity": "unmeasured",
     "evidence": "A4.9 a49/FINDINGS.md: gemma-4-12B-it served on GB200, prompt logprobs returned"},
    {"engine": ENGINE, "family": "gemma4_unified", "mode": "lora", "modality": "text", "fidelity": "unmeasured",
     "evidence": "A4.9 a49/FINDINGS.md: gemma-4-12B-it + A4.4 LoRA unmerged (--enable-lora, rank 16); LoRA applied "
                 "(served LoRA vs base 0.91 nats/token); token-level fidelity inside bf16 noise, not established"},
)


def lookup(family: str, mode: str, modality: str, engine: str = ENGINE) -> dict[str, Any] | None:
    """The measured cell for this deployment, or None when it has never been served here."""
    for cell in MEASURED_SERVING:
        if (cell["engine"], cell["family"], cell["mode"], cell["modality"]) == (engine, family, mode, modality):
            return dict(cell)
    return None
