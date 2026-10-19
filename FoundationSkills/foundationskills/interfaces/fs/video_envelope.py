"""The MEASURED video frame envelope (A4.4): the frame counts FoundationScale has actually trained at.

FS samples N centred-uniform frames per clip (``foundationscale.video``) and leaves N to the caller
(``FOUNDATIONSCALE_TRAIN_VIDEO_FRAMES``); it measures no envelope itself. The envelope here is a ledger of
measured runs, never a capability guess: the mixed image + video + text matrix ran every listed family at
16 frames per clip at 16384 and 32768 tokens (exit 0) and longer contexts with the same 16 frames
(validation_campaigns/vlm_2xh200/EVIDENCE.md:58-76, 86-92). A frame count above ``max_frames`` has never been
run, so every skill reports it UNMEASURED, never PASS. Widen the envelope only with a new measured row.
"""

from __future__ import annotations

from typing import Any

MEASURED_FRAME_ENVELOPE: dict[str, Any] = {
    "max_frames": 16,
    "sampling": "uniform",
    "families": ["gemma-4-12B-it", "gemma-4-26B-A4B-it", "gemma-4-31B-it", "Qwen3.6-27B", "Qwen3.6-35B-A3B"],
    "max_sequence_lengths": [16384, 32768],
    "evidence": "validation_campaigns/vlm_2xh200/EVIDENCE.md:58-76",
}


def frame_envelope() -> dict[str, Any]:
    """A copy of the measured envelope (callers may record it; nobody may widen it in place)."""
    return {k: list(v) if isinstance(v, list) else v for k, v in MEASURED_FRAME_ENVELOPE.items()}


def beyond_envelope(frames: Any) -> str | None:
    """The reason a declared frame count is UNMEASURED, or None when it lies inside the measured envelope.

    Only a positive integer above ``max_frames`` is judged here; a missing or malformed count is the
    declaration gates' job (TR-IN-006, FS-IN-006, EV-IN-007), never silently inside the envelope.
    """
    if isinstance(frames, bool):
        return None
    try:
        n = int(frames)
    except (TypeError, ValueError):
        return None
    limit = MEASURED_FRAME_ENVELOPE["max_frames"]
    if n <= limit:
        return None
    return (f"video_frames {n} exceeds the measured frame envelope ({limit} uniform frames per clip, "
            f"{MEASURED_FRAME_ENVELOPE['evidence']}): no run has trained at {n} frames, so it is UNMEASURED, "
            f"never PASS")
