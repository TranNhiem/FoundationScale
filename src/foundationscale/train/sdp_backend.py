"""Pure SDPA-backend pin logic for the #526/#527 finding.

train/loop.py owns the GLUE (print, _mark, _emit_manifest, os.environ);
this module owns every DECISION and every STRING, and it never imports
torch. WHY the split: the binding must happen inside train(), beside
attn_implementation and before from_pretrained, which puts the binding
site beyond the reach of any test that cannot import transformers and
CUDA. Factoring the decisions here -- each taking the torch.backends.cuda
object as a plain argument -- keeps the whole pin contract testable with
a recording fake. A gate that inspected nothing did not pass; this module
is what lets the inspection happen at all.
"""

from __future__ import annotations

from typing import Any, Literal

# One toggle per backend, and torch records NOTHING about which backend it
# ended up selecting -- that absence of a record is the measured defect,
# which is why every claim below is built from what THIS process toggled,
# never from asking torch after the fact.
SDP_BACKEND_TOGGLES: dict[str, str] = {
    "flash": "enable_flash_sdp",
    "mem_efficient": "enable_mem_efficient_sdp",
    "cudnn": "enable_cudnn_sdp",
    "math": "enable_math_sdp",
}

# The reason string IS the manifest value on the unpinned path: the
# contract for source == "unmeasured" is that `value` carries the reason,
# so the reason names the mechanism (torch selects per shape) and the
# consequence (not recorded; multi-GPU not reproducible), not merely None.
UNPINNED_REASON: str = (
    "unpinned: torch selects the SDPA backend per shape at runtime; not "
    "pinned or recorded by this run, so multi-GPU losses are not "
    "reproducible run-to-run (#526/#527)"
)


def sdp_pin_refusal_reason(backend: str, cuda_backends: Any, torch_version: str) -> str | None:
    """Return None iff the declared backend's toggle exists, else the 96 reason.

    WHY a refusal rather than a silent fallback: a pin that did not take,
    under a declaration that says it took, restores trust in single runs --
    the exact failure class the finding measured. The reason names the
    backend AND the torch version, so the refusal is itself a measurement
    the manifest can carry instead of a shrug.
    """
    toggle = SDP_BACKEND_TOGGLES[backend]
    if callable(getattr(cuda_backends, toggle, None)):
        return None
    return (
        f"sdp_backend={backend!r} declared, but this torch build "
        f"(torch {torch_version}) exposes no torch.backends.cuda.{toggle}; "
        "refusing (96) rather than leaving all four backends enabled under "
        "a pin that silently did not take -- the defect class of #526/#527"
    )


def apply_sdp_pin(backend: str, cuda_backends: Any) -> None:
    """Enable EXACTLY the declared backend and disable the other three.

    WHY disable rather than merely enable: the four toggles form a
    permissive mask, and torch still chooses per shape among everything
    left enabled -- enabling one while three stay on changes nothing the
    finding measured. A missing sibling toggle raises AttributeError out
    of this loop, loudly, which is correct: the declared toggle was
    already vetted by sdp_pin_refusal_reason, and a mask the process only
    half-set must never pass for a pin.
    """
    for name, toggle in SDP_BACKEND_TOGGLES.items():
        getattr(cuda_backends, toggle)(name == backend)


def sdp_pinned_announcement(backend: str) -> str:
    """Announcement for the pinned path: what is on, and what was turned off.

    WHY the disabled three are named: "pinned" is a claim about a MASK,
    and a mask claim that omits the disabled backends is half a claim --
    the reader could otherwise not distinguish "only math on" from "math
    on, who knows about the rest".
    """
    others = ", ".join(sorted(SDP_BACKEND_TOGGLES.keys() - {backend}))
    return (
        f"sdp_backend pinned: {backend} -- "
        f"torch.backends.cuda.{SDP_BACKEND_TOGGLES[backend]}(True) applied, "
        f"{others} disabled; recorded as measured in the run manifest (#526/#527)"
    )


def sdp_unpinned_announcement() -> str:
    """Announcement for the unpinned path -- an abstain path, so it must TALK.

    WHY the 94% number rides along: "torch decides" is easy to read as
    harmless, and the measured consequence is what makes the warning
    honest instead of boilerplate. The run is NOT refused -- unpinned is
    today's default and remains legitimate -- but silence is not an
    option; a silent default is the defect this repo hunts.
    """
    return (
        "sdp_backend UNPINNED: torch selects the SDPA backend per shape "
        "at runtime; measured on GB200 (#526/#527) the silently-chosen "
        "backend was load-bearing and DP=4 train_loss differed run-to-run "
        "by up to 94%, so multi-GPU losses from this run are NOT "
        "reproducible run-to-run. The run proceeds -- unpinned is "
        "today's default -- and the manifest records the axis unmeasured, "
        "with the reason."
    )


def sdp_backend_telemetry_pair(
    pinned: str | None,
) -> tuple[str, Literal["measured", "unmeasured"]]:
    """Build (value, source) for the manifest's sdp_backend TelemetryEntry.

    WHY the argument is the APPLIED pin rather than the config
    declaration: the only honest witness for "measured" is that this
    process actually set the mask. Config alone cannot distinguish pinned
    from pinned-in-intention, and that indistinguishability is the entire
    finding -- a manifest that recorded the declaration as the measurement
    would repeat it.
    """
    if pinned is not None:
        return pinned, "measured"
    return UNPINNED_REASON, "unmeasured"
