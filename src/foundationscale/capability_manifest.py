"""Capabilities this package declares for the FoundationSkills probe (A5.2).

A declaration is never trusted on its own: each entry names the zero-arg ``verify`` callable the
probe runs to measure it, and how the measurement must ``compare`` with what is ``declared``
(equal | subset: measured within declared and non-empty | superset: measured covers declared).
Keep this module import-light: the probe imports it just to read ``MANIFEST``.
"""

from __future__ import annotations

from typing import Any

MANIFEST: list[dict[str, Any]] = [
    {
        "key": "video.decode_backends",
        # every decoder foundationscale.video can drive; the probe measures which are installed
        "declared": ["torchcodec", "av", "cv2", "ffmpeg"],
        "compare": "subset",
        "verify": "foundationscale.video:available_backends",
        "doc": "decoders the SFT video path can use here, tried in this order",
    },
]
