"""FoundationScale's thin training entry.

Torch stays an optional extra: importing this package pulls in NOTHING heavy.
The implementation lives in :mod:`foundationscale.train.loop` and is loaded
lazily via PEP 562 so ``import foundationscale.train`` stays torch-free.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "EXIT_PASS",
    "EXIT_RED",
    "EXIT_REFUSE",
    "EXIT_UNMEASURED",
    "FoundationScaleSaveGate",
    "MARKERS",
    "TrainConfig",
    "train",
]


def __getattr__(name: str) -> Any:
    if name in __all__:
        from foundationscale.train import loop

        return getattr(loop, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
