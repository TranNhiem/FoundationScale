"""Campaign guard: without the checkpoints this campaign is UNMEASURED (exit 95).

These suites need real checkpoints (and CUDA for the parity test), which is why they
live outside ``tests/``. A run that cannot reach them must not report a green
"all skipped" session, so the session ends with exit 95 before collection.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

_REQUIRED = ("gemma-4-12B-it", "Qwen3.6-27B")


def pytest_sessionstart() -> None:
    models_dir = os.environ.get("FS_TEST_MODELS_DIR")
    missing = [m for m in _REQUIRED if not models_dir or not (Path(models_dir) / m).is_dir()]
    if missing:
        pytest.exit(
            f"FS_TEST_MODELS_DIR={models_dir!r} lacks {missing}: vlm_2xh200 campaign UNMEASURED",
            returncode=95,
        )
