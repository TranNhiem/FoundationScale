"""Atomic result-file writes for the NeMo CLIs (review 2026-11-28)."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path


def write_text_atomic(path: Path, text: str) -> None:
    """Write ``text`` to ``path`` so a reader never sees a truncated or half-replaced file.

    The bytes go to a uniquely named temp file in the same directory (``O_EXCL``), are flushed and
    fsynced, and then ``os.replace`` swaps it in; a failure leaves the previous file untouched.
    """
    path = Path(path)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent or ".")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        Path(tmp).replace(path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
