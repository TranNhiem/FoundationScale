"""Array transport between FS and a backend's policy server running in another environment.

Arrays travel as builtins only -- dtype string, shape, raw bytes -- so a pickle never carries a
NumPy object across environments that pin different NumPy versions (Long-WAM pins NumPy 1.x, the
FS eval environment may not).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np

__all__ = ["DEFAULT_AUTHKEY", "decode_array", "encode_array"]

# shared by the FS policy servers (multiprocessing.connection authkey; local hosts only)
DEFAULT_AUTHKEY = b"fs-longwam"


def encode_array(array: Any) -> dict[str, Any]:
    """An array as builtins only (dtype string, shape, raw bytes): safe across NumPy versions."""
    import numpy as np  # noqa: PLC0415

    contiguous = np.ascontiguousarray(array)
    return {
        "dtype": contiguous.dtype.str,
        "shape": list(contiguous.shape),
        "data": contiguous.tobytes(),
    }


def decode_array(payload: Mapping[str, Any]) -> np.ndarray:
    """The inverse of :func:`encode_array`, as a writable copy."""
    import numpy as np  # noqa: PLC0415

    flat = np.frombuffer(payload["data"], dtype=np.dtype(payload["dtype"]))
    return flat.reshape(tuple(payload["shape"])).copy()
