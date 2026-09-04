"""Shared test doubles for checkpoint-comparison tests.

Home: ``tests/_dcp_fakes.py``, not ``tests/conftest.py``.
Reason: ``MemSource`` is instantiated directly as a class, while
``conftest.py`` is for pytest wiring and fixtures.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

from foundationscale.checkpoint.dcp import Chunk, ReadResult


class MemSource:
    """In-memory ``WeightSource`` test double for ``compare_keys``.

    Implements the whole protocol surface so consuming tests exercise the real
    streaming, coverage and verdict logic. Refusing an empty mapping mirrors
    the real readers' non-empty guarantee: an empty source is the vacuous-truth
    bug one level up.
    """

    def __init__(self, tensors: dict[str, Any], path: str = "mem://test") -> None:
        if not tensors:
            raise ValueError("MemSource refuses an empty tensor set, as real readers do")
        self._tensors = dict(tensors)
        self.path = path

    def tensor_keys(self) -> tuple[str, ...]:
        return tuple(sorted(self._tensors))

    def nontensor_keys(self) -> tuple[str, ...]:
        return ()

    def shape(self, key: str) -> tuple[int, ...]:
        return tuple(int(d) for d in self._tensors[key].shape)

    def dtype(self, key: str) -> Any:
        return self._tensors[key].dtype

    def chunks(self, key: str) -> tuple[Chunk, ...]:
        shape = self.shape(key)
        return (Chunk(offsets=(0,) * len(shape), sizes=shape),)

    def read_chunk(self, key: str, offsets: Sequence[int]) -> Any:
        if any(int(o) != 0 for o in offsets):
            raise ValueError(f"MemSource tensors are single-chunk: {offsets}")
        return self._tensors[key]

    def read_box(self, key: str, lo: Sequence[int], hi: Sequence[int]) -> ReadResult:
        tensor = self._tensors[key]
        nd = tensor.dim()
        if len(lo) != nd or len(hi) != nd:
            raise ValueError(f"box rank mismatch for {key!r}: tensor is {nd}-D")
        idx = tuple(slice(int(lo_i), int(hi_i)) for lo_i, hi_i in zip(lo, hi, strict=True))
        block = tensor[idx]
        numel = math.prod(int(hi_i) - int(lo_i) for lo_i, hi_i in zip(lo, hi, strict=True))
        return ReadResult(
            key=key,
            tensor=block,
            chunks_read=1,
            elements_covered=int(numel),
            elements_expected=int(numel),
            bytes_read=block.numel() * block.element_size(),
        )

    def read_full(self, key: str) -> ReadResult:
        shape = self.shape(key)
        return self.read_box(key, (0,) * len(shape), shape)

    def close(self) -> None:
        return None
