"""Non-finite payloads must never launder ``compare_keys`` into a pass-grade verdict.

Why this file exists
--------------------
The defect pinned here (audit W-io-1): a single NaN on either side of
:func:`compare_keys` produced ``verdict="CLOSE"`` with ``max_abs_diff=0.0`` —
the machine-consumed pass grade — because three IEEE behaviours composed into
a lie: ``torch.equal`` refuses bitwise identity on NaN, ``max(0.0, nan)``
kept the running maximum at ``0.0``, and the NaN-poisoned norms forced
``cosine=None``, which satisfied the ``cosine is None or ...`` arm of the
CLOSE condition. A poisoned expert shard — the content the 87.5%-wrong
incident artifact was made of — was reported as *passing with zero
difference*.

Every test below failed against the pre-fix comparator:
``test_nan_never_close`` saw ``CLOSE``/``0.0`` where the fixed code reports
``NON_FINITE``/``inf``; the identical-±inf parity test died inside the
bitwise self-check, an ``AssertionError`` raised on content whose bytes
genuinely agreed; and the strict-JSON projections raised ``ValueError`` on
the ``inf`` the report itself carries. The in-memory source is the same full
``WeightSource`` protocol implementation used by the comparator's other
tests — the real code path, not a paraphrase.
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from typing import Any

import pytest

from foundationscale.checkpoint.dcp import (
    VERDICT_CLOSE,
    VERDICT_EXACT,
    VERDICT_NON_FINITE,
    Chunk,
    ReadResult,
    compare_keys,
)

# ---------------------------------------------------------------------------
# In-memory WeightSource double (full protocol — the real streaming path runs)
# ---------------------------------------------------------------------------


class MemSource:
    """In-memory ``WeightSource`` test double for ``compare_keys``.

    Implements the whole protocol surface so the tests below exercise the
    real streaming, coverage and verdict logic. Refusing an empty mapping
    mirrors the real readers' non-empty guarantee: an empty source is the
    vacuous-truth bug one level up.
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


# ---------------------------------------------------------------------------
# The defect trio: every NaN shape must block, at any stream granularity
# ---------------------------------------------------------------------------


def test_nan_never_close() -> None:
    """Positive control: a comparator that reports CLOSE over NaN is dead."""
    torch: Any = pytest.importorskip("torch")
    nan = float("nan")
    cases = {
        "nan_left_only": (
            torch.tensor([[nan, 1.0], [2.0, 3.0]]),
            torch.tensor([[0.5, 1.0], [2.0, 3.0]]),
            1,  # non-finite elements across both sides
            1,  # mismatched elements (nan != 0.5; the rest agree)
        ),
        "nan_both_same_positions": (
            torch.tensor([[nan, 1.0]]),
            torch.tensor([[nan, 1.0]]),
            2,
            1,  # nan != nan under IEEE, so the position counts as mismatched
        ),
        "nan_plus_large_finite_divergence": (
            torch.tensor([[nan, 100.0]]),
            torch.tensor([[0.5, -100.0]]),
            1,
            2,
        ),
    }
    for name, (ta, tb, nonfinite, mismatched) in cases.items():
        # block_rows=1 rechunks the 2-row case: poison must survive streaming.
        for block_rows in (4096, 1):
            cmp = compare_keys(
                MemSource({"w": ta}), MemSource({"w": tb}), "w", block_rows=block_rows
            )
            assert cmp.verdict == VERDICT_NON_FINITE, (
                f"{name} (block_rows={block_rows}): {cmp.to_dict()}"
            )
            assert cmp.verdict not in {VERDICT_CLOSE, VERDICT_EXACT}
            assert math.isinf(cmp.max_abs_diff), (
                f"{name}: max_abs_diff={cmp.max_abs_diff!r} launders poison as agreement"
            )
            assert cmp.cosine is None  # no finite direction exists to measure
            assert cmp.nonfinite_elements == nonfinite
            assert cmp.mismatched_elements == mismatched
            # A strict JSON consumer must survive the report of the poison.
            json.dumps(cmp.to_dict(), allow_nan=False)


def test_identical_finite_pair_stays_exact_with_zero_poison() -> None:
    """Negative control: the poison guard must not swallow a healthy EXACT.

    Pre-fix this fails on the surfacing field alone (``nonfinite_elements``
    did not exist); post-fix it proves the NON_FINITE branch stays quiet on
    legitimate content and the EXACT ground truth is unmoved.
    """
    torch: Any = pytest.importorskip("torch")
    gen = torch.Generator().manual_seed(29)
    t = torch.randn(64, 32, generator=gen)
    cmp = compare_keys(MemSource({"w": t}), MemSource({"w": t.clone()}), "w")

    assert cmp.verdict == VERDICT_EXACT
    assert cmp.bitwise_equal is True
    assert cmp.cosine == 1.0  # unchanged: the fixed code keeps finite parity exact
    assert cmp.nonfinite_elements == 0
    json.dumps(cmp.to_dict(), allow_nan=False)


def test_identical_inf_bytes_are_parity_not_a_self_check_crash() -> None:
    """Bitwise equality is a statement about bytes, including poisoned ones.

    Pre-fix the self-check fired on identical ±inf (no finite cosine exists,
    and the guard demanded one): ``compare_keys`` raised ``AssertionError``
    on content whose bytes agreed. Post-fix it reports EXACT with the poison
    counted, not hidden, and with no fabricated cosine.
    """
    torch: Any = pytest.importorskip("torch")
    t = torch.tensor([[float("inf"), 1.0], [-float("inf"), 2.0]])
    cmp = compare_keys(MemSource({"w": t}), MemSource({"w": t.clone()}), "w")

    assert cmp.verdict == VERDICT_EXACT
    assert cmp.bitwise_equal is True
    assert cmp.mismatched_elements == 0
    assert cmp.max_abs_diff == 0.0  # bytes agree; NaN in |inf-inf| cannot contradict parity
    assert cmp.nonfinite_elements == 4  # two infinities per side, counted together
    assert cmp.cosine is None  # no finite norm exists; inventing 1.0 would be a lie
    json.dumps(cmp.to_dict(), allow_nan=False)


def test_shape_mismatch_projection_survives_strict_json() -> None:
    """to_dict carries ``math.inf`` for SHAPE_MISMATCH; strict encoders must not crash."""
    torch: Any = pytest.importorskip("torch")
    cmp = compare_keys(
        MemSource({"w": torch.zeros(2, 3)}), MemSource({"w": torch.zeros(2, 4)}), "w"
    )

    assert cmp.max_abs_diff == math.inf  # the fact on the object is unchanged
    payload = json.dumps(cmp.to_dict(), allow_nan=False)
    assert '"inf"' in payload  # encoded by name, never a bare Infinity token
