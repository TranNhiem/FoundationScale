"""Zero-element tensors must abstain from ``compare_keys``, never pass as EXACT.

Why this file exists
--------------------
The comparator returned ``verdict="EXACT"`` with ``elements=0``,
``chunks_read=0``, ``bytes_read=0`` for two identical-shape ``(0, 2816)``
tensors: ``rows`` was 0, the stream loop never ran, and the ``bitwise =
True`` initialiser survived unopposed — the founding ``all([])`` incident
reproduced inside the framework's flagship comparator. The authors had
already defended the *adjacent* case: ``SHAPE_MISMATCH`` returns
``bitwise_equal=False``, unbounded diffs, ``elements=0`` "by design visibly
so". The fix extends that established "I compared nothing" expression to the
shape-*agreeing* zero-element case as a new abstaining verdict,
``NO_ELEMENTS``, which pre-fix consumers fail closed on by construction.
"""

from __future__ import annotations

import json
import math
from typing import Any

import pytest
from test_compare_keys_nonfinite import MemSource

from foundationscale.checkpoint import dcp
from foundationscale.checkpoint.dcp import (
    VERDICT_CLOSE,
    VERDICT_DIFFER,
    VERDICT_EXACT,
    VERDICT_NON_FINITE,
    VERDICT_SHAPE_MISMATCH,
    compare_keys,
)


def test_zero_height_tensors_abstain_instead_of_exact() -> None:
    """Positive control: the measured reproduction, two (0, 2816) bf16 tensors.

    Doctrine 1 demands a blocking/abstaining verdict that names 0 as the
    number examined; ``elements=0`` is that denominator.
    """
    torch: Any = pytest.importorskip("torch")
    t = torch.zeros(0, 2816, dtype=torch.bfloat16)
    cmp = compare_keys(MemSource({"w": t}), MemSource({"w": t.clone()}), "w")

    assert cmp.verdict == "NO_ELEMENTS", f"vacuous pass: {cmp.to_dict()}"
    assert cmp.elements == 0  # the abstention names the number examined
    assert cmp.chunks_read == 0
    assert cmp.bytes_read == 0
    assert cmp.bitwise_equal is False  # identity over unread content is not a fact
    assert math.isinf(cmp.max_abs_diff)  # the SHAPE_MISMATCH convention, unchanged
    assert cmp.cosine is None
    assert cmp.verdict not in {
        VERDICT_EXACT,
        VERDICT_CLOSE,
        VERDICT_DIFFER,
        VERDICT_NON_FINITE,
        VERDICT_SHAPE_MISMATCH,
    }  # a pre-fix consumer must meet a token it cannot map to any grade
    json.dumps(cmp.to_dict(), allow_nan=False)


def test_every_zero_extent_shape_family_abstains() -> None:
    """Family sweep: any zero extent anywhere means zero elements compared.

    (0,), (0, 0), (0, 5, 7) are the loop-never-runs family; (5, 0),
    (2, 0, 7), (7, 3, 0) are the family where rows > 0 made the loop execute
    vacuous blocks — the variant that launders "work happened" with
    chunks_read > 0.
    """
    torch: Any = pytest.importorskip("torch")
    shapes: list[tuple[int, ...]] = [
        (0,),
        (0, 0),
        (0, 5, 7),
        (5, 0),
        (2, 0, 7),
        (7, 3, 0),
    ]
    for shape in shapes:
        t = torch.zeros(*shape, dtype=torch.float32)
        cmp = compare_keys(MemSource({"w": t}), MemSource({"w": t.clone()}), "w")
        assert cmp.verdict == "NO_ELEMENTS", f"shape {shape}: {cmp.to_dict()}"
        assert cmp.elements == 0
        assert cmp.chunks_read == 0, (
            f"shape {shape}: the loop executed for a zero-element comparison; "
            f"chunks_read={cmp.chunks_read} launders work that compared nothing"
        )
        assert cmp.bytes_read == 0
        assert cmp.bitwise_equal is False
        assert math.isinf(cmp.max_abs_diff)
        assert math.isinf(cmp.mean_abs_diff)
        json.dumps(cmp.to_dict(), allow_nan=False)


def test_zero_d_scalar_still_compares_its_one_element() -> None:
    """Negative control: the abstention must not swallow a real comparison.

    A 0-D tensor has ``math.prod(()) == 1`` element, and that element is
    genuinely read, so EXACT remains the right verdict. If the new branch
    misfired on rank-0 tensors, this turns red.
    """
    torch: Any = pytest.importorskip("torch")
    t = torch.tensor(3.5, dtype=torch.float32)
    cmp = compare_keys(MemSource({"w": t}), MemSource({"w": t.clone()}), "w")

    # The export this control defends: it must exist, and it must never
    # appear on content-bearing comparisons.
    assert getattr(dcp, "VERDICT_NO_ELEMENTS", None) == "NO_ELEMENTS"
    assert cmp.verdict == VERDICT_EXACT
    assert cmp.elements == 1
    assert cmp.chunks_read == 2  # one block per source: the loop really ran
    assert cmp.bytes_read == 8  # four float32 bytes per side, moved for real
    assert cmp.bitwise_equal is True
    assert cmp.cosine == 1.0
    assert cmp.nonfinite_elements == 0
    json.dumps(cmp.to_dict(), allow_nan=False)


class _LyingShapeSource:
    """A source whose ``shape()`` reports extents no writer could produce.

    ``DcpReader.shape()`` is ``tuple(int(d) for d in self._meta(key).size)``
    over an unvalidated pickle, so a corrupt or hostile artifact reaches
    ``compare_keys`` exactly this way. ``read_box`` asserts because the whole
    point is that no read may be attributed to a shape that describes nothing.
    """

    def __init__(self, shape: tuple[int, ...], path: str = "mem://lying") -> None:
        self._shape = shape
        self.path = path

    def tensor_keys(self) -> tuple[str, ...]:
        return ("w",)

    def shape(self, key: str) -> tuple[int, ...]:
        return self._shape

    def dtype(self, key: str) -> Any:
        import torch

        return torch.float32

    def read_box(self, *args: Any, **kwargs: Any) -> Any:
        raise AssertionError("a read occurred against a negative-extent shape")


@pytest.mark.parametrize(
    "shape",
    [
        (-1,),  # the residual EXACT-over-zero-reads slice, first dim negative
        (-1, 2816),  # the realistic corrupt-embedding shape
        (3, -1),  # trailing negative: previously died inside _validate_box
        (0, -5),  # prod == 0: previously laundered corrupt as merely empty
        (-2, -3),  # prod == 6 > 0: previously read boxes off a fictional geometry
    ],
)
def test_negative_extent_is_a_format_error_not_a_verdict(shape: tuple[int, ...]) -> None:
    """MUST_FIRE: corrupt extents may be rendered as neither identity nor divergence.

    Pre-fix, ``(-1,)`` earned ``EXACT`` with ``elements=-1`` and
    ``chunks_read=0``: ``total == -1`` slipped past the ``total == 0``
    abstention and ``range(0, -1, block_rows)`` never ran, so the ``bitwise =
    True`` initialiser survived — the founding ``all([])`` incident one
    predicate below the guard that claims to close it. ``(0, -5)`` is the
    benign half of the same gap: an honest abstention, but it reports corrupt
    metadata as an empty tensor.
    """
    pytest.importorskip("torch")
    src = _LyingShapeSource(shape)

    with pytest.raises(dcp.CheckpointFormatError) as excinfo:
        compare_keys(src, _LyingShapeSource(shape), "w")

    msg = str(excinfo.value)
    assert "negative extent" in msg
    assert repr(shape) in msg  # the operator is told what the artifact claimed
    assert "key='w'" in msg and "mem://lying" in msg  # and where to look


def test_negative_extent_is_not_laundered_as_shape_mismatch() -> None:
    """Doctrine 5 symmetry: do not fix a false pass by minting a false failure.

    A corrupt ``(-1,)`` against a healthy ``(3,)`` compares equal to neither.
    Returning ``SHAPE_MISMATCH`` would assert an observed difference between a
    real tensor and a description of nothing, which is as unfounded as the
    ``EXACT`` this fix removes. Hence validation above the mismatch branch.
    """
    torch: Any = pytest.importorskip("torch")

    with pytest.raises(dcp.CheckpointFormatError):
        compare_keys(
            _LyingShapeSource((-1,), path="mem://corrupt"),
            MemSource({"w": torch.tensor([1.0, 2.0, 3.0])}),
            "w",
        )

    # ...and in the mirror direction, so neither argument position is special.
    with pytest.raises(dcp.CheckpointFormatError):
        compare_keys(
            MemSource({"w": torch.tensor([1.0, 2.0, 3.0])}),
            _LyingShapeSource((-1,), path="mem://corrupt"),
            "w",
        )


def test_healthy_shapes_still_compare_normally() -> None:
    """MUST_PASS: the extent guard is inert on every shape a real writer emits.

    A detector without a negative control is unproven. These four shapes span
    rank-0, rank-1, the zero-element abstention and a genuine 2-D weight; if
    the ``d < 0`` sweep ever fires on one of them the guard is over-broad and
    would block healthy checkpoints.
    """
    torch: Any = pytest.importorskip("torch")
    cases = {
        "scalar": (torch.tensor(3.5), VERDICT_EXACT),
        "vector": (torch.arange(8, dtype=torch.float32), VERDICT_EXACT),
        "empty": (torch.zeros(0, 4), "NO_ELEMENTS"),
        "matrix": (torch.ones(6, 5, dtype=torch.float32), VERDICT_EXACT),
    }
    for name, (t, expected) in cases.items():
        cmp = compare_keys(MemSource({"w": t}), MemSource({"w": t.clone()}), "w")
        assert cmp.verdict == expected, f"{name}: guard misfired -> {cmp.to_dict()}"
