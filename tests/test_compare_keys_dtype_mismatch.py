"""Cross-dtype comparisons must surface DTYPE_MISMATCH, never verify as EXACT.

Why this file exists
--------------------
``compare_keys`` fetched ``dtype_a`` and ``dtype_b``, recorded both on the
returned record, and never compared them. Identity was then decided by
``torch.equal``'s type promotion over *decoded values*: a bf16 export and its
f32 source whose values decoded equal (any lossless round-trip; every small
integral constant in scales and router buffers) returned ``verdict="EXACT"``
with ``bitwise_equal=True`` over artifacts holding different-width bits. The
requantized twin case minted an honest-looking ``DIFFER`` whose near-close
statistics buried the operative fact that the encoding had changed, and an
empty cross-encoding pair abstained ``NO_ELEMENTS`` over a difference the
function was holding in its hands. Three faces, one root: the declared
encodings were never adjudicated.

The fix resolves the disagreement from declared metadata, before any
streaming, as a positively observed finding in a new verdict token — not an
abstention (the difference WAS observed), not SHAPE_MISMATCH (the geometry
agrees), not DIFFER (no content was streamed), not a raise (valid writers
produce it). The tests below pin the verdict, the "compared nothing" field
expression, the precedence over both neighbouring branches, and — as the
MUST_PASS half of the detector — that same-dtype comparisons are untouched.
The in-memory source is the same full ``WeightSource`` double used by the
comparator's other tests: the real code path, not a paraphrase.
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
    VERDICT_NO_ELEMENTS,
    VERDICT_NON_FINITE,
    VERDICT_SHAPE_MISMATCH,
    compare_keys,
)

# ---------------------------------------------------------------------------
# MUST_FIRE: value-level equality can never launder an encoding change into
# a pass grade, in either argument position, across the realistic dtype pairs
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "dtype_a,dtype_b,values",
    [
        # The audited direction: lossy-width export of a full-width source.
        # 0.5/-0.25 are exactly representable in bf16, so the decoded values
        # agree exactly -- the worst case for the laundering, not the best.
        ("bfloat16", "float32", [0.0, 1.0, -2.0, 0.5, 4.0, -0.25]),
        # Mirrored, so neither argument position is special.
        ("float32", "bfloat16", [0.0, 1.0, -2.0, 0.5, 4.0, -0.25]),
        # IEEE against IEEE, narrower against wider.
        ("float16", "float32", [0.5, -0.25, 1.5, 2.5, -3.5, 4.0]),
        # Integer against IEEE: 1 in int8 and 1.0 in f32 promote and compare
        # equal -- torch.equal cannot see that the on-disk words differ.
        ("int8", "float32", [0.0, 1.0, -2.0, 3.0, 4.0, -5.0]),
        # Same width, different meaning: eight bytes read two ways.
        ("float64", "int64", [0.0, 1.0, -2.0, 3.0, 4.0, -5.0]),
    ],
)
def test_value_identity_cannot_launder_an_encoding_change(
    dtype_a: str, dtype_b: str, values: list[float]
) -> None:
    """MUST_FIRE: the dtype finding owns the verdict regardless of how equal the
    values are. Every element in ``values`` is exactly representable in both
    encodings, so the pre-fix promotion predicate genuinely held and the pair
    returned EXACT -- the laundered pass, reproduced and condemned. The
    denominators (elements/chunks/bytes all 0) prove nothing was streamed."""
    torch: Any = pytest.importorskip("torch")
    ta = torch.tensor(values, dtype=getattr(torch, dtype_a)).reshape(2, 3)
    tb = torch.tensor(values, dtype=getattr(torch, dtype_b)).reshape(2, 3)
    assert bool(torch.equal(ta, tb))  # the pre-fix laundering predicate really holds

    cmp = compare_keys(MemSource({"w": ta}), MemSource({"w": tb}), "w")

    assert cmp.verdict == "DTYPE_MISMATCH", f"laundered pass: {cmp.to_dict()}"
    assert cmp.verdict not in {
        VERDICT_EXACT,
        VERDICT_CLOSE,
        VERDICT_DIFFER,
        VERDICT_NON_FINITE,
        VERDICT_SHAPE_MISMATCH,
        VERDICT_NO_ELEMENTS,
    }  # a pre-fix consumer must meet a token it cannot map to any grade
    assert cmp.elements == 0  # the finding names how much content was compared: none
    assert cmp.chunks_read == 0
    assert cmp.bytes_read == 0
    assert cmp.bitwise_equal is False  # enforced by routing, never a clamped True
    assert cmp.mismatched_elements == 0, (
        "a mismatch count over unstreamed content is the mirrored lie: divergence nobody observed"
    )
    assert math.isinf(cmp.max_abs_diff)  # the established 'no bound measured' expression
    assert math.isinf(cmp.mean_abs_diff)
    assert cmp.cosine is None
    assert cmp.nonfinite_elements == 0
    assert cmp.dtype_a == f"torch.{dtype_a}"  # these two fields ARE the finding
    assert cmp.dtype_b == f"torch.{dtype_b}"
    assert cmp.shape_a == cmp.shape_b == (2, 3)  # geometry agreed: SHAPE_MISMATCH would lie
    payload = json.dumps(cmp.to_dict(), allow_nan=False)
    assert '"inf"' in payload  # strict consumers survive the report of the finding


def test_wild_value_divergence_is_not_minted_into_observed_statistics() -> None:
    """MUST_FIRE, mirrored: the fix must not trade a false pass for a false
    observation. The decoded values differ enormously, but nothing was
    streamed, so the record must carry no observed-divergence statistics:
    verdict stays DTYPE_MISMATCH (never DIFFER), mismatched keeps its
    0-of-the-named-0 empty shape, and max_abs_diff keeps the unbounded
    sentinel rather than a measured number that pretends a comparison ran.
    Pre-fix this pair returned DIFFER with mismatched_elements=6 and a
    measured max_abs_diff -- value divergence standing in front of the
    encoding change, which is the same defect wearing its other hat."""
    torch: Any = pytest.importorskip("torch")
    ta = torch.tensor([[1000.0, -1000.0]], dtype=torch.bfloat16)
    tb = torch.tensor([[1e-4, -1e-4]], dtype=torch.float32)

    cmp = compare_keys(MemSource({"w": ta}), MemSource({"w": tb}), "w")

    assert cmp.verdict == "DTYPE_MISMATCH", cmp.to_dict()
    assert cmp.verdict != VERDICT_DIFFER
    assert cmp.mismatched_elements == 0
    assert math.isinf(cmp.max_abs_diff)
    assert cmp.elements == 0 and cmp.chunks_read == 0 and cmp.bytes_read == 0
    json.dumps(cmp.to_dict(), allow_nan=False)


# ---------------------------------------------------------------------------
# MUST_FIRE: precedence -- the encoding finding outranks both neighbours,
# matching parity's adjudication order (dtype, then shape, then zero-element)
# ---------------------------------------------------------------------------


def test_dtype_finding_precedes_the_shape_branch() -> None:
    """When encodings AND geometry disagree, the encoding finding is named first.

    SHAPE_MISMATCH here was TRUE about the geometry -- the pin is precedence,
    not truth: parity has always adjudicated dtype before shape, and two
    layers of one framework must never disagree on which finding owns a key.
    Pre-fix this pair returned SHAPE_MISMATCH and left the encoding change to
    whichever layer happened to look. The geometry facts stay on the record."""
    torch: Any = pytest.importorskip("torch")
    ta = torch.zeros(2, 3, dtype=torch.bfloat16)
    tb = torch.zeros(6, dtype=torch.float32)

    cmp = compare_keys(MemSource({"w": ta}), MemSource({"w": tb}), "w")

    assert cmp.verdict == "DTYPE_MISMATCH", cmp.to_dict()
    assert cmp.verdict != VERDICT_SHAPE_MISMATCH
    assert cmp.shape_a == (2, 3) and cmp.shape_b == (6,)  # the geometry facts still recorded
    assert cmp.elements == 0
    json.dumps(cmp.to_dict(), allow_nan=False)


def test_dtype_finding_precedes_the_zero_element_abstention() -> None:
    """Empty tensors disagreeing in encoding are a finding, not an abstention.

    NO_ELEMENTS means 'nothing was examined, so nothing can be claimed'. Here
    the encodings WERE examined -- compare_keys fetches them before any branch
    -- and they differ; abstaining un-states an observed difference (doctrine
    5's mirrored lie). Pre-fix this pair abstained NO_ELEMENTS. This is also
    the cross-check on the zero-elements fix: its abstention is preserved for
    genuinely empty SAME-encoding pairs by the sweep just below."""
    torch: Any = pytest.importorskip("torch")
    ta = torch.zeros(0, 8, dtype=torch.bfloat16)
    tb = torch.zeros(0, 8, dtype=torch.float32)

    cmp = compare_keys(MemSource({"w": ta}), MemSource({"w": tb}), "w")

    assert cmp.verdict == "DTYPE_MISMATCH", cmp.to_dict()
    assert cmp.verdict != VERDICT_NO_ELEMENTS
    assert cmp.elements == 0 and cmp.chunks_read == 0
    json.dumps(cmp.to_dict(), allow_nan=False)


# ---------------------------------------------------------------------------
# MUST_PASS: same-dtype comparisons are byte-for-byte unchanged in behaviour.
# The new branch is unreachable when encodings agree, and these controls are
# what prove it -- if the predicate ever misfires on an agreed encoding,
# every gate in the estate starts blocking healthy checkpoints, and this
# sweep (over every scalar family both real readers can emit) turns red first.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "dtype",
    ["float16", "bfloat16", "float32", "float64", "int8", "int32", "int64"],
)
def test_same_dtype_comparisons_are_unchanged(dtype: str) -> None:
    """MUST_PASS: the detector's other failure direction -- a misfiring guard.

    Asserts the full pre-fix record shape for an identical-content pair:
    EXACT, the named denominators (elements/chunks/bytes all real), bitwise
    identity, and the self-checked cosine at the comparator's own 1e-9 slack.
    EXACT cannot be returned through the mismatch branch -- its constructor
    sets bitwise_equal=False and inf diffs -- so any misfire flips this red."""
    torch: Any = pytest.importorskip("torch")
    # The export this control defends: it must exist, and it must never
    # appear on an agreed-encoding comparison.
    assert getattr(dcp, "VERDICT_DTYPE_MISMATCH", None) == "DTYPE_MISMATCH"
    t = (torch.arange(24, dtype=torch.float64).reshape(4, 6) - 11.5).to(getattr(torch, dtype))
    cmp = compare_keys(MemSource({"w": t}), MemSource({"w": t.clone()}), "w")

    assert cmp.verdict == VERDICT_EXACT, f"guard misfired -> {cmp.to_dict()}"
    assert cmp.elements == 24
    assert cmp.chunks_read == 2  # one block per source: the stream really ran
    assert cmp.bytes_read == 48 * t.element_size()
    assert cmp.bitwise_equal is True
    assert cmp.mismatched_elements == 0
    assert cmp.max_abs_diff == 0.0
    assert cmp.cosine is not None and abs(cmp.cosine - 1.0) <= 1e-9  # the self-check's own slack
    assert cmp.nonfinite_elements == 0
    json.dumps(cmp.to_dict(), allow_nan=False)


def test_same_dtype_close_and_differ_paths_are_unchanged() -> None:
    """MUST_PASS for the tolerance machinery: CLOSE and DIFFER live below the
    new branch and must keep their pre-fix adjudications. Deterministic seed,
    unit-scale content: a +1e-3 uniform shift sits inside both default
    tolerances (CLOSE), a sign-flipped doubling violates all of them (DIFFER),
    and every element genuinely moves in both cases (ulp at these magnitudes
    is orders below both perturbations), so the mismatch counts are exact."""
    torch: Any = pytest.importorskip("torch")
    assert getattr(dcp, "VERDICT_DTYPE_MISMATCH", None) == "DTYPE_MISMATCH"
    gen = torch.Generator().manual_seed(97)
    base = torch.randn(8, 8, generator=gen)
    near = base + torch.full_like(base, 1e-3)
    far = base * -2.0

    close = compare_keys(MemSource({"w": base}), MemSource({"w": near}), "w")
    assert close.verdict == VERDICT_CLOSE, close.to_dict()
    assert close.elements == 64 and close.chunks_read == 2
    assert close.mismatched_elements == 64
    assert 0.0 < close.max_abs_diff <= 1e-2
    assert close.cosine is not None and close.cosine > 0.999

    differ = compare_keys(MemSource({"w": base}), MemSource({"w": far}), "w")
    assert differ.verdict == VERDICT_DIFFER, differ.to_dict()
    assert differ.mismatched_elements == 64
    json.dumps(close.to_dict(), allow_nan=False)
    json.dumps(differ.to_dict(), allow_nan=False)
