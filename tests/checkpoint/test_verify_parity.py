"""Adversarial tests for ``foundationscale.verify.parity``.

Why this suite exists
---------------------
This module is the checkpoint-side answer to the two audit incidents that define the
project: the 128-experts-aliased-to-16 checkpoint that passed every check, and the
verification tool that answered ``all_identity: true`` on it because it compared
zero keys and ``all([])`` is ``True``. Add the forensic probe that printed a tensor's
cosine *with itself* as 1.80 and shipped the number, and the failure shape is fixed:
**a comparison that never touched the data, or touched it with broken arithmetic,
reported success.**

These tests therefore attack parity the way the estate actually failed:

* they try to get PASS out of zero common keys and out of zero *elements*;
* they try to smuggle NaN and impossibly-large cosines through the tolerance checks;
* they try to get a numeric comparison (and a silent cast) out of a dtype mismatch;
* they verify every acquittal names the tolerance policy that granted it.

Every rejection test sits next to a positive control that proves the detector could
have fired. The heavy numerics are delegated to ``dcp.compare_keys``, which is
replaced here by a scripted double that records its calls — a metadata-level bug
must show up as *a call that never happened*, not merely as a suspicious report.
"""

from __future__ import annotations

import dataclasses
import json
import math
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

import foundationscale.checkpoint.dcp as dcp_module
import foundationscale.verify.parity as parity_module
from foundationscale.checkpoint.dcp import TensorComparison
from foundationscale.gates.core import (
    ControlKind,
    GateRegistry,
    Lifecycle,
    Verdict,
    verify_controls,
)
from foundationscale.verify.parity import (
    DEFAULT_TOLERANCE_POLICY,
    STRICT_TOLERANCE_POLICY,
    ParityGateContext,
    ParityInvariantError,
    ParityStatus,
    TolerancePolicy,
    ToleranceRule,
    Tolerances,
    WeightParityGate,
    compare_sources,
)

# ---------------------------------------------------------------------------
# Doubles.  ``_FakeSource`` satisfies the WeightSource surface parity actually
# touches (tensor_keys/shape/dtype/path/close); ``_CompareScript`` replaces
# ``dcp.compare_keys`` with per-key scripted comparisons and records every call,
# so "the comparison ran" is a fact, not a hope.
# ---------------------------------------------------------------------------

_Metadata = tuple[str, tuple[int, ...]]  # (dtype name, shape)


class _FakeSource:
    def __init__(self, path: str, spec: Mapping[str, _Metadata]) -> None:
        self.path = path
        self._spec = dict(spec)
        self.closed = False

    def tensor_keys(self) -> tuple[str, ...]:
        return tuple(sorted(self._spec))

    def shape(self, key: str) -> tuple[int, ...]:
        return self._spec[key][1]

    def dtype(self, key: str) -> str:
        return self._spec[key][0]

    def close(self) -> None:
        self.closed = True


def _comparison(
    key: str,
    *,
    elements: int = 16,
    mismatched: int = 0,
    bitwise: bool = True,
    max_abs_diff: float = 0.0,
    cosine: float | None = 1.0,
    rms_a: float = 1.0,
    rms_b: float = 1.0,
    shape: tuple[int, ...] = (4, 4),
) -> TensorComparison:
    """Defaults describe two bitwise-identical, non-degenerate tensors."""
    return TensorComparison(
        key=key,
        elements=elements,
        shape_a=shape,
        shape_b=shape,
        dtype_a="torch.float32",
        dtype_b="torch.float32",
        bitwise_equal=bitwise,
        mismatched_elements=mismatched,
        max_abs_diff=max_abs_diff,
        mean_abs_diff=0.0,
        cosine=cosine,
        rms_a=rms_a,
        rms_b=rms_b,
        chunks_read=2,
        bytes_read=128,
        verdict="SCRIPTED",
    )


class _CompareScript:
    """Drop-in for ``dcp.compare_keys``: returns scripted results, records calls."""

    def __init__(self, table: Mapping[str, TensorComparison | BaseException]) -> None:
        self._table = dict(table)
        self.calls: list[tuple[str, int, float, float]] = []

    def __call__(
        self,
        _source_a: Any,
        _source_b: Any,
        key: str,
        *,
        block_rows: int,
        close_max_abs_diff: float,
        close_min_cosine: float,
    ) -> TensorComparison:
        self.calls.append((key, block_rows, close_max_abs_diff, close_min_cosine))
        outcome = self._table.get(key)
        if outcome is None:
            raise AssertionError(f"compare_keys called for unscripted key {key!r}")
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


def _script_compare_keys(
    monkeypatch: pytest.MonkeyPatch,
    table: Mapping[str, TensorComparison | BaseException],
) -> _CompareScript:
    # parity imports ``compare_keys`` lazily inside the comparison, so this patch
    # lands on the name it actually resolves at call time.
    script = _CompareScript(table)
    monkeypatch.setattr(dcp_module, "compare_keys", script)
    return script


def _gate_ctx(
    left: _FakeSource,
    right: _FakeSource,
    policy: TolerancePolicy = DEFAULT_TOLERANCE_POLICY,
    label: str = "test",
) -> ParityGateContext:
    # The annotated field type is ``str | Path``, but ``compare_sources`` accepts a
    # live WeightSource; the doubles exercise exactly that runtime path.
    return ParityGateContext(left_path=left, right_path=right, policy=policy, label=label)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Tolerances: policy is explicit data, so invalid data must be unconstructible.
# ---------------------------------------------------------------------------


class TestTolerancesValidation:
    """Anonymous or nonsensical tolerance criteria are how magic numbers re-enter."""

    def test_well_formed_tolerances_construct_describe_and_serialize(self) -> None:
        # Positive control for every rejection in this class: valid data must work.
        tol = Tolerances("unit", max_abs_diff=0.001, min_cosine=0.99, max_rel_frob=0.01)
        assert tol.describe() == "unit(max_abs_diff<=0.001, cosine>=0.99, rel_frob<=0.01)"
        assert tol.to_dict() == {
            "name": "unit",
            "max_abs_diff": 0.001,
            "min_cosine": 0.99,
            "max_rel_frob": 0.01,
        }

    def test_blank_name_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="anonymous policy"):
            Tolerances("   ", max_abs_diff=0.0, min_cosine=1.0, max_rel_frob=0.0)

    def test_negative_max_abs_diff_is_rejected_but_zero_is_strict_and_legal(self) -> None:
        with pytest.raises(ValueError, match="max_abs_diff cannot be negative"):
            Tolerances("bad", max_abs_diff=-1e-9, min_cosine=1.0, max_rel_frob=0.0)
        strict = Tolerances("fine", max_abs_diff=0.0, min_cosine=1.0, max_rel_frob=0.0)
        assert strict.max_abs_diff == 0.0

    def test_min_cosine_outside_unit_interval_is_rejected_but_bounds_are_legal(self) -> None:
        for bad in (1.000_001, -1.5):
            with pytest.raises(ValueError, match="min_cosine must lie in"):
                Tolerances("bad", max_abs_diff=0.0, min_cosine=bad, max_rel_frob=0.0)
        assert Tolerances("top", 0.0, 1.0, 0.0).min_cosine == 1.0
        assert Tolerances("bottom", 0.0, -1.0, 0.0).min_cosine == -1.0

    def test_nan_min_cosine_is_rejected(self) -> None:
        # NaN is not in any interval; it must not slip through as "no constraint".
        with pytest.raises(ValueError, match="min_cosine must lie in"):
            Tolerances("bad", max_abs_diff=0.0, min_cosine=math.nan, max_rel_frob=0.0)

    def test_negative_rel_frob_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="max_rel_frob cannot be negative"):
            Tolerances("bad", max_abs_diff=0.0, min_cosine=1.0, max_rel_frob=-0.1)

    def test_tolerances_are_immutable(self) -> None:
        tol = Tolerances("unit", 0.1, 0.9, 0.1)
        with pytest.raises(dataclasses.FrozenInstanceError):
            tol.max_abs_diff = 900.0  # type: ignore[misc]

    def test_shipped_policies_describe_honestly(self) -> None:
        assert DEFAULT_TOLERANCE_POLICY.describe() == (
            "default-close(max_abs_diff<=0.01, cosine>=0.999, rel_frob<=0.01)"
        )
        assert STRICT_TOLERANCE_POLICY.describe() == (
            "strict-bitexact(max_abs_diff<=0, cosine>=1, rel_frob<=0)"
        )


class TestTolerancePolicy:
    """Rule matching order is adjudication order; it must be observable and stable."""

    def test_malformed_regex_fails_at_construction_not_mid_comparison(self) -> None:
        tol = Tolerances("unit", 0.1, 0.9, 0.1)
        with pytest.raises(re.error):
            ToleranceRule(r"layers[", tol)
        # Positive control: a valid pattern constructs and matches with re.search.
        rule = ToleranceRule(r"layers\.\d+\.experts", tol)
        assert rule.matches("model.layers.12.experts.w")
        assert not rule.matches("model.embed")

    def test_first_matching_rule_wins(self) -> None:
        first = Tolerances("first", 1.0, 0.5, 1.0)
        second = Tolerances("second", 2.0, 0.4, 2.0)
        policy = TolerancePolicy(
            default=Tolerances("base", 0.0, 1.0, 0.0),
            rules=(ToleranceRule(r"layers", first), ToleranceRule(r"layers\.0", second)),
        )
        # Both rules match; ordering, not specificity, decides — recorded as identity.
        assert policy.tolerances_for("model.layers.0.w") is first
        assert policy.tolerances_for("model.embed") is policy.default

    def test_describe_records_rules(self) -> None:
        loose = Tolerances("loose", 1.0, 0.9, 1.0)
        policy = TolerancePolicy(rules=(ToleranceRule(r"experts", loose),))
        rendered = policy.describe()
        assert "default-close" in rendered
        assert "->" in rendered
        assert "loose" in rendered


class TestParityStatus:
    def test_blocking_is_exactly_the_finding_statuses(self) -> None:
        # If an acquittal status ever becomes blocking (or vice versa), callers
        # triage wrong: "differs" and "not even the same dtype" drive different work.
        blocking = {status for status in ParityStatus if status.blocking}
        assert blocking == {
            ParityStatus.DIFFER,
            ParityStatus.DTYPE_MISMATCH,
            ParityStatus.SHAPE_MISMATCH,
        }


# ---------------------------------------------------------------------------
# compare_sources: routing, adjudication, skip bookkeeping.
# ---------------------------------------------------------------------------


class TestCompareSourcesHappyPaths:
    def test_bitwise_identical_keys_are_exact_and_the_report_acquits(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        script = _script_compare_keys(
            monkeypatch,
            {
                "a.weight": _comparison("a.weight"),
                "b.weight": _comparison("b.weight"),
            },
        )
        left = _FakeSource("left", {"a.weight": ("torch.float32", (4, 4)), "b.weight": ("x", (2,))})
        right = _FakeSource(
            "right", {"a.weight": ("torch.float32", (4, 4)), "b.weight": ("x", (2,))}
        )
        report = compare_sources(left, right)
        assert script.calls[0][0] == "a.weight"  # common keys ordered by name
        assert [entry.key for entry in report.keys] == ["a.weight", "b.weight"]
        assert all(entry.status is ParityStatus.EXACT for entry in report.keys)
        assert report.ok
        assert not report.is_vacuous
        assert report.findings == ()
        entry = report.keys[0]
        assert entry.rel_frob == 0.0
        assert entry.cosine == 1.0
        assert "16 elements bitwise identical" in entry.note
        assert report.render().startswith("weight parity [ok]: 2 common tensor keys")

    def test_close_but_not_bitwise_is_close_and_carries_its_tolerance(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        close_cmp = _comparison(
            "w",
            mismatched=2,
            bitwise=False,
            max_abs_diff=0.005,
            cosine=0.999999,
        )
        _script_compare_keys(monkeypatch, {"w": close_cmp})
        source = _FakeSource("s", {"w": ("torch.float32", (4, 4))})
        report = compare_sources(source, _FakeSource("r", {"w": ("torch.float32", (4, 4))}))
        entry = report.keys[0]
        assert entry.status is ParityStatus.CLOSE
        assert entry.mismatched_elements == 2
        # rel_frob is DERIVED from the streamed statistics, not taken on trust:
        # sqrt(1 + 1 - 2*0.999999) / 1 == sqrt(2e-6).
        assert math.isclose(entry.rel_frob, math.sqrt(2e-6), rel_tol=1e-9)
        # The acquittal names its own criteria, and the default policy object itself.
        assert entry.tolerances is DEFAULT_TOLERANCE_POLICY.default
        assert "default-close" in entry.note
        assert "within" in entry.note
        assert report.ok

    def test_per_key_rules_route_and_the_granting_tolerance_is_recorded(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        loose = Tolerances("loose-experts", max_abs_diff=1.0, min_cosine=0.9, max_rel_frob=1.0)
        tight = Tolerances(
            "tight-rest", max_abs_diff=0.0001, min_cosine=0.9999, max_rel_frob=0.0001
        )
        policy = TolerancePolicy(default=tight, rules=(ToleranceRule(r"experts", loose),))
        deviation = dict(mismatched=8, bitwise=False, max_abs_diff=0.5, cosine=0.95)
        _script_compare_keys(
            monkeypatch,
            {
                "m.experts.w": _comparison("m.experts.w", **deviation),
                "m.attention.w": _comparison("m.attention.w", **deviation),
            },
        )
        spec = {
            "m.experts.w": ("torch.float32", (4, 4)),
            "m.attention.w": ("torch.float32", (4, 4)),
        }
        report = compare_sources(_FakeSource("l", spec), _FakeSource("r", spec), policy=policy)
        by_key = {entry.key: entry for entry in report.keys}
        # Identical numbers, different verdicts — the only difference is the policy
        # rule. A report that cannot explain this distinction is laundering.
        assert by_key["m.experts.w"].status is ParityStatus.CLOSE
        assert by_key["m.experts.w"].tolerances is loose
        assert by_key["m.attention.w"].status is ParityStatus.DIFFER
        assert by_key["m.attention.w"].tolerances is tight
        assert not report.ok
        assert "EXCEEDS" in by_key["m.attention.w"].note

    def test_block_rows_and_tolerance_bounds_are_forwarded_unchanged(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        script = _script_compare_keys(monkeypatch, {"w": _comparison("w")})
        spec = {"w": ("torch.float32", (4, 4))}
        report = compare_sources(_FakeSource("l", spec), _FakeSource("r", spec), block_rows=7)
        assert report.block_rows == 7
        assert script.calls == [("w", 7, 0.01, 0.999)]

    def test_cosine_at_the_floor_is_within_tolerance(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # min_cosine is a floor: equality is admissible (>=).  A reduction that only
        # just reaches a declared floor must not flip a pass into a fail.
        floor_policy = TolerancePolicy(
            default=Tolerances("floor", max_abs_diff=1.0, min_cosine=0.9, max_rel_frob=10.0)
        )
        cmp = _comparison("w", mismatched=3, bitwise=False, max_abs_diff=0.1, cosine=0.9)
        _script_compare_keys(monkeypatch, {"w": cmp})
        spec = {"w": ("torch.float32", (4, 4))}
        report = compare_sources(
            _FakeSource("l", spec), _FakeSource("r", spec), policy=floor_policy
        )
        assert report.keys[0].status is ParityStatus.CLOSE


class TestFindingsAreDataNotExceptions:
    """compare_sources must never raise for *bad* data — only for impossible data."""

    def test_differing_key_is_a_blocking_finding_and_the_report_fails(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # rel_frob drives the failure: cosine is at the floor, max_abs_diff is small,
        # but the *aggregate distance* exceeds max_rel_frob. Does not raise.
        cmp = _comparison(
            "expert.3",
            mismatched=4,
            bitwise=False,
            max_abs_diff=0.005,
            cosine=0.9995,
        )
        _script_compare_keys(monkeypatch, {"expert.3": cmp})
        spec = {"expert.3": ("torch.float32", (4, 4))}
        report = compare_sources(_FakeSource("l", spec), _FakeSource("r", spec))
        entry = report.keys[0]
        assert math.isclose(entry.rel_frob, math.sqrt(0.001), rel_tol=1e-9)  # 0.0316 > 0.01
        assert entry.status is ParityStatus.DIFFER
        assert entry.blocking
        assert report.findings == (entry,)
        assert not report.ok
        line = f"{'DIFFER':>14} expert.3"
        assert line in report.render()

    def test_keys_present_on_one_side_only_are_named_and_fail_the_report(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The aliased-expert incident's shape: some keys compared fine, one side is
        # silently missing keys. Every common key may be EXACT and the report must
        # still say the sets do not correspond.
        _script_compare_keys(monkeypatch, {"common": _comparison("common")})
        left = _FakeSource(
            "l", {"common": ("torch.float32", (4, 4)), "zeta_only_left": ("x", (1,))}
        )
        right = _FakeSource(
            "r",
            {
                "common": ("torch.float32", (4, 4)),
                "beta_only_right": ("x", (1,)),
                "alpha_only_right": ("x", (1,)),
            },
        )
        report = compare_sources(left, right)
        assert report.keys[0].status is ParityStatus.EXACT
        assert report.only_in_left == ("zeta_only_left",)
        assert report.only_in_right == ("alpha_only_right", "beta_only_right")  # sorted
        assert not report.ok  # complete key sets is part of "ok", not an afterthought
        payload = report.to_dict()
        assert payload["only_in_left"] == ["zeta_only_left"]
        assert payload["only_in_right"] == ["alpha_only_right", "beta_only_right"]

    def test_reader_errors_propagate_out_of_compare_sources(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _script_compare_keys(monkeypatch, {"w": RuntimeError("shard fell off the disk")})
        spec = {"w": ("torch.float32", (4, 4))}
        with pytest.raises(RuntimeError, match="shard fell off the disk"):
            compare_sources(_FakeSource("l", spec), _FakeSource("r", spec))


class TestSkippedKeysAreNotSilentlyCountedAsMatching:
    """A key not compared numerically is coverage's business, not a quiet success."""

    def test_dtype_mismatch_is_a_metadata_finding_and_no_numeric_compare_runs(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # If this regresses, compare_keys runs against mismatched dtypes (the silent
        # cast) and ``script.calls`` records a call for "embed" — the detector fires.
        script = _script_compare_keys(monkeypatch, {"head": _comparison("head")})
        left = _FakeSource(
            "l",
            {"embed": ("torch.float32", (4, 4)), "head": ("torch.float32", (4, 4))},
        )
        right = _FakeSource(
            "r",
            {"embed": ("torch.float64", (4, 4)), "head": ("torch.float32", (4, 4))},
        )
        report = compare_sources(left, right)
        assert [call[0] for call in script.calls] == ["head"]  # embed never compared
        by_key = {entry.key: entry for entry in report.keys}
        entry = by_key["embed"]
        assert entry.status is ParityStatus.DTYPE_MISMATCH
        assert entry.blocking
        # 0 elements compared, visibly so — never folded into the EXACT count.
        assert entry.elements == 0
        assert entry.cosine is None
        assert math.isinf(entry.max_abs_diff)
        assert entry.dtype_left == "torch.float32"
        assert entry.dtype_right == "torch.float64"
        assert "dtype mismatch torch.float32 vs torch.float64" in entry.note
        # The skip bookkeeping names the key and the reason, once.
        assert len(report.skipped) == 1
        skipped_key, reason = report.skipped[0]
        assert skipped_key == "embed"
        assert "dtype mismatch" in reason
        assert "not numerically compared" in reason
        assert entry.to_dict()["status"] == "dtype_mismatch"
        # Crucially: it is neither "compared" nor an acquittal.
        assert [e.key for e in report.compared] == ["head"]
        payload = report.to_dict()
        assert payload["common_keys"] == 2
        assert payload["compared_keys"] == 1
        assert payload["skipped"] == [[skipped_key, reason]]
        assert not report.ok
        # Assert the fact, not the column width: render() right-aligns the status
        # label, and pinning the padding makes the test fail on a cosmetic change
        # while still passing if the key itself went missing.
        assert re.search(r"SKIPPED\s+embed\b", report.render())

    def test_shape_mismatch_is_distinguishable_from_a_value_difference(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # A caller seeing DIFFER re-runs a converter; a caller seeing SHAPE_MISMATCH
        # has a layout bug. Conflating them sends operators to the wrong fix.
        script = _script_compare_keys(monkeypatch, {})
        left = _FakeSource("l", {"w": ("torch.float32", (4, 4))})
        right = _FakeSource("r", {"w": ("torch.float32", (4, 8))})
        report = compare_sources(left, right)
        assert script.calls == []  # no numeric comparison was even attempted
        entry = report.keys[0]
        assert entry.status is ParityStatus.SHAPE_MISMATCH
        assert ParityStatus.SHAPE_MISMATCH is not ParityStatus.DIFFER
        assert entry.shape_left == (4, 4)
        assert entry.shape_right == (4, 8)
        assert entry.elements == 0
        assert entry.blocking
        assert "shape mismatch (4, 4) vs (4, 8)" in entry.note
        assert report.render().count("SHAPE_MISMATCH") == 1

    def test_dtype_mismatch_takes_priority_over_shape_mismatch(self) -> None:
        # Pinned ordering: when both metadata levels disagree, dtype is named first —
        # a caller must be able to rely on which finding they will see.
        left = _FakeSource("l", {"w": ("torch.float32", (4, 4))})
        right = _FakeSource("r", {"w": ("torch.bfloat16", (8, 8))})
        report = compare_sources(left, right)
        assert report.keys[0].status is ParityStatus.DTYPE_MISMATCH


class TestZeroDataMustNotReadAsAgreement:
    """The all([])-is-True incident, at every scale the module can produce it."""

    def test_zero_common_keys_is_vacuous_not_ok(self, monkeypatch: pytest.MonkeyPatch) -> None:
        script = _script_compare_keys(monkeypatch, {})
        report = compare_sources(
            _FakeSource("candidate", {"only_left_a": ("torch.float32", (4, 4))}),
            _FakeSource("reference", {"only_right_b": ("torch.float32", (4, 4))}),
        )
        assert script.calls == []  # nothing was compared — and the report says so
        assert report.is_vacuous
        assert not report.ok
        assert report.findings == ()
        rendered = report.render()
        assert "VACUOUS" in rendered
        assert "proves nothing" in rendered
        assert "all([])" in rendered
        payload = report.to_dict()
        assert payload["vacuous"] is True
        assert payload["ok"] is False
        assert payload["common_keys"] == 0

    def test_one_common_exact_key_is_a_positive_control_for_vacuity(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Same machinery, one real comparison: proves the vacuous verdict above is
        # about emptiness, not about the fixture.
        _script_compare_keys(monkeypatch, {"w": _comparison("w")})
        report = compare_sources(
            _FakeSource("l", {"w": ("torch.float32", (4, 4)), "left_extra": ("x", (1,))}),
            _FakeSource("r", {"w": ("torch.float32", (4, 4)), "right_extra": ("x", (1,))}),
        )
        assert not report.is_vacuous
        # …but the dangling keys on each side still fail the report: vacuity is not
        # the only path to "not ok".
        assert not report.ok

    def test_zero_element_tensor_is_disclosed_as_such(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Scripted on dcp's POST-fix NO_ELEMENTS contract: nothing read, unbounded
        # difference sentinels, cosine None. Pre-fix parity fed this shape into
        # _guard_stats, which inferred bitwise identity from 0-mismatched-of-0 and
        # raised ParityInvariantError; pre-dcp-fix it came back EXACT over zero
        # elements. Both are the all([]) shape; the correct outcome is a named,
        # skipped ABSTENTION that never enters the EXACT/CLOSE numerators.
        script = _script_compare_keys(
            monkeypatch,
            {
                "empty.w": _comparison(
                    "empty.w",
                    elements=0,
                    bitwise=False,
                    max_abs_diff=math.inf,
                    cosine=None,
                    rms_a=0.0,
                    rms_b=0.0,
                    shape=(0,),
                )
            },
        )
        spec = {"empty.w": ("torch.float32", (0,))}
        report = compare_sources(_FakeSource("l", spec), _FakeSource("r", spec))
        # The comparison never ran: zero elements are knowable from the shape
        # metadata, and streaming nothing is not a comparison.
        assert script.calls == []
        entry = report.keys[0]
        assert entry.status is ParityStatus.NO_ELEMENTS
        assert not entry.blocking  # an abstention, not a finding
        assert entry.elements == 0
        assert entry.cosine is None
        assert report.compared == ()  # excluded from the compared numerator
        assert report.skipped == (("empty.w", entry.note),)
        assert "0 elements" in entry.note
        # A report whose entire evidence base is one empty tensor is VACUOUS.
        assert report.is_vacuous
        assert not report.ok
        rendered = report.render()
        assert "VACUOUS" in rendered
        assert re.search(r"SKIPPED\s+empty\.w\b", rendered)
        assert json.dumps(report.to_dict())  # the disclosure must be serializable

    def test_a_source_reporting_an_empty_key_set_is_an_invariant_violation(self) -> None:
        # The readers refuse to construct empty sources; a bespoke WeightSource that
        # does must hit parity's own guard, not produce a vacuous report over nothing.
        healthy = _FakeSource("r", {"w": ("torch.float32", (4, 4))})
        with pytest.raises(ParityInvariantError, match="empty key set"):
            compare_sources(_FakeSource("empty-left", {}), healthy)
        with pytest.raises(ParityInvariantError, match="empty key set"):
            compare_sources(healthy, _FakeSource("empty-right", {}))
        # Positive control: the same helper with real keys reaches the comparison.
        _CompareScript({"w": _comparison("w")})  # constructs without error


# ---------------------------------------------------------------------------
# Degenerate payloads: NaN, inf, zero-norm.  Findings, never acquittals.
# ---------------------------------------------------------------------------


class TestZeroElementsAbstainAtMetadataLevel:
    """A zero-element key is a stated abstention: never EXACT, never minted DIFFER.

    The dcp NO_ELEMENTS contract crashed parity on the way in (the guard's empty
    numerator read as identity) and would have laundered into DIFFER had it survived
    (sentinel infs read as observed divergence). These tests pin the routing and the
    report arithmetic on scripted doubles encoding the fixed comparator's contract.
    """

    def test_mixed_report_passes_on_real_elements_with_the_empty_key_named(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        script = _script_compare_keys(monkeypatch, {"real.w": _comparison("real.w")})
        spec = {
            "empty.w": ("torch.float32", (0, 8)),
            "real.w": ("torch.float32", (4, 4)),
        }
        report = compare_sources(_FakeSource("l", spec), _FakeSource("r", spec))
        # Only the real key streamed; the empty one was adjudicated from metadata.
        assert script.calls == [("real.w", 4096, 0.01, 0.999)]
        by_key = {entry.key: entry for entry in report.keys}
        assert by_key["empty.w"].status is ParityStatus.NO_ELEMENTS
        assert by_key["real.w"].status is ParityStatus.EXACT
        assert ParityStatus.NO_ELEMENTS not in {s for s in ParityStatus if s.blocking}
        # The pass rests on the 16 compared elements; the empty key is named as
        # skipped, not folded into the EXACT numerator (which counts elements > 0).
        assert [entry.key for entry in report.compared] == ["real.w"]
        assert report.compared_elements == 16
        assert not report.is_vacuous
        assert report.ok
        assert [skipped_key for skipped_key, _ in report.skipped] == ["empty.w"]

    def test_a_report_of_only_empty_keys_is_vacuous_not_clean(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The case that matters most: every common tensor is empty, including the
        # (5, 0) shape whose nonzero row count would otherwise have burned real
        # reads over zero-element blocks. The report must block as VACUOUS, naming
        # 0 elements — never acquit, and never invent divergence findings.
        script = _script_compare_keys(monkeypatch, {})
        spec = {
            "pad.a": ("torch.float32", (0, 8)),
            "pad.b": ("torch.float32", (5, 0)),
        }
        report = compare_sources(_FakeSource("l", spec), _FakeSource("r", spec))
        assert script.calls == []
        assert {entry.status for entry in report.keys} == {ParityStatus.NO_ELEMENTS}
        assert report.findings == ()  # nothing observed, so nothing convicts
        assert report.compared_elements == 0
        assert report.is_vacuous
        assert not report.ok
        rendered = report.render()
        assert "VACUOUS" in rendered
        assert "0 elements compared" in rendered
        assert rendered.count("SKIPPED") == 2  # both empty keys are named


class TestDegeneratePayloads:
    def test_nan_statistics_from_a_real_mismatch_are_a_finding_not_an_acquittal(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # A NaN payload in the weights makes NaN diffs and a NaN cosine. Every
        # tolerance comparison involving NaN is False, so nothing can be CLOSE.
        cmp = _comparison(
            "w",
            mismatched=5,
            bitwise=False,
            max_abs_diff=math.nan,
            cosine=math.nan,
        )
        _script_compare_keys(monkeypatch, {"w": cmp})
        spec = {"w": ("torch.float32", (4, 4))}
        report = compare_sources(_FakeSource("l", spec), _FakeSource("r", spec))
        entry = report.keys[0]
        assert entry.status is ParityStatus.DIFFER  # not CLOSE, never EXACT
        assert math.isnan(entry.cosine)
        assert math.isnan(entry.rel_frob)
        assert "NaN/inf payload" in entry.note
        assert not report.ok

    def test_zero_reference_with_nonzero_candidate_yields_infinite_rel_frob(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # ||a - 0|| / ||0|| is infinite, not "small enough"; rounding it down would
        # be the friendliest possible lie.
        cmp = _comparison(
            "w",
            mismatched=4,
            bitwise=False,
            max_abs_diff=0.5,
            cosine=None,
            rms_a=1.0,
            rms_b=0.0,
        )
        _script_compare_keys(monkeypatch, {"w": cmp})
        spec = {"w": ("torch.float32", (4, 4))}
        report = compare_sources(_FakeSource("l", spec), _FakeSource("r", spec))
        entry = report.keys[0]
        assert entry.cosine is None  # a zero tensor has no direction
        assert math.isinf(entry.rel_frob)
        assert entry.status is ParityStatus.DIFFER
        assert "cosine undefined" in entry.note

    def test_two_all_zero_tensor_sides_are_exact_with_no_cosine(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Zero == zero is honestly EXACT, but inventing cosine=1.0 for it would be
        # inventing agreement between directions that do not exist.
        cmp = _comparison("w", cosine=None, rms_a=0.0, rms_b=0.0)
        _script_compare_keys(monkeypatch, {"w": cmp})
        spec = {"w": ("torch.float32", (4, 4))}
        report = compare_sources(_FakeSource("l", spec), _FakeSource("r", spec))
        entry = report.keys[0]
        assert entry.status is ParityStatus.EXACT
        assert entry.cosine is None
        assert entry.rel_frob == 0.0
        assert "cosine undefined" in entry.note
        assert report.ok


# ---------------------------------------------------------------------------
# The impossible-statistics guard: the 1.80 self-cosine must raise, not report.
# ---------------------------------------------------------------------------


class TestImpossibleStatisticsGuard:
    def test_cosine_above_one_is_a_broken_reduction_and_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The incident that named the guard. mismatched is nonzero on purpose: a
        # lying accumulator must not be laundered by being attached to real
        # differences — the lie itself is the detection.
        cmp = _comparison("w", mismatched=5, bitwise=False, max_abs_diff=0.1, cosine=1.80)
        _script_compare_keys(monkeypatch, {"w": cmp})
        spec = {"w": ("torch.float32", (4, 4))}
        with pytest.raises(ParityInvariantError, match="impossible cosine") as excinfo:
            compare_sources(_FakeSource("l", spec), _FakeSource("r", spec))
        assert "1.8" in str(excinfo.value)
        assert "'w'" in str(excinfo.value)

    def test_cosine_exactly_one_passes_the_guard(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Positive control: the guard is not "all cosines are suspicious".
        _script_compare_keys(monkeypatch, {"w": _comparison("w", cosine=1.0)})
        spec = {"w": ("torch.float32", (4, 4))}
        report = compare_sources(_FakeSource("l", spec), _FakeSource("r", spec))
        assert report.ok

    def test_cosine_within_float_slack_of_one_is_admissible(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # 1 + 1e-10 is float64 rounding at a legitimately attained bound, not a lie;
        # the guard's slack exists so honest accumulators are never red.
        cmp = _comparison("w", cosine=1.0 + 1e-10)
        _script_compare_keys(monkeypatch, {"w": cmp})
        spec = {"w": ("torch.float32", (4, 4))}
        report = compare_sources(_FakeSource("l", spec), _FakeSource("r", spec))
        assert report.keys[0].status is ParityStatus.EXACT

    def test_cosine_below_minus_one_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        cmp = _comparison("w", mismatched=5, bitwise=False, max_abs_diff=0.1, cosine=-1.5)
        _script_compare_keys(monkeypatch, {"w": cmp})
        spec = {"w": ("torch.float32", (4, 4))}
        with pytest.raises(ParityInvariantError, match="impossible cosine"):
            compare_sources(_FakeSource("l", spec), _FakeSource("r", spec))

    def test_bitwise_identical_tensors_with_nonzero_diff_stats_raise(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # mismatched_elements == 0 pins every statistic: a reduction that reports
        # differences between identical bit patterns is broken, and must not report.
        cmp = _comparison("w", mismatched=0, bitwise=True, max_abs_diff=0.25)
        _script_compare_keys(monkeypatch, {"w": cmp})
        spec = {"w": ("torch.float32", (4, 4))}
        with pytest.raises(ParityInvariantError, match="bitwise-identical tensors produced"):
            compare_sources(_FakeSource("l", spec), _FakeSource("r", spec))

    def test_same_numbers_with_real_mismatches_are_a_finding_not_a_crash(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Positive control for the test above: identical statistics attached to real
        # mismatches are possible data, so they adjudicate (DIFFER), not raise.
        cmp = _comparison("w", mismatched=3, bitwise=False, max_abs_diff=0.25)
        _script_compare_keys(monkeypatch, {"w": cmp})
        spec = {"w": ("torch.float32", (4, 4))}
        report = compare_sources(_FakeSource("l", spec), _FakeSource("r", spec))
        assert report.keys[0].status is ParityStatus.DIFFER

    def test_bitwise_identical_self_cosine_other_than_one_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # A tensor's cosine with itself is 1.0; 0.75 here is the reduction lying in
        # range — the guard catches what a bounded-cosine check would wave through.
        cmp = _comparison("w", cosine=0.75, rms_a=0.0, rms_b=0.0)
        _script_compare_keys(monkeypatch, {"w": cmp})
        spec = {"w": ("torch.float32", (4, 4))}
        with pytest.raises(ParityInvariantError, match="cosine with itself is 1.0"):
            compare_sources(_FakeSource("l", spec), _FakeSource("r", spec))

    def test_impossible_element_counts_raise(self, monkeypatch: pytest.MonkeyPatch) -> None:
        spec = {"w": ("torch.float32", (4, 4))}
        _script_compare_keys(monkeypatch, {"w": _comparison("w", mismatched=17)})
        with pytest.raises(ParityInvariantError, match="impossible element counts"):
            compare_sources(_FakeSource("l", spec), _FakeSource("r", spec))

    def test_negative_max_abs_diff_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        cmp = _comparison("w", mismatched=2, bitwise=False, max_abs_diff=-0.5, cosine=0.99)
        _script_compare_keys(monkeypatch, {"w": cmp})
        spec = {"w": ("torch.float32", (4, 4))}
        with pytest.raises(ParityInvariantError, match="negative max_abs_diff"):
            compare_sources(_FakeSource("l", spec), _FakeSource("r", spec))

    def test_invariant_error_is_an_arithmetic_error(self) -> None:
        assert issubclass(ParityInvariantError, ArithmeticError)

    # ---------------------------------------------------------------------------
    # Source lifecycle: paths are opened via the sniffer and closed; caller-owned
    # sources are never closed.
    # ---------------------------------------------------------------------------

    def test_zero_elements_compared_pins_nothing_but_impossible_stats_still_raise(self) -> None:
        # The repaired inference: 0 mismatched of 0 compared is an empty numerator,
        # not bitwise identity. The guard must not condemn the "compared nothing"
        # sentinels of dcp's NO_ELEMENTS contract — on the pre-fix tree this exact
        # call raised on max_abs_diff=inf and broke the dcp/parity seam.
        assert (
            parity_module._guard_stats(
                key="w",
                elements=0,
                mismatched=0,
                max_abs_diff=math.inf,
                rel_frob=math.inf,
                cosine=None,
            )
            is None
        )
        # Positive control: the identity pins still bite when identity was genuinely
        # examined — abstention at 0 elements did not amputate the guard.
        with pytest.raises(ParityInvariantError, match="bitwise-identical tensors produced"):
            parity_module._guard_stats(
                key="w",
                elements=16,
                mismatched=0,
                max_abs_diff=0.25,
                rel_frob=0.0,
                cosine=1.0,
            )
        # And abstention is not amnesty: an out-of-range cosine is impossible in any
        # data, at any element count, and still raises.
        with pytest.raises(ParityInvariantError, match="impossible cosine"):
            parity_module._guard_stats(
                key="w",
                elements=0,
                mismatched=0,
                max_abs_diff=math.inf,
                rel_frob=math.inf,
                cosine=1.80,
            )


class TestSourceLifecycle:
    def test_paths_are_opened_and_closed_after_comparison(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        opened: list[_FakeSource] = []

        def fake_open(path: Any) -> _FakeSource:
            src = _FakeSource(str(path), {"w": ("torch.float32", (4, 4))})
            opened.append(src)
            return src

        monkeypatch.setattr(dcp_module, "open_weights", fake_open)
        _script_compare_keys(monkeypatch, {"w": _comparison("w")})
        report = compare_sources(tmp_path / "left", str(tmp_path / "right"))
        assert report.ok
        assert len(opened) == 2
        assert all(src.closed for src in opened)

    def test_opened_sources_are_closed_even_when_the_comparison_raises(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        opened: list[_FakeSource] = []

        def fake_open(path: Any) -> _FakeSource:
            src = _FakeSource(str(path), {"w": ("torch.float32", (4, 4))})
            opened.append(src)
            return src

        monkeypatch.setattr(dcp_module, "open_weights", fake_open)
        _script_compare_keys(monkeypatch, {"w": RuntimeError("boom mid-stream")})
        with pytest.raises(RuntimeError, match="boom mid-stream"):
            compare_sources(tmp_path / "left", tmp_path / "right")
        assert len(opened) == 2
        assert all(src.closed for src in opened)

    def test_caller_owned_sources_are_never_opened_nor_closed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def forbidden_open(_path: Any) -> Any:
            raise AssertionError("open_weights must not be called for live sources")

        monkeypatch.setattr(dcp_module, "open_weights", forbidden_open)
        _script_compare_keys(monkeypatch, {"w": _comparison("w")})
        left = _FakeSource("mine-left", {"w": ("torch.float32", (4, 4))})
        right = _FakeSource("mine-right", {"w": ("torch.float32", (4, 4))})
        report = compare_sources(left, right)
        assert report.ok
        assert not left.closed
        assert not right.closed
        assert report.left_path == "mine-left"
        assert report.right_path == "mine-right"


# ---------------------------------------------------------------------------
# The gate: verdicts, coverage as a first-class fact, evidence.
# ---------------------------------------------------------------------------


class TestWeightParityGate:
    def test_identity_and_lifecycle_registration(self) -> None:
        gate = WeightParityGate()
        assert gate.id == "checkpoint.weight_parity"
        for event in (
            Lifecycle.FIRST_SAVE,
            Lifecycle.SAVE,
            Lifecycle.EXPORT,
            Lifecycle.PROMOTE,
        ):
            assert event in gate.events
        assert gate.description  # the subclass contract would reject otherwise

    def test_identical_sources_under_strict_policy_pass_with_full_coverage(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        spec = {f"w{i}": ("torch.float32", (4, 4)) for i in range(3)}
        _script_compare_keys(monkeypatch, {key: _comparison(key) for key in spec})
        result = WeightParityGate().run(
            _gate_ctx(_FakeSource("cand", spec), _FakeSource("ref", spec), STRICT_TOLERANCE_POLICY)
        )
        assert result.verdict is Verdict.PASS
        assert not result.blocking
        assert result.coverage.checked == 3
        assert result.coverage.expected == 3  # union equals intersection: full coverage
        assert "3 tensor keys adjudicated" in result.detail
        assert "3 bitwise-exact" in result.detail
        assert "strict-bitexact" in result.detail
        assert result.evidence["finding_count"] == 0
        assert result.evidence["only_in_left"] == []
        assert result.evidence["only_in_right"] == []

    def test_value_divergence_fails_with_first_finding_named(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        spec = {f"k{i:02d}": ("torch.float32", (4, 4)) for i in range(10)}
        table: dict[str, TensorComparison] = {key: _comparison(key) for key in spec}
        for i in range(9):
            key = f"k{i:02d}"
            table[key] = _comparison(key, bitwise=False, mismatched=4, max_abs_diff=0.5, cosine=0.1)
        _script_compare_keys(monkeypatch, table)
        result = WeightParityGate().run(
            _gate_ctx(_FakeSource("cand", spec), _FakeSource("ref", spec))
        )
        assert result.verdict is Verdict.FAIL
        assert result.blocking
        assert "9 keys outside tolerance" in result.detail
        assert "k00 (differ)" in result.detail  # findings ordered by key, first named
        assert result.evidence["finding_count"] == 9
        # Evidence truncates to 8 finding dicts, but the count stays honest.
        assert len(result.evidence["findings"]) == 8
        assert result.evidence["findings"][0]["key"] == "k00"
        assert result.evidence["findings"][0]["status"] == "differ"

    def test_dtype_mismatch_fails_as_dtype_not_as_differ(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        script = _script_compare_keys(monkeypatch, {"head": _comparison("head")})
        left = _FakeSource(
            "cand", {"embed": ("torch.float32", (4, 4)), "head": ("torch.float32", (4, 4))}
        )
        right = _FakeSource(
            "ref", {"embed": ("torch.float64", (4, 4)), "head": ("torch.float32", (4, 4))}
        )
        result = WeightParityGate().run(_gate_ctx(left, right))
        assert result.verdict is Verdict.FAIL
        assert script.calls[0][0] == "head"  # the cast-around path never ran
        assert "dtype_mismatch" in result.detail
        assert "embed" in result.detail
        assert result.evidence["skipped"][0][0] == "embed"

    def test_one_sided_keys_fail_and_coverage_records_the_shortfall(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _script_compare_keys(monkeypatch, {"w": _comparison("w")})
        left = _FakeSource("cand", {"w": ("torch.float32", (4, 4)), "orphan": ("x", (1,))})
        right = _FakeSource("ref", {"w": ("torch.float32", (4, 4))})
        result = WeightParityGate().run(_gate_ctx(left, right))
        assert result.verdict is Verdict.FAIL
        assert "1 keys only in left" in result.detail
        # checked < expected is kept as a fact even though the verdict was already
        # blocking: coverage is reported, not implied by the verdict.
        assert result.coverage.checked == 1
        assert result.coverage.expected == 2
        assert result.evidence["only_in_left"] == ["orphan"]

    def test_zero_common_keys_downgrades_to_vacuous_not_pass(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Incident #6 wearing this gate's id: PASS would mean the framework re-shipped
        # the exact bug it exists to prevent.
        script = _script_compare_keys(monkeypatch, {})
        result = WeightParityGate().run(
            _gate_ctx(
                _FakeSource("cand", {"alpha": ("torch.float32", (4, 4))}),
                _FakeSource("ref", {"beta": ("torch.float32", (4, 4))}),
            )
        )
        assert script.calls == []
        assert result.verdict is Verdict.VACUOUS
        assert result.verdict is not Verdict.PASS
        assert result.blocking
        assert "0 common tensor keys" in result.detail
        assert "proves nothing" in result.detail
        assert result.coverage.checked == 0
        assert result.coverage.is_vacuous
        assert result.evidence["only_in_left"] == ["alpha"]
        assert result.evidence["only_in_right"] == ["beta"]

    def test_a_creator_worthless_reader_error_fails_closed_as_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # In the audited estate a thrown exception counted as a pass. Here the gate
        # framework converts it to ERROR; this test pins that the gate cooperates by
        # not swallowing it first.
        _script_compare_keys(monkeypatch, {"w": RuntimeError("corrupt shard")})
        spec = {"w": ("torch.float32", (4, 4))}
        result = WeightParityGate().run(
            _gate_ctx(_FakeSource("cand", spec), _FakeSource("ref", spec))
        )
        assert result.verdict is Verdict.ERROR
        assert result.blocking
        assert "corrupt shard" in result.detail

    # Was a strict xfail, and it was Incident #6 one level down: Coverage counted KEYS,
    # so a report whose only key held a zero-element tensor reached checked == expected
    # == 1 with not one element compared, and the gate passed. A key is a container;
    # elements are the evidence. Fixed by surfacing the element total on the report and
    # adjudicating vacuity there, so a second caller cannot reintroduce it.
    def test_gate_must_not_pass_when_every_key_compared_zero_elements(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Scripted on dcp's POST-fix NO_ELEMENTS contract. The previous script fed
        # the old comparator shape (bitwise identity over zero elements, zeroed
        # statistics), and the old assertion accepted anything-but-PASS — so when
        # dcp started abstaining, this test watched the gate convert the resulting
        # parity crash into ERROR and waved it through. The required outcome is
        # narrower and is now pinned exactly: a blocking, stated VACUOUS.
        script = _script_compare_keys(
            monkeypatch,
            {
                "empty.w": _comparison(
                    "empty.w",
                    elements=0,
                    bitwise=False,
                    max_abs_diff=math.inf,
                    cosine=None,
                    rms_a=0.0,
                    rms_b=0.0,
                    shape=(0,),
                )
            },
        )
        spec = {"empty.w": ("torch.float32", (0,))}
        result = WeightParityGate().run(
            _gate_ctx(_FakeSource("cand", spec), _FakeSource("ref", spec))
        )
        assert script.calls == []  # zero elements are adjudicated from metadata
        assert result.verdict is not Verdict.PASS
        assert result.verdict is Verdict.VACUOUS
        assert result.blocking
        assert "0 elements were compared" in result.detail
        assert result.coverage.is_vacuous
        assert result.evidence["compared_elements"] == 0
        assert result.evidence["skipped"][0][0] == "empty.w"

    def test_controls_cover_both_directions(self) -> None:
        # A gate that can only pass is a rubber stamp; a gate that can only fire is
        # noise. The control set must prove both directions exist and are named.
        controls = WeightParityGate().controls()
        kinds = {control.kind for control in controls}
        assert ControlKind.MUST_FIRE in kinds
        assert ControlKind.MUST_PASS in kinds
        names = [control.name for control in controls]
        assert len(names) == len(set(names))
        assert any("expert" in name for name in names)

    def test_gate_passes_on_streamed_keys_while_naming_empty_keys_uncompared(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # An unallocated buffer (0 rows) in an otherwise bit-identical pair of
        # exports must not block the run — blocking on it would mint a failure from
        # content nobody read, doctrine 5's symmetry in reverse — but it must be
        # named as not-compared, never quietly acquitted.
        script = _script_compare_keys(monkeypatch, {"real": _comparison("real")})
        spec = {
            "empty.buf": ("torch.float32", (0, 8)),
            "real": ("torch.float32", (4, 4)),
        }
        result = WeightParityGate().run(
            _gate_ctx(_FakeSource("cand", spec), _FakeSource("ref", spec))
        )
        assert script.calls == [("real", 4096, 0.01, 0.999)]
        assert result.verdict is Verdict.PASS
        assert "empty.buf" in result.detail
        assert "0 elements" in result.detail
        assert result.evidence["finding_count"] == 0
        assert result.evidence["skipped"][0][0] == "empty.buf"
        assert result.evidence["compared_elements"] == 16


@pytest.mark.slow
class TestShippedControlsAgainstRealExports:
    """The control fixtures write real safetensors exports from real torch tensors.

    This is the end-to-end positive/negative control pair: if the gate cannot acquit
    two bit-identical exports, or cannot block the sign-flipped-expert corruption,
    every synthetic assertion above is moot. Requires torch + safetensors.
    """

    def test_all_shipped_controls_verify(self, fresh_registry: GateRegistry) -> None:
        pytest.importorskip("torch")
        pytest.importorskip("safetensors")
        fresh_registry.register(WeightParityGate())
        assert verify_controls(fresh_registry) == []

    def test_identical_exports_control_passes_with_full_key_coverage(self) -> None:
        pytest.importorskip("torch")
        pytest.importorskip("safetensors")
        gate = WeightParityGate()
        (control,) = [c for c in gate.controls() if c.kind is ControlKind.MUST_PASS]
        result = gate.run(control.make_ctx())
        assert result.verdict is Verdict.PASS
        assert result.coverage.checked == 3
        assert result.coverage.expected == 3
        # Float32 bit-exactness is honestly attainable: identical bytes in, EXACT out.
        assert "bitwise-exact" in result.detail

    def test_diverged_expert_control_blocks_and_names_the_expert_key(self) -> None:
        pytest.importorskip("torch")
        pytest.importorskip("safetensors")
        gate = WeightParityGate()
        fires = [c for c in gate.controls() if c.kind is ControlKind.MUST_FIRE]
        by_name = {c.name: c for c in fires}
        assert "wholesale-replaced-expert-tensor" in by_name
        assert "dtype-mismatch-report" in by_name
        diverged = gate.run(by_name["wholesale-replaced-expert-tensor"].make_ctx())
        assert diverged.blocking
        assert diverged.verdict is Verdict.FAIL
        assert "experts" in diverged.detail
        dtype = gate.run(by_name["dtype-mismatch-report"].make_ctx())
        assert dtype.blocking
        assert "dtype_mismatch" in dtype.detail
