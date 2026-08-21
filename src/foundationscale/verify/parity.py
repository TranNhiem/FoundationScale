"""Numerically honest weight parity between two checkpoint sources.

Why this module exists
----------------------
The audit that produced this framework found that the dangerous training defects are
the ones that *report success*. This module hardens three of them at once:

* Incident #1: a MoE checkpoint wrote 128 experts under 16 local tensor names aliased
  8 ways, and passed tensor-count, dtype, resume and loss checks for two full runs.
  Only a key-wise numeric comparison of examined — and counted — keys shows it.
* Incident #6: the tool written to detect exactly that answered ``all_identity:
  true`` on a known-corrupt artifact, because the compared set was empty and
  ``all([])`` is ``True``. Here the number of common keys is a first-class fact of
  :class:`ParityReport`: a report over zero common keys is VACUOUS and says so, and
  the wrapping :class:`WeightParityGate` lets the frozen gate contract downgrade that
  to a blocking verdict on its own.
* The forensic probe's float32 reduction defect printed a tensor's cosine *with
  itself* as 1.80 and shipped the number. Every statistic derived here passes through
  :func:`_guard_stats`, which raises :class:`ParityInvariantError` on mathematically
  impossible values (cosine outside [-1, 1], nonzero differences between
  bitwise-identical tensors) instead of laundering them into a report.

Two further rules from the record are enforced: dtype and shape mismatches are
*findings*, never silently cast around; and every key is judged against an explicit
:class:`Tolerances` value whose identity is recorded on the result — no magic numbers
hiding inside ``if`` statements.

The heavy lifting (chunked reads, self-validating offsets, float64 streaming
accumulation) is delegated to :mod:`foundationscale.checkpoint.dcp`, whose
``read_box``/``compare_keys`` pair refuses incomplete coverage rather than returning
zeros that look like data. torch and safetensors are therefore needed only when parity
is actually computed; this module imports in a bare stdlib environment.

Conventions: ``left`` is the artifact under test (the candidate), ``right`` is the
reference, and ``rel_frob`` is ``||left - right||_F / ||right||_F``.
"""

from __future__ import annotations

import math
import re
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..gates.core import Control, ControlKind, Coverage, Gate, GateResult, Lifecycle, register

if TYPE_CHECKING:
    from ..checkpoint.dcp import TensorComparison, WeightSource

__all__ = [
    "DEFAULT_TOLERANCE_POLICY",
    "STRICT_TOLERANCE_POLICY",
    "KeyParity",
    "ParityGateContext",
    "ParityInvariantError",
    "ParityReport",
    "ParityStatus",
    "TolerancePolicy",
    "ToleranceRule",
    "Tolerances",
    "WeightParityGate",
    "compare_sources",
]


class ParityInvariantError(ArithmeticError):
    """A derived parity statistic was mathematically impossible.

    The direct descendant of the 1.80 self-cosine incident: when the reduction cannot
    produce the number it produced, the number is evidence of a broken reduction, and
    the correct response is to fail loudly — never to print it.
    """


class ParityStatus(str, Enum):
    """The adjudication of one tensor key present in both sources."""

    EXACT = "exact"
    """Bitwise identical."""

    CLOSE = "close"
    """Not bitwise identical, but within the recorded tolerance on all three statistics."""

    DIFFER = "differ"
    """Outside tolerance. Blocks."""

    DTYPE_MISMATCH = "dtype_mismatch"
    """Same key, different dtypes. Numerically compared: never (no silent casts). Blocks."""

    SHAPE_MISMATCH = "shape_mismatch"
    """Same key, different shapes. Blocks."""

    @property
    def blocking(self) -> bool:
        return self in (
            ParityStatus.DIFFER,
            ParityStatus.DTYPE_MISMATCH,
            ParityStatus.SHAPE_MISMATCH,
        )


# ---------------------------------------------------------------------------
# Tolerance policy — explicit data, never magic numbers in comparisons
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Tolerances:
    """The numeric judgement criteria applied to one tensor key.

    Args:
        name: Stable identifier recorded on every :class:`KeyParity` it judges, so a
            report can say *which* policy acquitted each key.
        max_abs_diff: Ceiling on the largest absolute element difference.
        min_cosine: Floor on the cosine between the two tensors.
        max_rel_frob: Ceiling on ``||left - right||_F / ||right||_F``.
    """

    name: str
    max_abs_diff: float
    min_cosine: float
    max_rel_frob: float

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("Tolerances require a name — anonymous policy is magic numbers")
        if self.max_abs_diff < 0.0:
            raise ValueError(f"max_abs_diff cannot be negative: {self.max_abs_diff}")
        if not -1.0 <= self.min_cosine <= 1.0:
            raise ValueError(f"min_cosine must lie in [-1, 1]: {self.min_cosine}")
        if self.max_rel_frob < 0.0:
            raise ValueError(f"max_rel_frob cannot be negative: {self.max_rel_frob}")

    def describe(self) -> str:
        return (
            f"{self.name}(max_abs_diff<={self.max_abs_diff:g}, "
            f"cosine>={self.min_cosine:g}, rel_frob<={self.max_rel_frob:g})"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "max_abs_diff": self.max_abs_diff,
            "min_cosine": self.min_cosine,
            "max_rel_frob": self.max_rel_frob,
        }


_DEFAULT_TOLERANCES = Tolerances(
    name="default-close",
    max_abs_diff=1e-2,
    min_cosine=0.999,
    max_rel_frob=1e-2,
)
_STRICT_TOLERANCES = Tolerances(
    name="strict-bitexact",
    max_abs_diff=0.0,
    min_cosine=1.0,
    max_rel_frob=0.0,
)


@dataclass(frozen=True)
class ToleranceRule:
    """One ``pattern -> tolerances`` override, matched with ``re.search``."""

    pattern: str
    tolerances: Tolerances

    def __post_init__(self) -> None:
        re.compile(self.pattern)  # raise now, not mid-comparison, on a bad pattern

    def matches(self, key: str) -> bool:
        return re.search(self.pattern, key) is not None


@dataclass(frozen=True)
class TolerancePolicy:
    """The ordered rule set plus default a whole parity report is judged by."""

    default: Tolerances = _DEFAULT_TOLERANCES
    rules: tuple[ToleranceRule, ...] = ()

    def tolerances_for(self, key: str) -> Tolerances:
        """First matching rule wins; otherwise the default."""
        for rule in self.rules:
            if rule.matches(key):
                return rule.tolerances
        return self.default

    def describe(self) -> str:
        base = self.default.describe()
        if not self.rules:
            return base
        rules = ", ".join(f"{r.pattern!r}->{r.tolerances.name}" for r in self.rules)
        return f"{base} + [{rules}]"


DEFAULT_TOLERANCE_POLICY = TolerancePolicy()
STRICT_TOLERANCE_POLICY = TolerancePolicy(default=_STRICT_TOLERANCES)


# ---------------------------------------------------------------------------
# Per-key and whole-report results
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class KeyParity:
    """Everything known about one common tensor key after adjudication.

    Statistics were accumulated in float64 by ``dcp.compare_keys`` (per-key norms) and
    derived here (``rel_frob``), then validated by :func:`_guard_stats`. ``elements``
    is the number of elements actually compared; for metadata-level findings
    (dtype/shape mismatch) it is 0 by design, and visibly so.
    """

    key: str
    status: ParityStatus
    elements: int
    dtype_left: str
    dtype_right: str
    shape_left: tuple[int, ...]
    shape_right: tuple[int, ...]
    mismatched_elements: int
    max_abs_diff: float
    cosine: float | None
    rel_frob: float
    chunks_read: int
    bytes_read: int
    tolerances: Tolerances
    note: str = ""

    @property
    def blocking(self) -> bool:
        return self.status.blocking

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "status": self.status.value,
            "elements": self.elements,
            "shape_left": self.shape_left,
            "shape_right": self.shape_right,
            "dtype_left": self.dtype_left,
            "dtype_right": self.dtype_right,
            "mismatched_elements": self.mismatched_elements,
            "max_abs_diff": self.max_abs_diff,
            "cosine": self.cosine,
            "rel_frob": self.rel_frob,
            "chunks_read": self.chunks_read,
            "bytes_read": self.bytes_read,
            "tolerances": self.tolerances.to_dict(),
            "note": self.note,
        }


@dataclass(frozen=True)
class ParityReport:
    """The adjudication of two weight sources against each other.

    Coverage is stored, not implied: how many keys existed on only one side, how many
    were skipped and why, exactly which tolerances acquitted each compared key, and
    how many elements were actually streamed (:attr:`compared_elements`).
    :attr:`is_vacuous` — zero common keys, or zero compared elements — is the literal
    shape of the audit's ``all([])`` failure and is spelled out in :meth:`render` and
    :meth:`to_dict`.
    """

    left_path: str
    right_path: str
    keys: tuple[KeyParity, ...]
    only_in_left: tuple[str, ...]
    only_in_right: tuple[str, ...]
    skipped: tuple[tuple[str, str], ...]
    """``(key, reason)`` pairs for common keys deliberately not compared numerically."""

    policy: TolerancePolicy
    block_rows: int

    @property
    def compared_elements(self) -> int:
        """Total tensor elements actually streamed and compared across all keys.

        Keys are containers; elements are the evidence. A zero-sized tensor streams
        nothing and defensibly reports ``bitwise_equal=True`` over zero elements —
        counting its key as agreement is the ``all([])`` shape one level down.
        """
        return sum(entry.elements for entry in self.compared)

    @property
    def is_vacuous(self) -> bool:
        """True when nothing was compared — no common keys, or zero elements in total.

        A report over zero common keys compares nothing; so does a report whose
        common tensors are all zero-sized, because bitwise equality over zero
        elements is not evidence of agreement. Both are the ``all([])`` shape. A
        report carrying findings is never vacuous: a dtype or shape mismatch means
        metadata WAS examined and found wanting — evidence of a defect, not the
        absence of evidence.
        """
        return not self.findings and self.compared_elements == 0

    @property
    def findings(self) -> tuple[KeyParity, ...]:
        return tuple(entry for entry in self.keys if entry.blocking)

    @property
    def compared(self) -> tuple[KeyParity, ...]:
        """Keys that received a numeric comparison: not skipped, with elements streamed.

        A zero-element key was adjudicated from metadata alone — nothing flowed
        through the reduction — so it is reported as uncompared, never as matching.
        """
        return tuple(
            entry
            for entry in self.keys
            if (entry.key, entry.note) not in self.skipped_set and entry.elements > 0
        )

    @property
    def skipped_set(self) -> frozenset[tuple[str, str]]:
        return frozenset(self.skipped)

    @property
    def ok(self) -> bool:
        """Complete one-to-one key sets, no blocking findings, and elements examined.

        A vacuous report — zero common keys, or zero compared elements — never
        claims agreement, because no element supports the claim.
        """
        return (
            not self.is_vacuous
            and not self.findings
            and not self.only_in_left
            and not self.only_in_right
        )

    def render(self) -> str:
        if self.is_vacuous:
            verdict = "VACUOUS"
        elif self.ok:
            verdict = "ok"
        else:
            verdict = "FAIL"
        lines = [
            f"weight parity [{verdict}]: {len(self.keys)} common tensor keys, "
            f"{len(self.only_in_left)} only in left, {len(self.only_in_right)} only in "
            f"right, {len(self.skipped)} skipped"
        ]
        if self.is_vacuous:
            if not self.keys:
                lines.append(
                    "  0 common keys — parity over an empty set proves nothing "
                    "(the all([]) is True shape, and it blocks)"
                )
            else:
                lines.append(
                    f"  {len(self.keys)} common keys, 0 elements compared — parity over "
                    "zero elements proves nothing (the all([]) is True shape one level "
                    "down, and it blocks)"
                )
        for entry in self.keys:
            if entry.blocking:
                lines.append(
                    f"  {entry.status.value.upper():>14} {entry.key} "
                    f"[{entry.tolerances.name}] — {entry.note}"
                )
        for key, reason in self.skipped:
            lines.append(f"  {'SKIPPED':>14} {key} — {reason}")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "left_path": self.left_path,
            "right_path": self.right_path,
            "ok": self.ok,
            "vacuous": self.is_vacuous,
            "common_keys": len(self.keys),
            "compared_keys": len(self.compared),
            "compared_elements": self.compared_elements,
            "finding_count": len(self.findings),
            "only_in_left": list(self.only_in_left),
            "only_in_right": list(self.only_in_right),
            "skipped": [list(item) for item in self.skipped],
            "policy": self.policy.describe(),
            "block_rows": self.block_rows,
            "keys": [entry.to_dict() for entry in self.keys],
        }


# ---------------------------------------------------------------------------
# The comparison itself
# ---------------------------------------------------------------------------


def compare_sources(
    left: str | Path | WeightSource,
    right: str | Path | WeightSource,
    *,
    policy: TolerancePolicy = DEFAULT_TOLERANCE_POLICY,
    block_rows: int = 4096,
) -> ParityReport:
    """Compare two weight sources key by key, with float64 statistics and guards.

    Every key common to both sources is adjudicated: bitwise-equal or within its
    recorded tolerances, or a finding. Keys present on only one side and keys skipped
    (dtype/shape mismatch — never silently cast) are reported by name. A report with
    no common keys has :attr:`ParityReport.is_vacuous` true, which the wrapping gate
    converts into a blocking VACUOUS verdict.

    Args:
        left: Candidate checkpoint, as a path (sniffed by ``dcp.open_weights``) or an
            already-open :class:`~foundationscale.checkpoint.dcp.WeightSource`.
        right: Reference checkpoint, same accepted forms.
        policy: The explicit tolerances each key is judged against.
        block_rows: Row-block size forwarded to ``dcp.compare_keys``.

    Returns:
        A :class:`ParityReport`. This function never raises for *findings* — those are
        data. It raises :class:`ParityInvariantError` only when the numerics
        themselves are impossible, and propagates reader errors for corrupt sources.
    """
    left_source, own_left = _ensure_source(left)
    right_source, own_right = _ensure_source(right)
    try:
        return _compare_open_sources(
            left_source, right_source, policy=policy, block_rows=block_rows
        )
    finally:
        # Only close what we opened; a caller-owned source is the caller's problem.
        if own_left:
            left_source.close()
        if own_right:
            right_source.close()


def _ensure_source(obj: str | Path | WeightSource) -> tuple[WeightSource, bool]:
    """Return ``(source, we_opened_it)``. ``isinstance`` against the runtime protocol
    is avoided deliberately: ``WeightSource`` declares a data member (``path``), and
    runtime-checkable protocols with data members refuse ``isinstance`` entirely."""
    if isinstance(obj, (str, Path)):
        from ..checkpoint.dcp import open_weights

        return open_weights(obj), True
    return obj, False


def _compare_open_sources(
    left: WeightSource,
    right: WeightSource,
    *,
    policy: TolerancePolicy,
    block_rows: int,
) -> ParityReport:
    from ..checkpoint.dcp import compare_keys

    left_keys = set(left.tensor_keys())
    right_keys = set(right.tensor_keys())
    if not left_keys or not right_keys:
        # The readers make this unreachable by refusing empty sources at construction;
        # the guard stays because a future WeightSource implementation might not.
        raise ParityInvariantError("a live weight source reported an empty key set")

    common = sorted(left_keys & right_keys)
    only_in_left = tuple(sorted(left_keys - right_keys))
    only_in_right = tuple(sorted(right_keys - left_keys))

    entries: list[KeyParity] = []
    skipped: list[tuple[str, str]] = []
    for key in common:
        tolerances = policy.tolerances_for(key)
        dtype_left, dtype_right = left.dtype(key), right.dtype(key)
        shape_left, shape_right = left.shape(key), right.shape(key)
        if dtype_left != dtype_right:
            reason = f"dtype mismatch {dtype_left} vs {dtype_right}; not numerically compared"
            skipped.append((key, reason))
            entries.append(
                _metadata_entry(
                    key,
                    ParityStatus.DTYPE_MISMATCH,
                    shape_left,
                    shape_right,
                    dtype_left,
                    dtype_right,
                    tolerances,
                    reason,
                )
            )
        elif shape_left != shape_right:
            reason = f"shape mismatch {shape_left} vs {shape_right}; not numerically compared"
            skipped.append((key, reason))
            entries.append(
                _metadata_entry(
                    key,
                    ParityStatus.SHAPE_MISMATCH,
                    shape_left,
                    shape_right,
                    dtype_left,
                    dtype_right,
                    tolerances,
                    reason,
                )
            )
        else:
            cmp = compare_keys(
                left,
                right,
                key,
                block_rows=block_rows,
                close_max_abs_diff=tolerances.max_abs_diff,
                close_min_cosine=tolerances.min_cosine,
            )
            entries.append(_entry_from_comparison(cmp, tolerances))

    return ParityReport(
        left_path=left.path,
        right_path=right.path,
        keys=tuple(entries),
        only_in_left=only_in_left,
        only_in_right=only_in_right,
        skipped=tuple(skipped),
        policy=policy,
        block_rows=block_rows,
    )


def _metadata_entry(
    key: str,
    status: ParityStatus,
    shape_left: tuple[int, ...],
    shape_right: tuple[int, ...],
    dtype_left: Any,
    dtype_right: Any,
    tolerances: Tolerances,
    note: str,
) -> KeyParity:
    """A finding established from metadata alone — 0 elements compared, visibly so."""
    return KeyParity(
        key=key,
        status=status,
        elements=0,
        dtype_left=str(dtype_left),
        dtype_right=str(dtype_right),
        shape_left=shape_left,
        shape_right=shape_right,
        mismatched_elements=0,
        max_abs_diff=math.inf,
        cosine=None,
        rel_frob=math.inf,
        chunks_read=0,
        bytes_read=0,
        tolerances=tolerances,
        note=note,
    )


def _entry_from_comparison(cmp: TensorComparison, tolerances: Tolerances) -> KeyParity:
    rel_frob = _relative_frobenius(cmp)
    _guard_stats(
        key=cmp.key,
        elements=cmp.elements,
        mismatched=cmp.mismatched_elements,
        max_abs_diff=cmp.max_abs_diff,
        rel_frob=rel_frob,
        cosine=cmp.cosine,
    )
    status = _classify(
        bitwise=cmp.bitwise_equal,
        mismatched=cmp.mismatched_elements,
        cosine=cmp.cosine,
        max_abs_diff=cmp.max_abs_diff,
        rel_frob=rel_frob,
        tolerances=tolerances,
    )
    fragments: list[str] = []
    if cmp.elements == 0:
        fragments.append("empty tensor: 0 elements, adjudicated at metadata level")
    if cmp.cosine is None and cmp.elements > 0:
        fragments.append("cosine undefined: at least one side is all zeros")
    if math.isnan(cmp.max_abs_diff) or (math.isnan(cmp.cosine) if cmp.cosine else False):
        fragments.append("NaN/inf payload present in the weights")
    if status is ParityStatus.EXACT:
        head = f"{cmp.elements} elements bitwise identical"
    elif status is ParityStatus.CLOSE:
        head = f"within {tolerances.describe()}"
    else:
        head = f"EXCEEDS {tolerances.describe()}"
    stats = (
        f"cos={_fmt_opt(cmp.cosine)}, max_abs_diff={cmp.max_abs_diff:.6g}, "
        f"rel_frob={rel_frob:.6g}, mismatched={cmp.mismatched_elements}"
    )
    note = f"{head}: {stats}; streamed in {cmp.chunks_read} chunk reads, {cmp.bytes_read} bytes"
    if fragments:
        note += "; " + "; ".join(fragments)
    return KeyParity(
        key=cmp.key,
        status=status,
        elements=cmp.elements,
        dtype_left=cmp.dtype_a,
        dtype_right=cmp.dtype_b,
        shape_left=cmp.shape_a,
        shape_right=cmp.shape_b,
        mismatched_elements=cmp.mismatched_elements,
        max_abs_diff=cmp.max_abs_diff,
        cosine=cmp.cosine,
        rel_frob=rel_frob,
        chunks_read=cmp.chunks_read,
        bytes_read=cmp.bytes_read,
        tolerances=tolerances,
        note=note,
    )


def _relative_frobenius(cmp: TensorComparison) -> float:
    """Derive ``||a - b||_F / ||b||_F`` from the float64 per-key norms.

    ``compare_keys`` streams ``rms_a``, ``rms_b`` and the float64 dot product (via
    ``cosine``); the difference norm is ``rms_a² + rms_b² - 2·cos·rms_a·rms_b``, so no
    second pass over the data is needed. When ``cosine`` is ``None`` one side is all
    zeros, the dot term is provably zero, and a zero reference yields ``inf`` — stated
    as such rather than rounded to something friendlier.
    """
    ra, rb = cmp.rms_a, cmp.rms_b
    cosine = cmp.cosine if cmp.cosine is not None else 0.0
    var_diff = ra * ra + rb * rb - 2.0 * cosine * ra * rb
    if math.isnan(var_diff):
        return math.nan
    if var_diff < 0.0:
        # Float64 rounding at the bitwise-equality edge can dip a hair below zero;
        # anything larger fails _guard_stats through the identical-tensors clauses.
        var_diff = 0.0
    if rb > 0.0:
        return math.sqrt(var_diff) / rb
    return 0.0 if var_diff == 0.0 else math.inf


_COSINE_SLACK = 1e-9
"""Float64 error at a legitimately attained bound is ~1e-15 for these magnitudes;
1e-9 is generous slack that still laughs at 1.80."""

_SELF_COSINE_SLACK = 1e-9


def _guard_stats(
    *,
    key: str,
    elements: int,
    mismatched: int,
    max_abs_diff: float,
    rel_frob: float,
    cosine: float | None,
) -> None:
    """Raise :class:`ParityInvariantError` on mathematically impossible statistics.

    The separation of concerns: a statistic that is possible but bad (a NaN payload
    in the weights, which shows up as mismatched elements alongside NaN diffs) is a
    *finding*. A statistic that is impossible in any data (cosine 1.80, differing
    bitwise-identical tensors) is a *lying reduction*, and it raises.
    """
    if elements < 0 or mismatched < 0 or (elements > 0 and mismatched > elements):
        raise ParityInvariantError(
            f"impossible element counts on {key!r}: {mismatched} mismatched of {elements}"
        )
    if max_abs_diff < 0.0:
        raise ParityInvariantError(f"negative max_abs_diff {max_abs_diff!r} on {key!r}")
    if rel_frob < 0.0:
        raise ParityInvariantError(f"negative rel_frob {rel_frob!r} on {key!r}")
    if (
        cosine is not None
        and math.isfinite(cosine)
        and (cosine > 1.0 + _COSINE_SLACK or cosine < -1.0 - _COSINE_SLACK)
    ):
        raise ParityInvariantError(
            f"impossible cosine {cosine!r} on {key!r}: a cosine outside [-1, 1] is "
            f"a broken accumulator, not a property of the data (the 1.80 incident)"
        )
    if mismatched > 0:
        # Real mismatches make NaN statistics explainable (NaN payload in weights);
        # those are findings, handled downstream.
        return
    # Bitwise-identical tensors pin every remaining statistic to a known value.
    if max_abs_diff != 0.0:
        raise ParityInvariantError(
            f"bitwise-identical tensors produced max_abs_diff={max_abs_diff!r} on {key!r}"
        )
    if rel_frob != 0.0:
        raise ParityInvariantError(
            f"bitwise-identical tensors produced rel_frob={rel_frob!r} on {key!r}"
        )
    if (
        elements > 0
        and cosine is not None
        and (math.isnan(cosine) or abs(cosine - 1.0) > _SELF_COSINE_SLACK)
    ):
        raise ParityInvariantError(
            f"bitwise-identical tensors scored cosine={cosine!r} on {key!r}: a "
            f"tensor's cosine with itself is 1.0; any other printed value is a "
            f"broken reduction laundering itself into a report"
        )


def _classify(
    *,
    bitwise: bool,
    mismatched: int,
    cosine: float | None,
    max_abs_diff: float,
    rel_frob: float,
    tolerances: Tolerances,
) -> ParityStatus:
    if bitwise and mismatched == 0:
        return ParityStatus.EXACT
    within_cosine = cosine is None or math.isnan(cosine) or cosine >= tolerances.min_cosine
    if (
        max_abs_diff <= tolerances.max_abs_diff
        and rel_frob <= tolerances.max_rel_frob
        and within_cosine
    ):
        # NaN statistics land here as False comparisons, hence DIFFER — pathological
        # payloads are findings, not acquittals.
        return ParityStatus.CLOSE
    return ParityStatus.DIFFER


def _fmt_opt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.6g}"


# ---------------------------------------------------------------------------
# The gate
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ParityGateContext:
    """The context :class:`WeightParityGate` consumes.

    Args:
        left_path: Candidate checkpoint directory (or ``.safetensors`` file).
        right_path: Reference checkpoint.
        policy: Tolerances applied per key; strict bit-exactness is a choice, not a
            default, because legitimate converters may redistribute layouts.
        block_rows: Streaming row-block size for the underlying comparisons.
        label: Free-text provenance recorded in gate evidence.
        keepalive: Holds fixture resources (temporary directories) alive for the
            duration of the gate run; the gate never reads it.
    """

    left_path: str | Path
    right_path: str | Path
    policy: TolerancePolicy = DEFAULT_TOLERANCE_POLICY
    block_rows: int = 4096
    label: str = ""
    keepalive: tuple[Any, ...] = ()


@register
class WeightParityGate(Gate):
    """First-class-coverage weight parity between two checkpoint sources.

    This is the checkpoint-side answer to the audit's central lesson. It runs where a
    save defect is still cheap (``FIRST_SAVE``/``SAVE``) and again where a conversion
    defect still has the blast radius in front of it (``EXPORT``/``PROMOTE``). Its
    ``Coverage.checked`` is the number of common tensor keys adjudicated against an
    ``expected`` of the union of both key sets — so zero common keys downgrades to
    VACUOUS via ``Gate.ok`` rather than passing, exactly the path the ``all([])``
    verifier should have taken.
    """

    id = "checkpoint.weight_parity"
    description = (
        "Key-wise numeric parity between two checkpoint sources (e.g. a DCP "
        "checkpoint and its safetensors export); common-key coverage is a first-class "
        "fact, vacuity blocks, and statistics pass an impossible-value guard"
    )
    events = (Lifecycle.FIRST_SAVE, Lifecycle.SAVE, Lifecycle.EXPORT, Lifecycle.PROMOTE)
    context_type = ParityGateContext

    def check(self, ctx: ParityGateContext) -> GateResult:
        report = compare_sources(
            ctx.left_path,
            ctx.right_path,
            policy=ctx.policy,
            block_rows=ctx.block_rows,
        )
        coverage = Coverage(
            checked=len(report.keys),
            unit="tensor keys",
            expected=len(report.keys) + len(report.only_in_left) + len(report.only_in_right),
        )
        evidence: dict[str, Any] = {
            "label": ctx.label or None,
            "left_path": report.left_path,
            "right_path": report.right_path,
            "only_in_left": list(report.only_in_left),
            "only_in_right": list(report.only_in_right),
            "skipped": [list(item) for item in report.skipped],
            "finding_count": len(report.findings),
            "compared_elements": report.compared_elements,
            "findings": [entry.to_dict() for entry in report.findings[:8]],
            "policy": report.policy.describe(),
        }
        if report.is_vacuous:
            # self.ok with zero coverage downgrades to VACUOUS; claiming PASS here
            # would be Incident #6 wearing this gate's id.
            if report.keys:
                # Zero-sized tensors stream nothing: the keys agree, but no element
                # flowed through the reduction. Keys are containers, elements are the
                # evidence — this is the all([]) vacuity with an element denominator,
                # so it blocks on 0 tensor elements, not on 0 keys.
                return self.ok(
                    f"{len(report.keys)} common tensor keys but 0 elements were compared: "
                    "every common tensor is empty, and parity over zero elements proves "
                    "nothing",
                    Coverage.none("tensor elements"),
                    evidence=evidence,
                )
            return self.ok(
                "0 common tensor keys between the two sources; parity over an empty "
                "set proves nothing",
                coverage,
                evidence=evidence,
            )
        problems: list[str] = []
        if report.only_in_left:
            problems.append(
                f"{len(report.only_in_left)} keys only in left: {list(report.only_in_left[:5])}"
            )
        if report.only_in_right:
            problems.append(
                f"{len(report.only_in_right)} keys only in right: {list(report.only_in_right[:5])}"
            )
        if report.findings:
            worst = report.findings[0]
            problems.append(
                f"{len(report.findings)} keys outside tolerance; first: "
                f"{worst.key} ({worst.status.value})"
            )
        if problems:
            return self.fail("; ".join(problems), coverage, evidence=evidence)
        exact = sum(1 for e in report.keys if e.status is ParityStatus.EXACT and e.elements > 0)
        close = sum(1 for e in report.keys if e.status is ParityStatus.CLOSE and e.elements > 0)
        detail = (
            f"{len(report.keys)} tensor keys adjudicated: {exact} bitwise-exact, "
            f"{close} within tolerance ({report.policy.describe()})"
        )
        uncompared = [entry.key for entry in report.keys if entry.elements == 0]
        if uncompared:
            # Zero-sized tensors stream nothing; name them as uncompared rather
            # than folding them into the EXACT count and claiming agreement on
            # evidence nobody gathered.
            detail += (
                f"; {len(uncompared)} keys hold 0 elements and were not compared: {uncompared[:5]}"
            )
        return self.ok(detail, coverage, evidence=evidence)

    def controls(self) -> Sequence[Control]:
        return (
            Control(
                name="identical-exports",
                kind=ControlKind.MUST_PASS,
                make_ctx=_make_identical_sources_ctx,
                note="both exports written from the same tensors under the strict "
                "bit-exact policy; parity must acquit",
            ),
            Control(
                name="wholesale-replaced-expert-tensor",
                kind=ControlKind.MUST_FIRE,
                make_ctx=_make_diverged_sources_ctx,
                note="one expert tensor replaced by a sign-flipped, magnified, "
                "shifted copy — the numeric signature of the aliased-expert incident; "
                "all three statistics exceed every tolerance",
            ),
            Control(
                name="dtype-mismatch-report",
                kind=ControlKind.MUST_FIRE,
                make_ctx=_make_dtype_mismatched_sources_ctx,
                note="one key exported at a different dtype; the gate must flag it "
                "rather than silently cast and compare",
            ),
        )


# ---------------------------------------------------------------------------
# Control fixtures (real safetensors exports written under temp dirs)
# ---------------------------------------------------------------------------

_EXPERT_KEY = "model.layers.0.mlp.experts.experts.linear_fc1.weight"


def _fixture_tensors() -> dict[str, Any]:
    """Small deterministic tensors with non-degenerate norm and zero symmetry.

    Determinism matters: a control that can pass or fail on an RNG seed is a gate
    whose verdict nobody should trust (the lesson of Incident #6 applies upward).
    """
    import torch

    rows = torch.arange(64, dtype=torch.float64)
    cols = torch.arange(24, dtype=torch.float64)
    grid = torch.arange(8 * 24 * 32, dtype=torch.float64).reshape(8, 24, 32)
    return {
        "model.embed.weight": torch.sin(torch.outer(rows, cols) / 47.0).to(torch.float32),
        _EXPERT_KEY: torch.sin(grid / 113.0).to(torch.float32),
        "model.output.weight": torch.cos(torch.outer(cols, rows[:32]) / 89.0).to(torch.float32),
    }


def _write_export(tensors: Mapping[str, Any], directory: Path) -> None:
    from safetensors.torch import save_file

    directory.mkdir(parents=True, exist_ok=True)
    save_file(
        {key: tensor.contiguous() for key, tensor in tensors.items()},
        str(directory / "model.safetensors"),
    )


def _make_identical_sources_ctx() -> ParityGateContext:
    """MUST_PASS fixture: two exports of identical tensors, judged bit-exactly."""
    tmp = tempfile.TemporaryDirectory(prefix="fs-parity-pass-")
    root = Path(tmp.name)
    tensors = _fixture_tensors()
    left, right = root / "left", root / "right"
    _write_export(tensors, left)
    _write_export(tensors, right)
    return ParityGateContext(
        left_path=left,
        right_path=right,
        policy=STRICT_TOLERANCE_POLICY,
        label="control: identical exports",
        keepalive=(tmp,),
    )


def _make_diverged_sources_ctx() -> ParityGateContext:
    """MUST_FIRE fixture: one wholesale-replaced expert tensor.

    The replacement is sign-flipped, magnified and shifted, which drives cosine below
    every floor and both difference metrics above every ceiling — the defect cannot
    squeak through any one statistic.
    """
    tmp = tempfile.TemporaryDirectory(prefix="fs-parity-fire-")
    root = Path(tmp.name)
    tensors = _fixture_tensors()
    corrupted = dict(tensors)
    corrupted[_EXPERT_KEY] = (tensors[_EXPERT_KEY] * -3.0 + 0.7).contiguous()
    left, right = root / "left", root / "right"
    _write_export(tensors, left)
    _write_export(corrupted, right)
    return ParityGateContext(
        left_path=left,
        right_path=right,
        policy=DEFAULT_TOLERANCE_POLICY,
        label="control: wholesale-replaced expert tensor",
        keepalive=(tmp,),
    )


def _make_dtype_mismatched_sources_ctx() -> ParityGateContext:
    """MUST_FIRE fixture: one key exported at a different dtype."""
    tmp = tempfile.TemporaryDirectory(prefix="fs-parity-dtype-")
    root = Path(tmp.name)
    tensors = _fixture_tensors()
    recast = dict(tensors)
    recast["model.embed.weight"] = tensors["model.embed.weight"].to(_torch_float64())
    left, right = root / "left", root / "right"
    _write_export(tensors, left)
    _write_export(recast, right)
    return ParityGateContext(
        left_path=left,
        right_path=right,
        policy=DEFAULT_TOLERANCE_POLICY,
        label="control: dtype mismatch on one key",
        keepalive=(tmp,),
    )


def _torch_float64() -> Any:
    import torch

    return torch.float64
