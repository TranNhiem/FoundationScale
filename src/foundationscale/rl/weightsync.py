"""The ``WeightSync`` contract: train-view weights moved to generate-view weights.

This module closes the contract rollout.py deferred at lines 9-12: the
sibling edge that moves weights from the training view to the generation
view (design section 3.6, stage 3b). ``WeightSync`` is a SIBLING of
``RolloutSource``, never part of it -- one level wider and every
generation-engine adapter re-implements synchronisation policy. The
``sync`` mapping pairs each side's parameter names; both sides are
adapter-owned and the core knows nothing about what either name resolves
to. No transport lives here either: the design section 7 item 3
measurement is now TAKEN (verdict CLEAR_WITH_ABSTENTIONS) and its result
is that NO single transport wins -- resharding and a collective beat a
full copy per byte at the large end, the full copy wins at the small
per-call floor, and the largest cell was WITHHELD for spread over
tolerance. The transport is therefore chosen by the REALIZATION at
construction -- never by this contract, never by the call -- and is
EVIDENCED in the returned ``SyncReport.transport``. The per-transport
ratios are deliberately not restated here: they are countables sitting
in a slot no gate re-measures, they belong in the campaign record, and a
docstring carrying them goes stale the moment the measurement is re-run.

Consequently this module exposes NO throughput or bytes-per-second
property, on purpose. A rate is an in-transport quantity, not a
cross-transport one: full_copy declares 2x payload and collective/reshard
declare 3x, so a GB/s figure derived from ``bytes_moved / seconds`` would
compare payload conventions, not transports, and a number that compares
conventions reads as a comparison it is not. ``bytes_moved`` and
``seconds`` are recorded raw, separately, each abstainable.

WHAT THIS MODULE DOES NOT CLAIM: anything about which transport is best
-- the measurement abstained at scale; anything about staleness -- design
section 3.6 leaves staleness open because Phase 1 unknown 5 is unread, so
``is_stale`` is ``bool | None`` and never omitted and never defaulted:
OMITTING the field makes absence read as FRESH, and defaulting ``False``
asserts an unmeasured fact; anything about device, dtype, or sharding
semantics -- the mapping is names, plain strings on both sides; and
anything about parameter CONTENT -- a transferred name asserts movement,
never quality.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import isfinite
from typing import Protocol, runtime_checkable

__all__ = (
    "SyncCapabilities",
    "SyncCapabilityRefusal",
    "SyncReport",
    "SyncReportRefusal",
    "WeightSync",
    "check_sync_capabilities",
    "verify_sync",
)


class SyncCapabilityRefusal(ValueError):
    # Raised when the setup handshake fails: a required transport the
    # realization does not offer, a non-name in the requirement, or a
    # requirement so empty the check would answer vacuously.
    # CONSTRUCTION/CONFIGURATION-side only -- distinct from
    # SyncReportRefusal, which is the call-time data half. Fail closed;
    # never treat a claim as a delivery.
    pass


class SyncReportRefusal(ValueError):
    # Raised when a report cannot be built or fails verification: a
    # non-total transfer record, a malformed count, or a report whose
    # offered set disagrees with the mapping the sync was handed.
    # CALL-TIME DATA-side only -- distinct from SyncCapabilityRefusal,
    # which is the configuration half. A report of transferred-over-
    # nothing reading complete is the all([]) shape this class exists
    # to refuse.
    pass


def _checked_names(
    values: tuple[str, ...], *, field: str, refusal: type[ValueError]
) -> tuple[str, ...]:
    # Shared name-tuple validation: every entry a non-empty str (absence
    # of a name is not a name -- a sync cannot move something it cannot
    # name), no name twice (one parameter, one record -- a duplicated
    # name usually hides the entry the caller meant to declare).
    names = tuple(values)
    seen: dict[str, int] = {}
    for index, name in enumerate(names):
        if not isinstance(name, str) or not name:
            raise refusal(
                f"{field}[{index}]={name!r}: every parameter name must "
                f"be a non-empty str; absence of a name is not a name, "
                f"and a sync cannot move something it cannot name"
            )
        if name in seen:
            raise refusal(
                f"{field}[{index}] repeats {name!r}, first recorded at "
                f"{field}[{seen[name]}]: one parameter, one record -- "
                f"a duplicated name usually hides the entry the caller "
                f"meant to declare"
            )
        seen[name] = index
    return names


@dataclass(frozen=True, slots=True)
class SyncCapabilities:
    """What a weight-sync realization CLAIMS about itself at setup.

    ``transports`` names the transports the realization can run --
    non-empty (a realization offering nothing has nothing to check
    against and would pass vacuously), each a non-empty ``str``, no
    name twice. WHICH transport runs is chosen here, at construction, by
    the realization -- never by this contract and never by the call:
    the design section 7 item 3 measurement found no single winner and
    withheld its largest cell, so the contract keeps no opinion.
    ``reports_per_rank_failure`` and ``reports_staleness`` state whether
    the realization populates ``failed_ranks`` and ``is_stale`` on its
    reports; both are unmeasured claims recorded for the run manifest --
    no check in this module corroborates them.
    """

    transports: tuple[str, ...]
    reports_per_rank_failure: bool
    reports_staleness: bool

    def __post_init__(self) -> None:
        transports = tuple(self.transports)
        if not transports:
            raise SyncCapabilityRefusal(
                "transports is EMPTY: a realization offering no transport "
                "has nothing to check against, and checking it would pass "
                "vacuously -- an empty capability set is not a capability"
            )
        object.__setattr__(
            self,
            "transports",
            _checked_names(transports, field="transports", refusal=SyncCapabilityRefusal),
        )


@dataclass(frozen=True, slots=True)
class SyncReport:
    """The MEASUREMENT-half record of one ``sync`` call.

    ``transport`` names which transport RAN -- evidenced, not chosen by
    the caller (the design section 7 item 3 measurement withheld its
    largest cell and found no cross-size winner, so the realization
    chose at construction and the report records what happened).
    ``offered`` is the mapping keys the sync was handed; ``transferred``
    and ``skipped`` MUST partition ``offered`` exactly -- duplicate-free,
    disjoint, union equal to ``offered``. A non-total map is
    unrepresentable here rather than merely discouraged: a report of
    transferred-over-nothing reading complete is the all([]) shape.

    ``failed_ranks`` lists the ranks that failed, stored sorted; a
    per-rank failure is REPORT-valued, never exception-valued, and never
    summed away into ``complete`` -- see the property. ``bytes_moved``
    is ``None`` when unmeasured, never ``0``: an unmeasured contribution
    is None, absent is not zero, and a zero would sit in a denominator.
    ``seconds`` is ``None`` when withheld or abstained, never ``0.0``.
    ``is_stale`` is ``bool | None`` with NO default: design section 3.6
    leaves staleness open because Phase 1 unknown 5 is unread, omitting
    the field would make absence read as FRESH, and defaulting ``False``
    would assert an unmeasured fact. None means unmeasured.

    WHAT IS NOT CLAIMED: any rate. This module exposes no throughput
    property because a rate is an in-transport quantity, not a
    cross-transport one -- full_copy declares 2x payload and
    collective/reshard declare 3x, and ``bytes_moved / seconds`` would
    compare payload conventions while reading as a transport comparison.
    """

    transport: str
    offered: tuple[str, ...]
    transferred: tuple[str, ...]
    skipped: tuple[str, ...]
    failed_ranks: tuple[int, ...] = ()
    bytes_moved: int | None = None
    seconds: float | None = None
    is_stale: bool | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.transport, str) or not self.transport:
            raise SyncReportRefusal(
                f"transport={self.transport!r}: the transport that ran "
                f"must be named by a non-empty str; a sync that cannot "
                f"name what ran evidenced nothing"
            )
        offered = _checked_names(tuple(self.offered), field="offered", refusal=SyncReportRefusal)
        if not offered:
            raise SyncReportRefusal(
                "offered is EMPTY: a sync handed no parameters measured "
                "nothing, and a report over nothing reading complete is "
                "the all([]) shape -- 0 of 0 offered parameters are "
                "accounted for"
            )
        object.__setattr__(self, "offered", offered)
        transferred = _checked_names(
            tuple(self.transferred),
            field="transferred",
            refusal=SyncReportRefusal,
        )
        skipped = _checked_names(tuple(self.skipped), field="skipped", refusal=SyncReportRefusal)
        offered_set = frozenset(offered)
        skipped_set = frozenset(skipped)
        overlap = tuple(name for name in transferred if name in skipped_set)
        if overlap:
            raise SyncReportRefusal(
                f"{len(overlap)} names appear in BOTH transferred "
                f"({len(transferred)}) and skipped ({len(skipped)}): "
                f"{', '.join(overlap)}; a parameter either moved or it "
                f"did not -- the two records must be disjoint"
            )
        outside = tuple(name for name in transferred + skipped if name not in offered_set)
        if outside:
            raise SyncReportRefusal(
                f"{len(outside)} names in transferred ({len(transferred)}) "
                f"or skipped ({len(skipped)}) are not among the "
                f"{len(offered)} offered: {', '.join(outside)}; a report "
                f"may only account for parameters the sync was handed"
            )
        covered = frozenset(transferred) | frozenset(skipped)
        if covered != offered_set:
            missing = tuple(name for name in offered if name not in covered)
            raise SyncReportRefusal(
                f"{len(missing)} of {len(offered)} offered parameters are "
                f"in neither transferred ({len(transferred)}) nor skipped "
                f"({len(skipped)}): {', '.join(missing)}; transferred and "
                f"skipped must partition offered EXACTLY -- a non-total "
                f"report would read partial coverage as complete"
            )
        object.__setattr__(self, "transferred", transferred)
        object.__setattr__(self, "skipped", skipped)
        ranks = tuple(self.failed_ranks)
        seen_ranks: set[int] = set()
        for index, rank in enumerate(ranks):
            if isinstance(rank, bool) or not isinstance(rank, int):
                raise SyncReportRefusal(
                    f"failed_ranks[{index}]={rank!r}: a rank is an int; "
                    f"True is not 1 (bool is checked first because "
                    f"isinstance(True, int)) and no float names a rank"
                )
            if rank < 0:
                raise SyncReportRefusal(f"failed_ranks[{index}]={rank!r}: rank ids are >= 0")
            if rank in seen_ranks:
                raise SyncReportRefusal(
                    f"failed_ranks[{index}] repeats rank {rank}: one rank, one failure record"
                )
            seen_ranks.add(rank)
        object.__setattr__(self, "failed_ranks", tuple(sorted(ranks)))
        if self.bytes_moved is not None:
            if isinstance(self.bytes_moved, bool) or not isinstance(self.bytes_moved, int):
                raise SyncReportRefusal(
                    f"bytes_moved={self.bytes_moved!r}: bytes_moved is "
                    f"None (unmeasured) or an int; True is not a byte "
                    f"count"
                )
            if self.bytes_moved < 0:
                raise SyncReportRefusal(
                    f"bytes_moved={self.bytes_moved}: a byte count is "
                    f">= 0; None means unmeasured, never a negative or "
                    f"an invented 0"
                )
        if self.seconds is not None:
            # Admitted by value via float(), bool branch first because
            # isinstance(True, int) would admit it.
            if isinstance(self.seconds, bool):
                raise SyncReportRefusal(f"seconds={self.seconds!r}: True is not a duration")
            try:
                seconds = float(self.seconds)
            except (TypeError, ValueError) as exc:
                raise SyncReportRefusal(
                    f"seconds={self.seconds!r}: must be None or a real-valued duration"
                ) from exc
            if not isfinite(seconds):
                raise SyncReportRefusal(
                    f"seconds={seconds!r}: NaN and inf are not "
                    f"durations -- every comparison against NaN is "
                    f"False, so a NaN duration would pass any latency "
                    f"gate unchecked"
                )
            if seconds < 0.0:
                raise SyncReportRefusal(
                    f"seconds={seconds}: a duration is >= 0.0; None "
                    f"means withheld or abstained, never a negative"
                )
            object.__setattr__(self, "seconds", seconds)
        if self.is_stale is not None and not isinstance(self.is_stale, bool):
            raise SyncReportRefusal(
                f"is_stale={self.is_stale!r}: None (unmeasured) or a "
                f"real bool; staleness is open in design 3.6 and no "
                f"other value may stand in for it"
            )

    @property
    def complete(self) -> bool:
        """Every offered parameter transferred and no rank failed."""
        return self.failed_ranks == () and self.skipped == ()

    @property
    def partial(self) -> bool:
        """At least one rank failed. A per-rank failure is never summed
        away into complete -- even with zero skips, a failed rank means
        this report is partial."""
        return self.failed_ranks != ()


@runtime_checkable
class WeightSync(Protocol):
    """Moves weights train-view to generate-view without the core knowing
    how.

    ``sync`` takes the name mapping -- train-side name to generate-side
    name, both adapter-owned and opaque here -- and returns a
    ``SyncReport``. Partial transfer is REPORT-valued, never
    exception-valued: a rank failure or a skipped parameter arrives in
    ``failed_ranks``/``skipped`` for :func:`verify_sync` to refuse, not
    as a raised exception that hides the rest of the record.
    ``capabilities`` is the CLAIM half checked once at setup by
    :func:`check_sync_capabilities`. This protocol is a SIBLING of
    ``RolloutSource``, per the design section 3.3 sizing argument: one
    level wider and every adapter re-implements synchronisation policy.
    """

    def sync(self, mapping: Mapping[str, str]) -> SyncReport: ...

    def capabilities(self) -> SyncCapabilities: ...


def _checked_required(required: Sequence[str], *, caller: str) -> tuple[str, ...]:
    # Shape validation for the setup half. An empty requirement is
    # refused rather than passed: every requirement in an empty set is
    # trivially met because there are none, the check passes having
    # measured nothing, and a vacuous pass is the exact failure this
    # framework exists to prevent.
    names = tuple(required)
    if not names:
        raise SyncCapabilityRefusal(
            f"{caller} got an EMPTY required set: an empty requirement "
            f"set makes the check vacuously pass -- every requirement "
            f"in it is trivially met because there are none -- and a "
            f"vacuous pass is the exact failure this framework exists "
            f"to prevent"
        )
    for index, name in enumerate(names):
        if not isinstance(name, str) or not name:
            raise SyncCapabilityRefusal(
                f"required[{index}]={name!r}: every required transport "
                f"must be a non-empty str; absence of a name is not a "
                f"name"
            )
    return names


def check_sync_capabilities(
    *,
    required: Sequence[str],
    capabilities: SyncCapabilities,
    origin: str = "<weight-sync>",
) -> tuple[str, ...]:
    """The CLAIM half of the handshake, run once at setup.

    WHAT IS CLAIMED on success: every transport in ``required`` appears
    among the realization's offered transports. The validated required
    set is returned so the caller's denominator is exactly what was
    checked, never a superset it inferred.

    WHAT IS NOT CLAIMED: that any transport will be CHOSEN or will WIN
    -- the realization picks at construction, per the design section 7
    item 3 measurement, which found no single winner and withheld its
    largest cell -- and nothing about delivery: this function compares
    declarations and never sees a report, which is why
    :func:`verify_sync` exists.

    The refusal lists EVERY missing transport against the full
    requirement with the denominator stated ("N of M"): a half-met
    requirement must never read as a met one. An empty ``required`` is
    refused outright as VACUOUS.
    """
    names = _checked_required(required, caller="check_sync_capabilities")
    available = frozenset(capabilities.transports)
    missing = tuple(name for name in names if name not in available)
    if missing:
        raise SyncCapabilityRefusal(
            f"{len(missing)} of {len(names)} required transports are "
            f"unavailable from {origin}: {', '.join(missing)}; the "
            f"realization offers {capabilities.transports}. Refusing at "
            f"setup costs nothing; discovering the gap after weights "
            f"are stale costs a run"
        )
    return names


def verify_sync(
    report: SyncReport,
    *,
    expected: Mapping[str, str],
    origin: str = "<weight-sync>",
) -> int:
    """The MEASUREMENT half of the handshake, run after every ``sync``.

    ``capabilities()`` is a CLAIM and this function is the measurement
    that tests the run against the mapping the sync was actually handed.

    WHAT IS CLAIMED on success: the returned ``int`` is the number of
    parameters actually TRANSFERRED. That count -- not ``len(expected)``,
    not ``len(report.offered)``, not a capabilities claim -- is the
    denominator any downstream verdict about this sync must use. The two
    differ exactly when the sync skipped something, which is why the
    transferred count is what is returned: a skip is permitted, and
    reading ``len(expected)`` as though it were the transferred count is
    the reading this return value exists to prevent.

    WHAT IS NOT CLAIMED: that every offered parameter moved -- a skip is
    report-valued and passes here, carried out in the shrunken count
    rather than in an exception; anything about parameter content (a
    transferred name asserts movement, not quality); anything about the
    mapping VALUES (both sides are adapter-owned and opaque to the
    core); and any rate -- no throughput quantity is derivable here
    because payload conventions differ per transport.

    Four refusals, in order. An empty ``expected`` mapping is refused as
    VACUOUS: verifying a sync against nothing measures nothing. A report
    whose ``offered`` set disagrees with the expected mapping's keys is
    refused, naming the disagreement in both directions against both
    denominators -- the report must account for exactly what it was
    handed, and two sets of equal size can still be different sets. A
    PARTIAL report -- one with a failed rank -- is refused: a per-rank
    failure is never summed away into a pass, and unlike a skip it is
    not a smaller sync but a broken one. And a report that transferred
    NOTHING is refused rather than returned as 0: zero is not a
    denominator, and a caller handed 0 divides by it or reads it as a
    sync that had nothing to do.
    """
    expected_keys = tuple(expected.keys())
    if not expected_keys:
        raise SyncReportRefusal(
            f"{origin} was verified against an EMPTY expected mapping: a "
            f"sync handed no parameters measured nothing, so this check "
            f"would pass vacuously -- and a vacuous pass would read as a "
            f"healthy sync over an empty denominator"
        )
    offered_set = frozenset(report.offered)
    expected_set = frozenset(expected_keys)
    if offered_set != expected_set:
        unexpected = tuple(name for name in report.offered if name not in expected_set)
        unoffered = tuple(name for name in expected_keys if name not in offered_set)
        raise SyncReportRefusal(
            f"the report from {origin} does not account for the mapping "
            f"it was handed: {len(unexpected)} of {len(report.offered)} "
            f"offered parameters are not in the expected mapping "
            f"({', '.join(unexpected) or 'none'}) and {len(unoffered)} of "
            f"{len(expected_keys)} expected parameters were never offered "
            f"({', '.join(unoffered) or 'none'}). Two sets of equal size "
            f"can still be different sets, so the counts alone do not "
            f"settle it -- a report over a different set measures a "
            f"different sync"
        )
    if report.partial:
        raise SyncReportRefusal(
            f"the sync from {origin} is PARTIAL: {len(report.failed_ranks)} "
            f"ranks failed {report.failed_ranks}, having transferred "
            f"{len(report.transferred)} of {len(report.offered)} offered "
            f"parameters. A per-rank failure is report-valued, never "
            f"exception-valued, and is never summed away into a pass -- "
            f"unlike a skip, a failed rank is not a smaller sync"
        )
    if not report.transferred:
        raise SyncReportRefusal(
            f"the sync from {origin} transferred 0 of "
            f"{len(report.offered)} offered parameters (all "
            f"{len(report.skipped)} skipped): nothing moved, so nothing "
            f"was measured, and returning 0 would hand the caller an "
            f"empty denominator to divide by"
        )
    return len(report.transferred)
