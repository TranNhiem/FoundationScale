import math
from collections.abc import Mapping

import pytest

from foundationscale.rl.weightsync import (
    SyncCapabilities,
    SyncCapabilityRefusal,
    SyncReport,
    SyncReportRefusal,
    WeightSync,
    check_sync_capabilities,
    verify_sync,
)


def _report(**overrides: object) -> SyncReport:
    kwargs = {
        "transport": "reshard",
        "offered": ("w.a", "w.b"),
        "transferred": ("w.a", "w.b"),
        "skipped": (),
    }
    kwargs.update(overrides)
    return SyncReport(**kwargs)


def test_sync_report_refuses_a_non_str_transport() -> None:
    with pytest.raises(SyncReportRefusal):
        _report(transport=7)


def test_sync_report_refuses_an_empty_transport_name() -> None:
    with pytest.raises(SyncReportRefusal, match="non-empty str"):
        _report(transport="")


def test_sync_report_refuses_an_empty_offered_set() -> None:
    with pytest.raises(SyncReportRefusal):
        _report(offered=(), transferred=(), skipped=())


def test_sync_report_refuses_duplicate_entries_in_offered_naming_the_repeat() -> None:
    with pytest.raises(SyncReportRefusal, match="repeats"):
        _report(offered=("w.a", "w.a"), transferred=("w.a",), skipped=())


def test_sync_report_refuses_a_non_total_mapping_naming_both_counts() -> None:
    with pytest.raises(SyncReportRefusal, match="1 of 2"):
        _report(offered=("w.a", "w.b"), transferred=("w.a",), skipped=())


def test_sync_report_refuses_transferred_and_skipped_overlap() -> None:
    with pytest.raises(SyncReportRefusal, match="disjoint"):
        _report(
            offered=("w.a",),
            transferred=("w.a",),
            skipped=("w.a",),
        )


def test_sync_report_refuses_a_transferred_entry_not_in_offered() -> None:
    with pytest.raises(SyncReportRefusal):
        _report(
            offered=("w.a",),
            transferred=("w.a", "w.elsewhere"),
            skipped=(),
        )


def test_sync_report_refuses_duplicate_entries_in_transferred() -> None:
    with pytest.raises(SyncReportRefusal, match="repeat"):
        _report(
            offered=("w.a",),
            transferred=("w.a", "w.a"),
            skipped=(),
        )


def test_sync_report_refuses_duplicate_entries_in_skipped() -> None:
    with pytest.raises(SyncReportRefusal, match="repeat"):
        _report(
            offered=("w.a", "w.b"),
            transferred=("w.a",),
            skipped=("w.b", "w.b"),
        )


def test_sync_report_bytes_moved_none_round_trips_distinct_from_zero() -> None:
    unmeasured = _report(bytes_moved=None)
    measured_zero = _report(bytes_moved=0)
    assert unmeasured.bytes_moved is None
    assert measured_zero.bytes_moved == 0
    assert measured_zero.bytes_moved is not None


def test_sync_report_refuses_a_negative_bytes_moved() -> None:
    with pytest.raises(SyncReportRefusal):
        _report(bytes_moved=-1)


def test_sync_report_refuses_true_as_bytes_moved_because_bool_is_int() -> None:
    with pytest.raises(SyncReportRefusal):
        _report(bytes_moved=True)


def test_sync_report_seconds_none_round_trips_distinct_from_zero_point_zero() -> None:
    withheld = _report(seconds=None)
    measured_zero = _report(seconds=0.0)
    assert withheld.seconds is None
    assert measured_zero.seconds == 0.0
    assert measured_zero.seconds is not None


def test_sync_report_refuses_nan_seconds_because_every_comparison_against_nan_is_false() -> None:
    with pytest.raises(SyncReportRefusal):
        _report(seconds=float("nan"))


def test_sync_report_refuses_infinite_seconds() -> None:
    with pytest.raises(SyncReportRefusal):
        _report(seconds=math.inf)


def test_sync_report_accepts_seconds_admitted_by_value_via_float() -> None:
    report = _report(seconds=3)
    assert report.seconds == 3.0


def test_sync_report_is_stale_none_round_trips_distinct_from_false() -> None:
    unmeasured = _report(is_stale=None)
    freshest = _report(is_stale=False)
    assert unmeasured.is_stale is None
    assert freshest.is_stale is False


def test_sync_report_refuses_a_non_bool_non_none_is_stale() -> None:
    with pytest.raises(SyncReportRefusal):
        _report(is_stale=1)


def test_sync_report_failed_ranks_are_stored_sorted() -> None:
    report = _report(
        transferred=("w.a",),
        skipped=("w.b",),
        failed_ranks=(3, 0, 2),
    )
    assert report.failed_ranks == (0, 2, 3)


def test_sync_report_refuses_a_negative_failed_rank() -> None:
    with pytest.raises(SyncReportRefusal):
        _report(failed_ranks=(-1,))


def test_sync_report_refuses_a_bool_failed_rank() -> None:
    with pytest.raises(SyncReportRefusal):
        _report(failed_ranks=(True,))


def test_sync_report_refuses_duplicate_failed_ranks() -> None:
    with pytest.raises(SyncReportRefusal):
        _report(failed_ranks=(1, 1))


def test_sync_report_with_failed_ranks_and_no_skips_is_not_complete() -> None:
    report = _report(failed_ranks=(2,))
    assert report.skipped == ()
    assert report.failed_ranks != ()
    assert report.complete is False
    assert report.partial is True


def test_sync_report_happy_path_two_parameters_is_complete() -> None:
    report = _report(bytes_moved=128, seconds=0.01)
    assert report.complete is True
    assert report.partial is False


def test_sync_report_skipped_only_is_not_complete_but_not_partial() -> None:
    report = _report(transferred=("w.a",), skipped=("w.b",))
    assert report.complete is False
    assert report.partial is False


def test_sync_capabilities_refuses_an_empty_transports_tuple() -> None:
    with pytest.raises(SyncCapabilityRefusal):
        SyncCapabilities(transports=(), reports_per_rank_failure=True, reports_staleness=True)


def test_sync_capabilities_refuses_an_empty_transport_name() -> None:
    with pytest.raises(SyncCapabilityRefusal):
        SyncCapabilities(
            transports=("reshard", ""), reports_per_rank_failure=True, reports_staleness=True
        )


def test_sync_capabilities_refuses_duplicate_transports() -> None:
    with pytest.raises(SyncCapabilityRefusal):
        SyncCapabilities(
            transports=("reshard", "reshard"),
            reports_per_rank_failure=True,
            reports_staleness=True,
        )


def test_a_class_missing_sync_is_not_a_weight_sync_instance() -> None:
    class OnlyCapabilities:
        def capabilities(self) -> SyncCapabilities:
            return SyncCapabilities(
                transports=("reshard",),
                reports_per_rank_failure=True,
                reports_staleness=True,
            )

    assert not isinstance(OnlyCapabilities(), WeightSync)


def test_a_class_with_both_methods_is_a_weight_sync_instance() -> None:
    class FullSync:
        def sync(self, mapping: Mapping[str, str]) -> SyncReport:
            return _report()

        def capabilities(self) -> SyncCapabilities:
            return SyncCapabilities(
                transports=("reshard",),
                reports_per_rank_failure=True,
                reports_staleness=True,
            )

    assert isinstance(FullSync(), WeightSync)


def _capabilities(*transports: str) -> SyncCapabilities:
    return SyncCapabilities(
        transports=transports,
        reports_per_rank_failure=True,
        reports_staleness=True,
    )


def test_check_sync_capabilities_refuses_an_empty_required_set_as_vacuous() -> None:
    with pytest.raises(SyncCapabilityRefusal, match="vacuously"):
        check_sync_capabilities(required=(), capabilities=_capabilities("reshard"))


def test_check_sync_capabilities_refuses_a_missing_transport_naming_both_counts() -> None:
    with pytest.raises(SyncCapabilityRefusal, match="1 of 2"):
        check_sync_capabilities(
            required=("reshard", "collective"),
            capabilities=_capabilities("reshard"),
        )


def test_check_sync_capabilities_returns_exactly_the_validated_required_set() -> None:
    required = ("reshard", "collective")
    result = check_sync_capabilities(
        required=required, capabilities=_capabilities("reshard", "collective", "full_copy")
    )
    assert result == required


def test_verify_sync_refuses_offered_differing_from_expected_keys() -> None:
    report = _report(offered=("w.a",), transferred=("w.a",), skipped=())
    with pytest.raises(SyncReportRefusal, match="1 of 2"):
        verify_sync(report, expected={"w.a": "g.a", "w.b": "g.b"})


def test_verify_sync_refuses_a_partial_report() -> None:
    report = _report(failed_ranks=(1,))
    with pytest.raises(SyncReportRefusal):
        verify_sync(report, expected={"w.a": "g.a", "w.b": "g.b"})


def test_verify_sync_refuses_an_empty_expected_mapping_as_vacuous() -> None:
    report = _report()
    with pytest.raises(SyncReportRefusal, match="vacuously"):
        verify_sync(report, expected={})


def test_verify_sync_returns_the_transferred_count_on_the_happy_path() -> None:
    report = _report()
    count = verify_sync(report, expected={"w.a": "g.a", "w.b": "g.b"})
    assert count == 2


def test_verify_sync_returns_the_transferred_count_not_the_mapping_length() -> None:
    report = _report(transferred=("w.a",), skipped=("w.b",))
    count = verify_sync(report, expected={"w.a": "g.a", "w.b": "g.b"})
    assert count == 1


def test_verify_sync_refuses_a_report_that_transferred_nothing() -> None:
    # Every offered parameter skipped is not a partial report -- no rank
    # failed -- so it passes the partial check and would return 0. Zero is
    # not a denominator: a caller handed it divides by it, or reads it as a
    # sync that had nothing to do. Same refusal as verify_generated's
    # zero-row batch, for the same reason.
    report = _report(transferred=(), skipped=("w.a", "w.b"))
    assert report.partial is False
    with pytest.raises(SyncReportRefusal, match="transferred 0 of 2"):
        verify_sync(report, expected={"w.a": "g.a", "w.b": "g.b"})


def test_verify_sync_refuses_equal_sized_offered_and_expected_sets_that_differ() -> None:
    # The refusal must name the DISAGREEMENT, not just the two sizes. A
    # message built from counts alone reads "offered 2 parameters but the
    # expected mapping has 2 keys" here -- a refusal that states no reason,
    # over two sets that are genuinely different.
    report = _report(offered=("w.a", "w.b"), transferred=("w.a", "w.b"), skipped=())
    with pytest.raises(SyncReportRefusal) as excinfo:
        verify_sync(report, expected={"w.a": "g.a", "w.c": "g.c"})
    message = str(excinfo.value)
    assert "w.b" in message, message
    assert "w.c" in message, message
    assert "1 of 2" in message, message


def test_check_sync_capabilities_refuses_a_non_name_in_required() -> None:
    with pytest.raises(SyncCapabilityRefusal, match="absence of a name is not a name"):
        check_sync_capabilities(required=("reshard", ""), capabilities=_capabilities("reshard"))


def test_sync_report_exposes_no_rate_property() -> None:
    # Deliberate absence, asserted so it cannot be added back by convenience.
    # bytes_moved / seconds would compare payload CONVENTIONS -- a full copy
    # declares 2x payload, a collective or reshard 3x -- while reading as a
    # transport comparison. A rate is an in-transport quantity.
    report = _report(bytes_moved=1024, seconds=2.0)
    for forbidden in ("rate", "throughput", "gb_per_second", "bytes_per_second"):
        assert not hasattr(report, forbidden), (
            f"SyncReport grew a {forbidden!r} property: a rate derived across "
            f"transports compares payload conventions, not transports"
        )
