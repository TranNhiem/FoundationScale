# ===== FILE: test_ib_discovery.py
"""Tests for #520 runtime glob resolution, against fake sysfs trees only.

No real ``/sys`` is ever touched: every case builds its own device tree under
``tmp_path``. Nothing here may be skipped on any platform (FS_FORBID_SKIPS=1),
which is why the unreadable-file case uses a directory where a file is
expected -- portable everywhere, including as root, where ``chmod 000`` still
reads fine.
"""

from pathlib import Path

import pytest

from foundationscale.topology import DiscoveryVerdict, _ib_hca_value, discover_ib_devices

# The measured GB200 estate shape from #520: 0,1,4,5 are NDR InfiniBand,
# 2,3,6,7 are bond0 Ethernet. Kept as module constants so the regression test
# reads like the finding.
_IB_HCAS = ("mlx5_0", "mlx5_1", "mlx5_4", "mlx5_5")
_ETHERNET_DEVICES = ("mlx5_2", "mlx5_3", "mlx5_6", "mlx5_7")


def _make_device(sysfs_root: Path, name: str, link_layer: str, state: str) -> None:
    """Create one fake sysfs HCA with a port-1 link_layer and state."""
    port1 = sysfs_root / name / "ports" / "1"
    port1.mkdir(parents=True)
    (port1 / "link_layer").write_text(f"{link_layer}\n", encoding="utf-8")
    (port1 / "state").write_text(f"{state}\n", encoding="utf-8")


def test_inactive_named_state_is_rejected(tmp_path: Path) -> None:
    """REGRESSION (review defect 1): ``"7: INACTIVE"`` contains ACTIVE as a
    substring; a containment test would SELECT a port that is down. The state
    grammar must be parsed and the NAME compared, so INACTIVE -- and any other
    name embedding 'ACTIVE' -- never reaches the prefix list.
    """
    _make_device(tmp_path, "mlx5_0", "InfiniBand", "4: ACTIVE")
    _make_device(tmp_path, "mlx5_1", "InfiniBand", "7: INACTIVE")
    _make_device(tmp_path, "mlx5_2", "InfiniBand", "9: DEACTIVATE")

    selected, verdict, announcements = discover_ib_devices("mlx5_*", tmp_path)

    assert selected == ("mlx5_0",)
    assert verdict is DiscoveryVerdict.OK
    joined = "\n".join(announcements)
    assert "7: INACTIVE" in joined
    assert "9: DEACTIVATE" in joined
    assert _ib_hca_value(selected) == "mlx5_0:1"


def test_sysfs_root_that_is_a_file_reports_unmeasured(tmp_path: Path) -> None:
    """REGRESSION (review defect 2): a root that passes ``exists()`` but is a
    plain file makes the bare ``iterdir()`` in pre-fix code raise
    NotADirectoryError -- a crash, not an announcement. The outcome must be
    UNMEASURED, legibly.
    """
    not_a_dir = tmp_path / "not_a_dir"
    not_a_dir.write_text("i am a file\n", encoding="utf-8")

    selected, verdict, announcements = discover_ib_devices("mlx5_*", not_a_dir)

    assert selected == ()
    assert verdict is DiscoveryVerdict.UNMEASURED
    joined = "\n".join(announcements)
    assert "UNREADABLE" in joined
    assert "UNMEASURED" in joined


def test_verdict_discriminates_all_four_outcomes_by_type(tmp_path: Path) -> None:
    """REGRESSION (review defect 3): UNMEASURED, glob-matched-nothing,
    matched-but-none-survived, and declared-none all yield an empty selection.
    Pre-fix code discriminated only via announcement prose -- substring-matching
    English to derive a 0/5/95/96 exit code is the silent-wrong-answer defect
    relocated one layer up. The verdict must be a typed enum member, distinct
    per outcome.
    """
    ib_root = tmp_path / "with_ib"
    _make_device(ib_root, "mlx5_0", "InfiniBand", "4: ACTIVE")
    eth_root = tmp_path / "all_ethernet"
    _make_device(eth_root, "mlx5_2", "Ethernet", "4: ACTIVE")

    _, v_ok, _ = discover_ib_devices("mlx5_*", ib_root)
    _, v_declared, _ = discover_ib_devices("", ib_root)
    _, v_unmeasured, _ = discover_ib_devices("mlx5_*", tmp_path / "absent")
    _, v_no_match, _ = discover_ib_devices("does_not_match_*", ib_root)
    _, v_none_survived, _ = discover_ib_devices("mlx5_*", eth_root)

    assert v_ok is DiscoveryVerdict.OK
    assert v_declared is DiscoveryVerdict.DECLARED_NONE
    assert v_unmeasured is DiscoveryVerdict.UNMEASURED
    assert v_no_match is DiscoveryVerdict.NO_MATCH
    assert v_none_survived is DiscoveryVerdict.NONE_SURVIVED
    # Distinctness by identity, full-stop -- the exit-code mapping can never
    # collapse two different outcome shapes onto one rc.
    assert len({v_ok, v_declared, v_unmeasured, v_no_match, v_none_survived}) == 5


def test_empty_render_is_refused_not_empty_string() -> None:
    """REGRESSION (review defect 4): pre-fix, ``_ib_hca_value(())`` returned
    ``""`` with only a docstring telling callers not to export it. "Do not
    export" must be enforced: an empty selection raises, so no code path can
    produce an exported-but-empty NCCL_IB_HCA.
    """
    with pytest.raises(ValueError):
        _ib_hca_value(())


def test_gb200_estate_glob_selects_only_ib_hcas_by_name(tmp_path: Path) -> None:
    """REGRESSION for #520: 8-device estate, glob mlx5_* picks the 4 IB HCAs.

    The Ethernet four are asserted absent BY NAME. A pure count assertion would
    pass if the wrong four devices were ever selected, which is precisely the
    failure this finding describes.
    """
    for name in _IB_HCAS:
        _make_device(tmp_path, name, "InfiniBand", "4: ACTIVE")
    for name in _ETHERNET_DEVICES:
        _make_device(tmp_path, name, "Ethernet", "4: ACTIVE")

    selected, verdict, announcements = discover_ib_devices("mlx5_*", tmp_path)

    assert selected == _IB_HCAS
    assert verdict is DiscoveryVerdict.OK
    for ethernet_name in _ETHERNET_DEVICES:
        assert ethernet_name not in selected
    assert _ib_hca_value(selected) == "mlx5_0:1,mlx5_1:1,mlx5_4:1,mlx5_5:1"

    # 'Ethernet' vs 'InfiniBand' is the entire finding; the raw values must
    # reach the announcement, not a paraphrase.
    joined = "\n".join(announcements)
    assert "Ethernet" in joined
    assert "InfiniBand" in joined
    assert announcements


def test_glob_matching_only_ethernet_yields_empty_selection_and_loud_line(
    tmp_path: Path,
) -> None:
    """Matches-but-none-survived is the wrong-fabric defect, and must be loud."""
    for name in _ETHERNET_DEVICES:
        _make_device(tmp_path, name, "Ethernet", "4: ACTIVE")

    selected, verdict, announcements = discover_ib_devices("mlx5_*", tmp_path)

    assert selected == ()
    assert verdict is DiscoveryVerdict.NONE_SURVIVED
    assert any(
        "NONE" in line and "survived" in line and "wrong fabric" in line for line in announcements
    ), announcements


def test_missing_sysfs_root_reports_unmeasured_not_no_match(tmp_path: Path) -> None:
    """'Cannot look' and 'no IB present' must never read alike."""
    absent_root = tmp_path / "there_is_no_sysfs_here"

    selected, verdict, announcements = discover_ib_devices("mlx5_*", absent_root)

    assert selected == ()
    assert verdict is DiscoveryVerdict.UNMEASURED
    assert any("UNMEASURED" in line for line in announcements), announcements
    assert any("ABSENT" in line for line in announcements), announcements
    # ...and the unmeasured line must not be confusable with a clean walk that
    # simply matched nothing.
    assert not any("summary" in line for line in announcements), announcements


def test_glob_matching_nothing_is_a_clean_refusal_not_unmeasured(
    tmp_path: Path,
) -> None:
    """A readable surface whose glob misses everything is rc 5 refusal
    (NO_MATCH): we looked, and absence is the measurement."""
    _make_device(tmp_path, "mlx5_0", "InfiniBand", "4: ACTIVE")

    selected, verdict, announcements = discover_ib_devices("does_not_match_*", tmp_path)

    assert selected == ()
    assert verdict is DiscoveryVerdict.NO_MATCH
    assert any("summary" in line for line in announcements), announcements


def test_empty_pattern_means_the_profile_declared_none(tmp_path: Path) -> None:
    """Whitespace-only and empty are both undeclared, not errors."""
    for pattern in ("", "   ", "\t\n"):
        selected, verdict, announcements = discover_ib_devices(pattern, tmp_path)
        assert selected == ()
        assert verdict is DiscoveryVerdict.DECLARED_NONE
        joined = "\n".join(announcements)
        assert "declared none" in joined
        assert announcements


def test_inactive_ib_device_is_excluded_and_state_is_quoted(tmp_path: Path) -> None:
    """An InfiniBand HCA that is DOWN/INIT is off the fabric for this job."""
    _make_device(tmp_path, "mlx5_0", "InfiniBand", "4: ACTIVE")
    _make_device(tmp_path, "mlx5_1", "InfiniBand", "1: DOWN")
    _make_device(tmp_path, "mlx5_2", "InfiniBand", "2: INIT")

    selected, verdict, announcements = discover_ib_devices("mlx5_*", tmp_path)

    assert selected == ("mlx5_0",)
    assert verdict is DiscoveryVerdict.OK
    joined = "\n".join(announcements)
    # The actual state values must appear -- ' DOWN' vs ACTIVE is the reason
    # for the exclusion and has to be visible, not summarized away.
    assert "1: DOWN" in joined
    assert "2: INIT" in joined
    assert "mlx5_1" in joined
    assert "mlx5_2" in joined


def test_unreadable_link_layer_is_announced_and_walk_continues(
    tmp_path: Path,
) -> None:
    """A device we cannot interrogate is announced by name, never silently
    dropped, and never aborts the walk -- devices AFTER it are still selected.

    The unreadable file is a directory where a file is expected, not a chmod:
    that fails identically on every platform and for every uid, so the test
    needs no platform carve-out.
    """
    _make_device(tmp_path, "mlx5_0", "InfiniBand", "4: ACTIVE")
    broken_port1 = tmp_path / "mlx5_1" / "ports" / "1"
    broken_port1.mkdir(parents=True)
    (broken_port1 / "link_layer").mkdir()  # directory where a file is expected
    (broken_port1 / "state").write_text("4: ACTIVE\n", encoding="utf-8")
    _make_device(tmp_path, "mlx5_2", "InfiniBand", "4: ACTIVE")

    selected, verdict, announcements = discover_ib_devices("mlx5_*", tmp_path)

    assert selected == ("mlx5_0", "mlx5_2")
    assert verdict is DiscoveryVerdict.OK
    broken_lines = [line for line in announcements if "mlx5_1" in line]
    assert broken_lines, announcements
    assert any("UNREADABLE" in line for line in broken_lines)


def test_ib_hca_value_pins_port_one() -> None:
    assert _ib_hca_value(("mlx5_0", "mlx5_1")) == "mlx5_0:1,mlx5_1:1"
    assert _ib_hca_value(("mlx5_4",)) == "mlx5_4:1"


def test_announcements_are_never_empty_for_any_input(tmp_path: Path) -> None:
    """The no-silence invariant: every input shape speaks, and speaks in
    non-empty lines. A caller that prints nothing has a bug, not a quiet win."""
    ib_root = tmp_path / "with_ib"
    _make_device(ib_root, "mlx5_0", "InfiniBand", "4: ACTIVE")
    eth_root = tmp_path / "all_ethernet"
    _make_device(eth_root, "mlx5_2", "Ethernet", "4: ACTIVE")
    missing_root = tmp_path / "absent"

    inputs = [
        ("", ib_root),  # declared none
        ("   ", ib_root),  # declared none, whitespace
        ("mlx5_*", missing_root),  # unmeasured surface
        ("mlx5_*", ib_root),  # clean selection
        ("mlx5_*", eth_root),  # matched but none survived
        ("does_not_match_*", ib_root),  # glob matched nothing at all
    ]
    for pattern, root in inputs:
        _, _, announcements = discover_ib_devices(pattern, root)
        assert announcements, (pattern, root)
        assert all(line.strip() for line in announcements), (pattern, root)
        _, _, second_run = discover_ib_devices(pattern, root)
        # Stability: identical inputs produce identical announcements, because
        # these lines are meant to be diffable between runs.
        assert announcements == second_run
