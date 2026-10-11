"""Pinned fabric-declaration arms for ``foundationscale.topology.apply_fabric_declaration``.

The function's whole contract lives in four asymmetries that a passing build must keep
apart: one interface name feeding two consumers (NCCL and gloo agree, or they silently
disagree at runtime); an operator export that always wins, including the empty string;
``mnnvl_available`` that writes 0 when it is False, or when it is True but no nvidia-imex
daemon runs on the node; and ``ib_hca_pattern`` that reaches NCCL_IB_HCA only through sysfs
discovery of InfiniBand ports, never as the raw glob. Every test here calls the real
function with a real registered profile fetched through ``profile_by_name`` and reshaped
with ``dataclasses.replace`` -- ClusterProfile's constructor is not part of the surface
under test, so this module never guesses at it.

No monkeypatch, no GPU: ``environ`` is a parameter precisely so a plain ``dict`` stands in
for ``os.environ``, and the two host probes (the InfiniBand sysfs root and the IMEX daemon
check) are parameters too. The module-level wrapper below pins both to "absent" and
"unmeasured" so no test reads the host it happens to run on.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from foundationscale.topology import (
    PROFILES,
    ClusterProfile,
    _imex_daemon_running,
    profile_by_name,
)
from foundationscale.topology import apply_fabric_declaration as _apply_fabric_declaration

# A sysfs root that cannot exist, so discovery reads UNMEASURED unless a test builds a tree.
_ABSENT_SYSFS = Path("/nonexistent/fs-test/infiniband")
_IB_HCAS = ("mlx5_0", "mlx5_1", "mlx5_4", "mlx5_5")
_ETHERNET_DEVICES = ("mlx5_2", "mlx5_3", "mlx5_6", "mlx5_7")


def apply_fabric_declaration(
    profile: ClusterProfile, environ: dict[str, str], **probes: Any
) -> tuple[str, ...]:
    """The real function with both host probes pinned unless a test passes its own."""
    probes.setdefault("ib_sysfs_root", _ABSENT_SYSFS)
    probes.setdefault("imex_running", lambda: None)
    return _apply_fabric_declaration(profile, environ, **probes)


def _make_device(sysfs_root: Path, name: str, link_layer: str, state: str = "4: ACTIVE") -> None:
    """One fake sysfs HCA with a port-1 link_layer and state (the shape discovery reads)."""
    port1 = sysfs_root / name / "ports" / "1"
    port1.mkdir(parents=True)
    (port1 / "link_layer").write_text(f"{link_layer}\n", encoding="utf-8")
    (port1 / "state").write_text(f"{state}\n", encoding="utf-8")


def _two_tray_node(root: Path) -> Path:
    """A GB200 tray's eight HCAs: four InfiniBand compute ports, four Ethernet bond members."""
    for name in _IB_HCAS:
        _make_device(root, name, "InfiniBand")
    for name in _ETHERNET_DEVICES:
        _make_device(root, name, "Ethernet")
    return root


# The variables the function considers, in the order it considers them: two socket
# consumers, the MNNVL switch, and the field it declines. The shape test below pins
# exactly one announcement line per entry -- a variable considered but not reported is a
# quiet success, and this contract explicitly forbids those.
_EXPECTED_VARIABLES: tuple[str, ...] = (
    "NCCL_SOCKET_IFNAME",
    "GLOO_SOCKET_IFNAME",
    "NCCL_MNNVL_ENABLE",
    "NCCL_IB_HCA",
)


def _registered_profiles() -> list[ClusterProfile]:
    """Every profile in ``PROFILES``, resolved through ``profile_by_name``.

    The sweep property does not depend on the registry's container shape -- dict of
    name-to-profile or tuple of profiles -- so the helper absorbs both and always
    returns profile objects. A dict takes the ``profile_by_name`` path so the lookup
    half of the registry is exercised as well as the storage half.
    """
    if isinstance(PROFILES, Mapping):
        return [profile_by_name(name) for name in PROFILES]
    return list(PROFILES)


def _profile_with(**overrides: object) -> ClusterProfile:
    """A real registered profile with named fields replaced, never constructed raw.

    ``dataclasses.replace`` re-runs whatever validation ClusterProfile declares for
    itself, so a fixture that drifted out of contract fails here loudly instead of
    laundering a defect through a by-hand constructor call.
    """
    profiles = _registered_profiles()
    assert profiles, "PROFILES is empty: there is no registered profile to build variants from"
    return dataclasses.replace(profiles[0], **overrides)


def _line_for(announcements: tuple[str, ...], variable: str) -> str:
    """The one announcement line belonging to a variable.

    Returns a bare line rather than a list so a caller asserting on a duplicated or
    missing variable's line fails inside the helper, next to the assertion that
    depended on it, rather than with an IndexError three frames later.
    """
    matches = [line for line in announcements if line.startswith(variable)]
    assert len(matches) == 1, (
        f"expected exactly one announcement line for {variable}, got {matches!r}"
    )
    return matches[0]


def test_one_interface_name_reaches_both_socket_consumers() -> None:
    """A declared interface lands in NCCL_SOCKET_IFNAME AND GLOO_SOCKET_IFNAME:
    the data plane and the rendezvous plane are pointed at the same network,
    and asserting only the NCCL half would pass on the bug this pair exists to
    catch.
    """
    profile = _profile_with(nccl_socket_ifname="eth0", mnnvl_available=False)
    environ: dict[str, str] = {}

    announcements = apply_fabric_declaration(profile, environ)

    assert environ["NCCL_SOCKET_IFNAME"] == "eth0"
    assert environ["GLOO_SOCKET_IFNAME"] == "eth0"
    assert "NCCL_SOCKET_IFNAME=eth0" in _line_for(announcements, "NCCL_SOCKET_IFNAME")
    assert "GLOO_SOCKET_IFNAME=eth0" in _line_for(announcements, "GLOO_SOCKET_IFNAME")


def test_surrounding_whitespace_on_the_declaration_is_not_exported() -> None:
    """A declaration padded with whitespace is stripped before it reaches the
    environment: NCCL compares interface names literally, and " eth0 " is not
    an interface this machine has.
    """
    profile = _profile_with(nccl_socket_ifname="  eth0\t", mnnvl_available=False)
    environ: dict[str, str] = {}

    apply_fabric_declaration(profile, environ)

    assert environ["NCCL_SOCKET_IFNAME"] == "eth0"
    assert environ["GLOO_SOCKET_IFNAME"] == "eth0"


def test_operator_set_values_are_left_exactly_as_is() -> None:
    """A value the operator already exported wins over the profile, byte for
    byte: the profile is a default for an operator who did not choose, and an
    operator who exported chose -- possibly to work around the very default
    being applied here.
    """
    profile = _profile_with(nccl_socket_ifname="eth0", mnnvl_available=False)
    environ = {
        "NCCL_SOCKET_IFNAME": "ib0",
        "GLOO_SOCKET_IFNAME": "bond0",
        "NCCL_MNNVL_ENABLE": "1",
    }
    original = dict(environ)

    apply_fabric_declaration(profile, environ)

    assert environ == original


def test_an_empty_string_from_the_operator_is_still_a_choice() -> None:
    """The empty string is falsy but it is not absent: an operator export of ""
    is left exactly as is and the announcement reports the empty value, which an
    implementation guarded by ``if environ.get(var):`` would silently overwrite.
    """
    profile = _profile_with(nccl_socket_ifname="eth0", mnnvl_available=False)
    environ = {"NCCL_SOCKET_IFNAME": ""}

    announcements = apply_fabric_declaration(profile, environ)

    assert environ["NCCL_SOCKET_IFNAME"] == ""
    # The gloo half was not exported at all, so the profile default still
    # applies there -- only the variable the operator touched is protected.
    assert environ["GLOO_SOCKET_IFNAME"] == "eth0"
    nccl_line = _line_for(announcements, "NCCL_SOCKET_IFNAME")
    assert "left alone" in nccl_line
    assert "''" in nccl_line


def test_a_left_alone_announcement_names_the_operator_value_and_the_profile_value() -> None:
    """A variable left alone is announced with BOTH values visible -- what the
    operator exported and what the profile would have said -- so a log reader
    can reconstruct the decision without re-running anything.
    """
    profile = _profile_with(nccl_socket_ifname="eth0", mnnvl_available=False)
    environ = {"NCCL_SOCKET_IFNAME": "ib0"}

    announcements = apply_fabric_declaration(profile, environ)

    line = _line_for(announcements, "NCCL_SOCKET_IFNAME")
    # Both substring checks are independent properties: a fix that hard-coded
    # either half of the sentence would fail the other.
    assert "'ib0'" in line
    assert "'eth0'" in line
    assert "left alone" in line


def test_mnnvl_unavailable_disables_mnnvl() -> None:
    """A profile declaring MNNVL absent yields NCCL_MNNVL_ENABLE=0: without the
    pin a multi-rank job on this fabric can select multi-node NVLink and spin
    in its first collective at full GPU utilisation rather than fail.
    """
    profile = _profile_with(nccl_socket_ifname="eth0", mnnvl_available=False)
    environ: dict[str, str] = {}

    announcements = apply_fabric_declaration(profile, environ)

    assert environ["NCCL_MNNVL_ENABLE"] == "0"
    assert "NCCL_MNNVL_ENABLE=0" in _line_for(announcements, "NCCL_MNNVL_ENABLE")


@pytest.mark.parametrize(("imex", "state"), [(True, "running"), (None, "UNMEASURED")])
def test_mnnvl_available_sets_nothing_and_says_why(imex: bool | None, state: str) -> None:
    """MNNVL available is not MNNVL required: with the IMEX daemon running, or with
    its state unmeasurable, the True branch exports nothing -- the key is ABSENT
    from the mapping, not set to "1" -- and the line says which of the two it was.
    """
    profile = _profile_with(nccl_socket_ifname="eth0", mnnvl_available=True)
    environ: dict[str, str] = {}

    announcements = apply_fabric_declaration(profile, environ, imex_running=lambda: imex)

    assert "NCCL_MNNVL_ENABLE" not in environ
    line = _line_for(announcements, "NCCL_MNNVL_ENABLE")
    assert "nothing set" in line
    assert "AVAILABLE" in line
    assert state in line


def test_mnnvl_available_without_an_imex_daemon_is_disabled() -> None:
    """An NVLink Switch in the rack does not make multi-node NVLink usable: with no
    nvidia-imex daemon on the node NCCL would still select MNNVL and spin in its
    first collective, so the profile's "available" is overridden to 0 and the
    line names the missing daemon.
    """
    profile = _profile_with(nccl_socket_ifname="eth0", mnnvl_available=True)
    environ: dict[str, str] = {}

    announcements = apply_fabric_declaration(profile, environ, imex_running=lambda: False)

    assert environ["NCCL_MNNVL_ENABLE"] == "0"
    assert "nvidia-imex" in _line_for(announcements, "NCCL_MNNVL_ENABLE")


def test_an_operator_mnnvl_export_wins_over_an_unavailable_profile() -> None:
    """The operator's NCCL_MNNVL_ENABLE survives even a profile that would pin
    it to 0, and the announcement records both the operator's value and the
    profile's mnnvl_available declaration so the conflict is visible.
    """
    profile = _profile_with(nccl_socket_ifname="eth0", mnnvl_available=False)
    environ = {"NCCL_MNNVL_ENABLE": "1"}

    announcements = apply_fabric_declaration(profile, environ)

    assert environ["NCCL_MNNVL_ENABLE"] == "1"
    line = _line_for(announcements, "NCCL_MNNVL_ENABLE")
    assert "'1'" in line
    assert "mnnvl_available=False" in line
    assert "left alone" in line


@pytest.mark.parametrize("mnnvl_available", [False, True])
@pytest.mark.parametrize("ifname", ["eth0", ""])
def test_ib_hca_is_not_set_without_a_discovery_surface(ifname: str, mnnvl_available: bool) -> None:
    """With no readable InfiniBand sysfs root NCCL_IB_HCA is not written, whatever
    else the profile declares: the glob itself is never exported, and an
    unmeasured fabric gets no pin -- on every combination of the other two inputs.
    """
    profile = _profile_with(nccl_socket_ifname=ifname, mnnvl_available=mnnvl_available)
    environ: dict[str, str] = {}

    apply_fabric_declaration(profile, environ)

    assert "NCCL_IB_HCA" not in environ


def test_the_ib_hca_decline_names_the_declined_field_and_its_value() -> None:
    """The declined field is announced, not silent: the NCCL_IB_HCA line names
    ``ib_hca_pattern``, carries the pattern the profile actually declared, and
    states that nothing was set -- a field explicitly declined reads differently
    in a log than a field the function forgot.
    """
    profile = _profile_with(nccl_socket_ifname="eth0", mnnvl_available=False)
    environ: dict[str, str] = {}

    announcements = apply_fabric_declaration(profile, environ)

    line = _line_for(announcements, "NCCL_IB_HCA")
    assert repr(profile.ib_hca_pattern) in line
    assert "NOT set" in line
    assert "unmeasured" in line


def test_ib_hca_pins_exactly_the_infiniband_ports_of_a_gb200_tray(tmp_path: Path) -> None:
    """REGRESSION for the two-tray run: unpinned, NCCL paired a local InfiniBand HCA
    with a remote RoCE one and died in its first large collective. Discovery must
    export the four InfiniBand ports, port 1 each, and none of the Ethernet four.
    """
    profile = _profile_with(nccl_socket_ifname="eth0", mnnvl_available=False)
    environ: dict[str, str] = {}

    announcements = apply_fabric_declaration(
        profile, environ, ib_sysfs_root=_two_tray_node(tmp_path)
    )

    assert environ["NCCL_IB_HCA"] == "mlx5_0:1,mlx5_1:1,mlx5_4:1,mlx5_5:1"
    for name in _ETHERNET_DEVICES:
        assert name not in environ["NCCL_IB_HCA"]
    assert "applied" in _line_for(announcements, "NCCL_IB_HCA")


def test_ib_hca_is_not_set_when_only_ethernet_ports_match(tmp_path: Path) -> None:
    """The glob matched, the fabric did not: no InfiniBand port survives, so nothing
    is exported and the line names the discovery verdict.
    """
    for name in _ETHERNET_DEVICES:
        _make_device(tmp_path, name, "Ethernet")
    profile = _profile_with(nccl_socket_ifname="eth0", mnnvl_available=False)
    environ: dict[str, str] = {}

    announcements = apply_fabric_declaration(profile, environ, ib_sysfs_root=tmp_path)

    assert "NCCL_IB_HCA" not in environ
    assert "none_survived" in _line_for(announcements, "NCCL_IB_HCA")


@pytest.mark.parametrize("operator_value", ["mlx5_0", ""])
def test_an_operator_ib_hca_export_wins_over_discovery(tmp_path: Path, operator_value: str) -> None:
    """An operator's NCCL_IB_HCA, including the empty string, is never replaced by
    the discovered list -- even when discovery would have succeeded.
    """
    profile = _profile_with(nccl_socket_ifname="eth0", mnnvl_available=False)
    environ = {"NCCL_IB_HCA": operator_value}

    announcements = apply_fabric_declaration(
        profile, environ, ib_sysfs_root=_two_tray_node(tmp_path)
    )

    assert environ["NCCL_IB_HCA"] == operator_value
    assert "left alone" in _line_for(announcements, "NCCL_IB_HCA")


def test_a_blank_ib_hca_pattern_sets_nothing(tmp_path: Path) -> None:
    """A profile with no ib_hca_pattern has no opinion: nothing is discovered or set."""
    profile = _profile_with(nccl_socket_ifname="eth0", mnnvl_available=False, ib_hca_pattern="")
    environ: dict[str, str] = {}

    announcements = apply_fabric_declaration(
        profile, environ, ib_sysfs_root=_two_tray_node(tmp_path)
    )

    assert "NCCL_IB_HCA" not in environ
    assert "declares no ib_hca_pattern" in _line_for(announcements, "NCCL_IB_HCA")


def test_imex_probe_reads_process_names_and_abstains_without_proc(tmp_path: Path) -> None:
    """The IMEX probe answers True/False from ``<proc>/*/comm`` and None when there
    is no proc tree to read -- an absent surface is not evidence of an absent daemon.
    """
    proc = tmp_path / "proc"
    (proc / "1").mkdir(parents=True)
    (proc / "1" / "comm").write_text("systemd\n", encoding="utf-8")
    assert _imex_daemon_running(proc) is False
    (proc / "42").mkdir()
    (proc / "42" / "comm").write_text("nvidia-imex\n", encoding="utf-8")
    assert _imex_daemon_running(proc) is True
    assert _imex_daemon_running(tmp_path / "absent") is None


def test_imex_probe_skips_an_entry_that_vanished_mid_scan(tmp_path: Path) -> None:
    """A process that exits while /proc is being walked leaves a comm that cannot be
    read; that entry is skipped, not taken as evidence either way.
    """
    proc = tmp_path / "proc"
    (proc / "7" / "comm").mkdir(parents=True)  # read_text raises IsADirectoryError
    (proc / "9").mkdir()
    (proc / "9" / "comm").write_text("bash\n", encoding="utf-8")
    # No readable entry is the daemon, so the walk must visit the unreadable one
    # whatever order the filesystem lists them in -- and still answer False.
    assert _imex_daemon_running(proc) is False


@pytest.mark.parametrize("blank", ["", "   ", "\t"])
def test_a_blank_socket_declaration_sets_nothing_and_names_the_missing_field(blank: str) -> None:
    """An absent or whitespace-only ``nccl_socket_ifname`` declares nothing:
    neither socket variable appears in the mapping, and both announcements name
    the missing field -- NCCL's own selection is described as standing, not
    silently fallen back to.
    """
    profile = _profile_with(nccl_socket_ifname=blank, mnnvl_available=False)
    environ: dict[str, str] = {}

    announcements = apply_fabric_declaration(profile, environ)

    assert "NCCL_SOCKET_IFNAME" not in environ
    assert "GLOO_SOCKET_IFNAME" not in environ
    for variable in ("NCCL_SOCKET_IFNAME", "GLOO_SOCKET_IFNAME"):
        line = _line_for(announcements, variable)
        assert "nccl_socket_ifname" in line
        # The absence is stated outright, and so is its consequence. A line that
        # merely omitted the variable would read the same as a line reporting a
        # value that failed to apply.
        assert "declares no" in line
        assert "NCCL's own interface selection stands" in line


def test_every_variable_considered_earns_exactly_one_announcement_line() -> None:
    """The return value is a tuple with one line per variable considered, in a
    stable order, and empty is impossible: the profile fields exist, so there
    is always something to report, and a caller that printed nothing would be
    reporting a bug as success.
    """
    blank_profile = _profile_with(nccl_socket_ifname="", mnnvl_available=True)
    set_profile = _profile_with(nccl_socket_ifname="eth0", mnnvl_available=False)

    for profile, environ in (
        (set_profile, {}),
        (blank_profile, {"NCCL_SOCKET_IFNAME": "ib0", "NCCL_MNNVL_ENABLE": "1"}),
    ):
        announcements = apply_fabric_declaration(profile, environ)
        assert isinstance(announcements, tuple)
        assert announcements
        assert len(announcements) == len(_EXPECTED_VARIABLES)
        assert all(isinstance(line, str) and line for line in announcements)
        for variable, line in zip(_EXPECTED_VARIABLES, announcements, strict=True):
            assert line.startswith(variable)


def test_a_second_application_reports_its_own_writes_as_operator_choices() -> None:
    """Application is idempotent in values but not in announcements: the second
    call on the same mapping leaves every value unchanged and reports each
    previously-applied variable as set-and-left-alone -- from the function's
    inside, its own earlier write is indistinguishable from an operator's.
    """
    profile = _profile_with(nccl_socket_ifname="eth0", mnnvl_available=False)
    environ: dict[str, str] = {}

    first = apply_fabric_declaration(profile, environ)
    snapshot = dict(environ)
    second = apply_fabric_declaration(profile, environ)

    assert environ == snapshot
    assert environ == {
        "NCCL_SOCKET_IFNAME": "eth0",
        "GLOO_SOCKET_IFNAME": "eth0",
        "NCCL_MNNVL_ENABLE": "0",
    }
    for variable in ("NCCL_SOCKET_IFNAME", "GLOO_SOCKET_IFNAME", "NCCL_MNNVL_ENABLE"):
        assert "left alone" in _line_for(second, variable)
    # The first call announced an application; the second announces a no-op on
    # the same value. A log must show the transition, not two identical lines.
    assert _line_for(first, "NCCL_SOCKET_IFNAME") != _line_for(second, "NCCL_SOCKET_IFNAME")
    assert "left alone" not in _line_for(first, "NCCL_SOCKET_IFNAME")


def test_every_registered_profile_applies_without_exception_and_reports() -> None:
    """Every profile in the registry applies cleanly to a fresh mapping and
    returns a non-empty report: a sweep costs nothing and catches the failure
    mode that matters -- a newly registered profile whose field shape the
    function cannot read would surface here, not in a launcher log.
    """
    profiles = _registered_profiles()
    assert profiles, "PROFILES is empty: the sweep would vacuously pass over nothing"

    for profile in profiles:
        environ: dict[str, str] = {}
        announcements = apply_fabric_declaration(profile, environ)
        assert announcements, f"profile {profile.name!r} produced no announcements"
        assert all(isinstance(line, str) and line for line in announcements)
        # With no discovery surface, no profile's glob becomes an export.
        assert "NCCL_IB_HCA" not in environ, (
            f"profile {profile.name!r} caused NCCL_IB_HCA to be exported"
        )
        # Whatever any profile wrote, it wrote strings NCCL can read.
        assert all(isinstance(value, str) for value in environ.values())
