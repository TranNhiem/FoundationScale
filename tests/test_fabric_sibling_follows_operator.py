"""#547: an operator-set socket interface is inherited by its unset sibling.

WHAT IS CLAIMED: when the operator exports exactly one of NCCL_SOCKET_IFNAME /
GLOO_SOCKET_IFNAME, apply_fabric_declaration gives the other the SAME value, not
the profile's default. Measured failure this pins: NCCL_SOCKET_IFNAME=data0 by hand,
GLOO_SOCKET_IFNAME=eth0 from the profile, every rank dead at the first gloo group.
"""

from foundationscale.topology import apply_fabric_declaration, profile_by_name


def test_gloo_follows_operator_nccl() -> None:
    env = {"NCCL_SOCKET_IFNAME": "data0"}
    lines = apply_fabric_declaration(profile_by_name("slurm-generic"), env)
    assert env["GLOO_SOCKET_IFNAME"] == "data0"
    assert any("follows the operator's NCCL_SOCKET_IFNAME" in line for line in lines)


def test_nccl_follows_operator_gloo() -> None:
    env = {"GLOO_SOCKET_IFNAME": "data0"}
    apply_fabric_declaration(profile_by_name("slurm-generic"), env)
    assert env["NCCL_SOCKET_IFNAME"] == "data0"


def test_neither_set_both_take_the_profile() -> None:
    env: dict[str, str] = {}
    apply_fabric_declaration(profile_by_name("slurm-generic"), env)
    assert env["NCCL_SOCKET_IFNAME"] == env["GLOO_SOCKET_IFNAME"] == "eth0"


def test_both_set_both_left_alone() -> None:
    env = {"NCCL_SOCKET_IFNAME": "data0", "GLOO_SOCKET_IFNAME": "ib0"}
    apply_fabric_declaration(profile_by_name("slurm-generic"), env)
    assert env == {
        "NCCL_SOCKET_IFNAME": "data0",
        "GLOO_SOCKET_IFNAME": "ib0",
        "NCCL_MNNVL_ENABLE": "0",
    }
