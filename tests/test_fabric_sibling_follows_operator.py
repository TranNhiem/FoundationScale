"""#547: an operator-set socket interface is inherited by its unset sibling.

WHAT IS CLAIMED: when the operator exports exactly one of NCCL_SOCKET_IFNAME /
GLOO_SOCKET_IFNAME, apply_fabric_declaration gives the other the SAME value, not
the profile's default. Measured failure this pins: NCCL_SOCKET_IFNAME=data0 by hand,
GLOO_SOCKET_IFNAME=eth0 from the profile, every rank dead at the first gloo group.
"""

from pathlib import Path

from foundationscale.topology import apply_fabric_declaration, profile_by_name


def _apply(env: dict[str, str], tmp_path: Path) -> tuple[str, ...]:
    # Pin both host probes: an empty sysfs root (no InfiniBand devices) and no IMEX
    # daemon. Unpinned, a GB200 node adds NCCL_IB_HCA from its real mlx5 devices.
    return apply_fabric_declaration(
        profile_by_name("slurm-generic"), env, ib_sysfs_root=tmp_path, imex_running=lambda: False
    )


def test_gloo_follows_operator_nccl(tmp_path: Path) -> None:
    env = {"NCCL_SOCKET_IFNAME": "data0"}
    lines = _apply(env, tmp_path)
    assert env["GLOO_SOCKET_IFNAME"] == "data0"
    assert any("follows the operator's NCCL_SOCKET_IFNAME" in line for line in lines)


def test_nccl_follows_operator_gloo(tmp_path: Path) -> None:
    env = {"GLOO_SOCKET_IFNAME": "data0"}
    _apply(env, tmp_path)
    assert env["NCCL_SOCKET_IFNAME"] == "data0"


def test_neither_set_both_take_the_profile(tmp_path: Path) -> None:
    env: dict[str, str] = {}
    _apply(env, tmp_path)
    assert env["NCCL_SOCKET_IFNAME"] == env["GLOO_SOCKET_IFNAME"] == "eth0"


def test_both_set_both_left_alone(tmp_path: Path) -> None:
    env = {"NCCL_SOCKET_IFNAME": "data0", "GLOO_SOCKET_IFNAME": "ib0"}
    _apply(env, tmp_path)
    assert env == {
        "NCCL_SOCKET_IFNAME": "data0",
        "GLOO_SOCKET_IFNAME": "ib0",
        "NCCL_MNNVL_ENABLE": "0",
    }
