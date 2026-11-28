"""MasterWeightOptimizer master_device: fp32 masters on the param's device give identical updates
to host masters."""

import pytest

torch = pytest.importorskip("torch")

from foundationscale.rl.trainer import MasterWeightOptimizer  # noqa: E402


def _run(master_device):
    torch.manual_seed(0)
    p = torch.nn.Parameter(torch.randn(64, dtype=torch.bfloat16))
    opt = MasterWeightOptimizer([p], master_device=master_device, lr=1e-3)
    for _ in range(3):
        p.grad = torch.randn(64, dtype=torch.bfloat16)
        opt.step()
        opt.zero_grad()
    return p.detach().float(), opt


def test_device_masters_match_host_masters():
    a, _ = _run("cpu")
    b, opt = _run("device")
    assert torch.equal(a, b)
    assert all(
        m.device == torch.device("cpu") for m in opt.masters
    )  # params live on cpu in this test


def test_bad_master_device_is_refused():
    with pytest.raises(ValueError):
        MasterWeightOptimizer([torch.nn.Parameter(torch.zeros(2))], master_device="disk", lr=1e-3)


def test_device_masters_follow_the_parameter_device():
    # CPU stand-in for the GPU case (CI has no GPU and the skip guard forbids skips): "device"
    # places the fp32 masters on the PARAMETER's device, which on a GB200 run is the GPU.
    p = torch.nn.Parameter(torch.zeros(4, dtype=torch.bfloat16))
    opt = MasterWeightOptimizer([p], master_device="device", lr=1e-3)
    assert opt.masters[0].device == p.device
    assert opt.masters[0].dtype == torch.float32
