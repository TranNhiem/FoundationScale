"""max_grad_norm: global gradient-norm clipping before the optimizer step."""

import pytest

torch = pytest.importorskip("torch")

from foundationscale.rl.trainer import _clip_device_grads  # noqa: E402


def _params(scale):
    a, b = torch.nn.Parameter(torch.zeros(3)), torch.nn.Parameter(torch.zeros(4))
    a.grad, b.grad = torch.full((3,), scale), torch.full((4,), scale)
    return a, b


def test_large_gradients_are_clipped_to_max_norm():
    a, b = _params(10.0)
    opt = torch.optim.SGD([a, b], lr=0.1)
    pre = _clip_device_grads(opt, 1.0)
    post = torch.sqrt(a.grad.pow(2).sum() + b.grad.pow(2).sum()).item()
    assert pre == pytest.approx(10.0 * 7**0.5, rel=1e-5) and post == pytest.approx(1.0, rel=1e-4)


def test_small_gradients_are_untouched():
    a, b = _params(0.01)
    _clip_device_grads(torch.optim.SGD([a, b], lr=0.1), 1.0)
    assert torch.allclose(a.grad, torch.full((3,), 0.01))


def test_master_weight_style_optimizer_uses_device_params():
    a, b = _params(10.0)

    class Wrapper:
        device_params = [a, b]

    _clip_device_grads(Wrapper(), 2.0)
    assert torch.sqrt(a.grad.pow(2).sum() + b.grad.pow(2).sum()).item() == pytest.approx(
        2.0, rel=1e-4
    )


def test_no_grads_is_a_noop():
    p = torch.nn.Parameter(torch.zeros(2))
    assert _clip_device_grads(torch.optim.SGD([p], lr=0.1), 1.0) == 0.0


def test_non_finite_norm_skips_the_update():
    a, b = _params(1.0)
    a.grad[0] = float("nan")
    _clip_device_grads(torch.optim.SGD([a, b], lr=0.1), 1.0)
    assert a.grad is None and b.grad is None
