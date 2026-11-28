"""RolloutHost.async_prefetch: step k+1 is generated while the trainer works on k;
publish waits; staleness <= 1."""

from __future__ import annotations

import threading
import time

import pytest

from foundationscale.agentic_rl.rollout_host import RolloutHost


class Probe:
    """Stands in for RolloutHost.rollout_async: records which steps ran and when."""

    def __init__(self, delay=0.05, fail_at=None):
        self.calls, self.delay, self.fail_at, self.lock = [], delay, fail_at, threading.Lock()

    async def __call__(self, step):
        with self.lock:
            self.calls.append(("start", step, time.monotonic()))
        if step == self.fail_at:
            raise RuntimeError(f"boom at {step}")
        time.sleep(self.delay)
        with self.lock:
            self.calls.append(("end", step, time.monotonic()))
        return f"batch{step}"


def _host(probe, **kw):
    host = object.__new__(
        RolloutHost
    )  # bypass field validation; only the prefetch logic is under test
    object.__setattr__(host, "_state", {})
    object.__setattr__(host, "async_prefetch", kw.get("async_prefetch", True))
    object.__setattr__(host, "prefetch_limit", kw.get("prefetch_limit"))
    object.__setattr__(host, "rollout_async", probe)
    return host


def test_sync_mode_is_unchanged():
    p = Probe()
    h = _host(p, async_prefetch=False)
    assert h.rollout(0) == "batch0" and [c[1] for c in p.calls if c[0] == "start"] == [0]


def test_next_step_is_prefetched_and_returned():
    p = Probe()
    h = _host(p, prefetch_limit=3)
    assert h.rollout(0) == "batch0"
    time.sleep(0.15)  # the trainer "trains"; step 1 is generated meanwhile
    assert ("end", 1) in [(c[0], c[1]) for c in p.calls]
    assert h.rollout(1) == "batch1" and h.rollout(2) == "batch2"
    assert sorted({c[1] for c in p.calls}) == [0, 1, 2]  # no wasted step 3 (prefetch_limit)


def test_publish_waits_for_the_inflight_prefetch():
    p = Probe(delay=0.2)
    h = _host(p, prefetch_limit=5)
    h.rollout(0)
    h._await_prefetch()  # what publish() does first
    assert ("end", 1) in [(c[0], c[1]) for c in p.calls]


def test_prefetch_exception_surfaces_in_the_trainer():
    p = Probe(fail_at=1)
    h = _host(p, prefetch_limit=5)
    h.rollout(0)
    with pytest.raises(RuntimeError, match="boom at 1"):
        h.rollout(1)


def test_out_of_order_request_drains_stale_prefetch():
    p = Probe()
    h = _host(p, prefetch_limit=10)
    h.rollout(0)
    assert h.rollout(5) == "batch5"
