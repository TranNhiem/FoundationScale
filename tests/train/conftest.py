"""Directory-scoped fixtures for the training-plane suite.

This file exists to take a process-global out of the verdict.

``transformers.TrainingArguments`` resolves its device in ``_setup_devices``,
and the branch that selects MPS is ``elif is_torch_mps_available():`` -- a
helper in ``transformers.utils.import_utils`` decorated with
``functools.lru_cache``. The CUDA, XPU and NPU branches are the same shape.
So the probe is not consulted once per construction: it is consulted once per
PROCESS, and whichever caller reaches it first freezes the answer for every
caller after it. Patching ``torch.backends.mps.is_available`` after that point
changes nothing, because nothing calls it again.

``test_train_execution.py``'s autouse fixture patches exactly those torch
probes, and its control leg ``test_fixture_pins_execution_to_cpu`` asserts the
resolved device is CPU. Run alone, the module is the first caller and the
cache stores its patched ``False`` -- green. Run alongside the declared-axis
modules added for finding #373, which construct real ``TrainingArguments``
without any pin and sort before ``test_train_*`` alphabetically, the cache is
already ``True`` before the fixture ever runs, and the leg fails with
``expected CPU placement, resolved mps``. Same tree, same code, two answers
decided by collection order: the #289 class.

MEASURED, not assumed. In one process on an Apple-silicon host: an unpatched
``TrainingArguments`` resolves ``mps``; patching the torch probes and resetting
accelerate's ``PartialState`` still resolves ``mps``; patching the torch probes
and additionally clearing the import-probe caches resolves ``cpu``. The
accelerate reset alone is insufficient, which is why the cache clear is here
and why the docstring does not stop at the singleton the first diagnosis
blamed.

The fix belongs in this conftest rather than in either module. Putting it in
``test_train_execution.py`` would only re-order the same race, and putting it
in the new modules would make every future module in this directory
responsible for a hazard it did not create.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest


def _clear_device_probe_caches() -> None:
    """Re-open every memoised import/device probe so the next call re-measures.

    Every ``lru_cache`` in the two probe modules is cleared, rather than an
    allowlist of the handful that decide device placement today. An allowlist
    would be correct now and stale the next time transformers moves a branch
    behind a different helper -- the same reasoning that made
    ``_false_probe_like`` copy the original's shape instead of naming
    ``__wrapped__`` as a special case. These are import and capability probes
    (``find_spec`` calls and ``torch.*.is_available()`` reads); clearing them
    costs one re-probe per test and buys an order-independent verdict.

    Both modules are imported INSIDE the function. This conftest is collected
    for every test in the directory, including ones written to prove they need
    no torch, and a module-scope import would put torch on their import path
    (#354). ``ImportError`` is tolerated for the same reason: a torch-free
    invocation has nothing to clear, and a conftest that exploded there would
    trade one environment-dependent outcome for another.
    """
    modules = []
    try:
        import transformers.utils.import_utils as transformers_probes

        modules.append(transformers_probes)
    except ImportError:  # pragma: no cover -- torch-free legs never reach here
        pass
    try:
        import accelerate.utils.imports as accelerate_probes

        modules.append(accelerate_probes)
    except ImportError:  # pragma: no cover -- torch-free legs never reach here
        pass
    for module in modules:
        for obj in list(vars(module).values()):
            cache_clear = getattr(obj, "cache_clear", None)
            if callable(cache_clear):
                cache_clear()


def _reset_accelerate_state() -> None:
    """Drop accelerate's shared singletons so the next resolution is fresh.

    Not sufficient on its own -- the measurement above shows a reset alone
    still resolves ``mps`` -- but not redundant either: ``PartialState`` caches
    the resolved ``device`` object itself in a class-level shared dict, so
    without the reset a stale device survives a correctly re-measured probe.
    The two together are what make the resolution a function of the fixtures
    in scope rather than of collection order.

    ``_reset_state`` is accelerate's own API for this, so the reset stays
    correct if the internal layout of the shared-state dict changes.
    """
    try:
        from accelerate.state import AcceleratorState, PartialState
    except ImportError:  # pragma: no cover -- torch-free legs never reach here
        return
    AcceleratorState._reset_state(reset_partial_state=True)
    PartialState._reset_state()


@pytest.fixture(autouse=True)
def _isolate_device_resolution() -> Iterator[None]:
    """Re-open device resolution around EVERY training-plane test.

    Both directions are deliberate, and they close different holes.

    Resetting on the way IN means a test resolves the device under the
    fixtures actually in scope for it, rather than inheriting an answer cached
    before those fixtures existed. That is what makes the run green.

    Resetting on the way OUT is what keeps this fixture from creating the
    mirror-image defect. ``test_train_execution.py``'s pin is a ``monkeypatch``
    of the torch probes, so at teardown the real probes come back -- but the
    caches would still hold the ``False`` they learned while patched, and a
    later module would read a CPU verdict produced by a fixture that no longer
    applies to it. Conftest fixtures tear down after module fixtures, so by the
    time this clear runs the real probes are restored and what gets re-cached
    is the truth about the host.
    """
    _clear_device_probe_caches()
    _reset_accelerate_state()
    yield
    _clear_device_probe_caches()
    _reset_accelerate_state()
