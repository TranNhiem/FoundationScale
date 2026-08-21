"""Trainer-agnostic integration: run a lifecycle event against several context families.

Why this module exists
----------------------
A single training job hands its gates very different contexts: the checkpoint gates
want a :class:`~foundationscale.gates.checkpoint_gates.CheckpointGateContext`, the
parity gate a :class:`~foundationscale.verify.parity.ParityGateContext`, objective
gates their own. :meth:`~foundationscale.gates.core.GateRegistry.run` broadcasts
one object to every gate in an event; with several families registered that
degrades into raw ``TypeError``/``AttributeError`` ERRORs one layer down, named for
nothing.

:func:`run_event` (re-exported from :mod:`foundationscale.gates.core`, where the
machinery lives) dispatches on each gate's declared
:attr:`~foundationscale.gates.core.Gate.context_type`, so one SAVE sweep runs every
family in a single report::

    report = run_event(
        REGISTRY,
        "save",
        {
            CheckpointGateContext: CheckpointGateContext.from_path(ckpt_dir),
            ParityGateContext: ParityGateContext(ckpt_dir, reference_dir),
        },
        required=["checkpoint.expert_distinctness", "checkpoint.weight_parity"],
    )
    report.raise_if_blocking()

A gate whose context type was never supplied is a blocking ERROR — *unwired, not
healthy* — never a silent omission, unless the caller explicitly declares
``missing_ctx="report-skip"``. A sweep that ends up running zero gates is VACUOUS
and blocks, exactly as it does one level down inside a single gate: the ``all([])
is True`` rule applies at every altitude.
"""

from __future__ import annotations

from .gates.core import (
    REGISTRY,
    GateBlocked,
    GateRegistry,
    GateReport,
    Lifecycle,
    run_event,
)

__all__ = [
    "REGISTRY",
    "GateBlocked",
    "GateRegistry",
    "GateReport",
    "Lifecycle",
    "run_event",
]
