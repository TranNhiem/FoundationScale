"""FoundationScale — verifiable distributed training.

One import surface for the framework. Every public name below resolves lazily
via PEP 562, mirroring :mod:`foundationscale.train`, so ``import
foundationscale`` stays torch-free: nothing heavy loads until you touch a name
that needs it. That matters because the gate plane runs on hosts where torch is
absent by design.

Doctrine in one line: a verdict is a claim about a DENOMINATOR. Gates fail
closed, and "nothing was measured" is a declared state (:data:`EXIT_UNMEASURED`,
95), never a pass.

The training entry point is reached through its own module, ``from
foundationscale.train import train`` — see the note on the export map below.
"""

from __future__ import annotations

import importlib
from typing import Any

__version__ = "0.1.0"

# Public name -> home module. This mapping IS the public surface: it is the
# denominator that tests/test_front_door.py enumerates, so a name added here
# without a working home fails a test rather than an import at a user's site.
_EXPORTS: dict[str, str] = {
    # The gate plane and its verdicts.
    "Coverage": "foundationscale.gates.core",
    "GateRegistry": "foundationscale.gates.core",
    "GateReport": "foundationscale.gates.core",
    "REGISTRY": "foundationscale.gates.core",
    "Verdict": "foundationscale.gates.core",
    "verify_controls": "foundationscale.gates.core",
    # The exit-code contract: 0 PASS / 5 RED / 95 UNMEASURED / 96 REFUSE.
    "EXIT_PASS": "foundationscale.train",
    "EXIT_RED": "foundationscale.train",
    "EXIT_REFUSE": "foundationscale.train",
    "EXIT_UNMEASURED": "foundationscale.train",
    # Training. NOTE the absence of ``train`` itself: ``foundationscale.train``
    # is a subpackage, and once imported Python binds the submodule as a real
    # attribute here — real attributes win over ``__getattr__``, so the name
    # could only ever resolve to the module. Use ``from foundationscale.train
    # import train``. test_front_door.py gates the whole collision class.
    "FoundationScaleSaveGate": "foundationscale.train",
    "TrainConfig": "foundationscale.train",
    # Topology and provenance.
    "ClusterProfile": "foundationscale.topology",
    "Topology": "foundationscale.provenance",
    "TopologyConsistency": "foundationscale.provenance",
    # Checkpoints and adapters.
    "AdapterRefusal": "foundationscale.models.adapters",
    "CheckpointFormatError": "foundationscale.checkpoint.dcp",
    "select_adapter": "foundationscale.models.adapters",
}

__all__ = [*sorted(_EXPORTS), "__version__"]


def __getattr__(name: str) -> Any:
    """Resolve a public name from its home module, on first access only."""
    home = _EXPORTS.get(name)
    if home is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(importlib.import_module(home), name)


def __dir__() -> list[str]:
    return list(__all__)
