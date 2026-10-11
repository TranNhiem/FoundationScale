"""Isaac GR00T adapter: GR00T N1.x checkpoints read through their own published contract.

:mod:`.contract` reads a checkpoint's ``processor_config.json`` (JSON only -- the ``gr00t``
package is not imported) into the embodiment contract GR00T trained with, maps it onto
FoundationScale's ``ChunkSpec``, and builds GR00T's observation dict from FoundationScale's
data path. A new GR00T release is picked up by reading its checkpoint, not by copying config.
"""

from __future__ import annotations

from foundationscale.vla.adapters.gr00t.contract import (
    ACTION_REPS,
    Gr00tContract,
    Gr00tContractError,
    Gr00tGroup,
    build_gr00t_observation,
    check_contract_against,
    chunk_spec_from_contract,
    load_gr00t_contract,
)

__all__ = [
    "ACTION_REPS",
    "Gr00tContract",
    "Gr00tContractError",
    "Gr00tGroup",
    "build_gr00t_observation",
    "check_contract_against",
    "chunk_spec_from_contract",
    "load_gr00t_contract",
]
