"""Agentic-RL capabilities declared for the FoundationSkills probe (A5.2).

See foundationscale.capability_manifest for the entry contract.
"""

from __future__ import annotations

from typing import Any

MANIFEST: list[dict[str, Any]] = [
    {
        "key": "agentic_rl.env_backends",
        # "local" registers on import; other backends (e.g. enroot) register only when their
        # module is imported by the run, so the measurable floor is "local" -- the probe
        # reports what is registered
        "declared": ["local"],
        "compare": "superset",
        "verify": "foundationscale.agentic_rl.envs.base:available_env_backends",
        "doc": "episode environment backends registered in this process",
    },
]
