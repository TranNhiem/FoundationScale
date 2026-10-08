"""Environment backends for the FoundationScale agentic RL rollout plane.

WHAT IS CLAIMED HERE: the frozen declaration types every backend is built to
(``EnvSpec``, ``ExecResult``, ``EnvCapabilities``), the config/declaration
refusal :class:`EnvRefusal` and the runtime fault carrier :class:`EnvInfraError`
(whose ``kind`` the harness maps to ``abstention_reason=f"infra:{kind}"``), the
``Environment``/``EnvBackend`` structural protocols, the small name->factory
registry with the "local" port installed by default, and :func:`check_isolation`
-- the gate that must be called before a backend that runs model-chosen commands
on the HOST is ever used.

WHAT IS NOT CLAIMED: any sandboxing. The shipped ``local`` backend runs
subprocesses bare on the host (``capabilities().isolation == "none"``) and is
opted into explicitly through ``EnvSpec.allow_unsandboxed``. Container, VM and
remote backends are later slices and register through
:func:`register_env_backend` when they land. Nothing here imports torch and
nothing in it may.
"""

from __future__ import annotations

from foundationscale.agentic_rl.envs.base import (
    EnvBackend,
    EnvCapabilities,
    EnvInfraError,
    Environment,
    EnvRefusal,
    EnvSpec,
    ExecResult,
    available_env_backends,
    check_isolation,
    get_env_backend,
    register_env_backend,
)
from foundationscale.agentic_rl.envs.local import LocalEnvironment, LocalSubprocessBackend

# NOTE: "local" is registered once, lazily, at the bottom of envs/base.py (to avoid
# a circular import with this module's own import of envs.local above) -- do not
# register it again here, or the second call raises EnvRefusal for a duplicate name.

__all__ = (
    "EnvBackend",
    "EnvCapabilities",
    "EnvInfraError",
    "EnvRefusal",
    "EnvSpec",
    "Environment",
    "ExecResult",
    "LocalEnvironment",
    "LocalSubprocessBackend",
    "available_env_backends",
    "check_isolation",
    "get_env_backend",
    "register_env_backend",
)
