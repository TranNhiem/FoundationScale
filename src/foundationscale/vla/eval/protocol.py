"""The protocol every policy adapter of the FS evaluation harness implements.

FS owns the episode plan, the rollout loop, the success scoring and the result record of an
evaluation; a backend is reached only through this protocol, never through its own modules.
``vla/adapters/openpi/libero.py`` and ``vla/adapters/gr00t/libero.py`` implement it over the
backends' public serving protocols -- the openpi websocket policy server, the GR00T policy server
-- and import their client packages function-locally, so neither the harness nor the core install
carries a backend.

An adapter names itself, declares how many actions one inference buys, drops its episode state
between episodes, and turns one raw LIBERO observation plus the task text into an env-ready action
chunk ``[k, 7]``. The runner executes the chunk's first ``replan_steps`` rows and then infers
again. It refuses a chunk with fewer than ``replan_steps`` rows or a width other than 7, and it
records an adapter that raises, in both cases as a failed episode carrying the refusal as its
error. A short chunk would leave the runner to invent actions the policy never chose or to infer
again mid-chunk, either of which changes the cadence the adapter declared; a width other than 7 is
not LIBERO's ``[dx, dy, dz, droll, dpitch, dyaw, gripper]`` action and cannot be handed to the
env. A refused or crashed episode still counts as a trial: the denominator of a report never
shrinks.

Nothing here is guessed and nothing is imported at runtime: this module is a typing contract, so
numpy is named only in annotations, under ``TYPE_CHECKING``, and the core install needs none of it.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    import numpy as np

__all__ = ["PolicyAdapter"]


@runtime_checkable
class PolicyAdapter(Protocol):
    """One policy the evaluation harness rolls out, behind its own serving protocol.

    The contract is structural: any object carrying these four members is an adapter, and
    ``runtime_checkable`` lets a caller fail closed with ``isinstance`` before it trusts one. Such
    a check sees the members, not their values, so ``replan_steps >= 1`` and the chunk's
    ``[k >= replan_steps, 7]`` shape are refused where they are used, by the runner.

    ``name`` is the policy's identity and is carried verbatim into the report's ``policy`` field;
    it is never parsed, rewritten or defaulted. ``replan_steps`` is how many rows of every chunk
    the runner executes per inference and must be at least 1: a cadence of 0 would enqueue no
    action and spin the rollout without ever stepping the env.
    """

    name: str
    # The policy's identity, carried verbatim into the report.

    replan_steps: int
    # Rows of each chunk executed per inference; an adapter declares at least 1.

    def reset(self) -> None:
        """Drop everything one episode leaves behind, before the next episode starts.

        Called once per episode, before the env's first reset. An action queue, a backend's episode
        state or a cache carried across episodes would answer trial ``t`` with trial ``t - 1``'s
        leftovers, so the trials of a task would not be independent of each other.
        """
        ...

    def act(self, raw_obs: Mapping[str, Any], task_text: str) -> np.ndarray:
        """One env-ready action chunk ``[k >= replan_steps, 7]`` for ``raw_obs`` and ``task_text``.

        ``raw_obs`` is the LIBERO observation exactly as the env returned it -- the adapter picks
        the camera images and the state dims its backend encodes -- and ``task_text`` is the task's
        language instruction, sent as the prompt. Each row is one env action
        ``[dx, dy, dz, droll, dpitch, dyaw, gripper]`` with the gripper in ``[-1, 1]``, so the
        runner enqueues the first ``replan_steps`` rows and steps the env with them unchanged.

        A chunk with fewer than ``replan_steps`` rows or a width other than 7 is refused by the
        runner and the episode is recorded as a failure carrying that refusal (see the module
        docstring); an adapter that cannot answer raises instead, and the runner records that
        exception the same way. Either way the episode is a trial and a failure.
        """
        ...
