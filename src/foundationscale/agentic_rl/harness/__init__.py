"""Harness contracts and the native tool-calling loop for the agentic RL plane.

THIS NAMESPACE is the HARNESS plane of the agentic RL slice: the sampling
declarations (:class:`base.SamplingParams`), the generation/tokenizer protocols
(:class:`base.GenerationClient`, :class:`base.ChatTokenizer`), the episode
carriers (:class:`base.EpisodeTask`, :class:`base.EpisodeBudget`,
:class:`base.EpisodeOutcome`) and the adapter contract
(:class:`base.HarnessAdapter` / :func:`base.to_trajectory`) together with the
native tool-calling loop (:class:`native_tool_loop.NativeToolLoop`).

It DEPENDS on the contract plane (``foundationscale.agentic_rl.contracts``,
``foundationscale.agentic_rl.token_trace``, ``foundationscale.agentic_rl.markup``)
and the env plane (``foundationscale.agentic_rl.envs``) and imports them -- never
the other way round. There is no engine semantic here: a ``GenerationClient`` is
told what token to sample next and nothing about what sampling means outside it.

WHAT IS CLAIMED: typed declarations for one turn and one episode, and a loop that
turns model tool-call text into environment observations without re-rendering a
sampled token.

WHAT IS NOT CLAIMED: any reward. ``EpisodeOutcome`` asserts an abstention only for
a harness fault it is certain of (an INFRA termination); the RewardService scores
later and :func:`base.to_trajectory` records whatever verdict that caller supplies.
"""

from __future__ import annotations

from foundationscale.agentic_rl.harness.base import (
    ChatTokenizer,
    EpisodeBudget,
    EpisodeOutcome,
    EpisodeTask,
    Generation,
    GenerationClient,
    HarnessAdapter,
    HarnessRefusal,
    SamplingParams,
    to_trajectory,
)
from foundationscale.agentic_rl.harness.native_tool_loop import (
    NativeToolLoop,
    NativeToolLoopRefusal,
)

__all__ = (
    "ChatTokenizer",
    "EpisodeBudget",
    "EpisodeOutcome",
    "EpisodeTask",
    "Generation",
    "GenerationClient",
    "HarnessAdapter",
    "HarnessRefusal",
    "NativeToolLoop",
    "NativeToolLoopRefusal",
    "SamplingParams",
    "to_trajectory",
)
