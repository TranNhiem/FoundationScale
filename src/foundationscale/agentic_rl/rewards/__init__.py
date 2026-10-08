"""FoundationScale agentic RL rewards: pluggable, torch-free reward scorers.

WHAT IS CLAIMED HERE: re-export of the music reward adapter's two public
names, ``MusicReward`` and ``MusicScore``, from
:mod:`foundationscale.agentic_rl.rewards.music`, and the ``RewardFn`` contract
plane (``RewardFn``, ``RewardVerdict``, ``MusicRewardFn``) from
:mod:`foundationscale.agentic_rl.rewards.base`. Other reward domains are
later slices and register through their own submodules, the same one-way
rule the rest of this plane uses -- this package never grows a dispatch
table that would decide what a reward IS.

WHAT IS NOT CLAIMED: any behaviour beyond re-export. Import a domain's other
names (``extract_abc``, ``do``, ``SPEC``, ...) from its own submodule.
"""

from __future__ import annotations

from foundationscale.agentic_rl.rewards.base import MusicRewardFn, RewardFn, RewardVerdict
from foundationscale.agentic_rl.rewards.music import MusicReward, MusicScore

__all__ = (
    "MusicReward",
    "MusicRewardFn",
    "MusicScore",
    "RewardFn",
    "RewardVerdict",
)
