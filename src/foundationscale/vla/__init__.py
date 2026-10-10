"""Vision-language-action (VLA) data plane: robot episodes, modality config, norm stats.

Reads LeRobot v2 episode datasets, checks a GR00T-style ``modality.json`` against
them, and computes the per-dimension normalisation statistics a policy trains
and runs against. Everything here refuses rather than guesses: a dimension the
modality config does not cover, an episode whose frames are not contiguous, or a
statistic computed over nothing raises with the file and value named.
"""

from __future__ import annotations

from foundationscale.vla.chunking import ChunkError, ChunkSpec, valid_anchor_indices
from foundationscale.vla.frames import FrameDecodeError, decode_frames
from foundationscale.vla.lerobot import (
    SUPPORTED_CODEBASE_VERSIONS,
    EpisodeMeta,
    LeRobotDataset,
    LeRobotFormatError,
    open_lerobot,
    read_episode_columns,
)
from foundationscale.vla.modality import (
    ModalityError,
    ModalitySpec,
    SliceSpec,
    check_against_dataset,
    load_modality,
)
from foundationscale.vla.norm import (
    DEGENERATE_STD,
    STAT_NAMES,
    NormStats,
    compute_norm_stats,
    stats_digest,
    stats_from_json,
    stats_to_json,
)
from foundationscale.vla.sample import Normalizer, SampleError, VlaSample, build_sample

__all__ = [
    "ChunkError",
    "ChunkSpec",
    "FrameDecodeError",
    "Normalizer",
    "SampleError",
    "VlaSample",
    "build_sample",
    "decode_frames",
    "valid_anchor_indices",
    "DEGENERATE_STD",
    "STAT_NAMES",
    "SUPPORTED_CODEBASE_VERSIONS",
    "EpisodeMeta",
    "LeRobotDataset",
    "LeRobotFormatError",
    "ModalityError",
    "ModalitySpec",
    "NormStats",
    "SliceSpec",
    "check_against_dataset",
    "compute_norm_stats",
    "load_modality",
    "open_lerobot",
    "read_episode_columns",
    "stats_digest",
    "stats_from_json",
    "stats_to_json",
]
