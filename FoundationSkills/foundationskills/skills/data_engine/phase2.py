
"""Phase-2 capabilities: declared as protocols, not implemented.

Selecting one of these op names in a pipeline spec raises
``Phase2NotImplemented("phase-2: <name>")`` from ``run_pipeline``, which the
DataEngineSkill turns into a REFUSED result naming the missing capability.
This is intentional: an agent must see the refusal instead of a silent skip.
"""
from __future__ import annotations

from typing import Iterable, Iterator, Protocol

from foundationskills.skills.data_engine.ops.base import OpStats


class Phase2NotImplemented(NotImplementedError):
    """A phase-2 op was selected; the capability does not exist yet."""


class SemanticDedup(Protocol):
    """Embedding-based near-duplicate removal (beyond MinHash)."""

    def __call__(self, records: Iterable[dict], cfg: dict, stats: OpStats) -> Iterator[dict]: ...


class SyntheticGenerator(Protocol):
    """Teacher-model synthesis of new examples (requires an inference endpoint)."""

    def __call__(self, records: Iterable[dict], cfg: dict, stats: OpStats) -> Iterator[dict]: ...


class ToolCallFormatter(Protocol):
    """Normalize tool-calling traces into the target tool-call schema."""

    def __call__(self, records: Iterable[dict], cfg: dict, stats: OpStats) -> Iterator[dict]: ...


class VideoTextIngest(Protocol):
    """Ingest video (frames + ASR transcript) into canonical text records."""

    def __call__(self, records: Iterable[dict], cfg: dict, stats: OpStats) -> Iterator[dict]: ...


PHASE2: dict[str, str] = {
    "semantic_dedup": "embedding-based semantic near-deduplication (needs an embedding backend)",
    "synthesize": "teacher-model synthetic data generation (needs an inference endpoint)",
    "toolcall_format": "tool-calling trace normalization to a target tool schema",
    "video_ingest": "video ingestion (frames + ASR transcript alignment) into text records",
}
