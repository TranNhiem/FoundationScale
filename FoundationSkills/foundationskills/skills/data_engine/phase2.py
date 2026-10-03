
"""Phase-2 capabilities: declared as protocols, not implemented.

semantic_dedup, toolcall_format and video_ingest graduated to real ops in
ops/ (2026-10); only ``synthesize`` (superseded by llm_enhance) still refuses.

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


class SyntheticGenerator(Protocol):
    """Teacher-model synthesis of new examples (requires an inference endpoint)."""

    def __call__(self, records: Iterable[dict], cfg: dict, stats: OpStats) -> Iterator[dict]: ...


PHASE2: dict[str, str] = {
    "synthesize": "superseded by llm_enhance (modes rephrase/qa_synth/reasoning_trace/judge)",
}
