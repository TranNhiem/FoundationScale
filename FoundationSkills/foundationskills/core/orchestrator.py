
"""Ordered execution, confirmation gating, and journaling."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from foundationskills.core.contract import SkillContext, SkillResult
from foundationskills.core.provenance import sha256_json
from foundationskills.core.registry import SkillRegistry
from foundationskills.core.status import Status, worst


def plan_hash(plan_payload: dict[str, Any]) -> str:
    """First 16 hex chars of the canonical plan hash, used in confirmation tokens."""
    return sha256_json(plan_payload)[:16]


class ConfirmationRequired(Exception):
    """Raised when a mutating plan lacks a valid human confirmation token."""


def require_confirmation(plan_payload: dict[str, Any], token: str | None) -> None:
    expected = plan_hash(plan_payload)
    if token != expected:
        raise ConfirmationRequired(
            f"confirmation required for plan hash prefix {expected}; ask the user; never auto-fill"
        )


@dataclass(frozen=True)
class Step:
    skill: str
    request: dict[str, Any]
    allow_unmeasured: bool = False


class Orchestrator:
    """Execute skills in order, injecting matched prior artifact refs as inputs."""

    def __init__(self, registry: SkillRegistry, ctx: SkillContext) -> None:
        self.registry = registry
        self.ctx = ctx
        self._prior_refs: list[dict[str, str]] = []

    @staticmethod
    def _journal(path: Path, record: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n"
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(line)
            handle.flush()
            os.fsync(handle.fileno())

    def run(self, steps: Iterable[Step]) -> list[SkillResult]:
        results: list[SkillResult] = []
        journal_path = self.ctx.workdir / "journal.jsonl"
        for index, step in enumerate(steps):
            skill = self.registry.get(step.skill)
            request = dict(step.request)
            inputs = list(request.get("inputs", []))
            inputs.extend(ref for ref in self._prior_refs if ref["type"] in skill.consumes)
            request.setdefault("inputs", inputs)
            result = skill.execute(request, self.ctx)
            results.append(result)
            self._prior_refs.extend(a.to_dict() for a in result.artifacts)
            self._journal(
                journal_path,
                {
                    "step_index": index,
                    "skill": step.skill,
                    "status": result.status.value,
                    "exit_code": result.exit_code,
                    "artifact_ids": [a.id for a in result.artifacts],
                    "refusal": result.refusal,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
            )
            if result.status is not Status.PASS:
                if not (step.allow_unmeasured and result.status is Status.UNMEASURED):
                    break
        return results

    @staticmethod
    def final_status(results: Iterable[SkillResult]) -> Status:
        return worst(*(result.status for result in results))
