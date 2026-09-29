"""Eval policy: which benchmarks are contractual, and the one-sided regression band.

A benchmark regresses when ``drop > max(abs_epsilon, k * sqrt(se_b**2 + se_c**2))``.
Improvements never fail. A metric without a measured stderr is only comparable
when its policy entry names an explicit ``abs_epsilon``; otherwise the policy is
invalid for that result (a :class:`PolicyError`, which the skill refuses on).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from foundationskills.core.provenance import sha256_json


class PolicyError(ValueError):
    """The policy file, or a result judged against it, cannot yield a verdict."""


@dataclass(frozen=True)
class TaskPolicy:
    name: str
    metric: str
    higher_is_better: bool = True
    k: float = 2.0
    abs_epsilon: float = 0.01
    explicit_epsilon: bool = False
    num_fewshot: int | None = None
    dataset: str | None = None
    judge: bool = False
    fewshot_as_multiturn: bool = False
    system_instruction: str | None = None
    trust_remote_code: bool = False


@dataclass(frozen=True)
class Policy:
    version: int
    tasks: dict[str, TaskPolicy]
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def fingerprint(self) -> str:
        return sha256_json(self.raw)

    def task(self, name: str) -> TaskPolicy:
        if name not in self.tasks:
            raise PolicyError(f"missing input: policy entry for benchmark {name!r}")
        return self.tasks[name]


def _number(value: Any, what: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
        raise PolicyError(f"precondition failed: {what} must be a non-negative number, got {value!r}")
    return float(value)


def parse_policy(data: Any) -> Policy:
    if not isinstance(data, dict):
        raise PolicyError("precondition failed: eval policy is not a mapping")
    version = data.get("policy_version")
    if isinstance(version, bool) or not isinstance(version, int):
        raise PolicyError("precondition failed: eval policy needs an integer policy_version")
    default = data.get("default") or {}
    if not isinstance(default, dict):
        raise PolicyError("precondition failed: eval policy 'default' is not a mapping")
    k = _number(default.get("k", 2.0), "default.k")
    eps = _number(default.get("abs_epsilon", 0.01), "default.abs_epsilon")
    tasks_raw = data.get("tasks")
    if not isinstance(tasks_raw, dict) or not tasks_raw:
        raise PolicyError("precondition failed: eval policy has no tasks")
    tasks: dict[str, TaskPolicy] = {}
    for name, entry in tasks_raw.items():
        if not isinstance(entry, dict) or not isinstance(entry.get("metric"), str) or not entry["metric"]:
            raise PolicyError(f"precondition failed: policy task {name!r} needs a metric")
        fewshot = entry.get("num_fewshot")
        if fewshot is not None and (isinstance(fewshot, bool) or not isinstance(fewshot, int) or fewshot < 0):
            raise PolicyError(f"precondition failed: policy task {name!r} num_fewshot must be a non-negative int")
        tasks[str(name)] = TaskPolicy(
            name=str(name),
            metric=entry["metric"],
            higher_is_better=bool(entry.get("higher_is_better", True)),
            k=_number(entry.get("k", k), f"tasks.{name}.k"),
            abs_epsilon=_number(entry.get("abs_epsilon", eps), f"tasks.{name}.abs_epsilon"),
            explicit_epsilon="abs_epsilon" in entry,
            num_fewshot=fewshot,
            dataset=entry.get("dataset"),
            judge=bool(entry.get("judge", False)),
            fewshot_as_multiturn=bool(entry.get("fewshot_as_multiturn", False)),
            system_instruction=entry.get("system_instruction"),
            trust_remote_code=bool(entry.get("trust_remote_code", False)),
        )
    return Policy(version=version, tasks=tasks, raw=data)


def load_policy(path: str | Path) -> Policy:
    source = Path(path)
    if not source.is_file():
        raise PolicyError(f"missing input: eval policy file {source}")
    try:
        with source.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
    except yaml.YAMLError as exc:
        raise PolicyError(f"precondition failed: eval policy {source} is not valid YAML ({exc})") from exc
    return parse_policy(data)


@dataclass(frozen=True)
class Comparison:
    breach: bool
    drop: float
    threshold: float
    band: str  # "stderr" or "abs_epsilon"


def _stderr(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or math.isnan(value):
        return None
    return float(value)


def compare(task: TaskPolicy, score: float, baseline: float, se_c: Any = None, se_b: Any = None) -> Comparison:
    """One-sided regression test of ``score`` (checkpoint) against ``baseline``."""
    drop = (baseline - score) if task.higher_is_better else (score - baseline)
    sc, sb = _stderr(se_c), _stderr(se_b)
    if sc is None or sb is None:
        if not task.explicit_epsilon:
            raise PolicyError(
                f"precondition failed: benchmark {task.name!r} metric {task.metric!r} has no measured stderr "
                "and its policy entry sets no explicit abs_epsilon"
            )
        return Comparison(drop > task.abs_epsilon, drop, task.abs_epsilon, "abs_epsilon")
    noise = task.k * math.sqrt(sb * sb + sc * sc)
    threshold = max(task.abs_epsilon, noise)
    return Comparison(drop > threshold, drop, threshold, "stderr" if noise >= task.abs_epsilon else "abs_epsilon")
