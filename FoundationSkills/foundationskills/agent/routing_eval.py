"""Routing evaluation for FoundationSkills skill packages.

This module measures discoverability: whether an LLM agent routes a user request to the
right skill package when the only context it receives is the skill index built from each
package's ``SKILL.md`` frontmatter. It is the FoundationSkills version of NVSkills-Eval's
discoverability tier. Cases come from every package's ``evals/evals.json`` and one run uses
all packages' cases together, so negative cases (``expected_skill: null``) test that the
index does not over-trigger.

Ablation arms (:data:`ARMS`) show what the frontmatter adds:

* ``full`` -- name + description + ``when_to_use``; this is the gated arm (:data:`GATED_ARM`).
* ``description`` -- name + description only.
* ``names_only`` -- names only, the index before frontmatter existed.

Uplift is ``accuracy(full) - accuracy(names_only)`` (and versus ``description``).

Measurement policy
------------------
A call is *measured* only when the model answered and the answer parsed to the documented
JSON object with a ``skill`` key. An LLM error, an empty answer, unparseable text or a
missing ``skill`` key is UNMEASURED: a check that did not run is never PASS. A well-formed
answer naming a skill outside the index is measured and simply wrong (``unknown skill <x>``).
A null, empty, ``none`` or ``null`` answer means "no listed skill applies". Calls run at a
fixed ``temperature`` with ``seed=rep`` and JSON mode. Only the gated arm decides the verdict: its
accuracy and every package's trigger recall must reach ``threshold``; the other arms are
informational (uplift). A case verdict is the majority vote over reps; a tie has no majority.

Exit codes follow the repo doctrine: ``0`` PASS, ``5`` RED, ``95`` UNMEASURED. Refusals raise
:class:`RoutingEvalRefused` whose message names the missing input or precondition, and
:func:`write_benchmarks` refuses an unmeasured run because an unmeasured run must not become
a published benchmark. Reports are written as atomic JSON.

Secrets: this module never prints or stores an API key. The backend reads the key from the
environment variable named by ``api_key_env`` and only hands it to the endpoint.
"""
from __future__ import annotations

import hashlib
import importlib.resources
import json
import os
import sys
from collections import Counter
from collections.abc import Iterable, Mapping
from datetime import datetime, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

import yaml

from foundationskills.core import SkillRegistry
from foundationskills.core.artifacts import _atomic_write_json
from foundationskills.skills import register_builtin_skills
from foundationskills.skills.data_engine.llm_backend import (
    LLMBackend,
    LLMOpError,
    LLMResponse,
    make_backend,
    map_ordered,
    parse_json_object,
)

ARMS: tuple[str, ...] = ("full", "description", "names_only")
GATED_ARM = "full"
SCHEMA = "fskills.routing_eval/1"

EXIT_PASS = 0
EXIT_RED = 5
EXIT_UNMEASURED = 95


class RoutingEvalRefused(ValueError):
    """The routing eval refused to run; the message names the missing input."""


# --- index loading ---------------------------------------------------------------------


def load_index(registry: SkillRegistry | None = None) -> list[dict]:
    """Return one frontmatter/evals entry per skill package, sorted by name.

    Refuses when no package is registered, when a package has no SKILL.md frontmatter,
    no evals/evals.json or no eval cases.
    """
    if registry is None:
        registry = SkillRegistry()
        register_builtin_skills(registry)
    by_package: dict[str, list[str]] = {}
    for skill_name in registry.names():
        by_package.setdefault(_skill_package(registry.get(skill_name)), []).append(str(skill_name))
    if not by_package:
        raise RoutingEvalRefused("load_index: no skill packages are registered")
    index = [_read_package(package) for package in sorted(by_package)]
    index.sort(key=lambda entry: (str(entry["name"]), str(entry["package"])))
    return index


def _skill_package(skill: object) -> str:
    module = sys.modules.get(type(skill).__module__)
    package = getattr(module, "__package__", None)
    if not package:
        raise RoutingEvalRefused(f"load_index: class {type(skill).__qualname__} has no package to load files from")
    return str(package)


def _read_package(package: str) -> dict:
    root = importlib.resources.files(package)
    skill_md = root.joinpath("SKILL.md")
    if not skill_md.is_file():
        raise RoutingEvalRefused(f"load_index: {package}: SKILL.md is missing")
    md_bytes = skill_md.read_bytes()
    front = _parse_frontmatter(md_bytes.decode("utf-8"), package)
    evals_path = root.joinpath("evals", "evals.json")
    if not evals_path.is_file():
        raise RoutingEvalRefused(f"load_index: {package}: evals/evals.json is missing")
    evals_bytes = evals_path.read_bytes()
    return {
        "package": package,
        "name": front["name"],
        "description": front["description"],
        "when_to_use": front["when_to_use"],
        "evals": _parse_evals(evals_bytes, package),
        "skill_md_sha256": hashlib.sha256(md_bytes).hexdigest(),
        "evals_sha256": hashlib.sha256(evals_bytes).hexdigest(),
    }


def _parse_frontmatter(text: str, package: str) -> dict[str, Any]:
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        raise RoutingEvalRefused(f"load_index: {package}: SKILL.md has no frontmatter block")
    closing = next((i for i in range(1, len(lines)) if lines[i].strip() == "---"), None)
    if closing is None:
        raise RoutingEvalRefused(f"load_index: {package}: SKILL.md frontmatter is not terminated by ---")
    try:
        data = yaml.safe_load("\n".join(lines[1:closing]))
    except yaml.YAMLError as exc:
        raise RoutingEvalRefused(f"load_index: {package}: SKILL.md frontmatter is not valid YAML ({exc})") from exc
    if not isinstance(data, dict):
        raise RoutingEvalRefused(f"load_index: {package}: SKILL.md frontmatter is not a YAML mapping")
    name = data.get("name")
    description = data.get("description")
    when = data.get("when_to_use")
    if not isinstance(name, str) or not name.strip():
        raise RoutingEvalRefused(f"load_index: {package}: SKILL.md frontmatter has no name")
    if not isinstance(description, str):
        raise RoutingEvalRefused(f"load_index: {package}: SKILL.md frontmatter has no description")
    if not isinstance(when, list) or not all(isinstance(rule, str) for rule in when):
        raise RoutingEvalRefused(f"load_index: {package}: SKILL.md frontmatter has no when_to_use list of strings")
    return {"name": name.strip(), "description": description, "when_to_use": list(when)}


def _parse_evals(data: bytes, package: str) -> list[dict]:
    try:
        raw = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RoutingEvalRefused(f"load_index: {package}: evals/evals.json is not valid JSON ({exc})") from exc
    if not isinstance(raw, list) or not raw:
        raise RoutingEvalRefused(f"load_index: {package}: evals/evals.json has no eval cases")
    cases: list[dict] = []
    for item in raw:
        if not isinstance(item, dict) or not all(key in item for key in ("id", "question", "expected_skill")):
            raise RoutingEvalRefused(f"load_index: {package}: an eval case lacks id/question/expected_skill")
        if item["expected_skill"] is not None and not isinstance(item["expected_skill"], str):
            raise RoutingEvalRefused(f"load_index: {package}: eval case {item['id']!r} has a non-string expected_skill")
        cases.append(dict(item))
    return cases


# --- index rendering and prompting ------------------------------------------------------


def render_index(index: list[dict], arm: str) -> str:
    """Render the skill index for one ablation arm as deterministic plain text."""
    _require_arm(arm)
    blocks: list[str] = []
    for entry in index:
        name = f"- {entry['name']}"
        if arm == "names_only":
            blocks.append(name)
        elif arm == "description":
            blocks.append(f"{name}\n  description: {_one_line(entry['description'])}")
        else:
            lines = [name, f"  description: {_one_line(entry['description'])}"]
            for rule in entry["when_to_use"]:
                lines.append(f"  when_to_use: {_one_line(rule)}")
            blocks.append("\n".join(lines))
    return "\n".join(blocks) + "\n"


def _one_line(text: Any) -> str:
    return " ".join(str(text).split())


def build_messages(index: list[dict], arm: str, question: str) -> list[dict]:
    """Build the system/user message pair that asks for one routing decision."""
    system = (
        "You route user requests to FoundationSkills skill packages.\n\n"
        "Skill index:\n"
        f"{render_index(index, arm)}"
        "\nReply with ONLY a JSON object {\"skill\": \"<one name from the list>\"} when one listed skill applies,\n"
        "or {\"skill\": null} when no listed skill applies. Reply with nothing but that JSON object."
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": str(question)}]


# --- the run ----------------------------------------------------------------------------


def run_routing_eval(
    index: list[dict],
    backend: LLMBackend,
    *,
    arms: Iterable[str] = ARMS,
    reps: int = 3,
    workers: int = 4,
    temperature: float = 0.6,
    max_tokens: int = 2048,
    threshold: float = 0.9,
    created: datetime | str | None = None,
) -> dict:
    """Run the routing ablation over the index and return the report dictionary."""
    arm_list: tuple[str, ...] = (arms,) if isinstance(arms, str) else tuple(arms)
    _check_run_args(arm_list, reps, workers, threshold)
    endpoint = _resolve_backend(backend)
    names = sorted({str(entry["name"]) for entry in index})
    known = set(names)
    jobs = [
        (arm, entry, case, rep)
        for arm in arm_list
        for entry in index
        for case in entry["evals"]
        for rep in range(reps)
    ]

    def run_one(job: tuple[str, dict, dict, int]) -> dict:
        arm, entry, case, rep = job
        response = endpoint.complete(
            build_messages(index, arm, str(case["question"])),
            temperature=temperature,
            max_tokens=max_tokens,
            seed=rep,
            json_mode=True,
        )
        return _record_call(response, arm, entry, case, rep, known)

    calls = list(map_ordered(run_one, jobs, int(workers)))
    results = {arm: _score_arm(calls, arm, names) for arm in arm_list}
    verdict, exit_code, reasons = _verdict(calls, results, float(threshold))
    return {
        "schema": SCHEMA,
        "created": _stamp(created),
        "model": endpoint.model,
        "backend_kind": endpoint.kind,
        "arms": list(arm_list),
        "reps": reps,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "threshold": threshold,
        "gated_arm": GATED_ARM,
        "index": [_index_view(entry) for entry in index],
        "index_sha256": _index_sha256(index),
        "results": results,
        "uplift": _uplift(results, arm_list),
        "cases": _case_records(calls, index, arm_list),
        "calls": calls,
        "verdict": verdict,
        "exit_code": exit_code,
        "reasons": reasons,
    }


def _check_run_args(arms: tuple[str, ...], reps: int, workers: int, threshold: float) -> None:
    if isinstance(reps, bool) or not isinstance(reps, int) or reps < 1:
        raise RoutingEvalRefused(f"run_routing_eval: reps must be an integer >= 1, got {reps!r}")
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
        raise RoutingEvalRefused(f"run_routing_eval: workers must be an integer >= 1, got {workers!r}")
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or not 0.0 < threshold <= 1.0:
        raise RoutingEvalRefused(f"run_routing_eval: threshold must be in (0, 1], got {threshold!r}")
    for arm in arms:
        _require_arm(arm, "run_routing_eval")
    if GATED_ARM not in arms:
        raise RoutingEvalRefused(f"run_routing_eval: gated arm {GATED_ARM!r} is required in arms {list(arms)!r}")


def _require_arm(arm: str, where: str = "render_index") -> None:
    if arm not in ARMS:
        raise RoutingEvalRefused(f"{where}: unknown arm {arm!r}; expected one of {', '.join(ARMS)}")


def _resolve_backend(backend: LLMBackend | Mapping[str, Any]) -> LLMBackend:
    if isinstance(backend, Mapping):
        try:
            return make_backend(dict(backend), "routing_eval")
        except LLMOpError as exc:
            raise RoutingEvalRefused(f"run_routing_eval: backend configuration refused: {exc}") from exc
    return backend


def _record_call(response: LLMResponse, arm: str, entry: dict, case: dict, rep: int, known: set[str]) -> dict:
    expected = _expected_of(case)
    row: dict[str, Any] = {
        "arm": arm,
        "case_id": case["id"],
        "package": entry["name"],
        "rep": rep,
        "expected": expected,
        "predicted": None,
        "correct": False,
        "measured": False,
        "error": None,
        "completion_tokens": int(response.completion_tokens),
    }
    if response.error:
        row["error"] = response.error
        return row
    if response.content is None:
        row["error"] = "empty response content"
        return row
    parsed = parse_json_object(response.content)
    if not isinstance(parsed, dict) or "skill" not in parsed:
        row["error"] = "unparseable: " + response.content[:80]
        return row
    predicted, error = _normalise_skill(parsed["skill"], known)
    row["measured"] = True
    row["predicted"] = predicted
    row["error"] = error
    row["correct"] = bool(error is None and predicted == expected)
    return row


def _expected_of(case: dict) -> str | None:
    value = case.get("expected_skill")
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _normalise_skill(value: Any, known: set[str]) -> tuple[str | None, str | None]:
    """Return (predicted, error); an unknown name is measured and simply wrong."""
    if value is None:
        return None, None
    if not isinstance(value, str):
        return None, "skill field is not a string"
    text = value.strip()
    if text.lower() in ("", "none", "null"):
        return None, None
    if text not in known:
        return text, f"unknown skill {text}"
    return text, None


def _score_arm(calls: list[dict], arm: str, names: list[str]) -> dict:
    rows = [c for c in calls if c["arm"] == arm]
    measured = [c for c in rows if c["measured"]]
    correct = sum(1 for row in measured if row["correct"])
    return {
        "calls": len(rows),
        "measured": len(measured),
        "unmeasured": len(rows) - len(measured),
        "correct": correct,
        "accuracy": correct / len(measured) if measured else None,
        "packages": _package_stats(rows, names),
    }


def _package_stats(rows: list[dict], names: list[str]) -> dict[str, dict]:
    measured = [row for row in rows if row["measured"]]
    stats: dict[str, dict] = {}
    for name in names:
        hits = sum(1 for row in measured if row["expected"] == name and row["predicted"] == name)
        positives = sum(1 for row in measured if row["expected"] == name) or 0
        predicted = sum(1 for row in measured if row["predicted"] == name) or 0
        stats[name] = {
            "trigger_recall": hits / positives if positives else None,
            "trigger_precision": hits / predicted if predicted else None,
            "false_triggers": predicted - hits,
        }
    return stats


def _case_records(calls: list[dict], index: list[dict], arms: tuple[str, ...]) -> list[dict]:
    rows: list[dict] = []
    for arm in arms:
        for entry in index:
            for case in entry["evals"]:
                selected = [
                    c for c in calls
                    if c["arm"] == arm and c["package"] == entry["name"] and c["case_id"] == case["id"]
                ]
                rows.append(_majority_record(arm, case, selected))
    return rows


def _majority_record(arm: str, case: dict, rows: list[dict]) -> dict:
    expected = _expected_of(case)
    measured = [row for row in rows if row["measured"]]
    counts = Counter(_vote_key(row["predicted"]) for row in measured)
    ordered = dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))
    majority: bool | None = None
    if ordered:
        (top_key, top_count), *rest = ordered.items()
        # A tie has no majority; that is reported as undecidable rather than guessed.
        if not rest or top_count > max(count for _, count in rest):
            majority = top_key == (expected if expected is not None else "null")
    return {
        "case_id": case["id"],
        "arm": arm,
        "expected": expected,
        "votes": ordered,
        "majority_correct": majority,
    }


def _vote_key(predicted: str | None) -> str:
    return "null" if predicted is None else str(predicted)


def _uplift(results: dict[str, dict], arms: tuple[str, ...]) -> dict:

    def difference(left: str, right: str) -> float | None:
        if left not in arms or right not in arms:
            return None
        first, second = results[left]["accuracy"], results[right]["accuracy"]
        if first is None or second is None:
            return None
        return float(first) - float(second)

    return {
        "full_minus_names_only": difference("full", "names_only"),
        "full_minus_description": difference("full", "description"),
    }


def _verdict(calls: list[dict], results: dict[str, dict], threshold: float) -> tuple[str, int, list[str]]:
    gated = [c for c in calls if c["arm"] == GATED_ARM]
    if not gated:
        return "unmeasured", EXIT_UNMEASURED, [f"verdict: the gated arm {GATED_ARM!r} made no calls to measure"]
    missing = sum(1 for row in gated if not row["measured"])
    if missing:
        reason = [f"verdict: {missing} of {len(gated)} {GATED_ARM!r} calls were unmeasured; "
                  "an unmeasured run is never PASS"]
        return "unmeasured", EXIT_UNMEASURED, reason
    score = results.get(GATED_ARM) or {}
    accuracy = score.get("accuracy")
    if accuracy is None:
        return "unmeasured", EXIT_UNMEASURED, [f"verdict: the gated arm {GATED_ARM!r} has no measured calls"]
    reasons: list[str] = []
    red = False
    if accuracy < threshold:
        red = True
        reasons.append(f"verdict: {GATED_ARM!r} accuracy {_pct(accuracy)} is under the gate {_pct(threshold)}")
    for name, stats in sorted((score.get("packages") or {}).items()):
        recall = stats.get("trigger_recall")
        if recall is None:
            reasons.append(f"verdict: package {name} has no measured trigger case in the {GATED_ARM!r} arm")
            return "unmeasured", EXIT_UNMEASURED, reasons
        if recall < threshold:
            red = True
            reasons.append(f"verdict: package {name} trigger recall {_pct(recall)} is under the gate {_pct(threshold)}")
    return ("red", EXIT_RED, reasons) if red else ("pass", EXIT_PASS, [])


# --- report output ----------------------------------------------------------------------


def write_report(report: dict, path: Path) -> None:
    """Write the routing eval report as atomic JSON."""
    _atomic_write_json(Path(path), report)


def render_benchmark_md(report: dict, package_name: str) -> str:
    """Render the BENCHMARK.md body for one skill package from a run report."""
    entry, name = _report_entry(report, package_name)
    results: dict[str, dict] = report.get("results") or {}
    lines: list[str] = [
        f"# Routing evaluation: {name}",
        "",
        f"Generated by `fskills routing-eval` against skill index {_short_hash(report)}"
        f" and eval cases {str(entry.get('evals_sha256') or '')[:12]}; do not edit by hand.",
        "",
        "## Summary",
        "",
        f"- Verdict: {report.get('verdict')}",
        f"- Model: {report.get('model')}",
        f"- Created: {report.get('created')}",
        f"- Reps: {report.get('reps')}",
        f"- Temperature: {report.get('temperature')}",
        f"- Threshold: {_pct(report.get('threshold'))}",
        f"- Cases: {entry.get('cases')}",
        "",
        "## Results",
        "",
        "| arm | accuracy (all packages) | trigger recall | trigger precision | false triggers |",
        "| --- | --- | --- | --- | --- |",
    ]
    for arm in report.get("arms") or []:
        score = results.get(arm) or {}
        stats = (score.get("packages") or {}).get(name) or {}
        lines.append(f"| {arm} | {_pct(score.get('accuracy'))} | {_pct(stats.get('trigger_recall'))} "
                     f"| {_pct(stats.get('trigger_precision'))} | {stats.get('false_triggers', 0)} |")
    uplift = report.get("uplift") or {}
    lines += [
        "",
        "## Uplift",
        "",
        f"- full vs names_only: {_pct_delta(uplift.get('full_minus_names_only'))}",
        f"- full vs description: {_pct_delta(uplift.get('full_minus_description'))}",
        "",
        "names_only is the index before frontmatter existed: it lists skill package names and nothing else.",
    ]
    lines += ["", "## Cases", "", "| id | expected | votes | majority correct |", "| --- | --- | --- | --- |"]
    for record in _package_case_records(report, name):
        votes = ", ".join(f"{key}: {count}" for key, count in (record.get("votes") or {}).items()) or "n/a"
        expected = record.get("expected")
        expected_cell = expected if expected is not None else "null"
        majority = _flag(record.get("majority_correct"))
        lines.append(f"| {record.get('case_id')} | {expected_cell} | {votes} | {majority} |")
    lines += [
        "",
        "## Caveats",
        "",
        "- Small n: each package contributes a handful of eval cases.",
        "- LLM nondeterminism: repeated calls can differ even at a fixed temperature and seed.",
        "- Routing only: this measures skill selection, not task behaviour.",
        "- Measured against one model: results do not transfer to other models.",
    ]
    return "\n".join(lines) + "\n"


def _report_entry(report: dict, package_name: str) -> tuple[dict, str]:
    wanted = str(package_name)
    for entry in report.get("index") or []:
        if wanted in (str(entry.get("name")), str(entry.get("package"))):
            return entry, str(entry.get("name"))
    raise RoutingEvalRefused(f"render_benchmark_md: {wanted} is not in the report index")


def _package_case_records(report: dict, name: str) -> list[dict]:
    gated = [c for c in report.get("calls") or [] if c.get("arm") == GATED_ARM and c.get("package") == name]
    ordered_ids: list[str] = []
    for row in gated:
        case_id = str(row["case_id"])
        if case_id not in ordered_ids:
            ordered_ids.append(case_id)
    wanted = set(ordered_ids)
    by_id = {str(r["case_id"]): r for r in report.get("cases") or []
             if r.get("arm") == GATED_ARM and str(r["case_id"]) in wanted}
    return [by_id[case_id] for case_id in ordered_ids if case_id in by_id]


def write_benchmarks(report: dict, index: list[dict]) -> list[Path]:
    """Write one BENCHMARK.md per package next to its SKILL.md; refuse unmeasured runs."""
    if report.get("verdict") == "unmeasured":
        raise RoutingEvalRefused("write_benchmarks: an unmeasured run must not become a published benchmark")
    paths: list[Path] = []
    for entry in index:
        target = _package_dir(str(entry["package"])) / "BENCHMARK.md"
        _atomic_write_text(target, render_benchmark_md(report, str(entry["name"])))
        paths.append(target)
    return paths


def _package_dir(package: str) -> Path:
    """Directory holding the package's SKILL.md (a seam for tests)."""
    return Path(str(importlib.resources.files(package)))


def _atomic_write_text(path: Path, text: str) -> None:
    path = Path(path)
    temporary: Path | None = None
    try:
        with NamedTemporaryFile(
            "w", encoding="utf-8", dir=str(path.parent), prefix=path.name + ".", suffix=".tmp", delete=False
        ) as handle:
            temporary = Path(handle.name)
            handle.write(text)
        os.replace(temporary, path)
    except BaseException:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise


# --- small formatting and hashing helpers -------------------------------------------------


def _index_view(entry: dict) -> dict:
    return {
        "package": entry["package"],
        "name": entry["name"],
        "skill_md_sha256": entry["skill_md_sha256"],
        "evals_sha256": entry["evals_sha256"],
        "cases": len(entry["evals"]),
    }


def _index_sha256(index: list[dict]) -> str:
    return hashlib.sha256(render_index(index, GATED_ARM).encode("utf-8")).hexdigest()


def _short_hash(report: dict) -> str:
    return str(report.get("index_sha256", ""))[:12]


def _stamp(created: datetime | str | None) -> str:
    if created is None:
        return datetime.now(timezone.utc).isoformat(timespec="seconds")
    if isinstance(created, datetime):
        return created.isoformat()
    return str(created)


def _pct(value: Any) -> str:
    return "n/a" if isinstance(value, bool) or not isinstance(value, (int, float)) else f"{100.0 * float(value):.1f}%"


def _pct_delta(value: Any) -> str:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return "n/a"
    return f"{100.0 * float(value):+.1f}%"


def _flag(value: Any) -> str:
    return "n/a" if value is None else ("yes" if value else "no")
