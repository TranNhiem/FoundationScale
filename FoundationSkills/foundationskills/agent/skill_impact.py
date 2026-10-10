"""Skill-impact measurement for FoundationSkills (A2.1): paired uplift and the skill-impact ledger.

This module answers one question per skill package: what did the skill text add? It takes a
behaviour-eval report (``fskills.behaviour_eval/1``, see :mod:`foundationskills.agent.behaviour_eval`),
pairs every case against its own ablation and records the result as ``fskills.skill_impact/1``, the
record the append-only skill-impact ledger stores (A2.1 baselines first).

Arms (:data:`ARMS`) are the two arms the behaviour eval ran on the same case with the same plan-level
judge:

* ``with_skill`` -- the agent saw the package's ``SKILL.md`` (the gated arm).
* ``without_skill`` -- the same agent prompt without the skill text: the package's ablation.

Paired cells and abstentions
----------------------------
A case contributes one paired delta: ``item_pass_rate(with_skill) - item_pass_rate(ablation)`` where an
item pass rate is ``sum(items_met) / sum(items_graded)`` over MEASURED calls of that arm with all reps
summed (behaviour eval's policy: a check that did not run is never a result, so an unmeasured call
contributes to nothing). A case is PAIRED only when both arms have at least one measured call with
``items_graded > 0``; anything else ABSTAINS and records which arm was missing (``unmeasured:<arm>``,
``unmeasured:both``). ``mean_delta`` is the mean of the paired deltas, and the bootstrap interval
resamples those paired deltas with replacement (:func:`paired_uplift` draws from ``random.Random(seed)``
so one report and one seed give exactly one interval). With fewer than two paired cases there is no
interval at all.

Verdicts and exit codes
-----------------------
A package verdict is ``unmeasured`` -- and can never become a published result -- while any cell is
missing: fewer than :data:`MIN_REPS` reps in the report, an unmeasured call anywhere in the package, or
fewer than two paired cases. Otherwise ``uplift`` (the interval excludes 0 upwards), ``harm`` (it
excludes 0 downwards) or ``no_uplift``. Exit codes follow the repo doctrine: ``0`` when every package
reached a verdict, ``95`` UNMEASURED when any package is unmeasured. Refusals raise
:class:`SkillImpactRefused` and name the missing input (a report of the wrong schema, a report without
``calls`` or ``index``, a call for a package the index does not know).

The ledger
----------
:func:`ledger_rows` renders one row per package. :func:`append_ledger` appends rows to the append-only
markdown ledger: the file is created once with its title, its rule paragraph and the header table, and
afterwards only ever appended to -- a healthy ledger always ends in a table row of exactly
:data:`LEDGER_HEADER`'s width and a cell that would break its row (``|`` or a newline) is refused, so
the file is never rewritten. Ratios are ``%.3f``, an uncomputed value is ``n/a``, and digests are led by
their first 12 hex characters so a row still names its inputs. The ``delta`` column is the mean paired
delta, the same quantity the ``ci`` column bootstraps.

This module is stdlib-only and never touches a network: it reads one report dict and writes text.
"""
from __future__ import annotations

import math
import random
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

MIN_REPS = 3
SCHEMA = "fskills.skill_impact/1"
REPORT_SCHEMA = "fskills.behaviour_eval/1"
ARMS: tuple[str, ...] = ("with_skill", "without_skill")
EXIT_PASS = 0
EXIT_UNMEASURED = 95
LEDGER_HEADER: tuple[str, ...] = (
    "date",
    "change_family",
    "package",
    "cases",
    "reps",
    "agent",
    "judge",
    "skill_bundle",
    "cases_sha",
    "with",
    "without",
    "delta",
    "ci",
    "paired/abstained",
    "verdict",
    "report_sha256",
)

_ABBREV = 12
_TITLE = "# Skill-impact ledger"
_PARAGRAPH = (
    "Append-only: one row per measured change family (A2.1 baselines first). Rows are never edited "
    "or removed; a re-measurement is a new row."
)


class SkillImpactRefused(ValueError):
    """The skill-impact work refused; the message names the missing input or precondition."""


# --- small formatters --------------------------------------------------------------------------


def _as_int(value: Any) -> int:
    """A tolerant int for report numbers (a missing or unparsable count is 0)."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _rate(met: int, graded: int) -> float | None:
    """``sum(items_met) / sum(items_graded)``; None when nothing was graded."""
    return met / graded if graded > 0 else None


def _rate_cell(value: Any) -> str:
    """A ledger cell for a rate: ``%.3f`` or ``n/a``."""
    return "n/a" if value is None else f"{float(value):.3f}"


def _ci_cell(low: Any, high: Any) -> str:
    """A ledger cell for an interval: ``[%.3f, %.3f]`` or ``n/a``."""
    return "n/a" if low is None or high is None else f"[{float(low):.3f}, {float(high):.3f}]"


def _abbrev(digest: Any) -> str:
    """The leading hex of a digest, the ledger's short handle on an input."""
    return str(digest)[:_ABBREV]


def _require_cells(cells: Mapping[str, Any], where: str) -> dict[str, str]:
    """Clean ledger cells (no ``|``, no newline) covering every :data:`LEDGER_HEADER` column."""
    clean: dict[str, str] = {}
    for column in LEDGER_HEADER:
        if column not in cells:
            raise SkillImpactRefused(f"{where}: row is missing the {column!r} column")
        text = str(cells[column])
        if "|" in text or "\n" in text or "\r" in text:
            raise SkillImpactRefused(f"{where}: cell {column} contains '|' or a newline: {text!r}")
        clean[column] = text
    return clean


def _render_row(cells: Mapping[str, str]) -> str:
    """One markdown table row in :data:`LEDGER_HEADER` order."""
    return "| " + " | ".join(cells[column] for column in LEDGER_HEADER) + " |"


# --- input checks ------------------------------------------------------------------------------


def _require_report(report: Mapping[str, Any]) -> None:
    """Refuse anything that is not a ``fskills.behaviour_eval/1`` report carrying calls and an index."""
    if not isinstance(report, Mapping):
        raise SkillImpactRefused(f"paired_uplift: report is not a mapping: {report!r}")
    schema = str(report.get("schema", ""))
    if schema != REPORT_SCHEMA:
        raise SkillImpactRefused(f"paired_uplift: report schema {schema!r} is not {REPORT_SCHEMA!r}")
    for key in ("calls", "index"):
        if not isinstance(report.get(key), list):
            raise SkillImpactRefused(f"paired_uplift: report has no {key} list")


def _require_index(report: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    """The report's index entries, each a mapping (the call's ``package`` refers to its ``name``)."""
    index: list[Mapping[str, Any]] = []
    for entry in report["index"]:
        if not isinstance(entry, Mapping):
            raise SkillImpactRefused(f"paired_uplift: index entry is not a mapping: {entry!r}")
        index.append(entry)
    return index


# --- the measurement ---------------------------------------------------------------------------


def _group_calls(
    calls: Sequence[Mapping[str, Any]], names: Sequence[str]
) -> tuple[dict[str, Any], dict[str, int]]:
    """Per package: case -> arm -> measured/met/graded sums, plus the unmeasured call counts."""
    cells: dict[str, Any] = {name: {} for name in names}
    unmeasured: dict[str, int] = {name: 0 for name in names}
    for call in calls:
        name = str(call.get("package", ""))
        if name not in cells:
            raise SkillImpactRefused(
                f"paired_uplift: call for package {name!r} is not in the report index"
            )
        arm = str(call.get("arm", ""))
        if arm not in ARMS:
            raise SkillImpactRefused(f"paired_uplift: call for {name} has unknown arm {arm!r}")
        case_id = str(call.get("case_id", ""))
        cell = cells[name].setdefault(case_id, {}).setdefault(
            arm, {"met": 0, "graded": 0, "n_graded": 0}
        )
        if not bool(call.get("measured")):
            unmeasured[name] += 1
            continue
        met = _as_int(call.get("items_met"))
        graded = _as_int(call.get("items_graded"))
        cell["met"] += met
        cell["graded"] += graded
        if graded > 0:
            cell["n_graded"] += 1
    return cells, unmeasured


def _bootstrap_ci(
    deltas: Sequence[float], *, n_boot: int, seed: int, level: float
) -> tuple[float | None, float | None]:
    """Percentile interval over ``n_boot`` resamples of the paired deltas (None when < 2 pairs)."""
    n_pairs = len(deltas)
    if n_pairs < 2 or n_boot < 1:
        return None, None
    rng = random.Random(seed)
    means = [
        math.fsum(deltas[rng.randrange(n_pairs)] for _ in range(n_pairs)) / n_pairs
        for _ in range(n_boot)
    ]
    ordered = sorted(means)
    low = max(0, min(n_boot - 1, math.floor((1.0 - level) / 2.0 * n_boot)))
    high = max(0, min(n_boot - 1, math.ceil((1.0 + level) / 2.0 * n_boot) - 1))
    return ordered[low], ordered[high]


def _verdict(reps: int, row: Mapping[str, Any]) -> tuple[str, str]:
    """A package verdict and its short reason; a missing cell is never a result."""
    reasons: list[str] = []
    if reps < MIN_REPS:
        reasons.append(f"reps {reps} < {MIN_REPS}")
    unmeasured = int(row["unmeasured_calls"])
    if unmeasured > 0:
        reasons.append(f"{unmeasured} unmeasured call{'' if unmeasured == 1 else 's'}")
    pairs = int(row["n_paired"])
    if pairs < 2:
        reasons.append(f"{pairs} paired case{'' if pairs == 1 else 's'} < 2")
    ci_low, ci_high = row["ci_low"], row["ci_high"]
    if ci_low is None or ci_high is None:
        reasons.append("no bootstrap interval")
    if reasons:
        return "unmeasured", "; ".join(reasons)
    span = f"ci [{ci_low:.2f}, {ci_high:.2f}]"
    if ci_low > 0:
        return "uplift", f"{span} excludes 0"
    if ci_high < 0:
        return "harm", f"{span} excludes 0"
    return "no_uplift", f"{span} includes 0"


def _package_row(
    entry: Mapping[str, Any],
    name: str,
    cases: Mapping[str, Any],
    unmeasured: int,
    *,
    reps: int,
    n_boot: int,
    seed: int,
    level: float,
) -> dict[str, Any]:
    """One package's paired case rates, ablation deltas, interval and verdict."""
    arm_met = dict.fromkeys(ARMS, 0)
    arm_graded = dict.fromkeys(ARMS, 0)
    case_rows: dict[str, dict[str, Any]] = {}
    deltas: dict[str, float] = {}
    abstained: list[dict[str, str]] = []
    for case_id in sorted(cases):
        rates: dict[str, float | None] = {}
        missing: list[str] = []
        for arm in ARMS:
            cell = cases[case_id].get(arm, {"met": 0, "graded": 0, "n_graded": 0})
            arm_met[arm] += cell["met"]
            arm_graded[arm] += cell["graded"]
            rates[arm] = _rate(cell["met"], cell["graded"])
            if cell["n_graded"] < 1:
                missing.append(arm)
        if missing:
            reason = (
                "unmeasured:both" if len(missing) == len(ARMS) else f"unmeasured:{missing[0]}"
            )
            abstained.append({"case_id": case_id, "reason": reason})
            case_rows[case_id] = {
                "with_rate": rates["with_skill"],
                "without_rate": rates["without_skill"],
                "delta": None,
            }
            continue
        delta = float(rates["with_skill"]) - float(rates["without_skill"])
        deltas[case_id] = delta
        case_rows[case_id] = {
            "with_rate": rates["with_skill"],
            "without_rate": rates["without_skill"],
            "delta": delta,
        }
    values = [deltas[case_id] for case_id in sorted(deltas)]
    mean_delta = math.fsum(values) / len(values) if values else None
    ci_low, ci_high = _bootstrap_ci(values, n_boot=n_boot, seed=seed, level=level)
    row: dict[str, Any] = {
        "package": name,
        "skill_bundle_sha256": str(entry.get("skill_bundle_sha256", "")),
        "cases_sha256": str(entry.get("cases_sha256", "")),
        "cases": case_rows,
        "deltas": values,
        "mean_delta": mean_delta,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "with_rate": _rate(arm_met["with_skill"], arm_graded["with_skill"]),
        "without_rate": _rate(arm_met["without_skill"], arm_graded["without_skill"]),
        "n_paired": len(values),
        "n_abstained": len(abstained),
        "abstained": sorted(abstained, key=lambda item: item["case_id"]),
        "unmeasured_calls": unmeasured,
    }
    row["verdict"], row["verdict_reason"] = _verdict(reps, row)
    return row


def paired_uplift(
    report: Mapping[str, Any], *, n_boot: int = 2000, seed: int = 0, level: float = 0.95
) -> dict[str, Any]:
    """The per-package paired-uplift record (``fskills.skill_impact/1``) of one behaviour-eval report.

    Packages follow ``report["index"]`` order; every repository draws its bootstrap samples from its
    own ``random.Random(seed)`` so a package's interval never depends on its siblings.
    """
    _require_report(report)
    index = _require_index(report)
    names: list[str] = []
    for entry in index:
        name = str(entry.get("name", ""))
        if not name:
            raise SkillImpactRefused("paired_uplift: an index entry has no name")
        names.append(name)
    cells, unmeasured = _group_calls(list(report["calls"]), names)
    reps = _as_int(report.get("reps"))
    packages: dict[str, dict[str, Any]] = {}
    for entry, name in zip(index, names):
        packages[name] = _package_row(
            entry,
            name,
            cells[name],
            unmeasured[name],
            reps=reps,
            n_boot=n_boot,
            seed=seed,
            level=level,
        )
    exit_code = (
        EXIT_UNMEASURED
        if any(row["verdict"] == "unmeasured" for row in packages.values())
        else EXIT_PASS
    )
    return {
        "schema": SCHEMA,
        "cases": report.get("cases"),
        "reps": reps,
        "agent_model": report.get("agent_model"),
        "judge_model": report.get("judge_model"),
        "report_created": report.get("created"),
        "packages": packages,
        "exit_code": exit_code,
    }


# --- the ledger --------------------------------------------------------------------------------


def ledger_rows(
    impact: Mapping[str, Any], *, change_family: str, report_sha256: str, date: str
) -> list[dict[str, str]]:
    """One :data:`LEDGER_HEADER` row per package of a skill-impact record, in report order.

    ``delta`` is the mean paired delta (the quantity ``ci`` bootstraps); ``with``/``without`` are the
    package-level arm rates and every digest is abbreviated to its first 12 hex characters.
    """
    if not isinstance(impact, Mapping):
        raise SkillImpactRefused(f"ledger_rows: impact is not a mapping: {impact!r}")
    if str(impact.get("schema", "")) != SCHEMA:
        raise SkillImpactRefused(
            f"ledger_rows: impact schema {impact.get('schema')!r} is not {SCHEMA!r}"
        )
    packages = impact.get("packages")
    if not isinstance(packages, Mapping):
        raise SkillImpactRefused("ledger_rows: impact has no 'packages' map")
    rows: list[dict[str, str]] = []
    for name, row in packages.items():
        if not isinstance(row, Mapping):
            raise SkillImpactRefused(f"ledger_rows: package {name!r} is not a map")
        rows.append(
            _require_cells(
                {
                    "date": date,
                    "change_family": change_family,
                    "package": str(name),
                    "cases": str(impact.get("cases", "")),
                    "reps": str(impact.get("reps", "")),
                    "agent": str(impact.get("agent_model", "")),
                    "judge": str(impact.get("judge_model", "")),
                    "skill_bundle": _abbrev(row.get("skill_bundle_sha256", "")),
                    "cases_sha": _abbrev(row.get("cases_sha256", "")),
                    "with": _rate_cell(row.get("with_rate")),
                    "without": _rate_cell(row.get("without_rate")),
                    "delta": _rate_cell(row.get("mean_delta")),
                    "ci": _ci_cell(row.get("ci_low"), row.get("ci_high")),
                    "paired/abstained": (
                        f"{_as_int(row.get('n_paired'))}/{_as_int(row.get('n_abstained'))}"
                    ),
                    "verdict": str(row.get("verdict", "")),
                    "report_sha256": _abbrev(report_sha256),
                },
                "ledger_rows",
            )
        )
    return rows


def _preamble() -> str:
    """The ledger's title, rule paragraph and header table (written once, never rewritten)."""
    header = "| " + " | ".join(LEDGER_HEADER) + " |"
    separator = "|" + "|".join("---" for _ in LEDGER_HEADER) + "|"
    return f"{_TITLE}\n\n{_PARAGRAPH}\n\n{header}\n{separator}\n"


def _last_line(text: str) -> str:
    """The last non-blank line of a file (a file has no table edge case worth preserving)."""
    lines = [line for line in text.splitlines() if line.strip()]
    return lines[-1] if lines else ""


def _is_table_row(line: str) -> bool:
    """True for a markdown table row of exactly :data:`LEDGER_HEADER`'s width."""
    if not line.startswith("|") or not line.endswith("|") or len(line) < 2:
        return False
    return len(line.split("|")[1:-1]) == len(LEDGER_HEADER)


def append_ledger(path: Path, rows: Sequence[Mapping[str, str]]) -> int:
    """Append ``rows`` to the append-only markdown ledger at ``path``; return how many rows were written.

    A missing ledger is created once with its title, its rule paragraph and the header table; an
    existing one is only appended to and must end in a table row of :data:`LEDGER_HEADER`'s width, so
    a broken ledger is refused rather than rewritten. Every cell must survive a markdown row intact.
    """
    payload = [_render_row(_require_cells(row, "append_ledger")) for row in rows]
    existing = ""
    if path.exists():
        try:
            existing = path.read_text(encoding="utf-8")
        except OSError as exc:
            raise SkillImpactRefused(f"append_ledger: cannot read {path}: {exc}") from exc
        if not _is_table_row(_last_line(existing)):
            raise SkillImpactRefused(
                f"append_ledger: {path} does not end with a table row of {len(LEDGER_HEADER)} "
                "cells; refusing to rewrite"
            )
    body = "".join(f"{line}\n" for line in payload)
    if body and existing and not existing.endswith("\n"):
        body = f"\n{body}"
    chunk = body if existing.strip() else _preamble() + body
    if not existing:
        path.parent.mkdir(parents=True, exist_ok=True)  # a new ledger may live in a new evals/ directory
    with path.open("a", encoding="utf-8") as handle:
        handle.write(chunk)
    return len(payload)
