"""Unit tests for :mod:`foundationskills.agent.skill_impact` (synthetic reports, no network)."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest

from foundationskills.agent.skill_impact import (
    ARMS,
    LEDGER_HEADER,
    MIN_REPS,
    SCHEMA,
    SkillImpactRefused,
    append_ledger,
    ledger_rows,
    paired_uplift,
)

REPORT_SCHEMA = "fskills.behaviour_eval/1"
CREATED = "2024-05-01T09:30:00Z"
AGENT = "agent-m/plan"
JUDGE = "judge-m/json"
BUNDLE_SHA256 = "a1" * 32
CASES_SHA256 = "b2" * 32
REPORT_SHA256 = "f3" * 32
DATE = "2024-05-02"
FAMILY = "a2.1-baseline"


def _call(
    package: str,
    case_id: str,
    arm: str,
    rep: int = 0,
    *,
    measured: bool = True,
    items_met: int = 0,
    items_graded: int = 3,
) -> dict[str, Any]:
    """One synthetic behaviour-eval call (schema ``fskills.behaviour_eval/1``)."""
    return {
        "package": package,
        "case_id": case_id,
        "arm": arm,
        "rep": rep,
        "measured": measured,
        "items_met": items_met,
        "items_graded": items_graded,
    }


def _report(
    packages: Sequence[str],
    specs: Mapping[str, Sequence[tuple[str, int, int, int, int]]],
    *,
    reps: int = MIN_REPS,
    cases: str = "routing",
    calls: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """A synthetic report; a spec is ``(case_id, with met, with graded, without met, without graded)``."""
    index = [
        {
            "name": name,
            "package": f"skill-packages/{name}",
            "skill_bundle_sha256": BUNDLE_SHA256,
            "cases_sha256": CASES_SHA256,
        }
        for name in packages
    ]
    built: list[dict[str, Any]] = []
    for name in packages:
        for case_id, with_met, with_graded, without_met, without_graded in specs.get(name, ()):
            for rep in range(reps):
                built.append(
                    _call(name, case_id, "with_skill", rep, items_met=with_met, items_graded=with_graded)
                )
                built.append(
                    _call(
                        name,
                        case_id,
                        "without_skill",
                        rep,
                        items_met=without_met,
                        items_graded=without_graded,
                    )
                )
    built.extend(dict(call) for call in calls)
    return {
        "schema": REPORT_SCHEMA,
        "cases": cases,
        "reps": reps,
        "agent_model": AGENT,
        "judge_model": JUDGE,
        "created": CREATED,
        "index": index,
        "calls": built,
    }


def _cells(**extra: str) -> dict[str, str]:
    """A complete ledger row of filler cells."""
    row = {column: "x" for column in LEDGER_HEADER}
    row.update(extra)
    return row


def _impact(specs: Mapping[str, Sequence[tuple[str, int, int, int, int]]], *, n_boot: int = 2000, seed: int = 0,
            **kwargs: Any) -> dict[str, Any]:
    """A skill-impact record of the given per-package specs (report kwargs go to _report, bootstrap ones to paired_uplift)."""
    return paired_uplift(_report(list(specs), specs, **kwargs), n_boot=n_boot, seed=seed)


# --- paired uplift -----------------------------------------------------------------------------


def test_paired_uplift_all_positive_is_uplift() -> None:
    """A package whose cases all pass with the skill and all fail without it is an uplift."""
    report = _report(
        ["fskills-a", "fskills-b"],
        {
            "fskills-a": [("c1", 3, 3, 0, 3), ("c2", 3, 3, 0, 3), ("c3", 3, 3, 0, 3)],
            "fskills-b": [("c1", 3, 3, 2, 3), ("c2", 2, 3, 3, 3), ("c3", 1, 3, 1, 3)],
        },
    )
    impact = paired_uplift(report)
    assert impact["schema"] == SCHEMA
    assert impact["cases"] == "routing" and impact["reps"] == 3
    assert impact["agent_model"] == AGENT and impact["judge_model"] == JUDGE
    assert impact["report_created"] == CREATED
    assert list(impact["packages"]) == ["fskills-a", "fskills-b"]
    assert impact["exit_code"] == 0

    row = impact["packages"]["fskills-a"]
    assert row["verdict"] == "uplift"
    assert row["ci_low"] > 0 and row["ci_high"] >= row["ci_low"]
    assert row["mean_delta"] == pytest.approx(1.0)
    assert row["with_rate"] == pytest.approx(1.0) and row["without_rate"] == pytest.approx(0.0)
    assert row["n_paired"] == 3 and row["n_abstained"] == 0
    assert row["abstained"] == [] and row["unmeasured_calls"] == 0
    assert row["cases"]["c1"]["delta"] == pytest.approx(1.0)
    assert row["deltas"] == [pytest.approx(1.0)] * 3
    assert row["skill_bundle_sha256"] == BUNDLE_SHA256 and row["cases_sha256"] == CASES_SHA256
    assert row["verdict_reason"].startswith("ci [")

    other = impact["packages"]["fskills-b"]
    assert other["verdict"] == "no_uplift"
    assert other["mean_delta"] == pytest.approx(0.0)
    assert other["ci_low"] <= 0 <= other["ci_high"]


def test_paired_uplift_no_uplift_when_ci_spans_zero() -> None:
    """A mixed package that straddles 0 is measured and reports no uplift (exit code 0)."""
    row = _impact({"fskills-a": [("c1", 3, 3, 2, 3), ("c2", 2, 3, 3, 3), ("c3", 1, 3, 1, 3)]})
    row = row["packages"]["fskills-a"]
    assert row["verdict"] == "no_uplift"
    assert row["ci_low"] <= 0 <= row["ci_high"]
    assert row["n_paired"] == 3 and row["unmeasured_calls"] == 0
    assert "includes 0" in row["verdict_reason"]


def test_paired_uplift_harm_when_the_skill_hurts() -> None:
    """Every case worse with the skill is harm, measured, exit code 0."""
    impact = _impact(
        {"fskills-a": [("c1", 0, 3, 3, 3), ("c2", 0, 3, 3, 3), ("c3", 1, 3, 2, 3)]}
    )
    row = impact["packages"]["fskills-a"]
    assert row["verdict"] == "harm"
    assert row["ci_high"] < 0
    assert row["mean_delta"] < 0
    assert row["with_rate"] < row["without_rate"]
    assert row["verdict_reason"].endswith("excludes 0")
    assert impact["exit_code"] == 0


def test_reps_below_minimum_is_unmeasured() -> None:
    """Fewer than MIN_REPS reps means every package is unmeasured and the run exits 95."""
    impact = _impact(
        {"fskills-a": [("c1", 3, 3, 0, 3)], "fskills-b": [("c1", 3, 3, 0, 3)]},
        reps=MIN_REPS - 1,
    )
    for row in impact["packages"].values():
        assert row["verdict"] == "unmeasured"
        assert row["verdict_reason"].startswith(f"reps {MIN_REPS - 1} < {MIN_REPS}")
    assert impact["exit_code"] == 95


def test_unmeasured_call_marks_unmeasured_and_records_abstain_reason() -> None:
    """One unmeasured cell is never a result: the package is unmeasured and the case abstains."""
    calls = [_call("fskills-a", "c3", "with_skill", 0, measured=False)]
    calls += [
        _call("fskills-a", "c3", "without_skill", rep, items_met=0, items_graded=3)
        for rep in range(MIN_REPS)
    ]
    impact = _report(
        ["fskills-a", "fskills-b"],
        {
            "fskills-a": [("c1", 3, 3, 0, 3), ("c2", 3, 3, 0, 3)],
            "fskills-b": [("c1", 3, 3, 0, 3), ("c2", 3, 3, 0, 3), ("c3", 3, 3, 0, 3)],
        },
        calls=calls,
    )
    impact = paired_uplift(impact)
    row = impact["packages"]["fskills-a"]
    assert row["verdict"] == "unmeasured"
    assert row["unmeasured_calls"] == 1
    assert row["verdict_reason"] == "1 unmeasured call"
    assert row["n_paired"] == 2 and row["n_abstained"] == 1
    assert row["abstained"] == [{"case_id": "c3", "reason": "unmeasured:with_skill"}]
    assert row["cases"]["c3"]["delta"] is None
    assert impact["packages"]["fskills-b"]["verdict"] == "uplift"
    assert impact["exit_code"] == 95


def test_abstain_reason_when_both_arms_unmeasured() -> None:
    """A case with no measured graded call in either arm abstains as ``unmeasured:both``."""
    calls = [_call("fskills-a", "ghost", arm, 0, measured=False) for arm in ARMS]
    row = _impact({"fskills-a": [("c1", 3, 3, 0, 3), ("c2", 3, 3, 0, 3), ("c3", 3, 3, 0, 3)]}, calls=calls)
    row = row["packages"]["fskills-a"]
    assert row["abstained"] == [{"case_id": "ghost", "reason": "unmeasured:both"}]
    assert row["n_abstained"] == 1 and row["n_paired"] == 3
    assert row["verdict"] == "unmeasured" and row["unmeasured_calls"] == 2


def test_bootstrap_ci_is_deterministic_per_seed() -> None:
    """One seed gives exactly one interval; a different seed is allowed to give another one."""
    specs = {"fskills-a": [(f"c{i}", (i % 3) + 1, 3, i % 2, 3) for i in range(8)]}
    first = _impact(specs, n_boot=500, seed=11)["packages"]["fskills-a"]
    second = _impact(specs, n_boot=500, seed=11)["packages"]["fskills-a"]
    assert (first["ci_low"], first["ci_high"]) == (second["ci_low"], second["ci_high"])
    assert first["deltas"] == second["deltas"] == _impact(specs)["packages"]["fskills-a"]["deltas"]

    other = _impact(specs, n_boot=500, seed=12)["packages"]["fskills-a"]
    deltas = first["deltas"]
    assert min(deltas) <= other["ci_low"] <= other["ci_high"] <= max(deltas)


# --- refusals ----------------------------------------------------------------------------------


def test_refused_on_wrong_schema() -> None:
    """A report of any other schema is refused by name."""
    report = _report(["fskills-a"], {"fskills-a": [("c1", 3, 3, 0, 3)]})
    report["schema"] = "fskills.behaviour_eval/2"
    with pytest.raises(SkillImpactRefused, match="schema"):
        paired_uplift(report)


def test_refused_on_missing_calls_and_index() -> None:
    """A report without calls or without an index is refused by name."""
    report = _report(["fskills-a"], {"fskills-a": [("c1", 3, 3, 0, 3), ("c2", 3, 3, 0, 3)]})
    calls = report.pop("calls")
    with pytest.raises(SkillImpactRefused, match="calls"):
        paired_uplift(report)
    report["calls"] = calls
    del report["index"]
    with pytest.raises(SkillImpactRefused, match="index"):
        paired_uplift(report)


def test_refused_on_unknown_package_call() -> None:
    """A call the index does not know is refused, never silently dropped."""
    report = _report(
        ["fskills-a"],
        {"fskills-a": [("c1", 3, 3, 0, 3), ("c2", 3, 3, 0, 3)]},
        calls=[_call("fskills-ghost", "c1", "with_skill")],
    )
    with pytest.raises(SkillImpactRefused, match="fskills-ghost"):
        paired_uplift(report)


# --- the ledger --------------------------------------------------------------------------------


def test_ledger_rows_format() -> None:
    """One row per package in :data:`LEDGER_HEADER` order, rates %.3f and digests abbreviated."""
    rows = ledger_rows(
        _impact(
            {
                "fskills-a": [("c1", 3, 3, 0, 3), ("c2", 3, 3, 0, 3), ("c3", 3, 3, 0, 3)],
                "fskills-b": [("c1", 3, 3, 2, 3), ("c2", 2, 3, 3, 3), ("c3", 1, 3, 1, 3)],
            }
        ),
        change_family=FAMILY,
        report_sha256=REPORT_SHA256,
        date=DATE,
    )
    assert len(rows) == 2
    assert list(rows[0]) == list(LEDGER_HEADER)
    assert rows[0] == {
        **rows[0],
        "date": DATE,
        "change_family": FAMILY,
        "package": "fskills-a",
        "cases": "routing",
        "reps": "3",
        "agent": AGENT,
        "judge": JUDGE,
        "skill_bundle": BUNDLE_SHA256[:12],
        "cases_sha": CASES_SHA256[:12],
        "with": "1.000",
        "without": "0.000",
        "delta": "1.000",
        "ci": "[1.000, 1.000]",
        "paired/abstained": "3/0",
        "verdict": "uplift",
        "report_sha256": REPORT_SHA256[:12],
    }
    assert rows[1]["package"] == "fskills-b"
    assert rows[1]["delta"] == "0.000"
    assert rows[1]["paired/abstained"] == "3/0"
    assert rows[1]["verdict"] == "no_uplift"
    assert rows[1]["ci"].startswith("[-") or rows[1]["ci"].startswith("[0")


def test_ledger_rows_marks_unpaired_row_na() -> None:
    """A package without a paired case leads its uncomputed cells with ``n/a``."""
    calls = [_call("fskills-a", "c1", "with_skill", rep, measured=False) for rep in range(MIN_REPS)]
    calls += [
        _call("fskills-a", "c1", "without_skill", rep, items_met=1, items_graded=3)
        for rep in range(MIN_REPS)
    ]
    rows = ledger_rows(
        paired_uplift(_report(["fskills-a"], {}, calls=calls)),
        change_family=FAMILY,
        report_sha256=REPORT_SHA256,
        date=DATE,
    )
    row = rows[0]
    assert row["with"] == "n/a"
    assert row["without"] == "0.333"
    assert row["delta"] == "n/a"
    assert row["ci"] == "n/a"
    assert row["paired/abstained"] == "0/1"
    assert row["verdict"] == "unmeasured"


def test_append_ledger_creates_and_appends_only(tmp_path: Path) -> None:
    """A new ledger is created once with its title, rule and header, then only appended to."""
    path = tmp_path / "SKILL_IMPACT.md"
    rows = ledger_rows(
        _impact({"fskills-a": [("c1", 3, 3, 0, 3), ("c2", 3, 3, 0, 3)]}),
        change_family=FAMILY,
        report_sha256=REPORT_SHA256,
        date=DATE,
    )
    assert append_ledger(path, rows) == 1
    first = path.read_bytes()
    lines = first.decode("utf-8").splitlines()
    assert lines[0] == "# Skill-impact ledger"
    assert lines[1] == ""
    assert lines[2].startswith("Append-only: one row per measured change family (A2.1 baselines first).")
    assert lines[3] == ""
    assert lines[4] == "| " + " | ".join(LEDGER_HEADER) + " |"
    assert lines[5].split("|")[1:-1] == ["---"] * len(LEDGER_HEADER)
    assert lines[6].startswith(f"| {DATE} | {FAMILY} | fskills-a |")
    assert len(lines) == 7
    assert all(line.count("|") == len(LEDGER_HEADER) + 1 for line in lines[4:])

    more = ledger_rows(
        _impact({"fskills-a": [("c1", 1, 3, 1, 3), ("c2", 1, 3, 1, 3)]}),
        change_family="a2.1-second",
        report_sha256="8a" * 32,
        date="2024-06-01",
    )
    assert append_ledger(path, more) == 1
    second = path.read_bytes()
    assert second.startswith(first)  # prefix check: earlier bytes are unchanged
    assert len(second) > len(first)
    assert second.decode("utf-8").splitlines()[-1].startswith("| 2024-06-01 | a2.1-second | fskills-a |")


def test_append_ledger_refuses_non_table_tail(tmp_path: Path) -> None:
    """A ledger whose last line is not a full-width table row is refused, never rewritten."""
    path = tmp_path / "SKILL_IMPACT.md"
    path.write_text("| " + " | ".join(LEDGER_HEADER) + " |\n| a | b |\n", encoding="utf-8")
    before = path.read_bytes()
    with pytest.raises(SkillImpactRefused, match="table row"):
        append_ledger(path, [_cells()])
    assert path.read_bytes() == before

    path.write_text("not a table tail\n", encoding="utf-8")
    with pytest.raises(SkillImpactRefused, match="table row"):
        append_ledger(path, [_cells()])


def test_append_ledger_refuses_bad_cells(tmp_path: Path) -> None:
    """A cell that would break the row (``|`` or a newline) is refused before anything is written."""
    path = tmp_path / "SKILL_IMPACT.md"
    with pytest.raises(SkillImpactRefused, match="newline"):
        append_ledger(path, [_cells(change_family="a|b")])
    assert not path.exists()
    with pytest.raises(SkillImpactRefused, match="newline"):
        append_ledger(path, [_cells(date="2024\n05")])
    assert not path.exists()

    impact = _impact({"fskills-a": [("c1", 3, 3, 0, 3), ("c2", 3, 3, 0, 3)]})
    with pytest.raises(SkillImpactRefused, match="newline"):
        ledger_rows(impact, change_family="a|b", report_sha256=REPORT_SHA256, date=DATE)


def test_cli_uplift_appends_the_ledger_and_exits_by_the_worst_report(tmp_path: Path) -> None:
    """`fskills behaviour-eval uplift` reads reports, appends one ledger row per package and exits 0/95/96."""
    import json

    from foundationskills.cli import main

    good = tmp_path / "good.json"
    good.write_text(json.dumps(_report(["fskills-a"], {"fskills-a": [(f"c{i}", 3, 3, 0, 3) for i in range(4)]})))
    ledger = tmp_path / "skill-impact.md"
    assert main(["behaviour-eval", "uplift", "--report", str(good), "--ledger", str(ledger),
                 "--change-family", "baseline"]) == 0
    assert ledger.read_text().count("| baseline |") == 1
    short = tmp_path / "short.json"
    short.write_text(json.dumps(_report(["fskills-a"], {"fskills-a": [(f"c{i}", 3, 3, 0, 3) for i in range(4)]},
                                        reps=MIN_REPS - 1)))
    assert main(["behaviour-eval", "uplift", "--report", str(good), "--report", str(short)]) == 95
    assert main(["behaviour-eval", "uplift", "--report", str(good), "--ledger", str(ledger)]) == 96


def test_append_ledger_creates_a_missing_parent_directory(tmp_path: Path) -> None:
    """The first baseline writes FoundationSkills/evals/skill-impact.md before evals/ exists."""
    impact = _impact({"fskills-a": [(f"c{i}", 3, 3, 0, 3) for i in range(3)]})
    rows = ledger_rows(impact, change_family="baseline", report_sha256="f" * 64, date="2026-10-11")
    path = tmp_path / "evals" / "skill-impact.md"
    assert append_ledger(path, rows) == 1 and path.is_file()
