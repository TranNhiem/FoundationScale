"""A2.2 writing round: measure a candidate SKILL.md, keep it or restore every byte; record it."""

from __future__ import annotations

from pathlib import Path

import pytest
from foundationskills.agent import writing_round as wr


def report(cases: str, bundle: str, met: dict[str, int], *, unmeasured: int = 0) -> dict:
    calls = [
        {
            "package": "fskills-x",
            "arm": "with_skill",
            "case_id": c,
            "rep": r,
            "measured": True,
            "items_met": m,
            "items_graded": 3,
        }
        for c, m in met.items()
        for r in range(3)
    ]
    calls += [
        {
            "package": "fskills-x",
            "arm": "without_skill",
            "case_id": c,
            "rep": 0,
            "measured": True,
            "items_met": 0,
            "items_graded": 3,
        }
        for c in met
    ]
    calls += [
        {"package": "fskills-x", "arm": "with_skill", "case_id": "c0", "rep": 9, "measured": False}
        for _ in range(unmeasured)
    ]
    return {
        "schema": "fskills.behaviour_eval/1",
        "cases": cases,
        "agent_model": "a",
        "judge_model": "j",
        "reps": 3,
        "agent_temperature": 0.0,
        "calls": calls,
        "verdict": "pass",
        "index": [
            {
                "name": "fskills-x",
                "package": "pkg.x",
                "cases_sha256": f"{cases}-c",
                "evals_sha256": "e",
                "skill_bundle_sha256": bundle,
            }
        ],
    }


FLAT = {f"c{i}": 2 for i in range(8)}
UP = {f"c{i}": 3 for i in range(8)}
DOWN = {f"c{i}": 1 for i in range(8)}


@pytest.fixture()
def pkg(tmp_path: Path) -> Path:
    root = tmp_path / "pkg"
    root.mkdir()
    (root / "SKILL.md").write_text("ORIGINAL\n")
    (root / "BEHAVIOUR.md").write_text("old routing report\n")
    (root / "BEHAVIOUR_ORACLE.md").write_text("old oracle report\n")
    return root


def go(
    pkg: Path, outcome: dict[str, dict], tmp_path: Path, *, unmeasured: int = 0, boom: bool = False
):
    exports: list[int] = []

    def run_eval(cases: str) -> dict:
        if boom:
            raise RuntimeError("endpoint down")
        (pkg / wr.REPORT_FILES[cases]).write_text(f"new {cases} report\n")  # --write-behaviour
        return report(
            cases, "b1", outcome[cases], unmeasured=unmeasured if cases == "oracle" else 0
        )

    before = {c: report(c, "b0", FLAT) for c in wr.CASE_SETS}
    out = wr.run_round(
        pkg,
        "CANDIDATE\n",
        before,
        bundle_sha=lambda: "b0",
        run_eval=run_eval,
        export=lambda: exports.append(1),
        change_family="anti-rationalizations",
        ledger=tmp_path / "writing-ab.md",
        date="2026-10-17",
    )
    return out, exports


def test_a_kept_change_stays_and_is_recorded(pkg: Path, tmp_path: Path) -> None:
    out, exports = go(pkg, {"routing": FLAT | {"c0": 2}, "oracle": UP}, tmp_path)
    assert out["decision"] == "keep" and out["skill_md_restored"] is False
    assert (pkg / "SKILL.md").read_text() == "CANDIDATE\n" and (
        pkg / "BEHAVIOUR_ORACLE.md"
    ).read_text() != "old oracle report\n"
    rows = (tmp_path / "writing-ab.md").read_text().splitlines()
    assert rows[0] == wr.LEDGER_TITLE and [r.split("|")[-2].strip() for r in rows[-2:]] == [
        "keep",
        "keep",
    ]
    assert exports == [1]


def test_a_rejected_change_restores_every_byte_and_is_recorded(pkg: Path, tmp_path: Path) -> None:
    out, exports = go(pkg, {"routing": FLAT, "oracle": DOWN}, tmp_path)
    assert out["decision"] == "reject" and out["skill_md_restored"] is True
    assert (pkg / "SKILL.md").read_text() == "ORIGINAL\n"
    assert (pkg / "BEHAVIOUR.md").read_text() == "old routing report\n"
    assert (pkg / "BEHAVIOUR_ORACLE.md").read_text() == "old oracle report\n"
    assert exports == [1, 1]  # candidate exported, then the restore re-exported
    assert (tmp_path / "writing-ab.md").read_text().count("| reject |") == 2


def test_a_kept_case_set_with_a_significant_drop_elsewhere_is_rejected(
    pkg: Path, tmp_path: Path
) -> None:
    out, _ = go(pkg, {"routing": DOWN, "oracle": UP}, tmp_path)
    assert out["decision"] == "reject" and (pkg / "SKILL.md").read_text() == "ORIGINAL\n"


def test_unmeasured_restores_and_is_never_keep(pkg: Path, tmp_path: Path) -> None:
    out, _ = go(pkg, {"routing": UP, "oracle": UP}, tmp_path, unmeasured=1)
    assert out["decision"] == "unmeasured" and (pkg / "SKILL.md").read_text() == "ORIGINAL\n"


def test_a_crash_mid_round_restores_and_reraises(pkg: Path, tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="endpoint down"):
        go(pkg, {"routing": UP, "oracle": UP}, tmp_path, boom=True)
    assert (pkg / "SKILL.md").read_text() == "ORIGINAL\n"
    assert not (tmp_path / "writing-ab.md").exists()


def test_a_stale_baseline_or_identical_candidate_refuses_without_touching_anything(
    pkg: Path, tmp_path: Path
) -> None:
    before = {c: report(c, "b-old", FLAT) for c in wr.CASE_SETS}
    with pytest.raises(wr.WritingRoundRefused, match="re-measure the baseline"):
        wr.run_round(
            pkg,
            "CANDIDATE\n",
            before,
            bundle_sha=lambda: "b0",
            run_eval=lambda _c: {},
            export=lambda: None,
            change_family="x",
            ledger=tmp_path / "l.md",
        )
    fresh = {c: report(c, "b0", FLAT) for c in wr.CASE_SETS}
    with pytest.raises(wr.WritingRoundRefused, match="identical"):
        wr.run_round(
            pkg,
            "ORIGINAL\n",
            fresh,
            bundle_sha=lambda: "b0",
            run_eval=lambda _c: {},
            export=lambda: None,
            change_family="x",
            ledger=tmp_path / "l.md",
        )
    assert (pkg / "SKILL.md").read_text() == "ORIGINAL\n" and not (tmp_path / "l.md").exists()


def test_the_ledger_refuses_to_append_to_a_damaged_file(tmp_path: Path) -> None:
    path = tmp_path / "writing-ab.md"
    path.write_text("# Writing A/B ledger\n\nsomething edited by hand\n")
    with pytest.raises(wr.WritingRoundRefused, match="ledger table row"):
        wr.append_ledger(path, [["x"] * len(wr.LEDGER_COLUMNS)])


def test_a_failing_first_export_restores_the_skill_md(pkg: Path, tmp_path: Path) -> None:
    # measured: two parallel rounds raced on the export tree and the first export raised after the
    # candidate was written, leaving it in place; every write is now inside the restore guard
    calls = []

    def export() -> None:
        calls.append(1)
        if len(calls) == 1:
            raise FileNotFoundError("marketplace.json.tmp")

    before = {c: report(c, "b0", FLAT) for c in wr.CASE_SETS}
    with pytest.raises(FileNotFoundError):
        wr.run_round(
            pkg,
            "CANDIDATE\n",
            before,
            bundle_sha=lambda: "b0",
            run_eval=lambda _c: {},
            export=export,
            change_family="x",
            ledger=tmp_path / "l.md",
        )
    assert (pkg / "SKILL.md").read_text() == "ORIGINAL\n" and len(calls) == 2  # restore re-exported
    assert not (tmp_path / "l.md").exists()
