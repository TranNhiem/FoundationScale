"""A2.2 writing round for any skill package: one change family, measured, then kept or reverted.

``run_round`` takes a skill package, a candidate SKILL.md and the two behaviour reports measured
on the CURRENT SKILL.md (routing and oracle). It swaps the candidate in, re-exports, measures both
case sets with the same recipe, compares each with
:func:`foundationskills.agent.skill_ab.ab_compare`, and applies
the pre-registered rule (artifacts/research/a22_writing_protocol.md):

  KEEP    at least one case set is ``keep`` and the other shows no significant drop (ci_high >= 0)
  REJECT  otherwise -- SKILL.md and both BEHAVIOUR reports are restored byte for byte
  UNMEASURED  a report or comparison is unmeasured -- restored as well; never counted as keep

Every round, kept or rejected, appends one row per case set to the append-only A/B ledger
(``evals/writing-ab.md``). Measurement is injected (``run_eval``) so the CLI can run the real
``fskills behaviour-eval run`` and tests can run a fake.
"""

from __future__ import annotations

import datetime as dt
import json
import sys
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from foundationskills.agent.skill_ab import ab_compare

CASE_SETS = ("routing", "oracle")
REPORT_FILES = {"routing": "BEHAVIOUR.md", "oracle": "BEHAVIOUR_ORACLE.md"}
LEDGER_TITLE = "# Writing A/B ledger"
LEDGER_RULE = (
    "Append-only: one row per case set per A2.2 writing round (kept AND rejected). The decision "
    "follows the pre-registered rule in artifacts/research/a22_writing_protocol.md; rows are never "
    "edited or removed."
)
LEDGER_COLUMNS = (
    "date",
    "change_family",
    "package",
    "cases",
    "reps",
    "agent",
    "judge",
    "bundle_before",
    "bundle_after",
    "with_before",
    "with_after",
    "mean_delta",
    "ci",
    "paired",
    "ab_verdict",
    "decision",
)


class WritingRoundRefused(ValueError):
    """The round cannot start fairly (stale baseline, missing report); nothing was changed."""


def decide(results: Mapping[str, Mapping[str, Any]]) -> tuple[str, str]:
    """The pre-registered decision over the routing and oracle A/B results."""
    verdicts = {cases: str(r.get("verdict")) for cases, r in results.items()}
    if any(v == "unmeasured" for v in verdicts.values()):
        return "unmeasured", f"unmeasured comparison: {verdicts}"
    for cases, result in results.items():
        others = [r for c, r in results.items() if c != cases]
        if result["verdict"] == "keep" and all((o.get("ci_high") or 0.0) >= 0.0 for o in others):
            return (
                "keep",
                f"{cases} kept (ci_low {result['ci_low']:.3f} > 0); no significant drop elsewhere",
            )
    return "reject", f"no case set met the keep rule: {verdicts}"


def _fmt(value: Any) -> str:
    return "-" if value is None else f"{value:.3f}"


def ledger_row(
    result: Mapping[str, Any],
    report: Mapping[str, Any],
    *,
    change_family: str,
    decision: str,
    date: str,
) -> list[str]:
    return [
        date,
        change_family,
        str(result["package"]),
        str(result["cases"]),
        str(report.get("reps")),
        str(report.get("agent_model")),
        str(report.get("judge_model")),
        str(result["skill_bundle_before"])[:12],
        str(result["skill_bundle_after"])[:12],
        _fmt(result["with_before"]),
        _fmt(result["with_after"]),
        _fmt(result["mean_delta"]),
        f"[{_fmt(result['ci_low'])}, {_fmt(result['ci_high'])}]",
        str(result["n_paired"]),
        str(result["verdict"]),
        decision,
    ]


def append_ledger(path: Path, rows: list[list[str]]) -> int:
    """Append rows to the A/B ledger, creating it (title, rule, header) on first use."""
    if not path.exists():
        header = "| " + " | ".join(LEDGER_COLUMNS) + " |\n|" + "---|" * len(LEDGER_COLUMNS) + "\n"
        path.write_text(f"{LEDGER_TITLE}\n\n{LEDGER_RULE}\n\n{header}", encoding="utf-8")
    text = path.read_text(encoding="utf-8")
    last = text.rstrip("\n").splitlines()[-1]
    if not (last.startswith("|") and last.count("|") == len(LEDGER_COLUMNS) + 1):
        raise WritingRoundRefused(f"{path}: does not end in a ledger table row; refusing to append")
    with path.open("a", encoding="utf-8") as fh:
        for row in rows:
            fh.write("| " + " | ".join(cell.replace("|", "/") for cell in row) + " |\n")
    return len(rows)


def run_round(
    package_root: Path,
    candidate: str,
    before: Mapping[str, Mapping[str, Any]],
    *,
    bundle_sha: Callable[[], str],
    run_eval: Callable[[str], Mapping[str, Any]],
    export: Callable[[], object],
    change_family: str,
    ledger: Path,
    date: str | None = None,
) -> dict[str, Any]:
    """One writing round on ``package_root`` (the skill package dir holding SKILL.md)."""
    if set(before) != set(CASE_SETS):
        raise WritingRoundRefused(
            f"run_round: need before reports for {CASE_SETS}, got {sorted(before)}"
        )
    current = bundle_sha()
    for cases, report in before.items():
        measured = str(report["index"][0].get("skill_bundle_sha256"))
        if measured != current:
            raise WritingRoundRefused(
                f"run_round: the {cases} baseline measured bundle {measured[:12]}, "
                "but SKILL.md is now "
                f"{current[:12]}; re-measure the baseline first"
            )
    skill_md = package_root / "SKILL.md"
    if candidate == skill_md.read_text(encoding="utf-8"):
        raise WritingRoundRefused("run_round: the candidate is identical to the current SKILL.md")
    saved = {
        name: (package_root / name).read_bytes()
        for name in ("SKILL.md", *REPORT_FILES.values())
        if (package_root / name).exists()
    }

    def restore() -> None:
        for name, data in saved.items():
            (package_root / name).write_bytes(data)
        export()

    skill_md.write_text(candidate, encoding="utf-8")
    export()
    results: dict[str, dict[str, Any]] = {}
    after: dict[str, Mapping[str, Any]] = {}
    try:
        for cases in CASE_SETS:
            after[cases] = run_eval(cases)
            results[cases] = ab_compare(before[cases], after[cases])
    except Exception:
        restore()
        raise
    decision, reason = decide(results)
    if decision != "keep":
        restore()
    stamp = date or dt.datetime.now(dt.timezone.utc).date().isoformat()
    rows = [
        ledger_row(results[c], after[c], change_family=change_family, decision=decision, date=stamp)
        for c in CASE_SETS
    ]
    appended = append_ledger(ledger, rows)
    return {
        "schema": "fskills.writing_round/1",
        "change_family": change_family,
        "decision": decision,
        "reason": reason,
        "results": results,
        "ledger_rows_appended": appended,
        "skill_md_restored": decision != "keep",
    }


def run_cli(args: Any) -> int:
    """`fskills behaviour-eval round`: one A2.2 writing round with the published recipe."""
    import importlib.resources
    import subprocess

    from foundationskills.agent import export_skills, routing_eval

    entry = next((e for e in routing_eval.load_index() if e["name"] == args.package), None)
    if entry is None:
        print(f"[fskills:behaviour-eval:refuse] unknown skill {args.package!r}", file=sys.stderr)
        return 96
    root = Path(str(importlib.resources.files(entry["package"])))
    work = Path(args.workdir)
    work.mkdir(parents=True, exist_ok=True)
    passthrough = []
    for flag in (
        "agent_base_url",
        "agent_model",
        "judge_base_url",
        "judge_model",
        "agent_api_key_env",
        "judge_api_key_env",
        "agent_extra_body",
        "judge_extra_body",
        "reps",
        "workers",
        "agent_max_tokens",
        "judge_max_tokens",
        "timeout",
    ):
        value = getattr(args, flag)
        if value is not None:
            passthrough += ["--" + flag.replace("_", "-"), str(value)]

    def run_eval(cases: str) -> dict:
        out = work / f"{args.package}_{cases}.json"
        base = [
            sys.executable,
            "-m",
            "foundationskills.cli",
            "behaviour-eval",
            "run",
            "--packages",
            args.package,
            "--cases",
            cases,
            "--write-behaviour",
            *passthrough,
        ]
        subprocess.run([*base, "--out", str(out)], check=False)  # noqa: S603
        report = json.loads(out.read_text())
        if report.get("verdict") == "unmeasured":  # one resume: re-asks only the unmeasured calls
            resumed = work / f"{args.package}_{cases}_resumed.json"
            subprocess.run([*base, "--resume", str(out), "--out", str(resumed)], check=False)  # noqa: S603
            report = json.loads(resumed.read_text())
        return report

    try:
        before = {
            "routing": json.loads(Path(args.before_routing).read_text()),
            "oracle": json.loads(Path(args.before_oracle).read_text()),
        }
        ledger = (
            Path(args.ledger)
            if args.ledger
            else export_skills._roots()[1] / "evals" / "writing-ab.md"
        )
        summary = run_round(
            root,
            Path(args.candidate).read_text(encoding="utf-8"),
            before,
            bundle_sha=lambda: routing_eval.skill_bundle_sha256(root),
            run_eval=run_eval,
            export=lambda: export_skills.write_tree(*export_skills._roots()),
            change_family=args.change_family,
            ledger=ledger,
        )
    except (OSError, ValueError) as exc:
        print(f"[fskills:behaviour-eval:refuse] {exc}", file=sys.stderr)
        return 96
    print(json.dumps(summary, indent=2, default=str))
    return {"keep": 0, "reject": 5, "unmeasured": 95}[summary["decision"]]
