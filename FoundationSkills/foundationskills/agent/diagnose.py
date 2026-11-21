from __future__ import annotations

"""Mine a C9 measurement report for repeated with-skill item misses.

Groups (package, item) pairs missing min_count+ times into ranked hypotheses."""

from collections import defaultdict

SCHEMA = "fskills.self_improve.hypothesis/1"


def hypotheses(report: dict, *, min_count: int = 2) -> list[dict]:
    """Return ranked SCHEMA hypotheses mined from report["calls"] (with-skill arm).

    Never raises on missing keys; absent or malformed fields contribute
    no groups instead of blowing up.
    """
    misses: dict[tuple[str, str], int] = defaultdict(int)
    cases: dict[tuple[str, str], set[str]] = defaultdict(set)
    seen: dict[tuple[str, str], int] = defaultdict(int)

    calls = report.get("calls") if isinstance(report, dict) else None
    for call in calls or []:
        if not isinstance(call, dict):
            continue
        if call.get("arm") != "with_skill" or not call.get("measured"):
            continue
        items = call.get("item_texts") or []
        mets = call.get("met") or []
        if not isinstance(items, (list, tuple)) or not isinstance(mets, (list, tuple)):
            continue
        pairs = list(zip(items, mets))  # zip truncates to the shorter side
        if not pairs:
            continue
        pkg = str(call.get("package") or "")
        case_id = str(call.get("case_id") or call.get("case") or "")
        for item, ok in pairs:
            if ok:  # only met == False counts as a miss
                continue
            key = (pkg, str(item))
            misses[key] += 1
            if case_id:
                cases[key].add(case_id)
        # one "seen" per call per distinct item text in that package
        for key in {(pkg, str(item)) for item, _ in pairs}:
            seen[key] += 1

    out: list[dict] = []
    for key, count in misses.items():
        if count < min_count:
            continue
        out.append(
            {
                "schema": SCHEMA,
                "package": key[0],
                "item": key[1],
                "misses": count,
                "cases": sorted(cases.get(key, ())),
                "seen": seen.get(key, 0),
            }
        )
    out.sort(key=lambda h: (-h["misses"], h["package"], h["item"]))
    return out
