#!/usr/bin/env python3
"""Re-derive every countable the review drafts state (#220).

This exists because the first census was run as a shell one-liner and thrown
away, so when D8 landed there was no way to refresh the numbers except by
hand -- which is how they drifted in the first place. A countable a document
states must have a PRODUCER, not a memory. This script ships inside the tree
it measures and derives that tree from its own location, so any checkout of
the repo carries its own census with it.

Validation is built in: keys the tree has not touched (the h100_validation
subtree, the launchers, DECISIONS.md) are compared against the previous
census, and a mismatch there means THIS script is wrong, not the tree. A
census with no self-check is a number with no denominator.

Run:  python3 tools/countables_census.py --out ground_truth.json [--check prior.json]
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

# Derived from this file's own location, not configured: a census that can be
# pointed at a different tree than the one it ships in is an oracle with no
# anchor.
REPO = Path(__file__).resolve().parents[1]
# The interpreter actually running this census, so the tree is measured under
# whatever Python CI (or a developer) invoked -- not a .venv path that may
# not exist.
PY = sys.executable

# Directories that are build residue, never source. Counting __pycache__ would
# make every countable a function of whether someone had recently run pytest.
NOISE = {"__pycache__", ".venv", ".git", ".pytest_cache", ".mypy_cache", ".ruff_cache"}


def _py_files(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*.py") if not any(part in NOISE for part in p.parts))


def _loc(paths: list[Path]) -> int:
    # Physical lines, blanks and comments included -- the same "wc -l" notion
    # the drafts use. A "significant lines" count would be a different number
    # under the same word, which is the drift this whole exercise is about.
    return sum(len(p.read_text(encoding="utf-8", errors="replace").splitlines()) for p in paths)


def _count_subtree(root: Path) -> tuple[int, int]:
    files = _py_files(root)
    return len(files), _loc(files)


def _package_imports(root: Path) -> int:
    """Count IMPORT STATEMENTS that name the first-party package.

    AST, not regex: a regex counts the word `foundationscale` inside strings,
    comments and docstrings, and this repo's docstrings mention the package
    constantly. Relative imports are excluded and that is stated, because the
    drafts' figures may or may not include them -- naming the method is the
    point.
    """
    n = 0
    for p in _py_files(root):
        try:
            tree = ast.parse(p.read_text(encoding="utf-8", errors="replace"))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                n += sum(1 for a in node.names if a.name.split(".")[0] == "foundationscale")
            elif (
                isinstance(node, ast.ImportFrom)
                and node.level == 0
                and (node.module or "").split(".")[0] == "foundationscale"
            ):
                # level == 0 excludes relative imports: `from . import x` inside
                # the package is not a dependency ON the package.
                n += 1
    return n


def _shim_names() -> tuple[int, int, int]:
    """Names tools/live_save_gate.py re-exports from the moved decision API."""
    tree = ast.parse((REPO / "tools/live_save_gate.py").read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith(
            "foundationscale.gates.adjudication"
        ):
            for a in node.names:
                names.add(a.asname or a.name)
    private = {n for n in names if n.startswith("_")}
    return len(names), len(private), len(names) - len(private)


def _mutation_corpus() -> dict[str, int]:
    """Read the battery's corpus from the BATTERY, not from a re-derivation.

    Three re-derivations of this one number gave three answers: the JSON alone
    said 64, JSON + a regex over mutate.py said 64, and JSON + an AST over the
    embedded literal said 72. The battery's own ``--list`` footer says 73
    across 9 modules, and it is the thing that actually runs the mutants -- so
    it is the oracle and the others were guesses about its internals.

    The JSON is still read, but only as a CROSS-CHECK: its rows must be a
    strict subset of what the battery publishes, and the surplus must be
    exactly the embedded table. That way a JSON that grows a duplicate copy of
    an embedded row (which load_table refuses) also shows up here as an
    arithmetic contradiction rather than as a plausible larger number.
    """
    out = subprocess.run(
        [str(PY), "tools/mutate.py", "--list"], cwd=REPO, capture_output=True, text=True
    )
    if out.returncode != 0:
        raise SystemExit(f"mutate.py --list failed rc={out.returncode}; corpus UNMEASURED")
    footer = re.search(r"^(\d+) mutations across (\d+) module\(s\)", out.stdout, re.M)
    if not footer:
        raise SystemExit(
            "mutate.py --list printed no 'N mutations across M module(s)' footer; the "
            "corpus size is unmeasured. Refusing to substitute a hand count"
        )
    total, mods = int(footer.group(1)), int(footer.group(2))
    per_mod = {
        m.group(1): int(m.group(2))
        for m in re.finditer(r"^(\S+)\s+\((\d+) mutations\)", out.stdout, re.M)
    }
    if sum(per_mod.values()) != total or len(per_mod) != mods:
        raise SystemExit(
            f"battery footer ({total} rows / {mods} modules) disagrees with its own "
            f"per-module lines ({sum(per_mod.values())} / {len(per_mod)})"
        )
    js = _json_rows()
    embedded = total - js["json_rows"]
    if embedded < 0:
        raise SystemExit(
            f"mutations.json carries {js['json_rows']} rows but the battery publishes only "
            f"{total}; the JSON has grown a copy of an embedded row"
        )
    return {
        "total_rows": total,
        "mut_modules": mods,
        "json_rows": js["json_rows"],
        "json_mods": js["json_mods"],
        "embedded_rows": embedded,
        "must_pass": js["must_pass_json"] + _embedded_must_pass(),
        "must_fire": total - (js["must_pass_json"] + _embedded_must_pass()),
    }


def _json_rows() -> dict[str, int]:
    """Count the JSON half of the corpus.

    mutations.json is a MAPPING of module name -> list of rows, not a flat
    list. Reading it as a list yields zero rows and a zero-row battery reports
    "every mutant killed" over nothing -- the vacuous PASS this repo exists to
    refuse. The shape is asserted rather than assumed.
    """
    raw = json.loads((REPO / "tools/mutations.json").read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or not all(isinstance(v, list) for v in raw.values()):
        raise SystemExit(
            "mutations.json is not the expected {module: [rows]} mapping; refusing to "
            "guess a row count rather than report a wrong one"
        )
    rows = [r for mod in raw.values() for r in mod]
    if not rows:
        raise SystemExit("mutations.json parsed to ZERO rows; that is unmeasured, not empty")

    return {
        "json_rows": len(rows),
        "json_mods": len(raw),
        "must_pass_json": sum(1 for r in rows if r.get("must_survive")),
    }


def _embedded_must_pass() -> int:
    """MUST-PASS rows carried in mutate.py literals rather than the JSON.

    Read by AST over every module-level dict literal reachable from a `*_ROWS`
    binding OR from EMBEDDED_TABLE, because the inert control lives in the
    latter and not the former -- which is exactly why an AST count of the
    `*_ROWS` list alone came out one short of what the battery publishes.
    """
    tree = ast.parse((REPO / "tools/mutate.py").read_text(encoding="utf-8"))
    n = 0
    seen: list[str] = []
    for node in tree.body:
        # AnnAssign as well as Assign: EMBEDDED_TABLE carries an annotation
        # (`: dict[str, list[dict]]`), so an Assign-only walk skipped it and
        # under-counted MUST-PASS by exactly the one inert control it holds.
        if isinstance(node, ast.Assign):
            targets, value = node.targets, node.value
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            targets, value = [node.target], node.value
        else:
            continue
        names = [t.id for t in targets if isinstance(t, ast.Name)]
        if not any(x.endswith("_ROWS") or x == "EMBEDDED_TABLE" for x in names):
            continue
        seen.extend(names)
        for sub in ast.walk(value):
            if not isinstance(sub, ast.Dict):
                continue
            for k, v in zip(sub.keys, sub.values, strict=True):
                if (
                    isinstance(k, ast.Constant)
                    and k.value == "must_survive"
                    and isinstance(v, ast.Constant)
                    and v.value
                ):
                    n += 1
    if not seen:
        raise SystemExit(
            "no *_ROWS or EMBEDDED_TABLE binding found in tools/mutate.py; the embedded "
            "MUST-PASS count would be a vacuous 0"
        )
    return n


def _coverage() -> dict[str, float]:
    """Statement coverage, re-run rather than remembered."""
    out: dict[str, float] = {}
    with tempfile.TemporaryDirectory() as tmp:
        cov = Path(tmp) / "cov.json"
        xml = Path(tmp) / "junit_census.xml"
        subprocess.run(
            [
                str(PY),
                "-m",
                "pytest",
                "-q",
                "-p",
                "no:cacheprovider",
                "--cov=foundationscale",
                f"--cov-report=json:{cov}",
                f"--junit-xml={xml}",
            ],
            cwd=REPO,
            capture_output=True,
            text=True,
            env={**_env(), "FS_FORBID_SKIPS": "1"},
        )
        if cov.exists():
            d = json.loads(cov.read_text())["totals"]
            out["cov_stmts"] = d["num_statements"]
            out["cov_missed"] = d["missing_lines"]
            out["cov_pct"] = round(d["percent_covered"], 2)
        if xml.exists():
            import xml.etree.ElementTree as ET

            r = ET.parse(xml).getroot()
            s = r.find("testsuite") if r.tag == "testsuites" else r
            a = s.attrib
            out["junit_entries"] = int(a["tests"])
            out["junit_failures"] = int(a["failures"]) + int(a["errors"])
            out["junit_skipped"] = int(a["skipped"])
    return out


def _env() -> dict[str, str]:
    import os

    return dict(os.environ)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", type=Path, default=None, help="prior census to diff against")
    # Required with no default: a census writer with a default output path
    # silently overwrites someone's file.
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--no-coverage", action="store_true", help="skip the pytest+coverage run")
    args = ap.parse_args()

    gt: dict[str, object] = {}
    gt["src_files"], gt["src_loc"] = _count_subtree(REPO / "src/foundationscale")
    gt["tests_files"], gt["tests_loc"] = _count_subtree(REPO / "tests")
    gt["tools_files"], gt["tools_loc"] = _count_subtree(REPO / "tools")
    gt["gates_files"] = len(_py_files(REPO / "src/foundationscale/gates"))
    gt["root_init_loc"] = _loc([REPO / "src/foundationscale/__init__.py"])
    gt["adjudication_loc"] = _loc([REPO / "src/foundationscale/gates/adjudication.py"])
    gt["lsg_loc"] = _loc([REPO / "tools/live_save_gate.py"])
    gt["shim_names"], gt["shim_private"], gt["shim_public"] = _shim_names()
    gt["h100_files"], gt["h100_loc"] = _count_subtree(REPO / "h100_validation")

    lsh = sorted(
        p for p in (REPO / "launchers").rglob("*.sh") if not any(x in NOISE for x in p.parts)
    )
    gt["launch_sh_loc"] = _loc(lsh)
    gt["launch_py_loc"] = _loc(_py_files(REPO / "launchers"))
    gt["decisions_loc"] = _loc([REPO / "docs/DECISIONS.md"])

    # Repo-wide LOC is reported as SEVERAL keys, each naming its own method,
    # rather than one figure called "the repo size". The drafts carry a
    # repo-wide 125,697 that no method reproducible here reaches -- not .py
    # alone, not py+sh, not py+sh+md, and not any git-tracked variant of those
    # -- so its denominator is unrecoverable. Substituting a differently
    # derived number under the same word would hide that, so the old key is
    # deliberately NOT emitted: a claim whose method cannot be restated is
    # unmeasured, and unmeasured is a state, not a rounding error.
    def _ext_loc(ext: str) -> tuple[int, int]:
        ps = sorted(
            p
            for p in REPO.rglob(f"*{ext}")
            if not any(x in NOISE for x in p.relative_to(REPO).parts)
        )
        return len(ps), _loc(ps)

    gt["ondisk_py_files"], gt["ondisk_py_loc"] = _ext_loc(".py")
    gt["ondisk_sh_files"], gt["ondisk_sh_loc"] = _ext_loc(".sh")
    gt["ondisk_md_files"], gt["ondisk_md_loc"] = _ext_loc(".md")
    gt["ondisk_py_sh_loc"] = gt["ondisk_py_loc"] + gt["ondisk_sh_loc"]
    gt["repo_wide_loc_UNRECOVERABLE"] = (
        "prior drafts state 125,697 with no recorded method; no counting rule tried "
        "here reproduces it. Use the ondisk_* keys, which name their own denominator."
    )

    gt["imports_tests"] = _package_imports(REPO / "tests")
    gt["imports_tools"] = _package_imports(REPO / "tools")
    gt["imports_src"] = _package_imports(REPO / "src")

    gt.update(_mutation_corpus())
    if not args.no_coverage:
        gt.update(_coverage())

    args.out.write_text(json.dumps(gt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    for k in sorted(gt):
        print(f"  {k:22s} {gt[k]}")

    if args.check and args.check.exists() and args.check != args.out:
        prior = json.loads(args.check.read_text())
        # STABLE keys: subtrees D8 did not touch. A mismatch here indicts this
        # script's method, not the tree -- that is the positive control.
        stable = (
            "h100_files",
            "h100_loc",
            "launch_sh_loc",
            "launch_py_loc",
            "decisions_loc",
            "root_init_loc",
            "adjudication_loc",
            "lsg_loc",
            "shim_names",
            "shim_private",
            "shim_public",
        )
        bad = [(k, prior.get(k), gt.get(k)) for k in stable if prior.get(k) != gt.get(k)]
        print("\n--- method control (keys D8 could not have changed) ---")
        if bad:
            for k, was, now in bad:
                print(f"  MISMATCH {k}: prior {was} -> this run {now}")
            print(
                "  VERDICT: this census disagrees with the prior one on untouched "
                "subtrees, so its METHOD differs. Do not publish these numbers."
            )
            return 5
        print(f"  {len(stable)}/{len(stable)} stable keys reproduce exactly; method agrees.")
        print("\n--- keys that MOVED (expected: src/tests/suite, from D8) ---")
        for k in sorted(set(prior) | set(gt)):
            if prior.get(k) != gt.get(k):
                print(f"  {k}: {prior.get(k)} -> {gt.get(k)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
