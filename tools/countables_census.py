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
#
# #244: this set is not hygiene, it is a DEFINITION of "the repository", and it
# was the wrong kind of definition. It listed what to exclude, so it could only
# ever be as complete as the last machine someone ran it on: `build` was absent,
# one `pip install .` had left a byte-for-byte second copy of the package in
# build/lib/foundationscale/, and the repo-wide keys silently counted the
# package twice. CI's fresh checkout had no build/ and disagreed by exactly
# src_loc. A blocklist cannot be finished -- .codegraph/, artifacts/, htmlcov/,
# a downloaded dataset and the next tool's cache are all one command away.
#
# The repository already publishes an exact, machine-independent statement of
# what it contains: the git index. TRACKED membership is now the denominator and
# NOISE is only a second guard, kept so the method still refuses residue if a
# junk directory is ever committed by accident.
NOISE = {
    "__pycache__",
    ".venv",
    "venv",
    ".git",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".tox",
    ".eggs",
    ".codegraph",
    "build",
    "dist",
    "htmlcov",
    "site-packages",
    "node_modules",
}


def _tracked() -> set[Path]:
    """Every path in the git index, absolute. REFUSEs rather than guessing.

    Falling back to a bare filesystem walk when git is unavailable would put the
    census back on the blocklist without saying so -- the same number under the
    same key derived by a different method, which is the drift this file exists
    to prevent. A census that cannot name its denominator does not ship one.
    """
    r = subprocess.run(
        ["git", "-C", str(REPO), "ls-files", "-z"],
        capture_output=True,
        text=True,
        check=False,
    )
    if r.returncode != 0:
        raise SystemExit(
            f"REFUSE: `git ls-files` failed in {REPO} (rc={r.returncode}). The census "
            "measures the git index, not the working directory; without it there is "
            "no machine-independent denominator. See #244."
        )
    return {REPO / p for p in r.stdout.split("\0") if p}


_TRACKED: set[Path] | None = None


def _in_repo(p: Path) -> bool:
    """Tracked by git AND not residue. Both halves, in that order."""
    assert _TRACKED is not None, "call _tracked() into _TRACKED before measuring"
    return p in _TRACKED and not any(part in NOISE for part in p.parts)


def _files(root: Path, ext: str) -> list[Path]:
    return sorted(p for p in root.rglob(f"*{ext}") if _in_repo(p))


def _py_files(root: Path) -> list[Path]:
    return _files(root, ".py")


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


def self_test() -> int:
    """Controls for the #244 denominator: a stray build tree must not count.

    Two legs, and the second is the one that makes the first mean anything. A
    walker that returns nothing at all also "does not count the stray copy" --
    that PASS would be vacuous. So the MUST_MOVE leg plants a file that IS part
    of the repository and asserts the number moves by exactly its length.
    """
    global _TRACKED
    saved = _TRACKED
    ok = True
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        (root / "pkg").mkdir()
        (root / "pkg/a.py").write_text("x = 1\n" * 10, encoding="utf-8")
        (root / "doc.md").write_text("# t\n" * 5, encoding="utf-8")
        real = [root / "pkg/a.py", root / "doc.md"]
        _TRACKED = set(real)

        base_py = _loc(_files(root, ".py"))
        base_md = _loc(_files(root, ".md"))

        # Residue, in every shape that has actually appeared on a real machine:
        # the pip build tree (#244's live defect), a wheel dir, a coverage
        # report, a virtualenv's vendored packages, a code index.
        for rel, n in (
            ("build/lib/pkg/a.py", 10),
            ("dist/pkg/a.py", 10),
            ("htmlcov/x.md", 5),
            (".venv/lib/site-packages/dep.py", 400),
            (".codegraph/idx.py", 77),
            ("__pycache__/a.py", 10),
        ):
            f = root / rel
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_text("y = 2\n" * n, encoding="utf-8")

        legs: list[tuple[str, str, bool]] = [
            (
                "stray build/dist/htmlcov/.venv/.codegraph do not move .py",
                "MUST_PASS",
                _loc(_files(root, ".py")) == base_py,
            ),
            (
                "stray htmlcov/*.md does not move .md",
                "MUST_PASS",
                _loc(_files(root, ".md")) == base_md,
            ),
        ]

        # The positive control: plant a file the index DOES carry and require
        # the count to move by exactly its length. Without this leg the two
        # MUST_PASSes above are satisfied by a walker that measures nothing.
        (root / "pkg/b.py").write_text("z = 3\n" * 7, encoding="utf-8")
        _TRACKED = set(real) | {root / "pkg/b.py"}
        legs.append(
            (
                "a tracked file DOES move .py, by exactly its length",
                "MUST_MOVE",
                _loc(_files(root, ".py")) == base_py + 7,
            )
        )
        # And residue that is somehow in the index is still residue: NOISE is
        # the second guard, and a guard that never fires is not a guard.
        _TRACKED = set(real) | {root / "build/lib/pkg/a.py"}
        legs.append(
            (
                "an INDEXED build/ path is still refused by NOISE",
                "MUST_PASS",
                _loc(_files(root, ".py")) == base_py,
            )
        )

        for name, kind, passed in legs:
            print(f"  [{'OK ' if passed else 'FAIL'}] {kind:9s} {name}")
            ok &= passed
        n = len(legs)

    _TRACKED = saved
    print(f"census self-test: {n} controls, {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 5


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", type=Path, default=None, help="prior census to diff against")
    # Required with no default: a census writer with a default output path
    # silently overwrites someone's file.
    ap.add_argument("--out", type=Path, required=False)
    ap.add_argument("--no-coverage", action="store_true", help="skip the pytest+coverage run")
    ap.add_argument(
        "--self-test",
        action="store_true",
        help="run the #244 denominator controls and exit (no census is written)",
    )
    args = ap.parse_args()

    if args.self_test:
        return self_test()
    if args.out is None:
        ap.error("--out is required unless --self-test is given")

    global _TRACKED
    _TRACKED = _tracked()

    gt: dict[str, object] = {}
    gt["src_files"], gt["src_loc"] = _count_subtree(REPO / "src/foundationscale")
    gt["tests_files"], gt["tests_loc"] = _count_subtree(REPO / "tests")
    gt["tools_files"], gt["tools_loc"] = _count_subtree(REPO / "tools")
    # #241: checks/ is stated as an exclusion denominator in the Makefile and in
    # ci.yml ("N of N files under checks/ are unchecked"). Both said "3 of 3" in
    # the same commit that added the fourth file, because the clause copied
    # mypy's "Found 10 errors in 3 files (checked 4 source files)" and took the
    # ERROR-BEARING count for the denominator. A directory that appears in a
    # claim needs a census key, or the claim sits in nothing.
    gt["checks_files"], gt["checks_loc"] = _count_subtree(REPO / "checks")
    gt["gates_files"] = len(_py_files(REPO / "src/foundationscale/gates"))
    gt["root_init_loc"] = _loc([REPO / "src/foundationscale/__init__.py"])
    gt["adjudication_loc"] = _loc([REPO / "src/foundationscale/gates/adjudication.py"])
    gt["lsg_loc"] = _loc([REPO / "tools/live_save_gate.py"])
    gt["shim_names"], gt["shim_private"], gt["shim_public"] = _shim_names()
    gt["h100_files"], gt["h100_loc"] = _count_subtree(REPO / "h100_validation")

    lsh = _files(REPO / "launchers", ".sh")
    gt["launch_sh_files"] = len(lsh)
    gt["launch_sh_loc"] = _loc(lsh)
    gt["launch_py_loc"] = _loc(_py_files(REPO / "launchers"))
    # #243: h100_files/h100_loc are .py ONLY, but the review corpus states the
    # harness as "N Python LOC and M shell LOC" in the same clause. The shell
    # half had no key, so half of a two-number claim sat in nothing -- and the
    # half that IS anchored stayed correct while the unanchored half went
    # stale three lines away, with the gate reporting CLEAR.
    hsh = _files(REPO / "h100_validation", ".sh")
    gt["h100_sh_files"] = len(hsh)
    gt["h100_sh_loc"] = _loc(hsh)
    gt["decisions_loc"] = _loc([REPO / "docs/DECISIONS.md"])

    # Repo-wide LOC is reported as SEVERAL keys, each naming its own method,
    # rather than one figure called "the repo size". The drafts carry a
    # repo-wide 125,697 that no method reproducible here reaches -- not .py
    # alone, not py+sh, not py+sh+md, and not any git-tracked variant of those
    # -- so its denominator is unrecoverable. Substituting a differently
    # derived number under the same word would hide that, so the old key is
    # deliberately NOT emitted: a claim whose method cannot be restated is
    # unmeasured, and unmeasured is a state, not a rounding error.
    # #244: these were named ondisk_* and walked the working directory minus a
    # blocklist, so each one stated a property of the MACHINE. They are now
    # git-tracked, and RENAMED to say so -- a method change under an unchanged
    # key is exactly the failure this file exists to prevent, and renaming
    # forces every document that states one of these numbers to restate it in
    # words the gate must re-anchor.
    def _ext_loc(ext: str) -> tuple[int, int]:
        ps = _files(REPO, ext)
        return len(ps), _loc(ps)

    gt["tracked_py_files"], gt["tracked_py_loc"] = _ext_loc(".py")
    gt["tracked_sh_files"], gt["tracked_sh_loc"] = _ext_loc(".sh")
    gt["tracked_md_files"], gt["tracked_md_loc"] = _ext_loc(".md")
    gt["tracked_py_sh_loc"] = gt["tracked_py_loc"] + gt["tracked_sh_loc"]
    # The review corpus says "N ... lines repo-wide" -- a THIRD method again,
    # distinct from the two above. It gets its own key rather than being
    # approximated by tracked_py_sh_loc, because the whole point of this family
    # is that each name states its own denominator.
    #
    # This key counts .md, so the document that STATES it is inside it. That is
    # a fixed point only because the unit is the LINE and a token rewrite
    # preserves line count, so `--fix` converges in one pass. Re-express it in
    # characters or bytes and it never converges: correcting the number changes
    # the number.
    gt["tracked_py_sh_md_loc"] = gt["tracked_py_sh_loc"] + gt["tracked_md_loc"]
    gt["repo_wide_loc_UNRECOVERABLE"] = (
        "prior drafts state 125,697 with no recorded method; no counting rule tried "
        "here reproduces it. Use the tracked_* keys, which name their own denominator."
    )
    # The method and its exclusion set, emitted so the CONSUMER can check them.
    # checks/countables_drift.py holds its own EXCLUDED_PARTS for deciding which
    # documents to scan; before #244 the two lists differed and neither could
    # see the other, which is how a whole duplicated package entered a published
    # number in silence. The gate now refuses unless its list is a subset of
    # this one, so a future divergence is a verdict instead of a disagreement.
    gt["census_method"] = "git-tracked"
    gt["excluded_parts"] = sorted(NOISE)

    gt["imports_tests"] = _package_imports(REPO / "tests")
    gt["imports_tools"] = _package_imports(REPO / "tools")
    gt["imports_src"] = _package_imports(REPO / "src")
    # "tests = N of M total import statements" states a share, and a share needs
    # BOTH halves in a denominator or the fraction can rot from underneath. The
    # total is the sum of the three trees scanned above and nothing else --
    # named as such, not as "the repo's imports", which would be a fourth
    # method wearing the same word.
    gt["imports_total"] = gt["imports_tests"] + gt["imports_tools"] + gt["imports_src"]

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
