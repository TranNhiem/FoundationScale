#!/usr/bin/env python3
"""Build SWE-smith-style synthetic bug-fix tasks for agentic RL.

For each library directory under --libs (containing <name>.py and test_<name>.py)
this script filters the tests to those that pass on the gold module, generates
single-site mutants, keeps mutants that break 1-5 tests while leaving at least 3
passing, and emits task contexts plus JSONL records.

Usage:
    python make_swe_tasks.py --libs ./libs --out ./tasks --per-lib 12 \
        --eval-libs graphs,units,lru --seed 0 --jobs 4
    python make_swe_tasks.py --libs ./libs --out ./tasks --build --jobs 4

The --build path additionally builds container images and validates tasks in
enroot sandboxes; it requires enroot and the foundationscale package.
"""

from __future__ import annotations

import argparse
import ast
import copy
import json
import os
import random
import re
import subprocess
import sys
import tempfile
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

FENCE = "`" * 3

SYSTEM = (
    "You are a software engineer working in a Linux shell in the repository at /repo "
    "(Python 3.12, no network). Use the bash tool to inspect files, run Python and edit "
    "code (for example with sed or a python heredoc). The project's test suite is hidden "
    "from you; you may write and run your own checks. Fix the bug with a minimal change "
    "to the library code, then call the submit tool with a one-line summary of the fix."
)

ISSUE_STYLE = "test"  # set from --issue in main()
TEST_CMD = "python -m pytest -q -rA -p no:cacheprovider --rootdir /tests /tests"

STATUS_MAP = {
    "PASSED": "passed",
    "XPASS": "passed",
    "FAILED": "failed",
    "ERROR": "error",
    "SKIPPED": "skipped",
    "XFAIL": "skipped",
}

COMPARE_SWAPS: dict[type[ast.cmpop], type[ast.cmpop]] = {
    ast.Lt: ast.LtE,
    ast.LtE: ast.Lt,
    ast.Gt: ast.GtE,
    ast.GtE: ast.Gt,
    ast.Eq: ast.NotEq,
    ast.NotEq: ast.Eq,
    ast.In: ast.NotIn,
    ast.NotIn: ast.In,
    ast.Is: ast.IsNot,
    ast.IsNot: ast.Is,
}

BINOP_SWAPS: dict[type[ast.operator], type[ast.operator]] = {
    ast.Add: ast.Sub,
    ast.Sub: ast.Add,
    ast.Mult: ast.Add,
    ast.FloorDiv: ast.Div,
    ast.Mod: ast.FloorDiv,
}

AUGASSIGN_SWAPS: dict[type[ast.operator], type[ast.operator]] = dict(BINOP_SWAPS)

OP_LABELS: dict[type[Any], str] = {
    ast.Lt: "Lt",
    ast.LtE: "LtE",
    ast.Gt: "Gt",
    ast.GtE: "GtE",
    ast.Eq: "Eq",
    ast.NotEq: "NotEq",
    ast.In: "In",
    ast.NotIn: "NotIn",
    ast.Is: "Is",
    ast.IsNot: "IsNot",
    ast.Add: "Add",
    ast.Sub: "Sub",
    ast.Mult: "Mult",
    ast.FloorDiv: "FloorDiv",
    ast.Div: "Div",
    ast.Mod: "Mod",
    ast.And: "And",
    ast.Or: "Or",
}

RESULT_RE = re.compile(r"^(PASSED|FAILED|ERROR|XPASS|XFAIL|SKIPPED)\s+(\S+?)(?:\s+-\s+.*)?$")


def parse_pytest_output(text: str) -> dict[str, str]:
    """Parse pytest -rA output into a node-id -> status mapping."""
    lines = text.splitlines()
    header = "short test summary info"
    start = 0
    for i, line in enumerate(lines):
        if line.strip().lower() == header:
            start = i + 1
            break
    results: dict[str, str] = {}
    for line in lines[start:]:
        match = RESULT_RE.match(line.strip())
        if not match:
            continue
        keyword, raw_id = match.groups()
        node_id = re.split(r"\s+-\s+|\s", raw_id, maxsplit=1)[0]
        status = STATUS_MAP[keyword]
        if status == "skipped" and "::" not in node_id:
            continue
        results[node_id] = status
    return results


def run_pytest(python: str, repo_dir: Path, tests_dir: Path, timeout_s: float) -> tuple[int, str]:
    """Run the host pytest command and return (returncode, stdout)."""
    env = dict(os.environ)
    env["PYTHONPATH"] = str(repo_dir)
    tests_dir = tests_dir.resolve()  # macOS /var -> /private/var would skew node ids
    cmd = [
        python,
        "-m",
        "pytest",
        "-q",
        "-rA",
        "-p",
        "no:cacheprovider",
        "--rootdir",
        str(tests_dir),
        str(tests_dir),
    ]
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(tests_dir),
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired:
        return -1, ""
    return proc.returncode, proc.stdout


def test_func_name(node_id: str) -> str:
    return node_id.rsplit("::", 1)[-1]


def filter_tests(src: str, keep_names: set[str]) -> str:
    """Drop top-level test functions whose names are not in keep_names."""
    tree = ast.parse(src)
    kept: list[ast.stmt] = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name not in keep_names:
            continue
        kept.append(node)
    tree.body = kept
    ast.fix_missing_locations(tree)
    return ast.unparse(tree)


def gold_check(
    name: str,
    module_src: str,
    tests_src: str,
    python: str,
) -> tuple[str, str] | None:
    """Filter tests to those passing on gold; return (gold_src, tests_src)."""
    with tempfile.TemporaryDirectory(prefix="swe-gold-") as tmp:
        root = Path(tmp)
        repo_dir = root / "repo"
        tests_dir = root / "tests"
        repo_dir.mkdir()
        tests_dir.mkdir()
        (repo_dir / f"{name}.py").write_text(module_src, encoding="utf-8")
        (tests_dir / f"test_{name}.py").write_text(tests_src, encoding="utf-8")

        rc, out = run_pytest(python, repo_dir, tests_dir, 60.0)
        results = parse_pytest_output(out)
        if rc not in (0, 1) or not results:
            print(f"skip {name}: collection error (rc={rc}, results={len(results)})")
            return None

        keep = {test_func_name(nid) for nid, status in results.items() if status == "passed"}
        filtered = filter_tests(tests_src, keep)
        (tests_dir / f"test_{name}.py").write_text(filtered, encoding="utf-8")

        rc2, out2 = run_pytest(python, repo_dir, tests_dir, 60.0)
        results2 = parse_pytest_output(out2)
        if rc2 not in (0, 1) or not results2:
            print(f"skip {name}: collection error after filtering (rc={rc2})")
            return None
        passed = [nid for nid, st in results2.items() if st == "passed"]
        if len(passed) < 8:
            print(f"skip {name}: fewer than 8 passing tests remain ({len(passed)})")
            return None
        return ast.unparse(ast.parse(module_src)), filtered


class SiteCollector(ast.NodeVisitor):
    """Collect mutation sites as (line, description, apply_fn) records."""

    def __init__(self) -> None:
        self.sites: list[tuple[int, str, Callable[[ast.AST], None]]] = []
        self.func_stack: list[str] = []
        self.func_sites: dict[str, list[int]] = {}

    def _func_name(self) -> str:
        return self.func_stack[-1] if self.func_stack else "<module>"

    def _add(self, node: ast.AST, desc: str, apply: Callable[[ast.AST], None]) -> None:
        line = getattr(node, "lineno", 0)
        fname = self._func_name()
        self.sites.append((line, f"line {line}: {desc} in function {fname}", apply))
        self.func_sites.setdefault(fname, []).append(len(self.sites) - 1)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self.func_stack.append(node.name)
        self.generic_visit(node)
        self.func_stack.pop()

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_Compare(self, node: ast.Compare) -> None:
        for i, op in enumerate(node.ops):
            swap = COMPARE_SWAPS.get(type(op))
            if swap is None:
                continue
            old, new = OP_LABELS[type(op)], OP_LABELS[swap]

            def apply(root: ast.AST, i: int = i, swap: type[ast.cmpop] = swap) -> None:
                target = _find_node(root, node)
                assert isinstance(target, ast.Compare)
                target.ops[i] = swap()

            self._add(node, f"{old} -> {new}", apply)
        self.generic_visit(node)

    def visit_BinOp(self, node: ast.BinOp) -> None:
        swap = BINOP_SWAPS.get(type(node.op))
        if swap is not None:
            old, new = OP_LABELS[type(node.op)], OP_LABELS[swap]

            def apply(root: ast.AST, swap: type[ast.operator] = swap) -> None:
                target = _find_node(root, node)
                assert isinstance(target, ast.BinOp)
                target.op = swap()

            self._add(node, f"{old} -> {new}", apply)
        self.generic_visit(node)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        swap = AUGASSIGN_SWAPS.get(type(node.op))
        if swap is not None:
            old, new = OP_LABELS[type(node.op)], OP_LABELS[swap]

            def apply(root: ast.AST, swap: type[ast.operator] = swap) -> None:
                target = _find_node(root, node)
                assert isinstance(target, ast.AugAssign)
                target.op = swap()

            self._add(node, f"{old} -> {new}", apply)
        self.generic_visit(node)

    def visit_BoolOp(self, node: ast.BoolOp) -> None:
        swap: type[ast.boolop] = ast.Or if isinstance(node.op, ast.And) else ast.And
        old, new = OP_LABELS[type(node.op)], OP_LABELS[swap]

        def apply(root: ast.AST, swap: type[ast.boolop] = swap) -> None:
            target = _find_node(root, node)
            assert isinstance(target, ast.BoolOp)
            target.op = swap()

        self._add(node, f"{old} -> {new}", apply)
        self.generic_visit(node)

    def visit_UnaryOp(self, node: ast.UnaryOp) -> None:
        if isinstance(node.op, ast.Not):

            def apply(root: ast.AST) -> None:
                _replace_node(root, node, node.operand)

            self._add(node, "remove Not", apply)
        self.generic_visit(node)

    def visit_Constant(self, node: ast.Constant) -> None:
        value = node.value
        if isinstance(value, int) and not isinstance(value, bool) and 0 <= value <= 10:

            def apply(root: ast.AST, old: int = value) -> None:
                target = _find_node(root, node)
                assert isinstance(target, ast.Constant)
                target.value = old + 1

            self._add(node, f"Constant {value} -> {value + 1}", apply)


# Position of every gold-tree node in ast.walk order; a deep copy walks in the same order, so a
# gold node's position finds its twin in the copy (deepcopy creates new objects, identity fails).
_POSITION: dict[int, int] = {}


def _find_node(root: ast.AST, target: ast.AST) -> ast.AST | None:
    position = _POSITION.get(id(target))
    if position is None:
        return None
    nodes = list(ast.walk(root))
    return nodes[position] if position < len(nodes) else None


def _replace_node(root: ast.AST, target: ast.AST, replacement: ast.AST) -> None:
    twin = _find_node(root, target)
    twin_replacement = _find_node(root, replacement)
    for node in ast.walk(root):
        for field, value in ast.iter_fields(node):
            if isinstance(value, list):
                for i, item in enumerate(value):
                    if item is twin:
                        value[i] = twin_replacement
                        return
            elif value is twin:
                setattr(node, field, twin_replacement)
                return


def is_docstring_context(root: ast.AST, target: ast.AST) -> bool:
    for node in ast.walk(root):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Module)):
            body = node.body
            if (
                body
                and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and body[0].value is target
            ):
                return True
    return False


def enumerate_mutants(gold_src: str) -> list[tuple[str, str]]:
    """Return (mutant_src, description) pairs for every one-site change."""
    gold_tree = ast.parse(gold_src)
    _POSITION.clear()
    _POSITION.update({id(node): k for k, node in enumerate(ast.walk(gold_tree))})
    collector = SiteCollector()
    collector.visit(gold_tree)

    # Drop mutations inside private functions only if that leaves none.
    public = [i for i, (_, _, _) in enumerate(collector.sites)]
    private_only = all(not any(not fn.startswith("_") for fn in collector.func_sites) for _ in [0])
    if not private_only:
        public = [
            i
            for i, (_, _, _) in enumerate(collector.sites)
            if any(
                not fn.startswith("_") and i in idxs for fn, idxs in collector.func_sites.items()
            )
        ]
    sites = [collector.sites[i] for i in public] or collector.sites

    out: list[tuple[str, str]] = []
    for _, desc, apply in sites:
        tree = copy.deepcopy(gold_tree)
        try:
            apply(tree)
        except Exception:
            continue
        ast.fix_missing_locations(tree)
        try:
            out.append((ast.unparse(tree), desc))
        except Exception:
            continue
    return out


def evaluate_mutants(
    name: str,
    gold_src: str,
    tests_src: str,
    mutants: list[tuple[str, str]],
    python: str,
    per_lib: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Run tests against mutants and select up to per_lib tasks."""
    del gold_src  # unused here; kept so every per-library helper takes the same inputs
    with tempfile.TemporaryDirectory(prefix="swe-mut-") as tmp:
        root = Path(tmp)
        repo_dir = root / "repo"
        tests_dir = root / "tests"
        repo_dir.mkdir()
        tests_dir.mkdir()
        (tests_dir / f"test_{name}.py").write_text(tests_src, encoding="utf-8")

        candidates: list[dict[str, Any]] = []
        seen_sets: set[tuple[str, ...]] = set()
        for mutant_src, desc in mutants:
            (repo_dir / f"{name}.py").write_text(mutant_src, encoding="utf-8")
            rc, out = run_pytest(python, repo_dir, tests_dir, 30.0)
            if rc == -1:
                continue
            results = parse_pytest_output(out)
            if not results:
                continue
            failing = sorted(nid for nid, st in results.items() if st in ("failed", "error"))
            passing = sorted(nid for nid, st in results.items() if st == "passed")
            if not (1 <= len(failing) <= 5) or len(passing) < 3:
                continue
            key = tuple(failing)
            if key in seen_sets:
                continue
            seen_sets.add(key)
            candidates.append(
                {
                    "src": mutant_src,
                    "desc": desc,
                    "fail_to_pass": failing,
                    "pass_to_pass": passing,
                }
            )

        rng = random.Random(seed)
        rng.shuffle(candidates)
        candidates.sort(key=lambda c: (len(c["fail_to_pass"]), c["desc"]))
        return candidates[:per_lib]


def combine_mutants(
    name: str,
    gold_src: str,
    tests_src: str,
    singles: list[dict[str, Any]],
    python: str,
    per_lib: int,
    seed: int,
    bugs: int,
) -> list[dict[str, Any]]:
    """Tasks with ``bugs`` one-site mutants in DIFFERENT functions, merged line-wise.

    Every single mutant is ``ast.unparse`` of the gold tree with one change, so two changes in
    different functions touch different lines of equal-length sources and merge line by line.
    Keeps combinations failing 2..8 tests with at least 3 passing.
    """
    gold_lines = gold_src.splitlines()
    rng = random.Random(seed)
    pool = [m for m in singles if len(m["src"].splitlines()) == len(gold_lines)]
    out: list[dict[str, Any]] = []
    seen: set[tuple[str, ...]] = set()
    attempts = 0
    with tempfile.TemporaryDirectory(prefix="swe-combo-") as tmp:
        root = Path(tmp)
        (root / "repo").mkdir()
        (root / "tests").mkdir()
        (root / "tests" / f"test_{name}.py").write_text(tests_src, encoding="utf-8")
        while len(out) < per_lib and attempts < per_lib * 20 and len(pool) >= bugs:
            attempts += 1
            picks = rng.sample(pool, bugs)
            funcs = [re.findall(r"in function (\w+)", m["desc"]) for m in picks]
            if len({f[0] if f else "" for f in funcs}) < bugs:
                continue
            merged = list(gold_lines)
            clash = False
            for m in picks:
                for i, (g, line) in enumerate(zip(gold_lines, m["src"].splitlines(), strict=True)):
                    if line != g:
                        if merged[i] != g:
                            clash = True
                        merged[i] = line
            if clash:
                continue
            src = "\n".join(merged) + "\n"
            (root / "repo" / f"{name}.py").write_text(src, encoding="utf-8")
            rc, text = run_pytest(python, root / "repo", root / "tests", 30.0)
            if rc == -1:
                continue
            results = parse_pytest_output(text)
            failing = sorted(n for n, st in results.items() if st in ("failed", "error"))
            passing = sorted(n for n, st in results.items() if st == "passed")
            if not (2 <= len(failing) <= 8) or len(passing) < 3 or tuple(failing) in seen:
                continue
            seen.add(tuple(failing))
            out.append({
                "src": src,
                "desc": " + ".join(m["desc"] for m in picks),
                "fail_to_pass": failing,
                "pass_to_pass": passing,
            })  # fmt: skip
    return out


def build_vague_issue(name: str, desc: str) -> str:
    funcs = sorted(set(re.findall(r"in function (\w+)", desc)))
    named = ", ".join(f"`{f}()`" for f in funcs) or "some functions"
    return (
        f"Bug report for `{name}.py`: users report that {named} sometimes return wrong results "
        f"or raise errors for valid input. There may be more than one bug. Our test suite is "
        f"hidden from you; reproduce the problem yourself (write small checks with python), then "
        f"find and fix the bug(s) in the library (do not edit tests)."
    )


def build_issue(name: str, tests_src: str, failing_id: str) -> str:
    tree = ast.parse(tests_src)
    func_name = test_func_name(failing_id)
    source = ""
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == func_name:
            source = ast.get_source_segment(tests_src, node) or ""
            break
    return (
        f"Bug report for `{name}.py`: a behaviour is wrong. This check from our test "
        f"suite fails:\n\n{FENCE}python\n{source}\n{FENCE}\n\nFind and fix the bug in "
        f"the library (do not edit tests)."
    )


def write_task(
    out_root: Path,
    name: str,
    index: int,
    mutant_src: str,
    gold_src: str,
    tests_src: str,
    desc: str,
    fail_to_pass: list[str],
    pass_to_pass: list[str],
    validated: str,
) -> dict[str, Any]:
    uid = f"{name}-{index:03d}"
    ctx = out_root / "ctx" / uid
    repo_dir = ctx / "repo"
    tests_dir = ctx / "tests"
    gold_dir = out_root / "gold" / uid
    repo_dir.mkdir(parents=True, exist_ok=True)
    tests_dir.mkdir(parents=True, exist_ok=True)
    gold_dir.mkdir(parents=True, exist_ok=True)

    (repo_dir / f"{name}.py").write_text(mutant_src, encoding="utf-8")
    (tests_dir / f"test_{name}.py").write_text(tests_src, encoding="utf-8")
    (gold_dir / f"{name}.py").write_text(gold_src, encoding="utf-8")

    dockerfile = (
        "FROM python:3.12-slim\n"
        "RUN pip install -q pytest\n"
        "ENV PYTHONPATH=/repo\n"
        "WORKDIR /repo\n"
        "COPY repo/ /repo\n"
        "COPY tests/ /tests\n"
    )
    (ctx / "Dockerfile").write_text(dockerfile, encoding="utf-8")

    issue = (
        build_vague_issue(name, desc)
        if ISSUE_STYLE == "vague"
        else build_issue(name, tests_src, fail_to_pass[0])
    )
    image = str((out_root / "images" / f"{uid}.sqsh").resolve())
    return {
        "uid": uid,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": issue},
        ],
        "metadata": {
            "image": image,
            "workdir": "/repo",
            "test_cmd": TEST_CMD,
            "hidden_paths": json.dumps(["/tests"]),
            "test_timeout_s": "300",
            "fail_to_pass": json.dumps(fail_to_pass),
            "pass_to_pass": json.dumps(pass_to_pass),
            "source": "synthetic",
            "lib": name,
            "mutation": desc,
            "validated": validated,
        },
    }


def build_and_validate(
    task: dict[str, Any],
    out_root: Path,
    name: str,
    gold_src: str,
    roots: Any,
) -> dict[str, Any] | None:
    del gold_src  # unused here; kept so every per-library helper takes the same inputs
    from foundationscale.agentic_rl.envs.sandbox_task import parse_pytest_report
    from foundationscale.agentic_rl.sandbox.build import build_image
    from foundationscale.agentic_rl.sandbox.enroot import EnrootSandbox, SandboxSpec

    uid = task["uid"]
    ctx = out_root / "ctx" / uid
    sqsh = Path(task["metadata"]["image"])
    gold_file = out_root / "gold" / uid / f"{name}.py"

    if not sqsh.exists():
        build_image(
            str(ctx / "Dockerfile"),
            str(ctx),
            str(sqsh),
            data_root=roots.data_root,
            image_store=roots.image_store,
            scratch_root=roots.scratch_root,
            enroot_sysconf=roots.enroot_sysconf,
        )

    spec = SandboxSpec(
        name="fsar-build-" + uid,
        image=str(sqsh),
        data_root=roots.data_root,
        image_store=roots.image_store,
        scratch_root=roots.scratch_root,
        network="no-network",
        workdir="/repo",
        enroot_sysconf=roots.enroot_sysconf,
        isolate_network=True,
    )
    sb = EnrootSandbox(spec)
    try:
        sb.start()
        bug_res = sb.exec(TEST_CMD, cwd="/repo", timeout_s=300)
        bug = parse_pytest_report(bug_res.stdout)
        sb.upload_file(str(gold_file), f"/repo/{name}.py")
        gold_res = sb.exec(TEST_CMD, cwd="/repo", timeout_s=300)
        gold = parse_pytest_report(gold_res.stdout)
    finally:
        sb.stop(delete=True)

    fail_to_pass = sorted(nid for nid in bug if bug[nid] != "passed" and gold.get(nid) == "passed")
    pass_to_pass = sorted(nid for nid in bug if bug[nid] == "passed" and gold.get(nid) == "passed")
    if not fail_to_pass:
        print(f"drop {uid}: no failing tests after sandbox validation")
        return None
    if any(gold.get(nid) != "passed" for nid in fail_to_pass):
        print(f"drop {uid}: gold does not pass all failing tests")
        return None

    task["metadata"]["fail_to_pass"] = json.dumps(fail_to_pass)
    task["metadata"]["pass_to_pass"] = json.dumps(pass_to_pass)
    task["metadata"]["validated"] = "sandbox"
    return task


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build SWE-smith-style synthetic bug-fix tasks.")
    parser.add_argument("--libs", type=Path, required=True, help="library root directory")
    parser.add_argument("--out", type=Path, required=True, help="output directory")
    parser.add_argument("--per-lib", type=int, default=12, help="tasks per library")
    parser.add_argument(
        "--eval-libs",
        default="graphs,units,lru",
        help="comma-separated libraries held out for eval",
    )
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument(
        "--build",
        action="store_true",
        help="build images and validate in sandboxes (needs enroot + foundationscale)",
    )
    parser.add_argument("--jobs", type=int, default=4, help="parallel builds")
    parser.add_argument("--bugs-per-task", type=int, default=1, help="mutants per task (1 or 2+)")
    parser.add_argument(
        "--issue",
        choices=["test", "vague"],
        default="test",
        help="test: the issue shows one failing test; vague: it names only the buggy functions",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="interpreter for host pytest runs (must have pytest)",
    )
    args = parser.parse_args(argv)
    global ISSUE_STYLE
    ISSUE_STYLE = args.issue

    out_root: Path = args.out
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "images").mkdir(parents=True, exist_ok=True)
    eval_libs = {s.strip() for s in args.eval_libs.split(",") if s.strip()}

    train: list[dict[str, Any]] = []
    evals: list[dict[str, Any]] = []
    counts: dict[str, int] = {}

    for lib_dir in sorted(p for p in args.libs.iterdir() if p.is_dir()):
        name = lib_dir.name
        module_path = lib_dir / f"{name}.py"
        tests_path = lib_dir / f"test_{name}.py"
        if not module_path.exists() or not tests_path.exists():
            print(f"skip {name}: missing {name}.py or test_{name}.py")
            continue

        module_src = module_path.read_text(encoding="utf-8")
        tests_src = tests_path.read_text(encoding="utf-8")
        checked = gold_check(name, module_src, tests_src, args.python)
        if checked is None:
            continue
        gold_src, filtered_tests = checked

        mutants = enumerate_mutants(gold_src)
        if not mutants:
            print(f"skip {name}: no mutants generated")
            continue

        if args.bugs_per_task > 1:
            singles = evaluate_mutants(
                name, gold_src, filtered_tests, mutants, args.python, 10**6, args.seed
            )
            selected = combine_mutants(
                name,
                gold_src,
                filtered_tests,
                singles,
                args.python,
                args.per_lib,
                args.seed,
                args.bugs_per_task,
            )
        else:
            selected = evaluate_mutants(
                name, gold_src, filtered_tests, mutants, args.python, args.per_lib, args.seed
            )
        counts[name] = 0
        for i, cand in enumerate(selected, start=1):
            task = write_task(
                out_root,
                name,
                i,
                cand["src"],
                gold_src,
                filtered_tests,
                cand["desc"],
                cand["fail_to_pass"],
                cand["pass_to_pass"],
                "host",
            )
            (evals if name in eval_libs else train).append(task)
            counts[name] += 1

    if args.build:
        from foundationscale.agentic_rl.envs.sandbox_task import SandboxRoots

        roots = SandboxRoots.from_environ(os.environ)
        all_tasks = train + evals
        surviving: list[dict[str, Any]] = []
        with ThreadPoolExecutor(max_workers=args.jobs) as pool:
            futures = {
                pool.submit(
                    build_and_validate,
                    task,
                    out_root,
                    task["metadata"]["lib"],
                    (out_root / "gold" / task["uid"] / f"{task['metadata']['lib']}.py").read_text(
                        encoding="utf-8"
                    ),
                    roots,
                ): task
                for task in all_tasks
            }
            for fut in as_completed(futures):
                result = fut.result()
                if result is not None:
                    surviving.append(result)
        surviving.sort(key=lambda t: t["uid"])
        train = [t for t in surviving if t["metadata"]["lib"] not in eval_libs]
        evals = [t for t in surviving if t["metadata"]["lib"] in eval_libs]
        counts = {}
        for t in surviving:
            counts[t["metadata"]["lib"]] = counts.get(t["metadata"]["lib"], 0) + 1

    with (out_root / "swe_train.jsonl").open("w", encoding="utf-8") as fh:
        for task in train:
            fh.write(json.dumps(task) + "\n")
    with (out_root / "swe_eval.jsonl").open("w", encoding="utf-8") as fh:
        for task in evals:
            fh.write(json.dumps(task) + "\n")

    for name in sorted(counts):
        print(f"{name}: {counts[name]} tasks")
    print(f"train: {len(train)}  eval: {len(evals)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
