"""Gate: every published Python stage must exit with a declared contract code.

The trap this gate detects: `raise SystemExit("some message")` and
`sys.exit("some message")` print the message to stderr and exit **1**. A stage
that intends REFUSE -- a required input unset, no default by design -- and writes
`raise SystemExit("...required input missing...")` silently exits 1 instead of 96.
A consumer then cannot tell REFUSE from RED from an unhandled crash: the contract
collapses into noise, and every downstream retry/skip decision built on it is
arbitrary.

This was not unknown. Two files in this tree carry a docstring warning describing
exactly this trap, and twelve call sites in their sibling files violated it anyway.
A lesson recorded in prose next to the code is not a control: prose is read by the
person who already suspects the problem, and skipped by everyone else. Knowledge
that exists but is not enforced does not survive contact with the next editor. So
the rule is enforced here, mechanically, over the declared publish set, on every
build.

A count with no denominator is not a measurement either, so the gate prints how
many files it scanned and how many exit sites it judged, and it refuses to report
"clean" against a shrunken or undeclared file list.

EXIT CODES: 0 clean, 5 at least one contract violation found, 4 the gate failed
its own must-fire/must-pass controls, 95 measurement impossible (publish set
unreadable or fewer than 8 Python files resolve).
"""

from __future__ import annotations

import argparse
import ast
import sys
from dataclasses import dataclass, field
from pathlib import Path

CONTRACT_CODES: frozenset[int] = frozenset({0, 5, 95, 96})
MIN_DENOMINATOR = 8
PUBLISH_SET_REL = Path("h100") / "PUBLISH_SET.txt"


@dataclass
class Finding:
    path: str
    lineno: int
    reason: str
    segment: str
    is_message_defect: bool  # the string-message trap; wrong-integer findings use False

    def render(self) -> str:
        return f"{self.path}:{self.lineno}: {self.reason}  [{self.segment[:60]}]"


@dataclass
class ScanResult:
    sites: int = 0
    rejections: list[Finding] = field(default_factory=list)
    unjudged: list[tuple[str, int, str]] = field(default_factory=list)


def _leftmost_is_string(node: ast.expr) -> bool:
    # For `+` / `%` message construction the decisive operand is the leftmost
    # one: "…" + detail and "…" % detail both produce a string exit message.
    #
    # ast.JoinedStr belongs here as much as ast.Constant does, and leaving it out
    # is not a small omission: on the real pre-fix tree, 3 of the 5 published
    # violations were written as an f-string (or an implicitly-concatenated
    # str+f-string, which Python folds into ONE JoinedStr) followed by `+`. All
    # three fell through to UNJUDGED and the gate reported them as unrejected.
    # The two it did catch were the two written as a lone f-string. A control set
    # built only from the textbook shape measures the textbook, not the tree.
    while isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Mod)):
        node = node.left
    if isinstance(node, ast.JoinedStr):
        return True
    return isinstance(node, ast.Constant) and isinstance(node.value, str)


def _classify_arg(arg: ast.expr) -> str | None:
    """Return a rejection reason, or None meaning UNJUDGED/accepted-by-value.

    Callers distinguish accepted-by-value from UNJUDGED via _literal_value;
    this function only answers "definitely a defect?"
    """
    if isinstance(arg, ast.Constant):
        if isinstance(arg.value, str):
            return "string exit message: exits 1, not a declared contract code (use 96 for REFUSE)"
        if isinstance(arg.value, bool):
            return f"boolean exit code {arg.value!r}: not in {{0, 5, 95, 96}}"
        if isinstance(arg.value, int):
            if arg.value in CONTRACT_CODES:
                return ""  # accepted literal
            return f"integer exit code {arg.value} is not in the declared contract {{0, 5, 95, 96}}"
        return None
    if isinstance(arg, ast.JoinedStr):
        return "f-string exit message: exits 1, not a declared contract code (use 96 for REFUSE)"
    if isinstance(arg, ast.UnaryOp) and isinstance(arg.op, ast.USub) and isinstance(
        arg.operand, ast.Constant
    ) and isinstance(arg.operand.value, int) and not isinstance(arg.operand.value, bool):
        value = -arg.operand.value
        if value in CONTRACT_CODES:
            return ""
        return f"integer exit code {value} is not in the declared contract {{0, 5, 95, 96}}"
    if isinstance(arg, ast.BinOp) and isinstance(arg.op, (ast.Add, ast.Mod)):
        if _leftmost_is_string(arg):
            return "string-built exit message: exits 1, not a declared contract code (use 96 for REFUSE)"
        return None
    if (
        isinstance(arg, ast.Call)
        and isinstance(arg.func, ast.Attribute)
        and arg.func.attr == "format"
        and isinstance(arg.func.value, ast.Constant)
        and isinstance(arg.func.value.value, str)
    ):
        return "str.format exit message: exits 1, not a declared contract code (use 96 for REFUSE)"
    # Name / Attribute / Call / anything else: may resolve to an int at runtime
    # (e.g. a variable holding a return code). Not statically decidable.
    return None


def check_source(source: str, path: str) -> ScanResult:
    """Run the contract over one source text; shared by the scan and the controls."""
    result = ScanResult()
    try:
        tree = ast.parse(source, filename=path)
    except SyntaxError as exc:  # fail closed: a file we cannot parse is not "clean"
        result.sites += 1
        result.rejections.append(
            Finding(path, exc.lineno or 0, "file does not parse, exit behavior unverifiable", "", False)
        )
        return result

    def judge(call_or_raise: ast.AST, args: list[ast.expr]) -> None:
        result.sites += 1
        lineno = getattr(call_or_raise, "lineno", 0)
        segment = ast.get_source_segment(source, call_or_raise) or ""
        if not args:
            return  # raise SystemExit / sys.exit() exits 0, which is in contract
        if len(args) > 1:
            result.unjudged.append((path, lineno, segment))
            return
        verdict = _classify_arg(args[0])
        if verdict == "":
            return  # accepted literal contract code
        if verdict is not None:
            is_message = "message" in verdict or "string" in verdict
            result.rejections.append(Finding(path, lineno, verdict, segment, is_message))
            return
        result.unjudged.append((path, lineno, segment))

    for node in ast.walk(tree):
        if isinstance(node, ast.Raise) and node.exc is not None:
            exc = node.exc
            if isinstance(exc, ast.Name) and exc.id == "SystemExit":
                judge(node, [])  # bare `raise SystemExit` exits 0
            elif (
                isinstance(exc, ast.Call)
                and isinstance(exc.func, ast.Name)
                and exc.func.id == "SystemExit"
            ):
                judge(exc, exc.args)
        elif isinstance(node, ast.Call):
            func = node.func
            is_sys_exit = (
                isinstance(func, ast.Attribute)
                and func.attr == "exit"
                and isinstance(func.value, ast.Name)
                and func.value.id == "sys"
            )
            is_bare_exit = isinstance(func, ast.Name) and func.id == "exit"
            if is_sys_exit or is_bare_exit:
                judge(node, node.args)
    return result


def read_publish_set(root: Path) -> list[Path] | None:
    """Resolve the declared publish set to existing .py files, or None if undeclared.

    No glob fallback: if the denominator is undeclared, no scan can claim to
    cover it, and pretending to measure a self-selected subset is how a clean
    result gets manufactured.
    """
    listing = root / PUBLISH_SET_REL
    try:
        text = listing.read_text(encoding="utf-8")
    except OSError:
        return None
    resolved: list[Path] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("./"):
            line = line[2:]
        if not line.endswith(".py"):
            continue
        candidate = root / line
        if candidate.is_file():  # only resolvable entries count toward the denominator
            resolved.append(candidate)
    return resolved


def run_controls() -> str | None:
    """Prove the checker fires on the defects it exists for and passes the legal forms.

    A gate that cannot catch its own target defect on a trivial input cannot be
    trusted on the real tree, so neither can the build that ran it.
    """
    msg_defect = check_source('raise SystemExit("required input missing")\n', "<control:msg>")
    if len(msg_defect.rejections) != 1 or msg_defect.sites != 1:
        return "must-fire control failed: string SystemExit message did not produce exactly one rejection"
    int_defect = check_source("import sys\nsys.exit(1)\n", "<control:int>")
    if len(int_defect.rejections) != 1 or int_defect.sites != 1:
        return "must-fire control 2 failed: sys.exit(1) did not produce exactly one rejection"
    # The two shapes actually found in this tree, and the two the first draft of this
    # gate let through as UNJUDGED. They are controls, not extra coverage: without them
    # the gate is green on 3 of the 5 violations it was written to find.
    fstr_concat = check_source(
        'x = 1\nraise SystemExit(\n    f"missing {x}\\n"\n    + "  supply it\\n"\n)\n',
        "<control:fstr+>",
    )
    if len(fstr_concat.rejections) != 1 or fstr_concat.unjudged:
        return "must-fire control 3 failed: f-string + str concatenation was not rejected"
    implicit = check_source(
        'x = 1\nraise SystemExit(\n    "missing "\n    f"{x}\\n"\n    + "  supply it\\n"\n)\n',
        "<control:implicit>",
    )
    if len(implicit.rejections) != 1 or implicit.unjudged:
        return (
            "must-fire control 4 failed: implicitly concatenated str+f-string (one JoinedStr) "
            "followed by + was not rejected"
        )
    legal = check_source(
        "import sys\nraise SystemExit(96)\nsys.exit(0)\nraise SystemExit(rc)\n",
        "<control:pass>",
    )
    if legal.rejections or len(legal.unjudged) != 1 or legal.sites != 3:
        return (
            "must-pass control failed: SystemExit(96)/sys.exit(0)/SystemExit(rc) must give "
            "zero rejections and exactly one unjudged entry"
        )
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="build or publish tree root containing h100/PUBLISH_SET.txt "
        "(default: directory containing this script)",
    )
    args = parser.parse_args()
    root: Path = args.root

    control_error = run_controls()
    if control_error is not None:
        print(f"CONTROLS FAILED: {control_error}; the gate cannot be trusted, so neither can the build")
        return 4

    files = read_publish_set(root)
    if files is None:
        print(
            f"UNMEASURED: publish set {PUBLISH_SET_REL} is unreadable under {root}; "
            "the denominator is undeclared so no scan can claim to cover it"
        )
        return 95
    if len(files) < MIN_DENOMINATOR:
        print(
            f"UNMEASURED: only {len(files)} Python files resolve from the publish set "
            f"(floor is {MIN_DENOMINATOR}); a shrunken denominator reads exactly like a clean scan"
        )
        return 95

    total = ScanResult()
    for path in files:
        try:
            source = path.read_text(encoding="utf-8")
        except OSError as exc:
            total.sites += 1
            total.rejections.append(
                Finding(str(path), 0, f"file became unreadable during scan: {exc}", "", False)
            )
            continue
        sub = check_source(source, str(path.relative_to(root) if path.is_relative_to(root) else path))
        total.sites += sub.sites
        total.rejections.extend(sub.rejections)
        total.unjudged.extend(sub.unjudged)

    # Most-informative first: the string-message defects are the trap this gate
    # exists for, then wrong-integer findings, each stable by path and line.
    total.rejections.sort(key=lambda f: (not f.is_message_defect, f.path, f.lineno))
    for finding in total.rejections:
        print(finding.render())
    for path, lineno, segment in total.unjudged:
        print(f"UNJUDGED {path}:{lineno}: {segment[:60]}")

    print(
        f"scanned {len(files)} files, {total.sites} exit sites, "
        f"{len(total.rejections)} rejected, {len(total.unjudged)} unjudged"
    )
    return 5 if total.rejections else 0


if __name__ == "__main__":
    # `main()` here is intentionally UNJUDGED under the rule above (a Call that
    # may yield any int at runtime). That is the correct bucket: the gate's own
    # exit code is decided by main(), not by a literal, and the contract is
    # enforced by main()'s return values. Do not "fix" this into a literal.
    raise SystemExit(main())