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

Literal-only judging left the denominator hollow anyway: the universal idiom here
is `raise SystemExit(main())` / `sys.exit(main())`, whose argument is a Call, so
59 of 71 exit sites landed in UNJUDGED and every `return N` inside those entry
functions sat in NO denominator. The gate now follows that one hop: when an exit
site's sole argument is a zero-argument call to a locally defined name, the
literal returns inside that function are judged against the contract. The hop
surfaces 215 returns, of which 43 returned 1/2/3/4 -- outside {0, 5, 95, 96} --
and one of them was this gate's own `return 4`: the enforcer could not see
itself. An invisible self-defect is the strongest argument for the widening.

The hop still judged only literal returns, and 85 entry-function returns sat
UNJUDGED behind it: 48 returning a Call, 27 a Name, 7 an IfExp, 2 a Subscript,
1 an Attribute. Sixty-four of the 85 settle statically, and 30 of those return
a code outside {0, 5, 95, 96} -- a finding about the SYSTEM that the INSTRUMENT
was blind to. The gate now resolves each return to the SET of integers it can
evaluate to: a module-level constant name, a conditional whose branches both
resolve, one hop into a module-level helper's returns. A name bound twice, a
call to a function not defined in the module, a second hop -- none of these
settle, and the gate records UNJUDGED rather than guess. The exit SITES stay
unjudged either way: `raise SystemExit(main())` may yield any int at runtime,
and judging the site rather than the returns would claim the site decides the
code when it does not. That is a limit of the INSTRUMENT, honestly counted,
not a defect in the SYSTEM.

EXIT CODES: 0 clean, 5 at least one contract violation found, 95 measurement
impossible (publish set unreadable or fewer than 8 Python files resolve), 96 the
gate failed its own must-fire/must-pass controls and cannot be trusted.
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


def _resolve_int_set(
    value: ast.expr,
    consts: dict[str, int],
    funcs: dict[str, ast.FunctionDef | ast.AsyncFunctionDef],
    depth: int = 0,
) -> set[int] | None:
    """Resolve an expression to the SET of ints it can evaluate to, or None.

    None means the expression does not settle statically and must be recorded
    UNJUDGED, never guessed at. A Call to a module-level function is followed
    ONE HOP into that function's returns, and depth >= 1 never follows a call:
    no recursion, no second hop. The one-hop-only rule is deliberate -- a
    deeper walk would start judging values that flow through helpers rather
    than out of the process, and the gate would manufacture rejections the
    contract never made.
    """
    if (
        isinstance(value, ast.Constant)
        and isinstance(value.value, int)
        and not isinstance(value.value, bool)
    ):
        return {value.value}
    if (
        isinstance(value, ast.UnaryOp)
        and isinstance(value.op, ast.USub)
        and isinstance(value.operand, ast.Constant)
        and isinstance(value.operand.value, int)
        and not isinstance(value.operand.value, bool)
    ):
        return {-value.operand.value}
    if isinstance(value, ast.Name) and value.id in consts:
        return {consts[value.id]}
    if isinstance(value, ast.IfExp):
        body = _resolve_int_set(value.body, consts, funcs, depth)
        orelse = _resolve_int_set(value.orelse, consts, funcs, depth)
        if body is None or orelse is None:
            return None  # a half-resolved conditional is not resolved
        return body | orelse
    if (
        isinstance(value, ast.Call)
        and isinstance(value.func, ast.Name)
        and value.func.id in funcs
        and depth == 0
    ):
        returned: list[ast.Return] = []

        def collect(node: ast.AST) -> None:
            for child in ast.iter_child_nodes(node):
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
                    continue  # nested function bodies return values, not exit codes
                if isinstance(child, ast.Return):
                    returned.append(child)
                collect(child)

        collect(funcs[value.func.id])
        if not returned:
            return None  # a function with no returns does not settle to a code here
        settled: set[int] = set()
        for ret in returned:
            if ret.value is None:
                settled.add(0)  # SystemExit(None) is exit 0, as in judge_return
                continue
            one = _resolve_int_set(ret.value, consts, funcs, depth=1)
            if one is None:
                return None
            settled |= one
        return settled
    return None


def _module_tables(
    tree: ast.AST,
) -> tuple[dict[str, int], dict[str, ast.FunctionDef | ast.AsyncFunctionDef]]:
    """Module-level constant and function tables for the return resolver.

    Only the module BODY is read, not ast.walk: a name bound inside a function
    is not visible to an entry function's return in any way this gate may
    assume. A name assigned or defined more than once at module level is
    DROPPED entirely -- two bindings mean the name does not settle statically,
    and keeping the last one would be a guess.
    """
    consts: dict[str, int] = {}
    funcs: dict[str, ast.FunctionDef | ast.AsyncFunctionDef] = {}
    if not isinstance(tree, ast.Module):
        return consts, funcs
    seen_consts: set[str] = set()
    seen_funcs: set[str] = set()
    for node in tree.body:
        target: ast.expr | None = None
        value: ast.expr | None = None
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target, value = node.targets[0], node.value
        elif isinstance(node, ast.AnnAssign):
            target, value = node.target, node.value
        if isinstance(target, ast.Name):
            name = target.id
            if name in seen_consts:
                consts.pop(name, None)  # bound twice: does not settle statically
            else:
                seen_consts.add(name)
                if value is not None and isinstance(value, (ast.Constant, ast.UnaryOp)):
                    resolved = _resolve_int_set(value, {}, {})
                    if resolved is not None and len(resolved) == 1:
                        consts[name] = next(iter(resolved))
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name in seen_funcs:
                funcs.pop(node.name, None)  # defined twice: does not settle statically
            else:
                seen_funcs.add(node.name)
                funcs[node.name] = node
    return consts, funcs


def entry_functions(tree: ast.AST) -> set[str]:
    """Names of locally-defined functions called zero-arg as an exit site's argument.

    The universal entry idiom in this tree is `raise SystemExit(main())` /
    `sys.exit(main())`. Literal-only judging buckets every one of them UNJUDGED:
    on the measured tree that was 59 of 71 exit sites, leaving every `return N`
    inside those entry functions out of every denominator. Only bare Name
    callables with zero arguments count: `main(x)` is not the no-argument entry
    idiom, and `obj.main()` attributes a contract we cannot see, so both stay
    UNJUDGED. Restricting to locally-defined names keeps a harness's
    `sys.exit(thirdparty())` from dragging foreign returns into our contract.
    """
    defined = {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    names: set[str] = set()

    def inspect_exit(args: list[ast.expr]) -> None:
        if len(args) != 1:
            return
        arg = args[0]
        if (
            isinstance(arg, ast.Call)
            and not arg.args
            and not arg.keywords
            and isinstance(arg.func, ast.Name)
            and arg.func.id in defined
        ):
            names.add(arg.func.id)

    for node in ast.walk(tree):
        if isinstance(node, ast.Raise) and node.exc is not None:
            exc = node.exc
            if (
                isinstance(exc, ast.Call)
                and isinstance(exc.func, ast.Name)
                and exc.func.id == "SystemExit"
            ):
                inspect_exit(exc.args)
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
                inspect_exit(node.args)
    return names


def judge_entry_returns(tree: ast.AST, source: str, path: str, names: set[str]) -> ScanResult:
    """Judge the `return` statements of each named entry function, one hop only.

    Following `raise SystemExit(main())` into `main` is what judges the 215
    returns the literal-only rule never reached -- 43 of them returning 1/2/3/4,
    outside {0, 5, 95, 96}, and one of them this gate's own former `return 4`.
    The walk is explicit, not ast.walk, and refuses to descend into NESTED
    FunctionDef/AsyncFunctionDef/Lambda bodies: an inner helper's `return 1` is
    a value handed back into the entry function's own control flow, not an exit
    code, and judging it would manufacture rejections the contract never made.
    Each return is resolved to the set of ints it can settle to -- a literal, a
    module-level constant name, a conditional whose branches both resolve, one
    hop into a module-level helper's returns. What does not settle (Subscript,
    Attribute, BoolOp, a second hop, an unstable binding, ...) is recorded
    UNJUDGED so the count stays honest: no false widening.
    """
    result = ScanResult()
    consts, funcs = _module_tables(tree)

    def judge_return(node: ast.Return, fn_name: str) -> None:
        result.sites += 1
        lineno = node.lineno
        segment = ast.get_source_segment(source, node) or ""
        value = node.value
        if value is None:
            return  # bare `return` leaves SystemExit's argument None, which is exit 0
        if isinstance(value, ast.JoinedStr) or (
            isinstance(value, ast.Constant) and isinstance(value.value, str)
        ):
            result.rejections.append(
                Finding(
                    path,
                    lineno,
                    f"entry function {fn_name}() returns a string, which SystemExit turns "
                    "into exit 1, not a declared contract code (use 96 for REFUSE)",
                    segment,
                    True,
                )
            )
            return
        resolved = _resolve_int_set(value, consts, funcs)
        if resolved is not None:
            offending = sorted(v for v in resolved if v not in CONTRACT_CODES)
            if not offending:
                return  # every code this return can settle to is in contract
            codes = ", ".join(str(v) for v in offending)
            reason = (
                f"entry function {fn_name}() returns exit code {codes}, which is not in "
                "the declared contract {0, 5, 95, 96}"
            )
            if isinstance(value, ast.Call) and isinstance(value.func, ast.Name):
                reason += (
                    f" (resolved one hop through {value.func.id}(); the code is not on "
                    "the line you are reading)"
                )
            elif isinstance(value, ast.IfExp):
                reason += (
                    " (resolved through a conditional; the code is not on the line you "
                    "are reading)"
                )
            result.rejections.append(Finding(path, lineno, reason, segment, False))
            return
        result.unjudged.append((path, lineno, segment))

    def walk(node: ast.AST, fn_name: str) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
                continue  # nested function bodies return values, not exit codes
            if isinstance(child, ast.Return):
                judge_return(child, fn_name)
            walk(child, fn_name)

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in names:
            walk(node, node.name)
    return result


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

    # The Call-argument exit sites stay UNJUDGED as sites (a call may yield any
    # int at runtime), but the one-hop entry rule judges the RETURNS behind them.
    # Skip when nothing is named so plain scripts pay nothing.
    entry_names = entry_functions(tree)
    if entry_names:
        entry_result = judge_entry_returns(tree, source, path, entry_names)
        result.sites += entry_result.sites
        result.rejections.extend(entry_result.rejections)
        result.unjudged.extend(entry_result.unjudged)
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
    # The entry-return widening: the shape that hid 43 out-of-contract returns
    # behind `raise SystemExit(main())`, including this gate's own `return 4`.
    entry_int = check_source(
        "def main():\n    return 7\n\nraise SystemExit(main())\n", "<control:entry-7>"
    )
    if len(entry_int.rejections) != 1 or "7" not in entry_int.rejections[0].reason:
        return (
            "must-fire control 5 failed: `return 7` behind raise SystemExit(main()) did not "
            "produce exactly one rejection mentioning 7"
        )
    entry_msg = check_source(
        'def main():\n    return "boom"\n\nraise SystemExit(main())\n', "<control:entry-msg>"
    )
    if len(entry_msg.rejections) != 1 or not entry_msg.rejections[0].is_message_defect:
        return (
            "must-fire control 6 failed: a string returned from an entry function must give "
            "exactly one rejection flagged as the string-message trap"
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
    entry_contract = check_source(
        "def main():\n    return 96\n\nraise SystemExit(main())\n", "<control:entry-96>"
    )
    if entry_contract.rejections:
        return "must-pass control 2 failed: an entry function returning 96 must not be rejected"
    # A non-literal return is not statically decidable; widening must not
    # manufacture a rejection it cannot know, only record the honest UNJUDGED.
    entry_rc = check_source(
        "def main():\n    rc = 3\n    return rc\n\nraise SystemExit(main())\n",
        "<control:entry-rc>",
    )
    if entry_rc.rejections or not entry_rc.unjudged:
        return (
            "must-pass control 3 failed: non-literal `return rc` in an entry function must "
            "give zero rejections and at least one UNJUDGED entry"
        )
    # A nested helper's `return 1` is a value inside main()'s control flow, not
    # an exit code; descending into it would reject what the contract never made.
    entry_nest = check_source(
        "def main():\n    def helper():\n        return 1\n    return helper() and 0\n\n"
        "raise SystemExit(main())\n",
        "<control:entry-nest>",
    )
    if entry_nest.rejections:
        return (
            "must-pass control 4 failed: a nested helper's `return 1` was judged though only "
            "main()'s own returns are exit codes"
        )
    # Only functions actually named at an exit site are entries; a def that is
    # never called from an exit site returns values, not exit codes.
    entry_scope = check_source(
        "def helper():\n    return 7\n\ndef main():\n    return 0\n\nraise SystemExit(main())\n",
        "<control:entry-scope>",
    )
    if entry_scope.rejections:
        return (
            "must-pass control 5 failed: helper()'s `return 7` was judged though helper is "
            "never the argument of an exit site"
        )
    # The resolver widening: 85 entry-function returns sat UNJUDGED because only
    # a literal int was resolved, 64 of them settle statically, and 30 of those
    # are outside the contract. One must-fire per newly resolvable shape.
    entry_name = check_source(
        "EXIT_BAD = 3\n\ndef main():\n    return EXIT_BAD\n\nraise SystemExit(main())\n",
        "<control:entry-name>",
    )
    if len(entry_name.rejections) != 1 or "3" not in entry_name.rejections[0].reason:
        return (
            "must-fire control 7 failed: `return EXIT_BAD` with EXIT_BAD = 3 at module "
            "level did not produce exactly one rejection mentioning 3"
        )
    entry_ifexp = check_source(
        "def main():\n    ok = True\n    return 0 if ok else 6\n\nraise SystemExit(main())\n",
        "<control:entry-ifexp>",
    )
    if len(entry_ifexp.rejections) != 1 or "6" not in entry_ifexp.rejections[0].reason:
        return (
            "must-fire control 8 failed: `return 0 if ok else 6` did not produce exactly "
            "one rejection mentioning 6"
        )
    entry_call = check_source(
        "def _fail():\n    return 2\n\ndef main():\n    return _fail()\n\n"
        "raise SystemExit(main())\n",
        "<control:entry-call>",
    )
    if len(entry_call.rejections) != 1 or "2" not in entry_call.rejections[0].reason:
        return (
            "must-fire control 9 failed: `return _fail()` with _fail() returning 2 did "
            "not produce exactly one rejection mentioning 2"
        )
    # One must-pass per way the widening could go wrong: resolving what does
    # not settle would manufacture rejections the contract never made.
    entry_name_ok = check_source(
        "EXIT_OK = 0\n\ndef main():\n    return EXIT_OK\n\nraise SystemExit(main())\n",
        "<control:entry-name-ok>",
    )
    if entry_name_ok.rejections:
        return "must-pass control 6 failed: `return EXIT_OK` with EXIT_OK = 0 must not be rejected"
    entry_ifexp_ok = check_source(
        "def main():\n    ok = True\n    return 0 if ok else 5\n\nraise SystemExit(main())\n",
        "<control:entry-ifexp-ok>",
    )
    if entry_ifexp_ok.rejections:
        return (
            "must-pass control 7 failed: `return 0 if ok else 5` has both branches in "
            "contract and must not be rejected"
        )
    # A call to a function not defined in this module does not settle; the
    # unjudged count is the site plus the return, and both must stay UNJUDGED.
    entry_foreign = check_source(
        "def main():\n    return thirdparty()\n\nraise SystemExit(main())\n",
        "<control:entry-foreign>",
    )
    if entry_foreign.rejections or len(entry_foreign.unjudged) != 2:
        return (
            "must-pass control 8 failed: `return thirdparty()` for an undefined name must "
            "give zero rejections and exactly two UNJUDGED entries (site plus return), "
            "proving the widening does not guess"
        )
    # A name bound twice at module level does not settle; keeping the last
    # binding would be a guess, so the return must stay UNJUDGED.
    entry_unstable = check_source(
        "E = 0\nE = 7\n\ndef main():\n    return E\n\nraise SystemExit(main())\n",
        "<control:entry-unstable>",
    )
    if entry_unstable.rejections or len(entry_unstable.unjudged) != 2:
        return (
            "must-pass control 9 failed: a name bound twice at module level must not "
            "resolve; expected zero rejections and exactly two UNJUDGED entries (site "
            "plus return), proving an unstable binding is not resolved"
        )
    # Anti-widening: the top-level SystemExit(main()) site itself must stay
    # UNJUDGED even when main()'s returns all settle. A Call at an exit site
    # may yield any int at runtime; judging the site would claim it decides
    # the code when the returns do. The exact count fails any future change
    # that starts judging the 60 entry-point call sites.
    entry_site = check_source(
        "def main():\n    return 0\n\nraise SystemExit(main())\n",
        "<control:entry-site>",
    )
    if entry_site.rejections or len(entry_site.unjudged) != 1:
        return (
            "must-pass control 10 failed: the top-level SystemExit(main()) site itself "
            "must stay UNJUDGED (exactly one unjudged entry, zero rejections); judging "
            "the site rather than the returns claims the site decides the code"
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
        # 96 = CANNOT-MEASURE by a gate that cannot be trusted. This used to be
        # `return 4` -- out of the very contract this gate enforces, and invisible
        # to it until the entry-return hop judged the 43 returns that included it.
        print(
            f"CONTROLS FAILED: {control_error}; the gate cannot be trusted, so neither can "
            "the build (exit 96 = CANNOT-MEASURE, the contract's slot for exactly this)"
        )
        return 96

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
    # `main()` as the SystemExit argument is still UNJUDGED as an exit site (a
    # Call may yield any int at runtime, and a literal here would lie about who
    # decides). But it is no longer unmeasured: the one-hop entry rule now judges
    # main()'s RETURN values against the contract, which is how this gate's own
    # former `return 4` was finally seen. Do not "fix" this into a literal.
    raise SystemExit(main())
