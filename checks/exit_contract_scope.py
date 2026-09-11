#!/usr/bin/env python3
"""Gate: the package's SHIPPED trainers exit only with 0, 5, 95 or 96 -- never 1.

WHAT IS MEASURED
    Three axes, each reported separately with its own count, because one collapsed number
    cannot say which axis is inert.

    Axis RETURN: for each entry point declared in ENTRY_POINTS, every `ast.Return` of that
    function's own body (nested FunctionDef / AsyncFunctionDef / Lambda pruned -- their
    returns are not this function's contract) is resolved to a `set[int]` or to None
    (UNRESOLVED). Resolution covers: integer constants (bool excluded -- `True` is not a
    contract code), unary minus over a constant, `IfExp` as the union of both arms, bare
    `return` as {0} (`SystemExit(None)` raises status 0), names bound by module-level
    integer constants in the same file, names bound to module-level integer constants in a
    sibling module via `from pkg.mod import NAME` (cli.py's `EXIT_REFUSE` is unresolvable
    without this), names imported from a sibling module that name a FUNCTION (cli.py's
    `return train(cfg)` is unmeasurable without it -- and that is cli.py's principal
    return), names bound by simple intra-function assignments (loop.py's `return rc`,
    unioned over every modeled assignment; any assignment form the parser does not model
    poisons the name to UNRESOLVED rather than guessing), and calls to module-level
    functions of the same file at ONE hop. A duplicate function name is dropped from the
    callee table and its calls resolve to None.

    Axis MODULE-EXIT: in each file of MODULE_EXIT_FILES, every `raise SystemExit(<expr>)`,
    `sys.exit(<expr>)` and `os._exit(<expr>)` ANYWHERE in the file. `<expr>` is resolved
    by the same resolver; a call to a declared entry point resolves to that entry point's
    RETURN-axis set; a missing argument resolves to {0}; a NON-integer constant argument
    resolves to {1}, because the interpreter coerces `sys.exit("usage")` to status 1 and
    no AST read gets to pretend otherwise.

    Axis ESCAPE: for each declared entry point, whether an unexpected exception can escape
    it and become interpreter exit 1. The body is partitioned statement by statement after
    the docstring, and EVERY statement must be contained by one of two things: (a) it is
    an `ast.Try` whose handlers include `except Exception` or a bare `except:`, with every
    terminal path out of that handler an in-contract `return` or a deliberate re-`raise`
    and no `else:`/`finally:` clause (both run outside the handler); or (b) it is a tail
    `return <declared entry point>(...)` whose callee THIS RUN has already certified
    PROTECTED. Clause (b) is not optional: the shipped `cli.main` ends in a bare
    `return train(cfg)` deliberately -- folding it into cli's `except ValueError` would
    report a training failure as an operator refusal -- and a rule demanding a single
    enclosing `try` calls that correct shape UNPROTECTED. It cannot launder anything
    either, since the callee's status is computed: delegating to an UNPROTECTED entry
    point is UNPROTECTED, and a delegation cycle is UNMEASURED. Every other statement
    shape is UNPROTECTED and is named by its own line. The terminal-path check is
    syntactic and conservative: the handler block must end in Return/Raise on every branch
    (an `if` without an `else` is a fall-through and fails the check).

WHY IT IS A DEFECT (#381, measuring #380's defect class)
    The only exit-contract judge in the repository before this gate sat over a GENERATED
    campaign artifact (`h100/gen/fs_train.fixed.py`). The trainer the package actually
    ships -- loop.py, with ~40 EXIT_* returns and the contract stated in its own
    docstring -- sat in NO exit-contract denominator, and neither did cli.py. #380 found
    65 statements that could exit 1 from the shipped path and no gate could have seen it.
    A gate pointed at a generated copy reads as coverage of the thing it was generated
    from. Exit 1 is what the interpreter produces for an UNCAUGHT exception, so "never 1"
    is a claim about exception containment (axis ESCAPE) as much as about return values
    (axis RETURN).

WHAT IS NOT MEASURED (declared blind spots -- printed in the banner on every verdict)
    1. Third-party code called by an entry point can exit on its own -- notably argparse,
       which calls sys.exit(2) on a usage error and sys.exit(0) on --help/--version from
       inside the stdlib, so cli.py can produce 0 and 2 by a route no AST read of this
       tree can see. That is a real out-of-contract surface; it is named, not measured.
    2. `except Exception` does not catch BaseException: KeyboardInterrupt (status 130) and
       a SystemExit raised by a library pass through a PROTECTED entry point by design.
    3. Resolution is ONE call hop. A two-hop return is UNRESOLVED, never assumed good.
    4. The denominator is DECLARED, not discovered: a new entry point added to the package
       joins no axis until it is added to ENTRY_POINTS. The declared count is printed so
       the omission is visible.

DENOMINATOR
    The `ast.Return` nodes of the declared entry functions (axis RETURN), the
    SystemExit/sys.exit/os._exit sites of the declared module files (axis MODULE-EXIT),
    and the declared entry points themselves (axis ESCAPE). If the measured denominator is
    zero across axes 1 and 2, the verdict is UNMEASURED, never CLEAR: again the vacuous
    truth this repository's doctrine exists to refuse. An UNRESOLVED unit sinks the
    verdict to UNMEASURED for the same reason -- printing the abstention and then stamping
    CLEAR over it is #322's shape, where only the verdict is read. The shipped tree
    resolves every unit on all three axes, so this costs nothing today.

EXIT CODES
    0   CLEAR      -- every declared file parsed, no finding fired, no unit UNRESOLVED
    5   RED        -- at least one finding on any of the three axes
    95  UNMEASURED -- a declared file is missing or does not parse, zero units measured,
                      or any unit on any axis went UNRESOLVED
    96  REFUSE     -- bad CLI usage, or an unexpected internal error; never 1
"""

from __future__ import annotations

import ast
import sys
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

# The repository root is resolved relative to THIS FILE, not to the working directory, so
# that the verdict cannot be changed by where the gate is invoked from (#83/#229's defect
# class). A positional CLI argument overrides it, which is how the self-test points the
# live machinery at a fixture tree.
ROOT = Path(__file__).resolve().parents[1]

EXIT_CLEAR = 0
EXIT_RED = 5
EXIT_UNMEASURED = 95
EXIT_REFUSE = 96

GATE = "exit_contract_scope"

# The contract every shipped entry point is held to. 1 is absent on purpose: exit 1 is
# the interpreter's signature for an uncaught exception, which is #380's whole finding.
CONTRACT = (0, 5, 95, 96)
CONTRACT_SET = frozenset(CONTRACT)


@dataclass(frozen=True)
class EntryPoint:
    """One entry point the package ships, declared by hand -- never discovered (#381)."""

    path: str  # repository-relative, POSIX separators
    func: str  # module-level function whose returns and exception surface are measured


ENTRY_POINTS = (
    EntryPoint("src/foundationscale/train/loop.py", "train"),
    EntryPoint("src/foundationscale/train/cli.py", "main"),
)

MODULE_EXIT_FILES = (
    "src/foundationscale/train/cli.py",
    "src/foundationscale/train/__main__.py",
)


@dataclass
class Module:
    """A parsed file plus the three tables the resolver needs from it.

    `funcs` drops a name defined twice: a redefinition means "which function runs" depends
    on module-assembly detail this parser does not model, and guessing would be false
    precision. `imports` maps a locally bound name to its dotted module for `from x import
    y` only; relative imports are skipped rather than resolved, since a package-relative
    resolution table is exactly the kind of cleverness that silently goes wrong.
    """

    path: Path
    consts: dict[str, int]  # module-level NAME = <int constant> (USub allowed)
    funcs: dict[str, ast.FunctionDef | ast.AsyncFunctionDef]
    imports: dict[str, str]  # local name -> dotted `from` module


@dataclass(frozen=True)
class Item:
    """One RED finding or one UNMEASURED unit, anchored to a file and a physical line."""

    axis: str
    path: str
    line: int
    detail: str


@dataclass
class AxisCount:
    total: int = 0
    good: int = 0
    bad: int = 0
    unresolved: int = 0


@dataclass
class Assessment:
    """Everything the verdict is stated over; `rc` is derived from these fields alone."""

    rc: int
    axis1: AxisCount
    axis2: AxisCount
    escape: dict[str, str]  # "path::func" -> PROTECTED / UNPROTECTED / UNMEASURED
    escape_detail: dict[str, str]
    findings: list[Item]
    unmeasured: list[Item]
    notes: list[str]


def _const_int(node: ast.expr) -> int | None:
    """Return the integer value of a literal, with bool excluded.

    `True` is an int to Python but accepting it would let `return True` launder a 1 past
    the resolver; nobody who wrote `return True` meant EXIT code 1.
    """
    if isinstance(node, ast.Constant) and isinstance(node.value, int):
        if isinstance(node.value, bool):
            return None
        return node.value
    if (
        isinstance(node, ast.UnaryOp)
        and isinstance(node.op, ast.USub)
        and isinstance(node.operand, ast.Constant)
        and isinstance(node.operand.value, int)
    ):
        if isinstance(node.operand.value, bool):
            return None
        return -node.operand.value
    return None


def _load(path: Path, cache: dict[Path, Module | None]) -> Module | None:
    """Parse `path` and build its tables. Missing or unparseable is None, never a guess."""
    path = path.resolve()
    if path in cache:
        return cache[path]
    mod: Module | None = None
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
        tree = ast.parse(text, filename=str(path))
    except (OSError, ValueError, SyntaxError):
        cache[path] = None
        return None
    consts: dict[str, int] = {}
    funcs: dict[str, ast.FunctionDef | ast.AsyncFunctionDef] = {}
    dupes: set[str] = set()
    imports: dict[str, str] = {}
    for stmt in tree.body:
        if isinstance(stmt, ast.Assign) and stmt.targets:
            value = _const_int(stmt.value)
            if value is not None:
                for tgt in stmt.targets:
                    if isinstance(tgt, ast.Name):
                        consts[tgt.id] = value
        elif isinstance(stmt, ast.AnnAssign) and stmt.value is not None:
            value = _const_int(stmt.value)
            if value is not None and isinstance(stmt.target, ast.Name):
                consts[stmt.target.id] = value
        elif isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if stmt.name in funcs or stmt.name in dupes:
                funcs.pop(stmt.name, None)
                dupes.add(stmt.name)
            else:
                funcs[stmt.name] = stmt
        elif isinstance(stmt, ast.ImportFrom):
            if stmt.level == 0 and stmt.module and stmt.module != "__future__":
                for alias in stmt.names:
                    imports[alias.asname or alias.name] = stmt.module
    mod = Module(path=path, consts=consts, funcs=funcs, imports=imports)
    cache[path] = mod
    return mod


def _load_dotted(dotted: str, root: Path, cache: dict[Path, Module | None]) -> Module | None:
    """Locate a `from`-imported module on disk relative to the repository root.

    Two anchors are tried -- the root itself and root/src -- because this package is
    src-layout. Anything more imaginative (sys.path walks, importlib) would make the
    verdict depend on the interpreter reading the file, which #83/#111 rule out.
    """
    parts = dotted.split(".")
    for base in (root, root / "src"):
        candidate = base.joinpath(*parts).with_suffix(".py")
        if candidate.is_file():
            return _load(candidate, cache)
    return None


def _own_nodes(node: ast.AST) -> list[ast.AST]:
    """All nodes of `node`, pruning nested definitions: their returns are not ours (#381)."""
    out: list[ast.AST] = []

    def walk(n: ast.AST) -> None:
        for child in ast.iter_child_nodes(n):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
                continue
            out.append(child)
            walk(child)

    walk(node)
    return out


def _own_returns(fn: ast.AST) -> list[ast.Return]:
    return [n for n in _own_nodes(fn) if isinstance(n, ast.Return)]


def _poison_from(node: ast.AST, poisoned: set[str]) -> None:
    """Mark every Name reachable under `node` as locally unresolvable.

    Conservation, not laziness: a name assigned by tuple unpacking, augmented assignment,
    a for-loop target or a with-item has a value this resolver does not model, and a
    resolver that guesses one produces RED/CLEAR verdicts nobody verified (doctrine:
    UNMEASURED, loudly, rather than a fabricated measurement).
    """
    for sub in ast.walk(node):
        if isinstance(sub, ast.Name):
            poisoned.add(sub.id)


def _local_map(
    fn: ast.FunctionDef | ast.AsyncFunctionDef,
    mod: Module,
    root: Path,
    cache: dict[Path, Module | None],
    stack: frozenset[tuple[Path, str]],
    entries: dict[tuple[Path, str], set[int] | None],
) -> dict[str, set[int] | None]:
    """Resolve each function-local simple name to the union of everything assigned to it.

    loop.py's `rc` is bound by conditional expressions on several statements; unioning all
    of them is a sound over-approximation that never reads an execution order. Names
    assigned through any form this resolver does not model are poisoned to UNRESOLVED.
    """
    sources: dict[str, list[ast.expr]] = {}
    poisoned: set[str] = set()
    for n in _own_nodes(fn):
        if isinstance(n, ast.Assign):
            names = [t.id for t in n.targets if isinstance(t, ast.Name)]
            if n.targets and len(names) == len(n.targets):
                for name in names:
                    sources.setdefault(name, []).append(n.value)
            else:
                for tgt in n.targets:
                    _poison_from(tgt, poisoned)
        elif isinstance(n, ast.AnnAssign):
            if isinstance(n.target, ast.Name) and n.value is not None:
                sources.setdefault(n.target.id, []).append(n.value)
            else:
                _poison_from(n.target, poisoned)
        elif isinstance(n, (ast.AugAssign, ast.For, ast.AsyncFor)):
            _poison_from(n.target, poisoned)
        elif isinstance(n, (ast.With, ast.AsyncWith)):
            for item in n.items:
                if item.optional_vars is not None:
                    _poison_from(item.optional_vars, poisoned)
        elif isinstance(n, ast.ExceptHandler):
            if n.name:
                poisoned.add(n.name)
        elif isinstance(n, (ast.Import, ast.ImportFrom)):
            for alias in n.names:
                poisoned.add(alias.asname or alias.name.split(".")[0])
        elif isinstance(n, ast.NamedExpr):
            _poison_from(n.target, poisoned)

    # Fixed-point pass: a local may be bound from another local (`a = 0; b = a`). Names
    # whose sources never fully resolve -- including genuine cycles -- end as UNRESOLVED.
    resolved: dict[str, set[int] | None] = {}
    left = dict(sources)
    while left:
        moved = False
        retry: dict[str, list[ast.expr]] = {}
        for name, exprs in left.items():
            values: set[int] = set()
            pending = False
            for expr in exprs:
                s = _resolve(expr, mod, root, cache, 0, stack, resolved, entries)
                if s is None:
                    pending = True
                else:
                    values |= s
            if pending:
                retry[name] = exprs
            else:
                resolved[name] = values
                moved = True
        if not moved:
            for name in retry:
                resolved[name] = None
            break
        left = retry
    for name in poisoned:
        resolved[name] = None
    return resolved


def _resolve(
    node: ast.expr,
    mod: Module,
    root: Path,
    cache: dict[Path, Module | None],
    depth: int,
    stack: frozenset[tuple[Path, str]],
    locals_map: dict[str, set[int] | None] | None,
    entries: dict[tuple[Path, str], set[int] | None],
) -> set[int] | None:
    """Resolve one expression to its possible exit codes, or None (UNRESOLVED).

    ONE call hop: `depth` 0 may follow a call to a same-file helper; inside that helper
    (depth 1) further same-file calls stop. A call into a DECLARED entry point of another
    package module takes that entry's already-computed RETURN set instead -- this is the
    only route by which cli.py's `return train(cfg)` is measured, per #381.
    """
    lit = _const_int(node)
    if lit is not None:
        return {lit}
    if isinstance(node, ast.IfExp):
        a = _resolve(node.body, mod, root, cache, depth, stack, locals_map, entries)
        b = _resolve(node.orelse, mod, root, cache, depth, stack, locals_map, entries)
        if a is None or b is None:
            return None
        return a | b
    if isinstance(node, ast.Name):
        if locals_map is not None and node.id in locals_map:
            return locals_map[node.id]
        if node.id in mod.consts:
            return {mod.consts[node.id]}
        dotted = mod.imports.get(node.id)
        if dotted is not None:
            target = _load_dotted(dotted, root, cache)
            if target is not None and node.id in target.consts:
                return {target.consts[node.id]}
        return None
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        name = node.func.id
        if (mod.path, name) in entries:
            return entries[(mod.path, name)]
        if depth >= 1:
            return None
        fn = mod.funcs.get(name)
        if fn is not None:
            return _func_exit_set(mod, fn, root, cache, depth + 1, stack, entries)
        dotted = mod.imports.get(name)
        if dotted is not None:
            target = _load_dotted(dotted, root, cache)
            if target is not None:
                if (target.path, name) in entries:
                    return entries[(target.path, name)]
                tfn = target.funcs.get(name)
                if tfn is not None:
                    # The imported function gets its own hop budget: it IS the entry point
                    # its module ships, and its body was already measured on that basis.
                    return _func_exit_set(target, tfn, root, cache, 0, stack, entries)
        return None
    return None


def _func_exit_set(
    mod: Module,
    fn: ast.FunctionDef | ast.AsyncFunctionDef,
    root: Path,
    cache: dict[Path, Module | None],
    depth: int,
    stack: frozenset[tuple[Path, str]],
    entries: dict[tuple[Path, str], set[int] | None],
) -> set[int] | None:
    """Union of a function's own resolved returns; None if any return is UNRESOLVED.

    The `stack` cuts recursion: a function that (transitively) calls itself is
    UNRESOLVED, because following it would either loop or pretend a fixed point exists.
    """
    key = (mod.path, fn.name)
    if key in stack:
        return None
    returns = _own_returns(fn)
    if not returns:
        # A call to a function with no return statement produces None at the call site,
        # which SystemExit coerces to 0; but "no returns" from something named like an
        # exit path is more likely a half-measured helper, so it is UNRESOLVED, not {0}.
        return None
    inner = stack | {key}
    lmap = _local_map(fn, mod, root, cache, inner, entries)
    out: set[int] = set()
    for ret in returns:
        if ret.value is None:
            out.add(0)
            continue
        s = _resolve(ret.value, mod, root, cache, depth, inner, lmap, entries)
        if s is None:
            return None
        out |= s
    return out


def _fmt(values: set[int]) -> str:
    return "{" + ", ".join(str(v) for v in sorted(values)) + "}"


def _assess_entry(
    ep: EntryPoint,
    mod: Module,
    root: Path,
    cache: dict[Path, Module | None],
    entries: dict[tuple[Path, str], set[int] | None],
    a1: AxisCount,
    findings: list[Item],
    unmeasured: list[Item],
) -> set[int] | None:
    """Axis RETURN for one declared entry point. Returns the entry's combined exit set."""
    fn = mod.funcs.get(ep.func)
    if fn is None:
        unmeasured.append(
            Item(
                "RETURN",
                ep.path,
                0,
                f"declared entry function `{ep.func}` not found in parsed file",
            )
        )
        return None
    returns = _own_returns(fn)
    stack = frozenset({(mod.path, ep.func)})
    lmap = _local_map(fn, mod, root, cache, stack, entries)
    combined: set[int] = set()
    all_resolved = True
    for ret in returns:
        a1.total += 1
        s = (
            {0}
            if ret.value is None
            else _resolve(ret.value, mod, root, cache, 0, stack, lmap, entries)
        )
        if s is None:
            a1.unresolved += 1
            all_resolved = False
            unmeasured.append(
                Item(
                    "RETURN",
                    ep.path,
                    ret.lineno,
                    f"`return {ast.unparse(ret.value) if ret.value else ''}` does not resolve; "
                    "UNRESOLVED neither passes nor fails",
                )
            )
        elif s <= CONTRACT_SET:
            a1.good += 1
        else:
            a1.bad += 1
            findings.append(
                Item(
                    "RETURN",
                    ep.path,
                    ret.lineno,
                    f"return can produce {_fmt(s - CONTRACT_SET)} -- only 0/5/95/96 may exit",
                )
            )
        if s is not None:
            combined |= s
    return combined if all_resolved else None


def _catches_exception(handler_type: ast.expr) -> bool:
    """Does this handler type catch `Exception` (directly, dotted, or in a tuple)?"""
    if isinstance(handler_type, ast.Name) and handler_type.id == "Exception":
        return True
    if isinstance(handler_type, ast.Attribute) and handler_type.attr == "Exception":
        return True
    if isinstance(handler_type, ast.Tuple):
        return any(_catches_exception(e) for e in handler_type.elts)
    return False


def _block_terminals(stmts: list[ast.stmt]) -> list[ast.stmt] | None:
    """Terminal statements of a block, or None if any path falls out of its end.

    A caught exception whose handler falls out of its end leaves the function returning
    None by silent accident; that is not a contracted exit, so the handler fails the
    terminal-path check and the entry point is UNPROTECTED.
    """
    if not stmts:
        return None
    last = stmts[-1]
    if isinstance(last, ast.If):
        body_t = _block_terminals(last.body)
        else_t = _block_terminals(last.orelse) if last.orelse else None
        if body_t is None or else_t is None:
            return None
        return body_t + else_t
    return [last]


def _delegation_target(
    stmt: ast.stmt, mod: Module, root: Path, cache: dict[Path, Module | None]
) -> EntryPoint | None:
    """The declared entry point a tail `return NAME(...)` hands control to, if any.

    Only a bare `return NAME(...)` counts, and only when NAME resolves -- in this file, or
    through a `from x import NAME` -- to a member of ENTRY_POINTS. An attribute call, a
    call nested inside a larger expression, and a name that reaches an undeclared helper
    all return None, because the composition clause in `_escape_status` is sound only
    when the callee's totality was MEASURED on this same run rather than inferred from
    the shape of the call.

    Aliased imports (`from m import train as t`) are not followed: `Module.imports` keeps
    the LOCAL name only, which is the same limitation `_resolve` already carries at its
    cross-module hop. An alias therefore reads UNPROTECTED -- toward the finding, not away
    from it -- and that direction is the one a gate may be wrong in.
    """
    if not isinstance(stmt, ast.Return) or not isinstance(stmt.value, ast.Call):
        return None
    func = stmt.value.func
    if not isinstance(func, ast.Name):
        return None
    if func.id in mod.funcs:
        target_path = mod.path
    else:
        dotted = mod.imports.get(func.id)
        if dotted is None:
            return None
        target = _load_dotted(dotted, root, cache)
        if target is None:
            return None
        target_path = target.path
    for ep in ENTRY_POINTS:
        if ep.func == func.id and (root / ep.path).resolve() == target_path:
            return ep
    return None


def _guard_status(
    try_stmt: ast.Try,
    fn: ast.FunctionDef | ast.AsyncFunctionDef,
    ep: EntryPoint,
    mod: Module,
    root: Path,
    cache: dict[Path, Module | None],
    entries: dict[tuple[Path, str], set[int] | None],
) -> tuple[str, str]:
    """Does this one `try` convert EVERY exception raised under it into a contract code?"""
    if try_stmt.orelse or try_stmt.finalbody:
        clause = "else" if try_stmt.orelse else "finally"
        line = (try_stmt.orelse or try_stmt.finalbody)[0].lineno
        return (
            "UNPROTECTED",
            f"the guard at line {try_stmt.lineno} carries an `{clause}:` clause at line "
            f"{line}, and that clause runs OUTSIDE its own handlers -- a raise there "
            "escapes a try that otherwise looks total",
        )
    guard: ast.ExceptHandler | None = None
    for handler in try_stmt.handlers:
        if handler.type is None or _catches_exception(handler.type):
            guard = handler
            break
    if guard is None:
        line = try_stmt.body[0].lineno if try_stmt.body else try_stmt.lineno
        return (
            "UNPROTECTED",
            f"no `except Exception` / bare `except:` handler; the statement at line {line} "
            "can escape as interpreter exit 1",
        )
    stack = frozenset({(mod.path, ep.func)})
    lmap = _local_map(fn, mod, root, cache, stack, entries)
    for ret in _own_returns(guard):
        s = (
            {0}
            if ret.value is None
            else _resolve(ret.value, mod, root, cache, 0, stack, lmap, entries)
        )
        if s is None:
            return (
                "UNMEASURED",
                f"guard at line {guard.lineno} present, but its handler return at line "
                f"{ret.lineno} does not resolve",
            )
        if not s <= CONTRACT_SET:
            return (
                "UNPROTECTED",
                f"handler return at line {ret.lineno} can produce "
                f"{_fmt(s - CONTRACT_SET)} -- the guard converts the crash but not into a "
                "contract code",
            )
    terminals = _block_terminals(guard.body)
    if terminals is None:
        return (
            "UNPROTECTED",
            f"handler at line {guard.lineno} can fall out without returning or raising; a "
            "caught exception then leaks a None return",
        )
    if not all(isinstance(t, (ast.Return, ast.Raise)) for t in terminals):
        return (
            "UNPROTECTED",
            f"handler at line {guard.lineno} ends in something other than return/raise on "
            "some path",
        )
    return "PROTECTED", f"guard at line {guard.lineno}"


def _escape_status(
    ep: EntryPoint,
    mod: Module,
    root: Path,
    cache: dict[Path, Module | None],
    entries: dict[tuple[Path, str], set[int] | None],
    seen: frozenset[tuple[str, str]] = frozenset(),
) -> tuple[str, str]:
    """Axis ESCAPE: can an unexpected exception leave this entry point as interpreter 1?

    The body is partitioned statement by statement after the docstring, and EVERY
    statement must be contained by one of exactly two things:

    * it is a `try` whose blanket handler turns any exception under it into a contract
      code (`_guard_status`), or
    * it is a tail `return <declared entry point>(...)` whose callee this same run has
      already certified PROTECTED.

    The second clause is what the first draft of this gate lacked, and it mattered. The
    shipped `cli.main` is docstring + guard + `return train(cfg)`, and a rule demanding
    that the body be a SINGLE `try` called that correct shape UNPROTECTED, naming a line
    physically inside the guard. Worse, it contradicted the tree it guards:
    `loop.train`'s docstring and `test_train_boundary_is_total.py` both record that the
    bare handoff is DELIBERATE -- folding it into cli's `except ValueError` would report
    a training failure as an operator refusal (#384's distinction).

    The clause cannot launder anything, because the callee's status is computed rather
    than assumed: delegating to an UNPROTECTED entry point is UNPROTECTED and names the
    callee, delegating to an UNMEASURED one is UNMEASURED, and `seen` turns a delegation
    cycle into UNMEASURED instead of unbounded recursion or a free pass. Anything that is
    neither clause -- an assignment, a call statement, a plain `return`, a `try` narrowed
    to one exception type -- is UNPROTECTED and is named by its own line, which is what
    keeps the axis non-vacuous.
    """
    fn = mod.funcs.get(ep.func)
    if fn is None:
        return "UNMEASURED", f"declared entry function `{ep.func}` not found in parsed file"
    body = fn.body
    i = 0
    while i < len(body):
        stmt = body[i]
        if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant):
            i += 1
            continue
        break
    rest = body[i:]
    if not rest:
        return (
            "UNPROTECTED",
            f"`{ep.func}` has no executable statement after its docstring, so there is no "
            "guarded region to certify -- an empty denominator is not a pass",
        )
    reasons: list[str] = []
    for stmt in rest:
        if isinstance(stmt, ast.Try):
            status, detail = _guard_status(stmt, fn, ep, mod, root, cache, entries)
            if status != "PROTECTED":
                return status, detail
            reasons.append(detail)
            continue
        target = _delegation_target(stmt, mod, root, cache)
        if target is None:
            return (
                "UNPROTECTED",
                f"unguarded statement at line {stmt.lineno}: it sits inside no blanket "
                "`try` and is not a tail call to a PROTECTED entry point, so an exception "
                "there is interpreter exit 1 (#380's defect class)",
            )
        key = (target.path, target.func)
        if key == (ep.path, ep.func) or key in seen:
            return (
                "UNMEASURED",
                f"line {stmt.lineno} delegates to {target.path}::{target.func}, which is "
                "already on the delegation chain; a cycle is unmeasurable, not protected",
            )
        tmod = _load(root / target.path, cache)
        if tmod is None:
            return (
                "UNMEASURED",
                f"line {stmt.lineno} delegates to {target.path}::{target.func}, whose file "
                "is unreadable or unparseable",
            )
        tstatus, tdetail = _escape_status(
            target, tmod, root, cache, entries, seen | {(ep.path, ep.func)}
        )
        if tstatus != "PROTECTED":
            return (
                tstatus,
                f"line {stmt.lineno} hands control to {target.path}::{target.func}, which "
                f"is {tstatus}: {tdetail}",
            )
        reasons.append(
            f"line {stmt.lineno} delegates to {target.path}::{target.func}, itself "
            f"PROTECTED ({tdetail})"
        )
    return "PROTECTED", "; ".join(reasons)


def _site_values(
    args: list[ast.expr],
    mod: Module,
    root: Path,
    cache: dict[Path, Module | None],
    entries: dict[tuple[Path, str], set[int] | None],
) -> set[int] | None:
    """Resolve a SystemExit/sys.exit/os._exit argument list to exit codes."""
    if not args:
        return {0}
    if len(args) > 1:
        # sys.exit(1, 2) is a TypeError, and a TypeError here becomes interpreter 1.
        return {1}
    arg = args[0]
    if isinstance(arg, ast.Constant) and arg.value is None:
        return {0}
    if isinstance(arg, ast.Constant) and not isinstance(arg.value, (int, bool)):
        # The interpreter prints a non-integer argument and exits with status 1; the AST
        # route must not let a `sys.exit("usage")` read as UNMEASURED when it is RED.
        return {1}
    return _resolve(arg, mod, root, cache, 0, frozenset(), None, entries)


def _assess_module_exit(
    rel: str,
    mod: Module,
    root: Path,
    cache: dict[Path, Module | None],
    entries: dict[tuple[Path, str], set[int] | None],
    a2: AxisCount,
    findings: list[Item],
    unmeasured: list[Item],
) -> None:
    """Axis MODULE-EXIT for one declared file: every SystemExit/sys.exit/os._exit site."""
    for node in ast.walk(_own_tree(mod)):
        # Narrow before anything else: `ast.walk` is typed as yielding bare `ast.AST`,
        # which carries no `lineno`, and a finding whose line number came from a cast
        # is a citation nothing checks. Both shapes below are the only two this axis
        # recognises, and both are `_Positional` in the stubs.
        if not isinstance(node, (ast.Raise, ast.Call)):
            continue
        label: str | None = None
        args: list[ast.expr] = []
        if (
            isinstance(node, ast.Raise)
            and isinstance(node.exc, ast.Call)
            and isinstance(node.exc.func, ast.Name)
            and node.exc.func.id == "SystemExit"
        ):
            label = "raise SystemExit"
            args = node.exc.args
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
        ):
            if node.func.attr == "exit" and node.func.value.id == "sys":
                label = "sys.exit"
                args = node.args
            elif node.func.attr == "_exit" and node.func.value.id == "os":
                label = "os._exit"
                args = node.args
        if label is None:
            continue
        a2.total += 1
        s = _site_values(args, mod, root, cache, entries)
        if s is None:
            a2.unresolved += 1
            unmeasured.append(
                Item(
                    "MODULE-EXIT",
                    rel,
                    node.lineno,
                    f"`{label}(...)` argument does not resolve; UNRESOLVED neither passes "
                    "nor fails",
                )
            )
        elif s <= CONTRACT_SET:
            a2.good += 1
        else:
            a2.bad += 1
            findings.append(
                Item(
                    "MODULE-EXIT",
                    rel,
                    node.lineno,
                    f"`{label}` can produce {_fmt(s - CONTRACT_SET)} -- only 0/5/95/96 may exit",
                )
            )


def _own_tree(mod: Module) -> ast.AST:
    # Kept as one indirection because axis 2 scans the WHOLE parse tree (no pruning --
    # a SystemExit inside a helper is still a module exit), unlike _own_nodes callers.
    return _TREES[mod.path]


_TREES: dict[Path, ast.AST] = {}


def _remember_tree(mod: Module, tree: ast.AST) -> None:
    _TREES[mod.path] = tree


def assess(root: Path) -> Assessment:
    """Run all three axes over the declared denominator under `root`. Pure of printing."""
    root = root.resolve()
    cache: dict[Path, Module | None] = {}
    _TREES.clear()
    modules: dict[str, Module] = {}
    notes: list[str] = []
    declared_files = sorted({ep.path for ep in ENTRY_POINTS} | set(MODULE_EXIT_FILES))
    for rel in declared_files:
        p = (root / rel).resolve()
        mod = _load(p, cache)
        if mod is None:
            notes.append(f"declared file unreadable or unparseable: {rel}")
            continue
        try:
            _remember_tree(mod, ast.parse(p.read_text(encoding="utf-8", errors="replace")))
        except (OSError, ValueError, SyntaxError):
            # _load already succeeded, so this cannot fire; but a parse is a parse and a
            # second failure mode that silently different tables would be worse than the
            # odd defensive branch.
            notes.append(f"declared file unreadable or unparseable: {rel}")
            continue
        modules[rel] = mod

    a1, a2 = AxisCount(), AxisCount()
    escape: dict[str, str] = {}
    escape_detail: dict[str, str] = {}
    findings: list[Item] = []
    unmeasured: list[Item] = []
    entries: dict[tuple[Path, str], set[int] | None] = {}

    if notes:
        return Assessment(
            EXIT_UNMEASURED, a1, a2, escape, escape_detail, findings, unmeasured, notes
        )

    # Axis RETURN first, because axes MODULE-EXIT and resolver cross-module hops both
    # consume the entry points' combined exit sets.
    for ep in ENTRY_POINTS:
        combined = _assess_entry(
            ep, modules[ep.path], root, cache, entries, a1, findings, unmeasured
        )
        entries[(modules[ep.path].path, ep.func)] = combined

    for rel in MODULE_EXIT_FILES:
        _assess_module_exit(rel, modules[rel], root, cache, entries, a2, findings, unmeasured)

    for ep in ENTRY_POINTS:
        status, detail = _escape_status(ep, modules[ep.path], root, cache, entries)
        label = f"{ep.path}::{ep.func}"
        escape[label] = status
        escape_detail[label] = detail
        if status == "UNPROTECTED":
            findings.append(Item("ESCAPE", ep.path, 0, f"`{ep.func}` is UNPROTECTED: {detail}"))
        elif status == "UNMEASURED":
            unmeasured.append(Item("ESCAPE", ep.path, 0, f"`{ep.func}` is UNMEASURED: {detail}"))

    if findings:
        rc = EXIT_RED
    elif a1.total + a2.total == 0:
        # Zero measured units is not a pass; all([]) is True and that is the vacuous
        # truth this repository's doctrine exists to refuse.
        rc = EXIT_UNMEASURED
        notes.append("no RETURN nodes and no module-exit sites were measured")
    elif unmeasured:
        # An UNRESOLVED unit is not a clean one. Printing the abstention and then
        # stamping CLEAR over it is #322's shape: the line is emitted, the verdict says
        # everything passed, and only the verdict is read. The shipped tree resolves
        # every unit on all three axes, so 95 here costs nothing today -- and refuses to
        # launder the day that stops being true.
        rc = EXIT_UNMEASURED
    else:
        rc = EXIT_CLEAR
    return Assessment(rc, a1, a2, escape, escape_detail, findings, unmeasured, notes)


def _render(a: Assessment, root: Path) -> list[str]:
    """Turn an Assessment into the lines to print. The banner -- declared denominator and
    declared blind spots -- rides EVERY verdict, per #381."""
    word = {0: "CLEAR", 5: "RED", 95: "UNMEASURED", 96: "REFUSE"}.get(a.rc, "REFUSE")
    esc_prot = sum(1 for v in a.escape.values() if v == "PROTECTED")
    esc_unp = sum(1 for v in a.escape.values() if v == "UNPROTECTED")
    esc_unm = sum(1 for v in a.escape.values() if v == "UNMEASURED")
    lines = [
        f"{word} {GATE}: {len(a.findings)} finding(s) across 3 axes; root={root}",
        f"  declared denominator: {len(ENTRY_POINTS)} entry points "
        f"({', '.join(ep.func for ep in ENTRY_POINTS)}), {len(MODULE_EXIT_FILES)} "
        f"module-exit files -- DECLARED, never discovered (blind spot 4)",
        f"  axis RETURN:      {a.axis1.good} in-contract, {a.axis1.bad} out-of-contract, "
        f"{a.axis1.unresolved} unresolved over {a.axis1.total} return(s)",
        f"  axis MODULE-EXIT: {a.axis2.good} in-contract, {a.axis2.bad} out-of-contract, "
        f"{a.axis2.unresolved} unresolved over {a.axis2.total} site(s)",
        f"  axis ESCAPE:      {esc_prot} PROTECTED, {esc_unp} UNPROTECTED, "
        f"{esc_unm} UNMEASURED over {len(a.escape)} entry point(s)",
    ]
    for label in sorted(a.escape):
        lines.append(f"    {a.escape[label]:<11} {label}: {a.escape_detail[label]}")
    for item in a.findings:
        lines.append(f"  RED {item.axis} {item.path}:{item.line}: {item.detail}")
    for item in a.unmeasured:
        lines.append(f"  UNMEASURED {item.axis} {item.path}:{item.line}: {item.detail}")
    for note in a.notes:
        lines.append(f"  UNMEASURED {note} -- unreadable is not empty and it is not clean")
    lines.extend(
        [
            "  declared blind spots (named, not hidden):",
            "    1. third-party code can exit on its own: argparse calls sys.exit(2) on a",
            "       usage error and sys.exit(0) on --help/--version from inside the stdlib,",
            "       so cli.py can produce 0 and 2 by a route no AST read of this tree sees;",
            "       that out-of-contract surface is named, not measured.",
            "    2. except Exception does not catch BaseException: KeyboardInterrupt (130)",
            "       and a library SystemExit pass through a PROTECTED entry point by design.",
            "    3. resolution is ONE call hop; a two-hop return is UNRESOLVED, not assumed",
            "       good.",
            "    4. the denominator is DECLARED, not discovered: a new entry point joins",
            f"       no axis until it is added to ENTRY_POINTS (currently "
            f"{len(ENTRY_POINTS)} declared).",
        ]
    )
    return lines


def run(root: Path, quiet: bool = False) -> int:
    """Assess `root` and return the exit code, printing the verdict unless quiet."""
    a = assess(root)
    if not quiet:
        print("\n".join(_render(a, root)))
    return a.rc


# --------------------------------------------------------------------------------------------
# Controls.  Each one writes a fixture tree to a temp directory and runs the REAL assess()
# against it with the root override -- none re-implements the rule, so a control cannot pass
# while the shipped code is wrong.  MUST_FIRE plants the exact defect and asserts the
# detector fires; MUST_PASS plants correct shapes and asserts silence; MUST_BE_UNMEASURED
# plants a unit the resolver cannot own and asserts it is counted as such, not as a verdict.
# --------------------------------------------------------------------------------------------

_CONSTS = "EXIT_PASS = 0\nEXIT_RED = 5\nEXIT_UNMEASURED = 95\nEXIT_REFUSE = 96\n\n\n"


def _loop(body: str, consts: str = _CONSTS) -> str:
    return consts + body


_GOOD_LOOP = _loop(
    "def _train(cfg):\n"
    "    if cfg is None:\n"
    "        return EXIT_UNMEASURED\n"
    "    rc = EXIT_PASS if cfg else EXIT_RED\n"
    "    return rc\n"
    "\n\n"
    "def train(cfg):\n"
    "    try:\n"
    "        return _train(cfg)\n"
    "    except Exception:\n"
    "        return EXIT_RED\n"
)

_GOOD_CLI = (
    "from foundationscale.train.loop import train\n"
    "\n\n"
    "def main(argv=None):\n"
    "    try:\n"
    "        return train(argv)\n"
    "    except Exception:\n"
    "        return 5\n"
)

_GOOD_MAINMOD = "from foundationscale.train.cli import main\n\nraise SystemExit(main())\n"

_NARROW_LOOP = _loop(
    "def _train(cfg):\n"
    "    return EXIT_PASS\n"
    "\n\n"
    "def train(cfg):\n"
    "    try:\n"
    "        return _train(cfg)\n"
    "    except ValueError:\n"
    "        return EXIT_RED\n"
)

# The shipped handoff shape: no guard of its own, just a tail call into an entry point
# that IS total. Correct, and the reason the ESCAPE rule needs a composition clause.
_OPEN_CLI = (
    "from foundationscale.train.loop import train\n"
    "\n\n"
    "def main(argv=None):\n"
    "    return train(argv)\n"
)

# #381's actual defect: the handoff is fine, the WORK before it is in no guard at all.
_WORKING_CLI = (
    "from foundationscale.train.loop import train\n"
    "\n\n"
    "def _build(argv):\n"
    "    return argv\n"
    "\n\n"
    "def main(argv=None):\n"
    "    cfg = _build(argv)\n"
    "    return train(cfg)\n"
)

_NARROW_CLI = (
    "from foundationscale.train.loop import train\n"
    "\n\n"
    "def main(argv=None):\n"
    "    try:\n"
    "        return train(argv)\n"
    "    except ValueError:\n"
    "        return 5\n"
)


def _assess_pack(
    loop: str | None,
    cli: str | None,
    mainmod: str | None = _GOOD_MAINMOD,
) -> Assessment:
    """Write a fixture package tree to a temp dir and run the real machine over it."""
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        pkg = root / "src" / "foundationscale" / "train"
        pkg.mkdir(parents=True)
        if loop is not None:
            (pkg / "loop.py").write_text(loop, encoding="utf-8")
        if cli is not None:
            (pkg / "cli.py").write_text(cli, encoding="utf-8")
        if mainmod is not None:
            (pkg / "__main__.py").write_text(mainmod, encoding="utf-8")
        return assess(root)


def c_literal_one() -> bool:
    loop = _loop(
        "def train(cfg):\n    try:\n        return 1\n    except Exception:\n        return 5\n"
    )
    a = _assess_pack(loop, _GOOD_CLI)
    return a.rc == EXIT_RED and a.axis1.bad >= 1


def c_name_bound_to_one() -> bool:
    loop = _loop(
        "def train(cfg):\n    try:\n        return BAD\n    except Exception:\n        return 5\n",
        _CONSTS + "BAD = 1\n\n\n",
    )
    a = _assess_pack(loop, _GOOD_CLI)
    return a.rc == EXIT_RED and a.axis1.bad >= 1


def c_imported_out_of_contract_constant() -> bool:
    cli = (
        "from foundationscale.train.loop import OOPS\n\n"
        "def main(argv=None):\n"
        "    try:\n"
        "        return OOPS\n"
        "    except Exception:\n"
        "        return 5\n"
    )
    a = _assess_pack(_GOOD_LOOP + "\nOOPS = 7\n", cli)
    return a.rc == EXIT_RED and a.axis1.bad >= 1


def c_ifexp_with_one_bad_arm() -> bool:
    loop = _loop(
        "def train(cfg):\n"
        "    try:\n"
        "        return 0 if cfg else 3\n"
        "    except Exception:\n"
        "        return 5\n"
    )
    a = _assess_pack(loop, _GOOD_CLI)
    return a.rc == EXIT_RED and a.axis1.bad >= 1


def c_one_hop_helper_returns_one() -> bool:
    loop = _loop(
        "def _boom(cfg):\n"
        "    return 1\n"
        "\n\n"
        "def train(cfg):\n"
        "    try:\n"
        "        return _boom(cfg)\n"
        "    except Exception:\n"
        "        return 5\n"
    )
    a = _assess_pack(loop, _GOOD_CLI)
    return a.rc == EXIT_RED and a.axis1.bad >= 1


def c_module_scope_systemexit_one() -> bool:
    a = _assess_pack(_GOOD_LOOP, _GOOD_CLI + "\nraise SystemExit(1)\n")
    return a.rc == EXIT_RED and a.axis2.bad == 1 and a.axis1.bad == 0


def c_unguarded_statement_before_the_handoff() -> bool:
    # #381's own shape. `cfg = _build(argv)` sits in no guard, so a raise there is
    # interpreter 1 one function EARLIER than #380's hole. axis RETURN stays clean; the
    # fire must come from ESCAPE alone, or the control is measuring the wrong axis.
    a = _assess_pack(_GOOD_LOOP, _WORKING_CLI)
    return (
        a.rc == EXIT_RED
        and a.axis1.bad == 0
        and a.escape[f"{ENTRY_POINTS[1].path}::main"] == "UNPROTECTED"
    )


def c_delegation_to_a_protected_entry_point_is_contained() -> bool:
    # The composition clause. The shipped cli.main ends in a bare `return train(cfg)` ON
    # PURPOSE, and the first draft of this rule -- "the body must be one try" -- called
    # that UNPROTECTED. A gate that contradicts the tree it guards is the finding.
    a = _assess_pack(_GOOD_LOOP, _OPEN_CLI)
    key = f"{ENTRY_POINTS[1].path}::main"
    return (
        a.rc == EXIT_CLEAR
        and a.escape[key] == "PROTECTED"
        and "loop.py::train" in a.escape_detail[key]
    )


def c_delegation_to_an_unprotected_entry_point_fires() -> bool:
    # ...and the composition clause must not be a laundry. Same cli, callee narrowed to
    # `except ValueError`: main is UNPROTECTED THROUGH the delegation, and the detail
    # has to name the callee or the operator cannot act on it.
    a = _assess_pack(_NARROW_LOOP, _OPEN_CLI)
    key = f"{ENTRY_POINTS[1].path}::main"
    return (
        a.rc == EXIT_RED
        and a.escape[key] == "UNPROTECTED"
        and "loop.py::train" in a.escape_detail[key]
    )


def c_finally_clause_runs_outside_the_guard() -> bool:
    # A `finally:` executes OUTSIDE the handler that protects the try body, so a raise
    # there escapes a guard that otherwise looks total. Same for `else:`.
    loop = _loop(
        "def train(cfg):\n"
        "    try:\n"
        "        return EXIT_PASS\n"
        "    except Exception:\n"
        "        return EXIT_RED\n"
        "    finally:\n"
        "        _cleanup(cfg)\n"
    )
    a = _assess_pack(loop, _GOOD_CLI)
    return a.rc == EXIT_RED and a.escape[f"{ENTRY_POINTS[0].path}::train"] == "UNPROTECTED"


def c_handler_catches_only_valueerror() -> bool:
    a = _assess_pack(_GOOD_LOOP, _NARROW_CLI)
    return a.rc == EXIT_RED and a.escape[f"{ENTRY_POINTS[1].path}::main"] == "UNPROTECTED"


def c_correct_pack_stays_silent() -> bool:
    # Literal 5, resolved EXIT_* names, a local IfExp both of whose arms are in contract,
    # a cross-module function hop, and a SystemExit(main()) site -- the shipped shapes.
    a = _assess_pack(_GOOD_LOOP, _GOOD_CLI)
    return (
        a.rc == EXIT_CLEAR
        and a.axis1.bad == 0
        and a.axis1.unresolved == 0
        and a.axis2.bad == 0
        and all(v == "PROTECTED" for v in a.escape.values())
    )


def c_environ_return_is_unmeasured_not_a_verdict() -> bool:
    loop = _loop(
        "def train(cfg):\n"
        "    try:\n"
        '        return os.environ["X"]\n'
        "    except Exception:\n"
        "        return 5\n"
    )
    a = _assess_pack(loop, _GOOD_CLI)
    # TWO units go unresolved, not one: loop's own `return os.environ["X"]`, and then
    # cli's `return train(argv)`, which is that same unknown one hop along. And the
    # verdict is 95, not 0 -- an abstention reported under a CLEAR banner is read as a
    # pass, which is exactly #322.
    return a.rc == EXIT_UNMEASURED and a.axis1.unresolved == 2 and a.axis1.bad == 0


def c_missing_declared_file_is_unmeasured() -> bool:
    # Fail CLOSED: a declared file that is not there is UNMEASURED (95), never CLEAR.
    a = _assess_pack(_GOOD_LOOP, None, None)
    return a.rc == EXIT_UNMEASURED and any("cli.py" in n for n in a.notes)


def c_escape_axis_is_not_vacuous() -> bool:
    # The same body PROTECTED under `except Exception` must flip to UNPROTECTED when the
    # handler is narrowed to ValueError; otherwise the axis asserts nothing.
    protected = _assess_pack(_GOOD_LOOP, _GOOD_CLI)
    narrowed = _assess_pack(_NARROW_LOOP, _GOOD_CLI)
    key = f"{ENTRY_POINTS[0].path}::train"
    return protected.escape.get(key) == "PROTECTED" and narrowed.escape.get(key) == "UNPROTECTED"


CONTROLS: list[tuple[str, str, Callable[[], bool]]] = [
    ("return of a literal 1", "MUST_FIRE", c_literal_one),
    ("return of a Name bound to 1 at module level", "MUST_FIRE", c_name_bound_to_one),
    (
        "cross-module imported constant out of contract",
        "MUST_FIRE",
        c_imported_out_of_contract_constant,
    ),
    ("IfExp with one out-of-contract arm", "MUST_FIRE", c_ifexp_with_one_bad_arm),
    ("one-hop call into a helper returning 1", "MUST_FIRE", c_one_hop_helper_returns_one),
    ("module-scope raise SystemExit(1)", "MUST_FIRE", c_module_scope_systemexit_one),
    (
        "unguarded statement before the handoff",
        "MUST_FIRE",
        c_unguarded_statement_before_the_handoff,
    ),
    ("handler catching only ValueError", "MUST_FIRE", c_handler_catches_only_valueerror),
    (
        "ESCAPE not vacuous: narrowed handler flips verdict",
        "MUST_FIRE",
        c_escape_axis_is_not_vacuous,
    ),
    (
        "delegation to an UNPROTECTED entry point",
        "MUST_FIRE",
        c_delegation_to_an_unprotected_entry_point_fires,
    ),
    ("guard carrying a finally: clause", "MUST_FIRE", c_finally_clause_runs_outside_the_guard),
    (
        "in-contract literals, names, IfExp, protected pack",
        "MUST_PASS",
        c_correct_pack_stays_silent,
    ),
    (
        "bare handoff to a PROTECTED entry point",
        "MUST_PASS",
        c_delegation_to_a_protected_entry_point_is_contained,
    ),
    (
        "os.environ[...] return is unresolvable, not RED",
        "MUST_BE_UNMEASURED",
        c_environ_return_is_unmeasured_not_a_verdict,
    ),
    (
        "missing declared entry file -> 95",
        "MUST_BE_UNMEASURED",
        c_missing_declared_file_is_unmeasured,
    ),
]


def self_test() -> int:
    """Run every control. The DENOMINATOR line prints on both arms so an emptied control
    set cannot impersonate a pass (six sibling gates print this grammar already)."""
    failures: list[str] = []
    for name, kind, fn in CONTROLS:
        try:
            ok = fn()
        except Exception as exc:  # noqa: BLE001 - a crashing control is a failing control
            failures.append(f"{kind} RAISED {type(exc).__name__}: {name}: {exc}")
            continue
        if not ok:
            failures.append(f"{kind} FAILED: {name}")
    behaved = len(CONTROLS) - len(failures)
    if failures:
        print(
            f"SELF-TEST DENOMINATOR: {behaved} of {len(CONTROLS)} controls behaved; "
            + "; ".join(failures)
        )
        return EXIT_RED
    fires = sum(1 for _, k, _ in CONTROLS if k == "MUST_FIRE")
    unm = sum(1 for _, k, _ in CONTROLS if k == "MUST_BE_UNMEASURED")
    passes = len(CONTROLS) - fires - unm
    print(
        f"SELF-TEST DENOMINATOR: {behaved} of {len(CONTROLS)} controls behaved; "
        f"{fires} MUST_FIRE, {passes} MUST_PASS, {unm} MUST_BE_UNMEASURED"
    )
    return EXIT_CLEAR


def main(argv: list[str]) -> int:
    self_test_flag = False
    positionals: list[str] = []
    unknown: list[str] = []
    for arg in argv:
        if arg == "--self-test":
            self_test_flag = True
        elif arg.startswith("-"):
            unknown.append(arg)
        else:
            positionals.append(arg)
    if unknown or len(positionals) > 1:
        # Bad usage is REFUSE (96), not 2: this gate exits only with its own contract.
        print(
            f"REFUSE {GATE}: usage: python3 checks/{Path(__file__).name} [--self-test] [repo-root]"
        )
        return EXIT_REFUSE
    if self_test_flag:
        return self_test()
    root = Path(positionals[0]).resolve() if positionals else ROOT
    return run(root)


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv[1:]))
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001 - a crash is not a verdict
        # 1 is exactly the exit this gate exists to forbid, so the gate itself must never
        # produce it: any uncaught exception is contained and becomes REFUSE.
        print(f"REFUSE {GATE}: unexpected internal error: {type(exc).__name__}: {exc}")
        sys.exit(EXIT_REFUSE)
