#!/usr/bin/env python3
"""Close the four out-of-contract returns in the generated trainer's main().

WHAT / WHY. h100/gen/fs_train.fixed.py ends main(argv) with four returns.
Two are exception arms that PRINT one verdict and RETURN another: the
ContractError arm prints RUN_SUMMARY_JSON with verdict UNMEASURED and reason
contract_refused, then returns 2; the OperationFailure arm prints PHASE_JSON
plus RUN_SUMMARY_JSON with verdict UNMEASURED and reason str(exc), then
returns 3. The other two are calls: `return _run_selftest()` and
`return _run(config)`. Measured with a one-hop AST resolver over the
committed artifact, _run_selftest() has exactly one return, `0 if matched
== len(table) else 1`, and _run(config) has exactly one return, `0 if
verdict == "MEASURED" else 3`. The four codes 2, 3, 1, 3 all sit OUTSIDE
the plane's four-state exit contract {0, 5, 95, 96}, so one and the same
function emits a message and an exit code that contradict each other: the
JSON says UNMEASURED, the rc says something the contract does not name.
That is the finding about the SYSTEM.

THE FIX (four one-token edits, no redesign). The two exception arms become
95, the contract's UNMEASURED slot. _run's else-3 becomes 95: that arm
fires only when it has just printed "verdict": "UNMEASURED", and 95 is the
UNMEASURED slot, so the rc stops contradicting the JSON the same function
printed. _run_selftest's else-1 becomes 5: that arm fires when it has just
printed SELFTEST_FAIL, a measured failure, and 5 is the contract's slot for
a measured failure. Each patched line carries a trailing comment that says
which slot and why. Verified safe for consumers: the launcher pipes the
trainer rc through fs_map_run_verdict() in the container backend, which
reads the verdict line out of the run log and OUTRANKS the rc -- verdict
UNMEASURED with any rc maps to 95; verdict MEASURED maps rc 0 to 0 and any
non-zero rc to 5; no verdict line maps a non-zero rc to 5 and rc 0 to 95.
So _run's 3 was already mapped to 95, and _run_selftest's 1 (no verdict
line, non-zero rc) was already mapped to 5: neither code carried private
meaning. Grep over the campaign found no consumer anywhere that tests the
trainer for rc 1 or rc 3 -- the same argument, and the same measurement,
that justified the two exception arms.

THE GATE. The durable guarantee is not that these four lines changed but
that main() AS A WHOLE is now in contract, so after writing this stage
re-reads the file from disk, asserts all four patched anchors present and
all four pre-patch anchors absent, ast.parse()es the result (the target is
Python -- parsed, never shelled out to `bash -n`), and walks the AST with a
one-hop resolver: every return in main()'s OWN body (nested
FunctionDef/AsyncFunctionDef/Lambda not descended into) is resolved to the
SET of ints it can settle to -- an int constant (bool is not an int), a
negated int constant, a conditional whose BOTH arms resolve (a
half-resolved conditional is not resolved), or, at depth 0 only, a call to
a module-level function of this same file whose every own-body return
resolves at depth 1; a bare return settles to {0}, because
SystemExit(None) is exit 0. One hop and no more: a deeper walk would judge
values that flow through helpers rather than out of the process, and the
stage would manufacture findings the contract never made. A return that
does not resolve is UNRESOLVED: not a pass, not a fail, UNMEASURED,
reported by line as the honest limit of the instrument, so the count is
never mistaken for full coverage. The first cut of this gate (_exit_gate,
via _own_returns) collected only int-literal returns, so the two call arms
were invisible to it -- a finding about the INSTRUMENT, not about the tree:
it proved a contract over a denominator narrower than the contract it
claimed to prove, which is how a gate manufactures a false GREEN. A
zero-height denominator (main() with no returns at all) FAILS rather than
passes. Any future edit that reintroduces a private rc fails the build.

IDEMPOTENCE. A fresh extraction produces the PRE-patch state; the committed
artifact is PRE for the two call arms and POST for the two exception arms,
because it was written by this stage at an earlier version of itself. For
EACH of the four anchors, pre + post must be exactly 1 or the stage REFUSES
96 (missing or ambiguous spec, never a guess). Given that invariant every
arm is in exactly one well-formed state, application is monotone and
per-anchor, and a tree with SOME arms patched is a VERSION BUMP, not a
corruption: the write is one write_text, so a partial write truncates the
file and dies at G3 rather than converting one arm and not another. An
earlier cut of this rule called that mixed state half-applied and returned
5, which would have made the stage unable to roll its own artifact forward
-- every anchor added in future would read as corruption and sink the build
on the one state the stage exists to fix. So the stage applies whichever
arms are pre and NAMES the ones that were already in contract. All four
patched -> already applied, byte-idempotent no-op, re-prove the contract and
return 0.

EXIT CODES (the plane's four-state contract; this stage emits no other):
  0  clean apply, or already-applied no-op
  5  a measured finding about the tree: post-write verification failed,
     half-applied state, or the output no longer parses / is out of contract
  95 a required INPUT is absent: target file missing or unreadable
  96 the spec is unusable: an anchor count is not exactly 1 (missing or
     ambiguous), so the stage cannot act safely
"""

from __future__ import annotations

import ast
import collections
import pathlib
import sys

TARGET = pathlib.Path(__file__).resolve().parent / "h100" / "gen" / "fs_train.fixed.py"
MARK = "fs-exit-contract:"
CONTRACT = (0, 5, 95, 96)

# Needles are ASSEMBLED, never written as one literal: this stage counts them in the
# target, and a source that contains its own needle is inside its own denominator.
_ORIGIN_A = '                "dataset_origin": "UNKNOWN_' + 'NOT_RUN",\n'
_ORIGIN_B = '                "dataset_origin": "UNKNOWN_' + 'AFTER_FAILURE",\n'
_TAIL = (
    '                "real_data": None,\n'
    "            },\n"
    "        )\n"
)
ANCHOR_A_PRE = _ORIGIN_A + _TAIL + "        return " + "2"
ANCHOR_B_PRE = _ORIGIN_B + _TAIL + "        return " + "3"

_WHY = '  # this arm prints "verdict": "UNMEASURED"; 95 is the contract\'s UNMEASURED slot'
ANCHOR_A_POST = _ORIGIN_A + _TAIL + "        return 95" + _WHY
ANCHOR_B_POST = _ORIGIN_B + _TAIL + "        return 95" + _WHY

# Anchors C and D are the two call arms: main() returns _run_selftest() and
# _run(config), and one hop down each callee settles out of contract exactly
# once. Same assembly rule: the pre-patch codes 1 and 3 never join one literal.
ANCHOR_C_PRE = "    return 0 if matched == len(table) else " + "1"
ANCHOR_D_PRE = '    return 0 if verdict == "MEASURED" else ' + "3"
ANCHOR_C_POST = (
    "    return 0 if matched == len(table) else 5"
    "  # SELFTEST_FAIL is a measured failure: 5"
)
ANCHOR_D_POST = (
    '    return 0 if verdict == "MEASURED" else 95'
    "  # this arm printed UNMEASURED; 95 is its slot"
)


def _stderr(msg: str) -> None:
    print(msg, file=sys.stderr)


def _own_returns(func: ast.AST) -> list[ast.Return]:
    """Return nodes of one function's own body, in source order.

    Nested function scopes are not descended into: their returns are not this
    function's exit contract. The collector states the denominator; what each
    return RESOLVES to is _resolve()'s question, not this one's.
    """
    rets: list[ast.Return] = []
    stack = list(ast.iter_child_nodes(func))
    while stack:
        node = stack.pop()
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            continue
        if isinstance(node, ast.Return):
            rets.append(node)
        stack.extend(ast.iter_child_nodes(node))
    rets.sort(key=lambda r: r.lineno)
    return rets


def _resolve(expr: ast.AST, funcs: dict[str, ast.FunctionDef], depth: int) -> set[int] | None:
    """The set of ints one returned expression can settle to, or None.

    Rules, and only these: an int constant (bool is not an int); a negated
    int constant; a conditional whose BOTH arms resolve -- a half-resolved
    conditional is not resolved; and, at depth 0 only, a call to a
    module-level function of this same file whose every own-body return
    resolves at depth 1. depth >= 1 never follows a call: no recursion, no
    second hop. Anything else is None -- UNRESOLVED, which is not {0}.
    """
    if (
        isinstance(expr, ast.Constant)
        and isinstance(expr.value, int)
        and not isinstance(expr.value, bool)
    ):
        return {expr.value}
    if (
        isinstance(expr, ast.UnaryOp)
        and isinstance(expr.op, ast.USub)
        and isinstance(expr.operand, ast.Constant)
        and isinstance(expr.operand.value, int)
        and not isinstance(expr.operand.value, bool)
    ):
        return {-expr.operand.value}
    if isinstance(expr, ast.IfExp):
        body = _resolve(expr.body, funcs, depth)
        orelse = _resolve(expr.orelse, funcs, depth)
        if body is None or orelse is None:
            return None
        return body | orelse
    if isinstance(expr, ast.Call) and isinstance(expr.func, ast.Name) and depth == 0:
        callee = funcs.get(expr.func.id)
        if callee is None:
            return None
        rets = _own_returns(callee)
        if not rets:
            return None
        settled: set[int] = set()
        for ret in rets:
            values = _resolve_return(ret, funcs, depth + 1)
            if values is None:
                return None
            settled |= values
        return settled
    return None


def _resolve_return(
    ret: ast.Return, funcs: dict[str, ast.FunctionDef], depth: int
) -> set[int] | None:
    """What one return statement settles to; a bare return is {0}."""
    if ret.value is None:
        return {0}  # SystemExit(None) is exit 0
    return _resolve(ret.value, funcs, depth)


def _exit_gate(source: str) -> tuple[bool, str]:
    """Prove every return main() can settle to lands inside {0, 5, 95, 96}.

    Three counts, not one: returns RESOLVED in contract, returns RESOLVED out
    of contract (the FAIL, with line and codes), and returns UNRESOLVED --
    UNMEASURED, the honest limit of the instrument, named by line so the
    number is never mistaken for full coverage.
    """
    tree = ast.parse(source)
    mains = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "main"
    ]
    if len(mains) != 1:
        return False, f"FunctionDef main count={len(mains)} need=1"
    # A name bound to two module-level definitions does not settle statically, so
    # it is DROPPED rather than resolved to the last one: keeping the last would
    # be a guess, and a guess inside a resolver is how a gate invents a verdict.
    defs: list[ast.FunctionDef | ast.AsyncFunctionDef] = [
        n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    counts = collections.Counter(n.name for n in defs)
    funcs = {n.name: n for n in defs if counts[n.name] == 1}
    rets = _own_returns(mains[0])
    if not rets:
        return False, (
            "0 returns in main(); zero-height denominator -- "
            "95 UNMEASURED, never a pass"
        )
    good = 0
    bad: list[tuple[int, list[int]]] = []
    unresolved: list[int] = []
    seen: set[int] = set()
    for ret in rets:
        settled = _resolve_return(ret, funcs, 0)
        if settled is None:
            unresolved.append(ret.lineno)
        elif settled <= set(CONTRACT):
            good += 1
            seen |= settled
        else:
            bad.append((ret.lineno, sorted(settled - set(CONTRACT))))
    counts = (
        f"{good} resolved in contract, {len(bad)} resolved out of contract, "
        f"{len(unresolved)} unresolved"
    )
    note = ""
    if unresolved:
        lines = ", ".join(str(ln) for ln in unresolved)
        note = (
            f"; unresolved lines {lines} are UNMEASURED: the honest limit of "
            "the instrument, neither a pass nor a fail, and never coverage"
        )
    if bad:
        return False, (
            "main() returns outside {0, 5, 95, 96}: "
            + ", ".join(f"line {ln} can return {codes}" for ln, codes in bad)
            + f" ({counts})"
            + note
        )
    return True, (
        f"main() own-body returns all settle inside {{0, 5, 95, 96}}: "
        f"{counts}, values present {sorted(seen)}{note}"
    )


def _post_gates(source: str) -> list[tuple[str, bool, str]]:
    """The post-write gates, runnable against any candidate source text."""
    g: list[tuple[str, bool, str]] = []
    pa, pb = source.count(ANCHOR_A_POST), source.count(ANCHOR_B_POST)
    pc, pd = source.count(ANCHOR_C_POST), source.count(ANCHOR_D_POST)
    g.append(("G1", pa == 1 and pb == 1 and pc == 1 and pd == 1,
              f"patched anchors present exactly once: contract_refused arm x{pa}, "
              f"operation_failure arm x{pb}, selftest arm x{pc}, run arm x{pd} "
              "need=1/1/1/1"))
    qa, qb = source.count(ANCHOR_A_PRE), source.count(ANCHOR_B_PRE)
    qc, qd = source.count(ANCHOR_C_PRE), source.count(ANCHOR_D_PRE)
    g.append(("G2", qa == 0 and qb == 0 and qc == 0 and qd == 0,
              f"pre-patch anchors absent: return-2 arm x{qa}, return-3 arm x{qb}, "
              f"selftest else-1 arm x{qc}, run else-3 arm x{qd} need=0/0/0/0"))
    try:
        ast.parse(source)
        g.append(("G3", True, "ast.parse() clean (the target is Python: parsed, "
                              "never shelled out to bash -n)"))
    except SyntaxError as exc:
        g.append(("G3", False, f"ast.parse() SyntaxError: {exc}"))
        return g
    ok, detail = _exit_gate(source)
    g.append(("G4", ok, "exit-contract AST gate: " + detail))
    return g


def _print_gates(gates: list[tuple[str, bool, str]]) -> bool:
    ok = True
    for name, good, detail in gates:
        print(f"{name}: {'PASS' if good else 'FAIL'}  {detail}")
        ok = ok and good
    return ok


def main() -> int:
    # The build driver invokes every stage as `python3 <stage>` with NO arguments, so
    # bare invocation must APPLY. Requiring a flag here would make the stage a no-op
    # inside the build while passing by hand -- the #86 orphan shape, one layer up.
    check = False
    rest: list[str] = []
    for a in sys.argv[1:]:
        if a == "--check":
            check = True
        elif a == "--apply":
            pass
        elif a.startswith("--"):
            _stderr("usage: patch_trainer_exit_contract.py [--apply|--check] [target]"
                    "   (no argument == --apply)")
            return 96
        else:
            rest.append(a)
    if len(rest) > 1:
        _stderr("usage: patch_trainer_exit_contract.py [--apply|--check] [target]"
                "   (no argument == --apply)")
        return 96
    target = pathlib.Path(rest[0]) if rest else TARGET

    if not target.exists():
        _stderr(f"UNMEASURED 95: target missing: {target}")
        return 95
    try:
        text = target.read_text("utf-8")
    except OSError as exc:
        _stderr(f"UNMEASURED 95: target unreadable: {exc}")
        return 95

    pre_a, pre_b = text.count(ANCHOR_A_PRE), text.count(ANCHOR_B_PRE)
    pre_c, pre_d = text.count(ANCHOR_C_PRE), text.count(ANCHOR_D_PRE)
    post_a, post_b = text.count(ANCHOR_A_POST), text.count(ANCHOR_B_POST)
    post_c, post_d = text.count(ANCHOR_C_POST), text.count(ANCHOR_D_POST)
    if (pre_a + post_a != 1 or pre_b + post_b != 1
            or pre_c + post_c != 1 or pre_d + post_d != 1):
        _stderr(
            f"REFUSE 96: anchor count != 1 (missing or ambiguous): "
            f"contract_refused arm pre x{pre_a} post x{post_a}, operation_failure "
            f"arm pre x{pre_b} post x{post_b}, selftest arm pre x{pre_c} post "
            f"x{post_c}, run arm pre x{pre_d} post x{post_d}; need exactly one "
            "form of each. The spec is unusable against this file and the stage "
            "will not guess"
        )
        return 96
    # Every arm now satisfies pre + post == 1, so each is in exactly one
    # well-formed state and application is monotone and per-anchor. A tree with
    # SOME arms already patched is therefore NOT a corrupt tree: the write below
    # is one write_text, so a partial write truncates the file and dies at G3
    # (ast.parse) rather than converting one arm and not another. The only way
    # arms disagree is that this stage ran at an EARLIER VERSION of itself,
    # before anchors C and D existed -- which is exactly the state of the
    # committed artifact. Calling that half-applied and refusing 5, as the first
    # cut of this rule did, would make the stage unable to roll its own artifact
    # forward: every future anchor would read as corruption and sink the build on
    # the one state the stage exists to fix. What the operator needs is not a
    # refusal but a NAME for each arm that was already in contract.
    already = [
        name
        for name, post in (
            ("contract_refused", post_a), ("operation_failure", post_b),
            ("selftest", post_c), ("run", post_d),
        )
        if post
    ]

    if len(already) == 4:
        # Second run: byte-exact no-op, but re-prove the contract so an
        # already-applied yet out-of-contract file is RED (5), not a silent pass.
        ok = _print_gates(_post_gates(text))
        print("verdict: already applied; byte-idempotent no-op")
        return 0 if ok else 5

    if already:
        print(
            f"{MARK} {len(already)} of 4 arms already in contract "
            f"({', '.join(already)}) -- a prior version of this stage; "
            f"applying the remaining {4 - len(already)}"
        )

    new = (
        text.replace(ANCHOR_A_PRE, ANCHOR_A_POST)
        .replace(ANCHOR_B_PRE, ANCHOR_B_POST)
        .replace(ANCHOR_C_PRE, ANCHOR_C_POST)
        .replace(ANCHOR_D_PRE, ANCHOR_D_POST)
    )

    if check:
        ok = _print_gates(_post_gates(new))
        print("verdict: NOT APPLIED"
              + ("; the transform would apply cleanly" if ok
                 else "; the transform would NOT pass its own gates"))
        return 5

    target.write_text(new, "utf-8")
    # The post-write gates run against what is actually on disk, not against what
    # was meant to be there.
    try:
        disk = target.read_text("utf-8")
    except OSError as exc:
        _stderr(f"FAIL 5: wrote {target} but could not re-read it for verification: {exc}")
        return 5
    if not _print_gates(_post_gates(disk)):
        _stderr("FAIL 5: post-write verification failed; see gate lines above")
        return 5
    print(f"{MARK} all four arms now in contract; main() verified inside {{0, 5, 95, 96}}")
    return 0


def _guarded() -> int:
    # This stage exists to keep one function's exit codes inside a four-state
    # contract; it must not collapse its own: an unhandled exception is a REFUSE
    # with a named message, never a bare rc=1.
    try:
        return main()
    except Exception as exc:  # noqa: BLE001 - deliberate: no traceback may escape as rc=1
        _stderr(f"REFUSE 96: {MARK} stage raised {type(exc).__name__}: {exc}")
        return 96


if __name__ == "__main__":
    raise SystemExit(_guarded())
