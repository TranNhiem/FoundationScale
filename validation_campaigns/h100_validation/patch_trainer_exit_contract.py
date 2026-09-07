#!/usr/bin/env python3
"""Close the two out-of-contract returns in the generated trainer's main().

WHAT / WHY. h100/gen/fs_train.fixed.py ends main(argv) with two exception arms
that PRINT one verdict and RETURN another. The ContractError arm prints
RUN_SUMMARY_JSON with verdict UNMEASURED and reason contract_refused, then
returns 2. The OperationFailure arm prints PHASE_JSON plus RUN_SUMMARY_JSON
with verdict UNMEASURED and reason str(exc), then returns 3. Both 2 and 3 sit
OUTSIDE the plane's four-state exit contract {0, 5, 95, 96}, so one and the
same function emits a message and an exit code that contradict each other: the
JSON says UNMEASURED, the rc says something the contract does not name.

THE FIX (two one-token edits, no redesign). Both returns become 95, the
contract's UNMEASURED slot, each carrying a trailing comment that says why:
the arm declares UNMEASURED in its own JSON, and 95 is the UNMEASURED slot.
Now the exit code agrees with the verdict the arm itself printed. Verified
safe for consumers: the launcher pipes rc through fs_map_run_verdict(), which
reads the RUN_SUMMARY_JSON verdict from the log and maps UNMEASURED -> 95
regardless of rc, so 2 and 3 carried no private meaning.

THE GATE. The durable guarantee is not that these two lines changed but that
main() AS A WHOLE is now in contract, so after writing this stage re-reads the
file from disk, asserts both patched anchors present and both pre-patch
anchors absent, ast.parse()es the result (the target is Python -- parsed,
never shelled out to `bash -n`), and walks the AST: every int-literal return
in main()'s OWN body (nested FunctionDef/AsyncFunctionDef/Lambda not descended
into) must be in {0, 5, 95, 96}. Any future edit that reintroduces a private
rc fails the build.

IDEMPOTENCE. The committed artifact is the POST-patch state, but a fresh
extraction produces the PRE-patch state. Both arms patched and neither
pre-patch anchor present -> already applied, byte-idempotent no-op, re-prove
the contract and return 0. A MIXED tree (one arm patched, one not) is a
half-applied state -> 5, a measured finding about the tree, never a silent
pass and never a guess.

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


def _stderr(msg: str) -> None:
    print(msg, file=sys.stderr)


def _own_returns(func: ast.AST) -> list[tuple[int, int]]:
    """Int-literal returns of one function's own body, in source order.

    Nested function scopes are not descended into: their returns are not this
    function's exit contract.
    """
    rets: list[tuple[int, int]] = []
    stack = list(ast.iter_child_nodes(func))
    while stack:
        node = stack.pop()
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            continue
        if (
            isinstance(node, ast.Return)
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, int)
            and not isinstance(node.value.value, bool)
        ):
            rets.append((node.lineno, node.value.value))
        stack.extend(ast.iter_child_nodes(node))
    rets.sort(key=lambda r: r[0])
    return rets


def _exit_gate(source: str) -> tuple[bool, str]:
    """Prove every int-literal return in main()'s own body is in {0, 5, 95, 96}."""
    tree = ast.parse(source)
    mains = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "main"
    ]
    if len(mains) != 1:
        return False, f"FunctionDef main count={len(mains)} need=1"
    rets = _own_returns(mains[0])
    if not rets:
        return False, (
            "0 int-literal returns in main(); zero-height denominator -- "
            "95 UNMEASURED, never a pass"
        )
    bad = [(ln, v) for ln, v in rets if v not in CONTRACT]
    if bad:
        return False, (
            "main() returns outside {0, 5, 95, 96}: "
            + ", ".join(f"line {ln} returns {v}" for ln, v in bad)
        )
    vals = sorted({v for _, v in rets})
    return True, (
        f"main() own-body int returns all in {{0, 5, 95, 96}}: "
        f"{len(rets)} return(s), values present {vals}"
    )


def _post_gates(source: str) -> list[tuple[str, bool, str]]:
    """The post-write gates, runnable against any candidate source text."""
    g: list[tuple[str, bool, str]] = []
    pa, pb = source.count(ANCHOR_A_POST), source.count(ANCHOR_B_POST)
    g.append(("G1", pa == 1 and pb == 1,
              f"patched anchors present exactly once: contract_refused arm x{pa}, "
              f"operation_failure arm x{pb} need=1/1"))
    qa, qb = source.count(ANCHOR_A_PRE), source.count(ANCHOR_B_PRE)
    g.append(("G2", qa == 0 and qb == 0,
              f"pre-patch anchors absent: return-2 arm x{qa}, return-3 arm x{qb} "
              "need=0/0"))
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
    post_a, post_b = text.count(ANCHOR_A_POST), text.count(ANCHOR_B_POST)
    if pre_a + post_a != 1 or pre_b + post_b != 1:
        _stderr(
            f"REFUSE 96: anchor count != 1 (missing or ambiguous): "
            f"contract_refused arm pre x{pre_a} post x{post_a}, operation_failure "
            f"arm pre x{pre_b} post x{post_b}; need exactly one form of each. The "
            "spec is unusable against this file and the stage will not guess"
        )
        return 96
    if (pre_a == 1 and post_b == 1) or (post_a == 1 and pre_b == 1):
        _stderr(
            f"FAIL 5: half-applied tree: contract_refused arm is "
            f"{'pre-patch (returns 2)' if pre_a else 'patched (returns 95)'}, "
            f"operation_failure arm is "
            f"{'pre-patch (returns 3)' if pre_b else 'patched (returns 95)'}; one "
            "arm converted, one not -- a measured finding about the tree"
        )
        return 5

    if post_a == 1 and post_b == 1:
        # Second run: byte-exact no-op, but re-prove the contract so an
        # already-applied yet out-of-contract file is RED (5), not a silent pass.
        ok = _print_gates(_post_gates(text))
        print("verdict: already applied; byte-idempotent no-op")
        return 0 if ok else 5

    new = text.replace(ANCHOR_A_PRE, ANCHOR_A_POST).replace(ANCHOR_B_PRE, ANCHOR_B_POST)

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
    print(f"{MARK} both arms now return 95; main() verified inside {{0, 5, 95, 96}}")
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
