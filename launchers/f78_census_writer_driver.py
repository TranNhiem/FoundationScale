#!/usr/bin/env python3
"""f78 leg driver -- exercise the REAL persistence writer of
launchers/lora_target_census.py WITHOUT exec'ing that probe's top level.

Why this driver exists (the seam it welds)
------------------------------------------
Both F78 legs of launchers/test_launcher_contracts.sh died at
stage=F78_STAGE=exec-failed (python rc 15) BEFORE any assertion ran: zero
units exercised, so per doctrine 1/3 the legs currently record
UNMEASURED-red -- they are controls that never RUN, which doctrine 3 says
are not controls at all. The census probe is an in-container tool whose
module top level -- correctly, by design -- imports the training-interpreter
stack; the first version of the legs exec'd that whole module under a
hand-enumerated stub set ('megatron.bridge' only). The enumeration is the
defect, not the probe. The failing statement's CLASS is in evidence: a
top-level statement inside the probe's import block (audit listing covers
line 113 onward; lines 61-112 of 656 were never listed). Its exact spelling
is NOT in evidence and is not guessed here (doctrine 4). Three instance
hypotheses -- all removed by this single repair, because none of them
survives 'never exec the top level':
  (i)   a parenthesized multi-line import whose continuation lines a
        textual neuter leaves dangling as 'non-megatron' statements;
  (ii)  a hand-built megatron.bridge stub authored pre-#78, missing one
        newly imported name;
  (iii) an AST-whitelist neuter that dropped the #78-added module constant
        ARTIFACT_STRIP_SEGMENT, which _artifact_stem now requires.
Discriminate by looking, not guessing:
  sed -n '61,112p' launchers/lora_target_census.py
plus one re-run of the old driver with the exec traceback printed. The
repair does not need the answer; the audit trail does.

Repairs considered and rejected (recorded so they cannot be re-picked
silently):
  * probe-side lazy/guarded imports: bending production code to green a
    control -- the single worst outcome this round names;
  * extending the hand-enumerated stub set: works tonight, re-breaks on the
    next top-level import, and a stub whose semantics drift from the real
    symbol's is a false-green engine (doctrine 5, symmetric);
  * exec'ing the WHOLE module under AUTO-GENERATED sys.modules stubs:
    keeps the fragile top-level exec alive as the load-bearing party and
    still fabricates symbols -- ranked below function-extraction even by
    the reviewer that proposed it;
  * relaxing MUST_FIRE to 'rc != 0' or MUST_PASS to 'a file exists':
    drops exactly the assertion halves these legs exist to indict.

What this driver does
---------------------
ast.parse the probe (stdlib only). Lift, IN ORDER:
  * any leading 'from __future__ import ...' statements (so annotations are
    evaluated exactly the way the probe's own module evaluates them),
  * every module-scope assignment (plain or annotated, single Name target)
    whose value ast.literal_eval accepts -- bare constants AND literal
    containers; anything richer (a call, a join, a comprehension) is left
    out and, if the writer's closure needs it, surfaces LEGIBLY as
    F78_EXTRACT_UNRESOLVED=<name> -- never silently fabricated,
  * _persist_adapter_census plus the transitive closure of same-module
    functions it references (ast.Name Load walk over each kept body),
then exec NOTHING else, in a namespace whose __name__ is
'__f78_persistence__' (the probe's __main__ guard cannot fire even
transitively), over a stdlib-only prelude. The bytes executed are the
probe's own published source, compiled under the probe's own filename and
line numbers so tracebacks stay citable -- the same standard the probe
holds its matcher oracle to.

Facts printed (the driver's contract with the legs)
---------------------------------------------------
drive:
  stage=F78_STAGE=drove rc=0
  F78_EXTRACT_FUNCS=<k>            extraction denominator (doctrine 2)
  F78_EXTRACT_CONSTS=<c>
  F78_EXTRACT_FUTURES=<csv|none>
  F78_EXTRACT_UNRESOLVED=<csv|none>   module-scope names the closure wants
      that were NOT lifted. !=none means a subsequent NameError red is a
      DRIVER extraction gap, not a probe finding -- doctrine 5 requires
      this distinction be visible. Sentinal '<unresolved-check-failed>'
      means the check itself broke: red either way, never green.
  F78_RAISED=<none|Type: msg>      writer behaviour, including SystemExit
  F78_EXIT=<none|code>             under the probe's own 0/1/3 vocabulary
  F78_OUT_PATH=<path|unspecified>
  F78_OUT_EXISTS=<0|1|unknown>     doctrine 4: an out-path the driver
      cannot see is UNKNOWN, never 0 -- a forged 0 here would vacuously
      pass MUST_FIRE half (a). Legs pin the path via spec.out_path.
  F78_OUT_PARENT_PRESENT=<0|1|unknown>
  F78_VERDICT_TOKENS=<csv|none>    UNMEASURED/BLOCKED/CLEAR lifted from the
      writer's captured stdout+stderr, in order, duplicates kept
  F78_RESULT=<json|repr>           writer return value, when not None
  F78_TOOL|<line>                  the tool's own words, quoted
  rc is 0 even when the writer misbehaved: an unwritten artifact or an
  UNMEASURED verdict is a MEASURED fact about the code under test; the
  leg's own greps go red. The driver never pre-judges.
verify:
  stage=F78_STAGE=verified rc=0
  F78_JSON_PARSE=<ok|fail>         a missing/unparsable artifact the
      contract obliges the writer to produce is a probe red at rc 0
  F78_FQNS_FOUND=<k> of <n>        explicit denominator, n = len(expect_fqns)
  F78_FQNS_MISSING=<csv|none>
  F78_ARTIFACT_ROWS=<r> of <m>     when the artifact carries a rows list,
      else F78_ARTIFACT_ROWS=absent -- unmeasured is not zero (doctrine 4),
      and no fabricated '0 of m' line may satisfy the grep by accident.

Leg integration (one-line swap per leg at the exec site; every pre-existing
assertion grep is preserved byte-for-byte. The F78 leg block sits outside
the audited window of the 3676-line harness -- locate it with
  grep -n F78_STAGE launchers/test_launcher_contracts.sh
-- and doctrine 4 forbids shipping a blind byte-anchor against it, so the
swap ships as these verbatim instructions, not as an Edit):

  MUST_FIRE (refusal fixture authored by the leg):
      out=$(python3 launchers/f78_census_writer_driver.py drive launchers/lora_target_census.py "$f78_spec.json") || true
      keep asserting BOTH halves against "$out":
        (a) grep 'F78_OUT_EXISTS=0'                -- no census file afterwards
            ('unknown' NEVER matches: unmeasured is not pass)
        (b) grep 'F78_VERDICT_TOKENS=.*UNMEASURED' -- the verdict is UNMEASURED
      additive: grep 'F78_EXTRACT_UNRESOLVED=none' -- the driver's own
      extraction denominator; !=none re-reads the red as a DRIVER gap.

  MUST_PASS (real-artifact fixture authored by the leg):
      out=$(python3 launchers/f78_census_writer_driver.py drive launchers/lora_target_census.py "$f78_spec.json") || true
      ver=$(python3 launchers/f78_census_writer_driver.py verify "$f78_verify.json")
      keep asserting against "$ver":
        grep 'F78_JSON_PARSE=ok'
        each expected FQN individually and grep 'F78_FQNS_MISSING=none'
        grep the explicit denominator 'F78_FQNS_FOUND=<k> of <n>' and/or
        'F78_ARTIFACT_ROWS=<r> of <m>' -- never a bare numerator
      additive: grep 'F78_EXTRACT_UNRESOLVED=none' against "$out".

Spec shapes (JSON, authored by the leg fixtures):
  drive : {"kwargs": {"out_path": ..., "rows": [...], "population": N,
                      "hf_model_path": ..., "targets": [...], "total": M},
           "out_path": "optional explicit artifact path -- set it when the
                        writer's out-parameter is not named out_path, so
                        F78_OUT_EXISTS stays measured instead of unknown"}
          kwargs keys pass through as **kwargs to the real writer.
  verify: {"artifact": "/path/to/expected.json",
           "expect_fqns": ["module....linear_qkv", ...],
           "expect_denominator": <int printed as the 'of M'>;
                                 defaults to len(expect_fqns)}

Exit vocabulary (deliberately disjoint from the probe's 0/1/3):
  0  facts printed (a MEASURED probe misbehaviour is reported as F78_*
     facts at rc 0 so the leg's own greps go red);
  15 stage=F78_STAGE={extract,exec,drive,verified,driver}-failed --
     driver / fixture infrastructure; off-vocabulary exits are forbidden;
  2  argv misuse -- same convention as probe argparse.
"""

import ast
import builtins
import contextlib
import io
import json
import os
import re
import symtable
import sys

STAGE = "F78_STAGE"
RC_INFRA = 15
RC_USAGE = 2

WRITER_ENTRY = "_persist_adapter_census"

# stdlib curtain for the lifted bodies: everything a stdlib-pure persistence
# writer may legitimately reference. Anything beyond this (a lazy heavy
# import inside the writer, say) surfaces as F78_RAISED=NameError /
# ModuleNotFoundError AND as F78_EXTRACT_UNRESOLVED before it -- legible,
# never silently stubbed.
PRELUDE = "\n".join([
    "import argparse, collections, contextlib, functools, hashlib, io, json, "
    "os, pathlib, re, shutil, stat, sys, tempfile, time, typing",
    "from typing import *",
    "",
])
PRELUDE_NAMES = frozenset({
    "argparse", "collections", "contextlib", "functools", "hashlib", "io",
    "json", "os", "pathlib", "re", "shutil", "stat", "sys", "tempfile",
    "time", "typing",
})
_TYPING_PUBLIC = frozenset(dir(__import__("typing")))
_NS_DUNDERS = frozenset({
    "__name__", "__file__", "__doc__", "__builtins__", "__loader__",
    "__spec__", "__package__", "__cached__",
})

_VERDICT_RE = re.compile(r"UNMEASURED|BLOCKED|CLEAR")

USAGE = ("usage: f78_census_writer_driver.py drive PROBE SPEC.json"
         " | f78_census_writer_driver.py verify SPEC.json")


def _fail(stage, detail):
    print("stage=%s=%s-failed %s" % (STAGE, stage, detail))
    sys.exit(RC_INFRA)


def _read_spec(spec_path, stage):
    try:
        with open(spec_path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError) as exc:
        # doctrine 4: an unreadable spec is an INFRASTRUCTURE red, never
        # an empty fixture sailing through at rc 0.
        _fail(stage, "spec unreadable/unparseable: %s" % exc)


def _literal_assign(node):
    """Name bound by a module-scope literal assignment, else None.

    Plain or annotated, single Name target, value acceptable to
    ast.literal_eval (constants and containers of them). Lifting NEVER
    executes probe expressions: literal_eval only. Anything richer
    returns None here and, if the closure needs it, shows up as
    F78_EXTRACT_UNRESOLVED -- fabricated constants are forbidden.
    """
    if (isinstance(node, ast.Assign) and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)):
        target, value = node.targets[0].id, node.value
    elif (isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name) and node.value is not None):
        target, value = node.target.id, node.value
    else:
        return None
    try:
        ast.literal_eval(value)
    except Exception:
        return None
    return target


def _globals_needed(tbl):
    """Module-scope names referenced (or assigned via `global`) anywhere in
    symbol table `tbl` or its nested tables. symtable -- not a raw name walk
    -- so locals, parameters, nested scopes, defaults and decorators are
    scoped the way the compiler scopes them."""
    needed, stack, seen = set(), [tbl], set()
    while stack:
        cur = stack.pop()
        if id(cur) in seen:
            continue
        seen.add(id(cur))
        for sym in cur.get_symbols():
            if sym.is_global() and (sym.is_referenced() or sym.is_assigned()):
                needed.add(sym.get_name())
        stack.extend(cur.get_children())
    return needed


def _unresolved_names(src, probe_path, keep, const_names):
    """Module-scope names the kept closure will look up but this driver did
    not provide. Fail-SAFE bias: basename table matching may over-collect
    (noise a reviewer can read), never under-collect (silence that mints a
    false probe red). A broken check reports a sentinel, which is red under
    every leg grep -- never green."""
    provided = (set(keep) | const_names | PRELUDE_NAMES | _TYPING_PUBLIC
                | _NS_DUNDERS | set(dir(builtins)))
    try:
        root = symtable.symtable(src, probe_path, "exec")
        needed = set()
        for child in root.get_children():
            if child.get_name().split(".")[-1] in keep:
                needed |= _globals_needed(child)
        return needed - provided
    except Exception:
        return {"<unresolved-check-failed>"}


def _load_writer(probe_path):
    """Return (writer, report): the probe's REAL _persist_adapter_census --
    its published source bytes, its helper closure, its literal constants,
    its __future__ flags -- with the module's top-level statements NEVER
    executed."""
    try:
        with open(probe_path, "r", encoding="utf-8") as fh:
            src = fh.read()
    except OSError as exc:
        _fail("extract", "probe unreadable: %s" % exc)  # doctrine 4: unreadable is not empty
    try:
        tree = ast.parse(src, filename=probe_path)
    except SyntaxError as exc:
        _fail("extract", "probe did not parse: %s" % exc)

    futures, consts, funcs = [], [], {}
    const_names = set()
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module == "__future__":
            futures.append(node)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            funcs[node.name] = node
        elif _literal_assign(node) is not None:
            name = _literal_assign(node)
            if name not in const_names:
                consts.append(node)
                const_names.add(name)

    if WRITER_ENTRY not in funcs:
        _fail("extract", "%s not found at probe top level" % WRITER_ENTRY)

    keep, pending = set(), {WRITER_ENTRY}
    while pending:
        name = pending.pop()
        if name in keep or name not in funcs:
            continue
        keep.add(name)
        for sub in ast.walk(funcs[name]):
            if isinstance(sub, ast.Name) and isinstance(sub.ctx, ast.Load):
                pending.add(sub.id)

    unresolved = _unresolved_names(src, probe_path, keep, const_names)
    body = list(futures) + consts + [funcs[n] for n in sorted(keep)]
    module = ast.fix_missing_locations(ast.Module(body=body, type_ignores=[]))

    ns = {"__name__": "__f78_persistence__", "__file__": probe_path}
    try:
        exec(compile(PRELUDE, "<f78-prelude>", "exec"), ns)
        exec(compile(module, probe_path, "exec"), ns)
    except BaseException as exc:
        # SystemExit included: lifted def/const statements have no business
        # exiting; if they do, it is INFRA red on the published vocabulary,
        # never a bare traceback at an off-vocabulary rc.
        _fail("exec", "%s: %s" % (type(exc).__name__, exc))

    report = {
        "funcs": len(keep),
        "consts": len(consts),
        "futures": sorted({a.name for f in futures for a in f.names}),
        "unresolved": sorted(unresolved),
    }
    return ns[WRITER_ENTRY], report


def cmd_drive(probe_path, spec_path):
    spec = _read_spec(spec_path, "drive")
    if not isinstance(spec, dict) or not isinstance(spec.get("kwargs"), dict):
        _fail("drive", "spec must be an object carrying a 'kwargs' object "
                       "keyed on the writer's parameters")
    kwargs = spec["kwargs"]
    writer, report = _load_writer(probe_path)

    out_path = spec.get("out_path") or kwargs.get("out_path")
    if not isinstance(out_path, str) or not out_path:
        out_path = None

    tape = io.StringIO()
    raised, exited, result = "none", "none", None
    try:
        with contextlib.redirect_stdout(tape), contextlib.redirect_stderr(tape):
            result = writer(**kwargs)
    except SystemExit as exc:
        # the writer using the probe's own 0/1/3 exit vocabulary is a
        # MEASURED behaviour, reported, never re-judged
        raised, exited = "SystemExit", str(exc.code)
    except BaseException as exc:
        raised = "%s: %s" % (type(exc).__name__, exc)
    text = tape.getvalue()

    print("stage=%s=drove rc=0" % STAGE)
    print("F78_EXTRACT_FUNCS=%d" % report["funcs"])
    print("F78_EXTRACT_CONSTS=%d" % report["consts"])
    print("F78_EXTRACT_FUTURES=%s" % (",".join(report["futures"]) or "none"))
    print("F78_EXTRACT_UNRESOLVED=%s"
          % (",".join(report["unresolved"]) or "none"))
    print("F78_RAISED=%s" % raised)
    print("F78_EXIT=%s" % exited)
    if out_path is None:
        # doctrine 4: unmeasured is UNKNOWN, never a forged 0 that would
        # vacuously pass MUST_FIRE half (a).
        print("F78_OUT_PATH=unspecified")
        print("F78_OUT_EXISTS=unknown")
        print("F78_OUT_PARENT_PRESENT=unknown")
    else:
        print("F78_OUT_PATH=%s" % out_path)
        print("F78_OUT_EXISTS=%d" % (1 if os.path.isfile(out_path) else 0))
        print("F78_OUT_PARENT_PRESENT=%d"
              % (1 if os.path.isdir(os.path.dirname(os.path.abspath(out_path)))
                 else 0))
    tokens = _VERDICT_RE.findall(text)
    print("F78_VERDICT_TOKENS=%s" % (",".join(tokens) if tokens else "none"))
    if result is not None:
        try:
            rendered = json.dumps(result, sort_keys=True)
        except (TypeError, ValueError):
            rendered = repr(result)
        print("F78_RESULT=%s" % rendered)
    for line in text.splitlines():
        print("F78_TOOL|%s" % line)  # the tool's own words, quoted
    return 0


def _all_strings(node):
    if isinstance(node, str):
        yield node
    elif isinstance(node, dict):
        for k, v in node.items():
            yield from _all_strings(k)
            yield from _all_strings(v)
    elif isinstance(node, (list, tuple)):
        for item in node:
            yield from _all_strings(item)


def cmd_verify(spec_path):
    spec = _read_spec(spec_path, "verified")
    if not isinstance(spec, dict):
        _fail("verified", "spec must be an object")
    artifact = spec.get("artifact")
    if not isinstance(artifact, str) or not artifact:
        # path ABSENT FROM THE FIXTURE = fixture defect = infra red.
        # (A path whose FILE is absent is the opposite: a measured probe
        # red at rc 0, handled below.)
        _fail("verified", "spec.artifact missing: the leg must name the "
                          "artifact the contract obliges the writer to produce")
    fqns = spec.get("expect_fqns")
    if (not isinstance(fqns, list) or not fqns
            or any(not isinstance(f, str) or not f for f in fqns)):
        # doctrine 1: a zero-unit expectation is an UNMEASURED check, hence
        # a FIXTURE defect -- infra red here, never a vacuous 0-of-0 pass.
        _fail("verified", "spec.expect_fqns must be a non-empty list of strings")
    denominator = spec.get("expect_denominator", len(fqns))
    if not isinstance(denominator, int) or denominator < 0:
        _fail("verified", "spec.expect_denominator must be a non-negative int")

    parse_ok, payload = False, None
    try:
        with open(artifact, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
        parse_ok = True
    except (OSError, ValueError):
        parse_ok = False

    print("stage=%s=verified rc=0" % STAGE)
    if not parse_ok:
        # A missing/unparsable artifact the writer was contracted to produce
        # is a MEASURED fact about the code under test: rc 0, and every leg
        # grep below is driven red on purpose.
        print("F78_JSON_PARSE=fail")
        print("F78_FQNS_FOUND=0 of %d" % len(fqns))
        print("F78_FQNS_MISSING=%s" % ",".join(fqns))
        print("F78_ARTIFACT_ROWS=absent")
        return 0

    strings = set(_all_strings(payload))
    missing = [f for f in fqns if f not in strings]
    print("F78_JSON_PARSE=ok")
    print("F78_FQNS_FOUND=%d of %d" % (len(fqns) - len(missing), len(fqns)))
    print("F78_FQNS_MISSING=%s" % (",".join(missing) if missing else "none"))
    rows = payload.get("rows") if isinstance(payload, dict) else None
    if isinstance(rows, list):
        print("F78_ARTIFACT_ROWS=%d of %d" % (len(rows), denominator))
    else:
        # doctrine 4: unmeasured is not zero; no fabricated '0 of m' line
        # may satisfy an explicit-denominator grep by accident.
        print("F78_ARTIFACT_ROWS=absent")
    return 0


def main(argv):
    if len(argv) == 4 and argv[1] == "drive":
        return cmd_drive(argv[2], argv[3])
    if len(argv) == 3 and argv[1] == "verify":
        return cmd_verify(argv[2])
    sys.stderr.write(USAGE + "\n")
    return RC_USAGE


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv))
    except SystemExit:
        raise  # _fail()'s rc-15, usage rc-2 and sys.exit(main()) pass through
    except BaseException as exc:
        # off-vocabulary exits are forbidden: any unexpected driver bug is
        # INFRA red on the published vocabulary, never a bare rc-1 traceback.
        print("stage=%s=driver-failed %s: %s"
              % (STAGE, type(exc).__name__, exc))
        sys.exit(RC_INFRA)
