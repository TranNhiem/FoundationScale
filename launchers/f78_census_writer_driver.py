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
    drops exactly the assertion halves these legs exist to indict;
  * hand-authoring the writer's kwargs while the signature was NOT yet
    in evidence: REJECTED last merge as a claim broader than its
    evidence -- correctly, then. The signature and writer body are now
    the evidence (Leg integration below): the rejection is superseded,
    and recording the reversal HERE keeps it an audit event rather than
    a quiet re-pick.

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
    top-level DEFINITIONS it references -- functions AND classes -- via
    an ast.Name Load walk over each kept body, emitted in ORIGINAL
    top-level order (class bases/decorators and def defaults evaluate at
    exec time, exactly as the probe's own module evaluates them). The
    writer's refusal contract is a raised top-level class,
    _CensusRefusal: un-lifted, the guard dies as NameError, a DRIVER red
    impersonating probe behaviour on the leg that exists to watch it,
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
  F78_EXTRACT_FUNCS=<k>            top-level DEFINITIONS lifted --
      functions plus closure-referenced classes (e.g. _CensusRefusal);
      extraction denominator (doctrine 2)
  F78_EXTRACT_CONSTS=<c>
  F78_EXTRACT_FUTURES=<csv|none>
  F78_EXTRACT_UNRESOLVED=<csv|none>   module-scope names the closure wants
      that were NOT lifted. !=none means a subsequent NameError red is a
      DRIVER extraction gap, not a probe finding -- doctrine 5 requires
      this distinction be visible. Sentinal '<unresolved-check-failed>'
      means the check itself broke: red either way, never green.
  F78_SYNTH_MODE=<kwargs|fixture>  how the writer call was assembled;
      'fixture' reflects the LIVE signature (driver extension made for the
      legs: the heredoc it replaces regexed the def line of a neutered
      exec copy, and its substring classes would have mapped 'total' onto
      the path class via the substring 'out' -- matching here is by word
      segment, never by whole-name substring)
  F78_KWARGS_BOUND=<k> of <n>      explicit mode: the pre-call bind's own
      examined set -- k supplied keys against the n non-splat parameters
      of the LIVE signature (doctrine 2; fixture mode's denominator is
      F78_SYNTH_PATH/DATA instead, since its map comes FROM the signature)
  F78_SYNTH_PATH=<csv>             fixture mode: params mapped <- out_path
  F78_SYNTH_DATA=<csv>             fixture mode: params mapped <- fixture
  F78_FIXTURE_ROWS=<k>             fixture mode: attachment count actually
      sent. The refusal leg's k is 0 BY CONSTRUCTION (the mutation IS the
      empty set); printing it keeps 'empty' examined, not asserted
      (doctrine 1/2).
  F78_RAISED=<none|Type: msg>      writer behaviour, including SystemExit
  F78_EXIT=<none|code>             under the probe's own 0/1/3 vocabulary
  F78_OUT_PATH=<path|unspecified>
  F78_OUT_EXISTS=<0|1|unknown>     doctrine 4: an out-path the driver
      cannot see is UNKNOWN, never 0 -- a forged 0 here would vacuously
      pass MUST_FIRE half (a). Legs pin the path via spec.out_path.
  F78_OUT_PARENT_PRESENT=<0|1|unknown>
  F78_VERDICT_TOKENS=<csv|none>    UNMEASURED/BLOCKED/CLEAR, in order,
      duplicates kept -- indexed over BOTH captured channels the lifted
      call can speak on: the stdout+stderr tape AND the refusal's own
      exception/SystemExit message text. A refusal raised below main()'s
      verdict printer never reaches the tape, yet still speaks -- and its
      words are already quoted verbatim in F78_RAISED. INDEXED, never
      minted: if the code under test says no verdict word, none appears.
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
  F78_FQN_OK|<fqn>                 one line per FOUND expected fqn, positive
      form: leg greps must key on this, never the bare name -- the bare name
      also appears verbatim inside F78_FQNS_MISSING on failure, so a
      bare-name grep goes green exactly when the artifact is absent
  F78_ARTIFACT_ROWS=<r> of <m>     when the artifact carries a rows list,
      else F78_ARTIFACT_ROWS=absent -- unmeasured is not zero (doctrine 4),
      and no fabricated '0 of m' line may satisfy the grep by accident.

Leg integration (one-line swap per leg at the exec site; every pre-existing
assertion grep is preserved byte-for-byte. The F78 leg block sits outside
the audited window of the 3676-line harness -- locate it with
  grep -n F78_STAGE launchers/test_launcher_contracts.sh
-- and doctrine 4 forbids shipping a blind byte-anchor against it, so the
swap ships as these verbatim instructions, not as an Edit):

  BOTH legs ship EXPLICIT-MODE specs. The writer's LIVE signature and
  body are in evidence now (launchers/lora_target_census.py:238-240 ff.):

      def _persist_adapter_census(
          out_path, rows, population, hf_model_path, targets, total
      ) -> None:

  Hand-authoring this kwargs set was rejected last merge as naming
  parameters nowhere in evidence; the rejection is SUPERSEDED -- the
  signature above IS the evidence. The driver keeps
  inspect.signature(...).bind(**kwargs) in front of every explicit call,
  so a renamed/added/omitted parameter re-fails as a NAMED rc-15
  drive-failed before the writer runs: drift goes red, never silently
  mis-bound. Per-parameter sourcing -- read off the writer body,
  AUTHORED by the leg, never guessed by the driver:

    out_path      the leg's artifact path; ALSO pin spec.out_path to the
                  same string so F78_OUT_EXISTS stays measured. Feeds
                  _atomic_write_json(out_path, payload) (line 355).
    rows          the fixture's per-target records; each record carries
                  its FULL 'found' list (the writer's own docstring:
                  rows carries the FULL per-target match lists, the same
                  found lists CENSUS_SAMPLE previews only 2 of).
                  MUST_FIRE: every found is [] -- the zero attachment
                  set BY CONSTRUCTION; the empty-set guard must raise
                  _CensusRefusal BEFORE any temp file exists.
                  MUST_PASS: two records whose founds carry the two
                  expected names.
    population    {} -- the fixture authors NO module dims, so the
                  writer's all-or-nothing rule (lines 321-338) writes
                  every entry as a bare stem BY DESIGN; a partially
                  dimmed fixture would poke a gate-refusal contract the
                  leg is not aiming at.
    hf_model_path a leg-authored provenance string; in visible evidence
                  it is interpolated ONLY into the artifact 'source'
                  text (line 348) -- never opened, never globbed.
    targets       the fixture's pattern names, joined into 'source'
                  (line 348); e.g. the two 'target' keys of the rows.
    total         the offerable-module count the fixture CLAIMS, as an
                  int MATCHING the leg's own rows (line 349). The leg
                  authors the arithmetic; ANY driver-side derivation
                  (len(fixture), a constant, skipping the parameter) is
                  exactly the guessing the rc-15 refusal exists to
                  refuse (doctrine 4/5, symmetric).

  expect_fqns note (verify spec): with population={} the artifact's
  adapter_modules carry bare stems; the probe's _artifact_stem defines
  the found-name -> stem mapping. Author expect_fqns from
  _artifact_stem's REAL behaviour -- read it, do not guess it
  (doctrine 4). cmd_verify searches every string in the artifact JSON,
  so the leg greps key on F78_FQN_OK|<fqn> (positive form), never the
  bare name, which also appears inside F78_FQNS_MISSING on failure.

  MUST_FIRE (zero attachment set; explicit-kwargs spec as above):
      out=$(python3 launchers/f78_census_writer_driver.py drive \
            launchers/lora_target_census.py "$f78_spec.json") || true
      keep asserting BOTH halves against "$out":
        (a) grep 'F78_OUT_EXISTS=0'                -- no census file afterwards
            ('unknown' NEVER matches: unmeasured is not pass)
        (b) grep 'F78_VERDICT_TOKENS=.*UNMEASURED' -- the verdict is UNMEASURED
      additive: grep 'F78_EXTRACT_UNRESOLVED=none' -- the driver's own
      extraction denominator; !=none re-reads the red as a DRIVER gap;
      grep 'F78_RAISED=.*CensusRefusal' -- the refusal fires as the
      writer's PROMISED type, lifted, not a NameError miscarriage of an
      un-lifted guard; grep 'F78_KWARGS_BOUND=6 of 6' -- the bind's
      examined set, printed, never assumed.

  MUST_PASS (2-module explicit-kwargs spec as above):
      out=$(python3 launchers/f78_census_writer_driver.py drive \
            launchers/lora_target_census.py "$f78_spec.json") || true
      ver=$(python3 launchers/f78_census_writer_driver.py verify "$f78_verify.json")
      keep asserting against "$ver":
        grep 'F78_JSON_PARSE=ok'
        grep "F78_FQN_OK|<fqn>" for EACH expected fqn (positive form; the
        bare name also appears inside F78_FQNS_MISSING on failure, so a
        bare-name grep false-greens exactly when the artifact is absent)
        and grep 'F78_FQNS_MISSING=none'
        grep the explicit denominator 'F78_FQNS_FOUND=<k> of <n>' and/or
        'F78_ARTIFACT_ROWS=<r> of <m>' -- never a bare numerator
      additive: grep 'F78_EXTRACT_UNRESOLVED=none' against "$out", and
      'F78_KWARGS_BOUND=6 of 6' -- the whole signature bound and
      examined (doctrine 2), not just the happy path asserted.

Spec shapes (JSON, authored by the leg fixtures):
  drive : EITHER explicit mode:
            {"kwargs": {<writer's parameter names>: <values>},
             "out_path": "explicit artifact path -- set it when the
                          writer's out-parameter is not named out_path, so
                          F78_OUT_EXISTS stays measured instead of unknown"}
            kwargs keys pass through as **kwargs to the real writer, but
            only after they are BOUND against the lifted writer's LIVE
            signature (inspect.signature .bind): a key the signature
            rejects, or a required parameter the spec omits, is
            stage=F78_STAGE=drive-failed rc 15 naming the failure -- a
            stale-fixture red, never a mid-call TypeError at rc 0
            masquerading as a measured probe finding (doctrine 5).
          OR fixture mode (RETAINED -- every rc-15 refusal below stays
          live and reachable -- but NOT what the F78 legs ship):
            {"fixture": [<attachment elements>], "out_path": "..."}
            the driver reflects the writer's LIVE signature and maps
            path-class params (name segment out|path|dest|file|json) <-
            out_path and attachment-class params (segment prefix attach|
            module|row|entri|record|parent|found|result|population|
            census|target|fixture) <- list(fixture); defaulted params keep
            their own defaults; a param in NO class -- or in BOTH -- or a
            zero candidate on either side is stage=F78_STAGE=drive-failed
            rc 15 ('refusing to guess'), the same fail-closed contract the
            heredoc honoured at its exit 13, but mapped over the live
            signature instead of a regexed neutered copy. The EMPTY list
            is the refusal fixture; an absent 'fixture' key with no
            'kwargs' object is the fixture defect (doctrine 4). Fixture
            kwargs come FROM the signature and then RUN through the same
            inspect.signature(...).bind(**kwargs) guard explicit-kwargs
            mode keeps: an accept by construction that must still RUN to
            be a control (a never-run guard is no more a control than a
            never-fired detector, doctrine 3), going red as a NAMED rc 15
            the day the live signature drifts -- never a silent mis-bind
            or a mid-call TypeError at rc 0 that a leg could misread as a
            probe finding (doctrine 5).
            Against the #78 signature this mode rc-15s naming param=total
            BY DESIGN: total has no honest driver-side source (it is the
            offerable-population size, not the attachment-feed length),
            while hf_model_path would silently take the artifact path via
            its 'path' segment and population/targets the attachment
            list -- the guesses the refusal exists to refuse. The legs
            therefore author explicit kwargs (Leg integration). Fixture
            mode remains for writers whose signatures honestly map, and
            remains the standing proof that an unsourcable parameter is
            a NAMED rc 15, never a fabricated value.
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
import asyncio
import builtins
import contextlib
import inspect
import io
import json
import pathlib
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

# Param-name segment classes for fixture-mode synthesis (see
# _synthesize_kwargs): matched against word SEGMENTS of the live
# signature, never substrings of the whole name.
_PATH_SEGS = frozenset({"out", "path", "dest", "file", "json"})
_DATA_PREFIXES = ("attach", "module", "row", "entri", "record", "parent",
                  "found", "result", "population", "census", "target",
                  "fixture")

USAGE = ("usage: f78_census_writer_driver.py drive PROBE SPEC.json"
         " | f78_census_writer_driver.py verify SPEC.json")


def _fail(stage, detail):
    print(f"stage={STAGE}={stage}-failed {detail}")
    sys.exit(RC_INFRA)


def _read_spec(spec_path, stage):
    try:
        with pathlib.Path(spec_path).open(encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError) as exc:
        # doctrine 4: an unreadable spec is an INFRASTRUCTURE red, never
        # an empty fixture sailing through at rc 0.
        _fail(stage, f"spec unreadable/unparseable: {exc}")


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
        with pathlib.Path(probe_path).open(encoding="utf-8") as fh:
            src = fh.read()
    except OSError as exc:
        _fail("extract", f"probe unreadable: {exc}")  # doctrine 4: unreadable is not empty
    try:
        tree = ast.parse(src, filename=probe_path)
    except SyntaxError as exc:
        _fail("extract", f"probe did not parse: {exc}")

    futures, consts, defs, order = [], [], {}, []
    const_names = set()
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module == "__future__":
            futures.append(node)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef,
                               ast.ClassDef)):
            # Classes are lifted too: the writer's refusal CONTRACT is
            # 'raise _CensusRefusal' (its own docstring). Un-lifted, the
            # top-level class reads NameError in this namespace -- the
            # guard would miscarry as a DRIVER gap on exactly the leg
            # whose purpose is watching the guard fire (doctrine 3).
            if node.name not in defs:
                order.append(node.name)
            defs[node.name] = node
        elif _literal_assign(node) is not None:
            name = _literal_assign(node)
            if name not in const_names:
                consts.append(node)
                const_names.add(name)

    if WRITER_ENTRY not in defs:
        _fail("extract", f"{WRITER_ENTRY} not found at probe top level")

    keep, pending = set(), {WRITER_ENTRY}
    while pending:
        name = pending.pop()
        if name in keep or name not in defs:
            continue
        keep.add(name)
        for sub in ast.walk(defs[name]):
            if isinstance(sub, ast.Name) and isinstance(sub.ctx, ast.Load):
                pending.add(sub.id)

    unresolved = _unresolved_names(src, probe_path, keep, const_names)
    # ORIGINAL top-level order, not sorted: class bases/decorators (and
    # any def defaults) are evaluated at exec time, exactly the way the
    # probe's own module would evaluate them.
    body = (list(futures) + consts
            + [defs[n] for n in order if n in keep])
    module = ast.fix_missing_locations(ast.Module(body=body, type_ignores=[]))

    ns = {"__name__": "__f78_persistence__", "__file__": probe_path}
    try:
        exec(compile(PRELUDE, "<f78-prelude>", "exec"), ns)
        exec(compile(module, probe_path, "exec"), ns)
    except BaseException as exc:
        # SystemExit included: lifted def/const statements have no business
        # exiting; if they do, it is INFRA red on the published vocabulary,
        # never a bare traceback at an off-vocabulary rc.
        _fail("exec", f"{type(exc).__name__}: {exc}")

    report = {
        "funcs": len(keep),
        "consts": len(consts),
        "futures": sorted({a.name for f in futures for a in f.names}),
        "unresolved": sorted(unresolved),
    }
    return ns[WRITER_ENTRY], report


def _split_param_segments(name):
    """Identifier -> its lowercase word segments. Parameters are
    identifiers, so segments split on runs of non-letters (underscores,
    digits). Matching is by SEGMENT, never by substring of the whole
    name: the heredoc this driver replaces regexed whole names and would
    have mapped a 'total' parameter onto the PATH class via the substring
    'out' -- a fabricated value class is a false-green engine (doctrine
    5, symmetric)."""
    return [s for s in re.split(r"[^a-z]+", name.lower()) if s]


def _synthesize_kwargs(writer, spec):
    """Fixture mode: reflect the writer's LIVE signature and map by
    segment class -- path-class params take spec.out_path,
    attachment-class params take a copy of the fixture, defaulted params
    are left at the writer's own defaults, and a param in NO class (or in
    BOTH) fails closed at stage=drive-failed: guessing a value for it
    would paraphrase the writer's contract. A probe refactor therefore
    re-shapes the map LEGIBLY (an rc-15 stage-named red a reviewer can
    read) instead of stranding the leg the way the hand-enumerated stub
    set did."""
    out_path = spec.get("out_path")
    if not isinstance(out_path, str) or not out_path:
        _fail("drive", "arg-synthesis needs spec.out_path: the path-class "
                       "parameter takes it, and an un-pinned artifact path "
                       "reads F78_OUT_EXISTS=unknown -- never 0 (doctrine 4)")
    fixture = spec.get("fixture")
    if not isinstance(fixture, list):
        _fail("drive", "spec.fixture must be a list -- the EMPTY list is "
                       "the refusal fixture; an absent/non-list key is a "
                       "fixture defect, not an empty fixture (doctrine 4)")
    try:
        params = list(inspect.signature(writer).parameters.values())
    except (TypeError, ValueError) as exc:
        _fail("drive", f"arg-synthesis cannot reflect the writer: {exc}")
    kwargs, path_hits, data_hits = {}, [], []
    for p in params:
        if p.kind in (inspect.Parameter.VAR_POSITIONAL,
                      inspect.Parameter.VAR_KEYWORD):
            continue  # splats are never fabricated; the writer gets what it gets
        segs = _split_param_segments(p.name)
        is_path = any(s in _PATH_SEGS for s in segs)
        is_data = any(s.startswith(_DATA_PREFIXES) for s in segs)
        if is_path and is_data:
            _fail("drive", f"arg-synthesis param={p.name} matches BOTH the "
                           "path and the attachment class; refusing to guess")
        elif is_path:
            path_hits.append(p.name)
            kwargs[p.name] = out_path
        elif is_data:
            data_hits.append(p.name)
            kwargs[p.name] = list(fixture)
        elif p.default is not inspect.Parameter.empty:
            continue  # the writer's own default speaks for it
        else:
            _fail("drive", f"arg-synthesis param={p.name}: no honest value "
                           "available; refusing to guess")
    if not path_hits or not data_hits:
        _fail("drive", "arg-synthesis mapped path="
                       + (",".join(path_hits) or "none") + " attachment="
                       + (",".join(data_hits) or "none")
                       + ": a zero candidate on either side means the "
                       "writer's shape changed; refusing to run blind")
    # Fixture kwargs are bound against the LIVE signature as a guard, the
    # same control explicit-kwargs mode keeps in cmd_drive. Synthesis
    # builds kwargs FROM that signature, so bind() accepts them by
    # construction -- but a guard is only a control if it RUNS (doctrine
    # 3), and a future signature drift that ever breaks construction is
    # this named rc-15 red, never a silent mis-bind or a mid-call
    # TypeError at rc 0 that a leg could misread as a measured probe
    # finding (doctrine 5).
    try:
        inspect.signature(writer).bind(**kwargs)
    except TypeError as exc:
        _fail("drive", "synthesized kwargs do not bind the lifted writer's "
                       f"signature {inspect.signature(writer)}: {exc}")
    return kwargs, {"path": path_hits, "data": data_hits,
                    "fixture_rows": len(fixture)}


def cmd_drive(probe_path, spec_path):
    spec = _read_spec(spec_path, "drive")
    if not isinstance(spec, dict):
        _fail("drive", "spec must be an object carrying a 'kwargs' object "
                       "keyed on the writer's parameters, or a 'fixture' "
                       "list for signature-mapped synthesis")
    writer, report = _load_writer(probe_path)
    if isinstance(spec.get("kwargs"), dict):
        # Fail-closed fixture binding (the control the merge keeps from
        # the alternative sample): explicit kwargs are bound against the
        # lifted writer's REAL signature BEFORE the call, so a renamed,
        # added, or omitted required parameter is a named drive-failed at
        # rc 15 -- never a mid-call TypeError at rc 0 that a leg could
        # misread as a measured probe finding (doctrine 5).
        sig = inspect.signature(writer)
        try:
            bound = sig.bind(**spec["kwargs"])
        except TypeError as exc:
            _fail("drive", f"spec kwargs do not bind the lifted writer's "
                           f"signature {sig}: {exc}")
        kwargs, synth = spec["kwargs"], None
    elif "fixture" in spec:
        kwargs, synth = _synthesize_kwargs(writer, spec)
    else:
        _fail("drive", "spec carries neither a 'kwargs' object nor a "
                       "'fixture' list; there is no honest call to make")

    out_path = spec.get("out_path") or kwargs.get("out_path")
    if not isinstance(out_path, str) or not out_path:
        out_path = None

    tape = io.StringIO()
    raised, exited, result = "none", "none", None
    try:
        with contextlib.redirect_stdout(tape), contextlib.redirect_stderr(tape):
            result = writer(**kwargs)
            if inspect.iscoroutine(result):
                # an async writer awaited HERE is the driver's business: an
                # un-awaited coroutine would run zero units yet red the leg
                # in the probe's clothing (doctrine 1 units, doctrine 5 scope)
                result = asyncio.run(result)
    except SystemExit as exc:
        # the writer using the probe's own 0/1/3 exit vocabulary is a
        # MEASURED behaviour, reported, never re-judged. The code is kept
        # in the F78_RAISED text too, so a SystemExit(<msg>) refusal's
        # own words stay reachable by the verdict-token index below --
        # still within the F78_RAISED=<none|Type: msg> contract.
        raised, exited = f"SystemExit: {exc.code}", str(exc.code)
    except BaseException as exc:
        raised = f"{type(exc).__name__}: {exc}"
    text = tape.getvalue()

    print(f"stage={STAGE}=drove rc=0")
    print(f"F78_EXTRACT_FUNCS={report['funcs']}")
    print(f"F78_EXTRACT_CONSTS={report['consts']}")
    print(f"F78_EXTRACT_FUTURES={','.join(report['futures']) or 'none'}")
    print(f"F78_EXTRACT_UNRESOLVED={','.join(report['unresolved']) or 'none'}")
    print(f"F78_SYNTH_MODE={'fixture' if synth is not None else 'kwargs'}")
    if synth is not None:
        print(f"F78_SYNTH_PATH={','.join(synth['path'])}")
        print(f"F78_SYNTH_DATA={','.join(synth['data'])}")
        print(f"F78_FIXTURE_ROWS={synth['fixture_rows']}")
    else:
        # explicit mode: the pre-call BIND is the guard, so its examined
        # set gets a denominator too (doctrine 2) -- k supplied keys
        # against the n bindable (non-splat) parameters of the LIVE
        # signature; greppable as '<k> of <n>', never a bare numerator.
        n_bindable = sum(
            1 for p in sig.parameters.values()
            if p.kind not in (inspect.Parameter.VAR_POSITIONAL,
                              inspect.Parameter.VAR_KEYWORD))
        print(f"F78_KWARGS_BOUND={len(bound.arguments)} of {n_bindable}")
    print(f"F78_RAISED={raised}")
    print(f"F78_EXIT={exited}")
    if out_path is None:
        # doctrine 4: unmeasured is UNKNOWN, never a forged 0 that would
        # vacuously pass MUST_FIRE half (a).
        print("F78_OUT_PATH=unspecified")
        print("F78_OUT_EXISTS=unknown")
        print("F78_OUT_PARENT_PRESENT=unknown")
    else:
        print(f"F78_OUT_PATH={out_path}")
        print(f"F78_OUT_EXISTS={1 if pathlib.Path(out_path).is_file() else 0}")
        parent = pathlib.Path(out_path).resolve().parent
        print(f"F78_OUT_PARENT_PRESENT={1 if parent.is_dir() else 0}")
    # doctrine 4: scan BOTH captured channels the lifted call can speak
    # on -- the stdout+stderr tape AND the refusal's own message text
    # (already quoted verbatim in F78_RAISED). A refusal raised below
    # main()'s verdict printer never reaches the tape; its message is
    # still the code-under-test's own words, INDEXED here, never minted.
    tokens = _VERDICT_RE.findall(text) + _VERDICT_RE.findall(raised)
    print(f"F78_VERDICT_TOKENS={','.join(tokens) if tokens else 'none'}")
    if result is not None:
        try:
            rendered = json.dumps(result, sort_keys=True)
        except (TypeError, ValueError):
            rendered = repr(result)
        print(f"F78_RESULT={rendered}")
    for line in text.splitlines():
        print(f"F78_TOOL|{line}")  # the tool's own words, quoted
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
        with pathlib.Path(artifact).open(encoding="utf-8") as fh:
            payload = json.load(fh)
        parse_ok = True
    except (OSError, ValueError):
        parse_ok = False

    print(f"stage={STAGE}=verified rc=0")
    if not parse_ok:
        # A missing/unparsable artifact the writer was contracted to produce
        # is a MEASURED fact about the code under test: rc 0, and every leg
        # grep below is driven red on purpose.
        print("F78_JSON_PARSE=fail")
        print(f"F78_FQNS_FOUND=0 of {len(fqns)}")
        print(f"F78_FQNS_MISSING={','.join(fqns)}")
        print("F78_ARTIFACT_ROWS=absent")
        return 0

    strings = set(_all_strings(payload))
    missing = [f for f in fqns if f not in strings]
    print("F78_JSON_PARSE=ok")
    print(f"F78_FQNS_FOUND={len(fqns) - len(missing)} of {len(fqns)}")
    print(f"F78_FQNS_MISSING={','.join(missing) if missing else 'none'}")
    for f in fqns:
        if f not in missing:
            # positive found-evidence, one unit per line (doctrine 2): a leg
            # greps 'F78_FQN_OK|<fqn>' per expected name and can never be
            # satisfied by the F78_FQNS_MISSING list of a failed verify
            print(f"F78_FQN_OK|{f}")
    rows = payload.get("rows") if isinstance(payload, dict) else None
    if isinstance(rows, list):
        print(f"F78_ARTIFACT_ROWS={len(rows)} of {denominator}")
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
        print(f"stage={STAGE}=driver-failed {type(exc).__name__}: {exc}")
        sys.exit(RC_INFRA)
