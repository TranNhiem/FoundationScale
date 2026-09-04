#!/usr/bin/env python3
"""#192: one tolerance, two questions -- split the resume-proof knobs.

WHAT / WHY. _resume_continuity_verdict(pre_per_rank, post_per_rank, pre_scalar,
tolerance, world_size) took ONE tolerance and spent it on two unrelated questions:
restore fidelity (restore_delta = max_i |post[i] - pre[i]|; > tolerance is RED and
final -- the question the proof exists to answer) and cross-rank agreement
(rank_invariant = spread_before <= tolerance and spread_after <= tolerance -- a
property of the INSTRUMENT: do 8 ranks compute the same fixed-eval loss?, not of
resume). Measured on the real Gemma arm: job 37336 at --resume-tolerance 0.0005
read restore_delta 0.0 with cross_rank_spread_before_save ==
cross_rank_spread_after_resume == 0.2940967082977295, status unmeasured_cross_rank,
unmeasured:["resume.fixed_eval_rank_invariance"]; job 37319, the same arm at
--resume-tolerance 10.0, recorded unmeasured:[] -- zero abstentions. The second run
bought a cross-rank pass by raising a threshold that ALSO governs restore fidelity:
at 10.0 a completely broken restore -- a delta of 9.9 -- is a PASS. The knob that
admits a known instrument artifact silently disables the RED the proof exists for.
One scalar, two questions, and widening it for one destroys the other.

THE FIX. Two knobs, and a self-calibrating default for the second.
--resume-tolerance KEEPS its name and its meaning as the restore-fidelity
tolerance (existing configs, LAUNCH.md, the gates and the recorded runs all use
it). --rank-agreement-tolerance is added as an OPTIONAL float, routed through the
same _sourced machinery, the same env fallback (none) and the same validation
shape as resume_tolerance, with one difference: unset is legal, and when it IS set
it must be finite and greater than zero on the same terms. It is threaded into
_resume_continuity_verdict as a keyword parameter defaulting to None so every
existing call site and test keeps its meaning. The verdict now always records
cross_rank_spread_delta (after minus before, signed -- the resume-attributable
term; on job 37336 it is exactly 0.0), rank_agreement_tolerance (the effective
value), rank_agreement_tolerance_source ("explicit" or "self-calibrated"),
rank_agreement_absolute (bool, or None when the question was not asked) and
rank_agreement_preserved (spread_after <= max(spread_before, restore_tolerance)).
rank_invariant keeps its name and its meaning as the ABSOLUTE claim and may be
True ONLY under an explicit rank_agreement_tolerance with both spreads under it --
self-calibration must never mint a pass, because a floor derived from the same run
it judges cannot certify that run's absolute agreement. The status ladder's
precedence is unchanged: restore_delta > restore tolerance is RED and final no
matter how wide the rank knob is; an explicit rank tolerance with both spreads
under it is a pass; anything else is unmeasured_cross_rank whose reason STATES
what was measured -- and when resume WORSENED agreement the reason says so and
names both spreads. The legacy path (pre_per_rank is None) keeps its behaviour
byte-for-byte, restore_term_legacy marking included; its new fields are None. The
PHASE_JSON payload keeps the existing "tolerance" key with its restore meaning and
adds the new fields next to it, and every display string that said "within
tolerance" now says which one.

THE GATE. This stage locates the shipped text by unique assembled anchors, refuses
rather than guesses when any anchor count is off, requires a MUST_FIRE premise
(the pre-image genuinely spends one knob on both questions: zero occurrences of
rank_agreement and the conflated rank_invariant expression present), verifies
every post-condition on the produced text BEFORE anything is written, proves
byte-idempotence by applying the transform to its own output, and ast.parses and
compiles the result. Eight controls execute the REAL patched function: both images
are written to temp files in the target's own directory (so the mandatory
fs_model_root import resolves exactly as it does for the real entrypoint) and
imported as modules; the rule is never re-implemented inside a control. If Python
cannot import an image the stage is 95 UNMEASURED, printed and returned -- never
skipped. The build host has no torch and never will (the trainer runs inside a
container; the build runs on a laptop), so when an image import raises
ModuleNotFoundError the missing module is stubbed in sys.modules -- a DECLARED
stub, bounded at 12 retries, never twice for the same name, fully removed from
sys.modules afterwards -- and a contamination guard then proves by measurement
(inspect.getsource plus an AST walk, never a grep) that the rule under test
carries zero references a stub could resolve; a verdict a stub can reach is 95,
and if a stub can reach only C8's helpers then C8 alone reports UNMEASURED (stub
reachable) while the other seven controls still run and still count. Exit codes:
0 PASS / 5 RED / 95 UNMEASURED / 96 REFUSE.
"""

from __future__ import annotations

import ast
import importlib.util
import inspect
import linecache
import os
import pathlib
import sys
import types
from typing import Any

TARGET = pathlib.Path(__file__).resolve().parent / "h100" / "gen" / "fs_train.fixed.py"
MARK = "fs192:"
N_CONTROLS = 8
FN = "_resume" + "_continuity_verdict"

# Needles are ASSEMBLED, never written as one literal: this stage counts them in the
# target, and a source that contains its own needle is inside its own denominator.
N_FIELD = "    resume" + "_tolerance: SourcedValue"
N_ARG = 'add_argument("--resume' + '-tolerance", type=float)'
N_SRC = "resume" + "_tolerance = _sourced("
N_VAL = "resume" + "_tolerance must be finite and greater than zero"
N_CTOR = "resume" + "_tolerance=resume" + "_tolerance,"
N_FN = "def _resume" + "_continuity_verdict("
N_DEF = "def resume" + "_and_prove("
N_CALL = "continuity = _resume" + "_continuity_verdict("
N_TOL = '"tolerance": tolerance,'
N_DISP = "fixed-eval rank agreement MEASURED within"
N_SUM = "fixed loss restored within tolerance"
N_FEX = '"fixed_examples": Deno' + "minatedCount("
N_OLDCON = 'result["cross_rank_spread_before' + '_save"] <= tolerance'

# Post-image needles (assembled on the same terms).
P_ARG = 'add_argument("--rank' + '-agreement-tolerance", type=float)'
P_FIELD = "rank_agreement" + "_tolerance: SourcedValue"
P_SRC = "rank_agreement" + "_tolerance = _sourced("
P_CTOR = (
    "resume" + "_tolerance=resume" + "_tolerance,\n        rank_agreement"
    + "_tolerance=rank_agreement" + "_tolerance,"
)
P_CALL = "rank_agreement" + "_tolerance=rank_agreement" + "_tolerance,\n    )"
P_OLDSIG = (
    "def _resume" + "_continuity_verdict(pre_per_rank, post_per_rank, pre_scalar, "
    "tolerance, world_size):"
)
P_NEWSIG = (
    "def _resume" + "_continuity_verdict(pre_per_rank, post_per_rank, pre_scalar, "
    "tolerance, world_size, rank_agreement" + "_tolerance=None):"
)
P_SELF = '"self' + '-calibrated"'
P_EXPL = "spread_before <= rank_agreement" + "_tolerance"
P_LEGACY = "legacy (conflated) restore statistic"
P_RESTOL = '"restore' + '_tolerance": tolerance'
P_DELTA = '"cross_rank_spread' + '_delta"'
P_ABS = '"rank_agreement' + '_absolute"'
P_PRES = '"rank_agreement' + '_preserved"'
P_SRCKEY = '"rank_agreement' + '_tolerance_source"'
P_OLDDISP = "MEASURED within {tolerance"

# --- insertion blocks ---------------------------------------------------------
ARG_BLOCK = '    parser.add_argument("--rank-agreement-tolerance", type=float)\n'

SRC_BLOCK = '''    # fs192: the cross-rank knob. Same _sourced machinery, same env fallback
    # (none) and same validation shape as resume_tolerance -- with one difference:
    # unset is legal (required=False), and when it IS set it must be finite and
    # greater than zero on the same terms (checked below).
    rank_agreement_tolerance = _sourced(
        "rank_agreement_tolerance",
        args.rank_agreement_tolerance,
        env,
        None,
        str,
        required=False,
    )
'''

VAL_BLOCK = '''    if rank_agreement_tolerance.value is not None and (
        float(rank_agreement_tolerance.value) <= 0.0
        or not math.isfinite(float(rank_agreement_tolerance.value))
    ):
        raise ContractError(
            "rank_agreement_tolerance must be finite and greater than zero"
        )
'''

FIELD_BLOCK = "    rank_agreement_tolerance: SourcedValue\n"

CTOR_BLOCK = "        rank_agreement_tolerance=rank_agreement_tolerance,\n"

CALL_BLOCK = '''    # fs192: the cross-rank knob is OPTIONAL; None means the absolute question
    # was not asked, and the verdict self-calibrates only the preservation term.
    rank_agreement_raw = config.rank_agreement_tolerance.value
    rank_agreement_tolerance = (
        float(rank_agreement_raw) if rank_agreement_raw is not None else None
    )
    continuity = _resume_continuity_verdict(
        pre_per_rank,
        post_per_rank,
        float(fixed_pre),
        tolerance,
        bundle.world_size,
        rank_agreement_tolerance=rank_agreement_tolerance,
    )
'''

TOL1_BLOCK = '''            "restore_tolerance": tolerance,
            "rank_agreement_tolerance": continuity["rank_agreement_tolerance"],
            "rank_agreement_tolerance_source": continuity["rank_agreement_tolerance_source"],
            "cross_rank_spread_delta": continuity["cross_rank_spread_delta"],
'''

TOL2_BLOCK = '''            "rank_agreement_tolerance": continuity["rank_agreement_tolerance"],
            "rank_agreement_tolerance_source": continuity["rank_agreement_tolerance_source"],
            "rank_agreement_absolute": continuity["rank_agreement_absolute"],
            "rank_agreement_preserved": continuity["rank_agreement_preserved"],
            "cross_rank_spread_delta": continuity["cross_rank_spread_delta"],
'''

DISP_BLOCK = '''            "display": (
                "fixed-eval rank agreement MEASURED within the explicit "
                "rank-agreement tolerance "
                f"{continuity['rank_agreement_tolerance']:.8g}"
                if continuity["rank_invariant"]
                else "fixed-eval rank invariance UNMEASURED: the fixed-eval loss "
                "takes distinct bit-identical values across ranks (spread before "
                f"save {continuity['cross_rank_spread_before_save']}, spread after "
                f"resume {continuity['cross_rank_spread_after_resume']}, spread "
                f"delta {continuity['cross_rank_spread_delta']}); the restore "
                "verdict stands on its own per-rank terms against the restore "
                f"tolerance {tolerance:.8g}"
            ),
'''

SUM_BLOCK = '''            "display": "PROVED: optimizer step and fixed loss restored within the restore tolerance",
'''

# The patched pure decision rule. It replaces the fs178 image of the same function
# wholesale (the region between its def line and resume_and_prove). The legacy
# branch is carried over BYTE-FOR-BYTE; the rule stays free of torch/dist names --
# P3 proves that with an AST census, not a grep.
PURE_BLOCK = '''def _resume_continuity_verdict(pre_per_rank, post_per_rank, pre_scalar, tolerance, world_size, rank_agreement_tolerance=None):
    """Decide restore fidelity and cross-rank agreement as TWO statistics, never one.

    fs178: pure by construction -- plain lists and scalars in, a plain dict out; no
    torch, no distributed state. Measured basis (job 37319 plus a zero-training
    8-rank forensic run): rank 0's restore was bit-exact (before_save ==
    after_resume == 0.5986318588256836) while the shipped MAX of
    |own_post - rank0_pre| read 0.17570888996124268; within-rank restore delta 0.0
    and 0 of 341 parameter-fingerprint keys changed across save/load; the fixed-eval
    loss takes exactly two bit-identical values across ranks (e.g. spread
    1.1025075912475586) on a NEVER-SAVED runtime -- the ranks disagreeing is a
    property of the instrument, present before any checkpoint exists, and must never
    be named resume.

    fs192: the one tolerance used to be spent on BOTH questions. Job 37336
    (--resume-tolerance 0.0005) measured restore_delta 0.0 with
    cross_rank_spread_before_save == cross_rank_spread_after_resume ==
    0.2940967082977295 and abstained unmeasured_cross_rank; job 37319, the same arm
    at --resume-tolerance 10.0, recorded zero abstentions -- a cross-rank pass
    bought by raising the threshold that ALSO governs restore fidelity, and at 10.0
    a completely broken restore (delta 9.9) is a PASS. `tolerance` keeps its name
    and is now the RESTORE-fidelity knob only; `rank_agreement_tolerance` (keyword,
    default None) is the cross-rank knob. When it is unset, the before-save spread
    -- the measured noise floor of the instrument -- calibrates only the
    PRESERVATION question (did resume worsen the agreement the instrument already
    had?), never the absolute one: self-calibration must never set rank_invariant
    True, because a floor derived from the same run it judges cannot certify that
    run's absolute agreement.

    Statuses: "refuse" (malformed evidence -- a length that disagrees with
    world_size or a non-finite entry: refuse rather than guess); "red" (restore
    fidelity broken -- delta over the RESTORE tolerance, final, and never governed
    by the rank knob no matter how wide it is); "pass" (restore holds AND an
    EXPLICIT rank_agreement_tolerance was supplied AND both spreads come in under
    it); "unmeasured_cross_rank" (restore holds but the absolute cross-rank question
    was not asked or not satisfied, or the per-rank pre-save vector is absent so the
    restore term fell back to the legacy rank-0-scalar comparison -- a DECLARED
    UNMEASURED, never a clean pass).
    """
    result = {
        "restore_delta": None,
        "cross_rank_spread_before_save": None,
        "cross_rank_spread_after_resume": None,
        "cross_rank_spread_delta": None,
        "rank_agreement_tolerance": None,
        "rank_agreement_tolerance_source": None,
        "rank_agreement_absolute": None,
        "rank_agreement_preserved": None,
        "rank_invariant": False,
        "status": "refuse",
        "reason": "",
        "restore_term_legacy": False,
        "pre_per_rank": None,
        "post_per_rank": None,
    }

    def _coerce(vec, label):
        if not isinstance(vec, (list, tuple)) or len(vec) != world_size:
            got = len(vec) if isinstance(vec, (list, tuple)) else repr(vec)
            return None, (
                label + " length disagrees with world_size: observed " + str(got)
                + " of an expected " + str(world_size) + " entries; refusing to guess"
            )
        values = []
        for index, value in enumerate(vec):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
            ):
                return None, (
                    label + "[" + str(index) + "] is not a finite number: "
                    + repr(value) + "; refusing to guess"
                )
            values.append(float(value))
        return values, None

    if isinstance(world_size, bool) or not isinstance(world_size, int) or world_size < 1:
        result["reason"] = "world_size " + repr(world_size) + " is an invalid denominator"
        return result
    if (
        isinstance(tolerance, bool)
        or not isinstance(tolerance, (int, float))
        or not math.isfinite(float(tolerance))
        or float(tolerance) <= 0.0
    ):
        result["reason"] = "tolerance " + repr(tolerance) + " is not a positive finite number"
        return result
    if rank_agreement_tolerance is not None and (
        isinstance(rank_agreement_tolerance, bool)
        or not isinstance(rank_agreement_tolerance, (int, float))
        or not math.isfinite(float(rank_agreement_tolerance))
        or float(rank_agreement_tolerance) <= 0.0
    ):
        result["reason"] = (
            "rank_agreement_tolerance " + repr(rank_agreement_tolerance)
            + " is not a positive finite number"
        )
        return result
    if (
        isinstance(pre_scalar, bool)
        or not isinstance(pre_scalar, (int, float))
        or not math.isfinite(float(pre_scalar))
    ):
        result["reason"] = "the rank-0 manifest scalar " + repr(pre_scalar) + " is not finite"
        return result
    tolerance = float(tolerance)
    if rank_agreement_tolerance is not None:
        rank_agreement_tolerance = float(rank_agreement_tolerance)

    post, err = _coerce(post_per_rank, "post_per_rank")
    if err is not None:
        result["reason"] = err
        return result
    result["post_per_rank"] = post
    result["cross_rank_spread_after_resume"] = max(post) - min(post)

    if pre_per_rank is None:
        # Legacy fallback: a foreign or future writer left no per-rank pre-save values
        # on disk. Compare every rank's post against the rank-0 manifest scalar,
        # explicitly marked as the legacy CONFLATED statistic, and declare the
        # cross-rank term UNMEASURED -- never silently assume the ranks agreed.
        legacy_delta = max(abs(p - float(pre_scalar)) for p in post)
        result["restore_delta"] = legacy_delta
        result["restore_term_legacy"] = True
        result["rank_invariant"] = False
        if legacy_delta > tolerance:
            result["status"] = "red"
            result["reason"] = (
                "legacy (conflated) restore statistic " + format(legacy_delta, ".8g")
                + " exceeds tolerance " + format(tolerance, ".8g")
                + "; per-rank pre-save values were not recorded"
            )
        else:
            result["status"] = "unmeasured_cross_rank"
            result["reason"] = (
                "cross-rank term UNMEASURED: rank payload(s) carried no per-rank "
                "fixed-loss-before-save value; the restore term used the legacy "
                "rank-0 manifest scalar (the conflated statistic), marked legacy"
            )
        return result

    pre, err = _coerce(pre_per_rank, "pre_per_rank")
    if err is not None:
        result["reason"] = err
        return result
    result["pre_per_rank"] = pre
    result["cross_rank_spread_before_save"] = max(pre) - min(pre)
    restore_delta = max(abs(post[i] - pre[i]) for i in range(world_size))
    result["restore_delta"] = restore_delta
    spread_before = result["cross_rank_spread_before_save"]
    spread_after = result["cross_rank_spread_after_resume"]
    # fs192: the resume-attributable term, signed -- negative means resume
    # TIGHTENED agreement. On job 37336 it is exactly 0.0: resume did not worsen
    # rank agreement AT ALL, and no previous image of this proof could say so.
    result["cross_rank_spread_delta"] = spread_after - spread_before
    # Did resume preserve whatever agreement the instrument already had? The
    # before-save spread is the measured noise floor of the instrument; the
    # restore tolerance floors it so a bit-exact instrument is not punished for
    # last-bit kernel scheduling either.
    result["rank_agreement_preserved"] = spread_after <= max(spread_before, tolerance)
    if rank_agreement_tolerance is not None:
        result["rank_agreement_tolerance"] = rank_agreement_tolerance
        result["rank_agreement_tolerance_source"] = "explicit"
        # The boundary is strict: a spread exactly equal to the tolerance does NOT
        # fire (">", never ">="). The before-save spread is the measured noise
        # floor of the instrument -- asserting a tolerance without ever measuring
        # what the instrument can resolve is exactly how a bit-exact restore got
        # named a resume failure.
        rank_invariant = (
            spread_before <= rank_agreement_tolerance
            and spread_after <= rank_agreement_tolerance
        )
        result["rank_agreement_absolute"] = rank_invariant
    else:
        # Self-calibrated: the effective floor is derived from THIS run, so it can
        # judge only preservation -- never the absolute question. rank_invariant
        # stays False and rank_agreement_absolute stays None (the question was not
        # asked): a floor derived from the same run it judges cannot certify that
        # run's absolute agreement.
        result["rank_agreement_tolerance"] = max(spread_before, tolerance)
        result["rank_agreement_tolerance_source"] = "self-calibrated"
        rank_invariant = False
    result["rank_invariant"] = rank_invariant
    if restore_delta > tolerance:
        # RED and final: restore fidelity is compared against the RESTORE knob and
        # nothing else, no matter how wide rank_agreement_tolerance is -- that is
        # the whole point of #192. A real restore defect must never be laundered
        # into "the ranks merely disagree".
        result["status"] = "red"
        result["reason"] = (
            "restore fidelity broken: max over " + str(world_size) + " rank(s) of "
            "|own_after_resume - own_before_save| = " + format(restore_delta, ".8g")
            + " exceeds restore tolerance " + format(tolerance, ".8g")
        )
    elif rank_agreement_tolerance is not None and rank_invariant:
        result["status"] = "pass"
        result["reason"] = (
            "restore fidelity holds (restore_delta " + format(restore_delta, ".8g")
            + " <= restore tolerance " + format(tolerance, ".8g")
            + ") and the ranks agree within the explicit rank-agreement tolerance "
            + format(rank_agreement_tolerance, ".8g")
        )
    else:
        result["status"] = "unmeasured_cross_rank"
        spread_delta = result["cross_rank_spread_delta"]
        if result["rank_agreement_preserved"]:
            preserved_clause = (
                "resume did not worsen rank agreement (cross-rank spread delta "
                + format(spread_delta, ".8g") + ")"
            )
        else:
            # A real signal about resume, not an instrument artifact: name it and
            # name both spreads.
            preserved_clause = (
                "resume WORSENED rank agreement: spread before save "
                + format(spread_before, ".8g") + ", spread after resume "
                + format(spread_after, ".8g")
            )
        if rank_agreement_tolerance is None:
            absolute_clause = (
                "the absolute cross-rank question is UNMEASURED because no "
                "rank-agreement tolerance was declared"
            )
        else:
            absolute_clause = (
                "the ranks do not agree in absolute terms under the explicit "
                "rank-agreement tolerance " + format(rank_agreement_tolerance, ".8g")
            )
        result["reason"] = (
            "restore fidelity holds (restore_delta " + format(restore_delta, ".8g")
            + " <= restore tolerance " + format(tolerance, ".8g")
            + "); the ranks do not agree in absolute terms (spread after resume "
            + format(spread_after, ".8g") + "); " + preserved_clause + "; "
            + absolute_clause + "; declared UNMEASURED, never folded into the "
            "resume verdict"
        )
    return result


'''


def _stderr(msg: str) -> None:
    print(msg, file=sys.stderr)


def _locate(lines: list[str]) -> dict:
    """Locate the shipped text by assembled anchors; report position lists."""

    def hits(sub: str, start: int = 0) -> list[int]:
        return [i for i in range(start, len(lines)) if sub in lines[i]]

    locs: dict = {
        "field": hits(N_FIELD),
        "arg": hits(N_ARG),
        "src": hits(N_SRC),
        "val": hits(N_VAL),
        "ctor": hits(N_CTOR),
        "fn": hits(N_FN),
        "defn": hits(N_DEF),
        "call": hits(N_CALL),
        "tol": hits(N_TOL),
        "disp": hits(N_DISP),
        "summ": hits(N_SUM),
    }
    locs["j_call"] = None
    if locs["call"]:
        i = locs["call"][0]
        locs["j_call"] = next(
            (k for k in range(i + 1, len(lines)) if lines[k].strip() == ")"), None
        )
    locs["d0"] = None
    locs["d1"] = None
    if len(locs["tol"]) == 2:
        start = locs["tol"][1]
        d0 = next(
            (
                k
                for k in range(start, len(lines))
                if lines[k].strip().startswith('"display": (')
            ),
            None,
        )
        locs["d0"] = d0
        if d0 is not None:
            locs["d1"] = next(
                (k for k in range(d0 + 1, len(lines)) if lines[k].strip() == "),"),
                None,
            )
    return locs


def _counts_valid(locs: dict) -> bool:
    return (
        len(locs.get("field", [])) == 1
        and len(locs.get("arg", [])) == 1
        and len(locs.get("src", [])) == 1
        and len(locs.get("val", [])) == 1
        and len(locs.get("ctor", [])) == 1
        and len(locs.get("fn", [])) == 1
        and len(locs.get("defn", [])) == 1
        and len(locs.get("call", [])) == 1
        and len(locs.get("tol", [])) == 2
        and len(locs.get("disp", [])) == 1
        and len(locs.get("summ", [])) == 1
    )


def _locs_valid(locs: dict) -> bool:
    if not _counts_valid(locs):
        return False
    chain = (
        locs["field"][0] < locs["arg"][0] < locs["src"][0] < locs["val"][0]
        < locs["ctor"][0] < locs["fn"][0] < locs["defn"][0] < locs["call"][0]
        < locs["tol"][0] < locs["tol"][1] < locs["disp"][0] < locs["summ"][0]
    )
    derived = (
        locs.get("j_call") is not None
        and locs.get("d0") is not None
        and locs.get("d1") is not None
    )
    if derived:
        derived = (
            locs["call"][0] < locs["j_call"] < locs["tol"][0]
            and locs["tol"][1] < locs["d0"] < locs["disp"][0] < locs["d1"]
            and locs["d1"] < locs["summ"][0]
        )
    return chain and derived


def _transform(text: str) -> tuple[str, dict, bool]:
    """Apply the eleven edits; byte-exact no-op when the marker is already present."""
    if MARK in text:
        return text, {}, True
    lines = text.splitlines(keepends=True)
    locs = _locate(lines)
    if not _locs_valid(locs):
        return text, locs, False
    insert_after = {
        locs["arg"][0]: ARG_BLOCK,
        locs["src"][0]: SRC_BLOCK,
        locs["val"][0]: VAL_BLOCK,
        locs["field"][0]: FIELD_BLOCK,
        locs["ctor"][0]: CTOR_BLOCK,
        locs["tol"][0]: TOL1_BLOCK,
    }
    replace_span = {
        locs["fn"][0]: (locs["defn"][0] - 1, PURE_BLOCK),
        locs["call"][0]: (locs["j_call"], CALL_BLOCK),
        locs["tol"][1]: (locs["tol"][1], TOL2_BLOCK),
        locs["d0"]: (locs["d1"], DISP_BLOCK),
        locs["summ"][0]: (locs["summ"][0], SUM_BLOCK),
    }
    out: list[str] = []
    i = 0
    while i < len(lines):
        if i in replace_span:
            end, block = replace_span[i]
            out.append(block)
            i = end + 1
        else:
            out.append(lines[i])
            if i in insert_after:
                out.append(insert_after[i])
            i += 1
    return "".join(out), locs, False


def _pure_census(source: str) -> tuple[bool, int, int, int]:
    """AST census over the pure function: exists at module scope, references no
    torch/dist name. Returns (exists_module_scope, node_count, torch_refs, dist_refs).
    A census, not a grep -- the premise of this stage is host-certifiability."""
    tree = ast.parse(source)
    fn = None
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == FN:
            fn = node
            break
    if fn is None:
        return False, 0, 0, 0
    nodes = list(ast.walk(fn))
    tref = sum(
        1
        for n in nodes
        if (isinstance(n, ast.Name) and n.id == "torch")
        or (
            isinstance(n, ast.Attribute)
            and isinstance(n.value, ast.Name)
            and n.value.id == "torch"
        )
    )
    dref = sum(
        1
        for n in nodes
        if (isinstance(n, ast.Name) and n.id == "dist")
        or (
            isinstance(n, ast.Attribute)
            and isinstance(n.value, ast.Name)
            and n.value.id == "dist"
        )
    )
    return True, len(nodes), tref, dref


def _post_image_gates(text: str) -> list[tuple[str, bool, str]]:
    """Gates provable without locators (also reused on the already-applied path)."""
    g: list[tuple[str, bool, str]] = []
    n_arg = text.count(P_ARG)
    n_field = text.count(P_FIELD)
    n_src = text.count(P_SRC)
    n_ctor = text.count(P_CTOR)
    n_call = text.count(P_CALL)
    g.append((
        "P1",
        n_arg == 1 and n_field == 1 and n_src == 1 and n_ctor == 1 and n_call == 1,
        f"knob plumbing: argparse flag count={n_arg} need=1, config dataclass field "
        f"count={n_field} need=1, _sourced resolution count={n_src} need=1, "
        f"RunConfig constructor threading count={n_ctor} need=1, verdict call-site "
        f"keyword count={n_call} need=1",
    ))
    n_oldsig = text.count(P_OLDSIG)
    n_newsig = text.count(P_NEWSIG)
    g.append((
        "P2",
        n_oldsig == 0 and n_newsig == 1,
        f"signature split: old 5-positional signature count={n_oldsig} need=0, new "
        f"keyword-with-None-default signature count={n_newsig} need=1 (every "
        "existing call site keeps its meaning)",
    ))
    try:
        ok, n_nodes, tref, dref = _pure_census(text)
    except SyntaxError as exc:
        ok, n_nodes, tref, dref = False, 0, -1, -1
        g.append(("P3", False, f"pure-function census could not parse the image: {exc}"))
    else:
        g.append((
            "P3",
            ok and tref == 0 and dref == 0,
            f"pure function {FN} at module scope={ok}; AST census over {n_nodes} "
            f"node(s): torch refs={tref} need=0, dist refs={dref} need=0 (a census, "
            "not a grep -- the rule must be certifiable without torch or a cluster)",
        ))
    n_self = text.count(P_SELF)
    n_oldcon = text.count(N_OLDCON)
    n_expl = text.count(P_EXPL)
    n_leg = text.count(P_LEGACY)
    g.append((
        "P4",
        n_self == 1 and n_oldcon == 0 and n_expl == 1 and n_leg == 1,
        f"self-calibration never mints a pass: 'self-calibrated' source count="
        f"{n_self} need=1, the old conflated rank_invariant expression count="
        f"{n_oldcon} need=0, the explicit-only invariant expression count={n_expl} "
        f"need=1; legacy branch carried byte-for-byte (conflated-statistic reason "
        f"count={n_leg} need=1)",
    ))
    n_restol = text.count(P_RESTOL)
    n_tol = text.count(N_TOL)
    n_delta = text.count(P_DELTA)
    n_abs = text.count(P_ABS)
    n_pres = text.count(P_PRES)
    n_srckey = text.count(P_SRCKEY)
    n_sum = text.count(N_SUM)
    n_olddisp = text.count(P_OLDDISP)
    g.append((
        "P5",
        n_restol == 1 and n_tol == 1 and n_delta >= 3 and n_abs >= 2 and n_pres >= 2
        and n_srckey >= 3 and n_sum == 0 and n_olddisp == 0,
        f"payload attribution: keep 'tolerance' with its restore meaning count="
        f"{n_tol} need=1, added 'restore_tolerance' count={n_restol} need=1, "
        f"cross_rank_spread_delta count={n_delta} need>=3, rank_agreement_absolute "
        f"count={n_abs} need>=2, rank_agreement_preserved count={n_pres} need>=2, "
        f"rank_agreement_tolerance_source count={n_srckey} need>=3; ambiguous "
        f"displays gone: 'restored within tolerance' count={n_sum} need=0, "
        f"'MEASURED within {{tolerance' count={n_olddisp} need=0",
    ))
    try:
        ast.parse(text)
        compile(text, str(TARGET), "exec")
        g.append(("P6", True, "ast.parse + compile clean"))
    except (SyntaxError, ValueError) as exc:
        g.append(("P6", False, f"parse/compile: {type(exc).__name__}: {exc}"))
    return g


def _missing_module_name(exc: ModuleNotFoundError) -> str | None:
    """The missing module's dotted name: exc.name when the import system set it,
    else parsed out of the 'No module named ...' message."""
    name = getattr(exc, "name", None)
    if name:
        return str(name)
    message = str(exc)
    marker = "No module named "
    if marker not in message:
        return None
    tail = message.split(marker, 1)[1].lstrip()
    if tail[:1] in ("'", '"'):
        end = tail.find(tail[0], 1)
        if end > 1:
            return tail[1:end]
    words = tail.split()
    return words[0].strip("'\"") if words else None


class _AbsentStubBase:
    """The concrete base a stub resolves to when trainer code subclasses one.

    fs192: measured on this target -- `class IndexedSyntheticDataset(Dataset[dict[str,
    str]])` puts a stub INSTANCE in a bases tuple, so Python takes the metaclass from
    type(base) and calls _AbsentModuleStub(name, bases, ns): "TypeError: __init__()
    takes 3 positional arguments but 4 were given", and the whole image fails to
    import. __mro_entries__ on the stub redirects the base to this plain class, which
    is a real type and carries no behaviour a control could accidentally exercise.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    def __class_getitem__(cls, item: Any) -> Any:
        return cls


class _AbsentModuleStub(types.ModuleType):
    """A DECLARED stand-in for a module the build host does not have (torch and
    its submodules on this laptop; the trainer runs inside a container). It
    exists only so a control image can be IMPORTED and the pure rule executed,
    and it is never silent: _import_image reports every name it stubbed, and the
    contamination guard in _controls re-verifies by AST walk that the rule under
    test carries zero references a stub could resolve.

    Any attribute resolves to another stub, registered in sys.modules under its
    dotted name so `import torch.distributed as dist` and `from torch import nn`
    both work; the stub itself is callable, subscriptable, usable as a decorator
    and as a context manager, because module-scope trainer code does all of
    those. Dunder lookups raise AttributeError so the import system never
    mistakes the stub for a package."""

    def __init__(self, dotted: str, installed: dict) -> None:
        super().__init__(dotted)
        self.__dict__["_fs192_dotted"] = dotted
        self.__dict__["_fs192_installed"] = installed

    def __getattr__(self, name: str) -> Any:
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        dotted = self.__dict__["_fs192_dotted"] + "." + name
        current = sys.modules.get(dotted)
        if current is not None:
            return current
        stub = _AbsentModuleStub(dotted, self.__dict__["_fs192_installed"])
        self.__dict__["_fs192_installed"][dotted] = stub
        sys.modules[dotted] = stub
        return stub

    def __mro_entries__(self, bases: tuple) -> tuple:
        # Used as a base class: hand Python a real type. See _AbsentStubBase.
        return (_AbsentStubBase,)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return _AbsentModuleStub(
            self.__dict__["_fs192_dotted"] + "()",
            self.__dict__["_fs192_installed"],
        )

    def __getitem__(self, item: Any) -> Any:
        return _AbsentModuleStub(
            self.__dict__["_fs192_dotted"] + "[...]",
            self.__dict__["_fs192_installed"],
        )

    def __enter__(self) -> Any:
        return self

    def __exit__(self, *exc_info: Any) -> bool:
        return False


_CONTROL_IMAGES: list[pathlib.Path] = []


def _stub_reachable(source: str, stubbed: list[str]) -> str | None:
    """AST walk (a census, never a grep) over one function's source: the first
    Name/Attribute reference a stubbed module could resolve -- directly, or
    through a name an import bound from one -- else None."""
    tops = {name.split(".")[0] for name in stubbed}
    tree = ast.parse(source)
    bound: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] in tops:
                    bound.add(alias.asname or alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if (node.module or "").split(".")[0] in tops:
                for alias in node.names:
                    bound.add(alias.asname or alias.name)
    forbidden = tops | bound
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id in forbidden:
            return node.id
        if isinstance(node, ast.Attribute):
            parts: list[str] = []
            cursor = node
            while isinstance(cursor, ast.Attribute):
                parts.append(cursor.attr)
                cursor = cursor.value
            if isinstance(cursor, ast.Name) and cursor.id in forbidden:
                return ".".join([cursor.id] + parts[::-1])
    return None


def _import_image(source: str, tag: str) -> tuple[Any, str | None, list[str]]:
    """Write an image to a temp file in the TARGET's own directory -- so the
    mandatory fs_model_root import resolves exactly as it does for the real
    entrypoint -- and import it as a module.

    The build host has no torch and never will (the trainer runs inside a
    container; the build runs on a laptop), so when the import raises
    ModuleNotFoundError the missing module is stubbed in sys.modules -- a
    DECLARED _AbsentModuleStub, never a silent one -- and the import retried:
    until it succeeds, a different exception is raised, or the bounded retry
    count (12) is exhausted, and never twice for the same missing name (a repeat
    after stubbing is a stub defect and stops the loop). Every name installed is
    removed from sys.modules in the finally, any pre-existing entry restored,
    and the sorted names are returned as `stubbed` so the contamination guard in
    _controls can prove by measurement that no stub can participate in a
    verdict. Returns (module, None, stubbed) or (None, error, stubbed)."""
    directory = TARGET.parent
    path = directory / (".fs192_control_" + tag + "_" + str(os.getpid()) + ".py")
    # fs192: bound before the try so the finally can restore sys.modules even if
    # the write or the spec build fails.
    spec_name: str | None = None
    displaced_self = None
    keep_image = False
    try:
        path.write_text(source, "utf-8")
    except OSError as exc:
        return None, f"could not write control image {path}: {exc}", []
    installed: dict[str, Any] = {}
    displaced: dict[str, Any] = {}
    stubbed: set[str] = set()
    try:
        spec = importlib.util.spec_from_file_location(
            "fs192_control_" + tag + "_" + str(os.getpid()), path
        )
        if spec is None or spec.loader is None:
            return None, "importlib could not build a module spec", []
        module = importlib.util.module_from_spec(spec)
        # fs192: register the module under its own spec name BEFORE exec_module.
        # dataclasses._is_type resolves a decorated class's namespace with
        # sys.modules.get(cls.__module__).__dict__, and an unregistered module
        # makes that None -- "AttributeError: 'NoneType' object has no attribute
        # '__dict__'" at the first @dataclass, measured on this very target. Same
        # defect as fix68 D1. Removed again in the finally below.
        spec_name = spec.name
        displaced_self = sys.modules.get(spec_name)
        sys.modules[spec_name] = module
        sys.path.insert(0, str(directory))
        try:
            seen: set[str] = set()
            retries = 0
            while True:
                try:
                    spec.loader.exec_module(module)
                    break
                except ModuleNotFoundError as exc:
                    name = _missing_module_name(exc)
                    if name is None:
                        return None, f"{type(exc).__name__}: {exc}", sorted(stubbed)
                    if name in seen or name in stubbed:
                        return None, (
                            f"stub defect: {name!r} was already stubbed yet the "
                            "import still raises ModuleNotFoundError for it: "
                            f"{exc}"
                        ), sorted(stubbed)
                    if retries >= 12:
                        return None, (
                            "stub retry budget (12) exhausted; the import still "
                            f"misses {name!r} after stubbing: "
                            + ", ".join(sorted(stubbed))
                        ), sorted(stubbed)
                    seen.add(name)
                    retries += 1
                    parts = name.split(".")
                    for i in range(1, len(parts) + 1):
                        prefix = ".".join(parts[:i])
                        current = sys.modules.get(prefix)
                        if isinstance(current, _AbsentModuleStub):
                            continue
                        if current is not None and prefix != name:
                            continue  # a real parent module stays; never displace it
                        if current is not None:
                            displaced[prefix] = current
                        stub = _AbsentModuleStub(prefix, installed)
                        installed[prefix] = stub
                        sys.modules[prefix] = stub
                        stubbed.add(prefix)
        finally:
            try:
                sys.path.remove(str(directory))
            except ValueError:
                pass
        # fs192: priming linecache is NOT enough -- inspect.getsource calls
        # linecache.checkcache first, which stats the file and drops the entry the
        # moment it is gone, so the contamination guard died with "OSError: could
        # not get source code" on an image that had imported perfectly. Keep the
        # temp image on disk for as long as the controls need it and record the
        # path on the module; _controls unlinks it in its own finally.
        linecache.getlines(str(path), module.__dict__)
        module.__dict__["_fs192_control_path"] = str(path)
        _CONTROL_IMAGES.append(path)
        keep_image = True
        return module, None, sorted(stubbed)
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}", sorted(stubbed)
    finally:
        # fs192: undo the self-registration too. exec_module has finished by the
        # time anything here runs, so the caller's reference keeps the module
        # alive; sys.modules is left exactly as it was found.
        if displaced_self is not None:
            sys.modules[spec_name] = displaced_self
        elif spec_name is not None:
            sys.modules.pop(spec_name, None)
        for dotted in installed:
            if dotted in displaced:
                sys.modules[dotted] = displaced[dotted]
            else:
                sys.modules.pop(dotted, None)
        if not keep_image:
            try:
                path.unlink()
            except OSError:
                pass


def _reap_control_images() -> None:
    """Remove the control images kept alive for inspect.getsource. Called from the
    one place that owns the whole control run, so a guard can read real source and
    the build directory is still left clean."""
    while _CONTROL_IMAGES:
        leftover = _CONTROL_IMAGES.pop()
        try:
            leftover.unlink()
        except OSError:
            pass


def _controls(pre_text: str, new: str) -> tuple[int, int, list[str], str | None]:
    """Execute the REAL patched function: both images are written to temp files and
    imported as modules, and _resume_continuity_verdict is called directly. The
    rule is never re-implemented inside a control. When the build host lacks a
    module an image imports (torch on this laptop), _import_image stubs it --
    declared, bounded, and fully removed from sys.modules afterwards -- and the
    contamination guard then proves by measurement (inspect.getsource plus an
    AST walk, never a grep) that the rule under test carries zero references a
    stub could resolve. If the stub could contaminate the verdict the stage is
    95 UNMEASURED naming the offending reference; if it could contaminate only
    C8's helpers (build_parser/resolve_contract) then C8 alone reports
    UNMEASURED (stub reachable) and the pass denominator drops to 7 of 8 while
    the other seven controls still run and still count. Returns (passed, total,
    notes, unmeasured); a non-None unmeasured is the 95 contract."""
    notes: list[str] = []
    pre_mod, err, pre_stubbed = _import_image(pre_text, "pre")
    if pre_mod is None:
        msg = "could not import the PRE-image module: " + str(err)
        notes.append("UNMEASURED 95: " + msg)
        return 0, N_CONTROLS, notes, msg
    new_mod, err, new_stubbed = _import_image(new, "post")
    if new_mod is None:
        msg = "could not import the PATCHED module: " + str(err)
        notes.append("UNMEASURED 95: " + msg)
        return 0, N_CONTROLS, notes, msg
    stubbed = sorted(set(pre_stubbed) | set(new_stubbed))
    if stubbed:
        notes.append(
            "environment: both images imported with stub(s) for absent "
            "module(s): " + ", ".join(stubbed) + "\n"
            "  -- the build host has no torch; gate P3's AST census "
            "independently certifies the rule under test carries zero torch "
            "references, and the contamination guard re-verified that on the "
            "imported object"
        )
    else:
        notes.append("environment: no stubs; both images imported as-is")
    pre_fn = getattr(pre_mod, FN, None)
    post_fn = getattr(new_mod, FN, None)
    if not callable(pre_fn) or not callable(post_fn):
        msg = FN + " is not callable in an imported image"
        notes.append("UNMEASURED 95: " + msg)
        return 0, N_CONTROLS, notes, msg
    total = N_CONTROLS
    c8_trusted = True
    c8_detail = ""
    if stubbed:
        # The contamination guard -- the part that decides whether the stub is
        # honest: a stub that can influence a verdict is worse than no control.
        for label, fn_obj in (("PRE", pre_fn), ("PATCHED", post_fn)):
            try:
                offender = _stub_reachable(inspect.getsource(fn_obj), stubbed)
            except (OSError, TypeError, SyntaxError) as exc:
                msg = (
                    f"the contamination guard could not verify {FN} in the "
                    f"{label}-image ({type(exc).__name__}: {exc}); with stub(s) "
                    f"for {', '.join(stubbed)} in play, an unverifiable verdict "
                    "is not a verdict"
                )
                notes.append("UNMEASURED 95: " + msg)
                return 0, total, notes, msg
            if offender is not None:
                msg = (
                    f"the stub could contaminate the result: {FN} in the "
                    f"{label}-image references {offender!r}, a name the stub(s) "
                    f"for {', '.join(stubbed)} would resolve; refusing to run "
                    "controls a stub can reach"
                )
                notes.append("UNMEASURED 95: " + msg)
                return 0, total, notes, msg
        # C8 calls build_parser and resolve_contract; if EITHER genuinely
        # touches a stubbed module, C8 alone is untrustworthy -- record it and
        # drop the denominator; the other seven controls still run and count.
        offenders: list[str] = []
        for helper in ("build_parser", "resolve_contract"):
            helper_obj = getattr(new_mod, helper, None)
            if not callable(helper_obj):
                continue
            try:
                offender = _stub_reachable(inspect.getsource(helper_obj), stubbed)
            except (OSError, TypeError, SyntaxError) as exc:
                offenders.append(
                    f"{helper} (unverifiable: {type(exc).__name__}: {exc})"
                )
                continue
            if offender is not None:
                offenders.append(f"{helper} references {offender!r}")
        if offenders:
            c8_trusted = False
            total = N_CONTROLS - 1
            c8_detail = "; ".join(offenders)
    ok = 0

    # C1 MUST_FIRE -- the #192 vector itself: restore delta clearly above the
    # restore tolerance with a huge rank knob. Under the old single knob at 1000.0
    # this was a pass; that pass is the regression this drill exists to pin.
    try:
        r1 = post_fn([1.0] * 8, [1.5] * 8, 1.0, 0.0005, 8, rank_agreement_tolerance=1000.0)
        old1 = pre_fn([1.0] * 8, [1.5] * 8, 1.0, 1000.0, 8)
        good = (
            r1["status"] == "red"
            and r1["restore_delta"] is not None
            and abs(r1["restore_delta"] - 0.5) < 1e-12
            and "restore" in r1["reason"]
            and old1["status"] == "pass"
        )
        note = (
            "C1 MUST_FIRE/RESTORE_RED_SURVIVES_WIDE_RANK_KNOB: drove pre [1.0]*8, "
            "post [1.5]*8 (restore delta 0.5), restore tolerance 0.0005, "
            "rank_agreement_tolerance=1000.0 -> observed status="
            f"{r1['status']} restore_delta={r1['restore_delta']}; the identical "
            "vector under the PRE-image single knob at 1000.0 read status="
            f"{old1['status']} -- that pass is the #192 regression "
            + ("PASS " + r1["reason"] if good else "FAIL")
        )
    except Exception as exc:
        good = False
        note = f"C1 ERROR: {type(exc).__name__}: {exc}"
    ok += int(good)
    notes.append(note)

    # C2 MUST_FIRE -- the real 37336 numbers: both spreads 0.2940967082977295,
    # per-rank restore deltas 0.0, restore tolerance 0.0005, no rank tolerance.
    try:
        d2 = 0.2940967082977295
        vec2 = [2.0 if i % 2 == 0 else 2.0 + d2 for i in range(8)]
        r2 = post_fn(list(vec2), list(vec2), vec2[0], 0.0005, 8)
        good = (
            r2["status"] == "unmeasured_cross_rank"
            and r2["rank_invariant"] is False
            and r2["cross_rank_spread_delta"] == 0.0
            and r2["rank_agreement_tolerance_source"] == "self-calibrated"
            and r2["restore_delta"] == 0.0
            and r2["cross_rank_spread_before_save"] is not None
            and abs(r2["cross_rank_spread_before_save"] - d2) < 1e-12
            and r2["cross_rank_spread_after_resume"] is not None
            and abs(r2["cross_rank_spread_after_resume"] - d2) < 1e-12
        )
        note = (
            "C2 MUST_FIRE/GEMMA_VECTOR_ABSTAINS_AND_STATES_ITS_DELTA: drove the "
            "37336 vector (8 ranks, spread before == spread after == "
            f"{r2['cross_rank_spread_before_save']}, restore delta 0.0, restore "
            "tolerance 0.0005, no rank tolerance) -> observed status="
            f"{r2['status']} rank_invariant={r2['rank_invariant']} "
            f"cross_rank_spread_delta={r2['cross_rank_spread_delta']} source="
            f"{r2['rank_agreement_tolerance_source']} "
            + ("PASS " + r2["reason"] if good else "FAIL")
        )
    except Exception as exc:
        good = False
        note = f"C2 ERROR: {type(exc).__name__}: {exc}"
    ok += int(good)
    notes.append(note)

    # C3 MUST_PASS -- the same vector with an explicit rank tolerance of 1.0.
    try:
        r3 = post_fn(list(vec2), list(vec2), vec2[0], 0.0005, 8, rank_agreement_tolerance=1.0)
        good = (
            r3["status"] == "pass"
            and r3["rank_invariant"] is True
            and r3["rank_agreement_absolute"] is True
            and r3["rank_agreement_tolerance_source"] == "explicit"
            and r3["rank_agreement_tolerance"] == 1.0
        )
        note = (
            "C3 MUST_PASS/EXPLICIT_RANK_TOLERANCE_CAN_PASS: drove the 37336 "
            "vector with rank_agreement_tolerance=1.0, restore tolerance still "
            f"0.0005 -> observed status={r3['status']} rank_invariant="
            f"{r3['rank_invariant']} rank_agreement_absolute="
            f"{r3['rank_agreement_absolute']} source="
            f"{r3['rank_agreement_tolerance_source']} "
            + ("PASS " + r3["reason"] if good else "FAIL")
        )
    except Exception as exc:
        good = False
        note = f"C3 ERROR: {type(exc).__name__}: {exc}"
    ok += int(good)
    notes.append(note)

    # C4 MUST_FIRE -- a LARGE pre-existing spread (7.0 before and after), restore
    # delta 0.0, no explicit rank tolerance: preserved, yet never a pass.
    try:
        vec4 = [1.0 if i % 2 == 0 else 8.0 for i in range(8)]
        r4 = post_fn(list(vec4), list(vec4), 1.0, 0.0005, 8)
        good = (
            r4["status"] != "pass"
            and r4["rank_invariant"] is False
            and r4["rank_agreement_preserved"] is True
            and r4["status"] == "unmeasured_cross_rank"
        )
        note = (
            "C4 MUST_FIRE/SELF_CALIBRATION_NEVER_MINTS_A_PASS: drove spread 7.0 "
            "before and after, restore delta 0.0, no rank tolerance -> observed "
            f"status={r4['status']} rank_invariant={r4['rank_invariant']} "
            f"rank_agreement_preserved={r4['rank_agreement_preserved']} "
            f"effective_floor={r4['rank_agreement_tolerance']} (a floor derived "
            "from the same run it judges cannot certify that run's absolute "
            "agreement) " + ("PASS" if good else "FAIL (self-calibration minted a pass)")
        )
    except Exception as exc:
        good = False
        note = f"C4 ERROR: {type(exc).__name__}: {exc}"
    ok += int(good)
    notes.append(note)

    # C5 MUST_FIRE -- resume WORSENS agreement: before spread 0.001, after spread
    # 5.0. A restore delta of exactly 0.0 with unequal before/after spreads is
    # geometrically impossible (post == pre elementwise forces equal spreads), so
    # this drill drives the minimal restore delta consistent with the two named
    # spreads -- 2.4995 -- under a restore tolerance (3.0) that still holds.
    try:
        pre5 = [1.0, 1.001]
        post5 = [-1.4995, 3.5005]
        r5 = post_fn(list(pre5), list(post5), 1.0, 3.0, 2)
        sb5 = format(r5["cross_rank_spread_before_save"], ".8g")
        sa5 = format(r5["cross_rank_spread_after_resume"], ".8g")
        good = (
            r5["status"] != "pass"
            and r5["status"] == "unmeasured_cross_rank"
            and r5["rank_agreement_preserved"] is False
            and r5["restore_delta"] is not None
            and r5["restore_delta"] <= 3.0
            and abs(r5["cross_rank_spread_before_save"] - 0.001) < 1e-12
            and abs(r5["cross_rank_spread_after_resume"] - 5.0) < 1e-12
            and sb5 in r5["reason"]
            and sa5 in r5["reason"]
        )
        note = (
            "C5 MUST_FIRE/RESUME_WORSENS_AGREEMENT_IS_NAMED: drove before spread "
            f"{sb5}, after spread {sa5} (restore delta {r5['restore_delta']}, the "
            "minimum consistent with the two spreads), restore tolerance 3.0, no "
            f"rank tolerance -> observed status={r5['status']} "
            f"rank_agreement_preserved={r5['rank_agreement_preserved']} "
            f"reason names both spreads={sb5 in r5['reason'] and sa5 in r5['reason']} "
            + ("PASS " + r5["reason"] if good else "FAIL")
        )
    except Exception as exc:
        good = False
        note = f"C5 ERROR: {type(exc).__name__}: {exc}"
    ok += int(good)
    notes.append(note)

    # C6 MUST_PASS -- bit-exact: all ranks identical before and after, explicit
    # rank tolerance supplied.
    try:
        v6 = 0.5986318588256836
        r6 = post_fn([v6] * 8, [v6] * 8, v6, 0.0005, 8, rank_agreement_tolerance=0.001)
        good = (
            r6["status"] == "pass"
            and r6["rank_invariant"] is True
            and r6["restore_delta"] == 0.0
            and r6["cross_rank_spread_delta"] == 0.0
            and r6["rank_agreement_absolute"] is True
        )
        note = (
            "C6 MUST_PASS/BIT_EXACT_STILL_PASSES: drove all ranks identical "
            f"({v6}) before and after, restore tolerance 0.0005, "
            "rank_agreement_tolerance=0.001 -> observed status="
            f"{r6['status']} rank_invariant={r6['rank_invariant']} "
            f"restore_delta={r6['restore_delta']} "
            + ("PASS " + r6["reason"] if good else "FAIL")
        )
    except Exception as exc:
        good = False
        note = f"C6 ERROR: {type(exc).__name__}: {exc}"
    ok += int(good)
    notes.append(note)

    # C7 MUST_PASS -- the legacy path is byte-for-byte: drive it twice (under and
    # over the restore tolerance) on BOTH imported modules and compare the result
    # dicts field by field on the keys the pre-image has.
    try:
        drives = (
            ("under", [0.5] * 8, "unmeasured_cross_rank"),
            ("over", [0.5] * 7 + [0.6], "red"),
        )
        new_keys = (
            "cross_rank_spread_delta",
            "rank_agreement_tolerance",
            "rank_agreement_tolerance_source",
            "rank_agreement_absolute",
            "rank_agreement_preserved",
        )
        good = True
        details = []
        for label, post_vec, expected_status in drives:
            ra = pre_fn(None, list(post_vec), 0.5, 0.0005, 8)
            rb = post_fn(None, list(post_vec), 0.5, 0.0005, 8)
            shared = all(ra[k] == rb[k] for k in ra.keys())
            new_none = all(rb[k] is None for k in new_keys)
            good = (
                good
                and shared
                and new_none
                and ra["status"] == expected_status
                and rb["status"] == expected_status
            )
            details.append(
                f"{label}: status={rb['status']} (pre-image {ra['status']}) "
                f"restore_delta={rb['restore_delta']} restore_term_legacy="
                f"{rb['restore_term_legacy']} rank_invariant={rb['rank_invariant']} "
                f"shared_keys_identical={shared} new_fields_none={new_none}"
            )
        note = (
            "C7 MUST_PASS/LEGACY_PATH_UNCHANGED: drove pre_per_rank=None twice, "
            "once under and once over the restore tolerance, on BOTH imported "
            "modules and compared the result dicts field by field on the keys the "
            "pre-image has -- " + "; ".join(details) + " "
            + ("PASS" if good else "FAIL (the legacy path drifted)")
        )
    except Exception as exc:
        good = False
        note = f"C7 ERROR: {type(exc).__name__}: {exc}"
    ok += int(good)
    notes.append(note)

    # C8 MUST_PASS -- the default argv still parses: no --rank-agreement-tolerance,
    # and the config's restore tolerance equals what --resume-tolerance carried.
    # When the contamination guard found a stub reachable from build_parser or
    # resolve_contract, C8 alone reports UNMEASURED (stub reachable) and leaves
    # the denominator; the other seven controls still run and still count.
    if not c8_trusted:
        note = (
            "C8 UNMEASURED (stub reachable): the contamination guard found "
            + c8_detail
            + f" -- with stub(s) for {', '.join(stubbed)} needed to import the "
            "patched image, a verdict from these helpers cannot be trusted, so "
            "C8 says so by name rather than by silence and leaves the pass "
            f"denominator at {total} of {N_CONTROLS}; the other seven controls "
            "still run and still count"
        )
        notes.append(note)
    else:
        try:
            argv8 = new_mod._complete_required_argv() + [
                "--iteration-budget",
                "8",
                "--early-save-steps",
                "2",
                "--output-dir",
                "/tmp/fs192-control-output",
            ]
            parser8 = new_mod.build_parser()
            args8 = parser8.parse_args(argv8)
            cfg8 = new_mod.resolve_contract(argv8, {})
            good = (
                args8.rank_agreement_tolerance is None
                and cfg8 != "selftest"
                and float(cfg8.resume_tolerance.value) == 0.0005
                and cfg8.rank_agreement_tolerance.value is None
                and cfg8.rank_agreement_tolerance.source == "absent"
            )
            note = (
                "C8 MUST_PASS/DEFAULT_ARGV_STILL_PARSES: built the parser from the "
                "patched module and parsed an argv WITHOUT --rank-agreement-tolerance "
                "-> observed flag default "
                f"{args8.rank_agreement_tolerance!r}, config resume_tolerance="
                f"{cfg8.resume_tolerance.value!r} (source "
                f"{cfg8.resume_tolerance.source}), config rank_agreement_tolerance="
                f"{cfg8.rank_agreement_tolerance.value!r} (source "
                f"{cfg8.rank_agreement_tolerance.source}) "
                + ("PASS" if good else "FAIL")
            )
        except Exception as exc:
            good = False
            note = f"C8 ERROR: {type(exc).__name__}: {exc}"
        ok += int(good)
        notes.append(note)

    return ok, total, notes, None


def main() -> int:
    # The build driver invokes every stage as `python3 <stage>` with NO arguments, so
    # bare invocation must APPLY. Requiring a flag here would make the stage a no-op
    # inside the build while passing by hand -- the #86 orphan shape, one layer up.
    argv = sys.argv[1:]
    if argv not in ([], ["--apply"], ["--check"]):
        _stderr("usage: patch_resume_tolerance_split.py [--apply|--check]   (no argument == --apply)")
        return 96
    apply = argv != ["--check"]
    if not TARGET.exists():
        _stderr(f"UNMEASURED 95: target missing: {TARGET}")
        return 95
    try:
        text = TARGET.read_text("utf-8")
    except OSError as exc:
        _stderr(f"UNMEASURED 95: target unreadable: {exc}")
        return 95

    if MARK in text:
        # Second run: byte-exact no-op, but re-prove the post-image gates so an
        # already-applied yet mangled file is RED (5), not a silent pass.
        pg = _post_image_gates(text)
        good = all(g[1] for g in pg)
        print("verdict: already applied; byte-idempotent no-op")
        for name, ok, detail in pg:
            print(f"{name}: {'PASS' if ok else 'FAIL'}  {detail}")
        return 0 if good else 5

    new, locs, _already = _transform(text)

    gres: list[tuple[str, bool, str]] = []
    gres.append((
        "G1",
        _counts_valid(locs),
        "anchor uniqueness in the pre-image: dataclass field count="
        f"{len(locs.get('field', []))} need=1, argparse flag count="
        f"{len(locs.get('arg', []))} need=1, _sourced resolution count="
        f"{len(locs.get('src', []))} need=1, validation raise count="
        f"{len(locs.get('val', []))} need=1, constructor kwarg count="
        f"{len(locs.get('ctor', []))} need=1, pure-function def count="
        f"{len(locs.get('fn', []))} need=1, resume_and_prove def count="
        f"{len(locs.get('defn', []))} need=1, verdict call site count="
        f"{len(locs.get('call', []))} need=1, '\"tolerance\": tolerance,' count="
        f"{len(locs.get('tol', []))} need=2, invariance display count="
        f"{len(locs.get('disp', []))} need=1, summary display count="
        f"{len(locs.get('summ', []))} need=1",
    ))
    spans = "-/-"
    if _locs_valid(locs):
        spans = (
            f"pure function {locs['defn'][0] - locs['fn'][0]} lines, call site "
            f"{locs['j_call'] - locs['call'][0] + 1} lines, display block "
            f"{locs['d1'] - locs['d0'] + 1} lines"
        )
    gres.append((
        "G2",
        _locs_valid(locs),
        f"region boundaries recognised exactly ({spans}); anchors ordered "
        "field<arg<src<val<ctor<fn<def<call<tolerance[0]<tolerance[1]<display<"
        "summary -- the stage does not rewrite a file it does not recognise",
    ))
    premise = text.count("rank_agreement") == 0 and text.count(N_OLDCON) == 1
    gres.append((
        "G3",
        premise,
        f"MUST_FIRE premise, the pre-image genuinely exhibits #192: occurrences of "
        f"'rank_agreement' count={text.count('rank_agreement')} need=0 (no second "
        "knob exists), the conflated rank_invariant expression spending the ONE "
        f"tolerance on both spreads count={text.count(N_OLDCON)} need=1 -- one "
        "scalar, two questions, exactly as measured on jobs 37336/37319",
    ))
    gres.append((
        "G4",
        text.count(N_FEX) == 1 and new.count(N_FEX) == 1,
        f"fixed_examples denominator '<n> of <n>' preserved pre="
        f"{text.count(N_FEX)} post={new.count(N_FEX)} need=1/1 (a single-row proof "
        "cannot be read as broad coverage)",
    ))
    gres.extend(_post_image_gates(new))

    for name, ok, detail in gres:
        print(f"{name}: {'PASS' if ok else 'FAIL'}  {detail}")
    gates = sum(1 for _n, ok, _d in gres if ok)
    if gates != len(gres):
        _stderr(f"\nREFUSE 96: static gates {gates}/{len(gres)}; writing nothing "
                "(controls not run against an image the gates rejected)")
        return 96

    try:
        cok, ctot, cnotes, unmeasured = _controls(text, new)
    finally:
        # The control images outlive _import_image on purpose (the contamination
        # guard reads their real source); this is the one owner that reaps them.
        _reap_control_images()
    for n in cnotes:
        print("control " + n)
    if unmeasured is not None:
        _stderr(f"UNMEASURED 95: {unmeasured}; writing nothing")
        return 95
    if cok != ctot:
        _stderr(f"\nREFUSE 96: static gates {gates}/{len(gres)}, controls "
                f"{cok}/{ctot}; writing nothing")
        return 96
    again, _, already2 = _transform(new)
    if again != new or not already2:
        _stderr("REFUSE 96: byte-idempotence failed on own output; writing nothing")
        return 96
    if not apply:
        print(f"verdict: READY  6 insertion(s) and 5 replacement(s) would be "
              f"applied, {gates}/{len(gres)} static gates, {cok}/{ctot} controls")
        return 0
    TARGET.write_text(new, "utf-8")
    print(f"{MARK} {gates}/{len(gres)} static gates, {cok}/{ctot} controls")
    return 0


def _guarded() -> int:
    # This stage exists because one knob collapsed two questions; it must not
    # collapse its own exit states: an unhandled exception is a REFUSE with a named
    # message, never a bare rc=1.
    try:
        return main()
    except Exception as exc:  # noqa: BLE001 - deliberate: no traceback may escape as rc=1
        _stderr(f"REFUSE 96: {MARK} stage raised {type(exc).__name__}: {exc}")
        return 96


if __name__ == "__main__":
    raise SystemExit(_guarded())
