#!/usr/bin/env python3
"""#177/#178: stop the resume proof from blaming the wrong subsystem.

WHAT / WHY. resume_and_prove shipped ONE statistic to answer TWO questions: every rank
compared its own post-restore fixed-eval loss against RANK 0's recorded pre-save scalar,
and the per-rank differences were MAXed:

    fixed_pre = manifest.get("fixed_loss_before_save", None)   # rank 0's scalar
    local_difference = abs(fixed_post - float(fixed_pre))      # crosses ranks
    dist.all_reduce(difference_tensor, op=dist.ReduceOp.MAX)

Restore fidelity and cross-rank agreement are independent quantities; folded into one
number, a rank-disagreement was REPORTED as a resume failure. Measured on 8xH100,
Gemma-3-1b-it, job 37319 (tolerance raised to 10.0 so the refusal became a measurement):
before_save 0.5986318588256836 was bit-IDENTICAL to after_resume on rank 0, while
maximum_rank_difference read 0.17570888996124268 -- that number cannot be a restore
error. A dedicated zero-training 8-rank forensic run measured within-rank restore delta
0.0 and 0 of 341 parameter-fingerprint keys changed across save/load -- the restore is
exact -- while the fixed-eval loss takes exactly two bit-identical values across ranks
(spread 1.1025075912475586) on a fresh, NEVER-SAVED runtime, before any checkpoint
exists. Inputs verified byte-identical on 8 of 8 ranks. The ranks disagreeing is a
property of the instrument, present before any save; the shipped proof named it resume.

THE FIX. (1) Extract the entire decision rule into a PURE function,
_resume_continuity_verdict(pre_per_rank, post_per_rank, pre_scalar, tolerance,
world_size): plain lists and scalars in, a plain dict out, no torch, no distributed
state -- a rule certifiable on the build host rather than only by an 8-GPU allocation.
(2) In resume_and_prove, read each rank's OWN pre-save value off the rank payload it
ALREADY loaded -- _run computes pre_loss on every rank independently (there is no
all-reduce inside _evaluate_fixed_loss) and save_checkpoint wrote it into that rank's
own payload file, so the per-rank evidence is ALREADY ON DISK in every checkpoint this
framework has ever written; no gather at save time, no format change. Assert the
tolerance against per-rank restore fidelity ONLY (|own_post - own_pre|, MAX over
ranks), still raising OperationFailure("resume", "fixed_loss_continuity", ...) on a
genuine lossy restore -- RED outranks cross-rank divergence and is never laundered
into it. (3) Report the before-save spread (the measured noise floor of the
instrument), the after-resume spread, and both full per-rank vectors as a named
measurement, fixed_eval_rank_invariance, which the run ledger registers UNMEASURED
(fail-closed) exactly when the ranks diverge -- never a pass, never a resume failure.
Malformed evidence (vector length != world_size, non-finite entries) is refused; a
payload without the recorded key (foreign/future writer) falls the restore term back
to the legacy rank-0-scalar comparison, explicitly marked as the conflated statistic,
and declares the cross-rank term UNMEASURED. The tolerance is never widened -- that
would hide the anomaly and make the proof model-specific (Qwen3-4B reads 0.0 only
because its ranks happen to agree).

THE GATE. This stage locates the shipped arithmetic by unique anchors, requires a
MUST_FIRE premise (the single-scalar manifest read AND the MAX over abs(...) are both
present), replaces exactly two regions, inserts the pure function at module scope, and
then proves: the defect arithmetic is gone, the pure function references no torch/dist
name (an AST census, not a grep), the per-rank payload read is present, the
fixed_examples denominator is preserved, the post-image parses and compiles, and the
transform is byte-idempotent. Eight controls drive the pure function with plain python
-- including the REAL measured rank vector as a MUST_FIRE that is RED under the legacy
arithmetic and split apart by the new one, plus boundary, precedence, legacy-fallback,
and refuse cases. Controls need no torch and no cluster; the target is parsed, never
imported.
"""

from __future__ import annotations

import ast
import math
import pathlib
import sys

TARGET = pathlib.Path(__file__).resolve().parent / "h100" / "gen" / "fs_train.fixed.py"
MARK = "fs178:"
N_CONTROLS = 8
FN = "_resume" + "_continuity_verdict"

# Needles are ASSEMBLED, never written as one literal: this stage counts them in the
# target, and a source that contains its own needle is inside its own denominator.
N_DEF = "def resume" + "_and_prove("
N_CMP = "local" + "_difference = abs("
N_CMPMSG = "fixed loss changed" + " by"
N_FL = '"fixed' + '_loss": {'
N_CV = '"continuity' + '_verdict"'
N_LEGACY = "manifest.get(" + '"fixed_' + 'loss_before_save"'
N_ABS = "abs(fixed_post - " + "float(fixed_pre))"
N_MAX = "dist.all" + "_reduce(difference_tensor, op=dist.ReduceOp." + "MAX)"
N_FEX = '"fixed_examples": Deno' + "minatedCount("
N_PGET = "payload.get(" + '"fixed_' + 'loss_before_save"'
N_INV = '"fixed_eval_rank' + '_invariance"'

# The pure decision rule, inserted at module scope immediately before
# resume_and_prove. It must stay free of torch/dist names -- G6 proves that with an
# AST census -- so the rule is certifiable on the build host, not only by burning an
# 8-GPU allocation.
PURE_BLOCK = '''def _resume_continuity_verdict(pre_per_rank, post_per_rank, pre_scalar, tolerance, world_size):
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

    Statuses: "refuse" (malformed evidence -- a length that disagrees with
    world_size or a non-finite entry: refuse rather than guess); "red" (restore
    fidelity broken -- delta over tolerance, and RED outranks cross-rank divergence
    so a real restore defect can never be laundered into "the ranks merely
    disagree"); "pass" (restore holds AND the ranks agree); "unmeasured_cross_rank"
    (restore holds but the ranks diverge, or the per-rank pre-save vector is absent
    so the restore term fell back to the legacy rank-0-scalar comparison -- a
    DECLARED UNMEASURED, never a clean pass).
    """
    result = {
        "restore_delta": None,
        "cross_rank_spread_before_save": None,
        "cross_rank_spread_after_resume": None,
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
    if (
        isinstance(pre_scalar, bool)
        or not isinstance(pre_scalar, (int, float))
        or not math.isfinite(float(pre_scalar))
    ):
        result["reason"] = "the rank-0 manifest scalar " + repr(pre_scalar) + " is not finite"
        return result
    tolerance = float(tolerance)

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
    # The boundary is strict: a spread exactly equal to the tolerance does NOT fire
    # (">", never ">="). The before-save spread is the measured noise floor of the
    # instrument -- asserting a tolerance without ever measuring what the instrument
    # can resolve is exactly how a bit-exact restore got named a resume failure.
    rank_invariant = (
        result["cross_rank_spread_before_save"] <= tolerance
        and result["cross_rank_spread_after_resume"] <= tolerance
    )
    result["rank_invariant"] = rank_invariant
    if restore_delta > tolerance:
        # RED and final: restore fidelity outranks cross-rank divergence. A real
        # restore defect must never be laundered into "the ranks merely disagree".
        result["status"] = "red"
        result["reason"] = (
            "restore fidelity broken: max over " + str(world_size) + " rank(s) of "
            "|own_after_resume - own_before_save| = " + format(restore_delta, ".8g")
            + " exceeds tolerance " + format(tolerance, ".8g")
        )
    elif not rank_invariant:
        result["status"] = "unmeasured_cross_rank"
        result["reason"] = (
            "restore fidelity holds (restore_delta " + format(restore_delta, ".8g")
            + " <= " + format(tolerance, ".8g") + ") but the ranks do not agree: "
            "spread before save " + format(result["cross_rank_spread_before_save"], ".8g")
            + ", spread after resume " + format(result["cross_rank_spread_after_resume"], ".8g")
            + "; declared UNMEASURED, never folded into the resume verdict"
        )
    else:
        result["status"] = "pass"
        result["reason"] = (
            "restore fidelity holds (restore_delta " + format(restore_delta, ".8g")
            + ") and the ranks agree within tolerance " + format(tolerance, ".8g")
        )
    return result


'''

# Replacement for the shipped comparison block (the ONE-statistic arithmetic).
CMP_BLOCK = '''    # --- fs178: attribute the proof's two questions to two statistics (job 37319) ---
    # Job 37319 measured before_save bit-IDENTICAL to rank 0's after_resume
    # (0.5986318588256836) while the shipped statistic -- MAX over ranks of
    # |own_post - rank0's recorded pre scalar| -- read 0.17570888996124268 and
    # named resume. A zero-step 8-rank forensic run measured within-rank restore
    # delta 0.0 and 0 of 341 parameter-fingerprint keys changed across save/load,
    # and the fixed-eval loss takes exactly two bit-identical values across ranks
    # (spread 1.1025075912475586) on a fresh, NEVER-SAVED runtime: the divergence
    # precedes any checkpoint. The restore is exact; the instrument disagrees with
    # itself. Two questions, two statistics, and the instrument's noise floor is
    # measured (the before-save spread), not assumed -- the tolerance is NEVER
    # widened to hide this; that would make the proof model-specific.
    tolerance = float(config.resume_tolerance.value)
    # The per-rank pre-save value is ALREADY ON DISK: _run computed pre_loss on
    # every rank independently (there is no all-reduce inside _evaluate_fixed_loss,
    # so pre_loss is that rank's OWN value) and save_checkpoint wrote it into THIS
    # rank's own payload. No gather at save time, no format change -- every
    # checkpoint this framework has ever written carries it.
    own_pre_raw = payload.get("fixed_loss_before_save", None)
    own_pre_known = (
        isinstance(own_pre_raw, (int, float))
        and not isinstance(own_pre_raw, bool)
        and math.isfinite(float(own_pre_raw))
    )
    own_pre = float(own_pre_raw) if own_pre_known else None
    gather_packet = torch.tensor(
        [fixed_post, own_pre if own_pre_known else 0.0, 1.0 if own_pre_known else 0.0],
        dtype=torch.float64,
        device=bundle.device,
    )
    gathered = [torch.zeros_like(gather_packet) for _ in range(bundle.world_size)]
    dist.all_gather(gathered, gather_packet)
    post_per_rank = [float(slot[0].item()) for slot in gathered]
    pre_flags = [float(slot[2].item()) for slot in gathered]
    if all(flag == 1.0 for flag in pre_flags):
        pre_per_rank = [float(slot[1].item()) for slot in gathered]
    else:
        # A payload without the recorded key (a foreign or future writer): the
        # cross-rank term is declared UNMEASURED and the restore term falls back to
        # the legacy rank-0-scalar comparison, explicitly marked. Never silently
        # assume the ranks agreed.
        pre_per_rank = None
    continuity = _resume_continuity_verdict(
        pre_per_rank, post_per_rank, float(fixed_pre), tolerance, bundle.world_size
    )
    if continuity["status"] == "refuse":
        # Malformed evidence is refused, never smoothed over.
        raise OperationFailure("resume", "refuse", continuity["reason"])
    if continuity["status"] == "red":
        # A genuine lossy restore stays RED under the exact legacy phase/metric pair
        # so existing consumers keep working; restore fidelity outranks cross-rank
        # divergence and can never be laundered into "the ranks merely disagree".
        raise OperationFailure("resume", "fixed_loss_continuity", continuity["reason"])
    # --- end fs178 segment: comparison ---
'''

# Replacement for the fixed_loss / continuity_verdict metrics: both full per-rank
# vectors and both spreads, with the instrument term declared separately so the run
# ledger registers it UNMEASURED (fail-closed) exactly when the ranks diverge.
METRICS_BLOCK = '''        "fixed_loss": {
            "before_save": float(fixed_pre),
            "after_resume": fixed_post,
            "own_before_save": own_pre,
            "restore_delta": continuity["restore_delta"],
            "restore_term": (
                "legacy_rank0_scalar_conflated"
                if continuity["restore_term_legacy"]
                else "per_rank_own_payload"
            ),
            "cross_rank_spread_before_save": continuity["cross_rank_spread_before_save"],
            "cross_rank_spread_after_resume": continuity["cross_rank_spread_after_resume"],
            "pre_save_per_rank": continuity["pre_per_rank"],
            "after_resume_per_rank": continuity["post_per_rank"],
            "tolerance": tolerance,
            "status": "PROVED",
            "why_tolerance": (
                "restore fidelity is asserted per rank against that rank's OWN "
                "pre-save value (|own_after - own_pre|, MAX over ranks), never "
                "against another rank's scalar; cross-rank agreement is a separate "
                "named measurement below; finite-precision kernels and operation "
                "scheduling can alter the last bits, so continuity is bounded, not "
                "bit-for-bit -- and the tolerance is never widened to hide the "
                "instrument's measured spread"
            ),
        },
        # fs178: when the ranks do not agree, this entry lands in the run's
        # unmeasured set under this exact name -- a DECLARED UNMEASURED that names
        # the instrument (the divergence precedes any checkpoint), never a pass and
        # never a resume failure.
        "fixed_eval_rank_invariance": {
            "status": "measured" if continuity["rank_invariant"] else "unmeasured",
            "pre_save_per_rank": continuity["pre_per_rank"],
            "after_resume_per_rank": continuity["post_per_rank"],
            "cross_rank_spread_before_save": continuity["cross_rank_spread_before_save"],
            "cross_rank_spread_after_resume": continuity["cross_rank_spread_after_resume"],
            "tolerance": tolerance,
            "display": (
                f"fixed-eval rank agreement MEASURED within {tolerance:.8g}"
                if continuity["rank_invariant"]
                else "fixed-eval rank invariance UNMEASURED: the fixed-eval loss "
                "takes distinct bit-identical values across ranks (spread before "
                f"save {continuity['cross_rank_spread_before_save']}, spread after "
                f"resume {continuity['cross_rank_spread_after_resume']}) outside "
                f"tolerance {tolerance:.8g}; the restore verdict stands on its own "
                "per-rank terms"
            ),
        },
        "continuity_verdict": {
            "status": "PROVED",
            "restore_delta": continuity["restore_delta"],
            "reason": continuity["reason"],
        },
'''


def _stderr(msg: str) -> None:
    print(msg, file=sys.stderr)


def _locate(lines: list[str]) -> dict:
    """Locate the shipped arithmetic by assembled anchors; report position lists."""

    def hits(sub: str, start: int = 0) -> list[int]:
        return [i for i in range(start, len(lines)) if sub in lines[i]]

    locs: dict = {"def": hits(N_DEF), "cmp": hits(N_CMP), "fl": hits(N_FL)}
    locs["cmpmsg"] = hits(N_CMPMSG, locs["cmp"][0] if locs["cmp"] else 0)
    locs["j1"] = None
    locs["j2"] = None
    if locs["cmpmsg"]:
        i = locs["cmpmsg"][0]
        j1 = next((k for k in range(i, len(lines)) if lines[k].strip() == "),"), None)
        locs["j1"] = j1
        if j1 is not None:
            locs["j2"] = next(
                (k for k in range(j1 + 1, len(lines)) if lines[k].strip() == ")"),
                None,
            )
    locs["cv"] = hits(N_CV, locs["fl"][0] if locs["fl"] else 0)
    return locs


def _locs_valid(locs: dict) -> bool:
    return (
        len(locs.get("def", [])) == 1
        and len(locs.get("cmp", [])) == 1
        and len(locs.get("fl", [])) == 1
        and len(locs.get("cmpmsg", [])) >= 1
        and len(locs.get("cv", [])) >= 1
        and locs.get("j1") is not None
        and locs.get("j2") is not None
        and locs["def"][0] < locs["cmp"][0] < locs["j2"] < locs["fl"][0] < locs["cv"][0]
    )


def _transform(text: str) -> tuple[str, dict, bool]:
    """Apply the three edits; byte-exact no-op when the markers are already present."""
    if MARK in text:
        return text, {}, True
    lines = text.splitlines(keepends=True)
    locs = _locate(lines)
    if not _locs_valid(locs):
        return text, locs, False
    i_def = locs["def"][0]
    i_cmp = locs["cmp"][0]
    j2 = locs["j2"]
    i_fl = locs["fl"][0]
    i_cv = locs["cv"][0]
    out: list[str] = []
    i = 0
    while i < len(lines):
        if i == i_def:
            out.append(PURE_BLOCK)
            out.append(lines[i])
            i += 1
        elif i == i_cmp:
            out.append(CMP_BLOCK)
            i = j2 + 1
        elif i == i_fl:
            out.append(METRICS_BLOCK)
            i = i_cv + 1
        else:
            out.append(lines[i])
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


def _controls(new: str) -> tuple[int, list[str]]:
    """Drive the pure function with plain python -- no torch, no cluster.

    Extraction: ast-parse the POST-IMAGE, pull the function's own source segment, and
    exec just it with math in scope. The whole trainer is never imported.
    """
    notes: list[str] = []
    ok = 0
    fn = None
    seg = None
    try:
        tree = ast.parse(new)
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name == FN:
                seg = ast.get_source_segment(new, node)
                break
    except SyntaxError:
        seg = None
    if seg is not None:
        ns: dict = {"math": math}
        try:
            exec(compile(seg, "<fs178-pure>", "exec"), ns)
            cand = ns.get(FN)
            if callable(cand):
                fn = cand
        except Exception as exc:  # extraction failure fails every control below
            notes.append(
                f"extraction: pure function would not exec ({type(exc).__name__}: {exc})"
            )
    if fn is None:
        for tag in range(1, N_CONTROLS + 1):
            notes.append(f"C{tag} FAIL: pure function could not be extracted")
        return 0, notes

    # C1 MUST_FIRE -- the regression, seen RED under the legacy arithmetic and split
    # apart by the new one. The REAL measured 8-rank vector from the forensic run,
    # post == pre elementwise (a perfect restore).
    vec8 = [2.437912, 2.437912, 3.540419, 2.437912, 2.437912, 3.540419, 3.540419, 2.437912]
    legacy1 = max(abs(p - vec8[0]) for p in vec8)  # the shipped arithmetic, inline
    r1 = fn(list(vec8), list(vec8), vec8[0], 0.0005, 8)
    good = (
        legacy1 > 0.0005
        and abs(legacy1 - 1.1025075912475586) < 1e-3
        and r1["restore_delta"] == 0.0
        and r1["rank_invariant"] is False
        and r1["status"] not in ("red", "refuse")
        and r1["cross_rank_spread_before_save"] is not None
        and abs(r1["cross_rank_spread_before_save"] - legacy1) < 1e-12
    )
    ok += int(good)
    notes.append(
        f"C1 MUST_FIRE the real measured vector: legacy arithmetic legacy_delta="
        f"{legacy1:.8g} (>= tolerance 0.0005 -> the shipped proof REFUSED and named "
        f"resume); patched rule restore_delta={r1['restore_delta']} rank_invariant="
        f"{r1['rank_invariant']} status={r1['status']} spread_before="
        f"{r1['cross_rank_spread_before_save']:.8g} " + ("PASS" if good else "FAIL")
    )

    # C2 MUST_PASS: all ranks equal, post == pre.
    r2 = fn([0.5986318588256836] * 8, [0.5986318588256836] * 8, 0.5986318588256836, 0.0005, 8)
    good = (
        r2["restore_delta"] == 0.0
        and r2["rank_invariant"] is True
        and r2["status"] == "pass"
    )
    ok += int(good)
    notes.append(
        f"C2 MUST_PASS agreement with an exact restore: restore_delta="
        f"{r2['restore_delta']} rank_invariant={r2['rank_invariant']} "
        f"status={r2['status']} " + ("PASS" if good else "FAIL")
    )

    # C3 MUST_FIRE: ranks agree pre, one rank's post off by 10x tolerance -> RED.
    tol3 = 0.0005
    r3 = fn([1.0] * 8, [1.0] * 7 + [1.0 + 10.0 * tol3], 1.0, tol3, 8)
    good = (
        r3["status"] == "red"
        and r3["restore_delta"] is not None
        and abs(r3["restore_delta"] - 10.0 * tol3) < 1e-12
        and "restore" in r3["reason"]
    )
    ok += int(good)
    notes.append(
        f"C3 MUST_FIRE one rank 10x tolerance off: restore_delta={r3['restore_delta']} "
        f"status={r3['status']} (a genuine lossy restore still refuses naming resume) "
        + ("PASS " + r3["reason"] if good else "FAIL")
    )

    # C4 MUST_FIRE -- precedence: a real restore failure AND a large cross-rank
    # spread together must still be RED, never downgraded to the unmeasured state.
    r4 = fn(list(vec8), [v + (1.0 if i == 0 else 0.0) for i, v in enumerate(vec8)], vec8[0], 0.0005, 8)
    good = r4["status"] == "red" and r4["rank_invariant"] is False
    ok += int(good)
    notes.append(
        f"C4 MUST_FIRE precedence, RED outranks divergence: restore failure AND big "
        f"spread -> status={r4['status']} rank_invariant={r4['rank_invariant']} "
        + ("PASS" if good else "FAIL (RED was laundered into cross-rank UNMEASURED)")
    )

    # C5 MUST_FIRE: pre_per_rank is None -- a payload with no recorded per-rank
    # value -> declared UNMEASURED for the cross-rank term, never PASS, and the
    # restore term explicitly marked legacy.
    r5 = fn(None, [0.5] * 8, 0.5, 0.0005, 8)
    good = (
        r5["status"] == "unmeasured_cross_rank"
        and r5["restore_term_legacy"] is True
        and r5["rank_invariant"] is False
        and r5["restore_delta"] == 0.0
        and "legacy" in r5["reason"]
        and r5["cross_rank_spread_before_save"] is None
    )
    ok += int(good)
    notes.append(
        f"C5 MUST_FIRE missing per-rank record: status={r5['status']} "
        f"restore_term_legacy={r5['restore_term_legacy']} restore_delta="
        f"{r5['restore_delta']} (legacy conflated statistic, cross-rank term "
        f"declared UNMEASURED) " + ("PASS" if good else "FAIL")
    )

    # C6 MUST_FIRE: vector length disagrees with world_size -> refuse.
    r6 = fn([1.0] * 7, [1.0] * 7, 1.0, 0.0005, 8)
    good = r6["status"] == "refuse" and "length" in r6["reason"]
    ok += int(good)
    notes.append(
        f"C6 MUST_FIRE length 7 of 8: status={r6['status']} "
        + ("PASS " + r6["reason"] if good else "FAIL malformed length not refused")
    )

    # C7 MUST_FIRE: a non-finite entry -> refuse.
    r7 = fn([1.0] * 8, [1.0] * 7 + [float("nan")], 1.0, 0.0005, 8)
    good = r7["status"] == "refuse" and "finite" in r7["reason"]
    ok += int(good)
    notes.append(
        f"C7 MUST_FIRE non-finite entry: status={r7['status']} "
        + ("PASS " + r7["reason"] if good else "FAIL non-finite entry not refused")
    )

    # C8 MUST_PASS: spread exactly equal to the tolerance does not fire -- the
    # boundary is ">", never ">=", and it is stated.
    r8 = fn([1.0, 1.25], [1.0, 1.25], 1.0, 0.25, 2)
    good = (
        r8["cross_rank_spread_before_save"] == 0.25
        and r8["rank_invariant"] is True
        and r8["status"] == "pass"
        and r8["restore_delta"] == 0.0
    )
    ok += int(good)
    notes.append(
        f"C8 MUST_PASS boundary: spread {r8['cross_rank_spread_before_save']} == "
        f"tolerance 0.25 exactly -> rank_invariant={r8['rank_invariant']} "
        f"status={r8['status']} (the test is strict '>') " + ("PASS" if good else "FAIL")
    )
    return ok, notes


def _post_image_gates(text: str) -> list[tuple[str, bool, str]]:
    """Gates provable without locators (also reused on the already-applied path)."""
    g: list[tuple[str, bool, str]] = []
    n_abs = text.count(N_ABS)
    n_ld = text.count("local" + "_difference")
    g.append((
        "P1",
        n_abs == 0 and n_ld == 0,
        f"defect arithmetic gone: abs(fixed_post-float(fixed_pre)) count={n_abs} "
        f"need=0, local_difference spelling count={n_ld} need=0",
    ))
    try:
        ok, n_nodes, tref, dref = _pure_census(text)
    except SyntaxError as exc:
        ok, n_nodes, tref, dref = False, 0, -1, -1
        g.append(("P2", False, f"pure-function census could not parse the image: {exc}"))
    else:
        g.append((
            "P2",
            ok and tref == 0 and dref == 0,
            f"pure function {FN} at module scope={ok}; AST census over {n_nodes} "
            f"node(s): torch refs={tref} need=0, dist refs={dref} need=0 (a census, "
            "not a grep -- the rule must be certifiable without torch or a cluster)",
        ))
    n_pget = text.count(N_PGET)
    n_inv = text.count(N_INV)
    g.append((
        "P3",
        n_pget >= 1 and n_inv == 1,
        f"per-rank value read off the loaded rank payload count={n_pget} need>=1 "
        "(the central structural change: the evidence was ALREADY ON DISK); declared "
        f"unmeasured entry fixed_eval_rank_invariance count={n_inv} need=1",
    ))
    try:
        ast.parse(text)
        compile(text, str(TARGET), "exec")
        g.append(("P4", True, "ast.parse + compile clean"))
    except (SyntaxError, ValueError) as exc:
        g.append(("P4", False, f"parse/compile: {type(exc).__name__}: {exc}"))
    return g


def main() -> int:
    # The build driver invokes every stage as `python3 <stage>` with NO arguments, so
    # bare invocation must APPLY. Requiring a flag here would make the stage a no-op
    # inside the build while passing by hand -- the #86 orphan shape, one layer up.
    argv = sys.argv[1:]
    if argv not in ([], ["--apply"], ["--check"]):
        _stderr("usage: patch_resume_proof_attribution.py [--apply|--check]   (no argument == --apply)")
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
    g1 = (
        len(locs.get("def", [])) == 1
        and len(locs.get("cmp", [])) == 1
        and len(locs.get("fl", [])) == 1
    )
    gres.append((
        "G1",
        g1,
        f"anchor uniqueness in the pre-image: def resume_and_prove count="
        f"{len(locs.get('def', []))} need=1, local_difference=abs anchor count="
        f"{len(locs.get('cmp', []))} need=1, fixed_loss block anchor count="
        f"{len(locs.get('fl', []))} need=1",
    ))
    # The comparison block is 16 lines, not 15. Corrected against the target after
    # reading the located region rather than by tuning the constant until the gate
    # went green: the locators resolve to the `local_difference = abs(...)` line
    # through the closing paren of the `raise OperationFailure(...)` -- start, end and
    # contents all verified to be exactly the defective arithmetic and nothing else.
    # The expectation was wrong; the region was right. Had the CONTENTS not matched,
    # the correct move would have been to refuse, not to adjust the number.
    spans = "-/-"
    if _locs_valid(locs):
        spans = f"{locs['j2'] - locs['cmp'][0] + 1} of 16 expected, {locs['cv'][0] - locs['fl'][0] + 1} of 13 expected"
    g2 = (
        _locs_valid(locs)
        and locs["j2"] - locs["cmp"][0] + 1 == 16
        and locs["cv"][0] - locs["fl"][0] + 1 == 13
    )
    gres.append((
        "G2",
        g2,
        f"region boundaries recognised exactly (comparison block {spans}); end "
        "locators ordered def<cmp<cmp_end<fixed_loss<continuity_verdict -- the stage "
        "does not rewrite a file it does not recognise",
    ))
    premise = (
        text.count(N_LEGACY) == 1
        and text.count(N_ABS) == 1
        and text.count(N_MAX) == 1
    )
    gres.append((
        "G3",
        premise,
        f"MUST_FIRE premise, the pre-image genuinely exhibits the defect: "
        f"manifest.get(rank-0 scalar) count={text.count(N_LEGACY)} need=1, "
        f"abs(fixed_post-float(fixed_pre)) count={text.count(N_ABS)} need=1, "
        f"all_reduce MAX over the difference count={text.count(N_MAX)} need=1 -- "
        "ONE statistic answering two questions, exactly as measured on job 37319",
    ))
    gres.append((
        "G4",
        text.count(N_FEX) == 1 and new.count(N_FEX) == 1,
        f"fixed_examples denominator '<n> of <n>' preserved pre="
        f"{text.count(N_FEX)} post={new.count(N_FEX)} need=1/1 (a single-row proof "
        "cannot be read as broad coverage)",
    ))
    gres.extend([("G5a" if i == 0 else "G5b", ok, detail)
                 for i, (name, ok, detail) in enumerate(_post_image_gates(new)[:2])])
    p34 = _post_image_gates(new)[2:]
    gres.append(("G5c", p34[0][1], p34[0][2]))
    gres.append(("G5d", p34[1][1], p34[1][2]))

    for name, ok, detail in gres:
        print(f"{name}: {'PASS' if ok else 'FAIL'}  {detail}")
    gates = sum(1 for _n, ok, _d in gres if ok)
    cok, cnotes = _controls(new)
    for n in cnotes:
        print("control " + n)

    if gates != len(gres) or cok != N_CONTROLS:
        _stderr(f"\nREFUSE 96: static gates {gates}/{len(gres)}, controls "
                f"{cok}/{N_CONTROLS}; writing nothing")
        return 96
    if not apply:
        print(f"verdict: READY  1 insertion and 2 replacement(s) would be applied, "
              f"{gates}/{len(gres)} static gates, {cok}/{N_CONTROLS} controls")
        # Byte-idempotence is only meaningful against a write; still prove it here.
        again, _, already2 = _transform(new)
        if again != new or not already2:
            _stderr("REFUSE 96: byte-idempotence failed on own output")
            return 96
        return 0
    again, _, already2 = _transform(new)
    if again != new or not already2:
        _stderr("REFUSE 96: byte-idempotence failed on own output; writing nothing")
        return 96
    TARGET.write_text(new, "utf-8")
    print(f"{MARK} {gates}/{len(gres)} static gates, {cok}/{N_CONTROLS} controls")
    return 0


def _guarded() -> int:
    # This stage exists because one statistic collapsed two questions; it must not
    # collapse its own exit states: an unhandled exception is a REFUSE with a named
    # message, never a bare rc=1.
    try:
        return main()
    except Exception as exc:  # noqa: BLE001 - deliberate: no traceback may escape as rc=1
        _stderr(f"REFUSE 96: {MARK} stage raised {type(exc).__name__}: {exc}")
        return 96


if __name__ == "__main__":
    raise SystemExit(_guarded())