"""T1-7: adjudicate FoundationScale's DCP<->HF parity verifier.

WHAT THIS ROW TURNS OUT TO BE, AND WHAT IT IS NOT
--------------------------------------------------
The row said "ABSENT". It is not: checkpoint/dcp.py sniffs and reads both DCP
directories and safetensors behind one WeightSource protocol, and
verify/parity.py compares two sources key by key in float64 against an explicit,
named tolerance policy. That is a shipped, working subsystem that no matrix row
had ever exercised.

But the row's wording is "DCP<->HF conversion parity", and only half of that is
real. FoundationScale does NOT convert between the formats -- there is no
DCP-to-HF writer anywhere in the source. What it does is READ both and tell you
truthfully whether they agree. That is the more useful half, because it is the
instrument you would point at somebody else's converter, but it is a different
claim and this adjudicator refuses to let the two be confused.

WHY "ok" IS NOT THE MEASUREMENT
--------------------------------
`ParityReport.ok` means "agrees within the declared tolerance policy", not
"bitwise identical". A one-ULP bf16 difference passes, and SHOULD: the default
policy declares max_abs_diff <= 1e-2 and one bf16 ULP at that magnitude is
1.2e-4. So scoring this row on `ok` alone would be scoring the tolerance, not
the comparator. What must be shown instead is:

  * the comparator READS BYTES -- the one-ULP arm must report exactly one
    mismatched element and the exact ULP as its max_abs_diff. A structural
    comparison that never touched the data would report zero.
  * the grade DISTINGUISHES bitwise-equal from within-tolerance -- EXACT for
    the untouched arm, CLOSE for the nudged one. If both read EXACT, `ok` is
    hiding a real difference from anyone who looks closer.
  * the tolerance is a CEILING THAT CAN BE EXCEEDED -- a perturbation of 1.0
    must grade DIFFER, block, and name the key. Without this arm, a policy that
    acquits everything is indistinguishable from one that works.
  * a key-set gap is a DIFFERENT, separately named failure -- dropping a tensor
    must fail via only_in_right, not via a numeric finding.
  * zero common keys must be VACUOUS, never ok. That is the all([]) shape this
    framework exists to hunt, and it would be the worst possible place to have it.

Pure dict arithmetic over the payload: no torch, no checkpoint reader, no GPU.
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import Any

GREEN, RED, UNMEASURED, REFUSE = 0, 5, 95, 96

EXACT, ONE_ULP, BIG, MISSING, DISJOINT = "exact", "one_ulp", "big", "missing", "disjoint"
REQUIRED = (EXACT, ONE_ULP, BIG, MISSING, DISJOINT)


def _status(rec: dict[str, Any]) -> str | None:
    """Normalise 'ParityStatus.CLOSE' and 'CLOSE' to 'CLOSE'; absent stays None."""
    raw = rec.get("victim_status")
    if not isinstance(raw, str):
        return None
    return raw.rsplit(".", 1)[-1].upper()


def t1_7_verdict(payload: dict[str, Any]) -> tuple[int, str]:
    """Return (exit_code, one-line reason).

    Order is load-bearing and is the documented contract of this function:

     1. a required arm is missing                        -> UNMEASURED 95
     2. an arm could not write its DCP side              -> UNMEASURED 95
     3. an arm's comparison raised                       -> RED 5
     4. the exact arm compared nothing                   -> REFUSE 96
     5. the exact arm did not agree with itself          -> RED 5
     6. the exact arm was not graded bitwise-equal       -> RED 5
     7. the oversized perturbation did not block         -> RED 5
     8. it blocked without naming the perturbed key      -> RED 5
     9. the one-ULP arm saw no mismatched element        -> RED 5
    10. the one-ULP arm was not graded within-tolerance  -> RED 5
    11. the dropped tensor was not reported by key set   -> RED 5
    12. disjoint key sets were not vacuous, or were ok   -> RED 5
    13. everything above clean                           -> GREEN 0

    The arms that establish the instrument can fire (7, 8) are settled before
    the arms that depend on it being able to (9, 10).
    """
    arms = {name: (payload.get("arms") or {}).get(name) for name in REQUIRED}

    for name in REQUIRED:
        if arms[name] is None:
            return UNMEASURED, f"arm {name!r} is absent from the payload; nothing to adjudicate"

    for name in REQUIRED:
        rec = arms[name] or {}
        if not rec.get("dcp_written"):
            return UNMEASURED, (
                f"arm {name!r} could not write its DCP side ({rec.get('error', 'no reason')}); "
                "that is torch's writer, not FoundationScale, so it is a gap not a failure"
            )
        if not rec.get("compared"):
            return RED, (
                f"arm {name!r} raised inside compare_sources ({rec.get('error', 'no reason')}): "
                "a reader that crashes on a checkpoint it accepts cannot certify anything"
            )

    ex = arms[EXACT] or {}
    if ex.get("is_vacuous") or not ex.get("compared_elements"):
        return REFUSE, (
            "the exact arm compared no elements, so every other arm is being scored against "
            "an instrument that has not been shown to read anything"
        )
    if not ex.get("ok"):
        return RED, (
            "the exact arm reports disagreement between a DCP checkpoint and the safetensors "
            f"it was written from ({ex.get('n_findings')} finding(s), "
            f"{ex.get('n_only_in_left')}/{ex.get('n_only_in_right')} one-sided keys): "
            "the comparator disagrees with a checkpoint and itself"
        )
    if _status(ex) != "EXACT":
        return RED, (
            f"the untouched arm graded {_status(ex)!r} rather than EXACT: a round trip that "
            "changed no bytes must be reported as bitwise-equal, or the grade means nothing"
        )

    big = arms[BIG] or {}
    if big.get("ok"):
        return RED, (
            "a perturbation of 1.0 on a single element was ACQUITTED: the tolerance policy is "
            "not a ceiling that can be exceeded, so no parity verdict from this comparator "
            "carries information"
        )
    if payload.get("victim", {}).get("key") not in (big.get("finding_keys") or []):
        return RED, (
            "the oversized-perturbation arm blocked, but not on the key that was perturbed "
            f"(blocked on {big.get('finding_keys')}): it failed for the wrong reason"
        )

    ulp = arms[ONE_ULP] or {}
    if ulp.get("victim_mismatched_elements") != 1:
        return RED, (
            "the one-ULP arm reports "
            f"{ulp.get('victim_mismatched_elements')!r} mismatched elements where exactly one "
            "element was changed: the comparison is not reading the tensor data"
        )
    if _status(ulp) != "CLOSE":
        graded = _status(ulp)
        if graded == "EXACT":
            why = ("a real bitwise difference was reported as bitwise-equal, which hides it "
                   "from any reader who looks past `ok`")
        elif graded is None:
            why = "no grade was recorded at all, so there is nothing to read past `ok`"
        else:
            why = ("a difference well inside the declared policy was graded as blocking, so "
                   "the policy is not being applied")
        return RED, f"the one-ULP arm graded {graded!r} rather than CLOSE: {why}"

    miss = arms[MISSING] or {}
    if miss.get("ok") or miss.get("n_only_in_right") != 1:
        return RED, (
            "dropping one tensor from the DCP side was not reported as a one-sided key "
            f"(ok={miss.get('ok')}, only_in_right={miss.get('n_only_in_right')}): a key-set gap "
            "must be its own named failure, not a numeric one and not a pass"
        )

    dis = arms[DISJOINT] or {}
    if dis.get("ok") or not dis.get("is_vacuous"):
        return RED, (
            f"comparing disjoint key sets returned ok={dis.get('ok')} "
            f"vacuous={dis.get('is_vacuous')}: a comparison over zero common keys that does not "
            "declare itself vacuous is the all([]) defect, inside the instrument built to hunt it"
        )

    return GREEN, (
        "FoundationScale reads a DCP checkpoint and a safetensors checkpoint through one "
        f"interface and adjudicates them truthfully: {ex.get('n_compared_keys')} keys and "
        f"{ex.get('compared_elements')} elements compared bitwise-EXACT on identical inputs; a "
        "single element moved by one ULP is SEEN (exactly one mismatched element, max abs diff "
        f"{ulp.get('victim_max_abs_diff')}) and graded CLOSE against the declared policy "
        f"{ulp.get('victim_tolerances')!r} rather than silently absorbed; the same element moved "
        "by 1.0 grades DIFFER, blocks, and names the key, so the tolerance is a ceiling that can "
        "be exceeded; a dropped tensor is reported as a one-sided key rather than as a numeric "
        "finding; and disjoint key sets are VACUOUS rather than agreeing. What this does NOT "
        "establish: that FoundationScale can CONVERT between the two formats -- it cannot, there "
        "is no DCP-to-HF writer in the source, and this row measures the parity verifier only"
    )


# ---------------------------------------------------------------------------
# controls
# ---------------------------------------------------------------------------
VICTIM = "model.embed_tokens.weight"


def _mk(**over: Any) -> dict[str, Any]:
    """A payload that is GREEN unless an override breaks exactly one thing.

    Keys are 'arm.field'; a value of the sentinel DROP removes the arm entirely.
    """
    arms: dict[str, Any] = {
        EXACT: {
            "dcp_written": True, "compared": True, "ok": True, "is_vacuous": False,
            "n_compared_keys": 26, "compared_elements": 19743872, "n_findings": 0,
            "finding_keys": [], "n_only_in_left": 0, "n_only_in_right": 0,
            "victim_status": "ParityStatus.EXACT", "victim_mismatched_elements": 0,
            "victim_max_abs_diff": 0.0,
        },
        ONE_ULP: {
            "dcp_written": True, "compared": True, "ok": True, "is_vacuous": False,
            "n_compared_keys": 26, "compared_elements": 19743872, "n_findings": 0,
            "finding_keys": [], "n_only_in_left": 0, "n_only_in_right": 0,
            "victim_status": "ParityStatus.CLOSE", "victim_mismatched_elements": 1,
            "victim_max_abs_diff": 0.0001220703125,
            "victim_tolerances": "default-close(max_abs_diff<=0.01, cosine>=0.999, rel_frob<=0.01)",
        },
        BIG: {
            "dcp_written": True, "compared": True, "ok": False, "is_vacuous": False,
            "n_compared_keys": 26, "compared_elements": 19743872, "n_findings": 1,
            "finding_keys": [VICTIM], "n_only_in_left": 0, "n_only_in_right": 0,
            "victim_status": "ParityStatus.DIFFER", "victim_mismatched_elements": 1,
            "victim_max_abs_diff": 0.998046875,
        },
        MISSING: {
            "dcp_written": True, "compared": True, "ok": False, "is_vacuous": False,
            "n_compared_keys": 25, "compared_elements": 296064, "n_findings": 0,
            "finding_keys": [], "n_only_in_left": 0, "n_only_in_right": 1,
        },
        DISJOINT: {
            "dcp_written": True, "compared": True, "ok": False, "is_vacuous": True,
            "n_compared_keys": 0, "compared_elements": 0, "n_findings": 0,
            "finding_keys": [], "n_only_in_left": 26, "n_only_in_right": 26,
        },
    }
    for path, value in over.items():
        arm, _, field = path.partition("__")
        if field == "DROP":
            arms.pop(arm)
        else:
            arms[arm][field] = value
    return {"arms": arms, "victim": {"key": VICTIM}}


# (name, expected, payload, why). A control without a stated reason is a
# regression test for whatever the code happened to do on the day it was written.
CONTROLS: list[tuple[str, int, dict[str, Any], str]] = [
    ("green_clean", GREEN, _mk(),
     "the only shape that may pass: every grade distinct and every failure mode named"),
    ("missing_exact_arm", UNMEASURED, _mk(exact__DROP=1),
     "without the identity arm nothing else has a baseline to be scored against"),
    ("missing_big_arm", UNMEASURED, _mk(big__DROP=1),
     "without the tolerance-exceeding arm the policy has never been shown to block"),
    ("missing_disjoint_arm", UNMEASURED, _mk(disjoint__DROP=1),
     "the all([]) guard is the one this framework most needs run, not assumed"),
    ("dcp_write_failed", UNMEASURED, _mk(exact__dcp_written=False),
     "torch's writer failing is an environment gap, never evidence about FoundationScale"),
    ("compare_raised", RED, _mk(one_ulp__compared=False),
     "a reader that crashes on a checkpoint it agreed to open cannot certify anything"),
    ("exact_vacuous", REFUSE, _mk(exact__is_vacuous=True, exact__compared_elements=0),
     "an instrument that read nothing cannot be used to score the arms that depend on it"),
    ("exact_zero_elements", REFUSE, _mk(exact__compared_elements=0),
     "zero elements compared is the all([]) shape even when is_vacuous was not set"),
    ("exact_not_ok", RED, _mk(exact__ok=False, exact__n_findings=3),
     "the comparator disagrees with a checkpoint written from the very tensors it holds"),
    ("exact_graded_close", RED, _mk(exact__victim_status="ParityStatus.CLOSE"),
     "an unchanged tensor graded merely CLOSE means EXACT does not mean bitwise-equal"),
    ("exact_graded_differ", RED, _mk(exact__victim_status="ParityStatus.DIFFER", exact__ok=False),
     "reached through the not-ok branch first, but either way identity must not DIFFER"),
    ("big_acquitted", RED, _mk(big__ok=True),
     "a tolerance that cannot be exceeded makes every other verdict content-free"),
    ("big_blocked_wrong_key", RED, _mk(big__finding_keys=["model.layers.0.mlp.up_proj.weight"]),
     "blocking for an unrelated reason is not the detector firing on what was perturbed"),
    ("big_blocked_no_key", RED, _mk(big__finding_keys=[]),
     "a block with no named key cannot be attributed to the perturbation at all"),
    ("ulp_saw_nothing", RED, _mk(one_ulp__victim_mismatched_elements=0),
     "zero mismatched elements after one element changed means the data was never read"),
    ("ulp_mismatch_absent", RED, _mk(one_ulp__victim_mismatched_elements=None),
     "absent is not zero and is certainly not one; an unrecorded count must not pass"),
    ("ulp_saw_too_many", RED, _mk(one_ulp__victim_mismatched_elements=4096),
     "a count that does not match the perturbation means the comparison is not element-wise"),
    ("ulp_graded_exact", RED, _mk(one_ulp__victim_status="ParityStatus.EXACT"),
     "the dangerous one: a real difference reported as bitwise-equal is invisible past `ok`"),
    ("ulp_graded_differ", RED, _mk(one_ulp__victim_status="ParityStatus.DIFFER",
                                   one_ulp__ok=False),
     "blocking inside the declared tolerance means the named policy is not being applied"),
    ("ulp_status_absent", RED, _mk(one_ulp__victim_status=None),
     "an unrecorded grade is not a CLOSE grade"),
    ("missing_arm_passed", RED, _mk(missing__ok=True, missing__n_only_in_right=0),
     "a dropped tensor that compares clean is a key-set gap the comparator did not notice"),
    ("missing_as_numeric", RED, _mk(missing__n_only_in_right=0, missing__n_findings=1),
     "failing for a numeric reason confuses two failure modes that must stay distinguishable"),
    ("disjoint_not_vacuous", RED, _mk(disjoint__is_vacuous=False),
     "zero common keys that does not declare itself vacuous is the defect, not a near miss"),
    ("disjoint_ok", RED, _mk(disjoint__ok=True, disjoint__is_vacuous=True),
     "vacuous and ok at once would let an empty comparison read as agreement"),
]


def _self_test() -> int:
    failures: list[str] = []
    exercised: set[int] = set()
    for name, expected, payload, why in CONTROLS:
        got, reason = t1_7_verdict(payload)
        exercised.add(got)
        if got != expected:
            failures.append(f"{name}: expected {expected}, got {got} ({reason})")
        else:
            print(f"  ok  {name:24s} -> {got:2d}   {why}")
    if exercised != {GREEN, RED, UNMEASURED, REFUSE}:
        failures.append(
            f"controls never reach {sorted({GREEN, RED, UNMEASURED, REFUSE} - exercised)}"
        )
    if failures:
        for line in failures:
            print(f"  FAIL {line}")
        print(f"FAILED: {len(failures)} of {len(CONTROLS)}")
        return RED
    print(f"CLEAR: {len(CONTROLS)} of {len(CONTROLS)}")
    return GREEN


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--payload")
    args = ap.parse_args()

    if args.self_test:
        return _self_test()
    if not args.payload:
        print("UNMEASURED: no --payload given and no --self-test requested")
        return UNMEASURED

    with open(args.payload, encoding="utf-8") as fh:
        payload = json.load(fh)
    code, reason = t1_7_verdict(payload)
    label = {GREEN: "GREEN", RED: "RED", UNMEASURED: "UNMEASURED", REFUSE: "REFUSE"}[code]
    print(f"{label}: {reason}")
    return code


if __name__ == "__main__":
    sys.exit(main())
