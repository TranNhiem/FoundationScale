"""Adjudicate T1-18: LoRA through the PACKAGE API -- adapter trains, base frozen.

The row said "ABSENT from the package plane (#329)" on the strength of reading
the source, and the announcement draft said LoRA through the package API is
"explicitly NOT IMPLEMENTED". Both are false. `TrainConfig` carries five
adapter fields, `train` is a public callable, and a hand-built config -- no CLI
anywhere -- attaches peft, trains the adapter and freezes the base.

WHY THE EXIT CODE SETTLES NOTHING ON ITS OWN
--------------------------------------------
A LoRA declaration can fail in three ways that all look similar from outside:

  REFUSED  the plane says it cannot and exits 96. The row's claim is then
           false but honestly reported.
  IGNORED  it trains, but as a full fine-tune -- the declaration is accepted
           and silently dropped. rc is 0 and the run looks perfect. This is the
           #375 defect class and is the worst of the three, because a manifest
           would record a LoRA run that never happened.
  HONOURED adapter weights move, base does not.

IGNORED and HONOURED are both rc 0. So the verdict is decided on the ARTIFACTS
and the MANIFEST, never on rc alone.

THE THREE ARMS AND WHY EACH IS LOAD-BEARING
-------------------------------------------
  pkg_full      adapter=None, lr>0. Proves the package API trains at all, and
                -- the point -- that the base-movement detector CAN FIRE. A
                "base did not move" reading taken with a detector that never
                fires anywhere is an untested emptiness, not a control.
  pkg_lora      the deciding arm.
  pkg_lora_lr0  the null arm. lora_B is zero-initialised, so nonzero norm is
                movement with no threshold to calibrate -- but only if lr=0
                leaves it EXACTLY zero. If the null arm also moves, the probe
                is reading initialisation and the deciding arm says nothing.

BASE-FROZEN IS PROVEN TWICE, BY DIFFERENT MEANS
-----------------------------------------------
  1. the saved artifact carries adapter tensors and ZERO base tensors, while
     the full-finetune control writes base tensors and moves them; and
  2. the framework's own manifest records trainable_params/total_params, which
     must be a small fraction. (1) alone would be weak -- "the artifact has no
     base weights" is a statement about the save path, not about requires_grad.
     (2) alone would be weak -- it is the framework marking its own homework.
     Together they are two independent readings of the same fact.

Pure dict arithmetic over synthetic payloads: no torch, no peft, no GPU, so a
laptop and a tray reach the same verdict. Run with --self-test.
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import Any

GREEN, RED, UNMEASURED, REFUSE = 0, 5, 95, 96

BASELINE = "pkg_full"
DECIDER = "pkg_lora"
NULL = "pkg_lora_lr0"
REQUIRED = (BASELINE, DECIDER, NULL)

# peft on a 1.5B model with rank 8 on two projections is ~0.07% trainable. The
# bar is set two orders of magnitude looser than the real reading, because the
# claim is "the base is frozen", not "the rank is 8" -- a tight bar here would
# turn a legitimate change of rank or target set into a spurious RED.
MAX_TRAINABLE_FRACTION = 0.01


def _frac(payload: dict[str, Any]) -> float | None:
    """trainable/total from the manifest, or None if it was never recorded.

    Absent is not zero. A missing record must not read as "wonderfully frozen".
    """
    man = payload.get("manifest") or {}
    try:
        trainable = float(man["/config/adapter.trainable_params/value"])
        total = float(man["/config/adapter.total_params/value"])
    except (KeyError, TypeError, ValueError):
        return None
    return None if total <= 0 else trainable / total


def t1_18_verdict(payload: dict[str, dict[str, Any]]) -> tuple[int, str]:
    """Score the three arms.

    Order is load-bearing and is asserted by the controls:

     1. a required arm is missing                      -> UNMEASURED 95
     2. an arm died before the plane ran               -> UNMEASURED 95
     3. baseline did not train / wrote no weights      -> REFUSE 96
     4. baseline base-movement detector did not fire   -> REFUSE 96
     5. the deciding arm was refused                   -> REFUSE 96
     6. the deciding arm did not train                 -> REFUSE 96
     7. null arm moved lora_B                          -> REFUSE 96
     8. deciding arm attached no adapter tensors       -> RED 5
     9. manifest did not record the declaration        -> RED 5
    10. deciding arm moved base weights                -> RED 5
    11. deciding arm's lora_B did not move             -> RED 5
    12. trainable fraction absent or not small         -> RED 5
    13. everything above clean                         -> GREEN 0
    """
    missing = [a for a in REQUIRED if a not in payload]
    if missing:
        return UNMEASURED, f"arm(s) absent from the payload: {missing}"

    for arm in REQUIRED:
        got = payload[arm]
        if got.get("config_rejected"):
            return UNMEASURED, f"{arm}: TrainConfig rejected the declaration before the plane ran"
        if got.get("train_raised"):
            return UNMEASURED, f"{arm}: train() raised; the arm never reached a verdict"

    base = payload[BASELINE]
    if not base.get("trained") or base.get("ckpt_tensors_total", 0) <= 0:
        return REFUSE, f"{BASELINE}: the package API did not train at all; nothing else is readable"
    if base.get("ckpt_base_moved", 0) <= 0:
        return REFUSE, (
            f"{BASELINE}: base weights did not move under a full fine-tune, so the "
            "base-movement detector has never been shown to fire and 'base frozen' "
            "in the lora arm would be an untested emptiness"
        )

    dec = payload[DECIDER]
    if dec.get("any_known_refusal"):
        return REFUSE, f"{DECIDER}: the plane refused the declaration; the capability is not demonstrated"
    if not dec.get("trained"):
        return REFUSE, f"{DECIDER}: did not train, and did not refuse either"

    null = payload[NULL]
    if null.get("ckpt_lora_b_nonzero", 0) > 0 or null.get("ckpt_lora_b_max_norm", 0.0) > 0.0:
        return REFUSE, (
            f"{NULL}: lora_B moved at lr=0, so a nonzero norm in {DECIDER} cannot be "
            "attributed to training -- the movement probe is not valid here"
        )

    if dec.get("ckpt_lora_b", 0) <= 0:
        return RED, (
            f"{DECIDER}: declared adapter=lora and trained to completion, but the artifact "
            "carries no lora_B tensors -- the declaration was accepted and dropped"
        )

    man = dec.get("manifest") or {}
    if man.get("/config/adapter/value") != "lora":
        return RED, (
            f"{DECIDER}: trained with an adapter but the manifest does not record "
            f"adapter=lora (got {man.get('/config/adapter/value')!r}); the run is "
            "unattributable to the declaration that produced it"
        )

    if dec.get("ckpt_base_moved", 0) > 0:
        return RED, (
            f"{DECIDER}: {dec['ckpt_base_moved']} base tensor(s) moved under a LoRA "
            "declaration; the base is not frozen"
        )
    if dec.get("ckpt_lora_b_nonzero", 0) <= 0:
        return RED, (
            f"{DECIDER}: adapter attached but every lora_B tensor is still exactly zero; "
            "the adapter did not train"
        )

    frac = _frac(dec)
    if frac is None:
        return RED, (
            f"{DECIDER}: the manifest records no trainable/total parameter counts, so "
            "'base frozen' rests only on the save path having omitted base weights"
        )
    if frac > MAX_TRAINABLE_FRACTION:
        return RED, (
            f"{DECIDER}: {frac:.4%} of parameters are trainable, above the "
            f"{MAX_TRAINABLE_FRACTION:.0%} bar; the base is not frozen"
        )

    return GREEN, (
        f"LoRA through the package API is honoured, not refused and not dropped: "
        f"{dec['ckpt_lora_b_nonzero']} of {dec['ckpt_lora_b']} lora_B tensors moved "
        f"(max norm {dec['ckpt_lora_b_max_norm']:.6g}) while the null arm left all "
        f"{null['ckpt_lora_b']} at exactly 0.0; the artifact carries "
        f"{dec['ckpt_base_tensors']} base tensors against {base['ckpt_base_tensors']} "
        f"for the full fine-tune, whose {base['ckpt_base_moved']} moved base tensor(s) "
        f"prove the detector fires; and the manifest records {frac:.4%} of parameters "
        f"trainable"
    )


# ---------------------------------------------------------------------------
# controls
# ---------------------------------------------------------------------------
def _ok_null() -> dict[str, Any]:
    return {"trained": True, "ckpt_lora_b": 112, "ckpt_lora_b_nonzero": 0,
            "ckpt_lora_b_max_norm": 0.0, "ckpt_tensors_total": 224}


def _ok_base() -> dict[str, Any]:
    return {"trained": True, "ckpt_tensors_total": 676, "ckpt_base_tensors": 676,
            "ckpt_base_compared": 12, "ckpt_base_moved": 12}


def _ok_dec() -> dict[str, Any]:
    return {
        "trained": True, "any_known_refusal": False,
        "ckpt_tensors_total": 224, "ckpt_lora_a": 112, "ckpt_lora_b": 112,
        "ckpt_lora_b_nonzero": 112, "ckpt_lora_b_max_norm": 0.209401,
        "ckpt_base_tensors": 0, "ckpt_base_compared": 0, "ckpt_base_moved": 0,
        "manifest": {
            "/config/adapter/value": "lora",
            "/config/adapter.trainable_params/value": "1089536",
            "/config/adapter.total_params/value": "1544803840",
            "/config/stage/value": "done",
        },
    }


def _payload(**over: Any) -> dict[str, dict[str, Any]]:
    out = {BASELINE: _ok_base(), DECIDER: _ok_dec(), NULL: _ok_null()}
    for arm, patch in over.items():
        if patch is None:
            del out[arm]
        else:
            out[arm].update(patch)
    return out


def _controls() -> list[tuple[str, int, dict[str, Any], str]]:
    """(name, expected, payload, why) -- the `why` is the point of the control.

    A control without a stated reason is a regression test for whatever the
    code happened to do on the day it was written.
    """
    return [
        ("the measured reading", GREEN, _payload(),
         "the real arms, verbatim, must be GREEN"),

        # --- ordering: preconditions before the deciding arm -----------------
        ("no baseline arm", UNMEASURED, _payload(pkg_full=None),
         "a missing arm is unmeasured, never a pass and never a failure"),
        ("no deciding arm", UNMEASURED, _payload(pkg_lora=None),
         "the deciding arm's absence cannot be filled in by the other two"),
        ("no null arm", UNMEASURED, _payload(pkg_lora_lr0=None),
         "without the null arm a nonzero lora_B is uninterpretable"),
        ("config rejected on the deciding arm", UNMEASURED,
         _payload(pkg_lora={"config_rejected": True}),
         "a declaration the dataclass refused to build never reached the plane"),
        ("train() raised on the null arm", UNMEASURED,
         _payload(pkg_lora_lr0={"train_raised": True}),
         "a crashed control is not a passed control, even on a healthy decider"),

        # --- the detector must be shown able to fire -------------------------
        ("baseline never trained", REFUSE,
         _payload(pkg_full={"trained": False, "ckpt_tensors_total": 0}),
         "if the package API cannot train at all, nothing downstream is readable"),
        ("baseline moved no base weights", REFUSE,
         _payload(pkg_full={"ckpt_base_moved": 0}),
         "THE control: a base-movement detector that never fires cannot witness freezing"),
        ("baseline compared no base weights", REFUSE,
         _payload(pkg_full={"ckpt_base_compared": 0, "ckpt_base_moved": 0}),
         "nothing compared is indistinguishable from nothing moved, and must not pass"),

        # --- refusal and non-execution ---------------------------------------
        ("deciding arm refused", REFUSE, _payload(pkg_lora={"any_known_refusal": True}),
         "a refusal means the capability is absent -- honest, but not GREEN"),
        ("deciding arm neither trained nor refused", REFUSE,
         _payload(pkg_lora={"trained": False}),
         "silence is not evidence"),

        # --- the null arm is what makes movement mean anything ---------------
        ("null arm moved lora_B", REFUSE,
         _payload(pkg_lora_lr0={"ckpt_lora_b_nonzero": 3, "ckpt_lora_b_max_norm": 1e-6}),
         "if lr=0 also moves the adapter, the probe is reading initialisation"),
        ("null arm norm nonzero but count zero", REFUSE,
         _payload(pkg_lora_lr0={"ckpt_lora_b_max_norm": 1e-9}),
         "either signal alone invalidates the probe; they are checked with OR"),

        # --- the #375 class: accepted and dropped ----------------------------
        ("adapter declared, no lora_B written", RED, _payload(pkg_lora={"ckpt_lora_b": 0}),
         "trained to completion with the declaration silently dropped is the worst case"),
        ("manifest omits the adapter", RED,
         _payload(pkg_lora={"manifest": {"/config/stage/value": "done"}}),
         "an unrecorded declaration makes the run unattributable"),
        ("manifest records a different adapter", RED,
         _payload(pkg_lora={"manifest": dict(_ok_dec()["manifest"], **{"/config/adapter/value": "none"})}),
         "recording the wrong mode is as bad as recording nothing"),

        # --- base frozen ------------------------------------------------------
        ("base weights moved under lora", RED, _payload(pkg_lora={"ckpt_base_moved": 1}),
         "one moved base tensor falsifies the claim; the bar is zero, not a fraction"),
        ("lora_B present but all zero", RED, _payload(pkg_lora={"ckpt_lora_b_nonzero": 0}),
         "an attached adapter that never moved is not a trained adapter"),
        ("trainable fraction not recorded", RED,
         _payload(pkg_lora={"manifest": {"/config/adapter/value": "lora"}}),
         "absent is not zero: a missing record must not read as wonderfully frozen"),
        ("trainable fraction is the whole model", RED,
         _payload(pkg_lora={"manifest": dict(
             _ok_dec()["manifest"], **{"/config/adapter.trainable_params/value": "1544803840"})}),
         "100% trainable with an adapter attached means the base was never frozen"),
        ("trainable fraction just over the bar", RED,
         _payload(pkg_lora={"manifest": dict(
             _ok_dec()["manifest"], **{"/config/adapter.trainable_params/value": "20000000"})}),
         "the bar is enforced, not decorative"),
        ("total_params zero", RED,
         _payload(pkg_lora={"manifest": dict(
             _ok_dec()["manifest"], **{"/config/adapter.total_params/value": "0"})}),
         "a zero denominator must not divide, and must not pass"),

        # --- ordering proofs --------------------------------------------------
        ("missing arm outranks a refusal", UNMEASURED,
         _payload(pkg_full=None, pkg_lora={"any_known_refusal": True}),
         "an incomplete payload is scored before its contents are believed"),
        ("a dead detector outranks a dropped declaration", REFUSE,
         _payload(pkg_full={"ckpt_base_moved": 0}, pkg_lora={"ckpt_lora_b": 0}),
         "with no working detector the RED could not have been trusted anyway"),
        ("a broken null arm outranks moved base weights", REFUSE,
         _payload(pkg_lora_lr0={"ckpt_lora_b_nonzero": 1}, pkg_lora={"ckpt_base_moved": 1}),
         "an invalid probe is not a place to read a failure from"),
    ]


def _self_test() -> int:
    controls = _controls()
    failures = []
    exercised = set()
    for name, want, payload, why in controls:
        got, why_got = t1_18_verdict(payload)
        exercised.add(got)
        if got != want:
            failures.append(f"FAIL: {name}: want {want} got {got} -- {why_got} [{why}]")
        else:
            print(f"  ok  {name} -> {got}")
    for line in failures:
        print(line)
    # A suite that never produces one of the four states has a state it has
    # never actually checked.
    if exercised != {GREEN, RED, UNMEASURED, REFUSE}:
        failures.append(f"FAIL: controls never exercised {sorted({GREEN, RED, UNMEASURED, REFUSE} - exercised)}")
        print(failures[-1])
    print(f"{'CLEAR' if not failures else 'FAILED'}: {len(controls) - len(failures)} of {len(controls)}")
    return GREEN if not failures else RED


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--payload", help="t1_18_arms.json from the probe")
    args = ap.parse_args()

    if args.self_test:
        return _self_test()
    if not args.payload:
        print("UNMEASURED: no --payload and no --self-test")
        return UNMEASURED
    try:
        payload = json.loads(open(args.payload).read())
    except Exception as exc:  # noqa: BLE001
        print(f"UNMEASURED: payload unreadable: {type(exc).__name__}: {exc}")
        return UNMEASURED

    rc, why = t1_18_verdict(payload)
    print(f"{ {GREEN: 'GREEN', RED: 'RED', UNMEASURED: 'UNMEASURED', REFUSE: 'REFUSE'}[rc] }: {why}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
