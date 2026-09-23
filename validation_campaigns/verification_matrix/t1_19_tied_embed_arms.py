"""T1-19: adjudicate whether weight tying survives FoundationScale's save/load.

SCOPE, STATED UP FRONT BECAUSE THE ROW'S CLAIM IS COMPOUND
-----------------------------------------------------------
The matrix row claims "tying survives save/load AND sharding". Only the first
conjunct is reachable on this plane. T1-14 (#481) measured that a declared
sharding_strategy was REFUSED, because no sharded backend was wired then -- so
there was no sharded save to inspect, and no amount of care with this instrument
would have produced one. `fsdp` has since been wired and is unit-tested but not
yet measured on hardware, so the sharded conjunct is now unMEASURED rather than
unreachable; this adjudicator still does not answer it. This adjudicator therefore answers the save/load conjunct and
declares the sharding conjunct unreachable rather than letting a GREEN on half a
claim be read as a GREEN on all of it.

WHAT MAKES THIS MEASURABLE AT ALL
----------------------------------
Equality is NOT tying. A model that wrote two separate copies of one tensor and
reloaded them into two separate buffers is byte-identical to a correctly tied
one at rest, and broken in motion: the next update to the embedding would not
reach the output head. So the deciding check is STORAGE IDENTITY after a round
trip -- `input_embeddings.weight.data_ptr() == output_embeddings.weight.data_ptr()`
-- not `torch.equal`. Both are recorded; only the first decides.

THE CONTROL
-----------
The row declares "untied model as control", and the control's job here is
specific: to show that the round-trip detector is CAPABLE of reporting "not
tied". Without it, `shares_storage=True` on the tied arm is consistent with a
detector that returns True unconditionally. The control must therefore reload
with SEPARATE storage, and must have written an lm_head tensor of its own --
if it did not, the two arms did not actually diverge in the save path and the
tied arm's reading is untested.

Both arms are built from ONE config differing in exactly one field, so a
difference in outcome cannot be attributed to architecture, vocabulary or
checkpoint format. That is stronger than two different pretrained checkpoints
would have been, and it is also the only option: no untied model with real
weights exists on this estate and downloading is not permitted.

Pure dict arithmetic over the payload: no torch, no transformers, no GPU, so a
laptop and a tray reach the same verdict on the same bytes.
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import Any

GREEN, RED, UNMEASURED, REFUSE = 0, 5, 95, 96

TIED = "tied"
CONTROL = "untied"
REQUIRED = (TIED, CONTROL)


def _arm(payload: dict[str, Any], name: str) -> dict[str, Any] | None:
    return (payload.get("arms") or {}).get(name)


def t1_19_verdict(payload: dict[str, Any]) -> tuple[int, str]:
    """Return (exit_code, one-line reason).

    Order is load-bearing and is the documented contract of this function:

     1. a required arm is missing                         -> UNMEASURED 95
     2. an arm died before the plane ran                  -> UNMEASURED 95
     3. an arm did not train                              -> REFUSE 96
     4. an arm trained and wrote no artifact              -> RED 5
     5. an arm's artifact would not reload                -> RED 5
     6. the control was not untied in memory              -> REFUSE 96
     7. the control's artifact carries no lm_head         -> REFUSE 96
     8. the control reloaded SHARING storage              -> RED 5
     9. the tied arm was not tied in memory               -> REFUSE 96
    10. the tied arm's embedding did not move             -> REFUSE 96
    11. the tied artifact lost the tie flag               -> RED 5
    12. the tied arm reloaded NOT sharing storage         -> RED 5
    13. everything above clean                            -> GREEN 0

    Controls (6-8) are settled BEFORE the deciding arm (9-12) is read, so a
    verdict can never rest on a detector that has not been shown to fire.
    """
    arms = {name: _arm(payload, name) for name in REQUIRED}

    for name in REQUIRED:
        if arms[name] is None:
            return UNMEASURED, f"arm {name!r} is absent from the payload; nothing to adjudicate"

    for name in REQUIRED:
        rec = arms[name] or {}
        if rec.get("build_failed"):
            return UNMEASURED, f"arm {name!r} failed to build its subject; the plane never ran"
        if rec.get("config_rejected"):
            return UNMEASURED, f"arm {name!r} had its declaration rejected; the plane never ran"
        if rec.get("train_raised"):
            return UNMEASURED, f"arm {name!r} raised inside train(); no save path was exercised"

    for name in REQUIRED:
        rec = arms[name] or {}
        if not rec.get("trained"):
            return REFUSE, (
                f"arm {name!r} produced no training; a save path that was never reached "
                "cannot be shown to preserve anything"
            )

    for name in REQUIRED:
        rec = arms[name] or {}
        if not rec.get("artifact"):
            return RED, (
                f"arm {name!r} trained and wrote no artifact; there is nothing for a "
                "save/load round trip to survive"
            )
        if not (rec.get("reload") or {}).get("reload_ok"):
            why = (rec.get("reload") or {}).get("reload_error", "no reason recorded")
            return RED, f"arm {name!r} wrote an artifact that would not reload: {why}"

    # --- controls, settled first ------------------------------------------
    ctl = arms[CONTROL] or {}
    ctl_built = ctl.get("built") or {}
    ctl_art = ctl.get("artifact") or {}
    ctl_rel = ctl.get("reload") or {}

    if ctl_built.get("inmem_shares_storage") is not False:
        return REFUSE, (
            "the control was not actually untied in memory before saving, so it cannot "
            "witness that an untied model stays untied"
        )
    if not ctl_art.get("lm_head_in_artifact"):
        return REFUSE, (
            "the control's artifact carries no lm_head tensor, so the two arms did not "
            "diverge in the save path and the tied arm's reading is untested"
        )
    if ctl_rel.get("reload_shares_storage") is not False:
        return RED, (
            "the untied control reloaded with the head SHARING the embedding's storage: "
            "the round trip tied a model that was not tied, which is a defect in the "
            "save/load path and also voids the tied arm's evidence"
        )

    # --- the deciding arm --------------------------------------------------
    tie = arms[TIED] or {}
    tie_built = tie.get("built") or {}
    tie_art = tie.get("artifact") or {}
    tie_rel = tie.get("reload") or {}

    if tie_built.get("inmem_shares_storage") is not True:
        return REFUSE, (
            "the tied arm was not tied in memory before saving, so the round trip was "
            "never handed a tied model to preserve"
        )
    if not tie_rel.get("embed_moved"):
        return REFUSE, (
            "the tied arm's embedding did not move during training; an untouched tie is "
            "not the subject of this row, which is whether a TRAINED tie survives"
        )
    if tie_art.get("config_tie") is not True:
        return RED, (
            f"the tied artifact records tie_word_embeddings={tie_art.get('config_tie')!r}: "
            "the save path dropped the declaration, so a reloader has no way to know"
        )
    if tie_rel.get("reload_shares_storage") is not True:
        return RED, (
            "the tied arm reloaded with the head in SEPARATE storage from the embedding: "
            "the values may match today but the tie is gone, and the next update to the "
            "embedding would not reach the head"
        )

    head_kind = (
        "absent from the artifact" if not tie_art.get("lm_head_in_artifact")
        else "present in the artifact but re-tied on load"
    )
    return GREEN, (
        "tying survives the save/load round trip: the tied arm reloaded with the output "
        f"head sharing the input embedding's storage (lm_head {head_kind}, config "
        f"tie_word_embeddings={tie_art.get('config_tie')!r}) after the embedding actually "
        f"moved (max |delta| {tie_rel.get('embed_maxdiff')}), while the untied control "
        "wrote its own lm_head and reloaded with separate storage, proving the detector "
        "can report a mismatch. The sharding conjunct of this row is NOT measured here "
        "and is unreachable on this plane, which refuses sharded execution (T1-14)"
    )


# ---------------------------------------------------------------------------
# controls
# ---------------------------------------------------------------------------
def _mk(
    *,
    tied_over: dict[str, Any] | None = None,
    ctl_over: dict[str, Any] | None = None,
    drop: str | None = None,
) -> dict[str, Any]:
    """A payload that is GREEN unless an override breaks exactly one thing."""

    def arm(tie: bool) -> dict[str, Any]:
        return {
            "declared_tie": tie,
            "rc": 0,
            "trained": True,
            "build_failed": False,
            "config_rejected": False,
            "train_raised": False,
            "built": {"declared_tie": tie, "config_tie": tie, "inmem_shares_storage": tie},
            "artifact": {
                "n_tensors": 26 if tie else 27,
                "lm_head_in_artifact": not tie,
                "config_tie": tie,
            },
            "reload": {
                "reload_ok": True,
                "reload_shares_storage": tie,
                "reload_equal": tie,
                "embed_moved": True,
                "embed_maxdiff": 0.0031738,
            },
        }

    arms = {TIED: arm(True), CONTROL: arm(False)}
    for name, over in ((TIED, tied_over), (CONTROL, ctl_over)):
        for path, value in (over or {}).items():
            head, _, leaf = path.rpartition(".")
            target = arms[name]
            for part in filter(None, head.split(".")):
                target = target[part]
            target[leaf] = value
    if drop:
        arms.pop(drop)
    return {"arms": arms}


# (name, expected, payload, why). A control without a stated reason is a
# regression test for whatever the code happened to do on the day it was written.
CONTROLS: list[tuple[str, int, dict[str, Any], str]] = [
    ("green_clean", GREEN, _mk(),
     "both arms behave as the claim requires; this is the only shape that may pass"),
    ("missing_tied", UNMEASURED, _mk(drop=TIED),
     "no deciding arm at all is a gap in the measurement, not a failure of the framework"),
    ("missing_control", UNMEASURED, _mk(drop=CONTROL),
     "without the control the detector is unproven, and an unproven detector is not a pass"),
    ("tied_build_failed", UNMEASURED, _mk(tied_over={"build_failed": True}),
     "the subject could not be constructed, so the save path was never asked anything"),
    ("ctl_build_failed", UNMEASURED, _mk(ctl_over={"build_failed": True}),
     "same for the control: an environment fault is not evidence about the framework"),
    ("tied_config_rejected", UNMEASURED, _mk(tied_over={"config_rejected": True}),
     "a rejected declaration stops before the plane; that is a different row's question"),
    ("ctl_train_raised", UNMEASURED, _mk(ctl_over={"train_raised": True}),
     "a crash inside train() leaves no save path exercised and must not read as a refusal"),
    ("tied_not_trained", REFUSE, _mk(tied_over={"trained": False}),
     "a save path that was never reached cannot be shown to preserve a tie"),
    ("ctl_not_trained", REFUSE, _mk(ctl_over={"trained": False}),
     "an untrained control witnesses nothing, so the deciding arm cannot be certified"),
    ("tied_no_artifact", RED, _mk(tied_over={"artifact": None}),
     "training completed and wrote nothing; there is no round trip to survive"),
    ("ctl_no_artifact", RED, _mk(ctl_over={"artifact": None}),
     "the control must also produce something, or its separateness is unobserved"),
    ("tied_reload_failed", RED, _mk(tied_over={"reload.reload_ok": False}),
     "an artifact that will not load back has not preserved anything, whatever it contains"),
    ("ctl_reload_failed", RED, _mk(ctl_over={"reload.reload_ok": False}),
     "same for the control; a detector that cannot run is not a detector"),
    ("ctl_was_tied_in_memory", REFUSE, _mk(ctl_over={"built.inmem_shares_storage": True}),
     "a 'control' that was tied all along is a second copy of the run arm, not a control"),
    ("ctl_inmem_unknown", REFUSE, _mk(ctl_over={"built.inmem_shares_storage": None}),
     "absent is not False: an unrecorded build fact must not read as a satisfied control"),
    ("ctl_no_lm_head_saved", REFUSE, _mk(ctl_over={"artifact.lm_head_in_artifact": False}),
     "if the untied model saved no head either, the two arms never diverged in the save path"),
    ("ctl_reloaded_shared", RED, _mk(ctl_over={"reload.reload_shares_storage": True}),
     "the round trip TIED an untied model, which is a real defect and voids the tied arm"),
    ("ctl_reload_shared_unknown", RED, _mk(ctl_over={"reload.reload_shares_storage": None}),
     "an unrecorded control result is not a passing control result"),
    ("tied_not_tied_in_memory", REFUSE, _mk(tied_over={"built.inmem_shares_storage": False}),
     "the round trip was handed an untied model, so it cannot have preserved a tie"),
    ("tied_inmem_unknown", REFUSE, _mk(tied_over={"built.inmem_shares_storage": None}),
     "absent is not True; a missing build fact must not be read as a satisfied precondition"),
    ("tied_embed_did_not_move", REFUSE, _mk(tied_over={"reload.embed_moved": False}),
     "the row is about a TRAINED tie; an untouched tensor is a weaker thing entirely"),
    ("tied_config_lost_tie", RED, _mk(tied_over={"artifact.config_tie": False}),
     "the save path dropped the declaration, so any other reloader would get it wrong"),
    ("tied_config_tie_absent", RED, _mk(tied_over={"artifact.config_tie": None}),
     "an unrecorded flag is not a recorded True; absent must not pass"),
    ("tied_reloaded_separate", RED, _mk(tied_over={"reload.reload_shares_storage": False}),
     "the deciding failure: values may match today, but the tie is gone"),
    ("tied_equal_but_separate", RED,
     _mk(tied_over={"reload.reload_shares_storage": False, "reload.reload_equal": True}),
     "byte-equality is exactly the trap this row exists to catch; it must not rescue a RED"),
    ("tied_head_saved_but_retied", GREEN,
     _mk(tied_over={"artifact.lm_head_in_artifact": True, "artifact.n_tensors": 27}),
     "a redundant saved head is wasteful, not broken, when the round trip still shares "
     "storage; gating on artifact shape here would manufacture a false RED"),
]


def _self_test() -> int:
    failures: list[str] = []
    exercised: set[int] = set()
    for name, expected, payload, why in CONTROLS:
        got, reason = t1_19_verdict(payload)
        exercised.add(got)
        if got != expected:
            failures.append(f"{name}: expected {expected}, got {got} ({reason})")
        else:
            print(f"  ok  {name:28s} -> {got:2d}   {why}")
    if exercised != {GREEN, RED, UNMEASURED, REFUSE}:
        failures.append(f"controls never reach {sorted({GREEN, RED, UNMEASURED, REFUSE} - exercised)}")
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
    code, reason = t1_19_verdict(payload)
    label = {GREEN: "GREEN", RED: "RED", UNMEASURED: "UNMEASURED", REFUSE: "REFUSE"}[code]
    print(f"{label}: {reason}")
    return code


if __name__ == "__main__":
    sys.exit(main())
