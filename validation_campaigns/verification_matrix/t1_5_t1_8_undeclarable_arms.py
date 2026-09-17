"""T1-5 / T1-8: adjudicate the SHAPE of an absence, not its presence.

Resume-from-checkpoint and resharding are not implemented. That is settled and
this file does not relitigate it -- there is no resume surface and no reshard
surface in the training plane, only comments. GREEN here does NOT mean either
capability works. It means the absence is a SAFE absence.

The distinction is the whole point. A framework that accepts
--resume-from-checkpoint, ignores it, trains from scratch and exits 0 is
strictly worse than one that has no such flag, because the user believes they
resumed and has a clean exit code to back the belief. That is the #375 class:
a declaration accepted and silently dropped. Proving a capability is absent is
cheap; proving nobody can be fooled into thinking it is present is the part that
protects a reader, and it is what this adjudicates.

WHY THE EXIT CODE IS NOT THE EVIDENCE
--------------------------------------
This CLI maps argparse's usage error into the same 96 the training plane uses
for a designed refusal, so rc 96 on an unknown flag is consistent with BOTH "no
such flag" and "refused by design" -- and, for that matter, with a flag the
parser accepted while something else refused. The discriminator has to be the
marker argparse prints when it does not recognise an option. So the scoring key
is the marker, with rc recorded and never deciding. The known-flag control is
what makes the marker mean something: a parser that emitted it for every flag
would otherwise score identically to one that works.

Pure dict arithmetic over the payload: no torch, no CLI, no GPU.
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import Any

GREEN, RED, UNMEASURED, REFUSE = 0, 5, 95, 96

REQUIRED_AXES = ("resume", "reshard")


def t1_5_t1_8_verdict(payload: dict[str, Any]) -> tuple[int, str]:
    """Return (exit_code, one-line reason).

    Order is load-bearing and is the documented contract of this function:

    1. a required axis is missing                     -> UNMEASURED 95
    2. a control is missing                           -> UNMEASURED 95
    3. the known flag was ALSO reported unknown       -> REFUSE 96
    4. the known field raised at construction         -> REFUSE 96
    5. an invocation never reached argparse           -> UNMEASURED 95
    6. an axis probed no surface at all               -> UNMEASURED 95
    7. an arm's diagnosis is identical to the control  -> UNMEASURED 95
    8. an unimplemented flag was NOT rejected by name -> RED 5
    9. an unimplemented field was accepted            -> RED 5
   10. everything above clean                         -> GREEN 0

    The two controls settle before any deciding arm is read, because both of
    them are ways for every deciding arm to look good for the wrong reason.
    """
    axes = payload.get("axes") or {}
    controls = payload.get("controls") or {}

    for axis in REQUIRED_AXES:
        if axis not in axes:
            return UNMEASURED, f"axis {axis!r} is absent from the payload; nothing to adjudicate"
    for name in ("known_flag", "known_kwarg"):
        if name not in controls:
            return UNMEASURED, (
                f"control {name!r} is absent; without it a surface that rejects EVERYTHING "
                "would score exactly like one that rejects only what does not exist"
            )

    kf = controls["known_flag"]
    if kf.get("unknown_marker"):
        return REFUSE, (
            f"the control flag {kf.get('argv')} was itself reported unrecognised, so the parser "
            "rejects flags that do exist and the marker cannot witness anything about the ones "
            "that do not"
        )
    kk = controls["known_kwarg"]
    if kk.get("raised") is not None:
        return REFUSE, (
            f"the control field {kk.get('name')!r} raised {kk.get('raised')} at construction, so "
            "the constructor rejects fields that do exist and cannot witness the absence of "
            "fields that do not"
        )

    for rec in [kf, *(f for a in REQUIRED_AXES for f in (axes[a].get("flags") or []))]:
        if not rec.get("reached_parser", True):
            return UNMEASURED, (
                f"the invocation {rec.get('argv')} never reached argparse "
                f"(rc={rec.get('rc')}, last line {rec.get('stderr_tail')!r}): a run that died "
                "before the parser emits no unknown-flag marker for the same reason an ACCEPTED "
                "flag emits none, so this is a broken environment and not a framework defect"
            )

    for axis in REQUIRED_AXES:
        rec = axes[axis]
        flags = rec.get("flags") or []
        kwargs = rec.get("kwargs") or []
        if not flags or not kwargs:
            return UNMEASURED, (
                f"axis {axis!r} probed no surface (flags={len(flags)}, fields={len(kwargs)}): "
                "an absence nobody looked for is not an absence anybody measured"
            )
        for got in flags:
            if not got.get("unknown_marker") and got.get("diag") == kf.get("diag"):
                return UNMEASURED, (
                    f"{got.get('argv')} produced the SAME diagnosis as the control "
                    f"({got.get('diag')!r}), so the two arms are indistinguishable and nothing "
                    "was measured. This is what an incomplete command line looks like: argparse "
                    "checks required arguments inside parse_known_args and errors there, before "
                    "parse_args ever reports a leftover, so the unrecognized-argument message is "
                    "never printed and an unmeasurable run impersonates a framework that swallows "
                    "unknown flags"
                )
            if not got.get("unknown_marker"):
                return RED, (
                    f"the CLI did NOT reject {got.get('argv')} as an unrecognised option "
                    f"(rc={got.get('rc')}, last line {got.get('stderr_tail')!r}): {axis} is not "
                    "implemented, so a command line declaring it that is not refused by name is "
                    "a declaration accepted and silently dropped -- the #375 class, and the one "
                    "way an absence becomes dangerous"
                )
        for got in kwargs:
            if got.get("raised") != "TypeError":
                return RED, (
                    f"TrainConfig accepted the field {got.get('name')!r} "
                    f"(raised={got.get('raised')!r}): {axis} is not implemented, so a config "
                    "object that takes the field and does nothing with it lets the package API "
                    "make a promise the plane cannot keep"
                )

    n_flags = sum(len(axes[a]["flags"]) for a in REQUIRED_AXES)
    n_fields = sum(len(axes[a]["kwargs"]) for a in REQUIRED_AXES)
    return GREEN, (
        f"the absence is a SAFE absence on both surfaces a user can reach: all {n_flags} probed "
        f"resume/reshard command-line spellings are refused BY NAME as unrecognised options, and "
        f"all {n_fields} probed TrainConfig fields raise TypeError at construction, while a flag "
        "and a field that DO exist pass both checks -- so the rejection is about these "
        "capabilities and not about everything. This is NOT a measurement that resume or "
        "resharding work; both remain unimplemented and both rows remain ABSENT. It is a "
        "measurement that neither can be declared and silently dropped, which is the failure "
        "mode that would let a user believe they had resumed when they had trained from scratch"
    )


# ---------------------------------------------------------------------------
# controls
# ---------------------------------------------------------------------------
def _mk(**over: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "controls": {
            "known_flag": {"argv": ["--max-steps", "5"], "rc": 96, "unknown_marker": False, "reached_parser": True,
                           "diag": "[fs:train:prologue] dry run complete",
                           "stderr_tail": "[fs:train:prologue] dry run complete"},
            "known_kwarg": {"name": "max_steps", "raised": None, "msg": ""},
        },
        "axes": {
            "resume": {
                "flags": [{"argv": ["--resume-from-checkpoint", "/x"], "rc": 96,
                           "unknown_marker": True, "reached_parser": True,
                           "diag": "error: unrecognized arguments",
                           "stderr_tail": "unrecognized arguments"}],
                "kwargs": [{"name": "resume_from_checkpoint", "raised": "TypeError",
                            "msg": "unexpected keyword argument"}],
            },
            "reshard": {
                "flags": [{"argv": ["--reshard-from", "/x"], "rc": 96,
                           "unknown_marker": True, "reached_parser": True,
                           "diag": "error: unrecognized arguments",
                           "stderr_tail": "unrecognized arguments"}],
                "kwargs": [{"name": "reshard_from", "raised": "TypeError",
                            "msg": "unexpected keyword argument"}],
            },
        },
    }
    for path, value in over.items():
        head, _, tail = path.partition("__")
        if tail == "DROP_AXIS":
            payload["axes"].pop(head)
        elif tail == "DROP_CONTROL":
            payload["controls"].pop(head)
        elif tail == "EMPTY_FLAGS":
            payload["axes"][head]["flags"] = []
        elif tail == "EMPTY_KWARGS":
            payload["axes"][head]["kwargs"] = []
        elif head in payload["controls"]:
            payload["controls"][head][tail] = value
        else:
            axis, _, field = tail.partition("__")
            target = payload["axes"][head]["flags" if axis == "flag" else "kwargs"][0]
            target[field] = value
    return payload


CONTROLS: list[tuple[str, int, dict[str, Any], str]] = [
    ("green_safe_absence", GREEN, _mk(),
     "both surfaces refuse by name, both controls pass: the only shape that may pass"),
    ("missing_resume_axis", UNMEASURED, _mk(resume__DROP_AXIS=1),
     "one axis cannot stand in for the other; they are separate rows and separate spellings"),
    ("missing_reshard_axis", UNMEASURED, _mk(reshard__DROP_AXIS=1),
     "same in the other direction, so neither axis is silently inferred from its sibling"),
    ("missing_flag_control", UNMEASURED, _mk(known_flag__DROP_CONTROL=1),
     "without it, a parser that rejects every flag is indistinguishable from a working one"),
    ("missing_kwarg_control", UNMEASURED, _mk(known_kwarg__DROP_CONTROL=1),
     "without it, a constructor that rejects every field looks like correct strictness"),
    ("control_flag_also_unknown", REFUSE, _mk(known_flag__unknown_marker=True),
     "the marker fires on a flag that exists, so it carries no information about ones that do not"),
    ("control_kwarg_raised", REFUSE, _mk(known_kwarg__raised="TypeError"),
     "a constructor rejecting a real field cannot witness the rejection of an unreal one"),
    ("resume_no_flags_probed", UNMEASURED, _mk(resume__EMPTY_FLAGS=1),
     "an absence nobody looked for is not an absence anybody measured"),
    ("reshard_no_fields_probed", UNMEASURED, _mk(reshard__EMPTY_KWARGS=1),
     "the package API is a second reachable surface and skipping it is not a pass"),
    ("resume_flag_accepted", RED, _mk(resume__flag__unknown_marker=False),
     "the #375 class itself: a resume declaration accepted and silently dropped"),
    ("reshard_flag_accepted", RED, _mk(reshard__flag__unknown_marker=False),
     "same defect on the other axis, so neither is covered by the other's result"),
    ("resume_flag_zero_rc", RED, _mk(resume__flag__unknown_marker=False,
                                     resume__flag__rc=0),
     "the worst version: accepted AND exit 0, which is what a user would read as success"),
    ("resume_field_accepted", RED, _mk(resume__kwarg__raised=None),
     "TrainConfig taking the field would let the package API promise what the plane cannot keep"),
    ("reshard_field_accepted", RED, _mk(reshard__kwarg__raised=None),
     "a **kwargs sink added for convenience is exactly how a loud absence turns quiet"),
    ("field_raised_wrong_error", RED, _mk(resume__kwarg__raised="ValueError"),
     "a ValueError is the validator rejecting a VALUE, which implies the field exists"),
    ("control_parsed_cleanly_so_no_usage_line", GREEN,
     _mk(known_flag__diag="[fs:train:done] dry-run PASS: full validation prologue"),
     "a control that PARSES prints no usage line, and reading its absence as death before the "
     "parser would refuse every healthy run; this probe did exactly that on its second run"),
    ("import_died_before_parser", UNMEASURED,
     _mk(resume__flag__reached_parser=False, resume__flag__unknown_marker=False,
         resume__flag__rc=1, resume__flag__stderr_tail="ModuleNotFoundError: No module named 'torch'"),
     "a broken environment produces the same missing marker an ACCEPTED flag would; scoring it "
     "RED would blame the framework for the machine"),
    ("control_died_before_parser", UNMEASURED,
     _mk(known_flag__reached_parser=False),
     "if the control never reached the parser, neither did anything else it is meant to calibrate"),
    ("arm_indistinguishable_from_control", UNMEASURED,
     _mk(resume__flag__unknown_marker=False,
         resume__flag__diag="[fs:train:prologue] dry run complete"),
     "identical diagnosis to the control means the probe never reached the unknown-flag check; "
     "scoring that RED would report a probe bug as a framework defect, which this probe did on "
     "its first run"),
    ("flag_refused_but_not_by_name", RED,
     _mk(resume__flag__unknown_marker=False, resume__flag__rc=96,
         resume__flag__diag="[fs:train:refuse] declaration rejected"),
     "rc 96 alone is the trap this file exists to avoid: the CLI maps argparse's usage error "
     "onto the same code a designed refusal uses, so 96 without the marker proves nothing"),
]


def _self_test() -> int:
    failures: list[str] = []
    exercised: set[int] = set()
    for name, expected, payload, why in CONTROLS:
        got, reason = t1_5_t1_8_verdict(payload)
        exercised.add(got)
        if got != expected:
            failures.append(f"{name}: expected {expected}, got {got} ({reason})")
        else:
            print(f"  ok  {name:28s} -> {got:2d}   {why}")
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
    code, reason = t1_5_t1_8_verdict(payload)
    label = {GREEN: "GREEN", RED: "RED", UNMEASURED: "UNMEASURED", REFUSE: "REFUSE"}[code]
    print(f"{label}: {reason}")
    return code


if __name__ == "__main__":
    sys.exit(main())
