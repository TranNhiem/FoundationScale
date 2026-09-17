"""T1-3 adjudicator: is the nvfp4 refusal real, or a bf16 run wearing its name?

The row asserted "REFUSES by design (#345)" with no adjudicator and no finding.
An asserted refusal is the weakest thing in this matrix: the failure it names
(#342's defect class -- train at the loader's default dtype and let the
manifest say nvfp4) is INVISIBLE from the outside, because a silent fallback
and a working backend produce the same exit 0 and the same checkpoint.

WHY EXIT CODES ALONE DECIDE NOTHING. The train plane exits 96 from two
independent channels, and a bad `--precision` can reach either:

  validator  `TrainConfig.__post_init__` raises for a precision outside
             PRECISIONS; `cli.main` prints "declaration rejected" -> 96.
  plane      `_train` marks Step.REFUSE naming "the package plane has no nvfp4
             backend", emits a refused manifest -> 96.

nvfp4 is IN PRECISIONS, so it clears the validator and must be stopped by the
plane. Delete the plane's refusal and nvfp4 still clears the validator, then
trains at the loader default and exits 0. So "rc == 96" is consistent with a
validator rejection, a usage error (this CLI maps argparse's SystemExit into
the contract too) and a real refusal alike. Every check below therefore keys on
WHICH channel spoke, never on the number.

Verdict order -- controls are settled BEFORE the deciding arm is read, because
each one is a way for the deciding arm's answer to mean nothing:

  1. any arm unmeasured                      -> UNMEASURED 95
  2. any arm died in the parser              -> UNMEASURED 95  (reached no plane)
  3. bf16 + real model did not train         -> REFUSE 96  (fixtures dead here)
  4. bogus precision not rejected by name    -> REFUSE 96  (field unvalidated)
  5. bogus precision fired the plane marker  -> REFUSE 96  (channels not distinct)
  6. bf16 + absent model not fatal           -> REFUSE 96  (arm 7 would be vacuous)
  7. nvfp4 + absent model did not refuse     -> REFUSE 96  (order unproven)
  8. deciding arm trained                    -> RED 5      (the silent fallback)
  9. deciding arm refused but recorded a
     precision it was never given            -> RED 5      (#342 in writing)
 10. deciding arm refused, named nvfp4,
     trained nothing                         -> GREEN 0
 11. anything else                           -> REFUSE 96

Check 6 is the one that is easy to leave out and fatal to omit. Arm 7 proves
the refusal precedes model resolution -- but only if an unresolvable model
would otherwise have been fatal. Without arm 6 measuring that, arm 7's quiet
about the model is equally consistent with the plane not caring about models at
all.

Pure: `nvfp4_refusal_verdict` does no I/O and imports no torch. The controls
run under `python3 -S`.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROG = "t1_3_nvfp4_refusal_arms"
ROW_ID = "T1-3"

GREEN = 0
RED = 5
UNMEASURED = 95
REFUSE = 96

DECIDING_ARM = "nvfp4_real"
TRAIN_CONTROL = "bf16_real"
VALIDATOR_CONTROL = "bogus_precision"
FATALITY_CONTROL = "bf16_absent_model"
ORDER_CONTROL = "nvfp4_absent_model"
ARMS = (TRAIN_CONTROL, DECIDING_ARM, ORDER_CONTROL, FATALITY_CONTROL, VALIDATOR_CONTROL)


def nvfp4_refusal_verdict(payload: dict[str, Any]) -> tuple[int, str]:
    """Return (exit_code, one-line reason). Pure."""
    missing = [a for a in ARMS if a not in payload]
    if missing:
        return UNMEASURED, f"{ROW_ID}: arms absent from the payload: {', '.join(missing)}; 95"

    for arm in ARMS:
        row = payload[arm]
        if row.get("status") != "measured" or row.get("launcher_exit_code") is None:
            return (
                UNMEASURED,
                f"{ROW_ID}: arm {arm!r} did not produce a reading "
                f"(status={row.get('status')!r}); 95",
            )

    # 2. The parser is in front of every plane. A misdeclared arm exits 96 here
    #    exactly like a refusal, so it must be caught rather than counted.
    for arm in ARMS:
        if payload[arm].get("parser_usage_error"):
            return (
                UNMEASURED,
                f"{ROW_ID}: arm {arm!r} was rejected by the argument parser and "
                "never reached a training plane, so its exit says nothing about "
                "precision; 95",
            )

    # 3. Can these fixtures train at all on this host? Without this, a 96 from
    #    the deciding arm is as consistent with a broken environment as with a
    #    deliberate refusal.
    train = payload[TRAIN_CONTROL]
    if not train.get("wrote_checkpoint"):
        return (
            REFUSE,
            f"{ROW_ID}: the bf16 control wrote no checkpoint, so 'no checkpoint "
            "written' from the nvfp4 arm is an emptiness never shown to be "
            "fillable rather than evidence of a refusal; 96",
        )
    if not train.get("trained") or train.get("launcher_exit_code") != GREEN:
        return (
            REFUSE,
            f"{ROW_ID}: the bf16 control did not train to completion here "
            f"(exit={train.get('launcher_exit_code')}, trained={train.get('trained')}), "
            "so a refusal from the nvfp4 arm cannot be attributed to the "
            "precision; 96",
        )

    # 4/5. Is the precision field checked at all, and is the validator's 96
    #      distinguishable from the plane's 96?
    bogus = payload[VALIDATOR_CONTROL]
    if not (bogus.get("validator_rejected") and bogus.get("names_bogus_value")):
        return (
            REFUSE,
            f"{ROW_ID}: an unsupported precision was not rejected by name "
            f"(validator={bogus.get('validator_rejected')}, "
            f"named={bogus.get('names_bogus_value')}), so nothing shows the "
            "precision field is validated; 96",
        )
    if bogus.get("plane_refused_nvfp4"):
        return (
            REFUSE,
            f"{ROW_ID}: the unsupported-precision arm emitted the plane's nvfp4 "
            "refusal text, so the two 96 channels cannot be told apart in the "
            "output; 96",
        )

    # 6. Would an unresolvable model otherwise be fatal? This is what makes the
    #    ordering control below mean anything.
    fatal = payload[FATALITY_CONTROL]
    if fatal.get("plane_refused_nvfp4"):
        return (
            REFUSE,
            f"{ROW_ID}: the bf16 arm with an absent model emitted the nvfp4 "
            "refusal text, so that text is not attributable to the declared "
            "precision; 96",
        )
    if fatal.get("trained") or fatal.get("launcher_exit_code") == GREEN:
        return (
            REFUSE,
            f"{ROW_ID}: an absent model was not fatal at bf16 "
            f"(exit={fatal.get('launcher_exit_code')}, trained={fatal.get('trained')}), "
            "so the nvfp4 arm staying quiet about the model proves no ordering; 96",
        )

    # 7. Does the refusal really precede model resolution, as the loop's
    #    docstring claims ("refused before anything runs")?
    order = payload[ORDER_CONTROL]
    if not order.get("plane_refused_nvfp4") or order.get("launcher_exit_code") != REFUSE:
        return (
            REFUSE,
            f"{ROW_ID}: nvfp4 with an unresolvable model did not produce the "
            f"plane's refusal (exit={order.get('launcher_exit_code')}, "
            f"marker={order.get('plane_refused_nvfp4')}), so the refusal is not "
            "shown to precede the model load; 96",
        )
    if order.get("trained"):
        return REFUSE, f"{ROW_ID}: the ordering control trained, which it cannot have done; 96"

    # --- the deciding arm, now that its answer can carry weight -------------
    dec = payload[DECIDING_ARM]
    if dec.get("trained") or dec.get("launcher_exit_code") == GREEN:
        return (
            RED,
            f"{ROW_ID}: nvfp4 was declared and the plane trained anyway "
            f"(exit={dec.get('launcher_exit_code')}, trained={dec.get('trained')}) -- "
            "a silent fallback to another dtype under an nvfp4 label, which is "
            "the defect this row asserts cannot happen; 5",
        )

    # The row's own control_arm: "assert rc==96 AND no checkpoint written".
    # A refusal message with weights on disk beside it is not a refusal.
    if dec.get("wrote_checkpoint"):
        return (
            RED,
            f"{ROW_ID}: nvfp4 was refused but {dec.get('checkpoint_artifacts')} "
            "checkpoint artifact(s) were written anyway, so the run produced "
            "weights at a dtype nobody authorised; 5",
        )

    recorded = [p for p in dec.get("manifest_precisions", []) if p and p != "nvfp4"]
    if recorded:
        return (
            RED,
            f"{ROW_ID}: the run refused, but its manifest records precision "
            f"{recorded[0]!r} for a run that only ever declared nvfp4 -- a dtype "
            "the operator never authorised, written into the audit trail; 5",
        )

    if dec.get("launcher_exit_code") == REFUSE and dec.get("plane_refused_nvfp4"):
        return (
            GREEN,
            f"{ROW_ID}: a declared nvfp4 run refuses (96) from the plane's own "
            "refusal site, naming the missing backend, before any weights load "
            "and with nothing trained, while the identical bf16 arm trains to "
            "completion on the same fixtures; 0",
        )

    return (
        REFUSE,
        f"{ROW_ID}: the deciding arm neither trained nor produced the plane's "
        f"refusal (exit={dec.get('launcher_exit_code')}, "
        f"marker={dec.get('plane_refused_nvfp4')}); unclassifiable; 96",
    )


# --------------------------------------------------------------------------
# controls
# --------------------------------------------------------------------------


def _arm(**over: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "status": "measured",
        "launcher_exit_code": REFUSE,
        "validator_rejected": False,
        "plane_refused_nvfp4": False,
        "names_bogus_value": False,
        "names_absent_model": False,
        "parser_usage_error": False,
        "trained": False,
        "manifest_precisions": [],
        "manifest_refused_stage": False,
        "wrote_checkpoint": False,
        "checkpoint_artifacts": 0,
    }
    base.update(over)
    return base


def _green_payload() -> dict[str, Any]:
    """The shape actually measured on a tray: refusal real, controls all firing."""
    return {
        TRAIN_CONTROL: _arm(
            launcher_exit_code=GREEN,
            trained=True,
            manifest_precisions=["bf16"],
            wrote_checkpoint=True,
            checkpoint_artifacts=1,
        ),
        DECIDING_ARM: _arm(
            plane_refused_nvfp4=True,
            manifest_refused_stage=True,
            manifest_precisions=["nvfp4"],
        ),
        ORDER_CONTROL: _arm(plane_refused_nvfp4=True, names_absent_model=True),
        FATALITY_CONTROL: _arm(launcher_exit_code=RED, names_absent_model=True),
        VALIDATOR_CONTROL: _arm(validator_rejected=True, names_bogus_value=True),
    }


def _mutate(arm: str, **over: Any) -> dict[str, Any]:
    p = _green_payload()
    p[arm] = _arm(**{**p[arm], **over})
    return p


def _controls() -> list[tuple[str, int, dict[str, Any], str]]:
    drop = _green_payload()
    del drop[ORDER_CONTROL]
    return [
        ("measured-green", GREEN, _green_payload(), "the tray reading"),
        ("arm-missing", UNMEASURED, drop, "an absent arm is not a pass"),
        (
            "arm-timeout",
            UNMEASURED,
            _mutate(DECIDING_ARM, status="timeout", launcher_exit_code=None),
            "a timeout has no exit code to read",
        ),
        (
            "exit-none",
            UNMEASURED,
            _mutate(TRAIN_CONTROL, launcher_exit_code=None),
            "a missing exit code is unmeasured, never green",
        ),
        (
            "parser-error-deciding",
            UNMEASURED,
            _mutate(DECIDING_ARM, parser_usage_error=True),
            "a usage error exits 96 without reaching a plane",
        ),
        (
            "parser-error-control",
            UNMEASURED,
            _mutate(VALIDATOR_CONTROL, parser_usage_error=True),
            "a malformed control is not a satisfied control",
        ),
        (
            "fixtures-cannot-train",
            REFUSE,
            _mutate(TRAIN_CONTROL, trained=False),
            "if bf16 cannot train here, nvfp4's refusal is unattributable",
        ),
        (
            "bf16-control-not-green",
            REFUSE,
            _mutate(TRAIN_CONTROL, launcher_exit_code=RED, trained=True),
            "a bf16 arm that ends RED is not a working baseline",
        ),
        (
            "precision-field-unvalidated",
            REFUSE,
            _mutate(VALIDATOR_CONTROL, validator_rejected=False),
            "an unchecked field makes nvfp4's fate incidental",
        ),
        (
            "validator-silent-about-value",
            REFUSE,
            _mutate(VALIDATOR_CONTROL, names_bogus_value=False),
            "a rejection that does not name what it rejected proves little",
        ),
        (
            "channels-indistinguishable",
            REFUSE,
            _mutate(VALIDATOR_CONTROL, plane_refused_nvfp4=True),
            "both 96 channels emitting the same text collapses the attribution",
        ),
        (
            "absent-model-not-fatal",
            REFUSE,
            _mutate(FATALITY_CONTROL, launcher_exit_code=GREEN, trained=True),
            "if a missing model is survivable, ordering cannot be inferred",
        ),
        (
            "fatality-control-names-nvfp4",
            REFUSE,
            _mutate(FATALITY_CONTROL, plane_refused_nvfp4=True),
            "the refusal text must not appear for a bf16 declaration",
        ),
        (
            "ordering-unproven",
            REFUSE,
            _mutate(ORDER_CONTROL, plane_refused_nvfp4=False, launcher_exit_code=RED),
            "without this the refusal may follow the model load",
        ),
        (
            "ordering-control-trained",
            REFUSE,
            _mutate(ORDER_CONTROL, trained=True),
            "an arm with no resolvable model cannot have trained",
        ),
        (
            "silent-fallback-trained",
            RED,
            _mutate(DECIDING_ARM, trained=True, launcher_exit_code=GREEN),
            "the exact defect the row asserts cannot happen",
        ),
        (
            "silent-fallback-exit-zero",
            RED,
            _mutate(DECIDING_ARM, launcher_exit_code=GREEN),
            "exit 0 under a declared nvfp4 is a fallback even if quiet",
        ),
        (
            "checkpoint-detector-cannot-fire",
            REFUSE,
            _mutate(TRAIN_CONTROL, wrote_checkpoint=False, checkpoint_artifacts=0),
            "absence of weights means nothing if weights never appear here",
        ),
        (
            "refused-but-wrote-weights",
            RED,
            _mutate(
                DECIDING_ARM,
                plane_refused_nvfp4=True,
                wrote_checkpoint=True,
                checkpoint_artifacts=2,
            ),
            "the row's own control arm: rc==96 AND no checkpoint written",
        ),
        (
            "manifest-records-other-dtype",
            RED,
            _mutate(DECIDING_ARM, plane_refused_nvfp4=True, manifest_precisions=["bf16"]),
            "#342 laundering: a dtype recorded that was never declared",
        ),
        (
            "refused-without-the-marker",
            REFUSE,
            _mutate(DECIDING_ARM, plane_refused_nvfp4=False),
            "a 96 with no refusal text is unclassifiable, not a pass",
        ),
    ]


def _self_test() -> int:
    controls = _controls()
    failures = 0
    for name, want, payload, why in controls:
        got, reason = nvfp4_refusal_verdict(payload)
        ok = got == want
        failures += not ok
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: want {want} got {got} -- {why}")
        if not ok:
            print(f"         reason: {reason}")

    # A suite that never produces a verdict cannot detect losing it.
    exercised = {nvfp4_refusal_verdict(p)[0] for _, _, p, _ in controls}
    if exercised != {GREEN, RED, UNMEASURED, REFUSE}:
        print(f"[FAIL] control set exercises only {sorted(exercised)}")
        failures += 1

    print(f"{'CLEAR' if not failures else 'FAILING'}: {len(controls) - failures} "
          f"of {len(controls)} controls")
    return GREEN if not failures else RED


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog=PROG)
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--payload", help="t1_3_arms.json written by t1_3_probe.py")
    args = ap.parse_args(argv)

    if args.self_test:
        return _self_test()
    if not args.payload:
        print("UNMEASURED: --payload is required (or --self-test); 95", file=sys.stderr)
        return UNMEASURED

    # An unmet precondition is 95 or 96, never a traceback: exiting 1 here
    # would put this adjudicator outside the contract it exists to enforce.
    try:
        payload = json.loads(Path(args.payload).read_text())
    except OSError as exc:
        print(f"UNMEASURED: --payload {args.payload!r} is unreadable ({exc}); 95", file=sys.stderr)
        return UNMEASURED
    except ValueError as exc:
        print(f"REFUSE: --payload {args.payload!r} is not valid JSON ({exc}); 96", file=sys.stderr)
        return REFUSE

    code, reason = nvfp4_refusal_verdict(payload)
    print(reason)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
