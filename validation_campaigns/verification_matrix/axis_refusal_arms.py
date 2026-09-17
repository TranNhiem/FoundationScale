"""Adjudicate a declared-axis refusal on the training plane (T1-13/T1-14/T1-17).

Three rows assert the same shape: an axis can be DECLARED through the training
CLI, the plane has no executor for it, and the plane is said to refuse rather
than train something else under the declared label. None of the three had ever
been measured; all three said "REFUSED by the package plane" on the strength of
reading the source.

Reading the source is not enough, because the failure this guards against is
invisible from outside: a plane that silently ignored the declaration and
trained plain DDP would also exit cleanly, also write a checkpoint, and also
leave a manifest recording the axis. The only way to tell the two apart is to
vary one term at a time and score on WHICH refusal site spoke.

ONE ADJUDICATOR, THREE ROWS
---------------------------
The rows differ only in which flag is declared and which sentence the plane
prints, so they are one instrument parametrized by axis rather than three
near-copies. `--axis` selects; the shared controls are scored identically for
all three.

WHY `rc == 96` IS NOT THE MEASUREMENT
-------------------------------------
This CLI reaches 96 from an argparse usage error, from the config validator,
and from FOUR separate sites inside the training loop (nvfp4, unwired degrees,
sharding strategy, optimizer offload). A bare 96 is therefore consistent with
at least six different things, only one of which is the claim. Every arm is
scored on its marker, and an arm carrying ANOTHER axis's marker is a failure,
not a pass -- otherwise this file would credit T1-14 for a refusal that was
really about offload.

WHAT THE HONOUR ARM BUYS
------------------------
Each axis has an in-vocabulary value the plane does execute (`--ep 1`,
`--sharding-strategy ddp`, `--cpu-optimizer-offload false`). The honour arm
declares THE SAME FLAG with that value and must train to completion. Without
it, "the plane refuses whenever this flag appears at all" would remain a live
explanation, and the row would be evidence about a flag rather than about a
value. This is the control T1-3 could not have, because nvfp4 has no
honourable sibling on the same flag that the loader accepts.

VERDICT ORDER -- load-bearing, and fixed before the data was read
-----------------------------------------------------------------
  1. a required arm is missing                 -> UNMEASURED 95
  2. any scored arm died in the parser         -> UNMEASURED 95 (reached no
                                                   plane, so its 96 is noise)
  3. baseline did not train / wrote nothing    -> REFUSE 96 (fixtures dead;
                                                   every emptiness below would
                                                   be untested rather than
                                                   measured)
  4. baseline + absent model was not fatal     -> REFUSE 96 (arm 9 vacuous)
  5. honour arm did not train                  -> REFUSE 96 (the flag, not the
                                                   value, may be the trigger)
  6. refuse arm trained                        -> RED 5  (the silent fallback)
  7. refuse arm wrote weights                  -> RED 5  (refused on paper)
  8. refuse arm refused without recording it
     in the manifest                           -> RED 5  (a refusal nobody
                                                   downstream can see)
  9. refuse arm did not exit 96                -> REFUSE 96 (unclassifiable)
 10. refuse arm carries another axis's marker  -> REFUSE 96 (wrong site spoke)
 11. refuse arm lacks its own marker or does
     not echo the declared value               -> REFUSE 96 (unattributable)
 12. refuse + absent model did not refuse at
     the same marker                           -> REFUSE 96 (ordering unproven)
 13. everything above clean                    -> GREEN 0

Pure: no I/O, no torch, no datasets. `--self-test` drives it over synthetic
payloads under `python3 -S`.
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import Any

GREEN = 0
RED = 5
UNMEASURED = 95
REFUSE = 96

AXES = ("offload", "sharding", "ep")
ROW_OF = {"offload": "T1-13", "sharding": "T1-14", "ep": "T1-17"}

BASELINE = "baseline_real"
FATALITY = "baseline_absent_model"


def _arms_for(axis: str) -> tuple[str, str, str]:
    return (f"{axis}_honour_real", f"{axis}_refuse_real", f"{axis}_refuse_absent")


def axis_refusal_verdict(payload: dict[str, Any], axis: str) -> tuple[int, str]:
    """Return (exit_code, one-line reason) for one axis. Pure."""
    if axis not in AXES:
        return REFUSE, f"unknown axis {axis!r}; not one of {AXES}; {REFUSE}"
    row = ROW_OF[axis]
    arms = payload.get("arms", payload)
    honour, refuse, refuse_absent = _arms_for(axis)
    required = (BASELINE, FATALITY, honour, refuse, refuse_absent)

    missing = [name for name in required if name not in arms]
    if missing:
        return UNMEASURED, f"{row}: arms {sorted(missing)} were never run; {UNMEASURED}"

    scored = {name: arms[name] for name in required}

    parser_dead = sorted(n for n, a in scored.items() if a.get("parser_usage_error"))
    if parser_dead:
        return (
            UNMEASURED,
            f"{row}: {parser_dead} died in the argument parser, so they reached no plane and "
            f"their exit codes say nothing about refusal; {UNMEASURED}",
        )

    base = scored[BASELINE]
    if not base.get("trained") or not base.get("wrote_checkpoint"):
        return (
            REFUSE,
            f"{row}: the no-axis baseline did not train to completion and write weights "
            f"(trained={base.get('trained')}, artifacts={base.get('checkpoint_artifacts')}), so "
            f"the fixtures cannot support a differential and every empty arm below would be an "
            f"untested emptiness rather than a measured one; {REFUSE}",
        )

    fatal = scored[FATALITY]
    if fatal.get("rc") != RED or fatal.get("trained"):
        return (
            REFUSE,
            f"{row}: an unresolvable model was not otherwise fatal (rc={fatal.get('rc')}), so the "
            f"absent-model arm could not show that the refusal PRECEDES model resolution; "
            f"{REFUSE}",
        )

    hon = scored[honour]
    if hon.get("rc") != GREEN or not hon.get("trained") or not hon.get("wrote_checkpoint"):
        return (
            REFUSE,
            f"{row}: the honourable value on the SAME flag did not train "
            f"(rc={hon.get('rc')}, trained={hon.get('trained')}), so the refusal cannot be "
            f"attributed to the value rather than to the flag's mere presence; {REFUSE}",
        )

    ref = scored[refuse]
    if ref.get("trained"):
        return (
            RED,
            f"{row}: the declared axis was accepted and the run TRAINED anyway "
            f"(rc={ref.get('rc')}) -- the silent fallback this row exists to exclude; {RED}",
        )
    if ref.get("wrote_checkpoint"):
        return (
            RED,
            f"{row}: the run refused but left {ref.get('checkpoint_artifacts')} checkpoint "
            f"artifact(s) on disk, so weights outlived the refusal; {RED}",
        )
    if not ref.get("manifest_refused_stage"):
        return (
            RED,
            f"{row}: the run exited {ref.get('rc')} but its manifest records no refused stage, so "
            f"nothing downstream reading artifacts could tell a refusal from a run that never "
            f"happened; {RED}",
        )
    if ref.get("rc") != REFUSE:
        return (
            REFUSE,
            f"{row}: the deciding arm neither trained nor refused (rc={ref.get('rc')}); "
            f"unclassifiable; {REFUSE}",
        )

    foreign = sorted(a for a in AXES if a != axis and ref.get(f"marker_{a}"))
    if foreign:
        return (
            REFUSE,
            f"{row}: the deciding arm refused at {foreign}'s site, not this axis's -- the plane "
            f"holds four refusal sites returning the same code and the wrong one spoke; {REFUSE}",
        )
    if not ref.get(f"marker_{axis}"):
        return (
            REFUSE,
            f"{row}: exit 96 with no {axis} refusal text, so which of the plane's six routes to "
            f"96 was taken is unknown; {REFUSE}",
        )
    if not ref.get(f"echo_{axis}"):
        return (
            REFUSE,
            f"{row}: the refusal does not echo the declared value, so it cannot be attributed to "
            f"the declaration that caused it; {REFUSE}",
        )

    order = scored[refuse_absent]
    if order.get("rc") != REFUSE or not order.get(f"marker_{axis}"):
        return (
            REFUSE,
            f"{row}: with an unresolvable model the same declaration gave rc={order.get('rc')} "
            f"marker={order.get(f'marker_{axis}')}, so the refusal is not shown to precede model "
            f"resolution; {REFUSE}",
        )

    return (
        GREEN,
        f"{row}: the declared {axis} axis refuses (96) at its own site, echoing the declared "
        f"value, before the model resolves, training nothing and writing no weights -- while the "
        f"same flag at its honourable value trains to completion and writes {hon.get('checkpoint_artifacts')} "
        f"artifacts on identical fixtures; {GREEN}",
    )


# --------------------------------------------------------------------------
# controls
# --------------------------------------------------------------------------
def _good_arm(**over: Any) -> dict[str, Any]:
    arm: dict[str, Any] = {
        "rc": 0,
        "parser_usage_error": False,
        "trained": True,
        "wrote_checkpoint": True,
        "checkpoint_artifacts": 7,
        "manifest_refused_stage": False,
    }
    for a in AXES:
        arm[f"marker_{a}"] = False
        arm[f"echo_{a}"] = False
    arm.update(over)
    return arm


def _refusing_arm(axis: str, **over: Any) -> dict[str, Any]:
    arm = _good_arm(
        rc=REFUSE,
        trained=False,
        wrote_checkpoint=False,
        checkpoint_artifacts=0,
        manifest_refused_stage=True,
    )
    arm[f"marker_{axis}"] = True
    arm[f"echo_{axis}"] = True
    arm.update(over)
    return arm


def _payload(axis: str, **over: dict[str, Any]) -> dict[str, Any]:
    honour, refuse, refuse_absent = _arms_for(axis)
    arms = {
        BASELINE: _good_arm(),
        FATALITY: _good_arm(rc=RED, trained=False, wrote_checkpoint=False, checkpoint_artifacts=0),
        honour: _good_arm(),
        refuse: _refusing_arm(axis),
        refuse_absent: _refusing_arm(axis),
    }
    for name, patch in over.items():
        if patch is None:
            arms.pop(name, None)
        else:
            arms[name] = {**arms[name], **patch}
    return {"arms": arms}


def _controls() -> list[tuple[str, str, int, dict[str, Any], str]]:
    """Each control carries the axis it is to be scored under.

    Deriving the axis from the control's NAME would make the suite depend on a
    string split, and a renamed control would then be scored under the wrong
    axis while still reporting PASS.
    """
    out: list[tuple[str, str, int, dict[str, Any], str]] = []
    for axis in AXES:
        honour, refuse, refuse_absent = _arms_for(axis)
        out += [
            (
                f"{axis}: the measured shape",
                axis,
                GREEN,
                _payload(axis),
                "what the tray actually produced",
            ),
            (
                f"{axis}: silent fallback trains",
                axis,
                RED,
                _payload(axis, **{refuse: {"rc": 0, "trained": True, "wrote_checkpoint": True}}),
                "the defect the row exists to exclude",
            ),
            (
                f"{axis}: refused but wrote weights",
                axis,
                RED,
                _payload(axis, **{refuse: {"wrote_checkpoint": True, "checkpoint_artifacts": 3}}),
                "a refusal weights outlived",
            ),
            (
                f"{axis}: refusal absent from the manifest",
                axis,
                RED,
                _payload(axis, **{refuse: {"manifest_refused_stage": False}}),
                "invisible to anything reading artifacts",
            ),
            (
                f"{axis}: another axis's site spoke",
                axis,
                REFUSE,
                _payload(
                    axis,
                    **{refuse: {f"marker_{[a for a in AXES if a != axis][0]}": True}},
                ),
                "four sites share one exit code",
            ),
            (
                f"{axis}: 96 with no marker at all",
                axis,
                REFUSE,
                _payload(axis, **{refuse: {f"marker_{axis}": False}}),
                "an unattributable 96 is not a measurement",
            ),
            (
                f"{axis}: marker without the value echo",
                axis,
                REFUSE,
                _payload(axis, **{refuse: {f"echo_{axis}": False}}),
                "refused, but not demonstrably at this declaration",
            ),
            (
                f"{axis}: honour arm does not train",
                axis,
                REFUSE,
                _payload(axis, **{honour: {"rc": REFUSE, "trained": False}}),
                "then the flag, not the value, may be the trigger",
            ),
            (
                f"{axis}: ordering arm resolves instead of refusing",
                axis,
                REFUSE,
                _payload(axis, **{refuse_absent: {"rc": RED, f"marker_{axis}": False}}),
                "refusal not shown to precede model load",
            ),
            (
                f"{axis}: a required arm was never run",
                axis,
                UNMEASURED,
                _payload(axis, **{refuse: None}),
                "absence is not a verdict",
            ),
            (
                f"{axis}: an arm died in the parser",
                axis,
                UNMEASURED,
                _payload(axis, **{refuse: {"parser_usage_error": True}}),
                "a usage error is also 96 on this CLI",
            ),
        ]
    axis = AXES[0]
    out += [
        (
            "baseline did not train",
            axis,
            REFUSE,
            _payload(axis, **{BASELINE: {"trained": False}}),
            "dead fixtures make every empty arm untested",
        ),
        (
            "baseline wrote no weights",
            axis,
            REFUSE,
            _payload(axis, **{BASELINE: {"wrote_checkpoint": False, "checkpoint_artifacts": 0}}),
            "the checkpoint detector was never shown able to fire",
        ),
        (
            "absent model is not otherwise fatal",
            axis,
            REFUSE,
            _payload(axis, **{FATALITY: {"rc": 0, "trained": True}}),
            "then the ordering arm proves nothing",
        ),
        (
            "an axis this file does not know",
            "not-an-axis",
            REFUSE,
            _payload(axis),
            "a typo must not silently score a well-formed payload",
        ),
        (
            "cross-axis payload: offload arms scored as sharding",
            "sharding",
            UNMEASURED,
            _payload("offload"),
            "the sharding arms are simply absent; that is 95, not a pass",
        ),
    ]
    return out


def _self_test() -> int:
    controls = _controls()
    failures = 0
    for name, axis, want, payload, why in controls:
        got, reason = axis_refusal_verdict(payload, axis)
        ok = got == want
        failures += not ok
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: want {want} got {got} -- {why}")
        if not ok:
            print(f"         reason: {reason}")
    exercised = {axis_refusal_verdict(p, a)[0] for _, a, _, p, _ in controls}
    if exercised != {GREEN, RED, UNMEASURED, REFUSE}:
        print(f"FAIL: controls exercise only {sorted(exercised)}; all four verdicts must appear")
        failures += 1
    print(
        f"{'CLEAR' if not failures else 'RED'}: {len(controls) - failures} of {len(controls)} controls"
    )
    return GREEN if not failures else RED


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--payload")
    ap.add_argument("--axis", choices=AXES)
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args(argv)

    if args.self_test:
        return _self_test()
    if not args.payload or not args.axis:
        print("UNMEASURED: --payload and --axis are required (or --self-test); 95", file=sys.stderr)
        return UNMEASURED
    try:
        with open(args.payload, encoding="utf-8") as fh:
            payload = json.load(fh)
    except OSError as exc:
        print(f"UNMEASURED: --payload {args.payload!r} is unreadable ({exc}); 95", file=sys.stderr)
        return UNMEASURED
    except ValueError as exc:
        print(f"REFUSE: --payload {args.payload!r} is not valid JSON ({exc}); 96", file=sys.stderr)
        return REFUSE

    code, reason = axis_refusal_verdict(payload, args.axis)
    print(reason)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
