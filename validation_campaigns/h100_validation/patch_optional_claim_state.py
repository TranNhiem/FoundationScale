#!/usr/bin/env python3
"""fs207: separate "the operator did not ask" from "the framework could not measure".

THE DEFECT, AS MEASURED (job 37369, world=7, one 8-GPU H100 node, PROBE=1)
--------------------------------------------------------------------------
The run did all of its work and reported itself unmeasured.

    phases_completed   6 of 6 observable phases      status measured
    eval examples      8 of 8 held-out examples      status measured
    eval padding_trips 6 of 14 padding trips         status measured
    resume_proof       PROVED (restore_delta 0.0 <= restore tolerance 0.0005)
    RUN_SUMMARY verdict                              UNMEASURED
    rank 0 exit 3 -> launcher mapped 95 -> sacct FAILED 95:0

Exactly one metric sank it: `run.fixed_eval_rank_invariance`, whose status was
`unmeasured` because `--rank-agreement-tolerance` was not passed. Every cross-rank
spread it reports was 0.0 -- before save, after resume, and the delta.

`_resume_continuity_verdict` (the fs192 stage) already draws the distinction that
matters. It sets `rank_agreement_absolute` to True when an explicit tolerance was
declared and both spreads come in under it, to False when one was declared and they
do not, and leaves it None when none was declared -- its own comment says "the
question was not asked". One layer up, the ledger entry collapses all three into a
two-valued `"measured" if continuity["rank_invariant"] else "unmeasured"`, and
`MeasurementLedger._walk` flags `unmeasured`, so the run verdict sinks.

Three separate things go wrong with that collapse:

1.  A claim the operator never requested is reported as one the framework could not
    measure. Those are different states. The framework knows exactly how to decide
    this claim; the only missing input is an operator declaration. Contrast the
    genuinely unmeasured claims in this same report -- MoE expert-FQN handling has no
    MoE to measure and no flag can conjure one.

2.  Because `--rank-agreement-tolerance` is absent from the launcher and from both
    operator env files, the default configuration reaches that state on every run.
    The submit chain's first link is `--dependency=afterok:$probe_jid`, so a probe
    that exits 95 never releases production: post-fs192, the documented default
    configuration cannot advance past its own first link. fs192 shipped with the
    residual "no cluster job has yet run it"; this is what running it found.

3.  The else arm's prose is false whenever the spreads are zero. On 37369 it read
    "the fixed-eval loss takes distinct bit-identical values across ranks (spread
    before save 0.0, spread after resume 0.0, spread delta 0.0)". Spread 0.0 is
    agreement. The arm serves both "not asked" and "asked and refuted" and describes
    only the second, so it asserts a measured negative the data on the same line
    refutes. A claim mismatched to its evidence is a defect even when the code that
    produced the number is correct.

WHAT THIS STAGE CHANGES
-----------------------
`not_requested` becomes a first-class ledger state, distinct from `unmeasured`:

  * `MeasurementLedger` grows a `not_requested` list beside `unmeasured`.
  * `_walk` routes `status == "not_requested"` into it and does NOT sink the verdict --
    but keeps recursing, unlike the flagged branch. The flagged branch stops because
    the whole subtree is already condemned; stopping here would let an `unmeasured`
    child hide under a `not_requested` parent, which is the laundering direction.
  * Both payload writers -- the per-phase `PHASE_JSON` and the `RUN_SUMMARY_JSON` --
    emit the list whenever it is non-empty. A state that is recorded and never printed
    is invisible, and an invisible abstention is how a denominator shrinks silently.
  * `fixed_eval_rank_invariance` keys its status on `rank_agreement_absolute`, the
    three-valued field fs192 already computes, instead of on the two-valued
    `rank_invariant`:
        True  -> "measured"      the claim was requested and holds
        False -> "unmeasured"    requested and REFUTED; still sinks the verdict, and
                                 now says REFUTED rather than describing agreement
                                 it did not observe
        None  -> "not_requested" naming `--rank-agreement-tolerance` as the flag that
                                 would request it, and reporting the three spreads
                                 neutrally

REJECTED REMEDIES
-----------------
* Pass `--rank-agreement-tolerance` in the env files and call it fixed. That makes
  this estate's two arms green and leaves every other operator on the same cliff; it
  also invites picking the tolerance from the spread you already measured, which is
  the self-calibration fs192 exists to forbid.
* Change the chain's first dependency to `afterany`. That would let a genuinely
  broken probe release production, which deletes the probe's only purpose.
* Fold `rank_agreement_absolute is False` into "measured". A refuted claim must not
  produce a green run. It stays in the sinking set; only its label is corrected.

DECLARED RESIDUALS
------------------
* The `False` arm is certified at build time on synthetic input (C3) and is
  UNMEASURED on hardware: no run has yet declared an explicit tolerance, and after
  fs202 the Gemma spread is 0.0, so refuting it would need a deliberately divergent
  tree rather than an ordinary run.
* This stage does not document the flag. `--rank-agreement-tolerance` reaches the
  trainer through `FS_ENGINE_LAUNCH_CMD`, which is a free-form operator knob, so the
  path exists; that it is named in no operator-facing document is the fs170/fs185
  class and is recorded there, not patched here.
"""

from __future__ import annotations

import ast
import pathlib
import sys
from typing import Any

TARGET = pathlib.Path(__file__).resolve().parent / "h100" / "gen" / "fs_train.fixed.py"
MARK = "fs207:"
OPEN_MARK = "# --- fs207: an optional claim the operator did not request is not an"
N_CONTROLS = 8

# ---------------------------------------------------------------------------
# 1. the ledger gains a third state
# ---------------------------------------------------------------------------

OLD_LEDGER = '''class MeasurementLedger:
    """Accumulate unmeasured facts so the outcome degrades fail-closed."""

    def __init__(self) -> None:
        self.unmeasured: list[str] = []

    def check(self, namespace: str, payload: Mapping[str, Any]) -> None:
        for key, value in payload.items():
            self._walk(f"{namespace}.{key}", value)

    def _walk(self, path: str, value: Any) -> None:
        if isinstance(value, Mapping):
            if value.get("status") in {"unmeasured", "invalid_denominator", "UNVERIFIED"}:
                self.unmeasured.append(path)
            else:
                for key, child in value.items():
                    self._walk(f"{path}.{key}", child)
'''

NEW_LEDGER = '''class MeasurementLedger:
    """Accumulate unmeasured facts so the outcome degrades fail-closed."""

    # --- fs207: an optional claim the operator did not request is not an unmeasured one.
    # "unmeasured" means this machine could not decide the claim: no oracle, no data, a
    # detector that could not fire. "not_requested" means it could have, and was not asked
    # -- the only missing input is an operator declaration. Folding the second into the
    # first made every default-configured run report UNMEASURED while all six of its phases
    # measured (job 37369), and through the chain's afterok first link that made the
    # documented default unable to advance past its own probe.
    def __init__(self) -> None:
        self.unmeasured: list[str] = []
        self.not_requested: list[str] = []

    def check(self, namespace: str, payload: Mapping[str, Any]) -> None:
        for key, value in payload.items():
            self._walk(f"{namespace}.{key}", value)

    def _walk(self, path: str, value: Any) -> None:
        if isinstance(value, Mapping):
            if value.get("status") in {"unmeasured", "invalid_denominator", "UNVERIFIED"}:
                self.unmeasured.append(path)
                return
            if value.get("status") == "not_requested":
                # fs207: recorded, and the verdict is NOT sunk -- but recursion CONTINUES.
                # The branch above stops because a condemned subtree is already condemned;
                # stopping here would let an unmeasured child hide under a not_requested
                # parent, which is the laundering direction this state could be abused in.
                self.not_requested.append(path)
            for key, child in value.items():
                self._walk(f"{path}.{key}", child)
'''

# ---------------------------------------------------------------------------
# 2. the rank-invariance entry keys on the three-valued field, not the two-valued one
# ---------------------------------------------------------------------------

OLD_ENTRY = '''        "fixed_eval_rank_invariance": {
            "status": "measured" if continuity["rank_invariant"] else "unmeasured",'''

NEW_ENTRY = '''        "fixed_eval_rank_invariance": {
            # fs207: three states, because fs192 already computed three. `rank_invariant`
            # is True only on the certified arm, so `else` served both "refuted" and
            # "never asked" and reported both as unmeasured.
            "status": (
                "measured"
                if continuity["rank_agreement_absolute"] is True
                else "unmeasured"
                if continuity["rank_agreement_absolute"] is False
                else "not_requested"
            ),'''

OLD_DISPLAY = '''            "display": (
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
            ),'''

NEW_DISPLAY = '''            "display": (
                # fs207: the old else arm asserted "takes distinct bit-identical values
                # across ranks" on the same line as three spreads of 0.0. It described the
                # refuted case and was reached by the never-asked case as well.
                "fixed-eval rank agreement MEASURED within the explicit "
                "rank-agreement tolerance "
                f"{continuity['rank_agreement_tolerance']:.8g}"
                if continuity["rank_agreement_absolute"] is True
                else "fixed-eval rank agreement REFUTED under the explicit "
                "rank-agreement tolerance "
                f"{continuity['rank_agreement_tolerance']:.8g}: spread before "
                f"save {continuity['cross_rank_spread_before_save']}, spread after "
                f"resume {continuity['cross_rank_spread_after_resume']}, spread "
                f"delta {continuity['cross_rank_spread_delta']}; the restore "
                "verdict stands on its own per-rank terms against the restore "
                f"tolerance {tolerance:.8g}"
                if continuity["rank_agreement_absolute"] is False
                else "fixed-eval rank agreement NOT REQUESTED: no rank-agreement "
                "tolerance was declared, so the absolute cross-rank claim was not "
                "put to this run. Pass --rank-agreement-tolerance to request it. "
                f"Observed spread before save {continuity['cross_rank_spread_before_save']}, "
                f"after resume {continuity['cross_rank_spread_after_resume']}, "
                f"delta {continuity['cross_rank_spread_delta']}; the restore verdict "
                "stands on its own per-rank terms against the restore tolerance "
                f"{tolerance:.8g}"
            ),'''

# ---------------------------------------------------------------------------
# 3. both payload writers print the new state
# ---------------------------------------------------------------------------

OLD_PHASE_EMIT = '''    if ledger.unmeasured or status != "measured":
        payload["verdict"] = "UNMEASURED"
        payload["unmeasured"] = sorted(ledger.unmeasured)
    else:
        payload["verdict"] = "MEASURED"'''

NEW_PHASE_EMIT = '''    if ledger.unmeasured or status != "measured":
        payload["verdict"] = "UNMEASURED"
        payload["unmeasured"] = sorted(ledger.unmeasured)
    else:
        payload["verdict"] = "MEASURED"
    if ledger.not_requested:
        # fs207: printed whenever non-empty, on BOTH verdicts. An abstention that is
        # recorded and never shown is a denominator that shrank without saying so.
        payload["not_requested"] = sorted(set(ledger.not_requested))'''

OLD_RUN_EMIT = '''    ledger.check("run", summary_metrics)
    verdict = "UNMEASURED" if ledger.unmeasured else "MEASURED"
    _print_json(
        "RUN_SUMMARY_JSON",
        {
            "verdict": verdict,
            "unmeasured": sorted(set(ledger.unmeasured)),'''

NEW_RUN_EMIT = '''    ledger.check("run", summary_metrics)
    # fs207: only `unmeasured` sinks the run. `not_requested` is reported beside it.
    verdict = "UNMEASURED" if ledger.unmeasured else "MEASURED"
    _print_json(
        "RUN_SUMMARY_JSON",
        {
            "verdict": verdict,
            "unmeasured": sorted(set(ledger.unmeasured)),
            "not_requested": sorted(set(ledger.not_requested)),'''

REPLACEMENTS = (
    ("ledger", OLD_LEDGER, NEW_LEDGER),
    ("entry", OLD_ENTRY, NEW_ENTRY),
    ("display", OLD_DISPLAY, NEW_DISPLAY),
    ("phase_emit", OLD_PHASE_EMIT, NEW_PHASE_EMIT),
    ("run_emit", OLD_RUN_EMIT, NEW_RUN_EMIT),
)


def _stderr(msg: str) -> None:
    print(msg, file=sys.stderr)


def _transform(text: str) -> tuple[str, bool]:
    """Apply the five ordered replacements. Idempotent: a marked file is returned as-is."""
    if OPEN_MARK in text:
        return text, False
    out = text
    for _name, old, new in REPLACEMENTS:
        if out.count(old) != 1:
            return text, False
        out = out.replace(old, new, 1)
    return out, out != text


# ---------------------------------------------------------------------------
# controls
# ---------------------------------------------------------------------------

_HARNESS = '''
class _M(dict):
    pass


def _ledger_from(mod, payload):
    led = mod.MeasurementLedger()
    led.check("run", payload)
    return led
'''


def _load(source: str, name: str) -> Any:
    """Compile the trainer text into a module object without importing torch.

    The trainer guards its heavy imports, but exec'ing 4,700 lines to reach one class is
    both slow and an environment dependency this stage must not acquire. Only the two
    definitions the controls exercise are lifted, by ast surgery, into a fresh module.
    """
    import types

    tree = ast.parse(source)
    wanted = {"MeasurementLedger"}
    body: list[ast.stmt] = []
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name in wanted:
            body.append(node)
    if len(body) != len(wanted):
        raise RuntimeError(f"{name}: lifted {len(body)} of {len(wanted)} definitions")
    prelude = ast.parse(
        "from collections.abc import Mapping\nfrom typing import Any\n"
    ).body
    module_ast = ast.Module(body=prelude + body, type_ignores=[])
    ast.fix_missing_locations(module_ast)
    mod = types.ModuleType(name)
    exec(compile(module_ast, f"<{name}>", "exec"), mod.__dict__)  # noqa: S102
    return mod


def _entry_status(source: str, absolute: Any) -> tuple[str, str]:
    """Evaluate the shipped status/display expressions for one value of the 3-valued field.

    The expressions are lifted verbatim out of the target by ast, so this measures the
    text that ships rather than a paraphrase of it -- the fs150 lesson, applied to a
    conditional expression instead of a path.
    """
    tree = ast.parse(source)
    status_src = display_src = None
    for node in ast.walk(tree):
        if not isinstance(node, ast.Dict):
            continue
        for key, val in zip(node.keys, node.values):
            if not (isinstance(key, ast.Constant) and key.value == "fixed_eval_rank_invariance"):
                continue
            if not isinstance(val, ast.Dict):
                continue
            for k2, v2 in zip(val.keys, val.values):
                if isinstance(k2, ast.Constant) and k2.value == "status":
                    status_src = ast.unparse(v2)
                if isinstance(k2, ast.Constant) and k2.value == "display":
                    display_src = ast.unparse(v2)
    if status_src is None or display_src is None:
        raise RuntimeError("fixed_eval_rank_invariance status/display not found")
    continuity = {
        "rank_agreement_absolute": absolute,
        "rank_invariant": absolute is True,
        "rank_agreement_tolerance": 0.0005,
        "cross_rank_spread_before_save": 0.0,
        "cross_rank_spread_after_resume": 0.0,
        "cross_rank_spread_delta": 0.0,
    }
    env = {"continuity": continuity, "tolerance": 0.0005}
    return str(eval(status_src, {}, env)), str(eval(display_src, {}, env))  # noqa: S307


def _controls(new: str, old: str) -> tuple[int, list[str]]:
    ok = 0
    notes: list[str] = []

    def record(cid: str, kind: str, good: bool, detail: str) -> None:
        nonlocal ok
        ok += int(good)
        notes.append(f"{cid} {kind}: {'PASS' if good else 'FAIL'}  {detail}")

    pre = _load(old, "pre207")
    post = _load(new, "post207")

    # C1 MUST_FIRE -- reproduce 37369 on the PRE-image: the never-asked case is flagged.
    st_pre, disp_pre = _entry_status(old, None)
    led_pre = pre.MeasurementLedger()
    led_pre.check("run", {"fixed_eval_rank_invariance": {"status": st_pre}})
    record(
        "C1", "MUST_FIRE",
        st_pre == "unmeasured" and led_pre.unmeasured == ["run.fixed_eval_rank_invariance"],
        f"pre-image with rank_agreement_absolute=None -> status {st_pre!r}, "
        f"ledger.unmeasured={led_pre.unmeasured} (this is job 37369's single sinking metric)",
    )

    # C2 MUST_PASS -- the same input on the post-image is recorded and does not sink.
    st_post, disp_post = _entry_status(new, None)
    led_post = post.MeasurementLedger()
    led_post.check("run", {"fixed_eval_rank_invariance": {"status": st_post}})
    record(
        "C2", "MUST_PASS",
        st_post == "not_requested"
        and led_post.unmeasured == []
        and led_post.not_requested == ["run.fixed_eval_rank_invariance"],
        f"post-image None -> status {st_post!r}, unmeasured={led_post.unmeasured}, "
        f"not_requested={led_post.not_requested}",
    )

    # C3 MUST_FIRE -- ANTI-LAUNDERING. A refuted claim must still sink the run.
    st_ref, disp_ref = _entry_status(new, False)
    led_ref = post.MeasurementLedger()
    led_ref.check("run", {"fixed_eval_rank_invariance": {"status": st_ref}})
    record(
        "C3", "MUST_FIRE",
        st_ref == "unmeasured"
        and led_ref.unmeasured == ["run.fixed_eval_rank_invariance"]
        and led_ref.not_requested == [],
        f"post-image rank_agreement_absolute=False -> status {st_ref!r}, "
        f"unmeasured={led_ref.unmeasured}; a refuted claim is not laundered into "
        f"not_requested",
    )

    # C4 MUST_PASS -- the certified arm is untouched.
    st_true, disp_true = _entry_status(new, True)
    led_true = post.MeasurementLedger()
    led_true.check("run", {"fixed_eval_rank_invariance": {"status": st_true}})
    record(
        "C4", "MUST_PASS",
        st_true == "measured" and led_true.unmeasured == [] and led_true.not_requested == [],
        f"post-image rank_agreement_absolute=True -> status {st_true!r}, both lists empty",
    )

    # C5 MUST_FIRE -- ANTI-LAUNDERING. Recursion must not stop at a not_requested parent.
    nested = {
        "outer": {
            "status": "not_requested",
            "child": {"status": "unmeasured"},
        }
    }
    led_nest = post.MeasurementLedger()
    led_nest.check("run", nested)
    record(
        "C5", "MUST_FIRE",
        led_nest.unmeasured == ["run.outer.child"]
        and led_nest.not_requested == ["run.outer"],
        f"an unmeasured child under a not_requested parent still sinks: "
        f"unmeasured={led_nest.unmeasured}, not_requested={led_nest.not_requested}",
    )

    # C6 MUST_PASS -- the never-asked prose describes what was observed and names the flag.
    lowered = disp_post.lower()
    record(
        "C6", "MUST_PASS",
        "--rank-agreement-tolerance" in disp_post
        and "distinct" not in lowered
        and "do not agree" not in lowered
        and "not requested" in lowered,
        "never-asked display names the requesting flag and asserts no disagreement: "
        + disp_post[:96].replace("\\n", " "),
    )

    # C7 MUST_FIRE -- C6 measures a real change: the PRE-image prose fails it.
    record(
        "C7", "MUST_FIRE",
        "distinct" in disp_pre.lower() and "--rank-agreement-tolerance" not in disp_pre,
        "pre-image display for the SAME zero spreads claims distinct values: "
        + disp_pre[:96].replace("\\n", " "),
    )

    # C8 MUST_PASS -- the state is printed, not merely recorded, by BOTH writers.
    emit_sites = new.count('"not_requested"] = sorted(set(ledger.not_requested))') + new.count(
        '"not_requested": sorted(set(ledger.not_requested)),'
    )
    record(
        "C8", "MUST_PASS",
        emit_sites == 2,
        f"{emit_sites} of 2 payload writers emit the list (PHASE_JSON and RUN_SUMMARY_JSON); "
        f"a recorded-but-unprinted abstention is an invisible denominator",
    )

    return ok, notes


def main() -> int:
    apply = "--check" not in sys.argv
    if not TARGET.exists():
        _stderr(f"REFUSE 96: {TARGET} does not exist")
        return 96
    text = TARGET.read_text("utf-8")
    new, changed = _transform(text)

    gres: list[tuple[str, bool, str]] = []
    gres.append(("G1", MARK in new, "the fs207 mark is present in the patched text"))
    gres.append(
        ("G2", new.count("self.not_requested: list[str] = []") == 1,
         "the ledger declares exactly one not_requested list")
    )
    gres.append(
        ("G3", new.count('value.get("status") == "not_requested"') == 1,
         "_walk routes the new state at exactly one site")
    )
    gres.append(
        ("G4", '"not_requested"' not in text,
         "the pre-image knows nothing of the state -- this stage introduces it")
    )

    # G5: the sinking set is UNCHANGED. This stage may only add a state, never remove one.
    sink = '{"unmeasured", "invalid_denominator", "UNVERIFIED"}'
    gres.append(
        ("G5", new.count(sink) == 1 and text.count(sink) == 1,
         "the set of statuses that sink the verdict is byte-identical to the pre-image")
    )

    # G6: the entry keys on the three-valued field and no longer on the two-valued one.
    #     Scoped to the entry's own source, not the file: `rank_invariant` is a legitimate
    #     name elsewhere, and a file-wide count would measure a denominator wider than the
    #     claim -- the fs206 lesson.
    entry_ok = False
    try:
        st_none, _ = _entry_status(new, None)
        st_false, _ = _entry_status(new, False)
        st_true, _ = _entry_status(new, True)
        entry_ok = (st_none, st_false, st_true) == ("not_requested", "unmeasured", "measured")
    except Exception as exc:  # noqa: BLE001
        _stderr(f"G6 could not evaluate the shipped expression: {exc}")
    gres.append(
        ("G6", entry_ok,
         "the SHIPPED status expression maps None/False/True to "
         "not_requested/unmeasured/measured")
    )

    gres.append(
        ("G7", new.count('payload["not_requested"] = sorted(set(ledger.not_requested))') == 1
         and new.count('"not_requested": sorted(set(ledger.not_requested)),') == 1,
         "both payload writers emit the list, once each")
    )

    compiled = True
    try:
        compile(new, str(TARGET), "exec")
    except SyntaxError as exc:
        compiled = False
        _stderr(f"patched text does not compile: {exc}")
    gres.append(("G8", compiled, "the patched module compiles"))

    again, changed_again = _transform(new)
    gres.append(("G9", again == new and not changed_again,
                 "byte-idempotence on own output (a second run is a byte-exact no-op)"))
    gres.append(("G10", changed or OPEN_MARK in text,
                 "the transform either changed the file or the mark was already present"))

    gates = 0
    for name, good, detail in gres:
        print(f"{name}: {'PASS' if good else 'FAIL'}  {detail}")
        gates += int(good)
    try:
        cok, cnotes = _controls(new, text)
    except Exception as exc:  # noqa: BLE001
        cok, cnotes = 0, [f"controls raised {type(exc).__name__}: {exc}"]
    for n in cnotes:
        print("control " + n)

    if gates != len(gres) or cok != N_CONTROLS:
        _stderr(f"\nREFUSE 96: static gates {gates}/{len(gres)}, controls {cok}/{N_CONTROLS}; writing nothing")
        return 96
    if not apply:
        print(f"verdict: READY  {gates}/{len(gres)} static gates, {cok}/{N_CONTROLS} controls")
        return 0
    TARGET.write_text(new, "utf-8")
    print(f"{MARK} {gates}/{len(gres)} static gates, {cok}/{N_CONTROLS} controls")
    return 0


def _guarded() -> int:
    # A stage that makes a red run green must not be able to fail into silence: an
    # unhandled exception is a REFUSE, not a bare rc=1.
    try:
        return main()
    except Exception as exc:  # noqa: BLE001 - deliberate: no traceback may escape as rc=1
        _stderr(f"REFUSE 96: {MARK} stage raised {type(exc).__name__}: {exc}")
        return 96


if __name__ == "__main__":
    sys.exit(_guarded())
