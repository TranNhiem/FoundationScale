#!/usr/bin/env python3
"""#417 boundary control: an ENVIRONMENT fault must not be adjudicated as RED.

WHAT IS CLAIMED
  For EVERY shipped row with a main() -- t1_2, t1_4, t1_6, t1_9, t1_10, t1_11,
  t1_12, t1_21, t1_23 -- per row four dynamic legs (not text tripwires, #101):
    ENV           : _run raises a dyld/CUDA linker fault -> MUST return 95 (UNMEASURED)
    HARN          : _run raises a harness bug            -> MUST return 96 (CANNOT_MEASURE)
    PASSTHRU-RED  : _run RETURNS 5  (a reached verdict)  -> MUST return 5
    PASSTHRU-GREEN: _run RETURNS 0  (a reached verdict)  -> MUST return 0
  The PASSTHRU legs are the MUST_NOT_FIRE direction: the over-narrowing risk of
  #417's fix is that a genuine RED -- an actual refutation the row measured --
  gets laundered into an abstention. A handler that only ever abstains would
  pass the ENV/HARN legs and still be wrong.

WHAT IS NOT CLAIMED
  This control does not claim any row's science is right. It does not claim
  that 95/96 abstentions are well-calibrated in general, only that these two
  injected fault classes land on the correct side of the boundary. It does not
  claim the rows run end-to-end on this machine; _run is always patched out.
  Above all: a passing control proves nothing until the control has been shown
  able to fail ([[reference_positive_control_must_selfmatch]]). That is why
  --self-test is the PRIMARY entry point: it rebuilds a synthetic PRE-fix
  subject (everything -> 5 RED) plus a synthetic POST-fix subject, runs this
  control's own leg logic against both, and only then is a CLEAR reading on
  the real rows worth anything.

  #463 -- WHY THE DENOMINATOR IS NOW NINE AND THE ENTRY IS NOW WIRED. This
  control used to name four rows, and they were exactly the four #417 had
  already fixed, so it read green by construction and could not catch a
  regression. Worse, --expect postfix was invoked by no make target and no CI
  step, so even that reading was never taken. Both are closed: the denominator
  is every row with a main(), and `make boundary-control` runs the pair. Against
  the tree before #463 the widened control reports 9 failing legs of 36 -- t1_2
  and t1_4 answering 5 to an environment fault, t1_6 and t1_21 escaping main()
  to CPython's 1, t1_23 collapsing environment into harness at 96. Against this
  tree it reports 0 of 36, with every PASSTHRU leg green in both, so the fix did
  not launder a single verdict a row had actually reached.

PRIMARY ENTRY:  python3 control_boundary_classification.py --self-test
DEBUG ENTRY:    python3 control_boundary_classification.py --expect prefix|postfix

Exit contract: 0 CLEAR, 5 RED (a leg disagreed), 95 UNMEASURED,
96 CANNOT-MEASURE/REFUSE. Never 1, never 2 -- argparse's own exits and any
escaped exception are remapped into this contract (an escape is 96, a harness
fault, never 5, which would be a false refutation).
"""

from __future__ import annotations

import argparse
import contextlib
import io
import sys
import tempfile
from importlib.machinery import SourceFileLoader
from importlib.util import module_from_spec, spec_from_loader
from pathlib import Path

# The module lives IN the row directory, as a sibling of the rows it loads.
ROWDIR = Path(__file__).resolve().parent
# The rows import t1_interpreter_floor as a sibling, so they only resolve with
# their own directory on the path -- exactly as `python3 t1_9_...py` gives them.
sys.path.insert(0, str(ROWDIR))

# The exception each leg injects, and the code the boundary must answer with.
ENV_EXC = OSError(
    "dlopen(/usr/local/cuda/lib64/libcudart.so.12, 0x0006): "
    "Library not loaded: @rpath/libcudart.so.12"
)
HARN_EXC = RuntimeError("harness bug: arm table was never populated")

def _outdir(*extra: str):
    """argv builder for a row whose only required argument is --out-dir."""

    def build(td: str) -> list[str]:
        return ["--out-dir", td, *extra]

    return build


def _fixture_checkpoint(td: str) -> Path:
    """A minimal sharded checkpoint, enough to get past a row's existence checks."""
    ckpt = Path(td) / "ckpt"
    ckpt.mkdir(parents=True, exist_ok=True)
    (ckpt / "model.safetensors.index.json").write_text(
        '{"weight_map": {"a": "model-00001-of-00001.safetensors"}}', encoding="utf-8"
    )
    (ckpt / "model-00001-of-00001.safetensors").write_bytes(b"\x00" * 16)
    return ckpt


def _t1_2_argv(td: str) -> list[str]:
    return ["--out", str(Path(td) / "t1_2.json")]


def _t1_4_argv(td: str) -> list[str]:
    return ["--checkpoint", str(_fixture_checkpoint(td)), "--work-dir", str(Path(td) / "work")]


def _t1_6_argv(td: str) -> list[str]:
    # t1_6 takes the checkpoint as a bare positional, not a flag.
    return [str(_fixture_checkpoint(td))]


def _t1_21_argv(td: str) -> list[str]:
    rows = Path(td) / "rows.jsonl"
    rows.write_text('{"text": "x", "image": "y"}\n', encoding="utf-8")
    return [
        "--model", str(_fixture_checkpoint(td)),
        "--image-dataset", str(rows),
        "--text-dataset", str(rows),
        "--out-dir", str(Path(td) / "out"),
        "--vision-target", "vision",
        "--language-target", "language",
    ]


def _t1_23_argv(td: str) -> list[str]:
    return [
        "--model", str(_fixture_checkpoint(td)),
        "--work-dir", str(Path(td) / "work"),
        "--profile-name", "single-node",
    ]


def _payload(rc_in: int) -> dict:
    """A verdict payload shaped for the rows whose main() reads it back."""
    name = {0: "GREEN", 5: "RED"}[rc_in]
    return {
        "row": "control",
        "claim": "control passthrough",
        "verdict": {
            "row": "control",
            "rc": rc_in,
            "name": name,
            "reason": "passthrough leg: a verdict the row REACHED",
            "adjudicated_from_payload_fields_only": True,
        },
    }


# Each row: (file, raise_target, argv_builder, passthru_patches).
#
# WHY THIS IS NOT JUST A FILE LIST. Before this was widened, ROWS named four
# rows -- and those four were exactly the four that had already been fixed under
# #417. A control whose denominator is the set of subjects that already answer
# correctly reads green by construction; it cannot catch a regression and it
# never saw t1_2, t1_4, t1_6, t1_21 or t1_23, all five of which were wrong
# (two adjudicated an environment fault RED, two escaped main() and exited 1,
# one collapsed environment and harness faults into a single 96). The denominator
# is now every row with a main(), which is what the claim was always about.
#
# `raise_target` is the delegate whose raise must reach the boundary.
# `passthru_patches(rc)` returns the (attribute, return-value) stubs that make the
# row REACH the verdict `rc` without raising. It is a list, not one name, because
# the rows that adjudicate a payload need both the measurement and the
# adjudication neutralised -- stub only the adjudication and the leg runs the
# real measurement against a fixture checkpoint, which scores the fixture rather
# than the boundary. One hardcoded `_run` could express neither.
_ONLY_RUN = (lambda rc: [("_run", rc)])
ROWS = {
    "t1_2": ("t1_2_fp16_scaler_arms.py", "run_measurement", _t1_2_argv,
             lambda rc: [("run_measurement", {}), ("verdict", (rc, _payload(rc)))]),
    "t1_4": ("t1_4_save_load_parity.py", "run_measurement", _t1_4_argv,
             lambda rc: [("run_measurement", {}), ("verdict", (rc, _payload(rc)))]),
    "t1_6": ("t1_6_symlinked_shard.py", "_run_arms", _t1_6_argv,
             lambda rc: [("_run_arms", ["one arm disagreed"] if rc == 5 else [])]),
    "t1_9": ("t1_9_optimizer_arms.py", "_run", _outdir(), _ONLY_RUN),
    "t1_10": ("t1_10_accumulation_equivalence.py", "_run", _outdir(), _ONLY_RUN),
    "t1_11": ("t1_11_grad_checkpointing.py", "_run", _outdir(), _ONLY_RUN),
    "t1_12": ("t1_12_attention_impl.py", "_run", _outdir("--profile-name", "single-node"),
              _ONLY_RUN),
    "t1_21": ("t1_21_vision_tower_arms.py", "_measure", _t1_21_argv,
              lambda rc: [("_measure", rc)]),
    "t1_23": ("t1_23_audio_declaration_arms.py", "_run", _t1_23_argv, _ONLY_RUN),
}

# Synthetic subjects for --self-test. STANDALONE BY DESIGN: they must not
# import t1_interpreter_floor or any real row, so the synthetic legs stay
# independent of the module under test.
PREFIX_SUBJECT_SRC = """\
# Synthetic PRE-fix subject: every exception launders to RED (rc 5).
import argparse


def _run(args):
    raise AssertionError("the control patches _run; this body must never execute")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir")
    args, _extra = ap.parse_known_args(argv)
    try:
        return _run(args)
    except Exception:
        # The PRE-fix bug: environment, harness, and genuine faults all say RED.
        print("VERDICT RED: synthetic refutation")
        return 5
"""

POSTFIX_SUBJECT_SRC = """\
# Synthetic POST-fix subject: link fault -> 95, harness fault -> 96,
# and a verdict _run RETURNS is passed through untouched.
import argparse


def _run(args):
    raise AssertionError("the control patches _run; this body must never execute")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir")
    args, _extra = ap.parse_known_args(argv)
    try:
        return _run(args)
    except OSError as exc:
        print(f"VERDICT UNMEASURED: environment/link fault: {exc}")
        return 95
    except Exception as exc:
        print(f"CANNOT-MEASURE: harness fault: {exc}")
        return 96
"""


UNREACHABLE_SUBJECT_SRC = """\
# Synthetic UNREACHABLE subject (#466): returns on an ABSENT PRECONDITION before
# it ever calls the delegate the control patches. It is not wrong -- 95 for a
# missing precondition is exactly right -- so the control must neither pass nor
# fail its legs. This is the shape t1_2 takes on a runner with no transformers.
import argparse


def _run(args):
    raise AssertionError("the control patches _run; this body must never execute")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir")
    args, _extra = ap.parse_known_args(argv)
    print("VERDICT UNMEASURED: a precondition for the measurement is ABSENT here")
    return 95
    return _run(args)  # unreachable on purpose
"""


class HarnessFault(Exception):
    """The control itself could not run a leg -- exit 96, never 5."""


def _why(stdout: str) -> str:
    """The subject's own closing line, so an abstention names its cause.

    An abstention that says only "unreachable" is a subtraction from the
    denominator that nobody can audit. Quoting the row's own last verdict line
    makes the reason attributable to the subject rather than to this control.
    """
    lines = [ln.strip() for ln in stdout.splitlines() if ln.strip()]
    for ln in reversed(lines):
        if ln.startswith(("VERDICT ", "CANNOT-MEASURE")):
            return f"The row said: {ln[:200]}"
    return f"The row said: {lines[-1][:200]}" if lines else "The row printed nothing."


def load(path: Path):
    name = f"fs417ctl_{path.stem}"
    loader = SourceFileLoader(name, str(path))
    spec = spec_from_loader(name, loader)
    mod = module_from_spec(spec)
    sys.modules[name] = mod
    loader.exec_module(mod)
    return mod


def run_leg(mod, target: str, argv: list[str], exc: BaseException) -> tuple[int, str, bool]:
    """Patch `target` to raise `exc`, call main(argv); return (rc, stdout, reached).

    The signature is ``*_a, **_k`` because the delegates differ across rows --
    ``_run(args)`` in most, ``run_measurement(device, requested)`` in t1_2, and
    ``_run_arms(real, out)`` in t1_6. A one-argument stub would raise TypeError
    before the injected exception ever reached the boundary, which would score
    the row on the probe's own fault.

    `reached` is #466. A row may return on an ABSENT PRECONDITION before it ever
    calls the delegate -- t1_2 answers 95 at `import transformers` -- and then
    the injected exception never happened, so the leg measured staging rather
    than the boundary. Measured, not assumed: the stub records its own call, the
    same self-match discipline the f381 plant uses. Without it the ENV leg
    PASSES vacuously (the row's precondition 95 is coincidentally the wanted 95)
    while HARN and both PASSTHRU legs FAIL for a defect that is not there.
    """
    reached = False

    def boom(*_a, **_k):
        nonlocal reached
        reached = True
        raise exc

    original = getattr(mod, target)
    setattr(mod, target, boom)
    out, err = io.StringIO(), io.StringIO()
    try:
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            try:
                rc = mod.main(argv)
            except SystemExit as se:  # argparse or an un-caught exit
                rc = se.code if isinstance(se.code, int) else 96
            except Exception as esc:  # noqa: BLE001 -- an ESCAPE is the finding, not a fault
                # CPython would print this traceback and exit 1, which is outside
                # the row's declared contract. Report it as 1 so the leg fails.
                print(f"escaped main(): {type(esc).__name__}: {esc}")
                rc = 1
    finally:
        setattr(mod, target, original)
    return rc, out.getvalue() + err.getvalue(), reached


def run_passthrough(
    mod, patches: list[tuple[str, object]], argv: list[str]
) -> tuple[int, str, bool]:
    """The row REACHES a verdict (nothing raises); main() must pass it through.

    `patches` is a list because a row that adjudicates a payload needs two stubs,
    not one: the measurement must be neutralised as well as the adjudication, or
    the leg runs the real measurement against a fixture checkpoint and scores the
    row on the fixture rather than on the boundary.

    Returns (rc, stdout, reached); `reached` is #466, as in run_leg. ANY stub
    being called counts as reached: a row that calls the measurement and then
    drops the verdict on the floor HAS reached the boundary and its disagreement
    is a real finding, not an abstention.
    """
    originals = [(attr, getattr(mod, attr)) for attr, _ in patches]
    reached = False

    def stub_for(v):
        def stub(*_a, **_k):
            nonlocal reached
            reached = True
            return v

        return stub

    for attr, value in patches:
        setattr(mod, attr, stub_for(value))
    out, err = io.StringIO(), io.StringIO()
    try:
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            try:
                rc = mod.main(argv)
            except SystemExit as se:
                rc = se.code if isinstance(se.code, int) else 96
            except Exception:  # noqa: BLE001 -- an escape here is also out of contract
                rc = 1
        return rc, out.getvalue() + err.getvalue(), reached
    finally:
        for attr, original in originals:
            setattr(mod, attr, original)


def run_four_legs(
    mod,
    label: str,
    expect: str,
    note: str = "",
    fail_tag: str = "FAIL",
    *,
    raise_target: str = "_run",
    argv_of=_outdir(),
    passthru=lambda rc: [("_run", rc)],
) -> tuple[int, int]:
    """Run ENV / HARN / PASSTHRU-RED / PASSTHRU-GREEN against `mod`.

    Returns (failing legs, abstained legs). A harness-level raise inside a leg is
    a HarnessFault (exit 96), never a verdict.

    #466: a leg whose injection was never reached is ABSTAINED, not passed and
    not failed. It is named at its site with the row's own closing line, so an
    abstention is a declared state that a reader can audit -- never a quiet
    subtraction from the denominator.

    `fail_tag` names the token a DISAGREEING leg prints. It is "FAIL" everywhere a
    disagreement is bad news, and "FIRED" on the MUST_FIRE leg, where disagreement
    is the evidence being sought. Emitting "[FAIL]" for an intended firing would
    make an audit that greps the token self-hit on correct output -- the same
    use-vs-mention trap the "VERDICT RED" anchor below exists to avoid.
    """
    # PRE-fix the boundary answers RED to both. POST-fix it classifies.
    want = {"ENV": 5, "HARN": 5} if expect == "prefix" else {"ENV": 95, "HARN": 96}

    failures = 0
    abstained = 0
    for leg, exc in (("ENV", ENV_EXC), ("HARN", HARN_EXC)):
        with tempfile.TemporaryDirectory() as td:
            argv = argv_of(td)
            try:
                rc, stdout, reached = run_leg(mod, raise_target, argv, exc)
            except Exception as e:  # noqa: BLE001 -- a harness fault is 96, never a verdict
                print(f"[96] {label}/{leg}: harness raised {type(e).__name__}: {e}")
                raise HarnessFault(f"{label}/{leg}: {e}") from e
        # Anchor on the verdict token, NOT bare "RED": UNMEASURED *contains*
        # the substring RED, so a bare scan self-hits on the correct output.
        said_red = "VERDICT RED" in stdout
        if not reached:
            abstained += 1
            print(
                f"[ABSTAIN] {label}/{leg}: rc={rc}, but {raise_target}() was never called, "
                f"so the injected {type(exc).__name__} never happened and this leg measured "
                f"an absent precondition, not the boundary. {_why(stdout)}{note}"
            )
        else:
            ok = rc == want[leg]
            if not ok:
                failures += 1
            tag = "PASS" if ok else fail_tag
            print(f"[{tag}] {label}/{leg}: rc={rc} (want {want[leg]})  prints_RED={said_red}{note}")
        # A correct post-fix boundary must not print RED for either leg. This is
        # checked even on an abstained leg: a row that answers RED to a
        # precondition it could not meet is the #417 defect regardless of
        # whether the injection landed.
        if expect == "postfix" and said_red:
            failures += 1
            print(f"[{fail_tag}] {label}/{leg}: stdout asserts RED over an unmeasured claim{note}")

    # MUST_NOT_FIRE: a verdict the row actually REACHED must survive untouched.
    for name, rc_in in (("PASSTHRU-RED", 5), ("PASSTHRU-GREEN", 0)):
        with tempfile.TemporaryDirectory() as td:
            try:
                got, stdout, reached = run_passthrough(mod, passthru(rc_in), argv_of(td))
            except Exception as e:  # noqa: BLE001 -- a harness fault is 96, never a verdict
                print(f"[96] {label}/{name}: harness raised {type(e).__name__}: {e}")
                raise HarnessFault(f"{label}/{name}: {e}") from e
        if not reached:
            abstained += 1
            print(
                f"[ABSTAIN] {label}/{name}: rc={got}, but no stub was called, so the row "
                f"never reached a verdict to pass through. {_why(stdout)}{note}"
            )
            continue
        ok = got == rc_in
        if not ok:
            failures += 1
        print(f"[{'PASS' if ok else fail_tag}] {label}/{name}: rc={got} (want {rc_in}){note}")

    return failures, abstained


def run_self_test() -> int:
    """PRIMARY ENTRY: prove the control can fail before trusting that it passes.

    MUST_FIRE     : the synthetic PRE-fix subject (everything -> 5 RED) must be
                    REFUTED by this control's own leg logic.
    MUST_NOT_FIRE : the synthetic POST-fix subject (95 / 96 / passthrough) must
                    be CLEARED.
    Exit 0 only if zero self-test legs failed.
    """
    failures = 0
    with tempfile.TemporaryDirectory() as td:
        tdir = Path(td)
        mods = {}
        for stem, src in (
            ("synthetic_prefix_subject", PREFIX_SUBJECT_SRC),
            ("synthetic_postfix_subject", POSTFIX_SUBJECT_SRC),
            ("synthetic_unreachable_subject", UNREACHABLE_SUBJECT_SRC),
        ):
            path = tdir / f"{stem}.py"
            path.write_text(src, encoding="utf-8")
            try:
                mods[stem] = load(path)
            except Exception as exc:  # noqa: BLE001 -- cannot measure what we cannot load
                print(f"[96] self-test: cannot import {stem}: {type(exc).__name__}: {exc}")
                raise HarnessFault(f"self-test import: {exc}") from exc

        print("LEG MUST_FIRE: pre-fix synthetic subject must be refuted")
        pre_fail, pre_abs = run_four_legs(
            mods["synthetic_prefix_subject"],
            "synthetic_prefix",
            "postfix",
            note="  (expected: control MUST fire here)",
            fail_tag="FIRED",
        )
        if pre_abs:
            failures += 1
            print(
                f"[FAIL] MUST_FIRE: {pre_abs} leg(s) ABSTAINED on a subject that reaches "
                "its delegate every time -- the #466 reachability probe is over-firing "
                "and would mask real refutations"
            )
        if pre_fail > 0:
            print(f"[PASS] MUST_FIRE: control refuted the pre-fix subject ({pre_fail} leg(s))")
        else:
            failures += 1
            print(
                "[FAIL] MUST_FIRE: control PASSED a subject that answers 5 to every "
                "fault -- this control cannot fail, so its passes prove nothing"
            )

        print("LEG MUST_NOT_FIRE: post-fix synthetic subject must be cleared")
        post_fail, post_abs = run_four_legs(
            mods["synthetic_postfix_subject"], "synthetic_postfix", "postfix"
        )
        if post_fail == 0 and post_abs == 0:
            print("[PASS] MUST_NOT_FIRE: control cleared the post-fix subject")
        else:
            failures += 1
            print(
                f"[FAIL] MUST_NOT_FIRE: control refuted a correct post-fix subject "
                f"({post_fail} leg(s), {post_abs} abstained) -- it over-fires and "
                "cannot be trusted"
            )

        # #466. The abstention channel needs its own control, or it is an
        # untested escape hatch that could swallow every real finding: a probe
        # that always reports "unreachable" would turn this whole control green
        # while measuring nothing. The subject below returns 95 on an absent
        # precondition WITHOUT calling the delegate, which is correct behaviour.
        # All four legs must ABSTAIN -- not pass (the ENV leg's 95 would agree by
        # coincidence) and not fail (there is no defect to find).
        print("LEG MUST_ABSTAIN: a row that never reaches the delegate is not judged")
        unr_fail, unr_abs = run_four_legs(
            mods["synthetic_unreachable_subject"], "synthetic_unreachable", "postfix"
        )
        if unr_abs == 4 and unr_fail == 0:
            print("[PASS] MUST_ABSTAIN: all 4 legs abstained, 0 passed, 0 failed")
        else:
            failures += 1
            print(
                f"[FAIL] MUST_ABSTAIN: {unr_abs} of 4 legs abstained and {unr_fail} failed "
                "-- a leg that never reached the injection was scored as evidence"
            )

    print()
    print(f"SELF-TEST LEG COUNT: 3 run, {failures} failed")
    verdict = "CLEAR" if failures == 0 else "REFUTED"
    print(f"{verdict}: self-test {'shows the control can fail' if failures == 0 else 'FAILED'}")
    return 0 if failures == 0 else 5


def run_real_rows(expect: str) -> int:
    """SECONDARY/DEBUG ENTRY: run the four legs against every real row."""
    failures = 0
    abstained = 0
    legs = 0
    for row, (fn, raise_target, argv_of, passthru) in ROWS.items():
        path = ROWDIR / fn
        if not path.exists():
            print(f"[96] {row}: subject missing at {path}")
            return 96
        try:
            mod = load(path)
        except Exception as exc:  # noqa: BLE001 -- cannot measure a module it cannot load
            print(f"[96] {row}: could not import subject: {type(exc).__name__}: {exc}")
            return 96
        wanted = {raise_target, *(attr for attr, _ in passthru(5))}
        missing = sorted(a for a in wanted if not hasattr(mod, a))
        if missing:
            # A renamed delegate would silently make the legs measure nothing.
            print(f"[96] {row}: no attribute(s) {missing} to patch -- the spec is stale")
            return 96
        row_fail, row_abs = run_four_legs(
            mod,
            row,
            expect,
            raise_target=raise_target,
            argv_of=argv_of,
            passthru=passthru,
        )
        failures += row_fail
        abstained += row_abs
        legs += 4

    print()
    # The identity is printed, not implied: every leg is in exactly one column,
    # so an abstention can be seen to have been subtracted from neither.
    passed = legs - failures - abstained
    print(f"ROW LEG COUNT: {legs} run = {passed} passed + {failures} failed + "
          f"{abstained} abstained (each named at its site above)")
    if failures:
        print(f"REFUTED: {failures} failing leg(s), expect={expect}")
        return 5
    if passed == 0:
        # #466. Every leg abstaining is not a pass. This control would then have
        # certified the boundary rule while exercising it zero times, which is
        # the "unmeasured axis wearing a verdict" this whole file argues against.
        print(
            f"UNMEASURED: 0 of {legs} legs reached a boundary, so nothing about "
            f"the #417 rule was measured here; expect={expect}"
        )
        return 95
    print(f"CLEAR: 0 failing leg(s) over {passed} measured, expect={expect}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="#417 boundary control. PRIMARY entry: --self-test "
        "(proves the control can fail). DEBUG entry: --expect."
    )
    ap.add_argument(
        "--self-test",
        action="store_true",
        help="PRIMARY: build synthetic pre/post-fix subjects in a temp dir and prove "
        "this control refutes the former (MUST_FIRE) and clears the latter "
        "(MUST_NOT_FIRE) before any real-row result is trusted.",
    )
    ap.add_argument(
        "--expect",
        choices=("prefix", "postfix"),
        default=None,
        help="SECONDARY/DEBUG: run the four legs against the real rows, expecting "
        "pre-fix (5/5) or post-fix (95/96) boundary behaviour.",
    )
    return ap


def main(argv: list[str] | None = None) -> int:
    """Outermost handler: nothing escapes as a bare crash.

    Only 0 / 5 / 95 / 96 ever leave this function. argparse's own exits are
    remapped (--help's 0 stays 0; a bad flag's 2 becomes 96, a refusal). An
    escaped OSError is an environment fault (95); any other escaped exception
    is a harness fault (96) -- never 5, which would be a false refutation.
    """
    try:
        ap = build_parser()
        try:
            args = ap.parse_args(argv)
        except SystemExit as se:
            return 0 if se.code == 0 else 96

        if args.self_test:
            return run_self_test()
        if args.expect is not None:
            return run_real_rows(args.expect)
        print("[96] no mode selected: use --self-test (primary) or --expect prefix|postfix")
        return 96
    except HarnessFault as hf:
        print(f"[96] control harness fault: {hf}")
        return 96
    except OSError as exc:  # noqa: BLE001 -- an environment fault against the control itself
        print(f"[95] control environment fault: {type(exc).__name__}: {exc}")
        return 95
    except Exception as exc:  # noqa: BLE001 -- a harness fault, never a false refutation
        print(f"[96] control harness fault: {type(exc).__name__}: {exc}")
        return 96


if __name__ == "__main__":
    sys.exit(main())
