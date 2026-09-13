#!/usr/bin/env python3
"""#417 boundary control: an ENVIRONMENT fault must not be adjudicated as RED.

WHAT IS CLAIMED
  For each row under test (t1_9, t1_10, t1_11, t1_12), per row four dynamic
  legs (not text tripwires -- see #101):
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

ROWS = {
    "t1_9": ("t1_9_optimizer_arms.py", []),
    "t1_10": ("t1_10_accumulation_equivalence.py", []),
    "t1_11": ("t1_11_grad_checkpointing.py", []),
    "t1_12": ("t1_12_attention_impl.py", ["--profile-name", "single-node"]),
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


class HarnessFault(Exception):
    """The control itself could not run a leg -- exit 96, never 5."""


def load(path: Path):
    name = f"fs417ctl_{path.stem}"
    loader = SourceFileLoader(name, str(path))
    spec = spec_from_loader(name, loader)
    mod = module_from_spec(spec)
    sys.modules[name] = mod
    loader.exec_module(mod)
    return mod


def run_leg(mod, argv: list[str], exc: BaseException) -> tuple[int, str]:
    """Patch _run to raise `exc`, call main(argv), return (rc, captured stdout)."""

    def boom(_args):
        raise exc

    original = mod._run
    mod._run = boom
    out, err = io.StringIO(), io.StringIO()
    try:
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            try:
                rc = mod.main(argv)
            except SystemExit as se:  # argparse or an un-caught exit
                rc = se.code if isinstance(se.code, int) else 96
    finally:
        mod._run = original
    return rc, out.getvalue()


def run_passthrough(mod, argv: list[str], rc_in: int) -> int:
    """_run RETURNS a verdict (does not raise). main() must pass it through."""

    def quiet(_args):
        return rc_in

    original = mod._run
    mod._run = quiet
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            try:
                return mod.main(argv)
            except SystemExit as se:
                return se.code if isinstance(se.code, int) else 96
    finally:
        mod._run = original


def run_four_legs(
    mod, label: str, extra: list[str], expect: str, note: str = "", fail_tag: str = "FAIL"
) -> int:
    """Run ENV / HARN / PASSTHRU-RED / PASSTHRU-GREEN against `mod`.

    Returns the number of failing legs. A harness-level raise inside a leg is a
    HarnessFault (exit 96), never a verdict.

    `fail_tag` names the token a DISAGREEING leg prints. It is "FAIL" everywhere a
    disagreement is bad news, and "FIRED" on the MUST_FIRE leg, where disagreement
    is the evidence being sought. Emitting "[FAIL]" for an intended firing would
    make an audit that greps the token self-hit on correct output -- the same
    use-vs-mention trap the "VERDICT RED" anchor below exists to avoid.
    """
    # PRE-fix the boundary answers RED to both. POST-fix it classifies.
    want = {"ENV": 5, "HARN": 5} if expect == "prefix" else {"ENV": 95, "HARN": 96}

    failures = 0
    for leg, exc in (("ENV", ENV_EXC), ("HARN", HARN_EXC)):
        with tempfile.TemporaryDirectory() as td:
            argv = ["--out-dir", td, *extra]
            try:
                rc, stdout = run_leg(mod, argv, exc)
            except Exception as e:  # noqa: BLE001 -- a harness fault is 96, never a verdict
                print(f"[96] {label}/{leg}: harness raised {type(e).__name__}: {e}")
                raise HarnessFault(f"{label}/{leg}: {e}") from e
        ok = rc == want[leg]
        # Anchor on the verdict token, NOT bare "RED": UNMEASURED *contains*
        # the substring RED, so a bare scan self-hits on the correct output.
        said_red = "VERDICT RED" in stdout
        if not ok:
            failures += 1
        tag = "PASS" if ok else fail_tag
        print(f"[{tag}] {label}/{leg}: rc={rc} (want {want[leg]})  prints_RED={said_red}{note}")
        # A correct post-fix boundary must not print RED for either leg.
        if expect == "postfix" and said_red:
            failures += 1
            print(f"[{fail_tag}] {label}/{leg}: stdout asserts RED over an unmeasured claim{note}")

    # MUST_NOT_FIRE: a verdict the row actually REACHED must survive untouched.
    for name, rc_in in (("PASSTHRU-RED", 5), ("PASSTHRU-GREEN", 0)):
        with tempfile.TemporaryDirectory() as td:
            try:
                got = run_passthrough(mod, ["--out-dir", td, *extra], rc_in)
            except Exception as e:  # noqa: BLE001 -- a harness fault is 96, never a verdict
                print(f"[96] {label}/{name}: harness raised {type(e).__name__}: {e}")
                raise HarnessFault(f"{label}/{name}: {e}") from e
        ok = got == rc_in
        if not ok:
            failures += 1
        print(f"[{'PASS' if ok else fail_tag}] {label}/{name}: rc={got} (want {rc_in}){note}")

    return failures


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
        ):
            path = tdir / f"{stem}.py"
            path.write_text(src, encoding="utf-8")
            try:
                mods[stem] = load(path)
            except Exception as exc:  # noqa: BLE001 -- cannot measure what we cannot load
                print(f"[96] self-test: cannot import {stem}: {type(exc).__name__}: {exc}")
                raise HarnessFault(f"self-test import: {exc}") from exc

        print("LEG MUST_FIRE: pre-fix synthetic subject must be refuted")
        pre_fail = run_four_legs(
            mods["synthetic_prefix_subject"],
            "synthetic_prefix",
            [],
            "postfix",
            note="  (expected: control MUST fire here)",
            fail_tag="FIRED",
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
        post_fail = run_four_legs(
            mods["synthetic_postfix_subject"], "synthetic_postfix", [], "postfix"
        )
        if post_fail == 0:
            print("[PASS] MUST_NOT_FIRE: control cleared the post-fix subject")
        else:
            failures += 1
            print(
                f"[FAIL] MUST_NOT_FIRE: control refuted a correct post-fix subject "
                f"({post_fail} leg(s)) -- it over-fires and cannot be trusted"
            )

    print()
    print(f"SELF-TEST LEG COUNT: 2 run, {failures} failed")
    verdict = "CLEAR" if failures == 0 else "REFUTED"
    print(f"{verdict}: self-test {'shows the control can fail' if failures == 0 else 'FAILED'}")
    return 0 if failures == 0 else 5


def run_real_rows(expect: str) -> int:
    """SECONDARY/DEBUG ENTRY: run the four legs against the real rows."""
    failures = 0
    legs = 0
    for row, (fn, extra) in ROWS.items():
        path = ROWDIR / fn
        if not path.exists():
            print(f"[96] {row}: subject missing at {path}")
            return 96
        try:
            mod = load(path)
        except Exception as exc:  # noqa: BLE001 -- cannot measure a module it cannot load
            print(f"[96] {row}: could not import subject: {type(exc).__name__}: {exc}")
            return 96
        failures += run_four_legs(mod, row, extra, expect)
        legs += 4

    print()
    print(f"ROW LEG COUNT: {legs} run, {failures} failed")
    verdict = "CLEAR" if failures == 0 else "REFUTED"
    print(f"{verdict}: {failures} failing leg(s), expect={expect}")
    return 0 if failures == 0 else 5


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
