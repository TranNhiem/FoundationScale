#!/usr/bin/env python3
"""Partition every file the build touches into PRODUCED or DECLARED-UPSTREAM, and refuse
on anything else.

#136 and #137 were the same defect twice. In both cases a file the build READ was sitting
in the directory the build WRITES, so nobody could tell it apart from an artifact:

  #136  fs_container_backend.spliced.sh -- 73 KB, the shipped backend's entire base text,
        produced by no stage and removed by no rm. Every "rebuilt from scratch" run was
        built on top of it.
  #137  launchers__launch_fs_h100.sh -- the shipped launcher's entire base text, with no
        upstream anywhere in fs-repo, sitting three lines away from an `rm -f` over the
        same directory.

Both were found by accident. This gate is the thing that would have found them on purpose.

It deliberately does NOT try to work out what the stages read by parsing them. A static
reader of arbitrary Python path construction gets it wrong, and this project already has a
filed instance of that failure: an inline classifier that reported a default it had read out
of the inside of an error-message string. Instead the check is exact and needs no inference:

  I1  after a build, h100/gen/ contains no UNDECLARED file
  I1b every declared artifact is PRESENT

      Two legs, not one, because "no unexpected files" and "every declared file present" are
      different claims and a build that silently stopped producing one of them would
      otherwise read as clean. They also fail differently, which is #297: an undeclared file
      is evidence whether or not the build finished, but a declared file can be absent simply
      because the stage that writes it never ran. So I1 is unconditionally RED-able and I1b
      is DEFERRED to a refusal (96) when build_h100_plane.sh exports FS_BUILD_INCOMPLETE.
      Unset means "assume the build completed", so a by-hand or CI run keeps the strict
      reading, and a RED on I1 still dominates a deferral on I1b.
  I2  no upstream file shares a name with a produced one (an input shadowing an output is
      how a stale copy gets consumed while everyone reads the fresh one)
  I3  every file in h100/upstream/ appears in that directory's README table, so an input
      cannot arrive without its provenance being written down
  I4  MUST_FIRE: I1 is drilled with a planted file. A comparison that cannot go red reports
      "0 unexpected" for a dead check exactly as loudly as for a clean tree.
"""

from __future__ import annotations

import os
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent
GEN = ROOT / "h100" / "gen"
UPSTREAM = ROOT / "h100" / "upstream"

# The artifacts every build produces. This list is the CONTRACT: adding a stage that emits a
# new file means adding it here, which is the point -- an artifact nobody declared is exactly
# the state #136 lived in.
PRODUCED = {
    "launch_fs_h100.fixed.sh",
    "fs_container_backend.bound.sh",
    "fs_container_backend.spliced.sh",  # intermediate, but generated and removed like one
    "fs_train.fixed.py",
    "fs_model_root.py",
    "test_fs_model_root.py",
    # #141: the checkpoint adjudicator the launcher's required knob has been asking for since
    # #68 wired its call sites. Its suite is generated too, for the #133 reason -- a hand-kept
    # test beside a generated module drifts the moment the module's rule changes.
    "fs_ckpt_adjudicator.py",
    "test_fs_ckpt_adjudicator.py",
    # #183: the login-node argv preflight. Unlike its neighbours this one is not synthesised
    # from a template -- patch_argv_preflight.py COPIES it from the hand-authored source at
    # the build root. It is still produced: the copy is what the plane ships and what the
    # spliced call resolves, the build root holds the input, and nothing but the stage writes
    # here. Declaring the copy is what keeps the two in the same direction #137 established.
    "fs_argv_preflight.py",
}

# Interpreter byproducts, not artifacts. Named rather than pattern-matched so the exclusion
# is auditable: a wildcard here could hide a real file.
IGNORED_DIRS = {"__pycache__", ".pytest_cache"}


def _files(d: pathlib.Path) -> set[str]:
    return {p.name for p in d.iterdir() if p.is_file()} if d.is_dir() else set()


def main() -> int:
    gates: list[tuple[str, bool, str]] = []

    if not GEN.is_dir():
        print(f"REFUSING: {GEN} absent — run build_h100_plane.sh first", file=sys.stderr)
        return 95

    found = _files(GEN)
    extra = sorted(found - PRODUCED)
    missing = sorted(PRODUCED - found)

    # #297. The docstring above already says these are two different claims; until now the
    # code collapsed them into one verdict, and that is what made this gate report RED on a
    # tree nobody claimed was finished.
    #
    #   UNDECLARED (extra)  -- a file is here that no stage declares. Evidence regardless of
    #                          whether the build finished: an undeclared file does not appear
    #                          because a LATER stage refused. This stays RED, always.
    #   MISSING (absent)    -- a declared artifact is not on disk. On a completed build that is
    #                          RED. On a refused build it is a restatement of the refusal, and
    #                          the diagnosis printed below ("it will survive every from-scratch
    #                          rebuild") would be false. Downgraded to a refusal, and ONLY while
    #                          the build says so.
    #
    # The signal is the build's, not this gate's inference: FS_BUILD_INCOMPLETE is exported by
    # build_h100_plane.sh when a stage exits 95/96. Unset means "assume complete", which is the
    # safe direction -- a gate run by hand or by CI keeps the strict reading.
    incomplete = os.environ.get("FS_BUILD_INCOMPLETE", "").strip()

    gates.append((
        "I1 h100/gen/ holds no UNDECLARED file",
        not extra,
        f"{len(found)} present, {len(PRODUCED)} declared"
        + (f"; UNDECLARED: {extra}" if extra else ""),
    ))

    deferred = bool(missing) and bool(incomplete)
    if deferred:
        print(
            f"  I1b every declared artifact is present: DEFERRED — the build did not finish "
            f"(stage {incomplete} refused), so {len(missing)} declared artifact(s) are absent "
            f"because they were never produced, not because the declaration is wrong.\n"
            f"       ABSENT: {missing}"
        )
    else:
        gates.append((
            "I1b every declared artifact is present",
            not missing,
            f"{len(found)} present, {len(PRODUCED)} declared"
            + (f"; MISSING: {missing}" if missing else ""),
        ))

    up = _files(UPSTREAM)
    shadow = sorted(up & PRODUCED)
    gates.append((
        "I2 no upstream file shadows a produced artifact",
        not shadow,
        f"{len(up)} upstream file(s)" + (f"; SHADOWING: {shadow}" if shadow else ""),
    ))

    readme = UPSTREAM / "README.md"
    if not readme.is_file():
        gates.append(("I3 every upstream file is documented", False, "README.md absent"))
    else:
        text = readme.read_text("utf-8")
        undocumented = sorted(n for n in up if n != "README.md" and n not in text)
        gates.append((
            "I3 every upstream file is documented in README.md",
            not undocumented,
            f"{len(up) - 1} input(s) documented"
            + (f"; UNDOCUMENTED: {undocumented}" if undocumented else ""),
        ))

    # I4 -- drill I1. The planted file is written and removed here so the control runs on
    # every invocation rather than being a claim in a comment.
    planted = GEN / "gate_build_inputs.MUSTFIRE.tmp"
    try:
        planted.write_text("planted by I4\n", "utf-8")
        fired = bool(_files(GEN) - PRODUCED)
    finally:
        planted.unlink(missing_ok=True)
    gates.append(("I4 MUST_FIRE: I1 detects a planted undeclared file", fired, ""))

    ok = True
    for name, passed, detail in gates:
        ok &= passed
        print(f"  {name}: {'PASS' if passed else 'FAIL'}" + (f" ({detail})" if detail else ""))
    print(f"\n  {sum(1 for _, p, _ in gates if p)}/{len(gates)} input-partition gates green")
    if not ok:
        print(
            "\nINPUT PARTITION RED — a file the build touches is neither a declared artifact\n"
            "nor a documented upstream. That is the #136/#137 state: it will survive every\n"
            "'from scratch' rebuild and nobody will be able to say where it came from.",
            file=sys.stderr,
        )
        # RED dominates a deferral. A genuinely undeclared file is a finding on an unfinished
        # tree too, and reporting 96 here would let the refusal launder it.
        return 5
    if deferred:
        print(
            f"\nINPUT PARTITION REFUSED (96) — every claim this gate CAN decide on an "
            f"unfinished tree is green,\n  but the declared-artifact-present claim cannot be "
            f"decided: stage {incomplete} refused, so the\n  build never produced them. That "
            f"is CANNOT-MEASURE, not RED. Re-run on a completed build.",
            file=sys.stderr,
        )
        return 96
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
