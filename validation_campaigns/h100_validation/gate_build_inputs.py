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

  I1  after a build, h100/gen/ contains EXACTLY the declared produced set -- reported in
      both directions, because "no unexpected files" and "every declared file present" are
      different claims and a build that silently stopped producing one of them would
      otherwise read as clean
  I2  no upstream file shares a name with a produced one (an input shadowing an output is
      how a stale copy gets consumed while everyone reads the fresh one)
  I3  every file in h100/upstream/ appears in that directory's README table, so an input
      cannot arrive without its provenance being written down
  I4  MUST_FIRE: I1 is drilled with a planted file. A comparison that cannot go red reports
      "0 unexpected" for a dead check exactly as loudly as for a clean tree.
"""

from __future__ import annotations

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
        return 3

    found = _files(GEN)
    extra = sorted(found - PRODUCED)
    missing = sorted(PRODUCED - found)
    gates.append((
        "I1 h100/gen/ holds exactly the declared produced set",
        not extra and not missing,
        f"{len(found)} present, {len(PRODUCED)} declared"
        + (f"; UNDECLARED: {extra}" if extra else "")
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
    return 0 if ok else 5


if __name__ == "__main__":
    raise SystemExit(main())
