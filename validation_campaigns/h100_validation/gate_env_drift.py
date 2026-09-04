#!/usr/bin/env python3
"""#116: pin FS_ENV_ALLOWLIST against the launcher's exports so the two cannot drift.

WHY THIS GATE EXISTS. Two defects, found by hand, weeks apart, were the same defect:

  #115  the launcher exported 12 names; the allowlist carried 5. Seven were dropped
        silently, because a name absent from an allowlist produces no error -- it
        produces a variable that is simply not there, and a trainer that reads a
        default instead.
  #122  FS_RESUME_CKPT / FS_RESUME_STEP were exported under a runtime-specific alias
        (SINGULARITYENV_*), never reached the allowlist, and so never crossed under
        enroot. The resume proof would have passed BY NOT RESUMING.

Both were caught by a person reading two files side by side. That is not a control.
This gate makes the class mechanical, and it checks BOTH directions, because the two
directions fail differently:

  FORWARD   an exported name that does not cross  -> the container silently defaults
  REVERSE   an allowlisted name that nothing produces -> either dead weight, or a
            consumer that will fail closed on a variable no one ever sets

The reverse direction is the one a human never runs, and it is how MASTER_PORT came to
sit on the allowlist with zero producers anywhere in either file.

RULES
  D1  the launcher exports no SINGULARITYENV_* name. Runtime-specific injection is the
      backend's business; a launcher that names a runtime has re-created #109/#117/#122.
      This is a rule, not a waiver list, because the whole point of the two-axis seam is
      that the launcher does not know which runtime it is on.
  D2  every launcher export either crosses (on the allowlist) or is declared host-only
      with a STATED REASON. A waiver without a reason is just a longer allowlist.
  D3  every allowlisted name has a producer: exported by the launcher, exported by the
      backend, or supplied by the workload manager. An allowlist entry no one sets is
      not harmless -- it is a claim about the environment that nothing backs.
  D4  every verdict carries its denominator.
  D5  the detectors are drilled: a planted violation of D1, D2 and D3 must each be
      observed going red. A detector that cannot match reports zero and lies.

EXIT 0 only when every rule holds. This is a gate, not a report.
"""

from __future__ import annotations

import pathlib
import re
import sys

GEN = pathlib.Path(__file__).resolve().parent / "h100" / "gen"
BACKEND = GEN / "fs_container_backend.bound.sh"
LAUNCH = GEN / "launch_fs_h100.fixed.sh"

# D2: host-only by design. Each entry MUST carry why it does not need to cross.
HOST_ONLY = {
    "FS_CONTAINER_RUNTIME":
        "control-plane axis read by fs_backend_init, which is SOURCED into the "
        "launcher's own shell. It selects the arm; nothing inside the container reads it.",
    "FS_ALLOCATION":
        "the second control-plane axis, same reason: it decides who allocated the "
        "nodes, a question that is answered before any container exists.",
    "FS_BACKEND":
        "derived arm selector consumed by backend functions in the host shell "
        "(fs_launch_python branches on it). In-container code must never branch on the "
        "runtime -- that is the defect the two-axis seam exists to prevent.",
    "FS_PLANE_DIR":
        "#142's resolved plane directory. It answers a question that only exists on the "
        "host -- where do this launcher's siblings live -- and it is answered BEFORE the "
        "backend is sourced, let alone before a container exists. Deliberately not "
        "allowlisted: the path is a HOST path, and a host path crossing into a container "
        "under the same name is how a bind-mounted tree acquires two meanings (#133). "
        "It is also an accepted operator OVERRIDE, so letting it cross would let an "
        "in-container process silently re-point the next link of the submit chain.",
}

# D3: names the workload manager supplies to the job environment. Not produced by our
# code, and correctly allowlisted so they reach the container.
WLM_PROVIDED = {
    "SLURM_JOB_ID", "SLURM_JOB_NODELIST", "SLURM_NNODES", "SLURM_NTASKS",
    "SLURM_NTASKS_PER_NODE", "SLURM_SUBMIT_DIR",
}

EXPORT_RE = re.compile(r"^[ \t]*export[ \t]+(.+)$", re.M)
NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*")


def exported_names(src: str) -> dict[str, int]:
    """name -> first line number. Handles `export A=1` and `export A B C`."""
    out: dict[str, int] = {}
    for m in EXPORT_RE.finditer(src):
        line = src[:m.start()].count("\n") + 1
        for tok in m.group(1).split():
            if tok.startswith("-"):
                continue
            nm = NAME_RE.match(tok)
            if nm:
                out.setdefault(nm.group(0), line)
    return out


def allowlist(src: str) -> list[str]:
    m = re.search(r"^  FS_ENV_ALLOWLIST=\($", src, re.M)
    if not m:
        return []
    names = []
    for ln in src[m.end():].splitlines()[1:]:
        if ln.rstrip() == "  )":
            break
        tok = ln.split("#")[0].strip()
        if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", tok):
            names.append(tok)
    return names


def check(lau: str, back: str, quiet: bool = False) -> list[str]:
    """Returns a list of violation strings. Empty == clean."""
    v: list[str] = []
    lex = exported_names(lau)
    bex = exported_names(back)
    allow = allowlist(back)
    aset = set(allow)

    if not allow:
        return ["D0 allowlist parsed as EMPTY — the detector cannot certify anything"]

    # D1 -------------------------------------------------------------------
    sing = {n: l for n, l in lex.items() if n.startswith("SINGULARITYENV_")}
    if sing:
        v.append(f"D1 launcher exports {len(sing)} runtime-specific name(s): "
                 + ", ".join(f"{n}@L{l}" for n, l in sorted(sing.items())))
    elif not quiet:
        print(f"  PASS D1  0 SINGULARITYENV_* exports in the launcher "
              f"({len(lex)} exports scanned)")

    # D2 -------------------------------------------------------------------
    candidates = {n: l for n, l in lex.items() if not n.startswith("SINGULARITYENV_")}
    undeclared = {n: l for n, l in candidates.items()
                  if n not in aset and n not in HOST_ONLY}
    if undeclared:
        v.append(f"D2 {len(undeclared)} of {len(candidates)} launcher export(s) neither "
                 f"cross nor are declared host-only: "
                 + ", ".join(f"{n}@L{l}" for n, l in sorted(undeclared.items())))
    elif not quiet:
        crossing = sum(1 for n in candidates if n in aset)
        print(f"  PASS D2  {crossing} of {len(candidates)} launcher exports cross; "
              f"{len(candidates) - crossing} declared host-only with a stated reason")

    # D3 -------------------------------------------------------------------
    orphan = [n for n in allow
              if n not in lex and n not in bex and n not in WLM_PROVIDED]
    if orphan:
        v.append(f"D3 {len(orphan)} of {len(allow)} allowlisted name(s) have NO producer "
                 f"(not exported by launcher or backend, not workload-manager supplied): "
                 + ", ".join(sorted(orphan)))
    elif not quiet:
        # Disjoint by precedence, so the parts sum to the whole. Counting each
        # category independently over-counts names with two producers and prints
        # a breakdown larger than its own denominator -- the exact shape of claim
        # this gate exists to catch, in miniature.
        by = {"launcher": 0, "backend": 0, "workload manager": 0}
        for n in allow:
            if n in lex:
                by["launcher"] += 1
            elif n in bex:
                by["backend"] += 1
            else:
                by["workload manager"] += 1
        assert sum(by.values()) == len(allow)
        print(f"  PASS D3  {len(allow)} of {len(allow)} allowlisted names have a producer "
              + ", ".join(f"{v} {k}" for k, v in by.items())
              + " (disjoint; first producer wins)")

    return v


def drills(lau: str, back: str) -> bool:
    """D5. Each detector must be observed going red on a planted violation."""
    ok = True
    rows = [
        ("D1", lau + '\nexport SINGULARITYENV_PLANTED=1\n', back),
        ("D2", lau + '\nexport FS_PLANTED_DRIFT=1\n', back),
        ("D3", lau, back.replace("    NCCL_MNNVL_ENABLE\n",
                                 "    NCCL_MNNVL_ENABLE\n    FS_PLANTED_ORPHAN\n", 1)),
    ]
    for rule, l2, b2 in rows:
        if b2 == back and rule == "D3":
            print(f"  FAIL D5/{rule}  could not plant the violation — an unplantable "
                  f"drill proves nothing", file=sys.stderr)
            ok = False
            continue
        hit = [x for x in check(l2, b2, quiet=True) if x.startswith(rule)]
        if hit:
            print(f"  PASS D5/{rule} MUST_FIRE: planted violation observed going red")
        else:
            print(f"  FAIL D5/{rule} MUST_FIRE did not fire — the detector is dead",
                  file=sys.stderr)
            ok = False
    # MUST_PASS: the unplanted pair must not trip the drills' own machinery.
    if check(lau, back, quiet=True):
        pass  # real violations are reported by the main pass, not here
    return ok


def main() -> int:
    lau = LAUNCH.read_text("utf-8")
    back = BACKEND.read_text("utf-8")

    print(f"launcher: {len(lau.splitlines())} lines   "
          f"backend: {len(back.splitlines())} lines\n")
    violations = check(lau, back)
    for x in violations:
        print(f"  FAIL {x}", file=sys.stderr)
    print()
    drilled = drills(lau, back)

    if violations or not drilled:
        print(f"\nENV DRIFT GATE RED — {len(violations)} violation(s), "
              f"drills {'green' if drilled else 'RED'}", file=sys.stderr)
        return 5
    print("\nENV DRIFT GATE GREEN — both directions checked, all three detectors drilled")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
