#!/usr/bin/env python3
"""#126: the launcher defaults FS_ALLOCATION, defeating a required-no-default guard.

WHAT WAS MEASURED
  launcher :36   export FS_ALLOCATION="${FS_ALLOCATION:-slurm}"
  backend  :159  [[ -n "${FS_ALLOCATION:-}" ]] || fs_die "FS_ALLOCATION is unset/empty
                 (required, no default by design). ... It is never inferred ..."

The backend's guard states, in its own message, that the value is never inferred. Through
this launcher it is ALWAYS inferred, because the launcher supplies `slurm` one layer up
before the backend ever looks. The guard cannot fire. It is not a weak guard -- it is an
unreachable one, and an unreachable guard reads as protection while providing none.

The correct idiom is TWO LINES BELOW the defective one, which is what makes this a slip
rather than a design position:

  :38   export FS_CONTAINER_RUNTIME="${FS_CONTAINER_RUNTIME:-}"   <- defaults to EMPTY
  :39   [[ ... == singularity ]] || fail 96 "... (no default); got '<unset>'"

FS_ALLOCATION joins the required-with-no-default family for the reason already recorded in
that family's rationale: an unconfigured guard is a disabled standing rule. The family is
now FS_ALLOWED_NODE, FS_CONTAINER_RUNTIME, FS_ALLOWED_PATH_ROOTS, FS_ENGINE_LAUNCH_MODE
and FS_ALLOCATION.

WHAT THIS FIX DOES *NOT* CLAIM. After this patch the backend's :159 message still does not
fire through this launcher -- the launcher refuses first, at :37. That is correct and
intended: a launcher should refuse early, and the backend's guard remains the authority for
any other caller. The defect being closed is that the operator's silence was being
converted into an answer, not that a particular error string goes unprinted. The launcher's
own message is rewritten to carry the reason, so refusing early costs no information.

BREAKING CHANGE, stated plainly: any submission relying on the implicit `slurm` now fails
closed with an actionable message. /tmp/fs_phase3.sbatch is one such submission. This is
the third required-no-default variable added since that file was written, which is the
argument for generating its --export list from the same extractor that generates LAUNCH.md
rather than maintaining it by hand.

GATES
  A1 idempotent (marker)
  A2 anchor unique
  A3 bash -n clean on the patched text
  A4 EXECUTED behaviour rows -- the point of the patch is a runtime refusal, so asserting
     the text changed proves nothing:
       MUST_FIRE  unset            -> refused, message names the variable
       MUST_FIRE  empty            -> refused (the ${x:-} form must not paper over "")
       MUST_FIRE  FS_ALLOCATION=local -> refused BY THIS LAUNCHER (it is slurm-only), and
                  with a message distinguishable from the unset case
       MUST_PASS  FS_ALLOCATION=slurm  -> accepted
  A5 the defaulting form is GONE from the file. A patch that appends a guard while leaving
     the `:-slurm` expansion in place would satisfy A4 and change nothing.
  A6 INTEGRATION: gate_env_drift.py stays green (FS_ALLOCATION is a declared HOST_ONLY
     entry there; this patch must not disturb that).
"""

from __future__ import annotations

import importlib.util
import pathlib
import re
import subprocess
import sys
import tempfile

GEN = pathlib.Path(__file__).resolve().parent / "h100" / "gen"
LAUNCH = GEN / "launch_fs_h100.fixed.sh"
MARK = "fs126:"

ANCHOR = (
    'export FS_ALLOCATION="${FS_ALLOCATION:-slurm}"\n'
    '[[ "$FS_ALLOCATION" == slurm ]] || fail 96 '
    '"launch_fs_h100.sh only supports FS_ALLOCATION=slurm; got \'$FS_ALLOCATION\'"\n'
)

REPL = (
    '# fs126: no default. The backend\'s own guard calls FS_ALLOCATION "required, no default\n'
    '# by design ... never inferred", and this line used to infer it -- making that guard\n'
    '# unreachable through this launcher. Defaulting to EMPTY and refusing is the idiom used\n'
    '# for FS_CONTAINER_RUNTIME two lines below; an unconfigured guard is a disabled rule.\n'
    'export FS_ALLOCATION="${FS_ALLOCATION:-}"\n'
    '[[ "$FS_ALLOCATION" == slurm ]] || fail 96 '
    '"FS_ALLOCATION must be exactly \'slurm\' for this launcher (no default by design: who '
    'allocated these nodes is a separate axis from which runtime launches on them, and it is '
    'never inferred from SLURM_JOB_ID). got \'${FS_ALLOCATION:-<unset>}\'"\n'
)

# A4 harness: the two lines under test, lifted verbatim, with fail() stubbed.
DRILL = r"""
set -uo pipefail
fail() { printf 'REFUSED(%s): %s\n' "$1" "$2" >&2; exit 96; }
export FS_ALLOCATION="${FS_ALLOCATION:-}"
[[ "$FS_ALLOCATION" == slurm ]] || fail 96 "FS_ALLOCATION must be exactly 'slurm' for this launcher (no default by design: who allocated these nodes is a separate axis from which runtime launches on them, and it is never inferred from SLURM_JOB_ID). got '${FS_ALLOCATION:-<unset>}'"
printf 'ACCEPTED %s\n' "$FS_ALLOCATION"
"""


def _run(env_kv):
    import os
    e = {k: v for k, v in os.environ.items() if k != "FS_ALLOCATION"}
    e.update(env_kv)
    return subprocess.run(["bash", "-c", DRILL], capture_output=True, text=True, env=e)


def _syntax(text: str) -> bool:
    with tempfile.NamedTemporaryFile("w", suffix=".sh", delete=False) as fh:
        fh.write(text)
        tmp = fh.name
    r = subprocess.run(["bash", "-n", tmp], capture_output=True, text=True)
    if r.returncode:
        print(f"  FAIL A3  bash -n: {r.stderr.strip()[:250]}", file=sys.stderr)
        return False
    return True


def main() -> int:
    lau = LAUNCH.read_text("utf-8")

    if MARK in lau:                                                            # A1
        print("  A1  already applied — no-op (idempotent)")
        return 0
    print("  PASS A1  not yet applied")

    n = lau.count(ANCHOR)                                                      # A2
    if n != 1:
        print(f"  FAIL A2  anchor occurs {n}x (need exactly 1)", file=sys.stderr)
        return 5
    print("  PASS A2  anchor unique")

    new = lau.replace(ANCHOR, REPL, 1)

    ok = _syntax(new)                                                          # A3
    if ok:
        print("  PASS A3  bash -n clean")

    rows = []                                                                  # A4
    r = _run({})
    rows.append(("MUST_FIRE unset refused",
                 r.returncode == 96 and "FS_ALLOCATION" in r.stderr
                 and "<unset>" in r.stderr, f"rc={r.returncode}"))
    r = _run({"FS_ALLOCATION": ""})
    rows.append(("MUST_FIRE empty refused", r.returncode == 96, f"rc={r.returncode}"))
    r = _run({"FS_ALLOCATION": "local"})
    rows.append(("MUST_FIRE 'local' refused, distinguishably from unset",
                 r.returncode == 96 and "got 'local'" in r.stderr
                 and "<unset>" not in r.stderr, f"rc={r.returncode}"))
    r = _run({"FS_ALLOCATION": "slurm"})
    rows.append(("MUST_PASS 'slurm' accepted",
                 r.returncode == 0 and "ACCEPTED slurm" in r.stdout, f"rc={r.returncode}"))
    bad = [(nm, got) for nm, good, got in rows if not good]
    if bad:
        print(f"  FAIL A4  {len(bad)} of {len(rows)} rows wrong: {bad}", file=sys.stderr)
        ok = False
    else:
        print(f"  PASS A4  {len(rows)}/{len(rows)} rows EXECUTED: refuses unset, empty and "
              f"'local'; accepts 'slurm'")

    # A5 the defaulting form must be gone, not merely guarded.
    if re.search(r'FS_ALLOCATION:-\s*slurm', new):
        print("  FAIL A5  the ':-slurm' expansion survives — the operator's silence is "
              "still being converted into an answer", file=sys.stderr)
        ok = False
    else:
        print("  PASS A5  no ':-slurm' expansion remains in the launcher")

    if not ok:
        print("\nREFUSING TO WRITE — gates above are red", file=sys.stderr)
        return 5

    LAUNCH.write_text(new, "utf-8")

    spec = importlib.util.spec_from_file_location(                             # A6
        "gate_env_drift", str(pathlib.Path(__file__).resolve().parent / "gate_env_drift.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    back = (GEN / "fs_container_backend.bound.sh").read_text("utf-8")
    viol = mod.check(new, back, quiet=True)
    if viol:
        print(f"  FAIL A6  gate_env_drift went red: {viol}", file=sys.stderr)
        return 5
    print("  PASS A6  gate_env_drift.py still green on the patched pair")

    print(f"\nALL GATES GREEN -> {LAUNCH.name}")
    print("BREAKING: submissions relying on the implicit FS_ALLOCATION=slurm now fail closed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
