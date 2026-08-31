#!/usr/bin/env python3
"""#116 remediation: clear the two real reds gate_env_drift.py found.

Both were found by the gate, not by reading. Both were observed RED on the real
generated pair before this patch was written:

  D1  launcher exports SINGULARITYENV_PYTHONNOUSERSITE
  D3  MASTER_PORT is on FS_ENV_ALLOWLIST with ZERO producers anywhere

--- D1 -----------------------------------------------------------------------
DELETE, not relocate. Measured, so this is safe rather than merely tidy:
  * PYTHONNOUSERSITE=1 is exported on the line above it,
  * PYTHONNOUSERSITE is on FS_ENV_ALLOWLIST, so it crosses on BOTH arms, and
  * the backend's singularity arm exports SINGULARITYENV_PYTHONNOUSERSITE=1
    itself, unconditionally, inside run_in_container.
So the launcher line changes nothing on either arm. What it DOES do is teach the
next reader that naming a runtime in the launcher is normal -- which is how
#109, #117 and #122 each happened. Deleting it costs nothing and removes the
example.

--- D3 -----------------------------------------------------------------------
MASTER_PORT was allowlisted, consumed (fs_launch_python does ${MASTER_PORT:?...}),
and produced by nothing. It has not fired only because fs_launch_python currently
has zero call sites (#124). Minting it now means fixing #124 cannot detonate it.

WHERE: the converged tail of fs_backend_init, after `export FS_BACKEND ...`.
Both paths reach it -- the Slurm path (SLURM_JOB_ID supplied by the scheduler,
which the launcher has already proved non-empty) and the off-Slurm path (which
mints SLURM_JOB_ID a few lines earlier). Deriving it beside MASTER_ADDR would
have covered only the off-Slurm branch, which is the half that already works.

HOW: `29400 + SLURM_JOB_ID % 1000`, the form the backend's own comment already
names as the convention the GB200 launchers use. That comment describes it as
something "both launchers do" -- and the measured truth is that this launcher
does not, which is the whole defect. Deriving it from the job id is what gives
the collision-avoidance property the comment claims: two concurrent jobs on one
tray get different ports without coordinating.

FAIL CLOSED on a non-numeric job id rather than defaulting. A default port is
exactly the failure mode worth avoiding: two jobs silently sharing a rendezvous
port produce a hang or a cross-wired process group, not an error.

GATES
  F1 idempotent
  F2/F3 anchors unique
  F4 bash -n clean on both patched files
  F5 INTEGRATION: gate_env_drift.py must go GREEN on the patched pair. This is
     the gate that found the defects; it is also the proof they are gone.
  F6 the derivation is EXECUTED:
       MUST_PASS  a known job id yields the arithmetically expected port
       MUST_PASS  two different job ids yield DIFFERENT ports (the property the
                  mint exists for -- a constant would satisfy every other row)
       MUST_PASS  an operator-set MASTER_PORT is respected, not overwritten
       MUST_FIRE  a non-numeric job id is REFUSED, not defaulted
"""

from __future__ import annotations

import importlib.util
import pathlib
import re
import subprocess
import sys
import tempfile

GEN = pathlib.Path(__file__).resolve().parent / "h100" / "gen"
BACKEND = GEN / "fs_container_backend.bound.sh"
LAUNCH = GEN / "launch_fs_h100.fixed.sh"
MARK = "fs116:"

BACK_ANCHOR = "  export FS_BACKEND FS_CONTAINER_RUNTIME FS_ALLOCATION FS_USE_TORCHRUN\n}\n"

BACK_BLOCK = '''  # --- fs116: mint MASTER_PORT ------------------------------------------------
  # It was on FS_ENV_ALLOWLIST and consumed by fs_launch_python's
  # ${MASTER_PORT:?...} while being produced by NOTHING in either file. That is
  # an allowlist entry making a claim about the environment that nothing backs.
  #
  # Minted HERE, at the converged tail of fs_backend_init, because both paths
  # reach it: under Slurm the scheduler supplies SLURM_JOB_ID (the launcher has
  # already refused an empty one), and off-Slurm it was minted a few lines above.
  # Deriving it next to MASTER_ADDR would have covered only the off-Slurm branch.
  #
  # 29400 + jobid % 1000 is the convention this file's own comment names. Deriving
  # from the job id is what supplies the collision-avoidance it claims: two
  # concurrent jobs on one tray land on different ports without coordinating.
  #
  # Fail closed on a non-numeric id. A fallback port is the bad outcome here --
  # two jobs sharing a rendezvous port hang or cross-wire their process groups
  # instead of erroring, and the symptom appears in NCCL, far from the cause.
  if [[ -z "${MASTER_PORT:-}" ]]; then
    [[ "${SLURM_JOB_ID:-}" =~ ^[0-9]+$ ]] || fs_die "fs_backend_init: cannot derive MASTER_PORT — SLURM_JOB_ID is '${SLURM_JOB_ID:-<unset>}', not numeric. Refusing to fall back to a fixed port: two jobs sharing a rendezvous port hang or cross-wire instead of failing."
    MASTER_PORT=$(( 29400 + SLURM_JOB_ID % 1000 ))
  fi
  [[ "$MASTER_PORT" -ge 1024 && "$MASTER_PORT" -le 65535 ]] || fs_die "fs_backend_init: MASTER_PORT '$MASTER_PORT' outside the unprivileged range"
  export MASTER_PORT

  export FS_BACKEND FS_CONTAINER_RUNTIME FS_ALLOCATION FS_USE_TORCHRUN
}
'''

LAUNCH_ANCHOR = "export PYTHONNOUSERSITE=1\nexport SINGULARITYENV_PYTHONNOUSERSITE=1\n"

LAUNCH_BLOCK = (
    "export PYTHONNOUSERSITE=1\n"
    "# fs116: the SINGULARITYENV_PYTHONNOUSERSITE export that stood here is DELETED.\n"
    "# Measured to be a no-op: PYTHONNOUSERSITE is on FS_ENV_ALLOWLIST so it already\n"
    "# crosses on both arms, and the backend's singularity arm exports the\n"
    "# SINGULARITYENV_ form itself inside run_in_container. What it actually did was\n"
    "# demonstrate that naming a runtime in the launcher is acceptable -- the habit\n"
    "# behind three separate runtime-divergence defects.\n"
)

DRILL = r"""
set -uo pipefail
fs_die() { printf 'REFUSED: %%s\n' "$*" >&2; exit 1; }
derive() {
  if [[ -z "${MASTER_PORT:-}" ]]; then
    [[ "${SLURM_JOB_ID:-}" =~ ^[0-9]+$ ]] || fs_die "not numeric"
    MASTER_PORT=$(( 29400 + SLURM_JOB_ID %% 1000 ))
  fi
  [[ "$MASTER_PORT" -ge 1024 && "$MASTER_PORT" -le 65535 ]] || fs_die "out of range"
  printf '%%s\n' "$MASTER_PORT"
}
%s
"""


def _bash(body: str, env: dict) -> subprocess.CompletedProcess[str]:
    import os
    e = {k: v for k, v in os.environ.items() if k not in ("MASTER_PORT", "SLURM_JOB_ID")}
    e.update(env)
    return subprocess.run(["bash", "-c", DRILL % body], capture_output=True, text=True, env=e)


def _syntax(text: str, label: str) -> bool:
    with tempfile.NamedTemporaryFile("w", suffix=".sh", delete=False) as fh:
        fh.write(text); tmp = fh.name
    r = subprocess.run(["bash", "-n", tmp], capture_output=True, text=True)
    if r.returncode:
        print(f"  FAIL F4  bash -n {label}: {r.stderr.strip()[:250]}", file=sys.stderr)
        return False
    return True


def main() -> int:
    back = BACKEND.read_text("utf-8")
    lau = LAUNCH.read_text("utf-8")

    if MARK in back and MARK in lau:                                          # F1
        print("  F1  already applied to both — no-op (idempotent)")
        return 0
    if (MARK in back) != (MARK in lau):
        print(f"  FAIL F1  half-applied: backend={MARK in back} launcher={MARK in lau}",
              file=sys.stderr)
        return 5
    print("  PASS F1  neither file patched yet")

    ok = True
    for gate, name, text, anchor in (("F2", "backend mint site", back, BACK_ANCHOR),
                                     ("F3", "launcher delete site", lau, LAUNCH_ANCHOR)):
        n = text.count(anchor)
        if n != 1:
            print(f"  FAIL {gate}  {name} anchor occurs {n}x (need 1)", file=sys.stderr)
            ok = False
        else:
            print(f"  PASS {gate}  {name} anchor unique")
    if not ok:
        return 5

    back_new = back.replace(BACK_ANCHOR, BACK_BLOCK, 1)
    lau_new = lau.replace(LAUNCH_ANCHOR, LAUNCH_BLOCK, 1)

    if not (_syntax(back_new, BACKEND.name) & _syntax(lau_new, LAUNCH.name)):  # F4
        ok = False
    else:
        print("  PASS F4  bash -n clean on both")

    # --- F6 execute the derivation -----------------------------------------
    rows = []
    r = _bash("derive", {"SLURM_JOB_ID": "12345"})
    rows.append(("MUST_PASS derived port", r.stdout.strip() == "29745", r.stdout.strip()))
    # 12888 % 1000 == 888 -> 30288. Pinning the arithmetic literally, not just
    # asserting inequality: two ids differing only above the modulus (12345 vs
    # 13345) MUST collide, and a row that only checked "different" would call
    # that a pass while hiding a real same-tray collision case.
    r2 = _bash("derive", {"SLURM_JOB_ID": "12888"})
    rows.append(("MUST_PASS distinct ids -> distinct ports",
                 r2.stdout.strip() == "30288" and r2.stdout.strip() != r.stdout.strip(),
                 f"{r.stdout.strip()}/{r2.stdout.strip()}"))
    r5 = _bash("derive", {"SLURM_JOB_ID": "13345"})
    rows.append(("KNOWN LIMIT ids congruent mod 1000 collide (documented, not fixed: "
                 "Slurm ids on one tray are near-consecutive, so the 1000-wide window "
                 "is the convention's stated scope)",
                 r5.stdout.strip() == r.stdout.strip(), r5.stdout.strip()))
    r3 = _bash("derive", {"SLURM_JOB_ID": "1", "MASTER_PORT": "31000"})
    rows.append(("MUST_PASS operator override respected", r3.stdout.strip() == "31000",
                 r3.stdout.strip()))
    r4 = _bash("derive", {"SLURM_JOB_ID": "notanid"})
    rows.append(("MUST_FIRE non-numeric refused",
                 r4.returncode == 1 and "REFUSED" in r4.stderr, f"rc={r4.returncode}"))
    bad = [(n, got) for n, good, got in rows if not good]
    if bad:
        print(f"  FAIL F6  {len(bad)} of {len(rows)} control rows wrong: {bad}",
              file=sys.stderr)
        ok = False
    else:
        print(f"  PASS F6  {len(rows)}/{len(rows)} control rows: derivation correct, "
              f"job-id-dependent, override-respecting, and fails closed on garbage")

    if not ok:
        print("\nREFUSING TO WRITE — gates above are red", file=sys.stderr)
        return 5

    BACKEND.write_text(back_new, "utf-8")
    LAUNCH.write_text(lau_new, "utf-8")

    # --- F5 integration: the gate that found this must now be green ---------
    spec = importlib.util.spec_from_file_location(
        "gate_env_drift", str(pathlib.Path(__file__).resolve().parent / "gate_env_drift.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    viol = mod.check(lau_new, back_new, quiet=True)
    if viol:
        print(f"  FAIL F5  gate_env_drift still red after the patch: {viol}",
              file=sys.stderr)
        print("  (files were written; re-run gate_env_drift.py for the full report)",
              file=sys.stderr)
        return 5
    print("  PASS F5  gate_env_drift.py GREEN on the patched pair "
          "(it was RED on both D1 and D3 before this patch)")
    print(f"\nALL GATES GREEN -> {BACKEND.name}, {LAUNCH.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
