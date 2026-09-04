#!/usr/bin/env python3
"""#125: `wlm` launch mode is unimplementable off-Slurm, and a minted literal hides it.

This began as "the off-Slurm mint hard-codes SLURM_NTASKS=4, a GB200 tray fact". Measuring
it turned it into something worse: it is #124 reopened through the off-Slurm door.

THE MEASUREMENT
  backend :888   `--slurm-ntasks : srun-only`
  backend :1172, :1263, :1286   every srun call site is inside `if [[ "$FS_ALLOCATION" == slurm ]]`
  backend :362-363  off-Slurm the backend mints SLURM_NTASKS=4 / SLURM_NTASKS_PER_NODE=4
  backend :614   fs_compose_launch's wlm arm reads SLURM_NTASKS_PER_NODE and, if it equals
                 the measured gpu count, returns the BARE engine command

So on FS_ALLOCATION=local: the launcher passes --slurm-ntasks 8, run_in_container has no srun
to give it to and DISCARDS it, and nothing forks the ranks. One process starts on N GPUs --
precisely the defect #124 exists to prevent -- while every gate reports green.

The minted literal is what supplies the fake evidence:

  off-Slurm 8-GPU node   tpn=4  -> (( 4 == 8 )) fails -> caught, but only because a GB200
                                 tray literal happens to differ from this node's gpu count
  off-Slurm 4-GPU node   tpn=4  -> (( 4 == 4 )) passes -> ONE process on four GPUs, green

The guard is real; the value it checks against is fabricated. A comparison between a measured
count and a minted constant is not a measurement, and it passes exactly where the constant
was true when someone typed it.

THE FIX, and deliberately not the bigger one
  wlm mode is a CATEGORY ERROR off-Slurm, not a mismatch. There is no workload manager to
  supply tasks, by construction -- that is what FS_ALLOCATION=local MEANS. So the arm refuses
  on the allocation axis FIRST, before it reads a task count at all, and the message names
  torchrun as the mode that actually works there.

  Checking allocation before reading the count is the load-bearing ordering. Reversed, a
  4-GPU off-Slurm node still passes the numeric row and only then hits the allocation row --
  and the numeric row's success would be recorded on the way past.

WHAT THIS DOES NOT FIX, stated rather than quietly left
  The literal 4 remains at :362-363. Within these two files it is now dead: SLURM_NTASKS has
  ZERO readers, and SLURM_NTASKS_PER_NODE's only reader is the arm this patch refuses
  off-Slurm. Its comment claims "full-FT geometry divides by it", and that consumer is not in
  this tree -- it would be in the GB200 launchers, which are separate files on the cluster.
  Deleting a variable whose external readers I cannot enumerate would be trading a closed
  defect for an unmeasured one, so the mint is annotated, not removed. #125 stays open for
  that half, scoped to what it actually is.

GATES
  W1 idempotent
  W2 anchor unique
  W3 bash -n clean
  W4 EXECUTED rows against the real composer, sourced from the patched file -- the whole
     claim is a runtime refusal, so a text assertion proves nothing:
       MUST_FIRE  local + tpn==gpus (the 4-GPU node that used to PASS) -> refused
       MUST_FIRE  local + tpn!=gpus                                     -> refused, and on the
                  ALLOCATION ground, not the arithmetic one (ordering is load-bearing)
       MUST_FIRE  local + tpn unset                                     -> refused
       MUST_PASS  slurm + tpn==gpus                                     -> composes 'wlm'
       MUST_FIRE  slurm + tpn!=gpus                                     -> still refused (the
                  pre-existing #124 guard must survive this patch)
       MUST_PASS  local + torchrun mode                                 -> still composes, so
                  the patch has not made the off-Slurm path unlaunchable, only honest
  W5 annotation present at the mint
"""

from __future__ import annotations

import pathlib
import re
import subprocess
import sys
import tempfile

GEN = pathlib.Path(__file__).resolve().parent / "h100" / "gen"
BACKEND = GEN / "fs_container_backend.bound.sh"
MARK = "fs125:"

ANCHOR = "      local tpn=${SLURM_NTASKS_PER_NODE:-}\n"

REPL = (
    "      # fs125: allocation FIRST, before any task count is read. Off-Slurm there is no\n"
    "      # workload manager to fork ranks -- srun is absent by construction (--slurm-ntasks\n"
    "      # is srun-only and run_in_container DISCARDS it on the local path), so this arm\n"
    "      # would return the bare command and start ONE process on N GPUs. The minted\n"
    "      # SLURM_NTASKS_PER_NODE=4 made that look checked: on a 4-GPU off-Slurm node the\n"
    "      # numeric row below passes against a constant. Ordering is load-bearing -- reversed,\n"
    "      # that node clears the arithmetic row first and the pass is recorded on the way by.\n"
    '      [[ "${FS_ALLOCATION:-}" == slurm ]] || fs_die "fs_compose_launch wlm: '
    "FS_ALLOCATION='${FS_ALLOCATION:-<unset>}', not slurm. wlm mode delegates rank creation to "
    "the workload manager, and off-Slurm there is none -- srun is absent by construction, so "
    "--slurm-ntasks is discarded and ONE process would start on $gpus devices. Use "
    'FS_ENGINE_LAUNCH_MODE=torchrun off-Slurm."\n'
    "      local tpn=${SLURM_NTASKS_PER_NODE:-}\n"
)

MINT_ANCHOR = (
    "    export SLURM_NTASKS=4                        "
    "# 1 tray x 4 GB200; full-FT geometry divides by it\n"
)
MINT_REPL = (
    "    # fs125: these two are a GB200 tray fact minted on EVERY off-Slurm node, whatever its\n"
    "    # shape. In this tree they are now dead -- SLURM_NTASKS has zero readers, and\n"
    "    # SLURM_NTASKS_PER_NODE's only reader is fs_compose_launch's wlm arm, which now refuses\n"
    "    # off-Slurm before reading it. The 'full-FT geometry divides by it' consumer is NOT in\n"
    "    # this tree; it would be in the GB200 launchers. Left in place rather than deleted\n"
    "    # because removing a variable whose external readers cannot be enumerated from here\n"
    "    # trades a closed defect for an unmeasured one. Deriving it from nvidia-smi was\n"
    "    # considered and rejected: the composer's check would then compare nvidia-smi to\n"
    "    # itself, which is a tautology wearing a guard's clothes.\n"
    "    export SLURM_NTASKS=4                        "
    "# GB200 tray literal — see fs125 note above\n"
)

# W4: source the patched backend in a stubbed shell and drive fs_compose_launch directly.
DRILL = r"""
set -uo pipefail
fs_die() { printf 'REFUSED: %s\n' "$*" >&2; exit 1; }
export MASTER_ADDR=127.0.0.1 MASTER_PORT=29500
source "$BACKEND_PATH" 2>/dev/null || true
fs_compose_launch "$MODE" "$GPUS" "python train.py"
"""


def _row(mode: str, gpus: str, alloc: str, tpn: str | None, path: str):
    import os
    e = {k: v for k, v in os.environ.items()
         if k not in ("SLURM_NTASKS_PER_NODE", "FS_ALLOCATION", "FS_ENGINE_PROCS_PER_NODE")}
    e.update({"BACKEND_PATH": path, "MODE": mode, "GPUS": gpus, "FS_ALLOCATION": alloc,
              "SLURM_NNODES": "1", "SLURM_NODEID": "0"})
    if tpn is not None:
        e["SLURM_NTASKS_PER_NODE"] = tpn
    return subprocess.run(["bash", "-c", DRILL], capture_output=True, text=True, env=e)


def main() -> int:
    back = BACKEND.read_text("utf-8")

    if MARK in back:
        print("  W1  already applied — no-op (idempotent)")
        return 0
    print("  PASS W1  not yet applied")

    ok = True
    for gate, label, anc in (("W2a", "wlm arm", ANCHOR), ("W2b", "mint site", MINT_ANCHOR)):
        n = back.count(anc)
        if n != 1:
            print(f"  FAIL {gate}  {label} anchor occurs {n}x (need 1)", file=sys.stderr)
            ok = False
        else:
            print(f"  PASS {gate}  {label} anchor unique")
    if not ok:
        return 5

    new = back.replace(ANCHOR, REPL, 1).replace(MINT_ANCHOR, MINT_REPL, 1)

    with tempfile.NamedTemporaryFile("w", suffix=".sh", delete=False) as fh:
        fh.write(new)
        tmp = fh.name
    r = subprocess.run(["bash", "-n", tmp], capture_output=True, text=True)
    if r.returncode:
        print(f"  FAIL W3  bash -n: {r.stderr.strip()[:250]}", file=sys.stderr)
        return 5
    print("  PASS W3  bash -n clean")

    rows = []
    # The row that used to pass and must not: a 4-GPU off-Slurm node.
    x = _row("wlm", "4", "local", "4", tmp)
    rows.append(("MUST_FIRE local tpn==gpus (the silent 1-process case)",
                 x.returncode != 0 and "not slurm" in x.stderr, x.stderr.strip()[:70]))
    # Ordering: must die on allocation, NOT on arithmetic.
    x = _row("wlm", "8", "local", "4", tmp)
    rows.append(("MUST_FIRE local tpn!=gpus refused on ALLOCATION grounds, not arithmetic",
                 x.returncode != 0 and "not slurm" in x.stderr
                 and "!= measured gpus" not in x.stderr, x.stderr.strip()[:70]))
    x = _row("wlm", "8", "local", None, tmp)
    rows.append(("MUST_FIRE local tpn unset", x.returncode != 0 and "not slurm" in x.stderr,
                 x.stderr.strip()[:70]))
    x = _row("wlm", "8", "slurm", "8", tmp)
    rows.append(("MUST_PASS slurm tpn==gpus composes",
                 x.returncode == 0 and x.stdout.startswith("wlm\t"), x.stdout.strip()[:70]))
    x = _row("wlm", "8", "slurm", "4", tmp)
    rows.append(("MUST_FIRE slurm tpn!=gpus (the #124 guard survives)",
                 x.returncode != 0 and "!= measured gpus" in x.stderr, x.stderr.strip()[:70]))
    x = _row("torchrun", "8", "local", None, tmp)
    rows.append(("MUST_PASS local torchrun still composes (path not bricked)",
                 x.returncode == 0 and x.stdout.startswith("composed\t"),
                 x.stdout.strip()[:70]))

    bad = [(n, got) for n, good, got in rows if not good]
    if bad:
        print(f"  FAIL W4  {len(bad)} of {len(rows)} rows wrong:", file=sys.stderr)
        for n, got in bad:
            print(f"           {n}\n             got: {got}", file=sys.stderr)
        return 5
    print(f"  PASS W4  {len(rows)}/{len(rows)} rows EXECUTED against the patched composer")

    if "fs125" not in new.split("export SLURM_NTASKS=4")[0][-900:]:
        print("  FAIL W5  mint annotation missing", file=sys.stderr)
        return 5
    print("  PASS W5  mint annotated with its measured reader count and the rejected alternative")

    BACKEND.write_text(new, "utf-8")
    print(f"\nALL GATES GREEN -> {BACKEND.name}")
    print("PARTIAL: #125 stays open for the literal itself — external readers are not "
          "enumerable from this tree.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
