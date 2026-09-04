#!/usr/bin/env python3
"""#128: never invoke the rank launcher by name — bind it to the trainer's interpreter.

MEASURED on the real estate against the real training image, not inferred:

  host ~/.local/bin/torchrun shebang   #!<estate>/anaconda/envs/vllm_py312/bin/python3.12
  inside the hardened container (--no-home --cleanenv, PYTHONNOUSERSITE=1):
      command -v torchrun               -> /home/<user>/.local/bin/torchrun   (HOST)
      torchrun ...                      -> "cannot execute: required file not found"
      python3 -m torch.distributed.run  -> runs, torch 2.9.0a0+...nv25.09     (CONTAINER)

fs_compose_launch's torchrun arm emitted a bare `torchrun`. On this estate that name resolves
to a host script whose shebang names a host anaconda interpreter.

WHY THIS IS A BLOCKER RATHER THAN A NUISANCE — the remediations interact.
Today it fails loudly, and only by accident: the shebang interpreter lives under the estate
root, and that root is not mounted into the container, so the exec dies. #117's fix binds the
estate root so the model is visible. The moment it lands, that interpreter becomes resolvable
INSIDE the container and `torchrun` starts SUCCEEDING — on host anaconda python, importing
whatever torch that environment holds, while the trainer it forks imports the container's
torch. A hard failure quietly becomes a version split across the launcher/trainer boundary,
which is the kind of thing that surfaces as an incomprehensible NCCL or pickle error days
later. Fixing #117 re-arms #128; that is the reason to close it now rather than after.

Worth recording separately: `--no-home` did NOT hide the home directory (the user's .bashrc
still sourced and ~/.local/bin stayed on PATH). The hardening that closes #107's *import*
leak does not close the *binary lookup* leak. Two surfaces, one previously closed.

THE FIX IS AN ABSTRACTION, NOT AN ESTATE WORKAROUND.
Emit `<interpreter> -m torch.distributed.run`. The rule it encodes — the process that forks
the ranks must be the same interpreter that will import torch — is a general property of
distributed launch. It is true on any estate, under any container runtime, for any model. The
estate-specific fact (this cluster has a stray torchrun on PATH) is merely what made it
visible. Nothing here is conditioned on the cluster.

The interpreter is `${FS_PYTHON:-python3}`: overridable, but defaulting to the same bare
`python3` token the launcher ALREADY relies on at :386 to measure the device count with
`python3 -c 'import torch; print(torch.cuda.device_count())'`. That consistency is the point.
If those two tokens can resolve differently, the count that authorises the launch and the
torch that performs it came from different interpreters, and #124's measurement stops meaning
what it says. Defaulting rather than requiring is deliberate: this is not a policy variable
like FS_ENGINE_LAUNCH_MODE (where silence must not be an answer), it is a tool path with one
obviously correct in-container value, and a required-no-default here would add friction
without adding a decision.

The two `echo` banners that advertise "training will run via torchrun --nproc_per_node=N" are
corrected too. A banner describing a command the code no longer emits is a small lie aimed
squarely at whoever is debugging the launch.

GATES
  P1 idempotent
  P2 all three anchors unique
  P3 bash -n clean
  P4 EXECUTED against the patched composer:
       MUST_PASS  torchrun mode composes '-m torch.distributed.run'
       MUST_FIRE  the composed command contains NO bare `torchrun` token  <- the actual claim
       MUST_PASS  FS_PYTHON override is honoured
       MUST_PASS  nproc_per_node still carries the MEASURED gpu count (#124 must survive)
       MUST_PASS  the MASTER_ADDR/MASTER_PORT :? guards still fire when unset
  P5 no bare-`torchrun` emission survives anywhere outside mode names and comments — a
     file-wide sweep, because fixing one printf while another site still emits the name
     would satisfy every row above and change nothing where it mattered.
"""

from __future__ import annotations

import pathlib
import re
import subprocess
import sys
import tempfile

GEN = pathlib.Path(__file__).resolve().parent / "h100" / "gen"
BACKEND = GEN / "fs_container_backend.bound.sh"
MARK = "fs128:"

ANCHOR_CMD = (
    "      printf 'composed\\ttorchrun --nproc_per_node=%s --nnodes=%s --node_rank=%s "
    "--master_addr=%s --master_port=%s %s\\n' \\\n"
)
REPL_CMD = (
    "      # fs128: '-m torch.distributed.run', never the bare `torchrun` name. Measured on\n"
    "      # this estate: `torchrun` resolves to a HOST script whose shebang is a host anaconda\n"
    "      # python. It fails loudly only because that interpreter is not mounted -- and #117's\n"
    "      # estate-root bind makes it resolvable, at which point it would SUCCEED on the wrong\n"
    "      # interpreter and split the torch version across the launcher/trainer boundary.\n"
    "      # The general rule, independent of this estate: whatever forks the ranks must be the\n"
    "      # same interpreter that imports torch. ${FS_PYTHON:-python3} is the same token the\n"
    "      # launcher uses to MEASURE the device count, so the count that authorises the launch\n"
    "      # and the torch that performs it cannot come from two different interpreters.\n"
    "      printf 'composed\\t%s -m torch.distributed.run --nproc_per_node=%s --nnodes=%s "
    "--node_rank=%s --master_addr=%s --master_port=%s %s\\n' \\\n"
    '        "${FS_PYTHON:-python3}" \\\n'
)

BANNERS = [
    ("backend: container '$ENROOT_NAME' ready; training will run via torchrun "
     "--nproc_per_node=$gpus inside ONE enroot start",
     "backend: container '$ENROOT_NAME' ready; training will run via "
     "\\\"\\${FS_PYTHON:-python3} -m torch.distributed.run --nproc_per_node=$gpus\\\" "
     "inside ONE enroot start (fs128: rank launcher bound to the trainer's interpreter)"),
    ("backend: singularity image '$sqsh' ready; training will run via torchrun "
     "--nproc_per_node=$gpus inside singularity exec",
     "backend: singularity image '$sqsh' ready; training will run via "
     "\\\"\\${FS_PYTHON:-python3} -m torch.distributed.run --nproc_per_node=$gpus\\\" "
     "inside singularity exec (fs128: rank launcher bound to the trainer's interpreter)"),
]

DRILL = r"""
set -uo pipefail
fs_die() { printf 'REFUSED: %s\n' "$*" >&2; exit 1; }
source "$BACKEND_PATH" 2>/dev/null || true
fs_compose_launch "$MODE" "$GPUS" "python train.py --cfg a.yaml"
"""


def _row(gpus="8", env=None, path=""):
    import os
    e = {k: v for k, v in os.environ.items() if k not in ("FS_PYTHON", "MASTER_ADDR", "MASTER_PORT")}
    e.update({"BACKEND_PATH": path, "MODE": "torchrun", "GPUS": gpus,
              "MASTER_ADDR": "10.0.0.1", "MASTER_PORT": "29745",
              "SLURM_NNODES": "1", "SLURM_NODEID": "0", "FS_ALLOCATION": "slurm"})
    e.update(env or {})
    for k in [k for k, v in (env or {}).items() if v is None]:
        e.pop(k, None)
    return subprocess.run(["bash", "-c", DRILL], capture_output=True, text=True, env=e)


def main() -> int:
    back = BACKEND.read_text("utf-8")

    if MARK in back:
        print("  P1  already applied — no-op (idempotent)")
        return 0
    print("  PASS P1  not yet applied")

    anchors = [("P2a", "composed printf", ANCHOR_CMD)] + [
        (f"P2b{i}", f"banner {i}", a) for i, (a, _) in enumerate(BANNERS)
    ]
    ok = True
    for gate, label, anc in anchors:
        n = back.count(anc)
        if n != 1:
            print(f"  FAIL {gate}  {label} anchor occurs {n}x (need 1)", file=sys.stderr)
            ok = False
        else:
            print(f"  PASS {gate}  {label} anchor unique")
    if not ok:
        return 5

    new = back.replace(ANCHOR_CMD, REPL_CMD, 1)
    for a, b in BANNERS:
        new = new.replace(a, b, 1)

    with tempfile.NamedTemporaryFile("w", suffix=".sh", delete=False) as fh:
        fh.write(new)
        tmp = fh.name
    r = subprocess.run(["bash", "-n", tmp], capture_output=True, text=True)
    if r.returncode:
        print(f"  FAIL P3  bash -n: {r.stderr.strip()[:250]}", file=sys.stderr)
        return 5
    print("  PASS P3  bash -n clean")

    rows = []
    x = _row(path=tmp)
    out = x.stdout.strip()
    rows.append(("MUST_PASS composes -m torch.distributed.run",
                 x.returncode == 0 and "-m torch.distributed.run" in out, out[:80]))
    # The actual claim: no bare `torchrun` token survives into the command.
    cmd = out.split("\t", 1)[1] if "\t" in out else out
    rows.append(("MUST_FIRE no bare `torchrun` token in the composed command",
                 re.search(r"(?<![\w./-])torchrun(?![\w.-])", cmd) is None, cmd[:80]))
    rows.append(("MUST_PASS nproc_per_node carries the MEASURED gpu count (#124 survives)",
                 "--nproc_per_node=8" in out, out[:80]))
    x = _row(env={"FS_PYTHON": "/opt/venv/bin/python3"}, path=tmp)
    rows.append(("MUST_PASS FS_PYTHON override honoured",
                 "/opt/venv/bin/python3 -m torch.distributed.run" in x.stdout,
                 x.stdout.strip()[:80]))
    x = _row(env={"MASTER_ADDR": None}, path=tmp)
    rows.append(("MUST_PASS MASTER_ADDR :? guard still fires",
                 x.returncode != 0 and "MASTER_ADDR" in x.stderr, f"rc={x.returncode}"))
    x = _row(env={"MASTER_PORT": None}, path=tmp)
    rows.append(("MUST_PASS MASTER_PORT :? guard still fires",
                 x.returncode != 0 and "MASTER_PORT" in x.stderr, f"rc={x.returncode}"))

    bad = [(n, got) for n, good, got in rows if not good]
    if bad:
        print(f"  FAIL P4  {len(bad)} of {len(rows)} rows wrong:", file=sys.stderr)
        for n, got in bad:
            print(f"           {n}\n             got: {got}", file=sys.stderr)
        return 5
    print(f"  PASS P4  {len(rows)}/{len(rows)} rows EXECUTED against the patched composer")

    # P5 file-wide sweep. Legitimate survivors: the mode NAME (torchrun|wlm|self) and comments.
    survivors = []
    for i, ln in enumerate(new.splitlines(), 1):
        code = ln.split("#", 1)[0]
        if not re.search(r"(?<![\w./-])torchrun(?![\w.-])", code):
            continue
        if re.search(r"torchrun\|wlm\|self|mode.*torchrun|^\s*torchrun\)", code):
            continue
        survivors.append(f"L{i}: {code.strip()[:70]}")
    if survivors:
        print(f"  FAIL P5  {len(survivors)} bare-torchrun emission(s) survive:", file=sys.stderr)
        for s in survivors:
            print(f"           {s}", file=sys.stderr)
        return 5
    print("  PASS P5  file-wide sweep: 0 bare-torchrun emissions outside mode names/comments")

    BACKEND.write_text(new, "utf-8")
    print(f"\nALL GATES GREEN -> {BACKEND.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
