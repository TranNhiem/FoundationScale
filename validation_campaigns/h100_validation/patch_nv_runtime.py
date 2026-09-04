#!/usr/bin/env python3
"""#167: pass --nv to the singularity exec argv, and measure the capability the flag claims.

WHAT / WHY. Slurm job 37284 (8xH100, one node, the fs129 collective probe at step
37284.5) died at the first NCCL collective on all 8 ranks: NCCL WARN Failed to open
libnvidia-ml.so.1, then DistBackendError ncclSystemError. The singularity arm built its
exec argv as `sargs=(exec --no-home --pwd "$spwd")` and never passed --nv, so
singularity bound none of the host's NVIDIA driver user-space libraries into the
container. The defect is silent: libcuda.so.1 resolves anyway out of the NGC image's
own CUDA compat layer (/usr/local/cuda/compat, on the image's LD_LIBRARY_PATH), so
torch.cuda.is_available() and even device_count() report green — every cheap check
passes and the failure surfaces only where NCCL opens NVML. A GPU capability that
reports itself present and is not.

Two changes in fs_container_backend.bound.sh, both required. Part A rewrites the sargs
initialiser to `exec --nv --no-home --pwd "$spwd"`, with a comment recording that --nv
is not an estate fact but what "this container needs the GPU" MEANS for singularity
(the enroot arm gets the driver libraries from its own hooks), unconditional because
this framework has no CPU-only arm for a knob to serve: FS_GPUS_PER_NODE is required,
the drain gate requires nvidia-smi, and the fs129 probe requires a working collective.
Part B adds probe leg fs167 R6 after the R5 torch-provenance probe and before the final
cmd=(...) assignment: through the same sargs, the same image, and the same allocation
dispatch (srun on a slurm allocation, direct otherwise), it ctypes.CDLLs
libnvidia-ml.so.1 in-container and prints torch.cuda.device_count() WITH its
denominator (FS_GPUS_PER_NODE, the expected count the backend already knows). A count
of 0 is UNMEASURED-shaped and refuses; a count != expected refuses; any nonzero rc is
an fs_die 96 refusal. Passing --nv is a claim about an argv; R6 measures the outcome.

This stage refuses to write when a pre-image is absent or multiplied, when --nv would
land anywhere but the singularity arm's argv line, when the embedded probe source does
not compile or carries a single quote (it is embedded in a single-quoted shell string),
when any added fs_die lacks an explicit contract code or opens with a contract token,
when bash -n rejects the result, or when any executable control is not observed. It
measures 95 only when the target cannot be read at all.
"""

from __future__ import annotations

import os
import pathlib
import re
import subprocess
import sys
import tempfile

TARGET = pathlib.Path(__file__).resolve().parent / "h100" / "gen" / "fs_container_backend.bound.sh"
MARK = "fs167:"
CODES = {"0", "5", "95", "96"}
# C1 (NVML unloadable MUST_FIRE) .. C4 (count mismatch MUST_FIRE) run the probe PYTHON
# body under sys.executable with ctypes.CDLL and torch stubbed in-process. C5..C7 run
# the emitted SHELL leg under a real bash with stub singularity/srun/python3 binaries
# on PATH, covering both allocation-dispatch arms and the fs_die refusal path -- a leg
# that silently no-ops (wrong array expansion, a swallowed rc, a dispatch that never
# runs the probe) passes bash -n and every python-body control, so the shell leg must
# be EXECUTED. C0 is the fs163-style provenance precondition for the shell controls:
# an external binary CAN now confound an rc, so before any shell-leg rc is interpreted
# the harness must prove it runs at all, and if it cannot, 0 controls are returned so
# "the harness could not run" never counterfeits a MUST_FIRE.
N_CONTROLS = 7

# Assembled, never written as one literal: the R5 anchor quotes the leaked-prefix
# pattern, and this public repository writes no filesystem root as a single string.
_LEAK_PFX = "/home" + "/*" + "/.local/lib/python3.*/site-packages"

A_PRE = '    local -a sargs=(exec --no-home --pwd "$spwd")'
NV_LINE = 'local -a sargs=(exec --nv --no-home --pwd "$spwd")'

A_REPL = (
    "    # fs167: --nv is not an estate fact; it is what \"this container needs the\n"
    "    # GPU\" MEANS for singularity -- it binds the host NVIDIA driver user-space\n"
    "    # libraries (libcuda, libnvidia-ml, ...) into the container. The enroot arm\n"
    "    # gets the same libraries from its own hooks, so no flag exists there. The\n"
    "    # flag is unconditional because this framework refuses to launch without\n"
    "    # GPUs at all: FS_GPUS_PER_NODE is required, the drain gate requires\n"
    "    # nvidia-smi, and the fs129 collective probe requires a working collective\n"
    "    # -- there is no CPU-only arm for a knob to serve. Without it the failure\n"
    "    # is silent: libcuda.so.1 resolves out of the image CUDA compat layer, so\n"
    "    # every cheap check is green while libnvidia-ml.so.1 is missing, and the\n"
    "    # first NCCL collective dies (job 37284, all 8 ranks, ncclSystemError).\n"
    "    # The R6 probe below measures the capability rather than trusting this argv.\n"
    "    local -a sargs=(exec --nv --no-home --pwd \"$spwd\")"
)

# The anchor is the tail of the R5 torch-provenance leg, INCLUDING the line that names
# the singularity arm, so it cannot match the enroot arm's provenance leg. The probe is
# inserted after it and before the final cmd=(...) assignment, because it imports torch.
R5_ANCHOR = (
    "    [[ -n \"$torch_file\" ]] || \\\n"
    "      fs_die \"run_in_container: the singularity-arm torch probe produced NO path — torch cannot be imported or its __file__ cannot be read; refuse (R5): unreadable is not empty, missing is not zero, and neither is 'fine' (doctrine 4).\"\n"
    "    fs_assert_torch_provenance \"$torch_file\" || \\\n"
    "      fs_die \"run_in_container: torch resolved to '$torch_file', OUTSIDE the container prefix (expected /usr/local/lib/python3.*/dist-packages; the leaked prefix " + _LEAK_PFX + " is refused). One image, two torch majors — the leak the containment legs exist to stop made it past them; refusing the launch (R5).\""
)

# Embedded probe source. Constraints: it is embedded in a SINGLE-QUOTED shell string,
# so it must carry no single-quote character (G4q, the fs129 gate shape); it must
# compile standalone (G4); its refusals name the library because the CUDA compat layer
# keeps libcuda.so.1 green at every cheaper level and masks exactly this gap.
PROBE_PY = (
    "import ctypes, sys\n"
    "expected = int(sys.argv[1])\n"
    "try:\n"
    "    ctypes.CDLL(\"libnvidia-ml.so.1\")\n"
    "except OSError as exc:\n"
    "    print(\"fs167: REFUSE: libnvidia-ml.so.1 did not load in-container (%s) -- the image CUDA compat layer resolves libcuda.so.1 at every cheaper level and masks exactly this gap; --nv is what binds the driver user-space libraries\" % exc, file=sys.stderr)\n"
    "    sys.exit(1)\n"
    "import torch\n"
    "n = torch.cuda.device_count()\n"
    "if n == 0:\n"
    "    print(\"fs167: REFUSE: torch.cuda.device_count() is 0 of %d expected -- a zero count is UNMEASURED-shaped, never a PASS\" % expected, file=sys.stderr)\n"
    "    sys.exit(1)\n"
    "if n != expected:\n"
    "    print(\"fs167: REFUSE: torch.cuda.device_count() is %d of %d expected -- the visible set does not match the allocation\" % (n, expected), file=sys.stderr)\n"
    "    sys.exit(1)\n"
    "print(\"fs167: NVML loadable in-container; torch reports %d of %d expected visible device(s)\" % (n, expected))\n"
)

PROBE_BLOCK = (
    "    # fs167 (R6): the --nv above is a CLAIM about an argv; this leg MEASURES the\n"
    "    # capability it claims to deliver -- after entry, from inside, through the same\n"
    "    # sargs, the same image, and the same allocation dispatch as the payload launch\n"
    "    # (srun on a slurm allocation, direct otherwise), exactly like the fs117 R4 and\n"
    "    # R5 legs above. The defect shape is silent: libcuda.so.1 resolves out of the\n"
    "    # image CUDA compat layer, so is_available() and even device_count() report\n"
    "    # green while libnvidia-ml.so.1 -- the library NCCL opens at its first\n"
    "    # collective -- is absent (job 37284: all 8 ranks, ncclSystemError, every\n"
    "    # cheaper check green). The probe loads libnvidia-ml.so.1 with ctypes and\n"
    "    # prints torch.cuda.device_count() WITH its denominator -- FS_GPUS_PER_NODE,\n"
    "    # the expected count this backend already knows. A count of 0 is\n"
    "    # UNMEASURED-shaped and refuses; a count != expected refuses; any nonzero rc\n"
    "    # is a refusal. Unmeasured is never PASS.\n"
    "    local -a fs167_probe=(singularity \"${sargs[@]}\" \"$FS_CONTAINER_SQSH\" \\\n"
    "      env ${forward_env[@]+\"${forward_env[@]}\"} PYTHONNOUSERSITE=1 \\\n"
    "      python3 -c '" + PROBE_PY + "' \"$FS_GPUS_PER_NODE\")\n"
    "    if [[ \"$FS_ALLOCATION\" == slurm ]]; then\n"
    "      srun \"${fs167_probe[@]}\" || \\\n"
    "        fs_die 96 \"run_in_container: the fs167 NVML/device-count probe REFUSED (or failed to run) -- libnvidia-ml.so.1 must load in-container AND torch must report exactly FS_GPUS_PER_NODE visible device(s), printed with its denominator; the image CUDA compat layer keeps libcuda.so.1 green at every cheaper level, so this leg is the only measurement that sees the defect (fs167 R6, doctrine 4). Refusing the launch.\"\n"
    "    else\n"
    "      \"${fs167_probe[@]}\" || \\\n"
    "        fs_die 96 \"run_in_container: the fs167 NVML/device-count probe REFUSED (or failed to run) -- libnvidia-ml.so.1 must load in-container AND torch must report exactly FS_GPUS_PER_NODE visible device(s), printed with its denominator; the image CUDA compat layer keeps libcuda.so.1 green at every cheaper level, so this leg is the only measurement that sees the defect (fs167 R6, doctrine 4). Refusing the launch.\"\n"
    "    fi"
)

# Control harness: runs the probe body under sys.executable with ctypes.CDLL and torch
# stubbed BEFORE the probe source is exec'd, so the MUST_FIRE legs fail for the stated
# reason and the MUST_PASS leg passes for the stated reason. mode=block makes
# CDLL("libnvidia-ml.so.1") raise OSError, simulating NVML not bound into the container.
HARNESS_HDR = (
    "import ctypes, sys, types\n"
    "mode, expected, reported = sys.argv[1], sys.argv[2], int(sys.argv[3])\n"
)
HARNESS_BODY = (
    "def _cdll(name, *a, **k):\n"
    "    if mode == \"block\" and name == \"libnvidia-ml.so.1\":\n"
    "        raise OSError(\"simulated fs167 control: NVML not bound into this container\")\n"
    "    return object()\n"
    "ctypes.CDLL = _cdll\n"
    "torch = types.ModuleType(\"torch\")\n"
    "torch.cuda = types.SimpleNamespace(device_count=lambda: reported)\n"
    "sys.modules[\"torch\"] = torch\n"
    "sys.argv = [\"<fs167probe>\", expected]\n"
    "exec(compile(PROBE_SRC, \"<fs167probe>\", \"exec\"), {\"__name__\": \"__main__\"})\n"
)

# Shell-leg harness (C0, C5..C7): the emitted PROBE_BLOCK runs under a real bash with
# stub singularity/srun/python3 binaries on PATH. The singularity stub SKIPS its own
# leading arguments up to and including the image path, then execs the remaining
# `env ... python3 -c ...` command for real, so the probe body genuinely runs; the
# srun stub execs its arguments verbatim. Both touch a sentinel, so a leg that never
# dispatched is distinguishable from one that dispatched and passed. The python3 shim
# stubs ctypes.CDLL and torch exactly like the C1..C4 harness above, driven by the
# FS167_MODE / FS167_REPORTED environment variables so the same harness runs healthy
# and blocked. SHIM_WRAP_PY is the wrapper the python3 shim execs under the real
# interpreter: it installs the stubs, then execs the probe source carried in
# FS167_SRC, forwarding the probe's own argv.
SHIM_WRAP_PY = (
    "import ctypes, os, sys, types\n"
    "mode = os.environ.get(\"FS167_MODE\", \"ok\")\n"
    "reported = int(os.environ.get(\"FS167_REPORTED\", \"8\"))\n"
    "def _cdll(name, *a, **k):\n"
    "    if mode == \"block\" and name == \"libnvidia-ml.so.1\":\n"
    "        raise OSError(\"simulated fs167 control: NVML not bound into this container\")\n"
    "    return object()\n"
    "ctypes.CDLL = _cdll\n"
    "torch = types.ModuleType(\"torch\")\n"
    "torch.cuda = types.SimpleNamespace(device_count=lambda: reported)\n"
    "sys.modules[\"torch\"] = torch\n"
    "sys.argv = [\"<fs167probe>\"] + sys.argv[1:]\n"
    "exec(compile(os.environ[\"FS167_SRC\"], \"<fs167probe>\", \"exec\"), {\"__name__\": \"__main__\"})\n"
)

SINGULARITY_STUB = (
    "#!/usr/bin/env bash\n"
    "touch \"$FS167_SENT_DIR/singularity.ran\"\n"
    "while [[ $# -gt 0 ]]; do\n"
    "  a=\"$1\"; shift\n"
    "  if [[ \"$a\" == \"$FS167_SQSH\" ]]; then break; fi\n"
    "done\n"
    "exec \"$@\"\n"
)

SRUN_STUB = (
    "#!/usr/bin/env bash\n"
    "touch \"$FS167_SENT_DIR/srun.ran\"\n"
    "exec \"$@\"\n"
)

# The emitted leg, wrapped in a function because it uses `local`, with fs_die copied
# in shape from the patched backend: code-aware since fs163 -- a first argument in
# {0,5,95,96} becomes the exit code, anything else defaults to 96. The variables the
# block reads (sargs, FS_CONTAINER_SQSH, forward_env, FS_GPUS_PER_NODE, FS_ALLOCATION)
# are set to harness values from the FS167_* environment.
LEG_SCRIPT = (
    "#!/usr/bin/env bash\n"
    "fs_die() {\n"
    "  local code=96\n"
    "  case \"${1:-}\" in\n"
    "    0|5|95|96) code=\"$1\"; shift ;;\n"
    "  esac\n"
    "  printf 'fs_die[%s]: %s\\n' \"$code\" \"$*\" >&2\n"
    "  exit \"$code\"\n"
    "}\n"
    "run_leg() {\n"
    "  local spwd=\"$PWD\"\n"
    "  local -a sargs=(exec --nv --no-home --pwd \"$spwd\")\n"
    "  local FS_CONTAINER_SQSH=\"$FS167_SQSH\"\n"
    "  local -a forward_env=()\n"
    "  local FS_GPUS_PER_NODE=\"$FS167_GPUS\"\n"
    "  local FS_ALLOCATION=\"$FS167_ALLOC\"\n"
    + PROBE_BLOCK + "\n"
    "}\n"
    "run_leg\n"
)

# C0 provenance: a trivial dispatch through the same stub PATH. It must exit 0 AND
# leave both sentinels, proving bash, the stubs, and the PATH wiring all work before
# any C5..C7 rc is interpreted.
C0_SCRIPT = (
    "#!/usr/bin/env bash\n"
    "srun singularity exec --nv \"$FS167_SQSH\" true\n"
)


def _stderr(msg: str) -> None:
    print(msg, file=sys.stderr)


def _transform(text: str) -> tuple[str, dict[str, int], bool]:
    if MARK in text and A_PRE not in text:
        return text, {"a": 0, "r5": 0}, True
    counts = {"a": text.count(A_PRE), "r5": text.count(R5_ANCHOR)}
    new = text.replace(A_PRE, A_REPL, 1).replace(R5_ANCHOR, R5_ANCHOR + "\n" + PROBE_BLOCK, 1)
    return new, counts, False


def _g7_added_fs_die(block: str) -> list[str]:
    bad: list[str] = []
    for ln in block.splitlines():
        s = ln.strip()
        if s.startswith("#") or "fs_die" not in s:
            continue
        m = re.search(r"\bfs_die\s+(.+)$", s)
        if not m:
            continue
        toks = m.group(1).split()
        if not toks or toks[0] not in CODES:
            bad.append("no explicit contract code: " + s[:110])
            continue
        rest = toks[1:]
        if not rest:
            bad.append("code but no message: " + s[:110])
            continue
        first = rest[0].lstrip("\"'")
        if first in CODES or first.split(":", 1)[0] in CODES:
            bad.append("message begins with a contract token: " + s[:110])
    return bad


def _controls(new: str) -> tuple[int, list[str]]:
    notes: list[str] = []
    ok = 0
    if NV_LINE not in new or PROBE_BLOCK not in new:
        return 0, ["controls not buildable: patched --nv argv line or fs167 probe block absent"]
    del new
    with tempfile.TemporaryDirectory(prefix="fs167-") as td:
        tdp = pathlib.Path(td)
        bindir = tdp / "bin"
        bindir.mkdir()
        wrap = tdp / "fs167_wrap.py"
        wrap.write_text(SHIM_WRAP_PY, "utf-8")
        sqsh = tdp / "fs167_harness.sqsh"
        sqsh.write_text("fs167 harness image placeholder\n", "utf-8")

        def shim(name: str, text: str) -> None:
            p = bindir / name
            p.write_text(text, "utf-8")
            p.chmod(0o755)

        shim("singularity", SINGULARITY_STUB)
        shim("srun", SRUN_STUB)
        shim("python3", (
            "#!/usr/bin/env bash\n"
            "if [[ \"${1:-}\" == -c ]]; then\n"
            "  export FS167_SRC=\"$2\"\n"
            "  shift 2\n"
            "fi\n"
            f"exec \"{sys.executable}\" \"{wrap}\" \"$@\"\n"
        ))
        leg = tdp / "fs167_leg.sh"
        leg.write_text(LEG_SCRIPT, "utf-8")

        def shell_env(tag: str, alloc: str, mode: str, reported: int) -> tuple[dict[str, str], pathlib.Path]:
            sent = tdp / f"sent_{tag}"
            sent.mkdir()
            env = dict(os.environ)
            env["PATH"] = str(bindir) + os.pathsep + env.get("PATH", "")
            env["FS167_SQSH"] = str(sqsh)
            env["FS167_GPUS"] = "8"
            env["FS167_ALLOC"] = alloc
            env["FS167_MODE"] = mode
            env["FS167_REPORTED"] = str(reported)
            env["FS167_SENT_DIR"] = str(sent)
            return env, sent

        # C0 provenance (fs163 shape): before any shell-leg rc is interpreted, prove
        # the harness can run at all -- a trivial dispatch through the same stub PATH.
        c0 = tdp / "fs167_c0.sh"
        c0.write_text(C0_SCRIPT, "utf-8")
        env0, sent0 = shell_env("c0", "slurm", "ok", 8)
        try:
            r0 = subprocess.run(["bash", str(c0)], capture_output=True, text=True, timeout=60, env=env0)
        except (OSError, subprocess.SubprocessError) as exc:
            notes.append(f"C0 provenance: FAIL -- the shell harness could not run ({type(exc).__name__}: {exc}); "
                         "returning 0 controls so a harness failure cannot counterfeit a MUST_FIRE")
            return 0, notes
        c0_srun = (sent0 / "srun.ran").exists()
        c0_sing = (sent0 / "singularity.ran").exists()
        if r0.returncode != 0 or not c0_srun or not c0_sing:
            notes.append(f"C0 provenance: FAIL -- trivial harness rc={r0.returncode} expected=0 "
                         f"srun.sent={c0_srun} singularity.sent={c0_sing} " + r0.stderr.strip()[:140]
                         + "; returning 0 controls so a harness failure cannot counterfeit a MUST_FIRE")
            return 0, notes
        notes.append("C0 provenance: PASS -- trivial harness exited 0 through the stub PATH with both "
                     "sentinels observed, so the C5..C7 shell-leg rc's are interpretable. This REWRITES the "
                     "earlier vacuous note: real bash and real shim binaries ARE now invoked, so the "
                     "provenance precondition is executed, not assumed")

        def run(tag: str, mode: str, expected: int, reported: int) -> subprocess.CompletedProcess[str]:
            hp = pathlib.Path(td) / f"fs167_{tag}.py"
            hp.write_text(HARNESS_HDR + "PROBE_SRC = " + repr(PROBE_PY) + "\n" + HARNESS_BODY, "utf-8")
            return subprocess.run([sys.executable, str(hp), mode, str(expected), str(reported)],
                                  capture_output=True, text=True, timeout=60)

        def expect(name: str, r: subprocess.CompletedProcess[str], rc: int, needle: str) -> bool:
            good = r.returncode == rc and needle in (r.stdout + r.stderr)
            notes.append(f"{name}: rc={r.returncode} expected={rc} " + ("PASS" if good else "FAIL " + (r.stderr.strip()[:140])))
            return good

        if expect("C1 NVML-unloadable MUST_FIRE", run("c1", "block", 8, 8), 1, "libnvidia-ml.so.1"):
            ok += 1
        if expect("C2 healthy MUST_PASS", run("c2", "ok", 8, 8), 0,
                  "fs167: NVML loadable in-container; torch reports 8 of 8 expected visible device(s)"):
            ok += 1
        if expect("C3 zero-count MUST_FIRE", run("c3", "ok", 8, 0), 1, "UNMEASURED-shaped"):
            ok += 1
        if expect("C4 wrong-count MUST_FIRE", run("c4", "ok", 8, 7), 1, "7 of 8 expected"):
            ok += 1

        def run_leg(tag: str, alloc: str, mode: str, reported: int) -> tuple[subprocess.CompletedProcess[str], pathlib.Path]:
            env, sent = shell_env(tag, alloc, mode, reported)
            r = subprocess.run(["bash", str(leg)], capture_output=True, text=True, timeout=60, env=env)
            return r, sent

        r5, sent5 = run_leg("c5", "slurm", "ok", 8)
        s5_srun = (sent5 / "srun.ran").exists()
        g5 = (r5.returncode == 0 and s5_srun
              and "fs167: NVML loadable in-container" in r5.stdout)
        notes.append(f"C5 slurm-arm healthy MUST_PASS: rc={r5.returncode} expected=0 "
                     f"srun.sent={s5_srun} (dispatch proven, not skipped) "
                     + ("PASS" if g5 else "FAIL " + (r5.stderr.strip()[:140])))
        ok += int(g5)

        r6, sent6 = run_leg("c6", "slurm", "block", 8)
        g6 = r6.returncode == 96 and "libnvidia-ml.so.1" in r6.stderr
        notes.append(f"C6 slurm-arm NVML-blocked MUST_FIRE: rc={r6.returncode} expected=96 "
                     "(the fs163 code-aware fs_die contract code must survive the leg, not collapse to 1) "
                     + ("PASS" if g6 else "FAIL " + (r6.stderr.strip()[:140])))
        ok += int(g6)

        r7, sent7 = run_leg("c7", "local", "ok", 8)
        s7_srun = (sent7 / "srun.ran").exists()
        s7_sing = (sent7 / "singularity.ran").exists()
        g7c = r7.returncode == 0 and not s7_srun and s7_sing
        notes.append(f"C7 non-slurm-arm dispatches directly: rc={r7.returncode} expected=0 "
                     f"srun.sent={s7_srun} (must be ABSENT) singularity.sent={s7_sing} (must be PRESENT) "
                     + ("PASS" if g7c else "FAIL " + (r7.stderr.strip()[:140])))
        ok += int(g7c)
    return ok, notes


def main() -> int:
    # build_h100_plane.sh invokes every stage as `python3 <stage>` with NO arguments, so
    # bare invocation must APPLY. Requiring a flag here would have made the stage a no-op
    # inside the build while passing by hand -- the #86 orphan shape, one layer up.
    argv = sys.argv[1:]
    if argv not in ([], ["--apply"], ["--check"]):
        _stderr("usage: patch_nv_runtime.py [--apply|--check]   (no argument == --apply)")
        return 96
    apply = argv != ["--check"]
    if not TARGET.exists():
        _stderr(f"UNMEASURED 95: target missing: {TARGET}")
        return 95
    try:
        text = TARGET.read_text("utf-8")
    except OSError as exc:
        _stderr(f"UNMEASURED 95: target unreadable: {exc}")
        return 95

    new, counts, already = _transform(text)
    if already:
        print("verdict: already applied; byte-idempotent no-op")
        return 0

    gates = 0
    gres: list[tuple[str, bool, str]] = []
    gres.append(("G1", counts["a"] == 1, f"sargs pre-image count={counts['a']} need=1"))
    gres.append(("G2", counts["r5"] == 1, f"R5 torch-probe anchor count={counts['r5']} need=1"))
    gres.append(("G3", new.count(NV_LINE) == 1 and A_PRE not in new,
                 f"post-image --nv argv-line count={new.count(NV_LINE)} need=1 (singularity arm only); "
                 f"un-flagged pre-image absent={A_PRE not in new}"))
    try:
        compile(PROBE_PY, "<fs167probe>", "exec")
        g4, g4msg = True, "embedded probe compiles under <fs167probe>"
    except SyntaxError as exc:
        g4, g4msg = False, f"embedded probe SyntaxError: {exc}"
    gres.append(("G4", g4, g4msg))
    gres.append(("G4q", "'" not in PROBE_PY,
                 "embedded probe carries no single quote (it is embedded in a single-quoted shell string)"))
    with tempfile.NamedTemporaryFile("w", suffix=".sh", delete=False) as fh:
        fh.write(new)
        tmp = fh.name
    try:
        bn = subprocess.run(["bash", "-n", tmp], capture_output=True, text=True)
    finally:
        os.unlink(tmp)
    gres.append(("G5", bn.returncode == 0, "bash -n " + (bn.stderr.strip()[:160] if bn.returncode else "clean")))
    again, _, _ = _transform(new)
    gres.append(("G6", again == new, "byte-idempotence on own output"))
    bad = _g7_added_fs_die(A_REPL + "\n" + PROBE_BLOCK)
    gres.append(("G7", not bad, f"added fs_die explicit-code/contract-token violations={len(bad)} "
                 + ("; ".join(bad[:3]) if bad else "")))

    for name, good, detail in gres:
        print(f"{name}: {'PASS' if good else 'FAIL'}  {detail}")
        gates += int(good)
    cok, cnotes = _controls(new)
    for n in cnotes:
        print("control " + n)

    if gates != len(gres) or cok != N_CONTROLS:
        _stderr(f"\nREFUSE 96: static gates {gates}/{len(gres)}, controls {cok}/{N_CONTROLS}; writing nothing")
        return 96
    if not apply:
        print(f"verdict: READY  {sum(counts.values())} site(s) would be rewritten, {gates}/{len(gres)} static gates, {cok}/{N_CONTROLS} controls")
        return 0
    TARGET.write_text(new, "utf-8")
    print(f"{MARK} {sum(counts.values())} site(s) rewritten, {gates}/{len(gres)} static gates, {cok}/{N_CONTROLS} controls")
    return 0


def _guarded() -> int:
    # This stage exists to stop a silent green from masking a missing capability, so it
    # must not collapse states itself: an unhandled exception is a REFUSE, not a bare rc=1.
    try:
        return main()
    except Exception as exc:  # noqa: BLE001 - deliberate: no traceback may escape as rc=1
        _stderr(f"REFUSE 96: {MARK} stage raised {type(exc).__name__}: {exc}")
        return 96


if __name__ == "__main__":
    raise SystemExit(_guarded())
