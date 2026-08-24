"""fix32 — the container executor forwarded the host PATH and silently swapped
the interpreter.

The defect (measured on <compute-node>, 2026-08-24, container
``fs-g4e4b-nemo-automodel-26-04_compute``): ``run_in_container()``'s enroot arm
re-supplied every exported host variable via ``--env``, so the host ``PATH``
clobbered the image's ``PATH`` and the container resolved ``python3`` from the
host's anaconda (case 1: ``import torch`` FAILS) instead of ``/opt/venv/bin``
(case 3: torch 2.11.0a0+nv26.02, E4B entry points 2/2). Case 1 vs case 3
differ ONLY in the forwarded ``PATH``.

What this file pins
-------------------
* Task A — the named denylist ``fs_env_forward_denylisted()``: the convicted
  family (``PATH``; conda block; ``LD_LIBRARY_PATH``/``LD_PRELOAD``/
  ``PYTHONHOME``/``PYTHONSTARTUP``/``VIRTUAL_ENV``) never reaches
  ``enroot --env``; the load-bearing keep-set (``PYTHONPATH``,
  ``PYTHONNOUSERSITE``, ``CUDA_VISIBLE_DEVICES``, ``HOME``, ``USER``,
  ``LOGNAME``, ``LANG``, ``LC_CTYPE``, and the minted
  ``NCCL_*``/``GLOO_*``/``SLURM_*``/``FS_*``/``MASTER_*``/``RIC_*``/
  ``ENROOT_*`` families) keeps flowing byte-identically.
* Task B — ``FS_CONTAINER_WRAPPER`` (``fix32-container-tripwire-v1``): wired
  into BOTH executor arms; refuses (rc 95, naming the resolved interpreter AND
  the in-container PATH) when the container resolves its interpreter from
  host-mounted territory; passes with a counted denominator otherwise.
* MUST_FIRE / MUST_PASS (doctrine 3): the drill knob
  ``FS_REARM_HOST_PATH_FORWARD=1`` re-creates case 1 on purpose and the
  tripwire MUST fire; a clean launch MUST pass. Both legs below execute the
  SHIPPED wrapper text — it arrives at the stub container runtime as argv,
  never as a copy in this file (the no-paraphrase rule this estate applies to
  every control).

Fail-before / pass-after (stated per doctrine: a control that only passes
after is not a control):

    test_enroot_keeps_load_bearing_forwarding_and_drops_residue
        PASS before, PASS after — the anti-overreach control.
    test_enroot_denylist_drops_the_host_runtime_family
        FAIL before (PATH + 13 others observed in the --env stream), PASS after.
    test_enroot_invocation_shape_mounts_roots_wrapper
        FAIL before (mount literal /home/<group>/...; TRIPWIRE:absent), PASS after.
    test_tripwire_must_fire_when_host_path_is_rearmed
        FAIL before (rc 0; host python executes; no refusal exists to fire), PASS after.
    test_tripwire_must_pass_on_a_clean_launch
        FAIL before (no tripwire line; EXEC names the host interpreter), PASS after.
    test_slurm_arm_refuses_if_client_env_wins__conditional
        FAIL before (TRIPWIRE:absent; rc 0), PASS after.
    test_slurm_arm_passes_if_image_env_is_restored__conditional
        FAIL before (TRIPWIRE:absent), PASS after.
    test_denylist_predicate_is_named_and_total
        FAIL before (function absent), PASS after.
    test_denylist_comment_cites_the_measurement
        FAIL before (string absent), PASS after.

Harness idiom and its honesty (mirrors launchers/test_launcher_contracts.sh):
stub ``enroot``/``srun`` binaries in a tmp sandbox; the SHIPPED backend is
sourced and run unmodified. The stubs' EMULATED semantics are stated in their
own headers — enroot stub: image presents a baseline PATH, ``--env`` overrides
it (exactly the two measured properties fix32 turns on). srun stub: pyxis's
env merging is UNMEASURED on this estate (doctrine 5), so it runs one of two
*hypotheses* selected by ``FS32_SRUN_IMAGE_ENV``; the two slurm legs are
therefore conditional claims, covering both arms by construction while
abstaining on which hypothesis is true. These greens certify the LOGIC under
stubs, never a real launch: re-run case 1 vs case 3 on <compute-node>.

fix37 additions (hardware rerun of the fix32 legs, fix37_shared.md): the
CLEAN launch refused rc 95 on <compute-node> with the host anaconda PREPENDED to
an intact image PATH — measured cause: the wrapper own ``bash -c`` takes
bash's SSH_SOURCE_BASHRC branch (a Debian/Ubuntu compile option) whenever
SSH_CLIENT or SSH2_CLIENT rides the ``--env`` stream, sourcing the
host-mounted ``~/.bashrc`` conda block BEFORE the wrapper's first command.
The denylist therefore cannot be the load-bearing defence — the name space
is not enumerable by inspection (SSH2_CLIENT was found by reading bash's
source) — so the fix ships three layers: the SSH/session family and the
startup-pointer family (``BASH_ENV|ENV|ZDOTDIR`` — ``--norc`` does NOT
disarm ``BASH_ENV``) convicted in ``fs_env_forward_denylisted`` (the pair
marked PROVEN, the rest marked CATEGORY), ``bash --norc -c`` on the wrapper
invocation of BOTH executor arms, and a positional
``unset SSH_CLIENT SSH2_CLIENT BASH_ENV ENV ZDOTDIR`` at the end of the
shipped wrapper before exec — plus a second drill,
``FS_REARM_SSH_SESSION_FORWARD=1``, whose banner and refusal signature are
deliberately distinct from the PATH drill's.

Fail-before / pass-after for the fix37 legs:

    test_enroot_never_forwards_the_fix37_convicted_family
        FAIL before (all 14 exported SSH/session/startup-pointer names
        observed in the --env stream; FLAG/SCRUB records do not exist),
        PASS after.
    test_ssh_drill_re_arms_both_conjuncts__decision_only
        FAIL before (knob, banner and re-arm logic absent), PASS after.
        DECISIONS only: the sandbox run returns 0 BY CONSTRUCTION — no
        conda-bearing ~/.bashrc exists to source — and asserts no refusal.
    test_wrapper_scrubs_the_session_vector_before_exec__shipped_text
        FAIL before (no unset in the shipped wrapper text), PASS after.
    test_slurm_arm_carries_norc_and_scrub__conditional
        FAIL before (no --norc flag word, no SCRUB marker on the slurm
        arm), PASS after.

Declared strengthenings of pre-existing tests (house rule): PREDICATE_TABLE
gains the 14 fix37 deny verdicts (the test body is untouched), and
test_denylist_comment_cites_the_measurement gains the fix37 needles. The
two stubs learn FLAG:/SCRUB: logging with slot-index normalization chosen
so every pre-fix37 positional assertion reads the SAME numbers on either
argv shape — no existing test expectation was edited.

Hardware-owed legs (the EFFECT, which a sandbox cannot honestly reproduce;
each a denominator of exactly 1 launch on <compute-node>):
    H1  MUST_FIRE — FS_REARM_SSH_SESSION_FORWARD=1 from an ssh session: the
        shipped tripwire MUST refuse rc 95, name the anaconda interpreter,
        and show the conda-prepended-to-intact-image PATH signature; the
        banner reports N of the 2 measured trigger names (SSH2_CLIENT is
        host-dependent). A launch that PROCEEDS is proof the tripwire is
        dead.
    H2  MUST_PASS — the clean LEG1 rerun: rc 0 and the pass line naming
        'outside all 3 host-visible root(s)' — the refusal that caught this
        bug must also be proven liftable.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
BACKEND = REPO_ROOT / "launchers" / "fs_container_backend.sh"
BASH = shutil.which("bash") or "/bin/bash"

TRIPWIRE_MARKER_PRESENT = "TRIPWIRE:present"
WRAPPER_MARKER = "fix32-container-tripwire-v1"

# ---------------------------------------------------------------------------
# Stub programs. Each states its emulated semantics in its own header so a
# green can never be misread as a claim about the real binary.
# ---------------------------------------------------------------------------

STUB_PYTHON = """#!/bin/bash
# FS32TEST stub interpreter: answers only WHO was executed, by absolute path.
echo "EXEC:$0"
"""

STUB_ENROOT = """#!/bin/bash
# FS32TEST stub: enroot. EMULATED SEMANTICS, in full:
#   1. the image supplies a baseline env of exactly ONE var,
#      PATH=$FS32_FAKE_IMAGE/opt/venv/bin:/usr/bin:/bin — the measured image
#      behaviour behind fix32 case 3 (host PATH withheld -> /opt/venv wins);
#   2. each --env K=V replaces/inserts into that env — the measured behaviour
#      behind case 1 (--env PATH=... CLOBBERED the image PATH);
#   3. --mount/--rw/NAME are logged, NOT simulated: no VFS is needed because
#      the sandbox host tree already answers at its real path, and the
#      tripwire receives its host-root list as argv from the shipped code;
#   4. the payload argv is exec'd under the assembled env via env -i — the
#      shipped wrapper text runs verbatim; this stub never re-implements the
#      check it feeds;
#   5. fix37: the wrapper invocation gained a bash OPTION word (--norc)
#      between 'bash' and '-c'. Option words there are logged FLAG:<word>
#      and consume NO PAYLOAD_ARG slot — the slots keep the fix32-era logical
#      shape [0=bash 1=-c, wrapper unlogged, 3=name, 4..=roots/--/payload]
#      on either argv shape, so every pre-fix37 positional assertion in this
#      file reads the SAME numbers. The wrapper text is located DYNAMICALLY
#      as the word after '-c': pre-fix37 it was hardcoded slot 2, which once
#      --norc existed would silently have read '-c' as the wrapper — a
#      control reading its verdict off the wrong word. An argv with no '-c'
#      at all keeps the historic slot-2 rule, so a miswired invocation still
#      logs a comparable picture instead of silence;
#   6. the wrapper slot additionally logs SCRUB:present iff the SHIPPED text
#      contains the fix37 layer-3 scrub ('unset SSH_CLIENT SSH2_CLIENT'),
#      SCRUB:absent otherwise — the decision pin, never the effect.
img=$FS32_FAKE_IMAGE
log=$FS32_STUB_LOG
[ "${1:-}" = "start" ] && shift
declare -a envA=("PATH=$img/opt/venv/bin:/usr/bin:/bin")
while [ $# -gt 0 ]; do
  case "$1" in
    --rw) printf 'OPT:--rw\\n' >>"$log"; shift ;;
    --env)
      k=${2%%=*}
      declare -a keep=()
      for e in "${envA[@]}"; do [ "${e%%=*}" = "$k" ] || keep+=("$e"); done
      envA=("${keep[@]}" "$2")
      printf 'ENV:%s\\n' "$2" >>"$log"
      shift 2 ;;
    --mount) printf 'MOUNT:%s\\n' "$2" >>"$log"; shift 2 ;;
    *) printf 'NAME:%s\\n' "$1" >>"$log"; shift; break ;;
  esac
done
printf 'PAYLOAD_ARGC:%s\\n' "$#" >>"$log"
i=0
widx=1
for a in "$@"; do
  if [ "$a" = "-c" ]; then widx=$i; break; fi
  i=$((i+1))
done
i=0
for a in "$@"; do
  if [ "$i" -gt 0 ] && [ "$i" -lt "$widx" ]; then
    printf 'FLAG:%s\\n' "$a" >>"$log"
  elif [ "$i" -eq $((widx+1)) ]; then
    case "$a" in
      *fix32-container-tripwire-v1*) printf 'TRIPWIRE:present\\n' >>"$log" ;;
      *) printf 'TRIPWIRE:absent\\n' >>"$log" ;;
    esac
    case "$a" in
      *"unset SSH_CLIENT SSH2_CLIENT"*) printf 'SCRUB:present\\n' >>"$log" ;;
      *) printf 'SCRUB:absent\\n' >>"$log" ;;
    esac
  else
    n=$i
    if [ "$i" -ge "$widx" ]; then n=$((i - widx + 1)); fi
    printf 'PAYLOAD_ARG:%s:%s\\n' "$n" "$a" >>"$log"
  fi
  i=$((i+1))
done
exec env -i "${envA[@]}" "$@"
"""

STUB_SRUN = """#!/bin/bash
# FS32TEST stub: srun. Emulated: the option surface run_in_container uses
# (--opt=value forms) and payload-argv logging identical to the enroot stub.
# PYXIS-SEMANTICS ABSTENTION (doctrine 5): whether real pyxis re-imposes the
# image env over the env forwarded by --export=ALL is UNMEASURED on this
# estate. This stub does not guess; the test tells it which hypothesis to run:
#   FS32_SRUN_IMAGE_ENV unset/0 -> client env passes through wholesale
#       (the case-1-shaped hypothesis; the tripwire must REFUSE);
#   FS32_SRUN_IMAGE_ENV=1       -> image baseline PATH re-imposed
#       (the benign hypothesis; the tripwire must PASS).
# Neither leg is evidence about which hypothesis is true of real pyxis.
# fix37 logging, identical to the enroot stub's: bash OPTION words between
# 'bash' and '-c' (today exactly --norc) are logged FLAG:<word> and consume
# no PAYLOAD_ARG slot; the wrapper text is located as the word after '-c'
# (no '-c' at all keeps the historic slot-2 rule) and additionally yields
# SCRUB:present/absent for the fix37 layer-3 scrub marker. Positional slot
# indices are unchanged from fix32, so every pre-fix37 assertion in this
# file reads the same numbers on either arm's argv shape.
log=$FS32_STUB_LOG
while [ $# -gt 0 ]; do
  case "$1" in
    --container-image=*|--container-mounts=*|--container-workdir=*|--ntasks=*|--export=*)
      printf 'OPT:%s\\n' "$1" >>"$log"; shift ;;
    --) shift; break ;;
    *) break ;;
  esac
done
printf 'PAYLOAD_ARGC:%s\\n' "$#" >>"$log"
i=0
widx=1
for a in "$@"; do
  if [ "$a" = "-c" ]; then widx=$i; break; fi
  i=$((i+1))
done
i=0
for a in "$@"; do
  if [ "$i" -gt 0 ] && [ "$i" -lt "$widx" ]; then
    printf 'FLAG:%s\\n' "$a" >>"$log"
  elif [ "$i" -eq $((widx+1)) ]; then
    case "$a" in
      *fix32-container-tripwire-v1*) printf 'TRIPWIRE:present\\n' >>"$log" ;;
      *) printf 'TRIPWIRE:absent\\n' >>"$log" ;;
    esac
    case "$a" in
      *"unset SSH_CLIENT SSH2_CLIENT"*) printf 'SCRUB:present\\n' >>"$log" ;;
      *) printf 'SCRUB:absent\\n' >>"$log" ;;
    esac
  else
    n=$i
    if [ "$i" -ge "$widx" ]; then n=$((i - widx + 1)); fi
    printf 'PAYLOAD_ARG:%s:%s\\n' "$n" "$a" >>"$log"
  fi
  i=$((i+1))
done
if [ "${FS32_SRUN_IMAGE_ENV:-0}" = "1" ]; then
  exec env -i "PATH=$FS32_FAKE_IMAGE/opt/venv/bin:/usr/bin:/bin" "HOME=${HOME:-/tmp}" "$@"
else
  exec "$@"
fi
"""

# ---------------------------------------------------------------------------
# The measured environment, transcribed from the fix32 env dump. Marked KEEP /
# DROP by the decide rule the patch encodes; the values are the measured ones
# re-pointed into the sandbox.
# ---------------------------------------------------------------------------

# Load-bearing forwards — pinning these is the anti-overreach control: they
# must be present in the --env stream on the CURRENT tree AND after the patch.
KEEP_FORWARDED = {
    "PYTHONPATH": "/extras:/repo/src:/repo/3rdparty/Megatron-LM",  # README trap 1: EXTRAS first
    "PYTHONNOUSERSITE": "1",   # s7: ~/.local CPU-only torch shadows the CUDA build without it
    "CUDA_VISIBLE_DEVICES": "0,1,2,3",
    "USER": "testuser",        # enroot's own wrapper dies under set -u without it (measured)
    "LOGNAME": "testuser",
    "LANG": "en_US.UTF-8",
    "LC_CTYPE": "UTF-8",
    # everything fs_backend_init mints or sets must flow through the same loop
    "FS_CONTAINER_SQSH": "/sqsh/nemo-automodel-26-04_compute.sqsh",
    "FS_USE_TORCHRUN": "1",
    "NCCL_SOCKET_IFNAME": "bond0",
    "GLOO_SOCKET_IFNAME": "bond0",
    "NCCL_MNNVL_ENABLE": "0",
    "NCCL_NVLS_ENABLE": "1",
    "SLURM_JOB_ID": "1756100000000",
    "SLURM_JOB_NODELIST": "test-node-a",
    "SLURM_NNODES": "1",
    "SLURM_NTASKS": "4",
    "SLURM_NTASKS_PER_NODE": "4",
    "MASTER_ADDR": "127.0.0.1",
    "MASTER_PORT": "29999",
    "RIC_ACTIVE_CONTAINER": "fs32-test",
}

# The convicted family — measured dangerous or structurally dangerous per the
# fix32 classification. Values chosen to be loud if they leak.
def _denylisted(home: Path) -> dict[str, str]:
    return {
        "LD_LIBRARY_PATH": "/bogus/host-ld",
        "LD_PRELOAD": "/bogus/libhost-injection.so",
        "PYTHONHOME": "/bogus/host-pyhome",
        "PYTHONSTARTUP": "/bogus/host-startup.py",
        "VIRTUAL_ENV": str(home / "host-venv"),
        "CONDA_PREFIX": str(home / "anaconda3"),
        "CONDA_EXE": str(home / "anaconda3/bin/conda"),
        "CONDA_PYTHON_EXE": str(home / "anaconda3/bin/python"),
        "CONDA_DEFAULT_ENV": "base",
        "CONDA_SHLVL": "1",
        "CONDA_PROMPT_MODIFIER": "(base)",
        "_CE_CONDA": "",
        "_CE_M": "",
    }

# Caller-shell residue excluded before fix32 and required unchanged by it.
RESIDUE = ("_", "PWD", "OLDPWD", "SHLVL")


class Scenario:
    def __init__(self, tmp_path: Path, arm: str) -> None:
        assert BACKEND.is_file(), (
            f"cannot read {BACKEND} — an unreadable subject is BLOCK, "
            "not a vacuous green (doctrines 1/4)"
        )
        self.arm = arm
        self.root = tmp_path
        self.bin_dir = tmp_path / "bin"
        self.home = tmp_path / "home"
        self.image = tmp_path / "image"
        self.log = tmp_path / "stub.log"
        for d in (self.bin_dir, self.home / ".local/bin", self.home / "anaconda3/bin",
                  self.image / "opt/venv/bin"):
            d.mkdir(parents=True, exist_ok=True)
        (self.home / "prev").mkdir(parents=True, exist_ok=True)
        self.log.write_text("")
        # Measured host shape: anaconda lives UNDER $HOME, i.e. inside the one
        # tree both executor arms bind-mount — the invariant the tripwire reads.
        for stubbed in (self.home / "anaconda3/bin/python3",
                        self.image / "opt/venv/bin/python3"):
            stubbed.write_text(STUB_PYTHON)
            stubbed.chmod(0o755)
        (self.bin_dir / "enroot").write_text(STUB_ENROOT)
        (self.bin_dir / "enroot").chmod(0o755)
        (self.bin_dir / "srun").write_text(STUB_SRUN)
        (self.bin_dir / "srun").chmod(0o755)
        # The measured host PATH order: ~/.local/bin, then anaconda, then the
        # venv-less system dirs. The stub dir leads so the shipped code finds
        # `enroot`/`srun`; it contains no python3, so resolution scans past it.
        self.host_path = (
            f"{self.bin_dir}:{self.home}/.local/bin:{self.home}/anaconda3/bin:/usr/bin:/bin"
        )

    def env(self, drill: bool, srun_image_env: bool,
            ssh_drill: bool = False,
            extra_env: dict[str, str] | None = None) -> dict[str, str]:
        env: dict[str, str] = {
            "PATH": self.host_path,
            "HOME": str(self.home),
            "SHELL": "/bin/zsh",
            "FS_BACKEND": self.arm,
            "ENROOT_DATA_PATH": str(self.home / ".enroot"),
            "_": "/usr/bin/env",
            "PWD": str(self.home),
            "OLDPWD": str(self.home / "prev"),
            "SHLVL": "2",
            "FS32_STUB_LOG": str(self.log),
            "FS32_FAKE_IMAGE": str(self.image),
        }
        env.update(KEEP_FORWARDED)
        env.update(_denylisted(self.home))
        if drill:
            env["FS_REARM_HOST_PATH_FORWARD"] = "1"
        if ssh_drill:
            env["FS_REARM_SSH_SESSION_FORWARD"] = "1"
        if srun_image_env:
            env["FS32_SRUN_IMAGE_ENV"] = "1"
        if extra_env:
            env.update(extra_env)
        return env

    def run(self, *, drill: bool = False, srun_image_env: bool = False,
            ssh_drill: bool = False, extra_env: dict[str, str] | None = None,
            payload: tuple[str, ...] = ("true",)) -> subprocess.CompletedProcess[str]:
        # Drives the SHIPPED backend: source it, mint only what run_in_container
        # reads without a full fs_backend_init/runtime_setup cycle (ENROOT_NAME;
        # FS_BACKEND arrives via env), then call it exactly as the launchers do.
        script = (
            "set -uo pipefail\n"
            'source "$1"\n'
            "shift\n"
            'export ENROOT_NAME=fs32-test\n'
            '"$@"\n'
        )
        return subprocess.run(
            [BASH, "-s", "--", str(BACKEND), "run_in_container", "--", *payload],
            # `bash -s` reads the program from stdin; without input= it reads an
            # EMPTY stdin, runs nothing, and exits 0 with no output -- every
            # assertion below would then be reading silence. A driver that
            # executes 0 statements cannot certify anything (doctrine 1), so the
            # script must actually be delivered.
            input=script,
            env=self.env(drill=drill, srun_image_env=srun_image_env,
                         ssh_drill=ssh_drill, extra_env=extra_env),
            text=True,
            capture_output=True,
            timeout=120,
            check=False,
        )

    def records(self, prefix: str) -> list[str]:
        return [line[len(prefix) + 1:] for line in self.log.read_text().splitlines()
                if line.startswith(prefix + ":")]

    def env_stream(self) -> dict[str, str]:
        """name -> value of every --env the shipped loop handed the container."""
        out: dict[str, str] = {}
        for line in self.records("ENV"):
            k, _, v = line.partition("=")
            out[k] = v
        return out

    def payload_args(self) -> dict[int, str]:
        out: dict[int, str] = {}
        for line in self.records("PAYLOAD_ARG"):
            i, _, v = line.partition(":")
            out[int(i)] = v
        return out


# ---------------------------------------------------------------------------
# Task A legs — the --env stream itself (measured at the stub boundary).
# ---------------------------------------------------------------------------

def test_enroot_keeps_load_bearing_forwarding_and_drops_residue(tmp_path: Path) -> None:
    """ANTI-OVERREACH CONTROL: PASSES on the current tree and PASSES after the
    patch. A control that only passes after is not a control (brief, doctrine
    3): this leg is what convicts an over-broad denylist — drop PYTHONPATH or
    USER and this goes red on a 'fixed' tree."""
    sc = Scenario(tmp_path, "enroot")
    proc = sc.run(payload=("true",))
    assert proc.returncode == 0, proc.stderr
    env = sc.env_stream()
    missing = {k for k, v in KEEP_FORWARDED.items() if env.get(k) != v}
    assert not missing, (
        f"load-bearing variables dropped or mangled by the forwarder: {sorted(missing)}; "
        "a denylist that eats the keep-set is an allowlist wearing a denylist's name"
    )
    assert sc.records("NAME") == ["fs32-test"]
    leaked = [k for k in RESIDUE if k in env]
    assert not leaked, f"caller-shell residue leaked into the container: {leaked}"


def test_enroot_denylist_drops_the_host_runtime_family(tmp_path: Path) -> None:
    """FAILS on the current tree: PATH, the 8-name conda block, and the
    LD_*/PYTHON* injection vectors are ALL forwarded by the old loop — this
    test enumerates the offenders in its red message. PASSES after the patch."""
    sc = Scenario(tmp_path, "enroot")
    proc = sc.run(payload=("true",))
    env = sc.env_stream()
    # Control on this test's own denominator. `convicted` is an INTERSECTION
    # with the recorded stream, so a launch that died before recording any
    # --env would give env == {} and convict nothing -- the assertion below
    # would pass because nothing was measured. That is the all([]) shape this
    # repo exists to hunt, sitting inside one of its own tests. Both legs
    # (the launch succeeded, and it actually forwarded something) must hold
    # before the absence of convicted names is allowed to mean anything.
    assert proc.returncode == 0, f"the launch itself failed: {proc.stderr!r}"
    assert env, "no --env records at all: the census below would be vacuous"
    convicted = sorted({"PATH", *_denylisted(sc.home)} & set(env))
    assert not convicted, (
        f"convicted host variables forwarded into the container (the fix32 defect "
        f"class): {convicted}; PATH in particular resolves the container's python3 "
        "to host anaconda — measured case 1, <compute-node> 2026-08-24"
    )


def test_enroot_invocation_shape_mounts_roots_wrapper(tmp_path: Path) -> None:
    """FAILS before (mounts name the freestanding literal; no wrapper in argv;
    TRIPWIRE:absent). PASSES after: mounts derive from $HOME and the tripwire
    receives its roots from exactly those mounts — never a side-maintained
    copy that could drift."""
    sc = Scenario(tmp_path, "enroot")
    proc = sc.run(payload=("python3", "-c", "pass"))
    assert proc.returncode == 0, f"the launch itself failed: {proc.stderr!r}"
    assert "--rw" in sc.records("OPT")
    assert sc.records("MOUNT") == [f"{sc.home}:{sc.home}", "/dev:/dev", "/sys:/sys"]
    assert TRIPWIRE_MARKER_PRESENT in sc.log.read_text(), (
        "the enroot arm does not invoke the fix32 interpreter tripwire"
    )
    args = sc.payload_args()
    # bash -c <wrapper> fs-container-tripwire <roots...> -- <payload...>
    assert args.get(0) == "bash" and args.get(1) == "-c"
    assert args.get(3) == "fs-container-tripwire"
    assert args.get(4) == str(sc.home)
    assert args.get(5) == "/dev"
    assert args.get(6) == "/sys"
    assert args.get(7) == "--"
    assert args.get(8) == "python3"


# ---------------------------------------------------------------------------
# Task B legs — MUST_FIRE / MUST_PASS, executing the SHIPPED wrapper text
# end to end through the stub container runtime.
# ---------------------------------------------------------------------------

def test_tripwire_must_fire_when_host_path_is_rearmed(tmp_path: Path) -> None:
    """MUST_FIRE (doctrine 3). The drill knob FS_REARM_HOST_PATH_FORWARD=1
    re-creates the measured case 1 verbatim: PATH is forwarded, the stub
    container resolves python3 from anaconda UNDER the mounted host home, and
    the tripwire MUST refuse — naming the resolved interpreter and the actual
    PATH so the operator diagnoses without re-running a probe. FAILS on the
    current tree: nothing refuses, the host python EXECutes, rc 0."""
    sc = Scenario(tmp_path, "enroot")
    proc = sc.run(drill=True, payload=("python3", "-c", "pass"))
    assert TRIPWIRE_MARKER_PRESENT in sc.log.read_text()
    # Right-reason guard (the harness's isolation-self-check discipline): the
    # DRILL line proves PATH went in on purpose, so the refusal is the
    # tripwire's verdict on THAT condition — not some unrelated failure.
    assert "DRILL: FS_REARM_HOST_PATH_FORWARD=1" in proc.stderr
    assert sc.env_stream().get("PATH") == sc.host_path
    assert proc.returncode == 95, (
        f"expected the tripwire's own refusal code 95, got {proc.returncode}: "
        f"stdout={proc.stdout!r} stderr={proc.stderr!r}"
    )
    assert "FATAL: fix32 tripwire" in proc.stderr
    assert f"resolved python3 : {sc.home}/anaconda3/bin/python3" in proc.stderr
    assert f"host mount root  : {sc.home}" in proc.stderr
    assert f"in-container PATH: {sc.host_path}" in proc.stderr
    assert "EXEC:" not in proc.stdout, "the payload ran under a convicted interpreter"


def test_tripwire_must_pass_on_a_clean_launch(tmp_path: Path) -> None:
    """MUST_PASS (doctrine 3). No drill: PATH never enters the container, the
    image interpreter resolves, the tripwire prints its counted denominator (3
    roots on the enroot arm) and the payload runs under the CONTAINER python.
    FAILS on the current tree: no tripwire line exists, and the EXEC line
    names the host anaconda python — the fix32 bug, end to end."""
    sc = Scenario(tmp_path, "enroot")
    proc = sc.run(payload=("python3", "-c", "pass"))
    assert proc.returncode == 0, f"clean launch refused: {proc.stderr!r}"
    expected_tripwire = (
        f"fix32 tripwire: python3 resolves to {sc.image}/opt/venv/bin/python3 — "
        f"outside all 3 host-visible root(s) [{sc.home} /dev /sys]; executing payload."
    )
    assert expected_tripwire in proc.stdout, (
        f"tripwire pass-line with its denominator missing: {proc.stdout!r}"
    )
    assert f"EXEC:{sc.image}/opt/venv/bin/python3" in proc.stdout, (
        "the payload did not run under the container interpreter"
    )
    assert "FATAL" not in proc.stderr


# ---------------------------------------------------------------------------
# Task B legs — the slurm/pyxis arm. CONDITIONAL BY CONSTRUCTION (doctrine 5,
# both directions): the interpreter-swap failure mode is measured on the
# enroot arm ONLY. These two legs prove the SAME tripwire is wired into the
# slurm arm and is decision-relevant under EITHER hypothesis about how pyxis
# merges --export=ALL with the image env. Neither leg claims the slurm arm was
# broken; neither claims it is now fixed. That measurement is still owed.
# ---------------------------------------------------------------------------

def test_slurm_arm_refuses_if_client_env_wins__conditional(tmp_path: Path) -> None:
    """FAILS before (TRIPWIRE:absent; rc 0). After: under the hypothesis that
    --export=ALL's client env reaches the container wholesale, the tripwire
    must refuse exactly as on the enroot arm. Denominator: wiring + one
    conditional behaviour; pyxis's true merge semantics UNMEASURED."""
    sc = Scenario(tmp_path, "slurm")
    proc = sc.run(srun_image_env=False, payload=("python3", "-c", "pass"))
    assert TRIPWIRE_MARKER_PRESENT in sc.log.read_text(), (
        "the slurm arm does not invoke the fix32 interpreter tripwire"
    )
    assert "--export=ALL" in sc.records("OPT")
    args = sc.payload_args()
    assert args.get(0) == "bash" and args.get(1) == "-c"
    assert args.get(3) == "fs-container-tripwire"
    assert args.get(4) == str(sc.home)   # slurm arm declares exactly one host root
    assert args.get(5) == "--"
    assert proc.returncode == 95
    assert f"resolved python3 : {sc.home}/anaconda3/bin/python3" in proc.stderr
    assert "EXEC:" not in proc.stdout


def test_slurm_arm_passes_if_image_env_is_restored__conditional(tmp_path: Path) -> None:
    """FAILS before (TRIPWIRE:absent). After: under the hypothesis that pyxis
    re-imposes the image env, the tripwire must PASS — a guard that fires on
    both hypotheses would be a hair-trigger, not a detector."""
    sc = Scenario(tmp_path, "slurm")
    proc = sc.run(srun_image_env=True, payload=("python3", "-c", "pass"))
    assert TRIPWIRE_MARKER_PRESENT in sc.log.read_text()
    args = sc.payload_args()
    assert args.get(3) == "fs-container-tripwire"
    assert args.get(4) == str(sc.home)
    assert args.get(5) == "--"
    assert proc.returncode == 0, f"conditional pass-leg refused: {proc.stderr!r}"
    assert f"EXEC:{sc.image}/opt/venv/bin/python3" in proc.stdout


# ---------------------------------------------------------------------------
# Task A leg — the denylist is a NAMED, driveable rule, exercised as shipped.
# ---------------------------------------------------------------------------

PREDICATE_TABLE = {
    "PATH": "deny",
    "LD_LIBRARY_PATH": "deny",
    "LD_PRELOAD": "deny",
    "PYTHONHOME": "deny",
    "PYTHONSTARTUP": "deny",
    "VIRTUAL_ENV": "deny",
    "CONDA_PREFIX": "deny",
    "CONDA_EXE": "deny",
    "CONDA_PYTHON_EXE": "deny",
    "CONDA_DEFAULT_ENV": "deny",
    "CONDA_SHLVL": "deny",
    "CONDA_PROMPT_MODIFIER": "deny",
    "_CE_CONDA": "deny",
    "_CE_M": "deny",
    "_": "deny",
    "PWD": "deny",
    "OLDPWD": "deny",
    "SHLVL": "deny",
    # fix37 verdicts (declared strengthening; the test body is untouched):
    # the two PROVEN sshd-rc triggers, the CATEGORY host-session descriptors,
    # and the CATEGORY startup-file pointers (--norc does not disarm
    # BASH_ENV; see the FS_CONTAINER_WRAPPER scrub). Verdicts encode the
    # classification; provenance markers live in the backend comments.
    "SSH_CLIENT": "deny",
    "SSH2_CLIENT": "deny",
    "SSH_CONNECTION": "deny",
    "SSH_TTY": "deny",
    "SSH_AUTH_SOCK": "deny",
    "SSH_ASKPASS": "deny",
    "DBUS_SESSION_BUS_ADDRESS": "deny",
    "XDG_RUNTIME_DIR": "deny",
    "XDG_SESSION_ID": "deny",
    "XDG_SESSION_CLASS": "deny",
    "XDG_SESSION_TYPE": "deny",
    "BASH_ENV": "deny",
    "ENV": "deny",
    "ZDOTDIR": "deny",
    "PYTHONPATH": "forward",
    "PYTHONNOUSERSITE": "forward",
    "CUDA_VISIBLE_DEVICES": "forward",
    "HOME": "forward",
    "USER": "forward",
    "LOGNAME": "forward",
    "LANG": "forward",
    "LC_CTYPE": "forward",
    "NCCL_SOCKET_IFNAME": "forward",
    "NCCL_SOME_FUTURE_PIN": "forward",   # denylist, not allowlist: the unknown
    "FS_SOME_FUTURE_KNOB": "forward",    # minted families must default to flow
    "SLURM_JOB_ID": "forward",
    "MASTER_ADDR": "forward",
    "RIC_ACTIVE_CONTAINER": "forward",
    "ENROOT_DATA_PATH": "forward",
}


def test_denylist_predicate_is_named_and_total() -> None:
    """FAILS before: fs_env_forward_denylisted does not exist, so the loop
    below records 'forward' for every name (command-not-found lands on the
    else arm — visible in stderr). PASSES after. The 'forward' rows for
    invented NCCL_*/FS_* names pin the no-allowlist property of Task A.3."""
    script = (
        'source "$1"\n'
        "shift\n"
        'for v in "$@"; do\n'
        # Emitted NAME-first: the parser below builds {name: verdict} to compare
        # against PREDICATE_TABLE. Verdict-first would key every line on the
        # literal "deny"/"forward", making all 47 names read as "<no verdict>"
        # -- a total mismatch that looks like a dead predicate rather than a
        # mis-parse. Names contain no ':', so split(":", 1) is exact.
        '  if fs_env_forward_denylisted "$v"; then echo "$v:deny"; else echo "$v:forward"; fi\n'
        "done\n"
    )
    proc = subprocess.run(
        [BASH, "-s", "--", str(BACKEND), *PREDICATE_TABLE],
        input=script,  # see the driver above: `bash -s` without input= runs nothing
        text=True, capture_output=True, timeout=60, check=False,
    )
    verdicts = dict(line.split(":", 1) for line in proc.stdout.splitlines() if ":" in line)
    mismatches = {
        name: (verdicts.get(name, "<no verdict>"), want)
        for name, want in PREDICATE_TABLE.items()
        if verdicts.get(name) != want
    }
    assert not mismatches, (
        f"denylist verdicts disagree with the fix32 classification "
        f"(name: (got, want)): {mismatches}; stderr={proc.stderr!r}"
    )


def test_denylist_comment_cites_the_measurement() -> None:
    """Documentation pin (FAILS before the patch, PASSES after): Task A.1
    requires the denylist to cite the case-1-vs-case-3 measurement so the next
    reader cannot mistake the exclusion list for taste."""
    text = BACKEND.read_text()
    for needle in ("case 1 vs case 3", "fs_env_forward_denylisted()",
                   "FS_REARM_HOST_PATH_FORWARD",
                   # fix37 strengthening (declared): the classification must now
                   # also cite the SSH measurement, name the mechanism, pin the
                   # honest CATEGORY marker, and name the second drill.
                   "SSH_CLIENT", "SSH2_CLIENT", "SSH_SOURCE_BASHRC",
                   "CATEGORY", "FS_REARM_SSH_SESSION_FORWARD"):
        assert needle in text, f"missing load-bearing citation/name: {needle!r}"


# ---------------------------------------------------------------------------
# fix37 legs — the SSH_SOURCE_BASHRC mechanism. SANDBOX HONESTY (the fix37
# brief, doctrine 5): the sandbox has no conda-bearing host ~/.bashrc, no
# <compute-node>, and no guarantee of a Debian-patched bash, so these tests pin
# the DECISIONS — is the variable forwarded? is the wrapper invoked with
# --norc? does the drill re-forward the vector and drop --norc? is the
# layer-3 scrub present and ordered before exec? — and never the EFFECT
# (which interpreter a genuinely sourced conda block would produce). The
# effect legs are owed to hardware and are named, with their denominators,
# in the module docstring as H1/H2.
# ---------------------------------------------------------------------------

# The 14 fix37 convictions, exported with loud values: 2 measured sshd-rc
# triggers, 9 CATEGORY host-session descriptors, 3 startup-file pointers.
def _fix37_convicted(home: Path) -> dict[str, str]:
    return {
        "SSH_CLIENT": "10.62.207.17 46126 22",   # the measured host value (fix37_shared.md)
        "SSH2_CLIENT": "10.62.207.17 46126 22",  # measured DIRTY though unset on this host
        "SSH_CONNECTION": "10.62.207.17 46126 10.62.0.4 22",
        "SSH_TTY": "/dev/pts/7",
        "SSH_AUTH_SOCK": "/tmp/ssh-loud/agent.4242",
        "SSH_ASKPASS": "/usr/bin/ssh-askpass",
        "DBUS_SESSION_BUS_ADDRESS": "unix:path=/run/user/4242/bus",
        "XDG_RUNTIME_DIR": "/run/user/4242",
        "XDG_SESSION_ID": "4242",
        "XDG_SESSION_CLASS": "user",
        "XDG_SESSION_TYPE": "tty",
        "BASH_ENV": str(home / "bashrc-loud"),   # --norc does NOT disarm this one
        "ENV": str(home / "sh-env-loud"),
        "ZDOTDIR": str(home / "zdot-loud"),
    }


def test_enroot_never_forwards_the_fix37_convicted_family(tmp_path: Path) -> None:
    """FAILS on the current tree: the old loop forwards every one of the 14
    exported names (they appear verbatim in the --env stream), and the
    invocation carries neither the --norc flag word nor the layer-3 scrub.
    PASSES after the patch. Denominators asserted: 14 convicted names absent;
    1 clean launch rc 0; FLAG exactly ['--norc']; 1 TRIPWIRE + 1 SCRUB marker;
    1 payload execution under the image interpreter."""
    sc = Scenario(tmp_path, "enroot")
    proc = sc.run(extra_env=_fix37_convicted(sc.home), payload=("python3", "-c", "pass"))
    env = sc.env_stream()
    leaked = sorted(set(_fix37_convicted(sc.home)) & set(env))
    assert not leaked, (
        f"fix37-convicted variables forwarded into the container: {leaked}; "
        "SSH_CLIENT is the measured fix37 leak — bash's SSH_SOURCE_BASHRC branch "
        "over the host-mounted ~/.bashrc (<compute-node>, fix37_shared.md)"
    )
    assert proc.returncode == 0, f"clean launch refused: {proc.stderr!r}"
    assert sc.records("FLAG") == ["--norc"], (
        f"wrapper invocation carries {sc.records('FLAG')!r}, want exactly ['--norc'] — "
        "without it the wrapper's OWN bash takes the sshd-rc branch whenever any "
        "session variable slips the denylist"
    )
    log = sc.log.read_text()
    assert TRIPWIRE_MARKER_PRESENT in log and "SCRUB:present" in log
    assert f"EXEC:{sc.image}/opt/venv/bin/python3" in proc.stdout, (
        "the payload did not run under the container interpreter"
    )


def test_ssh_drill_re_arms_both_conjuncts__decision_only(tmp_path: Path) -> None:
    """DECISION leg of the fix37 MUST_FIRE drill (doctrine 3). FAILS before:
    the knob, the banner and the re-arm logic do not exist. PASSES after.
    Asserts the drill re-arms BOTH conjuncts the measured branch needs — the
    trigger names ride the forward stream again AND the invocation drops
    --norc (no_rc re-armed) — with a distinct banner carrying its
    denominator, and that both banner directions stay distinguishable from
    the PATH drill. The sandbox launch returns 0 BY CONSTRUCTION (no
    conda-bearing ~/.bashrc exists to source on any platform this file runs
    on) and asserts no refusal: the EFFECT leg — the tripwire refusing rc 95
    over a genuinely contaminated resolution — is owed to the <compute-node> rerun
    (module docstring, H1), exactly the part a sandbox cannot honestly
    reproduce."""
    triggers = {"SSH_CLIENT": "10.62.207.17 46126 22",
                "SSH2_CLIENT": "10.62.207.17 46126 22"}
    sc = Scenario(tmp_path / "ssh", "enroot")
    proc = sc.run(ssh_drill=True, extra_env=triggers, payload=("true",))
    assert "DRILL: FS_REARM_SSH_SESSION_FORWARD=1 — re-arming" in proc.stderr, (
        f"the SSH drill did not announce itself: stderr={proc.stderr!r}"
    )
    # Denominator: both measured trigger names were exported, so the banner
    # must SAY 2 of 2 — a drill that re-forwarded zero names may never read
    # as the full rehearsal (SSH2_CLIENT was measured DIRTY while unset).
    assert "2 of the 2 measured trigger names" in proc.stderr
    stream = sc.env_stream()
    assert stream.get("SSH_CLIENT") == triggers["SSH_CLIENT"]
    assert stream.get("SSH2_CLIENT") == triggers["SSH2_CLIENT"]
    assert "--norc" not in sc.records("FLAG"), (
        "the drill must re-arm the vulnerable invocation: --norc is still on the argv"
    )
    log = sc.log.read_text()
    assert TRIPWIRE_MARKER_PRESENT in log and "SCRUB:present" in log
    # Banner distinctness, direction A: the SSH drill never ACTIVATES the PATH
    # drill. It does name that knob once, deliberately — the banner tells an
    # operator reading cold logs which of the two leaks this refusal is about,
    # and that cross-reference is the whole point of tell (b). So the
    # discriminator is the ACTIVATION form, symmetric with direction B's
    # positive leg below; asserting the bare knob name is absent would forbid
    # the very sentence the banner exists to print (fix37 shipped both and they
    # contradicted: 1 red of 880).
    assert "DRILL: FS_REARM_HOST_PATH_FORWARD=1" not in proc.stderr
    # Direction B: the PATH drill never prints the SSH knob — two drills that
    # print the same line are one drill (the fix37 control problem).
    scb = Scenario(tmp_path / "path", "enroot")
    procb = scb.run(drill=True, payload=("true",))
    assert "DRILL: FS_REARM_HOST_PATH_FORWARD=1" in procb.stderr
    assert "FS_REARM_SSH_SESSION_FORWARD" not in procb.stderr


def test_wrapper_scrubs_the_session_vector_before_exec__shipped_text() -> None:
    """Positional pin on the SHIPPED wrapper text. FAILS before (no scrub
    exists), PASSES after. Three conjuncts over the 1 shipped string: the
    layer-3 unset targets the measured sshd-rc trigger pair; the
    startup-pointer family rides along (--norc does not disarm BASH_ENV);
    and the scrub sits after the checks and before the payload exec — the
    last environment surgery before handoff, so no descendant shell at any
    depth can take the branch."""
    text = BACKEND.read_text()
    wrapper = text.split("FS_CONTAINER_WRAPPER='", 1)[1].split("\n'", 1)[0]
    assert "unset SSH_CLIENT SSH2_CLIENT" in wrapper, (
        "the fix37 layer-3 scrub is absent from the shipped wrapper text"
    )
    assert "BASH_ENV" in wrapper, (
        "the startup-pointer family is missing from the scrub; "
        "--norc does not disarm BASH_ENV, so the scrub is its only positional defence"
    )
    assert wrapper.index("unset SSH_CLIENT SSH2_CLIENT") < wrapper.rindex('exec "$@"'), (
        "the scrub must precede the payload exec — after it, the scrub protects nothing"
    )


def test_slurm_arm_carries_norc_and_scrub__conditional(tmp_path: Path) -> None:
    """The slurm arm forwards --export=ALL BY DESIGN: SSH_CLIENT WILL reach
    the container there whenever the operator is on ssh, and the denylist
    (layer 1) never runs on that arm. Layers 2 and 3 are therefore the whole
    defence there; this pins their wiring. CONDITIONAL (doctrine 5, as with
    the fix32 slurm legs): run under the benign pyxis hypothesis
    (FS32_SRUN_IMAGE_ENV=1); whether real pyxis re-imposes the image env
    remains UNMEASURED and this test claims nothing about it. FAILS before
    (no --norc flag word, no SCRUB marker), PASSES after."""
    sc = Scenario(tmp_path, "slurm")
    proc = sc.run(srun_image_env=True,
                  extra_env={"SSH_CLIENT": "10.62.207.17 46126 22"},
                  payload=("python3", "-c", "pass"))
    assert proc.returncode == 0, f"conditional pass-leg refused: {proc.stderr!r}"
    assert sc.records("FLAG") == ["--norc"], (
        f"slurm-arm wrapper invocation carries {sc.records('FLAG')!r}; --export=ALL "
        "forwards SSH_CLIENT by design, so --norc is the only barrier in front of "
        "the tripwire's own shell on this arm"
    )
    log = sc.log.read_text()
    assert TRIPWIRE_MARKER_PRESENT in log and "SCRUB:present" in log
    assert f"EXEC:{sc.image}/opt/venv/bin/python3" in proc.stdout
