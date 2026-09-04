"""patch_collective_probe.py -- build_h100_plane patch stage for defect #129: the collective probe.

MEASUREMENT (all settled, measured on 8x H100 with nemo_25.11.sif; do not re-derive here):
  The backend verifies declared mounts from INSIDE the container (fs117 R4) and torch provenance
  from INSIDE (R5), but nothing verified that a COLLECTIVE completes. On this estate the image's
  bundled HPC-X NCCL net plugin (/opt/hpcx/nccl_rdma_sharp_plugin/lib/libnccl-net.so) is
  auto-loaded by NCCL and SIGSEGVs inside the FIRST all_reduce -- after set_device, cuda alloc
  and init_process_group(nccl) all succeed on every rank; the fault frames come from
  /opt/hpcx/ucx/lib/libucs.so.0. A training job therefore discovers a dead communication plane
  minutes in, after image start, model load and data load, with a stack naming a UCX library
  rather than anything the operator controls. Mounts measured, torch measured, collective
  UNMEASURED -- and unmeasured is never PASS.

THE FIX (three anchored, idempotent edits):
  EDIT 1 (launcher): a CONDITIONAL NCCL_NET_PLUGIN seam driven by FS_NCCL_NET_PLUGIN. The name
    is exported only when the operator speaks (F1: NCCL_NET_PLUGIN=none fixes the crash with
    NVLink/NVLS untouched; F4: an exported-but-EMPTY value means DISABLED, not unset; F5:
    compgen -e at backend :1138 would forward the empty export one layer down). The value is
    validated to `none` or an absolute path to an existing .so, else fail 96 naming the value.
    It is deliberately NOT defaulted -- converting operator silence into an answer is the #126
    defect class (FS_ALLOCATION), and on estates where the plugin works it is a performance
    feature (SHARP offload).
  EDIT 2 (backend): allowlist NCCL_NET_PLUGIN so the conditional value actually crosses into
    the container; probe and payload must agree.
  EDIT 3 (launcher): a fail-closed collective probe. w ranks are os.fork()'d BEFORE any torch
    import (F7: spawn is structurally impossible inside python3 -c), each rank contributes its
    rank id (sum 0..w-1, rank-distinct so silently duplicated ranks on one device -- the #124
    shape -- cannot pass), spread catches a partial reduction, and ANY nonzero rc refuses the
    launch with the measured remedy named. FS_PROBE_CORRUPT=1 poisons rank 0 as the MUST_FIRE
    control (observed got=29.0 vs expected=28.0).

DOES NOT CLAIM:
  - that the HPC-X plugin is broken on any other estate -- the export is conditional;
  - that NCCL_IB_DISABLE=1 helps (F2: it disables a TRANSPORT, not the plugin LOAD);
  - that NCCL_NET=Socket is a fix (F3: it forces socket for INTER-node traffic -- wrong for a
    multi-node framework);
  - anything about training convergence. One all_reduce of a rank-distinct payload is measured;
    nothing else.

GATES (all local, no GPU, no container):
  C1  idempotent: MARK already present in the launcher -> no-op, exit 0.
  C2  all three anchors unique (count == 1 each), reported by name.
  C3  the embedded probe source COMPILES: compile(src, "<probe>", "exec").
  C4  the probe source contains ZERO single-quote characters (it ships inside a single-quoted
      bash assignment; a quote would silently truncate the script on the GPU node).
  C5  bash -n clean on BOTH patched files (temp copies; nothing written yet).
  C6  EXECUTED conditional-export rows in stubbed bash (fail() stubbed), asserting on
      compgen -e -- NOT on [[ -n ]], because the empty-export bug is invisible to -n:
        unset    -> NCCL_NET_PLUGIN absent from compgen -e (count 0)   (the F4/F5 claim)
        none     -> exported, value none
        garbage  -> refused nonzero, message names the value
        /nope/x.so (absent path) -> refused nonzero, message names the path
  C7  EXECUTED probe-invocation rows with run_in_container stubbed; the block under test is
      LIFTED from the patched launcher text via the fs129 marker region, never re-typed:
        stub rc=0 -> exit 0 and the confirmation names the world size
        stub rc=1 -> refused nonzero and the refusal names FS_NCCL_NET_PLUGIN
  C8  INTEGRATION: importlib-load gate_env_drift.py and call check(new_launcher, new_backend,
      quiet=True); must return no violations (the F6/D3 producer claim, executed not argued).
  C9  the string 'NCCL_NET_PLUGIN:-' must not appear in the launcher -- no defaulting
      expansion crept in (same shape as #126's A5 gate).

Refuses to write if any gate is red; a red stage leaves the tree at the last good state.
"""

from __future__ import annotations

import importlib.util
import os
import pathlib
import subprocess
import sys
import tempfile
from fs_estate_pat import estate_partition_literal

# #157: the estate's short name is an INPUT. It appears here because this anchor has to
# reproduce the before-text verbatim to find it, and that text names the estate. A
# declared-empty estate (NONE) yields the estate-free anchor, not a hole in a sentence.
_SHORT = estate_partition_literal()
_ON_SHORT = f" on {_SHORT}" if _SHORT else ""

# Resolved from --root like every other stage in build_h100_plane.sh. An absolute
# build-host path baked into a stage is the #123 defect class one layer up: it makes the
# build plane itself non-relocatable, and this tree is published.
_ROOT = pathlib.Path(
    sys.argv[sys.argv.index("--root") + 1] if "--root" in sys.argv else "."
)
GEN = _ROOT / "h100" / "gen"
LAUNCH = GEN / "launch_fs_h100.fixed.sh"
BACKEND = GEN / "fs_container_backend.bound.sh"
MARK = "fs129:"

GATE_ENV_DRIFT = _ROOT / "gate_env_drift.py"


# ---------------------------------------------------------------------------
# anchors (exact, each expected to occur exactly once)
# ---------------------------------------------------------------------------

ANCHOR_EDIT1 = (
    "if [[ -n \"${FS_NCCL_IB_HCA:-}\" ]]; then\n"
    "  fail 96 \"FS_NCCL_IB_HCA pinning is unmeasured" + _ON_SHORT
    + "; leave unset unless measured and validated\"\n"
    "fi"
)

ANCHOR_EDIT2 = (
    "    NCCL_DEBUG                     # must cross: preserve NCCL diagnostics inside container"
)

ANCHOR_EDIT3 = (
    '[[ "$visible" == "$FS_GPUS_PER_NODE" ]] || fail 96 "visible GPUs in container ($visible) != requested FS_GPUS_PER_NODE ($FS_GPUS_PER_NODE)"'
)


# ---------------------------------------------------------------------------
# the probe source (VERBATIM; measured working; single-quote-free by construction)
# ---------------------------------------------------------------------------

PROBE_SOURCE = r"""import os, sys
w = int(sys.argv[1])
os.environ["MASTER_ADDR"] = "127.0.0.1"
os.environ["MASTER_PORT"] = os.environ.get("FS_PROBE_PORT", "29947")
def child(r, w):
    import torch, torch.distributed as dist
    torch.cuda.set_device(r)
    dist.init_process_group("nccl", rank=r, world_size=w)
    val = float(r) + (1.0 if os.environ.get("FS_PROBE_CORRUPT") == "1" and r == 0 else 0.0)
    t = torch.full((1024, 1024), val, device="cuda:%d" % r, dtype=torch.float32)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    torch.cuda.synchronize()
    rc = 0
    if r == 0:
        got, spread = t.min().item(), (t.max() - t.min()).item()
        exp = float(w * (w - 1) // 2)
        ok = abs(got - exp) < 1e-3 and spread < 1e-3
        print("FS_COLLECTIVE world=%d got=%s expected=%s spread=%s verdict=%s"
              % (w, got, exp, spread, "OK" if ok else "MISMATCH"), flush=True)
        rc = 0 if ok else 1
    dist.barrier()
    dist.destroy_process_group()
    return rc
kids = []
for r in range(1, w):
    pid = os.fork()
    if pid == 0:
        try:
            os._exit(child(r, w))
        except BaseException as e:
            sys.stderr.write("FS_COLLECTIVE child %d FAULT %s: %s\n" % (r, type(e).__name__, e))
            os._exit(1)
    kids.append(pid)
rc = 0
try:
    rc = child(0, w)
except BaseException as e:
    sys.stderr.write("FS_COLLECTIVE rank0 FAULT %s: %s\n" % (type(e).__name__, e))
    rc = 1
for p in kids:
    if os.waitpid(p, 0)[1] != 0:
        rc = 1
sys.exit(rc)"""


# ---------------------------------------------------------------------------
# replacement snippets
# ---------------------------------------------------------------------------

# NOTE on the expansion form: this guard deliberately reads ${FS_NCCL_NET_PLUGIN-} (no colon).
# For an [[ -n ]] test it is identical to :- in every case (unset -> empty; set-empty -> false),
# and it keeps the literal substring NCCL_NET_PLUGIN:- -- which C9 forbids anywhere in the
# launcher -- out even of the guard line, so C9 stays an honest test rather than a carved-out
# exception.
_EDIT1_SNIPPET = r"""

# fs129: conditional NCCL_NET_PLUGIN seam -- the only sanctioned producer of this name.
# Measured on 8x H100 nemo_25.11.sif (F1): NCCL_NET_PLUGIN=none makes the first all_reduce
# correct (world=8 got=28.0 expected=28.0 spread=0.0) with NVLink UNAFFECTED -- under
# NCCL_DEBUG=INFO the 8-rank log still shows "24 coll channels", 17 lines matching NVLS and 24
# matching "via P2P/CUMEM", so NVLS is still selected and P2P/CUMEM is still the intra-node
# path. Those are log-LINE counts with the patterns stated, not per-rank connection totals: an
# earlier revision of this comment claimed "192 P2P/CUMEM", which was 8 x 24 inferred from the
# rank count rather than a count of anything observed, and a reader of a generated file has no
# way to tell a measurement from an arithmetic guess.
# The framework deliberately does NOT default this to none: on an
# estate where the bundled plugin works it is a performance feature (SHARP offload), and
# converting operator silence into a decision is the #126 defect class (defaulting
# FS_ALLOCATION). F4: an exported-but-EMPTY NCCL_NET_PLUGIN means DISABLED, not unset. F5: the
# backend builds its forwarded env from compgen -e (backend :1138), which LISTS
# exported-but-empty names -- an unconditional export would forward NCCL_NET_PLUGIN= into the
# container and re-create F4 one layer down. So this export exists only when
# FS_NCCL_NET_PLUGIN is non-empty, and the value is validated because a mistyped VALUE is
# worse than an unset one: a typo like NCCL_NET_PLUGN=none changes nothing, but a wrong value
# launches wrong.
if [[ -n "${FS_NCCL_NET_PLUGIN-}" ]]; then
  case "$FS_NCCL_NET_PLUGIN" in
    none)
      export NCCL_NET_PLUGIN="none"
      ;;
    /*.so)
      [[ -f "$FS_NCCL_NET_PLUGIN" ]] || fail 96 "fs129: got FS_NCCL_NET_PLUGIN=$FS_NCCL_NET_PLUGIN -- want none or an absolute path to an existing .so"
      export NCCL_NET_PLUGIN="$FS_NCCL_NET_PLUGIN"
      ;;
    *)
      fail 96 "fs129: got FS_NCCL_NET_PLUGIN=$FS_NCCL_NET_PLUGIN -- refused: accept only none or an absolute path to an existing .so"
      ;;
  esac
fi
"""

_EDIT2_SNIPPET = r"""
    NCCL_NET_PLUGIN                # fs129: must cross: selects/disables the container's NCCL net
                                   #   plugin; the collective probe and the payload must agree
"""

_EDIT3_HEAD = r"""

# fs129: the collective gate. Mounts are verified from inside the container (fs117 R4) and torch
# provenance from inside (R5), but until now NOTHING verified that a collective completes -- and
# unmeasured is never PASS. This runs $visible ranks via os.fork BEFORE any torch import, so no
# CUDA context is ever inherited across a fork; torch.multiprocessing.spawn is structurally
# impossible inside python3 -c (F7). Each rank contributes its rank id (sum 0..w-1 = 28 at w=8),
# not 1.0 -- a uniform probe sums to w and passes even with ranks silently duplicated onto one
# device (the #124 shape). spread catches a partial reduction. FS_PROBE_CORRUPT=1 poisons rank 0
# as the MUST_FIRE control (observed got=29.0 vs expected=28.0). Same-interpreter rule from #128:
# the process forking ranks is the same interpreter that imports torch (${FS_PYTHON:-python3}).
fs129_collective_probe='"""

_EDIT3_TAIL = r"""'
run_in_container --slurm-ntasks 1 -- \
  "${FS_PYTHON:-python3}" -c "$fs129_collective_probe" "$visible"
fs129_rc=$?
if [[ "$fs129_rc" -ne 0 ]]; then
  fail 96 "fs129: collective plane UNMEASURED or BROKEN -- launch refused (probed world=$visible). Measured failure on 8x H100 nemo_25.11.sif: SIGSEGV inside the FIRST all_reduce with fault frames in /opt/hpcx/ucx/lib/libucs.so.0 while init_process_group(nccl) SUCCEEDED -- the auto-loaded HPC-X NCCL net plugin crashes. MEASURED remedy: FS_NCCL_NET_PLUGIN=none (correct at world=8, NVLink/NVLS unaffected). NCCL_IB_DISABLE=1 does NOT help: it disables a transport, not the plugin LOAD (F2)."
fi
echo "fs129: collective probe PASS -- all_reduce verified across world=$visible ranks"
"""


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _pass(tag: str, msg: str) -> None:
    print(f"  PASS {tag}  {msg}")


def _fail(tag: str, msg: str) -> None:
    print(f"  FAIL {tag}  {msg}", file=sys.stderr)


def _run_bash(script: str, *, syntax_only: bool = False) -> tuple[int, str, str]:
    """Run a bash script locally; return (rc, stdout, stderr).

    syntax_only adds -n. That distinction is load-bearing and was got wrong once: C5
    called this helper without it, so a gate LABELLED "bash -n clean" actually EXECUTED
    the launcher and reported the launcher's own runtime refusal (rc 96, "fs_container_
    backend.sh not readable") as a syntax error. Both files are in fact syntactically
    clean. C7 genuinely needs execution, so the choice is per-call, not global.
    """
    fd, path = tempfile.mkstemp(suffix=".sh", prefix="fs129_gate_")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(script)
        argv = ["/bin/bash", "-n", path] if syntax_only else ["/bin/bash", path]
        proc = subprocess.run(argv, capture_output=True, text=True, timeout=60)
        return proc.returncode, proc.stdout, proc.stderr
    finally:
        os.unlink(path)


def _lift(text: str, start_needle: str, end_pred) -> str:
    """Lift a shipped region out of patched text: from the line containing start_needle to
    the first subsequent line satisfying end_pred (inclusive). Never re-types the block."""
    lines = text.splitlines()
    start = None
    for i, ln in enumerate(lines):
        if start_needle in ln:
            start = i
            break
    if start is None:
        raise ValueError(f"lift start not found: {start_needle!r}")
    for j in range(start + 1, len(lines)):
        if end_pred(lines[j]):
            return "\n".join(lines[start : j + 1])
    raise ValueError(f"lift end not found after {start_needle!r}")


# ---------------------------------------------------------------------------
# gates
# ---------------------------------------------------------------------------


def _gate_c5(new_launcher: str, new_backend: str) -> bool:
    ok = True
    for name, text in ((LAUNCH.name, new_launcher), (BACKEND.name, new_backend)):
        rc, _out, err = _run_bash(text, syntax_only=True)
        if rc == 0:
            _pass("C5", f"bash -n clean: {name}")
        else:
            _fail("C5", f"bash -n rc={rc} on patched {name}: {err.strip()[:400]}")
            ok = False
    return ok


def _gate_c6(new_launcher: str) -> bool:
    try:
        block = _lift(
            new_launcher,
            "# fs129: conditional NCCL_NET_PLUGIN",
            lambda ln: ln.strip() == "fi",
        )
    except ValueError as exc:
        _fail("C6", f"could not lift conditional-export block from patched launcher: {exc}")
        return False
    harness = (
        "#!/bin/bash\n"
        "set -uo pipefail\n"
        'fail() { echo "fs129-stub fail 96: $*" >&2; exit 96; }\n'
        "__ROW_SETUP__\n"
        + block
        + "\n"
        # `compgen -e` lists exported NAMES, with no `=` and no value. Anchoring on
        # '^NCCL_NET_PLUGIN=' therefore NEVER matches, which made the unset row's
        # expected `compgen_count=0` green for the wrong reason -- a false green -- while
        # the none row went correctly red at count 0. Anchor on the name. The two rows
        # are now each other's control: none->1 proves the counter can fire at all, so
        # unset->0 is a measurement rather than a broken pattern.
        'echo "compgen_count=$(compgen -e | grep -c \'^NCCL_NET_PLUGIN$\' || true)"\n'
        'echo "value=${NCCL_NET_PLUGIN-<unset>}"\n'
    )
    rows = [
        # (row name, setup line, want_rc_zero, stdout needles, stderr needles)
        ("FS_NCCL_NET_PLUGIN unset", "unset FS_NCCL_NET_PLUGIN", True,
         ["compgen_count=0", "value=<unset>"], []),
        ("FS_NCCL_NET_PLUGIN=none", "FS_NCCL_NET_PLUGIN=none", True,
         ["compgen_count=1", "value=none"], []),
        ("FS_NCCL_NET_PLUGIN=garbage", "FS_NCCL_NET_PLUGIN=garbage", False,
         [], ["garbage"]),
        ("FS_NCCL_NET_PLUGIN=/nope/x.so", "FS_NCCL_NET_PLUGIN=/nope/x.so", False,
         [], ["/nope/x.so"]),
    ]
    problems: list[str] = []
    for name, setup, want_zero, out_needles, err_needles in rows:
        rc, out, err = _run_bash(harness.replace("__ROW_SETUP__", setup))
        if want_zero and rc != 0:
            problems.append(f"{name}: expected rc 0, got {rc} (stderr: {err.strip()[:200]})")
        if not want_zero and rc == 0:
            problems.append(f"{name}: expected refusal (nonzero rc), got 0")
        for needle in out_needles:
            if needle not in out:
                problems.append(f"{name}: stdout missing {needle!r} (got: {out.strip()[:200]})")
        for needle in err_needles:
            if needle not in err:
                problems.append(f"{name}: stderr missing {needle!r} (got: {err.strip()[:200]})")
    if problems:
        for p in problems:
            _fail("C6", p)
        return False
    _pass(
        "C6",
        "4 executed rows: unset->absent from compgen -e (count 0, the F4/F5 claim); "
        "none->exported(none); garbage->refused (names value); "
        "/nope/x.so (absent)->refused (names path)",
    )
    return True


def _gate_c7(new_launcher: str) -> bool:
    try:
        block = _lift(
            new_launcher,
            "# fs129: the collective gate",
            lambda ln: ln.startswith("echo ") and "collective probe PASS" in ln,
        )
    except ValueError as exc:
        _fail("C7", f"could not lift probe-invocation block from patched launcher: {exc}")
        return False
    harness = (
        "#!/bin/bash\n"
        "set -uo pipefail\n"
        'fail() { echo "fs129-stub fail 96: $*" >&2; exit 96; }\n'
        "run_in_container() {\n"
        '  if [[ "${FS129_STUB_RC}" == "0" ]]; then\n'
        '    echo "FS_COLLECTIVE world=${visible} got=28.0 expected=28.0 spread=0.0 verdict=OK"\n'
        "    return 0\n"
        "  fi\n"
        '  echo "FS_COLLECTIVE rank0 FAULT SIGSEGV: simulated HPC-X plugin crash in first all_reduce" >&2\n'
        "  return 1\n"
        "}\n"
        "visible=8\n"
        "FS129_STUB_RC=__STUB_RC__\n"
        + block
        + "\n"
    )
    problems: list[str] = []
    rc, out, err = _run_bash(harness.replace("__STUB_RC__", "0"))
    if rc != 0:
        problems.append(f"stub rc=0 row: expected exit 0, got {rc} (stderr: {err.strip()[:200]})")
    if "world=8" not in out:
        problems.append(f"stub rc=0 row: confirmation line missing world size (got: {out.strip()[:200]})")
    rc, out, err = _run_bash(harness.replace("__STUB_RC__", "1"))
    if rc == 0:
        problems.append("stub rc=1 row: expected refusal (nonzero rc), got 0")
    if "FS_NCCL_NET_PLUGIN" not in err:
        problems.append(f"stub rc=1 row: refusal message must name FS_NCCL_NET_PLUGIN (got: {err.strip()[:200]})")
    if problems:
        for p in problems:
            _fail("C7", p)
        return False
    _pass(
        "C7",
        "2 executed rows against the SHIPPED lifted block: stub rc=0 -> exit 0 with "
        "world=8 named; stub rc=1 -> refused nonzero, message names FS_NCCL_NET_PLUGIN",
    )
    return True


def _gate_c8(new_launcher: str, new_backend: str) -> bool:
    try:
        spec = importlib.util.spec_from_file_location("gate_env_drift", str(GATE_ENV_DRIFT))
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot load spec for {GATE_ENV_DRIFT}")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        violations = mod.check(new_launcher, new_backend, quiet=True)
    except Exception as exc:  # noqa: BLE001 -- a failed integration gate is a red gate
        _fail("C8", f"gate_env_drift integration could not run: {type(exc).__name__}: {exc}")
        return False
    if violations:
        _fail("C8", f"gate_env_drift.check returned violations: {violations}")
        return False
    _pass("C8", "gate_env_drift.check(new_launcher, new_backend, quiet=True) -> no violations (F6/D3 producer claim, executed)")
    return True


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> int:
    if not LAUNCH.exists() or not BACKEND.exists():
        _fail("C2", f"expected artifacts missing under {GEN}: "
                    f"{LAUNCH.name} exists={LAUNCH.exists()}, {BACKEND.name} exists={BACKEND.exists()}")
        return 5

    launch_text = LAUNCH.read_text("utf-8")
    backend_text = BACKEND.read_text("utf-8")

    # C1 -- idempotency
    if MARK in launch_text:
        _pass("C1", f"idempotent: {MARK} already present in {LAUNCH.name}; nothing to do")
        return 0
    _pass("C1", f"{MARK} absent from {LAUNCH.name}; applying stage")

    # C2 -- anchor uniqueness
    anchors = [
        ("EDIT1 launcher FS_NCCL_IB_HCA refuse-block", launch_text.count(ANCHOR_EDIT1)),
        ("EDIT2 backend NCCL_DEBUG allowlist line", backend_text.count(ANCHOR_EDIT2)),
        ("EDIT3 launcher visible-GPU check", launch_text.count(ANCHOR_EDIT3)),
    ]
    c2_ok = True
    for name, count in anchors:
        if count == 1:
            _pass("C2", f"anchor unique (count==1): {name}")
        else:
            _fail("C2", f"anchor count == {count} (need exactly 1): {name}")
            c2_ok = False
    if not c2_ok:
        _fail("WRITE", "C2 red; refusing to write; tree left at last good state")
        return 5

    new_launcher = launch_text.replace(ANCHOR_EDIT1, ANCHOR_EDIT1 + _EDIT1_SNIPPET, 1)
    new_launcher = new_launcher.replace(
        ANCHOR_EDIT3,
        ANCHOR_EDIT3 + _EDIT3_HEAD + PROBE_SOURCE + _EDIT3_TAIL,
        1,
    )
    new_backend = backend_text.replace(ANCHOR_EDIT2, ANCHOR_EDIT2 + _EDIT2_SNIPPET, 1)

    green = True

    # C3 -- probe compiles
    try:
        compile(PROBE_SOURCE, "<probe>", "exec")
        _pass("C3", "embedded probe source compiles: compile(src, \"<probe>\", \"exec\")")
    except SyntaxError as exc:
        _fail("C3", f"probe source does not compile: {exc}")
        green = False

    # C4 -- zero single quotes in the probe source
    n_quotes = PROBE_SOURCE.count("'")
    if n_quotes == 0:
        _pass("C4", "probe source contains ZERO single-quote characters (single-quoted-bash safe)")
    else:
        _fail("C4", f"probe source contains {n_quotes} single-quote character(s); would silently truncate the shipped bash string")
        green = False

    # C5 -- bash -n on both patched files
    if not _gate_c5(new_launcher, new_backend):
        green = False

    # C6 -- executed conditional-export rows (compgen -e, not [[ -n ]])
    if not _gate_c6(new_launcher):
        green = False

    # C7 -- executed probe-invocation rows against the shipped lifted block
    if not _gate_c7(new_launcher):
        green = False

    # C8 -- live gate_env_drift integration (D3 producer claim)
    if not _gate_c8(new_launcher, new_backend):
        green = False

    # C9 -- no defaulting expansion crept in (#126 A5 shape)
    if "NCCL_NET_PLUGIN:-" not in new_launcher:
        _pass("C9", "'NCCL_NET_PLUGIN:-' absent from launcher -- no defaulting expansion")
    else:
        _fail("C9", "'NCCL_NET_PLUGIN:-' present in launcher -- a defaulting expansion crept in (#126 A5 shape)")
        green = False

    if not green:
        _fail("WRITE", "one or more gates red; refusing to write; tree left at last good state")
        return 5

    LAUNCH.write_text(new_launcher, "utf-8")
    BACKEND.write_text(new_backend, "utf-8")
    print(f"ALL GATES GREEN -> {LAUNCH.name} {BACKEND.name}")
    print(
        "NOTE: FS_NCCL_NET_PLUGIN is deliberately NOT defaulted -- operator silence stays "
        "silence (#126 precedent); set it only to a measured value (none or an absolute .so)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
