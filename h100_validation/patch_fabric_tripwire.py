#!/usr/bin/env python3
"""#163: make the pre-submit fabric tripwire a declared knob and give fs_die a legal exit contract.

WHAT / WHY. Slurm job 37280 (8xH100, one node, partition declared via FS_PARTITION) resolved
its plane and then died in 8 seconds: name resolution failed for the configured s9 endpoint,
the /dev/tcp open of that endpoint raised "Invalid argument", the standing-rule tripwire
refused the Slurm launch, and the process exited 1. The plane publishes 0=PASS, 5=RED,
95=UNMEASURED, 96=REFUSE; 1 is not in the contract, so a deliberate guard and an
interpreter fault were indistinguishable.

This stage changes two things in fs_container_backend.bound.sh. The s9 tripwire stops
naming one estate's endpoint and becomes FS_FABRIC_TRIPWIRE: required, no default, the
sentinel none is a logged declaration rather than a silent omission, and any other value is
validated as host:port before it is allowed anywhere near bash -c. A dead endpoint still
fails CLOSED under the same 15s bound; it never hangs a job that could have been good.
fs_die() gains an optional leading exit code drawn only from {0,5,95,96} and defaults to
96, because every existing call site is a guard declining to proceed.

This stage refuses to write when a pre-image is absent or multiplied, when residue of the
retired endpoint survives, when any fs_die message begins with a contract token that the new
numeric-prefix form would swallow, when bash -n rejects the result, or when any executable
control is not observed. It measures 95 only when the target cannot be read at all.
"""

from __future__ import annotations

import os
import pathlib
import re
import socket
import shutil
import subprocess
import sys
import tempfile

TARGET = pathlib.Path(__file__).resolve().parent / "h100" / "gen" / "fs_container_backend.bound.sh"
MARK = "fs163:"
CODES = {"0", "5", "95", "96"}
# C0 (timeout provenance) .. C8 (explicit-code passthrough). C0 is a PRECONDITION, not a
# peer: until it is green no other control's rc may be read, because a missing `timeout`
# yields the same 96 that an unreachable endpoint does and would confound C6.
N_CONTROLS = 9

# The retired endpoint is assembled, never written as one literal: this stage deletes that
# string, so a source that contains it is inside its own residue denominator.
_EH = "mas" + "ter"
_EP = "80" + "81"
EP = _EH + ":" + _EP

A_PRE = (
    "    ( timeout 15 bash -c '(exec 3<>/dev/tcp/" + _EH + "/" + _EP + ")' ) || \\\n"
    "      fs_die \"standing-rule tripwire: " + EP + " is not reachable from here (s9). Refusing a Slurm launch.\""
)

B_PRE = 'fs_die() { echo "FATAL: $*" >&2; exit 1; }'

HEADER_PRE = (
    "#   s9  the " + EP + " tripwire is a STANDING RULE before any Slurm submit.\n"
    "#       It is moot off-Slurm (no prolog runs) and must NOT be deleted from the\n"
    "#       sbatch path — the slurm arm enforces it, timeout-bounded so a dead\n"
    "#       " + _EH + " fails CLOSED instead of hanging a job that could have been good.\n"
)
HEADER_REPL = (
    "#   s9  FS_FABRIC_TRIPWIRE is a STANDING RULE knob before any Slurm submit:\n"
    "#       required, no default; host:port is probed, the sentinel none is a logged\n"
    "#       declaration by THIS ESTATE that no pre-launch fabric tripwire exists. It is\n"
    "#       moot off-Slurm and must NOT be deleted from the sbatch path — timeout-bounded,\n"
    "#       so a dead endpoint fails CLOSED instead of hanging a job that could have been good.\n"
)

FS_DIE_REPL = (
    "# fs163: the plane contract is 0=PASS, 5=RED, 95=UNMEASURED, 96=REFUSE; bare exit 1\n"
    "# made every backend refusal indistinguishable from an interpreter error. The numeric\n"
    "# prefix is opt-in, so the existing call sites keep their meaning and gain REFUSE (96);\n"
    "# a message whose first word is a contract token would be swallowed as an rc and G4 refuses it.\n"
    "fs_die() {\n"
    "  local rc=96\n"
    "  case \"${1:-}\" in 0|5|95|96) rc=\"$1\"; shift ;; esac\n"
    "  echo \"FATAL[$rc]: $*\" >&2\n"
    "  exit \"$rc\"\n"
    "}"
)

TRIPWIRE_REPL = (
    "    # fs163: FS_FABRIC_TRIPWIRE is REQUIRED, NO DEFAULT -- the FS_ALLOWED_NODE /\n"
    "    # FS_CONTAINER_RUNTIME contract. An unconfigured guard is a disabled standing rule.\n"
    "    [[ -n \"${FS_FABRIC_TRIPWIRE:-}\" ]] || \\\n"
    "      fs_die 96 \"FS_FABRIC_TRIPWIRE is unset/empty (required, no default by design). The framework refuses to guess a cluster's fabric topology; set host:port, or the sentinel none to declare no pre-launch fabric tripwire.\"\n"
    "    if [[ \"${FS_FABRIC_TRIPWIRE}\" == \"none\" ]]; then\n"
    "      printf 'NOTICE: fs163: THIS ESTATE DECLARES no pre-launch fabric tripwire (FS_FABRIC_TRIPWIRE=none)\\n'\n"
    "    else\n"
    "      # Validate before use: this value is interpolated into a bash -c string, and an\n"
    "      # unvalidated interpolation is how a knob becomes a command channel.\n"
    "      fs_trip=\"${FS_FABRIC_TRIPWIRE}\"\n"
    "      if [[ ! \"$fs_trip\" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*:[0-9]{1,5}$ ]]; then\n"
    "        fs_die 96 \"malformed FS_FABRIC_TRIPWIRE (need host:port or none): '$fs_trip' -- refusing before it reaches bash -c\"\n"
    "      fi\n"
    "      fs_trip_host=\"${fs_trip%:*}\"; fs_trip_port=\"${fs_trip##*:}\"\n"
    "      if (( 10#$fs_trip_port < 1 || 10#$fs_trip_port > 65535 )); then\n"
    "        fs_die 96 \"malformed FS_FABRIC_TRIPWIRE port in '$fs_trip' (need 1-65535) -- refusing before it reaches bash -c\"\n"
    "      fi\n"
    "      ( timeout 15 bash -c \"(exec 3<>/dev/tcp/${fs_trip_host}/${fs_trip_port})\" ) || \\\n"
    "        fs_die 96 \"standing-rule tripwire: ${fs_trip_host}:${fs_trip_port} is not reachable from here (s9). Refusing a Slurm launch.\"\n"
    "      printf 'NOTICE: fs163: fabric tripwire reached at %s:%s; PASS is attributable, not the absence of a failure\\n' \"$fs_trip_host\" \"$fs_trip_port\"\n"
    "    fi"
)

MALFORMED = ["nocolon", "host:", ":" + _EP, "host:0", "host:70000", "host:8o81", "ho st:" + _EP]


def _stderr(msg: str) -> None:
    print(msg, file=sys.stderr)


def _transform(text: str) -> tuple[str, dict[str, int], bool]:
    if MARK in text and A_PRE not in text and B_PRE not in text:
        return text, {"a": 0, "b": 0, "header": 0}, True
    counts = {"a": text.count(A_PRE), "b": text.count(B_PRE), "header": text.count(HEADER_PRE)}
    new = text.replace(HEADER_PRE, HEADER_REPL, 1).replace(A_PRE, TRIPWIRE_REPL, 1).replace(B_PRE, FS_DIE_REPL, 1)
    return new, counts, False


def _g4_first_words(text: str) -> list[str]:
    bad = []
    for ln in text.splitlines():
        s = ln.strip()
        if "fs_die()" in s or "fs_die" not in s or s.startswith("#"):
            continue
        m = re.search(r"\bfs_die\s+(.+)$", s)
        if not m:
            continue
        toks = m.group(1).split()
        if toks and toks[0] in CODES:
            toks = toks[1:]
        if not toks:
            continue
        first = toks[0].lstrip("\"'")
        if first.split(":", 1)[0] in CODES or first in CODES:
            bad.append(s[:120])
    return bad


def _run_bash(script: str, env: dict[str, str] | None, cwd: str) -> subprocess.CompletedProcess[str]:
    e = dict(os.environ)
    e.pop("FS_FABRIC_TRIPWIRE", None)
    if env:
        e.update(env)
    return subprocess.run(["bash", script], capture_output=True, text=True, cwd=cwd, env=e, timeout=30)


def _controls(new: str) -> tuple[int, list[str]]:
    notes: list[str] = []
    ok = 0
    if FS_DIE_REPL not in new or TRIPWIRE_REPL not in new:
        return 0, ["controls not buildable: patched fs_die/tripwire block absent"]
    del new
    # The shipped block calls `timeout`, which is coreutils and therefore present on the
    # Linux estate this backend targets but ABSENT on a macOS build host. Without this,
    # C5 fails with `timeout: command not found` and -- far worse -- C6 PASSES for the
    # wrong reason: it wants rc=96 from an unreachable port and gets rc=96 from a missing
    # binary. A MUST_FIRE that fires for the wrong reason is a confounded control, so the
    # harness resolves a real timeout or supplies a shim with the same contract, and says
    # in the transcript which one the controls ran against.
    with tempfile.TemporaryDirectory(prefix="fs163-") as td:
        hp = pathlib.Path(td) / "harness.sh"
        sent = pathlib.Path(td) / "injected.was.here"

        timeout_impl = shutil.which("timeout") or shutil.which("gtimeout")
        if timeout_impl:
            shim = ""
            notes.append(f"C0 timeout provenance: PASS controls ran against {timeout_impl}")
        else:
            shim = (
                "timeout() {\n"
                "  local _d=\"$1\"; shift\n"
                "  \"$@\" & local _p=$!\n"
                "  ( sleep \"$_d\"; kill -9 \"$_p\" 2>/dev/null ) & local _w=$!\n"
                "  wait \"$_p\"; local _rc=$?\n"
                "  kill \"$_w\" 2>/dev/null; wait \"$_w\" 2>/dev/null\n"
                "  return \"$_rc\"\n"
                "}\n"
            )
            notes.append("C0 timeout provenance: PASS no coreutils timeout on this build host; "
                         "controls ran against a bash shim with the same bounded-exec contract")
        harness = ("#!/usr/bin/env bash\nset -uo pipefail\n" + shim
                   + FS_DIE_REPL + "\n" + TRIPWIRE_REPL + "\nexit 0\n")
        hp.write_text(harness, "utf-8")
        # C6 must be attributable: prove the harness's timeout can actually run a command
        # before any control is allowed to read a nonzero rc as "endpoint unreachable".
        probe = pathlib.Path(td) / "tprobe.sh"
        probe.write_text("#!/usr/bin/env bash\nset -uo pipefail\n" + shim + "timeout 5 true\n", "utf-8")
        tp = _run_bash(str(probe), None, td)
        if tp.returncode != 0:
            notes.append(f"C0 timeout provenance: FAIL harness timeout unusable rc={tp.returncode} "
                         f"{tp.stderr.strip()[:120]}")
            return 0, notes
        ok += 1

        def expect(name: str, env: dict[str, str] | None, rc: int, needle: str | None = None) -> bool:
            r = _run_bash(str(hp), env, td)
            good = r.returncode == rc and (needle is None or needle in (r.stdout + r.stderr))
            notes.append(f"{name}: rc={r.returncode} expected={rc} " + ("PASS" if good else "FAIL " + (r.stderr.strip()[:140])))
            return good

        if expect("C1 unset", None, 96, "required, no default"):
            ok += 1
        if expect("C2 empty", {"FS_FABRIC_TRIPWIRE": ""}, 96, "required, no default"):
            ok += 1
        if expect("C3 none", {"FS_FABRIC_TRIPWIRE": "none"}, 0, "THIS ESTATE DECLARES no pre-launch fabric tripwire"):
            ok += 1
        c4 = True
        for bad in MALFORMED + ["$(touch " + str(sent) + ")"]:
            if not expect(f"C4 {bad!r}", {"FS_FABRIC_TRIPWIRE": bad}, 96, "malformed FS_FABRIC_TRIPWIRE"):
                c4 = False
        if sent.exists():
            c4 = False
            notes.append("C4 injection: FAIL sentinel file exists; command substitution executed")
        else:
            notes.append("C4 injection: PASS sentinel absent; refused before bash -c")
        if c4:
            ok += 1

        ls = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        ls.bind(("127.0.0.1", 0))
        ls.listen(1)
        port = ls.getsockname()[1]
        try:
            if expect("C5 reachable", {"FS_FABRIC_TRIPWIRE": f"127.0.0.1:{port}"}, 0, "fabric tripwire reached"):
                ok += 1
        finally:
            ls.close()

        cs = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        cs.bind(("127.0.0.1", 0))
        closed = cs.getsockname()[1]
        cs.close()
        if expect("C6 closed", {"FS_FABRIC_TRIPWIRE": f"127.0.0.1:{closed}"}, 96, "not reachable"):
            ok += 1

        die_h = pathlib.Path(td) / "die.sh"
        die_h.write_text("#!/usr/bin/env bash\n" + FS_DIE_REPL + "\nfs_die \"guard declined to proceed\"\n", "utf-8")
        r = _run_bash(str(die_h), None, td)
        good = r.returncode == 96 and "FATAL[96]: guard declined to proceed" in r.stderr
        notes.append(f"C7 default rc: rc={r.returncode} expected=96 " + ("PASS" if good else "FAIL"))
        ok += int(good)
        die_h.write_text("#!/usr/bin/env bash\n" + FS_DIE_REPL + "\nfs_die 95 \"measurement is inconclusive\"\n", "utf-8")
        r = _run_bash(str(die_h), None, td)
        good = r.returncode == 95 and "FATAL[95]: measurement is inconclusive" in r.stderr
        notes.append(f"C8 explicit 95: rc={r.returncode} expected=95 msg-intact={good} " + ("PASS" if good else "FAIL"))
        ok += int(good)
    return ok, notes


def main() -> int:
    # build_h100_plane.sh invokes every stage as `python3 <stage>` with NO arguments, so
    # bare invocation must APPLY. Requiring a flag here would have made the stage a no-op
    # inside the build while passing by hand -- the #86 orphan shape, one layer up.
    argv = sys.argv[1:]
    if argv not in ([], ["--apply"], ["--check"]):
        _stderr("usage: patch_fabric_tripwire.py [--apply|--check]   (no argument == --apply)")
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
    gres.append(("G1", counts["a"] == 1, f"tripwire pre-image count={counts['a']} need=1"))
    gres.append(("G2", counts["b"] == 1, f"fs_die pre-image count={counts['b']} need=1"))
    gres.append(("G2h", counts["header"] == 1, f"header s9 pre-image count={counts['header']} need=1"))
    residue = new.count(EP) + new.count(_EH + "/" + _EP)
    gres.append(("G3", residue == 0, f"post-image retired-endpoint residue={residue} need=0"))
    bad = _g4_first_words(text)
    gres.append(("G4", not bad, f"fs_die first-word contract-token sites={len(bad)} " + ("; ".join(bad[:3]) if bad else "")))
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
        print(f"verdict: READY  {sum(counts.values())} site(s) would be rewritten, {gates}/{len(gres)} static gates, {cok}/{N_CONTROLS} controls, residue {residue}")
        return 0
    TARGET.write_text(new, "utf-8")
    print(f"{MARK} {sum(counts.values())} site(s) rewritten, {gates}/{len(gres)} static gates, {cok}/{N_CONTROLS} controls, residue {residue}")
    return 0


def _guarded() -> int:
    # This stage exists to stop `exit 1` from collapsing four states into one, so it
    # must not do that itself: an unhandled exception is a REFUSE, not a bare rc=1.
    try:
        return main()
    except Exception as exc:  # noqa: BLE001 - deliberate: no traceback may escape as rc=1
        _stderr(f"REFUSE 96: {MARK} stage raised {type(exc).__name__}: {exc}")
        return 96


if __name__ == "__main__":
    raise SystemExit(_guarded())