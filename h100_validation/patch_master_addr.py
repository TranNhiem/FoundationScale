#!/usr/bin/env python3
"""#168: mint MASTER_ADDR at the converged tail of fs_backend_init, on every allocation arm.

WHAT / WHY. fs_compose_launch's torchrun arm composes with ${MASTER_ADDR:?...} and its
own message names fs_backend_init as the producer, but fs_backend_init derived MASTER_ADDR
only on the off-Slurm (local allocation) arm. On a Slurm allocation nothing produced it
and the shell's :? expansion aborted the composer after every real probe had passed --
plane resolution, bind verification, the drain gate, container entry, the torch-provenance
and NVML/device-count probes, and a real 8-rank NCCL all_reduce. This is fs116's
acknowledged asymmetry one variable over: fs116's own comment admits that deriving
MASTER_PORT next to MASTER_ADDR "would have covered only the off-Slurm branch"; MASTER_ADDR
was never given the converged-tail treatment MASTER_PORT got.

The fix inserts an fs168 mint block immediately before the fs116 MASTER_PORT mint, so the
rendezvous pair is minted in one place on every allocation arm. A non-empty MASTER_ADDR
always wins and the workload manager is not consulted; otherwise, on the Slurm arm, the
address is derived from ground truth -- the FIRST host of `scontrol show hostnames` over
the job's node list, the node_rank 0 rendezvous convention torchrun expects.
SLURMD_NODENAME is deliberately not a fallback: it names the LOCAL node, so on a
multi-node job every node would elect itself master and the ranks would split into N
one-node worlds that hang instead of erroring -- a silent failure, worse than the refusal.
The block fails closed with fs_die 96 when scontrol is absent, when the node list is
unset/empty, or when the expansion yields no host, and on BOTH arms asserts MASTER_ADDR is
non-empty before exporting it, so an empty value can no longer travel to the composer.

This stage refuses to write when the fs116 anchor or the composer consumer is absent or
multiplied, when the off-Slurm export it must preserve is not present exactly once before
and after, when the mint would land outside fs_backend_init, when bash -n rejects the
result, when the transform is not byte-idempotent, when an added fs_die lacks an explicit
leading contract code, when an added line carries an estate literal, or when any executed
control is not observed. It measures 95 only when the target cannot be read at all.
"""

from __future__ import annotations

import difflib
import os
import pathlib
import re
import subprocess
import sys
import tempfile

TARGET = pathlib.Path(__file__).resolve().parent / "h100" / "gen" / "fs_container_backend.bound.sh"
MARK = "fs168:"
CODES = {"0", "5", "95", "96"}
# C0 (harness provenance PRECONDITION) + C1..C7 + C8a/C8b (the defect, both arms).
N_CONTROLS = 10

# Needles are ASSEMBLED, never written as one literal: this stage counts them, and a
# source that contains its own needle is inside its own denominator.
ANCHOR = "# --- fs116: mint " + "MASTER" + "_PORT"
CONSUMER = "${" + "MASTER_ADDR" + ":?" + "MASTER_ADDR must be exported by fs_backend_init before composition" + "}"
OFFSLURM = "export MASTER_ADDR=${MASTER_ADDR:-" + "127.0.0.1}"

OPEN_MARK = "# --- fs168: mint " + "MASTER" + "_ADDR"
CLOSE_MARK = "# --- end fs168 ---"

# Obviously synthetic harness values; they never enter the patched file.
H1 = "fs168stub-a.invalid"
H2 = "fs168stub-b.invalid"
PRESET = "fs168preset.invalid"

MINT_BLOCK = (
    "  # --- fs168: mint MASTER_ADDR ------------------------------------------------\n"
    "  # fs_compose_launch's torchrun arm expands ${MASTER_ADDR:?...} and names\n"
    "  # fs_backend_init as the producer, but the only producer was the off-Slurm arm's\n"
    "  # local default; on a Slurm allocation nothing minted it and the composer died in\n"
    "  # expansion after every real probe had passed. This is fs116's acknowledged\n"
    "  # asymmetry, one variable over: fs116's own comment admits that deriving\n"
    "  # MASTER_PORT next to MASTER_ADDR \"would have covered only the off-Slurm branch\"\n"
    "  # -- MASTER_ADDR was never given the converged-tail treatment MASTER_PORT got.\n"
    "  # Minted HERE, at the same converged tail, so every allocation arm passes through\n"
    "  # exactly one mint.\n"
    "  #\n"
    "  # The FIRST host of the allocation is the rendezvous convention torchrun expects:\n"
    "  # node_rank 0 binds the store there and every other rank connects to it. scontrol\n"
    "  # show hostnames expands the job's node list one host per line; head -n 1 takes\n"
    "  # the first.\n"
    "  #\n"
    "  # SLURMD_NODENAME is deliberately NOT a fallback: it names the LOCAL node, so on a\n"
    "  # multi-node job every node would elect itself master and the ranks would split\n"
    "  # into N one-node worlds that hang instead of erroring -- a silent failure, which\n"
    "  # is worse than the refusal.\n"
    "  #\n"
    "  # A launcher that already derived MASTER_ADDR under sbatch wins: the derivation\n"
    "  # runs only when the variable is empty, and the workload manager is not consulted\n"
    "  # when it is not needed.\n"
    "  if [[ -z \"${MASTER_ADDR:-}\" ]]; then\n"
    "    if [[ \"$FS_ALLOCATION\" == slurm ]]; then\n"
    "      command -v scontrol >/dev/null 2>&1 || fs_die 96 \"fs_backend_init: cannot derive MASTER_ADDR -- scontrol is not on PATH on a Slurm allocation; refusing to guess a rendezvous host\"\n"
    "      [[ -n \"${SLURM_JOB_NODELIST:-}\" ]] || fs_die 96 \"fs_backend_init: cannot derive MASTER_ADDR -- SLURM_JOB_NODELIST is unset/empty on a Slurm allocation; refusing to guess a rendezvous host\"\n"
    "      MASTER_ADDR=\"$(scontrol show hostnames \"$SLURM_JOB_NODELIST\" | head -n 1)\"\n"
    "      [[ -n \"$MASTER_ADDR\" ]] || fs_die 96 \"fs_backend_init: scontrol show hostnames '$SLURM_JOB_NODELIST' produced no host; refusing to mint an empty MASTER_ADDR\"\n"
    "    fi\n"
    "  fi\n"
    "  [[ -n \"${MASTER_ADDR:-}\" ]] || fs_die 96 \"fs_backend_init: MASTER_ADDR is empty at the converged tail (FS_ALLOCATION='$FS_ALLOCATION'); refusing to let an empty rendezvous address reach the composer\"\n"
    "  export MASTER_ADDR\n"
    "  # --- end fs168 ---\n"
)

FORBIDDEN = [
    ("filesystem-root literal", re.compile(r"/(?:home|Users|root|data|mnt|nfs|lustre|scratch|srv|opt)/")),
    ("IP literal", re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}\b")),
    ("DNS-style host name", re.compile(r"\b[a-z0-9][a-z0-9-]*(?:\.[a-z0-9][a-z0-9-]*)+\.(?:com|net|org|edu|gov|io|local|lan)\b", re.I)),
]


def _stderr(msg: str) -> None:
    print(msg, file=sys.stderr)


def _transform(text: str) -> tuple[str, dict[str, int], bool]:
    if OPEN_MARK in text:
        return text, {"anchor": text.count(ANCHOR)}, True
    counts = {"anchor": text.count(ANCHOR)}
    lines = text.splitlines(keepends=True)
    for i, ln in enumerate(lines):
        if ANCHOR in ln:
            lines.insert(i, MINT_BLOCK)
            break
    return "".join(lines), counts, False


def _added_lines(pre: str, post: str) -> list[str]:
    return [ln[2:] for ln in difflib.ndiff(pre.splitlines(), post.splitlines()) if ln.startswith("+ ")]


def _g7_bad_fs_die(added: list[str]) -> list[str]:
    bad = []
    for ln in added:
        s = ln.strip()
        if s.startswith("#") or "fs_die" not in s:
            continue
        m = re.search(r"\bfs_die\s+(.+)$", s)
        if not m:
            continue
        toks = m.group(1).split()
        if not toks or toks[0] not in CODES:
            bad.append("no explicit leading contract code: " + s[:100])
            continue
        rest = toks[1:]
        if rest:
            first = rest[0].lstrip("\"'")
            if first in CODES or first.split(":", 1)[0] in CODES:
                bad.append("message begins with a bare contract token: " + s[:100])
    return bad


def _g8_forbidden(added: list[str]) -> list[str]:
    bad = []
    for ln in added:
        for label, rx in FORBIDDEN:
            if rx.search(ln):
                bad.append(f"{label}: {ln.strip()[:100]}")
    return bad


def _slice_func(text: str, header: str) -> str:
    lines = text.splitlines()
    for i, ln in enumerate(lines):
        if ln.startswith(header):
            out = [ln]
            for ln2 in lines[i + 1:]:
                out.append(ln2)
                if ln2 == "}":
                    return "\n".join(out)
            return ""
    return ""


def _slice_block(text: str, start_needle: str, end_needle: str) -> str:
    out: list[str] = []
    on = False
    for ln in text.splitlines():
        if start_needle in ln:
            on = True
        if on:
            out.append(ln)
        if on and end_needle in ln:
            break
    return "\n".join(out)


def _run_bash(script: str, path: str | None, cwd: str) -> subprocess.CompletedProcess[str]:
    e = dict(os.environ)
    for v in ("MASTER_ADDR", "SLURM_JOB_NODELIST", "FS_ALLOCATION", "MASTER_PORT",
              "SLURM_NNODES", "SLURM_NODEID", "FS_PYTHON"):
        e.pop(v, None)
    if path is not None:
        e["PATH"] = path
    return subprocess.run(["bash", script], capture_output=True, text=True, cwd=cwd, env=e, timeout=30)


def _controls(new: str) -> tuple[int, list[str]]:
    notes: list[str] = []
    ok = 0
    fs_die_def = _slice_func(new, "fs_die()")
    mint = _slice_block(new, OPEN_MARK, CLOSE_MARK)
    composer = _slice_func(new, "fs_compose_launch()")
    if not fs_die_def or not mint or not composer:
        return 0, ["C0 provenance: FAIL could not slice fs_die / the fs168 mint block / fs_compose_launch "
                   "out of the patched text; controls not interpretable"]
    with tempfile.TemporaryDirectory(prefix="fs168-") as td:
        bindir = pathlib.Path(td) / "bin"
        bindir.mkdir()
        sentinel = pathlib.Path(td) / "scontrol.was.here"
        stub = bindir / "scontrol"
        healthy = ("#!/usr/bin/env bash\n"
                   "touch '" + str(sentinel) + "'\n"
                   "printf '%s\\n' '" + H1 + "' '" + H2 + "'\n")
        stub.write_text(healthy, "utf-8")
        stub.chmod(0o755)
        stubpath = str(bindir) + ":/usr/bin:/bin"
        nostub = "/usr/bin:/bin"

        # C0 PROVENANCE precondition: bash is an external binary here, and a missing bash
        # or a non-executable stub yields the same nonzero rc a real refusal does, which
        # would confound every MUST_FIRE below. Prove the harness mechanism -- the same
        # bash, the same stub PATH, and the fs_die shape actually extracted from the
        # target -- before any other control's rc may be interpreted.
        probe = pathlib.Path(td) / "probe.sh"
        probe.write_text("#!/usr/bin/env bash\nset -uo pipefail\n" + fs_die_def
                         + "\ncommand -v scontrol >/dev/null 2>&1 || exit 97\nexit 0\n", "utf-8")
        try:
            p = _run_bash(str(probe), stubpath, td)
        except OSError as exc:
            notes.append(f"C0 provenance: FAIL harness bash could not run ({exc}); controls not interpretable")
            return 0, notes
        if p.returncode != 0:
            notes.append(f"C0 provenance: FAIL trivial harness rc={p.returncode} through the control bash + stub PATH; "
                         "no MUST_FIRE below would be attributable, so the controls were not interpretable")
            return 0, notes
        ok += 1
        notes.append("C0 provenance: PASS harness bash, stub PATH and the fs_die shape extracted from the target are functional")

        hp = pathlib.Path(td) / "case.sh"

        def mint_script(alloc: str, nodelist: str | None, preset: str | None) -> str:
            lines = ["#!/usr/bin/env bash", "set -uo pipefail", fs_die_def,
                     "FS_ALLOCATION='" + alloc + "'"]
            lines.append("SLURM_JOB_NODELIST='" + nodelist + "'" if nodelist is not None
                         else "unset SLURM_JOB_NODELIST")
            if preset is not None:
                lines.append("MASTER_ADDR='" + preset + "'")
            lines += [mint, "printf 'RESULT=%s\\n' \"${MASTER_ADDR:-}\"", "exit 0"]
            return "\n".join(lines) + "\n"

        def run_case(name: str, script: str, path: str, rc: int, check=None) -> bool:
            hp.write_text(script, "utf-8")
            r = _run_bash(str(hp), path, td)
            good = r.returncode == rc and (check is None or check(r))
            notes.append(f"{name}: rc={r.returncode} expected={rc} "
                         + ("PASS" if good else "FAIL " + (r.stderr.strip()[:140])))
            return good

        sentinel.unlink(missing_ok=True)
        ok += int(run_case("C1 slurm derive first host", mint_script("slurm", "synth-[a-b]", None), stubpath, 0,
                           lambda r: r.stdout.strip() == "RESULT=" + H1 and H2 not in r.stdout and sentinel.exists()))
        sentinel.unlink(missing_ok=True)
        ok += int(run_case("C2 slurm preset short-circuits", mint_script("slurm", "synth-[a-b]", PRESET), stubpath, 0,
                           lambda r: r.stdout.strip() == "RESULT=" + PRESET and not sentinel.exists()))
        ok += int(run_case("C3 scontrol absent from PATH", mint_script("slurm", "synth-[a-b]", None), nostub, 96,
                           lambda r: "scontrol" in r.stderr))
        ok += int(run_case("C4 node list unset", mint_script("slurm", None, None), stubpath, 96))
        stub.write_text("#!/usr/bin/env bash\nexit 0\n", "utf-8")
        stub.chmod(0o755)
        ok += int(run_case("C5 expansion produces no host", mint_script("slurm", "synth-[a-b]", None), stubpath, 96))
        stub.write_text(healthy, "utf-8")
        stub.chmod(0o755)
        sentinel.unlink(missing_ok=True)
        ok += int(run_case("C6 local arm preset preserved", mint_script("local", None, PRESET), stubpath, 0,
                           lambda r: r.stdout.strip() == "RESULT=" + PRESET and not sentinel.exists()))
        ok += int(run_case("C7 local arm empty refuses", mint_script("local", None, None), stubpath, 96))

        base = ["#!/usr/bin/env bash", "set -uo pipefail", fs_die_def,
                "FS_ALLOCATION='slurm'", "SLURM_JOB_NODELIST='synth-[a-b]'", "MASTER_PORT=29500"]
        script_a = "\n".join(base + [mint, composer, "fs_compose_launch torchrun 8 'synth-engine'", "exit 0"]) + "\n"
        ok += int(run_case("C8a defect closed end to end", script_a, stubpath, 0,
                           lambda r: "--master_addr=" + H1 in r.stdout and H2 not in r.stdout))
        script_b = "\n".join(base + [composer, "fs_compose_launch torchrun 8 'synth-engine'", "exit 0"]) + "\n"
        hp.write_text(script_b, "utf-8")
        r = _run_bash(str(hp), stubpath, td)
        good = r.returncode != 0
        notes.append(f"C8b pre-image composer without the mint: rc={r.returncode} expected=nonzero "
                     + ("PASS" if good else
                        "FAIL composer passed with no producer; the premise is stale and the stage must refuse"))
        ok += int(good)
    return ok, notes


def main() -> int:
    # build_h100_plane.sh invokes every stage as `python3 <stage>` with NO arguments, so
    # bare invocation must APPLY. Requiring a flag here would have made the stage a no-op
    # inside the build while passing by hand -- the #86 orphan shape, one layer up.
    argv = sys.argv[1:]
    if argv not in ([], ["--apply"], ["--check"]):
        _stderr("usage: patch_master_addr.py [--apply|--check]   (no argument == --apply)")
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
    gres.append(("G1", counts["anchor"] == 1, f"fs116 anchor count={counts['anchor']} need=1"))
    gres.append(("G2", text.count(CONSUMER) == 1,
                 f"composer MASTER_ADDR consumer count={text.count(CONSUMER)} need=1 "
                 "(a consumer with no producer is this stage's whole premise)"))
    gres.append(("G3", text.count(OFFSLURM) == 1 and new.count(OFFSLURM) == 1,
                 f"off-Slurm MASTER_ADDR export pre={text.count(OFFSLURM)} post={new.count(OFFSLURM)} "
                 "need=1/1 unchanged (the fix ADDS a producer, it does not move one)"))
    lines = new.splitlines()
    hdr = next((i for i, l in enumerate(lines) if l.startswith("fs_backend_init()")), None)
    close = next((i for i in range(hdr + 1, len(lines)) if lines[i] == "}"), None) if hdr is not None else None
    om = next((i for i, l in enumerate(lines) if OPEN_MARK in l), None)
    cm = next((i for i, l in enumerate(lines) if CLOSE_MARK in l), None)
    ai = next((i for i, l in enumerate(lines) if ANCHOR in l), None)
    g4 = (new.count(OPEN_MARK) == 1 and new.count(CLOSE_MARK) == 1
          and None not in (hdr, close, om, cm, ai)
          and hdr < om < cm < ai < close)  # type: ignore[operator]
    gres.append(("G4", g4, f"markers once={new.count(OPEN_MARK)}/{new.count(CLOSE_MARK)} "
                           f"header={hdr} open={om} close={cm} anchor={ai} brace={close} "
                           "(the mint must sit inside fs_backend_init, before the fs116 mint)"))
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
    added = _added_lines(text, new)
    bad7 = _g7_bad_fs_die(added)
    gres.append(("G7", not bad7, f"added fs_die sites missing an explicit leading code or "
                                 f"carrying a contract-token message={len(bad7)} " + "; ".join(bad7[:2])))
    bad8 = _g8_forbidden(added)
    gres.append(("G8", not bad8, f"added lines carrying host/node/partition/user/fs-root "
                                 f"literals={len(bad8)} " + "; ".join(bad8[:2])))

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
        print(f"verdict: READY  {sum(counts.values())} site(s) would be rewritten, "
              f"{gates}/{len(gres)} static gates, {cok}/{N_CONTROLS} controls")
        return 0
    TARGET.write_text(new, "utf-8")
    print(f"{MARK} {sum(counts.values())} site(s) rewritten, {gates}/{len(gres)} static gates, {cok}/{N_CONTROLS} controls")
    return 0


def _guarded() -> int:
    # This stage exists to stop an empty MASTER_ADDR from reaching a guard that collapses
    # states, so it must not collapse its own: an unhandled exception is a REFUSE, not a
    # bare rc=1.
    try:
        return main()
    except Exception as exc:  # noqa: BLE001 - deliberate: no traceback may escape as rc=1
        _stderr(f"REFUSE 96: {MARK} stage raised {type(exc).__name__}: {exc}")
        return 96


if __name__ == "__main__":
    raise SystemExit(_guarded())
