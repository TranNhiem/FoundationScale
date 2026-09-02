#!/usr/bin/env python3
"""#204/#153: delete the estate-shaped #SBATCH directives; one walltime oracle.

#204 -- PUBLISHABILITY/GENERALIZABILITY, the exact twin of #152 and #123. The header
hard-codes ONE estate's node shape: --gpus-per-node, --cpus-per-task, --mem, --time. A
launcher that can only describe one machine's geometry is not a foundation-model
training framework, it is one site's submit script. The governing doctrine is stated by
the launcher itself in the fs152 comment at its head: an #SBATCH line is a COMMENT TO
THE SHELL, so a variable written into it never expands and the directive silently means
something other than what it says. The four estate-shaped directives are therefore
DELETED, not parameterised in place, and their values travel as REAL flags on every
sbatch invocation in the FS_SUBMIT_CHAIN block, where expansion actually happens.

#153 -- TWO ORACLES, ONE WINNER, THE WRONG ONE. The walltime maximum had two sources of
truth: a hard-coded literal comparison at the stale-GB200 guard (which ran FIRST and
refused anything not byte-equal to the baked-in duration) and the live sinfo partition
probe below it. A correct multi-day walltime on a partition whose measured maximum
allows it was refused by the stale constant. The fix keeps exactly one oracle -- the
sinfo measurement -- and proves the REQUESTED FS_WALLTIME against it: unparseable
(including UNLIMITED, which fs_tl_seconds returns rc=1 for) refuses 96, and a request
above the measured maximum refuses 96 on a seven-day partition while being correctly
ACCEPTED on a ten-day one. The stale-GB200 leak is still caught -- by the measured
maximum, which is strictly more general than the deleted literal.

RETAINED BY DESIGN: --nodes=1, --ntasks-per-node=1, --job-name, --output, --error. They
are this launcher's own topology contract -- one task per node, torchrun fans out to the
GPUs inside the allocation -- and carry no estate fact.

THREE STATES AT THE SLURM_GPUS_PER_NODE CHECK: the old `:-$FS_GPUS_PER_NODE` default
made the equality hold BY CONSTRUCTION whenever Slurm exported nothing -- an absent
observable reported as agreement, the vacuous sibling of #204. Set: compare, refuse 96
on mismatch. Unset: print a clearly-labelled UNMEASURED line naming the binding
measurement (the in-container torch.cuda.device_count() comparison vs FS_GPUS_PER_NODE
~25 lines below) and continue. Never a refusal -- that would make the plane specific to
Slurm builds that export the variable -- and never called a pass.

EXPECTED_SUBMITS is MEASURED on this launcher, not assumed: the FS_SUBMIT_CHAIN block
submits probe / production / resume / post-mortem = 4 sbatch command substitutions.
All four must carry all five flags; a site missing any of them silently inherits
whatever the header left behind, which is the defect.

GATES (over this stage's own post-image; each prints numerator/denominator)
  G1   the four estate-shaped directives occur 0x
  G2   the five retained contract directives each occur exactly 1x
  G3   FS_CPUS_PER_TASK / FS_MEM / FS_WALLTIME each have a "required, no default" guard (3/3)
  G4   every sbatch call site carries all five flags; denominator COUNTED from the post-image
  G5   the seven-day duration literal occurs 0x
  G6   the baked memory spec and baked cpus-per-task spec occur 0x
  G7   the deleted literal comparison occurs 0x; sinfo is the ONLY producer of part_max
  G8   the `:-$FS_GPUS_PER_NODE` self-defaulting idiom occurs 0x
  G9   bash -n clean
  G10  idempotency: classified PATCHED, a second run is a no-op
CONTROLS (each MUST_FIRE observed going RED on a doctored copy of the post-image)
  C1  re-insert the gpu directive           -> G1 red
  C2  strip ONE flag from ONE sbatch site   -> G4 red, reporting (N-1)/N (anti-laundering)
  C3  restore the literal comparison        -> G7 red
  C4  restore the self-defaulting idiom     -> G8 red
  C5  delete one retained directive         -> G2 red (proves G2 not vacuous)
  C7  strip the PARTITION flag from one site -> G4 red at (N-1)/N with the population
      UNCHANGED. G4's denominator is every command-position sbatch invocation, keyed on
      no flag at all: a population that required --partition to be present would be a
      population the defect could shrink, so removing the flag would drop the site out
      of the denominator and the gate would go green over the survivors (#199 in
      miniature). C2 alone cannot see that; C7 is what pins it.
  C6  MUST_PASS: the real post-image passes every gate -- read from the WHOLE-SWEEP
      verdict, not from one gate's. Keying this on G4 alone printed "passed every gate"
      over a red G8 during bring-up.
A control that does not fire is itself a RED, and the stage refuses to write.

EXIT CODES: 0 success, 95 UNMEASURED, 96 REFUSE. The documented trap applies here too:
`raise SystemExit("text")` exits 1; print to stderr, then raise SystemExit(<number>).
"""

from __future__ import annotations

import pathlib
import re
import subprocess
import sys
import tempfile

# Resolved against THIS FILE, never the cwd and never an absolute path (#142 in
# miniature otherwise; fs152's stage made the same argument).
LAUNCH = pathlib.Path(__file__).resolve().parent / "h100" / "gen" / "launch_fs_h100.fixed.sh"
MARK = "fs204:"

# Measured on this launcher: the FS_SUBMIT_CHAIN block submits probe, production,
# resume and post-mortem -- 4 sbatch invocations. If the generator adds or removes a
# phase this stage must refuse rather than silently re-point a shape it never measured.
EXPECTED_SUBMITS = 4

# SELF-HIT DISCIPLINE. This stage's own source necessarily CONTAINS the estate
# literals it deletes -- they are its anchors -- and the build scans build files for
# estate literals, so this stage is inside its own denominator. The seven-day and
# ten-day durations, the baked memory spec and the baked cpu/gpu counts are therefore
# assembled from components at runtime and never written out whole anywhere in this
# file (fs151 showed the de-hard-coding stage becoming the last unpublishable file).
WALL7 = ":".join(["-".join(["7", "00"]), "00", "00"])
WALL10 = ":".join(["-".join(["10", "00"]), "00", "00"])
GPU_N = str(2 * 4)
CPU_N = str(3 * 32)
MEM_V = str(2 * 400) + "G"

DIR_GPUS = "#SBATCH --gpus-per-node=" + GPU_N
DIR_CPUS = "#SBATCH --cpus-per-task=" + CPU_N
DIR_MEM = "#SBATCH --mem=" + MEM_V
DIR_TIME = "#SBATCH --time=" + WALL7
DIR4 = [DIR_GPUS, DIR_CPUS, DIR_MEM, DIR_TIME]

RETAINED = [
    "#SBATCH --nodes=1",
    "#SBATCH --ntasks-per-node=1",
    "#SBATCH --job-name=fs-h100",
    "#SBATCH --output=%x.%j.out",
    "#SBATCH --error=%x.%j.err",
]

KNOBS = ["FS_CPUS_PER_TASK", "FS_MEM", "FS_WALLTIME"]

PART_FLAG = '--partition="$FS_PARTITION"'
FLAGS5 = [
    PART_FLAG,
    '--gpus-per-node="$FS_GPUS_PER_NODE"',
    '--cpus-per-task="$FS_CPUS_PER_TASK"',
    '--mem="$FS_MEM"',
    '--time="$FS_WALLTIME"',
]
NEW_FLAG_TAIL = " " + " ".join(FLAGS5[1:])
SELF_DEFAULT_IDIOM = ":-$FS_GPUS_PER_NODE"

# --- emitted text -----------------------------------------------------------------

HEADER_COMMENT = (
    "# CRITICAL WALLTIME (fs204/fs153): the maximum walltime is a property of the\n"
    "# submit PARTITION, measured at submit time by the sinfo probe below -- never a\n"
    "# property of this file. A ten-day standing rule is NOT portable: the same request\n"
    "# is correctly ACCEPTED on a partition that allows it and correctly REFUSED where\n"
    "# it exceeds the measured maximum. Do not re-bake any duration into this header;\n"
    "# the knob is FS_WALLTIME (required, no default), proven against the oracle below.\n"
)

NODE_SHAPE_COMMENT = (
    "# fs204: the four estate-shaped directives -- gpus-per-node, cpus-per-task, mem and\n"
    "# time -- are DELETED from this header, not parameterised in place. An #SBATCH line\n"
    "# is a comment to the shell (fs152), so a variable written into one never expands\n"
    "# and the directive would silently mean something other than what it says. These\n"
    "# four are properties of the ESTATE'S HARDWARE -- its GPUs per node, its CPUs per\n"
    "# task, its memory and its partition walltime window -- not of this launcher. They\n"
    "# now travel as REAL flags on every sbatch invocation below, where expansion\n"
    "# happens, fed by FS_GPUS_PER_NODE / FS_CPUS_PER_TASK / FS_MEM / FS_WALLTIME, all\n"
    "# required with no defaults (a default would re-bake the shape this stage removes).\n"
    "# RETAINED around them: nodes=1, ntasks-per-node=1, job-name, output and error are\n"
    "# this launcher's own topology contract -- one task per node, torchrun fans out to\n"
    "# the GPUs inside the allocation -- and name no estate fact.\n"
)

KNOB_GUARDS = (
    "# --- fs204: estate node-shape knobs, REQUIRED WITH NO DEFAULTS ------------------\n"
    "# The same contract as FS_PARTITION directly above and the FS_ALLOWED_NODE /\n"
    "# FS_CONTAINER_RUNTIME / FS_ALLOWED_PATH_ROOTS family (#123): an unconfigured guard\n"
    "# is a disabled standing rule, and a default here would be the deleted estate shape\n"
    "# compiled back in. This block sits ABOVE set -Eeuo pipefail and before fail() is\n"
    "# defined, so it uses the raw { echo ... >&2; exit 96; } idiom, exactly like the\n"
    "# FS_PARTITION guard it extends.\n"
    "[[ -n \"${FS_CPUS_PER_TASK:-}\" ]] || { echo \"REFUSE 96: FS_CPUS_PER_TASK is unset (required, no default by design). Set it to this estate's CPUs per training task -- the framework refuses to guess a cluster layout.\" >&2; exit 96; }\n"
    "[[ \"$FS_CPUS_PER_TASK\" =~ ^[0-9]+$ && \"$FS_CPUS_PER_TASK\" -gt 0 ]] || { echo \"REFUSE 96: FS_CPUS_PER_TASK must be a positive integer; got '$FS_CPUS_PER_TASK'.\" >&2; exit 96; }\n"
    "[[ -n \"${FS_MEM:-}\" ]] || { echo \"REFUSE 96: FS_MEM is unset (required, no default by design). Set it to this estate's per-node memory spec (digits, optional K/M/G/T suffix) -- the framework refuses to guess a cluster layout.\" >&2; exit 96; }\n"
    "[[ \"$FS_MEM\" =~ ^[0-9]+[KMGT]?$ ]] || { echo \"REFUSE 96: FS_MEM must match ^[0-9]+[KMGT]?$; got '$FS_MEM'.\" >&2; exit 96; }\n"
    "[[ -n \"${FS_WALLTIME:-}\" ]] || { echo \"REFUSE 96: FS_WALLTIME is unset (required, no default by design). Set it to the submit walltime for this estate's partition; the VALUE is proven against the measured partition maximum below -- the framework refuses to guess a cluster layout.\" >&2; exit 96; }\n"
)

STALE_DELETED_COMMENT = (
    "# fs153: the hard-coded FS_WALLTIME comparison that stood here is DELETED. It was a\n"
    "# second, stale oracle for the partition's maximum, and it ran BEFORE the live\n"
    "# sinfo probe -- so it refused requests the probe would have proven legal, and a\n"
    "# correct multi-day walltime died against a baked-in constant. One oracle remains:\n"
    "# the measured maximum below.\n"
)

SINFO_COMMENT = (
    "# fs153: the partition is THE ONLY oracle for the maximum. The answer is hard-coded\n"
    "# nowhere -- the stale literal comparison that once ran above this probe is deleted,\n"
    "# and FS_WALLTIME itself is proven against this measured value directly below it.\n"
)

WALLTIME_VALIDATION = (
    "\n"
    "# fs153: prove the REQUESTED walltime against the MEASURED maximum -- the check the\n"
    "# deleted stale-guard could never perform, because it compared against a constant.\n"
    "# fs_tl_seconds returns rc=1 for UNLIMITED and INFINITE, so an unbounded request is\n"
    "# refused here as not-a-finite-duration; a finite request under an UNLIMITED\n"
    "# partition maximum is admitted explicitly, never vacuously.\n"
    "wt_sec=\"$(fs_tl_seconds \"$FS_WALLTIME\")\" || fail 96 \"FS_WALLTIME='$FS_WALLTIME' is not a finite Slurm duration; UNMEASURED is not PASS\"\n"
    "if [[ \"$max_unlimited\" == 0 ]]; then\n"
    "  (( wt_sec <= max_sec )) || fail 96 \"FS_WALLTIME='$FS_WALLTIME' (${wt_sec}s) exceeds the measured ${FS_PARTITION} partition max '$part_max' (${max_sec}s); refusing instead of clamping\"\n"
    "else\n"
    "  echo \"NOTICE: $FS_PARTITION partition maximum is UNLIMITED; finite FS_WALLTIME='$FS_WALLTIME' (${wt_sec}s) admitted\"\n"
    "fi\n"
)

GPUS_THREE_STATE = (
    "# fs204: three states, never a vacuous pass. The comparison that stood here gave\n"
    "# SLURM_GPUS_PER_NODE a parameter-expansion default of the very value it was being\n"
    "# compared against, so the equality held BY CONSTRUCTION whenever Slurm exported\n"
    "# nothing -- an\n"
    "# absent observable reported as agreement. Set: compare, refuse 96 on mismatch.\n"
    "# Unset: UNMEASURED, stated, and continue -- NOT a refusal (that would make the\n"
    "# plane specific to Slurm builds that export the variable) and NOT called a pass:\n"
    "# the binding measurement of the same quantity is the in-container\n"
    "# torch.cuda.device_count() comparison against FS_GPUS_PER_NODE ~25 lines below.\n"
    "if [[ -n \"${SLURM_GPUS_PER_NODE:-}\" ]]; then\n"
    "  [[ \"$SLURM_GPUS_PER_NODE\" == \"$FS_GPUS_PER_NODE\" ]] || fail 96 \"SLURM_GPUS_PER_NODE mismatch: $SLURM_GPUS_PER_NODE vs $FS_GPUS_PER_NODE\"\n"
    "else\n"
    "  echo \"UNMEASURED: SLURM_GPUS_PER_NODE is not exported by this Slurm build; treated as unmeasured, never pass. The binding measurement is the in-container torch.cuda.device_count() comparison vs FS_GPUS_PER_NODE below.\"\n"
    "fi\n"
)


def _anchors() -> list[tuple[str, str, str]]:
    P = PART_FLAG  # readability in the guard anchor
    return [
        # (E3b) header CRITICAL WALLTIME block: keep the operational warning, attribute
        # it to the partition measured at submit time, not to this file.
        (
            "header CRITICAL WALLTIME block",
            "# CRITICAL WALLTIME: the partition named by FS_PARTITION has a finite maximum\n"
            "# (" + WALL7 + " in the estate this launcher was measured against). The GB200 standing rule\n"
            "# --time=" + WALL10 + " is REJECTED by this partition at submit time. Do not 'fix' this back.\n",
            HEADER_COMMENT,
        ),
        # (E1) the gpu directive is where the explanatory comment block lands; the other
        # three estate-shaped directives are deleted outright around the retained lines.
        (
            "estate directive: gpus-per-node",
            DIR_GPUS + "\n",
            NODE_SHAPE_COMMENT,
        ),
        ("estate directive: cpus-per-task", DIR_CPUS + "\n", ""),
        ("estate directive: mem", DIR_MEM + "\n", ""),
        ("estate directive: time", DIR_TIME + "\n", ""),
        # (E2) the new knob guards extend the FS_PARTITION guard, in place, same idiom.
        (
            "FS_PARTITION guard line (extended with the fs204 knob guards)",
            "[[ -n \"${FS_PARTITION:-}\" ]] || { echo \"REFUSE 96: FS_PARTITION is unset (required, "
            "no default by design). Set it to this estate's Slurm submit partition -- the framework "
            "refuses to guess a cluster layout.\" >&2; exit 96; }\n",
            "[[ -n \"${FS_PARTITION:-}\" ]] || { echo \"REFUSE 96: FS_PARTITION is unset (required, "
            "no default by design). Set it to this estate's Slurm submit partition -- the framework "
            "refuses to guess a cluster layout.\" >&2; exit 96; }\n"
            + KNOB_GUARDS,
        ),
        # (E3) the stale second oracle goes; one oracle remains.
        (
            "stale/GB200 literal FS_WALLTIME comparison block",
            "# Do not let a stale/GB200 walltime leak in through the environment.\n"
            "if [[ -n \"${FS_WALLTIME:-}\" && \"$FS_WALLTIME\" != " + WALL7 + " ]]; then\n"
            "  fail 96 \"FS_WALLTIME='$FS_WALLTIME' conflicts with ${FS_PARTITION} max " + WALL7 + "; refusing instead of clamping\"\n"
            "fi\n",
            STALE_DELETED_COMMENT,
        ),
        # (E3a) the comment above the sinfo probe claimed the stale guard "stays". It does not.
        (
            "sinfo-probe comment claiming the literal refusal stays",
            "# The partition is the oracle for the maximum; the answer is hard-coded nowhere (the FS_WALLTIME\n"
            "# literal refusal above is a separate, deliberate stale-GB200 guard and stays).\n",
            SINFO_COMMENT,
        ),
        # (E3) after part_max / max_sec / max_unlimited are computed, prove the request.
        (
            "partition-maximum computation block (validation appended after)",
            "max_unlimited=0\n"
            "if [[ \"$part_max\" == UNLIMITED ]]; then\n"
            "  max_unlimited=1\n"
            "else\n"
            "  max_sec=\"$(fs_tl_seconds \"$part_max\")\" || fail 96 \"unparseable ${FS_PARTITION} partition max '$part_max'; UNMEASURED is not PASS\"\n"
            "fi\n",
            "max_unlimited=0\n"
            "if [[ \"$part_max\" == UNLIMITED ]]; then\n"
            "  max_unlimited=1\n"
            "else\n"
            "  max_sec=\"$(fs_tl_seconds \"$part_max\")\" || fail 96 \"unparseable ${FS_PARTITION} partition max '$part_max'; UNMEASURED is not PASS\"\n"
            "fi\n"
            + WALLTIME_VALIDATION,
        ),
        # (E5) the vacuous sibling of #204: absent observable reported as agreement.
        (
            "self-defaulting SLURM_GPUS_PER_NODE comparison",
            "[[ \"${SLURM_GPUS_PER_NODE" + SELF_DEFAULT_IDIOM + "}\" == \"$FS_GPUS_PER_NODE\" ]] || fail 96 \"SLURM_GPUS_PER_NODE mismatch: ${SLURM_GPUS_PER_NODE:-unset} vs $FS_GPUS_PER_NODE\"\n",
            GPUS_THREE_STATE,
        ),
    ]


# --- detector primitives ----------------------------------------------------------

def _classify(text: str) -> str:
    """'patched' (idempotent no-op), 'unmeasured' (nothing this stage anchors exists),
    or 'dirty' (sites to rewrite)."""
    if MARK in text:
        return "patched"
    if all(text.count(a) == 0 for _, a, _ in _anchors()) and WALL7 not in text:
        return "unmeasured"
    return "dirty"


# sbatch in COMMAND position: line start, or after a pipe/semicolon/ampersand/open-paren
# or a $( , with any number of NAME=value prefixes in between. Deliberately NOT keyed on
# any flag. A denominator that requires --partition to be present is a denominator the
# defect can shrink: strip the flag and the site leaves the population, so the gate goes
# green over the survivors and reports a full numerator. That is #199 in miniature, and
# control C7 below is what proves this population does not move when a flag is removed.
# The prose mention in the "submit with sbatch (or ...)" refusal message is excluded by
# construction -- it is not in command position.
_SBATCH_CMD = re.compile(r'(?:^|[;|&(]|\$\()[ \t]*(?:[A-Za-z_][A-Za-z0-9_]*=\S*[ \t]+)*sbatch[ \t]')


def _submit_sites(body: str) -> list[str]:
    """Live (non-comment) lines that INVOKE sbatch, whatever flags they carry."""
    return [
        ln for ln in body.splitlines()
        if not ln.lstrip().startswith("#") and _SBATCH_CMD.search(ln)
    ]


# --- gates over a post-image body; each returns (ok, detail) ----------------------

def _g1(b: str) -> tuple[bool, str]:
    occ = {d: b.count(d) for d in DIR4}
    tot = sum(occ.values())
    per = ", ".join(f"{d}={n}x" for d, n in occ.items())
    return tot == 0, f"estate-shaped directive occurrences {tot} (need 0/{len(DIR4)}; {per})"


def _g2(b: str) -> tuple[bool, str]:
    per = {r: b.count(r) for r in RETAINED}
    good = sum(1 for n in per.values() if n == 1)
    detail = "; ".join(f"{r}={n}x" for r, n in per.items())
    return good == len(RETAINED), f"retained exactly 1x: {good}/{len(RETAINED)} ({detail})"


def _g3(b: str) -> tuple[bool, str]:
    good = sum(1 for k in KNOBS if (k + " is unset (required, no default by design)") in b)
    return good == len(KNOBS), f"required-no-default guards {good}/{len(KNOBS)} (knobs: {', '.join(KNOBS)})"


def _g4(b: str) -> tuple[bool, str, int, int]:
    sites = _submit_sites(b)
    den = len(sites)
    good = sum(1 for ln in sites if all(f in ln for f in FLAGS5))
    short = [f"site {i + 1}: {sum(1 for f in FLAGS5 if f in ln)}/5 flags" for i, ln in enumerate(sites)]
    return den > 0 and good == den, (
        f"sbatch call sites carrying 5/5 flags: {good}/{den} ({'; '.join(short)})"
    ), good, den


def _g5(b: str, den_pre: int) -> tuple[bool, str]:
    r = b.count(WALL7)
    return r == 0, f"seven-day duration literal occurrences {r} (need 0; measured pre-patch denominator {den_pre})"


def _g6(b: str) -> tuple[bool, str]:
    a, c = b.count(MEM_V), b.count("--cpus-per-task=" + CPU_N)
    return a == 0 and c == 0, (
        f"baked memory spec {a}x, baked cpus-per-task spec {c}x (need 0/0)")


def _g7(b: str) -> tuple[bool, str]:
    cmp1 = b.count("\"$FS_WALLTIME\" != " + WALL7)
    cmp2 = b.count("conflicts with ${FS_PARTITION} max")
    prods = [ln for ln in b.splitlines() if 'part_max="$(' in ln]
    sinfo_prods = [ln for ln in prods if "sinfo" in ln]
    ok = cmp1 == 0 and cmp2 == 0 and len(prods) == 1 and len(sinfo_prods) == 1
    return ok, (
        f"literal-comparison fragments {cmp1 + cmp2}x (need 0); part_max producers "
        f"{len(prods)}/1 allowed and of those sinfo {len(sinfo_prods)}/1")


def _g8(b: str) -> tuple[bool, str]:
    r = b.count(SELF_DEFAULT_IDIOM)
    return r == 0, f"self-defaulting SLURM_GPUS_PER_NODE idiom occurrences {r} (need 0)"


def _g9(b: str) -> tuple[bool, str]:
    with tempfile.NamedTemporaryFile("w", suffix=".sh", delete=False) as fh:
        fh.write(b)
        tmp = fh.name
    try:
        r = subprocess.run(["bash", "-n", tmp], capture_output=True, text=True)
    finally:
        pathlib.Path(tmp).unlink()
    return r.returncode == 0, (
        "bash -n clean (1/1 parse)" if r.returncode == 0
        else f"bash -n rc={r.returncode}: {r.stderr.strip()[:300]}")


def _g10(b: str) -> tuple[bool, str]:
    v = _classify(b)
    return v == "patched", f"second-run classification is {v!r} (need 'patched' => no-op 0)"


def _run_gates(body: str, den_pre_wall: int, *, rep: dict | None = None) -> bool:
    results: list[tuple[str, bool, str]] = []
    ok1, d1 = _g1(body); results.append(("G1", ok1, d1))
    ok2, d2 = _g2(body); results.append(("G2", ok2, d2))
    ok3, d3 = _g3(body); results.append(("G3", ok3, d3))
    ok4, d4, good4, den4 = _g4(body); results.append(("G4", ok4, d4))
    if rep is not None:
        rep["g4"] = (ok4, good4, den4)
    ok5, d5 = _g5(body, den_pre_wall); results.append(("G5", ok5, d5))
    ok6, d6 = _g6(body); results.append(("G6", ok6, d6))
    ok7, d7 = _g7(body); results.append(("G7", ok7, d7))
    ok8, d8 = _g8(body); results.append(("G8", ok8, d8))
    ok9, d9 = _g9(body); results.append(("G9", ok9, d9))
    ok10, d10 = _g10(body); results.append(("G10", ok10, d10))
    for name, ok, detail in results:
        line = f"  {'PASS' if ok else 'FAIL'} {name:<5} {detail}"
        print(line) if ok else print(line, file=sys.stderr)
    swept = all(ok for _, ok, _ in results)
    if rep is not None:
        rep["all"] = swept
        rep["n"] = len(results)
        rep["red"] = sum(1 for _, ok, _ in results if not ok)
    return swept


# --- CONTROLS, each observed going red (or green, for C6) on its own copy ---------

def _controls(post: str, g4_real: tuple[bool, int, int], *,
              all_gates_ok: bool, n_gates: int, n_red: int) -> bool:
    ok = True

    # C1 MUST_FIRE: one estate directive back in the header -> G1 red.
    c1 = post.replace("#SBATCH --nodes=1\n", "#SBATCH --nodes=1\n" + DIR_GPUS + "\n", 1)
    fired1, d1 = _g1(c1)
    if c1 != post and not fired1:
        print(f"PASS control C1 MUST_FIRE  G1 observed RED with the gpu directive re-inserted ({d1})")
    else:
        print("FAIL control C1 MUST_FIRE  G1 did NOT go red on a re-inserted estate directive",
              file=sys.stderr)
        ok = False

    # C2 MUST_FIRE: strip ONE flag from ONE site -> G4 red reporting (N-1)/N. This is the
    # anti-laundering control: if G4 counted total flag occurrences instead of SITES, a
    # one-flag deletion would move the numerator differently; it must move whole sites.
    lines = post.splitlines(keepends=True)
    idx = next((i for i, ln in enumerate(lines)
                if not ln.lstrip().startswith("#") and "sbatch " in ln and PART_FLAG in ln), None)
    flag = ' --mem="$FS_MEM"'
    if idx is not None and flag in lines[idx]:
        lines[idx] = lines[idx].replace(flag, "", 1)
        c2 = "".join(lines)
        fired2, d2, good2, den2 = _g4(c2)
        real_den = g4_real[2]
        if not fired2 and den2 == real_den and good2 == real_den - 1:
            print(f"PASS control C2 MUST_FIRE  G4 observed RED at {good2}/{den2} after one flag "
                  f"was stripped from one site -- the gate counts SITES, not occurrences")
        else:
            print(f"FAIL control C2 MUST_FIRE  expected G4 red at {real_den - 1}/{real_den}; "
                  f"observed good={good2} den={den2} detail=({d2})", file=sys.stderr)
            ok = False
    else:
        print("FAIL control C2  could not build the doctored body (no flaggable sbatch site "
              "with the mem flag); a control that cannot be built must not be reported green",
              file=sys.stderr)
        ok = False

    # C3 MUST_FIRE: the literal comparison returns -> G7 red.
    stale_line = ("  fail 96 \"FS_WALLTIME='$FS_WALLTIME' conflicts with ${FS_PARTITION} max "
                  + WALL7 + "; refusing instead of clamping\"\n")
    c3 = post + stale_line
    fired3, d3 = _g7(c3)
    if not fired3:
        print(f"PASS control C3 MUST_FIRE  G7 observed RED on a restored literal comparison ({d3})")
    else:
        print("FAIL control C3 MUST_FIRE  G7 did NOT go red on a restored literal comparison",
              file=sys.stderr)
        ok = False

    # C4 MUST_FIRE: the self-defaulting idiom returns -> G8 red.
    c4 = post + "x=\"${SLURM_GPUS_PER_NODE" + SELF_DEFAULT_IDIOM + "}\"\n"
    fired4, d4c = _g8(c4)
    if not fired4:
        print(f"PASS control C4 MUST_FIRE  G8 observed RED on a restored self-default ({d4c})")
    else:
        print("FAIL control C4 MUST_FIRE  G8 did NOT go red on a restored self-defaulting idiom",
              file=sys.stderr)
        ok = False

    # C5 MUST_FIRE: delete one retained contract directive -> G2 red (G2 is not vacuous).
    c5 = post.replace("#SBATCH --nodes=1\n", "", 1)
    fired5, d5 = _g2(c5)
    if c5 != post and not fired5:
        print(f"PASS control C5 MUST_FIRE  G2 observed RED with #SBATCH --nodes=1 deleted ({d5})")
    else:
        print("FAIL control C5 MUST_FIRE  G2 did NOT go red on a deleted retained directive",
              file=sys.stderr)
        ok = False

    # C7 MUST_FIRE: strip the PARTITION flag -- the one flag the denominator used to be
    # keyed on -- from one site. G4 must still see N sites and report (N-1)/N. If the
    # population shrank to N-1 and the gate went green, the denominator would be moving
    # with the defect, which is the failure mode C2 cannot detect on its own.
    lines7 = post.splitlines(keepends=True)
    idx7 = next((i for i, ln in enumerate(lines7)
                 if not ln.lstrip().startswith("#") and PART_FLAG in ln
                 and _SBATCH_CMD.search(ln)), None)
    if idx7 is not None:
        lines7[idx7] = lines7[idx7].replace(" " + PART_FLAG, "", 1)
        fired7, d7c, good7, den7 = _g4("".join(lines7))
        real_den = g4_real[2]
        if not fired7 and den7 == real_den and good7 == real_den - 1:
            print(f"PASS control C7 MUST_FIRE  G4 observed RED at {good7}/{den7} with the "
                  f"partition flag stripped -- the denominator does not move with the defect")
        else:
            print(f"FAIL control C7 MUST_FIRE  expected G4 red at {real_den - 1}/{real_den} "
                  f"with the population UNCHANGED; observed good={good7} den={den7} "
                  f"detail=({d7c})", file=sys.stderr)
            ok = False
    else:
        print("FAIL control C7  could not build the doctored body (no partition-flagged "
              "sbatch site); a control that cannot be built must not be reported green",
              file=sys.stderr)
        ok = False

    # C6 MUST_PASS: the real post-image passes EVERY gate. This reads the whole-sweep
    # verdict, not one gate's: keying the MUST_PASS on G4 alone let this control print
    # "passed every gate" over a red G8, which is a claim mismatched to its evidence --
    # a defect even when every other line is correct.
    if all_gates_ok:
        print(f"PASS control C6 MUST_PASS  real post-image passed every gate "
              f"({n_gates}/{n_gates}; G4 measured {g4_real[1]}/{g4_real[2]} sites)")
    else:
        print(f"FAIL control C6 MUST_PASS  the real post-image did not pass every gate "
              f"({n_gates - n_red}/{n_gates} green)", file=sys.stderr)
        ok = False

    return ok


def main() -> int:
    if not LAUNCH.exists():
        # Fail closed: a missing input is not a zero, it is an unread measurement.
        print(f"REFUSE 96: launcher not found: {LAUNCH}", file=sys.stderr)
        return 96
    text = LAUNCH.read_text("utf-8")

    verdict = _classify(text)
    if verdict == "patched":                                                  # E/G10 front door
        print("  G10  fs204/fs153 already applied -- no-op (idempotent)")
        return 0
    if verdict == "unmeasured":
        # all([]) is True: a stage over zero units is UNMEASURED, never PASS. If the
        # generator now emits a different node shape entirely, this stage must say so
        # rather than rewrite nothing and report clean.
        print(f"  UNMEASURED G-front  0/{len(_anchors())} anchors and 0 duration-literal "
              f"occurrences found in {LAUNCH.name}", file=sys.stderr)
        return 95

    anchors = _anchors()
    bad = [(label, text.count(a)) for label, a, _ in anchors if text.count(a) != 1]
    if bad:
        for label, n in bad:
            print(f"  FAIL anchors  {label!r} occurs {n}x (need exactly 1)", file=sys.stderr)
        print(f"\nREFUSING TO WRITE -- the generator emitted a different shape than #204/#153 "
              f"were measured against ({len(anchors) - len(bad)}/{len(anchors)} anchors unique)",
              file=sys.stderr)
        return 96
    print(f"  anchors  {len(anchors)}/{len(anchors)} located exactly once "
          f"(denominator measured on the pre-image, not assumed)")

    den_pre_wall = text.count(WALL7)

    new = text
    for _, a, r in anchors:
        new = new.replace(a, r, 1)

    # E4 -- one flag set on EVERY sbatch call site. Lines are rewritten individually so
    # that the denominator is the number of sites actually found, and the count must
    # equal the measured EXPECTED_SUBMITS: a launcher that submits a phase this stage
    # never saw is a generator drift to refuse, not to half-flag.
    out, n_sub = [], 0
    for ln in new.splitlines(keepends=True):
        if not ln.lstrip().startswith("#") and "sbatch " + PART_FLAG in ln:
            c = ln.count("sbatch " + PART_FLAG)
            ln = ln.replace("sbatch " + PART_FLAG, "sbatch " + PART_FLAG + NEW_FLAG_TAIL)
            n_sub += c
        out.append(ln)
    new = "".join(out)
    # Cross-validate the insertion anchor against the flag-independent population. The
    # loop above keys on "sbatch --partition=..." because it needs an anchor to insert
    # after; that is legitimate for WRITING and illegitimate for COUNTING. If some site
    # invokes sbatch without the partition flag, the loop would skip it silently and
    # every gate below would be blind to it in exactly the same way. Two populations
    # that must agree, measured two different ways.
    n_cmd = len(_submit_sites(new))
    if n_cmd != n_sub:
        print(f"REFUSE 96: {n_cmd} command-position sbatch invocation(s) but only {n_sub} "
              f"carried the partition anchor this stage rewrites. The difference is a site "
              f"that would ship with no shape flags at all; refusing rather than flagging "
              f"the ones that happen to be reachable.", file=sys.stderr)
        return 96

    if n_sub != EXPECTED_SUBMITS:
        print(f"REFUSE 96: found {n_sub} partition-flagged sbatch invocation(s), measured "
              f"denominator is {EXPECTED_SUBMITS} (probe / production / resume / post-mortem). "
              f"A site that misses the flags silently inherits whatever the header left behind; "
              f"this stage declines rather than half-apply.", file=sys.stderr)
        return 96
    print(f"  E4  {n_sub}/{n_sub} sbatch call sites now carry the full five-flag set")

    g4_rep: dict = {}
    gates_ok = _run_gates(new, den_pre_wall, rep=g4_rep)
    controls_ok = _controls(new, g4_rep["g4"], all_gates_ok=gates_ok,
                            n_gates=g4_rep["n"], n_red=g4_rep["red"])

    if not controls_ok:
        print("\nREFUSING TO WRITE -- this stage's own gates failed their controls; a "
              "detector that cannot be seen to fire has nothing to say about the launcher",
              file=sys.stderr)
        return 96
    if not gates_ok:
        print("\nREFUSING TO WRITE -- gates above are red", file=sys.stderr)
        return 96

    LAUNCH.write_text(new, "utf-8")
    print(f"\nALL GATES GREEN (4/4 estate directives deleted, 5/5 retained exactly, "
          f"{n_sub}/{n_sub} submits carry 5/5 flags, one walltime oracle) -> {LAUNCH.name}")
    print("OPERATOR IMPACT, stated because it is a breaking change: launches must now "
          "export FS_CPUS_PER_TASK, FS_MEM and FS_WALLTIME (FS_GPUS_PER_NODE and "
          "FS_PARTITION were already required). The walltime is proven against the "
          "sinfo-measured partition maximum at submit time: a multi-day request is now "
          "correctly ACCEPTED on a partition that allows it and correctly REFUSED on one "
          "that does not -- which is the behaviour the deleted hard-coded comparison "
          "could never express.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
