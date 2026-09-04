#!/usr/bin/env python3
"""Add a singularity arm to launchers/fs_container_backend.sh by SPLICE.

WHY NOT REGENERATE THE WHOLE FILE
---------------------------------
The first attempt asked Kimi for the complete 791-line file with the enroot
arm "preserved verbatim". Two things went wrong and they are the same thing:

  1. MEASURED: without the current file in the prompt it produced a 491-line
     rewrite retaining 16 of 50 enroot references — it could not preserve what
     it had never been shown.
  2. MEASURED: with the file in the prompt (28k tokens), generation ran past
     83 minutes and had to be killed. A ~900-line single generation is simply
     the wrong unit of work.

Both are solved by never regenerating the enroot arm at all. Kimi is asked for
ONLY the functions that must change plus the new ones, and this script splices
them in mechanically. Everything it is not shown cannot be damaged by it —
that is a structural guarantee, not a promise extracted in a prompt. It also
makes the diff reviewable: the untouched functions are provably untouched.

The splice is verified, not trusted: enroot reference count must not drop, the
result must pass `bash -n`, and every function that existed before must still
exist after. A splice that silently dropped a function would otherwise look
exactly like a successful one.
"""

from __future__ import annotations

import json
import os
import pathlib
import re
import subprocess
import sys

HOME = pathlib.Path.home()
FANOUT = HOME / ".claude/skills/kimi-fanout/kimi_fanout.py"
# Both of these were absolute paths under one developer's home directory until #162.
# That is the same defect #123 and #151 fixed for the ESTATE root, recurring one layer
# down for the BUILD host: the build's own scans are tuned for estate identifiers and
# say nothing about `/Users/<someone>`, so this file passed every in-build check and was
# caught only by the pre-push gate, which has a wider vocabulary.
#
# Resolution order matches apply_splice.py deliberately -- same env var, same ancestor
# walk -- because "where is the upstream repo" must have ONE answer in this build.
_HERE = pathlib.Path(__file__).resolve().parent
_BACKEND_REL = pathlib.Path("launchers/fs_container_backend.sh")


def _resolve_src() -> pathlib.Path:
    """The upstream backend this module splices, or a refusal naming what was tried.

    Lazy, not module-level: `functions()` is imported by three other stages, and a
    missing upstream must not make importing a brace matcher fail.
    """
    tried: list[str] = []
    for label, cand in (
        ("$FS_UPSTREAM_REPO", os.environ.get("FS_UPSTREAM_REPO")),
        *(("ancestor of this file", str(a)) for a in [_HERE, *_HERE.parents]),
        ("legacy sibling", str(_HERE.parent / "fs-repo")),
    ):
        if not cand:
            continue
        path = pathlib.Path(cand).resolve() / _BACKEND_REL
        tried.append(f"{path}  ({label})")
        if path.is_file():
            return path
    seen, uniq = set(), []
    for t in tried:
        if t not in seen:
            seen.add(t)
            uniq.append(t)
    print(
        f"h100_backend_splice: upstream backend {_BACKEND_REL} not found.\n"
        + "".join(f"  tried: {t}\n" for t in uniq)
        + "  Point at the framework repo with $FS_UPSTREAM_REPO.",
        file=sys.stderr,
    )
    raise SystemExit(96)


OUT = _HERE / "h100"
BASE_URL = "http://localhost:18001"

# Only these change. Anything not listed is out of Kimi's reach by construction.
TARGETS = ["fs_backend_init", "fs_backend_runtime_setup",
           "fs_env_forward_denylisted", "run_in_container"]

FUNC_RE = re.compile(r"^([a-z_][a-z0-9_]*)\(\)\s*\{", re.M)


def functions(text: str) -> dict[str, tuple[int, int]]:
    """Map name -> (start_offset, end_offset) by brace matching.

    Brace-counting, not a regex for the closing brace: these bodies contain
    nested blocks and here-docs, and `^}` would end the function at the first
    dedented brace inside one.
    """
    out: dict[str, tuple[int, int]] = {}
    for m in FUNC_RE.finditer(text):
        i = text.index("{", m.start())
        depth, j, n = 0, i, len(text)
        while j < n:
            ch = text[j]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    break
            j += 1
        out[m.group(1)] = (m.start(), j + 1)
    return out


SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["replacements", "additions", "rationale", "gaps"],
    "properties": {
        "replacements": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["function", "body", "why"],
                "properties": {
                    "function": {"type": "string",
                                 "description": "exact existing function name"},
                    "body": {"type": "string",
                             "description": "COMPLETE replacement, from `name() {` to the "
                                            "matching closing brace inclusive"},
                    "why": {"type": "string"},
                },
            },
        },
        "additions": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["name", "body", "after", "why"],
                "properties": {
                    "name": {"type": "string"},
                    "body": {"type": "string"},
                    "after": {"type": "string",
                              "description": "existing function to insert after"},
                    "why": {"type": "string"},
                },
            },
        },
        "rationale": {"type": "string"},
        "gaps": {"type": "string",
                 "description": "what you could not verify. Empty string only if "
                                "genuinely nothing — a blank gaps field on a task "
                                "this size is itself a finding."},
    },
}

PROMPT = """You are extending a production bash library. Return ONLY changed and
new functions — never the whole file. The functions you are not shown are not
yours to touch, and that is deliberate: it is what guarantees the working
enroot arm survives this edit.

REFUTE FIRST, PATCH SECOND. If a requirement below is wrong, say so in
`gaps` and do not implement it. Do not make a correctly-red thing pass.

THE VACUOUS-TRUTH DOCTRINE — this code is a verification plane, so it is held
to it:
 1. `all([])` is True. Zero units checked is UNMEASURED, never PASS.
 2. Every claim carries a DENOMINATOR. "clean" without "of N" states nothing.
 3. Every detector ships CONTROLS: a MUST_FIRE observed going red, and a
    MUST_PASS. A detector never observed firing is not a control, and one
    that never runs is not either.
 4. Fail CLOSED. Unreadable is not empty; missing is not zero.
 5. A claim broader than its evidence is a defect even when the code is
    correct — and it is symmetric: a false alarm costs what a false green does.

MEASURED GROUND TRUTH FOR THIS EDIT
-----------------------------------
* Estate: singularity-ce 4.1.2 ONLY. No enroot, no docker, no podman binary.
  `sbatch` and `srun` both exist — but pyxis does NOT, so `srun
  --container-image` fails. srun existing is not pyxis existing.
* THE BLOCKER, and the reason this edit exists: default `singularity exec`
  imports torch 2.9.0+cu128 from the HOST user-site
  (`/home/<uid>/.local/lib/python3.12/site-packages`, 9.4 GB). With
  `SINGULARITYENV_PYTHONNOUSERSITE=1` the SAME image imports the container's
  torch 2.11.0a0+...nv26.02 from `/usr/local/lib/python3.12/dist-packages`.
  Two torch majors from one image depending on whose $HOME is mounted.
* ROOT CAUSE — a direction flip, not a missing flag: enroot forwards NOTHING
  unless told; singularity forwards the HOST ENVIRONMENT unless told not to.
  The existing denylist is correct for enroot and structurally insufficient
  here. It cannot be repaired by adding entries: a denylist must enumerate
  every hostile variable, and the one that bit us is not the last.
* Singularity ALSO binds $HOME, /tmp and $PWD implicitly, with no entry in any
  mounts array. Any host-root census derived from an explicit mounts list is
  incomplete in exactly the arm where it matters — the host user-site lives
  under the $HOME nobody asked for.
* This does NOT change interpreter RESOLUTION: same `python` binary, different
  `sys.path`. A check on which binary runs would not have fired on the real
  incident and must not be credited with covering it.

BINDING REQUIREMENTS
--------------------
R1. `FS_CONTAINER_RUNTIME` selects the arm: `enroot` | `singularity`.
    REQUIRED, NO DEFAULT, refuse if unset. Precedent in this codebase is
    FS_ALLOWED_NODE, whose comment reads "an unconfigured node guard is a
    disabled standing rule". Auto-detection is barred: it makes the runtime an
    accident of $PATH.
R2. Split the two axes the current code conflates. ALLOCATION (slurm | local)
    and RUNTIME (enroot | singularity) are independent. Any existing rule of
    the form "SLURM_JOB_ID is set therefore use the slurm/pyxis arm" is a bug
    on this estate and must be corrected.
R3. Replace the env DENYLIST with an ALLOWLIST enforced IDENTICALLY in both
    arms, so the forwarding direction becomes a property of FoundationScale
    rather than of whichever runtime is loaded. Existing denylist entries stay
    as a second, subordinate check — belt and braces — but the allowlist is
    what decides. Preserve every currently-forwarded variable the enroot arm
    depends on; dropping one silently is a worse failure than the leak.
R4. The singularity arm sets PYTHONNOUSERSITE=1 (via SINGULARITYENV_ AND
    inside the exec'd command) AND contains the implicit mounts (`--no-home`
    or an explicit non-host `--home`, plus explicit `--pwd`). Belt and braces
    again: the env var alone is a single point of failure, and this file's own
    doctrine is that a single uncontrolled export is not a guarantee.
R5. Add a torch-provenance ASSERTION, distinct from any interpreter-resolution
    check: capture the resolved `torch.__file__` in-container and refuse to
    proceed unless it is under the container prefix. Leaked prefix to reject:
    `/home/*/.local/lib/python3.*/site-packages`. Expected prefix:
    `/usr/local/lib/python3.*/dist-packages`. Fail CLOSED if torch cannot be
    imported or the path cannot be read — unreadable is not "fine".
R6. Give R5 a MUST_FIRE drill callable on demand (e.g.
    `fs_selftest_torch_provenance`) that recreates the leak — clears
    PYTHONNOUSERSITE with a host user-site reachable — and asserts the check
    goes RED, plus a MUST_PASS leg asserting it goes GREEN when contained.
    Report both as "k of N", never as a bare "ok".
R7. Do NOT invent a walltime, partition, node name, or image path. Those come
    from the caller. This partition's max is 7 days and the GB200's rule of 10
    is rejected here, but neither number belongs in this file.

STYLE: match the surrounding file exactly — `fs_die` for fatals, the same
quoting discipline, the same comment voice (comments explain WHY and cite the
measurement, they do not restate the code). Assume `set -euo pipefail`.

=== BEGIN THE FUNCTIONS YOU MAY CHANGE (verbatim from the live file) ===
{FUNCS}
=== END ===

=== FOR CONTEXT ONLY — names of every function in the file, so you do not
=== reinvent one that exists or call one that does not:
{NAMES}
=== END ===
"""


def main() -> int:
    src = _resolve_src()
    text = src.read_text("utf-8")
    fmap = functions(text)
    missing = [t for t in TARGETS if t not in fmap]
    if missing:
        print(f"REFUSING: target function(s) not found in {src}: {missing}", file=sys.stderr)
        return 3
    funcs = "\n\n".join(f"# ---- {t} ----\n{text[fmap[t][0]:fmap[t][1]]}" for t in TARGETS)
    prompt = PROMPT.replace("{FUNCS}", funcs).replace("{NAMES}", ", ".join(sorted(fmap)))

    OUT.mkdir(parents=True, exist_ok=True)
    sp = OUT / "schema_splice.json"
    sp.write_text(json.dumps(SCHEMA), "utf-8")
    dst = OUT / "backend_splice.json"
    print(f"prompt ~{len(prompt) // 4} tokens over {len(TARGETS)} functions")
    if "--dry-run" in sys.argv:
        return 0
    if not os.environ.get("KIMI_K3_API_KEY"):
        print("REFUSING: KIMI_K3_API_KEY unset", file=sys.stderr)
        return 3
    cmd = [sys.executable, str(FANOUT), "--base-url", BASE_URL, "--effort", "high",
           "--max-tokens", "30000", "--workers", "1", "--timeout", "3600",
           "--json-schema-file", str(sp), "--out", str(dst)]
    r = subprocess.run(cmd, input=json.dumps([{"prompt": prompt}]), text=True,
                       capture_output=True)
    sys.stderr.write(r.stderr[-1500:])
    print(f"rc={r.returncode} -> {dst}")
    return r.returncode


if __name__ == "__main__":
    raise SystemExit(main())
