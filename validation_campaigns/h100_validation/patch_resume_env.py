#!/usr/bin/env python3
"""#122: make the resume contract cross BOTH container runtimes, not just one.

THE DEFECT, measured before writing a line of fix:

  launch_fs_h100.fixed.sh:291-292   export SINGULARITYENV_FS_RESUME_CKPT=...
                                    export SINGULARITYENV_FS_RESUME_STEP=...
  fs_container_backend.bound.sh     grep -c RESUME  ->  0
  FS_ENV_ALLOWLIST                  FS_RESUME_CKPT 0 hits, FS_RESUME_STEP 0 hits
                                    (FS_ITERATION_BUDGET 1, FS_EARLY_SAVE_STEPS 1, OUT_DIR 1)

SINGULARITYENV_X is singularity's own private mechanism for injecting X into a
container. Under the enroot arm it is an ordinary host variable with a strange
name: not on the allowlist, therefore not forwarded, therefore FS_RESUME_CKPT
and FS_RESUME_STEP simply do not exist in-container. The trainer then finds no
checkpoint to restore, starts from step 0, trains its bounded budget, exits 0 --
and R5 records PASS for a resume that never resumed. That is the vacuous-truth
failure in its most expensive form: the gate is green BECAUSE the work was
skipped.

This is the THIRD instance of one pattern (#109, #117, now #122): a capability
wired to one runtime's private mechanism works there and vanishes silently on
the other. #117 was mounts, #109 was the allocation/runtime conflation, this is
environment. The general fix each time is the same shape -- declare the need in
FoundationScale's own vocabulary and let each arm materialise it -- so the fix
here is NOT "also export ENROOT_something". It is: the launcher exports the
PLAIN names, and the backend's single shared forwarding path (one forward_env
array, filtered by fs_env_forward_allowlisted, emitted as --env by enroot and as
an `env K=V` prefix by singularity) carries them on both arms at once.

The SINGULARITYENV_ exports are DELETED rather than kept as belt-and-braces. A
surviving runtime-specific duplicate would keep the singularity arm working for
a reason unrelated to the allowlist, which is precisely how this defect stayed
invisible: it would re-arm the trap for the next reader while looking harmless.

WHY A SCRIPT, NOT AN EDIT. Both files are generated -- bound.sh by apply_117.py,
the launcher by apply_113.py. A hand edit survives until the next regeneration
and then vanishes without a word: the fix is in the file you read and absent
from the file that runs. Pipeline order is therefore:

    python3 apply_113.py && python3 patch_bindpop.py && \
    python3 apply_117.py && python3 patch_resume_env.py

GATES. An unverified patch script is an edit with extra steps.
  Q1 idempotent (re-run is a no-op, never a double-insert)
  Q2 backend anchor unique
  Q3 launcher export anchor unique
  Q4 the stale "singularity is the only permitted runtime" comment anchor unique
     -- a comment that misstates the architecture is how the next reader
     reproduces the bug, so it is part of the defect, not cosmetics
  Q5 bash -n clean on BOTH patched files
  Q6 cross-check with denominators: 0 SINGULARITYENV_FS_RESUME_* survive, both
     plain names exported, both on the allowlist
  Q7 the allowlist is EXECUTED, not read: the patched array and the real
     fs_env_forward_allowlisted body are lifted out of bound.sh and run.
       MUST_PASS  FS_RESUME_CKPT, FS_RESUME_STEP forwarded
       MUST_FIRE  LD_PRELOAD refused (the list gates at all)
       MUST_FIRE  FS_RESUME_CKPT_EXTRA refused (exact match, not prefix -- a
                  prefix match would forward anything beginning with a legal
                  name and the MUST_PASS row alone could never detect it)
  Q8 the extraction has a denominator: names parsed out of the array must equal
     names counted in the file. A lifted-out control that silently extracted an
     empty array would pass Q7's MUST_FIRE rows for the wrong reason.
"""

from __future__ import annotations

import pathlib
import re
import subprocess
import sys
import tempfile

GEN = pathlib.Path(__file__).resolve().parent / "h100" / "gen"
BACKEND = GEN / "fs_container_backend.bound.sh"
LAUNCH = GEN / "launch_fs_h100.fixed.sh"

MARK = "fs122:"

# --- anchors ---------------------------------------------------------------

BACKEND_ANCHOR = (
    "    FS_EARLY_SAVE_STEPS            # must cross: same reader, same reason\n"
)

BACKEND_BLOCK = (
    "    FS_EARLY_SAVE_STEPS            # must cross: same reader, same reason\n"
    "    FS_RESUME_CKPT                 # fs122: must cross: the resume leg's checkpoint\n"
    "                                   #   path. Was exported as SINGULARITYENV_FS_RESUME_CKPT,\n"
    "                                   #   which is singularity's private injection mechanism and\n"
    "                                   #   a plain host variable to enroot -- so under enroot the\n"
    "                                   #   trainer saw no checkpoint, restarted from step 0, and\n"
    "                                   #   R5 recorded PASS for a resume that never resumed.\n"
    "    FS_RESUME_STEP                 # fs122: must cross: the RECORDED step the restored step\n"
    "                                   #   is compared against. Without it in-container there is\n"
    "                                   #   nothing to compare to and the proof degrades to 'it\n"
    "                                   #   did not crash', which is not evidence of resuming.\n"
)

LAUNCH_EXPORT_ANCHOR = (
    '  export SINGULARITYENV_FS_RESUME_CKPT="$resume_ckpt"\n'
    '  export SINGULARITYENV_FS_RESUME_STEP="$resume_step"\n'
)

LAUNCH_EXPORT_BLOCK = (
    "  # fs122: PLAIN names, not SINGULARITYENV_*. Both are on FS_ENV_ALLOWLIST, so the\n"
    "  # backend's single forwarding path carries them on BOTH arms (--env under enroot,\n"
    "  # an `env K=V` prefix under singularity) from one array. The SINGULARITYENV_ form\n"
    "  # is deleted rather than kept alongside: a runtime-specific duplicate would keep\n"
    "  # one arm working for a reason unrelated to the allowlist, which is exactly how\n"
    "  # this defect stayed invisible.\n"
    '  export FS_RESUME_CKPT="$resume_ckpt"\n'
    '  export FS_RESUME_STEP="$resume_step"\n'
)

LAUNCH_COMMENT_ANCHOR = (
    "  # container boundary via SINGULARITYENV_* (singularity is the only permitted runtime) and via\n"
    "  # FS_ITERATION_BUDGET / FS_EARLY_SAVE_STEPS (both on FS_ENV_ALLOWLIST). Distinct UNMEASURED\n"
)

LAUNCH_COMMENT_BLOCK = (
    "  # container boundary the same way every other in-container fact does: exported under its\n"
    "  # PLAIN name and carried by FS_ENV_ALLOWLIST (fs122). There is no runtime-specific path here\n"
    "  # -- FS_CONTAINER_RUNTIME is a required, never-inferred axis and singularity is what THIS\n"
    "  # estate happens to have, not what the framework is allowed to assume. FS_RESUME_CKPT,\n"
    "  # FS_RESUME_STEP, FS_ITERATION_BUDGET and FS_EARLY_SAVE_STEPS are all on the allowlist.\n"
    "  # Distinct UNMEASURED\n"
)

# --- Q7 harness -------------------------------------------------------------

CONTROL_TMPL = r"""
set -uo pipefail
%s
%s
probe() {  # name expected(0=forward 1=refuse)
  if fs_env_forward_allowlisted "$1"; then got=0; else got=1; fi
  if [[ "$got" == "$2" ]]; then printf 'ok %%s\n' "$1"
  else printf 'BAD %%s got=%%s want=%%s\n' "$1" "$got" "$2"; fi
}
printf 'names=%%s\n' "${#FS_ENV_ALLOWLIST[@]}"
probe FS_RESUME_CKPT 0
probe FS_RESUME_STEP 0
probe LD_PRELOAD 1
probe FS_RESUME_CKPT_EXTRA 1
probe FS_RESUME 1
"""


def _extract_block(src: str, start_re: str, end_line: str, what: str) -> str:
    m = re.search(start_re, src, re.M)
    if not m:
        # #161: RED is 5; a bare SystemExit(<string>) exits 1 and loses the contract.
        print(f"  FAIL Q7  could not locate {what} in {BACKEND.name}; "
              f"a control that cannot be built must not be reported green", file=sys.stderr)
        raise SystemExit(5)
    lines = src[m.start():].splitlines(keepends=True)
    out = [lines[0]]
    for ln in lines[1:]:
        out.append(ln)
        if ln.rstrip("\n") == end_line:
            return "".join(out)
    print(f"  FAIL Q7  unterminated {what}", file=sys.stderr)   # #161: RED is 5, not 1
    raise SystemExit(5)


def _bash(script: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["bash", "-c", script], capture_output=True, text=True)


def _syntax_ok(text: str, label: str) -> bool:
    with tempfile.NamedTemporaryFile("w", suffix=".sh", delete=False) as fh:
        fh.write(text)
        tmp = fh.name
    r = subprocess.run(["bash", "-n", tmp], capture_output=True, text=True)
    if r.returncode != 0:
        print(f"  FAIL Q5  bash -n {label}: {r.stderr.strip()[:300]}", file=sys.stderr)
        return False
    return True


def main() -> int:
    back = BACKEND.read_text("utf-8")
    lau = LAUNCH.read_text("utf-8")

    if MARK in back and MARK in lau:                                          # Q1
        print("  Q1  already applied to both files — no-op (idempotent)")
        return 0
    if (MARK in back) != (MARK in lau):
        print(f"  FAIL Q1  half-applied: backend={MARK in back} launcher={MARK in lau}. "
              f"One file was regenerated without the other; re-run the full pipeline.",
              file=sys.stderr)
        return 5
    print("  PASS Q1  neither file patched yet")

    ok = True
    for name, text, anchor, gate in (
        ("backend allowlist", back, BACKEND_ANCHOR, "Q2"),
        ("launcher export", lau, LAUNCH_EXPORT_ANCHOR, "Q3"),
        ("launcher comment", lau, LAUNCH_COMMENT_ANCHOR, "Q4"),
    ):
        n = text.count(anchor)
        if n != 1:
            print(f"  FAIL {gate}  {name} anchor occurs {n}x (need 1); "
                  f"{'the generator changed shape' if n == 0 else 'ambiguous site'}",
                  file=sys.stderr)
            ok = False
        else:
            print(f"  PASS {gate}  {name} anchor unique")
    if not ok:
        print("\nREFUSING TO WRITE — anchors are not what this patch was written against",
              file=sys.stderr)
        return 5

    back_new = back.replace(BACKEND_ANCHOR, BACKEND_BLOCK, 1)
    lau_new = lau.replace(LAUNCH_EXPORT_ANCHOR, LAUNCH_EXPORT_BLOCK, 1)
    lau_new = lau_new.replace(LAUNCH_COMMENT_ANCHOR, LAUNCH_COMMENT_BLOCK, 1)

    if not _syntax_ok(back_new, BACKEND.name):                                # Q5
        ok = False
    if not _syntax_ok(lau_new, LAUNCH.name):
        ok = False
    if ok:
        print("  PASS Q5  bash -n clean on both patched files")

    # --- Q6 cross-check, with denominators ----------------------------------
    leftover = len(re.findall(r"SINGULARITYENV_FS_RESUME_\w+", lau_new))
    plain = len(re.findall(r'^\s*export FS_RESUME_(CKPT|STEP)="', lau_new, re.M))
    listed = sum(1 for n in ("FS_RESUME_CKPT", "FS_RESUME_STEP")
                 if re.search(rf"^\s+{n}\s", back_new, re.M))
    if leftover or plain != 2 or listed != 2:
        print(f"  FAIL Q6  SINGULARITYENV_FS_RESUME_* surviving={leftover} (want 0); "
              f"plain exports={plain}/2; allowlist entries={listed}/2", file=sys.stderr)
        ok = False
    else:
        print("  PASS Q6  0 SINGULARITYENV_FS_RESUME_* survive; 2/2 plain exports; "
              "2/2 allowlist entries")

    # --- Q7/Q8 execute the patched allowlist --------------------------------
    arr = _extract_block(back_new, r"^  FS_ENV_ALLOWLIST=\($", "  )", "FS_ENV_ALLOWLIST")
    fn = _extract_block(back_new, r"^fs_env_forward_allowlisted\(\) \{$", "}",
                        "fs_env_forward_allowlisted")
    # Q8: the lifted array must contain the same names the file does. An empty
    # extraction would satisfy every MUST_FIRE row for entirely the wrong reason.
    lifted = [ln.split("#")[0].strip() for ln in arr.splitlines()[1:-1]]
    lifted = [n for n in lifted if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", n)]
    r = _bash(CONTROL_TMPL % (arr.replace("  FS_ENV_ALLOWLIST", "FS_ENV_ALLOWLIST", 1), fn))
    ran = re.search(r"names=(\d+)", r.stdout)
    if not ran:
        print(f"  FAIL Q8  harness produced no name count: {r.stderr.strip()[:200]}",
              file=sys.stderr)
        ok = False
    elif int(ran.group(1)) != len(lifted) or len(lifted) < 10:
        print(f"  FAIL Q8  extraction denominator mismatch: bash saw {ran.group(1)}, "
              f"parser saw {len(lifted)}", file=sys.stderr)
        ok = False
    else:
        print(f"  PASS Q8  allowlist lifted intact: {len(lifted)} names, bash agrees")

    bad = [ln for ln in r.stdout.splitlines() if ln.startswith("BAD")]
    okrows = [ln for ln in r.stdout.splitlines() if ln.startswith("ok ")]
    if bad or len(okrows) != 5:
        print(f"  FAIL Q7  {len(bad)} control row(s) wrong, {len(okrows)}/5 correct: "
              f"{bad[:4]}", file=sys.stderr)
        ok = False
    else:
        print("  PASS Q7  MUST_PASS FS_RESUME_CKPT/STEP forwarded; "
              "MUST_FIRE LD_PRELOAD, FS_RESUME_CKPT_EXTRA, FS_RESUME all REFUSED "
              "(the list gates, and it matches exactly — not by prefix)")

    if not ok:
        print("\nREFUSING TO WRITE — gates above are red", file=sys.stderr)
        return 5
    BACKEND.write_text(back_new, "utf-8")
    LAUNCH.write_text(lau_new, "utf-8")
    print(f"\nALL GATES GREEN -> {BACKEND.name}, {LAUNCH.name}")
    print("SCOPE, stated so it is not over-read: this proves the two names are "
          "FORWARDED by the shared path on both arms. Whether tools/fs_train.py "
          "READS them and asserts restored==recorded is C5's job in apply_phase3.py, "
          "and is UNMEASURED until that gate runs.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
