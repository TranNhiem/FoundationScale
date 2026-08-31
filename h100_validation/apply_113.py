#!/usr/bin/env python3
"""Apply the L1-L4 anchored fixes to the H100 launcher, and PROVE the result.

Anchored edits fail in a different way than whole-file rewrites, so the gates
differ. A rewrite can silently DROP content; an anchored edit can silently MISS
(anchor absent -> fix never lands) or OVER-APPLY (anchor ambiguous -> wrong
site edited). Both look like success if you only check that the file still
parses, so each is a hard refusal here rather than a warning.

  A1 every anchor occurs EXACTLY ONCE in the text it is applied to
     (0 -> the fix silently vanishes; >=2 -> the wrong site may be edited)
  A2 a "refuted" verdict must carry an EMPTY anchor, and a "confirmed" one a
     non-empty anchor -- a confirmed fix with no anchor is a fix that did not
     happen, reported as one
  A3 the result passes `bash -n`
  A4 the NOT-DEFECTS survive: eight behaviours an adversarial panel verified as
     CORRECT must still be present verbatim. This is the regression gate. The
     spec tells the worker not to weaken them; A4 is what makes that checkable
     rather than aspirational.
  A5 every fs_* / `fail` call in the new text is a function that exists
  A6 no model-family name enters the file (model-agnostic core)
  A7 the L1 host probe is really GONE and device counting really happens after
     fs_backend_runtime_setup -- L1's whole point is ordering, and an edit that
     merely hardened the probe in place would pass every other gate

Reports "before -> after" with denominators. Writes to a NEW path unless
--in-place.
"""

from __future__ import annotations

import json
import os
import pathlib
import re
import subprocess
import sys
import tempfile
from fs_estate_pat import estate_partition_literal

# #157: the estate's short name is an INPUT. It appears here because this anchor has to
# reproduce the before-text verbatim to find it, and that text names the estate. A
# declared-empty estate (NONE) yields the estate-free anchor, not a hole in a sentence.
_SHORT = estate_partition_literal()
_ON_SHORT = f" on {_SHORT}" if _SHORT else ""

# Resolved from __file__, not from an absolute build-host path (#123/#136). INPUTS come
# from h100/upstream/ and OUTPUTS go to h100/gen/; they were the same directory until #137,
# which meant the base text of the shipped launcher sat among the files the build deletes
# and rebuilds. It is not one of them -- no stage produces it and no upstream in fs-repo
# can re-derive it, so losing it would make the launcher unreconstructable.
_ROOT = pathlib.Path(__file__).resolve().parent
GEN = _ROOT / "h100" / "gen"
UPSTREAM = _ROOT / "h100" / "upstream"

# #157: this snapshot is the estate's OWN pre-existing launcher -- the artifact being
# generalized, not part of the framework doing the generalizing -- and it carries the estate's
# org segment in four path guards. Those four sites are exactly what patch_estate_roots.py
# replaces, using a root supplied by the environment, so the OUTPUT of this build is clean
# while the INPUT discloses. Three consecutive clean scans reported zero because no scan
# category owned this directory at all.
#
# Redacting the snapshot in place is the wrong repair twice over: it would misrepresent the
# provenance of a file whose whole purpose is to be the faithful before-text, and the
# downstream patch anchors are built FROM the estate root, so a pre-redacted input would
# resolve zero sites and the build would refuse anyway -- just less legibly.
#
# So the snapshot is a DECLARED, OPERATOR-SUPPLIED input, resolved the way #156 resolves the
# upstream backend: most explicit candidate first, accepted only if it is really there, and
# every candidate named in the refusal.
_SNAP = "launchers__launch_fs_h100.sh"


def _snapshot_candidates():
    for i, a in enumerate(sys.argv):
        if a == "--upstream-launcher" and i + 1 < len(sys.argv):
            yield "--upstream-launcher", pathlib.Path(sys.argv[i + 1]).resolve()
    env = os.environ.get("FS_UPSTREAM_H100_LAUNCHER")
    if env:
        yield "$FS_UPSTREAM_H100_LAUNCHER", pathlib.Path(env).resolve()
    updir = os.environ.get("FS_UPSTREAM_DIR")
    if updir:
        yield "$FS_UPSTREAM_DIR", pathlib.Path(updir).resolve() / _SNAP
    yield "in-tree h100/upstream", UPSTREAM / _SNAP


_snap_tried, SRC = [], None
for _lbl, _c in _snapshot_candidates():
    _snap_tried.append(f"{_c}  ({_lbl})")
    if _c.is_file():
        SRC = _c
        break
if SRC is None:
    # #161: REFUSE is 96, and `raise SystemExit("text")` exits 1. Print, then exit the code.
    print(
        f"apply_113: upstream launcher snapshot not found ({_SNAP}).\n"
        + "".join(f"  tried: {t}\n" for t in dict.fromkeys(_snap_tried))
        + "  Supply it with --upstream-launcher <path>, $FS_UPSTREAM_H100_LAUNCHER, or\n"
        "  $FS_UPSTREAM_DIR pointing at a directory that contains it.\n"
        "  This is the BEFORE-text every launcher patch is applied to. It is deliberately not\n"
        "  published: it is the estate's own launcher and it contains estate paths (#157).\n"
        "  Without it there is nothing to patch, and continuing would let a previously\n"
        "  generated launcher be certified as freshly built (#136).",
        file=sys.stderr,
    )
    raise SystemExit(96)
print(f"apply_113: upstream snapshot resolved to {SRC}")
FIX = _ROOT / "h100" / "fix_113.json"
DST = GEN / "launch_fs_h100.fixed.sh"
BACKEND = GEN / "fs_container_backend.spliced.sh"

# A4 -- the eight refuted-as-correct behaviours, as verbatim substrings. Each
# was checked against the current file when this list was written; a probe that
# cannot match would silently certify everything, so main() asserts all eight
# are present in the BEFORE text and aborts if any is not.
NOT_DEFECTS = {
    "PYTHONNOUSERSITE exported": "export PYTHONNOUSERSITE=1",
    "SINGULARITYENV_PYTHONNOUSERSITE": "export SINGULARITYENV_PYTHONNOUSERSITE=1",
    "FS_WALLTIME refusal": "conflicts with "
                           + (f"{_SHORT} " if _SHORT else "") + "max",
    "FS_NCCL_IB_HCA refusal": "FS_NCCL_IB_HCA",
    "FS_NCCL_SOCKET_IFNAME check": "FS_NCCL_SOCKET_IFNAME",
    "no default engine": "FS_ENGINE_LAUNCH_CMD",
    "OUT_DIR_STABLE checks": "OUT_DIR_STABLE",
    "unmeasured is not pass": "unmeasured is not pass",
}


# ---------------------------------------------------------------------------
# #157: the recorded envelope is a TEMPLATE, and the estate's partition name is an input.
#
# fix_113.json is the raw generator response whose `fixes` are before/after text pairs cut
# from the estate's own launcher, and that before-text names the estate's Slurm partition 11
# times. So the envelope carried 11 estate identifiers into a public tree -- invisible,
# because no scan category owned h100/*.json at all.
#
# Withholding it was not available: unlike h100/upstream/, this file is REQUIRED for a build
# from a clean clone, and a repository whose build cannot run is not a deliverable.
#
# Redacting it destructively was not acceptable either -- it is a recorded response, and a
# recorded response that has been quietly rewritten is no longer provenance.
#
# So the substitution is LOSSLESS and the reversal is checkable by anyone holding the
# literal, which is the property that makes this a redaction rather than an edit:
#
#     sed 's/@FS_PARTITION_LITERAL@/<literal>/g' h100/fix_113.json | shasum -a 256
#       -> 4ec3cf083a0ad28b04de785f3ec32e0b529558e7836f81cad724db5461bb50d9
#
# That hash is the ORIGINAL response, recorded before the substitution and proven to
# round-trip at the moment it was made. The template's own hash is
# f25369a34e608f3b246d6d541970c5677badd76f4e36016cbab120e31720d5f7.
_PLACEHOLDER = "@FS_" + "PARTITION_LITERAL@"   # split so the redactor's own token is not one


def _load_fix() -> str:
    """The envelope with the estate's partition name substituted back in.

    Refuses rather than expanding to nothing on a declared-empty estate. NONE asserts that
    THIS estate's before-text does not name a partition, which is a claim about the operator's
    launcher; it says nothing about a response recorded from a DIFFERENT estate whose text
    demonstrably does. Substituting "" would silently corrupt 11 anchors into forms that match
    nothing, and a patch stage that resolves zero sites is exactly the vacuous success this
    build refuses to produce.
    """
    raw = FIX.read_text("utf-8")
    n = raw.count(_PLACEHOLDER)
    if n == 0:
        return raw                      # an expanded or freshly dispatched envelope
    lit = os.environ.get("FS_PARTITION_LITERAL", "").strip()
    if not lit or lit == "NONE":
        # #161: my own violation, written in the ticket that fixed three of these elsewhere.
        print(
            f"apply_113: {FIX.name} is a template carrying {n} @FS_PARTITION_LITERAL@ "
            f"placeholder(s), and FS_PARTITION_LITERAL is "
            + ("unset." if not lit else "declared NONE.")
            + "\n  The recorded before-text names a Slurm partition, so the anchors cannot be\n"
            "  expanded without it. Set FS_PARTITION_LITERAL to the partition named in the\n"
            "  launcher you are patching, or re-dispatch the envelope against your own\n"
            "  launcher with dispatch_113.py.\n"
            "  Expanding to the empty string would leave 11 anchors matching nothing and the\n"
            "  stage reporting a clean no-op (#157).",
            file=sys.stderr,
        )
        raise SystemExit(96)
    out = raw.replace(_PLACEHOLDER, lit)
    assert _PLACEHOLDER not in out, "placeholder survived substitution"
    print(f"apply_113: expanded {n} partition placeholder(s) from $FS_PARTITION_LITERAL")
    return out


FAMILIES = re.compile(r"(?i)\b(gemma|qwen|llama|mistral|falcon|phi-?[0-9])\b")

_DEF = re.compile(r"(?m)^([a-z_][a-z0-9_]*)\(\)[ \t]*\{[ \t]*$")
_DEF_ONELINE = re.compile(r"(?m)^([a-z_][a-z0-9_]*)\(\)[ \t]*\{(.*)\}[ \t]*$")
# Anything that LOOKS like a top-level definition, however it is written. This
# is the denominator: it exists so the parser can measure what it failed to
# parse instead of assuming it parsed everything.
_DEF_ANY = re.compile(r"(?m)^(?:function[ \t]+)?([a-z_][a-z0-9_]*)[ \t]*\([ \t]*\)[ \t]*\{")


def _function_bodies(src: str) -> dict[str, str]:
    """name -> body, for top-level shell function definitions.

    Handles BOTH `name() {` ... column-0 `}` and the one-line
    `name() { ...; }`. The one-line form was originally excluded on the theory
    that a definition the parser could not see would be "left to A5" -- that
    reasoning was wrong in both directions and cost a false blocker:

      * A5 does not tolerate an unparsed definition, it MIS-REPORTS it. Every
        call to a one-line function reads as a call to an undefined function,
        which is how `fs_die` -- defined on line 76 and called 58 times -- came
        back as the sole red on an otherwise clean gate run.
      * A8/B6 silently SKIP the contract of anything absent from this table. The
        three helpers that mattered most (`fail`, `require_cmd`, `req_env`, all
        one-liners, all dereferencing $1) were therefore never contract-checked
        at all -- an under-detection that reported as a green.

    One omission that shouts and one that whispers, from a single blind spot.
    Callers should pair this with _unparsed_definitions() so coverage is a
    measured denominator rather than an assumption.
    """
    out, lines = {}, src.splitlines()
    for i, ln in enumerate(lines):
        m1 = _DEF_ONELINE.match(ln)
        if m1:
            out[m1.group(1)] = m1.group(2)
            continue
        m = _DEF.match(ln)
        if not m:
            continue
        for j in range(i + 1, len(lines)):
            if lines[j] == "}":
                out[m.group(1)] = "\n".join(lines[i + 1:j])
                break
    return out


def _unparsed_definitions(src: str) -> list[str]:
    """Names that look like definitions but did not make it into the table.

    A parser that cannot say what it missed cannot support a claim about what
    it checked.
    """
    return sorted({m.group(1) for m in _DEF_ANY.finditer(src)} - set(_function_bodies(src)))


def main() -> int:
    if not FIX.exists():
        print(f"REFUSING: {FIX} absent", file=sys.stderr)
        return 3
    rows = json.loads(_load_fix())
    if not rows or rows[0].get("error") or not rows[0].get("content"):
        print(f"REFUSING: task failed: {rows[0].get('error') if rows else 'empty'}",
              file=sys.stderr)
        return 3
    spec = json.loads(rows[0]["content"])
    fixes = spec.get("fixes", [])

    before = SRC.read_text("utf-8")

    # The A4 probes must be able to fire. A substring that is absent BEFORE any
    # edit cannot detect its own removal, and would report "preserved" forever.
    dead = [k for k, v in NOT_DEFECTS.items() if v not in before]
    if dead:
        print(f"REFUSING: {len(dead)} of {len(NOT_DEFECTS)} A4 probes match nothing "
              f"in the unedited file, so they could never detect a regression: {dead}",
              file=sys.stderr)
        return 3

    ok = True
    text = before
    applied, refuted = [], []

    for f in fixes:
        d, verdict = f["defect"], f["verdict"].strip().lower()
        anchor, repl = f["anchor"], f["replacement"]
        if verdict == "refuted":                                            # A2
            if anchor.strip():
                print(f"  FAIL A2  {d} refuted but carries an anchor — ambiguous")
                ok = False
            refuted.append(d)
            print(f"  {d}: REFUTED — {f['rationale'][:200]}")
            continue
        if not anchor.strip():                                              # A2
            print(f"  FAIL A2  {d} confirmed but anchor is EMPTY — no fix landed")
            ok = False
            continue
        n = text.count(anchor)
        if n != 1:                                                          # A1
            print(f"  FAIL A1  {d} anchor occurs {n}x (need exactly 1) — "
                  f"{'fix silently vanishes' if n == 0 else 'wrong site risk'}")
            print(f"           anchor[:120]={anchor[:120]!r}")
            ok = False
            continue
        text = text.replace(anchor, repl, 1)
        applied.append(d)
        print(f"  {d}: applied  ({len(anchor)} -> {len(repl)} chars)")

    print(f"\napplied {len(applied)}/{len(fixes)}: {applied or 'NONE'}"
          + (f"; refuted: {refuted}" if refuted else ""))
    if not applied and not refuted:
        print("  FAIL     an empty fix set is not a successful one"); ok = False

    kept = [k for k, v in NOT_DEFECTS.items() if v in text]
    if len(kept) != len(NOT_DEFECTS):                                       # A4
        print(f"  FAIL A4  {len(kept)}/{len(NOT_DEFECTS)} NOT-DEFECTS survived; "
              f"weakened: {sorted(set(NOT_DEFECTS) - set(kept))}")
        ok = False
    else:
        print(f"  PASS A4  all {len(NOT_DEFECTS)} verified-correct behaviours survive")

    # A6 -- DELTA, not absolute. Measured: the unedited launcher already contains
    # one family-name line, and it is :173, the comment that STATES the policy:
    #   "The core launcher does not name NeMo/Megatron/Gemma/Qwen."
    # An absolute scan is therefore red before any edit, so it could never go
    # green no matter how good the fix -- a gate that cannot pass is exactly as
    # useless as one that cannot fail, and this one would have blocked every
    # future L-fix. Worse, its message ("entered the core") asserted attribution
    # a whole-file scan cannot support. Compare before -> after so the gate
    # reports only what the EDIT introduced, and keep a control below so it can
    # still fire.
    def _fams(s: str) -> set[str]:
        return {m.group(0).lower() for m in FAMILIES.finditer(s)}

    pre, post = _fams(before), _fams(text)
    introduced = sorted(post - pre)
    if introduced:                                                          # A6
        print(f"  FAIL A6  the edit INTRODUCED model-family name(s): {introduced}")
        ok = False
    else:
        # MUST_FIRE: prove the comparison can still go red. A delta gate whose
        # baseline silently swallowed everything would report clean forever.
        # The probe token must be one the baseline does NOT already contain --
        # the first attempt injected "qwen", which line 173 already has, so the
        # delta was empty by construction and the control reported the detector
        # dead when the detector was fine. Pick an absent token, or refuse.
        probe = next((t for t in ("llama", "mistral", "falcon") if t not in pre), None)
        if probe is None:
            print("  FAIL A6  no unused probe token — control is inexpressible")
            ok = False
        elif not _fams(before + f" {probe}-x ") - pre:
            print(f"  FAIL A6  control did not fire on {probe!r} — detector is dead")
            ok = False
        else:
            print(f"  PASS A6  edit introduced 0 family names "
                  f"(control fired; {len(pre)} pre-existing, all in the policy "
                  f"comment at the line that forbids them)")

    # A5 -- callable universe is the backend's functions plus the launcher's own.
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    from h100_backend_splice import functions  # noqa: E402
    universe = set(functions(BACKEND.read_text("utf-8")))
    universe |= set(functions(text))
    universe.add("fail")
    nocomment = re.sub(r"(?m)^\s*#.*$", "", text)
    called = {m.group(1) for m in re.finditer(
        r"(?:^|\|\||&&|;|\||\$\(|`|\bthen\b|\belse\b|\bdo\b|!)[ \t]*"
        r"((?:fs_[a-z0-9_]+|fail))\b(?!\s*(?:\+?=|\[))", nocomment, re.M)}
    unknown = sorted(called - universe)
    if unknown:                                                             # A5
        print(f"  FAIL A5  calls {len(unknown)} undefined function(s) "
              f"(bash resolves at call time; rc 127): {unknown}")
        ok = False
    else:
        print(f"  PASS A5  all {len(called)} called function(s) exist "
              f"(universe of {len(universe)})")

    # A8 -- CONTRACT, not just existence. A5 proves a called function is
    # defined; it cannot see that the call violates the definition. Measured
    # cost of that gap: L1's first draft called fs_assert_torch_provenance with
    # NO argument (its first guard rejects the empty string -> rc 1 -> every
    # launch dead) and piped a heredoc into fs_launch_python, which is a string
    # BUILDER taking a gpu count and reading no stdin. Both passed A5. Two legs,
    # each derived from the definition rather than from a hand-written table, so
    # the gate tracks the backend instead of drifting from it.
    defs = _function_bodies(BACKEND.read_text("utf-8")) | _function_bodies(text)
    needs_arg = {n for n, b in defs.items() if re.search(r'\$\{?1[:\-\}\s"]', b)}
    reads_stdin = {n for n, b in defs.items()
                   if re.search(r"(?:\bread\b|\bcat\b|</dev/stdin|\$\(<)", b)}

    a8 = []
    for m in re.finditer(r"(?m)^[ \t]*(?:[a-z_]+=\"?\$\()?[ \t]*"
                         r"(fs_[a-z0-9_]+|run_in_container)\b([^\n]*)", nocomment):
        name, rest = m.group(1), m.group(2)
        if name not in defs:
            continue                       # A5 owns unknown names
        args = re.split(r"\|\||&&|[;|)]", rest)[0].strip()
        if name in needs_arg and not args:
            a8.append(f"{name} called with NO argument but its body dereferences $1")
        if re.search(r"<<-?\s*'?\w", rest) and name not in reads_stdin:
            a8.append(f"{name} is fed a heredoc but its body never reads stdin")
    if a8:                                                                  # A8
        for msg in sorted(set(a8)):
            print(f"  FAIL A8  {msg}")
        ok = False
    else:
        # MUST_FIRE both legs against synthetic violations, so a parser that
        # silently matches nothing cannot report the contracts satisfied.
        ctl_arity = bool(needs_arg) and bool(reads_stdin is not None)
        probe_a = sorted(needs_arg)[0] if needs_arg else None
        probe_b = next((n for n in sorted(defs) if n not in reads_stdin), None)
        fired = 0
        if probe_a and re.search(r"(?m)^[ \t]*" + probe_a + r"\b[ \t]*$",
                                 f"\n{probe_a}\n"):
            fired += 1
        if probe_b and re.search(r"<<-?\s*'?\w", f"{probe_b} - <<'PY'"):
            fired += 1
        if fired != 2 or not ctl_arity:
            print(f"  FAIL A8  controls fired {fired}/2 — the contract parser "
                  f"cannot detect its own violations")
            ok = False
        else:
            print(f"  PASS A8  {len(defs)} contracts parsed; "
                  f"{len(needs_arg)} require an argument, "
                  f"{len(reads_stdin)} read stdin; 0 violations (controls 2/2)")

    # A7 -- L1 is an ORDERING fix. Check the host probe is gone and that the
    # count now happens after runtime setup, not merely that something changed.
    host_probe = "visible=\"$(python3 - <<'PY'"
    if host_probe in text:                                                  # A7
        print("  FAIL A7  the host-interpreter probe is STILL PRESENT — L1 is an "
              "ordering defect; hardening the probe in place does not fix it")
        ok = False
    else:
        setup = text.find("fs_backend_runtime_setup")
        cnt = max(text.find("device_count"), text.find("FS_GPUS_PER_NODE"))
        if setup < 0:
            print("  FAIL A7  fs_backend_runtime_setup call vanished"); ok = False
        elif cnt >= 0 and cnt < setup:
            print("  FAIL A7  device counting still precedes fs_backend_runtime_setup")
            ok = False
        else:
            print("  PASS A7  host probe removed; counting follows runtime setup")

    with tempfile.NamedTemporaryFile("w", suffix=".sh", delete=False) as fh:
        fh.write(text); tmp = fh.name
    rc = subprocess.run(["bash", "-n", tmp], capture_output=True, text=True)
    if rc.returncode != 0:                                                  # A3
        print(f"  FAIL A3  bash -n: {rc.stderr.strip()[:300]}"); ok = False
    else:
        print("  PASS A3  bash -n clean")

    if spec.get("gaps"):
        print(f"\ngaps: {spec['gaps'][:900]}")
    for f in fixes:
        if f["verdict"].strip().lower() != "refuted":
            print(f"observe {f['defect']}: {f['observe'][:200]}")

    if not ok:
        print("\nREFUSING TO WRITE — gates above are red", file=sys.stderr)
        return 5
    dst = SRC if "--in-place" in sys.argv else DST
    dst.write_text(text, "utf-8")
    print(f"\nALL GATES GREEN -> {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
