#!/usr/bin/env python3
"""#152: replace the hard-coded Slurm partition name with a declared, required partition knob.

MEASURED BEFORE WRITING (literal partition name, on the generated launcher):
  launch_fs_h100.fixed.sh   13 hits
    :2    model-identity comment line
    :4    CRITICAL WALLTIME comment line
    :6    #SBATCH --partition=           <- the actual submit directive  (FUNCTIONAL)
    :116  fail 96 "... conflicts with <p> max 7-00:00:00 ..."
    :141  sinfo -h -p <p> -o '%l'        <- the walltime-cap probe       (FUNCTIONAL)
    :142  fail 96 '<p> partition max probe returned nothing; ...'        (single-quoted)
    :147  fail 96 "unparseable <p> partition max ..."
    :155  fail 96 "... exceeds finite <p> max ..."
    :157  fail 96 "cannot prove submitted TimeLimit <= <p> max; ..."
    :159  fail 96 "cannot prove submitted TimeLimit <= <p> max; got ..."
    :166  printf 'NOTICE: <p> partition reports UNLIMITED ...'           (single-quoted)
    :168  printf '<p> partition max walltime: %s (%ss)'                  (single-quoted)
    :178  fail 96 "FS_NCCL_IB_HCA pinning is unmeasured on <p>; ..."
Exactly two of the 13 are functional; 11 are message/comment text. ALL 13 must go: a partition
name left in an error message is still a published estate identifier, and it goes stale the
moment this launcher runs anywhere else -- a message naming the wrong partition is worse than
one naming none.

WHY THIS IS TWO DEFECTS IN ONE.

  (a) PUBLISHABILITY. This repository is public and the partition name is a cluster
      identifier. It was INVISIBLE to the build's blocklist for a reason worth recording:
      the blocklist's estate-identifier shapes are hostnames, paths and ticket-shaped
      tokens, and a bare lowercase partition name matches none of them. It surfaced only
      when a third scan, over published .md documents with no path-adjacency requirement,
      implicated the launcher as well. A detector blind to a class of identifier is not a
      green board, it is an UNMEASURED one.

  (b) GENERALIZABILITY, which is the requester's actual brief: model-specific components
      isolated from the core training infrastructure. The argument applies identically to
      ESTATE-specific components: a launcher that can only submit to one site's partition
      is not a foundation-model training framework, it is one site's script. This is the
      exact twin of #123 (the hard-coded filesystem root) and the fix is deliberately the
      same shape -- a required-no-default knob that fails closed at its point of use -- so
      the two read as one policy rather than two patches.

THE KNOB IS FS_PARTITION, REQUIRED WITH NO DEFAULT. No fallback is invented, not even a
"sensible" one: the house rule (stated by #123's stage, citing the FS_ALLOWED_NODE /
FS_CONTAINER_RUNTIME / FS_ALLOWED_PATH_ROOTS family) is that an unconfigured guard is a
disabled standing rule, and a default would re-bake the literal this stage removes. The
launcher refuses 96 when it is unset, saying what to set and why there is no default.

THE #SBATCH TRAP, HANDLED. An #SBATCH line is a COMMENT to the shell:
`#SBATCH --partition=$FS_PARTITION` never expands -- Slurm parses the dollar-sign text
literally. The directive therefore cannot be parameterised in place, and leaving a line
whose appearance lies about its meaning is worse than deleting it. The directive at :6 is
DELETED and the partition travels as a REAL command-line flag, --partition="$FS_PARTITION",
on the sbatch invocation the launcher already builds -- one flag, at the submit that
creates the allocation. Steps inside the allocation inherit it, which is why srun call
sites are deliberately untouched: adding the flag there would declare the same plane twice,
and a second declaration beside the live one is how the next reader concludes the wrong one
is authoritative. If this stage cannot resolve a UNIQUE sbatch invocation it REFUSES,
naming what it could not resolve; a patch stage that half-applies is worse than one that
declines.

TWO NAMES, ON PURPOSE. The literal being REMOVED arrives as FS_PARTITION_LITERAL; the knob
being INTRODUCED is FS_PARTITION. They must not share a name: one is an input to THIS STAGE
(so the stage can build the anchors it must match without checking the literal into the
repository -- #151 showed the de-hard-coding stage hard-coding the root ten times), the
other is an input to the GENERATED LAUNCHER at runtime. A reader who saw one name would
reasonably assume one of the two uses was redundant; neither is.

Deliberately NOT done: no default anywhere; no case-folding or "did you mean" on the
literal (an anchor that half-matches is a generator drift to refuse, not to repair); no
attempt to keep an #SBATCH-shaped line whose text cannot expand.

GATES
  E0  self-scan: this stage's own source carries no occurrence of the literal
  E1  idempotent: a second run is a reported no-op, not a failure
  E2  completeness with a real denominator (13/13 anchors located, 13 literal occurrences
      accounted for, unique sbatch invocation re-pointed, guard anchored); zero sites is
      UNMEASURED (95), never PASS
  E3  residue: zero occurrences of the literal survive, against the pre-patch denominator
  E4  bash -n clean on the patched launcher
  E5  the emitted FS_PARTITION guard is EXECUTED, not read:
        MUST_FIRE  unset FS_PARTITION  -> observed REFUSE 96
        MUST_PASS  set FS_PARTITION    -> observed admission, guard silent
CONTROLS, run before the real patch, on their own lines:
  MUST_FIRE  a synthetic body still carrying the literal is observed going RED on E3
  MUST_FIRE  a synthetic body with ZERO sites is observed becoming UNMEASURED (95), not PASS
  MUST_PASS  an already-patched synthetic body is observed as a clean idempotent no-op (0)
If any control misbehaves the stage REFUSES: a detector that failed its own controls has
nothing to say about the launcher.

EXIT CODES: 0 success, 95 UNMEASURED, 96 REFUSE. Note the documented trap:
`raise SystemExit("text")` prints the text but exits 1, silently breaking the contract.
Print to stderr, then raise SystemExit(<number>).
"""

from __future__ import annotations

import os
import pathlib
import re
import subprocess
import sys
import tempfile

# Resolved against THIS FILE, not the cwd and not an absolute literal. patch_estate_roots.py
# hard-codes a developer's home directory here; that works only because build_h100_plane.sh
# cds to its own dirname first, so the stage is silently coupled to one checkout on one
# machine. A build input that only resolves on the author's laptop is #142 in miniature --
# the launcher sourcing a filename the build never produces -- so this stage does not repeat
# it. patch_list_separators.py already uses the relative form; this is the same idea made
# cwd-independent.
LAUNCH = pathlib.Path(__file__).resolve().parent / "h100" / "gen" / "launch_fs_h100.fixed.sh"
MARK = "fs152:"

# Measured in finding #152: exactly 13 sites carry the literal, two functional, eleven
# message text. The denominator lives here, not in a docstring alone: if the generator
# drifts and the count no longer matches, the stage must refuse rather than patch blind.
EXPECTED_SITES = 13

LITERAL_SHAPE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def _literal() -> str:
    """The literal being REMOVED, supplied as an input -- never written into this file.

    #151 measured why: the stage that de-hard-codes the estate root hard-coded that root
    ten times in its own SITES table, making the de-hard-coding stage the last
    unpublishable file in the build. The anchors are text this stage must MATCH, so they
    cannot be invented -- but they can be an input, exactly as the FS_PARTITION policy this
    stage emits is an input: the operator migrating away from a partition name necessarily
    knows what that name is.
    """
    # `raise SystemExit("text")` prints the text but exits 1, which would quietly break the
    # declared 0 / 95 / 96 contract from inside the very stage that argues for it. Print,
    # then exit with the number that was promised.
    lit = os.environ.get("FS_PARTITION_LITERAL", "").strip()
    if not lit:
        print(
            "REFUSE 96: FS_PARTITION_LITERAL is unset (required, no default by design).\n"
            "  This stage rewrites a launcher that hard-codes one estate's Slurm partition.\n"
            "  Set FS_PARTITION_LITERAL to that name (the literal being REMOVED) so the\n"
            "  anchors can be built. It is deliberately not stored in this repository:\n"
            "  a checked-in partition name is a published partition name -- which is the\n"
            "  exact defect (#152a) this stage exists to remove from the launcher.\n"
            "  Note the deliberate pair: FS_PARTITION_LITERAL feeds THIS STAGE;\n"
            "  FS_PARTITION will feed the GENERATED LAUNCHER at runtime.", file=sys.stderr)
        raise SystemExit(96)
    if not LITERAL_SHAPE.match(lit):
        # A partition name with whitespace, slashes or quotes would corrupt the anchors and
        # the residue count rather than merely failing to match; refuse the input itself.
        print(f"REFUSE 96: FS_PARTITION_LITERAL is not a partition-shaped token: {lit!r}",
              file=sys.stderr)
        raise SystemExit(96)
    return lit


def _sites(L: str) -> list[tuple[str, str]]:
    """The (anchor, replacement) table, with the partition literal supplied rather than compiled in."""
    # The six double-quoted failure messages interpolate the VALUE by simple substitution;
    # the name of the knob in a message would be documentation, not diagnosis.
    dq = [
        '  fail 96 "FS_WALLTIME=\'$FS_WALLTIME\' conflicts with ' + L + ' max 7-00:00:00; refusing instead of clamping"\n',
        '  max_sec="$(fs_tl_seconds "$part_max")" || fail 96 "unparseable ' + L + ' partition max \'$part_max\'; UNMEASURED is not PASS"\n',
        '    [[ "$max_unlimited" == 1 ]] || fail 96 "submitted TimeLimit=UNLIMITED exceeds finite ' + L + ' max \'$part_max\'"\n',
        '    sub_sec="$(fs_tl_seconds "$submitted")" || fail 96 "cannot prove submitted TimeLimit <= ' + L + ' max; unparseable TimeLimit \'$submitted\'"\n',
        '      (( sub_sec <= max_sec )) || fail 96 "cannot prove submitted TimeLimit <= ' + L + ' max; got \'$submitted\' (${sub_sec}s) > \'$part_max\' (${max_sec}s)"\n',
        '  fail 96 "FS_NCCL_IB_HCA pinning is unmeasured on ' + L + '; leave unset unless measured and validated"\n',
    ]
    return [
        # :2 -- the launcher's self-description named one estate's partition as if it were
        # part of the framework's identity. The knob is an input, not an identity.
        (
            "# launch_fs_h100.sh -- model-agnostic single-node H100/" + L + " launcher for FoundationScale.\n",
            "# launch_fs_h100.sh -- model-agnostic single-node H100 launcher for FoundationScale.\n"
            "# fs152: the submit partition is an INPUT (FS_PARTITION, required, no default),\n"
            "# not a property of this file -- see the guard directly below the #SBATCH block.\n",
        ),
        # :4 -- the walltime figure is a measured property of the estate where the number
        # was taken, so the comment keeps the number but attaches it to the knob, not to a
        # baked-in name.
        (
            "# CRITICAL WALLTIME: " + L + " partition maximum is 7-00:00:00. The GB200 standing rule\n",
            "# CRITICAL WALLTIME: the partition named by FS_PARTITION has a finite maximum\n"
            "# (7-00:00:00 in the estate this launcher was measured against). The GB200 standing rule\n",
        ),
        # :6 -- DELETED, not parameterised in place. An #SBATCH line is a COMMENT to the
        # shell: `#SBATCH --partition=$FS_PARTITION` never expands, Slurm would parse the
        # dollar-sign text literally, and the failed submit would blame the operator's
        # spelling of a knob nothing had read. A directive that cannot mean what it says
        # must not keep a shape that implies it can; the partition travels as a real flag
        # on the sbatch invocation instead (gate E2b).
        (
            "#SBATCH --partition=" + L + "\n",
            "# fs152: the `#SBATCH --partition=...` directive that stood here is DELETED, not\n"
            "# parameterised in place -- an #SBATCH line is a comment to the shell, so an\n"
            "# expanded-looking form would silently mean something different from what it\n"
            "# says. The partition now travels as --partition=\"$FS_PARTITION\" on the sbatch\n"
            "# invocation below, where expansion actually happens.\n",
        ),
        # :116 -- double-quoted: plain value interpolation.
        (dq[0], dq[0].replace(L, "${FS_PARTITION}")),
        # :141 -- the walltime-cap probe must ask Slurm about the partition actually in use.
        (
            "part_max=\"$(sinfo -h -p " + L + " -o '%l' 2>/dev/null | head -n1 || true)\"\n",
            "part_max=\"$(sinfo -h -p \"$FS_PARTITION\" -o '%l' 2>/dev/null | head -n1 || true)\"\n",
        ),
        # :142 -- SINGLE-QUOTED: the quotes must change, or the message would print the
        # knob's NAME instead of its VALUE, and a diagnostic that reports
        # "$FS_PARTITION" to an operator mid-failure is a dead end dressed as help.
        (
            "[[ -n \"$part_max\" ]] || fail 96 '" + L + " partition max probe returned nothing; UNMEASURED is not PASS'\n",
            "[[ -n \"$part_max\" ]] || fail 96 \"$FS_PARTITION partition max probe returned nothing; UNMEASURED is not PASS\"\n",
        ),
        (dq[1], dq[1].replace(L, "${FS_PARTITION}")),
        (dq[2], dq[2].replace(L, "${FS_PARTITION}")),
        (dq[3], dq[3].replace(L, "${FS_PARTITION}")),
        (dq[4], dq[4].replace(L, "${FS_PARTITION}")),
        # :166 -- single-quoted printf format: keep the format single-quoted and pass the
        # value as an argument, so the $-text is never confused for format text.
        (
            "    printf 'NOTICE: " + L + " partition reports UNLIMITED max walltime\\n'\n",
            "    printf 'NOTICE: %s partition reports UNLIMITED max walltime\\n' \"$FS_PARTITION\"\n",
        ),
        # :168 -- same single-quoted-printf trap; the partition becomes the first %s.
        (
            "    printf '" + L + " partition max walltime: %s (%ss)\\n' \"$part_max\" \"$max_sec\"\n",
            "    printf '%s partition max walltime: %s (%ss)\\n' \"$FS_PARTITION\" \"$part_max\" \"$max_sec\"\n",
        ),
        # :178 -- "unmeasured on <partition>" must name the partition actually in use.
        (dq[5], dq[5].replace(L, "${FS_PARTITION}")),
    ]


# --- the emitted guard, inserted once directly below the #SBATCH block -------------------
# WHY IT IS SELF-CONTAINED (echo/exit, not the launcher's fail()): it is inserted directly
# under the #SBATCH block, BEFORE the launcher's fail() helper is defined. A guard that
# calls a helper it precedes refuses nothing -- the call itself would be the failure, with
# a message that blames the script instead of the missing configuration.
# WHY REQUIRED, NO DEFAULT: the same contract as FS_ALLOWED_NODE, FS_CONTAINER_RUNTIME and
# FS_ALLOWED_PATH_ROOTS (#123), for the same stated reason -- an unconfigured guard is a
# disabled standing rule. A default here would be the deleted literal compiled back in,
# and a checked-in partition name is a published estate identifier.
GUARD_BLOCK = (
    "\n"
    "# --- fs152: declared partition policy, replacing a hard-coded partition name --------\n"
    "# REQUIRED, NO DEFAULT -- the same contract as FS_ALLOWED_NODE, FS_CONTAINER_RUNTIME\n"
    "# and FS_ALLOWED_PATH_ROOTS. An unconfigured guard is a disabled standing rule; a\n"
    "# default here would be the deleted literal compiled back in.\n"
    "[[ -n \"${FS_PARTITION:-}\" ]] || { echo \"REFUSE 96: FS_PARTITION is unset (required, no default by design). Set it to this estate's Slurm submit partition -- the framework refuses to guess a cluster layout.\" >&2; exit 96; }\n"
)
GUARD_LINE = '[[ -n "${FS_PARTITION:-}" ]]'


# --- detector primitives, shared by the controls and the real run ------------------------
# The controls below exist to prove THESE functions can go red: doctrine 3 -- a check
# never seen to fail is not known to work, and all([]) is True, so "no sites found" must
# classify as UNMEASURED, never as PASS.

def _classify(text: str, lit: str) -> str:
    """One of 'patched' (idempotent no-op), 'unmeasured' (zero sites) or 'dirty'."""
    if MARK in text:
        return "patched"
    if lit not in text:
        return "unmeasured"
    return "dirty"


def _entry_rc(verdict: str) -> int | None:
    """The rc a verdict produces at the front door; None means 'sites exist, proceed'."""
    return {"patched": 0, "unmeasured": 95, "dirty": None}[verdict]


def _residue(text: str, lit: str) -> int:
    """E3's measurement: occurrences of the literal surviving in an output body."""
    return text.count(lit)


def _bash(script: str, env: dict | None = None) -> subprocess.CompletedProcess[str]:
    e = dict(os.environ)
    e.pop("FS_PARTITION", None)  # drills must not inherit the operator's shell
    if env:
        e.update(env)
    return subprocess.run(["bash", "-c", script], capture_output=True, text=True, env=e)


# --- CONTROLS: run before the real patch; a detector that failed its own controls has
# --- nothing to say about the launcher.

def _controls(lit: str) -> bool:
    ok = True

    # MUST_FIRE 1: a body still carrying the literal must drive the E3 measurement red.
    dirty = f'echo "submitting to {lit}"\nsleep 1\n'
    n = _residue(dirty, lit)
    if _classify(dirty, lit) == "dirty" and n > 0:
        print(f"PASS control MUST_FIRE  residue detector observed RED on a body still "
              f"carrying the literal (residue={n}/1)")
    else:
        print("FAIL control MUST_FIRE  residue detector did NOT go red on a body carrying "
              "the literal -- E3 cannot be trusted to catch a missed site", file=sys.stderr)
        ok = False

    # MUST_FIRE 2: a body with ZERO sites must classify UNMEASURED (95), not PASS. This is
    # the vacuous-truth drill this codebase exists for: if the launcher were regenerated
    # under a different partition, a naive 'no hits -> clean' stage would silently rewrite
    # nothing and report green.
    empty = "#!x\n# a launcher regenerated under a different partition\nset -u\n"
    if _classify(empty, lit) == "unmeasured" and _entry_rc(_classify(empty, lit)) == 95:
        print("PASS control MUST_FIRE  zero-site body observed becoming UNMEASURED (exit 95), "
              "not PASS")
    else:
        print("FAIL control MUST_FIRE  zero-site body did not classify as UNMEASURED -- "
              "E2 would report vacuous success on a regenerated launcher", file=sys.stderr)
        ok = False

    # MUST_PASS: an already-patched body must be a clean idempotent no-op, not a failure.
    marked = "#!x\n# fs152: partition is operator-supplied\nset -u\n"
    if _classify(marked, lit) == "patched" and _entry_rc(_classify(marked, lit)) == 0:
        print("PASS control MUST_PASS  already-patched body observed as a clean idempotent "
              "no-op (exit 0)")
    else:
        print("FAIL control MUST_PASS  already-patched body was not an idempotent no-op -- "
              "a second run of this stage would fail the build for no defect",
              file=sys.stderr)
        ok = False

    return ok


def main() -> int:
    L = _literal()

    # E0 (#151, #145): this stage's OWN source must carry no partition literal. It is the
    # file that deletes the literal from the launcher, so it was exactly where a literal
    # could hide while the build reported clean -- the build only ever scanned GENERATED
    # artifacts, never the generator. THE #145 HANDLING, stated precisely because the
    # template shows anchoring matters: the fs-root stage carries a BLOCKLIST pattern whose
    # own text legitimately contains the redacted vocabulary, so its self-scan excludes the
    # pattern's span. THIS stage carries no such definition -- the anchors above are built
    # at runtime from FS_PARTITION_LITERAL, so the literal's source text never appears in
    # this file, and the exempt-span list is EMPTY BY CONSTRUCTION, computed below rather
    # than assumed: if a future redaction pattern ever legitimately names a partition
    # shape, it slots into EXEMPT_SPANS -- as a recorded, located span -- without
    # weakening the check into a bare no-op.
    self_src = pathlib.Path(__file__).read_text("utf-8")
    exempt_spans: list[tuple[int, int]] = []
    real = [m for m in re.finditer(re.escape(L), self_src)
            if not any(a <= m.start() < b for a, b in exempt_spans)]
    if real:
        where = [f"L{self_src[:m.start()].count(chr(10)) + 1}:{m.group(0)}" for m in real[:6]]
        print(f"  FAIL E0  this stage's own source carries {len(real)} occurrence(s) of the "
              f"partition literal: {where}", file=sys.stderr)
        return 96
    print("  PASS E0  stage source carries no partition literal "
          "(0/0; the literal reaches this file only via FS_PARTITION_LITERAL)")

    # CONTROLS before the real patch: prove the detector can go red and can decline.
    if not _controls(L):
        print("\nREFUSING TO RUN -- this stage's own gate logic failed its controls; "
              "a detector that cannot fire has nothing to say about the launcher",
              file=sys.stderr)
        return 96

    if not LAUNCH.exists():
        # Fail closed: a missing input is not a zero, it is an unread measurement.
        print(f"REFUSE 96: launcher not found: {LAUNCH}", file=sys.stderr)
        return 96
    text = LAUNCH.read_text("utf-8")

    verdict = _classify(text, L)                                              # E1 / E2 front door
    if verdict == "patched":                                                  # E1
        print("  E1  already applied -- no-op (idempotent)")
        return 0
    if verdict == "unmeasured":                                               # E2 denominator is 0
        print("  UNMEASURED E2  0/{d} sites matched the literal in {p}\n"
              "  all([]) is True: zero units measured is UNMEASURED, never PASS. If the\n"
              "  generator now emits a different partition name, this stage must say so\n"
              "  rather than rewrite nothing and report clean.".format(
                  d=EXPECTED_SITES, p=LAUNCH.name), file=sys.stderr)
        return 95
    print("  PASS E1  not yet patched")

    sites = _sites(L)
    before = text.count(L)
    ok = True

    # E2a: every anchor unique (a shifted generator must not be patched blind), and the
    # total occurrence count must equal the number of anchors -- an occurrence OUTSIDE the
    # measured anchors would be a 14th site this stage has no replacement for.
    for i, (anchor, _) in enumerate(sites, 1):
        n = text.count(anchor)
        if n != 1:
            print(f"  FAIL E2  site {i:2d}/{EXPECTED_SITES} anchor occurs {n}x (need 1)",
                  file=sys.stderr)
            ok = False
    if before != EXPECTED_SITES:
        print(f"  FAIL E2  literal occurs {before}x in the launcher, measured denominator is "
              f"{EXPECTED_SITES} -- the generator emitted a different shape than #152 "
              f"measured", file=sys.stderr)
        ok = False
    if not ok:
        print("\nREFUSING TO WRITE -- the generator emitted a different shape than this "
              "patch was measured against", file=sys.stderr)
        return 96
    print(f"  PASS E2a  {EXPECTED_SITES}/{EXPECTED_SITES} sites located; literal occurrences "
          f"{before}/{EXPECTED_SITES} fully accounted for by anchors")

    new = text
    for anchor, repl in sites:
        new = new.replace(anchor, repl, 1)

    # E2b: re-point the submit. The dead #SBATCH directive is gone; the partition must
    # travel as a REAL flag on the one sbatch invocation the launcher builds. If a unique
    # sbatch invocation cannot be resolved, the stage REFUSES and names what it could not
    # resolve -- it does not guess, and it does not half-apply.
    # The idempotence probe must ignore COMMENTS. The replacement block installed at the
    # deleted #SBATCH directive explains, in prose, that the partition "now travels as
    # --partition=$FS_PARTITION on the sbatch invocation" -- so a bare substring test finds
    # the stage's own explanation and refuses on the first run. Same shape as #145 and #151:
    # a thing that describes X is not X, and a detector that cannot tell them apart flags
    # the code for documenting itself.
    already = [
        ln for ln in new.splitlines()
        if '--partition="$FS_PARTITION"' in ln and not ln.lstrip().startswith("#")
    ]
    if already:
        print("  FAIL E2b  --partition flag already present on a live line; refusing to "
              "declare the same plane twice", file=sys.stderr)
        return 96

    # This launcher RE-SUBMITS ITSELF: it sbatches "$0" once per phase (probe, production,
    # resume, post-mortem), so there are FOUR submit sites, not one, and each is nested
    # inside a `jid="$( ... )"` command substitution rather than sitting at the start of a
    # line. A line-anchored `^\s*sbatch` therefore matches ZERO of them.
    #
    # Every one of them needs the flag. Missing a single phase would not fail loudly -- that
    # phase would inherit the submitting job's partition when chained and fall to the cluster
    # default when launched directly, which is the kind of divergence that shows up as one
    # inexplicably-pending job days later.
    # COMMAND POSITION is the discriminator, and a bare `\bsbatch\b` is not it. Measured: a
    # word-boundary match found 6 sites in this launcher, and two were wrong --
    #   line 12   the explanatory comment this stage itself installs
    #   line 428  fail 96 'not in a Slurm allocation; submit with sbatch (or ...)'
    # The second is the dangerous one. It is SINGLE-quoted, so the injected
    # --partition="$FS_PARTITION" would never expand: an operator hitting that error would be
    # told to run a command containing a literal dollar-sign variable name. A rewrite that
    # lands in a diagnostic is worse than no rewrite, because it degrades the message an
    # operator reads precisely when something has already gone wrong.
    #
    # A real invocation sits in command position: after `$(`, a backtick, `;`, `&&`, `||`,
    # `|`, `(`, `&`, or the start of the line -- optionally behind VAR=value prefixes, which
    # is how three of the four real sites here are written. Prose and comments never satisfy
    # that. This is the same token-class-versus-context question as #145, in the rewriter.
    CMD_POS = re.compile(r"""(?:^|[;&|(]|\$\(|`)\s*(?:[A-Za-z_]\w*=\S*\s+)*$""", re.X)
    sb = []
    _off = 0
    for _ln in new.splitlines(keepends=True):
        if not _ln.lstrip().startswith("#"):
            for _m in re.finditer(r"(?<![\w-])sbatch(?=\s)", _ln):
                if CMD_POS.search(_ln[: _m.start()]):
                    sb.append(_off + _m.end())
        _off += len(_ln)
    if not sb:
        print("REFUSE 96: no sbatch invocation found to carry --partition=\"$FS_PARTITION\". "
              "The #SBATCH directive cannot be parameterised (it is a comment to the shell), "
              "so with no submit command at all this stage declines rather than half-apply.",
              file=sys.stderr)
        return 96
    # Right-to-left, so each insertion cannot shift the offsets of the sites not yet done.
    for at in reversed(sb):
        new = new[:at] + ' --partition="$FS_PARTITION"' + new[at:]
    # The discriminator is a detector and needs its own proof of life, run on a fixture that
    # contains one of each: a real command-substitution submit, a bare command-position
    # submit, an env-prefixed submit, a comment, and a single-quoted diagnostic. It must
    # rewrite exactly the first three. Written as a MUST_PASS *and* a MUST_FIRE in one:
    # 3 is the only answer that is neither blind nor over-eager.
    _fx = (
        '#!/bin/bash\n'
        '# submit with sbatch --parsable\n'                       # comment      -> no
        'jid="$(sbatch --parsable "$0")"\n'                       # $( )         -> yes
        'sbatch --hold "$0"\n'                                    # line start   -> yes
        'x="$(PROBE=1 FS_X=0 sbatch --parsable "$0")"\n'          # env-prefixed -> yes
        "fail 96 'not in an allocation; submit with sbatch (or set X=1)'\n"  # quoted -> no
    )
    _n = 0
    _o = 0
    for _l in _fx.splitlines(keepends=True):
        if not _l.lstrip().startswith("#"):
            for _mm in re.finditer(r"(?<![\w-])sbatch(?=\s)", _l):
                if CMD_POS.search(_l[: _mm.start()]):
                    _n += 1
        _o += len(_l)
    if _n != 3:
        print(f"REFUSE 96: command-position control failed -- matched {_n}/3 on a fixture "
              f"carrying 3 real submits, 1 comment and 1 single-quoted diagnostic. Too few "
              f"means real submit sites go un-flagged; too many means the rewrite lands in "
              f"prose, where $FS_PARTITION cannot even expand.", file=sys.stderr)
        return 96
    print("  PASS control  command-position discriminator 3/3 real submits, 0/2 "
          "comment-and-diagnostic false-positives")
    print(f"  PASS E2b  submit re-pointed: {len(sb)}/{len(sb)} sbatch invocation(s) now carry "
          "--partition=\"$FS_PARTITION\" (this launcher re-submits itself once per phase); "
          "srun call sites inherit the allocation and are deliberately untouched")

    # E2c: anchor the FS_PARTITION guard directly after the last remaining #SBATCH
    # directive -- executable code may not precede a directive Slurm still has to read.
    lines = new.splitlines(keepends=True)
    directive_idx = [i for i, ln in enumerate(lines) if ln.startswith("#SBATCH")]
    if not directive_idx:
        print("REFUSE 96: no #SBATCH directives remain to anchor the partition guard "
              "above; the guard would land before directives Slurm must still read, "
              "silently disabling them", file=sys.stderr)
        return 96
    lines.insert(directive_idx[-1] + 1, GUARD_BLOCK)
    new = "".join(lines)
    print(f"  PASS E2c  guard anchored after the last of {len(directive_idx)} retained "
          f"#SBATCH directive(s) (denominator: {len(directive_idx)})")

    after = _residue(new, L)                                                  # E3
    if after:
        hits = [f"L{new[:m.start()].count(chr(10)) + 1}:{m.group(0)}"
                for m in re.finditer(re.escape(L), new)]
        print(f"  FAIL E3  {after} occurrence(s) of the literal survive (was {before}): "
              f"{hits[:6]}", file=sys.stderr)
        ok = False
    else:
        print(f"  PASS E3  residue 0 (denominator: {before} pre-patch occurrences, "
              f"{before} -> 0)")

    with tempfile.NamedTemporaryFile("w", suffix=".sh", delete=False) as fh:
        fh.write(new)
        tmp = fh.name
    try:
        r = subprocess.run(["bash", "-n", tmp], capture_output=True, text=True)
    finally:
        os.unlink(tmp)
    if r.returncode != 0:                                                     # E4
        print(f"  FAIL E4  bash -n: {r.stderr.strip()[:300]}", file=sys.stderr)
        ok = False
    else:
        print("  PASS E4  bash -n clean")

    # E5 -- execute the emitted guard, do not read it (doctrine: the policy must be seen
    # to refuse and seen to admit, not merely be present).
    guard_line = next((ln for ln in new.splitlines() if ln.startswith(GUARD_LINE)), None)
    if guard_line is None:
        print("  FAIL E5  emitted FS_PARTITION guard not found; a control that cannot be "
              "built must not be reported green", file=sys.stderr)
        return 96
    r = _bash(guard_line)
    if r.returncode == 96 and "REFUSE 96" in r.stderr:
        print("  PASS E5a MUST_FIRE: unset FS_PARTITION observed being REFUSED (rc 96) by "
              "the emitted guard")
    else:
        print(f"  FAIL E5a unset FS_PARTITION was not refused by the emitted guard: "
              f"rc={r.returncode} {r.stderr.strip()[:150]}", file=sys.stderr)
        ok = False
    r = _bash(guard_line + '\necho ADMITTED\n', {"FS_PARTITION": "drillpartA"})
    if r.returncode == 0 and "ADMITTED" in r.stdout:
        print("  PASS E5b MUST_PASS: set FS_PARTITION observed admitted; the guard is "
              "silent when configured")
    else:
        print(f"  FAIL E5b configured FS_PARTITION was not admitted: rc={r.returncode}",
              file=sys.stderr)
        ok = False

    if not ok:
        print("\nREFUSING TO WRITE -- gates above are red", file=sys.stderr)
        return 96
    LAUNCH.write_text(new, "utf-8")
    print(f"\nALL GATES GREEN ({EXPECTED_SITES}/{EXPECTED_SITES} sites rewritten, residue 0) "
          f"-> {LAUNCH.name}")
    print("OPERATOR IMPACT, stated because it is a breaking change: every launch must now "
          "export FS_PARTITION, and sbatch receives it as a real --partition flag (the "
          "dead #SBATCH directive, which could never expand, is gone). That is the point -- "
          "the previous behaviour was not 'no configuration needed', it was 'one estate's "
          "partition compiled in'.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())