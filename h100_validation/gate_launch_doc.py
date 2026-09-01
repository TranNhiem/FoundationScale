#!/usr/bin/env python3
"""gate_launch_doc.py: pin h100/LAUNCH.md against the launcher *and the trainer* it claims to document.

WHY THIS GATE EXISTS. An operator document is a claim about a program. The first three
ways that claim drifts from the program were found in this estate before this gate was
written; the fourth was measured during a doc audit after the gate shipped:

  COVERAGE   The launcher refuses to start without a knob (exit 96); the document never
             mentions it. The operator learns the knob exists from the refusal message,
             in front of a deadline. The doc was never wrong about anything it said --
             it was wrong about what it omitted, and omissions produce no diff for a
             human reviewer to catch.
  CITATION   A doc row says "`FS_STEPS` is enforced at L:209". The launcher grows a
             function, L:209 is now a comment about NFS mounts, and the citation is a
             decorated guess. A renumbered line number is indistinguishable from a true
             one UNTIL YOU READ THE LINE. So this gate reads the line.
  REDACTION  The doc describes a real estate, and real estates have real paths, real
             partition names, real people. A literal that leaks once is archived
             forever. The patterns come from the environment so this gate never prints
             what it is looking for -- the gate's own output is DPIA-safe, which is not
             a nicety: a redaction gate whose failures name the secret is a leak with a
             green checkmark next to it.
  PARITY     The doc's single copy-pasteable engine command named a program no build
             stage produces and flags no parser declares. It would have passed every
             host-side gate, been granted an allocation, brought the container up,
             passed the collective probe -- and then died at argv parse, which is the
             most expensive place in the pipeline to discover a wrong flag spelling.
             Exit-code and output-marker claims about the in-container program drift
             the same way: the doc quoted an exit table the program does not implement
             and markers it never prints. L6-L11 exist so this class is caught by
             comparison, not by a human re-reading two files side by side.

A gate on any one of these was proposed and rejected as too easy to game by accident:
a coverage gate with zero knobs passes vacuously (`all([]) is True`), a citation gate
whose denominator silently shrinks passes vacuously, a redaction gate with no patterns
configured passes vacuously, and a parity gate over zero documented commands passes
vacuously. So every denominator is printed, a zero denominator is UNMEASURED and fails
the gate (exit 95), and the drills run on every invocation -- a detector that cannot be
observed going red is not a detector.

RULES
  L1  COVERAGE. Every required-no-default knob the launcher enforces is enumerated
      from the launcher itself -- lines matching the guard idiom
      `[[ -n "${NAME:-}" ]] ||` (the `||` arm exits 96 or calls `fail 96`). SLURM_*
      names are excluded (workload-manager supplied, not operator inputs). Each knob
      must be named in the doc. The knob list is NEVER hard-coded here.
  L2  CITATION RESOLUTION. Each doc line carrying `L:<n>` must resolve: launcher line
      <n> exists and contains the first backticked `FS_*` token on that doc line.
      A cited row with no backticked token counts in the denominator and is reported
      unresolvable -- dropping it is how this class of gate goes quietly vacuous.
  L3  REDACTION. Three patterns from the environment, each REQUIRED WITH NO DEFAULT:
      FS_ESTATE_ROOT (fixed string), FS_PARTITION_LITERAL (fixed string, word
      boundaries), FS_REDACT_EXTRA (already-formed regex alternation). Hits are
      reported by line number and count only. The pattern values never print.
  L4  Every verdict prints its denominator.
  L5  CONTROLS on every run: a MUST_FIRE drill per rule (covered knob deleted,
      citation renumbered, partition literal planted, unpublished script substituted,
      unknown flag appended, satisfied knob withdrawn, non-selected branch flag
      smuggled in, a documented exit code deleted, a bogus output marker inserted) and
      MUST_PASS on the unmodified pair. An unplantable drill is UNMEASURED and fails
      the gate -- it proves nothing.
  L6  SCRIPT IDENTITY. Every .py basename inside a documented FS_ENGINE_LAUNCH_CMD
      must be a file the publish set ships. A documented command referencing no .py
      at all hands the operator no program to run and is itself a finding.
  L7  FLAG EXISTENCE. Every `--flag` token in a documented engine command must be a
      flag the trainer's own `add_argument` calls declare -- enumerated by AST, never
      listed here.
  L8  REQUIRED-KNOB COVERAGE. Every unconditional required knob (from the trainer's
      sourcing calls, partitioned by AST ancestry) must be satisfied by a documented
      flag spelling or an `export` of its declared env fallback. Conditional knobs are
      enforced only on the branch the documented selector value selects; a selector
      that will not resolve makes the rule UNMEASURED, never skipped.
  L9  MODE EXCLUSIVITY. The trainer refuses knobs from the non-selected branch, so no
      flag classified into that branch may appear in the documented command. Branch
      membership comes from the same AST partition as L8; nothing about which flags
      conflict is written here.
  L10 EXIT-CODE TABLE COMPLETENESS. Every integer code the trainer's main() can
      return (plus argparse's exit-2-on-bad-argv whenever argparse is actually used)
      must appear in the doc's trainer-contract block as `**<n>**` or `` `<n>` ``.
  L11 OUTPUT-MARKER EXISTENCE. Every backticked SHOUTED token inside the
      trainer-contract block is a marker the doc attributes to that program, and must
      occur as a literal substring of the trainer source.

None of L6-L11 hard-codes an expectation: the script names, the flag spellings, the
knob list, the exit codes and the markers are all re-derived from the trainer source
or the publish set on every run. What this gate prints is names, codes, line numbers
and counts -- never the documented command string, never a path, never a doc line.

EXIT 0 all rules hold and all controls green; 5 RED; 95 UNMEASURED; 96 REFUSE.
"""

from __future__ import annotations

import ast
import os
import pathlib
import re
import sys

H100 = pathlib.Path(__file__).resolve().parent / "h100"
DOC = H100 / "LAUNCH.md"
LAUNCHER = H100 / "gen" / "launch_fs_h100.fixed.sh"
# Locations, not expectations: these two paths say WHERE the subjects of L6-L11 live.
# Every expectation about their contents (flags, knobs, codes, markers, basenames) is
# derived from the files at runtime; nothing else about them appears as a literal in
# this gate.
TRAINER = H100 / "gen" / "fs_train.fixed.py"
PUBLISH_SET = H100 / "PUBLISH_SET.txt"
BACKEND = H100 / "gen" / "fs_container_backend.bound.sh"

# L1: the launcher's own guard idiom for required-no-default knobs. Spacing-tolerant,
# but the SHAPE is fixed on purpose: `||` present, `${NAME:-}` empty-default probe.
GUARD_RE = re.compile(
    r'\[\[\s+-n\s+"\$\{([A-Za-z_][A-Za-z0-9_]*):-\}"\s+\]\]\s+\|\|'
)
# L2: a citation is <NOTATION>:<digits>; the row's subject is the FIRST backticked
# FS_ token. A citation is a line number OR an inclusive range: L:358, L:358-365,
# L:34–182. Both the ASCII hyphen and the en-dash a writer's editor substitutes for it,
# because a gate that only knows one of them silently reclassifies half the document's
# pointers as single-line citations and then reports them stale -- wrong, and wrong in
# the direction that manufactures work.
#
# #186: the notation is CAPTURED, not fixed to `L`. The doc's own legend declares two --
# L: launcher line, B: container-backend line -- and this pattern read only the first.
# The rule then printed `59/59 citation(s) resolve` and the gate's banner said "every
# citation read at its line" while 12 B: pointers were in no denominator at all. A rule
# whose denominator is a subset of its own claim is the defect this file exists to
# catch, and it had it. An unknown notation is now counted and reported, never dropped:
# a citation nobody can resolve must cost something.
CITE_RE = re.compile(r"\b([A-Z]):(\d+)(?:[-–—](\d+))?\b")

# notation -> the file whose line numbers it indexes. A LOCATION table, not an
# expectation: nothing here asserts what those files contain.
CITE_FILES = {"L": LAUNCHER, "B": BACKEND}

# Every identifier inside a backticked span, not the span itself. The first version of
# this required the whole backtick content to be a bare FS_ name, so `--partition="$FS_PARTITION"`
# -- a row that names its subject about as plainly as a row can -- registered as having
# no subject at all. Two separate narrowings, same defect: the oracle described a
# citation shape the document does not actually use.
SPAN_RE = re.compile(r"`([^`]+)`")
IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]{2,}")

# --- doc <-> trainer parity extraction (L6-L11) ---------------------------------

# The variable name is matched literally on purpose. Deriving it from the launcher's
# guard census was considered and read no cleaner: FS_ENGINE_LAUNCH_CMD is already a
# required knob L1 enumerates from the guard idiom, so the name is pinned by that rule
# and this regex is a second, independent read of the same name rather than a new
# expectation.
CMD_EXPORT_RE = re.compile(r'^\s*export\s+FS_ENGINE_LAUNCH_CMD\s*=\s*"(.*)"')

# NOT shlex. The documented value deliberately contains angle-bracket placeholders
# with spaces ("<absolute path to plane dir>") and shell expansions ($MODEL_DIR,
# \$OUT_DIR); shlex.split mangles both, and a parity gate that cannot read its own
# subject reads nothing. Script references are therefore gathered as runs of
# non-space, non-angle characters ending in .py (the run carries any path prefix,
# which is then dropped to a basename), and flags as --word tokens not preceded by a
# word character or a third hyphen (so "---" and "--opt=x--y" do not hallucinate).
SCRIPT_REF_RE = re.compile(r"[^\s<>]*\.py")
FLAG_TOKEN_RE = re.compile(r"(?<![\w-])(--[a-z0-9][a-z0-9-]*)")

# L11: inside the contract block, a backticked SHOUTED token is a marker the doc
# attributes to the program -- FSLEG/FSSUMMARY were exactly this shape and occur zero
# times in the shipped entrypoint. Bounded on both sides so myTOKEN and TOKENx do not
# partial-match.
MARKER_RE = re.compile(r"(?<![A-Za-z0-9_])([A-Z][A-Z0-9_]{3,})(?![A-Za-z0-9_])")


class Refusal(Exception):
    """An input is missing, unreadable, or a required pattern is unset. Exit 96."""


def load_patterns() -> list[tuple[str, "re.Pattern[str]"]]:
    """L3. All three patterns are REQUIRED WITH NO DEFAULT. Empty is unset."""
    root = os.environ.get("FS_ESTATE_ROOT")
    if not root:
        raise Refusal("FS_ESTATE_ROOT is unset or empty — the redaction gate cannot "
                      "certify a document against a secret it was never given")
    part = os.environ.get("FS_PARTITION_LITERAL")
    if not part:
        raise Refusal("FS_PARTITION_LITERAL is unset or empty — refusing rather than "
                      "letting L3 pass vacuously")
    extra = os.environ.get("FS_REDACT_EXTRA")
    if not extra:
        raise Refusal("FS_REDACT_EXTRA is unset or empty — refusing rather than "
                      "letting L3 pass vacuously")
    try:
        extra_re = re.compile(extra)
    except re.error as e:
        raise Refusal(f"FS_REDACT_EXTRA is not a compilable regex: {e}")
    return [
        ("FS_ESTATE_ROOT", re.compile(re.escape(root))),
        ("FS_PARTITION_LITERAL", re.compile(r"\b" + re.escape(part) + r"\b")),
        ("FS_REDACT_EXTRA", extra_re),
    ]


def enforced_knobs(lau_text: str) -> tuple[list[str], int, int]:
    """L1. Enumerate from the file, never from a hard-coded list.

    Returns (operator knob names in first-appearance order, guard lines seen,
    SLURM_* names skipped).
    """
    seen: set[str] = set()
    names: list[str] = []
    for m in GUARD_RE.finditer(lau_text):
        n = m.group(1)
        if n not in seen:
            seen.add(n)
            names.append(n)
    skipped = sum(1 for n in names if n.startswith("SLURM_"))
    operator = [n for n in names if not n.startswith("SLURM_")]
    return operator, len(names), skipped


def check_l1(doc_text: str, knobs: list[str]) -> list[str]:
    """Returns the knobs enforced by the launcher but never named in the doc."""
    return [k for k in knobs if k not in doc_text]


def row_subjects(dline: str) -> list[str]:
    """Every identifier a doc row names inside backticks -- its candidate subjects.

    Not just FS_ knobs. A row may point at `sinfo`, `sbatch` or a shell function, and
    those are exactly as checkable; restricting the vocabulary to FS_* did not make the
    gate stricter, it made whole categories of citation invisible and therefore
    unchecked. Deduplicated, order preserved, so the reported subject is stable.
    """
    seen: dict[str, None] = {}
    for span in SPAN_RE.findall(dline):
        for ident in IDENT_RE.findall(span):
            seen.setdefault(ident, None)
    return list(seen)


def block_bounds(doc_lines: list[str], i: int) -> tuple[int, int]:
    """The 0-based inclusive bounds of the block containing doc line i.

    THE UNIT OF A CITATION IS ITS BLOCK, NOT ITS PHYSICAL LINE. A markdown table row is
    one line, so for a table row the two are the same and nothing loosens. Prose is not:
    a sentence wraps, and the subject a citation is about routinely sits on the line
    above or below it -- `sinfo` named on one line, the (L:358-365) pointing at it on
    the previous one. Scoping subjects to the physical line reported seven such rows as
    "names no checkable subject", which was false: they name it, one line over.

    The tempting fix was to rewrite the prose so every citing line repeats a backticked
    token. That is contorting the artifact to fit the instrument. The instrument was
    wrong about what a row is.

    Blocks end at blank lines, at fence markers, and at table rows, so a paragraph can
    never absorb a neighbouring table's vocabulary.
    """
    def boundary(ln: str) -> bool:
        s = ln.strip()
        return not s or s.startswith("|") or s.startswith("```")

    if doc_lines[i].strip().startswith("|"):
        return i, i
    lo = hi = i
    while lo > 0 and not boundary(doc_lines[lo - 1]):
        lo -= 1
    while hi + 1 < len(doc_lines) and not boundary(doc_lines[hi + 1]):
        hi += 1
    return lo, hi


_CITE_SRC_CACHE: "dict[str, list[str]] | None" = None


def cite_sources() -> "dict[str, list[str]]":
    """Notation prefix -> the lines its citation numbers index.

    A notation whose file will not read is ABSENT from this map rather than present and
    empty. The difference is the whole point: absent makes every citation in that
    notation an unresolvable miss that names the notation, whereas present-and-empty
    would make them all "out of range" and read like a document error when the real
    fault is on the reading side.
    """
    global _CITE_SRC_CACHE
    if _CITE_SRC_CACHE is None:
        out: dict[str, list[str]] = {}
        for pfx, f in CITE_FILES.items():
            try:
                out[pfx] = f.read_text(encoding="utf-8", errors="replace").splitlines()
            except OSError:
                pass
        _CITE_SRC_CACHE = out
    return _CITE_SRC_CACHE


def check_l2(doc_lines: list[str],
             lau_lines: list[str]) -> tuple[list[str], int, int, int]:
    """L2. Returns (misses, resolved, total, exact).

    Every X:<n> or X:<a>-<b> on a doc line is one citation, in whichever notation the
    doc uses. The denominator counts ALL of them -- including rows that name no subject,
    and including notations this gate cannot resolve -- because count it, report it,
    never drop it is the only denominator discipline that survives a second notation
    being added to the legend (#186).

    A citation RESOLVES when some subject named on the row appears somewhere in the
    cited range OF THAT NOTATION'S FILE. `exact` separately counts the citations that
    land on the FIRST line of their own range, which is the strictest reading. Both
    numbers are printed: a range is a legitimate way to point at a block, but a gate
    that only reports the loose count lets a range quietly widen until it cannot fail.
    Reporting the strict count beside it means any such drift is visible in the ratio
    rather than hidden by it.

    `lau_lines` is still taken as an argument rather than read from the map, so the L2
    drill can poison the launcher view it hands in and watch this go red.
    """
    sources = dict(cite_sources())
    sources["L"] = lau_lines
    misses: list[str] = []
    resolved = 0
    exact = 0
    total = 0
    for dno, dline in enumerate(doc_lines, 1):
        cites = CITE_RE.findall(dline)
        if not cites:
            continue
        blo, bhi = block_bounds(doc_lines, dno - 1)
        subjects = row_subjects("\n".join(doc_lines[blo:bhi + 1]))
        for pfx, start_s, end_s in cites:
            total += 1
            lo = int(start_s)
            hi = int(end_s) if end_s else lo
            span = f"{pfx}:{lo}" + (f"-{hi}" if hi != lo else "")
            src = sources.get(pfx)
            if src is None:
                misses.append(f"{span} on doc line {dno}: notation `{pfx}:` indexes no "
                              f"source this gate can read, so the pointer is "
                              f"unverifiable — it stays in the denominator rather "
                              f"than being quietly skipped")
                continue
            if not subjects:
                misses.append(f"{span} on doc line {dno}: the row names no backticked "
                              f"subject, so nothing about it is checkable — stays in "
                              f"the denominator")
                continue
            if not (1 <= lo <= len(src) and 1 <= hi <= len(src) and lo <= hi):
                misses.append(f"{span} on doc line {dno} -> out of range (that source "
                              f"has {len(src)} line(s))")
                continue
            window = src[lo - 1:hi]
            hit = next((s for s in subjects if any(s in w for w in window)), None)
            if hit is None:
                misses.append(f"{'/'.join(subjects[:3])} {span} -> "
                              f"{src[lo - 1].strip()[:60]}")
                continue
            resolved += 1
            if hit in src[lo - 1]:
                exact += 1
    return misses, resolved, total, exact


def check_l3(doc_lines: list[str],
             patterns: list[tuple[str, "re.Pattern[str]"]]
             ) -> dict[str, list[tuple[int, int]]]:
    """L3. label -> [(doc line number, hits on that line)]. Values never leave here."""
    hits: dict[str, list[tuple[int, int]]] = {label: [] for label, _ in patterns}
    for label, rx in patterns:
        for lno, line in enumerate(doc_lines, 1):
            c = len(rx.findall(line))
            if c:
                hits[label].append((lno, c))
    return hits


# --- parity helpers ---------------------------------------------------------------


def _logical_lines(doc_lines: list[str]) -> list[tuple[int, str]]:
    """Backslash-continued lines folded into one, tagged with the physical line the
    logical line STARTS on.

    The doc writes its engine command the way an operator would paste it -- one flag
    per line, backslash-continued -- so a per-physical-line regex reads the opening
    quote, never finds the closing one, and matches nothing. That is exactly how
    L6-L9 first came back 0/0 UNMEASURED against a doc that plainly carries a
    command: the extractor could not see its own subject, and a denominator of zero
    is the failure this file exists to prevent, not a pass.
    """
    out: list[tuple[int, str]] = []
    i, n = 0, len(doc_lines)
    while i < n:
        start = i + 1
        parts = [doc_lines[i]]
        while parts[-1].rstrip().endswith("\\") and i + 1 < n:
            parts[-1] = parts[-1].rstrip()[:-1]
            i += 1
            parts.append(doc_lines[i])
        out.append((start, parts[0] if len(parts) == 1
                    else " ".join(x.strip() for x in parts)))
        i += 1
    return out


def documented_commands(doc_lines: list[str]) -> list[tuple[int, str]]:
    """Every `export FS_ENGINE_LAUNCH_CMD="..."` assignment: (1-based line, value).

    The denominator for L6-L9 is the length of this list: an operator doc with no
    engine command gives an operator nothing to paste, and a parity gate over zero
    commands is `all([])` in a trench coat.
    """
    out: list[tuple[int, str]] = []
    for lno, line in _logical_lines(doc_lines):
        m = CMD_EXPORT_RE.match(line)
        if m:
            out.append((lno, m.group(1)))
    return out


def script_refs(value: str) -> list[str]:
    """Basenames of every .py reference inside a documented command value."""
    return [r.rsplit("/", 1)[-1] for r in SCRIPT_REF_RE.findall(value)]


_SUPPLIED_CACHE: "set[str] | None" = None
_PLANE_CACHE: "str | None" = None


def plane_sources() -> str:
    """Launcher and backend text, concatenated, for substring oracles. Unreadable
    inputs contribute nothing, which can only make a rule stricter."""
    global _PLANE_CACHE
    if _PLANE_CACHE is None:
        parts = []
        for f in (LAUNCHER, BACKEND):
            try:
                parts.append(f.read_text(encoding="utf-8", errors="replace"))
            except OSError:
                pass
        _PLANE_CACHE = "\n".join(parts)
    return _PLANE_CACHE


def boundary_supplied() -> set[str]:
    """Env names the LAUNCH PLANE supplies to the trainer itself, so the operator need
    not: assigned somewhere in the launcher AND carried across the container boundary
    on the backend's allowlist.

    Both conditions are load-bearing. Assignment alone would credit a launcher-local
    working variable that never reaches the trainer; allowlist membership alone would
    credit an operator-supplied knob such as the engine command, which the launcher
    only ever reads. The intersection is exactly "the plane mints this AND delivers
    it", which is the property that discharges the operator's obligation.

    Measured, never listed. Without this arm L8 reports three permanent misses against
    knobs the doc correctly explains are launcher-minted, and a gate carrying rows that
    are known-false is a gate operators learn to scroll past. Unreadable inputs return
    the empty set, which credits nothing and can only make L8 stricter -- the safe
    direction for a rule whose job is to find omissions.
    """
    global _SUPPLIED_CACHE
    if _SUPPLIED_CACHE is None:
        try:
            lsrc = LAUNCHER.read_text(encoding="utf-8", errors="replace")
            bsrc = BACKEND.read_text(encoding="utf-8", errors="replace")
        except OSError:
            _SUPPLIED_CACHE = set()
            return _SUPPLIED_CACHE
        assigned = set(re.findall(r"^\s*(?:export\s+)?([A-Z_][A-Z0-9_]*)=",
                                  lsrc, re.M))
        # Close on a line that is only `)`, not the first `)` anywhere: the array's
        # entries carry explanatory trailing comments, and a comment containing a
        # parenthesis would truncate the census silently -- a short denominator that
        # still looks like a census. Entries likewise allow a trailing comment.
        m = re.search(r"FS_ENV_ALLOWLIST=\(\n(.*?)^\s*\)\s*$", bsrc, re.S | re.M)
        allow = (set(re.findall(r"^\s*([A-Z_][A-Z0-9_]*)\s*(?:#.*)?$",
                                m.group(1), re.M))
                 if m else set())
        _SUPPLIED_CACHE = assigned & allow
    return _SUPPLIED_CACHE


def knob_flag(knob: str) -> str:
    """The flag spelling of a sourced knob: `_` -> `-`, prefixed `--`. Derived, never listed."""
    return "--" + knob.replace("_", "-")


def _ast_const(node: ast.AST) -> "object":
    """The literal value of a constant node, tolerating ast.Str for older readers."""
    if isinstance(node, ast.Constant):
        return node.value
    str_t = getattr(ast, "Str", None)
    if str_t is not None and isinstance(node, str_t):
        return node.s
    return None


def trainer_flags(src: str) -> set[str]:
    """Every `--flag` the trainer's parser declares, from its `add_argument` calls.

    Walked out of the AST on every run: the day the trainer grows a nineteenth flag,
    this set grows with it instead of a human updating a list in a gate.
    """
    tree = ast.parse(src)
    flags: set[str] = set()
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "add_argument"):
            continue
        for arg in node.args:
            v = _ast_const(arg)
            if isinstance(v, str) and v.startswith("--"):
                flags.add(v)
    return flags


def required_knobs(src: str) -> tuple[list[tuple[str, str | None]],
                                      list[tuple[str, str | None, str]]]:
    """(unconditional, conditional) required knobs, from the trainer's sourcing calls.

    The sourcing helper binds (knob name, flag value, env name, parser) triples — we
    key on its bare-Name calls rather than hard-coding the knob list, so a new
    `_sourced(...)` line is enrolled in L8/L9 by being WRITTEN, not by being catalogued
    here. Calls with `required=False` explicitly are not required knobs and are
    excluded. A call with no enclosing `If` inside its own function is UNCONDITIONAL;
    one under an `If` is CONDITIONAL, recorded with the OUTERMOST enclosing branch
    (`body` vs `orelse`) and the source text of that `If`'s test — L8 needs the test
    to find the selector, L9 needs the branch to know what is mutually exclusive.
    The conditional triple packs branch and test as "branch|test" (branch never
    contains a pipe; partition with maxsplit 1 reads it back exactly).
    """
    tree = ast.parse(src)
    uncond: list[tuple[str, str | None]] = []
    cond: list[tuple[str, str | None, str]] = []
    seen_u: set[str] = set()
    seen_c: set[str] = set()

    def record(call: ast.Call, ifs: "tuple[tuple[ast.If, str], ...]") -> None:
        if not call.args:
            return
        knob = _ast_const(call.args[0])
        if not isinstance(knob, str):
            return  # not a shape this gate can read; better absent than invented
        if any(kw.arg == "required" and isinstance(kw.value, ast.Constant)
               and kw.value.value is False for kw in call.keywords):
            return
        env = _ast_const(call.args[3]) if len(call.args) > 3 else None
        if not isinstance(env, str):
            env = None  # a non-literal env expression cannot be resolved to a name
        if not ifs:
            if knob not in seen_u:
                seen_u.add(knob)
                uncond.append((knob, env))
        else:
            if knob not in seen_c:
                seen_c.add(knob)
                top_if, branch = ifs[0]
                test_src = ast.get_source_segment(src, top_if.test) or ""
                cond.append((knob, env, branch + "|" + test_src))

    def visit(node: ast.AST, ifs: "tuple[tuple[ast.If, str], ...]") -> None:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            # Branch ancestry is scoped to the enclosing FUNCTION (the spec's words):
            # entering a new function resets it.
            for child in ast.iter_child_nodes(node):
                visit(child, ())
            return
        if isinstance(node, ast.If):
            for child in node.body:
                visit(child, ifs + ((node, "body"),))
            for child in node.orelse:
                visit(child, ifs + ((node, "orelse"),))
            return
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id == "_sourced"):
            record(node, ifs)
        for child in ast.iter_child_nodes(node):
            visit(child, ifs)

    visit(tree, ())
    return uncond, cond


def published_basenames() -> set[str]:
    """Basenames of every shipped path in the publish set (blank/# lines dropped)."""
    out: set[str] = set()
    for raw in PUBLISH_SET.read_text("utf-8").splitlines():
        s = raw.strip()
        if not s or s.startswith("#"):
            continue
        out.add(s.rsplit("/", 1)[-1])
    return out


def parse_selector(test_src: str,
                   tflags: set[str]) -> "tuple[str, list[str]] | None":
    """Read a guard test of the form `<knob> == <string constant>` (Eq only).

    Returns (selector knob name, compared constants) or None for any shape this gate
    refuses to interpret — BoolOp, NotEq, calls, non-constant comparators. None is
    UNMEASURED at the call sites, never a skip: "we could not read the branch
    condition" is precisely the state in which silently enforcing nothing would be
    `all([])` again. When the compared expression is an attribute fetch
    (`x.y == "..."`), the knob is whichever of the base or the attribute corresponds
    to a flag the trainer actually declares — so both `args.dataset_mode` and
    `dataset_mode.value` resolve to the right knob without either spelling being
    named here.
    """
    try:
        node = ast.parse(test_src, mode="eval").body
    except SyntaxError:
        return None
    if not (isinstance(node, ast.Compare) and len(node.ops) == 1
            and isinstance(node.ops[0], ast.Eq)):
        return None
    left = node.left
    if isinstance(left, ast.Name):
        knob = left.id
    elif isinstance(left, ast.Attribute) and isinstance(left.value, ast.Name):
        base, attr = left.value.id, left.attr
        if knob_flag(base) in tflags:
            knob = base
        elif knob_flag(attr) in tflags:
            knob = attr
        else:
            knob = base if attr == "value" else attr
    else:
        return None
    consts: list[str] = []
    for comp in node.comparators:
        c = _ast_const(comp)
        if not isinstance(c, str):
            return None
        consts.append(c)
    return knob, consts


def command_flag_value(commands: list[tuple[int, str]], flag: str) -> "str | None":
    """The token a documented command assigns to `flag` (--flag v or --flag=v)."""
    rx = re.compile(r"(?<![\w-])" + re.escape(flag) + r"(?:\s+(\S+)|\s*=\s*(\S+))?")
    for _lno, value in commands:
        m = rx.search(value)
        if m:
            v = m.group(1) or m.group(2)
            if v is None:
                return None  # flag present, valueless: a selector cannot read it
            return v.rstrip("\"'")
    return None


def is_literal_value(v: str) -> bool:
    """True only for a value the gate may compare against a constant.

    A placeholder (<...>), a shell expansion ($VAR, \\$VAR), a quote, or another flag
    masquerading as the value all mean the selector WILL NOT RESOLVE, and an
    unresolvable selector is UNMEASURED -- the gate declines to guess which branch a
    runtime shell would have landed on, because guessing IS the drift being gated.
    """
    return bool(v) and not v.startswith("-") and not re.search(r"[$<>\\'\"`]", v)


def resolve_branches(commands: list[tuple[int, str]],
                     cond: list[tuple[str, str | None, str]],
                     tflags: set[str]
                     ) -> "list[tuple[list[tuple[str, str | None, str]], str, str | None, str | None, str | None]]":
    """Group conditional knobs by their guarding If and resolve each selector.

    One entry per distinct guard test: (knobs in group, test source, selector knob or
    None if the test is unreadable, documented value or None, selected branch or
    None). "body" is selected when the documented literal equals a compared constant,
    "orelse" otherwise — the partition only ever produces Eq guards, and any other
    shape already returned selector=None upstream.
    """
    grouped: dict[str, list[tuple[str, str | None, str]]] = {}
    order: list[str] = []
    for knob, env, packed in cond:
        branch, _, test_src = packed.partition("|")
        if test_src not in grouped:
            grouped[test_src] = []
            order.append(test_src)
        grouped[test_src].append((knob, env, branch))
    out = []
    for test_src in order:
        knobs_g = grouped[test_src]
        sel = parse_selector(test_src, tflags)
        if sel is None:
            out.append((knobs_g, test_src, None, None, None))
            continue
        sel_knob, consts = sel
        val = command_flag_value(commands, knob_flag(sel_knob))
        if val is None or not is_literal_value(val):
            out.append((knobs_g, test_src, sel_knob, val, None))
            continue
        out.append((knobs_g, test_src, sel_knob, val,
                    "body" if val in consts else "orelse"))
    return out


def doc_has_export(doc_lines: list[str], env: str) -> bool:
    """The doc contains an `export <ENV>=` line (arm (b) of knob satisfaction)."""
    rx = re.compile(r"^\s*export\s+" + re.escape(env) + r"\s*=")
    return any(rx.match(l) for l in doc_lines)


def trainer_exit_codes(src: str) -> "tuple[set[int], list[str]]":
    """The integer codes an operator can actually receive from the trainer.

    Returns (codes, unresolved) -- the second list is the point. A Constant-only walk
    of main()'s returns derived {2, 3} on the shipped trainer and printed
    `PASS L10 2/2`: a true sentence over half the contract, because main dispatches
    with `return _run(config)` and the helper ends `return 0 if verdict == "MEASURED"
    else 3`. Codes 0 and 1 are the ones an operator sees on nearly every run, and the
    rule certified a table without ever asking about them. A denominator that silently
    drops what it cannot parse is the same vacuous pass as `all([])`, one level down --
    so anything this resolver cannot reduce to an integer is NAMED and makes L10
    UNMEASURED rather than shrinking the census.

    Resolved shapes: an integer constant; a bare `return` or `return None` (CPython
    exits 0 on `SystemExit(None)`); `return A if c else B` on both arms; and one
    call-depth of `return helper(...)` into a function defined in the same file, since
    that is how a main() that is only a dispatcher spells its codes. Returns belonging
    to a function nested INSIDE the one being read are not its returns and are skipped.

    PLUS 2 whenever the trainer uses argparse at all: argparse's documented behaviour
    is to exit 2 on unrecognised argv, and an unrecognised flag is exactly what a
    drifting worked example produces, so a trainer that delegates parsing hands the
    operator a code no `return` statement will ever reveal. The +2 is CONDITIONAL on
    argparse being used, so a trainer that hand-rolls its parser is not charged for a
    library it does not link.
    """
    tree = ast.parse(src)
    funcs: dict[str, ast.AST] = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            funcs.setdefault(node.name, node)

    codes: set[int] = set()
    unresolved: list[str] = []

    def own_returns(fn) -> list[ast.Return]:
        out: list[ast.Return] = []

        def rec(n):
            for ch in ast.iter_child_nodes(n):
                if isinstance(ch, (ast.FunctionDef, ast.AsyncFunctionDef,
                                   ast.Lambda, ast.ClassDef)):
                    continue
                if isinstance(ch, ast.Return):
                    out.append(ch)
                rec(ch)

        rec(fn)
        return out

    def resolve(value, where: str, seen: "frozenset[str]", depth: int) -> None:
        if value is None:
            codes.add(0)
            return
        if isinstance(value, ast.Constant):
            if value.value is None:
                codes.add(0)
            elif isinstance(value.value, bool):
                unresolved.append(f"{where} returns a bool")
            elif isinstance(value.value, int):
                codes.add(value.value)
            else:
                unresolved.append(f"{where} returns a non-integer constant")
            return
        if isinstance(value, ast.IfExp):
            resolve(value.body, where, seen, depth)
            resolve(value.orelse, where, seen, depth)
            return
        if (isinstance(value, ast.Call) and isinstance(value.func, ast.Name)
                and value.func.id in funcs and value.func.id not in seen
                and depth > 0):
            callee = value.func.id
            rets = own_returns(funcs[callee])
            if not rets:
                unresolved.append(f"{where} delegates to {callee}(), which returns "
                                  f"nothing this walk can see")
                return
            for r in rets:
                resolve(r.value, f"{callee}():{r.lineno}", seen | {callee}, depth - 1)
            return
        unresolved.append(f"{where} returns a {type(value).__name__} this walk "
                          f"cannot reduce to an integer")

    main_fn = funcs.get("main")
    if main_fn is None:
        unresolved.append("the trainer declares no main(), so no code census exists")
    else:
        for r in own_returns(main_fn):
            resolve(r.value, f"main():{r.lineno}", frozenset({"main"}), 2)

    uses_argparse = any(
        isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
        and n.func.attr == "add_argument" for n in ast.walk(tree))
    if uses_argparse:
        codes.add(2)
    return codes, unresolved


FENCE_RE = re.compile(r"^\s*```")
HEADING_RE = re.compile(r"^#{1,6}\s")


def doc_sections(doc_lines: list[str]) -> list[tuple[int, int]]:
    """Heading-delimited sections as 0-based half-open [start, end) ranges.

    Fence-aware on purpose. The 60-second section is one long ```bash block whose
    shell comments (`# --- axis guards ---`) are indistinguishable from an h1 heading
    by regex alone, so a fence-blind reader shatters the document into dozens of
    spurious sections and no real one survives intact.
    """
    fence = False
    heads: list[int] = []
    for i, l in enumerate(doc_lines):
        if FENCE_RE.match(l):
            fence = not fence
            continue
        if not fence and HEADING_RE.match(l):
            heads.append(i)
    if not heads:
        return []
    bounds = heads + [len(doc_lines)]
    return [(a, b) for a, b in zip(bounds, bounds[1:])]


def contract_block(doc_lines: list[str],
                   basename: str) -> "tuple[list[str], int, int] | None":
    """The trainer-contract block: the heading-delimited section that names the
    trainer's basename AND actually tabulates codes (>=1 `**<n>**` or `` `<n>` `` span).

    A blank-line-delimited run was tried first and read UNMEASURED against a doc that
    plainly carries the table: the contract is written as several short paragraphs, so
    the basename lands in one run and the codes in another and no single run holds
    both. The heading is the unit the document itself uses to say "this is one
    contract", so it is the unit measured.

    Requiring a code literal rather than merely the word `exit` is what makes the
    choice unique -- prose elsewhere mentions exits in passing. If two sections still
    qualify, the doc has rival contracts and none of them is THE contract: return None
    and let L10/L11 declare UNMEASURED rather than silently crowning the first.
    """
    hits: list[tuple[list[str], int, int]] = []
    for a, b in doc_sections(doc_lines):
        block = doc_lines[a:b]
        if not any(basename in l for l in block):
            continue
        if not any(re.search(r"\*\*\d+\*\*|`\d+`", l) for l in block):
            continue
        hits.append((block, a, b))
    return hits[0] if len(hits) == 1 else None


def code_literal_re(n: int) -> "re.Pattern[str]":
    """The doc's two accepted spellings of an exit code: bold or backticked."""
    return re.compile(r"\*\*" + str(n) + r"\*\*|`" + str(n) + r"`")


# --- parity checks -----------------------------------------------------------------


def check_l6(commands: list[tuple[int, str]], published: set[str]
             ) -> tuple[list[tuple[int, str]], int, list[int]]:
    """L6. (unpublished (line, basename)s, total refs, lines of commands with no .py).

    A command referencing no .py hands the operator no program to run; that is a
    finding in its own right, reported by line number, not silently absorbed into a
    zero denominator.
    """
    missing: list[tuple[int, str]] = []
    unscripted: list[int] = []
    total = 0
    for lno, value in commands:
        refs = script_refs(value)
        total += len(refs)
        if not refs:
            unscripted.append(lno)
            continue
        for b in refs:
            if b not in published:
                missing.append((lno, b))
    return missing, total, unscripted


def check_l7(commands: list[tuple[int, str]], tflags: set[str]) -> tuple[list[str], int]:
    """L7. (unknown flags, distinct flags documented).

    This is the rule that would have caught --model/--ckpt-dir/--steps: three flag
    spellings, zero of which existed in a parser declaring seventeen.
    """
    used: set[str] = set()
    for _l, v in commands:
        used.update(FLAG_TOKEN_RE.findall(v))
    return sorted(used - tflags), len(used)


def check_l8(commands: list[tuple[int, str]], doc_lines: list[str],
             uncond: list[tuple[str, str | None]],
             cond: list[tuple[str, str | None, str]],
             tflags: set[str]
             ) -> tuple[list[tuple[str, "str | None"]],
                        list[tuple[str, "str | None"]],
                        list[tuple[tuple[str, ...], str]],
                        list[tuple[str, str]]]:
    """L8. (unconditional misses, conditional misses, unresolved selector groups,
    resolved selectors as (flag, branch)).

    A knob is SATISFIED by (a) its derived flag spelling appearing in a documented
    command, (b) an `export <ENV>=` line for its declared env fallback, or (c) the
    plane supplying that env itself (see `boundary_supplied`). Conditional
    knobs bind only on the branch the documented selector value selects.
    """
    all_flags = {f for _l, v in commands for f in FLAG_TOKEN_RE.findall(v)}

    def satisfied(knob: str, env: "str | None") -> bool:
        if knob_flag(knob) in all_flags:
            return True
        if env is not None and doc_has_export(doc_lines, env):
            return True
        return env is not None and env in boundary_supplied()

    u_missing = [(k, e) for k, e in uncond if not satisfied(k, e)]
    c_missing: list[tuple[str, "str | None"]] = []
    unresolved: list[tuple[tuple[str, ...], str]] = []
    selectors: list[tuple[str, str]] = []
    for knobs_g, _test, sel, _val, selected in resolve_branches(commands, cond, tflags):
        names = tuple(k for k, _e, _b in knobs_g)
        if sel is None:
            unresolved.append((names, "its guarding test is not a readable "
                                      "constant comparison"))
            continue
        if selected is None:
            unresolved.append((names, f"selector `{knob_flag(sel)}` has no literal "
                                      f"value in the documented command — absent, a "
                                      f"placeholder, or a shell expansion the gate "
                                      f"will not guess through"))
            continue
        selectors.append((knob_flag(sel), selected))
        for knob, env, branch in knobs_g:
            if branch == selected and not satisfied(knob, env):
                c_missing.append((knob, env))
    return u_missing, c_missing, unresolved, selectors


def check_l9(commands: list[tuple[int, str]],
             cond: list[tuple[str, str | None, str]],
             tflags: set[str]
             ) -> tuple[list[tuple[str, str, str]], list[tuple[tuple[str, ...], str]], int]:
    """L9. ((flag, its branch, selected branch) violations, unresolved groups, total).

    The trainer refuses knobs from the branch the selector did not choose, so a
    documented command carrying one is a guaranteed refusal discovered after
    allocation. WHICH flags conflict is read out of the same AST partition as L8 —
    hard-coding the pairs here would drift the day the trainer grows a third mode.
    """
    all_flags = {f for _l, v in commands for f in FLAG_TOKEN_RE.findall(v)}
    violations: list[tuple[str, str, str]] = []
    unresolved: list[tuple[tuple[str, ...], str]] = []
    for knobs_g, _test, sel, _val, selected in resolve_branches(commands, cond, tflags):
        names = tuple(k for k, _e, _b in knobs_g)
        if sel is None:
            unresolved.append((names, "its guarding test is not a readable "
                                      "constant comparison"))
            continue
        if selected is None:
            unresolved.append((names, f"selector `{knob_flag(sel)}` will not resolve "
                                      f"on the documented command"))
            continue
        for knob, _env, branch in knobs_g:
            if branch != selected and knob_flag(knob) in all_flags:
                violations.append((knob_flag(knob), branch, selected))
    return violations, unresolved, len(cond)


def check_l10(doc_lines: list[str], codes: set[int],
              basename: str) -> tuple[bool, list[int]]:
    """L10. (block found, codes the contract block fails to document).

    The measured defect: main() can hand back a code the doc's exit table omits, and
    the omitted code was ALSO argparse's bad-argv code — i.e. the exact code the
    drifting worked example produces was the only one undocumented.
    """
    blk = contract_block(doc_lines, basename)
    if blk is None:
        return False, []
    text = "\n".join(blk[0])
    return True, [c for c in sorted(codes) if not code_literal_re(c).search(text)]


def check_l11(doc_lines: list[str], src: str,
              basename: str) -> tuple[bool, list[str], int]:
    """L11. (block found, absent markers, token count).

    Every backticked SHOUTED token in the contract block is a token the doc claims
    the plane emits or honours. FSLEG/FSSUMMARY were claimed and occur zero times
    anywhere; RUN_SUMMARY_JSON/PHASE_JSON are printed and were unmentioned.

    The oracle is the whole plane -- trainer, launcher, backend -- not the trainer
    alone. The contract section deliberately tabulates the launcher's codes beside the
    trainer's, so it also names launcher-side tokens (`FATAL`, knob names read there),
    and scoring those against the trainer source alone reported five absences that are
    all present in the plane the doc actually describes. The defect worth catching is
    "the doc names a token nothing in the plane carries", and that is what this tests;
    attributing each token to one component would need a claim the section does not
    make.
    """
    blk = contract_block(doc_lines, basename)
    if blk is None:
        return False, [], 0
    seen: dict[str, None] = {}
    for line in blk[0]:
        for span in SPAN_RE.findall(line):
            for tok in MARKER_RE.findall(span):
                seen.setdefault(tok, None)
    toks = list(seen)
    plane = src + plane_sources()
    return True, [t for t in toks if t not in plane], len(toks)


def inject_into_command(doc_lines: list[str], start_lno: int,
                        token: str) -> "list[str] | None":
    """Splice a token into the END of a documented command's value, on a copy.

    The command is backslash-continued over several physical lines, so reaching for
    the last quote on the STARTING line finds the OPENING quote and splices the token
    outside the value. That does not make the command wrong, it makes it unparseable:
    the extractor then finds no command at all, the rule has nothing to object to, and
    the drill reports a healthy detector as dead. Land on the physical line where the
    logical line actually ends. None -> unplantable.
    """
    i, n = start_lno - 1, len(doc_lines)
    while i < n - 1 and doc_lines[i].rstrip().endswith("\\"):
        i += 1
    q = doc_lines[i].rfind('"')
    if q <= 0:
        return None
    out = list(doc_lines)
    out[i] = doc_lines[i][:q] + " " + token + doc_lines[i][q:]
    return out


def remove_flag_token(line: str, flag: str) -> str:
    """Excise one exact flag spelling from a line, boundary-safe.

    Lookaround both ways so `--model-path` cannot be clipped out of `--model-paths`
    and a leading word character or third hyphen cannot fake a token boundary.
    """
    rx = re.compile(r"(?<![\w-])" + re.escape(flag) + r"(?![\w-])")
    return rx.sub("", line)


def controls(doc_text: str, doc_lines: list[str], lau_lines: list[str],
             knobs: list[str], patterns: list[tuple[str, "re.Pattern[str]"]],
             trainer_src: str, published: set[str]
             ) -> tuple[bool, bool]:
    """L5. Returns (all_green, any_unplantable). Runs every time, not behind a flag.

    A gate that watches nine rules with three drills is three detectors and six hopes;
    each new rule ships a MUST_FIRE planted on a COPY of the input and observed going
    red, and an unplantable drill is UNMEASURED -- a detector never seen firing is not
    a detector. The parity oracles (flags, knobs, codes) are re-derived HERE from the
    trainer source rather than received as arguments, so a stale caller cannot hand the
    controls an oracle that disagrees with main()'s.
    """
    green = True
    unplantable = False

    tflags = trainer_flags(trainer_src)
    uncond_k, cond_k = required_knobs(trainer_src)
    # The drills exercise the codes the resolver COULD reduce; main() separately
    # refuses to certify L10 at all when any return shape stayed unresolved, so a
    # partial census can never reach a PASS by way of a drill that fired on the part
    # of it that parsed.
    codes, _codes_unresolved = trainer_exit_codes(trainer_src)
    basename = TRAINER.name
    commands0 = documented_commands(doc_lines)

    # Snapshot for the MUST_PASS at the bottom. The claim being made there is
    # NON-DESTRUCTIVENESS -- that the drills mutated only copies -- so the thing to
    # compare against is the rule output as it stood BEFORE any drill ran, not a
    # re-statement of one of the rules. Cheap, and it is the only way the claim
    # "the drill machinery added no violation" is actually backed by evidence.
    before = (check_l1(doc_text, knobs),
              check_l2(doc_lines, lau_lines),
              check_l3(doc_lines, patterns),
              check_l6(commands0, published),
              check_l7(commands0, tflags),
              check_l8(commands0, doc_lines, uncond_k, cond_k, tflags),
              check_l9(commands0, cond_k, tflags),
              check_l10(doc_lines, codes, basename),
              check_l11(doc_lines, trainer_src, basename))

    # MUST_FIRE L1: delete EVERY mention of one covered knob; L1 must go red.
    covered = [k for k in knobs if k in doc_text]
    if not covered:
        print("  FAIL L5/L1 MUST_FIRE cannot be planted — no covered knob exists to "
              "delete. An unplantable drill proves nothing. UNMEASURED.",
              file=sys.stderr)
        green = False
        unplantable = True
    else:
        victim = covered[0]
        poisoned = doc_text.replace(victim, "")
        if victim in check_l1(poisoned, knobs):
            print("  PASS L5/L1 MUST_FIRE: covered knob deleted from a doc copy, "
                  "L1 observed going red")
        else:
            print("  FAIL L5/L1 MUST_FIRE did not fire — a knob the launcher enforces "
                  "vanished from the doc and L1 stayed silent", file=sys.stderr)
            green = False

    # MUST_FIRE L2: renumber one resolving citation to a line that does not mention
    # its knob; L2 must go red.
    # NOT a precondition. The removed line here was `assert not probe_misses` -- it
    # required the live doc to be already clean before the drill would run, which is
    # backwards: this gate exists to be pointed at a doc that may be dirty, and on the
    # very first real run (19 of 19 citations stale) the assert took the process down
    # with a traceback and rc=1, outside the declared 0/5/95/96 contract. Worse, an
    # `assert` is erased by `python -O`, so under optimisation the unplantable case
    # would have gone silently green -- a control that disappears is the vacuous-truth
    # shape the whole doctrine is aimed at. The loop below already searches for a
    # resolving citation and reports UNMEASURED when none exists, which is the correct
    # handling; the count is printed so the drill's own denominator is visible.
    probe_misses, probe_resolved, probe_total, probe_exact = check_l2(doc_lines,
                                                                      lau_lines)
    print(f"  ---- drill population: {probe_resolved}/{probe_total} citation(s) "
          f"currently resolve ({probe_exact} on the first line of their own range) "
          f"and are therefore corruptible by the L2 drill")
    # Same view check_l2 gets: the poisoned `lau_lines` for L:, the real files for the
    # rest, so the drill and the rule cannot disagree about what a citation points at.
    drill_sources = dict(cite_sources())
    drill_sources["L"] = lau_lines
    planted_l2 = False
    for dno, dline in enumerate(doc_lines):
        cites = CITE_RE.findall(dline)
        if not cites:
            continue
        blo, bhi = block_bounds(doc_lines, dno)
        subjects = row_subjects("\n".join(doc_lines[blo:bhi + 1]))
        if not subjects:
            continue
        for m in CITE_RE.finditer(dline):
            # #186: the drill follows the notation. Retargeting a B: citation against
            # the LAUNCHER's line count would pick a number that is out of range in the
            # backend, and L2 would go red for being out of range rather than for
            # pointing somewhere wrong -- a control satisfiable by the wrong obstacle
            # demonstrates neither.
            pfx = m.group(1)
            src = drill_sources.get(pfx)
            if src is None:
                continue
            lo = int(m.group(2))
            hi = int(m.group(3)) if m.group(3) else lo
            if not (1 <= lo <= hi <= len(src)):
                continue
            window = src[lo - 1:hi]
            if not any(s in w for s in subjects for w in window):
                continue  # not a resolving citation; would not test the detector
            # Retarget to a line no subject on this row mentions. Must clear EVERY
            # subject, not just one: the resolver accepts any of them, so a target that
            # only defeats the first would leave the citation resolving and the drill
            # would report a dead detector as alive.
            target = next((i for i in range(1, len(src) + 1)
                           if not any(s in src[i - 1] for s in subjects)), None)
            if target is None:
                continue  # every line of that source names something on this row
            # m.group(0), not a reconstruction. The doc writes ranges with an en-dash;
            # rebuilding the citation as f"L:{lo}-{hi}" produces a string that is not in
            # the line, so .replace() would no-op and the "poisoned" copy would be the
            # clean copy -- a MUST_FIRE that silently tests nothing. Use the text the
            # regex actually matched.
            poisoned_lines = list(doc_lines)
            poisoned_lines[dno] = dline.replace(m.group(0), f"{pfx}:{target}", 1)
            misses, _, _, _ = check_l2(poisoned_lines, lau_lines)
            if any(f" {pfx}:{target} ->" in m for m in misses):
                print("  PASS L5/L2 MUST_FIRE: resolving citation renumbered on a doc "
                      "copy, L2 observed going red")
            else:
                print("  FAIL L5/L2 MUST_FIRE did not fire — a citation was pointed at "
                      "a line that never mentions its knob and L2 stayed silent",
                      file=sys.stderr)
                green = False
            planted_l2 = True
            break
        if planted_l2:
            break
    if not planted_l2:
        print("  FAIL L5/L2 MUST_FIRE cannot be planted — no resolving citation with a "
              "knob-free target line exists. An unplantable drill proves nothing. "
              "UNMEASURED.", file=sys.stderr)
        green = False
        unplantable = True

    # MUST_FIRE L3: plant the partition literal in a doc copy; L3 must go red.
    # The planted value is used, never printed.
    part = os.environ["FS_PARTITION_LITERAL"]  # load_patterns() already gated this
    poisoned_lines = doc_lines + ["", part]
    hits = check_l3(poisoned_lines, patterns)
    if sum(c for _, c in hits["FS_PARTITION_LITERAL"]) >= 1:
        print("  PASS L5/L3 MUST_FIRE: partition literal planted in a doc copy, "
              "L3 observed going red")
    else:
        print("  FAIL L5/L3 MUST_FIRE did not fire — the partition literal sits in a "
              "doc copy and L3 stayed silent", file=sys.stderr)
        green = False

    # MUST_FIRE L6: rewrite one documented script basename to a name the publish set
    # never shipped (still ending .py, or the poisoned copy would not be read as a
    # script reference at all and the drill would test nothing); L6 must go red.
    src_cmd = next(((lno, v) for lno, v in commands0 if script_refs(v)), None)
    if src_cmd is None:
        print("  FAIL L5/L6 MUST_FIRE cannot be planted — no documented engine command "
              "references any .py program, so there is no basename to rewrite. An "
              "unplantable drill proves nothing. UNMEASURED.", file=sys.stderr)
        green = False
        unplantable = True
    else:
        lno6, _v6 = src_cmd
        victim6 = script_refs(_v6)[0]
        stem6 = victim6[:-3] if victim6.endswith(".py") else victim6
        cand6 = next((c for c in (stem6 + "_drill.py", "zz_gate_" + victim6,
                                  stem6 + "_x.py")
                      if c != victim6 and c not in published), None)
        if cand6 is None:
            print("  FAIL L5/L6 MUST_FIRE cannot be planted — every derived "
                  "replacement basename is accidentally in the publish set. "
                  "UNMEASURED.", file=sys.stderr)
            green = False
            unplantable = True
        else:
            poisoned6 = list(doc_lines)
            poisoned6[lno6 - 1] = poisoned6[lno6 - 1].replace(victim6, cand6, 1)
            miss_p, _t_p, _z_p = check_l6(documented_commands(poisoned6), published)
            if any(b == cand6 for _l, b in miss_p):
                print("  PASS L5/L6 MUST_FIRE: documented script renamed to an "
                      "unpublished basename on a doc copy, L6 observed going red")
            else:
                print("  FAIL L5/L6 MUST_FIRE did not fire — a command pointing at a "
                      "file nothing ships stayed green under L6", file=sys.stderr)
                green = False

    # MUST_FIRE L7: append a flag the trainer never declares to a documented command;
    # L7 must go red. The planted spelling is derived to collide with nothing.
    if not commands0:
        print("  FAIL L5/L7 MUST_FIRE cannot be planted — the doc assigns no engine "
              "command to append a bogus flag to. UNMEASURED.", file=sys.stderr)
        green = False
        unplantable = True
    else:
        i7 = 0
        while f"--zz-drill-{i7}" in tflags:
            i7 += 1
        cand7 = f"--zz-drill-{i7}"
        poisoned7 = inject_into_command(doc_lines, commands0[0][0], cand7)
        if poisoned7 is None:
            print("  FAIL L5/L7 MUST_FIRE cannot be planted — the documented command "
                  "line carries no closing quote to inject before. UNMEASURED.",
                  file=sys.stderr)
            green = False
            unplantable = True
        else:
            unknown7, _tot7 = check_l7(documented_commands(poisoned7), tflags)
            if cand7 in unknown7:
                print("  PASS L5/L7 MUST_FIRE: undeclared flag appended to a "
                      "documented command on a doc copy, L7 observed going red")
            else:
                print(f"  FAIL L5/L7 MUST_FIRE did not fire — `{cand7}` sits in a "
                      f"documented command and L7 did not call it unknown",
                      file=sys.stderr)
                green = False

    # MUST_FIRE L8: withdraw one satisfied unconditional knob; L8 must go red.
    # TWO-OBSTACLE ATTRIBUTION: satisfaction is disjunctive — (a) the flag in the
    # documented command OR (b) an export of the knob's declared env fallback — so
    # deleting only the flag of a knob whose export line survives leaves the knob
    # satisfied, and the drill would report a dead detector as alive. Both routes the
    # victim actually uses are cut on the copy, otherwise the drill proves nothing.
    if not commands0 or not uncond_k:
        print("  FAIL L5/L8 MUST_FIRE cannot be planted — there is no documented "
              "command, or the trainer exposes no unconditional required knob to "
              "withdraw. UNMEASURED.", file=sys.stderr)
        green = False
        unplantable = True
    else:
        all_flags0 = {f for _l, v in commands0 for f in FLAG_TOKEN_RE.findall(v)}
        # Arm-(c) knobs are excluded from the victim pool: the plane supplies those
        # envs itself, that route lives in the launcher rather than in the doc copy,
        # so cutting the flag would leave the knob satisfied and the drill would read
        # a live detector as dead. Same two-obstacle rule as (a)/(b), third obstacle.
        _supplied = boundary_supplied()
        sat8 = [(k, e) for k, e in uncond_k
                if (e is None or e not in _supplied)
                and (knob_flag(k) in all_flags0 or (e is not None
                                                    and doc_has_export(doc_lines, e)))]
        if not sat8:
            print("  FAIL L5/L8 MUST_FIRE cannot be planted — every unconditional "
                  "required knob is ALREADY unsatisfied by the live doc, so none can "
                  "be withdrawn; L8 is red on its own evidence and its silence is not "
                  "what needs drilling in that state, but an unplantable drill proves "
                  "nothing. UNMEASURED.", file=sys.stderr)
            green = False
            unplantable = True
        else:
            vk, ve = sat8[0]
            exp_rx = (re.compile(r"^\s*export\s+" + re.escape(ve) + r"\s*=")
                      if ve is not None else None)
            poisoned8 = [remove_flag_token(l, knob_flag(vk)) for l in doc_lines
                         if not (exp_rx is not None and exp_rx.match(l))]
            um8_p, _cm, _ur, _ss = check_l8(documented_commands(poisoned8),
                                            poisoned8, uncond_k, cond_k, tflags)
            if any(k == vk for k, _e in um8_p):
                print("  PASS L5/L8 MUST_FIRE: satisfied unconditional knob withdrawn "
                      "(flag and, where declared, its export line) on a doc copy, "
                      "L8 observed going red")
            else:
                print(f"  FAIL L5/L8 MUST_FIRE did not fire — `{vk}` lost every "
                      f"documented route on a doc copy and L8 stayed silent",
                      file=sys.stderr)
                green = False

    # MUST_FIRE L9: smuggle a flag of the NON-SELECTED branch into the documented
    # command; L9 must go red. If the selector will not resolve on the live doc there
    # is no "non-selected" branch to borrow from — the same UNMEASURED L9 itself
    # reports — so the drill is unplantable rather than faked against a guessed branch.
    all_flags0 = {f for _l, v in commands0 for f in FLAG_TOKEN_RE.findall(v)}
    target9 = None
    if commands0 and cond_k:
        for knobs_g, _t, sel, _v, selected in resolve_branches(commands0, cond_k,
                                                               tflags):
            if selected is None:
                continue
            for k, _e, branch in knobs_g:
                if branch != selected and knob_flag(k) not in all_flags0:
                    target9 = k
                    break
            if target9 is not None:
                break
    if not commands0 or not cond_k:
        print("  FAIL L5/L9 MUST_FIRE cannot be planted — no documented command, or "
              "the trainer defines no conditional branch to smuggle a flag from. "
              "UNMEASURED.", file=sys.stderr)
        green = False
        unplantable = True
    elif target9 is None:
        print("  FAIL L5/L9 MUST_FIRE cannot be planted — the mode selector does not "
              "resolve on the live document (or every non-selected branch flag is "
              "already present, in which case L9 is red on its own evidence). There "
              "is no certified off-branch to borrow a flag from. UNMEASURED.",
              file=sys.stderr)
        green = False
        unplantable = True
    else:
        poisoned9 = inject_into_command(doc_lines, commands0[0][0],
                                        knob_flag(target9) + " 1")
        if poisoned9 is None:
            print("  FAIL L5/L9 MUST_FIRE cannot be planted — the documented command "
                  "carries no closing quote to inject before. UNMEASURED.",
                  file=sys.stderr)
            green = False
            unplantable = True
            poisoned9 = list(doc_lines)
        viol9, _ur9, _n9 = check_l9(documented_commands(poisoned9), cond_k, tflags)
        if any(f == knob_flag(target9) for f, _b, _s in viol9):
            print("  PASS L5/L9 MUST_FIRE: non-selected branch flag injected into a "
                  "documented command on a doc copy, L9 observed going red")
        else:
            print(f"  FAIL L5/L9 MUST_FIRE did not fire — an off-branch flag "
                  f"`{knob_flag(target9)}` rode into the command and L9 stayed silent",
                  file=sys.stderr)
            green = False

    # MUST_FIRE L10: delete one documented exit code's `**<n>**` / `<n>` literal from
    # inside the contract block; L10 must go red. No block means there is nothing to
    # delete from AND L10 is already UNMEASURED — a drill against absence proves
    # nothing, so it is reported unplantable.
    blk10 = contract_block(doc_lines, basename)
    planted10 = False
    if blk10 is None or not codes:
        print("  FAIL L5/L10 MUST_FIRE cannot be planted — the doc carries no "
              "trainer-contract block (or the trainer's exit-code set is empty), so "
              "there is no documented code to delete. An unplantable drill proves "
              "nothing. UNMEASURED.", file=sys.stderr)
        green = False
        unplantable = True
    else:
        _bl, blo10, bhi10 = blk10
        for c in sorted(codes):
            rx_c = code_literal_re(c)
            li = next((i for i in range(blo10, bhi10) if rx_c.search(doc_lines[i])),
                      None)
            if li is None:
                continue  # this code is already undocumented; deleting it tests nothing
            # Every occurrence inside the block, not merely the first. The contract
            # section tabulates the launcher's codes and the trainer's together, so a
            # code such as 0 is spelled twice; deleting one spelling leaves the other
            # standing, L10 stays green, and the drill reports a failure that has
            # nothing to do with the defect it meant to plant. That is the
            # two-obstacle attribution problem: a control satisfiable by either
            # obstacle alone demonstrates neither.
            poisoned10 = list(doc_lines)
            for _i in range(blo10, bhi10):
                poisoned10[_i] = rx_c.sub("n/a", poisoned10[_i])
            _found_p, miss10 = check_l10(poisoned10, codes, basename)
            if c in miss10:
                print(f"  PASS L5/L10 MUST_FIRE: exit code {c}'s literal deleted from "
                      f"the contract block on a doc copy, L10 observed going red")
            else:
                print(f"  FAIL L5/L10 MUST_FIRE did not fire — code {c} vanished from "
                      f"the contract block and L10 kept certifying the table",
                      file=sys.stderr)
                green = False
            planted10 = True
            break
        if not planted10:
            print("  FAIL L5/L10 MUST_FIRE cannot be planted — the contract block "
                  "documents none of the derived codes, so none can be deleted; L10 "
                  "is red on its own evidence there. An unplantable drill proves "
                  "nothing. UNMEASURED.", file=sys.stderr)
            green = False
            unplantable = True

    # MUST_FIRE L11: insert a backticked SHOUTED token that does not occur in the
    # trainer source into the contract block; L11 must go red. The token is minted,
    # not quoted, so it cannot accidentally be a real marker.
    blk11 = contract_block(doc_lines, basename)
    if blk11 is None:
        print("  FAIL L5/L11 MUST_FIRE cannot be planted — the doc carries no "
              "trainer-contract block to insert a phantom marker into. UNMEASURED.",
              file=sys.stderr)
        green = False
        unplantable = True
    else:
        _bl, blo11, _bhi11 = blk11
        n11 = 1
        tok = f"ZZ_DRILL_MARKER_{n11:03d}"
        while tok in trainer_src:
            n11 += 1
            tok = f"ZZ_DRILL_MARKER_{n11:03d}"
        poisoned11 = (doc_lines[:blo11 + 1]
                      + [f"  rank 0 also prints `{tok}` per the contract."]
                      + doc_lines[blo11 + 1:])
        _f_p, absent_p, _nt_p = check_l11(poisoned11, trainer_src, basename)
        if tok in absent_p:
            print("  PASS L5/L11 MUST_FIRE: phantom output marker inserted into the "
                  "contract block on a doc copy, L11 observed going red")
        else:
            print(f"  FAIL L5/L11 MUST_FIRE did not fire — `{tok}` never occurs in "
                  f"the trainer source and L11 certified it anyway", file=sys.stderr)
            green = False

    # MUST_PASS: the drill machinery added no violation to the real pair.
    #
    # What once stood here asserted that the LIVE doc is clean -- a rule restated,
    # a property of the artifact, not of the controls -- so it proved nothing about
    # the drills while guaranteeing a crash on exactly the docs this gate is for. The
    # claim worth making is non-destructiveness: re-derive every rule from the
    # originals now that every drill has run, and require identical answers to the
    # pre-drill snapshot. If a drill had mutated a shared list in place instead of a
    # copy, this and only this catches it.
    after = (check_l1(doc_text, knobs),
             check_l2(doc_lines, lau_lines),
             check_l3(doc_lines, patterns),
             check_l6(commands0, published),
             check_l7(commands0, tflags),
             check_l8(commands0, doc_lines, uncond_k, cond_k, tflags),
             check_l9(commands0, cond_k, tflags),
             check_l10(doc_lines, codes, basename),
             check_l11(doc_lines, trainer_src, basename))
    names = ("L1", "L2", "L3", "L6", "L7", "L8", "L9", "L10", "L11")
    if after == before:
        print("  PASS L5/MUST_PASS: all nine rules re-derive identically after the "
              "drills; every mutation landed on a copy")
    else:
        differing = [n for n, a, b in zip(names, after, before) if a != b]
        print(f"  FAIL L5/MUST_PASS: {', '.join(differing)} changed across the drill "
              f"run — a control mutated the real inputs, so this gate's own verdict on "
              f"the live pair is unattributable", file=sys.stderr)
        green = False

    return green, unplantable


def main() -> int:
    try:
        lau_text = LAUNCHER.read_text("utf-8")
    except OSError as e:
        print(f"REFUSE 96 cannot read launcher {LAUNCHER}: {e}", file=sys.stderr)
        return 96
    try:
        doc_text = DOC.read_text("utf-8")
    except OSError as e:
        print(f"REFUSE 96 cannot read doc {DOC}: {e}", file=sys.stderr)
        return 96
    try:
        patterns = load_patterns()
    except Refusal as e:
        print(f"REFUSE 96 {e}", file=sys.stderr)
        return 96
    try:
        trainer_src = TRAINER.read_text("utf-8")
    except OSError as e:
        print(f"REFUSE 96 cannot read trainer {TRAINER}: {e}", file=sys.stderr)
        return 96
    try:
        ast.parse(trainer_src)
    except SyntaxError as e:
        # An unparsable trainer is not a dirty doc: NONE of L6-L11 can be measured
        # against it, and a refusal names that instead of letting the rules guess.
        print(f"REFUSE 96 trainer does not parse, so its flags, knobs, exit codes "
              f"and markers cannot be derived: {e}", file=sys.stderr)
        return 96
    try:
        published = published_basenames()
    except OSError as e:
        print(f"REFUSE 96 cannot read publish set {PUBLISH_SET}: {e}", file=sys.stderr)
        return 96

    lau_lines = lau_text.splitlines()
    doc_lines = doc_text.splitlines()
    print(f"launcher: {len(lau_lines)} lines   doc: {len(doc_lines)} lines")

    # The parity census, printed once so every denominator below is attributable.
    tflags = trainer_flags(trainer_src)
    uncond_k, cond_k = required_knobs(trainer_src)
    codes, codes_unresolved = trainer_exit_codes(trainer_src)
    commands = documented_commands(doc_lines)
    unres_note = (f" ({len(codes_unresolved)} return shape(s) UNRESOLVED)"
                  if codes_unresolved else "")
    print(f"trainer: {len(tflags)} declared flag(s); required knobs from the "
          f"sourcing calls: {len(uncond_k)} unconditional + {len(cond_k)} "
          f"conditional; integer exit codes derivable from main(): {len(codes)}"
          f"{unres_note}; publish set: {len(published)} basename(s); documented "
          f"engine command assignments: {len(commands)}\n")

    unmeasured = False
    red = False

    # L1 --------------------------------------------------------------------
    knobs, guards, slurm_skipped = enforced_knobs(lau_text)
    total1 = len(knobs)
    if total1 == 0:
        print(f"  FAIL L1 UNMEASURED 0/0 — the launcher exposes no required-no-default "
              f"guard idiom, so coverage cannot be certified ({slurm_skipped} SLURM_* "
              f"skipped). all([]) is True; this gate is not.", file=sys.stderr)
        unmeasured = True
    else:
        missing = check_l1(doc_text, knobs)
        if missing:
            print(f"  FAIL L1  {total1 - len(missing)}/{total1} enforced knobs named "
                  f"in LAUNCH.md; missing: {', '.join(missing)} "
                  f"(from {guards} guard lines, {slurm_skipped} SLURM_* skipped)",
                  file=sys.stderr)
            red = True
        else:
            print(f"  PASS L1  {total1}/{total1} required-no-default knobs named in "
                  f"LAUNCH.md (enumerated from {guards} launcher guard lines; "
                  f"{slurm_skipped} SLURM_* skipped as workload-manager supplied)")

    # L2 --------------------------------------------------------------------
    misses2, resolved2, total2, exact2 = check_l2(doc_lines, lau_lines)
    if total2 == 0:
        print("  FAIL L2 UNMEASURED 0/0 — the doc carries no <NOTATION>:<n> citation "
              "to resolve, so citation grounding cannot be certified. all([]) is True; "
              "this gate is not.", file=sys.stderr)
        unmeasured = True
    elif misses2:
        print(f"  FAIL L2  {resolved2}/{total2} citation(s) resolve against the "
              f"source their own notation names; {len(misses2)} miss(es):",
              file=sys.stderr)
        for m in misses2:
            print(f"    MISS {m}", file=sys.stderr)
        red = True
    else:
        print(f"  PASS L2  {resolved2}/{total2} citation(s) across "
              f"{len(cite_sources())} notation(s) resolve: the cited line or range, in "
              f"the file that notation indexes, names a subject the doc row names — "
              f"{exact2} of them on the range's first line (the strict reading), "
              f"{resolved2 - exact2} elsewhere inside a declared range")

    # L3 --------------------------------------------------------------------
    total3 = len(patterns)
    if total3 == 0:
        print("  FAIL L3 UNMEASURED 0/0 — zero patterns configured. all([]) is True; "
              "this gate is not.", file=sys.stderr)
        unmeasured = True
    else:
        hits3 = check_l3(doc_lines, patterns)
        n_hits = sum(c for rows in hits3.values() for _, c in rows)
        if n_hits:
            print(f"  FAIL L3  {n_hits} estate-literal hit(s) across {total3} "
                  f"patterns (values never printed):", file=sys.stderr)
            for label, _ in patterns:
                rows = hits3[label]
                if rows:
                    locs = ", ".join(f"line {n} (x{c})" for n, c in rows)
                    print(f"    {label}: {sum(c for _, c in rows)} hit(s) at {locs}",
                          file=sys.stderr)
            red = True
        else:
            print(f"  PASS L3  0 estate-literal hits; {total3}/{total3} required "
                  f"patterns loaded and scanned over {len(doc_lines)} doc lines")

    # L6-L9 all share one denominator root: the number of documented engine command
    # assignments. Zero means the operator is handed nothing to paste and the parity
    # rules have no subject -- every one of them reports UNMEASURED rather than
    # passing on an empty list.
    no_cmd_msg = ("the doc assigns no FS_ENGINE_LAUNCH_CMD at all, so there is no "
                  "documented engine command to check. all([]) is True; this gate is "
                  "not.")

    # L6 --------------------------------------------------------------------
    if not commands:
        print(f"  FAIL L6 UNMEASURED 0/0 — {no_cmd_msg}", file=sys.stderr)
        unmeasured = True
    else:
        miss6, total6, zero6 = check_l6(commands, published)
        ok6 = total6 - len(miss6)
        if miss6 or zero6:
            print(f"  FAIL L6  {ok6}/{total6} documented .py reference(s) resolve to "
                  f"the publish set:", file=sys.stderr)
            for lno, b in miss6:
                print(f"    line {lno}: `{b}` is referenced by a documented command "
                      f"and shipped by NOTHING in the publish set", file=sys.stderr)
            for lno in zero6:
                print(f"    line {lno}: the documented command references no .py "
                      f"program at all — the operator is given nothing to run",
                      file=sys.stderr)
            red = True
        else:
            print(f"  PASS L6  {total6}/{total6} documented .py reference(s) across "
                  f"{len(commands)} command assignment(s) are files the publish set "
                  f"ships")

    # L7 --------------------------------------------------------------------
    if not commands:
        print(f"  FAIL L7 UNMEASURED 0/0 — {no_cmd_msg}", file=sys.stderr)
        unmeasured = True
    else:
        unknown7, total7 = check_l7(commands, tflags)
        if total7 == 0:
            print("  FAIL L7 UNMEASURED 0/0 — the documented command(s) carry no "
                  "--flag tokens to check against the trainer's parser. all([]) is "
                  "True; this gate is not.", file=sys.stderr)
            unmeasured = True
        elif unknown7:
            print(f"  FAIL L7  {total7 - len(unknown7)}/{total7} documented flag "
                  f"token(s) exist in the trainer's parser ({len(tflags)} declared); "
                  f"unknown: {', '.join(unknown7)} — argparse would exit 2 on the "
                  f"first one, AFTER allocation, container start and the collective "
                  f"probe", file=sys.stderr)
            red = True
        else:
            print(f"  PASS L7  {total7}/{total7} documented flag token(s) exist in "
                  f"the trainer's parser ({len(tflags)} flags declared by "
                  f"add_argument)")

    # L8 --------------------------------------------------------------------
    if not commands:
        print(f"  FAIL L8 UNMEASURED 0/0 — {no_cmd_msg}", file=sys.stderr)
        unmeasured = True
    else:
        um8, cm8, unres8, sels8 = check_l8(commands, doc_lines, uncond_k, cond_k,
                                           tflags)
        n_unc = len(uncond_k)
        n_cond = len(cond_k)
        if n_unc == 0:
            print("  FAIL L8 UNMEASURED 0/0 unconditional — the trainer's sourcing "
                  "calls yield no unconditional required knob. all([]) is True; this "
                  "gate is not.", file=sys.stderr)
            unmeasured = True
        elif um8:
            print(f"  FAIL L8  {n_unc - len(um8)}/{n_unc} unconditional required "
                  f"knobs are satisfied by the documented command or its exports:",
                  file=sys.stderr)
            for k, e in um8:
                if e is None:
                    detail = (f"flag `{knob_flag(k)}` absent from the documented "
                              f"command, and the knob declares no env fallback — "
                              f"the flag was its only route")
                else:
                    detail = (f"flag `{knob_flag(k)}` absent, and no `export {e}=` "
                              f"line exists either")
                print(f"    MISSING `{k}`: {detail}", file=sys.stderr)
            red = True
        else:
            print(f"  PASS L8  {n_unc}/{n_unc} unconditional required knobs satisfied "
                  f"by a documented flag spelling or an exported env fallback")
        # The conditional denominator is printed SEPARATELY: folding it into the
        # unconditional count is how a mode-gated knob quietly escapes measurement.
        if n_cond == 0:
            print("  FAIL L8 UNMEASURED 0/0 conditional — no branch-gated required "
                  "knobs exist to select. all([]) is True; this gate is not.",
                  file=sys.stderr)
            unmeasured = True
        elif unres8:
            print(f"  FAIL L8 UNMEASURED conditional ({n_cond} knob(s)) — the "
                  f"selector will not resolve, so the gate declines to guess a "
                  f"branch:", file=sys.stderr)
            for names, why in unres8:
                print(f"    knobs [{', '.join(names)}]: {why}", file=sys.stderr)
            if cm8:
                for k, e in cm8:
                    print(f"    MISSING `{k}` on a readable branch "
                          f"(flag `{knob_flag(k)}` absent"
                          + (f", no `export {e}=` line)" if e else ", no env fallback)"),
                          file=sys.stderr)
                red = True
            unmeasured = True
        elif cm8:
            print(f"  FAIL L8  {n_cond - len(cm8)}/{n_cond} conditional required "
                  f"knobs satisfied on the selected branch:", file=sys.stderr)
            for k, e in cm8:
                detail = (f"flag `{knob_flag(k)}` absent"
                          + (f" and no `export {e}=` line" if e
                             else " and no env fallback exists"))
                print(f"    MISSING `{k}`: {detail}", file=sys.stderr)
            red = True
        else:
            sel_txt = "; ".join(f"selector `{f}` selects the `{b}` branch"
                                for f, b in sels8)
            print(f"  PASS L8  {n_cond}/{n_cond} conditional required knobs "
                  f"satisfied on the selected branch(es) ({sel_txt})")

    # L9 --------------------------------------------------------------------
    if not commands:
        print(f"  FAIL L9 UNMEASURED 0/0 — {no_cmd_msg}", file=sys.stderr)
        unmeasured = True
    else:
        viol9, unres9, total9 = check_l9(commands, cond_k, tflags)
        if total9 == 0:
            print("  FAIL L9 UNMEASURED 0/0 — no conditional knobs classified into a "
                  "branch, so no exclusivity property exists to check. all([]) is "
                  "True; this gate is not.", file=sys.stderr)
            unmeasured = True
        elif unres9:
            print(f"  FAIL L9 UNMEASURED ({total9} conditional knob(s)) — the "
                  f"selector will not resolve and exclusivity cannot be certified "
                  f"against a guessed branch:", file=sys.stderr)
            for names, why in unres9:
                print(f"    knobs [{', '.join(names)}]: {why}", file=sys.stderr)
            if viol9:
                for f, b, s in viol9:
                    print(f"    OFF-BRANCH `{f}` (from the `{b}` branch; `{s}` "
                          f"selected)", file=sys.stderr)
                red = True
            unmeasured = True
        elif viol9:
            print(f"  FAIL L9  {len(viol9)} off-branch flag(s) in the documented "
                  f"command (denominator {total9} conditional knob(s) classified); "
                  f"the trainer refuses when the non-selected branch's knobs are "
                  f"supplied:", file=sys.stderr)
            for f, b, s in viol9:
                print(f"    OFF-BRANCH `{f}` belongs to the `{b}` branch; the "
                      f"documented selector value selects `{s}`", file=sys.stderr)
            red = True
        else:
            print(f"  PASS L9  0 off-branch flags in the documented command across "
                  f"{total9} conditional knob(s) classified into branches; only the "
                  f"selected branch's flags appear")

    # L10 -------------------------------------------------------------------
    if codes_unresolved:
        # A census that quietly drops the returns it cannot parse yields a PASS whose
        # denominator is a subset of the contract -- the Constant-only walk this
        # replaced derived {2,3} and certified 2/2 while 0 and 1 (the codes almost
        # every real run ends on) were never asked about. Partial is UNMEASURED.
        print(f"  FAIL L10 UNMEASURED {len(codes)} resolved code(s) but "
              f"{len(codes_unresolved)} return shape(s) under main() did not reduce "
              f"to an integer, so the census has no honest denominator: "
              f"{'; '.join(codes_unresolved)}", file=sys.stderr)
        unmeasured = True
    elif not codes:
        print("  FAIL L10 UNMEASURED 0/0 — no integer return under main() and "
              "argparse nowhere in the trainer, so no exit-code set exists to check "
              "the contract block against. all([]) is True; this gate is not.",
              file=sys.stderr)
        unmeasured = True
    else:
        found10, miss10 = check_l10(doc_lines, codes, TRAINER.name)
        if not found10:
            print(f"  FAIL L10 UNMEASURED 0/{len(codes)} — the doc carries no "
                  f"trainer-contract block (a run of non-empty lines naming the "
                  f"trainer's basename and speaking exits), so the exit-code table "
                  f"cannot be audited. UNMEASURED never passes.", file=sys.stderr)
            unmeasured = True
        elif miss10:
            print(f"  FAIL L10  {len(codes) - len(miss10)}/{len(codes)} trainer exit "
                  f"code(s) appear in the contract block; undocumented: "
                  f"{', '.join(str(c) for c in miss10)} — a code the operator can "
                  f"actually receive, missing from the only table they will read",
                  file=sys.stderr)
            red = True
        else:
            print(f"  PASS L10  {len(codes)}/{len(codes)} exit code(s) main() can "
                  f"produce (returns plus argparse's 2 when argparse is used) appear "
                  f"in the contract block as bolded or backticked literals")

    # L11 -------------------------------------------------------------------
    found11, absent11, ntok11 = check_l11(doc_lines, trainer_src, TRAINER.name)
    if not found11:
        print("  FAIL L11 UNMEASURED 0/0 — no unique trainer-contract block exists, so no "
              "output-marker claims can be collected. UNMEASURED never passes.",
              file=sys.stderr)
        unmeasured = True
    elif ntok11 == 0:
        print("  FAIL L11 UNMEASURED 0/0 — the contract block names no backticked "
              "SHOUTED markers at all. all([]) is True; this gate is not.",
              file=sys.stderr)
        unmeasured = True
    elif absent11:
        print(f"  FAIL L11  {ntok11 - len(absent11)}/{ntok11} documented output "
              f"marker(s) occur in the trainer source; claimed but never printed: "
              f"{', '.join(absent11)}", file=sys.stderr)
        red = True
    else:
        print(f"  PASS L11  {ntok11}/{ntok11} marker token(s) named in the contract "
              f"block occur as literals somewhere in the launch plane (trainer, "
              f"launcher, backend)")

    # L5 --------------------------------------------------------------------
    print("\nCONTROLS (every run, not behind a flag):")
    drill_green, drill_unplantable = controls(doc_text, doc_lines, lau_lines,
                                              knobs if total1 else [],
                                              patterns, trainer_src, published)
    unmeasured = unmeasured or drill_unplantable

    # PRECEDENCE: confirmed violations outrank an unplantable drill.
    #
    # The first real run of this gate reported UNMEASURED while L1, L2 and L3 were all
    # sitting there red, because the L2 drill could not be planted -- and it could not be
    # planted precisely BECAUSE 0 of 19 citations resolved. Reporting "unmeasured" when
    # the reason for it is the very defect being measured buries the actionable verdict
    # under a caveat about the verdict.
    #
    # The asymmetry that makes this sound: what an unplantable MUST_FIRE leaves
    # uncertified is the detector's ability to go correctly GREEN -- its false-NEGATIVE
    # rate. A rule that is red has already demonstrated it can fire on this very input.
    # So the uncertified property cannot change a red into anything else, and the caveat
    # is retained in the output rather than promoted over the finding. When nothing is
    # red, the same uncertainty is decisive and UNMEASURED stands.
    if red or not drill_green:
        print(f"\nLAUNCH DOC GATE RED — violations above, drills "
              f"{'green' if drill_green else 'RED'}"
              + ("; NOTE a drill was unplantable, so the detectors are certified only "
                 "for firing, not for staying silent correctly" if unmeasured else ""),
              file=sys.stderr)
        return 5
    if unmeasured:
        print("\nLAUNCH DOC GATE UNMEASURED — a zero denominator or unplantable drill "
              "makes the green part of this output meaningless. Fix the measurement, "
              "then trust the measurement.", file=sys.stderr)
        return 95
    print("\nLAUNCH DOC GATE GREEN — coverage full, every citation read at its line, "
          "zero estate literals, documented argv in parity with the trainer's parser, "
          "exit table and markers verified, all nine detectors drilled red-to-order")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())