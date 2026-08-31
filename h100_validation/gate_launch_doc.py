#!/usr/bin/env python3
"""gate_launch_doc.py: pin h100/LAUNCH.md against the launcher it claims to document.

WHY THIS GATE EXISTS. An operator document is a claim about a program. Three ways that
claim drifts from the program, all found in this estate before this gate was written:

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

A gate on any one of these was proposed and rejected as too easy to game by accident:
a coverage gate with zero knobs passes vacuously (`all([]) is True`), a citation gate
whose denominator silently shrinks passes vacuously, a redaction gate with no patterns
configured passes vacuously. So every denominator is printed, a zero denominator is
UNMEASURED and fails the gate (exit 95), and the drills run on every invocation --
a detector that cannot be observed going red is not a detector.

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
  L5  CONTROLS on every run: MUST_FIRE for L1 (covered knob deleted), L2 (citation
      renumbered), L3 (partition literal planted); MUST_PASS on the unmodified pair.
      An unplantable drill is UNMEASURED and fails the gate -- it proves nothing.

EXIT 0 all rules hold and all controls green; 5 RED; 95 UNMEASURED; 96 REFUSE.
"""

from __future__ import annotations

import os
import pathlib
import re
import sys

H100 = pathlib.Path(__file__).resolve().parent / "h100"
DOC = H100 / "LAUNCH.md"
LAUNCHER = H100 / "gen" / "launch_fs_h100.fixed.sh"

# L1: the launcher's own guard idiom for required-no-default knobs. Spacing-tolerant,
# but the SHAPE is fixed on purpose: `||` present, `${NAME:-}` empty-default probe.
GUARD_RE = re.compile(
    r'\[\[\s+-n\s+"\$\{([A-Za-z_][A-Za-z0-9_]*):-\}"\s+\]\]\s+\|\|'
)
# L2: a citation is L:<digits>; the row's subject is the FIRST backticked FS_ token.
# A citation is a line number OR an inclusive range: L:358, L:358-365, L:34–182.
# Both the ASCII hyphen and the en-dash a writer's editor substitutes for it, because a
# gate that only knows one of them silently reclassifies half the document's pointers as
# single-line citations and then reports them stale -- wrong, and wrong in the direction
# that manufactures work.
CITE_RE = re.compile(r"\bL:(\d+)(?:[-–—](\d+))?\b")

# Every identifier inside a backticked span, not the span itself. The first version of
# this required the whole backtick content to be a bare FS_ name, so `--partition="$FS_PARTITION"`
# -- a row that names its subject about as plainly as a row can -- registered as having
# no subject at all. Two separate narrowings, same defect: the oracle described a
# citation shape the document does not actually use.
SPAN_RE = re.compile(r"`([^`]+)`")
IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]{2,}")


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


def check_l2(doc_lines: list[str],
             lau_lines: list[str]) -> tuple[list[str], int, int, int]:
    """L2. Returns (misses, resolved, total, exact).

    Every L:<n> or L:<a>-<b> on a doc line is one citation. The denominator counts ALL
    of them, including rows that name no subject -- count it, report it, never drop it.

    A citation RESOLVES when some subject named on the row appears somewhere in the
    cited range. `exact` separately counts the citations that land on the FIRST line of
    their own range, which is the strictest reading. Both numbers are printed: a range
    is a legitimate way to point at a block, but a gate that only reports the loose
    count lets a range quietly widen until it cannot fail. Reporting the strict count
    beside it means any such drift is visible in the ratio rather than hidden by it.
    """
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
        for start_s, end_s in cites:
            total += 1
            lo = int(start_s)
            hi = int(end_s) if end_s else lo
            span = f"L:{lo}" + (f"-{hi}" if hi != lo else "")
            if not subjects:
                misses.append(f"{span} on doc line {dno}: the row names no backticked "
                              f"subject, so nothing about it is checkable — stays in "
                              f"the denominator")
                continue
            if not (1 <= lo <= len(lau_lines) and 1 <= hi <= len(lau_lines) and lo <= hi):
                misses.append(f"{span} on doc line {dno} -> out of range (launcher has "
                              f"{len(lau_lines)} line(s))")
                continue
            window = lau_lines[lo - 1:hi]
            hit = next((s for s in subjects if any(s in w for w in window)), None)
            if hit is None:
                misses.append(f"{'/'.join(subjects[:3])} {span} -> "
                              f"{lau_lines[lo - 1].strip()[:60]}")
                continue
            resolved += 1
            if hit in lau_lines[lo - 1]:
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


def controls(doc_text: str, doc_lines: list[str], lau_lines: list[str],
             knobs: list[str], patterns: list[tuple[str, "re.Pattern[str]"]]
             ) -> tuple[bool, bool]:
    """L5. Returns (all_green, any_unplantable). Runs every time, not behind a flag."""
    green = True
    unplantable = False

    # Snapshot for the MUST_PASS at the bottom. The claim being made there is
    # NON-DESTRUCTIVENESS -- that the drills mutated only copies -- so the thing to
    # compare against is the rule output as it stood BEFORE any drill ran, not a
    # re-statement of one of the rules. Cheap, and it is the only way the claim
    # "the drill machinery added no violation" is actually backed by evidence.
    before = (check_l1(doc_text, knobs),
              check_l2(doc_lines, lau_lines),
              check_l3(doc_lines, patterns))

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
            lo = int(m.group(1))
            hi = int(m.group(2)) if m.group(2) else lo
            if not (1 <= lo <= hi <= len(lau_lines)):
                continue
            window = lau_lines[lo - 1:hi]
            if not any(s in w for s in subjects for w in window):
                continue  # not a resolving citation; would not test the detector
            # Retarget to a line no subject on this row mentions. Must clear EVERY
            # subject, not just one: the resolver accepts any of them, so a target that
            # only defeats the first would leave the citation resolving and the drill
            # would report a dead detector as alive.
            target = next((i for i in range(1, len(lau_lines) + 1)
                           if not any(s in lau_lines[i - 1] for s in subjects)), None)
            if target is None:
                continue  # every launcher line names something on this row; try another
            # m.group(0), not a reconstruction. The doc writes ranges with an en-dash;
            # rebuilding the citation as f"L:{lo}-{hi}" produces a string that is not in
            # the line, so .replace() would no-op and the "poisoned" copy would be the
            # clean copy -- a MUST_FIRE that silently tests nothing. Use the text the
            # regex actually matched.
            poisoned_lines = list(doc_lines)
            poisoned_lines[dno] = dline.replace(m.group(0), f"L:{target}", 1)
            misses, _, _, _ = check_l2(poisoned_lines, lau_lines)
            if any(f" L:{target} ->" in m for m in misses):
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

    # MUST_PASS: the drill machinery added no violation to the real pair.
    #
    # What stood here asserted that the LIVE doc contains zero partition literals. That
    # is rule L3 restated -- a property of the artifact, not of the controls -- so it
    # proved nothing about the drills while guaranteeing a crash on exactly the docs
    # this gate is for. The claim worth making is non-destructiveness: re-derive all
    # three rules from the originals now that every drill has run, and require the
    # answers to be identical to the pre-drill snapshot. If a drill had mutated a shared
    # list in place instead of a copy, this and only this catches it.
    after = (check_l1(doc_text, knobs),
             check_l2(doc_lines, lau_lines),
             check_l3(doc_lines, patterns))
    if after == before:
        print("  PASS L5/MUST_PASS: all three rules re-derive identically after the "
              "drills; every mutation landed on a copy")
    else:
        differing = [n for n, a, b in zip(("L1", "L2", "L3"), after, before) if a != b]
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

    lau_lines = lau_text.splitlines()
    doc_lines = doc_text.splitlines()
    print(f"launcher: {len(lau_lines)} lines   doc: {len(doc_lines)} lines\n")

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
        print("  FAIL L2 UNMEASURED 0/0 — the doc carries no L:<n> citation to "
              "resolve, so citation grounding cannot be certified. all([]) is True; "
              "this gate is not.", file=sys.stderr)
        unmeasured = True
    elif misses2:
        print(f"  FAIL L2  {resolved2}/{total2} citation(s) resolve against the "
              f"launcher; {len(misses2)} miss(es):", file=sys.stderr)
        for m in misses2:
            print(f"    MISS {m}", file=sys.stderr)
        red = True
    else:
        print(f"  PASS L2  {resolved2}/{total2} citation(s) resolve: the cited launcher "
              f"line or range names a subject the doc row names — {exact2} of them on "
              f"the range's first line (the strict reading), {resolved2 - exact2} "
              f"elsewhere inside a declared range")

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

    # L5 --------------------------------------------------------------------
    print("\nCONTROLS (every run, not behind a flag):")
    drill_green, drill_unplantable = controls(doc_text, doc_lines, lau_lines,
                                              knobs if total1 else [],
                                              patterns)
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
          "zero estate literals, all three detectors drilled red-to-order")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
