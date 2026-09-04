#!/usr/bin/env python3
"""Gate: every stage-count claim in a shipped document must equal the build's own STAGES array.

WHY THIS EXISTS (#194)

Four published documents carried three different answers to one countable question.
`h100/DELIVERABLE_E_matrix.md` said `17/17`, `h100/DELIVERABLE_B_validation_report.md`
said `33 stages`, `h100/LAUNCH.md` said `17/17 stages` in its lead box, and the build
itself printed `BUILD GREEN -- 36 stages`. Nothing was wrong with the build. What was
wrong is that the number is TYPED in the documents and DERIVED in the build, so every
new stage silently falsified four sentences at once, and the drift was invisible because
each document was internally consistent.

This is the same shape as #157 and #190 -- a claim whose denominator has no producer --
and it fails in the direction people notice least. LAUNCH.md's lead box told operators
that Phase 3 "has never been executed" for a full campaign after job 37310 executed 8/8
legs. Understating is not the safe direction; it is the same defect, and a reader who
catches one understatement stops trusting the overstatements too.

WHAT IT MEASURES

For every document in DOCS that exists, every occurrence of a stage-count claim -- the
patterns in CLAIM_PATTERNS, which cover `N/N stages`, `N stages`, `N-stage` and
`at N stages` -- is extracted with its line number and compared against the ONE
authoritative count: len(STAGES) parsed out of build_h100_plane.sh.

    agreeing claim      ok
    disagreeing claim   RED, unless the line carries the historical marker

WHICH NUMBER, AND WHY THE GATE MUST SAY SO

The build has TWO counts and they are not equal: 36 stage ENTRIES and 35 unique stage
FILES, because `patch_list_separators.py` is deliberately invoked twice. `BUILD GREEN`
prints the entry count; `gate_stage_orphans.py` prints the unique count. Both are correct
about different questions, and the first version of THIS gate picked one silently -- the
defect it exists to close, reproduced inside the detector.

So: the authority is the ENTRY count, because a stage that runs twice runs twice and the
documents' sentences are about what the build does. Both numbers are printed on every
run, and every duplicated stage is named. A claim matching the unique count is still RED,
and the RED message says which number it matched and why that is a different question --
a reader who meant files should say "unique stage files", not "stages".

DENOMINATOR

The number of claims found, printed per document and in total, alongside the number of
documents scanned and the number that contained zero claims. Zero claims across every
document is UNMEASURED (95), never PASS: a scanner that matches nothing cannot
distinguish a clean corpus from a broken pattern, and all([]) is True.

THE HISTORICAL MARKER, AND WHY IT IS NOT AN ALLOWLIST

Some sentences must state an old count on purpose -- Deliverable A's "the measured record
lags the stage list" observation is ABOUT the lag, and rewriting its number would delete
the finding. A line may therefore carry a past-tense marker (HISTORICAL_MARKERS) and its
claims are then counted as HISTORICAL rather than RED.

The marker is deliberately NOT a filename allowlist and NOT a claim-value allowlist. It
attaches to the sentence, it is visible to a human reading that sentence, and it costs
the author a word that changes the prose's meaning ("reported" vs "reports"). An
allowlist keyed on file or number would let a live claim go stale under cover of a
declaration made years earlier -- which is exactly how the class this gate closes hides.
HISTORICAL claims are counted and printed; a document that is entirely historical is
reported as such, because a corpus that has quietly become all-past-tense has stopped
being checked.

CONTROLS

  MUST_FIRE  a planted document stating a count one above the real one goes RED, and the
             gate names the file, the line and both numbers.
  MUST_FIRE  a planted document whose only claim carries a historical marker does NOT go
             RED -- and a second planted copy with the marker removed DOES. The pair is
             the control on the marker itself: an escape hatch that is never observed
             both admitting and refusing is an untested escape hatch.
  MUST_PASS  a planted document stating the real count is green.
  MUST_FIRE  a planted document stating the UNIQUE FILE count instead of the ENTRY count
             goes RED, so "the authority is the entry count" is a measured decision and
             not a sentence in a docstring. Skipped, and said to be skipped, when the two
             counts happen to be equal -- a drill that cannot distinguish its two arms is
             not a drill.
  MUST_PASS  the real corpus, scanned with the real count.
  MUST_FIRE  zero claims across the corpus returns UNMEASURED (95), not PASS.

Controls run on temporary files and never touch the shipped documents.

WHAT IT DOES NOT COVER

  * Only the stage COUNT. The narrative status lines corrected alongside this gate --
    which jobs ran, which legs are UNMEASURED -- are prose about a job record, not a
    countable the build holds, and this gate makes no claim about them. That residual is
    stated in E's `stale status lines` row rather than papered over here.
  * Documents outside DOCS. The list is explicit and printed on every run.

EXIT CODES: 0 PASS, 5 RED, 95 UNMEASURED (no claims found, or the STAGES array could not
be parsed), 96 REFUSE (a control failed).
"""

import re
import sys
import tempfile
from pathlib import Path

EXIT_PASS = 0
EXIT_RED = 5
EXIT_UNMEASURED = 95
EXIT_REFUSE = 96

BUILD_SCRIPT = "build_h100_plane.sh"

# Explicit and printed: the gate's scope is a claim, not an implementation detail.
DOCS = (
    "h100/LAUNCH.md",
    "h100/DELIVERABLE_A_architecture_review.md",
    "h100/DELIVERABLE_B_validation_report.md",
    "h100/DELIVERABLE_E_matrix.md",
    "README.md",
)

# Each pattern's group 'n' is the claimed count. The N/N form additionally carries a
# group 'd' that must equal it -- "36/35 stages green" is incoherent on its own terms.
CLAIM_PATTERNS = (
    re.compile(r"(?P<n>\d{1,3})\s*/\s*(?P<d>\d{1,3})\s+stages?\b"),
    # '#' is in the lookbehind because '#191 stage/publish-set gate' is a FINDING ID
    # followed by a noun, not a count. Measured false positive on the first run.
    re.compile(r"(?<![-\w/#])(?P<n>\d{1,3})[- ]stage\b"),
    re.compile(r"(?<![-\w/#])(?P<n>\d{1,3})\s+stages?\b"),
    # NOUN BEFORE THE NUMBERS. Deliverable E's compatibility matrix writes the claim as a
    # table cell, `| stages green | **36/36** |`, and the first pattern above requires the
    # noun AFTER the fraction -- so that cell sat in no denominator and went stale by a
    # whole stage while the gate reported the file clean. Measured 2026-09-01 when #182
    # landed as stage 37. The window is bounded and may not cross a '/', so an unrelated
    # fraction further along the line (`8/8 legs`) cannot be captured by a distant 'stage'.
    # Narrowed after its first run: unqualified, this matched three modifier uses in
    # DELIVERABLE_E ("build stage 34, 5/5 controls", "stage counts (`17/17`)") and
    # manufactured three false REDs. Two independent narrowings, each with its own reason,
    # rather than an allowlist of the three lines:
    #   PLURAL only -- a count claim written noun-first is always plural ("stages green:
    #     36/36"); singular `stage N` is an ordinal and `stage counts` is a modifier.
    #   NO DIGIT in the window -- so `stages 34, 5/5` cannot bind the fraction across an
    #     intervening number even in the plural.
    # Both are drilled by MUST_PASS/SINGULAR_STAGE_IS_A_MODIFIER_NOT_A_COUNT.
    re.compile(r"\bstages\b[^\n/\d]{0,24}?(?<![\w/])(?P<n>\d{1,3})\s*/\s*(?P<d>\d{1,3})(?![\w/])"),
)

# Past tense, or an explicit restatement. Lowercased substring match on the whole line.
HISTORICAL_MARKERS = (
    "reported",
    "restated",
    "used to",
    "previously",
    "earlier ",
    "was stale",
    "stale `",
    "historical",
    "at audit time",
)


def _stderr(msg):
    sys.stderr.write(msg + "\n")


def parse_stages(text):
    """Return the STAGES=( ... ) entries, or None if the array cannot be located.

    Same reader as gate_stage_orphans.py: the array is the build's own declaration of
    what it runs, and deriving the count any other way (globbing patch_*.py, counting
    invocations) would substitute a proxy for the thing the documents claim."""
    entries = []
    in_array = False
    closed = False
    for line in text.splitlines():
        if not in_array:
            stripped = line.strip()
            if stripped.startswith("STAGES=("):
                in_array = True
                line = stripped[len("STAGES=("):]
                if not line:
                    continue
            else:
                continue
        stripped = line.strip()
        if stripped.startswith(")"):
            closed = True
            break
        code = line.split("#", 1)[0]
        for token in code.split():
            entries.append(token)
    if not in_array or not closed:
        return None
    return entries


def is_historical(line):
    low = line.lower()
    return any(m in low for m in HISTORICAL_MARKERS)


def scan_text(text, expected):
    """Return (agree, historical, bad) where bad is a list of (lineno, claim, snippet)."""
    agree = 0
    historical = 0
    bad = []
    for lineno, line in enumerate(text.splitlines(), 1):
        spans = []
        for pat in CLAIM_PATTERNS:
            for m in pat.finditer(line):
                # A later pattern re-matching text an earlier one already claimed would
                # double-count the same sentence and inflate the denominator.
                if any(s <= m.start() < e for s, e in spans):
                    continue
                spans.append((m.start(), m.end()))
                n = int(m.group("n"))
                d = m.groupdict().get("d")
                ok = n == expected and (d is None or int(d) == expected)
                if ok:
                    agree += 1
                elif is_historical(line):
                    historical += 1
                else:
                    bad.append((lineno, m.group(0).strip(), line.strip()[:140]))
    return agree, historical, bad


def scan_corpus(root, expected, docs=DOCS):
    """Return (rows, totals). rows: one per document that exists."""
    rows = []
    t_agree = t_hist = 0
    t_bad = []
    for rel in docs:
        path = root / rel
        if not path.exists():
            rows.append((rel, None, None, None))
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            rows.append((rel, "unreadable: %s" % exc, None, None))
            continue
        agree, hist, bad = scan_text(text, expected)
        rows.append((rel, agree, hist, bad))
        t_agree += agree
        t_hist += hist
        t_bad.extend((rel, ln, claim, snip) for ln, claim, snip in bad)
    return rows, (t_agree, t_hist, t_bad)


# --------------------------------------------------------------------------- controls

_C_REAL = "The plane builds green at %d stages and every stage is wired in.\n"
_C_WRONG = "The plane builds green at %d stages and every stage is wired in.\n"
_C_HIST = "E.5 reported %d/%d stages green, measured 2026-08-31.\n"
_C_NOHIST = "E.5 says %d/%d stages green.\n"
_C_NONE = "This document states no countable stage claim at all.\n"
# Noun BEFORE the numbers -- the notation Deliverable E's matrix uses and the one this
# gate was blind to until 2026-09-01. Kept as a distinct fixture from _C_NOHIST so that a
# regression in either word order is attributable to one control rather than to "the
# fraction rule".
_C_TABLE = "| stages green | **%d/%d** |\n"
# The bound on the new pattern's window, stated as a document. `stages` and `8/8` are on
# one line but 35 characters apart, so the fraction belongs to the legs and not to the
# stages. If the window is ever widened past 24 this control goes red instead of the claim
# quietly becoming wrong.
_C_FARFRAC = "The %d stages are green and the resume proof is 8/8 legs.\n"
# The two modifier uses the unqualified noun-first pattern read as count claims on its
# first run. Line 1 is a real claim, so agree>=1 proves the scanner actually read this
# document -- without it "0 red" would be satisfied by a scanner that read nothing.
_C_MODIFIER = (
    "The plane builds green at %d stages.\n"
    "That stage landed as build stage 34 (5/5 controls fired), and the citation gate's "
    "stage counts (`17/17`) were re-derived.\n"
)


def _plant(tmp, name, body):
    p = Path(tmp) / name
    p.write_text(body, encoding="utf-8")
    return name


def run_controls(expected, uniq=None):
    """Execute the real scanner over planted documents. Returns a list of
    (label, ok, observed) and never re-implements the rule under test."""
    out = []
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        n_wrong = _plant(tmp, "wrong.md", _C_WRONG % (expected + 1))
        n_right = _plant(tmp, "right.md", _C_REAL % expected)
        n_hist = _plant(tmp, "hist.md", _C_HIST % (expected - 1, expected - 1))
        n_nohist = _plant(tmp, "nohist.md", _C_NOHIST % (expected - 1, expected - 1))
        n_none = _plant(tmp, "none.md", _C_NONE)
        n_table = _plant(tmp, "table.md", _C_TABLE % (expected - 1, expected - 1))
        n_far = _plant(tmp, "far.md", _C_FARFRAC % expected)
        n_mod = _plant(tmp, "mod.md", _C_MODIFIER % expected)

        _, (_, _, bad) = scan_corpus(root, expected, (n_wrong,))
        out.append((
            "MUST_FIRE/STALE_COUNT_GOES_RED",
            len(bad) == 1 and str(expected + 1) in bad[0][2],
            "%d disagreeing claim(s); first=%r" % (len(bad), bad[0][2] if bad else None),
        ))

        _, (agree, _, bad) = scan_corpus(root, expected, (n_right,))
        out.append((
            "MUST_PASS/CURRENT_COUNT_IS_GREEN",
            agree == 1 and not bad,
            "agree=%d bad=%d" % (agree, len(bad)),
        ))

        _, (_, hist, bad) = scan_corpus(root, expected, (n_hist,))
        ok_hist = hist >= 1 and not bad
        _, (_, hist2, bad2) = scan_corpus(root, expected, (n_nohist,))
        ok_nohist = len(bad2) >= 1 and hist2 == 0
        out.append((
            "MUST_FIRE/HISTORICAL_MARKER_ADMITS_AND_ITS_ABSENCE_REFUSES",
            ok_hist and ok_nohist,
            "marked: historical=%d red=%d | unmarked: historical=%d red=%d"
            % (hist, len(bad), hist2, len(bad2)),
        ))

        if uniq is not None and uniq != expected:
            # Pins the decision the docstring makes. Without this drill, "the authority is
            # the entry count" is a sentence in a comment; with it, the near-miss value
            # that a careless author would most plausibly type is observed going red.
            n_uniq = _plant(tmp, "uniq.md", _C_REAL % uniq)
            _, (_, _, bad_u) = scan_corpus(root, expected, (n_uniq,))
            out.append((
                "MUST_FIRE/UNIQUE_FILE_COUNT_IS_NOT_THE_ENTRY_COUNT",
                len(bad_u) == 1,
                "a doc claiming the %d unique FILES rather than the %d ENTRIES: %d red"
                % (uniq, expected, len(bad_u)),
            ))

        # The measured miss. `| stages green | **36/36** |` sat in no denominator for a
        # whole stage because the fraction rule required the noun AFTER the numbers, and
        # the file it lives in was reported clean the entire time. Two legs, because the
        # widening has two ways to be wrong: too narrow (the cell stays invisible) and too
        # greedy (an unrelated fraction on a line that happens to say "stages" is read as
        # a stage count).
        _, (_, _, bad_t) = scan_corpus(root, expected, (n_table,))
        out.append((
            "MUST_FIRE/NOUN_BEFORE_THE_FRACTION_IS_STILL_A_CLAIM",
            len(bad_t) == 1 and str(expected - 1) in bad_t[0][2],
            "table-cell notation `| stages green | **N/N** |`: %d red" % len(bad_t),
        ))
        _, (agree_f, _, bad_f) = scan_corpus(root, expected, (n_far,))
        out.append((
            "MUST_PASS/A_DISTANT_FRACTION_IS_NOT_A_STAGE_CLAIM",
            agree_f >= 1 and not bad_f,
            "`%d stages ... 8/8 legs` (35 chars apart): agree=%d red=%d"
            % (expected, agree_f, len(bad_f)),
        ))

        _, (agree_m, _, bad_m) = scan_corpus(root, expected, (n_mod,))
        out.append((
            "MUST_PASS/SINGULAR_STAGE_IS_A_MODIFIER_NOT_A_COUNT",
            agree_m >= 1 and not bad_m,
            "`build stage 34 (5/5` + `stage counts (\\`17/17\\``: agree=%d red=%d"
            % (agree_m, len(bad_m)),
        ))

        _, (agree, hist, bad) = scan_corpus(root, expected, (n_none,))
        out.append((
            "MUST_FIRE/ZERO_CLAIMS_IS_NOT_A_PASS",
            agree == 0 and hist == 0 and not bad,
            "agree=0 historical=0 red=0 -> caller must map to UNMEASURED",
        ))
    return out


def main():
    root = Path(__file__).resolve().parent
    script = root / BUILD_SCRIPT
    print("DOC STAGE-COUNT GATE — every stage-count claim in a shipped document, read "
          "against the build's own STAGES array")
    print("  authority: %s  STAGES=( ... )" % BUILD_SCRIPT)
    print("  scope:     %d document(s): %s" % (len(DOCS), ", ".join(DOCS)))

    if not script.exists():
        _stderr("UNMEASURED: %s not found — the gate cannot derive its own authority, "
                "and a count guessed from the filesystem would be the defect this gate "
                "closes." % BUILD_SCRIPT)
        return EXIT_UNMEASURED
    stages = parse_stages(script.read_text(encoding="utf-8", errors="replace"))
    if stages is None:
        _stderr("UNMEASURED: no 'STAGES=(' ... ')' block in %s — refusing to guess."
                % BUILD_SCRIPT)
        return EXIT_UNMEASURED
    if not stages:
        _stderr("UNMEASURED: STAGES parsed to zero entries — zero units measured is "
                "UNMEASURED, not PASS.")
        return EXIT_UNMEASURED
    expected = len(stages)
    uniq = len(set(stages))
    dupes = sorted({x for x in stages if stages.count(x) > 1})
    print("  authoritative count: %d stage ENTRIES (the build's own `BUILD GREEN` figure)"
          % expected)
    print("  second, non-authoritative count: %d unique stage FILE(S)%s — printed because "
          "the build reports both and a gate that silently picks one is the defect it "
          "closes" % (uniq, (" (invoked more than once: %s)" % ", ".join(dupes)) if dupes else ""))

    controls = run_controls(expected, uniq)
    print("  controls:")
    for label, ok, observed in controls:
        print("    %-58s %-4s %s" % (label, "ok" if ok else "FAIL", observed))
    failed = [c for c in controls if not c[1]]
    if failed:
        _stderr("REFUSE 96: %d control(s) failed — the detector is not known to work, "
                "so its green means nothing." % len(failed))
        return EXIT_REFUSE

    rows, (t_agree, t_hist, t_bad) = scan_corpus(root, expected)
    print("  per document:")
    missing = 0
    silent = 0
    for rel, agree, hist, bad in rows:
        if agree is None:
            print("    %-48s absent (not scanned)" % rel)
            missing += 1
            continue
        if isinstance(agree, str):
            print("    %-48s %s" % (rel, agree))
            missing += 1
            continue
        total = agree + hist + len(bad)
        if total == 0:
            silent += 1
        note = ""
        if total and agree == 0 and not bad:
            note = "  <-- every claim here is HISTORICAL; nothing in this file is checked"
        print("    %-48s claims=%-3d agree=%-3d historical=%-3d stale=%-3d%s"
              % (rel, total, agree, hist, len(bad), note))

    denom = t_agree + t_hist + len(t_bad)
    print("  DENOMINATOR: %d claim(s) over %d document(s) scanned (%d absent, %d with "
          "no claim)" % (denom, len(DOCS) - missing, missing, silent))

    if denom == 0:
        _stderr("UNMEASURED: zero stage-count claims found across the whole corpus. A "
                "scanner that matches nothing cannot tell a clean corpus from a broken "
                "pattern — all([]) is True, so this is UNMEASURED, not PASS.")
        return EXIT_UNMEASURED

    if t_bad:
        _stderr("RED: %d stage-count claim(s) disagree with the build's %d stages:"
                % (len(t_bad), expected))
        for rel, lineno, claim, snip in t_bad:
            _stderr("  %s:%d  claims %r, build has %d" % (rel, lineno, claim, expected))
            _stderr("      %s" % snip)
            if uniq != expected and str(uniq) in claim:
                _stderr("      NOTE: %d is the count of unique stage FILES, not stage "
                        "entries. %s runs twice. If the sentence is about files, say "
                        "'unique stage files'; if it is about what the build does, the "
                        "number is %d."
                        % (uniq, ", ".join(dupes) or "a stage", expected))
        _stderr("Fix the document, or — if the sentence is deliberately about the past — "
                "make it past tense so a reader sees the same thing this gate does. "
                "Markers: %s" % ", ".join(repr(m) for m in HISTORICAL_MARKERS))
        return EXIT_RED

    print("DOC STAGE-COUNT GATE GREEN — %d claim(s) agree at %d stages, %d historical, "
          "0 stale" % (t_agree, expected, t_hist))
    return EXIT_PASS


if __name__ == "__main__":
    sys.exit(main())
