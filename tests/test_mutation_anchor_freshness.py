"""Anchor freshness: the cheap front gate against silent corpus rot.

Every pytest round, resolve EVERY published mutation-row anchor exactly once
-- the tools/mutations.json half AND tools/mutate.py's EMBEDDED_TABLE half,
so neither can orphan the other -- byte-exact and indentation included. Raw
str.count only: the drift this gate exists to catch was eight leading spaces
becoming four, and any matcher that strips or normalises greens it.

Ground truth is RECONSTRUCTED, never read. Reading the raw working tree as if
pristine is the text tripwire: mid-battery the source on disk IS one applied
mutant, so a raw reader reds every leg of that file regardless of behaviour
and takes the inert must-pass control down with it. Reading git HEAD is wrong
in the other direction: it lags any uncommitted repair round and reports a
stale row that is not stale, and a false alarm costs what a false green
costs. The corpus is its own inverse map, so the gate asks whether SOME
pristine text exists that binds every anchor exactly once and reaches the
live bytes in zero or one applications.

Denominator: 69 JSON rows over 8 modules + 9 EMBEDDED_TABLE rows over 1
module = 78 rows over 9 modules, pinned as constants below. #242 moved
64 -> 69: it gave a "must_survive" control row to each of the five modules
that had none, because CI now shards the battery per module and a shard
without its negative half is half a detector. Silent row
deletion reds here rather than shrinking the denominator; a legitimate corpus
change updates the pins.

Shape denominator: 3 insertion-style rows and 2 deletion-style rows, pinned
below for the same reason. A shape count of zero must never be used merely
to green a corpus edit: at zero that shape is unexercised by the real
corpus and its corresponding control has become vacuous.

Controls. MUST_PASS: the real shipped corpus, its pinned row-shape counts,
and the admissibility guard that a non-insertion-style anchor never
survives one application, plus the four row shapes that make "which row is
applied?" undecidable -- deletion-style, insertion-style, identical shared
anchors and nested anchors -- each asserted green in BOTH the pristine and
the applied state. MUST_FIRE: a stale anchor, a duplicated anchor, two rows
applied at once, a no-op row, candidate blow-up, the fail-closed set, and
the overlap-survivor documentation control whose literal fixture pins the
declared current limitation. Every verdict leg drives
mutate.check_anchor_freshness. The real-corpus shape guard reads real bytes
only to reconstruct through the gate's accepted candidates; it never calls
those bytes pristine. Every MUST_FIRE and shape-state control uses a
synthetic corpus over a tmp_path file the test wrote itself.

Residuals, stated honestly. The verdict is that anchors BIND, not that the
tree is pristine. Drift displacing no anchor is invisible unless it carries
replacement text from one of the 3 self-surviving insertion rows among the
shipped 73; the other 70 replacement strings are never freshness signals.
A blanket replacement rule would reject the 2 deletion-style rows whose
replacement necessarily occurs inside their intact anchor. The verdict is
per-MODULE, not per-ROW -- the 73 rows carry only 70 distinct (module,
anchor) pairs, so rows sharing an anchor are aliased and a green is not 73
independent verdicts. It classifies bytes, it does not date them: unrelated
drift that lands on some row's applied state reads as that applied state,
because with no external history the two are the same bytes. The general
overlap-survivor composition hole remains a stated limitation; the corpus
shape guard is what makes it unreachable. Findings accumulate ACROSS
modules, while a blank-row module masks its remaining rows for that run and
a zero-row corpus short-circuits. Whether the suite would kill any mutant
is the battery's verdict, never this gate's.
"""

import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MUTATE_PY = ROOT / "tools" / "mutate.py"
MUTATIONS_JSON = ROOT / "tools" / "mutations.json"

JSON_ROWS, JSON_MODS = 69, 8
EMBEDDED_ROWS, EMBEDDED_MODS = 9, 1
TOTAL_ROWS, TOTAL_MODS = 78, 9
# Rows are the corpus denominator; distinct (module, anchor) pairs are the
# DISTINGUISHABILITY denominator. They differ because `core` publishes 9 rows
# over 6 anchors, so a green verdict there resolves the anchor set and cannot
# say which of the aliased rows a state carries. Pinned so neither figure can
# be quoted as the other.
DISTINCT_ANCHORS = 75
# Row-shape counts are pins, not prose trivia. They are the reason the
# insertion composition control and deletion exactly-once control exercise
# real shipped shapes. Do not set either pin to 0 merely to pass after a
# corpus edit: a 0 means the shape is unexercised and its control is
# vacuous, which must red until an explicit reasoning round retires it.
INSERTION_STYLE_ROWS = 3
DELETION_STYLE_ROWS = 2


def _load_mutate():
    """Import tools/mutate.py by path (tools/ is not a package)."""
    spec = importlib.util.spec_from_file_location("mutate", MUTATE_PY)
    assert spec is not None and spec.loader is not None, (
        f"cannot load {MUTATE_PY} -- fail closed: the corpus tool being "
        "unreadable is a build red, never something to route around"
    )
    mod = importlib.util.module_from_spec(spec)
    # Register BEFORE exec: tools/mutate.py defines dataclasses, and
    # dataclasses resolves a class's annotations through
    # sys.modules[cls.__module__].__dict__. Exec'ing an unregistered module
    # makes that lookup return None -> AttributeError at class creation.
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _row(name, anchor, replacement):
    return {
        "name": name,
        "what": "synthetic",
        "anchor": anchor,
        "replacement": replacement,
    }


def _insertion_style(row):
    """True when the replacement text contains the anchor."""
    return bool(row["anchor"]) and row["replacement"].count(row["anchor"]) >= 1


def _deletion_style(row):
    """True when the anchor text contains its replacement."""
    return bool(row["replacement"]) and row["anchor"].count(row["replacement"]) >= 1


def _green_in_every_state(mutate, tmp_path, mod, pristine, rows):
    """Assert 0 problems in the pristine state AND each singly-applied one.

    This is the shape of every state a battery leg can put a source into:
    untouched, or with exactly one row applied. Both must certify fresh.
    """
    src = tmp_path / f"{mod}.py"
    states = [("pristine", pristine)]
    for row in rows:
        applied = pristine.replace(row["anchor"], row["replacement"], 1)
        assert applied != pristine, (
            f"fixture defect: applying {row['name']!r} changed nothing, so "
            "the applied leg would re-test the pristine state"
        )
        states.append((f"applied:{row['name']}", applied))
    for label, text in states:
        src.write_text(text, "utf-8")
        problems = mutate.check_anchor_freshness({mod: rows}, {mod: src})
        assert problems == [], f"{mod} [{label}] must certify fresh; got:\n" + "\n".join(
            f"  {p}" for p in problems
        )
    return states


def test_must_pass_shipped_corpus_binds_every_anchor_exactly_once():
    """MUST_PASS over the real shipped corpus: 73 rows, 9 modules, 0
    problems, with the per-source split stated.

    The gate reconstructs pristine per module from the LIVE bytes, so this
    is correct on a quiet tree and mid-battery alike, when this very suite
    is exec'd with one mutant written into a source. No git, no snapshot.
    """
    mutate = _load_mutate()
    # tools/mutations.json is read from the working tree deliberately: it is
    # not in MODULE_PATHS, so no battery leg ever applies a mutant to it.
    data = json.loads(MUTATIONS_JSON.read_text("utf-8"))
    json_rows = sum(len(rows) for rows in data.values())
    emb_rows = sum(len(rows) for rows in mutate.EMBEDDED_TABLE.values())
    denominator = (
        f"{json_rows} JSON rows over {len(data)} module(s) + {emb_rows} "
        f"EMBEDDED_TABLE rows over {len(mutate.EMBEDDED_TABLE)} module(s) "
        f"= {json_rows + emb_rows} rows over "
        f"{len(set(data) | set(mutate.EMBEDDED_TABLE))} module(s)"
    )
    assert (json_rows, len(data)) == (JSON_ROWS, JSON_MODS), f"JSON half drifted: {denominator}"
    assert (emb_rows, len(mutate.EMBEDDED_TABLE)) == (
        EMBEDDED_ROWS,
        EMBEDDED_MODS,
    ), f"embedded half drifted: {denominator}"
    assert len(mutate.MODULE_PATHS) == TOTAL_MODS, f"module map drifted: {denominator}"
    table = mutate.load_table(None)
    assert sum(len(rows) for rows in table.values()) == TOTAL_ROWS, denominator
    assert len(table) == TOTAL_MODS, denominator
    corpus_rows = [row for rows in table.values() for row in rows]
    insertion_rows = [row for row in corpus_rows if _insertion_style(row)]
    deletion_rows = [row for row in corpus_rows if _deletion_style(row)]
    assert len(insertion_rows) == INSERTION_STYLE_ROWS, (
        f"insertion-style shape denumerator drifted: "
        f"{len(insertion_rows)} of {TOTAL_ROWS} row(s). A count of 0 would "
        "leave this shape unexercised and its control vacuous."
    )
    assert len(deletion_rows) == DELETION_STYLE_ROWS, (
        f"deletion-style shape denominator drifted: "
        f"{len(deletion_rows)} of {TOTAL_ROWS} row(s). A count of 0 would "
        "leave this shape unexercised and its control vacuous."
    )
    distinct = len({(mod, r["anchor"]) for mod, rows in table.items() for r in rows})
    assert distinct == DISTINCT_ANCHORS, (
        f"distinguishability denominator drifted: {distinct} distinct "
        f"(module, anchor) pair(s) behind {TOTAL_ROWS} row(s). A green "
        "verdict resolves the ANCHOR set; rows sharing an anchor are "
        "aliased and are not independent verdicts."
    )
    paths = {mod: ROOT / rel for mod, rel in mutate.MODULE_PATHS.items()}
    problems = mutate.check_anchor_freshness(table, paths)
    assert problems == [], (
        f"anchor freshness over {denominator}: {len(problems)} problem(s), "
        "all accumulated:\n" + "\n".join(f"  {p}" for p in problems)
    )


def test_must_pass_non_insertion_anchors_never_survive_application():
    """MUST_PASS admissibility control over all 73 rows / 9 real modules.

    An insertion-style anchor survives its own application by construction;
    any OTHER survivor is the overlap shape left uncovered by the
    self-surviving clause. Today that measurement is zero. A future row of
    that shape reds here and points at D1 rather than shipping a silent
    false green.

    TEXT TRIPWIRE handling. The bytes read from MODULE_PATHS may be a live
    battery mutant, and this leg reads them anyway -- deliberately, because
    the LIVE bytes are exactly the state whose reachability is in question.
    It is sound because the property is invariant under any single-row
    application, which was MEASURED rather than argued:

      * over all 82 states the battery can reach (9 pristine + 73 singly
        applied), survivors = 0; and
      * 0 rows have a replacement that introduces another row's anchor, so
        applying one row cannot mint an occurrence for a different row.

    Routing through mutate._accepted_pristine_states instead would be
    UNSOUND here, not safer: on the quiet tree that call returns 11
    candidates over 9 modules, and the 2 surplus ones are counterfactual
    texts (``* width * width``, ``and bool(named) and bool(named)``) in
    which a deletion-style row does self-overlap. Those texts are not the
    real file and the battery never produces them, so measuring through
    candidates reds on healthy corpora -- a false alarm, which costs what a
    false green costs.
    """
    mutate = _load_mutate()
    table = mutate.load_table(None)
    paths = {mod: ROOT / rel for mod, rel in mutate.MODULE_PATHS.items()}
    assert (sum(len(rows) for rows in table.values()), len(table)) == (TOTAL_ROWS, TOTAL_MODS)
    survivors = []
    measured = 0
    for mod in sorted(table):
        rows = table[mod]
        live = paths[mod].read_text("utf-8")
        for row in rows:
            if not row["anchor"] or _insertion_style(row):
                continue
            measured += 1
            applied = live.replace(row["anchor"], row["replacement"], 1)
            if applied != live and applied.count(row["anchor"]) >= 1:
                survivors.append(f"{mod}/{row['name']}")
    assert measured == TOTAL_ROWS - INSERTION_STYLE_ROWS, (
        f"admissibility guard measured {measured} row(s), expected "
        f"{TOTAL_ROWS - INSERTION_STYLE_ROWS} -- zero units measured is "
        "UNMEASURED, never PASS, and a shrunken denominator is the same "
        "failure one step milder"
    )
    assert survivors == [], (
        f"non-insertion anchor survivor(s) over {measured} eligible row(s) "
        f"of {TOTAL_ROWS} / {TOTAL_MODS} module(s): " + ", ".join(survivors)
    )


def test_must_pass_deletion_style_row_is_fresh_pristine_and_applied(tmp_path):
    """MUST_PASS shape control: replacement is a STRICT SUBSTRING of its own
    anchor.

    The shipped count is pinned by DELETION_STYLE_ROWS. Its single
    occurrence of `replacement` in a pristine source IS the anchor's own
    occurrence, so a predicate demanding `count(replacement) == 0` can
    never accept a pristine file carrying such a row -- that defect
    reddened the real corpus and is what this leg pins. Synthetic corpus
    over a tmp_path source.
    """
    mutate = _load_mutate()
    anchor = "    fired = res.blocking\n    named = res.named\n"
    repl = "    fired = res.blocking\n"
    assert repl in anchor and repl != anchor, (
        "fixture defect: this leg must exercise a replacement that is a "
        "STRICT substring of its anchor, or it tests nothing"
    )
    pristine = (
        "def gate(res):\n"
        "    fired = res.blocking\n"
        "    named = res.named\n"
        "    return fired and named\n"
    )
    rows = [
        _row("deletion-style", anchor, repl),
        _row("plain-row", "    return fired and named\n", "    return True\n"),
    ]
    assert pristine.count(repl) == 1 and pristine.count(anchor) == 1, (
        "fixture defect: the pristine source must carry the anchor once and "
        "the replacement exactly once, inside it"
    )
    _green_in_every_state(mutate, tmp_path, "deletion", pristine, rows)


def test_must_pass_insertion_style_row_is_fresh_pristine_and_applied(tmp_path):
    """MUST_PASS shape control: the ANCHOR is a strict substring of its own
    REPLACEMENT.

    The shipped count is pinned by INSERTION_STYLE_ROWS. Applying such a
    row leaves its anchor still resolving exactly once, so the applied
    state is indistinguishable from pristine by anchor count -- and that is
    correct, because the gate's question is whether anchors BIND. Refusing
    this shape would be a false alarm on real rows. Synthetic corpus over a
    tmp_path source.
    """
    mutate = _load_mutate()
    anchor = "    total = 0\n"
    repl = "    total = 0\n    total += 1  # planted\n"
    assert anchor in repl and anchor != repl, (
        "fixture defect: this leg must exercise an anchor that is a STRICT "
        "substring of its replacement, or it tests nothing"
    )
    pristine = (
        "def count(xs):\n"
        "    total = 0\n"
        "    for x in xs:\n"
        "        total += len(x)\n"
        "    return total\n"
    )
    rows = [
        _row("insertion-style", anchor, repl),
        _row("plain-row", "    return total\n", "    return -1\n"),
    ]
    states = _green_in_every_state(mutate, tmp_path, "insert", pristine, rows)
    applied = dict(states)["applied:insertion-style"]
    assert applied.count(anchor) == 1, (
        "fixture defect: the whole point of this shape is that the anchor "
        "still binds exactly once in the applied state"
    )


def test_must_pass_rows_sharing_one_identical_anchor_are_fresh(tmp_path):
    """MUST_PASS shape control: several rows against ONE identical anchor.

    Module `core` publishes four rows this way, so applying any one of them
    drives all four anchors to 0x at once. A design that insists on
    identifying a unique applied row reds here; existence of a pristine
    state does not. Synthetic corpus over a tmp_path source.
    """
    mutate = _load_mutate()
    shared = "    return 'full'\n"
    rows = [
        _row("share-a", shared, "    return 'lora'\n"),
        _row("share-b", shared, "    return None\n"),
    ]
    assert rows[0]["anchor"] == rows[1]["anchor"], (
        "fixture defect: these rows must share one IDENTICAL anchor"
    )
    pristine = "def pick(cfg):\n    return 'full'\n"
    _green_in_every_state(mutate, tmp_path, "shared", pristine, rows)


def test_must_pass_nested_anchors_are_fresh_pristine_and_applied(tmp_path):
    """MUST_PASS shape control: one row's anchor NESTED inside another's.

    `parity` and `emit_run_manifest` both publish this, so applying the
    inner row consumes the outer row's anchor as collateral and two rows
    read 0x in a perfectly healthy state. Synthetic corpus over a tmp_path
    source.
    """
    mutate = _load_mutate()
    inner = "        return a + b\n"
    outer = "    if a and b:\n        return a + b\n"
    assert inner in outer and inner != outer, (
        "fixture defect: this leg must exercise one anchor strictly NESTED "
        "inside another, or it tests nothing"
    )
    pristine = "def score(a, b):\n    if a and b:\n        return a + b\n    return 0\n"
    rows = [
        _row("outer-row", outer, "    if True:\n        return a - b\n"),
        _row("inner-row", inner, "        return a * b\n"),
    ]
    _green_in_every_state(mutate, tmp_path, "nested", pristine, rows)


def test_must_fire_indentation_only_drift_is_refused(tmp_path):
    """MUST_FIRE, reasoned to red: the evicted-row shape in miniature.

    The source carries the target line at FOUR leading spaces; the corpus
    row remembers EIGHT. Raw str.count refuses at 0x -- a matcher that
    strips or normalises whitespace reports 1x and greens, and this leg
    keeps that 'robustness' red forever. The fresh row beside it must stay
    un-blamed. Synthetic corpus over a tmp_path source this test wrote.
    """
    mutate = _load_mutate()
    src = tmp_path / "synthetic_gate.py"
    src.write_text("def classify():\n    return 'full'\n", "utf-8")
    table = {
        "synthetic": [
            _row("fresh-row", "    return 'full'", "    return 'lora'"),
            _row("stale-row", "        return 'full'", "        return 'x'"),
        ]
    }
    problems = mutate.check_anchor_freshness(table, {"synthetic": src})
    assert any("stale-row" in p and "0x" in p for p in problems), problems
    assert not any("fresh-row" in p for p in problems), problems


def test_must_fire_anchor_matching_twice_is_refused(tmp_path):
    """MUST_FIRE, reasoned to red: exactly-once is part of the contract.

    An anchor occurring 2x would let the battery replace an arbitrary first
    hit and call it measurement, so n>1 is refused with the count named --
    0x and 2x are both failures, only 1x is fresh.
    """
    mutate = _load_mutate()
    src = tmp_path / "dup.py"
    src.write_text("x = 1\ny = 2\nx = 1\n", "utf-8")
    table = {"dup": [_row("dup-row", "x = 1", "x = 0")]}
    problems = mutate.check_anchor_freshness(table, {"dup": src})
    assert any("dup-row" in p and "2x" in p for p in problems), problems


def _accepted_without_reachability(live, rows):
    """The shipped enumeration with the REACHABILITY clause stripped out.

    Used only as a counterfactual inside the two legs below, so each can
    prove WHICH clause produced its red instead of asserting it.
    """
    cands = [live]
    for r in rows:
        at = live.find(r["replacement"])
        while at != -1:
            cands.append(live[:at] + r["anchor"] + live[at + len(r["replacement"]) :])
            at = live.find(r["replacement"], at + 1)
    return [c for c in cands if all(c.count(r["anchor"]) == 1 for r in rows)]


def test_must_fire_two_rows_applied_at_once_is_refused(tmp_path):
    """MUST_FIRE, reasoned to red: the battery applies ONE row per leg, so a
    source carrying two is a state nothing measured.

    Attribution, stated rather than assumed: this leg reds on EXACTLY-ONCE,
    not on reachability -- no candidate here binds both anchors, which the
    counterfactual below pins. It is therefore not a control on the
    reachability clause; that one is the self-overlap leg.
    """
    mutate = _load_mutate()
    first = _row("first-row", "a = 1", "a = 0")
    second = _row("second-row", "b = 2", "b = 0")
    live = "a = 0\nb = 0\n"
    doubled = tmp_path / "doubled.py"
    doubled.write_text(live, "utf-8")
    problems = mutate.check_anchor_freshness({"doubled": [first, second]}, {"doubled": doubled})
    assert any("first-row" in p and "second-row" in p for p in problems), problems
    assert _accepted_without_reachability(live, [first, second]) == [], (
        "attribution claim broken: this fixture is supposed to red on "
        "exactly-once alone, so the reachability-stripped predicate must "
        "also accept nothing"
    )


def test_must_fire_self_overlapping_anchor_pins_reachability(tmp_path):
    """MUST_FIRE, reasoned to red, and the ONLY leg that pins reachability.

    Rewriting a replacement back to its anchor normally makes that the
    anchor's sole occurrence, so re-applying restores the same bytes and
    reachability holds by construction. It bites only when the anchor
    SELF-OVERLAPS: here the rewrite yields a candidate whose first
    "x\\nx\\nx\\n" is NOT the one just created, so `str.replace(..., 1)`
    edits a different site and the round trip lands elsewhere.

    The counterfactual is asserted, not asserted-about: exactly-once alone
    ACCEPTS this candidate, and only the reachability clause refuses it. Cut
    that clause and this leg goes green -- which is what makes it a control.
    """
    mutate = _load_mutate()
    row = _row("overlap-row", "x\nx\nx\n", "y\n")
    live = "x\nx\ny\n"
    src = tmp_path / "overlap.py"
    src.write_text(live, "utf-8")
    slack = _accepted_without_reachability(live, [row])
    assert len(slack) == 1, (
        "fixture defect: exactly-once must ACCEPT here, or the leg would "
        f"red for the wrong reason; got {slack!r}"
    )
    assert slack[0].replace(row["anchor"], row["replacement"], 1) != live, (
        "fixture defect: the round trip must MISS, or reachability has nothing to refuse"
    )
    problems = mutate.check_anchor_freshness({"overlap": [row]}, {"overlap": src})
    assert any("overlap-row" in p for p in problems), problems


def test_must_fire_insertion_row_cannot_base_a_second_row(tmp_path):
    """MUST_FIRE against the composition hole: an insertion-style row's
    applied state must not serve as a pristine BASE.

    Such a row still binds its own anchor after it is applied. Accept that
    state as pristine and the gate's one-row budget silently becomes two,
    certifying a source carrying BOTH rows -- a state nothing measured. Two
    shipped rows have this shape, so the hole is reachable in production,
    and `tools/mutate.py` can strand a mutant on SIGTERM beside a live leg.

    The three legal states are asserted green in the same breath, so the
    repair cannot be the useless one of refusing insertion rows outright.
    """
    mutate = _load_mutate()
    ins = _row("ins-row", "a = 1\n", "a = 1\n# planted\n")
    ordinary = _row("ord-row", "b = 2\n", "b = 0\n")
    rows = [ins, ordinary]
    assert ins["anchor"] in ins["replacement"], (
        "fixture defect: this leg needs a row that SURVIVES its own "
        "application, or there is no composition to refuse"
    )
    pristine = "a = 1\nb = 2\n"
    both = pristine.replace(ins["anchor"], ins["replacement"], 1).replace(
        ordinary["anchor"], ordinary["replacement"], 1
    )
    src = tmp_path / "compose.py"
    for label, text in [
        ("pristine", pristine),
        ("applied:ins-row", pristine.replace(ins["anchor"], ins["replacement"], 1)),
        ("applied:ord-row", pristine.replace(ordinary["anchor"], ordinary["replacement"], 1)),
    ]:
        src.write_text(text, "utf-8")
        problems = mutate.check_anchor_freshness({"c": rows}, {"c": src})
        assert problems == [], f"{label} is a legal state and must green: {problems}"
    assert both.count(ins["anchor"]) == 1, (
        "fixture defect: the whole hole depends on the insertion row's "
        "anchor STILL binding once in the two-applied state"
    )
    src.write_text(both, "utf-8")
    problems = mutate.check_anchor_freshness({"c": rows}, {"c": src})
    assert any("ins-row" in p or "ord-row" in p for p in problems), (
        "two rows are applied at once and the gate certified it fresh: the "
        f"insertion row was laundered into a pristine base. {problems}"
    )


def test_must_fire_overlap_survivor_is_the_declared_limitation(tmp_path):
    """MUST_FIRE documentation control for a KNOWN limitation, not desired behaviour.

    Literal fixture: anchor "xx" -> replacement "x", ordinary anchor "b\n"
    -> replacement "c\n", pristine "xxxb\n", and live "xxc\n" after both
    applications. The count contract sees the first "xx" in pristine once
    because str.count is non-overlapping; after overlap-delete, "xx" still
    binds once. Today's gate therefore returns the false green asserted
    below.

    Unlike the other MUST_FIRE legs, this one pins the CURRENT wrong verdict
    rather than demanding a problem string immediately: exactly one accepted
    state, ["xxb\n"], and zero problems. That is the declared D1 residual.
    The real-corpus admissibility guard above is what prevents any shipped
    row from reaching this state. If this fixture silently changes class, or
    if a later general closure is proposed, this loud control forces that
    change to be reviewed against the measured 2-of-9 pristine and 15
    applied-state false alarms that refuted the predecessor remedy.
    """
    mutate = _load_mutate()
    rows = [
        _row("overlap-delete", "xx", "x"),
        _row("ordinary", "b\n", "c\n"),
    ]
    pristine = "xxxb\n"
    both = "xxc\n"
    assert (
        pristine.count(rows[0]["anchor"]),
        pristine.count(rows[1]["anchor"]),
    ) == (1, 1), "fixture defect: both anchors must bind exactly once before either row is applied"
    assert pristine.replace("xx", "x", 1).count("xx") == 1, (
        "fixture defect: the overlap row must SURVIVE its own application"
    )
    accepted = mutate._accepted_pristine_states(both, rows)
    assert accepted == ["xxb\n"], (
        "known limitation changed silently: the general overlap-survivor "
        f"composition hole currently accepts {accepted!r}"
    )
    src = tmp_path / "overlap_delete.py"
    src.write_text(both, "utf-8")
    problems = mutate.check_anchor_freshness({"overlap-delete": rows}, {"overlap-delete": src})
    assert problems == [], (
        "known limitation changed silently: today's overlapping two-row "
        f"state is declared as a false green, got {problems}"
    )


def test_must_fire_row_that_edits_nothing_is_refused(tmp_path):
    """MUST_FIRE, reasoned to red: a row whose replacement EQUALS its anchor
    mutates nothing, so the battery would score a guaranteed survivor as a
    real mutant. Zero such rows ship today; this keeps it that way."""
    mutate = _load_mutate()
    src = tmp_path / "noop.py"
    src.write_text("mode = 'strict'\n", "utf-8")
    same = "mode = 'strict'\n"
    problems = mutate.check_anchor_freshness(
        {"noop": [_row("noop-row", same, same)]}, {"noop": src}
    )
    assert any("noop-row" in p and "edits nothing" in p for p in problems), problems


def test_must_fire_candidate_blowup_refuses_rather_than_searching(tmp_path):
    """MUST_FIRE, reasoned to red: the enumeration bound is a refusal, not a
    silent truncation.

    The anchor here binds exactly once, so WITHOUT the bound this source
    would certify fresh -- which is what makes this a control on the bound
    itself rather than on the verdict.
    """
    mutate = _load_mutate()
    src = tmp_path / "flood.py"
    body = "UNIQUE_ANCHOR_LINE\n" + "z = 0\n" * (mutate.MAX_FRESHNESS_CANDIDATES + 40)
    src.write_text(body, "utf-8")
    row = _row("flood-row", "UNIQUE_ANCHOR_LINE\n", "z = 0\n")
    assert body.count(row["anchor"]) == 1, (
        "fixture defect: the anchor must bind exactly once, or this leg "
        "would red on the verdict instead of on the bound"
    )
    problems = mutate.check_anchor_freshness({"flood": [row]}, {"flood": src})
    assert any("candidate enumeration exceeded" in p for p in problems), problems


def test_must_fire_fail_closed_on_every_enumerated_refusal(tmp_path):
    """MUST_FIRE, reasoned to red on each fail-closed enumeration.

    An unmappable module, an unreadable source, a zero-row module, a
    zero-row corpus and a row with an empty anchor must EACH surface as
    their own problem string -- never a pass over nothing, never a guess --
    and the healthy row beside them must stay un-named.
    """
    mutate = _load_mutate()
    live = tmp_path / "live.py"
    live.write_text("x = 1\n", "utf-8")
    good = _row("live-row", "x = 1", "x = 2")
    problems = mutate.check_anchor_freshness(
        {"ghost": [good], "live": [good], "empty": [], "gone": [good]},
        {"live": live, "empty": live, "gone": tmp_path / "missing.py"},
    )
    assert any("ghost" in p and "no source mapping" in p for p in problems), problems
    assert any("gone" in p and "cannot read" in p for p in problems), problems
    assert any("empty" in p and "0 rows" in p for p in problems), problems
    assert not any("live-row" in p for p in problems), problems

    corpus_level = mutate.check_anchor_freshness({}, {})
    assert any("zero-row corpus" in p for p in corpus_level), corpus_level
    still_empty = mutate.check_anchor_freshness({"a": [], "b": []}, {})
    assert any("zero-row corpus" in p for p in still_empty), still_empty

    blank = mutate.check_anchor_freshness(
        {"blank": [_row("blank-row", "", "x = 3")]}, {"blank": live}
    )
    assert any("blank-row" in p and "empty anchor" in p for p in blank), blank
