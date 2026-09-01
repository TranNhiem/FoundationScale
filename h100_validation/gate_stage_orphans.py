#!/usr/bin/env python3
"""Gate: cross-read the build's two membership lists against each other and against disk.

build_h100_plane.sh keeps TWO independent membership lists and, until this gate,
nothing read either against the other:

  * STAGES=( ... )        -- the files the build EXECUTES
  * h100/PUBLISH_SET.txt  -- the files the build SHIPS to the public repository

Four findings were authored, verified and committed -- and then did nothing, because
the fix was in neither list (#136, #188, #189) or ran without shipping (#190). A fix
that is correct, committed, and in no execution list is indistinguishable from a fix
that was never written. This gate closes the CLASS, not those four files: no
finding's filename appears here as a special case.

WHAT IT MEASURES
  Every file in RUN (the STAGES array), SHIP (the .py lines of PUBLISH_SET.txt) or
  CANDIDATE (root *.py files in the declared prefix family, unioned with root *.py
  files named literally anywhere in the build script) is partitioned into exactly
  one bucket:

    RUN_AND_SHIPPED   ok
    RUN_NOT_SHIPPED   RED -- the published build invokes a file a clone does not get
    SHIPPED_NOT_RUN   needs a h100/STAGE_ROLES.tsv declaration whose role the gate
                      measures (gate / library / test / runtime-artifact)
    NEITHER           RED unless declared superseded / developer-tool, and measured so

  A file in RUN or SHIP that does not exist on disk is RED from either end, and a
  stale declaration (naming a missing file, naming a RUN_AND_SHIPPED file that needs
  none, or covering no partitioned file at all) is RED: a declaration list that
  accumulates rows covering nothing is how the next instance of this class hides.

DENOMINATOR
  |RUN ∪ SHIP ∪ CANDIDATE|, printed with the per-bucket counts. Zero parseable
  stages or zero shipped .py entries is UNMEASURED (95), never PASS -- all([]) is
  True, so zero units measured is UNMEASURED, and there is no glob fallback.

WHAT IT DOES NOT COVER
  * Files outside CANDIDATE. This root is a working directory holding dozens of
    earlier-campaign files that are no part of this build plane; the NEITHER bucket
    is scoped to the declared prefix family plus files the build script names, the
    scope is printed, and it is not a claim about every file in the directory.
  * The runtime-artifact role is ATTESTED, NOT MEASURED: the gate checks only that
    the file exists, says so in the output, and counts the rows carrying the role.

--propose
  Writes a candidate STAGE_ROLES.tsv to STDOUT for every file that would need a
  declaration, roles guessed from the same measurements, the reason column left as
  TODO, and exits 0. It never writes the contract file: a detector that authors the
  declaration it then verifies is checking its own homework.

EXIT CODES: 0 PASS, 5 RED, 95 UNMEASURED (the gate could not derive its own
inputs), 96 REFUSE (a control failed, or the declaration file is misconfigured).
"""

import argparse
import re
import sys
from collections import Counter
from pathlib import Path

EXIT_PASS = 0
EXIT_RED = 5
EXIT_UNMEASURED = 95
EXIT_REFUSE = 96

# The declared prefix family for CANDIDATE. Module-level and printed on every run:
# the NEITHER bucket is scoped to this family (plus files the build script names),
# and a scope that is not printed is a claim broader than its evidence.
PREFIX_FAMILY = ("patch_", "apply_", "extract_", "gate_", "emit_")

ROLES = ("gate", "build-driver", "library", "test", "runtime-artifact",
         "superseded", "developer-tool")
# 'build-driver' carries the SAME measurement as 'gate' -- the basename occurs as a
# literal in the build script -- and exists only so the declaration can be true.
# Several directly-invoked files (the stage extractor, the backend splicer) are not
# gates: they produce artifacts rather than adjudicating them. Labelling them 'gate'
# would put a claim in a machine-readable column that the file does not support, which
# is the defect this whole plane exists to refuse -- and it costs one tuple entry to
# avoid. The two roles are deliberately NOT merged: the distinction is invisible to the
# measurement and visible to a reader, which is exactly what a declaration is for.
EXCUSING_NEITHER = ("superseded", "developer-tool")

BUILD_SCRIPT = "build_h100_plane.sh"
PUBLISH_SET_REL = Path("h100") / "PUBLISH_SET.txt"
STAGE_ROLES_REL = Path("h100") / "STAGE_ROLES.tsv"

_FROM_RE = re.compile(r"^\s*from\s+([A-Za-z_][A-Za-z0-9_]*)\s+import\b")
_IMPORT_RE = re.compile(r"^\s*import\s+(.+)$")
_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_DEF_TEST_RE = re.compile(r"^\s*def\s+test_", re.M)


def _basename(path):
    return path.replace("\\", "/").rsplit("/", 1)[-1]


def _unmeasured(msg):
    print("UNMEASURED: " + msg, file=sys.stderr)
    return EXIT_UNMEASURED


def parse_stages(text):
    """Return the STAGES=( ... ) entries as an ordered list (a multiset: one stage
    may legitimately appear twice), or None if the array cannot be located."""
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


def read_ship(root):
    """Return the .py entries of the publish set (relative paths), or None if the
    set is unreadable. Comment blocks explaining deliberate absences are skipped."""
    try:
        text = (root / PUBLISH_SET_REL).read_text(encoding="utf-8")
    except OSError:
        return None
    entries = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("./"):
            line = line[2:]
        if line.endswith(".py"):
            entries.append(line)
    return entries


def read_declarations(path):
    """Return (declarations, error): declarations maps path -> (role, reason);
    error is None or ("unmeasured"|"refuse", message)."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return None, (
            "unmeasured",
            "STAGE ROLES UNREADABLE: %s is missing or unreadable — an unmeasured "
            "contract is not a passing one" % path,
        )
    declarations = {}
    for lineno, raw in enumerate(text.splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        cols = [c.strip() for c in raw.split("\t")]
        if len(cols) != 3:
            return None, (
                "refuse",
                "STAGE_ROLES.tsv row %d has %d column(s), expected 3 "
                "(path<TAB>role<TAB>reason)" % (lineno, len(cols)),
            )
        path_col, role, reason = cols
        if role not in ROLES:
            return None, (
                "refuse",
                "STAGE_ROLES.tsv row %d: unknown role '%s' (declared roles: %s) — "
                "misconfiguration, not a finding" % (lineno, role, ", ".join(ROLES)),
            )
        if path_col in declarations:
            return None, (
                "refuse",
                "STAGE_ROLES.tsv row %d: duplicate row for '%s'" % (lineno, path_col),
            )
        declarations[path_col] = (role, reason)
    return declarations, None


def modules_imported(text):
    """Module names named by import / from-import lines of one source text."""
    mods = set()
    for line in text.splitlines():
        m = _FROM_RE.match(line)
        if m:
            mods.add(m.group(1))
            continue
        m = _IMPORT_RE.match(line)
        if m:
            for token in _TOKEN_RE.findall(m.group(1)):
                if token != "as":
                    mods.add(token)
    return mods


class Refs(object):
    """Every textual reference the partition needs, precomputed by the caller so
    the partition itself never touches the filesystem."""

    def __init__(self, build_refs, importers, def_test):
        self.build_refs = frozenset(build_refs)  # basenames occurring literally in the build script
        self.importers = dict((m, frozenset(s)) for m, s in importers.items())  # module -> ship entries importing it
        self.def_test = frozenset(def_test)  # relative paths containing a `def test_` definition


def derive_disk_and_refs(root, build_text, ship, declarations):
    root_files = sorted(p.name for p in root.glob("*.py") if p.is_file())
    on_disk = set(root_files)
    for entry in ship:
        if (root / entry).is_file():
            on_disk.add(entry)
    for path in declarations:
        if (root / path).is_file():
            on_disk.add(path)

    names = set(root_files)
    names.update(_basename(e) for e in ship)
    names.update(_basename(p) for p in declarations)
    build_refs = set(n for n in names if n in build_text)

    importers = {}
    for entry in ship:
        if entry not in on_disk:
            continue
        try:
            text = (root / entry).read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for mod in modules_imported(text):
            importers.setdefault(mod, set()).add(entry)

    def_test = set()
    for rel in sorted(on_disk):
        try:
            text = (root / rel).read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if _DEF_TEST_RE.search(text):
            def_test.add(rel)

    return on_disk, Refs(build_refs, importers, def_test)


def partition(run, ship, on_disk, refs, declarations):
    """Pure function over (run, ship, on_disk, referenced, declarations). Controls
    mutate copies of these arguments; nothing here reads the repository."""
    run_set = set(run)
    ship_set = set(ship)
    ship_base = set(_basename(e) for e in ship_set)

    root_files = set(f for f in on_disk if "/" not in f and f.endswith(".py"))
    by_prefix = set(f for f in root_files if f.startswith(PREFIX_FAMILY))
    by_ref = set(f for f in root_files if f in refs.build_refs)
    candidate = by_prefix | by_ref

    run_and_shipped = set(f for f in run_set if f in ship_base)
    run_not_shipped = run_set - ship_base
    shipped_not_run = set(e for e in ship_set if _basename(e) not in run_set)
    neither = set(f for f in candidate if f not in run_set and f not in ship_base)

    findings = []  # (file, bucket, reason), all RED
    runtime_attested = set()

    for f in sorted(run_not_shipped):
        findings.append((
            f, "RUN_NOT_SHIPPED",
            "STAGES runs it but PUBLISH_SET.txt does not ship it; the published "
            "build invokes a file a clone does not receive (#190) — if the build "
            "runs it, it ships",
        ))
    for f in sorted(run_set):
        if f not in on_disk:
            findings.append((
                f, "MISSING_FILE",
                "STAGES names a file that does not exist on disk",
            ))
    for e in sorted(ship_set):
        if e not in on_disk:
            findings.append((
                e, "MISSING_FILE",
                "PUBLISH_SET.txt names a file that does not exist on disk",
            ))

    for e in sorted(shipped_not_run):
        decl = declarations.get(e)
        if decl is None:
            findings.append((
                e, "SHIPPED_NOT_RUN",
                "shipped but not a stage, with no STAGE_ROLES.tsv declaration",
            ))
            continue
        role, _reason = decl
        b = _basename(e)
        if role in ("gate", "build-driver"):
            if b not in refs.build_refs:
                findings.append((
                    e, "SHIPPED_NOT_RUN",
                    "declared %s but %s never names it" % (role, BUILD_SCRIPT),
                ))
        elif role == "library":
            mod = b[:-3] if b.endswith(".py") else b
            others = refs.importers.get(mod, frozenset()) - {e}
            if not others:
                findings.append((
                    e, "SHIPPED_NOT_RUN",
                    "declared library but no other shipped file imports module '%s'" % mod,
                ))
        elif role == "test":
            if not (b.startswith("test_") or e in refs.def_test):
                findings.append((
                    e, "SHIPPED_NOT_RUN",
                    "declared test but the basename does not start test_ and the "
                    "file holds no 'def test_' definition",
                ))
        elif role == "runtime-artifact":
            runtime_attested.add(e)  # existence already measured above; role attested, not measured
        else:
            findings.append((
                e, "SHIPPED_NOT_RUN",
                "declared '%s' but the file is SHIPPED; that role requires neither "
                "RUN nor SHIP" % role,
            ))

    for f in sorted(neither):
        decl = declarations.get(f)
        if decl is None:
            findings.append((
                f, "NEITHER",
                "candidate file in neither STAGES nor PUBLISH_SET.txt and "
                "undeclared (#136/#188/#189)",
            ))
            continue
        role, _reason = decl
        if role in EXCUSING_NEITHER:
            if f in refs.build_refs:
                findings.append((
                    f, "NEITHER",
                    "declared '%s' but %s still names it — a '%s' file the build "
                    "still invokes is RED" % (role, BUILD_SCRIPT, role),
                ))
        else:
            findings.append((
                f, "NEITHER",
                "in neither list; only superseded/developer-tool can excuse that, "
                "declared '%s'" % role,
            ))

    partitioned = run_set | ship_set | candidate
    for path in sorted(declarations):
        if path not in on_disk:
            findings.append((
                path, "STALE_DECLARATION",
                "declaration names a file that does not exist",
            ))
        elif _basename(path) in run_and_shipped:
            findings.append((
                path, "STALE_DECLARATION",
                "file is RUN_AND_SHIPPED and needs no declaration; remove the row",
            ))
        elif path not in partitioned and _basename(path) not in partitioned:
            findings.append((
                path, "STALE_DECLARATION",
                "declaration covers no partitioned file",
            ))

    return {
        "buckets": {
            "RUN_AND_SHIPPED": frozenset(run_and_shipped),
            "RUN_NOT_SHIPPED": frozenset(run_not_shipped),
            "SHIPPED_NOT_RUN": frozenset(shipped_not_run),
            "NEITHER": frozenset(neither),
        },
        "candidate": frozenset(candidate),
        "candidate_by_prefix": frozenset(by_prefix),
        "candidate_by_ref": frozenset(by_ref),
        "findings": tuple(sorted(findings)),
        "runtime_attested": frozenset(runtime_attested),
    }


def run_controls(run, ship, on_disk, refs, declarations, real):
    """Four MUST_FIRE drills over mutated argument copies (never the filesystem)
    and one MUST_PASS re-derivation. Returns (lines, failures, fired, total)."""
    lines = []
    failures = []
    fired = 0
    total = 4
    ras = sorted(real["buckets"]["RUN_AND_SHIPPED"])

    if not ras:
        failures.append(
            "MUST_FIRE/RUN_NOT_SHIPPED: no RUN_AND_SHIPPED file exists to mutate — "
            "the control cannot be constructed"
        )
    else:
        victim = ras[0]
        ship2 = set(e for e in ship if _basename(e) != victim)
        p = partition(run, ship2, on_disk, refs, declarations)
        if any(f[0] == victim and f[1] == "RUN_NOT_SHIPPED" for f in p["findings"]):
            fired += 1
            lines.append(
                "MUST_FIRE/RUN_NOT_SHIPPED   mutated: removed %s from SHIP; "
                "observed: RED, and the finding names %s" % (victim, victim)
            )
        else:
            failures.append(
                "MUST_FIRE/RUN_NOT_SHIPPED: removed %s from SHIP but no "
                "RUN_NOT_SHIPPED finding named it" % victim
            )

    synth = "patch_ctl_neither_mustfire.py"
    on_disk2 = set(on_disk)
    on_disk2.add(synth)
    p = partition(run, ship, on_disk2, refs, declarations)
    if any(f[0] == synth and f[1] == "NEITHER" for f in p["findings"]):
        fired += 1
        lines.append(
            "MUST_FIRE/NEITHER           mutated: injected candidate %s in neither "
            "list, undeclared; observed: RED naming it" % synth
        )
    else:
        failures.append(
            "MUST_FIRE/NEITHER: injected %s but no NEITHER finding named it" % synth
        )

    ghost = "patch_ctl_missing_mustfire.py"
    run2 = list(run)
    run2.append(ghost)
    p = partition(run2, ship, on_disk, refs, declarations)
    if any(f[0] == ghost and f[1] == "MISSING_FILE" for f in p["findings"]):
        fired += 1
        lines.append(
            "MUST_FIRE/MISSING_FILE      mutated: added stage %s absent from "
            "ON_DISK; observed: RED naming it" % ghost
        )
    else:
        failures.append(
            "MUST_FIRE/MISSING_FILE: added missing stage %s but no MISSING_FILE "
            "finding named it" % ghost
        )

    if not ras:
        failures.append(
            "MUST_FIRE/STALE_DECLARATION: no RUN_AND_SHIPPED file exists to "
            "declare — the control cannot be constructed"
        )
    else:
        victim = ras[-1]
        decls2 = dict(declarations)
        decls2[victim] = ("developer-tool", "planted by MUST_FIRE/STALE_DECLARATION")
        p = partition(run, ship, on_disk, refs, decls2)
        if any(f[0] == victim and f[1] == "STALE_DECLARATION" for f in p["findings"]):
            fired += 1
            lines.append(
                "MUST_FIRE/STALE_DECLARATION mutated: added a declaration row for "
                "%s (RUN_AND_SHIPPED); observed: RED naming the stale row" % victim
            )
        else:
            failures.append(
                "MUST_FIRE/STALE_DECLARATION: declared %s but no STALE_DECLARATION "
                "finding named it" % victim
            )

    again = partition(run, ship, on_disk, refs, declarations)
    if again == real:
        lines.append(
            "MUST_PASS                   re-derived the real partition after %d "
            "drills; observed: identical to the pre-drill partition, field by field" % total
        )
    else:
        failures.append(
            "MUST_PASS: the real partition re-derived after the drills differs "
            "from the one derived before them"
        )

    return lines, failures, fired, total


def print_report(root, run, ship, on_disk, declarations, real,
                 control_lines, control_failures, fired, total):
    buckets = real["buckets"]
    run_set = set(run)
    candidate = real["candidate"]
    denominator = len(run_set | set(ship) | candidate)
    counts = Counter(run)
    dups = sorted("%s x%d" % (k, v) for k, v in counts.items() if v > 1)

    print("STAGE-ORPHAN GATE — STAGES (what the build runs) cross-read against "
          "PUBLISH_SET.txt (what it ships)")
    print("  root:      %s" % root)
    print("  RUN:       %d stage entr(ies) parsed from STAGES= (%d unique%s)" % (
        len(run), len(run_set),
        "; deliberately repeated: " + ", ".join(dups) if dups else ""))
    print("  SHIP:      %d .py entr(ies) in %s" % (len(ship), PUBLISH_SET_REL))
    print("  ON_DISK:   %d root *.py file(s), non-recursive (earlier-campaign files "
          "outside CANDIDATE are not swept)" % len([f for f in on_disk if "/" not in f]))
    print("  CANDIDATE: %d file(s) — %d by declared prefix family (%s), %d named "
          "literally in %s; union of the two" % (
              len(candidate), len(real["candidate_by_prefix"]),
              " ".join(PREFIX_FAMILY), len(real["candidate_by_ref"]), BUILD_SCRIPT))
    print("  SCOPE:     the NEITHER bucket is scoped to CANDIDATE — it is not a "
          "claim about every file in the directory")
    print("")
    print("BUCKETS — %d file(s) partitioned (|RUN ∪ SHIP ∪ CANDIDATE|):" % denominator)

    role_counts = Counter()
    for e in buckets["SHIPPED_NOT_RUN"]:
        decl = declarations.get(e)
        role_counts[decl[0] if decl else "undeclared"] += 1
    neither_counts = Counter()
    for f in buckets["NEITHER"]:
        decl = declarations.get(f)
        if decl and decl[0] in EXCUSING_NEITHER:
            neither_counts[decl[0]] += 1
        else:
            neither_counts["unexcused"] += 1

    print("  RUN_AND_SHIPPED  %d" % len(buckets["RUN_AND_SHIPPED"]))
    print("  RUN_NOT_SHIPPED  %d   (RED if nonzero)" % len(buckets["RUN_NOT_SHIPPED"]))
    print("  SHIPPED_NOT_RUN  %d   (%s)" % (
        len(buckets["SHIPPED_NOT_RUN"]),
        ", ".join("%s=%d" % (k, role_counts[k]) for k in sorted(role_counts)) or "none"))
    print("  NEITHER          %d   (%s)" % (
        len(buckets["NEITHER"]),
        ", ".join("%s=%d" % (k, neither_counts[k]) for k in sorted(neither_counts)) or "none"))
    print("  NOTE: role runtime-artifact is ATTESTED, NOT MEASURED (existence "
          "only) — %d row(s) carry it" % len(real["runtime_attested"]))

    findings = real["findings"]
    if findings:
        print("")
        print("FINDINGS — %d file(s) in a state the contract forbids:" % len(findings))
        for name, bucket, reason in findings:
            print("  RED  %-44s [%s] %s" % (name, bucket, reason))

    print("")
    print("CONTROLS")
    for line in control_lines:
        print("  " + line)
    for line in control_failures:
        print("  CONTROL FAILED: " + line)

    print("")
    if control_failures:
        print("STAGE-ORPHAN GATE REFUSE — a control failed; a detector that cannot "
              "be shown to fire has not been shown to work, whatever the real "
              "partition says")
    elif findings:
        print("STAGE-ORPHAN GATE RED — %d stage(s), %d shipped module(s), %d "
              "candidate(s); %d finding(s); %d/%d drills fired" % (
                  len(run_set), len(ship), len(candidate), len(findings), fired, total))
    else:
        # "%d stage(s)" was ambiguous against the build's own `BUILD GREEN — N stages`,
        # which counts ENTRIES: this line counts unique FILES, and the two differ whenever
        # a stage is deliberately invoked twice. Two numbers under one word is the #194
        # class; the word now says which one this is.
        print("STAGE-ORPHAN GATE GREEN — %d unique stage file(s) from %d entr(ies), %d "
              "shipped module(s), %d candidate(s); every file in a declared state; "
              "%d/%d drills fired" % (
                  len(run_set), len(run), len(ship), len(candidate), fired, total))


def propose(run, ship, on_disk, refs, declarations):
    """Print a candidate STAGE_ROLES.tsv to stdout and exit 0. The gate never
    writes the contract file itself."""
    real = partition(run, ship, on_disk, refs, declarations)
    buckets = real["buckets"]
    rows = []
    notes = []
    for e in sorted(buckets["SHIPPED_NOT_RUN"]):
        if e in declarations:
            continue
        b = _basename(e)
        mod = b[:-3] if b.endswith(".py") else b
        if b in refs.build_refs:
            role = "gate"
        elif refs.importers.get(mod, frozenset()) - {e}:
            role = "library"
        elif b.startswith("test_") or e in refs.def_test:
            role = "test"
        else:
            role = "runtime-artifact"
        rows.append((e, role))
    for f in sorted(buckets["NEITHER"]):
        if f in declarations:
            continue
        if f in refs.build_refs:
            notes.append(
                "# %s is named by %s but is in neither STAGES nor PUBLISH_SET.txt — "
                "no role can excuse that; add it to the lists instead of declaring it"
                % (f, BUILD_SCRIPT))
            role = "developer-tool"  # placeholder; its measurement stays RED while the build names the file
        else:
            role = "superseded"
        rows.append((f, role))

    print("# Candidate %s proposed by gate_stage_orphans.py --propose" % STAGE_ROLES_REL)
    print("# Review every row before installing it. This gate never writes its own")
    print("# contract file: a detector that authors the declaration it then verifies")
    print("# is checking its own homework.")
    print("# Columns: path<TAB>role<TAB>reason — roles: %s" % ", ".join(ROLES))
    for note in notes:
        print(note)
    for path, role in rows:
        print("%s\t%s\tTODO: state why" % (path, role))
    if not rows:
        print("# (no file currently needs a declaration)")
    return EXIT_PASS


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="repository root containing %s (default: directory containing this "
             "script)" % BUILD_SCRIPT,
    )
    parser.add_argument(
        "--propose",
        action="store_true",
        help="print a candidate STAGE_ROLES.tsv to stdout and exit 0; never "
             "writes the contract file",
    )
    args = parser.parse_args()
    root = args.root

    build_path = root / BUILD_SCRIPT
    try:
        build_text = build_path.read_text(encoding="utf-8")
    except OSError as exc:
        return _unmeasured(
            "BUILD SCRIPT UNREADABLE: %s (%s) — RUN cannot be derived" % (build_path, exc))

    run = parse_stages(build_text)
    if run is None:
        return _unmeasured(
            "STAGES ARRAY NOT FOUND: no 'STAGES=(' ... ')' block in %s — refusing "
            "to fall back to a glob" % BUILD_SCRIPT)
    if not run:
        return _unmeasured(
            "STAGES ARRAY EMPTY: parsed zero entries — zero units measured is "
            "UNMEASURED, not PASS")

    ship = read_ship(root)
    if ship is None:
        return _unmeasured(
            "PUBLISH SET UNREADABLE: %s — SHIP cannot be derived" % (root / PUBLISH_SET_REL))
    if not ship:
        return _unmeasured(
            "PUBLISH SET EMPTY: zero .py entries in %s — an empty shipment covers "
            "vacuously" % PUBLISH_SET_REL)

    declarations, derr = read_declarations(root / STAGE_ROLES_REL)
    if derr is not None:
        kind, msg = derr
        if kind == "unmeasured" and args.propose:
            declarations = {}  # --propose exists to bootstrap the file
        elif kind == "unmeasured":
            return _unmeasured(msg)
        else:
            print("REFUSE: " + msg, file=sys.stderr)
            return EXIT_REFUSE

    on_disk, refs = derive_disk_and_refs(root, build_text, ship, declarations)

    if args.propose:
        return propose(run, ship, on_disk, refs, declarations)

    real = partition(run, ship, on_disk, refs, declarations)
    control_lines, control_failures, fired, total = run_controls(
        run, ship, on_disk, refs, declarations, real)
    print_report(root, run, ship, on_disk, declarations, real,
                 control_lines, control_failures, fired, total)

    if control_failures:
        return EXIT_REFUSE
    return EXIT_RED if real["findings"] else EXIT_PASS


if __name__ == "__main__":
    raise SystemExit(main())
