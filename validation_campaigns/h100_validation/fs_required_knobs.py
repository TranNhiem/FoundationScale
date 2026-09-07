"""Required-knob census for the launch-plane shell sources.

This module replaces two earlier gates that each answered "which environment
variables does the launch plane refuse to start without?" with a single,
half-complete pattern. One gate matched only the `req_env` helper; the other
matched only the raw `[[ -n ... ]] || refuse` idiom. The attempt to heal the
second gate by widening its pattern so it could also read refusal *messages*
is what produced its worst defect: it grabbed the first ALL-CAPS token out of
the refusal text, so `REFUSE 96: FS_PARTITION is unset` was attributed to a
phantom variable called REFUSE and four real knobs collapsed into it.

The lesson baked into this file is that the two idioms are *different
structures*, not one structure at two widths. Widening one pattern to cover
the other idiom is exactly what broke the previous extractors, so each idiom
(and each exclusion) is a separate named rule (R1, R2, R2neg, R2compound, R3,
R4, R5) with an explicit precedence order. Every name the extractor sees must
land in exactly one bucket -- `required` or `excluded[reason]` -- and anything
structurally a guard site but matched by no rule goes to `unclassified`,
which callers must treat as fatal. Nothing is silently dropped.
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

# A req_env *call*, word-boundary anchored so `xreq_env FOO` is not a site.
_REQ_ENV_CALL_RE = re.compile(r"\breq_env\s+([A-Za-z_][A-Za-z0-9_]*)")
# The req_env *definition* is recognisable syntactically (req_env() / req_env ()),
# never by shape similarity to a call. Its body is skipped wholesale, which is
# what keeps the indirection `${!n:-}` ("name" `!n`) out of the census.
_REQ_ENV_DEF_RE = re.compile(r"\breq_env\s*\(\s*\)")
# A -n test over a quoted direct expansion. `${X:-}` and `${X-}` (no colon) are
# the same intent; both are recognised here and therefore everywhere downstream.
_GUARD_RE = re.compile(r'-n\s+"\$\{(!?)([A-Za-z_][A-Za-z0-9_]*)(:-|-)\}"')
# An `if`/`elif` immediately (modulo whitespace and `!`) before the `[[` that
# contains the match means the test is a condition, not an enforcement.
_CONDITION_RE = re.compile(r"(?:^|[^A-Za-z0-9_])(?:if|elif)\s*!?\s*$")
# A refusal is a refusal because of these tokens, *not* merely because of `||`.
_REFUSAL_RE = re.compile(r"\bfail\s+96\b|\bfs_die\b|\bexit\s+96\b")
# Assignment forms: NAME=, export NAME=, local NAME=, declare -x NAME=.
# Anchored to statement boundaries so `==` comparisons and `VAR=` words inside
# quoted refusal messages do not register as production sites.
_ASSIGNMENT_RE = re.compile(
    r"(?:^|(?<=[;{&|]))\s*"
    r"(?:(?:export|local|readonly|declare|typeset)\s+(?:-[A-Za-z]+\s+)*)?"
    r"([A-Za-z_][A-Za-z0-9_]*)=(?!=)"
)
# Exclusion precedence among the three "structured but not required" verdicts.
# (R4 and R5 outrank even these and are checked first, per the contract.)
_NEGATIVE_ORDER = (
    "R3_conditional",
    "R2neg_nonrefusing_or",
    "R2compound_partial_test",
)


@dataclass
class Knob:
    name: str
    rule: str  # "R1_req_env" | "R2_refuse_on_unset"
    site: str  # "<basename>:<1-based line>"
    message: str | None  # refusal text from the SAME statement, or None for R1


@dataclass
class Census:
    required: dict[str, Knob] = field(default_factory=dict)  # name -> first winning site
    excluded: dict[str, tuple[str, str]] = field(default_factory=dict)  # name -> (rule, evidence)
    sites_seen: int = 0  # denominator: every guard site scanned
    unclassified: list[str] = field(default_factory=list)  # MUST be empty; non-empty = refuse


@dataclass
class _Site:
    """One observed guard site: a verdict candidate not yet subject to precedence."""

    name: str
    rule: str  # a rule name, or "UNCLASSIFIED"
    basename: str
    line: int
    message: str | None
    evidence: str | None

    @property
    def site(self) -> str:
        return f"{self.basename}:{self.line}"


@dataclass
class _Assignment:
    basename: str
    line: int
    text: str


def _logical_lines(text: str) -> list[tuple[str, int]]:
    """Join backslash continuations so a statement's RHS is seen whole.

    The refusal for an R2 test may live on the line(s) after the `||`; without
    this join, the refusal token would appear missing (a false R2neg) or be
    poached by a lookahead window from a neighbouring knob (the old defect).
    """
    logical: list[tuple[str, int]] = []
    buf = ""
    start = 1
    open_stmt = False
    for lineno, phys in enumerate(text.splitlines(), 1):
        stripped = phys.rstrip()
        continued = stripped.endswith("\\")
        piece = stripped[:-1] if continued else phys
        if not open_stmt:
            start = lineno
            buf = piece
            open_stmt = True
        else:
            buf += " " + piece.strip()
        if not continued:
            logical.append((buf, start))
            open_stmt = False
            buf = ""
    if open_stmt:
        logical.append((buf, start))
    return logical


def _collect_assignments(sources: dict[str, str]) -> dict[str, list[_Assignment]]:
    """Measure where each name is produced, in either artifact.

    R5 is defined by measurement over assignment sites, never by a hard-coded
    list of names -- that is what keeps ENROOT_NAME/MASTER_ADDR generic.
    """
    found: dict[str, list[_Assignment]] = {}
    for basename, text in sources.items():
        for lineno, phys in enumerate(text.splitlines(), 1):
            for match in _ASSIGNMENT_RE.finditer(phys):
                found.setdefault(match.group(1), []).append(
                    _Assignment(basename=basename, line=lineno, text=phys.strip())
                )
    return found


def _clip(text: str, limit: int = 160) -> str:
    return re.sub(r"\s+", " ", text).strip()[:limit]


def _clean_rhs(rhs: str) -> str:
    """Normalise a `||` right-hand side: drop the brace group and the stderr
    redirect, collapse whitespace. The text stays scoped to THIS statement."""
    cleaned = rhs.strip()
    if cleaned.startswith("{"):
        cleaned = cleaned[1:]
    cleaned = cleaned.rstrip()
    if cleaned.endswith("}"):
        cleaned = cleaned[:-1]
    cleaned = cleaned.replace(">&2", " ")
    return re.sub(r"\s+", " ", cleaned).strip()


def _classify_bracket(stmt: str, match: re.Match[str], open_idx: int) -> tuple[str, str | None, str | None]:
    """Classify one `-n "${NAME...}"` match known to sit inside a `[[` test.

    Returns (rule, message, evidence). Classification is structural: the same
    operator (`||`, `&&`) means different things depending on whether the -n
    test is the whole bracket expression and whether the RHS refuses.
    """
    if _CONDITION_RE.search(stmt[:open_idx]):
        # validated-if-set / forbidden-if-set / conditional-probe all share one
        # census verdict: not required. We deliberately do not distinguish them.
        return "R3_conditional", None, _clip(stmt)
    close_idx = stmt.find("]]", match.end())
    if close_idx == -1:
        return "UNCLASSIFIED", None, None
    # R2's first mandatory part: the -n test must be the ENTIRE bracket
    # expression, i.e. nothing but whitespace between `[[`, the test, and `]]`.
    whole = not stmt[open_idx + 2 : match.start()].strip() and not stmt[match.end() : close_idx].strip()
    after = stmt[close_idx + 2 :].lstrip()
    if after.startswith("&&"):
        return "R3_conditional", None, _clip(stmt)
    if not whole:
        if after.startswith("||"):
            # The statement refuses, but on a compound condition; unset-ness of
            # NAME alone is not what it enforces. Structure disqualifies it --
            # no special-casing of any particular variable (e.g. HOME).
            return "R2compound_partial_test", None, _clip(stmt)
        return "UNCLASSIFIED", None, None
    if after.startswith("||"):
        rhs = _clean_rhs(after[2:])
        if _REFUSAL_RE.search(rhs):
            return "R2_refuse_on_unset", rhs, None
        return "R2neg_nonrefusing_or", None, rhs
    return "UNCLASSIFIED", None, None


def _first_negative(sites: list[_Site]) -> _Site | None:
    """First (in discovery order) site per negative rule, honouring _NEGATIVE_ORDER."""
    best: dict[str, _Site] = {}
    for site in sites:
        if site.rule in _NEGATIVE_ORDER and site.rule not in best:
            best[site.rule] = site
    for rule in _NEGATIVE_ORDER:
        if rule in best:
            return best[rule]
    return None


def _produced_in_artifact(
    assignments: dict[str, list[_Assignment]], name: str, guard_line: int
) -> str | None:
    """R5 evidence if NAME is assigned anywhere, on a lower line than its guard."""
    for asg in assignments.get(name, []):
        if asg.line < guard_line:
            return f"{asg.basename}:{asg.line}: {asg.text}"
    return None


def extract(sources: dict[str, str]) -> Census:
    """Classify every guard site in `sources` (display basename -> full text).

    Pure: no filesystem, no environment, no argv. Scan order only determines
    what "first site" means; the verdict per name is precedence-driven.
    """
    assignments = _collect_assignments(sources)
    by_name: dict[str, list[_Site]] = {}
    sites_seen = 0
    for basename, text in sources.items():
        for stmt, start in _logical_lines(text):
            if stmt.lstrip().startswith("#"):
                # A COMMENT IS NOT A GUARD SITE. Measured: launch_fs_h100.fixed.sh:708
                # is prose -- "...already validated above this splice point (req_env
                # at :NNN)" -- and the word-anchored call regex captured `at` as a
                # required knob. A knob named `at` would then be demanded of every
                # operator by any doc generated from this census.
                #
                # Scope is exactly what was measured: 8 req_env matches across both
                # artifacts, 1 in a full-line comment, 0 in trailing comments. Trailing
                # comments are therefore NOT handled here, deliberately -- stripping at
                # the first `#` would corrupt `${VAR#prefix}` and `$#`, and writing a
                # real shell tokenizer to cover a case with zero present instances buys
                # nothing. The drill below plants both shapes so the trailing-comment
                # axis is recorded as known-and-unhandled rather than silently absent.
                continue
            if _REQ_ENV_DEF_RE.search(stmt):
                # Definition line skipped entirely: neither a site nor a name.
                continue
            for call in _REQ_ENV_CALL_RE.finditer(stmt):
                # Every name on the line counts; sites may be `;`-separated.
                sites_seen += 1
                by_name.setdefault(call.group(1), []).append(
                    _Site(call.group(1), "R1_req_env", basename, start, None, None)
                )
            for match in _GUARD_RE.finditer(stmt):
                if match.group(1):
                    # Indirect expansion `${!n:-}`: the "name" is !n, never a
                    # knob. Kept out here too, in case a definition spans lines.
                    continue
                open_idx = stmt.rfind("[[", 0, match.start())
                if open_idx == -1 or stmt.rfind("]]", 0, match.start()) > open_idx:
                    # Not inside an open `[[` test (e.g. `-n` inside a message).
                    continue
                sites_seen += 1
                rule, message, evidence = _classify_bracket(stmt, match, open_idx)
                name = match.group(2)
                by_name.setdefault(name, []).append(
                    _Site(name, rule, basename, start, message, evidence)
                )

    required: dict[str, Knob] = {}
    excluded: dict[str, tuple[str, str]] = {}
    unclassified: list[str] = []
    for name, sites in by_name.items():
        # R4 and R5 beat everything, including legitimate-looking refusal sites.
        if name.startswith("SLURM_"):
            excluded[name] = ("R4_scheduler_supplied", sites[0].site)
            continue
        evidence = _produced_in_artifact(assignments, name, min(s.line for s in sites))
        if evidence is not None:
            excluded[name] = ("R5_produced_in_artifact", evidence)
            continue
        # Negatives beat R1/R2: a name guarded conditionally somewhere and
        # enforced elsewhere is still not unconditionally required.
        negative = _first_negative(sites)
        if negative is not None:
            excluded[name] = (negative.rule, negative.evidence or negative.site)
            continue
        winner = next(
            (s for s in sites if s.rule in ("R1_req_env", "R2_refuse_on_unset")), None
        )
        if winner is not None:
            required[name] = Knob(name, winner.rule, winner.site, winner.message)
            continue
        # Structural guard site with no rule verdict: a hard error downstream,
        # never a silently dropped name.
        unclassified.append(name)
    return Census(required, excluded, sites_seen, unclassified)


def _all_names(census: Census) -> set[str]:
    return set(census.required) | set(census.excluded) | set(census.unclassified)


# -- self-test drills ------------------------------------------------------
# Each drill plants the exact pattern its rule matches and asserts the exact
# bucket -- positive first, then a near-miss negative control, so a drill
# cannot "pass" merely because nothing matched at all.


def _drill_r1() -> bool:
    text = (
        'req_env() { local n="$1"; [[ -n "${!n:-}" ]] || fail 96 "env $n unset"; }\n'
        "req_env MODEL_DIR; req_env DATASET_DIR; req_env CONFIG_FILE\n"
        "xreq_env STRAY_KNOB\n"
    )
    census = extract({"launcher.sh": text})
    for name in ("MODEL_DIR", "DATASET_DIR", "CONFIG_FILE"):
        knob = census.required.get(name)
        if knob is None or knob.rule != "R1_req_env" or knob.message is not None:
            return False
        if knob.site != "launcher.sh:2":
            return False
    return "STRAY_KNOB" not in _all_names(census)


def _drill_r1_definition_skipped() -> bool:
    text = 'req_env() { local n="$1"; [[ -n "${!n:-}" ]] || fail 96 "env $n unset"; }\n'
    census = extract({"helpers.sh": text})
    names = _all_names(census)
    return "!n" not in names and "n" not in names and census.sites_seen == 0


def _drill_r2_inline() -> bool:
    text = (
        '[[ -n "${FS_PARTITION:-}" ]] || { echo "REFUSE 96: FS_PARTITION is unset ..."'
        " >&2; exit 96; }\n"
    )
    census = extract({"backend.sh": text})
    knob = census.required.get("FS_PARTITION")
    if knob is None or knob.rule != "R2_refuse_on_unset":
        return False
    if knob.message is None or "REFUSE 96" not in knob.message:
        return False
    # REFUSE is a word in the message, never a variable. An extractor that
    # reports it has attributed a knob to its own error string.
    return "REFUSE" not in _all_names(census) and "FS_PARTITION" not in census.excluded


def _drill_r2_continuation() -> bool:
    text = (
        '[[ -n "${FS_ALLOWED_PATH_ROOTS:-}" ]] || fail 96 \\\n'
        '    "FS_ALLOWED_PATH_ROOTS is unset ..."\n'
        'echo "UNRELATED rationale for a different knob"\n'
    )
    census = extract({"backend.sh": text})
    knob = census.required.get("FS_ALLOWED_PATH_ROOTS")
    if knob is None or knob.rule != "R2_refuse_on_unset":
        return False
    if knob.message is None or "FS_ALLOWED_PATH_ROOTS is unset" not in knob.message:
        return False
    # The message must come from this statement only -- no lookahead window.
    return "UNRELATED" not in knob.message


def _drill_r2neg() -> bool:
    census = extract({"probe.sh": '[[ -n "${MOUNT_PROBE:-}" ]] || return 1\n'})
    verdict = census.excluded.get("MOUNT_PROBE")
    if verdict is None or verdict[0] != "R2neg_nonrefusing_or":
        return False
    if verdict[1] != "return 1":
        return False
    return "MOUNT_PROBE" not in census.required


def _drill_r2compound() -> bool:
    text = '[[ -n "${ROOT_PROBE:-}" && "$ROOT_PROBE" == /* ]] || fs_die 96 "need absolute"\n'
    census = extract({"checks.sh": text})
    verdict = census.excluded.get("ROOT_PROBE")
    if verdict is None or verdict[0] != "R2compound_partial_test":
        return False
    return "ROOT_PROBE" not in census.required


def _drill_r3_if() -> bool:
    text = 'if [[ -n "${SPARSE_KNOB:-}" ]]; then echo "validated-if-set"; fi\n'
    census = extract({"opts.sh": text})
    verdict = census.excluded.get("SPARSE_KNOB")
    if verdict is None or verdict[0] != "R3_conditional":
        return False
    return "SPARSE_KNOB" not in census.required


def _drill_r3_and() -> bool:
    text = '[[ -n "${TUNE_KNOB:-}" ]] && argv+=( --tune "${TUNE_KNOB}" )\n'
    census = extract({"opts.sh": text})
    verdict = census.excluded.get("TUNE_KNOB")
    if verdict is None or verdict[0] != "R3_conditional":
        return False
    return "TUNE_KNOB" not in census.required


def _drill_r3_nocolon_expansion() -> bool:
    text = '[[ -n "${LEGACY_KNOB-}" ]] && argv+=( --legacy )\n'
    census = extract({"opts.sh": text})
    verdict = census.excluded.get("LEGACY_KNOB")
    if verdict is None or verdict[0] != "R3_conditional":
        return False
    return census.sites_seen == 1 and "LEGACY_KNOB" not in census.required


def _drill_r4() -> bool:
    # Even a textbook refusal shape is not an operator knob if the scheduler
    # supplies the variable.
    census = extract({"sched.sh": '[[ -n "${SLURM_DRILL_KNOB:-}" ]] || exit 96\n'})
    verdict = census.excluded.get("SLURM_DRILL_KNOB")
    if verdict is None or verdict[0] != "R4_scheduler_supplied":
        return False
    return "SLURM_DRILL_KNOB" not in census.required


def _drill_r5() -> bool:
    text = (
        'CACHE_ROOT="$(mktemp -d)"\n'
        'render_config "${CACHE_ROOT}"\n'
        '[[ -n "${CACHE_ROOT:-}" ]] || exit 96\n'
        '[[ -n "${LATE_KNOB:-}" ]] || exit 96\n'
        "LATE_KNOB=written-after-the-guard\n"
    )
    census = extract({"sandbox.sh": text})
    verdict = census.excluded.get("CACHE_ROOT")
    if verdict is None or verdict[0] != "R5_produced_in_artifact":
        return False
    if not verdict[1].startswith("sandbox.sh:1:"):
        return False
    # Negative control: an assignment AFTER its guard site must not disqualify.
    return census.required.get("LATE_KNOB") is not None


def _drill_comment_not_a_site() -> bool:
    """Prose mentioning req_env must not mint a knob; real calls must still count.

    Positive control first: the negative half alone would pass on a scanner that
    matched nothing at all.
    """
    text = (
        "  # FS_GPUS_PER_NODE is validated above this splice point (req_env at :123)\n"
        "req_env REAL_KNOB\n"
    )
    census = extract({"launcher.sh": text})
    names = _all_names(census)
    if "at" in names:  # the measured defect
        return False
    if "REAL_KNOB" not in census.required:  # did not merely go blind
        return False
    return census.sites_seen == 1


def _drill_comment_axis_boundary() -> bool:
    """Records, honestly, that a TRAILING comment is not handled.

    This drill asserts the CURRENT behaviour rather than the desired one. It exists
    so the gap is a written-down fact with a test pinning it: if someone later adds
    trailing-comment handling, this drill fails and forces the claim to be updated
    instead of the axis quietly changing meaning. An unhandled case that no test
    mentions is indistinguishable from a case nobody thought of.
    """
    census = extract({"l.sh": 'echo hi   # see req_env PHANTOM at :1\n'})
    return "PHANTOM" in _all_names(census)  # known gap, 0 instances in the real corpus


def _drill_denominator_nonzero() -> bool:
    # A denominator that is a subset of the claim is the old defect; a source
    # with one guard must register one site.
    census = extract({"one.sh": "req_env SOLO_KNOB\n"})
    return census.sites_seen == 1 and "SOLO_KNOB" in census.required


def _drill_unclassified_is_fatal() -> bool:
    # A guard shape no rule buckets (no operator after `]]`) must surface in
    # unclassified rather than be dropped.
    text = '[[ -n "${MYSTERY_KNOB:-}" ]]  # bare probe with no operator\n'
    census = extract({"odd.sh": text})
    if census.unclassified != ["MYSTERY_KNOB"]:
        return False
    if census.sites_seen != 1:
        return False
    return "MYSTERY_KNOB" not in census.required and "MYSTERY_KNOB" not in census.excluded


def _self_test() -> int:
    drills: list[tuple[str, object]] = [
        ("R1", _drill_r1),
        ("R1_definition_skipped", _drill_r1_definition_skipped),
        ("R2_inline", _drill_r2_inline),
        ("R2_continuation", _drill_r2_continuation),
        ("R2neg", _drill_r2neg),
        ("R2compound", _drill_r2compound),
        ("R3_if", _drill_r3_if),
        ("R3_and", _drill_r3_and),
        ("R3_nocolon_expansion", _drill_r3_nocolon_expansion),
        ("R4", _drill_r4),
        ("R5", _drill_r5),
        ("comment_not_a_site", _drill_comment_not_a_site),
        ("comment_axis_boundary", _drill_comment_axis_boundary),
        ("denominator_nonzero", _drill_denominator_nonzero),
        ("unclassified_is_fatal", _drill_unclassified_is_fatal),
    ]
    passed = 0
    for name, drill in drills:
        try:
            ok = bool(drill())  # type: ignore[operator]
        except Exception:
            ok = False
        passed += int(ok)
        print(f"DRILL {name} {'PASS' if ok else 'FAIL'}")
    print(f"SELF-TEST {passed}/{len(drills)}")
    return 0 if passed == len(drills) else 1


def _cmd_census(paths: list[str]) -> int:
    sources: dict[str, str] = {}
    for raw in paths:
        try:
            sources[Path(raw).name] = Path(raw).read_text()
        except OSError as exc:
            print(f"cannot read {raw}: {exc}", file=sys.stderr)
            return 96
    census = extract(sources)
    for knob in census.required.values():
        print(f"{knob.name}  {knob.rule}  {knob.site}")
    by_reason: dict[str, list[tuple[str, str]]] = {}
    for name, (reason, evidence) in census.excluded.items():
        by_reason.setdefault(reason, []).append((name, evidence))
    for reason, members in by_reason.items():
        print(f"{reason}:")
        for name, evidence in members:
            print(f"  {name}  {evidence}")
    if census.unclassified:
        # A census with unclassified names is refused, not reported.
        print(f"UNCLASSIFIED: {', '.join(census.unclassified)}")
    print(f"SITES {census.sites_seen}  REQUIRED {len(census.required)}  EXCLUDED {len(census.excluded)}")
    return 1 if census.unclassified else 0


def main(argv: list[str]) -> int:
    if argv == ["--self-test"]:
        return _self_test()
    if len(argv) >= 2 and argv[0] == "--census":
        return _cmd_census(argv[1:])
    print(
        "usage: fs_required_knobs.py --self-test | --census <file.sh> [<file2.sh> ...]",
        file=sys.stderr,
    )
    return 96


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
