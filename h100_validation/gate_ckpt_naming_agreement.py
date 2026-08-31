#!/usr/bin/env python3
"""Fail the build when the checkpoint writer and the checkpoint adjudicator disagree about
how a checkpoint directory is named.

Finding #150 measured the disagreement by execution: fs_train.fixed.py writes
``checkpoint-step-{step:08d}`` at both save sites, while
fs_ckpt_adjudicator._directory_step fullmatches ``(?:step|checkpoint|ckpt)[_-](\\d+)``, which
sees "checkpoint", then "-", then a tail of "step-00000010" that is not all digits. Result,
measured by calling the parser directly: None on 100% of the directory names this framework
ever produces. A7b -- the only leg that cross-validates the directory against the manifest,
and therefore the leg that would catch a save into the wrong directory or a resume reading
the wrong checkpoint -- ABSTAINS on every real checkpoint, while both artifacts keep passing
their own generated suites, 31 tests green. ABSTAIN is correctly not FAIL, so nothing ever
surfaces the hole.

The two tempting repairs are both wrong, and this gate exists so that neither is mistaken
for a fix:

  * WIDEN THE REGEX. That repairs today's shape and leaves the next shape undetected. The
    two artifacts are emitted by two different build stages with no reason to stay in step,
    so the next drift is a matter of time.
  * SHARE A HELPER that writer and adjudicator both import. See the comment on
    _load_adjudicator_parser: that import is exactly the container bind-visibility
    dependency #146 exists to remove. The honest contract between the two artifacts is
    behavioural: the names the writer renders must be names the adjudicator parses, to the
    same step integer. This gate measures only that.

Checks:

  G1  writer naming formats were actually extracted. Zero extracted sites is UNMEASURED,
      not agreement -- the load-bearing line of the gate, because all([]) is True and a
      loop over zero formats reports 0/0 green for a measurement that never ran.
  G2  every rendered writer name parses under the shipped parser (rejection is one failure
      mode, reported with its denominator).
  G3  every parsed name yields the SAME step integer the writer encoded. Accepting a name
      but returning the wrong step is reported separately from rejecting it, because it is
      the worse failure: a refusal is loud, a wrong number is silently consumed.
  G4  the parser is not vacuously permissive: a name carrying no step must return None.
      G2/G3 only exercise names that contain a step; a parser that claimed every directory
      on disk would sail through both.

Controls, run BEFORE the real check; if any misbehaves the gate REFUSES and runs nothing
real, because a detector that failed its own controls has nothing to say about the code:

  C1  MUST_PASS  a synthetic writer format and a matching parser agree end to end
  C2  MUST_FIRE  a synthetic format the parser rejects is observed going RED
  C3  MUST_FIRE  a parser returning the WRONG step is observed going RED
  C4  MUST_FIRE  zero extracted writer sites is observed becoming UNMEASURED, never PASS
"""

from __future__ import annotations

import ast
import importlib.util
import pathlib
import re
import sys
from typing import Callable

ROOT = pathlib.Path(__file__).resolve().parent
GEN = ROOT / "h100" / "gen"
WRITER = GEN / "fs_train.fixed.py"
ADJUDICATOR = GEN / "fs_ckpt_adjudicator.py"

PASS = "PASS"
FAIL = "FAIL"
UNMEASURED = "UNMEASURED"

# Four outcomes must be distinguishable from the exit code alone: green, measured
# disagreement, declared unmeasured, and an untrusted gate.
EXIT_GREEN = 0
EXIT_UNMEASURED = 3
EXIT_CONTROLS = 4
EXIT_DISAGREEMENT = 5

# 0 is deliberate: an early save at global_step 0 is a real case, and a padded zero
# ("00000000") is precisely where naive step extractors break.
STEP_VALUES = (0, 1, 10, 200, 99999999)

# G4 probes. Both deliberately CARRY a naming token but no digits: a lazy parser that keys
# on the word and invents a number is exactly the vacuous-permissiveness being drilled.
NO_STEP_NAMES = ("checkpoint-no-step", "global-ledger")

_ZEROPAD_FIELD = re.compile(r"0\d")
_NAMING_TOKEN = re.compile(r"checkpoint|step", re.IGNORECASE)

Parser = Callable[[str], int | None]


# EXTRACTION RULE, and what it misses. Chosen: every f-string anywhere in the module whose
# own source text carries "checkpoint" or "step" next to a zero-padded integer field -- NOT
# save_checkpoint dataflow. The known call sites route the f-string through a Path `/` and
# a local (`early_dir = output_root / f"..."`) that is passed on later, and statically
# reconstructing arbitrary Python path construction is exactly the inference the
# input-partition gate refuses to attempt because it produces confident wrong answers. The
# cost of the chosen breadth: a log line that formats a padded step with "step" in its text
# is also extracted, and its renders are fed to the parser. That over-reads, and
# over-reading is the safe side -- a missed writer site is the silent #150 hole, while a
# spurious extra template merely adds rows that may fail loudly. What it still misses:
# directory names assembled by str concatenation, os.path.join, or %-formatting. If the
# writer ever builds a checkpoint path that way, G1's site count and template list are the
# only places a human will see that it happened.
def _spec_constants(spec: ast.expr | None) -> str:
    if spec is None:
        return ""
    if isinstance(spec, ast.JoinedStr):
        # A dynamic width ({w}08d) flattens to its constant parts. No writer should do
        # that; the flattening keeps the extraction honest about what it saw rather than
        # silently skipping the field.
        return "".join(str(v.value) for v in spec.values if isinstance(v, ast.Constant))
    return ""


def _has_zeropadded_field(node: ast.JoinedStr) -> bool:
    return any(
        isinstance(v, ast.FormattedValue)
        and _ZEROPAD_FIELD.search(_spec_constants(v.format_spec))
        for v in node.values
    )


def _as_template(node: ast.JoinedStr) -> str:
    # Every placeholder becomes the step variable, so a format with two numeric fields
    # renders a name that never occurs at runtime. That is the accepted cost of breadth;
    # the alternative -- guessing which FormattedValue is "the step" -- is the narrowing
    # this gate is built to avoid. Conversions (!r/!s) are dropped; no writer formats a
    # step with one.
    parts: list[str] = []
    for v in node.values:
        if isinstance(v, ast.Constant):
            parts.append(str(v.value).replace("{", "{{").replace("}", "}}"))
        elif isinstance(v, ast.FormattedValue):
            parts.append("{__STEP__:" + _spec_constants(v.format_spec) + "}")
    return "".join(parts)


def _extract_writer_templates(source: str) -> dict[str, list[int]]:
    tree = ast.parse(source)
    found: dict[str, list[int]] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.JoinedStr):
            continue
        segment = ast.get_source_segment(source, node) or ""
        if not _NAMING_TOKEN.search(segment):
            continue
        if not _has_zeropadded_field(node):
            continue
        found.setdefault(_as_template(node), []).append(node.lineno)
    return found


def _compare(
    templates: dict[str, list[int]], parser: Parser, step_values: tuple[int, ...]
) -> dict | None:
    # Zero templates must NOT fall through as "0/0 parsed, green": all([]) is True, and the
    # natural loop over zero rendered names would report success for a measurement that
    # never ran -- the same failure shape as A7b abstaining on 100% of real checkpoints.
    # Returning None here is the UNMEASURED declaration, and C4 drills that it holds.
    if not templates:
        return None
    total = 0
    accepted = 0
    rejected: list[str] = []
    wrong: list[str] = []
    for template in templates:
        for step in step_values:
            total += 1
            try:
                name = template.format(__STEP__=step)
            except Exception as exc:
                rejected.append(f"{template!r} @ step {step}: render raised {exc!r}")
                continue
            try:
                got = parser(name)
            except Exception as exc:
                rejected.append(f"{name!r} (step {step}): parser raised {exc!r}")
                continue
            if got is None:
                rejected.append(f"{name!r} (writer encoded step {step})")
            elif got == step:
                accepted += 1
            else:
                wrong.append(f"{name!r} -> {got!r} (writer encoded step {step})")
    return {"total": total, "accepted": accepted, "rejected": rejected, "wrong": wrong}


def _load_adjudicator_parser(path: pathlib.Path) -> Parser:
    # Note what is deliberately absent: no shared naming constant imported by both
    # artifacts that this gate could compare as strings. That helper would make the
    # adjudicator import from the training plane, and the adjudicator is executed INSIDE
    # the container as `python3 "$spec"` -- every import it grows is a bind-visibility
    # dependency, which is precisely what #146 exists to remove. A reader reaching for the
    # deduplication should read #146 first. Hence this gate compares BEHAVIOUR across the
    # two shipped files as they are: rendered names in, parsed steps out.
    spec = importlib.util.spec_from_file_location("fs_ckpt_adjudicator", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot build an import spec for {path}")
    module = importlib.util.module_from_spec(spec)
    # Register BEFORE exec_module. The adjudicator declares @dataclass classes, and on
    # 3.12+ dataclasses resolves each field annotation via sys.modules[cls.__module__],
    # which is None for a module that was built from a spec but never registered -- the
    # import dies AttributeError("'NoneType' object has no attribute '__dict__'") deep
    # inside dataclasses.py, nowhere near the real cause. This exact defect was already
    # found once in this project (fix68 D1, in a test that exec'd tools/mutate.py the
    # same way). Left unfixed here it is worse than a crash: the gate degrades to
    # UNMEASURED on all three real legs and still exits, so the naming disagreement it
    # was written to catch would go unreported behind a plausible-looking abstention.
    sys.modules.setdefault("fs_ckpt_adjudicator", module)
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop("fs_ckpt_adjudicator", None)
        raise
    parser = getattr(module, "_directory_step", None)
    if not callable(parser):
        raise ImportError(f"{path} exposes no callable _directory_step")
    return parser


def _run_controls() -> list[tuple[str, bool, str]]:
    # The controls run the gate's real machinery (extraction, templating, comparison)
    # against SYNTHETIC sources and parsers, so they exercise the same code path the real
    # check will take. A control that lives only in a comment is a claim, not a control.
    controls: list[tuple[str, bool, str]] = []

    # C1 -- the happy path must exist and be observable, or a green real run is
    # meaningless: a comparator that always says "agree" would pass everything.
    c1_templates = _extract_writer_templates('snap = root / f"synthetic-step-{n:06d}"\n')
    c1_shape = re.compile(r"synthetic-step-(\d+)")
    c1_parser: Parser = (
        lambda name: int(m.group(1)) if (m := c1_shape.fullmatch(name)) else None
    )
    r1 = _compare(c1_templates, c1_parser, STEP_VALUES)
    ok1 = bool(r1) and r1["accepted"] == r1["total"] and not r1["rejected"] and not r1["wrong"]
    controls.append((
        "C1 MUST_PASS: a synthetic writer format and a matching parser agree",
        ok1,
        (
            f"{r1['accepted']}/{r1['total']} synthetic names agreed "
            f"({len(c1_templates)} template(s) x {len(STEP_VALUES)} step values)"
            if r1 is not None
            else "comparison refused a non-empty synthetic template set"
        ),
    ))

    # C2 -- a writer format the parser cannot read at all must go RED. Without this drill,
    # the "rejected" bucket is an assertion never seen to fail, which is not the same as
    # known to work.
    c2_shape = re.compile(r"ckpt_(\d+)")
    c2_parser: Parser = (
        lambda name: int(m.group(1)) if (m := c2_shape.fullmatch(name)) else None
    )
    r2 = _compare(
        _extract_writer_templates('snap = root / f"synthetic-step-{n:08d}"\n'),
        c2_parser,
        STEP_VALUES,
    )
    fired2 = (
        r2 is not None
        and r2["accepted"] == 0
        and not r2["wrong"]
        and len(r2["rejected"]) == r2["total"]
        and r2["total"] > 0
    )
    controls.append((
        "C2 MUST_FIRE: a writer format the parser rejects is observed going RED",
        fired2,
        f"{len(r2['rejected'])}/{r2['total']} synthetic names rejected as expected"
        if r2 is not None
        else "comparison refused a non-empty synthetic template set",
    ))

    # C3 -- accepting the name but returning a DIFFERENT step is the failure A7b was built
    # to catch (wrong-directory saves and wrong-checkpoint resumes). The synthetic parser
    # parses faithfully and then adds one, so even step 0 fails: agreement by coincidence
    # on a zero would make this drill silent.
    c3_shape = re.compile(r"synthetic-step(\d+)")
    c3_parser: Parser = (
        lambda name: int(c3_shape.fullmatch(name).group(1)) + 1
        if c3_shape.fullmatch(name)
        else None
    )
    r3 = _compare(
        _extract_writer_templates('snap = root / f"synthetic-step{n:06d}"\n'),
        c3_parser,
        STEP_VALUES,
    )
    fired3 = (
        r3 is not None
        and r3["accepted"] == 0
        and not r3["rejected"]
        and len(r3["wrong"]) == r3["total"]
        and r3["total"] > 0
    )
    controls.append((
        "C3 MUST_FIRE: a parser returning the WRONG step is observed going RED",
        fired3,
        f"{len(r3['wrong'])}/{r3['total']} synthetic names mis-parsed as expected"
        if r3 is not None
        else "comparison refused a non-empty synthetic template set",
    ))

    # C4 -- the doctrine drill: a site-free source must yield zero templates AND the
    # comparator must declare zero templates UNMEASURED. If either half ever regressed,
    # a writer that renamed its formats would read as clean.
    c4_templates = _extract_writer_templates("x = 1\n")
    probe = _compare(c4_templates, lambda name: None, STEP_VALUES)
    fired4 = not c4_templates and probe is None
    controls.append((
        "C4 MUST_FIRE: zero extracted writer sites becomes UNMEASURED, not PASS",
        fired4,
        f"{len(c4_templates)}/0 templates from a site-free source; "
        + ("comparison refused to measure (declared UNMEASURED)"
           if probe is None else "comparison measured an empty set -- the all([]) trap"),
    ))

    return controls


def _print_rows(rows: list[tuple[str, str, str]]) -> None:
    for name, status, detail in rows:
        print(f"  {name}: {status}" + (f" ({detail})" if detail else ""))


def _legend() -> None:
    print()
    print("  exit 0  GREEN — writer and adjudicator agree on checkpoint naming")
    print("  exit 3  UNMEASURED — zero writer sites found, or the adjudicator would not import")
    print("  exit 4  CONTROLS FAILED — this gate failed its own controls and cannot be trusted")
    print("  exit 5  DISAGREEMENT — the writer and the adjudicator disagree about naming")


def main() -> int:
    controls = _run_controls()
    _print_rows([(n, PASS if ok else FAIL, d) for n, ok, d in controls])
    behaved = sum(1 for _, ok, _ in controls if ok)
    print(f"\n  {behaved}/{len(controls)} controls behaved")
    if behaved != len(controls):
        print(
            "\nNAMING-AGREEMENT GATE REFUSING: at least one control misbehaved, so this\n"
            "gate cannot be trusted. Not running the real check -- a detector that failed\n"
            "its own controls has nothing to say about the writer or the adjudicator.",
            file=sys.stderr,
        )
        _legend()
        return EXIT_CONTROLS

    if not WRITER.is_file():
        # Declared UNMEASURED, like the input-partition gate's absent-gen refusal: with no
        # writer source there is nothing to extract from, and nothing <-> agreement.
        print(f"\nREFUSING: {WRITER} absent — run build_h100_plane.sh first", file=sys.stderr)
        _legend()
        return EXIT_UNMEASURED

    try:
        source = WRITER.read_text("utf-8")
        templates = _extract_writer_templates(source)
    except (OSError, SyntaxError) as exc:
        print(
            f"\nREFUSING: {WRITER} could not be read and parsed ({exc!r}) — naming\n"
            "agreement is UNMEASURED, not agreed.",
            file=sys.stderr,
        )
        _legend()
        return EXIT_UNMEASURED

    rows: list[tuple[str, str, str]] = []
    site_lines = sorted({ln for lines in templates.values() for ln in lines})
    rows.append((
        "G1 writer naming formats extracted from fs_train.fixed.py",
        PASS if templates else UNMEASURED,
        (
            f"{len(site_lines)} writer site(s) at line(s) {site_lines}, "
            f"{len(templates)} unique format(s): "
            + "; ".join(repr(t) for t in templates)
            if templates
            else "0 writer sites found — zero measured is UNMEASURED, never agreement"
        ),
    ))
    if not templates:
        _print_rows(rows)
        greens = sum(1 for _, st, _ in rows if st == PASS)
        print(f"\n  {greens}/{len(rows)} naming-agreement checks green")
        print(
            "\nNAMING AGREEMENT UNMEASURED — no writer naming formats were found in\n"
            f"{WRITER}. This is the most important refusal in the gate: a loop over zero\n"
            "writer sites reports 0/0 green, which is exactly how #150 stayed invisible.\n"
            "Either the writer stopped naming checkpoints with a zero-padded 'checkpoint'/\n"
            "'step' f-string, or the extraction rule above went stale. Both are measured\n"
            "events; neither is agreement.",
            file=sys.stderr,
        )
        _legend()
        return EXIT_UNMEASURED

    try:
        parser = _load_adjudicator_parser(ADJUDICATOR)
    except Exception as exc:
        for label in (
            "G2 every rendered checkpoint name is parsed by the adjudicator",
            "G3 every parsed name yields the step the writer encoded",
            "G4 the parser is not vacuously permissive",
        ):
            rows.append((label, UNMEASURED, f"adjudicator would not import: {exc!r}"))
        _print_rows(rows)
        greens = sum(1 for _, st, _ in rows if st == PASS)
        print(f"\n  {greens}/{len(rows)} naming-agreement checks green")
        print(
            f"\nNAMING AGREEMENT UNMEASURED — {ADJUDICATOR} could not be imported to a\n"
            "callable _directory_step. Without the parser there is nothing to compare\n"
            "against, and nothing <-> agreement is the defect this gate exists to prevent.",
            file=sys.stderr,
        )
        _legend()
        return EXIT_UNMEASURED

    result = _compare(templates, parser, STEP_VALUES)
    assert result is not None  # templates non-empty here; the empty case returned above.
    total = result["total"]
    accepted = result["accepted"]
    rejected = result["rejected"]
    wrong = result["wrong"]
    parsed = accepted + len(wrong)

    rows.append((
        "G2 every rendered checkpoint name is parsed by the adjudicator",
        PASS if not rejected else FAIL,
        f"{parsed}/{total} rendered names parsed "
        f"({len(templates)} unique format(s) x {len(STEP_VALUES)} step values)"
        + (f"; REJECTED e.g. {rejected[:3]}" if rejected else ""),
    ))

    compared = parsed
    if wrong:
        g3_status: str = FAIL
    elif compared == 0:
        # Zero comparisons to judge is not a judgement: when nothing parsed, "no wrong
        # steps" would be the all([]) trap wearing a green hat.
        g3_status = UNMEASURED
    else:
        g3_status = PASS
    rows.append((
        "G3 every parsed name yields the step the writer encoded",
        g3_status,
        f"{accepted}/{total} rendered names parsed to the right step"
        + (f"; WRONG-STEP e.g. {wrong[:3]}" if wrong else "")
        + ("; nothing to compare (see G2)" if compared == 0 else ""),
    ))

    verdicts = [(n, parser(n)) for n in NO_STEP_NAMES]
    leaks = [f"{n!r} -> {v!r}" for n, v in verdicts if v is not None]
    rows.append((
        "G4 the parser is not vacuously permissive",
        PASS if not leaks else FAIL,
        f"{len(verdicts) - len(leaks)}/{len(verdicts)} step-less names returned None"
        + (f"; ACCEPTED ANYWAY: {leaks}" if leaks else ""),
    ))

    _print_rows(rows)
    greens = sum(1 for _, st, _ in rows if st == PASS)
    print(f"\n  {greens}/{len(rows)} naming-agreement checks green")

    # Disagreement outranks a leg that merely had nothing to compare: a real RED is
    # measured evidence, and it must not be laundered into UNMEASURED by an empty
    # companion leg.
    if any(st == FAIL for _, st, _ in rows):
        print(
            "\nWRITER/ADJUDICATOR NAMING DISAGREEMENT — the names fs_train.fixed.py writes\n"
            "are not parsed by fs_ckpt_adjudicator.py, or they parse to a different step\n"
            "than the writer encoded. That is the #150 state: A7b abstains on 100% of real\n"
            "checkpoints, and a save into the wrong directory or a resume off the wrong\n"
            "checkpoint surfaces as nothing while every suite stays green. Repair ONE\n"
            "artifact so the BEHAVIOUR matches; do not widen this gate to bless a\n"
            "mismatch, and do not import a shared helper into the adjudicator (see the\n"
            "comment on _load_adjudicator_parser and #146).",
            file=sys.stderr,
        )
        _legend()
        return EXIT_DISAGREEMENT

    if any(st == UNMEASURED for _, st, _ in rows):
        print(
            "\nNAMING AGREEMENT UNMEASURED — reached the comparison with all machinery\n"
            "alive, yet at least one leg had nothing to judge. UNMEASURED is a declared\n"
            "state with its own exit code; it is not PASS.",
            file=sys.stderr,
        )
        _legend()
        return EXIT_UNMEASURED

    _legend()
    return EXIT_GREEN


if __name__ == "__main__":
    raise SystemExit(main())