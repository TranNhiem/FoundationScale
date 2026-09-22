"""Hold every prose mention of the declaration-axes count to one source of truth.

loop.TrainConfig carries a subset of fields -- the declaration axes -- that default
to None, where None means NOT DECLARED (the abstention-is-not-a-claim rule, finding
#342). The count of those axes was once written as an English number word in eight
separate places -- comments in loop.py, a banner in cli.py, a test docstring, and
docs/VERIFICATION_MATRIX.md -- while the block itself had silently grown by
sdp_backend and logging_steps. A number repeated in eight places and enforced in
none is wrong somewhere. This module makes loop.DECLARATION_AXES the single source
of truth and holds every prose site to it.

The prose scanner matches "<word> declaration axes" phrases. Sites whose captured
word is not a number word ("the declaration axes are recorded") state no count,
have nothing to drift, and are ignored deliberately. This file contains no literal
count to find, by construction, so the scan does not exclude it by path.
"""

from __future__ import annotations

import dataclasses
import re
import subprocess
from pathlib import Path

from foundationscale.train.loop import DECLARATION_AXES, TrainConfig, _manifest_payload

_REPO_ROOT = Path(__file__).resolve().parents[2]

# The expected count word is derived as _NUMBER_WORDS[len(DECLARATION_AXES)], never
# written as a literal: this scanner lives inside the corpus it scans, so a hardcoded
# English number word here would itself become a drifting prose site.
_NUMBER_WORDS: tuple[str, ...] = (
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "eleven",
    "twelve",
    "thirteen",
    "fourteen",
    "fifteen",
    "sixteen",
    "seventeen",
    "eighteen",
    "nineteen",
    "twenty",
)

_COUNT_PHRASE = re.compile(r"([A-Za-z]+)\s+(?:\"declaration axes\"|declaration axes)")


def _prose_sites() -> list[tuple[Path, int, str]]:
    result = subprocess.run(
        ["git", "ls-files"],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    # The git index is the repo's denominator: a filesystem walk would pick up
    # untracked build output and count it.
    sites: list[tuple[Path, int, str]] = []
    for rel in result.stdout.split():
        path = Path(rel)
        if path.suffix not in {".py", ".md"}:
            continue
        text = (_REPO_ROOT / path).read_text(encoding="utf-8", errors="replace")
        for lineno, line in enumerate(text.splitlines(), start=1):
            for match in _COUNT_PHRASE.finditer(line):
                sites.append((path, lineno, match.group(1)))
    return sites


def test_every_axis_is_a_trainconfig_field_defaulting_to_none() -> None:
    assert DECLARATION_AXES, (
        "DECLARATION_AXES is empty: a gate that inspected nothing did not pass, "
        "and every other assertion in this module would be vacuously true"
    )
    assert len(set(DECLARATION_AXES)) == len(DECLARATION_AXES), (
        f"DECLARATION_AXES contains duplicate names: {DECLARATION_AXES!r}"
    )
    fields = {field.name: field for field in dataclasses.fields(TrainConfig)}
    for name in DECLARATION_AXES:
        assert name in fields, (
            f"DECLARATION_AXES names {name!r}, which is not a field of TrainConfig"
        )
        field = fields[name]
        assert field.default is not dataclasses.MISSING, (
            f"TrainConfig.{name} has no default; every declaration axis must "
            "default to None so that None can mean NOT DECLARED"
        )
        assert field.default is None, (
            f"TrainConfig.{name} defaults to {field.default!r}, not None; "
            "None means NOT DECLARED (abstention is not a claim, finding #342)"
        )


def test_manifest_records_every_axis_unconditionally() -> None:
    cfg = TrainConfig(
        model="sshleifer/tiny-gpt2",
        dataset="fancyzhx/ag_news",
        output_dir=Path("out/unused"),
        nodes=1,
        gpus_per_node=1,
        # TrainConfig fails closed: exactly one of profile / profile_path /
        # profile_name is required, because a cluster profile is a machine fact
        # with no default. The NAME form is used here because _manifest_payload
        # only records it -- nothing in this test resolves a profile, so naming
        # one keeps the construction legal without inventing a machine.
        profile_name="example",
    )
    payload = _manifest_payload(cfg, stage="train")
    recorded = payload["config"]
    for name in DECLARATION_AXES:
        assert name in recorded, (
            f"manifest config is missing {name!r}: present-carrying-None and absent "
            "are different facts -- present says the operator abstained, absent says "
            "this version of the loop never populated the field"
        )
        assert recorded[name] is None, (
            f"manifest config records {name!r} as {recorded[name]!r} for a config "
            "that declared nothing; an undeclared axis must be recorded as None"
        )


def test_every_prose_site_states_the_derived_count() -> None:
    expected = _NUMBER_WORDS[len(DECLARATION_AXES)]
    # EVERY stale site is collected before failing, deliberately. Asserting inside
    # the loop would name one site per run, and adding an axis touches all of them
    # -- the operator would rerun the suite once per site to discover the next one.
    # The whole point of this gate is that the count lives in several places, so
    # its failure has to describe all of them at once.
    stale = [
        f"  {path}:{lineno} states '{word} declaration axes'"
        for path, lineno, word in _prose_sites()
        if word in _NUMBER_WORDS and word != expected
    ]
    assert not stale, (
        f"{len(stale)} prose site(s) disagree with loop.DECLARATION_AXES, which has "
        f"{len(DECLARATION_AXES)} entries and so reads '{expected}':\n" + "\n".join(stale)
    )


def test_the_prose_scan_is_not_vacuous() -> None:
    sites = _prose_sites()
    counted = [(path, lineno) for path, lineno, word in sites if word in _NUMBER_WORDS]
    files = {path for path, _ in counted}
    # 6 and 3 are floors, not pins: prose sites may legitimately be added, and this
    # control must not have to change when they are. It exists so that a scan which
    # silently matches nothing fails loudly instead of passing by inspecting zero sites.
    assert len(counted) >= 6, (
        f"only {len(counted)} prose sites state a count; the scan is not inspecting "
        "enough of the corpus to mean anything"
    )
    assert len(files) >= 3, (
        f"count-stating sites span only {len(files)} file(s): {sorted(map(str, files))}"
    )
    loop_py = Path("src/foundationscale/train/loop.py")
    assert loop_py in files, f"no count-stating site found in {loop_py}"


def test_no_site_is_invisible_to_the_matcher() -> None:
    """Every line naming the axes must be a line the matcher can READ.

    An anchored scanner reports on the sites it can parse and says nothing about
    the ones it cannot, so a site that drifts out of the anchor's shape leaves the
    gate green while the prose rots -- the gate's coverage silently shrinks instead
    of failing. That is not hypothetical here: writing the doctrine comment this
    module was built to guard put the count word at the end of one line and the
    phrase at the start of the next, and the scanner stopped seeing that site
    while still reporting a healthy count over the ones it could still read.
    (The number word is deliberately not quoted in this paragraph -- naming it
    would plant in this file the very literal the module docstring promises is
    not here.)

    So the invariant is coverage, not correctness: every tracked line CONTAINING the
    phrase must also MATCH the phrase pattern. Re-wrap such a comment and this test
    fails, naming the line, before the drift gate goes quietly blind.
    """
    lines_with_phrase: list[tuple[Path, int, str]] = []
    listed = subprocess.run(
        ["git", "ls-files"], cwd=_REPO_ROOT, capture_output=True, text=True, check=True
    )
    for rel in listed.stdout.splitlines():
        path = Path(rel)
        if path.suffix not in {".py", ".md"}:
            continue
        # This module is excluded, and is the ONLY exclusion: its mentions are
        # regex samples assembled at runtime from DECLARATION_AXES (see the
        # "ni" + "ne" split above), deliberately unmatchable so the scanner does
        # not flag its own fixtures as prose. There is no literal count in here
        # to drift, which is what makes the exclusion safe rather than convenient.
        if path == Path(__file__).resolve().relative_to(_REPO_ROOT):
            continue
        text = (_REPO_ROOT / path).read_text(encoding="utf-8", errors="replace")
        for lineno, line in enumerate(text.splitlines(), start=1):
            if "declaration axes" in line:
                lines_with_phrase.append((path, lineno, line.strip()))

    visible = {(path, lineno) for path, lineno, _ in _prose_sites()}
    invisible = [(p, n, t) for p, n, t in lines_with_phrase if (p, n) not in visible]
    assert not invisible, "lines name the axes but the matcher cannot read them:\n" + "\n".join(
        f"  {p}:{n}  {t}" for p, n, t in invisible
    )
    # Non-vacuity for THIS test: if the phrase vanished from the corpus entirely the
    # loop above would find nothing and the assertion would pass having proved nothing.
    assert len(lines_with_phrase) >= 6, (
        f"only {len(lines_with_phrase)} line(s) name the axes; the coverage check "
        "is inspecting too little of the corpus to mean anything"
    )


def test_the_matcher_matches_its_own_anchor_shape() -> None:
    expected = _NUMBER_WORDS[len(DECLARATION_AXES)]

    bare = f"the {expected} declaration axes"
    match = _COUNT_PHRASE.search(bare)
    assert match is not None and match.group(1) == expected

    # "ni" + "ne" is a sample of the historical defect (the stale prose value), not
    # a claim about today's count; it is split so this source line stays unmatchable
    # and the scanner, which reads this file too, never flags the sample as a site.
    stale = "ni" + "ne"
    quoted = f'{stale} "declaration axes"'
    match = _COUNT_PHRASE.search(quoted)
    assert match is not None and match.group(1) == stale

    capital = f"{expected.capitalize()} declaration axes govern the block"
    match = _COUNT_PHRASE.search(capital)
    assert match is not None and match.group(1) == expected.capitalize()

    other_noun = f"{expected} protocol members"
    assert _COUNT_PHRASE.search(other_noun) is None
