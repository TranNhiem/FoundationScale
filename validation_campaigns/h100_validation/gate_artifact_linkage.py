#!/usr/bin/env python3
"""Verify the generated plane is not only well-formed but COMPOSABLE: that the artifacts the
build ships can find each other by name at run time.

#142 was measured on real hardware: an 8xH100 submit died on the launcher's FIRST executable
line --

    FATAL: fs_container_backend.sh not readable at <dir>/fs_container_backend.sh

The launcher sources a sibling by the upstream's literal filename. The build deliberately
ships that artifact as fs_container_backend.bound.sh: the .bound suffix is what keeps the
generated tree and the upstream tree from sharing names, which is how #136 and #137 happened,
so moving the ARTIFACT to the reference was considered and rejected -- the reference must move
to the artifact. Both files were individually well-formed. 18 stages green, parse gate green,
drift gate green, input-partition gate green, unit suite green -- and the plane could not
start, because every gate checked the NODES and none checked the EDGES. A launch plane is a
directed graph: artifacts are nodes, and run-time filename references between them are edges.
Build-time well-formedness is not deploy-time composability. This gate checks the edges.

What it measures:

  L1  the node set is PARSED from build_h100_plane.sh's artifact assignments, never restated
      here -- two hand-kept lists of names that must agree is exactly the drift this project
      keeps finding
  L2  every declared node is present on disk to be scanned
  L3  every extracted literal edge -- shell $SCRIPT_DIR/<name>, source/. of a sibling
      literal, python import of the plane's own fs_* namespace -- targets a declared node.
      Edges decided at run time (variables, substitutions) are extracted but UNRESOLVABLE:
      counted separately, never folded into pass or fail
  L4  extraction is live: zero edges from a non-empty plane is UNMEASURED, not PASS, because
      stale patterns are precisely how #142 survived 18 green stages (all([]) is True)

Exit codes: 0 every literal edge resolves; 5 a BROKEN edge or an absent node; 95 UNMEASURED --
unreadable build, implausible parse, stale extractor, or the controls below failing. 95 is a
declared third state; it is neither laundered into green nor reported as a defect in the tree.

Controls (run on every invocation, in a tempdir, no pytest, real tree untouched):
  C1 MUST_FIRE    the exact #142 shape, asserted to return 5 -- observed red, not assumed
  C2 MUST_PASS    a fully linked plane, asserted to return 0
  C3 MUST_ABSTAIN an edge-free plane, asserted to return 95 rather than 0
A control that misbehaves means the gate itself is unverified, and an unverified detector
issues no verdict on the real tree.
"""

from __future__ import annotations

import ast
import pathlib
import re
import sys
import tempfile
from typing import NamedTuple

ROOT = pathlib.Path(__file__).resolve().parent
BUILD = ROOT / "build_h100_plane.sh"

# Below this many parsed "VAR=$GEN/<name>" assignments the parse is assumed stale (variable
# renamed, block moved into a function) rather than the plane having shrunk -- today's build
# declares six. A parse that finds zero nodes would find zero broken edges and report green:
# the all([]) trap wearing this gate's face, so the gate abstains below the floor instead.
MIN_DECLARED = 3

GEN_LINE = re.compile(r"^GEN=(\S+)\s*$", re.M)
ASSIGNMENT = re.compile(r"^[A-Z][A-Z0-9_]*=\$\{?GEN\}?/([^\s#]+)\s*$", re.M)

SCRIPTDIR_TOKEN = re.compile(r"""\$\{?SCRIPT_DIR\}?/("[^"]*"|'[^']*'|[^\s"']+)""")
SOURCE_LINE = re.compile(r"""^\s*(?:source|\.)\s+("[^"]*"|'[^']*'|\S+)""")
_SCRIPT_DIR_PREFIXES = ("${SCRIPT_DIR}/", "$SCRIPT_DIR/")


class Assessment(NamedTuple):
    code: int
    edges: int
    resolved: tuple[tuple[str, str], ...]
    broken: tuple[tuple[str, str], ...]
    unresolvable: tuple[tuple[str, str], ...]
    missing: tuple[str, ...]


def parse_declared(build_text: str) -> set[str] | None:
    """Artifact basenames the build declares, or None when the parse is implausible."""
    if not GEN_LINE.search(build_text):
        return None
    names = {
        pathlib.PurePosixPath(m.group(1)).name for m in ASSIGNMENT.finditer(build_text)
    }
    return names if len(names) >= MIN_DECLARED else None


def _classify_shell_token(token: str, declared: set[str]) -> tuple[str, str] | None:
    """Bucket one referenced path as resolved / broken / unresolvable, or None.

    None means the token is not a sibling-relative reference at all (absolute path, or a
    path into a subdirectory): checking those is some other gate's claim, and a label wider
    than the measurement is itself a defect.
    """
    t = token.strip("\"'")
    for prefix in _SCRIPT_DIR_PREFIXES:
        if t.startswith(prefix):
            t = t[len(prefix):]
            break
    if not t:
        return None
    if "$" in t or "`" in t:
        # The target is decided at run time by a variable or command substitution. It is a
        # real edge, it was extracted, and it cannot be checked: the doctrine's third bucket.
        return ("unresolvable", token.strip("\"'"))
    if t.startswith("./"):
        t = t[2:]
    if "/" in t:
        return None
    return ("resolved", t) if t in declared else ("broken", t)


def _shell_edges(name: str, text: str, declared: set[str]) -> list[tuple[str, str]]:
    found: set[tuple[str, str]] = set()

    def take(raw: str) -> None:
        c = _classify_shell_token(raw, declared)
        if c is not None:
            found.add(c)  # (status, normalized target) -- dedupes the source-line re-match

    for line in text.splitlines():
        # Comments are stripped before matching: a prose mention of a filename is not a
        # run-time reference. The split is naive about quoted '#'; a false red is survivable,
        # a false green is the thing this gate exists to prevent.
        code = line.split("#", 1)[0]
        for m in SCRIPTDIR_TOKEN.finditer(code):
            take(m.group(1))
        sm = SOURCE_LINE.match(code)
        if sm:
            take(sm.group(1))
    return sorted(found, key=lambda st: (st[0], st[1]))


def _python_edges(name: str, text: str, declared: set[str]) -> list[tuple[str, str]]:
    try:
        tree = ast.parse(text, filename=name)
    except SyntaxError as e:
        # Fail closed: a node this gate cannot read is a node it cannot vouch for, and an
        # entrypoint with a broken import is exactly the state #133's stage C created.
        return [("broken", f"<unparseable python: {e.msg} at line {e.lineno}>")]
    found: set[tuple[str, str]] = set()
    for node in ast.walk(tree):
        mods: list[str] = []
        if isinstance(node, ast.Import):
            mods = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                # A relative import's target is decided by package layout at run time.
                found.add(("unresolvable", f"<relative import, level={node.level}>"))
            elif node.module:
                mods = [node.module]
        for mod in mods:
            base = mod.split(".")[0]
            # Only the plane's own namespace is claimed here. Deciding whether an arbitrary
            # module name is stdlib / third-party / sibling from a static scan is the exact
            # failure this project already filed -- a classifier that read provenance out of
            # contexts it did not understand. fs_* and every declared name is what #133 bound
            # into the entrypoint, and nobody has ever checked it resolves.
            if f"{base}.py" in declared or base.startswith(("fs_", "test_fs_")):
                target = f"{base}.py"
                found.add(("resolved" if target in declared else "broken", target))
    return sorted(found, key=lambda st: (st[0], st[1]))


def assess(declared: set[str], texts: dict[str, str | None]) -> Assessment:
    """Score one plane. Shared by the real tree and by the controls, so a green control
    demonstrates the production code path, not a copy of it."""
    missing = sorted(n for n in declared if texts.get(n) is None)
    buckets: dict[str, list[tuple[str, str]]] = {
        "resolved": [], "broken": [], "unresolvable": [],
    }
    for name in sorted(declared):
        text = texts.get(name)
        if text is None:
            continue
        edges = _python_edges(name, text, declared) if name.endswith(".py") \
            else _shell_edges(name, text, declared)
        for status, target in edges:
            buckets[status].append((name, target))
    total = sum(len(v) for v in buckets.values())
    # Order matters: zero extractions from a non-empty, fully-present plane means the
    # patterns went stale, so the gate abstains before it ever gets to judge. A known-broken
    # plane (absent node, broken edge) outranks abstention.
    if total == 0 and not missing:
        code = 95
    elif missing or buckets["broken"]:
        code = 5
    elif not buckets["resolved"]:
        # Every extracted edge is run-time-decided; nothing was actually verified.
        code = 95
    else:
        code = 0
    return Assessment(
        code, total,
        tuple(buckets["resolved"]), tuple(buckets["broken"]),
        tuple(buckets["unresolvable"]), tuple(missing),
    )


def _read_text(p: pathlib.Path) -> str | None:
    try:
        return p.read_text("utf-8")
    except (OSError, UnicodeDecodeError):
        return None


# ---- Controls. The MUST_FIRE build text reproduces #142 verbatim in miniature: the launcher
# sources the literal UPSTREAM name while the declared node set carries only the .bound name.

_MUSTFIRE_BUILD = """\
GEN=gen
LAUNCHER=$GEN/launch_fs_h100.fixed.sh
BACKEND=$GEN/fs_container_backend.bound.sh
ENTRY=$GEN/fs_train.fixed.py
"""
_MUSTFIRE_FILES = {
    "launch_fs_h100.fixed.sh": (
        "#!/usr/bin/env bash\n"
        'SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"\n'
        "# the #142 literal: the name the build never produces\n"
        'source "$SCRIPT_DIR/fs_container_backend.sh"\n'
    ),
    "fs_container_backend.bound.sh": "# generated backend\n",
    "fs_train.fixed.py": 'print("entry")\n',
}

_MUSTPASS_BUILD = """\
GEN=gen
LAUNCHER=$GEN/launch_fs_h100.fixed.sh
BACKEND=$GEN/fs_container_backend.bound.sh
ENTRY=$GEN/fs_train.fixed.py
MODELROOT=$GEN/fs_model_root.py
"""
_MUSTPASS_FILES = {
    "launch_fs_h100.fixed.sh": (
        "#!/usr/bin/env bash\n"
        'SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"\n'
        '. "$SCRIPT_DIR/fs_container_backend.bound.sh"\n'
    ),
    "fs_container_backend.bound.sh": "# generated backend\n",
    "fs_train.fixed.py": "import sys\nimport fs_model_root\n",
    "fs_model_root.py": "def resolve():\n    return '.'\n",
}

_MUSTABSTAIN_BUILD = _MUSTFIRE_BUILD
_MUSTABSTAIN_FILES = {
    "launch_fs_h100.fixed.sh": 'echo "no sibling references here"\n',
    "fs_container_backend.bound.sh": "# generated backend, references nothing\n",
    "fs_train.fixed.py": "import argparse\nprint(argparse.__name__)\n",
}


def run_controls() -> list[tuple[str, bool, str]]:
    specs = [
        ("C1 MUST_FIRE: synthetic #142 (source of a sibling the build never produces) returns 5",
         _MUSTFIRE_BUILD, _MUSTFIRE_FILES, 5),
        ("C2 MUST_PASS: fully linked synthetic plane returns 0",
         _MUSTPASS_BUILD, _MUSTPASS_FILES, 0),
        ("C3 MUST_ABSTAIN: edge-free synthetic plane returns 95, not 0",
         _MUSTABSTAIN_BUILD, _MUSTABSTAIN_FILES, 95),
    ]
    results: list[tuple[str, bool, str]] = []
    with tempfile.TemporaryDirectory(prefix="gate_artifact_linkage.") as td:
        base = pathlib.Path(td)
        for i, (label, build_text, files, expected) in enumerate(specs):
            d = base / f"c{i}"
            d.mkdir()
            texts: dict[str, str | None] = {}
            for fn, content in files.items():
                (d / fn).write_text(content, "utf-8")
                texts[fn] = content
            declared = parse_declared(build_text)
            if declared is None:
                results.append((label, False,
                                "the control's own build text failed to parse -- the control "
                                "is defective, which says nothing about the tree"))
                continue
            out = assess(declared, texts)
            results.append((label, out.code == expected,
                            f"expected {expected}, observed {out.code} "
                            f"({out.edges} edge(s), {len(out.broken)} broken)"))
    return results


def _print(checks: list[tuple[str, bool, str]]) -> None:
    for name, passed, detail in checks:
        print(f"  {name}: {'PASS' if passed else 'FAIL'}" + (f" ({detail})" if detail else ""))


def main() -> int:
    controls = run_controls()
    if not all(p for _, p, _ in controls):
        _print(controls)
        print(
            "\nLINKAGE GATE UNVERIFIED — a control above did not behave as asserted. A detector\n"
            "that cannot demonstrate its own red / green / abstain paths has measured nothing,\n"
            "so no verdict on the real tree will be issued by an unverified instrument.",
            file=sys.stderr,
        )
        return 95  # UNMEASURED: the instrument, not the tree

    try:
        build_text = BUILD.read_text("utf-8")
    except (OSError, UnicodeDecodeError):
        print(f"UNMEASURED: {BUILD} unreadable — the node set cannot be parsed, "
              "so no edge can be judged. Run the build, then re-gate.", file=sys.stderr)
        return 95

    declared = parse_declared(build_text)
    if declared is None:
        print(
            f"UNMEASURED: parsed fewer than {MIN_DECLARED} `VAR=$GEN/<name>` assignments from\n"
            f"{BUILD}. Either the build moved its declarations (this parser is stale) or GEN\n"
            "was renamed. Zero parsed nodes would find zero broken edges and report green, so\n"
            "the gate abstains instead.",
            file=sys.stderr,
        )
        return 95

    gen_dir = ROOT / GEN_LINE.search(build_text).group(1)  # type: ignore[union-attr]
    texts = {n: _read_text(gen_dir / n) for n in declared}
    out = assess(declared, texts)

    gates: list[tuple[str, bool, str]] = [
        ("L1 node set parsed from build_h100_plane.sh, never restated", True,
         f"{len(declared)} declared node(s); floor {MIN_DECLARED}"),
        ("L2 every declared node is present on disk to be scanned", not out.missing,
         f"{len(declared) - len(out.missing)}/{len(declared)} present"
         + (f"; ABSENT: {list(out.missing)}" if out.missing else "")),
    ]

    if out.edges == 0 and not out.missing:
        _print(gates + controls)
        print(
            f"\nUNMEASURED: 0 inter-artifact edges extracted from a non-empty plane of "
            f"{len(declared)} artifacts. The extractors' patterns have gone stale — which is\n"
            "precisely how #142 survived 18 green stages. all([]) is True, so an edge-less\n"
            "measurement can never be reported as PASS.",
            file=sys.stderr,
        )
        return 95
    if not out.resolved and not out.broken:
        _print(gates + controls)
        print(
            f"\nUNMEASURED: {out.edges} edge(s) extracted but every one is run-time-decided\n"
            "(variable or command substitution); zero edges were actually checked. Abstention\n"
            "is a declared state — it is laundered into neither green nor red.",
            file=sys.stderr,
        )
        return 95

    literal = len(out.resolved) + len(out.broken)
    gates.append((
        "L3 every literal inter-artifact edge resolves to a declared node",
        not out.broken,
        f"{len(out.resolved)}/{literal} literal edges resolve; "
        f"{len(out.unresolvable)} unresolvable (counted, not judged)"
        + (f"; BROKEN: {[f'{s} -> {t}' for s, t in out.broken]}" if out.broken else ""),
    ))
    # An UNRESOLVABLE edge is the one place the next #142 can hide: the extractor saw a
    # reference, could not decide it, and abstained -- which is correct, but a bare COUNT of
    # abstentions cannot be reviewed. Naming them is the difference between "2 things I did not
    # judge" and "2 things nobody will ever look at". Printed, deliberately not gated: promoting
    # them to failures would punish every legitimate runtime-computed path and pressure the code
    # toward hard-coding, which is the defect facing the other way.
    for src, tok in out.unresolvable:
        print(f"  UNRESOLVABLE (not judged) {src} -> {tok}")

    checks = gates + controls
    ok = all(p for _, p, _ in checks)
    _print(checks)
    print(f"\n  {sum(1 for _, p, _ in checks if p)}/{len(checks)} linkage checks green")
    if not ok:
        print(
            "\nARTIFACT LINKAGE RED — an artifact reaches for a sibling by a literal name the\n"
            "build never produces. That is the #142 state: every node well-formed, the graph\n"
            "broken, death on the first executable line at deploy time while every per-file\n"
            "gate stays green. Fix the REFERENCE to name the declared artifact — the .bound.sh\n"
            "suffix is deliberate (#136/#137): never rename the artifact to suit the reference.",
            file=sys.stderr,
        )
    return 0 if ok else 5


if __name__ == "__main__":
    raise SystemExit(main())
