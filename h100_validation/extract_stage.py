#!/usr/bin/env python3
"""Extract a generated build stage from its Kimi envelope, and gate it before it lands.

    extract_stage.py <envelope.json> <output.py> [must-appear-token ...]

Invocations used so far, kept here so each extraction is re-runnable rather than a thing
that happened once:

    extract_stage.py h100/gate133_stage_b.json patch_fs_model_root.py \\
        fs_model_root.py test_fs_model_root.py
    extract_stage.py h100/gate133_stage_c.json patch_fs_train_model_root.py \\
        fs_train.fixed.py resolve_model_root

This was `extract_patch_fs_model_root.py` until 2026-08-31, i.e. one hard-coded envelope
and one hard-coded destination. The second stage needed exactly the same eight controls,
and the honest options were to copy the file or to parameterise it. Copying is how two
detectors drift into disagreeing about what they check.

Everything here is a control on the extraction, not on the fix. The fix is judged by
running the patched module's own test suite afterwards; this file only refuses to put
a file on disk that cannot be what it claims to be. The distinction matters because a
truncated generation is still valid-looking Python: it parses, it imports, and the
first thing you notice is a stage that silently patches nothing.

Gates, each naming a specific way an extraction can look successful while being wrong:

  C1 the envelope holds exactly one task result   (a two-task envelope means the
     prompt was split and this reads only half of it)
  C2 the task reported ok
  C3 generation ran to a stop, not a length cap   (finish_reason=length is the
     truncation case above, and it is invisible to every later check)
  C4 the payload is plausibly a whole file        (>=100 lines)
  C5 no estate literal enters the tree            (this repo is public)
  C6 py_compile clean
  C7 the stage names every token the caller says it must touch. For stage B that was
     the resolver AND its test: a patch that updates the module and leaves the old
     assertion in the test turns the suite red and reads as a bad fix rather than a
     partial one. The tokens are an argument because they are stage-specific; the
     check that SOME set was demanded is not -- passing none is refused below.
  C8 the stage can REFUSE. A build stage that has no path to a non-zero exit cannot
     fail closed, and every stage in this plane is required to leave the tree at the
     last good state rather than half-patched.
"""

from __future__ import annotations

import ast
import json
import pathlib
import py_compile
import re
import sys
import tempfile
import os
from fs_estate_pat import estate_blocklist

ROOT = pathlib.Path(__file__).resolve().parent

# The standing public-repo pattern, applied to generated content before it lands.
#
# #144: this is TWO checks, not one, and merging them made the gate refuse a security control.
# A generated stage carried `REDACT_TOKENS = ("ghp_", ...)` -- a list of prefixes it redacts --
# and C5 called it a leak. The distinction the old single pattern could not draw:
#
#   an ESTATE IDENTIFIER is a disclosure BY ITSELF. A node name has no benign reason to appear
#   in a public repo, so bare matching is correct.
#
#   a SECRET PREFIX is not the secret. `ghp_` is documented by GitHub; what must never ship is
#   `ghp_` followed by a token BODY. Matching the bare prefix flags the vocabulary, and the only
#   way to satisfy it is to delete the redactor -- the gate pressuring the code toward the less
#   safe state.
#
# So: FATAL keeps the identifiers and adds token-SHAPED secrets. NOTICE reports bare prefixes with
# their line and does not fail. Narrowing without the NOTICE tier would trade a false positive for
# a blind spot, which is the worse bargain.
BLOCK = estate_blocklist()
# Bare secret vocabulary: visible, counted, never fatal. Real GitHub PATs carry 36 body
# characters, so the {20,} threshold above is conservative in the safe direction.
NOTICE = re.compile(r"ghp_(?![A-Za-z0-9]{20,})|-----BEGIN[A-Z ]*PRIVATE KEY", re.I)

# #145: #144 fixed the INSTANCE and not the CLASS, and the class bit back one stage later.
#
# #144 said "a secret PREFIX is not a secret" and split `ghp_` out of the fatal tier. But the very
# next generated stage carried
#     BLOCKLIST = ("<corp>", "<org>", "ghp_", "<ip-prefix>", "/work/", "dgx")
# -- its own redaction list -- and C5 called three of those entries a leak. The estate-identifier
# tier had exactly the same defect the secret tier had, and I had walked past it.
#
# The real rule is not "secret vs identifier". It is USE vs MENTION. A string is a disclosure when
# it appears as a VALUE the code operates on, and is not a disclosure when it is DECLARED as
# something to look for. `/work/` inside a tuple named BLOCKLIST is a detector's vocabulary;
# `/work/<org>/<project>/...` assigned to MODEL_DIR is an estate path. Same substring, opposite meaning,
# and only the syntactic position can tell them apart -- which is why this is decided on the AST
# and not by a cleverer regex. A regex about regexes cannot see the difference.
#
# The exemption is deliberately narrow in two independent ways, because "declared as a pattern" is
# otherwise a perfect hiding place:
#   * POSITION -- the literal must sit inside a collection assigned to a name in the redaction
#     vocabulary below. Not "near one", not "in a file that has one".
#   * SHAPE -- it must be a FRAGMENT, not an instance. A real path smuggled into a BLOCKLIST tuple
#     stays fatal. This is the leg that stops the exemption from being a laundering route, and
#     control K3 below is the proof it works.
REDACTION_NAME = re.compile(r"BLOCK|REDACT|DENY|FORBID|BANNED|SCRUB|PATTERN", re.I)


def _is_fragment(s: str) -> bool:
    """A redaction pattern is short and partial; an estate path is long and specific."""
    return len(s) <= 40 and s.count("/") <= 2


def _exempt_lines(src: str) -> set[int]:
    """Lines holding string constants that are DECLARED redaction patterns (use/mention, #145)."""
    try:
        tree = ast.parse(src)
    except SyntaxError:
        # Do not exempt anything on a broken parse. C8 will fail this payload anyway, and a
        # parse failure must never widen what is permitted -- that is the fail-closed rule.
        return set()
    out: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            targets, value = node.targets, node.value
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            targets, value = [node.target], node.value
        else:
            continue
        if not any(isinstance(t, ast.Name) and REDACTION_NAME.search(t.id) for t in targets):
            continue
        for sub in ast.walk(value):
            if isinstance(sub, ast.Constant) and isinstance(sub.value, str) and _is_fragment(sub.value):
                out.add(sub.lineno)
    return out


# The language tag was `(?:python)?`, so a reply wrapped in ```markdown missed this pattern
# entirely and fell through to the verbatim branch -- writing a file whose first line was the
# fence marker. Widened to any lowercase tag. Safe for the markdown path because the anchor
# still demands the fence open at the START of the reply, and a document that opens with a
# heading cannot match.
_WHOLE_FENCE = re.compile(r"\s*```[a-z]*\s*\n(.*?)\n```\s*\Z", re.S)
_ANY_FENCE = re.compile(r"```(?:python|py)?\s*\n(.*?)\n```", re.S)


def _unwrap(body: str, kind: str = ".py") -> tuple[str | None, str]:
    """Isolate the file from a reply that may carry prose around it.

    KIND-AWARE, and it has to be. Everything below assumes a fence is ENVELOPE -- a wrapper the
    model put around the artifact. That is true for a Python payload and flatly false for a
    markdown one, where fenced blocks are the document's own CONTENT. A launch document containing
    several ```bash examples would have hit the multi-fence branch and been REFUSED for being
    well-written. So for markdown the only fence that can be envelope is one wrapping the entire
    reply; inner fences are content and are left alone. The truncation risk that the multi-fence
    rule was really guarding against does not go unwatched -- it moves to the fence-balance gate in
    _syntax_gates, which catches a reply cut off inside a block.

    The original strip was anchored `re.match(...)\\Z`, so it only fired when the ENTIRE reply
    was one fenced block. Measured on a real dispatch: this model can spend its whole budget
    thinking IN the content channel, and a reply of the shape "<reasoning>\\n```python\\n<file>\\n```"
    was therefore written to disk verbatim and died at py_compile. That is fail-closed and so
    not dangerous, but it is the wrong FAILURE: the file was correct and the wrapper was not,
    and the tempting repair -- hand-lifting the code out -- puts the artifact beyond the build,
    which is the one thing this whole plane exists to prevent.

    Returns (payload, description); payload None means REFUSE.
    """
    # The fence regex captures up to the newline BEFORE the closing ```, so it eats the file's
    # final newline. Caught by U1/U2 the first time they ran. Cosmetic in Python, not cosmetic
    # here: several gates in this build compare artifacts BYTE-for-byte to prove idempotence,
    # and a silently-missing trailing newline would make a correct re-run look like a diff.
    def _norm(t: str) -> str:
        return t if t.endswith("\n") else t + "\n"

    m = _WHOLE_FENCE.match(body)
    if m:
        return _norm(m.group(1)), "whole reply was one fenced block"
    if kind == ".md":
        return _norm(body), "markdown: inner fences are content, reply used verbatim"
    blocks = _ANY_FENCE.findall(body)
    if len(blocks) == 1:
        # Report the discarded prose with a DENOMINATOR rather than dropping it silently: a
        # reply that is 90% prose is a signal about the dispatch, not a detail.
        kept = len(blocks[0])
        return _norm(blocks[0]), f"1 fenced block kept ({kept} chars), {len(body) - kept} chars of prose discarded"
    if len(blocks) > 1:
        sizes = [len(b) for b in blocks]
        return None, f"REFUSING: {len(blocks)} fenced blocks in the reply (sizes {sizes}); which one is the file is a guess"
    return _norm(body), "no fence found; reply used verbatim"


def classify(body: str) -> tuple[list[tuple[int, str]], list[tuple[int, str]]]:
    """Split blocklist matches into (fatal, notice) by line, applying the #145 exemption."""
    exempt = _exempt_lines(body)
    fatal: list[tuple[int, str]] = []
    notice: list[tuple[int, str]] = []
    for n, line in enumerate(body.splitlines(), 1):
        if BLOCK.search(line):
            (notice if n in exempt else fatal).append((n, line.strip()[:90]))
        elif NOTICE.search(line):
            notice.append((n, line.strip()[:90]))
    return fatal, notice


# The controls. C5 now makes a JUDGEMENT rather than running a regex, so it has to demonstrate the
# judgement -- an exemption nobody watched fire is indistinguishable from a hole. All three run on
# every invocation; if any misbehaves the tool refuses outright rather than reporting on the real
# payload, because a detector that cannot show its own failure path has measured nothing.
def _syntax_gates(kind: str, body: str) -> list[tuple[str, bool, str]]:
    """Well-formedness, chosen by ARTIFACT KIND rather than assumed to be Python.

    The refusal-path and py_compile pair below is right for a build stage and meaningless for a
    document. Pointing it at a markdown deliverable would paint a well-formed file red for the
    crime of not being Python -- a gate whose LABEL differs from what it MEASURES, which is the
    defect this project keeps finding in other people's code and therefore the one it is least
    entitled to ship itself.

    The tempting repair is to SKIP the syntax gate when the target is not .py. That is worse than
    it looks: a skipped check is UNMEASURED, not PASS, and skipping here would leave unguarded the
    single failure mode that actually occurs on this path -- a reply truncated part-way through a
    fenced block, which is exactly what a length-capped generation produces. So markdown gets its
    own falsifiable pair instead of an exemption.
    """
    if kind == ".md":
        fences = len(re.findall(r"(?m)^\s*```", body))
        headings = len(re.findall(r"(?m)^#{1,6}\s+\S", body))
        return [
            (
                "C7 markdown fenced blocks are balanced",
                fences % 2 == 0,
                f"{fences} fence marker(s); an odd count means the reply was cut inside a block",
            ),
            ("C8 markdown has at least one heading", headings > 0, f"{headings} heading(s)"),
        ]

    can_refuse = bool(re.search(r"return\s+[1-9]|SystemExit\(\s*[1-9]|sys\.exit\(\s*[1-9]", body))
    compiled, cerr = True, ""
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as fh:
        fh.write(body)
        tmp = fh.name
    try:
        py_compile.compile(tmp, doraise=True)
    except Exception as exc:  # noqa: BLE001 -- any compile failure is the same verdict
        compiled, cerr = False, str(exc)[:120]
    return [
        ("C7 stage has a refusal path (can fail closed)", can_refuse, ""),
        ("C8 py_compile clean", compiled, cerr),
    ]


_CONTROLS = (
    ("K1 MUST_FIRE: estate path in ordinary code is fatal", 'MODEL_DIR = "/work/acme-org/proj/w"\n', 1, 0),
    ("K2 MUST_PASS: same substrings inside a redaction list are not", 'BLOCKLIST = ("/work/", "dgx")\n', 0, 1),
    ("K3 MUST_FIRE: a real path hidden in a redaction list is still fatal", 'BLOCKLIST = ("/work/acme-org/proj/public/weights",)\n', 1, 0),
)


_FILE = "import os\nprint(os.getcwd())\n"
# The unwrapper decides WHICH BYTES become the artifact, so it is the highest-consequence
# function in this stage and it had zero controls until the prose-then-code shape was actually
# observed in a dispatch. U3 is the one that matters: two candidate blocks must REFUSE, because
# a wrong pick here is silent and every later gate would then certify the wrong file.
_UNWRAP_CONTROLS = (
    ("U1 MUST_PASS: whole reply is one fenced block", f"```python\n{_FILE}```\n", _FILE),
    ("U2 MUST_PASS: prose then one fenced block", f"We need to write...\n\n```python\n{_FILE}```\n", _FILE),
    ("U3 MUST_FIRE: two fenced blocks is a refusal, not a guess", f"a\n```python\n{_FILE}```\nb\n```python\nx = 1\n```\n", None),
    ("U4 MUST_PASS: unfenced reply is used verbatim", _FILE, _FILE),
)


_DOC = "# Title\n\nRun:\n\n```bash\necho a\n```\n\nThen:\n\n```bash\necho b\n```\n"
# M1 is a REGRESSION control, and it is the reason the kind-aware branch exists at all. Under the
# single rule that preceded it, a launch document containing two shell examples counted as "two
# fenced blocks" and was REFUSED -- the gate rejecting a correct artifact for being well written.
# M2 pins the other edge: a document the model chose to wrap must lose the wrapper, or the file
# begins with a fence marker.
_MD_UNWRAP_CONTROLS = (
    ("M1 MUST_PASS: a document's own fences are content, not envelope", _DOC, _DOC),
    ("M2 MUST_PASS: a wholly ```markdown-wrapped reply is unwrapped", f"```markdown\n{_DOC}```\n", _DOC),
)


_PY_OK = "import sys\n\n\ndef main() -> int:\n    if not sys.argv:\n        return 1\n    return 0\n"
_PY_BAD = _PY_OK + "\n\ndef broken(:\n"  # keeps the refusal path, so only the compile leg may go red
_MD_OK = "# Title\n\nText.\n\n```bash\necho hi\n```\n"
_MD_TRUNC = "# Title\n\n```bash\necho hi\n"  # fence opened and never closed: the truncation shape
_MD_NOHEAD = "Just prose, no structure at all.\n"
# A kind-aware gate is only worth having if BOTH kinds have been seen to reject something. S4 is
# the leg that earns the markdown branch: it is the observed failure mode of a length-capped
# generation, and the exemption this branch replaced would have passed it.
_SYNTAX_CONTROLS = (
    ("S1 MUST_PASS: valid python with a refusal path", ".py", _PY_OK, True),
    ("S2 MUST_FIRE: python that does not compile", ".py", _PY_BAD, False),
    ("S3 MUST_PASS: well-formed markdown", ".md", _MD_OK, True),
    ("S4 MUST_FIRE: markdown truncated inside a fence", ".md", _MD_TRUNC, False),
    ("S5 MUST_FIRE: markdown with no heading", ".md", _MD_NOHEAD, False),
)


def _run_controls() -> bool:
    ok = True
    for name, src, want_fatal, want_notice in _CONTROLS:
        f, n = classify(src)
        good = len(f) == want_fatal and len(n) == want_notice
        ok &= good
        print(f"  {name}: {'PASS' if good else 'FAIL'} (fatal {len(f)}/{want_fatal}, notice {len(n)}/{want_notice})")
    for name, src, want in _UNWRAP_CONTROLS:
        got, why = _unwrap(src)
        good = got == want
        ok &= good
        print(f"  {name}: {'PASS' if good else 'FAIL'} ({why})")
    for name, src, want in _MD_UNWRAP_CONTROLS:
        got, why = _unwrap(src, ".md")
        good = got == want
        ok &= good
        print(f"  {name}: {'PASS' if good else 'FAIL'} ({why})")
    for name, kind, src, want in _SYNTAX_CONTROLS:
        legs = _syntax_gates(kind, src)
        got = all(p for _, p, _ in legs)
        good = got == want
        ok &= good
        red = ",".join(n.split()[0] for n, p, _ in legs if not p) or "none"
        print(f"  {name}: {'PASS' if good else 'FAIL'} (all-green={got}, want={want}, red={red})")
    return ok


def main() -> int:
    argv = sys.argv[1:]
    # The controls used to be reachable ONLY by performing a real extraction, which meant the one
    # thing that certifies this gate could not be run on its own -- and a control suite that is
    # inconvenient to run is a control suite that stops being run. Exposed as a standalone verb so
    # it can sit in the build beside the other gates.
    if argv[:1] == ["--controls"]:
        print("=== extract_stage.py self-certification ===")
        ok = _run_controls()
        print(f"\ncontrols: {ok}")
        return 0 if ok else 6
    if len(argv) < 3:
        # Refuse rather than default. An extraction with no required tokens would pass C7
        # vacuously -- all([]) is True -- which is the exact failure this project is named
        # after, and it would pass it silently on the stage where it matters most.
        print(
            "usage: extract_stage.py <envelope.json> <output.py|output.md> <must-appear-token> ...\n"
            "       extract_stage.py --controls\n"
            "  at least one must-appear token is REQUIRED: C7 over an empty set is vacuous",
            file=sys.stderr,
        )
        return 2
    SRC = pathlib.Path(argv[0])
    DST = pathlib.Path(argv[1])
    NAMES = tuple(argv[2:])
    if not SRC.is_absolute():
        SRC = ROOT / SRC
    if not DST.is_absolute():
        DST = ROOT / DST

    if not SRC.is_file():
        print(f"REFUSING: {SRC} absent — nothing to extract", file=sys.stderr)
        return 3

    if not _run_controls():
        print(
            "\nREFUSING: the C5 classifier failed its own controls. The exemption it applies is\n"
            "unverified, so nothing it says about the payload can be trusted.",
            file=sys.stderr,
        )
        return 4

    rows = json.loads(SRC.read_text("utf-8"))
    gates: list[tuple[str, bool, str]] = []

    gates.append(("C1 envelope holds exactly one task result", len(rows) == 1, f"{len(rows)} result(s)"))
    if not rows:
        _report(gates)
        return 5
    task = rows[0]

    # bool() not truthiness-of-.get: a missing 'ok' key must fail, not read as absent-so-fine.
    gates.append(("C2 task reported ok", bool(task.get("ok")), str(task.get("error"))[:80]))
    fin = task.get("finish_reason")
    gates.append(("C3 generation ran to a stop, not a length cap", fin == "stop", str(fin)))

    body = task.get("content") or ""
    body, unwrap = _unwrap(body, DST.suffix)
    print(f"  envelope: {unwrap}")
    if body is None:
        # Refusing beats guessing. Choosing among several fenced blocks is an act of
        # interpretation, and the failure it invites is silent: the WRONG file is written,
        # every later gate runs against it, and the mistake is attributed to the model.
        gates.append(("C3b payload could be isolated from the reply", False, unwrap))
        _report(gates)
        return 5
    lines = body.count("\n") + 1 if body else 0
    gates.append((f"C4 payload is plausibly a whole file (>=100 lines)", lines >= 100, f"{lines} lines"))

    fatal, notice = classify(body)
    gates.append((
        "C5 no estate literal or token-shaped secret in a USED position",
        not fatal,
        f"{len(fatal)} fatal, {len(notice)} declared-pattern/notice",
    ))
    for n, text in fatal:
        print(f"  FATAL L{n}: {text}")
    # #144/#145: not gates. Printed so an exempted line stays VISIBLE after it stopped being fatal
    # -- every narrowing of this gate must leave a trail, or it becomes a place things can hide.
    for n, text in notice:
        print(f"  NOTICE (not fatal) L{n}: declared pattern or bare secret vocabulary — {text}")

    missing = [n for n in NAMES if n not in body]
    gates.append((
        f"C6 stage names all {len(NAMES)} token(s) it must touch",
        not missing,
        "missing: " + ",".join(missing) if missing else "all present: " + ",".join(NAMES),
    ))

    # THE SHARED HELPER, not a second copy of the Python branch. #149 taught _syntax_gates to
    # pick C7/C8 by artifact kind, and the controls harness (S3/S4/S5) has been certifying that
    # markdown behaviour ever since -- but main() still inlined the .py-only pair, so the LIVE
    # path ran py_compile on a markdown deliverable and painted a correct LAUNCH.md red for the
    # crime of not being Python. Controls green, real path wrong, because the rule existed
    # twice: #150's exact shape inside the tool that extracts #150's fix.
    gates.extend(_syntax_gates(DST.suffix, body))

    ok = _report(gates)
    if not ok:
        print("\nREFUSING TO WRITE — gates above are red", file=sys.stderr)
        return 5

    DST.write_text(body if body.endswith("\n") else body + "\n", "utf-8")
    print(f"\nALL GATES GREEN -> {DST}  ({lines} lines)")
    return 0


def _report(gates: list[tuple[str, bool, str]]) -> bool:
    ok = True
    for name, passed, detail in gates:
        ok &= passed
        suffix = f" ({detail})" if detail else ""
        print(f"  {name}: {'PASS' if passed else 'FAIL'}{suffix}")
    print(f"\n{sum(1 for _, p, _ in gates if p)}/{len(gates)} gates green")
    return ok


if __name__ == "__main__":
    raise SystemExit(main())
