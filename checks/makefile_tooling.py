#!/usr/bin/env python3
"""Gate: no bare Python-ecosystem tool name may sit in COMMAND position in a Makefile recipe.

WHAT IS MEASURED
    Every line of this repository's Makefile that begins with a literal TAB is a recipe line.
    Recipe lines are joined across trailing-backslash continuations into logical commands, and
    each logical command is tokenised with a quote- and substitution-aware scanner.  A token is
    in COMMAND position when it is the first word of a simple command: at the start of the
    logical line (after make's `@`/`-`/`+` prefixes), after `;`, `&&`, `||`, `|`, `&`, after an
    opening `(`/`` ` ``/`$$(`, after any run of `VAR=value` assignment prefixes, and after a
    transparent shell keyword (`then`, `do`, `env`, `xargs`, ...).  A command word is BARE when
    it names pytest / ruff / mypy / pip / coverage / python[N[.N]] with no path separator and no
    `$` expansion in it.

WHY IT IS A DEFECT (#232, reopened as a class by #247)
    Whether a bare name resolves is a property of the developer's PATH, not of the repository.
    `make lint` that runs `ruff` works on the machine that happens to have ruff installed
    globally and dies with `No module named ruff` -- or worse, silently runs a DIFFERENT ruff --
    on the machine that does not.  Routing every tool through `$(PY) -m <tool>` moves the
    verdict back inside the repository, where a gate can hold it.

WHAT IS NOT MEASURED (deliberately, per #83/#111)
    Nothing here asks which interpreter is running, what is installed, or what is on PATH.  A
    verdict that depends on the machine reading the file is a property of the MACHINE and may be
    reported as information but must never be RED.  This gate never shells out and never
    imports anything outside the standard library.

DENOMINATOR
    The physical recipe lines of the Makefile.  Zero recipe lines is UNMEASURED, never a pass:
    `all([])` is True, so an empty denominator that exits 0 is the exact vacuous truth this
    repository's doctrine exists to refuse.

EXIT CODES
    0   CLEAR      -- at least one recipe line was read and none invokes a bare tool name
    5   RED        -- one or more bare invocations
    95  UNMEASURED -- the Makefile is unreadable, or it contains zero recipe lines
    96  REFUSE     -- an unexpected internal error; a crash is not a verdict
"""

from __future__ import annotations

import re
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

# The Makefile is located relative to THIS FILE, not to the working directory, so that the
# verdict cannot be changed by where the gate is invoked from.
MAKEFILE = Path(__file__).resolve().parents[1] / "Makefile"

EXIT_CLEAR = 0
EXIT_RED = 5
EXIT_UNMEASURED = 95
EXIT_REFUSE = 96

# Tools whose bare invocation is the defect.  These are Python-ecosystem entry points that the
# repository installs into a virtualenv and that therefore may or may not be on any given PATH.
TOOL_NAMES = frozenset({"pytest", "ruff", "mypy", "pip", "coverage"})

# python, python3, python3.11, python3.11.2 -- any interpreter named without a path.
PYTHON_RE = re.compile(r"^python[0-9]*(?:\.[0-9]+)*$")

# `FOO=1 cmd` -- an assignment prefix does not end command position, it precedes it.
ASSIGN_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")

# Words that pass command position through to the next token.  `env pytest -q` and
# `xargs pytest` are pytest invocations; `then`/`do`/`else` open a new simple command.
TRANSPARENT = frozenset(
    {
        "if",
        "then",
        "elif",
        "else",
        "fi",
        "while",
        "until",
        "do",
        "done",
        "for",
        "case",
        "esac",
        "{",
        "}",
        "!",
        "time",
        "nohup",
        "exec",
        "env",
        "command",
        "builtin",
        "xargs",
        "sudo",
    }
)


@dataclass(frozen=True)
class Finding:
    """One bare tool invocation, anchored to the physical line its logical command starts on."""

    line: int
    tool: str
    text: str


def _unquote(word: str) -> str:
    """Strip one matching pair of surrounding quotes, if the result is still a single word.

    `"pytest" tests` really does invoke pytest, so a quoted command word must not be a hiding
    place.  A quoted run that contains whitespace (`'import pytest'` emitted by a printf) is not
    a command name and is returned unchanged, so it can never match a tool.
    """
    if len(word) >= 2 and word[0] == word[-1] and word[0] in "'\"":
        inner = word[1:-1]
        if inner and not any(c.isspace() for c in inner):
            return inner
    return word


def _skip_single_quoted(s: str, i: int) -> int:
    """Return the index just past a single-quoted run starting at `i`. No escapes inside."""
    j = s.find("'", i + 1)
    return len(s) if j < 0 else j + 1


def _skip_double_quoted(s: str, i: int) -> int:
    """Return the index just past a double-quoted run starting at `i`, honouring backslashes."""
    j = i + 1
    n = len(s)
    while j < n:
        if s[j] == "\\":
            j += 2
            continue
        if s[j] == '"':
            return j + 1
        j += 1
    return n


def _skip_make_var(s: str, i: int) -> int:
    """Return the index just past a `$(...)` make expansion starting at `i`, nesting-aware."""
    n = len(s)
    depth = 0
    j = i
    while j < n:
        if s.startswith("$(", j):
            depth += 1
            j += 2
            continue
        if s[j] == ")":
            depth -= 1
            j += 1
            if depth == 0:
                return j
            continue
        j += 1
    return n


def command_words(logical: str) -> list[str]:
    """Return every token of `logical` that sits in shell COMMAND position, in order.

    This is the whole instrument: everything else in the file is bookkeeping around it.  The
    scanner is deliberately conservative -- when it cannot tell, it stops treating the position
    as a command position rather than inventing an invocation.
    """
    words: list[str] = []
    i = 0
    n = len(logical)
    at_cmd = True
    first = True

    while i < n:
        ch = logical[i]

        if ch in " \t":
            i += 1
            continue

        # A quoted run that is NOT at command position is an argument; skip it wholesale so its
        # contents (a printf-generated `pytest.skip(...)`, say) can never be read as code.
        if ch == "'" and not at_cmd:
            i = _skip_single_quoted(logical, i)
            continue
        if ch == '"' and not at_cmd:
            i = _skip_double_quoted(logical, i)
            continue

        # `$$(` is shell command substitution -- make renders it as `$(` for the shell -- and it
        # opens a fresh command position.  `$(` alone is a MAKE variable and does not.
        if logical.startswith("$$(", i):
            i += 3
            at_cmd = True
            continue
        if logical.startswith("`", i):
            i += 1
            at_cmd = True
            continue
        if ch in "()":
            i += 1
            at_cmd = True
            continue
        if logical.startswith("&&", i) or logical.startswith("||", i):
            i += 2
            at_cmd = True
            continue
        if ch in ";|":
            i += 1
            at_cmd = True
            continue
        if ch == "&":
            prev = logical[i - 1] if i else ""
            i += 1
            if prev not in "><&":
                at_cmd = True
            continue

        # An ordinary token.  Quoted runs and `$(...)` expansions are absorbed into it so that
        # `$(PY)` and `"pytest-cov>=5"` stay single tokens.
        parts: list[str] = []
        while i < n:
            c = logical[i]
            if c in " \t;|&`()":
                break
            if logical.startswith("$$(", i):
                break
            if c == "'":
                end = _skip_single_quoted(logical, i)
                parts.append(logical[i:end])
                i = end
                continue
            if c == '"':
                end = _skip_double_quoted(logical, i)
                parts.append(logical[i:end])
                i = end
                continue
            if logical.startswith("$(", i):
                end = _skip_make_var(logical, i)
                parts.append(logical[i:end])
                i = end
                continue
            parts.append(c)
            i += 1
        word = "".join(parts)
        if not word:
            continue

        if at_cmd and ASSIGN_RE.match(word):
            # `FOO=1 pytest` and `output=$$(...)`: the assignment precedes the command.
            first = False
            continue

        if at_cmd:
            if first:
                word = word.lstrip("@-+")
            words.append(word)
            at_cmd = _unquote(word) in TRANSPARENT
        first = False

    return words


def bare_tool(word: str) -> str | None:
    """Return the tool name if `word` is a bare invocation of one, else None.

    A token carrying a path separator (`/usr/bin/python3`, `./ruff`) names a specific file and is
    not PATH-dependent; a token carrying `$` has been routed through a variable.  Neither is the
    defect.
    """
    candidate = _unquote(word)
    if not candidate or "/" in candidate or "$" in candidate:
        return None
    if candidate in TOOL_NAMES:
        return candidate
    if PYTHON_RE.match(candidate):
        return candidate
    return None


def logical_recipe_lines(text: str) -> tuple[list[tuple[int, str]], int]:
    """Split `text` into (first_physical_line, joined_command) pairs plus the physical count.

    The physical count is the DENOMINATOR the verdict is stated over: joining continuations is
    how the command is parsed, but "how much of the Makefile did I read" is a line question.
    """
    lines = text.split("\n")
    out: list[tuple[int, str]] = []
    physical = 0
    i = 0
    while i < len(lines):
        if not lines[i].startswith("\t"):
            i += 1
            continue
        start = i + 1
        parts: list[str] = []
        while True:
            physical += 1
            stripped = lines[i].lstrip("\t")
            continues = (
                stripped.endswith("\\")
                and not stripped.endswith("\\\\")
                and i + 1 < len(lines)
                and lines[i + 1].startswith("\t")
            )
            parts.append(stripped[:-1] if continues else stripped)
            i += 1
            if not continues:
                break
        out.append((start, " ".join(parts)))
    return out, physical


def scan(text: str) -> tuple[list[Finding], int]:
    """Return every bare invocation in `text`, and the physical recipe-line denominator."""
    findings: list[Finding] = []
    logical, physical = logical_recipe_lines(text)
    for line_no, command in logical:
        for word in command_words(command):
            tool = bare_tool(word)
            if tool is not None:
                findings.append(Finding(line=line_no, tool=tool, text=command.strip()[:120]))
    return findings, physical


def evaluate(text: str) -> tuple[int, list[str]]:
    """Turn Makefile text into an exit code and the lines to print. Pure; no I/O."""
    findings, denom = scan(text)
    if denom == 0:
        return EXIT_UNMEASURED, [
            "UNMEASURED makefile_tooling: 0 recipe lines found -- zero units is not a pass "
            "(doctrine 1). A Makefile with no TAB-indented recipe is either not a Makefile or "
            "was not read."
        ]
    if findings:
        out = [
            f"RED makefile_tooling: {len(findings)} bare tool invocation(s) over "
            f"{denom} recipe lines in {MAKEFILE.name}"
        ]
        for f in findings:
            out.append(f"  {MAKEFILE.name}:{f.line}: bare `{f.tool}` in command position")
            out.append(f"      {f.text}")
        out.append(
            "  Route each through the interpreter variable: `$(PY) -m <tool>`. A bare name "
            "resolves against the developer's PATH, which the repository does not control."
        )
        return EXIT_RED, out
    return EXIT_CLEAR, [
        f"CLEAR makefile_tooling: 0 bare tool invocations over {denom} recipe lines "
        f"in {MAKEFILE.name}"
    ]


# --------------------------------------------------------------------------------------------
# Controls.  Each one runs the REAL detector over a fixture; none re-implements the rule, so a
# control cannot pass while the shipped code is wrong.  MUST_FIRE plants the exact pattern and
# asserts the detector fires; MUST_PASS plants a clean or deliberately-confusable fixture and
# asserts it stays silent.
# --------------------------------------------------------------------------------------------

# Verbatim excerpts from the live Makefile.  These are the shapes that made a naive scanner
# wrong: a printf that emits Python source, an echo whose usage string quotes `$(PY)`, and a
# multi-line shell block that captures a subshell.
PRINTF_PYTHON_BLOCK = (
    "\t@printf '%s\\n' \\\n"
    '\t\t\'"""Throwaway MUST_FIRE probe for the FS_FORBID_SKIPS guard (tests/conftest.py).\' \\\n'
    "\t\t'' \\\n"
    "\t\t'import pytest' \\\n"
    "\t\t'' \\\n"
    "\t\t'def test_skip_guard_probe():' \\\n"
    "\t\t'    pytest.skip(\"deliberate skip; an armed skip guard must fail this run\")' \\\n"
    "\t\t> tests/test__skip_guard_probe.py\n"
)

SKIP_GUARD_BLOCK = (
    "\t@set +e; \\\n"
    "\toutput=$$(FS_FORBID_SKIPS=1 $(PY) -m pytest tests/test__skip_guard_probe.py 2>&1); \\\n"
    "\trc=$$?; \\\n"
    "\tset -e; \\\n"
    "\tprintf '%s\\n' \"$$output\"; \\\n"
    "\trm -f tests/test__skip_guard_probe.py; \\\n"
    "\tif [ $$rc -eq 0 ]; then \\\n"
    '\t\techo "skip-guard-probe: FAILED - guard armed but a skipped test exited 0"; \\\n'
    "\t\texit 1; \\\n"
    "\tfi\n"
)

USAGE_ECHO_LINE = (
    '\t@test -n "$(MODULE)" || { echo \'usage: make mutation-module MODULE=<name>   '
    '("$(PY) tools/mutate.py --list" names them)\'; exit 2; }\n'
)


def _fire(text: str, tool: str, count: int = 1) -> bool:
    findings, denom = scan(text)
    return denom > 0 and len(findings) == count and all(f.tool == tool for f in findings)


def _silent(text: str) -> bool:
    findings, denom = scan(text)
    return denom > 0 and findings == []


def c_bare_pytest() -> bool:
    return _fire("target:\n\tpytest tests\n", "pytest")


def c_at_prefixed_ruff() -> bool:
    return _fire("target:\n\t@ruff check src\n", "ruff")


def c_dash_prefixed_mypy() -> bool:
    return _fire("target:\n\t-mypy src\n", "mypy")


def c_assignment_prefix() -> bool:
    return _fire("target:\n\tFOO=1 pytest tests\n", "pytest")


def c_after_and_and() -> bool:
    return _fire("target:\n\tcd x && pip install -e .\n", "pip")


def c_inside_substitution() -> bool:
    # `coverage` opens the subshell; the `pytest` after `-m` is an argument, not a command word.
    return _fire('target:\n\toutput=$$(coverage run -m pytest); echo "$$output"\n', "coverage")


def c_bare_python3() -> bool:
    return _fire("target:\n\tpython3 tools/x.py\n", "python3")


def c_after_pipe() -> bool:
    return _fire("target:\n\techo a | pytest -q\n", "pytest")


def c_after_transparent_word() -> bool:
    return _fire("target:\n\tenv pytest -q\n", "pytest")


def c_quoted_command_word() -> bool:
    return _fire('target:\n\t"pytest" tests\n', "pytest")


def c_continuation_carries_the_defect() -> bool:
    # The bare name is on the SECOND physical line of one logical command; a per-line scanner
    # that never joins continuations would still see it, but one that joins must not lose it.
    return _fire("target:\n\tcd src && \\\n\t\truff check .\n", "ruff")


def c_non_recipe_line_is_not_a_command() -> bool:
    # No TAB: a variable assignment, a target line or a comment is not a recipe.
    text = "# run pytest before pushing\nCMD = pytest tests\ntarget:\n\t$(PY) -m pytest\n"
    return _silent(text)


def c_py_dash_m_is_the_fix() -> bool:
    return _silent("target:\n\t$(PY) -m pytest tests\n")


def c_tool_name_as_argument() -> bool:
    text = 'target:\n\t$(PY) -m pip install "pytest-cov>=5" --extra-index-url https://x/cpu\n'
    return _silent(text)


def c_printf_generated_python_is_data() -> bool:
    return _silent("target:\n" + PRINTF_PYTHON_BLOCK)


def c_usage_string_quoting_py() -> bool:
    return _silent("target:\n" + USAGE_ECHO_LINE)


def c_skip_guard_block() -> bool:
    return _silent("target:\n" + SKIP_GUARD_BLOCK)


def c_absolute_path_is_not_bare() -> bool:
    return _silent("target:\n\t/usr/bin/python3 tools/x.py\n")


def c_zero_recipe_lines_is_unmeasured() -> bool:
    rc, _ = evaluate("PY := python3\n\nall:\n")
    return rc == EXIT_UNMEASURED


def c_unreadable_makefile_is_unmeasured() -> bool:
    # Fail CLOSED: a Makefile that cannot be read is UNMEASURED, not CLEAR.
    with tempfile.TemporaryDirectory() as td:
        missing = Path(td) / "Makefile"
        return run(missing, quiet=True) == EXIT_UNMEASURED


def c_live_tree_denominator_is_nonzero() -> bool:
    # The instrument must actually reach this repository's Makefile from wherever it is invoked.
    if not MAKEFILE.is_file():
        return False
    _, denom = scan(MAKEFILE.read_text(encoding="utf-8", errors="replace"))
    return denom > 0


CONTROLS: list[tuple[str, str, object]] = [
    ("bare pytest at line start", "MUST_FIRE", c_bare_pytest),
    ("@-prefixed ruff", "MUST_FIRE", c_at_prefixed_ruff),
    ("--prefixed mypy", "MUST_FIRE", c_dash_prefixed_mypy),
    ("VAR=value assignment prefix", "MUST_FIRE", c_assignment_prefix),
    ("command word after &&", "MUST_FIRE", c_after_and_and),
    ("command word inside $$( )", "MUST_FIRE", c_inside_substitution),
    ("bare python3", "MUST_FIRE", c_bare_python3),
    ("command word after a pipe", "MUST_FIRE", c_after_pipe),
    ("command word after `env`", "MUST_FIRE", c_after_transparent_word),
    ("quoted command word", "MUST_FIRE", c_quoted_command_word),
    ("defect on a continuation line", "MUST_FIRE", c_continuation_carries_the_defect),
    ("non-recipe lines are not commands", "MUST_PASS", c_non_recipe_line_is_not_a_command),
    ("$(PY) -m <tool> is the fix", "MUST_PASS", c_py_dash_m_is_the_fix),
    ("tool name as an argument", "MUST_PASS", c_tool_name_as_argument),
    ("printf-generated python is data", "MUST_PASS", c_printf_generated_python_is_data),
    ("usage string quoting $(PY)", "MUST_PASS", c_usage_string_quoting_py),
    ("captured skip-guard block", "MUST_PASS", c_skip_guard_block),
    ("absolute interpreter path", "MUST_PASS", c_absolute_path_is_not_bare),
    ("zero recipe lines -> 95", "MUST_PASS", c_zero_recipe_lines_is_unmeasured),
    ("unreadable Makefile -> 95", "MUST_PASS", c_unreadable_makefile_is_unmeasured),
    ("live tree denominator > 0", "MUST_PASS", c_live_tree_denominator_is_nonzero),
]


def self_test() -> int:
    """Run every control. Exit 0 only if all pass; print the denominator either way."""
    failures: list[str] = []
    for name, kind, fn in CONTROLS:
        try:
            ok = bool(fn())  # type: ignore[operator]
        except Exception as exc:  # noqa: BLE001 - a crashing control is a failing control
            ok = False
            name = f"{name} (raised {type(exc).__name__}: {exc})"
        if not ok:
            failures.append(f"  {kind} FAILED: {name}")
    fires = sum(1 for _, k, _ in CONTROLS if k == "MUST_FIRE")
    passes = len(CONTROLS) - fires
    if failures:
        print("\n".join(failures))
        print(
            f"self-test: {len(CONTROLS) - len(failures)} of {len(CONTROLS)} controls ok "
            f"({fires} MUST_FIRE, {passes} MUST_PASS) -- FAILED"
        )
        return EXIT_RED
    print(
        f"self-test: {len(CONTROLS)} of {len(CONTROLS)} controls ok "
        f"({fires} MUST_FIRE, {passes} MUST_PASS)"
    )
    return EXIT_CLEAR


def run(makefile: Path, quiet: bool = False) -> int:
    """Evaluate `makefile` and return the exit code, printing the verdict unless quiet."""
    try:
        text = makefile.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        if not quiet:
            print(
                f"UNMEASURED makefile_tooling: cannot read {makefile}: {exc} -- unreadable is "
                "not empty and it is not clean (doctrine 4)"
            )
        return EXIT_UNMEASURED
    rc, lines = evaluate(text)
    if not quiet:
        print("\n".join(lines))
    return rc


def main(argv: list[str]) -> int:
    if "--self-test" in argv:
        return self_test()
    return run(MAKEFILE)


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv[1:]))
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001 - a crash is not a verdict
        print(f"REFUSE makefile_tooling: unexpected internal error: {type(exc).__name__}: {exc}")
        sys.exit(EXIT_REFUSE)
