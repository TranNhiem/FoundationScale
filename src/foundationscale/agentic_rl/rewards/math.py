"""Verifiable math-answer reward for agentic RL (tool-using math reasoning).

The policy solves a problem -- it may run Python through its tools -- and ends with a final
answer, normally ``\\boxed{...}``. :class:`MathAnswerReward` scores that answer against the
task's reference: 1.0 when they are equivalent, 0.0 otherwise, with a note saying why
(``correct``, ``wrong``, ``no_answer``, ``no_reference``). The score is deterministic and never
abstains, so a step's reward noise comes from the policy, not from the scorer.

Equivalence is decided in two stages. First, both answers are normalized (surrounding ``$``,
``\\left``/``\\right``, spacing commands, ``\\text{}`` wrappers, ``\\dfrac``, degree marks,
``%`` and thousands separators) and compared as strings. Then a small LaTeX subset (fractions,
roots, powers, pi, products) is rewritten into sympy syntax and the difference is simplified to
zero, with a numeric tolerance for decimals. Comma-separated answers (tuples, intervals, lists)
are compared element by element. Anything unparseable is simply "not equivalent" -- the scorer
never raises, and inputs longer than ``MAX_ANSWER_CHARS`` skip the symbolic stage so a
degenerate answer cannot stall a rollout.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import sympy  # type: ignore[import-untyped]

MAX_ANSWER_CHARS = 200
_REL_TOL = 1e-6

_BOXED = re.compile(r"\\(?:boxed|fbox)\s*\{")
_FINAL = re.compile(r"(?:final answer|answer)\s*[:：]\s*(.+)", re.IGNORECASE)
_WRAPPERS = re.compile(r"\\(?:text|mathrm|textbf|mathbf|operatorname)\s*\{([^{}]*)\}")
_SPACING = re.compile(r"\\[,;:! ]|\\left|\\right|\\displaystyle")


def _braced(text: str, start: int) -> str | None:
    """The content of the brace group opening at ``text[start] == '{'``, or None if unbalanced."""
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start + 1 : i]
    return None


def extract_final_answer(text: str) -> str | None:
    """The last ``\\boxed{...}`` / ``\\fbox{...}`` content, else the last "Answer: ..." line."""
    matches = list(_BOXED.finditer(text))
    for match in reversed(matches):
        content = _braced(text, match.end() - 1)
        if content is not None and content.strip():
            return content.strip()
    lines = _FINAL.findall(text)
    if lines and lines[-1].strip():
        return lines[-1].strip()
    return None


def normalize_answer(answer: str) -> str:
    """A canonical string form of ``answer`` (see the module docstring). Never raises."""
    s = answer.strip()
    if s.startswith("$") and s.endswith("$"):
        s = s.strip("$")
    prev = None
    while prev != s:  # nested wrappers: \text{\textbf{x}}
        prev, s = s, _WRAPPERS.sub(r"\1", s)
    s = _SPACING.sub("", s)
    s = s.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    s = s.replace("^{\\circ}", "").replace("^\\circ", "").replace("\\%", "").replace("%", "")
    s = re.sub(r"(?<=\d),(?=\d{3}(?!\d))", "", s)  # 1,000 -> 1000 (not a tuple's comma)
    s = re.sub(r"\s+", "", s)
    return s.rstrip(".")


def _split_top_level(s: str) -> list[str]:
    """Split on commas outside any (), [], {} -- '(1,2)' is one item, '1,2' two."""
    items, depth, cur = [], 0, ""
    for ch in s:
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        if ch == "," and depth == 0:
            items.append(cur)
            cur = ""
        else:
            cur += ch
    items.append(cur)
    return items


def _latex_to_sympy(s: str) -> str:
    """Rewrite a small LaTeX subset into sympy syntax; unknown commands are left to fail parsing."""
    out = s
    for _ in range(10):  # nested fractions/roots resolve inside-out
        new = re.sub(r"\\frac\{([^{}]*)\}\{([^{}]*)\}", r"((\1)/(\2))", out)
        new = re.sub(r"\\sqrt\[([^\[\]]*)\]\{([^{}]*)\}", r"((\2)**(1/(\1)))", new)
        new = re.sub(r"\\sqrt\{([^{}]*)\}", r"sqrt(\1)", new)
        if new == out:
            break
        out = new
    out = out.replace("\\pi", "pi").replace("\\cdot", "*").replace("\\times", "*")
    return out.replace("^", "**").replace("{", "(").replace("}", ")")


def _symbolic_equal(a: str, b: str) -> bool:
    try:
        ea = sympy.sympify(_latex_to_sympy(a), rational=False)
        eb = sympy.sympify(_latex_to_sympy(b), rational=False)
        if ea.is_number and eb.is_number:
            va, vb = complex(ea.evalf()), complex(eb.evalf())
            return abs(va - vb) <= _REL_TOL * max(1.0, abs(vb))
        return bool(sympy.simplify(ea - eb) == 0)
    except Exception:  # noqa: BLE001 -- any parse/eval failure means "not shown equivalent"
        return False


def _strip_outer(s: str) -> str:
    if len(s) >= 2 and s[0] in "([" and s[-1] in ")]":
        return s[1:-1]
    return s


def answers_equivalent(pred: str, truth: str) -> bool:
    """True if ``pred`` and ``truth`` are the same answer (string, then symbolic). Never raises."""
    p, t = normalize_answer(pred), normalize_answer(truth)
    if not p or not t:
        return False
    if p == t:
        return True
    if len(p) > MAX_ANSWER_CHARS or len(t) > MAX_ANSWER_CHARS:
        return False
    if p[:1] in "([" and t[:1] in "([" and (p[0] != t[0] or p[-1] != t[-1]):
        return False  # (1,2) vs [1,2]: an open and a closed interval are different answers
    p_items, t_items = _split_top_level(_strip_outer(p)), _split_top_level(_strip_outer(t))
    if len(p_items) != len(t_items):
        return False
    if len(p_items) > 1:
        return all(x == y or _symbolic_equal(x, y) for x, y in zip(p_items, t_items, strict=True))
    return _symbolic_equal(p, t)


@dataclass(frozen=True)
class MathScore:
    value: float
    note: str


@dataclass(frozen=True)
class MathAnswerReward:
    """Scores a response's final answer against the task's reference answer (1.0 or 0.0)."""

    def score(self, text: str, reference: str | None) -> MathScore:
        if reference is None or not str(reference).strip():
            return MathScore(0.0, "no_reference")
        answer = extract_final_answer(text)
        if answer is None:
            return MathScore(0.0, "no_answer")
        if answers_equivalent(answer, str(reference)):
            return MathScore(1.0, "correct")
        return MathScore(0.0, "wrong")
