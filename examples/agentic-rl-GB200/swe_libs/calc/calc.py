"""A small, dependency-free arithmetic expression evaluator.

Supports: ``+ - * /``, parentheses and unary minus.  The module exposes a
public :func:`calculate` helper and the lower-level building blocks
(:class:`Token`, :func:`tokenize`, :class:`Parser`, :func:`parse`) so that
tests can exercise each stage independently.

Grammar (precedence climbs low -> high)::

    expression := term (('+' | '-') term)*
    term       := factor (('*' | '/') factor)*
    factor     := ('-' | '+') factor | NUMBER | '(' expression ')'
"""

from __future__ import annotations

import re
from dataclasses import dataclass

__all__ = [
    "Token",
    "tokenize",
    "Parser",
    "parse",
    "calculate",
    "CalcError",
]


# --------------------------------------------------------------------------- #
# Errors
# --------------------------------------------------------------------------- #


class CalcError(ValueError):
    """Raised when an expression is invalid or cannot be evaluated."""


# --------------------------------------------------------------------------- #
# Tokens
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class Token:
    """A lexical token: *kind* and textual *value*."""

    kind: str  # "NUM", "OP", "LPAREN", "RPAREN", "END"
    value: str
    pos: int


_TOKEN_RE = re.compile(
    r"(?P<NUM>\d+(?:\.\d+)?)"
    r"|(?P<OP>[+\-*/])"
    r"|(?P<LPAREN>\()"
    r"|(?P<RPAREN>\))"
    r"|(?P<WS>\s+)"
)


def tokenize(text: str) -> list[Token]:
    """Convert *text* to a list of :class:`Token` objects.

    Raises :class:`CalcError` on any unexpected character.  The final token
    is always a synthetic ``END`` sentinel.
    """
    tokens: list[Token] = []
    pos = 0
    while pos < len(text):
        match = _TOKEN_RE.match(text, pos)
        if match is None:
            raise CalcError(f"unexpected character {text[pos]!r} at position {pos}")
        kind = match.lastgroup
        if kind != "WS":
            tokens.append(Token(kind, match.group(), match.start()))
        pos = match.end()
    tokens.append(Token("END", "", len(text)))
    return tokens


# --------------------------------------------------------------------------- #
# Parser (recursive descent, returns a float)
# --------------------------------------------------------------------------- #


class Parser:
    """Recursive-descent parser that evaluates as it parses.

    The parser consumes a token list and produces a numeric result.  It is
    deliberately explicit (no stack machine) so the control flow is easy to
    follow and easy to test in isolation.
    """

    def __init__(self, tokens: list[Token]) -> None:
        self._tokens = tokens
        self._index = 0

    # -- helpers ---------------------------------------------------------- #
    def _current(self) -> Token:
        return self._tokens[self._index]

    def _advance(self) -> Token:
        tok = self._tokens[self._index]
        self._index += 1
        return tok

    def _expect(self, kind: str) -> Token:
        tok = self._current()
        if tok.kind != kind:
            raise CalcError(
                f"expected {kind} but found {tok.kind} {tok.value!r} at position {tok.pos}"
            )
        return self._advance()

    # -- grammar ---------------------------------------------------------- #
    def parse(self) -> float:
        """Entry point: parse the full expression and return the result."""
        result = self._expression()
        if self._current().kind != "END":
            tok = self._current()
            raise CalcError(f"unexpected token {tok.value!r} at position {tok.pos}")
        return result

    def _expression(self) -> float:
        left = self._term()
        while self._current().kind == "OP" and self._current().value in ("+", "-"):
            op = self._advance().value
            right = self._term()
            if op == "+":  # noqa: SIM108 -- bug-injection target; kept as written
                left = left + right
            else:
                left = left - right
        return left

    def _term(self) -> float:
        left = self._factor()
        while self._current().kind == "OP" and self._current().value in ("*", "/"):
            op = self._advance().value
            right = self._factor()
            if op == "*":
                left = left * right
            elif right == 0:
                raise CalcError("division by zero")
            else:
                left = left / right
        return left

    def _factor(self) -> float:
        tok = self._current()
        if tok.kind == "OP" and tok.value in ("-", "+"):
            self._advance()
            value = self._factor()
            return -value if tok.value == "-" else value
        if tok.kind == "NUM":
            self._advance()
            return float(tok.value)
        if tok.kind == "LPAREN":
            self._advance()
            value = self._expression()
            self._expect("RPAREN")
            return value
        raise CalcError(f"unexpected token {tok.value!r} at position {tok.pos}")


def parse(text: str) -> float:
    """Convenience wrapper: tokenize then parse *text*."""
    return Parser(tokenize(text)).parse()


def calculate(text: str) -> float:
    """Evaluate an arithmetic expression string and return a float.

    >>> calculate("(2 + 3) * -4")
    -20.0
    >>> calculate("10 / 4 - 1")
    1.5
    """
    result = parse(text)
    if not isinstance(result, float):
        raise CalcError("expression did not evaluate to a number")
    return result
