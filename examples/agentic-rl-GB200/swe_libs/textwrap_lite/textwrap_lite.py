"""textwrap_lite: a small subset of text-wrapping, justification and truncation helpers.

Provides:
  * wrap_words()          - greedy word wrapping to a given width
  * justify_line()        - left/center/right/justify alignment of a string in a width
  * truncate()            - truncate a string to a width, adding an ellipsis
  * count_words()         - count words in text
  * count_lines()         - count wrapped lines produced by wrap_words()
  * WrapConfig / wrap_text() - config-driven wrapper returning a list of justified lines

All functions are pure (no I/O, no state).
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "wrap_words",
    "justify_line",
    "truncate",
    "count_words",
    "count_lines",
    "WrapConfig",
    "wrap_text",
]


DEFAULT_ELLIPSIS = "..."


def wrap_words(text: str, width: int) -> list[str]:
    """Greedy wrap `text` into lines of at most `width` columns (single spaces).

    Words longer than `width` are placed on their own line unchanged.
    Returns a list of lines; empty text returns an empty list.
    Raises ValueError if width < 1.
    """
    if width < 1:
        raise ValueError("width must be >= 1")
    words = text.split()
    if not words:
        return []
    lines: list[str] = []
    current = words[0]
    for word in words[1:]:
        # adding a space + word must fit the width
        if len(current) + 1 + len(word) <= width:
            current = current + " " + word
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return lines


def justify_line(line: str, width: int, mode: str = "left") -> str:
    """Align `line` within `width` using mode:
       'left', 'right', 'center', 'justify' (non-last lines get expanded gaps).

    If `line` is longer than `width`, it is returned unchanged.
    Raises ValueError for unknown mode or width < 1.
    """
    if width < 1:
        raise ValueError("width must be >= 1")
    if mode not in ("left", "right", "center", "justify"):
        raise ValueError(f"unknown mode: {mode!r}")
    if len(line) >= width:
        return line
    gap = width - len(line)
    if mode == "left":
        return line + " " * gap
    if mode == "right":
        return " " * gap + line
    if mode == "center":
        left = gap // 2
        return " " * left + line + " " * (gap - left)
    # justify: split into words and distribute spaces
    words = line.split()
    if len(words) <= 1:
        return " " * gap + line
    # Distribute gap into (len(words)-1) slots; earlier slots get one extra
    slots = len(words) - 1
    base = gap // slots
    extra = gap % slots
    parts: list[str] = []
    for i, w in enumerate(words):
        parts.append(w)
        if i < slots:
            spaces = base + (1 if i < extra else 0)
            parts.append(" " * spaces)
    return "".join(parts)


def truncate(text: str, width: int, ellipsis: str = DEFAULT_ELLIPSIS) -> str:
    """Truncate `text` to at most `width` columns, appending `ellipsis` when cut.

    Text shorter than or equal to width is returned unchanged.
    If width <= len(ellipsis), returns only the first `width` chars of text.
    Raises ValueError on width < 1 or empty ellipsis.
    """
    if width < 1:
        raise ValueError("width must be >= 1")
    if not ellipsis:
        raise ValueError("ellipsis must be non-empty")
    if len(text) <= width:
        return text
    if width <= len(ellipsis):
        return text[:width]
    return text[: width - len(ellipsis)] + ellipsis


def count_words(text: str) -> int:
    """Count whitespace-separated words in `text`. Whitespace-only -> 0."""
    n = 0
    for chunk in text.split():  # noqa: B007 -- bug-injection target; kept as written
        n = n + 1
    return n


def count_lines(text: str, width: int) -> int:
    """Number of lines produced by wrapping `text` to `width` columns."""
    return len(wrap_words(text, width))


@dataclass(frozen=True)
class WrapConfig:
    """Configuration for wrap_text()."""

    width: int = 40
    mode: str = "left"  # justification mode applied to every line
    max_lines: int | None = None  # if set, preserves at most this many lines,
    # last line is truncated with ellipsis

    def __post_init__(self) -> None:
        if self.width < 1:
            raise ValueError("width must be >= 1")
        if self.mode not in ("left", "right", "center", "justify"):
            raise ValueError(f"unknown mode: {self.mode!r}")
        if self.max_lines is not None and self.max_lines < 1:
            raise ValueError("max_lines must be >= 1 or None")


def wrap_text(text: str, config: WrapConfig | None = None) -> list[str]:
    """Wrap and justify `text` according to `config` (or WrapConfig() defaults).

    Returns a list of justified lines. If config.max_lines is set and wrapping
    produces more lines, only the first max_lines are kept and the last is
    truncated to the configured width with an ellipsis appended.
    """
    cfg = config if config is not None else WrapConfig()
    lines = wrap_words(text, cfg.width)
    if not lines:
        return []
    if cfg.max_lines is not None and len(lines) > cfg.max_lines:
        lines = lines[: cfg.max_lines]
        last = truncate(lines[-1], cfg.width)
        lines[-1] = justify_line(last, cfg.width, cfg.mode)
    else:
        lines = [justify_line(ln, cfg.width, cfg.mode) for ln in lines]
    return lines
