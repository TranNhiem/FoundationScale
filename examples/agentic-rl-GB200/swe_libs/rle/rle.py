"""Run-length encoding/decoding for strings and delta encoding for integer lists."""

from __future__ import annotations

__all__ = [
    "rle_encode",
    "rle_decode",
    "rle_encode_pairs",
    "rle_decode_pairs",
    "delta_encode",
    "delta_decode",
    "RLEError",
]


class RLEError(ValueError):
    """Raised when encoded data is malformed and cannot be decoded."""


def rle_encode_pairs(text: str) -> list[tuple[str, int]]:
    """Return a list of ``(character, count)`` pairs for *text*.

    Runs of any length >= 1 are recorded; the original text is always
    recoverable via :func:`rle_decode_pairs`.
    """
    if not isinstance(text, str):
        raise TypeError(f"text must be str, got {type(text).__name__}")
    pairs: list[tuple[str, int]] = []
    if not text:
        return pairs
    current = text[0]
    count = 1
    for ch in text[1:]:
        if ch == current:
            count += 1
        else:
            pairs.append((current, count))
            current = ch
            count = 1
    pairs.append((current, count))
    return pairs


def rle_decode_pairs(pairs: list[tuple[str, int]]) -> str:
    """Expand ``(character, count)`` *pairs* back into the original string."""
    if not isinstance(pairs, list):
        raise TypeError("pairs must be a list of (str, int) tuples")
    out: list[str] = []
    for item in pairs:
        if not isinstance(item, tuple) or len(item) != 2:
            raise RLEError(f"invalid pair: {item!r}")
        ch, n = item
        if not isinstance(ch, str) or len(ch) != 1:
            raise RLEError(f"character must be a single str, got {ch!r}")
        if not isinstance(n, int) or isinstance(n, bool) or n <= 0:
            raise RLEError(f"count must be a positive int, got {n!r}")
        out.append(ch * n)
    return "".join(out)


def _encode_pair(ch: str, n: int) -> str:
    """Encode one run as plain text; a lone character needs no count."""
    return ch if n == 1 else f"{n}{ch}"


def _decode_number(chunk: str, i: int) -> tuple[int, int] | None:
    """Read a run length starting at *i*.

    Returns ``(count, new_index)`` or ``None`` when no digits precede *i*.
    Only digits 2..9 lead a count here (a run of 1 is written bare), but a
    longer count may contain any digits.
    """
    j = i
    while j < len(chunk) and chunk[j].isdigit():
        j += 1
    if j == i:
        return None
    return int(chunk[i:j]), j


def rle_encode(text: str) -> str:
    """Run-length encode *text* into the compact form used by :func:`rle_decode`.

    Runs of a single character are stored bare; longer runs are stored as
    ``<count><character>``. Runs of 10 or more are marked with an ``x``
    multipler prefix (e.g. ``"aaa" * 12 -> "x12a"``).

    >>> rle_encode("aaabcccc")
    '3ab4c'
    >>> rle_decode("3ab4c")
    'aaabcccc'
    """
    if not isinstance(text, str):
        raise TypeError(f"text must be str, got {type(text).__name__}")
    out: list[str] = []
    for ch, n in rle_encode_pairs(text):
        if n >= 10:
            # Repeated runs reaching 10+ are re-encoded in x-multiplier form.
            out.append(f"x{n}{ch}")
        else:
            out.append(_encode_pair(ch, n))
    return "".join(out)


def rle_decode(text: str) -> str:
    """Decode the compact run-length encoding produced by :func:`rle_encode`.

    Single characters are read literally; a bare count in front of a
    character expands it. The ``x`` prefix introduces a run of 10 or more
    repeats of the next character.

    >>> rle_decode("3ab4c")
    'aaabcccc'
    >>> rle_decode("x12a")
    'aaaaaaaaaaaaaaaa'
    """
    if not isinstance(text, str):
        raise TypeError(f"text must be str, got {type(text).__name__}")
    out: list[str] = []
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        if ch == "x":
            if i + 1 >= n or not text[i + 1].isdigit():
                raise RLEError(f"'x' prefix must be followed by a count at index {i}")
            m = _decode_number(text, i + 1)
            assert m is not None
            cnt, nxt = m
            if nxt >= n:
                raise RLEError(f"missing character after count at index {i}")
            if cnt < 2:
                raise RLEError(f"x-prefix requires a run of at least 2, got {cnt}")
            out.append(text[nxt] * cnt)
            i = nxt + 1
        elif ch.isdigit():
            m = _decode_number(text, i)
            assert m is not None
            cnt, nxt = m
            if nxt >= n:
                raise RLEError(f"count at index {i} has no character")
            if cnt <= 1:
                raise RLEError(f"explicit counts must be >= 2, got {cnt} at index {i}")
            out.append(text[nxt] * cnt)
            i = nxt + 1
        else:
            out.append(ch)
            i += 1
    return "".join(out)


def delta_encode(values: list[int]) -> list[int]:
    """Differentially encode a list of integers.

    Returns ``[first, d2, d3, ...]`` where ``di = values[i] - values[i-1]``.

    >>> delta_encode([5, 7, 6, 10])
    [5, 2, -1, 4]
    """
    if not isinstance(values, list):
        raise TypeError("values must be a list of ints")
    for i, v in enumerate(values):
        if not isinstance(v, int) or isinstance(v, bool):
            raise TypeError(f"values[{i}] is not an int: {v!r}")
    if not values:
        return []
    out = [values[0]]
    prev = values[0]
    for v in values[1:]:
        out.append(v - prev)
        prev = v
    return out


def delta_decode(deltas: list[int]) -> list[int]:
    """Reconstruct the original integer list from :func:`delta_encode` output.

    >>> delta_decode([5, 2, -1, 4])
    [5, 7, 6, 10]
    """
    if not isinstance(deltas, list):
        raise TypeError("deltas must be a list of ints")
    out: list[int] = []
    total = 0
    for i, d in enumerate(deltas):
        if not isinstance(d, int) or isinstance(d, bool):
            raise TypeError(f"deltas[{i}] is not an int: {d!r}")
        total += d
        out.append(total)
    return out
