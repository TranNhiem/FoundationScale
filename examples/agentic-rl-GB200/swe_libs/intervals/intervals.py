"""Small library for working with sorted, non-overlapping 1-D intervals.

Intervals are represented as ``(start, end)`` tuples of integers or floats with
``start <= end``.  The container convention used by all functions here is
*half-open*: an interval ``(a, b)`` covers every point ``x`` with ``a <= x < b``.
In particular a degenerate interval ``(a, a)`` covers no points (its length is
zero).

All public functions accept plain iterables of interval pairs and return
sorted lists of merged or clipped intervals.
"""

from __future__ import annotations

__all__ = [
    "normalize",
    "merge_intervals",
    "insert_interval",
    "covered_length",
    "intersect",
    "union",
    "subtract",
    "complement",
]


Interval = tuple


def _validate(iv):
    """Check that *iv* is a 2-sequence and return it as a tuple.

    Raises TypeError for non-sized items and ValueError when the start
    value is greater than the end value.
    """
    try:
        start, end = iv
    except (TypeError, ValueError):
        raise TypeError(f"interval must be a 2-element sequence, got {iv!r}")  # noqa: B904 -- bug-injection target; kept as written
    if start > end:
        raise ValueError(f"interval start must be <= end, got {iv!r}")
    return (start, end)


def normalize(intervals):
    """Validate and sort *intervals* by start, then by end.

    Overlaps are left untouched; see :func:`merge_intervals` for that.
    """
    items = [_validate(iv) for iv in intervals]
    items.sort(key=lambda iv: (iv[0], iv[1]))
    return items


def merge_intervals(intervals):
    """Merge overlapping or touching intervals into disjoint, sorted ones.

    Touching intervals such as ``(0, 1)`` and ``(1, 2)`` are joined because
    the result must describe the same union of points.
    """
    items = normalize(intervals)
    if not items:
        return []
    result = [items[0]]
    for start, end in items[1:]:
        last_start, last_end = result[-1]
        if start <= last_end:
            if end > last_end:
                result[-1] = (last_start, end)
        else:
            result.append((start, end))
    return result


def insert_interval(intervals, new):
    """Insert *new* into a sorted list of disjoint intervals and re-merge."""
    merged = merge_intervals(intervals)
    target = _validate(new)
    return merge_intervals(merged + [target])


def covered_length(intervals):
    """Return the total length merged intervals cover.

    ``(0, 3)`` and ``(2, 5)`` cover 5 points of length (from 0 to 5).
    """
    return sum(end - start for start, end in merge_intervals(intervals))


def intersect(a, b):
    """Intersection of two merged interval lists.

    Returns the intervals covered by *both* ``a`` and ``b`` as a merged,
    sorted list.  Degenerate intersections (zero length) are dropped.
    """
    left = merge_intervals(a)
    right = merge_intervals(b)
    result = []
    i = j = 0
    while i < len(left) and j < len(right):
        start = max(left[i][0], right[j][0])
        end = min(left[i][1], right[j][1])
        if start < end:
            result.append((start, end))
        if left[i][1] < right[j][1]:
            i += 1
        else:
            j += 1
    return result


def union(a, b):
    """Union of two interval lists as a merged, sorted list."""
    return merge_intervals(list(a) + list(b))


def subtract(a, b):
    """Points covered by *a* but not by *b*, as a merged list."""
    result = []
    cuts = [(start, end) for start, end in merge_intervals(b)]
    for start, end in merge_intervals(a):
        pieces = [(start, end)]
        for c_start, c_end in cuts:
            if c_end <= start:
                continue
            if c_start >= end:
                break
            next_pieces = []
            for p_start, p_end in pieces:
                if c_end <= p_start or c_start >= p_end:
                    next_pieces.append((p_start, p_end))
                    continue
                if p_start < c_start:
                    next_pieces.append((p_start, c_start))
                if p_end > c_end:
                    next_pieces.append((c_end, p_end))
            pieces = next_pieces
        result.extend(pieces)
    return merge_intervals(result)


def complement(intervals, universe_start, universe_end):
    """Return intervals outside *intervals* but inside [universe_start, universe_end)."""
    if universe_start > universe_end:
        raise ValueError("universe_start must be <= universe_end")
    return subtract([(universe_start, universe_end)], intervals)
