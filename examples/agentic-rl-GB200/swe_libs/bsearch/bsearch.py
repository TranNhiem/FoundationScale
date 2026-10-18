"""Binary search helpers for sorted sequences.

All functions assume the input sequence is already sorted in ascending
order (unless noted otherwise) and support any sequence that implements
``__getitem__`` and ``__len__`` (lists, tuples, etc.).  Indices follow
normal Python conventions: negative counts from the end, slices-style
semantics where relevant.

Public API
----------
- ``lower_bound(seq, target)``  -> first index where ``seq[i] >= target``
- ``upper_bound(seq, target)``  -> first index where ``seq[i] >  target``
- ``index_of(seq, target)``     -> index of ``target`` or ``-1``
- ``count(seq, target)``        -> number of occurrences of ``target``
- ``equal_range(seq, target)``  -> ``(lower_bound, upper_bound)`` tuple
- ``insert_sorted(seq, value)`` -> list copy with ``value`` placed stably
- ``is_sorted(seq)``           -> ``True`` if non-decreasing
- ``search_rotated(seq, target)`` -> index of ``target`` in a rotated list
"""

from collections.abc import Sequence
from typing import Any


def lower_bound(seq: Sequence[Any], target: Any) -> int:
    """Return the first index i such that every element at indices < i is <
    target, and every element at indices >= i is >= target.  Equivalent to
    ``bisect.bisect_left``.  Returns an index in ``[0, len(seq)]``.
    """
    lo, hi = 0, len(seq)
    while lo < hi:
        mid = (lo + hi) // 2
        if seq[mid] < target:
            lo = mid + 1
        else:
            hi = mid
    return lo


def upper_bound(seq: Sequence[Any], target: Any) -> int:
    """Return the first index i such that seq[i] > target.  All elements at
    indices < i are <= target.  Equivalent to ``bisect.bisect_right``.
    """
    lo, hi = 0, len(seq)
    while lo < hi:
        mid = (lo + hi) // 2
        if seq[mid] <= target:
            lo = mid + 1
        else:
            hi = mid
    return lo


def index_of(seq: Sequence[Any], target: Any) -> int:
    """Return the index of ``target`` in ``seq``, or ``-1`` if not present.

    If ``target`` occurs multiple times, the first (leftmost) occurrence
    is returned.
    """
    i = lower_bound(seq, target)
    if i < len(seq) and seq[i] == target:
        return i
    return -1


def count(seq: Sequence[Any], target: Any) -> int:
    """Return the number of times ``target`` appears in ``seq``."""
    return upper_bound(seq, target) - lower_bound(seq, target)


def equal_range(seq: Sequence[Any], target: Any) -> tuple[int, int]:
    """Return ``(lower_bound(seq, target), upper_bound(seq, target))``.

    The slice ``seq[start:end]`` contains exactly the occurrences of
    ``target`` when they exist.
    """
    lo = lower_bound(seq, target)
    hi = upper_bound(seq, target)
    return (lo, hi)


def insert_sorted(seq: Sequence[Any], value: Any) -> list[Any]:
    """Return a new sorted list with ``value`` inserted into ``seq``.

    The insertion is *stable*: if ``seq`` already contains elements equal
    to ``value``, the new element is placed after them (i.e., at
    ``upper_bound``).  This preserves order of equal elements.
    """
    out = list(seq)  # copy
    pos = upper_bound(seq, value)
    out.insert(pos, value)
    return out


def is_sorted(seq: Sequence[Any]) -> bool:
    """Return ``True`` if ``seq`` is sorted in non-decreasing order."""
    n = len(seq)
    for i in range(1, n):  # noqa: SIM110 -- bug-injection target; kept as written
        if seq[i] < seq[i - 1]:
            return False
    return True


def search_rotated(seq: Sequence[Any], target: Any) -> int:
    """Search for ``target`` in a rotated sorted list.

    ``seq`` is assumed to have originally been sorted in non-decreasing
    order and then rotated right by 0..n-1 positions (e.g. [4, 5, 6, 0,
    1, 2]).  Returns the index of ``target`` or ``-1``.  Ties are broken
    arbitrarily because the algorithm returns *an* index where ``target``
    occurs; for lists with duplicate elements the same guarantee of "first
    occurrence" does not hold.
    """
    lo, hi = 0, len(seq) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if seq[mid] == target:
            return mid
        if seq[lo] <= seq[mid]:
            # Left half [lo..mid] is sorted
            if seq[lo] <= target < seq[mid]:
                hi = mid - 1
            else:
                lo = mid + 1
        else:
            # Right half [mid..hi] is sorted
            if seq[mid] < target <= seq[hi]:
                lo = mid + 1
            else:
                hi = mid - 1
    return -1
