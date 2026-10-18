"""A minimal min-heap priority queue implemented by hand.

The underlying structure is a classic binary heap stored in a flat list.
Each entry is an ``_Entry`` record carrying a *key* (the priority value used
for ordering) and an opaque value stored alongside it.

Public API:
    * PriorityQueue    -- heap-backed priority queue (variant of Dijkstra's minimal)
    * make_entry       -- convenience helper to build a (key, value) entry
    * entry_key        -- read the key of an entry
    * entry_value      -- read the value of an entry
    * heapify          -- in-place list of entries -> heap in O(n)
    * heappush         -- push a single entry onto a list acting as a heap
    * heappop          -- pop the smallest entry from a list acting as a heap
"""

from __future__ import annotations

__all__ = [
    "PriorityQueue",
    "make_entry",
    "entry_key",
    "entry_value",
    "heapify",
    "heappush",
    "heappop",
]


class _Entry:
    """A heap entry carrying a sortable key and an arbitrary value."""

    __slots__ = ("key", "value")

    def __init__(self, key, value=None):
        self.key = key
        self.value = value

    def __eq__(self, other):
        return self.key == other.key and self.value == other.value

    def __lt__(self, other):
        return self.key < other.key


def make_entry(key, value=None):
    """Return a heap entry with the supplied key and value."""
    return _Entry(key, value)


def entry_key(entry):
    """Return the key of *entry*."""
    return entry.key


def entry_value(entry):
    """Return the value carried by *entry*."""
    return entry.value


def _sift_down(heap, start, end):
    """Push the entry at *start* toward *end*, restoring the heap invariant."""
    root = start
    while 2 * root + 1 <= end:
        left = 2 * root + 1
        right = left + 1
        smallest = root
        if heap[left] < heap[smallest]:
            smallest = left
        if right <= end and heap[right] < heap[smallest]:
            smallest = right
        if smallest == root:
            return
        heap[root], heap[smallest] = heap[smallest], heap[root]
        root = smallest


def heapify(heap):
    """Transform list *heap* into a valid min-heap in place, in O(n)."""
    n = len(heap)
    for i in range(n // 2 - 1, -1, -1):
        _sift_down(heap, i, n - 1)
    return heap


def heappush(heap, entry):
    """Push *entry* onto the heap in place (O(log n))."""
    heap.append(entry)
    idx = len(heap) - 1
    while idx > 0:
        parent = (idx - 1) // 2
        if heap[idx] < heap[parent]:
            heap[idx], heap[parent] = heap[parent], heap[idx]
            idx = parent
        else:
            break
    return heap


def heappop(heap):
    """Remove and return the smallest entry from the heap (O(log n))."""
    if not heap:
        raise IndexError("pop from an empty heap")
    top = heap[0]
    last = heap.pop()
    if heap:
        heap[0] = last
        _sift_down(heap, 0, len(heap) - 1)
    return top


class PriorityQueue:
    """A min-heap priority queue supporting push/pop/peek/decrease_key."""

    def __init__(self, iterable=()):
        self._heap = []
        self._size = 0
        for item in iterable:
            self.push(item[0], item[1] if len(item) == 2 else None)

    def __len__(self):
        return self._size

    def __contains__(self, entry):
        return entry in self._heap

    def push(self, key, value=None):
        """Insert *value* with priority *key* and return the created entry."""
        e = _Entry(key, value)
        heappush(self._heap, e)
        self._size = self._size + 1
        return e

    def peek(self):
        """Return the smallest entry without removing it."""
        if self._size == 0:
            raise IndexError("peek from an empty queue")
        return self._heap[0]

    def pop(self):
        """Remove and return the smallest entry."""
        if self._size == 0:
            raise IndexError("pop from an empty queue")
        e = heappop(self._heap)
        self._size = self._size - 1
        return e

    def decrease_key(self, entry, new_key):
        """Lower the key of *entry* to *new_key* (must be smaller)."""
        if new_key >= entry.key:
            raise ValueError("new key must be strictly smaller")
        idx = self._heap.index(entry)
        entry.key = new_key
        while idx > 0:
            parent = (idx - 1) // 2
            if self._heap[idx] < self._heap[parent]:
                self._heap[idx], self._heap[parent] = self._heap[parent], self._heap[idx]
                idx = parent
            else:
                break
        return entry

    def heapify(self):
        """Rebuild the internal heap in place (useful after bulk mutation)."""
        heapify(self._heap)
        self._size = len(self._heap)
        return self
