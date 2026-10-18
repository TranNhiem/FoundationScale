"""A small, self-contained LRU cache module with hit/miss statistics.

The cache stores key/value pairs and evicts the *least recently used* entry when
its capacity is exceeded.  Every successful ``get`` bumps the entry to the most
recently used position.
"""

from __future__ import annotations

__all__ = ["LRUCache", "CacheStats", "make_cache"]


# ---------------------------------------------------------------------------
#
# ---------------------------------------------------------------------------
class CacheStats:
    """Running counters for cache accesses.

    Attributes
    ----------
    hits : int
        Number of ``get`` calls that found the key.
    misses : int
        Number of ``get`` calls that did not find the key (including keys that
        were evicted since they were written).
    """

    __slots__ = ("hits", "misses")

    def __init__(self) -> None:
        self.hits: int = 0
        self.misses: int = 0

    @property
    def total(self) -> int:
        """Total number of lookups (hits + misses)."""
        return self.hits + self.misses

    @property
    def hit_rate(self) -> float:
        """Fraction of lookups that were hits (0.0 if no lookups yet)."""
        total = self.total
        if total == 0:
            return 0.0
        return self.hits / total

    def reset(self) -> None:
        """Zero both counters."""
        self.hits = 0
        self.misses = 0

    def __repr__(self) -> str:
        return f"CacheStats(hits={self.hits}, misses={self.misses})"


# ---------------------------------------------------------------------------
#
# ---------------------------------------------------------------------------
class LRUCache:
    """Fixed-capacity LRU cache mapping arbitrary hashable keys to values.

    Parameters
    ----------
    capacity : int
        Maximum number of entries.  Must be a positive integer.

    Raises
    ------
    ValueError
        If ``capacity`` is not a positive integer.
    """

    def __init__(self, capacity: int) -> None:
        if not isinstance(capacity, int) or isinstance(capacity, bool):
            raise TypeError("capacity must be an int")
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self._capacity = capacity
        self._data: dict = {}  # insertion order == LRU -> MRU
        self._stats = CacheStats()

    # -- basic properties --------------------------------------------------
    @property
    def capacity(self) -> int:
        """Maximum number of entries the cache can hold."""
        return self._capacity

    @property
    def stats(self) -> CacheStats:
        """Shared :class:`CacheStats` object for this cache."""
        return self._stats

    def __len__(self) -> int:
        """Current number of stored entries."""
        return len(self._data)

    def __contains__(self, key) -> bool:
        """Membership test.  Does **not** bump recency or change stats."""
        return key in self._data

    def keys(self) -> list:
        """Return keys ordered from least to most recently used."""
        return list(self._data.keys())

    # -- core operations ---------------------------------------------------
    def get(self, key, default=None):
        """Look up *key*.  On hit the entry is bumped to MRU position.

        Returns *default* (``None`` unless given) on miss and increments the
        appropriate statistic counter.
        """
        try:
            value = self._data.pop(key)
        except KeyError:
            self._stats.misses += 1
            return default
        self._stats.hits += 1
        self._data[key] = value  # re-insert at MRU end
        return value

    def put(self, key, value) -> None:
        """Insert or update *key*.  Evict LRU entry if over capacity.

        Updating an existing key refreshes its recency without evicting.
        """
        if key in self._data:
            self._data.pop(key)
        self._data[key] = value
        while len(self._data) > self._capacity:
            lru_key = next(iter(self._data))
            del self._data[lru_key]

    def pop(self, key, default=None):
        """Remove *key* and return its value (or *default*).  No stat change."""
        return self._data.pop(key, default)

    def clear(self) -> None:
        """Remove all entries and reset statistics."""
        self._data.clear()
        self._stats.reset()


# ---------------------------------------------------------------------------
#
# ---------------------------------------------------------------------------
def make_cache(capacity: int) -> LRUCache:
    """Convenience factory that validates and returns an :class:`LRUCache`."""
    if capacity < 1:
        raise ValueError("capacity must be at least 1")
    return LRUCache(capacity)
