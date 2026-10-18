"""Tests for :mod:`lru`.  Each test checks exactly one behaviour."""

import pytest
from lru import LRUCache, make_cache


def test_capacity_is_preserved():
    c = LRUCache(3)
    assert c.capacity == 3


def test_len_is_zero_initially():
    c = LRUCache(5)
    assert len(c) == 0


def test_put_then_get_returns_value():
    c = LRUCache(5)
    c.put("a", 1)
    assert c.get("a") == 1


def test_len_after_two_puts():
    c = LRUCache(5)
    c.put("a", 1)
    c.put("b", 2)
    assert len(c) == 2


def test_get_missing_returns_default_none():
    c = LRUCache(5)
    assert c.get("nope") is None


def test_get_missing_returns_explicit_default():
    c = LRUCache(5)
    assert c.get("nope", -7) == -7


def test_contains_true_after_put():
    c = LRUCache(5)
    c.put("a", 1)
    assert ("a" in c) is True


def test_contains_false_for_absent_key():
    c = LRUCache(5)
    c.put("a", 1)
    assert ("b" in c) is False


def test_put_update_overwrites_value():
    c = LRUCache(5)
    c.put("a", 1)
    c.put("a", 2)
    assert c.get("a") == 2
    assert len(c) == 1


def test_evicts_least_recently_used_on_overflow():
    c = LRUCache(2)
    c.put("a", 1)
    c.put("b", 2)
    c.put("c", 3)
    assert ("a" in c) is False
    assert ("b" in c) is True
    assert ("c" in c) is True
    assert len(c) == 2


def test_get_bumps_recency_preventing_eviction():
    c = LRUCache(2)
    c.put("a", 1)
    c.put("b", 2)
    c.get("a")  # a becomes MRU
    c.put("c", 3)  # evicts b
    assert ("a" in c) is True
    assert ("b" in c) is False
    assert ("c" in c) is True


def test_put_existing_key_bumps_recency():
    c = LRUCache(2)
    c.put("a", 1)
    c.put("b", 2)
    c.put("a", 10)  # a becomes MRU
    c.put("c", 3)  # evicts b
    assert c.get("a") == 10
    assert ("b" in c) is False


def test_hit_counter_increments():
    c = LRUCache(5)
    c.put("a", 1)
    c.get("a")
    c.get("a")
    assert c.stats.hits == 2


def test_miss_counter_increments():
    c = LRUCache(5)
    c.get("x")
    c.get("y", 0)
    assert c.stats.misses == 2


def test_stats_not_changed_by_contains():
    c = LRUCache(5)
    c.put("a", 1)
    _ = "a" in c
    _ = "z" in c
    assert c.stats.hits == 0
    assert c.stats.misses == 0


def test_stats_total_and_hit_rate():
    c = LRUCache(5)
    c.put("a", 1)
    c.get("a")
    c.get("z")
    assert c.stats.total == 2
    assert c.stats.hit_rate == pytest.approx(0.5)


def test_hit_rate_zero_when_no_lookups():
    c = LRUCache(5)
    assert c.stats.hit_rate == 0.0


def test_pop_removes_and_returns_value():
    c = LRUCache(5)
    c.put("a", 1)
    assert c.pop("a") == 1
    assert len(c) == 0
    assert ("a" in c) is False


def test_pop_missing_returns_default():
    c = LRUCache(5)
    assert c.pop("zz", 42) == 42


def test_keys_returns_lru_to_mru_order():
    c = LRUCache(3)
    c.put("x", 1)
    c.put("y", 2)
    c.put("z", 3)
    assert c.keys() == ["x", "y", "z"]


def test_clear_resets_entries_and_stats():
    c = LRUCache(5)
    c.put("a", 1)
    c.get("a")
    c.clear()
    assert len(c) == 0
    assert c.stats.hits == 0
    assert c.stats.misses == 0


def test_zero_capacity_raises_value_error():
    with pytest.raises(ValueError):
        LRUCache(0)


def test_negative_capacity_raises_value_error():
    with pytest.raises(ValueError):
        LRUCache(-3)


def test_non_int_capacity_raises_type_error():
    with pytest.raises(TypeError):
        LRUCache("big")


def test_make_cache_factory_builds_lru_cache():
    c = make_cache(2)
    assert isinstance(c, LRUCache)
    assert c.capacity == 2
    c.put("k", "v")
    assert c.get("k") == "v"
