import pytest
from pqueue import (
    PriorityQueue,
    entry_key,
    entry_value,
    heapify,
    heappop,
    heappush,
    make_entry,
)


def test_make_entry_stores_key_and_value():
    e = make_entry(3, "three")
    assert entry_key(e) == 3
    assert entry_value(e) == "three"


def test_make_entry_default_value_is_none():
    e = make_entry(7)
    assert entry_key(e) == 7
    assert entry_value(e) is None


def test_heappush_orders_by_key():
    h = []
    heappush(h, make_entry(5))
    heappush(h, make_entry(2))
    heappush(h, make_entry(9))
    assert [entry_key(x) for x in h] == [2, 5, 9]


def test_heappop_returns_smallest_first():
    h = []
    for k in (4, 1, 7, 2):
        heappush(h, make_entry(k))
    order = []
    while h:
        order.append(entry_key(heappop(h)))
    assert order == [1, 2, 4, 7]


def test_heappop_empty_raises():
    with pytest.raises(IndexError):
        heappop([])


def test_heapify_three_element_list():
    lst = [make_entry(9), make_entry(2), make_entry(5)]
    heapify(lst)
    assert [entry_key(x) for x in lst] == [2, 9, 5]


def test_heapify_seven_element_list():
    lst = [make_entry(k) for k in (8, 3, 6, 1, 5, 4, 7)]
    heapify(lst)
    assert [entry_key(x) for x in lst] == [1, 3, 4, 8, 5, 6, 7]


def test_heapify_single_element_unchanged():
    lst = [make_entry(42)]
    heapify(lst)
    assert len(lst) == 1
    assert entry_key(lst[0]) == 42


def test_heapify_empty_list_no_op():
    lst = []
    heapify(lst)
    assert lst == []


def test_priority_queue_len_after_pushes():
    pq = PriorityQueue()
    pq.push(10, "a")
    pq.push(3, "b")
    pq.push(7, "c")
    assert len(pq) == 3


def test_priority_queue_pop_order():
    pq = PriorityQueue()
    pq.push(5, "five")
    pq.push(1, "one")
    pq.push(3, "three")
    assert entry_value(pq.pop()) == "one"
    assert entry_value(pq.pop()) == "three"
    assert entry_value(pq.pop()) == "five"


def test_priority_queue_pop_halves_len():
    pq = PriorityQueue()
    pq.push(2, "x")
    pq.push(1, "y")
    pq.pop()
    assert len(pq) == 1


def test_priority_queue_peek_does_not_remove():
    pq = PriorityQueue()
    pq.push(4, "four")
    pq.push(2, "two")
    first = pq.peek()
    assert entry_key(first) == 2
    assert len(pq) == 2
    assert entry_key(pq.peek()) == 2


def test_priority_queue_peek_on_empty_raises():
    pq = PriorityQueue()
    with pytest.raises(IndexError):
        pq.peek()


def test_priority_queue_pop_on_empty_raises():
    pq = PriorityQueue()
    with pytest.raises(IndexError):
        pq.pop()


def test_decrease_key_moves_entry_to_front():
    pq = PriorityQueue()
    a = pq.push(10, "alpha")
    pq.push(3, "beta")
    pq.push(7, "gamma")
    pq.decrease_key(a, 1)
    assert entry_value(pq.peek()) == "alpha"
    assert entry_key(pq.peek()) == 1


def test_decrease_key_reorders_pop_order():
    pq = PriorityQueue()
    a = pq.push(5, "a")
    pq.push(8, "b")
    pq.push(6, "c")
    pq.decrease_key(a, 2)
    got = [entry_value(pq.pop()) for _ in range(3)]
    assert got == ["a", "c", "b"]


def test_decrease_key_to_equal_raises():
    pq = PriorityQueue()
    a = pq.push(5, "a")
    with pytest.raises(ValueError):
        pq.decrease_key(a, 5)


def test_decrease_key_to_larger_raises():
    pq = PriorityQueue()
    a = pq.push(5, "a")
    with pytest.raises(ValueError):
        pq.decrease_key(a, 9)


def test_priority_queue_from_iterable():
    pq = PriorityQueue([(2, "x"), (1, "y"), (3, "z")])
    assert len(pq) == 3
    assert entry_value(pq.pop()) == "y"


def test_priority_queue_iteration_property_min_is_first():
    pq = PriorityQueue()
    for k in (9, 4, 13, 1, 7):
        pq.push(k, k * 10)
    m = pq.pop()
    assert entry_key(m) == 1
    assert entry_value(m) == 10


def test_contains_detects_entry():
    pq = PriorityQueue()
    a = pq.push(1, "kept")
    assert a in pq
    b = pq.push(2, "other")
    assert b in pq


def test_priority_queue_method_heapify_resets_size():
    pq = PriorityQueue()
    pq.push(3, "three")
    pq.push(1, "one")
    pq.heapify()
    assert len(pq) == 2
    assert entry_key(pq.peek()) == 1


def test_pops_from_lazy_insertion_match_sorted_order():
    pq = PriorityQueue()
    for k in (6, 2, 9, 2, 5, 1):
        pq.push(k, None)
    keys = []
    while len(pq) > 0:
        keys.append(entry_key(pq.pop()))
    assert keys == [1, 2, 2, 5, 6, 9]
