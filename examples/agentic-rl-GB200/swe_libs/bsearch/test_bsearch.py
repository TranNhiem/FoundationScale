"""pytest tests for bsearch module.

Each test checks exactly one behaviour with hand-computed expected
values.
"""

import pytest
from bsearch import (
    count,
    equal_range,
    index_of,
    insert_sorted,
    is_sorted,
    lower_bound,
    search_rotated,
    upper_bound,
)


def test_lower_bound_empty():
    assert lower_bound([], 5) == 0


def test_lower_bound_before_first():
    # all elements > target -> return 0
    assert lower_bound([3, 4, 5], 1) == 0


def test_lower_bound_after_last():
    # all elements < target -> return len
    assert lower_bound([1, 2, 3], 4) == 3


def test_lower_bound_with_duplicates():
    # first index where 5 is found
    assert lower_bound([1, 3, 5, 5, 9, 13], 5) == 2


def test_upper_bound_with_duplicates():
    # first index where value > 5
    assert upper_bound([1, 3, 5, 5, 9, 13], 5) == 4


def test_upper_bound_absent_value():
    # 7 is absent; first index with value > 7 is 4 (value 9)
    assert upper_bound([1, 3, 5, 5, 9, 13], 7) == 4


def test_index_of_present():
    assert index_of([2, 4, 6, 8, 10], 6) == 2


def test_index_of_absent_returns_minus_one():
    assert index_of([2, 4, 6, 8, 10], 7) == -1


def test_index_of_returns_first_occurrence():
    assert index_of([2, 2, 2, 2], 2) == 0


def test_count_zero():
    assert count([], 1) == 0


def test_count_three():
    assert count([1, 3, 3, 3, 5, 7], 3) == 3


def test_equal_range_exact_slice():
    assert equal_range([1, 3, 3, 3, 5, 7], 3) == (1, 4)


def test_insert_sorted_into_empty():
    assert insert_sorted([], 7) == [7]


def test_insert_sorted_after_duplicates():
    # stable: goes after existing 4s
    assert insert_sorted([1, 4, 4, 7], 4) == [1, 4, 4, 4, 7]


def test_insert_sorted_preserves_order():
    assert insert_sorted([1, 5, 9], 3) == [1, 3, 5, 9]


def test_is_sorted_false_at_inversion():
    assert is_sorted([1, 3, 2, 4]) is False


def test_is_sorted_empty_is_true():
    assert is_sorted([]) is True


def test_search_rotated_typical():
    # rotated by 3: [4,5,6,0,1,2]; target 6 is at index 2
    assert search_rotated([4, 5, 6, 0, 1, 2], 6) == 2


def test_search_rotated_after_pivot():
    # target 0 is at index 3 in [4,5,6,0,1,2]
    assert search_rotated([4, 5, 6, 0, 1, 2], 0) == 3


def test_search_rotated_absent():
    assert search_rotated([4, 5, 6, 0, 1, 2], 3) == -1


def test_search_rotated_single_element():
    assert search_rotated([7], 7) == 0
    assert search_rotated([7], 8) == -1


def test_search_rotated_not_rotated():
    # rotation by 0 -> plain sorted
    idx = search_rotated([1, 2, 3, 4, 5], 4)
    assert idx == 3


def test_lower_bound_exact_middle_value():
    # 5 in [1,2,3,5,7,8] is at position 3
    assert lower_bound([1, 2, 3, 5, 7, 8], 5) == 3


def test_errors_on_non_sequence():
    # Passing a non-sized type raises TypeError from len()
    with pytest.raises(TypeError):
        lower_bound(42, 1)
