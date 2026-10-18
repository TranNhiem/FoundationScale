import pytest
from intervals import (
    complement,
    covered_length,
    insert_interval,
    intersect,
    merge_intervals,
    normalize,
    subtract,
    union,
)


def test_normalize_sorts_pairs():
    assert normalize([(5, 6), (1, 3)]) == [(1, 3), (5, 6)]


def test_normalize_keeps_gap():
    assert normalize([(2, 5), (2, 4)]) == [(2, 4), (2, 5)]


def test_normalize_rejects_reversed():
    with pytest.raises(ValueError):
        normalize([(4, 1)])


def test_normalize_rejects_non_pair():
    with pytest.raises(TypeError):
        normalize([(1, 2, 3)])


def test_merge_intervals_empty():
    assert merge_intervals([]) == []


def test_merge_intervals_single():
    assert merge_intervals([(2, 4)]) == [(2, 4)]


def test_merge_intervals_full_overlap():
    assert merge_intervals([(1, 5), (2, 3)]) == [(1, 5)]


def test_merge_intervals_touching():
    assert merge_intervals([(0, 1), (1, 2), (2, 3)]) == [(0, 3)]


def test_merge_intervals_gap_kept():
    assert merge_intervals([(0, 1), (3, 4)]) == [(0, 1), (3, 4)]


def test_merge_intervals_extends_end():
    assert merge_intervals([(1, 3), (2, 7)]) == [(1, 7)]


def test_insert_interval_into_gap():
    assert insert_interval([(0, 1), (5, 6)], (2, 4)) == [(0, 1), (2, 4), (5, 6)]


def test_insert_interval_bridges_gap():
    assert insert_interval([(0, 1), (5, 6)], (1, 5)) == [(0, 6)]


def test_insert_interval_inside_existing():
    assert insert_interval([(2, 8)], (3, 5)) == [(2, 8)]


def test_insert_interval_after_all():
    assert insert_interval([(0, 1)], (9, 12)) == [(0, 1), (9, 12)]


def test_covered_length_disjoint():
    assert covered_length([(0, 3), (5, 8)]) == 6


def test_covered_length_overlapping():
    assert covered_length([(0, 4), (2, 6)]) == 6


def test_intersect_disjoint_is_empty():
    assert intersect([(0, 1)], [(3, 4)]) == []


def test_intersect_partial():
    assert intersect([(0, 5)], [(3, 8)]) == [(3, 5)]


def test_intersect_covered_strips():
    assert intersect([(0, 2), (5, 7), (9, 10)], [(1, 8)]) == [(1, 2), (5, 7)]


def test_intersect_point_overlap_is_dropped():
    assert intersect([(0, 3)], [(3, 5)]) == []


def test_union_combines_and_sorts():
    assert union([(3, 4)], [(0, 3)]) == [(0, 4)]


def test_subtract_middle_section():
    assert subtract([(0, 10)], [(4, 6)]) == [(0, 4), (6, 10)]


def test_complement_within_universe():
    assert complement([(3, 4)], 0, 8) == [(0, 3), (4, 8)]


def test_complement_rejects_bad_universe():
    with pytest.raises(ValueError):
        complement([], 7, 2)
