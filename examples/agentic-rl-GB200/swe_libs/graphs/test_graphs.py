import pytest
from graphs import (
    bfs_order,
    connected_components,
    degree_sequence,
    has_cycle,
    is_tree,
    shortest_path_length,
)


def test_bfs_order_simple_line():
    # 1-2-3-4 (undirected)
    g = {1: [2], 2: [1, 3], 3: [2, 4], 4: [3]}
    assert bfs_order(g, 1) == [1, 2, 3, 4]


def test_bfs_order_star():
    # star: 0 connected to 1, 2, 3
    g = {0: [1, 2, 3], 1: [0], 2: [0], 3: [0]}
    assert bfs_order(g, 0) == [0, 1, 2, 3]


def test_bfs_order_from_secondary_node():
    # line 0-1-2-3 starting at 2
    g = {0: [1], 1: [0, 2], 2: [1, 3], 3: [2]}
    assert bfs_order(g, 2) == [2, 1, 3, 0]


def test_bfs_order_missing_start_raises():
    g = {1: []}
    with pytest.raises(KeyError):
        bfs_order(g, 99)


def test_bfs_order_single_node():
    g = {7: []}
    assert bfs_order(g, 7) == [7]


def test_shortest_path_length_direct():
    g = {1: [2], 2: [1]}
    assert shortest_path_length(g, 1, 2) == 1


def test_shortest_path_length_same_node():
    g = {1: [2], 2: [1]}
    assert shortest_path_length(g, 1, 1) == 0


def test_shortest_path_length_three_edges():
    # 1-2-3-4
    g = {1: [2], 2: [1, 3], 3: [2, 4], 4: [3]}
    assert shortest_path_length(g, 1, 4) == 3


def test_shortest_path_length_empty_graph_raises():
    g = {}
    with pytest.raises(KeyError):
        shortest_path_length(g, 1, 2)


def test_connected_components_two_parts():
    # comp1: 1-2, comp2: 3-4-5
    g = {1: [2], 2: [1], 3: [4], 4: [3, 5], 5: [4]}
    assert connected_components(g) == [[1, 2], [3, 4, 5]]


def test_connected_components_with_isolated_node():
    g = {1: [2], 2: [1], 3: []}
    assert connected_components(g) == [[1, 2], [3]]


def test_connected_components_empty_graph():
    assert connected_components({}) == []


def test_has_cycle_square():
    # 0-1-2-3-0
    g = {0: [1, 3], 1: [0, 2], 2: [1, 3], 3: [2, 0]}
    assert has_cycle(g) is True


def test_has_cycle_no_cycle_tree():
    g = {1: [2], 2: [1, 3], 3: [2]}
    assert has_cycle(g) is False


def test_has_cycle_self_loop_is_cycle():
    g = {1: [1]}
    assert has_cycle(g) is True


def test_degree_sequence_ordering():
    g = {0: [1, 2, 3], 1: [0], 2: [0], 3: [0]}
    assert degree_sequence(g) == [3, 1, 1, 1]


def test_degree_sequence_of_isolates():
    g = {1: [], 2: [], 3: []}
    assert degree_sequence(g) == [0, 0, 0]


def test_is_tree_true_for_star():
    g = {0: [1, 2], 1: [0], 2: [0]}
    assert is_tree(g) is True


def test_is_tree_false_for_cycle():
    g = {0: [1, 2], 1: [0, 2], 2: [0, 1]}
    assert is_tree(g) is False


def test_is_tree_false_for_disconnected_acyclic():
    g = {1: [2], 2: [1], 3: [4], 4: [3]}
    assert is_tree(g) is False


def test_is_tree_false_for_empty_graph():
    assert is_tree({}) is False
