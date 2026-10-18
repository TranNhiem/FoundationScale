import pytest
from matrix import (
    determinant,
    identity,
    minor,
    multiply,
    shape,
    trace,
    transpose,
)


def test_shape_basic():
    assert shape([[1, 2], [3, 4], [5, 6]]) == (3, 2)


def test_shape_1x3():
    assert shape([[9, 8, 7]]) == (1, 3)


def test_shape_ragged_raises():
    with pytest.raises(ValueError):
        shape([[1, 2], [3]])


def test_shape_empty_row_raises():
    with pytest.raises(ValueError):
        shape([[], []])


def test_transpose_2x3():
    assert transpose([[1, 2, 3], [4, 5, 6]]) == [[1, 4], [2, 5], [3, 6]]


def test_transpose_square():
    assert transpose([[1, 2], [3, 4]]) == [[1, 3], [2, 4]]


def test_transpose_1x1():
    assert transpose([[7]]) == [[7]]


def test_multiply_2x2():
    a = [[1, 2], [3, 4]]
    b = [[5, 6], [7, 8]]
    assert multiply(a, b) == [[19, 22], [43, 50]]


def test_multiply_1x3_by_3x2():
    a = [[1, 2, 3]]
    b = [[4, 5], [6, 7], [8, 9]]
    assert multiply(a, b) == [[40, 50]]


def test_multiply_shape_mismatch_raises():
    with pytest.raises(ValueError):
        multiply([[1, 2, 3]], [[1, 2], [3, 4]])


def test_identity_3():
    assert identity(3) == [[1, 0, 0], [0, 1, 0], [0, 0, 1]]


def test_identity_1():
    assert identity(1) == [[1]]


def test_identity_nonpositive_raises():
    with pytest.raises(ValueError):
        identity(0)


def test_multiply_by_identity_is_noop():
    m = [[2, 0, 1], [3, 4, 2]]
    assert multiply(m, identity(3)) == m


def test_trace_3x3():
    assert trace([[1, 2, 3], [4, 5, 6], [7, 8, 9]]) == 15


def test_trace_1x1():
    assert trace([[-5]]) == -5


def test_trace_non_square_raises():
    with pytest.raises(ValueError):
        trace([[1, 2, 3], [4, 5, 6]])


def test_minor_3x3():
    m = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    assert minor(m, 1, 1) == [[1, 3], [7, 9]]


def test_determinant_1x1():
    assert determinant([[42]]) == 42


def test_determinant_2x2():
    assert determinant([[1, 2], [3, 4]]) == -2


def test_determinant_3x3():
    m = [[6, 1, 1], [4, -2, 5], [2, 8, 7]]
    assert determinant(m) == -306


def test_determinant_4x4():
    m = [
        [1, 0, 2, -1],
        [3, 0, 0, 5],
        [2, 1, 4, -3],
        [1, 0, 5, 0],
    ]
    assert determinant(m) == 30


def test_determinant_non_square_raises():
    with pytest.raises(ValueError):
        determinant([[1, 2, 3], [4, 5, 6]])
