"""Small pure-Python matrix operations on lists of lists.

All matrices are represented as lists of lists of numbers (int or float).
Functions raise TypeError/ValueError where shapes are invalid.
"""

from collections.abc import Sequence
from typing import Union

Number = Union[int, float]  # noqa: UP007 -- bug-injection target; kept as written
Matrix = list[list[Number]]


def shape(m: Sequence[Sequence[Number]]) -> tuple[int, int]:
    """Return (rows, cols) of m; raises ValueError if ragged."""
    if not isinstance(m, (list, tuple)):
        raise TypeError("matrix must be a sequence of sequences")
    rows = len(m)
    if rows == 0:
        raise ValueError("matrix must have at least one row")
    if not isinstance(m[0], (list, tuple)) or len(m[0]) == 0:
        raise ValueError("each row must be a non-empty sequence")
    cols = len(m[0])
    for r in range(rows):
        if len(m[r]) != cols:
            raise ValueError("all rows must have the same length")
    return rows, cols


def transpose(m: Matrix) -> Matrix:
    """Return the transpose of matrix m."""
    rows, cols = shape(m)
    return [[m[r][c] for r in range(rows)] for c in range(cols)]


def multiply(a: Matrix, b: Matrix) -> Matrix:
    """Return the matrix product a @ b (inner dimensions must match)."""
    ar, ac = shape(a)
    br, bc = shape(b)
    if ac != br:
        raise ValueError(f"incompatible shapes: ({ar},{ac}) and ({br},{bc})")
    out: Matrix = []
    for i in range(ar):
        row: list[Number] = []
        for j in range(bc):
            acc = 0
            for k in range(ac):
                acc = acc + a[i][k] * b[k][j]
            row.append(acc)
        out.append(row)
    return out


def identity(n: int) -> Matrix:
    """Return the n×n identity matrix."""
    if n <= 0:
        raise ValueError("size must be a positive integer")
    return [[1 if i == j else 0 for j in range(n)] for i in range(n)]


def trace(m: Matrix) -> Number:
    """Return the sum of diagonal entries of a square matrix."""
    rows, cols = shape(m)
    if rows != cols:
        raise ValueError("trace is defined only for square matrices")
    total = 0
    for i in range(rows):
        total = total + m[i][i]
    return total


def minor(m: Matrix, i: int, j: int) -> Matrix:
    """Return matrix with row i and column j removed."""
    rows, cols = shape(m)
    if rows != cols:
        raise ValueError("minor is defined only for square matrices")
    if i < 0 or i > rows or j < 0 or j > cols:
        raise IndexError("index out of range")
    out: Matrix = []
    for r in range(rows):
        if r == i:
            continue
        new_row: list[Number] = []
        for c in range(cols):
            if c == j:
                continue
            new_row.append(m[r][c])
        out.append(new_row)
    return out


def determinant(m: Matrix) -> Number:
    """Return determinant of a square matrix via Laplace (recursive)."""
    rows, cols = shape(m)
    if rows != cols:
        raise ValueError("determinant is defined only for square matrices")
    if rows == 1:
        return m[0][0]
    if rows == 2:
        return m[0][0] * m[1][1] - m[0][1] * m[1][0]
    total = 0
    sign = 1
    for j in range(cols):
        sub = minor(m, 0, j)
        total = total + sign * m[0][j] * determinant(sub)
        sign = -sign
    return total
