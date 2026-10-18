import pytest
from calc import (
    CalcError,
    Parser,
    Token,
    calculate,
    parse,
    tokenize,
)

# --------------------------------------------------------------------------- #
# Tokenizer tests
# --------------------------------------------------------------------------- #


def test_tokenize_simple_number():
    toks = tokenize("42")
    assert toks[-1].kind == "END"
    assert toks[0] == Token("NUM", "42", 0)


def test_tokenize_mixed_operators():
    toks = [t for t in tokenize("3+4") if t.kind != "END"]
    assert [(t.kind, t.value) for t in toks] == [
        ("NUM", "3"),
        ("OP", "+"),
        ("NUM", "4"),
    ]


def test_tokenize_parens_and_space():
    toks = [t for t in tokenize("( 12 )") if t.kind != "END"]
    assert [t.kind for t in toks] == ["LPAREN", "NUM", "RPAREN"]
    assert toks[1].value == "12"


def test_tokenize_decimal():
    toks = tokenize("1.25")
    assert toks[0].value == "1.25"


def test_tokenize_rejects_bad_character():
    with pytest.raises(CalcError):
        tokenize("2 @ 3")


# --------------------------------------------------------------------------- #
# Parser / evaluate tests
# --------------------------------------------------------------------------- #


def test_addition():
    assert calculate("1+2") == 3.0


def test_multiplication_precedence():
    assert calculate("2+3*4") == 14.0


def test_division_precedence():
    assert calculate("12/3-2") == 2.0


def test_parentheses_override_precedence():
    assert calculate("(2+3)*4") == 20.0


def test_left_associative_subtraction():
    assert calculate("10-3-2") == 5.0


def test_left_associative_division():
    assert calculate("100/10/2") == 5.0


def test_unary_minus():
    assert calculate("-5") == -5.0


def test_double_unary_minus():
    assert calculate("--5") == 5.0


def test_unary_minus_in_expression():
    assert calculate("2--3") == 5.0


def test_unary_plus():
    assert calculate("+7") == 7.0


def test_nested_parens():
    assert calculate("((1+2)*3)") == 9.0


def test_decimal_result():
    assert calculate("1/3*3") == 1.0


def test_division_by_zero_raises():
    with pytest.raises(CalcError, match="zero"):
        calculate("5/0")


def test_parse_empty_raises():
    with pytest.raises(CalcError):
        parse("")


def test_unbalanced_paren_raises():
    with pytest.raises(CalcError):
        calculate("(1+2")


def test_trailing_operator_raises():
    with pytest.raises(CalcError):
        calculate("1+*2")


def test_extra_token_raises():
    with pytest.raises(CalcError):
        calculate("3 4")


def test_parser_start_node_consumes_to_end():
    p = Parser(tokenize("2*5"))
    assert p.parse() == 10.0
