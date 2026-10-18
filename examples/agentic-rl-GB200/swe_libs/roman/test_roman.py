"""Test suite for the roman module."""

import pytest
from roman import (
    RangeError,
    ValidationError,
    is_ascii,
    is_valid,
    roman_range,
    to_arabic,
    to_roman,
)


def test_to_roman_min_value():
    assert to_roman(1) == "I"


def test_to_roman_max_value():
    assert to_roman(3999) == "MMMCMXCIX"


def test_to_roman_simple_subtractive():
    assert to_roman(4) == "IV"


def test_to_roman_compound_subtractive():
    assert to_roman(1994) == "MCMXCIV"


def test_to_roman_repeated_symbols():
    assert to_roman(3888) == "MMMDCCCLXXXVIII"


def test_to_roman_mid_value():
    assert to_roman(500) == "D"


def test_to_roman_range_error_low():
    with pytest.raises(RangeError):
        to_roman(0)


def test_to_roman_range_error_high():
    with pytest.raises(RangeError, match="4000"):
        to_roman(4000)


def test_to_roman_rejects_bool():
    with pytest.raises(TypeError):
        to_roman(True)


def test_to_arabic_simple():
    assert to_arabic("IX") == 9


def test_to_arabic_mixed_subtractive():
    assert to_arabic("MCM") == 1900


def test_to_arabic_lower_case():
    assert to_arabic("xiv") == 14


def test_to_arabic_rejects_empty():
    with pytest.raises(ValidationError):
        to_arabic("")


def test_to_arabic_rejects_invalid_char():
    with pytest.raises(ValidationError, match="Z"):
        to_arabic("MZ")


def test_to_arabic_rejects_noncanonical():
    with pytest.raises(ValidationError):
        to_arabic("IL")


def test_to_arabic_rejects_noncanonical_repeat():
    with pytest.raises(ValidationError):
        to_arabic("IIII")


def test_is_valid_true():
    assert is_valid("MMMCMXCIX") is True


def test_is_valid_false_nonroman():
    assert is_valid("ABC") is False


def test_is_valid_false_nonstring():
    assert is_valid(1234) is False


def test_is_ascii_true():
    assert is_ascii("CXLII") is True


def test_is_ascii_false():
    assert is_ascii("V\N{ARABIC-INDIC DIGIT ZERO}") is False


def test_roundtrip_consistency_small():
    for n in (1, 4, 5, 9, 10, 50, 100, 500, 1000):
        assert to_arabic(to_roman(n)) == n


def test_roman_range_start_and_end():
    it = roman_range()
    first = next(it)
    assert first == (1, "I")
    last = None
    for pair in it:
        last = pair
    assert last == (3999, "MMMCMXCIX")


def test_to_arabic_rejects_non_str():
    with pytest.raises(TypeError):
        to_arabic(1000)
