import pytest
from units import (
    UnitError,
    category,
    convert,
    from_base,
    known_units,
    parse,
    smart_convert,
    to_base,
)


def test_known_units_contains_expected():
    units = known_units()
    assert units[0] == "cm"
    assert units[-1] == "mg" or "K" in units
    assert "km" in units
    assert "degC" in units


def test_category_length():
    assert category("km") == "length"


def test_category_mass():
    assert category("kg") == "mass"


def test_category_temperature():
    assert category("degF") == "temperature"


def test_category_unknown_raises():
    with pytest.raises(UnitError):
        category("furlong")


def test_to_base_length():
    assert to_base(5, "km") == 5000.0


def test_to_base_mass():
    assert to_base(250, "g") == 0.25


def test_to_base_temperature_celsius():
    assert to_base(100, "degC") == 373.15


def test_from_base_temperature_fahrenheit():
    # 373.15 K -> degF: 373.15 * 9/5 - 459.67 = 212.0
    assert from_base(373.15, "degF") == pytest.approx(212.0)


def test_convert_length():
    assert convert(1500, "m", "km") == 1.5


def test_convert_mass():
    assert convert(2.5, "kg", "g") == 2500.0


def test_convert_temperature_c_to_f():
    assert convert(100, "degC", "degF") == pytest.approx(212.0)


def test_convert_temperature_f_to_c():
    assert convert(32, "degF", "degC") == pytest.approx(0.0)


def test_convert_incompatible_raises():
    with pytest.raises(UnitError):
        convert(1, "kg", "m")


def test_parse_valid():
    val, unit = parse("5 km")
    assert val == 5.0
    assert unit == "km"


def test_parse_negative():
    val, unit = parse("-2.5 degC")
    assert val == -2.5
    assert unit == "degC"


def test_parse_missing_space():
    with pytest.raises(ValueError):
        parse("5km")


def test_parse_bad_number():
    with pytest.raises(ValueError):
        parse("abc m")


def test_parse_no_unit():
    with pytest.raises(ValueError):
        parse("5.0")


def test_smart_convert_no_rounding():
    assert smart_convert("3 m", "cm") == 300.0


def test_smart_convert_with_rounding():
    assert smart_convert("5 km", "m", digits=0) == 5000.0


def test_smart_convert_rounding_temperature():
    result = smart_convert("75 degC", "degF", digits=1)
    assert result == 167.0


def test_smart_convert_negative_digits_raises():
    with pytest.raises(ValueError):
        smart_convert("1 m", "cm", digits=-1)


def test_smart_convert_incompatible_raises():
    with pytest.raises(UnitError):
        smart_convert("1 kg", "m")
