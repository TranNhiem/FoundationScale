import pytest
from inventory import Inventory, Item


def test_item_value():
    it = Item("widget", 3, 2.5)
    assert it.value() == 7.5


def test_add_new_item():
    inv = Inventory()
    inv.add("apple", 10, 0.5)
    assert inv.quantity_of("apple") == 10
    assert inv.price_of("apple") == 0.5


def test_add_existing_increases_quantity():
    inv = Inventory()
    inv.add("apple", 10, 0.5)
    inv.add("apple", 5, 0.75)
    assert inv.quantity_of("apple") == 15
    assert inv.price_of("apple") == 0.75


def test_add_rejects_negative_quantity():
    inv = Inventory()
    with pytest.raises(ValueError):
        inv.add("x", -1, 1.0)


def test_add_rejects_zero_price():
    inv = Inventory()
    with pytest.raises(ValueError):
        inv.add("x", 5, 0.0)


def test_remove_decreases_quantity():
    inv = Inventory()
    inv.add("apple", 10, 0.5)
    inv.remove("apple", 4)
    assert inv.quantity_of("apple") == 6


def test_remove_all_deletes_item():
    inv = Inventory()
    inv.add("apple", 3, 0.5)
    inv.remove("apple", 3)
    assert inv.quantity_of("apple") == 0


def test_remove_unknown_item_raises():
    inv = Inventory()
    with pytest.raises(KeyError):
        inv.remove("ghost", 1)


def test_remove_too_many_raises():
    inv = Inventory()
    inv.add("apple", 2, 0.5)
    with pytest.raises(ValueError):
        inv.remove("apple", 5)


def test_remove_zero_quantity_raises():
    inv = Inventory()
    inv.add("apple", 2, 0.5)
    with pytest.raises(ValueError):
        inv.remove("apple", 0)


def test_low_stock_report():
    inv = Inventory()
    inv.add("a", 3, 1.0)
    inv.add("b", 1, 1.0)
    inv.add("c", 7, 1.0)
    assert inv.low_stock(3) == ["a", "b"]


def test_low_stock_includes_equal_to_threshold():
    inv = Inventory()
    inv.add("a", 5, 1.0)
    inv.add("b", 6, 1.0)
    assert inv.low_stock(5) == ["a"]


def test_low_stock_negative_threshold_raises():
    inv = Inventory()
    with pytest.raises(ValueError):
        inv.low_stock(-1)


def test_total_value_no_discount():
    inv = Inventory()
    inv.add("a", 2, 3.0)
    inv.add("b", 1, 4.0)
    assert inv.total_value() == 10.0


def test_total_value_with_discount():
    inv = Inventory()
    inv.add("a", 4, 2.5)
    inv.add("b", 2, 5.0)
    assert inv.total_value(10) == 18.0


def test_total_value_full_discount():
    inv = Inventory()
    inv.add("a", 100, 1.5)
    assert inv.total_value(100) == 0.0


def test_total_value_discount_out_of_range():
    inv = Inventory()
    inv.add("a", 1, 1.0)
    with pytest.raises(ValueError):
        inv.total_value(101)
    with pytest.raises(ValueError):
        inv.total_value(-5)


def test_quantity_of_unknown_is_zero():
    inv = Inventory()
    assert inv.quantity_of("nothing") == 0


def test_price_of_unknown_raises():
    inv = Inventory()
    with pytest.raises(KeyError):
        inv.price_of("nothing")


def test_names_sorted():
    inv = Inventory()
    inv.add("zeta", 1, 1.0)
    inv.add("alpha", 1, 1.0)
    inv.add("mid", 1, 1.0)
    assert inv.names() == ["alpha", "mid", "zeta"]


def test_inventory_starts_empty():
    inv = Inventory()
    assert inv.names() == []
    assert inv.total_value() == 0.0


def test_add_positive_zero_quantity_rejected():
    inv = Inventory()
    with pytest.raises(ValueError):
        inv.add("a", 0, 1.0)
