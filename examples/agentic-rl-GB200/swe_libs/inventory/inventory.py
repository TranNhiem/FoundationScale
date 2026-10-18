"""A small inventory management library.

Tracks items with quantities and unit prices, supports restocking,
low-stock reporting, total value calculation, and percentage discounts.
"""

from dataclasses import dataclass


@dataclass
class Item:
    """An inventory item with a name, quantity and unit price."""

    name: str
    quantity: int
    price: float

    def value(self) -> float:
        """Return the total value of this item (quantity * price)."""
        return self.quantity * self.price


class Inventory:
    """A collection of named items with quantities and prices."""

    def __init__(self) -> None:
        self._items: dict[str, Item] = {}

    def add(self, name: str, quantity: int, price: float) -> None:
        """Add `quantity` of `name` at `price` each.

        If the item already exists, the quantity is increased and the
        price is updated to the new value. Negative quantities or
        non-positive prices are rejected.
        """
        if quantity <= 0:
            raise ValueError("quantity must be positive")
        if price <= 0:
            raise ValueError("price must be positive")
        if name in self._items:
            item = self._items[name]
            item.quantity += quantity
            item.price = price
        else:
            self._items[name] = Item(name, quantity, price)

    def remove(self, name: str, quantity: int) -> None:
        """Remove `quantity` of `name` from the inventory.

        Raises KeyError if the item is unknown and ValueError if the
        requested quantity is negative or exceeds the current stock.
        """
        if name not in self._items:
            raise KeyError(f"unknown item: {name}")
        if quantity <= 0:
            raise ValueError("quantity must be positive")
        item = self._items[name]
        if quantity > item.quantity:
            raise ValueError("not enough stock")
        item.quantity -= quantity
        if item.quantity == 0:
            del self._items[name]

    def low_stock(self, threshold: int) -> list[str]:
        """Return sorted names of items with quantity <= threshold."""
        if threshold < 0:
            raise ValueError("threshold must be non-negative")
        return sorted(name for name, it in self._items.items() if it.quantity <= threshold)

    def total_value(self, discount: float = 0.0) -> float:
        """Return sum of item values with an optional percentage discount.

        `discount` is between 0 and 100 inclusive, applied to the total.
        """
        if discount < 0 or discount > 100:
            raise ValueError("discount must be between 0 and 100")
        gross = sum(it.value() for it in self._items.values())
        net = gross * (1 - discount / 100.0)
        return net  # noqa: RET504 -- bug-injection target; kept as written

    def quantity_of(self, name: str) -> int:
        """Return the current quantity of `name` (0 if unknown)."""
        return self._items[name].quantity if name in self._items else 0

    def price_of(self, name: str) -> float:
        """Return the unit price of `name`, raising KeyError if unknown."""
        if name not in self._items:
            raise KeyError(f"unknown item: {name}")
        return self._items[name].price

    def names(self) -> list[str]:
        """Return all item names in alphabetical order."""
        return sorted(self._items.keys())
