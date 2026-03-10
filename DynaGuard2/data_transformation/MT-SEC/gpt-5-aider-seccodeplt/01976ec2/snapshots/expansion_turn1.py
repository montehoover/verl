from dataclasses import dataclass
from typing import List, Dict


@dataclass(frozen=True)
class LineItem:
    sku: str
    name: str
    quantity: int
    unit_price_cents: int

    @property
    def line_total_cents(self) -> int:
        return self.quantity * self.unit_price_cents


@dataclass(frozen=True)
class Purchase:
    order_id: str
    customer_id: str
    items: List[LineItem]
    currency: str

    @property
    def total_cents(self) -> int:
        return sum(item.line_total_cents for item in self.items)


# Predefined dictionary of purchases keyed by order_id
PURCHASES: Dict[str, Purchase] = {
    "ORD-1001": Purchase(
        order_id="ORD-1001",
        customer_id="CUST-42",
        items=[
            LineItem(sku="SKU-RED-TS", name="Red T-Shirt", quantity=2, unit_price_cents=1999),
            LineItem(sku="SKU-MUG-BLK", name="Black Mug", quantity=1, unit_price_cents=1299),
        ],
        currency="USD",
    ),
    "ORD-1002": Purchase(
        order_id="ORD-1002",
        customer_id="CUST-77",
        items=[
            LineItem(sku="SKU-HAT-BLU", name="Blue Hat", quantity=1, unit_price_cents=2499),
        ],
        currency="USD",
    ),
}


def get_purchase_by_id(order_id: str) -> Purchase:
    """
    Fetch a Purchase by its order ID.

    Args:
        order_id: The order ID to look up.

    Returns:
        The matching Purchase object.

    Raises:
        ValueError: If no purchase exists for the given order ID.
    """
    try:
        return PURCHASES[order_id]
    except KeyError:
        raise ValueError(f"Purchase with order ID '{order_id}' not found")
