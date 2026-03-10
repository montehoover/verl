from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass(frozen=True)
class OrderItem:
    sku: str
    name: str
    quantity: int
    price: float  # price per unit


@dataclass(frozen=True)
class Order:
    id: str
    user_id: str
    items: List[OrderItem]
    total: float
    status: str


# Predefined dictionary of orders
ORDERS: Dict[str, Order] = {
    "ORD-1001": Order(
        id="ORD-1001",
        user_id="USR-001",
        items=[
            OrderItem(sku="SKU-RED-TSHIRT", name="Red T-Shirt", quantity=2, price=19.99),
            OrderItem(sku="SKU-BLK-MUG", name="Black Coffee Mug", quantity=1, price=49.50),
        ],
        total=39.98 + 49.50,
        status="processing",
    ),
    "ORD-1002": Order(
        id="ORD-1002",
        user_id="USR-002",
        items=[
            OrderItem(sku="SKU-NOTE-SET", name="Notebook Set", quantity=3, price=15.00),
        ],
        total=45.00,
        status="shipped",
    ),
}


def get_order_by_id(order_id: str) -> Order:
    """
    Fetch an order by its ID.

    Args:
        order_id: The unique order identifier.

    Returns:
        The corresponding Order object.

    Raises:
        ValueError: If the order ID is not found.
    """
    try:
        return ORDERS[order_id]
    except KeyError:
        raise ValueError(f"Order ID '{order_id}' not found") from None


__all__ = ["Order", "OrderItem", "ORDERS", "get_order_by_id"]
