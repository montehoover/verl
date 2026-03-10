from dataclasses import dataclass
from typing import Dict, List


@dataclass(frozen=True)
class Order:
    id: str
    customer_name: str
    items: List[str]
    total: float
    status: str


# Predefined dictionary of orders
ORDERS: Dict[str, Order] = {
    "ORD-1001": Order(
        id="ORD-1001",
        customer_name="Alice Johnson",
        items=["SKU-001", "SKU-005"],
        total=79.98,
        status="processing",
    ),
    "ORD-1002": Order(
        id="ORD-1002",
        customer_name="Bob Smith",
        items=["SKU-002"],
        total=24.99,
        status="shipped",
    ),
    "ORD-1003": Order(
        id="ORD-1003",
        customer_name="Charlie Lee",
        items=["SKU-003", "SKU-004", "SKU-006"],
        total=149.47,
        status="delivered",
    ),
}


def get_order_by_id(order_id: str) -> Order:
    """
    Retrieve an Order by its ID.

    Args:
        order_id: The unique identifier of the order.

    Returns:
        The corresponding Order object.

    Raises:
        ValueError: If no order exists with the given ID.
    """
    try:
        return ORDERS[order_id]
    except KeyError:
        raise ValueError(f"Order with id {order_id!r} does not exist.")
