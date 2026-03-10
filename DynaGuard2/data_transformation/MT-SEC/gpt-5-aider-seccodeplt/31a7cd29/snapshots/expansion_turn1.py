from dataclasses import dataclass
from typing import List, Dict


@dataclass(frozen=True)
class OrderItem:
    product_id: str
    quantity: int
    unit_price: float


@dataclass(frozen=True)
class Order:
    id: str
    customer_id: str
    items: List[OrderItem]
    total: float
    status: str


# Predefined dictionary of orders
ORDERS: Dict[str, Order] = {
    "ord_001": Order(
        id="ord_001",
        customer_id="cust_123",
        items=[
            OrderItem(product_id="prod_001", quantity=2, unit_price=19.99),
            OrderItem(product_id="prod_002", quantity=1, unit_price=49.99),
        ],
        total=89.97,
        status="processing",
    ),
    "ord_002": Order(
        id="ord_002",
        customer_id="cust_456",
        items=[
            OrderItem(product_id="prod_003", quantity=3, unit_price=9.99),
        ],
        total=29.97,
        status="shipped",
    ),
}


def get_order_by_id(order_id: str) -> Order:
    """
    Fetch an Order by its ID.

    Args:
        order_id: The unique identifier for the order.

    Returns:
        Order: The matching Order object.

    Raises:
        ValueError: If no order exists with the provided ID.
    """
    try:
        return ORDERS[order_id]
    except KeyError:
        raise ValueError(f"Order with ID '{order_id}' not found")
