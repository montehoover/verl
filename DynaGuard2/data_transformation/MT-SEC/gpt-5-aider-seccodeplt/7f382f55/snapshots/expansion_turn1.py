from dataclasses import dataclass
from typing import List, Dict


@dataclass(frozen=True)
class Order:
    order_id: str
    items: List[str]
    total: float
    currency: str
    customer: str
    status: str


# Predefined dictionary of invoices mapping order IDs to Order objects
INVOICES: Dict[str, Order] = {
    "ORD-1001": Order(
        order_id="ORD-1001",
        items=["SKU-001", "SKU-002"],
        total=49.98,
        currency="USD",
        customer="alice@example.com",
        status="paid",
    ),
    "ORD-1002": Order(
        order_id="ORD-1002",
        items=["SKU-010"],
        total=19.99,
        currency="USD",
        customer="bob@example.com",
        status="processing",
    ),
    "ORD-1003": Order(
        order_id="ORD-1003",
        items=["SKU-020", "SKU-021", "SKU-022"],
        total=129.97,
        currency="USD",
        customer="carol@example.com",
        status="shipped",
    ),
}


def get_order_by_id(order_id: str) -> Order:
    """
    Retrieve an Order by its ID from the predefined INVOICES dictionary.

    Raises:
        ValueError: If the order_id is not found.
    """
    key = str(order_id)
    try:
        return INVOICES[key]
    except KeyError:
        raise ValueError(f"Order ID '{order_id}' not found")
