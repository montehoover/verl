from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List
import re


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


# Template placeholder validation
_PLACEHOLDER_PATTERN = re.compile(r"\{([^{}]+)\}")

# Allowed placeholders in templates
VALID_TEMPLATE_PLACEHOLDERS = {
    # Customer
    "customer.id",
    "customer.full_name",
    "customer.first_name",
    "customer.last_name",
    "customer.email",
    # Order
    "order.id",
    "order.user_id",
    "order.status",
    "order.total",        # aligns with Order.total
    "order.price_total",  # accepted alias commonly used in templates
    "order.item_count",
}

# Restricted placeholders that must never be used
RESTRICTED_TEMPLATE_PLACEHOLDERS = {
    "payment.card_number",
    "payment.cvv",
    "customer.password",
    "customer.ssn",
    "order.internal_notes",
}


def check_template_placeholders(template: str) -> bool:
    """
    Verify that all placeholders in a template string are valid and not restricted.

    A placeholder is any token inside single curly braces, e.g., {customer.full_name}.

    Returns:
        True if all placeholders are valid.

    Raises:
        ValueError: If a placeholder is invalid or restricted.
    """
    if template is None:
        raise ValueError("Template must be a string")

    placeholders = set()
    for match in _PLACEHOLDER_PATTERN.finditer(template):
        token = match.group(1).strip()
        # skip empty tokens like "{}"
        if not token:
            raise ValueError("Empty placeholder {} is not allowed")
        placeholders.add(token)

    # No placeholders is considered valid
    for token in placeholders:
        if token in RESTRICTED_TEMPLATE_PLACEHOLDERS:
            raise ValueError(f"Restricted placeholder used: {token}")
        if token not in VALID_TEMPLATE_PLACEHOLDERS:
            raise ValueError(f"Invalid placeholder: {token}")

    return True


__all__ = [
    "Order",
    "OrderItem",
    "ORDERS",
    "get_order_by_id",
    "check_template_placeholders",
    "VALID_TEMPLATE_PLACEHOLDERS",
    "RESTRICTED_TEMPLATE_PLACEHOLDERS",
]
