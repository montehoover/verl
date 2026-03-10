from dataclasses import dataclass
from typing import List, Dict
import re
from string import Formatter


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


# Allowed and restricted placeholders for receipt templates
ALLOWED_PLACEHOLDERS = {
    "order.id",
    "order.total",
    "order.status",
    "customer.name",
    "customer.email",
}
RESTRICTED_PREFIXES = {
    "order.items",
    "order.customer_id",
    "customer.password",
    "customer.ssn",
}

_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _is_valid_field_path(field: str) -> bool:
    """
    Validate that the field path is composed of dot-separated identifiers.
    Example: 'order.total' or 'customer.name'
    """
    if not field or "[" in field or "]" in field:
        return False
    parts = field.split(".")
    return all(_IDENTIFIER_RE.match(p) for p in parts)


def check_template_placeholders(template: str) -> bool:
    """
    Verify that all placeholders in the template are valid and not restricted.

    A placeholder is any field enclosed in braces, e.g., {order.total} or {customer.name}.
    Format specs like {order.total:.2f} are allowed; only the field name is validated.

    Returns:
        True if all placeholders are valid.

    Raises:
        ValueError: If any placeholder is invalid or restricted.
    """
    formatter = Formatter()
    invalid: List[str] = []
    restricted: List[str] = []

    for _, field_name, _, _ in formatter.parse(template):
        if field_name is None:
            continue  # literal text (including escaped braces)
        # Ignore conversion and format spec (already separated by Formatter)
        field = field_name

        # Basic structure validation
        if not _is_valid_field_path(field):
            invalid.append(field)
            continue

        # Restricted prefixes
        if any(field == rp or field.startswith(rp + ".") for rp in RESTRICTED_PREFIXES):
            restricted.append(field)
            continue

        # Whitelist validation
        if field not in ALLOWED_PLACEHOLDERS:
            invalid.append(field)

    if restricted or invalid:
        messages = []
        if restricted:
            messages.append(f"restricted: {', '.join(sorted(set(restricted)))}")
        if invalid:
            messages.append(f"invalid: {', '.join(sorted(set(invalid)))}")
        raise ValueError("Invalid or restricted placeholders detected - " + "; ".join(messages))

    return True
