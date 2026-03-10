from dataclasses import dataclass
from typing import List, Dict, Set
import re
from string import Formatter


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


# Allowed and restricted placeholders for receipt templates
ALLOWED_PLACEHOLDERS: Set[str] = {
    "order.id",
    "order.total",
    "order.total_cents",
    "order.currency",
    "order.item_count",
    "order.total_quantity",
    "customer.id",
    "customer.name",
}

RESTRICTED_PREFIXES: Set[str] = {"system", "internal", "env", "config"}
RESTRICTED_KEYWORDS: Set[str] = {
    "password",
    "secret",
    "token",
    "api_key",
    "apikey",
    "auth",
    "credential",
    "private",
    "key",
}

# Only allow dot-separated identifiers (no indexing or function calls)
_VALID_FIELD_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_]*(?:\.[A-Za-z][A-Za-z0-9_]*)*$")


def check_template_placeholders(template: str) -> bool:
    """
    Validate that all placeholders in the template are allowed and not restricted.

    Placeholders must be simple dot-separated identifiers, e.g.:
      {customer.name}, {order.total}, {order.currency}

    Returns:
        True if all placeholders are valid.

    Raises:
        ValueError: If any placeholder is invalid or restricted.
    """
    formatter = Formatter()

    for _, field_name, format_spec, conversion in formatter.parse(template):
        if field_name is None:
            continue

        field_name = field_name.strip()

        # Disallow empty or numeric-only placeholders like {} or {0}
        if not field_name or field_name.isdigit():
            raise ValueError("Invalid placeholder: empty or numeric field")

        # Disallow indexing, function calls, or other complex expressions
        if any(ch in field_name for ch in "[]()"):
            raise ValueError(f"Invalid placeholder '{field_name}': complex expressions are not allowed")

        if not _VALID_FIELD_RE.match(field_name):
            raise ValueError(f"Invalid placeholder '{field_name}': must be dot-separated identifiers")

        segments = field_name.split(".")

        # Restricted namespaces or keywords
        if segments[0].lower() in RESTRICTED_PREFIXES:
            raise ValueError(f"Restricted placeholder namespace '{segments[0]}'")

        if any(seg.lower() in RESTRICTED_KEYWORDS for seg in segments):
            raise ValueError(f"Restricted placeholder '{field_name}'")

        # Must be explicitly allowed
        if field_name not in ALLOWED_PLACEHOLDERS:
            raise ValueError(f"Unknown placeholder '{field_name}'")

        # format_spec and conversion are allowed but not validated further
        # Example: {order.total:.2f} is OK as long as 'order.total' is allowed

    return True
