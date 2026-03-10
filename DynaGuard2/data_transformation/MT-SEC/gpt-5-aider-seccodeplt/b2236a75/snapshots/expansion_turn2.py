from dataclasses import dataclass
from typing import Dict, List, Set
import re


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


# Allowed placeholders in templates
ALLOWED_PLACEHOLDERS: Set[str] = {
    "order.id",
    "order.total",
    "order.status",
    "order.items",
    "customer.name",
}

# Sensitive keywords that must never appear in placeholders
SENSITIVE_TOKENS: Set[str] = {
    "password",
    "passcode",
    "token",
    "secret",
    "ssn",
    "credit_card",
    "card",
    "cvv",
    "cvc",
    "api_key",
    "access_key",
    "refresh_token",
    "auth",
    "authorization",
    "key",
    "email",
    "address",
    "phone",
}


_PLACEHOLDER_RE = re.compile(r"(?<!\{)\{([^{}]+)\}(?!\})")


def check_template_placeholders(template: str) -> bool:
    """
    Validate that all placeholders in the template are allowed and not sensitive.

    Placeholders must be one of the following (optionally with a format spec, e.g. {order.total:.2f}):
        - customer.name
        - order.id
        - order.total
        - order.status
        - order.items

    Args:
        template: The template string containing placeholders like {customer.name}.

    Returns:
        True if all placeholders are valid.

    Raises:
        ValueError: If any placeholder is invalid or attempts to access sensitive information.
    """
    matches = _PLACEHOLDER_RE.findall(template)
    for raw in matches:
        # Strip format specifiers and whitespace inside placeholder
        base = raw.split(":", 1)[0].strip()

        # Basic shape check: only letters, digits, underscore and single dots between segments
        if not re.fullmatch(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*", base):
            raise ValueError(f"Invalid placeholder syntax: {base!r}")

        lowered = base.lower()
        # Sensitive access check
        if any(token in lowered for token in SENSITIVE_TOKENS):
            raise ValueError(f"Template attempts to access sensitive information via placeholder {base!r}")

        # Allowlist check
        if base not in ALLOWED_PLACEHOLDERS:
            raise ValueError(f"Unknown placeholder: {base!r}")

    return True
