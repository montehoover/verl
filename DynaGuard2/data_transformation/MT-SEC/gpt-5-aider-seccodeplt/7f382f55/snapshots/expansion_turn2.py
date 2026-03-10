from dataclasses import dataclass
from typing import List, Dict
import re


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


# Allowed placeholder schema for template validation
_ALLOWED_PLACEHOLDERS = {
    "customer": {
        "full_name": True,
        "first_name": True,
        "last_name": True,
        "email": True,
        "phone": True,
    },
    "order": {
        "id": True,
        "total_amount": True,
        "currency": True,
        "status": True,
        "item_count": True,
        "created_at": True,
    },
    "billing": {
        "address_line1": True,
        "address_line2": True,
        "city": True,
        "state": True,
        "postal_code": True,
        "country": True,
        "full_name": True,
    },
    "shipping": {
        "address_line1": True,
        "address_line2": True,
        "city": True,
        "state": True,
        "postal_code": True,
        "country": True,
        "full_name": True,
        "method": True,
        "tracking_number": True,
    },
    "store": {
        "name": True,
        "url": True,
        "support_email": True,
    },
}

# Restricted placeholders that must not appear. If a placeholder equals one of these
# or uses it as a prefix (e.g., "order.payment.card_number"), it's considered restricted.
_RESTRICTED_PLACEHOLDER_PREFIXES = {
    "customer.password",
    "customer.hashed_password",
    "customer.ssn",
    "customer.token",
    "customer.auth_token",
    "order.payment.card_number",
    "order.payment.cvv",
    "order.payment.cvc",
    "order.payment.pan",
    "order.payment.token",
    "order.internal_notes",
    "order.secret",
    "order.auth_code",
}


_PLACEHOLDER_PATTERN = re.compile(r"\{([^{}]+)\}")
_IDENTIFIER_PATTERN = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*(\.[a-zA-Z_][a-zA-Z0-9_]*)*$")


def _is_allowed_path(path: str) -> bool:
    """
    Check if the dot-delimited path is allowed based on _ALLOWED_PLACEHOLDERS.
    """
    parts = path.split(".")
    node = _ALLOWED_PLACEHOLDERS
    for i, part in enumerate(parts):
        if not isinstance(node, dict) or part not in node:
            return False
        node = node[part]
        if node is True:
            # If this is the last part, it's valid; otherwise, more nesting isn't allowed
            return i == len(parts) - 1
    # Ended on a dict, which means expecting further properties that weren't provided
    return False


def _is_restricted_path(path: str) -> bool:
    for prefix in _RESTRICTED_PLACEHOLDER_PREFIXES:
        if path == prefix or path.startswith(prefix + "."):
            return True
    return False


def check_template_placeholders(template: str) -> bool:
    """
    Verify that all placeholders in the template are valid and not restricted.

    A valid placeholder:
      - Is enclosed in single braces like {customer.full_name}
      - Consists of identifiers and dots: [a-zA-Z_][a-zA-Z0-9_]* segments
      - Matches an allowed path in the _ALLOWED_PLACEHOLDERS schema

    Raises:
        ValueError: If an invalid or restricted placeholder is encountered.

    Returns:
        bool: True if all placeholders are valid.
    """
    if not isinstance(template, str):
        raise ValueError("Template must be a string")

    placeholders = _PLACEHOLDER_PATTERN.findall(template)
    for raw in placeholders:
        placeholder = raw.strip()

        if not placeholder:
            raise ValueError("Empty placeholder {} is not allowed")

        if not _IDENTIFIER_PATTERN.match(placeholder):
            raise ValueError(f"Invalid placeholder syntax: {{{placeholder}}}")

        if _is_restricted_path(placeholder):
            raise ValueError(f"Restricted placeholder not allowed: {{{placeholder}}}")

        if not _is_allowed_path(placeholder):
            raise ValueError(f"Unknown placeholder: {{{placeholder}}}")

    return True
