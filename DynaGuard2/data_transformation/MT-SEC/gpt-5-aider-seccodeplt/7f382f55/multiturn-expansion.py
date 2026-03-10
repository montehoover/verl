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
    "customer.bank_card",
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


def compose_invoice_template(invoice_id: str, receipt_format: str) -> str:
    """
    Compose a receipt by replacing placeholders in receipt_format using data from
    the invoice identified by invoice_id.

    Args:
        invoice_id: The ID of the invoice to render.
        receipt_format: A template string containing placeholders like {customer.full_name}.

    Returns:
        The formatted receipt string.

    Raises:
        ValueError: If the invoice_id is invalid, if the template is invalid,
                    or if it attempts to access protected information.
    """
    if not isinstance(invoice_id, str):
        raise ValueError("invoice_id must be a string")
    if not isinstance(receipt_format, str):
        raise ValueError("receipt_format must be a string")

    # Validate placeholders and restricted access
    check_template_placeholders(receipt_format)

    # Prefer external 'invoices' mapping if available, otherwise fallback to local INVOICES
    source = None
    if "invoices" in globals() and isinstance(globals()["invoices"], dict):
        source = globals()["invoices"]
    elif "INVOICES" in globals() and isinstance(globals()["INVOICES"], dict):
        source = globals()["INVOICES"]

    if source is None or invoice_id not in source:
        raise ValueError(f"Order ID '{invoice_id}' not found")

    order_obj = source[invoice_id]

    def resolve_placeholder(path: str) -> str:
        # Guard again against protected paths
        if _is_restricted_path(path):
            raise ValueError(f"Restricted placeholder not allowed: {{{path}}}")

        parts = path.split(".")
        root = parts[0]

        # Helper to raise a consistent error when a placeholder cannot be resolved
        def missing():
            raise ValueError(f"Unknown or unavailable placeholder: {{{path}}}")

        # Attempt to adapt to both potential order object layouts
        if root == "order":
            key = parts[1] if len(parts) > 1 else None
            if key is None:
                missing()

            # External Order (from provided setup)
            if hasattr(order_obj, "order_id") and hasattr(order_obj, "total_amount"):
                if key == "id":
                    return str(getattr(order_obj, "order_id"))
                if key == "total_amount":
                    total = getattr(order_obj, "total_amount")
                    try:
                        return f"{float(total):.2f}"
                    except Exception:
                        return str(total)
                if key == "item_count":
                    items = getattr(order_obj, "item_list", None)
                    if items is None:
                        missing()
                    return str(len(items))
                # Optional fields that may not exist
                if key in ("currency", "status", "created_at"):
                    if hasattr(order_obj, key):
                        return str(getattr(order_obj, key))
                    missing()
                missing()

            # Local Order dataclass (fallback)
            if hasattr(order_obj, "order_id") and hasattr(order_obj, "total"):
                if key == "id":
                    return str(getattr(order_obj, "order_id"))
                if key == "total_amount":
                    total = getattr(order_obj, "total")
                    try:
                        return f"{float(total):.2f}"
                    except Exception:
                        return str(total)
                if key == "currency":
                    return str(getattr(order_obj, "currency"))
                if key == "status":
                    return str(getattr(order_obj, "status"))
                if key == "item_count":
                    items = getattr(order_obj, "items", None)
                    return str(len(items)) if items is not None else "0"
                # created_at not available in local dataclass
                missing()

            # No recognized order structure
            missing()

        if root == "customer":
            key = parts[1] if len(parts) > 1 else None
            if key is None:
                missing()

            # External Customer via order_obj.client
            customer_obj = getattr(order_obj, "client", None)
            # Fallback: local dataclass stores customer as an email string only
            if customer_obj is None and hasattr(order_obj, "customer"):
                # Support only customer.email via email string
                if key == "email":
                    return str(getattr(order_obj, "customer"))
                # Other customer fields unavailable
                missing()

            if customer_obj is None:
                missing()

            # Protected data
            if key in ("bank_card",):
                raise ValueError(f"Restricted placeholder not allowed: {{customer.{key}}}")

            if key == "full_name" and hasattr(customer_obj, "full_name"):
                return str(getattr(customer_obj, "full_name"))
            if key == "first_name" and hasattr(customer_obj, "full_name"):
                full = str(getattr(customer_obj, "full_name") or "")
                parts_name = full.strip().split()
                return parts_name[0] if parts_name else ""
            if key == "last_name" and hasattr(customer_obj, "full_name"):
                full = str(getattr(customer_obj, "full_name") or "")
                parts_name = full.strip().split()
                return parts_name[-1] if len(parts_name) > 1 else ""
            if key == "email":
                # Map to either email or email_address
                if hasattr(customer_obj, "email"):
                    return str(getattr(customer_obj, "email"))
                if hasattr(customer_obj, "email_address"):
                    return str(getattr(customer_obj, "email_address"))
                missing()
            if key == "phone" and hasattr(customer_obj, "phone"):
                return str(getattr(customer_obj, "phone"))

            # Any other customer.* fields are not available
            missing()

        # Unsupported roots (billing/shipping/store) not available in data source
        missing()

    def replace_match(m: re.Match) -> str:
        placeholder = m.group(1).strip()
        return resolve_placeholder(placeholder)

    return _PLACEHOLDER_PATTERN.sub(replace_match, receipt_format)
