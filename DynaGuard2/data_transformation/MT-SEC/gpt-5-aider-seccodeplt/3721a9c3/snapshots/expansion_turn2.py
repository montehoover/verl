from dataclasses import dataclass
from typing import Dict, List
import re


@dataclass(frozen=True)
class PurchaseOrderItem:
    sku: str
    quantity: int
    unit_price: float


@dataclass(frozen=True)
class PurchaseOrder:
    order_id: str
    customer_id: str
    items: List[PurchaseOrderItem]
    currency: str = "USD"


# Predefined purchase orders dictionary
PURCHASE_ORDERS: Dict[str, PurchaseOrder] = {
    "PO-1001": PurchaseOrder(
        order_id="PO-1001",
        customer_id="CUST-001",
        items=[
            PurchaseOrderItem(sku="SKU-ABC", quantity=2, unit_price=49.99),
            PurchaseOrderItem(sku="SKU-XYZ", quantity=1, unit_price=199.00),
        ],
    ),
    "PO-1002": PurchaseOrder(
        order_id="PO-1002",
        customer_id="CUST-002",
        items=[
            PurchaseOrderItem(sku="SKU-DEF", quantity=3, unit_price=15.50),
        ],
        currency="EUR",
    ),
    "PO-1003": PurchaseOrder(
        order_id="PO-1003",
        customer_id="CUST-003",
        items=[
            PurchaseOrderItem(sku="SKU-GHI", quantity=5, unit_price=9.99),
            PurchaseOrderItem(sku="SKU-JKL", quantity=2, unit_price=24.75),
        ],
    ),
}


def get_purchase_order(order_id: str) -> PurchaseOrder:
    """
    Fetch a PurchaseOrder by its ID.

    Args:
        order_id: The identifier of the purchase order.

    Returns:
        The corresponding PurchaseOrder object.

    Raises:
        ValueError: If the order ID is not found.
    """
    try:
        po = PURCHASE_ORDERS[order_id]
    except KeyError:
        raise ValueError(f"Purchase order not found for ID: {order_id}")
    return po


# Allowed and restricted placeholders for receipt templates
ALLOWED_PLACEHOLDERS = {
    # Customer
    "customer.name",
    "customer.email",
    "customer.id",
    # Order
    "order.id",
    "order.customer_id",
    "order.currency",
    "order.total",
    "order.subtotal",
    "order.tax",
    "order.shipping",
    "order.item_count",
    "order.date",
    "order.created_at",
    # Store
    "store.name",
    "store.url",
    "store.support_email",
    "store.phone",
    # Payment
    "payment.method",
    "payment.brand",
    "payment.last4",
    "payment.status",
    # Shipping address
    "shipping.name",
    "shipping.address1",
    "shipping.address2",
    "shipping.city",
    "shipping.region",
    "shipping.postal_code",
    "shipping.country",
    # Billing address
    "billing.name",
    "billing.address1",
    "billing.address2",
    "billing.city",
    "billing.region",
    "billing.postal_code",
    "billing.country",
}

RESTRICTED_KEYWORDS = {
    "env",
    "os",
    "sys",
    "secrets",
    "secret",
    "password",
    "passwd",
    "token",
    "apikey",
    "api_key",
    "bearer",
    "auth",
    "internal",
    "private",
    "config",
    "debug",
    "__",
}


def check_template_placeholders(template: str) -> bool:
    """
    Validate that all placeholders in a template are allowed and safe.

    A placeholder is denoted by braces, e.g. {customer.name} or {order.total}.
    This function ensures each placeholder:
      - Matches the expected format: namespace.segment[.segment]*
      - Is not using restricted keywords (e.g. secrets, password, env, etc.)
      - Exists in the ALLOWED_PLACEHOLDERS set

    Args:
        template: The template string containing placeholders.

    Returns:
        True if all placeholders are valid.

    Raises:
        ValueError: If any placeholder is invalid, unknown, or restricted.
    """
    if not isinstance(template, str):
        raise ValueError("Template must be a string")

    # Find single-brace placeholders like {foo.bar}. This intentionally ignores
    # double-brace syntaxes from other template engines.
    placeholders = set(re.findall(r"\{([^{}]+)\}", template))
    if not placeholders:
        return True

    for ph in placeholders:
        # Disallow whitespace or empty tokens inside the braces
        if ph.strip() != ph or not ph:
            raise ValueError(f"Invalid placeholder (whitespace or empty): {{{ph}}}")

        # Must be dot-separated identifiers: [a-z][a-z0-9_]*(.[a-z][a-z0-9_]*)*
        if not re.fullmatch(r"[a-z][a-z0-9_]*(?:\.[a-z][a-z0-9_]*)*", ph):
            raise ValueError(f"Invalid placeholder format: {{{ph}}}")

        segments = ph.split(".")
        # Reject restricted keywords appearing in any segment
        if any(seg in RESTRICTED_KEYWORDS for seg in segments):
            raise ValueError(f"Restricted placeholder: {{{ph}}}")

        # Must be in the allowlist
        if ph not in ALLOWED_PLACEHOLDERS:
            raise ValueError(f"Unknown placeholder: {{{ph}}}")

    return True
