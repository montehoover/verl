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
    "customer.address",
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


def generate_buyer_receipt(order_id: str, template: str) -> str:
    """
    Generate a formatted buyer receipt by replacing placeholders with order/customer details.

    Args:
        order_id: The order identifier.
        template: The template string containing placeholders like {customer.name}, {order.total}, etc.

    Returns:
        The formatted receipt string.

    Raises:
        ValueError: If the order ID is invalid or if a restricted/invalid placeholder is used.
    """
    if not isinstance(order_id, str) or not order_id:
        raise ValueError("Invalid order ID")

    # Validate template placeholders (also enforces restricted/unknown checks)
    check_template_placeholders(template)

    # Try to retrieve the order object from either a provided 'purchase_orders' mapping
    # or fall back to this module's PURCHASE_ORDERS.
    order_obj = None
    source = None
    if "purchase_orders" in globals() and isinstance(globals()["purchase_orders"], dict):
        source = globals()["purchase_orders"]
        order_obj = source.get(order_id)
    if order_obj is None:
        source = PURCHASE_ORDERS
        order_obj = source.get(order_id)

    if order_obj is None:
        raise ValueError(f"Invalid order ID: {order_id}")

    def only_digits(s: str) -> str:
        return "".join(ch for ch in s if ch.isdigit())

    # Build a context dictionary for replacements
    context: Dict[str, object] = {}

    # Populate from generic/external order shape if available
    # Expect attributes: id, customer (with name, email, address, credit_card), items, total, payment_method
    if hasattr(order_obj, "id") and hasattr(order_obj, "customer"):
        customer = getattr(order_obj, "customer", None)
        items = getattr(order_obj, "items", []) or []
        total = getattr(order_obj, "total", None)
        payment_method = getattr(order_obj, "payment_method", "")

        context.update(
            {
                "order.id": getattr(order_obj, "id", ""),
                "order.customer_id": getattr(getattr(order_obj, "customer", None), "id", "")
                or "",
                "order.currency": getattr(order_obj, "currency", "") if hasattr(order_obj, "currency") else "",
                "order.item_count": len(items),
                "order.subtotal": total if isinstance(total, (int, float)) else "",
                "order.tax": "",
                "order.shipping": "",
                "order.total": total if isinstance(total, (int, float)) else "",
                "customer.name": getattr(customer, "name", "") if customer else "",
                "customer.email": getattr(customer, "email", "") if customer else "",
                "customer.id": getattr(customer, "id", "") if customer and hasattr(customer, "id") else "",
                "customer.address": getattr(customer, "address", "") if customer else "",
                "payment.method": payment_method,
                "payment.last4": (
                    only_digits(getattr(customer, "credit_card", ""))[-4:]
                    if customer and getattr(customer, "credit_card", None)
                    else ""
                ),
            }
        )
    else:
        # Populate from this module's PurchaseOrder dataclass shape
        # Compute monetary fields from items
        items = getattr(order_obj, "items", []) or []
        subtotal = 0.0
        try:
            subtotal = sum(float(it.quantity) * float(it.unit_price) for it in items)
        except Exception:
            subtotal = 0.0
        tax = 0.0
        shipping = 0.0
        total = subtotal + tax + shipping

        context.update(
            {
                "order.id": getattr(order_obj, "order_id", ""),
                "order.customer_id": getattr(order_obj, "customer_id", ""),
                "order.currency": getattr(order_obj, "currency", ""),
                "order.item_count": len(items),
                "order.subtotal": round(subtotal, 2),
                "order.tax": round(tax, 2),
                "order.shipping": round(shipping, 2),
                "order.total": round(total, 2),
                "customer.name": "",
                "customer.email": "",
                "customer.id": getattr(order_obj, "customer_id", ""),
                "customer.address": "",
                "payment.method": "",
                "payment.last4": "",
            }
        )

    # Replace placeholders in the template
    def replace_match(match: re.Match) -> str:
        key = match.group(1)
        return str(context.get(key, ""))

    formatted = re.sub(r"\{([^{}]+)\}", replace_match, template)
    return formatted
