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
    # Customer (generic)
    "customer.id",
    "customer.full_name",
    "customer.first_name",
    "customer.last_name",
    "customer.email",           # generic alias
    # Customer (purchase setup specific)
    "customer.contact_email",   # maps to Customer.contact_email
    "customer.home_address",    # maps to Customer.home_address
    # Order (legacy dataclass)
    "order.id",
    "order.user_id",
    "order.status",
    "order.total",              # aligns with legacy Order.total
    # Order (purchase setup specific)
    "order.order_number",
    "order.price_total",        # maps to Order.price_total in purchase setup (alias to .total for legacy)
    "order.items_list",         # comma-separated string representation
    "order.item_count",
    "order.method_of_payment",
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


def create_purchase_receipt(order_id: str, receipt_template: str) -> str:
    """
    Generate a formatted receipt by replacing placeholders in the template with real data.

    This function supports two order data sources:
      - purchases: A dictionary keyed by order_id with objects having attributes
                   order_number, buyer (Customer with full_name, contact_email, home_address, payment_card),
                   items_list (list of str), price_total (number), method_of_payment (str).
      - ORDERS: Fallback legacy dataset defined in this module (limited placeholder coverage).

    Args:
        order_id: The ID of the order to render into the receipt.
        receipt_template: The template string containing placeholders.

    Returns:
        The formatted receipt string.

    Raises:
        ValueError: If the order ID is invalid or a restricted placeholder is used.
    """
    if not isinstance(receipt_template, str):
        raise ValueError("Template must be a string")
    if not isinstance(order_id, str) or not order_id:
        raise ValueError("Order ID must be a non-empty string")

    # Collect placeholders present in the template
    placeholders = [m.group(1).strip() for m in _PLACEHOLDER_PATTERN.finditer(receipt_template)]
    for token in placeholders:
        if token in RESTRICTED_TEMPLATE_PLACEHOLDERS:
            raise ValueError(f"Restricted placeholder used: {token}")

    # Determine data source and fetch order
    order_obj = None
    data_source = None  # "purchases" or "legacy"

    # Prefer purchases store if available
    purchases_store = globals().get("purchases")
    if isinstance(purchases_store, dict) and order_id in purchases_store:
        order_obj = purchases_store[order_id]
        data_source = "purchases"
    elif order_id in ORDERS:
        order_obj = ORDERS[order_id]
        data_source = "legacy"

    if order_obj is None:
        raise ValueError(f"Order ID '{order_id}' not found")

    # Build replacement map based on the detected data source
    replacements: Dict[str, str] = {}

    if data_source == "purchases":
        # Expect attributes per provided setup:
        # Order: order_number, buyer, items_list, price_total, method_of_payment
        # Customer: full_name, contact_email, home_address, payment_card
        buyer = getattr(order_obj, "buyer", None)

        # Customer placeholders
        if buyer is not None:
            full_name = getattr(buyer, "full_name", "")
            contact_email = getattr(buyer, "contact_email", "")
            home_address = getattr(buyer, "home_address", "")

            # derive optional first/last name if possible
            first_name = ""
            last_name = ""
            if isinstance(full_name, str) and full_name.strip():
                parts = full_name.strip().split()
                first_name = parts[0]
                last_name = parts[-1] if len(parts) > 1 else ""

            replacements.update({
                "customer.full_name": str(full_name),
                "customer.first_name": str(first_name),
                "customer.last_name": str(last_name),
                "customer.contact_email": str(contact_email),
                "customer.email": str(contact_email),  # alias
                "customer.home_address": str(home_address),
            })

        # Order placeholders
        order_number = getattr(order_obj, "order_number", "")
        price_total = getattr(order_obj, "price_total", "")
        items_list = getattr(order_obj, "items_list", [])
        method_of_payment = getattr(order_obj, "method_of_payment", "")

        # Normalize items list to a readable string
        if isinstance(items_list, list):
            items_list_str = ", ".join(str(x) for x in items_list)
            item_count = len(items_list)
        else:
            items_list_str = str(items_list)
            item_count = 0

        # Price formatting
        try:
            price_total_str = f"{float(price_total):.2f}"
        except (ValueError, TypeError):
            price_total_str = str(price_total)

        replacements.update({
            "order.order_number": str(order_number),
            "order.price_total": price_total_str,
            "order.items_list": items_list_str,
            "order.item_count": str(item_count),
            "order.method_of_payment": str(method_of_payment),
        })

        # Provide some generic aliases where sensible
        replacements.update({
            "order.id": str(order_number),
            "order.total": price_total_str,
        })

    elif data_source == "legacy":
        # Legacy Order dataclass defined in this module
        # We can fill order-centric placeholders and derive item_count,
        # but no customer details are available in this dataset.
        order_id_val = getattr(order_obj, "id", "")
        total_val = getattr(order_obj, "total", "")
        user_id_val = getattr(order_obj, "user_id", "")
        status_val = getattr(order_obj, "status", "")
        items = getattr(order_obj, "items", [])

        if isinstance(items, list):
            item_count = len(items)
            items_list_str = ", ".join(getattr(i, "name", str(i)) for i in items)
        else:
            item_count = 0
            items_list_str = ""

        try:
            total_str = f"{float(total_val):.2f}"
        except (ValueError, TypeError):
            total_str = str(total_val)

        replacements.update({
            "order.id": str(order_id_val),
            "order.user_id": str(user_id_val),
            "order.status": str(status_val),
            "order.total": total_str,
            "order.price_total": total_str,  # alias
            "order.item_count": str(item_count),
            "order.items_list": items_list_str,
        })

    # Replacement function that keeps unknown placeholders intact
    def _repl(match: re.Match) -> str:
        token = match.group(1).strip()
        if token in RESTRICTED_TEMPLATE_PLACEHOLDERS:
            # Defensive check during substitution
            raise ValueError(f"Restricted placeholder used: {token}")
        if token in replacements:
            return str(replacements[token])
        return match.group(0)

    return _PLACEHOLDER_PATTERN.sub(_repl, receipt_template)


__all__ = [
    "Order",
    "OrderItem",
    "ORDERS",
    "get_order_by_id",
    "check_template_placeholders",
    "VALID_TEMPLATE_PLACEHOLDERS",
    "RESTRICTED_TEMPLATE_PLACEHOLDERS",
    "create_purchase_receipt",
]
