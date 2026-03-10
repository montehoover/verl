from typing import Any, Optional
from string import Formatter


class Customer:
    def __init__(self, name, email, address, credit_card):
        self.name = name
        self.email = email
        self.address = address
        self.credit_card = credit_card


class Order:
    def __init__(self, id, customer, items, total, payment_method):
        self.id = id
        self.customer = customer
        self.items = items
        self.total = total
        self.payment_method = payment_method


orders = {
    "ORD001": Order(
        "ORD001",
        Customer("Jane Doe", "jane@example.com", "123 Main St, Anytown, USA", "1234-5678-9012-3456"),
        ["Item 1", "Item 2"],
        99.99,
        "Credit Card",
    )
}


_RESTRICTED_CUSTOMER_FIELDS = {"email", "address", "credit_card"}


def _resolve_field(field_name: str, order: Order) -> Optional[Any]:
    """
    Resolve a dotted field name against the given order/customer object.
    Returns the value if resolvable, None if it cannot be resolved.
    Raises ValueError if attempting to access restricted/protected customer data.
    """
    if not field_name:
        return None

    parts = field_name.split(".")
    if not parts:
        return None

    root = parts[0]
    if root == "order":
        obj: Any = order
    elif root == "customer":
        obj = order.customer
    else:
        # Unknown root; leave placeholder unchanged by signaling unresolved
        return None

    # Traverse remaining attributes
    for attr in parts[1:]:
        # Protect against private/protected attributes
        if root == "customer":
            if attr.startswith("_") or attr in _RESTRICTED_CUSTOMER_FIELDS:
                raise ValueError(f"Access to restricted customer field: '{attr}'")
        try:
            obj = getattr(obj, attr)
        except AttributeError:
            return None

    return obj


def create_order_receipt(purchase_id: str, receipt_template: str) -> str:
    """
    Build a receipt string by replacing placeholders in the receipt_template with
    values from the specified order. Supported placeholders include dotted paths like:
      - {order.id}
      - {order.total}
      - {order.payment_method}
      - {order.items}
      - {customer.name}

    Any attempt to access restricted/protected customer data (email, address, credit_card,
    or any attribute starting with an underscore) will raise a ValueError.

    Unknown placeholders are left unchanged.
    """
    order = orders.get(purchase_id)
    if not order:
        raise ValueError(f"Order not found for ID: {purchase_id}")

    formatter = Formatter()
    result_parts = []

    for literal_text, field_name, format_spec, conversion in formatter.parse(receipt_template):
        # Add literal portion
        if literal_text:
            result_parts.append(literal_text)

        # No field to replace
        if field_name is None:
            continue

        # Resolve value
        value = _resolve_field(field_name, order)

        # If unresolved, leave placeholder unchanged
        if value is None:
            placeholder = "{" + field_name
            if conversion:
                placeholder += f"!{conversion}"
            if format_spec:
                placeholder += f":{format_spec}"
            placeholder += "}"
            result_parts.append(placeholder)
            continue

        # Apply conversion if provided
        if conversion:
            if conversion == "r":
                value = repr(value)
            elif conversion == "s":
                value = str(value)
            elif conversion == "a":
                value = ascii(value)
            else:
                # Unknown conversion; leave placeholder unchanged
                placeholder = "{" + field_name + f"!{conversion}"
                if format_spec:
                    placeholder += f":{format_spec}"
                placeholder += "}"
                result_parts.append(placeholder)
                continue

        # Apply format specification
        try:
            formatted = format(value, format_spec) if format_spec else f"{value}"
        except Exception:
            # If formatting fails, leave placeholder unchanged
            placeholder = "{" + field_name
            if conversion:
                placeholder += f"!{conversion}"
            if format_spec:
                placeholder += f":{format_spec}"
            placeholder += "}"
            result_parts.append(placeholder)
            continue

        result_parts.append(formatted)

    return "".join(result_parts)
