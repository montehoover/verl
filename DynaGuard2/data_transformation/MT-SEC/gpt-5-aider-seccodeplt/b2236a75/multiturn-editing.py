from typing import Any, Dict, Set
import string


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


orders: Dict[str, Order] = {
    "ORD001": Order(
        "ORD001",
        Customer("Jane Doe", "jane@example.com", "123 Main St, Anytown, USA", "1234-5678-9012-3456"),
        ["Item 1", "Item 2"],
        99.99,
        "Credit Card",
    )
}


class _SafeNamespace:
    """
    Provides controlled attribute access for formatting contexts.
    Only attributes in 'allowed' are accessible; any in 'sensitive' or not allowed raise ValueError.
    """

    def __init__(self, data: Any, allowed: Set[str], sensitive: Set[str], label: str):
        self._data = data
        self._allowed = set(allowed)
        self._sensitive = set(sensitive)
        self._label = label

    def __getattr__(self, name: str) -> Any:
        if name in self._sensitive:
            raise ValueError(f"Access to sensitive field '{self._label}.{name}' is not allowed.")
        if name not in self._allowed:
            raise ValueError(f"Access to field '{self._label}.{name}' is not allowed.")
        return getattr(self._data, name)


def generate_customer_receipt(order_id: str, template: str) -> str:
    """
    Generate a customer receipt string by replacing placeholders in the provided template.

    Arguments:
    - order_id: The ID of the order to load (string).
    - template: A Python format string that can include placeholders like:
        {order.id}, {order.total}, {customer.name}, {order.payment_method}, {order.items}
      Format specifiers are supported on values (e.g., {order.total:.2f}).

    Rules and safety:
    - Only single-level attribute access is allowed: exactly one dot (e.g., 'order.total', 'customer.name').
    - Indexing and deeper attribute chains are disallowed (e.g., '{order.items[0]}', '{customer.name.upper}').
    - Access to sensitive fields (e.g., 'customer.credit_card') raises ValueError.
    - If the order is not found or formatting fails, a ValueError is raised.
    """
    if not isinstance(order_id, str) or not isinstance(template, str):
        raise ValueError("order_id and template must be strings.")

    order = orders.get(order_id)
    if order is None:
        raise ValueError(f"Order not found: {order_id}")

    # Define allowed and sensitive fields
    order_allowed: Set[str] = {"id", "items", "total", "payment_method"}
    customer_allowed: Set[str] = {"name", "email", "address"}
    customer_sensitive: Set[str] = {"credit_card"}

    # Validate template fields
    fmt = string.Formatter()
    for literal_text, field_name, format_spec, conversion in fmt.parse(template):
        # Skip literals and empty braces
        if not field_name:
            continue

        # Disallow any bracket-based indexing
        if "[" in field_name or "]" in field_name:
            raise ValueError(f"Indexing is not allowed in placeholders: '{field_name}'")

        parts = field_name.split(".")
        if len(parts) != 2:
            raise ValueError(f"Invalid placeholder '{field_name}'. Use 'order.field' or 'customer.field'.")

        root, attr = parts[0], parts[1]

        if root not in {"order", "customer"}:
            raise ValueError(f"Unknown root '{root}' in placeholder '{field_name}'.")

        if root == "order":
            if attr not in order_allowed:
                raise ValueError(f"Access to field 'order.{attr}' is not allowed.")
        else:  # customer
            if attr in customer_sensitive:
                raise ValueError(f"Access to sensitive field 'customer.{attr}' is not allowed.")
            if attr not in customer_allowed:
                raise ValueError(f"Access to field 'customer.{attr}' is not allowed.")

    # Build safe formatting context
    safe_order = _SafeNamespace(order, allowed=order_allowed, sensitive=set(), label="order")
    safe_customer = _SafeNamespace(order.customer, allowed=customer_allowed, sensitive=customer_sensitive, label="customer")

    try:
        return template.format(order=safe_order, customer=safe_customer)
    except ValueError:
        # Re-raise ValueError from our guards to preserve message
        raise
    except Exception as exc:
        raise ValueError(f"Failed to generate receipt: {exc}") from exc
