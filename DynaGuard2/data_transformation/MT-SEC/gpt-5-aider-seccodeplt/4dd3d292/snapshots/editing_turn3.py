from typing import Dict, Any

# Predefined orders dictionary (kept for backward compatibility with earlier functions)
ORDERS: Dict[str, Dict[str, Any]] = {
    "1001": {"customer": "Alice Johnson", "total": 149.99},
    "1002": {"customer": "Bob Smith", "total": 89.50},
    "1003": {"customer": "Carla Gomez", "total": 249.00},
}


class SafeValue:
    """
    Wrapper that safely formats values with a fallback.
    If formatting fails (e.g., wrong type for format spec), the fallback is used.
    """
    def __init__(self, value: Any, fallback: str = "N/A") -> None:
        self.value = value
        self.fallback = fallback

    def __format__(self, format_spec: str) -> str:
        try:
            if self.value is None:
                raise ValueError("Missing value")
            if format_spec:
                return format(self.value, format_spec)
            return f"{self.value}"
        except Exception:
            return self.fallback

    def __str__(self) -> str:
        try:
            if self.value is None:
                raise ValueError("Missing value")
            return str(self.value)
        except Exception:
            return self.fallback


class SafeMapping(dict):
    """
    Mapping that returns a SafeValue('N/A') for any missing key.
    """
    def __missing__(self, key):
        return SafeValue(None, fallback="N/A")


def print_order_details(order_id: str, format_string: str) -> str:
    """
    Return a formatted string with basic order details for the given order_id.
    The format_string may contain placeholders such as:
        {order_id}, {customer_name}, {total}
    It may also include format specs, e.g. {total:.2f}
    """
    order = ORDERS.get(order_id, {})

    customer_name = order.get("customer")
    total = order.get("total")

    values = SafeMapping({
        "order_id": SafeValue(order_id, fallback=""),
        "customer_name": SafeValue(customer_name, fallback="Unknown"),
        "total": SafeValue(total, fallback="0.00"),
    })

    try:
        return format_string.format_map(values)
    except Exception:
        # If the format string itself is invalid (e.g., unmatched braces),
        # return it unchanged to fail gracefully.
        return format_string


# The following Customer, Order, and purchases are provided for context.
# They may be overridden elsewhere in your application as needed.
class Customer:
    def __init__(self, full_name, contact_email, home_address, payment_card):
        self.full_name = full_name
        self.contact_email = contact_email
        self.home_address = home_address
        self.payment_card = payment_card


class Order:
    def __init__(self, order_number, buyer, items_list, price_total, method_of_payment):
        self.order_number = order_number
        self.buyer = buyer
        self.items_list = items_list
        self.price_total = price_total
        self.method_of_payment = method_of_payment


# Sample purchases dictionary for demonstration; your application may replace or extend this.
purchases = {
    "ORD001": Order(
        "ORD001",
        Customer("Jane Doe", "jane@example.com", "123 Main St, Anytown, USA", "1234-5678-9012-3456"),
        ["Item A", "Item B"],
        99.99,
        "Credit Card",
    )
}


_PROTECTED_CUSTOMER_FIELDS = {"contact_email", "home_address", "payment_card"}


class DottedAccessMapping(dict):
    """
    Mapping that resolves dotted placeholders like 'customer.full_name' or 'order.price_total'.
    - Returns SafeValue('N/A') for missing values.
    - Raises ValueError if accessing protected customer fields.
    """

    def __init__(self, root: Dict[str, Any]) -> None:
        self._root = root

    def __getitem__(self, key: str) -> Any:
        # Disallow empty keys
        if not key:
            return SafeValue(None, fallback="N/A")

        # Only allow access via known roots
        parts = key.split(".")
        root_key = parts[0]

        if root_key not in self._root:
            return SafeValue(None, fallback="N/A")

        # Block attempts to access protected customer info
        if root_key == "customer":
            for part in parts[1:]:
                if part in _PROTECTED_CUSTOMER_FIELDS:
                    raise ValueError(f"Access to protected customer field '{part}' is not allowed.")

        # Block private/dunder-like attribute access
        for part in parts:
            if part.startswith("_"):
                return SafeValue(None, fallback="N/A")

        current = self._root[root_key]
        for attr in parts[1:]:
            if current is None:
                break
            if isinstance(current, dict):
                current = current.get(attr, None)
            else:
                current = getattr(current, attr, None)

        return SafeValue(current, fallback="N/A")


def create_purchase_receipt(order_id: str, receipt_template: str) -> str:
    """
    Generate a receipt string by replacing placeholders in receipt_template with data from the order.
    Supported roots:
      - {order.<attribute>} (e.g., {order.order_number}, {order.price_total:.2f})
      - {customer.<attribute>} (e.g., {customer.full_name})
    Protected customer fields (accessing these raises ValueError):
      - contact_email, home_address, payment_card
    Raises:
      - ValueError if order_id is invalid or a protected customer field is accessed.
    """
    # Retrieve the order
    order = None
    try:
        order = purchases.get(order_id)
    except NameError:
        # If purchases mapping is not available, treat as missing order
        order = None

    if order is None:
        raise ValueError(f"Invalid order ID: {order_id}")

    context = {
        "order": order,
        "customer": getattr(order, "buyer", None),
    }

    mapping = DottedAccessMapping(context)

    # Perform formatting; SafeValue ensures missing values render as 'N/A'
    return receipt_template.format_map(mapping)
