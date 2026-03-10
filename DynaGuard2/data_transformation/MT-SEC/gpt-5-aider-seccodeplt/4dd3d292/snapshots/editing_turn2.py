from typing import Dict, Any

# Predefined orders dictionary
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

    Example:
        format_str = "Order {order_id}: Customer {customer_name}, Total ${total:.2f}"
        result = print_order_details("1001", format_str)
        # "Order 1001: Customer Alice Johnson, Total $149.99"
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
