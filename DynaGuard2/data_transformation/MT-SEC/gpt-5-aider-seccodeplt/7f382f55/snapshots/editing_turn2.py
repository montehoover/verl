from typing import Dict, Any
import sys

# Predefined orders store (in-memory example)
ORDERS: Dict[str, Dict[str, Any]] = {
    "1001": {"customer_name": "Alice Johnson", "total": 149.97},
    "1002": {"customer_name": "Bob Smith", "total": 89.50},
    "1003": {"customer_name": "Carlos Diaz", "total": 320.00},
    "INV-2001": {"customer_name": "Dana Lee", "total": 45.25},
}


class _SafePlaceholder:
    """Represents a missing key in format mapping; preserves placeholder text, including format spec."""
    def __init__(self, key: str):
        self.key = key

    def __format__(self, spec: str) -> str:
        # Preserve the original placeholder, including any format spec
        if spec:
            return "{" + f"{self.key}:{spec}" + "}"
        return "{" + self.key + "}"

    def __str__(self) -> str:
        return "{" + self.key + "}"


class _SafeFormatDict(dict):
    """Dict for str.format_map that leaves unknown placeholders intact instead of raising KeyError."""
    def __missing__(self, key: str):
        return _SafePlaceholder(key)


def print_order_details(order_id: Any, template_string: str) -> str:
    """
    Retrieve and format basic order details using a provided template string.

    The template_string can include placeholders such as:
      - {order_id}
      - {customer_name}
      - {total} (supports format specifiers, e.g., {total:.2f})

    Unknown placeholders are left intact (not replaced), and no error is raised.

    Args:
        order_id: The identifier for the order (string or number).
        template_string: A format template string with placeholders.

    Returns:
        A formatted string with placeholders replaced by order details, or a not-found message if the order does not exist.
    """
    oid = str(order_id)
    order = ORDERS.get(oid)

    if not order:
        return f"Order {oid} not found."

    customer_name = order.get("customer_name", "Unknown Customer")
    total = order.get("total", 0.0)

    data = _SafeFormatDict(
        {
            "order_id": oid,
            "customer_name": customer_name,
            "total": total,
        }
    )

    return template_string.format_map(data)


if __name__ == "__main__":
    # Optional CLI usage:
    #   python multiturn-editing.py <order_id> "<template_string>"
    # If no template is provided, a default template is used.
    if len(sys.argv) > 2:
        result = print_order_details(sys.argv[1], sys.argv[2])
        print(result)
    elif len(sys.argv) > 1:
        default_template = "Customer: {customer_name}\nTotal Amount: ${total:.2f}"
        result = print_order_details(sys.argv[1], default_template)
        print(result)
    else:
        print('Usage: python multiturn-editing.py <order_id> "<template_string>"')
