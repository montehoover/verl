from typing import Dict, Any


# Predefined orders stored in a dictionary
ORDERS: Dict[str, Dict[str, Any]] = {
    "1001": {"customer_name": "Alice Johnson", "total_amount": 149.99},
    "1002": {"customer_name": "Bob Smith", "total_amount": 89.50},
    "1003": {"customer_name": "Charlie Davis", "total_amount": 249.00},
}


def print_order_details(order_id, format_string: str = "Order ID: {order_id}\nCustomer: {customer_name}\nTotal: ${total_amount:.2f}") -> str:
    """
    Retrieve order details for the given order_id and return a formatted string.

    The format_string can contain placeholders:
    - {order_id}
    - {customer_name}
    - {total_amount} (supports format specifiers, e.g., {total_amount:.2f})

    If the order_id is not found, returns a not-found message string.
    """
    key = str(order_id)
    order = ORDERS.get(key)
    if not order:
        return f"Order not found: {key}"

    values = {
        "order_id": key,
        "customer_name": order.get("customer_name", "Unknown"),
        "total_amount": float(order.get("total_amount", 0.0)),
    }

    return format_string.format(**values)


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        fmt = sys.argv[2] if len(sys.argv) > 2 else "Order ID: {order_id}\nCustomer: {customer_name}\nTotal: ${total_amount:.2f}"
        result = print_order_details(sys.argv[1], fmt)
        print(result)
