ORDERS = {
    "ORD-1001": {"customer_name": "Alice Smith", "total_amount": 89.97, "currency": "USD"},
    "ORD-1002": {"customer_name": "Bob Johnson", "total_amount": 149.50, "currency": "USD"},
    "ORD-1003": {"customer_name": "Charlie Lee", "total_amount": 42.00, "currency": "USD"},
}


class _MissingPlaceholder:
    def __init__(self, key: str):
        self.key = key

    def __format__(self, format_spec: str) -> str:
        # Reconstruct the original placeholder (including any format spec) if missing
        if format_spec:
            return f"{{{self.key}:{format_spec}}}"
        return f"{{{self.key}}}"

    def __str__(self) -> str:
        return f"{{{self.key}}}"


class _SafeDict(dict):
    def __missing__(self, key):
        # Return a special object that leaves the placeholder intact
        return _MissingPlaceholder(key)


def print_order_details(order_id, format_string):
    """
    Retrieve basic order details for the given order_id and return a formatted string.

    The format_string can contain placeholders such as:
      - {order_id}
      - {customer_name}
      - {total_amount} (can be formatted, e.g., {total_amount:.2f})
      - {currency}

    Missing placeholders are handled gracefully: any unknown placeholder remains unchanged.
    """
    order = ORDERS.get(str(order_id))
    if not order:
        return f"Order not found for ID: {order_id}"

    context = _SafeDict(
        {
            "order_id": str(order_id),
            "customer_name": order.get("customer_name"),
            "total_amount": order.get("total_amount"),
            "currency": order.get("currency"),
        }
    )

    return format_string.format_map(context)
